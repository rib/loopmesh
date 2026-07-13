use smallvec::SmallVec;

use crate::loopmesh::*;

/// Parameters controlling [`LoopMesh::dissolve_faces`].
///
/// Mirrors the BMOP set-input shape described in
/// `radial-mesh-guides/operators/DISSOLVE_FACES.md` §1.
#[derive(Clone, Copy, Default)]
pub struct DissolveFacesOptions {
    /// Post-pass that collapses every vertex which was *not* an edge-pair
    /// before the operator started but became one as a consequence of the
    /// per-region merges. Documented in
    /// `radial-mesh-guides/operators/DISSOLVE_FACES.md` §6.
    pub use_verts: bool,
}

/// Output of [`LoopMesh::dissolve_faces`]. One entry per successful
/// per-region merge — the surviving merged face. Failed regions and
/// singletons contribute nothing.
///
/// Conceptually parallels the `region.out` slot described in
/// `radial-mesh-guides/operators/DISSOLVE_FACES.md` §1.
#[derive(Default)]
pub struct DissolveFacesResult {
    pub region: Vec<FaceKey>,
}

impl LoopMesh {
    /// Merge each manifoldly-connected region of `faces` into a single face.
    ///
    /// Follows the algorithm in
    /// `radial-mesh-guides/operators/DISSOLVE_FACES.md`:
    ///
    /// 1. Tag every input face (§3 walker face-mask).
    /// 2. From each unvisited input face, BFS across edges whose radial
    ///    cycle has exactly two faces *both* of which are tagged (§3 +
    ///    §3.3). Singleton regions are skipped per §3.1.
    /// 3. For each region of size ≥ 2, merge via the many-input
    ///    `faces_join` primitive (§4). The primitive kills the interior
    ///    shared edges and any vertex whose every incident edge was
    ///    interior (§4.1 steps 2 + 4).
    /// 4. Collect the surviving merged face into the result (§1
    ///    `region.out`).
    ///
    /// Supports the default-parameter manifold-region path and the optional
    /// `use_verts` post-pass (§6). The merge is invoked in the §4.2
    /// report-the-double form: when the union's perimeter coincides with a
    /// pre-existing face's, the primitive keeps both and names the double,
    /// and *this* operator kills the fresh union and reports the pre-existing
    /// face in `region.out`. Knowing the double is not a formality — §5's
    /// survivor detector prices it into the expected face count.
    pub fn dissolve_faces(
        &mut self,
        faces: &[FaceKey],
        options: DissolveFacesOptions,
    ) -> DissolveFacesResult {
        // §6.1 sub-phase 0 — snapshot which verts were *not* edge-pair
        // before any merge ran. The post-pass (sub-phase 1) collapses
        // exactly those of these verts whose status flipped to edge-pair
        // as a result of the merges; pre-existing edge-pair verts are
        // never marked and therefore never collapsed.
        let use_verts_mark = if options.use_verts {
            let layer = self.push_vert_op_flag_layer();
            let mut iter = MeshVerts::default();
            while let Some(v) = iter.next(self) {
                if !self.vert_is_edge_pair_for_dissolve_faces(v) {
                    self.vert_op_flag_set(layer, v, 0);
                }
            }
            Some(layer)
        } else {
            None
        };

        let mut result = DissolveFacesResult::default();
        if !faces.is_empty() {
            // §3 — face mark = "this face is in the input set". The walker
            // crosses an edge iff its radial cycle has exactly two faces
            // and both carry this mark.
            let in_input = self.push_face_op_flag_layer();
            for &f in faces {
                self.face_op_flag_set(in_input, f, 0);
            }
            // Independent "already visited" bit so that a single walker
            // step never re-seeds a face already pulled into another
            // region's BFS.
            let visited = self.push_face_op_flag_layer();
            // §5 — the survivor mark. Set on every face of every region that
            // is actually opened (size ≥ 2; singletons never enter a region
            // and so are never marked, per §3.1's no-op guarantee), then
            // cleared on the union of each successful merge and on every input
            // face of a failed one.
            let orig = self.push_face_op_flag_layer();
            let mut has_survivors = false;

            // Stable input-iteration order — the §11 customdata identity
            // rule ("first face in walker-emit order wins") depends on it.
            for &seed in faces {
                if self.face_op_flag_test(visited, seed, 0) {
                    continue;
                }
                let region = self.dissolve_faces_walk_region(seed, in_input, visited);
                if region.len() < 2 {
                    // §3.1 — singletons are skipped silently.
                    continue;
                }
                for &face in &region {
                    self.face_op_flag_set(orig, face, 0);
                }

                // §4 — the many-input merge primitive does the rest, in its §4.2
                // report-the-double calling form: it hands back both faces and
                // this operator settles the overlap.
                let face_count_before = self.face_count();
                match self.faces_join_report_double(&region) {
                    Some(outcome) => {
                        // §4.2 — the operator's half of the double: kill the fresh
                        // union, adopt the pre-existing face, report *that* one.
                        let (surviving, doubles) = match outcome.double {
                            Some(shadow) => {
                                self.face_kill(outcome.union);
                                (shadow, 1)
                            }
                            None => (outcome.union, 0),
                        };

                        // §5 — the mark comes off the region's *result*, whichever
                        // face that turned out to be. On the double path that is
                        // the adopted pre-existing face, which may itself be a
                        // still-marked §4.1.4 survivor of an earlier region: once
                        // it is this region's merge result it is a live output, and
                        // the mop-up must spare it. (On the plain path the mint
                        // carries no mark anyway — a recycled slot's op flags are
                        // zeroed at creation — so the clear is a guard there.)
                        self.face_op_flag_clear(orig, surviving, 0);
                        result.region.push(surviving);

                        // §5's survivor detector. The expectation is the naive
                        // "`n` faces in, one union out" (−(n − 1)), less one more
                        // for a double (whose union we just killed). The join only
                        // ever *adds* the union and *removes* input faces of this
                        // region, so an overshoot means it left survivors standing
                        // (§4.1.4) and the mop-up has work to do. Subtracting the
                        // double from the *expectation* — rather than reading it
                        // as a survivor's opposite — is what keeps the detector
                        // exact: it fires on one survivor whether or not the region
                        // also doubled.
                        let expected_face_count = face_count_before - (region.len() - 1) - doubles;
                        if self.face_count() > expected_face_count {
                            has_survivors = true;
                        }
                    }
                    None => {
                        // Failed merges (§4.3) leave the region's input faces
                        // intact — unmark them so the mop-up spares them — and
                        // contribute nothing to `region.out`.
                        for &face in &region {
                            self.face_op_flag_clear(orig, face, 0);
                        }
                    }
                }
            }

            // §5 — the mop-up runs once, *after every region*, never inside
            // the region loop. A survivor of region A is alive precisely
            // because it sits on a keep-alive chord, and later regions
            // classify that chord by counting the faces *currently* on it:
            // killing the survivor early would shorten the chord's radial and
            // flip it from keep-alive to killable for region B.
            if has_survivors {
                self.dissolve_faces_kill_marked_survivors(orig);
            }

            self.pop_face_op_flag_layer(orig);
            self.pop_face_op_flag_layer(visited);
            self.pop_face_op_flag_layer(in_input);
        }

        // §6.1 sub-phase 1 — collapse every surviving marked vert that is
        // now edge-pair. Verts killed by the merges (their entire disk was
        // interior) are gone by the time we iterate, so the lending vert
        // walker simply doesn't surface them; no stale-mark hazard. Verts
        // that were marked but did not flip to edge-pair fail the in-loop
        // re-test and are skipped.
        if let Some(layer) = use_verts_mark {
            self.dissolve_faces_collapse_marked_edge_pair_verts(layer);
            self.pop_vert_op_flag_layer(layer);
        }

        result
    }

    /// §5 — delete the input faces the joins left alive. Sweeping *live* faces
    /// and testing the mark is the only sound liveness probe here: a face key
    /// is a bare slab index, so a key held from before a join may name a slot
    /// that has since been freed or handed to a fresh face. A recycled slot
    /// carries no op flags (creation zeroes them), so the mark can only be
    /// true for a face that was walked into a successful region and never
    /// died — exactly the §4.1.4 survivors.
    ///
    /// The removal is a **faces**-context delete (§5.2, `DELETE.md` §4.2), not a
    /// faces-only kill. A survivor's keep-alive interior edges were spared by
    /// its own region's join on the strength of faces *outside that region* —
    /// a guarantee that expires the moment those faces are themselves consumed
    /// by another region. Whether such an edge outlives the *operator* is
    /// therefore decided here, by whether any face at all is still on it: the
    /// delete dooms the survivors' boundary, un-dooms whatever a surviving face
    /// or a surviving edge still holds, and sweeps the rest. That is what stops
    /// an emptied chord from standing as a wire edge (and an emptied vert as a
    /// loose point).
    fn dissolve_faces_kill_marked_survivors(&mut self, marked: crate::op_flags::FaceOpFlagLayer) {
        let mut survivors: SmallVec<[FaceKey; 4]> = SmallVec::new();
        let mut iter = MeshFaces::default();
        while let Some(face) = iter.next(self) {
            if self.face_op_flag_test(marked, face, 0) {
                survivors.push(face);
            }
        }
        self.delete(survivors.as_slice(), DeleteContext::Faces);
    }

    /// §6.1 predicate — true iff `v` has exactly two incident edges.
    /// Conceptually parallels the `is_edge_pair` snapshot used by the
    /// `use_verts` mark; lifted to a helper so the sub-phase 0 scan and
    /// the sub-phase 1 re-test ask the same question.
    #[inline]
    fn vert_is_edge_pair_for_dissolve_faces(&self, v: VertexKey) -> bool {
        self.vertex_disk_count_edges_capped(v, 3) == 2
    }

    /// §6.1 sub-phase 1 — single mesh-wide walk over live verts; for each
    /// vert that was marked pre-merge AND is now edge-pair, collapse it
    /// via the edge-pair-vert collapse primitive
    /// (`vert_collapse_edge` → JEKV). Iteration order is the underlying
    /// vert table's order; each collapse only frees one vert and one
    /// edge, never creates new ones, so the in-loop `MeshVerts` cursor
    /// stays valid.
    fn dissolve_faces_collapse_marked_edge_pair_verts(
        &mut self,
        marked: crate::op_flags::VertOpFlagLayer,
    ) {
        let mut iter = MeshVerts::default();
        while let Some(vertex) = iter.next(self) {
            if !self.vert_op_flag_test(marked, vertex, 0) {
                continue;
            }
            if !self.vert_is_edge_pair_for_dissolve_faces(vertex) {
                continue;
            }
            // Edge-pair: pick either of the two incident edges to kill;
            // the primitive retargets the other onto the surviving
            // neighbour. The `Option` return covers the rare double-edge
            // refusal — on that path the vert simply survives uncollapsed
            // (no harm, no use-after-free; iteration continues).
            let mut disk = self.disk_edges(vertex);
            let e_kill = match disk.next(self) {
                Some(e) => e,
                None => continue,
            };
            let _ = self.vert_collapse_edge(e_kill, vertex);
        }
    }

    /// BFS the manifold region rooted at `seed`. Visits every face
    /// reachable across an edge whose radial cycle is exactly the seed
    /// face plus one neighbour, where the neighbour also carries the
    /// `in_input` mark. Each visited face is stamped with `visited` so
    /// the caller's outer loop does not re-seed it.
    fn dissolve_faces_walk_region(
        &mut self,
        seed: FaceKey,
        in_input: crate::op_flags::FaceOpFlagLayer,
        visited: crate::op_flags::FaceOpFlagLayer,
    ) -> SmallVec<[FaceKey; 8]> {
        let mut region: SmallVec<[FaceKey; 8]> = SmallVec::new();
        let mut frontier: SmallVec<[FaceKey; 8]> = SmallVec::new();
        frontier.push(seed);
        self.face_op_flag_set(visited, seed, 0);

        while let Some(face) = frontier.pop() {
            region.push(face);

            // Snapshot the face's perimeter edges; the walker reads radial
            // cycles but does not mutate the mesh.
            let edges: SmallVec<[EdgeKey; 8]> =
                collect_iter::<_, 8>(self, FaceLoops::new(self, face))
                    .into_iter()
                    .map(|l| self.loops.get(l).edge)
                    .filter(|e| !e.is_null())
                    .collect();

            for edge in edges {
                // §3.3 crossability predicate: an edge is crossable iff
                // *exactly two* of its radial faces are in the input set.
                // The total radial length may be >2 (non-manifold spine
                // with one or more out-of-input faces on it); what gates
                // the cross is the input-set count, not the radial count.
                let mut input_face_count = 0usize;
                let mut neighbour: Option<FaceKey> = None;
                let mut radial = self.radial_loops(edge);
                while let Some(rloop) = radial.next(self) {
                    let rface = self.loops.get(rloop).face;
                    if !self.face_op_flag_test(in_input, rface, 0) {
                        continue;
                    }
                    input_face_count += 1;
                    if rface != face && neighbour.is_none() {
                        neighbour = Some(rface);
                    }
                }
                if input_face_count != 2 {
                    continue;
                }
                let Some(neighbour) = neighbour else { continue };
                if neighbour == face {
                    continue;
                }
                if self.face_op_flag_test(visited, neighbour, 0) {
                    continue;
                }
                self.face_op_flag_set(visited, neighbour, 0);
                frontier.push(neighbour);
            }
        }

        region
    }
}

#[cfg(test)]
mod tests {
    use glam::Vec3;

    use crate::loopmesh::*;

    /// Build a 3x3 face grid of quads with positions on the unit lattice.
    /// Returns `(verts[16], edges, faces[9])` where `faces[r * 3 + c]` is
    /// the quad whose lower-left vertex is `verts[r * 4 + c]`. Vertex
    /// numbering: `r * 4 + c` over rows `r in 0..4` and columns
    /// `c in 0..4`. Quad winding is CCW from above (lower-left → lower-
    /// right → upper-right → upper-left).
    fn build_grid_3x3(mesh: &mut LoopMesh) -> (Vec<VertexKey>, Vec<FaceKey>) {
        let mut verts: Vec<VertexKey> = Vec::with_capacity(16);
        for r in 0..4 {
            for c in 0..4 {
                verts.push(mesh.create_vertex(Vec3::new(c as f32, r as f32, 0.0)));
            }
        }
        let mut faces: Vec<FaceKey> = Vec::new();
        for r in 0..3 {
            for c in 0..3 {
                let v_bl = verts[r * 4 + c];
                let v_br = verts[r * 4 + c + 1];
                let v_tr = verts[(r + 1) * 4 + c + 1];
                let v_tl = verts[(r + 1) * 4 + c];
                let e_b = mesh.create_edge([v_bl, v_br], EdgeCreateFlags::NO_DUPLICATE);
                let e_r = mesh.create_edge([v_br, v_tr], EdgeCreateFlags::NO_DUPLICATE);
                let e_t = mesh.create_edge([v_tr, v_tl], EdgeCreateFlags::NO_DUPLICATE);
                let e_l = mesh.create_edge([v_tl, v_bl], EdgeCreateFlags::NO_DUPLICATE);
                faces.push(mesh.face_create(
                    &[v_bl, v_br, v_tr, v_tl],
                    &[e_b, e_r, e_t, e_l],
                    FaceCreateFlags::NONE,
                ));
            }
        }
        (verts, faces)
    }

    #[test]
    fn dissolve_faces_two_adjacent_quads_merge_into_rectangle() {
        // §7 row 2: a region of n=2 with i=1 interior edge and m=0
        // interior-only verts on a grid contributes -1 F, -1 E, 0 V.
        let mut mesh = LoopMesh::new();
        let (_verts, faces) = build_grid_3x3(&mut mesh);

        let v0 = mesh.vertex_count();
        let e0 = mesh.edge_count();
        let f0 = mesh.face_count();
        assert_eq!(v0, 16);
        assert_eq!(e0, 24, "3x3 grid: 4*3 horizontal + 4*3 vertical = 24 edges");
        assert_eq!(f0, 9);

        // Dissolve the bottom-left two quads (faces[0] and faces[1]) which
        // share the vertical edge between (1,0)-(1,1).
        let result = mesh.dissolve_faces(&[faces[0], faces[1]], Default::default());

        assert_eq!(result.region.len(), 1, "one region, one successful merge");
        let merged = result.region[0];

        assert_eq!(mesh.face_count(), f0 - 1, "-1 face per §7 row 2");
        assert_eq!(mesh.edge_count(), e0 - 1, "-1 edge per §7 row 2");
        assert_eq!(mesh.vertex_count(), v0, "0 verts removed per §7 row 2");

        let perimeter = mesh.face_vert_keys(merged);
        assert_eq!(
            perimeter.len(),
            6,
            "merged face has 6 corners: 4+4 minus the two consumed by the interior edge"
        );
        mesh.validate_topology();
    }

    #[test]
    fn dissolve_faces_empty_input_is_noop() {
        let mut mesh = LoopMesh::new();
        let (_verts, _faces) = build_grid_3x3(&mut mesh);
        let v0 = mesh.vertex_count();
        let e0 = mesh.edge_count();
        let f0 = mesh.face_count();

        let result = mesh.dissolve_faces(&[], Default::default());

        assert!(result.region.is_empty());
        assert_eq!(mesh.vertex_count(), v0);
        assert_eq!(mesh.edge_count(), e0);
        assert_eq!(mesh.face_count(), f0);
        mesh.validate_topology();
    }

    #[test]
    fn dissolve_faces_single_face_input_is_noop() {
        // §3.1 — a region of size 1 is silently skipped: the walker
        // requires at least one manifold step to open a region.
        let mut mesh = LoopMesh::new();
        let (_verts, faces) = build_grid_3x3(&mut mesh);
        let v0 = mesh.vertex_count();
        let e0 = mesh.edge_count();
        let f0 = mesh.face_count();

        let result = mesh.dissolve_faces(&[faces[0]], Default::default());

        assert!(
            result.region.is_empty(),
            "singleton region contributes nothing to region.out"
        );
        assert_eq!(mesh.vertex_count(), v0);
        assert_eq!(mesh.edge_count(), e0);
        assert_eq!(mesh.face_count(), f0);
        mesh.validate_topology();
    }

    #[test]
    fn dissolve_faces_merged_face_perimeter_keeps_outer_grid_verts() {
        // The merged face's six corners must be exactly the six outer
        // vertices of the two-quad region; the two endpoints of the
        // shared interior edge survive (they still border the rest of
        // the grid).
        let mut mesh = LoopMesh::new();
        let (verts, faces) = build_grid_3x3(&mut mesh);

        // Faces[0] perimeter = v0,v1,v5,v4. Faces[1] perimeter = v1,v2,v6,v5.
        // Shared edge sits between v1 and v5. Merged perimeter = {v0,v1,v2,v6,v5,v4}.
        let expected: std::collections::HashSet<VertexKey> =
            [verts[0], verts[1], verts[2], verts[6], verts[5], verts[4]]
                .into_iter()
                .collect();

        let result = mesh.dissolve_faces(&[faces[0], faces[1]], Default::default());
        let merged = result.region[0];
        let got: std::collections::HashSet<VertexKey> =
            mesh.face_vert_keys(merged).into_iter().collect();
        assert_eq!(got, expected);
    }

    /// Build a 5x5 face grid (36 verts, 60 edges, 25 faces). Same numbering
    /// scheme as `build_grid_3x3`: `verts[r * 6 + c]` for `r,c in 0..6`,
    /// `faces[r * 5 + c]` for `r,c in 0..5`.
    fn build_grid_5x5(mesh: &mut LoopMesh) -> (Vec<VertexKey>, Vec<FaceKey>) {
        let mut verts: Vec<VertexKey> = Vec::with_capacity(36);
        for r in 0..6 {
            for c in 0..6 {
                verts.push(mesh.create_vertex(Vec3::new(c as f32, r as f32, 0.0)));
            }
        }
        let mut faces: Vec<FaceKey> = Vec::new();
        for r in 0..5 {
            for c in 0..5 {
                let v_bl = verts[r * 6 + c];
                let v_br = verts[r * 6 + c + 1];
                let v_tr = verts[(r + 1) * 6 + c + 1];
                let v_tl = verts[(r + 1) * 6 + c];
                let e_b = mesh.create_edge([v_bl, v_br], EdgeCreateFlags::NO_DUPLICATE);
                let e_r = mesh.create_edge([v_br, v_tr], EdgeCreateFlags::NO_DUPLICATE);
                let e_t = mesh.create_edge([v_tr, v_tl], EdgeCreateFlags::NO_DUPLICATE);
                let e_l = mesh.create_edge([v_tl, v_bl], EdgeCreateFlags::NO_DUPLICATE);
                faces.push(mesh.face_create(
                    &[v_bl, v_br, v_tr, v_tl],
                    &[e_b, e_r, e_t, e_l],
                    FaceCreateFlags::NONE,
                ));
            }
        }
        (verts, faces)
    }

    /// `use_verts = true` headline: dissolving the bottom row of three quads
    /// in a 3x3 face grid demotes the two mid-rim verts (1 and 2) from
    /// 3-edge to 2-edge — newly edge-pair — and the §6 post-pass collapses
    /// them. Final mesh is a hexagon perimeter for the merged region with
    /// 14 V, 20 E, 7 F.
    #[test]
    fn dissolve_faces_use_verts_collapses_newly_edge_pair_verts() {
        let mut mesh = LoopMesh::new();
        let (verts, faces) = build_grid_3x3(&mut mesh);

        let v0_count = mesh.vertex_count();
        let e0_count = mesh.edge_count();
        let f0_count = mesh.face_count();
        assert_eq!(v0_count, 16);
        assert_eq!(e0_count, 24);
        assert_eq!(f0_count, 9);

        let v1 = verts[1];
        let v2 = verts[2];
        let v1_pre_count = mesh.vert_edge_keys(v1).len();
        let v2_pre_count = mesh.vert_edge_keys(v2).len();
        assert_eq!(
            v1_pre_count, 3,
            "v1 has 3 edges pre-merge → eligible for use_verts mark"
        );
        assert_eq!(
            v2_pre_count, 3,
            "v2 has 3 edges pre-merge → eligible for use_verts mark"
        );

        let result = mesh.dissolve_faces(
            &[faces[0], faces[1], faces[2]],
            DissolveFacesOptions { use_verts: true },
        );

        assert_eq!(
            result.region.len(),
            1,
            "one connected region merges into one face"
        );

        // §6 post-pass collapses verts 1 and 2, each contributing -1 V and -1 E.
        // The region merge itself is -2 F, -2 E (two interior edges between
        // the three faces), 0 V. Net: -2 F, -4 E, -2 V.
        assert_eq!(
            mesh.face_count(),
            f0_count - 2,
            "-2 faces (3 input → 1 merged)"
        );
        assert_eq!(
            mesh.edge_count(),
            e0_count - 4,
            "-2 from merge, -2 from use_verts collapses"
        );
        assert_eq!(
            mesh.vertex_count(),
            v0_count - 2,
            "-2 verts: v1 and v2 collapsed by post-pass"
        );
        mesh.validate_topology();
    }

    /// §6.1 carve-out: a vert that was *already* edge-pair before the
    /// operator started is never marked, and the post-pass leaves it
    /// alone even if it is still edge-pair post-merge. Here verts 0 and 3
    /// of the 3x3 grid live at the corners of the bottom row; each has
    /// exactly two incident edges pre-merge (one along the bottom rim,
    /// one going up to row 1) — those edges *survive* the merge (they
    /// are boundary edges of the merged region), so the corners remain
    /// 2-edge post-merge. The post-pass must NOT collapse them.
    #[test]
    fn dissolve_faces_use_verts_preserves_pre_existing_edge_pair_corners() {
        let mut mesh = LoopMesh::new();
        let (verts, faces) = build_grid_3x3(&mut mesh);

        let v_bottom_left = verts[0];
        let v_bottom_right = verts[3];
        assert_eq!(
            mesh.vert_edge_keys(v_bottom_left).len(),
            2,
            "v0 is edge-pair pre-merge"
        );
        assert_eq!(
            mesh.vert_edge_keys(v_bottom_right).len(),
            2,
            "v3 is edge-pair pre-merge"
        );

        // The edges that hold these corners on the perimeter — to vert 4 above
        // and to vert 7 above respectively — must survive both the merge and
        // the post-pass.
        let v_above_left = verts[4];
        let v_above_right = verts[7];
        let e_left_up = mesh
            .query_edge_exists(v_bottom_left, v_above_left)
            .expect("edge {v0,v4} exists pre-merge");
        let e_right_up = mesh
            .query_edge_exists(v_bottom_right, v_above_right)
            .expect("edge {v3,v7} exists pre-merge");
        let _ = (e_left_up, e_right_up);

        mesh.dissolve_faces(
            &[faces[0], faces[1], faces[2]],
            DissolveFacesOptions { use_verts: true },
        );

        // Corners survive — the post-pass left them alone.
        assert!(
            mesh.query_edge_exists(v_bottom_left, v_above_left)
                .is_some(),
            "v0's pre-existing edge to v4 survives the use_verts post-pass"
        );
        assert!(
            mesh.query_edge_exists(v_bottom_right, v_above_right)
                .is_some(),
            "v3's pre-existing edge to v7 survives the use_verts post-pass"
        );
        // The corner verts themselves survive too, still edge-pair.
        assert_eq!(mesh.vert_edge_keys(v_bottom_left).len(), 2);
        assert_eq!(mesh.vert_edge_keys(v_bottom_right).len(), 2);
        mesh.validate_topology();
    }

    /// §6.3 / §8: a vert whose every incident edge was interior to the
    /// dissolved region is killed by `faces_join` itself (step 4 of §4.1).
    /// By the time the §6.1 sub-phase 1 walk runs, that vert is gone — its
    /// mark dies with the slab slot. No double-collapse, no use-after-free,
    /// and the post-pass simply doesn't visit it.
    ///
    /// Fixture: 2x2 face block (faces 6, 7, 11, 12) in a 5x5 face grid.
    /// The block's centre vert (the four-way join between the four quads)
    /// has all four incident edges interior to the merged region. The merge
    /// strips those edges and the vert is killed by step 4.
    #[test]
    fn dissolve_faces_use_verts_skips_vert_killed_by_merge() {
        let mut mesh = LoopMesh::new();
        let (verts, faces) = build_grid_5x5(&mut mesh);

        let v0_count = mesh.vertex_count();
        let e0_count = mesh.edge_count();
        let f0_count = mesh.face_count();
        assert_eq!(v0_count, 36);
        assert_eq!(f0_count, 25);

        // The centre vert of the 2x2 block: faces 6, 7, 11, 12 sit at
        // (r,c) = (1,1), (1,2), (2,1), (2,2) — so the shared corner is
        // vert at (r,c) = (2,2) = index 2*6 + 2 = 14.
        let v_centre = verts[14];
        assert_eq!(
            mesh.vert_edge_keys(v_centre).len(),
            4,
            "interior 5x5 vert has 4 incident edges pre-merge"
        );

        mesh.dissolve_faces(
            &[faces[6], faces[7], faces[11], faces[12]],
            DissolveFacesOptions { use_verts: true },
        );

        // The block merges to one face: -3 F. The interior shared edges
        // (four of them: between f6/f7, f6/f11, f7/f12, f11/f12) are killed
        // by the merge: -4 E. The centre vert (v14) is killed by step 4 of
        // faces_join: -1 V.
        //
        // No other use_verts collapses are expected: the perimeter verts of
        // the 2x2 block are 5x5-interior verts with 4 edges each pre-merge.
        // Post-merge they each lose 1 edge (the one pointing into the
        // dissolved interior), leaving 3 — still not edge-pair, so the
        // post-pass marks but does not collapse them.
        assert_eq!(mesh.face_count(), f0_count - 3);
        assert_eq!(mesh.edge_count(), e0_count - 4);
        assert_eq!(
            mesh.vertex_count(),
            v0_count - 1,
            "centre vert killed by merge, no use_verts collapse"
        );

        // The centre vert is gone — it does not appear in the live vert key list.
        let live: std::collections::HashSet<VertexKey> = mesh.vert_keys().into_iter().collect();
        assert!(
            !live.contains(&v_centre),
            "v14 (block centre) is freed by faces_join step 4 — never visited by sub-phase 1"
        );
        mesh.validate_topology();
    }

    /// `use_verts = false` baseline contrast: same fixture and input as
    /// `dissolve_faces_use_verts_collapses_newly_edge_pair_verts`. Without
    /// the post-pass the mid-rim verts 1 and 2 survive as 2-edge perimeter
    /// nodes of the merged rectangle (16 V, 22 E, 7 F).
    #[test]
    fn dissolve_faces_use_verts_false_baseline_keeps_mid_rim_verts() {
        let mut mesh = LoopMesh::new();
        let (verts, faces) = build_grid_3x3(&mut mesh);

        let v0_count = mesh.vertex_count();
        let e0_count = mesh.edge_count();
        let f0_count = mesh.face_count();

        let v1 = verts[1];
        let v2 = verts[2];

        let result = mesh.dissolve_faces(
            &[faces[0], faces[1], faces[2]],
            DissolveFacesOptions { use_verts: false },
        );

        assert_eq!(result.region.len(), 1);
        // -2 F (3 input → 1 merged), -2 E (two interior edges), 0 V — the
        // post-pass is off, so the mid-rim verts survive as edge-pair
        // boundary nodes of the merged rectangle.
        assert_eq!(mesh.face_count(), f0_count - 2);
        assert_eq!(mesh.edge_count(), e0_count - 2);
        assert_eq!(mesh.vertex_count(), v0_count);

        assert_eq!(
            mesh.vert_edge_keys(v1).len(),
            2,
            "v1 sits on the merged perimeter as a 2-edge node"
        );
        assert_eq!(
            mesh.vert_edge_keys(v2).len(),
            2,
            "v2 sits on the merged perimeter as a 2-edge node"
        );
        mesh.validate_topology();
    }

    /// §4.2 BMOP-wrapper path: when the per-region merge would build a
    /// face whose perimeter coincides with a pre-existing face, the
    /// shadow wins. `DissolveFacesResult.region` carries the shadow as
    /// the per-region survivor; the two source quads and the
    /// newly-built merged face are gone.
    ///
    /// Fixture mirrors the A/B scaffold's `shadow_face_over_two_quad_region`:
    /// two adjacent unit quads sharing interior edge {v1,v2}, overlaid
    /// by a third hexagonal face whose perimeter coincides with the
    /// outer rim of the two-quad region.
    #[test]
    fn dissolve_faces_double_face_detection_shadow_wins() {
        let mut mesh = LoopMesh::new();
        let v: Vec<VertexKey> = [
            (0.0, 0.0),
            (1.0, 0.0),
            (1.0, 1.0),
            (0.0, 1.0),
            (2.0, 0.0),
            (2.0, 1.0),
        ]
        .into_iter()
        .map(|(x, y)| mesh.create_vertex(Vec3::new(x, y, 0.0)))
        .collect();

        let e_01 = mesh.create_edge([v[0], v[1]], EdgeCreateFlags::NO_DUPLICATE);
        let e_12 = mesh.create_edge([v[1], v[2]], EdgeCreateFlags::NO_DUPLICATE);
        let e_23 = mesh.create_edge([v[2], v[3]], EdgeCreateFlags::NO_DUPLICATE);
        let e_30 = mesh.create_edge([v[3], v[0]], EdgeCreateFlags::NO_DUPLICATE);
        let e_14 = mesh.create_edge([v[1], v[4]], EdgeCreateFlags::NO_DUPLICATE);
        let e_45 = mesh.create_edge([v[4], v[5]], EdgeCreateFlags::NO_DUPLICATE);
        let e_52 = mesh.create_edge([v[5], v[2]], EdgeCreateFlags::NO_DUPLICATE);

        let f_left = mesh.face_create(
            &[v[0], v[1], v[2], v[3]],
            &[e_01, e_12, e_23, e_30],
            FaceCreateFlags::NONE,
        );
        let f_right = mesh.face_create(
            &[v[1], v[4], v[5], v[2]],
            &[e_14, e_45, e_52, e_12],
            FaceCreateFlags::NONE,
        );
        let f_shadow = mesh.face_create(
            &[v[0], v[1], v[4], v[5], v[2], v[3]],
            &[e_01, e_14, e_45, e_52, e_23, e_30],
            FaceCreateFlags::NONE,
        );

        assert_eq!(mesh.vertex_count(), 6);
        assert_eq!(mesh.edge_count(), 7);
        assert_eq!(mesh.face_count(), 3);

        let result = mesh.dissolve_faces(&[f_left, f_right], DissolveFacesOptions::default());

        assert_eq!(result.region.len(), 1, "one region, one successful merge");
        assert_eq!(
            result.region[0], f_shadow,
            "region.out carries the shadow, not the merge result"
        );
        assert_eq!(mesh.vertex_count(), 6);
        assert_eq!(mesh.edge_count(), 6);
        assert_eq!(mesh.face_count(), 1);

        let live: std::collections::HashSet<FaceKey> = mesh.face_keys().into_iter().collect();
        assert!(live.contains(&f_shadow));
        assert!(!live.contains(&f_left));
        assert!(!live.contains(&f_right));
        mesh.validate_topology();
    }

    /// Build the `nonmanifold_book(3)` topology: three triangles all
    /// sharing one spine edge `{v0, v1}`. Returns `(verts[5], faces[3])`
    /// with verts laid out as `[spine_start, spine_end, apex_0, apex_1,
    /// apex_2]` and each face winding `[v0, v1, apex_i]` (all three
    /// walking the spine in the same direction — that's what makes this
    /// fixture a genuine non-manifold join, not just three coplanar
    /// triangles).
    ///
    /// Counts: 5 V, 7 E (1 spine + 2 per page), 3 F.
    fn build_nonmanifold_book_3(mesh: &mut LoopMesh) -> (Vec<VertexKey>, Vec<FaceKey>) {
        let v: Vec<VertexKey> = [
            (0.0, 0.0, 0.0),     // 0: spine start
            (1.0, 0.0, 0.0),     // 1: spine end
            (0.5, 1.0, 0.0),     // 2: apex of face 0
            (0.5, -0.5, 0.866),  // 3: apex of face 1
            (0.5, -0.5, -0.866), // 4: apex of face 2
        ]
        .into_iter()
        .map(|(x, y, z)| mesh.create_vertex(Vec3::new(x, y, z)))
        .collect();

        let e_spine = mesh.create_edge([v[0], v[1]], EdgeCreateFlags::NO_DUPLICATE);
        let mut faces = Vec::with_capacity(3);
        for apex_idx in 2..5 {
            let e_v0_apex = mesh.create_edge([v[0], v[apex_idx]], EdgeCreateFlags::NO_DUPLICATE);
            let e_apex_v1 = mesh.create_edge([v[apex_idx], v[1]], EdgeCreateFlags::NO_DUPLICATE);
            // Winding [v0, v1, apex_i] — all three spine loops walk v0→v1.
            faces.push(mesh.face_create(
                &[v[0], v[1], v[apex_idx]],
                &[e_spine, e_apex_v1, e_v0_apex],
                FaceCreateFlags::NONE,
            ));
        }
        (v, faces)
    }

    /// §3.3 headline: when only *two* of the spine's three radial faces
    /// are in the input set, the spine edge is crossable for the walker
    /// (its "input-set count" is exactly 2) even though its raw radial
    /// count is 3. The two input faces merge into a single 4-corner
    /// kite face; the third face survives untouched; the spine edge
    /// survives with its radial cycle shortened from 3 to 1.
    ///
    /// Contrast the §3.3 walker-crossability path (here) with the
    /// §4.3 `BM_faces_join` `rlen > 2` reject (covered by the
    /// all-three input scaffold) — same fixture, different code path.
    #[test]
    fn dissolve_faces_non_manifold_edge_splits_two_of_three_input() {
        let mut mesh = LoopMesh::new();
        let (verts, faces) = build_nonmanifold_book_3(&mut mesh);
        let v0 = verts[0];
        let v1 = verts[1];
        let apex_0 = verts[2];
        let apex_1 = verts[3];

        assert_eq!(mesh.vertex_count(), 5);
        assert_eq!(mesh.edge_count(), 7);
        assert_eq!(mesh.face_count(), 3);
        let spine = mesh
            .query_edge_exists(v0, v1)
            .expect("spine edge {v0,v1} must exist pre-merge");
        assert_eq!(
            mesh.edge_face_keys(spine).len(),
            3,
            "spine has radial cycle of length 3 pre-merge"
        );

        // Input = first two pages only — face 2 stays outside the set.
        let f2 = faces[2];
        let result = mesh.dissolve_faces(&[faces[0], faces[1]], DissolveFacesOptions::default());

        assert_eq!(
            result.region.len(),
            1,
            "the two-of-three input forms one region"
        );
        let kite = result.region[0];

        // §7 row 2 for the kite region: -1 F (2 source → 1 merged); no
        // edge killed (the spine survives with f2's loop on it); no
        // verts killed (every region vert retains at least one edge).
        assert_eq!(
            mesh.face_count(),
            2,
            "f0 + f1 merged into the kite; f2 untouched"
        );
        assert_eq!(
            mesh.edge_count(),
            7,
            "spine survives — its radial dropped to 1, not 0"
        );
        assert_eq!(mesh.vertex_count(), 5);

        // The spine edge is still alive and its radial cycle is now the
        // singleton {f2}.
        let spine_post = mesh
            .query_edge_exists(v0, v1)
            .expect("spine {v0,v1} survives the merge");
        let spine_radial = mesh.edge_face_keys(spine_post);
        assert_eq!(spine_radial.len(), 1, "radial cycle dropped from 3 to 1");
        assert_eq!(
            spine_radial[0], f2,
            "the surviving radial loop belongs to f2"
        );

        // f2 is still alive, still has v0 and v1 in its corner cycle.
        let live: std::collections::HashSet<FaceKey> = mesh.face_keys().into_iter().collect();
        assert!(
            live.contains(&f2),
            "f2 (the page left out of the input set) survives"
        );
        let f2_verts: std::collections::HashSet<VertexKey> =
            mesh.face_vert_keys(f2).into_iter().collect();
        assert!(f2_verts.contains(&v0));
        assert!(f2_verts.contains(&v1));

        // The kite has exactly 4 corners — the symmetric difference of
        // the two input perimeters is `{v0, v1, apex_0, apex_1}` minus
        // the spine `{v0, v1}` shared edge, leaving 4 boundary edges
        // and 4 corners: apex_0 → v0 → apex_1 → v1.
        let kite_verts: Vec<VertexKey> = mesh.face_vert_keys(kite);
        assert_eq!(kite_verts.len(), 4, "merged face is a 4-corner kite");
        let kite_set: std::collections::HashSet<VertexKey> = kite_verts.iter().copied().collect();
        let expected: std::collections::HashSet<VertexKey> =
            [v0, v1, apex_0, apex_1].into_iter().collect();
        assert_eq!(kite_set, expected);

        // §4.1.5 winding: the merged perimeter must walk in a single
        // consistent direction. Two checks pin this against a winding bug
        // in the same-direction splice:
        //   (a) the ordered corner cycle visits each corner exactly once
        //       (no repeat, which a self-intersecting splice would
        //       produce), and
        //   (b) every consecutive corner pair is a real edge whose radial
        //       cycle references the kite exactly once — i.e. the kite
        //       sits on each of its perimeter edges with a single loop,
        //       the hallmark of a non-degenerate, consistently-wound face.
        let mut seen: std::collections::HashSet<VertexKey> = std::collections::HashSet::new();
        for &corner in &kite_verts {
            assert!(
                seen.insert(corner),
                "kite corner {corner:?} visited twice — winding self-intersects"
            );
        }
        for window_start in 0..kite_verts.len() {
            let a = kite_verts[window_start];
            let b = kite_verts[(window_start + 1) % kite_verts.len()];
            let perimeter_edge = mesh
                .query_edge_exists(a, b)
                .expect("each consecutive kite corner pair must be a real edge");
            let kite_loops_on_edge = mesh
                .edge_face_keys(perimeter_edge)
                .into_iter()
                .filter(|&f| f == kite)
                .count();
            assert_eq!(
                kite_loops_on_edge, 1,
                "kite must sit on perimeter edge {{{a:?},{b:?}}} exactly once"
            );
        }

        mesh.validate_topology();
    }

    /// Build the `nonmanifold_book(3)` topology and additionally fan two
    /// extra triangles off page 1's non-spine edge `{v0, apex_1}`, so that
    /// edge becomes non-manifold (radial cycle length 3) *in addition* to
    /// the spine. Returns `(verts, faces)` where `faces[0..3]` are the
    /// three book pages (winding `[v0, v1, apex_i]`, all spine loops walk
    /// v0→v1) and `faces[3..5]` are the two fins sharing `{v0, apex_1}`.
    ///
    /// This is the design-blocker fixture: merging pages 0 and 1 makes
    /// page 1 the `fb` of the keep-edge splice, and page 1 carries the
    /// second non-manifold edge `{v0, apex_1}`. A face-wide reversal of
    /// page 1 would corrupt the fins' radial cycle on that edge; the local
    /// splice must leave it intact.
    fn build_nonmanifold_book_3_with_fin_on_page1(
        mesh: &mut LoopMesh,
    ) -> (Vec<VertexKey>, Vec<FaceKey>) {
        let v: Vec<VertexKey> = [
            (0.0, 0.0, 0.0),     // 0: spine start
            (1.0, 0.0, 0.0),     // 1: spine end
            (0.5, 1.0, 0.0),     // 2: apex of page 0
            (0.5, -0.5, 0.866),  // 3: apex of page 1
            (0.5, -0.5, -0.866), // 4: apex of page 2
            (-0.5, 0.5, 0.3),    // 5: fin A apex
            (-0.5, 0.5, -0.3),   // 6: fin B apex
        ]
        .into_iter()
        .map(|(x, y, z)| mesh.create_vertex(Vec3::new(x, y, z)))
        .collect();

        let e_spine = mesh.create_edge([v[0], v[1]], EdgeCreateFlags::NO_DUPLICATE);
        let mut faces = Vec::with_capacity(5);
        for apex_idx in 2..5 {
            let e_v0_apex = mesh.create_edge([v[0], v[apex_idx]], EdgeCreateFlags::NO_DUPLICATE);
            let e_apex_v1 = mesh.create_edge([v[apex_idx], v[1]], EdgeCreateFlags::NO_DUPLICATE);
            faces.push(mesh.face_create(
                &[v[0], v[1], v[apex_idx]],
                &[e_spine, e_apex_v1, e_v0_apex],
                FaceCreateFlags::NONE,
            ));
        }

        // Two fins fanning off page 1's non-spine edge {v0, apex_1}.
        let e_v0_apex1 = mesh
            .query_edge_exists(v[0], v[3])
            .expect("page 1's {v0, apex_1} edge exists");
        for fin_apex in [v[5], v[6]] {
            let e_apex1_fin = mesh.create_edge([v[3], fin_apex], EdgeCreateFlags::NO_DUPLICATE);
            let e_fin_v0 = mesh.create_edge([fin_apex, v[0]], EdgeCreateFlags::NO_DUPLICATE);
            faces.push(mesh.face_create(
                &[v[0], v[3], fin_apex],
                &[e_v0_apex1, e_apex1_fin, e_fin_v0],
                FaceCreateFlags::NONE,
            ));
        }

        (v, faces)
    }

    /// The design-blocker guard (§4.1.5 + the local-splice fix): merging
    /// two pages of a book spine where the `fb` page *also* carries a
    /// second non-manifold edge must not corrupt that other edge's radial
    /// cycle. A face-wide reversal of `fb` would disconnect and reconnect
    /// every one of `fb`'s edges, mangling the fins on `{v0, apex_1}`. The
    /// local splice touches only the two doomed spine loops, so the fins
    /// survive with their radial cycle intact.
    #[test]
    fn dissolve_faces_non_manifold_edge_keeps_other_nonmanifold_edge_intact() {
        let mut mesh = LoopMesh::new();
        let (verts, faces) = build_nonmanifold_book_3_with_fin_on_page1(&mut mesh);
        let v0 = verts[0];
        let apex_1 = verts[3];
        let fin_a = faces[3];
        let fin_b = faces[4];

        // Pre-merge: page 1's {v0, apex_1} edge is non-manifold (page 1 +
        // two fins = radial 3).
        let other_edge = mesh
            .query_edge_exists(v0, apex_1)
            .expect("page 1's second non-manifold edge exists pre-merge");
        let pre_radial: std::collections::HashSet<FaceKey> =
            mesh.edge_face_keys(other_edge).into_iter().collect();
        assert_eq!(
            pre_radial.len(),
            3,
            "{{v0, apex_1}} is non-manifold pre-merge"
        );
        assert!(pre_radial.contains(&faces[1]), "page 1 is on the radial");
        assert!(pre_radial.contains(&fin_a));
        assert!(pre_radial.contains(&fin_b));

        // Merge pages 0 and 1 — page 1 becomes the keep-edge splice's `fb`,
        // and its spine loop walks the spine in the same direction as page
        // 0's (both `[v0, v1, apex]`), exercising the §4.1.5 same-direction
        // path.
        let result = mesh.dissolve_faces(&[faces[0], faces[1]], DissolveFacesOptions::default());
        assert_eq!(result.region.len(), 1, "the two pages form one region");
        let kite = result.region[0];

        // The fins' radial cycle on {v0, apex_1} must be uncorrupted: the
        // edge survives, the two fins are still on it, and page 1's old
        // loop has been replaced by the merged kite's loop (the kite now
        // borders {v0, apex_1}). Radial length stays 3.
        let other_edge_post = mesh
            .query_edge_exists(v0, apex_1)
            .expect("{{v0, apex_1}} survives the merge");
        let post_radial: std::collections::HashSet<FaceKey> =
            mesh.edge_face_keys(other_edge_post).into_iter().collect();
        assert_eq!(
            post_radial.len(),
            3,
            "the second non-manifold edge keeps radial length 3 after the merge"
        );
        assert!(post_radial.contains(&fin_a), "fin A radial loop intact");
        assert!(post_radial.contains(&fin_b), "fin B radial loop intact");
        assert!(
            post_radial.contains(&kite),
            "the merged kite borders {{v0, apex_1}} in place of the consumed page 1"
        );

        // The fins are still well-formed triangles.
        let fin_a_live: std::collections::HashSet<FaceKey> = mesh.face_keys().into_iter().collect();
        assert!(fin_a_live.contains(&fin_a));
        assert!(fin_a_live.contains(&fin_b));
        assert_eq!(
            mesh.face_vert_keys(fin_a).len(),
            3,
            "fin A is still a triangle"
        );
        assert_eq!(
            mesh.face_vert_keys(fin_b).len(),
            3,
            "fin B is still a triangle"
        );

        mesh.validate_topology();
    }

    /// Sanity check that the walker change doesn't regress the
    /// all-three-input case documented in the matrix row
    /// `dissolve_faces_non_manifold_book_spine_declines`: when every
    /// page is in the input set, the §3.3 predicate "input-set count
    /// == 2" fails on the spine (count = 3) so the walker doesn't
    /// cross, every triangle becomes a singleton, and §3.1 skips them
    /// all. Mesh unchanged.
    #[test]
    fn dissolve_faces_non_manifold_book_spine_all_three_declines() {
        let mut mesh = LoopMesh::new();
        let (_verts, faces) = build_nonmanifold_book_3(&mut mesh);

        let v0_count = mesh.vertex_count();
        let e0_count = mesh.edge_count();
        let f0_count = mesh.face_count();

        let result = mesh.dissolve_faces(
            &[faces[0], faces[1], faces[2]],
            DissolveFacesOptions::default(),
        );

        assert!(
            result.region.is_empty(),
            "every region is a singleton — §3.1 skips them all"
        );
        assert_eq!(mesh.vertex_count(), v0_count, "mesh unchanged");
        assert_eq!(mesh.edge_count(), e0_count, "mesh unchanged");
        assert_eq!(mesh.face_count(), f0_count, "mesh unchanged");
        mesh.validate_topology();
    }

    /// §4.1.4's row-of-three fixture: three quads in a row whose two interior
    /// edges are one genuinely manifold edge (`{1,4}`) and one radial-4 chord
    /// (`{2,3}`, which two out-of-region fins also use). Returns
    /// `(verts[12], [f0, f1, f2, fin_a, fin_b])`; `f2` is the survivor-to-be —
    /// its only interior edge is the kept-alive chord, so no kill reaches it.
    ///
    /// The region's boundary ring is the octagon `[0, 1, 2, 6, 7, 3, 4, 5]`.
    /// Counts: 12 V, 16 E, 5 F.
    fn build_quad_row_with_nonmanifold_chord(
        mesh: &mut LoopMesh,
    ) -> (Vec<VertexKey>, [FaceKey; 5]) {
        let coords: [(f32, f32, f32); 12] = [
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (2.0, 0.0, 0.0),
            (2.0, 1.0, 0.0),
            (1.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
            (3.0, 0.0, 0.0),
            (3.0, 1.0, 0.0),
            (2.0, 0.0, 1.0),
            (2.0, 1.0, 1.0),
            (2.0, 0.0, -1.0),
            (2.0, 1.0, -1.0),
        ];
        let v: Vec<VertexKey> = coords
            .into_iter()
            .map(|(x, y, z)| mesh.create_vertex(Vec3::new(x, y, z)))
            .collect();
        let mut quad = |mesh: &mut LoopMesh, ring: [usize; 4]| -> FaceKey {
            let verts: Vec<VertexKey> = ring.iter().map(|&i| v[i]).collect();
            mesh.face_create_verts(&verts, FaceCreateFlags::NONE)
        };
        let f0 = quad(mesh, [0, 1, 4, 5]);
        let f1 = quad(mesh, [1, 2, 3, 4]);
        let f2 = quad(mesh, [2, 6, 7, 3]);
        let fin_a = quad(mesh, [2, 3, 9, 8]);
        let fin_b = quad(mesh, [3, 2, 10, 11]);

        (v, [f0, f1, f2, fin_a, fin_b])
    }

    /// §5 — the join-survivor mop-up, on §4.1.4's row-of-three fixture. The join
    /// leaves `f2` alive (its only interior edge is the kept-alive radial-4
    /// chord) and the union overlapping it; the operator's cleanup pass deletes
    /// that survivor, so the *operator*'s face delta is the naive `-(n-1)` even
    /// though the primitive's underneath it is not. The chord and the survivor's
    /// boundary edges are untouched by the delete — it kills the face only.
    #[test]
    fn dissolve_faces_deletes_the_join_survivor_of_a_kept_alive_chord() {
        let mut mesh = LoopMesh::new();
        let (v, [f0, f1, f2, fin_a, fin_b]) = build_quad_row_with_nonmanifold_chord(&mut mesh);
        assert_eq!(mesh.face_count(), 5);

        let result = mesh.dissolve_faces(&[f0, f1, f2], DissolveFacesOptions::default());

        assert_eq!(result.region.len(), 1, "one region, one merge");
        let union = result.region[0];

        let live: std::collections::HashSet<FaceKey> = mesh.face_keys().into_iter().collect();
        assert!(
            !live.contains(&f2),
            "§5 deletes the join's survivor, which the union overlapped"
        );
        assert!(
            live.contains(&fin_a) && live.contains(&fin_b),
            "the fins are untouched"
        );
        assert_eq!(
            (mesh.vertex_count(), mesh.edge_count(), mesh.face_count()),
            (12, 15, 3),
            "the operator's delta is the naive -(n-1) F: union + the two fins"
        );
        assert_eq!(
            mesh.face_vert_keys(union).len(),
            8,
            "the union is still the row-spanning octagon"
        );

        // The mop-up's faces-context delete (§5.2) sweeps only what the survivor
        // solely owned, and it owned nothing solely: its boundary edges are on the
        // union's perimeter, and the chord still carries both fins — so the sweep
        // un-dooms every one of them and the delete takes the face alone. Contrast
        // the multi-region fixture below, where the chord *is* emptied and swept.
        let chord = mesh
            .query_edge_exists(v[2], v[3])
            .expect("the chord survives — the fins are still on it");
        assert_eq!(
            mesh.edge_face_keys(chord).len(),
            2,
            "only the fins remain on it"
        );
        assert!(mesh.query_edge_exists(v[6], v[7]).is_some());

        mesh.validate_topology();
    }

    /// §5's detector, on the one input that separates "expected = naive" from
    /// "expected = naive − d": a region that leaves **both** a §4.1.4 survivor
    /// and a §4.2 double.
    ///
    /// §4.1.4's row-of-three, plus a pre-existing octagon laid over the region's
    /// boundary ring. The ring edges each still carry exactly one input face, so
    /// classification is untouched — `{1,4}` is collected, the chord is kept
    /// alive, `f2` survives — and the union the join mints coincides with the
    /// octagon, so the join reports it as a double and the operator adopts it.
    ///
    /// One survivor (`s = 1`) and one double (`d = 1`) cancel out against the
    /// *naive* `−(n − 1)` count: the mesh lands exactly where "n in, 1 out"
    /// predicts, and a detector that compares against the naive count alone sees
    /// no mismatch, never fires, and leaves `f2` standing under the adopted
    /// octagon. Pricing the double into the expectation restores the signal.
    #[test]
    fn dissolve_faces_survivor_and_double_in_the_same_region() {
        let mut mesh = LoopMesh::new();
        let (v, [f0, f1, f2, fin_a, fin_b]) = build_quad_row_with_nonmanifold_chord(&mut mesh);

        let ring: Vec<VertexKey> = [0, 1, 2, 6, 7, 3, 4, 5].iter().map(|&i| v[i]).collect();
        let octagon = mesh.face_create_verts(&ring, FaceCreateFlags::NONE);

        assert_eq!(
            (mesh.vertex_count(), mesh.edge_count(), mesh.face_count()),
            (12, 16, 6),
            "the octagon reuses the eight ring edges — it adds a face and nothing else"
        );

        let result = mesh.dissolve_faces(&[f0, f1, f2], DissolveFacesOptions::default());

        assert_eq!(result.region.len(), 1, "one region, one merge");
        assert_eq!(
            result.region[0], octagon,
            "§4.2: the fresh union dies and the pre-existing octagon is reported"
        );

        let live: std::collections::HashSet<FaceKey> = mesh.face_keys().into_iter().collect();
        assert_eq!(
            live,
            [octagon, fin_a, fin_b].into_iter().collect(),
            "the double's survivor is mopped up too — nothing overlaps the adopted octagon"
        );
        assert_eq!(
            (mesh.vertex_count(), mesh.edge_count(), mesh.face_count()),
            (12, 15, 3),
            "-1 E ({{1,4}}); -3 F: the three inputs go, the adopted octagon stays"
        );

        // Same as the survivor-only case: the survivor solely owned nothing, so
        // the §5.2 sweep un-dooms its whole boundary and the delete takes the
        // face alone.
        let chord = mesh
            .query_edge_exists(v[2], v[3])
            .expect("the chord survives — the fins still need it");
        assert_eq!(
            mesh.edge_face_keys(chord).len(),
            2,
            "only the fins remain on it"
        );
        assert_eq!(
            mesh.face_vert_keys(octagon).len(),
            8,
            "the adopted octagon keeps its own perimeter"
        );

        mesh.validate_topology();
    }

    /// §5 — the mop-up is a *single pass after every region*, not a per-region
    /// one. Two regions share a chord; region A leaves a survivor on it
    /// (§4.1.4), and region B classifies the same chord by the faces then on
    /// it. With A's survivor still standing the chord carries three faces
    /// (survivor + B's two) → total > 2 → keep-alive (§4.1.3), so B leaves it
    /// in place. Kill the survivor before B runs and the chord drops to two
    /// faces → B collects it into `kill_edges` and destroys an edge the
    /// reference keeps.
    ///
    /// Fixture (topology; coordinates are only there to make the mesh legal):
    ///
    /// - Three quads `fa0`, `fa1`, `fs` fan around a centre vert `c`, tiling a
    ///   hexagon. `fa0`/`fa1` share spoke `{c,d}` (genuinely manifold → killed
    ///   → both die). `fa1`/`fs` share spoke `{c,b}`, which an out-of-input
    ///   triangle `x` also uses → keep-alive → crossable by the walker (input
    ///   count 2) but collected by nobody. `fs`/`fa0` share the **chord**
    ///   `{c,a}`.
    /// - Two quads `g0`, `g1` also sit on that chord and additionally share
    ///   the manifold edge `{a,s}` with each other — that edge is what makes
    ///   them a region of their own (the chord itself carries four input faces,
    ///   so the §3.3 "exactly two" gate refuses to cross it).
    ///
    /// So `fs` is region A's survivor: both its interior edges are keep-alive.
    #[test]
    fn dissolve_faces_survivor_of_one_region_survives_until_all_regions_done() {
        let mut mesh = LoopMesh::new();
        let coords: [(f32, f32, f32); 11] = [
            (0.0, 0.0, 0.0),     // 0: c — hexagon centre
            (1.0, 0.0, 0.0),     // 1: a — chord endpoint
            (0.5, 0.866, 0.0),   // 2: p
            (-0.5, 0.866, 0.0),  // 3: b
            (-1.0, 0.0, 0.0),    // 4: q
            (-0.5, -0.866, 0.0), // 5: d
            (0.5, -0.866, 0.0),  // 6: r
            (-0.25, 0.433, 1.0), // 7: x — apex of the out-of-input triangle
            (1.5, -1.0, 0.0),    // 8: s
            (0.5, -1.2, 1.0),    // 9: t
            (0.5, -1.2, -1.0),   // 10: u
        ];
        let v: Vec<VertexKey> = coords
            .into_iter()
            .map(|(x, y, z)| mesh.create_vertex(Vec3::new(x, y, z)))
            .collect();
        let (c, a, p, b, q, d, r, x, s, t, u) = (
            v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8], v[9], v[10],
        );
        let mut face = |mesh: &mut LoopMesh, ring: &[VertexKey]| -> FaceKey {
            mesh.face_create_verts(ring, FaceCreateFlags::NONE)
        };
        let fs = face(&mut mesh, &[c, a, p, b]);
        let fa1 = face(&mut mesh, &[c, b, q, d]);
        let fa0 = face(&mut mesh, &[c, d, r, a]);
        let f_out = face(&mut mesh, &[b, c, x]);
        let g0 = face(&mut mesh, &[c, a, s, t]);
        let g1 = face(&mut mesh, &[a, c, u, s]);

        assert_eq!(
            (mesh.vertex_count(), mesh.edge_count(), mesh.face_count()),
            (11, 16, 6)
        );
        let chord = mesh
            .query_edge_exists(c, a)
            .expect("the shared chord {c,a} exists");
        assert_eq!(
            mesh.edge_face_keys(chord).len(),
            4,
            "the chord carries both of region A's faces and both of region B's"
        );

        let result = mesh.dissolve_faces(&[fa0, fa1, fs, g0, g1], DissolveFacesOptions::default());

        assert_eq!(result.region.len(), 2, "two regions, two successful merges");

        // Identify the live faces positively: the two unions plus the
        // out-of-input triangle, and nothing else. (Comparing against the dead
        // input keys would be unsound — a union minted after a kill can be
        // handed the dead face's recycled slot.)
        let live: std::collections::HashSet<FaceKey> = mesh.face_keys().into_iter().collect();
        let expected_live: std::collections::HashSet<FaceKey> =
            [result.region[0], result.region[1], f_out]
                .into_iter()
                .collect();
        assert_eq!(
            live, expected_live,
            "region A's survivor is mopped up — but only after region B ran"
        );
        assert_eq!(
            mesh.face_vert_keys(f_out).len(),
            3,
            "the out-of-input triangle is untouched"
        );

        // The load-bearing assertion: region B saw a three-face chord (A's
        // survivor plus its own two) and left it alone as keep-alive. A
        // per-region mop-up would have shortened it to two, B would have
        // collected it into `kill_edges`, and its cascade would have run
        // through it — a different classification, and a different union
        // perimeter in the general case.
        //
        // The chord nevertheless does not outlive the operator: once the
        // survivor — its last remaining face — is deleted, the §5 faces-context
        // sweep (§5.2) takes the emptied edge with it rather than leaving a
        // wire edge behind. Its endpoints live: `c` still sits on region B's
        // union and the spoke to the triangle, `a` on the hexagon rim.
        assert!(
            mesh.query_edge_exists(c, a).is_none(),
            "the chord is emptied by the survivor's deletion and swept with it"
        );
        assert!(
            mesh.query_edge_exists(c, b).is_some(),
            "the keep-alive spoke keeps the out-of-input triangle on it, so it stays"
        );
        assert_eq!(
            (mesh.vertex_count(), mesh.edge_count(), mesh.face_count()),
            (11, 13, 3),
            "-3 E: {{c,d}} (region A's join), {{a,s}} (region B's join), and the \
             chord (§5's sweep). No vert dies. -3 F: 5 input faces → 2 unions"
        );

        let hexagon = result.region[0];
        assert_eq!(
            mesh.face_vert_keys(hexagon).len(),
            6,
            "region A's union spans the whole hexagon rim"
        );
        assert_eq!(
            mesh.face_vert_keys(result.region[1]).len(),
            4,
            "region B's union is the quad on its four boundary edges"
        );

        mesh.validate_topology();
    }

    /// §5.2's second corollary: the survivor's deletion can orphan a **vertex**,
    /// and the faces-context sweep takes it too. Same mechanism as the orphaned
    /// chord — no extra machinery — but it needs a hub whose *entire* disk either
    /// dies in a join or empties out with the survivor.
    ///
    /// Fixture. `w` is the hub of a three-triangle wheel with rim `c, g, a`:
    /// `fs = (w,a,c)`, `m1 = (w,c,g)`, `m2 = (w,g,a)`. Region A also takes the
    /// quad `m0 = (a,c,g,z)` hanging off the rim.
    ///
    /// - `{w,c}` and `{w,a}` are **chords**: each carries two of region A's faces
    ///   *and* two faces of a two-quad region of its own (B on `{w,c}`, C on
    ///   `{w,a}`) — four input faces, so the §3.3 gate refuses to cross them and
    ///   both classify keep-alive for every region that sees them.
    /// - `{a,c}` is the crossable keep-alive spoke that pulls `fs` into region A
    ///   (`fs` + `m0` in the input, plus the out-of-input triangle `t_out`).
    /// - `{c,g}` and `{w,g}` are genuinely manifold interior edges of region A →
    ///   collected → their cascade takes `m0`, `m1`, `m2`. `fs`, whose every edge
    ///   is keep-alive, survives (§4.1.4).
    /// - Regions B and C are each a pinch pair sharing a manifold edge **at `w`**
    ///   (`{w,s_b}`, `{w,s_c}`), so their joins kill those and neither union has
    ///   `w` on its perimeter.
    ///
    /// So when §5 deletes `fs`, both chords are emptied — and they were `w`'s last
    /// two edges. A faces-only kill would leave two wire edges *and* a loose point.
    #[test]
    fn dissolve_faces_survivor_deletion_orphans_a_vert_which_is_swept_too() {
        let mut mesh = LoopMesh::new();
        let coords: [(f32, f32, f32); 12] = [
            (0.0, 0.0, 0.0),     // 0: w — the wheel hub
            (1.0, 0.0, 0.0),     // 1: c
            (-0.5, 0.866, 0.0),  // 2: g
            (-0.5, -0.866, 0.0), // 3: a
            (-0.2, 0.0, 1.5),    // 4: z
            (0.5, -0.5, -1.0),   // 5: x  — apex of the out-of-input triangle
            (0.8, 0.9, 1.2),     // 6: s_b
            (1.2, 0.2, 1.0),     // 7: t_b
            (1.2, 0.2, -1.0),    // 8: u_b
            (-1.0, -1.2, 1.0),   // 9: s_c
            (-0.8, -0.6, 1.2),   // 10: t_c
            (-0.8, -0.6, -1.2),  // 11: u_c
        ];
        let v: Vec<VertexKey> = coords
            .into_iter()
            .map(|(x, y, z)| mesh.create_vertex(Vec3::new(x, y, z)))
            .collect();
        let (w, c, g, a, z, x, s_b, t_b, u_b, s_c, t_c, u_c) = (
            v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8], v[9], v[10], v[11],
        );
        let mut face = |mesh: &mut LoopMesh, ring: &[VertexKey]| -> FaceKey {
            mesh.face_create_verts(ring, FaceCreateFlags::NONE)
        };
        let fs = face(&mut mesh, &[w, a, c]);
        let m0 = face(&mut mesh, &[a, c, g, z]);
        let m1 = face(&mut mesh, &[w, c, g]);
        let m2 = face(&mut mesh, &[w, g, a]);
        let t_out = face(&mut mesh, &[a, c, x]);
        let b0 = face(&mut mesh, &[c, w, s_b, t_b]);
        let b1 = face(&mut mesh, &[w, c, u_b, s_b]);
        let c0 = face(&mut mesh, &[a, w, s_c, t_c]);
        let c1 = face(&mut mesh, &[w, a, u_c, s_c]);

        assert_eq!(
            (mesh.vertex_count(), mesh.edge_count(), mesh.face_count()),
            (12, 20, 9)
        );
        let chord_wc = mesh.query_edge_exists(w, c).expect("chord {w,c}");
        let chord_wa = mesh.query_edge_exists(w, a).expect("chord {w,a}");
        assert_eq!(
            mesh.edge_face_keys(chord_wc).len(),
            4,
            "{{w,c}} is radial-4"
        );
        assert_eq!(
            mesh.edge_face_keys(chord_wa).len(),
            4,
            "{{w,a}} is radial-4"
        );
        assert_eq!(
            mesh.vert_edge_keys(w).len(),
            5,
            "w's disk: the two chords, the wheel spoke {{w,g}}, and B's and C's \
             own manifold edges"
        );

        let result = mesh.dissolve_faces(
            &[fs, m0, m1, m2, b0, b1, c0, c1],
            DissolveFacesOptions::default(),
        );

        assert_eq!(result.region.len(), 3, "regions A, B and C each merge");

        let live: std::collections::HashSet<FaceKey> = mesh.face_keys().into_iter().collect();
        let expected_live: std::collections::HashSet<FaceKey> =
            [result.region[0], result.region[1], result.region[2], t_out]
                .into_iter()
                .collect();
        assert_eq!(
            live, expected_live,
            "three unions plus the out-of-input triangle; region A's survivor is mopped up"
        );

        // The sweep's two halves: the emptied chords go, and w — whose last two
        // edges they were — goes with them. (w is dead, so probe it only through
        // the live-key listings; a disk walk from a freed vert is not a query,
        // it is a use-after-free.)
        let live_verts: std::collections::HashSet<VertexKey> =
            mesh.vert_keys().into_iter().collect();
        assert!(
            !live_verts.contains(&w),
            "w is left with an empty disk by the chord sweep, so the vert sweep takes it"
        );
        assert!(
            live_verts.contains(&c) && live_verts.contains(&a),
            "the chords' other endpoints keep other edges and survive"
        );
        for edge in mesh.edge_keys() {
            let [e0, e1] = mesh.edge_vert_keys(edge);
            assert!(
                e0 != w && e1 != w,
                "no edge may still reference w — both chords were swept"
            );
        }
        assert!(
            mesh.query_edge_exists(a, c).is_some(),
            "the survivor's third edge keeps the out-of-input triangle on it and stays"
        );

        // -6 E: {c,g} and {w,g} (region A's join), {w,s_b} (B's), {w,s_c} (C's),
        // and the two swept chords. -1 V: w. -5 F: 8 input faces → 3 unions.
        assert_eq!(
            (mesh.vertex_count(), mesh.edge_count(), mesh.face_count()),
            (11, 14, 4)
        );

        mesh.validate_topology();
    }
}
