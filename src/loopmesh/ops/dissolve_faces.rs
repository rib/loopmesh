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
    /// `use_verts` post-pass (§6). The double-face detection of §4.2 is not
    /// yet implemented; on inputs that would hit it the merged face is
    /// reported as the survivor.
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
                // §4 — the many-input merge primitive does the rest.
                if let Some(merged) = self.faces_join(&region) {
                    result.region.push(merged);
                }
                // Failed merges (§4.3) leave the region's input faces
                // intact; we contribute nothing to `region.out` for that
                // region.
            }

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
                // §3 + §3.3 — manifold gate: radial must be exactly length 2.
                let mut radial = self.radial_loops(edge);
                let l0 = match radial.next(self) {
                    Some(l) => l,
                    None => continue,
                };
                let l1 = match radial.next(self) {
                    Some(l) => l,
                    None => continue,
                };
                if radial.next(self).is_some() {
                    continue;
                }
                let f0 = self.loops.get(l0).face;
                let f1 = self.loops.get(l1).face;
                let neighbour = if f0 == face {
                    f1
                } else if f1 == face {
                    f0
                } else {
                    continue;
                };
                if neighbour == face {
                    continue;
                }
                if !self.face_op_flag_test(in_input, neighbour, 0) {
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

        let v0 = mesh.totvert();
        let e0 = mesh.totedge();
        let f0 = mesh.totface();
        assert_eq!(v0, 16);
        assert_eq!(e0, 24, "3x3 grid: 4*3 horizontal + 4*3 vertical = 24 edges");
        assert_eq!(f0, 9);

        // Dissolve the bottom-left two quads (faces[0] and faces[1]) which
        // share the vertical edge between (1,0)-(1,1).
        let result = mesh.dissolve_faces(&[faces[0], faces[1]], Default::default());

        assert_eq!(result.region.len(), 1, "one region, one successful merge");
        let merged = result.region[0];

        assert_eq!(mesh.totface(), f0 - 1, "-1 face per §7 row 2");
        assert_eq!(mesh.totedge(), e0 - 1, "-1 edge per §7 row 2");
        assert_eq!(mesh.totvert(), v0, "0 verts removed per §7 row 2");

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
        let v0 = mesh.totvert();
        let e0 = mesh.totedge();
        let f0 = mesh.totface();

        let result = mesh.dissolve_faces(&[], Default::default());

        assert!(result.region.is_empty());
        assert_eq!(mesh.totvert(), v0);
        assert_eq!(mesh.totedge(), e0);
        assert_eq!(mesh.totface(), f0);
        mesh.validate_topology();
    }

    #[test]
    fn dissolve_faces_single_face_input_is_noop() {
        // §3.1 — a region of size 1 is silently skipped: the walker
        // requires at least one manifold step to open a region.
        let mut mesh = LoopMesh::new();
        let (_verts, faces) = build_grid_3x3(&mut mesh);
        let v0 = mesh.totvert();
        let e0 = mesh.totedge();
        let f0 = mesh.totface();

        let result = mesh.dissolve_faces(&[faces[0]], Default::default());

        assert!(result.region.is_empty(), "singleton region contributes nothing to region.out");
        assert_eq!(mesh.totvert(), v0);
        assert_eq!(mesh.totedge(), e0);
        assert_eq!(mesh.totface(), f0);
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
        let expected: std::collections::HashSet<VertexKey> = [
            verts[0], verts[1], verts[2], verts[6], verts[5], verts[4],
        ]
        .into_iter()
        .collect();

        let result = mesh.dissolve_faces(&[faces[0], faces[1]], Default::default());
        let merged = result.region[0];
        let got: std::collections::HashSet<VertexKey> = mesh.face_vert_keys(merged).into_iter().collect();
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

        let v0_count = mesh.totvert();
        let e0_count = mesh.totedge();
        let f0_count = mesh.totface();
        assert_eq!(v0_count, 16);
        assert_eq!(e0_count, 24);
        assert_eq!(f0_count, 9);

        let v1 = verts[1];
        let v2 = verts[2];
        let v1_pre_count = mesh.vert_edge_keys(v1).len();
        let v2_pre_count = mesh.vert_edge_keys(v2).len();
        assert_eq!(v1_pre_count, 3, "v1 has 3 edges pre-merge → eligible for use_verts mark");
        assert_eq!(v2_pre_count, 3, "v2 has 3 edges pre-merge → eligible for use_verts mark");

        let result = mesh.dissolve_faces(
            &[faces[0], faces[1], faces[2]],
            DissolveFacesOptions { use_verts: true },
        );

        assert_eq!(result.region.len(), 1, "one connected region merges into one face");

        // §6 post-pass collapses verts 1 and 2, each contributing -1 V and -1 E.
        // The region merge itself is -2 F, -2 E (two interior edges between
        // the three faces), 0 V. Net: -2 F, -4 E, -2 V.
        assert_eq!(mesh.totface(), f0_count - 2, "-2 faces (3 input → 1 merged)");
        assert_eq!(mesh.totedge(), e0_count - 4, "-2 from merge, -2 from use_verts collapses");
        assert_eq!(mesh.totvert(), v0_count - 2, "-2 verts: v1 and v2 collapsed by post-pass");
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
        assert_eq!(mesh.vert_edge_keys(v_bottom_left).len(), 2, "v0 is edge-pair pre-merge");
        assert_eq!(mesh.vert_edge_keys(v_bottom_right).len(), 2, "v3 is edge-pair pre-merge");

        // The edges that hold these corners on the perimeter — to vert 4 above
        // and to vert 7 above respectively — must survive both the merge and
        // the post-pass.
        let v_above_left = verts[4];
        let v_above_right = verts[7];
        let e_left_up = mesh.query_edge_exists(v_bottom_left, v_above_left)
            .expect("edge {v0,v4} exists pre-merge");
        let e_right_up = mesh.query_edge_exists(v_bottom_right, v_above_right)
            .expect("edge {v3,v7} exists pre-merge");
        let _ = (e_left_up, e_right_up);

        mesh.dissolve_faces(
            &[faces[0], faces[1], faces[2]],
            DissolveFacesOptions { use_verts: true },
        );

        // Corners survive — the post-pass left them alone.
        assert!(
            mesh.query_edge_exists(v_bottom_left, v_above_left).is_some(),
            "v0's pre-existing edge to v4 survives the use_verts post-pass"
        );
        assert!(
            mesh.query_edge_exists(v_bottom_right, v_above_right).is_some(),
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

        let v0_count = mesh.totvert();
        let e0_count = mesh.totedge();
        let f0_count = mesh.totface();
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
        assert_eq!(mesh.totface(), f0_count - 3);
        assert_eq!(mesh.totedge(), e0_count - 4);
        assert_eq!(mesh.totvert(), v0_count - 1, "centre vert killed by merge, no use_verts collapse");

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

        let v0_count = mesh.totvert();
        let e0_count = mesh.totedge();
        let f0_count = mesh.totface();

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
        assert_eq!(mesh.totface(), f0_count - 2);
        assert_eq!(mesh.totedge(), e0_count - 2);
        assert_eq!(mesh.totvert(), v0_count);

        assert_eq!(mesh.vert_edge_keys(v1).len(), 2, "v1 sits on the merged perimeter as a 2-edge node");
        assert_eq!(mesh.vert_edge_keys(v2).len(), 2, "v2 sits on the merged perimeter as a 2-edge node");
        mesh.validate_topology();
    }
}
