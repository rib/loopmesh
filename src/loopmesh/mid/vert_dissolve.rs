use crate::loopmesh::*;

impl LoopMesh {
    /// Dissolves `v` — collapses it away by either killing it (isolated),
    /// collapsing the surrounding degree-2 chain, or merging its surrounding
    /// face fan and collapsing.
    ///
    /// Returns `true` on success. An **unsupported** vertex (non-manifold or
    /// wire/face-mixed disk, pure-boundary fan with no interior spoke, a fan
    /// pinched by a face that uses one spoke twice) is refused before any
    /// mutation: `false`, mesh unchanged.
    ///
    /// A vertex that *is* supported but whose surgery fails part-way through —
    /// a join or the final collapse rejecting a shape the pre-classification
    /// admits — returns `false` with the mesh **partially merged**. That is
    /// deliberate: the fan surgery is destructive-on-false in this shape
    /// (`DISSOLVE_VERTS.md` §6.7's UNCHARACTERISED box), so a caller must not
    /// read `false` as "nothing happened".
    pub fn vert_dissolve(&mut self, v: VertexKey) -> bool {
        // Snapshot v's disk edges + per-edge radial counts in a single walk,
        // off the heap for typical valences.
        let edges: smallvec::SmallVec<[EdgeKey; 8]> =
            collect_iter::<_, 8>(self, self.disk_edges(v));
        let ne = edges.len();

        // Isolated vertex: just kill it.
        if ne == 0 {
            self.vertex_kill(v);
            return true;
        }

        // Wire-only vertex with one edge: kill (cascades to the edge).
        if ne == 1 {
            self.vertex_kill(v);
            return true;
        }

        // Pre-classify the topology so we never start mutating before
        // confirming the operation is supported.
        let radial_counts: smallvec::SmallVec<[usize; 8]> = edges
            .iter()
            .map(|&e| {
                let mut it = self.radial_loops(e);
                let mut c = 0usize;
                while it.next(self).is_some() {
                    c += 1;
                }
                c
            })
            .collect();
        let is_wire = radial_counts.iter().all(|&c| c == 0);

        // Degree-2 wire chain → collapse via JEKV (vert_collapse_edge).
        if ne == 2 && is_wire {
            return self.vert_collapse_edge(edges[0], v).is_some();
        }

        // Degree-2 boundary vertex on a single face → also JEKV.
        // Count unique faces incident to v.
        if ne == 2 {
            let mut faces: smallvec::SmallVec<[FaceKey; 4]> = smallvec::SmallVec::new();
            let mut disk = self.disk_edges(v);
            while let Some(e) = disk.next(self) {
                let mut rl = self.radial_loops(e);
                while let Some(l) = rl.next(self) {
                    let f = self.loops.get(l).face;
                    if !faces.contains(&f) {
                        faces.push(f);
                    }
                }
            }
            if faces.len() == 1 {
                return self.vert_collapse_edge(edges[0], v).is_some();
            }
        }

        // The "disk dissolve" algorithm works for any vertex whose disk
        // contains only boundary (radial=1) and interior (radial=2) edges,
        // as long as at least one interior edge is present to fold the
        // surrounding fan down to a degree-2 boundary collapse. This
        // includes:
        //   - the pure manifold-interior case (all radial=2)
        //   - the mixed boundary-fan case (vertex sits on the open edge of
        //     a strip of faces; some edges boundary, some interior).
        let no_non_manifold_no_wire = radial_counts.iter().all(|&c| c == 1 || c == 2);
        let has_interior_edge = radial_counts.iter().any(|&c| c == 2);
        // A radial-2 spoke whose two loops belong to the *same* face pinches the
        // fan: no join can consume it, so the merge walk would stall half-way and
        // report failure over an already-mutated mesh. Refuse it here, while the
        // mesh is still untouched.
        let has_pinched_spoke = edges
            .iter()
            .zip(&radial_counts)
            .any(|(&e, &count)| count == 2 && self.spoke_face_pair(e).is_none());
        if no_non_manifold_no_wire && has_interior_edge && !has_pinched_spoke {
            return self.vert_dissolve_manifold_disk(v);
        }

        // Anything else (pure-boundary fan with no shared edges, non-manifold
        // edges, wire/face mix, bowtie): refuse.
        false
    }

    /// Inner: dissolve a vertex whose incident edges are all boundary
    /// (radial-1) or interior (radial-2). Caller must have verified the
    /// manifold precondition.
    ///
    /// A fully-interior fan with a reservable *keep* spoke runs the helper's
    /// **halves schedule** (`DISSOLVE_VERTS.md` §6.7): two spokes are held out
    /// of the merge, cutting the face ring into two arcs that are merged
    /// independently and joined last. That schedule — not §6.3's single
    /// running-survivor chain — is what fixes the merged n-gon's per-face
    /// customdata donor for this entry point.
    ///
    /// Everything else (the no-keep low-valence branch, and **every** boundary
    /// fan, whatever its disk order) keeps the plain "join any radial-2 spoke
    /// until degree-2, then collapse" walk. That walk fully dissolves a `k ≥ 3`
    /// boundary fan, where §6.7 records the reference as partially merging and
    /// reporting failure — an open divergence on an UNCHARACTERISED row, not a
    /// claim of parity.
    fn vert_dissolve_manifold_disk(&mut self, v: VertexKey) -> bool {
        if let Some((base_spoke, keep_spoke)) = self.vert_dissolve_reserved_spokes(v) {
            return self.vert_dissolve_halves_schedule(v, base_spoke, keep_spoke);
        }

        loop {
            // Snapshot current disk edges (off the heap for small valences).
            let edges: smallvec::SmallVec<[EdgeKey; 8]> =
                collect_iter::<_, 8>(self, self.disk_edges(v));
            if edges.len() <= 2 {
                break;
            }
            // Find any edge at v whose radial has exactly 2 faces and
            // join those two faces along it.
            let mut merged = false;
            for &e in &edges {
                let mut rl = self.radial_loops(e);
                let l0 = match rl.next(self) {
                    Some(l) => l,
                    None => continue,
                };
                let l1 = match rl.next(self) {
                    Some(l) => l,
                    None => continue,
                };
                if rl.next(self).is_some() {
                    continue;
                }
                let f0 = self.loops.get(l0).face;
                let f1 = self.loops.get(l1).face;
                if f0 != f1 && self.faces_join_pair(f0, f1, e).is_some() {
                    merged = true;
                    break;
                }
            }
            if !merged {
                // Shouldn't happen given the precondition, but guard against
                // infinite loops in case of pathological inputs.
                return false;
            }
        }

        // v is now degree-2 (one face on either side of two co-linear edges,
        // or a single face that touches v twice). Collapse it through the
        // reaping-enabled primitive so a triangle corner is reaped to a 2-gon
        // and any coincident doubled face the collapse produces is welded away
        // (DISSOLVE_VERTS.md §6.6) — this is what closes a degenerating fan onto
        // a single clean face rather than a doubled-face remnant.
        let final_edges: smallvec::SmallVec<[EdgeKey; 2]> =
            collect_iter::<_, 2>(self, self.disk_edges(v));
        if final_edges.len() != 2 {
            return false;
        }
        let surviving_edge = self
            .vert_collapse_edge(final_edges[0], v)
            .or_else(|| self.vert_collapse_edge(final_edges[1], v));
        let Some(surviving_edge) = surviving_edge else {
            return false;
        };

        // If the surviving edge is still a 2-manifold interior edge, the two
        // faces it sits between are the residue of the original disk and
        // should be merged.
        if self.edges.try_get(surviving_edge).is_none() {
            return true;
        }
        let mut rl = self.radial_loops(surviving_edge);
        let l0 = rl.next(self);
        let l1 = rl.next(self);
        let l2 = rl.next(self);
        if let (Some(l0), Some(l1), None) = (l0, l1, l2) {
            let f0 = self.loops.get(l0).face;
            let f1 = self.loops.get(l1).face;
            if f0 != f1 {
                self.faces_join_pair(f0, f1, surviving_edge);
            }
        }
        true
    }

    /// The two spokes the helper reserves on a fully-interior fan
    /// (`DISSOLVE_VERTS.md` §6.7): the **base** (`v`'s disk anchor) and the
    /// **keep** — the first spoke *in disk order* forward from the anchor that
    /// shares no face with it. Disk order is attachment order, not the
    /// geometric ring order, so the keep spoke is read off the disk cycle and
    /// never inferred from the fan's shape.
    ///
    /// `None` unless the **whole disk** is the fully-interior fan §6.7 derives
    /// the halves model for: every spoke radial-2 between two distinct faces.
    /// A boundary (radial-1) spoke is never joined by the merge scan, so it
    /// would still stand at `v` when the collapse runs and the schedule is not
    /// well-formed — that verdict is a property of the fan, so it must be
    /// decided over the whole disk and not fall out of where the keep search
    /// happens to stop. Also `None` when no keep spoke exists (every spoke
    /// shares a face with the anchor — §6.7's low-valence / no-keep branch).
    /// Both fall back to the plain merge-then-collapse walk.
    fn vert_dissolve_reserved_spokes(&self, v: VertexKey) -> Option<(EdgeKey, EdgeKey)> {
        let mut spokes: smallvec::SmallVec<[(EdgeKey, FaceKey, FaceKey); 8]> =
            smallvec::SmallVec::new();
        let mut disk = self.disk_edges(v);
        while let Some(spoke) = disk.next(self) {
            let (face_a, face_b) = self.spoke_face_pair(spoke)?;
            spokes.push((spoke, face_a, face_b));
        }

        let &(base_spoke, base_face_a, base_face_b) = spokes.first()?;
        spokes
            .iter()
            .skip(1)
            .find(|&&(_, face_a, face_b)| {
                let shares_face_with_base = face_a == base_face_a
                    || face_a == base_face_b
                    || face_b == base_face_a
                    || face_b == base_face_b;
                !shares_face_with_base
            })
            .map(|&(keep_spoke, _, _)| (base_spoke, keep_spoke))
    }

    /// §6.7 — merge the fan's two halves, collapse `v` along the base spoke,
    /// then join the halves across the surviving edge.
    ///
    /// The donor of every pairwise join is the **most recently created** of the
    /// two faces flanking the joined spoke, and that face is exactly the one the
    /// spoke's radial **entry** names (§6.3's entry-reseat rule) — so the donor
    /// is read straight off the entry, both for the fan's joins and for the
    /// post-collapse one. The entry of the surviving edge is re-read *after* the
    /// collapse, never cached across it (§6.3, bound 3).
    ///
    /// The reading stays true across joins because the merge used here is the
    /// *composed* (loop-minting) join, whose constructed union enters every
    /// perimeter edge it kept on its own loop — including across *calls*, which
    /// the overlapping fans of a repeated coarsening rely on: a merged face that
    /// a later fan re-joins must still score as the newest face on its own
    /// perimeter, not as pristine.
    fn vert_dissolve_halves_schedule(
        &mut self,
        v: VertexKey,
        base_spoke: EdgeKey,
        keep_spoke: EdgeKey,
    ) -> bool {
        loop {
            // Restart the scan from the anchor after every join, per §6.7 — the
            // join deletes the spoke it fires on, so the scan makes progress.
            let mut next_spoke = None;
            let mut disk = self.disk_edges(v);
            while let Some(spoke) = disk.next(self) {
                if spoke == base_spoke || spoke == keep_spoke {
                    continue;
                }
                if let Some(faces) = self.spoke_face_pair(spoke) {
                    next_spoke = Some((spoke, faces));
                    break;
                }
            }
            let Some((spoke, (donor, absorbed))) = next_spoke else {
                break;
            };
            if self.faces_join_pair(donor, absorbed, spoke).is_none() {
                return false;
            }
        }

        let Some(surviving_edge) = self.vert_collapse_edge(base_spoke, v) else {
            return false;
        };
        if self.edges.try_get(surviving_edge).is_none() {
            return true;
        }

        // The final join: both halves sit on the surviving edge, so its entry
        // names the half that finished merging last.
        if let Some((donor, absorbed)) = self.spoke_face_pair(surviving_edge) {
            self.faces_join_pair(donor, absorbed, surviving_edge);
        }
        true
    }

    /// The two distinct faces flanking a radial-2 `spoke`, the edge's radial
    /// **entry** face first — i.e. `(donor, absorbed)` for a join across it,
    /// since the entry is the most recently created face on the spoke (§6.3).
    /// `None` for a wire, boundary, non-manifold, or self-flanked spoke.
    fn spoke_face_pair(&self, spoke: EdgeKey) -> Option<(FaceKey, FaceKey)> {
        let mut radial = self.radial_loops(spoke);
        let entry_loop = radial.next(self)?;
        let partner_loop = radial.next(self)?;
        if radial.next(self).is_some() {
            return None;
        }
        let entry_face = self.loops.get(entry_loop).face;
        let partner_face = self.loops.get(partner_loop).face;
        if entry_face == partner_face {
            return None;
        }
        Some((entry_face, partner_face))
    }
}

#[cfg(test)]
mod tests {
    use glam::Vec3;

    use crate::loopmesh::*;

    /// Build a face from an ordered vertex ring under the perimeter contract of
    /// `DISSOLVE_VERTS.md` §3.1: the **closing** edge `v(n-1)–v0` is created
    /// first, then the opening edges in ascending ring order. That order is what
    /// fixes each vertex's disk anchor, and every §6.7 donor trace is stated
    /// against it.
    fn face_from_ring(mesh: &mut LoopMesh, ring: &[VertexKey]) -> FaceKey {
        let n = ring.len();
        let closing = mesh.create_edge([ring[n - 1], ring[0]], EdgeCreateFlags::NO_DUPLICATE);
        let mut edges: Vec<EdgeKey> = Vec::with_capacity(n);
        for i in 0..n - 1 {
            edges.push(mesh.create_edge([ring[i], ring[i + 1]], EdgeCreateFlags::NO_DUPLICATE));
        }
        edges.push(closing);
        mesh.face_create(ring, &edges, FaceCreateFlags::NONE)
    }

    /// A closed `n`-triangle fan: apex `0`, rim `1 ..= n`, faces built in ring
    /// order `f_i = [apex, rim_i, rim_{i+1}]` (§6.7's `disk_fan(n)`).
    fn build_disk_fan(mesh: &mut LoopMesh, n: usize) -> (VertexKey, Vec<FaceKey>) {
        let apex = mesh.create_vertex(Vec3::ZERO);
        let rim: Vec<VertexKey> = (0..n)
            .map(|i| {
                let theta = std::f32::consts::TAU * (i as f32) / (n as f32);
                mesh.create_vertex(Vec3::new(theta.cos(), theta.sin(), 0.0))
            })
            .collect();
        let faces = (0..n)
            .map(|i| face_from_ring(mesh, &[apex, rim[i], rim[(i + 1) % n]]))
            .collect();
        (apex, faces)
    }

    /// The 2x2 quad grid of §6.7's first worked example: verts `0..9` row-major
    /// on a 3x3 lattice, faces `f0..f3` in row-major order, centre vert `4`.
    fn build_grid_3x3(mesh: &mut LoopMesh) -> (Vec<VertexKey>, Vec<FaceKey>) {
        let verts: Vec<VertexKey> = (0..3)
            .flat_map(|row| (0..3).map(move |col| (row, col)))
            .map(|(row, col)| mesh.create_vertex(Vec3::new(col as f32, row as f32, 0.0)))
            .collect();
        let mut faces = Vec::new();
        for row in 0..2 {
            for col in 0..2 {
                let ring = [
                    verts[row * 3 + col],
                    verts[row * 3 + col + 1],
                    verts[(row + 1) * 3 + col + 1],
                    verts[(row + 1) * 3 + col],
                ];
                faces.push(face_from_ring(mesh, &ring));
            }
        }
        (verts, faces)
    }

    /// A 3x3 quad grid on a 4x4 lattice: verts `0..15` row-major, faces
    /// `f0..f8` row-major. The interior verts 5 and 10 have overlapping fans
    /// (they share the centre quad `f4`), which is the shape a repeated
    /// coarsening walks (`UNSUBDIVIDE.md` §5.1).
    fn build_grid_4x4(mesh: &mut LoopMesh) -> (Vec<VertexKey>, Vec<FaceKey>) {
        let verts: Vec<VertexKey> = (0..4)
            .flat_map(|row| (0..4).map(move |col| (row, col)))
            .map(|(row, col)| mesh.create_vertex(Vec3::new(col as f32, row as f32, 0.0)))
            .collect();
        let mut faces = Vec::new();
        for row in 0..3 {
            for col in 0..3 {
                let ring = [
                    verts[row * 4 + col],
                    verts[row * 4 + col + 1],
                    verts[(row + 1) * 4 + col + 1],
                    verts[(row + 1) * 4 + col],
                ];
                faces.push(face_from_ring(mesh, &ring));
            }
        }
        (verts, faces)
    }

    /// An **open** `n`-triangle fan — apex plus rim `r0 ..= rn`, faces
    /// `f_i = [apex, r_i, r_{i+1}]` — with the faces built in `build_order`,
    /// so a caller can choose which spokes attach at the apex first.
    fn build_open_disk_fan(
        mesh: &mut LoopMesh,
        n: usize,
        build_order: &[usize],
    ) -> (VertexKey, Vec<VertexKey>, Vec<FaceKey>) {
        let apex = mesh.create_vertex(Vec3::ZERO);
        let rim: Vec<VertexKey> = (0..=n)
            .map(|i| {
                let theta = std::f32::consts::PI * (i as f32) / (n as f32);
                mesh.create_vertex(Vec3::new(theta.cos(), theta.sin(), 0.0))
            })
            .collect();
        let mut faces: Vec<Option<FaceKey>> = vec![None; n];
        for &i in build_order {
            faces[i] = Some(face_from_ring(mesh, &[apex, rim[i], rim[i + 1]]));
        }
        (
            apex,
            rim,
            faces.into_iter().map(|f| f.expect("built")).collect(),
        )
    }

    fn tag_faces(mesh: &mut LoopMesh, faces: &[FaceKey]) -> ElementAttributeKey {
        let tag = mesh
            .register_face_attribute(ElementAttributeDesc::numeric(
                AttributeKey::custom("face_tag"),
                AttrType::Vec3,
            ))
            .unwrap();
        for (i, &face) in faces.iter().enumerate() {
            mesh.face_attribs_mut()
                .set_vec3(tag, face.as_usize(), [i as f32, 0.0, 0.0]);
        }
        tag
    }

    /// The one surviving face with `sides` corners, and the fixture index its
    /// per-face tag names.
    fn merged_face_donor_index(mesh: &LoopMesh, tag: ElementAttributeKey, sides: usize) -> usize {
        let mut merged: Option<FaceKey> = None;
        let mut faces = MeshFaces::default();
        while let Some(face) = faces.next(mesh) {
            if mesh.face_vert_keys(face).len() == sides {
                assert!(merged.is_none(), "exactly one {sides}-corner face survives");
                merged = Some(face);
            }
        }
        let merged = merged.expect("merged face exists");
        let tagged = mesh
            .face_attribs()
            .get_vec3(tag, merged.as_usize())
            .expect("merged face carries the tag");
        tagged[0].round() as usize
    }

    /// §6.7 — the helper reserves `base = v.e` and the first spoke in **disk**
    /// order sharing no face with it. On the `grid_3x3` centre that is
    /// `{1,4}` / `{4,7}`, which is the fact the whole donor derivation rests on.
    #[test]
    fn vert_dissolve_helper_reserves_base_and_first_non_neighbour_spoke() {
        let mut mesh = LoopMesh::new();
        let (verts, _faces) = build_grid_3x3(&mut mesh);
        let centre = verts[4];

        let disk: Vec<EdgeKey> = collect_iter::<_, 8>(&mesh, mesh.disk_edges(centre)).into_vec();
        let ends = |e: EdgeKey| {
            let [a, b] = mesh.edges.get(e).vertices;
            let mut pair = [
                verts.iter().position(|&v| v == a).unwrap(),
                verts.iter().position(|&v| v == b).unwrap(),
            ];
            pair.sort();
            pair
        };
        assert_eq!(
            disk.iter().map(|&e| ends(e)).collect::<Vec<_>>(),
            vec![[1, 4], [3, 4], [4, 5], [4, 7]],
            "§3.1 disk order at the grid centre is attachment order"
        );

        let (base, keep) = mesh
            .vert_dissolve_reserved_spokes(centre)
            .expect("interior 4-fan reserves two spokes");
        assert_eq!(ends(base), [1, 4], "base is the disk anchor v.e");
        assert_eq!(
            ends(keep),
            [4, 7],
            "keep is the first spoke in disk order sharing no face with base"
        );
    }

    /// §6.7 worked example 1 — `grid_3x3` centre: halves `{f0,f2}` and
    /// `{f1,f3}`, half B (carrying f3) finishes last, so the octagon takes
    /// fixture face **3**'s per-face customdata. (Valence 4 alone cannot
    /// discriminate §6.7's schedule from §6.3's — the fans below can.)
    #[test]
    fn vert_dissolve_helper_grid_centre_merged_face_cd_donor() {
        let mut mesh = LoopMesh::new();
        let (verts, faces) = build_grid_3x3(&mut mesh);
        let tag = tag_faces(&mut mesh, &faces);

        assert!(mesh.vert_dissolve(verts[4]));

        assert_eq!(mesh.face_count(), 1);
        assert_eq!(merged_face_donor_index(&mesh, tag, 8), 3);
        mesh.validate_topology();
    }

    /// §6.7 worked example 3 — `disk_fan(5)`: half A `{f1,f2}` completes in pass
    /// 2, half B `{f3,f4,f0}` in pass 3, so the merged pentagon carries fixture
    /// face **4**. The old whole-fan chain answered face 2 here.
    #[test]
    fn vert_dissolve_helper_disk_fan_5_merged_face_cd_donor() {
        let mut mesh = LoopMesh::new();
        let (apex, faces) = build_disk_fan(&mut mesh, 5);
        let tag = tag_faces(&mut mesh, &faces);

        assert!(mesh.vert_dissolve(apex));

        assert_eq!(mesh.face_count(), 1);
        assert_eq!(mesh.vertex_count(), 5, "the apex is gone; the rim survives");
        assert_eq!(mesh.edge_count(), 5, "every spoke is gone");
        assert_eq!(merged_face_donor_index(&mesh, tag, 5), 4);
        mesh.validate_topology();
    }

    /// §6.7 worked example 2 — `disk_fan(6)`: half B is assembled out of two
    /// sub-merges (`X` in pass 1, `Y` in pass 3) and closes in pass 4 carrying
    /// `Y`'s block, so the hexagon takes fixture face **4** — neither the first
    /// join's face (f5) nor the base spoke's original entry face (f1).
    #[test]
    fn vert_dissolve_helper_disk_fan_6_merged_face_cd_donor() {
        let mut mesh = LoopMesh::new();
        let (apex, faces) = build_disk_fan(&mut mesh, 6);
        let tag = tag_faces(&mut mesh, &faces);

        assert!(mesh.vert_dissolve(apex));

        assert_eq!(mesh.face_count(), 1);
        assert_eq!(mesh.vertex_count(), 6);
        assert_eq!(mesh.edge_count(), 6);
        assert_eq!(merged_face_donor_index(&mesh, tag, 6), 4);
        mesh.validate_topology();
    }

    /// §6.7's halves model is derived for the **fully-interior** fan, so the
    /// classification must be a property of the whole disk — not of wherever
    /// the keep search happens to stop. A boundary fan built so that its two
    /// boundary spokes attach *last* puts a keep candidate in front of them in
    /// disk order; reserving on it would run the halves schedule over a fan
    /// whose boundary spokes no join can consume.
    ///
    /// Apex disk order here is `s2, s1, s3, s4, s5, s0` (`s_i = {apex, rim_i}`,
    /// attachment order), with `s0`/`s5` the boundary spokes: a keep search that
    /// stops at the first radial-1 spoke it *meets* would reserve `s2`/`s4` and
    /// never see them.
    ///
    /// The fallback walk then dissolves the fan outright. That is loopmesh's
    /// long-standing behaviour on a `k ≥ 3` boundary fan and it is **not** a
    /// parity claim: §6.7 records the reference as merging the fan and then
    /// *failing* there, and marks the case UNCHARACTERISED.
    #[test]
    fn vert_dissolve_reserved_spokes_refuses_boundary_fan_whatever_the_disk_order() {
        let mut mesh = LoopMesh::new();
        let (apex, _rim, _faces) = build_open_disk_fan(&mut mesh, 5, &[1, 2, 3, 4, 0]);

        assert!(
            mesh.vert_dissolve_reserved_spokes(apex).is_none(),
            "a boundary fan never enters the halves schedule"
        );

        assert!(mesh.vert_dissolve(apex));
        assert_eq!(mesh.face_count(), 1);
        assert_eq!(mesh.vertex_count(), 6, "the apex is gone; the rim survives");
        assert_eq!(mesh.edge_count(), 6);
        mesh.validate_topology();
    }

    /// The donor's age channel is the stored radial **entry**, and it has to
    /// survive *across* calls: `unsubdivide` dissolves adjacent verts in
    /// sequence, so the second call's fan contains a face the first call merged.
    /// A merged face is the newest face on every perimeter edge it kept
    /// (§6.3's entry-reseat rule), so it wins every donor contest it enters
    /// next time — an age channel rebuilt per call would score it as pristine
    /// and read a stale build-order entry instead.
    ///
    /// On the 4x4 grid, dissolving vert 5 merges `{f0, f1, f3, f4}`; the spoke
    /// `{9,10}` of vert 10's fan then flanks that merged face and `f7`, and its
    /// *pristine* entry named `f7` (the later-built of the two). After the first
    /// dissolve it must name the merged face.
    #[test]
    fn vert_dissolve_helper_overlapping_fans_donor_reads_the_merged_face() {
        let mut mesh = LoopMesh::new();
        let (verts, faces) = build_grid_4x4(&mut mesh);
        let tag = tag_faces(&mut mesh, &faces);

        assert!(mesh.vert_dissolve(verts[5]));

        let spoke_9_10 = collect_iter::<_, 8>(&mesh, mesh.disk_edges(verts[10]))
            .into_iter()
            .find(|&e| mesh.edges.get(e).contains(verts[9]).is_some())
            .expect("{9,10} survives the first dissolve");
        let entry_face = mesh.loop_face(mesh.edge_radial_entry(spoke_9_10).expect("radial-2"));
        assert_eq!(
            mesh.face_attribs()
                .get_vec3(tag, entry_face.as_usize())
                .expect("tagged")[0]
                .round() as usize,
            4,
            "the merged face (donor f4's block) is the newest face on {{9,10}}"
        );

        assert!(mesh.vert_dissolve(verts[10]));

        assert_eq!(mesh.face_count(), 3, "the merged 12-gon plus f2 and f6");
        assert_eq!(mesh.vertex_count(), 14);
        assert_eq!(mesh.edge_count(), 16);
        assert_eq!(
            merged_face_donor_index(&mesh, tag, 12),
            8,
            "half B ({{f5, f8}}, donor f8) completes last"
        );
        mesh.validate_topology();
    }

    /// §6.7's no-keep branch: at valence 3 every spoke shares a face with the
    /// anchor, so nothing is reserved and the fan collapses through the single
    /// join + reaping collapse. The halves schedule must not claim this fan.
    #[test]
    fn vert_dissolve_helper_three_triangle_fan_takes_no_keep_branch() {
        let mut mesh = LoopMesh::new();
        let (apex, _faces) = build_disk_fan(&mut mesh, 3);
        assert!(
            mesh.vert_dissolve_reserved_spokes(apex).is_none(),
            "a 3-fan has no keep spoke"
        );

        assert!(mesh.vert_dissolve(apex));

        assert_eq!(mesh.face_count(), 1);
        assert_eq!(mesh.vertex_count(), 3, "collapses to a single triangle");
        assert_eq!(mesh.edge_count(), 3);
        mesh.validate_topology();
    }
}
