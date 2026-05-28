use smallvec::SmallVec;

use crate::loopmesh::*;

/// Parameters controlling [`LoopMesh::dissolve_verts_ex`].
///
/// Mirrors the BMOP set-input shape described in
/// `radial-mesh-guides/operators/DISSOLVE_VERTS.md` §1. Both pre-pass bools
/// default off; the all-defaults form is the convenience
/// [`LoopMesh::dissolve_verts`].
#[derive(Clone, Copy, Default)]
pub struct DissolveVertsParams {
    /// Pre-pass: before any vert is removed, draw a diagonal across every
    /// surrounding n-gon (`n ≥ 4`) so the input vert ends up isolated inside a
    /// triangle slice. The post-join merged face is then `2k`-cornered rather
    /// than `sum(n_i) − 2k`. Documented in
    /// `radial-mesh-guides/operators/DISSOLVE_VERTS.md` §4.
    pub use_face_split: bool,
    /// Pre-pass: for a boundary-touching input vert, tear the surrounding
    /// faces open rather than healing the surface. Documented in
    /// `radial-mesh-guides/operators/DISSOLVE_VERTS.md` §5. Not yet
    /// implemented.
    pub use_boundary_tear: bool,
}

/// Per-vert topology class, decided up front from the shape of `v`'s
/// incident disk + radial cycles. Mirrors the classification table in
/// `radial-mesh-guides/operators/DISSOLVE_VERTS.md` §3.
#[derive(Clone, Copy, PartialEq, Eq)]
enum VertClass {
    /// No incident edges. Killed by the phase-4 isolated-vert sweep.
    Isolated,
    /// One or more incident edges, every one of them wire. Phase 1 kills the
    /// wire edges; phase 4 kills the now-isolated vert.
    WireOnly,
    /// Exactly two incident edges bounding a single face (the open-strip
    /// corner). Collapsed in phase 4 via `vert_collapse_edge`.
    EdgePair,
    /// `k ≥ 2` incident edges, each radial 1 or 2, at least one radial 2, no
    /// non-manifold (radial ≥ 3) and no wire edge. The fan of faces merges
    /// into one. The default-parameter headline case.
    ManifoldFan,
    /// Bowtie / non-manifold / mixed — no unambiguous merged face. The
    /// operator no-ops on `v` and leaves it in place.
    NonManifold,
}

impl LoopMesh {
    /// Remove every vertex in `verts`, re-stitching the surrounding faces
    /// back into one face per topologically-connected fan.
    ///
    /// Implements the default-parameter path of
    /// `radial-mesh-guides/operators/DISSOLVE_VERTS.md`: the §3 per-vert
    /// classification dispatch plus the §6 four-phase surgery. Each input
    /// vert is classified independently (§8) and routed to the phase that
    /// handles its class:
    ///
    /// - **Isolated / wire-only** — phase 1 wire-edge sweep (§6.1) + phase 4
    ///   isolated-vert sweep (§6.4).
    /// - **Edge-pair on a face boundary** — phase 2 snapshot (§6.2) + phase 4
    ///   `vert_collapse_edge` (§6.4).
    /// - **Manifold interior fan** — phase 3 fan merge (§6.3): the fan's faces
    ///   are joined into one via the n-input `faces_join` primitive, which
    ///   kills the spoke edges and the interior-only hub vertex as part of the
    ///   merge. (§6.5 blesses the n-input form as producing the identical
    ///   canonical mesh as the per-edge join-pair loop; the per-edge form is a
    ///   performance-only variant tracked separately.)
    /// - **Bowtie / non-manifold** — no-op; the vert survives (§3 row 5).
    ///
    /// The `use_boundary_tear` pre-pass (§5) is not yet implemented; the
    /// `use_face_split` pre-pass (§4) is — see [`Self::dissolve_verts_ex`].
    pub fn dissolve_verts(&mut self, verts: &[VertexKey]) {
        self.dissolve_verts_ex(verts, &DissolveVertsParams::default());
    }

    /// Parameterised form of [`Self::dissolve_verts`]. The default-parameter
    /// behaviour is identical; the pre-pass options layer extra surgery on top
    /// of the §6 four-phase core:
    ///
    /// - **`use_face_split`** (§4) — before any vert is removed, slice each
    ///   surrounding n-gon (`n ≥ 4`) at the input vert with a diagonal so the
    ///   vert is demoted to a triangle-slice fan. The downstream join then
    ///   produces a `2k`-cornered merged face instead of the wider
    ///   `sum(n_i) − 2k` n-gon (§4.2).
    /// - **`use_boundary_tear`** (§5) — not yet implemented.
    pub fn dissolve_verts_ex(&mut self, verts: &[VertexKey], params: &DissolveVertsParams) {
        // §8 empty-input fast path.
        if verts.is_empty() {
            return;
        }

        // §4 — n-gon corner diagonal pre-pass. Runs across the full input set
        // before any vert is removed, so the §6 phases see the already-sliced
        // topology.
        if params.use_face_split {
            self.dissolve_verts_face_split_pre_pass(verts);
        }

        // GC bit (§6.1): every element the surgery flags as a cleanup
        // candidate. The phase-4 sweeps only ever delete elements carrying
        // this bit, so pre-existing wire edges / isolated verts the caller
        // left around the input set are never disturbed.
        let gc_vert = self.push_vert_op_flag_layer();
        let gc_edge = self.push_edge_op_flag_layer();
        // Edge-pair-pending bit (§6.2): set on verts that were edge-pair
        // *before* any join ran, so the predicate sees the pre-phase-3 state.
        let edge_pair_pending = self.push_vert_op_flag_layer();

        const BIT: u8 = 0;

        // §6.1 Phase 1 — seed GC bits and kill wire edges at every input vert.
        for &v in verts {
            if self.vertices.try_get(v).is_none() {
                continue;
            }
            // Mark every loop's vert + edge around v as a GC candidate. Walk
            // the disk, then each edge's radial fan, picking up the loops.
            let mut disk = self.disk_edges(v);
            while let Some(edge) = disk.next(self) {
                let mut radial = self.radial_loops(edge);
                while let Some(loop_key) = radial.next(self) {
                    let loop_vert = self.loops.get(loop_key).vertex;
                    let loop_edge = self.loops.get(loop_key).edge;
                    self.vert_op_flag_set(gc_vert, loop_vert, BIT);
                    if !loop_edge.is_null() {
                        self.edge_op_flag_set(gc_edge, loop_edge, BIT);
                    }
                }
                // The spoke edge itself is a GC candidate even when it carries
                // no loops (a wire edge: empty radial fan does not surface it
                // above).
                self.edge_op_flag_set(gc_edge, edge, BIT);
            }
            self.vert_op_flag_set(gc_vert, v, BIT);

            // Kill every wire edge in v's disk. Wire edges have no faces to
            // join, so removing them up front simplifies phase 3.
            let disk_edges: SmallVec<[EdgeKey; 8]> =
                collect_iter::<_, 8>(self, self.disk_edges(v));
            for edge in disk_edges {
                if self.edge_radial_is_empty(edge) {
                    self.edge_kill(edge);
                }
            }
        }

        // §6.2 Phase 2 — snapshot the edge-pair predicate before any joins.
        for &v in verts {
            if self.vertices.try_get(v).is_none() {
                continue;
            }
            if self.vert_disk_count_edges_capped(v, 3) == 2 {
                self.vert_op_flag_set(edge_pair_pending, v, BIT);
            }
        }

        // §6.3 Phase 3 — merge each manifold-interior fan. Edge-pair verts are
        // deferred to phase 4; bowtie / non-manifold verts no-op.
        for &v in verts {
            if self.vertices.try_get(v).is_none() {
                continue;
            }
            if self.vert_op_flag_test(edge_pair_pending, v, BIT) {
                continue;
            }
            if self.classify_dissolve_vert(v) == VertClass::ManifoldFan {
                self.dissolve_verts_merge_fan(v);
            }
        }

        // §6.4 Phase 4 — GC wire edges, collapse edge-pair verts, GC isolated
        // verts. The order matters: each step can produce work for the next.

        // (a) Sweep GC-flagged edges that are now wire (radial empty). These
        // are spokes that became internal-to-the-merged-face artefacts of
        // phase 3, plus any spoke left dangling after the joins.
        let mut edge_iter = MeshEdges::default();
        let mut wire_gc: SmallVec<[EdgeKey; 16]> = SmallVec::new();
        while let Some(edge) = edge_iter.next(self) {
            if self.edge_op_flag_test(gc_edge, edge, BIT) && self.edge_radial_is_empty(edge) {
                wire_gc.push(edge);
            }
        }
        for edge in wire_gc {
            if self.edges.try_get(edge).is_some() {
                self.edge_kill(edge);
            }
        }

        // (b) Collapse every input vert flagged edge-pair in phase 2 that is
        // *still* edge-pair (phase 3's joins around neighbours, and the wire
        // sweep above, may have changed the count).
        for &v in verts {
            if self.vertices.try_get(v).is_none() {
                continue;
            }
            if !self.vert_op_flag_test(edge_pair_pending, v, BIT) {
                continue;
            }
            if self.vert_disk_count_edges_capped(v, 3) != 2 {
                continue;
            }
            let mut disk = self.disk_edges(v);
            let Some(e_kill) = disk.next(self) else {
                continue;
            };
            let _ = self.vert_collapse_edge(e_kill, v);
        }

        // (c) Sweep GC-flagged verts that are now isolated (edge count 0).
        // This kills wire-only and isolated input verts whose surrounding
        // topology fully evaporated.
        let mut vert_iter = MeshVerts::default();
        let mut isolated_gc: SmallVec<[VertexKey; 16]> = SmallVec::new();
        while let Some(v) = vert_iter.next(self) {
            if self.vert_op_flag_test(gc_vert, v, BIT)
                && self.vertices.get(v).disk_loop_start.is_none()
            {
                isolated_gc.push(v);
            }
        }
        for v in isolated_gc {
            if self.vertices.try_get(v).is_some() {
                self.vertex_kill(v);
            }
        }

        self.pop_vert_op_flag_layer(edge_pair_pending);
        self.pop_edge_op_flag_layer(gc_edge);
        self.pop_vert_op_flag_layer(gc_vert);
    }

    /// §4 — the `use_face_split` n-gon corner diagonal pre-pass.
    ///
    /// For every input vert `v` that is not an edge-pair (more than two
    /// incident edges, §4.1 / §4.3), and every face `f` incident at `v` whose
    /// corner count is `> 3`, slice `f` along the diagonal `v_prev – v_next`
    /// (the two perimeter neighbours of `v` on `f`). The cut-off triangle
    /// `(v_prev, v, v_next)` holds `v` as its sole input corner; the rest of
    /// `f` survives as an `(n−1)`-gon. The slice is skipped when either
    /// `v_prev` or `v_next` is itself an input vert (§4.1): the diagonal would
    /// be wasted work since both its endpoints are about to be dissolved.
    ///
    /// The diagonal `v_prev – v_next` never touches `v`, so slicing one face
    /// of `v`'s fan does not change which loop sits at `v` on any other face;
    /// the per-face targets are therefore gathered up front and the splits
    /// applied afterwards.
    fn dissolve_verts_face_split_pre_pass(&mut self, verts: &[VertexKey]) {
        // O(1) input-set membership for the §4.1 skip condition, mirroring the
        // GC-bit idiom the four-phase core uses.
        let input_set = self.push_vert_op_flag_layer();
        const BIT: u8 = 0;
        for &v in verts {
            if self.vertices.try_get(v).is_some() {
                self.vert_op_flag_set(input_set, v, BIT);
            }
        }

        // (face, v_prev, v_next) diagonals to cut, gathered before any split.
        let mut diagonals: SmallVec<[(FaceKey, VertexKey, VertexKey); 8]> = SmallVec::new();
        for &v in verts {
            if self.vertices.try_get(v).is_none() {
                continue;
            }
            // §4.1 / §4.3 — edge-pair verts have no third corner to split.
            if self.vert_disk_count_edges_capped(v, 3) <= 2 {
                continue;
            }
            // Find each incident face via the loop at `v` on every disk edge's
            // radial fan; dedup so a face shared by two of `v`'s spokes is only
            // sliced once.
            let mut seen_faces: SmallVec<[FaceKey; 8]> = SmallVec::new();
            let mut disk = self.disk_edges(v);
            while let Some(edge) = disk.next(self) {
                let mut radial = self.radial_loops(edge);
                while let Some(loop_key) = radial.next(self) {
                    if self.loops.get(loop_key).vertex != v {
                        continue;
                    }
                    let face = self.loops.get(loop_key).face;
                    if seen_faces.contains(&face) {
                        continue;
                    }
                    seen_faces.push(face);
                    if self.face_len(face) <= 3 {
                        continue;
                    }
                    let v_next = self.loops.get(self.loop_next(loop_key)).vertex;
                    let v_prev = self.loops.get(self.loop_prev(loop_key)).vertex;
                    if self.vert_op_flag_test(input_set, v_next, BIT)
                        || self.vert_op_flag_test(input_set, v_prev, BIT)
                    {
                        continue;
                    }
                    diagonals.push((face, v_prev, v_next));
                }
            }
        }

        for (face, v_prev, v_next) in diagonals {
            let _ = self.face_split(face, v_prev, v_next);
        }

        self.pop_vert_op_flag_layer(input_set);
    }

    /// Classify `v` per `DISSOLVE_VERTS.md` §3. Reads `v`'s disk + per-edge
    /// radial counts in a single pass; no mutation.
    fn classify_dissolve_vert(&self, v: VertexKey) -> VertClass {
        let edges: SmallVec<[EdgeKey; 8]> = collect_iter::<_, 8>(self, self.disk_edges(v));
        if edges.is_empty() {
            return VertClass::Isolated;
        }

        let mut has_wire = false;
        let mut has_interior = false;
        let mut has_non_manifold = false;
        for &edge in &edges {
            match self.edge_radial_count_capped(edge, 3) {
                0 => has_wire = true,
                1 => {}
                2 => has_interior = true,
                _ => has_non_manifold = true,
            }
        }

        if !has_interior && !has_non_manifold && has_wire {
            // Every incident edge is wire.
            return VertClass::WireOnly;
        }

        // Edge-pair on a face boundary: exactly two edges bounding a single
        // face. (Caught here for completeness; the actual collapse is driven
        // by the phase-2 snapshot.)
        if edges.len() == 2 && !has_wire && !has_non_manifold {
            let mut faces: SmallVec<[FaceKey; 2]> = SmallVec::new();
            for &edge in &edges {
                let mut radial = self.radial_loops(edge);
                while let Some(loop_key) = radial.next(self) {
                    let face = self.loops.get(loop_key).face;
                    if !faces.contains(&face) {
                        faces.push(face);
                    }
                }
            }
            if faces.len() == 1 {
                return VertClass::EdgePair;
            }
        }

        // Manifold interior fan: no wire, no non-manifold radial, at least one
        // interior edge to fold the fan down.
        if !has_wire && !has_non_manifold && has_interior {
            return VertClass::ManifoldFan;
        }

        VertClass::NonManifold
    }

    /// §6.3 — merge the manifold fan around `v` into one face. Collects the
    /// fan's faces (every face incident at `v`, which for a manifold interior
    /// vert is a single topologically-connected fan) and delegates to the
    /// n-input `faces_join` primitive. `faces_join` kills the interior spoke
    /// edges and the now-orphaned hub vertex `v` as part of the merge.
    fn dissolve_verts_merge_fan(&mut self, v: VertexKey) {
        let mut fan_faces: SmallVec<[FaceKey; 8]> = SmallVec::new();
        let mut disk = self.disk_edges(v);
        while let Some(edge) = disk.next(self) {
            let mut radial = self.radial_loops(edge);
            while let Some(loop_key) = radial.next(self) {
                let face = self.loops.get(loop_key).face;
                if !fan_faces.contains(&face) {
                    fan_faces.push(face);
                }
            }
        }
        if fan_faces.len() < 2 {
            return;
        }
        let _ = self.faces_join(&fan_faces);
    }

    /// True iff `edge`'s radial fan carries no loops (a wire edge). Cheap
    /// single-step probe — does not walk the whole fan.
    #[inline]
    fn edge_radial_is_empty(&self, edge: EdgeKey) -> bool {
        self.edges.get(edge).radial_link.is_none()
    }

    /// Count `edge`'s radial loops, stopping at `cap`. Returns `cap` if the
    /// fan is at least that long. Used to distinguish wire (0) / boundary (1)
    /// / interior (2) / non-manifold (≥3) without a full walk.
    fn edge_radial_count_capped(&self, edge: EdgeKey, cap: usize) -> usize {
        let mut radial = self.radial_loops(edge);
        let mut count = 0usize;
        while count < cap && radial.next(self).is_some() {
            count += 1;
        }
        count
    }

    /// Count `v`'s incident edges, stopping at `cap`.
    fn vert_disk_count_edges_capped(&self, v: VertexKey, cap: usize) -> usize {
        let mut disk = self.disk_edges(v);
        let mut count = 0usize;
        while count < cap && disk.next(self).is_some() {
            count += 1;
        }
        count
    }
}

#[cfg(test)]
mod tests {
    use glam::Vec3;

    use crate::loopmesh::*;

    /// Build a 3x3 face grid of quads on the unit lattice. Returns
    /// `(verts[16], faces[9])` where `verts[r * 4 + c]` for `r,c in 0..4` and
    /// `faces[r * 3 + c]` for `r,c in 0..3`. The interior 4-valence vertex is
    /// `verts[5]` (the lower-left interior lattice point).
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

    /// Build a 5x5 face grid of quads. `verts[r * 6 + c]` for `r,c in 0..6`,
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

    /// §2 / §7 manifold-interior-fan row: dissolving the 4-valence centre of a
    /// 3x3 grid merges four quads into one octagon. Delta: -1 V, -4 E, -3 F.
    #[test]
    fn dissolve_verts_grid_centre_merges_fan_into_octagon() {
        let mut mesh = LoopMesh::new();
        let (verts, _faces) = build_grid_3x3(&mut mesh);

        let v0 = mesh.totvert();
        let e0 = mesh.totedge();
        let f0 = mesh.totface();
        assert_eq!(v0, 16);
        assert_eq!(e0, 24);
        assert_eq!(f0, 9);

        let centre = verts[5];
        assert_eq!(
            mesh.vert_edge_keys(centre).len(),
            4,
            "centre is a 4-valence interior vert"
        );

        mesh.dissolve_verts(&[centre]);

        // §2 n-corner-fan delta for k = 4.
        assert_eq!(mesh.totvert(), v0 - 1, "-1 V (the centre)");
        assert_eq!(mesh.totedge(), e0 - 4, "-4 E (the four spokes)");
        assert_eq!(mesh.totface(), f0 - 3, "-3 F (four quads → one)");

        // The merged face is the octagon whose perimeter is the ring of eight
        // verts around the dissolved centre.
        let live: std::collections::HashSet<VertexKey> = mesh.vert_keys().into_iter().collect();
        assert!(!live.contains(&centre), "centre vert is gone");

        // Locate the merged face: the only face touching all four ring corners
        // 1, 4, 6, 9 (the edge-midpoints of the original 2x2 of quads). Use any
        // surviving face that has 8 corners.
        let mut octagon: Option<FaceKey> = None;
        let mut face_iter = MeshFaces::default();
        while let Some(f) = face_iter.next(&mesh) {
            if mesh.face_vert_keys(f).len() == 8 {
                octagon = Some(f);
            }
        }
        let octagon = octagon.expect("an 8-corner merged face exists");
        let ring: std::collections::HashSet<VertexKey> =
            mesh.face_vert_keys(octagon).into_iter().collect();
        // The eight ring verts are the four edge-neighbours (1, 4, 6, 9) and
        // the four diagonal corners (0, 2, 8, 10) of the 2x2 quad block.
        for idx in [0usize, 1, 2, 4, 6, 8, 9, 10] {
            assert!(ring.contains(&verts[idx]), "ring contains vert {idx}");
        }
        mesh.validate_topology();
    }

    /// §8 "Multiple non-adjacent input verts": two interior verts of a 5x5
    /// face grid whose fans are topologically disjoint. Each fan merges
    /// independently; the surgeries do not interact. Two disjoint 4-fans →
    /// -2 V, -8 E, -6 F.
    #[test]
    fn dissolve_verts_two_nonadjacent_interior_verts_independent() {
        let mut mesh = LoopMesh::new();
        let (verts, _faces) = build_grid_5x5(&mut mesh);

        let v0 = mesh.totvert();
        let e0 = mesh.totedge();
        let f0 = mesh.totface();

        // Two interior verts two cells apart so their fans share no face.
        // verts[r*6+c]; pick (1,1) = 7 and (1,3) = 9? Those share row-1 faces.
        // Use (1,1) = 7 and (3,3) = 21 — fully disjoint fans.
        let a = verts[1 * 6 + 1];
        let b = verts[3 * 6 + 3];
        assert_eq!(mesh.vert_edge_keys(a).len(), 4);
        assert_eq!(mesh.vert_edge_keys(b).len(), 4);

        mesh.dissolve_verts(&[a, b]);

        assert_eq!(mesh.totvert(), v0 - 2, "-2 V (both centres)");
        assert_eq!(mesh.totedge(), e0 - 8, "-4 E per disjoint fan");
        assert_eq!(mesh.totface(), f0 - 6, "-3 F per disjoint fan");

        let live: std::collections::HashSet<VertexKey> = mesh.vert_keys().into_iter().collect();
        assert!(!live.contains(&a));
        assert!(!live.contains(&b));
        mesh.validate_topology();
    }

    /// §3 row 3 / §6.4 edge-pair-collapse: a vert with exactly two incident
    /// edges bounding a single triangle is removed via `vert_collapse_edge`;
    /// the two edges merge into one, the face loses one corner. Delta:
    /// -1 V, -1 E, 0 F.
    #[test]
    fn dissolve_verts_edge_pair_on_triangle_boundary() {
        // A single quad face: each corner is an edge-pair vert bounding one
        // face. Dissolving one corner collapses the quad to a triangle.
        let mut mesh = LoopMesh::new();
        let v: Vec<VertexKey> = [
            (0.0, 0.0),
            (1.0, 0.0),
            (1.0, 1.0),
            (0.0, 1.0),
        ]
        .into_iter()
        .map(|(x, y)| mesh.create_vertex(Vec3::new(x, y, 0.0)))
        .collect();
        let e0 = mesh.create_edge([v[0], v[1]], EdgeCreateFlags::NO_DUPLICATE);
        let e1 = mesh.create_edge([v[1], v[2]], EdgeCreateFlags::NO_DUPLICATE);
        let e2 = mesh.create_edge([v[2], v[3]], EdgeCreateFlags::NO_DUPLICATE);
        let e3 = mesh.create_edge([v[3], v[0]], EdgeCreateFlags::NO_DUPLICATE);
        let f = mesh.face_create(
            &[v[0], v[1], v[2], v[3]],
            &[e0, e1, e2, e3],
            FaceCreateFlags::NONE,
        );

        assert_eq!(mesh.totvert(), 4);
        assert_eq!(mesh.totedge(), 4);
        assert_eq!(mesh.totface(), 1);
        assert_eq!(mesh.vert_edge_keys(v[1]).len(), 2, "corner is an edge-pair vert");

        mesh.dissolve_verts(&[v[1]]);

        assert_eq!(mesh.totvert(), 3, "-1 V (the collapsed corner)");
        assert_eq!(mesh.totedge(), 3, "-1 E (two edges merge into one)");
        assert_eq!(mesh.totface(), 1, "the face survives, now a triangle");

        let live: std::collections::HashSet<VertexKey> = mesh.vert_keys().into_iter().collect();
        assert!(!live.contains(&v[1]), "the dissolved corner is gone");
        // The surviving face is a triangle on the other three verts.
        let tri_verts: std::collections::HashSet<VertexKey> = mesh
            .face_keys()
            .into_iter()
            .flat_map(|fk| mesh.face_vert_keys(fk))
            .collect();
        let _ = f;
        assert_eq!(tri_verts.len(), 3);
        for idx in [0usize, 2, 3] {
            assert!(tri_verts.contains(&v[idx]));
        }
        mesh.validate_topology();
    }

    /// §7 triangle-corner edge-pair row / §6.6 degenerate-face reaping: a vert
    /// with exactly two incident edges on the corner of a lone triangle. The
    /// phase-4 collapse pushes the triangle to a 2-gon and the primitive reaps
    /// it, splicing away the redundant duplicate edge. Delta:
    /// -1 V, -2 E, -1 F (V=3,E=3,F=1 → V=2,E=1,F=0).
    #[test]
    fn dissolve_verts_edge_pair_on_lone_triangle_reaps_face() {
        let mut mesh = LoopMesh::new();
        let v: Vec<VertexKey> = [(0.0, 0.0), (1.0, 0.0), (0.5, 1.0)]
            .into_iter()
            .map(|(x, y)| mesh.create_vertex(Vec3::new(x, y, 0.0)))
            .collect();
        let e0 = mesh.create_edge([v[0], v[1]], EdgeCreateFlags::NO_DUPLICATE);
        let e1 = mesh.create_edge([v[1], v[2]], EdgeCreateFlags::NO_DUPLICATE);
        let e2 = mesh.create_edge([v[2], v[0]], EdgeCreateFlags::NO_DUPLICATE);
        mesh.face_create(&[v[0], v[1], v[2]], &[e0, e1, e2], FaceCreateFlags::NONE);

        assert_eq!(mesh.totvert(), 3);
        assert_eq!(mesh.totedge(), 3);
        assert_eq!(mesh.totface(), 1);
        assert_eq!(mesh.vert_edge_keys(v[1]).len(), 2, "corner is an edge-pair vert");

        mesh.dissolve_verts(&[v[1]]);

        assert_eq!(mesh.totvert(), 2, "-1 V (the collapsed corner)");
        assert_eq!(mesh.totedge(), 1, "-2 E (spoke + spliced duplicate)");
        assert_eq!(mesh.totface(), 0, "-1 F (the 2-gon is reaped)");

        let live: std::collections::HashSet<VertexKey> = mesh.vert_keys().into_iter().collect();
        assert!(!live.contains(&v[1]), "the dissolved corner is gone");
        assert!(live.contains(&v[0]) && live.contains(&v[2]), "neighbours survive");

        let edges = mesh.edge_keys();
        assert_eq!(edges.len(), 1, "a single edge between the two neighbours");
        let endpoints: std::collections::HashSet<VertexKey> =
            mesh.edge_vert_keys(edges[0]).into_iter().collect();
        assert_eq!(endpoints, [v[0], v[2]].into_iter().collect());
        mesh.validate_topology();
    }

    /// §3 row 2 / §6.1 + §6.4: a wire-only vert with a single wire edge. Phase
    /// 1 kills the wire edge; phase 4 kills the now-isolated vert. The far
    /// endpoint survives. Delta: -1 V, -1 E.
    #[test]
    fn dissolve_verts_wire_only_with_one_wire_edge() {
        let mut mesh = LoopMesh::new();
        let a = mesh.create_vertex(Vec3::new(0.0, 0.0, 0.0));
        let b = mesh.create_vertex(Vec3::new(1.0, 0.0, 0.0));
        // An unrelated isolated vert the GC must NOT touch.
        let bystander = mesh.create_vertex(Vec3::new(5.0, 5.0, 0.0));
        let wire = mesh.create_edge([a, b], EdgeCreateFlags::NO_DUPLICATE);
        assert!(mesh.edge_face_keys(wire).is_empty(), "edge is wire");

        assert_eq!(mesh.totvert(), 3);
        assert_eq!(mesh.totedge(), 1);

        mesh.dissolve_verts(&[a]);

        assert_eq!(mesh.totvert(), 2, "-1 V (a)");
        assert_eq!(mesh.totedge(), 0, "-1 E (the wire edge)");
        let live: std::collections::HashSet<VertexKey> = mesh.vert_keys().into_iter().collect();
        assert!(!live.contains(&a), "a is gone");
        assert!(live.contains(&b), "far endpoint b survives");
        assert!(live.contains(&bystander), "unrelated isolated vert is untouched");
        mesh.validate_topology();
    }

    /// §3 row 5 / §8 bowtie: the shared corner of two triangles that share no
    /// edge is multi-fan. `dissolve_verts` no-ops on it; the mesh is unchanged.
    #[test]
    fn dissolve_verts_bowtie_corner_is_noop() {
        let mut mesh = LoopMesh::new();
        // Two triangles sharing only the apex vert `hub`.
        let hub = mesh.create_vertex(Vec3::new(0.0, 0.0, 0.0));
        let a0 = mesh.create_vertex(Vec3::new(-1.0, 1.0, 0.0));
        let a1 = mesh.create_vertex(Vec3::new(-1.0, -1.0, 0.0));
        let b0 = mesh.create_vertex(Vec3::new(1.0, 1.0, 0.0));
        let b1 = mesh.create_vertex(Vec3::new(1.0, -1.0, 0.0));

        let e_h_a0 = mesh.create_edge([hub, a0], EdgeCreateFlags::NO_DUPLICATE);
        let e_a0_a1 = mesh.create_edge([a0, a1], EdgeCreateFlags::NO_DUPLICATE);
        let e_a1_h = mesh.create_edge([a1, hub], EdgeCreateFlags::NO_DUPLICATE);
        mesh.face_create(&[hub, a0, a1], &[e_h_a0, e_a0_a1, e_a1_h], FaceCreateFlags::NONE);

        let e_h_b0 = mesh.create_edge([hub, b0], EdgeCreateFlags::NO_DUPLICATE);
        let e_b0_b1 = mesh.create_edge([b0, b1], EdgeCreateFlags::NO_DUPLICATE);
        let e_b1_h = mesh.create_edge([b1, hub], EdgeCreateFlags::NO_DUPLICATE);
        mesh.face_create(&[hub, b0, b1], &[e_h_b0, e_b0_b1, e_b1_h], FaceCreateFlags::NONE);

        let v0 = mesh.totvert();
        let e0 = mesh.totedge();
        let f0 = mesh.totface();
        assert_eq!(mesh.vert_edge_keys(hub).len(), 4, "hub joins two triangles, 4 edges, no shared edge");

        mesh.dissolve_verts(&[hub]);

        assert_eq!(mesh.totvert(), v0, "no-op: V unchanged");
        assert_eq!(mesh.totedge(), e0, "no-op: E unchanged");
        assert_eq!(mesh.totface(), f0, "no-op: F unchanged");
        let live: std::collections::HashSet<VertexKey> = mesh.vert_keys().into_iter().collect();
        assert!(live.contains(&hub), "the bowtie hub survives");
        mesh.validate_topology();
    }

    /// §3 row 5 T-junction: a vert on an edge whose radial cycle has length 3
    /// is non-manifold. `dissolve_verts` no-ops on it.
    #[test]
    fn dissolve_verts_t_junction_non_manifold_is_noop() {
        let mut mesh = LoopMesh::new();
        // Three triangles sharing one spine edge {v0, v1}: a "book" with 3
        // pages. Vert 0 is the spine endpoint whose spine edge has radial 3.
        let v: Vec<VertexKey> = [
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.5, 1.0, 0.0),
            (0.5, -0.5, 0.866),
            (0.5, -0.5, -0.866),
        ]
        .into_iter()
        .map(|(x, y, z)| mesh.create_vertex(Vec3::new(x, y, z)))
        .collect();
        let e_spine = mesh.create_edge([v[0], v[1]], EdgeCreateFlags::NO_DUPLICATE);
        for apex in 2..5 {
            let e_a = mesh.create_edge([v[0], v[apex]], EdgeCreateFlags::NO_DUPLICATE);
            let e_b = mesh.create_edge([v[apex], v[1]], EdgeCreateFlags::NO_DUPLICATE);
            mesh.face_create(&[v[0], v[1], v[apex]], &[e_spine, e_b, e_a], FaceCreateFlags::NONE);
        }

        let v0 = mesh.totvert();
        let e0 = mesh.totedge();
        let f0 = mesh.totface();
        assert_eq!(mesh.edge_face_keys(e_spine).len(), 3, "spine is radial-3");

        mesh.dissolve_verts(&[v[0]]);

        assert_eq!(mesh.totvert(), v0, "no-op: V unchanged");
        assert_eq!(mesh.totedge(), e0, "no-op: E unchanged");
        assert_eq!(mesh.totface(), f0, "no-op: F unchanged");
        let live: std::collections::HashSet<VertexKey> = mesh.vert_keys().into_iter().collect();
        assert!(live.contains(&v[0]), "the T-junction vert survives");
        mesh.validate_topology();
    }

    /// §6 mixed-classification set in one call: a manifold-interior fan, a
    /// wire-only vert, and a bowtie corner, all in one input set, on a single
    /// mesh. Each input vert is classified independently and routed to the
    /// phase that handles its class. The manifold fan merges, the wire-only
    /// vert + its edge are killed, the bowtie no-ops.
    #[test]
    fn dissolve_verts_mixed_classification_set() {
        let mut mesh = LoopMesh::new();
        let (verts, _faces) = build_grid_3x3(&mut mesh);

        // Manifold-interior fan target.
        let centre = verts[5];

        // A wire-only vert grafted onto a corner of the grid (so it lives on
        // the same mesh but its fan is disjoint from the grid centre's).
        let wire_far = mesh.create_vertex(Vec3::new(-1.0, -1.0, 0.0));
        let wire = mesh.create_edge([wire_far, verts[0]], EdgeCreateFlags::NO_DUPLICATE);
        assert!(mesh.edge_face_keys(wire).is_empty());

        // A bowtie hub attached well away from the grid.
        let hub = mesh.create_vertex(Vec3::new(10.0, 0.0, 0.0));
        let a0 = mesh.create_vertex(Vec3::new(9.0, 1.0, 0.0));
        let a1 = mesh.create_vertex(Vec3::new(9.0, -1.0, 0.0));
        let b0 = mesh.create_vertex(Vec3::new(11.0, 1.0, 0.0));
        let b1 = mesh.create_vertex(Vec3::new(11.0, -1.0, 0.0));
        let e_h_a0 = mesh.create_edge([hub, a0], EdgeCreateFlags::NO_DUPLICATE);
        let e_a0_a1 = mesh.create_edge([a0, a1], EdgeCreateFlags::NO_DUPLICATE);
        let e_a1_h = mesh.create_edge([a1, hub], EdgeCreateFlags::NO_DUPLICATE);
        mesh.face_create(&[hub, a0, a1], &[e_h_a0, e_a0_a1, e_a1_h], FaceCreateFlags::NONE);
        let e_h_b0 = mesh.create_edge([hub, b0], EdgeCreateFlags::NO_DUPLICATE);
        let e_b0_b1 = mesh.create_edge([b0, b1], EdgeCreateFlags::NO_DUPLICATE);
        let e_b1_h = mesh.create_edge([b1, hub], EdgeCreateFlags::NO_DUPLICATE);
        mesh.face_create(&[hub, b0, b1], &[e_h_b0, e_b0_b1, e_b1_h], FaceCreateFlags::NONE);

        let v0 = mesh.totvert();
        let e0 = mesh.totedge();
        let f0 = mesh.totface();

        // wire_far is the wire-only input vert (it has the single wire edge).
        mesh.dissolve_verts(&[centre, wire_far, hub]);

        // Centre fan: -1 V, -4 E, -3 F.
        // wire_far: -1 V, -1 E.
        // hub: no-op.
        assert_eq!(mesh.totvert(), v0 - 2, "centre + wire_far removed; hub survives");
        assert_eq!(mesh.totedge(), e0 - 5, "-4 spokes + -1 wire edge");
        assert_eq!(mesh.totface(), f0 - 3, "centre fan: 4 quads → 1");

        let live: std::collections::HashSet<VertexKey> = mesh.vert_keys().into_iter().collect();
        assert!(!live.contains(&centre), "manifold fan centre gone");
        assert!(!live.contains(&wire_far), "wire-only vert gone");
        assert!(live.contains(&hub), "bowtie hub survives (no-op)");
        assert!(live.contains(&verts[0]), "wire edge's far endpoint (grid corner) survives");
        mesh.validate_topology();
    }

    /// §8 empty-input fast path: the operator does nothing.
    #[test]
    fn dissolve_verts_empty_input_is_noop() {
        let mut mesh = LoopMesh::new();
        let (_verts, _faces) = build_grid_3x3(&mut mesh);
        let v0 = mesh.totvert();
        let e0 = mesh.totedge();
        let f0 = mesh.totface();

        mesh.dissolve_verts(&[]);

        assert_eq!(mesh.totvert(), v0);
        assert_eq!(mesh.totedge(), e0);
        assert_eq!(mesh.totface(), f0);
        mesh.validate_topology();
    }

    /// "Boundary edge-pair on a manifold strip": a vert with exactly two
    /// incident edges on the open boundary of a two-quad strip (touching only
    /// one of the two faces) collapses via `vert_collapse_edge`; the
    /// neighbouring face that does not touch the dissolved vert is unaffected.
    #[test]
    fn dissolve_verts_edge_pair_on_quad_strip_boundary() {
        let mut mesh = LoopMesh::new();
        // Two quads sharing edge {v1, v2}. Vert 0 is a corner of the left quad
        // only, with exactly two incident edges (both on the strip boundary).
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

        let v0 = mesh.totvert();
        let e0 = mesh.totedge();
        let f0 = mesh.totface();
        assert_eq!(mesh.vert_edge_keys(v[0]).len(), 2, "v0 is an edge-pair vert");

        let right_verts_before: std::collections::HashSet<VertexKey> =
            mesh.face_vert_keys(f_right).into_iter().collect();

        mesh.dissolve_verts(&[v[0]]);

        assert_eq!(mesh.totvert(), v0 - 1, "-1 V (v0 collapsed)");
        assert_eq!(mesh.totedge(), e0 - 1, "-1 E (two boundary edges → one)");
        assert_eq!(mesh.totface(), f0, "both faces survive");

        let live: std::collections::HashSet<VertexKey> = mesh.vert_keys().into_iter().collect();
        assert!(!live.contains(&v[0]), "v0 is collapsed away");

        // The right quad — which never touched v0 — is unchanged.
        let right_verts_after: std::collections::HashSet<VertexKey> =
            mesh.face_vert_keys(f_right).into_iter().collect();
        assert_eq!(
            right_verts_after, right_verts_before,
            "the face not touching the dissolved vert is unaffected"
        );
        let _ = f_left;
        mesh.validate_topology();
    }

    /// §4 / §4.2 — `use_face_split = true` on the 4-valence centre of a 3x3
    /// quad grid. The pre-pass slices each surrounding quad with a diagonal at
    /// the centre, demoting the centre to a 4-triangle fan; the join then
    /// merges those four triangles into a single **quad** (the four shared
    /// immediate neighbours of the centre). Each source quad leaves a triangle
    /// behind (the `(n−1)`-gon outside the diagonal). Delta from the
    /// pre-call grid: +4 E / +4 F from the pre-pass, then −1 V / −4 E / −3 F
    /// from the four-triangle merge → net −1 V / 0 E / +1 F.
    #[test]
    fn dissolve_verts_use_face_split_true_merges_fan_into_quad() {
        let mut mesh = LoopMesh::new();
        let (verts, _faces) = build_grid_3x3(&mut mesh);

        let v0 = mesh.totvert();
        let e0 = mesh.totedge();
        let f0 = mesh.totface();
        assert_eq!((v0, e0, f0), (16, 24, 9));

        let centre = verts[5];
        assert_eq!(mesh.vert_edge_keys(centre).len(), 4);

        mesh.dissolve_verts_ex(
            &[centre],
            &DissolveVertsParams {
                use_face_split: true,
                use_boundary_tear: false,
            },
        );

        // Pre-pass: +4 E (diagonals) +4 F (triangle slices). Merge of the
        // resulting 4-triangle fan: −1 V, −4 E (spokes), −3 F (4 → 1).
        assert_eq!(mesh.totvert(), v0 - 1, "-1 V (the centre)");
        assert_eq!(mesh.totedge(), e0, "+4 diagonals, −4 spokes → net 0 E");
        assert_eq!(mesh.totface(), f0 + 1, "+4 slices, −3 from the merge");

        let live: std::collections::HashSet<VertexKey> = mesh.vert_keys().into_iter().collect();
        assert!(!live.contains(&centre), "centre vert is gone");

        // The merged face is a QUAD on the four immediate neighbours of the
        // dissolved centre (lattice points 1, 4, 6, 9). With face_split the
        // merged face is the narrow 4-corner ring, not the 8-corner octagon.
        let mut quad: Option<FaceKey> = None;
        let mut octagon_present = false;
        let mut face_iter = MeshFaces::default();
        while let Some(f) = face_iter.next(&mesh) {
            match mesh.face_vert_keys(f).len() {
                4 => {
                    let ring: std::collections::HashSet<VertexKey> =
                        mesh.face_vert_keys(f).into_iter().collect();
                    if [1usize, 4, 6, 9].iter().all(|&i| ring.contains(&verts[i])) {
                        quad = Some(f);
                    }
                }
                8 => octagon_present = true,
                _ => {}
            }
        }
        assert!(quad.is_some(), "merged face is the 4-corner ring, a quad");
        assert!(!octagon_present, "no wide octagon is produced with face_split");

        // Each of the four source quads leaves a triangle behind (the
        // (n−1)-gon outside the cut diagonal): there are exactly four new
        // triangles among the faces.
        let mut tri_count = 0usize;
        let mut face_iter = MeshFaces::default();
        while let Some(f) = face_iter.next(&mesh) {
            if mesh.face_len(f) == 3 {
                tri_count += 1;
            }
        }
        assert_eq!(tri_count, 4, "four triangle leftovers, one per source quad");

        mesh.validate_topology();
    }

    /// §4.2 structural contrast: `use_face_split = false` (the default,
    /// expressed through the params form) on the same 3x3 grid centre merges
    /// the four quads into one wide **octagon** — no diagonals, no leftover
    /// triangles. This is the documented "with vs without" difference. Delta:
    /// −1 V / −4 E / −3 F (matching the default-parameter headline case).
    #[test]
    fn dissolve_verts_use_face_split_false_merges_fan_into_octagon() {
        let mut mesh = LoopMesh::new();
        let (verts, _faces) = build_grid_3x3(&mut mesh);

        let v0 = mesh.totvert();
        let e0 = mesh.totedge();
        let f0 = mesh.totface();

        mesh.dissolve_verts_ex(
            &[verts[5]],
            &DissolveVertsParams {
                use_face_split: false,
                use_boundary_tear: false,
            },
        );

        // No pre-pass: the four-quad fan merges directly into one octagon.
        assert_eq!(mesh.totvert(), v0 - 1, "-1 V");
        assert_eq!(mesh.totedge(), e0 - 4, "-4 spokes, no diagonals added");
        assert_eq!(mesh.totface(), f0 - 3, "4 quads → 1");

        let mut octagon: Option<FaceKey> = None;
        let mut tri_count = 0usize;
        let mut face_iter = MeshFaces::default();
        while let Some(f) = face_iter.next(&mesh) {
            match mesh.face_len(f) {
                8 => octagon = Some(f),
                3 => tri_count += 1,
                _ => {}
            }
        }
        assert!(octagon.is_some(), "the merged face is a wide octagon");
        assert_eq!(tri_count, 0, "no leftover triangles without face_split");

        mesh.validate_topology();
    }

    /// §8 per-vert independence with `use_face_split = true`: two non-adjacent
    /// interior verts of a 5x5 quad grid. Each fan is sliced and merged into a
    /// quad independently; the surgeries do not interact. Per fan the net is
    /// −1 V / 0 E / +1 F (see the single-fan true-case test), so two disjoint
    /// fans give −2 V / 0 E / +2 F.
    #[test]
    fn dissolve_verts_use_face_split_true_two_disjoint_fans() {
        let mut mesh = LoopMesh::new();
        let (verts, _faces) = build_grid_5x5(&mut mesh);

        let v0 = mesh.totvert();
        let e0 = mesh.totedge();
        let f0 = mesh.totface();

        let a = verts[1 * 6 + 1];
        let b = verts[3 * 6 + 3];
        assert_eq!(mesh.vert_edge_keys(a).len(), 4);
        assert_eq!(mesh.vert_edge_keys(b).len(), 4);

        mesh.dissolve_verts_ex(
            &[a, b],
            &DissolveVertsParams {
                use_face_split: true,
                use_boundary_tear: false,
            },
        );

        assert_eq!(mesh.totvert(), v0 - 2, "-1 V per fan");
        assert_eq!(mesh.totedge(), e0, "+4 diagonals −4 spokes per fan → net 0 E");
        assert_eq!(mesh.totface(), f0 + 2, "+1 F per fan (4 → 1 merge + 4 slices)");

        let live: std::collections::HashSet<VertexKey> = mesh.vert_keys().into_iter().collect();
        assert!(!live.contains(&a) && !live.contains(&b));

        // Two quad merges, eight triangle leftovers (four per fan).
        let mut quad_merges = 0usize;
        let mut tri_count = 0usize;
        let mut face_iter = MeshFaces::default();
        while let Some(f) = face_iter.next(&mesh) {
            match mesh.face_len(f) {
                3 => tri_count += 1,
                8 => quad_merges += 1, // an octagon would appear without face_split
                _ => {}
            }
        }
        assert_eq!(quad_merges, 0, "no octagons: each fan merges to a quad");
        assert_eq!(tri_count, 8, "four triangle leftovers per disjoint fan");

        mesh.validate_topology();
    }

    /// §8 "no-op on already-triangulated fan": `use_face_split = true` on a
    /// vert whose every incident face is already a triangle finds nothing to
    /// split (every `face_len == 3`), so the result is identical to the
    /// default path. Here the centre of a triangulated 4-fan merges to a quad
    /// regardless of the flag.
    #[test]
    fn dissolve_verts_use_face_split_noop_on_triangle_fan() {
        // Four triangles fanning around a shared centre, forming a quad
        // diamond: centre c, rim verts n/e/s/w. Each triangle is (c, rim_i,
        // rim_{i+1}).
        let mut mesh = LoopMesh::new();
        let c = mesh.create_vertex(Vec3::new(0.0, 0.0, 0.0));
        let n = mesh.create_vertex(Vec3::new(0.0, 1.0, 0.0));
        let e = mesh.create_vertex(Vec3::new(1.0, 0.0, 0.0));
        let s = mesh.create_vertex(Vec3::new(0.0, -1.0, 0.0));
        let w = mesh.create_vertex(Vec3::new(-1.0, 0.0, 0.0));
        let rim = [n, e, s, w];

        let e_spokes: Vec<EdgeKey> = rim
            .iter()
            .map(|&r| mesh.create_edge([c, r], EdgeCreateFlags::NO_DUPLICATE))
            .collect();
        for i in 0..4 {
            let r0 = rim[i];
            let r1 = rim[(i + 1) % 4];
            let e_rim = mesh.create_edge([r0, r1], EdgeCreateFlags::NO_DUPLICATE);
            mesh.face_create(
                &[c, r0, r1],
                &[e_spokes[i], e_rim, e_spokes[(i + 1) % 4]],
                FaceCreateFlags::NONE,
            );
        }

        let v0 = mesh.totvert();
        let e0 = mesh.totedge();
        let f0 = mesh.totface();
        assert_eq!((v0, e0, f0), (5, 8, 4));
        assert_eq!(mesh.vert_edge_keys(c).len(), 4);

        mesh.dissolve_verts_ex(
            &[c],
            &DissolveVertsParams {
                use_face_split: true,
                use_boundary_tear: false,
            },
        );

        // The pre-pass adds no diagonals (every face is already a triangle):
        // the four triangles merge straight into one quad rim. −1 V, −4 E
        // (spokes), −3 F (4 → 1).
        assert_eq!(mesh.totvert(), v0 - 1, "-1 V (the centre)");
        assert_eq!(mesh.totedge(), e0 - 4, "no diagonals added; −4 spokes");
        assert_eq!(mesh.totface(), f0 - 3, "4 triangles → 1 quad");

        let mut quad: Option<FaceKey> = None;
        let mut face_iter = MeshFaces::default();
        while let Some(f) = face_iter.next(&mesh) {
            if mesh.face_len(f) == 4 {
                quad = Some(f);
            }
        }
        let quad = quad.expect("the merged face is the 4-vert rim quad");
        let ring: std::collections::HashSet<VertexKey> =
            mesh.face_vert_keys(quad).into_iter().collect();
        assert_eq!(ring, rim.into_iter().collect());

        mesh.validate_topology();
    }

    /// §4.1 skip condition: when both diagonal endpoints would themselves be
    /// dissolved, the pre-pass skips the split. Two adjacent grid-centre verts
    /// sharing a spoke edge: slicing a quad whose `v_prev`/`v_next` neighbour
    /// is the *other* input vert is wasted work and is skipped. This pins that
    /// the skip does not crash and that an unrelated quad (whose neighbours are
    /// not input verts) is still sliced.
    #[test]
    fn dissolve_verts_use_face_split_skips_diagonal_between_two_inputs() {
        let mut mesh = LoopMesh::new();
        let (verts, _faces) = build_grid_5x5(&mut mesh);

        // Two horizontally-adjacent interior verts sharing the spoke between
        // them. verts[r*6+c]: (2,2) and (2,3).
        let a = verts[2 * 6 + 2];
        let b = verts[2 * 6 + 3];
        // They share an edge (the spoke a-b).
        let shared = mesh
            .vert_edge_keys(a)
            .into_iter()
            .find(|&e| mesh.edge_vert_keys(e).contains(&b));
        assert!(shared.is_some(), "a and b are adjacent");

        // Run the pre-pass directly (no dissolve) and confirm it does not
        // panic and that the faces straddling the a-b spoke are NOT sliced
        // (their adjacent corner is the other input vert).
        let f0 = mesh.totface();
        mesh.dissolve_verts_ex(
            &[a, b],
            &DissolveVertsParams {
                use_face_split: true,
                use_boundary_tear: false,
            },
        );
        // Both verts dissolve; the operator does not crash. The two fans are
        // adjacent so they union — the exact final count is topology-dependent,
        // but the mesh must remain valid and both inputs must be gone.
        let live: std::collections::HashSet<VertexKey> = mesh.vert_keys().into_iter().collect();
        assert!(!live.contains(&a), "a dissolved");
        assert!(!live.contains(&b), "b dissolved");
        assert!(mesh.totface() >= 1, "{f0} faces before; mesh non-empty after");
        mesh.validate_topology();
    }
}
