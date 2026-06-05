use smallvec::SmallVec;

use crate::loopmesh::*;

/// Per-vertex eligibility class for unsubdivide, decided by the radial mix of
/// the vert's incident edges. Mirrors the accept table in
/// `radial-mesh-guides/operators/UNSUBDIVIDE.md` §3.
#[derive(Clone, Copy, PartialEq, Eq)]
enum UnsubdivideShape {
    /// 4 edges, all interior (radial-2); 4 surrounding faces. A grid cell
    /// centre or an interior edge-midpoint vert.
    InteriorCross,
    /// 3 edges, all interior (radial-2); 3 surrounding faces. A grid vertex
    /// inside a triangulated patch.
    InteriorTriFan,
    /// 3 edges: 2 boundary (radial-1) + 1 interior (radial-2); 2 surrounding
    /// faces. A rim edge-midpoint vert.
    BoundaryTee,
    /// 2 edges, both wire (radial-0); 0 faces. A midpoint vert on a wire
    /// polyline.
    WirePair,
    /// Any other local shape: corner verts, valence ≥ 5, non-manifold spines,
    /// or a shape whose merged face would duplicate an existing one. Kept.
    Ineligible,
}

impl LoopMesh {
    /// Pattern-based grid coarsener. Each `iterations` step two-colours the
    /// eligible subgraph of the tagged `verts` (§3, §4) and dissolves one
    /// independent (checkerboard) class via the dissolve-family fan-merge (§5),
    /// welding the faces around each removed vert together.
    ///
    /// Implements the foundational core of
    /// `radial-mesh-guides/operators/UNSUBDIVIDE.md`: the every-other selection
    /// over the four-spoke / boundary-spoke / wire-pair dissolves. On a regular
    /// quad grid one step is a checkerboard half-coarsening, **not** an exact
    /// inverse of a subdivide (§2, §5.2): only an independent class is removed,
    /// so the welded faces come out as a mix of triangles and quads.
    ///
    /// No vertex is created, so every surviving vert keeps its exact input
    /// position (§8).
    ///
    /// `iterations` is clamped to at least `1`. The loop stops early on the
    /// first step that marks nothing for removal (§5.3): once a step dissolves
    /// nothing, no later step can find anything, so the operator converges
    /// rather than corrupting the mesh.
    pub fn unsubdivide(&mut self, verts: &[VertexKey], iterations: i32) {
        let iterations = iterations.max(1);
        for _ in 0..iterations {
            let removed = self.unsubdivide_step(verts);
            if removed == 0 {
                break;
            }
        }
    }

    /// One coarsening step. Re-runs eligibility (§3) over the *current*
    /// topology, two-colours the eligible subgraph (§4), and dissolves the
    /// *removed* class (§5). Returns the number of verts marked for removal so
    /// the [`Self::unsubdivide`] loop can early-out when a step does nothing.
    fn unsubdivide_step(&mut self, verts: &[VertexKey]) -> usize {
        // §3 / §4.1 — tagged-and-still-live membership wall. Untagged verts
        // (and verts a prior step removed) are never visited by the fill.
        let tagged = self.push_vert_op_flag_layer();
        const TAGGED_BIT: u8 = 0;
        for &v in verts {
            if self.vertices.try_get(v).is_some() {
                self.vert_op_flag_set(tagged, v, TAGGED_BIT);
            }
        }

        let removed = self.unsubdivide_two_colour(verts, tagged, TAGGED_BIT);

        self.pop_vert_op_flag_layer(tagged);

        if removed.is_empty() {
            return 0;
        }

        let count = removed.len();
        self.unsubdivide_dissolve_removed(&removed);
        count
    }

    /// §4 — the every-other two-colour flood fill. Breadth-first over the
    /// eligible subgraph (§4.2), one connected component at a time: the seed is
    /// coloured *kept*, then alternating waves flip *kept* ↔ *removed*. The
    /// returned set is the *removed* class — an independent set (§4.3): no two
    /// of its verts share an edge.
    fn unsubdivide_two_colour(
        &mut self,
        verts: &[VertexKey],
        tagged: crate::op_flags::VertOpFlagLayer,
        tagged_bit: u8,
    ) -> Vec<VertexKey> {
        // Colour layer: VISITED marks any vert the fill has reached; REMOVED
        // marks the dissolved class. A reached-but-not-removed vert is *kept*.
        let colour = self.push_vert_op_flag_layer();
        const VISITED_BIT: u8 = 0;
        const REMOVED_BIT: u8 = 1;

        let mut removed: Vec<VertexKey> = Vec::new();
        let mut frontier: Vec<VertexKey> = Vec::new();
        let mut next_frontier: Vec<VertexKey> = Vec::new();

        for &seed in verts {
            if self.vertices.try_get(seed).is_none() {
                continue;
            }
            if self.vert_op_flag_test(colour, seed, VISITED_BIT) {
                continue;
            }
            if !self.unsubdivide_is_eligible(seed, tagged, tagged_bit) {
                continue;
            }

            // Seed survives (§4.2): coloured visited, not removed.
            self.vert_op_flag_set(colour, seed, VISITED_BIT);
            frontier.clear();
            frontier.push(seed);
            // The seed's wave is *kept*, so its neighbours become *removed*.
            let mut next_wave_removes = true;

            while !frontier.is_empty() {
                next_frontier.clear();
                for &front_vert in &frontier {
                    let neighbours = self.unsubdivide_eligible_neighbours(
                        front_vert, tagged, tagged_bit,
                    );
                    for neighbour in neighbours {
                        if self.vert_op_flag_test(colour, neighbour, VISITED_BIT) {
                            continue;
                        }
                        self.vert_op_flag_set(colour, neighbour, VISITED_BIT);
                        if next_wave_removes {
                            self.vert_op_flag_set(colour, neighbour, REMOVED_BIT);
                            removed.push(neighbour);
                        }
                        next_frontier.push(neighbour);
                    }
                }
                std::mem::swap(&mut frontier, &mut next_frontier);
                next_wave_removes = !next_wave_removes;
            }
        }

        self.pop_vert_op_flag_layer(colour);
        removed
    }

    /// §5 — dissolve the *removed* class, one vert at a time (§5.1: "for one
    /// removed vert `v` … then dissolve `v`"). Face-bearing shapes (interior
    /// cross / interior tri-fan / boundary tee) are routed through the
    /// dissolve-family fan-merge with the n-gon corner-cut pre-pass on (§5.1);
    /// wire pairs are collapsed edge-into-edge so the two neighbours weld into
    /// one longer wire (§5.1).
    ///
    /// The removals are edge-independent (§4.3) but not face-independent: two
    /// removed verts can be diagonal corners of one quad and so share that
    /// face. Dissolving sequentially lets each vert's fan merge see the
    /// topology the previous removals left, so those overlapping fans weld
    /// through their shared faces — which is what produces the §5.2 mix of
    /// triangles and quads (and the −12 E / −3 F net) rather than nine
    /// disjoint per-vert merges.
    fn unsubdivide_dissolve_removed(&mut self, removed: &[VertexKey]) {
        let params = DissolveVertsParams {
            use_face_split: true,
            use_boundary_tear: false,
        };
        for &v in removed {
            if self.vertices.try_get(v).is_none() {
                continue;
            }
            if self.unsubdivide_is_wire_pair(v) {
                let mut disk = self.disk_edges(v);
                if let Some(edge) = disk.next(self) {
                    let _ = self.vert_collapse_edge(edge, v);
                }
            } else {
                self.dissolve_verts_ex(&[v], &params);
            }
        }
    }

    /// §3 — eligibility predicate. `v` must be tagged and match one of the
    /// dissolvable shapes by its incident-edge radial mix, and dissolving it
    /// must not duplicate an existing face (the already-bounds-a-face
    /// rejection).
    fn unsubdivide_is_eligible(
        &self,
        v: VertexKey,
        tagged: crate::op_flags::VertOpFlagLayer,
        tagged_bit: u8,
    ) -> bool {
        if !self.vert_op_flag_test(tagged, v, tagged_bit) {
            return false;
        }
        match self.unsubdivide_classify(v) {
            UnsubdivideShape::Ineligible => false,
            UnsubdivideShape::WirePair => true,
            _ => !self.unsubdivide_neighbour_ring_bounds_face(v),
        }
    }

    /// §3 — classify `v` by counting its incident edges' radial lengths into
    /// wire (0) / boundary (1) / interior (2) / non-manifold (≥3) buckets.
    fn unsubdivide_classify(&self, v: VertexKey) -> UnsubdivideShape {
        let mut wire = 0usize;
        let mut boundary = 0usize;
        let mut interior = 0usize;
        let mut non_manifold = 0usize;
        let mut total = 0usize;

        let mut disk = self.disk_edges(v);
        while let Some(edge) = disk.next(self) {
            total += 1;
            match self.unsubdivide_edge_radial_count_capped(edge, 3) {
                0 => wire += 1,
                1 => boundary += 1,
                2 => interior += 1,
                _ => non_manifold += 1,
            }
        }

        if non_manifold > 0 {
            return UnsubdivideShape::Ineligible;
        }
        if total == 4 && interior == 4 {
            return UnsubdivideShape::InteriorCross;
        }
        if total == 3 && interior == 3 {
            return UnsubdivideShape::InteriorTriFan;
        }
        if total == 3 && boundary == 2 && interior == 1 {
            return UnsubdivideShape::BoundaryTee;
        }
        if total == 2 && wire == 2 {
            return UnsubdivideShape::WirePair;
        }
        UnsubdivideShape::Ineligible
    }

    #[inline]
    fn unsubdivide_is_wire_pair(&self, v: VertexKey) -> bool {
        self.unsubdivide_classify(v) == UnsubdivideShape::WirePair
    }

    /// §3 — the already-bounds-a-face rejection. Dissolving `v` welds its fan
    /// into a single face whose perimeter is the ring of `v`'s neighbours. If
    /// that exact ring already bounds a face, joining would duplicate it, so
    /// `v` is kept. Checks every face incident to a neighbour rather than the
    /// whole mesh, keeping the probe local.
    fn unsubdivide_neighbour_ring_bounds_face(&self, v: VertexKey) -> bool {
        let mut ring: SmallVec<[VertexKey; 8]> = SmallVec::new();
        let mut disk = self.disk_edges(v);
        while let Some(edge) = disk.next(self) {
            let neighbour = {
                let [a, b] = self.edge_vert_keys(edge);
                if a == v {
                    b
                } else {
                    a
                }
            };
            if !ring.contains(&neighbour) {
                ring.push(neighbour);
            }
        }
        if ring.len() < 3 {
            return false;
        }

        let mut seen_faces: SmallVec<[FaceKey; 8]> = SmallVec::new();
        for &neighbour in &ring {
            for face in self.vert_face_keys(neighbour) {
                if seen_faces.contains(&face) {
                    continue;
                }
                seen_faces.push(face);
                let face_verts = self.face_vert_keys(face);
                if face_verts.len() != ring.len() {
                    continue;
                }
                if face_verts.iter().all(|fv| ring.contains(fv)) {
                    return true;
                }
            }
        }
        false
    }

    /// §4 — `v`'s eligible neighbours: the verts joined to `v` by a mesh edge
    /// that are themselves eligible (§3). These are the edges of the eligible
    /// subgraph the flood fill walks.
    fn unsubdivide_eligible_neighbours(
        &self,
        v: VertexKey,
        tagged: crate::op_flags::VertOpFlagLayer,
        tagged_bit: u8,
    ) -> SmallVec<[VertexKey; 8]> {
        let mut neighbours: SmallVec<[VertexKey; 8]> = SmallVec::new();
        let mut disk = self.disk_edges(v);
        while let Some(edge) = disk.next(self) {
            let neighbour = {
                let [a, b] = self.edge_vert_keys(edge);
                if a == v {
                    b
                } else {
                    a
                }
            };
            if neighbours.contains(&neighbour) {
                continue;
            }
            if self.unsubdivide_is_eligible(neighbour, tagged, tagged_bit) {
                neighbours.push(neighbour);
            }
        }
        neighbours
    }

    /// Count `edge`'s radial loops, stopping at `cap`. Distinguishes wire (0) /
    /// boundary (1) / interior (2) / non-manifold (≥3) without a full walk.
    fn unsubdivide_edge_radial_count_capped(&self, edge: EdgeKey, cap: usize) -> usize {
        let mut radial = self.radial_loops(edge);
        let mut count = 0usize;
        while count < cap && radial.next(self).is_some() {
            count += 1;
        }
        count
    }
}

#[cfg(test)]
mod tests {
    use glam::Vec3;
    use std::collections::HashSet;

    use crate::loopmesh::*;

    /// Build an `n × n`-cell quad grid (an `(n+1) × (n+1)` vertex lattice) on
    /// the integer lattice. Returns the verts in row-major order,
    /// `verts[r * (n + 1) + c]`.
    fn build_grid_n_faces(mesh: &mut LoopMesh, n: usize) -> Vec<VertexKey> {
        let stride = n + 1;
        let mut verts: Vec<VertexKey> = Vec::with_capacity(stride * stride);
        for r in 0..stride {
            for c in 0..stride {
                verts.push(mesh.create_vertex(Vec3::new(c as f32, r as f32, 0.0)));
            }
        }
        for r in 0..n {
            for c in 0..n {
                let v_bl = verts[r * stride + c];
                let v_br = verts[r * stride + c + 1];
                let v_tr = verts[(r + 1) * stride + c + 1];
                let v_tl = verts[(r + 1) * stride + c];
                let e_b = mesh.create_edge([v_bl, v_br], EdgeCreateFlags::NO_DUPLICATE);
                let e_r = mesh.create_edge([v_br, v_tr], EdgeCreateFlags::NO_DUPLICATE);
                let e_t = mesh.create_edge([v_tr, v_tl], EdgeCreateFlags::NO_DUPLICATE);
                let e_l = mesh.create_edge([v_tl, v_bl], EdgeCreateFlags::NO_DUPLICATE);
                mesh.face_create(
                    &[v_bl, v_br, v_tr, v_tl],
                    &[e_b, e_r, e_t, e_l],
                    FaceCreateFlags::NONE,
                );
            }
        }
        verts
    }

    fn face_corner_histogram(mesh: &LoopMesh) -> (usize, usize) {
        let mut triangles = 0usize;
        let mut quads = 0usize;
        for f in mesh.face_keys() {
            match mesh.face_vert_keys(f).len() {
                3 => triangles += 1,
                4 => quads += 1,
                _ => {}
            }
        }
        (triangles, quads)
    }

    /// §5.2 headline pin: `grid_n_faces(4)` whole-mesh, `iterations = 1`.
    /// 25 V / 40 E / 16 F → 16 V / 28 E / 13 F (8 triangles + 5 quads).
    #[test]
    fn unsubdivide_grid4_whole_mesh_one_step() {
        let mut mesh = LoopMesh::new();
        let verts = build_grid_n_faces(&mut mesh, 4);

        assert_eq!(mesh.totvert(), 25);
        assert_eq!(mesh.totedge(), 40);
        assert_eq!(mesh.totface(), 16);

        mesh.unsubdivide(&verts, 1);

        assert_eq!(mesh.totvert(), 16, "16 V — one independent class of 9 removed");
        assert_eq!(mesh.totedge(), 28, "28 E");
        assert_eq!(mesh.totface(), 13, "13 F");

        let (triangles, quads) = face_corner_histogram(&mesh);
        assert_eq!(triangles, 8, "8 triangles in the irregular coarsening");
        assert_eq!(quads, 5, "5 quads in the irregular coarsening");

        mesh.validate_topology();
    }

    /// §8 position exactness: no vertex is created, so every surviving vert
    /// keeps its exact input coordinate.
    #[test]
    fn unsubdivide_grid4_surviving_positions_bit_stable() {
        let mut mesh = LoopMesh::new();
        let verts = build_grid_n_faces(&mut mesh, 4);

        let original: Vec<(VertexKey, Vec3)> =
            verts.iter().map(|&v| (v, mesh.vertex_position(v))).collect();

        mesh.unsubdivide(&verts, 1);

        let live: HashSet<VertexKey> = mesh.vert_keys().into_iter().collect();
        let mut survivors = 0usize;
        for (v, pos) in original {
            if live.contains(&v) {
                survivors += 1;
                assert_eq!(
                    mesh.vertex_position(v),
                    pos,
                    "surviving vert {v:?} kept its exact input position"
                );
            }
        }
        assert_eq!(survivors, 16, "16 of the 25 input verts survive");
    }

    /// §4.3 independent-set guarantee: no two removed verts shared an edge —
    /// equivalently, every survivor edge endpoint is a survivor. Checked by
    /// confirming the removed count is exactly one checkerboard class (9) and
    /// the result is topologically valid (the dissolves did not interact).
    #[test]
    fn unsubdivide_grid4_removes_one_independent_class() {
        let mut mesh = LoopMesh::new();
        let verts = build_grid_n_faces(&mut mesh, 4);
        let v0 = mesh.totvert();

        mesh.unsubdivide(&verts, 1);

        assert_eq!(v0 - mesh.totvert(), 9, "exactly one bipartition class removed");
        mesh.validate_topology();
    }

    /// §4.3 / §5.3 early-out: an isolated lone eligible vert (all of whose
    /// eligible neighbours are walled off by untagged verts) seeds *kept* and
    /// removes nothing, leaving the mesh unchanged. Here only the centre of a
    /// 3×3-face grid is tagged; its neighbours are untagged, so the fill colours
    /// the centre kept and finds no eligible neighbour to remove.
    #[test]
    fn unsubdivide_lone_tagged_interior_vert_is_noop() {
        let mut mesh = LoopMesh::new();
        let verts = build_grid_n_faces(&mut mesh, 4);
        // verts[r*5+c]; (2,2) = 12 is a fully-interior 4-cross vert.
        let centre = verts[2 * 5 + 2];

        let v0 = mesh.totvert();
        let e0 = mesh.totedge();
        let f0 = mesh.totface();

        mesh.unsubdivide(&[centre], 1);

        assert_eq!(mesh.totvert(), v0, "no-op: V unchanged");
        assert_eq!(mesh.totedge(), e0, "no-op: E unchanged");
        assert_eq!(mesh.totface(), f0, "no-op: F unchanged");
        let live: HashSet<VertexKey> = mesh.vert_keys().into_iter().collect();
        assert!(live.contains(&centre), "the lone tagged vert survives");
        mesh.validate_topology();
    }

    /// §5.3 convergence pin: `grid_n_faces(4)` whole-mesh, `iterations = 2`.
    /// The second step re-runs eligibility over the *coarsened* topology and
    /// removes two more verts, stabilising at 14 V / 25 E / 12 F
    /// (10 triangles + 2 quads).
    #[test]
    fn unsubdivide_grid4_two_steps_converge() {
        let mut mesh = LoopMesh::new();
        let verts = build_grid_n_faces(&mut mesh, 4);

        mesh.unsubdivide(&verts, 2);

        assert_eq!(mesh.totvert(), 14, "14 V after the second step (§5.3)");
        assert_eq!(mesh.totedge(), 25, "25 E after the second step (§5.3)");
        assert_eq!(mesh.totface(), 12, "12 F after the second step (§5.3)");

        let (triangles, quads) = face_corner_histogram(&mesh);
        assert_eq!(triangles, 10, "10 triangles in the converged coarsening");
        assert_eq!(quads, 2, "2 quads in the converged coarsening");

        mesh.validate_topology();
    }

    /// §5.3 early termination: an `iterations` count past the grid's actual
    /// coarsening depth is harmless — the step that finds nothing to remove
    /// fires the early-out, so the result equals the two-step converged grid
    /// (14 V / 25 E / 12 F), unchanged by the excess steps.
    ///
    /// This also exercises stale-key safety: the caller's `verts` slice is
    /// tagged against the *original* mesh, so after step 1 removes ~9 of those
    /// keys the slice is full of freed keys. Each step must re-evaluate
    /// eligibility on the current topology and skip freed keys without panic or
    /// corruption (§5.3).
    #[test]
    fn unsubdivide_grid4_iterations_past_convergence_is_noop() {
        let mut mesh = LoopMesh::new();
        let verts = build_grid_n_faces(&mut mesh, 4);

        mesh.unsubdivide(&verts, 4);

        assert_eq!(mesh.totvert(), 14, "converged V unchanged by excess steps");
        assert_eq!(mesh.totedge(), 25, "converged E unchanged by excess steps");
        assert_eq!(mesh.totface(), 12, "converged F unchanged by excess steps");

        let (triangles, quads) = face_corner_histogram(&mesh);
        assert_eq!(triangles, 10);
        assert_eq!(quads, 2);

        mesh.validate_topology();
    }

    /// §5.3 stale-key direct check: running the operator one step at a time
    /// over the *same* original `verts` slice converges to the identical
    /// fixed point as a single two-step call, proving the stale keys left in
    /// the slice after each step are skipped (not dereferenced) and that
    /// eligibility is re-derived from current topology.
    #[test]
    fn unsubdivide_grid4_stepwise_matches_batched() {
        let mut stepwise = LoopMesh::new();
        let stepwise_verts = build_grid_n_faces(&mut stepwise, 4);
        // The slice is tagged against the original mesh; after the first call
        // most of its keys are freed. The second call must tolerate that.
        stepwise.unsubdivide(&stepwise_verts, 1);
        stepwise.unsubdivide(&stepwise_verts, 1);
        stepwise.unsubdivide(&stepwise_verts, 1);

        assert_eq!(stepwise.totvert(), 14);
        assert_eq!(stepwise.totedge(), 25);
        assert_eq!(stepwise.totface(), 12);
        stepwise.validate_topology();

        let mut batched = LoopMesh::new();
        let batched_verts = build_grid_n_faces(&mut batched, 4);
        batched.unsubdivide(&batched_verts, 2);

        assert_eq!(stepwise.totvert(), batched.totvert());
        assert_eq!(stepwise.totedge(), batched.totedge());
        assert_eq!(stepwise.totface(), batched.totface());
    }

    /// §6 partial-selection scope: tagging only a sub-region coarsens that
    /// region in place and walls the flood fill at the untagged border.
    ///
    /// Here exactly the two horizontally-adjacent interior-cross verts at grid
    /// positions (1,1) and (2,1) of a `grid_n_faces(3)` lattice (16 V / 24 E /
    /// 9 F) are tagged. Per §5.1/§5.2 they form a single eligible edge, so the
    /// fill keeps the seed and removes exactly **one** of them (§4.3 lone-edge
    /// component: seed kept, its one eligible neighbour removed). The removed
    /// vert is an interior cross, so the n-gon pre-pass re-triangulates its four
    /// surrounding quads — its four spokes are replaced by pre-pass diagonals
    /// (edges net-unchanged) and faces net +1. Measured delta:
    /// 16/24/9 → 15/24/10 (Δ −1 V / 0 E / +1 F). Euler-consistent: 15−24+10 = 1.
    #[test]
    fn unsubdivide_grid3_partial_two_interior_verts_one_step() {
        let mut mesh = LoopMesh::new();
        let verts = build_grid_n_faces(&mut mesh, 3);

        assert_eq!(mesh.totvert(), 16);
        assert_eq!(mesh.totedge(), 24);
        assert_eq!(mesh.totface(), 9);

        // stride = n + 1 = 4; verts[r * 4 + c] sits at Vec3(c, r, 0).
        // (1,1) = idx 5, (2,1) = idx 6 — two horizontally-adjacent interior
        // verts of the 4×4 lattice, matching the A/B scaffold's tag set.
        let partial = [verts[1 * 4 + 1], verts[1 * 4 + 2]];

        // Snapshot every untagged vert's position to prove survivors are
        // bit-stable: no vertex is created (§8), and only one tagged vert is
        // removed, so every untagged vert must keep its exact coordinate.
        let untagged_positions: Vec<(VertexKey, Vec3)> = verts
            .iter()
            .filter(|&&v| v != partial[0] && v != partial[1])
            .map(|&v| (v, mesh.vertex_position(v)))
            .collect();

        mesh.unsubdivide(&partial, 1);

        assert_eq!(mesh.totvert(), 15, "one of the two tagged verts removed");
        assert_eq!(mesh.totedge(), 24, "edges net-unchanged (spokes → diagonals)");
        assert_eq!(mesh.totface(), 10, "n-gon pre-pass nets +1 face");
        // Euler-consistent planar surface with one boundary loop.
        assert_eq!(
            mesh.totvert() as i64 - mesh.totedge() as i64 + mesh.totface() as i64,
            1,
        );

        // Exactly one of the two tagged verts survives; the untagged border is
        // untouched and every untagged vert keeps its exact position.
        let live: HashSet<VertexKey> = mesh.vert_keys().into_iter().collect();
        let tagged_survivors = partial.iter().filter(|&&v| live.contains(&v)).count();
        assert_eq!(tagged_survivors, 1, "seed kept, its one neighbour removed");
        for (v, pos) in untagged_positions {
            assert!(live.contains(&v), "untagged border vert {v:?} survives");
            assert_eq!(
                mesh.vertex_position(v),
                pos,
                "untagged vert {v:?} kept its exact input position"
            );
        }

        mesh.validate_topology();
    }

    /// §4.3 / §6 partial-selection no-op: a tagged set with no eligible pair
    /// dissolves nothing. Here the two tagged interior-cross verts are
    /// **diagonally** adjacent — not joined by a mesh edge — so the eligible
    /// subgraph has no edge: each is its own single-vert component, coloured
    /// *kept*, removing nothing. The mesh is unchanged.
    #[test]
    fn unsubdivide_grid3_partial_no_eligible_pair_is_noop() {
        let mut mesh = LoopMesh::new();
        let verts = build_grid_n_faces(&mut mesh, 3);

        let v0 = mesh.totvert();
        let e0 = mesh.totedge();
        let f0 = mesh.totface();

        // (1,1) = idx 5 and (2,2) = idx 10 are diagonal interior verts: both
        // eligible, but not joined by an edge, so the eligible subgraph is two
        // isolated nodes and nothing is removed.
        let diagonal = [verts[1 * 4 + 1], verts[2 * 4 + 2]];
        mesh.unsubdivide(&diagonal, 1);

        assert_eq!(mesh.totvert(), v0, "no-op: V unchanged");
        assert_eq!(mesh.totedge(), e0, "no-op: E unchanged");
        assert_eq!(mesh.totface(), f0, "no-op: F unchanged");
        let live: HashSet<VertexKey> = mesh.vert_keys().into_iter().collect();
        assert!(live.contains(&diagonal[0]) && live.contains(&diagonal[1]));
        mesh.validate_topology();
    }

    /// §1 — `iterations` is clamped to at least 1: a zero (or negative) count
    /// still runs one step.
    #[test]
    fn unsubdivide_iterations_clamped_to_one() {
        let mut mesh = LoopMesh::new();
        let verts = build_grid_n_faces(&mut mesh, 4);

        mesh.unsubdivide(&verts, 0);

        assert_eq!(mesh.totvert(), 16, "iterations=0 clamps to one step");
        assert_eq!(mesh.totface(), 13);
        mesh.validate_topology();
    }
}
