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
    /// cross / interior tri-fan / boundary tee) lay the n-gon corner-cut
    /// pre-pass first (§5.1) and are then routed through the **single-vert
    /// dissolve helper**, whose endgame is a reaping-enabled collapse: a
    /// triangle-corner collapse is reaped to nothing so a degenerating fan
    /// closes onto a single clean face rather than a doubled / 2-gon remnant
    /// (§5.1.1; DISSOLVE_VERTS.md §6.6). This is the load-bearing distinction
    /// from the set-input dissolve-verts join, which would keep that doubled
    /// face live (DISSOLVE_VERTS.md §8) and stall one removal short of the
    /// oracle on the `triangulated_fan(4)` → 3/3/1 and `mixed_ngon_patch`
    /// step-2 → 4/4/1 endgames. Wire pairs are collapsed edge-into-edge so the
    /// two neighbours weld into one longer wire (§5.1).
    ///
    /// The removals are edge-independent (§4.3) but not face-independent: two
    /// removed verts can be diagonal corners of one quad and so share that
    /// face. Dissolving sequentially lets each vert's fan merge see the
    /// topology the previous removals left, so those overlapping fans weld
    /// through their shared faces — which is what produces the §5.2 mix of
    /// triangles and quads (and the −12 E / −3 F net) rather than nine
    /// disjoint per-vert merges.
    fn unsubdivide_dissolve_removed(&mut self, removed: &[VertexKey]) {
        // §5.1 / §4.1 removed-set membership wall for the n-gon pre-pass skip:
        // a corner diagonal is wasted (and would only be re-walked) when one of
        // its endpoints is itself about to be dissolved.
        let removed_set = self.push_vert_op_flag_layer();
        const REMOVED_BIT: u8 = 0;
        for &v in removed {
            if self.vertices.try_get(v).is_some() {
                self.vert_op_flag_set(removed_set, v, REMOVED_BIT);
            }
        }

        for &v in removed {
            if self.vertices.try_get(v).is_none() {
                continue;
            }
            // §4.4 / §5.1 live-revalidation gate. The two-colour fill (§4) is an
            // over-approximation on a non-bipartite component (an odd cycle): it
            // can mark two adjacent verts for removal. Re-confirm the §3 shape
            // against the *current* topology — after the preceding removals in
            // this pass — and skip any vert whose fan no longer matches. On a
            // bipartite component this never fires (the removed set is genuinely
            // independent); on an odd cycle it is what prunes the over-marked set
            // down to the maximal non-interfering subset the oracle removes.
            if !self.unsubdivide_shape_still_dissolvable(v) {
                continue;
            }
            if self.unsubdivide_is_wire_pair(v) {
                let mut disk = self.disk_edges(v);
                if let Some(edge) = disk.next(self) {
                    let _ = self.vert_collapse_edge(edge, v);
                }
            } else {
                // §5.1 n-gon pre-pass: cut each surrounding face larger than a
                // triangle down to a clean corner triangle at `v`, so the fan
                // join sweeps those corner triangles into the merged face.
                self.unsubdivide_face_split_pre_pass(v, removed_set, REMOVED_BIT);
                // §5.1 / §5.1.1: dissolve `v` through the single-vert helper.
                // Its boundary-tee endgame is a reaping-enabled collapse, so a
                // triangle corner is reaped to nothing and the degenerating fan
                // lands on a single clean face — not a doubled-face remnant.
                self.vert_dissolve(v);
            }
        }

        self.pop_vert_op_flag_layer(removed_set);
    }

    /// §5.1 — the per-vert n-gon corner pre-pass. For every face incident at
    /// `v` whose corner count is larger than a triangle, slice the diagonal
    /// `v_prev – v_next` (the two perimeter neighbours of `v` on that face) so
    /// `v`'s corner becomes a clean triangle that the subsequent fan join can
    /// peel off. The diagonal is skipped when either endpoint is itself a
    /// removed vert (§5.1 / §4.1): that diagonal would be wasted work and only
    /// re-walked once the neighbour dissolves. The diagonal never touches `v`,
    /// so the per-face targets are gathered up front and the splits applied
    /// afterwards (a split of one fan face does not move `v`'s loop on another).
    fn unsubdivide_face_split_pre_pass(
        &mut self,
        v: VertexKey,
        removed_set: crate::op_flags::VertOpFlagLayer,
        removed_bit: u8,
    ) {
        let mut diagonals: SmallVec<[(FaceKey, VertexKey, VertexKey); 8]> = SmallVec::new();
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
                if self.vert_op_flag_test(removed_set, v_next, removed_bit)
                    || self.vert_op_flag_test(removed_set, v_prev, removed_bit)
                {
                    continue;
                }
                diagonals.push((face, v_prev, v_next));
            }
        }
        for (face, v_prev, v_next) in diagonals {
            let _ = self.face_split(face, v_prev, v_next);
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

    /// §4.4 / §5.1 — the live-revalidation predicate. A marked vert is dissolved
    /// only if its fan *still* presents one of the dissolvable §3 **radial-mix
    /// shapes** against the current topology (§4.4: "no longer match a
    /// boundary-tee against the now-collapsed fan"). The gate re-checks only the
    /// radial-mix classification, deliberately **not** the already-bounds-a-face
    /// rejection: that rejection is an *eligibility-stage* guard (it keeps a 3-fan
    /// from ever marking its rim verts, §7.1), whereas at dissolve time the
    /// over-marked odd-cycle verts are pruned purely by whether their fan still
    /// presents a dissolvable spoke shape after the preceding removals collapsed
    /// it. On a bipartite component every marked vert still matches, so this gate
    /// never fires; on an odd cycle it is what lands the achieved removed set on
    /// the maximal non-interfering subset (§4.4).
    fn unsubdivide_shape_still_dissolvable(&self, v: VertexKey) -> bool {
        self.unsubdivide_classify(v) != UnsubdivideShape::Ineligible
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

    /// Full face-arity histogram keyed by side count (3 = tri, 4 = quad,
    /// 5 = pentagon, …), for the §7.2 mixed-region pins where pentagons matter.
    fn face_arity_histogram(mesh: &LoopMesh) -> std::collections::BTreeMap<usize, usize> {
        let mut hist: std::collections::BTreeMap<usize, usize> = std::collections::BTreeMap::new();
        for f in mesh.face_keys() {
            *hist.entry(mesh.face_vert_keys(f).len()).or_insert(0) += 1;
        }
        hist
    }

    /// Build a closed **triangulated fan** of `s` spokes (§7.1): a central
    /// vertex ringed by `s` rim verts joined edge-to-edge into a closed ring,
    /// with `s` triangles `(center, rim[i], rim[i + 1])`. Yields `s + 1` V /
    /// `2s` E / `s` F, all triangles. Returns `(center, rim_verts)`.
    fn build_triangulated_fan(mesh: &mut LoopMesh, s: usize) -> (VertexKey, Vec<VertexKey>) {
        assert!(s >= 3);
        let center = mesh.create_vertex(Vec3::ZERO);
        let mut rim: Vec<VertexKey> = Vec::with_capacity(s);
        for i in 0..s {
            let angle = std::f32::consts::TAU * (i as f32) / (s as f32);
            rim.push(mesh.create_vertex(Vec3::new(angle.cos(), angle.sin(), 0.0)));
        }
        for i in 0..s {
            let a = rim[i];
            let b = rim[(i + 1) % s];
            let e_ca = mesh.create_edge([center, a], EdgeCreateFlags::NO_DUPLICATE);
            let e_ab = mesh.create_edge([a, b], EdgeCreateFlags::NO_DUPLICATE);
            let e_bc = mesh.create_edge([b, center], EdgeCreateFlags::NO_DUPLICATE);
            mesh.face_create(&[center, a, b], &[e_ca, e_ab, e_bc], FaceCreateFlags::NONE);
        }
        (center, rim)
    }

    /// Build a **uniformly triangulated grid** (§7.1): an `n × n`-cell quad
    /// lattice with every cell cut along its bottom-left→top-right diagonal into
    /// two triangles. Yields `(n+1)^2` V, `3n^2 + 2n` E, `2n^2` F (all
    /// triangles). Returns the verts row-major, `verts[r * (n + 1) + c]`.
    fn build_triangulated_grid(mesh: &mut LoopMesh, n: usize) -> Vec<VertexKey> {
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
                // Diagonal v_bl -- v_tr splits the cell into two triangles.
                let e_b = mesh.create_edge([v_bl, v_br], EdgeCreateFlags::NO_DUPLICATE);
                let e_r = mesh.create_edge([v_br, v_tr], EdgeCreateFlags::NO_DUPLICATE);
                let e_d = mesh.create_edge([v_tr, v_bl], EdgeCreateFlags::NO_DUPLICATE);
                let e_t = mesh.create_edge([v_tr, v_tl], EdgeCreateFlags::NO_DUPLICATE);
                let e_l = mesh.create_edge([v_tl, v_bl], EdgeCreateFlags::NO_DUPLICATE);
                mesh.face_create(&[v_bl, v_br, v_tr], &[e_b, e_r, e_d], FaceCreateFlags::NONE);
                mesh.face_create(&[v_bl, v_tr, v_tl], &[e_d, e_t, e_l], FaceCreateFlags::NONE);
            }
        }
        verts
    }

    /// Build the §7.2 canonical **mixed n-gon patch**: 9 V / 13 E / 5 F made of
    /// two triangles, two quads and one pentagon sharing the two interior verts
    /// `v4`/`v5`. Built vertex-for-vertex and face-for-face to the exact §7.2
    /// adjacency so it matches the canonical fixture topology (a patch with the
    /// same arity histogram but a different hub valence or pentagon cycle would
    /// coarsen differently). Returns the verts in index order so a caller can tag
    /// the whole patch (`&all`) or just the interior pair (`[v[4], v[5]]`).
    fn build_mixed_ngon_patch(mesh: &mut LoopMesh) -> Vec<VertexKey> {
        // §7.2 vertex table: the patch boundary is the heptagon
        // v0-v1-v2-v8-v7-v6-v3; v4 (hub of qA/qB/pent) and v5 (shared by
        // qB/triR/triT/pent) are the two interior verts. v2,v5,v8 are collinear
        // at x = 2, so triR=[2,8,5] is a zero-area but valid face cycle.
        let coords = [
            (0.0, 0.0), // 0  corner
            (1.0, 0.0), // 1  bottom-edge midpoint
            (2.0, 0.0), // 2  corner
            (0.0, 1.0), // 3  left-edge midpoint
            (1.0, 1.0), // 4  interior hub  (qA, qB, pent)
            (2.0, 1.0), // 5  interior       (qB, triR, triT, pent)
            (0.0, 2.0), // 6  corner
            (1.0, 2.0), // 7  top-edge midpoint
            (2.0, 2.0), // 8  corner
        ];
        let v: Vec<VertexKey> = coords
            .iter()
            .map(|&(x, y)| mesh.create_vertex(Vec3::new(x, y, 0.0)))
            .collect();
        let mut edge = |mesh: &mut LoopMesh, a: usize, b: usize| {
            mesh.create_edge([v[a], v[b]], EdgeCreateFlags::NO_DUPLICATE)
        };
        // The 13 §7.2 edges: 7 boundary + 6 interior. Each created once
        // (NO_DUPLICATE), then reused across the faces that share it.
        let e01 = edge(mesh, 0, 1); // boundary
        let e12 = edge(mesh, 1, 2); // boundary
        let e28 = edge(mesh, 2, 8); // boundary
        let e87 = edge(mesh, 8, 7); // boundary
        let e76 = edge(mesh, 7, 6); // boundary
        let e63 = edge(mesh, 6, 3); // boundary
        let e30 = edge(mesh, 3, 0); // boundary
        let e34 = edge(mesh, 3, 4); // interior
        let e14 = edge(mesh, 1, 4); // interior
        let e45 = edge(mesh, 4, 5); // interior
        let e52 = edge(mesh, 5, 2); // interior
        let e58 = edge(mesh, 5, 8); // interior
        let e57 = edge(mesh, 5, 7); // interior

        // qA = [0, 1, 4, 3] (quad).
        mesh.face_create(&[v[0], v[1], v[4], v[3]], &[e01, e14, e34, e30], FaceCreateFlags::NONE);
        // qB = [1, 2, 5, 4] (quad).
        mesh.face_create(&[v[1], v[2], v[5], v[4]], &[e12, e52, e45, e14], FaceCreateFlags::NONE);
        // triR = [2, 8, 5] (triangle, zero-area sliver).
        mesh.face_create(&[v[2], v[8], v[5]], &[e28, e58, e52], FaceCreateFlags::NONE);
        // triT = [5, 8, 7] (triangle).
        mesh.face_create(&[v[5], v[8], v[7]], &[e58, e87, e57], FaceCreateFlags::NONE);
        // pent = [3, 4, 5, 7, 6] (pentagon).
        mesh.face_create(&[v[3], v[4], v[5], v[7], v[6]], &[e34, e45, e57, e76, e63], FaceCreateFlags::NONE);
        v
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

    /// §7.1 triangulated fan, 6 spokes, whole mesh tagged, `iterations = 1`.
    /// 7 V / 12 E / 6 F → 4 V / 6 E / 3 F (three triangles). The center
    /// survives as the hub (excluded at eligibility on the edge-count cap for
    /// `s ≥ 5`); an independent rim subset is removed. The output is all
    /// triangles — no quad pre-pass remnant on a fan of triangles.
    #[test]
    fn unsubdivide_tri_fan6_whole_mesh_one_step() {
        let mut mesh = LoopMesh::new();
        let (center, rim) = build_triangulated_fan(&mut mesh, 6);
        let mut all = vec![center];
        all.extend_from_slice(&rim);

        assert_eq!(mesh.totvert(), 7);
        assert_eq!(mesh.totedge(), 12);
        assert_eq!(mesh.totface(), 6);

        mesh.unsubdivide(&all, 1);

        assert_eq!(mesh.totvert(), 4, "4 V — center + 3 surviving rim verts");
        assert_eq!(mesh.totedge(), 6, "6 E");
        assert_eq!(mesh.totface(), 3, "3 F");
        let (triangles, quads) = face_corner_histogram(&mesh);
        assert_eq!(triangles, 3, "all-triangle output (no quad pre-pass on a tri fan)");
        assert_eq!(quads, 0);

        let live: HashSet<VertexKey> = mesh.vert_keys().into_iter().collect();
        assert!(live.contains(&center), "center survives as the coarser fan hub");
        mesh.validate_topology();
    }

    /// §7.1 triangulated fan, 4 spokes, whole mesh tagged, `iterations = 1`.
    /// 5 V / 8 E / 4 F → 3 V / 3 E / 1 F (single triangle). The `s = 4` eligible
    /// subgraph is the non-bipartite wheel `W₄`, so §4.4 governs: the fill seeds
    /// the center *kept* and over-marks **all four** rim verts *removed*. The
    /// dissolve pass's live-revalidation gate (§5.1) then dissolves the first two
    /// marked rim verts cleanly; the remaining two no longer match a boundary-tee
    /// against the collapsed fan, so they are skipped and survive. The achieved
    /// removed set is **two adjacent rim verts** (the surviving two are also
    /// adjacent) — not edge-independent, which is only possible on an odd cycle.
    /// The center is held *kept* throughout as the seed.
    ///
    /// The single-triangle pin is reached via the §5.1.1 degenerating-fan
    /// endgame: removing `r0` lands a clean 3-triangle fan (4/6/3, no doubled
    /// face); removing `r1` is a triangle-corner collapse whose single-vert-
    /// helper **reaping** (DISSOLVE_VERTS.md §6.6) drops the fan onto one clean
    /// triangle (3/3/1); the §5.1 gate then rejects `r2`/`r3` (the live shape is
    /// a lone triangle corner matching no §3 shape).
    #[test]
    fn unsubdivide_tri_fan4_whole_mesh_one_step() {
        let mut mesh = LoopMesh::new();
        let (center, rim) = build_triangulated_fan(&mut mesh, 4);
        let mut all = vec![center];
        all.extend_from_slice(&rim);

        assert_eq!(mesh.totvert(), 5);
        assert_eq!(mesh.totedge(), 8);
        assert_eq!(mesh.totface(), 4);

        mesh.unsubdivide(&all, 1);

        assert_eq!(mesh.totvert(), 3, "3 V — two adjacent rim verts removed (§4.4)");
        assert_eq!(mesh.totedge(), 3, "3 E — single triangle");
        assert_eq!(mesh.totface(), 1, "1 F — collapsed to one triangle");
        let (triangles, quads) = face_corner_histogram(&mesh);
        assert_eq!(triangles, 1);
        assert_eq!(quads, 0);

        let live: HashSet<VertexKey> = mesh.vert_keys().into_iter().collect();
        assert!(live.contains(&center), "center survives (held kept as the seed)");
        mesh.validate_topology();
    }

    /// §7.1 triangulated fan, 3 spokes: a **complete no-op**. All three rim
    /// verts fail eligibility on the already-bounds-a-face test (their
    /// neighbour-ring already bounds the opposite fan triangle), so the eligible
    /// subgraph is just the center — a lone component, coloured *kept*, removing
    /// nothing. Verified for whole-mesh tagging here; the guide pins the same
    /// no-op for rim-only and center-only tagging.
    #[test]
    fn unsubdivide_tri_fan3_whole_mesh_is_noop() {
        let mut mesh = LoopMesh::new();
        let (center, rim) = build_triangulated_fan(&mut mesh, 3);
        let mut all = vec![center];
        all.extend_from_slice(&rim);

        assert_eq!(mesh.totvert(), 4);
        assert_eq!(mesh.totedge(), 6);
        assert_eq!(mesh.totface(), 3);

        mesh.unsubdivide(&all, 1);

        assert_eq!(mesh.totvert(), 4, "no-op: rim verts blocked at eligibility");
        assert_eq!(mesh.totedge(), 6, "no-op");
        assert_eq!(mesh.totface(), 3, "no-op");
        mesh.validate_topology();
    }

    /// §7.1 uniformly triangulated grid is a **complete no-op** at every
    /// iteration: none of the interior verts matches a dissolvable §3 shape
    /// (valence-6 verts exceed the cap; valence-4 verts hit the
    /// already-bounds-a-face rejection). Checked for `n = 2` and `n = 4` across
    /// several iteration counts.
    #[test]
    fn unsubdivide_triangulated_grid_is_noop() {
        for (n, exp_v, exp_e, exp_f) in [(2usize, 9usize, 16usize, 8usize), (4, 25, 56, 32)] {
            for iterations in [1, 2, 4] {
                let mut mesh = LoopMesh::new();
                let verts = build_triangulated_grid(&mut mesh, n);

                assert_eq!(mesh.totvert(), exp_v, "n={n} input V");
                assert_eq!(mesh.totedge(), exp_e, "n={n} input E");
                assert_eq!(mesh.totface(), exp_f, "n={n} input F");

                mesh.unsubdivide(&verts, iterations);

                assert_eq!(mesh.totvert(), exp_v, "n={n} it={iterations} no-op V");
                assert_eq!(mesh.totedge(), exp_e, "n={n} it={iterations} no-op E");
                assert_eq!(mesh.totface(), exp_f, "n={n} it={iterations} no-op F");
                mesh.validate_topology();
            }
        }
    }

    /// §7.2 canonical mixed n-gon patch, whole mesh tagged, `iterations = 1`.
    /// 9 V / 13 E / 5 F → 6 V / 10 E / 5 F: the three removed verts are `v2`, the
    /// hub `v4`, and `v7` (survivors `v0, v1, v3, v5, v6, v8`), and the welded
    /// neighbourhood is re-triangulated by the §5.1 pre-pass cuts, so the arity
    /// histogram flips `{tri:2, quad:2, pent:1}` → `{tri:5}` at unchanged face
    /// count. The eligible subgraph here is non-bipartite, so §4.4 over-mark /
    /// dissolve-time prune governs the achieved removed set.
    #[test]
    fn unsubdivide_mixed_ngon_whole_mesh_one_step() {
        let mut mesh = LoopMesh::new();
        let verts = build_mixed_ngon_patch(&mut mesh);

        assert_eq!(mesh.totvert(), 9);
        assert_eq!(mesh.totedge(), 13);
        assert_eq!(mesh.totface(), 5);
        assert_eq!(
            face_arity_histogram(&mesh),
            std::collections::BTreeMap::from([(3, 2), (4, 2), (5, 1)]),
            "input histogram {{tri:2, quad:2, pent:1}}"
        );

        mesh.unsubdivide(&verts, 1);

        assert_eq!(mesh.totvert(), 6, "6 V — three verts removed");
        assert_eq!(mesh.totedge(), 10, "10 E");
        assert_eq!(mesh.totface(), 5, "5 F — face count unchanged");
        assert_eq!(
            face_arity_histogram(&mesh),
            std::collections::BTreeMap::from([(3, 5)]),
            "histogram flips to {{tri:5}} via the §5.1 re-triangulation"
        );
        mesh.validate_topology();
    }

    /// §7.2 canonical mixed n-gon patch, whole mesh tagged, `iterations = 2`: the
    /// second step coarsens the triangle fan down to a single quad and the
    /// operator converges. 9/13/5 → 4 V / 4 E / 1 F (one quad); a third step is a
    /// no-op (§5.3 early-out).
    ///
    /// The §7.2 one-step result (6 V / 10 E / 5 F, {tri:5}) is reached, then the
    /// second step's five-triangle fan degenerates onto a single quad by the
    /// same §5.1.1 reaping endgame as the `triangulated_fan(4)` case: the final
    /// per-vert collapses reap their triangle corners rather than leaving doubled
    /// faces, landing the oracle's clean 4/4/1 single quad.
    #[test]
    fn unsubdivide_mixed_ngon_two_steps_converge() {
        let mut mesh = LoopMesh::new();
        let verts = build_mixed_ngon_patch(&mut mesh);

        mesh.unsubdivide(&verts, 2);

        assert_eq!(mesh.totvert(), 4, "4 V — converged to a single quad");
        assert_eq!(mesh.totedge(), 4, "4 E");
        assert_eq!(mesh.totface(), 1, "1 F — single quad");
        assert_eq!(
            face_arity_histogram(&mesh),
            std::collections::BTreeMap::from([(4, 1)]),
            "converged to a single quad"
        );
        mesh.validate_topology();

        // A third step finds nothing to remove (§5.3 early-out).
        let v_before = mesh.totvert();
        mesh.unsubdivide(&verts, 1);
        assert_eq!(mesh.totvert(), v_before, "converged: third step is a no-op");
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
