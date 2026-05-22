use crate::loopmesh::*;

impl LoopMesh {
    /// "Unglues" a face's corner at `l_sep` from its surrounding fan: peels
    /// `l_sep`'s face away along both edges adjacent to `l_sep` at its vertex,
    /// then separates the resulting disjoint fans into a fresh vertex.
    ///
    /// Returns `Some(new_vertex)` (the freshly created vertex carrying the
    /// peeled corner) or `Some(v_sep)` if no separation was needed (the loop
    /// was already isolated at its vertex).
    pub fn euler_unglue_region_make_vert(&mut self, l_sep: LoopKey) -> Option<VertexKey> {
        let v_sep = self.loops.get(l_sep).vertex;

        // For each of the two edges at l_sep's vertex (l_sep.edge and
        // l_sep.prev.edge), if it's an interior edge (radial > 1), separate
        // l_sep / l_sep.prev off onto a duplicate. Both loops have non-null
        // edges on a well-formed mesh; the is_null guard is defensive.
        let e_next = self.loops.get(l_sep).edge;
        if !e_next.is_null() && self.edge_radial_iter(e_next).count() > 1 {
            self.euler_edge_separate(e_next, l_sep);
        }
        let l_prev = self.loops.get(l_sep).face_loop_links.prev;
        let e_prev = self.loops.get(l_prev).edge;
        if !e_prev.is_null() && self.edge_radial_iter(e_prev).count() > 1 {
            self.euler_edge_separate(e_prev, l_prev);
        }

        // After the separates, l_sep's two edges are no longer shared with
        // other faces. The vertex v_sep may now have multiple disjoint fans
        // — run vert_separate to break them apart.
        let result = self.euler_vert_separate(v_sep);
        // vert_separate's contract: result[0] = v_sep, result[1..] are new,
        // and the last fan stays anchored on v_sep. We return the LAST
        // element, which is the freshly created vertex carrying the peeled
        // corner when len > 1, or v_sep otherwise.
        Some(*result.last().unwrap())
    }
}
