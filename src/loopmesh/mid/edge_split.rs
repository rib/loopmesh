use smallvec::SmallVec;

use crate::loopmesh::*;

impl LoopMesh {
    /// Parameterized edge split. Splits `edge` at a vertex positioned at
    /// `lerp(near_vertex.pos, far_endpoint.pos, fac)`, where `fac == 0`
    /// places the new vertex on top of `near_vertex` and `fac == 1` on
    /// top of the far endpoint. Returns `(new_vertex, new_edge)` — same
    /// shape as the underlying SEMV call.
    pub fn edge_split(
        &mut self,
        edge: EdgeKey,
        near_vertex: VertexKey,
        fac: f32,
    ) -> (VertexKey, EdgeKey) {
        let far = self
            .edges
            .get(edge)
            .other_vertex(near_vertex)
            .expect("near_vertex must belong to edge");
        let p_near = self.vertex_position(near_vertex);
        let p_far = self.vertex_position(far);

        // Pre-SEMV loop-attrib snapshot. For each loop on `edge`'s radial,
        // record (l, l.face_loop_links.next, is_at_near_side). SEMV inserts
        // a new loop between l and l.next in the face cycle — that's the
        // loop we need to interpolate, sourced from the original `l` and
        // `l_next`. Skipped entirely if no loop attribs are registered
        // (zero-cost fast path for the common case).
        let do_loop_interp = !self.loop_attribs.is_empty();
        let mut loop_snap: SmallVec<[(LoopKey, LoopKey, bool); 4]> = SmallVec::new();
        if do_loop_interp {
            let mut it = self.radial_loops(edge);
            while let Some(l) = it.next(self) {
                let lref = self.loops.get(l);
                let l_next = lref.face_loop_links.next;
                let is_at_near = lref.vertex == near_vertex;
                loop_snap.push((l, l_next, is_at_near));
            }
        }

        let (v_new, e_new) = self.euler_split_edge_make_vert(near_vertex, edge);
        self.set_vertex_position(v_new, p_near.lerp(p_far, fac));

        // Customdata interp: v_new is the linear blend of near/far by `fac`;
        // e_new copies the split edge's data.
        let sources = [(1.0 - fac, near_vertex.as_usize()), (fac, far.as_usize())];
        self.vertex_attribs
            .interpolate_into(v_new.as_usize(), &sources);
        self.edge_attribs
            .copy_into(e_new.as_usize(), edge.as_usize());

        // Post-SEMV loop interp. SEMV inserts a new loop at v_new between
        // each snapshotted `l` and its `l_next`. The new loop is reachable
        // as `l.face_loop_links.next` after SEMV runs. When `l` sits at
        // near_vertex, the new loop is `fac` of the way from `l` toward
        // `l_next`, so weights are (1-fac, l), (fac, l_next). When `l` sits
        // at far, the weights reverse.
        if do_loop_interp {
            for (l, l_next, is_at_near) in loop_snap {
                let l_new = self.loops.get(l).face_loop_links.next;
                let (w_l, w_next) = if is_at_near {
                    (1.0 - fac, fac)
                } else {
                    (fac, 1.0 - fac)
                };
                let src = [(w_l, l.as_usize()), (w_next, l_next.as_usize())];
                self.loop_attribs.interpolate_into(l_new.as_usize(), &src);
            }
        }

        (v_new, e_new)
    }

    /// Uniformly subdivides `edge` into `numcuts + 1` segments, returning the
    /// `numcuts` new vertices in order from `near_vertex` toward the far
    /// endpoint of `edge`.
    ///
    /// `numcuts == 0` returns an empty Vec. `numcuts == 1` returns a single
    /// midpoint.
    pub fn edge_split_n(
        &mut self,
        edge: EdgeKey,
        near_vertex: VertexKey,
        numcuts: u32,
    ) -> Vec<VertexKey> {
        let mut out = Vec::with_capacity(numcuts as usize);
        // After each SEMV with `current_near` as tail: `edge` (= current_edge)
        // still spans (far, new_vertex), so we keep splitting it but re-anchor
        // the named near vertex to the freshly created vertex so subsequent
        // `fac` values continue to march toward `far`.
        let mut current_edge = edge;
        let mut current_near = near_vertex;
        for i in 0..numcuts {
            // Remaining segment from current_near→far has (numcuts - i + 1)
            // sub-segments; the next cut sits 1/(numcuts - i + 1) of the way.
            let fac = 1.0_f32 / (numcuts as f32 - i as f32 + 1.0);
            let (v_new, _e_new) = self.edge_split(current_edge, current_near, fac);
            out.push(v_new);
            current_near = v_new;
        }
        out
    }
}
