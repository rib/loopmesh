use crate::loopmesh::*;

impl LoopMesh {
    /// Merges all radial loops from `e_src` onto `e_dst`, then kills `e_src`.
    /// Both edges must connect the same two vertices.
    ///
    /// Returns `false` if the edges don't share endpoints (the precondition
    /// the caller is expected to enforce).
    pub fn edge_splice(&mut self, e_dst: EdgeKey, e_src: EdgeKey) -> bool {
        if e_dst == e_src {
            return false;
        }
        let dst_v = self.edges.get(e_dst).vertices;
        if self.edges.get(e_src).contains(dst_v[0]).is_none()
            || self.edges.get(e_src).contains(dst_v[1]).is_none()
        {
            return false;
        }

        // §8.7: repeatedly take e_src's *entry* loop, remove it, and append it
        // to e_dst. Because removing the entry re-seats the entry forward onto
        // its successor (§6.5), the loops come off in e_src's §6.2 traversal
        // order — no snapshot needed, since each loop lands on a different edge
        // and e_src's cycle only ever shrinks. e_dst's post-splice entry is
        // therefore the loop moved last.
        while let Some(l) = self.edge_radial_entry(e_src) {
            self.face_loop_disconnect_edge(l);
            self.face_loop_connect_edge(e_dst, l);
        }

        // e_src is now an isolated edge — edge_kill simply removes it from the
        // two disk cycles and frees the slab entry (no radial recursion).
        self.edge_kill(e_src);
        true
    }
}

#[cfg(test)]
mod tests {
    use glam::Vec3;

    use crate::loopmesh::*;

    /// edge_splice rejects edges that don't share endpoints.
    #[test]
    fn test_edge_splice_rejects_non_parallel() {
        let mut mesh = LoopMesh::new();
        let v0 = mesh.create_vertex(Vec3::new(0.0, 0.0, 0.0));
        let v1 = mesh.create_vertex(Vec3::new(1.0, 0.0, 0.0));
        let v2 = mesh.create_vertex(Vec3::new(0.0, 1.0, 0.0));
        let e01 = mesh.create_edge([v0, v1], EdgeCreateFlags::NONE);
        let e02 = mesh.create_edge([v0, v2], EdgeCreateFlags::NONE);
        assert!(!mesh.edge_splice(e01, e02));
    }
}
