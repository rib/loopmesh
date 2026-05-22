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

        // Move every loop currently on e_src's radial cycle to e_dst's.
        let src_loops: Vec<LoopKey> = if self.edges.get(e_src).radial_link.is_some() {
            self.edge_radial_iter(e_src).collect()
        } else {
            Vec::new()
        };
        for &l in &src_loops {
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
