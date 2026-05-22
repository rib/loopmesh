use crate::loopmesh::*;

impl LoopMesh {
    /// Reverses the winding of `face`'s loop cycle. Loop vertices stay
    /// fixed; instead each loop's `edge` is rewritten to the edge that was
    /// previously held by the *preceding* loop in the original cycle, and
    /// `next`/`prev` pointers are swapped. The face's `first_edge_loop`
    /// pointer is unchanged.
    pub fn euler_loop_reverse(&mut self, face: FaceKey) {
        let first = self.faces.get(face).first_edge_loop;
        let loops: Vec<LoopKey> = self.face_loop_iter(first).collect();
        let n = loops.len();
        if n < 2 {
            return;
        }

        // Capture each loop's original edge before we start re-wiring radial cycles.
        let old_edges: Vec<EdgeKey> = loops.iter().map(|&l| self.loops.get(l).edge).collect();

        // Detach every loop from its current edge's radial cycle. We then
        // reconnect each loop l_i to the edge that used to belong to l_{i-1}.
        for &l in &loops {
            self.face_loop_disconnect_edge(l);
        }
        for (i, &l) in loops.iter().enumerate() {
            let prev_i = (i + n - 1) % n;
            let new_edge = old_edges[prev_i];
            if !new_edge.is_null() {
                self.face_loop_connect_edge(new_edge, l);
            }
        }

        // Finally swap face-loop next/prev for every loop so traversal walks
        // the cycle in the reverse direction.
        for &l in &loops {
            let old = self.loops.get(l).face_loop_links;
            self.loops.get_mut(l).face_loop_links = DoubleLinks {
                next: old.prev,
                prev: old.next,
            };
        }
    }
}

#[cfg(test)]
mod tests {
    use glam::Vec3;

    use crate::loopmesh::test_support::create_face_from_verts;
    use crate::loopmesh::*;

    /// euler_loop_reverse should flip face winding while keeping vertex set
    /// and edge set unchanged.
    #[test]
    fn test_euler_loop_reverse_quad() {
        let mut mesh = LoopMesh::new();
        let verts = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::new(2.0, 1.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        ];
        let face = create_face_from_verts(&mut mesh, &verts);
        let before: Vec<VertexKey> = mesh.face_vert_keys(face);
        let edges_before: std::collections::HashSet<EdgeKey> =
            mesh.face_edge_keys(face).into_iter().collect();

        mesh.euler_loop_reverse(face);

        let after: Vec<VertexKey> = mesh.face_vert_keys(face);
        let edges_after: std::collections::HashSet<EdgeKey> =
            mesh.face_edge_keys(face).into_iter().collect();

        assert_eq!(after.len(), before.len());
        assert_eq!(edges_before, edges_after, "edge set must be preserved");

        // Same starting vertex (first_edge_loop unchanged), then traversal in
        // reverse direction:  before = [a, b, c, d] -> after = [a, d, c, b].
        assert_eq!(after[0], before[0]);
        for i in 1..before.len() {
            assert_eq!(after[i], before[before.len() - i], "loop position {i}");
        }

        // Each edge's radial cycle still references the face (we didn't drop
        // it from the radial during the reattach).
        for e in mesh.face_edge_keys(face) {
            let radial_faces = mesh.edge_face_keys(e);
            assert_eq!(radial_faces.len(), 1);
            assert_eq!(radial_faces[0], face);
        }

        // Reversing twice should restore the original sequence.
        mesh.euler_loop_reverse(face);
        let restored = mesh.face_vert_keys(face);
        assert_eq!(restored, before);
    }
}
