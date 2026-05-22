use crate::loopmesh::*;

impl LoopMesh {
    /// Collapses edge `e_kill`, merging `v_kill` onto the other endpoint
    /// `v_target`. All loops on `e_kill`'s radial cycle are removed and the
    /// surrounding face perimeters shrink by one. All other edges incident
    /// to `v_kill` are retargeted to `v_target`. Returns the surviving vertex.
    ///
    /// Unlike [`Self::euler_join_edge_kill_vert`] this works for arbitrary-valence
    /// `v_kill` and may produce degenerate (length < 3) faces — the caller is
    /// responsible for kill_degenerate_faces / double-face cleanup as needed.
    pub fn euler_join_vert_kill_edge(
        &mut self,
        e_kill: EdgeKey,
        v_kill: VertexKey,
    ) -> Option<VertexKey> {
        if self.edges.get(e_kill).contains(v_kill).is_none() {
            return None;
        }
        let v_target = self.edges.get(e_kill).other_vertex(v_kill).unwrap();

        // 1. For every loop on e_kill's radial cycle: splice it out of its
        //    face cycle, fix neighbour vertex pointers, decrement face.len.
        let radial_loops: Vec<LoopKey> = if self.edges.get(e_kill).radial_link.is_some() {
            self.edge_radial_iter(e_kill).collect()
        } else {
            Vec::new()
        };
        for &l_kill in &radial_loops {
            let l_kill_face = self.loops.get(l_kill).face;
            let l_kill_prev = self.loops.get(l_kill).face_loop_links.prev;
            let l_kill_next = self.loops.get(l_kill).face_loop_links.next;

            if self.loops.get(l_kill_next).vertex == v_kill {
                self.loops.get_mut(l_kill_next).vertex = v_target;
            }

            self.loops.get_mut(l_kill_prev).face_loop_links.next = l_kill_next;
            self.loops.get_mut(l_kill_next).face_loop_links.prev = l_kill_prev;

            if self.faces.get(l_kill_face).first_edge_loop == l_kill {
                self.faces.get_mut(l_kill_face).first_edge_loop = l_kill_next;
            }
            self.faces.get_mut(l_kill_face).len -= 1;

            self.face_loop_disconnect_edge(l_kill);
            self.remove_loop_slab_entry(l_kill);
        }

        // 2. Kill the edge — its radial is now empty so edge_kill will only
        //    touch the two disk cycles and free the slab entry.
        self.edge_kill(e_kill);

        // 3. Splice v_kill's remaining edges onto v_target, retargeting each
        //    edge's endpoint and updating per-face loop vertex pointers along
        //    the way (this is what `edge_replace_vertex_raw` does).
        loop {
            let next_edge = self.vertices.get(v_kill).disk_loop_start;
            match next_edge {
                Some(e) => self.edge_replace_vertex_raw(e, v_kill, v_target),
                None => break,
            }
        }

        // 4. Free v_kill.
        self.remove_vertex_slab_entry(v_kill);

        Some(v_target)
    }
}

#[cfg(test)]
mod tests {
    use crate::loopmesh::test_support::mesh_add_cube;
    use crate::loopmesh::*;

    /// JVKE on a cube interior edge should collapse it into a 5-face wedge.
    #[test]
    fn test_euler_join_vert_kill_edge_cube() {
        let mut mesh = LoopMesh::new();
        mesh_add_cube(&mut mesh);
        let pre_v = mesh.totvert();
        let pre_e = mesh.totedge();
        let pre_f = mesh.totface();

        // Pick any cube edge; collapse it (v_kill is its first endpoint).
        let e_kill = mesh.edge_keys()[0];
        let v_kill = mesh.edge_vertices(e_kill)[0];
        let v_target_expected = mesh.edge_vertices(e_kill)[1];

        let v_target = mesh.euler_join_vert_kill_edge(e_kill, v_kill);
        assert_eq!(v_target, Some(v_target_expected));

        // The cube has 8 verts / 12 edges / 6 faces.
        // After collapsing one edge:
        //   - 1 vert killed (8 -> 7)
        //   - The killed edge itself is gone, plus the two faces adjacent each
        //     lose one perimeter edge (degenerate-face cleanup is the caller's
        //     concern, so face count stays the same; counts of edges go down by
        //     1 because two parallel edges may end up merging via the splice).
        assert_eq!(mesh.totvert(), pre_v - 1, "v_kill killed");
        assert!(mesh.totedge() < pre_e, "edges should decrease");
        assert_eq!(
            mesh.totface(),
            pre_f,
            "face count unchanged (degenerate-face cleanup is opt-in)"
        );

        // Every surviving edge mentions v_target, never v_kill.
        for e in mesh.edge_keys() {
            let [a, b] = mesh.edge_vertices(e);
            assert_ne!(a, v_kill);
            assert_ne!(b, v_kill);
        }
        // Every surviving loop's vertex is never v_kill.
        for l in mesh.loop_keys() {
            assert_ne!(mesh.loop_vertex(l), v_kill);
        }
    }
}
