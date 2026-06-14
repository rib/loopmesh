use crate::loopmesh::*;

impl LoopMesh {
    /// Collapses degree-2 `v_kill` by killing `e_kill` and merging it into
    /// the only other edge incident at `v_kill`. Returns `Some(surviving_edge)`,
    /// equal to that other edge, on success. The vertex itself is freed.
    ///
    /// Preconditions: `v_kill` is incident to `e_kill`, and `v_kill`'s disk
    /// contains exactly two edges. Returns `None` if either fails or if the
    /// collapse would create a double-edge.
    pub fn euler_join_edge_kill_vert(
        &mut self,
        e_kill: EdgeKey,
        v_kill: VertexKey,
    ) -> Option<EdgeKey> {
        if self.edges.get(e_kill).contains(v_kill).is_none() {
            return None;
        }
        if self.vertex_disk_count_edges_capped(v_kill, 3) != 2 {
            return None;
        }

        // The other edge at v_kill.
        let e_old = self
            .vertex_disk_iter(v_kill)
            .find(|&e| e != e_kill)
            .expect("disk count is 2 but no other edge found");

        let v_target = self.edges.get(e_kill).other_vertex(v_kill).unwrap();
        let v_old = self.edges.get(e_old).other_vertex(v_kill).unwrap();

        // Refuse if e_old already touches v_target — collapsing would
        // produce a duplicate edge.
        if self.edges.get(e_old).contains(v_target).is_some() {
            return None;
        }

        // Migrate e_old's endpoint from v_kill to v_target.
        self.vertex_disk_remove_edge(v_kill, e_old);
        let slot_old = if self.edges.get(e_old).vertices[0] == v_kill {
            0
        } else {
            1
        };
        self.edges.get_mut(e_old).vertices[slot_old] = v_target;
        self.vertex_disk_append_edge(v_target, e_old);

        // Remove e_kill from v_target's disk (v_kill's disk is wiped along with the vertex).
        self.vertex_disk_remove_edge(v_target, e_kill);

        // Snapshot loops on e_kill's radial cycle before mutating it.
        let radial_loops: Vec<LoopKey> = if self.edges.get(e_kill).radial_link.is_some() {
            self.edge_radial_iter(e_kill).collect()
        } else {
            Vec::new()
        };

        for &l_kill in &radial_loops {
            let l_kill_face = self.loops.get(l_kill).face;
            let l_kill_prev = self.loops.get(l_kill).face_loop_links.prev;
            let l_kill_next = self.loops.get(l_kill).face_loop_links.next;

            // If l_kill_next pointed at v_kill (Case A: l_kill.vertex was the v_target
            // side, and the loop went v_target -> v_kill -> v_old), redirect to v_target.
            // (Case B is handled implicitly via the e_old endpoint mutation above.)
            if self.loops.get(l_kill_next).vertex == v_kill {
                self.loops.get_mut(l_kill_next).vertex = v_target;
            }

            // Splice l_kill out of its face cycle.
            self.loops.get_mut(l_kill_prev).face_loop_links.next = l_kill_next;
            self.loops.get_mut(l_kill_next).face_loop_links.prev = l_kill_prev;

            if self.faces.get(l_kill_face).first_edge_loop == l_kill {
                self.faces.get_mut(l_kill_face).first_edge_loop = l_kill_next;
            }
            self.faces.get_mut(l_kill_face).len -= 1;

            // Remove l_kill from e_kill's radial cycle (book-keeping; the edge
            // is about to be freed) and drop it from the slab.
            self.face_loop_disconnect_edge(l_kill);
            self.remove_loop_slab_entry(l_kill);
        }

        // Free the killed edge and vertex.
        self.remove_edge_slab_entry(e_kill);
        self.remove_vertex_slab_entry(v_kill);

        let _ = v_old; // captured for clarity above; not needed beyond debug intuition

        Some(e_old)
    }
}

#[cfg(test)]
mod tests {
    use glam::Vec3;

    use crate::loopmesh::test_support::{create_face_from_verts, mesh_add_cube};
    use crate::loopmesh::*;

    /// SEMV followed by JEKV on the new vertex should restore the original mesh.
    #[test]
    fn test_euler_split_then_join_edge_kill_vert() {
        let mut mesh = LoopMesh::new();
        let verts = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::new(2.0, 1.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        ];
        let face = create_face_from_verts(&mut mesh, &verts);
        let pre_v = mesh.vertex_count();
        let pre_e = mesh.edge_count();
        let pre_f = mesh.face_count();
        let original_face_verts = mesh.face_vert_keys(face);
        let v0 = original_face_verts[0];
        let v1 = original_face_verts[1];
        let edge = mesh.query_edge_exists(v0, v1).unwrap();

        let (v_new, e_new) = mesh.euler_split_edge_make_vert(v1, edge);
        assert_eq!(mesh.vertex_count(), pre_v + 1);
        assert_eq!(mesh.edge_count(), pre_e + 1);
        assert_eq!(mesh.face_len(face), 5);

        // Collapse the new degree-2 vertex.
        let restored = mesh.euler_join_edge_kill_vert(e_new, v_new);
        assert!(
            restored.is_some(),
            "JEKV should succeed on a degree-2 vertex"
        );

        assert_eq!(mesh.vertex_count(), pre_v);
        assert_eq!(mesh.edge_count(), pre_e);
        assert_eq!(mesh.face_count(), pre_f);
        assert_eq!(mesh.face_len(face), 4);

        let after_verts: std::collections::HashSet<VertexKey> =
            mesh.face_vert_keys(face).into_iter().collect();
        let orig_set: std::collections::HashSet<VertexKey> =
            original_face_verts.into_iter().collect();
        assert_eq!(after_verts, orig_set);
    }

    /// JEKV refuses to collapse a vertex that doesn't have valence 2.
    #[test]
    fn test_euler_join_edge_kill_vert_rejects_non_degree2() {
        let mut mesh = LoopMesh::new();
        mesh_add_cube(&mut mesh);
        // Cube vertices all have valence 3 → JEKV should return None.
        let v0 = mesh.vert_keys()[0];
        let first_edge = mesh.vert_edge_keys(v0)[0];
        let result = mesh.euler_join_edge_kill_vert(first_edge, v0);
        assert!(result.is_none());
    }
}
