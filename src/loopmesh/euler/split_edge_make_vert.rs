use crate::loopmesh::*;

impl LoopMesh {
    /// Splits `edge` at a new vertex co-located with `tail_vertex`. Returns
    /// `(new_vertex, new_edge)`. After the split:
    ///
    /// - `edge` connects `v_old` (the original "other" endpoint) and `new_vertex`
    /// - `new_edge` connects `new_vertex` and `tail_vertex`
    /// - every face that referenced `edge` gains an extra loop at `new_vertex`,
    ///   correctly distributed across `edge` and `new_edge`
    pub fn euler_split_edge_make_vert(
        &mut self,
        tail_vertex: VertexKey,
        edge: EdgeKey,
    ) -> (VertexKey, EdgeKey) {
        debug_assert!(
            self.edges.get(edge).contains(tail_vertex).is_some(),
            "tail_vertex must belong to the edge being split"
        );

        let v_old = self
            .edges
            .get(edge)
            .other_vertex(tail_vertex)
            .expect("edge must have two distinct endpoints");
        let tv_pos = self.vertices.get(tail_vertex).position;

        // 1) Create the new vertex.
        let v_new = self.create_vertex(tv_pos);

        // 2) Create the new edge tail_vertex <-> v_new. create_edge appends
        //    it to both endpoints' disk cycles — exactly the final state we
        //    want for e_new, so no further disk manipulation is needed for it.
        let e_new = self.create_edge([tail_vertex, v_new], EdgeCreateFlags::NONE);

        // 3) Migrate `edge`'s endpoint from tail_vertex to v_new:
        //      a) remove edge from tail_vertex's disk
        //      b) overwrite the endpoint slot
        //      c) append edge to v_new's disk
        self.vertex_disk_remove_edge(tail_vertex, edge);
        let slot = if self.edges.get(edge).vertices[0] == tail_vertex {
            0
        } else {
            1
        };
        self.edges.get_mut(edge).vertices[slot] = v_new;
        self.vertex_disk_append_edge(v_new, edge);

        // 4) Split the radial cycle. Snapshot the loops first to make the
        //    walk independent of in-place radial mutations below.
        let original_loops: Vec<LoopKey> = if self.edges.get(edge).radial_link.is_some() {
            self.edge_radial_iter(edge).collect()
        } else {
            Vec::new()
        };

        // Detach every loop from `edge`'s radial cycle in one shot — the
        // per-loop reattach below decides which of (edge, e_new) it belongs to.
        self.edges.get_mut(edge).radial_link = None;
        for &l in &original_loops {
            let lref = self.loops.get_mut(l);
            // Transient null-edge state — the per-loop reattach below
            // assigns each one to either `edge` (re-anchored) or `e_new`.
            // The invariant "every reachable loop has a non-null edge"
            // is restored before SEMV returns.
            lref.edge = EdgeKey::NULL;
            lref.edge_radial_links = DoubleLinks { next: l, prev: l };
        }

        for &l in &original_loops {
            let l_face = self.loops.get(l).face;
            let l_next = self.loops.get(l).face_loop_links.next;
            let l_vertex = self.loops.get(l).vertex;

            // Bump face perimeter length — we're inserting one new loop.
            self.faces.get_mut(l_face).len += 1;

            // New loop sits at v_new, spliced between l and l_next:
            //     ... <-> l <-> l_new <-> l_next <-> ...
            let l_new = self.create_loop(v_new, l_face);
            self.loops.get_mut(l_new).face_loop_links.prev = l;
            self.loops.get_mut(l_new).face_loop_links.next = l_next;
            self.loops.get_mut(l).face_loop_links.next = l_new;
            self.loops.get_mut(l_next).face_loop_links.prev = l_new;

            // Each loop in our cycle covers (loop.vertex -> loop.next.vertex).
            // After the splice:
            //   l covers (l_vertex -> v_new)               via either `edge` or `e_new`
            //   l_new covers (v_new -> l_next.vertex)      via the other edge
            // `edge` now connects v_new <-> v_old; `e_new` connects v_new <-> tail_vertex.
            let (l_edge, l_new_edge) = if l_vertex == v_old {
                (edge, e_new)
            } else if l_vertex == tail_vertex {
                (e_new, edge)
            } else {
                panic!(
                    "loop vertex {} is not an endpoint of split edge ({} <-> {})",
                    l_vertex, v_old, tail_vertex
                );
            };
            self.face_loop_connect_edge(l_edge, l);
            self.face_loop_connect_edge(l_new_edge, l_new);
        }

        (v_new, e_new)
    }
}

#[cfg(test)]
mod tests {
    use glam::Vec3;

    use crate::loopmesh::test_support::create_face_from_verts;
    use crate::loopmesh::*;

    /// Sanity test for euler_split_edge_make_vert against a quad face.
    #[test]
    fn test_euler_split_edge_make_vert_quad() {
        let mut mesh = LoopMesh::new();

        let verts = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::new(2.0, 1.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        ];
        let face = create_face_from_verts(&mut mesh, &verts);
        let original_face_verts = mesh.face_vert_keys(face);
        assert_eq!(original_face_verts.len(), 4);

        // Pick the edge that runs between vert[0] and vert[1] in the face cycle.
        let v0 = original_face_verts[0];
        let v1 = original_face_verts[1];
        let edge = mesh
            .query_edge_exists(v0, v1)
            .expect("quad must have v0-v1 edge");

        // Split at v1 (i.e. tail_vertex = v1, so new vertex sits between v0 and v1
        // along the original edge, leaving `edge` connecting v0 <-> v_new and
        // a new edge connecting v_new <-> v1).
        let pre_totvert = mesh.totvert();
        let pre_totedge = mesh.totedge();
        let pre_totface = mesh.totface();

        let (v_new, e_new) = mesh.euler_split_edge_make_vert(v1, edge);

        // Topology checks.
        assert_eq!(mesh.totvert(), pre_totvert + 1, "one vertex added");
        assert_eq!(mesh.totedge(), pre_totedge + 1, "one edge added");
        assert_eq!(mesh.totface(), pre_totface, "no face change");

        // The face should now have 5 verts.
        let new_face_verts = mesh.face_vert_keys(face);
        assert_eq!(new_face_verts.len(), 5, "quad becomes pentagon");
        assert_eq!(mesh.face_len(face), 5);

        // v_new must appear exactly once in the new face cycle, sandwiched
        // between v0 and v1 in the original winding direction.
        let new_pos = new_face_verts
            .iter()
            .position(|&v| v == v_new)
            .expect("new vertex must appear in the face cycle");
        let prev_pos = (new_pos + new_face_verts.len() - 1) % new_face_verts.len();
        let next_pos = (new_pos + 1) % new_face_verts.len();
        assert_eq!(new_face_verts[prev_pos], v0);
        assert_eq!(new_face_verts[next_pos], v1);

        // Disk validation: v_new sits on exactly 2 edges (edge + e_new).
        assert_eq!(mesh.vertex_disk_count_edges(v_new), 2);
        let v_new_edges: Vec<EdgeKey> = mesh.vertex_disk_iter(v_new).collect();
        assert!(v_new_edges.contains(&edge));
        assert!(v_new_edges.contains(&e_new));

        // Edge endpoints: `edge` should now span v0 <-> v_new, and `e_new` v_new <-> v1.
        let edge_verts = mesh.edge_vertices(edge);
        assert!(
            (edge_verts == [v0, v_new]) || (edge_verts == [v_new, v0]),
            "edge endpoints after SEMV should be v0<->v_new, got {:?}",
            edge_verts
        );
        let e_new_verts = mesh.edge_vertices(e_new);
        assert!(
            (e_new_verts == [v_new, v1]) || (e_new_verts == [v1, v_new]),
            "e_new endpoints should be v_new<->v1, got {:?}",
            e_new_verts
        );

        // Each edge in the new pentagon has the face on its radial.
        for f_edge in mesh.face_edge_keys(face) {
            let radial: Vec<FaceKey> = mesh.edge_face_keys(f_edge);
            assert_eq!(radial.len(), 1);
            assert_eq!(radial[0], face);
        }
    }

    /// Splitting the same edge twice should give the same topology shape (hexagon).
    #[test]
    fn test_euler_split_edge_make_vert_twice() {
        let mut mesh = LoopMesh::new();
        let verts = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::new(2.0, 1.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        ];
        let face = create_face_from_verts(&mut mesh, &verts);
        let face_verts0 = mesh.face_vert_keys(face);
        let v0 = face_verts0[0];
        let v1 = face_verts0[1];
        let edge = mesh.query_edge_exists(v0, v1).unwrap();

        let (v_mid1, _e1) = mesh.euler_split_edge_make_vert(v1, edge);
        // After first split, `edge` now goes v0 <-> v_mid1. Split that again.
        let (_v_mid2, _e2) = mesh.euler_split_edge_make_vert(v_mid1, edge);

        assert_eq!(mesh.totvert(), 6);
        assert_eq!(mesh.totedge(), 6);
        assert_eq!(mesh.face_len(face), 6);
    }
}
