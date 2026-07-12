use crate::loopmesh::*;

impl LoopMesh {
    /// Joins two faces that share `edge` into a single face by killing the
    /// shared edge and the two loops along it. Returns `Some(surviving_face)`
    /// — equal to `f1` — on success, or `None` if preconditions fail.
    ///
    /// Preconditions: `f1 != f2`, `edge` is 2-manifold (radial length 2),
    /// both faces share exactly one edge (`edge`), and the shared edge's
    /// loops have opposite winding directions (the normal manifold case).
    ///
    /// **This is the raw Euler join: a pure re-splice.** It appends no fresh
    /// loop to any surviving edge — `f2`'s loops are re-stamped onto `f1` *in
    /// place* — so **no perimeter edge's radial entry moves**, in pointer or in
    /// cyclic position (`RADIAL_EDGE_STRUCTURE.md` §8.4, and §6.5's
    /// entry-motion discriminant: entries are never *set*, only appended-onto
    /// and re-seated-forward).
    ///
    /// Callers that want the *composed* face join — the one that constructs the
    /// union and therefore does re-seat every perimeter entry onto it — must
    /// call [`LoopMesh::faces_join_pair`] / [`LoopMesh::faces_join`] instead.
    /// The two primitives reach the same topology with **different** radial
    /// entries, observable at radial ≥ 3, so the choice is not an
    /// implementation detail (`RADIAL_EDGE_STRUCTURE.md` §8.4, §9).
    pub fn euler_join_face_kill_edge(
        &mut self,
        f1: FaceKey,
        f2: FaceKey,
        edge: EdgeKey,
    ) -> Option<FaceKey> {
        if f1 == f2 {
            return None;
        }

        let l_f1 = self.face_edge_share_loop(f1, edge)?;
        let l_f2 = self.face_edge_share_loop(f2, edge)?;

        // Radial cycle of the shared edge must contain exactly the two loops on f1 and f2.
        let radial_len = self.edge_radial_iter(edge).count();
        if radial_len != 2 {
            return None;
        }

        // The two loops must walk the shared edge in opposite directions:
        // l_f1.vertex != l_f2.vertex (one starts at v1->v2 while the other
        // starts at v2->v1).
        if self.loops.get(l_f1).vertex == self.loops.get(l_f2).vertex {
            return None;
        }

        // Reject "double" cases: don't allow merging if the result would
        // contain a duplicate vertex (excluding the two endpoints of `edge`).
        let f1_verts: Vec<VertexKey> = {
            let face_loops_first = self.faces.get(f1).first_edge_loop;
            self.face_loop_iter(face_loops_first)
                .filter(|&l| l != l_f1)
                .map(|l| self.loops.get(l).vertex)
                .collect()
        };
        let f2_verts: Vec<VertexKey> = {
            let face_loops_first = self.faces.get(f2).first_edge_loop;
            self.face_loop_iter(face_loops_first)
                .filter(|&l| l != l_f2)
                .map(|l| self.loops.get(l).vertex)
                .collect()
        };
        let mut shared_excluding_edge = 0usize;
        for v in &f2_verts {
            if f1_verts.contains(v) {
                shared_excluding_edge += 1;
            }
        }
        // Two endpoints of `edge` are shared by definition (one each from l_f1.next.vertex
        // and l_f1.vertex's neighbour on the other face). Anything beyond that means an
        // internal duplicate — refuse.
        if shared_excluding_edge > 2 {
            return None;
        }

        // ---- Splice the face cycles. ----
        let l_f1_prev = self.loops.get(l_f1).face_loop_links.prev;
        let l_f1_next = self.loops.get(l_f1).face_loop_links.next;
        let l_f2_prev = self.loops.get(l_f2).face_loop_links.prev;
        let l_f2_next = self.loops.get(l_f2).face_loop_links.next;

        self.loops.get_mut(l_f1_prev).face_loop_links.next = l_f2_next;
        self.loops.get_mut(l_f2_next).face_loop_links.prev = l_f1_prev;
        self.loops.get_mut(l_f1_next).face_loop_links.prev = l_f2_prev;
        self.loops.get_mut(l_f2_prev).face_loop_links.next = l_f1_next;

        // If f1's first_edge_loop was l_f1 (about to be killed), advance it.
        if self.faces.get(f1).first_edge_loop == l_f1 {
            self.faces.get_mut(f1).first_edge_loop = l_f1_next;
        }

        // ---- Remove l_f1 and l_f2 from edge's radial cycle (the edge will
        //      be killed below, but we still want consistent intermediate state). ----
        self.face_loop_disconnect_edge(l_f1);
        self.face_loop_disconnect_edge(l_f2);

        // ---- Re-stamp face pointers across the merged cycle. ----
        let new_first = self.faces.get(f1).first_edge_loop;
        let merged_loops: Vec<LoopKey> = self.face_loop_iter(new_first).collect();
        for &l in &merged_loops {
            self.loops.get_mut(l).face = f1;
        }
        self.faces.get_mut(f1).len = merged_loops.len() as u32;

        // ---- Free the two doomed loops, the shared edge and f2. ----
        let l_f1_key = l_f1;
        let l_f2_key = l_f2;
        self.remove_loop_slab_entry(l_f1_key);
        self.remove_loop_slab_entry(l_f2_key);

        // Remove the edge from both endpoints' disk cycles and free its slab entry.
        let edge_verts = self.edges.get(edge).vertices;
        self.vertex_disk_remove_edge(edge_verts[0], edge);
        self.vertex_disk_remove_edge(edge_verts[1], edge);
        self.remove_edge_slab_entry(edge);

        // Free f2's slab entry. (f2's loops have already been re-pointed to f1.)
        self.remove_face_slab_entry(f2);

        Some(f1)
    }
}

#[cfg(test)]
mod tests {
    use glam::Vec3;

    use crate::loopmesh::test_support::create_face_from_verts;
    use crate::loopmesh::*;

    /// SFME then JFKE should restore the original topology of a quad.
    #[test]
    fn test_euler_split_then_join_quad_diagonal() {
        let mut mesh = LoopMesh::new();
        let verts = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::new(2.0, 1.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        ];
        let face = create_face_from_verts(&mut mesh, &verts);
        let verts_orig: std::collections::HashSet<VertexKey> =
            mesh.face_vert_keys(face).into_iter().collect();
        let edges_orig: std::collections::HashSet<EdgeKey> =
            mesh.face_edge_keys(face).into_iter().collect();
        let v0 = mesh.face_vert_keys(face)[0];
        let v2 = mesh.face_vert_keys(face)[2];
        let face_loops = mesh.face_loop_keys(face);
        let l_v0 = *face_loops
            .iter()
            .find(|&&l| mesh.loop_vertex(l) == v0)
            .unwrap();
        let l_v2 = *face_loops
            .iter()
            .find(|&&l| mesh.loop_vertex(l) == v2)
            .unwrap();

        let (face2, new_edge, _l_new) = mesh.euler_split_face_make_edge(face, l_v0, l_v2);
        assert_eq!(mesh.face_count(), 2);

        let joined = mesh.euler_join_face_kill_edge(face, face2, new_edge);
        assert_eq!(joined, Some(face), "join should return the surviving face");

        assert_eq!(mesh.face_count(), 1);
        assert_eq!(mesh.edge_count(), 4, "split edge should have been killed");
        assert_eq!(mesh.face_len(face), 4, "restored quad");

        let verts_after: std::collections::HashSet<VertexKey> =
            mesh.face_vert_keys(face).into_iter().collect();
        let edges_after: std::collections::HashSet<EdgeKey> =
            mesh.face_edge_keys(face).into_iter().collect();
        assert_eq!(verts_after, verts_orig);
        assert_eq!(edges_after, edges_orig);

        // Each surviving edge should have face on its radial.
        for e in &edges_after {
            let radial = mesh.edge_face_keys(*e);
            assert_eq!(radial.len(), 1);
            assert_eq!(radial[0], face);
        }
    }

    /// The raw Euler join is *loop-preserving*: it appends no fresh loop to any
    /// surviving edge, so no perimeter entry moves
    /// (`RADIAL_EDGE_STRUCTURE.md` §8.4, §6.5's "there is no third rule").
    ///
    /// The discriminating edge is a perimeter edge shared with a face *outside*
    /// the join: its pristine entry names that outside face, and the join must
    /// leave it there. (The *composed* join —
    /// [`LoopMesh::faces_join_pair`] — is the one that seats the union.)
    #[test]
    fn test_join_face_kill_edge_moves_no_perimeter_radial_entry() {
        let mut mesh = LoopMesh::new();
        let strip = build_quad_strip(&mut mesh, 3);
        let seam_ab = shared_seam(&mesh, strip[0], strip[1]);
        let seam_bc = shared_seam(&mesh, strip[1], strip[2]);

        assert_eq!(
            mesh.loop_face(mesh.edge_radial_entry(seam_bc).unwrap()),
            strip[2],
            "pristine entry is the last face attached to the seam"
        );

        mesh.euler_join_face_kill_edge(strip[0], strip[1], seam_ab)
            .expect("radial-2 seam joins");

        assert_eq!(
            mesh.loop_face(mesh.edge_radial_entry(seam_bc).unwrap()),
            strip[2],
            "the raw Euler join re-splices in place: the outward seam's entry is untouched"
        );
        mesh.validate_topology();
    }

    /// The radial-≥3 reading of the same rule, where the three candidate join
    /// flavours part company (`RADIAL_EDGE_STRUCTURE.md` §6.5's discriminating
    /// trace). The absorbed face's loop is re-stamped onto the survivor **in its
    /// slot**: the cycle's order and entry are bit-for-bit what they were, and
    /// only the loop's owning face changed beneath them.
    #[test]
    fn test_join_face_kill_edge_leaves_radial_three_cycle_in_place() {
        let mut mesh = LoopMesh::new();
        let (strip, seam, rim, fins) = build_strip_with_rim_fins(&mut mesh);

        assert_eq!(
            mesh.edge_face_keys(rim),
            vec![fins[1], strip[1], fins[0]],
            "pristine radial-3 cycle, walked from its entry (the last fin appended)"
        );

        let merged = mesh
            .euler_join_face_kill_edge(strip[0], strip[1], seam)
            .expect("radial-2 seam joins");

        assert_eq!(
            mesh.edge_face_keys(rim),
            vec![fins[1], merged, fins[0]],
            "no fresh loop reached `rim`, so nothing on it moved — the absorbed \
             face's loop simply now names the survivor"
        );
        mesh.validate_topology();
    }

    /// A quad strip of `quads` faces, with two fins attached to the far rim edge
    /// of the last quad so that rim sits at radial 3 while staying on the union's
    /// perimeter. Returns `(strip, seam(strip0,strip1), rim, fins)`.
    fn build_strip_with_rim_fins(
        mesh: &mut LoopMesh,
    ) -> (Vec<FaceKey>, EdgeKey, EdgeKey, Vec<FaceKey>) {
        let strip = build_quad_strip(mesh, 2);
        let seam = shared_seam(mesh, strip[0], strip[1]);

        let rim = mesh
            .face_edge_keys(strip[1])
            .into_iter()
            .find(|&e| {
                mesh.edge_face_keys(e).len() == 1
                    && mesh
                        .edge_vert_keys(e)
                        .iter()
                        .all(|&v| mesh.vertex_position(v).x == 2.0)
            })
            .expect("far rim edge of the second quad");
        let [rim_a, rim_b] = mesh.edge_vert_keys(rim);

        let fins: Vec<FaceKey> = [1.0_f32, 2.0]
            .into_iter()
            .map(|z| {
                let a = mesh.create_vertex(Vec3::new(3.0, 0.0, z));
                let b = mesh.create_vertex(Vec3::new(3.0, 1.0, z));
                let ring = [rim_a, a, b, rim_b];
                let edges: Vec<EdgeKey> = (0..4)
                    .map(|k| {
                        mesh.create_edge(
                            [ring[k], ring[(k + 1) % 4]],
                            EdgeCreateFlags::NO_DUPLICATE,
                        )
                    })
                    .collect();
                mesh.face_create(&ring, &edges, FaceCreateFlags::NONE)
            })
            .collect();

        (strip, seam, rim, fins)
    }

    /// The composed join's counterpart to the test above, kept here so the two
    /// flavours can be read side by side: routed through
    /// [`LoopMesh::faces_join_pair`] the same merge *does* seat the union on the
    /// radial-3 rim, where an appended fresh loop would land — immediately after
    /// the pre-merge entry, with the two fins keeping their relative order
    /// (`RADIAL_EDGE_STRUCTURE.md` §6.5 case (a), §9).
    #[test]
    fn test_faces_join_pair_seats_the_union_on_a_radial_three_perimeter_edge() {
        let mut mesh = LoopMesh::new();
        let (strip, seam, rim, fins) = build_strip_with_rim_fins(&mut mesh);

        let merged = mesh
            .faces_join_pair(strip[0], strip[1], seam)
            .expect("radial-2 seam joins");

        assert_eq!(
            mesh.edge_face_keys(rim),
            vec![merged, fins[0], fins[1]],
            "the union enters where an appended fresh loop would: at the entry, \
             behind the two fins in their unchanged relative order"
        );
        mesh.validate_topology();
    }

    fn build_quad_strip(mesh: &mut LoopMesh, quads: usize) -> Vec<FaceKey> {
        let bottom: Vec<VertexKey> = (0..=quads)
            .map(|i| mesh.create_vertex(Vec3::new(i as f32, 0.0, 0.0)))
            .collect();
        let top: Vec<VertexKey> = (0..=quads)
            .map(|i| mesh.create_vertex(Vec3::new(i as f32, 1.0, 0.0)))
            .collect();
        (0..quads)
            .map(|i| {
                let ring = [bottom[i], bottom[i + 1], top[i + 1], top[i]];
                let edges: Vec<EdgeKey> = (0..4)
                    .map(|k| {
                        mesh.create_edge(
                            [ring[k], ring[(k + 1) % 4]],
                            EdgeCreateFlags::NO_DUPLICATE,
                        )
                    })
                    .collect();
                mesh.face_create(&ring, &edges, FaceCreateFlags::NONE)
            })
            .collect()
    }

    fn shared_seam(mesh: &LoopMesh, a: FaceKey, b: FaceKey) -> EdgeKey {
        mesh.face_edge_keys(a)
            .into_iter()
            .find(|&e| mesh.edge_face_keys(e).contains(&b))
            .expect("the two faces share a seam")
    }
}
