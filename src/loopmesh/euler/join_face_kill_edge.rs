use crate::loopmesh::*;

impl LoopMesh {
    /// Joins two faces that share `edge` into a single face by killing the
    /// shared edge and the two loops along it. Returns `Some(surviving_face)`
    /// — equal to `f1` — on success, or `None` if preconditions fail.
    ///
    /// Preconditions: `f1 != f2`, `edge` is 2-manifold (radial length 2),
    /// both faces share exactly one edge (`edge`), and the shared edge's
    /// loops have opposite winding directions (the normal manifold case).
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

    /// Seat `face` as the radial **entry** of every radial-≤2 edge on its
    /// perimeter — i.e. make it the *newest* face incident to those edges, the
    /// state a join that minted a fresh union face would have left behind
    /// (`RADIAL_EDGE_STRUCTURE.md` §6.2/§6.5's append rule, and the entry-reseat
    /// rule of `DISSOLVE_VERTS.md` §6.3). [`Self::euler_join_face_kill_edge`]
    /// preserves `f1`'s handle instead, so its survivor is *not* re-seated; a
    /// caller whose donor rule reads the entry as a face-age channel across
    /// successive joins calls this after each one.
    ///
    /// This is **not** folded into the join itself: doing so is only correct for
    /// callers that model the join as a fresh-face creation, and at least one
    /// pairwise-join consumer is pinned to the opposite reading (see the
    /// `dissolve_verts` row 52 notes in the parity dashboard). Applying it is
    /// the caller's decision, and belongs with the caller's donor rule.
    ///
    /// Scoped to radial-≤2 edges, the bound §6.3 attaches to the rule itself: on
    /// a longer cycle an append lands the new loop at a cyclic position a
    /// re-seat cannot reproduce, so those entries are left alone rather than
    /// half-modelled.
    pub(crate) fn face_reseat_perimeter_radial_entries(&mut self, face: FaceKey) {
        let first = self.faces.get(face).first_edge_loop;
        let perimeter: smallvec::SmallVec<[LoopKey; 8]> =
            self.face_loop_iter(first).collect::<smallvec::SmallVec<_>>();
        for l in perimeter {
            let perimeter_edge = self.loops.get(l).edge;
            let radial_next = self.loop_radial_next(l);
            if self.loop_radial_next(radial_next) != l {
                continue;
            }
            // Head-anchored storage: entry == head.prev, so seating the entry on
            // `l` means pointing the head at `l`'s radial successor.
            self.edges.get_mut(perimeter_edge).radial_link = Some(radial_next);
        }
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

    /// The join preserves `f1`'s handle, so it leaves the perimeter's radial
    /// entries where they were; `face_reseat_perimeter_radial_entries` is what
    /// makes the merged face the newest face on its own perimeter, the state a
    /// fresh-union-face join would have left (`RADIAL_EDGE_STRUCTURE.md`
    /// §6.2/§6.5, `DISSOLVE_VERTS.md` §6.3).
    ///
    /// The discriminating edge is a perimeter edge shared with a face *outside*
    /// the join: its pristine entry names that outside face, and only the
    /// re-seat moves it onto the merged one — which is what makes the entry a
    /// persistent age channel a later surgery can read.
    #[test]
    fn test_face_reseat_perimeter_radial_entries_ages_the_merged_face() {
        let mut mesh = LoopMesh::new();
        let bottom: Vec<VertexKey> = (0..4)
            .map(|i| mesh.create_vertex(Vec3::new(i as f32, 0.0, 0.0)))
            .collect();
        let top: Vec<VertexKey> = (0..4)
            .map(|i| mesh.create_vertex(Vec3::new(i as f32, 1.0, 0.0)))
            .collect();
        let strip: Vec<FaceKey> = (0..3)
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
            .collect();
        let seam_ab = mesh
            .face_edge_keys(strip[0])
            .into_iter()
            .find(|&e| mesh.edge_face_keys(e).contains(&strip[1]))
            .expect("f0 and f1 share a seam");
        let seam_bc = mesh
            .face_edge_keys(strip[1])
            .into_iter()
            .find(|&e| mesh.edge_face_keys(e).contains(&strip[2]))
            .expect("f1 and f2 share a seam");

        assert_eq!(
            mesh.loop_face(mesh.edge_radial_entry(seam_bc).unwrap()),
            strip[2],
            "pristine entry is the last face attached to the seam"
        );

        let merged = mesh
            .euler_join_face_kill_edge(strip[0], strip[1], seam_ab)
            .expect("radial-2 seam joins");

        assert_eq!(
            mesh.loop_face(mesh.edge_radial_entry(seam_bc).unwrap()),
            strip[2],
            "the bare join leaves the entry put — it reuses f1's handle"
        );

        mesh.face_reseat_perimeter_radial_entries(merged);

        assert_eq!(
            mesh.loop_face(mesh.edge_radial_entry(seam_bc).unwrap()),
            merged,
            "the merged face is now the newest face on the outward seam"
        );
        for l in mesh.face_loop_keys(merged) {
            let e = mesh.loop_edge(l);
            assert_eq!(
                mesh.loop_face(mesh.edge_radial_entry(e).unwrap()),
                merged,
                "every perimeter edge of the merged face enters on its loop"
            );
        }
        mesh.validate_topology();
    }
}
