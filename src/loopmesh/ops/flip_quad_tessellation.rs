use crate::loopmesh::*;

impl LoopMesh {
    /// Flip the implied tessellation diagonal of every named quad by advancing
    /// its first loop one corner around the loop cycle.
    ///
    /// A quad fanned from its first corner draws its split diagonal between
    /// corners 0 and 2 of the current start; advancing the start by one corner
    /// moves that diagonal to the other pair of opposite corners. This operator
    /// performs exactly that advance for each 4-corner face in `faces`: the loop
    /// that follows the old start becomes the new start. The loop cycle is left
    /// structurally identical — no loop's vertex, edge, `next`/`prev`, or
    /// customdata changes, and winding, geometry, and normals are untouched.
    /// The only change is which loop the face calls its start, and hence the
    /// diagonal a start-respecting triangulator (the `FIXED` quad method) will
    /// pick.
    ///
    /// Faces that are not quads are visited and left untouched: a triangle has
    /// no diagonal to flip and an `n>4` gon has no single diagonal a one-corner
    /// advance cleanly swaps. A mixed-arity set is therefore legal — each quad's
    /// start advances and every other face is skipped. Each face is handled in
    /// isolation, so the result is order-independent.
    ///
    /// Advancing the start has period 4 (four advances restore the original
    /// start) but the implied diagonal has period 2 (two advances restore the
    /// diagonal while the start has only travelled halfway around). Applying the
    /// operator twice is therefore an identity for the tessellation.
    ///
    /// When at least one quad's start advances, the cached loop index table is
    /// invalidated: walking from the new start renumbers the loops, so any
    /// derived loop-index array is stale until rebuilt. This is bookkeeping, not
    /// an observable mesh change. An input with no quads is a whole-operator
    /// no-op and leaves the table valid.
    pub fn flip_quad_tessellation(&mut self, faces: &[FaceKey]) {
        let mut advanced_any = false;

        for &face in faces {
            if self.face_len(face) != 4 {
                continue;
            }

            let old_start = self.face_first_loop(face);
            let new_start = self.loop_next(old_start);
            self.faces.get_mut(face).first_edge_loop = new_start;
            advanced_any = true;
        }

        if advanced_any {
            self.flags |= LoopMeshFlags::LOOPS_INDEX_DIRTY;
        }
    }
}

#[cfg(test)]
mod tests {
    use glam::Vec3;

    use crate::loopmesh::test_support::create_face_from_verts;
    use crate::loopmesh::*;

    fn ngon(mesh: &mut LoopMesh, n: usize) -> FaceKey {
        let verts: Vec<Vec3> = (0..n)
            .map(|i| {
                let a = std::f32::consts::TAU * (i as f32) / (n as f32);
                Vec3::new(a.cos(), a.sin(), 0.0)
            })
            .collect();
        create_face_from_verts(mesh, &verts)
    }

    /// The implied fan-from-first-corner diagonal: the unordered vertex pair
    /// (start corner, opposite corner) — corner 0 and corner 2 of the current
    /// start. Returned sorted so it compares as an unordered pair.
    fn implied_diagonal(mesh: &LoopMesh, face: FaceKey) -> [usize; 2] {
        let start = mesh.face_first_loop(face);
        let opposite = mesh.loop_next(mesh.loop_next(start));
        let a = mesh.loop_vertex(start).as_usize();
        let b = mesh.loop_vertex(opposite).as_usize();
        if a <= b {
            [a, b]
        } else {
            [b, a]
        }
    }

    /// §2 / §5 — one application advances the start by exactly one corner: the
    /// loop that followed the old start becomes the new start, and the cycle is
    /// otherwise untouched.
    #[test]
    fn single_quad_advances_start() {
        let mut mesh = LoopMesh::new();
        let face = ngon(&mut mesh, 4);

        let old_start = mesh.face_first_loop(face);
        let expected_new = mesh.loop_next(old_start);
        let next_after = mesh.loop_next(expected_new);
        let prev_after = mesh.loop_prev(expected_new);

        mesh.flip_quad_tessellation(&[face]);

        let new_start = mesh.face_first_loop(face);
        assert_eq!(new_start, expected_new, "start advanced by one corner");
        // The links around the cycle are unchanged.
        assert_eq!(mesh.loop_next(new_start), next_after);
        assert_eq!(mesh.loop_prev(new_start), prev_after);
        // The corner vertex set of the face is unchanged.
        assert_eq!(mesh.face_len(face), 4);
    }

    /// §3 / §6.5 — one application flips the implied diagonal to the opposite
    /// vertex pair; the post-flip diagonal differs from the un-flipped one.
    #[test]
    fn single_flip_changes_implied_diagonal() {
        let mut mesh = LoopMesh::new();
        let face = ngon(&mut mesh, 4);

        let before = implied_diagonal(&mesh, face);
        mesh.flip_quad_tessellation(&[face]);
        let after = implied_diagonal(&mesh, face);

        assert_ne!(before, after, "one flip moves the diagonal to the other pair");
    }

    /// §5.1 — period of the start (4) vs period of the diagonal (2): two
    /// advances restore the implied diagonal but NOT the start; four advances
    /// restore the start exactly.
    #[test]
    fn start_period_four_diagonal_period_two() {
        let mut mesh = LoopMesh::new();
        let face = ngon(&mut mesh, 4);

        let start0 = mesh.face_first_loop(face);
        let diag0 = implied_diagonal(&mesh, face);

        mesh.flip_quad_tessellation(&[face]);
        let start1 = mesh.face_first_loop(face);
        let diag1 = implied_diagonal(&mesh, face);
        assert_ne!(start1, start0, "one advance moves the start");
        assert_ne!(diag1, diag0, "one advance flips the diagonal");

        mesh.flip_quad_tessellation(&[face]);
        let start2 = mesh.face_first_loop(face);
        let diag2 = implied_diagonal(&mesh, face);
        assert_ne!(start2, start0, "two advances have NOT restored the start");
        assert_eq!(diag2, diag0, "two advances restore the implied diagonal");

        mesh.flip_quad_tessellation(&[face]);
        mesh.flip_quad_tessellation(&[face]);
        let start4 = mesh.face_first_loop(face);
        let diag4 = implied_diagonal(&mesh, face);
        assert_eq!(start4, start0, "four advances restore the start exactly");
        assert_eq!(diag4, diag0, "and with it the diagonal");
    }

    /// §4 — non-quad faces are visited and left untouched: a triangle and a
    /// pentagon keep their start loop. Their loop index table is not even
    /// dirtied (whole-operator no-op when no quad is present).
    #[test]
    fn non_quads_skipped() {
        let mut mesh = LoopMesh::new();
        let tri = ngon(&mut mesh, 3);
        let pentagon = ngon(&mut mesh, 5);
        mesh.loops_ensure_index();

        let tri_start = mesh.face_first_loop(tri);
        let pent_start = mesh.face_first_loop(pentagon);

        mesh.flip_quad_tessellation(&[tri, pentagon]);

        assert_eq!(mesh.face_first_loop(tri), tri_start, "triangle untouched");
        assert_eq!(
            mesh.face_first_loop(pentagon),
            pent_start,
            "pentagon untouched"
        );
        assert!(
            !mesh.is_loops_index_dirty(),
            "no quad advanced ⇒ loop index table stays valid"
        );
    }

    /// §5 / §6.6 — a mixed-arity set: only the quad's start advances; the
    /// triangle and pentagon are skipped. The presence of a quad dirties the
    /// loop index table.
    #[test]
    fn mixed_arity_routing() {
        let mut mesh = LoopMesh::new();
        let tri = ngon(&mut mesh, 3);
        let quad = ngon(&mut mesh, 4);
        let pentagon = ngon(&mut mesh, 5);
        mesh.loops_ensure_index();

        let tri_start = mesh.face_first_loop(tri);
        let pent_start = mesh.face_first_loop(pentagon);
        let quad_start = mesh.face_first_loop(quad);
        let expected_quad = mesh.loop_next(quad_start);

        mesh.flip_quad_tessellation(&[tri, quad, pentagon]);

        assert_eq!(mesh.face_first_loop(quad), expected_quad, "quad advanced");
        assert_eq!(mesh.face_first_loop(tri), tri_start, "triangle skipped");
        assert_eq!(
            mesh.face_first_loop(pentagon),
            pent_start,
            "pentagon skipped"
        );
        assert!(
            mesh.is_loops_index_dirty(),
            "a quad advanced ⇒ loop index table is dirty"
        );
    }

    /// §5 — order-independence and multi-quad routing: flipping a set of quads
    /// advances every quad's start regardless of order.
    #[test]
    fn multi_quad_each_flips() {
        let mut mesh = LoopMesh::new();
        let quad_a = ngon(&mut mesh, 4);
        let quad_b = ngon(&mut mesh, 4);

        let expected_a = mesh.loop_next(mesh.face_first_loop(quad_a));
        let expected_b = mesh.loop_next(mesh.face_first_loop(quad_b));

        mesh.flip_quad_tessellation(&[quad_b, quad_a]);

        assert_eq!(mesh.face_first_loop(quad_a), expected_a);
        assert_eq!(mesh.face_first_loop(quad_b), expected_b);
    }

    /// §7 — empty input is a no-op and does not dirty the loop index table.
    #[test]
    fn empty_input_is_noop() {
        let mut mesh = LoopMesh::new();
        let quad = ngon(&mut mesh, 4);
        mesh.loops_ensure_index();
        let start = mesh.face_first_loop(quad);

        mesh.flip_quad_tessellation(&[]);

        assert_eq!(mesh.face_first_loop(quad), start);
        assert!(!mesh.is_loops_index_dirty());
    }
}
