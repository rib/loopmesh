use crate::loopmesh::*;

impl LoopMesh {
    /// Rotates an interior edge. The edge must be 2-manifold (exactly two
    /// loops on its radial cycle). The two faces sharing the edge are joined,
    /// then split along the new diagonal connecting the "opposite" corners.
    ///
    /// `ccw` controls which pair of corners becomes the new diagonal: the new
    /// edge joins the corner reached by stepping **forward** around *each* of the
    /// two faces from the shared edge (`ccw=true`), or **backward** around each
    /// (`ccw=false`). Both faces are stepped the same way, so the pair is a
    /// property of the faces' winding and does not depend on which radial loop
    /// the walk happened to yield first. On two triangles the two directions
    /// coincide — a triangle pair has only one other diagonal.
    ///
    /// Returns the new edge, or `None` if the precondition fails (non-manifold
    /// edge, or the new diagonal already exists between the two corners).
    pub fn edge_rotate(&mut self, edge: EdgeKey, ccw: bool) -> Option<EdgeKey> {
        // Precondition: exactly two loops on the radial cycle.
        let radial: Vec<LoopKey> = self.edge_radial_iter(edge).collect();
        if radial.len() != 2 {
            return None;
        }
        let l_a = radial[0];
        let l_b = radial[1];
        let f_a = self.loops.get(l_a).face;
        let f_b = self.loops.get(l_b).face;
        if f_a == f_b {
            return None;
        }

        let (corner_a, corner_b) = self.rotate_corner_pair(edge, l_a, l_b, ccw)?;

        // Refuse if the diagonal already exists (would create a duplicate edge).
        if self.query_edge_exists(corner_a, corner_b).is_some() {
            return None;
        }

        // Join then split.
        let merged = self.euler_join_face_kill_edge(f_a, f_b, edge)?;
        let (_new_face, new_edge) = self.face_split(merged, corner_a, corner_b)?;
        Some(new_edge)
    }

    /// The corner each of the edge's two radial loops contributes to the new
    /// diagonal. From a loop on the shared edge, stepping *forward* around its
    /// face lands on the corner just past the edge (the loop two steps on);
    /// stepping *backward* lands on the corner just before it (the previous
    /// loop). Applied identically to both loops, so swapping the two loops only
    /// swaps the returned pair — never the diagonal it names.
    fn rotate_corner_pair(
        &self,
        edge: EdgeKey,
        l_a: LoopKey,
        l_b: LoopKey,
        ccw: bool,
    ) -> Option<(VertexKey, VertexKey)> {
        let [v1, v2] = self.edges.get(edge).vertices;
        let corner_of = |l: LoopKey| -> Option<VertexKey> {
            let links = self.loops.get(l).face_loop_links;
            let corner_loop = if ccw {
                self.loops.get(links.next).face_loop_links.next
            } else {
                links.prev
            };
            let corner = self.loops.get(corner_loop).vertex;
            // A face with fewer than three corners has no off-edge corner to
            // rotate onto.
            (corner != v1 && corner != v2).then_some(corner)
        };
        let corner_a = corner_of(l_a)?;
        let corner_b = corner_of(l_b)?;
        if corner_a == corner_b {
            return None;
        }
        Some((corner_a, corner_b))
    }

    /// Compute the corner pair the rotation of `edge` (with direction `ccw`)
    /// would connect — the endpoints of the "other diagonal" of the union
    /// quad. Returns `None` if the edge is not 2-manifold or the geometry does
    /// not yield a well-defined diagonal. This is the read-only counterpart of
    /// the corner selection inside [`Self::edge_rotate`], exposed so the
    /// rotate-edges operator can run its eligibility guards without committing
    /// the surgery.
    pub(crate) fn edge_rotate_diagonal(
        &self,
        edge: EdgeKey,
        ccw: bool,
    ) -> Option<(VertexKey, VertexKey)> {
        let mut radial = self.edge_radial_iter(edge);
        let l_a = radial.next()?;
        let l_b = radial.next()?;
        if radial.next().is_some() {
            return None;
        }
        if self.loops.get(l_a).face == self.loops.get(l_b).face {
            return None;
        }
        self.rotate_corner_pair(edge, l_a, l_b, ccw)
    }
}

#[cfg(test)]
mod tests {
    use crate::loopmesh::*;

    /// Two quads sharing `v1–v2`. `second_face_first` swaps the order the faces are
    /// created in, which swaps which of the shared edge's two loops is its radial
    /// entry — and so which loop a radial walk yields first.
    fn two_quads(second_face_first: bool) -> (LoopMesh, EdgeKey, [VertexKey; 6]) {
        let mut mesh = LoopMesh::new();
        let v0 = mesh.create_vertex(Vec3::new(0.0, 0.0, 0.0));
        let v1 = mesh.create_vertex(Vec3::new(1.0, 0.0, 0.0));
        let v2 = mesh.create_vertex(Vec3::new(1.0, 1.0, 0.0));
        let v3 = mesh.create_vertex(Vec3::new(0.0, 1.0, 0.0));
        let v4 = mesh.create_vertex(Vec3::new(2.0, 0.0, 0.0));
        let v5 = mesh.create_vertex(Vec3::new(2.0, 1.0, 0.0));

        let e01 = mesh.create_edge([v0, v1], EdgeCreateFlags::NONE);
        let shared = mesh.create_edge([v1, v2], EdgeCreateFlags::NONE);
        let e23 = mesh.create_edge([v2, v3], EdgeCreateFlags::NONE);
        let e30 = mesh.create_edge([v3, v0], EdgeCreateFlags::NONE);
        let e14 = mesh.create_edge([v1, v4], EdgeCreateFlags::NONE);
        let e45 = mesh.create_edge([v4, v5], EdgeCreateFlags::NONE);
        let e52 = mesh.create_edge([v5, v2], EdgeCreateFlags::NONE);

        // Left quad crosses the shared edge v1 -> v2; right quad crosses it v2 -> v1.
        let mut left = |mesh: &mut LoopMesh| {
            mesh.face_create(
                &[v0, v1, v2, v3],
                &[e01, shared, e23, e30],
                FaceCreateFlags::NONE,
            );
        };
        let mut right = |mesh: &mut LoopMesh| {
            mesh.face_create(
                &[v1, v4, v5, v2],
                &[e14, e45, e52, shared],
                FaceCreateFlags::NONE,
            );
        };
        if second_face_first {
            right(&mut mesh);
            left(&mut mesh);
        } else {
            left(&mut mesh);
            right(&mut mesh);
        }
        (mesh, shared, [v0, v1, v2, v3, v4, v5])
    }

    fn endpoints(mesh: &LoopMesh, edge: EdgeKey) -> [VertexKey; 2] {
        let [a, b] = mesh.edge_vertices(edge);
        if a.as_usize() <= b.as_usize() {
            [a, b]
        } else {
            [b, a]
        }
    }

    /// RADIAL_EDGE_STRUCTURE.md §9 / ROTATE_EDGES.md §1: the direction flag steps
    /// *forward* (ccw) or *backward* (cw) around **each** adjacent face from the shared
    /// edge. Both faces are stepped the same way, so the corner pair is a property of
    /// the faces' winding — not of which loop the radial walk yielded first. Building
    /// the two quads in either order re-seats the shared edge's radial entry onto the
    /// other face's loop; the rotation must still land on the same diagonal.
    #[test]
    fn rotation_lands_the_same_diagonal_whichever_face_was_built_first() {
        for ccw in [true, false] {
            let mut landed: Vec<[usize; 2]> = Vec::new();
            for second_face_first in [false, true] {
                let (mut mesh, shared, [v0, _v1, _v2, v3, v4, v5]) = two_quads(second_face_first);
                let expected = if ccw { [v3, v4] } else { [v0, v5] };
                let new_edge = mesh.edge_rotate(shared, ccw).expect("manifold quad pair");
                let got = endpoints(&mesh, new_edge);
                assert_eq!(
                    got.map(|v| v.as_usize()),
                    {
                        let mut e = expected.map(|v| v.as_usize());
                        e.sort();
                        e
                    },
                    "ccw={ccw}, second_face_first={second_face_first}",
                );
                landed.push(got.map(|v| v.as_usize()));
                mesh.validate_topology();
            }
            assert_eq!(landed[0], landed[1]);
        }
    }

    /// The two directions connect **different** corner pairs on a quad union — the flag
    /// is part of the operator's identity (§9, "Direction and guards"). The read-only
    /// [`LoopMesh::edge_rotate_diagonal`] must agree with what the surgery commits.
    #[test]
    fn ccw_and_cw_name_different_diagonals_and_the_preview_agrees() {
        let (mut mesh, shared, [v0, _v1, _v2, v3, v4, v5]) = two_quads(false);

        let preview_ccw = mesh.edge_rotate_diagonal(shared, true).expect("2-manifold");
        let preview_cw = mesh
            .edge_rotate_diagonal(shared, false)
            .expect("2-manifold");
        assert_ne!(preview_ccw, preview_cw);

        let mut sorted = |(a, b): (VertexKey, VertexKey)| {
            let mut pair = [a.as_usize(), b.as_usize()];
            pair.sort();
            pair
        };
        let mut expect_ccw = [v3.as_usize(), v4.as_usize()];
        expect_ccw.sort();
        let mut expect_cw = [v0.as_usize(), v5.as_usize()];
        expect_cw.sort();
        assert_eq!(sorted(preview_ccw), expect_ccw);
        assert_eq!(sorted(preview_cw), expect_cw);

        let new_edge = mesh.edge_rotate(shared, true).expect("manifold quad pair");
        assert_eq!(
            endpoints(&mesh, new_edge).map(|v| v.as_usize()),
            expect_ccw,
            "the committed rotation lands where the preview said it would",
        );
    }

    /// A triangle pair has only one other diagonal, so both directions name it — the
    /// forward and backward steps meet at the same apex.
    #[test]
    fn triangle_pair_has_one_diagonal_for_both_directions() {
        let mut mesh = LoopMesh::new();
        let a = mesh.create_vertex(Vec3::new(0.0, 0.0, 0.0));
        let b = mesh.create_vertex(Vec3::new(1.0, 0.0, 0.0));
        let c = mesh.create_vertex(Vec3::new(1.0, 1.0, 0.0));
        let d = mesh.create_vertex(Vec3::new(0.0, 1.0, 0.0));

        let ab = mesh.create_edge([a, b], EdgeCreateFlags::NONE);
        let bc = mesh.create_edge([b, c], EdgeCreateFlags::NONE);
        let ca = mesh.create_edge([c, a], EdgeCreateFlags::NONE);
        let cd = mesh.create_edge([c, d], EdgeCreateFlags::NONE);
        let da = mesh.create_edge([d, a], EdgeCreateFlags::NONE);
        mesh.face_create(&[a, b, c], &[ab, bc, ca], FaceCreateFlags::NONE);
        mesh.face_create(&[a, c, d], &[ca, cd, da], FaceCreateFlags::NONE);

        let ccw = mesh.edge_rotate_diagonal(ca, true).expect("2-manifold");
        let cw = mesh.edge_rotate_diagonal(ca, false).expect("2-manifold");
        let normalise = |(x, y): (VertexKey, VertexKey)| {
            let mut pair = [x.as_usize(), y.as_usize()];
            pair.sort();
            pair
        };
        assert_eq!(normalise(ccw), normalise(cw));
        let mut expected = [b.as_usize(), d.as_usize()];
        expected.sort();
        assert_eq!(normalise(ccw), expected);
    }
}
