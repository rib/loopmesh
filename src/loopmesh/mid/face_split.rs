use glam::Vec3;

use crate::loopmesh::*;

impl LoopMesh {
    /// High-level wrapper around [`Self::euler_split_face_make_edge`].
    /// Finds the loops in `face` that sit at vertices `v1` and `v2`, then
    /// invokes SFME. Returns `(new_face, new_edge)`, or `None` if the SFME
    /// preconditions (RADIAL_EDGE_STRUCTURE.md §8.3) fail: `v1 == v2`, either
    /// vertex is absent from the face, or the two are **adjacent** corners of
    /// the face (a perimeter edge already joins them, so there is no diagonal
    /// to draw). On refusal the mesh is left untouched.
    pub fn face_split(
        &mut self,
        face: FaceKey,
        v1: VertexKey,
        v2: VertexKey,
    ) -> Option<(FaceKey, EdgeKey)> {
        self.face_split_reusing(face, v1, v2, false)
    }

    /// [`Self::face_split`] with the SFME edge-reuse variant
    /// (RADIAL_EDGE_STRUCTURE.md §8.3.1). When `reuse_existing_edge` is set and
    /// `v1`–`v2` are already joined by an interior/wire edge, that edge is
    /// adopted as the splitting diagonal instead of a duplicate being created.
    /// The reuse path can leave a duplicate sub-face; the caller is responsible
    /// for detecting and removing it.
    pub fn face_split_reusing(
        &mut self,
        face: FaceKey,
        v1: VertexKey,
        v2: VertexKey,
        reuse_existing_edge: bool,
    ) -> Option<(FaceKey, EdgeKey)> {
        if v1 == v2 {
            return None;
        }
        let mut l_v1 = None;
        let mut l_v2 = None;
        let mut it = self.face_loops(face);
        while let Some(l) = it.next(self) {
            let here = self.loops.get(l).vertex;
            if here == v1 && l_v1.is_none() {
                l_v1 = Some(l);
            } else if here == v2 && l_v2.is_none() {
                l_v2 = Some(l);
            }
            if l_v1.is_some() && l_v2.is_some() {
                break;
            }
        }
        let l_v1 = l_v1?;
        let l_v2 = l_v2?;

        // §8.3's non-adjacency precondition, measured on the *loop cycle*: with
        // one corner per vertex, v1 and v2 are cycle-adjacent exactly when one's
        // successor corner is the other. A perimeter edge then already joins
        // them and the "split" would carve off a two-sided sliver instead of
        // partitioning the face. The wrap-around pair needs no special case —
        // the loop cycle is a cycle. Note this is deliberately *not* an edge-
        // existence test: under §8.3.1 an interior/wire edge may already join
        // two non-adjacent corners, and that split must still run (reusing it).
        // Every mutation happens after this point, so refusing leaves the mesh
        // untouched.
        if self.loop_vertex(self.loop_next(l_v1)) == v2
            || self.loop_vertex(self.loop_next(l_v2)) == v1
        {
            return None;
        }

        let (new_face, new_edge, l_f2) =
            self.euler_split_face_make_edge_reusing(face, l_v1, l_v2, reuse_existing_edge);

        // Customdata interp: new_face copies the source face's CD. new_edge is
        // a fresh interior diagonal — no source edge to inherit from, so it
        // stays at default values.
        self.face_attribs
            .copy_into(new_face.as_usize(), face.as_usize());

        // Loop interp. SFME creates two new loops sitting on the
        // diagonal: l_f1 at v2 in the surviving half of `face`, l_f2 at v1
        // in `new_face`. Each takes its CD from the existing same-vertex
        // loop (the corner mate on the opposite half).
        // - l_f2 sits at v1 — copy from the original l_v1.
        // - l_f1 is the other loop on new_edge's radial — copy from l_v2.
        if !self.loop_attribs.is_empty() {
            self.loop_attribs
                .copy_into(l_f2.as_usize(), l_v1.as_usize());
            // Walk new_edge's two-loop radial to find l_f1.
            let mut it = self.radial_loops(new_edge);
            while let Some(l) = it.next(self) {
                if l != l_f2 {
                    self.loop_attribs.copy_into(l.as_usize(), l_v2.as_usize());
                    break;
                }
            }
        }

        Some((new_face, new_edge))
    }

    /// Splits `face` along v1–v2 (like [`Self::face_split`]) and then inserts
    /// `cut_positions.len()` intermediate vertices along the new diagonal at
    /// the supplied positions. Returns `(new_face, new_edge, intermediates)`
    /// where `intermediates[i]` is the vertex placed at `cut_positions[i]` —
    /// ordered from `v1` toward `v2`.
    ///
    /// Empty `cut_positions` is valid (degenerate to plain `face_split`).
    /// Returns `None` whenever [`Self::face_split`] refuses (`v1 == v2`, a
    /// vertex absent from the face, or adjacent corners).
    pub fn face_split_n(
        &mut self,
        face: FaceKey,
        v1: VertexKey,
        v2: VertexKey,
        cut_positions: &[Vec3],
    ) -> Option<(FaceKey, EdgeKey, Vec<VertexKey>)> {
        let (new_face, new_edge) = self.face_split(face, v1, v2)?;
        let mids = self.edge_split_n(new_edge, v1, cut_positions.len() as u32);
        for (v_mid, &pos) in mids.iter().zip(cut_positions.iter()) {
            self.set_vertex_position(*v_mid, pos);
        }
        Some((new_face, new_edge, mids))
    }
}

#[cfg(test)]
mod tests {
    use glam::Vec3;

    use crate::loopmesh::test_support::create_face_from_verts;
    use crate::loopmesh::*;

    /// face_split wrapper produces the same result as SFME but accepts vertex keys.
    #[test]
    fn test_face_split_quad_diagonal() {
        let mut mesh = LoopMesh::new();
        let verts = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::new(2.0, 1.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        ];
        let face = create_face_from_verts(&mut mesh, &verts);
        let fv = mesh.face_vert_keys(face);
        let (new_face, new_edge) = mesh.face_split(face, fv[0], fv[2]).unwrap();
        assert_eq!(mesh.face_len(face), 3);
        assert_eq!(mesh.face_len(new_face), 3);
        let nv = mesh.edge_vertices(new_edge);
        assert!(nv == [fv[0], fv[2]] || nv == [fv[2], fv[0]]);
    }

    fn quad(mesh: &mut LoopMesh) -> FaceKey {
        let verts = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::new(2.0, 1.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        ];
        create_face_from_verts(mesh, &verts)
    }

    fn counts(mesh: &LoopMesh) -> (usize, usize, usize) {
        (mesh.vertex_count(), mesh.edge_count(), mesh.face_count())
    }

    /// §8.3 precondition: adjacent corners have no diagonal — the split refuses
    /// (`None`) and mutates nothing, rather than minting a 2-gon.
    #[test]
    fn test_face_split_refuses_adjacent_corners() {
        let mut mesh = LoopMesh::new();
        let face = quad(&mut mesh);
        let fv = mesh.face_vert_keys(face);
        let before = counts(&mesh);

        assert!(mesh.face_split(face, fv[0], fv[1]).is_none());
        assert!(mesh.face_split(face, fv[1], fv[0]).is_none());

        assert_eq!(counts(&mesh), before);
        assert_eq!(mesh.face_len(face), 4);
        assert_eq!(mesh.face_vert_keys(face), fv);
        mesh.validate_topology();
    }

    /// The wrap-around adjacency (last corner + first corner) is the same
    /// perimeter edge and refuses just the same.
    #[test]
    fn test_face_split_refuses_wraparound_adjacent_corners() {
        let mut mesh = LoopMesh::new();
        let face = quad(&mut mesh);
        let fv = mesh.face_vert_keys(face);
        let before = counts(&mesh);

        assert!(mesh.face_split(face, fv[3], fv[0]).is_none());
        assert!(mesh.face_split(face, fv[0], fv[3]).is_none());

        assert_eq!(counts(&mesh), before);
        assert_eq!(mesh.face_vert_keys(face), fv);

        // The legal diagonal on the same, untouched face still splits.
        let (new_face, _) = mesh.face_split(face, fv[1], fv[3]).unwrap();
        assert_eq!(mesh.face_len(face), 3);
        assert_eq!(mesh.face_len(new_face), 3);
        assert_eq!(mesh.face_count(), before.2 + 1);
        assert_eq!(mesh.edge_count(), before.1 + 1);
        mesh.validate_topology();
    }

    /// The reuse variant carries the same precondition (§8.3.1 changes only
    /// which edge the split is built around).
    #[test]
    fn test_face_split_reusing_refuses_adjacent_corners() {
        let mut mesh = LoopMesh::new();
        let face = quad(&mut mesh);
        let fv = mesh.face_vert_keys(face);
        let before = counts(&mesh);

        assert!(mesh.face_split_reusing(face, fv[1], fv[2], true).is_none());

        assert_eq!(counts(&mesh), before);
        assert_eq!(mesh.face_vert_keys(face), fv);
        mesh.validate_topology();
    }

    /// §8.3's precondition is about the face's *loop cycle*, not about edge
    /// incidence: two opposite corners already joined by an interior/wire edge
    /// are still non-adjacent on the cycle, so the split must run — that is
    /// exactly §8.3.1's reuse case. Rewriting the walk as a `query_edge_exists`
    /// test would kill this path.
    #[test]
    fn test_face_split_splits_nonadjacent_corners_already_joined_by_an_edge() {
        let mut mesh = LoopMesh::new();
        let face = quad(&mut mesh);
        let fv = mesh.face_vert_keys(face);
        // A wire diagonal across the quad: fv[0]–fv[2] are joined by an edge but
        // are not consecutive corners of `face`.
        let wire = mesh.create_edge([fv[0], fv[2]], EdgeCreateFlags::NONE);
        let before = counts(&mesh);

        let (new_face, new_edge) = mesh.face_split_reusing(face, fv[0], fv[2], true).unwrap();

        assert_eq!(new_edge, wire, "the reuse variant adopts the existing edge");
        assert_eq!(mesh.face_len(face), 3);
        assert_eq!(mesh.face_len(new_face), 3);
        assert_eq!(mesh.edge_count(), before.1, "no duplicate edge is minted");
        assert_eq!(mesh.face_count(), before.2 + 1);
        mesh.validate_topology();
    }

    /// The same non-adjacent-but-already-joined pair through the plain (no
    /// reuse) entry point: it also splits, laying its own diagonal alongside the
    /// pre-existing edge.
    #[test]
    fn test_face_split_no_reuse_splits_corners_already_joined_by_an_edge() {
        let mut mesh = LoopMesh::new();
        let face = quad(&mut mesh);
        let fv = mesh.face_vert_keys(face);
        let wire = mesh.create_edge([fv[0], fv[2]], EdgeCreateFlags::NONE);
        let before = counts(&mesh);

        let (new_face, new_edge) = mesh.face_split(face, fv[0], fv[2]).unwrap();

        assert_ne!(new_edge, wire);
        assert_eq!(mesh.face_len(face), 3);
        assert_eq!(mesh.face_len(new_face), 3);
        assert_eq!(mesh.edge_count(), before.1 + 1);
        mesh.validate_topology();
    }

    /// `face_split_n` is built on `face_split`, so it inherits the §8.3 refusal
    /// (its doc promises this) — and inherits "mutates nothing" with it.
    #[test]
    fn test_face_split_n_inherits_adjacent_corner_refusal() {
        let mut mesh = LoopMesh::new();
        let face = quad(&mut mesh);
        let fv = mesh.face_vert_keys(face);
        let before = counts(&mesh);

        let cuts = [Vec3::new(1.0, 0.0, 0.0)];
        assert!(mesh.face_split_n(face, fv[0], fv[1], &cuts).is_none());
        assert!(mesh.face_split_n(face, fv[3], fv[0], &[]).is_none());

        assert_eq!(counts(&mesh), before);
        assert_eq!(mesh.face_vert_keys(face), fv);
        mesh.validate_topology();
    }

    #[test]
    fn test_face_split_rejects_same_vertex() {
        let mut mesh = LoopMesh::new();
        let verts = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::new(2.0, 1.0, 0.0),
        ];
        let face = create_face_from_verts(&mut mesh, &verts);
        let v = mesh.face_vert_keys(face)[0];
        assert!(mesh.face_split(face, v, v).is_none());
    }
}
