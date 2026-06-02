use crate::loopmesh::*;

impl LoopMesh {
    /// Rotates an interior edge. The edge must be 2-manifold (exactly two
    /// loops on its radial cycle). The two faces sharing the edge are joined,
    /// then split along the new diagonal connecting the "opposite" corners.
    ///
    /// `ccw` controls which pair of corners becomes the new diagonal:
    /// `ccw=true` follows the first-found face's radial loop, `ccw=false`
    /// swaps the two faces.
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

        let [v1, v2] = self.edges.get(edge).vertices;

        // Choose the corner pair for the new diagonal; `ccw` selects which of
        // the two faces is fa vs fb (swap when !ccw). The new edge connects:
        //   diag_for_v1 = step in fb at v1 away from v2
        //   diag_for_v2 = step in fa at v2 away from v1
        let (fa, fb) = if ccw { (f_a, f_b) } else { (f_b, f_a) };

        let diag_for_v1 = self.face_step_away_from(fb, v1, v2)?;
        let diag_for_v2 = self.face_step_away_from(fa, v2, v1)?;

        // Refuse if the diagonal already exists (would create a duplicate edge).
        if self.query_edge_exists(diag_for_v1, diag_for_v2).is_some() {
            return None;
        }

        // Join then split.
        let merged = self.euler_join_face_kill_edge(f_a, f_b, edge)?;
        let (_new_face, new_edge) = self.face_split(merged, diag_for_v1, diag_for_v2)?;
        Some(new_edge)
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
        let f_a = self.loops.get(l_a).face;
        let f_b = self.loops.get(l_b).face;
        if f_a == f_b {
            return None;
        }
        let [v1, v2] = self.edges.get(edge).vertices;
        let (fa, fb) = if ccw { (f_a, f_b) } else { (f_b, f_a) };
        let diag_for_v1 = self.face_step_away_from(fb, v1, v2)?;
        let diag_for_v2 = self.face_step_away_from(fa, v2, v1)?;
        Some((diag_for_v1, diag_for_v2))
    }

    /// In `face`'s loop cycle, find the loop at vertex `v`, then return the
    /// vertex of the *adjacent* loop that is **not** `v_other`. This is the
    /// "walk one step in the face cycle away from v_other" primitive.
    fn face_step_away_from(
        &self,
        face: FaceKey,
        v: VertexKey,
        v_other: VertexKey,
    ) -> Option<VertexKey> {
        let l_at_v = self
            .face_loop_iter(self.faces.get(face).first_edge_loop)
            .find(|&l| self.loops.get(l).vertex == v)?;
        let links = self.loops.get(l_at_v).face_loop_links;
        let prev_v = self.loops.get(links.prev).vertex;
        let next_v = self.loops.get(links.next).vertex;
        if prev_v == v_other {
            Some(next_v)
        } else if next_v == v_other {
            Some(prev_v)
        } else {
            None
        }
    }
}
