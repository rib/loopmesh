use super::*;

impl LoopMesh {
    /// Normalise to unit length, or return the all-zero vector when the
    /// input is (near-)zero or contains a NaN. The test is on the squared
    /// length against a threshold far below float epsilon, so any face with
    /// measurable area normalises normally; only a genuinely degenerate
    /// (zero-area) face takes the zero branch.
    pub(crate) fn normalize_or_zero(v: Vec3) -> Vec3 {
        let len_sq = v.dot(v);
        if len_sq > 1.0e-35 {
            v / len_sq.sqrt()
        } else {
            Vec3::ZERO
        }
    }

    pub(crate) fn calc_tri_cross(corners: [Vec3; 3]) -> Vec3 {
        let side0 = corners[0] - corners[1];
        let side1 = corners[1] - corners[2];
        side0.cross(side1)
    }

    pub(crate) fn calc_tri_normal(corners: [Vec3; 3]) -> Vec3 {
        LoopMesh::normalize_or_zero(LoopMesh::calc_tri_cross(corners))
    }

    pub(crate) fn calc_tri_area(corners: [Vec3; 3]) -> f32 {
        LoopMesh::calc_tri_cross(corners).length() * 0.5
    }

    pub(crate) fn calc_quad_normal(corners: [Vec3; 4]) -> Vec3 {
        let diagonal0 = corners[0] - corners[2];
        let diagonal1 = corners[1] - corners[3];
        let cross = diagonal0.cross(diagonal1);
        LoopMesh::normalize_or_zero(cross)
    }

    pub(crate) fn calc_quad_area(corners: [Vec3; 4]) -> f32 {
        let diagonal0 = corners[0] - corners[2];
        let diagonal1 = corners[1] - corners[3];
        let cross = diagonal0.cross(diagonal1);
        cross.length() * 0.5
    }

    pub(crate) fn calc_poly_newell_normal_not_normalized(corners: &[Vec3]) -> Vec3 {
        /*
        Newell Method:

        Pseudo code from: https://www.khronos.org/opengl/wiki/Calculating_a_Surface_Normal

        Set Vertex Normal to (0, 0, 0)

        Begin Cycle for Index in [0, Polygon.vertexNumber)

            Set Vertex Current to Polygon.verts[Index]
            Set Vertex Next    to Polygon.verts[(Index plus 1) mod Polygon.vertexNumber]

            Set Normal.x to Sum of Normal.x and (multiply (Current.y minus Next.y) by (Current.z plus Next.z))
            Set Normal.y to Sum of Normal.y and (multiply (Current.z minus Next.z) by (Current.x plus Next.x))
            Set Normal.z to Sum of Normal.z and (multiply (Current.x minus Next.x) by (Current.y plus Next.y))

        End Cycle

        Returning Normalize(Normal)
         */

        let mut normal = Vec3::ZERO;
        for i in 0..corners.len() {
            let cur = corners[i];
            let next = corners[(i + 1) % corners.len()];

            normal.x += (cur.y - next.y) * (cur.z + next.z);
            normal.y += (cur.z - next.z) * (cur.x + next.x);
            normal.z += (cur.x - next.x) * (cur.y + next.y);
        }

        normal
    }

    pub(crate) fn calc_poly_normal(corners: &[Vec3]) -> Vec3 {
        LoopMesh::normalize_or_zero(LoopMesh::calc_poly_newell_normal_not_normalized(corners))
    }

    pub(crate) fn calc_iter_newell_normal_not_normalized(
        mut corner_iter: impl Iterator<Item = Vec3>,
    ) -> Vec3 {
        match corner_iter.next() {
            Some(first_corner) => {
                let mut normal = Vec3::ZERO;
                let mut prev = first_corner;
                let mut end = false;
                loop {
                    let cur = match corner_iter.next() {
                        Some(cur) => cur,
                        None => {
                            end = true;
                            first_corner
                        }
                    };

                    normal.x += (prev.y - cur.y) * (prev.z + cur.z);
                    normal.y += (prev.z - cur.z) * (prev.x + cur.x);
                    normal.z += (prev.x - cur.x) * (prev.y + cur.y);

                    if end {
                        break;
                    }

                    prev = cur;
                }

                normal
            }
            None => Vec3::ONE,
        }
    }

    pub(crate) fn calc_iter_poly_normal(corner_iter: impl Iterator<Item = Vec3>) -> Vec3 {
        LoopMesh::normalize_or_zero(LoopMesh::calc_iter_newell_normal_not_normalized(
            corner_iter,
        ))
    }

    pub(crate) fn calc_poly_area(corners: &[Vec3]) -> f32 {
        f32::abs(LoopMesh::calc_poly_newell_normal_not_normalized(corners).length() * 0.5)
    }

    pub(crate) fn calc_iter_poly_area(corner_iter: impl Iterator<Item = Vec3>) -> f32 {
        f32::abs(LoopMesh::calc_iter_newell_normal_not_normalized(corner_iter).length() * 0.5)
    }

    pub fn face_calc_normal(&mut self, face: FaceKey) {
        let normal = self.face_compute_normal(face);
        self.faces.get_mut(face).normal = normal;
    }

    /// Compute (but do not store) the face's normal from its current loop
    /// positions. Uses Newell's method for non-triangle / non-quad faces
    /// so non-planar polygons get a reasonable result. The output is unit
    /// length for a non-degenerate face; for a fully degenerate (zero-area)
    /// face — collinear or coincident corners — the output is the all-zero
    /// vector `(0, 0, 0)`, never a `NaN` or a fixed-axis fallback.
    pub fn face_compute_normal(&self, face: FaceKey) -> Vec3 {
        let (face_len, first_edge_loop) = {
            let face_data = self.faces.get(face);
            (face_data.len, face_data.first_edge_loop)
        };
        let iter = self.face_loop_iter(first_edge_loop);
        let mut corner_iter = iter.map(|l| self.vertices.get(self.loops.get(l).vertex).position);
        match face_len {
            3 => {
                let mut pos = [Vec3::ZERO; 3];
                for i in 0..3 {
                    pos[i] = corner_iter.next().unwrap();
                }
                LoopMesh::calc_tri_normal(pos)
            }
            4 => {
                let mut pos = [Vec3::ZERO; 4];
                for i in 0..4 {
                    pos[i] = corner_iter.next().unwrap();
                }
                LoopMesh::calc_quad_normal(pos)
            }
            _ => LoopMesh::calc_iter_poly_normal(corner_iter),
        }
    }

    pub fn face_calc_area(&mut self, face: FaceKey) -> f32 {
        let face_data = self.faces.get(face);
        let iter = self.face_loop_iter(face_data.first_edge_loop);
        let mut corner_iter = iter.map(|l| self.vertices.get(self.loops.get(l).vertex).position);
        match face_data.len {
            3 => {
                let mut pos = [Vec3::ZERO; 3];
                for i in 0..3 {
                    pos[i] = corner_iter.next().unwrap();
                }
                LoopMesh::calc_tri_area(pos)
            }
            4 => {
                let mut pos = [Vec3::ZERO; 4];
                for i in 0..4 {
                    pos[i] = corner_iter.next().unwrap();
                }
                LoopMesh::calc_quad_area(pos)
            }
            _ => LoopMesh::calc_iter_poly_area(corner_iter),
        }
    }

    pub fn calc_face_normals(&mut self) {
        let mut it = self.mesh_faces();
        while let Some(face) = it.next(self) {
            self.face_calc_normal(face);
        }
    }

    /// Recompute a single face's normal. Alias for [`Self::face_calc_normal`].
    pub fn face_normal_update(&mut self, face: FaceKey) {
        self.face_calc_normal(face);
    }

    /// Recompute a single vertex's normal as the area-weighted average of
    /// incident face normals, normalised. Falls back to `Vec3::Z` for
    /// isolated / wire-only vertices.
    ///
    /// Note: callers should ensure face normals are current first (e.g.
    /// via [`Self::mesh_normals_update`]); the area weight is read from
    /// each face's stored `normal` field.
    pub fn vert_normal_update(&mut self, v: VertexKey) {
        // Walk v's incident faces (with dedup) using lending iterators —
        // SmallVec keeps the dedup set off the heap for typical valences.
        let mut seen: smallvec::SmallVec<[FaceKey; 8]> = smallvec::SmallVec::new();
        let mut disk = self.disk_edges(v);
        while let Some(e) = disk.next(self) {
            let mut radial = self.radial_loops(e);
            while let Some(l) = radial.next(self) {
                let f = self.loops.get(l).face;
                if !seen.contains(&f) {
                    seen.push(f);
                }
            }
        }
        if seen.is_empty() {
            self.set_vertex_normal(v, Vec3::Z);
            return;
        }
        let mut acc = Vec3::ZERO;
        for &f in &seen {
            let area = self.face_calc_area(f);
            let n_f = self.faces.get(f).normal;
            acc += area * n_f;
        }
        let n = if acc.length_squared() > 0.0 {
            acc.normalize()
        } else {
            Vec3::Z
        };
        self.set_vertex_normal(v, n);
    }

    /// Recompute every face normal, then every vertex normal. This is the
    /// canonical "freshen all cached normals" entry point.
    pub fn mesh_normals_update(&mut self) {
        let mut it_f = self.mesh_faces();
        while let Some(f) = it_f.next(self) {
            self.face_calc_normal(f);
        }
        let mut it_v = self.mesh_verts();
        while let Some(v) = it_v.next(self) {
            self.vert_normal_update(v);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::loopmesh::test_support::{create_face_from_verts, mesh_add_cube};

    #[test]
    fn update_cube_normals() {
        let mut mesh = LoopMesh::new();
        mesh_add_cube(&mut mesh);
        mesh.calc_face_normals();

        let normals: Vec<Vec3> = mesh
            .faces
            .iter()
            .map(|f| mesh.faces.get(f).normal)
            .collect();
        assert_eq!(normals[0], Vec3::new(0.0, 0.0, -1.0));
        assert_eq!(normals[1], Vec3::new(-1.0, 0.0, 0.0));
        assert_eq!(normals[2], Vec3::new(0.0, 0.0, 1.0));
        assert_eq!(normals[3], Vec3::new(1.0, 0.0, 0.0));
        assert_eq!(normals[4], Vec3::new(0.0, 1.0, 0.0));
        assert_eq!(normals[5], Vec3::new(0.0, -1.0, 0.0));
    }

    #[test]
    fn test_calc_tri_quad_poly_area() {
        let area = LoopMesh::calc_tri_area([
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
        ]);
        assert_eq!(area, 0.5);

        let area = LoopMesh::calc_tri_area([
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 2.0, 0.0),
            Vec3::new(0.0, 2.0, 2.0),
        ]);
        assert_eq!(area, 2.0);

        let area = LoopMesh::calc_quad_area([
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 2.0, 0.0),
            Vec3::new(0.0, 2.0, 2.0),
            Vec3::new(0.0, 0.0, 2.0),
        ]);
        assert_eq!(area, 4.0);

        let area = LoopMesh::calc_poly_area(&[
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 2.0, 0.0),
            Vec3::new(0.0, 2.0, 2.0),
            Vec3::new(0.0, 0.0, 2.0),
        ]);
        assert_eq!(area, 4.0);

        //             5        4
        //             ______________ 3 (2,1)
        //            /|        |    /
        //          /  |        |  /
        // (-1,0) /____|________ /
        //        0    1        2
        let area = LoopMesh::calc_poly_area(&[
            Vec3::new(-1.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(2.0, 1.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        ]);
        assert_eq!(area, 2.0);
    }

    #[test]
    fn test_calc_tri_quad_poly_normals() {
        let normal = LoopMesh::calc_tri_normal([
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::new(2.0, 2.0, 0.0),
        ]);
        assert_eq!(normal, Vec3::new(0.0, 0.0, 1.0));

        let normal = LoopMesh::calc_quad_normal([
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::new(2.0, 1.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        ]);
        assert_eq!(normal, Vec3::new(0.0, 0.0, 1.0));

        let normal = LoopMesh::calc_poly_normal(&[
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(1.0, 2.0, 0.0),
            Vec3::new(0.0, 2.0, 0.0),
        ]);
        assert_eq!(normal, Vec3::new(0.0, 0.0, 1.0));

        //             5        4
        //             ______________ 3 (2,1)
        //            /|        |    /
        //          /  |        |  /
        // (-1,0) /____|________ /
        //        0    1        2
        let normal = LoopMesh::calc_poly_normal(&[
            Vec3::new(-1.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(2.0, 1.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        ]);
        assert_eq!(normal, Vec3::new(0.0, 0.0, 1.0));

        let normal = LoopMesh::calc_iter_poly_normal(
            [
                Vec3::new(-1.0, 0.0, 0.0),
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(2.0, 1.0, 0.0),
                Vec3::new(1.0, 1.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
            ]
            .into_iter(),
        );
        assert_eq!(normal, Vec3::new(0.0, 0.0, 1.0));
    }

    #[test]
    fn test_normalize_or_zero_unit_and_degenerate() {
        // Non-degenerate input normalises to unit length.
        let n = LoopMesh::normalize_or_zero(Vec3::new(0.0, 0.0, 5.0));
        assert_eq!(n, Vec3::new(0.0, 0.0, 1.0));

        // True zero falls back to the all-zero vector, never NaN.
        let n = LoopMesh::normalize_or_zero(Vec3::ZERO);
        assert_eq!(n, Vec3::ZERO);

        // A NaN component fails the `> 1.0e-35` test and takes the zero branch.
        let n = LoopMesh::normalize_or_zero(Vec3::new(f32::NAN, 0.0, 0.0));
        assert_eq!(n, Vec3::ZERO);
    }

    #[test]
    fn test_degenerate_tri_normal_is_zero() {
        // Collinear corners -> zero-area triangle -> zero normal.
        let normal = LoopMesh::calc_tri_normal([
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
        ]);
        assert_eq!(normal, Vec3::ZERO);
    }

    #[test]
    fn test_degenerate_quad_normal_is_zero() {
        // Parallel diagonals (both along X) -> zero diagonal cross -> zero normal,
        // even though the quad's edges are non-zero.
        let normal = LoopMesh::calc_quad_normal([
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::new(3.0, 0.0, 0.0),
        ]);
        assert_eq!(normal, Vec3::ZERO);

        // Coincident-corner sliver quad (two pairs collapse) -> zero normal.
        let normal = LoopMesh::calc_quad_normal([
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
        ]);
        assert_eq!(normal, Vec3::ZERO);
    }

    #[test]
    fn test_degenerate_poly_normal_is_zero() {
        // Collinear n-gon -> Newell accumulator is zero -> zero normal.
        let corners = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::new(3.0, 0.0, 0.0),
            Vec3::new(4.0, 0.0, 0.0),
        ];
        assert_eq!(LoopMesh::calc_poly_normal(&corners), Vec3::ZERO);
        assert_eq!(
            LoopMesh::calc_iter_poly_normal(corners.into_iter()),
            Vec3::ZERO
        );
    }

    #[test]
    fn test_calc_face_normals() {
        let mut mesh = LoopMesh::new();

        let verts = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::new(2.0, 2.0, 0.0),
        ];
        let face = create_face_from_verts(&mut mesh, &verts);
        mesh.face_calc_normal(face);
        let normal = mesh.faces.get(face).normal;
        assert_eq!(normal, Vec3::new(0.0, 0.0, 1.0));

        let verts = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::new(2.0, 1.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        ];
        let face = create_face_from_verts(&mut mesh, &verts);
        mesh.face_calc_normal(face);
        let normal = mesh.faces.get(face).normal;
        assert_eq!(normal, Vec3::new(0.0, 0.0, 1.0));

        //             5        4
        //             ______________ 3 (2,1)
        //            /|        |    /
        //          /  |        |  /
        // (-1,0) /____|________ /
        //        0    1        2
        let verts = [
            Vec3::new(-1.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(2.0, 1.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        ];
        let face = create_face_from_verts(&mut mesh, &verts);
        mesh.face_calc_normal(face);
        let normal = mesh.faces.get(face).normal;
        assert_eq!(normal, Vec3::new(0.0, 0.0, 1.0));
    }

    #[test]
    fn test_calc_face_area() {
        let mut mesh = LoopMesh::new();

        let verts = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
        ];
        let face = create_face_from_verts(&mut mesh, &verts);
        let area = mesh.face_calc_area(face);
        assert_eq!(area, 0.5);

        let verts = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 2.0, 0.0),
            Vec3::new(0.0, 2.0, 2.0),
            Vec3::new(0.0, 0.0, 2.0),
        ];
        let face = create_face_from_verts(&mut mesh, &verts);
        let area = mesh.face_calc_area(face);
        assert_eq!(area, 4.0);

        //             5        4
        //             ______________ 3 (2,1)
        //            /|        |    /
        //          /  |        |  /
        // (-1,0) /____|________ /
        //        0    1        2
        let verts = [
            Vec3::new(-1.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(2.0, 1.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        ];
        let face = create_face_from_verts(&mut mesh, &verts);
        let area = mesh.face_calc_area(face);
        assert_eq!(area, 2.0);
    }
}
