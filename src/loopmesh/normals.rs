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

    /// Polynomial approximation of `acos` on `[-1, 1]`, in radians on `[0, π]`.
    ///
    /// Form (publicly known clamp + `sqrt(1 - |x|)` × polynomial, the standard
    /// extension of Abramowitz & Stegun 4.4.45 to one more term): for `x ≥ 0`,
    /// `acos(x) ≈ sqrt(1 - x) · P(x)` with `P` a degree-4 polynomial fitted
    /// over `[0, 1]`; the negative half uses the symmetry
    /// `acos(-x) = π - acos(x)`. The operand is clamped to `[-1, 1]` first, so
    /// the `sqrt` argument is never negative and the result never `NaN`.
    /// Maximum absolute error over `[-1, 1]` is ~1.97e-5 rad, below the guide's
    /// 4.5e-5 target (see `acos_approx_max_error` self-test).
    fn acos_approx(x: f32) -> f32 {
        let x = x.clamp(-1.0, 1.0);
        let abs_x = x.abs();
        // Horner evaluation of the degree-4 fit of acos(t)/sqrt(1-t) over [0,1].
        let poly = 1.570_776_8
            + abs_x
                * (-0.213_973_87
                    + abs_x * (0.084_163_88 + abs_x * (-0.035_166_71 + abs_x * 0.008_428_128)));
        let result = (1.0 - abs_x).sqrt() * poly;
        if x < 0.0 {
            core::f32::consts::PI - result
        } else {
            result
        }
    }

    /// Recompute a single vertex's normal as the angle-weighted sum of the
    /// incident faces' *stored* normals (Thürmer–Wüthrich weighting), one
    /// contribution per corner. There is no per-face dedup: a vertex visited by
    /// several corners of the same face contributes once per corner. When the
    /// accumulator normalises to zero — a wire/isolated vertex, fully cancelled
    /// contributions, or only degenerate incident faces — the stored normal is
    /// the vertex's own position treated as a direction from the origin
    /// (`normalize(position)`, and `(0,0,0)` at the origin).
    ///
    /// Note: callers should ensure face normals are current first (e.g. via
    /// [`Self::mesh_normals_update`]); each weight scales that face's stored
    /// `normal` field.
    pub fn vert_normal_update(&mut self, v: VertexKey) {
        let apex = self.vertices.get(v).position;
        let mut acc = Vec3::ZERO;
        // Per-corner gather: walk v's disk, and for every radial corner that
        // sits *at* v, weight that face's stored normal by the interior corner
        // angle. Lending iterators keep the walk allocation-free.
        let mut disk = self.disk_edges(v);
        while let Some(e) = disk.next(self) {
            // The *outgoing* edge from `v` along disk edge `e` is shared by
            // every radial corner this edge contributes at `v`, so its unit ray
            // is normalised once per disk edge — not per corner. The corner's
            // far endpoint is the loop's other vertex, i.e. `e`'s far endpoint.
            let far_vertex = match self.edges.get(e).other_vertex(v) {
                Some(w) => w,
                None => continue,
            };
            let out_ray =
                LoopMesh::normalize_or_zero(self.vertices.get(far_vertex).position - apex);
            let mut radial = self.radial_loops(e);
            while let Some(l) = radial.next(self) {
                let loop_data = self.loops.get(l);
                if loop_data.vertex != v {
                    continue;
                }
                let face = loop_data.face;
                // Only the loop-cycle predecessor edge's ray is normalised
                // per corner; `out_ray` is reused from above.
                let prev_vertex = self.loops.get(self.loop_prev(l)).vertex;
                let in_ray =
                    LoopMesh::normalize_or_zero(self.vertices.get(prev_vertex).position - apex);
                let dot = in_ray.dot(out_ray).clamp(-1.0, 1.0);
                let weight = LoopMesh::acos_approx(dot);
                acc += weight * self.faces.get(face).normal;
            }
        }
        let unit = LoopMesh::normalize_or_zero(acc);
        let normal = if unit != Vec3::ZERO {
            unit
        } else {
            LoopMesh::normalize_or_zero(apex)
        };
        self.set_vertex_normal(v, normal);
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

    /// The polynomial `acos` approximation stays under the guide's 4.5e-5 rad
    /// max-abs-error target across a dense sweep of `[-1, 1]`, and is exact at
    /// the clamp endpoints / midpoint (no NaN, monotone-ish).
    #[test]
    fn acos_approx_max_error() {
        let samples = 200_001;
        let mut max_err = 0.0_f32;
        for i in 0..samples {
            let x = -1.0 + 2.0 * (i as f32) / ((samples - 1) as f32);
            let approx = LoopMesh::acos_approx(x);
            assert!(approx.is_finite(), "acos_approx({x}) not finite");
            let err = (approx - x.acos()).abs();
            if err > max_err {
                max_err = err;
            }
        }
        assert!(
            max_err < 4.5e-5,
            "acos_approx max error {max_err} exceeds 4.5e-5 rad target"
        );
        // Out-of-domain operands are clamped, never NaN.
        assert!(LoopMesh::acos_approx(2.0).is_finite());
        assert!(LoopMesh::acos_approx(-2.0).is_finite());
        // A degenerate-corner dot of 0 weights at exactly π/2 (within tol).
        assert!((LoopMesh::acos_approx(0.0) - core::f32::consts::FRAC_PI_2).abs() < 4.5e-5);
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

    /// Two triangles share a single apex vertex. Both corners at the apex have
    /// the same interior angle (45°) but vastly different areas (10000:1). The
    /// vertex normal must bisect the two face normals — angle weighting cancels
    /// the area difference. Area weighting would collapse the result onto the
    /// large face's normal, so this guards the angle-weighting fix.
    #[test]
    fn vert_normal_is_angle_weighted_not_area_weighted() {
        let mut mesh = LoopMesh::new();

        let apex = mesh.create_vertex(Vec3::new(0.0, 0.0, 0.0));

        // Small face in plane z = 0, corner angle 45° at apex, normal +Z.
        let s = 0.1_f32;
        let b = mesh.create_vertex(Vec3::new(s, 0.0, 0.0));
        let c = mesh.create_vertex(Vec3::new(s, s, 0.0));
        let eab = mesh.create_edge([apex, b], EdgeCreateFlags::NONE);
        let ebc = mesh.create_edge([b, c], EdgeCreateFlags::NONE);
        let eca = mesh.create_edge([c, apex], EdgeCreateFlags::NONE);
        mesh.face_create(&[apex, b, c], &[eab, ebc, eca], FaceCreateFlags::NONE);

        // Large face in plane x = 0, corner angle 45° at apex, normal +X.
        let large = 10.0_f32;
        let d = mesh.create_vertex(Vec3::new(0.0, large, 0.0));
        let e = mesh.create_vertex(Vec3::new(0.0, large, large));
        let ead = mesh.create_edge([apex, d], EdgeCreateFlags::NONE);
        let ede = mesh.create_edge([d, e], EdgeCreateFlags::NONE);
        let eea = mesh.create_edge([e, apex], EdgeCreateFlags::NONE);
        mesh.face_create(&[apex, d, e], &[ead, ede, eea], FaceCreateFlags::NONE);

        mesh.mesh_normals_update();

        let n = mesh.vertex_normal(apex);
        let expected = Vec3::new(1.0, 0.0, 1.0).normalize();
        assert!(
            (n - expected).length() < 1.0e-4,
            "apex normal {n:?} should bisect the two face normals (got != {expected:?})"
        );
    }

    /// A vertex whose incident faces' normals cancel falls back to the vertex's
    /// own position normalised, not +Z or zero. Two coplanar triangles with
    /// opposite winding give equal-and-opposite normals at the shared apex.
    #[test]
    fn vert_normal_fallback_is_normalized_position() {
        let mut mesh = LoopMesh::new();

        let apex = mesh.create_vertex(Vec3::new(2.0, 0.0, 0.0));

        // Face 1: CCW in z = 0, normal +Z, 90° corner at apex.
        let b = mesh.create_vertex(Vec3::new(3.0, 0.0, 0.0));
        let c = mesh.create_vertex(Vec3::new(2.0, 1.0, 0.0));
        let e1 = mesh.create_edge([apex, b], EdgeCreateFlags::NONE);
        let e2 = mesh.create_edge([b, c], EdgeCreateFlags::NONE);
        let e3 = mesh.create_edge([c, apex], EdgeCreateFlags::NONE);
        mesh.face_create(&[apex, b, c], &[e1, e2, e3], FaceCreateFlags::NONE);

        // Face 2: CW in z = 0, normal -Z, same 90° corner at apex.
        let d = mesh.create_vertex(Vec3::new(1.0, 0.0, 0.0));
        let f = mesh.create_vertex(Vec3::new(2.0, -1.0, 0.0));
        let e4 = mesh.create_edge([apex, f], EdgeCreateFlags::NONE);
        let e5 = mesh.create_edge([f, d], EdgeCreateFlags::NONE);
        let e6 = mesh.create_edge([d, apex], EdgeCreateFlags::NONE);
        mesh.face_create(&[apex, f, d], &[e4, e5, e6], FaceCreateFlags::NONE);

        mesh.mesh_normals_update();

        // Normals cancel -> fallback = normalize(position) = +X.
        let n = mesh.vertex_normal(apex);
        assert!(
            (n - Vec3::new(1.0, 0.0, 0.0)).length() < 1.0e-5,
            "cancelled accumulator should fall back to normalize(position); got {n:?}"
        );
    }

    /// Near-cancellation (not exact-zero): the angle-weighted accumulator is a
    /// tiny-but-nonzero residual whose squared length sits below the
    /// `normalize_or_zero` guard. The old `length_squared() > 0.0` gate would
    /// have passed this residual into a raw `normalize()`, producing a
    /// wildly-scaled / non-finite normal; the guarded gate must instead take the
    /// `normalize(position)` fallback. Drive the vertex pass with two
    /// stored-normal contributions that nearly cancel.
    #[test]
    fn vert_normal_near_cancellation_uses_position_fallback() {
        let mut mesh = LoopMesh::new();

        let apex = mesh.create_vertex(Vec3::new(2.0, 0.0, 0.0));

        // Two coplanar triangles sharing the apex, 90° corner each, so both
        // gather weights are exactly π/2 and the accumulator is π/2 * (n1 + n2)
        // for whatever stored face normals we plant below.
        let b = mesh.create_vertex(Vec3::new(3.0, 0.0, 0.0));
        let c = mesh.create_vertex(Vec3::new(2.0, 1.0, 0.0));
        let e1 = mesh.create_edge([apex, b], EdgeCreateFlags::NONE);
        let e2 = mesh.create_edge([b, c], EdgeCreateFlags::NONE);
        let e3 = mesh.create_edge([c, apex], EdgeCreateFlags::NONE);
        let f1 = mesh.face_create(&[apex, b, c], &[e1, e2, e3], FaceCreateFlags::NONE);

        let d = mesh.create_vertex(Vec3::new(1.0, 0.0, 0.0));
        let g = mesh.create_vertex(Vec3::new(2.0, -1.0, 0.0));
        let e4 = mesh.create_edge([apex, g], EdgeCreateFlags::NONE);
        let e5 = mesh.create_edge([g, d], EdgeCreateFlags::NONE);
        let e6 = mesh.create_edge([d, apex], EdgeCreateFlags::NONE);
        let f2 = mesh.face_create(&[apex, g, d], &[e4, e5, e6], FaceCreateFlags::NONE);

        // Plant stored face normals that nearly cancel. f1 = (tiny, 0, +1),
        // f2 = (0, 0, -1); their sum is the residual (tiny, 0, 0). With both
        // gather weights π/2 the accumulator is (π/2 * tiny, 0, 0), whose
        // squared length ~6.2e-36 is strictly > 0 (passes the old `> 0.0`
        // gate) yet < the 1e-35 `normalize_or_zero` guard — so the guarded
        // path must reject it and take the position fallback. (Raw normalize()
        // of this residual would yield a non-finite / unit-X vector instead.)
        let tiny = 1.0e-18_f32;
        assert!(tiny > 0.0);
        mesh.faces.get_mut(f1).normal = Vec3::new(tiny, 0.0, 1.0);
        mesh.faces.get_mut(f2).normal = Vec3::new(0.0, 0.0, -1.0);

        mesh.vert_normal_update(apex);

        let n = mesh.vertex_normal(apex);
        // Residual after cancellation is below the guard -> position fallback.
        assert!(
            n.is_finite(),
            "near-cancelled accumulator must not produce a non-finite normal; got {n:?}"
        );
        assert!(
            (n.length() - 1.0).abs() < 1.0e-5,
            "fallback must be a unit vector; got length {} ({n:?})",
            n.length()
        );
        assert!(
            (n - Vec3::new(1.0, 0.0, 0.0)).length() < 1.0e-5,
            "near-cancelled accumulator should fall back to normalize(position) = +X; got {n:?}"
        );
    }

    /// A wire/isolated vertex (no faces) falls back to normalize(position)
    /// under the whole-mesh sweep, and a vertex at the origin yields (0,0,0).
    #[test]
    fn vert_normal_wire_vertex_uses_position_fallback() {
        let mut mesh = LoopMesh::new();
        let off_origin = mesh.create_vertex(Vec3::new(0.0, 0.0, 5.0));
        let at_origin = mesh.create_vertex(Vec3::new(0.0, 0.0, 0.0));

        mesh.mesh_normals_update();

        assert_eq!(mesh.vertex_normal(off_origin), Vec3::new(0.0, 0.0, 1.0));
        assert_eq!(mesh.vertex_normal(at_origin), Vec3::ZERO);
    }

    /// On a unit cube every vertex normal is the symmetric diagonal of its
    /// three incident faces — independent of weighting since all three corner
    /// angles are 90° and all three faces have equal area. Sanity check that the
    /// rebuilt gather still produces the expected closed-manifold result.
    #[test]
    fn vert_normal_cube_diagonals() {
        let mut mesh = LoopMesh::new();
        mesh_add_cube(&mut mesh);
        mesh.mesh_normals_update();

        let mut it = mesh.mesh_verts();
        while let Some(v) = it.next(&mesh) {
            let pos = mesh.vertices.get(v).position;
            let expected = pos.normalize(); // cube is centred at origin
            let n = mesh.vertex_normal(v);
            assert!(
                (n - expected).length() < 1.0e-5,
                "cube vertex {v:?} at {pos:?}: normal {n:?} != {expected:?}"
            );
        }
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
