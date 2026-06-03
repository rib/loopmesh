use glam::Mat4;

use crate::loopmesh::*;

/// Parameters for [`LoopMesh::transform`] (TRANSFORM.md §1).
///
/// Transform is a pure per-vertex coordinate edit: it applies a 4×4 affine map
/// to each named vert's position and touches nothing else (no topology, no
/// customdata, §5). The only derived state it refreshes is the cached normals
/// on the affected geometry (§5.1).
#[derive(Clone, Copy, Debug)]
pub struct TransformParams {
    /// The transform applied to each input vert's position, as a column-major
    /// affine 4×4 mapped over the homogeneous point `(x, y, z, 1)` — the
    /// translation column moves verts as well as the linear block
    /// rotating/scaling/shearing them (§3).
    pub matrix: Mat4,
    /// The frame the transform is applied *in* (§4). The all-zeros matrix
    /// ([`Mat4::ZERO`]) is the "no space" sentinel: apply `matrix` directly in
    /// world space. Any other value drives the conjugation
    /// `space⁻¹ · matrix · space`, so `matrix` happens expressed in the `space`
    /// frame (e.g. scale-about-a-point rather than scale-about-the-origin).
    pub space: Mat4,
    /// When set, the same matrix would also apply to each input vert's stored
    /// shape-key coordinates (§5.2). loopmesh has no shape-key model, so this is
    /// an accepted-but-no-effect deferral: the base position result is identical
    /// with the flag set or clear, and no topology or base position is changed.
    /// Mirrors the `use_shapekey` deferral on `mirror` / `symmetrize`.
    pub use_shapekey: bool,
}

/// Determinant magnitude at or below which a `space` matrix is treated as
/// singular and given a zeroed inverse rather than a NaN/inf-laden one
/// (TRANSFORM.md §4.1).
const SINGULAR_SPACE_EPSILON: f32 = 1e-12;

/// Invert `matrix`, clamping a singular (zero / near-zero determinant) input to
/// the all-zeros 4×4 instead of propagating a NaN/inf inverse (TRANSFORM.md
/// §4.1). A singular `space` fed through the conjugation therefore collapses the
/// effective transform to the zero matrix, sending every named vert to the
/// origin — a finite result, never NaN.
fn invert_or_zero(matrix: Mat4) -> Mat4 {
    if matrix.determinant().abs() <= SINGULAR_SPACE_EPSILON {
        Mat4::ZERO
    } else {
        matrix.inverse()
    }
}

impl LoopMesh {
    /// Apply a 4×4 affine matrix to the positions of `verts` (TRANSFORM.md §6).
    ///
    /// Each named vert's position is overwritten with the effective-transform
    /// image of its current position (§3). The effective transform is `matrix`
    /// when `space` is the all-zeros [`Mat4::ZERO`] sentinel (world space, §4),
    /// otherwise the conjugation `space⁻¹ · matrix · space` — `space` applied
    /// innermost, so a `+c` translation frame puts a scale's fixed point at `−c`
    /// (§4's sign trap).
    ///
    /// The transform is applied independently per vert: no ordering dependency,
    /// no closure (naming a subset of a face's corners moves only those corners,
    /// distorting the face, §7). Topology and customdata are untouched — every
    /// element identity, incidence, loop cycle, and CD byte survives unchanged
    /// (§5.2). A negative-determinant matrix reflects positions and flips the
    /// geometric normals but **never rewinds a loop cycle**; the combinatorial
    /// winding is invariant under transform (§3.1). A singular matrix is applied
    /// as given, collapsing the named verts onto a plane/line/point — there is no
    /// guard against degeneracy (§7).
    ///
    /// After moving the verts, the cached face and vertex normals on the affected
    /// geometry are recomputed from the new positions (§5.1), matching loopmesh's
    /// eager-recompute model for position-mutating ops. Empty `verts` is a no-op
    /// (§7).
    ///
    /// See `radial-mesh-guides/operators/TRANSFORM.md` for the full algorithm and
    /// parity contract.
    pub fn transform(&mut self, verts: &[VertexKey], params: TransformParams) {
        if verts.is_empty() {
            return;
        }

        // §4 — compose the effective world-space transform. The all-zeros
        // `space` sentinel means "no space": apply `matrix` directly without
        // inverting the singular zero matrix or multiplying by it (which would
        // collapse every vert to the origin).
        let effective = if params.space == Mat4::ZERO {
            params.matrix
        } else {
            // §4.1 — a non-zero but singular `space` is NOT the sentinel, so it
            // takes the conjugation branch. Its inverse is the zeroed 4×4 by
            // convention (never a NaN/inf-laden result), so the product collapses
            // to the zero matrix and every named vert maps to the origin.
            invert_or_zero(params.space) * params.matrix * params.space
        };

        // §3 — apply the effective affine map to each named vert's homogeneous
        // position. Independent per vert; `transform_point3` includes the
        // translation column.
        for &vert in verts {
            let position = self.vertex_position(vert);
            self.set_vertex_position(vert, effective.transform_point3(position));
        }

        // §5.1 — refresh the cached normals on the affected geometry: every face
        // incident to a moved vert, then those faces' verts. No topology and no
        // customdata is touched.
        self.refresh_normals_around(verts);
    }

    /// Recompute cached face normals on every face incident to a moved vert, and
    /// vertex normals on every vert of those faces (§5.1). Scoped to the affected
    /// geometry rather than the whole mesh so the cost tracks the selection.
    fn refresh_normals_around(&mut self, verts: &[VertexKey]) {
        let mut affected_faces: smallvec::SmallVec<[FaceKey; 16]> = smallvec::SmallVec::new();
        for &vert in verts {
            let mut disk = self.disk_edges(vert);
            while let Some(edge) = disk.next(self) {
                let mut radial = self.radial_loops(edge);
                while let Some(loop_key) = radial.next(self) {
                    let face = self.loops.get(loop_key).face;
                    if !affected_faces.contains(&face) {
                        affected_faces.push(face);
                    }
                }
            }
        }

        let mut affected_verts: smallvec::SmallVec<[VertexKey; 32]> = smallvec::SmallVec::new();
        for &face in &affected_faces {
            self.face_calc_normal(face);
            for vert in self.face_vert_keys(face) {
                if !affected_verts.contains(&vert) {
                    affected_verts.push(vert);
                }
            }
        }
        for &vert in &affected_verts {
            self.vert_normal_update(vert);
        }
    }
}

#[cfg(test)]
mod tests {
    use glam::{Mat4, Quat, Vec3};

    use crate::loopmesh::*;

    /// A unit quad in the `z = 0` plane: verts at (0,0),(1,0),(1,1),(0,1), one
    /// face wound `0→1→2→3`. Returns the four verts in that order and the face.
    fn unit_quad(mesh: &mut LoopMesh) -> ([VertexKey; 4], FaceKey) {
        let v0 = mesh.create_vertex(Vec3::new(0.0, 0.0, 0.0));
        let v1 = mesh.create_vertex(Vec3::new(1.0, 0.0, 0.0));
        let v2 = mesh.create_vertex(Vec3::new(1.0, 1.0, 0.0));
        let v3 = mesh.create_vertex(Vec3::new(0.0, 1.0, 0.0));
        let verts = [v0, v1, v2, v3];
        let edges: Vec<EdgeKey> = (0..4)
            .map(|i| mesh.create_edge([verts[i], verts[(i + 1) % 4]], EdgeCreateFlags::NO_DUPLICATE))
            .collect();
        let face = mesh.face_create(&verts, &edges, FaceCreateFlags::NONE);
        (verts, face)
    }

    fn params(matrix: Mat4) -> TransformParams {
        TransformParams {
            matrix,
            space: Mat4::ZERO,
            use_shapekey: false,
        }
    }

    fn assert_close(a: Vec3, b: Vec3) {
        assert!(
            (a - b).length() < 1e-5,
            "expected {b:?}, got {a:?} (delta {})",
            (a - b).length()
        );
    }

    /// §2 / §3: a combined translate + scale + rotate maps every vert to the
    /// homogeneous-point image of its input — the headline full-affine pin. Each
    /// vert must land exactly at `matrix.transform_point3(p)`.
    #[test]
    fn affine_translate_scale_rotate_moves_verts() {
        let mut mesh = LoopMesh::new();
        let (verts, face) = unit_quad(&mut mesh);
        let before: Vec<Vec3> = verts.iter().map(|&v| mesh.vertex_position(v)).collect();

        let matrix = Mat4::from_scale_rotation_translation(
            Vec3::new(2.0, 0.5, 1.5),
            Quat::from_rotation_z(std::f32::consts::FRAC_PI_6),
            Vec3::new(3.0, -1.0, 2.0),
        );

        mesh.transform(&verts, params(matrix));

        for (i, &v) in verts.iter().enumerate() {
            assert_close(mesh.vertex_position(v), matrix.transform_point3(before[i]));
        }
        // Topology untouched: still one quad, four verts.
        assert_eq!((mesh.totvert(), mesh.totedge(), mesh.totface()), (4, 4, 1));
        assert_eq!(mesh.face_vert_keys(face).len(), 4);
        mesh.validate_topology();
    }

    /// §3: a translation-only matrix honours the translation column — the verts
    /// shift by exactly the translation vector.
    #[test]
    fn pure_translation_honours_translation_column() {
        let mut mesh = LoopMesh::new();
        let (verts, _face) = unit_quad(&mut mesh);
        let before: Vec<Vec3> = verts.iter().map(|&v| mesh.vertex_position(v)).collect();

        let shift = Vec3::new(5.0, -2.0, 1.0);
        mesh.transform(&verts, params(Mat4::from_translation(shift)));

        for (i, &v) in verts.iter().enumerate() {
            assert_close(mesh.vertex_position(v), before[i] + shift);
        }
        mesh.validate_topology();
    }

    /// §3: a diagonal scale about the origin scales the linear block, no
    /// translation applied.
    #[test]
    fn pure_scale_about_origin() {
        let mut mesh = LoopMesh::new();
        let (verts, _face) = unit_quad(&mut mesh);
        let before: Vec<Vec3> = verts.iter().map(|&v| mesh.vertex_position(v)).collect();

        let scale = Vec3::new(2.0, 3.0, 0.5);
        mesh.transform(&verts, params(Mat4::from_scale(scale)));

        for (i, &v) in verts.iter().enumerate() {
            assert_close(mesh.vertex_position(v), before[i] * scale);
        }
        mesh.validate_topology();
    }

    /// §7: the identity matrix is a no-op on positions.
    #[test]
    fn identity_is_noop() {
        let mut mesh = LoopMesh::new();
        let (verts, _face) = unit_quad(&mut mesh);
        let before: Vec<Vec3> = verts.iter().map(|&v| mesh.vertex_position(v)).collect();

        mesh.transform(&verts, params(Mat4::IDENTITY));

        for (i, &v) in verts.iter().enumerate() {
            assert_close(mesh.vertex_position(v), before[i]);
        }
        mesh.validate_topology();
    }

    /// §7: empty input is a no-op even with a non-identity matrix; no position
    /// or topology changes.
    #[test]
    fn empty_input_is_noop() {
        let mut mesh = LoopMesh::new();
        let (verts, _face) = unit_quad(&mut mesh);
        let before: Vec<Vec3> = verts.iter().map(|&v| mesh.vertex_position(v)).collect();
        let counts = (mesh.totvert(), mesh.totedge(), mesh.totface());

        mesh.transform(&[], params(Mat4::from_translation(Vec3::new(9.0, 9.0, 9.0))));

        for (i, &v) in verts.iter().enumerate() {
            assert_close(mesh.vertex_position(v), before[i]);
        }
        assert_eq!((mesh.totvert(), mesh.totedge(), mesh.totface()), counts);
        mesh.validate_topology();
    }

    /// §5.2: a pure position edit touches no customdata layer. A vert weight
    /// layer and a loop UV layer must survive a non-trivial transform unchanged.
    #[test]
    fn customdata_is_untouched() {
        let mut mesh = LoopMesh::new();
        let (verts, face) = unit_quad(&mut mesh);

        let weight = mesh
            .register_vertex_attribute(ElementAttributeDesc::numeric(
                AttributeKey::custom("weight"),
                Format::new(ComponentType::F32, 1),
            ))
            .expect("vert weight layer registers");
        for (i, &v) in verts.iter().enumerate() {
            mesh.vertex_attribs_mut()
                .set_f32(weight, v.as_usize(), 0.1 * (i as f32 + 1.0));
        }

        let uv = mesh
            .register_loop_attribute(ElementAttributeDesc::numeric(
                AttributeKey::custom("uv"),
                Format::new(ComponentType::F32, 2),
            ))
            .expect("uv layer registers");
        let loop_uvs: Vec<(LoopKey, [f32; 2])> = mesh
            .face_loop_keys(face)
            .into_iter()
            .enumerate()
            .map(|(i, l)| {
                let value = [0.1 * i as f32, 0.2 * i as f32];
                mesh.loop_attribs_mut().set_vec2(uv, l.as_usize(), value);
                (l, value)
            })
            .collect();

        let matrix = Mat4::from_scale_rotation_translation(
            Vec3::new(1.5, 2.0, 0.75),
            Quat::from_rotation_y(std::f32::consts::FRAC_PI_4),
            Vec3::new(-1.0, 2.0, 0.5),
        );
        mesh.transform(&verts, params(matrix));

        for (i, &v) in verts.iter().enumerate() {
            let w = mesh.vertex_attribs().get_f32(weight, v.as_usize()).unwrap();
            assert!((w - 0.1 * (i as f32 + 1.0)).abs() < 1e-6, "vert weight survives");
        }
        for (l, before) in loop_uvs {
            let after = mesh.loop_attribs().get_vec2(uv, l.as_usize()).unwrap();
            assert_eq!(after, before, "loop UV survives the transform");
        }
        mesh.validate_topology();
    }

    /// §4 the sign trap: `matrix` = scale ×2 about the origin, `space` =
    /// "translate by c". Because `space` is applied innermost, the effective map
    /// is `p ↦ 2·p + c` (scale ×2 about the fixed point `−c`, NOT `+c`). Pins the
    /// conjugation centre's sign: a known vert must land at its `2·p + c` image.
    #[test]
    fn space_conjugation_centre_sign() {
        let mut mesh = LoopMesh::new();
        let (verts, _face) = unit_quad(&mut mesh);
        let before: Vec<Vec3> = verts.iter().map(|&v| mesh.vertex_position(v)).collect();

        let c = Vec3::new(4.0, 0.0, 0.0);
        mesh.transform(
            &verts,
            TransformParams {
                matrix: Mat4::from_scale(Vec3::splat(2.0)),
                space: Mat4::from_translation(c),
                use_shapekey: false,
            },
        );

        // Effective map p ↦ 2·p + c (centre at −c). A wrong-sign implementation
        // (centre at +c, i.e. p ↦ 2·p − c) would fail here.
        for (i, &v) in verts.iter().enumerate() {
            assert_close(mesh.vertex_position(v), 2.0 * before[i] + c);
        }

        // The fixed point is at −c: a vert placed there would not move. Verify
        // the map's fixed point analytically: 2·(−c) + c = −c.
        let fixed = -c;
        assert_close(2.0 * fixed + c, fixed);
        mesh.validate_topology();
    }

    /// §4.1: a non-zero but singular `space` (e.g. `diag(1,1,0)`, det 0) is NOT
    /// the all-zeros sentinel, so the conjugation branch runs with a zeroed
    /// inverse — collapsing every named vert to a finite origin `(0,0,0)`, never
    /// to NaN. A naive `space.inverse()` would write NaN here.
    #[test]
    fn singular_nonzero_space_collapses_to_origin() {
        let mut mesh = LoopMesh::new();
        let (verts, _face) = unit_quad(&mut mesh);

        // diag(1,1,0): non-zero entries (not the sentinel) but determinant 0.
        let singular_space = Mat4::from_scale(Vec3::new(1.0, 1.0, 0.0));
        assert_ne!(singular_space, Mat4::ZERO, "fixture must not be the sentinel");
        assert_eq!(singular_space.determinant(), 0.0, "fixture must be singular");

        // A non-trivial matrix: the convention must collapse to the origin
        // regardless of `matrix` (and regardless of each vert's input position).
        let matrix = Mat4::from_scale_rotation_translation(
            Vec3::new(2.0, 0.5, 1.5),
            Quat::from_rotation_z(std::f32::consts::FRAC_PI_3),
            Vec3::new(3.0, -1.0, 2.0),
        );

        mesh.transform(
            &verts,
            TransformParams {
                matrix,
                space: singular_space,
                use_shapekey: false,
            },
        );

        for &v in &verts {
            let position = mesh.vertex_position(v);
            assert!(
                position.is_finite(),
                "singular space must yield finite coords, got {position:?}"
            );
            assert_close(position, Vec3::ZERO);
        }
        mesh.validate_topology();
    }

    /// §3 / §7: a partial vert selection moves only the named corners — no
    /// closure pulls in the rest of the face. The un-named verts stay put.
    #[test]
    fn partial_selection_moves_only_named_verts() {
        let mut mesh = LoopMesh::new();
        let (verts, _face) = unit_quad(&mut mesh);
        let before: Vec<Vec3> = verts.iter().map(|&v| mesh.vertex_position(v)).collect();

        let shift = Vec3::new(10.0, 0.0, 0.0);
        // Name only the first two corners.
        mesh.transform(&verts[..2], params(Mat4::from_translation(shift)));

        assert_close(mesh.vertex_position(verts[0]), before[0] + shift);
        assert_close(mesh.vertex_position(verts[1]), before[1] + shift);
        assert_close(mesh.vertex_position(verts[2]), before[2]);
        assert_close(mesh.vertex_position(verts[3]), before[3]);
        // Still one face — no topology change despite the sheared geometry.
        assert_eq!(mesh.totface(), 1);
        mesh.validate_topology();
    }
}
