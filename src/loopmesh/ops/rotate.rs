use glam::{Mat4, Vec3};

use crate::loopmesh::*;

/// Parameters for [`LoopMesh::rotate`] (ROTATE.md §1).
///
/// Rotate is the (rotation, centre)-shaped front door to
/// [`LoopMesh::transform`]: it spins each named vert about an axis through
/// `cent`, leaving topology, customdata, and all relative geometry untouched
/// (§2). It reduces exactly to `transform` fed the **pivot-recentred** rotation
/// matrix `T(cent)·R·T(-cent)` — the rotation `R` in the linear block, `cent -
/// R·cent` in the translation column — so it forwards `space` and
/// `use_shapekey` to transform unchanged.
#[derive(Clone, Copy, Debug)]
pub struct RotateParams {
    /// The rotation applied to each input vert's position, expressed as a 4×4
    /// whose **linear (upper-left 3×3) block is the rotation** and whose
    /// translation column is identity/zero (§1). A caller thinking in axis +
    /// angle builds this matrix (e.g. quaternion-to-matrix) before handing it
    /// to rotate; rotate then recentres it about `cent` (§2). Same column-major
    /// layout as transform's `matrix`.
    pub matrix: Mat4,
    /// The **centre of rotation** — the point left fixed by the operator (§1).
    /// Rotate recentres `matrix` about this point so a vert at `cent` does not
    /// move and every other vert spins about the axis through `cent`. The
    /// default `Vec3::ZERO` rotates about the axis through the world origin.
    pub cent: Vec3,
    /// The frame the rotation is applied *in* (§3), identical in meaning to
    /// transform's `space`. The all-zeros [`Mat4::ZERO`] sentinel means world
    /// space. Unlike translate, rotate's recentred matrix has a non-trivial
    /// linear block, so `space` is genuinely observable: a non-trivial frame
    /// re-expresses the whole rotation-about-`cent` via the conjugation
    /// `space⁻¹ · matrix_pivoted · space`.
    pub space: Mat4,
    /// Forwarded to transform unchanged (§1). loopmesh has no shape-key model,
    /// so this is an accepted-but-no-effect deferral — the base position result
    /// is identical with the flag set or clear.
    pub use_shapekey: bool,
}

impl LoopMesh {
    /// Rotate the positions of `verts` about an axis through `cent` (ROTATE.md §2).
    ///
    /// Rotate is [`LoopMesh::transform`] restricted to a rotation matrix and
    /// recentred about `cent`. It builds the **pivot recentre**
    /// `T(cent)·R·T(-cent)` — the rotation `R` in the linear block, the pivot
    /// offset `cent - R·cent` in the translation column — and forwards it
    /// through the transform path with the same `space` and `use_shapekey`. A
    /// vert at `p` lands at `R·(p - cent) + cent` in the no-space case; the
    /// fixed point is exactly `cent` (a vert there does not move). With a
    /// non-trivial linear-block `space` the whole rotation is re-expressed
    /// through that frame (§3).
    ///
    /// Topology and customdata are untouched — rotate moves positions only and
    /// inherits transform's contract byte-for-byte (§2). The affected normals
    /// are recomputed from the new positions (rotated by `R`), but the
    /// combinatorial winding is never touched (a rotation has determinant +1,
    /// never a reflection, §2). Empty `verts` and an identity / zero-angle
    /// `matrix` are both accepted no-ops, the latter for *any* `cent` since
    /// `T(cent)·I·T(-cent) = I` (§4).
    ///
    /// See `radial-mesh-guides/operators/ROTATE.md` for the full reduction and
    /// parity contract.
    pub fn rotate(&mut self, verts: &[VertexKey], params: RotateParams) {
        // §2 — pivot recentre: set the translation column to `cent - R·cent`
        // so the rotation fixes `cent` instead of the world origin. `matrix`
        // carries the rotation in its linear block with a zero translation
        // column, so left-multiplying by the pure translation `T(cent - R·cent)`
        // simply installs that column: the result is `p ↦ R·p + (cent - R·cent)`.
        let pivot_offset = params.cent - params.matrix.transform_vector3(params.cent);
        let matrix_pivoted = Mat4::from_translation(pivot_offset) * params.matrix;

        self.transform(
            verts,
            TransformParams {
                matrix: matrix_pivoted,
                space: params.space,
                use_shapekey: params.use_shapekey,
            },
        );
    }
}

#[cfg(test)]
mod tests {
    use glam::{Mat4, Quat, Vec3};

    use crate::loopmesh::*;

    /// A unit cube spanning `[0,1]³`, eight verts, six quad faces. Returns the
    /// eight vertex keys in corner order.
    fn unit_cube(mesh: &mut LoopMesh) -> Vec<VertexKey> {
        let corners = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(1.0, 0.0, 1.0),
            Vec3::new(1.0, 1.0, 1.0),
            Vec3::new(0.0, 1.0, 1.0),
        ];
        let verts: Vec<VertexKey> = corners.iter().map(|&p| mesh.create_vertex(p)).collect();

        let quads = [
            [0usize, 3, 2, 1], // bottom (z = 0)
            [4, 5, 6, 7],      // top    (z = 1)
            [0, 1, 5, 4],      // front  (y = 0)
            [2, 3, 7, 6],      // back   (y = 1)
            [1, 2, 6, 5],      // right  (x = 1)
            [0, 4, 7, 3],      // left   (x = 0)
        ];
        for quad in quads {
            let face_verts: Vec<VertexKey> = quad.iter().map(|&i| verts[i]).collect();
            let face_edges: Vec<EdgeKey> = (0..4)
                .map(|i| {
                    mesh.create_edge(
                        [face_verts[i], face_verts[(i + 1) % 4]],
                        EdgeCreateFlags::NO_DUPLICATE,
                    )
                })
                .collect();
            mesh.face_create(&face_verts, &face_edges, FaceCreateFlags::NONE);
        }
        verts
    }

    fn params(matrix: Mat4, cent: Vec3) -> RotateParams {
        RotateParams {
            matrix,
            cent,
            space: Mat4::ZERO,
            use_shapekey: false,
        }
    }

    fn assert_close(actual: Vec3, expected: Vec3) {
        assert!(
            (actual - expected).length() < 1e-5,
            "expected {expected:?}, got {actual:?} (delta {})",
            (actual - expected).length()
        );
    }

    /// §2 / §5 headline: a rotation about the origin (`cent = origin`) maps every
    /// named vert to `R·p` and leaves the cube's topology invariant.
    #[test]
    fn rotation_about_origin_maps_each_vert_to_r_p() {
        let mut mesh = LoopMesh::new();
        let verts = unit_cube(&mut mesh);
        let before: Vec<Vec3> = verts.iter().map(|&v| mesh.vertex_position(v)).collect();

        let rotation = Mat4::from_quat(Quat::from_rotation_z(std::f32::consts::FRAC_PI_3));
        mesh.rotate(&verts, params(rotation, Vec3::ZERO));

        for (corner, &vert) in verts.iter().enumerate() {
            // cent = origin, so the pivot offset is zero: p ↦ R·p.
            assert_close(
                mesh.vertex_position(vert),
                rotation.transform_point3(before[corner]),
            );
        }
        assert_eq!((mesh.totvert(), mesh.totedge(), mesh.totface()), (8, 12, 6));
        mesh.validate_topology();
    }

    /// §2 / §4: a rotation about a non-origin `cent` fixes `cent` (a vert there
    /// stays put) and maps every other vert to `R·(p - cent) + cent`. The result
    /// differs from the origin-centred rotation — the pivot-recentre pin.
    #[test]
    fn rotation_about_center_fixes_center() {
        let mut mesh = LoopMesh::new();
        let verts = unit_cube(&mut mesh);
        let before: Vec<Vec3> = verts.iter().map(|&v| mesh.vertex_position(v)).collect();

        // Pick a cent that coincides with a vertex so we can pin "stays put":
        // vert 6 at (1,1,1) is the fixed centre.
        let cent = before[6];
        let rotation = Mat4::from_quat(Quat::from_rotation_z(std::f32::consts::FRAC_PI_3));
        mesh.rotate(&verts, params(rotation, cent));

        // The vert sitting exactly at cent does not move.
        assert_close(mesh.vertex_position(verts[6]), cent);

        // Every vert lands at R·(p - cent) + cent.
        for (corner, &vert) in verts.iter().enumerate() {
            let expected = rotation.transform_vector3(before[corner] - cent) + cent;
            assert_close(mesh.vertex_position(vert), expected);
        }

        // The recentred result differs from a bare origin-centred rotation for at
        // least one vert (the pivot recentre is genuinely consulted).
        let origin_image = rotation.transform_point3(before[0]);
        assert!(
            (mesh.vertex_position(verts[0]) - origin_image).length() > 1e-4,
            "cent != origin must differ from the origin-centred rotation"
        );

        assert_eq!((mesh.totvert(), mesh.totedge(), mesh.totface()), (8, 12, 6));
        mesh.validate_topology();
    }

    /// §3: a non-trivial `space` re-expresses the rotation in that frame via the
    /// conjugation `space⁻¹ · matrix_pivoted · space`. Observable because
    /// rotate's matrix has a non-trivial linear block — the result differs from
    /// the no-space run.
    #[test]
    fn space_reexpresses_rotation() {
        let rotation = Mat4::from_quat(Quat::from_rotation_z(std::f32::consts::FRAC_PI_4));
        let space = Mat4::from_quat(Quat::from_rotation_x(std::f32::consts::FRAC_PI_2));

        let mut mesh_no_space = LoopMesh::new();
        let verts_no_space = unit_cube(&mut mesh_no_space);
        let inputs: Vec<Vec3> = verts_no_space
            .iter()
            .map(|&v| mesh_no_space.vertex_position(v))
            .collect();
        mesh_no_space.rotate(&verts_no_space, params(rotation, Vec3::ZERO));

        let mut mesh_space = LoopMesh::new();
        let verts_space = unit_cube(&mut mesh_space);
        mesh_space.rotate(
            &verts_space,
            RotateParams {
                matrix: rotation,
                cent: Vec3::ZERO,
                space,
                use_shapekey: false,
            },
        );

        // The conjugation must reproduce `space⁻¹ · R · space` applied per vert.
        let effective = space.inverse() * rotation * space;
        let mut differs = false;
        for (corner, &vert) in verts_space.iter().enumerate() {
            assert_close(
                mesh_space.vertex_position(vert),
                effective.transform_point3(inputs[corner]),
            );
            if (mesh_space.vertex_position(vert)
                - mesh_no_space.vertex_position(verts_no_space[corner]))
            .length()
                > 1e-4
            {
                differs = true;
            }
        }
        assert!(
            differs,
            "a non-trivial space must change the result vs the no-space run"
        );
        mesh_space.validate_topology();
    }

    /// §4: the identity (zero-angle) rotation is a no-op on positions for *any*
    /// `cent` — recentring the identity yields the identity.
    #[test]
    fn identity_is_noop_for_any_center() {
        for cent in [Vec3::ZERO, Vec3::new(3.0, -2.0, 5.0)] {
            let mut mesh = LoopMesh::new();
            let verts = unit_cube(&mut mesh);
            let before: Vec<Vec3> = verts.iter().map(|&v| mesh.vertex_position(v)).collect();

            mesh.rotate(&verts, params(Mat4::IDENTITY, cent));

            for (corner, &vert) in verts.iter().enumerate() {
                assert_close(mesh.vertex_position(vert), before[corner]);
            }
            assert_eq!((mesh.totvert(), mesh.totedge(), mesh.totface()), (8, 12, 6));
            mesh.validate_topology();
        }
    }

    /// §5 / §8: a rotation is a pure position edit, so it leaves every customdata
    /// layer byte-identical on every element — sibling of translate's and
    /// transform's CD-untouched pins. A vert weight layer and a loop UV layer must
    /// survive a non-trivial rotate (about a non-origin centre) unchanged.
    #[test]
    fn customdata_is_untouched() {
        let mut mesh = LoopMesh::new();
        let verts = unit_cube(&mut mesh);

        let weight = mesh
            .register_vertex_attribute(ElementAttributeDesc::numeric(
                AttributeKey::custom("weight"),
                AttrType::F32,
            ))
            .expect("vert weight layer registers");
        for (corner, &vert) in verts.iter().enumerate() {
            mesh.vertex_attribs_mut()
                .set_f32(weight, vert.as_usize(), 0.1 * (corner as f32 + 1.0));
        }

        let uv = mesh
            .register_loop_attribute(ElementAttributeDesc::numeric(
                AttributeKey::custom("uv"),
                AttrType::Vec2,
            ))
            .expect("uv layer registers");
        let face = mesh.face_keys()[0];
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

        let rotation = Mat4::from_quat(Quat::from_rotation_z(std::f32::consts::FRAC_PI_3));
        mesh.rotate(&verts, params(rotation, Vec3::new(1.0, 1.0, 1.0)));

        for (corner, &vert) in verts.iter().enumerate() {
            let stored = mesh
                .vertex_attribs()
                .get_f32(weight, vert.as_usize())
                .unwrap();
            assert!(
                (stored - 0.1 * (corner as f32 + 1.0)).abs() < 1e-6,
                "vert weight survives the rotate"
            );
        }
        for (l, before) in loop_uvs {
            let after = mesh.loop_attribs().get_vec2(uv, l.as_usize()).unwrap();
            assert_eq!(after, before, "loop UV survives the rotate");
        }
        assert_eq!((mesh.totvert(), mesh.totedge(), mesh.totface()), (8, 12, 6));
        mesh.validate_topology();
    }

    /// §4: empty input is an accepted no-op even with a non-trivial rotation and
    /// centre — no position or topology change.
    #[test]
    fn empty_input_is_noop() {
        let mut mesh = LoopMesh::new();
        let verts = unit_cube(&mut mesh);
        let before: Vec<Vec3> = verts.iter().map(|&v| mesh.vertex_position(v)).collect();
        let counts = (mesh.totvert(), mesh.totedge(), mesh.totface());

        let rotation = Mat4::from_quat(Quat::from_rotation_y(std::f32::consts::FRAC_PI_4));
        mesh.rotate(&[], params(rotation, Vec3::new(2.0, 2.0, 2.0)));

        for (corner, &vert) in verts.iter().enumerate() {
            assert_close(mesh.vertex_position(vert), before[corner]);
        }
        assert_eq!((mesh.totvert(), mesh.totedge(), mesh.totface()), counts);
        mesh.validate_topology();
    }
}
