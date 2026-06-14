use glam::{Mat4, Vec3};

use crate::loopmesh::*;

/// Parameters for [`LoopMesh::scale`] (SCALE.md §1).
///
/// Scale is the factor-vector-shaped front door to [`LoopMesh::transform`]: it
/// stretches or shrinks each named vert by a per-axis factor `vec`, leaving
/// topology, customdata, and all relative incidence untouched (§2). It reduces
/// exactly to `transform` fed the **bare** diagonal matrix `diag(vec.x, vec.y,
/// vec.z)` — the factors in the linear block, a **zero translation column** that
/// fixes the world origin — so it forwards `space` and `use_shapekey` to
/// transform unchanged.
///
/// Unlike [`RotateParams`], scale carries **no `cent` parameter** and performs
/// no pivot recentre of its own (§1): the diagonal matrix is a pure linear map
/// that scales about the origin, and a caller relocates that centre entirely
/// through a translation `space` frame (§3).
#[derive(Clone, Copy, Debug)]
pub struct ScaleParams {
    /// The **per-axis scale factor** applied to each input vert's position (§1):
    /// `vec.x` scales `x`, `vec.y` scales `y`, `vec.z` scales `z`. Scale builds
    /// the diagonal `diag(vec.x, vec.y, vec.z)` with a zero translation column,
    /// so a vert at `p` lands at `(vec.x·p.x, vec.y·p.y, vec.z·p.z)` in the
    /// no-space case (§2). A uniform scale is the special case `vec = (s, s, s)`;
    /// the default `(1, 1, 1)` is the identity (a no-op). Negative entries
    /// reflect, zero entries collapse onto the complementary subspace — both
    /// applied as given, no guard (§4).
    pub vec: Vec3,
    /// The frame the scale is applied *in* (§3), identical in meaning to
    /// transform's `space`. The all-zeros [`Mat4::ZERO`] sentinel means world
    /// space. Because scale's matrix has a non-trivial linear block, `space` is
    /// genuinely observable (like rotate, unlike translate): a non-trivial frame
    /// re-expresses the whole scale via the conjugation `space⁻¹ · matrix ·
    /// space`. This is also the **only** way to recentre the scale about a point
    /// — a `+c` translation frame scales about `−c` (§3's sign trap).
    pub space: Mat4,
    /// Forwarded to transform unchanged (§1). loopmesh has no shape-key model, so
    /// this is an accepted-but-no-effect deferral — the base position result is
    /// identical with the flag set or clear.
    pub use_shapekey: bool,
}

impl LoopMesh {
    /// Scale the positions of `verts` by a per-axis factor `vec` (SCALE.md §2).
    ///
    /// Scale is [`LoopMesh::transform`] restricted to a diagonal scale matrix. It
    /// builds the **bare** `diag(vec.x, vec.y, vec.z)` — the factors in the
    /// linear block, a zero translation column — and forwards it through the
    /// transform path with the same `space` and `use_shapekey`. A vert at `p`
    /// lands at `(vec.x·p.x, vec.y·p.y, vec.z·p.z)` in the no-space case, scaling
    /// about the **world origin**; with a translation `space` the scale is
    /// recentred about a point (§3). Scale performs **no** pivot recentre of its
    /// own — there is no `cent` parameter, unlike rotate (§1).
    ///
    /// Topology and customdata are untouched — scale moves positions only and
    /// inherits transform's contract byte-for-byte (§2). The affected normals are
    /// recomputed from the new positions; a non-uniform or negative scale
    /// genuinely changes the geometry's shape, but the combinatorial winding is
    /// never touched — scale rewinds no face even under a negative-determinant
    /// (reflecting) factor (§4). A zero factor collapses the named verts onto the
    /// complementary plane/line/point **in place**, applied as given with no
    /// guard — a finite result, distinct from a singular `space` collapse onto
    /// the origin (§3, §4). Empty `verts` and an identity `vec = (1, 1, 1)` are
    /// both accepted no-ops (§4).
    ///
    /// See `radial-mesh-guides/operators/SCALE.md` for the full reduction and
    /// parity contract.
    pub fn scale(&mut self, verts: &[VertexKey], params: ScaleParams) {
        // §2 — the bare diagonal: factors in the linear block, zero translation
        // column. `Mat4::from_scale` builds exactly `diag(vec.x, vec.y, vec.z)`
        // with an identity (zero-offset) translation, so the bare scale fixes the
        // world origin and any recentre is carried by `space` (§3), not `matrix`.
        self.transform(
            verts,
            TransformParams {
                matrix: Mat4::from_scale(params.vec),
                space: params.space,
                use_shapekey: params.use_shapekey,
            },
        );
    }
}

#[cfg(test)]
mod tests {
    use glam::{Mat4, Vec3};

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

    fn params(vec: Vec3) -> ScaleParams {
        ScaleParams {
            vec,
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

    /// §2 / §5 headline: a uniform factor `(s, s, s)` maps every named vert to
    /// `s·p` and leaves the cube's topology invariant.
    #[test]
    fn uniform_scale_multiplies_every_vert() {
        let mut mesh = LoopMesh::new();
        let verts = unit_cube(&mut mesh);
        let before: Vec<Vec3> = verts.iter().map(|&v| mesh.vertex_position(v)).collect();

        let factor = 3.0;
        mesh.scale(&verts, params(Vec3::splat(factor)));

        for (corner, &vert) in verts.iter().enumerate() {
            assert_close(mesh.vertex_position(vert), before[corner] * factor);
        }
        assert_eq!(
            (mesh.vertex_count(), mesh.edge_count(), mesh.face_count()),
            (8, 12, 6)
        );
        mesh.validate_topology();
    }

    /// §4 / §5: a non-uniform factor `(a, b, c)` scales the three axes
    /// *independently* — the diagonal, not a single uniform factor. A regression
    /// that applied one factor to all axes lands different positions.
    #[test]
    fn non_uniform_scale_is_per_axis() {
        let mut mesh = LoopMesh::new();
        let verts = unit_cube(&mut mesh);
        let before: Vec<Vec3> = verts.iter().map(|&v| mesh.vertex_position(v)).collect();

        let vec = Vec3::new(2.0, 5.0, 0.5);
        mesh.scale(&verts, params(vec));

        for (corner, &vert) in verts.iter().enumerate() {
            let expected = before[corner] * vec;
            assert_close(mesh.vertex_position(vert), expected);
        }
        assert_eq!(
            (mesh.vertex_count(), mesh.edge_count(), mesh.face_count()),
            (8, 12, 6)
        );
        mesh.validate_topology();
    }

    /// §4: a negative factor (one negative entry) reflects positions across the
    /// `x = 0` plane and rewinds no face — topology is invariant.
    #[test]
    fn negative_factor_reflects_without_rewind() {
        let mut mesh = LoopMesh::new();
        let verts = unit_cube(&mut mesh);
        let before: Vec<Vec3> = verts.iter().map(|&v| mesh.vertex_position(v)).collect();

        let vec = Vec3::new(-1.0, 1.0, 1.0);
        mesh.scale(&verts, params(vec));

        for (corner, &vert) in verts.iter().enumerate() {
            let expected = before[corner] * vec;
            assert_close(mesh.vertex_position(vert), expected);
        }
        // Reflection moves verts only; the loop cycles are exactly as they were.
        assert_eq!(
            (mesh.vertex_count(), mesh.edge_count(), mesh.face_count()),
            (8, 12, 6)
        );
        mesh.validate_topology();
    }

    /// §4: a zero factor (one zero entry) collapses the named verts onto the
    /// complementary plane **in place** — applied as given, no guard. The result
    /// is finite (no NaN) and topology counts are invariant.
    #[test]
    fn zero_factor_collapses_in_place_and_finite() {
        let mut mesh = LoopMesh::new();
        let verts = unit_cube(&mut mesh);
        let before: Vec<Vec3> = verts.iter().map(|&v| mesh.vertex_position(v)).collect();

        let vec = Vec3::new(1.0, 1.0, 0.0); // flatten into z = 0
        mesh.scale(&verts, params(vec));

        for (corner, &vert) in verts.iter().enumerate() {
            let position = mesh.vertex_position(vert);
            assert!(position.is_finite(), "zero-factor collapse must be finite");
            assert_close(position, before[corner] * vec);
            assert!((position.z).abs() < 1e-6, "every vert lands on z = 0");
        }
        // Collapse onto a plane is in-place, not onto the origin (which a singular
        // `space` would produce) — the x/y spread survives.
        assert!(mesh.vertex_position(verts[6]).x.abs() > 1e-6);
        assert_eq!(
            (mesh.vertex_count(), mesh.edge_count(), mesh.face_count()),
            (8, 12, 6)
        );
        mesh.validate_topology();
    }

    /// §3: scale has no `cent`; a translation `space` recentres it. A `+c`
    /// translation frame scales about `−c` (the sign trap), so the vert at `−c`
    /// is the one left fixed. The result also differs from the no-space run,
    /// proving `space` is consulted.
    #[test]
    fn space_recentres_scale_with_sign() {
        // Choose c so that -c coincides with a cube vertex we can pin as fixed.
        // Fix vert 6 at (1,1,1): we need -c = (1,1,1), i.e. c = (-1,-1,-1).
        let c = Vec3::new(-1.0, -1.0, -1.0);
        let space = Mat4::from_translation(c); // `space · p = p + c`
        let factor = 4.0;

        let mut mesh = LoopMesh::new();
        let verts = unit_cube(&mut mesh);
        let inputs: Vec<Vec3> = verts.iter().map(|&v| mesh.vertex_position(v)).collect();
        mesh.scale(
            &verts,
            ScaleParams {
                vec: Vec3::splat(factor),
                space,
                use_shapekey: false,
            },
        );

        // The fixed point of a `+c` frame scale is -c; vert 6 sits there and must
        // not move.
        let fixed = -c;
        assert_close(mesh.vertex_position(verts[6]), fixed);

        // effective(p) = space⁻¹ · diag · space · p = s·(p + c) − c.
        let diag = Mat4::from_scale(Vec3::splat(factor));
        let effective = space.inverse() * diag * space;
        let mut differs = false;
        for (corner, &vert) in verts.iter().enumerate() {
            assert_close(
                mesh.vertex_position(vert),
                effective.transform_point3(inputs[corner]),
            );
            // contrast against the no-space (origin-centred) image.
            let origin_image = inputs[corner] * factor;
            if (mesh.vertex_position(vert) - origin_image).length() > 1e-4 {
                differs = true;
            }
        }
        assert!(
            differs,
            "a translation space must change the result vs the no-space run"
        );
        mesh.validate_topology();
    }

    /// §4: the identity factor `(1, 1, 1)` is a no-op on positions.
    #[test]
    fn identity_factor_is_noop() {
        let mut mesh = LoopMesh::new();
        let verts = unit_cube(&mut mesh);
        let before: Vec<Vec3> = verts.iter().map(|&v| mesh.vertex_position(v)).collect();

        mesh.scale(&verts, params(Vec3::ONE));

        for (corner, &vert) in verts.iter().enumerate() {
            assert_close(mesh.vertex_position(vert), before[corner]);
        }
        assert_eq!(
            (mesh.vertex_count(), mesh.edge_count(), mesh.face_count()),
            (8, 12, 6)
        );
        mesh.validate_topology();
    }

    /// §4: empty input is an accepted no-op even with a non-trivial factor — no
    /// position or topology change.
    #[test]
    fn empty_input_is_noop() {
        let mut mesh = LoopMesh::new();
        let verts = unit_cube(&mut mesh);
        let before: Vec<Vec3> = verts.iter().map(|&v| mesh.vertex_position(v)).collect();
        let counts = (mesh.vertex_count(), mesh.edge_count(), mesh.face_count());

        mesh.scale(&[], params(Vec3::new(7.0, 7.0, 7.0)));

        for (corner, &vert) in verts.iter().enumerate() {
            assert_close(mesh.vertex_position(vert), before[corner]);
        }
        assert_eq!(
            (mesh.vertex_count(), mesh.edge_count(), mesh.face_count()),
            counts
        );
        mesh.validate_topology();
    }

    /// §4 / TRANSFORM.md §7: a partial vert selection scales only the named
    /// corners — no closure pulls in the rest of a shared face. The un-named
    /// verts stay put and the counts are invariant.
    #[test]
    fn partial_selection_scales_only_named_verts() {
        let mut mesh = LoopMesh::new();
        let verts = unit_cube(&mut mesh);
        let before: Vec<Vec3> = verts.iter().map(|&v| mesh.vertex_position(v)).collect();

        let vec = Vec3::splat(2.0);
        let named = &verts[..4]; // bottom corners only
        mesh.scale(named, params(vec));

        for corner in 0..4 {
            assert_close(mesh.vertex_position(verts[corner]), before[corner] * vec);
        }
        for corner in 4..8 {
            assert_close(mesh.vertex_position(verts[corner]), before[corner]);
        }
        assert_eq!(
            (mesh.vertex_count(), mesh.edge_count(), mesh.face_count()),
            (8, 12, 6)
        );
        mesh.validate_topology();
    }

    /// §2 / §5: a position-only edit touches no customdata. A vert weight layer
    /// and a loop UV layer must survive a non-uniform scale unchanged.
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

        mesh.scale(&verts, params(Vec3::new(2.0, 5.0, 0.5)));

        for (corner, &vert) in verts.iter().enumerate() {
            let stored = mesh
                .vertex_attribs()
                .get_f32(weight, vert.as_usize())
                .unwrap();
            assert!(
                (stored - 0.1 * (corner as f32 + 1.0)).abs() < 1e-6,
                "vert weight survives the scale"
            );
        }
        for (l, before) in loop_uvs {
            let after = mesh.loop_attribs().get_vec2(uv, l.as_usize()).unwrap();
            assert_eq!(after, before, "loop UV survives the scale");
        }
        assert_eq!(
            (mesh.vertex_count(), mesh.edge_count(), mesh.face_count()),
            (8, 12, 6)
        );
        mesh.validate_topology();
    }
}
