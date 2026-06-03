use glam::{Mat4, Vec3};

use crate::loopmesh::*;

/// Parameters for [`LoopMesh::translate`] (TRANSLATE.md §1).
///
/// Translate is the vector-shaped front door to [`LoopMesh::transform`]: it
/// moves each named vert by a single `offset`, leaving topology, customdata, and
/// all relative geometry untouched (§2). It reduces exactly to `transform` fed a
/// pure-translation matrix — identity linear block, `offset` in the translation
/// column — so it forwards `space` and `use_shapekey` to transform unchanged.
#[derive(Clone, Copy, Debug)]
pub struct TranslateParams {
    /// The displacement added to each input vert's position (§1). Translate
    /// builds the identity 4×4 with `offset` in its translation column and hands
    /// that matrix to transform, so a vert at `p` lands at `p + offset` in the
    /// no-space case (§2).
    pub offset: Vec3,
    /// The frame the translation is applied *in* (§3), identical in meaning to
    /// transform's `space`. The all-zeros [`Mat4::ZERO`] sentinel means world
    /// space. A pure-translation or sentinel `space` is a no-op for translate
    /// because it commutes with the offset matrix; only a `space` with a
    /// non-trivial linear block (rotation/scale) re-aims the offset vector (§3).
    pub space: Mat4,
    /// Forwarded to transform unchanged (§1). loopmesh has no shape-key model, so
    /// this is an accepted-but-no-effect deferral — the base position result is
    /// identical with the flag set or clear.
    pub use_shapekey: bool,
}

impl LoopMesh {
    /// Shift the positions of `verts` by a single `offset` vector (TRANSLATE.md §2).
    ///
    /// Translate is [`LoopMesh::transform`] restricted to a pure-translation
    /// matrix: it builds `Mat4::from_translation(offset)` — identity linear
    /// block, `offset` in the translation column — and forwards it through the
    /// transform path with the same `space` and `use_shapekey`. A vert at `p`
    /// lands at `p + offset` in the no-space case; with a non-trivial linear-block
    /// `space` the offset is re-aimed through that frame (§3), but the underlying
    /// matrix is always a pure translation so winding is never flipped and no
    /// geometry collapses (§4).
    ///
    /// Topology and customdata are untouched — translate moves positions only and
    /// inherits transform's contract byte-for-byte (§2). The affected normals are
    /// recomputed from the new positions, but a pure translation is rigid so every
    /// recomputed normal lands on its previous value (§2). Empty `verts` and a
    /// zero `offset` are both accepted no-ops (§4).
    ///
    /// See `radial-mesh-guides/operators/TRANSLATE.md` for the full reduction and
    /// parity contract.
    pub fn translate(&mut self, verts: &[VertexKey], params: TranslateParams) {
        self.transform(
            verts,
            TransformParams {
                matrix: Mat4::from_translation(params.offset),
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

    fn params(offset: Vec3) -> TranslateParams {
        TranslateParams {
            offset,
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

    /// §2 / §5 headline: a non-zero offset moves every named vert to `p + offset`
    /// and leaves the cube's topology invariant.
    #[test]
    fn core_offset_shifts_every_vert() {
        let mut mesh = LoopMesh::new();
        let verts = unit_cube(&mut mesh);
        let before: Vec<Vec3> = verts.iter().map(|&v| mesh.vertex_position(v)).collect();

        let offset = Vec3::new(5.0, -2.0, 3.0);
        mesh.translate(&verts, params(offset));

        for (corner, &vert) in verts.iter().enumerate() {
            assert_close(mesh.vertex_position(vert), before[corner] + offset);
        }
        assert_eq!((mesh.totvert(), mesh.totedge(), mesh.totface()), (8, 12, 6));
        mesh.validate_topology();
    }

    /// §4: a zero offset maps every vert to itself — a no-op on positions.
    #[test]
    fn zero_offset_is_noop() {
        let mut mesh = LoopMesh::new();
        let verts = unit_cube(&mut mesh);
        let before: Vec<Vec3> = verts.iter().map(|&v| mesh.vertex_position(v)).collect();

        mesh.translate(&verts, params(Vec3::ZERO));

        for (corner, &vert) in verts.iter().enumerate() {
            assert_close(mesh.vertex_position(vert), before[corner]);
        }
        assert_eq!((mesh.totvert(), mesh.totedge(), mesh.totface()), (8, 12, 6));
        mesh.validate_topology();
    }

    /// §4: empty input is an accepted no-op even with a non-zero offset — no
    /// position or topology change.
    #[test]
    fn empty_input_is_noop() {
        let mut mesh = LoopMesh::new();
        let verts = unit_cube(&mut mesh);
        let before: Vec<Vec3> = verts.iter().map(|&v| mesh.vertex_position(v)).collect();
        let counts = (mesh.totvert(), mesh.totedge(), mesh.totface());

        mesh.translate(&[], params(Vec3::new(9.0, 9.0, 9.0)));

        for (corner, &vert) in verts.iter().enumerate() {
            assert_close(mesh.vertex_position(vert), before[corner]);
        }
        assert_eq!((mesh.totvert(), mesh.totedge(), mesh.totface()), counts);
        mesh.validate_topology();
    }

    /// §2 / §5: a position-only edit touches no customdata. A vert weight layer
    /// must survive a non-zero translate unchanged.
    #[test]
    fn customdata_is_untouched() {
        let mut mesh = LoopMesh::new();
        let verts = unit_cube(&mut mesh);

        let weight = mesh
            .register_vertex_attribute(ElementAttributeDesc::numeric(
                AttributeKey::custom("weight"),
                Format::new(ComponentType::F32, 1),
            ))
            .expect("vert weight layer registers");
        for (corner, &vert) in verts.iter().enumerate() {
            mesh.vertex_attribs_mut()
                .set_f32(weight, vert.as_usize(), 0.1 * (corner as f32 + 1.0));
        }

        mesh.translate(&verts, params(Vec3::new(-4.0, 1.0, 2.0)));

        for (corner, &vert) in verts.iter().enumerate() {
            let stored = mesh
                .vertex_attribs()
                .get_f32(weight, vert.as_usize())
                .unwrap();
            assert!(
                (stored - 0.1 * (corner as f32 + 1.0)).abs() < 1e-6,
                "vert weight survives the translate"
            );
        }
        mesh.validate_topology();
    }
}
