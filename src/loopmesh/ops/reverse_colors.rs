use crate::loopmesh::*;

impl LoopMesh {
    /// Reverse the selected colour layer's per-corner values around each named
    /// face.
    ///
    /// For every face the per-loop values of the selected colour layer are
    /// permuted so the value at corner `i` becomes the value that sat at the
    /// mirror corner `n - 1 - i`. Nothing else changes: no vertex moves, no
    /// element is created or killed, no edge is relinked, no winding is flipped,
    /// and no other customdata layer (including non-selected colour layers and
    /// any UV layer) is touched. Each loop keeps its vertex, its edges, and its
    /// place in the loop cycle; only the selected-colour payload riding on it is
    /// overwritten.
    ///
    /// The colour layer is selected by colour index, defaulting to `0` — the
    /// first colour-type (4-component) loop layer in stored order. If no colour
    /// layer resolves (none present, or the index is out of range) the operator
    /// cancels and leaves the mesh untouched — it does not error and does not
    /// create a layer. Each face is reversed independently, so the result is
    /// order-independent, and reversing the same face set twice restores the
    /// original colours exactly (the operator is its own inverse). For odd-`n`
    /// faces the centre corner is its own mirror and keeps its value.
    pub fn reverse_colors(&mut self, faces: &[FaceKey]) {
        self.reverse_colors_indexed(faces, 0);
    }

    /// Reverse the `color_index`-th colour loop layer's per-corner values around
    /// each named face. See [`reverse_colors`](Self::reverse_colors); this is the
    /// general form carrying the layer-selection index.
    pub fn reverse_colors_indexed(&mut self, faces: &[FaceKey], color_index: usize) {
        let color_layer = match self.active_color_layer(color_index) {
            Some(layer) => layer,
            None => return,
        };

        // Loop keys gathered once into a stack buffer for index addressing; the
        // permutation itself is an allocation-free two-pointer swap from both
        // ends toward the centre, so the centre corner of an odd-n face is never
        // touched.
        let mut loop_keys: smallvec::SmallVec<[LoopKey; 8]> = smallvec::SmallVec::new();

        for &face in faces {
            loop_keys.clear();
            for loop_key in self.iter_face_loops(face) {
                loop_keys.push(loop_key);
            }

            let n = loop_keys.len();
            for i in 0..n / 2 {
                let front = loop_keys[i].as_usize();
                let back = loop_keys[n - 1 - i].as_usize();

                let front_value = self
                    .loop_attribs()
                    .get_vec4(color_layer, front)
                    .expect("selected colour layer is a vec4 loop attribute");
                let back_value = self
                    .loop_attribs()
                    .get_vec4(color_layer, back)
                    .expect("selected colour layer is a vec4 loop attribute");

                self.loop_attribs_mut().set_vec4(color_layer, front, back_value);
                self.loop_attribs_mut().set_vec4(color_layer, back, front_value);
            }
        }
    }

    /// Resolve the `color_index`-th colour loop layer: the `color_index`-th
    /// loop-domain layer whose type is a colour type, counted in stored order.
    /// Colour layers are the 4-component loop layers; the canonical colour
    /// semantic is preferred for index `0` when present, otherwise selection is
    /// by position among the 4-component loop layers. `None` when the index does
    /// not resolve to a colour layer (none present, or out of range).
    pub(crate) fn active_color_layer(&self, color_index: usize) -> Option<ElementAttributeKey> {
        let attribs = self.loop_attribs();
        // Among the 4-component loop layers this op handles, prefer those
        // explicitly tagged with the Color role; if none is tagged, fall back
        // to the layout convention (any Vec4 loop layer is treated as colour).
        let is_vec4 = |key| matches!(attribs.storage_of(key), Some(AttributeStorage::Vec4(_)));
        let any_tagged = attribs
            .iter_descriptors()
            .any(|(key, d)| d.role == AttrRole::Color && is_vec4(key));
        attribs
            .iter_descriptors()
            .filter(|(key, d)| is_vec4(*key) && (!any_tagged || d.role == AttrRole::Color))
            .map(|(key, _)| key)
            .nth(color_index)
    }
}

#[cfg(test)]
mod tests {
    use glam::Vec3;

    use crate::loopmesh::test_support::create_face_from_verts;
    use crate::loopmesh::*;

    fn register_color(mesh: &mut LoopMesh) -> ElementAttributeKey {
        mesh.register_loop_attribute(ElementAttributeDesc::numeric(
            AttributeKey::custom("color"),
            AttrType::Vec4,
        ))
        .expect("colour layer registers")
    }

    fn ngon(mesh: &mut LoopMesh, n: usize) -> FaceKey {
        let verts: Vec<Vec3> = (0..n)
            .map(|i| {
                let a = std::f32::consts::TAU * (i as f32) / (n as f32);
                Vec3::new(a.cos(), a.sin(), 0.0)
            })
            .collect();
        create_face_from_verts(mesh, &verts)
    }

    /// Seed distinct, asymmetric colours corner-by-corner in loop-cycle order;
    /// return them.
    fn seed_corner_colors(
        mesh: &mut LoopMesh,
        face: FaceKey,
        color: ElementAttributeKey,
    ) -> Vec<[f32; 4]> {
        let loops = mesh.face_loop_keys(face);
        let mut values = Vec::with_capacity(loops.len());
        for (i, &loop_key) in loops.iter().enumerate() {
            let value = [i as f32, 10.0 + i as f32, 20.0 + i as f32, 1.0];
            mesh.loop_attribs_mut()
                .set_vec4(color, loop_key.as_usize(), value);
            values.push(value);
        }
        values
    }

    fn read_corner_colors(
        mesh: &LoopMesh,
        face: FaceKey,
        color: ElementAttributeKey,
    ) -> Vec<[f32; 4]> {
        mesh.face_loop_keys(face)
            .into_iter()
            .map(|l| mesh.loop_attribs().get_vec4(color, l.as_usize()).unwrap())
            .collect()
    }

    // Registration rejects a role/type pairing the role can't accept (set the
    // `role` field directly to bypass the `with_role` debug-assert).
    #[test]
    fn registering_invalid_role_type_pairing_is_rejected() {
        let mut mesh = LoopMesh::new();
        let mut desc = ElementAttributeDesc::numeric(AttributeKey::custom("x"), AttrType::Bool);
        desc.role = AttrRole::Color; // a bool colour is nonsense
        assert!(mesh.register_loop_attribute(desc).is_err());

        let mut desc = ElementAttributeDesc::numeric(AttributeKey::custom("y"), AttrType::F32);
        desc.role = AttrRole::Uv; // a scalar UV is nonsense
        assert!(mesh.register_loop_attribute(desc).is_err());
    }

    #[test]
    #[should_panic]
    fn with_role_debug_asserts_invalid_pairing() {
        let _ = ElementAttributeDesc::numeric(AttributeKey::custom("x"), AttrType::F32)
            .with_role(AttrRole::Uv);
    }

    // When a Color-tagged layer exists, lookup prefers it over an
    // earlier-registered untagged Vec4 layer.
    #[test]
    fn color_lookup_prefers_role_tagged_layer() {
        let mut mesh = LoopMesh::new();
        let generic = mesh
            .register_loop_attribute(ElementAttributeDesc::numeric(
                AttributeKey::custom("generic4"),
                AttrType::Vec4,
            ))
            .unwrap();
        let tagged = mesh
            .register_loop_attribute(ElementAttributeDesc::color(
                AttributeKey::custom("col"),
                AttrType::Vec4,
            ))
            .unwrap();
        assert_eq!(mesh.active_color_layer(0), Some(tagged));
        assert_ne!(mesh.active_color_layer(0), Some(generic));
    }

    // With no tagged layer, lookup falls back to the layout convention
    // (first Vec4 loop layer).
    #[test]
    fn color_lookup_falls_back_to_layout_when_untagged() {
        let mut mesh = LoopMesh::new();
        let first = mesh
            .register_loop_attribute(ElementAttributeDesc::numeric(
                AttributeKey::custom("a"),
                AttrType::Vec4,
            ))
            .unwrap();
        assert_eq!(mesh.active_color_layer(0), Some(first));
    }

    /// §2 / §5.1 — even `n`: every corner moves; corner `i` gets corner
    /// `n-1-i`'s value. No corner is fixed.
    #[test]
    fn quad_reverses_all_corners() {
        let mut mesh = LoopMesh::new();
        let color = register_color(&mut mesh);
        let face = ngon(&mut mesh, 4);
        let before = seed_corner_colors(&mut mesh, face, color);

        mesh.reverse_colors(&[face]);

        let after = read_corner_colors(&mesh, face, color);
        let n = before.len();
        for i in 0..n {
            assert_eq!(after[i], before[n - 1 - i], "corner {i} takes mirror corner");
        }
    }

    /// §5 / §6.2 — multiple faces of a cube reverse independently; topology is
    /// unchanged (a topology-only view sees a no-op).
    #[test]
    fn cube_all_faces_independent() {
        let mut mesh = LoopMesh::new();
        let color = register_color(&mut mesh);

        // Six independent quads standing in for the cube's six faces; each
        // reverses against its own corner colours.
        let mut faces = Vec::new();
        for _ in 0..6 {
            faces.push(ngon(&mut mesh, 4));
        }

        let befores: Vec<Vec<[f32; 4]>> = faces
            .iter()
            .map(|&f| seed_corner_colors(&mut mesh, f, color))
            .collect();
        let verts_before: Vec<Vec<VertexKey>> =
            faces.iter().map(|&f| mesh.face_vert_keys(f)).collect();

        mesh.reverse_colors(&faces);

        for (idx, &face) in faces.iter().enumerate() {
            let after = read_corner_colors(&mesh, face, color);
            let before = &befores[idx];
            let n = before.len();
            for i in 0..n {
                assert_eq!(after[i], before[n - 1 - i], "face {idx} corner {i}");
            }
            assert_eq!(
                mesh.face_vert_keys(face),
                verts_before[idx],
                "face {idx} topology unchanged"
            );
        }
    }

    /// §5.1 — mixed arity: two triangles and a quad reversed together pin the
    /// odd-`n` centre-fixed and even-`n` no-fixed-corner behaviours at once.
    #[test]
    fn mixed_tri_quad_reverses() {
        let mut mesh = LoopMesh::new();
        let color = register_color(&mut mesh);
        let tri_a = ngon(&mut mesh, 3);
        let tri_b = ngon(&mut mesh, 3);
        let quad = ngon(&mut mesh, 4);

        let tri_a_before = seed_corner_colors(&mut mesh, tri_a, color);
        let tri_b_before = seed_corner_colors(&mut mesh, tri_b, color);
        let quad_before = seed_corner_colors(&mut mesh, quad, color);

        mesh.reverse_colors(&[tri_a, tri_b, quad]);

        for (face, before) in [(tri_a, &tri_a_before), (tri_b, &tri_b_before)] {
            let after = read_corner_colors(&mesh, face, color);
            assert_eq!(after[1], before[1], "tri centre fixed");
            assert_eq!(after[0], before[2]);
            assert_eq!(after[2], before[0]);
        }

        let quad_after = read_corner_colors(&mesh, quad, color);
        let n = quad_before.len();
        for i in 0..n {
            assert_eq!(quad_after[i], quad_before[n - 1 - i], "quad corner {i}");
        }
    }

    /// §5.1 / §7 — odd `n`: the centre corner is its own mirror and keeps its
    /// value; the flanking corners exchange in pairs.
    #[test]
    fn pentagon_keeps_centre_corner() {
        let mut mesh = LoopMesh::new();
        let color = register_color(&mut mesh);
        let face = ngon(&mut mesh, 5);
        let before = seed_corner_colors(&mut mesh, face, color);

        mesh.reverse_colors(&[face]);

        let after = read_corner_colors(&mesh, face, color);
        assert_eq!(after[2], before[2], "centre corner is self-fixed");
        assert_eq!(after[0], before[4]);
        assert_eq!(after[4], before[0]);
        assert_eq!(after[1], before[3]);
        assert_eq!(after[3], before[1]);
    }

    /// §5 / §7 — reversing a face's colours twice restores the original
    /// labelling exactly: the operator is its own inverse.
    #[test]
    fn double_reverse_is_identity() {
        let mut mesh = LoopMesh::new();
        let color = register_color(&mut mesh);
        let face = ngon(&mut mesh, 4);
        let before = seed_corner_colors(&mut mesh, face, color);

        mesh.reverse_colors(&[face]);
        mesh.reverse_colors(&[face]);

        assert_eq!(read_corner_colors(&mesh, face, color), before);
    }

    /// §4.1 / §7 — with no colour layer the whole operator cancels: it does not
    /// panic, leaves topology unchanged, and creates no layer.
    #[test]
    fn no_color_layer_cancels() {
        let mut mesh = LoopMesh::new();
        let face = ngon(&mut mesh, 4);
        let verts_before = mesh.face_vert_keys(face);

        mesh.reverse_colors(&[face]);

        assert_eq!(mesh.face_vert_keys(face), verts_before);
        assert!(mesh.loop_attribs().is_empty(), "no layer was created");
    }

    /// §7 — empty input is a no-op even with a colour layer present.
    #[test]
    fn empty_input_is_noop() {
        let mut mesh = LoopMesh::new();
        let color = register_color(&mut mesh);
        let face = ngon(&mut mesh, 4);
        let before = seed_corner_colors(&mut mesh, face, color);

        mesh.reverse_colors(&[]);

        assert_eq!(read_corner_colors(&mesh, face, color), before);
    }

    /// §4 / §4.1 — only the selected colour layer is reversed; a second
    /// co-resident vec4 colour layer keeps its values. The semantic colour
    /// layer resolves at `color_index = 0` via the positional fallback.
    #[test]
    fn non_selected_color_layer_untouched() {
        let mut mesh = LoopMesh::new();
        let selected = register_color(&mut mesh);
        let other = mesh
            .register_loop_attribute(ElementAttributeDesc::numeric(
                AttributeKey::custom("color2"),
                AttrType::Vec4,
            ))
            .expect("second colour layer registers");
        let face = ngon(&mut mesh, 4);

        let selected_before = seed_corner_colors(&mut mesh, face, selected);
        let other_before: Vec<[f32; 4]> = {
            let loops = mesh.face_loop_keys(face);
            let mut vals = Vec::new();
            for (i, &l) in loops.iter().enumerate() {
                let v = [100.0 + i as f32, 200.0 + i as f32, 300.0 + i as f32, 1.0];
                mesh.loop_attribs_mut().set_vec4(other, l.as_usize(), v);
                vals.push(v);
            }
            vals
        };

        mesh.reverse_colors(&[face]);

        let selected_after = read_corner_colors(&mesh, face, selected);
        let n = selected_before.len();
        for i in 0..n {
            assert_eq!(
                selected_after[i],
                selected_before[n - 1 - i],
                "selected layer reversed"
            );
        }
        assert_eq!(
            read_corner_colors(&mesh, face, other),
            other_before,
            "non-selected colour layer untouched"
        );
    }
}
