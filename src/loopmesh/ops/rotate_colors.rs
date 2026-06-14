use crate::loopmesh::*;

impl LoopMesh {
    /// Rotate the selected colour layer's per-corner values around each named
    /// face by one corner.
    ///
    /// For every face the per-loop values of the selected colour layer are
    /// cyclically shifted one corner around the loop cycle. With
    /// `use_ccw = false` (the default, forward) the value at corner `i` becomes
    /// the value that sat at corner `i - 1 (mod n)`, so the whole sequence slides
    /// one corner along the loop direction with the tail wrapping to the head
    /// (`A,B,C,D` becomes `D,A,B,C`). With `use_ccw = true` (backward) corner `i`
    /// takes corner `i + 1 (mod n)`, sliding against the loop direction
    /// (`A,B,C,D` becomes `B,C,D,A`). The two directions are exact inverses.
    ///
    /// Nothing else changes: no vertex moves, no element is created or killed,
    /// no edge is relinked, no winding is flipped, and no other customdata layer
    /// (including non-selected colour layers and any UV layer) is touched. Each
    /// loop keeps its vertex, its edges, and its place in the loop cycle; only
    /// the selected-colour payload riding on it is overwritten. Unlike
    /// `reverse_colors`, rotate holds no corner fixed for any `n > 1`.
    ///
    /// The colour layer is selected by colour index, defaulting to `0` — the
    /// first colour-type (4-component) loop layer in stored order. If no colour
    /// layer resolves (none present, or the index is out of range) the operator
    /// cancels and leaves the mesh untouched — it does not error and does not
    /// create a layer. Each face is rotated independently, so the result is
    /// order-independent.
    pub fn rotate_colors(&mut self, faces: &[FaceKey], use_ccw: bool) {
        self.rotate_colors_indexed(faces, use_ccw, 0);
    }

    /// Rotate the `color_index`-th colour loop layer's per-corner values around
    /// each named face by one corner. See [`rotate_colors`](Self::rotate_colors);
    /// this is the general form carrying the layer-selection index.
    pub fn rotate_colors_indexed(&mut self, faces: &[FaceKey], use_ccw: bool, color_index: usize) {
        let color_layer = match self.active_color_layer(color_index) {
            Some(layer) => layer,
            None => return,
        };

        // Loop keys and their original colour values gathered once into stack
        // buffers; the one-corner shift overwrites a value an adjacent corner
        // still needs, so the originals must be saved before any write.
        let mut loop_keys: smallvec::SmallVec<[LoopKey; 8]> = smallvec::SmallVec::new();
        let mut saved: smallvec::SmallVec<[[f32; 4]; 8]> = smallvec::SmallVec::new();

        for &face in faces {
            loop_keys.clear();
            saved.clear();

            for loop_key in self.iter_face_loops(face) {
                loop_keys.push(loop_key);
            }

            let n = loop_keys.len();
            if n == 0 {
                continue;
            }

            for &loop_key in &loop_keys {
                let value = self
                    .loop_attribs()
                    .get_vec4(color_layer, loop_key.as_usize())
                    .expect("selected colour layer is a vec4 loop attribute");
                saved.push(value);
            }

            for i in 0..n {
                let source = if use_ccw {
                    (i + 1) % n
                } else {
                    (i + n - 1) % n
                };
                self.loop_attribs_mut().set_vec4(
                    color_layer,
                    loop_keys[i].as_usize(),
                    saved[source],
                );
            }
        }
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

    /// §2.1 / §5.1 — forward shift on a quad: `A,B,C,D` becomes `D,A,B,C`;
    /// corner `i` takes corner `i-1`'s value, tail wraps to head.
    #[test]
    fn quad_forward_shift() {
        let mut mesh = LoopMesh::new();
        let color = register_color(&mut mesh);
        let face = ngon(&mut mesh, 4);
        let before = seed_corner_colors(&mut mesh, face, color);

        mesh.rotate_colors(&[face], false);

        let after = read_corner_colors(&mesh, face, color);
        let n = before.len();
        for i in 0..n {
            assert_eq!(
                after[i],
                before[(i + n - 1) % n],
                "corner {i} takes predecessor"
            );
        }
    }

    /// §2.2 / §5.1 — backward shift on a quad: `A,B,C,D` becomes `B,C,D,A`;
    /// corner `i` takes corner `i+1`'s value, head wraps to tail.
    #[test]
    fn quad_backward_shift() {
        let mut mesh = LoopMesh::new();
        let color = register_color(&mut mesh);
        let face = ngon(&mut mesh, 4);
        let before = seed_corner_colors(&mut mesh, face, color);

        mesh.rotate_colors(&[face], true);

        let after = read_corner_colors(&mesh, face, color);
        let n = before.len();
        for i in 0..n {
            assert_eq!(after[i], before[(i + 1) % n], "corner {i} takes successor");
        }
    }

    /// §6.2 — forward and backward land different per-corner labellings on the
    /// same distinct-colour input: the `use_ccw` flag is itself pinned.
    #[test]
    fn forward_differs_from_backward() {
        let mut forward = LoopMesh::new();
        let fcolor = register_color(&mut forward);
        let fface = ngon(&mut forward, 4);
        seed_corner_colors(&mut forward, fface, fcolor);
        forward.rotate_colors(&[fface], false);

        let mut backward = LoopMesh::new();
        let bcolor = register_color(&mut backward);
        let bface = ngon(&mut backward, 4);
        seed_corner_colors(&mut backward, bface, bcolor);
        backward.rotate_colors(&[bface], true);

        assert_ne!(
            read_corner_colors(&forward, fface, fcolor),
            read_corner_colors(&backward, bface, bcolor),
            "forward and backward produce different labellings"
        );
    }

    /// §5 / §6.2 — multiple faces rotate independently; topology is unchanged.
    #[test]
    fn multi_face_independent() {
        let mut mesh = LoopMesh::new();
        let color = register_color(&mut mesh);

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

        mesh.rotate_colors(&faces, false);

        for (idx, &face) in faces.iter().enumerate() {
            let after = read_corner_colors(&mesh, face, color);
            let before = &befores[idx];
            let n = before.len();
            for i in 0..n {
                assert_eq!(after[i], before[(i + n - 1) % n], "face {idx} corner {i}");
            }
            assert_eq!(
                mesh.face_vert_keys(face),
                verts_before[idx],
                "face {idx} topology unchanged"
            );
        }
    }

    /// §5.1 — mixed arity: the triangles and the quad both rotate by the same
    /// one-step rule, and on the odd-`n` triangles no corner is held fixed
    /// (the rotate-vs-reverse tell).
    #[test]
    fn mixed_tri_quad_no_corner_fixed() {
        let mut mesh = LoopMesh::new();
        let color = register_color(&mut mesh);
        let tri_a = ngon(&mut mesh, 3);
        let tri_b = ngon(&mut mesh, 3);
        let quad = ngon(&mut mesh, 4);

        let tri_a_before = seed_corner_colors(&mut mesh, tri_a, color);
        let tri_b_before = seed_corner_colors(&mut mesh, tri_b, color);
        let quad_before = seed_corner_colors(&mut mesh, quad, color);

        mesh.rotate_colors(&[tri_a, tri_b, quad], false);

        for (face, before) in [(tri_a, &tri_a_before), (tri_b, &tri_b_before)] {
            let after = read_corner_colors(&mesh, face, color);
            let n = before.len();
            for i in 0..n {
                assert_ne!(
                    after[i], before[i],
                    "tri corner {i} moved (no fixed corner)"
                );
                assert_eq!(after[i], before[(i + n - 1) % n]);
            }
        }

        let quad_after = read_corner_colors(&mesh, quad, color);
        let n = quad_before.len();
        for i in 0..n {
            assert_ne!(quad_after[i], quad_before[i], "quad corner {i} moved");
            assert_eq!(quad_after[i], quad_before[(i + n - 1) % n]);
        }
    }

    /// §5.2 / §7 — forward then backward restores the original labelling: the
    /// two directions are exact inverses.
    #[test]
    fn forward_then_backward_is_identity() {
        let mut mesh = LoopMesh::new();
        let color = register_color(&mut mesh);
        let face = ngon(&mut mesh, 4);
        let before = seed_corner_colors(&mut mesh, face, color);

        mesh.rotate_colors(&[face], false);
        mesh.rotate_colors(&[face], true);

        assert_eq!(read_corner_colors(&mesh, face, color), before);
    }

    /// §4.1 / §7 — with no colour layer the whole operator cancels: it does not
    /// panic, leaves topology unchanged, and creates no layer.
    #[test]
    fn no_color_layer_cancels() {
        let mut mesh = LoopMesh::new();
        let face = ngon(&mut mesh, 4);
        let verts_before = mesh.face_vert_keys(face);

        mesh.rotate_colors(&[face], false);
        mesh.rotate_colors(&[face], true);

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

        mesh.rotate_colors(&[], false);

        assert_eq!(read_corner_colors(&mesh, face, color), before);
    }

    /// §4 / §4.1 — only the selected colour layer is rotated; a second
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

        mesh.rotate_colors(&[face], false);

        let selected_after = read_corner_colors(&mesh, face, selected);
        let n = selected_before.len();
        for i in 0..n {
            assert_eq!(
                selected_after[i],
                selected_before[(i + n - 1) % n],
                "selected layer rotated"
            );
        }
        assert_eq!(
            read_corner_colors(&mesh, face, other),
            other_before,
            "non-selected colour layer untouched"
        );
    }
}
