use crate::loopmesh::*;

impl LoopMesh {
    /// Reverse the active UV layer's per-corner values around each named face.
    ///
    /// For every face the per-loop values of the active UV layer are permuted
    /// so the value at corner `i` becomes the value that sat at the mirror
    /// corner `n - 1 - i`. Nothing else changes: no vertex moves, no element
    /// is created or killed, no edge is relinked, no winding is flipped, and
    /// no other customdata layer (including non-active UV layers) is touched.
    /// Each loop keeps its vertex, its edges, and its place in the loop cycle;
    /// only the active-UV payload riding on it is overwritten.
    ///
    /// The active UV layer is resolved once for the whole call. If the mesh
    /// carries no UV layer the operator is a no-op — it does not error and does
    /// not create a layer. Each face is reversed independently, so the result
    /// is order-independent, and reversing the same face set twice restores the
    /// original UVs exactly (the operator is its own inverse). For odd-`n`
    /// faces the centre corner is its own mirror and keeps its value.
    pub fn reverse_uvs(&mut self, faces: &[FaceKey]) {
        let uv_layer = match self.active_uv_loop_layer() {
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
                    .get_vec2(uv_layer, front)
                    .expect("active UV layer is a vec2 loop attribute");
                let back_value = self
                    .loop_attribs()
                    .get_vec2(uv_layer, back)
                    .expect("active UV layer is a vec2 loop attribute");

                self.loop_attribs_mut()
                    .set_vec2(uv_layer, front, back_value);
                self.loop_attribs_mut()
                    .set_vec2(uv_layer, back, front_value);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use glam::Vec3;

    use crate::loopmesh::test_support::create_face_from_verts;
    use crate::loopmesh::*;

    fn register_uv(mesh: &mut LoopMesh) -> ElementAttributeKey {
        mesh.register_loop_attribute(ElementAttributeDesc::numeric(
            AttributeKey::custom("uv"),
            AttrType::Vec2,
        ))
        .expect("uv layer registers")
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

    /// Seed distinct UVs corner-by-corner in loop-cycle order; return them.
    fn seed_corner_uvs(
        mesh: &mut LoopMesh,
        face: FaceKey,
        uv: ElementAttributeKey,
    ) -> Vec<[f32; 2]> {
        let loops = mesh.face_loop_keys(face);
        let mut values = Vec::with_capacity(loops.len());
        for (i, &loop_key) in loops.iter().enumerate() {
            let value = [i as f32, 10.0 + i as f32];
            mesh.loop_attribs_mut()
                .set_vec2(uv, loop_key.as_usize(), value);
            values.push(value);
        }
        values
    }

    fn read_corner_uvs(mesh: &LoopMesh, face: FaceKey, uv: ElementAttributeKey) -> Vec<[f32; 2]> {
        mesh.face_loop_keys(face)
            .into_iter()
            .map(|l| mesh.loop_attribs().get_vec2(uv, l.as_usize()).unwrap())
            .collect()
    }

    /// §2 / §5.1 — even `n`: every corner moves; corner `i` gets corner
    /// `n-1-i`'s value. No corner is fixed.
    #[test]
    fn quad_reverses_all_corners() {
        let mut mesh = LoopMesh::new();
        let uv = register_uv(&mut mesh);
        let face = ngon(&mut mesh, 4);
        let before = seed_corner_uvs(&mut mesh, face, uv);

        mesh.reverse_uvs(&[face]);

        let after = read_corner_uvs(&mesh, face, uv);
        let n = before.len();
        for i in 0..n {
            assert_eq!(
                after[i],
                before[n - 1 - i],
                "corner {i} takes mirror corner"
            );
        }
    }

    /// §5.1 / §7 — odd `n`: the centre corner is its own mirror and keeps its
    /// value; the flanking corners exchange in pairs.
    #[test]
    fn pentagon_keeps_centre_corner() {
        let mut mesh = LoopMesh::new();
        let uv = register_uv(&mut mesh);
        let face = ngon(&mut mesh, 5);
        let before = seed_corner_uvs(&mut mesh, face, uv);

        mesh.reverse_uvs(&[face]);

        let after = read_corner_uvs(&mesh, face, uv);
        assert_eq!(after[2], before[2], "centre corner is self-fixed");
        assert_eq!(after[0], before[4]);
        assert_eq!(after[4], before[0]);
        assert_eq!(after[1], before[3]);
        assert_eq!(after[3], before[1]);
    }

    /// §5 / §7 — reversing a face's UVs twice restores the original labelling
    /// exactly: the operator is its own inverse.
    #[test]
    fn double_reverse_is_identity() {
        let mut mesh = LoopMesh::new();
        let uv = register_uv(&mut mesh);
        let face = ngon(&mut mesh, 4);
        let before = seed_corner_uvs(&mut mesh, face, uv);

        mesh.reverse_uvs(&[face]);
        mesh.reverse_uvs(&[face]);

        assert_eq!(read_corner_uvs(&mesh, face, uv), before);
    }

    /// §4 / §7 — with no UV layer the whole operator is a no-op: it does not
    /// panic and creates no layer.
    #[test]
    fn no_uv_layer_is_noop() {
        let mut mesh = LoopMesh::new();
        let face = ngon(&mut mesh, 4);
        let verts_before = mesh.face_vert_keys(face);

        mesh.reverse_uvs(&[face]);

        assert_eq!(mesh.face_vert_keys(face), verts_before);
        assert!(mesh.loop_attribs().is_empty(), "no layer was created");
    }

    /// §4 — only the active UV layer is reversed; a second vec2 loop layer
    /// riding alongside keeps its values. The first-registered vec2 layer is
    /// the active one.
    #[test]
    fn non_active_uv_layer_untouched() {
        let mut mesh = LoopMesh::new();
        let active = register_uv(&mut mesh);
        let other = mesh
            .register_loop_attribute(ElementAttributeDesc::numeric(
                AttributeKey::custom("uv2"),
                AttrType::Vec2,
            ))
            .expect("second uv layer registers");
        let face = ngon(&mut mesh, 4);

        let active_before = seed_corner_uvs(&mut mesh, face, active);
        let other_before: Vec<[f32; 2]> = {
            let loops = mesh.face_loop_keys(face);
            let mut vals = Vec::new();
            for (i, &l) in loops.iter().enumerate() {
                let v = [100.0 + i as f32, 200.0 + i as f32];
                mesh.loop_attribs_mut().set_vec2(other, l.as_usize(), v);
                vals.push(v);
            }
            vals
        };

        mesh.reverse_uvs(&[face]);

        let active_after = read_corner_uvs(&mesh, face, active);
        let n = active_before.len();
        for i in 0..n {
            assert_eq!(active_after[i], active_before[n - 1 - i], "active reversed");
        }
        assert_eq!(
            read_corner_uvs(&mesh, face, other),
            other_before,
            "non-active layer untouched"
        );
    }

    /// §5 — multiple faces reverse independently; order-independent. Topology
    /// is unchanged (a topology-only view sees a no-op).
    #[test]
    fn multi_face_independent() {
        let mut mesh = LoopMesh::new();
        let uv = register_uv(&mut mesh);
        let quad = ngon(&mut mesh, 4);
        let tri = ngon(&mut mesh, 3);

        let quad_before = seed_corner_uvs(&mut mesh, quad, uv);
        let tri_before = seed_corner_uvs(&mut mesh, tri, uv);
        let quad_verts = mesh.face_vert_keys(quad);
        let tri_verts = mesh.face_vert_keys(tri);

        mesh.reverse_uvs(&[tri, quad]);

        let quad_after = read_corner_uvs(&mesh, quad, uv);
        for i in 0..quad_before.len() {
            assert_eq!(quad_after[i], quad_before[quad_before.len() - 1 - i]);
        }
        let tri_after = read_corner_uvs(&mesh, tri, uv);
        assert_eq!(tri_after[1], tri_before[1], "tri centre fixed");
        assert_eq!(tri_after[0], tri_before[2]);
        assert_eq!(tri_after[2], tri_before[0]);

        assert_eq!(mesh.face_vert_keys(quad), quad_verts, "topology unchanged");
        assert_eq!(mesh.face_vert_keys(tri), tri_verts, "topology unchanged");
    }

    /// §7 — empty input is a no-op.
    #[test]
    fn empty_input_is_noop() {
        let mut mesh = LoopMesh::new();
        let uv = register_uv(&mut mesh);
        let face = ngon(&mut mesh, 4);
        let before = seed_corner_uvs(&mut mesh, face, uv);

        mesh.reverse_uvs(&[]);

        assert_eq!(read_corner_uvs(&mesh, face, uv), before);
    }
}
