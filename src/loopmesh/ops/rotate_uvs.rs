use crate::loopmesh::*;

impl LoopMesh {
    /// Rotate the active UV layer's per-corner values around each named face by
    /// one corner.
    ///
    /// For every face the per-loop values of the active UV layer are cyclically
    /// shifted one corner around the loop cycle. With `use_ccw = false` (the
    /// default, forward) the value at corner `i` becomes the value that sat at
    /// corner `i - 1 (mod n)`, so the whole sequence slides one corner along the
    /// loop direction with the tail wrapping to the head (`A,B,C,D` becomes
    /// `D,A,B,C`). With `use_ccw = true` (backward) corner `i` takes corner
    /// `i + 1 (mod n)`, sliding against the loop direction (`A,B,C,D` becomes
    /// `B,C,D,A`). The two directions are exact inverses.
    ///
    /// Nothing else changes: no vertex moves, no element is created or killed,
    /// no edge is relinked, no winding is flipped, and no other customdata layer
    /// (including non-active UV layers) is touched. Each loop keeps its vertex,
    /// its edges, and its place in the loop cycle; only the active-UV payload
    /// riding on it is overwritten. Unlike `reverse_uvs`, rotate holds no corner
    /// fixed for any `n > 1`.
    ///
    /// The active UV layer is resolved once for the whole call. If the mesh
    /// carries no UV layer the operator is a no-op — it does not error and does
    /// not create a layer. Each face is rotated independently, so the result is
    /// order-independent.
    pub fn rotate_uvs(&mut self, faces: &[FaceKey], use_ccw: bool) {
        let uv_layer = match self.active_uv_layer_for_rotate() {
            Some(layer) => layer,
            None => return,
        };

        // Loop keys and their original UV values gathered once into stack
        // buffers; the shift overwrites a value an adjacent corner still needs,
        // so the originals must be saved before any write.
        let mut loop_keys: smallvec::SmallVec<[LoopKey; 8]> = smallvec::SmallVec::new();
        let mut saved: smallvec::SmallVec<[[f32; 2]; 8]> = smallvec::SmallVec::new();

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
                    .get_vec2(uv_layer, loop_key.as_usize())
                    .expect("active UV layer is a vec2 loop attribute");
                saved.push(value);
            }

            for i in 0..n {
                let source = if use_ccw {
                    (i + 1) % n
                } else {
                    (i + n - 1) % n
                };
                self.loop_attribs_mut()
                    .set_vec2(uv_layer, loop_keys[i].as_usize(), saved[source]);
            }
        }
    }

    /// Resolve the active UV layer: the loop-domain layer the operator targets
    /// by default. Prefer the canonical primary-UV semantic; otherwise fall
    /// back to the first registered 2-component loop layer (the established
    /// "a vec2 loop layer is a UV layer" convention). `None` when the mesh
    /// carries no UV layer at all.
    fn active_uv_layer_for_rotate(&self) -> Option<ElementAttributeKey> {
        let attribs = self.loop_attribs();

        if let Some(key) = attribs.find(&AttributeKey::Semantic(AttributeSemantic::Uv0)) {
            if matches!(attribs.storage_of(key), Some(AttributeStorage::Vec2(_))) {
                return Some(key);
            }
        }

        attribs
            .iter_descriptors()
            .map(|(key, _)| key)
            .find(|&key| matches!(attribs.storage_of(key), Some(AttributeStorage::Vec2(_))))
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
            Format::new(ComponentType::F32, 2),
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
    fn seed_corner_uvs(mesh: &mut LoopMesh, face: FaceKey, uv: ElementAttributeKey) -> Vec<[f32; 2]> {
        let loops = mesh.face_loop_keys(face);
        let mut values = Vec::with_capacity(loops.len());
        for (i, &loop_key) in loops.iter().enumerate() {
            let value = [i as f32, 10.0 + i as f32];
            mesh.loop_attribs_mut().set_vec2(uv, loop_key.as_usize(), value);
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

    /// §2.1 / §5.1 — forward shift on a quad: `A,B,C,D` becomes `D,A,B,C`;
    /// corner `i` takes corner `i-1`'s value, tail wraps to head.
    #[test]
    fn quad_forward_shift() {
        let mut mesh = LoopMesh::new();
        let uv = register_uv(&mut mesh);
        let face = ngon(&mut mesh, 4);
        let before = seed_corner_uvs(&mut mesh, face, uv);

        mesh.rotate_uvs(&[face], false);

        let after = read_corner_uvs(&mesh, face, uv);
        let n = before.len();
        for i in 0..n {
            assert_eq!(after[i], before[(i + n - 1) % n], "corner {i} takes predecessor");
        }
    }

    /// §2.2 / §5.1 — backward shift on a quad: `A,B,C,D` becomes `B,C,D,A`;
    /// corner `i` takes corner `i+1`'s value, head wraps to tail.
    #[test]
    fn quad_backward_shift() {
        let mut mesh = LoopMesh::new();
        let uv = register_uv(&mut mesh);
        let face = ngon(&mut mesh, 4);
        let before = seed_corner_uvs(&mut mesh, face, uv);

        mesh.rotate_uvs(&[face], true);

        let after = read_corner_uvs(&mesh, face, uv);
        let n = before.len();
        for i in 0..n {
            assert_eq!(after[i], before[(i + 1) % n], "corner {i} takes successor");
        }
    }

    /// §6.2 — forward and backward land different per-corner labellings on the
    /// same distinct-UV input: the `use_ccw` flag is itself pinned.
    #[test]
    fn forward_differs_from_backward() {
        let mut forward = LoopMesh::new();
        let fuv = register_uv(&mut forward);
        let fface = ngon(&mut forward, 4);
        seed_corner_uvs(&mut forward, fface, fuv);
        forward.rotate_uvs(&[fface], false);

        let mut backward = LoopMesh::new();
        let buv = register_uv(&mut backward);
        let bface = ngon(&mut backward, 4);
        seed_corner_uvs(&mut backward, bface, buv);
        backward.rotate_uvs(&[bface], true);

        assert_ne!(
            read_corner_uvs(&forward, fface, fuv),
            read_corner_uvs(&backward, bface, buv),
            "forward and backward produce different labellings"
        );
    }

    /// §5 — multiple faces rotate independently; topology is unchanged.
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

        mesh.rotate_uvs(&[tri, quad], false);

        let quad_after = read_corner_uvs(&mesh, quad, uv);
        for i in 0..quad_before.len() {
            assert_eq!(quad_after[i], quad_before[(i + quad_before.len() - 1) % quad_before.len()]);
        }
        let tri_after = read_corner_uvs(&mesh, tri, uv);
        for i in 0..tri_before.len() {
            assert_eq!(tri_after[i], tri_before[(i + tri_before.len() - 1) % tri_before.len()]);
        }

        assert_eq!(mesh.face_vert_keys(quad), quad_verts, "topology unchanged");
        assert_eq!(mesh.face_vert_keys(tri), tri_verts, "topology unchanged");
    }

    /// §5.1 — mixed arity: the triangle and the quad both rotate by the same
    /// one-step rule, and on the odd-`n` triangle no corner is held fixed
    /// (the rotate-vs-reverse tell).
    #[test]
    fn mixed_tri_quad_no_corner_fixed() {
        let mut mesh = LoopMesh::new();
        let uv = register_uv(&mut mesh);
        let tri = ngon(&mut mesh, 3);
        let quad = ngon(&mut mesh, 4);

        let tri_before = seed_corner_uvs(&mut mesh, tri, uv);
        let quad_before = seed_corner_uvs(&mut mesh, quad, uv);

        mesh.rotate_uvs(&[tri, quad], false);

        let tri_after = read_corner_uvs(&mesh, tri, uv);
        for i in 0..tri_before.len() {
            assert_ne!(tri_after[i], tri_before[i], "tri corner {i} moved");
            assert_eq!(tri_after[i], tri_before[(i + tri_before.len() - 1) % tri_before.len()]);
        }
        let quad_after = read_corner_uvs(&mesh, quad, uv);
        for i in 0..quad_before.len() {
            assert_ne!(quad_after[i], quad_before[i], "quad corner {i} moved");
        }
    }

    /// §5.2 / §7 — forward then backward restores the original labelling: the
    /// two directions are exact inverses.
    #[test]
    fn forward_then_backward_is_identity() {
        let mut mesh = LoopMesh::new();
        let uv = register_uv(&mut mesh);
        let face = ngon(&mut mesh, 4);
        let before = seed_corner_uvs(&mut mesh, face, uv);

        mesh.rotate_uvs(&[face], false);
        mesh.rotate_uvs(&[face], true);

        assert_eq!(read_corner_uvs(&mesh, face, uv), before);
    }

    /// §4 / §7 — with no UV layer the whole operator is a no-op: it does not
    /// panic and creates no layer.
    #[test]
    fn no_uv_layer_is_noop() {
        let mut mesh = LoopMesh::new();
        let face = ngon(&mut mesh, 4);
        let verts_before = mesh.face_vert_keys(face);

        mesh.rotate_uvs(&[face], false);
        mesh.rotate_uvs(&[face], true);

        assert_eq!(mesh.face_vert_keys(face), verts_before);
        assert!(mesh.loop_attribs().is_empty(), "no layer was created");
    }
}
