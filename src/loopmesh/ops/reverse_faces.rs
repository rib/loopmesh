use crate::loopmesh::*;

impl LoopMesh {
    /// Flip the winding of every named face and flip its cached normal.
    ///
    /// For each face this reverses the loop cycle (the loop-reverse Euler
    /// operator: loop vertices stay fixed, each loop's edge moves to its
    /// predecessor's edge, `next`/`prev` swap, radial cycles relink) and
    /// negates the stored face normal so it follows the new winding. No
    /// element is created or killed and no ordinary per-loop / per-face
    /// customdata is copied or interpolated — each loop carries its own
    /// data through the reversal unchanged; only the cyclic order in which
    /// the loops are encountered reverses.
    ///
    /// Each face is reversed independently, so the result is identical for
    /// any ordering of `faces`, and reversing the same set twice restores
    /// the original winding exactly. Reversing a face left out of `faces`
    /// is untouched, so a partial selection of an edge-adjacent pair leaves
    /// the pair inconsistently wound across the shared edge — this is not
    /// policed.
    ///
    /// `flip_multires` governs re-orientation of a per-loop multi-resolution
    /// displacement layer to follow the reversed winding. loopmesh carries
    /// no such layer, so the flag has no observable effect either way; it is
    /// accepted for parity with the wider operator surface.
    ///
    /// After every face is reversed, the vertex normals of the affected
    /// corners are refreshed so the averaged vertex normals agree with the
    /// freshly-flipped face normals.
    pub fn reverse_faces(&mut self, faces: &[FaceKey], flip_multires: bool) {
        let _ = flip_multires;

        if faces.is_empty() {
            return;
        }

        let mut touched_verts: smallvec::SmallVec<[VertexKey; 16]> = smallvec::SmallVec::new();

        for &face in faces {
            for vert in self.face_vert_keys(face) {
                if !touched_verts.contains(&vert) {
                    touched_verts.push(vert);
                }
            }

            self.euler_loop_reverse(face);

            // The winding flip inverts the implied normal. Negating the
            // cached value keeps a degenerate face's all-zero normal at
            // (0, 0, 0) rather than introducing a spurious direction.
            let flipped = -self.faces.get(face).normal;
            self.faces.get_mut(face).normal = flipped;
        }

        for vert in touched_verts {
            self.vert_normal_update(vert);
        }
    }
}

#[cfg(test)]
mod tests {
    use glam::Vec3;

    use crate::loopmesh::test_support::create_face_from_verts;
    use crate::loopmesh::*;

    /// Build two quads sharing the edge (v1, v2):
    ///   left:  v0 v1 v2 v3   right: v1 v4 v5 v2
    /// Returns (left_face, right_face, shared_edge).
    fn two_quads_shared_edge(mesh: &mut LoopMesh) -> (FaceKey, FaceKey, EdgeKey) {
        let v0 = mesh.create_vertex(Vec3::new(0.0, 0.0, 0.0));
        let v1 = mesh.create_vertex(Vec3::new(1.0, 0.0, 0.0));
        let v2 = mesh.create_vertex(Vec3::new(1.0, 1.0, 0.0));
        let v3 = mesh.create_vertex(Vec3::new(0.0, 1.0, 0.0));
        let v4 = mesh.create_vertex(Vec3::new(2.0, 0.0, 0.0));
        let v5 = mesh.create_vertex(Vec3::new(2.0, 1.0, 0.0));

        let e01 = mesh.create_edge([v0, v1], EdgeCreateFlags::NONE);
        let e12 = mesh.create_edge([v1, v2], EdgeCreateFlags::NONE);
        let e23 = mesh.create_edge([v2, v3], EdgeCreateFlags::NONE);
        let e30 = mesh.create_edge([v3, v0], EdgeCreateFlags::NONE);
        let e14 = mesh.create_edge([v1, v4], EdgeCreateFlags::NONE);
        let e45 = mesh.create_edge([v4, v5], EdgeCreateFlags::NONE);
        let e52 = mesh.create_edge([v5, v2], EdgeCreateFlags::NONE);

        let left = mesh.face_create(
            &[v0, v1, v2, v3],
            &[e01, e12, e23, e30],
            FaceCreateFlags::NONE,
        );
        let right = mesh.face_create(
            &[v1, v4, v5, v2],
            &[e14, e45, e52, e12],
            FaceCreateFlags::NONE,
        );
        (left, right, e12)
    }

    /// §6.2 headline — reversing one of two edge-adjacent faces flips that
    /// face's normal relative to its un-reversed neighbour. The shared edge
    /// is now traversed the same direction by both faces (a normals
    /// discontinuity); the neighbour's normal is unchanged.
    #[test]
    fn one_of_two_adjacent_flips_normal_vs_neighbour() {
        let mut mesh = LoopMesh::new();
        let (left, right, shared) = two_quads_shared_edge(&mut mesh);
        mesh.calc_face_normals();

        let left_before = mesh.face_normal(left);
        let right_before = mesh.face_normal(right);
        let left_verts_before: std::collections::HashSet<VertexKey> =
            mesh.face_vert_keys(left).into_iter().collect();

        mesh.reverse_faces(&[left], false);

        assert!(
            (mesh.face_normal(left) + left_before).length() < 1e-6,
            "reversed face normal must point the opposite way"
        );
        assert!(
            (mesh.face_normal(right) - right_before).length() < 1e-6,
            "un-reversed neighbour keeps its normal"
        );

        // Topology is untouched: both faces still share the edge, vertices unchanged.
        assert_eq!(mesh.edge_face_keys(shared).len(), 2);
        assert_eq!(
            left_verts_before,
            mesh.face_vert_keys(left).into_iter().collect(),
            "reversed face keeps its corner vertex set"
        );
    }

    /// §6.2 contrast — reverse *every* face: a topology no-op. Same verts,
    /// edges, faces, same per-face vertex set; only every normal flips.
    #[test]
    fn reverse_all_is_topology_noop() {
        let mut mesh = LoopMesh::new();
        let (left, right, shared) = two_quads_shared_edge(&mut mesh);
        mesh.calc_face_normals();

        let left_verts_before: std::collections::HashSet<VertexKey> =
            mesh.face_vert_keys(left).into_iter().collect();
        let right_verts_before: std::collections::HashSet<VertexKey> =
            mesh.face_vert_keys(right).into_iter().collect();
        let left_n = mesh.face_normal(left);
        let right_n = mesh.face_normal(right);

        mesh.reverse_faces(&[left, right], false);

        assert_eq!(mesh.edge_face_keys(shared).len(), 2, "still shared");
        assert_eq!(
            left_verts_before,
            mesh.face_vert_keys(left).into_iter().collect(),
            "left face keeps its vertex set"
        );
        assert_eq!(
            right_verts_before,
            mesh.face_vert_keys(right).into_iter().collect(),
            "right face keeps its vertex set"
        );
        assert!((mesh.face_normal(left) + left_n).length() < 1e-6);
        assert!((mesh.face_normal(right) + right_n).length() < 1e-6);
    }

    /// §6.2 / §7 — reversing an isolated face flips its normal and reverses
    /// its loop traversal, but keeps the same corner vertex set. The
    /// "same starting corner, reversed remainder" sequence pins the winding
    /// flip directly (a topology-only view cannot see it).
    #[test]
    fn single_isolated_face_reverses_winding() {
        let mut mesh = LoopMesh::new();
        let face = create_face_from_verts(
            &mut mesh,
            &[
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(2.0, 0.0, 0.0),
                Vec3::new(2.0, 1.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
            ],
        );
        mesh.calc_face_normals();
        let before = mesh.face_vert_keys(face);
        let normal_before = mesh.face_normal(face);

        mesh.reverse_faces(&[face], false);

        let after = mesh.face_vert_keys(face);
        assert_eq!(after[0], before[0], "first corner is fixed");
        for i in 1..before.len() {
            assert_eq!(after[i], before[before.len() - i], "corner {i} reversed");
        }
        assert!((mesh.face_normal(face) + normal_before).length() < 1e-6);
    }

    /// §5 / §7 — reversing a face set twice restores the original winding
    /// exactly (loop reverse is its own inverse; double-negating the normal
    /// is the identity).
    #[test]
    fn double_reverse_is_identity() {
        let mut mesh = LoopMesh::new();
        let (left, right, _) = two_quads_shared_edge(&mut mesh);
        mesh.calc_face_normals();

        let left_seq = mesh.face_vert_keys(left);
        let right_seq = mesh.face_vert_keys(right);
        let left_n = mesh.face_normal(left);
        let right_n = mesh.face_normal(right);

        mesh.reverse_faces(&[left, right], false);
        mesh.reverse_faces(&[left, right], false);

        assert_eq!(mesh.face_vert_keys(left), left_seq);
        assert_eq!(mesh.face_vert_keys(right), right_seq);
        assert!((mesh.face_normal(left) - left_n).length() < 1e-6);
        assert!((mesh.face_normal(right) - right_n).length() < 1e-6);
    }

    /// §3.1 — per-loop customdata stays attached to its loop. Each loop
    /// keeps its own value and its own vertex, so the set of (corner, value)
    /// pairs is identical; only the cyclic order in which they are
    /// encountered reverses. A reversed quad's UVs are not mirrored or
    /// recomputed.
    #[test]
    fn customdata_follows_reversed_loop_order() {
        let mut mesh = LoopMesh::new();
        let uv = mesh
            .register_loop_attribute(ElementAttributeDesc::numeric(
                AttributeKey::custom("uv"),
                AttrType::Vec2,
            ))
            .expect("uv layer registers");

        let face = create_face_from_verts(
            &mut mesh,
            &[
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(2.0, 0.0, 0.0),
                Vec3::new(2.0, 1.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
            ],
        );

        // Tag each loop's UV with its corner vertex so we can prove the
        // (corner, UV) pairing is preserved through the reversal.
        let loops = mesh.face_loop_keys(face);
        let pair_value = |i: usize| [i as f32, 10.0 + i as f32];
        let mut expected: std::collections::HashMap<VertexKey, [f32; 2]> =
            std::collections::HashMap::new();
        for (i, &loop_key) in loops.iter().enumerate() {
            let value = pair_value(i);
            mesh.loop_attribs_mut()
                .set_vec2(uv, loop_key.as_usize(), value);
            expected.insert(mesh.loop_vertex(loop_key), value);
        }

        mesh.reverse_faces(&[face], false);

        // Every loop still carries the UV its corner had before — no copy,
        // no interpolation, no mirroring.
        for loop_key in mesh.face_loop_keys(face) {
            let corner = mesh.loop_vertex(loop_key);
            let got = mesh
                .loop_attribs()
                .get_vec2(uv, loop_key.as_usize())
                .unwrap();
            assert_eq!(
                got, expected[&corner],
                "loop at corner {corner:?} keeps its own UV"
            );
        }

        // And the traversal order of the UVs is reversed (rotation-fixed at
        // the first corner).
        let after_values: Vec<[f32; 2]> = mesh
            .face_loop_keys(face)
            .into_iter()
            .map(|l| mesh.loop_attribs().get_vec2(uv, l.as_usize()).unwrap())
            .collect();
        let n = after_values.len();
        assert_eq!(after_values[0], pair_value(0));
        for i in 1..n {
            assert_eq!(
                after_values[i],
                pair_value(n - i),
                "UV order reversed at {i}"
            );
        }
    }

    /// §3.2 / §7 — a degenerate (zero-area, collinear) face keeps its
    /// all-zero normal through the reversal rather than acquiring a spurious
    /// direction or NaN.
    #[test]
    fn degenerate_face_normal_stays_zero() {
        let mut mesh = LoopMesh::new();
        let face = create_face_from_verts(
            &mut mesh,
            &[
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(2.0, 0.0, 0.0),
                Vec3::new(3.0, 0.0, 0.0),
            ],
        );
        mesh.calc_face_normals();
        assert_eq!(mesh.face_normal(face), Vec3::ZERO);

        mesh.reverse_faces(&[face], false);

        assert_eq!(mesh.face_normal(face), Vec3::ZERO);
    }

    /// §7 — empty input is a no-op.
    #[test]
    fn empty_input_is_noop() {
        let mut mesh = LoopMesh::new();
        let (left, _, _) = two_quads_shared_edge(&mut mesh);
        mesh.calc_face_normals();
        let before = mesh.face_vert_keys(left);
        let normal = mesh.face_normal(left);

        mesh.reverse_faces(&[], false);

        assert_eq!(mesh.face_vert_keys(left), before);
        assert_eq!(mesh.face_normal(left), normal);
    }
}
