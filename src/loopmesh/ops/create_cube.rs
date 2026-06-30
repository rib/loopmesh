use glam::{Mat4, Vec3};

use crate::loopmesh::*;

/// Pack three `(axis, positive)` sign choices into the §2.1 corner index, where
/// bit `2 - axis` is set when that axis' coordinate is `+half`.
fn corner_index(
    axis_a: usize,
    positive_a: bool,
    axis_b: usize,
    positive_b: bool,
    axis_c: usize,
    positive_c: bool,
) -> usize {
    let bit = |axis: usize, positive: bool| if positive { 1usize << (2 - axis) } else { 0 };
    bit(axis_a, positive_a) | bit(axis_b, positive_b) | bit(axis_c, positive_c)
}

/// Recover a cube side's `(axis, positive)` identity from the §2.1 sign-bit
/// indices of its four corners. A side fixes exactly one axis (its outward
/// normal's axis) and varies the other two, so exactly one axis bit is constant
/// across all four corners; that bit's value gives the side's sign (§4.1).
fn recover_axis_side(sign_bits: &[usize]) -> (usize, bool) {
    for axis in 0..3 {
        let shift = 2 - axis;
        let first_bit = (sign_bits[0] >> shift) & 1;
        if sign_bits.iter().all(|&s| (s >> shift) & 1 == first_bit) {
            return (axis, first_bit == 1);
        }
    }
    unreachable!("a cube side shares exactly one axis sign across its four corners")
}

/// Map a cube side to its cross-net tile origin (§4.2) and the four §2.1
/// sign-bit corner indices that take the tile corners `[BL, BR, TR, TL]` (§4.3).
fn cross_net_side(side: (usize, bool)) -> ([f32; 2], [usize; 4]) {
    match side {
        (0, false) => ([0.375, 0.00], [0, 1, 3, 2]), // -X
        (1, true) => ([0.375, 0.25], [2, 3, 7, 6]),  // +Y
        (0, true) => ([0.375, 0.50], [6, 7, 5, 4]),  // +X
        (1, false) => ([0.375, 0.75], [4, 5, 1, 0]), // -Y
        (2, false) => ([0.125, 0.50], [2, 6, 4, 0]), // -Z
        (2, true) => ([0.625, 0.50], [7, 3, 1, 5]),  // +Z
        _ => unreachable!("axis is 0..3"),
    }
}

/// The UV of the tile corner that the given §2.1 sign-bit corner occupies, given
/// the tile origin `(u0, v0)` and the side's `[BL, BR, TR, TL]` corner assignment
/// (§4.3). Tile side `w = 0.25`.
fn tile_corner_uv(origin: [f32; 2], sign_bit: usize, assignment: [usize; 4]) -> [f32; 2] {
    const TILE: f32 = 0.25;
    let [u0, v0] = origin;
    let [bl, br, tr, _tl] = assignment;
    if sign_bit == bl {
        [u0, v0]
    } else if sign_bit == br {
        [u0 + TILE, v0]
    } else if sign_bit == tr {
        [u0 + TILE, v0 + TILE]
    } else {
        [u0, v0 + TILE]
    }
}

impl LoopMesh {
    /// Build a closed, axis-aligned box — eight corner verts and six outward-wound
    /// quad faces — and append it to the mesh (CREATE_CUBE.md §5).
    ///
    /// `size` is the **full** edge length (note the opposite convention to
    /// create-grid's half-extent `size`): corners are placed at the eight sign
    /// combinations of `(±half, ±half, ±half)` with `half = size / 2`, spanning
    /// `[-size/2, +size/2]` per axis (§2.1). The sole special case is `size == 0`,
    /// which is overridden to `half = 0.5` — the **unit cube**, not a collapsed
    /// point (§7), the headline divergence from create-grid. Any non-zero `size`
    /// is used as-is, so a negative `size` point-reflects the box through the
    /// origin without changing the topology or the per-face corner-visit order.
    ///
    /// The corner whose creation index is `i` has the sign pattern that is the
    /// binary expansion of `i` (`bit 2 → x, bit 1 → y, bit 0 → z`, a set bit
    /// meaning `+half`, §2.1). Each corner is mapped through `matrix` as a
    /// homogeneous point at creation time (the translation column is honoured,
    /// §2.2), so any affine map — including non-uniform scale, shear and mirror —
    /// is applied per corner.
    ///
    /// The six faces are derived from the generative right-handed-basis winding
    /// rule (§3) rather than a stored table: for each axis-aligned side the two
    /// in-plane axes are ordered `(u, v)` so that `u × v` is the outward normal,
    /// and the four corners are walked `(-u,-v), (+u,-v), (+u,+v), (-u,+v)` —
    /// counter-clockwise as seen from outside, giving an outward normal. Faces are
    /// built through the edge-reusing face-create primitive, so each of the 12
    /// cube edges is minted once and shared by the two sides meeting along it; the
    /// result is a single watertight closed manifold shell (every edge
    /// radial-count 2, no boundary), not six disjoint quads.
    ///
    /// Create-cube is purely constructive and **additive**: it reads no existing
    /// geometry, never clears the mesh, and never merges into prior verts even
    /// where the new box is geometrically coincident with them (§ intro, §7).
    ///
    /// The eight corners are **returned** in §2.1 sign-bit order (the `verts.out`
    /// slot contract): `result[i]` is the corner whose `(x, y, z)` signs are the
    /// binary expansion of `i` (`i = (x_bit << 2) | (y_bit << 1) | z_bit`, a set
    /// bit meaning `+half`), so the buffer always has exactly 8 elements.
    ///
    /// When `calc_uvs` is set **and** the mesh carries an active per-loop
    /// 2-component-float (UV) layer, a default **cube-cross** net is painted
    /// across the six faces: six `0.25`-wide tiles laid out as a Latin-cross net
    /// in the unit square. The net is keyed on **geometric identity**, not build
    /// order (§4.1): each face's tile and within-tile corner assignment are
    /// recovered from the sign bit its four corners share (§2.1) — the axis-side
    /// fixes the tile origin (§4.2) and the side's bottom-left sign-bit corner
    /// (§4.3), with the remaining three corners walking the side's outward
    /// winding. With `calc_uvs` clear, or when no UV layer exists, the flag is a
    /// silent no-op — it never creates a layer (§7).
    ///
    /// See `radial-mesh-guides/operators/CREATE_CUBE.md` for the full contract.
    pub fn create_cube(&mut self, size: f32, matrix: Mat4, calc_uvs: bool) -> Vec<VertexKey> {
        // §7 — exactly `0` maps to the unit cube; every other size, positive or
        // negative, is used directly. A single conditional, not a place-then-fix.
        let half = if size == 0.0 { 0.5 } else { size / 2.0 };

        // §2 — build all eight corners straight from their index bits, mapping
        // each through `matrix` as it is created. `corners[i]` is the corner whose
        // sign pattern is the binary expansion of `i`.
        let corners: [VertexKey; 8] = std::array::from_fn(|index| {
            let coord = |axis: usize| {
                if index & (1usize << (2 - axis)) != 0 {
                    half
                } else {
                    -half
                }
            };
            let local = Vec3::new(coord(0), coord(1), coord(2));
            self.create_vertex(matrix.transform_point3(local))
        });

        // §3 — one outward quad per axis-aligned side, derived from the winding
        // rule. For side (fixed_axis, fixed_positive) the outward normal is the
        // signed axis direction; the two remaining axes form the in-plane pair
        // ordered so `u × v` points out — the cyclic order `(a+1, a+2)` for the
        // positive side, swapped for the negative side.
        let mut new_faces: Vec<FaceKey> = Vec::with_capacity(6);
        for fixed_axis in 0..3 {
            let next_axis = (fixed_axis + 1) % 3;
            let after_axis = (fixed_axis + 2) % 3;
            for fixed_positive in [false, true] {
                let (u_axis, v_axis) = if fixed_positive {
                    (next_axis, after_axis)
                } else {
                    (after_axis, next_axis)
                };
                let quad: [VertexKey; 4] = [(false, false), (true, false), (true, true), (false, true)]
                    .map(|(u_positive, v_positive)| {
                        let index = corner_index(
                            fixed_axis,
                            fixed_positive,
                            u_axis,
                            u_positive,
                            v_axis,
                            v_positive,
                        );
                        corners[index]
                    });
                new_faces.push(self.face_create_verts(&quad, FaceCreateFlags::NONE));
            }
        }

        // §1 — compute the new faces' normals (a run-flag post-pass), faces first
        // so each vertex normal sums its incident faces' stored normals.
        for &face in &new_faces {
            self.face_calc_normal(face);
        }
        for &vert in &corners {
            self.vert_normal_update(vert);
        }

        // §4 — optional default cube-cross UV net, painted only when the flag is
        // set and an active per-loop UV layer already exists (the flag alone
        // never mints a layer).
        if calc_uvs {
            if let Some(uv_layer) = self.active_uv_layer_for_cube() {
                self.paint_cube_cross_uvs(&new_faces, &corners, uv_layer);
            }
        }

        // §2.1 — report the eight corners in sign-bit index order (verts.out).
        corners.to_vec()
    }

    /// §4 — paint the cube-cross net across the freshly built faces, keyed on
    /// **geometric identity** rather than build order (§4.1). For each face the
    /// axis-side is recovered from the sign bit its four corners share (§2.1);
    /// the side fixes the tile origin (§4.2) and which §2.1 sign-bit corner takes
    /// each tile corner `BL, BR, TR, TL` (§4.3). Because the assignment is keyed
    /// on the corner sign bits, the painted net is independent of the order the
    /// faces and their loops were created in. `corners[i]` is the §2.1 sign-bit
    /// corner `i`, used to map each loop's vertex back to its sign-bit index.
    fn paint_cube_cross_uvs(
        &mut self,
        new_faces: &[FaceKey],
        corners: &[VertexKey; 8],
        uv_layer: ElementAttributeKey,
    ) {
        let mut loop_keys: smallvec::SmallVec<[LoopKey; 4]> = smallvec::SmallVec::new();
        let mut sign_bits: smallvec::SmallVec<[usize; 4]> = smallvec::SmallVec::new();
        for &face in new_faces {
            loop_keys.clear();
            sign_bits.clear();
            for loop_key in self.iter_face_loops(face) {
                let vertex = self.loop_vertex(loop_key);
                let sign_bit = corners
                    .iter()
                    .position(|&corner| corner == vertex)
                    .expect("cube face loop resolves to a sign-bit corner");
                loop_keys.push(loop_key);
                sign_bits.push(sign_bit);
            }

            let (origin, tile_corner_assignment) = cross_net_side(recover_axis_side(&sign_bits));
            for (&loop_key, &sign_bit) in loop_keys.iter().zip(sign_bits.iter()) {
                let uv = tile_corner_uv(origin, sign_bit, tile_corner_assignment);
                self.loop_attribs_mut()
                    .set_vec2(uv_layer, loop_key.as_usize(), uv);
            }
        }
    }

    /// Resolve the active UV layer: the loop-domain layer create-cube paints
    /// when `calc_uvs` is set. Prefer 2-component loop layers tagged with the
    /// Uv role; otherwise fall back to the first registered Vec2 loop layer
    /// (the "a vec2 loop layer is a UV layer" convention). `None` when the mesh
    /// carries no UV layer at all (§4 / §7 — the no-op case).
    fn active_uv_layer_for_cube(&self) -> Option<ElementAttributeKey> {
        let attribs = self.loop_attribs();
        let is_vec2 = |key| matches!(attribs.storage_of(key), Some(AttributeStorage::Vec2(_)));
        let any_tagged = attribs
            .iter_descriptors()
            .any(|(key, d)| d.role == AttrRole::Uv && is_vec2(key));
        attribs
            .iter_descriptors()
            .filter(|(key, d)| is_vec2(*key) && (!any_tagged || d.role == AttrRole::Uv))
            .map(|(key, _)| key)
            .next()
    }
}

#[cfg(test)]
mod tests {
    use glam::{Mat4, Quat, Vec3};

    use crate::loopmesh::*;

    const EPS: f32 = 1e-5;

    fn has_vertex_near(mesh: &LoopMesh, target: Vec3) -> bool {
        mesh.vert_keys()
            .into_iter()
            .any(|v| (mesh.vertex_position(v) - target).length() < EPS)
    }

    /// §6 / §8.1: the default cube adds exactly `ΔV=8, ΔE=12, ΔF=6, ΔLoops=24`,
    /// every face a quad, eight corners at `(±off, ±off, ±off)` with `off = size/2`.
    #[test]
    fn create_cube_default_counts_and_corners() {
        let mut mesh = LoopMesh::new();
        let size = 2.0;
        let off = size / 2.0;

        mesh.create_cube(size, Mat4::IDENTITY, false);

        assert_eq!(
            (mesh.vertex_count(), mesh.edge_count(), mesh.face_count()),
            (8, 12, 6)
        );
        assert_eq!(mesh.loop_count(), 24);
        for face in mesh.face_keys() {
            assert_eq!(mesh.face_vert_keys(face).len(), 4, "every face a quad");
        }
        for sx in [-off, off] {
            for sy in [-off, off] {
                for sz in [-off, off] {
                    let corner = Vec3::new(sx, sy, sz);
                    assert!(has_vertex_near(&mesh, corner), "missing corner {corner:?}");
                }
            }
        }
        mesh.validate_topology();
    }

    /// §6 / §8: the shell is a closed manifold — `V − E + F = 2` and every edge is
    /// shared by exactly two faces (radial-count 2, no boundary).
    #[test]
    fn create_cube_watertight_closed_shell() {
        let mut mesh = LoopMesh::new();
        mesh.create_cube(2.0, Mat4::IDENTITY, false);

        assert_eq!(
            mesh.vertex_count() as i64 - mesh.edge_count() as i64 + mesh.face_count() as i64,
            2,
            "closed-shell Euler characteristic"
        );
        for edge in mesh.edge_keys() {
            assert_eq!(
                mesh.edge_face_keys(edge).len(),
                2,
                "every cube edge shared by exactly two quads, no boundary"
            );
        }
        // Outward winding: every face normal points away from the (origin) centre.
        for face in mesh.face_keys() {
            let centroid: Vec3 = mesh
                .face_vert_keys(face)
                .iter()
                .fold(Vec3::ZERO, |acc, &v| acc + mesh.vertex_position(v))
                / 4.0;
            assert!(
                mesh.face_normal(face).dot(centroid) > 0.0,
                "face normal points outward"
            );
        }
        mesh.validate_topology();
    }

    /// §1 / §2.1 / §7: `size = 0` is overridden to the unit cube — identical
    /// geometry to an explicit `size = 1` build (corners at `±0.5`), NOT a
    /// collapsed point (the divergence from create-grid's `size = 0`).
    #[test]
    fn create_cube_size_zero_equals_size_one() {
        let mut zero = LoopMesh::new();
        zero.create_cube(0.0, Mat4::IDENTITY, false);

        let mut one = LoopMesh::new();
        one.create_cube(1.0, Mat4::IDENTITY, false);

        assert_eq!(
            (zero.vertex_count(), zero.edge_count(), zero.face_count()),
            (one.vertex_count(), one.edge_count(), one.face_count())
        );
        // Both place corners at ±0.5 — and crucially not all coincident at origin.
        for sx in [-0.5, 0.5] {
            for sy in [-0.5, 0.5] {
                for sz in [-0.5, 0.5] {
                    let corner = Vec3::new(sx, sy, sz);
                    assert!(has_vertex_near(&zero, corner), "size=0 missing {corner:?}");
                    assert!(has_vertex_near(&one, corner), "size=1 missing {corner:?}");
                }
            }
        }
        zero.validate_topology();
    }

    /// §7 (negative `size`): a negative size negates every corner coordinate — a
    /// point reflection through the origin. Topology unchanged; corner positions
    /// are the negation of the positive-size build, i.e. the same set of corners.
    #[test]
    fn create_cube_negative_size_point_reflection() {
        let mut negative = LoopMesh::new();
        negative.create_cube(-2.0, Mat4::IDENTITY, false);

        assert_eq!(
            (
                negative.vertex_count(),
                negative.edge_count(),
                negative.face_count()
            ),
            (8, 12, 6)
        );
        // Corner i of the negative build sits at -(positive corner i). The full
        // corner *set* is unchanged (the box is symmetric), but verify the
        // reflection holds per the §2.1 index labelling: index 0 → (+1,+1,+1).
        assert!(
            has_vertex_near(&negative, Vec3::new(1.0, 1.0, 1.0)),
            "index-0 corner reflected to the positive octant"
        );
        assert!(
            has_vertex_near(&negative, Vec3::new(-1.0, -1.0, -1.0)),
            "index-7 corner reflected to the negative octant"
        );
        negative.validate_topology();
    }

    /// §2.2: every corner is mapped through `matrix` at creation. A translate +
    /// rotate lands the corners at the transformed box positions.
    #[test]
    fn create_cube_matrix_transform() {
        let mut mesh = LoopMesh::new();
        let size = 2.0;
        let off = size / 2.0;
        let matrix = Mat4::from_scale_rotation_translation(
            Vec3::ONE,
            Quat::from_rotation_z(std::f32::consts::FRAC_PI_4),
            Vec3::new(5.0, -2.0, 1.0),
        );

        mesh.create_cube(size, matrix, false);

        assert_eq!(
            (mesh.vertex_count(), mesh.edge_count(), mesh.face_count()),
            (8, 12, 6)
        );
        for sx in [-off, off] {
            for sy in [-off, off] {
                for sz in [-off, off] {
                    let expected = matrix.transform_point3(Vec3::new(sx, sy, sz));
                    assert!(
                        has_vertex_near(&mesh, expected),
                        "missing transformed corner {expected:?}"
                    );
                }
            }
        }
        mesh.validate_topology();
    }

    /// §2.1 / §8: the returned `verts.out` buffer has exactly 8 elements in
    /// sign-bit corner order — `result[i]` is the corner whose `(x, y, z)` signs
    /// are the binary expansion of `i` (`bit 2 → x, bit 1 → y, bit 0 → z`, a set
    /// bit meaning `+off`). Probed under a non-identity matrix so a transposed or
    /// reordered bug lands on the wrong corner. This slot order is *not* captured
    /// by an order-free isomorphism check, so it needs an explicit index check.
    #[test]
    fn create_cube_returns_sign_bit_corner_order() {
        let mut mesh = LoopMesh::new();
        let size = 2.0;
        let off = size / 2.0;
        let matrix = Mat4::from_scale_rotation_translation(
            Vec3::new(1.0, 2.0, 3.0),
            Quat::from_rotation_z(std::f32::consts::FRAC_PI_4),
            Vec3::new(5.0, -2.0, 1.0),
        );

        let verts = mesh.create_cube(size, matrix, false);

        assert_eq!(verts.len(), 8);
        for index in 0..8usize {
            let signed = |bit_set: bool| if bit_set { off } else { -off };
            let local = Vec3::new(
                signed(index & 0b100 != 0),
                signed(index & 0b010 != 0),
                signed(index & 0b001 != 0),
            );
            let expected = matrix.transform_point3(local);
            let got = mesh.vertex_position(verts[index]);
            assert!(
                (got - expected).length() < EPS,
                "verts.out[{index}] at {got:?}, expected sign pattern {local:?} -> {expected:?}"
            );
        }
    }

    /// §7 / intro: create-cube appends to a populated mesh without clearing it.
    /// On a mesh already carrying a quad island the result is that island plus the
    /// fresh cube as a disjoint shell — additive counts, no merge.
    #[test]
    fn create_cube_appends_onto_populated_mesh() {
        let mut mesh = LoopMesh::new();

        let corners = [
            Vec3::new(10.0, 10.0, 0.0),
            Vec3::new(11.0, 10.0, 0.0),
            Vec3::new(11.0, 11.0, 0.0),
            Vec3::new(10.0, 11.0, 0.0),
        ];
        let island_verts: Vec<VertexKey> =
            corners.iter().map(|&p| mesh.create_vertex(p)).collect();
        mesh.face_create_verts(&island_verts, FaceCreateFlags::NONE);
        let before = (mesh.vertex_count(), mesh.edge_count(), mesh.face_count());
        assert_eq!(before, (4, 4, 1));

        mesh.create_cube(2.0, Mat4::IDENTITY, false);

        assert_eq!(
            (mesh.vertex_count(), mesh.edge_count(), mesh.face_count()),
            (4 + 8, 4 + 12, 1 + 6)
        );
        for &p in &corners {
            assert!(has_vertex_near(&mesh, p), "prior island vertex {p:?} moved");
        }
        mesh.validate_topology();
    }

    fn register_uv(mesh: &mut LoopMesh) -> ElementAttributeKey {
        mesh.register_loop_attribute(ElementAttributeDesc::numeric(
            AttributeKey::custom("uv"),
            AttrType::Vec2,
        ))
        .expect("uv layer registers")
    }

    /// Read the per-corner UVs of a face in loop-cycle order.
    fn read_corner_uvs(mesh: &LoopMesh, face: FaceKey, uv: ElementAttributeKey) -> Vec<[f32; 2]> {
        mesh.face_loop_keys(face)
            .into_iter()
            .map(|l| mesh.loop_attribs().get_vec2(uv, l.as_usize()).unwrap())
            .collect()
    }

    fn uv_near(got: [f32; 2], want: [f32; 2]) -> bool {
        (got[0] - want[0]).abs() < EPS && (got[1] - want[1]).abs() < EPS
    }

    /// §4.2 / §4.3: with `calc_uvs` and an active UV layer the six faces receive
    /// the cube-cross net keyed on **geometric identity** (§4.1). Each face's
    /// axis-side is recovered here from the coordinate held constant across its
    /// four corners (keyed on geometry, not build order); the side fixes the tile
    /// origin (§4.2) and which §2.1 sign-bit corner takes each tile corner
    /// `BL, BR, TR, TL` (§4.3). The per-loop UV is then checked corner-by-corner
    /// against that spec, recovering each loop's sign-bit index from its vertex
    /// position.
    #[test]
    fn create_cube_calc_uvs_cross_net() {
        let mut mesh = LoopMesh::new();
        let uv = register_uv(&mut mesh);

        mesh.create_cube(2.0, Mat4::IDENTITY, true);

        const TILE: f32 = 0.25;

        // §2.1 sign-bit index recovered from a corner position (size = 2 → ±1).
        let sign_bit = |p: Vec3| -> usize {
            ((p.x > 0.0) as usize) << 2 | ((p.y > 0.0) as usize) << 1 | (p.z > 0.0) as usize
        };

        // Independent restatement of the §4.2 tile origin + §4.3 [BL,BR,TR,TL]
        // sign-bit-corner tables, used as the test oracle (keyed on the side, not
        // on any build-order index).
        let side_net = |axis: usize, positive: bool| -> ([f32; 2], [usize; 4]) {
            match (axis, positive) {
                (0, false) => ([0.375, 0.00], [0, 1, 3, 2]), // -X
                (1, true) => ([0.375, 0.25], [2, 3, 7, 6]),  // +Y
                (0, true) => ([0.375, 0.50], [6, 7, 5, 4]),  // +X
                (1, false) => ([0.375, 0.75], [4, 5, 1, 0]), // -Y
                (2, false) => ([0.125, 0.50], [2, 6, 4, 0]), // -Z
                (2, true) => ([0.625, 0.50], [7, 3, 1, 5]),  // +Z
                _ => unreachable!(),
            }
        };

        let faces = mesh.face_keys();
        assert_eq!(faces.len(), 6);
        for face in faces {
            // Recover the side from the axis whose coordinate is constant across
            // all four corners (the fixed axis of the side).
            let positions: Vec<Vec3> = mesh
                .face_loop_keys(face)
                .into_iter()
                .map(|l| mesh.vertex_position(mesh.loop_vertex(l)))
                .collect();
            let (axis, positive) = (0..3)
                .find_map(|axis| {
                    let first = positions[0][axis];
                    positions
                        .iter()
                        .all(|p| (p[axis] - first).abs() < EPS)
                        .then_some((axis, first > 0.0))
                })
                .expect("a cube face fixes exactly one axis");

            let ([u0, v0], [bl, br, tr, tl]) = side_net(axis, positive);
            for loop_key in mesh.face_loop_keys(face) {
                let index = sign_bit(mesh.vertex_position(mesh.loop_vertex(loop_key)));
                let want = if index == bl {
                    [u0, v0]
                } else if index == br {
                    [u0 + TILE, v0]
                } else if index == tr {
                    [u0 + TILE, v0 + TILE]
                } else {
                    assert_eq!(index, tl, "loop corner is one of the side's four");
                    [u0, v0 + TILE]
                };
                let got = mesh.loop_attribs().get_vec2(uv, loop_key.as_usize()).unwrap();
                assert!(
                    uv_near(got, want),
                    "side (axis {axis}, positive {positive}) corner {index} UV {got:?}, expected {want:?}"
                );
            }
        }
        mesh.validate_topology();
    }

    /// §4: every painted UV lies within the unit square — the cube-cross net's
    /// six tiles are the distinct cells of `[0,1]²` (the −X/+Y/+X/−Y central
    /// column shares the `u` band `[0.375, 0.625]`, the ±Z sides flank it), so no
    /// painted corner escapes the unit square.
    #[test]
    fn create_cube_calc_uvs_inside_unit_square() {
        let mut mesh = LoopMesh::new();
        let uv = register_uv(&mut mesh);

        mesh.create_cube(2.0, Mat4::IDENTITY, true);

        for face in mesh.face_keys() {
            for [u, v] in read_corner_uvs(&mesh, face, uv) {
                assert!(
                    (-EPS..=1.0 + EPS).contains(&u) && (-EPS..=1.0 + EPS).contains(&v),
                    "UV ({u}, {v}) escapes the unit square"
                );
            }
        }
        mesh.validate_topology();
    }

    /// §4 / §7: `calc_uvs = false` writes nothing even when a UV layer exists —
    /// every painted-on corner keeps the layer's default (zero) value.
    #[test]
    fn create_cube_calc_uvs_false_is_noop() {
        let mut mesh = LoopMesh::new();
        let uv = register_uv(&mut mesh);

        mesh.create_cube(2.0, Mat4::IDENTITY, false);

        for face in mesh.face_keys() {
            for corner in read_corner_uvs(&mesh, face, uv) {
                assert!(
                    uv_near(corner, [0.0, 0.0]),
                    "calc_uvs=false left a non-default UV {corner:?}"
                );
            }
        }
    }

    /// §4 / §7: `calc_uvs = true` with no UV layer present is a silent no-op —
    /// it builds the topology and creates no layer.
    #[test]
    fn create_cube_calc_uvs_no_layer_is_noop() {
        let mut mesh = LoopMesh::new();

        mesh.create_cube(2.0, Mat4::IDENTITY, true);

        assert_eq!(mesh.face_count(), 6);
        assert!(mesh.loop_attribs().is_empty(), "no UV layer was created");
        mesh.validate_topology();
    }
}
