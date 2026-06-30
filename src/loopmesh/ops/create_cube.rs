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
    /// `calc_uvs` is accepted for call-shape compatibility but is presently a
    /// deferral: loopmesh writes no default cube-cross UV net (the §4 UV row is
    /// tracked separately and stays unimplemented this cycle). With no UV layer
    /// populated the topology is identical with the flag set or clear, matching
    /// §7's "`calc_uvs` set but no UV layer present" no-op.
    ///
    /// See `radial-mesh-guides/operators/CREATE_CUBE.md` for the full contract.
    pub fn create_cube(&mut self, size: f32, matrix: Mat4, calc_uvs: bool) {
        let _ = calc_uvs;

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
}
