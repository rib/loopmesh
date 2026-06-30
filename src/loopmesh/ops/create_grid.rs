use glam::{Mat4, Vec3};

use crate::loopmesh::*;

impl LoopMesh {
    /// Lay down a flat `x_segments` × `y_segments` lattice of quad faces and
    /// append it to the mesh (CREATE_GRID.md §5).
    ///
    /// The patch is built in the local `z = 0` plane spanning `[-size, +size]²`
    /// (a half-extent, so the full side length is `2 · size`, §2.1), then every
    /// vertex is mapped through `matrix` as a homogeneous point at creation time
    /// (the translation column is honoured, §2.3). `x_segments` and `y_segments`
    /// are each **clamped up to a minimum of 1** (§7), so the smallest result is
    /// a single quad; there is no empty-grid or failure outcome.
    ///
    /// Vertices are minted in **row-major** order — the bottom row left to right,
    /// then upward — with linear index `index(ix, iy) = ix + iy·(nx+1)` (§2.2).
    /// One counter-clockwise quad is built per 2×2 cell through the edge-reusing
    /// face-create primitive (§3), so interior edges between adjacent cells are
    /// minted once and shared; only the rim edges stay boundary, making the
    /// result a single connected manifold patch rather than `nx·ny` disjoint
    /// quads. The corner visit order `(ix-1,iy-1) → (ix,iy-1) → (ix,iy) →
    /// (ix-1,iy)` winds CCW as seen from `+z` in local space (§3).
    ///
    /// Create-grid is purely constructive and **additive**: it reads no existing
    /// geometry, never clears the mesh, and never merges into prior verts even
    /// where the new lattice is geometrically coincident with them (§ intro, §7).
    /// With `size = 0` the full topology is still built — correct vertex, edge and
    /// face counts and connectivity — with every vertex collapsed onto the single
    /// point `matrix·(0,0,0)`; the coincident verts are not de-duplicated (§7).
    ///
    /// `calc_uvs` is accepted for call-shape compatibility but is presently a
    /// deferral: loopmesh writes no default UV unwrap (the §4 UV row is tracked
    /// separately and stays unimplemented this cycle). With no UV layer populated
    /// the topology is identical with the flag set or clear, matching §7's
    /// "`calc_uvs` set but no UV layer present" no-op.
    ///
    /// See `radial-mesh-guides/operators/CREATE_GRID.md` for the full contract.
    pub fn create_grid(
        &mut self,
        x_segments: i32,
        y_segments: i32,
        size: f32,
        matrix: Mat4,
        calc_uvs: bool,
    ) {
        let _ = calc_uvs;

        // §7 — clamp each segment count up to a minimum of 1.
        let nx = x_segments.max(1) as usize;
        let ny = y_segments.max(1) as usize;

        let vert_columns = nx + 1;
        let vert_rows = ny + 1;

        // §2 — lay down the (nx+1)×(ny+1) lattice in row-major order, mapping each
        // local z=0 position through `matrix` at creation. Indexed by
        // `index(ix, iy) = ix + iy·(nx+1)` (§2.2).
        let mut lattice: Vec<VertexKey> = Vec::with_capacity(vert_columns * vert_rows);
        for iy in 0..vert_rows {
            let local_y = ((iy as f32 * 2.0 / ny as f32) - 1.0) * size;
            for ix in 0..vert_columns {
                let local_x = ((ix as f32 * 2.0 / nx as f32) - 1.0) * size;
                let position = matrix.transform_point3(Vec3::new(local_x, local_y, 0.0));
                lattice.push(self.create_vertex(position));
            }
        }
        let index = |ix: usize, iy: usize| ix + iy * vert_columns;

        // §3 — one CCW quad per 2×2 cell through the edge-reusing face-create
        // primitive, so interior edges are minted once and shared. Cells are
        // visited in row-major order, fixing the face index order.
        let mut new_faces: Vec<FaceKey> = Vec::with_capacity(nx * ny);
        for iy in 1..=ny {
            for ix in 1..=nx {
                let corners = [
                    lattice[index(ix - 1, iy - 1)],
                    lattice[index(ix, iy - 1)],
                    lattice[index(ix, iy)],
                    lattice[index(ix - 1, iy)],
                ];
                new_faces.push(self.face_create_verts(&corners, FaceCreateFlags::NONE));
            }
        }

        // §1 — compute the new faces' normals (a run-flag post-pass). Face
        // normals first, then the lattice verts (each vertex normal is the
        // angle-weighted sum of its incident faces' stored normals).
        for &face in &new_faces {
            self.face_calc_normal(face);
        }
        for &vert in &lattice {
            self.vert_normal_update(vert);
        }
    }
}

#[cfg(test)]
mod tests {
    use glam::{Mat4, Quat, Vec3};

    use crate::loopmesh::*;

    const EPS: f32 = 1e-5;

    /// True when some mesh vertex sits within `EPS` of `target`.
    fn has_vertex_near(mesh: &LoopMesh, target: Vec3) -> bool {
        mesh.vert_keys()
            .into_iter()
            .any(|v| (mesh.vertex_position(v) - target).length() < EPS)
    }

    /// §3 / §6 / §7: the minimal `1 × 1` grid is a single CCW quad spanning
    /// `[-size, +size]²` — `ΔV = 4, ΔE = 4, ΔF = 1`, one 4-gon, the four corners
    /// at the half-extent.
    #[test]
    fn create_grid_1x1_single_quad() {
        let mut mesh = LoopMesh::new();
        let size = 2.0;

        mesh.create_grid(1, 1, size, Mat4::IDENTITY, false);

        assert_eq!(
            (mesh.vertex_count(), mesh.edge_count(), mesh.face_count()),
            (4, 4, 1)
        );
        assert_eq!(mesh.loop_count(), 4);
        let face = mesh.face_keys()[0];
        assert_eq!(mesh.face_vert_keys(face).len(), 4);
        for corner in [
            Vec3::new(-size, -size, 0.0),
            Vec3::new(size, -size, 0.0),
            Vec3::new(size, size, 0.0),
            Vec3::new(-size, size, 0.0),
        ] {
            assert!(has_vertex_near(&mesh, corner), "missing corner {corner:?}");
        }
        // CCW in the z=0 plane gives a +z face normal.
        assert!(mesh.face_normal(face).z > 0.9, "quad faces +z");
        mesh.validate_topology();
    }

    /// §2 / §3 / §6: a rectangular `3 × 2` lattice has `(nx+1)(ny+1) = 12` verts,
    /// `nx·ny = 6` quads, and `nx·(ny+1) + ny·(nx+1) = 9 + 8 = 17` edges. The
    /// edge count is the load-bearing assert: 6 disjoint quads would carry 24
    /// edges, so 17 pins the shared-edge fusion of the edge-reusing face-create.
    #[test]
    fn create_grid_3x2_rectangular_lattice_shared_edges() {
        let mut mesh = LoopMesh::new();
        let size = 1.0;

        mesh.create_grid(3, 2, size, Mat4::IDENTITY, false);

        assert_eq!(mesh.vertex_count(), 12);
        assert_eq!(mesh.face_count(), 6);
        assert_eq!(mesh.edge_count(), 17, "interior edges shared, not duplicated");
        assert_eq!(mesh.loop_count(), 4 * 6);
        for face in mesh.face_keys() {
            assert_eq!(mesh.face_vert_keys(face).len(), 4, "every face a quad");
        }
        // Spacing 2·size/nx × 2·size/ny: columns at x ∈ {-1, -1/3, 1/3, 1},
        // rows at y ∈ {-1, 0, 1}. Probe a couple of interior lattice points.
        assert!(has_vertex_near(&mesh, Vec3::new(-1.0 / 3.0, 0.0, 0.0)));
        assert!(has_vertex_near(&mesh, Vec3::new(1.0 / 3.0, 0.0, 0.0)));
        // Euler relation for a simply-connected sheet: V − E + F = 1.
        assert_eq!(
            mesh.vertex_count() as i64 - mesh.edge_count() as i64 + mesh.face_count() as i64,
            1
        );
        mesh.validate_topology();
    }

    /// §2.3: every vertex is mapped through `matrix` at creation. With a
    /// translate+rotate the `1 × 1` corners land at the transformed half-extent
    /// points, not the local ones.
    #[test]
    fn create_grid_matrix_transform() {
        let mut mesh = LoopMesh::new();
        let size = 1.0;
        let matrix = Mat4::from_scale_rotation_translation(
            Vec3::ONE,
            Quat::from_rotation_z(std::f32::consts::FRAC_PI_4),
            Vec3::new(5.0, -2.0, 1.0),
        );

        mesh.create_grid(1, 1, size, matrix, false);

        assert_eq!(
            (mesh.vertex_count(), mesh.edge_count(), mesh.face_count()),
            (4, 4, 1)
        );
        for local in [
            Vec3::new(-size, -size, 0.0),
            Vec3::new(size, -size, 0.0),
            Vec3::new(size, size, 0.0),
            Vec3::new(-size, size, 0.0),
        ] {
            let expected = matrix.transform_point3(local);
            assert!(
                has_vertex_near(&mesh, expected),
                "missing transformed corner {expected:?}"
            );
        }
        mesh.validate_topology();
    }

    /// §7: `x_segments = 0` (or any value below 1) clamps up to 1 — identical to
    /// an explicit `x_segments = 1` build, never an empty or zero-face result.
    #[test]
    fn create_grid_segment_clamp_zero_equals_one() {
        let mut clamped = LoopMesh::new();
        clamped.create_grid(0, 2, 1.0, Mat4::IDENTITY, false);

        let mut explicit = LoopMesh::new();
        explicit.create_grid(1, 2, 1.0, Mat4::IDENTITY, false);

        assert_eq!(clamped.vertex_count(), explicit.vertex_count());
        assert_eq!(clamped.edge_count(), explicit.edge_count());
        assert_eq!(clamped.face_count(), explicit.face_count());
        // 1×2 grid: 2·3 = 6 verts, 2 quads, 1·3 + 2·2 = 7 edges.
        assert_eq!(
            (clamped.vertex_count(), clamped.edge_count(), clamped.face_count()),
            (6, 7, 2)
        );
        // A negative count clamps the same way.
        let mut negative = LoopMesh::new();
        negative.create_grid(-5, 2, 1.0, Mat4::IDENTITY, false);
        assert_eq!(negative.face_count(), explicit.face_count());
        clamped.validate_topology();
        negative.validate_topology();
    }

    /// §7 / intro: create-grid appends to a populated mesh without clearing it.
    /// On a mesh already carrying a quad island the result is that island plus
    /// the fresh grid as a disjoint patch — additive counts, no merge.
    #[test]
    fn create_grid_appends_onto_populated_mesh() {
        let mut mesh = LoopMesh::new();

        // An unrelated quad island off to the side.
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

        // Append a 3×2 grid: ΔV = 12, ΔE = 17, ΔF = 6.
        mesh.create_grid(3, 2, 1.0, Mat4::IDENTITY, false);

        assert_eq!(
            (mesh.vertex_count(), mesh.edge_count(), mesh.face_count()),
            (4 + 12, 4 + 17, 1 + 6)
        );
        // The prior island's corners are untouched.
        for &p in &corners {
            assert!(has_vertex_near(&mesh, p), "prior island vertex {p:?} moved");
        }
        mesh.validate_topology();
    }

    /// §7: with `size = 0` every local position collapses to the origin, so all
    /// verts land on the single point `matrix·(0,0,0)` — yet the full topology is
    /// built (correct counts, connectivity) with no de-duplication.
    #[test]
    fn create_grid_size_zero_collapsed_patch() {
        let mut mesh = LoopMesh::new();

        mesh.create_grid(2, 2, 0.0, Mat4::IDENTITY, false);

        // Full 2×2-cell topology survives: 9 verts, 4 faces, 12 edges.
        assert_eq!(mesh.vertex_count(), 9);
        assert_eq!(mesh.face_count(), 4);
        assert_eq!(mesh.edge_count(), 12);
        // All verts coincident at the origin (no merge).
        for v in mesh.vert_keys() {
            assert!((mesh.vertex_position(v) - Vec3::ZERO).length() < EPS);
        }
        for face in mesh.face_keys() {
            assert_eq!(mesh.face_vert_keys(face).len(), 4);
        }
        mesh.validate_topology();
    }
}
