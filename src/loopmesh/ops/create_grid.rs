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
    /// They are **returned** in that same row-major order (the `verts.out` slot
    /// contract, §2.2): the returned buffer has `(nx+1)·(ny+1)` elements and
    /// `result[ix + iy·(nx+1)]` is the vertex at lattice coordinate `(ix, iy)`.
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
    /// When `calc_uvs` is set **and** the mesh carries an active per-loop
    /// 2-component-float (UV) layer, a default unit-square unwrap is painted
    /// across the grid: `du = 1/nx`, `dv = 1/ny`, each cell occupying a
    /// `du × dv` tile that mirrors its spatial position one-to-one, with UV
    /// `(0,0)` at the bottom-left corner of the patch and `(1,1)` at the
    /// top-right (§4). With `calc_uvs` clear, or when no UV layer exists, the
    /// flag is a silent no-op — it never creates a layer (§7).
    ///
    /// See `radial-mesh-guides/operators/CREATE_GRID.md` for the full contract.
    pub fn create_grid(
        &mut self,
        x_segments: i32,
        y_segments: i32,
        size: f32,
        matrix: Mat4,
        calc_uvs: bool,
    ) -> Vec<VertexKey> {
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

        // §4 — optional default unit-square UV unwrap, painted only when the
        // flag is set and an active per-loop UV layer already exists (the flag
        // alone never mints a layer).
        if calc_uvs {
            if let Some(uv_layer) = self.active_uv_loop_layer() {
                self.paint_grid_uvs(&new_faces, nx, ny, uv_layer);
            }
        }

        // §2.2 — report the created verts in row-major order (the verts.out slot).
        lattice
    }

    /// §4 — paint the seamless `[0,1]²` planar unwrap across the freshly built
    /// grid faces. `new_faces` is in row-major cell build order (§3), so the
    /// face at position `face_index` sits at 0-based cell column
    /// `ix = face_index mod nx` and row `iy = face_index div nx`. Each cell
    /// occupies a `du × dv` tile, and the four loop corners — in the
    /// `bottom-left → bottom-right → top-right → top-left` visit order the
    /// faces were created with — receive the tile's corners so the unwrap
    /// matches the spatial layout one-to-one.
    fn paint_grid_uvs(
        &mut self,
        new_faces: &[FaceKey],
        nx: usize,
        ny: usize,
        uv_layer: ElementAttributeKey,
    ) {
        let du = 1.0 / nx as f32;
        let dv = 1.0 / ny as f32;

        let mut loop_keys: smallvec::SmallVec<[LoopKey; 4]> = smallvec::SmallVec::new();
        for (face_index, &face) in new_faces.iter().enumerate() {
            let cell_ix = (face_index % nx) as f32;
            let cell_iy = (face_index / nx) as f32;
            let u_lo = cell_ix * du;
            let u_hi = (cell_ix + 1.0) * du;
            let v_lo = cell_iy * dv;
            let v_hi = (cell_iy + 1.0) * dv;
            let corner_uvs = [
                [u_lo, v_lo], // corner 0: bottom-left
                [u_hi, v_lo], // corner 1: bottom-right
                [u_hi, v_hi], // corner 2: top-right
                [u_lo, v_hi], // corner 3: top-left
            ];

            loop_keys.clear();
            for loop_key in self.iter_face_loops(face) {
                loop_keys.push(loop_key);
            }
            for (corner, &loop_key) in loop_keys.iter().enumerate() {
                self.loop_attribs_mut()
                    .set_vec2(uv_layer, loop_key.as_usize(), corner_uvs[corner]);
            }
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

    /// §2.2: the returned `verts.out` buffer has `(nx+1)·(ny+1)` elements in
    /// row-major order — `result[ix + iy·(nx+1)]` is the vertex at lattice
    /// coordinate `(ix, iy)`. Probed on a rectangular `nx ≠ ny` grid so a
    /// transposed / column-major bug fails (the strides differ: 4 vs 3).
    #[test]
    fn create_grid_returns_row_major_verts_out() {
        let mut mesh = LoopMesh::new();
        let (nx, ny, size) = (3usize, 2usize, 1.0f32);

        let verts = mesh.create_grid(nx as i32, ny as i32, size, Mat4::IDENTITY, false);

        let vert_columns = nx + 1;
        assert_eq!(verts.len(), vert_columns * (ny + 1));

        // Expected local lattice position for coordinate (ix, iy).
        let expected = |ix: usize, iy: usize| {
            Vec3::new(
                ((ix as f32 * 2.0 / nx as f32) - 1.0) * size,
                ((iy as f32 * 2.0 / ny as f32) - 1.0) * size,
                0.0,
            )
        };

        // Probe several cells, including asymmetric ones (ix != iy) and the far
        // corners, so a column-major layout would land on the wrong vertex.
        for (ix, iy) in [(0, 0), (3, 0), (0, 2), (3, 2), (1, 0), (0, 1), (2, 1)] {
            let key = verts[ix + iy * vert_columns];
            let got = mesh.vertex_position(key);
            let want = expected(ix, iy);
            assert!(
                (got - want).length() < EPS,
                "verts.out[{ix}+{iy}·{vert_columns}] at {got:?}, expected ({ix},{iy}) = {want:?}"
            );
        }
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

    /// §4: a `1 × 1` grid with `calc_uvs` and an active UV layer maps its single
    /// quad onto the whole unit square — `(0,0)` bottom-left through `(1,1)`
    /// top-right, in the `BL → BR → TR → TL` corner order the face was built.
    #[test]
    fn create_grid_calc_uvs_1x1_unit_square() {
        let mut mesh = LoopMesh::new();
        let uv = register_uv(&mut mesh);

        mesh.create_grid(1, 1, 1.0, Mat4::IDENTITY, true);

        let face = mesh.face_keys()[0];
        let corners = read_corner_uvs(&mesh, face, uv);
        let expected = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        for (i, want) in expected.iter().enumerate() {
            assert!(
                uv_near(corners[i], *want),
                "corner {i} UV {:?}, expected {want:?}",
                corners[i]
            );
        }
        mesh.validate_topology();
    }

    /// §4: a `2 × 2` grid tiles UV space into four `0.5 × 0.5` cells matching
    /// the spatial layout one-to-one. Cell `(ix, iy)` (row-major face order)
    /// occupies `[ix·du, (ix+1)·du] × [iy·dv, (iy+1)·dv]`; the shared spatial
    /// vertex carries the same UV from each incident face (the unwrap is
    /// continuous), and `(0,0)`/`(1,1)` land on the patch corners.
    #[test]
    fn create_grid_calc_uvs_2x2_tiles() {
        let mut mesh = LoopMesh::new();
        let uv = register_uv(&mut mesh);

        let faces_before = mesh.face_count();
        mesh.create_grid(2, 2, 1.0, Mat4::IDENTITY, true);
        assert_eq!(mesh.face_count() - faces_before, 4);

        let du = 0.5;
        let dv = 0.5;
        // Faces are emitted in row-major cell order (§3): index k → cell
        // (k mod 2, k div 2).
        for (face_index, face) in mesh.face_keys().into_iter().enumerate() {
            let cell_ix = (face_index % 2) as f32;
            let cell_iy = (face_index / 2) as f32;
            let u_lo = cell_ix * du;
            let u_hi = (cell_ix + 1.0) * du;
            let v_lo = cell_iy * dv;
            let v_hi = (cell_iy + 1.0) * dv;
            let expected = [
                [u_lo, v_lo],
                [u_hi, v_lo],
                [u_hi, v_hi],
                [u_lo, v_hi],
            ];
            let corners = read_corner_uvs(&mesh, face, uv);
            for (i, want) in expected.iter().enumerate() {
                assert!(
                    uv_near(corners[i], *want),
                    "face {face_index} corner {i} UV {:?}, expected {want:?}",
                    corners[i]
                );
            }
        }
        mesh.validate_topology();
    }

    /// §4 / §7: `calc_uvs = false` writes nothing even when a UV layer exists —
    /// every painted-on corner keeps the layer's default (zero) value.
    #[test]
    fn create_grid_calc_uvs_false_is_noop() {
        let mut mesh = LoopMesh::new();
        let uv = register_uv(&mut mesh);

        mesh.create_grid(2, 2, 1.0, Mat4::IDENTITY, false);

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
    fn create_grid_calc_uvs_no_layer_is_noop() {
        let mut mesh = LoopMesh::new();

        mesh.create_grid(2, 2, 1.0, Mat4::IDENTITY, true);

        assert_eq!(mesh.face_count(), 4);
        assert!(mesh.loop_attribs().is_empty(), "no UV layer was created");
        mesh.validate_topology();
    }
}
