use std::collections::HashSet;

use glam::{Mat4, Vec3};

use crate::loopmesh::*;

impl LoopMesh {
    /// Lay down a **geodesic icosphere** — a fixed base icosahedron (12 vertices,
    /// 20 outward-wound triangles) optionally refined into an all-triangle
    /// geodesic dome — and append it to the mesh
    /// (CREATE_ICOSPHERE.md §2 / §3).
    ///
    /// Stage one builds the base icosahedron (§2.1): two axis apexes at
    /// `(0, 0, ±radius)` plus two staggered antiprism pentagons of circumradius
    /// `radius·2/√5` at `z = ±radius/√5`, the lower pentagon rotated 36° from the
    /// upper so the equatorial band triangulates with no vertical inter-ring
    /// edges. Every one of the 20 triangles is wound so its normal faces outward
    /// (§2.2); every base vertex lands at distance `radius` from the local origin.
    ///
    /// Stage two, taken only when `subdivisions > 1`, hands the base shell's 30
    /// edges to [`LoopMesh::subdivide_edges`] in its triangular grid-fill /
    /// spherical-projection mode (§3): each base edge is cut into
    /// `k = 2^(subdivisions − 1)` equal segments (`cuts = k − 1`), each base
    /// triangle is filled with a `k²` triangular grid whose split-edge vertices are
    /// **shared** between the two incident grids, and every newly minted vertex is
    /// projected radially out onto the sphere of radius `radius`. The 12 base
    /// vertices already lie at radius `radius`, so after projection the whole
    /// refined shell lies on that sphere — a pole-free, seam-free geodesic dome.
    ///
    /// `subdivisions ≤ 1` is **not** a clamp and **not** a no-op (§7): levels `1`,
    /// `0`, and any negative value all reach the same "no subdivision" branch and
    /// build the full 12-vertex / 20-face base icosahedron.
    ///
    /// The 4×4 `matrix` is applied to **every** generated vertex in one final pass
    /// *after* the spherical subdivision (§2.4): the projection must run on the
    /// canonical (un-transformed) sphere, so transforming first would project onto
    /// the wrong surface.
    ///
    /// `calc_uvs` is accepted for signature parity but BMesh's icosphere UV net is
    /// a build-order/loop-index-keyed table (a flat unfolded net with seam
    /// duplication, **not** a position formula, §4) — a documented don't-care for
    /// parity, so no unwrap is painted here.
    ///
    /// Create-icosphere is purely constructive and **additive**: it reads no
    /// existing geometry, never clears the mesh, and never merges into prior verts
    /// (§ intro / §7). The created vertices are **returned**; the slot order is a
    /// documented don't-care (§8.2), emitted base-first then the subdivision verts.
    ///
    /// See `radial-mesh-guides/operators/CREATE_ICOSPHERE.md` for the full contract.
    pub fn create_icosphere(
        &mut self,
        subdivisions: i32,
        radius: f32,
        matrix: Mat4,
        _calc_uvs: bool,
    ) -> Vec<VertexKey> {
        // §2.1 — the base icosahedron at its canonical (pre-`matrix`) positions.
        // Build the trig in f64 and cast only on the final write, the same
        // tie-stable one-time constructor cost create-uvsphere pays.
        let radius = radius as f64;
        let ring_radius = radius * 2.0 / 5.0_f64.sqrt();
        let ring_height = radius / 5.0_f64.sqrt();

        let mut base: Vec<VertexKey> = Vec::with_capacity(12);
        base.push(self.create_vertex(Vec3::new(0.0, 0.0, radius as f32))); // 0: top apex
        base.push(self.create_vertex(Vec3::new(0.0, 0.0, -radius as f32))); // 1: bottom apex

        const PENTAGON_STEP: f64 = std::f64::consts::TAU / 5.0;
        const STAGGER: f64 = std::f64::consts::PI / 5.0; // 36°

        for i in 0..5 {
            // 2..7: upper pentagon
            let theta = PENTAGON_STEP * i as f64;
            base.push(self.create_vertex(Vec3::new(
                (ring_radius * theta.cos()) as f32,
                (ring_radius * theta.sin()) as f32,
                ring_height as f32,
            )));
        }
        for i in 0..5 {
            // 7..12: lower pentagon, staggered 36° from the upper
            let theta = STAGGER + PENTAGON_STEP * i as f64;
            base.push(self.create_vertex(Vec3::new(
                (ring_radius * theta.cos()) as f32,
                (ring_radius * theta.sin()) as f32,
                -ring_height as f32,
            )));
        }

        let top = 0usize;
        let bottom = 1usize;
        let upper = |i: usize| 2 + i % 5;
        let lower = |i: usize| 7 + i % 5;

        // §2.2 — the 20 triangles in three bands. The explicit corner triples are
        // wound outward by `face_create_outward`, which orders each triangle so its
        // normal points away from the origin regardless of the listed order.
        let mut base_faces: Vec<FaceKey> = Vec::with_capacity(20);
        for i in 0..5 {
            // 5 top-cap triangles: apex fanned to consecutive upper verts.
            base_faces.push(self.face_create_outward([
                base[top],
                base[upper(i)],
                base[upper(i + 1)],
            ]));
            // 5 bottom-cap triangles: apex fanned to consecutive lower verts.
            base_faces.push(self.face_create_outward([
                base[bottom],
                base[lower(i)],
                base[lower(i + 1)],
            ]));
            // 10 equatorial triangles: the staggered up/down zig-zag band. Each
            // upper edge caps down to the lower vert below its gap, each lower edge
            // caps up to the upper vert above its gap.
            base_faces.push(self.face_create_outward([
                base[upper(i)],
                base[lower(i)],
                base[upper(i + 1)],
            ]));
            base_faces.push(self.face_create_outward([
                base[upper(i + 1)],
                base[lower(i)],
                base[lower(i + 1)],
            ]));
        }

        // §3 — past level 1, refine the base shell's 30 edges into a geodesic dome.
        let mut subdivision_verts: Vec<VertexKey> = Vec::new();
        let mut subdivided_faces: Vec<FaceKey> = Vec::new();
        if subdivisions > 1 {
            let frequency: u32 = 1 << (subdivisions - 1);
            let cuts = frequency - 1;

            let mut base_edges_set: HashSet<EdgeKey> = HashSet::new();
            for &face in &base_faces {
                for edge in self.face_edge_keys(face) {
                    base_edges_set.insert(edge);
                }
            }
            let base_edges: Vec<EdgeKey> = base_edges_set.into_iter().collect();

            // `use_sphere` overloads `smooth` as the projection radius (§3): every
            // inserted vert lands at `normalize(chord) * radius`, base corners stay.
            let params = SubdivideParams::default()
                .cuts(cuts)
                .use_sphere(true)
                .smooth(radius as f32);
            match self.subdivide_edges(&base_edges, params) {
                SubdivideResult::Ran(geom) => {
                    subdivision_verts = geom.verts;
                    subdivided_faces = geom.faces;
                }
                SubdivideResult::Unsupported => {
                    panic!("spherical grid-fill subdivision is required for the icosphere refinement")
                }
            }
        }

        // §2.4 — transform every generated vertex in one final pass, AFTER the
        // projection ran on the canonical sphere.
        if matrix != Mat4::IDENTITY {
            for &vert in base.iter().chain(subdivision_verts.iter()) {
                let local = self.vertex_position(vert);
                self.set_vertex_position(vert, matrix.transform_point3(local));
            }
        }

        // §1 — recompute the new faces' then verts' normals over the final
        // (transformed) geometry; a run-flag post-pass, not a topological contract.
        let faces_to_update: &[FaceKey] = if subdivided_faces.is_empty() {
            &base_faces
        } else {
            &subdivided_faces
        };
        for &face in faces_to_update {
            self.face_calc_normal(face);
        }
        for &vert in base.iter().chain(subdivision_verts.iter()) {
            self.vert_normal_update(vert);
        }

        // §8.2 — verts.out: slot order is a don't-care, base-first then the
        // subdivision verts in the subdivision operator's traversal order.
        let mut out = base;
        out.extend_from_slice(&subdivision_verts);
        out
    }

    /// Build a triangle from `corners`, ordering them so the face normal points
    /// away from the local origin (CREATE_ICOSPHERE.md §2.2). The base
    /// icosahedron's verts sit on an origin-centred sphere, so the outward
    /// direction at a face is its centroid direction; if the listed order winds
    /// inward the last two corners are swapped.
    fn face_create_outward(&mut self, corners: [VertexKey; 3]) -> FaceKey {
        let pa = self.vertex_position(corners[0]);
        let pb = self.vertex_position(corners[1]);
        let pc = self.vertex_position(corners[2]);
        let normal = (pb - pa).cross(pc - pa);
        let centroid = (pa + pb + pc) / 3.0;
        let ordered = if normal.dot(centroid) < 0.0 {
            [corners[0], corners[2], corners[1]]
        } else {
            corners
        };
        self.face_create_verts(&ordered, FaceCreateFlags::NONE)
    }
}

#[cfg(test)]
mod tests {
    use glam::{Mat4, Vec3};

    use crate::loopmesh::*;

    const EPS: f32 = 1e-4;

    fn euler(mesh: &LoopMesh) -> i64 {
        mesh.vertex_count() as i64 - mesh.edge_count() as i64 + mesh.face_count() as i64
    }

    /// `k = 2^(subdivisions − 1)` (with `subdivisions ≤ 1` → `k = 1`).
    fn freq(subdivisions: i32) -> usize {
        1usize << (subdivisions.max(1) - 1)
    }

    /// Assert the closed-shell geodesic contract for a freshly-built icosphere:
    /// the §6 counts for the relevant `k`, χ = 2, every face a triangle, every
    /// vertex on the sphere of `radius`, every triangle wound outward.
    fn assert_icosphere(mesh: &LoopMesh, subdivisions: i32, radius: f32) {
        let k = freq(subdivisions);
        assert_eq!(
            (mesh.vertex_count(), mesh.edge_count(), mesh.face_count()),
            (10 * k * k + 2, 30 * k * k, 20 * k * k),
            "§6 counts for k = {k}"
        );
        assert_eq!(euler(mesh), 2, "closed sphere shell (χ = 2)");

        for face in mesh.face_keys() {
            let verts = mesh.face_vert_keys(face);
            assert_eq!(verts.len(), 3, "every icosphere face is a triangle");
            let centroid = verts
                .iter()
                .map(|&v| mesh.vertex_position(v))
                .fold(Vec3::ZERO, |a, b| a + b)
                / verts.len() as f32;
            assert!(
                mesh.face_normal(face).dot(centroid) > 0.0,
                "icosphere faces wind outward"
            );
        }
        for vert in mesh.vert_keys() {
            assert!(
                (mesh.vertex_position(vert).length() - radius).abs() < EPS,
                "every vertex at distance radius"
            );
        }
        mesh.validate_topology();
    }

    /// Every vertex of the regular icosahedron / geodesic sphere is five- or
    /// six-valent (the 12 base verts five, the subdivision verts six).
    fn assert_geodesic_valence(mesh: &LoopMesh) {
        let mut fives = 0;
        for vert in mesh.vert_keys() {
            let valence = mesh.vert_face_keys(vert).len();
            assert!(
                valence == 5 || valence == 6,
                "geodesic vertex valence is 5 or 6, got {valence}"
            );
            if valence == 5 {
                fives += 1;
            }
        }
        assert_eq!(fives, 12, "exactly the 12 base verts stay five-valent");
    }

    /// §2 / §6 / §8.1 the bare icosahedron (`subdivisions = 1`): 12 / 30 / 20,
    /// χ = 2, all triangles, every vertex five-valent and at distance radius.
    #[test]
    fn create_icosphere_bare_icosahedron() {
        let mut mesh = LoopMesh::new();
        let r = 1.0;
        mesh.create_icosphere(1, r, Mat4::IDENTITY, false);
        assert_icosphere(&mesh, 1, r);
        assert_geodesic_valence(&mesh);

        // Both apexes present on the axis.
        let on_axis = |z: f32| {
            mesh.vert_keys()
                .into_iter()
                .any(|v| (mesh.vertex_position(v) - Vec3::new(0.0, 0.0, z)).length() < EPS)
        };
        assert!(on_axis(r) && on_axis(-r), "axis apexes at (0,0,±radius)");
    }

    /// §3 / §6 / §8.1 the first geodesic refinement (`subdivisions = 2`, k = 2):
    /// 42 / 120 / 80, and it is a clean topological switch away from the bare base.
    #[test]
    fn create_icosphere_subdivided_level2() {
        let mut mesh = LoopMesh::new();
        let r = 1.0;
        mesh.create_icosphere(2, r, Mat4::IDENTITY, false);
        assert_icosphere(&mesh, 2, r);
        assert_geodesic_valence(&mesh);

        let mut bare = LoopMesh::new();
        bare.create_icosphere(1, r, Mat4::IDENTITY, false);
        assert_ne!(
            mesh.face_count(),
            bare.face_count(),
            "subdivision is a topological switch: 80 faces vs 20"
        );
    }

    /// §3 / §6 deeper refinement (`subdivisions = 3`, k = 4): 162 / 480 / 320 —
    /// pins the frequency-doubling beyond the first level.
    #[test]
    fn create_icosphere_subdivided_level3() {
        let mut mesh = LoopMesh::new();
        let r = 1.0;
        mesh.create_icosphere(3, r, Mat4::IDENTITY, false);
        assert_icosphere(&mesh, 3, r);
        assert_geodesic_valence(&mesh);
    }

    /// §7 / §8.1 `subdivisions ≤ 1` is the bare icosahedron, not a clamp and not a
    /// no-op: `0 ≡ 1` and `−3 ≡ 1`, both the full non-empty 12 / 30 / 20 shell.
    #[test]
    fn create_icosphere_subdivisions_le_one_is_bare_icosahedron() {
        let counts = |subdiv: i32| {
            let mut mesh = LoopMesh::new();
            mesh.create_icosphere(subdiv, 1.0, Mat4::IDENTITY, false);
            (mesh.vertex_count(), mesh.edge_count(), mesh.face_count())
        };
        let one = counts(1);
        assert_eq!(one, (12, 30, 20), "subdivisions = 1 → bare icosahedron");
        assert_eq!(counts(0), one, "subdivisions = 0 ≡ 1 (no subdivision)");
        assert_eq!(counts(-3), one, "negative subdivisions ≡ 1 (no subdivision)");
        assert!(one.0 > 0, "the ≤ 1 guard is not a build-nothing no-op");
    }

    /// §2.1 / §3 / §8.1 radius scaling: every vertex lands at distance `radius`,
    /// so distinct radii place the spheres at distinct positions.
    #[test]
    fn create_icosphere_radius_scaling() {
        for &r in &[1.0_f32, 2.0_f32] {
            let mut mesh = LoopMesh::new();
            mesh.create_icosphere(3, r, Mat4::IDENTITY, false);
            assert_icosphere(&mesh, 3, r);
        }

        // A radius-2 sphere puts no vertex within the radius-1 shell.
        let mut big = LoopMesh::new();
        big.create_icosphere(3, 2.0, Mat4::IDENTITY, false);
        assert!(
            big.vert_keys()
                .into_iter()
                .all(|v| big.vertex_position(v).length() > 1.5),
            "radius = 2 verts are all beyond the radius = 1 shell"
        );
    }

    /// §2.4 / §8.1 matrix transform: a non-identity matrix moves every generated
    /// vertex off the local sphere, applied after the projection. The local apex
    /// no longer survives; its transformed image does.
    #[test]
    fn create_icosphere_matrix_transform() {
        let r = 1.0;
        let matrix = Mat4::from_translation(Vec3::new(3.0, -2.0, 1.0))
            * Mat4::from_rotation_x(std::f32::consts::FRAC_PI_4);

        let mut mesh = LoopMesh::new();
        mesh.create_icosphere(2, r, matrix, false);

        let has_near = |target: Vec3| {
            mesh.vert_keys()
                .into_iter()
                .any(|v| (mesh.vertex_position(v) - target).length() < EPS)
        };
        assert!(
            has_near(matrix.transform_point3(Vec3::new(0.0, 0.0, r))),
            "transformed top apex present"
        );
        assert!(
            !has_near(Vec3::new(0.0, 0.0, r)),
            "untransformed apex does not survive"
        );

        // The transformed sphere still has the right topology and is centred on
        // the transformed origin (every vert at distance radius from it).
        let centre = matrix.transform_point3(Vec3::ZERO);
        for vert in mesh.vert_keys() {
            assert!(
                (mesh.vertex_position(vert) - centre).length() - r < EPS,
                "projection ran on the canonical sphere before transform"
            );
        }
        assert_eq!(
            (mesh.vertex_count(), mesh.edge_count(), mesh.face_count()),
            (42, 120, 80)
        );
        mesh.validate_topology();
    }

    /// intro / §3 / §7 / §8.1 append onto a populated mesh: the sphere is a
    /// disjoint island, prior geometry untouched, additive counts, no merge.
    #[test]
    fn create_icosphere_appends_onto_populated_mesh() {
        let mut mesh = LoopMesh::new();
        let corners = [
            Vec3::new(10.0, 10.0, 0.0),
            Vec3::new(11.0, 10.0, 0.0),
            Vec3::new(11.0, 11.0, 0.0),
            Vec3::new(10.0, 11.0, 0.0),
        ];
        let island: Vec<VertexKey> = corners.iter().map(|&p| mesh.create_vertex(p)).collect();
        mesh.face_create_verts(&island, FaceCreateFlags::NONE);
        assert_eq!(
            (mesh.vertex_count(), mesh.edge_count(), mesh.face_count()),
            (4, 4, 1)
        );

        mesh.create_icosphere(2, 1.0, Mat4::IDENTITY, false);

        // Quad counts plus the §6 k = 2 icosphere deltas (42 / 120 / 80).
        assert_eq!(
            (mesh.vertex_count(), mesh.edge_count(), mesh.face_count()),
            (4 + 42, 4 + 120, 1 + 80)
        );
        for &p in &corners {
            assert!(
                mesh.vert_keys()
                    .into_iter()
                    .any(|v| (mesh.vertex_position(v) - p).length() < EPS),
                "prior island vertex {p:?} untouched"
            );
        }
        mesh.validate_topology();
    }
}
