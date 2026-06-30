use glam::{Mat4, Vec3};

use crate::loopmesh::*;

impl LoopMesh {
    /// Lay down a regular `segments`-gon — a ring of evenly spaced verts on the
    /// circle of radius `radius` in the local `z = 0` plane — and append it to the
    /// mesh (CREATE_CIRCLE.md §2 / §3).
    ///
    /// Ring vertex `a` (for `0 ≤ a < segments`) is placed at angle
    /// `θ = 2π·a/segments` at local `(-radius·sin θ, radius·cos θ, 0)`, so vertex
    /// `0` sits at `(0, +radius, 0)` and the ring is traced counter-clockwise as
    /// seen from `+z` (§2.1). Every vertex — the ring and, when capped, the centre
    /// apex — is mapped through `matrix` as a homogeneous point at creation time
    /// (the translation column is honoured, §2.3).
    ///
    /// Filling is governed by the two cap flags (§3):
    /// * `cap_ends = false`: a bare wire loop — `segments` verts joined by
    ///   `segments` ring edges, no face. `cap_tris` is ignored.
    /// * `cap_ends = true, cap_tris = false`: a single n-gon spanning the ring —
    ///   `segments` verts, `segments` edges, one face, no interior vertex. The
    ///   guide's reference route builds a triangle fan then dissolves it into one
    ///   face; §3 explicitly blesses minting the n-gon directly from the ring,
    ///   which yields the identical topology (one CCW `+z` face, no centre, no
    ///   spokes), so that is the route taken here.
    /// * `cap_ends = true, cap_tris = true`: a triangle fan from a centre apex —
    ///   `segments + 1` verts, `2·segments` edges, `segments` triangles. The
    ///   `segments` spoke edges are minted once and shared by adjacent triangles
    ///   through the edge-reusing face-create primitive.
    ///
    /// Create-circle is purely constructive and **additive**: it reads no existing
    /// geometry, never clears the mesh, and never merges into prior verts even
    /// where the new ring is geometrically coincident with them (§ intro, §7).
    /// `segments = 0` is a **no-op** — nothing is built (§7), unlike create-grid's
    /// clamp-to-1 or create-cube's unit-cube override. With `radius = 0` the full
    /// topology is still built with every ring vertex collapsed onto the
    /// transformed origin, undeduplicated (§7).
    ///
    /// `calc_uvs` is accepted for call-shape compatibility but is presently a
    /// deferral: loopmesh writes no default radial UV unwrap (the §4 UV row is
    /// tracked separately and stays unimplemented this cycle). With no UV layer
    /// populated the topology is identical with the flag set or clear, matching
    /// §7's "`calc_uvs` set but no UV layer present" no-op.
    ///
    /// See `radial-mesh-guides/operators/CREATE_CIRCLE.md` for the full contract.
    pub fn create_circle(
        &mut self,
        cap_ends: bool,
        cap_tris: bool,
        segments: i32,
        radius: f32,
        matrix: Mat4,
        calc_uvs: bool,
    ) {
        let _ = calc_uvs;

        // §7 — `segments = 0` (or any non-positive value) builds nothing.
        if segments <= 0 {
            return;
        }
        let segment_count = segments as usize;

        // §2.2 — the centre apex is minted only for the fan cap. The n-gon path
        // builds the face straight from the ring, so it needs no centre (the
        // guide's dissolve route would create then remove it, §2.2); the uncapped
        // path never has one.
        let want_fan = cap_ends && cap_tris;
        let centre = if want_fan {
            Some(self.create_vertex(matrix.transform_point3(Vec3::ZERO)))
        } else {
            None
        };

        // §2.1 — place the ring counter-clockwise from +z, transforming each
        // local position at creation (§2.3).
        let mut ring: Vec<VertexKey> = Vec::with_capacity(segment_count);
        for a in 0..segment_count {
            let theta = std::f32::consts::TAU * (a as f32) / (segment_count as f32);
            let local = Vec3::new(-radius * theta.sin(), radius * theta.cos(), 0.0);
            ring.push(self.create_vertex(matrix.transform_point3(local)));
        }

        let mut new_faces: Vec<FaceKey> = Vec::new();
        if let Some(apex) = centre {
            // §3 fan cap — one CCW triangle (centre, a-1, a) per ring edge, plus a
            // closing triangle (centre, last, first). The edge-reusing face-create
            // mints each spoke once and shares it between adjacent triangles, and
            // mints the ring edges as the bases.
            new_faces.reserve(segment_count);
            for a in 0..segment_count {
                let prev = ring[a];
                let next = ring[(a + 1) % segment_count];
                new_faces.push(self.face_create_verts(&[apex, prev, next], FaceCreateFlags::NONE));
            }
        } else if cap_ends {
            // §3 n-gon cap — a single CCW face straight from the ring. The
            // face-create mints the `segments` ring edges.
            new_faces.push(self.face_create_verts(&ring, FaceCreateFlags::NONE));
        } else {
            // §3 uncapped — a bare wire loop, no face. Mint the `segments` ring
            // edges explicitly, closing the cycle back to the first vertex.
            for a in 0..segment_count {
                let next = (a + 1) % segment_count;
                self.create_edge([ring[a], ring[next]], EdgeCreateFlags::NO_DUPLICATE);
            }
        }

        // §1 — compute the new faces' normals (a run-flag post-pass), faces first
        // then verts. An uncapped ring has no faces, so its verts carry the
        // zeroed loose-style normal.
        for &face in &new_faces {
            self.face_calc_normal(face);
        }
        if let Some(apex) = centre {
            self.vert_normal_update(apex);
        }
        for &vert in &ring {
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

    /// §3 / §6 / §8.1 uncapped headline: `cap_ends = false` builds a bare
    /// `n`-vertex / `n`-edge closed loop with no face — `ΔV = n, ΔE = n, ΔF = 0`.
    /// The ring verts sit at the radius-`r` cardinal points.
    #[test]
    fn create_circle_uncapped_ring() {
        let mut mesh = LoopMesh::new();
        let n = 8;
        let r = 2.0;

        mesh.create_circle(false, false, n, r, Mat4::IDENTITY, false);

        assert_eq!(
            (mesh.vertex_count(), mesh.edge_count(), mesh.face_count()),
            (n as usize, n as usize, 0)
        );
        assert_eq!(mesh.loop_count(), 0);
        // Vertex 0 at (0, +r, 0); the quarter-turn vertex (a = n/4 = 2) at
        // (-r, 0, 0) by the §2.1 sign convention (CCW from +z).
        assert!(has_vertex_near(&mesh, Vec3::new(0.0, r, 0.0)));
        assert!(has_vertex_near(&mesh, Vec3::new(-r, 0.0, 0.0)));
        assert!(has_vertex_near(&mesh, Vec3::new(0.0, -r, 0.0)));
        assert!(has_vertex_near(&mesh, Vec3::new(r, 0.0, 0.0)));
        // V − E + F = 0 for a bare 1-cycle.
        assert_eq!(
            mesh.vertex_count() as i64 - mesh.edge_count() as i64 + mesh.face_count() as i64,
            0
        );
        mesh.validate_topology();
    }

    /// §3 / §6 / §8.1 n-gon cap: `cap_ends = true, cap_tris = false` builds a
    /// single `n`-gon spanning the ring — `ΔV = n, ΔE = n, ΔF = 1, ΔLoops = n`,
    /// no interior vertex, wound CCW (`+z` normal).
    #[test]
    fn create_circle_ngon_cap() {
        let mut mesh = LoopMesh::new();
        let n = 6;
        let r = 1.5;

        mesh.create_circle(true, false, n, r, Mat4::IDENTITY, false);

        assert_eq!(
            (mesh.vertex_count(), mesh.edge_count(), mesh.face_count()),
            (n as usize, n as usize, 1)
        );
        assert_eq!(mesh.loop_count(), n as usize);
        let face = mesh.face_keys()[0];
        assert_eq!(mesh.face_vert_keys(face).len(), n as usize, "a single n-gon");
        assert!(mesh.face_normal(face).z > 0.9, "cap faces +z");
        // Disc Euler relation V − E + F = 1.
        assert_eq!(
            mesh.vertex_count() as i64 - mesh.edge_count() as i64 + mesh.face_count() as i64,
            1
        );
        mesh.validate_topology();
    }

    /// §3 / §6 / §8.1 fan cap (`cap_tris` contrast): `cap_ends = true,
    /// cap_tris = true` builds `n` triangles around a centre apex — `ΔV = n+1,
    /// ΔE = 2n, ΔF = n, ΔLoops = 3n`. Paired with the n-gon case this pins
    /// `cap_tris` as a genuine topological switch: the two cap forms differ in
    /// every count.
    #[test]
    fn create_circle_fan_cap() {
        let mut mesh = LoopMesh::new();
        let n = 6;
        let r = 1.5;

        mesh.create_circle(true, true, n, r, Mat4::IDENTITY, false);

        let nn = n as usize;
        assert_eq!(
            (mesh.vertex_count(), mesh.edge_count(), mesh.face_count()),
            (nn + 1, 2 * nn, nn)
        );
        assert_eq!(mesh.loop_count(), 3 * nn);
        for face in mesh.face_keys() {
            assert_eq!(mesh.face_vert_keys(face).len(), 3, "every fan face a triangle");
            assert!(mesh.face_normal(face).z > 0.9, "fan triangle faces +z");
        }
        // A surviving centre apex at the transformed origin.
        assert!(has_vertex_near(&mesh, Vec3::ZERO), "centre apex present");
        // Disc Euler relation V − E + F = 1.
        assert_eq!(
            mesh.vertex_count() as i64 - mesh.edge_count() as i64 + mesh.face_count() as i64,
            1
        );
        mesh.validate_topology();

        // Contrast: the n-gon cap of the same ring differs in every count.
        let mut ngon = LoopMesh::new();
        ngon.create_circle(true, false, n, r, Mat4::IDENTITY, false);
        assert_ne!(mesh.vertex_count(), ngon.vertex_count());
        assert_ne!(mesh.edge_count(), ngon.edge_count());
        assert_ne!(mesh.face_count(), ngon.face_count());
    }

    /// §2.1 / §8.1 segments + ring layout: distinct `segments` give distinct ring
    /// counts (a 12-gon ≠ an 8-gon), and the verts land on the radius-`r` circle
    /// at the evenly spaced angles.
    #[test]
    fn create_circle_segments_variation() {
        let mut eight = LoopMesh::new();
        eight.create_circle(false, false, 8, 1.0, Mat4::IDENTITY, false);
        assert_eq!(eight.vertex_count(), 8);
        assert_eq!(eight.edge_count(), 8);

        let mut twelve = LoopMesh::new();
        twelve.create_circle(false, false, 12, 1.0, Mat4::IDENTITY, false);
        assert_eq!(twelve.vertex_count(), 12);
        assert_eq!(twelve.edge_count(), 12);
        assert_ne!(eight.vertex_count(), twelve.vertex_count());

        // Every vertex sits on the radius-1 circle in the z = 0 plane.
        for v in twelve.vert_keys() {
            let p = twelve.vertex_position(v);
            assert!((p.length() - 1.0).abs() < EPS, "ring vert on unit circle");
            assert!(p.z.abs() < EPS, "ring vert in z = 0 plane");
        }
        eight.validate_topology();
        twelve.validate_topology();
    }

    /// §1 / §2.1 / §8.1 radius scaling: the ring sits at distance `radius` from the
    /// local origin, so two distinct radii bucket to distinct positions.
    #[test]
    fn create_circle_radius_scaling() {
        for &r in &[1.0_f32, 3.0_f32] {
            let mut mesh = LoopMesh::new();
            mesh.create_circle(false, false, 4, r, Mat4::IDENTITY, false);
            // Cardinal points of a 4-gon land on the axes at distance r.
            assert!(has_vertex_near(&mesh, Vec3::new(0.0, r, 0.0)));
            assert!(has_vertex_near(&mesh, Vec3::new(-r, 0.0, 0.0)));
            assert!(has_vertex_near(&mesh, Vec3::new(0.0, -r, 0.0)));
            assert!(has_vertex_near(&mesh, Vec3::new(r, 0.0, 0.0)));
            mesh.validate_topology();
        }
        // The two radii are genuinely distinct: radius 3 has no vertex on the
        // radius-1 circle's +y cardinal point.
        let mut three = LoopMesh::new();
        three.create_circle(false, false, 4, 3.0, Mat4::IDENTITY, false);
        assert!(!has_vertex_near(&three, Vec3::new(0.0, 1.0, 0.0)));
    }

    /// §2.3 / §8.1 matrix transform: every vertex (centre + ring) is mapped
    /// through `matrix` as a homogeneous point at creation, so the fan apex and
    /// ring land at their transformed circle positions.
    #[test]
    fn create_circle_matrix_transform() {
        let mut mesh = LoopMesh::new();
        let n = 8;
        let r = 1.0;
        let matrix = Mat4::from_scale_rotation_translation(
            Vec3::ONE,
            Quat::from_rotation_z(std::f32::consts::FRAC_PI_4),
            Vec3::new(5.0, -2.0, 1.0),
        );

        mesh.create_circle(true, true, n, r, matrix, false);

        // The centre apex sits at the transformed origin (the matrix translation).
        assert!(has_vertex_near(&mesh, matrix.transform_point3(Vec3::ZERO)));
        // Vertex 0's local (0, +r, 0) maps through the matrix.
        let local_zero = Vec3::new(0.0, r, 0.0);
        assert!(has_vertex_near(&mesh, matrix.transform_point3(local_zero)));
        // No untransformed vertex survives (identity build would differ).
        assert!(!has_vertex_near(&mesh, local_zero));
        mesh.validate_topology();
    }

    /// intro / §7 / §8.1 append onto a populated mesh: the circle is added as a
    /// disjoint island, prior geometry untouched, additive counts, no merge.
    #[test]
    fn create_circle_appends_onto_populated_mesh() {
        let mut mesh = LoopMesh::new();

        // An unrelated quad island off to the side.
        let corners = [
            Vec3::new(10.0, 10.0, 0.0),
            Vec3::new(11.0, 10.0, 0.0),
            Vec3::new(11.0, 11.0, 0.0),
            Vec3::new(10.0, 11.0, 0.0),
        ];
        let island: Vec<VertexKey> = corners.iter().map(|&p| mesh.create_vertex(p)).collect();
        mesh.face_create_verts(&island, FaceCreateFlags::NONE);
        let before = (mesh.vertex_count(), mesh.edge_count(), mesh.face_count());
        assert_eq!(before, (4, 4, 1));

        // Append an n-gon circle: ΔV = n, ΔE = n, ΔF = 1.
        let n = 6;
        mesh.create_circle(true, false, n, 1.0, Mat4::IDENTITY, false);

        assert_eq!(
            (mesh.vertex_count(), mesh.edge_count(), mesh.face_count()),
            (4 + n as usize, 4 + n as usize, 1 + 1)
        );
        // The prior island's corners are untouched.
        for &p in &corners {
            assert!(has_vertex_near(&mesh, p), "prior island vertex {p:?} moved");
        }
        mesh.validate_topology();
    }

    /// §7: `segments = 0` is a no-op — nothing is built (not a clamp, not an
    /// override). Guards the headline divergence from create-grid/create-cube.
    #[test]
    fn create_circle_zero_segments_is_noop() {
        let mut mesh = LoopMesh::new();
        mesh.create_circle(true, true, 0, 1.0, Mat4::IDENTITY, false);
        assert_eq!(
            (mesh.vertex_count(), mesh.edge_count(), mesh.face_count()),
            (0, 0, 0)
        );
        mesh.validate_topology();
    }
}
