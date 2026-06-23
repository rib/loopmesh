use std::collections::HashMap;

use glam::Vec3;

use crate::loopmesh::*;

/// Squared distance below which a vertex is considered to have not moved, used
/// for the per-iteration early-out (PLANAR_FACES.md §5.1 / §5.3). A vertex whose
/// target is within this band of its current position is left untouched and
/// triggers no re-marking; an iteration in which every vertex is within the band
/// breaks the loop.
const MOVE_EPSILON_SQ: f32 = 1e-10;

impl LoopMesh {
    /// Iteratively flatten `faces` toward their own plane (PLANAR_FACES.md §2 / §4).
    ///
    /// A **position-only** relaxation (`ΔV = ΔE = ΔF = 0`, no Euler operator):
    /// every element identity, incidence, loop cycle, and customdata byte survives
    /// unchanged — only the position layer and the cached normals it feeds change.
    /// The declared `geom.out` slot is never populated, so there is nothing to
    /// return (§1.1).
    ///
    /// Each selected **non-triangle** face contributes a point-and-normal plane:
    /// the point is the face's length-weighted median center, **frozen** at the
    /// input geometry and reused every iteration (§3.2 / §4.3); the orientation is
    /// the face's **live cached normal**, read fresh each pass so the plane pivots
    /// about the fixed center as the face flattens (§4.1 / §4.3). Triangles are
    /// **skipped on side count** (not planarity): they contribute no plane, their
    /// verts are not registered, and they are never marked (§3.1) — a coplanar
    /// quad, by contrast, stays in the working set and simply does not move.
    ///
    /// One pass projects every selected face's verts orthogonally onto that face's
    /// plane, folds each projection into the vertex's running **arithmetic mean**
    /// over its incident selected faces (§4.2), then lerps every vertex from its
    /// **current** position toward that mean by `factor` (§5.1). A moved vertex
    /// re-marks the non-triangle faces incident to it for the next pass (§5.2). The
    /// loop breaks the first pass that moves nothing (§5.3). `factor` is not clamped
    /// to `[0, 1]` — `factor = 0` is a no-op, `factor > 1` overshoots. `iterations
    /// <= 0` and empty input are no-ops (§7).
    ///
    /// See `radial-mesh-guides/operators/PLANAR_FACES.md` for the full algorithm
    /// and parity contract.
    pub fn planar_faces(&mut self, faces: &[FaceKey], iterations: i32, factor: f32) {
        if faces.is_empty() || iterations <= 0 {
            return;
        }

        // §3.1 — build the working set. A face of exactly three sides is skipped
        // on its side count (it is already planar); every face with four or more
        // sides enters the set with its frozen plane center (§3.2). The center is
        // computed once here, from the input geometry, and never recomputed.
        let mut working_faces: Vec<FaceKey> = Vec::new();
        let mut centers: Vec<Vec3> = Vec::new();
        let mut working_index: HashMap<FaceKey, usize> = HashMap::new();
        for &face in faces {
            if self.face_len(face) <= 3 {
                continue;
            }
            if working_index.contains_key(&face) {
                continue;
            }
            working_index.insert(face, working_faces.len());
            centers.push(self.face_plane_center(face));
            working_faces.push(face);
        }
        if working_faces.is_empty() {
            return;
        }

        // The marked set: parallel to `working_faces`, every working face starts
        // marked "to flatten" (§3.1). The flatten phase walks the working faces and
        // acts only on those currently marked; a moved vertex re-marks the incident
        // working faces for the next pass (§5.2). Re-marks on faces outside the
        // working set are inert (the flatten phase never visits them), so we track
        // marks only on working faces and look them up by `working_index`.
        let mut marked: Vec<bool> = vec![true; working_faces.len()];

        let mut targets: HashMap<VertexKey, (Vec3, u32)> = HashMap::new();

        for _ in 0..iterations {
            // §5.1 flatten phase — accumulate every marked working face's
            // projections into the per-vertex running mean, then clear its mark.
            targets.clear();
            for index in 0..working_faces.len() {
                if !marked[index] {
                    continue;
                }
                let face = working_faces[index];
                let normal = self.face_normal(face);
                let center = centers[index];
                let plane_d = -normal.dot(center);
                for loop_key in self.face_loop_keys(face) {
                    let vert = self.loops.get(loop_key).vertex;
                    let position = self.vertex_position(vert);
                    let side = normal.dot(position) + plane_d;
                    let projected = position - normal * side;
                    let entry = targets.entry(vert).or_insert((Vec3::ZERO, 0));
                    entry.1 += 1;
                    entry.0 += (projected - entry.0) / entry.1 as f32;
                }
                marked[index] = false;
            }

            // §5.1 apply phase — lerp every vertex whose mean target differs from
            // its current position by more than epsilon, recording that something
            // moved and re-marking that vertex's incident working faces (§5.2). All
            // projections above were read before any vertex moved, so the pass is
            // order-independent.
            let mut moved_verts: smallvec::SmallVec<[VertexKey; 32]> = smallvec::SmallVec::new();
            for (&vert, &(target, _)) in &targets {
                let position = self.vertex_position(vert);
                if (target - position).length_squared() <= MOVE_EPSILON_SQ {
                    continue;
                }
                self.set_vertex_position(vert, position + (target - position) * factor);
                moved_verts.push(vert);
            }

            // §5.3 early-out — a pass that moved nothing breaks the loop.
            if moved_verts.is_empty() {
                break;
            }

            // §5.2 re-mark every non-triangle working face incident to a moved
            // vertex, and refresh the cached normals so the next pass projects
            // against a normal that tracks the (now flatter) geometry (§4.3).
            for &vert in &moved_verts {
                let mut disk = self.disk_edges(vert);
                while let Some(edge) = disk.next(self) {
                    let mut radial = self.radial_loops(edge);
                    while let Some(loop_key) = radial.next(self) {
                        let face = self.loops.get(loop_key).face;
                        if let Some(&index) = working_index.get(&face) {
                            marked[index] = true;
                        }
                    }
                }
            }
            self.refresh_planar_normals(&moved_verts);
        }
    }

    /// Recompute the cached face normals on every face incident to a moved vertex,
    /// then the vertex normals on those faces' verts (PLANAR_FACES.md §1 / §4.3,
    /// the same scoped refresh `smooth_vert` / `transform` use). Keeps each face's
    /// cached normal current so the next iteration's plane orientation tracks the
    /// flattening geometry, and leaves the post-op normals valid.
    fn refresh_planar_normals(&mut self, verts: &[VertexKey]) {
        let mut affected_faces: smallvec::SmallVec<[FaceKey; 16]> = smallvec::SmallVec::new();
        for &vert in verts {
            let mut disk = self.disk_edges(vert);
            while let Some(edge) = disk.next(self) {
                let mut radial = self.radial_loops(edge);
                while let Some(loop_key) = radial.next(self) {
                    let face = self.loops.get(loop_key).face;
                    if !affected_faces.contains(&face) {
                        affected_faces.push(face);
                    }
                }
            }
        }

        let mut affected_verts: smallvec::SmallVec<[VertexKey; 32]> = smallvec::SmallVec::new();
        for &face in &affected_faces {
            self.face_calc_normal(face);
            for vert in self.face_vert_keys(face) {
                if !affected_verts.contains(&vert) {
                    affected_verts.push(vert);
                }
            }
        }
        for &vert in &affected_verts {
            self.vert_normal_update(vert);
        }
    }

    /// The face's length-weighted median center (PLANAR_FACES.md §3.2): each loop
    /// vertex weighted by the sum of the lengths of its two incident face edges,
    /// the center the weighted mean of the vertex positions. A zero total weight
    /// (an all-coincident face) yields the accumulated point rather than a
    /// divide-by-zero.
    fn face_plane_center(&self, face: FaceKey) -> Vec3 {
        let loops = self.face_loop_keys(face);
        let count = loops.len();
        if count == 0 {
            return Vec3::ZERO;
        }
        let mut accumulated = Vec3::ZERO;
        let mut total_weight = 0.0f32;
        for current in 0..count {
            let previous = if current == 0 { count - 1 } else { current - 1 };
            let next = if current + 1 == count { 0 } else { current + 1 };
            let position = self.vertex_position(self.loops.get(loops[current]).vertex);
            let previous_position = self.vertex_position(self.loops.get(loops[previous]).vertex);
            let next_position = self.vertex_position(self.loops.get(loops[next]).vertex);
            let weight = (position - previous_position).length() + (next_position - position).length();
            accumulated += position * weight;
            total_weight += weight;
        }
        if total_weight != 0.0 {
            accumulated / total_weight
        } else {
            accumulated
        }
    }
}

#[cfg(test)]
mod tests {
    use glam::Vec3;

    use crate::loopmesh::*;

    fn assert_close(a: Vec3, b: Vec3) {
        assert!(
            (a - b).length() < 1e-4,
            "expected {b:?}, got {a:?} (delta {})",
            (a - b).length()
        );
    }

    /// Build a single n-gon face from the given corner positions, wound in order.
    fn ngon(mesh: &mut LoopMesh, positions: &[Vec3]) -> (Vec<VertexKey>, FaceKey) {
        let verts: Vec<VertexKey> = positions.iter().map(|&p| mesh.create_vertex(p)).collect();
        let edges: Vec<EdgeKey> = (0..verts.len())
            .map(|i| {
                mesh.create_edge(
                    [verts[i], verts[(i + 1) % verts.len()]],
                    EdgeCreateFlags::NO_DUPLICATE,
                )
            })
            .collect();
        let face = mesh.face_create(&verts, &edges, FaceCreateFlags::NONE);
        mesh.face_calc_normal(face);
        (verts, face)
    }

    /// A regular pentagon in the `z = 0` plane with its first vertex (the apex)
    /// lifted by `ridge` along `+z`, forming a shallow tent. Returns the verts and
    /// the face; the apex is `verts[0]`.
    fn nonplanar_pentagon(mesh: &mut LoopMesh, ridge: f32) -> (Vec<VertexKey>, FaceKey) {
        let mut positions = Vec::new();
        for i in 0..5 {
            let angle = std::f32::consts::TAU * i as f32 / 5.0 + std::f32::consts::FRAC_PI_2;
            positions.push(Vec3::new(angle.cos(), angle.sin(), 0.0));
        }
        positions[0].z = ridge;
        ngon(mesh, &positions)
    }

    /// §2 / §4: the headline flatten. The lifted apex of a non-planar pentagon
    /// drops onto the face plane after one `factor = 1` pass, and every vertex
    /// ends coplanar (zero signed distance to the plane through the frozen center
    /// with the converged normal).
    #[test]
    fn flattens_nonplanar_pentagon() {
        let mut mesh = LoopMesh::new();
        let (verts, face) = nonplanar_pentagon(&mut mesh, 0.6);
        let apex_before = mesh.vertex_position(verts[0]);

        mesh.planar_faces(&[face], 10, 1.0);

        // The apex moved (it was the only out-of-plane vertex).
        assert!(
            (mesh.vertex_position(verts[0]) - apex_before).length() > 1e-3,
            "apex should have dropped toward the plane"
        );

        // All verts coplanar: fit a plane through the (recomputed) normal and the
        // centroid and check every vertex's signed distance is ~0.
        let normal = mesh.face_normal(face);
        let positions: Vec<Vec3> = verts.iter().map(|&v| mesh.vertex_position(v)).collect();
        let centroid = positions.iter().fold(Vec3::ZERO, |acc, &p| acc + p) / positions.len() as f32;
        for p in &positions {
            assert!(
                normal.dot(*p - centroid).abs() < 1e-3,
                "vertex {p:?} should lie on the flattened plane (dist {})",
                normal.dot(*p - centroid).abs()
            );
        }
        assert_eq!(
            (mesh.vertex_count(), mesh.edge_count(), mesh.face_count()),
            (5, 5, 1)
        );
        mesh.validate_topology();
    }

    /// §7: `factor = 0` is a no-op — every position stays at its input.
    #[test]
    fn factor_zero_is_noop() {
        let mut mesh = LoopMesh::new();
        let (verts, face) = nonplanar_pentagon(&mut mesh, 0.6);
        let before: Vec<Vec3> = verts.iter().map(|&v| mesh.vertex_position(v)).collect();

        mesh.planar_faces(&[face], 8, 0.0);

        for (i, &v) in verts.iter().enumerate() {
            assert_close(mesh.vertex_position(v), before[i]);
        }
        mesh.validate_topology();
    }

    /// §7: `iterations = 0` is a no-op — zero passes run.
    #[test]
    fn iterations_zero_is_noop() {
        let mut mesh = LoopMesh::new();
        let (verts, face) = nonplanar_pentagon(&mut mesh, 0.6);
        let before: Vec<Vec3> = verts.iter().map(|&v| mesh.vertex_position(v)).collect();

        mesh.planar_faces(&[face], 0, 1.0);

        for (i, &v) in verts.iter().enumerate() {
            assert_close(mesh.vertex_position(v), before[i]);
        }
        mesh.validate_topology();
    }

    /// §7: empty input is a no-op.
    #[test]
    fn empty_input_is_noop() {
        let mut mesh = LoopMesh::new();
        let (verts, _face) = nonplanar_pentagon(&mut mesh, 0.6);
        let before: Vec<Vec3> = verts.iter().map(|&v| mesh.vertex_position(v)).collect();

        mesh.planar_faces(&[], 8, 1.0);

        for (i, &v) in verts.iter().enumerate() {
            assert_close(mesh.vertex_position(v), before[i]);
        }
        mesh.validate_topology();
    }

    /// §4.2 / §6: connected-patch convergence. Two quads share an edge; each quad
    /// is an individually non-planar saddle, so the shared-edge verts belong to
    /// two non-planar faces and are pulled toward the mean of their two
    /// projections. Over several iterations with a mid `factor` the patch relaxes
    /// — every quad ends (near) planar.
    #[test]
    fn converges_twisted_quad_pair() {
        let mut mesh = LoopMesh::new();
        // Two quads sharing the edge (b, c). Left quad a-b-c-d, right quad b-e-f-c.
        // Twist each quad into a saddle so it is individually non-planar.
        let twist = 0.4;
        let a = mesh.create_vertex(Vec3::new(-2.0, -1.0, twist));
        let b = mesh.create_vertex(Vec3::new(0.0, -1.0, -twist));
        let c = mesh.create_vertex(Vec3::new(0.0, 1.0, twist));
        let d = mesh.create_vertex(Vec3::new(-2.0, 1.0, -twist));
        let e = mesh.create_vertex(Vec3::new(2.0, -1.0, -twist));
        let f = mesh.create_vertex(Vec3::new(2.0, 1.0, twist));

        let make_quad = |mesh: &mut LoopMesh, vs: [VertexKey; 4]| -> FaceKey {
            let edges: Vec<EdgeKey> = (0..4)
                .map(|i| {
                    mesh.create_edge([vs[i], vs[(i + 1) % 4]], EdgeCreateFlags::NO_DUPLICATE)
                })
                .collect();
            let face = mesh.face_create(&vs, &edges, FaceCreateFlags::NONE);
            mesh.face_calc_normal(face);
            face
        };
        let left = make_quad(&mut mesh, [a, b, c, d]);
        let right = make_quad(&mut mesh, [b, e, f, c]);

        // Planarity measure: max abs signed distance of each face's corners to the
        // plane through its centroid with its current normal.
        let planarity = |mesh: &LoopMesh, face: FaceKey| -> f32 {
            let normal = mesh.face_normal(face);
            let positions: Vec<Vec3> = mesh
                .face_vert_keys(face)
                .iter()
                .map(|&v| mesh.vertex_position(v))
                .collect();
            let centroid = positions.iter().fold(Vec3::ZERO, |acc, &p| acc + p) / positions.len() as f32;
            positions
                .iter()
                .map(|p| normal.dot(*p - centroid).abs())
                .fold(0.0f32, f32::max)
        };

        let before = planarity(&mesh, left).max(planarity(&mesh, right));
        assert!(before > 0.1, "fixture must start non-planar");

        mesh.planar_faces(&[left, right], 40, 0.5);

        let after = planarity(&mesh, left).max(planarity(&mesh, right));
        assert!(
            after < before * 0.1,
            "patch should have relaxed: before {before}, after {after}"
        );
        assert_eq!(
            (mesh.vertex_count(), mesh.edge_count(), mesh.face_count()),
            (6, 7, 2)
        );
        mesh.validate_topology();
    }

    /// §6: a folded patch of individually planar quads is a no-op. Two flat quads
    /// share an edge and fold along it into a roof; each quad already lies on its
    /// own plane, so every projection is the identity and nothing moves.
    #[test]
    fn folded_planar_quads_is_noop() {
        let mut mesh = LoopMesh::new();
        // Left flat quad in a tilted plane, right flat quad in another — they share
        // the crease edge (b, c) on the y-axis. Each quad is planar on its own.
        let a = mesh.create_vertex(Vec3::new(-1.0, -1.0, 1.0));
        let b = mesh.create_vertex(Vec3::new(0.0, -1.0, 0.0));
        let c = mesh.create_vertex(Vec3::new(0.0, 1.0, 0.0));
        let d = mesh.create_vertex(Vec3::new(-1.0, 1.0, 1.0));
        let e = mesh.create_vertex(Vec3::new(1.0, -1.0, 1.0));
        let f = mesh.create_vertex(Vec3::new(1.0, 1.0, 1.0));

        let make_quad = |mesh: &mut LoopMesh, vs: [VertexKey; 4]| -> FaceKey {
            let edges: Vec<EdgeKey> = (0..4)
                .map(|i| {
                    mesh.create_edge([vs[i], vs[(i + 1) % 4]], EdgeCreateFlags::NO_DUPLICATE)
                })
                .collect();
            let face = mesh.face_create(&vs, &edges, FaceCreateFlags::NONE);
            mesh.face_calc_normal(face);
            face
        };
        let left = make_quad(&mut mesh, [a, b, c, d]);
        let right = make_quad(&mut mesh, [b, e, f, c]);

        let verts = [a, b, c, d, e, f];
        let before: Vec<Vec3> = verts.iter().map(|&v| mesh.vertex_position(v)).collect();

        mesh.planar_faces(&[left, right], 20, 0.5);

        for (i, &v) in verts.iter().enumerate() {
            assert_close(mesh.vertex_position(v), before[i]);
        }
        mesh.validate_topology();
    }

    /// §3.1 / §7: triangle skip on side count. A non-planar pentagon and a triangle
    /// are both selected; the triangle's verts that are not shared with the
    /// pentagon must be unchanged (the triangle contributes no plane), while the
    /// pentagon flattens.
    #[test]
    fn skips_triangle_in_input_set() {
        let mut mesh = LoopMesh::new();
        // A non-planar pentagon, plus a disjoint triangle whose verts the pentagon
        // does not share.
        let (pent_verts, pentagon) = nonplanar_pentagon(&mut mesh, 0.6);

        let t0 = mesh.create_vertex(Vec3::new(5.0, 0.0, 0.5));
        let t1 = mesh.create_vertex(Vec3::new(6.0, 0.0, -0.5));
        let t2 = mesh.create_vertex(Vec3::new(5.5, 1.0, 0.3));
        let tri_verts = [t0, t1, t2];
        let tri_edges: Vec<EdgeKey> = (0..3)
            .map(|i| mesh.create_edge([tri_verts[i], tri_verts[(i + 1) % 3]], EdgeCreateFlags::NO_DUPLICATE))
            .collect();
        let triangle = mesh.face_create(&tri_verts, &tri_edges, FaceCreateFlags::NONE);
        mesh.face_calc_normal(triangle);

        let tri_before: Vec<Vec3> = tri_verts.iter().map(|&v| mesh.vertex_position(v)).collect();
        let apex_before = mesh.vertex_position(pent_verts[0]);

        mesh.planar_faces(&[pentagon, triangle], 10, 1.0);

        // Triangle verts untouched — the triangle was skipped on side count.
        for (i, &v) in tri_verts.iter().enumerate() {
            assert_close(mesh.vertex_position(v), tri_before[i]);
        }
        // Pentagon flattened — its apex moved.
        assert!(
            (mesh.vertex_position(pent_verts[0]) - apex_before).length() > 1e-3,
            "pentagon apex should have flattened"
        );
        mesh.validate_topology();
    }

    /// §7: a coplanar quad is in the working set (skip is on side count, not
    /// planarity) but does not move — every projection equals the input.
    #[test]
    fn coplanar_quad_is_noop() {
        let mut mesh = LoopMesh::new();
        let (verts, face) = ngon(
            &mut mesh,
            &[
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(1.0, 1.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
            ],
        );
        let before: Vec<Vec3> = verts.iter().map(|&v| mesh.vertex_position(v)).collect();

        mesh.planar_faces(&[face], 8, 1.0);

        for (i, &v) in verts.iter().enumerate() {
            assert_close(mesh.vertex_position(v), before[i]);
        }
        mesh.validate_topology();
    }
}
