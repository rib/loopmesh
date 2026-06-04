use glam::Vec3;

use crate::loopmesh::*;

/// Parameters for [`LoopMesh::smooth_vert`] (SMOOTH_VERT.md §1).
///
/// Smooth-vert is a position-only relaxation: each named vert moves toward the
/// unweighted average of its connected neighbours, blended from its original
/// position by `factor`, then written back only on the enabled axes. It changes
/// no topology and no customdata (§2 / §8) — only the position layer and the
/// cached normals that position feeds.
#[derive(Clone, Copy, Debug)]
pub struct SmoothVertParams {
    /// The blend weight between a vert's **original** position and its neighbour
    /// average (§3.3). `0` leaves the vert in place; `1` snaps it fully onto the
    /// average; intermediate values lerp. Not clamped to `[0, 1]` — `factor > 1`
    /// over-relaxes past the average and `factor < 0` moves away from it (§7).
    pub factor: f32,
    /// Per-axis mirror-clip (§6). When set, a vert whose **original** coordinate
    /// on that axis is within `clip_dist` of the `axis = 0` plane has its target
    /// coordinate on that axis snapped to exactly `0`.
    pub mirror_clip_x: bool,
    pub mirror_clip_y: bool,
    pub mirror_clip_z: bool,
    /// The tolerance band for the three `mirror_clip_*` flags: a vert is "on the
    /// mirror plane" for axis *a* when `|original_co[a]| <= clip_dist` (§6).
    pub clip_dist: f32,
    /// Per-axis write-back enable (§5). The full 3-D target is always computed;
    /// these flags gate only which coordinates of the result reach the live
    /// position. A disabled axis keeps the vert's original coordinate on that
    /// axis. With all three clear the operator is a position no-op.
    pub use_axis_x: bool,
    pub use_axis_y: bool,
    pub use_axis_z: bool,
}

impl LoopMesh {
    /// Smooth `verts` toward their neighbour averages by one Laplacian pass
    /// (SMOOTH_VERT.md §3 / §4).
    ///
    /// For each named vert the target is the unweighted mean of the **other
    /// endpoint of every incident edge** (§3.1) — counted per incident edge, so a
    /// double edge double-weights its neighbour (§3.1.1), and wire edges count
    /// equally to face edges (§3.1). The target is then a lerp from the vert's
    /// original position toward that average by `factor` (§3.3), with a
    /// mirror-clip snap to the `axis = 0` plane for any axis whose flag is set and
    /// whose **original** coordinate started within `clip_dist` of the plane (§6).
    ///
    /// The pass is **double-buffered** (§4): every target is computed against the
    /// pre-pass snapshot of positions before any write lands, so the result is
    /// order-independent — a smoothed vert's neighbour average always reads its
    /// neighbours' pre-pass positions even when those neighbours are themselves in
    /// the smoothed set. The per-axis `use_axis_*` flags gate the write-back (§5):
    /// a target component reaches the live position only on an enabled axis;
    /// disabled axes keep the original coordinate.
    ///
    /// A vert with **zero incident edges** has no neighbour average and stays put
    /// (§3.2) — the branch keys on edge count, so a wire-only vert (incident edges
    /// but no faces) is smoothed normally. After the apply pass the cached normals
    /// on the affected geometry are recomputed from the new positions (§1). Empty
    /// `verts` is a no-op (§7).
    ///
    /// See `radial-mesh-guides/operators/SMOOTH_VERT.md` for the full algorithm
    /// and parity contract.
    pub fn smooth_vert(&mut self, verts: &[VertexKey], params: SmoothVertParams) {
        if verts.is_empty() {
            return;
        }

        // §4 pass 1 — compute every target against the input snapshot. Nothing is
        // written to the live mesh yet, so each neighbour read sees its pre-pass
        // position. The scratch buffer is indexed in lockstep with `verts`.
        let mut targets: smallvec::SmallVec<[Vec3; 32]> = smallvec::SmallVec::with_capacity(verts.len());
        for &vert in verts {
            let original = self.vertex_position(vert);

            // §3.1 — average the other endpoint of every incident edge. Walk the
            // full disk: per incident edge (not per distinct neighbour, §3.1.1),
            // including wire edges. `edge_count == 0` is the isolated-vert branch.
            let mut sum = Vec3::ZERO;
            let mut edge_count = 0u32;
            let mut disk = self.disk_edges(vert);
            while let Some(edge) = disk.next(self) {
                if let Some(other) = self.edges.get(edge).other_vertex(vert) {
                    sum += self.vertex_position(other);
                    edge_count += 1;
                }
            }

            let mut target = if edge_count == 0 {
                // §3.2 — no incident edges: stay put.
                original
            } else {
                let average = sum / edge_count as f32;
                // §3.3 — lerp from the original toward the average by `factor`.
                original + (average - original) * params.factor
            };

            // §6 — mirror-clip, keyed on the ORIGINAL coordinate, acting on the
            // target coordinate. Computed into the full target regardless of which
            // axes are enabled; the axis gate in pass 2 decides if it lands.
            if params.mirror_clip_x && original.x.abs() <= params.clip_dist {
                target.x = 0.0;
            }
            if params.mirror_clip_y && original.y.abs() <= params.clip_dist {
                target.y = 0.0;
            }
            if params.mirror_clip_z && original.z.abs() <= params.clip_dist {
                target.z = 0.0;
            }

            targets.push(target);
        }

        // §4 pass 2 — apply, honouring the per-axis enable flags (§5). A target
        // component reaches the live position only on an enabled axis.
        for (&vert, &target) in verts.iter().zip(&targets) {
            let mut position = self.vertex_position(vert);
            if params.use_axis_x {
                position.x = target.x;
            }
            if params.use_axis_y {
                position.y = target.y;
            }
            if params.use_axis_z {
                position.z = target.z;
            }
            self.set_vertex_position(vert, position);
        }

        // §1 — refresh the cached normals on the affected geometry. No topology
        // and no customdata is touched.
        self.refresh_normals_around_verts(verts);
    }

    /// Recompute cached face normals on every face incident to a moved vert, and
    /// vertex normals on every vert of those faces (SMOOTH_VERT.md §1, the same
    /// scoped refresh `transform` uses). Scoped to the affected geometry so the
    /// cost tracks the selection.
    fn refresh_normals_around_verts(&mut self, verts: &[VertexKey]) {
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
}

#[cfg(test)]
mod tests {
    use glam::Vec3;

    use crate::loopmesh::*;

    fn all_axes(factor: f32) -> SmoothVertParams {
        SmoothVertParams {
            factor,
            mirror_clip_x: false,
            mirror_clip_y: false,
            mirror_clip_z: false,
            clip_dist: 0.0,
            use_axis_x: true,
            use_axis_y: true,
            use_axis_z: true,
        }
    }

    fn assert_close(a: Vec3, b: Vec3) {
        assert!(
            (a - b).length() < 1e-5,
            "expected {b:?}, got {a:?} (delta {})",
            (a - b).length()
        );
    }

    /// A plus-shaped star: a centre vert at `centre` with four arm verts at the
    /// given offsets, one edge from the centre to each arm. Returns the centre
    /// and the four arms. No faces — pure edge connectivity, enough to pin the
    /// neighbour-average core.
    fn star(mesh: &mut LoopMesh, centre: Vec3, arms: [Vec3; 4]) -> (VertexKey, [VertexKey; 4]) {
        let c = mesh.create_vertex(centre);
        let a: [VertexKey; 4] = arms.map(|p| mesh.create_vertex(p));
        for &arm in &a {
            mesh.create_edge([c, arm], EdgeCreateFlags::NO_DUPLICATE);
        }
        (c, a)
    }

    /// §3 / §3.1: the headline relaxation. The centre vert sits above the mean of
    /// its four neighbours; a factor = 1 pass drops it exactly onto that mean.
    #[test]
    fn relaxes_centre_onto_neighbour_mean() {
        let mut mesh = LoopMesh::new();
        let arms = [
            Vec3::new(-1.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, -1.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        ];
        // Centre lifted to z = 2 — the four arms average to the origin.
        let (centre, _) = star(&mut mesh, Vec3::new(0.0, 0.0, 2.0), arms);

        mesh.smooth_vert(&[centre], all_axes(1.0));

        assert_close(mesh.vertex_position(centre), Vec3::ZERO);
        // A factor = 0.5 from z = 2 would land halfway; verify the lerp too on a
        // fresh mesh.
        let mut mesh2 = LoopMesh::new();
        let (centre2, _) = star(&mut mesh2, Vec3::new(0.0, 0.0, 2.0), arms);
        mesh2.smooth_vert(&[centre2], all_axes(0.5));
        assert_close(mesh2.vertex_position(centre2), Vec3::new(0.0, 0.0, 1.0));
        mesh.validate_topology();
    }

    /// §7: factor = 0 is a no-op — every position stays at its original.
    #[test]
    fn factor_zero_is_noop() {
        let mut mesh = LoopMesh::new();
        let arms = [
            Vec3::new(-1.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, -1.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        ];
        let (centre, _) = star(&mut mesh, Vec3::new(0.0, 0.0, 2.0), arms);

        mesh.smooth_vert(&[centre], all_axes(0.0));

        assert_close(mesh.vertex_position(centre), Vec3::new(0.0, 0.0, 2.0));
        mesh.validate_topology();
    }

    /// §5: per-axis gating. With only `use_axis_z` enabled the centre slides on
    /// `z` toward the neighbour mean's `z` while `x`/`y` stay frozen, even though
    /// the full 3-D target would have moved every axis.
    #[test]
    fn per_axis_gate_writes_only_enabled_axis() {
        let mut mesh = LoopMesh::new();
        // Asymmetric arms so the neighbour mean is off-origin on every axis.
        let arms = [
            Vec3::new(-1.0, 0.0, 0.0),
            Vec3::new(3.0, 0.0, 0.0),
            Vec3::new(0.0, -1.0, 0.0),
            Vec3::new(0.0, 5.0, 0.0),
        ];
        // mean of arms = (0.5, 1.0, 0.0). Centre starts at (10, 10, 4).
        let start = Vec3::new(10.0, 10.0, 4.0);
        let (centre, _) = star(&mut mesh, start, arms);

        mesh.smooth_vert(
            &[centre],
            SmoothVertParams {
                factor: 1.0,
                mirror_clip_x: false,
                mirror_clip_y: false,
                mirror_clip_z: false,
                clip_dist: 0.0,
                use_axis_x: false,
                use_axis_y: false,
                use_axis_z: true,
            },
        );

        // z written to the mean's z (0.0); x and y frozen at the original.
        assert_close(mesh.vertex_position(centre), Vec3::new(10.0, 10.0, 0.0));
        mesh.validate_topology();
    }

    /// §6: mirror-clip fires for a vert that started within `clip_dist` of the
    /// `x = 0` plane — its target `x` snaps to exactly 0 — and the same input with
    /// the flag clear keeps the free blend instead.
    #[test]
    fn mirror_clip_x_fires_and_clears() {
        // Centre near the x = 0 plane (x = 0.05); neighbours pull it toward a
        // non-zero x mean so the free blend would NOT land on the plane.
        let arms = [
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::new(2.0, -1.0, 0.0),
            Vec3::new(2.0, 1.0, 0.0),
        ];
        let start = Vec3::new(0.05, 0.0, 0.0);

        // Clip on: target x forced to 0.
        let mut clipped = LoopMesh::new();
        let (c1, _) = star(&mut clipped, start, arms);
        clipped.smooth_vert(
            &[c1],
            SmoothVertParams {
                factor: 1.0,
                mirror_clip_x: true,
                mirror_clip_y: false,
                mirror_clip_z: false,
                clip_dist: 0.2,
                use_axis_x: true,
                use_axis_y: true,
                use_axis_z: true,
            },
        );
        assert!(
            clipped.vertex_position(c1).x.abs() < 1e-6,
            "mirror_clip_x must snap target x to 0, got {}",
            clipped.vertex_position(c1).x
        );

        // Clip off: target x takes the free neighbour-average blend (mean x = 2).
        let mut free = LoopMesh::new();
        let (c2, _) = star(&mut free, start, arms);
        free.smooth_vert(
            &[c2],
            SmoothVertParams {
                factor: 1.0,
                mirror_clip_x: false,
                mirror_clip_y: false,
                mirror_clip_z: false,
                clip_dist: 0.2,
                use_axis_x: true,
                use_axis_y: true,
                use_axis_z: true,
            },
        );
        assert_close(free.vertex_position(c2), Vec3::new(2.0, 0.0, 0.0));
    }

    /// §6: a vert that STARTED outside the `clip_dist` band is never clipped, even
    /// if its smoothed target would land inside it.
    #[test]
    fn mirror_clip_keys_on_original_not_target() {
        // Centre starts far from the plane (x = 5) but its neighbours average to
        // x = 0.1 — inside the band. Keyed on the original (5 > clip_dist), so no
        // clip: the target keeps the free blend (x = 0.1).
        let arms = [
            Vec3::new(0.1, 0.0, 0.0),
            Vec3::new(0.1, 0.0, 0.0),
            Vec3::new(0.1, -1.0, 0.0),
            Vec3::new(0.1, 1.0, 0.0),
        ];
        let mut mesh = LoopMesh::new();
        let (centre, _) = star(&mut mesh, Vec3::new(5.0, 0.0, 0.0), arms);

        mesh.smooth_vert(
            &[centre],
            SmoothVertParams {
                factor: 1.0,
                mirror_clip_x: true,
                mirror_clip_y: false,
                mirror_clip_z: false,
                clip_dist: 0.2,
                use_axis_x: true,
                use_axis_y: true,
                use_axis_z: true,
            },
        );

        assert_close(mesh.vertex_position(centre), Vec3::new(0.1, 0.0, 0.0));
    }

    /// §3.2 / §7: a vert with zero incident edges stays put even at factor = 1.
    #[test]
    fn no_neighbour_vert_stays_put() {
        let mut mesh = LoopMesh::new();
        let loose = mesh.create_vertex(Vec3::new(3.0, -4.0, 5.0));

        mesh.smooth_vert(&[loose], all_axes(1.0));

        assert_close(mesh.vertex_position(loose), Vec3::new(3.0, -4.0, 5.0));
    }

    /// §3.1.1: a double edge double-weights its neighbour. Centre joined to `a` by
    /// two edges and to `b` by one: the mean is `(2a + b) / 3`, not `(a + b) / 2`.
    #[test]
    fn double_edge_double_counts_neighbour() {
        let mut mesh = LoopMesh::new();
        let centre = mesh.create_vertex(Vec3::new(0.0, 0.0, 0.0));
        let a = mesh.create_vertex(Vec3::new(6.0, 0.0, 0.0));
        let b = mesh.create_vertex(Vec3::new(0.0, 6.0, 0.0));
        // Two distinct edges centre–a (a double edge), one edge centre–b.
        mesh.create_edge([centre, a], EdgeCreateFlags::NONE);
        mesh.create_edge([centre, a], EdgeCreateFlags::NONE);
        mesh.create_edge([centre, b], EdgeCreateFlags::NONE);

        mesh.smooth_vert(&[centre], all_axes(1.0));

        // (2*a + b) / 3 = ((12,6,0)) / 3 = (4, 2, 0).
        assert_close(mesh.vertex_position(centre), Vec3::new(4.0, 2.0, 0.0));
    }

    /// §4: the pass is double-buffered. Two adjacent verts both in the smoothed
    /// set read each other's PRE-pass positions, so the result is order-
    /// independent. A naive in-place update would feed the moved neighbour into
    /// the second vert's average.
    #[test]
    fn double_buffered_is_order_independent() {
        // A path p0 - p1 - p2 - p3, smoothing the two interior verts p1, p2.
        let mut mesh = LoopMesh::new();
        let p0 = mesh.create_vertex(Vec3::new(0.0, 0.0, 0.0));
        let p1 = mesh.create_vertex(Vec3::new(1.0, 0.0, 0.0));
        let p2 = mesh.create_vertex(Vec3::new(2.0, 4.0, 0.0));
        let p3 = mesh.create_vertex(Vec3::new(3.0, 0.0, 0.0));
        mesh.create_edge([p0, p1], EdgeCreateFlags::NO_DUPLICATE);
        mesh.create_edge([p1, p2], EdgeCreateFlags::NO_DUPLICATE);
        mesh.create_edge([p2, p3], EdgeCreateFlags::NO_DUPLICATE);

        // p1 mean = (p0 + p2)/2 = ((0,0)+(2,4))/2 = (1, 2).
        // p2 mean = (p1 + p3)/2 = ((1,0)+(3,0))/2 = (2, 0).
        // Both computed from the SNAPSHOT, so swapping which we list first must
        // not change the outcome.
        mesh.smooth_vert(&[p1, p2], all_axes(1.0));
        assert_close(mesh.vertex_position(p1), Vec3::new(1.0, 2.0, 0.0));
        assert_close(mesh.vertex_position(p2), Vec3::new(2.0, 0.0, 0.0));

        let mut mesh2 = LoopMesh::new();
        let q0 = mesh2.create_vertex(Vec3::new(0.0, 0.0, 0.0));
        let q1 = mesh2.create_vertex(Vec3::new(1.0, 0.0, 0.0));
        let q2 = mesh2.create_vertex(Vec3::new(2.0, 4.0, 0.0));
        let q3 = mesh2.create_vertex(Vec3::new(3.0, 0.0, 0.0));
        mesh2.create_edge([q0, q1], EdgeCreateFlags::NO_DUPLICATE);
        mesh2.create_edge([q1, q2], EdgeCreateFlags::NO_DUPLICATE);
        mesh2.create_edge([q2, q3], EdgeCreateFlags::NO_DUPLICATE);
        // Reverse order.
        mesh2.smooth_vert(&[q2, q1], all_axes(1.0));
        assert_close(mesh2.vertex_position(q1), Vec3::new(1.0, 2.0, 0.0));
        assert_close(mesh2.vertex_position(q2), Vec3::new(2.0, 0.0, 0.0));
    }

    /// §7: empty input is a no-op.
    #[test]
    fn empty_input_is_noop() {
        let mut mesh = LoopMesh::new();
        let v = mesh.create_vertex(Vec3::new(1.0, 2.0, 3.0));
        mesh.smooth_vert(&[], all_axes(1.0));
        assert_close(mesh.vertex_position(v), Vec3::new(1.0, 2.0, 3.0));
    }
}
