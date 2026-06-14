use glam::Vec3;

use crate::loopmesh::*;

/// Selects the formula that computes the poke centre vertex's base position
/// from the input face's corner positions.
///
/// See [`POKE.md`](../../../../radial-mesh-guides/operators/POKE.md) §3.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
pub enum CenterMode {
    /// Uniform-weight centroid: the arithmetic mean of the corner positions
    /// (§3.1).
    Mean,
    /// Edge-length-weighted centroid: each corner is weighted by the sum of
    /// the lengths of its two incident face edges (§3.2). This is the
    /// default, matching BMesh's `poke`.
    #[default]
    MeanWeighted,
    /// Axis-aligned bounding-box centre: the midpoint of the per-axis min and
    /// max of the corner positions (§3.3).
    Bounds,
}

/// The two output buffers of a multi-face [`poke`](LoopMesh::poke)
/// ([`POKE.md`](../../../../radial-mesh-guides/operators/POKE.md) §7).
///
/// `verts` and `faces` are independent: `verts` has one entry per input face,
/// `faces` has `Σ corner_count(f)` entries. Neither buffer contains the killed
/// input faces nor any surviving perimeter vert/edge — only the freshly-created
/// centre verts and triangles (§1).
#[derive(Default)]
pub struct PokeResult {
    /// The new centre vertex of every poked face, one per input face, in input
    /// (`faces`-slice) order (§7).
    pub verts: Vec<VertexKey>,
    /// The new triangle faces. An input face of `n` corners contributes its `n`
    /// triangles in loop-cycle order; the per-face groups are concatenated in
    /// input order (§5.2 / §7).
    pub faces: Vec<FaceKey>,
}

impl LoopMesh {
    /// Multi-face poke: fan-triangulate each input face independently,
    /// inserting a fresh centre vertex per face, and report both output buffers
    /// ([`POKE.md`](../../../../radial-mesh-guides/operators/POKE.md) §7).
    ///
    /// Each face is processed in isolation through the same per-face surgery as
    /// the single-face [`face_poke_offset`](Self::face_poke_offset); the input
    /// set has no notion of a shared region. Two adjacent input faces sharing a
    /// perimeter edge produce non-interacting output — the shared edge keeps its
    /// radial-2 configuration, with one triangle from each fan occupying a slot
    /// (§5.3).
    ///
    /// Returns a [`PokeResult`] whose `verts` lists each new centre vertex in
    /// input order, and whose `faces` concatenates each face's `n` new triangles
    /// (in that face's loop-cycle order) in input order.
    pub fn poke(
        &mut self,
        faces: &[FaceKey],
        center_mode: CenterMode,
        offset: f32,
        use_relative_offset: bool,
    ) -> PokeResult {
        let mut result = PokeResult::default();
        for &face in faces {
            let v_center = self.face_poke_into(
                face,
                center_mode,
                offset,
                use_relative_offset,
                Some(&mut result.faces),
            );
            result.verts.push(v_center);
        }
        result
    }

    /// Inserts a center vertex at the median position of `face` and
    /// fan-triangulates: each existing perimeter edge becomes the base of
    /// a triangle that includes the new center vertex. The original face
    /// is killed; `face.len()` new triangle faces are created. Returns
    /// the new center vertex.
    ///
    /// Equivalent to [`face_poke_mode`](Self::face_poke_mode) with
    /// [`CenterMode::Mean`] (the uniform centroid). Existing callers that
    /// want the uniform centre keep this entry point; new callers select a
    /// centre formula explicitly through `face_poke_mode`.
    pub fn face_poke(&mut self, face: FaceKey) -> VertexKey {
        self.face_poke_mode(face, CenterMode::Mean)
    }

    /// Like [`face_poke`](Self::face_poke), but the centre vertex's base
    /// position is computed by the chosen [`CenterMode`] formula
    /// ([`POKE.md`](../../../../radial-mesh-guides/operators/POKE.md) §3).
    /// Only the centre position differs across modes; the fan-triangulation
    /// surgery and customdata interpolation are identical.
    pub fn face_poke_mode(&mut self, face: FaceKey, center_mode: CenterMode) -> VertexKey {
        self.face_poke_offset(face, center_mode, 0.0, false)
    }

    /// Like [`face_poke_mode`](Self::face_poke_mode), but additionally lifts the
    /// centre vertex along the input face's normal by `offset`
    /// ([`POKE.md`](../../../../radial-mesh-guides/operators/POKE.md) §4).
    ///
    /// The lift direction is the face's outward unit normal (§4.1), recomputed
    /// for the input face just before the surgery so the displacement follows
    /// the face's current geometry. A degenerate (zero-area) face has no
    /// well-defined normal; when the recomputed normal is non-finite or
    /// effectively zero-length the lift collapses to nothing and the centre
    /// stays at the flat base, regardless of `offset` (§8 degenerate case).
    ///
    /// `offset` is a signed world-space distance by default; positive lifts
    /// along the normal, negative against it. When `use_relative_offset` is set
    /// the lift is scaled by the mean distance from the corners to the base
    /// centre (§4.2), turning `offset` into a multiplier of the face's
    /// characteristic radius. The scale is measured at the *base* centre, before
    /// the lift, so it is independent of the final lifted position.
    ///
    /// The lift is applied *after* the centre's customdata interpolation (§4
    /// step ordering): the per-vertex and apex-loop blends are computed at the
    /// flat base centre, then the position alone is displaced.
    pub fn face_poke_offset(
        &mut self,
        face: FaceKey,
        center_mode: CenterMode,
        offset: f32,
        use_relative_offset: bool,
    ) -> VertexKey {
        self.face_poke_into(face, center_mode, offset, use_relative_offset, None)
    }

    /// Per-face poke surgery shared by the single-face and multi-face entry
    /// points. When `out_faces` is `Some`, the spawned triangles are appended to
    /// it in loop-cycle order (the `faces.out` contract of §7); the returned
    /// centre vertex is always the `verts.out` entry for this face.
    fn face_poke_into(
        &mut self,
        face: FaceKey,
        center_mode: CenterMode,
        offset: f32,
        use_relative_offset: bool,
        mut out_faces: Option<&mut Vec<FaceKey>>,
    ) -> VertexKey {
        // Snapshot the face's perimeter (verts + their outgoing edges + the
        // original loop at each corner) in a single forward walk. SmallVec
        // keeps small fans off the heap. The original-loop snapshot serves
        // the customdata copy further below.
        let mut verts: smallvec::SmallVec<[VertexKey; 8]> = smallvec::SmallVec::new();
        let mut perim_edges: smallvec::SmallVec<[EdgeKey; 8]> = smallvec::SmallVec::new();
        let mut orig_loops: smallvec::SmallVec<[LoopKey; 8]> = smallvec::SmallVec::new();
        let mut it = self.face_loops(face);
        while let Some(l) = it.next(self) {
            let lref = self.loops.get(l);
            verts.push(lref.vertex);
            orig_loops.push(l);
            debug_assert!(
                !lref.edge.is_null(),
                "face_poke: loop on perimeter has no edge"
            );
            perim_edges.push(lref.edge);
        }
        let n = verts.len();
        debug_assert!(n >= 3, "face_poke needs at least 3 vertices");
        debug_assert_eq!(perim_edges.len(), n);

        // Corner positions, in loop-cycle order, gathered once and reused by
        // every centre formula.
        let positions: smallvec::SmallVec<[Vec3; 8]> =
            verts.iter().map(|&v| self.vertex_position(v)).collect();

        // §3 — base centre position, by mode.
        let center = match center_mode {
            CenterMode::Mean => {
                let mut acc = Vec3::ZERO;
                for &p in &positions {
                    acc += p;
                }
                acc / n as f32
            }
            CenterMode::MeanWeighted => {
                // Each corner i is weighted by the sum of the lengths of its
                // two incident face edges. Perimeter edge i runs from corner
                // i to corner i+1, so corner i's incident edges are
                // perimeter edges i and i-1 (mod n).
                let mut acc = Vec3::ZERO;
                let mut total_weight = 0.0_f32;
                for i in 0..n {
                    let next = (i + 1) % n;
                    let prev = (i + n - 1) % n;
                    let curr_edge_len = positions[i].distance(positions[next]);
                    let prev_edge_len = positions[prev].distance(positions[i]);
                    let weight = curr_edge_len + prev_edge_len;
                    acc += positions[i] * weight;
                    total_weight += weight;
                }
                if total_weight > 0.0 {
                    acc / total_weight
                } else {
                    // Degenerate (all corners coincident): fall back to the
                    // uniform mean (§3.2).
                    let mut mean = Vec3::ZERO;
                    for &p in &positions {
                        mean += p;
                    }
                    mean / n as f32
                }
            }
            CenterMode::Bounds => {
                let mut min = Vec3::splat(f32::INFINITY);
                let mut max = Vec3::splat(f32::NEG_INFINITY);
                for &p in &positions {
                    min = min.min(p);
                    max = max.max(p);
                }
                (min + max) * 0.5
            }
        };

        // §4 — the optional normal lift. The direction is the input face's
        // current outward unit normal (§4.1); recompute it for this face so a
        // stale stored normal can't tilt the lift. A degenerate (zero-area)
        // face has no well-defined normal — the normal-computation fallback can
        // hand back a non-finite or zero-length vector — so we treat any such
        // normal as "no lift" and the centre stays at the flat base, regardless
        // of `offset` (§8 degenerate case). The scale (§4.2) is the mean
        // corner-to-base-centre distance under `use_relative_offset`, measured
        // at the flat base centre before the lift; otherwise 1.0.
        const NORMAL_EPS_SQ: f32 = 1e-12;
        let lift = if offset != 0.0 {
            self.face_normal_update(face);
            let face_normal = self.face_normal(face);
            if face_normal.is_finite() && face_normal.length_squared() > NORMAL_EPS_SQ {
                let scale = if use_relative_offset {
                    let mut total = 0.0_f32;
                    for &p in &positions {
                        total += p.distance(center);
                    }
                    total / n as f32
                } else {
                    1.0
                };
                face_normal * offset * scale
            } else {
                Vec3::ZERO
            }
        } else {
            Vec3::ZERO
        };

        // Snapshot the source face's slot — face_kill at the end recycles
        // it, but the slot's CD bucket entry stays addressable for the copy
        // calls in the new-triangle loop.
        let face_slot = face.as_usize();
        let do_loop_interp = !self.loop_attribs.is_empty();

        // NOTE: We delay face_kill until after the new triangles' loops
        // have been copy_into'd from the original loops. Killing the face
        // up front would recycle the original loop slab slots, breaking
        // the loop CD source-of-truth. The perimeter edges happily carry
        // two loops on their radial in the meantime — the original (from
        // `face`) plus the new triangle's. face_kill at the end removes
        // the originals.

        // Build the center vertex.
        let v_center = self.create_vertex(center);

        // Customdata: the centre vertex is the uniform average of every
        // corner.
        let w = 1.0_f32 / n as f32;
        let vert_sources: smallvec::SmallVec<[(f32, usize); 8]> =
            verts.iter().map(|v| (w, v.as_usize())).collect();
        self.vertex_attribs
            .interpolate_into(v_center.as_usize(), &vert_sources);

        // Pre-compute the loop-average source list (one entry per original
        // corner-loop, uniform weight). Reused per spawned triangle for the
        // centre-vertex loop's interp.
        let loop_avg_sources: smallvec::SmallVec<[(f32, usize); 8]> = if do_loop_interp {
            orig_loops.iter().map(|l| (w, l.as_usize())).collect()
        } else {
            smallvec::SmallVec::new()
        };

        // Create the N center→perimeter edges (stack-allocated for small faces).
        let mut center_edges: smallvec::SmallVec<[EdgeKey; 8]> = smallvec::SmallVec::new();
        for &v in &verts {
            center_edges.push(self.create_edge([v_center, v], EdgeCreateFlags::NONE));
        }

        // Build N triangles, each (verts[i], verts[i+1], v_center) with
        // edges (perim_edges[i], center_edges[i+1], center_edges[i]).
        // Each fan face inherits the source face's CD (any custom face
        // attribs). Loop-level CD: the two perimeter corners
        // copy from the original loops at the same vertices; the centre
        // corner takes the uniform average.
        for i in 0..n {
            let j = (i + 1) % n;
            let tri_verts = [verts[i], verts[j], v_center];
            let tri_edges = [perim_edges[i], center_edges[j], center_edges[i]];
            let f_new = self.face_create(&tri_verts, &tri_edges, FaceCreateFlags::NONE);
            self.face_attribs.copy_into(f_new.as_usize(), face_slot);
            if let Some(buffer) = out_faces.as_deref_mut() {
                buffer.push(f_new);
            }

            if do_loop_interp {
                // The new face's loops sit at verts[i], verts[j], v_center
                // in that order in the loop cycle.
                let l_i = self.faces.get(f_new).first_edge_loop;
                let l_j = self.loops.get(l_i).face_loop_links.next;
                let l_c = self.loops.get(l_j).face_loop_links.next;

                self.loop_attribs
                    .copy_into(l_i.as_usize(), orig_loops[i].as_usize());
                self.loop_attribs
                    .copy_into(l_j.as_usize(), orig_loops[j].as_usize());
                self.loop_attribs
                    .interpolate_into(l_c.as_usize(), &loop_avg_sources);
            }
        }

        // §4 step ordering — apply the lift after the centre's customdata
        // interpolation (done above on the flat base centre). Only the position
        // moves; the blended attributes are untouched. `lift` is the single
        // source of truth for "is there a lift": it is finite and non-zero only
        // when offset and the face normal were both usable.
        if lift != Vec3::ZERO {
            let base = self.vertex_position(v_center);
            self.set_vertex_position(v_center, base + lift);
        }

        // Now safe to kill the original face — its loops were the source
        // for the copies above and are no longer needed.
        self.face_kill(face);

        v_center
    }
}

#[cfg(test)]
mod tests {
    use glam::Vec3;

    use super::CenterMode;
    use crate::loopmesh::test_support::create_face_from_verts;
    use crate::loopmesh::*;

    /// An irregular convex polygon where MEAN, MEAN_WEIGHTED, and BOUNDS all
    /// land in distinct places. Corner 2 is pushed far out so the
    /// long-edge weighting visibly pulls the weighted mean toward it relative
    /// to the uniform mean, and the bbox is asymmetric.
    fn irregular_corners() -> [Vec3; 4] {
        [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(4.0, 0.0, 0.0),
            Vec3::new(5.0, 6.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
        ]
    }

    fn uniform_mean(corners: &[Vec3]) -> Vec3 {
        let mut acc = Vec3::ZERO;
        for &p in corners {
            acc += p;
        }
        acc / corners.len() as f32
    }

    fn weighted_mean(corners: &[Vec3]) -> Vec3 {
        let n = corners.len();
        let mut acc = Vec3::ZERO;
        let mut total = 0.0_f32;
        for i in 0..n {
            let next = (i + 1) % n;
            let prev = (i + n - 1) % n;
            let weight = corners[i].distance(corners[next]) + corners[prev].distance(corners[i]);
            acc += corners[i] * weight;
            total += weight;
        }
        acc / total
    }

    fn bounds_center(corners: &[Vec3]) -> Vec3 {
        let mut min = Vec3::splat(f32::INFINITY);
        let mut max = Vec3::splat(f32::NEG_INFINITY);
        for &p in corners {
            min = min.min(p);
            max = max.max(p);
        }
        (min + max) * 0.5
    }

    #[test]
    fn mean_weighted_centre_matches_edge_length_weighted_formula() {
        let corners = irregular_corners();
        let mut mesh = LoopMesh::new();
        let face = create_face_from_verts(&mut mesh, &corners);
        let v_center = mesh.face_poke_mode(face, CenterMode::MeanWeighted);
        let expected = weighted_mean(&corners);
        let got = mesh.vertex_position(v_center);
        assert!(
            got.distance(expected) < 1e-5,
            "weighted centre {got:?} != expected {expected:?}"
        );
    }

    #[test]
    fn bounds_centre_matches_aabb_midpoint_formula() {
        let corners = irregular_corners();
        let mut mesh = LoopMesh::new();
        let face = create_face_from_verts(&mut mesh, &corners);
        let v_center = mesh.face_poke_mode(face, CenterMode::Bounds);
        let expected = bounds_center(&corners);
        let got = mesh.vertex_position(v_center);
        assert!(
            got.distance(expected) < 1e-5,
            "bounds centre {got:?} != expected {expected:?}"
        );
    }

    #[test]
    fn three_modes_differ_on_irregular_face() {
        let corners = irregular_corners();
        let mean = uniform_mean(&corners);
        let weighted = weighted_mean(&corners);
        let bounds = bounds_center(&corners);
        assert!(mean.distance(weighted) > 1e-3);
        assert!(mean.distance(bounds) > 1e-3);
        assert!(weighted.distance(bounds) > 1e-3);
    }

    #[test]
    fn face_poke_defaults_to_uniform_mean() {
        let corners = irregular_corners();
        let mut mesh = LoopMesh::new();
        let face = create_face_from_verts(&mut mesh, &corners);
        let v_center = mesh.face_poke(face);
        let expected = uniform_mean(&corners);
        let got = mesh.vertex_position(v_center);
        assert!(
            got.distance(expected) < 1e-5,
            "default face_poke centre {got:?} != uniform mean {expected:?}"
        );
    }

    #[test]
    fn mean_weighted_collapses_to_mean_on_regular_polygon() {
        // Regular polygon: uniform edge lengths, so MEAN_WEIGHTED == MEAN.
        let corners: Vec<Vec3> = (0..6)
            .map(|i| {
                let theta = (i as f32) * std::f32::consts::TAU / 6.0;
                Vec3::new(theta.cos(), theta.sin(), 0.0)
            })
            .collect();
        let mut mesh = LoopMesh::new();
        let face = create_face_from_verts(&mut mesh, &corners);
        let v_center = mesh.face_poke_mode(face, CenterMode::MeanWeighted);
        let expected = uniform_mean(&corners);
        let got = mesh.vertex_position(v_center);
        assert!(
            got.distance(expected) < 1e-5,
            "regular-polygon weighted centre {got:?} != mean {expected:?}"
        );
    }

    /// Unit square in the XY plane (corners CCW so the normal is +Z), base
    /// centre at (0.5, 0.5, 0).
    fn unit_square() -> [Vec3; 4] {
        [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        ]
    }

    #[test]
    fn offset_lifts_centre_along_normal() {
        // POKE.md §4/§4.3 — absolute offset, no relative scaling. The unit
        // square's normal is ±Z; the lift magnitude equals |offset| and the
        // in-plane (x, y) base centre is unchanged.
        let corners = unit_square();
        let mut mesh = LoopMesh::new();
        let face = create_face_from_verts(&mut mesh, &corners);
        let v_center = mesh.face_poke_offset(face, CenterMode::Mean, 2.0, false);
        let got = mesh.vertex_position(v_center);
        assert!((got.x - 0.5).abs() < 1e-5 && (got.y - 0.5).abs() < 1e-5);
        assert!(
            (got.z.abs() - 2.0).abs() < 1e-5,
            "expected |z| ~= 2.0 after lift, got {got:?}"
        );
    }

    #[test]
    fn offset_zero_stays_flat() {
        // POKE.md §4.3 — offset 0 leaves the centre at the flat base position.
        let corners = unit_square();
        let mut mesh = LoopMesh::new();
        let face = create_face_from_verts(&mut mesh, &corners);
        let v_center = mesh.face_poke_offset(face, CenterMode::Mean, 0.0, false);
        let got = mesh.vertex_position(v_center);
        assert!(
            got.distance(Vec3::new(0.5, 0.5, 0.0)) < 1e-6,
            "offset 0 should stay flat, got {got:?}"
        );
    }

    #[test]
    fn offset_sign_flips_lift_direction() {
        // POKE.md §8 — sign is just a multiplier; +offset and -offset mirror
        // across the face's plane.
        let corners = unit_square();
        let mut up_mesh = LoopMesh::new();
        let up_face = create_face_from_verts(&mut up_mesh, &corners);
        let up = up_mesh.face_poke_offset(up_face, CenterMode::Mean, 1.0, false);
        let up_z = up_mesh.vertex_position(up).z;

        let mut down_mesh = LoopMesh::new();
        let down_face = create_face_from_verts(&mut down_mesh, &corners);
        let down = down_mesh.face_poke_offset(down_face, CenterMode::Mean, -1.0, false);
        let down_z = down_mesh.vertex_position(down).z;

        assert!(up_z.abs() > 1e-5, "lift should be non-zero, got {up_z}");
        assert!(
            (up_z + down_z).abs() < 1e-5,
            "opposite signs should mirror: {up_z} vs {down_z}"
        );
    }

    #[test]
    fn relative_offset_scales_lift_by_mean_corner_distance() {
        // POKE.md §4.2 — relative scale is the mean corner-to-base-centre
        // distance. For the unit square the centre is (0.5,0.5,0) and every
        // corner is sqrt(0.5) away, so offset 1.0 lifts by exactly sqrt(0.5).
        let corners = unit_square();
        let mut mesh = LoopMesh::new();
        let face = create_face_from_verts(&mut mesh, &corners);
        let v_center = mesh.face_poke_offset(face, CenterMode::Mean, 1.0, true);
        let lift = mesh.vertex_position(v_center).z.abs();
        let expected = 0.5_f32.sqrt();
        assert!(
            (lift - expected).abs() < 1e-5,
            "relative lift expected ~= {expected}, got {lift}"
        );
    }

    #[test]
    fn relative_offset_differs_from_absolute() {
        // The relative scale (sqrt(0.5) ~= 0.707) genuinely departs from the
        // absolute lift on the same offset, so the flag is observable.
        let corners = unit_square();
        let mut abs_mesh = LoopMesh::new();
        let abs_face = create_face_from_verts(&mut abs_mesh, &corners);
        let abs_v = abs_mesh.face_poke_offset(abs_face, CenterMode::Mean, 1.0, false);
        let abs_z = abs_mesh.vertex_position(abs_v).z.abs();

        let mut rel_mesh = LoopMesh::new();
        let rel_face = create_face_from_verts(&mut rel_mesh, &corners);
        let rel_v = rel_mesh.face_poke_offset(rel_face, CenterMode::Mean, 1.0, true);
        let rel_z = rel_mesh.vertex_position(rel_v).z.abs();

        assert!(
            (abs_z - rel_z).abs() > 1e-3,
            "relative ({rel_z}) and absolute ({abs_z}) lifts should differ"
        );
    }

    #[test]
    fn degenerate_face_offset_collapses_to_no_lift() {
        // POKE.md §8 — a zero-area (collinear) face has no well-defined normal,
        // so a non-zero offset must produce zero lift: the centre stays at the
        // flat base centre and its position must be finite (no NaN leak from the
        // degenerate normal). Checked for both relative-offset settings.
        //
        // Four collinear corners along X give zero area. The base centre under
        // CenterMode::Mean is their arithmetic mean, (1.5, 0, 0).
        let corners = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::new(3.0, 0.0, 0.0),
        ];
        let expected_base = Vec3::new(1.5, 0.0, 0.0);

        for use_relative in [false, true] {
            let mut mesh = LoopMesh::new();
            let face = create_face_from_verts(&mut mesh, &corners);
            let v_center = mesh.face_poke_offset(face, CenterMode::Mean, 5.0, use_relative);
            let got = mesh.vertex_position(v_center);
            assert!(
                got.is_finite(),
                "degenerate poke (use_relative={use_relative}) leaked non-finite position {got:?}"
            );
            assert!(
                got.distance(expected_base) < 1e-5,
                "degenerate poke (use_relative={use_relative}) lifted off the flat base: {got:?}"
            );
        }
    }

    // ---- Multi-face poke wrapper (PokeResult / §7 output buffers) ----

    fn corner_verts(mesh: &LoopMesh, face: FaceKey) -> Vec<VertexKey> {
        let mut out = Vec::new();
        let mut it = mesh.face_loops(face);
        while let Some(l) = it.next(mesh) {
            out.push(mesh.loops.get(l).vertex);
        }
        out
    }

    fn tri_corner_verts(mesh: &LoopMesh, face: FaceKey) -> [VertexKey; 3] {
        let l0 = mesh.faces.get(face).first_edge_loop;
        let l1 = mesh.loops.get(l0).face_loop_links.next;
        let l2 = mesh.loops.get(l1).face_loop_links.next;
        [
            mesh.loops.get(l0).vertex,
            mesh.loops.get(l1).vertex,
            mesh.loops.get(l2).vertex,
        ]
    }

    /// Two quads sharing the edge v1->v2. Quad A = (v0,v1,v2,v3),
    /// quad B = (v1,v4,v5,v2). Returns (face_a, face_b, perimeter verts of A,
    /// perimeter verts of B).
    fn two_adjacent_quads(
        mesh: &mut LoopMesh,
    ) -> (FaceKey, FaceKey, Vec<VertexKey>, Vec<VertexKey>) {
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

        let face_a = mesh.face_create(
            &[v0, v1, v2, v3],
            &[e01, e12, e23, e30],
            FaceCreateFlags::NONE,
        );
        // Quad B walks v1->v4->v5->v2->v1, sharing edge e12 (v1<->v2) with A.
        let face_b = mesh.face_create(
            &[v1, v4, v5, v2],
            &[e14, e45, e52, e12],
            FaceCreateFlags::NONE,
        );

        (face_a, face_b, vec![v0, v1, v2, v3], vec![v1, v4, v5, v2])
    }

    #[test]
    fn poke_single_face_buffers_match_corner_count() {
        // §7 — one centre vert, n triangles for an n-corner face.
        let corners = unit_square();
        let mut mesh = LoopMesh::new();
        let face = create_face_from_verts(&mut mesh, &corners);
        let perimeter = corner_verts(&mesh, face);

        let result = mesh.poke(&[face], CenterMode::Mean, 0.0, false);

        assert_eq!(result.verts.len(), 1, "one centre vert per input face");
        assert_eq!(result.faces.len(), 4, "quad yields four triangles");
        // The centre vert is freshly created — not one of the perimeter verts.
        assert!(!perimeter.contains(&result.verts[0]));
    }

    #[test]
    fn poke_faces_out_grouped_in_loop_cycle_order() {
        // §5.2 / §7 — triangle i has corners (v_i, v_{i+1}, v_c) and the
        // group is emitted in loop-cycle order. Verify each spawned triangle's
        // perimeter corners are the consecutive perimeter pair for its index.
        let corners = unit_square();
        let mut mesh = LoopMesh::new();
        let face = create_face_from_verts(&mut mesh, &corners);
        let perimeter = corner_verts(&mesh, face);
        let n = perimeter.len();

        let result = mesh.poke(&[face], CenterMode::Mean, 0.0, false);
        let v_center = result.verts[0];

        assert_eq!(result.faces.len(), n);
        for (i, &tri) in result.faces.iter().enumerate() {
            let tri_verts = tri_corner_verts(&mesh, tri);
            let expected = [perimeter[i], perimeter[(i + 1) % n], v_center];
            assert_eq!(
                tri_verts, expected,
                "triangle {i} corners {tri_verts:?} != expected loop-cycle order {expected:?}"
            );
        }
    }

    #[test]
    fn poke_multi_face_verts_out_in_input_order() {
        // §7 — verts.out follows input-faces-slice order, one per face. Two
        // adjacent quads sharing a perimeter edge are poked independently.
        let mut mesh = LoopMesh::new();
        let (face_a, face_b, perim_a, perim_b) = two_adjacent_quads(&mut mesh);

        let result = mesh.poke(&[face_a, face_b], CenterMode::Mean, 0.0, false);

        assert_eq!(result.verts.len(), 2, "one centre vert per input face");
        // faces.out is grouped per input face: first quad A's 4 triangles, then
        // quad B's 4 triangles (§7).
        assert_eq!(result.faces.len(), 8);

        let centre_a = result.verts[0];
        let centre_b = result.verts[1];
        // Each centre vert is new (not a perimeter vert of either quad).
        assert!(!perim_a.contains(&centre_a) && !perim_b.contains(&centre_a));
        assert!(!perim_a.contains(&centre_b) && !perim_b.contains(&centre_b));

        // Quad A's group (faces 0..4) fans from centre_a; quad B's (4..8) from
        // centre_b.
        for tri in &result.faces[0..4] {
            assert!(tri_corner_verts(&mesh, *tri).contains(&centre_a));
        }
        for tri in &result.faces[4..8] {
            assert!(tri_corner_verts(&mesh, *tri).contains(&centre_b));
        }
    }

    #[test]
    fn poke_adjacent_quads_share_perimeter_edge_centres_distinct() {
        // §5.3 — poking two faces that share a perimeter edge produces two
        // distinct centre verts; the shared edge is not consumed by either fan.
        let mut mesh = LoopMesh::new();
        let (face_a, face_b, _perim_a, _perim_b) = two_adjacent_quads(&mut mesh);

        let result = mesh.poke(&[face_a, face_b], CenterMode::Mean, 0.0, false);
        assert_ne!(result.verts[0], result.verts[1]);
    }

    #[test]
    fn poke_triangle_ngon_mix_face_counts() {
        // §7 — faces.out length == Σ corner counts across a triangle + pentagon
        // mix; verts.out length == number of input faces.
        let mut mesh = LoopMesh::new();
        let tri = create_face_from_verts(
            &mut mesh,
            &[
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
            ],
        );
        let pentagon: Vec<Vec3> = (0..5)
            .map(|i| {
                let theta = (i as f32) * std::f32::consts::TAU / 5.0;
                Vec3::new(10.0 + theta.cos(), theta.sin(), 0.0)
            })
            .collect();
        let pent = create_face_from_verts(&mut mesh, &pentagon);

        let result = mesh.poke(&[tri, pent], CenterMode::MeanWeighted, 0.0, false);

        assert_eq!(result.verts.len(), 2);
        assert_eq!(result.faces.len(), 3 + 5, "Σ corner counts");
        // First three faces are the triangle's fan, the next five the pentagon's.
        for face in &result.faces[0..3] {
            assert!(tri_corner_verts(&mesh, *face).contains(&result.verts[0]));
        }
        for face in &result.faces[3..8] {
            assert!(tri_corner_verts(&mesh, *face).contains(&result.verts[1]));
        }
    }

    #[test]
    fn poke_empty_input_yields_empty_buffers() {
        // §8 — empty faces input does nothing; both buffers empty.
        let mut mesh = LoopMesh::new();
        let result = mesh.poke(&[], CenterMode::Mean, 0.0, false);
        assert!(result.verts.is_empty() && result.faces.is_empty());
    }
}
