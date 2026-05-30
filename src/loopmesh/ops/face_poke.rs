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

impl LoopMesh {
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
}
