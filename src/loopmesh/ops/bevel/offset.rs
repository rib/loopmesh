//! Offset modes and the overlap clamp (BEVEL.md §4.3 / §4.4 / §8.7.2).

use super::*;

/// The corner axis `Σ_i normalize(v − A_i)` (BEVEL.md §4.3): the un-normalised
/// sum of the inward edge directions. Only its direction matters — it is the
/// reference each edge's angle `φ_i` is measured against.
pub(super) fn corner_axis(vertex: Vec3, far_positions: &[Vec3]) -> Vec3 {
    far_positions.iter().fold(Vec3::ZERO, |axis, &far| {
        let inward = vertex - far;
        let length = inward.length();
        if length <= f32::EPSILON {
            axis
        } else {
            axis + inward / length
        }
    })
}

/// The largest offset a single beveled edge can tolerate before its two ring
/// verts collide (BEVEL.md §4.4.2): `L / (k_a + k_b)`, where `k_a`, `k_b` are the
/// per-offset slide rates at the two ends (`k_b = 0` when the far end is not
/// beveled). When neither end slides, the edge never limits.
pub(super) fn edge_collision_offset(length: f32, rate_near: f32, rate_far: f32) -> f32 {
    let rate_sum = rate_near + rate_far;
    if rate_sum <= 0.0 {
        f32::INFINITY
    } else {
        length / rate_sum
    }
}

/// The gated per-corner bevel weight for `affect = Vertices` (BEVEL.md §8.7.2).
/// The three-way rule: gate off ⇒ `1.0` (full offset, weights ignored); gate on
/// and the vertex present in the slice ⇒ its stored float; gate on and absent ⇒
/// `0.0` (no slide — the asymmetric "weighting on but nothing to read" default).
pub(super) fn corner_weight(
    use_weights: bool,
    vert_weights: &[(VertexKey, f32)],
    vertex: VertexKey,
) -> f32 {
    if !use_weights {
        return 1.0;
    }
    vert_weights
        .iter()
        .find_map(|&(key, weight)| (key == vertex).then_some(weight))
        .unwrap_or(0.0)
}

/// The §4.4 global clamp factor: gather every beveled edge's collision offset,
/// take the smallest, and shrink everything by `min_collision / requested` if
/// that minimum falls below the requested offset (BEVEL.md §4.4.2). A shared
/// edge contributes both endpoints' slide rates; an edge beveled at one end only
/// contributes its near rate (far rate 0). Returns a factor in `(0, 1]`: every
/// beveled vertex's resolved slide is multiplied by it uniformly (§4.4.3).
pub(super) fn clamp_factor(
    requested_offset: f32,
    beveled: &[VertexKey],
    frames: &[BevelFrame],
) -> f32 {
    let mut min_collision = f32::INFINITY;
    for frame in frames {
        for placement in &frame.placements {
            let length = (placement.far_position - frame.vertex_position).length();
            let rate_near = placement.rate;
            // §4.4.2: a two-ended collision. If the far endpoint is also beveled,
            // look up *its* slide rate on this same edge so both ends' slides are
            // summed; otherwise the far rate is 0 and the edge collides only at
            // its own far endpoint.
            let rate_far = if beveled.contains(&placement.far_vertex) {
                far_end_rate(frames, placement.edge, frame.vertex)
            } else {
                0.0
            };
            let collision = edge_collision_offset(length, rate_near, rate_far);
            if collision < min_collision {
                min_collision = collision;
            }
        }
    }

    if min_collision < requested_offset && requested_offset > 0.0 {
        min_collision / requested_offset
    } else {
        1.0
    }
}

/// The slide rate at the far end of `edge`: the rate the beveled frame whose
/// vertex is *not* `near_vertex` assigns to `edge`. Returns 0 if no other frame
/// bevels `edge` (BEVEL.md §4.4.2).
pub(super) fn far_end_rate(frames: &[BevelFrame], edge: EdgeKey, near_vertex: VertexKey) -> f32 {
    for frame in frames {
        if frame.vertex == near_vertex {
            continue;
        }
        for placement in &frame.placements {
            if placement.edge == edge {
                return placement.rate;
            }
        }
    }
    0.0
}

/// The smallest slide a near-degenerate angle mode is allowed to produce
/// (BEVEL.md §4.3): a hundredth of `offset`, the established fallback when the
/// `WIDTH` / `DEPTH` denominator underflows rather than dividing by ~0.
pub(super) const DEGENERATE_SLIDE_FRACTION: f32 = 0.01;

/// Per-edge slide distance `d_i` for a single-segment vertex bevel (BEVEL.md
/// §4 / §4.3). Each `offset_type` interprets `offset` as a different physical
/// quantity and solves for the slide that realises it; the surgery downstream
/// is mode-agnostic.
pub(super) fn slide_distance(
    offset: f32,
    offset_type: BevelOffsetType,
    vertex: Vec3,
    far: Vec3,
    corner_axis: Vec3,
) -> f32 {
    match offset_type {
        // §4 / §4.3: plain and fixed slide — `d_i = offset`. The two coincide at
        // the foundational no-weights level (§4.3).
        BevelOffsetType::Offset | BevelOffsetType::Absolute => offset,

        // §4.3: a per-edge fraction of that edge's own length, no angle term.
        BevelOffsetType::Percent => (far - vertex).length() * offset / 100.0,

        // §4.3 angle modes: re-measure against `φ_i`, the angle between the
        // edge's inward direction and the corner axis.
        BevelOffsetType::Width => {
            let phi = inward_axis_angle(vertex, far, corner_axis);
            let denominator = (2.0 * phi.sin()).abs();
            if denominator <= f32::EPSILON {
                offset * DEGENERATE_SLIDE_FRACTION
            } else {
                offset / denominator
            }
        }
        BevelOffsetType::Depth => {
            let phi = inward_axis_angle(vertex, far, corner_axis);
            let denominator = phi.cos().abs();
            if denominator <= f32::EPSILON {
                offset * DEGENERATE_SLIDE_FRACTION
            } else {
                offset / denominator
            }
        }
    }
}

/// `φ_i` — the angle in `0..π` between edge `e_i`'s inward direction
/// `normalize(v − A_i)` and the corner axis (BEVEL.md §4.3).
pub(super) fn inward_axis_angle(vertex: Vec3, far: Vec3, corner_axis: Vec3) -> f32 {
    let inward = vertex - far;
    let inward_length = inward.length();
    let axis_length = corner_axis.length();
    if inward_length <= f32::EPSILON || axis_length <= f32::EPSILON {
        return 0.0;
    }
    let cosine = (inward / inward_length).dot(corner_axis / axis_length);
    cosine.clamp(-1.0, 1.0).acos()
}
