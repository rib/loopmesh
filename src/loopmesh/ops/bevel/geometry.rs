//! Small geometric helpers shared across the bevel stages.

use super::*;

/// The closest point on the segment `[a, b]` to `p` (clamped to the segment ends).
///
/// The only guard is against a segment of exactly zero length, which has no direction to
/// project onto. It is deliberately *not* an epsilon: `length_squared` is a length², so any
/// fixed floor on it is a floor on the caller's world scale — `f32::EPSILON` would answer
/// `a` for every segment shorter than `3.4e-4`, which at a small bevel offset is every
/// segment there is. Nothing else needs a floor: the parameter is clamped to `[0, 1]`, so a
/// denominator small enough to overflow the quotient still lands on the nearer end, which
/// is the answer a degenerate segment wants.
pub(super) fn closest_point_on_segment(p: Vec3, a: Vec3, b: Vec3) -> Vec3 {
    let axis = b - a;
    let length_squared = axis.length_squared();
    if length_squared == 0.0 {
        return a;
    }
    let t = ((p - a).dot(axis) / length_squared).clamp(0.0, 1.0);
    a + axis * t
}

/// Squared distance from `p` to the segment `[a, b]` (BEVEL.md §5.2.1.1.1(i)).
pub(super) fn dist_sq_to_segment(p: Vec3, a: Vec3, b: Vec3) -> f32 {
    (p - closest_point_on_segment(p, a, b)).length_squared()
}

/// The point-in-polygon **ray-crossing parity** test of BEVEL.md §5.2.1.1.1(ii), over a
/// polygon already flattened to 2D through the donor's axis-dominant projection. Deliberately
/// *not* tolerance-guarded and *not* a strict-interior test: a point that projects exactly onto
/// the boundary is decided by the parity arithmetic and may fall either way, exactly as the
/// guide requires.
pub(super) fn point_in_polygon_parity(polygon: &[Vec2], q: Vec2) -> bool {
    let n = polygon.len();
    if n < 3 {
        return false;
    }
    let mut inside = false;
    let mut j = n - 1;
    for i in 0..n {
        let pi = polygon[i];
        let pj = polygon[j];
        if (pi.y > q.y) != (pj.y > q.y) {
            let crossing_x = pi.x + (q.y - pi.y) / (pj.y - pi.y) * (pj.x - pi.x);
            if q.x < crossing_x {
                inside = !inside;
            }
        }
        j = i;
    }
    inside
}

impl LoopMesh {
    /// A face's geometric normal, computed on the fly from its loop cycle by
    /// Newell's method (BEVEL.md §7.3 needs an in-plane perpendicular; the stored
    /// face normal may be stale during surgery, so this reads positions directly).
    pub(super) fn face_plane_normal(&self, face: FaceKey) -> Vec3 {
        self.face_newell_normal(face).unwrap_or(Vec3::Z)
    }

    /// The face's unit Newell normal, or `None` when the polygon is degenerate
    /// (near-zero area). `face_plane_normal` substitutes `Vec3::Z` for the `None`
    /// case; the §8.9.1 reflex reference (`reflex_reference_normal`) instead uses it
    /// to fall through to the next candidate in the forward → back → `n_v` chain.
    ///
    /// The unitisation is §7.6.2's shared `normalise` **arithmetic** — one `sqrt`, one
    /// scalar reciprocal, three multiplies — not three per-component divisions. This is
    /// directly upstream of the offset lines (a case-1/2 sign reference, §7.5.1's
    /// regime-2 plane normal outright), and the guide's `2.5%`-of-shallow-sectors
    /// normalisation axis is measured on exactly that spelling. The **floor** stays this
    /// function's own — it reports `None` where the shared helper would report the zero
    /// vector, and callers key on that distinction.
    pub(super) fn face_newell_normal(&self, face: FaceKey) -> Option<Vec3> {
        let verts = self.face_vert_keys(face);
        let mut normal = Vec3::ZERO;
        for i in 0..verts.len() {
            let current = self.vertex_position(verts[i]);
            let next = self.vertex_position(verts[(i + 1) % verts.len()]);
            normal.x += (current.y - next.y) * (current.z + next.z);
            normal.y += (current.z - next.z) * (current.x + next.x);
            normal.z += (current.x - next.x) * (current.y + next.y);
        }
        let length = normal.length();
        if length <= f32::EPSILON {
            None
        } else {
            Some(normal * (1.0 / length))
        }
    }

    /// The midpoint of a face's axis-aligned extent — key 4 of BEVEL.md §6.6.1.4's
    /// tie-break, which is the bounding-box centre and *not* the area centroid.
    pub(super) fn face_bounding_box_center(&self, face: FaceKey) -> Vec3 {
        let mut min = Vec3::splat(f32::INFINITY);
        let mut max = Vec3::splat(f32::NEG_INFINITY);
        for vertex in self.face_vert_keys(face) {
            let position = self.vertex_position(vertex);
            min = min.min(position);
            max = max.max(position);
        }
        (min + max) * 0.5
    }

    /// A face's centroid (mean of its corner positions), used to orient the
    /// in-face strip-slide direction toward the face interior (BEVEL.md §7.3).
    pub(super) fn face_center_of_mass(&self, face: FaceKey) -> Vec3 {
        let verts = self.face_vert_keys(face);
        let mut sum = Vec3::ZERO;
        for &v in &verts {
            sum += self.vertex_position(v);
        }
        sum / verts.len() as f32
    }
}

/// Squared-sine floor below which two **unit** directions count as parallel for the
/// callers that need a tolerant parallel test of their own. The two-line solve's own
/// bail is an exact zero (§7.6.2); this is the extra guard §8.2.2.1's `mid` solve puts
/// in front of it, where a near-parallel pair has no useful crossing to report at all.
///
/// Read on `length_squared()` of the cross of two **already unitised** directions, so
/// the bound is `sin²θ` and independent of the inputs' lengths.
pub(super) const LINE_PARALLEL_SIN_SQ_TOLERANCE: f32 = 1.0e-7;

/// The point on the first line (`p0 + t·d0`) nearest the second line
/// (`p1 + s·d1`) — the first-line foot of the mutual closest-approach pair
/// (BEVEL.md §8.2.2.1 `mid` solve). `None` when either direction is degenerate or the
/// lines are parallel.
///
/// The single solve lives in `closest_approach`; this is the point-and-direction spelling
/// of it, with both directions unitised first so that `LINE_PARALLEL_SIN_SQ_TOLERANCE` reads as a
/// bound on `sin²θ` independent of the inputs' lengths. That bound is this caller's own:
/// the shared solve bails only on an exact zero (§7.6.2), which would let a
/// near-parallel pair here report a crossing tens of units away instead of `None`.
pub(super) fn line_line_nearest_on_first(
    first_point: Vec3,
    first_direction: Vec3,
    second_point: Vec3,
    second_direction: Vec3,
) -> Option<Vec3> {
    let first_unit = first_direction.normalize_or_zero();
    let second_unit = second_direction.normalize_or_zero();
    if first_unit == Vec3::ZERO || second_unit == Vec3::ZERO {
        return None;
    }
    if first_unit.cross(second_unit).length_squared() <= LINE_PARALLEL_SIN_SQ_TOLERANCE {
        return None;
    }
    closest_approach(
        &Line {
            point: first_point,
            direction: first_unit,
        },
        &Line {
            point: second_point,
            direction: second_unit,
        },
    )
    .map(|(on_first, _)| on_first)
}

/// The sphere (center, radius) through four points (BEVEL.md §8.5.2: the rounded
/// surface the ADJ tri-corner's ring + chain verts lie on). Solves the three
/// equidistance planes `|x − p_i|² = |x − p_0|²`. `None` when the four points are
/// (near-)coplanar so the system is singular.
pub(super) fn circumsphere(points: [Vec3; 4]) -> Option<(Vec3, f32)> {
    let base = points[0];
    let mut rows = [Vec3::ZERO; 3];
    let mut rhs = [0.0_f32; 3];
    for i in 0..3 {
        let p = points[i + 1];
        rows[i] = 2.0 * (p - base);
        rhs[i] = p.length_squared() - base.length_squared();
    }
    let m = glam::Mat3::from_cols(
        Vec3::new(rows[0].x, rows[1].x, rows[2].x),
        Vec3::new(rows[0].y, rows[1].y, rows[2].y),
        Vec3::new(rows[0].z, rows[1].z, rows[2].z),
    );
    if m.determinant().abs() <= 1.0e-9 {
        return None;
    }
    let center = m.inverse() * Vec3::new(rhs[0], rhs[1], rhs[2]);
    Some((center, (center - base).length()))
}

/// Intersect the line `point + t·direction` with the plane through `plane_point`
/// with normal `plane_normal` (BEVEL.md §8.2.2.1 projection). `None` when the
/// line is parallel to the plane.
///
/// **The parallel arm is a required landing point, not a convenience.** §8.2.2.1 ("The
/// other degeneracy"): *"when the ray direction is parallel to the plane, the sample must
/// end up at the raw input point, unmoved. How you spell that is free — an
/// `intersect_line_plane` that returns the input point itself, or one that reports 'no
/// intersection' and leaves the caller to keep the raw point, reach the same place."*
/// This is the second spelling, and the sole caller ([`ProfileCell::sample`]) keeps the
/// raw point. What is not free is dividing anyway: on a zero-cell-axis profile at a
/// §8.3.0.3 weld both the denominator *and* the numerator vanish — the projection
/// direction lies in the weld plane and the raw point is already in it — so the literal
/// quotient is `NaN` on every interior vert of that configuration.
///
/// This arm is distinct from the **zero-`projection_dir`** arm, which §8.2.2.1 pins
/// separately and [`ProfileCell::sample`] tests before calling here: no direction to cast
/// along, versus a direction that never leaves the plane.
pub(super) fn intersect_line_plane(
    point: Vec3,
    direction: Vec3,
    plane_point: Vec3,
    plane_normal: Vec3,
) -> Option<Vec3> {
    let denominator = direction.dot(plane_normal);
    if denominator.abs() <= 1.0e-9 {
        return None;
    }
    let t = (plane_point - point).dot(plane_normal) / denominator;
    Some(point + direction * t)
}
