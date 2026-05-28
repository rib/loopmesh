//! Polygon-interior attribute resample weights — `radial-mesh-guides/BMESH.md`
//! §10.1. Given a source polygon's corners and a query point, both flattened
//! to a shared 2D plane, produce one weight per corner that blends the
//! corners' attributes at the query point. The weights are **mean value
//! coordinates** (Floater / Meyer et al.): they generalise barycentric
//! coordinates to an arbitrary convex-or-concave polygon, stay smooth across
//! the interior, and reduce to ordinary edge-linear blends on the boundary.

use glam::{Vec2, Vec3};

/// The axis-dominant 2D projection of §10.1.1: pick the axis along which the
/// normal's component is largest in magnitude and drop it, keeping the other
/// two with a fixed handedness so the projection is orientation-stable.
///
/// Returns the two basis selectors as a function the caller applies to every
/// corner and to the query point, guaranteeing one shared 2D frame.
#[derive(Clone, Copy)]
pub struct PlaneProjection {
    /// 0 = drop x (keep y,z), 1 = drop y (keep z,x), 2 = drop z (keep x,y).
    dominant: u8,
}

impl PlaneProjection {
    pub fn from_normal(normal: Vec3) -> Self {
        let a = normal.abs();
        let dominant = if a.x >= a.y && a.x >= a.z {
            0
        } else if a.y >= a.z {
            1
        } else {
            2
        };
        Self { dominant }
    }

    pub fn project(&self, p: Vec3) -> Vec2 {
        match self.dominant {
            // Keep the remaining two axes in a right-handed cyclic order so the
            // projection is a consistent affine flattening (§10.1.1).
            0 => Vec2::new(p.y, p.z),
            1 => Vec2::new(p.z, p.x),
            _ => Vec2::new(p.x, p.y),
        }
    }
}

/// Signed 2D cross product (parallelogram area) of two spokes.
fn cross_2d(a: Vec2, b: Vec2) -> f64 {
    (a.x as f64) * (b.y as f64) - (a.y as f64) * (b.x as f64)
}

/// Half-tangent of the angle between two spokes emanating from the query
/// point: `tan(theta/2) = (|d_a||d_b| - d_a·d_b) / signed_cross_2d(d_a, d_b)`.
/// Built without an explicit `atan` so the sign of the swept angle is carried
/// (load-bearing for concave polygons and exterior query points). Accumulated
/// in `f64` (§10.1.2 numerical notes).
fn half_tan(d_a: Vec2, d_b: Vec2) -> f64 {
    let area = cross_2d(d_a, d_b);
    if area == 0.0 {
        return 0.0;
    }
    let dot = (d_a.x as f64) * (d_b.x as f64) + (d_a.y as f64) * (d_b.y as f64);
    let len = (d_a.length() as f64) * (d_b.length() as f64);
    let result = (len - dot) / area;
    if result.is_finite() {
        result
    } else {
        0.0
    }
}

/// Mean value coordinates of `q` over the projected polygon `p` (in loop-cycle
/// order), wired with the §10.1.3 boundary short-circuits *before* the
/// §10.1.2 smooth-interior formula. Returns `n` weights summing to one.
///
/// The on-corner branch is the identity-rebake guarantee: a query point that
/// coincides with corner `i` yields the unit vector `e_i`, so an attribute
/// resample returns that corner's stored value exactly.
pub fn mean_value_weights(p: &[Vec2], q: Vec2) -> Vec<f32> {
    let n = p.len();
    let mut w = vec![0.0f32; n];
    if n == 0 {
        return w;
    }

    // Scale-aware epsilon: track the largest coordinate magnitude seen so the
    // boundary tolerances follow the data's scale (§10.1.3).
    let mut scale = q.x.abs().max(q.y.abs());
    for pi in p {
        scale = scale.max(pi.x.abs()).max(pi.y.abs());
    }
    let eps = 1.0e-6_f32 * scale.max(1.0);
    let eps_sq = eps * eps;

    let d: Vec<Vec2> = p.iter().map(|&pi| pi - q).collect();

    // §10.1.3 on-corner: q == p[i]. Returns the identity unit vector.
    for i in 0..n {
        if d[i].length_squared() <= eps_sq {
            w[i] = 1.0;
            return w;
        }
    }

    // §10.1.3 on-edge: q on segment (p[i], p[i+1]) without coinciding with an
    // endpoint → clamped linear blend of the two endpoints.
    for i in 0..n {
        let j = (i + 1) % n;
        let seg = p[j] - p[i];
        let seg_len_sq = seg.length_squared();
        if seg_len_sq <= eps_sq {
            continue;
        }
        let t = (q - p[i]).dot(seg) / seg_len_sq;
        if t <= 0.0 || t >= 1.0 {
            continue;
        }
        let closest = p[i] + seg * t;
        if (q - closest).length_squared() <= eps_sq {
            w[i] = 1.0 - t;
            w[j] = t;
            return w;
        }
    }

    // §10.1.2 smooth interior. ht_edge[i] = tan(half-angle) subtended by edge
    // (i, i+1) at q.
    let mut ht_edge = vec![0.0f64; n];
    for i in 0..n {
        let j = (i + 1) % n;
        ht_edge[i] = half_tan(d[i], d[j]);
    }

    let mut total = 0.0f32;
    for i in 0..n {
        let prev = (i + n - 1) % n;
        let r_i = d[i].length();
        // The on-corner identity is handled above; this guard only avoids a
        // divide-by-zero on the smooth path.
        let wi = if r_i == 0.0 {
            0.0
        } else {
            ((ht_edge[prev] + ht_edge[i]) / r_i as f64) as f32
        };
        w[i] = wi;
        total += wi;
    }

    if total != 0.0 {
        for wi in &mut w {
            *wi /= total;
        }
    }
    w
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn on_corner_is_identity() {
        let p = [
            Vec2::new(0.0, 0.0),
            Vec2::new(1.0, 0.0),
            Vec2::new(1.0, 1.0),
            Vec2::new(0.0, 1.0),
        ];
        let w = mean_value_weights(&p, Vec2::new(1.0, 1.0));
        assert!((w[2] - 1.0).abs() < 1.0e-6);
        for (i, &wi) in w.iter().enumerate() {
            if i != 2 {
                assert!(wi.abs() < 1.0e-6);
            }
        }
    }

    #[test]
    fn on_edge_is_linear_blend() {
        let p = [
            Vec2::new(0.0, 0.0),
            Vec2::new(2.0, 0.0),
            Vec2::new(2.0, 2.0),
            Vec2::new(0.0, 2.0),
        ];
        // Quarter of the way along edge 0->1.
        let w = mean_value_weights(&p, Vec2::new(0.5, 0.0));
        assert!((w[0] - 0.75).abs() < 1.0e-5, "w0={}", w[0]);
        assert!((w[1] - 0.25).abs() < 1.0e-5, "w1={}", w[1]);
        assert!(w[2].abs() < 1.0e-6);
        assert!(w[3].abs() < 1.0e-6);
    }

    #[test]
    fn partition_of_unity_and_centre_reproduces_point() {
        let p = [
            Vec2::new(0.0, 0.0),
            Vec2::new(2.0, 0.0),
            Vec2::new(2.0, 2.0),
            Vec2::new(0.0, 2.0),
        ];
        let q = Vec2::new(0.7, 1.3);
        let w = mean_value_weights(&p, q);
        let sum: f32 = w.iter().sum();
        assert!((sum - 1.0).abs() < 1.0e-5, "sum={sum}");
        // Mean value coords reproduce the query point for a convex polygon.
        let mut rx = 0.0;
        let mut ry = 0.0;
        for (i, &wi) in w.iter().enumerate() {
            rx += wi * p[i].x;
            ry += wi * p[i].y;
        }
        assert!((rx - q.x).abs() < 1.0e-4 && (ry - q.y).abs() < 1.0e-4);
    }

    #[test]
    fn quad_centre_is_non_uniform_off_centroid() {
        // A non-square quad: a point that is NOT the centroid must give
        // non-uniform weights — distinguishes mean-value from a plain average.
        let p = [
            Vec2::new(0.0, 0.0),
            Vec2::new(4.0, 0.0),
            Vec2::new(4.0, 1.0),
            Vec2::new(0.0, 1.0),
        ];
        let w = mean_value_weights(&p, Vec2::new(0.5, 0.5));
        // Nearer the left edge → corners 0,3 should outweigh 1,2.
        assert!(w[0] > w[1], "{w:?}");
        assert!(w[3] > w[2], "{w:?}");
    }
}
