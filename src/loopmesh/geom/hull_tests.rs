//! Tier-2 property tests for the in-tree quickhull (general-position clouds).
//!
//! No oracle: a hull is self-checking. We assert the invariants the
//! prerequisite file lists —
//!   1. every input point is on or inside the hull;
//!   2. the hull is globally convex (every input below or on every face plane);
//!   3. every face is outward-wound (and, being a triangle, trivially planar
//!      and convex).
//! Exact orientation in the assertions uses the same `robust::orient3d`
//! predicate the implementation does, so "on or inside" is decided exactly.
//!
//! Degenerate clouds (coplanar / collinear / coincident / cospherical) are out
//! of scope for this first cut and are not exercised here.

use glam::DVec3;
use robust::{orient3d, Coord3D};

use super::{convex_hull, ConvexHullProvider, HullTopology, Quickhull};

const EPS: f64 = 1e-9;

fn coord(point: DVec3) -> Coord3D<f64> {
    Coord3D {
        x: point.x,
        y: point.y,
        z: point.z,
    }
}

/// Orientation of `d` against the face plane `(a, b, c)`. With outward-wound
/// faces, a point strictly inside the hull is "below" → strictly positive.
fn orient(a: DVec3, b: DVec3, c: DVec3, d: DVec3) -> f64 {
    orient3d(coord(a), coord(b), coord(c), coord(d))
}

/// Assert the three hull invariants on `topo` over `points`.
fn assert_hull_invariants(points: &[DVec3], topo: &HullTopology) {
    assert!(!topo.faces.is_empty(), "hull produced no faces");
    assert!(topo.vertices.len() >= 4, "hull needs at least a tetrahedron");

    // The on-hull vertex set must be a subset of the input indices, sorted and
    // unique, and every face index must be on the hull.
    assert!(
        topo.vertices.windows(2).all(|w| w[0] < w[1]),
        "hull vertices not strictly ascending / unique"
    );
    for face in &topo.faces {
        for index in [face.a, face.b, face.c] {
            assert!(index < points.len(), "face references out-of-range index");
            assert!(
                topo.vertices.binary_search(&index).is_ok(),
                "face index {index} missing from hull vertex set"
            );
        }
    }

    for (face_number, face) in topo.faces.iter().enumerate() {
        let a = points[face.a];
        let b = points[face.b];
        let c = points[face.c];

        // Invariant 3: outward-wound — every OTHER input point is on or inside
        // (>= 0) relative to this face plane. A negative value would mean a
        // point outside the face (hull not closed / wound inward).
        for (index, &point) in points.iter().enumerate() {
            if index == face.a || index == face.b || index == face.c {
                continue;
            }
            let side = orient(a, b, c, point);
            assert!(
                side >= -EPS,
                "point {index} lies outside face {face_number}: orient={side}"
            );
        }

        // Sanity: the face is non-degenerate (the three points are not
        // collinear). A general-position hull never emits a sliver.
        let normal = (b - a).cross(c - a);
        assert!(
            normal.length_squared() > EPS,
            "face {face_number} is degenerate (collinear vertices)"
        );
    }

    // Invariant 1 & 2 combined: every input point is on or inside the hull,
    // which for a convex polytope is exactly "on or below every face plane".
    for (index, &point) in points.iter().enumerate() {
        for (face_number, face) in topo.faces.iter().enumerate() {
            let side = orient(points[face.a], points[face.b], points[face.c], point);
            assert!(
                side >= -EPS,
                "input point {index} is outside face {face_number}: orient={side}"
            );
        }
    }
}

/// Deterministic, dependency-free PRNG (SplitMix64) so the random clouds are
/// reproducible without pulling in `rand`.
struct SplitMix64(u64);

impl SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform f64 in [-1, 1).
    fn next_unit(&mut self) -> f64 {
        let bits = self.next_u64() >> 11; // 53-bit mantissa
        let unit = (bits as f64) / ((1u64 << 53) as f64); // [0, 1)
        unit * 2.0 - 1.0
    }

    fn point(&mut self) -> DVec3 {
        DVec3::new(self.next_unit(), self.next_unit(), self.next_unit())
    }
}

#[test]
fn tetrahedron_is_its_own_hull() {
    let points = vec![
        DVec3::new(0.0, 0.0, 0.0),
        DVec3::new(1.0, 0.0, 0.0),
        DVec3::new(0.0, 1.0, 0.0),
        DVec3::new(0.0, 0.0, 1.0),
    ];
    let topo = convex_hull(&points).expect("tetrahedron has a hull");
    assert_eq!(topo.vertices, vec![0, 1, 2, 3]);
    assert_eq!(topo.faces.len(), 4);
    assert_hull_invariants(&points, &topo);
}

#[test]
fn interior_point_is_excluded() {
    // A regular-ish tetrahedron plus its centroid; the centroid must not appear
    // on the hull.
    let mut points = vec![
        DVec3::new(1.0, 1.0, 1.0),
        DVec3::new(1.0, -1.0, -1.0),
        DVec3::new(-1.0, 1.0, -1.0),
        DVec3::new(-1.0, -1.0, 1.0),
    ];
    let centroid = points.iter().sum::<DVec3>() / points.len() as f64;
    points.push(centroid);
    let interior_index = points.len() - 1;

    let topo = convex_hull(&points).expect("non-degenerate cloud has a hull");
    assert!(
        topo.vertices.binary_search(&interior_index).is_err(),
        "interior centroid leaked onto the hull"
    );
    assert_eq!(topo.faces.len(), 4);
    assert_hull_invariants(&points, &topo);
}

#[test]
fn octahedron_general_position() {
    // Six axis vertices, jittered off the axes so no four points are coplanar
    // (keeps it strictly general-position for the first cut). Plus an interior
    // point that must be culled.
    let points = vec![
        DVec3::new(1.0, 0.01, -0.02),
        DVec3::new(-1.0, -0.02, 0.01),
        DVec3::new(0.015, 1.0, 0.02),
        DVec3::new(-0.01, -1.0, -0.015),
        DVec3::new(0.02, -0.015, 1.0),
        DVec3::new(-0.018, 0.012, -1.0),
        DVec3::new(0.0, 0.0, 0.0),
    ];
    let topo = convex_hull(&points).expect("octahedron has a hull");
    // The interior origin must be culled; the six extreme points remain.
    assert_eq!(topo.vertices.len(), 6, "expected six hull vertices");
    assert!(topo.vertices.binary_search(&6).is_err());
    assert_hull_invariants(&points, &topo);
}

#[test]
fn random_clouds_satisfy_invariants() {
    let mut rng = SplitMix64(0xDEAD_BEEF_CAFE_F00D);
    for trial in 0..64 {
        let count = 8 + (rng.next_u64() % 40) as usize; // 8..=47 points
        let points: Vec<DVec3> = (0..count).map(|_| rng.point()).collect();

        let topo = Quickhull
            .convex_hull(&points)
            .unwrap_or_else(|| panic!("trial {trial}: random cloud produced no hull"));
        assert_hull_invariants(&points, &topo);
    }
}

#[test]
fn random_clouds_on_a_sphere() {
    // Points on a sphere are all extreme, so every input must end up on the
    // hull — a strong stress for the visible-face/horizon stitch.
    let mut rng = SplitMix64(0x0123_4567_89AB_CDEF);
    for trial in 0..32 {
        let count = 8 + (rng.next_u64() % 24) as usize;
        let points: Vec<DVec3> = (0..count)
            .map(|_| rng.point().normalize_or_zero())
            .filter(|p| p.length_squared() > 0.5)
            .collect();
        if points.len() < 4 {
            continue;
        }

        let topo = convex_hull(&points)
            .unwrap_or_else(|| panic!("trial {trial}: sphere cloud produced no hull"));
        assert_hull_invariants(&points, &topo);
        assert_eq!(
            topo.vertices.len(),
            points.len(),
            "trial {trial}: every point on a sphere should be on the hull"
        );
    }
}

#[test]
fn fewer_than_four_points_has_no_hull() {
    let points = vec![
        DVec3::new(0.0, 0.0, 0.0),
        DVec3::new(1.0, 0.0, 0.0),
        DVec3::new(0.0, 1.0, 0.0),
    ];
    assert!(convex_hull(&points).is_none());
}
