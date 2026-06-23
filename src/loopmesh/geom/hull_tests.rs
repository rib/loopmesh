//! Tier-2 property tests for the in-tree quickhull.
//!
//! No oracle: a hull is self-checking. We assert the invariants the
//! prerequisite file lists —
//!   1. every input point is on or inside the hull;
//!   2. the hull is globally convex (every input below or on every face plane);
//!   3. every face is outward-wound, planar, and convex (faces are now n-gons,
//!      so planarity and convexity are explicit checks, not free triangle
//!      properties);
//!   4. coplanar input points merge into a single face.
//! Exact orientation in the assertions uses the same `robust::orient3d`
//! predicate the implementation does, so "on or inside" is decided exactly.

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

/// Assert the hull invariants on a full-dimensional `topo` over `points`.
///
/// Faces are n-gons, so planarity and convexity are explicit. Each face's plane
/// is taken from its first three vertices; planarity asserts every further face
/// vertex lies exactly in that plane, and global convexity asserts every input
/// point is on or inside every face plane.
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
        assert!(face.vertices.len() >= 3, "face has fewer than three vertices");
        for &index in &face.vertices {
            assert!(index < points.len(), "face references out-of-range index");
            assert!(
                topo.vertices.binary_search(&index).is_ok(),
                "face index {index} missing from hull vertex set"
            );
        }
        // No repeated vertex within a single face.
        let mut sorted = face.vertices.clone();
        sorted.sort_unstable();
        let len = sorted.len();
        sorted.dedup();
        assert_eq!(len, sorted.len(), "face has a repeated vertex");
    }

    for (face_number, face) in topo.faces.iter().enumerate() {
        let a = points[face.vertices[0]];
        let b = points[face.vertices[1]];
        let c = points[face.vertices[2]];

        // Non-degenerate seed triangle (the first three face vertices are not
        // collinear).
        let normal = (b - a).cross(c - a);
        assert!(
            normal.length_squared() > EPS,
            "face {face_number} has a collinear seed triangle"
        );

        // Planarity: every further vertex of the face lies exactly in the plane.
        for &index in &face.vertices[3..] {
            let side = orient(a, b, c, points[index]);
            assert_eq!(
                side, 0.0,
                "face {face_number} is non-planar at vertex {index}"
            );
        }

        // Convexity: walking the boundary, every turn bends the same way. The
        // cross product of consecutive edges must stay parallel (not anti-
        // parallel) to the face normal at every corner.
        let count = face.vertices.len();
        for corner in 0..count {
            let prev = points[face.vertices[(corner + count - 1) % count]];
            let here = points[face.vertices[corner]];
            let nextp = points[face.vertices[(corner + 1) % count]];
            let turn = (here - prev).cross(nextp - here);
            assert!(
                turn.dot(normal) >= -EPS,
                "face {face_number} is non-convex / mis-wound at corner {corner}"
            );
        }

        for (index, &point) in points.iter().enumerate() {
            if face.vertices.contains(&index) {
                continue;
            }
            let side = orient(a, b, c, point);
            assert!(
                side >= -EPS,
                "point {index} lies outside face {face_number}: orient={side}"
            );
        }
    }

    // Invariant 1 & 2 combined: every input point is on or inside the hull,
    // which for a convex polytope is exactly "on or below every face plane".
    for (index, &point) in points.iter().enumerate() {
        for (face_number, face) in topo.faces.iter().enumerate() {
            let a = points[face.vertices[0]];
            let b = points[face.vertices[1]];
            let c = points[face.vertices[2]];
            let side = orient(a, b, c, point);
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
fn lower_dimensional_clouds_resolve_to_degenerate_hulls() {
    // Three non-collinear coplanar points are a genuine (flat) hull: one
    // triangular face over the three boundary points. The pre-merge first cut
    // returned `None` here; the degenerate handling now resolves it.
    let triangle = vec![
        DVec3::new(0.0, 0.0, 0.0),
        DVec3::new(1.0, 0.0, 0.0),
        DVec3::new(0.0, 1.0, 0.0),
    ];
    let topo = convex_hull(&triangle).expect("coplanar triangle is a flat hull");
    assert_eq!(topo.vertices, vec![0, 1, 2]);
    assert_eq!(topo.faces.len(), 1, "flat triangle hull is one face");
    assert_eq!(topo.faces[0].vertices.len(), 3);

    // A hull only fails to exist when fewer than two distinct points remain.
    let single = vec![DVec3::new(1.0, 1.0, 1.0), DVec3::new(1.0, 1.0, 1.0)];
    assert!(convex_hull(&single).is_none(), "one distinct point has no hull");
    assert!(convex_hull(&[]).is_none(), "empty cloud has no hull");
}

#[test]
fn coincident_points_are_deduplicated() {
    // A tetrahedron whose apex is duplicated three times. The duplicates are
    // coincident with a true hull vertex, so a correct hull lists that corner
    // exactly once. The first cut already culls strict duplicates here (they sit
    // on every incident face plane), so this dedup invariant holds today — it is
    // a real, currently-passing test, not a row-5 target. (Full dedup across all
    // degenerate inputs — e.g. coincident points that are not all extreme — is
    // still row-5 work; this fixture's extreme-corner case is covered now.)
    let points = vec![
        DVec3::new(0.0, 0.0, 0.0),
        DVec3::new(1.0, 0.0, 0.0),
        DVec3::new(0.0, 1.0, 0.0),
        DVec3::new(0.0, 0.0, 1.0),
        DVec3::new(0.0, 0.0, 1.0), // dup of index 3
        DVec3::new(0.0, 0.0, 1.0), // dup of index 3
    ];
    let topo = convex_hull(&points).expect("tetra-with-duplicates has a hull");

    assert!(
        hull_positions_are_distinct(&points, &topo),
        "coincident points were not deduplicated on the hull"
    );
    // Exactly one of the three coincident apex indices represents the corner;
    // a deduped hull has four distinct hull vertices, not six.
    assert_eq!(
        topo.vertices.len(),
        4,
        "deduped tetra hull must have four distinct vertices, got {}",
        topo.vertices.len()
    );
    assert_hull_invariants(&points, &topo);
}

// ---------------------------------------------------------------------------
// Degenerate-input fixtures (decomposition row 4).
//
// These encode the *target* post-merge / post-dedup invariants the convex_hull
// contract requires (prerequisites/convex_hull.md row 5: coplanar-face merge +
// degenerate combinatorics). The first cut is general-position only, so each of
// these fails today — `#[ignore]` keeps them out of routine runs while making
// the gap a runnable record. They run under
// `cargo test -p loopmesh --lib -- --ignored`.
// ---------------------------------------------------------------------------

/// True when no two on-hull vertices share a position (the dedup invariant): a
/// correct hull never lists two coincident input points as distinct hull verts.
fn hull_positions_are_distinct(points: &[DVec3], topo: &HullTopology) -> bool {
    for (i, &vi) in topo.vertices.iter().enumerate() {
        for &vj in &topo.vertices[i + 1..] {
            if points[vi] == points[vj] {
                return false;
            }
        }
    }
    true
}

#[test]
fn cube_merges_coplanar_faces() {
    // Eight corners of the unit cube. Every cube face is four coplanar points,
    // so the operator contract is six merged quad faces — NOT a 12-triangle fan.
    // The general-position first cut triangulates each square into two triangles
    // (12 faces), so the face-count assertion fails until coplanar merge lands.
    let points = vec![
        DVec3::new(0.0, 0.0, 0.0),
        DVec3::new(1.0, 0.0, 0.0),
        DVec3::new(1.0, 1.0, 0.0),
        DVec3::new(0.0, 1.0, 0.0),
        DVec3::new(0.0, 0.0, 1.0),
        DVec3::new(1.0, 0.0, 1.0),
        DVec3::new(1.0, 1.0, 1.0),
        DVec3::new(0.0, 1.0, 1.0),
    ];
    let topo = convex_hull(&points).expect("cube has a hull");

    // Every corner is extreme → all eight inputs must be on the hull.
    assert_eq!(topo.vertices, (0..8).collect::<Vec<_>>(), "every cube corner is on-hull");

    // Target contract: six coplanar-merged faces, not a triangle fan.
    assert_eq!(
        topo.faces.len(),
        6,
        "cube must merge to six coplanar faces, got {} (triangle fan?)",
        topo.faces.len()
    );

    assert_hull_invariants(&points, &topo);
}

#[test]
fn coplanar_slab_yields_flat_hull() {
    // A single-plane point cloud (z == 0): a 3x3 grid plus interior jitter. No
    // positive-volume seed simplex exists, so the first cut returns None. The
    // target contract is a degenerate (flat) hull: a single planar polygon over
    // the cloud's 2D convex boundary (the four grid corners), interior points
    // culled. We assert that target; it fails today on the None.
    let points = vec![
        DVec3::new(0.0, 0.0, 0.0),
        DVec3::new(2.0, 0.0, 0.0),
        DVec3::new(2.0, 2.0, 0.0),
        DVec3::new(0.0, 2.0, 0.0),
        DVec3::new(1.0, 0.0, 0.0),
        DVec3::new(1.0, 2.0, 0.0),
        DVec3::new(1.0, 1.0, 0.0), // interior → culled
    ];
    let topo = convex_hull(&points).expect("coplanar slab must produce a flat hull");

    // The four corners bound the planar region; the interior centre is culled.
    assert!(
        topo.vertices.binary_search(&6).is_err(),
        "interior coplanar point leaked onto the flat hull"
    );
    for corner in [0usize, 1, 2, 3] {
        assert!(
            topo.vertices.binary_search(&corner).is_ok(),
            "boundary corner {corner} missing from the flat hull"
        );
    }
    assert!(hull_positions_are_distinct(&points, &topo));
}

#[test]
fn collinear_cloud_yields_degenerate_hull() {
    // All points on a single line: no area, no volume. The first cut returns
    // None (no seed simplex). The target contract reduces this to the two
    // extreme endpoints of the segment — a 1D degenerate hull whose on-hull set
    // is exactly those two extremes, interior collinear points culled.
    let points = vec![
        DVec3::new(0.0, 0.0, 0.0),
        DVec3::new(1.0, 1.0, 1.0),
        DVec3::new(2.0, 2.0, 2.0),
        DVec3::new(3.0, 3.0, 3.0),
        DVec3::new(0.5, 0.5, 0.5), // interior on the segment → culled
    ];
    let topo = convex_hull(&points).expect("collinear cloud must produce a degenerate hull");

    // Only the two endpoints are extreme.
    assert_eq!(
        topo.vertices,
        vec![0, 3],
        "collinear hull must reduce to the two segment endpoints"
    );
    assert!(hull_positions_are_distinct(&points, &topo));
}

#[test]
fn cospherical_with_coplanar_subsets_merges_faces() {
    // A cospherical octagonal pyramid: eight base points around a circle in the
    // exact plane z = z0, plus an apex at the north pole. Every point is extreme
    // (all radius 1 → on the unit sphere), so all nine are on the hull.
    //
    // The merge target here is genuine, not incidental: the eight base points lie
    // in the supporting plane z = z0 (the apex is the *only* point above it), so
    // `{0,1,..,7}` is a true hull facet — an octagon. A correct merged hull emits
    // that octagon as ONE face plus the eight triangular side faces to the apex:
    // nine faces total. The general-position first cut instead fans the octagon
    // into six triangles (eight base + apex = 14 triangles overall), so the
    // merged-vs-triangulated face count is the load-bearing assertion. The
    // triangulated count (14) equals the Euler bound 2V-4 exactly, and the merged
    // count (9) sits a comfortable five faces below it — no incidental-quad
    // margin.
    let z0 = -0.5;
    let rho = (1.0_f64 - z0 * z0).sqrt(); // base circle radius so every point is on the unit sphere
    let mut points: Vec<DVec3> = (0..8)
        .map(|k| {
            let angle = std::f64::consts::TAU * (k as f64) / 8.0;
            DVec3::new(rho * angle.cos(), rho * angle.sin(), z0)
        })
        .collect();
    points.push(DVec3::new(0.0, 0.0, 1.0)); // apex at the north pole

    let topo = convex_hull(&points).expect("cospherical cloud has a hull");

    // Every point lies on the sphere → every input must be on the hull.
    assert_eq!(
        topo.vertices.len(),
        points.len(),
        "every cospherical point must be on the hull"
    );

    // The eight base points (indices 0..=7) lie in the supporting plane z = z0
    // and must merge into a single octagonal face, not a six-triangle fan. A
    // correctly merged hull therefore emits strictly fewer faces than the
    // all-triangle count (9 merged vs 14 triangulated = 2V-4). The first cut
    // triangulates the octagon, hitting the bound, so this fails until the merge
    // lands.
    //
    let triangulated_upper_bound = 2 * topo.vertices.len() - 4; // Euler bound for a triangulated hull
    assert!(
        topo.faces.len() < triangulated_upper_bound,
        "coplanar octagon must merge: got {} faces, a merged hull is strictly \
         below the triangulated bound {triangulated_upper_bound}",
        topo.faces.len()
    );

    // Concretely: the base merges to exactly one octagonal face, plus eight
    // triangular side faces to the apex (nine faces total).
    assert_eq!(topo.faces.len(), 9, "expected one octagon + eight side triangles");
    let octagons = topo
        .faces
        .iter()
        .filter(|face| face.vertices.len() == 8)
        .count();
    assert_eq!(octagons, 1, "the eight coplanar base points must form one octagon");

    assert_hull_invariants(&points, &topo);
}

/// Each emitted facet must be anchored on its smallest vertex index — the
/// hash-independent anchor the deterministic merge pass guarantees (the boundary
/// walk starts at the minimum boundary index). A floating anchor would leave the
/// facet winding at the mercy of the per-process hash seed, the root of the
/// convex_hull reuse-path nondeterminism, and would let a downstream
/// find-or-create miss an existing face.
///
/// We also assert the full emission is stable run-to-run by recomputing the hull
/// and comparing the two outputs structurally — within a process the hash seed
/// is fixed, so this catches any *intra*-run reordering and pins the output as a
/// pure function of the input.
fn assert_facet_emission_is_deterministic(points: &[DVec3], topo: &HullTopology) {
    for face in &topo.faces {
        let min_index = *face.vertices.iter().min().expect("facet has vertices");
        assert_eq!(
            face.vertices[0], min_index,
            "facet must be anchored on its smallest vertex index for a stable winding"
        );
    }

    let again = convex_hull(points).expect("hull recomputes");
    assert_eq!(again.vertices, topo.vertices, "on-hull set is a pure function of input");
    assert_eq!(
        again.faces.len(),
        topo.faces.len(),
        "facet count is a pure function of input"
    );
    for (lhs, rhs) in topo.faces.iter().zip(&again.faces) {
        assert_eq!(lhs.vertices, rhs.vertices, "each facet re-emits identically");
    }
}

#[test]
fn cube_facet_emission_is_deterministic() {
    // The cube exercises the coplanar-merge pass (six quad facets). Pin that the
    // merge emits each facet anchored on its minimum vertex and the facets in
    // ascending-anchor order, so the downstream face find-or-create sees a stable
    // anchor regardless of the hash seed.
    let points = vec![
        DVec3::new(0.0, 0.0, 0.0),
        DVec3::new(1.0, 0.0, 0.0),
        DVec3::new(1.0, 1.0, 0.0),
        DVec3::new(0.0, 1.0, 0.0),
        DVec3::new(0.0, 0.0, 1.0),
        DVec3::new(1.0, 0.0, 1.0),
        DVec3::new(1.0, 1.0, 1.0),
        DVec3::new(0.0, 1.0, 1.0),
    ];
    let topo = convex_hull(&points).expect("cube has a hull");
    assert_eq!(topo.faces.len(), 6, "cube merges to six coplanar quads");
    assert_facet_emission_is_deterministic(&points, &topo);
    assert_hull_invariants(&points, &topo);
}

#[test]
fn bipyramid_facet_emission_is_deterministic() {
    // A triangular bipyramid: two apexes over a shared equatorial triangle, all
    // six facets genuine triangles (no coplanar merge). This is the reuse-path
    // fixture's geometry; pin its facet anchors/order are hash-independent.
    let points = vec![
        DVec3::new(1.0, 0.0, 0.0),
        DVec3::new(-0.5, 0.866_025, 0.0),
        DVec3::new(-0.5, -0.866_025, 0.0),
        DVec3::new(0.0, 0.0, 1.0),
        DVec3::new(0.0, 0.0, -1.0),
    ];
    let topo = convex_hull(&points).expect("bipyramid has a hull");
    assert_eq!(topo.faces.len(), 6, "triangular bipyramid has six triangular facets");
    for face in &topo.faces {
        assert_eq!(face.vertices.len(), 3, "every bipyramid facet is a triangle");
    }
    assert_facet_emission_is_deterministic(&points, &topo);
    assert_hull_invariants(&points, &topo);
}
