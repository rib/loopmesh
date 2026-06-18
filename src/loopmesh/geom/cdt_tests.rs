use super::cdt::{triangulate, CdtInput, CdtProvider, CdtTriangulation, SpadeCdt};
use glam::DVec2;
use std::collections::HashSet;

fn boundary_only(points: Vec<DVec2>) -> CdtInput {
    let boundary = (0..points.len()).collect();
    CdtInput {
        points,
        boundary,
        holes: Vec::new(),
        edges: Vec::new(),
    }
}

/// Set of undirected edges present in the output, as sorted index pairs.
fn output_edges(result: &CdtTriangulation) -> HashSet<(usize, usize)> {
    let mut edges = HashSet::new();
    for triangle in &result.triangles {
        let [a, b, c] = triangle.vertices;
        for (u, v) in [(a, b), (b, c), (c, a)] {
            edges.insert((u.min(v), u.max(v)));
        }
    }
    edges
}

fn constraint_present(result: &CdtTriangulation, from: usize, to: usize) -> bool {
    output_edges(result).contains(&(from.min(to), from.max(to)))
}

/// Total area of the output triangles (input-indexed, |signed area| summed).
fn total_area(points: &[DVec2], result: &CdtTriangulation) -> f64 {
    result
        .triangles
        .iter()
        .map(|triangle| {
            let [a, b, c] = triangle.vertices;
            let pa = points[a];
            let pb = points[b];
            let pc = points[c];
            ((pb - pa).perp_dot(pc - pa)).abs() * 0.5
        })
        .sum()
}

fn point_in_triangle(points: &[DVec2], triangle: [usize; 3], probe: DVec2) -> bool {
    let [a, b, c] = triangle;
    let pa = points[a];
    let pb = points[b];
    let pc = points[c];
    let d1 = (probe - pa).perp_dot(pb - pa);
    let d2 = (probe - pb).perp_dot(pc - pb);
    let d3 = (probe - pc).perp_dot(pa - pc);
    let has_neg = d1 < 0.0 || d2 < 0.0 || d3 < 0.0;
    let has_pos = d1 > 0.0 || d2 > 0.0 || d3 > 0.0;
    !(has_neg && has_pos)
}

#[test]
fn convex_square_triangulates_to_two_triangles() {
    let points = vec![
        DVec2::new(0.0, 0.0),
        DVec2::new(2.0, 0.0),
        DVec2::new(2.0, 2.0),
        DVec2::new(0.0, 2.0),
    ];
    let input = boundary_only(points.clone());
    let result = triangulate(&input).expect("square triangulates");

    assert_eq!(result.triangles.len(), 2);
    assert_eq!(result.vertices, vec![0, 1, 2, 3]);

    // Every boundary constraint edge survives.
    for (from, to) in [(0, 1), (1, 2), (2, 3), (3, 0)] {
        assert!(
            constraint_present(&result, from, to),
            "boundary edge {from}-{to} missing"
        );
    }

    // Full coverage, no drift.
    assert!((total_area(&points, &result) - 4.0).abs() < 1e-9);
}

#[test]
fn forced_diagonal_constraint_appears() {
    // A square whose diagonal 0-2 is forced as a free constraint edge. The
    // unconstrained Delaunay diagonal for a square is ambiguous; forcing it
    // pins the result.
    let points = vec![
        DVec2::new(0.0, 0.0),
        DVec2::new(4.0, 0.0),
        DVec2::new(4.0, 4.0),
        DVec2::new(0.0, 4.0),
    ];
    let mut input = boundary_only(points.clone());
    input.edges.push([0, 2]);

    let result = triangulate(&input).expect("constrained square triangulates");
    assert!(
        constraint_present(&result, 0, 2),
        "forced diagonal 0-2 not present"
    );
    assert!((total_area(&points, &result) - 16.0).abs() < 1e-9);
}

#[test]
fn polygon_with_hole_excludes_hole_interior() {
    // 6x6 outer square, 2x2 hole centred at (3,3).
    let points = vec![
        // outer 0..4
        DVec2::new(0.0, 0.0),
        DVec2::new(6.0, 0.0),
        DVec2::new(6.0, 6.0),
        DVec2::new(0.0, 6.0),
        // hole 4..8
        DVec2::new(2.0, 2.0),
        DVec2::new(4.0, 2.0),
        DVec2::new(4.0, 4.0),
        DVec2::new(2.0, 4.0),
    ];
    let input = CdtInput {
        points: points.clone(),
        boundary: vec![0, 1, 2, 3],
        holes: vec![vec![4, 5, 6, 7]],
        edges: Vec::new(),
    };
    let result = triangulate(&input).expect("holed polygon triangulates");

    // Area = outer (36) minus hole (4) = 32, proving the hole is carved out
    // with no gaps elsewhere.
    assert!(
        (total_area(&points, &result) - 32.0).abs() < 1e-9,
        "area {} != 32",
        total_area(&points, &result)
    );

    // No triangle covers the hole centre.
    let centre = DVec2::new(3.0, 3.0);
    for triangle in &result.triangles {
        assert!(
            !point_in_triangle(&points, triangle.vertices, centre),
            "a triangle covers the hole interior"
        );
    }

    // Every hole-boundary constraint survives.
    for (from, to) in [(4, 5), (5, 6), (6, 7), (7, 4)] {
        assert!(
            constraint_present(&result, from, to),
            "hole edge {from}-{to} missing"
        );
    }

    // No positional drift: every output index is an input index.
    for &index in &result.vertices {
        assert!(index < points.len());
    }
}

#[test]
fn nonconvex_polygon_covered_without_overflow() {
    // An L-shaped (non-convex) hexagon. Its area is 12 - the convex hull area
    // would be 16, so a correct CDT must exclude the reflex notch.
    let points = vec![
        DVec2::new(0.0, 0.0),
        DVec2::new(4.0, 0.0),
        DVec2::new(4.0, 2.0),
        DVec2::new(2.0, 2.0),
        DVec2::new(2.0, 4.0),
        DVec2::new(0.0, 4.0),
    ];
    let input = boundary_only(points.clone());
    let result = triangulate(&input).expect("L-shape triangulates");

    assert!(
        (total_area(&points, &result) - 12.0).abs() < 1e-9,
        "area {} != 12 (notch not excluded?)",
        total_area(&points, &result)
    );

    // The reflex corner (3,3) lies outside the L; no triangle should cover it.
    let notch = DVec2::new(3.0, 3.0);
    for triangle in &result.triangles {
        assert!(!point_in_triangle(&points, triangle.vertices, notch));
    }

    for (from, to) in [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)] {
        assert!(constraint_present(&result, from, to));
    }
}

#[test]
fn degenerate_inputs_return_none() {
    // Fewer than three points.
    let two = boundary_only(vec![DVec2::new(0.0, 0.0), DVec2::new(1.0, 0.0)]);
    assert!(triangulate(&two).is_none());

    let one = boundary_only(vec![DVec2::new(0.0, 0.0)]);
    assert!(triangulate(&one).is_none());

    let none = boundary_only(Vec::new());
    assert!(triangulate(&none).is_none());

    // Three collinear points have no positive-area triangulation.
    let collinear = boundary_only(vec![
        DVec2::new(0.0, 0.0),
        DVec2::new(1.0, 0.0),
        DVec2::new(2.0, 0.0),
    ]);
    assert!(
        triangulate(&collinear).is_none(),
        "collinear cloud must yield no triangulation"
    );
}

#[test]
fn self_crossing_hole_loop_returns_none() {
    // A square outer boundary plus a "bowtie" hole loop whose two diagonals
    // cross each other: visiting 4 -> 5 -> 6 -> 7 in this order makes edges
    // 4-5 and 6-7 the crossing diagonals of the bowtie. The hole loop is a
    // self-intersecting constraint, which must fail cleanly rather than panic.
    let points = vec![
        DVec2::new(0.0, 0.0),
        DVec2::new(6.0, 0.0),
        DVec2::new(6.0, 6.0),
        DVec2::new(0.0, 6.0),
        // bowtie corners
        DVec2::new(2.0, 2.0),
        DVec2::new(4.0, 4.0),
        DVec2::new(4.0, 2.0),
        DVec2::new(2.0, 4.0),
    ];
    let input = CdtInput {
        points,
        boundary: vec![0, 1, 2, 3],
        holes: vec![vec![4, 5, 6, 7]],
        edges: Vec::new(),
    };
    assert!(
        triangulate(&input).is_none(),
        "self-crossing hole loop must yield None, not panic"
    );
}

#[test]
fn crossing_free_constraints_return_none() {
    // Two free constraint edges across a square that cross each other in the
    // interior (0-2 and 1-3 are the two diagonals). The second one cannot be
    // inserted without intersecting the first constraint, so the whole
    // triangulation fails cleanly.
    let points = vec![
        DVec2::new(0.0, 0.0),
        DVec2::new(4.0, 0.0),
        DVec2::new(4.0, 4.0),
        DVec2::new(0.0, 4.0),
    ];
    let mut input = boundary_only(points);
    input.edges.push([0, 2]);
    input.edges.push([1, 3]);
    assert!(
        triangulate(&input).is_none(),
        "crossing free constraints must yield None, not panic"
    );
}

#[test]
fn two_disjoint_holes_both_excluded() {
    // 10x6 outer rectangle with two separate 2x2 holes: one on the left,
    // centred at (2.5, 3), one on the right, centred at (7.5, 3). The
    // flood-fill's parity logic has to flip in and back out twice, once per
    // hole, independently — the single-hole test cannot exercise that.
    let points = vec![
        // outer 0..4
        DVec2::new(0.0, 0.0),
        DVec2::new(10.0, 0.0),
        DVec2::new(10.0, 6.0),
        DVec2::new(0.0, 6.0),
        // left hole 4..8, centred at (2.5, 3)
        DVec2::new(1.5, 2.0),
        DVec2::new(3.5, 2.0),
        DVec2::new(3.5, 4.0),
        DVec2::new(1.5, 4.0),
        // right hole 8..12, centred at (7.5, 3)
        DVec2::new(6.5, 2.0),
        DVec2::new(8.5, 2.0),
        DVec2::new(8.5, 4.0),
        DVec2::new(6.5, 4.0),
    ];
    let input = CdtInput {
        points: points.clone(),
        boundary: vec![0, 1, 2, 3],
        holes: vec![vec![4, 5, 6, 7], vec![8, 9, 10, 11]],
        edges: Vec::new(),
    };
    let result = triangulate(&input).expect("two-holed polygon triangulates");

    // Area = outer (60) minus two 2x2 holes (4 each) = 52. A parity bug that
    // fails to re-enter "outside" after the first hole would mis-fill the
    // second, throwing this off.
    let area = total_area(&points, &result);
    assert!((area - 52.0).abs() < 1e-9, "area {area} != 52");

    // A probe inside each hole hits no triangle.
    let left_centre = DVec2::new(2.5, 3.0);
    let right_centre = DVec2::new(7.5, 3.0);
    for triangle in &result.triangles {
        assert!(
            !point_in_triangle(&points, triangle.vertices, left_centre),
            "a triangle covers the left hole interior"
        );
        assert!(
            !point_in_triangle(&points, triangle.vertices, right_centre),
            "a triangle covers the right hole interior"
        );
    }

    // Both holes' boundary edges survive as constraints.
    for (from, to) in [(4, 5), (5, 6), (6, 7), (7, 4)] {
        assert!(
            constraint_present(&result, from, to),
            "left hole edge {from}-{to} missing"
        );
    }
    for (from, to) in [(8, 9), (9, 10), (10, 11), (11, 8)] {
        assert!(
            constraint_present(&result, from, to),
            "right hole edge {from}-{to} missing"
        );
    }
}

#[test]
fn edge_net_with_no_boundary_carves_no_region() {
    // A free edge-net with no enclosing boundary loop. Per the input contract,
    // free edges are forced into the triangulation but "do not by themselves
    // bound a region". With nothing to mark interior, the parity flood-fill
    // leaves every face outside, so no faces are emitted and `triangulate`
    // reports no region (`None`) rather than dumping the unbounded convex hull.
    // Fill operators always supply a boundary; an edge-net alone bounds nothing.
    let points = vec![
        DVec2::new(0.0, 0.0),
        DVec2::new(4.0, 0.0),
        DVec2::new(4.0, 4.0),
        DVec2::new(0.0, 4.0),
    ];
    let input = CdtInput {
        points,
        boundary: Vec::new(),
        holes: Vec::new(),
        edges: vec![[0, 2]],
    };
    assert!(
        triangulate(&input).is_none(),
        "an edge-net with no boundary bounds no region, so triangulate returns None"
    );
}

#[test]
fn coincident_input_points_collapse_cleanly() {
    // A unit-square boundary whose corner (4,0) is supplied twice: index 1 and
    // index 4 share a position. spade collapses them onto one handle. The
    // triangulation must still cover the square, never reference the same input
    // index twice in a triangle, never reference both collapsed indices in the
    // same triangle, and never panic (exercising the degenerate-triple skip).
    let points = vec![
        DVec2::new(0.0, 0.0),
        DVec2::new(4.0, 0.0),
        DVec2::new(4.0, 4.0),
        DVec2::new(0.0, 4.0),
        DVec2::new(4.0, 0.0), // duplicate position of index 1
    ];
    let input = CdtInput {
        points: points.clone(),
        boundary: vec![0, 1, 2, 3],
        holes: Vec::new(),
        edges: Vec::new(),
    };
    let result = triangulate(&input).expect("coincident-cornered square triangulates");

    // Still a full, gap-free covering of the 4x4 square.
    let area = total_area(&points, &result);
    assert!((area - 16.0).abs() < 1e-9, "area {area} != 16");

    // No triangle references one input index twice, and none references both
    // collapsed indices (1 and 4) — they are the same point, which would be a
    // zero-width triangle.
    for triangle in &result.triangles {
        let [a, b, c] = triangle.vertices;
        assert!(a != b && b != c && a != c, "degenerate triple {a},{b},{c}");
        let touches_one = a == 1 || b == 1 || c == 1;
        let touches_four = a == 4 || b == 4 || c == 4;
        assert!(
            !(touches_one && touches_four),
            "triangle {a},{b},{c} spans both coincident corners"
        );
    }

    // Output vertex set: ascending, all valid input indices, and the corner is
    // represented by exactly one of the two coincident indices, not both.
    assert!(result.vertices.windows(2).all(|w| w[0] < w[1]));
    for &index in &result.vertices {
        assert!(index < points.len());
    }
    let has_one = result.vertices.contains(&1);
    let has_four = result.vertices.contains(&4);
    assert!(
        has_one ^ has_four,
        "exactly one of the coincident corner indices should survive (1={has_one}, 4={has_four})"
    );
}

#[test]
fn provider_default_matches_free_function() {
    let points = vec![
        DVec2::new(0.0, 0.0),
        DVec2::new(2.0, 0.0),
        DVec2::new(1.0, 2.0),
    ];
    let input = boundary_only(points);
    let direct = triangulate(&input).unwrap();
    let via_provider = SpadeCdt.triangulate(&input).unwrap();
    assert_eq!(direct.triangles.len(), via_provider.triangles.len());
    assert_eq!(direct.vertices, via_provider.vertices);
}
