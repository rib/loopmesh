//! 3D convex hull (quickhull) — a shared comp-geom prerequisite.
//!
//! This is a self-contained primitive: points in, hull topology out. It does
//! not touch `LoopMesh` topology; the `convex_hull` operator will translate the
//! neutral index-based output into radial-edge surgery on top of this.
//!
//! Exact orientation is delegated to the `robust` crate's adaptive-precision
//! `orient3d` predicate (Shewchuk's algorithm). We never hand-roll the
//! predicate: only the combinatorial quickhull skeleton lives here.
//!
//! This first cut is **general-position correct**. Degenerate combinatorics
//! (coincident / collinear / coplanar / cospherical inputs) are deferred; see
//! `bmesh-parity/prerequisites/convex_hull.md`.

use glam::DVec3;
use robust::{orient3d, Coord3D};

/// A hull face as a triangle of indices into the provider's input point slice.
///
/// Indices are **into the input cloud** (not a remapped hull-vertex set): the
/// output is neutral and index-based so any consumer can map them back to its
/// own vertex keys without positional drift. The winding is counter-clockwise
/// as seen from outside the hull (outward-facing normal by the right-hand
/// rule).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HullFace {
    pub a: usize,
    pub b: usize,
    pub c: usize,
}

/// The topology of a convex hull: a set of outward-wound triangular faces over
/// the input point indices, plus the set of input indices that lie on the hull.
///
/// `vertices` is the on-hull subset of the input indices, in ascending order.
/// `faces` reference the original input indices (so they index the caller's
/// cloud directly).
#[derive(Clone, Debug, Default)]
pub struct HullTopology {
    pub vertices: Vec<usize>,
    pub faces: Vec<HullFace>,
}

/// A provider that turns a point cloud into hull topology.
///
/// The boundary is deliberately coarse — one call computes a whole hull — so
/// that an injected `&dyn ConvexHullProvider` costs a single virtual dispatch
/// per hull. The production path uses the zero-sized [`Quickhull`] default via
/// static dispatch, so the abstraction inlines to a direct call and taxes the
/// loopmesh-native path nothing.
pub trait ConvexHullProvider {
    /// Compute the convex hull of `points`.
    ///
    /// Returns `None` when no full-dimensional hull exists for this provider
    /// (e.g. fewer than four points, or — for this first cut — a degenerate
    /// lower-dimensional input the provider cannot resolve).
    fn convex_hull(&self, points: &[DVec3]) -> Option<HullTopology>;
}

/// The in-tree default convex-hull provider: a quickhull over `robust`'s
/// `orient3d`. Zero-sized, so the trait abstraction is free on the production
/// path.
#[derive(Clone, Copy, Debug, Default)]
pub struct Quickhull;

impl ConvexHullProvider for Quickhull {
    #[inline]
    fn convex_hull(&self, points: &[DVec3]) -> Option<HullTopology> {
        convex_hull(points)
    }
}

#[inline]
fn coord(point: DVec3) -> Coord3D<f64> {
    Coord3D {
        x: point.x,
        y: point.y,
        z: point.z,
    }
}

/// Signed orientation of `d` against the oriented plane through `a, b, c`.
///
/// Positive when `d` is on the side from which `a, b, c` appear
/// counter-clockwise (the "below" side); negative on the other side; zero when
/// coplanar. We treat a triangle `(a, b, c)` of a hull face as outward-wound
/// when every other hull point is strictly on the positive side.
#[inline]
fn orient(a: DVec3, b: DVec3, c: DVec3, d: DVec3) -> f64 {
    orient3d(coord(a), coord(b), coord(c), coord(d))
}

/// A face of the working hull during construction. `points` is winding order
/// (outward-facing CCW from outside).
#[derive(Clone, Copy)]
struct WorkFace {
    points: [usize; 3],
    /// `false` once the face is removed (visible to a processed point).
    alive: bool,
}

impl WorkFace {
    #[inline]
    fn edges(&self) -> [(usize, usize); 3] {
        let [a, b, c] = self.points;
        [(a, b), (b, c), (c, a)]
    }
}

/// Compute the 3D convex hull of `points` (general-position inputs).
///
/// Returns `None` when the cloud has fewer than four points or is degenerate
/// (all points coplanar / collinear / coincident, so no positive-volume seed
/// simplex exists). Handling those degeneracies as real hulls is deferred.
pub fn convex_hull(points: &[DVec3]) -> Option<HullTopology> {
    if points.len() < 4 {
        return None;
    }

    let seed = seed_simplex(points)?;
    let mut faces = seed_faces(points, seed);

    // Assign every remaining point to the first face it sits outside of. A
    // point assigned to no face is interior and never re-examined.
    let mut outside: Vec<Vec<usize>> = faces.iter().map(|_| Vec::new()).collect();
    for index in 0..points.len() {
        if seed.contains(&index) {
            continue;
        }
        assign_to_face(points, &faces, index, &mut outside);
    }

    // Process faces that still own outside points. Each iteration pops the
    // furthest such point, carves out every face it can see, and stitches the
    // horizon back to it.
    let mut face_queue: Vec<usize> = (0..faces.len()).collect();
    while let Some(face_index) = face_queue.pop() {
        if !faces[face_index].alive || outside[face_index].is_empty() {
            continue;
        }

        let apex = furthest_outside_point(points, &faces[face_index], &outside[face_index]);

        // Find every face visible from the apex (flood from the seed face
        // across shared edges) and collect their orphaned outside points.
        let visible = visible_faces(points, &faces, face_index, apex);
        let horizon = horizon_edges(&faces, &visible);

        let mut orphans: Vec<usize> = Vec::new();
        for &vf in &visible {
            faces[vf].alive = false;
            orphans.append(&mut outside[vf]);
        }

        // Stitch a new face from each horizon edge to the apex; redistribute
        // the orphaned points onto the new faces.
        for (edge_start, edge_end) in horizon {
            let new_index = faces.len();
            faces.push(WorkFace {
                points: [edge_start, edge_end, apex],
                alive: true,
            });
            outside.push(Vec::new());

            for &orphan in &orphans {
                if orphan == apex {
                    continue;
                }
                if point_outside(points, &faces[new_index], orphan) {
                    outside[new_index].push(orphan);
                }
            }
            if !outside[new_index].is_empty() {
                face_queue.push(new_index);
            }
        }
    }

    Some(finalize(&faces))
}

/// Pick four input points that form a positive-volume (non-degenerate) seed
/// tetrahedron. Returns `None` if the cloud is degenerate (coplanar etc.).
fn seed_simplex(points: &[DVec3]) -> Option<[usize; 4]> {
    let n = points.len();

    // First two distinct points span an edge.
    let p0 = 0;
    let p1 = (1..n).find(|&i| points[i] != points[p0])?;

    // Third point not collinear with the edge: maximize the cross-product
    // magnitude for numerical comfort (orient3d still decides the simplex).
    let line = points[p1] - points[p0];
    let mut p2 = None;
    let mut best_area = 0.0f64;
    for i in 0..n {
        if i == p0 || i == p1 {
            continue;
        }
        let area = line.cross(points[i] - points[p0]).length_squared();
        if area > best_area {
            best_area = area;
            p2 = Some(i);
        }
    }
    let p2 = p2?;
    if best_area == 0.0 {
        return None;
    }

    // Fourth point off the plane of the first three, chosen by the largest
    // exact orientation magnitude.
    let mut p3 = None;
    let mut best_vol = 0.0f64;
    for i in 0..n {
        if i == p0 || i == p1 || i == p2 {
            continue;
        }
        let vol = orient(points[p0], points[p1], points[p2], points[i]).abs();
        if vol > best_vol {
            best_vol = vol;
            p3 = Some(i);
        }
    }
    let p3 = p3?;
    if best_vol == 0.0 {
        return None;
    }

    Some([p0, p1, p2, p3])
}

/// Build the four outward-wound faces of the seed tetrahedron.
fn seed_faces(points: &[DVec3], seed: [usize; 4]) -> Vec<WorkFace> {
    let [a, b, c, d] = seed;
    // Orient each face so its fourth vertex is strictly "below" it (interior).
    // `orient(x, y, z, w) > 0` means w is below the plane of (x, y, z); we want
    // the opposite vertex below, so flip winding when it is above.
    let mut faces = Vec::with_capacity(4);
    for (tri, opposite) in [
        ([a, b, c], d),
        ([a, c, d], b),
        ([a, d, b], c),
        ([b, d, c], a),
    ] {
        faces.push(oriented_face(points, tri, opposite));
    }
    faces
}

/// Return `tri` (or its reverse) wound so that `opposite` lies below the face
/// plane (i.e. on the interior side, with the face normal pointing outward).
fn oriented_face(points: &[DVec3], tri: [usize; 3], opposite: usize) -> WorkFace {
    let [x, y, z] = tri;
    if orient(points[x], points[y], points[z], points[opposite]) > 0.0 {
        WorkFace {
            points: [x, y, z],
            alive: true,
        }
    } else {
        WorkFace {
            points: [x, z, y],
            alive: true,
        }
    }
}

/// True when `index` lies strictly outside `face` (on the outward side of its
/// plane). The face is outward-wound, so an outside point is "above" it:
/// `orient(a, b, c, p) < 0`.
#[inline]
fn point_outside(points: &[DVec3], face: &WorkFace, index: usize) -> bool {
    let [a, b, c] = face.points;
    orient(points[a], points[b], points[c], points[index]) < 0.0
}

fn assign_to_face(
    points: &[DVec3],
    faces: &[WorkFace],
    index: usize,
    outside: &mut [Vec<usize>],
) {
    for face_index in 0..faces.len() {
        if faces[face_index].alive && point_outside(points, &faces[face_index], index) {
            outside[face_index].push(index);
            return;
        }
    }
}

/// The outside point of `face` furthest above its plane (largest exact
/// orientation magnitude on the outward side).
fn furthest_outside_point(points: &[DVec3], face: &WorkFace, candidates: &[usize]) -> usize {
    let [a, b, c] = face.points;
    let mut best = candidates[0];
    let mut best_dist = f64::NEG_INFINITY;
    for &index in candidates {
        // Outward side is negative orient; negate so larger == further out.
        let dist = -orient(points[a], points[b], points[c], points[index]);
        if dist > best_dist {
            best_dist = dist;
            best = index;
        }
    }
    best
}

/// All alive faces visible from `apex`, found by flooding from `start` across
/// shared edges (the visible set is edge-connected on a convex hull).
fn visible_faces(
    points: &[DVec3],
    faces: &[WorkFace],
    start: usize,
    apex: usize,
) -> Vec<usize> {
    let mut visible = Vec::new();
    let mut seen = vec![false; faces.len()];
    let mut stack = vec![start];
    seen[start] = true;
    while let Some(face_index) = stack.pop() {
        if !faces[face_index].alive || !point_outside(points, &faces[face_index], apex) {
            continue;
        }
        visible.push(face_index);
        // Cross every edge to neighbouring faces.
        for (edge_start, edge_end) in faces[face_index].edges() {
            for neighbour in 0..faces.len() {
                if seen[neighbour] || !faces[neighbour].alive {
                    continue;
                }
                if shares_edge(&faces[neighbour], edge_end, edge_start) {
                    seen[neighbour] = true;
                    stack.push(neighbour);
                }
            }
        }
    }
    visible
}

/// True when `face` contains the directed edge `(u, v)` in its winding.
#[inline]
fn shares_edge(face: &WorkFace, u: usize, v: usize) -> bool {
    face.edges().iter().any(|&(s, e)| s == u && e == v)
}

/// The horizon: directed boundary edges of the visible region — edges of a
/// visible face whose opposite (reverse) edge does not belong to any other
/// visible face. Each horizon edge becomes the base of one new face.
fn horizon_edges(faces: &[WorkFace], visible: &[usize]) -> Vec<(usize, usize)> {
    let mut horizon = Vec::new();
    for &face_index in visible {
        for (edge_start, edge_end) in faces[face_index].edges() {
            let shared = visible.iter().any(|&other| {
                other != face_index && shares_edge(&faces[other], edge_end, edge_start)
            });
            if !shared {
                horizon.push((edge_start, edge_end));
            }
        }
    }
    horizon
}

fn finalize(faces: &[WorkFace]) -> HullTopology {
    let mut topo = HullTopology::default();
    let mut on_hull = vec![false; 0];
    let mut max_index = 0usize;
    for face in faces {
        if face.alive {
            max_index = max_index.max(face.points[0].max(face.points[1]).max(face.points[2]));
        }
    }
    on_hull.resize(max_index + 1, false);
    for face in faces {
        if !face.alive {
            continue;
        }
        let [a, b, c] = face.points;
        topo.faces.push(HullFace { a, b, c });
        on_hull[a] = true;
        on_hull[b] = true;
        on_hull[c] = true;
    }
    for (index, &present) in on_hull.iter().enumerate() {
        if present {
            topo.vertices.push(index);
        }
    }
    topo
}
