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
//! The hull is full-dimensional in the general case, but mesh-vertex inputs are
//! disproportionately degenerate (cubes, grids, coplanar rings, collinear
//! runs). So the output is n-gon faces with coplanar input merged into a single
//! face, and lower-dimensional clouds (coplanar / collinear) resolve to their
//! genuine degenerate hulls rather than `None`.

use glam::DVec3;
use robust::{orient3d, Coord3D};

/// A hull face as a CCW-wound loop of indices into the provider's input point
/// slice.
///
/// Indices are **into the input cloud** (not a remapped hull-vertex set): the
/// output is neutral and index-based so any consumer can map them back to its
/// own vertex keys without positional drift. The winding is counter-clockwise
/// as seen from outside the hull (outward-facing normal by the right-hand
/// rule).
///
/// A face has at least three vertices. Coplanar input points that share a
/// supporting plane are emitted as a single n-gon face, not a triangle fan, so
/// a cube hull is six quads and an octagonal-base pyramid's base is one octagon.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HullFace {
    pub vertices: Vec<usize>,
}

/// The topology of a convex hull: a set of outward-wound faces over the input
/// point indices, plus the set of input indices that lie on the hull.
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
///
/// # Contract
///
/// This list is the **whole specification** an implementor gets — a hull engine
/// wrapped behind an adapter has to satisfy every clause, because consumers rely
/// on all of them and a slip corrupts their output silently rather than failing
/// loudly. The convex-hull operator audits these clauses on entry in debug
/// builds, naming the broken one.
///
/// An implementor **must**:
///
/// 1. Return `None` **exactly when** the cloud holds fewer than two distinct
///    points — i.e. `None` is *required* there, and *forbidden* anywhere else.
///    (A cloud of three coincident points has no hull of any dimension: return
///    `None`, not a one-vertex topology.) Distinctness is exact positional
///    equality.
/// 2. Return a genuine degenerate hull, never `None`, for a cloud that is merely
///    lower-dimensional: a collinear cloud's hull is its two extreme endpoints
///    (`vertices.len() == 2`, `faces` empty — the *only* legitimate facet-less
///    topology); a coplanar cloud's hull is a single flat n-gon facet.
/// 3. Index every output — both `HullTopology::vertices` and each
///    `HullFace::vertices` — into the **input `points` slice**, never into a
///    remapped hull-local vertex set. Every index must be `< points.len()`.
/// 4. Emit `HullTopology::vertices` as the on-hull index subset in **strictly
///    ascending** order: sorted, with no duplicates.
/// 5. Make `HullTopology::vertices` **exactly** the set of indices its own facets
///    reference, whenever `faces` is non-empty — neither a proper subset (an
///    index built into a face but reported on no output slot) nor a proper
///    superset (an index reported on-hull that no face uses).
/// 6. Give every facet at least three corners, wound counter-clockwise as seen
///    from **outside** the hull, with coplanar input merged into a single n-gon
///    facet rather than split into a triangle fan.
pub trait ConvexHullProvider {
    /// Compute the convex hull of `points`, upholding the trait's contract.
    ///
    /// Returns `None` **exactly when** fewer than two distinct points exist (no
    /// hull of any dimension). Coplanar and collinear clouds are *not* that case:
    /// they resolve to their genuine degenerate hulls.
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

/// Compute the 3D convex hull of `points`.
///
/// Returns `None` exactly when fewer than two distinct points exist (no hull of
/// any dimension). Collinear clouds reduce to their two extreme endpoints;
/// coplanar clouds reduce to a single flat polygon over their 2D boundary;
/// full-dimensional clouds yield a closed polytope with coplanar faces merged.
pub fn convex_hull(points: &[DVec3]) -> Option<HullTopology> {
    let basis = dimension_basis(points)?;
    match basis {
        DimensionBasis::Volume(seed) => Some(full_hull(points, seed)),
        DimensionBasis::Plane { origin, edge, area } => {
            Some(planar_hull(points, origin, edge, area))
        }
        DimensionBasis::Line { origin, direction } => {
            Some(collinear_hull(points, origin, direction))
        }
    }
}

/// The dimensionality of a point cloud and a basis for its affine hull.
enum DimensionBasis {
    /// Full-dimensional: a positive-volume seed tetrahedron.
    Volume([usize; 4]),
    /// Coplanar (rank 2): a point on the plane and two in-plane vectors. `edge`
    /// and `area` span the plane; their cross product is the plane normal.
    Plane {
        origin: usize,
        edge: usize,
        area: usize,
    },
    /// Collinear (rank 1): a point and a direction vector index.
    Line { origin: usize, direction: usize },
}

/// Classify `points` by affine dimension and return a non-degenerate basis.
///
/// Returns `None` only when fewer than two distinct points exist. Distinctness
/// is decided by exact equality of positions, so coincident duplicates collapse
/// before any predicate runs.
fn dimension_basis(points: &[DVec3]) -> Option<DimensionBasis> {
    let n = points.len();
    if n == 0 {
        return None;
    }

    // First two distinct points span the line.
    let origin = 0;
    let direction = (1..n).find(|&i| points[i] != points[origin])?;

    // Third point off the line: maximize the cross-product magnitude (orient3d
    // still decides the simplex below; this is only for numerical comfort).
    let line = points[direction] - points[origin];
    let mut area = None;
    let mut best_area = 0.0f64;
    for i in 0..n {
        if i == origin || i == direction {
            continue;
        }
        let span = line.cross(points[i] - points[origin]).length_squared();
        if span > best_area {
            best_area = span;
            area = Some(i);
        }
    }
    let area = match area {
        Some(index) if best_area > 0.0 => index,
        _ => {
            return Some(DimensionBasis::Line { origin, direction });
        }
    };

    // Fourth point off the plane, chosen by largest exact orientation magnitude.
    let mut apex = None;
    let mut best_vol = 0.0f64;
    for i in 0..n {
        if i == origin || i == direction || i == area {
            continue;
        }
        let vol = orient(points[origin], points[direction], points[area], points[i]).abs();
        if vol > best_vol {
            best_vol = vol;
            apex = Some(i);
        }
    }
    match apex {
        Some(index) if best_vol > 0.0 => {
            Some(DimensionBasis::Volume([origin, direction, area, index]))
        }
        _ => Some(DimensionBasis::Plane {
            origin,
            edge: direction,
            area,
        }),
    }
}

// ---------------------------------------------------------------------------
// Full-dimensional hull
// ---------------------------------------------------------------------------

/// Build the closed polytope from a positive-volume seed simplex, then merge
/// coplanar faces.
fn full_hull(points: &[DVec3], seed: [usize; 4]) -> HullTopology {
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
        let adjacency = build_adjacency(&faces);
        let visible = visible_faces(points, &faces, &adjacency, face_index, apex);
        let horizon = horizon_edges(&faces, &visible);

        let mut orphans: Vec<usize> = Vec::new();
        for &vf in &visible {
            faces[vf].alive = false;
            orphans.append(&mut outside[vf]);
        }

        // Stitch a new face from each horizon edge to the apex; redistribute
        // the orphaned points onto the first new face each can see (mirroring
        // the single-owner assignment of `assign_to_face`).
        let new_face_start = faces.len();
        for (edge_start, edge_end) in horizon {
            faces.push(WorkFace {
                points: [edge_start, edge_end, apex],
                alive: true,
            });
            outside.push(Vec::new());
        }
        let new_faces = new_face_start..faces.len();

        for &orphan in &orphans {
            if orphan == apex {
                continue;
            }
            for new_index in new_faces.clone() {
                if point_outside(points, &faces[new_index], orphan) {
                    outside[new_index].push(orphan);
                    break;
                }
            }
        }
        for new_index in new_faces {
            if !outside[new_index].is_empty() {
                face_queue.push(new_index);
            }
        }
    }

    merge_coplanar(points, &faces)
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

/// A map from each directed edge `(start, end)` to the alive face that owns it.
/// On a closed orientable hull each directed edge belongs to exactly one face,
/// so this gives O(1) neighbour lookup (cross an edge by reversing it).
type Adjacency = std::collections::HashMap<(usize, usize), usize>;

fn build_adjacency(faces: &[WorkFace]) -> Adjacency {
    let mut adjacency =
        std::collections::HashMap::with_capacity(faces.iter().filter(|f| f.alive).count() * 3);
    for (face_index, face) in faces.iter().enumerate() {
        if !face.alive {
            continue;
        }
        for edge in face.edges() {
            adjacency.insert(edge, face_index);
        }
    }
    adjacency
}

/// All alive faces visible from `apex`, found by flooding from `start` across
/// shared edges (the visible set is edge-connected on a convex hull). Neighbour
/// lookup is O(1) via the directed-edge adjacency map.
fn visible_faces(
    points: &[DVec3],
    faces: &[WorkFace],
    adjacency: &Adjacency,
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
        for (edge_start, edge_end) in faces[face_index].edges() {
            if let Some(&neighbour) = adjacency.get(&(edge_end, edge_start)) {
                if !seen[neighbour] && faces[neighbour].alive {
                    seen[neighbour] = true;
                    stack.push(neighbour);
                }
            }
        }
    }
    visible
}

/// The horizon: directed boundary edges of the visible region — edges of a
/// visible face whose opposite (reverse) edge does not belong to any other
/// visible face. Each horizon edge becomes the base of one new face.
fn horizon_edges(faces: &[WorkFace], visible: &[usize]) -> Vec<(usize, usize)> {
    let visible_set: std::collections::HashSet<(usize, usize)> = visible
        .iter()
        .flat_map(|&face_index| faces[face_index].edges())
        .collect();
    let mut horizon = Vec::new();
    for &face_index in visible {
        for (edge_start, edge_end) in faces[face_index].edges() {
            if !visible_set.contains(&(edge_end, edge_start)) {
                horizon.push((edge_start, edge_end));
            }
        }
    }
    horizon
}

// ---------------------------------------------------------------------------
// Coplanar-face merge
// ---------------------------------------------------------------------------

/// Merge maximally-coplanar adjacent triangles of the finished polytope into
/// single n-gon faces, then collect the on-hull vertex set.
///
/// Two edge-adjacent triangles belong to the same face iff they are exactly
/// coplanar (the neighbour's off-edge vertex lies in the triangle's plane).
/// Each merged component's boundary is the chain of directed edges whose
/// reverse is not internal to the component; that chain is the n-gon's CCW
/// winding.
fn merge_coplanar(points: &[DVec3], faces: &[WorkFace]) -> HullTopology {
    let alive: Vec<usize> = faces
        .iter()
        .enumerate()
        .filter(|(_, face)| face.alive)
        .map(|(index, _)| index)
        .collect();

    // Position in `alive` for each global face index, so union-find works over
    // a compact range.
    let mut slot = vec![usize::MAX; faces.len()];
    for (compact, &face_index) in alive.iter().enumerate() {
        slot[face_index] = compact;
    }

    let edge_owner = build_adjacency(faces);
    let mut groups = UnionFind::new(alive.len());

    for (compact, &face_index) in alive.iter().enumerate() {
        let [a, b, c] = faces[face_index].points;
        for (edge_start, edge_end) in faces[face_index].edges() {
            if let Some(&neighbour) = edge_owner.get(&(edge_end, edge_start)) {
                if !faces[neighbour].alive {
                    continue;
                }
                let off_edge = off_edge_vertex(&faces[neighbour], edge_end, edge_start);
                if orient(points[a], points[b], points[c], points[off_edge]) == 0.0 {
                    groups.union(compact, slot[neighbour]);
                }
            }
        }
    }

    // Gather each component's triangles in a deterministic order. We iterate the
    // alive faces in ascending index order and bucket each into its union-find
    // root; the root's *first appearance* fixes the component's emission slot, so
    // the facet list is hash-independent (a default-hasher map here would emit
    // facets — and therefore their anchor vertices downstream — in a per-process
    // order, the root of the reuse-path nondeterminism).
    let mut component_slot = vec![usize::MAX; alive.len()];
    let mut component_triangles: Vec<Vec<usize>> = Vec::new();
    for (compact, &face_index) in alive.iter().enumerate() {
        let root = groups.find(compact);
        let slot = if component_slot[root] == usize::MAX {
            let slot = component_triangles.len();
            component_slot[root] = slot;
            component_triangles.push(Vec::new());
            slot
        } else {
            component_slot[root]
        };
        component_triangles[slot].push(face_index);
    }

    let mut topo = HullTopology::default();
    let mut on_hull = vec![false; points.len()];

    for triangles in &component_triangles {
        let face = extract_boundary_polygon(faces, triangles);
        for &index in &face.vertices {
            on_hull[index] = true;
        }
        topo.faces.push(face);
    }

    for (index, &present) in on_hull.iter().enumerate() {
        if present {
            topo.vertices.push(index);
        }
    }
    topo
}

/// The vertex of `triangle` that is not on the directed edge `(u, v)` (nor its
/// reverse).
#[inline]
fn off_edge_vertex(triangle: &WorkFace, u: usize, v: usize) -> usize {
    triangle.points[triangle
        .points
        .iter()
        .position(|&p| p != u && p != v)
        .expect("triangle has a vertex off any of its edges")]
}

/// Chain the boundary directed edges of a coplanar triangle component into a
/// single CCW n-gon. Internal edges (those whose reverse is also in the
/// component) cancel; the surviving boundary edges form one closed loop because
/// each component is a disc of a convex polytope facet.
fn extract_boundary_polygon(faces: &[WorkFace], triangles: &[usize]) -> HullFace {
    let internal: std::collections::HashSet<(usize, usize)> = triangles
        .iter()
        .flat_map(|&face_index| faces[face_index].edges())
        .collect();

    // next[start] = end for each surviving boundary edge. A correct convex-hull
    // facet boundary is a simple disc, so each vertex has exactly one outgoing
    // boundary edge; a collision would mean the component pinched at a vertex,
    // which is unreachable for valid hull input. A `BTreeMap` keeps the keys in
    // ascending order so the walk's start vertex (the smallest boundary index) is
    // deterministic — the emitted n-gon's anchor vertex must not depend on the
    // hash seed, or a downstream find-or-create can miss an existing face.
    let mut next: std::collections::BTreeMap<usize, usize> = std::collections::BTreeMap::new();
    for &face_index in triangles {
        for (edge_start, edge_end) in faces[face_index].edges() {
            if !internal.contains(&(edge_end, edge_start)) {
                let previous = next.insert(edge_start, edge_end);
                assert!(
                    previous.is_none(),
                    "coplanar component is not a simple disc: vertex has two boundary edges"
                );
            }
        }
    }

    // The smallest boundary vertex index is the deterministic anchor; the walk
    // then follows the CCW `next` chain, so the winding is preserved.
    let start = *next
        .keys()
        .next()
        .expect("coplanar component has a boundary");
    let mut vertices = Vec::with_capacity(next.len());
    let mut current = start;
    loop {
        vertices.push(current);
        current = next[&current];
        if current == start {
            break;
        }
        debug_assert!(
            vertices.len() <= next.len(),
            "boundary walk did not close into a single loop"
        );
    }
    // The walk closed; a simple disc consumes every boundary edge exactly once,
    // so an under-length loop means the boundary was not a single cycle.
    assert!(
        vertices.len() == next.len(),
        "boundary walk closed before consuming every boundary edge"
    );
    HullFace { vertices }
}

/// Minimal union-find over a compact index range, for grouping coplanar faces.
struct UnionFind {
    parent: Vec<usize>,
}

impl UnionFind {
    fn new(count: usize) -> Self {
        UnionFind {
            parent: (0..count).collect(),
        }
    }

    fn find(&mut self, mut node: usize) -> usize {
        while self.parent[node] != node {
            self.parent[node] = self.parent[self.parent[node]];
            node = self.parent[node];
        }
        node
    }

    fn union(&mut self, a: usize, b: usize) {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra != rb {
            self.parent[ra] = rb;
        }
    }
}

// ---------------------------------------------------------------------------
// Degenerate (lower-dimensional) hulls
// ---------------------------------------------------------------------------

/// A coplanar cloud reduces to a single flat polygon over its 2D convex
/// boundary, with interior points culled.
///
/// We work in the plane spanned by `(points[edge] - points[origin])` and the
/// in-plane normal `n x edge` (so the 2D coordinates stay exact in spirit and
/// the gift-wrap below uses exact `orient3d` for every turn test). The polygon
/// is wound CCW about the plane normal `edge x area`.
fn planar_hull(points: &[DVec3], origin: usize, edge: usize, area: usize) -> HullTopology {
    let normal = (points[edge] - points[origin]).cross(points[area] - points[origin]);

    // A reference point strictly off the plane on the positive side, so the
    // 3D `orient` predicate decides 2D left/right turns exactly.
    let apex = points[origin] + normal;

    let boundary = gift_wrap_2d(points, apex);
    let mut topo = HullTopology::default();
    if boundary.len() >= 3 {
        topo.vertices = sorted_unique(&boundary);
        topo.faces.push(HullFace { vertices: boundary });
    } else {
        // Fewer than three distinct boundary points means the cloud collapsed
        // to a segment after dedup; fall back to the collinear hull.
        return collinear_hull(points, origin, edge);
    }
    topo
}

/// Gift-wrap the 2D convex boundary of a coplanar cloud. `apex` is a fixed
/// reference off the plane used to turn 2D turn-tests into exact 3D
/// orientations.
///
/// Each hull edge `(current, next)` is the one for which every other point lies
/// to the left (`orient(current, next, apex, other) >= 0` with no point strictly
/// to the right). Starts from the lexicographically smallest position so the
/// result is deterministic.
fn gift_wrap_2d(points: &[DVec3], apex: DVec3) -> Vec<usize> {
    let n = points.len();

    // Deduplicate coincident positions: keep the lowest index per position.
    let mut representative: Vec<usize> = Vec::new();
    'outer: for i in 0..n {
        for &kept in &representative {
            if points[i] == points[kept] {
                continue 'outer;
            }
        }
        representative.push(i);
    }
    if representative.len() < 3 {
        return representative;
    }

    // Start at the extreme along the plane's `edge` direction (lowest lexicographic
    // position) — guaranteed on the boundary.
    let start = *representative
        .iter()
        .min_by(|&&a, &&b| {
            let pa = points[a];
            let pb = points[b];
            (pa.x, pa.y, pa.z)
                .partial_cmp(&(pb.x, pb.y, pb.z))
                .unwrap()
        })
        .unwrap();

    let mut boundary = Vec::new();
    let mut current = start;
    loop {
        boundary.push(current);
        // Tentative next: any point other than `current`.
        let mut next = *representative.iter().find(|&&p| p != current).unwrap();
        for &candidate in &representative {
            if candidate == current || candidate == next {
                continue;
            }
            // `orient(current, next, apex, candidate)`: with `apex` off-plane on
            // the positive side, this is negative when `candidate` is to the
            // right of the directed edge (current -> next). A right-turn means
            // `candidate` is a more clockwise next vertex; keep sweeping until
            // every point lies left of (current -> next).
            let side = orient(points[current], points[next], apex, points[candidate]);
            if side < 0.0 {
                next = candidate;
            }
        }
        if next == start {
            break;
        }
        current = next;
        if boundary.len() > representative.len() {
            break;
        }
    }
    boundary
}

/// A collinear cloud reduces to its two extreme endpoints along the line.
fn collinear_hull(points: &[DVec3], origin: usize, direction: usize) -> HullTopology {
    let axis = points[direction] - points[origin];

    let mut min_index = origin;
    let mut max_index = origin;
    let mut min_t = axis.dot(points[origin] - points[origin]);
    let mut max_t = min_t;
    for i in 0..points.len() {
        let t = axis.dot(points[i] - points[origin]);
        if t < min_t {
            min_t = t;
            min_index = i;
        }
        if t > max_t {
            max_t = t;
            max_index = i;
        }
    }

    let mut topo = HullTopology::default();
    topo.vertices = sorted_unique(&[min_index, max_index]);
    // A 1D hull has no area face; the two endpoints carry the result. The
    // `faces` list stays empty for a genuinely 1D hull.
    topo
}

/// Sort and deduplicate a small index set in ascending order.
fn sorted_unique(indices: &[usize]) -> Vec<usize> {
    let mut out = indices.to_vec();
    out.sort_unstable();
    out.dedup();
    out
}
