//! Constrained Delaunay triangulation (CDT) — a shared comp-geom prerequisite.
//!
//! Triangulates a planar polygon (with holes and/or a free edge-net) that is
//! Delaunay where it can be while being *forced* to contain a prescribed set of
//! constraint edges. The constraint set is what mesh-fill needs: the input
//! loop's edges must survive and holes must be carved out of the result.
//!
//! This is a self-contained primitive: a 2D point cloud plus constraint edges
//! in, an index-based triangulation out. It does not touch `LoopMesh`
//! topology; fill operators translate the neutral output into radial-edge
//! surgery on top of this.
//!
//! The triangulation core is delegated to the `spade` crate. The output is
//! *neutral and index-based*: triangles reference the **input** vertex indices,
//! so a consumer maps them back to its own vertex keys with no positional drift
//! — a CDT reuses its input vertices, it never moves or re-coordinates them.

use glam::DVec2;
use spade::{ConstrainedDelaunayTriangulation, HasPosition, InsertionError, Point2, Triangulation};

/// A planar polygon-with-constraints to triangulate.
///
/// `points` is the input vertex cloud; every other field references indices
/// into it. The boundary, holes, and free edges are all *constraint edges* —
/// segments the output is forced to contain — they differ only in how they are
/// supplied (and holes additionally carve out an excluded region).
///
/// The vertex set of the output equals the set of `points` actually referenced
/// by the triangulation; positions are never altered.
#[derive(Clone, Debug, Default)]
pub struct CdtInput {
    /// The 2D vertex cloud. Output triangles index into this slice.
    pub points: Vec<DVec2>,
    /// The outer boundary loop as a sequence of vertex indices. The closing
    /// edge (last -> first) is implied; an open path is permitted but then
    /// describes no enclosed region on its own.
    pub boundary: Vec<usize>,
    /// Inner hole loops, each a sequence of vertex indices wound as a closed
    /// loop (closing edge implied). Triangles inside any hole are excluded
    /// from the output.
    pub holes: Vec<Vec<usize>>,
    /// Free constraint edges (an edge-net) as index pairs. These are forced to
    /// appear in the triangulation but do not by themselves bound a region.
    pub edges: Vec<[usize; 2]>,
}

impl CdtInput {
    /// All constraint edges as `(from, to)` index pairs: the boundary loop, the
    /// hole loops, and the free edges. The boundary and each hole contribute
    /// their implied closing edge.
    fn constraint_edges(&self) -> Vec<[usize; 2]> {
        let mut out = Vec::new();
        push_loop_edges(&self.boundary, &mut out);
        for hole in &self.holes {
            push_loop_edges(hole, &mut out);
        }
        out.extend_from_slice(&self.edges);
        out
    }
}

fn push_loop_edges(loop_indices: &[usize], out: &mut Vec<[usize; 2]>) {
    if loop_indices.len() < 2 {
        return;
    }
    for window in loop_indices.windows(2) {
        out.push([window[0], window[1]]);
    }
    let first = loop_indices[0];
    let last = loop_indices[loop_indices.len() - 1];
    if first != last {
        out.push([last, first]);
    }
}

/// A single output triangle as a triple of indices into [`CdtInput::points`].
///
/// Vertices are wound counter-clockwise. The triple is order-independent as a
/// *set* for invariant comparison, but the winding is meaningful for normals.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CdtTriangle {
    pub vertices: [usize; 3],
}

/// The result of a constrained triangulation: triangles over the input vertex
/// indices, with hole interiors excluded.
///
/// `vertices` is the ascending set of input indices that the triangulation
/// references (equal to the input set when no input point is isolated). The
/// triangulation introduces no new vertices, so this is always a subset of the
/// input indices — there is no positional drift.
#[derive(Clone, Debug, Default)]
pub struct CdtTriangulation {
    pub vertices: Vec<usize>,
    pub triangles: Vec<CdtTriangle>,
}

/// A provider that turns a constrained polygon into a triangulation.
///
/// The boundary is deliberately coarse — one call triangulates a whole polygon
/// — so an injected `&dyn CdtProvider` costs a single virtual dispatch per
/// triangulation. The production path uses the zero-sized [`SpadeCdt`] default
/// via static dispatch, so the abstraction inlines to a direct call and taxes
/// the loopmesh-native path nothing.
pub trait CdtProvider {
    /// Triangulate `input`.
    ///
    /// Returns `None` when the input bounds no region or is malformed for a
    /// constrained triangulation:
    ///
    /// - no triangulation of positive area exists — fewer than three distinct
    ///   points, or an all-collinear cloud;
    /// - no face is interior to the constrained region — an open boundary or a
    ///   free edge-net with no enclosing boundary loop carves out nothing, so
    ///   the parity flood-fill marks every face outside and there is nothing to
    ///   emit; or
    /// - a constraint edge cannot be inserted because it intersects another
    ///   constraint edge (a self-crossing hole loop, two crossing hole loops,
    ///   or an edge-net segment that crosses the boundary or another segment).
    ///   Dropping the offending constraint would silently break the
    ///   "every constraint edge appears in the output" invariant, so the whole
    ///   triangulation fails cleanly instead.
    fn triangulate(&self, input: &CdtInput) -> Option<CdtTriangulation>;
}

/// The in-tree default CDT provider: a `spade`-backed constrained Delaunay
/// triangulation. Zero-sized, so the trait abstraction is free on the
/// production path.
#[derive(Clone, Copy, Debug, Default)]
pub struct SpadeCdt;

impl CdtProvider for SpadeCdt {
    #[inline]
    fn triangulate(&self, input: &CdtInput) -> Option<CdtTriangulation> {
        triangulate(input)
    }
}

/// A spade vertex that remembers which input index it came from, so the output
/// maps back to the caller's cloud regardless of how spade orders or merges
/// coincident vertices internally.
#[derive(Clone, Copy)]
struct IndexedPoint {
    position: Point2<f64>,
    input_index: usize,
}

impl HasPosition for IndexedPoint {
    type Scalar = f64;
    #[inline]
    fn position(&self) -> Point2<f64> {
        self.position
    }
}

/// Triangulate a constrained polygon with the `spade` backend.
///
/// Returns `None` when fewer than three non-collinear distinct points exist (no
/// positive-area triangulation), or when a constraint edge cannot be inserted
/// because it intersects another constraint edge.
pub fn triangulate(input: &CdtInput) -> Option<CdtTriangulation> {
    let mut cdt: ConstrainedDelaunayTriangulation<IndexedPoint> =
        ConstrainedDelaunayTriangulation::new();

    // Insert every input point, recording the spade handle each maps to.
    // Coincident inputs collapse onto the same handle; that is fine — the
    // constraint indices below all resolve through `handles`.
    let mut handles = Vec::with_capacity(input.points.len());
    for (input_index, point) in input.points.iter().enumerate() {
        match insert_point(&mut cdt, *point, input_index) {
            Ok(handle) => handles.push(handle),
            Err(_) => return None,
        }
    }

    for [from, to] in input.constraint_edges() {
        let (Some(&a), Some(&b)) = (handles.get(from), handles.get(to)) else {
            continue;
        };
        if a == b {
            continue;
        }
        // Non-panicking insertion: spade's `add_constraint` panics when the new
        // edge crosses an existing constraint edge, but malformed fill regions
        // (self-crossing holes, crossing edge-nets) are an expected input case
        // here. `try_add_constraint` instead leaves the triangulation unchanged
        // and returns an empty list when the constraint is un-insertable. Since
        // `a != b` is already guaranteed, an empty result means an intersection
        // conflict, so the whole triangulation fails cleanly.
        if cdt.try_add_constraint(a, b).is_empty() {
            return None;
        }
    }

    let triangulation = collect(&cdt, input);
    if triangulation.triangles.is_empty() {
        // No interior face survived: either no positive-area triangulation
        // exists (fewer than three distinct points, or an all-collinear cloud),
        // or nothing bounds a region (an open boundary, or a free edge-net with
        // no enclosing boundary loop, which the parity flood-fill leaves wholly
        // outside).
        return None;
    }
    Some(triangulation)
}

fn insert_point(
    cdt: &mut ConstrainedDelaunayTriangulation<IndexedPoint>,
    point: DVec2,
    input_index: usize,
) -> Result<spade::handles::FixedVertexHandle, InsertionError> {
    cdt.insert(IndexedPoint {
        position: Point2::new(point.x, point.y),
        input_index,
    })
}

/// Walk the finished triangulation, keep only the faces interior to the
/// constrained region, and emit them as input-indexed triangles.
fn collect(
    cdt: &ConstrainedDelaunayTriangulation<IndexedPoint>,
    input: &CdtInput,
) -> CdtTriangulation {
    let inside = inside_faces(cdt);

    let mut triangulation = CdtTriangulation::default();
    let mut referenced = vec![false; input.points.len()];

    for face in cdt.inner_faces() {
        if !inside[face.fix().index()] {
            continue;
        }
        let vertices = face.vertices();
        let triple = [
            vertices[0].data().input_index,
            vertices[1].data().input_index,
            vertices[2].data().input_index,
        ];
        // A constraint that collapses two distinct input indices onto one
        // spade vertex could in principle pinch a triangle; drop any degenerate
        // triple defensively rather than emit a zero-area face.
        if triple[0] == triple[1] || triple[1] == triple[2] || triple[0] == triple[2] {
            continue;
        }
        for index in triple {
            referenced[index] = true;
        }
        triangulation
            .triangles
            .push(CdtTriangle { vertices: triple });
    }

    for (index, &present) in referenced.iter().enumerate() {
        if present {
            triangulation.vertices.push(index);
        }
    }
    triangulation
}

/// Classify every face (by its fixed index) as inside or outside the
/// constrained region via a parity flood-fill.
///
/// Start at the single outer face, which is by definition *outside*. Flood
/// across shared edges; crossing a constraint edge flips the in/out parity,
/// crossing a non-constraint edge preserves it. The polygon boundary and every
/// hole loop are constraint edges, so the exterior, the polygon interior, and
/// each hole interior end up correctly alternating: the polygon interior is one
/// constraint-crossing in from the exterior, and a hole interior is two
/// crossings in (back to outside).
fn inside_faces(cdt: &ConstrainedDelaunayTriangulation<IndexedPoint>) -> Vec<bool> {
    let face_count = cdt.num_all_faces();

    // Build the dual graph: which faces each undirected edge separates. Every
    // undirected edge borders exactly two faces (an inner face borders the
    // outer face on the convex boundary), so neighbour lookup is by index.
    let mut adjacency: Vec<Vec<(usize, bool)>> = (0..face_count).map(|_| Vec::new()).collect();
    for edge in cdt.undirected_edges() {
        let is_constraint = edge.is_constraint_edge();
        let directed = edge.as_directed();
        let a = directed.face().fix().index();
        let b = directed.rev().face().fix().index();
        adjacency[a].push((b, is_constraint));
        adjacency[b].push((a, is_constraint));
    }

    let mut inside = vec![false; face_count];
    let mut visited = vec![false; face_count];

    let outer = cdt.outer_face().fix().index();
    visited[outer] = true;
    let mut stack = vec![(outer, false)];

    while let Some((face_index, face_is_inside)) = stack.pop() {
        inside[face_index] = face_is_inside;
        for &(neighbour, crosses_constraint) in &adjacency[face_index] {
            if visited[neighbour] {
                continue;
            }
            visited[neighbour] = true;
            stack.push((neighbour, face_is_inside ^ crosses_constraint));
        }
    }

    inside
}
