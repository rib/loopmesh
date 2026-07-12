use std::collections::HashMap;

use glam::{Vec2, Vec3};

use crate::loopmesh::*;
use crate::triangulate::{
    self, axis_dominant_basis, axis_dominant_basis_negated, beautify_polygon, ear_clip,
};

/// Order a vertex-key pair so the two orientations of an undirected edge hash to
/// the same key — used to build the §6 beautify boundary-protection set.
fn unordered_pair(a: VertexKey, b: VertexKey) -> (VertexKey, VertexKey) {
    if a.as_usize() <= b.as_usize() {
        (a, b)
    } else {
        (b, a)
    }
}

/// Axis-aligned 2D bounding box of a component's projected vertices,
/// used by the §1.2.3 hole-coalesce pass.
#[derive(Copy, Clone, Debug)]
struct BoundingBox2D {
    min: Vec2,
    max: Vec2,
}

impl BoundingBox2D {
    fn empty() -> Self {
        Self {
            min: Vec2::splat(f32::INFINITY),
            max: Vec2::splat(f32::NEG_INFINITY),
        }
    }

    fn expand(&mut self, p: Vec2) {
        self.min = self.min.min(p);
        self.max = self.max.max(p);
    }

    fn union(&mut self, other: &BoundingBox2D) {
        self.min = self.min.min(other.min);
        self.max = self.max.max(other.max);
    }

    fn overlaps(&self, other: &BoundingBox2D) -> bool {
        self.min.x <= other.max.x
            && other.min.x <= self.max.x
            && self.min.y <= other.max.y
            && other.min.y <= self.max.y
    }

    fn area(&self) -> f32 {
        let extent = (self.max - self.min).max(Vec2::ZERO);
        extent.x * extent.y
    }
}

/// Triangulate a bridged keyhole polygon (§1.2.4) — a simple polygon whose
/// vertex sequence may repeat the two bridge endpoints — by ear-clipping with a
/// **full** interior-vertex rejection test. Unlike the single-loop ear-clip,
/// every candidate ear is checked against *all* remaining vertices (not only
/// reflex ones), which is what keeps any ear from escaping across the hole: a
/// hole-loop vertex inside a candidate triangle always rejects it. Emits index
/// triples into `coords`.
///
/// Auto-detects the polygon winding from its signed area so the orientation
/// convention matches the caller's projection. Produces *a* valid triangulation
/// of the region-with-holes (§1.1 "any valid triangulation"), leaving the hole
/// empty.
fn triangle_fill_ear_clip_with_holes(coords: &[Vec2], out_tris: &mut Vec<[u32; 3]>) {
    let n = coords.len();
    if n < 3 {
        return;
    }

    // Signed area: positive ⇒ CCW. The interior of a convex corner is on the
    // left for a CCW polygon, on the right for CW; `interior_sign` carries that
    // so the ear/inclusion tests work for either winding.
    let mut twice_area = 0.0f32;
    for i in 0..n {
        let a = coords[i];
        let b = coords[(i + 1) % n];
        twice_area += a.x * b.y - b.x * a.y;
    }
    let interior_sign = if twice_area >= 0.0 { 1.0 } else { -1.0 };

    let cross = |o: Vec2, a: Vec2, b: Vec2| (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);
    // Strict point-in-triangle (apex `b`, neighbours `a`,`c`) for the polygon's
    // winding. A point shared with a triangle corner is treated as outside so a
    // repeated bridge vertex never rejects its own ear.
    let point_in_tri = |p: Vec2, a: Vec2, b: Vec2, c: Vec2| {
        let d1 = interior_sign * cross(a, b, p);
        let d2 = interior_sign * cross(b, c, p);
        let d3 = interior_sign * cross(c, a, p);
        d1 > 1e-9 && d2 > 1e-9 && d3 > 1e-9
    };

    let mut next: Vec<usize> = (0..n).map(|i| (i + 1) % n).collect();
    let mut prev: Vec<usize> = (0..n).map(|i| (i + n - 1) % n).collect();
    let mut remaining = n;
    let mut cursor = 0usize;
    let mut guard = 0usize;
    let guard_limit = n * n + n;

    while remaining > 3 {
        guard += 1;
        if guard > guard_limit {
            break;
        }
        let i_prev = prev[cursor];
        let i_next = next[cursor];
        let a = coords[i_prev];
        let b = coords[cursor];
        let c = coords[i_next];

        // Convex (an ear must turn toward the interior) and non-degenerate.
        let turn = interior_sign * cross(a, b, c);
        let is_convex = turn > 1e-12;

        let mut is_ear = is_convex;
        if is_ear {
            // Full inclusion test: reject if any *other* remaining vertex lies
            // strictly inside the candidate triangle (§1.2.4 hole guard).
            let mut probe = next[i_next];
            while probe != i_prev {
                if probe != cursor
                    && coords[probe] != a
                    && coords[probe] != b
                    && coords[probe] != c
                    && point_in_tri(coords[probe], a, b, c)
                {
                    is_ear = false;
                    break;
                }
                probe = next[probe];
            }
        }

        if is_ear {
            out_tris.push([i_prev as u32, cursor as u32, i_next as u32]);
            next[i_prev] = i_next;
            prev[i_next] = i_prev;
            remaining -= 1;
            cursor = i_prev;
            guard = 0;
        } else {
            cursor = i_next;
        }
    }

    if remaining == 3 {
        let i_prev = prev[cursor];
        let i_next = next[cursor];
        out_tris.push([i_prev as u32, cursor as u32, i_next as u32]);
    }
}

/// Projected bounding box of a single loop, for the §1.2.3 outer-loop ordering.
fn loop_box(
    mesh: &LoopMesh,
    loop_verts: &[VertexKey],
    project: &impl Fn(&LoopMesh, VertexKey) -> Vec2,
) -> BoundingBox2D {
    let mut bbox = BoundingBox2D::empty();
    for &v in loop_verts {
        bbox.expand(project(mesh, v));
    }
    bbox
}

/// Squared-length floor below which a normal (derived corner normal, accumulated
/// fill normal, or supplied `normal`) is treated as zero. Guards the projection
/// basis and the supplied-vs-derived choice against a degenerate-length vector.
const MIN_NORMAL_LEN_SQ: f32 = 1e-12;

/// Parameters for [`LoopMesh::triangle_fill`], mirroring the operator's BMOP
/// slot knobs.
///
/// Algorithm in
/// [`TRIANGLE_FILL.md`](../../../../radial-mesh-guides/operators/TRIANGLE_FILL.md)
/// §1.
#[derive(Copy, Clone, Debug, Default)]
pub struct TriangleFillParams {
    /// When set, run the beautify edge-rotation pass after the raw fill (§6):
    /// rotate the fill's interior diagonals toward a Delaunay-like
    /// triangulation, before the §7 dissolve (so beautify runs before dissolve
    /// when both flags are set). Applied to the whole fill region — the
    /// single-loop (no-holes) fill and the bridged with-holes/keyhole fill
    /// (§1.2.4) alike; on the keyhole, the region's loop edges, hole rims, and
    /// the zero-width bridge slit are protected so only genuine interior
    /// diagonals rotate and no diagonal escapes into the hole.
    pub use_beauty: bool,
    /// When set, dissolve the fill's new interior edges into larger n-gons
    /// after the fill (§7).
    pub use_dissolve: bool,
    /// Projection-plane normal (§1/§3). `None` (the default) derives a fill
    /// normal from the input geometry (§3); a supplied non-zero value is used
    /// directly as the projection plane and skips the §4 winding vote.
    pub normal: Option<Vec3>,
}

/// The `geom.out` of [`LoopMesh::triangle_fill`]: every **new** interior edge
/// and **new** fill face the operator created (§1). Input boundary edges are
/// never reported. After a dissolve pass the surviving joined faces and
/// surviving fill edges are reported.
#[derive(Clone, Debug, Default)]
pub struct TriangleFillResult {
    edges: Vec<EdgeKey>,
    faces: Vec<FaceKey>,
}

impl TriangleFillResult {
    pub fn edges(&self) -> &[EdgeKey] {
        &self.edges
    }
    pub fn faces(&self) -> &[FaceKey] {
        &self.faces
    }
    pub fn is_empty(&self) -> bool {
        self.edges.is_empty() && self.faces.is_empty()
    }
}

impl LoopMesh {
    /// Span the empty region bounded by an undirected set of `edges` with
    /// triangles (§1). The operator never creates a vertex (every corner is
    /// an existing input endpoint, §2) and never deletes an input edge.
    ///
    /// Algorithm in
    /// [`TRIANGLE_FILL.md`](../../../../radial-mesh-guides/operators/TRIANGLE_FILL.md):
    /// the fill is a planar triangulation of the bounded region (§1.1 — the
    /// contract is *any* valid triangulation, optionally beautified), the
    /// projection plane comes from the derived or supplied normal (§3), the
    /// winding agrees with any existing boundary faces (§4), each triangle is
    /// materialised with the no-double edge discipline (§5), and — when
    /// `params.use_dissolve` is set — the new interior edges are JFKE-joined
    /// back into n-gons (§7).
    ///
    /// A supplied non-zero `params.normal` is used directly as the projection
    /// plane and skips the §4 winding vote; the fill orients to that normal.
    ///
    /// **Scope:** a single simple closed loop rides the fast path
    /// ([`Self::triangle_fill_order_single_loop`] → ear-clip). An edge set
    /// that is *not* one simple closed loop — multiple disjoint loops, an
    /// annulus/hole, an open chain, or loose edges (§8) — is handled by the
    /// region-discovery path (§1.2): connected-component labelling (§1.2.1),
    /// loose-edge / dangling-chain trim (§1.2.2), bounding-box coalescing of
    /// components into fill regions (§1.2.3), and a with-holes fill that leaves
    /// an enclosed inner loop empty (§1.2.4). Multiple disjoint regions fill
    /// independently in the one call; loose edges create nothing.
    ///
    /// **`use_beauty`:** when set, the beautify post-pass (§6) rotates the
    /// fill's interior diagonals toward a Delaunay-like triangulation before the
    /// §7 dissolve. Applied to the whole fill region — both the single-loop
    /// (no-holes) fill and the bridged with-holes/keyhole fill (§1.2.4); on the
    /// keyhole, loop edges, hole rims, and the bridge slit are protected so only
    /// interior diagonals rotate.
    pub fn triangle_fill(
        &mut self,
        edges: &[EdgeKey],
        params: &TriangleFillParams,
    ) -> TriangleFillResult {
        let mut result = TriangleFillResult::default();

        // §8: fewer than three edges can bound no region.
        if edges.len() < 3 {
            return result;
        }

        let mut interior_edges: Vec<EdgeKey> = Vec::new();
        let mut created_faces: Vec<FaceKey> = Vec::new();
        // §7/§8 — the §1.2.4 bridge/slit edges that separate a band from its
        // hole. The dissolve pass must never JFKE-join across one of these, or
        // the hole rim merges into the outer disk and the hole fills (χ would
        // collapse from 0 to 1). Only the with-holes path ever populates this;
        // the single-simple-loop path leaves it empty, so its dissolve is
        // bit-for-bit unchanged.
        let mut protected_edges: std::collections::HashSet<EdgeKey> =
            std::collections::HashSet::new();

        if let Some(loop_verts) = self.triangle_fill_order_single_loop(edges) {
            // Fast path: the whole edge set is one simple closed loop.
            self.triangle_fill_region(
                &[loop_verts],
                edges,
                params,
                &mut interior_edges,
                &mut created_faces,
                &mut protected_edges,
            );
        } else {
            // §1.2 region-discovery path: multiple loops, an annulus/hole, an
            // open chain, or loose edges. Decompose the net into regions, trim
            // dangling edges, and fill each region (holes left empty).
            for region_loops in self.triangle_fill_discover_regions(edges) {
                self.triangle_fill_region(
                    &region_loops,
                    edges,
                    params,
                    &mut interior_edges,
                    &mut created_faces,
                    &mut protected_edges,
                );
            }
        }

        if created_faces.is_empty() {
            return result;
        }

        if params.use_dissolve {
            // Seed the survivor set with *every* fill face just created, so a
            // face whose edges are never visited as a manifold interior edge
            // (e.g. the lone triangle of a 3-edge loop) is still reported in
            // geom.out (§1/§7).
            self.triangle_fill_dissolve(
                &interior_edges,
                created_faces,
                &protected_edges,
                &mut result,
            );
        } else {
            result.faces = created_faces;
            result.edges = interior_edges;
        }

        result
    }

    /// Fill one region — an outer loop plus zero or more enclosed hole loops
    /// (§1.2.4) — appending its new interior diagonals and fill faces to the
    /// caller's accumulators. The fill is a planar triangulation of the
    /// outer-with-holes region; a hole is bridged into the outer boundary so
    /// the band between them is tiled while the hole stays empty.
    ///
    /// `region_loops[0]` is the outer loop; `region_loops[1..]` are holes. A
    /// single-loop region (no holes) is the §1.1 simple-polygon case.
    fn triangle_fill_region(
        &mut self,
        region_loops: &[Vec<VertexKey>],
        boundary_edges: &[EdgeKey],
        params: &TriangleFillParams,
        interior_edges: &mut Vec<EdgeKey>,
        created_faces: &mut Vec<FaceKey>,
        protected_edges: &mut std::collections::HashSet<EdgeKey>,
    ) {
        let outer = &region_loops[0];
        if outer.len() < 3 {
            return;
        }

        // §3 — projection-plane normal: supplied (authoritative) or derived
        // from the outer loop's corner turns.
        let (fill_normal, normal_was_derived) = match params.normal {
            Some(n) if n.length_squared() > MIN_NORMAL_LEN_SQ => (n.normalize(), false),
            _ => (self.triangle_fill_derive_normal(outer), true),
        };
        // §3 — a supplied normal sets the projection plane *and* the default
        // orientation: the fill must face `+normal` (wound CCW seen from
        // `+normal`). The `+n` frame's third axis is `+n`, so a CCW projected
        // loop materialises facing `+n`. The derived-normal path uses the `-n`
        // frame (its CW projection is what the §4 winding vote and the
        // ear-clip kernel expect) — a free loop has no inherent front and the
        // vote corrects orientation against any boundary faces.
        let (axis_u, axis_w) = if normal_was_derived {
            axis_dominant_basis_negated(fill_normal)
        } else {
            axis_dominant_basis(fill_normal)
        };
        let project = |mesh: &LoopMesh, v: VertexKey| {
            let p = mesh.vertex_position(v);
            Vec2::new(axis_u.dot(p), axis_w.dot(p))
        };

        // §1.2.4 — bridge each hole loop into the outer boundary, yielding a
        // single simple-polygon vertex sequence whose triangulation leaves the
        // hole empty. With no holes this is just the outer loop verbatim.
        let has_holes = region_loops.len() > 1;
        let (sweep_verts, bridge_pairs) = self.triangle_fill_bridge_holes(region_loops, &project);
        let projected: Vec<Vec2> = sweep_verts.iter().map(|&v| project(self, v)).collect();

        let mut tris: Vec<[u32; 3]> = Vec::with_capacity(sweep_verts.len());
        if has_holes {
            // A bridged keyhole is a simple polygon, but its repeated bridge
            // vertices and zero-width slit defeat the single-loop ear-clip's
            // convex-skip fast path (§1.2.4 needs the interior-vertex test to
            // run against *every* candidate, not only reflex ones). Use the
            // full-inclusion triangulator so no ear escapes across the hole.
            triangle_fill_ear_clip_with_holes(&projected, &mut tris);
        } else {
            ear_clip(&projected, 0, &mut tris);
        }
        if tris.is_empty() {
            return;
        }

        // §6 — beautify post-pass: rotate the fill's interior diagonals toward
        // a Delaunay-like triangulation before any faces are materialised. Only
        // which interior diagonals exist changes; the boundary edges and the
        // vertex set are untouched. Mutating `tris` here (before face creation,
        // hence before the §7 dissolve in `triangle_fill`) is what satisfies
        // §6's "beautify before dissolve" ordering.
        if params.use_beauty {
            if has_holes {
                // The bridged keyhole is one polygon, but its sequence repeats
                // the two bridge endpoints and carries a zero-width slit, so the
                // simple-polygon index-adjacency perimeter rule is unsafe (it
                // could rotate a diagonal into the slit or across the hole).
                // Drive beautify with an explicit boundary predicate over the
                // region's *actual* loop edges (outer loop + every hole rim),
                // keyed on VertexKey so a vertex repeated at several sweep
                // indices is protected at all of them. Beautify then only
                // rotates genuine interior diagonals (§6).
                let boundary_pairs = self.triangle_fill_boundary_pairs(region_loops);
                let is_boundary = |a: u32, b: u32| {
                    let va = sweep_verts[a as usize];
                    let vb = sweep_verts[b as usize];
                    // A repeated bridge vertex at two distinct sweep indices is
                    // a zero-width slit edge — always fixed.
                    va == vb || boundary_pairs.contains(&unordered_pair(va, vb))
                };
                triangulate::beautify_polygon_protected(&projected, &mut tris, is_boundary);
            } else {
                beautify_polygon(&projected, &mut tris);
            }
        }

        // §4 — winding vote: only when the normal was derived and some input
        // edge already carries a face (rim-of-a-hole). The fill is already
        // consistently wound, so the decision is global.
        let flip = normal_was_derived && self.triangle_fill_winding_vote(&sweep_verts, &tris);

        // §5 — materialise each triangle with no-double edge creation. A
        // bridge introduces the same vertex twice in `sweep_verts`; a triangle
        // that would collapse onto a repeated vertex is degenerate (zero area
        // in the band) and is skipped rather than creating a self-edge.
        let mut interior_seen: std::collections::HashSet<EdgeKey> =
            interior_edges.iter().copied().collect();
        for tri in &tris {
            let (a, b, c) = if flip {
                (tri[0], tri[2], tri[1])
            } else {
                (tri[0], tri[1], tri[2])
            };
            let va = sweep_verts[a as usize];
            let vb = sweep_verts[b as usize];
            let vc = sweep_verts[c as usize];
            if va == vb || vb == vc || vc == va {
                continue;
            }

            let face = self.face_create_verts(&[va, vb, vc], FaceCreateFlags::NO_DUPLICATE);
            created_faces.push(face);

            for edge in self.face_edge_keys(face) {
                if interior_seen.contains(&edge) {
                    continue;
                }
                // A diagonal is interior iff it is not one of the input
                // boundary edges. Input edges pre-existed and must survive the
                // dissolve pass (§5/§7).
                if !boundary_edges.contains(&edge) {
                    interior_seen.insert(edge);
                    interior_edges.push(edge);
                }
            }
        }

        // §7/§8 — protect each bridge edge (the §1.2.4 splice that joins a hole
        // rim into the outer boundary) from the dissolve pass. A bridge separates
        // the band from the hole; JFKE-joining across it would merge the hole rim
        // into the outer disk and fill the hole. Keeping it as a surviving slit
        // edge leaves the dissolved annulus with its hole empty (χ = 0).
        for (va, vb) in bridge_pairs {
            if let Some(edge) = self.query_edge_exists(va, vb) {
                protected_edges.insert(edge);
            }
        }
    }

    /// §1.2.4 — splice every hole loop of a region into its outer loop with a
    /// bridge diagonal, producing a single simple-polygon vertex sequence. The
    /// bridge re-enters the outer chain so the band between outer and hole is
    /// fillable while the hole interior is left empty. Each hole is wound
    /// opposite to the outer loop (in projection) so the bridged polygon stays
    /// simple.
    ///
    /// Returns the spliced chain and, for each hole, the undirected VertexKey
    /// pair of its bridge endpoints (outer vertex, hole vertex). Those pairs are
    /// the bridge/slit edges the §7 dissolve must protect: a bridge separates the
    /// band from the hole, so joining across it would merge the hole rim into the
    /// outer disk and fill the hole (§8 requires the hole to stay empty).
    fn triangle_fill_bridge_holes(
        &self,
        region_loops: &[Vec<VertexKey>],
        project: &impl Fn(&LoopMesh, VertexKey) -> Vec2,
    ) -> (Vec<VertexKey>, Vec<(VertexKey, VertexKey)>) {
        let mut chain: Vec<VertexKey> = region_loops[0].clone();
        let mut bridge_pairs: Vec<(VertexKey, VertexKey)> = Vec::new();
        if region_loops.len() == 1 {
            return (chain, bridge_pairs);
        }

        let outer_ccw = self.triangle_fill_loop_is_ccw(&chain, project);
        for hole in &region_loops[1..] {
            if hole.len() < 3 {
                continue;
            }
            // Wind the hole opposite to the outer loop so the bridged polygon
            // is simple (the band stays on one side of the bridge).
            let mut hole_seq = hole.clone();
            if self.triangle_fill_loop_is_ccw(&hole_seq, project) == outer_ccw {
                hole_seq.reverse();
            }

            // Bridge at the closest outer/hole vertex pair (a short bridge is
            // the least likely to cross the band geometry).
            let (outer_index, hole_index) =
                self.triangle_fill_closest_pair(&chain, &hole_seq, project);
            bridge_pairs.push(unordered_pair(chain[outer_index], hole_seq[hole_index]));

            // Splice: ... outer[outer_index], hole[hole_index..], hole[hole_index],
            // outer[outer_index] ... — the repeated outer and hole vertices are
            // the two bridge endpoints, traversed out along the bridge, around
            // the hole, and back.
            let mut spliced: Vec<VertexKey> = Vec::with_capacity(chain.len() + hole_seq.len() + 2);
            spliced.extend_from_slice(&chain[..=outer_index]);
            for offset in 0..hole_seq.len() {
                spliced.push(hole_seq[(hole_index + offset) % hole_seq.len()]);
            }
            spliced.push(hole_seq[hole_index]);
            spliced.extend_from_slice(&chain[outer_index..]);
            chain = spliced;
        }
        (chain, bridge_pairs)
    }

    /// Collect every region-boundary edge as an undirected VertexKey pair: the
    /// outer loop's edges and every hole rim's edges (§1.2.4). The §6 beautify
    /// pass treats these as fixed — it may rotate only genuine interior
    /// diagonals, never a loop edge — which is what keeps a rotation from
    /// escaping the region or covering a hole.
    fn triangle_fill_boundary_pairs(
        &self,
        region_loops: &[Vec<VertexKey>],
    ) -> std::collections::HashSet<(VertexKey, VertexKey)> {
        let mut pairs = std::collections::HashSet::new();
        for loop_verts in region_loops {
            let n = loop_verts.len();
            if n < 2 {
                continue;
            }
            for i in 0..n {
                let a = loop_verts[i];
                let b = loop_verts[(i + 1) % n];
                pairs.insert(unordered_pair(a, b));
            }
        }
        pairs
    }

    /// Signed-area (shoelace) test on a loop's projected coordinates: true when
    /// the loop is wound counter-clockwise in the projection plane.
    fn triangle_fill_loop_is_ccw(
        &self,
        loop_verts: &[VertexKey],
        project: &impl Fn(&LoopMesh, VertexKey) -> Vec2,
    ) -> bool {
        let n = loop_verts.len();
        let mut twice_area = 0.0f32;
        for i in 0..n {
            let a = project(self, loop_verts[i]);
            let b = project(self, loop_verts[(i + 1) % n]);
            twice_area += a.x * b.y - b.x * a.y;
        }
        twice_area > 0.0
    }

    /// Index pair `(outer_index, hole_index)` minimising the projected
    /// distance between an outer-loop vertex and a hole-loop vertex — the
    /// bridge endpoints for [`Self::triangle_fill_bridge_holes`].
    fn triangle_fill_closest_pair(
        &self,
        outer: &[VertexKey],
        hole: &[VertexKey],
        project: &impl Fn(&LoopMesh, VertexKey) -> Vec2,
    ) -> (usize, usize) {
        let mut best = (0usize, 0usize);
        let mut best_dist = f32::INFINITY;
        for (oi, &ov) in outer.iter().enumerate() {
            let op = project(self, ov);
            for (hi, &hv) in hole.iter().enumerate() {
                let dist = op.distance_squared(project(self, hv));
                if dist < best_dist {
                    best_dist = dist;
                    best = (oi, hi);
                }
            }
        }
        best
    }

    /// §1.2.1–§1.2.3 — decompose the input edge net into fill regions. Each
    /// returned region is a list of closed loops: index 0 the outer loop, the
    /// rest enclosed holes. Loose edges and dangling chains (§1.2.2) are
    /// trimmed and contribute no region.
    fn triangle_fill_discover_regions(&self, edges: &[EdgeKey]) -> Vec<Vec<Vec<VertexKey>>> {
        // §1.2.2 — iterative leaf-trim: repeatedly drop every edge incident to
        // a valence-1 vertex until only edges that lie on a cycle survive.
        let mut surviving: Vec<EdgeKey> = Vec::new();
        let mut degree: HashMap<VertexKey, u32> = HashMap::new();
        for &edge in edges {
            let [v0, v1] = self.edge_vert_keys(edge);
            if v0 == v1 {
                continue;
            }
            surviving.push(edge);
            *degree.entry(v0).or_default() += 1;
            *degree.entry(v1).or_default() += 1;
        }
        let mut changed = true;
        while changed {
            changed = false;
            let mut kept: Vec<EdgeKey> = Vec::with_capacity(surviving.len());
            for &edge in &surviving {
                let [v0, v1] = self.edge_vert_keys(edge);
                if degree[&v0] == 1 || degree[&v1] == 1 {
                    *degree.get_mut(&v0).unwrap() -= 1;
                    *degree.get_mut(&v1).unwrap() -= 1;
                    changed = true;
                } else {
                    kept.push(edge);
                }
            }
            surviving = kept;
        }
        if surviving.is_empty() {
            return Vec::new();
        }

        // §1.2.1 — connected-component labelling over the surviving cycle
        // edges, then order each component into its closed loops.
        let mut component_of: HashMap<VertexKey, usize> = HashMap::new();
        let mut adjacency: HashMap<VertexKey, smallvec::SmallVec<[VertexKey; 4]>> = HashMap::new();
        for &edge in &surviving {
            let [v0, v1] = self.edge_vert_keys(edge);
            adjacency.entry(v0).or_default().push(v1);
            adjacency.entry(v1).or_default().push(v0);
        }
        let mut component_loops: Vec<Vec<Vec<VertexKey>>> = Vec::new();
        let mut stack: Vec<VertexKey> = Vec::new();
        for &edge in &surviving {
            let [seed, _] = self.edge_vert_keys(edge);
            if component_of.contains_key(&seed) {
                continue;
            }
            let component_index = component_loops.len();
            let mut component_verts: Vec<VertexKey> = Vec::new();
            stack.push(seed);
            component_of.insert(seed, component_index);
            while let Some(v) = stack.pop() {
                component_verts.push(v);
                for &neighbour in &adjacency[&v] {
                    if !component_of.contains_key(&neighbour) {
                        component_of.insert(neighbour, component_index);
                        stack.push(neighbour);
                    }
                }
            }
            let loops = self.triangle_fill_extract_loops(&component_verts, &adjacency);
            component_loops.push(loops);
        }

        // §1.2.3 — coalesce components whose 2D bounding boxes overlap into one
        // region (no point-in-polygon / nesting test). The projection plane is
        // derived per-region later; for the box test, project on whichever
        // plane the region's first loop derives — close enough for the overlap
        // decision, which only chooses which loops share a sweep.
        self.triangle_fill_coalesce_regions(component_loops)
    }

    /// Order one component's vertices into its constituent closed loops. A
    /// component is usually one loop; a figure-eight or multi-cycle component
    /// yields several. Each loop walks degree-2 chains; higher-degree junctions
    /// pick an unused incident edge.
    fn triangle_fill_extract_loops(
        &self,
        component_verts: &[VertexKey],
        adjacency: &HashMap<VertexKey, smallvec::SmallVec<[VertexKey; 4]>>,
    ) -> Vec<Vec<VertexKey>> {
        // Track unused undirected edges as a multiset of neighbour slots.
        let mut remaining: HashMap<VertexKey, Vec<VertexKey>> = HashMap::new();
        for &v in component_verts {
            remaining.insert(v, adjacency[&v].to_vec());
        }
        let take_edge =
            |remaining: &mut HashMap<VertexKey, Vec<VertexKey>>, from: VertexKey, to: VertexKey| {
                if let Some(slots) = remaining.get_mut(&from) {
                    if let Some(pos) = slots.iter().position(|&n| n == to) {
                        slots.swap_remove(pos);
                    }
                }
                if let Some(slots) = remaining.get_mut(&to) {
                    if let Some(pos) = slots.iter().position(|&n| n == from) {
                        slots.swap_remove(pos);
                    }
                }
            };

        let mut loops: Vec<Vec<VertexKey>> = Vec::new();
        for &start in component_verts {
            while !remaining[&start].is_empty() {
                let mut loop_verts: Vec<VertexKey> = vec![start];
                let first_step = remaining[&start][0];
                take_edge(&mut remaining, start, first_step);
                let mut next = first_step;
                while next != start {
                    loop_verts.push(next);
                    let following = remaining
                        .get(&next)
                        .and_then(|slots| slots.first().copied());
                    let Some(following) = following else {
                        break;
                    };
                    take_edge(&mut remaining, next, following);
                    next = following;
                }
                if loop_verts.len() >= 3 && next == start {
                    loops.push(loop_verts);
                }
            }
        }
        loops
    }

    /// §1.2.3 — group components into regions by transitive 2D bounding-box
    /// overlap. The region's loops are concatenated with the largest-box loop
    /// first (the outer loop); the rest are treated as holes by the fill.
    fn triangle_fill_coalesce_regions(
        &self,
        component_loops: Vec<Vec<Vec<VertexKey>>>,
    ) -> Vec<Vec<Vec<VertexKey>>> {
        let component_count = component_loops.len();

        // Derive one shared projection basis from the first non-empty loop so
        // the box test is computed in a consistent plane. The overlap decision
        // only chooses which loops share a sweep, so the exact plane is not
        // load-bearing (§1.2.3).
        let mut basis_normal = Vec3::Z;
        for loops in &component_loops {
            if let Some(first) = loops.first() {
                if first.len() >= 3 {
                    basis_normal = self.triangle_fill_derive_normal(first);
                    break;
                }
            }
        }
        let (axis_u, axis_w) = axis_dominant_basis_negated(basis_normal);
        let project = |mesh: &LoopMesh, v: VertexKey| {
            let p = mesh.vertex_position(v);
            Vec2::new(axis_u.dot(p), axis_w.dot(p))
        };

        let mut boxes: Vec<BoundingBox2D> = vec![BoundingBox2D::empty(); component_count];
        for (i, loops) in component_loops.iter().enumerate() {
            for loop_verts in loops {
                for &v in loop_verts {
                    boxes[i].expand(project(self, v));
                }
            }
        }

        // Union-find by transitive box overlap.
        let mut target: Vec<usize> = (0..component_count).collect();
        fn find(target: &mut [usize], mut i: usize) -> usize {
            while target[i] != i {
                target[i] = target[target[i]];
                i = target[i];
            }
            i
        }
        for a in 0..component_count {
            for b in (a + 1)..component_count {
                if boxes[a].overlaps(&boxes[b]) {
                    let ra = find(&mut target, a);
                    let rb = find(&mut target, b);
                    if ra != rb {
                        target[rb] = ra;
                        let merged = boxes[rb];
                        boxes[ra].union(&merged);
                    }
                }
            }
        }

        // Gather each region's loops; order so the largest-box loop is first
        // (the outer boundary, with the rest as holes).
        let mut regions: HashMap<usize, Vec<Vec<VertexKey>>> = HashMap::new();
        for (i, loops) in component_loops.into_iter().enumerate() {
            let root = find(&mut target, i);
            regions.entry(root).or_default().extend(loops);
        }
        let mut out: Vec<Vec<Vec<VertexKey>>> = Vec::new();
        for (_, mut loops) in regions {
            loops.retain(|l| l.len() >= 3);
            if loops.is_empty() {
                continue;
            }
            loops.sort_by(|a, b| {
                let box_a = loop_box(self, a, &project);
                let box_b = loop_box(self, b, &project);
                box_b
                    .area()
                    .partial_cmp(&box_a.area())
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            out.push(loops);
        }
        out
    }

    /// Order an undirected edge set into a single simple closed loop's vertex
    /// cycle, or `None` if the edges do not form exactly one such loop (every
    /// vertex must have degree exactly two within the set, and the set must be
    /// a single connected cycle covering all its edges).
    fn triangle_fill_order_single_loop(&self, edges: &[EdgeKey]) -> Option<Vec<VertexKey>> {
        let mut adjacency: HashMap<VertexKey, smallvec::SmallVec<[VertexKey; 2]>> = HashMap::new();
        for &edge in edges {
            let [v0, v1] = self.edge_vert_keys(edge);
            if v0 == v1 {
                return None;
            }
            adjacency.entry(v0).or_default().push(v1);
            adjacency.entry(v1).or_default().push(v0);
        }
        // A single simple cycle has every vertex at degree exactly two.
        if adjacency.values().any(|neighbors| neighbors.len() != 2) {
            return None;
        }

        let start = *adjacency.keys().next()?;
        let mut ordered: Vec<VertexKey> = Vec::with_capacity(adjacency.len());
        let mut current = start;
        let mut previous: Option<VertexKey> = None;
        loop {
            ordered.push(current);
            let neighbors = &adjacency[&current];
            let next = if Some(neighbors[0]) == previous {
                neighbors[1]
            } else {
                neighbors[0]
            };
            previous = Some(current);
            current = next;
            if current == start {
                break;
            }
            // Unreachable safety net: a degree-2 net traversed from `start`
            // either closes back to `start` within `adjacency.len()` steps or is
            // a disjoint cycle (rejected by the post-loop coverage check). This
            // guards against an unbounded walk should that invariant ever break.
            if ordered.len() > adjacency.len() {
                return None;
            }
        }
        // The walk closed back to `start` — reject if it did not cover every
        // vertex of the net (disjoint cycles).
        if ordered.len() != adjacency.len() {
            return None;
        }
        Some(ordered)
    }

    /// Derive a robust fill normal from the loop's corner turns (§3).
    fn triangle_fill_derive_normal(&self, loop_verts: &[VertexKey]) -> Vec3 {
        let n = loop_verts.len();

        // §3 step 1-2: per-corner cross product, sign-normalised by a fixed,
        // order-independent edge-length rule so the same corner always yields
        // the same-signed normal regardless of input edge order.
        let mut corner_normals: Vec<(f32, Vec3)> = Vec::with_capacity(n);
        for i in 0..n {
            let prev = self.vertex_position(loop_verts[(i + n - 1) % n]);
            let here = self.vertex_position(loop_verts[i]);
            let next = self.vertex_position(loop_verts[(i + 1) % n]);
            let incoming = here - prev;
            let outgoing = next - here;
            let mut corner = incoming.cross(outgoing);
            // Order-independent sign rule: flip so the corner normal is
            // oriented consistently for the unordered pair {incoming, outgoing}.
            // The tie-broken convention keys off the two edge lengths.
            let incoming_len = incoming.length_squared();
            let outgoing_len = outgoing.length_squared();
            if outgoing_len < incoming_len {
                corner = -corner;
            }
            corner_normals.push((corner.length_squared(), corner));
        }

        // §3 step 3: seed from the squarest corner, align the rest.
        corner_normals.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        let mut accumulated = Vec3::ZERO;
        for (magnitude, corner) in &corner_normals {
            if *magnitude <= MIN_NORMAL_LEN_SQ {
                continue;
            }
            let oriented = if accumulated.dot(*corner) < 0.0 {
                -*corner
            } else {
                *corner
            };
            accumulated += oriented;
        }

        // §3 step 4: no two-edge corner contributed — average stored normals.
        if accumulated.length_squared() <= MIN_NORMAL_LEN_SQ {
            for &v in loop_verts {
                accumulated += self.vertex_normal(v);
            }
        }

        // §3 step 5: final degenerate guard → +Z.
        if accumulated.length_squared() <= MIN_NORMAL_LEN_SQ {
            Vec3::Z
        } else {
            accumulated.normalize()
        }
    }

    /// §4 winding vote: returns true if the whole fill should be flipped to
    /// agree with the existing surface across the shared boundary edges.
    /// Fires only when some input edge already carries exactly one face.
    fn triangle_fill_winding_vote(&self, loop_verts: &[VertexKey], tris: &[[u32; 3]]) -> bool {
        let mut tally: i32 = 0;
        for tri in tris {
            // Each directed corner pair (a→b) of the fill triangle.
            for &(a, b) in &[(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])] {
                let va = loop_verts[a as usize];
                let vb = loop_verts[b as usize];
                let Some(edge) = self.query_edge_exists(va, vb) else {
                    continue;
                };
                // Only boundary input edges (exactly one existing face) vote.
                let radial = self.edge_loop_keys(edge);
                if radial.len() != 1 {
                    continue;
                }
                let neighbour_loop = radial[0];
                // The neighbour traverses the edge starting at this loop's
                // vertex. Coherent normals require the fill to traverse the
                // shared edge in the *opposite* direction to the neighbour.
                let neighbour_start = self.loop_vertex(neighbour_loop);
                if neighbour_start == va {
                    // Fill and neighbour traverse va→vb the same way →
                    // incoherent → vote to flip.
                    tally += 1;
                } else {
                    tally -= 1;
                }
            }
        }
        tally > 0
    }

    /// True when `face_a` and `face_b` share — besides `joining_edge` — at least
    /// one *protected* edge. Joining the pair across `joining_edge` would then
    /// strand that protected edge as a same-face filament (and the join surgery
    /// would dissolve it), filling a hole the §1.2.4 bridge was keeping open.
    fn faces_share_protected_edge(
        &self,
        face_a: FaceKey,
        face_b: FaceKey,
        joining_edge: EdgeKey,
        protected_edges: &std::collections::HashSet<EdgeKey>,
    ) -> bool {
        let edges_b: std::collections::HashSet<EdgeKey> =
            self.face_edge_keys(face_b).into_iter().collect();
        self.face_edge_keys(face_a).into_iter().any(|edge| {
            edge != joining_edge && protected_edges.contains(&edge) && edges_b.contains(&edge)
        })
    }

    /// §7 dissolve pass: JFKE-join the two fill triangles flanking each new
    /// manifold interior edge, removing the edge; report the surviving joined
    /// faces and any surviving fill edges in `geom.out`.
    fn triangle_fill_dissolve(
        &mut self,
        interior_edges: &[EdgeKey],
        created_faces: Vec<FaceKey>,
        protected_edges: &std::collections::HashSet<EdgeKey>,
        result: &mut TriangleFillResult,
    ) {
        // Seed with every fill face created during materialisation (§5); each
        // join removes the two consumed faces and inserts the survivor, so a
        // face never touched by an interior-edge join (e.g. the single triangle
        // of a 3-edge loop) still survives to geom.out (§1/§7).
        let mut surviving_faces: std::collections::HashSet<FaceKey> =
            created_faces.into_iter().collect();
        for &edge in interior_edges {
            // The edge may already be gone (killed when an earlier neighbour
            // edge's join removed it) — skip stale keys.
            if self.edges.try_get(edge).is_none() {
                continue;
            }
            // §7/§8 — a §1.2.4 bridge/slit edge separates the band from its
            // hole; JFKE-joining across it would fill the hole. Leave it as a
            // surviving fill edge so a dissolved annulus keeps its hole empty
            // (χ = 0). The no-holes path never protects any edge, so its
            // dissolve is unaffected.
            if protected_edges.contains(&edge) {
                result.edges.push(edge);
                continue;
            }
            let radial = self.edge_loop_keys(edge);
            match radial.len() {
                2 => {
                    let face_a = self.loop_face(radial[0]);
                    let face_b = self.loop_face(radial[1]);
                    if face_a == face_b {
                        // Same-face filament — joining is undefined; leave it.
                        result.edges.push(edge);
                        continue;
                    }
                    // §7/§8 — when the two flanking faces share a *protected*
                    // edge in addition to `edge` (the case where dissolving the
                    // band's last spoke would close the annular ring), the join
                    // would consume the protected §1.2.4 bridge as a same-face
                    // filament and fill the hole. The JFKE precondition (a clean
                    // single-shared-edge merge that introduces no internal
                    // duplicate vertex) is violated, so the join is refused and
                    // `edge` survives — leaving the dissolved annulus open
                    // (χ = 0). Only the with-holes path populates `protected_
                    // edges`, so the no-holes dissolve is unaffected.
                    if !protected_edges.is_empty()
                        && self.faces_share_protected_edge(face_a, face_b, edge, protected_edges)
                    {
                        result.edges.push(edge);
                        continue;
                    }
                    if let Some(joined) = self.faces_join(&[face_a, face_b]) {
                        surviving_faces.remove(&face_a);
                        surviving_faces.remove(&face_b);
                        surviving_faces.insert(joined);
                    } else {
                        result.edges.push(edge);
                    }
                }
                0 => {
                    // Loose interior edge (degenerate net) — kill it (§7).
                    self.edge_kill(edge);
                }
                _ => {
                    // One or three-plus faces — joining undefined; leave it.
                    result.edges.push(edge);
                }
            }
        }
        result.faces = surviving_faces.into_iter().collect();
    }
}

#[cfg(test)]
mod tests {
    use std::f32::consts::TAU;

    use glam::{Vec2, Vec3};

    use super::TriangleFillParams;
    use crate::loopmesh::*;

    /// Build a free wire ring of `n` verts on a unit circle in the z=0 plane,
    /// returning the ring edges in arbitrary (set) order.
    fn convex_wire_ring(mesh: &mut LoopMesh, n: usize) -> Vec<EdgeKey> {
        let verts: Vec<VertexKey> = (0..n)
            .map(|i| {
                let theta = (i as f32) * TAU / (n as f32);
                mesh.create_vertex(Vec3::new(theta.cos(), theta.sin(), 0.0))
            })
            .collect();
        (0..n)
            .map(|i| {
                mesh.create_edge(
                    [verts[i], verts[(i + 1) % n]],
                    EdgeCreateFlags::NO_DUPLICATE,
                )
            })
            .collect()
    }

    #[test]
    fn convex_hexagon_dissolve_to_single_ngon() {
        let mut mesh = LoopMesh::new();
        let edges = convex_wire_ring(&mut mesh, 6);
        let result = mesh.triangle_fill(
            &edges,
            &TriangleFillParams {
                use_beauty: false,
                use_dissolve: true,
                normal: None,
            },
        );
        assert_eq!(mesh.vertex_count(), 6, "no vertex created");
        assert_eq!(mesh.edge_count(), 6, "only boundary edges survive");
        assert_eq!(mesh.face_count(), 1, "convex loop dissolves to 1 n-gon");
        let face = mesh.face_keys()[0];
        assert_eq!(mesh.face_len(face), 6, "the n-gon spans all six corners");
        assert_eq!(
            result.faces().len(),
            1,
            "geom.out reports the single surviving dissolved n-gon"
        );
        mesh.validate_topology();
    }

    #[test]
    fn triangle_loop_dissolve_reports_single_fill_face() {
        // A 3-edge loop fills to one triangle and has *zero* interior
        // diagonals, so the dissolve pass joins nothing — the lone fill face
        // must still be reported in geom.out.
        let mut mesh = LoopMesh::new();
        let edges = convex_wire_ring(&mut mesh, 3);
        let result = mesh.triangle_fill(
            &edges,
            &TriangleFillParams {
                use_beauty: false,
                use_dissolve: true,
                normal: None,
            },
        );
        assert_eq!(mesh.vertex_count(), 3, "no vertex created");
        assert_eq!(mesh.edge_count(), 3, "only the three boundary edges");
        assert_eq!(mesh.face_count(), 1, "one fill triangle, live in the mesh");
        assert_eq!(
            result.faces().len(),
            1,
            "geom.out must report the lone fill triangle (no interior edges to join)"
        );
        mesh.validate_topology();
    }

    #[test]
    fn use_beauty_does_not_panic_and_fills() {
        // use_beauty is a caller-settable input: it must not abort, and the
        // beautified fill must still be a valid triangulation with the boundary
        // preserved.
        let mut mesh = LoopMesh::new();
        let edges = convex_wire_ring(&mut mesh, 6);
        let result = mesh.triangle_fill(
            &edges,
            &TriangleFillParams {
                use_beauty: true,
                use_dissolve: false,
                normal: None,
            },
        );
        assert_eq!(mesh.vertex_count(), 6, "no vertex created");
        assert_eq!(mesh.face_count(), 4, "hexagon → n-2 = 4 triangles");
        assert_eq!(result.faces().len(), 4);
        for face in mesh.face_keys() {
            assert_eq!(mesh.face_len(face), 3, "every fill face is a triangle");
        }
        // All six boundary edges survive (the beautify pass only re-routes
        // interior diagonals, never boundary edges).
        for &boundary in &edges {
            assert!(
                mesh.edges.try_get(boundary).is_some(),
                "boundary edge {boundary:?} must survive beautify"
            );
        }
        mesh.validate_topology();
    }

    /// A "thin" non-convex quad-ish loop whose raw ear-clip diagonal differs
    /// from the beautified (Delaunay-like) diagonal. The four corners sit on a
    /// very flat rhombus: the horizontal diagonal is far shorter than the
    /// vertical one, so a quality-driven rotation prefers the short diagonal.
    fn thin_quad_ring(mesh: &mut LoopMesh) -> [VertexKey; 4] {
        // Corners 0,1,2,3 in CCW order. 0 and 2 are close together on the x
        // axis (short diagonal 0-2); 1 and 3 are far apart on the y axis (long
        // diagonal 1-3).
        let v0 = mesh.create_vertex(Vec3::new(-0.2, 0.0, 0.0));
        let v1 = mesh.create_vertex(Vec3::new(0.0, 3.0, 0.0));
        let v2 = mesh.create_vertex(Vec3::new(0.2, 0.0, 0.0));
        let v3 = mesh.create_vertex(Vec3::new(0.0, -3.0, 0.0));
        [v0, v1, v2, v3]
    }

    /// Returns the single interior diagonal of a quad fill as the unordered
    /// vertex pair `{a, b}` plus the live mesh: a fill of a 4-vertex loop emits
    /// 2 triangles sharing exactly one interior diagonal. `quad` is the four
    /// corner verts in CCW order, so `quad[0]`/`quad[2]` is the short diagonal
    /// and `quad[1]`/`quad[3]` the long one.
    fn quad_fill_diagonal(use_beauty: bool) -> ([VertexKey; 4], VertexKey, VertexKey, LoopMesh) {
        let mut mesh = LoopMesh::new();
        let quad = thin_quad_ring(&mut mesh);
        let [v0, v1, v2, v3] = quad;
        let e0 = mesh.create_edge([v0, v1], EdgeCreateFlags::NO_DUPLICATE);
        let e1 = mesh.create_edge([v1, v2], EdgeCreateFlags::NO_DUPLICATE);
        let e2 = mesh.create_edge([v2, v3], EdgeCreateFlags::NO_DUPLICATE);
        let e3 = mesh.create_edge([v3, v0], EdgeCreateFlags::NO_DUPLICATE);
        let result = mesh.triangle_fill(
            &[e0, e1, e2, e3],
            &TriangleFillParams {
                use_beauty,
                use_dissolve: false,
                normal: None,
            },
        );
        assert_eq!(result.faces().len(), 2, "quad → 2 triangles");
        assert_eq!(
            result.edges().len(),
            1,
            "quad fill has one interior diagonal"
        );
        let diagonal = result.edges()[0];
        let [a, b] = mesh.edge_vert_keys(diagonal);
        (quad, a, b, mesh)
    }

    #[test]
    fn use_beauty_rotates_interior_diagonal() {
        // The thin quad's two diagonals are very different lengths: the short
        // one is quad[0]-quad[2] (|x| ≈ 0.4), the long one quad[1]-quad[3]
        // (|y| = 6). The beautify pass must always settle on the short,
        // Delaunay-preferred diagonal regardless of which one the raw ear-clip
        // happened to pick (its starting corner is set-ordering dependent).
        let (quad, beauty_a, beauty_b, beauty_mesh) = quad_fill_diagonal(true);
        let short_pair = (quad[0], quad[2]);
        let on_short = (beauty_a == short_pair.0 && beauty_b == short_pair.1)
            || (beauty_a == short_pair.1 && beauty_b == short_pair.0);
        assert!(
            on_short,
            "beautify must land on the short (Delaunay-preferred) diagonal"
        );
        let beauty_len = beauty_mesh
            .vertex_position(beauty_a)
            .distance(beauty_mesh.vertex_position(beauty_b));
        let long_len = beauty_mesh
            .vertex_position(quad[1])
            .distance(beauty_mesh.vertex_position(quad[3]));
        assert!(
            beauty_len < long_len,
            "beautified diagonal ({beauty_len}) must be the short one, not the long ({long_len})"
        );
        beauty_mesh.validate_topology();
    }

    /// Fill a fresh n-gon ring with the supplied projection normal and return
    /// each fill face's normal projected onto the supplied axis. The ring is
    /// rebuilt each call so vertex/edge ordering is identical across calls —
    /// only the supplied normal differs.
    fn ring_fill_face_dots(n: usize, supplied: Vec3) -> Vec<f32> {
        let mut mesh = LoopMesh::new();
        let edges = convex_wire_ring(&mut mesh, n);
        let result = mesh.triangle_fill(
            &edges,
            &TriangleFillParams {
                use_beauty: false,
                use_dissolve: false,
                normal: Some(supplied),
            },
        );
        assert_eq!(result.faces().len(), n - 2, "n-gon → n-2 triangles");
        mesh.validate_topology();
        result
            .faces()
            .iter()
            .map(|&face| {
                mesh.face_normal_update(face);
                mesh.face_normal(face).dot(supplied.normalize())
            })
            .collect()
    }

    #[test]
    fn explicit_normal_supplied_projects_and_winds_consistently() {
        // A supplied non-zero normal is authoritative (§3/§4): it is used
        // directly as the projection plane, the §4 winding vote is skipped, and
        // §3 fixes the default orientation so the fill *faces the supplied
        // normal* (wound CCW seen from +normal). Every fill face's normal must
        // therefore agree with the supplied normal (positive dot), and the fill
        // must be consistently wound.
        let dots = ring_fill_face_dots(5, Vec3::new(0.0, 0.3, 1.0));
        assert!(
            dots.iter().all(|d| *d > 0.0),
            "fill must face the supplied normal (positive dot): {dots:?}"
        );
    }

    #[test]
    fn explicit_normal_sign_flips_fill_orientation() {
        // §3: the supplied normal's sign drives the fill orientation. A +Z
        // normal yields fill faces pointing +Z; a −Z normal yields faces
        // pointing −Z. Flipping the supplied normal flips the fill.
        for &n in &[5usize, 6] {
            let plus = ring_fill_face_dots(n, Vec3::Z);
            assert!(
                plus.iter().all(|d| *d > 0.0),
                "+Z supplied normal must make fill faces point +Z: {plus:?}"
            );

            let minus = ring_fill_face_dots(n, -Vec3::Z);
            assert!(
                minus.iter().all(|d| *d > 0.0),
                "−Z supplied normal must make fill faces point −Z: {minus:?}"
            );

            // Cross-check the raw (un-projected) Z component flips sign with the
            // supplied normal: +Z fill faces have positive Z, −Z negative Z.
            let mut mesh_plus = LoopMesh::new();
            let edges_plus = convex_wire_ring(&mut mesh_plus, n);
            let res_plus = mesh_plus.triangle_fill(
                &edges_plus,
                &TriangleFillParams {
                    use_beauty: false,
                    use_dissolve: false,
                    normal: Some(Vec3::Z),
                },
            );
            let mut mesh_minus = LoopMesh::new();
            let edges_minus = convex_wire_ring(&mut mesh_minus, n);
            let res_minus = mesh_minus.triangle_fill(
                &edges_minus,
                &TriangleFillParams {
                    use_beauty: false,
                    use_dissolve: false,
                    normal: Some(-Vec3::Z),
                },
            );
            for &face in res_plus.faces() {
                mesh_plus.face_normal_update(face);
                assert!(mesh_plus.face_normal(face).z > 0.0, "+Z fill face faces +Z");
            }
            for &face in res_minus.faces() {
                mesh_minus.face_normal_update(face);
                assert!(
                    mesh_minus.face_normal(face).z < 0.0,
                    "−Z fill face faces −Z"
                );
            }
        }
    }

    #[test]
    fn convex_pentagon_dissolve_to_single_ngon() {
        let mut mesh = LoopMesh::new();
        let edges = convex_wire_ring(&mut mesh, 5);
        mesh.triangle_fill(
            &edges,
            &TriangleFillParams {
                use_beauty: false,
                use_dissolve: true,
                normal: None,
            },
        );
        assert_eq!(mesh.vertex_count(), 5, "no vertex created");
        assert_eq!(mesh.edge_count(), 5, "only boundary edges survive");
        assert_eq!(mesh.face_count(), 1, "odd-arity loop dissolves to 1 n-gon");
        assert_eq!(mesh.face_len(mesh.face_keys()[0]), 5);
        mesh.validate_topology();
    }

    #[test]
    fn raw_fill_emits_n_minus_two_triangles() {
        let mut mesh = LoopMesh::new();
        let n = 7;
        let edges = convex_wire_ring(&mut mesh, n);
        let result = mesh.triangle_fill(
            &edges,
            &TriangleFillParams {
                use_beauty: false,
                use_dissolve: false,
                normal: None,
            },
        );
        assert_eq!(mesh.vertex_count(), n, "no vertex created");
        assert_eq!(mesh.face_count(), n - 2, "simple n-gon → n-2 triangles");
        for face in mesh.face_keys() {
            assert_eq!(mesh.face_len(face), 3, "every fill face is a triangle");
        }
        // n boundary edges survive; the raw fill adds n-3 interior diagonals.
        assert_eq!(mesh.edge_count(), n + (n - 3));
        assert_eq!(
            result.edges().len(),
            n - 3,
            "geom.out reports only the new interior diagonals"
        );
        assert_eq!(result.faces().len(), n - 2);
        mesh.validate_topology();
    }

    #[test]
    fn fewer_than_three_edges_fills_nothing() {
        let mut mesh = LoopMesh::new();
        let v0 = mesh.create_vertex(Vec3::ZERO);
        let v1 = mesh.create_vertex(Vec3::X);
        let v2 = mesh.create_vertex(Vec3::Y);
        let e0 = mesh.create_edge([v0, v1], EdgeCreateFlags::NO_DUPLICATE);
        let e1 = mesh.create_edge([v1, v2], EdgeCreateFlags::NO_DUPLICATE);
        let result = mesh.triangle_fill(&[e0, e1], &TriangleFillParams::default());
        assert!(result.is_empty());
        assert_eq!(mesh.face_count(), 0);
    }

    /// Build a free wire ring of `n` verts on a circle of the given `radius`,
    /// centred at `center` in the z=0 plane, returning the ring edges.
    fn wire_ring_at(mesh: &mut LoopMesh, n: usize, center: Vec3, radius: f32) -> Vec<EdgeKey> {
        let verts: Vec<VertexKey> = (0..n)
            .map(|i| {
                let theta = (i as f32) * TAU / (n as f32);
                mesh.create_vertex(
                    center + Vec3::new(radius * theta.cos(), radius * theta.sin(), 0.0),
                )
            })
            .collect();
        (0..n)
            .map(|i| {
                mesh.create_edge(
                    [verts[i], verts[(i + 1) % n]],
                    EdgeCreateFlags::NO_DUPLICATE,
                )
            })
            .collect()
    }

    #[test]
    fn two_disjoint_loops_both_fill_independently() {
        // §1.2.1 / §8: two disjoint closed loops in one edge set — each is a
        // connected component (boxes do not overlap) and fills independently.
        let mut mesh = LoopMesh::new();
        let mut edges = wire_ring_at(&mut mesh, 4, Vec3::new(-5.0, 0.0, 0.0), 1.0);
        edges.extend(wire_ring_at(&mut mesh, 5, Vec3::new(5.0, 0.0, 0.0), 1.0));

        let result = mesh.triangle_fill(&edges, &TriangleFillParams::default());

        // quad → 2 tris, pentagon → 3 tris.
        assert_eq!(mesh.face_count(), 2 + 3, "both regions filled");
        assert_eq!(mesh.vertex_count(), 9, "no vertex created");
        assert_eq!(result.faces().len(), 5);
        for face in mesh.face_keys() {
            assert_eq!(mesh.face_len(face), 3, "every fill face is a triangle");
        }
        mesh.validate_topology();
    }

    #[test]
    fn annulus_fills_band_and_leaves_hole_empty() {
        // §1.2.3 / §1.2.4 / §8: an outer loop enclosing an inner loop. The two
        // loops coalesce (inner box ⊂ outer box) into one region; the band is
        // tiled and the inner loop is left empty (no face covers the hole).
        let mut mesh = LoopMesh::new();
        let outer = wire_ring_at(&mut mesh, 4, Vec3::ZERO, 4.0);
        let inner = wire_ring_at(&mut mesh, 4, Vec3::ZERO, 1.0);
        let mut edges = outer.clone();
        edges.extend(inner.iter().copied());

        let result = mesh.triangle_fill(&edges, &TriangleFillParams::default());

        assert_eq!(mesh.vertex_count(), 8, "no vertex created");
        assert!(!result.faces().is_empty(), "band must be filled");
        for face in mesh.face_keys() {
            assert_eq!(mesh.face_len(face), 3, "every band face is a triangle");
        }
        // The four inner edges still bound the empty hole: each carries exactly
        // one fill face (the band side), never two (which would mean the hole
        // got covered).
        for &inner_edge in &inner {
            let face_count = mesh.edge_loop_keys(inner_edge).len();
            assert_eq!(
                face_count, 1,
                "inner edge {inner_edge:?} must have the band on one side only (hole empty)"
            );
        }
        // No fill triangle may have all three corners on the inner loop — that
        // would be a triangle inside the hole.
        let inner_verts: std::collections::HashSet<VertexKey> =
            inner.iter().flat_map(|&e| mesh.edge_vert_keys(e)).collect();
        for face in mesh.face_keys() {
            let all_inner = mesh
                .face_vert_keys(face)
                .iter()
                .all(|v| inner_verts.contains(v));
            assert!(!all_inner, "no triangle may fill the hole interior");
        }
        mesh.validate_topology();
    }

    #[test]
    fn annulus_dissolve_keeps_hole_empty() {
        // §7/§8 + §1.2.4: a dissolved annulus must keep its hole empty. The
        // dissolve pass JFKE-joins the band's interior diagonals into n-gons,
        // but it must NOT join across the §1.2.4 bridge/slit edge that splices
        // the hole rim into the outer boundary — joining there would merge the
        // hole rim into the outer disk and fill the hole (χ would collapse from
        // 0 to 1). The surviving annulus has χ = 0.
        let mut mesh = LoopMesh::new();
        let outer = wire_ring_at(&mut mesh, 4, Vec3::ZERO, 4.0);
        let inner = wire_ring_at(&mut mesh, 4, Vec3::ZERO, 1.0);
        let mut edges = outer.clone();
        edges.extend(inner.iter().copied());

        let result = mesh.triangle_fill(
            &edges,
            &TriangleFillParams {
                use_beauty: false,
                use_dissolve: true,
                normal: None,
            },
        );

        assert_eq!(mesh.vertex_count(), 8, "no vertex created");
        assert!(!result.faces().is_empty(), "dissolved band must survive");

        // The four inner edges still bound the empty hole: each carries exactly
        // one face (the band side), never two — the hole stays empty after the
        // dissolve.
        for &inner_edge in &inner {
            let face_count = mesh.edge_loop_keys(inner_edge).len();
            assert_eq!(
                face_count, 1,
                "inner edge {inner_edge:?} keeps the band on one side only (hole empty)"
            );
        }
        // The four outer edges likewise survive verbatim (input edges are never
        // dissolved, §7).
        for &outer_edge in &outer {
            assert!(
                mesh.edges.try_get(outer_edge).is_some(),
                "outer boundary edge {outer_edge:?} must survive the dissolve"
            );
        }

        // No surviving face may have all corners on the inner loop — that would
        // mean the hole got covered.
        let inner_verts: std::collections::HashSet<VertexKey> =
            inner.iter().flat_map(|&e| mesh.edge_vert_keys(e)).collect();
        for face in mesh.face_keys() {
            let all_inner = mesh
                .face_vert_keys(face)
                .iter()
                .all(|v| inner_verts.contains(v));
            assert!(!all_inner, "no surviving face may fill the hole interior");
        }

        // χ = V − E + F = 0 for a topological annulus (one face per side of the
        // band's loop, two boundary loops, hole empty). A filled hole would push
        // χ to 1.
        let chi = mesh.vertex_count() as i64 - mesh.edge_count() as i64 + mesh.face_count() as i64;
        assert_eq!(
            chi, 0,
            "dissolved annulus has Euler characteristic 0 (hole empty)"
        );

        // The band dissolves as far as the manifold-join rule allows (§7/§8):
        // the §1.2.4 bridge survives, and so does the one ring-closing spoke
        // whose join would have consumed the bridge as a filament (closing the
        // annular ring would fill the hole). The band ends up as two large
        // faces separated by bridge + spoke. 8 verts, 8 boundary + 2 surviving
        // interior = 10 edges, 2 faces (χ = 8 − 10 + 2 = 0).
        assert_eq!(
            mesh.edge_count(),
            10,
            "8 boundary edges + the bridge + the ring-closing spoke"
        );
        assert_eq!(
            mesh.face_count(),
            2,
            "the band dissolves to two large n-gons, not one ring"
        );

        mesh.validate_topology();
    }

    /// Build an axis-aligned rectangular annulus: an outer rectangle ring and a
    /// concentric inner rectangle ring (the hole), returning `(outer, inner)`
    /// edge sets. The four corners of each ring are placed explicitly so the
    /// band quads can be made elongated (driving the beautify contrast).
    fn rect_annulus(
        mesh: &mut LoopMesh,
        outer_half: Vec2,
        inner_half: Vec2,
    ) -> (Vec<EdgeKey>, Vec<EdgeKey>) {
        let ring = |mesh: &mut LoopMesh, h: Vec2| {
            let corners = [
                Vec3::new(-h.x, -h.y, 0.0),
                Vec3::new(h.x, -h.y, 0.0),
                Vec3::new(h.x, h.y, 0.0),
                Vec3::new(-h.x, h.y, 0.0),
            ];
            let verts: Vec<VertexKey> = corners.iter().map(|&c| mesh.create_vertex(c)).collect();
            (0..4)
                .map(|i| {
                    mesh.create_edge(
                        [verts[i], verts[(i + 1) % 4]],
                        EdgeCreateFlags::NO_DUPLICATE,
                    )
                })
                .collect::<Vec<_>>()
        };
        let outer = ring(mesh, outer_half);
        let inner = ring(mesh, inner_half);
        (outer, inner)
    }

    /// Like [`rect_annulus`] but the two rectangles are given by explicit
    /// lo/hi corners, so the hole can be placed off-centre — which skews the
    /// band quads and makes the raw bridged ear-clip pick a non-Delaunay
    /// diagonal that beautify then rotates.
    fn rect_annulus_box(
        mesh: &mut LoopMesh,
        outer_lo: Vec2,
        outer_hi: Vec2,
        inner_lo: Vec2,
        inner_hi: Vec2,
    ) -> (Vec<EdgeKey>, Vec<EdgeKey>) {
        let ring = |mesh: &mut LoopMesh, lo: Vec2, hi: Vec2| {
            let corners = [
                Vec3::new(lo.x, lo.y, 0.0),
                Vec3::new(hi.x, lo.y, 0.0),
                Vec3::new(hi.x, hi.y, 0.0),
                Vec3::new(lo.x, hi.y, 0.0),
            ];
            let verts: Vec<VertexKey> = corners.iter().map(|&c| mesh.create_vertex(c)).collect();
            (0..4)
                .map(|i| {
                    mesh.create_edge(
                        [verts[i], verts[(i + 1) % 4]],
                        EdgeCreateFlags::NO_DUPLICATE,
                    )
                })
                .collect::<Vec<_>>()
        };
        let outer = ring(mesh, outer_lo, outer_hi);
        let inner = ring(mesh, inner_lo, inner_hi);
        (outer, inner)
    }

    /// The set of interior-diagonal endpoint pairs of a fill, keyed by rounded
    /// position so it is comparable across separate meshes. A diagonal is an
    /// edge of the live mesh that is *not* one of the supplied boundary edges.
    fn interior_diagonal_positions(
        mesh: &LoopMesh,
        boundary: &[EdgeKey],
    ) -> std::collections::HashSet<((i32, i32), (i32, i32))> {
        let boundary_set: std::collections::HashSet<EdgeKey> = boundary.iter().copied().collect();
        let key = |p: Vec3| ((p.x * 1000.0) as i32, (p.y * 1000.0) as i32);
        let mut out = std::collections::HashSet::new();
        for edge in mesh.edge_keys() {
            if boundary_set.contains(&edge) {
                continue;
            }
            let [a, b] = mesh.edge_vert_keys(edge);
            let ka = key(mesh.vertex_position(a));
            let kb = key(mesh.vertex_position(b));
            out.insert(if ka <= kb { (ka, kb) } else { (kb, ka) });
        }
        out
    }

    #[test]
    fn beauty_on_annulus_preserves_hole_and_boundary() {
        // §6 over a with-holes fill (§1.2.4): beautify must rotate only genuine
        // interior diagonals — never a loop edge, hole rim, or the bridge slit —
        // so the hole stays empty, every boundary/hole-rim edge survives, and
        // the band is still a valid all-triangle fill.
        let mut mesh = LoopMesh::new();
        let (outer, inner) = rect_annulus(&mut mesh, Vec2::new(4.0, 4.0), Vec2::new(1.0, 1.0));
        let mut edges = outer.clone();
        edges.extend(inner.iter().copied());

        let result = mesh.triangle_fill(
            &edges,
            &TriangleFillParams {
                use_beauty: true,
                use_dissolve: false,
                normal: None,
            },
        );

        assert_eq!(mesh.vertex_count(), 8, "beautify creates no vertex");
        assert!(!result.faces().is_empty(), "band must be filled");
        for face in mesh.face_keys() {
            assert_eq!(mesh.face_len(face), 3, "every band face is a triangle");
        }
        // Every outer-loop and hole-rim edge survives beautify untouched.
        for &boundary_edge in outer.iter().chain(inner.iter()) {
            assert!(
                mesh.edges.try_get(boundary_edge).is_some(),
                "boundary edge {boundary_edge:?} must survive beautify"
            );
        }
        // The hole stays empty: each inner edge carries exactly the band face on
        // one side, never two.
        for &inner_edge in &inner {
            assert_eq!(
                mesh.edge_loop_keys(inner_edge).len(),
                1,
                "inner edge {inner_edge:?} must keep the hole empty after beautify"
            );
        }
        // No triangle has all three corners on the inner loop (a hole-covering
        // triangle), and no fill edge connects two inner-loop vertices that are
        // not an original hole rim — that would be a diagonal rotated across the
        // hole.
        let inner_verts: std::collections::HashSet<VertexKey> =
            inner.iter().flat_map(|&e| mesh.edge_vert_keys(e)).collect();
        for face in mesh.face_keys() {
            let all_inner = mesh
                .face_vert_keys(face)
                .iter()
                .all(|v| inner_verts.contains(v));
            assert!(!all_inner, "beautify must not fill the hole interior");
        }
        let inner_rim: std::collections::HashSet<EdgeKey> = inner.iter().copied().collect();
        for edge in mesh.edge_keys() {
            let [a, b] = mesh.edge_vert_keys(edge);
            if inner_verts.contains(&a) && inner_verts.contains(&b) && !inner_rim.contains(&edge) {
                panic!("beautify rotated a diagonal across the hole: {edge:?}");
            }
        }
        mesh.validate_topology();
    }

    #[test]
    fn beauty_changes_annulus_raw_fill() {
        // Contrast: an annulus with a strongly *off-centre* hole (hugging the
        // left of a wide outer rectangle). The skewed band quads make the raw
        // bridged ear-clip pick a long across-the-band diagonal; beautify must
        // rotate at least one such diagonal toward the shorter, Delaunay-like
        // choice — so the two interior-diagonal sets must differ.
        let outer_lo = Vec2::new(-10.0, -3.0);
        let outer_hi = Vec2::new(10.0, 3.0);
        let inner_lo = Vec2::new(-9.0, -2.0);
        let inner_hi = Vec2::new(-2.0, 2.0);

        let mut raw_mesh = LoopMesh::new();
        let (raw_outer, raw_inner) =
            rect_annulus_box(&mut raw_mesh, outer_lo, outer_hi, inner_lo, inner_hi);
        let mut raw_edges = raw_outer.clone();
        raw_edges.extend(raw_inner.iter().copied());
        raw_mesh.triangle_fill(
            &raw_edges,
            &TriangleFillParams {
                use_beauty: false,
                use_dissolve: false,
                normal: None,
            },
        );
        let raw_boundary: Vec<EdgeKey> =
            raw_outer.iter().chain(raw_inner.iter()).copied().collect();
        let raw_diagonals = interior_diagonal_positions(&raw_mesh, &raw_boundary);
        raw_mesh.validate_topology();

        let mut beauty_mesh = LoopMesh::new();
        let (beauty_outer, beauty_inner) =
            rect_annulus_box(&mut beauty_mesh, outer_lo, outer_hi, inner_lo, inner_hi);
        let mut beauty_edges = beauty_outer.clone();
        beauty_edges.extend(beauty_inner.iter().copied());
        beauty_mesh.triangle_fill(
            &beauty_edges,
            &TriangleFillParams {
                use_beauty: true,
                use_dissolve: false,
                normal: None,
            },
        );
        let beauty_boundary: Vec<EdgeKey> = beauty_outer
            .iter()
            .chain(beauty_inner.iter())
            .copied()
            .collect();
        let beauty_diagonals = interior_diagonal_positions(&beauty_mesh, &beauty_boundary);
        beauty_mesh.validate_topology();

        assert_ne!(
            raw_diagonals, beauty_diagonals,
            "beautify must change the with-holes interior triangulation"
        );
        // The beautified fill is still a valid all-triangle annulus with an
        // empty hole (sanity, so "changed" can't mean "corrupted").
        for face in beauty_mesh.face_keys() {
            assert_eq!(beauty_mesh.face_len(face), 3);
        }
        for &inner_edge in &beauty_inner {
            assert_eq!(
                beauty_mesh.edge_loop_keys(inner_edge).len(),
                1,
                "hole stays empty in the beautified off-centre annulus"
            );
        }
    }

    #[test]
    fn loose_dangling_edges_create_nothing() {
        // §1.2.2: a closed triangle with a dangling tail. The tail is trimmed
        // by the leaf-trim pass; only the triangle fills. A *purely* dangling
        // chain (no cycle) fills nothing.
        let mut mesh = LoopMesh::new();

        // Pure dangling chain: a 3-edge open path off to one side.
        let p0 = mesh.create_vertex(Vec3::new(10.0, 0.0, 0.0));
        let p1 = mesh.create_vertex(Vec3::new(11.0, 0.0, 0.0));
        let p2 = mesh.create_vertex(Vec3::new(12.0, 0.0, 0.0));
        let p3 = mesh.create_vertex(Vec3::new(13.0, 0.5, 0.0));
        let d0 = mesh.create_edge([p0, p1], EdgeCreateFlags::NO_DUPLICATE);
        let d1 = mesh.create_edge([p1, p2], EdgeCreateFlags::NO_DUPLICATE);
        let d2 = mesh.create_edge([p2, p3], EdgeCreateFlags::NO_DUPLICATE);

        let result = mesh.triangle_fill(&[d0, d1, d2], &TriangleFillParams::default());
        assert!(result.is_empty(), "a pure dangling chain bounds no region");
        assert_eq!(mesh.face_count(), 0);

        // A closed loop plus a dangling spur off one of its vertices: the spur
        // is trimmed, the loop fills.
        let mut mesh2 = LoopMesh::new();
        let ring = wire_ring_at(&mut mesh2, 4, Vec3::ZERO, 1.0);
        let spur_tip = mesh2.create_vertex(Vec3::new(3.0, 0.0, 0.0));
        let ring_v = mesh2.edge_vert_keys(ring[0])[0];
        let spur = mesh2.create_edge([ring_v, spur_tip], EdgeCreateFlags::NO_DUPLICATE);
        let mut edges = ring.clone();
        edges.push(spur);

        let result2 = mesh2.triangle_fill(&edges, &TriangleFillParams::default());
        assert_eq!(mesh2.face_count(), 2, "the quad fills (spur trimmed)");
        assert_eq!(result2.faces().len(), 2);
        // The spur edge survives untouched (the op never deletes input edges).
        assert!(mesh2.edges.try_get(spur).is_some(), "spur edge untouched");
        mesh2.validate_topology();
    }

    #[test]
    fn open_chain_fills_nothing() {
        // Three edges that form an open path, not a closed loop.
        let mut mesh = LoopMesh::new();
        let v0 = mesh.create_vertex(Vec3::new(0.0, 0.0, 0.0));
        let v1 = mesh.create_vertex(Vec3::new(1.0, 0.0, 0.0));
        let v2 = mesh.create_vertex(Vec3::new(2.0, 0.0, 0.0));
        let v3 = mesh.create_vertex(Vec3::new(3.0, 0.0, 0.0));
        let e0 = mesh.create_edge([v0, v1], EdgeCreateFlags::NO_DUPLICATE);
        let e1 = mesh.create_edge([v1, v2], EdgeCreateFlags::NO_DUPLICATE);
        let e2 = mesh.create_edge([v2, v3], EdgeCreateFlags::NO_DUPLICATE);
        let result = mesh.triangle_fill(&[e0, e1, e2], &TriangleFillParams::default());
        assert!(result.is_empty(), "open chain bounds no region");
        assert_eq!(mesh.face_count(), 0);
    }
}
