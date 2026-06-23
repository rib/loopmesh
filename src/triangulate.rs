//! 2D polygon-triangulation primitives.
//!
//! The kernel here is independent of [`crate::loopmesh::LoopMesh`]: it
//! operates on `&[Vec2]` slices and yields triangle index triples into
//! that slice. The intended consumers are
//! [`crate::triangulate_corners`] (the non-destructive
//! corner-triple producer used by `to_nub_mesh`) and a future destructive
//! triangulate operator on [`LoopMesh`](crate::loopmesh::LoopMesh).
//!
//! Algorithms are documented in
//! `radial-mesh-guides/BMESH_POLYGON_TRIANGULATION.md`.

use std::collections::BinaryHeap;

use glam::{Vec2, Vec3};

/// Build the two orthonormal basis vectors used by the **negated**
/// axis-dominant projection from 3D to 2D (guide §3.1).
///
/// The projection of a 3D point `v` to 2D is `(u.dot(v), w.dot(v))`,
/// where `(u, w)` is the returned pair. `u` and `w` are unit length,
/// mutually orthogonal, and lie in the plane perpendicular to `normal`.
/// The pair, together with `-normal`, forms a right-handed orthonormal
/// frame. The resulting 2D embedding gives a clockwise winding to a 3D
/// face whose loops are CCW with respect to `normal`, matching the
/// ear-clipping kernel's expected input winding (§7.1).
///
/// `normal` is expected to be unit length. When `normal` is nearly the
/// zero vector or when its XY component is degenerate, a stable fallback
/// is selected so the result is still a valid orthonormal pair.
pub fn axis_dominant_basis_negated(normal: Vec3) -> (Vec3, Vec3) {
    ortho_basis_perp(-normal)
}

/// Build the two orthonormal basis vectors used by the **non-negated**
/// axis-dominant projection (guide §3.1).
///
/// Same shape as [`axis_dominant_basis_negated`] but applied to `+normal`
/// rather than `-normal`. The resulting frame is right-handed with the
/// third axis along `+normal`. This variant is used by the inlined 3D
/// beautify cost (§8.6).
pub fn axis_dominant_basis(normal: Vec3) -> (Vec3, Vec3) {
    ortho_basis_perp(normal)
}

fn ortho_basis_perp(n: Vec3) -> (Vec3, Vec3) {
    // u lives in the XY plane and is perpendicular to n's XY component
    // (rotated 90°, then normalised by that component's length). w is
    // the right-handed completion n × u.
    let xy_length_sq = n.x * n.x + n.y * n.y;
    if xy_length_sq > f32::EPSILON {
        let xy_length_inv = 1.0 / xy_length_sq.sqrt();
        let u = Vec3::new(n.y * xy_length_inv, -n.x * xy_length_inv, 0.0);
        let w = Vec3::new(-n.z * u.y, n.z * u.x, n.x * u.y - n.y * u.x);
        (u, w)
    } else {
        // n is essentially along ±z; the rotated-XY trick degenerates.
        // Fall back to the closest sign-stable axis-aligned right-handed
        // frame instead.
        let u_x_sign = if n.z < 0.0 { -1.0 } else { 1.0 };
        let u = Vec3::new(u_x_sign, 0.0, 0.0);
        let w = Vec3::new(0.0, 1.0, 0.0);
        (u, w)
    }
}

// ============================== Quad BEAUTY math ==============================

/// Diagonal-flip predicate for the §6.3 BEAUTY quad-method.
///
/// Vertices in cycle order. Returns a 2-bit flag:
/// - Bit 0 set when cutting along diagonal `v0-v2` produces two
///   triangles whose normals point in opposite directions (i.e. the
///   diagonal exits the quad through a concave corner).
/// - Bit 1 set when the same holds for diagonal `v1-v3`.
///
/// BEAUTY picks whichever diagonal does *not* flip; if both are valid
/// it falls through to a cost-based tie-break.
pub fn is_quad_flip(v0: Vec3, v1: Vec3, v2: Vec3, v3: Vec3) -> u32 {
    // The two triangles formed by a diagonal have same-side normals iff
    // the cross products of their two "free" pairs of consecutive edges
    // point the same way (dot product non-negative).
    let edge_v1_to_v0 = v0 - v1;
    let edge_v2_to_v1 = v1 - v2;
    let edge_v3_to_v2 = v2 - v3;
    let edge_v0_to_v3 = v3 - v0;

    let mut flip_mask = 0u32;

    // Diagonal v0-v2 splits the quad into (v0, v1, v2) and (v2, v3, v0).
    let normal_at_v1 = edge_v1_to_v0.cross(edge_v2_to_v1);
    let normal_at_v3 = edge_v3_to_v2.cross(edge_v0_to_v3);
    if normal_at_v1.dot(normal_at_v3) < 0.0 {
        flip_mask |= 1;
    }

    // Diagonal v1-v3 splits the quad into (v1, v2, v3) and (v3, v0, v1).
    let normal_at_v2 = edge_v2_to_v1.cross(edge_v3_to_v2);
    let normal_at_v0 = edge_v0_to_v3.cross(edge_v1_to_v0);
    if normal_at_v2.dot(normal_at_v0) < 0.0 {
        flip_mask |= 2;
    }

    flip_mask
}

/// 2D core of the §8.2 area-over-perimeter cost rule. Args follow the
/// §8.2 vertex convention — `(v1, v2, v3, v4)` — where the *existing*
/// diagonal under evaluation is `v2-v4` and the *candidate* diagonal
/// is `v1-v3`.
///
/// Returns the existing diagonal's score minus the candidate diagonal's
/// score: positive when keeping the existing diagonal is better,
/// negative when swapping to the candidate improves the triangulation.
/// Degenerate configurations per §8.3 return `±f32::MAX`.
pub fn beautify_quad_cost_2d(v1: Vec2, v2: Vec2, v3: Vec2, v4: Vec2, lock_degenerate: bool) -> f32 {
    beautify_quad_cost_2d_ex(v1, v2, v3, v4, lock_degenerate).0
}

/// Like [`beautify_quad_cost_2d`] but also returns the local quad area,
/// which the n-gon BEAUTY pass uses for its area-scaled epsilon threshold
/// during re-scoring.
pub fn beautify_quad_cost_2d_ex(
    v1: Vec2,
    v2: Vec2,
    v3: Vec2,
    v4: Vec2,
    lock_degenerate: bool,
) -> (f32, f32) {
    const ZERO_AREA_EPS: f32 = 1e-12;

    // Each diagonal splits the quad into a pair of corner triangles.
    // For each triangle we need its (doubled) signed 2D area both for
    // the §8.3 degeneracy guards and for the area-over-perimeter score.
    let existing_first_signed_area = signed_area_2x(v2, v3, v4);
    let existing_second_signed_area = signed_area_2x(v2, v4, v1);
    let candidate_first_signed_area = signed_area_2x(v1, v2, v3);
    let candidate_second_signed_area = signed_area_2x(v1, v3, v4);

    let quad_total_area = (existing_first_signed_area.abs()
        + existing_second_signed_area.abs()
        + candidate_first_signed_area.abs()
        + candidate_second_signed_area.abs())
        / 8.0;

    // §8.3: refuse the candidate when its two corner triangles disagree
    // on signed winding, or when either is below the zero-area epsilon.
    if (candidate_first_signed_area >= 0.0) != (candidate_second_signed_area >= 0.0)
        || candidate_first_signed_area.abs() <= ZERO_AREA_EPS
        || candidate_second_signed_area.abs() <= ZERO_AREA_EPS
    {
        return (f32::MAX, quad_total_area);
    }

    // §8.3: the existing diagonal's corner triangles disagreeing on
    // winding means the quad is already non-convex along the existing
    // split. With `lock_degenerate` the caller wants us to refuse the
    // swap (don't ever rotate out of a degenerate state when subsequent
    // swaps could flip it back); otherwise the swap is always preferred.
    if (existing_first_signed_area >= 0.0) != (existing_second_signed_area >= 0.0) {
        let result = if lock_degenerate { f32::MAX } else { -f32::MAX };
        return (result, quad_total_area);
    }
    if existing_first_signed_area.abs() <= ZERO_AREA_EPS
        || existing_second_signed_area.abs() <= ZERO_AREA_EPS
    {
        return (-f32::MAX, quad_total_area);
    }

    // Perimeter components. Each corner edge appears in exactly two of
    // the four corner triangles (one per diagonal split). Each diagonal
    // contributes its own perimeter component to both of its triangles.
    let perimeter_v1_to_v2 = v1.distance(v2);
    let perimeter_v2_to_v3 = v2.distance(v3);
    let perimeter_v3_to_v4 = v3.distance(v4);
    let perimeter_v4_to_v1 = v4.distance(v1);
    let candidate_diagonal_length = v1.distance(v3);
    let existing_diagonal_length = v2.distance(v4);

    let existing_score = diagonal_area_over_perimeter(
        existing_first_signed_area,
        existing_second_signed_area,
        perimeter_v2_to_v3 + perimeter_v3_to_v4 + existing_diagonal_length,
        perimeter_v4_to_v1 + perimeter_v1_to_v2 + existing_diagonal_length,
    );
    let candidate_score = diagonal_area_over_perimeter(
        candidate_first_signed_area,
        candidate_second_signed_area,
        perimeter_v1_to_v2 + perimeter_v2_to_v3 + candidate_diagonal_length,
        perimeter_v3_to_v4 + perimeter_v4_to_v1 + candidate_diagonal_length,
    );

    (existing_score - candidate_score, quad_total_area)
}

/// One diagonal's contribution to the §8.2 score: sum the two corner
/// triangles' `|signed area 2x| / perimeter`. Higher is better.
fn diagonal_area_over_perimeter(
    first_triangle_signed_area: f32,
    second_triangle_signed_area: f32,
    first_triangle_perimeter: f32,
    second_triangle_perimeter: f32,
) -> f32 {
    first_triangle_signed_area.abs() / first_triangle_perimeter
        + second_triangle_signed_area.abs() / second_triangle_perimeter
}

/// 3D entry point to the §8.2 cost rule. Args are in cycle order
/// `(v0, v1, v2, v3)`. The *existing* diagonal under evaluation is
/// `v0-v2`; the *candidate* is `v1-v3`. Positive return ⇒ keep 0-2;
/// negative ⇒ swap to 1-3.
///
/// Follows the §8.6 inlined-3D recipe: build the projection normal from
/// the two existing-diagonal corner triangles (not the face's cached
/// normal) so the score is independent of normal-cache state.
pub fn beautify_quad_cost(v0: Vec3, v1: Vec3, v2: Vec3, v3: Vec3, lock_degenerate: bool) -> f32 {
    // §8.6: derive a projection normal from the two existing-diagonal
    // corner triangles' cross products. Their unnormalised sum's length
    // doubles as a scale factor for the signum tolerance below.
    let corner_normal_v0v2v3 = (v3 - v2).cross(v0 - v2);
    let corner_normal_v0v1v2 = (v0 - v2).cross(v1 - v2);
    let corner_normal_sum = corner_normal_v0v2v3 + corner_normal_v0v1v2;
    let projection_scale = corner_normal_sum.length();
    if projection_scale == 0.0 {
        return f32::MAX;
    }
    let projection_normal = corner_normal_sum / projection_scale;

    let (basis_u, basis_w) = axis_dominant_basis(projection_normal);
    let project_to_2d = |p: Vec3| Vec2::new(basis_u.dot(p), basis_w.dot(p));
    let v0_proj = project_to_2d(v0);
    let v1_proj = project_to_2d(v1);
    let v2_proj = project_to_2d(v2);
    let v3_proj = project_to_2d(v3);

    // §8.6 scale-invariant signum guard: refuse to score when the two
    // existing-diagonal corner triangles already disagree on 2D winding.
    // The tolerance is normalised by `projection_scale` so the check
    // survives arbitrary input scale.
    const SIGNUM_EPS: f32 = 1e-5;
    let first_corner_signed_area = signed_area_2x(v2_proj, v3_proj, v0_proj);
    let second_corner_signed_area = signed_area_2x(v2_proj, v0_proj, v1_proj);
    let first_tolerant_sign = signum_eps(first_corner_signed_area / projection_scale, SIGNUM_EPS);
    let second_tolerant_sign = signum_eps(second_corner_signed_area / projection_scale, SIGNUM_EPS);
    if first_tolerant_sign + second_tolerant_sign == 0 {
        return f32::MAX;
    }

    beautify_quad_cost_2d(v1_proj, v2_proj, v3_proj, v0_proj, lock_degenerate)
}

fn signed_area_2x(a: Vec2, b: Vec2, c: Vec2) -> f32 {
    (b.x - a.x) * (c.y - a.y) - (c.x - a.x) * (b.y - a.y)
}

fn signum_eps(x: f32, eps: f32) -> i32 {
    if x > eps {
        1
    } else if x < -eps {
        -1
    } else {
        0
    }
}

// ============================== N-gon BEAUTY pass ==============================

/// A single rotated diagonal recorded by [`beautify_polygon`].
///
/// Both `before` and `after` are pairs of indices into the original
/// `coords` array passed to `beautify_polygon`. `before` is the edge
/// being rotated; `after` is the edge that replaces it.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct BeautifySwap {
    pub before: (u32, u32),
    pub after: (u32, u32),
}

/// Apply the §8 area-over-perimeter diagonal-swap beautify pass to a
/// 2D triangulation.
///
/// `coords` is the polygon's 2D vertex coordinates (in the same
/// projection used to produce `tris`). `tris` is the `n - 2` index
/// triples from [`ear_clip`]. The pass modifies `tris` in place and
/// returns the sequence of edge rotations performed, in the order
/// applied — a downstream destructive operator can replay each swap
/// against the live mesh by rotating the diagonal between
/// `coords[swap.before.0]` and `coords[swap.before.1]`.
///
/// Implements the §8.4 iterative pass: an initial scan of internal
/// edges pushes any negative-cost diagonals into a min-heap; the loop
/// pops the cheapest swap, applies it, and re-scores the four
/// surrounding edges with an area-scaled epsilon threshold to suppress
/// endless near-zero swaps.
pub fn beautify_polygon(coords: &[Vec2], tris: &mut [[u32; 3]]) -> Vec<BeautifySwap> {
    // Simple-polygon perimeter rule (§8.1): an edge is fixed when its two
    // endpoints are consecutive in the polygon cycle (indices `i`,`i+1`, or the
    // `0`/`last` wrap). Every other shared edge is an interior diagonal and may
    // rotate. This is exact only when `coords` *is* the cycle in order.
    let last_coord_index = (coords.len() - 1) as u32;
    beautify_polygon_protected(coords, tris, |a, b| {
        is_perimeter_edge_pair(sorted_endpoints(a, b), last_coord_index)
    })
}

/// Like [`beautify_polygon`] but with a caller-supplied **boundary predicate**
/// instead of the index-adjacency perimeter rule, for triangulations whose
/// vertex sequence is *not* a single simple cycle in index order — notably the
/// bridged-keyhole fill (`TRIANGLE_FILL.md` §1.2.4), whose sequence repeats the
/// two bridge endpoints and carries a zero-width slit.
///
/// `is_boundary(a, b)` returns `true` for an undirected coord-index pair that
/// must **never** rotate: the region's outer-loop edges, every hole-rim edge,
/// and the bridge/slit edges. Beautify then only ever rotates a *genuine
/// interior diagonal* between two adjacent fill triangles, and additionally
/// refuses any rotation whose replacement diagonal would (a) collapse onto a
/// repeated vertex (a zero-width slit edge — both endpoints share a position) or
/// (b) coincide with a protected boundary edge. Together these keep a diagonal
/// from ever being rotated into the slit or across the hole (§6).
pub fn beautify_polygon_protected(
    coords: &[Vec2],
    tris: &mut [[u32; 3]],
    is_boundary: impl Fn(u32, u32) -> bool,
) -> Vec<BeautifySwap> {
    let tri_count = tris.len();
    if tri_count < 2 {
        return Vec::new();
    }
    let half_edge_count = tri_count * 3;

    let mut half_edges: Vec<BeautifyHalfEdge> =
        vec![BeautifyHalfEdge::placeholder(); half_edge_count];

    // For each input triangle, lay down three half-edges whose `next`
    // pointers form a 3-cycle. Half-edge `tri * 3 + slot` starts at
    // `tris[tri][slot]` and points at `tris[tri][(slot + 1) % 3]`.
    for tri in 0..tri_count {
        for slot in 0..3 {
            let here = tri * 3 + slot;
            let then = tri * 3 + (slot + 1) % 3;
            half_edges[here].origin = tris[tri][slot];
            half_edges[here].next = then as u32;
        }
    }

    // Pair half-edges across each internal edge (edges whose endpoints
    // are not consecutive in the original polygon). Perimeter edges of
    // the polygon are left with `twin == NONE` and excluded from
    // rotation. Each internal pair gets a stable id so the rotation
    // bookkeeping below can address it independently of which half is
    // currently being scored.
    let mut pending_edges: std::collections::HashMap<(u32, u32), u32> =
        std::collections::HashMap::with_capacity(half_edge_count);
    let mut next_internal_id: u32 = 0;
    for here in 0..half_edge_count {
        let endpoints = {
            let from = half_edges[here].origin;
            let to = half_edges[half_edges[here].next as usize].origin;
            sorted_endpoints(from, to)
        };
        if is_boundary(endpoints.0, endpoints.1) {
            continue;
        }
        match pending_edges.remove(&endpoints) {
            Some(twin) => {
                let assigned = next_internal_id;
                next_internal_id += 1;
                half_edges[here].twin = twin;
                half_edges[twin as usize].twin = here as u32;
                half_edges[here].internal_edge_id = assigned;
                half_edges[twin as usize].internal_edge_id = assigned;
            }
            None => {
                pending_edges.insert(endpoints, here as u32);
            }
        }
    }
    let internal_edge_count = next_internal_id as usize;

    // Each internal edge gets a generation counter — incremented each
    // time we push a fresh score, used to discard stale heap entries
    // (Rust's `BinaryHeap` has no in-place decrease-key) — and an
    // `active` flag so we can skip popped entries belonging to edges
    // whose score has since moved above the swap threshold.
    let mut latest_generation: Vec<u32> = vec![0; internal_edge_count];
    let mut still_pending: Vec<bool> = vec![false; internal_edge_count];
    let mut swap_queue: BinaryHeap<SwapCandidate> = BinaryHeap::new();

    // Initial heap build. Each internal edge has two half-edges; we
    // only score from the lower-indexed one so each edge enters the
    // heap at most once.
    for here in 0..half_edge_count {
        let twin = half_edges[here].twin;
        if twin == BeautifyHalfEdge::NONE || (twin as usize) >= here {
            continue;
        }
        let (cost, _) =
            score_internal_edge_for_swap(coords, &half_edges, twin as usize, &is_boundary);
        if cost < 0.0 {
            let edge_id = half_edges[here].internal_edge_id as usize;
            latest_generation[edge_id] = 1;
            still_pending[edge_id] = true;
            swap_queue.push(SwapCandidate {
                cost,
                half_edge_idx: twin,
                generation: 1,
            });
        }
    }

    let mut swaps: Vec<BeautifySwap> = Vec::new();

    while let Some(candidate) = swap_queue.pop() {
        let head = candidate.half_edge_idx as usize;
        let edge_id = half_edges[head].internal_edge_id;
        if edge_id == BeautifyHalfEdge::NONE
            || latest_generation[edge_id as usize] != candidate.generation
            || !still_pending[edge_id as usize]
        {
            continue;
        }
        still_pending[edge_id as usize] = false;

        let tail = half_edges[head].twin as usize;
        let before_endpoints = (half_edges[head].origin, half_edges[tail].origin);
        let after_endpoints = (
            half_edges[third_corner_in_triangle(&half_edges, head)].origin,
            half_edges[third_corner_in_triangle(&half_edges, tail)].origin,
        );
        swaps.push(BeautifySwap {
            before: before_endpoints,
            after: after_endpoints,
        });

        rotate_internal_edge(&mut half_edges, head);

        // The two triangles around the rotated diagonal touch four
        // perimeter half-edges (two in each new triangle); each of
        // those may now be a better or worse swap candidate.
        let perimeter_around_swap = [
            half_edges[head].next,
            half_edges[half_edges[head].next as usize].next,
            half_edges[tail].next,
            half_edges[half_edges[tail].next as usize].next,
        ];
        for &neighbour in &perimeter_around_swap {
            rescore_after_swap(
                &half_edges,
                neighbour as usize,
                coords,
                &mut latest_generation,
                &mut still_pending,
                &mut swap_queue,
                &is_boundary,
            );
        }
    }

    // Read the post-swap triangle list back out by walking each
    // half-edge's `next` cycle exactly once.
    let mut visited = vec![false; half_edge_count];
    let mut out_idx = 0usize;
    for start in 0..half_edge_count {
        if visited[start] {
            continue;
        }
        let mut tri = [0u32; 3];
        let mut walk = start;
        for slot in tri.iter_mut() {
            *slot = half_edges[walk].origin;
            visited[walk] = true;
            walk = half_edges[walk].next as usize;
        }
        tris[out_idx] = tri;
        out_idx += 1;
    }

    swaps
}

/// A single half-edge in [`beautify_polygon`]'s internal triangle
/// graph. Each triangle owns three half-edges, and every internal
/// polygon edge has exactly two — one in each triangle that shares
/// it — so the standard half-edge `next` / `twin` link names apply.
#[derive(Copy, Clone)]
struct BeautifyHalfEdge {
    /// Vertex this half-edge starts at (index into the caller's
    /// `coords` array).
    origin: u32,
    /// Next half-edge around the same triangle (3-cycle).
    next: u32,
    /// Opposing half-edge across the shared triangle edge, or
    /// [`Self::NONE`] if this half-edge lies on the polygon perimeter.
    twin: u32,
    /// Identifier for the internal edge this half-edge belongs to.
    /// Both twins share the same id. [`Self::NONE`] for perimeter
    /// half-edges.
    internal_edge_id: u32,
}

impl BeautifyHalfEdge {
    const NONE: u32 = u32::MAX;

    fn placeholder() -> Self {
        Self {
            origin: Self::NONE,
            next: Self::NONE,
            twin: Self::NONE,
            internal_edge_id: Self::NONE,
        }
    }
}

/// One queued swap candidate. The heap is ordered ascending by `cost`
/// (cheapest swap first). A `generation` tag lets the consumer reject
/// stale entries when a later push has superseded an earlier one.
#[derive(Copy, Clone)]
struct SwapCandidate {
    cost: f32,
    half_edge_idx: u32,
    generation: u32,
}

impl PartialEq for SwapCandidate {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == std::cmp::Ordering::Equal
    }
}
impl Eq for SwapCandidate {}
impl PartialOrd for SwapCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for SwapCandidate {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Rust's BinaryHeap is a max-heap; invert the ordering so we
        // pop the lowest-cost candidate first. Half-edge index breaks
        // cost ties deterministically.
        other
            .cost
            .total_cmp(&self.cost)
            .then(other.half_edge_idx.cmp(&self.half_edge_idx))
    }
}

/// Score one internal edge by feeding the four quad corners around it
/// into the §8.2 cost rule. Returns `(cost, quad_total_area)`.
fn score_internal_edge_for_swap(
    coords: &[Vec2],
    half_edges: &[BeautifyHalfEdge],
    diagonal_in_first_tri: usize,
    is_boundary: &impl Fn(u32, u32) -> bool,
) -> (f32, f32) {
    let diagonal_in_second_tri = half_edges[diagonal_in_first_tri].twin as usize;
    let opposite_in_first_tri = third_corner_in_triangle(half_edges, diagonal_in_first_tri);
    let opposite_in_second_tri = third_corner_in_triangle(half_edges, diagonal_in_second_tri);

    // Candidate (post-rotation) diagonal connects the two opposite corners.
    // Refuse the swap outright when that diagonal would land on a protected
    // boundary edge or collapse onto a repeated vertex (a zero-width slit edge
    // of the bridged keyhole, §1.2.4) — either would rotate a diagonal into the
    // slit / across the hole. A refused swap reports a non-negative cost so it
    // never enters the heap.
    let candidate_a = half_edges[opposite_in_first_tri].origin;
    let candidate_b = half_edges[opposite_in_second_tri].origin;
    if is_boundary(candidate_a, candidate_b)
        || coords[candidate_a as usize] == coords[candidate_b as usize]
    {
        return (f32::MAX, 0.0);
    }

    let v1 = coords[candidate_a as usize];
    let v2 = coords[half_edges[diagonal_in_first_tri].origin as usize];
    let v3 = coords[candidate_b as usize];
    let v4 = coords[half_edges[diagonal_in_second_tri].origin as usize];

    beautify_quad_cost_2d_ex(v1, v2, v3, v4, false)
}

/// Half-edge rewiring for the §8.1 swap operation. The diagonal
/// half-edge at `diagonal_idx` lies inside one of the two triangles
/// sharing the rotated edge; its twin lies in the other. After this
/// call the diagonal connects the two corners that were previously
/// *opposite* the diagonal (i.e. the third vertex of each triangle),
/// and the two `next` cycles describe the post-swap triangles.
fn rotate_internal_edge(half_edges: &mut [BeautifyHalfEdge], diagonal_idx: usize) {
    let diag_in_first_tri = diagonal_idx;
    let succ_in_first_tri = half_edges[diag_in_first_tri].next as usize;
    let pred_in_first_tri = half_edges[succ_in_first_tri].next as usize;

    let diag_in_second_tri = half_edges[diag_in_first_tri].twin as usize;
    let succ_in_second_tri = half_edges[diag_in_second_tri].next as usize;
    let pred_in_second_tri = half_edges[succ_in_second_tri].next as usize;

    // Build the post-swap 3-cycle for the first new triangle:
    // diagonal → predecessor-from-first → successor-from-second → back.
    half_edges[diag_in_first_tri].next = pred_in_first_tri as u32;
    half_edges[pred_in_first_tri].next = succ_in_second_tri as u32;
    half_edges[succ_in_second_tri].next = diag_in_first_tri as u32;

    // ...and for the second new triangle.
    half_edges[diag_in_second_tri].next = pred_in_second_tri as u32;
    half_edges[pred_in_second_tri].next = succ_in_first_tri as u32;
    half_edges[succ_in_first_tri].next = diag_in_second_tri as u32;

    // The rotated diagonal now originates at the previously-opposite
    // corner on each side.
    half_edges[diag_in_first_tri].origin = half_edges[pred_in_second_tri].origin;
    half_edges[diag_in_second_tri].origin = half_edges[pred_in_first_tri].origin;
}

/// Re-score one internal edge after the swap that bordered it and push
/// any negative-cost entry back onto the heap with a fresh generation.
fn rescore_after_swap(
    half_edges: &[BeautifyHalfEdge],
    half_edge_idx: usize,
    coords: &[Vec2],
    latest_generation: &mut [u32],
    still_pending: &mut [bool],
    swap_queue: &mut BinaryHeap<SwapCandidate>,
    is_boundary: &impl Fn(u32, u32) -> bool,
) {
    if half_edges[half_edge_idx].twin == BeautifyHalfEdge::NONE {
        return;
    }
    let edge_id = half_edges[half_edge_idx].internal_edge_id as usize;
    let (cost, quad_area) =
        score_internal_edge_for_swap(coords, half_edges, half_edge_idx, is_boundary);
    // §8.4 area-scaled epsilon: suppresses endless near-zero swaps on
    // large quads where the cost would otherwise oscillate around 0.
    let acceptance_threshold = -1e-6_f32 * quad_area.max(1.0);
    if cost < acceptance_threshold {
        latest_generation[edge_id] += 1;
        still_pending[edge_id] = true;
        swap_queue.push(SwapCandidate {
            cost,
            half_edge_idx: half_edge_idx as u32,
            generation: latest_generation[edge_id],
        });
    } else {
        still_pending[edge_id] = false;
    }
}

/// In a triangle's 3-cycle, the half-edge whose origin is the third
/// vertex (the corner opposite the given half-edge's edge).
fn third_corner_in_triangle(half_edges: &[BeautifyHalfEdge], here: usize) -> usize {
    half_edges[half_edges[here].next as usize].next as usize
}

fn sorted_endpoints(a: u32, b: u32) -> (u32, u32) {
    if a < b {
        (a, b)
    } else {
        (b, a)
    }
}

fn is_perimeter_edge_pair(endpoints: (u32, u32), last_coord_index: u32) -> bool {
    debug_assert!(endpoints.0 < endpoints.1);
    endpoints.0 + 1 == endpoints.1 || (endpoints.0 == 0 && endpoints.1 == last_coord_index)
}

// ============================== Ear-clipping ==============================

/// Triangulate a 2D polygon by ear-clipping
/// (`BMESH_POLYGON_TRIANGULATION.md` §7).
///
/// `coords` is the polygon's vertices in cycle order. `coords_sign` is a
/// winding hint:
///
/// - `+1` if `coords` is wound clockwise in 2D.
/// - `-1` if counter-clockwise.
/// - `0` to auto-detect from the polygon's signed area.
///
/// The render-time path (used by [`crate::triangulate_corners`]) feeds
/// the kernel CW-in-2D polygons via the negated axis-dominant projection
/// (§3.1), so callers should pass `+1` when feeding the output of
/// [`axis_dominant_basis_negated`].
///
/// Output: `n - 2` triangle index triples appended to `out_tris`, where
/// each triple is an `[u32; 3]` of indices into `coords`. Each triple is
/// emitted in `(prev_index, ear_index, next_index)` order at the moment
/// of clipping (§7.6) — un-projecting these back to 3D produces
/// triangles whose normal matches the source face.
///
/// `n < 3` produces no output. `n == 3` emits one triple `[0, 1, 2]`.
/// Severely degenerate inputs follow §7.5: every vertex appears in some
/// triple, but the resulting triangles may overlap.
pub fn ear_clip(coords: &[Vec2], coords_sign: i32, out_tris: &mut Vec<[u32; 3]>) {
    let vertex_count = coords.len();
    if vertex_count < 3 {
        return;
    }
    if vertex_count == 3 {
        out_tris.push([0, 1, 2]);
        return;
    }

    let mut state = EarClipState::new(coords, coords_sign);
    state.run(coords, out_tris);
}

const SIGN_CONCAVE: i8 = -1;
const SIGN_TANGENTIAL: i8 = 0;
const SIGN_CONVEX: i8 = 1;

#[derive(Copy, Clone)]
struct PolyNode {
    prev: u32,
    next: u32,
    /// Index into the input `coords` slice.
    coord_idx: u32,
    /// `SIGN_CONCAVE` / `SIGN_TANGENTIAL` / `SIGN_CONVEX`.
    sign: i8,
}

struct EarClipState {
    nodes: Vec<PolyNode>,
    /// Any surviving node, used as the starting point when the algorithm
    /// needs an entry into the linked list.
    head: u32,
    remaining: u32,
    /// Number of non-CONVEX vertices among surviving nodes. When zero,
    /// every remaining vertex is a valid ear and the inclusion test can
    /// be skipped (the §7.3 convex-skip fast path).
    non_convex_count: u32,
}

impl EarClipState {
    fn new(coords: &[Vec2], coords_sign: i32) -> Self {
        let vertex_count = coords.len() as u32;
        // Resolve auto-detect winding using the §7.1 shoelace-sum rule:
        // accumulate (b.x - a.x) * (b.y + a.y) over consecutive vertex
        // pairs. Non-positive sum ⇒ CW (hint value 1), positive sum ⇒
        // CCW (hint value -1).
        let effective_sign = if coords_sign == 0 {
            let mut shoelace_sum = 0.0f32;
            for i in 0..vertex_count as usize {
                let a = coords[i];
                let b = coords[(i + 1) % vertex_count as usize];
                shoelace_sum += (b.x - a.x) * (b.y + a.y);
            }
            if shoelace_sum <= 0.0 {
                1
            } else {
                -1
            }
        } else {
            coords_sign
        };

        // Build the linked list per §7.1: nodes[i].coord_idx selects
        // which coord this node references; the linked-list neighbour
        // pointers are always prev = i - 1, next = i + 1 (modulo n).
        // When the input is CCW we map nodes[i].coord_idx = n - 1 - i
        // so walking next-ward visits coords in reverse — letting the
        // kernel process a CCW polygon as if it were CW without
        // rebuilding the coordinate buffer.
        let mut nodes: Vec<PolyNode> = (0..vertex_count)
            .map(|i| PolyNode {
                prev: if i == 0 { vertex_count - 1 } else { i - 1 },
                next: if i == vertex_count - 1 { 0 } else { i + 1 },
                coord_idx: if effective_sign == 1 {
                    i
                } else {
                    vertex_count - 1 - i
                },
                sign: 0,
            })
            .collect();

        // Initial classification of every vertex.
        let mut non_convex_count: u32 = 0;
        for i in 0..vertex_count as usize {
            let corner_sign = span_tri_sign(
                coords[nodes[nodes[i].prev as usize].coord_idx as usize],
                coords[nodes[i].coord_idx as usize],
                coords[nodes[nodes[i].next as usize].coord_idx as usize],
            );
            nodes[i].sign = corner_sign;
            if corner_sign != SIGN_CONVEX {
                non_convex_count += 1;
            }
        }

        Self {
            nodes,
            head: 0,
            remaining: vertex_count,
            non_convex_count,
        }
    }

    fn run(&mut self, coords: &[Vec2], out_tris: &mut Vec<[u32; 3]>) {
        let mut next_search_start = self.head;
        let mut sweep_reversed = false;

        while self.remaining > 3 {
            let ear_node = self.find_ear_tip(coords, next_search_start, sweep_reversed);
            let prev_node = self.nodes[ear_node as usize].prev;
            let next_node = self.nodes[ear_node as usize].next;

            out_tris.push([
                self.nodes[prev_node as usize].coord_idx,
                self.nodes[ear_node as usize].coord_idx,
                self.nodes[next_node as usize].coord_idx,
            ]);

            if self.nodes[ear_node as usize].sign != SIGN_CONVEX {
                self.non_convex_count -= 1;
            }

            // Cut the ear from the linked list.
            self.nodes[prev_node as usize].next = next_node;
            self.nodes[next_node as usize].prev = prev_node;
            if self.head == ear_node {
                self.head = next_node;
            }
            self.remaining -= 1;

            // The two surviving neighbours of the clipped ear now have
            // a new corner triangle (formed with their new neighbour);
            // reclassify them.
            self.reclassify_node(coords, prev_node);
            self.reclassify_node(coords, next_node);

            // Advance the ear-search start; toggle sweep direction when
            // the natural next candidate is non-convex (§7.4).
            let candidate = if sweep_reversed {
                self.nodes[prev_node as usize].prev
            } else {
                self.nodes[next_node as usize].next
            };
            if self.nodes[candidate as usize].sign != SIGN_CONVEX {
                next_search_start = if sweep_reversed {
                    self.nodes[candidate as usize].prev
                } else {
                    self.nodes[candidate as usize].next
                };
                sweep_reversed = !sweep_reversed;
            } else {
                next_search_start = candidate;
            }
        }

        // Final triangle: emit the three surviving nodes in linked-list
        // order starting from `head`.
        if self.remaining == 3 {
            let first = self.head;
            let second = self.nodes[first as usize].next;
            let third = self.nodes[second as usize].next;
            out_tris.push([
                self.nodes[first as usize].coord_idx,
                self.nodes[second as usize].coord_idx,
                self.nodes[third as usize].coord_idx,
            ]);
        }
    }

    /// Re-classify one surviving neighbour's corner sign after an ear
    /// was clipped next to it. A vertex that was reflex/tangential may
    /// now be convex, dropping the non-convex count; we don't need to
    /// recompute already-convex neighbours since clipping cannot make
    /// a convex corner reflex.
    fn reclassify_node(&mut self, coords: &[Vec2], node: u32) {
        if self.nodes[node as usize].sign == SIGN_CONVEX {
            return;
        }
        let new_sign = span_tri_sign(
            coords[self.nodes[self.nodes[node as usize].prev as usize].coord_idx as usize],
            coords[self.nodes[node as usize].coord_idx as usize],
            coords[self.nodes[self.nodes[node as usize].next as usize].coord_idx as usize],
        );
        if new_sign == SIGN_CONVEX {
            self.non_convex_count -= 1;
        }
        self.nodes[node as usize].sign = new_sign;
    }

    /// Two-pass ear search (§7.4), with desperate-mode fallback (§7.5).
    fn find_ear_tip(&self, coords: &[Vec2], search_start: u32, reverse: bool) -> u32 {
        for &accepted_sign in &[SIGN_CONVEX, SIGN_TANGENTIAL] {
            let mut cursor = search_start;
            let mut remaining_steps = self.remaining;
            while remaining_steps > 0 {
                if self.is_acceptable_ear(coords, cursor, accepted_sign) {
                    return cursor;
                }
                cursor = if reverse {
                    self.nodes[cursor as usize].prev
                } else {
                    self.nodes[cursor as usize].next
                };
                remaining_steps -= 1;
            }
        }

        // Desperate mode: every surviving vertex failed the ear check.
        // Return any non-CONCAVE vertex; if even that fails, return the
        // last visited.
        let mut cursor = search_start;
        let mut remaining_steps = self.remaining;
        while remaining_steps > 0 {
            if self.nodes[cursor as usize].sign != SIGN_CONCAVE {
                return cursor;
            }
            cursor = self.nodes[cursor as usize].next;
            remaining_steps -= 1;
        }
        cursor
    }

    fn is_acceptable_ear(&self, coords: &[Vec2], ear_node: u32, accepted_sign: i8) -> bool {
        // Fast path: the whole polygon is convex → every vertex is a
        // valid ear, no inclusion test needed.
        if self.non_convex_count == 0 {
            return true;
        }
        if self.nodes[ear_node as usize].sign != accepted_sign {
            return false;
        }

        let prev = self.nodes[ear_node as usize].prev;
        let next = self.nodes[ear_node as usize].next;
        let v_prev = coords[self.nodes[prev as usize].coord_idx as usize];
        let v_ear = coords[self.nodes[ear_node as usize].coord_idx as usize];
        let v_next = coords[self.nodes[next as usize].coord_idx as usize];

        // Walk all surviving vertices except prev/ear/next. Convex
        // vertices can't fall inside a candidate ear (§7.3), so they
        // are skipped. Stop once we've examined every non-convex
        // vertex in the polygon.
        let mut non_convex_examined: u32 = 0;
        let non_convex_to_examine = self.non_convex_count;
        let mut cursor = self.nodes[next as usize].next;
        while cursor != prev {
            if self.nodes[cursor as usize].sign != SIGN_CONVEX {
                let v_test = coords[self.nodes[cursor as usize].coord_idx as usize];
                // For a CW polygon, "v_test inside triangle
                // (v_prev, v_ear, v_next)" means every edge-side test
                // is non-CONCAVE. The next-prev edge fails most often,
                // so test it first to short-circuit cheaply.
                if span_tri_sign(v_next, v_prev, v_test) != SIGN_CONCAVE
                    && span_tri_sign(v_prev, v_ear, v_test) != SIGN_CONCAVE
                    && span_tri_sign(v_ear, v_next, v_test) != SIGN_CONCAVE
                {
                    return false;
                }
                non_convex_examined += 1;
                if non_convex_examined == non_convex_to_examine {
                    break;
                }
            }
            cursor = self.nodes[cursor as usize].next;
        }

        true
    }
}

/// Sign of the signed 2× area of triangle `(v3, v2, v1)` — note the
/// reversed argument order compared to the obvious read. Positive when
/// the triangle is wound clockwise in standard XY space, which is the
/// ear orientation the §7.1 CW-in-2D convention expects.
fn span_tri_sign(v1: Vec2, v2: Vec2, v3: Vec2) -> i8 {
    // Compute the z-component of `(v2 - v3) × (v1 - v3)`. Its sign is
    // the winding sign of the triangle.
    let edge_v3_to_v2 = v2 - v3;
    let edge_v3_to_v1 = v1 - v3;
    let signed_area_2x = edge_v3_to_v2.x * edge_v3_to_v1.y - edge_v3_to_v1.x * edge_v3_to_v2.y;
    if signed_area_2x > 0.0 {
        SIGN_CONVEX
    } else if signed_area_2x == 0.0 {
        SIGN_TANGENTIAL
    } else {
        SIGN_CONCAVE
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::vec2;

    fn project(u: Vec3, w: Vec3, p: Vec3) -> Vec2 {
        vec2(u.dot(p), w.dot(p))
    }

    fn approx_eq(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() <= eps
    }

    #[test]
    fn basis_is_orthonormal_for_axis_aligned_normals() {
        for n in [
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(-1.0, 0.0, 0.0),
            Vec3::new(0.0, -1.0, 0.0),
            Vec3::new(0.0, 0.0, -1.0),
        ] {
            let (u, w) = axis_dominant_basis_negated(n);
            assert!(approx_eq(u.length(), 1.0, 1e-5), "u not unit for n={n:?}");
            assert!(approx_eq(w.length(), 1.0, 1e-5), "w not unit for n={n:?}");
            assert!(approx_eq(u.dot(w), 0.0, 1e-5), "u·w != 0 for n={n:?}");
            // Right-handed frame `(u, w, -n)`: u × w should equal -n
            // (within sign of normal). For axis-aligned normals the
            // cross is exact.
            let cross = u.cross(w);
            assert!(
                cross.distance(-n) < 1e-4,
                "u × w != -n for n={n:?} (got {cross:?})"
            );
        }
    }

    #[test]
    fn basis_for_oblique_normal_is_orthonormal() {
        let n = Vec3::new(1.0, 2.0, 3.0).normalize();
        let (u, w) = axis_dominant_basis_negated(n);
        assert!(approx_eq(u.length(), 1.0, 1e-5));
        assert!(approx_eq(w.length(), 1.0, 1e-5));
        assert!(approx_eq(u.dot(w), 0.0, 1e-5));
        assert!(approx_eq(u.dot(n), 0.0, 1e-5));
        assert!(approx_eq(w.dot(n), 0.0, 1e-5));
    }

    #[test]
    fn ccw_3d_face_projects_to_clockwise_2d() {
        // A unit CCW triangle in the XY plane (normal = +Z).
        let n = Vec3::Z;
        let (u, w) = axis_dominant_basis_negated(n);
        let p0 = project(u, w, Vec3::new(0.0, 0.0, 0.0));
        let p1 = project(u, w, Vec3::new(1.0, 0.0, 0.0));
        let p2 = project(u, w, Vec3::new(0.0, 1.0, 0.0));
        // Signed area in 2D: positive => CCW, negative => CW. The
        // negated axis-dominant projection of a CCW-in-3D face must be
        // CW-in-2D (guide §3.1), so we expect negative signed area.
        let signed_area = 0.5 * ((p1.x - p0.x) * (p2.y - p0.y) - (p2.x - p0.x) * (p1.y - p0.y));
        assert!(
            signed_area < 0.0,
            "expected CW (negative signed area), got {signed_area}"
        );
    }

    // ---- ear_clip tests ----

    fn cw_unit_quad() -> [Vec2; 4] {
        // CW-in-2D unit quad (the §7.1 projection convention).
        [
            vec2(0.0, 0.0),
            vec2(0.0, 1.0),
            vec2(1.0, 1.0),
            vec2(1.0, 0.0),
        ]
    }

    fn assert_covers_all_indices(tris: &[[u32; 3]], n: u32) {
        let mut seen = vec![false; n as usize];
        for t in tris {
            for &i in t {
                assert!(i < n, "index {i} out of range");
                seen[i as usize] = true;
            }
        }
        for (i, &s) in seen.iter().enumerate() {
            assert!(s, "vertex {i} missing from triangulation");
        }
    }

    fn assert_no_zero_area_tris(coords: &[Vec2], tris: &[[u32; 3]]) {
        for &t in tris {
            let v1 = coords[t[0] as usize];
            let v2 = coords[t[1] as usize];
            let v3 = coords[t[2] as usize];
            let area = ((v2.x - v1.x) * (v3.y - v1.y) - (v3.x - v1.x) * (v2.y - v1.y)).abs();
            assert!(area > 0.0, "triangle {t:?} has zero area");
        }
    }

    fn assert_all_triangles_cw(coords: &[Vec2], tris: &[[u32; 3]]) {
        for &t in tris {
            let v1 = coords[t[0] as usize];
            let v2 = coords[t[1] as usize];
            let v3 = coords[t[2] as usize];
            assert!(
                span_tri_sign(v1, v2, v3) == SIGN_CONVEX,
                "expected CW winding (SIGN_CONVEX) on tri {t:?}",
            );
        }
    }

    #[test]
    fn ear_clip_triangle_passes_through() {
        let coords = [vec2(0.0, 0.0), vec2(0.0, 1.0), vec2(1.0, 0.0)];
        let mut out = Vec::new();
        ear_clip(&coords, 1, &mut out);
        assert_eq!(out, vec![[0, 1, 2]]);
    }

    #[test]
    fn ear_clip_unit_quad_emits_two_triangles_covering_all_corners() {
        let coords = cw_unit_quad();
        let mut out = Vec::new();
        ear_clip(&coords, 1, &mut out);
        assert_eq!(out.len(), 2);
        assert_covers_all_indices(&out, 4);
        assert_no_zero_area_tris(&coords, &out);
        assert_all_triangles_cw(&coords, &out);
    }

    #[test]
    fn ear_clip_concave_quad_emits_two_non_overlapping_triangles() {
        // CW concave quad ("arrow"): vertex 2 is pushed inward so the
        // 0-2 diagonal lies outside the polygon. A naive fan from
        // vertex 0 would flip; ear-clipping must split along 1-3.
        let coords = [
            vec2(0.0, 0.0),
            vec2(0.0, 1.0),
            vec2(0.5, 0.5),
            vec2(1.0, 0.0),
        ];
        let mut out = Vec::new();
        ear_clip(&coords, 1, &mut out);
        assert_eq!(out.len(), 2);
        assert_covers_all_indices(&out, 4);
        assert_no_zero_area_tris(&coords, &out);
        assert_all_triangles_cw(&coords, &out);
    }

    #[test]
    fn ear_clip_regular_hexagon_emits_four_tris() {
        // CW regular hexagon (every vertex strictly convex).
        let n = 6;
        let coords: Vec<Vec2> = (0..n)
            .map(|i| {
                let theta = -(i as f32) * (std::f32::consts::TAU / n as f32);
                vec2(theta.cos(), theta.sin())
            })
            .collect();
        let mut out = Vec::new();
        ear_clip(&coords, 1, &mut out);
        assert_eq!(out.len(), n - 2);
        assert_covers_all_indices(&out, n as u32);
        assert_no_zero_area_tris(&coords, &out);
        assert_all_triangles_cw(&coords, &out);
    }

    #[test]
    fn ear_clip_concave_pentagon() {
        // CW "Pac-Man" pentagon: one strongly concave bite.
        let coords = [
            vec2(0.0, 0.0),
            vec2(0.0, 2.0),
            vec2(1.0, 1.0),
            vec2(2.0, 2.0),
            vec2(2.0, 0.0),
        ];
        let mut out = Vec::new();
        ear_clip(&coords, 1, &mut out);
        assert_eq!(out.len(), 3);
        assert_covers_all_indices(&out, 5);
        assert_no_zero_area_tris(&coords, &out);
        assert_all_triangles_cw(&coords, &out);
    }

    #[test]
    fn ear_clip_auto_winding_detection() {
        // Same hexagon as above but CCW input — auto-detect should
        // walk it in reverse and still produce 4 valid triangles.
        let n = 6;
        let coords: Vec<Vec2> = (0..n)
            .map(|i| {
                let theta = (i as f32) * (std::f32::consts::TAU / n as f32);
                vec2(theta.cos(), theta.sin())
            })
            .collect();
        let mut out = Vec::new();
        ear_clip(&coords, 0, &mut out);
        assert_eq!(out.len(), n - 2);
        assert_covers_all_indices(&out, n as u32);
        assert_no_zero_area_tris(&coords, &out);
    }

    #[test]
    fn ear_clip_collinear_polygon_falls_back_safely() {
        // All vertices on a horizontal line — fully degenerate.
        // §7.5 says the algorithm still terminates with n-2 triples
        // and every vertex appears in at least one. Triangles will
        // have zero area, which is acceptable for the degenerate case.
        let coords = [
            vec2(0.0, 0.0),
            vec2(1.0, 0.0),
            vec2(2.0, 0.0),
            vec2(3.0, 0.0),
        ];
        let mut out = Vec::new();
        ear_clip(&coords, 1, &mut out);
        assert_eq!(out.len(), 2);
        assert_covers_all_indices(&out, 4);
    }
}
