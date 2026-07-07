use std::collections::{HashMap, HashSet, VecDeque};

use glam::Vec3;
use smallvec::SmallVec;

use crate::loopmesh::*;

/// Parameters for [`LoopMesh::grid_fill`], mirroring the operator's BMOP slot
/// knobs.
///
/// Algorithm in
/// [`GRID_FILL.md`](../../../../radial-mesh-guides/operators/GRID_FILL.md) §1.
#[derive(Copy, Clone, Debug, Default)]
pub struct GridFillParams {
    /// Material index stamped onto every quad the operator creates (§1/§7).
    pub mat_nr: i32,
    /// When set, each created quad is flagged smooth-shaded (§1/§7). A per-face
    /// attribute only; it does not change topology.
    pub use_smooth: bool,
    /// Interior-vertex positioning mode (§1/§5/§6). Clear (default) follows the
    /// two side-loop shapes via a per-row transform blend; set selects a simple
    /// bilinear weighting of the four boundary sides. Affects interior-vertex
    /// *coordinates* only, never topology.
    pub use_interp_simple: bool,
}

/// The `faces.out` of [`LoopMesh::grid_fill`]: every **new** quad the operator
/// created — `(cols−1) × (rows−1)` of them for a `cols × rows` vertex lattice
/// (§7). An input that fails any §3 gate fills nothing and reports an empty set.
#[derive(Clone, Debug, Default)]
pub struct GridFillResult {
    faces: Vec<FaceKey>,
}

impl GridFillResult {
    pub fn faces(&self) -> &[FaceKey] {
        &self.faces
    }
    pub fn is_empty(&self) -> bool {
        self.faces.is_empty()
    }
}

/// A `cols × rows` lattice of vertex keys addressed `grid[x * rows + y]` (§2).
struct Lattice {
    cols: usize,
    rows: usize,
    cells: Vec<VertexKey>,
}

impl Lattice {
    fn new(cols: usize, rows: usize) -> Self {
        Self {
            cols,
            rows,
            // Sentinel; every cell is written (perimeter then interior) before
            // any read.
            cells: vec![VertexKey(u32::MAX); cols * rows],
        }
    }
    #[inline]
    fn get(&self, x: usize, y: usize) -> VertexKey {
        self.cells[x * self.rows + y]
    }
    #[inline]
    fn set(&mut self, x: usize, y: usize, v: VertexKey) {
        self.cells[x * self.rows + y] = v;
    }
}

impl LoopMesh {
    /// Span the rectangular region between **two opposing open edge loops** with
    /// a regular grid of quads (§1). The input `edges` are the two opposing side
    /// loops; the two connecting **rails** are discovered from the surrounding
    /// wire/boundary edges (§3.2); the interior lattice vertices are **created**
    /// by interpolation (§5/§6); the perimeter is reused.
    ///
    /// Algorithm in
    /// [`GRID_FILL.md`](../../../../radial-mesh-guides/operators/GRID_FILL.md):
    /// the §3.1 two-open-loops gate (edge-loop discovery over the marked input,
    /// rejecting any count other than two and any closed side), the §3.2 rail
    /// discovery (a wire/boundary path search between the loops' endpoints with
    /// the side edges hidden), the §3.3 endpoint pairing and flip (natural
    /// pairing first, the crossed pairing reversing loop B), the §4 lattice
    /// addressing and radial-data winding vote, the §5 default transform-blend /
    /// §6 simple-bilinear interior interpolation, and the §7 quad build with
    /// `mat_nr` / `use_smooth` stamping into `faces.out`.
    ///
    /// All-or-nothing (§3): any failed gate fills nothing and returns an empty
    /// result. Span equalisation (§3.4) fills mismatched side or rail counts by
    /// expanding the shorter member with scaffold vertices and collapsing them
    /// back out after the build, yielding a fan of quads plus `|Δ|` triangles.
    pub fn grid_fill(&mut self, edges: &[EdgeKey], params: &GridFillParams) -> GridFillResult {
        let empty = GridFillResult::default();

        // §3.1 — discover the two opposing side loops. Exactly two, both open.
        let Some(loops) = self.grid_discover_open_loops(edges) else {
            return empty;
        };
        if loops.len() != 2 {
            return empty;
        }
        let mut loop_a = loops[0].clone();
        let mut loop_b = loops[1].clone();
        if loop_a.len() < 2 || loop_b.len() < 2 {
            return empty;
        }

        // The side edges are hidden during rail discovery so the search cannot
        // wander back along the input loops (§3.2).
        let side_edges: HashSet<EdgeKey> = edges.iter().copied().collect();

        let a_first = *loop_a.first().unwrap();
        let a_last = *loop_a.last().unwrap();
        let b_first = *loop_b.first().unwrap();
        let b_last = *loop_b.last().unwrap();

        // §3.3 — try the natural pairing, then the crossed pairing (which
        // reverses loop B so the rails do not twist).
        let (mut rail_left, mut rail_right) = match (
            self.grid_find_rail(a_first, b_first, &side_edges),
            self.grid_find_rail(a_last, b_last, &side_edges),
        ) {
            (Some(left), Some(right)) => (left, right),
            _ => match (
                self.grid_find_rail(a_first, b_last, &side_edges),
                self.grid_find_rail(a_last, b_first, &side_edges),
            ) {
                (Some(left), Some(right)) => {
                    loop_b.reverse();
                    (left, right)
                }
                // §3.2 — no connecting wire/boundary path → not joinable.
                _ => return empty,
            },
        };

        // §3.4 — anchor the lattice origin deterministically. §3.3 leaves two
        // free flips (which side is A vs B, which endpoint each loop's discovery
        // walk labels "first"); both are arbitrary walk-order artifacts, so the
        // span-equalisation collapse — which is keyed to the lattice origin
        // `grid[0][0]`, not to geometry — would otherwise land its surviving
        // corner on a hash-order-dependent physical vertex. We pin `grid[0][0]`
        // to the corner with the lowest element index (the lattice addressed by
        // element order, never enumeration order), choosing among the four valid
        // non-twisted orientations. This makes the §3.4 collapse reproducible.
        {
            let c00 = loop_a[0];
            let c10 = *loop_a.last().unwrap();
            let c01 = loop_b[0];
            let c11 = *loop_b.last().unwrap();
            let origin = [c00, c10, c01, c11].into_iter().min_by_key(|v| v.0).unwrap();
            if origin == c10 {
                // Horizontal flip: the far bottom corner becomes the origin.
                loop_a.reverse();
                loop_b.reverse();
                std::mem::swap(&mut rail_left, &mut rail_right);
            } else if origin == c01 {
                // Vertical flip: the near top corner becomes the origin.
                std::mem::swap(&mut loop_a, &mut loop_b);
                rail_left.reverse();
                rail_right.reverse();
            } else if origin == c11 {
                // Both flips: the far top corner becomes the origin.
                std::mem::swap(&mut loop_a, &mut loop_b);
                loop_a.reverse();
                loop_b.reverse();
                rail_left.reverse();
                rail_right.reverse();
                std::mem::swap(&mut rail_left, &mut rail_right);
            }
            // else origin == c00: identity, the lattice already starts at it.
        }

        // §3.4 — the lattice must be rectangular, so the two opposing sides are
        // reconciled to a common `cols` and the two rails to a common `rows`.
        // Equal members need no work; a shorter member is *expanded* with
        // temporary scaffold vertices (not rejected), and the scaffold collapses
        // back out after the quads are built — degenerating the corner-losing
        // cells into triangles.
        let cols = loop_a.len().max(loop_b.len());
        let rows = rail_left.len().max(rail_right.len());
        if cols < 2 || rows < 2 {
            return empty;
        }

        // §3.3 — overlapping rails cannot bound a clean rectangle.
        if grid_rails_overlap(&rail_left, &rail_right) {
            return empty;
        }

        // §3.4 Phase 1+2 — expand each boundary member to the full lattice count.
        // `*_pos` carry the genuine (scaffold-subdivided) boundary positions that
        // feed the §5/§6 interior interpolation; `*_key` carry the per-slot
        // vertex, with every scaffold slot already mapped to the perimeter vertex
        // it collapses into, so no scaffold vertex survives in the result.
        let (bottom_pos, bottom_key) = self.grid_expand_member(&loop_a, cols);
        let (top_pos, top_key) = self.grid_expand_member(&loop_b, cols);
        let (left_pos, left_key) = self.grid_expand_member(&rail_left, rows);
        let (right_pos, right_key) = self.grid_expand_member(&rail_right, rows);

        // §4.1 — write the perimeter into the lattice. The four corners are
        // written by more than one side; §3.3 pinned them to the same vertices,
        // and slot 0 / slot max are always real (never scaffold), so the corners
        // stay coherent under expansion.
        let mut grid = Lattice::new(cols, rows);
        for x in 0..cols {
            grid.set(x, 0, bottom_key[x]);
            grid.set(x, rows - 1, top_key[x]);
        }
        for y in 0..rows {
            grid.set(0, y, left_key[y]);
            grid.set(cols - 1, y, right_key[y]);
        }

        // §5/§6 — create the genuine interior vertices by interpolation over the
        // expanded boundary profiles.
        self.grid_build_interior(
            &mut grid,
            &bottom_pos,
            &top_pos,
            &left_pos,
            &right_pos,
            params.use_interp_simple,
        );

        // §4.2 — settle the build winding against any bordering surface.
        let flip = self.grid_winding_flip(&grid);

        // §7 / §3.4 Phase 3 — one face per lattice cell. A cell with four distinct
        // corners is a quad; a cell that lost a corner to a scaffold collapse has
        // a consecutive duplicate, which drops out to leave a triangle. A fully
        // degenerate cell produces no face.
        let mut result = GridFillResult::default();
        for x in 0..cols - 1 {
            for y in 0..rows - 1 {
                let raw: [VertexKey; 4] = if flip {
                    [
                        grid.get(x, y),
                        grid.get(x + 1, y),
                        grid.get(x + 1, y + 1),
                        grid.get(x, y + 1),
                    ]
                } else {
                    [
                        grid.get(x, y),
                        grid.get(x, y + 1),
                        grid.get(x + 1, y + 1),
                        grid.get(x + 1, y),
                    ]
                };
                let mut corners: SmallVec<[VertexKey; 4]> = SmallVec::new();
                for &vertex in &raw {
                    if corners.last() != Some(&vertex) {
                        corners.push(vertex);
                    }
                }
                if corners.len() > 1 && corners.first() == corners.last() {
                    corners.pop();
                }
                if corners.len() < 3 {
                    continue;
                }
                let face = self.face_create_verts(&corners, FaceCreateFlags::NO_DUPLICATE);
                if params.mat_nr != 0 {
                    self.set_face_material_index(face, params.mat_nr);
                }
                if params.use_smooth {
                    self.set_face_smooth(face, true);
                }
                result.faces.push(face);
            }
        }

        result
    }

    /// §3.1 — discover the open edge loops in the marked input. Returns `None`
    /// when the input is malformed for a grid fill: a self-loop edge, a branch
    /// vertex (valence > 2 over the input set), or a leftover closed loop (a
    /// cycle with no endpoints — the single-closed-loop case of §1.2/§8). Each
    /// returned loop is an ordered open vertex chain.
    fn grid_discover_open_loops(&self, edges: &[EdgeKey]) -> Option<Vec<Vec<VertexKey>>> {
        if edges.is_empty() {
            return None;
        }

        // Adjacency over the (deduplicated) input edge set.
        let mut adjacency: HashMap<VertexKey, SmallVec<[(VertexKey, EdgeKey); 2]>> = HashMap::new();
        let mut seen_input: HashSet<EdgeKey> = HashSet::new();
        let mut edge_list: Vec<EdgeKey> = Vec::with_capacity(edges.len());
        for &edge in edges {
            if !seen_input.insert(edge) {
                continue;
            }
            let [a, b] = self.edge_vert_keys(edge);
            if a == b {
                return None;
            }
            adjacency.entry(a).or_default().push((b, edge));
            adjacency.entry(b).or_default().push((a, edge));
            edge_list.push(edge);
        }

        // A grid side is a simple chain: every vertex carries one or two input
        // edges. A branch vertex makes the side ambiguous (§3.1).
        for incident in adjacency.values() {
            if incident.len() > 2 {
                return None;
            }
        }
        let endpoints: Vec<VertexKey> = adjacency
            .iter()
            .filter(|(_, incident)| incident.len() == 1)
            .map(|(&v, _)| v)
            .collect();

        let mut visited_edges: HashSet<EdgeKey> = HashSet::new();
        let mut loops: Vec<Vec<VertexKey>> = Vec::new();
        for &start in &endpoints {
            let start_edge = adjacency[&start][0].1;
            if visited_edges.contains(&start_edge) {
                continue;
            }
            let mut path = vec![start];
            let mut current = start;
            let mut arrived: Option<EdgeKey> = None;
            loop {
                let mut advanced = false;
                for &(next_vertex, next_edge) in &adjacency[&current] {
                    if Some(next_edge) == arrived {
                        continue;
                    }
                    visited_edges.insert(next_edge);
                    path.push(next_vertex);
                    current = next_vertex;
                    arrived = Some(next_edge);
                    advanced = true;
                    break;
                }
                if !advanced {
                    break;
                }
            }
            loops.push(path);
        }

        // Any edge not reached from an endpoint belongs to a closed loop, which
        // has no endpoints to attach rails to (§3.1).
        if visited_edges.len() != edge_list.len() {
            return None;
        }

        Some(loops)
    }

    /// §3.2 — find a rail: an ordered path of wire-or-boundary edges from
    /// `start` to `target`, with the `hidden` side edges excluded so the search
    /// cannot run along the input loops. Returns the vertex chain `start..=target`
    /// or `None` when no such path exists.
    fn grid_find_rail(
        &self,
        start: VertexKey,
        target: VertexKey,
        hidden: &HashSet<EdgeKey>,
    ) -> Option<Vec<VertexKey>> {
        if start == target {
            return None;
        }
        let mut predecessor: HashMap<VertexKey, VertexKey> = HashMap::new();
        let mut visited: HashSet<VertexKey> = HashSet::new();
        let mut queue: VecDeque<VertexKey> = VecDeque::new();
        visited.insert(start);
        queue.push_back(start);
        while let Some(vertex) = queue.pop_front() {
            if vertex == target {
                break;
            }
            for edge in self.iter_disk_edges(vertex) {
                if hidden.contains(&edge) {
                    continue;
                }
                // A rail edge is wire (no face) or boundary (one face).
                if self.iter_radial_loops(edge).count() > 1 {
                    continue;
                }
                let [v0, v1] = self.edge_vert_keys(edge);
                let other = if v0 == vertex { v1 } else { v0 };
                if visited.insert(other) {
                    predecessor.insert(other, vertex);
                    queue.push_back(other);
                }
            }
        }
        if !visited.contains(&target) {
            return None;
        }
        let mut path = vec![target];
        let mut current = target;
        while current != start {
            current = predecessor[&current];
            path.push(current);
        }
        path.reverse();
        Some(path)
    }

    /// §3.4 Phase 1+2 — reconcile a boundary member of `members` real vertices to
    /// the lattice's full `target` slot count. Every original member vertex is
    /// preserved as a lattice slot; the `target - count` extra slots are
    /// *scaffold* vertices inserted into the member's segments by a count-driven
    /// even-distribution schedule walked **from the origin end** (the member is
    /// pre-oriented origin-first, so segment `s` is `[members[s], members[s+1]]`).
    /// The schedule — segment `s` receives
    /// `floor((s+1)·extra / segments) − floor(s·extra / segments)` scaffolds — is
    /// a function of the counts alone, never of input ordering or geometry.
    ///
    /// Returns the per-slot positions and the per-slot vertex keys. Each scaffold
    /// slot is a **parameter-0 split (§3.4a-a)**: its position is placed
    /// *coincident with* its collapse-survivor endpoint — not at the segment
    /// midpoint — so the profile handed to the §5/§6 interior solver repeats an
    /// endpoint coordinate at each scaffold slot rather than an even-subdivision
    /// point. The survivor is that segment's origin-ward — lower-index — endpoint,
    /// the deterministic collapse target of §3.4 (the endpoint reached first
    /// walking from `grid[0][0]`), and the later Phase-2 collapse into that same
    /// endpoint is a geometric no-op. The two member endpoints are always slot 0
    /// and slot `target-1`, so the four lattice corners stay real. An already-full
    /// member is the identity.
    fn grid_expand_member(
        &self,
        members: &[VertexKey],
        target: usize,
    ) -> (Vec<Vec3>, Vec<VertexKey>) {
        let real_positions: Vec<Vec3> =
            members.iter().map(|&v| self.vertex_position(v)).collect();
        let count = members.len();
        if count == target {
            return (real_positions, members.to_vec());
        }

        let extra = target - count;
        let segments = count - 1;
        let mut positions = Vec::with_capacity(target);
        let mut keys = Vec::with_capacity(target);
        let mut inserted = 0usize;
        for s in 0..segments {
            // The segment's origin-ward endpoint — a real member vertex, and the
            // collapse survivor for every scaffold this segment carries.
            positions.push(real_positions[s]);
            keys.push(members[s]);

            // Even-distribution schedule (Bresenham-style spread from the origin
            // end): the running count up to and including this segment, minus what
            // earlier segments already took.
            let upto = (s + 1) * extra / segments;
            let here = upto - inserted;
            inserted = upto;
            for _ in 0..here {
                // §3.4a-a — parameter-0 split: the scaffold is placed *coincident*
                // with its collapse-survivor (the segment's origin-ward endpoint),
                // so the boundary profile fed to the §5/§6 interior solver carries a
                // repeated endpoint coordinate here, not an even-subdivision
                // midpoint. Collapsing into that same endpoint (below) moves no
                // geometry.
                positions.push(real_positions[s]);
                keys.push(members[s]);
            }
        }
        // The final real member vertex closes the walk at slot `target-1`.
        positions.push(real_positions[count - 1]);
        keys.push(members[count - 1]);
        (positions, keys)
    }

    /// §5/§6 — fill every interior lattice cell with a freshly created vertex,
    /// interpolated from the four expanded boundary profiles (§3.4 Phase 1).
    fn grid_build_interior(
        &mut self,
        grid: &mut Lattice,
        bottom_side: &[Vec3],
        top_side: &[Vec3],
        left_rail: &[Vec3],
        right_rail: &[Vec3],
        use_interp_simple: bool,
    ) {
        let cols = grid.cols;
        let rows = grid.rows;
        if cols < 3 || rows < 3 {
            return;
        }

        // §5.1 — the bottom and top row frames are fixed for the whole fill.
        let frame_bottom = grid_row_frame(left_rail, right_rail, 0);
        let frame_top = grid_row_frame(left_rail, right_rail, rows - 1);

        for x in 1..cols - 1 {
            for y in 1..rows - 1 {
                let u = x as f32 / (cols - 1) as f32;
                let v = y as f32 / (rows - 1) as f32;

                let position = if use_interp_simple {
                    // §6.1 — reciprocal-distance convex partition of unity over
                    // the four boundary verts sharing this cell's column/row.
                    grid_simple_blend(u, v, bottom_side[x], left_rail[y], top_side[x], right_rail[y])
                } else {
                    // §5.3 — carry this column's bottom and top profile verts
                    // through their own row frame into the current row's frame,
                    // then blend by the row's fractional height. The per-row
                    // frame rotation is what makes a curved boundary yield a
                    // curved interior rather than the flat blend of §6.
                    let frame_row = grid_row_frame(left_rail, right_rail, y);
                    let co_a = grid_frame_transform(bottom_side[x], frame_bottom, frame_row);
                    let co_b = grid_frame_transform(top_side[x], frame_top, frame_row);
                    co_a.lerp(co_b, v)
                };

                let vertex = self.create_vertex(position);

                // Interior vertex custom-data always blends the four boundary
                // verts by the §6.1 reciprocal-distance partition of unity —
                // the same weights simple-mode position uses, and unconditional
                // of `use_interp_simple` (§5.3). The linear side-weighting would
                // only agree at a centered interior vertex.
                if !self.vertex_attribs.is_empty() {
                    let [w_bottom, w_left, w_top, w_right] = grid_recip_weights(u, v);
                    let sources = [
                        (w_bottom, grid.get(x, 0).as_usize()),
                        (w_left, grid.get(0, y).as_usize()),
                        (w_top, grid.get(x, rows - 1).as_usize()),
                        (w_right, grid.get(cols - 1, y).as_usize()),
                    ];
                    self.vertex_attribs
                        .interpolate_into(vertex.as_usize(), &sources);
                }

                grid.set(x, y, vertex);
            }
        }
    }

    /// §4.2 — winding vote. Each perimeter edge that already borders exactly one
    /// face votes on whether the grid's default build would wind incoherently
    /// with that surface; a positive tally flips the build. A free wire frame
    /// scores zero and keeps the default orientation.
    fn grid_winding_flip(&self, grid: &Lattice) -> bool {
        let cols = grid.cols;
        let rows = grid.rows;
        let mut tally: i32 = 0;

        // The directed perimeter edges as the *default* build traverses them.
        // Opposite sides run in opposite senses, which the directions encode
        // directly (so no separate per-side sign is needed).
        let mut directed: Vec<(VertexKey, VertexKey)> = Vec::new();
        for x in 0..cols - 1 {
            directed.push((grid.get(x + 1, 0), grid.get(x, 0)));
            directed.push((grid.get(x, rows - 1), grid.get(x + 1, rows - 1)));
        }
        for y in 0..rows - 1 {
            directed.push((grid.get(0, y), grid.get(0, y + 1)));
            directed.push((grid.get(cols - 1, y + 1), grid.get(cols - 1, y)));
        }

        for (from_vertex, to_vertex) in directed {
            let Some(edge) = self.query_edge_exists(from_vertex, to_vertex) else {
                continue;
            };
            let radial = self.edge_loop_keys(edge);
            if radial.len() != 1 {
                continue;
            }
            // Coherent neighbours traverse a shared edge in opposite senses. If
            // the bordering face already traverses from→to the way the grid
            // would, the grid is incoherent → vote to flip.
            if self.loop_vertex(radial[0]) == from_vertex {
                tally += 1;
            } else {
                tally -= 1;
            }
        }

        tally > 0
    }
}

/// §3.3 — the two rails overlap when they share any vertex (a clean rectangle's
/// rails share only the — distinct — corners, which live one on each rail).
fn grid_rails_overlap(rail_left: &[VertexKey], rail_right: &[VertexKey]) -> bool {
    let left: HashSet<VertexKey> = rail_left.iter().copied().collect();
    rail_right.iter().any(|v| left.contains(v))
}

/// §6.1 — the four reciprocal-distance weights `[bottom, left, top, right]`,
/// normalised to a partition of unity. Each side's unnormalised weight is the
/// reciprocal of the lattice distance to it, so the blend tracks the nearest
/// boundary. Drives both the simple-mode interior position and — unconditional
/// of `use_interp_simple` — the interior vertex-CD blend (§5.3).
fn grid_recip_weights(u: f32, v: f32) -> [f32; 4] {
    let w_bottom = 1.0 / v;
    let w_left = 1.0 / u;
    let w_top = 1.0 / (1.0 - v);
    let w_right = 1.0 / (1.0 - u);
    let sum = w_bottom + w_left + w_top + w_right;
    [w_bottom / sum, w_left / sum, w_top / sum, w_right / sum]
}

/// §6.1 — the reciprocal-distance convex blend of the four boundary verts.
fn grid_simple_blend(u: f32, v: f32, bottom: Vec3, left: Vec3, top: Vec3, right: Vec3) -> Vec3 {
    let [w_bottom, w_left, w_top, w_right] = grid_recip_weights(u, v);
    w_bottom * bottom + w_left * left + w_top * top + w_right * right
}

/// §5.1 — the row frame: a triangle `[L, R, apex]` whose base is the row's
/// left-rail to right-rail segment and whose apex sits a base-length away along
/// the averaged direction in which the two rails advance from this row toward
/// its neighbour(s). The apex direction is what makes the interior follow the
/// boundary's bow rather than lie flat.
fn grid_row_frame(left: &[Vec3], right: &[Vec3], y: usize) -> [Vec3; 3] {
    let l = left[y];
    let r = right[y];
    let progression = grid_progression(left, right, y);
    let apex = (l + r) * 0.5 + progression * (r - l).length();
    [l, r, apex]
}

/// §5.1 — the unit progression direction for row `y`: the averaged rail step
/// toward the neighbouring row(s). The bottom row steps forward, the top row
/// steps backward and negates (so its apex points the same way as the bottom's),
/// and an interior row averages a forward and a backward estimate.
fn grid_progression(left: &[Vec3], right: &[Vec3], y: usize) -> Vec3 {
    let rows = left.len();
    let unit = |v: Vec3| v.normalize_or_zero();
    // Averaged unit step of the two rails from row `a` to row `b`.
    let step = |a: usize, b: usize| unit(unit(left[b] - left[a]) + unit(right[b] - right[a]));
    if y == 0 {
        step(0, 1)
    } else if y == rows - 1 {
        -step(rows - 1, rows - 2)
    } else {
        unit(step(y, y + 1) + step(y - 1, y))
    }
}

/// §5.2 — carry a point from a source row frame into a target row frame. The
/// in-plane part is handled by (un-clamped) triangle barycentric coordinates;
/// the signed height above the source plane is carried across and rescaled by
/// the linear size ratio of the two triangles.
fn grid_frame_transform(point: Vec3, src: [Vec3; 3], tar: [Vec3; 3]) -> Vec3 {
    let n_src = grid_tri_normal(src);
    let n_tar = grid_tri_normal(tar);

    // Orthonormal frame whose z axis is the source normal; (x, y) lie in the
    // source plane, z is signed height.
    let (e1, e2) = n_src.any_orthonormal_pair();
    let project = |q: Vec3| Vec3::new(q.dot(e1), q.dot(e2), q.dot(n_src));
    let p = project(point);
    let s = [project(src[0]), project(src[1]), project(src[2])];

    // Affine (un-clamped) barycentric weights in the source plane.
    let w = grid_barycentric_2d(p.truncate(), s[0].truncate(), s[1].truncate(), s[2].truncate());
    let in_plane = w[0] * tar[0] + w[1] * tar[1] + w[2] * tar[2];

    // Out-of-plane height, scaled by the linear size ratio of the triangles.
    let height = p.z - s[0].z;
    let area_tar = grid_tri_area_3d(tar);
    let area_src = grid_tri_area_2d(s[0].truncate(), s[1].truncate(), s[2].truncate());
    let scale = if area_src > 1.0e-20 {
        area_tar.sqrt() / area_src.sqrt()
    } else {
        1.0
    };
    in_plane + n_tar * (height * scale)
}

/// Unit normal of a triangle, from its fixed `[a, b, c]` vertex order. The §5.2
/// result is invariant to the sign convention as long as it is consistent.
fn grid_tri_normal(t: [Vec3; 3]) -> Vec3 {
    (t[1] - t[0]).cross(t[2] - t[0]).normalize_or_zero()
}

fn grid_tri_area_3d(t: [Vec3; 3]) -> f32 {
    0.5 * (t[1] - t[0]).cross(t[2] - t[0]).length()
}

fn grid_tri_area_2d(a: glam::Vec2, b: glam::Vec2, c: glam::Vec2) -> f32 {
    0.5 * (b - a).perp_dot(c - a).abs()
}

/// Affine barycentric weights of `p` within triangle `(a, b, c)` in 2D. The
/// three weights sum to one but may fall outside `[0, 1]` (extrapolation).
fn grid_barycentric_2d(p: glam::Vec2, a: glam::Vec2, b: glam::Vec2, c: glam::Vec2) -> [f32; 3] {
    let v0 = b - a;
    let v1 = c - a;
    let v2 = p - a;
    let den = v0.perp_dot(v1);
    // A collinear/degenerate source frame (den ≈ 0) has no well-defined
    // barycentric coordinates; dividing would emit inf/NaN into the interior
    // vertex position. Collapse to the first source vertex — a finite,
    // deterministic fallback. Guarded at the same magnitude as `area_src` in
    // `grid_frame_transform`; unreachable on non-degenerate frames.
    if den.abs() <= 1.0e-20 {
        return [1.0, 0.0, 0.0];
    }
    let w1 = v2.perp_dot(v1) / den;
    let w2 = v0.perp_dot(v2) / den;
    [1.0 - w1 - w2, w1, w2]
}

#[cfg(test)]
mod tests {
    use glam::Vec3;

    use super::GridFillParams;
    use crate::loopmesh::*;

    /// A perimeter-only rectangular wire ring: `cols × rows` lattice with only
    /// its boundary present. Returns the bottom+top side edges (the grid-fill
    /// input) — the two vertical rails are left in the mesh for discovery.
    /// `warp` lifts the boundary into a saddle so the two interpolation modes
    /// diverge.
    fn rect_wire_ring(mesh: &mut LoopMesh, cols: usize, rows: usize, warp: f32) -> Vec<EdgeKey> {
        let on_perimeter = |x: usize, y: usize| x == 0 || x == cols - 1 || y == 0 || y == rows - 1;
        let is_corner =
            |x: usize, y: usize| (x == 0 || x == cols - 1) && (y == 0 || y == rows - 1);
        let z = |x: usize, y: usize| -> f32 {
            // Bow the boundary: every non-corner boundary vertex is lifted to
            // `warp`, the four corners stay at z = 0. This is the GRID_FILL.md
            // §5/§6 worked-check frame (a flat warp = 0 frame is the regular
            // lattice; a non-zero warp makes the two interp modes diverge).
            if on_perimeter(x, y) && !is_corner(x, y) {
                warp
            } else {
                0.0
            }
        };
        let mut vert = vec![vec![VertexKey(u32::MAX); rows]; cols];
        for x in 0..cols {
            for y in 0..rows {
                if on_perimeter(x, y) {
                    vert[x][y] = mesh.create_vertex(Vec3::new(x as f32, y as f32, z(x, y)));
                }
            }
        }
        let edge = |mesh: &mut LoopMesh, a: VertexKey, b: VertexKey| {
            mesh.create_edge([a, b], EdgeCreateFlags::NO_DUPLICATE)
        };
        let mut side_edges = Vec::new();
        // Bottom (y=0) and top (y=rows-1) sides — the input loops.
        for x in 0..cols - 1 {
            side_edges.push(edge(mesh, vert[x][0], vert[x + 1][0]));
            side_edges.push(edge(mesh, vert[x][rows - 1], vert[x + 1][rows - 1]));
        }
        // Left (x=0) and right (x=cols-1) rails — discovered, not input.
        for y in 0..rows - 1 {
            edge(mesh, vert[0][y], vert[0][y + 1]);
            edge(mesh, vert[cols - 1][y], vert[cols - 1][y + 1]);
        }
        side_edges
    }

    #[test]
    fn flat_rings_fill_to_quad_grids() {
        // (cols, rows, expected quads, expected new interior verts).
        for &(cols, rows, quads, interior) in &[
            (2usize, 2usize, 1usize, 0usize),
            (3, 3, 4, 1),
            (4, 3, 6, 2),
            (4, 4, 9, 4),
        ] {
            let mut mesh = LoopMesh::new();
            let side_edges = rect_wire_ring(&mut mesh, cols, rows, 0.0);
            let v0 = mesh.vertex_count();
            let f0 = mesh.face_count();

            let result = mesh.grid_fill(&side_edges, &GridFillParams::default());

            assert_eq!(
                result.faces().len(),
                quads,
                "({cols}x{rows}) faces.out lists every new quad",
            );
            assert_eq!(
                mesh.face_count() - f0,
                quads,
                "({cols}x{rows}) (cols-1)(rows-1) quads created",
            );
            assert_eq!(
                mesh.vertex_count() - v0,
                interior,
                "({cols}x{rows}) only the (cols-2)(rows-2) interior verts created",
            );
            for &face in result.faces() {
                assert_eq!(mesh.face_len(face), 4, "({cols}x{rows}) every cell is a quad");
            }
            mesh.validate_topology();
        }
    }

    #[test]
    fn interior_vertex_sits_at_center_of_flat_3x3() {
        let mut mesh = LoopMesh::new();
        let side_edges = rect_wire_ring(&mut mesh, 3, 3, 0.0);
        let v_before = mesh.vertex_count();
        let result = mesh.grid_fill(&side_edges, &GridFillParams::default());
        assert_eq!(result.faces().len(), 4);
        assert_eq!(mesh.vertex_count() - v_before, 1, "one interior vert");

        // The lone interior vertex is the one not on the perimeter; on a flat
        // regular 3x3 frame it sits at the centre (1, 1, 0).
        let mut interior = None;
        let mut it = mesh.mesh_verts();
        while let Some(v) = it.next(&mesh) {
            let p = mesh.vertex_position(v);
            if (p.x - 1.0).abs() < 1.0e-4 && (p.y - 1.0).abs() < 1.0e-4 {
                interior = Some(p);
            }
        }
        let p = interior.expect("interior vertex at the lattice centre");
        assert!((p - Vec3::new(1.0, 1.0, 0.0)).length() < 1.0e-4);
        mesh.validate_topology();
    }

    #[test]
    fn interp_modes_diverge_on_warped_frame() {
        // The lone interior vertex of a 3x3 saddle lands at different points for
        // the two interpolation modes (the discriminating §5/§6 condition).
        let interior_pos = |use_interp_simple: bool| -> Vec3 {
            let mut mesh = LoopMesh::new();
            let side_edges = rect_wire_ring(&mut mesh, 3, 3, 1.2);
            let v_before = mesh.vertex_count();
            mesh.grid_fill(
                &side_edges,
                &GridFillParams {
                    use_interp_simple,
                    ..Default::default()
                },
            );
            assert_eq!(mesh.vertex_count() - v_before, 1);
            // The newest vertex is the interior one.
            let mut last = None;
            let mut it = mesh.mesh_verts();
            while let Some(v) = it.next(&mesh) {
                last = Some(v);
            }
            mesh.vertex_position(last.unwrap())
        };

        let simple = interior_pos(true);
        let transform = interior_pos(false);
        assert!(
            (simple - transform).length() > 1.0e-3,
            "the two interpolation modes must place the interior vert differently \
             (simple={simple:?} transform={transform:?})",
        );
    }

    /// The position of the (unique) vertex whose `xy` lands near `(x, y)`.
    fn vert_near(mesh: &LoopMesh, x: f32, y: f32) -> Vec3 {
        let mut found = None;
        let mut it = mesh.mesh_verts();
        while let Some(v) = it.next(mesh) {
            let p = mesh.vertex_position(v);
            if (p.x - x).abs() < 1.0e-3 && (p.y - y).abs() < 1.0e-3 {
                assert!(found.is_none(), "more than one vertex near ({x}, {y})");
                found = Some(p);
            }
        }
        found.unwrap_or_else(|| panic!("no vertex near ({x}, {y})"))
    }

    #[test]
    fn simple_interp_centered_saddle_3x3() {
        // §6.1 worked check: u = v = 0.5 ⇒ all four reciprocal weights are 2,
        // normalise to 0.25; the four boundary midpoints all sit at z = 0.6, so
        // the lone interior vert lands at (1, 1, 0.6) — a convex blend (a
        // partition-of-unity bug would double the height to 1.2).
        let mut mesh = LoopMesh::new();
        let side_edges = rect_wire_ring(&mut mesh, 3, 3, 0.6);
        mesh.grid_fill(
            &side_edges,
            &GridFillParams {
                use_interp_simple: true,
                ..Default::default()
            },
        );
        let p = vert_near(&mesh, 1.0, 1.0);
        assert!(
            (p - Vec3::new(1.0, 1.0, 0.6)).length() < 1.0e-4,
            "simple-mode interior vert at (1,1,0.6), got {p:?}",
        );
        mesh.validate_topology();
    }

    #[test]
    fn simple_interp_offcenter_4x3() {
        // §6.1 off-center worked check: grid[1][1] sits at u = 1/3, v = 1/2 with
        // asymmetric weights (2, 3, 2, 1.5)/8.5, landing it at (1, 1, 0.6) —
        // above its own column, not drifted to x = 1.25 as a uniform 0.25-each
        // (symmetric) weighting would. Its mirror grid[2][1] lands at (2,1,0.6).
        let mut mesh = LoopMesh::new();
        let side_edges = rect_wire_ring(&mut mesh, 4, 3, 0.6);
        mesh.grid_fill(
            &side_edges,
            &GridFillParams {
                use_interp_simple: true,
                ..Default::default()
            },
        );
        let p1 = vert_near(&mesh, 1.0, 1.0);
        let p2 = vert_near(&mesh, 2.0, 1.0);
        assert!(
            (p1 - Vec3::new(1.0, 1.0, 0.6)).length() < 1.0e-4,
            "grid[1][1] at (1,1,0.6), got {p1:?}",
        );
        assert!(
            (p2 - Vec3::new(2.0, 1.0, 0.6)).length() < 1.0e-4,
            "grid[2][1] at (2,1,0.6), got {p2:?}",
        );
        mesh.validate_topology();
    }

    #[test]
    fn default_interp_saddle_3x3_lifts_above_simple() {
        // §5.3 worked check: the per-row frame apexes bow the interior well
        // above the z = 0.6 plane of the four side midpoints — BMesh lands the
        // lone interior vert at z ≈ 1.115, distinct from the simple mode's 0.6.
        let mut mesh = LoopMesh::new();
        let side_edges = rect_wire_ring(&mut mesh, 3, 3, 0.6);
        mesh.grid_fill(&side_edges, &GridFillParams::default());
        let p = vert_near(&mesh, 1.0, 1.0);
        assert!(
            (p.x - 1.0).abs() < 1.0e-4 && (p.y - 1.0).abs() < 1.0e-4,
            "default interior vert centred over its column, got {p:?}",
        );
        assert!(
            (p.z - 1.115).abs() < 2.0e-2,
            "default-mode interior vert lifted to z ≈ 1.115, got z = {}",
            p.z,
        );
        mesh.validate_topology();
    }

    #[test]
    fn interior_vertex_cd_uses_reciprocal_weights_offcenter() {
        // §5.3/§6.1: interior vertex custom-data blends the four boundary verts
        // by the reciprocal-distance partition of unity, *unconditional* of
        // `use_interp_simple`. On a 4x3 frame the interior vert grid[1][1] sits
        // off-center at u = 1/3, v = 1/2, where the reciprocal bottom weight is
        // 2/8.5 ≈ 0.2353 — distinct from the linear side-weighting's 0.25. A
        // centered 3x3 interior vert (u = v = 1/2) cannot discriminate the two.
        let recip_bottom = 2.0_f32 / 8.5; // ≈ 0.23529
        let linear_bottom = 0.25_f32;

        // Run in both modes: vertex-CD must not be gated by `use_interp_simple`.
        for use_interp_simple in [false, true] {
            let mut mesh = LoopMesh::new();
            let side_edges = rect_wire_ring(&mut mesh, 4, 3, 0.0);

            let cd = mesh
                .register_vertex_attribute(ElementAttributeDesc::numeric(
                    AttributeKey::custom("weightprobe"),
                    AttrType::F32,
                ))
                .unwrap();
            // Isolate the bottom source of grid[1][1]: only the boundary vert at
            // (1, 0) carries 1.0; every other boundary vert carries 0.0. The
            // interior blend then reads back exactly that vertex's normalised
            // weight.
            let mut it = mesh.mesh_verts();
            while let Some(v) = it.next(&mesh) {
                let p = mesh.vertex_position(v);
                let value = if (p.x - 1.0).abs() < 1.0e-3 && p.y.abs() < 1.0e-3 {
                    1.0
                } else {
                    0.0
                };
                mesh.vertex_attribs_mut().set_f32(cd, v.as_usize(), value);
            }

            mesh.grid_fill(
                &side_edges,
                &GridFillParams {
                    use_interp_simple,
                    ..Default::default()
                },
            );

            // grid[1][1] lands over its own column at (1, 1, 0) in both modes on
            // this flat frame.
            let mut interior = None;
            let mut it = mesh.mesh_verts();
            while let Some(v) = it.next(&mesh) {
                let p = mesh.vertex_position(v);
                if (p.x - 1.0).abs() < 1.0e-3 && (p.y - 1.0).abs() < 1.0e-3 {
                    assert!(interior.is_none(), "one interior vert near (1, 1)");
                    interior = Some(v);
                }
            }
            let interior = interior.expect("interior vert grid[1][1]");
            let blended = mesh
                .vertex_attribs()
                .get_f32(cd, interior.as_usize())
                .expect("probe layer present on interior vert");

            assert!(
                (blended - recip_bottom).abs() < 1.0e-4,
                "interior vertex-CD is the reciprocal bottom weight {recip_bottom} \
                 (simple={use_interp_simple}), got {blended}",
            );
            assert!(
                (blended - linear_bottom).abs() > 1.0e-2,
                "interior vertex-CD must NOT be the linear side-weight {linear_bottom} \
                 (simple={use_interp_simple}), got {blended}",
            );
        }
    }

    #[test]
    fn mat_nr_and_use_smooth_stamp_every_quad() {
        let mut mesh = LoopMesh::new();
        let side_edges = rect_wire_ring(&mut mesh, 4, 3, 0.0);
        let params = GridFillParams {
            mat_nr: 2,
            use_smooth: true,
            use_interp_simple: false,
        };
        let result = mesh.grid_fill(&side_edges, &params);
        assert_eq!(result.faces().len(), 6);
        for &face in result.faces() {
            assert_eq!(mesh.face_material_index(face), 2, "mat_nr stamped on the quad");
            assert!(mesh.is_face_smooth(face), "use_smooth stamped on the quad");
        }
        mesh.validate_topology();
    }

    #[test]
    fn default_params_leave_quads_at_slot_zero_and_flat() {
        let mut mesh = LoopMesh::new();
        let side_edges = rect_wire_ring(&mut mesh, 3, 3, 0.0);
        let result = mesh.grid_fill(&side_edges, &GridFillParams::default());
        assert_eq!(result.faces().len(), 4);
        for &face in result.faces() {
            assert_eq!(mesh.face_material_index(face), 0, "default mat_nr is slot 0");
            assert!(!mesh.is_face_smooth(face), "default quads are flat");
        }
        mesh.validate_topology();
    }

    #[test]
    fn unconnected_parallel_loops_create_nothing() {
        // Two parallel open chains with no rails between them: rail discovery
        // finds no wire/boundary path → reject (§3.2).
        let mut mesh = LoopMesh::new();
        let bottom: Vec<VertexKey> = (0..3)
            .map(|x| mesh.create_vertex(Vec3::new(x as f32, 0.0, 0.0)))
            .collect();
        let top: Vec<VertexKey> = (0..3)
            .map(|x| mesh.create_vertex(Vec3::new(x as f32, 2.0, 0.0)))
            .collect();
        let mut side_edges = Vec::new();
        for x in 0..2 {
            side_edges.push(mesh.create_edge([bottom[x], bottom[x + 1]], EdgeCreateFlags::NO_DUPLICATE));
            side_edges.push(mesh.create_edge([top[x], top[x + 1]], EdgeCreateFlags::NO_DUPLICATE));
        }
        let (v0, e0, f0) = (mesh.vertex_count(), mesh.edge_count(), mesh.face_count());

        let result = mesh.grid_fill(&side_edges, &GridFillParams::default());

        assert!(result.is_empty(), "no rails → nothing filled");
        assert_eq!(
            (mesh.vertex_count(), mesh.edge_count(), mesh.face_count()),
            (v0, e0, f0),
            "mesh left unchanged",
        );
        mesh.validate_topology();
    }

    #[test]
    fn single_closed_loop_is_rejected() {
        // A single closed quad ring discovers as one (closed) loop → the
        // two-open-loops gate rejects it (§3.1/§8).
        let mut mesh = LoopMesh::new();
        let ring: Vec<VertexKey> = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        ]
        .iter()
        .map(|&p| mesh.create_vertex(p))
        .collect();
        let edges: Vec<EdgeKey> = (0..4)
            .map(|i| mesh.create_edge([ring[i], ring[(i + 1) % 4]], EdgeCreateFlags::NO_DUPLICATE))
            .collect();
        let (v0, e0, f0) = (mesh.vertex_count(), mesh.edge_count(), mesh.face_count());

        let result = mesh.grid_fill(&edges, &GridFillParams::default());

        assert!(result.is_empty(), "single closed loop → rejected");
        assert_eq!(
            (mesh.vertex_count(), mesh.edge_count(), mesh.face_count()),
            (v0, e0, f0),
        );
        mesh.validate_topology();
    }

    /// §3.4 mismatched-sides fixture: a bottom side of `bottom_cols` vertices
    /// opposing a top side of `top_cols`, joined by single-edge rails. Returns the
    /// side edges (the grid-fill input); the two rails are left for discovery.
    fn unequal_sided_wire_ring(
        mesh: &mut LoopMesh,
        bottom_cols: usize,
        top_cols: usize,
    ) -> Vec<EdgeKey> {
        let width = bottom_cols.max(top_cols) - 1;
        let bottom: Vec<VertexKey> = (0..bottom_cols)
            .map(|x| {
                let fx = x as f32 * width as f32 / (bottom_cols - 1) as f32;
                mesh.create_vertex(Vec3::new(fx, 0.0, 0.0))
            })
            .collect();
        let top: Vec<VertexKey> = (0..top_cols)
            .map(|x| {
                let fx = x as f32 * width as f32 / (top_cols - 1) as f32;
                mesh.create_vertex(Vec3::new(fx, 2.0, 0.0))
            })
            .collect();
        let edge = |mesh: &mut LoopMesh, a: VertexKey, b: VertexKey| {
            mesh.create_edge([a, b], EdgeCreateFlags::NO_DUPLICATE)
        };
        let mut side_edges = Vec::new();
        for x in 0..bottom_cols - 1 {
            side_edges.push(edge(mesh, bottom[x], bottom[x + 1]));
        }
        for x in 0..top_cols - 1 {
            side_edges.push(edge(mesh, top[x], top[x + 1]));
        }
        // Single-edge rails connecting the matching corners — discovered, not input.
        edge(mesh, bottom[0], top[0]);
        edge(mesh, bottom[bottom_cols - 1], top[top_cols - 1]);
        side_edges
    }

    /// §3.4 mismatched-rails fixture: single-edge opposing sides joined by a left
    /// rail of `left_rows` vertices opposing a right rail of `right_rows`. Returns
    /// the side edges (the input); the rails are left for discovery.
    fn unequal_railed_wire_ring(
        mesh: &mut LoopMesh,
        left_rows: usize,
        right_rows: usize,
    ) -> Vec<EdgeKey> {
        let height = left_rows.max(right_rows) - 1;
        let left: Vec<VertexKey> = (0..left_rows)
            .map(|y| {
                let fy = y as f32 * height as f32 / (left_rows - 1) as f32;
                mesh.create_vertex(Vec3::new(0.0, fy, 0.0))
            })
            .collect();
        let right: Vec<VertexKey> = (0..right_rows)
            .map(|y| {
                let fy = y as f32 * height as f32 / (right_rows - 1) as f32;
                mesh.create_vertex(Vec3::new(2.0, fy, 0.0))
            })
            .collect();
        let edge = |mesh: &mut LoopMesh, a: VertexKey, b: VertexKey| {
            mesh.create_edge([a, b], EdgeCreateFlags::NO_DUPLICATE)
        };
        // Single-edge bottom + top sides — the input loops.
        let side_edges = vec![
            edge(mesh, left[0], right[0]),
            edge(mesh, left[left_rows - 1], right[right_rows - 1]),
        ];
        // The two rails — discovered, not input.
        for y in 0..left_rows - 1 {
            edge(mesh, left[y], left[y + 1]);
        }
        for y in 0..right_rows - 1 {
            edge(mesh, right[y], right[y + 1]);
        }
        side_edges
    }

    /// §3.4a non-minimal fixture: `cols`-vertex opposing sides (so a genuine
    /// interior vertex is created) joined by a left rail of `left_rows` vertices
    /// opposing a right rail of `right_rows`. Returns the side edges (the input);
    /// the two rails are left for discovery.
    fn unequal_railed_interior_wire_ring(
        mesh: &mut LoopMesh,
        cols: usize,
        left_rows: usize,
        right_rows: usize,
    ) -> Vec<EdgeKey> {
        let width = (cols - 1) as f32;
        let height = (left_rows.max(right_rows) - 1) as f32;
        let bottom: Vec<VertexKey> = (0..cols)
            .map(|x| mesh.create_vertex(Vec3::new(x as f32, 0.0, 0.0)))
            .collect();
        let top: Vec<VertexKey> = (0..cols)
            .map(|x| mesh.create_vertex(Vec3::new(x as f32, height, 0.0)))
            .collect();
        let left: Vec<VertexKey> = (0..left_rows)
            .map(|y| {
                let fy = y as f32 * height / (left_rows - 1) as f32;
                // The two corners coincide with the side endpoints; only the
                // genuine interior rail vertices are fresh.
                if y == 0 {
                    bottom[0]
                } else if y == left_rows - 1 {
                    top[0]
                } else {
                    mesh.create_vertex(Vec3::new(0.0, fy, 0.0))
                }
            })
            .collect();
        let right: Vec<VertexKey> = (0..right_rows)
            .map(|y| {
                let fy = y as f32 * height / (right_rows - 1) as f32;
                if y == 0 {
                    bottom[cols - 1]
                } else if y == right_rows - 1 {
                    top[cols - 1]
                } else {
                    mesh.create_vertex(Vec3::new(width, fy, 0.0))
                }
            })
            .collect();
        let edge = |mesh: &mut LoopMesh, a: VertexKey, b: VertexKey| {
            mesh.create_edge([a, b], EdgeCreateFlags::NO_DUPLICATE)
        };
        let mut side_edges = Vec::new();
        for x in 0..cols - 1 {
            side_edges.push(edge(mesh, bottom[x], bottom[x + 1]));
            side_edges.push(edge(mesh, top[x], top[x + 1]));
        }
        for y in 0..left_rows - 1 {
            edge(mesh, left[y], left[y + 1]);
        }
        for y in 0..right_rows - 1 {
            edge(mesh, right[y], right[y + 1]);
        }
        side_edges
    }

    #[test]
    fn interior_scaffold_sits_at_coincident_endpoint() {
        // §3.4a worked check (surviving interior, 3-vertex sides, 2-vs-3 rails).
        // The left rail's lone scaffold `grid[0][1]` is a parameter-0 split placed
        // *coincident with* the origin corner `(0,0,0)`, not at the rail midpoint
        // `(0,1,0)`. The surviving interior `grid[1][1]` therefore blends its four
        // boundary neighbours over the coincident-endpoint profile:
        //   simple  (all §6.1 weights ¼): ¼·((1,0,0)+(0,0,0)+(1,2,0)+(2,1,0))
        //                                  = (1.0, 0.75, 0)
        //   default (§5.3 transform):      (1.0, 0.5, 0)
        // A midpoint (even-subdivision) scaffold at `(0,1,0)` would instead pull
        // the interior to `y = 1.0` in either mode — the diagnosed §3.4a divergence.
        // Topology is identical either way (3 quads + 1 triangle, one surviving
        // interior vertex); only this position distinguishes the two.
        let interior_pos = |use_interp_simple: bool| -> (Vec3, usize, usize) {
            let mut mesh = LoopMesh::new();
            let side_edges = unequal_railed_interior_wire_ring(&mut mesh, 3, 2, 3);
            let v0 = mesh.vertex_count();
            let result = mesh.grid_fill(
                &side_edges,
                &GridFillParams {
                    use_interp_simple,
                    ..Default::default()
                },
            );
            assert_eq!(mesh.vertex_count() - v0, 1, "one surviving interior vertex");
            mesh.validate_topology();
            let (quads, tris) = quad_tri_split(&mesh, &result);
            // The newest vertex is the surviving interior one.
            let mut last = None;
            let mut it = mesh.mesh_verts();
            while let Some(v) = it.next(&mesh) {
                last = Some(v);
            }
            (mesh.vertex_position(last.unwrap()), quads, tris)
        };

        let (simple, q_s, t_s) = interior_pos(true);
        assert_eq!((q_s, t_s), (3, 1), "3 quads + 1 triangle (simple)");
        assert!(
            (simple - Vec3::new(1.0, 0.75, 0.0)).length() < 1.0e-4,
            "simple-mode interior at the coincident-endpoint blend (1, 0.75, 0), got {simple:?}",
        );
        // Must NOT be the midpoint-subdivision result (y = 1.0).
        assert!(
            (simple.y - 1.0).abs() > 0.2,
            "simple-mode interior must not sit at the midpoint-scaffold y = 1.0, got {simple:?}",
        );

        let (default, q_d, t_d) = interior_pos(false);
        assert_eq!((q_d, t_d), (3, 1), "3 quads + 1 triangle (default)");
        assert!(
            (default - Vec3::new(1.0, 0.5, 0.0)).length() < 1.0e-4,
            "default-mode interior at the coincident-endpoint blend (1, 0.5, 0), got {default:?}",
        );
        assert!(
            (default.y - 1.0).abs() > 0.2,
            "default-mode interior must not sit at the midpoint-scaffold y = 1.0, got {default:?}",
        );
    }

    /// Partition a fill result into (quad count, triangle count).
    fn quad_tri_split(mesh: &LoopMesh, result: &super::GridFillResult) -> (usize, usize) {
        let mut quads = 0;
        let mut tris = 0;
        for &face in result.faces() {
            match mesh.face_len(face) {
                4 => quads += 1,
                3 => tris += 1,
                n => panic!("unexpected face of {n} verts in span-equalised fill"),
            }
        }
        (quads, tris)
    }

    #[test]
    fn unequal_sided_ring_fills_to_quad_tri_fan() {
        // §3.4 worked check: a bottom side of 4 opposing a top side of 3 (single
        // edge rails) → 2 quads + 1 triangle; the lone scaffold that expanded the
        // top to 4 collapses back into the top-middle vertex, so the vertex set is
        // unchanged and the fan apex is shared by all three cells.
        let mut mesh = LoopMesh::new();
        let side_edges = unequal_sided_wire_ring(&mut mesh, 4, 3);
        let v0 = mesh.vertex_count();
        let e0 = mesh.edge_count();
        let f0 = mesh.face_count();
        let perimeter: std::collections::HashSet<VertexKey> = {
            let mut set = std::collections::HashSet::new();
            let mut it = mesh.mesh_verts();
            while let Some(v) = it.next(&mesh) {
                set.insert(v);
            }
            set
        };

        let result = mesh.grid_fill(&side_edges, &GridFillParams::default());

        assert_eq!(result.faces().len(), 3, "fan of 3 faces");
        assert_eq!(quad_tri_split(&mesh, &result), (2, 1), "2 quads + 1 triangle");
        assert_eq!(mesh.vertex_count(), v0, "no scaffold vertex survives (V unchanged)");
        assert_eq!(mesh.face_count() - f0, 3, "F 0 -> 3");
        assert_eq!(mesh.edge_count() - e0, 2, "two interior fan edges created (E 7 -> 9)");

        // The perimeter vertex set is exactly the input's — every scaffold collapsed.
        let mut after = std::collections::HashSet::new();
        let mut it = mesh.mesh_verts();
        while let Some(v) = it.next(&mesh) {
            after.insert(v);
        }
        assert_eq!(after, perimeter, "perimeter vertex set unchanged");
        mesh.validate_topology();
    }

    #[test]
    fn unequal_railed_ring_fills_to_quad_tri_fan() {
        // §3.4 worked check: single-edge opposing sides with a left rail of 2
        // opposing a right rail of 3 → 1 quad + 1 triangle. The right rail's
        // genuine middle vertex is an input perimeter vertex and is retained; the
        // left-rail scaffold collapses into the lattice origin corner, so the
        // vertex set is unchanged.
        let mut mesh = LoopMesh::new();
        let side_edges = unequal_railed_wire_ring(&mut mesh, 2, 3);
        let v0 = mesh.vertex_count();
        let f0 = mesh.face_count();

        let result = mesh.grid_fill(&side_edges, &GridFillParams::default());

        assert_eq!(result.faces().len(), 2, "fan of 2 faces");
        assert_eq!(quad_tri_split(&mesh, &result), (1, 1), "1 quad + 1 triangle");
        assert_eq!(mesh.vertex_count(), v0, "no scaffold vertex survives (V unchanged)");
        assert_eq!(mesh.face_count() - f0, 2, "F 0 -> 2");

        // §3.4 determinate collapse: the scaffold survives into the origin-ward
        // (bottom, low-y) corner, so the **bottom** cell degenerates to the
        // triangle — never the top. The fixture's bottom side is at y = 0, the
        // top side at y = 2; the triangle must therefore carry no top-side
        // vertex (max y < 2), and the quad must carry both top-side vertices.
        let max_y = |face: FaceKey| -> f32 {
            mesh.face_vert_keys(face)
                .iter()
                .map(|&v| mesh.vertex_position(v).y)
                .fold(f32::MIN, f32::max)
        };
        let triangle = result
            .faces()
            .iter()
            .copied()
            .find(|&f| mesh.face_len(f) == 3)
            .unwrap();
        let quad = result
            .faces()
            .iter()
            .copied()
            .find(|&f| mesh.face_len(f) == 4)
            .unwrap();
        assert!(
            max_y(triangle) < 1.5,
            "triangle degenerates the bottom (origin) cell, not the top",
        );
        assert!(
            max_y(quad) > 1.5,
            "the quad spans up to the top side",
        );

        mesh.validate_topology();
    }

    #[test]
    fn equal_sides_unaffected_by_equalisation_path() {
        // §3.4 is a no-op for equal members: the regular 3x3 grid still fills to 4
        // quads with one new interior vertex (guards against the expand-collapse
        // path regressing the equal-sided case).
        let mut mesh = LoopMesh::new();
        let side_edges = rect_wire_ring(&mut mesh, 3, 3, 0.0);
        let v0 = mesh.vertex_count();
        let result = mesh.grid_fill(&side_edges, &GridFillParams::default());
        assert_eq!(quad_tri_split(&mesh, &result), (4, 0), "4 quads, no triangles");
        assert_eq!(mesh.vertex_count() - v0, 1, "one interior vertex created");
        mesh.validate_topology();
    }

    #[test]
    fn faces_out_is_exactly_the_new_quads() {
        let mut mesh = LoopMesh::new();
        let side_edges = rect_wire_ring(&mut mesh, 4, 4, 0.0);
        let result = mesh.grid_fill(&side_edges, &GridFillParams::default());
        // faces.out is exactly the grid; the ring had no faces before.
        assert_eq!(result.faces().len(), 9);
        assert_eq!(mesh.face_count(), 9, "no face exists beyond the reported grid");
        let unique: std::collections::HashSet<FaceKey> = result.faces().iter().copied().collect();
        assert_eq!(unique.len(), 9, "faces.out lists each new quad once");
        mesh.validate_topology();
    }

    #[test]
    fn degenerate_source_frame_stays_finite() {
        // A collinear source triple gives a zero barycentric denominator; the
        // guard must yield finite weights (and a finite transformed position)
        // rather than dividing by ~zero into inf/NaN.
        let a = glam::Vec2::new(0.0, 0.0);
        let b = glam::Vec2::new(1.0, 0.0);
        let c = glam::Vec2::new(2.0, 0.0); // collinear with a, b ⇒ den == 0
        let w = super::grid_barycentric_2d(glam::Vec2::new(0.5, 0.3), a, b, c);
        assert!(w.iter().all(|v| v.is_finite()), "weights finite, got {w:?}");

        // End-to-end through the frame transform with a collinear source
        // triangle: every component of the carried point must be finite.
        let src = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
        ];
        let tar = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
            Vec3::new(2.0, 0.0, 1.0),
        ];
        let out = super::grid_frame_transform(Vec3::new(0.5, 0.5, 0.5), src, tar);
        assert!(
            out.x.is_finite() && out.y.is_finite() && out.z.is_finite(),
            "transformed position finite on a degenerate source frame, got {out:?}",
        );
    }
}
