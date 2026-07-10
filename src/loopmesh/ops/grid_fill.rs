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
        let loop_a = loops[0].clone();
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
        let (rail_left, rail_right) = match (
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

        // §3.4 — the lattice must be rectangular, so the two opposing sides are
        // reconciled to a common `cols` and the two rails to a common `rows`.
        // Equal members need no work; a shorter member is *expanded* with
        // temporary scaffold vertices (not rejected), and the scaffold collapses
        // back out after the build — degenerating the corner-losing cells into
        // triangles. When neither pair mismatches the expansion is the identity
        // and no collapse occurs, so the member expansions are inert and skipped;
        // they exist only to make the §3.4 collapse reproducible. The lattice
        // origin `grid[0][0]` is already pinned by the §3.4a discovery labeling
        // (loop A's seed-oriented far terminus, `a_first`) — no separate anchor
        // pass is needed here.
        let cols_mismatch = loop_a.len() != loop_b.len();
        let rows_mismatch = rail_left.len() != rail_right.len();

        let cols = loop_a.len().max(loop_b.len());
        let rows = rail_left.len().max(rail_right.len());
        if cols < 2 || rows < 2 {
            return empty;
        }

        // §3.3 — overlapping rails cannot bound a clean rectangle. The membership
        // test is orientation-independent, so the rails' §3.3 pairing orientation
        // does not affect it.
        if grid_rails_overlap(&rail_left, &rail_right) {
            return empty;
        }

        if !(cols_mismatch || rows_mismatch) {
            // Equal-side path: every member already carries the full lattice
            // count, so expansion is the identity. Read each member's positions
            // for the §5/§6 interior solver, but hand the perimeter write the
            // member key slices directly — no scaffold, no collapse, and no
            // redundant per-member key clone.
            let bottom_pos: Vec<Vec3> = loop_a.iter().map(|&v| self.vertex_position(v)).collect();
            let top_pos: Vec<Vec3> = loop_b.iter().map(|&v| self.vertex_position(v)).collect();
            let left_pos: Vec<Vec3> = rail_left.iter().map(|&v| self.vertex_position(v)).collect();
            let right_pos: Vec<Vec3> =
                rail_right.iter().map(|&v| self.vertex_position(v)).collect();
            return self.grid_assemble(
                cols,
                rows,
                &bottom_pos,
                &top_pos,
                &left_pos,
                &right_pos,
                &loop_a,
                &loop_b,
                &rail_left,
                &rail_right,
                params,
            );
        }

        // §3.4a — the lattice origin `grid[0][0]` is `a_first` (loop A's
        // seed-oriented far terminus, `loop_a[0]`), fixed upstream by discovery.
        // The §3.3 pairing already ran loop B and the rails the same way around
        // the lattice, so no re-orientation is needed before expansion; the
        // collapse survivor keyed to `grid[0][0]` is therefore reproducible.

        // §3.4 Phase 1+2 — expand each boundary member to the full lattice count.
        // `*_pos` carry the genuine (scaffold-subdivided) boundary positions that
        // feed the §5/§6 interior interpolation; `*_key` carry the per-slot
        // vertex, with every scaffold slot already mapped to the perimeter vertex
        // it collapses into, so no scaffold vertex survives in the result.
        let (bottom_pos, bottom_key) = self.grid_expand_member(&loop_a, cols);
        let (top_pos, top_key) = self.grid_expand_member(&loop_b, cols);
        let (left_pos, left_key) = self.grid_expand_member(&rail_left, rows);
        let (right_pos, right_key) = self.grid_expand_member(&rail_right, rows);
        self.grid_assemble(
            cols,
            rows,
            &bottom_pos,
            &top_pos,
            &left_pos,
            &right_pos,
            &bottom_key,
            &top_key,
            &left_key,
            &right_key,
            params,
        )
    }

    /// §4.1/§4.2/§7 — assemble a populated `cols × rows` lattice into faces,
    /// shared by the equal-side path (identity expansion, borrowed member key
    /// slices) and the span-equalised path (scaffold-expanded owned key vecs).
    /// The perimeter comes from the `*_key` slots; the `*_pos` profiles drive the
    /// §5/§6 interior interpolation. Behaviour is identical for both callers — the
    /// only difference is whether the key slots were cloned/expanded or borrowed.
    #[allow(clippy::too_many_arguments)]
    fn grid_assemble(
        &mut self,
        cols: usize,
        rows: usize,
        bottom_pos: &[Vec3],
        top_pos: &[Vec3],
        left_pos: &[Vec3],
        right_pos: &[Vec3],
        bottom_key: &[VertexKey],
        top_key: &[VertexKey],
        left_key: &[VertexKey],
        right_key: &[VertexKey],
        params: &GridFillParams,
    ) -> GridFillResult {
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
        // (possibly expanded) boundary profiles.
        self.grid_build_interior(
            &mut grid,
            bottom_pos,
            top_pos,
            left_pos,
            right_pos,
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
    /// cycle with no endpoints — the single-closed-loop case of §1.2/§8).
    ///
    /// The returned loops are ordered and oriented by the §3.4a discovery
    /// labeling — the one upstream quantity the fill's orientation is a locked
    /// function of, and which must never depend on hash iteration order. Loops
    /// come back in ascending order of their **lowest-indexed member edge** (loop
    /// A first, owning the global-lowest marked edge), and each chain is oriented
    /// so its slot 0 is `a_first`: the far terminus reached by crossing that
    /// loop's seed edge in its stored `first → second` order and continuing.
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
        // §3.4a(ii) — seed each loop from the lowest-indexed marked edge it
        // contains, scanning the marked set in ascending edge-index order. The
        // global-lowest edge seeds loop A; building A consumes its edges, so the
        // lowest remaining edge seeds loop B. This makes both the loop ordering
        // and (via the seed's stored endpoint order) each loop's orientation a
        // pure function of element indices, never of hash iteration order.
        let mut seed_order = edge_list.clone();
        seed_order.sort_unstable_by_key(|edge| edge.0);

        let mut visited_edges: HashSet<EdgeKey> = HashSet::new();
        let mut loops: Vec<Vec<VertexKey>> = Vec::new();
        for &seed in &seed_order {
            if !visited_edges.insert(seed) {
                continue;
            }
            let [first, second] = self.edge_vert_keys(seed);

            // Walk the two arms away from the seed edge. Every internal vertex of
            // a grid side has exactly one continuing edge (valence ≤ 2, checked
            // above), so each arm ends at a valence-one terminus. An arm that
            // returns to the opposite seed endpoint means the component is a
            // closed loop — no endpoints for rails → reject (§3.1).
            let forward = grid_walk_arm(&adjacency, seed, first, second, &mut visited_edges)?;
            let backward = grid_walk_arm(&adjacency, seed, second, first, &mut visited_edges)?;

            // §3.4a(iii) — orient the chain so slot 0 is `a_first`: the far
            // terminus reached by crossing the seed `first → second` and
            // continuing, i.e. the end of the `forward` arm. The chain therefore
            // runs `forward` reversed then `backward`.
            let mut chain = Vec::with_capacity(forward.len() + backward.len());
            chain.extend(forward.iter().rev().copied());
            chain.extend(backward.iter().copied());
            loops.push(chain);
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

    /// §3.4 Phase 1+2 / §3.4a(b) — reconcile a boundary member of `members` real
    /// vertices to the lattice's full `target` slot count. Every original member
    /// vertex is preserved as a lattice slot; the `target − count` extra slots are
    /// *scaffold* vertices whose placement, and whose collapse survivor, follow
    /// the **doubling-then-spread expand schedule** of §3.4a(b). That schedule is
    /// a pure function of `count` and `target` — never of input ordering or
    /// geometry — and is what lands the parity-correct outer-cell fan (`4→6 →
    /// [m0, m0, m1, m2, m3, m3]`) rather than an inner-cell one.
    ///
    /// Each slot is resolved to the member index it carries: a real slot to its
    /// own vertex, a scaffold slot to the collapse **survivor** it is coincident
    /// with. Because a scaffold is a **parameter-0 split (§3.4a(a))** placed on
    /// top of its survivor, both its position and its key are the survivor's, so
    /// the profile handed to the §5/§6 interior solver repeats an endpoint
    /// coordinate at each scaffold slot (not an even-subdivision midpoint) and the
    /// later Phase-2 collapse into that survivor moves no geometry.
    ///
    /// A running **swap flag** (§3.4a(b)) picks each split's survivor: **set** →
    /// the slot being processed, **clear** → the neighbour slot sharing the split
    /// segment. It starts set, toggles after every scaffold, and toggles once more
    /// at the end of each doubling pass, carrying unbroken into the spread pass.
    /// The two member endpoints are always slot 0 and slot `target−1`, so the four
    /// lattice corners stay real. An already-full member is the identity.
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

        // Each slot resolves to the member index it carries; scaffolds resolve to
        // their collapse survivor. Start as the identity over the real vertices.
        let mut slots: Vec<usize> = (0..count).collect();
        let mut survive_processed = true;

        // Doubling pass — while the member is under half the target, insert one
        // scaffold per current slot, taking `current → 2·current`.
        while 2 * slots.len() < target {
            let base = std::mem::take(&mut slots);
            let mut next = Vec::with_capacity(base.len() * 2);
            for j in 0..base.len() {
                if j + 1 < base.len() {
                    // Forward split of the segment ahead: the processed slot is
                    // the origin-ward endpoint, the neighbour is its follower.
                    let survivor = if survive_processed { base[j] } else { base[j + 1] };
                    next.push(base[j]);
                    next.push(survivor);
                } else {
                    // The final slot has no segment ahead, so it splits the
                    // segment behind it; the neighbour is its predecessor (the
                    // scaffold just emitted for the previous slot).
                    let predecessor = *next.last().unwrap();
                    let survivor = if survive_processed { base[j] } else { predecessor };
                    next.push(survivor);
                    next.push(base[j]);
                }
                survive_processed = !survive_processed;
            }
            // One extra toggle closes the doubling pass, carrying into the next.
            survive_processed = !survive_processed;
            slots = next;
        }

        // Spread pass — scatter the remaining scaffolds one per even-width bucket
        // over the current slots (§3.4a(b) `spread_indices`). Indices refer to the
        // pre-spread sequence; inserting in ascending order keeps a running offset.
        let current = slots.len();
        let indices = spread_indices(current, target - current);
        let mut offset = 0usize;
        for index in indices {
            let position = index + offset;
            let survivor = if index + 1 < current {
                // Forward split: this slot (set) or its follower (clear).
                if survive_processed {
                    slots[position]
                } else {
                    slots[position + 1]
                }
            } else {
                // Final slot: backward split against its predecessor.
                if survive_processed {
                    slots[position]
                } else {
                    slots[position - 1]
                }
            };
            let insert_at = if index + 1 < current { position + 1 } else { position };
            slots.insert(insert_at, survivor);
            survive_processed = !survive_processed;
            offset += 1;
        }

        let positions: Vec<Vec3> = slots.iter().map(|&m| real_positions[m]).collect();
        let keys: Vec<VertexKey> = slots.iter().map(|&m| members[m]).collect();
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

/// §3.1/§3.4a — walk one arm of a side loop away from its seed edge. Starting at
/// `start` (a seed endpoint) having arrived via `seed`, step to the single
/// continuing edge at each vertex until a valence-one terminus, collecting the
/// vertices (including `start`). Returns `None` when the arm reaches
/// `far_endpoint` (the seed's opposite endpoint), which means the component is a
/// closed loop and cannot be a grid side.
fn grid_walk_arm(
    adjacency: &HashMap<VertexKey, SmallVec<[(VertexKey, EdgeKey); 2]>>,
    seed: EdgeKey,
    far_endpoint: VertexKey,
    start: VertexKey,
    visited_edges: &mut HashSet<EdgeKey>,
) -> Option<Vec<VertexKey>> {
    let mut arm = vec![start];
    let mut current = start;
    let mut arrived = seed;
    loop {
        let mut next = None;
        for &(next_vertex, next_edge) in &adjacency[&current] {
            if next_edge == arrived {
                continue;
            }
            next = Some((next_vertex, next_edge));
            break;
        }
        let Some((next_vertex, next_edge)) = next else {
            break;
        };
        if next_vertex == far_endpoint {
            return None;
        }
        visited_edges.insert(next_edge);
        arm.push(next_vertex);
        current = next_vertex;
        arrived = next_edge;
    }
    Some(arm)
}

/// §3.3 — the two rails overlap when they share any vertex (a clean rectangle's
/// rails share only the — distinct — corners, which live one on each rail).
fn grid_rails_overlap(rail_left: &[VertexKey], rail_right: &[VertexKey]) -> bool {
    let left: HashSet<VertexKey> = rail_left.iter().copied().collect();
    rail_right.iter().any(|v| left.contains(v))
}

/// §3.4a(b) — the spread pass's scaffold slots: one mark per equal-width bucket
/// over `0..count`, dropped at the integer slot **just below** the bucket centre
/// `c_k = (2k+1)·count / (2·extra)` (the largest integer strictly less than
/// `c_k`). Equal buckets keep the marks as evenly spaced as an integer grid
/// allows; the low-biased "just below" tie rule fixes the placement when a centre
/// lands exactly on an integer slot. Both operands are non-negative, so this is
/// ordinary truncating integer division. `spread_indices(4, 2) == [0, 2]`,
/// `spread_indices(6, 1) == [2]`.
fn spread_indices(count: usize, extra: usize) -> Vec<usize> {
    (0..extra)
        .map(|k| ((2 * k + 1) * count - 1) / (2 * extra))
        .collect()
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
    // deterministic fallback. `den` is twice `area_src` (the latter is
    // ½·|perp_dot|), so guarding `|den| <= 2.0e-20` fires at the same frame
    // magnitude the sibling `area_src > 1.0e-20` check in `grid_frame_transform`
    // rejects; unreachable on non-degenerate frames.
    if den.abs() <= 2.0e-20 {
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
    fn side_edge_input_order_is_inert_on_asymmetric_frame() {
        // §3.4a(i): the input edge array's order is *inert*. The operator stamps
        // a mark on each input edge as a membership set, discards the incoming
        // order, and rediscovers the two loops by scanning the marked set in
        // ascending edge-index order (§3.4a(ii)) — seeding loop A from the
        // lowest-indexed marked edge and walking it from that seed's stored
        // `first -> second` endpoints (§3.4a(iii)). Permuting `side_edges` must
        // therefore leave the fill *identical*, not merely isomorphic.
        //
        // This is exercised on `unequal_railed_interior_wire_ring(3, 2, 3)`
        // precisely because it *would* discriminate a reintroduced order
        // dependence: loop A (the 3-vertex bottom side) owns **two** marked
        // edges, so seeding from the passed order rather than the lowest index
        // could pick a different seed and mirror the grid; and the frame is
        // rail-asymmetric (short 2-vertex left rail vs 3-vertex right rail), so a
        // mirror moves the surviving interior vertex and the triangle's cell —
        // observably, not just up to isomorphism. (Contrast a single-edge loop A
        // or a symmetric frame, where the property holds vacuously.)
        //
        // Same fixture built in one creation order (⇒ identical element indices),
        // filled three times with `side_edges` forward, reversed, and rotated.
        // All three must agree on the face split, the surviving interior vertex
        // key, and its position exactly.
        let fill = |permute: &dyn Fn(&mut Vec<EdgeKey>)| -> (usize, usize, VertexKey, Vec3) {
            let mut mesh = LoopMesh::new();
            let mut side_edges = unequal_railed_interior_wire_ring(&mut mesh, 3, 2, 3);
            permute(&mut side_edges);
            let v0 = mesh.vertex_count();
            let result = mesh.grid_fill(&side_edges, &GridFillParams::default());
            let (quads, tris) = quad_tri_split(&mesh, &result);
            assert_eq!(mesh.vertex_count() - v0, 1, "one surviving interior vertex");
            // The lone surviving interior vertex is the newest one.
            let mut interior = None;
            let mut it = mesh.mesh_verts();
            while let Some(v) = it.next(&mesh) {
                interior = Some(v);
            }
            let interior = interior.unwrap();
            mesh.validate_topology();
            (quads, tris, interior, mesh.vertex_position(interior))
        };

        let forward = fill(&|_edges| {});
        let reversed = fill(&|edges| edges.reverse());
        let rotated = fill(&|edges| edges.rotate_left(1));

        assert_eq!(
            forward, reversed,
            "reversing side_edges must leave the fill identical (interior key, position, split)",
        );
        assert_eq!(
            forward, rotated,
            "rotating side_edges must leave the fill identical (interior key, position, split)",
        );

        // Pin the orientation concretely (the §3.4a worked check): the fill lands
        // 3 quads + 1 triangle and the surviving interior vertex sits at the
        // coincident-endpoint blend (1, 0.5, 0) — never the y = 1.0 an
        // even-subdivision (or mirror-opposite) placement would give.
        let (quads, tris, _interior, interior_pos) = forward;
        assert_eq!((quads, tris), (3, 1), "2-vs-3 rail fill: 3 quads + 1 triangle");
        assert!(
            (interior_pos - Vec3::new(1.0, 0.5, 0.0)).length() < 1.0e-4,
            "interior vertex at the coincident-endpoint blend (1, 0.5, 0), got {interior_pos:?}",
        );
        assert!(
            (interior_pos.y - 1.0).abs() > 0.2,
            "interior vertex must not sit at the mirror/midpoint y = 1.0, got {interior_pos:?}",
        );
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

    #[test]
    fn spread_indices_matches_guide_traces() {
        // §3.4a(b) worked traces: two even-width buckets over 0..4 pick slots 0
        // and 2; a single bucket over 0..6 picks the slot just below centre 3.0.
        assert_eq!(super::spread_indices(4, 2), vec![0, 2]);
        assert_eq!(super::spread_indices(6, 1), vec![2]);
    }

    /// The §3.4a(b) collapsed slot profile of `grid_expand_member`, expressed as
    /// the member index each slot resolves to (a scaffold resolves to its collapse
    /// survivor). `[0, 0, 1, 2, 3, 3]` is the outer-cell fan of a `4→6` expand.
    fn expand_profile(mesh: &LoopMesh, members: &[VertexKey], target: usize) -> Vec<usize> {
        let (positions, keys) = mesh.grid_expand_member(members, target);
        assert_eq!(positions.len(), target, "one position per lattice slot");
        assert_eq!(keys.len(), target, "one key per lattice slot");
        keys.iter()
            .map(|k| {
                members
                    .iter()
                    .position(|m| m == k)
                    .expect("every slot resolves to a real member vertex")
            })
            .collect()
    }

    #[test]
    fn expand_schedule_matches_worked_traces() {
        // §3.4a(b) — the doubling-then-spread expand schedule, pinned against the
        // guide's two fully-worked traces and the minimal-axis regressions.
        let mut mesh = LoopMesh::new();
        let make = |mesh: &mut LoopMesh, n: usize| -> Vec<VertexKey> {
            (0..n)
                .map(|i| mesh.create_vertex(Vec3::new(i as f32, 0.0, 0.0)))
                .collect()
        };

        // 4→6: doubling does not run (2·4 = 8 is not < 6); the flag is still set,
        // spread [0, 2] collapses segment 0 origin-ward (m0) and segment 2
        // destination-ward (m3) → the outer-cell fan.
        let m4 = make(&mut mesh, 4);
        assert_eq!(expand_profile(&mesh, &m4, 6), vec![0, 0, 1, 2, 3, 3]);
        // Identity when already full.
        assert_eq!(expand_profile(&mesh, &m4, 4), vec![0, 1, 2, 3]);

        // 3→7: one doubling pass (2·3 = 6 < 7) then a single spread; the flag's
        // set → clear → set cadence plus the end-of-pass toggle lands the spread
        // scaffold origin-ward on m1.
        let m3 = make(&mut mesh, 3);
        assert_eq!(expand_profile(&mesh, &m3, 7), vec![0, 0, 1, 1, 2, 2, 2]);

        // Minimal single-scaffold axes are unchanged from the earlier ✅ rows.
        let m2 = make(&mut mesh, 2);
        assert_eq!(expand_profile(&mesh, &m2, 3), vec![0, 0, 1]); // 2→3 origin-ward
        let m3b = make(&mut mesh, 3);
        assert_eq!(expand_profile(&mesh, &m3b, 4), vec![0, 1, 1, 2]); // 3→4 middle segment

        // Single-segment multi-scaffold chains every scaffold onto the first
        // corner — the toggle is a no-op once the segment degenerates.
        let m2b = make(&mut mesh, 2);
        assert_eq!(expand_profile(&mesh, &m2b, 5), vec![0, 0, 0, 0, 1]); // 2→5
        assert_eq!(expand_profile(&mesh, &m2b, 4), vec![0, 0, 0, 1]); // 2→4
    }

    #[test]
    fn multisegment_expand_lands_outer_cell_fan() {
        // §3.4a(b) parity target: a 6-vertex side opposing a 4-vertex side (single
        // edge rails) expands the short side 4→6 with the outer-cell fan profile
        // [m0, m0, m1, m2, m3, m3], so the two **outer** lattice cells degenerate
        // to triangles and the three interior cells stay quads. The bottom corners
        // at (0,0,0) and (5,0,0) each belong to a triangle — the observable that
        // distinguishes this from the inner-cell fan an even-per-segment schedule
        // would produce (where those corners land in quads).
        let mut mesh = LoopMesh::new();
        let side_edges = unequal_sided_wire_ring(&mut mesh, 6, 4);
        let result = mesh.grid_fill(&side_edges, &GridFillParams::default());

        assert_eq!(result.faces().len(), 5, "fan of 5 faces");
        assert_eq!(quad_tri_split(&mesh, &result), (3, 2), "3 quads + 2 triangles");

        // The face carrying a given position, and whether it is a triangle.
        let face_at = |x: f32, y: f32| -> FaceKey {
            let target = result.faces().iter().copied().find(|&f| {
                mesh.face_vert_keys(f).iter().any(|&v| {
                    let p = mesh.vertex_position(v);
                    (p.x - x).abs() < 1.0e-3 && (p.y - y).abs() < 1.0e-3
                })
            });
            target.unwrap_or_else(|| panic!("no fill face carries ({x}, {y})"))
        };

        // Both bottom corners degenerate → the outer cells are the triangles.
        assert_eq!(
            mesh.face_len(face_at(0.0, 0.0)),
            3,
            "the origin-corner (outer) cell degenerates to a triangle",
        );
        assert_eq!(
            mesh.face_len(face_at(5.0, 0.0)),
            3,
            "the far-corner (outer) cell degenerates to a triangle",
        );
        mesh.validate_topology();
    }
}
