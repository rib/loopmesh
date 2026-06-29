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
    /// result. Span equalisation (§3.4, mismatched side counts) is not yet
    /// implemented — sides or rails of unequal length are rejected.
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

        let cols = loop_a.len();
        let rows = rail_left.len();

        // The lattice must be rectangular. Span equalisation (§3.4) is out of
        // scope, so unequal opposing sides or unequal rails are rejected here
        // rather than expanded.
        if loop_b.len() != cols || rail_right.len() != rows {
            return empty;
        }
        if cols < 2 || rows < 2 {
            return empty;
        }

        // §3.3 — overlapping rails cannot bound a clean rectangle.
        if grid_rails_overlap(&rail_left, &rail_right) {
            return empty;
        }

        // §4.1 — write the perimeter into the lattice. The four corners are
        // written by more than one side; §3.3 pinned them to the same vertices.
        let mut grid = Lattice::new(cols, rows);
        for x in 0..cols {
            grid.set(x, 0, loop_a[x]);
            grid.set(x, rows - 1, loop_b[x]);
        }
        for y in 0..rows {
            grid.set(0, y, rail_left[y]);
            grid.set(cols - 1, y, rail_right[y]);
        }

        // §5/§6 — create the interior vertices by interpolation.
        self.grid_build_interior(&mut grid, params.use_interp_simple);

        // §4.2 — settle the build winding against any bordering surface.
        let flip = self.grid_winding_flip(&grid);

        // §7 — one quad per cell, stamped and recorded.
        let mut result = GridFillResult::default();
        for x in 0..cols - 1 {
            for y in 0..rows - 1 {
                let corners: [VertexKey; 4] = if flip {
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

    /// §5/§6 — fill every interior lattice cell with a freshly created vertex.
    fn grid_build_interior(&mut self, grid: &mut Lattice, use_interp_simple: bool) {
        let cols = grid.cols;
        let rows = grid.rows;
        if cols < 3 || rows < 3 {
            return;
        }

        let position = |g: &Lattice, x: usize, y: usize, mesh: &LoopMesh| -> Vec3 {
            mesh.vertex_position(g.get(x, y))
        };

        for x in 1..cols - 1 {
            for y in 1..rows - 1 {
                let u = x as f32 / (cols - 1) as f32;
                let v = y as f32 / (rows - 1) as f32;

                let bottom = position(grid, x, 0, self);
                let top = position(grid, x, rows - 1, self);
                let left = position(grid, 0, y, self);
                let right = position(grid, cols - 1, y, self);
                let p00 = position(grid, 0, 0, self);
                let p10 = position(grid, cols - 1, 0, self);
                let p01 = position(grid, 0, rows - 1, self);
                let p11 = position(grid, cols - 1, rows - 1, self);

                let pos = if use_interp_simple {
                    // §6 — Coons bilinear blend of the four boundary sides.
                    grid_coons(u, v, bottom, top, left, right, p00, p10, p01, p11)
                } else {
                    // §5 — carry the bottom and top side profiles inward along
                    // the rails. Each side has a local frame (its run direction
                    // and the across-to-next-row direction); the profile point
                    // is re-expressed in the current row's frame, then the two
                    // mapped positions are blended by the row height. The frame
                    // rotation is what makes a curved boundary yield a curved
                    // interior rather than the flat bilinear sheet of §6.
                    let row_origin = position(grid, 0, y, self);
                    let row_run = position(grid, cols - 1, y, self) - row_origin;
                    let row_across = position(grid, 0, y + 1, self) - row_origin;

                    let bottom_across = position(grid, 0, 1, self) - p00;
                    let carried_bottom = grid_frame_transfer(
                        bottom, p00, p10 - p00, bottom_across, row_origin, row_run, row_across,
                    );

                    let top_across = p01 - position(grid, 0, rows - 2, self);
                    let carried_top = grid_frame_transfer(
                        top, p01, p11 - p01, top_across, row_origin, row_run, row_across,
                    );

                    carried_bottom.lerp(carried_top, v)
                };

                let vertex = self.create_vertex(pos);

                // Interior vertex attributes blend the four boundary verts in
                // the cell's row and column (§5).
                if !self.vertex_attribs.is_empty() {
                    let half = 0.5_f32;
                    let sources = [
                        (half * (1.0 - v), grid.get(x, 0).as_usize()),
                        (half * v, grid.get(x, rows - 1).as_usize()),
                        (half * (1.0 - u), grid.get(0, y).as_usize()),
                        (half * u, grid.get(cols - 1, y).as_usize()),
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

/// §6 — the Coons bilinear blend: the four boundary sides summed, minus the
/// corner bilinear term, so a flat regular frame reproduces the regular lattice.
#[allow(clippy::too_many_arguments)]
fn grid_coons(
    u: f32,
    v: f32,
    bottom: Vec3,
    top: Vec3,
    left: Vec3,
    right: Vec3,
    p00: Vec3,
    p10: Vec3,
    p01: Vec3,
    p11: Vec3,
) -> Vec3 {
    let sides = (1.0 - v) * bottom + v * top + (1.0 - u) * left + u * right;
    let corners = (1.0 - u) * (1.0 - v) * p00
        + u * (1.0 - v) * p10
        + (1.0 - u) * v * p01
        + u * v * p11;
    sides - corners
}

/// §5 — re-express `point` from a source corner frame into a destination corner
/// frame. Each frame is the orthonormalised `(run, across)` pair plus their
/// cross product; the source-relative coordinates are rebuilt in the destination
/// basis, scaled by the run-length ratio. A rotation between the frames carries
/// a curved profile through the row's orientation instead of merely translating
/// it (which is what distinguishes §5 from the §6 bilinear sheet on a warped
/// frame).
fn grid_frame_transfer(
    point: Vec3,
    src_origin: Vec3,
    src_run: Vec3,
    src_across: Vec3,
    dst_origin: Vec3,
    dst_run: Vec3,
    dst_across: Vec3,
) -> Vec3 {
    let (se1, se2, se3, src_len) = grid_ortho_frame(src_run, src_across);
    let (de1, de2, de3, dst_len) = grid_ortho_frame(dst_run, dst_across);
    let relative = point - src_origin;
    if src_len < 1.0e-12 {
        return dst_origin + relative;
    }
    let a = relative.dot(se1);
    let b = relative.dot(se2);
    let c = relative.dot(se3);
    let scale = dst_len / src_len;
    dst_origin + (a * de1 + b * de2 + c * de3) * scale
}

/// Orthonormal frame `(e1, e2, e3)` from a run direction and an across
/// direction, returning the run length so the caller can scale. `e1` follows
/// the run; `e2` is the across component orthogonal to `e1`; `e3` completes the
/// right-handed basis.
fn grid_ortho_frame(run: Vec3, across: Vec3) -> (Vec3, Vec3, Vec3, f32) {
    let run_len = run.length();
    let e1 = if run_len > 1.0e-12 {
        run / run_len
    } else {
        Vec3::X
    };
    let mut e2 = across - across.dot(e1) * e1;
    if e2.length_squared() < 1.0e-20 {
        e2 = e1.any_orthonormal_vector();
    } else {
        e2 = e2.normalize();
    }
    let e3 = e1.cross(e2);
    (e1, e2, e3, run_len)
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
        let z = |x: usize, y: usize| -> f32 {
            // Saddle: positive along one diagonal, negative along the other.
            let u = x as f32 / (cols - 1) as f32 - 0.5;
            let v = y as f32 / (rows - 1) as f32 - 0.5;
            warp * (u * u - v * v)
        };
        let mut vert = vec![vec![VertexKey(u32::MAX); rows]; cols];
        let on_perimeter = |x: usize, y: usize| x == 0 || x == cols - 1 || y == 0 || y == rows - 1;
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
}
