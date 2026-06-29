use std::collections::{HashMap, HashSet, VecDeque};

use crate::loopmesh::*;

/// Parameters for [`LoopMesh::holes_fill`], mirroring the operator's BMOP slot
/// knobs.
///
/// Algorithm in
/// [`HOLES_FILL.md`](../../../../radial-mesh-guides/operators/HOLES_FILL.md) §1.
#[derive(Copy, Clone, Debug, Default)]
pub struct HolesFillParams {
    /// The **maximum** perimeter vertex count a cap may have to be kept (§5).
    /// After every hole is filled, any cap with **more than** `sides` corners is
    /// deleted. `0` is the sentinel for *no limit* — every cap survives
    /// regardless of size.
    pub sides: usize,
}

/// The `faces.out` of [`LoopMesh::holes_fill`]: every **new** cap the operator
/// created and kept — one n-gon per filled hole that survived the `sides` filter
/// (§1/§5). A hole left unfilled (too many sides, or no closable cycle)
/// contributes nothing.
#[derive(Clone, Debug, Default)]
pub struct HolesFillResult {
    faces: Vec<FaceKey>,
}

impl HolesFillResult {
    pub fn faces(&self) -> &[FaceKey] {
        &self.faces
    }
    pub fn is_empty(&self) -> bool {
        self.faces.is_empty()
    }
}

/// Which of the two opposing breadth-first fronts has claimed a vertex during the
/// §4.2 two-front walk.
#[derive(Copy, Clone, PartialEq, Eq)]
enum Front {
    Near,
    Far,
}

impl LoopMesh {
    /// Discover the minimal closed cycles bounded by an undirected set of boundary
    /// `edges` and cap each with a **single n-gon** (§1). The operator creates no
    /// vertex and (for a simple hole) no edge — every cap side is an existing
    /// edge of the tagged net — and performs no triangulation (§1.1).
    ///
    /// Algorithm in
    /// [`HOLES_FILL.md`](../../../../radial-mesh-guides/operators/HOLES_FILL.md):
    /// the §3 boundary-net tagging, the §4 two-front minimal-cycle fill (§4.1
    /// fillable / one-shot-seed edges, §4.2 two-front breadth-first walk, §4.3
    /// spans-distinct-faces gate, §4.4 overlap rejection, §4.5 termination, §4.6
    /// cap build), the §5 `sides` post-fill deletion of oversized caps, and the §6
    /// winding coherence (§6.2) with the §6.3 free-wire geometric-normal fallback.
    ///
    /// `sides` is **not** a search constraint: the fill runs unconstrained, then
    /// any cap with more than `sides` corners is deleted (`sides == 0` keeps all,
    /// §5). The surviving caps are returned in `faces.out`.
    pub fn holes_fill(&mut self, edges: &[EdgeKey], params: &HolesFillParams) -> HolesFillResult {
        let mut result = HolesFillResult::default();

        // §3 — tag the boundary net: only these edges are ever stepped along or
        // seeded from. A degenerate self-edge can bound no hole and is dropped.
        let mut tagged: HashSet<EdgeKey> = HashSet::with_capacity(edges.len());
        for &edge in edges {
            let [a, b] = self.edge_vert_keys(edge);
            if a != b {
                tagged.insert(edge);
            }
        }
        // §7 — fewer than three fillable edges can close no cycle.
        if tagged.len() < 3 {
            return result;
        }

        // §4.5 — each tagged edge is offered as a seed at most once. Iterate in a
        // stable, deduplicated input order so the outer loop is deterministic.
        let mut seed_order: Vec<EdgeKey> = Vec::with_capacity(tagged.len());
        let mut seeded: HashSet<EdgeKey> = HashSet::with_capacity(tagged.len());
        for &edge in edges {
            if tagged.contains(&edge) && seeded.insert(edge) {
                seed_order.push(edge);
            }
        }

        let mut new_caps: Vec<FaceKey> = Vec::new();
        for &seed in &seed_order {
            // §4.1 — a seed must currently be fillable: tagged and not yet
            // manifold. A surface-hole rim edge that an earlier cap already turned
            // manifold drops out here.
            if !self.holes_fill_fillable(seed, &tagged) {
                continue;
            }
            if let Some(cycle) = self.holes_fill_min_cycle(seed, &tagged) {
                // §4.6 + §6.2 — materialise the cap, wound coherently with any
                // surrounding surface.
                let cap = self.holes_fill_build_cap(&cycle);
                new_caps.push(cap);
            }
        }

        // §6.3 / NORMALS_CALC — refresh every cap's face normal. A surface cap was
        // already wound coherently (§6.2); a free-wire cap has no surface to
        // inherit from, so this geometric recompute is its only normal source.
        for &cap in &new_caps {
            self.face_normal_update(cap);
        }

        // §5 — the `sides` perimeter cap is a post-fill deletion, not a search
        // constraint: every hole was filled above, and only now is each oversized
        // cap killed. `sides == 0` skips the deletion entirely.
        if params.sides != 0 {
            new_caps.retain(|&cap| {
                if self.face_len(cap) > params.sides {
                    self.face_kill(cap);
                    false
                } else {
                    true
                }
            });
        }

        result.faces = new_caps;
        result
    }

    /// §4.1 — an edge is fillable (step-ok / seed-ok) when it carries the working
    /// tag *and* borders zero or one face (wire or boundary). An edge already
    /// bordering two faces is full; capping it again would make it non-manifold.
    fn holes_fill_fillable(&self, edge: EdgeKey, tagged: &HashSet<EdgeKey>) -> bool {
        tagged.contains(&edge) && self.edge_loop_keys(edge).len() <= 1
    }

    /// §4.2 — the two-front breadth-first walk. Plant the seed edge's two
    /// endpoints as the roots of opposing fronts and grow both outward along
    /// fillable tagged edges until they touch across a connecting edge, closing
    /// the smallest cycle reachable from the seed. Returns the cycle's ordered
    /// vertex ring, or `None` if no acceptable cycle closes.
    fn holes_fill_min_cycle(
        &self,
        seed: EdgeKey,
        tagged: &HashSet<EdgeKey>,
    ) -> Option<Vec<VertexKey>> {
        let [root_near, root_far] = self.edge_vert_keys(seed);

        let mut claimed: HashMap<VertexKey, Front> = HashMap::new();
        let mut pred: HashMap<VertexKey, (VertexKey, EdgeKey)> = HashMap::new();
        let mut queue: VecDeque<VertexKey> = VecDeque::new();

        claimed.insert(root_near, Front::Near);
        claimed.insert(root_far, Front::Far);
        // Both roots seeded at depth 0: a strict FIFO then expands in lockstep
        // breadth-first rings, so the first connecting edge found closes a
        // minimal-length cycle (§4.2).
        queue.push_back(root_near);
        queue.push_back(root_far);

        while let Some(vertex) = queue.pop_front() {
            let front = claimed[&vertex];
            // Collect incident edges up front so the BFS bookkeeping below does not
            // alias the disk walk.
            let incident: Vec<EdgeKey> = self.vertex_disk_iter(vertex).collect();
            for edge in incident {
                // Never step along the seed itself — it is the implied closing
                // edge that joins the two roots, not an outward step.
                if edge == seed {
                    continue;
                }
                if !self.holes_fill_fillable(edge, tagged) {
                    continue;
                }
                let [ea, eb] = self.edge_vert_keys(edge);
                let other = if ea == vertex { eb } else { ea };
                match claimed.get(&other) {
                    None => {
                        claimed.insert(other, front);
                        pred.insert(other, (vertex, edge));
                        queue.push_back(other);
                    }
                    Some(&other_front) if other_front == front => {
                        // Same front already owns it — not a closure.
                    }
                    Some(_) => {
                        // The opposite front owns `other`: the two fronts touch
                        // across the connecting edge (vertex, other), closing a
                        // candidate cycle.
                        let cycle =
                            self.holes_fill_assemble_cycle(vertex, other, front, &pred);
                        if self.holes_fill_cycle_accepted(&cycle) {
                            return Some(cycle);
                        }
                        // §4.4 — refused; keep searching for another connecting
                        // edge without claiming `other`.
                    }
                }
            }
        }
        None
    }

    /// Retrace the two half-paths from a connecting edge back to their roots and
    /// splice them into one closed vertex ring (§4.2): the near half-path followed
    /// by the reversed far half-path, whose wrap-around closing edge is the seed.
    fn holes_fill_assemble_cycle(
        &self,
        meeting_a: VertexKey,
        meeting_b: VertexKey,
        front_a: Front,
        pred: &HashMap<VertexKey, (VertexKey, EdgeKey)>,
    ) -> Vec<VertexKey> {
        let (near_end, far_end) = if front_a == Front::Near {
            (meeting_a, meeting_b)
        } else {
            (meeting_b, meeting_a)
        };
        let near_path = Self::holes_fill_retrace(near_end, pred);
        let mut far_path = Self::holes_fill_retrace(far_end, pred);
        far_path.reverse();
        let mut cycle = near_path;
        cycle.extend(far_path);
        cycle
    }

    /// Follow predecessor links from `vertex` back to its (link-less) root,
    /// returning the path root → … → vertex.
    fn holes_fill_retrace(
        vertex: VertexKey,
        pred: &HashMap<VertexKey, (VertexKey, EdgeKey)>,
    ) -> Vec<VertexKey> {
        let mut path = vec![vertex];
        let mut current = vertex;
        while let Some(&(parent, _edge)) = pred.get(&current) {
            path.push(parent);
            current = parent;
        }
        path.reverse();
        path
    }

    /// §4.3 + §4.4 — accept a candidate cycle only when it genuinely bounds a
    /// hole. It must span distinct surrounding regions — its edges border two or
    /// more existing faces, or include a wire edge (open space) — rather than
    /// lying within a single existing face (§4.3); and it must not duplicate
    /// geometry — no existing face may have all of its vertices among the cycle's
    /// (§4.4).
    fn holes_fill_cycle_accepted(&self, cycle: &[VertexKey]) -> bool {
        let n = cycle.len();
        if n < 3 {
            return false;
        }

        // §4.3 — the cycle's edges are the near half-path, the connecting edge,
        // the far half-path, and the seed closing edge; tally the existing faces
        // they border.
        let mut bordering: HashSet<FaceKey> = HashSet::new();
        let mut touches_open_space = false;
        for i in 0..n {
            let from = cycle[i];
            let to = cycle[(i + 1) % n];
            let Some(edge) = self.query_edge_exists(from, to) else {
                // A side that is not an existing edge means the walk did not trace
                // a closed net cycle — reject defensively.
                return false;
            };
            let faces = self.edge_face_keys(edge);
            if faces.is_empty() {
                touches_open_space = true;
            }
            for face in faces {
                bordering.insert(face);
            }
        }
        if !touches_open_space && bordering.len() < 2 {
            return false;
        }

        // §4.4 — overlap rejection: refuse a cap that would stack over a region an
        // existing face already covers.
        let cycle_set: HashSet<VertexKey> = cycle.iter().copied().collect();
        for &vertex in cycle {
            for face in self.vert_face_keys(vertex) {
                if self
                    .face_vert_keys(face)
                    .iter()
                    .all(|fv| cycle_set.contains(fv))
                {
                    return false;
                }
            }
        }
        true
    }

    /// §4.6 + §6.2 — build the single n-gon cap over the cycle, reversing the walk
    /// order when the §6.2 winding vote says the cap would otherwise be wound
    /// incoherently with a surrounding surface.
    fn holes_fill_build_cap(&mut self, cycle: &[VertexKey]) -> FaceKey {
        if self.holes_fill_winding_flip(cycle) {
            let mut reversed = cycle.to_vec();
            reversed.reverse();
            self.face_create_verts(&reversed, FaceCreateFlags::NO_DUPLICATE)
        } else {
            self.face_create_verts(cycle, FaceCreateFlags::NO_DUPLICATE)
        }
    }

    /// §6.2 winding vote: returns true when the cap, built in walk order, would be
    /// wound incoherently with an existing bordering surface and should be
    /// reversed. Two coherent faces sharing a manifold edge traverse it in
    /// opposite senses; only cycle edges that already carry exactly one face vote,
    /// so a free wire loop tallies zero and keeps walk order (§6.3). This is the
    /// same coherence rule as `edgeloop_fill` §5 / `triangle_fill` §4.
    fn holes_fill_winding_flip(&self, cycle: &[VertexKey]) -> bool {
        let n = cycle.len();
        let mut tally: i32 = 0;
        for i in 0..n {
            let from = cycle[i];
            let to = cycle[(i + 1) % n];
            let Some(edge) = self.query_edge_exists(from, to) else {
                continue;
            };
            let radial = self.edge_loop_keys(edge);
            if radial.len() != 1 {
                continue;
            }
            // The neighbour traverses the shared edge starting at this loop's
            // vertex; if it traverses from→to the same way our walk does, our
            // walk is incoherent → vote to flip.
            if self.loop_vertex(radial[0]) == from {
                tally += 1;
            } else {
                tally -= 1;
            }
        }
        tally > 0
    }
}

#[cfg(test)]
mod tests {
    use std::f32::consts::TAU;

    use glam::Vec3;

    use super::HolesFillParams;
    use crate::loopmesh::*;

    /// Build an `n`×`n` grid of unit quad cells in the z=0 plane with the centre
    /// cell omitted, leaving an open hole bordered by four ring faces. Returns the
    /// mesh and the four hole-rim edges. `n` must be odd so a single centre cell
    /// exists; this helper uses a 3×3 grid (16 verts, 8 faces).
    fn grid_3x3_center_hole() -> (LoopMesh, [EdgeKey; 4]) {
        let mut mesh = LoopMesh::new();
        // 4×4 vertex lattice in row-major order (index = row*4 + col).
        let verts: Vec<VertexKey> = (0..4)
            .flat_map(|row| (0..4).map(move |col| (row, col)))
            .map(|(row, col)| mesh.create_vertex(Vec3::new(col as f32, row as f32, 0.0)))
            .collect();
        let at = |row: usize, col: usize| verts[row * 4 + col];
        // Build all nine cells except the centre one (row 1, col 1).
        for row in 0..3 {
            for col in 0..3 {
                if row == 1 && col == 1 {
                    continue;
                }
                let ring = [
                    at(row, col),
                    at(row, col + 1),
                    at(row + 1, col + 1),
                    at(row + 1, col),
                ];
                mesh.face_create_verts(&ring, FaceCreateFlags::NO_DUPLICATE);
            }
        }
        // The centre cell's four corner vertices and rim edges.
        let c = [at(1, 1), at(1, 2), at(2, 2), at(2, 1)];
        let rim = [
            mesh.query_edge_exists(c[0], c[1]).expect("top rim"),
            mesh.query_edge_exists(c[1], c[2]).expect("right rim"),
            mesh.query_edge_exists(c[2], c[3]).expect("bottom rim"),
            mesh.query_edge_exists(c[3], c[0]).expect("left rim"),
        ];
        (mesh, rim)
    }

    /// Build a free wire ring of `n` verts on a unit circle in the z=0 plane,
    /// returning the ring edges in arbitrary (set) order.
    fn convex_wire_ring(mesh: &mut LoopMesh, n: usize, radius: f32) -> Vec<EdgeKey> {
        let verts: Vec<VertexKey> = (0..n)
            .map(|i| {
                let theta = (i as f32) * TAU / (n as f32);
                mesh.create_vertex(Vec3::new(radius * theta.cos(), radius * theta.sin(), 0.0))
            })
            .collect();
        (0..n)
            .map(|i| mesh.create_edge([verts[i], verts[(i + 1) % n]], EdgeCreateFlags::NO_DUPLICATE))
            .collect()
    }

    #[test]
    fn surface_hole_caps_with_one_coherent_quad() {
        let (mut mesh, rim) = grid_3x3_center_hole();
        assert_eq!(mesh.face_count(), 8, "centre cell omitted");

        let result = mesh.holes_fill(&rim, &HolesFillParams { sides: 1000 });

        assert_eq!(result.faces().len(), 1, "one cap per hole");
        assert_eq!(mesh.vertex_count(), 16, "no vertex created");
        assert_eq!(mesh.edge_count(), 24, "no edge created (rim reused)");
        assert_eq!(mesh.face_count(), 9, "hole capped by exactly one quad");

        let cap = result.faces()[0];
        assert_eq!(mesh.face_len(cap), 4, "the cap is a quad over the four rim verts");

        // §6.2 — the cap must be wound coherently with the surrounding ring: it
        // traverses each shared rim edge opposite to the neighbour ring face.
        for shared in mesh.face_edge_keys(cap) {
            let radial = mesh.edge_loop_keys(shared);
            assert_eq!(radial.len(), 2, "each rim edge now borders cap + ring face");
            let loop_a = radial[0];
            let loop_b = radial[1];
            assert_ne!(
                mesh.loop_vertex(loop_a),
                mesh.loop_vertex(loop_b),
                "coherent neighbours traverse the shared edge in opposite senses"
            );
        }
        mesh.validate_topology();
    }

    #[test]
    fn sides_unbounded_keeps_oversized_quad_cap() {
        let (mut mesh, rim) = grid_3x3_center_hole();
        // sides large enough to keep a 4-perimeter cap (and the formal no-limit
        // sentinel 0 behaves the same for this fixture).
        let result = mesh.holes_fill(&rim, &HolesFillParams { sides: 1000 });
        assert_eq!(result.faces().len(), 1, "the quad cap survives the sides filter");
        assert_eq!(mesh.face_count(), 9, "hole stays capped (8 ring + 1 cap)");

        let (mut mesh0, rim0) = grid_3x3_center_hole();
        let result0 = mesh0.holes_fill(&rim0, &HolesFillParams { sides: 0 });
        assert_eq!(result0.faces().len(), 1, "sides==0 is no-limit: cap kept");
        assert_eq!(mesh0.face_count(), 9);
        mesh.validate_topology();
        mesh0.validate_topology();
    }

    #[test]
    fn sides_below_perimeter_deletes_cap_and_reopens_hole() {
        let (mut mesh, rim) = grid_3x3_center_hole();
        // The hole is still filled by the unconstrained search, then its quad cap
        // (perimeter 4 > 3) is deleted by the post-fill pass (§5).
        let result = mesh.holes_fill(&rim, &HolesFillParams { sides: 3 });
        assert!(result.is_empty(), "the oversized cap is deleted → empty faces.out");
        assert_eq!(mesh.face_count(), 8, "hole reopened: only the 8 ring faces remain");
        assert_eq!(mesh.vertex_count(), 16, "no vertex left behind");
        assert_eq!(mesh.edge_count(), 24, "rim edges remain (cap kill leaves them)");
        mesh.validate_topology();
    }

    #[test]
    fn free_wire_loop_caps_to_one_ngon() {
        // A free convex hexagon ring with no surrounding surface: capped by one
        // 6-gon through the §6.3 geometric-normal fallback (orientation
        // non-determinate, so only the topology is asserted).
        let mut mesh = LoopMesh::new();
        let ring = convex_wire_ring(&mut mesh, 6, 1.0);

        let result = mesh.holes_fill(&ring, &HolesFillParams { sides: 1000 });

        assert_eq!(result.faces().len(), 1, "free wire loop caps to one face");
        assert_eq!(mesh.vertex_count(), 6, "no vertex created");
        assert_eq!(mesh.edge_count(), 6, "no edge created (loop reused)");
        assert_eq!(mesh.face_count(), 1, "exactly one cap");
        let cap = result.faces()[0];
        assert_eq!(mesh.face_len(cap), 6, "the cap spans all six corners");
        assert!(
            mesh.face_normal(cap).length_squared() > 1e-6,
            "§6.3 fallback gives the isolated cap a well-defined normal"
        );
        mesh.validate_topology();
    }

    #[test]
    fn under_edged_input_creates_nothing() {
        // Two wire edges cannot close a cycle (§7).
        let mut mesh = LoopMesh::new();
        let a = mesh.create_vertex(Vec3::new(0.0, 0.0, 0.0));
        let b = mesh.create_vertex(Vec3::new(1.0, 0.0, 0.0));
        let c = mesh.create_vertex(Vec3::new(0.5, 1.0, 0.0));
        let e0 = mesh.create_edge([a, b], EdgeCreateFlags::NO_DUPLICATE);
        let e1 = mesh.create_edge([b, c], EdgeCreateFlags::NO_DUPLICATE);

        let result = mesh.holes_fill(&[e0, e1], &HolesFillParams { sides: 0 });

        assert!(result.is_empty(), "sub-three-edge input fills nothing");
        assert_eq!(mesh.face_count(), 0);
        mesh.validate_topology();
    }
}
