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
    /// fillable edges / failed-seed net removal, §4.2 two-front breadth-first
    /// walk, §4.3 per-front spans-distinct-faces gate, §4.4 overlap rejection,
    /// §4.5 termination, §4.6 cap build), the §5 `sides` post-fill deletion of
    /// oversized caps, and the §6
    /// winding coherence (§6.2) with the §6.3 free-wire geometric-normal fallback.
    ///
    /// `sides` is **not** a search constraint: the fill runs unconstrained, then
    /// any cap with more than `sides` corners is deleted (`sides == 0` keeps all,
    /// §5). The surviving caps are returned in `faces.out`.
    pub fn holes_fill(&mut self, edges: &[EdgeKey], params: &HolesFillParams) -> HolesFillResult {
        let mut result = HolesFillResult::default();

        // §3 + §4 — the shared minimal-cycle core: tag the net, cap every enclosed
        // cell with one n-gon, and refresh each cap's normal (NORMALS_CALC).
        let mut new_caps = self.edgenet_minimal_cycle_fill(edges);

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

        // §6.1 — flood the surrounding surface's attributes into the surviving
        // caps. A free-wire cap the flood never reaches keeps its defaults (§6.3).
        self.holes_fill_inherit_attributes(&new_caps);

        result.faces = new_caps;
        result
    }

    /// §3 + §4 — the shared **edgenet minimal-cycle fill** core, reused verbatim
    /// by [`LoopMesh::holes_fill`] and [`LoopMesh::edgenet_fill`]. Tags the
    /// working net (§3), then repeatedly finds the smallest fillable cycle in the
    /// tagged subset and caps it with a single n-gon (§4.1–§4.6) until no fillable
    /// tagged edge remains, returning every new cap with a refreshed geometric
    /// normal (NORMALS_CALC / §6.3).
    ///
    /// Neither the `sides` post-fill deletion (holes-fill §5) nor the §6 attribute
    /// flood is part of the core — those are the per-operator wrappers that the
    /// two fill operators apply around this identical engine.
    pub(crate) fn edgenet_minimal_cycle_fill(&mut self, edges: &[EdgeKey]) -> Vec<FaceKey> {
        // §3 — tag the working net: only these edges are ever stepped along or
        // seeded from. A degenerate self-edge can bound no cell and is dropped.
        let mut tagged: HashSet<EdgeKey> = HashSet::with_capacity(edges.len());
        for &edge in edges {
            let [a, b] = self.edge_vert_keys(edge);
            if a != b {
                tagged.insert(edge);
            }
        }
        // §7 — fewer than three fillable edges can close no cycle.
        if tagged.len() < 3 {
            return Vec::new();
        }

        // §4.3 / §4.5 — seeds are drawn in mesh/insertion order, not input order:
        // the per-front gate's refusal of an interior-edge net depends on offering
        // a face-bordering edge before a wire diagonal (§4.3 worked example), and
        // `edge_keys` enumerates the slab in ascending insertion order. Only tagged
        // edges can seed, so the candidate list is the tagged subset in that order.
        let mut candidates: Vec<EdgeKey> = self.edge_keys();
        candidates.retain(|edge| tagged.contains(edge));

        let mut new_caps: Vec<FaceKey> = Vec::new();
        // The net shrinks monotonically: a seed that closes no acceptable cycle is
        // *removed* from the tagged net (§4.1), so it can never serve as a path or
        // connecting edge again — that is what makes a per-front refusal final
        // (§4.3) and what terminates the free-wire case (§4.5). A successful cap's
        // rim edges, by contrast, stay tagged and may be re-seeded; re-capping is
        // prevented by the §4.4 overlap rejection, not by a one-shot-seed guard.
        loop {
            // §4.1 — the next still-fillable tagged edge in insertion order. A rim
            // edge an earlier cap turned manifold drops out here; a failed seed has
            // been untagged and is likewise skipped.
            let Some(seed) = candidates
                .iter()
                .copied()
                .find(|&edge| self.holes_fill_fillable(edge, &tagged))
            else {
                break;
            };
            if let Some(cycle) = self.holes_fill_min_cycle(seed, &tagged) {
                // §4.6 + §6.2 — materialise the cap, wound coherently with any
                // surrounding surface. The seed and its rim stay tagged.
                let cap = self.holes_fill_build_cap(&cycle);
                new_caps.push(cap);
            } else {
                // §4.1 — failed seed: remove it from the net so no later seeding
                // can reconstruct a cycle across it.
                tagged.remove(&seed);
            }
        }

        // §6.3 / NORMALS_CALC — refresh every cap's face normal. A surface cap was
        // already wound coherently (§6.2); a free-wire cap has no surface to
        // inherit from, so this geometric recompute is its only normal source.
        for &cap in &new_caps {
            self.face_normal_update(cap);
        }

        new_caps
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

        // §4.3 detail 1 — the spans-distinct-faces flag is seeded from the *seed
        // edge's* bordering face. A wire seed (no bordering face) raises both
        // fronts' flags from the outset; otherwise both fronts start confined to
        // the seed's single bordering face and only a step that leaves that face
        // (or a wire step) raises the flag.
        let seed_face: Option<FaceKey> = self.edge_face_keys(seed).first().copied();
        let root_flag = seed_face.is_none();

        let mut claimed: HashMap<VertexKey, Front> = HashMap::new();
        let mut pred: HashMap<VertexKey, (VertexKey, EdgeKey)> = HashMap::new();
        // §4.3 detail 2 — the flag is per-front and *sticky*: once raised on a
        // path it stays raised. Carried per claimed vertex (the flag of the path
        // root → that vertex).
        let mut spans: HashMap<VertexKey, bool> = HashMap::new();
        let mut queue: VecDeque<VertexKey> = VecDeque::new();

        claimed.insert(root_near, Front::Near);
        claimed.insert(root_far, Front::Far);
        spans.insert(root_near, root_flag);
        spans.insert(root_far, root_flag);
        // Both roots seeded at depth 0, then a strict FIFO expands in lockstep
        // breadth-first rings, so the first connecting edge found closes a
        // minimal-length cycle (§4.2). The far root is enqueued first: when both
        // fronts race for one shared vertex in the same ring, the far front claims
        // it and the near front's step becomes the bare connecting edge — the order
        // the §4.3 worked example relies on for the interior-net refusal.
        queue.push_back(root_far);
        queue.push_back(root_near);

        while let Some(vertex) = queue.pop_front() {
            let front = claimed[&vertex];
            let vertex_spans = spans[&vertex];
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
                        // §4.3 detail 1+2 — propagate the flag stickily: it rises
                        // the moment this step leaves the seed face (a wire step, or
                        // a step onto an edge bordering a different face).
                        let raised = vertex_spans || self.holes_fill_step_raises(edge, seed_face);
                        spans.insert(other, raised);
                        pred.insert(other, (vertex, edge));
                        queue.push_back(other);
                    }
                    Some(&other_front) if other_front == front => {
                        // Same front already owns it — not a closure.
                    }
                    Some(_) => {
                        // The opposite front owns `other`: the two fronts touch
                        // across the connecting edge (vertex, other), closing a
                        // candidate cycle. §4.3 detail 3 — the connecting edge's own
                        // face is *not* consulted; acceptance reads only the two
                        // meeting fronts' accumulated flags.
                        let spans_distinct = vertex_spans || spans[&other];
                        if spans_distinct {
                            let cycle =
                                self.holes_fill_assemble_cycle(vertex, other, front, &pred);
                            if self.holes_fill_cycle_accepted(&cycle) {
                                return Some(cycle);
                            }
                        }
                        // §4.3 / §4.4 — refused (the cycle lies within one face, or
                        // would duplicate geometry); keep searching for another
                        // connecting edge without claiming `other`.
                    }
                }
            }
        }
        None
    }

    /// §4.3 detail 1 — whether stepping along `edge` lifts a front's
    /// spans-distinct-faces flag: a wire step (no bordering face) always does, and
    /// so does a step onto an edge whose single bordering face differs from the
    /// seed's confining face. A fillable edge borders at most one face (§4.1), so
    /// only its first bordering face is consulted.
    fn holes_fill_step_raises(&self, edge: EdgeKey, seed_face: Option<FaceKey>) -> bool {
        match self.edge_face_keys(edge).first().copied() {
            None => true,
            Some(face) => Some(face) != seed_face,
        }
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

    /// §4.4 — overlap rejection (the spans-distinct-faces gate of §4.3 is decided
    /// per-front during the walk, *before* this is called). A candidate cycle that
    /// passed the per-front gate is still refused here if it would duplicate
    /// geometry: no existing face may have all of its vertices among the cycle's,
    /// which would stack a second cap over a region that face already covers.
    /// Also rejects defensively if the assembled ring is not a closed loop of
    /// existing edges.
    fn holes_fill_cycle_accepted(&self, cycle: &[VertexKey]) -> bool {
        let n = cycle.len();
        if n < 3 {
            return false;
        }

        // Defensive: every side of the ring must be an existing tagged-net edge —
        // the walk traces along existing edges, so a missing side means the cycle
        // was not assembled from a real closed net loop.
        for i in 0..n {
            let from = cycle[i];
            let to = cycle[(i + 1) % n];
            if self.query_edge_exists(from, to).is_none() {
                return false;
            }
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

    /// §6.1 — inherit the surrounding surface's attributes into the surviving
    /// caps. Face-domain attributes (material index, smooth flag, face-domain
    /// custom-data) flood outward across shared edges: every cap touching an
    /// existing face inherits directly, then caps reachable only through other
    /// caps inherit transitively from the first cap that touched the surface.
    /// Per-corner loop attributes (UVs / colours) are pulled across each cap
    /// corner's shared edge from the existing face at the same vertex. A cap the
    /// flood never reaches — a free wire loop with no adjacent existing face —
    /// keeps its defaults (§6.3).
    pub(crate) fn holes_fill_inherit_attributes(&mut self, caps: &[FaceKey]) {
        if caps.is_empty() {
            return;
        }
        let cap_set: HashSet<FaceKey> = caps.iter().copied().collect();

        // Face-domain flood. Seed from every cap adjacent to an existing (non-cap)
        // face, then propagate to caps reachable only through already-resolved
        // caps. Reuses the same radial adjacency walk as the §6.2 winding vote.
        let mut resolved: HashSet<FaceKey> = HashSet::with_capacity(caps.len());
        let mut queue: VecDeque<FaceKey> = VecDeque::new();
        for &cap in caps {
            if let Some(donor) = self.holes_fill_surface_neighbor(cap, &cap_set) {
                self.holes_fill_copy_face_attribs(donor, cap);
                resolved.insert(cap);
                queue.push_back(cap);
            }
        }
        while let Some(donor_cap) = queue.pop_front() {
            for shared in self.face_edge_keys(donor_cap) {
                for radial in self.edge_loop_keys(shared) {
                    let neighbor = self.loop_face(radial);
                    if cap_set.contains(&neighbor) && resolved.insert(neighbor) {
                        self.holes_fill_copy_face_attribs(donor_cap, neighbor);
                        queue.push_back(neighbor);
                    }
                }
            }
        }

        // Per-corner loop inheritance across the shared boundary.
        if !self.loop_attribs.is_empty() {
            for &cap in caps {
                self.holes_fill_copy_loop_data(cap, &cap_set);
            }
        }
    }

    /// Find an existing (non-cap) face adjacent to `cap` across any of its rim
    /// edges — the donor the cap inherits face attributes from. Returns `None`
    /// for a free wire cap with no surrounding surface.
    fn holes_fill_surface_neighbor(
        &self,
        cap: FaceKey,
        cap_set: &HashSet<FaceKey>,
    ) -> Option<FaceKey> {
        for shared in self.face_edge_keys(cap) {
            for radial in self.edge_loop_keys(shared) {
                let neighbor = self.loop_face(radial);
                if neighbor != cap && !cap_set.contains(&neighbor) {
                    return Some(neighbor);
                }
            }
        }
        None
    }

    /// Copy `donor`'s face-domain attributes into `cap`: the material index and
    /// any face-domain custom-data (both live in the face attribute table, so a
    /// single copy carries them together) plus the persistent smooth header bit.
    ///
    /// Shared with [`LoopMesh::face_attribute_fill`], which is the same bare
    /// face-domain inheritance pass exposed on its own.
    pub(crate) fn holes_fill_copy_face_attribs(&mut self, donor: FaceKey, cap: FaceKey) {
        self.face_attribs.copy_into(cap.as_usize(), donor.as_usize());
        let smooth = self.is_face_smooth(donor);
        self.set_face_smooth(cap, smooth);
    }

    /// Pull each cap corner's loop data from the existing face across its shared
    /// edge, matched at the same vertex (§6.1). A corner whose edges all face open
    /// space or another cap keeps its default loop data.
    ///
    /// Shared with [`LoopMesh::face_attribute_fill`]: passing the live set of
    /// still-unresolved destinations as `cap_set` gives the per-corner,
    /// non-transitive loop-data inheritance both operators share.
    pub(crate) fn holes_fill_copy_loop_data(&mut self, cap: FaceKey, cap_set: &HashSet<FaceKey>) {
        for cap_loop in self.face_loop_keys(cap) {
            let vertex = self.loop_vertex(cap_loop);
            let donor_face = self
                .holes_fill_loop_donor(self.loop_edge(cap_loop), cap, cap_set)
                .or_else(|| {
                    self.holes_fill_loop_donor(
                        self.loop_edge(self.loop_prev(cap_loop)),
                        cap,
                        cap_set,
                    )
                });
            let Some(donor_face) = donor_face else {
                continue;
            };
            if let Some(donor_loop) = self.holes_fill_face_loop_at(donor_face, vertex) {
                self.loop_attribs
                    .copy_into(cap_loop.as_usize(), donor_loop.as_usize());
            }
        }
    }

    /// The existing (non-cap) face bordering `edge`, if any — the donor a cap
    /// corner on `edge` inherits its loop data from.
    fn holes_fill_loop_donor(
        &self,
        edge: EdgeKey,
        cap: FaceKey,
        cap_set: &HashSet<FaceKey>,
    ) -> Option<FaceKey> {
        if edge.is_null() {
            return None;
        }
        self.edge_loop_keys(edge)
            .into_iter()
            .map(|radial| self.loop_face(radial))
            .find(|&face| face != cap && !cap_set.contains(&face))
    }

    /// The loop of `face` whose corner sits at `vertex`.
    fn holes_fill_face_loop_at(&self, face: FaceKey, vertex: VertexKey) -> Option<LoopKey> {
        self.face_loop_keys(face)
            .into_iter()
            .find(|&l| self.loop_vertex(l) == vertex)
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
    fn surface_hole_cap_inherits_material_smooth_and_corner_uv() {
        // §6.1 — every ring face uniformly carries material 3, the smooth flag,
        // and a constant corner UV; the cap inherits all three across its rim.
        let (mut mesh, rim) = grid_3x3_center_hole();
        let uv = mesh
            .register_loop_attribute(ElementAttributeDesc::numeric(
                AttributeKey::custom("uv"),
                AttrType::Vec2,
            ))
            .expect("uv layer registers");

        for face in mesh.face_keys() {
            mesh.set_face_material_index(face, 3);
            mesh.set_face_smooth(face, true);
        }
        for l in mesh.loop_keys() {
            mesh.loop_attribs_mut().set_vec2(uv, l.as_usize(), [0.25, 0.75]);
        }

        let result = mesh.holes_fill(&rim, &HolesFillParams { sides: 0 });
        assert_eq!(result.faces().len(), 1, "one cap per hole");
        let cap = result.faces()[0];

        assert_eq!(
            mesh.face_material_index(cap),
            3,
            "cap inherits the surrounding material index"
        );
        assert!(mesh.is_face_smooth(cap), "cap inherits the smooth flag");
        for cap_loop in mesh.face_loop_keys(cap) {
            let got = mesh.loop_attribs().get_vec2(uv, cap_loop.as_usize()).unwrap();
            assert_eq!(
                got, [0.25, 0.75],
                "every cap corner inherits the surrounding corner UV"
            );
        }
        mesh.validate_topology();
    }

    #[test]
    fn free_wire_cap_keeps_default_attributes() {
        // §6.3 — a cap the flood never reaches has no surface to inherit from, so
        // its material / smooth / loop data stay at their defaults.
        let mut mesh = LoopMesh::new();
        let uv = mesh
            .register_loop_attribute(ElementAttributeDesc::numeric(
                AttributeKey::custom("uv"),
                AttrType::Vec2,
            ))
            .expect("uv layer registers");
        let ring = convex_wire_ring(&mut mesh, 6, 1.0);

        let result = mesh.holes_fill(&ring, &HolesFillParams { sides: 0 });
        assert_eq!(result.faces().len(), 1, "free wire loop caps to one face");
        let cap = result.faces()[0];

        assert_eq!(mesh.face_material_index(cap), 0, "default material slot 0");
        assert!(!mesh.is_face_smooth(cap), "default cap is flat");
        for cap_loop in mesh.face_loop_keys(cap) {
            let got = mesh.loop_attribs().get_vec2(uv, cap_loop.as_usize()).unwrap();
            assert_eq!(got, [0.0, 0.0], "isolated cap keeps default corner UV");
        }
        mesh.validate_topology();
    }

    #[test]
    fn interior_wire_net_in_face_creates_nothing() {
        // §4.3 / §4.5 — an isolated quad 0-1-2-3 (the lone face) plus a wire
        // diagonal (0,2), selecting two perimeter edges and the diagonal. The
        // minimal cycle 0-1-2 is two quad-perimeter edges plus the wire diagonal.
        // The per-front gate, seeding a perimeter edge first (mesh order), leaves
        // the wire as the bare connecting edge → both fronts' flags stay low → the
        // triangle is refused, and the failed seeds are pruned so no later seeding
        // can reconstruct it. Nothing is created.
        let mut mesh = LoopMesh::new();
        let v0 = mesh.create_vertex(Vec3::new(0.0, 0.0, 0.0));
        let v1 = mesh.create_vertex(Vec3::new(1.0, 0.0, 0.0));
        let v2 = mesh.create_vertex(Vec3::new(1.0, 1.0, 0.0));
        let v3 = mesh.create_vertex(Vec3::new(0.0, 1.0, 0.0));
        // Quad first → its perimeter edges precede the diagonal in insertion order.
        mesh.face_create_verts(&[v0, v1, v2, v3], FaceCreateFlags::NO_DUPLICATE);
        let diagonal = mesh.create_edge([v0, v2], EdgeCreateFlags::NO_DUPLICATE);

        let perim_01 = mesh.query_edge_exists(v0, v1).expect("perimeter 0-1");
        let perim_12 = mesh.query_edge_exists(v1, v2).expect("perimeter 1-2");

        let result = mesh.holes_fill(
            &[perim_01, perim_12, diagonal],
            &HolesFillParams { sides: 0 },
        );

        assert!(result.is_empty(), "interior-edge-net triangle is refused");
        assert_eq!(mesh.face_count(), 1, "only the original quad remains");
        assert_eq!(mesh.vertex_count(), 4, "no vertex created");
        assert_eq!(mesh.edge_count(), 5, "the four quad edges plus the diagonal");
        mesh.validate_topology();
    }

    #[test]
    fn face_boundary_only_creates_nothing() {
        // §4.3 / §7 — selecting exactly a face's own boundary is refused: both
        // fronts stay confined to that single face (neither flag rises) and the
        // §4.4 overlap rejection independently sees the candidate covers the face.
        let mut mesh = LoopMesh::new();
        let v0 = mesh.create_vertex(Vec3::new(0.0, 0.0, 0.0));
        let v1 = mesh.create_vertex(Vec3::new(1.0, 0.0, 0.0));
        let v2 = mesh.create_vertex(Vec3::new(1.0, 1.0, 0.0));
        let v3 = mesh.create_vertex(Vec3::new(0.0, 1.0, 0.0));
        mesh.face_create_verts(&[v0, v1, v2, v3], FaceCreateFlags::NO_DUPLICATE);
        let boundary = [
            mesh.query_edge_exists(v0, v1).unwrap(),
            mesh.query_edge_exists(v1, v2).unwrap(),
            mesh.query_edge_exists(v2, v3).unwrap(),
            mesh.query_edge_exists(v3, v0).unwrap(),
        ];

        let result = mesh.holes_fill(&boundary, &HolesFillParams { sides: 0 });

        assert!(result.is_empty(), "a face's own boundary is not re-capped");
        assert_eq!(mesh.face_count(), 1, "only the original quad remains");
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
