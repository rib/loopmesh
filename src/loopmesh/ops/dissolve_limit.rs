use std::collections::BinaryHeap;
use std::collections::HashMap;

use crate::loopmesh::*;

/// Cost sentinel standing in for `+∞`: any heap top carrying it always fails
/// the strict-less-than pop gate, so a delimited / non-manifold / degenerate
/// candidate is never popped (`DISSOLVE_LIMIT.md` §3.2 / §5.1 / §6.3).
const COST_INVALID: f32 = f32::INFINITY;

/// Float-noise slack on the §6.3 flat-collinear short-circuit's
/// `|cos(corner)| < 1 − ε` gate. Mirrors the dissolve-edges `ANGLE_EPS`.
const COS_EPS: f32 = 0.0001;

/// Bit numbering of the `delimit` mask (`DISSOLVE_LIMIT.md` §3). The integer
/// values are part of the operator's public surface and must match exactly.
pub const DELIMIT_NORMAL: u32 = 1 << 0;
pub const DELIMIT_MATERIAL: u32 = 1 << 1;
pub const DELIMIT_SEAM: u32 = 1 << 2;
pub const DELIMIT_SHARP: u32 = 1 << 3;
pub const DELIMIT_UV: u32 = 1 << 4;

/// Parameters controlling [`LoopMesh::dissolve_limit`].
///
/// Mirrors the BMOP set-input shape described in
/// `radial-mesh-guides/operators/DISSOLVE_LIMIT.md` §1.
///
/// The core `delimit = 0` priority sweep — the §4 clamp, the §5 edge pass,
/// the §6 heap-driven vert pass (with the §6.3 degeneracy guard), and the §7
/// `use_dissolve_boundaries` linear sweep — is implemented. The §3
/// `delimit`-flag filtering (NORMAL / MATERIAL / SEAM / SHARP / UV) is a
/// deferred follow-up: a non-zero `delimit` declines through
/// [`DissolveLimitResult::Unsupported`] rather than silently producing a
/// wrong result.
#[derive(Clone, Copy)]
pub struct DissolveLimitParams {
    /// Dihedral / corner-angle tolerance in radians. Clamped to π/2 before
    /// either pass runs (§4). At `0` the operator is a complete no-op; at π/2
    /// (or larger) every non-delimited candidate passes the gate.
    pub angle_limit: f32,
    /// Vert-pass mode toggle (§7). When `false`, the vert pass uses a heap +
    /// corner-angle gate (§6). When `true`, it becomes an unconditional linear
    /// sweep that collapses every still-edge-pair candidate with no angle test.
    pub use_dissolve_boundaries: bool,
    /// Hard candidate filter (§3). `0` is the only implemented value this
    /// cycle; any non-zero mask declines via [`DissolveLimitResult::Unsupported`].
    pub delimit: u32,
}

impl Default for DissolveLimitParams {
    fn default() -> Self {
        Self {
            angle_limit: std::f32::consts::FRAC_PI_2,
            use_dissolve_boundaries: false,
            delimit: 0,
        }
    }
}

/// Outcome of [`LoopMesh::dissolve_limit`].
///
/// `Ran` carries the §1 `region.out` analogue — the surviving merge anchors
/// produced by the edge pass (one per live anchor after the FACE_NEW-style
/// sweep, *not* one per heap pop). `Unsupported` reports a parameter tuple
/// outside the implemented surface; the mesh is left untouched.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum DissolveLimitResult {
    Ran { region_out: Vec<FaceKey> },
    Unsupported,
}

/// One queued candidate in the edge or vert priority heap. The heap is
/// ordered ascending by `cost` (cheapest popped first). `input_index` is the
/// candidate's slot in the input array — the §5.3 "tagged-pointer" indirection
/// loopmesh resolves through the `generation` / `still_pending` side tables
/// rather than a settable `elem_index`. `generation` discards stale entries
/// left behind by a re-score (`BinaryHeap` has no in-place decrease-key).
#[derive(Copy, Clone)]
struct HeapCandidate {
    cost: f32,
    input_index: u32,
    generation: u32,
}

impl PartialEq for HeapCandidate {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == std::cmp::Ordering::Equal
    }
}
impl Eq for HeapCandidate {}
impl PartialOrd for HeapCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for HeapCandidate {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // `BinaryHeap` is a max-heap; invert so the lowest cost pops first.
        // The input index breaks cost ties deterministically (the heap
        // tie-breaker `DISSOLVE_LIMIT.md` §11 leaves to the implementation).
        other
            .cost
            .total_cmp(&self.cost)
            .then(other.input_index.cmp(&self.input_index))
    }
}

/// A live-but-supplanted heap entry guard: a `generation` table lets a
/// re-score push a fresh entry and lazily skip the stale one on pop.
struct PriorityHeap {
    heap: BinaryHeap<HeapCandidate>,
    /// Per-input-index latest generation. A popped entry whose generation is
    /// behind this value is stale and skipped.
    generation: Vec<u32>,
    /// Per-input-index live cost, mirrored so the §6.2-step-5 reopening can
    /// test "was this previously `COST_INVALID`" without popping.
    cost: Vec<f32>,
    /// Per-input-index liveness (the guide's null `heap_table[i]` slot).
    active: Vec<bool>,
}

impl PriorityHeap {
    fn with_capacity(len: usize) -> Self {
        Self {
            heap: BinaryHeap::new(),
            generation: vec![0; len],
            cost: vec![COST_INVALID; len],
            active: vec![false; len],
        }
    }

    /// Seed input index `i` at `cost`.
    fn insert(&mut self, input_index: usize, cost: f32) {
        let generation = self.generation[input_index] + 1;
        self.generation[input_index] = generation;
        self.cost[input_index] = cost;
        self.active[input_index] = true;
        self.heap.push(HeapCandidate {
            cost,
            input_index: input_index as u32,
            generation,
        });
    }

    /// Re-score input index `i` (the §5.2-step-4 / §6.2-step-4 update). Stale
    /// entries for `i` are discarded lazily by their behind-generation tag.
    fn update_cost(&mut self, input_index: usize, cost: f32) {
        if !self.active[input_index] {
            return;
        }
        let generation = self.generation[input_index] + 1;
        self.generation[input_index] = generation;
        self.cost[input_index] = cost;
        self.heap.push(HeapCandidate {
            cost,
            input_index: input_index as u32,
            generation,
        });
    }

    /// Mark input index `i` consumed (its `heap_table` slot nulled, §5.3).
    fn deactivate(&mut self, input_index: usize) {
        self.active[input_index] = false;
    }

    /// Pop the live cheapest candidate whose cost is strictly below
    /// `threshold`, or `None` once the cheapest live candidate meets/exceeds
    /// it (the §5.2 / §6.2 step-1 pop gate).
    fn pop_below(&mut self, threshold: f32) -> Option<usize> {
        while let Some(top) = self.heap.peek().copied() {
            let i = top.input_index as usize;
            if !self.active[i] || self.generation[i] != top.generation {
                self.heap.pop();
                continue;
            }
            if !(top.cost < threshold) {
                return None;
            }
            self.heap.pop();
            return Some(i);
        }
        None
    }

    fn current_cost(&self, input_index: usize) -> f32 {
        self.cost[input_index]
    }
}

impl LoopMesh {
    /// Greedily merge co-planar faces and collinear vert chains until removing
    /// the next candidate would alter the surface by more than `angle_limit`.
    ///
    /// Implements the `delimit = 0` path of
    /// `radial-mesh-guides/operators/DISSOLVE_LIMIT.md`:
    ///
    /// - **§4** — `angle_limit` is clamped to π/2 before either pass runs.
    /// - **§5 edge pass** — a min-heap over `edges`, keyed by the signed
    ///   negated-dot cost (§5.1), pops the lowest-cost candidate while its cost
    ///   is strictly below `−cos(angle_limit)` and merges its two flanking
    ///   faces via the n==2 `faces_join` form (the same JFKE merge
    ///   `dissolve_edges` uses, `DISSOLVE_EDGES.md` §3.1). Each merged face's
    ///   surviving perimeter edges are re-scored against the fresh normal
    ///   (§5.2 step 4). The first-passed loop's face wins per-face customdata
    ///   (§11).
    /// - **§5.4 cleanup** — edges that became wire as a side effect of a merge
    ///   are killed, and any endpoint left isolated is killed and struck from
    ///   the vert candidate set.
    /// - **§6 vert pass** — a min-heap over `verts`, keyed by the §6.1
    ///   modulated corner-angle cost, pops while the cost is below
    ///   `angle_limit` and collapses each via the vert-collapse-edge primitive
    ///   (`DISSOLVE_VERTS.md` §6.4). The §6.3 degeneracy guard demotes a
    ///   would-be-self-intersecting collapse to `COST_INVALID` in place (rather
    ///   than removing it), and the §6.2-step-5 reopening re-evaluates
    ///   neighbours previously demoted.
    /// - **§7** — with `use_dissolve_boundaries`, the vert pass is replaced by
    ///   an unconditional linear sweep with no heap, cost, or degeneracy test.
    ///
    /// Returns the §1 `region.out` analogue: the surviving merge anchors. A
    /// non-zero `delimit` is **not** implemented this cycle and declines via
    /// [`DissolveLimitResult::Unsupported`] (delimit-flag filtering is a
    /// deferred follow-up).
    pub fn dissolve_limit(
        &mut self,
        edges: &[EdgeKey],
        verts: &[VertexKey],
        params: &DissolveLimitParams,
    ) -> DissolveLimitResult {
        // §3 delimit-flag filtering is deferred; only the unfiltered path is
        // implemented. Decline rather than silently mis-filter.
        if params.delimit != 0 {
            return DissolveLimitResult::Unsupported;
        }

        // §4 — clamp the tolerance to π/2 and derive the negated-cosine form
        // the edge pass compares against.
        let angle_limit = params.angle_limit.min(std::f32::consts::FRAC_PI_2);
        let edge_threshold = -angle_limit.cos();

        // §5.2 — tag every edge that is wire *on entry*, so the §5.4 cleanup can
        // tell "was wire on entry" from "became wire because of a merge" and
        // only reaps the latter. The tag is a pushed op-flag layer that vanishes
        // when the operator returns.
        let wire_on_entry = self.push_edge_op_flag_layer();
        const WIRE_BIT: u8 = 0;
        {
            let mut edge_iter = self.mesh_edges();
            let mut wires: smallvec::SmallVec<[EdgeKey; 16]> = smallvec::SmallVec::new();
            while let Some(edge) = edge_iter.next(self) {
                if self.edges.get(edge).radial_link.is_none() {
                    wires.push(edge);
                }
            }
            for edge in wires {
                self.edge_op_flag_set(wire_on_entry, edge, WIRE_BIT);
            }
        }

        // Face normals drive the edge-pass cost; refresh them so a freshly
        // built mesh (whose faces may carry an un-computed normal) scores
        // correctly. The edge pass refreshes each merged face itself.
        self.calc_face_normals();

        let region_out = self.dissolve_limit_edge_pass(edges, edge_threshold);

        // §5.4 — the vert input set may lose entries the cleanup orphaned;
        // build a live, deduplicated working set before the vert pass.
        let mut vert_input: Vec<Option<VertexKey>> = verts
            .iter()
            .map(|&v| {
                if self.vertices.try_get(v).is_some() {
                    Some(v)
                } else {
                    None
                }
            })
            .collect();
        self.dissolve_limit_edge_pass_cleanup(&mut vert_input, wire_on_entry, WIRE_BIT);
        self.pop_edge_op_flag_layer(wire_on_entry);

        if params.use_dissolve_boundaries {
            // §7 — unconditional linear sweep, no heap / angle / degeneracy.
            for slot in vert_input.iter() {
                let Some(v) = *slot else { continue };
                if self.vertices.try_get(v).is_none() {
                    continue;
                }
                if self.dissolve_limit_vert_edge_count(v) == 2 {
                    let Some(e_kill) = self.vertex_disk_iter(v).next() else {
                        continue;
                    };
                    let _ = self.vert_collapse_edge(e_kill, v);
                }
            }
        } else {
            self.dissolve_limit_vert_pass(&vert_input, angle_limit);
        }

        DissolveLimitResult::Ran { region_out }
    }

    /// §5 — the edge pass. Returns the §1 `region.out` surviving merge anchors.
    fn dissolve_limit_edge_pass(
        &mut self,
        edges: &[EdgeKey],
        edge_threshold: f32,
    ) -> Vec<FaceKey> {
        // §5.3 — the edge → input-index map standing in for `BM_elem_index`.
        // A later re-score looks an edge up here to find its heap slot.
        let mut index_of: HashMap<EdgeKey, usize> = HashMap::with_capacity(edges.len());
        let mut heap = PriorityHeap::with_capacity(edges.len());
        for (i, &edge) in edges.iter().enumerate() {
            if self.edges.try_get(edge).is_none() {
                continue;
            }
            // First-wins if the caller passed a duplicate edge.
            index_of.entry(edge).or_insert(i);
            let cost = self.dissolve_limit_edge_cost(edge);
            heap.insert(i, cost);
        }

        // §1 / §5.2 step 4 — merged faces are tagged so the tail FACE_NEW-style
        // sweep can collect the surviving anchors (chained merges re-tag the
        // survivor, so an absorbed intermediate anchor drops out naturally).
        let region_tag = self.push_face_op_flag_layer();
        const REGION_BIT: u8 = 0;

        while let Some(i) = heap.pop_below(edge_threshold) {
            let edge = edges[i];
            if self.edges.try_get(edge).is_none() {
                heap.deactivate(i);
                continue;
            }
            // §5.2 step 2 — re-verify manifoldness against the current topology.
            let Some((survivor, other)) = self.dissolve_limit_edge_face_pair(edge) else {
                heap.update_cost(i, COST_INVALID);
                continue;
            };

            // §5.2 step 3 — merge via the n==2 `faces_join` form. The survivor
            // (the `e.l` face) is passed first so it keeps its identity and CD
            // (§11), matching the dissolve-edges merge.
            let merged = self.faces_join(&[survivor, other]);

            match merged {
                Some(f_new) => {
                    // §5.2 step 4 — consume this edge, refresh the merged
                    // normal, emit to region.out, and re-score the perimeter.
                    heap.deactivate(i);
                    index_of.remove(&edge);
                    self.face_normal_update(f_new);
                    self.face_op_flag_set(region_tag, f_new, REGION_BIT);

                    let perimeter: smallvec::SmallVec<[EdgeKey; 16]> = {
                        let mut out: smallvec::SmallVec<[EdgeKey; 16]> = smallvec::SmallVec::new();
                        let mut loops = self.face_loops(f_new);
                        while let Some(loop_key) = loops.next(self) {
                            out.push(self.loops.get(loop_key).edge);
                        }
                        out
                    };
                    for perimeter_edge in perimeter {
                        if let Some(&j) = index_of.get(&perimeter_edge) {
                            let cost = self.dissolve_limit_edge_cost(perimeter_edge);
                            heap.update_cost(j, cost);
                        }
                    }
                }
                None => {
                    // §5.2 step 5 — the merge could not run; pin the edge out.
                    heap.update_cost(i, COST_INVALID);
                }
            }
        }

        // §1 — FACE_NEW sweep: surviving anchors are the still-live tagged faces.
        let mut region_out: Vec<FaceKey> = Vec::new();
        let mut face_iter = self.mesh_faces();
        while let Some(face) = face_iter.next(self) {
            if self.face_op_flag_test(region_tag, face, REGION_BIT) {
                region_out.push(face);
            }
        }
        self.pop_face_op_flag_layer(region_tag);
        region_out
    }

    /// §5.4 — kill edges that became wire as a side effect of the merges (an
    /// edge that bounds no face yet was *not* wire on entry, per the
    /// `wire_on_entry` tag), and strike any endpoint left isolated from the
    /// vert candidate set so the vert pass never touches a freed vert.
    fn dissolve_limit_edge_pass_cleanup(
        &mut self,
        vert_input: &mut [Option<VertexKey>],
        wire_on_entry: crate::op_flags::EdgeOpFlagLayer,
        wire_bit: u8,
    ) {
        // Reverse map from vert to its slot(s) in the candidate set.
        let mut slot_of: HashMap<VertexKey, smallvec::SmallVec<[usize; 2]>> = HashMap::new();
        for (slot, entry) in vert_input.iter().enumerate() {
            if let Some(v) = entry {
                slot_of.entry(*v).or_default().push(slot);
            }
        }

        // A merge can strand a perimeter edge with no adjacent face. Such an
        // edge is now wire and was not wire on entry; kill it, then kill any
        // endpoint it isolates.
        let wire_orphans: Vec<EdgeKey> = {
            let mut out: Vec<EdgeKey> = Vec::new();
            let mut edge_iter = self.mesh_edges();
            while let Some(edge) = edge_iter.next(self) {
                if self.edges.get(edge).radial_link.is_none()
                    && !self.edge_op_flag_test(wire_on_entry, edge, wire_bit)
                {
                    out.push(edge);
                }
            }
            out
        };

        for edge in wire_orphans {
            if self.edges.try_get(edge).is_none() {
                continue;
            }
            let endpoints = self.edges.get(edge).vertices;
            self.edge_kill(edge);
            for v in endpoints {
                if self.vertices.try_get(v).is_none() {
                    continue;
                }
                if self.vertices.get(v).disk_loop_start.is_none() {
                    if let Some(slots) = slot_of.get(&v) {
                        for &slot in slots {
                            vert_input[slot] = None;
                        }
                    }
                    self.vertex_kill(v);
                }
            }
        }
    }

    /// §6 — the heap-driven vert pass.
    fn dissolve_limit_vert_pass(&mut self, vert_input: &[Option<VertexKey>], angle_limit: f32) {
        let mut index_of: HashMap<VertexKey, usize> = HashMap::with_capacity(vert_input.len());
        let mut heap = PriorityHeap::with_capacity(vert_input.len());
        for (i, slot) in vert_input.iter().enumerate() {
            let Some(v) = *slot else { continue };
            if self.vertices.try_get(v).is_none() {
                continue;
            }
            index_of.entry(v).or_insert(i);
            let cost = self.dissolve_limit_vert_cost(v);
            heap.insert(i, cost);
        }

        while let Some(i) = heap.pop_below(angle_limit) {
            let Some(v) = vert_input[i] else {
                heap.deactivate(i);
                continue;
            };
            if self.vertices.try_get(v).is_none() {
                heap.deactivate(i);
                continue;
            }

            // §6.2 step 2 / §6.3 — degeneracy guard: a failing collapse is
            // demoted to COST_INVALID in place, never removed.
            if self.dissolve_limit_vert_collapse_is_degenerate(v) {
                heap.update_cost(i, COST_INVALID);
                continue;
            }

            let Some(e_kill) = self.vertex_disk_iter(v).next() else {
                heap.update_cost(i, COST_INVALID);
                continue;
            };
            let collapsed = self.vert_collapse_edge(e_kill, v);

            match collapsed {
                Some(e_new) => {
                    heap.deactivate(i);
                    index_of.remove(&v);

                    // §6.2 step 4 — re-score the surviving edge's endpoints.
                    let endpoints = self.edges.get(e_new).vertices;
                    for v_other in endpoints {
                        if let Some(&j) = index_of.get(&v_other) {
                            let cost = self.dissolve_limit_vert_cost(v_other);
                            heap.update_cost(j, cost);
                        }
                    }

                    // §6.2 step 5 — refresh incident face normals and reopen
                    // any perimeter neighbour previously demoted as degenerate.
                    let incident_faces: smallvec::SmallVec<[FaceKey; 4]> = {
                        let mut out: smallvec::SmallVec<[FaceKey; 4]> = smallvec::SmallVec::new();
                        let mut radial = self.radial_loops(e_new);
                        while let Some(loop_key) = radial.next(self) {
                            let face = self.loops.get(loop_key).face;
                            if !out.contains(&face) {
                                out.push(face);
                            }
                        }
                        out
                    };
                    for face in incident_faces {
                        self.face_normal_update(face);
                        let perimeter: smallvec::SmallVec<[VertexKey; 16]> = {
                            let mut out: smallvec::SmallVec<[VertexKey; 16]> =
                                smallvec::SmallVec::new();
                            let mut loops = self.face_loops(face);
                            while let Some(loop_key) = loops.next(self) {
                                out.push(self.loops.get(loop_key).vertex);
                            }
                            out
                        };
                        for neighbour in perimeter {
                            if endpoints.contains(&neighbour) {
                                continue;
                            }
                            if let Some(&j) = index_of.get(&neighbour) {
                                if heap.current_cost(j) == COST_INVALID {
                                    let cost = self.dissolve_limit_vert_cost(neighbour);
                                    heap.update_cost(j, cost);
                                }
                            }
                        }
                    }
                }
                None => {
                    heap.update_cost(i, COST_INVALID);
                }
            }
        }
    }

    /// §5.1 — per-edge dissolve cost. `COST_INVALID` for a non-manifold edge;
    /// otherwise the signed negated dot of the two face normals (the sign flip
    /// makes "coplanar / no fold" the lowest cost for both contiguous and
    /// non-contiguous edges).
    fn dissolve_limit_edge_cost(&self, edge: EdgeKey) -> f32 {
        let Some((face_a, face_b)) = self.dissolve_limit_edge_face_pair(edge) else {
            return COST_INVALID;
        };
        let normal_a = self.faces.get(face_a).normal;
        let normal_b = self.faces.get(face_b).normal;
        let dot = normal_a.dot(normal_b);
        if self.dissolve_limit_edge_is_contiguous(edge) {
            -dot
        } else {
            dot
        }
    }

    /// §6.1 — per-vert dissolve cost. The corner angle, modulated by the
    /// incident-edge dihedral when the canonical edge is manifold (the
    /// multiplicative collinear-and-coplanar form). A non-edge-pair vert is
    /// `COST_INVALID` (it has no two-edge corner to collapse).
    fn dissolve_limit_vert_cost(&self, v: VertexKey) -> f32 {
        let Some(corner_angle) = self.dissolve_limit_vert_corner_angle(v) else {
            return COST_INVALID;
        };
        // delimit = 0 this cycle: every manifold-canonical-edge vert takes the
        // modulated form; delimiter-aware routing is a deferred follow-up.
        let primary_edge = self
            .vertices
            .get(v)
            .disk_loop_start
            .expect("edge-pair vert has a disk-head edge");
        if let Some(edge_angle) = self.dissolve_limit_edge_dihedral(primary_edge) {
            corner_angle * edge_angle / std::f32::consts::FRAC_PI_2
        } else {
            corner_angle
        }
    }

    /// §6.1 — corner angle for an edge-pair vert; `None` for any other valence
    /// (the §6.1 sentinel). The cost is the *deviation from straight*, so a
    /// collinear corner yields 0 and a right-angle corner yields π/2 (per the
    /// §2 / §8 worked examples where "collinear ⇒ corner angle ≈ 0"). This is
    /// `π − ∠(a − v, b − v)`, the same convention the sibling edge-pair angle
    /// in `dissolve_edges` uses (`DISSOLVE_EDGES.md` §5.2.1).
    fn dissolve_limit_vert_corner_angle(&self, v: VertexKey) -> Option<f32> {
        if self.dissolve_limit_vert_edge_count(v) != 2 {
            return None;
        }
        let mut disk = self.vertex_disk_iter(v);
        let edge_a = disk.next()?;
        let edge_b = disk.next()?;
        let far_a = self.edges.get(edge_a).other_vertex(v)?;
        let far_b = self.edges.get(edge_b).other_vertex(v)?;
        let position = self.vertices.get(v).position;
        let direction_a = self.vertices.get(far_a).position - position;
        let direction_b = self.vertices.get(far_b).position - position;
        Some(std::f32::consts::PI - direction_a.angle_between(direction_b))
    }

    /// §6.1 — dihedral of a manifold edge: the angle between its two face
    /// normals, `[0, π]`. `None` when the edge is not a manifold face pair.
    fn dissolve_limit_edge_dihedral(&self, edge: EdgeKey) -> Option<f32> {
        let (face_a, face_b) = self.dissolve_limit_edge_face_pair(edge)?;
        let normal_a = self.faces.get(face_a).normal;
        let normal_b = self.faces.get(face_b).normal;
        Some(normal_a.angle_between(normal_b))
    }

    /// §6.3 — degeneracy guard. Returns `true` when collapsing `v` would
    /// produce a self-intersecting face or a flipped corner.
    ///
    /// 1. Non-edge-pair verts cannot collapse → degenerate.
    /// 2. Wire passthrough: a wire incident edge has no face geometry to
    ///    invalidate → allow.
    /// 3. Flat-collinear short-circuit: a near-perfectly-straight corner
    ///    (`|cos(corner)| ≥ 1 − ε`) needs no 2D test → allow. Otherwise run
    ///    the per-face corner-flip + point-in-triangle checks on each incident
    ///    face with more than three corners.
    fn dissolve_limit_vert_collapse_is_degenerate(&self, v: VertexKey) -> bool {
        if self.dissolve_limit_vert_edge_count(v) != 2 {
            return true;
        }

        let mut disk = self.vertex_disk_iter(v);
        let edge_a = disk.next().expect("edge-pair vert has a first edge");
        let edge_b = disk.next().expect("edge-pair vert has a second edge");

        // §6.3 step 2 — wire passthrough.
        if self.edges.get(edge_a).radial_link.is_none()
            || self.edges.get(edge_b).radial_link.is_none()
        {
            return false;
        }

        let far_a = match self.edges.get(edge_a).other_vertex(v) {
            Some(v) => v,
            None => return true,
        };
        let far_b = match self.edges.get(edge_b).other_vertex(v) {
            Some(v) => v,
            None => return true,
        };
        let position = self.vertices.get(v).position;
        let direction_a = (self.vertices.get(far_a).position - position).normalize_or_zero();
        let direction_b = (self.vertices.get(far_b).position - position).normalize_or_zero();
        let cosine = direction_a.dot(direction_b);

        // §6.3 step 3 — a near-straight corner needs no 2D test.
        if cosine.abs() >= 1.0 - COS_EPS {
            return false;
        }

        // For each incident face with more than three corners, test the
        // collapse against corner-flip and point-in-triangle.
        let incident_faces: smallvec::SmallVec<[FaceKey; 4]> = {
            let mut out: smallvec::SmallVec<[FaceKey; 4]> = smallvec::SmallVec::new();
            for edge in [edge_a, edge_b] {
                let mut radial = self.radial_loops(edge);
                while let Some(loop_key) = radial.next(self) {
                    let face = self.loops.get(loop_key).face;
                    if !out.contains(&face) {
                        out.push(face);
                    }
                }
            }
            out
        };

        for face in incident_faces {
            if self.face_len(face) <= 3 {
                continue;
            }
            if self.dissolve_limit_face_collapse_is_degenerate(face, v, far_a, far_b) {
                return true;
            }
        }
        false
    }

    /// §6.3 step 3 — the per-face 2D degeneracy test. The collapsed corner is
    /// `v` between `far_a` and `far_b`; the new edge runs `far_a – far_b`. The
    /// test projects the face into its dominant axis plane and rejects the
    /// collapse when (a) the cut-off corner triangle winds *against* the face
    /// (a corner flip), or (b) any other face vertex falls inside the cut-off
    /// triangle `(far_a, v, far_b)` (a self-intersection).
    fn dissolve_limit_face_collapse_is_degenerate(
        &self,
        face: FaceKey,
        v: VertexKey,
        far_a: VertexKey,
        far_b: VertexKey,
    ) -> bool {
        let normal = self.faces.get(face).normal;
        let axis = dominant_axis(normal);

        let pivot = project_to_axis(self.vertices.get(v).position, axis);
        let corner_prev = project_to_axis(self.vertices.get(far_a).position, axis);
        let corner_next = project_to_axis(self.vertices.get(far_b).position, axis);

        // Face winding sign in the projected plane: positive when the face's
        // loop order is CCW about the +normal-axis. The projection drops the
        // dominant axis, so a face with a negative dominant-axis normal winds
        // clockwise in the kept plane — sample the winding directly rather than
        // assuming a sign.
        let face_winding = self.dissolve_limit_projected_face_winding(face, axis);

        // Corner-flip: the cut-off corner triangle (far_a → v → far_b) must
        // wind the same way as the face. A flipped sign inverts the corner.
        let corner_winding = signed_area(corner_prev, pivot, corner_next);
        if face_winding != 0.0 && corner_winding * face_winding < 0.0 {
            return true;
        }

        // Self-intersection: no other corner may sit inside the cut-off
        // triangle.
        let mut loops = self.face_loops(face);
        while let Some(loop_key) = loops.next(self) {
            let corner_vert = self.loops.get(loop_key).vertex;
            if corner_vert == v || corner_vert == far_a || corner_vert == far_b {
                continue;
            }
            let point = project_to_axis(self.vertices.get(corner_vert).position, axis);
            if point_in_triangle(point, corner_prev, pivot, corner_next) {
                return true;
            }
        }
        false
    }

    /// §6.3 — the signed area of the face's perimeter projected onto the kept
    /// plane (the dominant axis dropped). Its sign is the face's winding
    /// direction in that plane, used to validate the cut-corner orientation.
    fn dissolve_limit_projected_face_winding(&self, face: FaceKey, axis: usize) -> f32 {
        let mut area = 0.0f32;
        let mut loops = self.face_loops(face);
        let mut first: Option<glam::Vec2> = None;
        let mut prev: Option<glam::Vec2> = None;
        while let Some(loop_key) = loops.next(self) {
            let point = project_to_axis(
                self.vertices.get(self.loops.get(loop_key).vertex).position,
                axis,
            );
            if first.is_none() {
                first = Some(point);
            }
            if let Some(previous) = prev {
                area += previous.x * point.y - point.x * previous.y;
            }
            prev = Some(point);
        }
        if let (Some(last), Some(start)) = (prev, first) {
            area += last.x * start.y - start.x * last.y;
        }
        area
    }

    /// True when `edge`'s two radial loops traverse it in opposite directions
    /// (the §5.1 contiguous case). For a manifold pair the two loops start at
    /// different endpoints exactly when the faces wind consistently.
    fn dissolve_limit_edge_is_contiguous(&self, edge: EdgeKey) -> bool {
        let mut radial = self.radial_loops(edge);
        let Some(loop_a) = radial.next(self) else {
            return true;
        };
        let Some(loop_b) = radial.next(self) else {
            return true;
        };
        self.loops.get(loop_a).vertex != self.loops.get(loop_b).vertex
    }

    /// The two distinct faces flanking a manifold `edge`, ordered `e.l` first
    /// so the merge keeps the §11 customdata winner. `None` for wire /
    /// boundary / self-radial / non-manifold edges.
    ///
    /// Mirrors the dissolve-edges §3.1 survivor resolution: loopmesh's radial
    /// cycle emits the first-built face's loop at the head, so the `e.l`
    /// survivor is the head's radial-next (the second-built face), passed first.
    fn dissolve_limit_edge_face_pair(&self, edge: EdgeKey) -> Option<(FaceKey, FaceKey)> {
        let mut radial = self.radial_loops(edge);
        let head = radial.next(self)?;
        let second = radial.next(self)?;
        if radial.next(self).is_some() {
            return None;
        }
        let head_face = self.loops.get(head).face;
        let second_face = self.loops.get(second).face;
        if head_face == second_face {
            return None;
        }
        Some((second_face, head_face))
    }

    /// Count `v`'s incident edges, capped at 3 (edge-pair detection).
    fn dissolve_limit_vert_edge_count(&self, v: VertexKey) -> usize {
        let mut disk = self.disk_edges(v);
        let mut count = 0usize;
        while count < 3 && disk.next(self).is_some() {
            count += 1;
        }
        count
    }
}

/// The axis (0 = X, 1 = Y, 2 = Z) of largest absolute normal component — the
/// face's dominant axis, dropped to project the face into 2D (§6.3 step 3).
fn dominant_axis(normal: Vec3) -> usize {
    let abs = normal.abs();
    if abs.x >= abs.y && abs.x >= abs.z {
        0
    } else if abs.y >= abs.z {
        1
    } else {
        2
    }
}

/// Drop the dominant axis component, yielding a 2D point.
fn project_to_axis(position: Vec3, axis: usize) -> glam::Vec2 {
    match axis {
        0 => glam::Vec2::new(position.y, position.z),
        1 => glam::Vec2::new(position.x, position.z),
        _ => glam::Vec2::new(position.x, position.y),
    }
}

/// Twice the signed area of triangle `(a, b, c)`.
fn signed_area(a: glam::Vec2, b: glam::Vec2, c: glam::Vec2) -> f32 {
    (b.x - a.x) * (c.y - a.y) - (c.x - a.x) * (b.y - a.y)
}

/// True when `point` lies inside (or on) triangle `(a, b, c)`, orientation-
/// agnostic.
fn point_in_triangle(point: glam::Vec2, a: glam::Vec2, b: glam::Vec2, c: glam::Vec2) -> bool {
    let d1 = signed_area(point, a, b);
    let d2 = signed_area(point, b, c);
    let d3 = signed_area(point, c, a);
    let has_neg = d1 < 0.0 || d2 < 0.0 || d3 < 0.0;
    let has_pos = d1 > 0.0 || d2 > 0.0 || d3 > 0.0;
    !(has_neg && has_pos)
}

#[cfg(test)]
mod tests {
    use std::f32::consts::FRAC_PI_2;

    use glam::Vec3;

    use crate::loopmesh::*;

    /// Build a flat coplanar strip of `n` quads in the XY plane. The bottom row
    /// verts are `verts[0..=n]` at `(i, 0, 0)`, the top row `verts[n+1..=2n+1]`
    /// at `(i, 1, 0)`. Quad `i` (CCW) is `(b_i, b_{i+1}, t_{i+1}, t_i)`.
    ///
    /// Returns `(verts, interior_edges, interior_verts)` where
    /// `interior_edges[i]` is the vertical edge shared between quad `i` and quad
    /// `i+1` (the edge `b_{i+1} – t_{i+1}`), and `interior_verts` are the
    /// two-per-seam rim verts `b_{i+1}` / `t_{i+1}` that the vert pass collapses.
    fn build_coplanar_strip(
        mesh: &mut LoopMesh,
        n: usize,
    ) -> (Vec<VertexKey>, Vec<EdgeKey>, Vec<VertexKey>) {
        let mut bottom: Vec<VertexKey> = Vec::with_capacity(n + 1);
        let mut top: Vec<VertexKey> = Vec::with_capacity(n + 1);
        for i in 0..=n {
            bottom.push(mesh.create_vertex(Vec3::new(i as f32, 0.0, 0.0)));
        }
        for i in 0..=n {
            top.push(mesh.create_vertex(Vec3::new(i as f32, 1.0, 0.0)));
        }
        for i in 0..n {
            let b0 = bottom[i];
            let b1 = bottom[i + 1];
            let t1 = top[i + 1];
            let t0 = top[i];
            let e_b = mesh.create_edge([b0, b1], EdgeCreateFlags::NO_DUPLICATE);
            let e_r = mesh.create_edge([b1, t1], EdgeCreateFlags::NO_DUPLICATE);
            let e_t = mesh.create_edge([t1, t0], EdgeCreateFlags::NO_DUPLICATE);
            let e_l = mesh.create_edge([t0, b0], EdgeCreateFlags::NO_DUPLICATE);
            mesh.face_create(&[b0, b1, t1, t0], &[e_b, e_r, e_t, e_l], FaceCreateFlags::NONE);
        }

        let mut verts: Vec<VertexKey> = bottom.clone();
        verts.extend_from_slice(&top);

        let mut interior_edges = Vec::with_capacity(n - 1);
        let mut interior_verts = Vec::with_capacity(2 * (n - 1));
        for i in 1..n {
            interior_edges.push(
                mesh.query_edge_exists(bottom[i], top[i])
                    .expect("interior seam edge present"),
            );
            interior_verts.push(bottom[i]);
            interior_verts.push(top[i]);
        }
        (verts, interior_edges, interior_verts)
    }

    #[test]
    fn all_flat_strip_full_collapse() {
        // §8 "Coplanar 4-quad strip": every interior edge has cost −1, every
        // interior vert corner angle 0, so the edge pass merges to one face and
        // the vert pass strips the rim back to a quad. 4 quads → one quad.
        let mut mesh = LoopMesh::new();
        let (_verts, edges, interior_verts) = build_coplanar_strip(&mut mesh, 4);

        assert_eq!((mesh.totvert(), mesh.totface()), (10, 4));

        let params = DissolveLimitParams {
            angle_limit: std::f32::consts::FRAC_PI_4,
            use_dissolve_boundaries: false,
            delimit: 0,
        };
        let result = mesh.dissolve_limit(&edges, &interior_verts, &params);
        assert!(matches!(result, DissolveLimitResult::Ran { .. }));

        assert_eq!(mesh.totface(), 1, "all four quads merge into one");
        assert_eq!(mesh.totvert(), 4, "the rim collapses to a single quad");
        assert_eq!(mesh.totedge(), 4);
        let face = mesh.face_keys()[0];
        assert_eq!(mesh.face_len(face), 4, "the surviving face is a quad");
        mesh.validate_topology();
    }

    #[test]
    fn angle_limit_zero_is_noop() {
        // §4 — at angle_limit 0 the strict-less-than gate rejects even exactly
        // coplanar edges (−1 < −cos(0) = −1 is false). Complete no-op.
        let mut mesh = LoopMesh::new();
        let (_verts, edges, interior_verts) = build_coplanar_strip(&mut mesh, 4);
        let v0 = mesh.totvert();
        let e0 = mesh.totedge();
        let f0 = mesh.totface();

        let params = DissolveLimitParams {
            angle_limit: 0.0,
            use_dissolve_boundaries: false,
            delimit: 0,
        };
        let result = mesh.dissolve_limit(&edges, &interior_verts, &params);
        match result {
            DissolveLimitResult::Ran { region_out } => {
                assert!(region_out.is_empty(), "no merges, so no region.out anchors");
            }
            DissolveLimitResult::Unsupported => panic!("delimit 0 must run"),
        }
        assert_eq!((mesh.totvert(), mesh.totedge(), mesh.totface()), (v0, e0, f0));
        mesh.validate_topology();
    }

    #[test]
    fn partial_fold_ridge_preserved() {
        // §8 "Strip with one sharp ridge": a 3-quad strip folded 90° at the
        // second interior seam. With angle_limit = π/4 the flat seam (cost −1)
        // dissolves but the 90° ridge (cost 0, threshold −0.707) survives.
        let mut mesh = LoopMesh::new();
        // Quad 0: (0,0)-(1,0) flat; Quad 1: (1,0)-(2,0) flat; Quad 2 folds up.
        let b: Vec<VertexKey> = vec![
            mesh.create_vertex(Vec3::new(0.0, 0.0, 0.0)),
            mesh.create_vertex(Vec3::new(1.0, 0.0, 0.0)),
            mesh.create_vertex(Vec3::new(2.0, 0.0, 0.0)),
            mesh.create_vertex(Vec3::new(2.0, 0.0, 1.0)), // folded up 90°
        ];
        let t: Vec<VertexKey> = vec![
            mesh.create_vertex(Vec3::new(0.0, 1.0, 0.0)),
            mesh.create_vertex(Vec3::new(1.0, 1.0, 0.0)),
            mesh.create_vertex(Vec3::new(2.0, 1.0, 0.0)),
            mesh.create_vertex(Vec3::new(2.0, 1.0, 1.0)),
        ];
        for i in 0..3 {
            let e_b = mesh.create_edge([b[i], b[i + 1]], EdgeCreateFlags::NO_DUPLICATE);
            let e_r = mesh.create_edge([b[i + 1], t[i + 1]], EdgeCreateFlags::NO_DUPLICATE);
            let e_t = mesh.create_edge([t[i + 1], t[i]], EdgeCreateFlags::NO_DUPLICATE);
            let e_l = mesh.create_edge([t[i], b[i]], EdgeCreateFlags::NO_DUPLICATE);
            mesh.face_create(
                &[b[i], b[i + 1], t[i + 1], t[i]],
                &[e_b, e_r, e_t, e_l],
                FaceCreateFlags::NONE,
            );
        }
        let flat_seam = mesh.query_edge_exists(b[1], t[1]).unwrap();
        let ridge_seam = mesh.query_edge_exists(b[2], t[2]).unwrap();

        let params = DissolveLimitParams {
            angle_limit: std::f32::consts::FRAC_PI_4,
            use_dissolve_boundaries: false,
            delimit: 0,
        };
        // No verts in this call: isolate the edge-pass angle gate.
        mesh.dissolve_limit(&[flat_seam, ridge_seam], &[], &params);

        assert!(
            mesh.query_edge_exists(b[1], t[1]).is_none(),
            "the flat seam dissolves"
        );
        assert!(
            mesh.query_edge_exists(b[2], t[2]).is_some(),
            "the 90° ridge seam survives the angle gate"
        );
        assert_eq!(mesh.totface(), 2, "two flat quads merge; the folded one stays");
        mesh.validate_topology();
    }

    #[test]
    fn angle_limit_pi_clamps_to_pi_over_2() {
        // §4 — angle_limit = π behaves identically to π/2. A 60° fold dissolves
        // under both (cost −cos(60°) = −0.5 < threshold 0).
        for angle_limit in [FRAC_PI_2, std::f32::consts::PI] {
            let mut mesh = LoopMesh::new();
            // Two quads sharing edge {1,2}; second quad folded 60° about it.
            let fold = 60.0_f32.to_radians();
            let v: Vec<VertexKey> = vec![
                mesh.create_vertex(Vec3::new(0.0, 0.0, 0.0)),
                mesh.create_vertex(Vec3::new(1.0, 0.0, 0.0)),
                mesh.create_vertex(Vec3::new(1.0, 1.0, 0.0)),
                mesh.create_vertex(Vec3::new(0.0, 1.0, 0.0)),
                mesh.create_vertex(Vec3::new(1.0 + fold.cos(), 0.0, fold.sin())),
                mesh.create_vertex(Vec3::new(1.0 + fold.cos(), 1.0, fold.sin())),
            ];
            let e01 = mesh.create_edge([v[0], v[1]], EdgeCreateFlags::NO_DUPLICATE);
            let e12 = mesh.create_edge([v[1], v[2]], EdgeCreateFlags::NO_DUPLICATE);
            let e23 = mesh.create_edge([v[2], v[3]], EdgeCreateFlags::NO_DUPLICATE);
            let e30 = mesh.create_edge([v[3], v[0]], EdgeCreateFlags::NO_DUPLICATE);
            mesh.face_create(&[v[0], v[1], v[2], v[3]], &[e01, e12, e23, e30], FaceCreateFlags::NONE);
            let e14 = mesh.create_edge([v[1], v[4]], EdgeCreateFlags::NO_DUPLICATE);
            let e45 = mesh.create_edge([v[4], v[5]], EdgeCreateFlags::NO_DUPLICATE);
            let e52 = mesh.create_edge([v[5], v[2]], EdgeCreateFlags::NO_DUPLICATE);
            mesh.face_create(&[v[1], v[4], v[5], v[2]], &[e14, e45, e52, e12], FaceCreateFlags::NONE);

            let seam = mesh.query_edge_exists(v[1], v[2]).unwrap();
            let params = DissolveLimitParams {
                angle_limit,
                use_dissolve_boundaries: false,
                delimit: 0,
            };
            mesh.dissolve_limit(&[seam], &[], &params);
            assert_eq!(
                mesh.totface(),
                1,
                "60° fold dissolves at angle_limit {angle_limit} (π clamps to π/2)"
            );
            mesh.validate_topology();
        }
    }

    #[test]
    fn angle_gate_rejects_60_fold_below_threshold() {
        // §5.1 / §8 — at angle_limit = π/4 the 60° fold's cost −0.5 is NOT
        // strictly below threshold −cos(π/4) ≈ −0.707, so the edge survives.
        let mut mesh = LoopMesh::new();
        let fold = 60.0_f32.to_radians();
        let v: Vec<VertexKey> = vec![
            mesh.create_vertex(Vec3::new(0.0, 0.0, 0.0)),
            mesh.create_vertex(Vec3::new(1.0, 0.0, 0.0)),
            mesh.create_vertex(Vec3::new(1.0, 1.0, 0.0)),
            mesh.create_vertex(Vec3::new(0.0, 1.0, 0.0)),
            mesh.create_vertex(Vec3::new(1.0 + fold.cos(), 0.0, fold.sin())),
            mesh.create_vertex(Vec3::new(1.0 + fold.cos(), 1.0, fold.sin())),
        ];
        let e01 = mesh.create_edge([v[0], v[1]], EdgeCreateFlags::NO_DUPLICATE);
        let e12 = mesh.create_edge([v[1], v[2]], EdgeCreateFlags::NO_DUPLICATE);
        let e23 = mesh.create_edge([v[2], v[3]], EdgeCreateFlags::NO_DUPLICATE);
        let e30 = mesh.create_edge([v[3], v[0]], EdgeCreateFlags::NO_DUPLICATE);
        mesh.face_create(&[v[0], v[1], v[2], v[3]], &[e01, e12, e23, e30], FaceCreateFlags::NONE);
        let e14 = mesh.create_edge([v[1], v[4]], EdgeCreateFlags::NO_DUPLICATE);
        let e45 = mesh.create_edge([v[4], v[5]], EdgeCreateFlags::NO_DUPLICATE);
        let e52 = mesh.create_edge([v[5], v[2]], EdgeCreateFlags::NO_DUPLICATE);
        mesh.face_create(&[v[1], v[4], v[5], v[2]], &[e14, e45, e52, e12], FaceCreateFlags::NONE);

        let seam = mesh.query_edge_exists(v[1], v[2]).unwrap();
        let params = DissolveLimitParams {
            angle_limit: std::f32::consts::FRAC_PI_4,
            use_dissolve_boundaries: false,
            delimit: 0,
        };
        mesh.dissolve_limit(&[seam], &[], &params);
        assert_eq!(mesh.totface(), 2, "60° fold survives below the π/4 gate");
        assert!(mesh.query_edge_exists(v[1], v[2]).is_some());
        mesh.validate_topology();
    }

    #[test]
    fn edge_then_vert_chain_collapses_strip() {
        // §5 → §6 — passing both the interior edges AND the soon-to-be-edge-pair
        // verts fully collapses the strip; passing the edges alone merges to one
        // face but leaves the rim verts. Pins that the vert pass takes its own
        // candidate set, not the edge pass's side effects.
        let mut mesh = LoopMesh::new();
        let (_verts, edges, interior_verts) = build_coplanar_strip(&mut mesh, 3);

        // Edge pass only: one merged face, the four rim verts survive.
        let _ = &interior_verts;
        {
            let params = DissolveLimitParams {
                angle_limit: FRAC_PI_2,
                use_dissolve_boundaries: false,
                delimit: 0,
            };
            mesh.dissolve_limit(&edges, &[], &params);
            assert_eq!(mesh.totface(), 1, "edge pass merges three quads to one");
            assert_eq!(mesh.totvert(), 8, "the four rim verts survive");
        }

        // Fresh mesh, both inputs: full collapse to a quad.
        let mut mesh = LoopMesh::new();
        let (_verts, edges, interior_verts) = build_coplanar_strip(&mut mesh, 3);
        let params = DissolveLimitParams {
            angle_limit: FRAC_PI_2,
            use_dissolve_boundaries: false,
            delimit: 0,
        };
        mesh.dissolve_limit(&edges, &interior_verts, &params);
        assert_eq!(mesh.totface(), 1);
        assert_eq!(mesh.totvert(), 4, "rim collapses to a quad");
        assert_eq!(mesh.totedge(), 4);
        let _ = interior_verts;
        mesh.validate_topology();
    }

    #[test]
    fn use_dissolve_boundaries_collapses_sharp_corners() {
        // §7 — the unconditional sweep collapses every still-edge-pair candidate
        // regardless of corner angle. A single quad's four corners are all
        // edge-pairs; with the flag they all collapse (reducing the quad away),
        // where the heap pass at a tight angle_limit would reject them.
        let mut mesh = LoopMesh::new();
        let v: Vec<VertexKey> = vec![
            mesh.create_vertex(Vec3::new(0.0, 0.0, 0.0)),
            mesh.create_vertex(Vec3::new(1.0, 0.0, 0.0)),
            mesh.create_vertex(Vec3::new(1.0, 1.0, 0.0)),
            mesh.create_vertex(Vec3::new(0.0, 1.0, 0.0)),
        ];
        let e0 = mesh.create_edge([v[0], v[1]], EdgeCreateFlags::NO_DUPLICATE);
        let e1 = mesh.create_edge([v[1], v[2]], EdgeCreateFlags::NO_DUPLICATE);
        let e2 = mesh.create_edge([v[2], v[3]], EdgeCreateFlags::NO_DUPLICATE);
        let e3 = mesh.create_edge([v[3], v[0]], EdgeCreateFlags::NO_DUPLICATE);
        mesh.face_create(&[v[0], v[1], v[2], v[3]], &[e0, e1, e2, e3], FaceCreateFlags::NONE);

        // false branch at a tight angle_limit: the 90° corners are rejected.
        {
            let mut probe = LoopMesh::new();
            let p: Vec<VertexKey> = (0..4)
                .map(|i| {
                    let pos = [
                        Vec3::new(0.0, 0.0, 0.0),
                        Vec3::new(1.0, 0.0, 0.0),
                        Vec3::new(1.0, 1.0, 0.0),
                        Vec3::new(0.0, 1.0, 0.0),
                    ][i];
                    probe.create_vertex(pos)
                })
                .collect();
            let pe0 = probe.create_edge([p[0], p[1]], EdgeCreateFlags::NO_DUPLICATE);
            let pe1 = probe.create_edge([p[1], p[2]], EdgeCreateFlags::NO_DUPLICATE);
            let pe2 = probe.create_edge([p[2], p[3]], EdgeCreateFlags::NO_DUPLICATE);
            let pe3 = probe.create_edge([p[3], p[0]], EdgeCreateFlags::NO_DUPLICATE);
            probe.face_create(&[p[0], p[1], p[2], p[3]], &[pe0, pe1, pe2, pe3], FaceCreateFlags::NONE);
            let params = DissolveLimitParams {
                angle_limit: std::f32::consts::FRAC_PI_4,
                use_dissolve_boundaries: false,
                delimit: 0,
            };
            probe.dissolve_limit(&[], &p, &params);
            assert_eq!(probe.totvert(), 4, "heap pass rejects the 90° corners");
        }

        // true branch: unconditional collapse strips the quad away entirely.
        let params = DissolveLimitParams {
            angle_limit: std::f32::consts::FRAC_PI_4,
            use_dissolve_boundaries: true,
            delimit: 0,
        };
        mesh.dissolve_limit(&[], &v, &params);
        assert!(
            mesh.totvert() < 4,
            "unconditional sweep collapses edge-pair corners regardless of angle"
        );
        mesh.validate_topology();
    }

    #[test]
    fn re_score_after_merge_z_fold() {
        // §5.2 step 4 — a Z-fold of three quads with two shallow fold seams. The
        // first merge changes the merged face's normal; the second seam must be
        // re-scored against the new normal and still pop. Both dissolve, leaving
        // one face. Folds shallow enough (π/6) that both seams pass at π/2.
        let mut mesh = LoopMesh::new();
        let a = std::f32::consts::PI / 6.0;
        // Three quad panels meeting at two seams, each turned by ±π/6.
        let b: Vec<VertexKey> = vec![
            mesh.create_vertex(Vec3::new(0.0, 0.0, 0.0)),
            mesh.create_vertex(Vec3::new(1.0, 0.0, 0.0)),
            mesh.create_vertex(Vec3::new(1.0 + a.cos(), 0.0, a.sin())),
            mesh.create_vertex(Vec3::new(1.0 + 2.0 * a.cos(), 0.0, 0.0)),
        ];
        let t: Vec<VertexKey> = vec![
            mesh.create_vertex(Vec3::new(0.0, 1.0, 0.0)),
            mesh.create_vertex(Vec3::new(1.0, 1.0, 0.0)),
            mesh.create_vertex(Vec3::new(1.0 + a.cos(), 1.0, a.sin())),
            mesh.create_vertex(Vec3::new(1.0 + 2.0 * a.cos(), 1.0, 0.0)),
        ];
        for i in 0..3 {
            let e_b = mesh.create_edge([b[i], b[i + 1]], EdgeCreateFlags::NO_DUPLICATE);
            let e_r = mesh.create_edge([b[i + 1], t[i + 1]], EdgeCreateFlags::NO_DUPLICATE);
            let e_t = mesh.create_edge([t[i + 1], t[i]], EdgeCreateFlags::NO_DUPLICATE);
            let e_l = mesh.create_edge([t[i], b[i]], EdgeCreateFlags::NO_DUPLICATE);
            mesh.face_create(
                &[b[i], b[i + 1], t[i + 1], t[i]],
                &[e_b, e_r, e_t, e_l],
                FaceCreateFlags::NONE,
            );
        }
        let seam1 = mesh.query_edge_exists(b[1], t[1]).unwrap();
        let seam2 = mesh.query_edge_exists(b[2], t[2]).unwrap();
        let params = DissolveLimitParams {
            angle_limit: FRAC_PI_2,
            use_dissolve_boundaries: false,
            delimit: 0,
        };
        mesh.dissolve_limit(&[seam1, seam2], &[], &params);
        assert_eq!(mesh.totface(), 1, "both shallow seams dissolve after re-score");
        mesh.validate_topology();
    }

    #[test]
    fn merged_face_keeps_first_loop_cd() {
        // §11 — the merged face keeps the per-face CD of the `e.l` face, the
        // same first-passed-loop rule dissolve_edges follows: on the standard
        // build order the second-built flanker of the seam wins.
        let mut mesh = LoopMesh::new();
        let v: Vec<VertexKey> = vec![
            mesh.create_vertex(Vec3::new(0.0, 0.0, 0.0)),
            mesh.create_vertex(Vec3::new(1.0, 0.0, 0.0)),
            mesh.create_vertex(Vec3::new(1.0, 1.0, 0.0)),
            mesh.create_vertex(Vec3::new(0.0, 1.0, 0.0)),
            mesh.create_vertex(Vec3::new(2.0, 0.0, 0.0)),
            mesh.create_vertex(Vec3::new(2.0, 1.0, 0.0)),
        ];
        let e01 = mesh.create_edge([v[0], v[1]], EdgeCreateFlags::NO_DUPLICATE);
        let e12 = mesh.create_edge([v[1], v[2]], EdgeCreateFlags::NO_DUPLICATE);
        let e23 = mesh.create_edge([v[2], v[3]], EdgeCreateFlags::NO_DUPLICATE);
        let e30 = mesh.create_edge([v[3], v[0]], EdgeCreateFlags::NO_DUPLICATE);
        let face_first = mesh.face_create(
            &[v[0], v[1], v[2], v[3]],
            &[e01, e12, e23, e30],
            FaceCreateFlags::NONE,
        );
        let e14 = mesh.create_edge([v[1], v[4]], EdgeCreateFlags::NO_DUPLICATE);
        let e45 = mesh.create_edge([v[4], v[5]], EdgeCreateFlags::NO_DUPLICATE);
        let e52 = mesh.create_edge([v[5], v[2]], EdgeCreateFlags::NO_DUPLICATE);
        let face_second = mesh.face_create(
            &[v[1], v[4], v[5], v[2]],
            &[e14, e45, e52, e12],
            FaceCreateFlags::NONE,
        );

        let color = mesh
            .register_face_attribute(ElementAttributeDesc::numeric(
                AttributeKey::custom("face_color"),
                Format::new(ComponentType::F32, 3),
            ))
            .unwrap();
        let first_color = [1.0_f32, 0.0, 0.0];
        let second_color = [0.0_f32, 1.0, 0.0];
        mesh.face_attribs_mut()
            .set_vec3(color, face_first.as_usize(), first_color);
        mesh.face_attribs_mut()
            .set_vec3(color, face_second.as_usize(), second_color);

        let seam = mesh.query_edge_exists(v[1], v[2]).unwrap();
        let params = DissolveLimitParams {
            angle_limit: FRAC_PI_2,
            use_dissolve_boundaries: false,
            delimit: 0,
        };
        mesh.dissolve_limit(&[seam], &[], &params);

        let merged = mesh.face_keys();
        assert_eq!(merged.len(), 1);
        let got = mesh
            .face_attribs()
            .get_vec3(color, merged[0].as_usize())
            .expect("merged face carries the CD layer");
        // The `e.l` survivor is the second-built flanker (the same rule as
        // dissolve_edges' §3.1 worked example).
        for axis in 0..3 {
            assert!(
                (got[axis] - second_color[axis]).abs() < 1e-6,
                "merged face must keep the second-built (e.l) face's CD: got {got:?}"
            );
        }
        mesh.validate_topology();
    }

    #[test]
    fn delimit_nonzero_declines() {
        // §3 deferral — a non-zero delimit is not implemented this cycle and
        // must decline cleanly without touching the mesh.
        let mut mesh = LoopMesh::new();
        let (_verts, edges, interior_verts) = build_coplanar_strip(&mut mesh, 2);
        let v0 = mesh.totvert();
        let e0 = mesh.totedge();
        let f0 = mesh.totface();
        let params = DissolveLimitParams {
            angle_limit: FRAC_PI_2,
            use_dissolve_boundaries: false,
            delimit: DELIMIT_NORMAL,
        };
        let result = mesh.dissolve_limit(&edges, &interior_verts, &params);
        assert_eq!(result, DissolveLimitResult::Unsupported);
        assert_eq!((mesh.totvert(), mesh.totedge(), mesh.totface()), (v0, e0, f0));
        mesh.validate_topology();
    }
}
