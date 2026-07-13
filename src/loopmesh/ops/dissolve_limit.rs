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

/// The subset of `delimit` bits this implementation honours. Every documented
/// bit (`NORMAL` | `MATERIAL` | `SEAM` | `SHARP` | `UV`) is implemented; a
/// `delimit` value with any *undocumented* bit set declines via
/// [`DissolveLimitResult::Unsupported`].
const DELIMIT_SUPPORTED: u32 =
    DELIMIT_NORMAL | DELIMIT_MATERIAL | DELIMIT_SEAM | DELIMIT_SHARP | DELIMIT_UV;

/// Parameters controlling [`LoopMesh::dissolve_limit`].
///
/// Mirrors the BMOP set-input shape described in
/// `radial-mesh-guides/operators/DISSOLVE_LIMIT.md` §1.
///
/// The core `delimit = 0` priority sweep — the §4 clamp, the §5 edge pass,
/// the §6 heap-driven vert pass (with the §6.3 degeneracy guard), and the §7
/// `use_dissolve_boundaries` linear sweep — is implemented, as is the full §3
/// `delimit`-flag filtering for every documented bit: the geometry-only
/// `NORMAL` / `UV` bits (including the §3.1 zero-UV-layer auto-clear) and the
/// semantic `MATERIAL` / `SEAM` / `SHARP` bits. A `delimit` value with an
/// *undocumented* bit set declines through [`DissolveLimitResult::Unsupported`]
/// rather than silently producing a wrong result.
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
    /// Hard candidate filter (§3). Every documented bit is honoured (any
    /// combination runs): `DELIMIT_NORMAL`, `DELIMIT_MATERIAL`, `DELIMIT_SEAM`,
    /// `DELIMIT_SHARP`, and `DELIMIT_UV`. A mask with an undocumented bit set
    /// declines via [`DissolveLimitResult::Unsupported`].
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

/// The operator's priority queue: an **indexed array binary min-heap** with the
/// three tie conventions of `DISSOLVE_LIMIT.md` §5.5.1, plus the §5.3
/// `heap_table[input_slot] -> node` indirection so a re-score is an in-place
/// update.
///
/// The conventions are the specification, not an implementation detail. Exactly
/// tied costs are this operator's *normal* regime (every coplanar face pair
/// scores −1.0 bit-exactly), the tied pop order is observable through the
/// per-face customdata donation chain, and §5.5.2/§5.5.3 establish that **no
/// comparator over a total order can express it** — a four-way tie pops
/// `3, 0, 1, 2`, which is not sorted under any fixed key. So the array layout
/// has to be reproduced directly:
///
/// 1. **Insertion promotes ties.** Sift-up stops only on a *strictly* cheaper
///    parent, so a newly inserted node that ties everything climbs to the root.
/// 2. **Extraction does not sink ties.** Sift-down descends only into a
///    *strictly* cheaper child, so the hoisted tail node stays at the root under
///    a full tie.
/// 3. **A re-score to an unchanged value is a no-op.** It must not touch the
///    array — which is also why a lazy-deletion heap (re-pushing a duplicate
///    entry on re-score) cannot be used here, whatever its comparator.
struct PriorityHeap {
    /// The heap array. Each element is an input slot; the ordering key is
    /// `cost[slot]`.
    tree: Vec<u32>,
    /// Per-input-slot cost, live for removed slots too so the §6.2-step-5
    /// reopening can ask "was this demoted to `COST_INVALID`?".
    cost: Vec<f32>,
    /// Per-input-slot position in `tree` — the §5.3 `heap_table`. `None` is the
    /// guide's nulled slot: the candidate is no longer in the heap.
    position: Vec<Option<u32>>,
}

impl PriorityHeap {
    fn with_capacity(len: usize) -> Self {
        Self {
            tree: Vec::with_capacity(len),
            cost: vec![COST_INVALID; len],
            position: vec![None; len],
        }
    }

    fn cost_at(&self, node: usize) -> f32 {
        self.cost[self.tree[node] as usize]
    }

    fn swap_nodes(&mut self, a: usize, b: usize) {
        self.tree.swap(a, b);
        self.position[self.tree[a] as usize] = Some(a as u32);
        self.position[self.tree[b] as usize] = Some(b as u32);
    }

    /// Convention 1 — climb while the parent is not *strictly* cheaper, so an
    /// exact tie swaps.
    fn sift_up(&mut self, mut node: usize) {
        while node > 0 {
            let parent = (node - 1) / 2;
            if self.cost_at(parent) < self.cost_at(node) {
                break;
            }
            self.swap_nodes(parent, node);
            node = parent;
        }
    }

    /// Convention 2 — descend only into a *strictly* cheaper child, so an exact
    /// tie stays put.
    fn sift_down(&mut self, mut node: usize) {
        loop {
            let left = node * 2 + 1;
            let right = left + 1;
            let mut cheapest = node;
            if left < self.tree.len() && self.cost_at(left) < self.cost_at(cheapest) {
                cheapest = left;
            }
            if right < self.tree.len() && self.cost_at(right) < self.cost_at(cheapest) {
                cheapest = right;
            }
            if cheapest == node {
                return;
            }
            self.swap_nodes(node, cheapest);
            node = cheapest;
        }
    }

    /// Seed input slot `i` at `cost` (appended at the tail, then sifted up).
    fn insert(&mut self, input_index: usize, cost: f32) {
        self.cost[input_index] = cost;
        let node = self.tree.len();
        self.tree.push(input_index as u32);
        self.position[input_index] = Some(node as u32);
        self.sift_up(node);
    }

    /// Re-score input slot `i` in place (the §5.2-step-4 / §6.2-step-4 update).
    /// An unchanged cost touches nothing (convention 3); a slot already out of
    /// the heap is ignored (its `heap_table` entry is null).
    fn update_cost(&mut self, input_index: usize, cost: f32) {
        let Some(node) = self.position[input_index] else {
            return;
        };
        let previous = self.cost[input_index];
        if cost == previous {
            return;
        }
        self.cost[input_index] = cost;
        if cost < previous {
            self.sift_up(node as usize);
        } else {
            self.sift_down(node as usize);
        }
    }

    /// Take input slot `i` out of the heap (its §5.3 `heap_table` slot nulled).
    /// The node is hoisted unconditionally to the root along its parent chain,
    /// then the root is extracted the way §5.5.1's convention 2 requires: the
    /// array's tail moves into the root slot and sifts down.
    fn remove(&mut self, input_index: usize) {
        let Some(node) = self.position[input_index] else {
            return;
        };
        let mut node = node as usize;
        while node > 0 {
            let parent = (node - 1) / 2;
            self.swap_nodes(parent, node);
            node = parent;
        }
        let tail = self.tree.pop().expect("the removed node is in the array");
        self.position[input_index] = None;
        if tail != input_index as u32 {
            self.tree[0] = tail;
            self.position[tail as usize] = Some(0);
            self.sift_down(0);
        }
    }

    /// The §5.2 / §6.2 step-1 gate: the root candidate, if its cost is strictly
    /// below `threshold`. The pass *peeks* — it never pops — so a candidate that
    /// fails its merge stays in the array at its own position and is demoted
    /// there via [`Self::update_cost`].
    fn peek_below(&self, threshold: f32) -> Option<usize> {
        let slot = *self.tree.first()? as usize;
        if self.cost[slot] < threshold {
            Some(slot)
        } else {
            None
        }
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
    /// §3 `delimit` filtering is honoured for every documented bit:
    /// `DELIMIT_NORMAL` (a non-contiguous edge — the two flanking faces wind
    /// the same direction along the shared edge), `DELIMIT_MATERIAL` (the two
    /// flanking faces have different material indices), `DELIMIT_SEAM` (the
    /// edge is seam-flagged), `DELIMIT_SHARP` (the edge is sharp / not smooth),
    /// and `DELIMIT_UV` (the edge is a UV-island boundary — its per-loop UVs
    /// disagree across some Float2 loop layer), including the §3.1
    /// zero-UV-layer auto-clear. A delimited edge takes the `COST_INVALID`
    /// sentinel and is never popped (§3.2); a vert touching a delimited
    /// manifold edge takes the un-modulated corner-angle cost (§3.3).
    ///
    /// Returns the §1 `region.out` analogue: the surviving merge anchors. A
    /// `delimit` value with an undocumented bit set declines via
    /// [`DissolveLimitResult::Unsupported`].
    pub fn dissolve_limit(
        &mut self,
        edges: &[EdgeKey],
        verts: &[VertexKey],
        params: &DissolveLimitParams,
    ) -> DissolveLimitResult {
        // Every documented delimiter bit is implemented; an undocumented bit
        // declines rather than silently mis-filtering.
        if params.delimit & !DELIMIT_SUPPORTED != 0 {
            return DissolveLimitResult::Unsupported;
        }

        // §3.1 — the UV bit auto-clears when the mesh carries no UV (Float2
        // loop) layers: with no UV data to delimit against the bit is inert,
        // and clearing it skips the per-edge loop-CD test entirely.
        let mut delimit = params.delimit;
        if delimit & DELIMIT_UV != 0 && self.dissolve_limit_uv_layer_count() == 0 {
            delimit &= !DELIMIT_UV;
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

        let region_out = self.dissolve_limit_edge_pass(edges, edge_threshold, delimit);

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
            self.dissolve_limit_vert_pass(&vert_input, angle_limit, delimit);
        }

        DissolveLimitResult::Ran { region_out }
    }

    /// §5 — the edge pass. Returns the §1 `region.out` surviving merge anchors.
    fn dissolve_limit_edge_pass(
        &mut self,
        edges: &[EdgeKey],
        edge_threshold: f32,
        delimit: u32,
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
            let cost = self.dissolve_limit_edge_cost(edge, delimit);
            heap.insert(i, cost);
        }

        // §1 / §5.2 step 4 — merged faces are tagged so the tail FACE_NEW-style
        // sweep can collect the surviving anchors (chained merges re-tag the
        // survivor, so an absorbed intermediate anchor drops out naturally).
        let region_tag = self.push_face_op_flag_layer();
        const REGION_BIT: u8 = 0;

        while let Some(i) = heap.peek_below(edge_threshold) {
            let edge = edges[i];
            if self.edges.try_get(edge).is_none() {
                heap.remove(i);
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
                    heap.remove(i);
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
                            let cost = self.dissolve_limit_edge_cost(perimeter_edge, delimit);
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
    fn dissolve_limit_vert_pass(
        &mut self,
        vert_input: &[Option<VertexKey>],
        angle_limit: f32,
        delimit: u32,
    ) {
        let mut index_of: HashMap<VertexKey, usize> = HashMap::with_capacity(vert_input.len());
        let mut heap = PriorityHeap::with_capacity(vert_input.len());
        for (i, slot) in vert_input.iter().enumerate() {
            let Some(v) = *slot else { continue };
            if self.vertices.try_get(v).is_none() {
                continue;
            }
            index_of.entry(v).or_insert(i);
            let cost = self.dissolve_limit_vert_cost(v, delimit);
            heap.insert(i, cost);
        }

        while let Some(i) = heap.peek_below(angle_limit) {
            let Some(v) = vert_input[i] else {
                heap.remove(i);
                continue;
            };
            if self.vertices.try_get(v).is_none() {
                heap.remove(i);
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
                    heap.remove(i);
                    index_of.remove(&v);

                    // §6.2 step 4 — re-score the surviving edge's endpoints.
                    let endpoints = self.edges.get(e_new).vertices;
                    for v_other in endpoints {
                        if let Some(&j) = index_of.get(&v_other) {
                            let cost = self.dissolve_limit_vert_cost(v_other, delimit);
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
                                    let cost = self.dissolve_limit_vert_cost(neighbour, delimit);
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

    /// §5.1 — per-edge dissolve cost. `COST_INVALID` for a non-manifold edge,
    /// or (§3.2) when `delimit != 0` and the edge crosses a set delimiter;
    /// otherwise the signed negated dot of the two face normals (the sign flip
    /// makes "coplanar / no fold" the lowest cost for both contiguous and
    /// non-contiguous edges).
    fn dissolve_limit_edge_cost(&self, edge: EdgeKey, delimit: u32) -> f32 {
        let Some((face_a, face_b)) = self.dissolve_limit_edge_face_pair(edge) else {
            return COST_INVALID;
        };
        if delimit != 0 && self.dissolve_limit_edge_is_delimiter(edge, delimit) {
            return COST_INVALID;
        }
        let normal_a = self.faces.get(face_a).normal;
        let normal_b = self.faces.get(face_b).normal;
        let dot = normal_a.dot(normal_b);
        if self.dissolve_limit_edge_is_contiguous(edge) {
            -dot
        } else {
            dot
        }
    }

    /// §6.1 / §3.3 — per-vert dissolve cost. The corner angle, modulated by the
    /// incident-edge dihedral when the canonical edge is manifold *and* the
    /// vert is not a delimiter (the multiplicative collinear-and-coplanar
    /// form). A non-edge-pair vert is `COST_INVALID` (it has no two-edge corner
    /// to collapse). A delimiter vert — any incident manifold edge crosses a
    /// set delimiter — takes the un-modulated corner angle (§3.3), the more
    /// permissive form that still lets an "easy" collinear corner collapse on a
    /// delimiter line.
    fn dissolve_limit_vert_cost(&self, v: VertexKey, delimit: u32) -> f32 {
        let Some(corner_angle) = self.dissolve_limit_vert_corner_angle(v) else {
            return COST_INVALID;
        };
        let primary_edge = self
            .vertices
            .get(v)
            .disk_loop_start
            .expect("edge-pair vert has a disk-head edge");
        let is_delimiter = delimit != 0 && self.dissolve_limit_vert_is_delimiter(v, delimit);
        if !is_delimiter {
            if let Some(edge_angle) = self.dissolve_limit_edge_dihedral(primary_edge) {
                return corner_angle * edge_angle / std::f32::consts::FRAC_PI_2;
            }
        }
        corner_angle
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
            if self.dissolve_limit_face_collapse_is_degenerate(face, v) {
                return true;
            }
        }
        false
    }

    /// §6.3 step 3 — the per-face 2D degeneracy test. The collapsed corner is
    /// `v`; in this face's loop cycle its ear-adjacent corners are `prev` and
    /// `next` (the loop-neighbours of `v`), and the weld replaces the two edges
    /// `prev → v` and `v → next` with the single edge `prev → next`. The cut-off
    /// **ear triangle** is `(prev, v, next)`. The test projects the face into
    /// its dominant axis plane and rejects the collapse when:
    ///
    /// - **Adjacent-corner non-flip.** For each ear-adjacent corner (`prev` /
    ///   `next`) that is *not* itself an edge-pair vert, the corner must keep
    ///   its orientation across the weld: the cross-product sign of its other
    ///   arm against the *old* corner edge (toward `v`) must match its sign
    ///   against the *new* weld edge (`prev → next`). A sign change folds that
    ///   neighbouring corner inside-out.
    /// - **Point-in-triangle.** Any other perimeter vert lying inside the ear
    ///   triangle `(prev, v, next)` would be trapped by the weld → a
    ///   self-intersection.
    fn dissolve_limit_face_collapse_is_degenerate(&self, face: FaceKey, v: VertexKey) -> bool {
        let normal = self.faces.get(face).normal;
        let axis = dominant_axis(normal);

        // Locate `v`'s loop within this face so the ear-adjacent corners are the
        // loop-neighbours in the face's own winding order, not the disk order.
        let Some(loop_v) = self.dissolve_limit_face_loop_at_vertex(face, v) else {
            return false;
        };
        let loop_prev = self.loop_prev(loop_v);
        let loop_next = self.loop_next(loop_v);
        let prev_vert = self.loops.get(loop_prev).vertex;
        let next_vert = self.loops.get(loop_next).vertex;

        let pivot = project_to_axis(self.vertices.get(v).position, axis);
        let prev = project_to_axis(self.vertices.get(prev_vert).position, axis);
        let next = project_to_axis(self.vertices.get(next_vert).position, axis);

        // Adjacent-corner non-flip for the `prev` corner. Its other arm runs to
        // the vert before `prev`; the old corner edge points at `v`, the new
        // weld edge points at `next`.
        if !self.dissolve_limit_vert_is_edge_pair(prev_vert) {
            let prev_prev = project_to_axis(
                self.vertices
                    .get(self.loops.get(self.loop_prev(loop_prev)).vertex)
                    .position,
                axis,
            );
            let old_side = signed_area(prev, prev_prev, pivot);
            let new_side = signed_area(prev, prev_prev, next);
            if old_side * new_side < 0.0 {
                return true;
            }
        }

        // Adjacent-corner non-flip for the `next` corner. Its other arm runs to
        // the vert after `next`; the old corner edge points at `v`, the new weld
        // edge points at `prev`.
        if !self.dissolve_limit_vert_is_edge_pair(next_vert) {
            let next_next = project_to_axis(
                self.vertices
                    .get(self.loops.get(self.loop_next(loop_next)).vertex)
                    .position,
                axis,
            );
            let old_side = signed_area(next, next_next, pivot);
            let new_side = signed_area(next, next_next, prev);
            if old_side * new_side < 0.0 {
                return true;
            }
        }

        // Self-intersection: no other corner may sit inside the ear triangle.
        let mut loops = self.face_loops(face);
        while let Some(loop_key) = loops.next(self) {
            let corner_vert = self.loops.get(loop_key).vertex;
            if corner_vert == v || corner_vert == prev_vert || corner_vert == next_vert {
                continue;
            }
            let point = project_to_axis(self.vertices.get(corner_vert).position, axis);
            if point_in_triangle(point, prev, pivot, next) {
                return true;
            }
        }
        false
    }

    /// Find the loop in `face`'s cycle whose vertex is `v`, or `None` if `v` is
    /// not a corner of `face`.
    fn dissolve_limit_face_loop_at_vertex(&self, face: FaceKey, v: VertexKey) -> Option<LoopKey> {
        let mut loops = self.face_loops(face);
        while let Some(loop_key) = loops.next(self) {
            if self.loops.get(loop_key).vertex == v {
                return Some(loop_key);
            }
        }
        None
    }

    /// §6.3 — an edge-pair vert has exactly two incident edges. Used by the
    /// adjacent-corner non-flip check: an ear-adjacent corner that is itself an
    /// edge-pair vert is exempt from the flip test (it has its own collapse
    /// candidacy and no third arm to fold).
    fn dissolve_limit_vert_is_edge_pair(&self, v: VertexKey) -> bool {
        self.dissolve_limit_vert_edge_count(v) == 2
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

    /// §3 — the per-edge delimiter predicate. Returns `true` when `edge`
    /// crosses any delimiter set in `delimit`. Only the geometry-only bits are
    /// implemented:
    ///
    /// - `DELIMIT_NORMAL` — the edge is **not contiguous** (§3 row / §5.1): its
    ///   two flanking faces' loops run the *same* direction along the shared
    ///   edge rather than opposite directions, i.e. one face is wound
    ///   inside-out relative to the other.
    /// - `DELIMIT_MATERIAL` — the two faces flanking the edge have different
    ///   material indices (§3 row / §8 worked example): `mat(f_a) != mat(f_b)`.
    /// - `DELIMIT_SEAM` — the edge carries the seam flag (§3 row).
    /// - `DELIMIT_SHARP` — the edge is *not* smooth, i.e. sharp (§3 row). The
    ///   smooth state is the positive-presence one, so "sharp" is its negation;
    ///   loopmesh stores this directly as the per-edge sharp flag.
    /// - `DELIMIT_UV` — the edge is a UV-island boundary: it is not loop-CD
    ///   contiguous across every Float2 loop layer (§3 row, §3.2). The bit is
    ///   only ever tested when at least one such layer exists (§3.1 auto-clear
    ///   strips it otherwise).
    ///
    /// A non-manifold / boundary edge has no flanking pair, so the
    /// face-relative bits (`NORMAL` / `MATERIAL` / `UV`) cannot fire on it; the
    /// non-manifold case already yields `COST_INVALID` independently (§5.1). The
    /// per-edge bits (`SEAM` / `SHARP`) read the edge's own flag and so apply
    /// regardless of the flanking pair.
    fn dissolve_limit_edge_is_delimiter(&self, edge: EdgeKey, delimit: u32) -> bool {
        if delimit & DELIMIT_NORMAL != 0 && !self.dissolve_limit_edge_is_contiguous(edge) {
            return true;
        }
        if delimit & DELIMIT_MATERIAL != 0 {
            if let Some((face_a, face_b)) = self.dissolve_limit_edge_face_pair(edge) {
                if self.face_material_index(face_a) != self.face_material_index(face_b) {
                    return true;
                }
            }
        }
        if delimit & DELIMIT_SEAM != 0 && self.edge_seam(edge) {
            return true;
        }
        if delimit & DELIMIT_SHARP != 0 && self.edge_sharp(edge) {
            return true;
        }
        if delimit & DELIMIT_UV != 0 && !self.dissolve_limit_edge_is_uv_contiguous(edge) {
            return true;
        }
        false
    }

    /// §3.3 — a vert is a delimiter vert when any incident manifold edge is
    /// itself a delimiter; such a vert takes the un-modulated corner-angle
    /// cost. A wire / non-manifold incident edge never carries a delimiter
    /// classification (the delimiter predicate requires a flanking face pair).
    fn dissolve_limit_vert_is_delimiter(&self, v: VertexKey, delimit: u32) -> bool {
        let mut disk = self.disk_edges(v);
        while let Some(edge) = disk.next(self) {
            if self.dissolve_limit_edge_face_pair(edge).is_some()
                && self.dissolve_limit_edge_is_delimiter(edge, delimit)
            {
                return true;
            }
        }
        false
    }

    /// Count the mesh's Float2 (UV) loop layers. The §3 `DELIMIT_UV` test runs
    /// against every such layer; a count of zero triggers the §3.1 auto-clear.
    fn dissolve_limit_uv_layer_count(&self) -> usize {
        let attribs = self.loop_attribs();
        attribs
            .iter_descriptors()
            .filter(|(key, _)| {
                matches!(
                    attribs.storage_of(*key),
                    Some(crate::loopmesh::AttributeStorage::Vec2(_))
                )
            })
            .count()
    }

    /// §3 `DELIMIT_UV` — `true` when `edge` is loop-CD contiguous across every
    /// Float2 (UV) loop layer: the per-loop UVs agree at *both* endpoints of
    /// the edge across its two flanking faces. A disagreement on any layer at
    /// either endpoint makes the edge a UV-island boundary (returns `false`).
    ///
    /// For a manifold edge with radial loops `la` (on face A) and `lb` (on
    /// face B), `la` runs from `la.vert` to `loop_next(la).vert` and likewise
    /// for `lb`; the loop↔edge invariant (`RADIAL_EDGE_STRUCTURE.md` §5.1)
    /// guarantees those two endpoint vertices are the edge's two ends. The four
    /// corners at the edge pair up by shared vertex:
    ///   - at `la.vert`:        `la`            vs `loop_next(lb)`
    ///   - at `loop_next(la).vert`: `loop_next(la)` vs `lb`
    /// (the pairing holds whether or not the edge is contiguous, since
    /// `loop_next` lands on the matching endpoint on the opposite face). Each
    /// pair's UVs are compared for exact per-component equality.
    ///
    /// NOTE — the guide (`DISSOLVE_LIMIT.md` §3, the `UV` row) defines this
    /// per-loop UV-contiguity predicate but does not specify a comparison
    /// *tolerance*; this implementation uses exact per-component equality (the
    /// literal reading of "the per-loop UVs ... don't match"). A near-equal
    /// island seam may therefore classify differently from a tolerant compare.
    /// See the cycle report's guide-gap note.
    fn dissolve_limit_edge_is_uv_contiguous(&self, edge: EdgeKey) -> bool {
        let mut radial = self.radial_loops(edge);
        let Some(loop_a) = radial.next(self) else {
            return true;
        };
        let Some(loop_b) = radial.next(self) else {
            return true;
        };
        if radial.next(self).is_some() {
            return true;
        }

        let a_start = loop_a;
        let a_end = self.loop_next(loop_a);
        let b_start = loop_b;
        let b_end = self.loop_next(loop_b);

        let vertex_a = self.loops.get(a_start).vertex;
        // Pair each face's corner by the shared endpoint vertex. `b_start`'s
        // vertex is one endpoint; whichever of {b_start, b_end} shares
        // `vertex_a` is the corner to compare against `a_start`.
        let (b_at_a_start, b_at_a_end) = if self.loops.get(b_start).vertex == vertex_a {
            (b_start, b_end)
        } else {
            (b_end, b_start)
        };

        let uv_layers: smallvec::SmallVec<[crate::loopmesh::ElementAttributeKey; 4]> = {
            let attribs = self.loop_attribs();
            attribs
                .iter_descriptors()
                .filter(|(key, _)| {
                    matches!(
                        attribs.storage_of(*key),
                        Some(crate::loopmesh::AttributeStorage::Vec2(_))
                    )
                })
                .map(|(key, _)| key)
                .collect()
        };
        for layer in uv_layers {
            if self.dissolve_limit_loop_uv(layer, a_start)
                != self.dissolve_limit_loop_uv(layer, b_at_a_start)
                || self.dissolve_limit_loop_uv(layer, a_end)
                    != self.dissolve_limit_loop_uv(layer, b_at_a_end)
            {
                return false;
            }
        }
        true
    }

    /// Read a loop's UV from a Float2 layer, defaulting to the zero corner if
    /// the slot is unset.
    fn dissolve_limit_loop_uv(
        &self,
        layer: crate::loopmesh::ElementAttributeKey,
        loop_key: LoopKey,
    ) -> [f32; 2] {
        self.loop_attribs()
            .get_vec2(layer, loop_key.as_usize())
            .unwrap_or([0.0; 2])
    }

    /// The two distinct faces flanking a manifold `edge`, ordered `e.l` first
    /// so the merge keeps the §11 customdata winner. `None` for wire /
    /// boundary / self-radial / non-manifold edges.
    ///
    /// Mirrors the dissolve-edges §3.1 survivor resolution: the `e.l` survivor
    /// is the edge's radial **entry**, which the §6.2 radial traversal emits
    /// first.
    fn dissolve_limit_edge_face_pair(&self, edge: EdgeKey) -> Option<(FaceKey, FaceKey)> {
        let mut radial = self.radial_loops(edge);
        let entry = radial.next(self)?;
        let other = radial.next(self)?;
        if radial.next(self).is_some() {
            return None;
        }
        let entry_face = self.loops.get(entry).face;
        let other_face = self.loops.get(other).face;
        if entry_face == other_face {
            return None;
        }
        Some((entry_face, other_face))
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
            mesh.face_create(
                &[b0, b1, t1, t0],
                &[e_b, e_r, e_t, e_l],
                FaceCreateFlags::NONE,
            );
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

        assert_eq!((mesh.vertex_count(), mesh.face_count()), (10, 4));

        let params = DissolveLimitParams {
            angle_limit: std::f32::consts::FRAC_PI_4,
            use_dissolve_boundaries: false,
            delimit: 0,
        };
        let result = mesh.dissolve_limit(&edges, &interior_verts, &params);
        assert!(matches!(result, DissolveLimitResult::Ran { .. }));

        assert_eq!(mesh.face_count(), 1, "all four quads merge into one");
        assert_eq!(mesh.vertex_count(), 4, "the rim collapses to a single quad");
        assert_eq!(mesh.edge_count(), 4);
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
        let v0 = mesh.vertex_count();
        let e0 = mesh.edge_count();
        let f0 = mesh.face_count();

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
        assert_eq!(
            (mesh.vertex_count(), mesh.edge_count(), mesh.face_count()),
            (v0, e0, f0)
        );
        mesh.validate_topology();
    }

    /// §5.5.4 — the donor fingerprint. On a flat `n`-quad strip every candidate
    /// edge scores −1.0 bit-exactly, so the whole heap is tied and the surviving
    /// face's per-face customdata reads the tied pop order back out: each merge
    /// donates from the popped edge's radial entry, and the entry-reseat rule
    /// carries the running union forward as its own donor.
    ///
    /// The table is the guide's oracle-confirmed one. Reverse-input-index would
    /// answer `f_{n−1}` at every row — it agrees only up to a three-way tie
    /// (§5.5.2's trap), and `n = 5` (a four-way tie) is the first row that
    /// separates them. Ascending-index would answer `f_1` at every row, which
    /// `n = 3` separates. Only the whole family pins the heap.
    #[test]
    fn equal_cost_pop_order_matches_the_heap_donor_fingerprint() {
        // (quads `n`, surviving donor face index) — the tie size is `m = n − 1`.
        const DONOR_FINGERPRINT: [(usize, usize); 8] = [
            (2, 1),
            (3, 2),
            (4, 3),
            (5, 1),
            (6, 1),
            (7, 6),
            (8, 5),
            (9, 1),
        ];

        for (quads, expected_donor) in DONOR_FINGERPRINT {
            let mut mesh = LoopMesh::new();
            let (_verts, edges, _interior_verts) = build_coplanar_strip(&mut mesh, quads);
            let tag = mesh
                .register_face_attribute(ElementAttributeDesc::numeric(
                    AttributeKey::custom("face_tag"),
                    AttrType::Vec3,
                ))
                .expect("face tag layer registers");
            for face in mesh.face_keys() {
                let index = strip_quad_index(&mesh, face);
                mesh.face_attribs_mut()
                    .set_vec3(tag, face.as_usize(), [index as f32, 0.0, 0.0]);
            }

            let params = DissolveLimitParams {
                angle_limit: FRAC_PI_2,
                use_dissolve_boundaries: false,
                delimit: 0,
            };
            // Empty vert set: the edge pass alone decides the donation chain.
            mesh.dissolve_limit(&edges, &[], &params);

            assert_eq!(mesh.face_count(), 1, "n = {quads}: the strip merges to one");
            let survivor = mesh.face_keys()[0];
            let donor = mesh
                .face_attribs()
                .get_vec3(tag, survivor.as_usize())
                .expect("survivor carries a tag")[0]
                .round() as usize;
            assert_eq!(
                donor,
                expected_donor,
                "n = {quads} (m = {}): the surviving customdata must name f_{expected_donor}",
                quads - 1
            );
            mesh.validate_topology();
        }
    }

    /// The index of a `build_coplanar_strip` quad, read off its centroid: quad
    /// `i` spans x ∈ [i, i+1].
    fn strip_quad_index(mesh: &LoopMesh, face: FaceKey) -> usize {
        let verts = mesh.face_vert_keys(face);
        let sum: f32 = verts
            .iter()
            .map(|&v| mesh.vertex_position(v).x)
            .sum::<f32>();
        (sum / verts.len() as f32).floor() as usize
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
        assert_eq!(
            mesh.face_count(),
            2,
            "two flat quads merge; the folded one stays"
        );
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
            mesh.face_create(
                &[v[0], v[1], v[2], v[3]],
                &[e01, e12, e23, e30],
                FaceCreateFlags::NONE,
            );
            let e14 = mesh.create_edge([v[1], v[4]], EdgeCreateFlags::NO_DUPLICATE);
            let e45 = mesh.create_edge([v[4], v[5]], EdgeCreateFlags::NO_DUPLICATE);
            let e52 = mesh.create_edge([v[5], v[2]], EdgeCreateFlags::NO_DUPLICATE);
            mesh.face_create(
                &[v[1], v[4], v[5], v[2]],
                &[e14, e45, e52, e12],
                FaceCreateFlags::NONE,
            );

            let seam = mesh.query_edge_exists(v[1], v[2]).unwrap();
            let params = DissolveLimitParams {
                angle_limit,
                use_dissolve_boundaries: false,
                delimit: 0,
            };
            mesh.dissolve_limit(&[seam], &[], &params);
            assert_eq!(
                mesh.face_count(),
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
        mesh.face_create(
            &[v[0], v[1], v[2], v[3]],
            &[e01, e12, e23, e30],
            FaceCreateFlags::NONE,
        );
        let e14 = mesh.create_edge([v[1], v[4]], EdgeCreateFlags::NO_DUPLICATE);
        let e45 = mesh.create_edge([v[4], v[5]], EdgeCreateFlags::NO_DUPLICATE);
        let e52 = mesh.create_edge([v[5], v[2]], EdgeCreateFlags::NO_DUPLICATE);
        mesh.face_create(
            &[v[1], v[4], v[5], v[2]],
            &[e14, e45, e52, e12],
            FaceCreateFlags::NONE,
        );

        let seam = mesh.query_edge_exists(v[1], v[2]).unwrap();
        let params = DissolveLimitParams {
            angle_limit: std::f32::consts::FRAC_PI_4,
            use_dissolve_boundaries: false,
            delimit: 0,
        };
        mesh.dissolve_limit(&[seam], &[], &params);
        assert_eq!(mesh.face_count(), 2, "60° fold survives below the π/4 gate");
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
            assert_eq!(mesh.face_count(), 1, "edge pass merges three quads to one");
            assert_eq!(mesh.vertex_count(), 8, "the four rim verts survive");
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
        assert_eq!(mesh.face_count(), 1);
        assert_eq!(mesh.vertex_count(), 4, "rim collapses to a quad");
        assert_eq!(mesh.edge_count(), 4);
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
        mesh.face_create(
            &[v[0], v[1], v[2], v[3]],
            &[e0, e1, e2, e3],
            FaceCreateFlags::NONE,
        );

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
            probe.face_create(
                &[p[0], p[1], p[2], p[3]],
                &[pe0, pe1, pe2, pe3],
                FaceCreateFlags::NONE,
            );
            let params = DissolveLimitParams {
                angle_limit: std::f32::consts::FRAC_PI_4,
                use_dissolve_boundaries: false,
                delimit: 0,
            };
            probe.dissolve_limit(&[], &p, &params);
            assert_eq!(probe.vertex_count(), 4, "heap pass rejects the 90° corners");
        }

        // true branch: unconditional collapse strips the quad away entirely.
        let params = DissolveLimitParams {
            angle_limit: std::f32::consts::FRAC_PI_4,
            use_dissolve_boundaries: true,
            delimit: 0,
        };
        mesh.dissolve_limit(&[], &v, &params);
        assert!(
            mesh.vertex_count() < 4,
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
        assert_eq!(
            mesh.face_count(),
            1,
            "both shallow seams dissolve after re-score"
        );
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
                AttrType::Vec3,
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
    fn delimit_undocumented_bit_declines() {
        // §3 — every documented bit (NORMAL / MATERIAL / SEAM / SHARP / UV) is
        // implemented; only an *undocumented* bit declines cleanly without
        // touching the mesh.
        let mut mesh = LoopMesh::new();
        let (_verts, edges, interior_verts) = build_coplanar_strip(&mut mesh, 2);
        let v0 = mesh.vertex_count();
        let e0 = mesh.edge_count();
        let f0 = mesh.face_count();
        let undocumented = 1u32 << 5;
        let params = DissolveLimitParams {
            angle_limit: FRAC_PI_2,
            use_dissolve_boundaries: false,
            delimit: undocumented,
        };
        let result = mesh.dissolve_limit(&edges, &interior_verts, &params);
        assert_eq!(result, DissolveLimitResult::Unsupported);
        assert_eq!(
            (mesh.vertex_count(), mesh.edge_count(), mesh.face_count()),
            (v0, e0, f0)
        );
        mesh.validate_topology();
    }

    /// Two coplanar contiguous quads sharing the single interior edge of a
    /// 2-quad strip (the `b[1]–t[1]` seam). Returns `(mesh, shared_edge,
    /// face_a, face_b)`. The seam is contiguous and coplanar, so at
    /// `delimit = 0` and `angle_limit = π/2` the edge pass merges the pair into
    /// one face — the baseline every semantic-bit test contrasts against.
    fn build_coplanar_pair() -> (LoopMesh, EdgeKey, FaceKey, FaceKey) {
        let mut mesh = LoopMesh::new();
        let (_verts, edges, _interior_verts) = build_coplanar_strip(&mut mesh, 2);
        let shared = edges[0];
        let faces = mesh.face_keys();
        (mesh, shared, faces[0], faces[1])
    }

    #[test]
    fn delimit_material_blocks_material_boundary_edge() {
        // §3 MATERIAL row / §8 worked example — a coplanar edge between two
        // faces of *different* material indices is delimited and survives with
        // DELIMIT_MATERIAL set; at delimit = 0 the same edge dissolves.

        // delimit = 0: matching-or-ignored materials ⇒ the edge dissolves.
        let (mut mesh, shared, face_a, face_b) = build_coplanar_pair();
        mesh.set_face_material_index(face_a, 0);
        mesh.set_face_material_index(face_b, 1);
        let params = DissolveLimitParams {
            angle_limit: FRAC_PI_2,
            use_dissolve_boundaries: false,
            delimit: 0,
        };
        mesh.dissolve_limit(&[shared], &[], &params);
        assert_eq!(
            mesh.face_count(),
            1,
            "without MATERIAL the boundary edge dissolves"
        );
        mesh.validate_topology();

        // DELIMIT_MATERIAL with mismatched indices: the boundary edge survives.
        let (mut mesh, shared, face_a, face_b) = build_coplanar_pair();
        mesh.set_face_material_index(face_a, 0);
        mesh.set_face_material_index(face_b, 1);
        let params = DissolveLimitParams {
            angle_limit: FRAC_PI_2,
            use_dissolve_boundaries: false,
            delimit: DELIMIT_MATERIAL,
        };
        assert!(matches!(
            mesh.dissolve_limit(&[shared], &[], &params),
            DissolveLimitResult::Ran { region_out } if region_out.is_empty()
        ));
        assert_eq!(
            mesh.face_count(),
            2,
            "DELIMIT_MATERIAL preserves the material boundary"
        );
        assert!(mesh.edges.try_get(shared).is_some());
        mesh.validate_topology();

        // DELIMIT_MATERIAL with matching indices: not a boundary, dissolves.
        let (mut mesh, shared, face_a, face_b) = build_coplanar_pair();
        mesh.set_face_material_index(face_a, 2);
        mesh.set_face_material_index(face_b, 2);
        let params = DissolveLimitParams {
            angle_limit: FRAC_PI_2,
            use_dissolve_boundaries: false,
            delimit: DELIMIT_MATERIAL,
        };
        mesh.dissolve_limit(&[shared], &[], &params);
        assert_eq!(
            mesh.face_count(),
            1,
            "matching materials are not a boundary; edge dissolves"
        );
        mesh.validate_topology();
    }

    #[test]
    fn delimit_seam_blocks_seam_flagged_edge() {
        // §3 SEAM row — a seam-flagged edge is delimited and survives with
        // DELIMIT_SEAM set; without the bit the seam flag is inert and the
        // coplanar edge dissolves.

        // delimit = 0: the seam flag is ignored, the edge dissolves.
        let (mut mesh, shared, _a, _b) = build_coplanar_pair();
        mesh.set_edge_seam(shared, true);
        let params = DissolveLimitParams {
            angle_limit: FRAC_PI_2,
            use_dissolve_boundaries: false,
            delimit: 0,
        };
        mesh.dissolve_limit(&[shared], &[], &params);
        assert_eq!(
            mesh.face_count(),
            1,
            "without SEAM the flag is inert; edge dissolves"
        );
        mesh.validate_topology();

        // DELIMIT_SEAM with the flag set: the edge survives.
        let (mut mesh, shared, _a, _b) = build_coplanar_pair();
        mesh.set_edge_seam(shared, true);
        let params = DissolveLimitParams {
            angle_limit: FRAC_PI_2,
            use_dissolve_boundaries: false,
            delimit: DELIMIT_SEAM,
        };
        assert!(matches!(
            mesh.dissolve_limit(&[shared], &[], &params),
            DissolveLimitResult::Ran { region_out } if region_out.is_empty()
        ));
        assert_eq!(
            mesh.face_count(),
            2,
            "DELIMIT_SEAM preserves the seam-flagged edge"
        );
        assert!(mesh.edges.try_get(shared).is_some());
        mesh.validate_topology();

        // DELIMIT_SEAM with no flag set: not a delimiter, dissolves.
        let (mut mesh, shared, _a, _b) = build_coplanar_pair();
        let params = DissolveLimitParams {
            angle_limit: FRAC_PI_2,
            use_dissolve_boundaries: false,
            delimit: DELIMIT_SEAM,
        };
        mesh.dissolve_limit(&[shared], &[], &params);
        assert_eq!(
            mesh.face_count(),
            1,
            "an un-flagged edge is not a seam; it dissolves"
        );
        mesh.validate_topology();
    }

    #[test]
    fn delimit_sharp_blocks_sharp_flagged_edge() {
        // §3 SHARP row — sharp = non-smooth. A sharp-flagged edge is delimited
        // and survives with DELIMIT_SHARP set; without the bit the sharp flag
        // is inert and the coplanar edge dissolves.

        // delimit = 0: the sharp flag is ignored, the edge dissolves.
        let (mut mesh, shared, _a, _b) = build_coplanar_pair();
        mesh.set_edge_sharp(shared, true);
        let params = DissolveLimitParams {
            angle_limit: FRAC_PI_2,
            use_dissolve_boundaries: false,
            delimit: 0,
        };
        mesh.dissolve_limit(&[shared], &[], &params);
        assert_eq!(
            mesh.face_count(),
            1,
            "without SHARP the flag is inert; edge dissolves"
        );
        mesh.validate_topology();

        // DELIMIT_SHARP with the flag set (non-smooth): the edge survives.
        let (mut mesh, shared, _a, _b) = build_coplanar_pair();
        mesh.set_edge_sharp(shared, true);
        let params = DissolveLimitParams {
            angle_limit: FRAC_PI_2,
            use_dissolve_boundaries: false,
            delimit: DELIMIT_SHARP,
        };
        assert!(matches!(
            mesh.dissolve_limit(&[shared], &[], &params),
            DissolveLimitResult::Ran { region_out } if region_out.is_empty()
        ));
        assert_eq!(
            mesh.face_count(),
            2,
            "DELIMIT_SHARP preserves the sharp (non-smooth) edge"
        );
        assert!(mesh.edges.try_get(shared).is_some());
        mesh.validate_topology();

        // DELIMIT_SHARP with a smooth edge (default): not a delimiter,
        // dissolves. The smooth state is the positive-presence one.
        let (mut mesh, shared, _a, _b) = build_coplanar_pair();
        let params = DissolveLimitParams {
            angle_limit: FRAC_PI_2,
            use_dissolve_boundaries: false,
            delimit: DELIMIT_SHARP,
        };
        mesh.dissolve_limit(&[shared], &[], &params);
        assert_eq!(
            mesh.face_count(),
            1,
            "a smooth edge is not sharp; it dissolves"
        );
        mesh.validate_topology();
    }

    #[test]
    fn delimit_seam_or_material_blocks_either_boundary() {
        // §8 "delimit = SEAM | MATERIAL | SHARP simultaneously" — the OR
        // composition delimits the union: an edge that is SEAM-flagged *or* a
        // material boundary is blocked. Here the shared edge is both at once,
        // and the combined mask preserves it.
        let (mut mesh, shared, face_a, face_b) = build_coplanar_pair();
        mesh.set_edge_seam(shared, true);
        mesh.set_face_material_index(face_a, 0);
        mesh.set_face_material_index(face_b, 1);
        let params = DissolveLimitParams {
            angle_limit: FRAC_PI_2,
            use_dissolve_boundaries: false,
            delimit: DELIMIT_SEAM | DELIMIT_MATERIAL,
        };
        assert!(matches!(
            mesh.dissolve_limit(&[shared], &[], &params),
            DissolveLimitResult::Ran { region_out } if region_out.is_empty()
        ));
        assert_eq!(
            mesh.face_count(),
            2,
            "SEAM | MATERIAL preserves the doubly-delimited edge"
        );
        assert!(mesh.edges.try_get(shared).is_some());
        mesh.validate_topology();

        // The OR fires even when only *one* condition holds: a material
        // boundary with no seam flag is still blocked by the combined mask.
        let (mut mesh, shared, face_a, face_b) = build_coplanar_pair();
        mesh.set_face_material_index(face_a, 0);
        mesh.set_face_material_index(face_b, 1);
        let params = DissolveLimitParams {
            angle_limit: FRAC_PI_2,
            use_dissolve_boundaries: false,
            delimit: DELIMIT_SEAM | DELIMIT_MATERIAL,
        };
        mesh.dissolve_limit(&[shared], &[], &params);
        assert_eq!(
            mesh.face_count(),
            2,
            "the OR blocks on the material boundary alone"
        );
        mesh.validate_topology();

        // And neither condition holding ⇒ the edge dissolves under the same
        // combined mask (confirms the mask's effect is the conditions).
        let (mut mesh, shared, _a, _b) = build_coplanar_pair();
        let params = DissolveLimitParams {
            angle_limit: FRAC_PI_2,
            use_dissolve_boundaries: false,
            delimit: DELIMIT_SEAM | DELIMIT_MATERIAL,
        };
        mesh.dissolve_limit(&[shared], &[], &params);
        assert_eq!(
            mesh.face_count(),
            1,
            "no seam and matching materials ⇒ edge dissolves"
        );
        mesh.validate_topology();
    }

    /// Two quads sharing a seam `s0 – v – s1` (two consecutive seam edges per
    /// face), gently folded about the seam. The middle seam vert `v` is an
    /// edge-pair vert whose two incident seam edges are both manifold (each
    /// flanks faces A and B). `v` is kinked off the `s0 – s1` line so its
    /// corner angle is a non-trivial ~43.6°, while the inter-face fold is a
    /// *shallow* ~39.8° dihedral.
    ///
    /// This split between a large corner angle and a small dihedral is what
    /// makes the §3.3 carve-out observable in the *outcome* (not just the
    /// cost): the §6.1 modulated form `corner · dihedral / (π/2)` shrinks the
    /// cost well below the gate (the shallow dihedral discounts it), so a
    /// *non-delimiter* `v` collapses; the §3.3 *un-modulated* form is the raw
    /// corner angle, above the gate, so a *delimiter* `v` survives. Returns
    /// `(mesh, v, seam_edge_a, seam_edge_b)`.
    fn build_folded_seam_with_kinked_vert() -> (LoopMesh, VertexKey, EdgeKey, EdgeKey) {
        let mut mesh = LoopMesh::new();
        // Seam s0 – v – s1 runs along +Y; `v` kinked toward +X by 0.4.
        let s0 = mesh.create_vertex(Vec3::new(0.0, 0.0, 0.0));
        let v = mesh.create_vertex(Vec3::new(0.4, 1.0, 0.0));
        let s1 = mesh.create_vertex(Vec3::new(0.0, 2.0, 0.0));
        // Left wing (face A), flat in the z = 0 plane.
        let wing_a = mesh.create_vertex(Vec3::new(-1.5, 1.0, 0.0));
        // Right wing (face B), lifted in +Z to a shallow ~40° fold about the seam.
        let wing_b = mesh.create_vertex(Vec3::new(1.0, 1.0, 0.5));

        let e_s0v = mesh.create_edge([s0, v], EdgeCreateFlags::NO_DUPLICATE);
        let e_vs1 = mesh.create_edge([v, s1], EdgeCreateFlags::NO_DUPLICATE);
        let e_s1wa = mesh.create_edge([s1, wing_a], EdgeCreateFlags::NO_DUPLICATE);
        let e_was0 = mesh.create_edge([wing_a, s0], EdgeCreateFlags::NO_DUPLICATE);
        // Face A loop: s0 -> v -> s1 -> wing_a (seam half-edges s0-v, v-s1).
        mesh.face_create(
            &[s0, v, s1, wing_a],
            &[e_s0v, e_vs1, e_s1wa, e_was0],
            FaceCreateFlags::NONE,
        );
        // Face B loop: s1 -> v -> s0 -> wing_b, reusing the two seam edges so
        // each seam edge is manifold (flanked by A and B).
        let e_s0wb = mesh.create_edge([s0, wing_b], EdgeCreateFlags::NO_DUPLICATE);
        let e_wbs1 = mesh.create_edge([wing_b, s1], EdgeCreateFlags::NO_DUPLICATE);
        mesh.face_create(
            &[s1, v, s0, wing_b],
            &[e_vs1, e_s0v, e_s0wb, e_wbs1],
            FaceCreateFlags::NONE,
        );
        mesh.calc_face_normals();
        (mesh, v, e_s0v, e_vs1)
    }

    #[test]
    fn delimit_sharp_vert_takes_unmodulated_corner_cost() {
        // §3.3 / §6.1 — vert-pass transitivity through a *semantic* delimiter.
        //
        // `v` is an edge-pair vert whose two seam edges are manifold; the
        // build gives it a large corner angle (~43.6°, raw cost ~0.761) but a
        // shallow inter-face fold (~39.8° dihedral). With angle_limit = 40°
        // (~0.698 rad) the two §6.1 cost forms land on opposite sides of the
        // gate:
        //   • non-delimiter (delimit = 0): modulated cost
        //     corner·dihedral/(π/2) ≈ 0.337 < 0.698  ⇒  v COLLAPSES.
        //   • delimiter (DELIMIT_SHARP on a seam edge): the §3.3 un-modulated
        //     corner angle ≈ 0.761 ≥ 0.698  ⇒  v SURVIVES.
        // The delimiter makes v *more conservative* — it forgoes the
        // shallow-dihedral discount the modulated form grants, which is the
        // §3.3 carve-out: a vert on a delimiter still dissolves if its own
        // corner is shallow, but it does not borrow the dihedral modulation.
        let angle_limit = 40.0_f32.to_radians();

        // First pin the §6.1 cost forms directly (the mechanism), so the
        // outcome assertions below are anchored to the documented carve-out and
        // not to incidental geometry.
        {
            let (mut mesh, v, seam_a, _seam_b) = build_folded_seam_with_kinked_vert();
            let modulated = mesh.dissolve_limit_vert_cost(v, 0);
            assert!(
                modulated < angle_limit,
                "non-delimiter vert takes the modulated form, discounted below the gate \
                 (cost {modulated}, limit {angle_limit})"
            );
            mesh.set_edge_sharp(seam_a, true);
            let unmodulated = mesh.dissolve_limit_vert_cost(v, super::DELIMIT_SHARP);
            assert!(
                unmodulated >= angle_limit,
                "§3.3 delimiter vert takes the un-modulated corner angle, above the gate \
                 (cost {unmodulated}, limit {angle_limit})"
            );
            // The collapse is geometrically clean (gentle fold), so the §6.3
            // degeneracy guard does not interfere with either outcome.
            assert!(
                !mesh.dissolve_limit_vert_collapse_is_degenerate(v),
                "the gentle fold keeps the collapse non-degenerate"
            );
        }

        // delimit = 0: the modulated cost pops v; the vert pass collapses it.
        let (mut mesh, v, _seam_a, _seam_b) = build_folded_seam_with_kinked_vert();
        let v0 = mesh.vertex_count();
        let params = DissolveLimitParams {
            angle_limit,
            use_dissolve_boundaries: false,
            delimit: 0,
        };
        let result = mesh.dissolve_limit(&[], &[v], &params);
        assert!(matches!(result, DissolveLimitResult::Ran { .. }));
        assert!(
            mesh.vertices.try_get(v).is_none(),
            "without a delimiter the modulated cost dissolves the seam vert"
        );
        assert_eq!(
            mesh.vertex_count(),
            v0 - 1,
            "exactly the one seam vert collapses"
        );
        mesh.validate_topology();

        // DELIMIT_SHARP with a sharp-flagged seam edge: v is a §3.3 delimiter
        // vert, takes the un-modulated corner cost (above the gate), survives.
        let (mut mesh, v, seam_a, _seam_b) = build_folded_seam_with_kinked_vert();
        mesh.set_edge_sharp(seam_a, true);
        let v0 = mesh.vertex_count();
        let params = DissolveLimitParams {
            angle_limit,
            use_dissolve_boundaries: false,
            delimit: super::DELIMIT_SHARP,
        };
        let result = mesh.dissolve_limit(&[], &[v], &params);
        assert!(matches!(result, DissolveLimitResult::Ran { .. }));
        assert!(
            mesh.vertices.try_get(v).is_some(),
            "the §3.3 delimiter vert forgoes the dihedral discount and survives"
        );
        assert_eq!(
            mesh.vertex_count(),
            v0,
            "no collapse: the delimiter vert is preserved"
        );
        mesh.validate_topology();
    }

    /// Two coplanar quads sharing edge `v1–v2`, the second wound *with* the
    /// shared edge (loop traverses `v1 → v2` on both faces) so the edge is
    /// **non-contiguous** (§3 row `NORMAL` / §5.1) and face B's normal flips
    /// relative to face A. Returns `(mesh, shared_edge)`.
    fn build_noncontiguous_coplanar_pair() -> (LoopMesh, EdgeKey) {
        let mut mesh = LoopMesh::new();
        let v: Vec<VertexKey> = vec![
            mesh.create_vertex(Vec3::new(0.0, 0.0, 0.0)), // v0
            mesh.create_vertex(Vec3::new(1.0, 0.0, 0.0)), // v1
            mesh.create_vertex(Vec3::new(1.0, 1.0, 0.0)), // v2
            mesh.create_vertex(Vec3::new(0.0, 1.0, 0.0)), // v3
            mesh.create_vertex(Vec3::new(2.0, 0.0, 0.0)), // v4
            mesh.create_vertex(Vec3::new(2.0, 1.0, 0.0)), // v5
        ];
        // Face A: CCW [v0,v1,v2,v3], shared edge v1->v2.
        let e01 = mesh.create_edge([v[0], v[1]], EdgeCreateFlags::NO_DUPLICATE);
        let e12 = mesh.create_edge([v[1], v[2]], EdgeCreateFlags::NO_DUPLICATE);
        let e23 = mesh.create_edge([v[2], v[3]], EdgeCreateFlags::NO_DUPLICATE);
        let e30 = mesh.create_edge([v[3], v[0]], EdgeCreateFlags::NO_DUPLICATE);
        mesh.face_create(
            &[v[0], v[1], v[2], v[3]],
            &[e01, e12, e23, e30],
            FaceCreateFlags::NONE,
        );
        // Face B: [v1,v2,v5,v4] — also traverses the shared edge v1->v2, so the
        // edge is non-contiguous and B winds opposite to A (flipped normal).
        let e25 = mesh.create_edge([v[2], v[5]], EdgeCreateFlags::NO_DUPLICATE);
        let e54 = mesh.create_edge([v[5], v[4]], EdgeCreateFlags::NO_DUPLICATE);
        let e41 = mesh.create_edge([v[4], v[1]], EdgeCreateFlags::NO_DUPLICATE);
        mesh.face_create(
            &[v[1], v[2], v[5], v[4]],
            &[e12, e25, e54, e41],
            FaceCreateFlags::NONE,
        );

        let shared = mesh.query_edge_exists(v[1], v[2]).unwrap();
        (mesh, shared)
    }

    #[test]
    fn delimit_normal_invalidates_noncontiguous_edge_cost() {
        // §3 row `NORMAL` / §5.1 / §3.2 — a non-contiguous coplanar edge
        // (winding-discrepant: B's normal is flipped vs A) scores a low,
        // poppable cost +dot ≈ −1 at delimit = 0, but DELIMIT_NORMAL marks it a
        // delimiter so its cost becomes the COST_INVALID sentinel and it is
        // never popped.
        let (mut mesh, shared) = build_noncontiguous_coplanar_pair();
        mesh.calc_face_normals();
        assert!(
            !mesh.dissolve_limit_edge_is_contiguous(shared),
            "the engineered pair shares a same-direction edge (non-contiguous)"
        );
        let cost_open = mesh.dissolve_limit_edge_cost(shared, 0);
        assert!(
            cost_open.is_finite() && cost_open < 0.0,
            "without NORMAL the non-contiguous coplanar edge is poppable (cost {cost_open})"
        );
        let cost_delim = mesh.dissolve_limit_edge_cost(shared, DELIMIT_NORMAL);
        assert!(
            cost_delim == super::COST_INVALID,
            "DELIMIT_NORMAL marks the non-contiguous edge a delimiter (cost {cost_delim})"
        );
    }

    #[test]
    fn delimit_normal_preserves_noncontiguous_edge() {
        // §3 row `NORMAL` — running the operator with DELIMIT_NORMAL over the
        // non-contiguous edge leaves it intact: it is never popped, so no merge
        // is attempted and region.out is empty. (loopmesh's `faces_join`
        // independently declines a non-contiguous merge, so the delimit = 0
        // baseline also leaves two faces here; the delimiter's effect is pinned
        // at the cost level in `delimit_normal_invalidates_noncontiguous_edge_cost`.)
        let (mut mesh, shared) = build_noncontiguous_coplanar_pair();
        let params = DissolveLimitParams {
            angle_limit: FRAC_PI_2,
            use_dissolve_boundaries: false,
            delimit: DELIMIT_NORMAL,
        };
        assert!(matches!(
            mesh.dissolve_limit(&[shared], &[], &params),
            DissolveLimitResult::Ran { region_out } if region_out.is_empty()
        ));
        assert_eq!(
            mesh.face_count(),
            2,
            "DELIMIT_NORMAL preserves the non-contiguous edge"
        );
        assert!(mesh.edges.try_get(shared).is_some());
        mesh.validate_topology();
    }

    /// A coplanar quad pair sharing edge `v1–v2`, contiguous (the normal
    /// manifold winding), carrying a per-loop UV layer. `split_uv` controls
    /// whether the two faces' UVs agree at the seam (a single UV island) or
    /// disagree (a UV-island boundary). Returns `(mesh, shared_edge, uv_layer)`.
    fn build_coplanar_pair_with_uv(split_uv: bool) -> (LoopMesh, EdgeKey) {
        let mut mesh = LoopMesh::new();
        let v: Vec<VertexKey> = vec![
            mesh.create_vertex(Vec3::new(0.0, 0.0, 0.0)), // v0
            mesh.create_vertex(Vec3::new(1.0, 0.0, 0.0)), // v1
            mesh.create_vertex(Vec3::new(1.0, 1.0, 0.0)), // v2
            mesh.create_vertex(Vec3::new(0.0, 1.0, 0.0)), // v3
            mesh.create_vertex(Vec3::new(2.0, 0.0, 0.0)), // v4
            mesh.create_vertex(Vec3::new(2.0, 1.0, 0.0)), // v5
        ];
        let e01 = mesh.create_edge([v[0], v[1]], EdgeCreateFlags::NO_DUPLICATE);
        let e12 = mesh.create_edge([v[1], v[2]], EdgeCreateFlags::NO_DUPLICATE);
        let e23 = mesh.create_edge([v[2], v[3]], EdgeCreateFlags::NO_DUPLICATE);
        let e30 = mesh.create_edge([v[3], v[0]], EdgeCreateFlags::NO_DUPLICATE);
        let face_a = mesh.face_create(
            &[v[0], v[1], v[2], v[3]],
            &[e01, e12, e23, e30],
            FaceCreateFlags::NONE,
        );
        let e14 = mesh.create_edge([v[1], v[4]], EdgeCreateFlags::NO_DUPLICATE);
        let e45 = mesh.create_edge([v[4], v[5]], EdgeCreateFlags::NO_DUPLICATE);
        let e52 = mesh.create_edge([v[5], v[2]], EdgeCreateFlags::NO_DUPLICATE);
        let face_b = mesh.face_create(
            &[v[1], v[4], v[5], v[2]],
            &[e14, e45, e52, e12],
            FaceCreateFlags::NONE,
        );

        let uv = mesh
            .register_loop_attribute(ElementAttributeDesc::numeric(
                AttributeKey::custom("uv"),
                AttrType::Vec2,
            ))
            .unwrap();
        // Assign UVs by vertex position so both faces agree at the seam verts
        // v1, v2 (a single island). When `split_uv`, face B's seam corners are
        // pushed off by a large offset so the edge becomes an island boundary.
        let seam_offset = if split_uv { 10.0 } else { 0.0 };
        for l in mesh.face_loop_keys(face_a) {
            let p = mesh.vertex_position(mesh.loops.get(l).vertex);
            mesh.loop_attribs_mut()
                .set_vec2(uv, l.as_usize(), [p.x, p.y]);
        }
        for l in mesh.face_loop_keys(face_b) {
            let vert = mesh.loops.get(l).vertex;
            let p = mesh.vertex_position(vert);
            let on_seam = vert == v[1] || vert == v[2];
            let off = if on_seam { seam_offset } else { 0.0 };
            mesh.loop_attribs_mut()
                .set_vec2(uv, l.as_usize(), [p.x + off, p.y]);
        }

        let shared = mesh.query_edge_exists(v[1], v[2]).unwrap();
        (mesh, shared)
    }

    #[test]
    fn delimit_uv_preserves_island_boundary() {
        // §3 row `UV` / §3.2 — an edge whose per-loop UVs disagree across the
        // seam is a UV-island boundary. With DELIMIT_UV it is delimited and
        // survives; with matching UVs (single island) or delimit = 0 it
        // dissolves.

        // delimit = 0: the split-UV edge dissolves (UV ignored).
        let (mut mesh, shared) = build_coplanar_pair_with_uv(true);
        let params = DissolveLimitParams {
            angle_limit: FRAC_PI_2,
            use_dissolve_boundaries: false,
            delimit: 0,
        };
        mesh.dissolve_limit(&[shared], &[], &params);
        assert_eq!(
            mesh.face_count(),
            1,
            "UV split edge dissolves when delimit = 0"
        );
        mesh.validate_topology();

        // DELIMIT_UV with mismatched seam UVs: the island boundary survives.
        let (mut mesh, shared) = build_coplanar_pair_with_uv(true);
        let params = DissolveLimitParams {
            angle_limit: FRAC_PI_2,
            use_dissolve_boundaries: false,
            delimit: DELIMIT_UV,
        };
        assert!(matches!(
            mesh.dissolve_limit(&[shared], &[], &params),
            DissolveLimitResult::Ran { region_out } if region_out.is_empty()
        ));
        assert_eq!(
            mesh.face_count(),
            2,
            "DELIMIT_UV preserves the UV-island boundary"
        );
        assert!(mesh.edges.try_get(shared).is_some());
        mesh.validate_topology();

        // DELIMIT_UV with matching seam UVs (single island): not delimited,
        // dissolves as if delimit = 0.
        let (mut mesh, shared) = build_coplanar_pair_with_uv(false);
        let params = DissolveLimitParams {
            angle_limit: FRAC_PI_2,
            use_dissolve_boundaries: false,
            delimit: DELIMIT_UV,
        };
        mesh.dissolve_limit(&[shared], &[], &params);
        assert_eq!(
            mesh.face_count(),
            1,
            "matching seam UVs are contiguous; the edge dissolves"
        );
        mesh.validate_topology();
    }

    #[test]
    fn delimit_uv_auto_clears_with_no_uv_layers() {
        // §3.1 — DELIMIT_UV with zero Float2 loop layers auto-clears before the
        // pass runs, so the bit is inert and the mesh behaves as delimit = 0.
        // A plain coplanar strip (no UV layer) fully collapses under DELIMIT_UV
        // exactly as it would under delimit = 0.
        let mut mesh = LoopMesh::new();
        let (_verts, edges, interior_verts) = build_coplanar_strip(&mut mesh, 4);
        let params = DissolveLimitParams {
            angle_limit: std::f32::consts::FRAC_PI_4,
            use_dissolve_boundaries: false,
            delimit: DELIMIT_UV,
        };
        let result = mesh.dissolve_limit(&edges, &interior_verts, &params);
        assert!(
            matches!(result, DissolveLimitResult::Ran { .. }),
            "UV bit auto-clears, runs"
        );
        assert_eq!(
            mesh.face_count(),
            1,
            "no UV layers ⇒ UV bit inert ⇒ full collapse"
        );
        assert_eq!(mesh.vertex_count(), 4);
        mesh.validate_topology();
    }

    /// Build a single CCW face in the XY plane from `positions` and return
    /// `(vert keys in order, face)`.
    fn build_single_face(mesh: &mut LoopMesh, positions: &[Vec3]) -> (Vec<VertexKey>, FaceKey) {
        let verts: Vec<VertexKey> = positions.iter().map(|&p| mesh.create_vertex(p)).collect();
        let mut edges: Vec<EdgeKey> = Vec::with_capacity(verts.len());
        for i in 0..verts.len() {
            let a = verts[i];
            let b = verts[(i + 1) % verts.len()];
            edges.push(mesh.create_edge([a, b], EdgeCreateFlags::NO_DUPLICATE));
        }
        let face = mesh.face_create(&verts, &edges, FaceCreateFlags::NONE);
        (verts, face)
    }

    #[test]
    fn reflex_empty_ear_corner_collapses() {
        // §6.3 step 3 — a reflex corner whose ear triangle is empty must be
        // allowed to collapse. The ear of a reflex corner winds *opposite* to a
        // CCW face; the degeneracy guard must not reject it on that basis alone.
        //
        // Pentagon, CCW in XY. `r` = (2, 0.1) is a near-collinear reflex vertex
        // (pushed slightly into the interior off the straight bottom edge), so
        // its corner cost is small (passes the angle gate) and its ear triangle
        // (v0, r, v2) is a thin outward sliver containing no other vertex.
        let mut mesh = LoopMesh::new();
        let (verts, _face) = build_single_face(
            &mut mesh,
            &[
                Vec3::new(0.0, 0.0, 0.0), // v0  (ear-prev of r)
                Vec3::new(2.0, 0.1, 0.0), // r   (reflex, near-collinear)
                Vec3::new(4.0, 0.0, 0.0), // v2  (ear-next of r)
                Vec3::new(4.0, 3.0, 0.0), // v3
                Vec3::new(0.0, 3.0, 0.0), // v4
            ],
        );
        let r = verts[1];

        let params = DissolveLimitParams {
            angle_limit: FRAC_PI_2,
            use_dissolve_boundaries: false,
            delimit: 0,
        };
        mesh.dissolve_limit(&[], &[r], &params);

        assert_eq!(
            mesh.vertex_count(),
            4,
            "the reflex empty-ear vert collapses away"
        );
        assert_eq!(mesh.face_count(), 1, "the face survives as a quad");
        assert!(
            mesh.query_edge_exists(verts[0], verts[2]).is_some(),
            "the weld edge v0–v2 replaces the two arms through r"
        );
        mesh.validate_topology();
    }

    #[test]
    fn guard_fires_when_witness_in_ear() {
        // §6.3 step 3 — contrast to the reflex-empty-ear case: a wide-open corner
        // whose ear triangle traps another vertex must be rejected by the
        // point-in-triangle sub-check. The candidate passes the angle gate but
        // the guard demotes it, so nothing collapses.
        //
        // CCW hexagon in XY. `c` = (6, 3) is a wide-open corner (corner cost
        // ≈ 0.927 < π/2); its loop-neighbours are (12, 0) and (0, 0), so its ear
        // triangle is ((12,0), (6,3), (0,0)). The witness `w` = (6, 2) — a sharp
        // spike rising from below the base — sits inside that ear.
        let mut mesh = LoopMesh::new();
        let (verts, _face) = build_single_face(
            &mut mesh,
            &[
                Vec3::new(2.0, -3.0, 0.0),  // 0
                Vec3::new(6.0, 2.0, 0.0),   // 1  w  (witness inside c's ear)
                Vec3::new(10.0, -3.0, 0.0), // 2
                Vec3::new(12.0, 0.0, 0.0),  // 3  ear-next of c
                Vec3::new(6.0, 3.0, 0.0),   // 4  c  (wide-open candidate)
                Vec3::new(0.0, 0.0, 0.0),   // 5  ear-prev of c
            ],
        );
        let c = verts[4];

        let params = DissolveLimitParams {
            angle_limit: FRAC_PI_2,
            use_dissolve_boundaries: false,
            delimit: 0,
        };
        mesh.dissolve_limit(&[], &[c], &params);

        assert_eq!(mesh.vertex_count(), 6, "the guard fires; nothing collapses");
        assert_eq!(mesh.face_count(), 1);
        assert!(
            mesh.query_edge_exists(verts[3], verts[5]).is_none(),
            "the would-be weld edge is never created"
        );
        mesh.validate_topology();
    }

    #[test]
    fn clear_ear_corner_collapses() {
        // §6.3 step 3 — the contrast partner to `guard_fires_when_witness_in_ear`:
        // the same wide-open corner with the witness lifted out of the ear
        // collapses normally. Isolates the guard's firing to ear occupancy.
        // Witness `w` = (6, -1) now sits below the base, outside c's ear.
        let mut mesh = LoopMesh::new();
        let (verts, _face) = build_single_face(
            &mut mesh,
            &[
                Vec3::new(2.0, -3.0, 0.0),  // 0
                Vec3::new(6.0, -1.0, 0.0),  // 1  w  (witness below the base)
                Vec3::new(10.0, -3.0, 0.0), // 2
                Vec3::new(12.0, 0.0, 0.0),  // 3  ear-next of c
                Vec3::new(6.0, 3.0, 0.0),   // 4  c  (wide-open candidate)
                Vec3::new(0.0, 0.0, 0.0),   // 5  ear-prev of c
            ],
        );
        let c = verts[4];

        let params = DissolveLimitParams {
            angle_limit: FRAC_PI_2,
            use_dissolve_boundaries: false,
            delimit: 0,
        };
        mesh.dissolve_limit(&[], &[c], &params);

        assert_eq!(mesh.vertex_count(), 5, "clear ear ⇒ the corner collapses");
        assert_eq!(mesh.face_count(), 1);
        assert!(
            mesh.query_edge_exists(verts[3], verts[5]).is_some(),
            "weld edge (ear-prev)–(ear-next) replaces the arms through c"
        );
        mesh.validate_topology();
    }
}
