use smallvec::SmallVec;

use crate::loopmesh::*;

/// Float-noise slack on the `angle_threshold` comparisons (§5.2). Below this
/// floor `use_verts` is forced off entirely (§1); above `π − ε` every
/// edge-pair candidate qualifies for collapse.
const ANGLE_EPS: f32 = 0.0001;

/// Parameters controlling [`LoopMesh::dissolve_edges`].
///
/// Mirrors the BMOP set-input shape described in
/// `radial-mesh-guides/operators/DISSOLVE_EDGES.md` §1.
///
/// The default-parameter path, the full `use_verts` post-pass — wide,
/// near-zero, *and* intermediate `angle_threshold` (the §5.2 blended-angle
/// gate) — and the `use_face_split` pre-pass (§4) are all implemented.
/// [`LoopMesh::dissolve_edges`] reports its run through
/// [`DissolveEdgesResult::Ran`].
#[derive(Clone, Copy)]
pub struct DissolveEdgesParams {
    /// Post-pass: collapse adjacent verts that became edge-pairs after the
    /// per-edge merges (§5). Supported with a wide or near-zero
    /// `angle_threshold` (see the type docs).
    pub use_verts: bool,
    /// Pre-pass: triangulate n-gon corners around verts that would become
    /// edge-pairs after the merges (§4). Runs before any per-edge merge.
    pub use_face_split: bool,
    /// Gate for `use_verts` (§5.2). Only consulted when `use_verts` is on.
    /// At the default `π` every edge-pair candidate collapses; at `0` the
    /// `use_verts` pass is short-circuited entirely (§1); at an intermediate
    /// value each still-edge-pair candidate's collapse is gated on its
    /// blended dihedral angle being strictly below `angle_threshold − ε`.
    pub angle_threshold: f32,
}

impl Default for DissolveEdgesParams {
    fn default() -> Self {
        Self {
            use_verts: false,
            use_face_split: false,
            angle_threshold: std::f32::consts::PI,
        }
    }
}

/// Outcome of [`LoopMesh::dissolve_edges`]. Distinguishes a real run from a
/// declined-because-unsupported configuration, so callers can detect the
/// latter without inspecting the mesh.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DissolveEdgesResult {
    /// The operator ran (the mesh may or may not have changed, depending on
    /// the per-edge classifications).
    Ran,
    /// The parameter tuple is outside the implemented surface. The mesh was
    /// left untouched. No parameter currently triggers this; the variant is
    /// retained so future unimplemented knobs can decline without panicking.
    Unsupported,
}

/// How the `use_verts` post-pass treats the marked candidates after the
/// merges, decided up front from `angle_threshold` (§1 / §5.2 / §6.5).
#[derive(Clone, Copy, PartialEq)]
enum UseVertsMode {
    /// Post-pass disabled: either `use_verts` was off, or the threshold sits
    /// at/below the epsilon floor and forces it off (§1).
    Off,
    /// Threshold at/above `π − ε` (the `dissolve_all` fast path): every
    /// still-edge-pair candidate collapses without an angle test (§5.2
    /// default branch / §6.5 sub-step 3 first bullet).
    CollapseAll,
    /// Intermediate threshold (above `ε`, below `π − ε`): each still-edge-pair
    /// candidate's collapse is gated by the §5.2 blended-angle test against
    /// the carried threshold, unless the candidate was reached via the §5.1
    /// chain-walk (chain verts always collapse — the §6.5 chain override).
    Gated { angle_threshold: f32 },
}

/// Per-edge classification of an input edge (§3). Drives whether the
/// per-edge merge runs at all: only `ManifoldPair` does work; everything
/// else is a silent no-op for that edge.
///
/// For `ManifoldPair`, `face_a` is the **survivor** — the face on the
/// dissolved edge's primary loop `e.l` (§3.1 / §11). The merged face keeps
/// `face_a`'s identity and per-face customdata, so `face_a` must be passed
/// first to [`LoopMesh::faces_join`] (whose two-face form keeps its first
/// argument's face). `face_b` is the other flanking face.
enum EdgeClass {
    ManifoldPair { face_a: FaceKey, face_b: FaceKey },
    NoOp,
}

impl LoopMesh {
    /// Remove every input edge that flanks two distinct faces, merging the
    /// two faces into one.
    ///
    /// Follows the algorithm in
    /// `radial-mesh-guides/operators/DISSOLVE_EDGES.md`:
    ///
    /// 1. §3 per-edge classification: an edge is acted on only when its
    ///    radial cycle has exactly two loops belonging to two *distinct*
    ///    faces (the manifold face-pair case). Wire (radial 0), boundary
    ///    (radial 1), self-radial (radial 2, same face), and non-manifold
    ///    (radial ≥ 3) edges are silently skipped.
    /// 2. §6.4 per-edge merge: each manifold-pair edge is dissolved by
    ///    merging its two faces via the n==2 `faces_join` form, which is
    ///    the same JFKE surgery §3.1 / §6.4 names — the shared edge and its
    ///    two loops die, and the merged face's perimeter is the symmetric
    ///    difference of the two source perimeters (§3.1).
    /// 3. §6.4 / §8 multi-edge iteration: the classification is re-tested
    ///    afresh for every input edge, so a later edge sees the topology
    ///    left by earlier merges (the shared-endpoint case picks up the
    ///    post-first-merge face pair; disjoint pairs are order-independent).
    ///
    /// The default-parameter path, the `use_face_split` pre-pass (§4), and the
    /// `use_verts` post-pass (§5) with any `angle_threshold` are implemented:
    ///
    /// - **Phase 1 (§6.1)** — when `use_face_split` or `use_verts` is on, tag
    ///   every input edge with the `EDGE_TAG` bit so the pre-/post-passes can
    ///   recognise input-set membership.
    /// - **Phase 2 (§6.2)** — when `use_verts` is on, mark the collapse
    ///   candidates *before* the merges run, so the §5.1 predicates
    ///   (tagged-edge counts, "touches an unselected face") read the
    ///   pre-merge topology. The marking applies the three §5.1 filters: the
    ///   triangle-to-quad anti-degeneration guard, the chain-walk to the
    ///   chain terminus, and the T-junction / incidental-touch filter.
    /// - **Phase 3 (§6.3 / §4)** — when `use_face_split` is on, mark every
    ///   vert whose untagged-edge count is ≤ 2 and triangulate the n-gon
    ///   corners around the marked (non-edge-pair) verts, *before* any merge.
    /// - **Phase 4 (§6.4)** — the per-edge merges (the default path).
    /// - **Phase 5 (§6.5)** — for each marked vert that is *still* an
    ///   edge-pair after the merges, collapse it via `vert_collapse_edge`
    ///   (§5.3). The angle is sampled once per candidate before any collapse
    ///   runs (§6.5 snapshot-then-act); at a wide `angle_threshold` every
    ///   candidate qualifies so no angle math is needed this far.
    pub fn dissolve_edges(
        &mut self,
        edges: &[EdgeKey],
        params: &DissolveEdgesParams,
    ) -> DissolveEdgesResult {
        let mode = if !params.use_verts {
            UseVertsMode::Off
        } else if params.angle_threshold <= ANGLE_EPS {
            // §1 — at/below the epsilon floor `use_verts` is forced off.
            UseVertsMode::Off
        } else if params.angle_threshold > std::f32::consts::PI - ANGLE_EPS {
            // §5.2 default branch — every edge-pair candidate collapses.
            UseVertsMode::CollapseAll
        } else {
            // §5.2 blended-angle gate for an intermediate threshold.
            UseVertsMode::Gated {
                angle_threshold: params.angle_threshold,
            }
        };

        // §6.1 — the `EDGE_TAG` (input-set membership) bit is needed by either
        // pre-/post-pass: the §6.3 pre-pass counts each vert's *untagged*
        // edges, and the §5.1 candidate marking counts each vert's *tagged*
        // edges. It is seeded once here and kept alive through phase 3, then
        // popped before the phase-4 merges. The layer is a pushed op-flag layer
        // (the BMOP flag-bit layer of §10) so it vanishes when the operator
        // returns.
        let tag_inputs = params.use_face_split || mode != UseVertsMode::Off;
        let tagged = if tag_inputs {
            let tagged = self.push_edge_op_flag_layer();
            for &edge in edges {
                if self.edges.try_get(edge).is_none() {
                    continue;
                }
                self.edge_op_flag_set(tagged, edge, 0);
            }
            Some(tagged)
        } else {
            None
        };

        // §6.2 — when the post-pass is on, mark the §5.1 collapse candidates
        // against the pre-merge topology.
        let candidate_marks = if mode == UseVertsMode::Off {
            None
        } else {
            let tagged = tagged.expect("EDGE_TAG seeded whenever use_verts is on");
            // §6.1 — `EDGE_CHAIN`: set on input edges that flank two faces and
            // have an edge-pair endpoint. Used as the chain-walk abort marker
            // (§8): the §5.1 walk stops if it runs into a *different* input
            // chain edge, avoiding double-marking intersecting selections.
            let chain = self.push_edge_op_flag_layer();
            // §6.5 chain override — verts reached through the §5.1 chain-walk
            // skip the §5.2 angle test entirely and always collapse. Recorded
            // as a separate vert layer so the gated collapse pass can honour
            // the override without re-walking.
            let chain_target = self.push_vert_op_flag_layer();
            for &edge in edges {
                if self.edges.try_get(edge).is_none() {
                    continue;
                }
                if matches!(
                    self.dissolve_edges_classify(edge),
                    EdgeClass::ManifoldPair { .. }
                ) {
                    let [v1, v2] = self.edge_vert_keys(edge);
                    let v1_pair = self.vertex_disk_count_edges_capped(v1, 3) == 2;
                    let v2_pair = self.vertex_disk_count_edges_capped(v2, 3) == 2;
                    if v1_pair || v2_pair {
                        self.edge_op_flag_set(chain, edge, 0);
                    }
                }
            }
            // §6.1 — extend the tag set along each input edge's chain so the
            // §5.1 tagged-edge counts at the chain termini reflect the
            // unselected chain edges that dissolve as a consequence.
            self.dissolve_edges_tag_chains(edges, tagged);
            let marked = self.push_vert_op_flag_layer();
            self.dissolve_edges_mark_use_verts_candidates(
                edges,
                tagged,
                chain,
                marked,
                chain_target,
            );
            self.pop_edge_op_flag_layer(chain);
            Some((marked, chain_target))
        };

        // §6.3 — the `use_face_split` pre-pass, run after the §6.2 marking but
        // before the phase-4 merges (the §6 phase ordering). The two passes are
        // independent (§6.3) — they share neither marker bit nor effect.
        if params.use_face_split {
            let tagged = tagged.expect("EDGE_TAG seeded whenever use_face_split is on");
            self.dissolve_edges_face_split_pre_pass(tagged);
        }

        // The `EDGE_TAG` set has served both passes; drop it before the merges.
        if let Some(tagged) = tagged {
            self.pop_edge_op_flag_layer(tagged);
        }

        // §6.4 — per-edge dispatch in input order. Each iteration
        // re-classifies the edge against the *current* topology so the
        // shared-endpoint case (§8) sees the previous merges.
        for &edge in edges {
            if self.edges.try_get(edge).is_none() {
                // A prior merge already consumed this edge (it was the
                // shared interior edge of a pair). Skip it silently — the
                // re-test-afresh contract treats an already-dissolved edge
                // as a no-op rather than an error.
                continue;
            }
            match self.dissolve_edges_classify(edge) {
                EdgeClass::ManifoldPair { face_a, face_b } => {
                    // n==2 `faces_join` is the JFKE merge of §3.1 / §6.4:
                    // it finds the shared edge, kills it and its two loops,
                    // and splices the two perimeters into one face.
                    let _ = self.faces_join(&[face_a, face_b]);
                }
                EdgeClass::NoOp => {}
            }
        }

        // §6.5 sub-step 3 — collapse the marked candidates that are still
        // edge-pairs after the merges, gating each on the §5.2 blended angle
        // when the mode demands it.
        if let Some((marked, chain_target)) = candidate_marks {
            self.dissolve_edges_collapse_marked_candidates(marked, chain_target, mode);
            self.pop_vert_op_flag_layer(chain_target);
            self.pop_vert_op_flag_layer(marked);
        }

        DissolveEdgesResult::Ran
    }

    /// §6.3 / §4 — the `use_face_split` pre-pass. Mirrors the dissolve-verts §4
    /// pre-pass in shape but keys the candidate marking off the *untagged*-edge
    /// count rather than input-set membership.
    ///
    /// Sub-pass 1 (§4.1 / §6.3): mark every vert in the mesh whose count of
    /// incident edges *not* carrying the `EDGE_TAG` (input-set) bit is ≤ 2 — a
    /// vert that would become an edge-pair (or less) once the input edges are
    /// dissolved.
    ///
    /// Sub-pass 2 (§4.1): for every marked vert `v` that is **not** already an
    /// edge-pair (it currently has more than two incident edges, so the mark
    /// came from many input edges meeting at `v` rather than from `v` already
    /// being a corner), and every incident face whose corner count is `> 3`,
    /// draw the diagonal `(v_prev, v_next)` provided **neither** `v_prev` nor
    /// `v_next` is itself a marked vert. The cut-off triangle `(v_prev, v,
    /// v_next)` holds `v` as its sole marked corner; the rest of the face
    /// survives as an `(n−1)`-gon. The diagonals are gathered before any split
    /// because the diagonal `v_prev – v_next` never touches `v`, so slicing one
    /// face of `v`'s fan does not change which loop sits at `v` on any other.
    fn dissolve_edges_face_split_pre_pass(&mut self, tagged: crate::op_flags::EdgeOpFlagLayer) {
        const MARK_BIT: u8 = 0;
        // Sub-pass 1 — mark candidate verts and gather them for sub-pass 2's
        // membership test (the diagonal skip uses the *marked* set, §4.1).
        let marked = self.push_vert_op_flag_layer();
        let mut candidates: SmallVec<[VertexKey; 16]> = SmallVec::new();
        let mut vert_iter = MeshVerts::default();
        while let Some(vertex) = vert_iter.next(self) {
            if self.vert_untagged_edge_count_capped(vertex, tagged, 3) <= 2 {
                self.vert_op_flag_set(marked, vertex, MARK_BIT);
                candidates.push(vertex);
            }
        }

        // Sub-pass 2 — gather the (face, v_prev, v_next) diagonals to cut.
        let mut diagonals: SmallVec<[(FaceKey, VertexKey, VertexKey); 8]> = SmallVec::new();
        for &vertex in &candidates {
            // §4.1 — edge-pair verts (≤ 2 incident edges) have no third corner
            // to slice; they are skipped (the §4.4 "already an edge-pair" no-op).
            if self.vertex_disk_count_edges_capped(vertex, 3) <= 2 {
                continue;
            }
            let mut seen_faces: SmallVec<[FaceKey; 8]> = SmallVec::new();
            let mut disk = self.disk_edges(vertex);
            while let Some(edge) = disk.next(self) {
                let mut radial = self.radial_loops(edge);
                while let Some(loop_key) = radial.next(self) {
                    if self.loops.get(loop_key).vertex != vertex {
                        continue;
                    }
                    let face = self.loops.get(loop_key).face;
                    if seen_faces.contains(&face) {
                        continue;
                    }
                    seen_faces.push(face);
                    // §4.1 / §4.4 — already-triangle faces have no n-gon diagonal.
                    if self.face_len(face) <= 3 {
                        continue;
                    }
                    let v_next = self.loops.get(self.loop_next(loop_key)).vertex;
                    let v_prev = self.loops.get(self.loop_prev(loop_key)).vertex;
                    // §4.1 — skip when either diagonal endpoint is itself marked.
                    if self.vert_op_flag_test(marked, v_next, MARK_BIT)
                        || self.vert_op_flag_test(marked, v_prev, MARK_BIT)
                    {
                        continue;
                    }
                    diagonals.push((face, v_prev, v_next));
                }
            }
        }

        for (face, v_prev, v_next) in diagonals {
            // §4.1 — the same shared SFME `face_split` helper the dissolve-verts
            // §4 pre-pass uses (§9 Euler inventory).
            let _ = self.face_split(face, v_prev, v_next);
        }

        self.pop_vert_op_flag_layer(marked);
    }

    /// §4.1 / §6.3 helper — count `v`'s incident edges *not* carrying the
    /// `EDGE_TAG` (input-set) bit, stopping at `cap`. The §6.3 candidate test
    /// marks a vert when this count is ≤ 2.
    fn vert_untagged_edge_count_capped(
        &self,
        v: VertexKey,
        tagged: crate::op_flags::EdgeOpFlagLayer,
        cap: usize,
    ) -> usize {
        let mut count = 0usize;
        let mut disk = self.disk_edges(v);
        while count < cap {
            let Some(edge) = disk.next(self) else { break };
            if !self.edge_op_flag_test(tagged, edge, 0) {
                count += 1;
            }
        }
        count
    }

    /// §6.1 — extend the `EDGE_TAG` set along the chains rooted at each input
    /// edge. An input edge is a chain edge when it flanks two distinct faces
    /// and at least one endpoint is an edge-pair (so removing the input edge
    /// pulls the unselected chain edges out with it). From each edge-pair
    /// endpoint the disk-next walk tags every edge along the chain up to (but
    /// not including the edge past) the first non-edge-pair terminus, so the
    /// §5.1 tagged-edge count at the terminus reflects the chain it caps.
    fn dissolve_edges_tag_chains(
        &mut self,
        edges: &[EdgeKey],
        tagged: crate::op_flags::EdgeOpFlagLayer,
    ) {
        for &edge in edges {
            if self.edges.try_get(edge).is_none() {
                continue;
            }
            if !matches!(
                self.dissolve_edges_classify(edge),
                EdgeClass::ManifoldPair { .. }
            ) {
                continue;
            }
            let endpoints = self.edge_vert_keys(edge);
            for v_endpoint in endpoints {
                if self.vertex_disk_count_edges_capped(v_endpoint, 3) != 2 {
                    continue;
                }
                // Walk from the edge-pair endpoint, tagging each chain edge,
                // until the terminus (a vert with more than two edges) is
                // reached.
                let mut current = v_endpoint;
                let mut came_from = edge;
                loop {
                    let Some(next_edge) =
                        self.vertex_disk_iter(current).find(|&e| e != came_from)
                    else {
                        break;
                    };
                    if self.edge_op_flag_test(tagged, next_edge, 0) {
                        // Already tagged: end of this chain segment / loop guard.
                        break;
                    }
                    self.edge_op_flag_set(tagged, next_edge, 0);
                    let Some(next_vert) = self.edges.get(next_edge).other_vertex(current) else {
                        break;
                    };
                    if self.vertex_disk_count_edges_capped(next_vert, 3) != 2 {
                        break;
                    }
                    current = next_vert;
                    came_from = next_edge;
                }
            }
        }
    }

    /// §6.2 / §5.1 — mark the `use_verts` collapse candidates against the
    /// pre-merge topology. For every input edge that flanks two distinct
    /// faces, both endpoints are tested independently; each survives or is
    /// redirected through the three §5.1 filters before its `VERT_MARK` is
    /// set. `tagged` carries the `EDGE_TAG` (input-set membership) bit;
    /// `marked` receives the `VERT_MARK` collapse bit.
    fn dissolve_edges_mark_use_verts_candidates(
        &mut self,
        edges: &[EdgeKey],
        tagged: crate::op_flags::EdgeOpFlagLayer,
        chain: crate::op_flags::EdgeOpFlagLayer,
        marked: crate::op_flags::VertOpFlagLayer,
        chain_target: crate::op_flags::VertOpFlagLayer,
    ) {
        for &edge in edges {
            if self.edges.try_get(edge).is_none() {
                continue;
            }
            let EdgeClass::ManifoldPair { face_a, face_b } = self.dissolve_edges_classify(edge)
            else {
                continue;
            };
            let both_tris = self.face_len(face_a) == 3 && self.face_len(face_b) == 3;
            let endpoints = self.edge_vert_keys(edge);
            for v_edge in endpoints {
                // §5.1 first sub-rule — triangle-to-quad anti-degeneration
                // guard. (use_face_split is always false on this path, so the
                // "not applied when use_face_split is on" carve-out is moot.)
                if both_tris && self.vert_tagged_edge_count_capped(v_edge, tagged, 2) == 1 {
                    continue;
                }

                // §5.1 second sub-rule — if v_edge is already an edge-pair,
                // walk the chain to its terminus and mark that instead. A
                // walk that runs into a *different* input chain edge mid-walk
                // (intersecting selections, §8) yields None → skip.
                let reached_via_chain_walk = self.vertex_disk_count_edges_capped(v_edge, 3) == 2;
                let v_target = if reached_via_chain_walk {
                    match self.dissolve_edges_find_chain_end(v_edge, edge, chain) {
                        Some(v) => v,
                        None => continue,
                    }
                } else {
                    v_edge
                };

                // §5.1 third sub-rule — T-junction / incidental-touch filter:
                // skip a vert that touches an unselected face unless exactly
                // one tagged edge ends at it (the canonical T-on-a-loop-cut).
                if self.vert_touches_unselected_face(v_target, tagged)
                    && self.vert_tagged_edge_count_capped(v_target, tagged, 2) != 1
                {
                    continue;
                }

                self.vert_op_flag_set(marked, v_target, 0);
                if reached_via_chain_walk {
                    // §6.5 chain override marker — this candidate is the
                    // terminus of a selected chain, so its collapse bypasses
                    // the §5.2 angle test (§5.2.6).
                    self.vert_op_flag_set(chain_target, v_target, 0);
                }
            }
        }
    }

    /// §5.1 second sub-rule — walk the edge-pair chain starting at `v_start`,
    /// leaving along its incident edge other than `arrived_via`, until a vert
    /// with more than two incident edges is reached; return that terminus.
    ///
    /// Returns `None` if the walk steps onto a *different* input chain edge
    /// (one carrying the `EDGE_CHAIN` bit other than the one we started from,
    /// §8) — the caller then leaves the vert unmarked to avoid double-marking
    /// a shared chain between two intersecting selections.
    fn dissolve_edges_find_chain_end(
        &self,
        v_start: VertexKey,
        arrived_via: EdgeKey,
        chain: crate::op_flags::EdgeOpFlagLayer,
    ) -> Option<VertexKey> {
        let mut current = v_start;
        let mut came_from = arrived_via;
        loop {
            // The next chain edge is the one in `current`'s 2-edge disk that
            // we did not arrive on.
            let next_edge = self
                .vertex_disk_iter(current)
                .find(|&e| e != came_from)?;
            if next_edge != arrived_via && self.edge_op_flag_test(chain, next_edge, 0) {
                // Ran into a different selected chain edge mid-walk: abandon.
                return None;
            }
            let next_vert = self.edges.get(next_edge).other_vertex(current)?;
            if self.vertex_disk_count_edges_capped(next_vert, 3) != 2 {
                return Some(next_vert);
            }
            current = next_vert;
            came_from = next_edge;
        }
    }

    /// §6.5 sub-step 3 / §5.3 — the two-sweep `use_verts` post-pass. An
    /// angle-test sweep gathers the candidates that are still edge-pairs and
    /// pass the §5.2 gate (without touching topology), then a collapse sweep
    /// acts on the survivors. Splitting the sweeps is what makes every
    /// blended angle a *post-merge* sample uncontaminated by an earlier
    /// collapse (§6.5 snapshot-then-act).
    ///
    /// - [`UseVertsMode::CollapseAll`] (the `dissolve_all` fast path, §6.5
    ///   first bullet) gathers every still-edge-pair candidate with no angle
    ///   math.
    /// - [`UseVertsMode::Gated`] keeps a candidate iff it is a chain terminus
    ///   (`chain_target`, the §6.5 chain override) or its blended angle is
    ///   strictly below `angle_threshold − ε` (§5.2.6).
    fn dissolve_edges_collapse_marked_candidates(
        &mut self,
        marked: crate::op_flags::VertOpFlagLayer,
        chain_target: crate::op_flags::VertOpFlagLayer,
        mode: UseVertsMode,
    ) {
        let mut to_collapse: SmallVec<[VertexKey; 16]> = SmallVec::new();
        let mut iter = MeshVerts::default();
        while let Some(vertex) = iter.next(self) {
            if !self.vert_op_flag_test(marked, vertex, 0) {
                continue;
            }
            if self.vertex_disk_count_edges_capped(vertex, 3) != 2 {
                // The merges may have demoted this candidate below edge-pair
                // status (§6.5 angle-test sweep first guard).
                continue;
            }
            let keep = match mode {
                UseVertsMode::Off => false,
                UseVertsMode::CollapseAll => true,
                UseVertsMode::Gated { angle_threshold } => {
                    if self.vert_op_flag_test(chain_target, vertex, 0) {
                        // §6.5 chain override — chain termini always collapse.
                        true
                    } else {
                        let angle = self.blended_edge_pair_angle(vertex);
                        angle < angle_threshold - ANGLE_EPS
                    }
                }
            };
            if keep {
                to_collapse.push(vertex);
            }
        }

        for vertex in to_collapse {
            // Re-test: an earlier collapse may have demoted this neighbour
            // out of edge-pair status (§5.3 / §6.5).
            if self.vertices.try_get(vertex).is_none() {
                continue;
            }
            if self.vertex_disk_count_edges_capped(vertex, 3) != 2 {
                continue;
            }
            let mut disk = self.disk_edges(vertex);
            let Some(e_kill) = disk.next(self) else {
                continue;
            };
            let _ = self.vert_collapse_edge(e_kill, vertex);
        }
    }

    /// §5.2 / §9 helper — the blended edge-pair angle at a candidate vert,
    /// sampled live on the current post-merge mesh (§5.2.0; nothing is
    /// captured pre-merge).
    ///
    /// `vertex` is known to be an edge-pair (the caller re-tested it). The
    /// baseline `α_raw = π − ∠(v_a, v, v_b)` is always formed from the two
    /// surviving edges' far-endpoint positions (§5.2.1). The face-pair branch
    /// (§5.2.2) is taken *only* when the vert's primary edge `v.e` still
    /// flanks two distinct faces post-merge; then both flanking faces' normals
    /// are recomputed live (their cached normals are stale after the merges),
    /// the raw-blend factor `r = clamp(−dot(no_a, no_b), 0, 1)` and the
    /// normal-axis angle `α_normal = π − ∠axis(v_a, v, v_b, v.no)` are formed,
    /// and the result is `lerp(α_raw, α_normal, r)`. When the branch is not
    /// taken — single-face perimeter edge, wire, or non-manifold `v.e` — the
    /// result is exactly `α_raw` (§5.2.2 no-live-two-face-edge fallback), which
    /// is the common inverse-of-a-cut case (§5.2.5).
    fn blended_edge_pair_angle(&self, vertex: VertexKey) -> f32 {
        let pi = std::f32::consts::PI;

        // The two surviving incident edges. The first one emitted by the disk
        // iterator is the primary edge `v.e` (the disk-cycle entry edge).
        let mut disk = self.vertex_disk_iter(vertex);
        let primary_edge = disk.next().expect("edge-pair vert has a primary edge");
        let second_edge = disk.next().expect("edge-pair vert has a second edge");

        let far_a = self
            .edges
            .get(primary_edge)
            .other_vertex(vertex)
            .expect("primary edge has a far endpoint");
        let far_b = self
            .edges
            .get(second_edge)
            .other_vertex(vertex)
            .expect("second edge has a far endpoint");

        let position = self.vertex_position(vertex);
        let position_a = self.vertex_position(far_a);
        let position_b = self.vertex_position(far_b);

        // §5.2.1 — baseline, always computed.
        let raw_angle = pi - corner_angle(position_a, position, position_b);

        // §5.2.2 — face-pair branch, taken iff `v.e` currently flanks two
        // distinct faces. If it does not, `α = α_raw` and no normal is read.
        let Some((face_a, face_b)) = self.dissolve_edges_edge_face_pair(primary_edge) else {
            return raw_angle;
        };

        // §5.2.2 / §9 note — derive both normals from the faces' live corner
        // positions; the cached per-face normals are stale after the merges.
        let normal_a = self.face_compute_normal(face_a);
        let normal_b = self.face_compute_normal(face_b);
        let raw_blend = (-normal_a.dot(normal_b)).clamp(0.0, 1.0);

        // §5.2.3 — unsigned corner angle in the plane ⊥ `v.no`.
        let axis = self.vertex_normal(vertex);
        let normal_angle = pi - corner_angle_on_axis(position_a, position, position_b, axis);

        // §5.2.2 — lerp toward the normal-axis angle as the dihedral folds.
        raw_angle + raw_blend * (normal_angle - raw_angle)
    }

    /// §5.2.2 helper — the two distinct faces flanking `edge` in the current
    /// post-merge mesh, or `None` when the radial cycle is not exactly two
    /// loops belonging to two distinct faces (wire, boundary, single-face
    /// perimeter edge, self-radial, or non-manifold). Shares the radial-cycle
    /// inspection shape with [`Self::dissolve_edges_classify`].
    fn dissolve_edges_edge_face_pair(&self, edge: EdgeKey) -> Option<(FaceKey, FaceKey)> {
        match self.dissolve_edges_classify(edge) {
            EdgeClass::ManifoldPair { face_a, face_b } => Some((face_a, face_b)),
            EdgeClass::NoOp => None,
        }
    }

    /// Count `v`'s incident edges carrying the `EDGE_TAG` bit, stopping at
    /// `cap`. Used by the §5.1 anti-degeneration ("exactly one tagged edge")
    /// and T-junction ("more than one tagged edge") predicates.
    fn vert_tagged_edge_count_capped(
        &self,
        v: VertexKey,
        tagged: crate::op_flags::EdgeOpFlagLayer,
        cap: usize,
    ) -> usize {
        let mut count = 0usize;
        let mut disk = self.disk_edges(v);
        while count < cap {
            let Some(edge) = disk.next(self) else { break };
            if self.edge_op_flag_test(tagged, edge, 0) {
                count += 1;
            }
        }
        count
    }

    /// §5.1 third sub-rule helper — true iff `v` is a corner of at least one
    /// *unselected* face: a face neither of whose two perimeter edges meeting
    /// at `v` carries the `EDGE_TAG` (input-set) bit.
    fn vert_touches_unselected_face(
        &self,
        v: VertexKey,
        tagged: crate::op_flags::EdgeOpFlagLayer,
    ) -> bool {
        let mut disk = self.disk_edges(v);
        while let Some(edge) = disk.next(self) {
            let mut radial = self.radial_loops(edge);
            while let Some(loop_key) = radial.next(self) {
                // Find the loop sitting at `v` on this face so its two
                // perimeter edges at `v` can be tested.
                if self.loops.get(loop_key).vertex != v {
                    continue;
                }
                let edge_here = self.loops.get(loop_key).edge;
                let edge_prev = self.loops.get(self.loop_prev(loop_key)).edge;
                let here_tagged =
                    !edge_here.is_null() && self.edge_op_flag_test(tagged, edge_here, 0);
                let prev_tagged =
                    !edge_prev.is_null() && self.edge_op_flag_test(tagged, edge_prev, 0);
                if !here_tagged && !prev_tagged {
                    return true;
                }
            }
        }
        false
    }

    /// §3 classification of a single input edge against the current
    /// topology. Returns `ManifoldPair` only when the radial cycle has
    /// exactly two loops belonging to two distinct faces; every other shape
    /// (wire, boundary, self-radial, non-manifold) is `NoOp`.
    fn dissolve_edges_classify(&self, edge: EdgeKey) -> EdgeClass {
        let mut radial = self.radial_loops(edge);
        let Some(head_loop) = radial.next(self) else {
            // Radial cycle empty: wire edge (§3 row 1).
            return EdgeClass::NoOp;
        };
        let Some(second_loop) = radial.next(self) else {
            // Radial cycle length 1: boundary edge (§3 row 2).
            return EdgeClass::NoOp;
        };
        if radial.next(self).is_some() {
            // Radial cycle length ≥ 3: non-manifold edge (§3 row 5).
            return EdgeClass::NoOp;
        }
        // §3.1 / §11 survivor rule: the merged face keeps the per-face
        // customdata of the face on the dissolved edge's primary loop `e.l` —
        // the **second-built** of the two flanking faces. In this crate's
        // radial cycle the *first-built* face's loop sits at the radial head
        // (`radial_loops` emits it first); the second-built face's loop is the
        // head's radial-next, i.e. `second_loop` here. So the survivor `face_a`
        // is `second_loop`'s face, and `face_b` is the head loop's face.
        let head_face = self.loops.get(head_loop).face;
        let second_face = self.loops.get(second_loop).face;
        if head_face == second_face {
            // Radial cycle length 2 but both loops on the same face: a
            // self-radial fin (§3 row 4).
            return EdgeClass::NoOp;
        }
        EdgeClass::ManifoldPair {
            face_a: second_face,
            face_b: head_face,
        }
    }
}

/// §5.2.1 — the corner angle `∠(corner_a, apex, corner_b)`: the unsigned angle
/// at `apex` between the two edge directions, in `[0, π]`. Uses glam's
/// `angle_between`, the safe angle-between-vectors primitive that clamps its
/// argument and returns a finite angle rather than a NaN on degenerate input
/// (§5.2.4).
fn corner_angle(corner_a: Vec3, apex: Vec3, corner_b: Vec3) -> f32 {
    (corner_a - apex).angle_between(corner_b - apex)
}

/// §5.2.3 — the corner angle at `apex` after projecting both edge directions
/// into the plane through the origin perpendicular to `axis`. The result is
/// the unsigned angle between the projected directions, in `[0, π]`; there is
/// no winding sign. `axis` is taken as a unit vector (`v.no` is stored
/// normalised). A near-zero projected direction (an incident edge nearly
/// parallel to `axis`) flows through the same safe angle primitive as
/// [`corner_angle`] (§5.2.4).
fn corner_angle_on_axis(corner_a: Vec3, apex: Vec3, corner_b: Vec3, axis: Vec3) -> f32 {
    let direction_a = corner_a - apex;
    let direction_b = corner_b - apex;
    let projected_a = direction_a - direction_a.dot(axis) * axis;
    let projected_b = direction_b - direction_b.dot(axis) * axis;
    projected_a.angle_between(projected_b)
}

#[cfg(test)]
mod tests {
    use glam::Vec3;

    use crate::loopmesh::*;

    /// Build a 3x3 face grid of quads on the unit lattice. Returns
    /// `(verts[16], faces[9])` where `verts[r * 4 + c]` is the lattice
    /// point at column `c`, row `r`, and `faces[r * 3 + c]` is the quad
    /// whose lower-left corner is `verts[r * 4 + c]`. CCW winding.
    fn build_grid_3x3(mesh: &mut LoopMesh) -> (Vec<VertexKey>, Vec<FaceKey>) {
        let mut verts: Vec<VertexKey> = Vec::with_capacity(16);
        for r in 0..4 {
            for c in 0..4 {
                verts.push(mesh.create_vertex(Vec3::new(c as f32, r as f32, 0.0)));
            }
        }
        let mut faces: Vec<FaceKey> = Vec::new();
        for r in 0..3 {
            for c in 0..3 {
                let v_bl = verts[r * 4 + c];
                let v_br = verts[r * 4 + c + 1];
                let v_tr = verts[(r + 1) * 4 + c + 1];
                let v_tl = verts[(r + 1) * 4 + c];
                let e_b = mesh.create_edge([v_bl, v_br], EdgeCreateFlags::NO_DUPLICATE);
                let e_r = mesh.create_edge([v_br, v_tr], EdgeCreateFlags::NO_DUPLICATE);
                let e_t = mesh.create_edge([v_tr, v_tl], EdgeCreateFlags::NO_DUPLICATE);
                let e_l = mesh.create_edge([v_tl, v_bl], EdgeCreateFlags::NO_DUPLICATE);
                faces.push(mesh.face_create(
                    &[v_bl, v_br, v_tr, v_tl],
                    &[e_b, e_r, e_t, e_l],
                    FaceCreateFlags::NONE,
                ));
            }
        }
        (verts, faces)
    }

    /// Two triangles sharing the diagonal edge {v1, v2}. Square corners are
    /// v0=(0,0), v1=(1,0), v2=(0,1), v3=(1,1); the diagonal runs v1→v2.
    /// Triangle A = (v0, v1, v2); triangle B = (v1, v3, v2).
    fn build_two_tris_shared_diagonal(
        mesh: &mut LoopMesh,
    ) -> (Vec<VertexKey>, EdgeKey, Vec<FaceKey>) {
        let v: Vec<VertexKey> = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)]
            .into_iter()
            .map(|(x, y)| mesh.create_vertex(Vec3::new(x, y, 0.0)))
            .collect();
        let e_01 = mesh.create_edge([v[0], v[1]], EdgeCreateFlags::NO_DUPLICATE);
        let e_12 = mesh.create_edge([v[1], v[2]], EdgeCreateFlags::NO_DUPLICATE);
        let e_20 = mesh.create_edge([v[2], v[0]], EdgeCreateFlags::NO_DUPLICATE);
        let e_13 = mesh.create_edge([v[1], v[3]], EdgeCreateFlags::NO_DUPLICATE);
        let e_32 = mesh.create_edge([v[3], v[2]], EdgeCreateFlags::NO_DUPLICATE);
        let f_a = mesh.face_create(
            &[v[0], v[1], v[2]],
            &[e_01, e_12, e_20],
            FaceCreateFlags::NONE,
        );
        let f_b = mesh.face_create(
            &[v[1], v[3], v[2]],
            &[e_13, e_32, e_12],
            FaceCreateFlags::NONE,
        );
        (v, e_12, vec![f_a, f_b])
    }

    /// Single triangle (3 boundary edges) at the origin.
    fn build_triangle(mesh: &mut LoopMesh) -> (Vec<VertexKey>, Vec<EdgeKey>, FaceKey) {
        let v: Vec<VertexKey> = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]
            .into_iter()
            .map(|(x, y)| mesh.create_vertex(Vec3::new(x, y, 0.0)))
            .collect();
        let e_01 = mesh.create_edge([v[0], v[1]], EdgeCreateFlags::NO_DUPLICATE);
        let e_12 = mesh.create_edge([v[1], v[2]], EdgeCreateFlags::NO_DUPLICATE);
        let e_20 = mesh.create_edge([v[2], v[0]], EdgeCreateFlags::NO_DUPLICATE);
        let f = mesh.face_create(&[v[0], v[1], v[2]], &[e_01, e_12, e_20], FaceCreateFlags::NONE);
        (v, vec![e_01, e_12, e_20], f)
    }

    /// `n` triangles all sharing the spine edge {v0, v1} (a radial-`n`
    /// non-manifold "book"). Triangle `i` is (v0, v1, apex_i).
    fn build_nonmanifold_book(mesh: &mut LoopMesh, n: usize) -> (VertexKey, VertexKey, EdgeKey) {
        let v0 = mesh.create_vertex(Vec3::new(0.0, 0.0, 0.0));
        let v1 = mesh.create_vertex(Vec3::new(1.0, 0.0, 0.0));
        let spine = mesh.create_edge([v0, v1], EdgeCreateFlags::NO_DUPLICATE);
        for i in 0..n {
            let angle = std::f32::consts::PI * (i as f32 + 1.0) / (n as f32 + 1.0);
            let apex = mesh.create_vertex(Vec3::new(0.5, angle.cos(), angle.sin()));
            let e_1a = mesh.create_edge([v1, apex], EdgeCreateFlags::NO_DUPLICATE);
            let e_a0 = mesh.create_edge([apex, v0], EdgeCreateFlags::NO_DUPLICATE);
            mesh.face_create(&[v0, v1, apex], &[spine, e_1a, e_a0], FaceCreateFlags::NONE);
        }
        (v0, v1, spine)
    }

    #[test]
    fn dissolve_edges_interior_edge_between_two_quads() {
        // Default-parameter manifold face-pair (§3 row 3, §7 row 3): the
        // shared edge between two grid quads dissolves, merging the two
        // quads into a hexagon. -1 E, -1 F, 0 V.
        let mut mesh = LoopMesh::new();
        let (verts, _faces) = build_grid_3x3(&mut mesh);

        let v0 = mesh.totvert();
        let e0 = mesh.totedge();
        let f0 = mesh.totface();
        assert_eq!((v0, e0, f0), (16, 24, 9));

        // Vertical interior edge between faces[0] and faces[1]: verts[1]-verts[5].
        let edge = mesh
            .query_edge_exists(verts[1], verts[5])
            .expect("interior edge {1,5} present");

        mesh.dissolve_edges(&[edge], &Default::default());

        assert_eq!(mesh.totvert(), v0, "0 verts removed (§7 row 3)");
        assert_eq!(mesh.totedge(), e0 - 1, "-1 edge (§7 row 3)");
        assert_eq!(mesh.totface(), f0 - 1, "-1 face (§7 row 3)");
        assert!(
            mesh.query_edge_exists(verts[1], verts[5]).is_none(),
            "the dissolved edge is gone"
        );
        // 4 + 4 corners minus the two consumed by the shared edge = 6.
        let merged = mesh
            .face_keys()
            .into_iter()
            .find(|&f| mesh.face_len(f) == 6)
            .expect("a hexagonal merged face exists");
        assert_eq!(mesh.face_len(merged), 6);
        mesh.validate_topology();
    }

    #[test]
    fn dissolve_edges_diagonal_of_two_triangles_makes_quad() {
        // Inverse-of-triangulate base case (§7 row 3 on the smallest
        // non-trivial face pair): two triangles sharing a diagonal merge
        // into a quad. Verts unchanged under default parameters.
        let mut mesh = LoopMesh::new();
        let (_verts, diagonal, _faces) = build_two_tris_shared_diagonal(&mut mesh);

        let v0 = mesh.totvert();
        let e0 = mesh.totedge();
        let f0 = mesh.totface();
        assert_eq!((v0, e0, f0), (4, 5, 2));

        mesh.dissolve_edges(&[diagonal], &Default::default());

        assert_eq!(mesh.totvert(), v0, "0 verts removed");
        assert_eq!(mesh.totedge(), e0 - 1, "-1 edge (the diagonal)");
        assert_eq!(mesh.totface(), f0 - 1, "-1 face (two tris → one quad)");

        let merged = mesh.face_keys();
        assert_eq!(merged.len(), 1);
        assert_eq!(mesh.face_len(merged[0]), 4, "3 + 3 - 2 = 4 corners");
        mesh.validate_topology();
    }

    #[test]
    fn dissolve_edges_merged_face_keeps_second_built_face_cd() {
        // §3.1 / §11 — the merged face retains the per-face customdata of the
        // face on the dissolved edge's primary loop `e.l`, which under the
        // standard build order is the *second-built* of the two flanking
        // faces. `build_two_tris_shared_diagonal` creates triangle A first and
        // triangle B second, both sharing the diagonal `e_12`, so the survivor
        // must carry triangle B's color — not triangle A's.
        let mut mesh = LoopMesh::new();
        let (_verts, diagonal, faces) = build_two_tris_shared_diagonal(&mut mesh);
        let first_built = faces[0];
        let second_built = faces[1];

        let color = mesh
            .register_face_attribute(ElementAttributeDesc::numeric(
                AttributeKey::custom("face_color"),
                Format::new(ComponentType::F32, 3),
            ))
            .unwrap();
        let first_color = [1.0_f32, 0.0, 0.0];
        let second_color = [0.0_f32, 1.0, 0.0];
        mesh.face_attribs_mut()
            .set_vec3(color, first_built.as_usize(), first_color);
        mesh.face_attribs_mut()
            .set_vec3(color, second_built.as_usize(), second_color);

        mesh.dissolve_edges(&[diagonal], &Default::default());

        let merged = mesh.face_keys();
        assert_eq!(merged.len(), 1, "two tris collapse to one quad");
        let got = mesh
            .face_attribs()
            .get_vec3(color, merged[0].as_usize())
            .expect("merged face carries the customdata layer");

        // Direction-explicit: the survivor is the second-built face's CD.
        for axis in 0..3 {
            assert!(
                (got[axis] - second_color[axis]).abs() < 1e-6,
                "merged face must keep the second-built (`e.l`) face's CD: \
                 got {got:?}, expected {second_color:?} (regression to \
                 first-built {first_color:?} would fail here)"
            );
        }
        assert!(
            (got[0] - first_color[0]).abs() > 1e-6 || (got[1] - first_color[1]).abs() > 1e-6,
            "merged face must NOT keep the first-built face's CD {first_color:?}"
        );
        mesh.validate_topology();
    }

    #[test]
    fn dissolve_edges_wire_edge_is_noop() {
        // §3 row 1: a wire edge (empty radial cycle) fails the face-pair
        // test; the merge is skipped and the edge survives.
        let mut mesh = LoopMesh::new();
        let (_verts, _faces) = build_grid_3x3(&mut mesh);
        // Add a free-standing wire edge between two fresh verts.
        let w0 = mesh.create_vertex(Vec3::new(10.0, 0.0, 0.0));
        let w1 = mesh.create_vertex(Vec3::new(11.0, 0.0, 0.0));
        let wire = mesh.create_edge([w0, w1], EdgeCreateFlags::NO_DUPLICATE);

        let v0 = mesh.totvert();
        let e0 = mesh.totedge();
        let f0 = mesh.totface();

        mesh.dissolve_edges(&[wire], &Default::default());

        assert_eq!((mesh.totvert(), mesh.totedge(), mesh.totface()), (v0, e0, f0));
        assert!(
            mesh.query_edge_exists(w0, w1).is_some(),
            "wire edge survives (BMesh does not kill wire edges directly)"
        );
        mesh.validate_topology();
    }

    #[test]
    fn dissolve_edges_boundary_edge_is_noop() {
        // §3 row 2: a boundary edge (radial length 1) fails the face-pair
        // test; the merge is skipped and the edge survives.
        let mut mesh = LoopMesh::new();
        let (verts, edges, _f) = build_triangle(&mut mesh);

        let v0 = mesh.totvert();
        let e0 = mesh.totedge();
        let f0 = mesh.totface();

        mesh.dissolve_edges(&[edges[0]], &Default::default());

        assert_eq!((mesh.totvert(), mesh.totedge(), mesh.totface()), (v0, e0, f0));
        assert!(
            mesh.query_edge_exists(verts[0], verts[1]).is_some(),
            "boundary edge survives"
        );
        mesh.validate_topology();
    }

    #[test]
    fn dissolve_edges_two_disjoint_interior_edges() {
        // Multi-edge input, disjoint pairs (§8 first bullet): two
        // non-adjacent interior edges, each between a distinct quad pair.
        // The merges are topologically independent: -2 E, -2 F, 0 V.
        let mut mesh = LoopMesh::new();
        let (verts, _faces) = build_grid_3x3(&mut mesh);

        let v0 = mesh.totvert();
        let e0 = mesh.totedge();
        let f0 = mesh.totface();

        // Edge {1,5} sits between faces[0]/faces[1] (bottom row).
        // Edge {10,14} sits between faces[6]/faces[7] (top row) — disjoint.
        let edge_a = mesh.query_edge_exists(verts[1], verts[5]).expect("edge {1,5}");
        let edge_b = mesh.query_edge_exists(verts[10], verts[14]).expect("edge {10,14}");

        mesh.dissolve_edges(&[edge_a, edge_b], &Default::default());

        assert_eq!(mesh.totvert(), v0, "0 verts removed");
        assert_eq!(mesh.totedge(), e0 - 2, "-2 edges");
        assert_eq!(mesh.totface(), f0 - 2, "-2 faces");
        mesh.validate_topology();
    }

    #[test]
    fn dissolve_edges_two_disjoint_interior_edges_order_independent() {
        // Same disjoint-pair input as above but reversed order: the result
        // is identical, pinning the order-independence claim of §8.
        let mut mesh = LoopMesh::new();
        let (verts, _faces) = build_grid_3x3(&mut mesh);
        let v0 = mesh.totvert();
        let e0 = mesh.totedge();
        let f0 = mesh.totface();

        let edge_a = mesh.query_edge_exists(verts[1], verts[5]).expect("edge {1,5}");
        let edge_b = mesh.query_edge_exists(verts[10], verts[14]).expect("edge {10,14}");

        mesh.dissolve_edges(&[edge_b, edge_a], &Default::default());

        assert_eq!(mesh.totvert(), v0);
        assert_eq!(mesh.totedge(), e0 - 2);
        assert_eq!(mesh.totface(), f0 - 2);
        mesh.validate_topology();
    }

    #[test]
    fn dissolve_edges_two_edges_sharing_endpoint() {
        // Multi-edge input, shared endpoint (§6.4 / §8 "two adjacent input
        // edges sharing a vert"): two interior edges meeting at the centre
        // vert of the grid. The first merge reshapes the face pair the
        // second sees; the per-edge dispatch re-classifies afresh, so both
        // merges run and consume one edge / one face each.
        let mut mesh = LoopMesh::new();
        let (verts, _faces) = build_grid_3x3(&mut mesh);
        let v0 = mesh.totvert();
        let e0 = mesh.totedge();
        let f0 = mesh.totface();

        // Centre vert is verts[5]. Two interior edges meeting at it:
        // {1,5} (between faces[0]/faces[1]) and {5,6} (between faces[1]/faces[2]).
        let edge_a = mesh.query_edge_exists(verts[1], verts[5]).expect("edge {1,5}");
        let edge_b = mesh.query_edge_exists(verts[5], verts[6]).expect("edge {5,6}");

        mesh.dissolve_edges(&[edge_a, edge_b], &Default::default());

        // Two distinct face pairs each merge: faces[0]+faces[1], then the
        // result + faces[2] is NOT what happens here — edge_b's faces are
        // faces[1] (now merged with faces[0]) and faces[2]. The second
        // merge joins that merged face with faces[2]. Net over both edges:
        // -2 E, -2 F, 0 V.
        assert_eq!(mesh.totvert(), v0, "0 verts removed under default params");
        assert_eq!(mesh.totedge(), e0 - 2, "-2 edges (both dissolved)");
        assert_eq!(mesh.totface(), f0 - 2, "-2 faces (both merges ran)");
        assert!(mesh.query_edge_exists(verts[1], verts[5]).is_none());
        assert!(mesh.query_edge_exists(verts[5], verts[6]).is_none());
        mesh.validate_topology();
    }

    #[test]
    fn dissolve_edges_non_manifold_radial_3_is_noop() {
        // §3 row 5: three faces share the spine edge (radial length 3);
        // the face-pair test fails and the merge is skipped.
        let mut mesh = LoopMesh::new();
        let (s0, s1, spine) = build_nonmanifold_book(&mut mesh, 3);

        let v0 = mesh.totvert();
        let e0 = mesh.totedge();
        let f0 = mesh.totface();

        mesh.dissolve_edges(&[spine], &Default::default());

        assert_eq!((mesh.totvert(), mesh.totedge(), mesh.totface()), (v0, e0, f0));
        assert!(
            mesh.query_edge_exists(s0, s1).is_some(),
            "non-manifold spine edge survives"
        );
        mesh.validate_topology();
    }

    #[test]
    fn dissolve_edges_closed_loop_boundary_perimeter_is_noop() {
        // §8 "an entire closed loop of input edges", boundary-perimeter
        // sub-case: the three edges of a lone triangle are all boundary
        // (radial length 1), so every per-edge dispatch is a no-op
        // regardless of iteration order.
        let mut mesh = LoopMesh::new();
        let (_verts, edges, _f) = build_triangle(&mut mesh);
        let v0 = mesh.totvert();
        let e0 = mesh.totedge();
        let f0 = mesh.totface();

        mesh.dissolve_edges(&edges, &Default::default());

        assert_eq!((mesh.totvert(), mesh.totedge(), mesh.totface()), (v0, e0, f0));
        mesh.validate_topology();
    }

    #[test]
    fn dissolve_edges_empty_input_is_noop() {
        let mut mesh = LoopMesh::new();
        let (_verts, _faces) = build_grid_3x3(&mut mesh);
        let v0 = mesh.totvert();
        let e0 = mesh.totedge();
        let f0 = mesh.totface();

        mesh.dissolve_edges(&[], &Default::default());

        assert_eq!((mesh.totvert(), mesh.totedge(), mesh.totface()), (v0, e0, f0));
        mesh.validate_topology();
    }

    // ---- use_verts post-pass (§5) ----

    /// Build a 2x2 face grid of quads on a 3x3 vertex lattice. Returns
    /// `(verts[9], faces[4])`. Vert numbering is row-major `verts[r * 3 + c]`
    /// for `r,c in 0..3`; `verts[1]` is mid-bottom `(1,0)`, `verts[4]` the
    /// 4-valence centre `(1,1)`, `verts[5]` mid-right `(2,1)`. Faces are
    /// `faces[r * 2 + c]` whose lower-left corner is `verts[r * 3 + c]`. This
    /// matches the A/B scaffold's `Fixture::grid_3x3`.
    fn build_face_grid_2x2(mesh: &mut LoopMesh) -> (Vec<VertexKey>, Vec<FaceKey>) {
        let mut verts: Vec<VertexKey> = Vec::with_capacity(9);
        for r in 0..3 {
            for c in 0..3 {
                verts.push(mesh.create_vertex(Vec3::new(c as f32, r as f32, 0.0)));
            }
        }
        let mut faces: Vec<FaceKey> = Vec::new();
        for r in 0..2 {
            for c in 0..2 {
                let v_bl = verts[r * 3 + c];
                let v_br = verts[r * 3 + c + 1];
                let v_tr = verts[(r + 1) * 3 + c + 1];
                let v_tl = verts[(r + 1) * 3 + c];
                let e_b = mesh.create_edge([v_bl, v_br], EdgeCreateFlags::NO_DUPLICATE);
                let e_r = mesh.create_edge([v_br, v_tr], EdgeCreateFlags::NO_DUPLICATE);
                let e_t = mesh.create_edge([v_tr, v_tl], EdgeCreateFlags::NO_DUPLICATE);
                let e_l = mesh.create_edge([v_tl, v_bl], EdgeCreateFlags::NO_DUPLICATE);
                faces.push(mesh.face_create(
                    &[v_bl, v_br, v_tr, v_tl],
                    &[e_b, e_r, e_t, e_l],
                    FaceCreateFlags::NONE,
                ));
            }
        }
        (verts, faces)
    }

    fn use_verts(angle_threshold: f32) -> DissolveEdgesParams {
        DissolveEdgesParams {
            use_verts: true,
            use_face_split: false,
            angle_threshold,
        }
    }

    /// §5 headline (`use_verts = true`, wide threshold): dissolving edge
    /// `{1,4}` of the 2x2 face grid merges faces 0 and 1 into a hexagon, then
    /// the post-pass collapses the endpoint that became edge-pair (vert 1)
    /// while leaving the endpoint that is still 3-valent (vert 4). Net
    /// `-1 V, -2 E, -1 F` (V=9→8, E=12→10, F=4→3).
    #[test]
    fn dissolve_edges_use_verts_true_collapses_edge_pair_endpoint() {
        let mut mesh = LoopMesh::new();
        let (verts, _faces) = build_face_grid_2x2(&mut mesh);
        assert_eq!((mesh.totvert(), mesh.totedge(), mesh.totface()), (9, 12, 4));

        let edge = mesh.query_edge_exists(verts[1], verts[4]).expect("edge {1,4}");

        let result = mesh.dissolve_edges(&[edge], &use_verts(std::f32::consts::PI));
        assert_eq!(result, DissolveEdgesResult::Ran);

        assert_eq!(mesh.totvert(), 8, "-1 V (vert 1 collapsed)");
        assert_eq!(mesh.totedge(), 10, "-1 E from merge, -1 E from collapse");
        assert_eq!(mesh.totface(), 3, "-1 F (faces 0+1 merged)");

        let live: std::collections::HashSet<VertexKey> = mesh.vert_keys().into_iter().collect();
        assert!(!live.contains(&verts[1]), "vert 1 (became edge-pair) collapsed");
        assert!(live.contains(&verts[4]), "vert 4 (still 3-valent) survives");
        // The merged-then-collapsed face is the pentagon 0,2,5,4,3.
        assert!(
            mesh.query_edge_exists(verts[0], verts[2]).is_some(),
            "collapse merged {{0,1}} and {{1,2}} into {{0,2}}"
        );
        mesh.validate_topology();
    }

    /// §1 / §5.2 short-circuit (`use_verts = true`, `angle_threshold = 0`):
    /// the post-pass is forced off, so the result matches the
    /// default-parameter outcome on the same input — `-1 E, -1 F, 0 V`, and
    /// vert 1 survives uncollapsed. Direct contrast with the headline test.
    #[test]
    fn dissolve_edges_use_verts_true_angle_threshold_zero_short_circuits() {
        let mut mesh = LoopMesh::new();
        let (verts, _faces) = build_face_grid_2x2(&mut mesh);

        let edge = mesh.query_edge_exists(verts[1], verts[4]).expect("edge {1,4}");

        let result = mesh.dissolve_edges(&[edge], &use_verts(0.0));
        assert_eq!(result, DissolveEdgesResult::Ran);

        assert_eq!(mesh.totvert(), 9, "0 V removed — use_verts forced off");
        assert_eq!(mesh.totedge(), 11, "-1 E (only the merge)");
        assert_eq!(mesh.totface(), 3, "-1 F (faces 0+1 merged)");

        let live: std::collections::HashSet<VertexKey> = mesh.vert_keys().into_iter().collect();
        assert!(live.contains(&verts[1]), "vert 1 survives — no collapse");
        assert_eq!(mesh.vert_edge_keys(verts[1]).len(), 2, "vert 1 is now edge-pair but uncollapsed");
        mesh.validate_topology();
    }

    /// §5.1 first sub-rule — triangle-to-quad anti-degeneration guard.
    /// Dissolving the `{0,2}` diagonal of two triangles with `use_verts`
    /// on: both faces are triangles and each endpoint has exactly one tagged
    /// edge, so the guard fires at both endpoints. The merged quad survives
    /// intact — identical to the `use_verts = false` outcome.
    #[test]
    fn dissolve_edges_use_verts_anti_degeneration_guard_preserves_triangle() {
        // Two triangles sharing diagonal {0,2}: corners 0=(0,0), 1=(1,0),
        // 2=(0,1), 3=(-1,1). Triangle A = (0,1,2); triangle B = (0,2,3).
        let mut mesh = LoopMesh::new();
        let v: Vec<VertexKey> = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (-1.0, 1.0)]
            .into_iter()
            .map(|(x, y)| mesh.create_vertex(Vec3::new(x, y, 0.0)))
            .collect();
        let e_01 = mesh.create_edge([v[0], v[1]], EdgeCreateFlags::NO_DUPLICATE);
        let e_12 = mesh.create_edge([v[1], v[2]], EdgeCreateFlags::NO_DUPLICATE);
        let e_20 = mesh.create_edge([v[2], v[0]], EdgeCreateFlags::NO_DUPLICATE);
        let e_23 = mesh.create_edge([v[2], v[3]], EdgeCreateFlags::NO_DUPLICATE);
        let e_30 = mesh.create_edge([v[3], v[0]], EdgeCreateFlags::NO_DUPLICATE);
        mesh.face_create(&[v[0], v[1], v[2]], &[e_01, e_12, e_20], FaceCreateFlags::NONE);
        mesh.face_create(&[v[0], v[2], v[3]], &[e_20, e_23, e_30], FaceCreateFlags::NONE);

        let diagonal = mesh.query_edge_exists(v[0], v[2]).expect("diagonal {0,2}");

        let result = mesh.dissolve_edges(&[diagonal], &use_verts(std::f32::consts::PI));
        assert_eq!(result, DissolveEdgesResult::Ran);

        // The guard fires at both endpoints: neither collapses. The two tris
        // become one quad; verts unchanged. Same as use_verts = false.
        assert_eq!(mesh.totvert(), 4, "0 V removed — guard preserved both endpoints");
        assert_eq!(mesh.totedge(), 4, "-1 E (the diagonal)");
        assert_eq!(mesh.totface(), 1, "-1 F (two tris → one quad)");
        let faces = mesh.face_keys();
        assert_eq!(mesh.face_len(faces[0]), 4, "the merged face is an intact quad");
        mesh.validate_topology();
    }

    /// §5.1 third sub-rule — T-junction / incidental-touch filter. On the
    /// 2x2 face grid, dissolve edges `{1,4}` and `{4,5}`, both incident at
    /// the 4-valence centre vert 4. Vert 4 touches unselected faces (faces 2
    /// and 3, which contain neither input edge) AND has more than one tagged
    /// edge → the filter fires, so vert 4 is NOT collapsed. The other
    /// endpoints (verts 1 and 5) have only one tagged edge each, so the
    /// carve-out marks them and they collapse.
    #[test]
    fn dissolve_edges_use_verts_t_junction_filter_preserves_corner() {
        let mut mesh = LoopMesh::new();
        let (verts, _faces) = build_face_grid_2x2(&mut mesh);

        let edge_a = mesh.query_edge_exists(verts[1], verts[4]).expect("edge {1,4}");
        let edge_b = mesh.query_edge_exists(verts[4], verts[5]).expect("edge {4,5}");

        let result = mesh.dissolve_edges(&[edge_a, edge_b], &use_verts(std::f32::consts::PI));
        assert_eq!(result, DissolveEdgesResult::Ran);

        let live: std::collections::HashSet<VertexKey> = mesh.vert_keys().into_iter().collect();
        assert!(
            live.contains(&verts[4]),
            "vert 4 touches unselected faces with >1 tagged edge — filter preserves it"
        );
        assert!(!live.contains(&verts[1]), "vert 1 (one tagged edge) collapses");
        assert!(!live.contains(&verts[5]), "vert 5 (one tagged edge) collapses");
        mesh.validate_topology();
    }

    /// Build the §4.4 multi-shared-edge chain fixture: two faces sharing a
    /// chord `1 → 4 → 5 → 3` subdivided at midpoints 4 and 5. Both faces
    /// traverse all three chord segments (one forward, one reversed), so
    /// each segment is shared (radial 2, distinct faces) and the midpoints 4
    /// and 5 are pre-existing edge-pair verts; verts 1 and 3 are the
    /// 3-valent chain termini. Returns `verts[6]`.
    fn build_chain_through_edge_pair_verts(mesh: &mut LoopMesh) -> Vec<VertexKey> {
        // 0,1,2,3 outer corners; 4,5 chord midpoints.
        let v: Vec<VertexKey> = [
            (-2.0, 0.0),
            (0.0, -1.0),
            (2.0, 0.0),
            (0.0, 1.0),
            (0.0, -0.33),
            (0.0, 0.33),
        ]
        .into_iter()
        .map(|(x, y)| mesh.create_vertex(Vec3::new(x, y, 0.0)))
        .collect();
        let e_01 = mesh.create_edge([v[0], v[1]], EdgeCreateFlags::NO_DUPLICATE);
        let e_14 = mesh.create_edge([v[1], v[4]], EdgeCreateFlags::NO_DUPLICATE);
        let e_45 = mesh.create_edge([v[4], v[5]], EdgeCreateFlags::NO_DUPLICATE);
        let e_53 = mesh.create_edge([v[5], v[3]], EdgeCreateFlags::NO_DUPLICATE);
        let e_30 = mesh.create_edge([v[3], v[0]], EdgeCreateFlags::NO_DUPLICATE);
        let e_12 = mesh.create_edge([v[1], v[2]], EdgeCreateFlags::NO_DUPLICATE);
        let e_23 = mesh.create_edge([v[2], v[3]], EdgeCreateFlags::NO_DUPLICATE);
        // Left face: 0 → 1 → 4 → 5 → 3 → 0 (chord walked 1→4→5→3).
        mesh.face_create(
            &[v[0], v[1], v[4], v[5], v[3]],
            &[e_01, e_14, e_45, e_53, e_30],
            FaceCreateFlags::NONE,
        );
        // Right face: 1 → 2 → 3 → 5 → 4 → 1 (chord walked 3→5→4→1).
        mesh.face_create(
            &[v[1], v[2], v[3], v[5], v[4]],
            &[e_12, e_23, e_53, e_45, e_14],
            FaceCreateFlags::NONE,
        );
        v
    }

    /// §5.1 second sub-rule — chain-walk to the chain terminus. Dissolving
    /// the middle chord segment `{4,5}` with `use_verts` on: both endpoints
    /// (verts 4 and 5) are pre-merge edge-pair, so the chain-walk redirects
    /// each mark from the mid-chain vert to the 3-valent terminus (vert 1
    /// from vert 4 via `{1,4}`; vert 3 from vert 5 via `{5,3}`). The
    /// mid-chain verts themselves are never directly marked.
    #[test]
    fn dissolve_edges_use_verts_chain_walk_through_subdivided_chord() {
        let mut mesh = LoopMesh::new();
        let v = build_chain_through_edge_pair_verts(&mut mesh);
        assert_eq!(mesh.vert_edge_keys(v[4]).len(), 2, "vert 4 is pre-merge edge-pair");
        assert_eq!(mesh.vert_edge_keys(v[5]).len(), 2, "vert 5 is pre-merge edge-pair");
        assert_eq!(mesh.vert_edge_keys(v[1]).len(), 3, "vert 1 is a 3-valent terminus");
        assert_eq!(mesh.vert_edge_keys(v[3]).len(), 3, "vert 3 is a 3-valent terminus");

        let edge = mesh.query_edge_exists(v[4], v[5]).expect("middle chord segment {4,5}");

        let result = mesh.dissolve_edges(&[edge], &use_verts(std::f32::consts::PI));
        assert_eq!(result, DissolveEdgesResult::Ran);

        // The chain-walk targets the termini, not the mid-chain verts: verts
        // 1 and 3 are the marked collapse candidates. Verts 4 and 5 are
        // consumed by the merge of {4,5} (their shared segment dies). The
        // termini collapse, removing them from the live set.
        let live: std::collections::HashSet<VertexKey> = mesh.vert_keys().into_iter().collect();
        assert!(!live.contains(&v[1]), "terminus vert 1 collapsed (chain-walk target)");
        assert!(!live.contains(&v[3]), "terminus vert 3 collapsed (chain-walk target)");
        mesh.validate_topology();
    }

    /// §11 — customdata identity on a `use_verts`-collapsed edge. The
    /// surviving edge keeps the kept-side edge's per-element attribute,
    /// inherited through the shared `vert_collapse_edge` primitive. Pinned
    /// with a per-edge attribute on the 2x2 grid: dissolve `{1,4}` with
    /// `use_verts` so vert 1 collapses, merging its two incident edges, and
    /// confirm the surviving edge carries the propagated value.
    #[test]
    fn dissolve_edges_use_verts_collapsed_edge_cd_identity() {
        let mut mesh = LoopMesh::new();
        let (verts, _faces) = build_face_grid_2x2(&mut mesh);

        let tint = mesh
            .register_edge_attribute(ElementAttributeDesc::numeric(
                AttributeKey::custom("tint"),
                Format::new(ComponentType::F32, 3),
            ))
            .unwrap();

        // Vert 1's two perimeter edges that survive the merge and feed the
        // collapse are {0,1} and {1,2}; one of them becomes the survivor
        // edge {0,2}, the other is killed (§11 keeps the kept-side edge's
        // CD). Tag both with the same value so the survivor must carry the
        // propagated (non-default) value regardless of which side is kept.
        let cd_value = [0.25_f32, 0.5, 0.75];
        let e_01 = mesh.query_edge_exists(verts[0], verts[1]).expect("edge {0,1}");
        let e_12 = mesh.query_edge_exists(verts[1], verts[2]).expect("edge {1,2}");
        mesh.edge_attribs_mut().set_vec3(tint, e_01.as_usize(), cd_value);
        mesh.edge_attribs_mut().set_vec3(tint, e_12.as_usize(), cd_value);

        let edge = mesh.query_edge_exists(verts[1], verts[4]).expect("edge {1,4}");
        let result = mesh.dissolve_edges(&[edge], &use_verts(std::f32::consts::PI));
        assert_eq!(result, DissolveEdgesResult::Ran);

        // Vert 1 collapsed; its two edges {0,1} and {1,2} merged into {0,2}.
        let survivor = mesh
            .query_edge_exists(verts[0], verts[2])
            .expect("collapse produced the merged edge {0,2}");
        let got = mesh
            .edge_attribs()
            .get_vec3(tint, survivor.as_usize())
            .expect("survivor edge carries the tint layer");
        for i in 0..3 {
            assert!(
                (got[i] - cd_value[i]).abs() < 1e-6,
                "survivor carries the kept-side edge CD through the collapse: \
                 got {got:?}, expected {cd_value:?}"
            );
        }
        mesh.validate_topology();
    }

    // ---- §5.2 blended-angle gate (intermediate angle_threshold) ----

    /// Build the §5.2.5 worked-example fixture: two unit quads sharing edge
    /// `{1,2}`, the second folded out of the first's plane. Vert numbering
    /// matches the guide: flat quad `[0,1,2,3]` lies in z=0 extending in −x
    /// from the shared edge; the second quad `[1,4,5,2]` extends from the
    /// shared edge along `displacement`. With `displacement = (0,0,1)` this is
    /// the canonical 90° fold (vert 4 = `(1,0,1)`, vert 5 = `(1,1,1)`). The
    /// vertex normals are populated so the §5.2.2 face-pair branch — when it
    /// is reached — has a real `v.no` to project against. Returns `verts[6]`.
    fn build_folded_two_quads(mesh: &mut LoopMesh, displacement: Vec3) -> Vec<VertexKey> {
        let positions = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0) + displacement,
            Vec3::new(1.0, 1.0, 0.0) + displacement,
        ];
        let v: Vec<VertexKey> = positions
            .iter()
            .map(|&p| mesh.create_vertex(p))
            .collect();

        let e_01 = mesh.create_edge([v[0], v[1]], EdgeCreateFlags::NO_DUPLICATE);
        let e_12 = mesh.create_edge([v[1], v[2]], EdgeCreateFlags::NO_DUPLICATE);
        let e_23 = mesh.create_edge([v[2], v[3]], EdgeCreateFlags::NO_DUPLICATE);
        let e_30 = mesh.create_edge([v[3], v[0]], EdgeCreateFlags::NO_DUPLICATE);
        let e_14 = mesh.create_edge([v[1], v[4]], EdgeCreateFlags::NO_DUPLICATE);
        let e_45 = mesh.create_edge([v[4], v[5]], EdgeCreateFlags::NO_DUPLICATE);
        let e_52 = mesh.create_edge([v[5], v[2]], EdgeCreateFlags::NO_DUPLICATE);

        // Flat quad [0,1,2,3] and folded quad [1,4,5,2], both CCW so the
        // shared edge {1,2} flanks two distinct faces.
        mesh.face_create(
            &[v[0], v[1], v[2], v[3]],
            &[e_01, e_12, e_23, e_30],
            FaceCreateFlags::NONE,
        );
        mesh.face_create(
            &[v[1], v[4], v[5], v[2]],
            &[e_14, e_45, e_52, e_12],
            FaceCreateFlags::NONE,
        );

        mesh.mesh_normals_update();
        v
    }

    /// §5.2.5 worked example / the targeted parity row. Two unit quads folded
    /// at 90° sharing edge `{1,2}`; dissolving it with `use_verts = true` and
    /// an intermediate `angle_threshold = 0.5`. The merge produces one hexagon;
    /// both endpoints (verts 1 and 2) land on the single merged-face perimeter,
    /// so each candidate's primary edge has only one flanking face — the §5.2.2
    /// face-pair branch is *not* taken and `α = α_raw = π/2 ≈ 1.5708`. Since
    /// `1.5708 ≥ 0.5 − ε`, both collapses are rejected. Net `−1 E, −1 F`,
    /// 0 V removed: 6 V, 6 E, 1 F.
    #[test]
    fn dissolve_edges_blended_angle_rejects_ninety_degree_fold() {
        let mut mesh = LoopMesh::new();
        let v = build_folded_two_quads(&mut mesh, Vec3::new(0.0, 0.0, 1.0));
        assert_eq!((mesh.totvert(), mesh.totedge(), mesh.totface()), (6, 7, 2));

        let shared = mesh.query_edge_exists(v[1], v[2]).expect("shared edge {1,2}");

        let result = mesh.dissolve_edges(&[shared], &use_verts(0.5));
        assert_eq!(result, DissolveEdgesResult::Ran);

        assert_eq!(mesh.totvert(), 6, "0 V removed — both candidates rejected by the gate");
        assert_eq!(mesh.totedge(), 6, "-1 E (only the merge)");
        assert_eq!(mesh.totface(), 1, "-1 F (the two quads merge into one hexagon)");

        let live: std::collections::HashSet<VertexKey> = mesh.vert_keys().into_iter().collect();
        assert!(live.contains(&v[1]), "vert 1 survives — α_raw = π/2 ≥ 0.5");
        assert!(live.contains(&v[2]), "vert 2 survives — α_raw = π/2 ≥ 0.5");
        let face = mesh.face_keys();
        assert_eq!(face.len(), 1);
        assert_eq!(mesh.face_len(face[0]), 6, "merged hexagon perimeter 0-1-4-5-2-3");
        mesh.validate_topology();
    }

    /// Same 90° fold, but a wide-enough intermediate threshold (`2.0 > π/2`):
    /// `α_raw = π/2 ≈ 1.5708 < 2.0 − ε`, so the gate now *accepts* both
    /// collapses. Direct contrast with the reject case above — only the
    /// threshold differs. Both endpoints collapse: net `−1 V` each (`−2 V`
    /// total), the merged hexagon shrinking to a quad: 4 V, 4 E, 1 F.
    #[test]
    fn dissolve_edges_blended_angle_accepts_ninety_degree_fold_with_wide_threshold() {
        let mut mesh = LoopMesh::new();
        let v = build_folded_two_quads(&mut mesh, Vec3::new(0.0, 0.0, 1.0));

        let shared = mesh.query_edge_exists(v[1], v[2]).expect("shared edge {1,2}");

        let result = mesh.dissolve_edges(&[shared], &use_verts(2.0));
        assert_eq!(result, DissolveEdgesResult::Ran);

        let live: std::collections::HashSet<VertexKey> = mesh.vert_keys().into_iter().collect();
        assert!(!live.contains(&v[1]), "vert 1 collapses — α_raw = π/2 < 2.0");
        assert!(!live.contains(&v[2]), "vert 2 collapses — α_raw = π/2 < 2.0");
        assert_eq!(mesh.totvert(), 4, "-2 V (both endpoints collapsed)");
        assert_eq!(mesh.totface(), 1, "-1 F (the merge)");
        mesh.validate_topology();
    }

    /// Intermediate-threshold gate on a *straight* (near-collinear) chain: the
    /// 2x2 face grid, dissolving `{1,4}`. Vert 1 becomes an edge-pair whose two
    /// surviving edges `{0,1}` and `{1,2}` are collinear along x, so
    /// `α_raw = π − π = 0`. Its primary edge is a single-face grid-boundary
    /// edge post-merge, so the face-pair branch is skipped and `α = α_raw ≈ 0`.
    /// At `angle_threshold = 0.5` the gate accepts (`0 < 0.5 − ε`), so vert 1
    /// collapses — matching the wide-threshold headline outcome exactly while
    /// going through the gated code path. Vert 4 stays 3-valent and survives.
    #[test]
    fn dissolve_edges_blended_angle_accepts_straight_chain() {
        let mut mesh = LoopMesh::new();
        let (verts, _faces) = build_face_grid_2x2(&mut mesh);

        let edge = mesh.query_edge_exists(verts[1], verts[4]).expect("edge {1,4}");

        let result = mesh.dissolve_edges(&[edge], &use_verts(0.5));
        assert_eq!(result, DissolveEdgesResult::Ran);

        assert_eq!(mesh.totvert(), 8, "-1 V (vert 1, α_raw≈0, collapses)");
        assert_eq!(mesh.totedge(), 10, "-1 E merge, -1 E collapse");
        assert_eq!(mesh.totface(), 3, "-1 F (faces 0+1 merged)");

        let live: std::collections::HashSet<VertexKey> = mesh.vert_keys().into_iter().collect();
        assert!(!live.contains(&verts[1]), "vert 1 (straight chain) collapsed by the gate");
        assert!(live.contains(&verts[4]), "vert 4 (still 3-valent) survives");
        mesh.validate_topology();
    }

    // ---- use_face_split pre-pass (§4 / §6.3) ----

    fn use_face_split() -> DissolveEdgesParams {
        DissolveEdgesParams {
            use_verts: false,
            use_face_split: true,
            angle_threshold: std::f32::consts::PI,
        }
    }

    /// Count faces with `face_len == 3` in the mesh.
    fn count_triangles(mesh: &LoopMesh) -> usize {
        let mut tris = 0usize;
        let mut face_iter = MeshFaces::default();
        while let Some(f) = face_iter.next(mesh) {
            if mesh.face_len(f) == 3 {
                tris += 1;
            }
        }
        tris
    }

    /// §4 / §6.3 headline + §4.2 structural contrast — the targeted parity row.
    /// Two input edges `{1,4}` and `{3,4}` converge on the 4-valence centre
    /// vert 4 of the 2x2 face grid. Vert 4's untagged-edge count is 2 (4 edges,
    /// 2 tagged), so it is marked and — being a non-edge-pair n-gon corner — its
    /// surrounding quads are diagonal-sliced *before* the merges.
    ///
    /// Of vert 4's four quads, three have a marked vert on *both* corners of the
    /// candidate diagonal (verts 1 and 3 are also marked) and are skipped per
    /// §4.1; only `face3 = (4,5,8,7)`, whose neighbours 5 and 7 are unmarked,
    /// is sliced — adding the diagonal `{5,7}` (+1 E, +1 F). The two merges of
    /// `{1,4}` and `{3,4}` do not touch face3, so the `{5,7}` diagonal and the
    /// triangle it produces survive into the final mesh. The contrast against
    /// `use_face_split = false` (no diagonal, no triangle) is the documented
    /// "with vs without" pre-pass difference (§4.2 / §7).
    #[test]
    fn dissolve_edges_use_face_split_slices_corner_around_shared_vert() {
        let mut mesh = LoopMesh::new();
        let (verts, _faces) = build_face_grid_2x2(&mut mesh);
        assert_eq!((mesh.totvert(), mesh.totedge(), mesh.totface()), (9, 12, 4));
        assert_eq!(mesh.vert_edge_keys(verts[4]).len(), 4, "vert 4 is the 4-valence centre");

        let edge_a = mesh.query_edge_exists(verts[1], verts[4]).expect("edge {1,4}");
        let edge_b = mesh.query_edge_exists(verts[3], verts[4]).expect("edge {3,4}");

        let result = mesh.dissolve_edges(&[edge_a, edge_b], &use_face_split());
        assert_eq!(result, DissolveEdgesResult::Ran);

        // Pre-pass: +1 E (diag {5,7}) +1 F. Merge {1,4}: -1 E, -1 F. Merge
        // {3,4}: -1 E, -1 F. Net from 9/12/4: 0 V, -1 E, -1 F.
        assert_eq!(mesh.totvert(), 9, "0 V (pre-pass adds no verts)");
        assert_eq!(mesh.totedge(), 11, "+1 diagonal, -2 merges");
        assert_eq!(mesh.totface(), 3, "+1 slice, -2 merges");

        assert!(
            mesh.query_edge_exists(verts[5], verts[7]).is_some(),
            "the pre-pass sliced face3 along the diagonal {{5,7}}"
        );
        // The single slice cuts quad face3 into two triangles; the merges
        // (which act on the other three quads around vert 4) leave them intact.
        assert_eq!(
            count_triangles(&mesh),
            2,
            "two triangles from splitting the one untouched quad along {{5,7}}"
        );
        mesh.validate_topology();
    }

    /// §4.2 structural contrast: the *same* input with `use_face_split = false`.
    /// No pre-pass runs, so no `{5,7}` diagonal is drawn and no triangle is
    /// produced; the two merges consume one more face than the sliced run. This
    /// is the direct "without" half of the headline contrast.
    #[test]
    fn dissolve_edges_no_face_split_leaves_corner_intact() {
        let mut mesh = LoopMesh::new();
        let (verts, _faces) = build_face_grid_2x2(&mut mesh);

        let edge_a = mesh.query_edge_exists(verts[1], verts[4]).expect("edge {1,4}");
        let edge_b = mesh.query_edge_exists(verts[3], verts[4]).expect("edge {3,4}");

        let result = mesh.dissolve_edges(&[edge_a, edge_b], &Default::default());
        assert_eq!(result, DissolveEdgesResult::Ran);

        // No pre-pass: 9/12/4 → merge {1,4} → 9/11/3 → merge {3,4} → 9/10/2.
        assert_eq!(mesh.totvert(), 9, "0 V");
        assert_eq!(mesh.totedge(), 10, "-2 merges, no diagonal added");
        assert_eq!(mesh.totface(), 2, "-2 merges, no slice");

        assert!(
            mesh.query_edge_exists(verts[5], verts[7]).is_none(),
            "without the pre-pass no {{5,7}} diagonal exists"
        );
        assert_eq!(count_triangles(&mesh), 0, "no triangles without the pre-pass");
        mesh.validate_topology();
    }

    /// §4.4 no-op — already-triangulated mesh. `use_face_split = true` on two
    /// triangles sharing a diagonal finds no n-gon corner to slice (every face
    /// is already a triangle, the §4.4 "corner count > 3 fails" no-op), so the
    /// result is identical to the default-parameter run: the two triangles
    /// merge into a quad with no extra geometry.
    #[test]
    fn dissolve_edges_use_face_split_noop_on_triangulated_mesh() {
        let mut mesh = LoopMesh::new();
        let (_verts, diagonal, _faces) = build_two_tris_shared_diagonal(&mut mesh);
        assert_eq!((mesh.totvert(), mesh.totedge(), mesh.totface()), (4, 5, 2));

        let result = mesh.dissolve_edges(&[diagonal], &use_face_split());
        assert_eq!(result, DissolveEdgesResult::Ran);

        // Same outcome as `use_face_split = false`: -1 E (diagonal), -1 F.
        assert_eq!(mesh.totvert(), 4, "0 V");
        assert_eq!(mesh.totedge(), 4, "-1 E (the diagonal), no slice added");
        assert_eq!(mesh.totface(), 1, "-1 F (two tris → one quad)");

        let faces = mesh.face_keys();
        assert_eq!(faces.len(), 1);
        assert_eq!(mesh.face_len(faces[0]), 4, "merged face is an intact quad");
        mesh.validate_topology();
    }

    /// §4.4 no-op — every candidate vert is already an edge-pair. On the 3x3
    /// face grid, dissolving the deep-interior edge `{5,6}` leaves both
    /// endpoints with 3 untagged edges (4 incident − 1 tagged), so neither is
    /// marked. The only verts with ≤ 2 untagged edges are the grid corners,
    /// which are pre-existing edge-pairs (2 incident edges) — the §4.1
    /// sub-pass-2 "not already an edge-pair" guard skips them, so no diagonal
    /// is drawn. The result matches the `use_face_split = false` merge exactly.
    #[test]
    fn dissolve_edges_use_face_split_noop_when_candidates_are_edge_pairs() {
        let mut mesh = LoopMesh::new();
        let (verts, _faces) = build_grid_3x3(&mut mesh);
        assert_eq!((mesh.totvert(), mesh.totedge(), mesh.totface()), (16, 24, 9));

        // verts[5] and verts[6] are deep-interior 4-valence verts sharing an
        // interior edge between faces[1] and faces[4].
        let edge = mesh
            .query_edge_exists(verts[5], verts[6])
            .expect("interior edge {5,6}");
        assert_eq!(mesh.vert_edge_keys(verts[5]).len(), 4);
        assert_eq!(mesh.vert_edge_keys(verts[6]).len(), 4);

        let result = mesh.dissolve_edges(&[edge], &use_face_split());
        assert_eq!(result, DissolveEdgesResult::Ran);

        // Pure merge, no slice: -1 E, -1 F, 0 V.
        assert_eq!(mesh.totvert(), 16, "0 V");
        assert_eq!(mesh.totedge(), 23, "-1 E (the merge only)");
        assert_eq!(mesh.totface(), 8, "-1 F (the merge only)");
        assert_eq!(
            count_triangles(&mesh),
            0,
            "no triangles: every candidate was an edge-pair, nothing sliced"
        );
        // The merged face is the hexagon spanning the two source quads.
        let merged = mesh
            .face_keys()
            .into_iter()
            .find(|&f| mesh.face_len(f) == 6)
            .expect("a hexagonal merged face exists");
        assert_eq!(mesh.face_len(merged), 6);
        mesh.validate_topology();
    }

    /// §4.4 no-op — no candidate verts at all. A single interior edge `{5,6}`
    /// in a mesh with no low-valence verts would be the previous test; here we
    /// confirm the empty-input degenerate: `use_face_split = true` with an empty
    /// edge set tags nothing, so every vert has its full (untagged) disk. On the
    /// 3x3 face grid the corner verts are still ≤ 2 untagged but edge-pairs, the
    /// boundary-midpoints are 3 untagged, the interior verts 4 untagged — so no
    /// diagonal is ever drawn and the mesh is untouched.
    #[test]
    fn dissolve_edges_use_face_split_empty_input_is_noop() {
        let mut mesh = LoopMesh::new();
        let (_verts, _faces) = build_grid_3x3(&mut mesh);
        let v0 = mesh.totvert();
        let e0 = mesh.totedge();
        let f0 = mesh.totface();

        let result = mesh.dissolve_edges(&[], &use_face_split());
        assert_eq!(result, DissolveEdgesResult::Ran);

        assert_eq!((mesh.totvert(), mesh.totedge(), mesh.totface()), (v0, e0, f0));
        assert_eq!(count_triangles(&mesh), 0, "no slices on empty input");
        mesh.validate_topology();
    }
}
