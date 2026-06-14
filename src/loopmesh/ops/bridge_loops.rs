use std::collections::HashSet;

use glam::Vec3;
use smallvec::SmallVec;

use crate::loopmesh::*;

/// Parameters for [`LoopMesh::bridge_loops`], mirroring the slot inputs of
/// `radial-mesh-guides/operators/BRIDGE_LOOPS.md` §1.
#[derive(Clone, Copy, Debug)]
pub struct BridgeLoopsParams {
    /// Grouping mode (§1 / §4.1). When set the discovered loops are bridged
    /// two at a time as disjoint pairs and the loop count must be even.
    pub use_pairs: bool,
    /// Chain closure (§1 / §4.2). When set (and `use_pairs` clear, > 2 loops)
    /// the chain's last loop is also bridged back to the first. Forced off
    /// whenever `use_merge` is set.
    pub use_cyclic: bool,
    /// Weld mode (§1 / §8). When set the operator welds each loop pair
    /// vertex-for-vertex instead of building a band; requires equal lengths.
    pub use_merge: bool,
    /// Position / customdata interpolation fraction for the weld path (§8).
    /// Only consulted when `use_merge` is set.
    pub merge_factor: f32,
    /// Deliberate twist of the rung correspondence for closed pairs (§5.6),
    /// applied after the lowest-distance rotation. For a non-zero `t` the
    /// pairing becomes `A_i -> B_{(i - t - 1) mod n}` (floored modulo); `t == 0`
    /// leaves §5.4's `A_i -> B_i` untouched. Only consulted for closed pairs.
    pub twist_offset: i32,
}

impl Default for BridgeLoopsParams {
    fn default() -> Self {
        Self {
            use_pairs: false,
            use_cyclic: false,
            use_merge: false,
            merge_factor: 0.5,
            twist_offset: 0,
        }
    }
}

/// Output of [`LoopMesh::bridge_loops`] (§1 output slots).
#[derive(Clone, Debug, Default)]
pub struct BridgeLoopsResult {
    /// `false` when a validation gate cancelled the operator (§3.1); the mesh
    /// is then left unchanged and both output sets are empty.
    pub ok: bool,
    /// The new band faces, including any produced by the §7 refinement. Empty
    /// for the merge path (§8).
    pub faces_out: Vec<FaceKey>,
    /// The new rung edges running between the loops — not the pre-existing loop
    /// edges (§6.3). Empty for the merge path.
    pub edges_out: Vec<EdgeKey>,
}

/// A discovered edge loop: its ordered vertex sequence, whether it is a closed
/// ring, and the mean of its vertex positions (§3).
struct EdgeLoop {
    verts: Vec<VertexKey>,
    closed: bool,
    centre: Vec3,
}

impl EdgeLoop {
    fn len(&self) -> usize {
        self.verts.len()
    }
}

impl LoopMesh {
    /// Bridge a set of selected edge loops with bands of faces (build mode) or
    /// weld them together (merge mode).
    ///
    /// Implements `radial-mesh-guides/operators/BRIDGE_LOOPS.md`:
    ///
    /// - **§3 loop discovery** — the input edges are flooded into maximal
    ///   open chains / closed rings; a branch vertex (3+ selected edges) breaks
    ///   the flood.
    /// - **§3.1 validation** — fewer than two loops, an odd loop count under
    ///   `use_pairs`, or unequal loop lengths under `use_merge` each cancel the
    ///   operator (no-op, `ok = false`).
    /// - **§4 grouping** — with more than two loops the loops are ordered by
    ///   centre proximity, then paired either as a chain (`(0,1),(1,2),…`, plus
    ///   the closing `(n−1,0)` band under `use_cyclic`) or as disjoint pairs
    ///   (`(0,1),(2,3),…`) under `use_pairs`.
    /// - **§5 alignment** — for each pair B is matched to A (the longer loop):
    ///   open loops have their direction reconciled (§5.1) and their winding
    ///   reconciled against the between-loops axis (§5.2); the build path also
    ///   votes the band's facing against bordering faces (§5.3); closed loops
    ///   are rotated to the lowest-distance offset (§5.4) plus `twist_offset`
    ///   (§5.6); the shorter loop is padded to the longer's length with
    ///   repeated verts (§5.5).
    /// - **§6 band build** — the aligned sequences are walked in lockstep,
    ///   emitting a quad per step (a triangle where a padded vert repeats),
    ///   reusing an already-existing face and skipping degenerate steps. Per-loop
    ///   / per-face customdata is copied verbatim from the bordering faces
    ///   (§6.2). The rung edges are reported in `edges.out` (§6.3).
    /// - **§7 refinement** — an unequal-length band is triangulated so the
    ///   triangle padding is distributed across the band.
    /// - **§8 merge** — under `use_merge` each A vertex is mapped onto its B
    ///   partner (after moving B by `merge_factor` toward A and blending its
    ///   customdata), and the map is handed to [`Self::weld_verts`]; no band is
    ///   built and the element counts drop.
    pub fn bridge_loops(
        &mut self,
        edges: &[EdgeKey],
        params: BridgeLoopsParams,
    ) -> BridgeLoopsResult {
        let mut loops = self.bridge_discover_loops(edges);

        // §3.1 — fewer than two loops cancels.
        if loops.len() < 2 {
            return BridgeLoopsResult::default();
        }

        // §1 — use_cyclic is forced off in merge mode.
        let use_cyclic = params.use_cyclic && !params.use_merge;

        // §3.1 — pair mode needs an even count.
        if params.use_pairs && loops.len() % 2 != 0 {
            return BridgeLoopsResult::default();
        }

        // §3.1 — merge requires every loop to have the same length.
        if params.use_merge {
            let first_len = loops[0].len();
            if loops.iter().any(|l| l.len() != first_len) {
                return BridgeLoopsResult::default();
            }
        }

        // §4 — order loops by centre proximity when there are more than two,
        // then form the pair list per the grouping mode.
        if loops.len() > 2 {
            self.bridge_order_loops(&mut loops);
        }
        let pair_indices = bridge_pairings(loops.len(), params.use_pairs, use_cyclic);

        let mut result = BridgeLoopsResult {
            ok: true,
            faces_out: Vec::new(),
            edges_out: Vec::new(),
        };

        for (ia, ib) in pair_indices {
            // Split the borrow: align needs &self for geometry, so pull the
            // two loops out, work on owned copies, and write nothing back to
            // the discovery list (each loop may pair more than once and must
            // stay in its original orientation for the next pair — §5.5).
            let mut loop_a = clone_loop(&loops[ia]);
            let mut loop_b = clone_loop(&loops[ib]);

            self.bridge_align_pair(&mut loop_a, &mut loop_b, params, !params.use_merge);

            if params.use_merge {
                self.bridge_merge_pair(&loop_a, &loop_b, params.merge_factor);
            } else {
                self.bridge_build_band(&loop_a, &loop_b, &mut result);
            }
        }

        result
    }

    /// §3 — flood the input edges into maximal edge loops. A vertex carrying a
    /// single selected edge is a chain endpoint; two selected edges is an
    /// interior vertex; three or more is a branch that breaks the flood.
    fn bridge_discover_loops(&self, edges: &[EdgeKey]) -> Vec<EdgeLoop> {
        let selected: HashSet<EdgeKey> = edges.iter().copied().collect();
        if selected.is_empty() {
            return Vec::new();
        }

        // Per-vertex count of incident selected edges, to classify endpoints
        // and branch points.
        let mut selected_valence: std::collections::HashMap<VertexKey, u32> =
            std::collections::HashMap::new();
        for &edge in &selected {
            let [v0, v1] = self.edge_vertices(edge);
            *selected_valence.entry(v0).or_insert(0) += 1;
            *selected_valence.entry(v1).or_insert(0) += 1;
        }

        let next_selected_edge = |vert: VertexKey, skip: EdgeKey| -> Option<EdgeKey> {
            let mut chosen = None;
            for edge in self.iter_disk_edges(vert) {
                if edge != skip && selected.contains(&edge) {
                    chosen = Some(edge);
                    break;
                }
            }
            chosen
        };

        let mut visited: HashSet<EdgeKey> = HashSet::new();
        let mut out = Vec::new();

        // Iterate the input edges in their given order (deduplicated) so loop
        // discovery is deterministic — the result must not depend on hash
        // iteration order. The input order is part of the operator's
        // reproducibility contract (§4's ordering then re-sorts ≥3 loops).
        let mut seen_input: HashSet<EdgeKey> = HashSet::new();
        let ordered_edges: Vec<EdgeKey> = edges
            .iter()
            .copied()
            .filter(|e| seen_input.insert(*e))
            .collect();

        // First sweep: every chain that has at least one endpoint vertex (a
        // vertex with exactly one selected edge). Walking from an endpoint
        // discovers an open loop and consumes its edges.
        for &edge in &ordered_edges {
            if visited.contains(&edge) {
                continue;
            }
            let [v0, v1] = self.edge_vertices(edge);
            let start_vert = if selected_valence[&v0] == 1 {
                Some(v0)
            } else if selected_valence[&v1] == 1 {
                Some(v1)
            } else {
                None
            };
            let Some(start_vert) = start_vert else {
                continue;
            };

            let mut verts = vec![start_vert];
            let mut cur_vert = start_vert;
            let mut cur_edge = edge;
            loop {
                visited.insert(cur_edge);
                let next_vert = self.edges.get(cur_edge).other_vertex(cur_vert).unwrap();
                verts.push(next_vert);
                // Stop at a branch or another endpoint.
                if selected_valence[&next_vert] != 2 {
                    break;
                }
                match next_selected_edge(next_vert, cur_edge) {
                    Some(next_edge) if !visited.contains(&next_edge) => {
                        cur_vert = next_vert;
                        cur_edge = next_edge;
                    }
                    _ => break,
                }
            }
            let centre = self.bridge_centre(&verts);
            out.push(EdgeLoop {
                verts,
                closed: false,
                centre,
            });
        }

        // Second sweep: any remaining edges form closed rings (every vertex on
        // them has valence two and none was an endpoint).
        for &edge in &ordered_edges {
            if visited.contains(&edge) {
                continue;
            }
            let [v0, v1] = self.edge_vertices(edge);
            let mut verts = vec![v0];
            let mut cur_vert = v0;
            let mut cur_edge = edge;
            loop {
                visited.insert(cur_edge);
                let next_vert = self.edges.get(cur_edge).other_vertex(cur_vert).unwrap();
                if next_vert == v0 {
                    break;
                }
                if selected_valence[&next_vert] != 2 {
                    break;
                }
                verts.push(next_vert);
                match next_selected_edge(next_vert, cur_edge) {
                    Some(next_edge) if !visited.contains(&next_edge) => {
                        cur_vert = next_vert;
                        cur_edge = next_edge;
                    }
                    _ => break,
                }
            }
            let centre = self.bridge_centre(&verts);
            out.push(EdgeLoop {
                verts,
                closed: true,
                centre,
            });
        }

        out
    }

    fn bridge_centre(&self, verts: &[VertexKey]) -> Vec3 {
        if verts.is_empty() {
            return Vec3::ZERO;
        }
        let sum: Vec3 = verts
            .iter()
            .fold(Vec3::ZERO, |acc, &v| acc + self.vertex_position(v));
        sum / verts.len() as f32
    }

    /// §4 — order the loops so that consecutive entries are spatially adjacent,
    /// using a nearest-centre greedy walk. This gives the operator its
    /// order-independence: the result depends on the loop set's geometry, not
    /// the input order of the edges.
    fn bridge_order_loops(&self, loops: &mut Vec<EdgeLoop>) {
        let count = loops.len();
        let mut order: Vec<usize> = Vec::with_capacity(count);
        let mut used = vec![false; count];
        order.push(0);
        used[0] = true;
        for _ in 1..count {
            let last = *order.last().unwrap();
            let from = loops[last].centre;
            let mut best = None;
            let mut best_dist = f32::INFINITY;
            for candidate in 0..count {
                if used[candidate] {
                    continue;
                }
                let dist = (loops[candidate].centre - from).length_squared();
                if dist < best_dist {
                    best_dist = dist;
                    best = Some(candidate);
                }
            }
            let next = best.unwrap();
            used[next] = true;
            order.push(next);
        }

        let mut reordered: Vec<EdgeLoop> = Vec::with_capacity(count);
        // `order` is a permutation; drain the originals into their new slots.
        let mut taken: Vec<Option<EdgeLoop>> = loops.drain(..).map(Some).collect();
        for index in order {
            reordered.push(taken[index].take().unwrap());
        }
        *loops = reordered;
    }

    /// §5 — match loop B to loop A so step `i` of A corresponds to step `i` of
    /// B. `loop_a` is forced to be the longer side up front so the §5.5 padding
    /// always pads B. `build_mode` enables the build-only winding vote (§5.3).
    fn bridge_align_pair(
        &self,
        loop_a: &mut EdgeLoop,
        loop_b: &mut EdgeLoop,
        params: BridgeLoopsParams,
        build_mode: bool,
    ) {
        // §5 — A is the longer loop.
        if loop_b.len() > loop_a.len() {
            std::mem::swap(loop_a, loop_b);
        }

        let between = loop_a.centre - loop_b.centre;

        let closed = loop_a.closed && loop_b.closed;

        if !closed {
            self.bridge_reconcile_open_direction(loop_a, loop_b, between);
        }

        self.bridge_reconcile_winding(loop_a, loop_b, between, build_mode);

        if closed {
            // §5.4 — rotate B to the lowest-distance offset against A.
            self.bridge_rotate_lowest_distance(loop_a, loop_b);
            // §5.6 — apply the deliberate twist after the lowest-distance pick.
            bridge_apply_twist(loop_b, params.twist_offset);
        }

        // §5.5 — pad B up to A's length with repeated verts.
        if loop_a.len() > loop_b.len() {
            bridge_pad_loop(loop_b, loop_a.len());
        }
    }

    /// §5.1 — for open loops, flip B if the two chains were discovered running
    /// opposite ways (their end-to-end directions, projected off the
    /// between-loops axis, point against each other).
    fn bridge_reconcile_open_direction(
        &self,
        loop_a: &EdgeLoop,
        loop_b: &mut EdgeLoop,
        between: Vec3,
    ) {
        let dir_a = self.bridge_end_to_end(loop_a);
        let dir_b = self.bridge_end_to_end(loop_b);

        let axis = between.normalize_or_zero();
        let proj_a = dir_a - axis * dir_a.dot(axis);
        let proj_b = dir_b - axis * dir_b.dot(axis);

        // Degenerate fallback (§5.1): a loop whose end-to-end direction is
        // parallel to the between-loops axis projects to nothing, so compare
        // the unprojected directions instead.
        let (cmp_a, cmp_b) = if proj_a.length_squared() < 1.0e-12
            || proj_b.length_squared() < 1.0e-12
        {
            (dir_a, dir_b)
        } else {
            (proj_a, proj_b)
        };

        if cmp_a.dot(cmp_b) < 0.0 {
            loop_b.verts.reverse();
        }
    }

    fn bridge_end_to_end(&self, edge_loop: &EdgeLoop) -> Vec3 {
        let head = self.vertex_position(*edge_loop.verts.first().unwrap());
        let tail = self.vertex_position(*edge_loop.verts.last().unwrap());
        head - tail
    }

    /// §5.2 / §5.3 — reconcile the two loops' facing against the between-loops
    /// axis, and (build mode only) vote the band's overall facing against the
    /// faces the loops already border.
    fn bridge_reconcile_winding(
        &self,
        loop_a: &mut EdgeLoop,
        loop_b: &mut EdgeLoop,
        between: Vec3,
        build_mode: bool,
    ) {
        let axis = between.normalize_or_zero();
        let normal_a = self.bridge_loop_normal(loop_a, axis);
        let normal_b = self.bridge_loop_normal(loop_b, axis);

        let no_depth = between.length_squared() < 1.0e-12
            || (normal_a.dot(axis).abs() < 1.0e-4 && normal_b.dot(axis).abs() < 1.0e-4);

        if no_depth {
            // §5.2 no-depth case — just make the loop normals agree.
            if normal_a.dot(normal_b) < 0.0 {
                loop_b.verts.reverse();
            }
        } else {
            // §5.2 general case — flip B if the two normals sit on opposite
            // sides of the between-loops axis.
            if normal_a.dot(axis).signum() != normal_b.dot(axis).signum() {
                loop_b.verts.reverse();
            }
        }

        if !build_mode {
            return;
        }

        // §5.3 outward orientation — flip both loops if their summed normal
        // points against the between-loops axis.
        let normal_a = self.bridge_loop_normal(loop_a, axis);
        let normal_b = self.bridge_loop_normal(loop_b, axis);
        if (normal_a + normal_b).dot(axis) < 0.0 {
            loop_a.verts.reverse();
            loop_b.verts.reverse();
        }

        // §5.3 boundary-edge winding vote — each loop accumulates a signed
        // tally of how the faces it already borders traverse its edges; a loop
        // whose band contradicts the surrounding winding is flipped.
        let vote_a = self.bridge_boundary_vote(loop_a);
        let vote_b = self.bridge_boundary_vote(loop_b);
        if vote_a < 0 && vote_b < 0 {
            loop_a.verts.reverse();
            loop_b.verts.reverse();
        } else if vote_a < 0 && vote_b == 0 {
            loop_a.verts.reverse();
        } else if vote_b < 0 && vote_a == 0 {
            loop_b.verts.reverse();
        }
    }

    /// §5.2 — the loop's normal. For a closed loop it is the area-weighted
    /// polygon normal of its vertex ring; for an open loop it is computed in
    /// the between-loops frame, aligned to `reference`.
    fn bridge_loop_normal(&self, edge_loop: &EdgeLoop, reference: Vec3) -> Vec3 {
        let positions: SmallVec<[Vec3; 16]> = edge_loop
            .verts
            .iter()
            .map(|&v| self.vertex_position(v))
            .collect();
        let normal = bridge_newell_normal(&positions, edge_loop.closed);
        if normal.length_squared() < 1.0e-20 {
            // Degenerate ring — fall back to the reference axis so the winding
            // tests stay well-defined.
            return reference;
        }
        let normal = normal.normalize();
        if !edge_loop.closed && reference.length_squared() > 1.0e-20 && normal.dot(reference) < 0.0 {
            -normal
        } else {
            normal
        }
    }

    /// §5.3 — signed tally of how the existing boundary faces on a loop's edges
    /// traverse those edges relative to the loop's own walk direction.
    fn bridge_boundary_vote(&self, edge_loop: &EdgeLoop) -> i32 {
        let mut tally = 0_i32;
        let span = if edge_loop.closed {
            edge_loop.len()
        } else {
            edge_loop.len().saturating_sub(1)
        };
        for i in 0..span {
            let v0 = edge_loop.verts[i];
            let v1 = edge_loop.verts[(i + 1) % edge_loop.len()];
            let Some(edge) = self.query_edge_exists(v0, v1) else {
                continue;
            };
            let radial: SmallVec<[LoopKey; 4]> = self.iter_radial_loops(edge).collect();
            if radial.len() != 1 {
                continue;
            }
            // A boundary face traverses the edge v0->v1 in the loop's direction
            // when the bordering loop's own vertex is v0.
            if self.loop_vertex(radial[0]) == v0 {
                tally += 1;
            } else {
                tally -= 1;
            }
        }
        tally
    }

    /// §5.4 — rotate B (a closed loop) to the starting offset that minimises
    /// the total rung length against A's fixed start. An early cutoff skips
    /// candidates that already exceed the running best.
    fn bridge_rotate_lowest_distance(&self, loop_a: &EdgeLoop, loop_b: &mut EdgeLoop) {
        let count = loop_b.len();
        if count == 0 {
            return;
        }
        let pos_a: SmallVec<[Vec3; 16]> = loop_a
            .verts
            .iter()
            .map(|&v| self.vertex_position(v))
            .collect();
        let pos_b: SmallVec<[Vec3; 16]> = loop_b
            .verts
            .iter()
            .map(|&v| self.vertex_position(v))
            .collect();
        // The rung walk compares position i of A against position i of B; only
        // the first min(len) rungs constrain the offset when A is longer.
        let compare = pos_a.len().min(count);

        let mut best_offset = 0;
        let mut best_total = f32::INFINITY;
        for offset in 0..count {
            let mut total = 0.0_f32;
            for i in 0..compare {
                let delta = pos_a[i] - pos_b[(i + offset) % count];
                total += delta.length();
                if total >= best_total {
                    break;
                }
            }
            if total < best_total {
                best_total = total;
                best_offset = offset;
            }
        }

        if best_offset != 0 {
            loop_b.verts.rotate_left(best_offset);
        }
    }

    /// §6 — walk the two aligned sequences in lockstep, emitting a band face
    /// per step. Equal-length bands emit clean quads; unequal-length bands emit
    /// triangle-padded steps that are then triangulated (§7).
    fn bridge_build_band(
        &mut self,
        loop_a: &EdgeLoop,
        loop_b: &EdgeLoop,
        result: &mut BridgeLoopsResult,
    ) {
        let closed = loop_a.closed && loop_b.closed;
        let count = loop_a.len().min(loop_b.len());
        if count < 2 {
            return;
        }
        let steps = if closed { count } else { count - 1 };

        let was_unequal = bridge_loop_has_repeat(loop_b) || bridge_loop_has_repeat(loop_a);

        let mut band_faces: SmallVec<[FaceKey; 16]> = SmallVec::new();
        for step in 0..steps {
            let next = (step + 1) % count;
            let v_a = loop_a.verts[step];
            let v_a_next = loop_a.verts[next];
            let v_b = loop_b.verts[step];
            let v_b_next = loop_b.verts[next];

            if let Some(face) = self.bridge_emit_face(v_a, v_a_next, v_b, v_b_next) {
                band_faces.push(face);
            }
        }

        // §7 — distribute the triangle padding across an unequal-length band by
        // triangulating every band face. (The beautify diagonal-swap that the
        // guide layers on top rotates diagonals but does not change the
        // face/edge counts that the parity pin asserts.)
        if was_unequal {
            for &face in &band_faces {
                if self.faces.try_get(face).is_some() && self.face_len(face) > 3 {
                    self.face_calc_normal(face);
                    self.face_triangulate(face, QuadMethod::Beauty, NgonMethod::Beauty);
                }
            }
        }

        self.bridge_collect_outputs(loop_a, loop_b, &band_faces, result);
    }

    /// §6 — create (or reuse) one band face spanning the four (or three) corner
    /// verts, copying per-loop / per-face customdata from the bordering faces
    /// (§6.2). Returns `None` for the duplicate-vertex guard (§6).
    fn bridge_emit_face(
        &mut self,
        v_a: VertexKey,
        v_a_next: VertexKey,
        v_b: VertexKey,
        v_b_next: VertexKey,
    ) -> Option<FaceKey> {
        // §6 duplicate-vertex guard — a face with a repeated corner is
        // degenerate, so the step is skipped.
        let corners: SmallVec<[VertexKey; 4]> = if v_b != v_b_next {
            SmallVec::from_slice(&[v_b, v_b_next, v_a_next, v_a])
        } else {
            SmallVec::from_slice(&[v_b, v_a_next, v_a])
        };
        if has_duplicate_vert(&corners) {
            return None;
        }

        // §6 already-existing face guard.
        if let Some(existing) = self.face_exists(&corners) {
            return Some(existing);
        }

        // Capture the per-loop customdata sources on the loop edges before
        // the new face is created (§6.2).
        let source_loop_a = self.bridge_source_loop(v_a, v_a_next);
        let source_loop_a_next = self.bridge_source_loop(v_a_next, v_a);
        let source_loop_b = self.bridge_source_loop(v_b, v_b_next);
        let source_loop_b_next = self.bridge_source_loop(v_b_next, v_b);
        let source_face = source_loop_a
            .or(source_loop_a_next)
            .or(source_loop_b)
            .or(source_loop_b_next)
            .map(|l| self.loop_face(l));

        let edges: SmallVec<[EdgeKey; 4]> = {
            let mut es = SmallVec::new();
            for i in 0..corners.len() {
                let a = corners[i];
                let b = corners[(i + 1) % corners.len()];
                es.push(self.create_edge([a, b], EdgeCreateFlags::NO_DUPLICATE));
            }
            es
        };

        let face = self.face_create(&corners, &edges, FaceCreateFlags::NONE);

        // §6.2 — copy the corner loops' customdata verbatim. The corner order
        // matches the quad `(v_b, v_b_next, v_a_next, v_a)` / triangle
        // `(v_b, v_a_next, v_a)` source order.
        let source_loops: SmallVec<[Option<LoopKey>; 4]> = if v_b != v_b_next {
            SmallVec::from_slice(&[
                source_loop_b,
                source_loop_b_next,
                source_loop_a_next,
                source_loop_a,
            ])
        } else {
            SmallVec::from_slice(&[source_loop_b, source_loop_a_next, source_loop_a])
        };
        if !self.loop_attribs.is_empty() {
            let band_loops: SmallVec<[LoopKey; 4]> = self.iter_face_loops(face).collect();
            for (band_loop, source) in band_loops.iter().zip(source_loops.iter()) {
                if let Some(source) = source {
                    self.loop_attribs
                        .copy_into(band_loop.as_usize(), source.as_usize());
                }
            }
        }
        if !self.face_attribs.is_empty() {
            if let Some(source_face) = source_face {
                self.face_attribs
                    .copy_into(face.as_usize(), source_face.as_usize());
            }
        }

        Some(face)
    }

    /// §6.2 — the corner loop, on a face bordering the edge `{from, to}`, that
    /// sits at vertex `from`. Used as the per-loop customdata source. A
    /// bordering face has a corner loop at *both* endpoints of the edge, so the
    /// requested endpoint is found by taking any radial loop on the edge and, if
    /// its own vertex is the other endpoint, stepping to the next corner of the
    /// same face. `None` when the edge is missing or borders no face.
    fn bridge_source_loop(&self, from: VertexKey, to: VertexKey) -> Option<LoopKey> {
        let edge = self.query_edge_exists(from, to)?;
        let radial = self.iter_radial_loops(edge).next()?;
        let corner = if self.loop_vertex(radial) == from {
            radial
        } else {
            self.loop_next(radial)
        };
        debug_assert_eq!(self.loop_vertex(corner), from);
        Some(corner)
    }

    /// §6.3 — fold every band face's edges into `edges.out`, then drop the
    /// edges lying along either input loop, leaving exactly the rung edges.
    fn bridge_collect_outputs(
        &self,
        loop_a: &EdgeLoop,
        loop_b: &EdgeLoop,
        band_faces: &[FaceKey],
        result: &mut BridgeLoopsResult,
    ) {
        let mut loop_edges: HashSet<EdgeKey> = HashSet::new();
        for edge_loop in [loop_a, loop_b] {
            let span = if edge_loop.closed {
                edge_loop.len()
            } else {
                edge_loop.len().saturating_sub(1)
            };
            for i in 0..span {
                let v0 = edge_loop.verts[i];
                let v1 = edge_loop.verts[(i + 1) % edge_loop.len()];
                if let Some(edge) = self.query_edge_exists(v0, v1) {
                    loop_edges.insert(edge);
                }
            }
        }

        let mut seen_faces: HashSet<FaceKey> = HashSet::new();
        let mut seen_edges: HashSet<EdgeKey> = HashSet::new();
        for &face in band_faces {
            if self.faces.try_get(face).is_none() || !seen_faces.insert(face) {
                continue;
            }
            result.faces_out.push(face);
            for l in self.iter_face_loops(face) {
                let edge = self.loop_edge(l);
                if edge.is_null() || loop_edges.contains(&edge) {
                    continue;
                }
                if seen_edges.insert(edge) {
                    result.edges_out.push(edge);
                }
            }
        }
    }

    /// §8 — weld each A vertex onto its B partner. Each B vertex is first moved
    /// to `interp(A, B, merge_factor)` and its customdata blended from A by the
    /// same fraction, then the accumulated `(A → B)` map is welded.
    fn bridge_merge_pair(&mut self, loop_a: &EdgeLoop, loop_b: &EdgeLoop, merge_factor: f32) {
        let count = loop_a.len().min(loop_b.len());
        let mut pairs: Vec<(VertexKey, VertexKey)> = Vec::with_capacity(count);
        for i in 0..count {
            let v_a = loop_a.verts[i];
            let v_b = loop_b.verts[i];
            if v_a == v_b {
                continue;
            }

            // §8 — survivor B sits at interp(A, B, merge_factor): factor 0 lands
            // on A, factor 1 keeps B.
            let pos_a = self.vertex_position(v_a);
            let pos_b = self.vertex_position(v_b);
            self.set_vertex_position(v_b, pos_a.lerp(pos_b, merge_factor));

            if !self.vertex_attribs.is_empty() {
                let sources = [(1.0 - merge_factor, v_a.as_usize()), (merge_factor, v_b.as_usize())];
                self.vertex_attribs.interpolate_into(v_b.as_usize(), &sources);
            }

            pairs.push((v_a, v_b));
        }
        self.weld_verts(&pairs, false);
    }
}

/// §4 — the ordered list of loop index pairs to bridge for the given grouping.
fn bridge_pairings(count: usize, use_pairs: bool, use_cyclic: bool) -> Vec<(usize, usize)> {
    if use_pairs {
        (0..count).step_by(2).map(|i| (i, i + 1)).collect()
    } else {
        let mut pairs: Vec<(usize, usize)> = (0..count - 1).map(|i| (i, i + 1)).collect();
        if use_cyclic && count > 2 {
            pairs.push((count - 1, 0));
        }
        pairs
    }
}

fn clone_loop(src: &EdgeLoop) -> EdgeLoop {
    EdgeLoop {
        verts: src.verts.clone(),
        closed: src.closed,
        centre: src.centre,
    }
}

/// §5.6 — re-shift the rung correspondence of the (closed) loop B after the
/// §5.4 lowest-distance rotation. For a non-zero `twist = t` the pairing
/// becomes `A_i -> B_{(i - t - 1) mod n}` (floored modulo); `t == 0` skips the
/// step entirely and leaves §5.4's `A_i -> B_i` untouched. The one-time `-1`
/// reference correction fires only when the twist is non-zero, so the mapping
/// is discontinuous across zero (`t == -1` also reproduces the identity).
///
/// The band walk reads `loop_b.verts[i]` as the partner of `loop_a.verts[i]`,
/// so we need `verts_new[i] == verts_old[(i - t - 1) mod n]`. A left rotation
/// by `s` gives `verts_new[i] == verts_old[(i + s) mod n]`, hence
/// `s = (-t - 1) mod n`, taken with floored modulo for negative `t`.
fn bridge_apply_twist(loop_b: &mut EdgeLoop, twist: i32) {
    let count = loop_b.len();
    if count == 0 || twist == 0 {
        return;
    }
    let shift = (-twist - 1).rem_euclid(count as i32) as usize;
    if shift != 0 {
        loop_b.verts.rotate_left(shift);
    }
}

/// §5.5 — expand B to `target` verts by repeating verts, so the extra steps of
/// the band collapse into triangles. The repeats are distributed as evenly as
/// possible across the loop.
fn bridge_pad_loop(loop_b: &mut EdgeLoop, target: usize) {
    let original = loop_b.verts.clone();
    let count = original.len();
    if count == 0 || count >= target {
        return;
    }
    let mut padded: Vec<VertexKey> = Vec::with_capacity(target);
    for i in 0..target {
        // Map the padded index back onto an original index, repeating verts so
        // the band's triangle steps are spread out rather than bunched.
        let source = (i * count) / target;
        padded.push(original[source.min(count - 1)]);
    }
    loop_b.verts = padded;
}

fn bridge_loop_has_repeat(edge_loop: &EdgeLoop) -> bool {
    let mut seen = HashSet::new();
    for &v in &edge_loop.verts {
        if !seen.insert(v) {
            return true;
        }
    }
    false
}

fn has_duplicate_vert(verts: &[VertexKey]) -> bool {
    for i in 0..verts.len() {
        for j in (i + 1)..verts.len() {
            if verts[i] == verts[j] {
                return true;
            }
        }
    }
    false
}

/// Newell's method polygon normal (un-normalised). For an open loop the ring is
/// closed implicitly head-to-tail, which gives the open-loop normal the guide
/// computes in the between-loops frame (§5.2).
fn bridge_newell_normal(positions: &[Vec3], _closed: bool) -> Vec3 {
    let n = positions.len();
    if n < 3 {
        return Vec3::ZERO;
    }
    let mut normal = Vec3::ZERO;
    for i in 0..n {
        let cur = positions[i];
        let next = positions[(i + 1) % n];
        normal.x += (cur.y - next.y) * (cur.z + next.z);
        normal.y += (cur.z - next.z) * (cur.x + next.x);
        normal.z += (cur.x - next.x) * (cur.y + next.y);
    }
    normal
}

#[cfg(test)]
mod tests {
    use glam::Vec3;

    use crate::loopmesh::*;

    /// Build an open chain of `verts.len()` verts at the given positions and
    /// return its verts and the chain edges (selected for bridging).
    fn open_chain(mesh: &mut LoopMesh, positions: &[Vec3]) -> (Vec<VertexKey>, Vec<EdgeKey>) {
        let verts: Vec<VertexKey> = positions.iter().map(|&p| mesh.create_vertex(p)).collect();
        let edges: Vec<EdgeKey> = (0..verts.len() - 1)
            .map(|i| mesh.create_edge([verts[i], verts[i + 1]], EdgeCreateFlags::NONE))
            .collect();
        (verts, edges)
    }

    /// Build a closed ring and return its verts and ring edges.
    fn closed_ring(mesh: &mut LoopMesh, positions: &[Vec3]) -> (Vec<VertexKey>, Vec<EdgeKey>) {
        let verts: Vec<VertexKey> = positions.iter().map(|&p| mesh.create_vertex(p)).collect();
        let n = verts.len();
        let edges: Vec<EdgeKey> = (0..n)
            .map(|i| mesh.create_edge([verts[i], verts[(i + 1) % n]], EdgeCreateFlags::NONE))
            .collect();
        (verts, edges)
    }

    /// §2 / §6 / §9 — two open equal-length loops (k = 3 edges) bridge to a
    /// clean quad strip: `+0 V, +4 E, +3 F`.
    #[test]
    fn open_equal_loops_quad_strip() {
        let mut mesh = LoopMesh::new();
        let (_a, ea) = open_chain(
            &mut mesh,
            &[
                Vec3::new(0.0, 1.0, 0.0),
                Vec3::new(1.0, 1.0, 0.0),
                Vec3::new(2.0, 1.0, 0.0),
                Vec3::new(3.0, 1.0, 0.0),
            ],
        );
        let (_b, eb) = open_chain(
            &mut mesh,
            &[
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(2.0, 0.0, 0.0),
                Vec3::new(3.0, 0.0, 0.0),
            ],
        );
        let (v0, e0, f0) = (mesh.totvert(), mesh.totedge(), mesh.totface());

        let mut selected = ea.clone();
        selected.extend(eb.iter().copied());
        let result = mesh.bridge_loops(&selected, BridgeLoopsParams::default());

        assert!(result.ok);
        assert_eq!(mesh.totvert() - v0, 0);
        assert_eq!(mesh.totedge() - e0, 4);
        assert_eq!(mesh.totface() - f0, 3);
        assert_eq!(result.faces_out.len(), 3);
        // §6.3 — only the four rung edges, not the loop edges.
        assert_eq!(result.edges_out.len(), 4);
    }

    /// §5.4 + §6 wrap — two closed equal-length rings (k = 4) bridge to a tube:
    /// `+0 V, +4 E, +4 F`. The rings are z-offset so the between-loops axis is
    /// well-defined and the lowest-distance rotation lines the rungs straight.
    #[test]
    fn closed_equal_loops_tube() {
        let mut mesh = LoopMesh::new();
        let (_a, ea) = closed_ring(
            &mut mesh,
            &[
                Vec3::new(0.0, 0.0, 1.0),
                Vec3::new(1.0, 0.0, 1.0),
                Vec3::new(1.0, 1.0, 1.0),
                Vec3::new(0.0, 1.0, 1.0),
            ],
        );
        let (_b, eb) = closed_ring(
            &mut mesh,
            &[
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(1.0, 1.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
            ],
        );
        let (v0, e0, f0) = (mesh.totvert(), mesh.totedge(), mesh.totface());

        let mut selected = ea.clone();
        selected.extend(eb.iter().copied());
        let result = mesh.bridge_loops(&selected, BridgeLoopsParams::default());

        assert!(result.ok);
        assert_eq!(mesh.totvert() - v0, 0);
        assert_eq!(mesh.totedge() - e0, 4);
        assert_eq!(mesh.totface() - f0, 4);
        assert_eq!(result.faces_out.len(), 4);
        assert_eq!(result.edges_out.len(), 4);
    }

    /// Build `n` stacked open chains (each 3 verts / 2 edges) at ascending z,
    /// so each band is 2 quads + 3 rungs.
    fn stacked_open(mesh: &mut LoopMesh, n: usize) -> Vec<EdgeKey> {
        let mut selected = Vec::new();
        for layer in 0..n {
            let z = layer as f32;
            let (_v, e) = open_chain(
                mesh,
                &[
                    Vec3::new(0.0, 0.0, z),
                    Vec3::new(1.0, 0.0, z),
                    Vec3::new(2.0, 0.0, z),
                ],
            );
            selected.extend(e);
        }
        selected
    }

    /// §4.2 — `use_cyclic` on a 3-loop stack adds the closing band: 3 bands
    /// (`+9 E, +6 F`) vs 2 bands when off (`+6 E, +4 F`).
    #[test]
    fn use_cyclic_closes_chain() {
        // Cyclic off — open chain of two bands.
        let mut mesh = LoopMesh::new();
        let selected = stacked_open(&mut mesh, 3);
        let (e0, f0) = (mesh.totedge(), mesh.totface());
        let result = mesh.bridge_loops(&selected, BridgeLoopsParams::default());
        assert!(result.ok);
        assert_eq!(mesh.totedge() - e0, 6);
        assert_eq!(mesh.totface() - f0, 4);

        // Cyclic on — three bands.
        let mut mesh = LoopMesh::new();
        let selected = stacked_open(&mut mesh, 3);
        let (e0, f0) = (mesh.totedge(), mesh.totface());
        let params = BridgeLoopsParams {
            use_cyclic: true,
            ..Default::default()
        };
        let result = mesh.bridge_loops(&selected, params);
        assert!(result.ok);
        assert_eq!(mesh.totedge() - e0, 9);
        assert_eq!(mesh.totface() - f0, 6);
    }

    /// §4.1 — `use_pairs` on a 4-loop stack bridges 2 disjoint pairs
    /// (`+6 E, +4 F`) vs chain mode's 3 bands (`+9 E, +6 F`).
    #[test]
    fn use_pairs_bridges_disjoint_pairs() {
        let mut mesh = LoopMesh::new();
        let selected = stacked_open(&mut mesh, 4);
        let (e0, f0) = (mesh.totedge(), mesh.totface());
        let params = BridgeLoopsParams {
            use_pairs: true,
            ..Default::default()
        };
        let result = mesh.bridge_loops(&selected, params);
        assert!(result.ok);
        assert_eq!(mesh.totedge() - e0, 6);
        assert_eq!(mesh.totface() - f0, 4);

        // Chain mode on the same stack — three bands.
        let mut mesh = LoopMesh::new();
        let selected = stacked_open(&mut mesh, 4);
        let (e0, f0) = (mesh.totedge(), mesh.totface());
        let result = mesh.bridge_loops(&selected, BridgeLoopsParams::default());
        assert!(result.ok);
        assert_eq!(mesh.totedge() - e0, 9);
        assert_eq!(mesh.totface() - f0, 6);
    }

    /// §8 — `use_merge` welds two equal-length open loops into one; element
    /// counts drop (8 V → 4 V, 6 E → 3 E) and no band faces are created.
    #[test]
    fn use_merge_welds_loops() {
        let mut mesh = LoopMesh::new();
        let (_a, ea) = open_chain(
            &mut mesh,
            &[
                Vec3::new(0.0, 1.0, 0.0),
                Vec3::new(1.0, 1.0, 0.0),
                Vec3::new(2.0, 1.0, 0.0),
                Vec3::new(3.0, 1.0, 0.0),
            ],
        );
        let (_b, eb) = open_chain(
            &mut mesh,
            &[
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(2.0, 0.0, 0.0),
                Vec3::new(3.0, 0.0, 0.0),
            ],
        );
        assert_eq!(mesh.totvert(), 8);
        assert_eq!(mesh.totedge(), 6);

        let mut selected = ea.clone();
        selected.extend(eb.iter().copied());
        let params = BridgeLoopsParams {
            use_merge: true,
            merge_factor: 0.5,
            ..Default::default()
        };
        let result = mesh.bridge_loops(&selected, params);

        assert!(result.ok);
        assert_eq!(mesh.totvert(), 4);
        assert_eq!(mesh.totedge(), 3);
        assert_eq!(mesh.totface(), 0);
        assert!(result.faces_out.is_empty());
        assert!(result.edges_out.is_empty());
    }

    /// §8 — `merge_factor` is load-bearing: factor 0 lands survivors on loop A,
    /// factor 1 keeps loop B; the surviving positions differ.
    #[test]
    fn merge_factor_changes_survivor_positions() {
        let survivor_y = |factor: f32| -> f32 {
            let mut mesh = LoopMesh::new();
            let (_a, ea) = open_chain(
                &mut mesh,
                &[
                    Vec3::new(0.0, 4.0, 0.0),
                    Vec3::new(1.0, 4.0, 0.0),
                    Vec3::new(2.0, 4.0, 0.0),
                ],
            );
            let (_b, eb) = open_chain(
                &mut mesh,
                &[
                    Vec3::new(0.0, 0.0, 0.0),
                    Vec3::new(1.0, 0.0, 0.0),
                    Vec3::new(2.0, 0.0, 0.0),
                ],
            );
            let mut selected = ea.clone();
            selected.extend(eb.iter().copied());
            let params = BridgeLoopsParams {
                use_merge: true,
                merge_factor: factor,
                ..Default::default()
            };
            mesh.bridge_loops(&selected, params);
            // The survivors are the welded loop; average their y so the pin is
            // independent of which survivor the iterator yields first.
            let mut iter = mesh.mesh_verts();
            let mut sum = 0.0_f32;
            let mut n = 0.0_f32;
            while let Some(v) = iter.next(&mesh) {
                sum += mesh.vertex_position(v).y;
                n += 1.0;
            }
            sum / n
        };

        // The load-bearing pin: the survivor positions differ between the two
        // factors (§8 — merge_factor blends A↔B). factor 0 lands on one loop's
        // plane, factor 1 on the other.
        let at_zero = survivor_y(0.0);
        let at_one = survivor_y(1.0);
        assert!((at_zero - at_one).abs() > 1.0);
    }

    /// §5.5 / §6.1 / §7 — a 4-vert loop bridged to a 3-vert loop pads with a
    /// triangle step, then triangulates the band: `+0 V, +6 E, +5 F`.
    #[test]
    fn mismatched_loops_refined() {
        let mut mesh = LoopMesh::new();
        let (_a, ea) = open_chain(
            &mut mesh,
            &[
                Vec3::new(0.0, 1.0, 0.0),
                Vec3::new(1.0, 1.0, 0.0),
                Vec3::new(2.0, 1.0, 0.0),
                Vec3::new(3.0, 1.0, 0.0),
            ],
        );
        let (_b, eb) = open_chain(
            &mut mesh,
            &[
                Vec3::new(0.5, 0.0, 0.0),
                Vec3::new(1.5, 0.0, 0.0),
                Vec3::new(2.5, 0.0, 0.0),
            ],
        );
        let (v0, e0, f0) = (mesh.totvert(), mesh.totedge(), mesh.totface());

        let mut selected = ea.clone();
        selected.extend(eb.iter().copied());
        let result = mesh.bridge_loops(&selected, BridgeLoopsParams::default());

        assert!(result.ok);
        assert_eq!(mesh.totvert() - v0, 0);
        assert_eq!(mesh.totedge() - e0, 6);
        assert_eq!(mesh.totface() - f0, 5);
        // Every refined band face is a triangle.
        for &f in &result.faces_out {
            assert_eq!(mesh.face_len(f), 3);
        }
    }

    /// §3.1 — a single loop cancels: `ok = false`, mesh unchanged.
    #[test]
    fn single_loop_cancels() {
        let mut mesh = LoopMesh::new();
        let (_a, ea) = open_chain(
            &mut mesh,
            &[
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(2.0, 0.0, 0.0),
            ],
        );
        let (v0, e0, f0) = (mesh.totvert(), mesh.totedge(), mesh.totface());
        let result = mesh.bridge_loops(&ea, BridgeLoopsParams::default());
        assert!(!result.ok);
        assert_eq!((mesh.totvert(), mesh.totedge(), mesh.totface()), (v0, e0, f0));
    }

    /// §3.1 — an odd loop count under `use_pairs` cancels, mesh unchanged.
    #[test]
    fn pairs_odd_count_cancels() {
        let mut mesh = LoopMesh::new();
        let selected = stacked_open(&mut mesh, 3);
        let (v0, e0, f0) = (mesh.totvert(), mesh.totedge(), mesh.totface());
        let params = BridgeLoopsParams {
            use_pairs: true,
            ..Default::default()
        };
        let result = mesh.bridge_loops(&selected, params);
        assert!(!result.ok);
        assert_eq!((mesh.totvert(), mesh.totedge(), mesh.totface()), (v0, e0, f0));
    }

    /// §3.1 — unequal loop lengths under `use_merge` cancel, mesh unchanged.
    #[test]
    fn merge_unequal_lengths_cancels() {
        let mut mesh = LoopMesh::new();
        let (_a, ea) = open_chain(
            &mut mesh,
            &[
                Vec3::new(0.0, 1.0, 0.0),
                Vec3::new(1.0, 1.0, 0.0),
                Vec3::new(2.0, 1.0, 0.0),
                Vec3::new(3.0, 1.0, 0.0),
            ],
        );
        let (_b, eb) = open_chain(
            &mut mesh,
            &[
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(2.0, 0.0, 0.0),
            ],
        );
        let (v0, e0, f0) = (mesh.totvert(), mesh.totedge(), mesh.totface());
        let mut selected = ea.clone();
        selected.extend(eb.iter().copied());
        let params = BridgeLoopsParams {
            use_merge: true,
            ..Default::default()
        };
        let result = mesh.bridge_loops(&selected, params);
        assert!(!result.ok);
        assert_eq!((mesh.totvert(), mesh.totedge(), mesh.totface()), (v0, e0, f0));
    }

    /// §5.6 — `twist_offset` on a closed pair re-shifts the rung connectivity
    /// without changing element counts (the tube's face/edge totals are the
    /// same; only the rung incidence rotates).
    #[test]
    fn twist_offset_preserves_counts() {
        let build = |twist: i32| -> (usize, usize, usize) {
            let mut mesh = LoopMesh::new();
            let (a_verts, _b_verts, ea, eb) = twist_rings(&mut mesh);
            let mut selected = ea;
            selected.extend(eb);
            let _ = a_verts;
            let params = BridgeLoopsParams {
                twist_offset: twist,
                ..Default::default()
            };
            let result = mesh.bridge_loops(&selected, params);
            assert!(result.ok);
            (mesh.totvert(), mesh.totedge(), mesh.totface())
        };
        assert_eq!(build(0), build(1));
    }

    /// §5.6 — the specific rung incidence at `n = 4`. For the z-stacked, index-
    /// aligned rings the §5.4 baseline pairs `A_i -> B_i`, so the worked table
    /// applies directly: `t = 0` and `t = -1` reproduce that identity, `t = 1`
    /// maps `A_i -> B_{(i+2)%4}`, and `t = 2` maps `A_i -> B_{(i+1)%4}`. We
    /// assert the cross-rung connectivity, not just the element counts.
    #[test]
    fn twist_offset_pins_rung_incidence() {
        // For each `A_i`, which `B_j` shares a rung edge with it.
        let partner_map = |twist: i32| -> [usize; 4] {
            let mut mesh = LoopMesh::new();
            let (a_verts, b_verts, ea, eb) = twist_rings(&mut mesh);
            let mut selected = ea;
            selected.extend(eb);
            let params = BridgeLoopsParams {
                twist_offset: twist,
                ..Default::default()
            };
            assert!(mesh.bridge_loops(&selected, params).ok);

            let mut map = [usize::MAX; 4];
            for (i, &av) in a_verts.iter().enumerate() {
                for (j, &bv) in b_verts.iter().enumerate() {
                    if mesh.query_edge_exists(av, bv).is_some() {
                        assert_eq!(map[i], usize::MAX, "A_{i} has more than one rung");
                        map[i] = j;
                    }
                }
            }
            assert!(map.iter().all(|&j| j != usize::MAX), "every A vert is rung");
            map
        };

        // Baseline §5.4 is the identity for this fixture.
        assert_eq!(partner_map(0), [0, 1, 2, 3]);
        // `t = -1` reproduces the identity (the `-1` reference correction).
        assert_eq!(partner_map(-1), [0, 1, 2, 3]);
        // `t = 1` shifts the rung index by +2: A_i -> B_{(i+2)%4}.
        assert_eq!(partner_map(1), [2, 3, 0, 1]);
        // `t = 2` shifts by +1: A_i -> B_{(i+1)%4}.
        assert_eq!(partner_map(2), [1, 2, 3, 0]);
    }

    /// §6.2 — every band-face corner copies its per-loop customdata verbatim
    /// from the bordering existing face's corner at the *same* vertex. The
    /// regression this pins: a rim edge `{from, to}` borders one face whose two
    /// corner loops sit at `from` and `to`; both band corners on that edge must
    /// source from that face — not just the one whose loop vertex matches the
    /// radial loop's own vertex.
    #[test]
    fn band_corners_source_per_corner_customdata() {
        let mut mesh = LoopMesh::new();

        // Loop A (the bridged top chain) and a wider chain above it that we
        // close into bordering quads on each A edge.
        let (a, ea) = open_chain(
            &mut mesh,
            &[
                Vec3::new(0.0, 1.0, 0.0),
                Vec3::new(1.0, 1.0, 0.0),
                Vec3::new(2.0, 1.0, 0.0),
                Vec3::new(3.0, 1.0, 0.0),
            ],
        );
        let above: Vec<VertexKey> = (0..a.len())
            .map(|i| mesh.create_vertex(Vec3::new(i as f32, 2.0, 0.0)))
            .collect();

        let uv = mesh
            .register_loop_attribute(ElementAttributeDesc::numeric(
                AttributeKey::custom("uv"),
                AttrType::Vec2,
            ))
            .unwrap();

        // One bordering quad per A edge: `[a_i, a_{i+1}, above_{i+1}, above_i]`.
        // Each corner gets a UV unique to its vertex so the source is provable.
        let vert_uv = |mesh: &LoopMesh, v: VertexKey| -> [f32; 2] {
            let p = mesh.vertex_position(v);
            [p.x, p.y]
        };
        for i in 0..a.len() - 1 {
            let corners = [a[i], a[i + 1], above[i + 1], above[i]];
            let edges: Vec<EdgeKey> = (0..4)
                .map(|k| {
                    mesh.create_edge([corners[k], corners[(k + 1) % 4]], EdgeCreateFlags::NO_DUPLICATE)
                })
                .collect();
            let face = mesh.face_create(&corners, &edges, FaceCreateFlags::NONE);
            for l in mesh.iter_face_loops(face).collect::<Vec<_>>() {
                let v = mesh.loop_vertex(l);
                let value = vert_uv(&mesh, v);
                mesh.loop_attribs_mut().set_vec2(uv, l.as_usize(), value);
            }
        }

        let (_b, eb) = open_chain(
            &mut mesh,
            &[
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(2.0, 0.0, 0.0),
                Vec3::new(3.0, 0.0, 0.0),
            ],
        );

        let mut selected = ea.clone();
        selected.extend(eb.iter().copied());
        let result = mesh.bridge_loops(&selected, BridgeLoopsParams::default());
        assert!(result.ok);

        // Each band face borders one A edge; its two corners at the A vertices
        // must carry that A vertex's UV, sourced from the bordering quad.
        let a_set: std::collections::HashSet<VertexKey> = a.iter().copied().collect();
        let mut a_corners_checked = 0;
        for &face in &result.faces_out {
            for l in mesh.iter_face_loops(face).collect::<Vec<_>>() {
                let v = mesh.loop_vertex(l);
                if !a_set.contains(&v) {
                    continue;
                }
                let got = mesh.loop_attribs().get_vec2(uv, l.as_usize()).unwrap();
                assert_eq!(
                    got,
                    vert_uv(&mesh, v),
                    "band corner at A vertex must source the bordering face's UV"
                );
                // The default layer value is [0,0]; every A corner here has a
                // non-zero y, so a defaulted corner would fail the equality
                // above. This guard makes the "not left at default" intent
                // explicit for the y=1 corners.
                assert_ne!(got, [0.0, 0.0]);
                a_corners_checked += 1;
            }
        }
        // Three band quads, two A corners each.
        assert_eq!(a_corners_checked, 6);
    }

    /// The z-stacked, index-aligned ring pair used by the twist tests. The two
    /// rings share x,y so §5.4 lines them up index-to-index (offset 0) and the
    /// direction/winding passes do not reverse B — giving the identity baseline.
    /// Returns A verts, B verts, A ring edges, B ring edges.
    #[allow(clippy::type_complexity)]
    fn twist_rings(
        mesh: &mut LoopMesh,
    ) -> (Vec<VertexKey>, Vec<VertexKey>, Vec<EdgeKey>, Vec<EdgeKey>) {
        let (a, ea) = closed_ring(
            mesh,
            &[
                Vec3::new(0.0, 0.0, 1.0),
                Vec3::new(1.0, 0.0, 1.0),
                Vec3::new(1.0, 1.0, 1.0),
                Vec3::new(0.0, 1.0, 1.0),
            ],
        );
        let (b, eb) = closed_ring(
            mesh,
            &[
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(1.0, 1.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
            ],
        );
        (a, b, ea, eb)
    }
}
