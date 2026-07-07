use std::collections::HashSet;

use glam::Vec3;

use crate::loopmesh::*;

/// The `edges.out` of [`LoopMesh::edgenet_prepare`]: the **whole prepared net** —
/// every input edge that survived validation *plus* any new connecting edge(s),
/// deduplicated
/// ([`EDGENET_PREPARE.md`](../../../../radial-mesh-guides/operators/EDGENET_PREPARE.md)
/// §6.1). So a single open chain reports its chain edges plus one bridge; a
/// facing pair reports both chains' edges plus two bridges; an already-closed
/// loop or a lone single edge reports just its (unchanged) input edges; only a
/// branch-rejected input (§3) reports an empty set.
///
/// The no-duplicate wrinkle (§5.3): a connecting edge that coincides with a
/// pre-existing edge is reported here **without** a new edge being created — the
/// reused edge is surfaced in the slot even though the mesh edge set is
/// unchanged for that bridge.
#[derive(Clone, Debug, Default)]
pub struct EdgenetPrepareResult {
    edges: Vec<EdgeKey>,
}

impl EdgenetPrepareResult {
    pub fn edges(&self) -> &[EdgeKey] {
        &self.edges
    }
    pub fn is_empty(&self) -> bool {
        self.edges.is_empty()
    }
}

/// An ordered open chain traced out of the net: the edges from one free end to
/// the other, plus the two free-endpoint vertices (§4 / §5).
struct OpenChain {
    edges: Vec<EdgeKey>,
    start: VertexKey,
    end: VertexKey,
}

impl LoopMesh {
    /// Close an *almost-closed* edge net — one or two **open chains** with
    /// dangling free endpoints — by adding the one or two **connecting edges**
    /// that make it a fillable loop (the input-side companion to
    /// [`LoopMesh::edgenet_fill`], which caps the closed result). Prepare is
    /// purely additive of *edges*: it creates no vertex, no face, and deletes
    /// nothing (§2).
    ///
    /// Algorithm in
    /// [`EDGENET_PREPARE.md`](../../../../radial-mesh-guides/operators/EDGENET_PREPARE.md):
    /// §3 tags the input `edges` as the working net and rejects the **whole**
    /// input if any vertex is incident to three or more tagged edges (a branch /
    /// T-junction — prepare does **not** split it); §4 traces at most two open
    /// chains (a closed component is never seeded, and the §4.1 guard drops any
    /// chain that closed on itself); §5 bridges the free endpoints — one chain of
    /// more than one edge gets one connecting edge (§5.1; a lone single edge is
    /// left alone), a facing pair gets two connecting edges chosen to avoid a
    /// self-crossing bowtie (§5.2). Every bridge is made with a no-duplicate
    /// guarantee (§5.3).
    ///
    /// Returns the [`EdgenetPrepareResult`] `edges.out` slot: the whole prepared
    /// net (input edges + bridges, deduped), empty only on branch rejection
    /// (§6.1).
    pub fn edgenet_prepare(&mut self, edges: &[EdgeKey]) -> EdgenetPrepareResult {
        // §3 — tag the working net. Operator-internal tagging only; no persistent
        // flag survives the call.
        let tagged: HashSet<EdgeKey> = edges.iter().copied().collect();

        // §3 — validation pass: any vertex incident to three or more tagged edges
        // is a branch the operator cannot handle as a simple chain, so the WHOLE
        // input is rejected (all-or-nothing, not per-component) and nothing —
        // not even the inputs — is reported.
        for &edge in &tagged {
            for vertex in self.edge_vert_keys(edge) {
                if self.edgenet_prepare_tagged_degree(vertex, &tagged) > 2 {
                    return EdgenetPrepareResult::default();
                }
            }
        }

        // §4 — trace at most two open chains, then apply the §4.1 already-closed
        // guard.
        let chains = self.edgenet_prepare_trace_chains(edges, &tagged);

        // §5 — bridge the free endpoints of the surviving chains.
        let mut bridges: Vec<EdgeKey> = Vec::new();
        match chains.as_slice() {
            [] => {}
            [chain] => {
                // §5.1 — one chain of more than one edge closes with one bridge;
                // a lone single edge is left alone (bridging would duplicate it).
                if chain.edges.len() > 1 {
                    bridges.push(
                        self.create_edge([chain.start, chain.end], EdgeCreateFlags::NO_DUPLICATE),
                    );
                }
            }
            [first, second, ..] => {
                // §5.2 — two chains close with two bridges, the endpoint pairing
                // chosen to avoid a self-crossing bowtie.
                let (bridge_a, bridge_b) = self.edgenet_prepare_clean_pairing(first, second);
                bridges.push(self.create_edge(bridge_a, EdgeCreateFlags::NO_DUPLICATE));
                bridges.push(self.create_edge(bridge_b, EdgeCreateFlags::NO_DUPLICATE));
            }
        }

        // §6.1 — the slot reports the whole prepared net: every input edge plus
        // the bridges, deduplicated. A reused pre-existing bridge (§5.3) is
        // surfaced here even though no edge was created.
        let mut seen: HashSet<EdgeKey> = HashSet::with_capacity(edges.len() + bridges.len());
        let mut net: Vec<EdgeKey> = Vec::with_capacity(edges.len() + bridges.len());
        for &edge in edges.iter().chain(bridges.iter()) {
            if seen.insert(edge) {
                net.push(edge);
            }
        }
        EdgenetPrepareResult { edges: net }
    }

    /// Count how many **tagged** edges touch `vertex` (§3 / §4). Interior chain
    /// vertices count 2, free endpoints count 1; a count above 2 is a branch.
    fn edgenet_prepare_tagged_degree(&self, vertex: VertexKey, tagged: &HashSet<EdgeKey>) -> usize {
        self.iter_disk_edges(vertex)
            .filter(|edge| tagged.contains(edge))
            .count()
    }

    /// The endpoint of `edge` that is not `vertex`.
    fn edgenet_prepare_other_vert(&self, edge: EdgeKey, vertex: VertexKey) -> VertexKey {
        let [a, b] = self.edge_vert_keys(edge);
        if a == vertex {
            b
        } else {
            a
        }
    }

    /// §4 — seed a new chain at each unvisited tagged edge that has a free
    /// endpoint (tagged-degree exactly one) and walk it to its other free end,
    /// collecting **at most two** traced chains, then apply the §4.1 guard that
    /// drops any chain that closed on itself. Closed components (no free
    /// endpoint) are never seeded.
    fn edgenet_prepare_trace_chains(
        &self,
        edges: &[EdgeKey],
        tagged: &HashSet<EdgeKey>,
    ) -> Vec<OpenChain> {
        let mut visited: HashSet<EdgeKey> = HashSet::with_capacity(tagged.len());
        let mut traced: Vec<OpenChain> = Vec::new();

        for &seed in edges {
            // §4 — the hard two-chain limit: once two chains are traced, no
            // further seeding happens (a third-and-beyond chain is left open).
            if traced.len() == 2 {
                break;
            }
            if visited.contains(&seed) {
                continue;
            }
            let [v0, v1] = self.edge_vert_keys(seed);
            let free = if self.edgenet_prepare_tagged_degree(v0, tagged) == 1 {
                v0
            } else if self.edgenet_prepare_tagged_degree(v1, tagged) == 1 {
                v1
            } else {
                // Not a chain end (interior edge, or part of a closed loop with no
                // free endpoint) — never seeded here.
                continue;
            };
            traced.push(self.edgenet_prepare_walk(seed, free, tagged, &mut visited));
        }

        // §4.1 — drop any traced chain that turned out closed (its two free
        // endpoints coincide); order-preserving `retain` promotes a surviving
        // secondary to primary. With a validated net this never fires, but a
        // faithful implementation includes it so no-op cases agree.
        traced.retain(|chain| chain.start != chain.end);
        traced
    }

    /// §4 — walk from `seed` (whose free endpoint is `free`) hop-by-hop to the
    /// chain's other free end, recording the ordered edge list and marking each
    /// visited so it is never re-walked. Each vertex has tagged-degree at most
    /// two (validation, §3), so at each frontier there is at most one unvisited
    /// tagged neighbour to follow.
    fn edgenet_prepare_walk(
        &self,
        seed: EdgeKey,
        free: VertexKey,
        tagged: &HashSet<EdgeKey>,
        visited: &mut HashSet<EdgeKey>,
    ) -> OpenChain {
        visited.insert(seed);
        let mut chain_edges = vec![seed];
        let mut frontier = self.edgenet_prepare_other_vert(seed, free);

        loop {
            let next = self
                .iter_disk_edges(frontier)
                .find(|edge| tagged.contains(edge) && !visited.contains(edge));
            match next {
                Some(next_edge) => {
                    visited.insert(next_edge);
                    chain_edges.push(next_edge);
                    frontier = self.edgenet_prepare_other_vert(next_edge, frontier);
                }
                None => break,
            }
        }

        OpenChain {
            edges: chain_edges,
            start: free,
            end: frontier,
        }
    }

    /// §5.2 — pick the bowtie-avoiding endpoint pairing for two chains with free
    /// ends `(a, b)` and `(c, d)`. The two ways to close the gap are `{a-c, b-d}`
    /// and `{a-d, b-c}`; one threads a clean ring, the other crosses into a
    /// self-intersecting bowtie. Each pairing closes the quad on the four
    /// endpoints along one diagonal, splitting it into two triangles; the clean
    /// pairing is the one whose two triangle normals are more consistently wound
    /// (larger normalized dot). Deterministic in the four endpoint positions.
    fn edgenet_prepare_clean_pairing(
        &self,
        first: &OpenChain,
        second: &OpenChain,
    ) -> ([VertexKey; 2], [VertexKey; 2]) {
        let a = first.start;
        let b = first.end;
        let c = second.start;
        let d = second.end;

        let pa = self.vertex_position(a);
        let pb = self.vertex_position(b);
        let pc = self.vertex_position(c);
        let pd = self.vertex_position(d);

        // Candidate {a-c, b-d}: ring a -> b -> d -> c.
        let dot_ac_bd = triangle_pair_normal_dot(pa, pb, pd, pc);
        // Candidate {a-d, b-c}: ring a -> b -> c -> d.
        let dot_ad_bc = triangle_pair_normal_dot(pa, pb, pc, pd);

        if dot_ac_bd >= dot_ad_bc {
            ([a, c], [b, d])
        } else {
            ([a, d], [b, c])
        }
    }
}

/// How consistently wound the two triangles of the quad `p0 -> p1 -> p2 -> p3`
/// are, split along the `p0..p2` diagonal: the dot of the two triangles' unit
/// normals. Close to `+1` for a near-planar, non-crossing ring; small or
/// negative when the ring folds or crosses on itself (the bowtie). See §5.2.
fn triangle_pair_normal_dot(p0: Vec3, p1: Vec3, p2: Vec3, p3: Vec3) -> f32 {
    let normal_first = (p1 - p0).cross(p2 - p0).normalize_or_zero();
    let normal_second = (p2 - p0).cross(p3 - p0).normalize_or_zero();
    normal_first.dot(normal_second)
}

#[cfg(test)]
mod tests {
    use glam::Vec3;

    use crate::loopmesh::*;

    /// Build a straight open wire chain of `n` edges along +x and return the
    /// chain edges plus its two free-end vertices.
    fn open_chain(mesh: &mut LoopMesh, n: usize, y: f32) -> (Vec<EdgeKey>, VertexKey, VertexKey) {
        let verts: Vec<VertexKey> = (0..=n)
            .map(|i| mesh.create_vertex(Vec3::new(i as f32, y, 0.0)))
            .collect();
        let edges: Vec<EdgeKey> = (0..n)
            .map(|i| mesh.create_edge([verts[i], verts[i + 1]], EdgeCreateFlags::NO_DUPLICATE))
            .collect();
        (edges, verts[0], verts[n])
    }

    #[test]
    fn single_open_chain_bridges_one_edge() {
        // §5.1 — a single open chain of more than one edge is closed by one new
        // edge between its two free ends.
        let mut mesh = LoopMesh::new();
        let (chain, free_start, free_end) = open_chain(&mut mesh, 3, 0.0);

        let out = mesh.edgenet_prepare(&chain);

        assert_eq!(mesh.vertex_count(), 4, "no vertex created");
        assert_eq!(mesh.edge_count(), 4, "exactly one bridge added");
        assert_eq!(mesh.face_count(), 0, "prepare never makes faces");
        assert!(
            mesh.query_edge_exists(free_start, free_end).is_some(),
            "the two free ends are now joined",
        );
        // §6.1 — the slot reports the whole prepared net: 3 chain edges + 1 bridge.
        assert_eq!(out.edges().len(), 4, "edges.out is the whole prepared net");
        mesh.validate_topology();
    }

    #[test]
    fn lone_single_edge_creates_nothing() {
        // §5.1 / §7 — a lone single edge is left alone: joining its two ends
        // would merely duplicate it.
        let mut mesh = LoopMesh::new();
        let (chain, _, _) = open_chain(&mut mesh, 1, 0.0);

        let out = mesh.edgenet_prepare(&chain);

        assert_eq!(mesh.edge_count(), 1, "no bridge added for a lone edge");
        // §6.1 — a lone edge still reports its (unchanged) input in the slot.
        assert_eq!(out.edges().len(), 1, "slot is the unchanged input, not empty");
        assert!(!out.is_empty());
        mesh.validate_topology();
    }

    #[test]
    fn closed_loop_creates_nothing() {
        // §4 / §4.1 — a closed loop has no free endpoint, so it is never seeded
        // and gets no connecting edge.
        let mut mesh = LoopMesh::new();
        let verts: Vec<VertexKey> = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        ]
        .into_iter()
        .map(|p| mesh.create_vertex(p))
        .collect();
        let loop_edges: Vec<EdgeKey> = (0..4)
            .map(|i| {
                mesh.create_edge(
                    [verts[i], verts[(i + 1) % 4]],
                    EdgeCreateFlags::NO_DUPLICATE,
                )
            })
            .collect();

        let out = mesh.edgenet_prepare(&loop_edges);

        assert_eq!(mesh.edge_count(), 4, "already-closed loop gains no edge");
        assert_eq!(out.edges().len(), 4, "slot reports the unchanged loop edges");
        mesh.validate_topology();
    }

    #[test]
    fn facing_pair_bridges_clean_ring() {
        // §5.2 — two facing chains (ends left/right on each) are joined by two
        // vertical bridges, not the crossing diagonals.
        let mut mesh = LoopMesh::new();
        let (top, top_left, top_right) = open_chain(&mut mesh, 2, 1.0);
        let (bottom, bottom_left, bottom_right) = open_chain(&mut mesh, 2, 0.0);
        let net: Vec<EdgeKey> = top.iter().chain(bottom.iter()).copied().collect();

        let out = mesh.edgenet_prepare(&net);

        assert_eq!(mesh.edge_count(), 6, "two bridges added to the four chain edges");
        assert!(mesh.query_edge_exists(top_left, bottom_left).is_some(), "left-to-left");
        assert!(mesh.query_edge_exists(top_right, bottom_right).is_some(), "right-to-right");
        assert!(
            mesh.query_edge_exists(top_left, bottom_right).is_none(),
            "the crossing diagonal is not created",
        );
        assert!(
            mesh.query_edge_exists(top_right, bottom_left).is_none(),
            "the crossing diagonal is not created",
        );
        assert_eq!(out.edges().len(), 6, "whole net = 4 chain edges + 2 bridges");
        mesh.validate_topology();
    }

    #[test]
    fn crossing_pair_picks_clean_ring_not_bowtie() {
        // §5.2 — the upper chain is reversed in x, so a naive "in-order" pairing
        // would cross. The bowtie-avoidance heuristic picks the clean vertical
        // ring instead.
        let mut mesh = LoopMesh::new();
        // Upper chain reversed in x: its start end sits on the right (x=2), its
        // far end on the left (x=0).
        let top_right = mesh.create_vertex(Vec3::new(2.0, 1.0, 0.0));
        let top_mid = mesh.create_vertex(Vec3::new(1.0, 1.0, 0.0));
        let top_left = mesh.create_vertex(Vec3::new(0.0, 1.0, 0.0));
        let te0 = mesh.create_edge([top_right, top_mid], EdgeCreateFlags::NO_DUPLICATE);
        let te1 = mesh.create_edge([top_mid, top_left], EdgeCreateFlags::NO_DUPLICATE);
        let (bottom, bottom_left, bottom_right) = open_chain(&mut mesh, 2, 0.0);
        let net: Vec<EdgeKey> = [te0, te1].into_iter().chain(bottom.iter().copied()).collect();

        let out = mesh.edgenet_prepare(&net);

        assert_eq!(mesh.edge_count(), 6, "exactly two bridges added");
        // Clean vertical ring: right-to-right and left-to-left.
        assert!(mesh.query_edge_exists(top_right, bottom_right).is_some(), "vertical right");
        assert!(mesh.query_edge_exists(top_left, bottom_left).is_some(), "vertical left");
        // The bowtie diagonals must NOT appear.
        assert!(
            mesh.query_edge_exists(top_right, bottom_left).is_none(),
            "bowtie diagonal rejected",
        );
        assert!(
            mesh.query_edge_exists(top_left, bottom_right).is_none(),
            "bowtie diagonal rejected",
        );
        assert_eq!(out.edges().len(), 6);
        mesh.validate_topology();
    }

    #[test]
    fn branch_net_creates_nothing() {
        // §3 — a vertex incident to three tagged edges is a branch: the whole
        // input is rejected and nothing is created, with an empty slot.
        let mut mesh = LoopMesh::new();
        let center = mesh.create_vertex(Vec3::new(0.0, 0.0, 0.0));
        let arm_a = mesh.create_vertex(Vec3::new(1.0, 0.0, 0.0));
        let arm_b = mesh.create_vertex(Vec3::new(-1.0, 0.0, 0.0));
        let arm_c = mesh.create_vertex(Vec3::new(0.0, 1.0, 0.0));
        let net = [
            mesh.create_edge([center, arm_a], EdgeCreateFlags::NO_DUPLICATE),
            mesh.create_edge([center, arm_b], EdgeCreateFlags::NO_DUPLICATE),
            mesh.create_edge([center, arm_c], EdgeCreateFlags::NO_DUPLICATE),
        ];

        let out = mesh.edgenet_prepare(&net);

        assert_eq!(mesh.edge_count(), 3, "branch rejection creates nothing");
        assert!(out.is_empty(), "a branch-rejected input reports an empty slot");
        mesh.validate_topology();
    }

    #[test]
    fn three_chains_only_first_two_bridged() {
        // §4 — the hard two-chain limit: three disjoint open chains, but only the
        // first two are bridged (two new edges), the third left open.
        let mut mesh = LoopMesh::new();
        let (chain0, _, _) = open_chain(&mut mesh, 2, 0.0);
        let (chain1, _, _) = open_chain(&mut mesh, 2, 2.0);
        let (chain2, _, _) = open_chain(&mut mesh, 2, 4.0);
        let net: Vec<EdgeKey> = chain0
            .iter()
            .chain(chain1.iter())
            .chain(chain2.iter())
            .copied()
            .collect();

        let _out = mesh.edgenet_prepare(&net);

        assert_eq!(
            mesh.edge_count(),
            8,
            "6 input edges + exactly 2 bridges (third chain left open)",
        );
        mesh.validate_topology();
    }

    #[test]
    fn out_reuses_preexisting_bridge_edge() {
        // §5.3 / §6.1 — a bridge that coincides with a pre-existing edge adds no
        // edge, but the reused edge is still surfaced in the slot alongside the
        // input chain edges.
        let mut mesh = LoopMesh::new();
        let v0 = mesh.create_vertex(Vec3::new(0.0, 0.0, 0.0));
        let v1 = mesh.create_vertex(Vec3::new(1.0, 0.0, 0.0));
        let v2 = mesh.create_vertex(Vec3::new(0.0, 1.0, 0.0));
        // Two-edge chain v0-v1-v2 as the working net.
        let e01 = mesh.create_edge([v0, v1], EdgeCreateFlags::NO_DUPLICATE);
        let e12 = mesh.create_edge([v1, v2], EdgeCreateFlags::NO_DUPLICATE);
        // The would-be bridge (v0,v2) already exists but is NOT part of the net.
        let e02 = mesh.create_edge([v0, v2], EdgeCreateFlags::NO_DUPLICATE);

        let out = mesh.edgenet_prepare(&[e01, e12]);

        assert_eq!(mesh.edge_count(), 3, "no duplicate edge created for the reused bridge");
        assert_eq!(out.edges().len(), 3, "slot = 2 input chain edges + reused bridge");
        assert!(out.edges().contains(&e02), "the reused pre-existing bridge is reported");
        mesh.validate_topology();
    }
}
