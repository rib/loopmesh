use std::collections::HashSet;

use glam::Vec3;
use smallvec::SmallVec;

use crate::loopmesh::*;

impl LoopMesh {
    /// Connected-selection vertex-merge: take an input edge set, flood each
    /// edge's connected vertex shell into a component, and fuse every vertex of
    /// a component down to one survivor parked at the component's
    /// edge-endpoint-weighted centroid.
    ///
    /// Implements `radial-mesh-guides/operators/COLLAPSE.md`:
    ///
    /// - **§3 / §5 connected-component flood** — the input is an *edge* set, not
    ///   a vertex set. Each component is the vertex shell reachable from a seed
    ///   edge by stepping only across *other input edges*; a non-input edge is
    ///   invisible to the flood. A branch (a vert with three incident input
    ///   edges) is one component, not one-per-branch.
    /// - **§4.1 centroid** — the survivor lands at the edge-endpoint-weighted
    ///   centroid: the sum of every input edge's *two* endpoint positions
    ///   divided by the endpoint count, so a vert shared by two input edges is
    ///   weighted twice. Every component vert is snapped to this centroid
    ///   *before* the weld so the non-survivors are coincident at delete time.
    /// - **§4.2 survivor identity** — the survivor is the first endpoint of the
    ///   *last* edge the flood yields (the most recently visited edge in the
    ///   component). This is a flood-order-dependent contract; the survivor's
    ///   position is order-independent (it is always the centroid) but its
    ///   identity tracks the flood order.
    /// - **§3 phase 4 / §10 delegation** — the assembled `(non-survivor →
    ///   survivor)` targetmap is handed to [`Self::weld_verts`] with
    ///   `use_centroid = false`: collapse has *already* parked every vert at the
    ///   component centroid, so re-centroiding would double-average. All
    ///   edge/face rewrite, collapsed-edge removal, degenerate-face reap, and
    ///   deferred delete are the weld contract, inherited unchanged — exactly as
    ///   [`Self::pointmerge`] / [`Self::remove_doubles`] compose it.
    /// - **§7 edge cases** — an empty `edges` input marks nothing, floods
    ///   nothing, and is a complete no-op. A wire edge (radial cycle empty)
    ///   collapses its endpoints like any other edge. A non-manifold edge
    ///   (radial 3+) is walked like any other; the weld re-points every incident
    ///   face's corner independently with no non-manifold rejection. Two
    ///   disjoint input edges produce two independent survivors.
    ///
    /// `uvs` requests the §6 per-loop customdata blend (the `collapse_uvs`
    /// island min/max-midpoint snap, run *before* the topology merge). The
    /// `uvs = false` path is fully implemented; the `uvs = true` blend is not
    /// yet implemented (see the note in the body) and currently behaves
    /// identically to `uvs = false` for topology and survivor position — the
    /// loops that survive carry whatever the weld copies through unchanged.
    pub fn collapse(&mut self, edges: &[EdgeKey], uvs: bool) {
        // §7 — empty input marks nothing and floods nothing; a complete no-op.
        if edges.is_empty() {
            return;
        }

        // §6 — the optional per-loop customdata blend runs BEFORE the topology
        // merge. Not yet implemented; see the report. The `uvs = true` path
        // currently produces the same topology and survivor position as
        // `uvs = false`, only the surviving loops' customdata differs.
        let _ = uvs;

        // Phase 2 (§3) — mark the input edges so the flood recognises them.
        let marked: HashSet<EdgeKey> = edges.iter().copied().collect();

        let mut targetmap: Vec<(VertexKey, VertexKey)> = Vec::new();
        let mut consumed_edges: HashSet<EdgeKey> = HashSet::new();
        let mut mapped_verts: HashSet<VertexKey> = HashSet::new();

        // Phase 3 (§4, §5) — per-component flood, centroid, survivor. Seed from
        // the input edges in input order so the flood order (and therefore the
        // §4.2 survivor identity) is the documented order-dependent contract.
        for &seed in edges {
            if consumed_edges.contains(&seed) {
                continue;
            }

            let component_edges =
                self.collapse_flood_component(seed, &marked, &mut consumed_edges);

            let mut centroid = Vec3::ZERO;
            let mut endpoint_count = 0;
            for &edge in &component_edges {
                let [v0, v1] = self.edge_vertices(edge);
                centroid += self.vertex_position(v0);
                centroid += self.vertex_position(v1);
                endpoint_count += 2;
            }
            centroid /= endpoint_count as f32;

            // §4.2 — survivor is the first endpoint of the LAST edge the flood
            // yielded (the most recently visited edge in the component).
            let last_edge = *component_edges.last().expect("flood yields ≥ 1 edge");
            let survivor = self.edge_vertices(last_edge)[0];

            // Snap every component vert to the centroid (§4.1) and map each
            // non-survivor onto the survivor exactly once, even when the flood
            // reaches it along two distinct paths within the component (§4.2).
            for &edge in &component_edges {
                for vert in self.edge_vertices(edge) {
                    self.set_vertex_position(vert, centroid);
                    if vert != survivor && mapped_verts.insert(vert) {
                        targetmap.push((vert, survivor));
                    }
                }
            }
        }

        // Phase 4 (§3 / §10) — delegate the rewrite to weld-verts with
        // use_centroid = false; the verts are already at the centroid.
        self.weld_verts(&targetmap, false);
    }

    /// §5 — flood a connected component starting from `seed`. The walk is a
    /// vertex-shell walk masked to the marked (input) edge set: from each
    /// component vert, step across its other *marked* incident edges. Returns
    /// the component's edges in flood order (the last entry is the most
    /// recently visited edge, which §4.2 reads the survivor from), and records
    /// every visited edge in `consumed_edges` so the outer seed loop skips
    /// edges already claimed by an earlier component.
    fn collapse_flood_component(
        &self,
        seed: EdgeKey,
        marked: &HashSet<EdgeKey>,
        consumed_edges: &mut HashSet<EdgeKey>,
    ) -> SmallVec<[EdgeKey; 16]> {
        let mut component_edges: SmallVec<[EdgeKey; 16]> = SmallVec::new();
        let mut vert_stack: SmallVec<[VertexKey; 16]> = SmallVec::new();
        let mut visited_verts: HashSet<VertexKey> = HashSet::new();

        consumed_edges.insert(seed);
        component_edges.push(seed);
        for vert in self.edge_vertices(seed) {
            if visited_verts.insert(vert) {
                vert_stack.push(vert);
            }
        }

        while let Some(vert) = vert_stack.pop() {
            for edge in self.vert_edge_keys(vert) {
                if !marked.contains(&edge) || !consumed_edges.insert(edge) {
                    continue;
                }
                component_edges.push(edge);
                if let Some(other) = self.edges.get(edge).other_vertex(vert) {
                    if visited_verts.insert(other) {
                        vert_stack.push(other);
                    }
                }
            }
        }

        component_edges
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use glam::Vec3;

    use crate::loopmesh::*;

    /// Two quads sharing the vertical edge `{b, c}`. Returns the six verts and
    /// the shared edge handle. Mirrors the `two_quads_shared_edge` fixture
    /// shape the matrix rows lean on.
    fn two_quads_shared_edge(
        mesh: &mut LoopMesh,
    ) -> ([VertexKey; 6], EdgeKey) {
        let a0 = mesh.create_vertex(Vec3::new(0.0, 0.0, 0.0));
        let b = mesh.create_vertex(Vec3::new(1.0, 0.0, 0.0));
        let c = mesh.create_vertex(Vec3::new(1.0, 1.0, 0.0));
        let a1 = mesh.create_vertex(Vec3::new(0.0, 1.0, 0.0));
        let d0 = mesh.create_vertex(Vec3::new(2.0, 0.0, 0.0));
        let d1 = mesh.create_vertex(Vec3::new(2.0, 1.0, 0.0));

        let mk = |mesh: &mut LoopMesh, vs: [VertexKey; 4]| {
            let es: Vec<EdgeKey> = (0..4)
                .map(|i| mesh.create_edge([vs[i], vs[(i + 1) % 4]], EdgeCreateFlags::NO_DUPLICATE))
                .collect();
            mesh.face_create(&vs, &es, FaceCreateFlags::NONE);
        };
        mk(mesh, [a0, b, c, a1]);
        mk(mesh, [b, d0, d1, c]);

        let shared = mesh.query_edge_exists(b, c).expect("shared edge present");
        ([a0, b, c, a1, d0, d1], shared)
    }

    /// A collinear chain of `n` edges over `n + 1` verts on the x-axis, spaced
    /// one unit apart. Returns the verts and the chain edges in order.
    fn collinear_chain(mesh: &mut LoopMesh, n: usize) -> (Vec<VertexKey>, Vec<EdgeKey>) {
        let verts: Vec<VertexKey> = (0..=n)
            .map(|i| mesh.create_vertex(Vec3::new(i as f32, 0.0, 0.0)))
            .collect();
        let edges: Vec<EdgeKey> = (0..n)
            .map(|i| mesh.create_edge([verts[i], verts[i + 1]], EdgeCreateFlags::NO_DUPLICATE))
            .collect();
        (verts, edges)
    }

    /// §2 / §7 — a single edge between two quads: its two endpoints are the
    /// whole component and fuse to the edge midpoint; both quads lose a corner
    /// and become triangles.
    #[test]
    fn single_edge_two_quad_strip_collapses_to_midpoint() {
        let mut mesh = LoopMesh::new();
        let ([_, b, c, ..], shared) = two_quads_shared_edge(&mut mesh);
        let midpoint = (mesh.vertex_position(b) + mesh.vertex_position(c)) * 0.5;

        mesh.collapse(&[shared], false);

        assert_eq!(mesh.totvert(), 5, "the shared pair fused to one survivor");
        assert_eq!(mesh.totface(), 2, "both quads survive as triangles");
        for face in mesh.face_keys() {
            assert_eq!(mesh.face_len(face), 3, "each quad lost the collapsed corner");
        }
        let live: HashSet<VertexKey> = mesh.vert_keys().into_iter().collect();
        let survivor = if live.contains(&b) { b } else { c };
        assert_eq!(
            mesh.vertex_position(survivor),
            midpoint,
            "the survivor lands at the edge midpoint"
        );
        mesh.validate_topology();
    }

    /// §2 / §5 — a connected chain of input edges floods to ONE component and
    /// collapses to ONE survivor, not several pairwise midpoints.
    #[test]
    fn connected_chain_floods_to_single_survivor() {
        let mut mesh = LoopMesh::new();
        let (_verts, edges) = collinear_chain(&mut mesh, 3);

        mesh.collapse(&edges, false);

        assert_eq!(mesh.totvert(), 1, "the whole chain fused to one survivor");
        assert_eq!(mesh.totedge(), 0, "every collapsed chain edge is reaped");
        mesh.validate_topology();
    }

    /// §5 — a star of input edges at one vertex floods all branches into ONE
    /// component → one survivor (a branch is not several components).
    #[test]
    fn branching_cluster_floods_to_single_survivor() {
        let mut mesh = LoopMesh::new();
        // Four wire spokes from a central hub; all four are input edges.
        let hub = mesh.create_vertex(Vec3::new(0.0, 0.0, 0.0));
        let tips: Vec<VertexKey> = [
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(-1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, -1.0, 0.0),
        ]
        .into_iter()
        .map(|p| mesh.create_vertex(p))
        .collect();
        let spokes: Vec<EdgeKey> = tips
            .iter()
            .map(|&tip| mesh.create_edge([hub, tip], EdgeCreateFlags::NO_DUPLICATE))
            .collect();

        mesh.collapse(&spokes, false);

        assert_eq!(mesh.totvert(), 1, "the whole star fused to one survivor");
        assert_eq!(mesh.totedge(), 0, "all four spokes collapsed away");
        mesh.validate_topology();
    }

    /// §4.1 — the centroid is edge-endpoint-weighted: a vert shared by two
    /// input edges is weighted twice. An asymmetric three-edge chain over
    /// `{0, 1, 2, 3}` weights the two interior verts double.
    #[test]
    fn edge_endpoint_weighted_centroid_survivor_position() {
        let mut mesh = LoopMesh::new();
        // Non-uniform spacing separates the endpoint-weighted centroid from the
        // plain vert mean. Interior verts (1, 2) are each shared by two edges
        // and so counted twice.
        let xs = [0.0_f32, 1.0, 2.0, 10.0];
        let verts: Vec<VertexKey> = xs
            .iter()
            .map(|&x| mesh.create_vertex(Vec3::new(x, 0.0, 0.0)))
            .collect();
        let edges: Vec<EdgeKey> = (0..3)
            .map(|i| mesh.create_edge([verts[i], verts[i + 1]], EdgeCreateFlags::NO_DUPLICATE))
            .collect();
        // Endpoint-weighted: ((0+1)+(1+2)+(2+10)) / 6 = 16/6 ≈ 2.6667.
        // Plain vert mean: (0+1+2+10)/4 = 3.25. Distinct → weighting observable.
        let expected_x = (0.0 + 1.0 + 1.0 + 2.0 + 2.0 + 10.0) / 6.0;

        mesh.collapse(&edges, false);

        assert_eq!(mesh.totvert(), 1, "the chain fused to one survivor");
        let survivor = mesh.vert_keys()[0];
        let pos = mesh.vertex_position(survivor);
        assert!(
            (pos.x - expected_x).abs() < 1e-5,
            "survivor at edge-endpoint-weighted centroid {expected_x}, got {}",
            pos.x
        );
        mesh.validate_topology();
    }

    /// §7 — a face-less wire edge collapses its two endpoints to the midpoint,
    /// leaving one vert and no edges.
    #[test]
    fn wire_edge_collapses_to_single_vert() {
        let mut mesh = LoopMesh::new();
        let a = mesh.create_vertex(Vec3::new(-1.0, 0.0, 0.0));
        let b = mesh.create_vertex(Vec3::new(1.0, 0.0, 0.0));
        let wire = mesh.create_edge([a, b], EdgeCreateFlags::NO_DUPLICATE);

        mesh.collapse(&[wire], false);

        assert_eq!(mesh.totvert(), 1, "the wire endpoints fused to one vert");
        assert_eq!(mesh.totedge(), 0, "the wire edge collapsed away");
        assert_eq!(mesh.totface(), 0, "no face involved");
        let survivor = mesh.vert_keys()[0];
        assert_eq!(
            mesh.vertex_position(survivor),
            Vec3::new(0.0, 0.0, 0.0),
            "the survivor lands at the wire midpoint"
        );
        mesh.validate_topology();
    }

    /// §7 — empty `edges` input is a complete no-op; the mesh is unchanged.
    #[test]
    fn empty_input_is_noop() {
        let mut mesh = LoopMesh::new();
        let (_verts, shared) = two_quads_shared_edge(&mut mesh);
        let _ = shared;
        let before = (mesh.totvert(), mesh.totedge(), mesh.totface());

        mesh.collapse(&[], false);

        assert_eq!((mesh.totvert(), mesh.totedge(), mesh.totface()), before);
        mesh.validate_topology();
    }

    /// §7 — two disjoint input edges in separate islands produce two
    /// independent survivors at two centroids; the islands never interact.
    #[test]
    fn disjoint_components_two_survivors() {
        let mut mesh = LoopMesh::new();
        let a0 = mesh.create_vertex(Vec3::new(-10.0, -1.0, 0.0));
        let a1 = mesh.create_vertex(Vec3::new(-10.0, 1.0, 0.0));
        let e0 = mesh.create_edge([a0, a1], EdgeCreateFlags::NO_DUPLICATE);
        let b0 = mesh.create_vertex(Vec3::new(10.0, -1.0, 0.0));
        let b1 = mesh.create_vertex(Vec3::new(10.0, 1.0, 0.0));
        let e1 = mesh.create_edge([b0, b1], EdgeCreateFlags::NO_DUPLICATE);

        mesh.collapse(&[e0, e1], false);

        assert_eq!(mesh.totvert(), 2, "two independent survivors");
        assert_eq!(mesh.totedge(), 0, "both wire edges collapsed away");
        let positions: HashSet<[i32; 2]> = mesh
            .vert_keys()
            .into_iter()
            .map(|v| {
                let p = mesh.vertex_position(v);
                [(p.x * 1000.0) as i32, (p.y * 1000.0) as i32]
            })
            .collect();
        assert!(
            positions.contains(&[-10000, 0]) && positions.contains(&[10000, 0]),
            "the two survivors sit at their own island midpoints"
        );
        mesh.validate_topology();
    }

    /// §7 — a non-manifold spine edge (radial 3+) collapses with every incident
    /// face's corner re-pointed independently by the weld; degenerate faces
    /// reap, the rest keep the survivor. No non-manifold rejection.
    #[test]
    fn nonmanifold_radial3_spine_collapses() {
        let mut mesh = LoopMesh::new();
        // Three quads sharing the radial-3 spine edge {spine0, spine1}.
        let spine0 = mesh.create_vertex(Vec3::new(0.0, 0.0, 0.0));
        let spine1 = mesh.create_vertex(Vec3::new(0.0, 1.0, 0.0));
        let spine = mesh.create_edge([spine0, spine1], EdgeCreateFlags::NO_DUPLICATE);

        let wings = [
            [Vec3::new(1.0, 0.0, 0.0), Vec3::new(1.0, 1.0, 0.0)],
            [Vec3::new(-1.0, 0.0, 0.0), Vec3::new(-1.0, 1.0, 0.0)],
            [Vec3::new(0.0, 0.0, 1.0), Vec3::new(0.0, 1.0, 1.0)],
        ];
        for [outer0, outer1] in wings {
            let o0 = mesh.create_vertex(outer0);
            let o1 = mesh.create_vertex(outer1);
            let vs = [spine0, o0, o1, spine1];
            let es: Vec<EdgeKey> = (0..4)
                .map(|i| mesh.create_edge([vs[i], vs[(i + 1) % 4]], EdgeCreateFlags::NO_DUPLICATE))
                .collect();
            mesh.face_create(&vs, &es, FaceCreateFlags::NONE);
        }
        assert_eq!(mesh.totface(), 3, "three quads share the spine");

        mesh.collapse(&[spine], false);

        // The spine endpoints fuse; each quad loses a corner → three triangles.
        let live: HashSet<VertexKey> = mesh.vert_keys().into_iter().collect();
        assert!(
            !live.contains(&spine0) || !live.contains(&spine1),
            "one spine endpoint fused away"
        );
        assert_eq!(mesh.totface(), 3, "all three wings survive");
        for face in mesh.face_keys() {
            assert_eq!(mesh.face_len(face), 3, "each wing dropped to a triangle");
        }
        mesh.validate_topology();
    }
}
