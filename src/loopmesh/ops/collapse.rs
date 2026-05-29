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
    /// island min/max-midpoint snap, run *before* the topology merge). When set,
    /// every has-math per-loop layer (§6.1 — Float2 UV, four-channel float
    /// colour; scalar and three-vector layers are skipped) has its loop-data
    /// islands (§6.2) snapped to the per-scalar-component bounding-box midpoint
    /// (§6.3) before the weld reaps the collapsed corners. With `uvs` clear no
    /// blend runs and the surviving loops carry whatever the weld copies through
    /// unchanged; the topology outcome is identical either way.
    pub fn collapse(&mut self, edges: &[EdgeKey], uvs: bool) {
        // §7 — empty input marks nothing and floods nothing; a complete no-op.
        if edges.is_empty() {
            return;
        }

        // Phase 2 (§3) — mark the input edges so the flood recognises them.
        let marked: HashSet<EdgeKey> = edges.iter().copied().collect();

        // §6 — the optional per-loop customdata blend runs BEFORE the topology
        // merge, while the loops it averages still exist as distinct corners.
        if uvs {
            self.collapse_uvs(&marked);
        }

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

    /// §6 — the `collapse_uvs` per-loop customdata blend. Runs *before* the
    /// topology merge so the loops it averages still exist as distinct corners.
    ///
    /// For every has-math per-loop layer (§6.1) it floods each loop-data island
    /// (§6.2) and snaps every island loop to the per-scalar-component min/max
    /// midpoint of the island's value range (§6.3 — the centre of the value's
    /// axis-aligned bounding box, *not* a mean). Each layer is processed
    /// independently with its own island flood and its own snap.
    fn collapse_uvs(&mut self, marked: &HashSet<EdgeKey>) {
        let participating: SmallVec<[ElementAttributeKey; 4]> = self
            .loop_attribs()
            .iter_descriptors()
            .filter_map(|(key, _)| {
                if loop_layer_participates_in_collapse_blend(self.loop_attribs(), key) {
                    Some(key)
                } else {
                    None
                }
            })
            .collect();

        // §6.1 early-out: no has-math layer means the whole pass is a no-op and
        // the result is identical to `uvs = false`.
        if participating.is_empty() {
            return;
        }

        let all_loops = self.loop_keys();
        for layer in participating {
            let mut visited_loops: HashSet<LoopKey> = HashSet::new();
            for &seed in &all_loops {
                if visited_loops.contains(&seed) {
                    continue;
                }
                // §6.2 seeding: a loop seeds an island only if its own edge is a
                // marked input edge.
                if !marked.contains(&self.loop_edge(seed)) {
                    continue;
                }
                let island =
                    self.collapse_uvs_flood_island(seed, marked, layer, &mut visited_loops);
                self.collapse_uvs_snap_island(layer, &island);
            }
        }
    }

    /// §6.2 — flood one loop-data island in `layer` from `seed`. The walk yields
    /// loops; it conducts across marked input edges and applies the
    /// value-equality gate at each around-the-vertex step, so a single
    /// collapsing shell can host several islands (one per distinct pre-existing
    /// corner value), which preserves UV seams. Every loop is visited at most
    /// once across the whole layer pass via `visited_loops`.
    fn collapse_uvs_flood_island(
        &self,
        seed: LoopKey,
        marked: &HashSet<EdgeKey>,
        layer: ElementAttributeKey,
        visited_loops: &mut HashSet<LoopKey>,
    ) -> SmallVec<[LoopKey; 16]> {
        let mut island: SmallVec<[LoopKey; 16]> = SmallVec::new();
        let mut frontier: SmallVec<[LoopKey; 16]> = SmallVec::new();

        visited_loops.insert(seed);
        island.push(seed);
        frontier.push(seed);

        while let Some(loop_key) = frontier.pop() {
            // §6.2 leaf rule: a loop whose own edge is not marked contributes its
            // value but expands no further.
            if !marked.contains(&self.loop_edge(loop_key)) {
                continue;
            }

            // The two loops at the ends of this loop's edge: the near loop sits
            // at `loop_key.vert`, the far loop at `loop_key.next.vert`.
            for pivot in [loop_key, self.loop_next(loop_key)] {
                let pivot_vert = self.loop_vertex(pivot);
                let pivot_value = self.loop_value(layer, pivot);

                for candidate in self.loops_around_vertex(pivot_vert) {
                    if visited_loops.contains(&candidate) {
                        continue;
                    }

                    let candidate_at_pivot = self.loop_vertex(candidate) == pivot_vert;

                    // Gate 1, edge-mask: a candidate sitting at the *far* vertex
                    // of its own edge relative to the pivot must itself be on a
                    // marked input edge; a candidate sitting *at* the pivot
                    // vertex waives this gate.
                    if !candidate_at_pivot && !marked.contains(&self.loop_edge(candidate)) {
                        continue;
                    }

                    // Gate 2, value-equality: compare the corner that sits *at*
                    // the pivot vertex. For a candidate whose corner is at the
                    // pivot that is the candidate itself; otherwise it is the
                    // candidate's next corner.
                    let candidate_corner = if candidate_at_pivot {
                        candidate
                    } else {
                        self.loop_next(candidate)
                    };
                    let candidate_value = self.loop_value(layer, candidate_corner);
                    if !loop_values_equal(candidate_value, pivot_value) {
                        continue;
                    }

                    visited_loops.insert(candidate);
                    island.push(candidate);
                    frontier.push(candidate);
                }
            }
        }

        island
    }

    /// §6.3 — snap every loop in `island` to the per-scalar-component midpoint of
    /// the island's value range: `mid[k] = (min[k] + max[k]) / 2`. The min and
    /// max are taken independently per component, so the result is the centre of
    /// the value's axis-aligned bounding box and is generally not any loop's
    /// original value and not the mean.
    fn collapse_uvs_snap_island(
        &mut self,
        layer: ElementAttributeKey,
        island: &[LoopKey],
    ) {
        let first = match island.first() {
            Some(&first) => first,
            None => return,
        };

        let mut lo = self.loop_value(layer, first);
        let mut hi = lo;
        for &loop_key in &island[1..] {
            let value = self.loop_value(layer, loop_key);
            for component in 0..4 {
                lo.components[component] = lo.components[component].min(value.components[component]);
                hi.components[component] = hi.components[component].max(value.components[component]);
            }
        }

        let mut mid = lo;
        for component in 0..4 {
            mid.components[component] =
                (lo.components[component] + hi.components[component]) * 0.5;
        }

        for &loop_key in island {
            self.set_loop_value(layer, loop_key, mid);
        }
    }

    /// Read a loop's value in a has-math layer as a fixed-width 4-component
    /// scalar tuple. Unused components are padded so the per-component min/max
    /// of §6.3 sees a stable sentinel.
    fn loop_value(&self, layer: ElementAttributeKey, loop_key: LoopKey) -> LoopBlendValue {
        let slot = loop_key.as_usize();
        match self.loop_attribs().storage_of(layer) {
            Some(AttributeStorage::Vec2(_)) => {
                let v = self.loop_attribs().get_vec2(layer, slot).unwrap_or([0.0; 2]);
                LoopBlendValue {
                    components: [v[0], v[1], 0.0, 0.0],
                    used: 2,
                }
            }
            Some(AttributeStorage::Vec4(_)) => {
                let v = self.loop_attribs().get_vec4(layer, slot).unwrap_or([0.0; 4]);
                LoopBlendValue {
                    components: v,
                    used: 4,
                }
            }
            _ => LoopBlendValue {
                components: [0.0; 4],
                used: 0,
            },
        }
    }

    /// Write a snapped value back into a has-math layer.
    fn set_loop_value(
        &mut self,
        layer: ElementAttributeKey,
        loop_key: LoopKey,
        value: LoopBlendValue,
    ) {
        let slot = loop_key.as_usize();
        match self.loop_attribs().storage_of(layer) {
            Some(AttributeStorage::Vec2(_)) => {
                self.loop_attribs_mut().set_vec2(
                    layer,
                    slot,
                    [value.components[0], value.components[1]],
                );
            }
            Some(AttributeStorage::Vec4(_)) => {
                self.loop_attribs_mut()
                    .set_vec4(layer, slot, value.components);
            }
            _ => {}
        }
    }

    /// Every loop incident to `vertex` — the full disk fan of corners, in both
    /// directions across every face meeting there. A loop is incident either
    /// because its own corner sits at `vertex` (its edge leaves `vertex`) or
    /// because its next corner sits at `vertex` (its edge arrives at `vertex`).
    /// Deduplicated so each loop appears once.
    fn loops_around_vertex(&self, vertex: VertexKey) -> SmallVec<[LoopKey; 16]> {
        let mut loops: SmallVec<[LoopKey; 16]> = SmallVec::new();
        for edge in self.vertex_disk_iter(vertex) {
            for radial in self.edge_radial_iter(edge) {
                for candidate in [radial, self.loop_next(radial)] {
                    let touches = self.loop_vertex(candidate) == vertex
                        || self.loop_vertex(self.loop_next(candidate)) == vertex;
                    if touches && !loops.contains(&candidate) {
                        loops.push(candidate);
                    }
                }
            }
        }
        loops
    }
}

/// A has-math per-loop value, held as a fixed-width 4-component scalar tuple so
/// the §6.3 per-component min/max midpoint can run uniformly across the Float2
/// UV and four-channel colour layer types. `used` records the live component
/// count; the padding components are identical across all loops of a layer so
/// they fold to a stable midpoint and are never written back beyond `used`.
#[derive(Clone, Copy)]
struct LoopBlendValue {
    components: [f32; 4],
    used: usize,
}

/// §6.1 has-math predicate, keyed on the layer's storage type alone (no value
/// inspection). True exactly for the per-loop types that carry the full
/// equals / add / multiply / min-max value-operation set: the Float2 UV layer
/// and the four-channel float colour layer. A scalar Float and a three-vector
/// Float3 layer interpolate under face-split but carry no equals / min-max, so
/// they fail the predicate and are left untouched.
fn loop_layer_participates_in_collapse_blend(
    attribs: &ElementAttributes,
    layer: ElementAttributeKey,
) -> bool {
    matches!(
        attribs.storage_of(layer),
        Some(AttributeStorage::Vec2(_)) | Some(AttributeStorage::Vec4(_))
    )
}

/// The §6.1 tolerant value-equality test used by the island flood (§6.2). Two
/// values are equal when every live scalar component agrees within a small
/// tolerance.
fn loop_values_equal(a: LoopBlendValue, b: LoopBlendValue) -> bool {
    const TOLERANCE: f32 = 1e-6;
    let count = a.used.max(b.used);
    (0..count).all(|component| (a.components[component] - b.components[component]).abs() <= TOLERANCE)
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

    // ----- §6 collapse_uvs per-loop customdata blend -----

    fn register_uv(mesh: &mut LoopMesh) -> ElementAttributeKey {
        mesh.register_loop_attribute(ElementAttributeDesc::numeric(
            AttributeKey::custom("uv"),
            Format::new(ComponentType::F32, 2),
        ))
        .expect("uv layer registers")
    }

    /// Set the UV on every loop whose corner sits at `vert`.
    fn set_uv_at_vert(mesh: &mut LoopMesh, uv: ElementAttributeKey, vert: VertexKey, value: [f32; 2]) {
        let slots: Vec<usize> = mesh
            .loop_keys()
            .into_iter()
            .filter(|&l| mesh.loop_vertex(l) == vert)
            .map(|l| l.as_usize())
            .collect();
        for slot in slots {
            mesh.loop_attribs_mut().set_vec2(uv, slot, value);
        }
    }

    /// §6.4 worked example — a three-loop Float2 island with the documented
    /// three corner UVs snaps to the per-component bounding-box midpoint
    /// `(0.20, 0.50)`, which is *not* any input value and *not* the mean
    /// `(0.1667, 0.40)`. Exercises the §6.3 snap primitive on a hand-built
    /// island so the midpoint contract is pinned independent of flood order.
    #[test]
    fn snap_island_is_component_wise_midpoint_not_mean() {
        let mut mesh = LoopMesh::new();
        let uv = register_uv(&mut mesh);
        let v0 = mesh.create_vertex(Vec3::new(0.0, 0.0, 0.0));
        let v1 = mesh.create_vertex(Vec3::new(1.0, 0.0, 0.0));
        let v2 = mesh.create_vertex(Vec3::new(0.0, 1.0, 0.0));
        let es: Vec<EdgeKey> = [[v0, v1], [v1, v2], [v2, v0]]
            .into_iter()
            .map(|vs| mesh.create_edge(vs, EdgeCreateFlags::NO_DUPLICATE))
            .collect();
        let face = mesh.face_create(&[v0, v1, v2], &es, FaceCreateFlags::NONE);

        let loops = mesh.face_loop_keys(face);
        assert_eq!(loops.len(), 3, "triangle has three loops");
        let values = [[0.0, 0.2], [0.4, 0.1], [0.1, 0.9]];
        for (loop_key, value) in loops.iter().zip(values) {
            mesh.loop_attribs_mut().set_vec2(uv, loop_key.as_usize(), value);
        }

        mesh.collapse_uvs_snap_island(uv, &loops);

        for &loop_key in &loops {
            let snapped = mesh.loop_attribs().get_vec2(uv, loop_key.as_usize()).unwrap();
            assert!(
                (snapped[0] - 0.20).abs() < 1e-6 && (snapped[1] - 0.50).abs() < 1e-6,
                "island snaps to bounding-box midpoint (0.20, 0.50), got {snapped:?}"
            );
        }
    }

    /// §6.1 participation predicate — a Float2 (Vec2) loop layer participates
    /// and is blended; a Float3 (Vec3) loop layer does not (it interpolates but
    /// carries no equals / min-max) and is left byte-for-byte untouched.
    #[test]
    fn participation_predicate_blends_uv_skips_vec3() {
        let mut mesh = LoopMesh::new();
        let uv = register_uv(&mut mesh);
        let attr3 = mesh
            .register_loop_attribute(ElementAttributeDesc::numeric(
                AttributeKey::custom("vec3"),
                Format::new(ComponentType::F32, 3),
            ))
            .unwrap();
        let ([_, b, c, ..], shared) = two_quads_shared_edge(&mut mesh);

        // Distinct UV per shared-edge endpoint so the island spans the two
        // extremes whose midpoint the blend must produce; all other corners
        // default to (0,0) and are outside the island (their edges are
        // unmarked). The Vec3 layer gets a distinct value per endpoint too.
        set_uv_at_vert(&mut mesh, uv, b, [0.0, 0.0]);
        set_uv_at_vert(&mut mesh, uv, c, [1.0, 1.0]);
        for l in mesh.loop_keys() {
            let v = mesh.loop_vertex(l);
            let tag = if v == b { 10.0 } else { 20.0 };
            mesh.loop_attribs_mut().set_vec3(attr3, l.as_usize(), [tag, tag, tag]);
        }
        let vec3_before: Vec<[f32; 3]> = mesh
            .loop_keys()
            .into_iter()
            .map(|l| mesh.loop_attribs().get_vec3(attr3, l.as_usize()).unwrap())
            .collect();

        mesh.collapse(&[shared], true);
        mesh.validate_topology();

        // Find the survivor of the b/c merge, so we know which corners were the
        // blended island.
        let survivor = mesh
            .vert_keys()
            .into_iter()
            .find(|&v| v == b || v == c)
            .expect("one of b/c survives");

        // Vec2 UV: the surviving loops at the merged corner (the b/c island)
        // carry the island midpoint (0.5, 0.5), not a copied-through endpoint.
        let survivor_corners: Vec<[f32; 2]> = mesh
            .loop_keys()
            .into_iter()
            .filter(|&l| mesh.loop_vertex(l) == survivor)
            .map(|l| mesh.loop_attribs().get_vec2(uv, l.as_usize()).unwrap())
            .collect();
        assert!(!survivor_corners.is_empty(), "the survivor has corners");
        for uvv in survivor_corners {
            assert!(
                (uvv[0] - 0.5).abs() < 1e-6 && (uvv[1] - 0.5).abs() < 1e-6,
                "blended UV is the (0,0)/(1,1) midpoint (0.5, 0.5), got {uvv:?}"
            );
        }

        // Vec3: the predicate skips it, so every surviving loop keeps a value
        // that appears in the pre-collapse set (no midpoint was written).
        for l in mesh.loop_keys() {
            let after = mesh.loop_attribs().get_vec3(attr3, l.as_usize()).unwrap();
            assert!(
                vec3_before.contains(&after),
                "Vec3 layer left untouched by the blend, got {after:?}"
            );
        }
    }

    /// §6 contrast — with `uvs = false` no blend runs and the surviving loops
    /// carry whatever the weld copies through (an original endpoint value),
    /// never the midpoint. Same topology as the `uvs = true` path.
    #[test]
    fn uvs_false_does_not_blend_loop_uv() {
        let mut mesh = LoopMesh::new();
        let uv = register_uv(&mut mesh);
        let ([_, b, c, ..], shared) = two_quads_shared_edge(&mut mesh);
        set_uv_at_vert(&mut mesh, uv, b, [0.0, 0.0]);
        set_uv_at_vert(&mut mesh, uv, c, [1.0, 1.0]);

        mesh.collapse(&[shared], false);
        mesh.validate_topology();

        let endpoints = [[0.0_f32, 0.0], [1.0, 1.0]];
        for face in mesh.face_keys() {
            for l in mesh.face_loop_keys(face) {
                let uvv = mesh.loop_attribs().get_vec2(uv, l.as_usize()).unwrap();
                let is_endpoint = endpoints
                    .iter()
                    .any(|e| (uvv[0] - e[0]).abs() < 1e-6 && (uvv[1] - e[1]).abs() < 1e-6);
                let is_midpoint = (uvv[0] - 0.5).abs() < 1e-6 && (uvv[1] - 0.5).abs() < 1e-6;
                assert!(is_endpoint, "uvs=false copies endpoints through, got {uvv:?}");
                assert!(!is_midpoint, "uvs=false must not produce a midpoint");
            }
        }
    }

    /// §6.2 seam preservation — when the two sides of the collapsed edge carry
    /// *distinct* UV values, the value-equality gate keeps them as separate
    /// islands, so each side snaps within itself and the seam is preserved
    /// rather than averaged across.
    #[test]
    fn distinct_corner_values_form_separate_islands() {
        let mut mesh = LoopMesh::new();
        let uv = register_uv(&mut mesh);
        let ([_, b, c, ..], shared) = two_quads_shared_edge(&mut mesh);

        // Per-loop UV keyed on (vertex, face) so the two faces disagree at the
        // shared corners — a UV seam across the collapsed edge.
        let face_keys = mesh.face_keys();
        for (face_index, &face) in face_keys.iter().enumerate() {
            for l in mesh.face_loop_keys(face) {
                let base = if mesh.loop_vertex(l) == b || mesh.loop_vertex(l) == c {
                    0.0
                } else {
                    5.0
                };
                let seam_offset = face_index as f32 * 100.0;
                mesh.loop_attribs_mut()
                    .set_vec2(uv, l.as_usize(), [base + seam_offset, base + seam_offset]);
            }
        }

        mesh.collapse(&[shared], true);
        mesh.validate_topology();

        // No surviving loop blends across the seam: every value stays on one
        // side's scale (near 0/100 for the shared corners of each face), never
        // a cross-seam average like ~50.
        for face in mesh.face_keys() {
            for l in mesh.face_loop_keys(face) {
                let uvv = mesh.loop_attribs().get_vec2(uv, l.as_usize()).unwrap();
                let cross_seam_average = (uvv[0] - 50.0).abs() < 5.0;
                assert!(
                    !cross_seam_average,
                    "seam preserved: no loop averages across the two islands, got {uvv:?}"
                );
            }
        }
    }

    /// §6.1 early-out / §7 — `uvs = true` on a mesh with no has-math loop layer
    /// finds nothing to blend and is identical to `uvs = false`: topology
    /// collapses the same and no panic occurs.
    #[test]
    fn uvs_true_no_math_layer_is_noop_blend() {
        let mut mesh = LoopMesh::new();
        let (_verts, shared) = two_quads_shared_edge(&mut mesh);

        mesh.collapse(&[shared], true);

        assert_eq!(mesh.totvert(), 5, "topology collapses regardless of the flag");
        assert_eq!(mesh.totface(), 2);
        mesh.validate_topology();
    }
}
