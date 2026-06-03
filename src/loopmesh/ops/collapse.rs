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
            self.collapse_uvs_blend(&marked);
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

    /// Standalone "snap each connected UV island to its bounding-box midpoint"
    /// operator. Takes an input edge set and, for *every* has-math per-loop
    /// customdata layer, snaps the like-valued loop islands the edges stitch
    /// together down to a single value — the component-wise midpoint of the
    /// island's value range. No topology changes.
    ///
    /// Implements `radial-mesh-guides/operators/COLLAPSE_UVS.md`:
    ///
    /// - **§1.1 / §3 conduction graph** — the input edges are the *conductor*
    ///   set for the island flood, not a corner list and not a face set. A
    ///   corner joins an island only by being reached across a marked input edge
    ///   under the value-equality gate. An edge not in the input is a wall.
    /// - **§4 all has-math layers** — unlike the `faces`-driven single-active-
    ///   layer siblings, this runs the flood/snap for *every* participating
    ///   per-loop layer (every UV layer and every colour layer), each with its
    ///   own independent island flood.
    /// - **§4.1 early-out** — if no per-loop layer is has-math the operator
    ///   returns immediately, creating no layer; the empty-input case (§7) is
    ///   likewise a no-op (no edge marked → no seed loop).
    /// - **§5 topology-invariance / idempotence** — no Euler operator runs, no
    ///   geometry moves; the entire effect lives in the snapped loop layers.
    ///   Running twice equals running once (the second pass's island range is a
    ///   single point).
    ///
    /// This is the pure-customdata sibling of [`Self::collapse`] with
    /// `uvs = true` (the §6 pre-merge blend) — the identical island flood and
    /// midpoint snap, minus the weld. Both delegate to the same
    /// [`Self::collapse_uvs_blend`] primitive (§8).
    pub fn collapse_uvs(&mut self, edges: &[EdgeKey]) {
        // §7 — empty input marks nothing and floods nothing; a no-op. (The
        // blend below would also early-out, but this skips the set build.)
        if edges.is_empty() {
            return;
        }
        let marked: HashSet<EdgeKey> = edges.iter().copied().collect();
        self.collapse_uvs_blend(&marked);
    }

    /// §6 — the per-loop customdata blend shared by [`Self::collapse`]'s
    /// `uvs = true` pre-merge pass and the standalone [`Self::collapse_uvs`]
    /// operator. When called from `collapse` it runs *before* the topology merge
    /// so the loops it averages still exist as distinct corners; when called
    /// standalone it *is* the whole operator.
    ///
    /// For every has-math per-loop layer (§6.1) it floods each loop-data island
    /// (§6.2) and snaps every island loop to the per-scalar-component min/max
    /// midpoint of the island's value range (§6.3 — the centre of the value's
    /// axis-aligned bounding box, *not* a mean). Each layer is processed
    /// independently with its own island flood and its own snap.
    fn collapse_uvs_blend(&mut self, marked: &HashSet<EdgeKey>) {
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

    /// Standalone "flatten the face-corner customdata around a set of vertices
    /// to one value" operator. Takes an input *vertex* set and, for every
    /// has-math per-loop customdata layer, gathers every corner sitting at any
    /// selected vert — the gateless per-vert loop fan — and overwrites them all
    /// with one value per layer: the component-wise midpoint of the gathered
    /// corners' value range. No topology changes.
    ///
    /// Implements `radial-mesh-guides/operators/AVERAGE_VERT_FACEDATA.md`:
    ///
    /// - **§3 gateless per-vert fan** — for each selected vert, every face
    ///   corner anchored at it is gathered. There is *no* edge mask and *no*
    ///   value-equality gate: a corner is gathered solely because it sits at a
    ///   selected vert. So a UV/colour seam meeting at a selected vert is
    ///   *flattened* (both sides pooled together), the opposite of
    ///   [`Self::collapse_uvs`]'s seam-preserving island flood (§4.2).
    /// - **§4 / §4.1 one global pool per layer** — the min/max accumulator is
    ///   built *once per layer* over the union of all selected verts' corners,
    ///   *not* reset between verts. Every gathered corner across every selected
    ///   vert lands on the single shared bounding-box midpoint. Calling once
    ///   with two verts differs from calling twice with one vert each.
    /// - **§2 bbox midpoint, not mean** — the snapped value is the
    ///   per-component `(min + max) / 2` of the pool, the identical
    ///   [`Self::collapse_uvs_snap_island`] snap collapse uses (§6.3 there);
    ///   independent of how many corners carry intermediate values.
    /// - **§5 every has-math loop layer** — every UV layer and every colour
    ///   layer is pooled and snapped independently, each with its own pool. The
    ///   same has-math participation predicate as the collapse blend (§6.1).
    /// - **§5.2 / §7 edge cases** — no has-math loop layer is a whole-operator
    ///   no-op (no layer created). An empty input vert set, or one naming only
    ///   loop-less (wire/isolated) verts, gathers no corners and is a no-op for
    ///   that layer. Idempotent: a second pass finds the pool a single point.
    ///
    /// This is the pure-customdata sibling of [`Self::collapse_uvs`] — it shares
    /// the identical midpoint snap and the all-has-math-layers iteration, but
    /// gathers corners by a vert's *gateless* loop fan and pools them into *one*
    /// global accumulator rather than per-island.
    pub fn average_vert_facedata(&mut self, verts: &[VertexKey]) {
        // §7 — no verts named gathers nothing; a complete no-op.
        if verts.is_empty() {
            return;
        }

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

        // §5.2 early-out: no has-math layer means the whole operator is a no-op
        // and creates no layer.
        if participating.is_empty() {
            return;
        }

        for layer in participating {
            // §4 / §4.1 — one pool over the union of all selected verts'
            // corners, NOT reset between verts. Each corner appears once.
            let mut pool: SmallVec<[LoopKey; 32]> = SmallVec::new();
            for &vert in verts {
                for corner in self.corners_at_vertex(vert) {
                    if !pool.contains(&corner) {
                        pool.push(corner);
                    }
                }
            }

            // §6.1 — an empty pool (every selected vert loop-less) writes
            // nothing for this layer.
            if pool.is_empty() {
                continue;
            }

            // §2 / §6.2 / §6.3 — one component-wise bbox midpoint over the whole
            // pool, written back to every gathered corner.
            self.collapse_uvs_snap_island(layer, &pool);
        }
    }

    /// Every face corner anchored at `vertex` — the vert's gateless loop fan,
    /// one loop per face incident to `vertex` (the loop whose own corner sits at
    /// `vertex`). A wire/isolated vert with no incident face yields none. This
    /// is the gather of `AVERAGE_VERT_FACEDATA.md` §3 (and the loops-of-vert
    /// fan the collapse blend walks around a pivot, §6.2 there) *without* any
    /// edge mask or value-equality gate.
    fn corners_at_vertex(&self, vertex: VertexKey) -> SmallVec<[LoopKey; 16]> {
        let mut corners: SmallVec<[LoopKey; 16]> = SmallVec::new();
        for edge in self.vertex_disk_iter(vertex).collect::<SmallVec<[EdgeKey; 16]>>() {
            for loop_key in self.edge_radial_iter(edge).collect::<SmallVec<[LoopKey; 8]>>() {
                if self.loop_vertex(loop_key) != vertex {
                    continue;
                }
                if !corners.contains(&loop_key) {
                    corners.push(loop_key);
                }
            }
        }
        corners
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

    // ----- standalone collapse_uvs operator (COLLAPSE_UVS.md) -----

    fn register_color(mesh: &mut LoopMesh) -> ElementAttributeKey {
        mesh.register_loop_attribute(ElementAttributeDesc::numeric(
            AttributeKey::custom("color"),
            Format::new(ComponentType::F32, 4),
        ))
        .expect("color layer registers")
    }

    /// Snapshot every loop's Float2 value in a layer, keyed by loop slot.
    fn uv_snapshot(mesh: &LoopMesh, uv: ElementAttributeKey) -> Vec<(usize, [f32; 2])> {
        mesh.loop_keys()
            .into_iter()
            .map(|l| (l.as_usize(), mesh.loop_attribs().get_vec2(uv, l.as_usize()).unwrap()))
            .collect()
    }

    /// Seed each face's four shared-edge corners with a spread of distinct UVs so
    /// the island's bounding box is non-degenerate and its midpoint differs from
    /// every corner. Returns the expected per-component midpoint of the spread.
    fn seed_like_valued_island(
        mesh: &mut LoopMesh,
        uv: ElementAttributeKey,
        island_verts: [VertexKey; 2],
    ) -> [f32; 2] {
        // The two faces must *agree* at each shared vertex (coherent seam) so the
        // value-equality gate floods all four shared corners into ONE island; the
        // spread that the bbox midpoint averages is between the two verts'
        // distinct values (so the midpoint differs from each corner).
        let value_at = |v: VertexKey| -> [f32; 2] {
            if v == island_verts[0] {
                [0.0, 0.2]
            } else {
                [0.4, 0.8]
            }
        };
        for face in mesh.face_keys() {
            for l in mesh.face_loop_keys(face) {
                let v = mesh.loop_vertex(l);
                if v == island_verts[0] || v == island_verts[1] {
                    mesh.loop_attribs_mut().set_vec2(uv, l.as_usize(), value_at(v));
                }
            }
        }
        // bbox midpoint over the two distinct values.
        let a = value_at(island_verts[0]);
        let b = value_at(island_verts[1]);
        [
            (a[0].min(b[0]) + a[0].max(b[0])) * 0.5,
            (a[1].min(b[1]) + a[1].max(b[1])) * 0.5,
        ]
    }

    /// §2 / §3 / §5 headline — marking the shared edge floods both faces'
    /// shared-edge corners into one island and snaps every island corner to the
    /// component-wise bounding-box midpoint (not a mean, not any input value).
    #[test]
    fn collapse_uvs_shared_edge_snaps_island_to_midpoint() {
        let mut mesh = LoopMesh::new();
        let uv = register_uv(&mut mesh);
        let ([_, b, c, ..], shared) = two_quads_shared_edge(&mut mesh);
        let midpoint = seed_like_valued_island(&mut mesh, uv, [b, c]);

        mesh.collapse_uvs(&[shared]);
        mesh.validate_topology();

        // Topology is untouched (§5 / §7.1).
        assert_eq!(mesh.totvert(), 6);
        assert_eq!(mesh.totedge(), 7);
        assert_eq!(mesh.totface(), 2);

        // Every shared-edge corner now carries the island midpoint.
        for l in mesh.loop_keys() {
            let v = mesh.loop_vertex(l);
            if v == b || v == c {
                let uvv = mesh.loop_attribs().get_vec2(uv, l.as_usize()).unwrap();
                assert!(
                    (uvv[0] - midpoint[0]).abs() < 1e-6 && (uvv[1] - midpoint[1]).abs() < 1e-6,
                    "shared-edge corner snapped to bbox midpoint {midpoint:?}, got {uvv:?}"
                );
            }
        }
    }

    /// §1.1 / §7 — an empty input edge set marks no conductor, seeds no flood,
    /// and is a complete no-op: the UV layer is byte-for-byte unchanged.
    #[test]
    fn collapse_uvs_empty_input_is_noop() {
        let mut mesh = LoopMesh::new();
        let uv = register_uv(&mut mesh);
        let ([_, b, c, ..], _shared) = two_quads_shared_edge(&mut mesh);
        let _ = seed_like_valued_island(&mut mesh, uv, [b, c]);
        let before = uv_snapshot(&mesh, uv);

        mesh.collapse_uvs(&[]);
        mesh.validate_topology();

        assert_eq!(uv_snapshot(&mesh, uv), before, "empty input leaves UVs untouched");
    }

    /// §1.1 / §3 — the same like-valued island, but the shared edge is *not* in
    /// the input set (a different, within-quad edge is). The unmarked shared edge
    /// is a wall: nothing floods across it and the shared-edge corners keep their
    /// original spread (no snap to the cross-face midpoint).
    #[test]
    fn collapse_uvs_unmarked_edge_blocks_flood() {
        let mut mesh = LoopMesh::new();
        let uv = register_uv(&mut mesh);
        let ([a0, b, c, a1, ..], _shared) = two_quads_shared_edge(&mut mesh);
        let midpoint = seed_like_valued_island(&mut mesh, uv, [b, c]);
        let before = uv_snapshot(&mesh, uv);

        // Mark a boundary edge of the first quad ({a0, a1}), not the shared one.
        let unrelated = mesh.query_edge_exists(a1, a0).expect("boundary edge present");
        mesh.collapse_uvs(&[unrelated]);
        mesh.validate_topology();

        // The shared-edge corners must not have been snapped to the cross-face
        // midpoint — the wall held. (Their value may change only if the {a0,a1}
        // island happens to touch them, which it does not here.)
        let after = uv_snapshot(&mesh, uv);
        let _ = before;
        for l in mesh.loop_keys() {
            let v = mesh.loop_vertex(l);
            if v == b || v == c {
                let uvv = mesh.loop_attribs().get_vec2(uv, l.as_usize()).unwrap();
                let snapped = (uvv[0] - midpoint[0]).abs() < 1e-6 && (uvv[1] - midpoint[1]).abs() < 1e-6;
                assert!(!snapped, "unmarked shared edge must not flood corner, got {uvv:?}");
            }
        }
        // a0/a1 corners are not part of any cross-face island either (no marked
        // edge connects two faces), so nothing was averaged across faces.
        assert_eq!(after.len(), before.len());
    }

    /// §3 / §6.1 — a genuine UV seam: the two faces carry *different* values at
    /// the shared corners. The value-equality gate keeps the two sides as two
    /// islands snapped separately; the seam survives (no value lands on the
    /// cross-seam average).
    #[test]
    fn collapse_uvs_seam_preserved_two_islands() {
        let mut mesh = LoopMesh::new();
        let uv = register_uv(&mut mesh);
        let ([_, b, c, ..], shared) = two_quads_shared_edge(&mut mesh);

        // Per-loop UV keyed on (vertex, face): the two faces disagree at the
        // shared corners — a seam. Within each face the two shared corners carry
        // a small spread so each island's snap is observable.
        for (face_index, face) in mesh.face_keys().into_iter().enumerate() {
            for l in mesh.face_loop_keys(face) {
                let v = mesh.loop_vertex(l);
                if v == b || v == c {
                    let base = face_index as f32 * 100.0;
                    let bump = if v == b { 0.0 } else { 2.0 };
                    mesh.loop_attribs_mut().set_vec2(uv, l.as_usize(), [base + bump, base + bump]);
                }
            }
        }

        mesh.collapse_uvs(&[shared]);
        mesh.validate_topology();

        // Face 0's island snaps within [0..2] → mid 1; face 1's within [100..102]
        // → mid 101. No corner ever lands on a cross-seam average (~51).
        for face in mesh.face_keys() {
            for l in mesh.face_loop_keys(face) {
                let v = mesh.loop_vertex(l);
                if v == b || v == c {
                    let uvv = mesh.loop_attribs().get_vec2(uv, l.as_usize()).unwrap();
                    let cross_seam = (uvv[0] - 51.0).abs() < 10.0;
                    assert!(!cross_seam, "seam preserved, no cross-seam average, got {uvv:?}");
                }
            }
        }
    }

    /// §4 — every has-math loop layer is collapsed, not just one: with two UV
    /// layers present, both are flooded and snapped independently.
    #[test]
    fn collapse_uvs_all_layers_multi_uv() {
        let mut mesh = LoopMesh::new();
        let uv0 = register_uv(&mut mesh);
        let uv1 = mesh
            .register_loop_attribute(ElementAttributeDesc::numeric(
                AttributeKey::custom("uv1"),
                Format::new(ComponentType::F32, 2),
            ))
            .unwrap();
        let ([_, b, c, ..], shared) = two_quads_shared_edge(&mut mesh);
        let mid0 = seed_like_valued_island(&mut mesh, uv0, [b, c]);
        let mid1 = seed_like_valued_island(&mut mesh, uv1, [b, c]);

        mesh.collapse_uvs(&[shared]);
        mesh.validate_topology();

        for l in mesh.loop_keys() {
            let v = mesh.loop_vertex(l);
            if v == b || v == c {
                let a = mesh.loop_attribs().get_vec2(uv0, l.as_usize()).unwrap();
                let d = mesh.loop_attribs().get_vec2(uv1, l.as_usize()).unwrap();
                assert!(
                    (a[0] - mid0[0]).abs() < 1e-6 && (a[1] - mid0[1]).abs() < 1e-6,
                    "first UV layer snapped, got {a:?}"
                );
                assert!(
                    (d[0] - mid1[0]).abs() < 1e-6 && (d[1] - mid1[1]).abs() < 1e-6,
                    "second UV layer also snapped, got {d:?}"
                );
            }
        }
    }

    /// §4 — the participation predicate admits colour: a four-channel Color
    /// layer is snapped by the same component-wise midpoint over its channels.
    #[test]
    fn collapse_uvs_color_layer_participates() {
        let mut mesh = LoopMesh::new();
        let _uv = register_uv(&mut mesh);
        let color = register_color(&mut mesh);
        let ([_, b, c, ..], shared) = two_quads_shared_edge(&mut mesh);

        // Coherent across the seam at each vertex (both faces agree per vertex),
        // distinct between the two verts so the four corners flood into one
        // island whose four-channel bbox midpoint differs from each corner.
        let value_at = |v: VertexKey| -> [f32; 4] {
            if v == b {
                [0.0, 0.0, 0.2, 1.0]
            } else {
                [0.4, 0.6, 0.8, 1.0]
            }
        };
        for face in mesh.face_keys() {
            for l in mesh.face_loop_keys(face) {
                let v = mesh.loop_vertex(l);
                if v == b || v == c {
                    mesh.loop_attribs_mut().set_vec4(color, l.as_usize(), value_at(v));
                }
            }
        }
        let lo = value_at(b);
        let hi = value_at(c);
        let mid = [
            (lo[0].min(hi[0]) + lo[0].max(hi[0])) * 0.5,
            (lo[1].min(hi[1]) + lo[1].max(hi[1])) * 0.5,
            (lo[2].min(hi[2]) + lo[2].max(hi[2])) * 0.5,
            (lo[3].min(hi[3]) + lo[3].max(hi[3])) * 0.5,
        ];

        mesh.collapse_uvs(&[shared]);
        mesh.validate_topology();

        for l in mesh.loop_keys() {
            let v = mesh.loop_vertex(l);
            if v == b || v == c {
                let rgba = mesh.loop_attribs().get_vec4(color, l.as_usize()).unwrap();
                for k in 0..4 {
                    assert!(
                        (rgba[k] - mid[k]).abs() < 1e-6,
                        "colour channel {k} snapped to midpoint {}, got {}",
                        mid[k],
                        rgba[k]
                    );
                }
            }
        }
    }

    /// §4.1 / §7 — with no has-math loop layer the operator returns immediately:
    /// no layer is created and the topology is untouched.
    #[test]
    fn collapse_uvs_no_loop_layer_is_noop() {
        let mut mesh = LoopMesh::new();
        let (_verts, shared) = two_quads_shared_edge(&mut mesh);
        let before = (mesh.totvert(), mesh.totedge(), mesh.totface());
        let layers_before = mesh.loop_attribs().iter_descriptors().count();

        mesh.collapse_uvs(&[shared]);
        mesh.validate_topology();

        assert_eq!((mesh.totvert(), mesh.totedge(), mesh.totface()), before);
        assert_eq!(
            mesh.loop_attribs().iter_descriptors().count(),
            layers_before,
            "no loop layer was created by the early-out"
        );
    }

    /// §5 / §7 — idempotence: running twice with the same input edges equals
    /// running once. After the first pass each island is a single point, so the
    /// second pass's range is that point and the snap is a no-op.
    #[test]
    fn collapse_uvs_idempotent_double_application() {
        let mut mesh = LoopMesh::new();
        let uv = register_uv(&mut mesh);
        let ([_, b, c, ..], shared) = two_quads_shared_edge(&mut mesh);
        let _ = seed_like_valued_island(&mut mesh, uv, [b, c]);

        mesh.collapse_uvs(&[shared]);
        let once = uv_snapshot(&mesh, uv);
        mesh.collapse_uvs(&[shared]);
        let twice = uv_snapshot(&mesh, uv);

        assert_eq!(once, twice, "second application is a no-op (idempotent)");
        mesh.validate_topology();
    }

    // ---- average_vert_facedata (AVERAGE_VERT_FACEDATA.md) ----

    /// A triangle fan of `n` faces sharing one central hub vertex. Triangle `i`
    /// is wound `[hub, rim_i, rim_{(i+1)%n}]`, so the hub corner of face `i` is
    /// the loop anchored at the hub. Returns `(hub, rim_verts)`.
    fn build_triangle_fan(mesh: &mut LoopMesh, n: usize) -> (VertexKey, Vec<VertexKey>) {
        use std::f32::consts::TAU;
        let hub = mesh.create_vertex(Vec3::ZERO);
        let rim: Vec<VertexKey> = (0..n)
            .map(|i| {
                let a = TAU * i as f32 / n as f32;
                mesh.create_vertex(Vec3::new(a.cos(), a.sin(), 0.0))
            })
            .collect();
        for i in 0..n {
            let vs = [hub, rim[i], rim[(i + 1) % n]];
            let es: Vec<EdgeKey> = (0..3)
                .map(|k| mesh.create_edge([vs[k], vs[(k + 1) % 3]], EdgeCreateFlags::NO_DUPLICATE))
                .collect();
            mesh.face_create(&vs, &es, FaceCreateFlags::NONE);
        }
        (hub, rim)
    }

    /// All loops whose corner sits at `vert`, as `(slot, uv)` pairs.
    fn corner_uvs_at(mesh: &LoopMesh, uv: ElementAttributeKey, vert: VertexKey) -> Vec<[f32; 2]> {
        mesh.loop_keys()
            .into_iter()
            .filter(|&l| mesh.loop_vertex(l) == vert)
            .map(|l| mesh.loop_attribs().get_vec2(uv, l.as_usize()).unwrap())
            .collect()
    }

    fn approx_uv(a: [f32; 2], b: [f32; 2]) -> bool {
        (a[0] - b[0]).abs() < 1e-6 && (a[1] - b[1]).abs() < 1e-6
    }

    /// §2 / §3 / §4 headline — selecting the hub of a fan whose hub corners carry
    /// a distinct UV spread gathers all hub corners into one pool and snaps them
    /// to the component-wise bbox midpoint (not a mean, not any input value).
    #[test]
    fn average_vert_facedata_fan_hub_snaps_to_bbox_midpoint() {
        let mut mesh = LoopMesh::new();
        let uv = register_uv(&mut mesh);
        let (hub, _rim) = build_triangle_fan(&mut mesh, 6);

        // Assign each hub corner a distinct UV; record the bbox the pool spans.
        let hub_corners: Vec<LoopKey> = mesh
            .loop_keys()
            .into_iter()
            .filter(|&l| mesh.loop_vertex(l) == hub)
            .collect();
        assert_eq!(hub_corners.len(), 6, "fan hub has one corner per face");
        let values = [
            [0.0, 0.5],
            [0.4, 0.1],
            [0.2, 0.9],
            [0.8, 0.3],
            [0.6, 0.7],
            [0.1, 0.2],
        ];
        for (l, v) in hub_corners.iter().zip(values) {
            mesh.loop_attribs_mut().set_vec2(uv, l.as_usize(), v);
        }
        // bbox: u in [0.0..0.8] -> 0.4 ; v in [0.1..0.9] -> 0.5
        let expected = [0.4, 0.5];

        mesh.average_vert_facedata(&[hub]);
        mesh.validate_topology();

        for got in corner_uvs_at(&mesh, uv, hub) {
            assert!(approx_uv(got, expected), "hub corner snapped to {expected:?}, got {got:?}");
        }
    }

    /// §2 — the snapped value is the bbox midpoint `(min+max)/2`, NOT the
    /// arithmetic mean: a clustered spread with one outlier lands the midpoint,
    /// which the cluster would pull the mean away from.
    #[test]
    fn average_vert_facedata_bbox_midpoint_not_mean() {
        let mut mesh = LoopMesh::new();
        let uv = register_uv(&mut mesh);
        let (hub, _rim) = build_triangle_fan(&mut mesh, 6);

        // Five corners clustered near the origin, one outlier at (1,1).
        // min=(0,0), max=(1,1) -> midpoint (0.5,0.5); mean is pulled near origin.
        let hub_corners: Vec<LoopKey> = mesh
            .loop_keys()
            .into_iter()
            .filter(|&l| mesh.loop_vertex(l) == hub)
            .collect();
        for (i, l) in hub_corners.iter().enumerate() {
            let v = if i + 1 == hub_corners.len() {
                [1.0, 1.0]
            } else {
                [0.01 * i as f32, 0.02 * i as f32]
            };
            mesh.loop_attribs_mut().set_vec2(uv, l.as_usize(), v);
        }

        mesh.average_vert_facedata(&[hub]);
        mesh.validate_topology();

        for got in corner_uvs_at(&mesh, uv, hub) {
            assert!(approx_uv(got, [0.5, 0.5]), "midpoint (0.5,0.5), not mean, got {got:?}");
        }
    }

    /// §4 / §4.1 — the discriminating case: two verts with disjoint corner
    /// spreads pool into ONE global bbox, so all four corners land on one shared
    /// midpoint. A per-vertex implementation would land each vert's corners on
    /// its own midpoint (two different values). Verified by contrast against the
    /// per-vert midpoints.
    #[test]
    fn average_vert_facedata_global_single_pool_multi_vert() {
        let mut mesh = LoopMesh::new();
        let uv = register_uv(&mut mesh);
        let (_hub, rim) = build_triangle_fan(&mut mesh, 6);
        // Two non-adjacent rim verts (share no face): rim[0] and rim[3].
        let a = rim[0];
        let b = rim[3];

        // a's corners near origin, b's corners far offset; disjoint ranges.
        let set_corners = |mesh: &mut LoopMesh, vert: VertexKey, base: [f32; 2]| {
            let slots: Vec<usize> = mesh
                .loop_keys()
                .into_iter()
                .filter(|&l| mesh.loop_vertex(l) == vert)
                .map(|l| l.as_usize())
                .collect();
            for (k, slot) in slots.iter().enumerate() {
                mesh.loop_attribs_mut().set_vec2(
                    uv,
                    *slot,
                    [base[0] + 0.1 * k as f32, base[1] + 0.2 * k as f32],
                );
            }
        };
        set_corners(&mut mesh, a, [0.0, 0.0]);
        set_corners(&mut mesh, b, [5.0, 6.0]);

        // a's own spread: each rim vert is shared by two faces -> two corners.
        // a: (0.0,0.0),(0.1,0.2) -> per-vert mid (0.05,0.1)
        // b: (5.0,6.0),(5.1,6.2) -> per-vert mid (5.05,6.1)
        // global pool min=(0,0) max=(5.1,6.2) -> mid (2.55,3.1) — distinct from both.
        let global_mid = [2.55, 3.1];
        let per_vert_a = [0.05, 0.1];
        let per_vert_b = [5.05, 6.1];

        mesh.average_vert_facedata(&[a, b]);
        mesh.validate_topology();

        for got in corner_uvs_at(&mesh, uv, a).into_iter().chain(corner_uvs_at(&mesh, uv, b)) {
            assert!(approx_uv(got, global_mid), "global pool midpoint {global_mid:?}, got {got:?}");
            assert!(!approx_uv(got, per_vert_a), "not vert a's own midpoint");
            assert!(!approx_uv(got, per_vert_b), "not vert b's own midpoint");
        }
    }

    /// §4 / §4.1 contrast — one call with both verts differs from two calls with
    /// one vert each. Pins the not-reset-per-vert accumulator at the call level.
    #[test]
    fn average_vert_facedata_one_call_two_verts_differs_from_two_calls() {
        let build = || {
            let mut mesh = LoopMesh::new();
            let uv = register_uv(&mut mesh);
            let (_hub, rim) = build_triangle_fan(&mut mesh, 6);
            let (a, b) = (rim[0], rim[3]);
            for (vert, base) in [(a, [0.0, 0.0]), (b, [5.0, 6.0])] {
                let slots: Vec<usize> = mesh
                    .loop_keys()
                    .into_iter()
                    .filter(|&l| mesh.loop_vertex(l) == vert)
                    .map(|l| l.as_usize())
                    .collect();
                for (k, slot) in slots.iter().enumerate() {
                    mesh.loop_attribs_mut().set_vec2(
                        uv,
                        *slot,
                        [base[0] + 0.1 * k as f32, base[1] + 0.2 * k as f32],
                    );
                }
            }
            (mesh, uv, a, b)
        };

        let (mut one_call, uv1, a1, b1) = build();
        one_call.average_vert_facedata(&[a1, b1]);

        let (mut two_calls, uv2, a2, b2) = build();
        two_calls.average_vert_facedata(&[a2]);
        two_calls.average_vert_facedata(&[b2]);

        assert_ne!(
            corner_uvs_at(&one_call, uv1, a1),
            corner_uvs_at(&two_calls, uv2, a2),
            "global pool (one call) must differ from per-vert pools (two calls)"
        );
    }

    /// §3 / §4.2 — a UV seam meeting at the selected vert is FLATTENED, not
    /// preserved: both seam corners are pooled and land on the one midpoint
    /// (opposite of collapse_uvs's value-equality gate).
    #[test]
    fn average_vert_facedata_seam_flattened_not_preserved() {
        let mut mesh = LoopMesh::new();
        let uv = register_uv(&mut mesh);
        let (hub, _rim) = build_triangle_fan(&mut mesh, 6);

        // Two adjacent hub corners disagree (a seam at the hub) at the extremes;
        // the rest coherent in the middle. min=(0,0) max=(0.8,0.8) -> mid 0.4.
        let hub_corners: Vec<LoopKey> = mesh
            .loop_keys()
            .into_iter()
            .filter(|&l| mesh.loop_vertex(l) == hub)
            .collect();
        for (i, l) in hub_corners.iter().enumerate() {
            let v = match i {
                0 => [0.0, 0.0],
                1 => [0.8, 0.8],
                _ => [0.4, 0.4],
            };
            mesh.loop_attribs_mut().set_vec2(uv, l.as_usize(), v);
        }

        mesh.average_vert_facedata(&[hub]);
        mesh.validate_topology();

        // Every hub corner now reads the same value (seam welded to one midpoint).
        let got = corner_uvs_at(&mesh, uv, hub);
        for v in &got {
            assert!(approx_uv(*v, [0.4, 0.4]), "seam flattened to midpoint, got {v:?}");
        }
    }

    /// §5 — every has-math layer is flattened, each with its own independent
    /// pool: two UV layers are both snapped to their own bbox midpoints.
    #[test]
    fn average_vert_facedata_all_has_math_layers_flattened() {
        let mut mesh = LoopMesh::new();
        let uv0 = register_uv(&mut mesh);
        let uv1 = mesh
            .register_loop_attribute(ElementAttributeDesc::numeric(
                AttributeKey::custom("uv1"),
                Format::new(ComponentType::F32, 2),
            ))
            .unwrap();
        let (hub, _rim) = build_triangle_fan(&mut mesh, 4);

        let hub_corners: Vec<LoopKey> = mesh
            .loop_keys()
            .into_iter()
            .filter(|&l| mesh.loop_vertex(l) == hub)
            .collect();
        // uv0 spread in [0..1]^2 -> mid (0.5,0.5); uv1 offset by +10 -> mid (10.5,10.5).
        let v0 = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        for (l, v) in hub_corners.iter().zip(v0) {
            mesh.loop_attribs_mut().set_vec2(uv0, l.as_usize(), v);
            mesh.loop_attribs_mut().set_vec2(uv1, l.as_usize(), [v[0] + 10.0, v[1] + 10.0]);
        }

        mesh.average_vert_facedata(&[hub]);
        mesh.validate_topology();

        for got in corner_uvs_at(&mesh, uv0, hub) {
            assert!(approx_uv(got, [0.5, 0.5]), "uv0 snapped, got {got:?}");
        }
        for l in mesh.loop_keys().into_iter().filter(|&l| mesh.loop_vertex(l) == hub) {
            let got = mesh.loop_attribs().get_vec2(uv1, l.as_usize()).unwrap();
            assert!(approx_uv(got, [10.5, 10.5]), "uv1 snapped independently, got {got:?}");
        }
    }

    /// §5 — the participation predicate admits colour: a four-channel Color
    /// layer is snapped by the same component-wise midpoint.
    #[test]
    fn average_vert_facedata_color_layer_participates() {
        let mut mesh = LoopMesh::new();
        let color = register_color(&mut mesh);
        let (hub, _rim) = build_triangle_fan(&mut mesh, 4);

        let hub_corners: Vec<LoopKey> = mesh
            .loop_keys()
            .into_iter()
            .filter(|&l| mesh.loop_vertex(l) == hub)
            .collect();
        let cols = [
            [0.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 0.5, 1.0],
            [0.0, 1.0, 0.5, 1.0],
            [1.0, 1.0, 1.0, 1.0],
        ];
        for (l, c) in hub_corners.iter().zip(cols) {
            mesh.loop_attribs_mut().set_vec4(color, l.as_usize(), c);
        }
        // bbox: r[0..1]->0.5, g[0..1]->0.5, b[0..1]->0.5, a[1..1]->1.0
        let expected = [0.5, 0.5, 0.5, 1.0];

        mesh.average_vert_facedata(&[hub]);
        mesh.validate_topology();

        for l in mesh.loop_keys().into_iter().filter(|&l| mesh.loop_vertex(l) == hub) {
            let rgba = mesh.loop_attribs().get_vec4(color, l.as_usize()).unwrap();
            for k in 0..4 {
                assert!(
                    (rgba[k] - expected[k]).abs() < 1e-6,
                    "colour channel {k} snapped to {}, got {}",
                    expected[k],
                    rgba[k]
                );
            }
        }
    }

    /// §3 / §8.2 — corners at unselected verts keep their values exactly: the
    /// gather is by corner-vertex membership, not whole-face.
    #[test]
    fn average_vert_facedata_unselected_corners_untouched() {
        let mut mesh = LoopMesh::new();
        let uv = register_uv(&mut mesh);
        let (hub, rim) = build_triangle_fan(&mut mesh, 6);

        // Hub corners get a spread; rim corners get a far-offset band.
        for l in mesh.loop_keys() {
            let v = mesh.loop_vertex(l);
            let value = if v == hub {
                [0.1 * (l.as_usize() % 7) as f32, 0.1 * (l.as_usize() % 5) as f32]
            } else {
                [100.0 + l.as_usize() as f32, 200.0 + l.as_usize() as f32]
            };
            mesh.loop_attribs_mut().set_vec2(uv, l.as_usize(), value);
        }
        let rim_before: Vec<Vec<[f32; 2]>> =
            rim.iter().map(|&v| corner_uvs_at(&mesh, uv, v)).collect();

        mesh.average_vert_facedata(&[hub]);
        mesh.validate_topology();

        let rim_after: Vec<Vec<[f32; 2]>> =
            rim.iter().map(|&v| corner_uvs_at(&mesh, uv, v)).collect();
        assert_eq!(rim_before, rim_after, "rim corners at unselected verts untouched (§3)");
    }

    /// §5.2 / §7 — no has-math loop layer is a whole-operator no-op: nothing
    /// changes and no layer is created.
    #[test]
    fn average_vert_facedata_no_loop_layer_is_noop() {
        let mut mesh = LoopMesh::new();
        let (hub, _rim) = build_triangle_fan(&mut mesh, 6);
        let before = (mesh.totvert(), mesh.totedge(), mesh.totface());
        let layers_before = mesh.loop_attribs().iter_descriptors().count();

        mesh.average_vert_facedata(&[hub]);
        mesh.validate_topology();

        assert_eq!((mesh.totvert(), mesh.totedge(), mesh.totface()), before);
        assert_eq!(
            mesh.loop_attribs().iter_descriptors().count(),
            layers_before,
            "no loop layer created by the early-out"
        );
    }

    /// §7 — selecting a loop-less (wire/isolated) vert gathers no corners and is
    /// a no-op: a present UV layer is left byte-for-byte unchanged.
    #[test]
    fn average_vert_facedata_wire_vert_is_noop() {
        let mut mesh = LoopMesh::new();
        let uv = register_uv(&mut mesh);
        let (hub, _rim) = build_triangle_fan(&mut mesh, 6);
        for l in mesh.loop_keys() {
            mesh.loop_attribs_mut()
                .set_vec2(uv, l.as_usize(), [0.1 * l.as_usize() as f32, 0.2 * l.as_usize() as f32]);
        }
        // An isolated vert with no incident face.
        let loose = mesh.create_vertex(Vec3::new(9.0, 9.0, 9.0));
        let before = uv_snapshot(&mesh, uv);

        mesh.average_vert_facedata(&[loose]);
        mesh.validate_topology();

        assert_eq!(uv_snapshot(&mesh, uv), before, "loose vert gathers nothing");
        let _ = hub;
    }

    /// §6 / §7 — idempotence: a second pass with the same verts finds the pool a
    /// single point and snaps it to itself.
    #[test]
    fn average_vert_facedata_idempotent_double_application() {
        let mut mesh = LoopMesh::new();
        let uv = register_uv(&mut mesh);
        let (hub, _rim) = build_triangle_fan(&mut mesh, 6);
        let hub_corners: Vec<LoopKey> = mesh
            .loop_keys()
            .into_iter()
            .filter(|&l| mesh.loop_vertex(l) == hub)
            .collect();
        for (i, l) in hub_corners.iter().enumerate() {
            mesh.loop_attribs_mut()
                .set_vec2(uv, l.as_usize(), [0.1 * i as f32, 0.2 * i as f32]);
        }

        mesh.average_vert_facedata(&[hub]);
        let once = uv_snapshot(&mesh, uv);
        mesh.average_vert_facedata(&[hub]);
        let twice = uv_snapshot(&mesh, uv);

        assert_eq!(once, twice, "second application is a no-op (idempotent)");
        mesh.validate_topology();
    }

    /// §7 — an empty input vert set gathers nothing and is a complete no-op.
    #[test]
    fn average_vert_facedata_empty_input_is_noop() {
        let mut mesh = LoopMesh::new();
        let uv = register_uv(&mut mesh);
        let (_hub, _rim) = build_triangle_fan(&mut mesh, 6);
        for l in mesh.loop_keys() {
            mesh.loop_attribs_mut()
                .set_vec2(uv, l.as_usize(), [0.3 * l.as_usize() as f32, 0.4 * l.as_usize() as f32]);
        }
        let before = uv_snapshot(&mesh, uv);

        mesh.average_vert_facedata(&[]);
        mesh.validate_topology();

        assert_eq!(uv_snapshot(&mesh, uv), before, "empty input leaves UVs untouched");
    }
}
