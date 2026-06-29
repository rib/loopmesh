use std::collections::HashMap;

use crate::loopmesh::*;

/// A single entry in a [`LoopMesh::duplicate`] selection buffer. Verts, edges,
/// and faces are mixed freely in one slice; each entry is routed by its kind
/// (DUPLICATE.md §3). `VertexKey` / `EdgeKey` / `FaceKey` all `From`-convert
/// into this, so callers write `face.into()`, `vert.into()`, `edge.into()`.
#[derive(PartialEq, Eq, Clone, Copy, Debug, Hash)]
pub enum GeomKey {
    Vert(VertexKey),
    Edge(EdgeKey),
    Face(FaceKey),
}

impl From<VertexKey> for GeomKey {
    fn from(v: VertexKey) -> Self {
        GeomKey::Vert(v)
    }
}
impl From<EdgeKey> for GeomKey {
    fn from(e: EdgeKey) -> Self {
        GeomKey::Edge(e)
    }
}
impl From<FaceKey> for GeomKey {
    fn from(f: FaceKey) -> Self {
        GeomKey::Face(f)
    }
}

/// Correspondence maps reported by [`LoopMesh::duplicate`] (DUPLICATE.md §5,
/// §6). The three per-kind maps are **bidirectional** — every copied element
/// is recorded `source -> clone` *and* `clone -> source`. The two filtered
/// maps are unidirectional `source -> clone` subsets selected by the
/// selection-relative classification of §6.
#[derive(Debug, Default)]
pub struct DuplicateResult {
    /// The new elements — every cloned vert, edge, and face (DUPLICATE.md §1,
    /// the `geom.out` slot). Partitioned by kind.
    pub geom_out: Vec<GeomKey>,

    /// Bidirectional vert correspondence (§5): `source -> clone` and
    /// `clone -> source` for every copied vert.
    pub vert_map: HashMap<VertexKey, VertexKey>,
    /// Bidirectional edge correspondence (§5).
    pub edge_map: HashMap<EdgeKey, EdgeKey>,
    /// Bidirectional face correspondence (§5).
    pub face_map: HashMap<FaceKey, FaceKey>,

    /// Frontier edges (§6.1): the `source -> clone` subset of `edge_map` for
    /// edges whose in-selection radial count is fewer than two — wire edges
    /// (count 0) and single-selected-face boundary edges (count 1), but not
    /// fully-interior edges (>= 2).
    pub boundary_map: HashMap<EdgeKey, EdgeKey>,
    /// Isolated seed verts (§6.2): the `source -> clone` subset of `vert_map`
    /// for selected verts touching no selected face and no selected edge.
    pub isovert_map: HashMap<VertexKey, VertexKey>,
}

/// The four per-domain customdata remap plans (DUPLICATE.md §8.5.2),
/// destination-driven and matched by `(type, name)`. Built once before the
/// cross-mesh closure walk and replayed for every cloned element of each
/// domain. Each plan collapses to the verbatim copy of §4 when the two meshes
/// share a layout.
struct RemapPlans {
    vert: RemapPlan,
    edge: RemapPlan,
    face: RemapPlan,
    loops: RemapPlan,
}

impl LoopMesh {
    /// Clone a mixed selection into new, disjoint, geometrically-coincident
    /// geometry in place.
    ///
    /// The input is a single buffer mixing verts, edges, and faces. Duplicate
    /// walks the selection's **closure** — a face implies its edges and verts,
    /// an edge implies its endpoints — and copies each distinct closure element
    /// exactly once, sharing nothing with the originals. The originals are left
    /// entirely untouched; the clone is a separate connected component sitting
    /// at the same coordinates (the operator moves nothing, so the caller is
    /// expected to displace the clone afterward via the returned `geom_out`).
    ///
    /// Each clone is a **verbatim** per-element customdata copy of its single
    /// source — never an interpolation (DUPLICATE.md §4). The returned maps
    /// report the source/clone correspondence (§5, §6).
    ///
    /// When `use_edge_flip_from_face` is set, a cloned **boundary** edge (its
    /// source edge has exactly one incident face) whose source face traverses
    /// it "v1-first" has its two clone endpoints swapped, so a downstream
    /// extrude building a wall off the clone gets a consistent winding
    /// (DUPLICATE.md §7). The swap touches only the clone edge's internal
    /// endpoint order — never which two verts it connects, nor any incidence —
    /// so it is invisible to a topology-only comparison.
    ///
    /// See `radial-mesh-guides/operators/DUPLICATE.md` for the full algorithm
    /// and parity contract.
    /// Per-domain selection membership keyed by source slot: `(vert, edge,
    /// face)` boolean vectors sized to each domain's key watermark, with a
    /// `true` at every slot named in `geom`. Shared by the in-place
    /// [`duplicate`](Self::duplicate) and the cross-mesh
    /// [`duplicate_into`](Self::duplicate_into) so the §6 "is this incident
    /// element itself selected?" classification cannot drift between the two
    /// paths.
    fn selection_flags(&self, geom: &[GeomKey]) -> (Vec<bool>, Vec<bool>, Vec<bool>) {
        let mut vert_selected = vec![false; self.vertices.key_watermark()];
        let mut edge_selected = vec![false; self.edges.key_watermark()];
        let mut face_selected = vec![false; self.faces.key_watermark()];
        for entry in geom {
            match *entry {
                GeomKey::Vert(v) => vert_selected[v.as_usize()] = true,
                GeomKey::Edge(e) => edge_selected[e.as_usize()] = true,
                GeomKey::Face(f) => face_selected[f.as_usize()] = true,
            }
        }
        (vert_selected, edge_selected, face_selected)
    }

    pub fn duplicate(
        &mut self,
        geom: &[GeomKey],
        use_edge_flip_from_face: bool,
    ) -> DuplicateResult {
        let mut result = DuplicateResult::default();

        // Selection membership, keyed by source slot, so the §6 classification
        // can ask "is this incident element itself in the selection?".
        let (vert_selected, edge_selected, face_selected) = self.selection_flags(geom);

        // The two ledgers (DUPLICATE.md §3): source element -> clone, so an
        // element reached through several selected elements is copied once.
        let mut vert_ledger: HashMap<VertexKey, VertexKey> = HashMap::new();
        let mut edge_ledger: HashMap<EdgeKey, EdgeKey> = HashMap::new();

        // ---- Pass A — selected verts, with the isolation test (§3.1) ----
        for entry in geom {
            let GeomKey::Vert(source) = *entry else {
                continue;
            };
            if vert_ledger.contains_key(&source) {
                continue;
            }
            let clone = self.clone_vertex(source, &mut result);
            vert_ledger.insert(source, clone);

            if self.vert_isolated_in_selection(source, &edge_selected, &face_selected) {
                result.isovert_map.insert(source, clone);
            }
        }

        // ---- Pass B — selected edges, closing endpoints first (§3.2) ----
        for entry in geom {
            let GeomKey::Edge(source) = *entry else {
                continue;
            };
            if edge_ledger.contains_key(&source) {
                continue;
            }
            self.clone_edge(
                source,
                &mut vert_ledger,
                &mut edge_ledger,
                &face_selected,
                use_edge_flip_from_face,
                &mut result,
            );
        }

        // ---- Pass C — selected faces: corners, perimeter, then assemble
        //               (§3.3) ----
        for entry in geom {
            let GeomKey::Face(source) = *entry else {
                continue;
            };
            if result.face_map.contains_key(&source) {
                continue;
            }
            self.clone_face(
                source,
                &mut vert_ledger,
                &mut edge_ledger,
                &face_selected,
                use_edge_flip_from_face,
                &mut result,
            );
        }

        result
    }

    /// §3.1 / §5: copy one vert verbatim, record both map directions, tag the
    /// clone for `geom_out`. The shared copy routine for all three passes.
    fn clone_vertex(&mut self, source: VertexKey, result: &mut DuplicateResult) -> VertexKey {
        let clone = self.create_vertex(self.vertex_position(source));
        self.vertex_attribs
            .copy_into(clone.as_usize(), source.as_usize());
        result.vert_map.insert(source, clone);
        result.vert_map.insert(clone, source);
        result.geom_out.push(GeomKey::Vert(clone));
        clone
    }

    /// §3.2 / §6.1 / §5: close the edge's endpoints (looking each up in or
    /// adding it to the vert ledger), copy the edge verbatim joining the two
    /// endpoint clones, record both map directions, classify it for
    /// `boundary_map`, and record it in the edge ledger.
    fn clone_edge(
        &mut self,
        source: EdgeKey,
        vert_ledger: &mut HashMap<VertexKey, VertexKey>,
        edge_ledger: &mut HashMap<EdgeKey, EdgeKey>,
        face_selected: &[bool],
        use_edge_flip_from_face: bool,
        result: &mut DuplicateResult,
    ) -> EdgeKey {
        let [v0, v1] = self.edge_vertices(source);
        let c0 = self.closure_vertex(v0, vert_ledger, result);
        let c1 = self.closure_vertex(v1, vert_ledger, result);

        // §7: a boundary edge whose sole source face traverses it v1-first has
        // its clone endpoints swapped, fixing the downstream extrude wall's
        // winding. The endpoint *set* is unchanged — only the internal order.
        let endpoints = if use_edge_flip_from_face && self.edge_flips_from_face(source) {
            [c1, c0]
        } else {
            [c0, c1]
        };

        let clone = self.create_edge(endpoints, EdgeCreateFlags::NONE);
        self.edge_attribs
            .copy_into(clone.as_usize(), source.as_usize());
        result.edge_map.insert(source, clone);
        result.edge_map.insert(clone, source);
        result.geom_out.push(GeomKey::Edge(clone));

        if self.edge_in_selection_radial_count(source, face_selected) < 2 {
            result.boundary_map.insert(source, clone);
        }

        edge_ledger.insert(source, clone);
        clone
    }

    /// §3.3: clone the corner verts, the perimeter edges, then assemble the
    /// clone face from the ledger lookups, copying per-face and per-loop
    /// customdata verbatim in loop order.
    fn clone_face(
        &mut self,
        source: FaceKey,
        vert_ledger: &mut HashMap<VertexKey, VertexKey>,
        edge_ledger: &mut HashMap<EdgeKey, EdgeKey>,
        face_selected: &[bool],
        use_edge_flip_from_face: bool,
        result: &mut DuplicateResult,
    ) -> FaceKey {
        // Snapshot the source face's loops in cycle order so the clone is built
        // in the same order and per-loop customdata copies position-for-position.
        let source_loops: Vec<LoopKey> = self.face_loop_keys(source);
        let source_verts: Vec<VertexKey> =
            source_loops.iter().map(|&l| self.loop_vertex(l)).collect();
        let source_edges: Vec<EdgeKey> = source_loops.iter().map(|&l| self.loop_edge(l)).collect();

        // §3.3 step 1: clone corner verts via the ledger.
        let clone_verts: Vec<VertexKey> = source_verts
            .iter()
            .map(|&v| self.closure_vertex(v, vert_ledger, result))
            .collect();

        // §3.3 step 2: clone perimeter edges via the ledger (including the
        // boundary-map classification through `clone_edge`). An edge already in
        // the ledger — shared with another selected face, or named directly —
        // is reused, so two faces sharing an edge share its clone.
        let clone_edges: Vec<EdgeKey> = source_edges
            .iter()
            .map(|&e| match edge_ledger.get(&e) {
                Some(&existing) => existing,
                None => self.clone_edge(
                    e,
                    vert_ledger,
                    edge_ledger,
                    face_selected,
                    use_edge_flip_from_face,
                    result,
                ),
            })
            .collect();

        // §3.3 step 3: assemble. The clone verts and edges are already in the
        // source loop-cycle order, so the clone face's loops align with the
        // source's one-to-one.
        let clone = self.face_create(&clone_verts, &clone_edges, FaceCreateFlags::NONE);
        self.face_attribs
            .copy_into(clone.as_usize(), source.as_usize());

        if !self.loop_attribs.is_empty() {
            let clone_loops: Vec<LoopKey> = self.face_loop_keys(clone);
            for (clone_loop, source_loop) in clone_loops.iter().zip(source_loops.iter()) {
                self.loop_attribs
                    .copy_into(clone_loop.as_usize(), source_loop.as_usize());
            }
        }

        result.face_map.insert(source, clone);
        result.face_map.insert(clone, source);
        result.geom_out.push(GeomKey::Face(clone));
        clone
    }

    /// Closure-walk vert copy (§3.2 / §3.3): a vert already in the ledger is
    /// reused; otherwise it is cloned (no isolation test — an edge endpoint or
    /// face corner is by definition not isolated, §3.1).
    fn closure_vertex(
        &mut self,
        source: VertexKey,
        vert_ledger: &mut HashMap<VertexKey, VertexKey>,
        result: &mut DuplicateResult,
    ) -> VertexKey {
        if let Some(&existing) = vert_ledger.get(&source) {
            return existing;
        }
        let clone = self.clone_vertex(source, result);
        vert_ledger.insert(source, clone);
        clone
    }

    /// §6.2 isolation test: true when `source` touches no selected face and no
    /// selected edge. Reads the *source* element's incidence against the
    /// selection flags.
    fn vert_isolated_in_selection(
        &self,
        source: VertexKey,
        edge_selected: &[bool],
        face_selected: &[bool],
    ) -> bool {
        for edge in self.iter_disk_edges(source) {
            if edge_selected[edge.as_usize()] {
                return false;
            }
            for loop_key in self.iter_radial_loops(edge) {
                let face = self.loops.get(loop_key).face;
                if face_selected[face.as_usize()] {
                    return false;
                }
            }
        }
        true
    }

    /// §7 swap precondition: true when `source` is a **boundary** edge (exactly
    /// one incident face) and that face traverses the edge **v1-first** — its
    /// loop on the edge starts at the edge's `v1` (loop direction v1 -> v0
    /// relative to the stored `[v0, v1]` order). A loop's vertex is the corner
    /// it emanates from, so "starts at v1" is `loop_vertex(loop) == v1`.
    fn edge_flips_from_face(&self, source: EdgeKey) -> bool {
        let [_v0, v1] = self.edge_vertices(source);
        let mut radial = self.iter_radial_loops(source);
        let Some(only_loop) = radial.next() else {
            return false;
        };
        if radial.next().is_some() {
            return false; // not a boundary edge (>= 2 incident faces)
        }
        self.loop_vertex(only_loop) == v1
    }

    /// §6.1 in-selection radial count: how many of `source`'s incident faces are
    /// themselves in the selection.
    fn edge_in_selection_radial_count(&self, source: EdgeKey, face_selected: &[bool]) -> usize {
        let mut count = 0;
        for loop_key in self.iter_radial_loops(source) {
            let face = self.loops.get(loop_key).face;
            if face_selected[face.as_usize()] {
                count += 1;
            }
        }
        count
    }

    /// Clone a mixed selection into a **separate** destination mesh
    /// (DUPLICATE.md §8.5). The source mesh `self` is left entirely untouched;
    /// every clone vert, edge, face, and loop is born in `dest`.
    ///
    /// The §3 A→B→C closure walk, the shared vert/edge ledgers, and the §6
    /// boundary/isovert classification run exactly as the in-place
    /// [`duplicate`](Self::duplicate) — every classification reads the
    /// **source** mesh's incidence and selection flags, and the ledgers map each
    /// source element to its destination clone (§8.5.1). The §4 verbatim
    /// per-element customdata copy is replaced by the destination-driven
    /// `(type, name)` layer-layout remap of §8.5.2: a layer present on both
    /// meshes is copied source→clone, a destination-only layer is left at its
    /// default on the clone, and a source-only layer is dropped. It is still a
    /// copy, never an interpolation — one source per clone.
    ///
    /// Map contract cross-mesh (§8.5.1): `vert_map` / `edge_map` / `face_map`
    /// carry `source → clone` and, together with `boundary_map` / `isovert_map`,
    /// are the **primary** clone enumeration (their values are the destination
    /// clones). They are **forward-only** here — unlike the in-place §5
    /// bidirectional maps — because source and clone keys live in independent
    /// per-mesh index spaces and a single map cannot hold both directions
    /// without key collision (a clone in an empty `dest` reuses the source's low
    /// indices). `geom_out` comes back **empty** cross-mesh (§8.5.1): it is the
    /// same-mesh "new elements in this mesh" convenience, and the clones live in
    /// `dest`; a cross-mesh caller reads the maps instead.
    ///
    /// See `radial-mesh-guides/operators/DUPLICATE.md` §8.5.
    pub fn duplicate_into(
        &self,
        geom: &[GeomKey],
        use_edge_flip_from_face: bool,
        dest: &mut LoopMesh,
    ) -> DuplicateResult {
        let mut result = DuplicateResult::default();

        // §8.5.2: one destination-driven remap plan per domain, built before the
        // walk and replayed per cloned element.
        let plans = RemapPlans {
            vert: dest.vertex_attribs.build_remap_from(&self.vertex_attribs),
            edge: dest.edge_attribs.build_remap_from(&self.edge_attribs),
            face: dest.face_attribs.build_remap_from(&self.face_attribs),
            loops: dest.loop_attribs.build_remap_from(&self.loop_attribs),
        };

        // Selection membership reads source-side (§8.5.1) so the §6
        // classification can ask "is this incident element itself selected?".
        let (vert_selected, edge_selected, face_selected) = self.selection_flags(geom);

        // Ledgers keyed by source identity, value is the destination clone.
        let mut vert_ledger: HashMap<VertexKey, VertexKey> = HashMap::new();
        let mut edge_ledger: HashMap<EdgeKey, EdgeKey> = HashMap::new();

        // ---- Pass A — selected verts, with the isolation test (§3.1) ----
        for entry in geom {
            let GeomKey::Vert(source) = *entry else {
                continue;
            };
            if vert_ledger.contains_key(&source) {
                continue;
            }
            let clone = self.clone_vertex_into(source, dest, &plans, &mut result);
            vert_ledger.insert(source, clone);

            if self.vert_isolated_in_selection(source, &edge_selected, &face_selected) {
                result.isovert_map.insert(source, clone);
            }
        }

        // ---- Pass B — selected edges, closing endpoints first (§3.2) ----
        for entry in geom {
            let GeomKey::Edge(source) = *entry else {
                continue;
            };
            if edge_ledger.contains_key(&source) {
                continue;
            }
            self.clone_edge_into(
                source,
                dest,
                &mut vert_ledger,
                &mut edge_ledger,
                &face_selected,
                use_edge_flip_from_face,
                &plans,
                &mut result,
            );
        }

        // ---- Pass C — selected faces: corners, perimeter, then assemble
        //               (§3.3) ----
        for entry in geom {
            let GeomKey::Face(source) = *entry else {
                continue;
            };
            if result.face_map.contains_key(&source) {
                continue;
            }
            self.clone_face_into(
                source,
                dest,
                &mut vert_ledger,
                &mut edge_ledger,
                &face_selected,
                use_edge_flip_from_face,
                &plans,
                &mut result,
            );
        }

        result
    }

    /// Cross-mesh analogue of [`clone_vertex`](Self::clone_vertex): create the
    /// clone in `dest`, remap per-vert customdata (§8.5.2), and record the
    /// forward `source → clone` correspondence. No `geom_out` push — cross-mesh
    /// `geom.out` is empty (§8.5.1).
    fn clone_vertex_into(
        &self,
        source: VertexKey,
        dest: &mut LoopMesh,
        plans: &RemapPlans,
        result: &mut DuplicateResult,
    ) -> VertexKey {
        let clone = dest.create_vertex(self.vertex_position(source));
        dest.vertex_attribs.apply_remap_from(
            &plans.vert,
            clone.as_usize(),
            &self.vertex_attribs,
            source.as_usize(),
        );
        result.vert_map.insert(source, clone);
        clone
    }

    /// Cross-mesh analogue of [`clone_edge`](Self::clone_edge): close endpoints
    /// into `dest` via the ledger, create the clone edge in `dest`, remap
    /// per-edge customdata (§8.5.2), record the forward correspondence, and
    /// classify for `boundary_map` source-side (§6.1).
    #[allow(clippy::too_many_arguments)]
    fn clone_edge_into(
        &self,
        source: EdgeKey,
        dest: &mut LoopMesh,
        vert_ledger: &mut HashMap<VertexKey, VertexKey>,
        edge_ledger: &mut HashMap<EdgeKey, EdgeKey>,
        face_selected: &[bool],
        use_edge_flip_from_face: bool,
        plans: &RemapPlans,
        result: &mut DuplicateResult,
    ) -> EdgeKey {
        let [v0, v1] = self.edge_vertices(source);
        let c0 = self.closure_vertex_into(v0, dest, vert_ledger, plans, result);
        let c1 = self.closure_vertex_into(v1, dest, vert_ledger, plans, result);

        // §7 swap reads the source edge's boundary status and winding (§8.5.3:
        // orthogonal to the remap — it touches endpoint order, never attributes).
        let endpoints = if use_edge_flip_from_face && self.edge_flips_from_face(source) {
            [c1, c0]
        } else {
            [c0, c1]
        };

        let clone = dest.create_edge(endpoints, EdgeCreateFlags::NONE);
        dest.edge_attribs.apply_remap_from(
            &plans.edge,
            clone.as_usize(),
            &self.edge_attribs,
            source.as_usize(),
        );
        result.edge_map.insert(source, clone);

        if self.edge_in_selection_radial_count(source, face_selected) < 2 {
            result.boundary_map.insert(source, clone);
        }

        edge_ledger.insert(source, clone);
        clone
    }

    /// Cross-mesh analogue of [`clone_face`](Self::clone_face): clone the corner
    /// verts and perimeter edges into `dest` via the ledgers, assemble the clone
    /// face in `dest`, then remap per-face and per-loop customdata in loop order
    /// (§8.5.2).
    #[allow(clippy::too_many_arguments)]
    fn clone_face_into(
        &self,
        source: FaceKey,
        dest: &mut LoopMesh,
        vert_ledger: &mut HashMap<VertexKey, VertexKey>,
        edge_ledger: &mut HashMap<EdgeKey, EdgeKey>,
        face_selected: &[bool],
        use_edge_flip_from_face: bool,
        plans: &RemapPlans,
        result: &mut DuplicateResult,
    ) -> FaceKey {
        let source_loops: Vec<LoopKey> = self.face_loop_keys(source);
        let source_verts: Vec<VertexKey> =
            source_loops.iter().map(|&l| self.loop_vertex(l)).collect();
        let source_edges: Vec<EdgeKey> = source_loops.iter().map(|&l| self.loop_edge(l)).collect();

        let clone_verts: Vec<VertexKey> = source_verts
            .iter()
            .map(|&v| self.closure_vertex_into(v, dest, vert_ledger, plans, result))
            .collect();

        let clone_edges: Vec<EdgeKey> = source_edges
            .iter()
            .map(|&e| match edge_ledger.get(&e) {
                Some(&existing) => existing,
                None => self.clone_edge_into(
                    e,
                    dest,
                    vert_ledger,
                    edge_ledger,
                    face_selected,
                    use_edge_flip_from_face,
                    plans,
                    result,
                ),
            })
            .collect();

        let clone = dest.face_create(&clone_verts, &clone_edges, FaceCreateFlags::NONE);
        dest.face_attribs.apply_remap_from(
            &plans.face,
            clone.as_usize(),
            &self.face_attribs,
            source.as_usize(),
        );

        if !plans.loops.is_empty() {
            let clone_loops: Vec<LoopKey> = dest.face_loop_keys(clone);
            for (clone_loop, source_loop) in clone_loops.iter().zip(source_loops.iter()) {
                dest.loop_attribs.apply_remap_from(
                    &plans.loops,
                    clone_loop.as_usize(),
                    &self.loop_attribs,
                    source_loop.as_usize(),
                );
            }
        }

        result.face_map.insert(source, clone);
        clone
    }

    /// Cross-mesh closure-walk vert copy (§3.2 / §3.3): reuse the ledger clone
    /// if present, else clone the source vert into `dest`.
    fn closure_vertex_into(
        &self,
        source: VertexKey,
        dest: &mut LoopMesh,
        vert_ledger: &mut HashMap<VertexKey, VertexKey>,
        plans: &RemapPlans,
        result: &mut DuplicateResult,
    ) -> VertexKey {
        if let Some(&existing) = vert_ledger.get(&source) {
            return existing;
        }
        let clone = self.clone_vertex_into(source, dest, plans, result);
        vert_ledger.insert(source, clone);
        clone
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use glam::Vec3;

    use crate::loopmesh::test_support::create_face_from_verts;
    use crate::loopmesh::*;

    fn unit_quad(mesh: &mut LoopMesh) -> FaceKey {
        create_face_from_verts(
            mesh,
            &[
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(1.0, 1.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
            ],
        )
    }

    fn clone_verts(result: &DuplicateResult) -> Vec<VertexKey> {
        result
            .geom_out
            .iter()
            .filter_map(|g| match g {
                GeomKey::Vert(v) => Some(*v),
                _ => None,
            })
            .collect()
    }
    fn clone_edges(result: &DuplicateResult) -> Vec<EdgeKey> {
        result
            .geom_out
            .iter()
            .filter_map(|g| match g {
                GeomKey::Edge(e) => Some(*e),
                _ => None,
            })
            .collect()
    }
    fn clone_faces(result: &DuplicateResult) -> Vec<FaceKey> {
        result
            .geom_out
            .iter()
            .filter_map(|g| match g {
                GeomKey::Face(f) => Some(*f),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn single_quad_full_closure() {
        // §3.3 / §8: naming one face clones its whole closure — 4 corner verts,
        // 4 perimeter edges, the face, every loop — disjoint and coincident.
        let mut mesh = LoopMesh::new();
        let face = unit_quad(&mut mesh);
        let orig_verts: HashSet<VertexKey> = mesh.face_vert_keys(face).into_iter().collect();
        let orig_edges: HashSet<EdgeKey> = mesh.face_edge_keys(face).into_iter().collect();

        let result = mesh.duplicate(&[face.into()], false);

        // Additive: original untouched, full coincident copy alongside.
        assert_eq!(mesh.vertex_count(), 8, "4 originals + 4 clones");
        assert_eq!(mesh.edge_count(), 8);
        assert_eq!(mesh.face_count(), 2);
        assert_eq!(clone_verts(&result).len(), 4);
        assert_eq!(clone_edges(&result).len(), 4);
        assert_eq!(clone_faces(&result).len(), 1);
        assert_eq!(result.geom_out.len(), 9, "4 verts + 4 edges + 1 face");

        // Bidirectional maps: N elements -> 2N entries (§5).
        assert_eq!(result.vert_map.len(), 8);
        assert_eq!(result.edge_map.len(), 8);
        assert_eq!(result.face_map.len(), 2);

        // Every perimeter edge is radial-1 -> count 1 < 2 -> all 4 in boundary.
        assert_eq!(result.boundary_map.len(), 4);
        assert!(result.isovert_map.is_empty());

        // Disjoint: no clone vert/edge coincides with a source element.
        for v in clone_verts(&result) {
            assert!(!orig_verts.contains(&v), "clone vert is fresh");
        }
        for e in clone_edges(&result) {
            assert!(!orig_edges.contains(&e), "clone edge is fresh");
        }
        // Coincident: each clone vert sits on a source position.
        let src_positions: HashSet<[u32; 3]> = orig_verts
            .iter()
            .map(|&v| {
                let p = mesh.vertex_position(v);
                [p.x.to_bits(), p.y.to_bits(), p.z.to_bits()]
            })
            .collect();
        for v in clone_verts(&result) {
            let p = mesh.vertex_position(v);
            assert!(src_positions.contains(&[p.x.to_bits(), p.y.to_bits(), p.z.to_bits()]));
        }
        // The clone face shares no edge with the original.
        let clone_face = clone_faces(&result)[0];
        for e in mesh.face_edge_keys(clone_face) {
            assert!(!orig_edges.contains(&e));
        }
        mesh.validate_topology();
    }

    #[test]
    fn loose_vert_isovert() {
        // §6.2 / §8: a lone selected vert with no incident edge/face -> one
        // clone, one isovert-map entry, empty boundary map.
        let mut mesh = LoopMesh::new();
        let v = mesh.create_vertex(Vec3::new(0.5, 0.5, 0.0));

        let result = mesh.duplicate(&[v.into()], false);

        assert_eq!(mesh.vertex_count(), 2, "original + clone");
        assert_eq!(clone_verts(&result).len(), 1);
        assert_eq!(result.geom_out.len(), 1);
        assert_eq!(result.vert_map.len(), 2, "bidirectional");
        assert_eq!(result.isovert_map.len(), 1, "the lone vert is isolated");
        assert!(result.boundary_map.is_empty());
        assert!(result.edge_map.is_empty());

        let clone = clone_verts(&result)[0];
        assert_eq!(result.isovert_map.get(&v), Some(&clone));
        assert_eq!(mesh.vertex_position(clone), mesh.vertex_position(v));
        mesh.validate_topology();
    }

    #[test]
    fn wire_edge_boundary() {
        // §6.1 / §8: a faceless wire edge -> in-selection radial count 0 -> in
        // boundary_map. Both endpoints are closed by the closure walk but, held
        // by the cloned wire edge, are NOT isoverts.
        let mut mesh = LoopMesh::new();
        let a = mesh.create_vertex(Vec3::new(0.0, 0.0, 0.0));
        let b = mesh.create_vertex(Vec3::new(1.0, 0.0, 0.0));
        let wire = mesh.create_edge([a, b], EdgeCreateFlags::NONE);

        let result = mesh.duplicate(&[wire.into()], false);

        assert_eq!(mesh.vertex_count(), 4, "2 originals + 2 closed endpoints");
        assert_eq!(mesh.edge_count(), 2);
        assert_eq!(clone_verts(&result).len(), 2, "both endpoints closed");
        assert_eq!(clone_edges(&result).len(), 1);
        assert_eq!(result.geom_out.len(), 3);

        assert_eq!(result.vert_map.len(), 4);
        assert_eq!(result.edge_map.len(), 2);
        assert_eq!(result.boundary_map.len(), 1, "wire edge count 0 < 2");
        assert!(
            result.isovert_map.is_empty(),
            "endpoints held by the wire clone"
        );

        let clone_edge = clone_edges(&result)[0];
        let [ca, cb] = mesh.edge_vertices(clone_edge);
        assert!(ca != a && ca != b && cb != a && cb != b, "fresh endpoints");
        mesh.validate_topology();
    }

    #[test]
    fn mixed_buffer_face_edge_vert() {
        // §3 three-pass routing in one buffer: a face + a disjoint wire edge +
        // a loose vert. geom_out = 13 (4+4+1 face closure, 2+1 wire, 1 vert);
        // boundary_map = 5 (4 perimeter + 1 wire); isovert_map = 1 (loose vert).
        let mut mesh = LoopMesh::new();
        let face = create_face_from_verts(
            &mut mesh,
            &[
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(1.0, 1.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
            ],
        );
        let wa = mesh.create_vertex(Vec3::new(3.0, 0.0, 0.0));
        let wb = mesh.create_vertex(Vec3::new(4.0, 0.0, 0.0));
        let wire = mesh.create_edge([wa, wb], EdgeCreateFlags::NONE);
        let loose = mesh.create_vertex(Vec3::new(6.0, 6.0, 0.0));

        let result = mesh.duplicate(&[face.into(), wire.into(), loose.into()], false);

        // Closure: 4 face verts + 4 face edges + 1 face + 2 wire verts + 1 wire
        // edge + 1 loose vert = 13.
        assert_eq!(result.geom_out.len(), 13);
        assert_eq!(clone_verts(&result).len(), 7, "4 face + 2 wire + 1 loose");
        assert_eq!(clone_edges(&result).len(), 5, "4 face + 1 wire");
        assert_eq!(clone_faces(&result).len(), 1);

        assert_eq!(result.vert_map.len(), 14);
        assert_eq!(result.edge_map.len(), 10);
        assert_eq!(result.face_map.len(), 2);

        assert_eq!(
            result.boundary_map.len(),
            5,
            "4 quad perimeter (count 1) + 1 wire (count 0)"
        );
        assert_eq!(result.isovert_map.len(), 1, "only the loose vert");
        let loose_clone = result.vert_map.get(&loose).copied().unwrap();
        assert_eq!(result.isovert_map.get(&loose), Some(&loose_clone));
        mesh.validate_topology();
    }

    #[test]
    fn two_shared_faces_ledger_and_boundary() {
        // §3.4 / §6.1: two faces sharing an edge -> the shared edge cloned
        // ONCE (6 clone verts, 7 clone edges), and the shared clone edge is
        // interior (in-selection radial count 2) so it is ABSENT from
        // boundary_map; only the 6 outer perimeter edges are present.
        let mut mesh = LoopMesh::new();
        // Two quads sharing the edge between (1,0) and (1,1).
        let v0 = mesh.create_vertex(Vec3::new(0.0, 0.0, 0.0));
        let v1 = mesh.create_vertex(Vec3::new(1.0, 0.0, 0.0));
        let v2 = mesh.create_vertex(Vec3::new(1.0, 1.0, 0.0));
        let v3 = mesh.create_vertex(Vec3::new(0.0, 1.0, 0.0));
        let v4 = mesh.create_vertex(Vec3::new(2.0, 0.0, 0.0));
        let v5 = mesh.create_vertex(Vec3::new(2.0, 1.0, 0.0));
        let e0 = mesh.create_edge([v0, v1], EdgeCreateFlags::NONE);
        let e1 = mesh.create_edge([v1, v2], EdgeCreateFlags::NONE); // shared
        let e2 = mesh.create_edge([v2, v3], EdgeCreateFlags::NONE);
        let e3 = mesh.create_edge([v3, v0], EdgeCreateFlags::NONE);
        let e4 = mesh.create_edge([v1, v4], EdgeCreateFlags::NONE);
        let e5 = mesh.create_edge([v4, v5], EdgeCreateFlags::NONE);
        let e6 = mesh.create_edge([v5, v2], EdgeCreateFlags::NONE);
        let fa = mesh.face_create(&[v0, v1, v2, v3], &[e0, e1, e2, e3], FaceCreateFlags::NONE);
        let fb = mesh.face_create(&[v1, v4, v5, v2], &[e4, e5, e6, e1], FaceCreateFlags::NONE);

        let result = mesh.duplicate(&[fa.into(), fb.into()], false);

        assert_eq!(clone_verts(&result).len(), 6, "shared corners deduped");
        assert_eq!(clone_edges(&result).len(), 7, "shared edge cloned once");
        assert_eq!(clone_faces(&result).len(), 2);

        // The shared clone edge is radial-2 in the copy: both clone faces use it.
        let clone_fa = result.face_map.get(&fa).copied().unwrap();
        let clone_fb = result.face_map.get(&fb).copied().unwrap();
        let fa_edges: HashSet<EdgeKey> = mesh.face_edge_keys(clone_fa).into_iter().collect();
        let fb_edges: HashSet<EdgeKey> = mesh.face_edge_keys(clone_fb).into_iter().collect();
        let shared: Vec<EdgeKey> = fa_edges.intersection(&fb_edges).copied().collect();
        assert_eq!(
            shared.len(),
            1,
            "clone region welded across one shared edge"
        );

        // boundary_map = 6 outer perimeter edges; the shared edge (count 2) is
        // absent.
        assert_eq!(result.boundary_map.len(), 6, "outer perimeter only");
        assert!(
            !result.boundary_map.contains_key(&e1),
            "interior shared edge absent from boundary_map"
        );
        assert!(result.isovert_map.is_empty());
        mesh.validate_topology();
    }

    #[test]
    fn two_disjoint_faces() {
        // §8: two non-adjacent faces -> independent clones, no ledger collision.
        // Every perimeter edge (count 1) is a boundary-map entry: 8 total.
        let mut mesh = LoopMesh::new();
        let fa = create_face_from_verts(
            &mut mesh,
            &[
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(1.0, 1.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
            ],
        );
        let fb = create_face_from_verts(
            &mut mesh,
            &[
                Vec3::new(5.0, 0.0, 0.0),
                Vec3::new(6.0, 0.0, 0.0),
                Vec3::new(6.0, 1.0, 0.0),
                Vec3::new(5.0, 1.0, 0.0),
            ],
        );

        let result = mesh.duplicate(&[fa.into(), fb.into()], false);

        assert_eq!(clone_verts(&result).len(), 8, "no shared verts");
        assert_eq!(clone_edges(&result).len(), 8, "no shared edges");
        assert_eq!(clone_faces(&result).len(), 2);
        assert_eq!(
            result.boundary_map.len(),
            8,
            "all perimeter edges are frontier"
        );
        assert!(result.isovert_map.is_empty());

        // The two clone faces share nothing.
        let clone_fa = result.face_map.get(&fa).copied().unwrap();
        let clone_fb = result.face_map.get(&fb).copied().unwrap();
        let fa_verts: HashSet<VertexKey> = mesh.face_vert_keys(clone_fa).into_iter().collect();
        let fb_verts: HashSet<VertexKey> = mesh.face_vert_keys(clone_fb).into_iter().collect();
        assert!(fa_verts.is_disjoint(&fb_verts), "disjoint clones");
        mesh.validate_topology();
    }

    /// Build a single quad whose boundary edge `{v0, v1}` is traversed
    /// **v1-first** by the face — the §7 swap precondition. The face visits the
    /// corners a -> b -> c -> d, but the a–b edge is *stored* as `[b, a]`, so
    /// the face loop on it starts at the edge's `v1` (= a). Returns the face and
    /// that boundary edge.
    fn quad_with_v1_first_edge(mesh: &mut LoopMesh) -> (FaceKey, EdgeKey) {
        let a = mesh.create_vertex(Vec3::new(0.0, 0.0, 0.0));
        let b = mesh.create_vertex(Vec3::new(1.0, 0.0, 0.0));
        let c = mesh.create_vertex(Vec3::new(1.0, 1.0, 0.0));
        let d = mesh.create_vertex(Vec3::new(0.0, 1.0, 0.0));
        // Stored [b, a]: v0 = b, v1 = a. The face loop runs a -> b on it, so the
        // loop starts at a = the edge's v1 -> v1-first.
        let e_ab = mesh.create_edge([b, a], EdgeCreateFlags::NONE);
        let e_bc = mesh.create_edge([b, c], EdgeCreateFlags::NONE);
        let e_cd = mesh.create_edge([c, d], EdgeCreateFlags::NONE);
        let e_da = mesh.create_edge([d, a], EdgeCreateFlags::NONE);
        let face = mesh.face_create(&[a, b, c, d], &[e_ab, e_bc, e_cd, e_da], FaceCreateFlags::NONE);
        (face, e_ab)
    }

    #[test]
    fn edge_flip_from_face_reverses_boundary_clone_edge_order() {
        // §7 / §7.1: selecting a face's boundary edge WITHOUT the face produces a
        // face-less wire clone whose *source* still satisfies the boundary
        // precondition (one incident face). With use_edge_flip_from_face = true
        // and the source face traversing that edge v1-first, the clone edge's
        // endpoint order is the exact REVERSE of the flag-false clone order,
        // while the endpoint *set* is unchanged.
        let edge_order = |flag: bool| -> ([VertexKey; 2], [Vec3; 2]) {
            let mut mesh = LoopMesh::new();
            let (_face, edge) = quad_with_v1_first_edge(&mut mesh);

            let result = mesh.duplicate(&[edge.into()], flag);
            let clone = result.boundary_map.get(&edge).copied().unwrap();
            let [c0, c1] = mesh.edge_vertices(clone);
            ([c0, c1], [mesh.vertex_position(c0), mesh.vertex_position(c1)])
        };

        let (false_keys, false_pos) = edge_order(false);
        let (true_keys, true_pos) = edge_order(true);

        // The flag reverses the clone edge's internal endpoint order.
        assert_eq!(
            true_pos,
            [false_pos[1], false_pos[0]],
            "flag=true clone edge order is the reverse of flag=false (§7.1)"
        );
        // The endpoint *set* (by position) is unchanged — same two verts joined.
        let set = |p: [Vec3; 2]| {
            let mut s = [p[0], p[1]];
            s.sort_by(|x, y| {
                (x.x, x.y, x.z)
                    .partial_cmp(&(y.x, y.y, y.z))
                    .unwrap()
            });
            s
        };
        assert_eq!(set(false_pos), set(true_pos), "endpoint set unchanged");
        // Distinct clone identities either way (the order axis is internal).
        assert_ne!(true_keys, false_keys);
    }

    #[test]
    fn edge_flip_from_face_default_keeps_source_order() {
        // §7 contrast: with the flag OFF, the boundary edge's clone keeps the
        // source edge's stored [v0, v1] order regardless of face traversal.
        let mut mesh = LoopMesh::new();
        let (_face, edge) = quad_with_v1_first_edge(&mut mesh);
        let [v0, v1] = mesh.edge_vertices(edge);
        let src_pos = [mesh.vertex_position(v0), mesh.vertex_position(v1)];

        let result = mesh.duplicate(&[edge.into()], false);
        let clone = result.boundary_map.get(&edge).copied().unwrap();
        let [c0, c1] = mesh.edge_vertices(clone);
        assert_eq!(
            [mesh.vertex_position(c0), mesh.vertex_position(c1)],
            src_pos,
            "flag=false clone keeps source endpoint order"
        );
        mesh.validate_topology();
    }

    #[test]
    fn edge_flip_from_face_no_swap_when_v0_first() {
        // §7: the swap fires ONLY when the source face is v1-first. A standard
        // quad built by create_face_from_verts traverses every perimeter edge
        // v0-first, so even with the flag ON the clone keeps the source order.
        let mut mesh = LoopMesh::new();
        let face = unit_quad(&mut mesh);
        let edge = mesh.face_edge_keys(face)[0];
        let [v0, v1] = mesh.edge_vertices(edge);
        let src_pos = [mesh.vertex_position(v0), mesh.vertex_position(v1)];

        let result = mesh.duplicate(&[edge.into()], true);
        let clone = result.boundary_map.get(&edge).copied().unwrap();
        let [c0, c1] = mesh.edge_vertices(clone);
        assert_eq!(
            [mesh.vertex_position(c0), mesh.vertex_position(c1)],
            src_pos,
            "v0-first source face: no swap even with flag on"
        );
        mesh.validate_topology();
    }

    #[test]
    fn copies_all_cd_homes_verbatim() {
        // §4: verbatim per-element customdata copy across all four CD homes —
        // vert Vec3, edge f32, face Vec3, loop Vec2. Each clone equals its
        // single source byte-for-byte, never interpolated.
        let mut mesh = LoopMesh::new();
        let v_color = mesh
            .register_vertex_attribute(ElementAttributeDesc::numeric(
                AttributeKey::custom("v_color"),
                AttrType::Vec3,
            ))
            .unwrap();
        let e_crease = mesh
            .register_edge_attribute(ElementAttributeDesc::numeric(
                AttributeKey::custom("e_crease"),
                AttrType::F32,
            ))
            .unwrap();
        let f_color = mesh
            .register_face_attribute(ElementAttributeDesc::numeric(
                AttributeKey::custom("f_color"),
                AttrType::Vec3,
            ))
            .unwrap();
        let uv = mesh
            .register_loop_attribute(ElementAttributeDesc::numeric(
                AttributeKey::custom("uv"),
                AttrType::Vec2,
            ))
            .unwrap();

        let face = unit_quad(&mut mesh);

        // Decorate the source.
        for v in mesh.face_vert_keys(face) {
            let p = mesh.vertex_position(v);
            mesh.vertex_attribs_mut()
                .set_vec3(v_color, v.as_usize(), [p.x, p.y, p.x + p.y]);
        }
        for (i, e) in mesh.face_edge_keys(face).into_iter().enumerate() {
            mesh.edge_attribs_mut()
                .set_f32(e_crease, e.as_usize(), 0.25 * (i as f32 + 1.0));
        }
        mesh.face_attribs_mut()
            .set_vec3(f_color, face.as_usize(), [0.7, 0.2, 0.1]);
        for (i, l) in mesh.face_loop_keys(face).into_iter().enumerate() {
            mesh.loop_attribs_mut()
                .set_vec2(uv, l.as_usize(), [i as f32, i as f32 * 2.0]);
        }

        let result = mesh.duplicate(&[face.into()], false);

        // Vert CD: each clone vert equals its source.
        for (&src, &clone) in result.vert_map.iter() {
            // vert_map is bidirectional; only assert the source->clone half.
            if mesh.vertex_position(src) == mesh.vertex_position(clone) {
                let a = mesh
                    .vertex_attribs()
                    .get_vec3(v_color, src.as_usize())
                    .unwrap();
                let b = mesh
                    .vertex_attribs()
                    .get_vec3(v_color, clone.as_usize())
                    .unwrap();
                assert_eq!(a, b, "vert CD copied verbatim");
            }
        }
        // Edge CD.
        for (&src, &clone) in result.edge_map.iter() {
            let a = mesh
                .edge_attribs()
                .get_f32(e_crease, src.as_usize())
                .unwrap();
            let b = mesh
                .edge_attribs()
                .get_f32(e_crease, clone.as_usize())
                .unwrap();
            assert_eq!(a, b, "edge CD copied verbatim");
        }
        // Face CD.
        let clone_face = clone_faces(&result)[0];
        assert_eq!(
            mesh.face_attribs()
                .get_vec3(f_color, face.as_usize())
                .unwrap(),
            mesh.face_attribs()
                .get_vec3(f_color, clone_face.as_usize())
                .unwrap(),
            "face CD copied verbatim"
        );
        // Loop CD: clone loops carry identical UVs in loop order.
        let src_loops = mesh.face_loop_keys(face);
        let clone_loops = mesh.face_loop_keys(clone_face);
        assert_eq!(src_loops.len(), clone_loops.len());
        let src_uvs: Vec<[f32; 2]> = src_loops
            .iter()
            .map(|&l| mesh.loop_attribs().get_vec2(uv, l.as_usize()).unwrap())
            .collect();
        let clone_uvs: Vec<[f32; 2]> = clone_loops
            .iter()
            .map(|&l| mesh.loop_attribs().get_vec2(uv, l.as_usize()).unwrap())
            .collect();
        // Same multiset of loop UVs (loop cycle may start at any corner).
        let mut a = src_uvs.clone();
        let mut b = clone_uvs.clone();
        a.sort_by(|x, y| x.partial_cmp(y).unwrap());
        b.sort_by(|x, y| x.partial_cmp(y).unwrap());
        assert_eq!(a, b, "loop CD copied verbatim in loop order");
        mesh.validate_topology();
    }

    // ---- Cross-mesh `dest` (DUPLICATE.md §8.5) ----

    #[test]
    fn dest_single_face_into_empty_dest() {
        // §8.5.1: naming one face clones its whole closure into `dest`; the
        // source is untouched, `geom_out` is empty cross-mesh, and the maps
        // (forward `source -> clone`) are the clone enumeration.
        let mut src = LoopMesh::new();
        let face = unit_quad(&mut src);
        let mut dest = LoopMesh::new();

        let result = src.duplicate_into(&[face.into()], false, &mut dest);

        // Clones born in dest.
        assert_eq!(dest.vertex_count(), 4);
        assert_eq!(dest.edge_count(), 4);
        assert_eq!(dest.face_count(), 1);
        // Source entirely untouched.
        assert_eq!(
            (src.vertex_count(), src.edge_count(), src.face_count()),
            (4, 4, 1)
        );

        // geom_out empty cross-mesh (§8.5.1); forward-only maps carry the clones.
        assert!(result.geom_out.is_empty(), "geom.out empty cross-mesh");
        assert_eq!(result.vert_map.len(), 4, "forward-only source -> clone");
        assert_eq!(result.edge_map.len(), 4);
        assert_eq!(result.face_map.len(), 1);
        assert_eq!(result.boundary_map.len(), 4, "every perimeter edge frontier");
        assert!(result.isovert_map.is_empty());

        // Each clone vert lands in `dest` at its source's exact position (§2).
        for sv in src.face_vert_keys(face) {
            let cv = result.vert_map.get(&sv).copied().unwrap();
            assert_eq!(dest.vertex_position(cv), src.vertex_position(sv));
        }
        dest.validate_topology();
        src.validate_topology();
    }

    #[test]
    fn dest_cd_layer_layout_remap_copy_default_drop() {
        // §8.5.2: source layout {v_color, v_temp} vs dest layout {v_color,
        // v_extra}. v_color matches both sides -> copied; v_extra is dest-only
        // -> defaulted on the clone; v_temp is source-only -> dropped.
        let mut src = LoopMesh::new();
        let v_color = src
            .register_vertex_attribute(ElementAttributeDesc::numeric(
                AttributeKey::custom("v_color"),
                AttrType::Vec3,
            ))
            .unwrap();
        let v_temp = src
            .register_vertex_attribute(ElementAttributeDesc::numeric(
                AttributeKey::custom("v_temp"),
                AttrType::F32,
            ))
            .unwrap();
        let face = unit_quad(&mut src);
        for (i, v) in src.face_vert_keys(face).into_iter().enumerate() {
            let f = i as f32;
            src.vertex_attribs_mut()
                .set_vec3(v_color, v.as_usize(), [f, f + 1.0, f + 2.0]);
            src.vertex_attribs_mut().set_f32(v_temp, v.as_usize(), 9.0);
        }

        let mut dest = LoopMesh::new();
        let d_color = dest
            .register_vertex_attribute(ElementAttributeDesc::numeric(
                AttributeKey::custom("v_color"),
                AttrType::Vec3,
            ))
            .unwrap();
        let d_extra = dest
            .register_vertex_attribute(ElementAttributeDesc::numeric(
                AttributeKey::custom("v_extra"),
                AttrType::Vec3,
            ))
            .unwrap();

        let result = src.duplicate_into(&[face.into()], false, &mut dest);

        assert_eq!(result.vert_map.len(), 4);
        // v_temp has no dest slot at all -> dropped.
        assert!(
            dest.vertex_attribs()
                .find(&AttributeKey::custom("v_temp"))
                .is_none(),
            "source-only layer dropped"
        );
        for (&sv, &cv) in result.vert_map.iter() {
            let want = src.vertex_attribs().get_vec3(v_color, sv.as_usize()).unwrap();
            let got = dest.vertex_attribs().get_vec3(d_color, cv.as_usize()).unwrap();
            assert_eq!(want, got, "matched layer copied source -> clone");
            assert_eq!(
                dest.vertex_attribs().get_vec3(d_extra, cv.as_usize()).unwrap(),
                [0.0; 3],
                "dest-only layer defaulted on the clone"
            );
        }
        let _ = v_temp;
        dest.validate_topology();
    }

    #[test]
    fn dest_matching_layout_copies_all_cd_homes_including_loops() {
        // §8.5.2 degenerate (identity) remap: matching layouts on both meshes
        // across all four CD homes -> every clone element equals its source,
        // exercising the per-loop remap path too.
        fn register(m: &mut LoopMesh) -> (
            ElementAttributeKey,
            ElementAttributeKey,
            ElementAttributeKey,
            ElementAttributeKey,
        ) {
            let vc = m
                .register_vertex_attribute(ElementAttributeDesc::numeric(
                    AttributeKey::custom("v_color"),
                    AttrType::Vec3,
                ))
                .unwrap();
            let ec = m
                .register_edge_attribute(ElementAttributeDesc::numeric(
                    AttributeKey::custom("e_crease"),
                    AttrType::F32,
                ))
                .unwrap();
            let fc = m
                .register_face_attribute(ElementAttributeDesc::numeric(
                    AttributeKey::custom("f_color"),
                    AttrType::Vec3,
                ))
                .unwrap();
            let uv = m
                .register_loop_attribute(ElementAttributeDesc::numeric(
                    AttributeKey::custom("uv"),
                    AttrType::Vec2,
                ))
                .unwrap();
            (vc, ec, fc, uv)
        }

        let mut src = LoopMesh::new();
        let (vc, ec, fc, uv) = register(&mut src);
        let face = unit_quad(&mut src);
        for v in src.face_vert_keys(face) {
            let p = src.vertex_position(v);
            src.vertex_attribs_mut()
                .set_vec3(vc, v.as_usize(), [p.x, p.y, 1.0]);
        }
        for (i, e) in src.face_edge_keys(face).into_iter().enumerate() {
            src.edge_attribs_mut()
                .set_f32(ec, e.as_usize(), 0.5 * (i as f32 + 1.0));
        }
        src.face_attribs_mut()
            .set_vec3(fc, face.as_usize(), [0.3, 0.4, 0.5]);
        for (i, l) in src.face_loop_keys(face).into_iter().enumerate() {
            src.loop_attribs_mut()
                .set_vec2(uv, l.as_usize(), [i as f32, i as f32 * 3.0]);
        }

        let mut dest = LoopMesh::new();
        let (dvc, dec, dfc, duv) = register(&mut dest);

        let result = src.duplicate_into(&[face.into()], false, &mut dest);

        for (&sv, &cv) in result.vert_map.iter() {
            assert_eq!(
                src.vertex_attribs().get_vec3(vc, sv.as_usize()),
                dest.vertex_attribs().get_vec3(dvc, cv.as_usize()),
                "vert CD remapped (identity)"
            );
        }
        for (&se, &ce) in result.edge_map.iter() {
            assert_eq!(
                src.edge_attribs().get_f32(ec, se.as_usize()),
                dest.edge_attribs().get_f32(dec, ce.as_usize()),
                "edge CD remapped (identity)"
            );
        }
        let clone_face = *result.face_map.values().next().unwrap();
        assert_eq!(
            src.face_attribs().get_vec3(fc, face.as_usize()),
            dest.face_attribs().get_vec3(dfc, clone_face.as_usize()),
            "face CD remapped (identity)"
        );
        let mut src_uv: Vec<[f32; 2]> = src
            .face_loop_keys(face)
            .iter()
            .map(|&l| src.loop_attribs().get_vec2(uv, l.as_usize()).unwrap())
            .collect();
        let mut clone_uv: Vec<[f32; 2]> = dest
            .face_loop_keys(clone_face)
            .iter()
            .map(|&l| dest.loop_attribs().get_vec2(duv, l.as_usize()).unwrap())
            .collect();
        src_uv.sort_by(|x, y| x.partial_cmp(y).unwrap());
        clone_uv.sort_by(|x, y| x.partial_cmp(y).unwrap());
        assert_eq!(src_uv, clone_uv, "loop CD remapped in loop order");
        dest.validate_topology();
    }

    #[test]
    fn dest_into_populated_dest_is_additive() {
        // §8.5.3: cloning into a dest that already holds a triangle appends the
        // clone disjointly; the triangle and the source are left untouched and
        // no clone aliases a pre-existing dest element.
        let mut src = LoopMesh::new();
        let face = unit_quad(&mut src);

        let mut dest = LoopMesh::new();
        let tri = create_face_from_verts(
            &mut dest,
            &[
                Vec3::new(10.0, 0.0, 0.0),
                Vec3::new(11.0, 0.0, 0.0),
                Vec3::new(10.0, 1.0, 0.0),
            ],
        );
        let tri_verts: HashSet<VertexKey> = dest.face_vert_keys(tri).into_iter().collect();

        let result = src.duplicate_into(&[face.into()], false, &mut dest);

        // Additive: triangle (3V/3E/1F) + quad clone (4V/4E/1F).
        assert_eq!(dest.vertex_count(), 7);
        assert_eq!(dest.edge_count(), 7);
        assert_eq!(dest.face_count(), 2);

        // No clone aliases a pre-existing dest vert.
        for cv in result.vert_map.values() {
            assert!(
                !tri_verts.contains(cv),
                "clone never aliases a pre-existing dest element"
            );
        }
        // The pre-existing triangle is intact.
        assert_eq!(dest.face_vert_keys(tri).len(), 3);
        // Source untouched.
        assert_eq!(
            (src.vertex_count(), src.edge_count(), src.face_count()),
            (4, 4, 1)
        );
        dest.validate_topology();
        src.validate_topology();
    }

    #[test]
    fn dest_only_layer_defaults_over_recycled_slot_stale_value() {
        // §8.5.2/§8.5.3: a destination-only layer must be written to its type
        // default on the clone, *not* left to the freshly-created slot's
        // contents. When the clone is minted into a recycled slab slot, that
        // slot still carries the stale value its previous occupant left (slab
        // removal never resets the attribute buckets, and create's grow is a
        // resize no-op for a recycled key), so the default must be stamped on.
        let mut src = LoopMesh::new();
        let source_vert = src.create_vertex(Vec3::new(2.0, 3.0, 4.0));

        let mut dest = LoopMesh::new();
        // A dest-only layer the source has no `(type, name)` match for.
        let scratch = dest
            .register_vertex_attribute(ElementAttributeDesc::numeric(
                AttributeKey::custom("scratch"),
                AttrType::I32,
            ))
            .unwrap();

        // Occupy the lowest slab slot, poison its `scratch` bucket entry, then
        // free the slot so the next `create_vertex` recycles it (slab hands out
        // the lowest vacant key).
        let throwaway = dest.create_vertex(Vec3::ZERO);
        dest.vertex_attribs_mut()
            .set_i32(scratch, throwaway.as_usize(), 999);
        dest.remove_vertex_slab_entry(throwaway);

        let result = src.duplicate_into(&[source_vert.into()], false, &mut dest);

        let clone = result.vert_map.get(&source_vert).copied().unwrap();
        // The clone reused the recycled slot (the only freed key in `dest`).
        assert_eq!(
            clone.as_usize(),
            throwaway.as_usize(),
            "clone minted into the recycled slot",
        );
        // Old skip-the-layer behaviour would leak 999; the explicit default
        // branch stamps the type default.
        assert_eq!(
            dest.vertex_attribs().get_i32(scratch, clone.as_usize()),
            Some(0),
            "dest-only layer defaulted on the clone, not left stale",
        );
        dest.validate_topology();
    }
}
