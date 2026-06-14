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
    /// `use_edge_flip_from_face` is accepted for interface stability but its
    /// non-default (`true`) behaviour — the §7 boundary-edge endpoint swap for
    /// the duplicate-then-extrude composition — is a by-design deferral and is
    /// not yet implemented; only the default (`false`) path runs.
    ///
    /// See `radial-mesh-guides/operators/DUPLICATE.md` for the full algorithm
    /// and parity contract.
    pub fn duplicate(
        &mut self,
        geom: &[GeomKey],
        use_edge_flip_from_face: bool,
    ) -> DuplicateResult {
        debug_assert!(
            !use_edge_flip_from_face,
            "duplicate: use_edge_flip_from_face=true is a by-design deferral (DUPLICATE.md §7)"
        );
        let _ = use_edge_flip_from_face;

        let mut result = DuplicateResult::default();

        // Selection membership, keyed by source slot, so the §6 classification
        // can ask "is this incident element itself in the selection?".
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
        result: &mut DuplicateResult,
    ) -> EdgeKey {
        let [v0, v1] = self.edge_vertices(source);
        let c0 = self.closure_vertex(v0, vert_ledger, result);
        let c1 = self.closure_vertex(v1, vert_ledger, result);

        let clone = self.create_edge([c0, c1], EdgeCreateFlags::NONE);
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
                None => self.clone_edge(e, vert_ledger, edge_ledger, face_selected, result),
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
        assert_eq!(mesh.totvert(), 8, "4 originals + 4 clones");
        assert_eq!(mesh.totedge(), 8);
        assert_eq!(mesh.totface(), 2);
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

        assert_eq!(mesh.totvert(), 2, "original + clone");
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

        assert_eq!(mesh.totvert(), 4, "2 originals + 2 closed endpoints");
        assert_eq!(mesh.totedge(), 2);
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
}
