use smallvec::SmallVec;

use crate::loopmesh::*;

impl LoopMesh {
    /// Extrude a set of selected edges with no face input (edge-only extrude).
    ///
    /// Each input edge and its endpoints are duplicated; one "wall" quad is then
    /// built per input edge, spanning the original edge to its duplicate. Where
    /// two contiguous input edges share an endpoint, the vertical edge lifted at
    /// that shared endpoint is created once and reused by both walls (the §4.5
    /// vertical-edge-sharing rule), so a contiguous strip of input edges lifts as
    /// a continuous ribbon rather than a stack of independent quads.
    ///
    /// Unlike region / discrete extrude there is **no deletion phase**: the
    /// original edges are kept, because they may still be attached to non-input
    /// faces in the surround.
    ///
    /// This is a pure-topology op; the caller displaces the new vertices after it
    /// returns.
    ///
    /// Customdata: every duplicate vertex copies its source vertex's attribs;
    /// every duplicate edge copies its source edge's attribs. Each wall's four
    /// corner loops follow the unified §4.3 column rule, sourcing per-corner loop
    /// customdata from the *other radial loop across the original edge* — i.e.
    /// the surround face's loop when the input edge has a face on it. For a
    /// genuinely faceless wire input edge there is no surround loop at all, so the
    /// wall's loop customdata is default-initialised (§10 final paragraph). The
    /// newly-created vertical edges take default attribute values.
    ///
    /// See `radial-mesh-guides/operators/EXTRUDE.md` §10 for the full algorithm
    /// and parity contract.
    pub fn extrude_edge_only(&mut self, edges: &[EdgeKey]) -> ExtrudeResult {
        self.extrude_edge_only_ex(edges, false)
    }

    /// [`extrude_edge_only`](Self::extrude_edge_only) with the `use_normal_flip`
    /// parameter exposed: when set, each wall's winding is reversed
    /// (EXTRUDE.md §4.2 / §5). The default (`false`) winds each wall so its loop
    /// on the original edge runs in the original-edge direction implied by the
    /// edge's first surround loop; flipping reverses that choice, so the wall's
    /// outward normal points the other way. Topology (counts, incidence) is
    /// identical either way — only the per-wall loop direction differs.
    pub fn extrude_edge_only_ex(
        &mut self,
        edges: &[EdgeKey],
        use_normal_flip: bool,
    ) -> ExtrudeResult {
        let mut ledger = ExtrudeVertDupLedger::new(self.vertices.key_watermark());
        self.extrude_edge_only_seeded(edges, use_normal_flip, &mut ledger)
    }

    /// Edge-only extrude that shares its vertex-duplication and vertical-edge
    /// bookkeeping with other extrude passes through `ledger` (EXTRUDE.md §8.4).
    ///
    /// When a mixed-input extrude routes a directly-named wire edge whose
    /// endpoint is *also* a corner of an input face, that corner must be
    /// duplicated exactly once across the whole operation, and the face
    /// region's wall and this edge's wall must meet at the *same* duplicate
    /// corner, sharing one vertical edge. The region pass seeds `ledger` with
    /// the duplicate it minted for that corner and the vertical edge it lifted
    /// there; this pass then reuses both instead of minting parallel ones —
    /// the §4.5 "does an edge already exist between `v_old` and `v_dup`?" reuse
    /// rule, extended across passes. A vertex not present in the seed is
    /// duplicated here and recorded for any later pass.
    pub(crate) fn extrude_edge_only_seeded(
        &mut self,
        edges: &[EdgeKey],
        use_normal_flip: bool,
        ledger: &mut ExtrudeVertDupLedger,
    ) -> ExtrudeResult {
        self.extrude_edge_only_seeded_exclude(edges, &[], use_normal_flip, ledger)
    }

    /// [`extrude_edge_only_seeded`](Self::extrude_edge_only_seeded) honouring an
    /// `edges_exclude` set (EXTRUDE.md §4.6 / §8.3).
    ///
    /// Each input edge is still duplicated (verts and edge), so a shared corner
    /// stays consistent across passes through `ledger`. The exclusion only
    /// changes what happens at *wall* time:
    ///
    /// - An **excluded wire** edge (no face on either side) takes the §4.6
    ///   *before wall construction* kill: its freshly-created duplicate edge is
    ///   killed outright, and any duplicate endpoint vert the kill leaves with
    ///   no other incident edge is killed too. No wall. The original wire pair
    ///   is untouched. This is the §8.3 wire-boundary kill.
    /// - An **excluded face-bounded** edge keeps its duplicate edge/vert pair
    ///   but builds no wall (§4.6 second bullet).
    /// - A non-excluded edge behaves exactly as the plain seeded pass.
    pub(crate) fn extrude_edge_only_seeded_exclude(
        &mut self,
        edges: &[EdgeKey],
        edges_exclude: &[EdgeKey],
        use_normal_flip: bool,
        ledger: &mut ExtrudeVertDupLedger,
    ) -> ExtrudeResult {
        let do_loop_cd = !self.loop_attribs.is_empty();

        let vert_watermark = self.vertices.key_watermark();
        let edge_watermark = self.edges.key_watermark();

        let mut edge_excluded: Vec<bool> = vec![false; edge_watermark];
        for &e in edges_exclude {
            let slot = e.as_usize();
            if slot < edge_watermark {
                edge_excluded[slot] = true;
            }
        }

        // ---- Phase 1: tag the input edges and their endpoints. Dedup verts and
        //               edges via slab-indexed presence flags. ----
        let mut vert_in_input: Vec<bool> = vec![false; vert_watermark];
        let mut edge_in_input: Vec<bool> = vec![false; edge_watermark];
        for &e in edges {
            if edge_in_input[e.as_usize()] {
                continue;
            }
            edge_in_input[e.as_usize()] = true;
            let [v0, v1] = self.edge_vertices(e);
            vert_in_input[v0.as_usize()] = true;
            vert_in_input[v1.as_usize()] = true;
        }

        // ---- Phase 2: snapshot, per input edge, the source loop across it for
        //               wall customdata and the winding cue. There are no input
        //               faces, so the "other radial loop across e_old" is simply
        //               the edge's first radial loop (a surround face). A faceless
        //               wire edge has none — its wall CD defaults (§10). ----
        struct EdgeSource {
            // The two surround corner-loops on the original edge, keyed by the
            // endpoint vertex each sits at. Empty for a faceless wire edge.
            source_loop_at: SmallVec<[(VertexKey, LoopKey); 2]>,
            // Winding cue: true when the edge's first surround loop runs from the
            // edge's *second* endpoint, i.e. against the edge's stored v0->v1
            // order. Faceless edges fall back to an arbitrary-but-consistent cue.
            edge_normal_flip: bool,
        }
        let mut edge_source: Vec<Option<EdgeSource>> = (0..edge_watermark).map(|_| None).collect();
        for slot in 0..edge_watermark {
            if !edge_in_input[slot] {
                continue;
            }
            let e = EdgeKey::from_usize(slot);
            let [v0, _v1] = self.edge_vertices(e);
            let first_loop = self.edges.get(e).radial_link;
            let (source_loop_at, edge_normal_flip) = match first_loop {
                Some(l) => {
                    let lref = self.loops.get(l);
                    let v_here = lref.vertex;
                    let l_next = lref.face_loop_links.next;
                    let v_next = self.loops.get(l_next).vertex;
                    let mut src: SmallVec<[(VertexKey, LoopKey); 2]> = SmallVec::new();
                    src.push((v_here, l));
                    src.push((v_next, l_next));
                    // §5: orient as the surround face's loop on the edge. The loop
                    // anchored at the edge's second endpoint means the surround
                    // face traverses the edge against its stored order.
                    (src, v_here != v0)
                }
                None => (SmallVec::new(), false),
            };
            edge_source[slot] = Some(EdgeSource {
                source_loop_at,
                edge_normal_flip,
            });
        }

        // ---- Phase 3: duplicate verts and edges. A vertex already duplicated
        //               by another pass (its corner shared with an input face,
        //               §8.4) reuses that duplicate rather than minting a fresh
        //               one. ----
        let mut vert_to_dup: Vec<VertexKey> =
            vec![VertexKey::from_usize(usize::MAX); vert_watermark];
        for slot in 0..vert_watermark {
            if !vert_in_input[slot] {
                continue;
            }
            let v_old = VertexKey::from_usize(slot);
            let dup = match ledger.dup_of(v_old) {
                Some(existing) => existing,
                None => {
                    let dup = self.create_vertex(self.vertex_position(v_old));
                    self.vertex_attribs.copy_into(dup.as_usize(), slot);
                    ledger.record_dup(v_old, dup);
                    dup
                }
            };
            vert_to_dup[slot] = dup;
        }
        let mut edge_to_dup: Vec<EdgeKey> = vec![EdgeKey::NULL; edge_watermark];
        for slot in 0..edge_watermark {
            if !edge_in_input[slot] {
                continue;
            }
            let e_old = EdgeKey::from_usize(slot);
            let [v0, v1] = self.edge_vertices(e_old);
            let dup = self.create_edge(
                [vert_to_dup[v0.as_usize()], vert_to_dup[v1.as_usize()]],
                EdgeCreateFlags::NONE,
            );
            edge_to_dup[slot] = dup;
            self.edge_attribs.copy_into(dup.as_usize(), slot);
        }

        // ---- Phase 4: build one wall quad per input edge. Vertical edges are
        //               lazily created and shared across walls at the same source
        //               vertex (§4.5) — and, through the ledger, shared with the
        //               face region pass at a cross-kind corner (§8.4). ----
        let mut wall_faces: Vec<FaceKey> = Vec::new();
        // Side edges this pass freshly created (a reused, ledger-seeded edge is
        // owned by the pass that minted it, so it is not reported again here).
        let mut side_edges_out: Vec<EdgeKey> = Vec::new();
        // §8.3 wire-boundary kill drops some just-minted duplicates; their slots
        // are excluded from the output maps so callers never see a dead element.
        let mut edge_dup_killed: Vec<bool> = vec![false; edge_watermark];
        let mut vert_dup_killed: Vec<bool> = vec![false; vert_watermark];

        for slot in 0..edge_watermark {
            let source = match &edge_source[slot] {
                Some(s) => s,
                None => continue,
            };
            let e_old = EdgeKey::from_usize(slot);
            let [v0_old, v1_old] = self.edge_vertices(e_old);
            let v0_new = vert_to_dup[v0_old.as_usize()];
            let v1_new = vert_to_dup[v1_old.as_usize()];
            let e_new = edge_to_dup[slot];

            // §4.6 / §8.3: an excluded edge builds no wall. A non-wire excluded
            // edge keeps its duplicate edge/verts; a wire excluded edge (no
            // surround loop) would otherwise leave its duplicate as a stranded
            // wire, so the *before wall construction* kill removes the duplicate
            // edge outright, plus any duplicate endpoint the kill leaves with an
            // empty disk. The original wire pair is untouched. These killed
            // duplicates are not recorded in the output maps below.
            if edge_excluded[slot] {
                if source.source_loop_at.is_empty() {
                    self.edge_kill(e_new);
                    edge_dup_killed[slot] = true;
                    for (endpoint_old, endpoint_dup) in [(v0_old, v0_new), (v1_old, v1_new)] {
                        if self.vertex_disk_iter(endpoint_dup).next().is_none() {
                            self.vertex_kill(endpoint_dup);
                            vert_dup_killed[endpoint_old.as_usize()] = true;
                        }
                    }
                }
                continue;
            }

            // §5 vert order. `a_is_v0` decides which endpoint leads the wall's
            // loop on the original edge; `use_normal_flip` swaps the two
            // endpoints, reversing the quad cycle and the wall's outward normal.
            let a_is_v0 = source.edge_normal_flip != use_normal_flip;
            let (v_a_old, v_b_old, v_a_new, v_b_new) = if a_is_v0 {
                (v0_old, v1_old, v0_new, v1_new)
            } else {
                (v1_old, v0_old, v1_new, v0_new)
            };

            let side_a = match ledger.side_edge_of(v_a_old) {
                Some(existing) => existing,
                None => {
                    let e = self.create_edge([v_a_old, v_a_new], EdgeCreateFlags::NONE);
                    ledger.record_side_edge(v_a_old, e);
                    side_edges_out.push(e);
                    e
                }
            };
            let side_b = match ledger.side_edge_of(v_b_old) {
                Some(existing) => existing,
                None => {
                    let e = self.create_edge([v_b_old, v_b_new], EdgeCreateFlags::NONE);
                    ledger.record_side_edge(v_b_old, e);
                    side_edges_out.push(e);
                    e
                }
            };

            // Wall quad: v_a_old -> v_b_old -> v_b_new -> v_a_new (outward winding).
            let wall = self.face_create(
                &[v_a_old, v_b_old, v_b_new, v_a_new],
                &[e_old, side_b, e_new, side_a],
                FaceCreateFlags::NONE,
            );

            // Wall loop customdata flows continuously across the wall from the
            // surround face (§4.3): each of the four corners copies from the
            // source loop sitting at the corner's *original* endpoint, so the
            // duplicate-side corners take the same source as the original-side
            // corners they sit above. For a faceless wire edge `source_loop_at`
            // is empty and the wall loops keep their default-initialised CD.
            if do_loop_cd && !source.source_loop_at.is_empty() {
                let wall_source_at = |v: VertexKey| -> Option<LoopKey> {
                    source
                        .source_loop_at
                        .iter()
                        .find(|(sv, _)| *sv == v)
                        .map(|(_, l)| *l)
                };
                let wall_loop_a_old = self.faces.get(wall).first_edge_loop;
                let wall_loop_b_old = self.loops.get(wall_loop_a_old).face_loop_links.next;
                let wall_loop_b_new = self.loops.get(wall_loop_b_old).face_loop_links.next;
                let wall_loop_a_new = self.loops.get(wall_loop_b_new).face_loop_links.next;
                for (wall_loop, origin_vert) in [
                    (wall_loop_a_old, v_a_old),
                    (wall_loop_b_old, v_b_old),
                    (wall_loop_b_new, v_b_old),
                    (wall_loop_a_new, v_a_old),
                ] {
                    if let Some(src_loop) = wall_source_at(origin_vert) {
                        self.loop_attribs
                            .copy_into(wall_loop.as_usize(), src_loop.as_usize());
                    }
                }
            }

            wall_faces.push(wall);
        }

        // Build the output maps in dense-index order.
        let mut vert_map_out: Vec<(VertexKey, VertexKey)> = Vec::new();
        for slot in 0..vert_watermark {
            if vert_in_input[slot] && !vert_dup_killed[slot] {
                vert_map_out.push((VertexKey::from_usize(slot), vert_to_dup[slot]));
            }
        }
        let mut edge_map_out: Vec<(EdgeKey, EdgeKey)> = Vec::new();
        for slot in 0..edge_watermark {
            if edge_in_input[slot] && !edge_dup_killed[slot] {
                edge_map_out.push((EdgeKey::from_usize(slot), edge_to_dup[slot]));
            }
        }
        ExtrudeResult {
            vert_map: vert_map_out,
            edge_map: edge_map_out,
            side_edges: side_edges_out,
            wall_faces,
            cap_faces: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use glam::Vec3;

    use crate::loopmesh::test_support::create_face_from_verts;
    use crate::loopmesh::*;

    fn pos_key(p: Vec3) -> [u32; 3] {
        [p.x.to_bits(), p.y.to_bits(), p.z.to_bits()]
    }

    // Two N-cycles describe the same polygon with opposite winding iff one is
    // the reverse rotation of the other.
    fn cycles_are_reversed(a: &[Vec3], b: &[Vec3]) -> bool {
        let n = a.len();
        if b.len() != n {
            return false;
        }
        let rev: Vec<Vec3> = b.iter().rev().copied().collect();
        for shift in 0..n {
            if (0..n).all(|i| a[i] == rev[(i + shift) % n]) {
                return true;
            }
        }
        false
    }

    fn unit_quad(mesh: &mut LoopMesh) -> FaceKey {
        let verts = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        ];
        create_face_from_verts(mesh, &verts)
    }

    #[test]
    fn edge_only_single_perimeter_edge_one_wall() {
        // A lone quad. Extruding one of its perimeter edges duplicates that
        // edge's 2 verts + 1 edge and builds exactly one wall quad spanning the
        // original edge and its duplicate. The original quad is untouched (no
        // delete phase) and stays on the original edge.
        let mut mesh = LoopMesh::new();
        let face = unit_quad(&mut mesh);
        // Pick the perimeter edge between the first two loop verts.
        let face_verts = mesh.face_vert_keys(face);
        let (va, vb) = (face_verts[0], face_verts[1]);
        let edge = mesh
            .face_edge_keys(face)
            .into_iter()
            .find(|&e| {
                let [x, y] = mesh.edge_vertices(e);
                (x == va && y == vb) || (x == vb && y == va)
            })
            .expect("perimeter edge exists");

        let result = mesh.extrude_edge_only(&[edge]);

        assert_eq!(result.vert_map.len(), 2, "2 endpoints duplicated");
        assert_eq!(result.edge_map.len(), 1, "1 edge duplicated");
        assert_eq!(result.side_edges.len(), 2, "2 vertical edges");
        assert_eq!(result.wall_faces.len(), 1, "one wall per input edge");

        // 4 original verts + 2 duplicates.
        assert_eq!(mesh.vertex_count(), 6);
        // original quad + 1 wall.
        assert_eq!(mesh.face_count(), 2);
        // The original face survives (no delete phase).
        assert!(mesh.face_keys().contains(&face), "original face is kept");

        // The single wall references both original endpoints.
        let wall = result.wall_faces[0];
        let wall_verts: HashSet<VertexKey> = mesh.face_vert_keys(wall).into_iter().collect();
        assert!(wall_verts.contains(&va) && wall_verts.contains(&vb));
        assert_eq!(mesh.face_vert_keys(wall).len(), 4, "wall is a quad");
        mesh.validate_topology();
    }

    #[test]
    fn edge_only_contiguous_strip_shares_vertical_edge() {
        // The §4.5 load-bearing case. A path of two edges v0–v1–v2 sharing the
        // middle vertex v1. Edge-only extrude lifts both, and the vertical edge
        // at the shared endpoint v1 is created once and reused by both walls —
        // so the two walls form a continuous ribbon rather than two stacked
        // quads with paired-but-distinct verticals.
        let mut mesh = LoopMesh::new();
        let v0 = mesh.create_vertex(Vec3::new(0.0, 0.0, 0.0));
        let v1 = mesh.create_vertex(Vec3::new(1.0, 0.0, 0.0));
        let v2 = mesh.create_vertex(Vec3::new(2.0, 0.0, 0.0));
        let e0 = mesh.create_edge([v0, v1], EdgeCreateFlags::NONE);
        let e1 = mesh.create_edge([v1, v2], EdgeCreateFlags::NONE);

        let result = mesh.extrude_edge_only(&[e0, e1]);

        // 3 unique endpoints duplicated; 2 edges duplicated.
        assert_eq!(result.vert_map.len(), 3, "v0,v1,v2 deduped to 3");
        assert_eq!(result.edge_map.len(), 2);
        // Vertical edges: one per unique source vert, so 3 — NOT 4. The shared
        // endpoint v1 contributes a single vertical edge to the ribbon.
        assert_eq!(
            result.side_edges.len(),
            3,
            "shared endpoint v1 contributes a single shared vertical edge"
        );
        assert_eq!(result.wall_faces.len(), 2, "one wall per input edge");

        // Confirm both walls reference the *same* vertical edge at v1: the edge
        // between v1 and its duplicate must appear in both wall perimeters.
        let v1_dup = result
            .vert_map
            .iter()
            .find(|(s, _)| *s == v1)
            .map(|(_, d)| *d)
            .unwrap();
        let shared_vertical = mesh
            .vert_edge_keys(v1)
            .into_iter()
            .find(|&e| {
                let [x, y] = mesh.edge_vertices(e);
                (x == v1 && y == v1_dup) || (x == v1_dup && y == v1)
            })
            .expect("vertical edge at shared endpoint exists");
        let mut walls_using_shared_vertical = 0;
        for &wf in &result.wall_faces {
            if mesh.face_edge_keys(wf).contains(&shared_vertical) {
                walls_using_shared_vertical += 1;
            }
        }
        assert_eq!(
            walls_using_shared_vertical, 2,
            "both walls reuse the single vertical edge at the shared endpoint"
        );
        mesh.validate_topology();
    }

    #[test]
    fn edge_only_faceless_wire_edge_default_init_wall() {
        // §10 final paragraph: a genuinely faceless wire edge has no surround
        // loop, so the wall is still built but its loop customdata is
        // default-initialised. A loop UV layer is registered; the wall's four
        // corners must read the layer default (zero), not garbage.
        let mut mesh = LoopMesh::new();
        let v0 = mesh.create_vertex(Vec3::new(0.0, 0.0, 0.0));
        let v1 = mesh.create_vertex(Vec3::new(1.0, 0.0, 0.0));
        let wire = mesh.create_edge([v0, v1], EdgeCreateFlags::NONE);

        let uv = mesh
            .register_loop_attribute(ElementAttributeDesc::numeric(
                AttributeKey::custom("uv"),
                AttrType::Vec2,
            ))
            .unwrap();

        let result = mesh.extrude_edge_only(&[wire]);

        assert_eq!(result.vert_map.len(), 2);
        assert_eq!(result.edge_map.len(), 1);
        assert_eq!(result.wall_faces.len(), 1, "wire edge still gets a wall");
        let wall = result.wall_faces[0];
        assert_eq!(mesh.face_vert_keys(wall).len(), 4, "wall is a quad");

        // All four corner loops carry the layer default (zero) — no surround
        // loop existed to source from.
        for l in mesh.iter_face_loops(wall).collect::<Vec<_>>() {
            let got = mesh.loop_attribs().get_vec2(uv, l.as_usize()).unwrap();
            assert_eq!(
                got,
                [0.0, 0.0],
                "faceless wall loop CD is default-initialised"
            );
        }
        mesh.validate_topology();
    }

    #[test]
    fn edge_only_normal_flip_reverses_wall_winding() {
        // use_normal_flip must reverse the wall's vertex cycle relative to the
        // default. Extrude the same perimeter edge twice (default vs flipped)
        // and assert the wall perimeter is the reverse cycle. CanonicalMesh is
        // winding-insensitive, so this in-crate cycle check is what truly pins
        // the flag.
        let edge_of = |mesh: &LoopMesh, face: FaceKey| -> EdgeKey {
            let fv = mesh.face_vert_keys(face);
            let (va, vb) = (fv[0], fv[1]);
            mesh.face_edge_keys(face)
                .into_iter()
                .find(|&e| {
                    let [x, y] = mesh.edge_vertices(e);
                    (x == va && y == vb) || (x == vb && y == va)
                })
                .unwrap()
        };

        let mut mesh_default = LoopMesh::new();
        let face_default = unit_quad(&mut mesh_default);
        let e_default = edge_of(&mesh_default, face_default);
        let res_default = mesh_default.extrude_edge_only(&[e_default]);

        let mut mesh_flip = LoopMesh::new();
        let face_flip = unit_quad(&mut mesh_flip);
        let e_flip = edge_of(&mesh_flip, face_flip);
        let res_flip = mesh_flip.extrude_edge_only_ex(&[e_flip], true);

        assert_eq!(res_default.wall_faces.len(), 1);
        assert_eq!(res_flip.wall_faces.len(), 1);

        let def_pos: Vec<Vec3> = mesh_default
            .face_vert_keys(res_default.wall_faces[0])
            .iter()
            .map(|&v| mesh_default.vertex_position(v))
            .collect();
        let flip_pos: Vec<Vec3> = mesh_flip
            .face_vert_keys(res_flip.wall_faces[0])
            .iter()
            .map(|&v| mesh_flip.vertex_position(v))
            .collect();

        assert!(
            cycles_are_reversed(&def_pos, &flip_pos),
            "flip wall must be the reverse cycle of the default wall"
        );
        // Sanity: positions identical between the two meshes (same fixture).
        let def_set: HashSet<[u32; 3]> = def_pos.iter().map(|&p| pos_key(p)).collect();
        let flip_set: HashSet<[u32; 3]> = flip_pos.iter().map(|&p| pos_key(p)).collect();
        assert_eq!(def_set, flip_set, "same four corner positions");

        mesh_default.validate_topology();
        mesh_flip.validate_topology();
    }

    #[test]
    fn edge_only_default_matches_ex_false() {
        // The convenience form must delegate to _ex(.., false).
        let mut mesh_a = LoopMesh::new();
        let v0a = mesh_a.create_vertex(Vec3::new(0.0, 0.0, 0.0));
        let v1a = mesh_a.create_vertex(Vec3::new(1.0, 0.0, 0.0));
        let ea = mesh_a.create_edge([v0a, v1a], EdgeCreateFlags::NONE);
        let res_a = mesh_a.extrude_edge_only(&[ea]);

        let mut mesh_b = LoopMesh::new();
        let v0b = mesh_b.create_vertex(Vec3::new(0.0, 0.0, 0.0));
        let v1b = mesh_b.create_vertex(Vec3::new(1.0, 0.0, 0.0));
        let eb = mesh_b.create_edge([v0b, v1b], EdgeCreateFlags::NONE);
        let res_b = mesh_b.extrude_edge_only_ex(&[eb], false);

        let a_pos: Vec<Vec3> = mesh_a
            .face_vert_keys(res_a.wall_faces[0])
            .iter()
            .map(|&v| mesh_a.vertex_position(v))
            .collect();
        let b_pos: Vec<Vec3> = mesh_b
            .face_vert_keys(res_b.wall_faces[0])
            .iter()
            .map(|&v| mesh_b.vertex_position(v))
            .collect();
        let n = a_pos.len();
        let same_rotation = (0..n).any(|shift| (0..n).all(|i| a_pos[i] == b_pos[(i + shift) % n]));
        assert!(
            same_rotation,
            "extrude_edge_only must equal _ex(false) winding"
        );
    }
}
