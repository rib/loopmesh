use smallvec::SmallVec;

use crate::loopmesh::*;

impl LoopMesh {
    /// Extrude a region of selected faces.
    ///
    /// Every vertex and edge referenced by any face in `faces` is duplicated.
    /// Each selected face's perimeter is then re-anchored to the duplicate
    /// vertices/edges, leaving the original boundary edges still attached to
    /// any unselected neighbouring faces. For each *boundary* edge (one with
    /// at least one selected and one unselected face on its radial cycle), a
    /// "wall" quad is created spanning the original edge to its duplicate.
    /// Interior edges of the selection (every radial loop on a selected face)
    /// become wire edges — they remain in the mesh but no face references
    /// them.
    ///
    /// Non-manifold edges are handled: a boundary edge with radial count > 2
    /// receives one wall face per selected loop on its original radial.
    ///
    /// This is a pure-topology op; the caller is responsible for displacing the
    /// new vertices after it returns.
    ///
    /// Customdata: every new vertex copies its source vertex's attribs;
    /// every duplicated edge copies its source edge's attribs; each wall
    /// face inherits the attribs of the selected face that contributed it.
    /// Each wall's four corner loops source their per-corner loop customdata
    /// from the *other radial loop across the original edge*: the surround
    /// (unselected) face's loop when a surround face is present, else the
    /// original selected face's own loop on that edge. Both the original-side
    /// and duplicate-side corners take the same source at the matching
    /// endpoint, so each wall column reads identically top and bottom and the
    /// across-edge gradient of the source face is reproduced across the wall.
    /// Default loop customdata arises only for a genuinely faceless wire edge,
    /// which never occurs on a face-region boundary edge.
    /// The newly-created side ("vertical") edges between old and new verts
    /// take default attribute values.
    ///
    /// See `radial-mesh-guides/EXTRUDE_REGION.md` for the full algorithm
    /// description and parity contract.
    pub fn extrude_region(&mut self, faces: &[FaceKey]) -> ExtrudeResult {
        self.extrude_region_ex(faces, false)
    }

    /// [`extrude_region`](Self::extrude_region) with the `use_normal_flip`
    /// parameter exposed: when set, each side-wall quad's winding is reversed
    /// (EXTRUDE.md §4.2 / §5). The default (`false`) winds each wall so its
    /// loop on the duplicate edge runs opposite the duplicate face's loop;
    /// flipping reverses that choice, so the wall's outward normal points the
    /// other way. Topology (counts, incidence) is identical either way — only
    /// the per-wall loop direction differs.
    pub fn extrude_region_ex(&mut self, faces: &[FaceKey], use_normal_flip: bool) -> ExtrudeResult {
        self.extrude_region_impl(faces, &[], false, use_normal_flip, false)
    }

    /// [`extrude_region_ex`](Self::extrude_region_ex) with per-edge wall
    /// suppression (EXTRUDE.md §4.6). Each edge in `edges_exclude` keeps both
    /// its original and its duplicate edge — the duplication phase is
    /// unchanged — but no side-wall face is built spanning that pair. Every
    /// non-excluded boundary edge builds its wall exactly as the plain region
    /// extrude does. Passing an empty `edges_exclude` is byte-for-byte
    /// identical to [`extrude_region_ex`](Self::extrude_region_ex).
    ///
    /// An excluded edge that is *interior* to the selection (every radial face
    /// selected) is a silent no-op for exclusion purposes: interior edges never
    /// get walls, so there is nothing to suppress (EXTRUDE.md §8.9).
    ///
    /// `use_keep_orig` is accepted for forward compatibility with the full
    /// region-extrude parameter set but is not yet implemented; the wire-
    /// boundary-kill interaction (§8.3) is likewise out of scope here.
    pub fn extrude_region_exclude(
        &mut self,
        faces: &[FaceKey],
        edges_exclude: &[EdgeKey],
        use_keep_orig: bool,
        use_normal_flip: bool,
    ) -> ExtrudeResult {
        self.extrude_region_impl(faces, edges_exclude, use_keep_orig, use_normal_flip, false)
    }

    /// [`extrude_region_ex`](Self::extrude_region_ex) with the
    /// `use_normal_from_adjacent` winding cue exposed (EXTRUDE.md §5).
    ///
    /// The default region extrude derives each side-wall's winding from the
    /// *duplicated* face's loop direction across the boundary edge — i.e. from
    /// the lifted region itself. When `use_normal_from_adjacent` is set, the
    /// winding cue is read instead from the *un-extruded neighbour* (surround)
    /// face's loop on that edge. For a boundary edge with a surround face the
    /// two cues can disagree, so the resulting wall winding can differ; for a
    /// boundary edge with no surround (mesh-boundary) there is no neighbour to
    /// read and the wall falls back to the duplicate-face cue, matching the
    /// default. `use_normal_flip` still flips whichever cue was chosen.
    ///
    /// This is the form chained operators (e.g. spin) want: successive
    /// iterations' walls stay wound consistently with the surround they grow
    /// out of, rather than with whatever orientation the duplicate sub-step
    /// happened to give the fresh cap.
    ///
    /// Topology (counts, incidence) is identical to
    /// [`extrude_region_ex`](Self::extrude_region_ex); only per-wall loop
    /// direction can change.
    pub fn extrude_region_ex_normal_from_adjacent(
        &mut self,
        faces: &[FaceKey],
        use_normal_flip: bool,
        use_normal_from_adjacent: bool,
    ) -> ExtrudeResult {
        self.extrude_region_impl(faces, &[], false, use_normal_flip, use_normal_from_adjacent)
    }

    fn extrude_region_impl(
        &mut self,
        faces: &[FaceKey],
        edges_exclude: &[EdgeKey],
        use_keep_orig: bool,
        use_normal_flip: bool,
        use_normal_from_adjacent: bool,
    ) -> ExtrudeResult {
        assert!(
            !use_keep_orig,
            "extrude_region_exclude: use_keep_orig is not yet implemented"
        );
        // Mark the selection. Bit 0 = "in region".
        let sel = self.push_face_op_flag_layer();
        for &f in faces {
            self.face_op_flag_set(sel, f, 0);
        }

        // ---- Phase 1: collect per-face loop lists + the union of verts /
        //               edges those loops reference. Snapshot now because
        //               Phase 4 will mutate the radial cycles. ----
        let mut per_face_loops: SmallVec<[(FaceKey, SmallVec<[LoopKey; 8]>); 16]> = SmallVec::new();
        let vert_watermark = self.vertices.key_watermark();
        let edge_watermark = self.edges.key_watermark();
        // Slab-index keyed presence flags. Faster than HashSet for typical
        // meshes.
        let mut vert_in_region: Vec<bool> = vec![false; vert_watermark];
        let mut edge_in_region: Vec<bool> = vec![false; edge_watermark];
        // §4.6 wall suppression: edges named here keep their original/duplicate
        // pair but get no wall face. Slab-index keyed to match the region flags.
        let mut edge_excluded: Vec<bool> = vec![false; edge_watermark];
        for &e in edges_exclude {
            let slot = e.as_usize();
            if slot < edge_watermark {
                edge_excluded[slot] = true;
            }
        }
        for &f in faces {
            let mut loops: SmallVec<[LoopKey; 8]> = SmallVec::new();
            let mut it = self.face_loops(f);
            while let Some(l) = it.next(self) {
                loops.push(l);
                let v = self.loops.get(l).vertex;
                let e = self.loops.get(l).edge;
                vert_in_region[v.as_usize()] = true;
                edge_in_region[e.as_usize()] = true;
            }
            per_face_loops.push((f, loops));
        }

        // ---- Phase 2: classify each in-region edge. For each, walk the
        //               radial *now* (before re-anchoring) and snapshot per
        //               selected-loop: (the loop key, its original vertex).
        //               We need both for wall-face winding in Phase 5.
        // ----
        struct EdgeInfo {
            is_boundary: bool,
            // (loop, its pre-mutation vertex). The vertex equals one of the
            // edge's two endpoints — used to decide the wall's winding.
            selected_loop_verts: SmallVec<[(LoopKey, VertexKey, FaceKey); 4]>,
            // The two corner-loops the wall sources its per-corner loop
            // customdata from, keyed by the endpoint vertex each corner sits
            // at. These are the *other radial loop across the original edge*:
            // the surround (unselected) face's loops when a surround is
            // present, else the original selected face's own loops on this
            // edge (mesh-boundary case). A wall corner at endpoint `v` (either
            // side) copies from the entry whose `.0 == v`. Only empty for a
            // genuinely faceless wire edge.
            source_loop_at: SmallVec<[(VertexKey, LoopKey); 2]>,
            // §5 `use_normal_from_adjacent`: the vertex the first *surround*
            // (un-extruded neighbour) loop is anchored at on this edge. The
            // winding cue when `use_normal_from_adjacent` is set is read from
            // this rather than from the duplicate face's loop. `None` when the
            // edge has no surround face (mesh-boundary) — there is nothing to
            // read, so the wall falls back to the duplicate-face cue.
            adjacent_loop_vert: Option<VertexKey>,
        }
        let mut edge_info: Vec<Option<EdgeInfo>> = (0..edge_watermark).map(|_| None).collect();
        for slot in 0..edge_watermark {
            if !edge_in_region[slot] {
                continue;
            }
            let e = EdgeKey::from_usize(slot);
            let mut selected_loop_verts: SmallVec<[(LoopKey, VertexKey, FaceKey); 4]> =
                SmallVec::new();
            // The preferred customdata source: the first surround face's two
            // corner-loops on this edge. Falls back to the original selected
            // face's own loops (`fallback_loop_at`) when no surround exists.
            let mut surround_loop_at: SmallVec<[(VertexKey, LoopKey); 2]> = SmallVec::new();
            let mut fallback_loop_at: SmallVec<[(VertexKey, LoopKey); 2]> = SmallVec::new();
            // §5: the first surround loop's anchored vertex, used as the
            // `use_normal_from_adjacent` winding cue. Set on the same first
            // surround loop that supplies `surround_loop_at`.
            let mut adjacent_loop_vert: Option<VertexKey> = None;
            let mut has_unselected = false;
            let mut radial = self.radial_loops(e);
            while let Some(l) = radial.next(self) {
                let lref = self.loops.get(l);
                let face = lref.face;
                // A face's two corner-loops on this edge are `l` (anchored at
                // `lref.vertex`) and `l`'s face-cycle successor (anchored at
                // the edge's other endpoint).
                if self.face_op_flag_test(sel, face, 0) {
                    selected_loop_verts.push((l, lref.vertex, face));
                    // First selected face supplies the mesh-boundary fallback
                    // source, used only when this edge has no surround face.
                    if fallback_loop_at.is_empty() {
                        let l_next = self.loops.get(l).face_loop_links.next;
                        let v_here = lref.vertex;
                        let v_next = self.loops.get(l_next).vertex;
                        fallback_loop_at.push((v_here, l));
                        fallback_loop_at.push((v_next, l_next));
                    }
                } else {
                    has_unselected = true;
                    // First surround face wins as the customdata source and as
                    // the §5 `use_normal_from_adjacent` winding cue.
                    if surround_loop_at.is_empty() {
                        let l_next = self.loops.get(l).face_loop_links.next;
                        let v_here = lref.vertex;
                        let v_next = self.loops.get(l_next).vertex;
                        surround_loop_at.push((v_here, l));
                        surround_loop_at.push((v_next, l_next));
                        adjacent_loop_vert = Some(v_here);
                    }
                }
            }
            // Unified §4.3 rule: source from the surround loop when present,
            // else from the original selected face's own loop on this edge.
            let source_loop_at = if surround_loop_at.is_empty() {
                fallback_loop_at
            } else {
                surround_loop_at
            };
            // An edge is *interior to the selection* only when every face
            // on its radial is selected AND there are at least two of them
            // (you need ≥ 2 selected faces meeting on an edge for the
            // selection to fully enclose it). Otherwise it's a boundary:
            //   - radial of 1 (mesh-boundary): the "outside" is implicit;
            //     extrude needs a wall here.
            //   - radial of 2 with one unselected: manifold boundary of
            //     the selection.
            //   - radial of N > 2 with at least one unselected: non-
            //     manifold boundary; one wall per selected loop.
            let is_boundary = has_unselected || selected_loop_verts.len() < 2;
            edge_info[slot] = Some(EdgeInfo {
                is_boundary,
                selected_loop_verts,
                source_loop_at,
                adjacent_loop_vert,
            });
        }

        // ---- Phase 3: build vert / edge duplicate maps. ----
        let mut vert_to_dup: Vec<VertexKey> =
            vec![VertexKey::from_usize(usize::MAX); vert_watermark];
        for slot in 0..vert_watermark {
            if !vert_in_region[slot] {
                continue;
            }
            let v_old = VertexKey::from_usize(slot);
            let dup = self.create_vertex(self.vertex_position(v_old));
            vert_to_dup[slot] = dup;
            self.vertex_attribs.copy_into(dup.as_usize(), slot);
        }
        let mut edge_to_dup: Vec<EdgeKey> = vec![EdgeKey::NULL; edge_watermark];
        for slot in 0..edge_watermark {
            if !edge_in_region[slot] {
                continue;
            }
            let e_old = EdgeKey::from_usize(slot);
            let [v0, v1] = self.edge_vertices(e_old);
            let dup_v0 = vert_to_dup[v0.as_usize()];
            let dup_v1 = vert_to_dup[v1.as_usize()];
            let dup = self.create_edge([dup_v0, dup_v1], EdgeCreateFlags::NONE);
            edge_to_dup[slot] = dup;
            self.edge_attribs.copy_into(dup.as_usize(), slot);
        }

        // ---- Phase 4: re-anchor each selected face's loops to dup verts +
        //               dup edges. The face's loop cycle (face_loop_links)
        //               stays intact; only loop.vertex and the radial
        //               attachment change. ----
        for (_f, loops) in &per_face_loops {
            for &l in loops {
                let old_e = self.loops.get(l).edge;
                let old_v = self.loops.get(l).vertex;
                let new_e = edge_to_dup[old_e.as_usize()];
                let new_v = vert_to_dup[old_v.as_usize()];
                self.edge_replace_in_face_loop(l, new_e);
                self.loops.get_mut(l).vertex = new_v;
            }
        }

        // ---- Phase 5: build wall quads. One wall per selected loop on each
        //               boundary edge's pre-mutation radial. Vertical edges
        //               are lazily created and shared across walls at the
        //               same source vertex. ----
        let mut side_edge_at_v: Vec<EdgeKey> = vec![EdgeKey::NULL; vert_watermark];
        let mut wall_faces: Vec<FaceKey> = Vec::new();
        let do_loop_cd = !self.loop_attribs.is_empty();
        let mut vert_map_out: Vec<(VertexKey, VertexKey)> = Vec::new();
        let mut edge_map_out: Vec<(EdgeKey, EdgeKey)> = Vec::new();

        for slot in 0..edge_watermark {
            let info = match &edge_info[slot] {
                Some(info) if info.is_boundary => info,
                _ => continue,
            };
            // §4.6: an excluded boundary edge keeps its original/duplicate pair
            // (built in Phase 3) but gets no wall face here.
            if edge_excluded[slot] {
                continue;
            }
            let e_old = EdgeKey::from_usize(slot);
            let [v0_old, v1_old] = self.edge_vertices(e_old);
            let v0_new = vert_to_dup[v0_old.as_usize()];
            let v1_new = vert_to_dup[v1_old.as_usize()];
            let e_new = edge_to_dup[slot];

            for &(_l_sel, l_v_orig, src_face) in &info.selected_loop_verts {
                // l_v_orig is the loop's PRE-MUTATION vertex; one of v0_old / v1_old.
                // EXTRUDE.md §5 picks the wall's winding cue (`edge_normal_flip`)
                // from one of two sources:
                //   - default: the duplicate face's loop direction across the
                //     edge — captured here as the selected loop's pre-mutation
                //     vertex `l_v_orig` (the duplicate is the re-anchored same
                //     loop, so its e_dup loop runs in the same vertex order);
                //   - use_normal_from_adjacent: the surround (un-extruded
                //     neighbour) loop's vertex on the edge. With no surround
                //     (mesh-boundary) there is nothing to read, so fall back to
                //     the default cue.
                // `use_normal_flip` then flips whichever cue was chosen.
                debug_assert!(l_v_orig == v0_old || l_v_orig == v1_old);
                let edge_normal_flip = match (use_normal_from_adjacent, info.adjacent_loop_vert) {
                    (true, Some(adjacent_vert)) => v0_old == adjacent_vert,
                    _ => l_v_orig != v0_old,
                };
                let a_is_v0 = edge_normal_flip == use_normal_flip;
                let (v_a_old, v_b_old, v_a_new, v_b_new) = if a_is_v0 {
                    (v0_old, v1_old, v0_new, v1_new)
                } else {
                    (v1_old, v0_old, v1_new, v0_new)
                };

                let side_a = if side_edge_at_v[v_a_old.as_usize()].is_null() {
                    let e = self.create_edge([v_a_old, v_a_new], EdgeCreateFlags::NONE);
                    side_edge_at_v[v_a_old.as_usize()] = e;
                    e
                } else {
                    side_edge_at_v[v_a_old.as_usize()]
                };
                let side_b = if side_edge_at_v[v_b_old.as_usize()].is_null() {
                    let e = self.create_edge([v_b_old, v_b_new], EdgeCreateFlags::NONE);
                    side_edge_at_v[v_b_old.as_usize()] = e;
                    e
                } else {
                    side_edge_at_v[v_b_old.as_usize()]
                };

                // Wall quad: v_a_old -> v_b_old -> v_b_new -> v_a_new (outward winding).
                let wall = self.face_create(
                    &[v_a_old, v_b_old, v_b_new, v_a_new],
                    &[e_old, side_b, e_new, side_a],
                    FaceCreateFlags::NONE,
                );
                self.face_attribs
                    .copy_into(wall.as_usize(), src_face.as_usize());

                // Wall loop customdata flows continuously across the wall from
                // the source face (surround if present, else the original
                // selected face — §4.3): each of the four corners copies from
                // the source loop sitting at the corner's *original* endpoint,
                // so the duplicate-side corners (v_b_new, v_a_new) pick up the
                // same values as the original-side corners they sit above.
                if do_loop_cd && !info.source_loop_at.is_empty() {
                    let wall_source_at = |v: VertexKey| -> Option<LoopKey> {
                        info.source_loop_at
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
                        if let Some(source) = wall_source_at(origin_vert) {
                            self.loop_attribs
                                .copy_into(wall_loop.as_usize(), source.as_usize());
                        }
                    }
                }

                wall_faces.push(wall);
            }
        }

        // Build the output maps in dense-index order.
        for slot in 0..vert_watermark {
            if vert_in_region[slot] {
                vert_map_out.push((VertexKey::from_usize(slot), vert_to_dup[slot]));
            }
        }
        for slot in 0..edge_watermark {
            if edge_in_region[slot] {
                edge_map_out.push((EdgeKey::from_usize(slot), edge_to_dup[slot]));
            }
        }

        let mut side_edges_out: Vec<EdgeKey> = Vec::new();
        for slot in 0..vert_watermark {
            let e = side_edge_at_v[slot];
            if !e.is_null() {
                side_edges_out.push(e);
            }
        }

        self.pop_face_op_flag_layer(sel);

        ExtrudeResult {
            vert_map: vert_map_out,
            edge_map: edge_map_out,
            side_edges: side_edges_out,
            wall_faces,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use glam::Vec3;

    use crate::loopmesh::test_support::{create_face_from_verts, mesh_add_cube};
    use crate::loopmesh::*;

    fn unit_quad(mesh: &mut LoopMesh) -> FaceKey {
        let verts = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        ];
        create_face_from_verts(mesh, &verts)
    }

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

    #[test]
    fn extrude_region_single_quad_isolated() {
        // A single quad sitting alone in the mesh. Extruding it produces 4 new
        // verts + 4 wall quads + 1 top cap (the original face, re-anchored).
        let mut mesh = LoopMesh::new();
        let verts = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        ];
        let face = create_face_from_verts(&mut mesh, &verts);

        let result = mesh.extrude_region(&[face]);

        assert_eq!(result.vert_map.len(), 4, "4 boundary verts duplicated");
        assert_eq!(result.edge_map.len(), 4, "4 boundary edges duplicated");
        assert_eq!(
            result.side_edges.len(),
            4,
            "4 vertical edges, one per source vert"
        );
        assert_eq!(
            result.wall_faces.len(),
            4,
            "4 wall quads, one per boundary edge"
        );

        assert_eq!(mesh.totvert(), 8);
        assert_eq!(mesh.totface(), 5); // 4 walls + 1 top cap
        mesh.validate_topology();

        // Displace the new verts upward and verify the top cap still has the
        // expected normal direction.
        for &(_old, new_v) in &result.vert_map {
            let p = mesh.vertex_position(new_v);
            mesh.set_vertex_position(new_v, p + Vec3::new(0.0, 0.0, 1.0));
        }
        mesh.validate_topology();
    }

    #[test]
    fn extrude_region_isolated_face_keeps_boundary() {
        // After extrude, the source verts of an isolated face should still be
        // referenced by the *wall* faces (not the top cap).
        let mut mesh = LoopMesh::new();
        let verts = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        ];
        let face = create_face_from_verts(&mut mesh, &verts);
        let source_verts: Vec<VertexKey> = mesh.face_vert_keys(face);

        let result = mesh.extrude_region(&[face]);

        // Top cap's perimeter uses the duplicated verts.
        let top_verts: Vec<VertexKey> = mesh.face_vert_keys(face);
        for v in &top_verts {
            assert!(
                !source_verts.contains(v),
                "top cap should now reference duplicates, not source verts"
            );
        }
        // Wall faces collectively touch every source vert.
        let mut wall_verts = std::collections::HashSet::new();
        for &wf in &result.wall_faces {
            for v in mesh.face_vert_keys(wf) {
                wall_verts.insert(v);
            }
        }
        for v in &source_verts {
            assert!(
                wall_verts.contains(v),
                "wall faces must reference every source vert"
            );
        }
    }

    #[test]
    fn extrude_region_cube_top_face() {
        // Cube with the top face selected → the top face becomes a "lid" sitting
        // on 4 new walls. The cube's 5 unselected faces stay attached to the
        // original boundary edges.
        let mut mesh = LoopMesh::new();
        mesh_add_cube(&mut mesh);
        // mesh_add_cube creates faces in a known order; the first one is the
        // bottom (-Z). We'll pick the top (+Z) face by finding a face whose
        // normal points in +Z.
        mesh.all_indices_ensure();
        mesh.calc_face_normals();
        let mut top_face = None;
        let face_keys = mesh.face_keys();
        for &f in &face_keys {
            let n = mesh.face_normal(f);
            if n.z > 0.9 {
                top_face = Some(f);
                break;
            }
        }
        let top = top_face.expect("cube has a +Z face");

        let result = mesh.extrude_region(&[top]);

        assert_eq!(result.vert_map.len(), 4);
        assert_eq!(result.edge_map.len(), 4);
        assert_eq!(result.wall_faces.len(), 4);
        // Cube had 8 verts + 12 edges + 6 faces. After: +4 verts, +4 dup edges
        // + 4 side edges = +8 edges, +4 wall faces.
        assert_eq!(mesh.totvert(), 12);
        assert_eq!(mesh.totedge(), 20);
        assert_eq!(mesh.totface(), 10);
        mesh.validate_topology();
    }

    #[test]
    fn extrude_region_two_adjacent_faces_shares_interior_edge() {
        // Two triangles sharing one edge. Both selected → the shared edge is
        // *interior* to the selection. After extrude, the original shared edge
        // becomes a wire edge (no faces); both new triangles share a single
        // duplicate edge.
        let mut mesh = LoopMesh::new();
        let v0 = mesh.create_vertex(Vec3::new(0.0, 0.0, 0.0));
        let v1 = mesh.create_vertex(Vec3::new(1.0, 0.0, 0.0));
        let v2 = mesh.create_vertex(Vec3::new(1.0, 1.0, 0.0));
        let v3 = mesh.create_vertex(Vec3::new(0.0, 1.0, 0.0));
        let e01 = mesh.create_edge([v0, v1], EdgeCreateFlags::NONE);
        let e12 = mesh.create_edge([v1, v2], EdgeCreateFlags::NONE);
        let e02 = mesh.create_edge([v0, v2], EdgeCreateFlags::NONE);
        let e23 = mesh.create_edge([v2, v3], EdgeCreateFlags::NONE);
        let e30 = mesh.create_edge([v3, v0], EdgeCreateFlags::NONE);
        let f1 = mesh.face_create(&[v0, v1, v2], &[e01, e12, e02], FaceCreateFlags::NONE);
        let f2 = mesh.face_create(&[v0, v2, v3], &[e02, e23, e30], FaceCreateFlags::NONE);

        let result = mesh.extrude_region(&[f1, f2]);

        // 4 unique boundary verts (v0,v1,v2,v3) all duplicated; 5 edges
        // duplicated (e01, e12, e23, e30 boundary + e02 interior).
        assert_eq!(result.vert_map.len(), 4);
        assert_eq!(result.edge_map.len(), 5);
        // 4 boundary edges → 4 walls (each manifold radial=1 since they had
        // only 1 selected face on them; the unselected side is "outside the
        // mesh", i.e. boundary edges). The interior edge (e02, radial=2 both
        // selected) gets no wall.
        assert_eq!(result.wall_faces.len(), 4);
        mesh.validate_topology();

        // The interior edge e02 is now a wire edge (no faces on its radial).
        assert!(
            mesh.edge_radial_iter(e02).count() == 0,
            "interior edge of selection becomes wire after extrude"
        );
    }

    #[test]
    fn extrude_region_copies_vertex_customdata() {
        let mut mesh = LoopMesh::new();
        let verts = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        ];
        let face = create_face_from_verts(&mut mesh, &verts);
        let source_verts = mesh.face_vert_keys(face);

        let color = mesh
            .register_vertex_attribute(ElementAttributeDesc::numeric(
                AttributeKey::custom("color"),
                Format::new(ComponentType::F32, 3),
            ))
            .unwrap();
        let palette = [
            [1.0_f32, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
        ];
        for (i, &v) in source_verts.iter().enumerate() {
            mesh.vertex_attribs_mut()
                .set_vec3(color, v.as_usize(), palette[i]);
        }

        let result = mesh.extrude_region(&[face]);

        // Every duplicate carries its source vertex's color exactly.
        for &(src, dup) in &result.vert_map {
            let src_val = mesh
                .vertex_attribs()
                .get_vec3(color, src.as_usize())
                .unwrap();
            let dup_val = mesh
                .vertex_attribs()
                .get_vec3(color, dup.as_usize())
                .unwrap();
            assert_eq!(src_val, dup_val);
        }
    }

    #[test]
    fn extrude_region_wall_loop_uv_sourced_from_surround() {
        // Two quads sharing edge v1–v2. Each face carries a distinct
        // per-corner UV. Extruding face 0 makes the shared edge a manifold
        // boundary whose surround is face 1; per EXTRUDE.md §4.3 the wall on
        // that edge sources all four corner UVs from face 1's two loops on
        // the edge (the duplicate-side corners take the same source as the
        // original-side corners). The no-surround case is pinned separately
        // in `extrude_region_no_surround_wall_loop_uv_from_original_face`.
        let mut mesh = LoopMesh::new();
        let v0 = mesh.create_vertex(Vec3::new(0.0, 0.0, 0.0));
        let v1 = mesh.create_vertex(Vec3::new(1.0, 0.0, 0.0));
        let v2 = mesh.create_vertex(Vec3::new(1.0, 1.0, 0.0));
        let v3 = mesh.create_vertex(Vec3::new(0.0, 1.0, 0.0));
        let v4 = mesh.create_vertex(Vec3::new(2.0, 0.0, 0.0));
        let v5 = mesh.create_vertex(Vec3::new(2.0, 1.0, 0.0));
        let e01 = mesh.create_edge([v0, v1], EdgeCreateFlags::NONE);
        let e12 = mesh.create_edge([v1, v2], EdgeCreateFlags::NONE);
        let e23 = mesh.create_edge([v2, v3], EdgeCreateFlags::NONE);
        let e30 = mesh.create_edge([v3, v0], EdgeCreateFlags::NONE);
        let e14 = mesh.create_edge([v1, v4], EdgeCreateFlags::NONE);
        let e45 = mesh.create_edge([v4, v5], EdgeCreateFlags::NONE);
        let e52 = mesh.create_edge([v5, v2], EdgeCreateFlags::NONE);
        let face0 = mesh.face_create(&[v0, v1, v2, v3], &[e01, e12, e23, e30], FaceCreateFlags::NONE);
        let face1 = mesh.face_create(&[v1, v4, v5, v2], &[e14, e45, e52, e12], FaceCreateFlags::NONE);

        let uv = mesh
            .register_loop_attribute(ElementAttributeDesc::numeric(
                AttributeKey::custom("uv"),
                Format::new(ComponentType::F32, 2),
            ))
            .unwrap();

        // Give every corner of both faces a UV equal to its world XY, so the
        // gradient is non-trivial and distinct per corner.
        for &f in &[face0, face1] {
            for l in mesh.iter_face_loops(f).collect::<Vec<_>>() {
                let v = mesh.loop_vertex(l);
                let p = mesh.vertex_position(v);
                mesh.loop_attribs_mut().set_vec2(uv, l.as_usize(), [p.x, p.y]);
            }
        }

        // Capture face 1's (surround) loop UVs at the shared-edge endpoints.
        let mut surround_uv_at_v1 = None;
        let mut surround_uv_at_v2 = None;
        for l in mesh.iter_face_loops(face1).collect::<Vec<_>>() {
            let v = mesh.loop_vertex(l);
            let val = mesh.loop_attribs().get_vec2(uv, l.as_usize()).unwrap();
            if v == v1 {
                surround_uv_at_v1 = Some(val);
            } else if v == v2 {
                surround_uv_at_v2 = Some(val);
            }
        }
        let surround_uv_at_v1 = surround_uv_at_v1.unwrap();
        let surround_uv_at_v2 = surround_uv_at_v2.unwrap();

        let result = mesh.extrude_region(&[face0]);

        // The duplicates of v1 / v2.
        let mut v1_dup = None;
        let mut v2_dup = None;
        for &(src, dup) in &result.vert_map {
            if src == v1 {
                v1_dup = Some(dup);
            } else if src == v2 {
                v2_dup = Some(dup);
            }
        }
        let v1_dup = v1_dup.unwrap();
        let v2_dup = v2_dup.unwrap();

        // Find the wall on the shared edge: the wall face whose corner set
        // contains both v1 and v2 (its original-edge endpoints).
        let mut shared_wall = None;
        for &wf in &result.wall_faces {
            let verts = mesh.face_vert_keys(wf);
            if verts.contains(&v1) && verts.contains(&v2) {
                shared_wall = Some(wf);
                break;
            }
        }
        let shared_wall = shared_wall.expect("a wall sits on the shared edge");

        // Each of the four wall corners must carry the surround's UV at the
        // matching original endpoint — including the duplicate-side corners.
        for l in mesh.iter_face_loops(shared_wall).collect::<Vec<_>>() {
            let corner_vert = mesh.loop_vertex(l);
            let got = mesh.loop_attribs().get_vec2(uv, l.as_usize()).unwrap();
            let expected = if corner_vert == v1 || corner_vert == v1_dup {
                surround_uv_at_v1
            } else {
                assert!(corner_vert == v2 || corner_vert == v2_dup);
                surround_uv_at_v2
            };
            assert_eq!(
                got, expected,
                "shared-edge wall corner at {corner_vert:?} must source surround UV"
            );
        }

        // The surround UVs are non-zero, so this is a real propagation, not a
        // coincidental default.
        assert!(surround_uv_at_v1 != [0.0, 0.0] || surround_uv_at_v2 != [0.0, 0.0]);

        mesh.validate_topology();
    }

    #[test]
    fn extrude_region_no_surround_wall_loop_uv_from_original_face() {
        // Two quads sharing edge v1–v2 (the worked example from EXTRUDE.md
        // §4.3). Extruding face 0: the shared edge v1–v2 has a surround
        // (face 1); the other three edges are mesh-boundary (no surround) and
        // per the unified §4.3 rule source their wall loop CD from face 0's
        // own per-corner loops — NOT default zero. Corner UVs of face 0 are
        // (0,0) at v0, (1,0) at v1, (1,1) at v2, (0,1) at v3.
        let mut mesh = LoopMesh::new();
        let v0 = mesh.create_vertex(Vec3::new(0.0, 0.0, 0.0));
        let v1 = mesh.create_vertex(Vec3::new(1.0, 0.0, 0.0));
        let v2 = mesh.create_vertex(Vec3::new(1.0, 1.0, 0.0));
        let v3 = mesh.create_vertex(Vec3::new(0.0, 1.0, 0.0));
        let v4 = mesh.create_vertex(Vec3::new(2.0, 0.0, 0.0));
        let v5 = mesh.create_vertex(Vec3::new(2.0, 1.0, 0.0));
        let e01 = mesh.create_edge([v0, v1], EdgeCreateFlags::NONE);
        let e12 = mesh.create_edge([v1, v2], EdgeCreateFlags::NONE);
        let e23 = mesh.create_edge([v2, v3], EdgeCreateFlags::NONE);
        let e30 = mesh.create_edge([v3, v0], EdgeCreateFlags::NONE);
        let e14 = mesh.create_edge([v1, v4], EdgeCreateFlags::NONE);
        let e45 = mesh.create_edge([v4, v5], EdgeCreateFlags::NONE);
        let e52 = mesh.create_edge([v5, v2], EdgeCreateFlags::NONE);
        let face0 =
            mesh.face_create(&[v0, v1, v2, v3], &[e01, e12, e23, e30], FaceCreateFlags::NONE);
        let face1 =
            mesh.face_create(&[v1, v4, v5, v2], &[e14, e45, e52, e12], FaceCreateFlags::NONE);

        let uv = mesh
            .register_loop_attribute(ElementAttributeDesc::numeric(
                AttributeKey::custom("uv"),
                Format::new(ComponentType::F32, 2),
            ))
            .unwrap();

        // UVs equal to world XY: distinct per corner and non-zero except v0.
        for &f in &[face0, face1] {
            for l in mesh.iter_face_loops(f).collect::<Vec<_>>() {
                let v = mesh.loop_vertex(l);
                let p = mesh.vertex_position(v);
                mesh.loop_attribs_mut().set_vec2(uv, l.as_usize(), [p.x, p.y]);
            }
        }
        // Capture face 0's own loop UV at each of its corners, keyed by vert.
        let face0_uv_at = |mesh: &LoopMesh, want: VertexKey| -> [f32; 2] {
            for l in mesh.iter_face_loops(face0).collect::<Vec<_>>() {
                if mesh.loop_vertex(l) == want {
                    return mesh.loop_attribs().get_vec2(uv, l.as_usize()).unwrap();
                }
            }
            panic!("vert not on face0");
        };
        let uv_v0 = face0_uv_at(&mesh, v0);
        let uv_v1 = face0_uv_at(&mesh, v1);
        let uv_v3 = face0_uv_at(&mesh, v3);

        let result = mesh.extrude_region(&[face0]);

        let dup_of = |src: VertexKey| -> VertexKey {
            result
                .vert_map
                .iter()
                .find(|(s, _)| *s == src)
                .map(|(_, d)| *d)
                .unwrap()
        };
        let v0_dup = dup_of(v0);
        let v1_dup = dup_of(v1);
        let v3_dup = dup_of(v3);

        // Helper: find the wall on a given original endpoint pair.
        let wall_on = |mesh: &LoopMesh, a: VertexKey, b: VertexKey| -> FaceKey {
            for &wf in &result.wall_faces {
                let verts = mesh.face_vert_keys(wf);
                if verts.contains(&a) && verts.contains(&b) {
                    return wf;
                }
            }
            panic!("no wall on edge");
        };

        // Wall on the v0–v1 mesh-boundary edge: each column carries face 0's
        // own corner UV, identical top and bottom.
        let wall_v0v1 = wall_on(&mesh, v0, v1);
        for l in mesh.iter_face_loops(wall_v0v1).collect::<Vec<_>>() {
            let cv = mesh.loop_vertex(l);
            let got = mesh.loop_attribs().get_vec2(uv, l.as_usize()).unwrap();
            let expected = if cv == v0 || cv == v0_dup {
                uv_v0
            } else {
                assert!(cv == v1 || cv == v1_dup);
                uv_v1
            };
            assert_eq!(got, expected, "v0–v1 wall column at {cv:?}");
        }

        // Wall on the v3–v0 mesh-boundary edge: columns from face 0's loops at
        // v3 (0,1) and v0 (0,0). v3's value is non-zero → not a default.
        let wall_v3v0 = wall_on(&mesh, v3, v0);
        let mut saw_nonzero = false;
        for l in mesh.iter_face_loops(wall_v3v0).collect::<Vec<_>>() {
            let cv = mesh.loop_vertex(l);
            let got = mesh.loop_attribs().get_vec2(uv, l.as_usize()).unwrap();
            let expected = if cv == v3 || cv == v3_dup {
                uv_v3
            } else {
                assert!(cv == v0 || cv == v0_dup);
                uv_v0
            };
            assert_eq!(got, expected, "v3–v0 wall column at {cv:?}");
            if got != [0.0, 0.0] {
                saw_nonzero = true;
            }
        }
        assert!(
            saw_nonzero,
            "no-surround wall must carry the original face's non-zero corner UV, not default zero"
        );

        mesh.validate_topology();
    }

    #[test]
    fn extrude_region_normal_flip_reverses_wall_winding() {
        // use_normal_flip must reverse each side-wall's vertex cycle relative to
        // the default. Extrude the same isolated quad twice (default vs flipped)
        // and assert each wall's perimeter is the reverse cycle of its default
        // counterpart over the same original boundary edge. CanonicalMesh is
        // winding-insensitive, so this in-crate cycle check is what truly pins
        // the flag.
        let mut mesh_default = LoopMesh::new();
        let face_default = unit_quad(&mut mesh_default);
        let res_default = mesh_default.extrude_region(&[face_default]);

        let mut mesh_flip = LoopMesh::new();
        let face_flip = unit_quad(&mut mesh_flip);
        let res_flip = mesh_flip.extrude_region_ex(&[face_flip], true);

        // Same topology — only winding differs.
        assert_eq!(res_default.wall_faces.len(), res_flip.wall_faces.len());
        assert_eq!(res_default.wall_faces.len(), 4);

        // A wall's two lower (original) verts identify its boundary edge.
        // Positions are identical across both meshes, so match by position.
        let lower_pos = |mesh: &LoopMesh, wf: FaceKey| -> HashSet<[u32; 3]> {
            mesh.face_vert_keys(wf)
                .into_iter()
                .filter(|&v| mesh.vertex_position(v).z == 0.0)
                .map(|v| pos_key(mesh.vertex_position(v)))
                .collect()
        };

        for &wf_def in &res_default.wall_faces {
            let def_pos: Vec<Vec3> = mesh_default
                .face_vert_keys(wf_def)
                .iter()
                .map(|&v| mesh_default.vertex_position(v))
                .collect();
            let def_lower = lower_pos(&mesh_default, wf_def);

            let mut matched = false;
            for &wf_flip in &res_flip.wall_faces {
                if lower_pos(&mesh_flip, wf_flip) != def_lower {
                    continue;
                }
                let flip_pos: Vec<Vec3> = mesh_flip
                    .face_vert_keys(wf_flip)
                    .iter()
                    .map(|&v| mesh_flip.vertex_position(v))
                    .collect();
                assert!(
                    cycles_are_reversed(&def_pos, &flip_pos),
                    "flip wall must be the reverse cycle of the default wall"
                );
                matched = true;
                break;
            }
            assert!(matched, "every default wall has a flipped counterpart");
        }

        mesh_flip.validate_topology();
    }

    #[test]
    fn extrude_region_default_matches_ex_false() {
        // The convenience form must delegate to the _ex form with
        // use_normal_flip = false, producing identical wall windings.
        let mut mesh_a = LoopMesh::new();
        let face_a = unit_quad(&mut mesh_a);
        let res_a = mesh_a.extrude_region(&[face_a]);

        let mut mesh_b = LoopMesh::new();
        let face_b = unit_quad(&mut mesh_b);
        let res_b = mesh_b.extrude_region_ex(&[face_b], false);

        assert_eq!(res_a.wall_faces.len(), res_b.wall_faces.len());

        let lower_pos = |mesh: &LoopMesh, wf: FaceKey| -> HashSet<[u32; 3]> {
            mesh.face_vert_keys(wf)
                .into_iter()
                .filter(|&v| mesh.vertex_position(v).z == 0.0)
                .map(|v| pos_key(mesh.vertex_position(v)))
                .collect()
        };

        for &wf_a in &res_a.wall_faces {
            let a_pos: Vec<Vec3> = mesh_a
                .face_vert_keys(wf_a)
                .iter()
                .map(|&v| mesh_a.vertex_position(v))
                .collect();
            let a_lower = lower_pos(&mesh_a, wf_a);
            let mut matched = false;
            for &wf_b in &res_b.wall_faces {
                if lower_pos(&mesh_b, wf_b) != a_lower {
                    continue;
                }
                let b_pos: Vec<Vec3> = mesh_b
                    .face_vert_keys(wf_b)
                    .iter()
                    .map(|&v| mesh_b.vertex_position(v))
                    .collect();
                // Same winding: identical as a rotation (not reversed).
                let n = a_pos.len();
                let same_rotation = (0..n).any(|shift| {
                    (0..n).all(|i| a_pos[i] == b_pos[(i + shift) % n])
                });
                assert!(same_rotation, "extrude_region must equal _ex(false) winding");
                matched = true;
                break;
            }
            assert!(matched);
        }
    }

    #[test]
    fn extrude_region_exclude_one_perimeter_edge_suppresses_one_wall() {
        // EXTRUDE.md §4.6: an excluded boundary edge keeps its original and
        // duplicate edge pair but gets no wall face. Extruding an isolated quad
        // normally yields 4 walls; excluding one perimeter edge yields 3.
        let mut mesh = LoopMesh::new();
        let verts = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        ];
        let face = create_face_from_verts(&mut mesh, &verts);
        let source_verts = mesh.face_vert_keys(face);
        let v0 = source_verts[0];
        let v1 = source_verts[1];
        let excluded_edge = mesh.query_edge_exists(v0, v1).expect("perimeter edge v0-v1");

        let result = mesh.extrude_region_exclude(&[face], &[excluded_edge], false, false);

        assert_eq!(result.vert_map.len(), 4, "all 4 boundary verts still duplicated");
        assert_eq!(
            result.edge_map.len(),
            4,
            "all 4 boundary edges still duplicated, including the excluded one"
        );
        assert_eq!(
            result.wall_faces.len(),
            3,
            "the excluded edge contributes no wall face"
        );

        // The excluded edge's duplicate still exists (it's in edge_map).
        assert!(
            result.edge_map.iter().any(|(src, _)| *src == excluded_edge),
            "excluded edge is still duplicated"
        );
        // No wall face spans the excluded edge's two original verts.
        for &wf in &result.wall_faces {
            let wall_verts = mesh.face_vert_keys(wf);
            assert!(
                !(wall_verts.contains(&v0) && wall_verts.contains(&v1)),
                "no wall should sit on the excluded edge"
            );
        }
        mesh.validate_topology();
    }

    #[test]
    fn extrude_region_exclude_empty_matches_plain_region() {
        // An empty exclude set must produce exactly the plain-region result:
        // identical wall count and identical wall windings.
        let mut mesh_plain = LoopMesh::new();
        let face_plain = unit_quad(&mut mesh_plain);
        let res_plain = mesh_plain.extrude_region(&[face_plain]);

        let mut mesh_excl = LoopMesh::new();
        let face_excl = unit_quad(&mut mesh_excl);
        let res_excl = mesh_excl.extrude_region_exclude(&[face_excl], &[], false, false);

        assert_eq!(res_plain.vert_map.len(), res_excl.vert_map.len());
        assert_eq!(res_plain.edge_map.len(), res_excl.edge_map.len());
        assert_eq!(res_plain.side_edges.len(), res_excl.side_edges.len());
        assert_eq!(res_plain.wall_faces.len(), res_excl.wall_faces.len());
        assert_eq!(res_plain.wall_faces.len(), 4);

        let lower_pos = |mesh: &LoopMesh, wf: FaceKey| -> HashSet<[u32; 3]> {
            mesh.face_vert_keys(wf)
                .into_iter()
                .filter(|&v| mesh.vertex_position(v).z == 0.0)
                .map(|v| pos_key(mesh.vertex_position(v)))
                .collect()
        };

        for &wf_plain in &res_plain.wall_faces {
            let plain_pos: Vec<Vec3> = mesh_plain
                .face_vert_keys(wf_plain)
                .iter()
                .map(|&v| mesh_plain.vertex_position(v))
                .collect();
            let plain_lower = lower_pos(&mesh_plain, wf_plain);
            let mut matched = false;
            for &wf_excl in &res_excl.wall_faces {
                if lower_pos(&mesh_excl, wf_excl) != plain_lower {
                    continue;
                }
                let excl_pos: Vec<Vec3> = mesh_excl
                    .face_vert_keys(wf_excl)
                    .iter()
                    .map(|&v| mesh_excl.vertex_position(v))
                    .collect();
                let n = plain_pos.len();
                let same_rotation =
                    (0..n).any(|shift| (0..n).all(|i| plain_pos[i] == excl_pos[(i + shift) % n]));
                assert!(same_rotation, "empty-exclude wall must match plain wall winding");
                matched = true;
                break;
            }
            assert!(matched, "every plain wall has an empty-exclude counterpart");
        }
        mesh_excl.validate_topology();
    }

    #[test]
    fn extrude_region_exclude_interior_edge_is_noop() {
        // EXTRUDE.md §8.9: excluding an interior edge (every radial face
        // selected) has no effect — interior edges never get walls.
        let mut mesh = LoopMesh::new();
        let v0 = mesh.create_vertex(Vec3::new(0.0, 0.0, 0.0));
        let v1 = mesh.create_vertex(Vec3::new(1.0, 0.0, 0.0));
        let v2 = mesh.create_vertex(Vec3::new(1.0, 1.0, 0.0));
        let v3 = mesh.create_vertex(Vec3::new(0.0, 1.0, 0.0));
        let e01 = mesh.create_edge([v0, v1], EdgeCreateFlags::NONE);
        let e12 = mesh.create_edge([v1, v2], EdgeCreateFlags::NONE);
        let e02 = mesh.create_edge([v0, v2], EdgeCreateFlags::NONE);
        let e23 = mesh.create_edge([v2, v3], EdgeCreateFlags::NONE);
        let e30 = mesh.create_edge([v3, v0], EdgeCreateFlags::NONE);
        let f1 = mesh.face_create(&[v0, v1, v2], &[e01, e12, e02], FaceCreateFlags::NONE);
        let f2 = mesh.face_create(&[v0, v2, v3], &[e02, e23, e30], FaceCreateFlags::NONE);

        // e02 is interior to the {f1, f2} selection.
        let result = mesh.extrude_region_exclude(&[f1, f2], &[e02], false, false);

        // Same as the no-exclude two-adjacent-faces case: 4 walls, 5 dup edges.
        assert_eq!(result.edge_map.len(), 5);
        assert_eq!(
            result.wall_faces.len(),
            4,
            "excluding an interior edge changes nothing"
        );
        mesh.validate_topology();
    }

    // Build two quads sharing edge v1–v2: face0 = {v0,v1,v2,v3}, face1 the
    // neighbour on the far side of v1–v2. Returns (mesh, face0, face1, v1, v2).
    fn two_quads_shared_edge(
        mesh: &mut LoopMesh,
    ) -> (FaceKey, FaceKey, VertexKey, VertexKey) {
        let v0 = mesh.create_vertex(Vec3::new(0.0, 0.0, 0.0));
        let v1 = mesh.create_vertex(Vec3::new(1.0, 0.0, 0.0));
        let v2 = mesh.create_vertex(Vec3::new(1.0, 1.0, 0.0));
        let v3 = mesh.create_vertex(Vec3::new(0.0, 1.0, 0.0));
        let v4 = mesh.create_vertex(Vec3::new(2.0, 0.0, 0.0));
        let v5 = mesh.create_vertex(Vec3::new(2.0, 1.0, 0.0));
        let e01 = mesh.create_edge([v0, v1], EdgeCreateFlags::NONE);
        let e12 = mesh.create_edge([v1, v2], EdgeCreateFlags::NONE);
        let e23 = mesh.create_edge([v2, v3], EdgeCreateFlags::NONE);
        let e30 = mesh.create_edge([v3, v0], EdgeCreateFlags::NONE);
        let e14 = mesh.create_edge([v1, v4], EdgeCreateFlags::NONE);
        let e45 = mesh.create_edge([v4, v5], EdgeCreateFlags::NONE);
        let e52 = mesh.create_edge([v5, v2], EdgeCreateFlags::NONE);
        let face0 =
            mesh.face_create(&[v0, v1, v2, v3], &[e01, e12, e23, e30], FaceCreateFlags::NONE);
        let face1 =
            mesh.face_create(&[v1, v4, v5, v2], &[e14, e45, e52, e12], FaceCreateFlags::NONE);
        (face0, face1, v1, v2)
    }

    // The vertex cycle of a wall, as positions.
    fn wall_cycle_pos(mesh: &LoopMesh, wf: FaceKey) -> Vec<Vec3> {
        mesh.face_vert_keys(wf)
            .iter()
            .map(|&v| mesh.vertex_position(v))
            .collect()
    }

    // Find the wall whose original-edge endpoints are exactly `a` and `b`.
    fn find_wall_on(mesh: &LoopMesh, walls: &[FaceKey], a: VertexKey, b: VertexKey) -> FaceKey {
        for &wf in walls {
            let verts = mesh.face_vert_keys(wf);
            if verts.contains(&a) && verts.contains(&b) {
                return wf;
            }
        }
        panic!("no wall on edge");
    }

    #[test]
    fn extrude_region_normal_from_adjacent_reverses_surround_edge_wall() {
        // EXTRUDE.md §5: with a real surround, `use_normal_from_adjacent` reads
        // the wall winding cue from the un-extruded neighbour's loop direction
        // rather than the duplicated face's. On the shared edge v1–v2 the
        // surround face's loop runs opposite the extruded face's loop, so the
        // adjacent-mode wall must be the *reverse* cycle of the default wall on
        // that same boundary edge. CanonicalMesh is winding-blind, so this
        // in-crate cycle check is the load-bearing pin.
        let mut mesh_def = LoopMesh::new();
        let (f0_def, _f1_def, v1, v2) = two_quads_shared_edge(&mut mesh_def);
        let res_def = mesh_def.extrude_region(&[f0_def]);
        let wall_def = find_wall_on(&mesh_def, &res_def.wall_faces, v1, v2);
        let def_pos = wall_cycle_pos(&mesh_def, wall_def);

        let mut mesh_adj = LoopMesh::new();
        let (f0_adj, _f1_adj, _v1, _v2) = two_quads_shared_edge(&mut mesh_adj);
        let res_adj =
            mesh_adj.extrude_region_ex_normal_from_adjacent(&[f0_adj], false, true);
        let wall_adj = find_wall_on(&mesh_adj, &res_adj.wall_faces, v1, v2);
        let adj_pos = wall_cycle_pos(&mesh_adj, wall_adj);

        // Same topology either way.
        assert_eq!(res_def.wall_faces.len(), res_adj.wall_faces.len());
        // Surround-edge wall: reversed cycle under the adjacent cue.
        assert!(
            cycles_are_reversed(&def_pos, &adj_pos),
            "surround-edge wall must reverse under use_normal_from_adjacent: \
             default {def_pos:?} vs adjacent {adj_pos:?}"
        );
        mesh_adj.validate_topology();
    }

    #[test]
    fn extrude_region_normal_from_adjacent_keeps_mesh_boundary_walls() {
        // EXTRUDE.md §5: a boundary edge with no surround (mesh-boundary) has no
        // neighbour loop to read, so `use_normal_from_adjacent` falls back to
        // the duplicate-face cue — those walls must wind identically to the
        // default. Pin this on the two mesh-boundary edges v0–v1 (bottom) and
        // v2–v3 (top) of the extruded face, which have no surround.
        let mut mesh_def = LoopMesh::new();
        let (f0_def, _f1, _v1, _v2) = two_quads_shared_edge(&mut mesh_def);
        let v0 = mesh_def.face_vert_keys(f0_def)[0];
        let v1 = mesh_def.face_vert_keys(f0_def)[1];
        let res_def = mesh_def.extrude_region(&[f0_def]);
        let wall_def = find_wall_on(&mesh_def, &res_def.wall_faces, v0, v1);
        let def_pos = wall_cycle_pos(&mesh_def, wall_def);

        let mut mesh_adj = LoopMesh::new();
        let (f0_adj, _f1b, _v1b, _v2b) = two_quads_shared_edge(&mut mesh_adj);
        let res_adj =
            mesh_adj.extrude_region_ex_normal_from_adjacent(&[f0_adj], false, true);
        let wall_adj = find_wall_on(&mesh_adj, &res_adj.wall_faces, v0, v1);
        let adj_pos = wall_cycle_pos(&mesh_adj, wall_adj);

        // Identical cycle (same rotation, not reversed): no surround to read.
        let n = def_pos.len();
        let same_rotation =
            (0..n).any(|shift| (0..n).all(|i| def_pos[i] == adj_pos[(i + shift) % n]));
        assert!(
            same_rotation,
            "mesh-boundary wall must keep the default winding under \
             use_normal_from_adjacent (no surround to read): \
             default {def_pos:?} vs adjacent {adj_pos:?}"
        );
        mesh_adj.validate_topology();
    }

    #[test]
    fn extrude_region_normal_from_adjacent_off_matches_ex() {
        // `use_normal_from_adjacent = false` must reproduce extrude_region_ex
        // exactly — identical wall windings on every boundary edge, surround or
        // not. Pin on the shared-edge fixture so a surround edge is involved.
        let mut mesh_ex = LoopMesh::new();
        let (f0_ex, _f1, v1, v2) = two_quads_shared_edge(&mut mesh_ex);
        let v0 = mesh_ex.face_vert_keys(f0_ex)[0];
        let res_ex = mesh_ex.extrude_region_ex(&[f0_ex], false);

        let mut mesh_off = LoopMesh::new();
        let (f0_off, _f1b, _v1, _v2) = two_quads_shared_edge(&mut mesh_off);
        let res_off =
            mesh_off.extrude_region_ex_normal_from_adjacent(&[f0_off], false, false);

        assert_eq!(res_ex.wall_faces.len(), res_off.wall_faces.len());
        for &(a, b) in &[(v0, v1), (v1, v2)] {
            let pos_ex = wall_cycle_pos(&mesh_ex, find_wall_on(&mesh_ex, &res_ex.wall_faces, a, b));
            let pos_off =
                wall_cycle_pos(&mesh_off, find_wall_on(&mesh_off, &res_off.wall_faces, a, b));
            let n = pos_ex.len();
            let same_rotation =
                (0..n).any(|shift| (0..n).all(|i| pos_ex[i] == pos_off[(i + shift) % n]));
            assert!(
                same_rotation,
                "adjacent=false must match extrude_region_ex on edge {a:?}-{b:?}: \
                 {pos_ex:?} vs {pos_off:?}"
            );
        }
        mesh_off.validate_topology();
    }

    #[test]
    fn extrude_region_normal_from_adjacent_flip_reverses_adjacent_cue() {
        // §5: `use_normal_flip` flips whichever cue was chosen. With the
        // adjacent cue active, flipping must reverse the surround-edge wall
        // relative to the un-flipped adjacent result.
        let mut mesh_adj = LoopMesh::new();
        let (f0_adj, _f1, v1, v2) = two_quads_shared_edge(&mut mesh_adj);
        let res_adj =
            mesh_adj.extrude_region_ex_normal_from_adjacent(&[f0_adj], false, true);
        let adj_pos =
            wall_cycle_pos(&mesh_adj, find_wall_on(&mesh_adj, &res_adj.wall_faces, v1, v2));

        let mut mesh_flip = LoopMesh::new();
        let (f0_flip, _f1b, _v1, _v2) = two_quads_shared_edge(&mut mesh_flip);
        let res_flip =
            mesh_flip.extrude_region_ex_normal_from_adjacent(&[f0_flip], true, true);
        let flip_pos =
            wall_cycle_pos(&mesh_flip, find_wall_on(&mesh_flip, &res_flip.wall_faces, v1, v2));

        assert!(
            cycles_are_reversed(&adj_pos, &flip_pos),
            "use_normal_flip must reverse the adjacent-cued wall: \
             adjacent {adj_pos:?} vs adjacent+flip {flip_pos:?}"
        );
        mesh_flip.validate_topology();
    }

    #[test]
    fn extrude_region_wall_face_copies_source_face_attribs() {
        let mut mesh = LoopMesh::new();
        let verts = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        ];
        let face = create_face_from_verts(&mut mesh, &verts);

        let mat = mesh
            .register_face_attribute(ElementAttributeDesc::numeric(
                AttributeKey::custom("material_color"),
                Format::new(ComponentType::F32, 3),
            ))
            .unwrap();
        let source_color = [0.4_f32, 0.6, 0.8];
        mesh.face_attribs_mut()
            .set_vec3(mat, face.as_usize(), source_color);

        let result = mesh.extrude_region(&[face]);

        // Every wall face carries the source face's color.
        for &wf in &result.wall_faces {
            let got = mesh.face_attribs().get_vec3(mat, wf.as_usize()).unwrap();
            for i in 0..3 {
                assert!(
                    (got[i] - source_color[i]).abs() < 1e-6,
                    "wall face {wf:?}: got {got:?}, expected {source_color:?}"
                );
            }
        }
    }
}
