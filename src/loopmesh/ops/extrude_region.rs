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
    /// from the surround (unselected) face's two loops on the boundary edge,
    /// so the wall is attribute-transparent with the surround; both the
    /// original-side and duplicate-side corners take the same surround source
    /// at the matching endpoint. A wall on a boundary edge with no surround
    /// (a mesh-boundary radial-1 edge) keeps default loop customdata.
    /// The newly-created side ("vertical") edges between old and new verts
    /// take default attribute values.
    ///
    /// See `radial-mesh-guides/EXTRUDE_REGION.md` for the full algorithm
    /// description and parity contract.
    pub fn extrude_region(&mut self, faces: &[FaceKey]) -> ExtrudeResult {
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
            // The surround face's two corner-loops on this edge, keyed by the
            // endpoint vertex each corner sits at. A wall corner at endpoint
            // `v` (original or duplicate side) sources its loop customdata
            // from `surround_loop_at[i]` whose `.0 == v`. Absent when the
            // edge has no unselected (surround) face — e.g. a mesh-boundary
            // radial-1 edge — in which case wall loops keep default values.
            surround_loop_at: SmallVec<[(VertexKey, LoopKey); 2]>,
        }
        let mut edge_info: Vec<Option<EdgeInfo>> = (0..edge_watermark).map(|_| None).collect();
        for slot in 0..edge_watermark {
            if !edge_in_region[slot] {
                continue;
            }
            let e = EdgeKey::from_usize(slot);
            let mut selected_loop_verts: SmallVec<[(LoopKey, VertexKey, FaceKey); 4]> =
                SmallVec::new();
            let mut surround_loop_at: SmallVec<[(VertexKey, LoopKey); 2]> = SmallVec::new();
            let mut has_unselected = false;
            let mut radial = self.radial_loops(e);
            while let Some(l) = radial.next(self) {
                let lref = self.loops.get(l);
                let face = lref.face;
                if self.face_op_flag_test(sel, face, 0) {
                    selected_loop_verts.push((l, lref.vertex, face));
                } else {
                    has_unselected = true;
                    // First surround face wins as the customdata source. Its
                    // two corner-loops on this edge are `l` (anchored at
                    // `lref.vertex`) and `l`'s face-cycle successor (anchored
                    // at the edge's other endpoint).
                    if surround_loop_at.is_empty() {
                        let l_next = self.loops.get(l).face_loop_links.next;
                        let v_here = lref.vertex;
                        let v_next = self.loops.get(l_next).vertex;
                        surround_loop_at.push((v_here, l));
                        surround_loop_at.push((v_next, l_next));
                    }
                }
            }
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
                surround_loop_at,
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
            let e_old = EdgeKey::from_usize(slot);
            let [v0_old, v1_old] = self.edge_vertices(e_old);
            let v0_new = vert_to_dup[v0_old.as_usize()];
            let v1_new = vert_to_dup[v1_old.as_usize()];
            let e_new = edge_to_dup[slot];

            for &(_l_sel, l_v_orig, src_face) in &info.selected_loop_verts {
                // l_v_orig is the loop's PRE-MUTATION vertex; one of v0_old / v1_old.
                let (v_a_old, v_b_old, v_a_new, v_b_new) = if l_v_orig == v0_old {
                    (v0_old, v1_old, v0_new, v1_new)
                } else {
                    debug_assert_eq!(l_v_orig, v1_old);
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
                // the surround face: each of the four corners copies from the
                // surround loop sitting at the corner's *original* endpoint, so
                // the duplicate-side corners (v_b_new, v_a_new) pick up the same
                // values as the original-side corners they sit above.
                if do_loop_cd && !info.surround_loop_at.is_empty() {
                    let surround_source_at = |v: VertexKey| -> Option<LoopKey> {
                        info.surround_loop_at
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
                        if let Some(source) = surround_source_at(origin_vert) {
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
    use glam::Vec3;

    use crate::loopmesh::test_support::{create_face_from_verts, mesh_add_cube};
    use crate::loopmesh::*;

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
        // original-side corners). The other three walls sit on mesh-boundary
        // edges with no surround and keep default (zero) loop UVs.
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

        // A mesh-boundary wall (e.g. on edge v0–v3, no surround) keeps default
        // zero UVs on every corner.
        let mut boundary_wall = None;
        for &wf in &result.wall_faces {
            let verts = mesh.face_vert_keys(wf);
            if verts.contains(&v0) && verts.contains(&v3) {
                boundary_wall = Some(wf);
                break;
            }
        }
        let boundary_wall = boundary_wall.expect("a wall sits on the v0–v3 mesh-boundary edge");
        for l in mesh.iter_face_loops(boundary_wall).collect::<Vec<_>>() {
            let got = mesh.loop_attribs().get_vec2(uv, l.as_usize()).unwrap();
            assert_eq!(got, [0.0, 0.0], "mesh-boundary wall has no surround source");
        }

        mesh.validate_topology();
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
