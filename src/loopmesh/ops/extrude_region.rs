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
        }
        let mut edge_info: Vec<Option<EdgeInfo>> = (0..edge_watermark).map(|_| None).collect();
        for slot in 0..edge_watermark {
            if !edge_in_region[slot] {
                continue;
            }
            let e = EdgeKey::from_usize(slot);
            let mut selected_loop_verts: SmallVec<[(LoopKey, VertexKey, FaceKey); 4]> =
                SmallVec::new();
            let mut has_unselected = false;
            let mut radial = self.radial_loops(e);
            while let Some(l) = radial.next(self) {
                let lref = self.loops.get(l);
                let face = lref.face;
                if self.face_op_flag_test(sel, face, 0) {
                    selected_loop_verts.push((l, lref.vertex, face));
                } else {
                    has_unselected = true;
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
