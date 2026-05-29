use smallvec::SmallVec;

use crate::loopmesh::*;

impl LoopMesh {
    /// Extrude each selected face independently (discrete / per-face extrude).
    ///
    /// Unlike [`extrude_region`](Self::extrude_region), there is no notion of a
    /// boundary between selected and unselected faces: every input face is its
    /// own one-face region. Each input face is copied in place to a fresh,
    /// structurally-disconnected duplicate (new verts, new edges, customdata
    /// copied), every loop of the duplicate is ringed by a four-vert wall
    /// spanning the original edge to its duplicate, and finally the original
    /// face header is removed (its edges and verts survive as the bottom of the
    /// walls).
    ///
    /// The consequence is per-face isolation: two adjacent input faces produce
    /// *two* lifted caps, each with its own wall on the formerly-shared edge —
    /// they split apart rather than staying one patch (contrast region extrude,
    /// which keeps a single lifted patch with no wall on the interior edge).
    ///
    /// Customdata: every new vertex copies its source vertex's attribs; every
    /// duplicated edge copies its source edge's attribs; the duplicate cap copies
    /// the original face's face/loop attribs corner-for-corner. Each wall face
    /// inherits the original face's attribs, and each wall corner sources its
    /// per-corner loop customdata from the original face's own loop at that
    /// corner's original vertex (the §4.3 column pattern, with the original face
    /// as the source since discrete extrude has no surround context). The
    /// vertical edges between original and duplicate verts take default attribute
    /// values.
    ///
    /// This is a pure-topology op; the caller displaces the new vertices after it
    /// returns.
    ///
    /// See `radial-mesh-guides/operators/EXTRUDE.md` §9 for the full algorithm
    /// and parity contract.
    pub fn extrude_discrete_faces(&mut self, faces: &[FaceKey]) -> ExtrudeResult {
        self.extrude_discrete_faces_ex(faces, false)
    }

    /// [`extrude_discrete_faces`](Self::extrude_discrete_faces) with the
    /// `use_normal_flip` parameter exposed: when set, each wall's winding is
    /// reversed (EXTRUDE.md §4.2 / §5).
    pub fn extrude_discrete_faces_ex(
        &mut self,
        faces: &[FaceKey],
        use_normal_flip: bool,
    ) -> ExtrudeResult {
        let do_loop_cd = !self.loop_attribs.is_empty();

        let mut vert_map_out: Vec<(VertexKey, VertexKey)> = Vec::new();
        let mut edge_map_out: Vec<(EdgeKey, EdgeKey)> = Vec::new();
        let mut side_edges_out: Vec<EdgeKey> = Vec::new();
        let mut wall_faces: Vec<FaceKey> = Vec::new();
        let mut originals_to_delete: SmallVec<[FaceKey; 16]> = SmallVec::new();

        for &face in faces {
            // ---- Step 1: snapshot the original face's loop ring. Each entry is
            //              the loop, its anchor vertex, and the edge it lies on.
            //              Snapshot before any mutation — building the cap and
            //              walls reweaves radial cycles. ----
            let mut org_loops: SmallVec<[(LoopKey, VertexKey, EdgeKey); 8]> = SmallVec::new();
            let mut it = self.face_loops(face);
            while let Some(l) = it.next(self) {
                let lref = self.loops.get(l);
                org_loops.push((l, lref.vertex, lref.edge));
            }
            let perimeter = org_loops.len();
            if perimeter == 0 {
                continue;
            }

            // ---- Step 2: duplicate verts and edges for this face only. The
            //              duplicate set is disconnected from the rest of the
            //              mesh and from any other input face's duplicates. ----
            let mut dup_verts: SmallVec<[VertexKey; 8]> = SmallVec::new();
            for &(_l, v_org, _e) in &org_loops {
                let dup = self.create_vertex(self.vertex_position(v_org));
                self.vertex_attribs.copy_into(dup.as_usize(), v_org.as_usize());
                vert_map_out.push((v_org, dup));
                dup_verts.push(dup);
            }
            // Each perimeter edge connects loop `i`'s vertex to loop `i+1`'s
            // vertex (face-cycle ordering). The matching duplicate edge spans the
            // corresponding duplicate verts.
            let mut dup_edges: SmallVec<[EdgeKey; 8]> = SmallVec::new();
            for i in 0..perimeter {
                let next = (i + 1) % perimeter;
                let e_org = org_loops[i].2;
                let dup = self.create_edge([dup_verts[i], dup_verts[next]], EdgeCreateFlags::NONE);
                self.edge_attribs.copy_into(dup.as_usize(), e_org.as_usize());
                edge_map_out.push((e_org, dup));
                dup_edges.push(dup);
            }

            // ---- Step 3: build the duplicate cap face, in the same cycle order
            //              as the original, and copy face + per-corner loop
            //              customdata across. ----
            let cap_verts: SmallVec<[VertexKey; 8]> = dup_verts.iter().copied().collect();
            let cap = self.face_create(&cap_verts, &dup_edges, FaceCreateFlags::NONE);
            self.face_attribs.copy_into(cap.as_usize(), face.as_usize());
            if do_loop_cd {
                let mut cap_loop = self.faces.get(cap).first_edge_loop;
                for &(l_org, _v, _e) in &org_loops {
                    self.loop_attribs
                        .copy_into(cap_loop.as_usize(), l_org.as_usize());
                    cap_loop = self.loops.get(cap_loop).face_loop_links.next;
                }
            }

            // ---- Step 4: build one wall per perimeter edge. Vertical edges are
            //              created once per original vertex and shared between the
            //              two walls of this face that meet at that vertex. ----
            let mut side_edge_at_corner: SmallVec<[EdgeKey; 8]> =
                (0..perimeter).map(|_| EdgeKey::NULL).collect();
            for i in 0..perimeter {
                let next = (i + 1) % perimeter;
                let (l_org, va_org, e_org) = org_loops[i];
                let vb_org = org_loops[next].1;
                let va_dup = dup_verts[i];
                let vb_dup = dup_verts[next];
                let e_dup = dup_edges[i];

                // Winding (EXTRUDE.md §5): orient as a loop of the duplicate edge
                // so the wall's loop on e_dup runs opposite the cap's loop. With
                // no surround, `use_normal_from_adjacent` is irrelevant; the cue
                // is always the duplicate face's loop direction.
                let [e_dup_v0, _e_dup_v1] = self.edge_vertices(e_dup);
                let dup_first_loop = self.edges.get(e_dup).radial_link;
                let edge_normal_flip = match dup_first_loop {
                    Some(l) => self.loops.get(l).vertex != e_dup_v0,
                    None => {
                        // The cap is always built before the walls, so the dup
                        // edge always has the cap's loop; this branch is a guard.
                        true
                    }
                };

                // §5 vert order. The "v1" of each edge is its first endpoint;
                // map the duplicate-edge ordering back onto original verts via
                // the loop ring (loop `i` is anchored at `va_org`/`va_dup`).
                let (wa_org, wb_org, wa_dup, wb_dup) = if edge_normal_flip == use_normal_flip {
                    (va_org, vb_org, va_dup, vb_dup)
                } else {
                    (vb_org, va_org, vb_dup, va_dup)
                };

                // Vertical edges, shared at each original corner of this face.
                let corner_a = if wa_org == va_org { i } else { next };
                let corner_b = if corner_a == i { next } else { i };
                let side_a = if side_edge_at_corner[corner_a].is_null() {
                    let e = self.create_edge([wa_org, wa_dup], EdgeCreateFlags::NONE);
                    side_edge_at_corner[corner_a] = e;
                    e
                } else {
                    side_edge_at_corner[corner_a]
                };
                let side_b = if side_edge_at_corner[corner_b].is_null() {
                    let e = self.create_edge([wb_org, wb_dup], EdgeCreateFlags::NONE);
                    side_edge_at_corner[corner_b] = e;
                    e
                } else {
                    side_edge_at_corner[corner_b]
                };

                // Wall quad: wa_org -> wb_org -> wb_dup -> wa_dup.
                let wall = self.face_create(
                    &[wa_org, wb_org, wb_dup, wa_dup],
                    &[e_org, side_b, e_dup, side_a],
                    FaceCreateFlags::NONE,
                );
                self.face_attribs.copy_into(wall.as_usize(), face.as_usize());

                // Wall loop customdata (§4.3): the source face is the original
                // face itself. Its two loops on e_org are `l_org` (at `va_org`)
                // and `l_org`'s face-successor (at `vb_org`). Each wall column
                // (near + far corner over the same original vertex) copies the
                // source loop at that vertex.
                if do_loop_cd {
                    let l_org_next = self.loops.get(l_org).face_loop_links.next;
                    let source_at = |v: VertexKey| -> LoopKey {
                        if v == va_org {
                            l_org
                        } else {
                            debug_assert_eq!(v, vb_org);
                            l_org_next
                        }
                    };
                    let wall_loop_0 = self.faces.get(wall).first_edge_loop;
                    let wall_loop_1 = self.loops.get(wall_loop_0).face_loop_links.next;
                    let wall_loop_2 = self.loops.get(wall_loop_1).face_loop_links.next;
                    let wall_loop_3 = self.loops.get(wall_loop_2).face_loop_links.next;
                    for (wall_loop, origin_vert) in [
                        (wall_loop_0, wa_org),
                        (wall_loop_1, wb_org),
                        (wall_loop_2, wb_org),
                        (wall_loop_3, wa_org),
                    ] {
                        let source = source_at(origin_vert);
                        self.loop_attribs
                            .copy_into(wall_loop.as_usize(), source.as_usize());
                    }
                }

                wall_faces.push(wall);
            }

            for e in side_edge_at_corner {
                if !e.is_null() {
                    side_edges_out.push(e);
                }
            }

            originals_to_delete.push(face);
        }

        // ---- Step 5: remove the original face headers only (DEL_ONLYFACES);
        //              their edges and verts survive as the bottom of the walls.
        // ----
        self.delete(&originals_to_delete[..], DeleteContext::FacesOnly);

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

    use crate::loopmesh::test_support::create_face_from_verts;
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

    #[test]
    fn discrete_single_quad_lifts_with_own_walls() {
        // A lone quad: discrete extrude duplicates its 4 verts + 4 edges, builds
        // the cap, rings it with 4 walls, and deletes the original face header.
        let mut mesh = LoopMesh::new();
        let face = unit_quad(&mut mesh);
        let source_verts: Vec<VertexKey> = mesh.face_vert_keys(face);

        let result = mesh.extrude_discrete_faces(&[face]);

        assert_eq!(result.vert_map.len(), 4, "4 verts duplicated");
        assert_eq!(result.edge_map.len(), 4, "4 edges duplicated");
        assert_eq!(result.side_edges.len(), 4, "4 vertical edges");
        assert_eq!(result.wall_faces.len(), 4, "4 walls, one per perimeter edge");

        // 4 original verts + 4 duplicates.
        assert_eq!(mesh.totvert(), 8);
        // 4 walls + 1 cap; the original face header is gone.
        assert_eq!(mesh.totface(), 5);
        assert!(
            !mesh.face_keys().contains(&face),
            "original face header is removed (DEL_ONLYFACES analog)"
        );

        // The original verts survive (bottom of the walls) and are still
        // referenced by the walls.
        let live_verts: HashSet<VertexKey> = mesh.vert_keys().into_iter().collect();
        let mut wall_verts = HashSet::new();
        for &wf in &result.wall_faces {
            for v in mesh.face_vert_keys(wf) {
                wall_verts.insert(v);
            }
        }
        for v in &source_verts {
            assert!(live_verts.contains(v), "original vert survives");
            assert!(wall_verts.contains(v), "wall references original vert");
        }
        mesh.validate_topology();
    }

    #[test]
    fn discrete_two_adjacent_faces_split_apart() {
        // The load-bearing isolation case. Two quads share edge v1–v2. Region
        // extrude would keep one patch and wall only the outer perimeter; the
        // discrete variant treats each face independently, so the formerly
        // shared edge gets *two* walls (one per face's own duplicate of it) and
        // the two caps share no geometry.
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
        let f0 = mesh.face_create(&[v0, v1, v2, v3], &[e01, e12, e23, e30], FaceCreateFlags::NONE);
        let f1 = mesh.face_create(&[v1, v4, v5, v2], &[e14, e45, e52, e12], FaceCreateFlags::NONE);

        let result = mesh.extrude_discrete_faces(&[f0, f1]);

        // Each face independently duplicates ALL of its verts/edges — including
        // its own copy of the shared edge v1–v2 and the shared verts v1, v2. So
        // the dup count is 8 verts (4 per face, with v1/v2 duplicated twice) and
        // 8 edges (4 per face, with e12 duplicated twice).
        assert_eq!(result.vert_map.len(), 8, "each face dups all 4 of its verts");
        assert_eq!(result.edge_map.len(), 8, "each face dups all 4 of its edges");
        // 4 walls per face = 8 walls; the shared edge is walled twice (once by
        // each face's own duplicate), which is the isolation signature.
        assert_eq!(result.wall_faces.len(), 8, "each face rings itself with 4 walls");

        // Exactly two walls sit on the original shared edge v1–v2 — proof the
        // faces split apart rather than merging into one patch.
        let mut walls_on_shared = 0;
        for &wf in &result.wall_faces {
            let vs = mesh.face_vert_keys(wf);
            if vs.contains(&v1) && vs.contains(&v2) {
                walls_on_shared += 1;
            }
        }
        assert_eq!(
            walls_on_shared, 2,
            "the formerly-shared edge is walled by BOTH faces (split apart)"
        );

        // Both originals removed; the two caps + 8 walls remain.
        let live_faces: HashSet<FaceKey> = mesh.face_keys().into_iter().collect();
        assert!(!live_faces.contains(&f0) && !live_faces.contains(&f1));
        assert_eq!(mesh.totface(), 10, "2 caps + 8 walls");
        mesh.validate_topology();
    }

    #[test]
    fn discrete_triangle_ngon_path() {
        // n-gon != 4: a single triangle. 3 verts/edges duplicated, 3 walls,
        // 3 vertical edges, 1 cap.
        let mut mesh = LoopMesh::new();
        let verts = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        ];
        let tri = create_face_from_verts(&mut mesh, &verts);

        let result = mesh.extrude_discrete_faces(&[tri]);

        assert_eq!(result.vert_map.len(), 3);
        assert_eq!(result.edge_map.len(), 3);
        assert_eq!(result.side_edges.len(), 3);
        assert_eq!(result.wall_faces.len(), 3);
        assert_eq!(mesh.totface(), 4, "3 walls + 1 cap");
        for &wf in &result.wall_faces {
            assert_eq!(mesh.face_vert_keys(wf).len(), 4, "each wall is a quad");
        }
        mesh.validate_topology();
    }

    #[test]
    fn discrete_normal_flip_reverses_wall_winding() {
        // The use_normal_flip variant must reverse each wall's vertex order
        // relative to the default. We extrude the same quad twice (default and
        // flipped) and assert each wall's perimeter is the reverse cycle of its
        // default counterpart.
        let mut mesh_default = LoopMesh::new();
        let face_default = unit_quad(&mut mesh_default);
        let res_default = mesh_default.extrude_discrete_faces(&[face_default]);

        let mut mesh_flip = LoopMesh::new();
        let face_flip = unit_quad(&mut mesh_flip);
        let res_flip = mesh_flip.extrude_discrete_faces_ex(&[face_flip], true);

        // Topology counts are identical; only the winding differs.
        assert_eq!(res_default.wall_faces.len(), res_flip.wall_faces.len());

        // For each default wall, find the flipped wall over the same original
        // edge (same original-vertex pair) and check the vertex cycle is
        // reversed.
        let key_pair = |mesh: &LoopMesh, wf: FaceKey| -> (VertexKey, VertexKey) {
            // The two original (lower) verts of the wall are the ones whose
            // positions match the z=0 plane.
            let mut lower: Vec<VertexKey> = mesh
                .face_vert_keys(wf)
                .into_iter()
                .filter(|&v| mesh.vertex_position(v).z == 0.0)
                .collect();
            lower.sort_by_key(|v| v.as_usize());
            (lower[0], lower[1])
        };

        for &wf_def in &res_default.wall_faces {
            let def_cycle = mesh_default.face_vert_keys(wf_def);
            let def_lower = key_pair(&mesh_default, wf_def);
            // Lower-vert pair identifies the edge; positions are identical
            // across the two meshes, so match by position.
            let def_pos: Vec<Vec3> = def_cycle
                .iter()
                .map(|&v| mesh_default.vertex_position(v))
                .collect();

            let mut matched = false;
            for &wf_flip in &res_flip.wall_faces {
                let flip_lower = key_pair(&mesh_flip, wf_flip);
                let flip_lower_pos: HashSet<[u32; 3]> = [flip_lower.0, flip_lower.1]
                    .iter()
                    .map(|&v| pos_key(mesh_flip.vertex_position(v)))
                    .collect();
                let def_lower_pos: HashSet<[u32; 3]> = [def_lower.0, def_lower.1]
                    .iter()
                    .map(|&v| pos_key(mesh_default.vertex_position(v)))
                    .collect();
                if flip_lower_pos != def_lower_pos {
                    continue;
                }
                // Same edge. The flipped wall's position cycle must be the
                // reverse of the default's.
                let flip_pos: Vec<Vec3> = mesh_flip
                    .face_vert_keys(wf_flip)
                    .iter()
                    .map(|&v| mesh_flip.vertex_position(v))
                    .collect();
                assert!(
                    cycles_are_reversed(&def_pos, &flip_pos),
                    "flip wall must be reverse cycle of default wall"
                );
                matched = true;
                break;
            }
            assert!(matched, "every default wall has a flipped counterpart");
        }

        mesh_flip.validate_topology();
    }

    #[test]
    fn discrete_copies_vertex_and_face_customdata() {
        let mut mesh = LoopMesh::new();
        let face = unit_quad(&mut mesh);
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
            mesh.vertex_attribs_mut().set_vec3(color, v.as_usize(), palette[i]);
        }

        let mat = mesh
            .register_face_attribute(ElementAttributeDesc::numeric(
                AttributeKey::custom("material_color"),
                Format::new(ComponentType::F32, 3),
            ))
            .unwrap();
        let face_color = [0.2_f32, 0.5, 0.9];
        mesh.face_attribs_mut().set_vec3(mat, face.as_usize(), face_color);

        let result = mesh.extrude_discrete_faces(&[face]);

        for &(src, dup) in &result.vert_map {
            let src_val = mesh.vertex_attribs().get_vec3(color, src.as_usize()).unwrap();
            let dup_val = mesh.vertex_attribs().get_vec3(color, dup.as_usize()).unwrap();
            assert_eq!(src_val, dup_val, "duplicate vert carries source color");
        }
        // Every wall face inherits the original face's material color.
        for &wf in &result.wall_faces {
            let got = mesh.face_attribs().get_vec3(mat, wf.as_usize()).unwrap();
            for i in 0..3 {
                assert!((got[i] - face_color[i]).abs() < 1e-6, "wall inherits face color");
            }
        }
    }

    fn pos_key(p: Vec3) -> [u32; 3] {
        [p.x.to_bits(), p.y.to_bits(), p.z.to_bits()]
    }

    // Two 4-cycles describe the same quad with opposite winding iff one is the
    // reverse rotation of the other.
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
}
