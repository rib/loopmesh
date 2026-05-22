use std::collections::HashMap;

use glam::Vec3;

use crate::loopmesh::*;

/// Parameters controlling [`LoopMesh::inset_region`]. Each field gates one
/// of the toggles described in `radial-mesh-guides/operators/INSET.md` (see
/// the operator's rustdoc for which guide section each one governs).
#[derive(Clone, Copy)]
pub struct InsetRegionParams<'a> {
    pub thickness: f32,
    pub depth: f32,
    pub use_boundary: bool,
    pub use_even_offset: bool,
    pub use_interpolate: bool,
    pub use_relative_offset: bool,
    pub use_edge_rail: bool,
    pub use_outset: bool,
    pub faces_exclude: &'a [FaceKey],
}

impl<'a> Default for InsetRegionParams<'a> {
    fn default() -> Self {
        Self {
            thickness: 0.0,
            depth: 0.0,
            use_boundary: true,
            use_even_offset: true,
            use_interpolate: false,
            use_relative_offset: false,
            use_edge_rail: false,
            use_outset: false,
            faces_exclude: &[],
        }
    }
}

#[derive(Default)]
pub struct InsetRegionResult {
    /// The newly-created wall faces, one per seam edge (and one per selected
    /// loop at non-manifold seams).
    pub wall_faces: Vec<FaceKey>,
}

struct SplitEdge {
    /// The edge identity that ends up carrying the *untagged* side after seam
    /// separation. For a mesh-boundary seam this is a freshly-fabricated wire
    /// edge; otherwise it is the original edge identity.
    e_old: EdgeKey,
    /// The edge identity that ends up carrying the *tagged* side after seam
    /// separation. Equals the original edge identity except when
    /// `is_mesh_boundary` is set.
    e_new: EdgeKey,
    /// The tagged loop on this seam edge after separation. Always sits on
    /// `e_new`.
    seam_loop: LoopKey,
    /// The tagged face the seam loop belongs to. Used as the customdata
    /// source for the wall face.
    face: FaceKey,
    /// Inward edge tangent in `face`'s plane, oriented to point into the
    /// tagged face. Filled in during phase two.
    tangent: Vec3,
    /// Endpoint shared with `e_new`/`e_old` in seam-loop traversal order.
    /// `v0` is `seam_loop.vertex`; `v1` is the next vertex around the face.
    v0: VertexKey,
    v1: VertexKey,
    /// True if this seam edge is on the mesh boundary (radial-1, tagged-side
    /// only); `use_boundary` controls whether such edges become seams.
    is_mesh_boundary: bool,
    /// Pre-inset length of the edge — used by `use_relative_offset`.
    length: f32,
}

impl LoopMesh {
    /// Inset a region of selected faces — shrink their outer boundary inward
    /// and stitch a ribbon of "wall" quads into the gap.
    ///
    /// The implementation follows the five-phase pipeline described in
    /// `radial-mesh-guides/operators/INSET.md`:
    ///
    /// 1. **Classify** every edge touching a tagged face. An edge is a *seam*
    ///    iff its radial cycle contains exactly one tagged loop and at least
    ///    one untagged loop, or (when `use_boundary` is set) its radial is
    ///    radial-1 and its only loop is tagged. Edges whose radial is entirely
    ///    tagged or entirely untagged are not seams; edges with two-or-more
    ///    tagged loops on a mixed radial are *declined* and left un-inset.
    /// 2. **Seam-separate** each seam edge with `euler_edge_separate` so the
    ///    tagged loop sits on a fresh `e_new` and the original `e_old` keeps
    ///    the untagged loops. For mesh-boundary seams the original edge
    ///    already has only the tagged loop on it; instead, a fresh parallel
    ///    edge is fabricated as `e_old`.
    /// 3. **Vertex-separate + offset.** For each endpoint of each seam edge,
    ///    `euler_vert_separate` splits the vertex into one fan-instance per
    ///    distinct face-fan. The fan-instance that carries the tagged-side
    ///    seam edges (its disk contains one or more `e_new` keys) is the
    ///    tagged corner: its position is the original position plus a
    ///    bisector offset of the two seam-edge tangents (one-tangent for the
    ///    single-seam corner case), optionally scaled by `1/cos(half-angle)`
    ///    when `use_even_offset` is set.
    /// 4. *(Not yet implemented: face re-interpolation when `use_interpolate`
    ///    is set; the depth pass when `depth != 0.0`; `use_outset`,
    ///    `use_edge_rail`, and `use_relative_offset` offset modes.)*
    /// 5. **Build wall quads.** For each seam edge, build a face spanning
    ///    `(v0_old, v1_old, v1_new, v0_new)` with edges `(e_old, side_v1,
    ///    e_new, side_v0)`. Side edges between matched tagged/untagged
    ///    endpoints are cached so adjacent walls share them.
    pub fn inset_region(
        &mut self,
        faces: &[FaceKey],
        params: InsetRegionParams<'_>,
    ) -> InsetRegionResult {
        if faces.is_empty() {
            return InsetRegionResult::default();
        }

        // Phase 0: tag the selected faces. For `use_outset`, invert the tag
        // — every face *except* the input set (and `faces_exclude`) becomes
        // the inset region; `use_boundary` is forced off in that mode since
        // the inverted selection includes mesh-boundary faces.
        let face_tag = self.push_face_op_flag_layer();
        if params.use_outset {
            let exclude: std::collections::HashSet<FaceKey> = faces
                .iter()
                .chain(params.faces_exclude.iter())
                .copied()
                .collect();
            for f in self.face_keys() {
                if !exclude.contains(&f) {
                    self.face_op_flag_set(face_tag, f, 0);
                }
            }
        } else {
            for &f in faces {
                self.face_op_flag_set(face_tag, f, 0);
            }
        }

        // Phase 1: classify every edge incident to a tagged face and build
        // the SplitEdge[] array.
        let mut split_edges: Vec<SplitEdge> = Vec::new();
        let edge_watermark = self.edges.key_watermark();
        let mut edge_seen: Vec<bool> = vec![false; edge_watermark];
        for &f in faces {
            let face_loops: Vec<LoopKey> = {
                let mut it = self.face_loops(f);
                let mut acc = Vec::new();
                while let Some(l) = it.next(self) {
                    acc.push(l);
                }
                acc
            };
            for l in face_loops {
                let e = self.loop_edge(l);
                let idx = e.as_usize();
                if edge_seen[idx] {
                    continue;
                }
                edge_seen[idx] = true;

                // Walk the radial: count tagged/untagged loops and remember
                // the first tagged loop we see.
                let mut tagged_count = 0usize;
                let mut untagged_count = 0usize;
                let mut first_tagged: Option<LoopKey> = None;
                let radial: Vec<LoopKey> = self.edge_radial_iter(e).collect();
                for rl in radial {
                    let face = self.loop_face(rl);
                    if self.face_op_flag_test(face_tag, face, 0) {
                        if first_tagged.is_none() {
                            first_tagged = Some(rl);
                        }
                        tagged_count += 1;
                    } else {
                        untagged_count += 1;
                    }
                }

                let (is_seam, is_mesh_boundary, seam_loop) =
                    if tagged_count == 1 && untagged_count >= 1 {
                        (true, false, first_tagged.unwrap())
                    } else if tagged_count == 1
                        && untagged_count == 0
                        && params.use_boundary
                        && !params.use_outset
                    {
                        (true, true, first_tagged.unwrap())
                    } else {
                        (false, false, LoopKey(0))
                    };
                if !is_seam {
                    continue;
                }

                let v0 = self.loop_vertex(seam_loop);
                let v1 = self.loop_vertex(self.loop_next(seam_loop));
                let length = (self.vertex_position(v0) - self.vertex_position(v1)).length();
                let face = self.loop_face(seam_loop);
                split_edges.push(SplitEdge {
                    e_old: e,
                    e_new: e,
                    seam_loop,
                    face,
                    tangent: Vec3::ZERO,
                    v0,
                    v1,
                    is_mesh_boundary,
                    length,
                });
            }
        }

        if split_edges.is_empty() {
            self.pop_face_op_flag_layer(face_tag);
            return InsetRegionResult::default();
        }

        // Phase 2: snapshot face normals (the tangent calculation needs them),
        // then seam-separate every seam edge.
        self.calc_face_normals();
        for i in 0..split_edges.len() {
            let s = &split_edges[i];
            let face = s.face;
            let n = self.face_normal(face);
            let p0 = self.vertex_position(s.v0);
            let p1 = self.vertex_position(s.v1);
            // edge_dir runs v0 → v1 in seam-loop order; rotating 90° about the
            // face normal points it into the face interior.
            let edge_dir = (p1 - p0).normalize_or_zero();
            let tangent = n.cross(edge_dir).normalize_or_zero();
            split_edges[i].tangent = tangent;

            if !split_edges[i].is_mesh_boundary {
                let e_orig = split_edges[i].e_old;
                let seam = split_edges[i].seam_loop;
                let e_new = self.euler_edge_separate(e_orig, seam);
                split_edges[i].e_new = e_new;
            } else {
                // Mesh boundary: fabricate a parallel wire edge to act as the
                // untagged side. The seam loop stays on the original edge.
                let s = &split_edges[i];
                let new_old = self.create_edge([s.v0, s.v1], EdgeCreateFlags::NONE);
                self.edge_attribs
                    .copy_into(new_old.as_usize(), s.e_old.as_usize());
                split_edges[i].e_new = split_edges[i].e_old;
                split_edges[i].e_old = new_old;
            }
        }

        // Build the e_new → split-edge-index lookup for phase three.
        let mut e_new_to_split: HashMap<EdgeKey, usize> = HashMap::new();
        let mut e_old_to_split: HashMap<EdgeKey, usize> = HashMap::new();
        for (i, s) in split_edges.iter().enumerate() {
            e_new_to_split.insert(s.e_new, i);
            e_old_to_split.insert(s.e_old, i);
        }

        // Gather the set of pre-mutation endpoint vertices.
        let mut endpoint_set: Vec<VertexKey> = Vec::with_capacity(split_edges.len() * 2);
        for s in &split_edges {
            endpoint_set.push(s.v0);
            endpoint_set.push(s.v1);
        }
        endpoint_set.sort_unstable_by_key(|v| v.as_usize());
        endpoint_set.dedup();

        // Phase 3: at each endpoint, vert-separate and offset the tagged fan.
        // `endpoint_tagged_v[(split_edge_idx, role)]` → the post-separate
        // tagged-side vertex for split-edge `split_edge_idx` at endpoint
        // `role` (0 = v0, 1 = v1). Same for untagged.
        let mut endpoint_tagged_v: HashMap<(usize, usize), VertexKey> = HashMap::new();
        let mut endpoint_untagged_v: HashMap<(usize, usize), VertexKey> = HashMap::new();

        for &v_pre in &endpoint_set {
            let pre_pos = self.vertex_position(v_pre);
            let fans = self.euler_vert_separate(v_pre);

            // Vertex-glue step (INSET.md §5.4). When vert_separate produces
            // three or more fan-instances at one source vertex — only
            // reachable on mesh-boundary seams (each fabricated wire edge
            // becomes its own fan) or at non-manifold pinches — collect every
            // fan-instance that carries *no* tagged-side seam edge and merge
            // them into a single vertex via vert_splice.
            let mut glue_candidates: Vec<VertexKey> = Vec::new();
            if fans.len() >= 3 {
                for &fv in &fans {
                    let has_e_new = self
                        .vertex_disk_iter(fv)
                        .any(|e| e_new_to_split.contains_key(&e));
                    if !has_e_new {
                        glue_candidates.push(fv);
                    }
                }
                if glue_candidates.len() >= 2 {
                    let dst = glue_candidates[0];
                    for &src in &glue_candidates[1..] {
                        self.euler_vert_splice(dst, src);
                    }
                }
            }
            // Re-collect the surviving fan-instances after splicing — every
            // glue_candidates[1..] was deleted.
            let surviving_fans: Vec<VertexKey> = if glue_candidates.len() >= 2 {
                let dead: std::collections::HashSet<VertexKey> =
                    glue_candidates[1..].iter().copied().collect();
                fans.into_iter().filter(|fv| !dead.contains(fv)).collect()
            } else {
                fans
            };

            for &fv in &surviving_fans {
                let disk: Vec<EdgeKey> = self.vertex_disk_iter(fv).collect();
                let mut seam_new: Vec<usize> = Vec::new();
                let mut seam_old: Vec<usize> = Vec::new();
                for e in &disk {
                    if let Some(&idx) = e_new_to_split.get(e) {
                        seam_new.push(idx);
                    }
                    if let Some(&idx) = e_old_to_split.get(e) {
                        seam_old.push(idx);
                    }
                }

                // Record vertex identities for wall construction.
                for &idx in &seam_new {
                    let role = if split_edges[idx].v0 == v_pre { 0 } else { 1 };
                    endpoint_tagged_v.insert((idx, role), fv);
                }
                for &idx in &seam_old {
                    let role = if split_edges[idx].v0 == v_pre { 0 } else { 1 };
                    endpoint_untagged_v.insert((idx, role), fv);
                }

                // A fan-instance is the *tagged* corner iff it carries one or
                // more e_new edges. Apply the in-plane offset.
                if seam_new.is_empty() {
                    continue;
                }
                let (tvec, relative) = match seam_new.len() {
                    1 => {
                        // One-seam corner — only reachable at a mesh-boundary
                        // seam vertex (INSET.md §5.6). Use the tangent
                        // directly; the even-offset scaling for this case
                        // is not yet wired up.
                        let s = &split_edges[seam_new[0]];
                        (s.tangent, s.length)
                    }
                    _ => {
                        // Two-or-more-seam case (INSET.md §5.1, §5.2).
                        // First-two-wins for 3+ seams.
                        let sa = &split_edges[seam_new[0]];
                        let sb = &split_edges[seam_new[1]];
                        let bisector = (sa.tangent + sb.tangent).normalize_or_zero();
                        let dir = if params.use_even_offset {
                            let cos_half = sa.tangent.dot(bisector).abs();
                            if cos_half > 1.0e-6 {
                                bisector / cos_half
                            } else {
                                bisector
                            }
                        } else {
                            bisector
                        };
                        (dir, 0.5 * (sa.length + sb.length))
                    }
                };
                let scale = if params.use_relative_offset {
                    relative
                } else {
                    1.0
                };
                let new_pos = pre_pos + tvec * params.thickness * scale;
                self.set_vertex_position(fv, new_pos);
            }
        }

        // Phase 5: build wall faces.
        let mut wall_faces: Vec<FaceKey> = Vec::with_capacity(split_edges.len());
        let mut side_edge_cache: HashMap<(VertexKey, VertexKey), EdgeKey> = HashMap::new();

        for (i, s) in split_edges.iter().enumerate() {
            let v0_tag = endpoint_tagged_v.get(&(i, 0)).copied().unwrap_or(s.v0);
            let v1_tag = endpoint_tagged_v.get(&(i, 1)).copied().unwrap_or(s.v1);
            let v0_un = endpoint_untagged_v.get(&(i, 0)).copied().unwrap_or(s.v0);
            let v1_un = endpoint_untagged_v.get(&(i, 1)).copied().unwrap_or(s.v1);

            // Side edges between matched tagged/untagged endpoints. Shared
            // between adjacent walls at the same source vertex.
            let side_v0 =
                if v0_tag == v0_un {
                    None
                } else {
                    Some(*side_edge_cache.entry((v0_un, v0_tag)).or_insert_with(|| {
                        self.create_edge([v0_un, v0_tag], EdgeCreateFlags::NONE)
                    }))
                };
            let side_v1 =
                if v1_tag == v1_un {
                    None
                } else {
                    Some(*side_edge_cache.entry((v1_un, v1_tag)).or_insert_with(|| {
                        self.create_edge([v1_un, v1_tag], EdgeCreateFlags::NONE)
                    }))
                };

            // Degenerate wall (both endpoints coincide) — skip per INSET.md
            // §6.1. Reachable only on pathological inputs where neither
            // endpoint produced an extra fan.
            if side_v0.is_none() && side_v1.is_none() {
                continue;
            }

            // Assemble verts and edges in winding order so the wall's normal
            // points away from the tagged face (i.e. matches the untagged
            // side's orientation).
            //
            // Vert order: v0_un → v1_un → v1_tag → v0_tag.
            // Edge order: e_old (v0_un→v1_un), side_v1 (v1_un→v1_tag),
            //             e_new (v1_tag→v0_tag — opposite of seam loop dir),
            //             side_v0 (v0_tag→v0_un).
            let (verts, edges): (Vec<VertexKey>, Vec<EdgeKey>) = match (side_v0, side_v1) {
                (Some(sv0), Some(sv1)) => (
                    vec![v0_un, v1_un, v1_tag, v0_tag],
                    vec![s.e_old, sv1, s.e_new, sv0],
                ),
                (Some(sv0), None) => {
                    // v1_tag == v1_un — triangle.
                    (vec![v0_un, v1_un, v0_tag], vec![s.e_old, s.e_new, sv0])
                }
                (None, Some(sv1)) => {
                    // v0_tag == v0_un — triangle.
                    (vec![v0_un, v1_un, v1_tag], vec![s.e_old, sv1, s.e_new])
                }
                (None, None) => unreachable!(),
            };

            let face = self.face_create(&verts, &edges, FaceCreateFlags::NONE);
            self.face_attribs
                .copy_into(face.as_usize(), s.face.as_usize());
            wall_faces.push(face);
        }

        // Phase 6: depth pass (INSET.md §8). Lift every tagged-region vertex
        // along its shell normal — the area-less normal averaged from the
        // seam loops on tagged faces touching the vertex.
        //
        // For flat regions the shell factor is 1.0, which is what this
        // implementation assumes. The general curved-surface form
        // (`shell_factor_at(v)`, INSET.md §8.2) is not yet pinned down in
        // the guide and not yet implemented here.
        if params.depth != 0.0 {
            // Collect the set of tagged-region vertex identities. After
            // Phase 5, the inset boundary lives on the tagged-side verts
            // recorded in `endpoint_tagged_v`. Interior tagged verts (with
            // no seam edge incident) currently fall through unchanged — the
            // BFS-relaxation fallback for their seam-length average
            // (INSET.md §8.1) is not yet implemented.
            let mut tagged_verts: std::collections::HashSet<VertexKey> =
                std::collections::HashSet::new();
            for &v in endpoint_tagged_v.values() {
                tagged_verts.insert(v);
            }

            // Shell normal per tagged vertex: sum the face normals of every
            // seam loop now incident to the vertex on the tagged side, then
            // normalize.
            let mut shell_normal: HashMap<VertexKey, Vec3> = HashMap::new();
            let mut seam_len_sum: HashMap<VertexKey, (f32, u32)> = HashMap::new();
            for (i, s) in split_edges.iter().enumerate() {
                // Tagged-side endpoint identities for this seam edge.
                let v0 = endpoint_tagged_v.get(&(i, 0)).copied().unwrap_or(s.v0);
                let v1 = endpoint_tagged_v.get(&(i, 1)).copied().unwrap_or(s.v1);
                let n = self.face_normal(s.face);
                for &v in &[v0, v1] {
                    *shell_normal.entry(v).or_insert(Vec3::ZERO) += n;
                    let entry = seam_len_sum.entry(v).or_insert((0.0, 0));
                    entry.0 += s.length;
                    entry.1 += 1;
                }
            }

            // Compute final positions into a scratch buffer to keep the
            // per-vertex calculation independent of write order (§8 step 6).
            let mut moves: Vec<(VertexKey, Vec3)> = Vec::with_capacity(tagged_verts.len());
            for v in tagged_verts {
                let n = shell_normal
                    .get(&v)
                    .copied()
                    .unwrap_or(Vec3::ZERO)
                    .normalize_or_zero();
                let avg_len = seam_len_sum
                    .get(&v)
                    .map(|(sum, count)| {
                        if *count > 0 {
                            *sum / *count as f32
                        } else {
                            1.0
                        }
                    })
                    .unwrap_or(1.0);
                let scale = if params.use_relative_offset {
                    avg_len
                } else {
                    1.0
                };
                // shell_factor = 1.0 (flat-region approximation).
                let offset = params.depth * scale;
                let pos = self.vertex_position(v);
                moves.push((v, pos + n * offset));
            }
            for (v, p) in moves {
                self.set_vertex_position(v, p);
            }
        }

        self.pop_face_op_flag_layer(face_tag);
        InsetRegionResult { wall_faces }
    }
}

#[cfg(test)]
mod tests {
    use glam::Vec3;

    use super::{InsetRegionParams, InsetRegionResult};
    use crate::loopmesh::test_support::create_face_from_verts;
    use crate::loopmesh::*;

    // Keep the unused-import lint happy when assertions don't touch the
    // result type's fields.
    #[allow(dead_code)]
    fn _result_type_used(r: InsetRegionResult) -> usize {
        r.wall_faces.len()
    }

    fn make_grid_3x3_faces(mesh: &mut LoopMesh) -> Vec<FaceKey> {
        // 4×4 vertex grid → 9 quads.
        let mut verts = Vec::with_capacity(16);
        for y in 0..4 {
            for x in 0..4 {
                verts.push(mesh.create_vertex(Vec3::new(x as f32, y as f32, 0.0)));
            }
        }
        let mut faces = Vec::with_capacity(9);
        for y in 0..3 {
            for x in 0..3 {
                let i0 = y * 4 + x;
                let i1 = y * 4 + (x + 1);
                let i2 = (y + 1) * 4 + (x + 1);
                let i3 = (y + 1) * 4 + x;
                let v = [verts[i0], verts[i1], verts[i2], verts[i3]];
                let e0 = mesh.create_edge([v[0], v[1]], EdgeCreateFlags::NO_DUPLICATE);
                let e1 = mesh.create_edge([v[1], v[2]], EdgeCreateFlags::NO_DUPLICATE);
                let e2 = mesh.create_edge([v[2], v[3]], EdgeCreateFlags::NO_DUPLICATE);
                let e3 = mesh.create_edge([v[3], v[0]], EdgeCreateFlags::NO_DUPLICATE);
                faces.push(mesh.face_create(&v, &[e0, e1, e2, e3], FaceCreateFlags::NONE));
            }
        }
        faces
    }

    #[test]
    fn inset_region_centre_face_of_3x3_grid() {
        let mut mesh = LoopMesh::new();
        let faces = make_grid_3x3_faces(&mut mesh);
        let centre = faces[4];

        let result = mesh.inset_region(
            &[centre],
            InsetRegionParams {
                thickness: 0.1,
                ..Default::default()
            },
        );

        // Four seam edges → four wall quads.
        assert_eq!(result.wall_faces.len(), 4);
        // 16 original verts + 4 inset verts.
        assert_eq!(mesh.totvert(), 20);
        // Original 24 edges + 4 seam-separated duplicates + 4 side edges.
        assert_eq!(mesh.totedge(), 32);
        // Original 9 faces + 4 walls.
        assert_eq!(mesh.totface(), 13);
        mesh.validate_topology();
    }

    #[test]
    fn inset_region_no_seams_is_no_op() {
        // Single isolated quad with `use_boundary = false` — no seam edges,
        // so the operator should leave the mesh untouched.
        let mut mesh = LoopMesh::new();
        let verts = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        ];
        let face = create_face_from_verts(&mut mesh, &verts);
        let v0 = mesh.totvert();
        let e0 = mesh.totedge();
        let f0 = mesh.totface();

        let result = mesh.inset_region(
            &[face],
            InsetRegionParams {
                thickness: 0.1,
                use_boundary: false,
                ..Default::default()
            },
        );

        assert!(result.wall_faces.is_empty());
        assert_eq!(mesh.totvert(), v0);
        assert_eq!(mesh.totedge(), e0);
        assert_eq!(mesh.totface(), f0);
    }

    #[test]
    fn inset_region_two_adjacent_grid_cells() {
        let mut mesh = LoopMesh::new();
        let faces = make_grid_3x3_faces(&mut mesh);

        // Two adjacent centre-row faces. Their shared edge is interior to
        // the selection (radial-2, both tagged) and must not be a seam.
        let result = mesh.inset_region(
            &[faces[3], faces[4]],
            InsetRegionParams {
                thickness: 0.1,
                ..Default::default()
            },
        );

        // Six seam edges form the outer rectangle of the two-cell region:
        // 2 top + 2 bottom + 1 left + 1 right.
        assert_eq!(result.wall_faces.len(), 6);
        mesh.validate_topology();
    }
}
