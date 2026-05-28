use std::collections::HashMap;

use glam::{Vec2, Vec3};

use super::mean_value::{mean_value_weights, PlaneProjection};
use crate::loopmesh::*;

/// One snapshotted face's customdata baseline (INSET.md §4.5). Captured before
/// the offset solver moves any corner so that phase four (§7) can resample the
/// *original* corner attributes at the *new* corner positions.
struct FaceSnapshot {
    /// The face's loops in cycle order. Loop keys are stable across phases
    /// 2–5, so each entry still names the live corner at rebake time. The
    /// corner vertices are re-read live through these loops at rebake time
    /// (their identities may change under vert-separate).
    loops: Vec<LoopKey>,
    /// Original corner positions projected into the face-local 2D plane.
    proj_corners: Vec<Vec2>,
    /// Detached copy of each corner's loop customdata block.
    loop_cd: SlotSnapshot,
    /// Detached copy of each corner-vertex's customdata block.
    vert_cd: SlotSnapshot,
    /// Axis-dominant projection shared by the snapshot corners and the query
    /// points (§10.1.1).
    projection: PlaneProjection,
}

/// One wall's inner-edge (tagged-corner) loop and the source loops behind it,
/// recorded for the §6.3 corner customdata merge. Two such records share a
/// corner vertex when two walls meet there.
struct CornerLoop {
    /// The wall loop sitting on the new (tagged) edge at the corner vertex.
    wall_loop: LoopKey,
    /// The tagged inset face's loop at the corner ("inner" loop, §6.3).
    tagged_src: LoopKey,
    /// The untagged original face's loop at the corner ("outer" loop), if any.
    outer_src: Option<LoopKey>,
}

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
    ///    When `use_edge_rail` is set and a two-seam corner sits on a common
    ///    non-seam edge shared by its two tagged faces (INSET.md §5.5), the
    ///    move direction is the slide along that existing edge — toward the
    ///    shared far vertex, using its pre-inset position — instead of the
    ///    tangent bisector. Corners without a shared edge fall back to the
    ///    bisector.
    /// 4. *(Not yet implemented: face re-interpolation when `use_interpolate`
    ///    is set.)*
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

        // (Optional) customdata snapshot — INSET.md §4.5. Capture the original
        // per-corner loop/vert customdata and projected corner positions for
        // every tagged face incident to a seam endpoint, before phase three
        // moves any corner. Phase four (§7) rebakes from this baseline.
        let face_snapshots: Vec<FaceSnapshot> = if params.use_interpolate
            && (!self.loop_attribs.is_empty() || !self.vertex_attribs.is_empty())
        {
            // Tagged faces incident to any seam endpoint. The walk goes
            // through each endpoint's live disk and keeps faces still tagged.
            let mut snap_faces: Vec<FaceKey> = Vec::new();
            let mut seen: std::collections::HashSet<FaceKey> = std::collections::HashSet::new();
            for &v in &endpoint_set {
                let disk: Vec<EdgeKey> = self.vertex_disk_iter(v).collect();
                for e in disk {
                    for l in self.edge_radial_iter(e).collect::<Vec<_>>() {
                        let f = self.loop_face(l);
                        if self.face_op_flag_test(face_tag, f, 0) && seen.insert(f) {
                            snap_faces.push(f);
                        }
                    }
                }
            }
            snap_faces
                .into_iter()
                .map(|f| self.capture_face_snapshot(f))
                .collect()
        } else {
            Vec::new()
        };

        // For `use_edge_rail` (INSET.md §5.5) the slide direction is measured
        // toward a far vertex along a shared non-seam edge, using *pre-inset*
        // coordinates — phase three moves boundary verts as it walks, so a
        // far vertex read live could feed back its own displacement. Snapshot
        // every vertex position that exists before any move.
        let original_coords: HashMap<VertexKey, Vec3> = if params.use_edge_rail {
            let live: Vec<VertexKey> = self.vertices.iter().collect();
            live.into_iter()
                .map(|v| (v, self.vertex_position(v)))
                .collect()
        } else {
            HashMap::new()
        };

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
                        let idx_a = seam_new[0];
                        let idx_b = seam_new[1];
                        let tangent_a = split_edges[idx_a].tangent;
                        let tangent_b = split_edges[idx_b].tangent;

                        // `use_edge_rail` (INSET.md §5.5): if the two seam edges
                        // share a common non-seam edge leading out of the corner
                        // — their tagged faces' "other" edge at this corner
                        // points at the same far vertex — slide along that
                        // existing edge instead of along the tangent bisector.
                        let rail_dir = if params.use_edge_rail {
                            let far_a = self.rail_far_vertex(&split_edges[idx_a], v_pre);
                            let far_b = self.rail_far_vertex(&split_edges[idx_b], v_pre);
                            match (far_a, far_b) {
                                (Some(fa), Some(fb)) if fa == fb => {
                                    let far_pos = original_coords
                                        .get(&fa)
                                        .copied()
                                        .unwrap_or_else(|| self.vertex_position(fa));
                                    let dir = (far_pos - pre_pos).normalize_or_zero();
                                    if dir == Vec3::ZERO {
                                        None
                                    } else {
                                        Some(dir)
                                    }
                                }
                                _ => None,
                            }
                        } else {
                            None
                        };

                        let dir = match rail_dir {
                            Some(slide) => {
                                if params.use_even_offset {
                                    // Scale the slide so its perpendicular
                                    // projection onto a seam tangent equals
                                    // `thickness` — use whichever tangent makes
                                    // the larger angle with the slide (the
                                    // smaller |dot|), matching §5.5.
                                    let dot_a = tangent_a.dot(slide).abs();
                                    let dot_b = tangent_b.dot(slide).abs();
                                    let cos = dot_a.min(dot_b);
                                    if cos > 1.0e-6 {
                                        slide / cos
                                    } else {
                                        slide
                                    }
                                } else {
                                    slide
                                }
                            }
                            None => {
                                let bisector = (tangent_a + tangent_b).normalize_or_zero();
                                if params.use_even_offset {
                                    let cos_half = tangent_a.dot(bisector).abs();
                                    if cos_half > 1.0e-6 {
                                        bisector / cos_half
                                    } else {
                                        bisector
                                    }
                                } else {
                                    bisector
                                }
                            }
                        };
                        let relative =
                            0.5 * (split_edges[idx_a].length + split_edges[idx_b].length);
                        (dir, relative)
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

        // Phase 4: re-interpolate face attributes (INSET.md §7). Runs after
        // the offset solver moved the corners but before phase five builds the
        // walls, so the wall's §6.2 loop-CD copy reads the already-rebaked
        // tagged-face corners and §6.3 midpoints those rebaked values.
        if params.use_interpolate {
            self.inset_reinterpolate_faces(&face_snapshots);
        }

        // Phase 5: build wall faces.
        let do_loop_interp = !self.loop_attribs.is_empty();
        let mut wall_faces: Vec<FaceKey> = Vec::with_capacity(split_edges.len());
        let mut side_edge_cache: HashMap<(VertexKey, VertexKey), EdgeKey> = HashMap::new();
        // §6.3: wall inner-edge loops grouped by their tagged corner vertex.
        let mut corner_records: HashMap<VertexKey, Vec<CornerLoop>> = HashMap::new();

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

            // §6.2 wall loop customdata copy (attribute-transparent wall) and
            // §6.3 per-corner merge bookkeeping — only when loop layers exist.
            if do_loop_interp {
                // Tagged-side source loops: the seam loop sits at v0_tag going
                // toward v1_tag, so it owns the v0_tag corner; its successor
                // owns the v1_tag corner (INSET.md §6.2).
                let src_v0_tag = s.seam_loop;
                let src_v1_tag = self.loop_next(s.seam_loop);
                // Untagged-side source loops: the loop on e_old whose face is
                // untagged. None for a fabricated mesh-boundary e_old.
                let (src_v0_un, src_v1_un) = self.untagged_seam_loops(s, face_tag, v0_un, v1_un);

                // Map each wall loop to its corner vertex and copy from the
                // matching source. Walls inherit the tagged corner's (now
                // §7-rebaked) values on the inner edge.
                let mut wl = self.faces.get(face).first_edge_loop;
                for _ in 0..verts.len() {
                    let wv = self.loop_vertex(wl);
                    let src = if wv == v0_tag {
                        Some(src_v0_tag)
                    } else if wv == v1_tag {
                        Some(src_v1_tag)
                    } else if wv == v0_un {
                        src_v0_un.or(Some(src_v0_tag))
                    } else if wv == v1_un {
                        src_v1_un.or(Some(src_v1_tag))
                    } else {
                        None
                    };
                    if let Some(src) = src {
                        self.loop_attribs.copy_into(wl.as_usize(), src.as_usize());
                    }
                    // Record the wall's inner-edge (tagged-corner) loops for
                    // the §6.3 corner merge.
                    if params.use_interpolate && (wv == v0_tag || wv == v1_tag) {
                        let outer = if wv == v0_tag { src_v0_un } else { src_v1_un };
                        corner_records.entry(wv).or_default().push(CornerLoop {
                            wall_loop: wl,
                            tagged_src: if wv == v0_tag { src_v0_tag } else { src_v1_tag },
                            outer_src: outer,
                        });
                    }
                    wl = self.loop_next(wl);
                }
            }

            wall_faces.push(face);
        }

        // §6.3 corner loop-CD merge. Run after every wall is built so each
        // tagged corner has the full set of incident wall inner-edge loops.
        if params.use_interpolate && do_loop_interp {
            self.inset_merge_corner_loops(&corner_records);
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

    /// For `use_edge_rail` (INSET.md §5.5): the far endpoint of the *other*
    /// edge of `split.face` incident to the corner `corner` — i.e. the
    /// non-seam edge of the tagged face leading out of the corner. Two seam
    /// edges that return the same far vertex share a common edge through it,
    /// which is the rail the inset corner slides along.
    ///
    /// `corner` is the pre-separation endpoint identity; the seam loop walks
    /// `split.v0 → split.v1`, so the corner is reached either at the seam
    /// loop itself (`corner == v0`) or at its successor (`corner == v1`).
    fn rail_far_vertex(&self, split: &SplitEdge, corner: VertexKey) -> Option<VertexKey> {
        // Two face loops touch the corner: the one whose `loop_vertex` is the
        // corner, and its predecessor. One sits on the seam edge; the other
        // sits on the non-seam edge that is the candidate rail.
        let corner_loop = if split.v0 == corner {
            split.seam_loop
        } else if split.v1 == corner {
            self.loop_next(split.seam_loop)
        } else {
            return None;
        };
        let prev_loop = self.loop_prev(corner_loop);
        let seam_edge = self.loop_edge(split.seam_loop);
        // The non-seam loop's edge leads to the far vertex along the rail; its
        // far endpoint is whichever of the loop's two corner-adjacent verts is
        // not the corner itself.
        let non_seam_loop = if self.loop_edge(corner_loop) == seam_edge {
            prev_loop
        } else {
            corner_loop
        };
        // The corner vertex's live identity (after this fan's vert-separate)
        // is `corner_loop`'s own vertex; the rail's far end is the non-seam
        // edge's other endpoint.
        let corner_live = self.loop_vertex(corner_loop);
        let a = self.loop_vertex(non_seam_loop);
        let b = self.loop_vertex(self.loop_next(non_seam_loop));
        Some(if a == corner_live { b } else { a })
    }

    /// INSET.md §4.5 — capture one face's loop/vert customdata baseline and
    /// its corners' axis-dominant 2D projection, for phase-four rebaking.
    fn capture_face_snapshot(&self, face: FaceKey) -> FaceSnapshot {
        let normal = self.face_normal(face);
        let projection = PlaneProjection::from_normal(normal);

        let mut loops: Vec<LoopKey> = Vec::new();
        let mut verts: Vec<VertexKey> = Vec::new();
        let mut proj_corners: Vec<Vec2> = Vec::new();
        let mut it = self.face_loops(face);
        while let Some(l) = it.next(self) {
            let v = self.loop_vertex(l);
            loops.push(l);
            verts.push(v);
            proj_corners.push(projection.project(self.vertex_position(v)));
        }

        let loop_slots: Vec<usize> = loops.iter().map(|l| l.as_usize()).collect();
        let vert_slots: Vec<usize> = verts.iter().map(|v| v.as_usize()).collect();
        let loop_cd = self.loop_attribs.snapshot_slots(&loop_slots);
        let vert_cd = self.vertex_attribs.snapshot_slots(&vert_slots);

        FaceSnapshot {
            loops,
            proj_corners,
            loop_cd,
            vert_cd,
            projection,
        }
    }

    /// INSET.md §7 — rebake each snapshotted face's per-corner customdata at
    /// the post-displacement corner positions using the §10.1 mean-value
    /// resample. Reads the §4.5 baseline (immune to the in-place write) and
    /// writes the blended values back to the live loops and corner verts.
    fn inset_reinterpolate_faces(&mut self, snapshots: &[FaceSnapshot]) {
        for snap in snapshots {
            let n = snap.loops.len();
            if n == 0 {
                continue;
            }
            for i in 0..n {
                // Live corner identity may have changed under vert-separate;
                // re-read the loop's current vertex and its moved position.
                let live_vert = self.loop_vertex(snap.loops[i]);
                let q = snap.projection.project(self.vertex_position(live_vert));
                let weights = mean_value_weights(&snap.proj_corners, q);
                self.loop_attribs.interpolate_from_snapshot(
                    snap.loops[i].as_usize(),
                    &snap.loop_cd,
                    &weights,
                );
                self.vertex_attribs.interpolate_from_snapshot(
                    live_vert.as_usize(),
                    &snap.vert_cd,
                    &weights,
                );
            }
        }
    }

    /// Locate the untagged-side loops on a seam edge's `e_old` at the two
    /// corner vertices, for the §6.2 wall loop-CD copy. Returns
    /// `(loop_at_v0_un, loop_at_v1_un)`; either is `None` when no untagged
    /// loop sits on `e_old` (mesh-boundary seam: `e_old` is a fabricated wire
    /// edge with an empty radial).
    fn untagged_seam_loops(
        &self,
        s: &SplitEdge,
        face_tag: crate::op_flags::FaceOpFlagLayer,
        v0_un: VertexKey,
        v1_un: VertexKey,
    ) -> (Option<LoopKey>, Option<LoopKey>) {
        let mut at_v0 = None;
        let mut at_v1 = None;
        for l in self.edge_radial_iter(s.e_old).collect::<Vec<_>>() {
            if self.face_op_flag_test(face_tag, self.loop_face(l), 0) {
                continue;
            }
            // The untagged face's two loops on this edge: `l` and its next.
            let l_next = self.loop_next(l);
            for cl in [l, l_next] {
                let v = self.loop_vertex(cl);
                if v == v0_un {
                    at_v0 = Some(cl);
                } else if v == v1_un {
                    at_v1 = Some(cl);
                }
            }
        }
        (at_v0, at_v1)
    }

    /// INSET.md §6.3 — merge per-corner loop customdata where two wall faces
    /// meet at a shared tagged corner. For each corner with exactly two walls:
    /// gate on the two outer (untagged) loops being equal (no pre-existing UV
    /// seam), then write the midpoint of the two inner (tagged inset) loops to
    /// both inner loops and to both walls' inner-edge loops.
    fn inset_merge_corner_loops(&mut self, corner_records: &HashMap<VertexKey, Vec<CornerLoop>>) {
        for records in corner_records.values() {
            if records.len() != 2 {
                // The four-loop find of §6.3 is defined for exactly the two
                // walls meeting at a corner. Lone walls (boundary) and
                // higher-valence pinches keep their §6.2-copied values.
                continue;
            }
            let a = &records[0];
            let b = &records[1];

            // Outer-pair-equality gate: skip when the original untagged face
            // already carried a seam (diverging outer loop values) here.
            match (a.outer_src, b.outer_src) {
                (Some(oa), Some(ob)) => {
                    if !self.loop_cd_equal(oa, ob) {
                        continue;
                    }
                }
                // Missing outer loop(s) → mesh-boundary corner, no merge.
                _ => continue,
            }

            // Midpoint of the two inner (tagged inset) loop values. Snapshot
            // the two source corners first so the in-place writes below don't
            // feed back through the still-being-read source slots.
            let snapshot = self
                .loop_attribs
                .snapshot_slots(&[a.tagged_src.as_usize(), b.tagged_src.as_usize()]);
            let weights = [0.5_f32, 0.5_f32];
            // Write into both inner loops and both walls' inner-edge loops.
            for dst in [a.tagged_src, b.tagged_src, a.wall_loop, b.wall_loop] {
                self.loop_attribs
                    .interpolate_from_snapshot(dst.as_usize(), &snapshot, &weights);
            }
        }
    }

    /// Loop customdata equality test (byte-fingerprint) for the §6.3 outer-pair
    /// gate.
    fn loop_cd_equal(&self, a: LoopKey, b: LoopKey) -> bool {
        let mut fa = Vec::new();
        let mut fb = Vec::new();
        self.loop_attribs.write_fingerprint(a.as_usize(), &mut fa);
        self.loop_attribs.write_fingerprint(b.as_usize(), &mut fb);
        fa == fb
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

    /// Build a two-triangle fan whose shared (interior) edge `c → far` is the
    /// rail. The two outer edges `c–A` and `c–B` are made mixed-radial by
    /// flanking unselected triangles, so they become the two seams meeting at
    /// the corner `c`. `far` is placed off the symmetry axis so the rail
    /// slide direction (c → far) differs from the tangent bisector.
    ///
    /// Returns `(selected_faces, c_key, far_position)`.
    fn make_rail_fan(mesh: &mut LoopMesh) -> (Vec<FaceKey>, VertexKey, Vec3) {
        let c = mesh.create_vertex(Vec3::new(0.0, 0.0, 0.0));
        let far = mesh.create_vertex(Vec3::new(1.0, 3.0, 0.0));
        let a = mesh.create_vertex(Vec3::new(2.0, 0.0, 0.0));
        let b = mesh.create_vertex(Vec3::new(-2.0, 0.0, 0.0));
        let ao = mesh.create_vertex(Vec3::new(1.0, -2.0, 0.0));
        let bo = mesh.create_vertex(Vec3::new(-1.0, -2.0, 0.0));

        let tri = |mesh: &mut LoopMesh, vs: [VertexKey; 3]| {
            let e0 = mesh.create_edge([vs[0], vs[1]], EdgeCreateFlags::NO_DUPLICATE);
            let e1 = mesh.create_edge([vs[1], vs[2]], EdgeCreateFlags::NO_DUPLICATE);
            let e2 = mesh.create_edge([vs[2], vs[0]], EdgeCreateFlags::NO_DUPLICATE);
            mesh.face_create(&vs, &[e0, e1, e2], FaceCreateFlags::NONE)
        };

        // Selected fan: both triangles share edge c–far.
        let f_a = tri(mesh, [c, a, far]);
        let f_b = tri(mesh, [c, far, b]);
        // Unselected flanks making c–A and c–B mixed-radial seams.
        let _u_a = tri(mesh, [a, c, ao]);
        let _u_b = tri(mesh, [c, b, bo]);

        (vec![f_a, f_b], c, Vec3::new(1.0, 3.0, 0.0))
    }

    /// The inset corner that came from `c`: the moved vertex nearest the
    /// origin whose position is not one of the original (unmoved) verts.
    fn moved_corner_near_origin(mesh: &LoopMesh) -> Vec3 {
        let originals = [
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::new(-2.0, 0.0, 0.0),
            Vec3::new(1.0, 3.0, 0.0),
            Vec3::new(1.0, -2.0, 0.0),
            Vec3::new(-1.0, -2.0, 0.0),
        ];
        let mut best: Option<Vec3> = None;
        for v in mesh.vertices.iter() {
            let p = mesh.vertex_position(v);
            if originals.iter().any(|o| (*o - p).length() < 1.0e-4) {
                continue;
            }
            // Skip the unmoved untagged copies still sitting at the origin.
            if p.length() < 1.0e-4 {
                continue;
            }
            best = match best {
                Some(cur) if cur.length() <= p.length() => Some(cur),
                _ => Some(p),
            };
        }
        best.expect("expected a displaced inset corner")
    }

    #[test]
    fn inset_region_edge_rail_slides_along_shared_edge() {
        let mut mesh = LoopMesh::new();
        let (faces, _c, far) = make_rail_fan(&mut mesh);

        let result = mesh.inset_region(
            &faces,
            InsetRegionParams {
                thickness: 0.3,
                use_boundary: false,
                use_even_offset: false,
                use_edge_rail: true,
                ..Default::default()
            },
        );
        assert_eq!(result.wall_faces.len(), 2);
        mesh.validate_topology();

        let moved = moved_corner_near_origin(&mesh);
        // The corner slid along the existing edge c → far.
        let rail_dir = far.normalize();
        let moved_dir = moved.normalize();
        assert!(
            (moved_dir - rail_dir).length() < 1.0e-4,
            "edge-rail corner should slide toward far vertex; moved={moved:?}"
        );
    }

    #[test]
    fn inset_region_edge_rail_differs_from_bisector() {
        // Same fixture, rail off: the corner follows the tangent bisector,
        // which for this asymmetric fan is *not* parallel to c → far.
        let mut mesh = LoopMesh::new();
        let (faces, _c, far) = make_rail_fan(&mut mesh);

        mesh.inset_region(
            &faces,
            InsetRegionParams {
                thickness: 0.3,
                use_boundary: false,
                use_even_offset: false,
                use_edge_rail: false,
                ..Default::default()
            },
        );

        let moved = moved_corner_near_origin(&mesh);
        let rail_dir = far.normalize();
        let moved_dir = moved.normalize();
        assert!(
            (moved_dir - rail_dir).length() > 1.0e-3,
            "bisector corner should not coincide with the rail; moved={moved:?}"
        );
    }

    #[test]
    fn inset_region_edge_rail_falls_back_to_bisector_without_shared_edge() {
        // Centre face of a 3×3 grid: the four seam corners are "free" — the
        // two seam edges at each corner belong to faces that do not share a
        // non-seam edge through the corner. `use_edge_rail` must therefore
        // produce the same result as the default bisector path.
        let mut rail = LoopMesh::new();
        let rail_faces = make_grid_3x3_faces(&mut rail);
        rail.inset_region(
            &[rail_faces[4]],
            InsetRegionParams {
                thickness: 0.2,
                use_edge_rail: true,
                ..Default::default()
            },
        );

        let mut plain = LoopMesh::new();
        let plain_faces = make_grid_3x3_faces(&mut plain);
        plain.inset_region(
            &[plain_faces[4]],
            InsetRegionParams {
                thickness: 0.2,
                use_edge_rail: false,
                ..Default::default()
            },
        );

        rail.validate_topology();
        assert_eq!(rail.totvert(), plain.totvert());
        assert_eq!(rail.totface(), plain.totface());
        // The four inset-boundary verts should land at identical positions.
        let mut rail_pos: Vec<[f32; 3]> = rail
            .vertices
            .iter()
            .map(|v| rail.vertex_position(v).to_array())
            .collect();
        let mut plain_pos: Vec<[f32; 3]> = plain
            .vertices
            .iter()
            .map(|v| plain.vertex_position(v).to_array())
            .collect();
        let key = |a: &[f32; 3]| {
            (
                (a[0] * 1.0e4) as i64,
                (a[1] * 1.0e4) as i64,
                (a[2] * 1.0e4) as i64,
            )
        };
        rail_pos.sort_by_key(|a| key(a));
        plain_pos.sort_by_key(|a| key(a));
        assert_eq!(rail_pos, plain_pos);
    }

    // ---- use_interpolate: §7 face re-interpolation + §6.3 corner merge ----

    use super::super::mean_value::{mean_value_weights, PlaneProjection};
    use glam::Vec2;

    /// Register a Vec2 loop "uv" attribute and a Vec3 vertex "color"
    /// attribute, returning their keys.
    fn register_uv_color(mesh: &mut LoopMesh) -> (ElementAttributeKey, ElementAttributeKey) {
        let uv = mesh
            .register_loop_attribute(ElementAttributeDesc::numeric(
                AttributeKey::custom("uv"),
                Format::new(ComponentType::F32, 2),
            ))
            .unwrap();
        let color = mesh
            .register_vertex_attribute(ElementAttributeDesc::numeric(
                AttributeKey::custom("color"),
                Format::new(ComponentType::F32, 3),
            ))
            .unwrap();
        (uv, color)
    }

    /// Expected mean-value blend of `corner_vals` at the post-displacement
    /// projected position of a corner, using the documented §10.1 routine.
    fn expected_blend2(
        proj_corners: &[Vec2],
        corner_vals: &[[f32; 2]],
        q: Vec2,
    ) -> [f32; 2] {
        let w = mean_value_weights(proj_corners, q);
        let mut out = [0.0_f32; 2];
        for i in 0..w.len() {
            out[0] += w[i] * corner_vals[i][0];
            out[1] += w[i] * corner_vals[i][1];
        }
        out
    }

    #[test]
    fn interpolate_rebakes_nonaffine_uv_via_mean_value() {
        // Centre face of a 3×3 grid carries a deliberately NON-affine UV
        // (u = x*y, v = x²) so that any affine / barycentric scheme would
        // disagree with mean value coordinates. After inset, each moved
        // corner's UV must equal the mean-value resample of the original
        // corner UVs at the corner's new projected position.
        let mut mesh = LoopMesh::new();
        let faces = make_grid_3x3_faces(&mut mesh);
        let centre = faces[4];
        let (uv, _color) = register_uv_color(&mut mesh);

        // Snapshot original corner positions + UVs in loop-cycle order.
        let mut orig_loops = Vec::new();
        let mut orig_pos = Vec::new();
        let mut it = mesh.face_loops(centre);
        while let Some(l) = it.next(&mesh) {
            let p = mesh.vertex_position(mesh.loop_vertex(l));
            let uv_val = [p.x * p.y, p.x * p.x];
            mesh.loop_attribs_mut().set_vec2(uv, l.as_usize(), uv_val);
            orig_loops.push(l);
            orig_pos.push(p);
        }
        let projection = PlaneProjection::from_normal(Vec3::Z);
        let proj_corners: Vec<Vec2> = orig_pos.iter().map(|&p| projection.project(p)).collect();
        let corner_uvs: Vec<[f32; 2]> = orig_pos
            .iter()
            .map(|&p| [p.x * p.y, p.x * p.x])
            .collect();

        mesh.inset_region(
            &[centre],
            InsetRegionParams {
                thickness: 0.2,
                use_interpolate: true,
                ..Default::default()
            },
        );

        // The four loops of the inset (shrunken) centre face are the same
        // loop keys (loops are stable); their verts moved inward. Verify the
        // rebaked UV equals the mean-value resample at the moved position.
        let mut checked = 0;
        for (i, &l) in orig_loops.iter().enumerate() {
            let new_p = mesh.vertex_position(mesh.loop_vertex(l));
            let q = projection.project(new_p);
            let want = expected_blend2(&proj_corners, &corner_uvs, q);
            let got = mesh.loop_attribs().get_vec2(uv, l.as_usize()).unwrap();
            assert!(
                (got[0] - want[0]).abs() < 1e-4 && (got[1] - want[1]).abs() < 1e-4,
                "corner {i}: got {got:?} want {want:?} (new_p={new_p:?})"
            );
            // The corner genuinely moved, so this is the smooth-interior path,
            // not the identity short-circuit.
            assert!((new_p - orig_pos[i]).length() > 1e-3, "corner {i} did not move");
            checked += 1;
        }
        assert_eq!(checked, 4);
    }

    #[test]
    fn interpolate_identity_rebake_for_unmoved_untagged_face() {
        // An untagged face adjacent to the seam is over-snapshotted (§7) but
        // none of its corners move, so it must rebake byte-exactly to its
        // original UVs (the on-corner identity short-circuit, §10.1.3).
        let mut mesh = LoopMesh::new();
        let faces = make_grid_3x3_faces(&mut mesh);
        let centre = faces[4];
        let neighbour = faces[1]; // shares an edge with the centre.
        let (uv, _color) = register_uv_color(&mut mesh);

        let mut neigh_loops = Vec::new();
        let mut neigh_orig = Vec::new();
        let mut it = mesh.face_loops(neighbour);
        while let Some(l) = it.next(&mesh) {
            let p = mesh.vertex_position(mesh.loop_vertex(l));
            // Non-affine, distinctive per-corner UVs.
            let uv_val = [p.x + 2.0 * p.y, (p.x - p.y) * (p.x - p.y)];
            mesh.loop_attribs_mut().set_vec2(uv, l.as_usize(), uv_val);
            neigh_loops.push(l);
            neigh_orig.push(uv_val);
        }

        mesh.inset_region(
            &[centre],
            InsetRegionParams {
                thickness: 0.15,
                use_interpolate: true,
                ..Default::default()
            },
        );

        for (l, want) in neigh_loops.iter().zip(neigh_orig.iter()) {
            let got = mesh.loop_attribs().get_vec2(uv, l.as_usize()).unwrap();
            assert_eq!(got, *want, "unmoved corner must rebake byte-exact");
        }
    }

    #[test]
    fn interpolate_rebakes_vertex_color_with_same_weights() {
        // The per-vertex domain rebakes by the same weight vector as the
        // per-loop domain (§10.1.4). Verify a vertex color channel matches the
        // mean-value blend at the moved corner.
        let mut mesh = LoopMesh::new();
        let faces = make_grid_3x3_faces(&mut mesh);
        let centre = faces[4];
        let (_uv, color) = register_uv_color(&mut mesh);

        let mut corner_loops = Vec::new();
        let mut orig_pos = Vec::new();
        let mut corner_cols = Vec::new();
        let mut it = mesh.face_loops(centre);
        while let Some(l) = it.next(&mesh) {
            let v = mesh.loop_vertex(l);
            let p = mesh.vertex_position(v);
            // Non-affine color.
            let c = [p.x * p.x, p.y * p.y, p.x * p.y];
            mesh.vertex_attribs_mut().set_vec3(color, v.as_usize(), c);
            corner_loops.push(l);
            orig_pos.push(p);
            corner_cols.push(c);
        }
        let projection = PlaneProjection::from_normal(Vec3::Z);
        let proj_corners: Vec<Vec2> = orig_pos.iter().map(|&p| projection.project(p)).collect();

        mesh.inset_region(
            &[centre],
            InsetRegionParams {
                thickness: 0.2,
                use_interpolate: true,
                ..Default::default()
            },
        );

        for (i, &l) in corner_loops.iter().enumerate() {
            let v = mesh.loop_vertex(l);
            let q = projection.project(mesh.vertex_position(v));
            let w = mean_value_weights(&proj_corners, q);
            let mut want = [0.0_f32; 3];
            for k in 0..4 {
                for c in 0..3 {
                    want[c] += w[k] * corner_cols[k][c];
                }
            }
            let got = mesh.vertex_attribs().get_vec3(color, v.as_usize()).unwrap();
            for c in 0..3 {
                assert!(
                    (got[c] - want[c]).abs() < 1e-4,
                    "corner {i} chan {c}: got {} want {}",
                    got[c],
                    want[c]
                );
            }
        }
    }

    #[test]
    fn corner_merge_equalises_inner_loops_when_outer_equal() {
        // §6.3: two walls meet at each inset corner. With UV ≡ a continuous
        // function of position (so the untagged "outer" loops at the shared
        // original vertex are equal), the two walls' inner-edge loops at the
        // corner must end up holding the same midpointed UV.
        let mut mesh = LoopMesh::new();
        let faces = make_grid_3x3_faces(&mut mesh);
        let centre = faces[4];
        let (uv, _color) = register_uv_color(&mut mesh);

        // Continuous per-vertex UV shared by every loop at that vertex, so the
        // untagged outer loops on either side of each corner agree (no seam).
        let set_uv_from_pos = |mesh: &mut LoopMesh, uv: ElementAttributeKey| {
            let loops: Vec<LoopKey> = mesh.loops.iter().collect();
            for l in loops {
                let p = mesh.vertex_position(mesh.loop_vertex(l));
                mesh.loop_attribs_mut()
                    .set_vec2(uv, l.as_usize(), [p.x, p.y]);
            }
        };
        set_uv_from_pos(&mut mesh, uv);

        let result = mesh.inset_region(
            &[centre],
            InsetRegionParams {
                thickness: 0.2,
                use_interpolate: true,
                ..Default::default()
            },
        );
        assert_eq!(result.wall_faces.len(), 4);
        mesh.validate_topology();

        // For each inset-boundary (tagged) vertex, every wall loop sitting on
        // that vertex's inner edge must carry the merged (equal) UV.
        // Collect, per tagged vertex, the wall loops on the inner edge.
        let mut by_vert: std::collections::HashMap<VertexKey, Vec<[f32; 2]>> =
            std::collections::HashMap::new();
        for &wf in &result.wall_faces {
            let mut it = mesh.face_loops(wf);
            while let Some(l) = it.next(&mesh) {
                let v = mesh.loop_vertex(l);
                let got = mesh.loop_attribs().get_vec2(uv, l.as_usize()).unwrap();
                by_vert.entry(v).or_default().push(got);
            }
        }
        // A corner where two walls meet contributes two inner-edge loops at the
        // tagged vertex; after the merge their UVs are equal.
        let mut merged_corners = 0;
        for vals in by_vert.values() {
            if vals.len() >= 2 {
                let first = vals[0];
                if vals
                    .iter()
                    .all(|v| (v[0] - first[0]).abs() < 1e-5 && (v[1] - first[1]).abs() < 1e-5)
                {
                    merged_corners += 1;
                }
            }
        }
        assert!(merged_corners >= 4, "expected merged corners, got {merged_corners}");
    }

    #[test]
    fn corner_merge_skipped_when_outer_uv_seam_present() {
        // §6.3 gate: if the original untagged faces already disagree on UV at
        // the corner (a user UV seam), no merge happens. We give each untagged
        // face its own distinct per-corner UVs, so the outer-pair-equality
        // gate fails and the inner loops keep their §7-rebaked (per-face,
        // diverged) values. The operator must still complete cleanly.
        let mut mesh = LoopMesh::new();
        let faces = make_grid_3x3_faces(&mut mesh);
        let centre = faces[4];
        let (uv, _color) = register_uv_color(&mut mesh);

        // Per-FACE UV islands: each loop's UV depends on its face index, so no
        // two faces agree at a shared corner.
        let face_keys: Vec<FaceKey> = mesh.face_keys();
        for (fi, &f) in face_keys.iter().enumerate() {
            let loops: Vec<LoopKey> = {
                let mut acc = Vec::new();
                let mut it = mesh.face_loops(f);
                while let Some(l) = it.next(&mesh) {
                    acc.push(l);
                }
                acc
            };
            for l in loops {
                mesh.loop_attribs_mut()
                    .set_vec2(uv, l.as_usize(), [fi as f32, fi as f32 * 2.0]);
            }
        }

        let result = mesh.inset_region(
            &[centre],
            InsetRegionParams {
                thickness: 0.2,
                use_interpolate: true,
                ..Default::default()
            },
        );
        assert_eq!(result.wall_faces.len(), 4);
        mesh.validate_topology();
        // No panic / clean topology is the assertion: the gate declined the
        // merge and the rebaked per-face values were left in place.
    }
}
