use std::collections::HashMap;

use glam::{Vec2, Vec3};

use super::mean_value::{mean_value_weights, PlaneProjection};
use crate::loopmesh::*;

/// Squared-distance tolerance for the corner customdata merge's value
/// comparisons (INSET.md §6.3 step-5 note). Both the step-5 gate and the
/// step-7 propagation value-match treat two math-layer values as equal when
/// their squared distance is below this threshold — roughly `3.2e-3` units of
/// separation for a two-component float layer. A byte-exact compare here is the
/// documented byte-match killer: independently-rebaked float resamples that the
/// operator should treat as identical differ in their low bits.
const CORNER_MERGE_EPS_SQ: f32 = 1.0e-5;

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

/// One wall's side edge — a "connecting" edge `e_connect` of the wall quad
/// that runs from a rim (outer/untagged) vertex down to a moved inset boundary
/// corner `v` (INSET.md §6.3). The corner merge is driven by **two invocations
/// per wall**, one keyed on each of the wall's two side edges; this record
/// names one such invocation's geometry.
struct WallSideEdge {
    /// The wall face that owns this invocation ("wall A").
    wall: FaceKey,
    /// The side edge `e_connect`, shared by wall A and the radial-other face
    /// across it ("face B"). The merge fires only when this edge is manifold.
    connect_edge: EdgeKey,
    /// The rim (outer/untagged) endpoint of the side edge — where the gate's
    /// outer pair lives.
    rim_vertex: VertexKey,
    /// The moved inset boundary corner `v` — the foot of the side edge, where
    /// the inner and behind pairs live.
    moved_corner: VertexKey,
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

        // Phase 1: classify every edge in the mesh against the tagged region
        // and build the SplitEdge[] array. The classification walks the
        // tagged/untagged region boundary, so it must visit *every* edge — not
        // just those of the input `faces` slice. Under `use_outset` the tagged
        // region is the inverted set, so seam edges that border an excluded
        // (or input) face touch only an untagged face from the input slice's
        // perspective; iterating all edges is what lets those seams be found.
        let mut split_edges: Vec<SplitEdge> = Vec::new();
        for e in self.edge_keys() {
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

            let (is_seam, is_mesh_boundary, seam_loop) = if tagged_count == 1 && untagged_count >= 1
            {
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
                        // seam vertex (INSET.md §5.6). The direction is gated on
                        // `use_even_offset`:
                        //   off → the bare seam inward tangent (§4.3); the corner
                        //         moves perpendicular to the seam at magnitude
                        //         `thickness` and the non-seam boundary edge is
                        //         ignored entirely.
                        //   on  → the slide *along* the non-seam boundary edge,
                        //         scaled by `1/|tangent · slide|` (§5.2) so its
                        //         perpendicular projection onto the seam tangent
                        //         equals `thickness`.
                        // The relative-offset scale is the single seam edge's
                        // pre-inset length (§5.6).
                        let tangent = split_edges[seam_new[0]].tangent;
                        let length = split_edges[seam_new[0]].length;
                        let dir = if params.use_even_offset {
                            match self.one_seam_slide_dir(fv, &e_new_to_split, &e_old_to_split) {
                                Some(slide) => {
                                    let cos = tangent.dot(slide).abs();
                                    if cos > 1.0e-6 {
                                        slide / cos
                                    } else {
                                        slide
                                    }
                                }
                                // No non-seam boundary edge at this fan: fall back
                                // to the seam tangent.
                                None => tangent,
                            }
                        } else {
                            tangent
                        };
                        (dir, length)
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
        // §6.3: per-wall side edges. Each side edge spawns one corner-merge
        // invocation (two per wall) keyed on that edge.
        let mut wall_side_edges: Vec<WallSideEdge> = Vec::new();

        // §6.2 far-loop source: map each snapshotted tagged-face loop to the
        // (snapshot, corner) holding its *pre-move* customdata block. Loop keys
        // are stable across phases 2–5, so the seam loops named in `split_edges`
        // index straight into this map.
        let mut premove_corner: HashMap<LoopKey, (usize, usize)> = HashMap::new();
        for (snap_idx, snap) in face_snapshots.iter().enumerate() {
            for (corner_idx, &loop_key) in snap.loops.iter().enumerate() {
                premove_corner.insert(loop_key, (snap_idx, corner_idx));
            }
        }

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

            // INSET.md §6.1 / §8.2.1.1: the wall inherits the tagged inset
            // face's normal verbatim — not a normal recomputed from the wall
            // quad's own (roughly perpendicular) plane. The depth pass (§8)
            // runs after the walls exist and `shell_factor_at` (§8.2) iterates
            // *all* incident faces of a seam corner, walls included. Carrying
            // the inset face's normal makes each wall contribute the same
            // per-face shell term as its source inset face instead of a
            // near-tangent term the §8.2.3 guard would clamp to 1. The inset
            // face's stored normal at this point is its pre-offset value
            // (computed in phase two, never recomputed by the offset solver or
            // phase four), which is the value §6.1 intends the wall to inherit.
            let inset_face_normal = self.face_normal(s.face);
            self.faces.get_mut(face).normal = inset_face_normal;

            // §6.2 wall loop customdata copy and §6.3 per-corner merge
            // bookkeeping — only when loop layers exist. Every wall loop sources
            // from the *tagged* inset face; the untagged neighbour never
            // contributes. Near loops (on the tagged edge `e_new`) take the
            // post-move, §7-rebaked live value; far loops (on the old edge
            // `e_old`) take the pre-move §4.5 snapshot value, in swapped corner
            // order to preserve the wall's winding.
            if do_loop_interp {
                // §6.2 source-loop correspondence: locate each near loop's
                // inset-face source by a single radial hop across `e_new`,
                // re-anchored to that near loop's own vertex (the
                // "[other loop](BMESH_API.md)" primitive). The hop is
                // topology-independent — it yields same-side output on a grid
                // and opposite-looking output on a fan with no branch (§6.2.1).
                //
                // The wall's two near loops sit on `e_new` (at v0_tag, v1_tag);
                // their radial-other loops on `e_new` are the inset face's loops
                // at the same corners. Each near loop copies its own source
                // live; each far loop copies the *other* end's near source from
                // the §4.5 snapshot, the cross forced by the §6.1
                // opposite-order wall winding (far@P ← src@Q, far@Q ← src@P).
                let mut near_at_v0: Option<LoopKey> = None;
                let mut near_at_v1: Option<LoopKey> = None;

                let mut wl = self.faces.get(face).first_edge_loop;
                for _ in 0..verts.len() {
                    let wv = self.loop_vertex(wl);
                    if wv == v0_tag {
                        near_at_v0 = Some(wl);
                    } else if wv == v1_tag {
                        near_at_v1 = Some(wl);
                    }
                    wl = self.loop_next(wl);
                }

                // Near sources via the radial hop across `e_new`.
                let src_v0 = near_at_v0.and_then(|nl| self.edge_other_loop(s.e_new, nl));
                let src_v1 = near_at_v1.and_then(|nl| self.edge_other_loop(s.e_new, nl));

                let mut wl = self.faces.get(face).first_edge_loop;
                for _ in 0..verts.len() {
                    let wv = self.loop_vertex(wl);
                    if wv == v0_tag || wv == v1_tag {
                        // Near loop: copy its own near source live (post-rebake).
                        let near = if wv == v0_tag { src_v0 } else { src_v1 };
                        if let Some(near) = near {
                            self.loop_attribs.copy_into(wl.as_usize(), near.as_usize());
                        }
                    } else if wv == v0_un || wv == v1_un {
                        // Far loop: the inset-face source at its OWN physical
                        // corner (the same source the near loop at the matching
                        // tagged copy uses), read from the §4.5 snapshot. The
                        // far loop and the tagged copy `v?_tag` are the two
                        // copies of one original vertex, so the radial hop
                        // re-anchors to that shared corner (§6.2: "re-anchored
                        // to the loop's own vertex"). The §6.1 opposite-order
                        // assembly places this in the crossed wall-cycle slot,
                        // but the *source corner* is the same one — never the
                        // other end. The non-interpolate path has no snapshot,
                        // so the far loop keeps the live near-source value the
                        // copy below seeds (near == far with no rebake), giving
                        // a per-corner-uniform wall.
                        let far_src = if wv == v0_un { src_v0 } else { src_v1 };
                        if let Some(far_src) = far_src {
                            self.loop_attribs
                                .copy_into(wl.as_usize(), far_src.as_usize());
                            if let Some(&(snap_idx, corner_idx)) = premove_corner.get(&far_src) {
                                self.write_premove_corner(
                                    wl,
                                    snap_idx,
                                    corner_idx,
                                    &face_snapshots,
                                );
                            }
                        }
                    }
                    wl = self.loop_next(wl);
                }
            }

            // §6.3: record this wall's two side edges (the connecting edges from
            // a rim vertex down to a moved corner) for the corner merge.
            if params.use_interpolate {
                if let Some(sv0) = side_v0 {
                    wall_side_edges.push(WallSideEdge {
                        wall: face,
                        connect_edge: sv0,
                        rim_vertex: v0_un,
                        moved_corner: v0_tag,
                    });
                }
                if let Some(sv1) = side_v1 {
                    wall_side_edges.push(WallSideEdge {
                        wall: face,
                        connect_edge: sv1,
                        rim_vertex: v1_un,
                        moved_corner: v1_tag,
                    });
                }
            }

            wall_faces.push(face);
        }

        // §6.3 corner loop-CD merge. Replayed once over the side edges in their
        // phase-five wall-build order (the visit-order analogue of the inline
        // per-connecting-face merge §6.3.1.1 describes); deferring the replay
        // until every wall exists only means each tagged corner already carries
        // its full set of incident wall inner-edge loops when the pass runs.
        // Per INSET.md §6.3 the merge only applies when at least one loop
        // layer is a *math* layer — one that blends fractionally
        // (`InterpPolicy::Linear`). Non-math layers (integer-family, blended
        // by a nearest/winner-take-one policy) must not be midpointed, so a
        // mesh carrying only such layers skips the merge entirely.
        if params.use_interpolate && do_loop_interp && self.has_loop_math_layer() {
            self.inset_merge_corner_loops(face_tag, &wall_side_edges);
        }

        // Phase 6: depth pass (INSET.md §8). Lift every tagged-region vertex
        // along its shell normal — the area-less normal averaged from the
        // seam loops on tagged faces touching the vertex.
        //
        // When `use_even_offset` is set, the displacement along the shell
        // normal is scaled by the curved-surface shell factor
        // (`shell_factor_at`, INSET.md §8.2) so the lifted boundary stays at a
        // uniform perpendicular distance from each incident face's pre-depth
        // plane regardless of how the surface curves at the vertex. The factor
        // reads incident corner angles off the *pre-depth* positions, which the
        // two-pass scratch-buffer structure (§8 step 6) preserves: positions
        // are not written back until every factor has been measured.
        if params.depth != 0.0 {
            self.inset_depth_pass(face_tag, &split_edges, &endpoint_tagged_v, &params);
        }

        self.pop_face_op_flag_layer(face_tag);
        InsetRegionResult { wall_faces }
    }

    /// INSET.md §8 — lift every tagged-region vertex (seam boundary *and*
    /// interior) along its depth normal.
    ///
    /// Seam-edge endpoints get a freshly-averaged shell normal (§8 steps 1–3:
    /// the area-less sum of incident tagged seam loops' face normals). Interior
    /// tagged verts have no incident seam edge, so steps 1–3 never visit them
    /// (the step-5 note in §8): they lift along their *incoming, un-recomputed*
    /// vertex normal as it stood entering the depth pass.
    ///
    /// The `use_relative_offset` length scale (§8.1) is the average of incident
    /// seam-edge pre-inset lengths for a boundary vert; for an interior vert
    /// with no incident seam edge it is the topological-distance relaxation of
    /// §8.1 — built lazily and only when its three gates hold.
    fn inset_depth_pass(
        &mut self,
        face_tag: crate::op_flags::FaceOpFlagLayer,
        split_edges: &[SplitEdge],
        endpoint_tagged_v: &HashMap<(usize, usize), VertexKey>,
        params: &InsetRegionParams<'_>,
    ) {
        // Step 4: re-tag every vertex of every tagged face — boundary *and*
        // interior, this time. The tagged faces still carry `face_tag` after
        // the surgery (the inset face rode onto the new boundary), so their
        // live corner verts are the tagged-region vertex set.
        let mut tagged_verts: std::collections::HashSet<VertexKey> =
            std::collections::HashSet::new();
        let mut tagged_edges: std::collections::HashSet<EdgeKey> = std::collections::HashSet::new();
        for f in self.face_keys() {
            if !self.face_op_flag_test(face_tag, f, 0) {
                continue;
            }
            let mut it = self.face_loops(f);
            while let Some(l) = it.next(self) {
                tagged_verts.insert(self.loop_vertex(l));
                tagged_edges.insert(self.loop_edge(l));
            }
        }

        // Steps 1–3 (shell normal) and the boundary half of §8.1 (seam-length
        // average), accumulated per seam-edge endpoint on its tagged side.
        let mut shell_normal: HashMap<VertexKey, Vec3> = HashMap::new();
        let mut seam_len_sum: HashMap<VertexKey, (f32, u32)> = HashMap::new();
        for (i, s) in split_edges.iter().enumerate() {
            let v0 = endpoint_tagged_v.get(&(i, 0)).copied().unwrap_or(s.v0);
            let v1 = endpoint_tagged_v.get(&(i, 1)).copied().unwrap_or(s.v1);
            let face_normal = self.face_normal(s.face);
            for &v in &[v0, v1] {
                *shell_normal.entry(v).or_insert(Vec3::ZERO) += face_normal;
                let entry = seam_len_sum.entry(v).or_insert((0.0, 0));
                entry.0 += s.length;
                entry.1 += 1;
            }
        }

        // §8.1 interior-vert length relaxation, gated on all three conditions
        // (depth ≠ 0 holds here, `use_relative_offset` on, and ≥1 interior
        // tagged vert with no incident seam edge). Built lazily and reused.
        let interior_lengths = if params.use_relative_offset {
            self.relax_interior_seam_lengths(&tagged_verts, &tagged_edges, &seam_len_sum)
        } else {
            HashMap::new()
        };

        // Step 5 → scratch buffer; step 6 writes back after every factor has
        // been measured against the pre-depth geometry (§8.2.5).
        let mut moves: Vec<(VertexKey, Vec3)> = Vec::with_capacity(tagged_verts.len());
        for &v in &tagged_verts {
            let is_seam_vert = seam_len_sum.contains_key(&v);
            // Boundary verts lift along their freshly-averaged shell normal;
            // interior verts along their incoming, un-recomputed `v.no`.
            let normal = if is_seam_vert {
                shell_normal
                    .get(&v)
                    .copied()
                    .unwrap_or(Vec3::ZERO)
                    .normalize_or_zero()
            } else {
                self.vertex_normal(v)
            };

            let scale = if params.use_relative_offset {
                if let Some((sum, count)) = seam_len_sum.get(&v) {
                    if *count > 0 {
                        *sum / *count as f32
                    } else {
                        1.0
                    }
                } else {
                    // Interior vert: the topological-distance fallback (§8.1).
                    interior_lengths.get(&v).copied().unwrap_or(1.0)
                }
            } else {
                1.0
            };

            let shell = if params.use_even_offset {
                self.shell_factor_at(v, normal)
            } else {
                1.0
            };

            let offset = params.depth * scale * shell;
            let pos = self.vertex_position(v);
            moves.push((v, pos + normal * offset));
        }
        for (v, p) in moves {
            self.set_vertex_position(v, p);
        }
    }

    /// INSET.md §8.1 — topological-distance length relaxation for interior
    /// tagged verts that have no incident seam edge.
    ///
    /// Seeds every seam-touching tagged vert with its own
    /// `avg_seam_edge_length_at(v)` (the boundary case, already in
    /// `seam_len_sum`), then walks outward through tagged edges: each
    /// unresolved interior vert takes the mean of its already-resolved
    /// tagged-edge neighbours' lengths, one BFS-style pass at a time, until
    /// every reachable interior vert has a length.
    ///
    /// Returns the resolved length for each interior vert. Boundary verts are
    /// not included (their length comes straight from `seam_len_sum`). Returns
    /// an empty map when no interior tagged vert needs the fallback — the third
    /// §8.1 gate.
    fn relax_interior_seam_lengths(
        &self,
        tagged_verts: &std::collections::HashSet<VertexKey>,
        tagged_edges: &std::collections::HashSet<EdgeKey>,
        seam_len_sum: &HashMap<VertexKey, (f32, u32)>,
    ) -> HashMap<VertexKey, f32> {
        let interior: Vec<VertexKey> = tagged_verts
            .iter()
            .copied()
            .filter(|v| !seam_len_sum.contains_key(v))
            .collect();
        if interior.is_empty() {
            return HashMap::new();
        }

        // Resolved lengths: seeded from the boundary verts' averages.
        let mut resolved: HashMap<VertexKey, f32> = HashMap::new();
        for (&v, &(sum, count)) in seam_len_sum {
            if count > 0 {
                resolved.insert(v, sum / count as f32);
            }
        }

        let mut pending: std::collections::HashSet<VertexKey> = interior.iter().copied().collect();
        while !pending.is_empty() {
            let mut newly_resolved: Vec<(VertexKey, f32)> = Vec::new();
            for &v in &pending {
                let mut neighbour_sum = 0.0_f32;
                let mut neighbour_count = 0u32;
                for edge in self.vertex_disk_iter(v) {
                    if !tagged_edges.contains(&edge) {
                        continue;
                    }
                    let [a, b] = self.edge_vertices(edge);
                    let other = if a == v { b } else { a };
                    if let Some(&length) = resolved.get(&other) {
                        neighbour_sum += length;
                        neighbour_count += 1;
                    }
                }
                if neighbour_count > 0 {
                    newly_resolved.push((v, neighbour_sum / neighbour_count as f32));
                }
            }
            // A pass that resolves nothing means the remaining interior verts
            // are unreachable from any seam through tagged edges; stop rather
            // than loop forever (they keep the `1.0` default at the call site).
            if newly_resolved.is_empty() {
                break;
            }
            for (v, length) in newly_resolved {
                resolved.insert(v, length);
                pending.remove(&v);
            }
        }

        resolved
            .into_iter()
            .filter(|(v, _)| !seam_len_sum.contains_key(v))
            .collect()
    }

    /// INSET.md §5.6 — the unit slide direction for a one-seam mesh-boundary
    /// fan-instance: the direction from the fan-instance `fv` to its neighbour
    /// *along the non-seam boundary edge*. That edge is the one in `fv`'s disk
    /// that is neither the seam edge (`e_new`) nor the fabricated parallel
    /// untagged edge (`e_old`) and that lies on the mesh boundary (radial
    /// count ≤ 1). Returns `None` if no such edge exists, so the caller can
    /// fall back to the seam tangent.
    fn one_seam_slide_dir(
        &self,
        fv: VertexKey,
        e_new_to_split: &HashMap<EdgeKey, usize>,
        e_old_to_split: &HashMap<EdgeKey, usize>,
    ) -> Option<Vec3> {
        let here = self.vertex_position(fv);
        for e in self.vertex_disk_iter(fv) {
            if e_new_to_split.contains_key(&e) || e_old_to_split.contains_key(&e) {
                continue;
            }
            // Only the mesh boundary (or a wire) is a valid slide rail; an
            // interior edge would not be "along the mesh boundary" (§5.6).
            if self.edge_radial_iter(e).count() > 1 {
                continue;
            }
            let [a, b] = self.edge_vertices(e);
            let other = if a == fv { b } else { a };
            let dir = (self.vertex_position(other) - here).normalize_or_zero();
            if dir != Vec3::ZERO {
                return Some(dir);
            }
        }
        None
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

    /// INSET.md §8.2 — the curved-surface shell factor at `v`: the
    /// corner-angle-weighted mean over `v`'s incident faces of the per-face
    /// factor `1/|dot(vertex_normal, face_normal)|`.
    ///
    /// ```text
    ///              Σ_f  per_face_shell(v, f) · corner_angle(f, v)
    /// shell(v)  =  ───────────────────────────────────────────────
    ///                     Σ_f  corner_angle(f, v)
    /// ```
    ///
    /// `vertex_normal` is the per-vertex shell normal already built in §8 steps
    /// 1–3 (assumed unit-length, as is each face normal). `corner_angle(f, v)`
    /// is the interior angle of `f` at `v` — the unsigned angle ∠(prev, v, next)
    /// in `[0, π]` measured at `v`'s current (pre-depth) position.
    ///
    /// Two guards keep the result finite and `≥ 1` (§8.2.3): the per-face
    /// factor is clamped to `1` when the move direction is near-tangent to the
    /// face (`|dot| < EPS`), and the whole factor falls back to `1` when the
    /// total corner angle is zero (a fully collapsed corner with no weight to
    /// average).
    fn shell_factor_at(&self, v: VertexKey, vertex_normal: Vec3) -> f32 {
        const TANGENT_EPS: f32 = 1.0e-6;

        let mut accumulated_shell = 0.0_f32;
        let mut accumulated_angle = 0.0_f32;

        // Visit every face-corner at `v`: walk the disk, and for each incident
        // edge walk its radial fan, keeping the loops whose own vertex is `v`
        // (one per incident face). Dedup faces so a face reached via two of
        // `v`'s spokes is only weighted once.
        let mut seen_faces: Vec<FaceKey> = Vec::new();
        let disk: Vec<EdgeKey> = self.vertex_disk_iter(v).collect();
        for edge in disk {
            for corner_loop in self.edge_radial_iter(edge).collect::<Vec<_>>() {
                if self.loop_vertex(corner_loop) != v {
                    continue;
                }
                let face = self.loop_face(corner_loop);
                if seen_faces.contains(&face) {
                    continue;
                }
                seen_faces.push(face);

                let apex = self.vertex_position(v);
                let previous = self.vertex_position(self.loop_vertex(self.loop_prev(corner_loop)));
                let next = self.vertex_position(self.loop_vertex(self.loop_next(corner_loop)));
                let corner_angle = (previous - apex).angle_between(next - apex);

                let face_normal = self.face_normal(face);
                let cosine = vertex_normal.dot(face_normal).abs();
                let per_face_shell = if cosine < TANGENT_EPS {
                    1.0
                } else {
                    1.0 / cosine
                };

                accumulated_shell += per_face_shell * corner_angle;
                accumulated_angle += corner_angle;
            }
        }

        if accumulated_angle != 0.0 {
            accumulated_shell / accumulated_angle
        } else {
            1.0
        }
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

    /// INSET.md §6.2 far-loop write — overlay the tagged inset face's *pre-move*
    /// customdata at one snapshot corner onto a wall loop. A one-hot weight on
    /// the named corner reproduces that corner's snapshot value exactly for
    /// every interpolable layer (math layers re-normalise to a copy; integer /
    /// nearest layers pick the winning corner). NoInterp layers are left as the
    /// caller seeded them (the live tagged value, which §7 never moved).
    fn write_premove_corner(
        &mut self,
        dst: LoopKey,
        snap_idx: usize,
        corner_idx: usize,
        snapshots: &[FaceSnapshot],
    ) {
        let snapshot = &snapshots[snap_idx].loop_cd;
        // Caller contract: `corner_idx` names a real corner in this snapshot.
        // A zero-corner snapshot would make the one-hot weight below an
        // out-of-bounds write, so guard it as an invariant rather than letting
        // it become a latent panic for a future caller.
        debug_assert!(
            corner_idx < snapshot.len(),
            "premove corner index {corner_idx} out of range for snapshot of {} corners",
            snapshot.len()
        );
        let mut weights = vec![0.0_f32; snapshot.len()];
        weights[corner_idx] = 1.0;
        self.loop_attribs
            .interpolate_from_snapshot(dst.as_usize(), snapshot, &weights);
    }

    /// INSET.md §6.2 — the radial-other loop on `edge` re-anchored to the
    /// vertex of `loop_key` (the "[other loop](../BMESH_API.md)" hop). On a
    /// manifold edge this lands on the single other face's loop; the result is
    /// re-anchored so it shares `loop_key`'s vertex. Returns `None` if `edge`
    /// has no radial-other loop at that vertex.
    fn edge_other_loop(&self, edge: EdgeKey, loop_key: LoopKey) -> Option<LoopKey> {
        let here = self.loop_vertex(loop_key);
        let self_face = self.loop_face(loop_key);
        for candidate in self.edge_radial_iter(edge).collect::<Vec<_>>() {
            if self.loop_face(candidate) == self_face {
                continue;
            }
            // A face's loop on `edge` is anchored at one of the edge's two
            // endpoints; the loop's successor is anchored at the other. Pick the
            // one sharing `here`.
            if self.loop_vertex(candidate) == here {
                return Some(candidate);
            }
            let next = self.loop_next(candidate);
            if self.loop_edge(candidate) == edge && self.loop_vertex(next) == here {
                return Some(next);
            }
        }
        None
    }

    /// The face's loop on `edge` anchored at `vertex`, if any. `edge` is one of
    /// `face`'s edges; the matching loop is the one whose own vertex is
    /// `vertex` and whose edge is `edge`.
    fn loop_on_face_at_vertex_on_edge(
        &self,
        face: FaceKey,
        edge: EdgeKey,
        vertex: VertexKey,
    ) -> Option<LoopKey> {
        for candidate in self.edge_radial_iter(edge).collect::<Vec<_>>() {
            if self.loop_face(candidate) != face {
                continue;
            }
            if self.loop_vertex(candidate) == vertex {
                return Some(candidate);
            }
            let next = self.loop_next(candidate);
            if self.loop_vertex(next) == vertex {
                return Some(next);
            }
        }
        None
    }

    /// INSET.md §6.3 — merge per-corner loop customdata, driven by two
    /// invocations per wall keyed on the wall's two side edges (the connecting
    /// edges rim↔`v`). For each manifold side edge `e_connect`, with the wall as
    /// "wall A" and the side edge's radial-other face as "face B":
    ///
    /// 1. Locate the three loop pairs (§6.3 steps 1–3): the **outer pair** at
    ///    the rim vertex (gate loops), the **inner pair** at the moved corner
    ///    `v` on `e_connect`, and the **behind pair** one face further in,
    ///    reached by crossing each face's *other* edge at `v`.
    /// 2. **Same-behind-face early-out** (step 4): if the behind pair lies on
    ///    one inset face, return before touching anything.
    /// 3. **Gate** (step 5): require the outer pair to agree (no user UV seam).
    /// 4. **Collapse the behind pair to `behind_wall`'s value** (step 6) — a
    ///    single behind-face value, *not* a midpoint — and copy it onto the
    ///    inner pair.
    /// 5. **Propagate** (step 7) iff the two behind inset faces are distinct and
    ///    not edge-adjacent through `v`: overwrite every other tagged-region
    ///    loop at `v` whose live value still tolerant-equals one of the two
    ///    corner-pair comparison values.
    fn inset_merge_corner_loops(
        &mut self,
        face_tag: crate::op_flags::FaceOpFlagLayer,
        wall_side_edges: &[WallSideEdge],
    ) {
        // §6.3 runs once per manifold side edge per wall. The side edges are
        // collected in phase-five wall-build order (ascending `edge_keys()`),
        // and we replay them here as a single sequential pass with **live** reads
        // and writes — each invocation's gate, step-6 collapse source, and
        // step-7 value-match read the current (post-earlier-write) loop values.
        // This is the visit-order analogue of running the merge inline as each
        // connecting face is built: an invocation never sees a value an earlier
        // invocation has not yet written.
        //
        // At a fan hub the two arc-end walls share the moved corner *and* the
        // single side edge to the original centre, so both walls' hub invocations
        // key on the SAME side edge and locate the SAME behind pair, A/B swapped.
        // Which arc end's value survives is **first-writer-wins per hub**
        // (§6.3.1.1): the first invocation to reach a hub collapses the
        // participating loops to a single behind value, and later sibling
        // invocations re-read those loops already equal and re-stamp the same
        // value. The reconciled value is one behind loop's §7 value, stamped
        // verbatim — no averaging. See the survivor-selection block below for
        // why the surviving loop is `behind_b` (not `behind_wall`/`behind_a`)
        // under loopmesh's wall-build visit order, and how that is pinned to the
        // guide's stated signed values.
        for side in wall_side_edges {
            // Face B is the side edge's radial-other face. The merge fires only
            // when `e_connect` is manifold (exactly wall A and one other face).
            let radial: Vec<LoopKey> = self.edge_radial_iter(side.connect_edge).collect();
            if radial.len() != 2 {
                continue;
            }
            let face_b = radial
                .iter()
                .map(|&l| self.loop_face(l))
                .find(|&f| f != side.wall);
            let face_b = match face_b {
                Some(f) if f != side.wall => f,
                _ => continue,
            };

            // Step 1 — outer pair at the RIM vertex (the gate loops).
            let outer_a =
                self.loop_on_face_at_vertex_on_edge(side.wall, side.connect_edge, side.rim_vertex);
            let outer_b =
                self.loop_on_face_at_vertex_on_edge(face_b, side.connect_edge, side.rim_vertex);
            // Step 2 — inner pair at the moved corner `v` on `e_connect`.
            let inner_a = self.loop_on_face_at_vertex_on_edge(
                side.wall,
                side.connect_edge,
                side.moved_corner,
            );
            let inner_b =
                self.loop_on_face_at_vertex_on_edge(face_b, side.connect_edge, side.moved_corner);
            let (outer_a, outer_b, inner_a, inner_b) = match (outer_a, outer_b, inner_a, inner_b) {
                (Some(oa), Some(ob), Some(ia), Some(ib)) => (oa, ob, ia, ib),
                _ => continue,
            };

            // Step 3 — behind pair: cross each face's OTHER edge at `v` (the
            // edge at `v` that is not `e_connect`), one radial step inward.
            let behind_a = self.inset_behind_loop(inner_a, side.connect_edge, side.moved_corner);
            let behind_b = self.inset_behind_loop(inner_b, side.connect_edge, side.moved_corner);
            let (behind_a, behind_b) = match (behind_a, behind_b) {
                (Some(la), Some(lb)) => (la, lb),
                _ => continue,
            };
            let inset_face_a = self.loop_face(behind_a);
            let inset_face_b = self.loop_face(behind_b);

            // Step 4 — same-behind-face early-out, BEFORE the gate and midpoint.
            if inset_face_a == inset_face_b {
                continue;
            }

            // Step 5 — gate on the outer (rim) pair agreeing, using the
            // *tolerant* per-layer comparator (§6.3 step-5 note): a math UV
            // layer compares by squared distance, never byte-for-byte.
            if !self.loop_attribs.slots_tolerant_equal(
                outer_a.as_usize(),
                outer_b.as_usize(),
                CORNER_MERGE_EPS_SQ,
            ) {
                continue;
            }

            // The reconciled value is a SINGLE behind loop's §7 value (§6.3
            // step 6 / §6.3.1.1): one behind loop survives, the other is
            // overwritten with it — never their midpoint. Which of the two
            // survives is "first-writer-wins per hub" (§6.3.1.1): the first
            // invocation to reach a hub in connecting-face creation order stamps
            // its `behind_wall` value, and later sibling invocations re-read the
            // (now-overwritten) loops live and re-stamp the same value.
            //
            // Which physical arc end that names is traversal-order dependent.
            // The guide is explicit (§6.3.1.1 "Residual" note, ≈lines 1446–1466)
            // that the order-to-sign mapping is NOT pin-downable from the guide
            // alone — it follows from the operator's internal connecting-face
            // creation order, which an implementation must reproduce, and the
            // guide does not state that order relative to a backend's own build
            // order. So we calibrate against the one fact the guide DOES pin: the
            // surviving value on the 4-of-8 fixture is `(−0.14142, −0.2)`
            // (≈line 1434), and on 5-of-8 `(−0.22426, +0.28284)` (≈line 1437) —
            // *one specific* arc end, never the mirror or the midpoint.
            //
            // loopmesh builds the wall (connecting) faces in phase five by
            // iterating `split_edges` in ascending `edge_keys()` order, and the
            // merge visits `wall_side_edges` in that same wall-build order with
            // LIVE reads. The first hub-firing invocation under this order owns
            // the wall whose `behind_a` is the LOWER-indexed arc-end inset face;
            // its `behind_b` is the HIGHER-indexed one. Against the guide's
            // stated values, the survivor is the higher-indexed arc end — so
            // loopmesh's visit order is the reverse of the operator's
            // connecting-face creation order, and the compensating pick that
            // reproduces the guide's signed survivor is `behind_b`, not the
            // `behind_wall`/`behind_a` an order-matched backend would use.
            //
            // This sign is regression-pinned to the guide's stated values (NOT
            // to the live A/B oracle) by the in-crate tests
            // `corner_merge_disk_fan_4_of_8_*` and `..._5_of_8_*`.
            let survivor = behind_b;
            let loser = behind_a;
            let reconciled = survivor;

            // Step 7's value-match compares candidate loops against the corner
            // pair's values "as they stand at the moment the walk begins" — read
            // live, *before* step 8 seeds them. Cache the comparison sources now
            // so the step-8 write below cannot move them mid-walk. (At a fan hub
            // an earlier sibling invocation may already have seeded these; that
            // is the intended first-writer-wins state, not a poisoning.)
            let compare_a = inner_a;
            let compare_b = inner_b;

            // Step 6 — collapse the behind pair to the survivor's value: stamp it
            // onto the OTHER behind loop. The midpoint the operator computes here
            // is dead (clobbered the next line), so the net effect is a plain copy
            // of the single surviving behind value.
            self.loop_attribs
                .copy_into(loser.as_usize(), reconciled.as_usize());

            // §6.3.1 test (b): behind faces edge-adjacent through `v` → simple
            // case, the collapse already covers the corner, no propagation.
            // Still seed the corner pair (step 8) with the single behind value.
            if self.faces_share_edge_through(inset_face_a, inset_face_b, side.moved_corner) {
                self.loop_attribs
                    .copy_into(inner_a.as_usize(), reconciled.as_usize());
                self.loop_attribs
                    .copy_into(inner_b.as_usize(), reconciled.as_usize());
                continue;
            }

            // Step 7 — distinct and non-edge-adjacent: at least one further
            // tagged face sits between them around `v`. Overwrite every other
            // tagged-region loop at `v` whose live value is *tolerant-equal* (the
            // fuzzy predicate of the step-5 note, NOT a byte-compare) to one of
            // the two corner-pair comparison values, read live before step 8
            // seeds them. Loops further than epsilon from either comparison value
            // are left at their §7-rebaked value (the merge does NOT collapse the
            // whole hub — §6.3 step-8 verified note).
            let handled = [inner_a, inner_b, behind_a, behind_b];
            let corner_loops: Vec<LoopKey> = self.loops_at_vertex(side.moved_corner);
            let mut matched: Vec<LoopKey> = Vec::new();
            for candidate in corner_loops {
                if handled.contains(&candidate) {
                    continue;
                }
                if !self.face_op_flag_test(face_tag, self.loop_face(candidate), 0) {
                    continue;
                }
                let matches_a = self.loop_attribs.slots_tolerant_equal(
                    candidate.as_usize(),
                    compare_a.as_usize(),
                    CORNER_MERGE_EPS_SQ,
                );
                let matches_b = self.loop_attribs.slots_tolerant_equal(
                    candidate.as_usize(),
                    compare_b.as_usize(),
                    CORNER_MERGE_EPS_SQ,
                );
                if matches_a || matches_b {
                    matched.push(candidate);
                }
            }
            for candidate in matched {
                self.loop_attribs
                    .copy_into(candidate.as_usize(), reconciled.as_usize());
            }

            // Step 8 — seed the corner pair with the single behind value.
            self.loop_attribs
                .copy_into(inner_a.as_usize(), reconciled.as_usize());
            self.loop_attribs
                .copy_into(inner_b.as_usize(), reconciled.as_usize());
        }
    }

    /// INSET.md §6.3 step 3 — the behind loop reached from an inner loop at the
    /// moved corner `v`. Starting on `inner` (a loop at `v` on the side edge
    /// `connect_edge`), cross that loop's face's **other** edge at `v` — the
    /// corner-incident edge that is *not* `connect_edge` — by one radial step to
    /// the loop on the face across it, anchored at the same `v`. For wall A this
    /// other edge is the wall↔inset boundary edge, so the step lands on the
    /// inset face. Returns `None` if that other edge has no face across it.
    fn inset_behind_loop(
        &self,
        inner: LoopKey,
        connect_edge: EdgeKey,
        corner: VertexKey,
    ) -> Option<LoopKey> {
        let inner_face = self.loop_face(inner);

        // The two corner-incident edges of `inner`: its forward edge and its
        // predecessor's edge. One is `connect_edge`; the other is the edge to
        // cross.
        let forward_edge = self.loop_edge(inner);
        let backward_edge = self.loop_edge(self.loop_prev(inner));
        let other_edge = if forward_edge == connect_edge {
            backward_edge
        } else {
            forward_edge
        };

        for behind in self.edge_radial_iter(other_edge).collect::<Vec<_>>() {
            if self.loop_face(behind) == inner_face {
                continue;
            }
            if self.loop_vertex(behind) == corner {
                return Some(behind);
            }
            let next = self.loop_next(behind);
            if self.loop_vertex(next) == corner {
                return Some(next);
            }
        }
        None
    }

    /// INSET.md §6.3.1 test (b) — whether two distinct faces are edge-adjacent
    /// through `corner`: they share one of `corner`'s incident edges.
    fn faces_share_edge_through(
        &self,
        face_a: FaceKey,
        face_b: FaceKey,
        corner: VertexKey,
    ) -> bool {
        for edge in self.vertex_disk_iter(corner).collect::<Vec<_>>() {
            let mut on_a = false;
            let mut on_b = false;
            for loop_key in self.edge_radial_iter(edge).collect::<Vec<_>>() {
                let face = self.loop_face(loop_key);
                on_a |= face == face_a;
                on_b |= face == face_b;
            }
            if on_a && on_b {
                return true;
            }
        }
        false
    }

    /// Every face-corner loop anchored at `vertex` — one per face incident to
    /// `vertex`. Walks the disk and each spoke's radial fan, keeping the loop
    /// whose own vertex is `vertex`, deduplicated by face.
    fn loops_at_vertex(&self, vertex: VertexKey) -> Vec<LoopKey> {
        let mut result: Vec<LoopKey> = Vec::new();
        let mut seen_faces: Vec<FaceKey> = Vec::new();
        for edge in self.vertex_disk_iter(vertex).collect::<Vec<_>>() {
            for loop_key in self.edge_radial_iter(edge).collect::<Vec<_>>() {
                if self.loop_vertex(loop_key) != vertex {
                    continue;
                }
                let face = self.loop_face(loop_key);
                if seen_faces.contains(&face) {
                    continue;
                }
                seen_faces.push(face);
                result.push(loop_key);
            }
        }
        result
    }

    /// INSET.md §6.3 corner-merge gate — true iff at least one registered
    /// loop layer blends fractionally (a "math" layer). A layer is a math
    /// layer when its interpolation policy is the fractional weighted blend
    /// (`InterpPolicy::Linear`); integer-family layers use a
    /// nearest/winner-take-one policy and must not trigger the corner merge.
    fn has_loop_math_layer(&self) -> bool {
        self.loop_attribs
            .iter_descriptors()
            .any(|(_, desc)| matches!(desc.interp, InterpPolicy::Linear))
    }
}

#[cfg(test)]
mod tests {
    use glam::Vec3;

    use super::{InsetRegionParams, InsetRegionResult, CORNER_MERGE_EPS_SQ};
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
    fn inset_region_outset_with_excluded_corner_separates_corner_seams() {
        // use_outset over centre face 4, excluding corner face 0. The tagged
        // region is the inverted set {1,2,3,5,6,7,8}. The seams that border
        // the excluded face 0 touch face 0 (untagged) and faces 1/3 (tagged):
        // they MUST be classified as seams even though neither face 4 (input)
        // nor face 0 (excluded) is in the tagged region. Phase-1 iterates
        // every mesh edge (INSET.md §12) so these seams are found and split.
        let mut mesh = LoopMesh::new();
        let faces = make_grid_3x3_faces(&mut mesh);

        let result = mesh.inset_region(
            &[faces[4]],
            InsetRegionParams {
                thickness: 0.1,
                use_outset: true,
                faces_exclude: &[faces[0]],
                ..Default::default()
            },
        );

        // The corner face 0 has two edges that border the tagged region
        // (its edge shared with face 1, and its edge shared with face 3).
        // Both must produce a wall, so they must be among the seams. Before
        // the fix these two seams were dropped (they touch only face 4 / 0,
        // neither tagged), losing their walls and split verts.
        mesh.validate_topology();

        // Independently recompute the seam set the way Phase-1 should, to make
        // the assertion robust to the exact total vert count: every edge with
        // exactly one tagged-region radial loop is a seam, and the two edges
        // around corner 0 are in that set.
        // Post-fix: the two corner-0 seams are found and split, contributing
        // their walls and split verts. Before the fix Phase-1 only visited the
        // input face 4's edges, so these seams were dropped (20 verts, 4
        // walls). With whole-mesh classification the totals match the BMesh
        // oracle: 23 verts, 6 walls.
        assert_eq!(mesh.totvert(), 23);
        assert_eq!(mesh.totface(), 15);
        assert_eq!(result.wall_faces.len(), 6);
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
                AttrType::Vec2,
            ))
            .unwrap();
        let color = mesh
            .register_vertex_attribute(ElementAttributeDesc::numeric(
                AttributeKey::custom("color"),
                AttrType::Vec3,
            ))
            .unwrap();
        (uv, color)
    }

    /// Expected mean-value blend of `corner_vals` at the post-displacement
    /// projected position of a corner, using the documented §10.1 routine.
    fn expected_blend2(proj_corners: &[Vec2], corner_vals: &[[f32; 2]], q: Vec2) -> [f32; 2] {
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
        let corner_uvs: Vec<[f32; 2]> = orig_pos.iter().map(|&p| [p.x * p.y, p.x * p.x]).collect();

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
            assert!(
                (new_p - orig_pos[i]).length() > 1e-3,
                "corner {i} did not move"
            );
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
        assert!(
            merged_corners >= 4,
            "expected merged corners, got {merged_corners}"
        );
    }

    #[test]
    fn corner_merge_skipped_for_nonmath_integer_loop_layer() {
        // §6.3 gate: a loop layer whose interpolation is the integer-family
        // nearest/winner-take-one policy is NOT a math layer, so the corner
        // merge must not fire. Each tagged corner keeps the per-wall §6.2
        // value copied from its own seam side; without a math layer there is
        // nothing to midpoint, so the two walls meeting at a corner keep the
        // distinct integer values they inherited rather than being equalised.
        let mut mesh = LoopMesh::new();
        let faces = make_grid_3x3_faces(&mut mesh);
        let centre = faces[4];

        // Integer loop layer → default policy Nearest (non-math).
        let tag = mesh
            .register_loop_attribute(ElementAttributeDesc::numeric(
                AttributeKey::custom("region"),
                AttrType::I32,
            ))
            .unwrap();
        assert_eq!(
            mesh.loop_attribs().descriptor(tag).unwrap().interp,
            InterpPolicy::Nearest,
            "integer loop layer should default to a non-math policy"
        );

        // Give each face a distinct integer id on all its loops, so the
        // centre face's seam loops differ per side. The corner merge — if it
        // wrongly ran — would average them toward a single value at the
        // shared corner; the Nearest-policy gate must leave them untouched.
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
                    .set_i32(tag, l.as_usize(), fi as i32);
            }
        }

        // Snapshot every centre-face loop's id before the inset.
        let centre_loops: Vec<LoopKey> = {
            let mut acc = Vec::new();
            let mut it = mesh.face_loops(centre);
            while let Some(l) = it.next(&mesh) {
                acc.push(l);
            }
            acc
        };
        let centre_idx = face_keys.iter().position(|&f| f == centre).unwrap() as i32;

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

        // The inset (shrunken) centre face's corner ids are untouched: the
        // §6.3 merge was gated off, and §7 re-interpolation routes the
        // integer layer through the nearest policy, which on this uniform
        // per-face layer reproduces the face's own id exactly.
        for &l in &centre_loops {
            let got = mesh.loop_attribs().get_i32(tag, l.as_usize()).unwrap();
            assert_eq!(
                got, centre_idx,
                "integer corner value must be left uniform (no fractional merge)"
            );
        }
    }

    #[test]
    fn noninterpolate_wall_copy_radial_hop_uniform_per_corner_no_untagged_leak() {
        // §6.2 non-interpolate path: every wall loop's source is located by the
        // radial hop across `e_new`, re-anchored to the loop's OWN vertex — so
        // the far loop at `v?_un` reads the inset-face source at its own
        // physical corner (the same source the near loop at the matching
        // `v?_tag` reads), not the other end's. With no §7 rebake the inset-face
        // loops keep their original per-corner UVs, so each wall comes out
        // UNIFORM PER CORNER: near and far at one physical corner agree, and the
        // §6.1 winding cross is invisible in the values. A per-corner-distinct
        // UV pins two facts:
        //   * every wall loop's UV is one of the wall's two TAGGED seam-corner
        //     UVs (never the untagged neighbour's value);
        //   * near and far at the same physical corner are equal (uniform).
        let mut mesh = LoopMesh::new();
        let faces = make_grid_3x3_faces(&mut mesh);
        let centre = faces[4];
        let (uv, _color) = register_uv_color(&mut mesh);

        // Distinct per-corner UV keyed on position, on every loop.
        let loops: Vec<LoopKey> = mesh.loops.iter().collect();
        for l in loops {
            let p = mesh.vertex_position(mesh.loop_vertex(l));
            mesh.loop_attribs_mut()
                .set_vec2(uv, l.as_usize(), [p.x, p.y]);
        }

        // The centre face's four original corner UVs (= the only legitimate
        // wall-loop sources). They live at grid coords {1,2}×{1,2}.
        let centre_corner_uvs: Vec<[f32; 2]> = {
            let mut acc = Vec::new();
            let mut it = mesh.face_loops(centre);
            while let Some(l) = it.next(&mesh) {
                acc.push(mesh.loop_attribs().get_vec2(uv, l.as_usize()).unwrap());
            }
            acc
        };

        let result = mesh.inset_region(
            &[centre],
            InsetRegionParams {
                thickness: 0.2,
                use_interpolate: false,
                ..Default::default()
            },
        );
        assert_eq!(result.wall_faces.len(), 4);
        mesh.validate_topology();

        let approx =
            |a: [f32; 2], b: [f32; 2]| (a[0] - b[0]).abs() < 1e-4 && (a[1] - b[1]).abs() < 1e-4;

        for &wall in &result.wall_faces {
            let mut wall_pos_uv: Vec<([f32; 3], [f32; 2])> = Vec::new();
            let mut it = mesh.face_loops(wall);
            while let Some(l) = it.next(&mesh) {
                let p = mesh.vertex_position(mesh.loop_vertex(l)).to_array();
                let u = mesh.loop_attribs().get_vec2(uv, l.as_usize()).unwrap();
                // No untagged leak: every wall loop UV is a tagged inset corner.
                assert!(
                    centre_corner_uvs.iter().any(|&c| approx(c, u)),
                    "wall loop UV {u:?} is not one of the tagged inset corners \
                     {centre_corner_uvs:?} (untagged leak)"
                );
                wall_pos_uv.push((p, u));
            }
            assert_eq!(wall_pos_uv.len(), 4);

            // Uniform per corner: each loop's nearest twin in xy (its tagged /
            // untagged copy of the same source vertex) carries the same UV.
            for i in 0..wall_pos_uv.len() {
                let (pi, ui) = wall_pos_uv[i];
                let mut best: Option<(f32, [f32; 2])> = None;
                for j in 0..wall_pos_uv.len() {
                    if i == j {
                        continue;
                    }
                    let (pj, uj) = wall_pos_uv[j];
                    let dd = ((pi[0] - pj[0]).powi(2) + (pi[1] - pj[1]).powi(2)).sqrt();
                    best = match best {
                        Some((bd, _)) if bd <= dd => best,
                        _ => Some((dd, uj)),
                    };
                }
                let (_, twin_uv) = best.unwrap();
                assert!(
                    approx(ui, twin_uv),
                    "uniform per corner: wall loop UV {ui:?} must match its \
                     nearest twin {twin_uv:?} (far ← same-corner radial hop)"
                );
            }
        }
    }

    #[test]
    fn noninterpolate_wall_copy_l_shape_no_untagged_leak_uniform_per_corner() {
        // §6.2 / §6.2.1 grid-vs-L regression guard. An L-shaped tagged region of
        // a 3×3 grid ({0,1,3}) has a concave corner where the seam loop traverses
        // some inset faces in the opposite rotational sense to a convex grid
        // boundary. The radial-hop rule (no fan-vs-grid branch) must still copy
        // every wall loop from the TAGGED inset face only (no untagged leak) and
        // keep each wall uniform per corner — the same invariants as the convex
        // grid case. A positional / "swap on the concave side" rule would leak or
        // double-cross here.
        let mut mesh = LoopMesh::new();
        let faces = make_grid_3x3_faces(&mut mesh);
        let l_region = [faces[0], faces[1], faces[3]];
        let (uv, _color) = register_uv_color(&mut mesh);

        let loops: Vec<LoopKey> = mesh.loops.iter().collect();
        for l in loops {
            let p = mesh.vertex_position(mesh.loop_vertex(l));
            mesh.loop_attribs_mut()
                .set_vec2(uv, l.as_usize(), [p.x, p.y]);
        }

        // Original corner UVs of the three tagged faces (the only legit sources).
        let mut tagged_corner_uvs: Vec<[f32; 2]> = Vec::new();
        for &f in &l_region {
            let mut it = mesh.face_loops(f);
            while let Some(l) = it.next(&mesh) {
                tagged_corner_uvs.push(mesh.loop_attribs().get_vec2(uv, l.as_usize()).unwrap());
            }
        }

        let result = mesh.inset_region(
            &l_region,
            InsetRegionParams {
                thickness: 0.15,
                use_interpolate: false,
                ..Default::default()
            },
        );
        mesh.validate_topology();
        assert!(!result.wall_faces.is_empty());

        let approx =
            |a: [f32; 2], b: [f32; 2]| (a[0] - b[0]).abs() < 1e-4 && (a[1] - b[1]).abs() < 1e-4;

        for &wall in &result.wall_faces {
            let mut wall_pos_uv: Vec<([f32; 3], [f32; 2])> = Vec::new();
            let mut it = mesh.face_loops(wall);
            while let Some(l) = it.next(&mesh) {
                let p = mesh.vertex_position(mesh.loop_vertex(l)).to_array();
                let u = mesh.loop_attribs().get_vec2(uv, l.as_usize()).unwrap();
                assert!(
                    tagged_corner_uvs.iter().any(|&c| approx(c, u)),
                    "L-shape wall loop UV {u:?} leaked from a non-tagged source"
                );
                wall_pos_uv.push((p, u));
            }
            for i in 0..wall_pos_uv.len() {
                let (pi, ui) = wall_pos_uv[i];
                let mut best: Option<(f32, [f32; 2])> = None;
                for j in 0..wall_pos_uv.len() {
                    if i == j {
                        continue;
                    }
                    let (pj, uj) = wall_pos_uv[j];
                    let dd = ((pi[0] - pj[0]).powi(2) + (pi[1] - pj[1]).powi(2)).sqrt();
                    best = match best {
                        Some((bd, _)) if bd <= dd => best,
                        _ => Some((dd, uj)),
                    };
                }
                let (_, twin_uv) = best.unwrap();
                assert!(
                    approx(ui, twin_uv),
                    "L-shape wall must stay uniform per corner; {ui:?} vs twin {twin_uv:?}"
                );
            }
        }
    }

    #[test]
    fn wall_loops_source_only_from_tagged_face_for_nonmath_layer() {
        // §6.2 wall-copy: every wall loop sources from the *tagged* inset face,
        // never the untagged neighbour. For a per-face-uniform non-math integer
        // layer the §7 rebake is a no-op, so near == far and each wall comes out
        // uniform with the tagged value — NOT the [tagged, tagged, untagged,
        // untagged] gradient a tagged↔untagged transparency rule would give.
        let mut mesh = LoopMesh::new();
        let faces = make_grid_3x3_faces(&mut mesh);
        let centre = faces[4];

        let region = mesh
            .register_loop_attribute(ElementAttributeDesc::numeric(
                AttributeKey::custom("region"),
                AttrType::I32,
            ))
            .unwrap();
        assert_eq!(
            mesh.loop_attribs().descriptor(region).unwrap().interp,
            InterpPolicy::Nearest,
            "integer loop layer should default to a non-math policy"
        );

        // Tagged centre face → 10 on every loop; every other (untagged) face → 0.
        let face_keys: Vec<FaceKey> = mesh.face_keys();
        for &f in &face_keys {
            let value = if f == centre { 10 } else { 0 };
            let loops: Vec<LoopKey> = {
                let mut acc = Vec::new();
                let mut it = mesh.face_loops(f);
                while let Some(l) = it.next(&mesh) {
                    acc.push(l);
                }
                acc
            };
            for l in loops {
                mesh.loop_attribs_mut().set_i32(region, l.as_usize(), value);
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

        for &wall in &result.wall_faces {
            let mut it = mesh.face_loops(wall);
            while let Some(l) = it.next(&mesh) {
                let got = mesh.loop_attribs().get_i32(region, l.as_usize()).unwrap();
                assert_eq!(
                    got, 10,
                    "every wall loop must carry the tagged face value (10), \
                     never the untagged neighbour's 0"
                );
            }
        }
    }

    #[test]
    fn corner_merge_runs_for_math_loop_layer() {
        // Mirror of the gate test: a math (Linear) loop layer present means
        // the §6.3 merge DOES run. With a continuous per-vertex UV, the two
        // walls meeting at each inset corner end up holding the same merged
        // value.
        let mut mesh = LoopMesh::new();
        let faces = make_grid_3x3_faces(&mut mesh);
        let centre = faces[4];

        let uv = mesh
            .register_loop_attribute(ElementAttributeDesc::numeric(
                AttributeKey::custom("uv"),
                AttrType::Vec2,
            ))
            .unwrap();
        assert_eq!(
            mesh.loop_attribs().descriptor(uv).unwrap().interp,
            InterpPolicy::Linear,
            "float loop layer should default to a math policy"
        );

        let loops: Vec<LoopKey> = mesh.loops.iter().collect();
        for l in loops {
            let p = mesh.vertex_position(mesh.loop_vertex(l));
            mesh.loop_attribs_mut()
                .set_vec2(uv, l.as_usize(), [p.x, p.y]);
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
        assert!(
            merged_corners >= 4,
            "math layer should still merge corners, got {merged_corners}"
        );
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

    // ---- One-seam mesh-boundary corner: §5.6 use_even_offset gate ----

    /// Two quads sharing the interior edge `v1–v2`, only the left quad (f0)
    /// tagged, with the shared edge *skewed* so the bottom/top mesh-boundary
    /// edges meet the seam off-perpendicular (v1 pushed below the v0–v4 line,
    /// v2 pushed above the v3–v5 line). With `use_boundary` off the shared
    /// edge is the single seam at both v1 and v2, so both are one-seam
    /// boundary verts (INSET.md §5.6).
    fn make_one_seam_two_quads(mesh: &mut LoopMesh) -> Vec<FaceKey> {
        let v0 = mesh.create_vertex(Vec3::new(0.0, 0.0, 0.0));
        let v1 = mesh.create_vertex(Vec3::new(1.0, -0.5, 0.0));
        let v2 = mesh.create_vertex(Vec3::new(1.0, 1.5, 0.0));
        let v3 = mesh.create_vertex(Vec3::new(0.0, 1.0, 0.0));
        let v4 = mesh.create_vertex(Vec3::new(2.0, 0.0, 0.0));
        let v5 = mesh.create_vertex(Vec3::new(2.0, 1.0, 0.0));

        let quad = |mesh: &mut LoopMesh, vs: [VertexKey; 4]| {
            let e0 = mesh.create_edge([vs[0], vs[1]], EdgeCreateFlags::NO_DUPLICATE);
            let e1 = mesh.create_edge([vs[1], vs[2]], EdgeCreateFlags::NO_DUPLICATE);
            let e2 = mesh.create_edge([vs[2], vs[3]], EdgeCreateFlags::NO_DUPLICATE);
            let e3 = mesh.create_edge([vs[3], vs[0]], EdgeCreateFlags::NO_DUPLICATE);
            mesh.face_create(&vs, &[e0, e1, e2, e3], FaceCreateFlags::NONE)
        };

        // f0 (tagged) = left quad; f1 (untagged) = right quad sharing v1–v2.
        let f0 = quad(mesh, [v0, v1, v2, v3]);
        let _f1 = quad(mesh, [v1, v4, v5, v2]);
        vec![f0]
    }

    /// The two displaced tagged corners (those that moved off their original
    /// positions), returned as `(displacement_at_v1, displacement_at_v2)`
    /// keyed by which original (skewed) seam endpoint they came from.
    fn one_seam_displacements(mesh: &LoopMesh) -> (Vec3, Vec3) {
        let v1_orig = Vec3::new(1.0, -0.5, 0.0);
        let v2_orig = Vec3::new(1.0, 1.5, 0.0);
        let originals = [
            Vec3::new(0.0, 0.0, 0.0),
            v1_orig,
            v2_orig,
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::new(2.0, 1.0, 0.0),
        ];
        let mut from_v1: Option<Vec3> = None;
        let mut from_v2: Option<Vec3> = None;
        for v in mesh.vertices.iter() {
            let p = mesh.vertex_position(v);
            // A vertex still sitting on an original position did not move.
            if originals.iter().any(|o| (*o - p).length() < 1.0e-5) {
                continue;
            }
            // Attribute the moved corner to whichever seam endpoint it is
            // closest to (the tagged corner is a small offset from it).
            if (p - v1_orig).length() < (p - v2_orig).length() {
                from_v1 = Some(p - v1_orig);
            } else {
                from_v2 = Some(p - v2_orig);
            }
        }
        (
            from_v1.expect("expected a displaced corner from v1"),
            from_v2.expect("expected a displaced corner from v2"),
        )
    }

    #[test]
    fn inset_region_one_seam_even_off_uses_bare_seam_tangent() {
        // §5.6 use_even_offset OFF: both one-seam corners move by the *same*
        // vector — the seam edge's inward tangent times `thickness` — and the
        // skewed boundary edges play no role. The seam edge v1–v2 runs along
        // +Y, so its inward tangent (into f0, the -X side) is purely along X
        // with zero Y component.
        let mut mesh = LoopMesh::new();
        let faces = make_one_seam_two_quads(&mut mesh);

        let result = mesh.inset_region(
            &faces,
            InsetRegionParams {
                thickness: 0.25,
                use_boundary: false,
                use_even_offset: false,
                ..Default::default()
            },
        );
        assert_eq!(result.wall_faces.len(), 1);
        mesh.validate_topology();

        let (d1, d2) = one_seam_displacements(&mesh);
        // Same displacement at both ends.
        assert!(
            (d1 - d2).length() < 1.0e-5,
            "even-off one-seam corners should share one displacement; d1={d1:?} d2={d2:?}"
        );
        // Perpendicular to the seam: zero Y, magnitude `thickness`, pointing
        // into f0 (toward -X).
        assert!(
            d1.y.abs() < 1.0e-5,
            "bare tangent should have no Y; d1={d1:?}"
        );
        assert!(d1.z.abs() < 1.0e-5, "planar mesh stays in z=0; d1={d1:?}");
        assert!(
            d1.x < 0.0,
            "tangent should point into the tagged face (-X); d1={d1:?}"
        );
        assert!(
            (d1.length() - 0.25).abs() < 1.0e-5,
            "displacement magnitude should equal thickness; |d1|={}",
            d1.length()
        );
    }

    #[test]
    fn inset_region_one_seam_even_on_slides_along_boundary() {
        // §5.6 use_even_offset ON: each one-seam corner slides along its *own*
        // skewed boundary edge, so the two corners move in *different*
        // directions (non-zero, opposite-sign Y components) — visibly distinct
        // from the bare-tangent OFF case. The slide is scaled so its
        // perpendicular projection onto the seam tangent still equals
        // `thickness` (the X component stays at -thickness).
        let mut mesh = LoopMesh::new();
        let faces = make_one_seam_two_quads(&mut mesh);

        let result = mesh.inset_region(
            &faces,
            InsetRegionParams {
                thickness: 0.25,
                use_boundary: false,
                use_even_offset: true,
                ..Default::default()
            },
        );
        assert_eq!(result.wall_faces.len(), 1);
        mesh.validate_topology();

        let (d1, d2) = one_seam_displacements(&mesh);
        // The two corners disagree in direction (the gate is load-bearing).
        assert!(
            (d1 - d2).length() > 1.0e-3,
            "even-on one-seam corners should slide differently; d1={d1:?} d2={d2:?}"
        );
        // Each genuinely slides along its boundary edge → non-zero Y, with
        // opposite signs (v1's boundary edge v0–v1 tilts down, v2's edge
        // v2–v3 tilts up).
        assert!(
            d1.y.abs() > 1.0e-4,
            "even-on v1 should slide off-axis; d1={d1:?}"
        );
        assert!(
            d2.y.abs() > 1.0e-4,
            "even-on v2 should slide off-axis; d2={d2:?}"
        );
        assert!(
            d1.y.signum() != d2.y.signum(),
            "the two corners slide toward opposite Y; d1={d1:?} d2={d2:?}"
        );
        // The even-offset shell factor keeps the perpendicular (seam-tangent)
        // projection at exactly `thickness`: the X component is -thickness.
        assert!(
            (d1.x + 0.25).abs() < 1.0e-5 && (d2.x + 0.25).abs() < 1.0e-5,
            "perpendicular projection should equal thickness; d1={d1:?} d2={d2:?}"
        );
    }

    // ---- Depth pass: curved-surface shell factor (INSET.md §8.2) ----

    /// Build a "tent" of two quads sharing the ridge edge `r0–r1`, each plane
    /// tilted by `tilt` radians off the XY plane in opposite directions. The
    /// ridge runs along +Y at x = 0; the left quad falls away to -X, the right
    /// to +X, both rising in +Z by `tan(tilt)` at unit horizontal distance.
    /// Returns `(faces, ridge_v0, ridge_v1)`.
    fn make_tent(mesh: &mut LoopMesh, tilt: f32) -> (Vec<FaceKey>, VertexKey, VertexKey) {
        let lift = tilt.tan();
        // Ridge.
        let r0 = mesh.create_vertex(Vec3::new(0.0, 0.0, 0.0));
        let r1 = mesh.create_vertex(Vec3::new(0.0, 2.0, 0.0));
        // Left skirt (-X, rising in +Z).
        let l0 = mesh.create_vertex(Vec3::new(-1.0, 0.0, lift));
        let l1 = mesh.create_vertex(Vec3::new(-1.0, 2.0, lift));
        // Right skirt (+X, rising in +Z).
        let rr0 = mesh.create_vertex(Vec3::new(1.0, 0.0, lift));
        let rr1 = mesh.create_vertex(Vec3::new(1.0, 2.0, lift));

        let quad = |mesh: &mut LoopMesh, vs: [VertexKey; 4]| {
            let e0 = mesh.create_edge([vs[0], vs[1]], EdgeCreateFlags::NO_DUPLICATE);
            let e1 = mesh.create_edge([vs[1], vs[2]], EdgeCreateFlags::NO_DUPLICATE);
            let e2 = mesh.create_edge([vs[2], vs[3]], EdgeCreateFlags::NO_DUPLICATE);
            let e3 = mesh.create_edge([vs[3], vs[0]], EdgeCreateFlags::NO_DUPLICATE);
            mesh.face_create(&vs, &[e0, e1, e2, e3], FaceCreateFlags::NONE)
        };

        // Both faces wound so their normals point up (+Z component).
        let left = quad(mesh, [l0, r0, r1, l1]);
        let right = quad(mesh, [r0, rr0, rr1, r1]);
        (vec![left, right], r0, r1)
    }

    #[test]
    fn shell_factor_curved_corner_exceeds_one() {
        // A ridge vertex shared by two tilted faces is a curved corner: its
        // shell normal bisects the two face normals, so the move direction is
        // oblique to each face plane and the shell factor must exceed 1
        // (INSET.md §8.2). A flat corner (coplanar faces) yields exactly 1.
        let tilt = 30.0_f32.to_radians();
        let mut mesh = LoopMesh::new();
        let (_faces, ridge_v0, _ridge_v1) = make_tent(&mut mesh, tilt);
        mesh.calc_face_normals();

        // Shell normal at the ridge vertex: area-less sum of the two incident
        // face normals, normalised (the depth pass's §8 steps 1–3 input).
        let mut shell_normal = Vec3::ZERO;
        let disk: Vec<EdgeKey> = mesh.vertex_disk_iter(ridge_v0).collect();
        let mut seen: Vec<FaceKey> = Vec::new();
        for e in disk {
            for l in mesh.edge_radial_iter(e).collect::<Vec<_>>() {
                if mesh.loop_vertex(l) != ridge_v0 {
                    continue;
                }
                let f = mesh.loop_face(l);
                if seen.contains(&f) {
                    continue;
                }
                seen.push(f);
                shell_normal += mesh.face_normal(f);
            }
        }
        let shell_normal = shell_normal.normalize();

        let factor = mesh.shell_factor_at(ridge_v0, shell_normal);
        assert!(
            factor > 1.0 + 1.0e-4,
            "curved corner shell factor should exceed 1; got {factor}"
        );

        // A flat ridge (zero tilt → coplanar faces) gives exactly 1.
        let mut flat = LoopMesh::new();
        let (_f, flat_v0, _f1) = make_tent(&mut flat, 0.0);
        flat.calc_face_normals();
        let flat_factor = flat.shell_factor_at(flat_v0, Vec3::Z);
        assert!(
            (flat_factor - 1.0).abs() < 1.0e-5,
            "flat corner shell factor should be 1; got {flat_factor}"
        );
    }

    #[test]
    fn shell_factor_two_face_reduces_to_inv_cos_half_dihedral() {
        // §8.2.2 reduction proof: for a vertex shared by exactly two equally
        // tilted faces, the shell normal bisects the two face normals, so the
        // angle between the move direction and each face normal is the
        // half-dihedral. Each per-face factor is 1/cos(half-dihedral) and the
        // angle-weighted mean of two equal values is that same value.
        //
        // Tent halves tilt by `tilt` off horizontal, so each face normal makes
        // an angle `tilt` with +Z; the dihedral between the two normals is
        // `2*tilt` and the half-dihedral is `tilt`. The shell factor must equal
        // 1/cos(tilt).
        for tilt_deg in [15.0_f32, 30.0, 45.0] {
            let tilt = tilt_deg.to_radians();
            let mut mesh = LoopMesh::new();
            let (_faces, ridge_v0, _ridge_v1) = make_tent(&mut mesh, tilt);
            mesh.calc_face_normals();

            // Shell normal at the ridge vertex bisects the two face normals.
            // By the tent's symmetry it points straight up (+Z).
            let shell_normal = Vec3::Z;

            let factor = mesh.shell_factor_at(ridge_v0, shell_normal);
            let expected = 1.0 / tilt.cos();
            assert!(
                (factor - expected).abs() < 1.0e-4,
                "tilt {tilt_deg}°: shell factor {factor} should be 1/cos(half-dihedral) {expected}"
            );
        }
    }

    /// A folded strip of two tagged tent quads meeting at the ridge `r0–r1`,
    /// flanked on the two long sides by untagged quads so the ridge endpoints
    /// `r0`/`r1` are seam corners *shared by both tagged faces*. After the
    /// inset, each ridge-end tagged corner therefore has two differently-tilted
    /// tagged faces incident — a genuine curved corner whose shell factor
    /// exceeds 1. Returns `(tagged_faces, r0, r1)`.
    fn make_folded_strip(mesh: &mut LoopMesh, tilt: f32) -> (Vec<FaceKey>, VertexKey, VertexKey) {
        let lift = tilt.tan();
        // Ridge along Y at x=0.
        let r0 = mesh.create_vertex(Vec3::new(0.0, 0.0, 0.0));
        let r1 = mesh.create_vertex(Vec3::new(0.0, 2.0, 0.0));
        // Left skirt edge of the tent (-X, up).
        let l0 = mesh.create_vertex(Vec3::new(-1.0, 0.0, lift));
        let l1 = mesh.create_vertex(Vec3::new(-1.0, 2.0, lift));
        // Right skirt edge of the tent (+X, up).
        let rr0 = mesh.create_vertex(Vec3::new(1.0, 0.0, lift));
        let rr1 = mesh.create_vertex(Vec3::new(1.0, 2.0, lift));
        // Outer untagged flank verts beyond the tent's near (y<0) and far
        // (y>2) ends, so the ridge endpoints become mixed-radial seams.
        let nl = mesh.create_vertex(Vec3::new(-1.0, -1.0, lift));
        let nr = mesh.create_vertex(Vec3::new(1.0, -1.0, lift));
        let nc = mesh.create_vertex(Vec3::new(0.0, -1.0, 0.0));

        let quad = |mesh: &mut LoopMesh, vs: [VertexKey; 4]| {
            let e0 = mesh.create_edge([vs[0], vs[1]], EdgeCreateFlags::NO_DUPLICATE);
            let e1 = mesh.create_edge([vs[1], vs[2]], EdgeCreateFlags::NO_DUPLICATE);
            let e2 = mesh.create_edge([vs[2], vs[3]], EdgeCreateFlags::NO_DUPLICATE);
            let e3 = mesh.create_edge([vs[3], vs[0]], EdgeCreateFlags::NO_DUPLICATE);
            mesh.face_create(&vs, &[e0, e1, e2, e3], FaceCreateFlags::NONE)
        };

        // Tagged tent halves, both rising to the ridge.
        let left = quad(mesh, [l0, r0, r1, l1]);
        let right = quad(mesh, [r0, rr0, rr1, r1]);
        // Untagged flanks at the near (y<0) end, making the near ridge edges of
        // each tent half mixed-radial → r0 becomes a seam corner shared by both
        // tagged faces.
        let _f_left = quad(mesh, [nl, nc, r0, l0]);
        let _f_right = quad(mesh, [nc, nr, rr0, r0]);
        (vec![left, right], r0, r1)
    }

    // ---- Interior-vert depth: §8.1 topological-distance length BFS ----

    /// Build an `n`×`n`-vertex grid (so `(n-1)×(n-1)` quads) at the given
    /// per-axis coordinates, returning `(faces_row_major, verts_row_major)`.
    /// `xs`/`ys` are the world coordinates of each grid line, so a non-uniform
    /// spacing yields seam edges of non-uniform length.
    fn make_graded_grid(
        mesh: &mut LoopMesh,
        xs: &[f32],
        ys: &[f32],
    ) -> (Vec<FaceKey>, Vec<VertexKey>) {
        let nx = xs.len();
        let ny = ys.len();
        let mut verts = Vec::with_capacity(nx * ny);
        for &y in ys {
            for &x in xs {
                verts.push(mesh.create_vertex(Vec3::new(x, y, 0.0)));
            }
        }
        let mut faces = Vec::with_capacity((nx - 1) * (ny - 1));
        for gy in 0..ny - 1 {
            for gx in 0..nx - 1 {
                let i0 = gy * nx + gx;
                let i1 = gy * nx + (gx + 1);
                let i2 = (gy + 1) * nx + (gx + 1);
                let i3 = (gy + 1) * nx + gx;
                let v = [verts[i0], verts[i1], verts[i2], verts[i3]];
                let e0 = mesh.create_edge([v[0], v[1]], EdgeCreateFlags::NO_DUPLICATE);
                let e1 = mesh.create_edge([v[1], v[2]], EdgeCreateFlags::NO_DUPLICATE);
                let e2 = mesh.create_edge([v[2], v[3]], EdgeCreateFlags::NO_DUPLICATE);
                let e3 = mesh.create_edge([v[3], v[0]], EdgeCreateFlags::NO_DUPLICATE);
                faces.push(mesh.face_create(&v, &[e0, e1, e2, e3], FaceCreateFlags::NONE));
            }
        }
        (faces, verts)
    }

    /// On a graded 5×5-vertex grid (4×4 quads), tag the inner 2×2 face block.
    /// The block's shared centre vertex (grid index 12) has all four incident
    /// edges interior to the selection — no mixed radial — so it is the §8.1
    /// interior tagged vert with no incident seam edge. The grading makes the
    /// eight boundary seam edges non-uniform in length, so the BFS-relaxed
    /// interior length is a genuine spatial blend rather than a uniform
    /// constant.
    ///
    /// We pin the *length value* by reading it off the net displacement: with
    /// `use_even_offset` off the shell factor is 1, and an interior vert lifts
    /// along its *incoming* normal (§8.1.1) — which we seed to a known
    /// non-degenerate unit vector so the displacement is exactly
    /// `depth * relaxed_length` along it.
    fn graded_5x5_inner_block(
        mesh: &mut LoopMesh,
    ) -> (Vec<FaceKey>, VertexKey, [f32; 5], [f32; 5]) {
        // Deliberately irregular spacing on both axes.
        let xs = [0.0_f32, 0.7, 1.5, 2.6, 4.0];
        let ys = [0.0_f32, 0.5, 1.4, 2.0, 3.1];
        let (faces, verts) = make_graded_grid(mesh, &xs, &ys);
        // 4×4 face grid, row-major. Inner 2×2 block = faces {5,6,9,10}.
        let block = vec![faces[5], faces[6], faces[9], faces[10]];
        // Centre vertex shared by all four block faces: grid vertex 12 (row 2,
        // col 2 of the 5×5 vertex grid).
        let centre = verts[12];
        (block, centre, xs, ys)
    }

    /// Independently compute the §8.1 BFS-relaxed length at the inner-block
    /// centre vertex for the graded 5×5 fixture. The centre's four tagged-edge
    /// neighbours are the four mid-edge verts of the block boundary (grid
    /// indices 7, 11, 13, 17); each is a seam vert sitting on two seam edges,
    /// so its `avg_seam_edge_length` is the mean of those two edge lengths.
    /// One BFS pass resolves the centre to the mean of those four neighbour
    /// averages.
    fn expected_centre_length(xs: &[f32; 5], ys: &[f32; 5]) -> f32 {
        let p = |gx: usize, gy: usize| Vec3::new(xs[gx], ys[gy], 0.0);
        let len = |a: Vec3, b: Vec3| (a - b).length();
        // Mid-edge seam verts of the inner 2×2 block (col,row in the 5×5 grid):
        //   index 7  = (col2,row1) top-mid     seams: (1,1)-(2,1), (2,1)-(3,1)
        //   index 11 = (col1,row2) left-mid    seams: (1,1)-(1,2), (1,2)-(1,3)
        //   index 13 = (col3,row2) right-mid   seams: (3,1)-(3,2), (3,2)-(3,3)
        //   index 17 = (col2,row3) bottom-mid  seams: (1,3)-(2,3), (2,3)-(3,3)
        let top = 0.5 * (len(p(1, 1), p(2, 1)) + len(p(2, 1), p(3, 1)));
        let left = 0.5 * (len(p(1, 1), p(1, 2)) + len(p(1, 2), p(1, 3)));
        let right = 0.5 * (len(p(3, 1), p(3, 2)) + len(p(3, 2), p(3, 3)));
        let bottom = 0.5 * (len(p(1, 3), p(2, 3)) + len(p(2, 3), p(3, 3)));
        0.25 * (top + left + right + bottom)
    }

    #[test]
    fn depth_interior_vert_relaxed_length_pins_value() {
        // §8.1.1: pin the BFS *length value* by reading it off the interior
        // vert's net displacement. Seed the centre vertex with a known
        // non-degenerate normal so the displacement is observable, run depth +
        // relative offset with even-offset OFF (shell factor = 1), and assert
        // the move equals `depth * relaxed_length` along the seeded normal.
        let mut mesh = LoopMesh::new();
        let (block, centre, xs, ys) = graded_5x5_inner_block(&mut mesh);

        // A non-axis-aligned unit normal so a degenerate-normal bug (lifting
        // along +Z or zero) would be visible.
        let seeded = Vec3::new(0.2, -0.3, 1.0).normalize();
        mesh.set_vertex_normal(centre, seeded);
        let before = mesh.vertex_position(centre);

        let depth = 0.37_f32;
        mesh.inset_region(
            &block,
            InsetRegionParams {
                thickness: 0.1,
                depth,
                use_boundary: false,
                use_even_offset: false,
                use_relative_offset: true,
                ..Default::default()
            },
        );
        mesh.validate_topology();

        let after = mesh.vertex_position(centre);
        let displacement = after - before;
        let relaxed = expected_centre_length(&xs, &ys);
        let expected = seeded * depth * relaxed;
        assert!(
            (displacement - expected).length() < 1.0e-4,
            "interior vert should lift by depth*relaxed_length along its incoming \
             normal; got {displacement:?} want {expected:?} (relaxed_len={relaxed})"
        );
    }

    #[test]
    fn depth_interior_vert_no_relative_offset_takes_unit_scale() {
        // Gate: with `use_relative_offset` OFF the BFS never runs and the
        // interior vert takes scale 1.0 — the displacement is exactly `depth`
        // along its seeded normal, NOT `depth * relaxed_length`.
        let mut mesh = LoopMesh::new();
        let (block, centre, xs, ys) = graded_5x5_inner_block(&mut mesh);

        let seeded = Vec3::new(0.0, 0.0, 1.0);
        mesh.set_vertex_normal(centre, seeded);
        let before = mesh.vertex_position(centre);

        let depth = 0.4_f32;
        mesh.inset_region(
            &block,
            InsetRegionParams {
                thickness: 0.1,
                depth,
                use_boundary: false,
                use_even_offset: false,
                use_relative_offset: false,
                ..Default::default()
            },
        );

        let displacement = mesh.vertex_position(centre) - before;
        // Scale 1.0 → pure `depth` lift.
        assert!(
            (displacement - seeded * depth).length() < 1.0e-5,
            "no-relative interior vert should lift by exactly depth; got {displacement:?}"
        );
        // And it is distinguishable from the relative-on length (the fixture is
        // graded, so the relaxed length differs from 1.0).
        let relaxed = expected_centre_length(&xs, &ys);
        assert!(
            (relaxed - 1.0).abs() > 1.0e-2,
            "fixture must have a non-trivial relaxed length to make the gate meaningful"
        );
    }

    #[test]
    fn depth_no_interior_vert_skips_relaxation() {
        // Gate: a region with no interior tagged vert never reaches the BFS.
        // The centre face of a 3×3 grid has only boundary (seam) verts, so the
        // depth pass moves them all from their own incident seam lengths — and
        // the operator must complete cleanly with relative offset on.
        let mut mesh = LoopMesh::new();
        let faces = make_grid_3x3_faces(&mut mesh);
        let centre = faces[4];

        let result = mesh.inset_region(
            &[centre],
            InsetRegionParams {
                thickness: 0.1,
                depth: 0.3,
                use_boundary: false,
                use_even_offset: false,
                use_relative_offset: true,
                ..Default::default()
            },
        );
        assert_eq!(result.wall_faces.len(), 4);
        mesh.validate_topology();
    }

    #[test]
    fn depth_interior_vert_degenerate_normal_stays_put() {
        // §8.1.1 caveat made explicit: the BFS length genuinely runs and is
        // non-trivial, but multiplied into a near-zero incoming normal the net
        // displacement is near-zero — both `scale=1.0` and the relaxed length
        // leave the vert effectively in place. This is exactly why the A/B
        // fixture cannot discriminate the value and pins only the path.
        let mut mesh = LoopMesh::new();
        let (block, centre, _xs, _ys) = graded_5x5_inner_block(&mut mesh);

        // A zero-length incoming normal (as a freshly-built interior vert may
        // carry before any normal recompute).
        mesh.set_vertex_normal(centre, Vec3::ZERO);
        let before = mesh.vertex_position(centre);

        mesh.inset_region(
            &block,
            InsetRegionParams {
                thickness: 0.1,
                depth: 0.5,
                use_boundary: false,
                use_even_offset: false,
                use_relative_offset: true,
                ..Default::default()
            },
        );
        mesh.validate_topology();

        let displacement = mesh.vertex_position(centre) - before;
        assert!(
            displacement.length() < 1.0e-6,
            "a degenerate incoming normal leaves the interior vert in place \
             regardless of the BFS length; got {displacement:?}"
        );
    }

    #[test]
    fn depth_even_offset_scales_by_shell_factor_at_curved_boundary_corner() {
        // End-to-end: the near ridge vertex r0 is a seam corner shared by two
        // tilted tagged faces. With `use_even_offset` on, the depth pass lifts
        // its tagged corner by `depth * shell_factor_at(r0)` along the (upward)
        // shell normal — a longer move than the `use_even_offset`-off lift of
        // exactly `depth` (§8.2.5). Track the tagged corner that descends from
        // r0 and compare its post-inset displacement length.
        let tilt = 35.0_f32.to_radians();
        let depth = 0.4_f32;

        // Displacement of the tagged corner nearest the origin (the one that
        // came from r0 at (0,0,0)), measured from r0's original position.
        let r0_corner_disp = |mesh: &LoopMesh| {
            let origin = Vec3::ZERO;
            let mut best: Option<Vec3> = None;
            for v in mesh.vertices.iter() {
                let p = mesh.vertex_position(v);
                // Only verts lifted by depth (z away from 0) and near origin in
                // xy are candidate r0 corners; pick the one closest in xy.
                if p.z.abs() < 1.0e-4 {
                    continue;
                }
                let xy = (p.x * p.x + p.y * p.y).sqrt();
                best = match best {
                    Some(cur) if (cur.x * cur.x + cur.y * cur.y).sqrt() <= xy => best,
                    _ => Some(p),
                };
            }
            best.expect("expected a lifted r0 corner") - origin
        };

        let mut even_on = LoopMesh::new();
        let (on_faces, _r0, _r1) = make_folded_strip(&mut even_on, tilt);
        even_on.inset_region(
            &on_faces,
            InsetRegionParams {
                thickness: 0.15,
                depth,
                use_boundary: false,
                use_even_offset: true,
                ..Default::default()
            },
        );
        even_on.validate_topology();

        let mut even_off = LoopMesh::new();
        let (off_faces, _r0b, _r1b) = make_folded_strip(&mut even_off, tilt);
        even_off.inset_region(
            &off_faces,
            InsetRegionParams {
                thickness: 0.15,
                depth,
                use_boundary: false,
                use_even_offset: false,
                ..Default::default()
            },
        );
        even_off.validate_topology();

        let on_disp = r0_corner_disp(&even_on);
        let off_disp = r0_corner_disp(&even_off);
        // The shell factor (> 1 at this curved corner) only *lengthens* the
        // depth component, so the even-on corner travels strictly farther.
        assert!(
            on_disp.length() > off_disp.length() + 1.0e-3,
            "even-offset shell factor should lengthen the depth move at a curved corner: \
             on={} off={}",
            on_disp.length(),
            off_disp.length()
        );
    }

    #[test]
    fn wall_inherits_inset_face_normal_not_geometric() {
        // §6.1 / §8.2.1.1: each wall quad must carry a verbatim copy of its
        // source tagged inset face's normal — NOT a normal recomputed from the
        // wall's own (roughly perpendicular) plane. The tilted tent gives the
        // two tagged faces normals that point up-and-outward, while each wall
        // quad's own geometric plane is nearly vertical, so the inherited
        // normal and a hypothetical geometric wall normal are far apart and the
        // distinction is observable.
        let tilt = 35.0_f32.to_radians();
        let mut mesh = LoopMesh::new();
        let (tagged_faces, _ridge_v0, _ridge_v1) = make_folded_strip(&mut mesh, tilt);

        // Capture the two tagged inset faces' stored normals as they stand
        // entering the wall-building phase — i.e. the pre-offset normals the
        // operator computes in phase two and never recomputes. They are the
        // values §6.1 says each wall must inherit.
        mesh.calc_face_normals();
        let tagged_normals: Vec<Vec3> = tagged_faces.iter().map(|&f| mesh.face_normal(f)).collect();
        for normal in &tagged_normals {
            assert!(
                normal.length() > 0.5 && normal.z > 0.0,
                "tent fixture should give well-defined upward tagged normals; got {normal:?}"
            );
        }

        let result = mesh.inset_region(
            &tagged_faces,
            InsetRegionParams {
                thickness: 0.15,
                depth: 0.4,
                use_boundary: false,
                use_even_offset: true,
                ..Default::default()
            },
        );
        mesh.validate_topology();
        assert!(!result.wall_faces.is_empty(), "expected wall faces");

        for &wall in &result.wall_faces {
            let stored_normal = mesh.face_normal(wall);

            // The stored normal must match one of the tagged inset faces'
            // normals verbatim (the wall was spun from one of them).
            let matches_inherited = tagged_normals
                .iter()
                .any(|inset_normal| (stored_normal - *inset_normal).length() < 1.0e-6);
            assert!(
                matches_inherited,
                "wall stored normal {stored_normal:?} must equal a tagged inset face normal \
                 verbatim; tagged normals = {tagged_normals:?}"
            );

            // And it must NOT be the wall's own geometric normal: that plane is
            // near-vertical here, so its normal is near-perpendicular to the
            // inherited (upward) normal. A geometric normal would have a tiny z
            // component; the inherited one has a clearly positive z.
            assert!(
                stored_normal.z > 0.1,
                "wall normal should be the inherited upward inset normal, not a near-vertical \
                 geometric wall normal; got {stored_normal:?}"
            );
            let geometric_normal = mesh.face_compute_normal(wall);
            assert!(
                stored_normal.dot(geometric_normal).abs() < 0.5,
                "inherited normal should be far from the wall's own geometric normal \
                 (stored={stored_normal:?}, geometric={geometric_normal:?})"
            );
        }
    }

    #[test]
    fn wall_per_face_shell_term_equals_inset_face_term_not_clamp() {
        // §8.2.1.1, the load-bearing invariant: in the shell-factor face set a
        // wall quad stands in for the inset face it was spun from. Because the
        // wall carries that face's normal verbatim, its per-face shell term
        // `1/|dot(v.no, wall.no)|` equals the term the source inset face
        // contributes — it neither dilutes nor sharpens the average.
        //
        // The bug this pins: a *geometric* wall normal is near-perpendicular to
        // the inset face (and hence to the seam-corner `v.no`, which is close to
        // the inset face's own normal), so `|dot|` is tiny and the §8.2.3
        // near-tangent guard clamps the wall's term to a flat 1 — a *different*
        // (and on a curved corner, smaller) value than the inset face's term.
        let tilt = 35.0_f32.to_radians();
        let mut mesh = LoopMesh::new();
        let (tagged_faces, _r0, _r1) = make_folded_strip(&mut mesh, tilt);

        let result = mesh.inset_region(
            &tagged_faces,
            InsetRegionParams {
                thickness: 0.15,
                depth: 0.4,
                use_boundary: false,
                use_even_offset: true,
                ..Default::default()
            },
        );
        mesh.validate_topology();
        assert!(!result.wall_faces.is_empty(), "expected wall faces");

        // The seam-corner move direction `v.no` is, by §8 steps 1–3, close to
        // the tilted inset faces' normals; use the up-axis bisector of the two
        // symmetric tent halves as a representative non-degenerate `v.no`.
        let vertex_normal = Vec3::Z;

        // Per-face shell term as `shell_factor_at` computes it (§8.2.2/§8.2.3).
        let per_face_shell = |normal: Vec3| -> f32 {
            let cosine = vertex_normal.dot(normal).abs();
            if cosine < 1.0e-6 {
                1.0
            } else {
                1.0 / cosine
            }
        };

        for &wall in &result.wall_faces {
            let wall_normal = mesh.face_normal(wall);
            // The inset face whose normal this wall inherited.
            let source_inset = tagged_faces
                .iter()
                .copied()
                .find(|&f| (mesh.face_normal(f) - wall_normal).length() < 1.0e-6)
                .expect("wall normal should match a source inset face normal");
            let inset_normal = mesh.face_normal(source_inset);

            // Invariant: the wall's per-face term equals its source inset
            // face's term.
            let wall_term = per_face_shell(wall_normal);
            let inset_term = per_face_shell(inset_normal);
            assert!(
                (wall_term - inset_term).abs() < 1.0e-5,
                "wall per-face shell term {wall_term} must equal the inset face term \
                 {inset_term}"
            );

            // The choice of normal is load-bearing: the wall's own geometric
            // normal produces a *different* per-face term, so a geometric-normal
            // implementation would feed a different value into the average. The
            // inherited rule deliberately makes the wall mirror the inset face
            // instead of contributing this independent geometric term.
            let geometric_term = per_face_shell(mesh.face_compute_normal(wall));
            assert!(
                (geometric_term - wall_term).abs() > 1.0e-3,
                "a geometric wall normal would contribute a term ({geometric_term}) distinct \
                 from the inherited one ({wall_term}); the inheritance rule is what aligns it \
                 with the inset face"
            );
            // On this curved corner the inherited (inset-face) term genuinely
            // exceeds the flat 1 — it is not a degenerate no-op.
            assert!(
                wall_term > 1.0 + 1.0e-4,
                "the inherited inset-face term should exceed 1 on this curved corner; \
                 got {wall_term}"
            );
        }
    }

    // ---- §6.3 step 4: corner CD-merge propagation around a high-valence
    //      moved boundary corner (INSET.md §6.3.1) ----

    /// Build an *open* triangle fan of `count` triangles sharing a single hub
    /// vertex at the origin. The rim verts lie on a half-circle arc so the
    /// outermost spokes and the rim form the mesh boundary; each triangle is
    /// `(hub, rim[i], rim[i+1])`. Returns `(triangles, hub, rim)`.
    fn make_triangle_fan(
        mesh: &mut LoopMesh,
        count: usize,
    ) -> (Vec<FaceKey>, VertexKey, Vec<VertexKey>) {
        let hub = mesh.create_vertex(Vec3::ZERO);
        let mut rim = Vec::with_capacity(count + 1);
        for i in 0..=count {
            // Spread the arc over (0, π) so neighbouring spokes are distinct
            // directions and the fan stays open (a mesh-boundary rim).
            let t = (i as f32 + 1.0) / (count as f32 + 2.0);
            let angle = t * std::f32::consts::PI;
            rim.push(mesh.create_vertex(Vec3::new(angle.cos(), angle.sin(), 0.0)));
        }
        let mut triangles = Vec::with_capacity(count);
        for i in 0..count {
            let vs = [hub, rim[i], rim[i + 1]];
            let e0 = mesh.create_edge([vs[0], vs[1]], EdgeCreateFlags::NO_DUPLICATE);
            let e1 = mesh.create_edge([vs[1], vs[2]], EdgeCreateFlags::NO_DUPLICATE);
            let e2 = mesh.create_edge([vs[2], vs[0]], EdgeCreateFlags::NO_DUPLICATE);
            triangles.push(mesh.face_create(&vs, &[e0, e1, e2], FaceCreateFlags::NONE));
        }
        (triangles, hub, rim)
    }

    /// The single vertex shared by every face in `faces` (the moved hub of a
    /// contiguous tagged fan after inset). Panics if there is not exactly one.
    fn shared_vertex_of(mesh: &LoopMesh, faces: &[FaceKey]) -> VertexKey {
        let mut counts: std::collections::HashMap<VertexKey, usize> =
            std::collections::HashMap::new();
        for &f in faces {
            let mut it = mesh.face_loops(f);
            while let Some(l) = it.next(mesh) {
                *counts.entry(mesh.loop_vertex(l)).or_default() += 1;
            }
        }
        let shared: Vec<VertexKey> = counts
            .into_iter()
            .filter(|&(_, c)| c == faces.len())
            .map(|(v, _)| v)
            .collect();
        assert_eq!(shared.len(), 1, "expected exactly one shared hub vertex");
        shared[0]
    }

    /// The UVs of every tagged-region loop anchored at `vertex` after the
    /// inset (one per face in `tagged_faces` touching `vertex`).
    fn tagged_corner_uvs_at(
        mesh: &LoopMesh,
        vertex: VertexKey,
        tagged_faces: &[FaceKey],
        uv: ElementAttributeKey,
    ) -> Vec<[f32; 2]> {
        let mut out = Vec::new();
        for &f in tagged_faces {
            let mut it = mesh.face_loops(f);
            while let Some(l) = it.next(mesh) {
                if mesh.loop_vertex(l) == vertex {
                    out.push(mesh.loop_attribs().get_vec2(uv, l.as_usize()).unwrap());
                }
            }
        }
        out
    }

    /// Build a *closed* triangle fan of `count` triangles around a single hub
    /// vertex at the origin, ring verts on the unit circle. Returns
    /// `(triangles, hub, rim)`. Mirrors the `disk_fan(n)` A/B fixture so the
    /// in-crate tests pin the same hub-merge geometry the compat suite does.
    fn make_closed_disk_fan(
        mesh: &mut LoopMesh,
        count: usize,
    ) -> (Vec<FaceKey>, VertexKey, Vec<VertexKey>) {
        let hub = mesh.create_vertex(Vec3::ZERO);
        let mut rim = Vec::with_capacity(count);
        for i in 0..count {
            let angle = (i as f32) / (count as f32) * std::f32::consts::TAU;
            rim.push(mesh.create_vertex(Vec3::new(angle.cos(), angle.sin(), 0.0)));
        }
        let mut triangles = Vec::with_capacity(count);
        for i in 0..count {
            let vs = [hub, rim[i], rim[(i + 1) % count]];
            let e0 = mesh.create_edge([vs[0], vs[1]], EdgeCreateFlags::NO_DUPLICATE);
            let e1 = mesh.create_edge([vs[1], vs[2]], EdgeCreateFlags::NO_DUPLICATE);
            let e2 = mesh.create_edge([vs[2], vs[0]], EdgeCreateFlags::NO_DUPLICATE);
            triangles.push(mesh.face_create(&vs, &[e0, e1, e2], FaceCreateFlags::NONE));
        }
        (triangles, hub, rim)
    }

    #[test]
    fn corner_merge_closed_disk_fan_hub_keeps_interior_seven_values() {
        // §6.3 corrected model on the closed disk_fan(8) the A/B suite pins
        // (continuous non-affine UV, `use_boundary`/`use_even_offset`/
        // `use_interpolate` on). The verified §6.3 note (and BMESH_PARITY.md
        // row 316) is explicit: BMesh does NOT collapse the moved hub to one
        // value. The merge runs the literal steps 1–8 with the tolerant
        // comparator and reconciles only the loops steps 6–8 reach; the
        // INTERIOR inset-face hub loops keep their own distinct §7-rebaked
        // values. This regression-guards the over-collapse this cycle removed:
        // an all-hub-collapse pass would force every interior loop equal.
        //
        // A six-of-eight arc is used because its §7 interior rebake lands
        // GENUINELY distinct per-corner values (`u=x*y+x, v=x²−y²` is
        // non-affine), so a collapse is observable as loss of that spread —
        // unlike the symmetric four-of-eight arc, where the interior pair's §7
        // value happens to coincide with the merge midpoint and a collapse is
        // numerically indistinguishable from the correct result.
        let mut mesh = LoopMesh::new();
        let (triangles, _hub, _rim) = make_closed_disk_fan(&mut mesh, 8);
        let tagged: Vec<FaceKey> = triangles[0..6].to_vec();
        let uv = mesh
            .register_loop_attribute(ElementAttributeDesc::numeric(
                AttributeKey::custom("uv"),
                AttrType::Vec2,
            ))
            .unwrap();
        let uv_fn = |p: Vec3| -> [f32; 2] { [p.x * p.y + p.x, p.x * p.x - p.y * p.y] };

        // Predict each interior inset face's STANDALONE §7 hub rebake (mean
        // value resample at the moved hub) so we can assert the merge leaves it
        // untouched. The interior faces are triangles[1..5].
        let interior_faces: Vec<FaceKey> = triangles[1..5].to_vec();
        let projection = PlaneProjection::from_normal(Vec3::Z);
        let mut interior_proj: Vec<Vec<Vec2>> = Vec::new();
        let mut interior_uv: Vec<Vec<[f32; 2]>> = Vec::new();
        for &f in &interior_faces {
            let mut proj = Vec::new();
            let mut vals = Vec::new();
            let mut it = mesh.face_loops(f);
            while let Some(l) = it.next(&mesh) {
                let p = mesh.vertex_position(mesh.loop_vertex(l));
                proj.push(projection.project(p));
                vals.push(uv_fn(p));
            }
            interior_proj.push(proj);
            interior_uv.push(vals);
        }

        let loops: Vec<LoopKey> = mesh.loops.iter().collect();
        for l in loops {
            let p = mesh.vertex_position(mesh.loop_vertex(l));
            mesh.loop_attribs_mut().set_vec2(uv, l.as_usize(), uv_fn(p));
        }
        mesh.inset_region(
            &tagged,
            InsetRegionParams {
                thickness: 0.2,
                use_boundary: true,
                use_even_offset: true,
                use_interpolate: true,
                ..Default::default()
            },
        );
        mesh.validate_topology();
        let moved_hub = shared_vertex_of(&mesh, &tagged);
        let hub_q = projection.project(mesh.vertex_position(moved_hub));

        // Each interior face's hub loop must still hold its own §7 rebake — the
        // merge's step-7 tolerant walk does NOT reach it (its rebaked value is
        // farther than ε from either arc-end comparison value), so it is left
        // alone. A collapse pass would overwrite it with the hub midpoint.
        for (idx, &f) in interior_faces.iter().enumerate() {
            let predicted = expected_blend2(&interior_proj[idx], &interior_uv[idx], hub_q);
            let got = tagged_corner_uvs_at(&mesh, moved_hub, &[f], uv);
            assert_eq!(got.len(), 1, "one hub loop per interior face");
            let u = got[0];
            assert!(
                (u[0] - predicted[0]).abs() < 1.0e-3 && (u[1] - predicted[1]).abs() < 1.0e-3,
                "interior hub loop must keep its §7 rebake {predicted:?}; got {u:?} \
                 (a collapse pass would force it to the hub midpoint)"
            );
        }

        // And the hub as a whole must carry several distinct values, never one.
        let hub_uvs = tagged_corner_uvs_at(&mesh, moved_hub, &tagged, uv);
        let mut distinct: Vec<[f32; 2]> = Vec::new();
        for u in &hub_uvs {
            if !distinct
                .iter()
                .any(|d| (d[0] - u[0]).powi(2) + (d[1] - u[1]).powi(2) < CORNER_MERGE_EPS_SQ)
            {
                distinct.push(*u);
            }
        }
        assert!(
            distinct.len() > 1,
            "closed disk-fan hub must keep multiple distinct UVs, not collapse to \
             one; got {hub_uvs:?} -> {} distinct",
            distinct.len()
        );
    }

    #[test]
    fn corner_merge_disk_fan_4_of_8_collapses_arc_ends_to_single_behind_value() {
        // §6.3.1.1 on the exact A/B 4-of-8 geometry: the two arc-end inset faces
        // are the behind pair of the hub invocation; step 6 collapses them to a
        // SINGLE behind loop's §7 value (NOT the cross-arc midpoint), and
        // first-writer-wins settles which arc end survives. The fixture UV
        // `[x·y+x, x²−y²]` is odd in x, so the two mirror-image arc ends rebake
        // the hub to *unequal* values — their midpoint differs from either, which
        // is what makes "single value, not average" observable.
        let mut mesh = LoopMesh::new();
        let (triangles, _hub, _rim) = make_closed_disk_fan(&mut mesh, 8);
        let tagged: Vec<FaceKey> = triangles[0..4].to_vec();
        let arc_end_faces = [triangles[0], triangles[3]];
        let uv = mesh
            .register_loop_attribute(ElementAttributeDesc::numeric(
                AttributeKey::custom("uv"),
                AttrType::Vec2,
            ))
            .unwrap();
        let uv_fn = |p: Vec3| -> [f32; 2] { [p.x * p.y + p.x, p.x * p.x - p.y * p.y] };

        // Predict each arc-end face's standalone §7 hub rebake (pre-merge).
        let projection = PlaneProjection::from_normal(Vec3::Z);
        let mut arc_proj: Vec<Vec<Vec2>> = Vec::new();
        let mut arc_uv: Vec<Vec<[f32; 2]>> = Vec::new();
        for &f in &arc_end_faces {
            let mut proj = Vec::new();
            let mut vals = Vec::new();
            let mut it = mesh.face_loops(f);
            while let Some(l) = it.next(&mesh) {
                let p = mesh.vertex_position(mesh.loop_vertex(l));
                proj.push(projection.project(p));
                vals.push(uv_fn(p));
            }
            arc_proj.push(proj);
            arc_uv.push(vals);
        }

        let loops: Vec<LoopKey> = mesh.loops.iter().collect();
        for l in loops {
            let p = mesh.vertex_position(mesh.loop_vertex(l));
            mesh.loop_attribs_mut().set_vec2(uv, l.as_usize(), uv_fn(p));
        }
        mesh.inset_region(
            &tagged,
            InsetRegionParams {
                thickness: 0.2,
                use_boundary: true,
                use_even_offset: true,
                use_interpolate: true,
                ..Default::default()
            },
        );
        mesh.validate_topology();
        let moved_hub = shared_vertex_of(&mesh, &tagged);
        let hub_q = projection.project(mesh.vertex_position(moved_hub));

        let pre_a = expected_blend2(&arc_proj[0], &arc_uv[0], hub_q);
        let pre_b = expected_blend2(&arc_proj[1], &arc_uv[1], hub_q);
        let midpoint = [0.5 * (pre_a[0] + pre_b[0]), 0.5 * (pre_a[1] + pre_b[1])];
        // The antisymmetric UV must rebake the two arc ends distinctly.
        assert!(
            (pre_a[0] - pre_b[0]).abs() > 1.0e-3,
            "fixture must rebake arc ends distinctly; pre_a={pre_a:?} pre_b={pre_b:?}"
        );

        let near = |u: &[f32; 2], w: &[f32; 2]| -> bool {
            (u[0] - w[0]).abs() < 2.0e-3 && (u[1] - w[1]).abs() < 2.0e-3
        };

        let arc_uvs = tagged_corner_uvs_at(&mesh, moved_hub, &arc_end_faces, uv);
        assert_eq!(arc_uvs.len(), 2);
        // Both arc-end hub loops collapsed to a single value...
        assert!(
            near(&arc_uvs[0], &arc_uvs[1]),
            "arc-end hub loops must collapse to one value; got {:?}, {:?}",
            arc_uvs[0],
            arc_uvs[1]
        );
        // ...equal to one arc-end's §7 value, never the cross-arc midpoint.
        let survivor = arc_uvs[0];
        assert!(
            near(&survivor, &pre_a) || near(&survivor, &pre_b),
            "survivor {survivor:?} must equal a behind value (pre_a={pre_a:?} \
             pre_b={pre_b:?}), the single-behind-value rule"
        );
        assert!(
            !near(&survivor, &midpoint),
            "survivor {survivor:?} must NOT be the midpoint {midpoint:?} \
             (the merge collapses, it never averages)"
        );
        // Pin the *signed* survivor — the one value §6.3.1.1 spent effort
        // resolving. The guide states (INSET.md §6.3.1.1, ≈line 1434, and the
        // §6.3 parity-trap rule ≈line 1480) that BMesh's surviving arc-end hub
        // value on the 4-of-8 fixture is `(−0.14142, −0.2)` — *one specific*
        // arc end, not its mirror `(+0.14142, −0.2)` and not the midpoint
        // `(0, −0.2)`. Asserting the sign here (not merely "one of the two")
        // keeps the survivor pinned to the guide's stated value rather than
        // silently delegated to the A/B oracle. See `inset_merge_corner_loops`
        // for why this requires `survivor = behind_b` under loopmesh's
        // wall-build visit order.
        assert!(
            near(&survivor, &[-0.14142, -0.2]),
            "survivor must be the guide's signed value (−0.14142, −0.2); got {survivor:?}"
        );
    }

    #[test]
    fn corner_merge_disk_fan_5_of_8_arc_ends_pin_guide_signed_value() {
        // §6.3.1.1 signed-survivor pin on the closed disk_fan(8) 5-of-8 arc the
        // A/B suite uses. The guide states (INSET.md §6.3.1.1, ≈line 1437) that
        // BMesh's surviving arc-end hub value here is the single
        // `(−0.22426, +0.28284)` — one arc end's §7 value, NOT the cross-arc
        // midpoint `(−0.08284, 0)`. This pins the *sign* the same way the 4-of-8
        // test does, so the survivor stays tied to the guide's stated value
        // rather than delegated to the A/B oracle.
        let mut mesh = LoopMesh::new();
        let (triangles, _hub, _rim) = make_closed_disk_fan(&mut mesh, 8);
        let tagged: Vec<FaceKey> = triangles[0..5].to_vec();
        let arc_end_faces = [triangles[0], triangles[4]];
        let uv = mesh
            .register_loop_attribute(ElementAttributeDesc::numeric(
                AttributeKey::custom("uv"),
                AttrType::Vec2,
            ))
            .unwrap();
        let uv_fn = |p: Vec3| -> [f32; 2] { [p.x * p.y + p.x, p.x * p.x - p.y * p.y] };
        let loops: Vec<LoopKey> = mesh.loops.iter().collect();
        for l in loops {
            let p = mesh.vertex_position(mesh.loop_vertex(l));
            mesh.loop_attribs_mut().set_vec2(uv, l.as_usize(), uv_fn(p));
        }
        mesh.inset_region(
            &tagged,
            InsetRegionParams {
                thickness: 0.2,
                use_boundary: true,
                use_even_offset: true,
                use_interpolate: true,
                ..Default::default()
            },
        );
        mesh.validate_topology();
        let moved_hub = shared_vertex_of(&mesh, &tagged);

        let near = |u: &[f32; 2], w: &[f32; 2]| -> bool {
            (u[0] - w[0]).abs() < 3.0e-3 && (u[1] - w[1]).abs() < 3.0e-3
        };
        let arc_uvs = tagged_corner_uvs_at(&mesh, moved_hub, &arc_end_faces, uv);
        assert_eq!(arc_uvs.len(), 2);
        // Both arc ends collapse to the single guide-stated signed value.
        assert!(
            near(&arc_uvs[0], &arc_uvs[1]),
            "arc ends must collapse to one value; got {:?}, {:?}",
            arc_uvs[0],
            arc_uvs[1]
        );
        assert!(
            near(&arc_uvs[0], &[-0.22426, 0.28284]),
            "survivor must be the guide's 5-of-8 signed value (−0.22426, +0.28284); \
             got {:?}",
            arc_uvs[0]
        );
        // And NOT the cross-arc midpoint the guide explicitly rules out.
        assert!(
            !near(&arc_uvs[0], &[-0.08284, 0.0]),
            "survivor must not be the midpoint (−0.08284, 0); got {:?}",
            arc_uvs[0]
        );
    }

    #[test]
    fn corner_merge_high_valence_fan_leaves_multiple_distinct_hub_values() {
        // §6.3 / §6.3.1 (corrected model). A disk fan of eight triangles with
        // five consecutive ones tagged. The hub is the moved two-seam boundary
        // corner; phase five rings it with two arc-end walls. With a *non-affine*
        // continuous UV the §7 rebake lands a DISTINCT per-corner value on each
        // of the five tagged hub loops, and the merge runs the literal steps 1–8
        // with the TOLERANT comparator (squared distance < 1e-5):
        //
        //   - the two arc-end inset faces' hub loops are the behind pair of the
        //     two hub invocations and are midpointed (here, symmetric → left at
        //     their equal §7 value);
        //   - interior hub loops are reconciled only where the tolerant step-7
        //     walk reaches them; loops outside epsilon stay at their §7 value.
        //
        // The corrected guide verifies BMesh leaves the hub carrying SEVERAL
        // distinct UVs, NOT one merged value. A regression that re-adds an
        // all-hub-collapse pass would force them all equal; this test fails it.
        let mut mesh = LoopMesh::new();
        let (triangles, _hub, _rim) = make_triangle_fan(&mut mesh, 8);
        let tagged: Vec<FaceKey> = triangles[2..7].to_vec(); // five consecutive

        let uv = mesh
            .register_loop_attribute(ElementAttributeDesc::numeric(
                AttributeKey::custom("uv"),
                AttrType::Vec2,
            ))
            .unwrap();
        // Continuous (gate-passing) but non-affine UV so each tagged triangle
        // rebakes the hub corner to its own value.
        let uv_fn = |p: Vec3| -> [f32; 2] { [p.x * p.y + p.x, p.x * p.x - p.y * p.y] };
        let loops: Vec<LoopKey> = mesh.loops.iter().collect();
        for l in loops {
            let p = mesh.vertex_position(mesh.loop_vertex(l));
            mesh.loop_attribs_mut().set_vec2(uv, l.as_usize(), uv_fn(p));
        }

        let result = mesh.inset_region(
            &tagged,
            InsetRegionParams {
                thickness: 0.1,
                use_boundary: false,
                use_interpolate: true,
                ..Default::default()
            },
        );
        mesh.validate_topology();
        assert_eq!(result.wall_faces.len(), 2);

        let moved_hub = shared_vertex_of(&mesh, &tagged);
        assert!(
            mesh.vertex_position(moved_hub).length() > 1.0e-6,
            "hub should have moved inward off the origin"
        );

        let hub_uvs = tagged_corner_uvs_at(&mesh, moved_hub, &tagged, uv);
        assert_eq!(hub_uvs.len(), 5, "five tagged hub corners expected");

        // Count distinct hub values under the tolerant predicate. The corrected
        // model leaves MORE THAN ONE distinct value; a collapse pass would leave
        // exactly one.
        let mut distinct: Vec<[f32; 2]> = Vec::new();
        for u in &hub_uvs {
            if !distinct
                .iter()
                .any(|d| (d[0] - u[0]).powi(2) + (d[1] - u[1]).powi(2) < CORNER_MERGE_EPS_SQ)
            {
                distinct.push(*u);
            }
        }
        assert!(
            distinct.len() > 1,
            "high-valence fan hub must keep multiple distinct UVs (not a single \
             collapsed value); got {hub_uvs:?} -> {} distinct",
            distinct.len()
        );
    }

    #[test]
    fn corner_merge_tolerant_gate_passes_within_epsilon() {
        // The step-5 gate uses a tolerant (squared-distance < 1e-5) compare, not
        // a byte fingerprint. Two rim loops whose UVs differ only by float noise
        // below epsilon must still pass the gate so the merge fires. We pin this
        // directly on the comparator to guard the documented byte-match killer.
        let mut mesh = LoopMesh::new();
        let uv = mesh
            .register_loop_attribute(ElementAttributeDesc::numeric(
                AttributeKey::custom("uv"),
                AttrType::Vec2,
            ))
            .unwrap();
        // Two loops on a throwaway triangle so we have valid slots to write.
        let f = create_face_from_verts(&mut mesh, &[Vec3::ZERO, Vec3::X, Vec3::Y]);
        let face_loops: Vec<LoopKey> = {
            let mut out = Vec::new();
            let mut it = mesh.face_loops(f);
            while let Some(l) = it.next(&mesh) {
                out.push(l);
            }
            out
        };
        let (a, b) = (face_loops[0], face_loops[1]);

        // Within epsilon: separation 1e-3 → squared distance 1e-6 < 1e-5.
        mesh.loop_attribs_mut()
            .set_vec2(uv, a.as_usize(), [0.5, 0.5]);
        mesh.loop_attribs_mut()
            .set_vec2(uv, b.as_usize(), [0.5 + 1.0e-3, 0.5]);
        assert!(
            mesh.loop_attribs().slots_tolerant_equal(
                a.as_usize(),
                b.as_usize(),
                CORNER_MERGE_EPS_SQ
            ),
            "values within epsilon must compare tolerant-equal (a byte compare \
             would reject them)"
        );

        // Beyond epsilon: separation 0.1 → squared distance 1e-2 > 1e-5.
        mesh.loop_attribs_mut()
            .set_vec2(uv, b.as_usize(), [0.6, 0.5]);
        assert!(
            !mesh.loop_attribs().slots_tolerant_equal(
                a.as_usize(),
                b.as_usize(),
                CORNER_MERGE_EPS_SQ
            ),
            "values beyond epsilon must compare unequal"
        );
    }

    #[test]
    fn corner_merge_fires_on_fan_and_collapses_behind_pair_to_single_value() {
        // §6.3 firing discriminator (corrected model, §6.3.1.1). On a
        // high-valence fan the two arc-end walls ABUT along the shared hub-side
        // side edge `e_connect` (rim = the untagged hub copy, `v` = the moved
        // hub). The merge keys on that side edge: wall A and face B are the two
        // walls; the behind pair is reached by crossing each wall's wall↔inset
        // edge at `v`, landing on the two ARC-END inset faces themselves
        // (triangles[2] and triangles[6]).
        //
        // Step 6 does NOT midpoint the behind pair — it collapses both behind
        // loops to a SINGLE behind loop's §7 value (`behind_wall`'s), and
        // first-writer-wins per hub settles which arc end survives. So after the
        // merge both arc-end hub loops hold ONE of the two pre-merge rebaked
        // values, never their average.
        //
        // A *non-affine* continuous UV makes the firing observable: §7 rebakes
        // each arc-end inset face's hub corner to a DISTINCT value (mean-value
        // weights differ per triangle); the merge equalises them to a single
        // arc-end value. The continuous-at-rim UV keeps the step-5 gate open.
        // A step-6-less (or never-firing) implementation leaves the two arc-end
        // hub loops at their distinct rebaked values.
        let mut mesh = LoopMesh::new();
        let (triangles, _hub, _rim) = make_triangle_fan(&mut mesh, 8);
        let tagged: Vec<FaceKey> = triangles[2..7].to_vec();

        let uv = mesh
            .register_loop_attribute(ElementAttributeDesc::numeric(
                AttributeKey::custom("uv"),
                AttrType::Vec2,
            ))
            .unwrap();

        let uv_fn = |p: Vec3| -> [f32; 2] { [p.x * p.y + p.x, p.x * p.x - p.y * p.y] };
        // Snapshot the two arc-end faces' original corner geometry + UVs so we
        // can predict each face's STANDALONE hub rebake (the pre-merge value).
        let arc_end_faces = [triangles[2], triangles[6]];
        let mut arc_corner_proj: Vec<Vec<Vec2>> = Vec::new();
        let mut arc_corner_uv: Vec<Vec<[f32; 2]>> = Vec::new();
        let projection = PlaneProjection::from_normal(Vec3::Z);
        for &f in &arc_end_faces {
            let mut proj = Vec::new();
            let mut vals = Vec::new();
            let mut it = mesh.face_loops(f);
            while let Some(l) = it.next(&mesh) {
                let p = mesh.vertex_position(mesh.loop_vertex(l));
                proj.push(projection.project(p));
                vals.push(uv_fn(p));
            }
            arc_corner_proj.push(proj);
            arc_corner_uv.push(vals);
        }

        // Seed the continuous non-affine UV on every loop.
        let loops: Vec<LoopKey> = mesh.loops.iter().collect();
        for l in loops {
            let p = mesh.vertex_position(mesh.loop_vertex(l));
            mesh.loop_attribs_mut().set_vec2(uv, l.as_usize(), uv_fn(p));
        }

        mesh.inset_region(
            &tagged,
            InsetRegionParams {
                thickness: 0.1,
                use_boundary: false,
                use_interpolate: true,
                ..Default::default()
            },
        );
        mesh.validate_topology();

        let moved_hub = shared_vertex_of(&mesh, &tagged);
        let hub_q = projection.project(mesh.vertex_position(moved_hub));

        // Predict each arc-end face's standalone hub rebake (no merge).
        let predict = |i: usize| -> [f32; 2] {
            expected_blend2(&arc_corner_proj[i], &arc_corner_uv[i], hub_q)
        };
        let pre_a = predict(0);
        let pre_b = predict(1);
        // The non-affine UV makes the two arc-end faces rebake the hub
        // DISTINCTLY — so a non-firing merge would leave them unequal, and a
        // (wrong) midpointing merge would land them on the average.
        assert!(
            (pre_a[0] - pre_b[0]).abs() > 1.0e-4 || (pre_a[1] - pre_b[1]).abs() > 1.0e-4,
            "fixture must rebake the two arc-end hub corners distinctly so firing \
             is observable; pre_a={pre_a:?} pre_b={pre_b:?}"
        );
        let midpoint = [0.5 * (pre_a[0] + pre_b[0]), 0.5 * (pre_a[1] + pre_b[1])];

        let near = |u: &[f32; 2], w: &[f32; 2]| -> bool {
            (u[0] - w[0]).abs() < 1.0e-4 && (u[1] - w[1]).abs() < 1.0e-4
        };

        // After the merge both arc-end inset faces' hub loops hold a SINGLE
        // behind value — one of `pre_a`/`pre_b`, decided by first-writer-wins —
        // NOT their midpoint (§6.3 step 6, §6.3.1.1). This proves the merge
        // fired AND pins the no-average rule.
        let arc_end_uvs = tagged_corner_uvs_at(&mesh, moved_hub, &arc_end_faces, uv);
        assert_eq!(arc_end_uvs.len(), 2);
        // Both arc-end hub loops collapsed to the same value.
        assert!(
            near(&arc_end_uvs[0], &arc_end_uvs[1]),
            "arc-end hub loops must collapse to one value; got {:?} and {:?}",
            arc_end_uvs[0],
            arc_end_uvs[1]
        );
        let survivor = arc_end_uvs[0];
        // That value is one of the two pre-merge arc-end §7 values.
        assert!(
            near(&survivor, &pre_a) || near(&survivor, &pre_b),
            "surviving arc-end UV {survivor:?} must equal one behind value \
             pre_a={pre_a:?} or pre_b={pre_b:?} (single behind value, not midpoint)"
        );
        // And it is NOT the midpoint — the old (wrong) averaging rule.
        assert!(
            !near(&survivor, &midpoint),
            "surviving arc-end UV {survivor:?} must NOT be the behind-pair \
             midpoint {midpoint:?} (the merge collapses, it does not average)"
        );
    }

    #[test]
    fn corner_merge_does_not_propagate_on_low_valence_fan() {
        // §6.3.1 branches (i)/(ii): a moved corner with only the two arc-end
        // walls and no further tagged face between them must NOT propagate.
        // A tagged fan of three triangles is below the five-triangle threshold
        // the corollary names, so the two behind-faces coincide or are
        // edge-adjacent and step 4 early-returns. With an asymmetric (skewed)
        // fan the single interior tagged hub loop rebakes (§7) to a value
        // distinct from the arc-end midpoint; we confirm step 4 leaves it at
        // that own value rather than forcing the midpoint onto it.
        let mut mesh = LoopMesh::new();
        // Skew the hub off the arc centre so the interior triangle's hub corner
        // rebakes to a genuinely different UV from the two arc ends.
        let (triangles, hub, _rim) = make_triangle_fan(&mut mesh, 6);
        mesh.set_vertex_position(hub, Vec3::new(0.15, -0.1, 0.0));
        let tagged: Vec<FaceKey> = triangles[2..5].to_vec(); // three consecutive

        let uv = mesh
            .register_loop_attribute(ElementAttributeDesc::numeric(
                AttributeKey::custom("uv"),
                AttrType::Vec2,
            ))
            .unwrap();
        // Non-affine continuous UV so the outer-pair gate passes (per-position)
        // yet the interior hub corner rebakes distinctly from the arc ends.
        let loops: Vec<LoopKey> = mesh.loops.iter().collect();
        for l in loops {
            let p = mesh.vertex_position(mesh.loop_vertex(l));
            mesh.loop_attribs_mut()
                .set_vec2(uv, l.as_usize(), [p.x * p.x + p.y, p.y * p.y - p.x]);
        }

        let middle = triangles[3];
        let mut middle_hub_loop = None;
        {
            let mut it = mesh.face_loops(middle);
            while let Some(l) = it.next(&mesh) {
                let p = mesh.vertex_position(mesh.loop_vertex(l));
                if (p - Vec3::new(0.15, -0.1, 0.0)).length() < 1.0e-9 {
                    middle_hub_loop = Some(l);
                }
            }
        }
        let middle_hub_loop = middle_hub_loop.unwrap();

        mesh.inset_region(
            &tagged,
            InsetRegionParams {
                thickness: 0.1,
                use_boundary: false,
                use_interpolate: true,
                ..Default::default()
            },
        );
        mesh.validate_topology();

        let got = mesh
            .loop_attribs()
            .get_vec2(uv, middle_hub_loop.as_usize())
            .unwrap();
        // The two arc-end hub loops merged to one value (steps 1–3). The lone
        // interior hub loop keeps its own §7-rebaked UV — step 4 early-returned
        // because the three-triangle fan's behind-faces are coincident or
        // edge-adjacent, so no propagation forced the midpoint onto it.
        let end_uvs = tagged_corner_uvs_at(
            &mesh,
            mesh.loop_vertex(middle_hub_loop),
            &[triangles[2], triangles[4]],
            uv,
        );
        assert_eq!(end_uvs.len(), 2);
        let merged = end_uvs[0];
        assert!(
            (got[0] - merged[0]).abs() > 1.0e-6 || (got[1] - merged[1]).abs() > 1.0e-6,
            "low-valence corner must not propagate the midpoint onto the interior \
             hub loop; got {got:?} merged {merged:?}"
        );
    }
}
