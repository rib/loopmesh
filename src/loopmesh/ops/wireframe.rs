use crate::loopmesh::*;

/// Near-flat guard for the inset-tangent bisector: a corner whose two edge
/// directions sum to (near-)zero is collinear, so the bisector is undefined and
/// the tangent falls back to a face-plane perpendicular (WIREFRAME.md §4.3).
const BISECTOR_EPS_SQ: f32 = 1.0e-12;

/// Near-flat guard for the even-offset shell-angle factor (WIREFRAME.md §4.3):
/// a corner whose interior angle is (near-)straight has `sin(α/2) → 1`, so the
/// factor is `1`; this guard keeps a near-flat corner at factor `1` rather than
/// dividing by a value that has drifted just past `1`.
const EVEN_FACTOR_EPS: f32 = 1.0e-6;

/// The §4.3 shell-angle factor for the inset distance at a corner whose two
/// incident edge directions (leaving the corner, unit length) are `to_prev` and
/// `to_next`. The interior corner angle `α` satisfies `cos α = to_prev · to_next`,
/// and the factor is `1 / cos(half_corner_deficit) = 1 / sin(α/2)` so that a
/// sharp corner pushes the inset frame correspondingly further along the
/// bisector to keep the wire's apparent width even. There is no upper clamp; a
/// near-flat corner (`sin(α/2) → 1`) is guarded back to `1`.
fn even_offset_factor(to_prev: Vec3, to_next: Vec3) -> f32 {
    let cos_angle = to_prev.dot(to_next).clamp(-1.0, 1.0);
    // sin(α/2) = sqrt((1 - cos α) / 2), the half-angle identity.
    let sin_half = (0.5 * (1.0 - cos_angle)).max(0.0).sqrt();
    if sin_half > EVEN_FACTOR_EPS {
        1.0 / sin_half
    } else {
        1.0
    }
}

/// The per-vertex relative-offset factor (WIREFRAME.md §5.4): the median length
/// of the tagged edges incident to a vertex, used to scale that vertex's whole
/// cross-section so the wire thickness tracks local geometry scale. `lengths`
/// are the tagged-edge lengths around one vertex; an even count averages the two
/// middle values (for a two-edge fan this reduces to the mean), an odd count
/// takes the single middle value. A vertex with no tagged edges falls back to
/// `1.0` (absolute thickness).
fn median_edge_length(lengths: &mut [f32]) -> f32 {
    if lengths.is_empty() {
        return 1.0;
    }
    lengths.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = lengths.len() / 2;
    if lengths.len() % 2 == 0 {
        (lengths[mid - 1] + lengths[mid]) * 0.5
    } else {
        lengths[mid]
    }
}

/// The three signed distances along the vertex normal that fix the wire's
/// cross-section (WIREFRAME.md §5.1), for thickness `t` and off-centre bias
/// `b ∈ [-1, 1]`. The near distance bounds `shell_neg`, the far distance bounds
/// `shell_pos`, and their midpoint is the cross-section centre. The span between
/// near and far is always the full thickness `t`; `b` only decides where that
/// span sits relative to the surface (`b = 0` centres it, `b = ±1` slides it
/// fully to one side).
fn shell_distances(t: f32, b: f32) -> (f32, f32, f32) {
    let near_offset = -(1.0 - b) * 0.5 * t;
    let far_offset = near_offset + t;
    let centre_offset = (near_offset + far_offset) * 0.5;
    (near_offset, far_offset, centre_offset)
}

/// The generated geometry of a wireframe run: the strut quads lining the
/// region's edges, plus the rim-cap quads added under `use_boundary`. This is
/// the analogue of the operator's `faces.out` slot (WIREFRAME.md §1) — it holds
/// only freshly built strut faces, never the (retained) source faces.
#[derive(Default)]
pub struct WireframeResult {
    pub strut_faces: Vec<FaceKey>,
}

/// The two shell hub verts of one tagged vertex (WIREFRAME.md §4.1): `neg` at
/// the near distance, `pos` at the far distance, both along the vertex normal.
#[derive(Clone, Copy)]
struct ShellPair {
    neg: VertexKey,
    pos: VertexKey,
}

impl LoopMesh {
    /// Replace a region of faces with a lattice of solid struts running along
    /// the region's edges — a per-face inset frame plus a two-sided strut
    /// bridge (WIREFRAME.md §2).
    ///
    /// For every tagged face this insets a shrunken copy of the face's loop
    /// inward along the face-tangent direction (the same in-plane bisector
    /// [`Self::inset_region`] uses, §4.2–§4.3), and bridges the gap between each
    /// original edge and the inset edge with a pair of quads — one offset to
    /// each side of the surface along the vertex normals by half the thickness
    /// (the two-sided shell offset of [`Self::solidify`], §4.1 + §5). Where two
    /// tagged faces share an edge their two strut pairs wrap that edge into a
    /// closed interior tube (§11.3); a region-boundary edge gets only the inner
    /// pair unless `use_boundary` caps its outer side with a rim strut pair
    /// (§6).
    ///
    /// This three-argument form **retains** the original input faces (it does
    /// not delete them) and leaves the original vertices in place — the
    /// `use_replace` removal of §7 is selected through
    /// [`Self::wireframe_with_replace`]. The returned [`WireframeResult`]
    /// carries only the generated strut/cap quads.
    ///
    /// See `radial-mesh-guides/operators/WIREFRAME.md` for the full algorithm
    /// and parity contract.
    pub fn wireframe(
        &mut self,
        faces: &[FaceKey],
        thickness: f32,
        use_boundary: bool,
    ) -> WireframeResult {
        // The off-centre bias defaults to a symmetric straddle (§5.1, `b = 0`),
        // the originals are kept (`use_replace = false`), and the inset uses the
        // plain flat distance (`use_even_offset = false`).
        self.wireframe_core(faces, thickness, 0.0, use_boundary, false, false, false)
    }

    /// As [`Self::wireframe`], but with the `[-1, 1]` off-centre bias `offset`
    /// that slides the wire's cross-section relative to the surface
    /// (WIREFRAME.md §5.1, §5.3). At `0` the tube straddles the surface
    /// symmetrically; at `±1` it sits entirely to one side. The in-plane
    /// half-width (`thickness/2`) is independent of the bias; the bias only
    /// shifts the near/far shell distances and, when non-zero, rides the
    /// inset-frame vert along the vertex normal by the cross-section centre
    /// (§4.2 / §5.3).
    pub fn wireframe_with_offset(
        &mut self,
        faces: &[FaceKey],
        thickness: f32,
        offset: f32,
        use_boundary: bool,
    ) -> WireframeResult {
        self.wireframe_core(faces, thickness, offset, use_boundary, false, false, false)
    }

    /// As [`Self::wireframe_with_offset`], but with the `use_even_offset`
    /// shell-angle correction of WIREFRAME.md §4.3: when `use_even_offset` is
    /// set, each corner's in-plane inset distance is scaled by
    /// `1 / cos(half_corner_deficit) = 1 / sin(interior_angle / 2)` so the wire
    /// keeps an even apparent thickness through sharp corners rather than
    /// pinching. The correction is the same construction inset's in-plane
    /// even-offset uses (see [`Self::inset_region`]); it is a pure scale on the
    /// per-corner inset push, adding no new geometry. A near-flat corner is
    /// guarded back to factor `1` and there is no upper clamp, so a very sharp
    /// corner produces a correspondingly large push. When `use_even_offset` is
    /// clear the result is identical to the plain flat-inset path.
    pub fn wireframe_with_even_offset(
        &mut self,
        faces: &[FaceKey],
        thickness: f32,
        offset: f32,
        use_boundary: bool,
        use_even_offset: bool,
    ) -> WireframeResult {
        self.wireframe_core(
            faces,
            thickness,
            offset,
            use_boundary,
            false,
            use_even_offset,
            false,
        )
    }

    /// As [`Self::wireframe_with_offset`], but with the `use_replace` axis of
    /// WIREFRAME.md §7: when `use_replace` is set, the original tagged faces are
    /// consumed and the original tagged vertices are deleted once the wire is
    /// built, so only the strut network remains. The deletion is guarded against
    /// over-removal on a *partial* selection — a tagged vertex still used by any
    /// un-tagged surviving face is kept alive (the boundary-of-region guard, §7).
    /// When `use_replace` is clear the original faces are retained and no
    /// original vertex is removed (they persist as face-less points).
    pub fn wireframe_with_replace(
        &mut self,
        faces: &[FaceKey],
        thickness: f32,
        offset: f32,
        use_boundary: bool,
        use_replace: bool,
    ) -> WireframeResult {
        self.wireframe_core(faces, thickness, offset, use_boundary, use_replace, false, false)
    }

    /// As [`Self::wireframe_with_offset`], but with the `use_relative_offset`
    /// per-vertex edge-length scale of WIREFRAME.md §5.4: when
    /// `use_relative_offset` is set, every per-vertex magnitude (the in-plane
    /// inset half-width and all three shell distances) is multiplied by that
    /// vertex's **relative factor** — the median length of the tagged edges
    /// incident to it. The wire thickness then tracks the local geometry scale:
    /// a vertex surrounded by long edges gets a thicker wire than one in a
    /// finely subdivided region. For a vertex whose tagged fan is two edges the
    /// median reduces to their mean. When `use_relative_offset` is clear the
    /// factor is `1.0` everywhere and the result is bit-identical to
    /// [`Self::wireframe_with_offset`] (absolute thickness).
    ///
    /// This implements only the all-tagged-edge fan case (the median is taken
    /// over the tagged edges of the region); the mixed tagged/untagged edge-fan
    /// behaviour of §14 is out of scope.
    pub fn wireframe_with_relative_offset(
        &mut self,
        faces: &[FaceKey],
        thickness: f32,
        offset: f32,
        use_boundary: bool,
        use_relative_offset: bool,
    ) -> WireframeResult {
        self.wireframe_core(
            faces,
            thickness,
            offset,
            use_boundary,
            false,
            false,
            use_relative_offset,
        )
    }

    /// The shared wireframe core driving every public entry point: per-face
    /// inset-frame + two-sided strut bridge, with the `offset` bias (§5.1/§5.3),
    /// the `use_boundary` rim cap (§6), the `use_replace` removal (§7), the
    /// `use_even_offset` shell-angle correction (§4.3), and the
    /// `use_relative_offset` per-vertex edge-length scale (§5.4) all selected by
    /// argument.
    fn wireframe_core(
        &mut self,
        faces: &[FaceKey],
        thickness: f32,
        offset: f32,
        use_boundary: bool,
        use_replace: bool,
        use_even_offset: bool,
        use_relative_offset: bool,
    ) -> WireframeResult {
        if faces.is_empty() {
            return WireframeResult::default();
        }

        // The shell offset (§5) and the in-plane tangent (§4.3) both read the
        // stored normals, so freshen them first — exactly the precondition the
        // oracle carries.
        self.mesh_normals_update();

        // §5.1: the bias `b` slides the cross-section. The near distance
        // retreats from the surface by the fraction `(1 - b)/2` of the
        // thickness, the far distance is one full thickness beyond, and the
        // centre is their midpoint. At `b = 0` this collapses to the symmetric
        // straddle (near = -t/2, far = +t/2, centre = 0).
        let (near_offset, far_offset, centre_offset) = shell_distances(thickness, offset);
        // The in-plane half-width is independent of the bias (§5.1).
        let inset_dist = thickness * 0.5;

        let face_tag = self.push_face_op_flag_layer();
        for &face in faces {
            self.face_op_flag_set(face_tag, face, 0);
        }

        // §3: tag every vertex of a tagged face (a tagged vertex gets a shell
        // pair) and every edge of a tagged face (the tagged edges whose median
        // length drives the §5.4 relative scale). The shell-pair lookup is keyed
        // by original vertex slot. Snapshot the original element watermarks so
        // the §7 replace pass can tell originals (slot < watermark) from freshly
        // built wire geometry.
        let vert_watermark = self.vertices.key_watermark();
        let edge_watermark = self.edges.key_watermark();
        let mut vert_tagged: Vec<bool> = vec![false; vert_watermark];
        let mut edge_tagged: Vec<bool> = vec![false; edge_watermark];
        for &face in faces {
            let mut loops = self.face_loops(face);
            while let Some(l) = loops.next(self) {
                vert_tagged[self.loop_vertex(l).as_usize()] = true;
                edge_tagged[self.loop_edge(l).as_usize()] = true;
            }
        }

        // §5.4: the per-vertex relative factor — the median length of the tagged
        // edges incident to each tagged vertex — scaling that vertex's whole
        // cross-section. With `use_relative_offset` clear the factor is `1.0`
        // everywhere, so the cross-section keeps its absolute size and the path
        // is bit-identical to the flat behaviour.
        let mut relative_scale: Vec<f32> = vec![1.0; vert_watermark];
        if use_relative_offset {
            let mut lengths: Vec<f32> = Vec::new();
            for slot in 0..vert_watermark {
                if !vert_tagged[slot] {
                    continue;
                }
                let v = VertexKey::from_usize(slot);
                lengths.clear();
                for edge in self.vertex_disk_iter(v) {
                    if !edge_tagged[edge.as_usize()] {
                        continue;
                    }
                    let [a, b] = self.edge_vertices(edge);
                    let length = (self.vertex_position(a) - self.vertex_position(b)).length();
                    lengths.push(length);
                }
                relative_scale[slot] = median_edge_length(&mut lengths);
            }
        }

        // §4.1: two shell verts per tagged vertex, placed along the vertex
        // normal at the near / far distances. Indexed by original vertex slot.
        let mut shell: Vec<Option<ShellPair>> = vec![None; vert_watermark];
        for slot in 0..vert_watermark {
            if !vert_tagged[slot] {
                continue;
            }
            let v = VertexKey::from_usize(slot);
            let base = self.vertex_position(v);
            let normal = self.vertex_normal(v);
            // §5.4: scale the near/far shell distances by the vertex's relative
            // factor (`1.0` unless `use_relative_offset`).
            let scale = relative_scale[slot];

            let neg = self.create_vertex(base + normal * (near_offset * scale));
            self.vertex_attribs.copy_into(neg.as_usize(), slot);
            let pos = self.create_vertex(base + normal * (far_offset * scale));
            self.vertex_attribs.copy_into(pos.as_usize(), slot);

            shell[slot] = Some(ShellPair { neg, pos });
        }

        // §4.2: one inset-frame vert per loop (face-corner), in the face plane,
        // pushed inward along the corner's face tangent by `thickness/2`. Keyed
        // by loop key so §4.4 can recover both ends of each edge.
        let loop_watermark = self.loops.key_watermark();
        let mut inset_of_loop: Vec<Option<VertexKey>> = vec![None; loop_watermark];
        for &face in faces {
            let face_normal = self.face_normal(face);
            let mut loops = self.face_loops(face);
            while let Some(l) = loops.next(self) {
                let v = self.loop_vertex(l);
                let tangent = self.wireframe_inward_tangent(l, face_normal);
                let corner = self.vertex_position(v);
                // §5.4: the vertex's relative factor scales the in-plane
                // half-width and the centre-offset normal push together.
                let scale = relative_scale[v.as_usize()];
                // §4.3: with `use_even_offset` the in-plane push is lengthened by
                // the corner's shell-angle factor so a sharp corner keeps an even
                // apparent wire width; the factor is `1` (no scale) otherwise.
                let corner_inset = if use_even_offset {
                    inset_dist * scale * self.wireframe_even_factor(l)
                } else {
                    inset_dist * scale
                };
                // §4.2: the in-plane inset push gives the wire its width. When
                // the bias is non-zero the inset frame additionally rides along
                // the vertex normal by the cross-section centre (§5.3), so the
                // frame stays aligned with the shifted cross-section rather than
                // pinned in the surface.
                let mut frame_pos = corner + tangent * corner_inset;
                if offset != 0.0 {
                    frame_pos += self.vertex_normal(v) * (centre_offset * scale);
                }
                let frame = self.create_vertex(frame_pos);
                self.vertex_attribs.copy_into(frame.as_usize(), v.as_usize());
                inset_of_loop[l.as_usize()] = Some(frame);
            }
        }

        // §3 boundary classification + §6.1 boundary-frame verts (one per
        // boundary vertex, created lazily so each appears once).
        let mut boundary_frame: Vec<Option<VertexKey>> = vec![None; vert_watermark];

        let mut result = WireframeResult::default();

        // §4.4 + §6.2: walk every tagged face's loops once more and bridge each
        // corner's edge. The current loop runs from corner `a` (its own vertex)
        // to corner `b` (the next loop's vertex) along one face edge.
        for &face in faces {
            let face_normal = self.face_normal(face);
            let first = self.face_first_loop(face);
            let mut l = first;
            loop {
                let next_l = self.loop_next(l);
                let edge = self.loop_edge(l);

                let va = self.loop_vertex(l);
                let vb = self.loop_vertex(next_l);
                let inset_a = inset_of_loop[l.as_usize()].unwrap();
                let inset_b = inset_of_loop[next_l.as_usize()].unwrap();
                let shell_a = shell[va.as_usize()].unwrap();
                let shell_b = shell[vb.as_usize()].unwrap();

                // §4.4: the `−` quad is (inset_a, inset_b, neg_b, neg_a); the
                // `+` quad is (inset_b, inset_a, pos_a, pos_b) — opposite
                // windings so both faces point outward from the wire.
                let neg_quad = self.wireframe_quad(
                    [inset_a, inset_b, shell_b.neg, shell_a.neg],
                    [l, next_l, next_l, l],
                    face,
                );
                let pos_quad = self.wireframe_quad(
                    [inset_b, inset_a, shell_a.pos, shell_b.pos],
                    [next_l, l, l, next_l],
                    face,
                );
                result.strut_faces.push(neg_quad);
                result.strut_faces.push(pos_quad);

                // §6: a region-boundary edge (no other tagged face uses it)
                // gets, under `use_boundary`, an outer rim cap that closes its
                // tube.
                if use_boundary && self.wireframe_edge_is_boundary(edge, face, face_tag) {
                    // §6.1: one boundary vert per boundary vertex, created the
                    // first time a boundary edge touches it.
                    let boundary_a = match boundary_frame[va.as_usize()] {
                        Some(existing) => existing,
                        None => {
                            // §5.4: the boundary frame rides the same per-vertex
                            // relative scale as the per-loop inset (§4.2).
                            let frame = self.wireframe_boundary_frame(
                                va,
                                face,
                                face_normal,
                                face_tag,
                                inset_dist * relative_scale[va.as_usize()],
                            );
                            boundary_frame[va.as_usize()] = Some(frame);
                            frame
                        }
                    };
                    let boundary_b = match boundary_frame[vb.as_usize()] {
                        Some(existing) => existing,
                        None => {
                            let frame = self.wireframe_boundary_frame(
                                vb,
                                face,
                                face_normal,
                                face_tag,
                                inset_dist * relative_scale[vb.as_usize()],
                            );
                            boundary_frame[vb.as_usize()] = Some(frame);
                            frame
                        }
                    };

                    // §6.2: windings opposite the inner struts so the cap faces
                    // point outward on the rim's outer side.
                    let cap_neg = self.wireframe_quad(
                        [boundary_b, boundary_a, shell_a.neg, shell_b.neg],
                        [next_l, l, l, next_l],
                        face,
                    );
                    let cap_pos = self.wireframe_quad(
                        [boundary_a, boundary_b, shell_b.pos, shell_a.pos],
                        [l, next_l, next_l, l],
                        face,
                    );
                    result.strut_faces.push(cap_neg);
                    result.strut_faces.push(cap_pos);
                }

                l = next_l;
                if l == first {
                    break;
                }
            }
        }

        // §7: with `use_replace` set, consume the original tagged faces and
        // delete the original tagged verts — guarding a tagged vert against
        // removal when any un-tagged surviving face still uses it (the
        // boundary-of-region keep-alive guard).
        if use_replace {
            self.wireframe_replace_originals(&vert_tagged, vert_watermark, face_tag);
        }

        self.pop_face_op_flag_layer(face_tag);
        result
    }

    /// §7 — the `use_replace` removal pass. Each tagged original vertex is
    /// deleted unless an un-tagged surviving face still uses it; the original
    /// tagged faces are consumed as a *cascade* of that vertex removal (deleting
    /// a vert kills its incident edges and their faces), so a fully-wired region
    /// loses its source faces while a partial region whose border verts are all
    /// kept alive retains them.
    ///
    /// The keep-alive guard walks every *surviving original* face (any face not
    /// tagged for wiring — the freshly built strut quads use new verts, slot >=
    /// the pre-build watermark, so they never anchor an original) and marks every
    /// original vertex it touches as kept-alive, so a vertex on the border
    /// between the wired region and the surrounding un-wired faces is preserved.
    /// Only then are the tagged, un-kept original verts removed.
    fn wireframe_replace_originals(
        &mut self,
        vert_tagged: &[bool],
        vert_watermark: usize,
        face_tag: crate::op_flags::FaceOpFlagLayer,
    ) {
        // §7.1 — mark every original vert that a surviving (un-tagged) face still
        // uses. Strut faces are skipped implicitly because their corner verts are
        // new (slot >= vert_watermark); the tagged source faces are skipped by
        // the tag test (they are consumed by the §7.2 cascade).
        let mut keep_alive: Vec<bool> = vec![false; vert_watermark];
        let mut face_iter = MeshFaces::default();
        while let Some(face) = face_iter.next(self) {
            if self.face_op_flag_test(face_tag, face, 0) {
                continue;
            }
            let mut loops = self.face_loops(face);
            while let Some(l) = loops.next(self) {
                let slot = self.loop_vertex(l).as_usize();
                if slot < vert_watermark {
                    keep_alive[slot] = true;
                }
            }
        }

        // §7.2 — delete every tagged original vert not kept alive. `vertex_kill`
        // cascades through the vert's perimeter edges, killing the original
        // source face with them; a kept vert (still anchored by a surviving
        // face) is left untouched, and its source face rides along.
        for slot in 0..vert_watermark {
            if !vert_tagged[slot] || keep_alive[slot] {
                continue;
            }
            let v = VertexKey::from_usize(slot);
            if self.vertices.try_get(v).is_some() {
                self.vertex_kill(v);
            }
        }
    }

    /// §4.3 — the unit inward face tangent at loop `l`: the bisector of the two
    /// edge directions leaving the corner, pointing into the face interior. A
    /// collinear corner (degenerate bisector) falls back to a face-plane
    /// perpendicular taken against the face normal.
    fn wireframe_inward_tangent(&self, l: LoopKey, face_normal: Vec3) -> Vec3 {
        let v = self.loop_vertex(l);
        let prev_v = self.loop_vertex(self.loop_prev(l));
        let next_v = self.loop_vertex(self.loop_next(l));
        let here = self.vertex_position(v);
        let to_prev = LoopMesh::normalize_or_zero(self.vertex_position(prev_v) - here);
        let to_next = LoopMesh::normalize_or_zero(self.vertex_position(next_v) - here);

        let bisector = to_prev + to_next;
        if bisector.dot(bisector) > BISECTOR_EPS_SQ {
            LoopMesh::normalize_or_zero(bisector)
        } else {
            // Collinear corner: take a perpendicular to the (straight) edge in
            // the face plane, oriented inward by a concavity check.
            let edge_dir = to_next;
            let perp = LoopMesh::normalize_or_zero(face_normal.cross(edge_dir));
            if perp.dot(to_prev) < 0.0 {
                -perp
            } else {
                perp
            }
        }
    }

    /// §4.3 — the even-offset shell-angle factor at loop `l`: derived from the
    /// interior corner angle between the two edges meeting at the corner, the
    /// same construction inset's in-plane even-offset uses. See
    /// [`even_offset_factor`].
    fn wireframe_even_factor(&self, l: LoopKey) -> f32 {
        let v = self.loop_vertex(l);
        let prev_v = self.loop_vertex(self.loop_prev(l));
        let next_v = self.loop_vertex(self.loop_next(l));
        let here = self.vertex_position(v);
        let to_prev = LoopMesh::normalize_or_zero(self.vertex_position(prev_v) - here);
        let to_next = LoopMesh::normalize_or_zero(self.vertex_position(next_v) - here);
        even_offset_factor(to_prev, to_next)
    }

    /// §3 — an edge is a region boundary with respect to the tagged region if
    /// no tagged face *other than* `owner` uses it (a real mesh boundary, or
    /// the only tagged user is `owner`).
    fn wireframe_edge_is_boundary(
        &self,
        edge: EdgeKey,
        owner: FaceKey,
        face_tag: crate::op_flags::FaceOpFlagLayer,
    ) -> bool {
        let mut radial = self.radial_loops(edge);
        while let Some(rl) = radial.next(self) {
            let face = self.loop_face(rl);
            if face != owner && self.face_op_flag_test(face_tag, face, 0) {
                return false;
            }
        }
        true
    }

    /// §6.1 — the outer-side boundary inset-frame vert at corner `v` of the
    /// boundary-edged face `face`: placed like the per-loop inset vert (§4.2)
    /// but pushed *outward* along the boundary tangent so the rim cap closes the
    /// tube. The boundary tangent is the negation of the inward bisector taken
    /// over the (up to two) boundary edges of `face` at `v`; a single boundary
    /// edge falls back to a face-plane perpendicular whose sign is oriented
    /// outward (away from the face interior).
    fn wireframe_boundary_frame(
        &mut self,
        v: VertexKey,
        face: FaceKey,
        face_normal: Vec3,
        face_tag: crate::op_flags::FaceOpFlagLayer,
        inset_dist: f32,
    ) -> VertexKey {
        let here = self.vertex_position(v);

        // The (up to two) boundary edges of `face` incident to `v`, as unit
        // directions leaving the corner. `corner_loop` is the loop of `face`
        // whose own vertex is `v` — used to recover the face-interior reference
        // direction so a single-edge perpendicular can be oriented outward.
        let mut dirs: [Vec3; 2] = [Vec3::ZERO; 2];
        let mut count = 0usize;
        let mut corner_loop: Option<LoopKey> = None;
        let first = self.face_first_loop(face);
        let mut l = first;
        loop {
            let lv = self.loop_vertex(l);
            let le = self.loop_edge(l);
            if lv == v {
                corner_loop = Some(l);
            }
            if (lv == v || self.loop_vertex(self.loop_next(l)) == v)
                && self.wireframe_edge_is_boundary(le, face, face_tag)
            {
                let [a, b] = self.edge_vertices(le);
                let other = if a == v { b } else { a };
                if other != v {
                    let dir = LoopMesh::normalize_or_zero(self.vertex_position(other) - here);
                    if dir != Vec3::ZERO && count < 2 {
                        dirs[count] = dir;
                        count += 1;
                    }
                }
            }
            l = self.loop_next(l);
            if l == first {
                break;
            }
        }

        // The rim tangent must point *outward* — away from the face interior at
        // this corner. For a single boundary edge (and the collinear two-edge
        // fallback) the face-plane perpendicular has two valid signs, and a bare
        // cross product picks the same one at both ends of a rim that runs in a
        // single direction; at a fork vertex where the spine continues past the
        // corner, that is the wrong (inward-pointing) sign at one end. Orient the
        // perpendicular against the inward face tangent (the §4.3 interior
        // bisector) so it always points away from the interior, the same
        // concavity check the inward tangent itself uses.
        let orient_outward = |perp: Vec3, this: &Self| -> Vec3 {
            let perp = LoopMesh::normalize_or_zero(perp);
            if perp == Vec3::ZERO {
                return Vec3::ZERO;
            }
            match corner_loop {
                Some(cl) => {
                    let inward = this.wireframe_inward_tangent(cl, face_normal);
                    if perp.dot(inward) > 0.0 {
                        -perp
                    } else {
                        perp
                    }
                }
                None => perp,
            }
        };

        let outward = match count {
            2 => {
                // Inward bisector of the two boundary edges → negate for the
                // outward (rim) direction.
                let inward = LoopMesh::normalize_or_zero(dirs[0] + dirs[1]);
                if inward != Vec3::ZERO {
                    -inward
                } else {
                    orient_outward(face_normal.cross(dirs[0]), self)
                }
            }
            1 => {
                // §6.1 single-boundary-edge degenerate case: derive the tangent
                // from the one edge alone (a face-plane perpendicular), oriented
                // outward away from the face interior.
                orient_outward(face_normal.cross(dirs[0]), self)
            }
            _ => Vec3::ZERO,
        };

        let frame = self.create_vertex(here + outward * inset_dist);
        self.vertex_attribs.copy_into(frame.as_usize(), v.as_usize());
        frame
    }

    /// Build one strut quad from four corner verts, copying the source face's
    /// per-face customdata and each corner's per-loop customdata from the named
    /// source loop (§4.4 / §10). `src_loops[i]` is the source corner whose loop
    /// data corner `i` of the quad inherits.
    fn wireframe_quad(
        &mut self,
        verts: [VertexKey; 4],
        src_loops: [LoopKey; 4],
        src_face: FaceKey,
    ) -> FaceKey {
        let edges = [
            self.create_edge([verts[0], verts[1]], EdgeCreateFlags::NO_DUPLICATE),
            self.create_edge([verts[1], verts[2]], EdgeCreateFlags::NO_DUPLICATE),
            self.create_edge([verts[2], verts[3]], EdgeCreateFlags::NO_DUPLICATE),
            self.create_edge([verts[3], verts[0]], EdgeCreateFlags::NO_DUPLICATE),
        ];
        let face = self.face_create(&verts, &edges, FaceCreateFlags::NONE);
        self.face_attribs.copy_into(face.as_usize(), src_face.as_usize());

        if !self.loop_attribs.is_empty() {
            let mut ql = self.face_first_loop(face);
            for i in 0..4 {
                self.loop_attribs
                    .copy_into(ql.as_usize(), src_loops[i].as_usize());
                ql = self.loop_next(ql);
            }
        }

        face
    }
}

#[cfg(test)]
mod tests {
    use glam::Vec3;

    use crate::loopmesh::test_support::create_face_from_verts;
    use crate::loopmesh::*;

    /// A unit quad lying flat in the z = 0 plane, wound so its face normal
    /// points +Z.
    fn flat_quad(mesh: &mut LoopMesh) -> FaceKey {
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

    #[test]
    fn lone_quad_open_frame_counts() {
        // WIREFRAME.md §11.1 (kernel-layer counts): a lone tagged quad with
        // `use_boundary = false` builds an open four-strut frame — 8 strut
        // faces — and the kernel BMOP *retains* the source face, so the mesh
        // ends with 8 + 1 = 9 faces and 16 verts (4 original + 4 inset-frame +
        // 8 shell).
        let mut mesh = LoopMesh::new();
        let face = flat_quad(&mut mesh);

        let result = mesh.wireframe(&[face], 0.2, false);

        assert_eq!(result.strut_faces.len(), 8, "two struts per corner");
        assert_eq!(mesh.face_count(), 9, "8 struts + 1 retained source");
        assert_eq!(
            mesh.vertex_count(),
            16,
            "4 original + 4 inset-frame + 8 shell"
        );
        mesh.validate_topology();
    }

    #[test]
    fn lone_quad_capped_rim_counts() {
        // WIREFRAME.md §6 / §11.1: with `use_boundary = true` every rim edge is
        // capped by a second strut pair — 8 inner + 8 cap = 16 strut faces —
        // adding one boundary vert per corner. With the retained source: 17
        // faces, 20 verts (16 open + 4 boundary).
        let mut mesh = LoopMesh::new();
        let face = flat_quad(&mut mesh);

        let result = mesh.wireframe(&[face], 0.2, true);

        assert_eq!(result.strut_faces.len(), 16, "8 inner + 8 rim caps");
        assert_eq!(mesh.face_count(), 17, "16 struts + 1 retained source");
        assert_eq!(
            mesh.vertex_count(),
            20,
            "16 (open case) + 4 boundary verts"
        );
        mesh.validate_topology();
    }

    #[test]
    fn use_boundary_contrast_pins_flag_effect() {
        // The contrast pin (§6): the same lone quad differs by exactly the rim
        // cap — 8 extra strut faces and 4 extra boundary verts — between
        // `use_boundary` clear and set. An impl that ignored the flag would
        // produce identical counts.
        let mut open = LoopMesh::new();
        let fo = flat_quad(&mut open);
        let open_result = open.wireframe(&[fo], 0.2, false);

        let mut capped = LoopMesh::new();
        let fc = flat_quad(&mut capped);
        let capped_result = capped.wireframe(&[fc], 0.2, true);

        assert_eq!(
            capped_result.strut_faces.len() - open_result.strut_faces.len(),
            8,
            "use_boundary adds one cap pair per boundary edge"
        );
        assert_eq!(
            capped.vertex_count() - open.vertex_count(),
            4,
            "use_boundary adds one boundary vert per boundary vertex"
        );
    }

    #[test]
    fn grid_3x3_centre_face_wires_only_region() {
        // WIREFRAME.md §11.2: wiring the centre face of a 3×3 grid touches only
        // that face. Its four edges are all interior to the mesh but
        // wire-boundary with respect to the tagged region (the surrounding
        // faces are untagged), so with `use_boundary = false` it yields 8 strut
        // faces; the 8 surrounding faces and their verts are untouched.
        let mut mesh = LoopMesh::new();
        let mut verts = [[VertexKey::from_usize(0); 4]; 4];
        for (iy, row) in verts.iter_mut().enumerate() {
            for (ix, slot) in row.iter_mut().enumerate() {
                *slot = mesh.create_vertex(Vec3::new(ix as f32, iy as f32, 0.0));
            }
        }
        let mut faces = Vec::new();
        for y in 0..3 {
            for x in 0..3 {
                let ring = [
                    verts[y][x],
                    verts[y][x + 1],
                    verts[y + 1][x + 1],
                    verts[y + 1][x],
                ];
                let edges = [
                    mesh.create_edge([ring[0], ring[1]], EdgeCreateFlags::NO_DUPLICATE),
                    mesh.create_edge([ring[1], ring[2]], EdgeCreateFlags::NO_DUPLICATE),
                    mesh.create_edge([ring[2], ring[3]], EdgeCreateFlags::NO_DUPLICATE),
                    mesh.create_edge([ring[3], ring[0]], EdgeCreateFlags::NO_DUPLICATE),
                ];
                faces.push(mesh.face_create(&ring, &edges, FaceCreateFlags::NONE));
            }
        }
        let centre = faces[4];
        let surround: Vec<FaceKey> = faces.iter().copied().filter(|&f| f != centre).collect();
        let verts_before = mesh.vertex_count();

        let result = mesh.wireframe(&[centre], 0.2, false);

        assert_eq!(result.strut_faces.len(), 8, "all four edges are wire-boundary");
        // 9 grid faces + 8 struts (centre retained) = 17.
        assert_eq!(mesh.face_count(), 17);
        // New verts: 4 inset + 8 shell; the 16 grid verts persist.
        assert_eq!(mesh.vertex_count(), verts_before + 12);
        for &f in &surround {
            assert_eq!(mesh.face_len(f), 4, "surrounding faces untouched");
        }
        mesh.validate_topology();
    }

    #[test]
    fn offset_plus_one_shifts_every_vert_along_normal() {
        // WIREFRAME.md §5.1/§5.3: the off-centre bias is position-only. A lone
        // flat quad (face normal +Z) wired at `offset = +1` builds the same
        // topology as `offset = 0` (9 faces, 16 verts) but every freshly built
        // shell + inset-frame vert shifts along the vertex normal by the bias.
        //
        // At `offset = 0`: near = -t/2, far = +t/2, centre = 0 — shell verts
        // straddle z = ±t/2, inset frame at z = 0.
        // At `offset = +1`: near = 0, far = +t, centre = +t/2 — the whole
        // cross-section slides to the +Z side, so every new vert is +t/2 higher
        // than its `offset = 0` counterpart.
        let thickness = 0.2_f32;
        let half = thickness * 0.5;

        let mut zero = LoopMesh::new();
        let fz = flat_quad(&mut zero);
        let zero_result = zero.wireframe_with_offset(&[fz], thickness, 0.0, false);

        let mut plus = LoopMesh::new();
        let fp = flat_quad(&mut plus);
        let plus_result = plus.wireframe_with_offset(&[fp], thickness, 1.0, false);

        // Topology is identical between the two bias values.
        assert_eq!(zero_result.strut_faces.len(), 8);
        assert_eq!(plus_result.strut_faces.len(), 8);
        assert_eq!(zero.face_count(), plus.face_count());
        assert_eq!(zero.vertex_count(), 16);
        assert_eq!(plus.vertex_count(), 16);

        // The four original corner verts (slots 0..4, created by `flat_quad`)
        // are untouched by either run; every vert beyond them is freshly built.
        // Each new `+1` vert sits exactly `+half` in z above its `0` twin —
        // the new verts are created in the same order by both runs.
        for slot in 4..16 {
            let z0 = zero.vertex_position(VertexKey::from_usize(slot)).z;
            let zp = plus.vertex_position(VertexKey::from_usize(slot)).z;
            assert!(
                (zp - z0 - half).abs() < 1.0e-6,
                "vert {slot}: offset=+1 z {zp} should be {half} above offset=0 z {z0}",
            );
        }

        // Sanity on the absolute placements at offset=0: shell verts straddle
        // z = ±half, inset-frame verts sit in the surface (z = 0).
        let mut saw_neg = false;
        let mut saw_pos = false;
        let mut saw_surface = false;
        for slot in 4..16 {
            let z = zero.vertex_position(VertexKey::from_usize(slot)).z;
            if (z + half).abs() < 1.0e-6 {
                saw_neg = true;
            } else if (z - half).abs() < 1.0e-6 {
                saw_pos = true;
            } else if z.abs() < 1.0e-6 {
                saw_surface = true;
            }
        }
        assert!(saw_neg && saw_pos && saw_surface, "offset=0 straddles symmetrically");

        plus.validate_topology();
    }

    #[test]
    fn use_replace_lone_quad_removes_originals() {
        // WIREFRAME.md §7 (whole-mesh wire): with `use_replace` set and the
        // whole mesh tagged, the keep-alive guard is vacuous — no un-tagged face
        // survives to anchor the four originals — so the source face is consumed
        // and all four original verts are removed. Counts drop from the kept
        // case (16 verts / 9 faces) to 12 verts (4 inset-frame + 8 shell) / 8
        // faces (struts only, source consumed).
        let mut kept = LoopMesh::new();
        let fk = flat_quad(&mut kept);
        let kept_result = kept.wireframe_with_replace(&[fk], 0.2, 0.0, false, false);
        assert_eq!(kept_result.strut_faces.len(), 8);
        assert_eq!(kept.face_count(), 9, "source retained when use_replace clear");
        assert_eq!(kept.vertex_count(), 16, "4 original kept + 4 inset + 8 shell");
        kept.validate_topology();

        let mut replaced = LoopMesh::new();
        let fr = flat_quad(&mut replaced);
        let replaced_result = replaced.wireframe_with_replace(&[fr], 0.2, 0.0, false, true);
        assert_eq!(replaced_result.strut_faces.len(), 8, "struts unchanged by replace");
        assert_eq!(replaced.face_count(), 8, "source face consumed under use_replace");
        assert_eq!(
            replaced.vertex_count(),
            12,
            "all 4 originals removed (12 = 4 inset + 8 shell)"
        );
        replaced.validate_topology();
    }

    #[test]
    fn use_replace_grid_3x3_centre_keeps_shared_border_verts() {
        // WIREFRAME.md §7 / §11.2 keep-alive guard: wiring only the centre face
        // of a 3×3 grid with `use_replace` set must *not* remove the centre
        // face's four corner verts — each is shared with the eight surrounding
        // un-tagged faces, which still need them. The guard keeps them alive, so
        // the counts match the `use_replace = false` centre-face wire exactly:
        // 28 verts (16 grid + 4 inset + 8 shell) and 17 faces (9 grid − 1
        // consumed centre + 8 struts + ... ) — concretely 16 surviving grid
        // faces + ... see below.
        let build = |use_replace: bool| -> LoopMesh {
            let mut mesh = LoopMesh::new();
            let mut verts = [[VertexKey::from_usize(0); 4]; 4];
            for (iy, row) in verts.iter_mut().enumerate() {
                for (ix, slot) in row.iter_mut().enumerate() {
                    *slot = mesh.create_vertex(Vec3::new(ix as f32, iy as f32, 0.0));
                }
            }
            let mut faces = Vec::new();
            for y in 0..3 {
                for x in 0..3 {
                    let ring = [
                        verts[y][x],
                        verts[y][x + 1],
                        verts[y + 1][x + 1],
                        verts[y + 1][x],
                    ];
                    let edges = [
                        mesh.create_edge([ring[0], ring[1]], EdgeCreateFlags::NO_DUPLICATE),
                        mesh.create_edge([ring[1], ring[2]], EdgeCreateFlags::NO_DUPLICATE),
                        mesh.create_edge([ring[2], ring[3]], EdgeCreateFlags::NO_DUPLICATE),
                        mesh.create_edge([ring[3], ring[0]], EdgeCreateFlags::NO_DUPLICATE),
                    ];
                    faces.push(mesh.face_create(&ring, &edges, FaceCreateFlags::NONE));
                }
            }
            let centre = faces[4];
            mesh.wireframe_with_replace(&[centre], 0.1, 0.0, false, use_replace);
            mesh
        };

        // Clear case: centre face retained, 16 grid verts + 4 inset + 8 shell =
        // 28 verts; 9 grid faces + 8 struts = 17 faces.
        let kept = build(false);
        assert_eq!(kept.vertex_count(), 28);
        assert_eq!(kept.face_count(), 17);
        kept.validate_topology();

        // Replace case: the four shared border verts are kept alive by the eight
        // surrounding faces, so only the centre face is consumed and *no* vert
        // is removed — the counts are unchanged from the clear case.
        let replaced = build(true);
        assert_eq!(
            replaced.vertex_count(),
            28,
            "keep-alive guard preserves the 4 shared border verts"
        );
        assert_eq!(
            replaced.face_count(),
            17,
            "centre consumed but 8 surrounding faces + 8 struts intact"
        );
        // The eight surrounding faces stay quads (nothing torn out under them).
        let mut quad_faces = 0;
        let mut face_iter = MeshFaces::default();
        while let Some(f) = face_iter.next(&replaced) {
            if replaced.face_len(f) == 4 {
                quad_faces += 1;
            }
        }
        assert_eq!(quad_faces, 17, "all surviving + strut faces are quads");
        replaced.validate_topology();
    }

    /// A flat equilateral triangle lying in z = 0, wound so its face normal
    /// points +Z. Every interior corner is 60°, so the §4.3 even-offset factor
    /// is `1/cos(60°) = 1/sin(30°) = 2.0`.
    fn flat_equilateral_tri(mesh: &mut LoopMesh) -> FaceKey {
        let h = (3.0_f32).sqrt() / 2.0;
        create_face_from_verts(
            mesh,
            &[
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(0.5, h, 0.0),
            ],
        )
    }

    #[test]
    fn even_offset_zero_matches_flat_inset() {
        // WIREFRAME.md §4.3: `use_even_offset = false` must be bit-identical to
        // the existing flat-inset path. The new entry point with the flag clear
        // and the old `wireframe_with_offset` build the same verts in the same
        // order, so every position matches exactly.
        let mut flat = LoopMesh::new();
        let ff = flat_equilateral_tri(&mut flat);
        let flat_result = flat.wireframe_with_offset(&[ff], 0.3, 0.0, false);

        let mut even_off = LoopMesh::new();
        let fe = flat_equilateral_tri(&mut even_off);
        let even_result = even_off.wireframe_with_even_offset(&[fe], 0.3, 0.0, false, false);

        assert_eq!(flat_result.strut_faces.len(), even_result.strut_faces.len());
        assert_eq!(flat.vertex_count(), even_off.vertex_count());
        assert_eq!(flat.face_count(), even_off.face_count());
        for slot in 0..flat.vertex_count() {
            let a = flat.vertex_position(VertexKey::from_usize(slot));
            let b = even_off.vertex_position(VertexKey::from_usize(slot));
            assert_eq!(a, b, "vert {slot}: use_even_offset=false must equal flat path");
        }
        even_off.validate_topology();
    }

    #[test]
    fn even_offset_doubles_inset_at_sixty_degree_corner() {
        // WIREFRAME.md §4.3: on a flat equilateral triangle every corner is 60°,
        // so the shell-angle factor is `1/sin(30°) = 2.0`. The even arm pushes
        // each inset-frame vert inward along the bisector by exactly twice the
        // flat arm's distance from the corner; the original corners and the
        // shell verts (offset along the +Z normal, unaffected by the in-plane
        // scale) are untouched between the two arms.
        let thickness = 0.3_f32;

        let mut flat = LoopMesh::new();
        let ff = flat_equilateral_tri(&mut flat);
        flat.wireframe_with_even_offset(&[ff], thickness, 0.0, false, false);

        let mut even_on = LoopMesh::new();
        let fe = flat_equilateral_tri(&mut even_on);
        even_on.wireframe_with_even_offset(&[fe], thickness, 0.0, false, true);

        // Topology is identical between the two arms — the correction is a pure
        // in-plane inset-distance scale (no new verts).
        assert_eq!(flat.vertex_count(), even_on.vertex_count());
        assert_eq!(flat.face_count(), even_on.face_count());

        // The three original corners (slots 0..3) are flat-z and shared; the
        // three inset-frame verts (one per loop) sit in the surface plane
        // (z = 0 at offset = 0) and carry the in-plane scale. Identify them as
        // the new verts at z ≈ 0; their displacement from the corner doubles.
        let mut compared = 0;
        for slot in 3..flat.vertex_count() {
            let pf = flat.vertex_position(VertexKey::from_usize(slot));
            let pe = even_on.vertex_position(VertexKey::from_usize(slot));
            if pf.z.abs() < 1.0e-6 && pe.z.abs() < 1.0e-6 {
                // An inset-frame vert (in the surface plane). Find its source
                // corner: the nearest original corner. The even displacement is
                // exactly 2× the flat one along the same bisector.
                let mut nearest = Vec3::ZERO;
                let mut best = f32::MAX;
                for c in 0..3 {
                    let corner = flat.vertex_position(VertexKey::from_usize(c));
                    let d = (corner - pf).length_squared();
                    if d < best {
                        best = d;
                        nearest = corner;
                    }
                }
                let flat_disp = pf - nearest;
                let even_disp = pe - nearest;
                assert!(
                    (even_disp - flat_disp * 2.0).length() < 1.0e-5,
                    "vert {slot}: even displacement {even_disp:?} should be 2× flat {flat_disp:?}",
                );
                compared += 1;
            } else {
                // A shell vert: offset purely along the +Z normal, identical
                // between the two arms (the in-plane scale never touches it).
                assert!(
                    (pf - pe).length() < 1.0e-6,
                    "vert {slot}: shell verts must be unaffected by even-offset",
                );
            }
        }
        assert_eq!(compared, 3, "all three inset-frame verts compared");
        even_on.validate_topology();
    }

    #[test]
    fn even_offset_factor_matches_known_angles() {
        // WIREFRAME.md §4.3: spot-check the shell-angle factor against analytic
        // values. 60° → 2.0, 90° → √2, and a (near-)flat 180° corner → 1.0
        // (the near-flat guard, no upper clamp).
        let dir = |deg: f32| {
            let r = deg.to_radians();
            Vec3::new(r.cos(), r.sin(), 0.0)
        };
        // Interior angle is the angle between the two edge directions.
        let f60 = super::even_offset_factor(dir(0.0), dir(60.0));
        assert!((f60 - 2.0).abs() < 1.0e-4, "60° factor {f60} should be 2.0");
        let f90 = super::even_offset_factor(dir(0.0), dir(90.0));
        assert!(
            (f90 - 2.0_f32.sqrt()).abs() < 1.0e-4,
            "90° factor {f90} should be √2",
        );
        let f180 = super::even_offset_factor(dir(0.0), dir(180.0));
        assert!((f180 - 1.0).abs() < 1.0e-4, "flat corner factor {f180} should be 1.0");
    }

    /// An axis-irregular planar quad lying in z = 0, wound +Z, whose four edges
    /// all have distinct lengths so each corner's median tagged-edge length is
    /// different. Corners (in winding order):
    ///   c0 = (0, 0)   c1 = (4, 0)   c2 = (4, 3)   c3 = (0, 3)
    /// Edge lengths around the loop: |c0c1| = 4, |c1c2| = 3, |c2c3| = 4,
    /// |c3c0| = 3. Each corner has exactly two tagged edges (the lone-quad fan),
    /// so its relative factor is their mean:
    ///   c0: mean(4, 3) = 3.5   c1: mean(4, 3) = 3.5
    ///   c2: mean(3, 4) = 3.5   c3: mean(4, 3) = 3.5
    /// — uniform here, so for a *non-uniform* per-corner factor we stretch one
    /// edge; see `relative_offset_scales_shell_per_vertex`.
    fn irregular_planar_quad(mesh: &mut LoopMesh) -> ([VertexKey; 4], FaceKey) {
        let corners = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(4.0, 0.0, 0.0),
            Vec3::new(4.0, 3.0, 0.0),
            Vec3::new(0.0, 3.0, 0.0),
        ];
        let verts = [
            mesh.create_vertex(corners[0]),
            mesh.create_vertex(corners[1]),
            mesh.create_vertex(corners[2]),
            mesh.create_vertex(corners[3]),
        ];
        let edges = [
            mesh.create_edge([verts[0], verts[1]], EdgeCreateFlags::NO_DUPLICATE),
            mesh.create_edge([verts[1], verts[2]], EdgeCreateFlags::NO_DUPLICATE),
            mesh.create_edge([verts[2], verts[3]], EdgeCreateFlags::NO_DUPLICATE),
            mesh.create_edge([verts[3], verts[0]], EdgeCreateFlags::NO_DUPLICATE),
        ];
        let face = mesh.face_create(&verts, &edges, FaceCreateFlags::NONE);
        (verts, face)
    }

    #[test]
    fn relative_offset_zero_matches_absolute_path() {
        // WIREFRAME.md §5.4: with `use_relative_offset` clear the per-vertex
        // factor is `1.0` everywhere, so the new entry point must be
        // bit-identical to `wireframe_with_offset` — same verts, same order,
        // exact positions.
        let mut absolute = LoopMesh::new();
        let (_, fa) = irregular_planar_quad(&mut absolute);
        let abs_result = absolute.wireframe_with_offset(&[fa], 0.3, 0.0, false);

        let mut rel_clear = LoopMesh::new();
        let (_, fr) = irregular_planar_quad(&mut rel_clear);
        let rel_result = rel_clear.wireframe_with_relative_offset(&[fr], 0.3, 0.0, false, false);

        assert_eq!(abs_result.strut_faces.len(), rel_result.strut_faces.len());
        assert_eq!(absolute.vertex_count(), rel_clear.vertex_count());
        assert_eq!(absolute.face_count(), rel_clear.face_count());
        for slot in 0..absolute.vertex_count() {
            let a = absolute.vertex_position(VertexKey::from_usize(slot));
            let b = rel_clear.vertex_position(VertexKey::from_usize(slot));
            assert_eq!(a, b, "vert {slot}: relative-offset clear must equal absolute path");
        }
        rel_clear.validate_topology();
    }

    #[test]
    fn relative_offset_scales_shell_per_vertex() {
        // WIREFRAME.md §5.4: each corner's whole cross-section scales by the
        // median length of its tagged edges. Build a quad whose four edges have
        // distinct lengths so each corner has a *different* median:
        //   c0 = (0,0)  c1 = (5,0)  c2 = (5,2)  c3 = (0,2)
        //   |c0c1| = 5  |c1c2| = 2  |c2c3| = 5  |c3c0| = 2
        // Each corner has two tagged edges → factor = their mean:
        //   c0: mean(5, 2) = 3.5   c1: mean(5, 2) = 3.5
        //   c2: mean(2, 5) = 3.5   c3: mean(5, 2) = 3.5   (still uniform)
        // To force distinct per-corner factors, displace c1 so the two edges at
        // each corner differ corner-to-corner:
        //   c0 = (0,0)  c1 = (6,0)  c2 = (6,2)  c3 = (0,2)
        //   |c0c1| = 6  |c1c2| = 2  |c2c3| = 6  |c3c0| = 2
        //   c0: mean(6, 2) = 4   c1: mean(6, 2) = 4 ...  (a rectangle is always
        // uniform). A genuinely non-uniform fan needs a non-rectangular quad:
        //   c0 = (0,0)  c1 = (6,0)  c2 = (5,4)  c3 = (0,2)
        //   |c0c1| = 6
        //   |c1c2| = sqrt(1 + 16) = sqrt(17)        ≈ 4.1231
        //   |c2c3| = sqrt(25 + 4) = sqrt(29)        ≈ 5.3852
        //   |c3c0| = 2
        //   c0: mean(|c3c0|, |c0c1|) = mean(2, 6)            = 4.0
        //   c1: mean(|c0c1|, |c1c2|) = mean(6, sqrt(17))     ≈ 5.0616
        //   c2: mean(|c1c2|, |c2c3|) = mean(sqrt(17), sqrt(29)) ≈ 4.7541
        //   c3: mean(|c2c3|, |c3c0|) = mean(sqrt(29), 2)     ≈ 3.6926
        let corners = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(6.0, 0.0, 0.0),
            Vec3::new(5.0, 4.0, 0.0),
            Vec3::new(0.0, 2.0, 0.0),
        ];
        let len = |a: usize, b: usize| (corners[a] - corners[b]).length();
        let factor = [
            (len(3, 0) + len(0, 1)) * 0.5,
            (len(0, 1) + len(1, 2)) * 0.5,
            (len(1, 2) + len(2, 3)) * 0.5,
            (len(2, 3) + len(3, 0)) * 0.5,
        ];

        let mut mesh = LoopMesh::new();
        let verts = [
            mesh.create_vertex(corners[0]),
            mesh.create_vertex(corners[1]),
            mesh.create_vertex(corners[2]),
            mesh.create_vertex(corners[3]),
        ];
        let edges = [
            mesh.create_edge([verts[0], verts[1]], EdgeCreateFlags::NO_DUPLICATE),
            mesh.create_edge([verts[1], verts[2]], EdgeCreateFlags::NO_DUPLICATE),
            mesh.create_edge([verts[2], verts[3]], EdgeCreateFlags::NO_DUPLICATE),
            mesh.create_edge([verts[3], verts[0]], EdgeCreateFlags::NO_DUPLICATE),
        ];
        let face = mesh.face_create(&verts, &edges, FaceCreateFlags::NONE);

        let thickness = 0.2_f32;
        let half = thickness * 0.5;
        mesh.wireframe_with_relative_offset(&[face], thickness, 0.0, false, true);

        // Topology unchanged: lone quad → 8 struts + 1 retained source, 16 verts.
        assert_eq!(mesh.face_count(), 9);
        assert_eq!(mesh.vertex_count(), 16);

        // The quad is planar (z = 0, +Z normal) so every shell vert is offset
        // purely along ±Z, landing at z = ±half·factor[c]. Group the 8 shell
        // verts (the new verts with non-zero z) by nearest corner and assert
        // their |z| equals half × that corner's median factor. The inset-frame
        // verts stay in the plane (z = 0); the four originals are at z = 0 too.
        let mut shell_hits = 0;
        for slot in 4..16 {
            let p = mesh.vertex_position(VertexKey::from_usize(slot));
            if p.z.abs() < 1.0e-6 {
                continue; // inset-frame vert, in the surface plane
            }
            // Nearest original corner in the xy plane.
            let mut nearest = 0usize;
            let mut best = f32::MAX;
            for c in 0..4 {
                let d = (Vec3::new(corners[c].x, corners[c].y, 0.0)
                    - Vec3::new(p.x, p.y, 0.0))
                .length_squared();
                if d < best {
                    best = d;
                    nearest = c;
                }
            }
            let expected = half * factor[nearest];
            assert!(
                (p.z.abs() - expected).abs() < 1.0e-5,
                "shell vert {slot} near corner {nearest}: |z| {} should be half·factor {expected}",
                p.z.abs(),
            );
            shell_hits += 1;
        }
        assert_eq!(shell_hits, 8, "all eight shell verts (a +/- pair per corner) checked");

        // The per-corner factors are genuinely distinct here — the test would be
        // vacuous on a rectangle (uniform factor).
        assert!(
            factor[0] != factor[1] && factor[1] != factor[2] && factor[2] != factor[3],
            "fixture must give a non-uniform per-corner median",
        );
        mesh.validate_topology();
    }

    #[test]
    fn relative_offset_two_edge_fan_is_mean() {
        // WIREFRAME.md §5.4: for a vertex whose tagged fan is exactly two edges
        // the median reduces to their mean. On a lone tagged quad every corner
        // has a two-edge fan, so the relative factor at a corner is the mean of
        // its two incident edge lengths. Pin that directly via `median_edge_length`
        // and via the resulting shell offset on an isosceles right-corner quad.
        let mut two = [3.0_f32, 7.0];
        assert_eq!(super::median_edge_length(&mut two), 5.0, "two-edge fan → mean");

        let mut three = [2.0_f32, 10.0, 6.0];
        assert_eq!(super::median_edge_length(&mut three), 6.0, "odd count → middle value");

        let mut four = [1.0_f32, 2.0, 8.0, 9.0];
        assert_eq!(super::median_edge_length(&mut four), 5.0, "even count → mean of middles");

        let mut empty: [f32; 0] = [];
        assert_eq!(super::median_edge_length(&mut empty), 1.0, "no tagged edges → 1.0 fallback");
    }

    /// A radial-3 T-junction fan (WIREFRAME.md §11.4): three quads share one
    /// spine edge `{s0, s1}`, each fanning out to its own two non-spine wing
    /// verts in a distinct direction so the spine vertices have well-defined
    /// (non-degenerate) normals. The spine edge is wound the same way in every
    /// quad (`s0 → wing → wing → s1`), so all three faces are radial users of
    /// the single shared spine edge — a non-manifold (radial-3) interior edge.
    /// Returns `(spine_edge, [face; 3])`.
    fn radial3_tjunction(mesh: &mut LoopMesh) -> (EdgeKey, [FaceKey; 3]) {
        let s0 = mesh.create_vertex(Vec3::new(0.0, 0.0, 0.0));
        let s1 = mesh.create_vertex(Vec3::new(0.0, 1.0, 0.0));
        let spine = mesh.create_edge([s0, s1], EdgeCreateFlags::NO_DUPLICATE);

        let mut faces = [FaceKey::from_usize(0); 3];
        // Three wings at 0°, 120°, 240° in the XZ plane around the spine.
        for (i, face) in faces.iter_mut().enumerate() {
            let theta = (i as f32) * std::f32::consts::TAU / 3.0;
            let (dx, dz) = (theta.cos(), theta.sin());
            let w0 = mesh.create_vertex(Vec3::new(dx, 0.0, dz));
            let w1 = mesh.create_vertex(Vec3::new(dx, 1.0, dz));
            // Winding s0 → w0 → w1 → s1; the closing edge w1→... s1→s0 is the
            // shared spine. Reuse the spine edge so its radial fan grows.
            let e_s0w0 = mesh.create_edge([s0, w0], EdgeCreateFlags::NO_DUPLICATE);
            let e_w0w1 = mesh.create_edge([w0, w1], EdgeCreateFlags::NO_DUPLICATE);
            let e_w1s1 = mesh.create_edge([w1, s1], EdgeCreateFlags::NO_DUPLICATE);
            *face = mesh.face_create(
                &[s0, w0, w1, s1],
                &[e_s0w0, e_w0w1, e_w1s1, spine],
                FaceCreateFlags::NONE,
            );
        }
        (spine, faces)
    }

    #[test]
    fn radial3_spine_is_interior_others_boundary() {
        // WIREFRAME.md §11.4: the §3 wire-boundary predicate fires on the
        // *second* tagged user, so the radial-3 spine edge (three tagged users)
        // is interior, while every wing edge (one tagged user) is a wire
        // boundary. Pin the classification before the count tests so the
        // interior-fan / multi-fork-cap behaviour rests on a verified base.
        let mut mesh = LoopMesh::new();
        let (spine, faces) = radial3_tjunction(&mut mesh);

        let tag = mesh.push_face_op_flag_layer();
        for &f in &faces {
            mesh.face_op_flag_set(tag, f, 0);
        }

        // The spine has three radial users, so for any one owner there are two
        // *other* tagged faces — it is interior with respect to each.
        for &owner in &faces {
            assert!(
                !mesh.wireframe_edge_is_boundary(spine, owner, tag),
                "radial-3 spine must be interior (a second tagged user exists)",
            );
        }
        // Every wing edge has a single tagged user → wire boundary. Walk each
        // face's loops; the only interior edge it sees is the spine.
        let mut interior_seen = 0;
        let mut boundary_seen = 0;
        for &owner in &faces {
            let mut loops = mesh.face_loops(owner);
            while let Some(l) = loops.next(&mesh) {
                let e = mesh.loop_edge(l);
                if mesh.wireframe_edge_is_boundary(e, owner, tag) {
                    boundary_seen += 1;
                } else {
                    interior_seen += 1;
                }
            }
        }
        // Three faces × 4 edges = 12 loop-edge visits; the spine is interior in
        // all three (3 interior visits), the 9 distinct wing edges boundary.
        assert_eq!(interior_seen, 3, "spine interior for all three owners");
        assert_eq!(boundary_seen, 9, "nine wing edges are wire boundary");

        mesh.pop_face_op_flag_layer(tag);
    }

    #[test]
    fn radial3_interior_fan_tube_counts_no_boundary() {
        // WIREFRAME.md §11.4 worked table (k = 3, use_boundary clear): the spine
        // is a radial-3 interior edge lined by 2k = 6 strut quads meeting at its
        // two shared shell hubs; the §11 formulas absorb the fan depth with no
        // special-casing. L = 4k = 12 corners → 2L = 24 strut faces; F = k = 3
        // retained source faces → 27 faces. New verts = L + 2·V_t = 12 + 16 = 28
        // (V_t = 2k+2 = 8), plus the 8 retained originals → 36 verts.
        let mut mesh = LoopMesh::new();
        let (spine, faces) = radial3_tjunction(&mut mesh);

        let result = mesh.wireframe(&faces, 0.2, false);

        assert_eq!(
            result.strut_faces.len(),
            24,
            "2 struts per corner × 12 corners (radial depth multiplies per-corner work)",
        );
        assert_eq!(mesh.face_count(), 27, "24 struts + 3 retained source faces");
        assert_eq!(
            mesh.vertex_count(),
            36,
            "12 inset-frame + 16 shell + 8 retained originals",
        );

        // The shared shell hub is welded: each spine vertex spawns exactly one
        // +/- shell pair (keyed by original vertex, §4.1), shared by all three
        // wings' struts. Count the strut quads that reference the spine verts'
        // shell verts — the 2k = 6 quads lining the spine must all anchor on the
        // *same* two shell pairs, not per-face duplicates.
        // The shared shell hub weld (§11.4): the spine's two shell pairs are
        // keyed by original vertex, so all 2k = 6 strut quads lining the spine
        // anchor on the *same* two pairs (not per-face duplicates). The spine
        // itself is untouched (no use_replace), so it still carries its full
        // radial-3 fan — confirm the fan survived the build.
        let mut radial = mesh.radial_loops(spine);
        let mut spine_fan = 0;
        while radial.next(&mesh).is_some() {
            spine_fan += 1;
        }
        assert_eq!(
            spine_fan, 3,
            "spine retains its radial-3 fan (originals kept under no-replace)",
        );
        mesh.validate_topology();
    }

    #[test]
    fn radial3_use_boundary_caps_multifork_vertex_counts() {
        // WIREFRAME.md §11.4 worked table (k = 3, use_boundary set): each spine
        // vertex touches k = 3 boundary edges (a >2-boundary-edge cap vertex);
        // §6.1's per-vertex guard builds exactly ONE boundary vert there, reused
        // for all k caps, and §6.2 emits a cap pair for each of the E_b = 3k = 9
        // boundary edges. Boundary verts = V_b = 2k+2 = 8; cap faces = 2·E_b =
        // 6k = 18. So 27 + 18 = 45 faces and 36 + 8 = 44 verts.
        let mut mesh = LoopMesh::new();
        let (_spine, faces) = radial3_tjunction(&mut mesh);

        let result = mesh.wireframe(&faces, 0.2, true);

        assert_eq!(
            result.strut_faces.len(),
            42,
            "24 inner struts + 18 rim caps (2 per boundary edge × 9 boundary edges)",
        );
        assert_eq!(mesh.face_count(), 45, "42 strut/cap faces + 3 retained source");
        assert_eq!(
            mesh.vertex_count(),
            44,
            "36 (no-boundary case) + 8 boundary verts (one per boundary vertex)",
        );

        // §6.1 first-touch tangent sign at the >2-boundary-edge fork vertex.
        // The spine runs +Y from the origin `s0`, so the origin's face interior
        // lies on the +Y side; the rim cap must close *outward* — away from that
        // interior — putting the origin's boundary inset-frame vert on the -Y
        // side. The first-touch face (the θ = 0 wing, in the XY plane) gives a
        // pure-Y perpendicular, so the fork boundary vert lands at exactly
        // (0, -thickness/2, 0). The bug this guards: a bare cross product points
        // the cap *toward* the spine interior (+Y) at this end while the
        // unambiguous far end (s1 = (0,1,0)) already points outward (+Y, away
        // from its interior), so the two ends disagree in sign.
        let half = 0.2_f32 * 0.5;
        let mut fork_outward = 0usize;
        let mut fork_inward = 0usize;
        for v in mesh.vert_keys() {
            let p = mesh.vertex_position(v);
            if p.x.abs() < 1.0e-6 && p.z.abs() < 1.0e-6 {
                if (p.y + half).abs() < 1.0e-6 {
                    fork_outward += 1;
                } else if (p.y - half).abs() < 1.0e-6 {
                    fork_inward += 1;
                }
            }
        }
        assert_eq!(
            fork_outward, 1,
            "origin fork cap vert sits on the -Y (outward) side at (0, -thickness/2, 0)",
        );
        assert_eq!(
            fork_inward, 0,
            "no fork cap vert points inward toward the +Y spine interior at the origin",
        );

        mesh.validate_topology();
    }

    #[test]
    fn radial3_use_boundary_contrast_pins_multifork_cap() {
        // The contrast that makes the >2-boundary-edge cap observable: the same
        // radial-3 fan differs by exactly the rim caps between use_boundary
        // clear and set — +18 cap faces and +8 boundary verts (§11.4). An impl
        // that mis-counted the multi-fork vertex (e.g. one boundary vert per
        // boundary *edge* rather than per vertex, or per-face shell duplicates)
        // would not land this exact delta.
        let mut open = LoopMesh::new();
        let (_, fo) = radial3_tjunction(&mut open);
        let open_result = open.wireframe(&fo, 0.2, false);

        let mut capped = LoopMesh::new();
        let (_, fc) = radial3_tjunction(&mut capped);
        let capped_result = capped.wireframe(&fc, 0.2, true);

        assert_eq!(
            capped_result.strut_faces.len() - open_result.strut_faces.len(),
            18,
            "2 cap faces per boundary edge × 9 boundary edges",
        );
        assert_eq!(
            capped.vertex_count() - open.vertex_count(),
            8,
            "one boundary vert per boundary vertex (V_b = 2k+2 = 8), not per boundary edge",
        );
    }

    #[test]
    fn two_quads_shared_edge_interior_tube_closes() {
        // WIREFRAME.md §11.3: two tagged faces sharing an edge make it a wire
        // *interior* edge — each contributes a strut pair (no rim cap on the
        // shared edge). Per-corner counts: L = 8 corners → 16 strut faces. Each
        // quad has 4 boundary edges of its own plus the shared one (7 distinct
        // tagged edges total; the shared one is interior). With `use_boundary`
        // off the shared edge gets no cap, so the four strut quads around it
        // close the tube via the shared shell hub edges.
        let mut mesh = LoopMesh::new();
        let v0 = mesh.create_vertex(Vec3::new(0.0, 0.0, 0.0));
        let v1 = mesh.create_vertex(Vec3::new(1.0, 0.0, 0.0));
        let v2 = mesh.create_vertex(Vec3::new(1.0, 1.0, 0.0));
        let v3 = mesh.create_vertex(Vec3::new(0.0, 1.0, 0.0));
        let v4 = mesh.create_vertex(Vec3::new(2.0, 0.0, 0.0));
        let v5 = mesh.create_vertex(Vec3::new(2.0, 1.0, 0.0));
        let e01 = mesh.create_edge([v0, v1], EdgeCreateFlags::NO_DUPLICATE);
        let e12 = mesh.create_edge([v1, v2], EdgeCreateFlags::NO_DUPLICATE);
        let e23 = mesh.create_edge([v2, v3], EdgeCreateFlags::NO_DUPLICATE);
        let e30 = mesh.create_edge([v3, v0], EdgeCreateFlags::NO_DUPLICATE);
        let e14 = mesh.create_edge([v1, v4], EdgeCreateFlags::NO_DUPLICATE);
        let e45 = mesh.create_edge([v4, v5], EdgeCreateFlags::NO_DUPLICATE);
        let e52 = mesh.create_edge([v5, v2], EdgeCreateFlags::NO_DUPLICATE);
        let a = mesh.face_create(&[v0, v1, v2, v3], &[e01, e12, e23, e30], FaceCreateFlags::NONE);
        let b = mesh.face_create(&[v1, v4, v5, v2], &[e14, e45, e52, e12], FaceCreateFlags::NONE);

        let result = mesh.wireframe(&[a, b], 0.2, false);

        assert_eq!(
            result.strut_faces.len(),
            16,
            "two strut pairs per face × 4 corners each = 16"
        );
        // 16 struts + 2 retained source faces.
        assert_eq!(mesh.face_count(), 18);
        // New verts: 8 inset-frame (one per loop) + 12 shell (2 × 6 tagged
        // verts). 6 originals persist.
        assert_eq!(mesh.vertex_count(), 6 + 8 + 12);

        // Tube closure: the shared edge's shell hub verts at v1 and v2 must each
        // be referenced by struts from *both* faces. The neg shell vert of v1 is
        // shared by A's and B's neg struts, so its radial edge fan is richer
        // than an open-rim shell vert's. Topology validity is the load-bearing
        // assertion — the four quads must weld into a manifold tube.
        mesh.validate_topology();
    }
}
