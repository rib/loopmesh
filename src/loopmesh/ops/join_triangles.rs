use std::collections::BinaryHeap;
use std::f32::consts::PI;

use crate::loopmesh::*;

/// Float-noise slack: two face normals whose squared difference is within this
/// bound contribute exactly zero to the planarity term, matching the
/// "near-flat quad adds nothing" rule in
/// `radial-mesh-guides/operators/JOIN_TRIANGLES.md` §5.1 term 1.
const NORMAL_EPS: f32 = 0.0001;

/// §6.3 cap. The influence discount can drive a candidate's error toward a
/// good neighbour quad's error but never *all the way* to it: a sliver of the
/// original error is always kept so the relative ordering among influenced
/// candidates survives, and so a non-zero-error competitor can never be
/// out-pulled by a candidate against a perfect (zero-error) rival (§6.4
/// mechanism 1). The multiplier `topology_influence × alignment` is clamped to
/// this cap.
const INFLUENCE_CAP: f32 = 0.95;

/// §6.1 seed over-prioritisation. A pre-existing input quad's own §5.1 error is
/// scaled *down* by this factor before it seeds its neighbours, so "a good
/// existing quad is treated as near-perfect" and influences strongly — a lower
/// `err_q` makes the §6.3 discount `(err_q − err_e) × multiplier` more negative,
/// pulling grid-continuing candidates harder than a freshly-merged quad
/// (§6.2, which uses the raw error). The guide pins the *direction and intent*
/// of this scaling (over-prioritise toward near-perfect) but not the exact
/// scalar; this fraction is the loopmesh approximation and the precise value is
/// a parity surface the A/B un-stub must confirm.
const SEED_ERROR_SCALE: f32 = 0.25;

/// Parameters controlling [`LoopMesh::join_triangles`].
///
/// Mirrors the operator's BMOP slot set described in
/// `radial-mesh-guides/operators/JOIN_TRIANGLES.md` §1. Both angle thresholds
/// default to `π` (gate disabled); the bool gates default to `false`;
/// `topology_influence` defaults to `0` (pure quad-quality order).
#[derive(Clone, Copy)]
pub struct JoinTrianglesParams {
    /// §4.1 — reject an edge carrying the seam flag.
    pub cmp_seam: bool,
    /// §4.1 — reject an edge that is not smooth (the sharp / hard edge).
    pub cmp_sharp: bool,
    /// §4.1 — reject an edge whose two triangles carry different materials.
    pub cmp_materials: bool,
    /// §4.2 — reject an edge that is a UV-island boundary.
    pub cmp_uvs: bool,
    /// §4.2 — reject an edge that is a vertex-colour-island boundary.
    pub cmp_vcols: bool,
    /// §4.3 — fold gate. When below `π`, reject an edge whose two triangles'
    /// face normals diverge by more than this angle (radians). At `π` disabled.
    pub angle_face_threshold: f32,
    /// §4.4 — shape gate. When below `π`, reject an edge whose resulting quad
    /// would deviate from a rectangle at any corner by more than this angle, or
    /// whose quad would be flipped (self-overlapping). At `π` disabled.
    pub angle_shape_threshold: f32,
    /// §6 — grid-flow re-prioritisation strength (0–2). At `0` (default) the
    /// merge order is pure quad-quality (§5.1). When non-zero, pre-existing and
    /// freshly-merged good quads discount their grid-continuing neighbour
    /// candidate edges so the merge front flows into existing grid topology; a
    /// purely *ordering* heuristic that can flip which contested candidate wins.
    pub topology_influence: f32,
    /// §7 — output-selection control. When `false` (default) the result's
    /// `joined` carries the merged quads; when `true` it instead reports the
    /// input triangles that were *not* merged in `unmerged`.
    pub deselect_joined: bool,
}

impl Default for JoinTrianglesParams {
    fn default() -> Self {
        Self {
            cmp_seam: false,
            cmp_sharp: false,
            cmp_materials: false,
            cmp_uvs: false,
            cmp_vcols: false,
            angle_face_threshold: PI,
            angle_shape_threshold: PI,
            topology_influence: 0.0,
            deselect_joined: false,
        }
    }
}

/// Outcome of [`LoopMesh::join_triangles`] (`JOIN_TRIANGLES.md` §7).
///
/// `joined` lists the merged quads produced by the sweep (the default
/// `faces.out` content). `unmerged` lists the input triangles that survived
/// un-merged (the `deselect_joined` content). Both are always populated so a
/// caller can read either regardless of the `deselect_joined` flag; the flag
/// only changes which set BMesh's `faces.out` slot would report.
#[derive(Clone, PartialEq, Eq, Debug, Default)]
pub struct JoinTrianglesResult {
    pub joined: Vec<FaceKey>,
    pub unmerged: Vec<FaceKey>,
}

/// One queued candidate edge in the merge heap, ordered ascending by `error`
/// (best-quad-first, §5.1). `insertion_order` breaks error ties by the edge's
/// mesh-iteration order (§8), the deterministic tiebreak BMesh's queue uses.
#[derive(Copy, Clone)]
struct Candidate {
    error: f32,
    insertion_order: u32,
    edge: EdgeKey,
}

impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == std::cmp::Ordering::Equal
    }
}
impl Eq for Candidate {}
impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // `BinaryHeap` is a max-heap; invert so the lowest error pops first,
        // and the earlier insertion order wins an error tie.
        other
            .error
            .total_cmp(&self.error)
            .then(other.insertion_order.cmp(&self.insertion_order))
    }
}

/// The precomputed delimit-gate configuration for one operator run (§4).
struct DelimitConfig {
    cmp_seam: bool,
    cmp_sharp: bool,
    cmp_materials: bool,
    cmp_uvs: bool,
    cmp_vcols: bool,
    /// `Some(cos θ)` when the §4.3 fold gate is active (`angle_face_threshold <
    /// π`); `None` when disabled.
    angle_face_cos: Option<f32>,
    /// `Some(θ)` when the §4.4 shape gate is active; `None` when disabled.
    angle_shape_threshold: Option<f32>,
    /// The UV (Vec2) loop layers consulted by the §4.2 `cmp_uvs` gate.
    uv_layers: Vec<crate::loopmesh::ElementAttributeKey>,
    /// The colour (Vec4) loop layers consulted by the §4.2 `cmp_vcols` gate.
    vcol_layers: Vec<crate::loopmesh::ElementAttributeKey>,
    /// The resolved per-face `material_index` layer key consulted by the §4.1
    /// `cmp_materials` gate, resolved once per run (mirroring `uv_layers` /
    /// `vcol_layers`) so the per-edge check needs no descriptor scan. `None`
    /// when `cmp_materials` is off or no material layer exists — in the latter
    /// case every face reads as the default `0`.
    material_layer: Option<crate::loopmesh::ElementAttributeKey>,
    /// The resolved per-edge `seam` layer key consulted by the §4.1 `cmp_seam`
    /// gate, resolved once per run (mirroring `material_layer`) so the per-edge
    /// check needs no descriptor scan. `None` when `cmp_seam` is off or no seam
    /// layer exists — in the latter case every edge reads as "no seam".
    seam_layer: Option<crate::loopmesh::ElementAttributeKey>,
    /// The resolved per-edge `sharp` layer key consulted by the §4.1 `cmp_sharp`
    /// gate, resolved once per run (mirroring `seam_layer`) so the per-edge check
    /// needs no descriptor scan. `None` when `cmp_sharp` is off or no sharp layer
    /// exists — in the latter case every edge reads as **smooth** (not sharp), so
    /// the gate does not fire by default.
    sharp_layer: Option<crate::loopmesh::ElementAttributeKey>,
}

/// The per-face attribute layer name carrying each face's material index, the
/// convention the §4.1 `cmp_materials` gate reads. Stored as a single-component
/// `I32` numeric layer; a face with no stored value (or no layer at all) is
/// material `0`, the default. See `radial-mesh-guides/operators/DISSOLVE_LIMIT.md`
/// §3 (the `MATERIAL` delimiter) and `JOIN_TRIANGLES.md` §4.1.
pub const MATERIAL_INDEX_ATTRIBUTE: &str = "material_index";

/// The per-edge attribute layer name carrying each edge's seam flag, the
/// convention the §4.1 `cmp_seam` gate reads. Stored as a single-component
/// `I32` numeric layer (mirroring the per-face material-index choice): a
/// non-zero value means "this edge is a seam", and an edge with no stored value
/// (or no layer at all) is not a seam. The per-domain boolean-attribute model
/// for persistent edge state such as seams is the one endorsed in
/// `radial-mesh-guides/BMESH.md` §6; see `JOIN_TRIANGLES.md` §4.1.
pub const SEAM_ATTRIBUTE: &str = "seam";

/// The per-edge attribute layer name carrying each edge's sharp (non-smooth)
/// flag, the convention the §4.1 `cmp_sharp` gate reads. Stored as a
/// single-component `I32` numeric layer (mirroring [`SEAM_ATTRIBUTE`]): a
/// non-zero value means "this edge is sharp" (the smooth shading hint is
/// cleared), and an edge with no stored value (or no layer at all) is **smooth**
/// — i.e. not sharp.
///
/// Note the polarity, pinned by `radial-mesh-guides/operators/JOIN_TRIANGLES.md`
/// §4.1 and `radial-mesh-guides/BMESH.md` §6: an edge is "sharp" precisely when
/// it is *not smooth*, and the default state of an edge is **smooth**. The
/// per-domain boolean-attribute model that endorses representing shading
/// sharpness as a `sharp_edge` attribute (rather than a smooth header bit) is the
/// one described in `BMESH.md` §6. Because the default reads as not-sharp, an
/// unset/default edge never blocks the merge under `cmp_sharp` — the same net
/// "default does not block" effect [`SEAM_ATTRIBUTE`] has, despite the inverted
/// meaning of the stored bit.
pub const SHARP_ATTRIBUTE: &str = "sharp";

impl LoopMesh {
    /// Whether `edge` carries the seam flag, or the default `false` when no
    /// per-edge seam layer exists or the edge has no stored value.
    ///
    /// The seam flag is a per-edge attribute under the [`SEAM_ATTRIBUTE`]
    /// convention; the §4.1 `cmp_seam` gate of [`LoopMesh::join_triangles`]
    /// rejects an edge that carries it.
    pub fn edge_seam(&self, edge: EdgeKey) -> bool {
        let attribs = self.edge_attribs();
        match attribs.find(&AttributeKey::custom(SEAM_ATTRIBUTE)) {
            Some(layer) => attribs.get_i32(layer, edge.as_usize()).unwrap_or(0) != 0,
            None => false,
        }
    }

    /// Set `edge`'s seam flag under the [`SEAM_ATTRIBUTE`] convention,
    /// registering the per-edge `I32` layer on first use.
    ///
    /// Idempotent in the layer: repeated calls reuse the single registered
    /// layer. Edges never assigned a value keep the default `false`.
    pub fn set_edge_seam(&mut self, edge: EdgeKey, seam: bool) {
        let key = AttributeKey::custom(SEAM_ATTRIBUTE);
        let layer = match self.edge_attribs().find(&key) {
            Some(layer) => layer,
            None => self
                .register_edge_attribute(ElementAttributeDesc::numeric(
                    key,
                    AttrType::I32,
                ))
                .expect("I32x1 is a supported edge attribute format"),
        };
        self.edge_attribs_mut()
            .set_i32(layer, edge.as_usize(), seam as i32);
    }

    /// Whether `edge` is **sharp** (the smooth shading hint is cleared), or the
    /// default `false` (smooth) when no per-edge sharp layer exists or the edge
    /// has no stored value.
    ///
    /// The sharp flag is a per-edge attribute under the [`SHARP_ATTRIBUTE`]
    /// convention; the §4.1 `cmp_sharp` gate of [`LoopMesh::join_triangles`]
    /// rejects an edge that is sharp (not smooth). The default state of an edge
    /// is smooth, so an unset edge reads as not-sharp and never blocks the merge.
    pub fn edge_sharp(&self, edge: EdgeKey) -> bool {
        let attribs = self.edge_attribs();
        match attribs.find(&AttributeKey::custom(SHARP_ATTRIBUTE)) {
            Some(layer) => attribs.get_i32(layer, edge.as_usize()).unwrap_or(0) != 0,
            None => false,
        }
    }

    /// Set `edge`'s sharp flag under the [`SHARP_ATTRIBUTE`] convention,
    /// registering the per-edge `I32` layer on first use. Setting `true` marks
    /// the edge sharp (not smooth); setting `false` marks it smooth.
    ///
    /// Idempotent in the layer: repeated calls reuse the single registered
    /// layer. Edges never assigned a value keep the default `false` (smooth).
    pub fn set_edge_sharp(&mut self, edge: EdgeKey, sharp: bool) {
        let key = AttributeKey::custom(SHARP_ATTRIBUTE);
        let layer = match self.edge_attribs().find(&key) {
            Some(layer) => layer,
            None => self
                .register_edge_attribute(ElementAttributeDesc::numeric(
                    key,
                    AttrType::I32,
                ))
                .expect("I32x1 is a supported edge attribute format"),
        };
        self.edge_attribs_mut()
            .set_i32(layer, edge.as_usize(), sharp as i32);
    }

    /// The material index assigned to `face`, or the default `0` when no
    /// per-face material layer exists or the face has no stored value.
    ///
    /// The material index is a per-face integer attribute under the
    /// [`MATERIAL_INDEX_ATTRIBUTE`] convention; the §4.1 `cmp_materials` gate of
    /// [`LoopMesh::join_triangles`] rejects an edge whose two faces disagree on
    /// this value.
    pub fn face_material_index(&self, face: FaceKey) -> i32 {
        let attribs = self.face_attribs();
        match attribs.find(&AttributeKey::custom(MATERIAL_INDEX_ATTRIBUTE)) {
            Some(layer) => attribs.get_i32(layer, face.as_usize()).unwrap_or(0),
            None => 0,
        }
    }

    /// Assign `face`'s material index under the [`MATERIAL_INDEX_ATTRIBUTE`]
    /// convention, registering the per-face `I32` layer on first use.
    ///
    /// Idempotent in the layer: repeated calls reuse the single registered
    /// layer. Faces never assigned a value keep the default `0`.
    pub fn set_face_material_index(&mut self, face: FaceKey, index: i32) {
        let key = AttributeKey::custom(MATERIAL_INDEX_ATTRIBUTE);
        let layer = match self.face_attribs().find(&key) {
            Some(layer) => layer,
            None => self
                .register_face_attribute(ElementAttributeDesc::numeric(
                    key,
                    AttrType::I32,
                ))
                .expect("I32x1 is a supported face attribute format"),
        };
        self.face_attribs_mut().set_i32(layer, face.as_usize(), index);
    }

    /// Merge adjacent input triangles back into quads.
    ///
    /// Implements `radial-mesh-guides/operators/JOIN_TRIANGLES.md`:
    ///
    /// - **§3.1** — tag every input face that is a triangle; non-triangles are
    ///   ignored.
    /// - **§3.2 / §3.3** — gather every manifold edge whose two faces are both
    ///   input triangles and which passes every enabled delimit gate (§4),
    ///   scoring each by the §5.1 quad-error of the quad it would produce.
    /// - **§5.2 / §5.3** — pop the best-quad-first candidate and merge its two
    ///   triangles via the JFKE primitive ([`LoopMesh::faces_join`],
    ///   `RADIAL_EDGE_STRUCTURE.md` §8.4); a candidate whose triangle was
    ///   already consumed by an earlier merge is declined (stale-skip).
    /// - **§4** — the delimit gates: flag/material (§4.1), customdata-island
    ///   (§4.2), fold-angle (§4.3), quad-shape/sliver (§4.4).
    /// - **§7** — the output: by default the merged quads; with
    ///   `deselect_joined`, the un-merged input triangles.
    ///
    /// The §4.2 UV / vertex-colour island gates read loop customdata, which
    /// loopmesh stores directly. The §4.1 material gate reads the per-face
    /// material attribute layer ([`LoopMesh::face_material_index`] /
    /// [`LoopMesh::set_face_material_index`]). The §4.1 seam gate reads the
    /// per-edge seam attribute layer ([`LoopMesh::edge_seam`] /
    /// [`LoopMesh::set_edge_seam`]). The §4.1 sharp gate reads the per-edge sharp
    /// attribute layer ([`LoopMesh::edge_sharp`] / [`LoopMesh::set_edge_sharp`]);
    /// an edge is sharp when it is *not smooth*, and the default state is smooth,
    /// so an unset edge never blocks the merge.
    ///
    /// - **§6** — when `topology_influence` is non-zero, pre-existing input
    ///   quads (§6.1) and each freshly-merged quad (§6.2) discount their
    ///   grid-continuing neighbour candidate edges (§6.3), reordering the heap
    ///   so the merge front flows into existing grid topology. At `0` (default)
    ///   none of §6 runs and the order is pure §5.1 quality.
    pub fn join_triangles(
        &mut self,
        faces: &[FaceKey],
        params: JoinTrianglesParams,
    ) -> JoinTrianglesResult {
        let delimit = self.join_triangles_build_delimit(&params);

        // Face normals drive the §4.3 fold gate and the §5.1 quad-error
        // planarity term; refresh them so a freshly built mesh scores correctly.
        self.calc_face_normals();

        // §3.1 — tag the input triangles. A pushed op-flag layer that vanishes
        // when the operator returns.
        let input_tag = self.push_face_op_flag_layer();
        const INPUT_BIT: u8 = 0;
        for &face in faces {
            if self.faces.try_get(face).is_some() && self.face_len(face) == 3 {
                self.face_op_flag_set(input_tag, face, INPUT_BIT);
            }
        }

        let influence_active = params.topology_influence != 0.0;

        // §3.2 — gather candidate edges into the merge heap. When §6 is active
        // we also keep an edge→error map of the *authoritative* score for each
        // queued candidate; a re-score (§6.3) lowers that score and pushes a
        // fresh heap entry, and a stale (superseded) entry is dropped on pop.
        let mut heap: BinaryHeap<Candidate> = BinaryHeap::new();
        let mut current_error: std::collections::HashMap<EdgeKey, (f32, u32)> =
            std::collections::HashMap::new();
        let mut insertion_order: u32 = 0;
        {
            let mut edge_iter = self.mesh_edges();
            while let Some(edge) = edge_iter.next(self) {
                if let Some((face_a, face_b)) =
                    self.join_triangles_edge_tri_pair(edge, input_tag, INPUT_BIT)
                {
                    if self.join_triangles_is_delimit(edge, face_a, face_b, &delimit) {
                        continue;
                    }
                    let corners = self.join_triangles_quad_corners(edge);
                    let error = join_triangles_quad_error(&corners);
                    heap.push(Candidate {
                        error,
                        insertion_order,
                        edge,
                    });
                    if influence_active {
                        current_error.insert(edge, (error, insertion_order));
                    }
                    insertion_order += 1;
                }
            }
        }

        // §6.1 — seed the influence from pre-existing input quads, before any
        // merge, so a good existing quad pulls its grid-continuing neighbour
        // candidates toward the front of the queue.
        if influence_active && !heap.is_empty() {
            for &face in faces {
                if self.faces.try_get(face).is_some() && self.face_len(face) == 4 {
                    // §6.1 — seed quads are over-prioritised: their error is
                    // scaled toward near-perfect so they pull harder than a
                    // freshly-merged quad (§6.2) does.
                    let seed_error =
                        join_triangles_quad_error(&self.join_triangles_quad_corners_of_face(face))
                            * SEED_ERROR_SCALE;
                    self.join_triangles_reprioritise_quad(
                        face,
                        seed_error,
                        params.topology_influence,
                        input_tag,
                        INPUT_BIT,
                        &mut heap,
                        &mut current_error,
                    );
                }
            }
        }

        // §5.2 — greedy pop-and-merge.
        let mut joined: Vec<FaceKey> = Vec::new();
        while let Some(candidate) = heap.pop() {
            // §6 lazy-deletion: skip an entry whose score has since been
            // superseded by a lower re-scored entry for the same edge.
            if influence_active {
                match current_error.get(&candidate.edge) {
                    Some(&(error, order))
                        if error == candidate.error
                            && order == candidate.insertion_order => {}
                    _ => continue,
                }
            }
            if let Some(quad) = self.join_triangles_try_merge(candidate.edge) {
                joined.push(quad);
                if influence_active {
                    current_error.remove(&candidate.edge);
                    // §6.2 — the freshly-built quad re-scores its neighbours so
                    // a clean merge propagates the grid front. Unlike the §6.1
                    // seed pass, the post-merge re-score uses the quad's *raw*
                    // §5.1 error (no over-prioritisation scaling).
                    let quad_error = join_triangles_quad_error(
                        &self.join_triangles_quad_corners_of_face(quad),
                    );
                    self.join_triangles_reprioritise_quad(
                        quad,
                        quad_error,
                        params.topology_influence,
                        input_tag,
                        INPUT_BIT,
                        &mut heap,
                        &mut current_error,
                    );
                }
            } else if influence_active {
                current_error.remove(&candidate.edge);
            }
        }

        // §7 — the un-merged input triangles are the input faces that are still
        // live triangles after the sweep. The slot is a set (§8), so a face
        // that appears more than once in `faces` is reported once: the still-set
        // INPUT tag doubles as the "not yet recorded" marker, cleared on record.
        let mut unmerged: Vec<FaceKey> = Vec::new();
        for &face in faces {
            if self.faces.try_get(face).is_some()
                && self.face_len(face) == 3
                && self.face_op_flag_test(input_tag, face, INPUT_BIT)
            {
                unmerged.push(face);
                self.face_op_flag_clear(input_tag, face, INPUT_BIT);
            }
        }

        self.pop_face_op_flag_layer(input_tag);
        JoinTrianglesResult { joined, unmerged }
    }

    /// §4 — assemble the gate configuration once per run, resolving the angle
    /// thresholds to their active forms and collecting the loop layers the
    /// customdata-island gates consult.
    fn join_triangles_build_delimit(&self, params: &JoinTrianglesParams) -> DelimitConfig {
        let angle_face_cos = if params.angle_face_threshold < PI {
            Some(params.angle_face_threshold.cos())
        } else {
            None
        };
        let angle_shape_threshold = if params.angle_shape_threshold < PI {
            Some(params.angle_shape_threshold)
        } else {
            None
        };

        let attribs = self.loop_attribs();
        let uv_layers: Vec<crate::loopmesh::ElementAttributeKey> = if params.cmp_uvs {
            attribs
                .iter_descriptors()
                .filter(|(key, _)| {
                    matches!(
                        attribs.storage_of(*key),
                        Some(crate::loopmesh::AttributeStorage::Vec2(_))
                    )
                })
                .map(|(key, _)| key)
                .collect()
        } else {
            Vec::new()
        };
        let vcol_layers: Vec<crate::loopmesh::ElementAttributeKey> = if params.cmp_vcols {
            attribs
                .iter_descriptors()
                .filter(|(key, _)| {
                    matches!(
                        attribs.storage_of(*key),
                        Some(crate::loopmesh::AttributeStorage::Vec4(_))
                    )
                })
                .map(|(key, _)| key)
                .collect()
        } else {
            Vec::new()
        };

        let material_layer = if params.cmp_materials {
            self.face_attribs()
                .find(&AttributeKey::custom(MATERIAL_INDEX_ATTRIBUTE))
        } else {
            None
        };

        let seam_layer = if params.cmp_seam {
            self.edge_attribs().find(&AttributeKey::custom(SEAM_ATTRIBUTE))
        } else {
            None
        };

        let sharp_layer = if params.cmp_sharp {
            self.edge_attribs().find(&AttributeKey::custom(SHARP_ATTRIBUTE))
        } else {
            None
        };

        DelimitConfig {
            cmp_seam: params.cmp_seam,
            cmp_sharp: params.cmp_sharp,
            cmp_materials: params.cmp_materials,
            cmp_uvs: params.cmp_uvs,
            cmp_vcols: params.cmp_vcols,
            angle_face_cos,
            angle_shape_threshold,
            uv_layers,
            vcol_layers,
            material_layer,
            seam_layer,
            sharp_layer,
        }
    }

    /// §3.2 — the two distinct faces flanking a manifold `edge`, returned only
    /// when both are INPUT-tagged triangles. `None` for boundary / non-manifold
    /// edges or when either face is not a tagged triangle.
    fn join_triangles_edge_tri_pair(
        &self,
        edge: EdgeKey,
        input_tag: crate::op_flags::FaceOpFlagLayer,
        input_bit: u8,
    ) -> Option<(FaceKey, FaceKey)> {
        let mut radial = self.radial_loops(edge);
        let head = radial.next(self)?;
        let second = radial.next(self)?;
        if radial.next(self).is_some() {
            return None;
        }
        let head_face = self.loops.get(head).face;
        let second_face = self.loops.get(second).face;
        if head_face == second_face {
            return None;
        }
        if !self.face_op_flag_test(input_tag, head_face, input_bit)
            || !self.face_op_flag_test(input_tag, second_face, input_bit)
        {
            return None;
        }
        Some((head_face, second_face))
    }

    /// §3.3 — the four corners of the quad a merge across `edge` would produce,
    /// read off the radial-edge structure without performing the merge. The
    /// quad winds: `edge.start → opposite-tri apex → edge.end → own-tri apex`.
    fn join_triangles_quad_corners(&self, edge: EdgeKey) -> [Vec3; 4] {
        let loop_a = self.edges.get(edge).radial_link.expect("manifold edge has a loop");
        let loop_op = self.loop_radial_next(loop_a);

        let start = self.loops.get(loop_a).vertex;
        let opposite_apex = self.loops.get(self.loop_prev(loop_op)).vertex;
        let end = self.loops.get(self.loop_next(loop_a)).vertex;
        let own_apex = self.loops.get(self.loop_prev(loop_a)).vertex;

        [
            self.vertices.get(start).position,
            self.vertices.get(opposite_apex).position,
            self.vertices.get(end).position,
            self.vertices.get(own_apex).position,
        ]
    }

    /// §4 — true when `edge` fails any enabled delimit gate; checked in §4.1 →
    /// §4.4 order, short-circuiting on the first failure.
    fn join_triangles_is_delimit(
        &self,
        edge: EdgeKey,
        face_a: FaceKey,
        face_b: FaceKey,
        delimit: &DelimitConfig,
    ) -> bool {
        // §4.1 — flag and material gates.
        if delimit.cmp_seam && self.join_triangles_edge_is_seam_cached(delimit.seam_layer, edge) {
            return true;
        }
        if delimit.cmp_sharp && self.join_triangles_edge_is_sharp_cached(delimit.sharp_layer, edge) {
            return true;
        }
        if delimit.cmp_materials
            && self.join_triangles_face_material_cached(delimit.material_layer, face_a)
                != self.join_triangles_face_material_cached(delimit.material_layer, face_b)
        {
            return true;
        }

        // §4.2 — customdata-island gates.
        if delimit.cmp_uvs && !self.join_triangles_edge_loop_cd_contiguous(edge, &delimit.uv_layers) {
            return true;
        }
        if delimit.cmp_vcols
            && !self.join_triangles_edge_loop_cd_contiguous(edge, &delimit.vcol_layers)
        {
            return true;
        }

        // §4.3 — fold gate.
        if let Some(angle_face_cos) = delimit.angle_face_cos {
            let normal_a = self.faces.get(face_a).normal;
            let normal_b = self.faces.get(face_b).normal;
            if normal_a.dot(normal_b) < angle_face_cos {
                return true;
            }
        }

        // §4.4 — shape gate (flipped quad or corner deviation).
        if let Some(threshold) = delimit.angle_shape_threshold {
            let corners = self.join_triangles_quad_corners(edge);
            if join_triangles_quad_is_flipped(&corners)
                || join_triangles_quad_corner_deviation_exceeds(&corners, threshold)
            {
                return true;
            }
        }

        false
    }

    /// §4.2 — true when `edge` is loop-CD contiguous across *every* given loop
    /// layer: each layer's per-loop value agrees at both shared endpoints
    /// across the edge's two flanking faces. A mismatch on any layer at either
    /// endpoint is an island boundary (returns `false`). With no layers the
    /// edge is trivially contiguous.
    ///
    /// The two flanking loops `la`, `lb` each run from their `.vertex` to
    /// `loop_next(.).vertex`; the loop↔edge invariant
    /// (`RADIAL_EDGE_STRUCTURE.md` §5.1) makes those the edge's two endpoints.
    /// Corners pair by shared endpoint vertex.
    fn join_triangles_edge_loop_cd_contiguous(
        &self,
        edge: EdgeKey,
        layers: &[crate::loopmesh::ElementAttributeKey],
    ) -> bool {
        if layers.is_empty() {
            return true;
        }
        let mut radial = self.radial_loops(edge);
        let Some(loop_a) = radial.next(self) else {
            return true;
        };
        let Some(loop_b) = radial.next(self) else {
            return true;
        };

        let a_start = loop_a;
        let a_end = self.loop_next(loop_a);
        let b_start = loop_b;
        let b_end = self.loop_next(loop_b);

        let vertex_a = self.loops.get(a_start).vertex;
        let (b_at_a_start, b_at_a_end) = if self.loops.get(b_start).vertex == vertex_a {
            (b_start, b_end)
        } else {
            (b_end, b_start)
        };

        for &layer in layers {
            if self.join_triangles_loop_cd(layer, a_start)
                != self.join_triangles_loop_cd(layer, b_at_a_start)
                || self.join_triangles_loop_cd(layer, a_end)
                    != self.join_triangles_loop_cd(layer, b_at_a_end)
            {
                return false;
            }
        }
        true
    }

    /// Read a loop's value from a Vec2 or Vec4 loop layer as a 4-vector for the
    /// §4.2 contiguity comparison; the unused components stay zero. Exact
    /// per-component equality, matching the dissolve-family island predicate.
    fn join_triangles_loop_cd(
        &self,
        layer: crate::loopmesh::ElementAttributeKey,
        loop_key: LoopKey,
    ) -> [f32; 4] {
        let attribs = self.loop_attribs();
        let slot = loop_key.as_usize();
        if let Some(uv) = attribs.get_vec2(layer, slot) {
            return [uv[0], uv[1], 0.0, 0.0];
        }
        if let Some(col) = attribs.get_vec4(layer, slot) {
            return col;
        }
        [0.0; 4]
    }

    /// §5.3 — merge the two triangles flanking `edge` into one quad via the
    /// JFKE primitive, or `None` when either side is no longer a triangle (a
    /// stale candidate whose triangle was already consumed). The merge keeps
    /// the JFKE customdata convention from [`LoopMesh::faces_join`].
    fn join_triangles_try_merge(&mut self, edge: EdgeKey) -> Option<FaceKey> {
        if self.edges.try_get(edge).is_none() {
            return None;
        }
        let mut radial = self.radial_loops(edge);
        let head = radial.next(self)?;
        let second = radial.next(self)?;
        if radial.next(self).is_some() {
            return None;
        }
        let face_a = self.loops.get(head).face;
        let face_b = self.loops.get(second).face;
        if face_a == face_b {
            return None;
        }
        if self.face_len(face_a) != 3 || self.face_len(face_b) != 3 {
            return None;
        }
        self.faces_join(&[face_a, face_b])
    }

    /// §4.1 — whether `edge` carries the seam flag, read through the layer key
    /// resolved once per run in [`Self::join_triangles_build_delimit`]. A `None`
    /// key (no seam layer registered) or an edge with no stored value reports
    /// the default `false`, so the `cmp_seam` gate only fires on a genuinely
    /// seam-flagged edge. This is the hot-path equivalent of the public
    /// [`LoopMesh::edge_seam`], avoiding the per-call descriptor scan and
    /// `AttributeKey` allocation that accessor performs.
    fn join_triangles_edge_is_seam_cached(
        &self,
        seam_layer: Option<crate::loopmesh::ElementAttributeKey>,
        edge: EdgeKey,
    ) -> bool {
        match seam_layer {
            Some(layer) => self.edge_attribs().get_i32(layer, edge.as_usize()).unwrap_or(0) != 0,
            None => false,
        }
    }

    /// §4.1 — whether `edge` is sharp (the smooth bit cleared), read through the
    /// layer key resolved once per run in [`Self::join_triangles_build_delimit`].
    /// A `None` key (no sharp layer registered) or an edge with no stored value
    /// reports the default `false` (smooth), so the `cmp_sharp` gate only fires
    /// on a genuinely sharp edge and never blocks the merge by default. This is
    /// the hot-path equivalent of the public [`LoopMesh::edge_sharp`], avoiding
    /// the per-call descriptor scan and `AttributeKey` allocation that accessor
    /// performs.
    fn join_triangles_edge_is_sharp_cached(
        &self,
        sharp_layer: Option<crate::loopmesh::ElementAttributeKey>,
        edge: EdgeKey,
    ) -> bool {
        match sharp_layer {
            Some(layer) => self.edge_attribs().get_i32(layer, edge.as_usize()).unwrap_or(0) != 0,
            None => false,
        }
    }

    /// §4.1 — the face's material index, read through the layer key resolved
    /// once per run in [`Self::join_triangles_build_delimit`]. A `None` key (no
    /// material layer registered) or a face with no stored value reports the
    /// default `0`, so the `cmp_materials` gate only fires when two adjacent
    /// faces carry *different* indices. This is the hot-path equivalent of the
    /// public [`LoopMesh::face_material_index`], avoiding the per-call
    /// descriptor scan and `AttributeKey` allocation that accessor performs.
    fn join_triangles_face_material_cached(
        &self,
        material_layer: Option<crate::loopmesh::ElementAttributeKey>,
        face: FaceKey,
    ) -> i32 {
        match material_layer {
            Some(layer) => self.face_attribs().get_i32(layer, face.as_usize()).unwrap_or(0),
            None => 0,
        }
    }

    /// §6.1 / §6.2 — re-score the candidate edges of `quad`'s neighbouring
    /// input triangles, discounting each grid-continuing candidate toward
    /// `quad_error` (the seed quad's scaled error in §6.1, the freshly-merged
    /// quad's raw error in §6.2).
    ///
    /// For every perimeter edge of `quad`, the face on the other side of that
    /// edge — when it is an input triangle — is a neighbour whose mergeable
    /// candidate edges should be pulled toward the front of the queue (§6.1).
    /// Each such candidate is discounted per §6.3, with grid alignment measured
    /// between the two quads across the shared edge.
    ///
    /// §10 budgets this as a constant (≤ 8 neighbours) with no allocation, so
    /// the walk is over lending iterators: the quad's loop cycle and each
    /// neighbour's loop cycle, never a heap-`Vec` of keys.
    fn join_triangles_reprioritise_quad(
        &self,
        quad: FaceKey,
        quad_error: f32,
        topology_influence: f32,
        input_tag: crate::op_flags::FaceOpFlagLayer,
        input_bit: u8,
        heap: &mut BinaryHeap<Candidate>,
        current_error: &mut std::collections::HashMap<EdgeKey, (f32, u32)>,
    ) {
        let mut quad_loops = self.face_loops(quad);
        while let Some(quad_loop) = quad_loops.next(self) {
            let shared_edge = self.loop_edge(quad_loop);
            let Some(neighbour) = self.join_triangles_edge_other_face(shared_edge, quad) else {
                continue;
            };
            if self.face_len(neighbour) != 3
                || !self.face_op_flag_test(input_tag, neighbour, input_bit)
            {
                continue;
            }
            let mut neighbour_loops = self.face_loops(neighbour);
            while let Some(neighbour_loop) = neighbour_loops.next(self) {
                let candidate_edge = self.loop_edge(neighbour_loop);
                if candidate_edge == shared_edge {
                    continue;
                }
                let Some(&(error_e, order_e)) = current_error.get(&candidate_edge) else {
                    continue; // not a live queued candidate (merged, gated, or non-manifold)
                };
                if quad_error >= error_e {
                    continue; // a re-score that would raise error is ignored (§6.3)
                }
                let alignment =
                    self.join_triangles_grid_alignment(quad, shared_edge, candidate_edge);
                if alignment <= 0.0 {
                    continue;
                }
                let multiplier = (topology_influence * alignment).min(INFLUENCE_CAP);
                let new_error = error_e + (quad_error - error_e) * multiplier;
                if new_error < error_e {
                    current_error.insert(candidate_edge, (new_error, order_e));
                    heap.push(Candidate {
                        error: new_error,
                        insertion_order: order_e,
                        edge: candidate_edge,
                    });
                }
            }
        }
    }

    /// The face on the radial-opposite side of a manifold `edge` from `face`,
    /// or `None` when the edge is not shared by exactly two distinct faces.
    fn join_triangles_edge_other_face(&self, edge: EdgeKey, face: FaceKey) -> Option<FaceKey> {
        let mut radial = self.radial_loops(edge);
        let first = radial.next(self)?;
        let second = radial.next(self)?;
        if radial.next(self).is_some() {
            return None;
        }
        let face_a = self.loops.get(first).face;
        let face_b = self.loops.get(second).face;
        if face_a == face {
            Some(face_b)
        } else if face_b == face {
            Some(face_a)
        } else {
            None
        }
    }

    /// §6.3 alignment — how well the candidate's would-be quad lines up with
    /// the `neighbour_quad` once both are rotated into a common plane about
    /// their `shared_edge`. A candidate that continues the neighbour quad's
    /// grid row aligns near `1`; a diagonal or mismatched candidate near `0`.
    ///
    /// The measure compares the two quads' edge-direction *flows* across the
    /// shared edge, and is **fold-invariant** by construction: for each quad it
    /// takes the direction of the edge opposite the shared edge (the quad's
    /// "far rail" — the row the grid continues into) and decomposes it about the
    /// shared-edge axis into (a) its component *along* the shared edge and (b)
    /// the magnitude of its component *perpendicular* to the shared edge within
    /// that quad's own plane. Both parts are invariant under rotation about the
    /// shared-edge axis, so unfolding the natural fold between the two quads
    /// into a common plane leaves the measure unchanged (§6.3). The two quads'
    /// (along, perp) flow vectors are compared by cosine similarity; a grid
    /// continuation makes them parallel, giving `1`.
    ///
    /// Returns `0.0` (no influence) when the candidate edge is no longer
    /// manifold — a post-merge re-score (§6.2) runs after topology has mutated,
    /// so the candidate quad may have become unreconstructable; falling back to
    /// zero matches the non-manifold handling of
    /// [`Self::join_triangles_edge_other_face`] and avoids panicking on an
    /// invariant that now spans a mutation.
    fn join_triangles_grid_alignment(
        &self,
        neighbour_quad: FaceKey,
        shared_edge: EdgeKey,
        candidate_edge: EdgeKey,
    ) -> f32 {
        let [s_a, s_b] = self.edges.get(shared_edge).vertices;
        let pos_s_a = self.vertices.get(s_a).position;
        let pos_s_b = self.vertices.get(s_b).position;
        let axis = (pos_s_b - pos_s_a).normalize_or_zero();
        if axis == Vec3::ZERO {
            return 0.0;
        }

        let neighbour_corners = self.join_triangles_face_corners_verts(neighbour_quad);
        let Some(candidate_corners) = self.join_triangles_quad_corners_verts(candidate_edge) else {
            return 0.0;
        };

        let Some(neighbour_flow) =
            self.join_triangles_far_rail_flow(&neighbour_corners, s_a, s_b, axis)
        else {
            return 0.0;
        };
        let Some(candidate_flow) =
            self.join_triangles_far_rail_flow(&candidate_corners, s_a, s_b, axis)
        else {
            return 0.0;
        };

        let neighbour_dir = neighbour_flow.normalize_or_zero();
        let candidate_dir = candidate_flow.normalize_or_zero();
        if neighbour_dir == glam::Vec2::ZERO || candidate_dir == glam::Vec2::ZERO {
            return 0.0;
        }
        neighbour_dir.dot(candidate_dir).clamp(0.0, 1.0)
    }

    /// §6.3 helper — the fold-invariant, winding-consistent 2D flow of a quad's
    /// far rail about the shared edge. `corners` is the quad's 4-cycle of
    /// vertices; `s_a`/`s_b` are the shared-edge endpoints (which must appear
    /// consecutively in the cycle); `axis` is the unit shared-edge direction.
    ///
    /// The far rail is the cycle edge opposite the shared edge. To make the two
    /// quads' flows directly comparable regardless of each cycle's winding
    /// sense, the rail is oriented relative to **this quad's own traversal of
    /// the shared edge** (the direction `corners[i] → corners[i+1]`, always
    /// `±axis`): `along` is the far rail's component along that per-quad
    /// traversal, `perp` is its component along the in-plane direction pointing
    /// from the shared edge out toward the far rail. Both are invariant under
    /// rotation about the shared-edge axis (the fold-invariance §6.3 requires)
    /// and both are sign-consistent between the two quads. Returns `None` if the
    /// shared edge is not an edge of the cycle or the quad is degenerate.
    fn join_triangles_far_rail_flow(
        &self,
        corners: &[VertexKey; 4],
        s_a: VertexKey,
        s_b: VertexKey,
        axis: Vec3,
    ) -> Option<glam::Vec2> {
        let mut shared_index = None;
        for i in 0..4 {
            let a = corners[i];
            let b = corners[(i + 1) % 4];
            if (a == s_a && b == s_b) || (a == s_b && b == s_a) {
                shared_index = Some(i);
                break;
            }
        }
        let shared_index = shared_index?;

        // The cycle traverses the shared edge corners[si] → corners[si+1]; this
        // is ±axis and fixes the orientation against which the far rail is read.
        let shared_start = self.vertices.get(corners[shared_index]).position;
        let shared_end = self.vertices.get(corners[(shared_index + 1) % 4]).position;
        let traverse = (shared_end - shared_start).normalize_or_zero();
        if traverse == Vec3::ZERO {
            return None;
        }

        let far_a = corners[(shared_index + 2) % 4];
        let far_b = corners[(shared_index + 3) % 4];
        let far_dir = self.vertices.get(far_b).position - self.vertices.get(far_a).position;

        // The perpendicular reference points from the shared edge toward the far
        // rail (out into the quad), so the perpendicular sign is consistent
        // between the two quads being compared.
        let shared_mid = (shared_start + shared_end) * 0.5;
        let far_mid = (self.vertices.get(far_a).position + self.vertices.get(far_b).position) * 0.5;
        let outward = far_mid - shared_mid;
        let outward_perp = (outward - axis * outward.dot(axis)).normalize_or_zero();
        if outward_perp == Vec3::ZERO {
            return None;
        }

        let along = far_dir.dot(traverse);
        let perp = far_dir.dot(outward_perp);
        Some(glam::Vec2::new(along, perp))
    }

    /// §3.3 corner reconstruction returning the four corner *vertices* (cycle
    /// order) of the quad a merge across `edge` would build, or `None` when the
    /// edge is no longer manifold (a post-merge §6.2 re-score may reach an edge
    /// whose radial cycle has changed). Mirrors the graceful non-manifold path
    /// of [`Self::join_triangles_edge_other_face`].
    fn join_triangles_quad_corners_verts(&self, edge: EdgeKey) -> Option<[VertexKey; 4]> {
        let loop_a = self.edges.try_get(edge).and_then(|e| e.radial_link)?;
        let loop_op = self.loop_radial_next(loop_a);
        Some([
            self.loops.get(loop_a).vertex,
            self.loops.get(self.loop_prev(loop_op)).vertex,
            self.loops.get(self.loop_next(loop_a)).vertex,
            self.loops.get(self.loop_prev(loop_a)).vertex,
        ])
    }

    /// The four corner *vertices* of an existing quad `face` in cycle order, for
    /// the §6.3 alignment comparison. Uses the face-loop lending iterator (no
    /// allocation), matching the §10 constant-cost budget.
    fn join_triangles_face_corners_verts(&self, face: FaceKey) -> [VertexKey; 4] {
        let mut loops = self.face_loops(face);
        // A re-prioritise seed/merge target is always a 4-cycle (face_len == 4),
        // so the cycle yields exactly four loops; seed the array with the first
        // loop's vertex as a placeholder rather than carry a sentinel constant.
        let placeholder = self.loops.get(self.face_first_loop(face)).vertex;
        let mut corners = [placeholder; 4];
        let mut index = 0;
        while let Some(loop_key) = loops.next(self) {
            if index < 4 {
                corners[index] = self.loops.get(loop_key).vertex;
            }
            index += 1;
        }
        corners
    }

    /// The four corner positions of an existing quad `face` in cycle order, for
    /// scoring its §5.1 quad-error as an influence seed (§6.1 / §6.2).
    fn join_triangles_quad_corners_of_face(&self, face: FaceKey) -> [Vec3; 4] {
        let mut positions = [Vec3::ZERO; 4];
        let mut loops = self.face_loops(face);
        let mut index = 0;
        while let Some(loop_key) = loops.next(self) {
            if index < 4 {
                positions[index] = self.vertices.get(self.loops.get(loop_key).vertex).position;
            }
            index += 1;
        }
        positions
    }
}

/// §5.1 — the quad-error join-quality measure: the sum of the non-planarity,
/// non-rectangularity, and concavity terms. Lower is a better quad and merges
/// first.
fn join_triangles_quad_error(quad: &[Vec3; 4]) -> f32 {
    join_triangles_planarity_term(quad)
        + join_triangles_rectangularity_term(quad)
        + join_triangles_concavity_term(quad)
}

/// §5.1 term 1 — non-planarity. The quad is split along each diagonal into two
/// triangles; the angle between the two triangle normals is measured per
/// diagonal, the two summed, and normalised by 2π. A flat quad contributes 0;
/// normals that match within `NORMAL_EPS` contribute 0 for that split.
fn join_triangles_planarity_term(quad: &[Vec3; 4]) -> f32 {
    let [v1, v2, v3, v4] = *quad;
    let split_angle = |a: Vec3, b: Vec3, c: Vec3, d: Vec3, e: Vec3, f: Vec3| -> f32 {
        let normal_first = (b - a).cross(c - a);
        let normal_second = (e - d).cross(f - d);
        if (normal_first - normal_second).length_squared() < NORMAL_EPS {
            return 0.0;
        }
        normal_first.angle_between(normal_second)
    };
    // Diagonal v1–v3: triangles (v1,v2,v3) and (v1,v3,v4).
    let diagonal_a = split_angle(v1, v2, v3, v1, v3, v4);
    // Diagonal v2–v4: triangles (v2,v3,v4) and (v2,v4,v1).
    let diagonal_b = split_angle(v2, v3, v4, v2, v4, v1);
    (diagonal_a + diagonal_b) / (2.0 * PI)
}

/// §5.1 term 2 — non-rectangularity. The four perimeter edge directions are
/// taken as unit vectors; at each corner the angle between successive
/// directions is compared to π/2; the four absolute deviations are summed and
/// normalised by 2π. Four right-angle corners contribute 0.
fn join_triangles_rectangularity_term(quad: &[Vec3; 4]) -> f32 {
    let directions = join_triangles_perimeter_directions(quad);
    let mut total = 0.0f32;
    for corner in 0..4 {
        // At each corner the incoming edge reversed meets the outgoing edge.
        let incoming = -directions[(corner + 3) % 4];
        let outgoing = directions[corner];
        let angle = incoming.angle_between(outgoing);
        total += (angle - std::f32::consts::FRAC_PI_2).abs();
    }
    total / (2.0 * PI)
}

/// §5.1 term 3 — concavity (area imbalance). The quad is triangulated both ways;
/// the total area of each pairing is computed; the term is `1 − area_min /
/// area_max` (or `1` when the larger area is zero). A balanced convex quad
/// contributes near 0; a lopsided concave one near 1.
fn join_triangles_concavity_term(quad: &[Vec3; 4]) -> f32 {
    let [v1, v2, v3, v4] = *quad;
    let area = |a: Vec3, b: Vec3, c: Vec3| -> f32 { (b - a).cross(c - a).length() * 0.5 };
    let pairing_a = area(v1, v2, v3) + area(v1, v3, v4);
    let pairing_b = area(v2, v3, v4) + area(v4, v1, v2);
    let (area_min, area_max) = if pairing_a <= pairing_b {
        (pairing_a, pairing_b)
    } else {
        (pairing_b, pairing_a)
    };
    if area_max <= 0.0 {
        1.0
    } else {
        1.0 - area_min / area_max
    }
}

/// The four perimeter edge directions of the quad (`v1→v2`, `v2→v3`, `v3→v4`,
/// `v4→v1`) as unit vectors. A zero-length edge yields a zero vector.
fn join_triangles_perimeter_directions(quad: &[Vec3; 4]) -> [Vec3; 4] {
    [
        (quad[1] - quad[0]).normalize_or_zero(),
        (quad[2] - quad[1]).normalize_or_zero(),
        (quad[3] - quad[2]).normalize_or_zero(),
        (quad[0] - quad[3]).normalize_or_zero(),
    ]
}

/// §4.4 item 2 — true when any of the quad's four corner angles deviates from a
/// right angle (π/2) by more than `threshold`.
fn join_triangles_quad_corner_deviation_exceeds(quad: &[Vec3; 4], threshold: f32) -> bool {
    let directions = join_triangles_perimeter_directions(quad);
    for corner in 0..4 {
        let incoming = -directions[(corner + 3) % 4];
        let outgoing = directions[corner];
        let angle = incoming.angle_between(outgoing);
        if (angle - std::f32::consts::FRAC_PI_2).abs() > threshold {
            return true;
        }
    }
    false
}

/// §4.4 item 1 — true when the quad is flipped (self-overlapping, "bow-tie").
///
/// The four corners arrive in the §3.3 winding `start → opposite-apex → end →
/// own-apex`, so `start`/`end` are the merge edge's endpoints and the other two
/// are the two source triangles' apexes. A simple quad — convex *or* concave —
/// has the two apexes on opposite sides of the shared-edge line; the two source
/// triangles then meet edge-to-edge without overlapping. A bow-tie is exactly
/// the case where both apexes fall on the *same* side: the two triangles
/// overlap and the merged perimeter crosses itself.
///
/// We read the side of each apex from the sign of `edge × (apex − start)` about
/// the same reference vector. Opposite sides give a negative dot product
/// (simple); a non-negative dot means same-side (overlap) or collinear, both of
/// which are rejected as not a valid simple quad.
fn join_triangles_quad_is_flipped(quad: &[Vec3; 4]) -> bool {
    let [start, opposite_apex, end, own_apex] = *quad;
    let edge_direction = end - start;
    let opposite_side = edge_direction.cross(opposite_apex - start);
    let own_side = edge_direction.cross(own_apex - start);
    opposite_side.dot(own_side) >= 0.0
}

#[cfg(test)]
mod tests {
    use glam::Vec3;

    use super::JoinTrianglesParams;
    use crate::loopmesh::*;

    /// Build a single triangle from three fresh verts at the given positions,
    /// returning the face key. Edges are deduplicated so two triangles sharing
    /// an edge share its edge key.
    fn build_tri(mesh: &mut LoopMesh, positions: [Vec3; 3]) -> FaceKey {
        let verts: Vec<VertexKey> = positions.iter().map(|&p| mesh.create_vertex(p)).collect();
        build_tri_from_verts(mesh, [verts[0], verts[1], verts[2]])
    }

    fn build_tri_from_verts(mesh: &mut LoopMesh, verts: [VertexKey; 3]) -> FaceKey {
        let e0 = mesh.create_edge([verts[0], verts[1]], EdgeCreateFlags::NO_DUPLICATE);
        let e1 = mesh.create_edge([verts[1], verts[2]], EdgeCreateFlags::NO_DUPLICATE);
        let e2 = mesh.create_edge([verts[2], verts[0]], EdgeCreateFlags::NO_DUPLICATE);
        mesh.face_create(&verts, &[e0, e1, e2], FaceCreateFlags::NONE)
    }

    /// Two coplanar triangles forming the unit square `{0,1,2,3}` split along
    /// the diagonal `0–2`. Returns `(face_lower, face_upper)`.
    fn two_tris_shared_edge(mesh: &mut LoopMesh) -> (FaceKey, FaceKey, [VertexKey; 4]) {
        let v0 = mesh.create_vertex(Vec3::new(0.0, 0.0, 0.0));
        let v1 = mesh.create_vertex(Vec3::new(1.0, 0.0, 0.0));
        let v2 = mesh.create_vertex(Vec3::new(1.0, 1.0, 0.0));
        let v3 = mesh.create_vertex(Vec3::new(0.0, 1.0, 0.0));
        // Lower triangle 0-1-2, upper triangle 0-2-3, sharing edge 0-2.
        let lower = build_tri_from_verts(mesh, [v0, v1, v2]);
        let upper = build_tri_from_verts(mesh, [v0, v2, v3]);
        (lower, upper, [v0, v1, v2, v3])
    }

    #[test]
    fn single_tri_pair_merges_to_quad() {
        let mut mesh = LoopMesh::new();
        let (lower, upper, _) = two_tris_shared_edge(&mut mesh);
        assert_eq!((mesh.totvert(), mesh.totedge(), mesh.totface()), (4, 5, 2));

        let result = mesh.join_triangles(&[lower, upper], JoinTrianglesParams::default());

        assert_eq!(mesh.totvert(), 4, "no vert removed");
        assert_eq!(mesh.totedge(), 4, "the shared diagonal is gone (−1 E)");
        assert_eq!(mesh.totface(), 1, "one quad (−1 F)");
        assert_eq!(result.joined.len(), 1);
        assert_eq!(mesh.face_len(result.joined[0]), 4);
        assert!(result.unmerged.is_empty());
        mesh.validate_topology();
    }

    #[test]
    fn material_index_round_trips_with_default_zero() {
        let mut mesh = LoopMesh::new();
        let (lower, upper, _) = two_tris_shared_edge(&mut mesh);

        // Unassigned faces are material 0 (no layer registered yet).
        assert_eq!(mesh.face_material_index(lower), 0);
        assert_eq!(mesh.face_material_index(upper), 0);

        mesh.set_face_material_index(lower, 2);
        // The second assignment reuses the registered layer (idempotent layer).
        mesh.set_face_material_index(upper, 5);

        assert_eq!(mesh.face_material_index(lower), 2);
        assert_eq!(mesh.face_material_index(upper), 5);
    }

    #[test]
    fn cmp_materials_blocks_merge_across_material_boundary() {
        // §4.1 — two adjacent triangles carrying different material indices must
        // not be merged when `cmp_materials` is enabled.
        let mut mesh = LoopMesh::new();
        let (lower, upper, _) = two_tris_shared_edge(&mut mesh);
        mesh.set_face_material_index(lower, 0);
        mesh.set_face_material_index(upper, 1);

        let result = mesh.join_triangles(
            &[lower, upper],
            JoinTrianglesParams {
                cmp_materials: true,
                ..Default::default()
            },
        );

        assert!(result.joined.is_empty(), "the material boundary blocks the merge");
        assert_eq!(mesh.totface(), 2, "both triangles survive");
        assert_eq!(mesh.face_len(lower), 3);
        assert_eq!(mesh.face_len(upper), 3);
        mesh.validate_topology();
    }

    #[test]
    fn cmp_materials_blocks_explicit_nonzero_vs_unset_default() {
        // The asymmetry the cached-key path must preserve: one face carries an
        // explicit non-zero index, the other is never assigned (default 0). The
        // material layer exists (registered by the first assignment) but the
        // unset face has no stored value, so the gate must still read it as 0
        // and fire on the mismatch.
        let mut mesh = LoopMesh::new();
        let (lower, upper, _) = two_tris_shared_edge(&mut mesh);
        mesh.set_face_material_index(lower, 5);
        // `upper` deliberately left unset → default 0.

        let result = mesh.join_triangles(
            &[lower, upper],
            JoinTrianglesParams {
                cmp_materials: true,
                ..Default::default()
            },
        );

        assert!(
            result.joined.is_empty(),
            "explicit 5 vs default 0 is a material boundary"
        );
        assert_eq!(mesh.totface(), 2, "both triangles survive");
        assert_eq!(mesh.face_material_index(upper), 0, "unset face reads default 0");
        mesh.validate_topology();
    }

    #[test]
    fn cmp_materials_allows_merge_for_equal_materials() {
        // Equal explicit materials, and the default (unset → 0) case, both let
        // the edge merge with `cmp_materials` enabled.
        for assign_equal in [false, true] {
            let mut mesh = LoopMesh::new();
            let (lower, upper, _) = two_tris_shared_edge(&mut mesh);
            if assign_equal {
                mesh.set_face_material_index(lower, 3);
                mesh.set_face_material_index(upper, 3);
            }

            let result = mesh.join_triangles(
                &[lower, upper],
                JoinTrianglesParams {
                    cmp_materials: true,
                    ..Default::default()
                },
            );

            assert_eq!(result.joined.len(), 1, "equal materials merge (assigned={assign_equal})");
            assert_eq!(mesh.totface(), 1);
            assert_eq!(mesh.face_len(result.joined[0]), 4);
            mesh.validate_topology();
        }
    }

    #[test]
    fn cmp_materials_false_ignores_material_difference() {
        // The default path: with `cmp_materials` off, differing materials are
        // irrelevant and the edge merges exactly as the no-material case does.
        let mut mesh = LoopMesh::new();
        let (lower, upper, _) = two_tris_shared_edge(&mut mesh);
        mesh.set_face_material_index(lower, 0);
        mesh.set_face_material_index(upper, 7);

        let result = mesh.join_triangles(&[lower, upper], JoinTrianglesParams::default());

        assert_eq!(result.joined.len(), 1, "material difference ignored when gate is off");
        assert_eq!(mesh.totface(), 1);
        assert_eq!(mesh.face_len(result.joined[0]), 4);
        mesh.validate_topology();
    }

    /// The shared diagonal edge of the `two_tris_shared_edge` fixture (the
    /// `0-2` edge between the lower and upper triangles).
    fn shared_diagonal(mesh: &LoopMesh, verts: [VertexKey; 4]) -> EdgeKey {
        mesh.query_edge_exists(verts[0], verts[2])
            .expect("the two triangles share the 0-2 diagonal")
    }

    #[test]
    fn edge_seam_round_trips_with_default_false() {
        let mut mesh = LoopMesh::new();
        let (_lower, _upper, verts) = two_tris_shared_edge(&mut mesh);
        let diagonal = shared_diagonal(&mesh, verts);
        let perimeter = mesh
            .query_edge_exists(verts[0], verts[1])
            .expect("perimeter edge 0-1 exists");

        // Unflagged edges are not seams (no layer registered yet).
        assert!(!mesh.edge_seam(diagonal));
        assert!(!mesh.edge_seam(perimeter));

        mesh.set_edge_seam(diagonal, true);
        // The second assignment reuses the registered layer (idempotent layer).
        mesh.set_edge_seam(perimeter, false);

        assert!(mesh.edge_seam(diagonal));
        assert!(!mesh.edge_seam(perimeter), "explicit false reads as no seam");

        // Clearing a previously-set seam round-trips back to false.
        mesh.set_edge_seam(diagonal, false);
        assert!(!mesh.edge_seam(diagonal));
    }

    #[test]
    fn cmp_seam_blocks_merge_across_seam_edge() {
        // §4.1 — a seam-flagged shared edge must not be merged across when
        // `cmp_seam` is enabled.
        let mut mesh = LoopMesh::new();
        let (lower, upper, verts) = two_tris_shared_edge(&mut mesh);
        let diagonal = shared_diagonal(&mesh, verts);
        mesh.set_edge_seam(diagonal, true);

        let result = mesh.join_triangles(
            &[lower, upper],
            JoinTrianglesParams {
                cmp_seam: true,
                ..Default::default()
            },
        );

        assert!(result.joined.is_empty(), "the seam blocks the merge");
        assert_eq!(mesh.totface(), 2, "both triangles survive");
        assert_eq!(mesh.face_len(lower), 3);
        assert_eq!(mesh.face_len(upper), 3);
        mesh.validate_topology();
    }

    #[test]
    fn cmp_seam_allows_merge_when_no_seam() {
        // With `cmp_seam` on, an edge with no seam (unset → default false, or
        // explicit false) merges exactly as the no-seam case does.
        for assign_false in [false, true] {
            let mut mesh = LoopMesh::new();
            let (lower, upper, verts) = two_tris_shared_edge(&mut mesh);
            if assign_false {
                let diagonal = shared_diagonal(&mesh, verts);
                mesh.set_edge_seam(diagonal, false);
            }

            let result = mesh.join_triangles(
                &[lower, upper],
                JoinTrianglesParams {
                    cmp_seam: true,
                    ..Default::default()
                },
            );

            assert_eq!(result.joined.len(), 1, "no seam merges (assigned={assign_false})");
            assert_eq!(mesh.totface(), 1);
            assert_eq!(mesh.face_len(result.joined[0]), 4);
            mesh.validate_topology();
        }
    }

    #[test]
    fn cmp_seam_false_ignores_seam_flag() {
        // The default path: with `cmp_seam` off, a seam-flagged edge is
        // irrelevant and the pair merges exactly as the no-seam case does.
        let mut mesh = LoopMesh::new();
        let (lower, upper, verts) = two_tris_shared_edge(&mut mesh);
        let diagonal = shared_diagonal(&mesh, verts);
        mesh.set_edge_seam(diagonal, true);

        let result = mesh.join_triangles(&[lower, upper], JoinTrianglesParams::default());

        assert_eq!(result.joined.len(), 1, "seam ignored when gate is off");
        assert_eq!(mesh.totface(), 1);
        assert_eq!(mesh.face_len(result.joined[0]), 4);
        mesh.validate_topology();
    }

    #[test]
    fn edge_sharp_round_trips_with_default_false() {
        let mut mesh = LoopMesh::new();
        let (_lower, _upper, verts) = two_tris_shared_edge(&mut mesh);
        let diagonal = shared_diagonal(&mesh, verts);
        let perimeter = mesh
            .query_edge_exists(verts[0], verts[1])
            .expect("perimeter edge 0-1 exists");

        // The default state of an edge is smooth, i.e. not sharp (no layer
        // registered yet). Polarity check: default reads as not-sharp.
        assert!(!mesh.edge_sharp(diagonal));
        assert!(!mesh.edge_sharp(perimeter));

        mesh.set_edge_sharp(diagonal, true);
        // The second assignment reuses the registered layer (idempotent layer).
        mesh.set_edge_sharp(perimeter, false);

        assert!(mesh.edge_sharp(diagonal));
        assert!(!mesh.edge_sharp(perimeter), "explicit false reads as smooth");

        // Clearing a previously-set sharp edge round-trips back to smooth.
        mesh.set_edge_sharp(diagonal, false);
        assert!(!mesh.edge_sharp(diagonal));
    }

    #[test]
    fn cmp_sharp_blocks_merge_across_sharp_edge() {
        // §4.1 — a sharp (non-smooth) shared edge must not be merged across when
        // `cmp_sharp` is enabled.
        let mut mesh = LoopMesh::new();
        let (lower, upper, verts) = two_tris_shared_edge(&mut mesh);
        let diagonal = shared_diagonal(&mesh, verts);
        mesh.set_edge_sharp(diagonal, true);

        let result = mesh.join_triangles(
            &[lower, upper],
            JoinTrianglesParams {
                cmp_sharp: true,
                ..Default::default()
            },
        );

        assert!(result.joined.is_empty(), "the sharp edge blocks the merge");
        assert_eq!(mesh.totface(), 2, "both triangles survive");
        assert_eq!(mesh.face_len(lower), 3);
        assert_eq!(mesh.face_len(upper), 3);
        mesh.validate_topology();
    }

    #[test]
    fn cmp_sharp_allows_merge_when_smooth() {
        // With `cmp_sharp` on, a smooth edge (unset → default smooth, or
        // explicit false) merges exactly as the no-sharp case does. The default
        // (unset) path is the parity subtlety: a default edge is smooth, not
        // sharp, so the gate must not block it.
        for assign_smooth in [false, true] {
            let mut mesh = LoopMesh::new();
            let (lower, upper, verts) = two_tris_shared_edge(&mut mesh);
            if assign_smooth {
                let diagonal = shared_diagonal(&mesh, verts);
                mesh.set_edge_sharp(diagonal, false);
            }

            let result = mesh.join_triangles(
                &[lower, upper],
                JoinTrianglesParams {
                    cmp_sharp: true,
                    ..Default::default()
                },
            );

            assert_eq!(result.joined.len(), 1, "smooth merges (assigned={assign_smooth})");
            assert_eq!(mesh.totface(), 1);
            assert_eq!(mesh.face_len(result.joined[0]), 4);
            mesh.validate_topology();
        }
    }

    #[test]
    fn cmp_sharp_false_ignores_sharp_flag() {
        // The default path: with `cmp_sharp` off, a sharp-flagged edge is
        // irrelevant and the pair merges exactly as the smooth case does.
        let mut mesh = LoopMesh::new();
        let (lower, upper, verts) = two_tris_shared_edge(&mut mesh);
        let diagonal = shared_diagonal(&mesh, verts);
        mesh.set_edge_sharp(diagonal, true);

        let result = mesh.join_triangles(&[lower, upper], JoinTrianglesParams::default());

        assert_eq!(result.joined.len(), 1, "sharp ignored when gate is off");
        assert_eq!(mesh.totface(), 1);
        assert_eq!(mesh.face_len(result.joined[0]), 4);
        mesh.validate_topology();
    }

    #[test]
    fn deselect_joined_reports_unmerged_triangles() {
        let mut mesh = LoopMesh::new();
        let (lower, upper, _) = two_tris_shared_edge(&mut mesh);

        let result = mesh.join_triangles(
            &[lower, upper],
            JoinTrianglesParams {
                deselect_joined: true,
                ..Default::default()
            },
        );

        // Both input triangles were consumed by the single merge, so the
        // un-merged-triangle set is empty; the quad still formed.
        assert_eq!(mesh.totface(), 1);
        assert!(result.unmerged.is_empty());
        mesh.validate_topology();
    }

    #[test]
    fn deselect_joined_reports_surviving_unmerged_triangle() {
        // A mergeable tri-pair plus a lone triangle that shares no mergeable
        // edge with anything: the pair merges, the lone triangle survives and
        // must appear in `unmerged`.
        let mut mesh = LoopMesh::new();
        let (lower, upper, _) = two_tris_shared_edge(&mut mesh);
        let lone = build_tri(
            &mut mesh,
            [
                Vec3::new(5.0, 0.0, 0.0),
                Vec3::new(6.0, 0.0, 0.0),
                Vec3::new(5.0, 1.0, 0.0),
            ],
        );

        let result = mesh.join_triangles(
            &[lower, upper, lone],
            JoinTrianglesParams {
                deselect_joined: true,
                ..Default::default()
            },
        );

        assert_eq!(result.joined.len(), 1, "the tri-pair merges");
        assert_eq!(result.unmerged, vec![lone], "the lone triangle survives un-merged");
        assert_eq!(mesh.totface(), 2, "merged quad + surviving triangle");
        assert_eq!(mesh.face_len(lone), 3);
        mesh.validate_topology();
    }

    #[test]
    fn quad_is_flipped_rejects_bowtie_accepts_convex() {
        // Corners arrive in the §3.3 winding `start, opposite-apex, end,
        // own-apex`. A convex quad has the two apexes on opposite sides of the
        // shared-edge line; a bow-tie has them on the same side.
        let convex = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, -1.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
        ];
        assert!(
            !super::join_triangles_quad_is_flipped(&convex),
            "a convex quad is not flipped"
        );

        // Same shared edge, but both apexes above it -> the two source
        // triangles overlap -> bow-tie.
        let bowtie = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::new(1.0, 2.0, 0.0),
        ];
        assert!(
            super::join_triangles_quad_is_flipped(&bowtie),
            "a same-side bow-tie quad is flipped"
        );

        // A concave (arrowhead) but simple quad keeps the apexes on opposite
        // sides, so it must not be rejected as flipped.
        let concave = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, -1.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::new(1.0, 0.2, 0.0),
        ];
        assert!(
            !super::join_triangles_quad_is_flipped(&concave),
            "a concave simple quad is not flipped"
        );
    }

    #[test]
    fn tri_strip_greedy_merges_squares_first() {
        // A strip of four triangles forming two unit squares:
        //   square A: verts 0,1,4,3   split by diagonal 1-3 → tris (0,1,3),(1,4,3)
        //   square B: verts 1,2,5,4   split by diagonal 2-4 → tris (1,2,4),(2,5,4)
        // The two per-square diagonals are well-shaped (square) merges; the
        // shared vertical 1-4 between the squares is contested. Greedy
        // best-quad-first merges each square's diagonal and leaves the vertical.
        let mut mesh = LoopMesh::new();
        let v: Vec<VertexKey> = [
            (0.0, 0.0),
            (1.0, 0.0),
            (2.0, 0.0),
            (0.0, 1.0),
            (1.0, 1.0),
            (2.0, 1.0),
        ]
        .iter()
        .map(|&(x, y)| mesh.create_vertex(Vec3::new(x, y, 0.0)))
        .collect();

        let t0 = build_tri_from_verts(&mut mesh, [v[0], v[1], v[3]]);
        let t1 = build_tri_from_verts(&mut mesh, [v[1], v[4], v[3]]);
        let t2 = build_tri_from_verts(&mut mesh, [v[1], v[2], v[4]]);
        let t3 = build_tri_from_verts(&mut mesh, [v[2], v[5], v[4]]);

        assert_eq!((mesh.totvert(), mesh.totedge(), mesh.totface()), (6, 9, 4));

        let result = mesh.join_triangles(&[t0, t1, t2, t3], JoinTrianglesParams::default());

        assert_eq!(mesh.totface(), 2, "two quads form");
        assert_eq!(result.joined.len(), 2);
        for &quad in &result.joined {
            assert_eq!(mesh.face_len(quad), 4);
        }
        // 6 V, 9 E, 4 F → −2 E, −2 F: 6 V, 7 E, 2 F.
        assert_eq!(mesh.totvert(), 6);
        assert_eq!(mesh.totedge(), 7);
        mesh.validate_topology();
    }

    #[test]
    fn non_triangle_input_is_ignored() {
        // A tri-pair plus an already-quad face sharing no mergeable edge with
        // the tris: only the tri-pair merges; the quad is untouched.
        let mut mesh = LoopMesh::new();
        let (lower, upper, _) = two_tris_shared_edge(&mut mesh);

        // A standalone quad off to the side.
        let q0 = mesh.create_vertex(Vec3::new(3.0, 0.0, 0.0));
        let q1 = mesh.create_vertex(Vec3::new(4.0, 0.0, 0.0));
        let q2 = mesh.create_vertex(Vec3::new(4.0, 1.0, 0.0));
        let q3 = mesh.create_vertex(Vec3::new(3.0, 1.0, 0.0));
        let eq0 = mesh.create_edge([q0, q1], EdgeCreateFlags::NO_DUPLICATE);
        let eq1 = mesh.create_edge([q1, q2], EdgeCreateFlags::NO_DUPLICATE);
        let eq2 = mesh.create_edge([q2, q3], EdgeCreateFlags::NO_DUPLICATE);
        let eq3 = mesh.create_edge([q3, q0], EdgeCreateFlags::NO_DUPLICATE);
        let quad = mesh.face_create(&[q0, q1, q2, q3], &[eq0, eq1, eq2, eq3], FaceCreateFlags::NONE);

        let result = mesh.join_triangles(&[lower, upper, quad], JoinTrianglesParams::default());

        assert_eq!(result.joined.len(), 1, "only the tri-pair merges");
        assert_eq!(mesh.totface(), 2, "merged quad + the untouched input quad");
        assert!(mesh.faces.try_get(quad).is_some(), "the input quad survives");
        assert_eq!(mesh.face_len(quad), 4);
        mesh.validate_topology();
    }

    #[test]
    fn angle_face_threshold_blocks_and_allows_fold() {
        // Two triangles folded 90° across their shared edge 0-2. The shared
        // edge runs along x=... ; fold the upper triangle up out of the plane.
        let build = |mesh: &mut LoopMesh| -> (FaceKey, FaceKey) {
            let v0 = mesh.create_vertex(Vec3::new(0.0, 0.0, 0.0));
            let v1 = mesh.create_vertex(Vec3::new(1.0, 0.0, 0.0));
            let v2 = mesh.create_vertex(Vec3::new(1.0, 1.0, 0.0));
            // Upper apex folded up: same footprint but z != 0 so the dihedral
            // across edge 0-2 is 90°.
            let v3 = mesh.create_vertex(Vec3::new(0.0, 0.0, 1.0));
            let lower = build_tri_from_verts(mesh, [v0, v1, v2]);
            let upper = build_tri_from_verts(mesh, [v0, v2, v3]);
            (lower, upper)
        };

        // Tight threshold blocks the fold.
        let mut blocked = LoopMesh::new();
        let (l, u) = build(&mut blocked);
        let blocked_result = blocked.join_triangles(
            &[l, u],
            JoinTrianglesParams {
                angle_face_threshold: 0.1,
                ..Default::default()
            },
        );
        assert!(blocked_result.joined.is_empty(), "tight fold gate blocks merge");
        assert_eq!(blocked.totface(), 2, "both triangles survive");
        blocked.validate_topology();

        // Disabled threshold (π) permits the fold.
        let mut allowed = LoopMesh::new();
        let (l, u) = build(&mut allowed);
        let allowed_result = allowed.join_triangles(&[l, u], JoinTrianglesParams::default());
        assert_eq!(allowed_result.joined.len(), 1, "disabled fold gate merges");
        assert_eq!(allowed.totface(), 1);
        allowed.validate_topology();
    }

    #[test]
    fn angle_shape_threshold_blocks_and_allows_sliver() {
        // A long thin lozenge: corners deviate far from right angles, so a
        // tight shape gate rejects the merge but the disabled gate allows it.
        let build = |mesh: &mut LoopMesh| -> (FaceKey, FaceKey) {
            let v0 = mesh.create_vertex(Vec3::new(0.0, 0.0, 0.0));
            let v1 = mesh.create_vertex(Vec3::new(4.0, 0.2, 0.0));
            let v2 = mesh.create_vertex(Vec3::new(8.0, 0.0, 0.0));
            let v3 = mesh.create_vertex(Vec3::new(4.0, -0.2, 0.0));
            // Quad 0-1-2-3 split along diagonal 0-2.
            let lower = build_tri_from_verts(mesh, [v0, v1, v2]);
            let upper = build_tri_from_verts(mesh, [v0, v2, v3]);
            (lower, upper)
        };

        let mut blocked = LoopMesh::new();
        let (l, u) = build(&mut blocked);
        let blocked_result = blocked.join_triangles(
            &[l, u],
            JoinTrianglesParams {
                angle_shape_threshold: 0.1,
                ..Default::default()
            },
        );
        assert!(blocked_result.joined.is_empty(), "tight shape gate rejects sliver");
        assert_eq!(blocked.totface(), 2);
        blocked.validate_topology();

        let mut allowed = LoopMesh::new();
        let (l, u) = build(&mut allowed);
        let allowed_result = allowed.join_triangles(&[l, u], JoinTrianglesParams::default());
        assert_eq!(allowed_result.joined.len(), 1, "disabled shape gate merges sliver");
        allowed.validate_topology();
    }

    #[test]
    fn cmp_uvs_blocks_uv_island_boundary() {
        // A tri-pair whose shared diagonal is a UV-island seam: the per-loop
        // UVs disagree across the edge. cmp_uvs gates it out; off it merges.
        let layer_name = "uv";
        let build = |mesh: &mut LoopMesh| -> (FaceKey, FaceKey) {
            let (lower, upper, _) = two_tris_shared_edge(mesh);
            (lower, upper)
        };

        let mut blocked = LoopMesh::new();
        let uv = register_uv_layer(&mut blocked, layer_name);
        let (l, u) = build(&mut blocked);
        // Give every loop a UV; make the two faces disagree at the shared
        // diagonal endpoints so the edge is an island boundary.
        set_loop_uvs_island_split(&mut blocked, l, u, uv);
        let blocked_result = blocked.join_triangles(
            &[l, u],
            JoinTrianglesParams {
                cmp_uvs: true,
                ..Default::default()
            },
        );
        assert!(blocked_result.joined.is_empty(), "cmp_uvs gates the UV seam");
        assert_eq!(blocked.totface(), 2);
        blocked.validate_topology();

        let mut allowed = LoopMesh::new();
        let uv = register_uv_layer(&mut allowed, layer_name);
        let (l, u) = build(&mut allowed);
        set_loop_uvs_island_split(&mut allowed, l, u, uv);
        let allowed_result = allowed.join_triangles(&[l, u], JoinTrianglesParams::default());
        assert_eq!(allowed_result.joined.len(), 1, "cmp_uvs off merges across the seam");
        allowed.validate_topology();
    }

    #[test]
    fn cmp_vcols_blocks_colour_island_boundary() {
        let layer_name = "col";
        let build = |mesh: &mut LoopMesh| -> (FaceKey, FaceKey) {
            let (lower, upper, _) = two_tris_shared_edge(mesh);
            (lower, upper)
        };

        let mut blocked = LoopMesh::new();
        let col = register_col_layer(&mut blocked, layer_name);
        let (l, u) = build(&mut blocked);
        set_loop_cols_island_split(&mut blocked, l, u, col);
        let blocked_result = blocked.join_triangles(
            &[l, u],
            JoinTrianglesParams {
                cmp_vcols: true,
                ..Default::default()
            },
        );
        assert!(blocked_result.joined.is_empty(), "cmp_vcols gates the colour seam");
        assert_eq!(blocked.totface(), 2);
        blocked.validate_topology();

        let mut allowed = LoopMesh::new();
        let col = register_col_layer(&mut allowed, layer_name);
        let (l, u) = build(&mut allowed);
        set_loop_cols_island_split(&mut allowed, l, u, col);
        let allowed_result = allowed.join_triangles(&[l, u], JoinTrianglesParams::default());
        assert_eq!(allowed_result.joined.len(), 1, "cmp_vcols off merges across the seam");
        allowed.validate_topology();
    }

    /// Build the §6.4 contested-influence fixture: a pre-existing quad `Q`
    /// plus a contested apex triangle `Ta` whose two candidate partners (`Tc`
    /// on the grid-aligned left, `Tb` on the unaligned right) build *congruent*
    /// quads (equal §5.1 error). Returns `(faces, [Q, Tc, Ta, Tb])`.
    ///
    /// Layout (XY plane):
    /// ```text
    ///   3 ------- 4 ------------ 5
    ///   |    Q    |\    Ta      /|
    ///   |         | \    ___---/ |
    ///   0 ------- 1 -- 2 ------- 6
    /// ```
    /// `Q = [0,1,4,3]`, `Tc = [1,2,4]`, `Ta = [2,5,4]`, `Tb = [2,6,5]`.
    /// Candidate edges: `{2,4}` (Tc|Ta) and `{2,5}` (Ta|Tb).
    ///
    /// The shared `{2,5}` edge is created *before* `{2,4}` so that, at
    /// `topology_influence == 0`, the equal-error tie is broken by insertion
    /// order toward the unaligned `{2,5}` merge (consuming Ta+Tb, leaving Tc).
    /// The influence then has to flip the tie the *other* way to be observable.
    fn build_contested_influence(mesh: &mut LoopMesh) -> (Vec<FaceKey>, [FaceKey; 4]) {
        let p = |x: f32, y: f32| Vec3::new(x, y, 0.0);
        let v: Vec<VertexKey> = [
            p(0.0, 0.0), // 0
            p(1.0, 0.0), // 1
            p(2.0, 0.0), // 2
            p(0.0, 1.0), // 3
            p(1.0, 1.0), // 4
            p(3.0, 1.0), // 5
            p(3.0, 0.0), // 6
        ]
        .iter()
        .map(|&pos| mesh.create_vertex(pos))
        .collect();

        // Pre-create the two contested edges in the order that gives the
        // unaligned {2,5} merge the earlier insertion tiebreak at influence 0.
        mesh.create_edge([v[2], v[5]], EdgeCreateFlags::NO_DUPLICATE);
        mesh.create_edge([v[2], v[4]], EdgeCreateFlags::NO_DUPLICATE);

        let make_face = |mesh: &mut LoopMesh, idx: &[usize]| -> FaceKey {
            let verts: Vec<VertexKey> = idx.iter().map(|&i| v[i]).collect();
            let edges: Vec<EdgeKey> = (0..verts.len())
                .map(|i| {
                    mesh.create_edge(
                        [verts[i], verts[(i + 1) % verts.len()]],
                        EdgeCreateFlags::NO_DUPLICATE,
                    )
                })
                .collect();
            mesh.face_create(&verts, &edges, FaceCreateFlags::NONE)
        };

        let q = make_face(mesh, &[0, 1, 4, 3]);
        let tc = make_face(mesh, &[1, 2, 4]);
        let ta = make_face(mesh, &[2, 5, 4]);
        let tb = make_face(mesh, &[2, 6, 5]);
        (vec![q, tc, ta, tb], [q, tc, ta, tb])
    }

    /// The §6.4 tie-break flip: at `topology_influence == 0` the congruent-quad
    /// tie is decided by insertion order and one triangle survives; at
    /// `topology_influence == 2` the pre-existing quad `Q`'s grid-pull discounts
    /// the aligned candidate below the tie and a *different* triangle survives.
    #[test]
    fn topology_influence_flips_contested_tie_break() {
        // Influence 0 — pure §5.1 order; the insertion-order tiebreak merges the
        // unaligned {2,5} pair (Ta+Tb), leaving Tc = [1,2,4].
        let mut zero = LoopMesh::new();
        let (faces, [_q, tc, _ta, _tb]) = build_contested_influence(&mut zero);
        let zero_result = zero.join_triangles(&faces, JoinTrianglesParams::default());

        assert_eq!((zero.totvert(), zero.totedge(), zero.totface()), (7, 9, 3));
        assert_eq!(zero_result.joined.len(), 1, "one tri-pair merges at influence 0");
        assert_eq!(zero_result.unmerged, vec![tc], "Tc survives at influence 0");
        assert!(zero.faces.try_get(tc).is_some());
        assert_eq!(zero.face_len(tc), 3);
        zero.validate_topology();

        // Influence 2 — Q discounts its grid-continuing neighbour candidate
        // {2,4} below the tie; it now pops first, merges Tc+Ta, leaving Tb.
        let mut two = LoopMesh::new();
        let (faces, [_q, tc2, _ta2, tb2]) = build_contested_influence(&mut two);
        let two_result = two.join_triangles(
            &faces,
            JoinTrianglesParams {
                topology_influence: 2.0,
                ..Default::default()
            },
        );

        assert_eq!((two.totvert(), two.totedge(), two.totface()), (7, 9, 3));
        assert_eq!(two_result.joined.len(), 1, "one tri-pair merges at influence 2");
        assert_eq!(two_result.unmerged, vec![tb2], "Tb survives at influence 2");
        assert!(two.faces.try_get(tb2).is_some());
        assert_eq!(two.face_len(tb2), 3, "Tb is still a triangle at influence 2");
        let _ = tc2;
        two.validate_topology();
    }

    /// Build the contested-influence fixture but with the candidate-side
    /// geometry **folded 90° about the shared `{1,4}` edge** (the line `x = 1`):
    /// the region `x > 1` is rotated into `z`, so verts 2,5,6 leave the `Q`
    /// plane. `Q` stays flat; the fold is exactly the natural fold across the
    /// shared edge that §6.3 says alignment must *not* penalise.
    ///
    /// This pins the §6.3 structural contract two ways: alignment is a quad-vs-
    /// quad measure (it consults the now-out-of-plane candidate quad against the
    /// flat `Q`), and it is fold-invariant (the 90° fold leaves the influence
    /// outcome identical to the flat fixture). A candidate-only far-rail proxy
    /// would have scored the folded candidate's intrinsic rail unchanged and so
    /// could not distinguish this — but neither could it have produced the flat
    /// fixture's correct flip, which the quad-vs-quad measure does.
    fn build_contested_influence_folded(mesh: &mut LoopMesh) -> (Vec<FaceKey>, [FaceKey; 4]) {
        // Fold the candidate plane up about x = 1: (x, y, 0) with x > 1 maps to
        // (1, y, x - 1). Q's verts (x <= 1) stay in the z = 0 plane.
        let fold = |x: f32, y: f32| {
            if x > 1.0 {
                Vec3::new(1.0, y, x - 1.0)
            } else {
                Vec3::new(x, y, 0.0)
            }
        };
        let v: Vec<VertexKey> = [
            (0.0, 0.0), // 0
            (1.0, 0.0), // 1
            (2.0, 0.0), // 2
            (0.0, 1.0), // 3
            (1.0, 1.0), // 4
            (3.0, 1.0), // 5
            (3.0, 0.0), // 6
        ]
        .iter()
        .map(|&(x, y)| mesh.create_vertex(fold(x, y)))
        .collect();

        mesh.create_edge([v[2], v[5]], EdgeCreateFlags::NO_DUPLICATE);
        mesh.create_edge([v[2], v[4]], EdgeCreateFlags::NO_DUPLICATE);

        let make_face = |mesh: &mut LoopMesh, idx: &[usize]| -> FaceKey {
            let verts: Vec<VertexKey> = idx.iter().map(|&i| v[i]).collect();
            let edges: Vec<EdgeKey> = (0..verts.len())
                .map(|i| {
                    mesh.create_edge(
                        [verts[i], verts[(i + 1) % verts.len()]],
                        EdgeCreateFlags::NO_DUPLICATE,
                    )
                })
                .collect();
            mesh.face_create(&verts, &edges, FaceCreateFlags::NONE)
        };

        let q = make_face(mesh, &[0, 1, 4, 3]);
        let tc = make_face(mesh, &[1, 2, 4]);
        let ta = make_face(mesh, &[2, 5, 4]);
        let tb = make_face(mesh, &[2, 6, 5]);
        (vec![q, tc, ta, tb], [q, tc, ta, tb])
    }

    /// §6.3 fold-invariance: folding the candidate side 90° about the shared
    /// edge must leave the influence outcome unchanged. At `topology_influence
    /// == 2` the folded fixture must still flip the contested tie toward the
    /// grid-continuing candidate and leave Tb surviving — exactly as the flat
    /// fixture does — proving alignment compares the two quads' flows in a
    /// common plane rather than penalising the natural fold.
    #[test]
    fn topology_influence_alignment_is_fold_invariant() {
        let mut folded = LoopMesh::new();
        let (faces, [_q, _tc, _ta, tb]) = build_contested_influence_folded(&mut folded);
        let result = folded.join_triangles(
            &faces,
            JoinTrianglesParams {
                topology_influence: 2.0,
                ..Default::default()
            },
        );

        assert_eq!((folded.totvert(), folded.totedge(), folded.totface()), (7, 9, 3));
        assert_eq!(result.joined.len(), 1, "one tri-pair merges across the fold");
        assert_eq!(
            result.unmerged,
            vec![tb],
            "the fold does not change the flip: Tb still survives at influence 2"
        );
        assert_eq!(folded.face_len(tb), 3);
        folded.validate_topology();
    }

    /// The two settings reach the *same* V/E/F counts but a different survivor,
    /// i.e. a non-isomorphic divergence — the cleanest observable signature of
    /// the parameter (§6.4). This is the in-crate analogue of the A/B
    /// `assert_topology_influence_diverges` backstop.
    #[test]
    fn topology_influence_divergence_is_non_isomorphic() {
        let mut zero = LoopMesh::new();
        let (faces, _) = build_contested_influence(&mut zero);
        let zero_survivor = {
            let r = zero.join_triangles(&faces, JoinTrianglesParams::default());
            assert_eq!(r.unmerged.len(), 1);
            zero.face_vert_keys(r.unmerged[0])
        };

        let mut two = LoopMesh::new();
        let (faces, _) = build_contested_influence(&mut two);
        let two_survivor = {
            let r = two.join_triangles(
                &faces,
                JoinTrianglesParams {
                    topology_influence: 2.0,
                    ..Default::default()
                },
            );
            assert_eq!(r.unmerged.len(), 1);
            two.face_vert_keys(r.unmerged[0])
        };

        // Same counts both ways.
        assert_eq!(zero.totvert(), two.totvert());
        assert_eq!(zero.totedge(), two.totedge());
        assert_eq!(zero.totface(), two.totface());
        // Different surviving triangle ⇒ non-isomorphic.
        assert_ne!(
            zero_survivor, two_survivor,
            "the leftover triangle must differ between influence 0 and 2"
        );
    }

    /// `topology_influence` is an *ordering* heuristic: where no triangle is
    /// contested, every candidate merges regardless of order, so a non-zero
    /// influence changes nothing observable. Default (`0`) and `2` agree here.
    #[test]
    fn topology_influence_unobservable_without_contention() {
        let build = |mesh: &mut LoopMesh| {
            let (lower, upper, _) = two_tris_shared_edge(mesh);
            vec![lower, upper]
        };

        let mut zero = LoopMesh::new();
        let faces = build(&mut zero);
        let zero_result = zero.join_triangles(&faces, JoinTrianglesParams::default());

        let mut two = LoopMesh::new();
        let faces = build(&mut two);
        let two_result = two.join_triangles(
            &faces,
            JoinTrianglesParams {
                topology_influence: 2.0,
                ..Default::default()
            },
        );

        assert_eq!(zero_result.joined.len(), two_result.joined.len());
        assert_eq!(zero.totface(), two.totface());
        assert_eq!((zero.totvert(), zero.totedge(), zero.totface()), (4, 4, 1));
        two.validate_topology();
    }

    fn register_uv_layer(mesh: &mut LoopMesh, name: &'static str) -> ElementAttributeKey {
        mesh.register_loop_attribute(ElementAttributeDesc::numeric(
            AttributeKey::custom(name),
            AttrType::Vec2,
        ))
        .expect("uv layer registers")
    }

    fn register_col_layer(mesh: &mut LoopMesh, name: &'static str) -> ElementAttributeKey {
        mesh.register_loop_attribute(ElementAttributeDesc::numeric(
            AttributeKey::custom(name),
            AttrType::Vec4,
        ))
        .expect("colour layer registers")
    }

    /// Set per-loop UVs so the two triangles disagree at the shared-diagonal
    /// endpoints: the lower face's corners get one constant UV, the upper
    /// face's another, so the shared edge is an island boundary.
    fn set_loop_uvs_island_split(
        mesh: &mut LoopMesh,
        lower: FaceKey,
        upper: FaceKey,
        layer: ElementAttributeKey,
    ) {
        for face in [lower, upper] {
            let value = if face == lower { 0.0 } else { 0.5 };
            for loop_key in mesh.face_loop_keys(face) {
                mesh.loop_attribs_mut()
                    .set_vec2(layer, loop_key.as_usize(), [value, value]);
            }
        }
    }

    fn set_loop_cols_island_split(
        mesh: &mut LoopMesh,
        lower: FaceKey,
        upper: FaceKey,
        layer: ElementAttributeKey,
    ) {
        for face in [lower, upper] {
            let value = if face == lower { 0.0 } else { 0.5 };
            for loop_key in mesh.face_loop_keys(face) {
                mesh.loop_attribs_mut().set_vec4(
                    layer,
                    loop_key.as_usize(),
                    [value, value, value, 1.0],
                );
            }
        }
    }
}
