use std::collections::BinaryHeap;
use std::f32::consts::PI;

use crate::loopmesh::*;

/// Float-noise slack: two face normals whose squared difference is within this
/// bound contribute exactly zero to the planarity term, matching the
/// "near-flat quad adds nothing" rule in
/// `radial-mesh-guides/operators/JOIN_TRIANGLES.md` §5.1 term 1.
const NORMAL_EPS: f32 = 0.0001;

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
    /// §6 — grid-flow re-prioritisation strength (0–2). At `0` the merge order
    /// is pure quad-quality. Non-zero is not implemented this cycle.
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
}

impl LoopMesh {
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
    /// loopmesh stores directly. The §4.1 seam / sharp / material gates read
    /// per-element header state loopmesh does not yet expose; their accessors
    /// ([`LoopMesh::join_triangles_edge_is_seam`],
    /// [`LoopMesh::join_triangles_edge_is_sharp`],
    /// [`LoopMesh::join_triangles_face_material`]) currently report "no seam /
    /// smooth / default material", so those gates never fire until that state
    /// lands.
    ///
    /// `topology_influence` (§6) is not implemented; a non-zero value runs the
    /// pure quad-quality order (§5.1) unchanged.
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

        // §3.2 — gather candidate edges into the merge heap.
        let mut heap: BinaryHeap<Candidate> = BinaryHeap::new();
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
                    insertion_order += 1;
                }
            }
        }

        // §5.2 — greedy pop-and-merge.
        let mut joined: Vec<FaceKey> = Vec::new();
        while let Some(candidate) = heap.pop() {
            if let Some(quad) = self.join_triangles_try_merge(candidate.edge) {
                joined.push(quad);
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
        if delimit.cmp_seam && self.join_triangles_edge_is_seam(edge) {
            return true;
        }
        if delimit.cmp_sharp && self.join_triangles_edge_is_sharp(edge) {
            return true;
        }
        if delimit.cmp_materials
            && self.join_triangles_face_material(face_a) != self.join_triangles_face_material(face_b)
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

    /// §4.1 — whether `edge` carries the seam flag. loopmesh has no per-edge
    /// seam header today, so this always reports "no seam"; the `cmp_seam` gate
    /// never fires until a per-edge seam attribute lands.
    fn join_triangles_edge_is_seam(&self, _edge: EdgeKey) -> bool {
        false
    }

    /// §4.1 — whether `edge` is sharp (the smooth bit cleared). loopmesh has no
    /// per-edge smooth header today, so this always reports "smooth" (not
    /// sharp); the `cmp_sharp` gate never fires until that state lands.
    fn join_triangles_edge_is_sharp(&self, _edge: EdgeKey) -> bool {
        false
    }

    /// §4.1 — the face's material index. loopmesh has no per-face material
    /// today, so every face reports `0`; the `cmp_materials` gate never fires
    /// until a per-face material attribute lands.
    fn join_triangles_face_material(&self, _face: FaceKey) -> u32 {
        0
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

    fn register_uv_layer(mesh: &mut LoopMesh, name: &'static str) -> ElementAttributeKey {
        mesh.register_loop_attribute(ElementAttributeDesc::numeric(
            AttributeKey::custom(name),
            Format::new(ComponentType::F32, 2),
        ))
        .expect("uv layer registers")
    }

    fn register_col_layer(mesh: &mut LoopMesh, name: &'static str) -> ElementAttributeKey {
        mesh.register_loop_attribute(ElementAttributeDesc::numeric(
            AttributeKey::custom(name),
            Format::new(ComponentType::F32, 4),
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
