//! The shared slide primitive (BEVEL.md §4.1) and the `loop_slide` placement (§8.8).

use super::*;

impl LoopMesh {
    /// Try to place a sector's boundvert by the `loop_slide` rule (BEVEL.md §8.8):
    /// slide it onto a single unbeveled leg's line instead of the §7.6.2
    /// meet-point. Returns `Some(position)` when the sector fires (§8.8.2) and the
    /// slide is accepted, `None` to fall back to the plain meet-point.
    pub(super) fn slide_boundvert(
        &self,
        vertex: VertexKey,
        vertex_position: Vec3,
        sector: &Sector,
        is_beveled: &dyn Fn(EdgeKey) -> bool,
    ) -> Option<SlideRecord> {
        if sector.bounding_edges.len() != 2 {
            return None;
        }

        // The sector's corners are the ordered (edge, face) stream the disk walk
        // produced, opening on `g_k` and closing on `g_(k+1)` (§7.6.1). Each
        // consecutive pair shares the next entry's edge as the leg between their two
        // faces, so the unbeveled outgoing edges between the two bounding beveled
        // edges *are* the legs, already in disk order — the same representation
        // `solve_meet_point` reads. Take them straight from that stream rather than
        // a face-dedup → shared-edge round trip (which mis-orders if a face recurs
        // non-contiguously).
        let mut legs: Vec<(EdgeKey, FaceKey, FaceKey)> = Vec::new();
        for pair in sector.corners.windows(2) {
            let (leg, face_after) = pair[1];
            let face_before = pair[0].1;
            // `leg` is the closing beveled edge for the final pair and any beveled
            // edge mid-sector; only unbeveled legs slide. A non-shared/degenerate
            // pair is skipped, not aborted: the firing decision must still see the
            // legs that genuinely exist (§8.8.2).
            if is_beveled(leg) {
                continue;
            }
            legs.push((leg, face_before, face_after));
        }
        if legs.is_empty() {
            return None;
        }

        // Classify each leg in-plane vs silhouette by the dot of its two flanking
        // face normals (§8.8.2): coplanar (≈ ±1) is in-plane, else silhouette.
        let silhouette: Vec<&(EdgeKey, FaceKey, FaceKey)> = legs
            .iter()
            .filter(|&&(_, fa, fb)| {
                let dot = self.face_plane_normal(fa).dot(self.face_plane_normal(fb));
                dot.abs() < 1.0 - LOOP_SLIDE_TOLERANCE
            })
            .collect();

        // Firing condition (§8.8.2) part 1: exactly one silhouette leg, or — absent
        // any silhouette leg — exactly one in-plane leg. Anything else falls back.
        let slide_leg = if !silhouette.is_empty() {
            if silhouette.len() != 1 {
                return None;
            }
            silhouette[0].0
        } else {
            if legs.len() != 1 {
                return None;
            }
            legs[0].0
        };

        let leg_direction = self.edge_direction_from(slide_leg, vertex, vertex_position)?;
        let edge_before = sector.bounding_edges[0];
        let edge_after = sector.bounding_edges[1];
        let dir_before = self.edge_direction_from(edge_before, vertex, vertex_position)?;
        let dir_after = self.edge_direction_from(edge_after, vertex, vertex_position)?;

        // Firing condition (§8.8.2.1) part 2 — the "good offset" guard, taken in the
        // genuine junction *vertex-normal* frame `n_v`, never a sector face-plane
        // frame. Both sub-meets `g_k → e_mid` and `e_mid → g_(k+1)` must sweep
        // *forward* and open a non-degenerate angle; a reflex sub-meet would crease
        // the surface backward, so the slide is rejected (return `None`). Forward vs
        // reflex is the sign of the swept angle viewed from `n_v` (`cross · n_v < 0`
        // is reflex); `n_v` is the incident-face-average vertex normal (vertex-normal
        // fallback) shared with §7.5.1/§8.8.3. A *sector face-plane* normal tilts away
        // from the true corner on an irregular junction's off-plane sectors and flips
        // this sign, wrongly rejecting the very sectors that should slide (the
        // under-fire); `n_v` is sign-stable across every sector of the junction.
        // Degeneracy is measured on the genuine un-projected opening angle (within
        // ~0.1 rad of 0 or π), separately from the sign.
        let vertex_normal = self.surface_normal_at(vertex);
        if !subsweep_is_good(dir_before, leg_direction, vertex_normal)
            || !subsweep_is_good(leg_direction, dir_after, vertex_normal)
        {
            return None;
        }

        // The two sub-meets sit `offset / sin(α)` along the leg from `v`, where α is
        // the (true) opening angle of the flanking beveled edge against the leg
        // (§8.8.3). The gate above guarantees both are forward and non-degenerate, so
        // §8.8.3's single-survivor (one sub-meet reflex) and both-degenerate (all
        // three lines collinear) fallback branches are intentionally not built here.
        // The position itself is not computed in this construct pass: it is re-derived
        // by the caller's §7.6.4.5 rebuild off the reconciled offsets (§8.8.3.1), where
        // the reconciled pair makes the two sub-meets coincide — so only the slide
        // parameters (`d_leg`, `sin α₁`, the flank-sine ratio) are returned.
        let sin_alpha_before = dir_before.cross(leg_direction).length();
        let sin_alpha_after = dir_after.cross(leg_direction).length();

        // §8.8.4: record the flank-sine ratio `sin(α₂)/sin(α₁)` — a pure measure of
        // how lopsided the two flanking angles are — for the §7.6.4 width pass to
        // thread into the far-end offsets. The explicit guard: if either sine
        // collapses (a flank gone collinear with the leg) the ratio diverges, so
        // override it to `1` rather than evaluate the blow-up — an unconstrained side
        // makes equal demands on the leg, so a balanced ratio adds no bias.
        let flank_sine_ratio =
            if sin_alpha_before < LOOP_SLIDE_TOLERANCE || sin_alpha_after < LOOP_SLIDE_TOLERANCE {
                1.0
            } else {
                sin_alpha_after / sin_alpha_before
            };

        Some(SlideRecord {
            flank_sine_ratio,
            leg_direction,
            sin_alpha_before,
        })
    }

    /// [`slide_along_edge`] addressed by an incident edge rather than by a far
    /// position: place a vertex `distance` from `vertex` along `edge`, carrying
    /// §4.1's overshoot guard.
    ///
    /// Every on-edge slide in the edge-bevel builders goes through here. The two
    /// **line solves** (the terminal leg meet of §7.5.1 and the junction meet of
    /// §7.6.2) deliberately do not: they take a distance along no edge, so there
    /// is nothing to hand this, and their overshoot is §7.5.2's span clamp — which
    /// snaps *onto* the endpoint where this lands short of it. Their shared
    /// near-collinear ladder is the exception that proves it: its doubled-back branch
    /// solves no lines and *is* a plain slide, so it comes here (at a terminal, via
    /// `slide_along_edge` directly) and lands short like any other.
    pub(super) fn slide_along_incident_edge(
        &self,
        vertex: VertexKey,
        edge: EdgeKey,
        distance: f32,
    ) -> Vec3 {
        slide_along_edge(
            self.vertex_position(vertex),
            self.edge_far_position(edge, vertex),
            distance,
        )
    }

    /// The position of `edge`'s **other** endpoint, seen from `vertex` — the `far` every
    /// §4.1 slide is measured against.
    pub(super) fn edge_far_position(&self, edge: EdgeKey, vertex: VertexKey) -> Vec3 {
        let [a, b] = self.edge_vert_keys(edge);
        self.vertex_position(if a == vertex { b } else { a })
    }

    /// The unit direction of `edge` pointing away from `vertex`, or `None` for a
    /// degenerate (zero-length) edge.
    ///
    /// Unitised by §7.6.2's shared `normalise` arithmetic — one `sqrt`, one scalar
    /// reciprocal, three multiplies. These directions reach the offset lines through
    /// both sector-normal recipes (the case-1/2 edge span, the case-3 corner crosses),
    /// so the per-component-division spelling would leave the guide's normalisation
    /// axis half-migrated one frame above the base points.
    pub(super) fn edge_direction_from(
        &self,
        edge: EdgeKey,
        vertex: VertexKey,
        vertex_position: Vec3,
    ) -> Option<Vec3> {
        let [a, b] = self.edge_vert_keys(edge);
        let far = if a == vertex { b } else { a };
        let segment = self.vertex_position(far) - vertex_position;
        let length = segment.length();
        (length > f32::EPSILON).then(|| segment * (1.0 / length))
    }
}

/// The slide parameters of a firing `loop_slide` sector (BEVEL.md §8.8.3 / §8.8.4),
/// returned by `slide_boundvert` so the caller can run the §7.6.4.5 *rebuild* pass:
/// re-derive both the apex boundvert (§8.8.3.1) and the flanking terminal offsets
/// (§7.6.4.3) from the reconciled offsets, rather than placing the apex once off the
/// spec offset. `leg_direction` and `sin_alpha_before` are the un-projected leg
/// direction `d_leg` and `sin α₁` of §8.8.3 — the apex rebuild lands at
/// `v + (o₁ / sin α₁) · d_leg`, where the reconciled pair makes the two sub-meets
/// coincide.
pub(super) struct SlideRecord {
    pub(super) flank_sine_ratio: f32,
    pub(super) leg_direction: Vec3,
    pub(super) sin_alpha_before: f32,
}

/// The absolute world-unit gap the §4.1 overshoot guard leaves between a clamped
/// slide and the edge's far endpoint. Never scaled by the edge length, the
/// offset, or the model scale — the landing point is position-comparable, so a
/// relative epsilon here is a divergence.
pub(super) const SLIDE_CLAMP_GAP: f32 = 5.0e-5;

/// Floor of the slide direction's normalisation, stated on the **squared** length
/// (BEVEL.md §4.1). At or below it the two endpoints are the same `f32` point and
/// there is no direction to slide along, so the placement collapses onto `v`
/// rather than emitting a non-finite coordinate.
pub(super) const SLIDE_DIRECTION_FLOOR_SQ: f32 = 1.0e-35;

/// BEVEL.md §7.6.2's document-wide `normalise`: the unit vector, or the **zero**
/// vector at or below the shared direction floor. Never a non-finite coordinate, and
/// never a substituted axis — a caller that scales this by a distance therefore
/// leaves its base point exactly where it was, which is the derived (not stipulated)
/// answer for a degenerate input on both halves of the near-collinear ladder.
///
/// The floor is [`SLIDE_DIRECTION_FLOOR_SQ`], stated on the squared length, and it is
/// the same constant §4.1 states for the slide direction. Every §7.6.2 normalisation —
/// both offset-line perpendiculars in each of the three sector cases, and the ladder's
/// straight-through push — routes through here, so the floor is stated once.
///
/// The arithmetic is stated, not incidental (§7.6.2 "The arithmetic, not just the
/// floor"): the guarded quantity is the **squared** length, the test is a **strict**
/// `>` so a squared length exactly on the floor takes the zero branch, and above the
/// floor the unit vector is a **scaling** — one `sqrt`, one scalar reciprocal,
/// multiplied into all three components. Spelling it as three per-component divisions
/// is a different computation (one correctly rounded division each, against one
/// rounding for the reciprocal plus one per component), and although the two agree to
/// within an ulp per component, §7.6.2's offset lines feed a solve that amplifies that
/// ulp into a whole canonical quantum on 2.5% of shallow planar sectors.
pub(super) fn normalise_at_direction_floor(vector: Vec3) -> Vec3 {
    let length_squared = vector.length_squared();
    if length_squared > SLIDE_DIRECTION_FLOOR_SQ {
        vector * (1.0 / length_squared.sqrt())
    } else {
        Vec3::ZERO
    }
}

/// The one shared slide primitive (BEVEL.md §4.1): place a vertex `distance` from
/// `vertex` along the edge running to `far`.
///
/// Every placement in this operator phrased as "put a vertex a distance `d` from
/// `v` along edge `e`" routes through here, so the guard lives in one place
/// instead of on the branch that happens to reach it first. The guard is
/// **profile-blind**: the concave / custom `√2` leg lengthening (§8.3.0.1) is only
/// one of the routes that can hand it an overshooting distance — the `WIDTH` /
/// `DEPTH` re-measurement of §4.3 reaches it from a modest offset on any profile,
/// as do `PERCENT > 100` and an `OFFSET` longer than the edge.
///
/// Three behaviours are separately observable and all fall out of the shape below:
///
/// - The trigger is a **strict** `>`: a distance that resolves to exactly the edge
///   length is left alone and lands the vertex precisely on `far`.
/// - A clamped vertex lands [`SLIDE_CLAMP_GAP`] **short** of `far`, whatever the
///   overshoot was — the landing point is constant past the knee. This is *not*
///   the §7.5.2 span clamp, which snaps a *meet* exactly onto the endpoint with no
///   gap; the two guards' landing points differ by exactly that gap.
/// - On an edge shorter than the gap but longer than the direction floor the
///   clamped distance goes negative and the vertex is **reflected** behind
///   `vertex`; at or below the direction floor it lands on `vertex` itself.
pub(super) fn slide_along_edge(vertex: Vec3, far: Vec3, distance: f32) -> Vec3 {
    let toward_far = far - vertex;
    let length_squared = toward_far.length_squared();
    if length_squared <= SLIDE_DIRECTION_FLOOR_SQ {
        return vertex;
    }
    let edge_length = length_squared.sqrt();
    let clamped = if distance > edge_length {
        edge_length - SLIDE_CLAMP_GAP
    } else {
        distance
    };
    // Reciprocal scaling then the step, matching the shared normalisation's arithmetic
    // (§4.1's primitive, §7.6.2's box) rather than a per-component division.
    vertex + toward_far * (1.0 / edge_length) * clamped
}

/// Coincidence / co-linearity tolerance for the `loop_slide` placement
/// (BEVEL.md §8.8.2 / §8.8.3). The guide is silent on the numeric value; this
/// reuses the same `1e-4` length tolerance `corner_verts_coincide` already
/// applies to position coincidence in this file, so the silhouette classifier's
/// normal-dot test and the sub-meet `sin α` degeneracy test share one
/// crate-local convention rather than introducing a fresh magic number.
pub(super) const LOOP_SLIDE_TOLERANCE: f32 = 1.0e-4;

/// Angular band (radians) within which a §8.8.2.1 sub-meet's un-projected opening
/// angle is rejected as degenerate — within ~0.1 rad of either `0` or `π`. This is
/// the *magnitude* half of the "good offset" guard (the *sign* half is the
/// vertex-normal-frame forward/reflex test); a leg collinear with a flanking
/// beveled edge (angle near 0 or π) carries no usable slide direction.
pub(super) const LOOP_SLIDE_ANGLE_BAND: f32 = 0.1;

/// The §8.8.2.1 "good offset" sub-meet guard, evaluated in the genuine junction
/// vertex-normal frame `n_v`. The swept angle from `dir_first` to `dir_second`
/// must be **forward** — viewed from `n_v` it turns positively, i.e.
/// `cross(dir_first, dir_second) · n_v > 0`; a reflex (`< 0`) sweep folds the
/// surface backward — **and** **non-degenerate** — its genuine un-projected
/// opening angle in `[0, π]` must lie outside the `~0.1 rad` band of both `0` and
/// `π`. Sign and magnitude are read separately (§8.8.2.1): the sign from the
/// `n_v` projection, the magnitude from the un-projected `atan2(|cross|, dot)`.
pub(super) fn subsweep_is_good(dir_first: Vec3, dir_second: Vec3, vertex_normal: Vec3) -> bool {
    let cross = dir_first.cross(dir_second);
    if cross.dot(vertex_normal) <= 0.0 {
        return false;
    }
    let opening_angle = cross.length().atan2(dir_first.dot(dir_second));
    opening_angle > LOOP_SLIDE_ANGLE_BAND
        && opening_angle < std::f32::consts::PI - LOOP_SLIDE_ANGLE_BAND
}
