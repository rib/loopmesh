//! The mesh-free terminal leg meet (BEVEL.md §7.5.1).

use super::*;

/// The §7.5.1 **direction** angle of a terminal leg meet: the angle between the two
/// edge directions *as the meet binds them* — the pair's **first** member (in the
/// endpoint's rotational order) taken **into** the endpoint, its **second** taken
/// **out** of it. A small value therefore means the two edges run nearly straight
/// *through* the endpoint; a value near `π` means they double back on each other.
///
/// This is **not** the wedge ("opening") angle the two outward directions enclose —
/// the two are supplements, and every §7.5.1 threshold is stated on this one. Reading
/// a threshold on the wedge inverts the dispatch: it would fire the shallow branch on
/// a razor-pinched corner instead of a near-straight one.
///
/// `leg_precedes_edge` decides which member is first, and the returned value is
/// *invariant* under it — swapping the pair negates both directions. Keeping the
/// parameter states §7.5.1's binding literally, so the invariance is a checkable
/// property (see the unit test) rather than an unstated assumption baked into a
/// one-sided formula.
pub(super) fn terminal_direction_angle(
    edge_out: Vec3,
    leg_out: Vec3,
    leg_precedes_edge: bool,
) -> f32 {
    let (first_dir, second_dir) = terminal_meet_directions(edge_out, leg_out, leg_precedes_edge);
    unit_chord_angle_between(first_dir, second_dir)
}

/// §7.5.1's binding of the pair's two directions: the **first** member of the pair (in
/// the endpoint's rotational order) taken *into* the endpoint, the **second** taken
/// *out* of it. Every §7.5.1 quantity stated on "first"/"second" — the direction angle,
/// the offset-line perpendiculars, regime 1's common direction and its slide edge — is
/// read off this pair, so it is bound once here.
pub(super) fn terminal_meet_directions(
    edge_out: Vec3,
    leg_out: Vec3,
    leg_precedes_edge: bool,
) -> (Vec3, Vec3) {
    if leg_precedes_edge {
        (-leg_out, edge_out)
    } else {
        (-edge_out, leg_out)
    }
}

/// §7.5.1 regime 2's angular band: at least `EPS_ANG` (outside the near-degenerate
/// band, which has no meet plane at all) and strictly under `SMALL_ANG`.
pub(super) fn is_shallow_direction_angle(direction_angle: f32) -> bool {
    (MEET_DIRECTION_EPS_ANGLE..TERMINAL_DIRECTION_SMALL_ANGLE).contains(&direction_angle)
}

/// The §7.5.1 per-sector orientation input, in the one shape that cannot state the
/// normal and the reference inconsistently: with a usable sector face plane the two
/// *are* the same vector, and without one there is no candidate plane normal at all —
/// only a reference for the sign test.
///
/// `NoFaceNormal` is §7.5.1's genuinely face-less sector, and only that. A sector face
/// whose plane normal is **degenerate** does *not* come here: source 1 takes the face's
/// normal directly, zero and all, so it stays a `Face` — the fallback is keyed on the
/// sector having no face, not on the normal being usable. Nothing reaches this variant
/// from `terminal_strip_vert`, whose `face` argument is always one of `E`'s two radial
/// faces.
#[derive(Clone, Copy)]
pub(super) enum SectorPlane {
    Face(Vec3),
    NoFaceNormal { reference: Vec3 },
}

impl SectorPlane {
    /// The vector the §7.5.1 regime-3 sign test is run against.
    pub(super) fn reference(self) -> Vec3 {
        match self {
            SectorPlane::Face(normal) => normal,
            SectorPlane::NoFaceNormal { reference } => reference,
        }
    }
}

/// The mesh-free inputs to one §7.5.1 terminal leg meet, read off the pristine mesh at
/// the endpoint.
pub(super) struct TerminalLegMeet {
    pub(super) endpoint: Vec3,
    /// `E`'s far endpoint, and the leg's. Everything §7.5.1 needs of the two flanks is
    /// read off these: the meet's two **raw** directions (`far − endpoint`, the
    /// full-length vectors §7.5.1's pseudocode binds), and regime 1's doubled-back
    /// slide, which is a §4.1 slide like any other and so is clamped against the
    /// length of whichever of the two it runs along.
    pub(super) edge_far: Vec3,
    pub(super) leg_far: Vec3,
    pub(super) leg_precedes_edge: bool,
    pub(super) sector_plane: SectorPlane,
    pub(super) offset: f32,
}

/// Where a §7.5.1 terminal placement came from, which is what decides whether the
/// §7.5.2 span clamp may touch it.
///
/// §7.5.1 returns regime 1's closed forms **before** reaching the clamp, and that
/// ordering is load-bearing rather than incidental. Regime 1's perpendicular push is
/// square to the *common* direction, so it leans `θ_dir/2` off square to each edge and
/// its along-leg projection is `−(offset / cos(θ_dir/2)) · sin(θ_dir/2) / |leg|` —
/// negative for every direction angle above zero, on either side of a well-wound sector.
/// Clamping it would snap the strip vert bodily onto the endpoint for one of each
/// terminal's two pair orders, which is §7.8.2's mis-wound collapse fired on a mesh that
/// is wound correctly, and the exact opposite of regime 1's purpose (a push of magnitude
/// ≈ `offset`).
#[derive(Clone, Copy)]
pub(super) enum TerminalPlacement {
    /// A line solve against the zero-offset leg's own line: the meet can run off the
    /// leg's span, so §7.5.2 arms.
    LineSolve(Vec3),
    /// Regime 1's closed form, which no clamp may see.
    Degenerate(Vec3),
}

impl TerminalPlacement {
    pub(super) fn position(self) -> Vec3 {
        match self {
            TerminalPlacement::LineSolve(position) | TerminalPlacement::Degenerate(position) => {
                position
            }
        }
    }
}

/// §7.5.1's leg meet, before the §7.5.2 span clamp: the beveled edge's offset line
/// against the zero-offset leg's own line, resolved at their closest approach when the
/// two pass skew.
///
/// Both flanks are carried **raw** — `far − endpoint`, full length — exactly as
/// §7.5.1's own pseudocode binds them (`first_dir = endpoint − first.other_end()`,
/// `second_dir = second.other_end() − endpoint`), and the pair is handed to the solve
/// as two points per line so the round trip through the tip runs here too. There is no
/// terminal-only exemption from §7.6.2's direction or carriage axes: the terminal meet
/// differs from the junction only in where its *plane normal* comes from.
pub(super) fn terminal_leg_meet(meet: TerminalLegMeet) -> TerminalPlacement {
    let TerminalLegMeet {
        endpoint,
        edge_far,
        leg_far,
        leg_precedes_edge,
        sector_plane,
        offset,
    } = meet;

    let edge_raw = edge_far - endpoint;
    let leg_raw = leg_far - endpoint;

    // §7.5.4 zero-direction degenerate guard. A zero-length beveled edge has no direction,
    // so its offset line has zero direction and a base at the endpoint: it is not a line but
    // a single **point** at the endpoint. The meet of a leg line against a point line returns
    // that point, so both boundverts the edge contributes here pin to the endpoint —
    // regardless of the first/second disk-order designation. This must fire *before* the
    // regime dispatch: the direction-angle test, the perpendiculars, and the two-line meet
    // solve below all build an offset line from the edge direction and would normalise a
    // zero vector into NaN (or fall to the regime-1 push, which is for two *well-formed*
    // near-collinear legs, not a directionless edge — §7.5.4 keeps them distinct).
    if edge_raw.length() <= f32::EPSILON {
        return TerminalPlacement::Degenerate(endpoint);
    }

    let direction_angle = terminal_direction_angle(edge_raw, leg_raw, leg_precedes_edge);
    // The pair's far endpoints in the pair's own order. The straight-through push sums
    // the **raw** directions, which telescope to `second_far − first_far`, and the
    // doubled-back slide runs along the pair's **second** member — so that is also the
    // edge §4.1's clamp is measured against: `E` on the (leg, E) side and the leg itself
    // on the (E, leg) side.
    let (first_far, second_far) = if leg_precedes_edge {
        (leg_far, edge_far)
    } else {
        (edge_far, leg_far)
    };

    // §7.5.1 regime 1 (near-degenerate direction angle): no meet plane is well defined,
    // so neither offset line is built and no line solve runs. §7.5.1 returns this ahead
    // of the §7.5.2 clamp, so it is reported as clamp-ineligible — see `TerminalPlacement`.
    if is_degenerate_direction_angle(direction_angle) {
        return TerminalPlacement::Degenerate(degenerate_meet_placement(
            endpoint,
            first_far,
            second_far,
            direction_angle,
            // The sector normal, in regime 1's own role: the plane the perpendicular is
            // built *in*. The face-less fallback is already folded into `reference()`.
            sector_plane.reference(),
            offset,
        ));
    }

    let (dir_first, dir_second) = terminal_meet_directions(edge_raw, leg_raw, leg_precedes_edge);

    // §7.5.1 regime 2 (shallow direction angle): the two edges run near-straight
    // through the endpoint, so their cross product is numerically fragile and can come
    // out almost square to the surface. When the sector has a face, that face's normal
    // *is* the plane normal — no sign test, because nothing ambiguous is left to
    // orient. Without a usable sector face normal this regime is unreachable and the
    // angle falls through to the cross product.
    let local_normal = match sector_plane {
        SectorPlane::Face(normal) if is_shallow_direction_angle(direction_angle) => normal,
        _ => {
            // Regime 3: the LOCAL edge-cross normal, oriented to the surface's outward
            // side — the decisive §7.5.1 departure from §7.3's per-face plane normal.
            // The cross is sign-ambiguous; the reference picks the outward branch. Taken
            // on the bound pair in §7.5.1's own order, `cross(second_dir, first_dir)`,
            // and on the raw directions: the sign the pair's order gives it is undone by
            // the reference test, but the magnitudes are what the normalisation rounds.
            let cross_normal = dir_second.cross(dir_first);
            let length = cross_normal.length();
            if length <= f32::EPSILON {
                // Meant as an unreachable division guard: collinear to floating-point
                // exactness is deep inside regime 1's band, so the dispatch above should
                // already have taken it, and this routes to the same placement so it
                // cannot answer differently.
                //
                // It is not actually unreachable. This is the operator's one surviving
                // hand-rolled `normalise`: the floor is `f32::EPSILON` on the **unsquared**
                // length with the `<=` sense, against `normalise_at_direction_floor`'s
                // `SLIDE_DIRECTION_FLOOR_SQ` on the squared length with `>`. And because
                // `dir_first` / `dir_second` are **raw**, the guard scales with the mesh:
                // at a world scale of `~1e-4` an ordinary right-angle terminal crosses to
                // `~1e-8` and is diverted here with `direction_angle` nowhere near regime 1.
                // Unifying the two spellings is its own change — the sign fix below has to
                // move with the floor, and the degenerate-return contract changes with it.
                return TerminalPlacement::Degenerate(degenerate_meet_placement(
                    endpoint,
                    first_far,
                    second_far,
                    direction_angle,
                    sector_plane.reference(),
                    offset,
                ));
            }
            let unit_normal = cross_normal * (1.0 / length);
            if unit_normal.dot(sector_plane.reference()) < 0.0 {
                -unit_normal
            } else {
                unit_normal
            }
        }
    };

    // Both offset lines get the one uniform §7.6.2 perpendicular,
    // `normalise(direction × plane_normal)`, on the pair's bound directions — the first
    // taken *into* the endpoint, the second *out* of it. The role is what carries the
    // sign, so `E` second in the pair pushes one way and `E` first the other, with no
    // separate negation and no per-leg re-derivation of the normal.
    let perp_first = normalise_at_direction_floor(dir_first.cross(local_normal));
    let perp_second = normalise_at_direction_floor(dir_second.cross(local_normal));

    // Exactly one of the pair carries the offset; the unbeveled leg carries 0, so its
    // base point sits on the endpoint whatever its perpendicular came out to be.
    //
    // At a non-planar terminal the two lines generally pass skew, and §7.5.1's
    // resolution is the closest-approach point on the **first** line — the pair's disk
    // predecessor — never the midpoint of the pair. So the two terminal strip verts are
    // asymmetric by construction: with the leg first the meet rides the leg's own line,
    // while with `E` first it rides `E`'s offset line, laterally `offset` from the leg.
    // Symmetrising this matches on one vert of the two.
    let (first_offset, second_offset) = if leg_precedes_edge {
        (0.0, offset)
    } else {
        (offset, 0.0)
    };
    let line_first = Line {
        point: endpoint + perp_first * first_offset,
        direction: dir_first,
    };
    let line_second = Line {
        point: endpoint + perp_second * second_offset,
        direction: dir_second,
    };
    match closest_approach(&line_first, &line_second) {
        Some((on_first, _)) => TerminalPlacement::LineSolve(on_first),
        // The pair came out (anti)parallel to the line solve despite a direction angle
        // outside regime 1's band — reachable only as floating-point slop right at the
        // band edge, since the two lines' directions *are* the two edge directions.
        // Regime 1's placement is the right answer for a pair with no isolated crossing,
        // and it carries regime 1's clamp exemption with it for the same reason.
        None => TerminalPlacement::Degenerate(degenerate_meet_placement(
            endpoint,
            first_far,
            second_far,
            direction_angle,
            sector_plane.reference(),
            offset,
        )),
    }
}

/// §7.5.1 `SMALL_ANG` — the shallow/ordinary threshold (10°, in radians) on the
/// terminal **direction** angle. Below it (and only with a usable sector face normal)
/// the sector face's normal is the meet plane's normal outright; at or above it the
/// sign-fixed edge-cross product is.
pub(super) const TERMINAL_DIRECTION_SMALL_ANGLE: f32 = 0.174_532_92;
