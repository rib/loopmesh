use super::*;

/// §7.5.1's offset-line meet for a **given** plane normal: push `E`'s line sideways
/// by `offset` square to `E`, intersect it with the zero-offset leg's own line, and
/// resolve a skew pair at the closest-approach point on the **first** line (the disk
/// predecessor: the leg when `leg_precedes_edge`, else `E`). The regime dispatch is
/// *not* transcribed — the whole point is to compare each regime's claimed normal
/// against the placement it must produce.
///
/// **What this discriminates, and what it does not.** The solve below is §7.6.2's
/// stated arithmetic written out longhand, which is now the *same* arithmetic
/// `solve_two_lines` runs — so it is not an independent check of the solve, and
/// nothing here would catch the two lines being carried or associated differently in
/// both places at once. What it does discriminate is the **plane normal**: each
/// caller feeds it a different candidate and asserts which one the implementation
/// picked. The longhand is written out rather than borrowed only so the assertions
/// can be stated at `1e-5` — at these shallow direction angles a `2 × 2` Gram
/// spelling of the same algebra lands a hundred ulps away, which is enough to blur a
/// normal-discriminating assertion at that tolerance.
fn leg_meet_oracle(
    endpoint: Vec3,
    edge_out: Vec3,
    leg_out: Vec3,
    leg_precedes_edge: bool,
    plane_normal: Vec3,
    offset: f32,
) -> Vec3 {
    let mut edge_perp = edge_out.cross(plane_normal).normalize();
    if !leg_precedes_edge {
        edge_perp = -edge_perp;
    }
    let edge_base = endpoint + edge_perp * offset;
    // §7.5.1's binding: the pair's first member taken *into* the endpoint, its
    // second taken *out* of it.
    let ((base_first, step_first), (base_second, step_second)) = if leg_precedes_edge {
        ((endpoint, -leg_out), (edge_base, edge_out))
    } else {
        ((edge_base, -edge_out), (endpoint, leg_out))
    };
    let dir_first = (base_first + step_first) - base_first;
    let dir_second = (base_second + step_second) - base_second;
    let direction_cross = dir_first.cross(dir_second);
    let between_bases = base_second - base_first;
    let coplanarity = between_bases.dot(direction_cross);
    if coplanarity.abs() <= 1.0e-6 {
        let along = between_bases.cross(dir_second).dot(direction_cross)
            / direction_cross.dot(direction_cross);
        return base_first + dir_first * along;
    }
    let gap = direction_cross
        * ((base_first - base_second).dot(direction_cross) / direction_cross.dot(direction_cross));
    let moved_base = base_second + gap;
    let moved_direction = (base_second + step_second + gap) - moved_base;
    let moved_cross = dir_first.cross(moved_direction);
    let along = (moved_base - base_first)
        .cross(moved_direction)
        .dot(moved_cross)
        / moved_cross.dot(moved_cross);
    base_first + dir_first * along
}

/// The retracted rule, kept only so the tests below can show they discriminate
/// against it: the midpoint of the two closest-approach points.
fn leg_meet_midpoint_oracle(
    endpoint: Vec3,
    edge_out: Vec3,
    leg_out: Vec3,
    leg_precedes_edge: bool,
    plane_normal: Vec3,
    offset: f32,
) -> Vec3 {
    let mut edge_perp = edge_out.cross(plane_normal).normalize();
    if !leg_precedes_edge {
        edge_perp = -edge_perp;
    }
    let edge_base = endpoint + edge_perp * offset;
    let separation = edge_base - endpoint;
    let along_edge = edge_out.dot(edge_out);
    let cross_term = edge_out.dot(leg_out);
    let along_leg = leg_out.dot(leg_out);
    let edge_projection = edge_out.dot(separation);
    let leg_projection = leg_out.dot(separation);
    let denominator = along_edge * along_leg - cross_term * cross_term;
    let edge_parameter = (cross_term * leg_projection - along_leg * edge_projection) / denominator;
    let leg_parameter = (along_edge * leg_projection - cross_term * edge_projection) / denominator;
    ((edge_base + edge_out * edge_parameter) + (endpoint + leg_out * leg_parameter)) * 0.5
}

/// A terminal frame with the given §7.5.1 **direction** angle: `E` runs out along
/// `+x`, the leg turns back to within `direction_angle` of straight-through. The
/// sector face's normal is deliberately tilted off the `+z` edge-cross normal, so
/// the two candidate plane normals produce measurably different placements — the
/// non-planar sector that makes regime 2 observable at all.
fn shallow_terminal_frame(direction_angle_deg: f32) -> (Vec3, Vec3, Vec3, Vec3) {
    let angle = direction_angle_deg.to_radians();
    let edge_out = Vec3::new(1.0, 0.0, 0.0);
    let leg_out = Vec3::new(-angle.cos(), angle.sin(), 0.0);
    // The tilt has to be square to `E` to matter at all: `cross(edge_out, n)` is
    // blind to `n`'s component along `E`.
    let sector_normal = Vec3::new(0.0, 0.55, 1.0).normalize();
    (Vec3::ZERO, edge_out, leg_out, sector_normal)
}

fn leg_meet(
    endpoint: Vec3,
    edge_out: Vec3,
    leg_out: Vec3,
    sector_plane: SectorPlane,
    offset: f32,
) -> Vec3 {
    terminal_leg_meet(TerminalLegMeet {
        endpoint,
        edge_far: endpoint + edge_out,
        leg_far: endpoint + leg_out,
        leg_precedes_edge: true,
        sector_plane,
        offset,
    })
    .position()
}

/// §7.5.1's dispatch quantity is the **direction** angle — the pair's first member
/// taken *into* the endpoint, its second taken *out* of it — and not the wedge the
/// two outward directions enclose. The two are supplements, and the value must not
/// depend on which of the pair the rotational order happens to put first.
#[test]
fn terminal_direction_angle_is_the_supplement_of_the_outward_wedge() {
    for direction_angle_deg in [1.0_f32, 5.0, 10.0, 45.0, 175.0] {
        let (_, edge_out, leg_out, _) = shallow_terminal_frame(direction_angle_deg);
        let expected = direction_angle_deg.to_radians();
        let wedge = edge_out.dot(leg_out).clamp(-1.0, 1.0).acos();
        assert!(
            (wedge - (std::f32::consts::PI - expected)).abs() < 1.0e-4,
            "frame built at wedge = 180° − {direction_angle_deg}°, got {}°",
            wedge.to_degrees(),
        );
        for leg_precedes_edge in [true, false] {
            let angle = terminal_direction_angle(edge_out, leg_out, leg_precedes_edge);
            assert!(
                (angle - expected).abs() < 1.0e-4,
                "direction angle at {direction_angle_deg}° (leg first = \
                 {leg_precedes_edge}) came out {}°",
                angle.to_degrees(),
            );
        }
    }
}

/// §7.5.1 regime 2: under `SMALL_ANG = 10°` with a face-bearing sector, the sector
/// face's normal **is** the plane normal — used directly, with no sign test — not a
/// sign-fixed edge-cross product. Observable only because the sector is non-planar.
#[test]
fn terminal_leg_meet_shallow_direction_angle_uses_the_sector_face_normal_directly() {
    let offset = 0.25;
    let (endpoint, edge_out, leg_out, sector_normal) = shallow_terminal_frame(5.0);
    let placement = leg_meet(
        endpoint,
        edge_out,
        leg_out,
        SectorPlane::Face(sector_normal),
        offset,
    );

    let by_face_normal = leg_meet_oracle(endpoint, edge_out, leg_out, true, sector_normal, offset);
    let cross_normal = edge_out.cross(leg_out).normalize();
    let by_cross_normal = leg_meet_oracle(endpoint, edge_out, leg_out, true, cross_normal, offset);

    assert!(
        (placement - by_face_normal).length() < 1.0e-5,
        "shallow branch must place from the sector face's normal, got \
         {placement:?} vs {by_face_normal:?}",
    );
    assert!(
        (by_face_normal - by_cross_normal).length() > 1.0e-3,
        "the fixture must actually discriminate the two normals",
    );
}

/// `terminal_leg_meet`'s contract when it is handed no sector plane normal: "when
/// the sector has no face this regime is unreachable and the angle falls through to
/// regime 3" (§7.5.1). The same shallow angle then takes the sign-fixed edge-cross
/// product, oriented by the fallback reference.
///
/// This pins the *function's* behaviour, not a state the operator can reach today:
/// `terminal_strip_vert` is always handed one of `E`'s radial faces, and a
/// **degenerate** one of those stays a `Face` carrying the zero normal (§7.5.1's
/// source 1 is a direct take) rather than falling through to here.
#[test]
fn terminal_leg_meet_shallow_direction_angle_falls_through_without_a_sector_plane_normal() {
    let offset = 0.25;
    let (endpoint, edge_out, leg_out, vertex_normal) = shallow_terminal_frame(5.0);
    let placement = leg_meet(
        endpoint,
        edge_out,
        leg_out,
        SectorPlane::NoFaceNormal {
            reference: vertex_normal,
        },
        offset,
    );

    let cross_normal = edge_out.cross(leg_out).normalize();
    assert!(
        cross_normal.dot(vertex_normal) > 0.0,
        "the reference must already agree with the cross product's sign here",
    );
    let by_cross_normal = leg_meet_oracle(endpoint, edge_out, leg_out, true, cross_normal, offset);
    let by_reference = leg_meet_oracle(endpoint, edge_out, leg_out, true, vertex_normal, offset);

    assert!(
        (placement - by_cross_normal).length() < 1.0e-5,
        "a face-less shallow sector must fall through to the cross product, got \
         {placement:?} vs {by_cross_normal:?}",
    );
    assert!(
        (placement - by_reference).length() > 1.0e-3,
        "and must not have used the fallback reference as the plane normal",
    );
}

/// §7.5.1 regime 3: at or above `SMALL_ANG` the plane normal is the edge-cross
/// product, sign-fixed against the sector face's normal — the sector face normal is
/// an *orientation reference* here, not the plane normal.
#[test]
fn terminal_leg_meet_ordinary_direction_angle_uses_the_sign_fixed_edge_cross() {
    let offset = 0.25;
    let (endpoint, edge_out, leg_out, sector_normal) = shallow_terminal_frame(35.0);
    let placement = leg_meet(
        endpoint,
        edge_out,
        leg_out,
        SectorPlane::Face(sector_normal),
        offset,
    );

    let cross_normal = edge_out.cross(leg_out).normalize();
    let by_cross_normal = leg_meet_oracle(endpoint, edge_out, leg_out, true, cross_normal, offset);
    let by_face_normal = leg_meet_oracle(endpoint, edge_out, leg_out, true, sector_normal, offset);

    assert!(
        (placement - by_cross_normal).length() < 1.0e-5,
        "ordinary branch must place from the edge-cross normal, got {placement:?} \
         vs {by_cross_normal:?}",
    );
    assert!(
        (placement - by_face_normal).length() > 1.0e-3,
        "and must not have used the sector face's normal as the plane normal",
    );
}

/// The §7.5.1 band edges, on all three regimes: `EPS_ANG = 2°` is inclusive at the
/// bottom of regime 2 (so `[0, 2°)` and `(178°, 180°]` belong to regime 1 and the
/// shallow branch must **not** swallow them — the mistake §7.5.1 warns about) and
/// `SMALL_ANG = 10°` is exclusive at the top (10° itself is already ordinary).
#[test]
fn terminal_leg_meet_direction_angle_band_edges_dispatch_per_the_thresholds() {
    let offset = 0.25;
    for (direction_angle_deg, expected_regime) in [
        (1.9_f32, 1),
        (2.1, 2),
        (9.9, 2),
        (10.1, 3),
        (177.9, 3),
        (178.1, 1),
    ] {
        let (endpoint, edge_out, leg_out, sector_normal) =
            shallow_terminal_frame(direction_angle_deg);
        let placement = leg_meet(
            endpoint,
            edge_out,
            leg_out,
            SectorPlane::Face(sector_normal),
            offset,
        );
        let expected = match expected_regime {
            1 => degenerate_placement_oracle(
                endpoint,
                edge_out,
                leg_out,
                true,
                sector_normal,
                offset,
            ),
            2 => leg_meet_oracle(endpoint, edge_out, leg_out, true, sector_normal, offset),
            _ => leg_meet_oracle(
                endpoint,
                edge_out,
                leg_out,
                true,
                edge_out.cross(leg_out).normalize(),
                offset,
            ),
        };
        assert!(
            (placement - expected).length() < 1.0e-5,
            "at {direction_angle_deg}° the dispatch should reach regime \
             {expected_regime}, got {placement:?} vs {expected:?}",
        );
    }
}

/// An independent transcription of §7.5.1 regime 1 / §7.6.2's two special cases,
/// stated straight from the guide's prose: sum the two bound directions, cross with
/// the sector normal, push by `offset / cos(θ_dir/2)` when the directions run
/// straight through; slide along the *second* edge by `offset` when they double back.
fn degenerate_placement_oracle(
    endpoint: Vec3,
    edge_out: Vec3,
    leg_out: Vec3,
    leg_precedes_edge: bool,
    plane_normal: Vec3,
    offset: f32,
) -> Vec3 {
    let (first_dir, second_dir) = if leg_precedes_edge {
        (-leg_out, edge_out)
    } else {
        (-edge_out, leg_out)
    };
    let direction_angle = first_dir
        .cross(second_dir)
        .length()
        .atan2(first_dir.dot(second_dir));
    if direction_angle > std::f32::consts::FRAC_PI_2 {
        return endpoint + second_dir.normalize() * offset;
    }
    let perpendicular = (first_dir + second_dir).cross(plane_normal).normalize();
    endpoint + perpendicular * (offset / (direction_angle * 0.5).cos())
}

/// §7.5.1 regime 1, near-collinear half: the two edges run straight through the
/// endpoint, so there is no isolated crossing to solve for and the meet is pushed
/// straight out perpendicular to both by `offset / cos(θ_dir/2)`.
///
/// The expectation is a closed form worked by hand in a frame chosen to make one
/// available: with the sector plane's normal `+z`, `E` along `+x` and the leg at
/// `α` from straight-through, the common direction is `(1 + cos α, −sin α, 0)` of
/// length `2 cos(α/2)`, so the unit perpendicular is exactly
/// `(−sin(α/2), −cos(α/2), 0)`. That `−y` sense is the *continuity* check: it is
/// the side regimes 2 and 3 push `E`'s own offset line to in this frame.
#[test]
fn terminal_leg_meet_near_collinear_direction_angle_pushes_out_the_common_perpendicular() {
    let offset = 0.25;
    let endpoint = Vec3::new(0.3, -0.7, 1.1);
    let sector_normal = Vec3::Z;
    let edge_out = Vec3::X;
    for direction_angle_deg in [0.0_f32, 1.0, 1.9] {
        let alpha = direction_angle_deg.to_radians();
        let leg_out = Vec3::new(-alpha.cos(), alpha.sin(), 0.0);
        let placement = leg_meet(
            endpoint,
            edge_out,
            leg_out,
            SectorPlane::Face(sector_normal),
            offset,
        );

        let half = alpha * 0.5;
        let expected = endpoint + Vec3::new(-half.sin(), -half.cos(), 0.0) * (offset / half.cos());
        assert!(
            (placement - expected).length() < 1.0e-5,
            "at {direction_angle_deg}° the perpendicular push should land at \
             {expected:?}, got {placement:?}",
        );
        // The distance is the whole point of the `1 / cos(θ_dir/2)` factor.
        assert!(
            ((placement - endpoint).length() - offset / half.cos()).abs() < 1.0e-6,
            "the push distance must be offset / cos(θ_dir/2)",
        );
        // Square to the *common* direction — which is what "perpendicular to both
        // edges" means once the two directions differ: the push leans `α/2` off
        // square to each of them individually, and that lean is exactly what the
        // `1 / cos(θ_dir/2)` factor lengthens the push to compensate for.
        let push = (placement - endpoint).normalize();
        let common_direction = (-leg_out + edge_out).normalize();
        assert!(
            push.dot(common_direction).abs() < 1.0e-5,
            "the push must be square to the common direction, got {push:?}",
        );
        assert!(
            (push.dot(edge_out) + half.sin()).abs() < 1.0e-5,
            "and must lean exactly θ_dir/2 off square to each edge",
        );
    }
}

/// BEVEL.md §7.6.2's raw-sum rule, on the terminal half of the same ladder: the common
/// direction is the sum of the pair's **raw** directions (first taken into the
/// endpoint, second out of it), which telescopes to `second_far − first_far`. Summing
/// the unit directions instead drops the lean toward the longer flank.
///
/// The frame is the guide's own oracle pin: flanking lengths `2` and `3.00042`, bearings
/// `0.955°` apart. At `offset = 0.5` the raw sum puts the strip vert `0.00500` off the
/// axis where the unit sum puts it `0.00417` — **83 quanta** of the canonical `1e-5`
/// grid, from a direction angle under one degree; at `offset = 6` the same choice moves
/// it from `0.0600` to `0.0500`, a thousand quanta. Neither offset is clamped: regime 1
/// returns ahead of §7.5.2's span clamp and the straight-through half slides along
/// nothing, so the push is free to run past both flanks' lengths.
#[test]
fn terminal_straight_through_push_sums_raw_directions_not_unit_ones() {
    let endpoint = Vec3::ZERO;
    let direction_angle = 0.955_f32.to_radians();
    let leg_out = Vec3::new(-1.0, 0.0, 0.0);
    let edge_out = Vec3::new(direction_angle.cos(), direction_angle.sin(), 0.0);
    let leg_far = endpoint + leg_out * 2.0;
    let edge_far = endpoint + edge_out * 3.00042;

    for (offset, raw_x, unit_x) in [
        (0.5_f32, 0.0050006_f32, 0.0041671_f32),
        (6.0, 0.0600078, 0.0500050),
    ] {
        let placement = terminal_leg_meet(TerminalLegMeet {
            endpoint,
            edge_far,
            leg_far,
            leg_precedes_edge: true,
            sector_plane: SectorPlane::Face(Vec3::Z),
            offset,
        })
        .position();

        assert!(
            (placement.x - raw_x).abs() < 1.0e-5,
            "at offset {offset} the raw sum puts the strip vert {raw_x} off the axis, \
             got {placement:?}",
        );
        assert!(
            (placement.x - unit_x).abs() > 5.0e-4,
            "at offset {offset} the unit-vector sum ({unit_x}) is the failure mode",
        );
        // The push is square to the common direction and `offset / cos(θ_dir/2)` long,
        // whichever sum built it — so the lean is the *only* thing under test here.
        assert!(
            ((placement - endpoint).length() - offset / (direction_angle * 0.5).cos()).abs()
                < 1.0e-5,
            "the push distance must stay `offset / cos(θ_dir/2)`",
        );
    }
}

/// §7.5.1 regime 1, doubled-back half: the two edges leave the endpoint the same way
/// and lie back on one another, enclosing a zero-area sliver with no wedge to bisect.
/// The meet sits **on the common line**, slid out along the *second* edge of the pair
/// by the larger of the two offsets — at a terminal, the beveled edge's. There is no
/// perpendicular component at all, which is what discriminates it from the other half.
///
/// The pair's order decides which edge is slid along, so both orders are exercised:
/// with the leg first it is `E`, with `E` first it is the leg.
#[test]
fn terminal_leg_meet_doubled_back_direction_angle_slides_along_the_second_edge() {
    let offset = 0.25;
    let endpoint = Vec3::new(-1.0, 2.0, 0.5);
    let sector_normal = Vec3::new(0.0, 0.55, 1.0).normalize();
    for direction_angle_deg in [178.1_f32, 179.5, 180.0] {
        let alpha = direction_angle_deg.to_radians();
        let edge_out = Vec3::X;
        let leg_out = Vec3::new(-alpha.cos(), alpha.sin(), 0.0);
        for (leg_precedes_edge, slide_along) in [(true, edge_out), (false, leg_out)] {
            let placement = terminal_leg_meet(TerminalLegMeet {
                endpoint,
                edge_far: endpoint + edge_out,
                leg_far: endpoint + leg_out,
                leg_precedes_edge,
                sector_plane: SectorPlane::Face(sector_normal),
                offset,
            })
            .position();
            let expected = endpoint + slide_along * offset;
            assert!(
                (placement - expected).length() < 1.0e-5,
                "at {direction_angle_deg}° (leg first = {leg_precedes_edge}) the meet \
                 should slide along the second edge to {expected:?}, got {placement:?}",
            );
            assert!(
                distance_from_line(placement, endpoint, slide_along) < 1.0e-5,
                "the doubled-back meet takes no perpendicular offset at all",
            );
        }
    }
}

/// The doubled-back half of regime 1 is a **plain on-edge slide**, so it routes
/// through §4.1's shared primitive like every other slide in the operator — and the
/// clamp is therefore measured against the length of the edge actually slid along,
/// the pair's **second** member. §7.5.2's span clamp stays out of it (it is armed on
/// a *meet*, and regime 1 solves no lines), so this is the guard.
///
/// The frame gives `E` length `0.3` and the leg length `1.0`, and asks for `0.5`: with
/// the leg first the slide runs along `E` and overshoots, landing `5e-5` short of `E`'s
/// far end; with `E` first it runs along the leg and fits, sliding the full `0.5`. One
/// offset, one geometry, two landings — which is what makes the clamp's target visible.
#[test]
fn terminal_doubled_back_slide_is_clamped_against_the_second_edge() {
    let offset = 0.5;
    let endpoint = Vec3::new(-1.0, 2.0, 0.5);
    let sector_normal = Vec3::new(0.0, 0.55, 1.0).normalize();
    let edge_length = 0.3;
    let leg_length = 1.0;
    for direction_angle_deg in [178.1_f32, 179.5, 180.0] {
        let alpha = direction_angle_deg.to_radians();
        let edge_out = Vec3::X;
        let leg_out = Vec3::new(-alpha.cos(), alpha.sin(), 0.0);
        let edge_far = endpoint + edge_out * edge_length;
        let leg_far = endpoint + leg_out * leg_length;
        for (leg_precedes_edge, second_out, second_length) in
            [(true, edge_out, edge_length), (false, leg_out, leg_length)]
        {
            let placement = terminal_leg_meet(TerminalLegMeet {
                endpoint,
                edge_far,
                leg_far,
                leg_precedes_edge,
                sector_plane: SectorPlane::Face(sector_normal),
                offset,
            })
            .position();

            let travelled = if offset > second_length {
                second_length - SLIDE_CLAMP_GAP
            } else {
                offset
            };
            let expected = endpoint + second_out * travelled;
            assert!(
                (placement - expected).length() < 1.0e-6,
                "at {direction_angle_deg}° (leg first = {leg_precedes_edge}) the slide \
                 should land at {expected:?}, got {placement:?}",
            );
            // The clamp lands *short* of the far endpoint by the absolute gap — it
            // does not snap onto it, which is the §7.5.2 rule for a different family.
            if offset > second_length {
                let far = endpoint + second_out * second_length;
                assert!(
                    ((far - placement).length() - SLIDE_CLAMP_GAP).abs() < 1.0e-7,
                    "the clamped slide must stop exactly {SLIDE_CLAMP_GAP} short of \
                     the second edge's far endpoint",
                );
            }
        }
    }
}

/// BEVEL.md §7.6.2's `#zero-area-sector-collapse`, on the terminal side of the same
/// ladder: §7.5.1's `reference` takes the sector face's normal **directly**, so a
/// zero-area face hands over the zero vector, the cross with the common direction is
/// zero, and the zero-returning normalisation leaves the straight-through push with
/// nothing to scale — the strip vert stays **exactly on the endpoint**.
///
/// The likeliest mis-implementation is not a rescued normal but a non-finite
/// coordinate, so finiteness is asserted separately from the position.
#[test]
fn terminal_zero_area_sector_push_collapses_onto_the_endpoint() {
    let offset = 0.25;
    let endpoint = Vec3::new(0.3, -0.7, 1.1);
    let edge_out = Vec3::X;
    for direction_angle_deg in [0.0_f32, 1.0, 1.9] {
        let alpha = direction_angle_deg.to_radians();
        let leg_out = Vec3::new(-alpha.cos(), alpha.sin(), 0.0);
        let placement = leg_meet(
            endpoint,
            edge_out,
            leg_out,
            SectorPlane::Face(Vec3::ZERO),
            offset,
        );

        assert!(
            placement.is_finite(),
            "a zero-area sector face must not produce a non-finite coordinate at \
             {direction_angle_deg}°, got {placement:?}",
        );
        assert_eq!(
            placement, endpoint,
            "at {direction_angle_deg}° the collapsed push must stay on the endpoint",
        );

        // With a usable sector plane the same frame pushes a whole `offset` out —
        // which is the distance a rescued normal would put between the two readings.
        let rescued = leg_meet(
            endpoint,
            edge_out,
            leg_out,
            SectorPlane::Face(Vec3::Z),
            offset,
        );
        assert!(
            (rescued - endpoint).length() > offset * 0.9,
            "fixture sanity: a substituted normal would move the strip vert",
        );
    }
}

/// The whole of BEVEL.md §7.6.2's near-collinear ladder, driven through the operator
/// on the degenerate strip the `#zero-area-sector-collapse` pin names: four collinear
/// verts closed by two **zero-area** triangles sharing `{v0, v1}`. Beveling that edge
/// gives each endpoint one doubled-back sector and one straight-through sector — the
/// split "What varies within one vertex" describes — so both halves of the ladder run
/// on one mesh, and both faces being zero-area collapses both pushes:
///
/// ```text
///    at v = (0,0,0):   slide → (0.1, 0, 0)   push → (0, 0, 0)   ← ON v
///    at v = (2,0,0):   slide → (1.9, 0, 0)   push → (2, 0, 0)   ← ON v
/// ```
///
/// The two collapsed verts land on the original coordinates, so the mesh is easy to
/// misread as "the endpoints survived" — they did not; they were consumed and replaced
/// by strip verts that happen to land there. A rescued normal reads `(0, ±0.1, 0)` and
/// `(2, ±0.1, 0)` instead, one whole offset off-axis, with `V`/`E`/`F` unchanged —
/// which is what the off-axis assertion below excludes.
#[test]
fn bevel_edge_zero_area_collinear_strip_collapses_both_pushes_onto_the_endpoints() {
    let mut mesh = LoopMesh::new();
    let v0 = mesh.create_vertex(Vec3::ZERO);
    let v1 = mesh.create_vertex(Vec3::new(2.0, 0.0, 0.0));
    let left_ear = mesh.create_vertex(Vec3::new(-1.0, 0.0, 0.0));
    let right_ear = mesh.create_vertex(Vec3::new(3.0, 0.0, 0.0));
    mesh.face_create_verts(&[v0, v1, left_ear], FaceCreateFlags::NONE);
    mesh.face_create_verts(&[v1, v0, right_ear], FaceCreateFlags::NONE);
    let beveled = mesh.query_edge_exists(v0, v1).expect("the shared edge");

    mesh.bevel(
        &[beveled.into()],
        BevelOptions {
            offset: 0.1,
            segments: 1,
            affect: BevelAffect::Edges,
            ..Default::default()
        },
    );

    let positions: Vec<Vec3> = mesh
        .vert_keys()
        .into_iter()
        .map(|vertex| mesh.vertex_position(vertex))
        .collect();
    assert!(
        positions.iter().all(|p| p.is_finite()),
        "the collapsed pushes must not emit a non-finite coordinate: {positions:?}",
    );
    for expected in [
        Vec3::new(0.1, 0.0, 0.0),
        Vec3::ZERO,
        Vec3::new(1.9, 0.0, 0.0),
        Vec3::new(2.0, 0.0, 0.0),
    ] {
        assert!(
            positions.iter().any(|&p| (p - expected).length() < 1.0e-5),
            "the ladder should place a vert at {expected:?}; got {positions:?}",
        );
    }
    assert!(
        positions
            .iter()
            .all(|p| p.y.abs() < 1.0e-5 && p.z.abs() < 1.0e-5),
        "a collapsed push moves nothing off the common line; got {positions:?}",
    );
}

fn distance_from_line(point: Vec3, base: Vec3, direction: Vec3) -> f32 {
    let unit = direction.normalize();
    let offset_from_base = point - base;
    (offset_from_base - unit * offset_from_base.dot(unit)).length()
}

/// §7.5.1's skew resolution: the meet is the closest-approach point on the **first**
/// line — the pair's disk predecessor — and *not* the midpoint of the closest pair.
///
/// The consequence is an asymmetry, and it is the whole content of this test: with
/// the leg first the meet rides the **leg's own line**; with `E` first it rides
/// **`E`'s offset line**, exactly `offset` off `E`. Regime 2 is the only branch that
/// can make the two lines skew at all, so the frame is shallow and non-planar.
#[test]
fn terminal_leg_meet_skew_pair_takes_the_closest_point_on_the_first_line() {
    let offset = 0.1;
    let (endpoint, edge_out, leg_out, sector_normal) = shallow_terminal_frame(8.0);

    let leg_first = terminal_leg_meet(TerminalLegMeet {
        endpoint,
        edge_far: endpoint + edge_out,
        leg_far: endpoint + leg_out,
        leg_precedes_edge: true,
        sector_plane: SectorPlane::Face(sector_normal),
        offset,
    })
    .position();
    let edge_first = terminal_leg_meet(TerminalLegMeet {
        endpoint,
        edge_far: endpoint + edge_out,
        leg_far: endpoint + leg_out,
        leg_precedes_edge: false,
        sector_plane: SectorPlane::Face(sector_normal),
        offset,
    })
    .position();

    // The frame must genuinely produce a skew pair, or nothing below discriminates.
    let midpoint =
        leg_meet_midpoint_oracle(endpoint, edge_out, leg_out, true, sector_normal, offset);
    assert!(
        (leg_first - midpoint).length() > 1.0e-3,
        "the frame must make the two lines skew: first-line point and midpoint \
         coincide at {leg_first:?}",
    );

    assert!(
        distance_from_line(leg_first, endpoint, leg_out) < 1.0e-5,
        "leg-first meet must land on the leg's own line, it is {} off",
        distance_from_line(leg_first, endpoint, leg_out),
    );
    assert!(
        (distance_from_line(edge_first, endpoint, edge_out) - offset).abs() < 1.0e-5,
        "edge-first meet must land on `E`'s offset line, it is {} from `E`",
        distance_from_line(edge_first, endpoint, edge_out),
    );
    assert!(
        distance_from_line(edge_first, endpoint, leg_out) > 1.0e-3,
        "edge-first meet must NOT also sit on the leg's line — that is the \
         symmetrisation §7.5.1 forbids",
    );
}

/// BEVEL.md §7.5.1 carries both flanks **raw** — `first_dir = endpoint −
/// first.other_end()`, `second_dir = second.other_end() − endpoint`, full length —
/// and hands the solve two *points* per line (`first_base`, `first_base +
/// first_dir`). There is no terminal exemption from §7.6.2's direction and carriage
/// axes; what the terminal owns is only where its *plane normal* comes from.
///
/// Every other terminal fixture in this file is built from unit direction vectors
/// one length-unit out, where the raw and unit carriages are the same vectors and the
/// axis is invisible. This one is deliberately unequal and non-unit — `E` of length
/// `2.1` against a leg of `0.3` at a `2.148°` direction angle, the sweep's
/// widest-parting station — where the two spellings land `6.7e-6` apart, two thirds
/// of a canonical quantum.
#[test]
fn terminal_leg_meet_carries_the_flanks_raw_and_as_two_points() {
    let offset = 0.1;
    let (_, edge_out, leg_out, sector_normal) = shallow_terminal_frame(2.148);
    let endpoint = Vec3::new(0.3, -0.7, 1.1);
    let edge_far = endpoint + edge_out * 2.1;
    let leg_far = endpoint + leg_out * 0.3;

    let placement = terminal_leg_meet(TerminalLegMeet {
        endpoint,
        edge_far,
        leg_far,
        leg_precedes_edge: true,
        sector_plane: SectorPlane::Face(sector_normal),
        offset,
    })
    .position();

    let by_raw = leg_meet_oracle(
        endpoint,
        edge_far - endpoint,
        leg_far - endpoint,
        true,
        sector_normal,
        offset,
    );
    let by_unit = leg_meet_oracle(endpoint, edge_out, leg_out, true, sector_normal, offset);

    assert_eq!(
        placement, by_raw,
        "the terminal meet must be the raw two-point carriage, bit for bit",
    );
    assert!(
        (placement - by_unit).length() > 5.0e-6,
        "fixture sanity: the unit carriage must be a measurably different answer \
         here, it is {} away",
        (placement - by_unit).length(),
    );
}

/// §7.5.2's along-leg projection: `s_leg` is the same number whichever of the two
/// closest-approach points it is measured from, because the segment joining them is
/// perpendicular to both line directions and so to `leg_vec` in particular. This is
/// the property that lets the clamp be stated once for both strip verts. No clamp is
/// invoked here — this pins the projection's pair-independence on its own, which is
/// also a check on `closest_approach` returning a genuinely mutual pair.
#[test]
fn terminal_along_leg_projection_is_pair_independent() {
    let offset = 0.1;
    let (endpoint, edge_out, leg_out, sector_normal) = shallow_terminal_frame(8.0);
    let leg_vec = leg_out; // unit, so the projection is a distance along the leg

    let on_leg_line = terminal_leg_meet(TerminalLegMeet {
        endpoint,
        edge_far: endpoint + edge_out,
        leg_far: endpoint + leg_out,
        leg_precedes_edge: true,
        sector_plane: SectorPlane::Face(sector_normal),
        offset,
    })
    .position();
    // The *sibling* of `on_leg_line`: the other end of the same closest-approach
    // segment, on the same pair of lines (same offset-push sign).
    let sibling = {
        let edge_perp = edge_out.cross(sector_normal).normalize();
        let edge_base = endpoint + edge_perp * offset;
        let line_edge = Line {
            point: edge_base,
            direction: edge_out,
        };
        let line_leg = Line {
            point: endpoint,
            direction: leg_out,
        };
        closest_approach(&line_edge, &line_leg)
            .expect("skew pair")
            .0
    };

    let projection = |point: Vec3| (point - endpoint).dot(leg_vec);
    assert!(
        (projection(on_leg_line) - projection(sibling)).abs() < 1.0e-4,
        "s_leg must be pair-independent: {} vs {}",
        projection(on_leg_line),
        projection(sibling),
    );
    assert!(
        (on_leg_line - sibling).length() > 1.0e-3,
        "the two closest points must actually differ, or the check is vacuous",
    );
}

/// A terminal whose **both** sectors are shallow (regime 2) and non-planar, so both
/// strip verts are placed from a genuinely skew line pair. `E` runs out along `+x`;
/// the two legs turn ~8° off straight-through and are tilted out of the `z = 0` plane
/// in opposite senses. Returns the mesh, the beveled edge, and the two legs' far
/// vertices.
fn skew_terminal_quad_pair() -> (LoopMesh, EdgeKey, [VertexKey; 2]) {
    let mut mesh = LoopMesh::new();
    let near = mesh.create_vertex(Vec3::ZERO);
    let far = mesh.create_vertex(Vec3::new(1.0, 0.0, 0.0));
    let left_leg_end = mesh.create_vertex(Vec3::new(-0.99046, 0.09744, 0.09744));
    let right_leg_end = mesh.create_vertex(Vec3::new(-0.99046, -0.09744, -0.09744));
    let left_far = mesh.create_vertex(Vec3::new(1.3, 1.0, 0.6));
    let right_far = mesh.create_vertex(Vec3::new(1.3, -1.0, -0.6));

    let mut edge = |mesh: &mut LoopMesh, a: VertexKey, b: VertexKey| {
        mesh.create_edge([a, b], EdgeCreateFlags::NONE)
    };
    let beveled = edge(&mut mesh, near, far);
    let left_ring = [
        beveled,
        edge(&mut mesh, far, left_far),
        edge(&mut mesh, left_far, left_leg_end),
        edge(&mut mesh, left_leg_end, near),
    ];
    mesh.face_create(
        &[near, far, left_far, left_leg_end],
        &left_ring,
        FaceCreateFlags::NONE,
    );
    let right_ring = [
        edge(&mut mesh, near, right_leg_end),
        edge(&mut mesh, right_leg_end, right_far),
        edge(&mut mesh, right_far, far),
        beveled,
    ];
    mesh.face_create(
        &[near, right_leg_end, right_far, far],
        &right_ring,
        FaceCreateFlags::NONE,
    );
    (mesh, beveled, [left_leg_end, right_leg_end])
}

/// §7.5.1's asymmetry, driven through the whole operator: at a non-planar terminal
/// the strip vert **before** `E` in rotational order rides its own leg's line, while
/// the one **after** `E` rides `E`'s **offset** line — `offset` away from `E`, and
/// off its leg entirely.
///
/// The two are matched assignment-free: of the two near-end strip verts exactly one
/// must sit on a leg, and the other must sit exactly `offset` from `E`. A test that
/// only sampled the along-leg station could not see this — both verts share the
/// station; it is the lateral component that discriminates.
#[test]
fn edge_bevel_skew_terminal_strip_verts_ride_different_lines() {
    let offset = 0.1;
    let (mut mesh, beveled, legs) = skew_terminal_quad_pair();
    let [near, far] = mesh.edge_vert_keys(beveled);
    let near_position = mesh.vertex_position(near);
    let far_position = mesh.vertex_position(far);
    let edge_out = (far_position - near_position).normalize();
    let leg_directions =
        legs.map(|leg_end| (mesh.vertex_position(leg_end) - near_position).normalize());

    let result = mesh.bevel(
        &[beveled.into()],
        BevelOptions {
            offset,
            affect: BevelAffect::Edges,
            ..Default::default()
        },
    );
    mesh.validate_topology();

    let mut on_a_leg = 0;
    let mut on_the_offset_line = 0;
    let mut seen = 0;
    for &vert in &result.verts {
        let position = mesh.vertex_position(vert);
        if (position - near_position).length() > (position - far_position).length() {
            continue;
        }
        seen += 1;
        let leg_gap = leg_directions
            .iter()
            .map(|&leg_out| distance_from_line(position, near_position, leg_out))
            .fold(f32::INFINITY, f32::min);
        let edge_gap = distance_from_line(position, near_position, edge_out);
        // Every strip vert must be on exactly one of the two lines, never both.
        let rides_a_leg = leg_gap < 1.0e-4;
        let rides_the_offset_line = (edge_gap - offset).abs() < 1.0e-4;
        assert!(
            rides_a_leg != rides_the_offset_line,
            "strip vert {position:?} is {leg_gap} off its leg and {edge_gap} from \
             `E` (offset {offset}) — the two cases must be exclusive here, or the \
             asymmetry is unobservable on this fixture",
        );
        on_a_leg += usize::from(rides_a_leg);
        on_the_offset_line += usize::from(rides_the_offset_line);
    }

    assert_eq!(seen, 2, "the terminal contributes two strip verts");
    assert_eq!(
        (on_a_leg, on_the_offset_line),
        (1, 1),
        "§7.5.1: exactly one strip vert rides its leg and exactly one rides `E`'s \
         offset line — symmetrising puts both on the same kind of line",
    );
}

/// §7.5.2 property (3): once the along-leg projection leaves `[0, 1]` the snap is
/// **bodily onto the leg's endpoint**, lateral component and all. So past the flip
/// *both* strip verts sit exactly on their own leg's far vertex — including the
/// after-`E` one, which thereby leaves `E`'s offset line entirely. "Lands `offset`
/// from `E`" is a property of the unclamped meet only.
#[test]
fn edge_bevel_skew_terminal_span_clamp_snaps_both_verts_onto_their_legs() {
    let offset = 0.5; // well past the flip: `s_leg` crosses 1 an order sooner
    let (mut mesh, beveled, legs) = skew_terminal_quad_pair();
    let [near, far] = mesh.edge_vert_keys(beveled);
    let near_position = mesh.vertex_position(near);
    let far_position = mesh.vertex_position(far);
    let leg_ends = legs.map(|leg_end| mesh.vertex_position(leg_end));

    let result = mesh.bevel(
        &[beveled.into()],
        BevelOptions {
            offset,
            affect: BevelAffect::Edges,
            ..Default::default()
        },
    );
    mesh.validate_topology();

    // The clamp is per sector, so the two near-end strip verts must land on the two
    // *distinct* leg ends — one each. Folding a `min` over both would pass a bug that
    // snapped both verts onto the same leg.
    assert!(
        (leg_ends[0] - leg_ends[1]).length() > 1.0e-2,
        "the two leg ends must be distinct, or the pairing check is vacuous",
    );
    let mut claimed = [false; 2];
    let mut matched = 0;
    for &vert in &result.verts {
        let position = mesh.vertex_position(vert);
        if (position - near_position).length() > (position - far_position).length() {
            continue;
        }
        let landed_on = leg_ends
            .iter()
            .position(|&leg_end| (position - leg_end).length() < 1.0e-4);
        let Some(index) = landed_on else {
            panic!(
                "a clamped terminal strip vert must land ON a leg's far vertex, it is \
                 at {position:?}, legs at {leg_ends:?}",
            );
        };
        assert!(
            !claimed[index],
            "both strip verts snapped onto leg {index} — §7.5.2 clamps each vert \
             against its OWN adjacent leg",
        );
        claimed[index] = true;
        matched += 1;
    }
    assert_eq!(matched, 2);
}

/// The §7.5.1 shallow regime driven through the whole operator: two non-planar quads
/// straddle the beveled edge, and at one endpoint each leg turns only ~5° off
/// straight-through. Each near-side strip vert must sit where the **sector face's**
/// normal puts it, measurably away from where the edge-cross normal would.
///
/// The offset is deliberately small: at a 5° direction angle the meet runs out to
/// roughly `offset / sin(5°)` along the leg, so anything much larger is snapped by
/// the §7.5.2 span clamp onto the leg's far endpoint — which is the *same* point
/// under either normal, and the test would stop discriminating the regimes at all.
#[test]
fn edge_bevel_shallow_direction_angle_terminal_places_strip_verts_by_the_sector_normal() {
    let leg_turn = 5.0_f32.to_radians();
    let mut mesh = LoopMesh::new();
    let near = mesh.create_vertex(Vec3::ZERO);
    let far = mesh.create_vertex(Vec3::new(1.0, 0.0, 0.0));
    // The two legs at `near` turn only `leg_turn` off straight-through, so the
    // direction angle there is ~5° — inside regime 2's band.
    let left_leg_end = mesh.create_vertex(Vec3::new(-leg_turn.cos(), leg_turn.sin(), 0.0));
    let right_leg_end = mesh.create_vertex(Vec3::new(-leg_turn.cos(), -leg_turn.sin(), 0.0));
    // Lifted out of z = 0 so neither quad is planar and the sector normal genuinely
    // differs from the corner's edge-cross normal.
    let left_far = mesh.create_vertex(Vec3::new(1.3, 1.0, 0.6));
    let right_far = mesh.create_vertex(Vec3::new(1.3, -1.0, -0.6));

    let mut edge = |mesh: &mut LoopMesh, a: VertexKey, b: VertexKey| {
        mesh.create_edge([a, b], EdgeCreateFlags::NONE)
    };
    let beveled = edge(&mut mesh, near, far);
    let left_ring = [
        beveled,
        edge(&mut mesh, far, left_far),
        edge(&mut mesh, left_far, left_leg_end),
        edge(&mut mesh, left_leg_end, near),
    ];
    let left_face = mesh.face_create(
        &[near, far, left_far, left_leg_end],
        &left_ring,
        FaceCreateFlags::NONE,
    );
    let right_ring = [
        edge(&mut mesh, near, right_leg_end),
        edge(&mut mesh, right_leg_end, right_far),
        edge(&mut mesh, right_far, far),
        beveled,
    ];
    let right_face = mesh.face_create(
        &[near, right_leg_end, right_far, far],
        &right_ring,
        FaceCreateFlags::NONE,
    );

    // Both endpoints of the beveled edge are killed by the strip, so the frame has
    // to be read off the pristine mesh.
    let near_position = mesh.vertex_position(near);
    let far_position = mesh.vertex_position(far);
    let edge_out = (far_position - near_position).normalize();
    let sectors =
        [(left_leg_end, left_face), (right_leg_end, right_face)].map(|(leg_end, face)| {
            let leg_out = (mesh.vertex_position(leg_end) - near_position).normalize();
            (leg_out, mesh.face_plane_normal(face).normalize())
        });

    let offset = 0.02;
    let result = mesh.bevel(
        &[beveled.into()],
        BevelOptions {
            offset,
            affect: BevelAffect::Edges,
            ..Default::default()
        },
    );
    mesh.validate_topology();

    // The rotational order fixes which side of `E` each leg's offset line is pushed
    // to; that sign is pinned by the unit tests above, so here take the nearer of
    // the two and let the *normal* be what discriminates.
    let nearest_over_signs = |plane_normal: Vec3, leg_out: Vec3, position: Vec3| {
        [true, false]
            .map(|leg_precedes_edge| {
                let expected = leg_meet_oracle(
                    near_position,
                    edge_out,
                    leg_out,
                    leg_precedes_edge,
                    plane_normal,
                    offset,
                );
                (expected - position).length()
            })
            .into_iter()
            .fold(f32::INFINITY, f32::min)
    };

    let mut matched = 0;
    for &vert in &result.verts {
        let position = mesh.vertex_position(vert);
        if (position - near_position).length() > (position - far_position).length() {
            continue; // a strip vert belonging to the other endpoint
        }
        let cross_gap = sectors
            .iter()
            .map(|&(leg_out, _)| {
                nearest_over_signs(edge_out.cross(leg_out).normalize(), leg_out, position)
            })
            .fold(f32::INFINITY, f32::min);
        let sector_gap = sectors
            .iter()
            .map(|&(leg_out, sector_normal)| nearest_over_signs(sector_normal, leg_out, position))
            .fold(f32::INFINITY, f32::min);
        assert!(
            sector_gap < 1.0e-4,
            "shallow terminal strip vert is {sector_gap} off the sector-normal meet",
        );
        assert!(
            cross_gap > 1.0e-2,
            "the fixture must discriminate: the edge-cross normal would put this \
             strip vert only {cross_gap} away",
        );
        matched += 1;
    }
    assert_eq!(
        matched, 2,
        "the shallow terminal contributes two strip verts"
    );
}
