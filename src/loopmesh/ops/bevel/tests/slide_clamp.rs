use super::*;

/// The position a slide of `distance` from `from` toward `to` lands on once
/// §4.1's guard has clamped it: a fixed `5e-5` short of `to`, not on it.
fn clamped_slide_landing(from: Vec3, to: Vec3) -> Vec3 {
    let toward = to - from;
    let length = toward.length();
    from + toward / length * (length - 5.0e-5)
}

/// BEVEL.md §4.1 / §8.3.0.1's knee: the concave (`profile < 0.25`) leg slide is
/// `offset·√2`, so on a `0.4`-long leg it overshoots from `offset = 0.283`
/// upward. At `offset = 0.35` it wants `0.494975` — well past the leg — and
/// §4.1's shared primitive lands the leg vert `5e-5` **short** of the leg's far
/// endpoint. Not on the endpoint (that is §7.5.2's span clamp, a different
/// guard for a different placement), and not at the un-clamped position `0.095`
/// beyond it.
#[test]
fn terminal_leg_vert_slide_clamps_five_e_minus_five_short_of_the_far_endpoint() {
    let (mesh, result, terminal, leg_far) = bevel_hex_tube_middle_edge(0.35, 0.0, 1);

    let toward_far = leg_far - terminal;
    let leg_length = toward_far.length();
    assert!(
        (leg_length - 0.4).abs() < 1.0e-5,
        "the hex tube's circumferential legs are 0.4 long",
    );

    assert_eq!(
        result_has_vert_near(&mesh, &result, clamped_slide_landing(terminal, leg_far)),
        1,
        "the leg vert lands 5e-5 short of the leg's far endpoint",
    );
    assert_eq!(
        result_has_vert_near(&mesh, &result, leg_far),
        0,
        "and NOT bodily on the endpoint — that is the §7.5.2 span clamp's landing",
    );
    let unclamped = terminal + toward_far / leg_length * (0.35 * std::f32::consts::SQRT_2);
    assert_eq!(
        result_has_vert_near(&mesh, &result, unclamped),
        0,
        "nor at the un-clamped `offset·√2` position past the far endpoint",
    );
}

/// BEVEL.md §4.1 property 3(a): the gap is absolute, so the clamped landing is
/// **independent of how badly the slide overshot** — every offset past the knee
/// reads the same coordinate while the un-clamped answer keeps running away.
/// Below the knee (`offset = 0.28`, wanting `0.39598`) the slide is untouched.
#[test]
fn terminal_leg_vert_slide_clamp_landing_is_constant_past_the_knee() {
    let (mesh, result, terminal, leg_far) = bevel_hex_tube_middle_edge(0.28, 0.0, 1);
    let toward_far = leg_far - terminal;
    let leg_length = toward_far.length();
    let below_knee = terminal + toward_far / leg_length * (0.28 * std::f32::consts::SQRT_2);
    assert_eq!(
        result_has_vert_near(&mesh, &result, below_knee),
        1,
        "below the knee the concave slide is not clamped at all",
    );

    for offset in [0.30_f32, 0.35, 0.39] {
        let (mesh, result, terminal, leg_far) = bevel_hex_tube_middle_edge(offset, 0.0, 1);
        assert_eq!(
            result_has_vert_near(&mesh, &result, clamped_slide_landing(terminal, leg_far)),
            1,
            "offset {offset} past the knee reads the same clamped coordinate",
        );
    }
}

/// The guard sits in the slide primitive, not in the `segments = 1` close, so
/// the multi-segment terminal-chain builder clamps its leg vert identically
/// (BEVEL.md §8.3.0.1: the reading is unchanged by `segments`).
#[test]
fn terminal_leg_vert_slide_clamp_fires_at_segments_two() {
    let (mesh, result, terminal, leg_far) = bevel_hex_tube_middle_edge(0.35, 0.0, 2);
    assert_eq!(
        result_has_vert_near(&mesh, &result, clamped_slide_landing(terminal, leg_far)),
        1,
        "the clamp is not segment-count-gated",
    );
    assert_eq!(
        result_has_vert_near(&mesh, &result, leg_far),
        0,
        "still short of the endpoint, not on it",
    );
}

/// BEVEL.md §4.1, "profile is not the only route in": the guard is
/// **profile-blind**, and the plainest route to it has no profile in sight — a
/// vertex-mode `OFFSET` bevel whose offset simply exceeds the incident edge.
/// On the unit cube (edges of length `2`) at the default round profile,
/// `offset = 3` with `clamp_overlap` off slides every ring vert to `2 − 5e-5`
/// from the corner, not to `3` and not onto the neighbour.
#[test]
fn vertex_ring_vert_clamps_on_a_plain_offset_overshoot_round_profile() {
    let mut mesh = LoopMesh::new();
    mesh_add_cube(&mut mesh);
    let corner = vert_at(&mesh, Vec3::new(1.0, -1.0, -1.0));
    let corner_position = mesh.vertex_position(corner);
    let neighbours = [
        Vec3::new(-1.0, -1.0, -1.0),
        Vec3::new(1.0, 1.0, -1.0),
        Vec3::new(1.0, -1.0, 1.0),
    ];

    let result = mesh.bevel(
        &[corner.into()],
        BevelOptions {
            offset: 3.0,
            clamp_overlap: false,
            ..Default::default()
        },
    );
    mesh.validate_topology();

    for neighbour in neighbours {
        assert_eq!(
            result_has_vert_near(
                &mesh,
                &result,
                clamped_slide_landing(corner_position, neighbour),
            ),
            1,
            "the ring vert toward {neighbour} lands 5e-5 short of it",
        );
        assert_eq!(
            result_has_vert_near(&mesh, &result, neighbour),
            0,
            "and not coincident with the neighbour itself",
        );
    }
}

/// BEVEL.md §4.1 property 2: the trigger is a **strict** `>`, so a slide that
/// resolves to exactly the edge length lands precisely on the far endpoint and
/// one representable step above it jumps back by the whole `5e-5` gap.
#[test]
fn slide_primitive_trigger_is_strict_against_the_edge_length() {
    let origin = Vec3::ZERO;
    let far = Vec3::new(0.4, 0.0, 0.0);

    assert_eq!(
        slide_along_edge(origin, far, 0.4),
        far,
        "a slide of exactly the edge length is not clamped",
    );
    let just_over = slide_along_edge(origin, far, f32::from_bits(0.4_f32.to_bits() + 1));
    assert!(
        ((far.x - just_over.x) - 5.0e-5).abs() < 1.0e-7,
        "one step above the edge length the vertex jumps back the whole gap",
    );
    assert!(
        (slide_along_edge(origin, far, 40.0).x - just_over.x).abs() < 1.0e-7,
        "the landing is independent of how far the slide overshot",
    );
    assert_eq!(
        slide_along_edge(origin, far, 0.1),
        Vec3::new(0.1, 0.0, 0.0),
        "a slide that fits is untouched",
    );
}

/// BEVEL.md §4.1 property 4, both degenerate regimes. Strictly between the
/// direction floor and the gap the clamped distance goes **negative** and the
/// vertex is reflected behind the corner; at or below the direction floor the
/// direction collapses and the vertex lands exactly on the corner — never NaN,
/// and never reflected.
#[test]
fn slide_primitive_reflects_on_a_sub_gap_edge_and_pins_a_directionless_one() {
    let origin = Vec3::ZERO;

    let sub_gap = Vec3::new(2.0e-5, 0.0, 0.0);
    let reflected = slide_along_edge(origin, sub_gap, 0.1);
    assert!(
        (reflected.x - (2.0e-5 - 5.0e-5)).abs() < 1.0e-9,
        "an edge shorter than the gap reflects the vertex behind the corner, at {reflected:?}",
    );

    assert_eq!(
        slide_along_edge(origin, origin, 0.1),
        origin,
        "coincident endpoints place the vertex on the corner",
    );
    let below_floor = Vec3::new(1.0e-18, 0.0, 0.0);
    assert_eq!(
        slide_along_edge(origin, below_floor, 0.1),
        origin,
        "so does any separation at or below the squared-length direction floor",
    );
}
