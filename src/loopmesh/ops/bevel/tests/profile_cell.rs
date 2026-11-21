use super::*;

/// BEVEL.md §8.2.2.1 (the degenerate-`mid` rescue, tier 1): a planar corner's
/// closest-approach `mid` lands *on* the `start`–`end` line and supplies no bulge.
/// The bulge is then taken from the original vertex — off the chord, in the plane the
/// legs span — so the profile stays a **curve**, and the samples drop orthogonally
/// into that plane rather than being cast along the edge.
#[test]
fn profile_cell_collinear_corner_rescues_the_bulge_to_the_vertex() {
    let edge_point = Vec3::ZERO;
    let edge_dir = Vec3::X;
    let start = Vec3::new(2.0, 1.0, 0.0);
    let end = Vec3::new(4.0, 3.0, 0.0);

    let plain = ProfileCell::new(start, end, edge_point, edge_dir);
    assert!(
        plain.collinear,
        "the start–end line meets the edge's line, so the on-edge mid is collinear",
    );

    let mut rescued = plain;
    rescued.rescue_degenerate_mid(&MidRescue {
        vertex: Vec3::new(2.0, 3.0, 0.0),
        neighbour_dirs: None,
    });
    assert!(!rescued.collinear, "tier 1 recovers a frame");
    assert_eq!(
        rescued.mid,
        Vec3::new(2.0, 3.0, 0.0),
        "the bulge is the vertex"
    );
    assert!(
        (rescued.projection_dir - rescued.mid_plane_normal).length() < 1.0e-6,
        "a rescued profile projects along its own plane normal, not along the edge",
    );

    // Tier 3: with the vertex itself on the start–end line even the re-derived
    // triangle is degenerate, and only then is the profile a straight chord.
    let mut collapsed = plain;
    collapsed.rescue_degenerate_mid(&MidRescue {
        vertex: Vec3::new(0.0, -1.0, 0.0),
        neighbour_dirs: None,
    });
    assert!(collapsed.collinear, "a still-collinear triangle collapses");
    assert_eq!(
        collapsed.projection_dir,
        Vec3::ZERO,
        "a collapsed profile has no projection setup — the signature both overrides key on",
    );
    assert_eq!(
        collapsed.mid,
        (start + end) * 0.5,
        "the chord's plain midpoint"
    );
}

/// BEVEL.md §8.3.0.3's merge rule, and the construct it scopes its exception to.
///
/// The rule is *"`w_t` is the midpoint of `p_a` and `p_b` — except that for a
/// non-custom profile in which **exactly one** of the two has degenerated to a
/// straight chord (§8.2.2.1's **tier-3 collapse**), `w_t` is the curved one's sample
/// instead."* The named construct is the tier-3 collapse, whose signature §8.2.2.1
/// gives as *"'`projection_dir` is zero' is thus the machine-checkable signature of a
/// collapsed profile"* — **not** the cell being chord-*shaped*, which is a different
/// question with its own flag. The last arm below is the one that separates them.
#[test]
fn weld_merge_exception_is_scoped_to_the_tier3_collapse_not_to_a_chord_shape() {
    let start = Vec3::new(2.0, 1.0, 0.0);
    let end = Vec3::new(4.0, 3.0, 0.0);
    let from_first = Vec3::new(0.0, 0.0, 0.0);
    let from_second = Vec3::new(1.0, 0.0, 0.0);
    let midpoint = (from_first + from_second) * 0.5;

    let curved = {
        let mut cell = ProfileCell::new(start, end, Vec3::ZERO, Vec3::X);
        cell.rescue_degenerate_mid(&MidRescue::tier1_only(Vec3::new(2.0, 3.0, 0.0)));
        assert!(!cell.is_tier3_collapsed(), "tier 1 keeps a projection");
        cell
    };
    let collapsed = {
        let mut cell = ProfileCell::new(start, end, Vec3::ZERO, Vec3::X);
        cell.rescue_degenerate_mid(&MidRescue::tier1_only(Vec3::new(0.0, -1.0, 0.0)));
        assert!(cell.is_tier3_collapsed(), "tier 3 zeroes the projection");
        cell
    };

    assert_eq!(
        merge_welded_sample(&curved, &curved, false, from_first, from_second),
        midpoint,
        "two curved profiles: the plain midpoint",
    );
    assert_eq!(
        merge_welded_sample(&collapsed, &collapsed, false, from_first, from_second),
        midpoint,
        "two collapsed profiles: still the plain midpoint — the exception needs \
         EXACTLY one",
    );
    assert_eq!(
        merge_welded_sample(&collapsed, &curved, false, from_first, from_second),
        from_second,
        "exactly one collapsed: the curved one's sample stands",
    );
    assert_eq!(
        merge_welded_sample(&curved, &collapsed, false, from_first, from_second),
        from_first,
        "and symmetrically, whichever side carries the chord",
    );

    // The exception's *other* conjunct: it fires only when NEITHER profile's
    // `profile_type` is `CUSTOM`. §8.3.0.3: *"A profile whose `profile_type` is `CUSTOM`
    // always takes the midpoint, flagged or not."*
    for (first, second) in [(&collapsed, &curved), (&curved, &collapsed)] {
        assert_eq!(
            merge_welded_sample(first, second, true, from_first, from_second),
            midpoint,
            "a CUSTOM cross-section takes the midpoint even with exactly one collapse",
        );
    }

    // The separating arm, posed on the one **production-reachable** cell that emits a
    // straight chord without being the collapse: §8.2.2.1's zero cell axis. The crossed
    // edge's line meets the chord line *at* `end`, so `mid = end` and `b_axis` is zero.
    // The rescue-entry predicate reads that cell as `60°` and skips it, so it keeps its
    // projection direction and is never flagged linear; only the frame builder declines.
    // This is exactly the folded tee's spoke half, and §8.3.0.3 scopes its exception to
    // the collapse, so this pair takes the plain midpoint.
    let zero_axis = {
        let mut cell = ProfileCell::new(start, end, end, Vec3::Z);
        cell.rescue_degenerate_mid(&MidRescue::tier1_only(Vec3::new(2.0, 3.0, 0.0)));
        cell
    };
    assert_eq!(zero_axis.mid, end, "the meet lands on the chain endpoint");
    assert!(
        zero_axis.emits_straight_chord() && !zero_axis.collinear,
        "the frame declines, but the rescue never ran, so nothing flagged it linear",
    );
    assert!(
        !zero_axis.is_tier3_collapsed(),
        "and it kept its projection direction — it is not the tier-3 collapse",
    );
    assert_eq!(
        merge_welded_sample(&zero_axis, &curved, false, from_first, from_second),
        midpoint,
        "a chord-emitting cell that is not the tier-3 collapse the exception names, so \
         the merge averages rather than picking the curved sample",
    );
}

/// BEVEL.md §8.2.2.1: a profile cell's degeneracy is an **angle** band — `EPS_ANG =
/// 2°` from straight or from folded-back, on the two *normalised* axes — and never a
/// floor on the raw cross-product length. An area reading is wrong in both
/// directions, and both errors are exercised here.
#[test]
fn profile_cell_degeneracy_is_a_two_degree_angle_band_not_an_area() {
    // A cell with `mid` at the origin and axes `a_axis`, `b_axis` enclosing `apex`.
    let cell_at = |apex_degrees: f32, leg: f32| {
        let apex = apex_degrees.to_radians();
        let a_axis = Vec3::new(leg, 0.0, 0.0);
        let b_axis = Vec3::new(leg * apex.cos(), leg * apex.sin(), 0.0);
        ProfileCell::from_control_points(-a_axis, -b_axis, Vec3::ZERO)
    };

    // The §8.9.4.1 flanking cell: `1.31°` from straight on legs of `≈ 0.184`. Its raw
    // cross is three orders of magnitude above a `1e-6` area floor, so an area-gated
    // cell never rescues here — the whole of the wide-`spread` miter divergence.
    let flank = cell_at(180.0 - 1.3079, 0.184);
    let raw_cross = (flank.mid - flank.start)
        .cross(flank.mid - flank.end)
        .length();
    assert!(
        (7.0e-4..9.0e-4).contains(&raw_cross),
        "the near-straight flanking cell carries plenty of raw area: {raw_cross}",
    );
    assert!(
        flank.collinear,
        "yet it is inside the 2° band, so it rescues"
    );

    // The contrast arm: `2.84°` from straight is outside the band and keeps its own
    // `mid`. The band is behavioural, so it must not over-fire either.
    assert!(
        !cell_at(180.0 - 2.84, 0.184).collinear,
        "a 2.84° cell is well clear of the band",
    );
    assert!(!cell_at(180.0 - 2.01, 0.184).collinear, "just outside");
    assert!(cell_at(180.0 - 1.99, 0.184).collinear, "just inside");
    assert!(cell_at(1.99, 0.184).collinear, "the folded-back half too");
    assert!(!cell_at(2.01, 0.184).collinear, "and its far side");

    // The other direction: a *small* cell clears an area floor at any angle at all.
    let small = cell_at(45.0, 1.0e-3);
    assert!(
        (small.mid - small.start)
            .cross(small.mid - small.end)
            .length()
            < 1.0e-6,
        "a 45° cell on 1e-3 legs reads below a 1e-6 area floor",
    );
    assert!(!small.collinear, "but 45° is nowhere near the band");
}

/// BEVEL.md §8.6.3.1's `angle_between` convention — the one measure the operator
/// carries — and the `60°` its degenerate input yields.
///
/// §8.6.3.1: *"a zero input gives `dot(d̂, â) = 0`, which takes the **acute** arm. With
/// one input zero the difference chord is the surviving unit vector's own length, `1`:
/// `turn = 2·asin(1/2) = π/3 = 60°` — a fixed constant, independent of the other
/// input."* And: *"**Both tangents zero is a different answer** … the difference chord
/// is `|0 − 0| = 0`, and the segment scores `0°`."*
///
/// The two spellings §8.6.3.1 names as wrong are pinned as non-answers, because one of
/// them inverts §8.2.2.1's zero-cell-axis branch: *"`atan2(cross, dot)` … returns `0°`
/// … `0°` falls inside the `2°` band, so the rescue runs and the welded chain moves."*
/// The other, normalise-then-`acos(dot)`, returns `90°` and reaches the same verdict
/// here for the wrong reason.
#[test]
fn angle_between_is_the_chord_measure_and_reads_a_zero_input_as_sixty_degrees() {
    let sixty = std::f32::consts::FRAC_PI_3;
    for other in [Vec3::X, Vec3::new(0.0, -3.0, 4.0), Vec3::splat(-1.0)] {
        for degenerate in [
            unit_chord_angle_between(Vec3::ZERO, other),
            unit_chord_angle_between(other, Vec3::ZERO),
        ] {
            assert!(
                (degenerate - sixty).abs() < 1.0e-6,
                "a zero input scores a constant 60°, independent of the other input; \
                 got {}°",
                degenerate.to_degrees(),
            );
            assert!(
                degenerate.to_degrees() > 2.0,
                "and 60° is outside EPS_ANG — the `atan2` spelling's 0° is not",
            );
        }
    }
    assert_eq!(
        unit_chord_angle_between(Vec3::ZERO, Vec3::ZERO),
        0.0,
        "both inputs zero is a different answer: a zero chord, so 0°",
    );

    // The ordinary range still agrees with the textbook angle, to well inside the
    // `2°` band the predicate is stated against.
    for degrees in [1.0_f32, 45.0, 90.0, 135.0, 179.0] {
        let radians = degrees.to_radians();
        let measured =
            unit_chord_angle_between(Vec3::X, Vec3::new(radians.cos(), radians.sin(), 0.0));
        assert!(
            (measured.to_degrees() - degrees).abs() < 1.0e-3,
            "{degrees}° measured as {}°",
            measured.to_degrees(),
        );
    }

    assert!(
        !near_parallel_directions(Vec3::ZERO, Vec3::X),
        "so the rescue-entry predicate answers FALSE on a zero axis — the cell never \
         reaches the ladder at all",
    );
}

/// BEVEL.md §8.2.2.1 ["The other degeneracy"]: `mid` coincident with a chain endpoint —
/// a zero cell axis — is **not** a tier-3 collapse, and the rescue is never entered.
///
/// The three consequences the guide lists, in order:
///
/// 1. *"The rescue does not fire, and the cell is not a collapse. … The profile keeps
///    its cross-section exponent (it is **not** flagged linear), keeps `plane_point =
///    start`, and keeps `projection_dir = edge_dir`. … The only frame field that does
///    degenerate is `mid_plane_normal` … and is therefore **zero**."*
/// 2. *"The frame cannot be built, and the builder's own test is what catches it."*
/// 3. *"When the builder returns nothing, the chain's interior is the even straight
///    subdivision `lerp(start, end, k / segments)`"* — *"not the cross-section's own
///    spacing"*.
///
/// **Which threshold this fixture is degenerate under.** It is degenerate under
/// [`cell_frame_is_wellformed`]'s clause (1) — `b_axis` is `Vec3::ZERO` **exactly, component-wise**,
/// which is the guide's own test verbatim. It is *not* degenerate under the rescue's
/// `2°` band (it reads the guide's stated `60°`), and it is *not* degenerate under any
/// loopmesh length floor: no floor is consulted on this path at all.
#[test]
fn zero_cell_axis_declines_the_frame_without_collapsing_the_profile() {
    // The §8.3.0.3 asymmetric weld's spoke: `E` along `x` through the origin, and the
    // chord line meeting it exactly at `end`.
    let start = Vec3::new(-0.25, 0.25, 0.0);
    let end = Vec3::new(0.25, 0.0, 0.0);
    let edge_dir = Vec3::new(-1.0, 0.0, 0.0);
    let mut cell = ProfileCell::new(start, end, Vec3::ZERO, edge_dir);

    assert_eq!(
        cell.mid, end,
        "the meet lands exactly on the chain endpoint"
    );
    assert_eq!(
        cell.mid - cell.end,
        Vec3::ZERO,
        "so one cell axis is exactly zero, component-wise",
    );
    assert!(
        !cell.collinear,
        "consequence 1: the rescue-entry predicate reads 60°, so it answers false",
    );
    assert_eq!(
        cell.projection_dir, edge_dir,
        "consequence 1: `projection_dir` is untouched — only tier 3 zeroes it",
    );
    assert!(
        !cell.is_tier3_collapsed(),
        "consequence 1: and the profile is therefore not flagged linear",
    );
    assert_eq!(
        cell.mid_plane_normal,
        Vec3::ZERO,
        "consequence 1: the one field that does degenerate — the axes' cross product",
    );
    assert!(
        !cell.frame_is_wellformed(),
        "consequence 2: the frame builder's clause (1) declines",
    );

    // Consequence 1 again, from the other end: running the rescue changes nothing,
    // because its gate never opens.
    let before = cell.mid;
    cell.rescue_degenerate_mid(&MidRescue::tier1_only(Vec3::ZERO));
    assert_eq!(cell.mid, before, "the rescue is not entered");
    assert_eq!(cell.projection_dir, edge_dir, "nor re-planed");

    // Consequence 3: the even straight subdivision, at `k / segments` — NOT the
    // cross-section's own spacing, which at `segments = 2` would put the sample at
    // chord fraction `0.7071` (§8.2.2.1's rung 2).
    for segments in [2u32, 3, 4] {
        let chain = cell.chain_positions(segments, 0.5, None, ChainEmission::Raw);
        for k in 0..=segments {
            let want = start + (end - start) * (k as f32 / segments as f32);
            assert!(
                (chain[k as usize] - want).length() < 1.0e-6,
                "segments = {segments}, k = {k}: expected {want:?}, got {:?}",
                chain[k as usize],
            );
        }
    }
}

/// BEVEL.md §8.2.2.1's cell frame builder, clause by clause. The two clauses differ in
/// shape on purpose and neither is a restatement of the rescue's band: *"clause (1) is
/// an exact equality on a quantity the band cannot express, and clause (2) is one-sided
/// where the band is two-sided."*
///
/// **Clause (1) is exact, and its discontinuity is the reference's.** §8.2.2.1: *"When
/// the coincidence is exact the builder declines … When it is off by even a single ulp
/// the builder **succeeds**."* The perturbed cell below is degenerate under **neither**
/// threshold: not the guide's exact-zero clause (1), and not loopmesh's shared direction
/// floor — §8.2.2.1 measures that band as empty here, *"that floor is
/// `DIRECTION_FLOOR_SQ = 1e-35`, a length of `3.2e-18` … the smallest non-zero `f32`
/// difference is already about `3e-8`, ten orders of magnitude above it."*
///
/// **Clause (2) is the `180°` end only**, at the guide's `EPS_ANG = 2°`; the `0°` arm is
/// absent because *"a `0°` pair means `mid` sits on the chord line outside the segment,
/// which the rescue upstream has already caught"*. Both arms are probed a tenth of a
/// degree either side of `2°`, so the pin is on the guide's threshold rather than on
/// any loopmesh epsilon.
#[test]
fn cell_frame_clause_one_is_exact_and_clause_two_is_one_sided() {
    let start = Vec3::new(-0.25, 0.25, 0.0);
    let end = Vec3::new(0.25, 0.0, 0.0);

    assert!(
        !cell_frame_is_wellformed(start, end, end),
        "clause (1): `mid` on `end` — `b_axis` exactly zero",
    );
    assert!(
        !cell_frame_is_wellformed(start, start, end),
        "clause (1): and its mirror, `mid` on `start`",
    );

    // One ulp off in `x` — a miss of ≈ 1.5e-8 world units, ten orders above the
    // direction floor and so a perfectly ordinary unit vector once normalised.
    let barely_off = Vec3::new(f32::from_bits(end.x.to_bits() + 1), end.y, end.z);
    let miss = (barely_off - end).length();
    assert!(
        (1.0e-9..1.0e-6).contains(&miss),
        "fixture sanity: the miss is ULP-scale but far above any length floor: {miss}",
    );
    assert!(
        cell_frame_is_wellformed(start, barely_off, end),
        "clause (1) is an exact equality: one ulp off and the frame IS built",
    );

    // Clause (2): the anti-parallel end of the band, and only that end. Build a `mid`
    // that sits on the chord line at a controlled angular departure from anti-parallel
    // by shifting it off the chord by a known perpendicular amount.
    let chord = end - start;
    let perpendicular = Vec3::new(-chord.y, chord.x, 0.0).normalize();
    let anti_parallel_at = |departure_degrees: f32| {
        // `mid` at the chord midpoint pushed sideways: the two axes then enclose
        // `180° − 2·atan(sideways / half_chord)`.
        let half = chord.length() * 0.5;
        let sideways = half * (departure_degrees.to_radians() * 0.5).tan();
        (start + end) * 0.5 + perpendicular * sideways
    };
    assert!(
        !cell_frame_is_wellformed(start, anti_parallel_at(1.9), end),
        "clause (2): 1.9° from anti-parallel is inside EPS_ANG, so the frame declines",
    );
    assert!(
        cell_frame_is_wellformed(start, anti_parallel_at(2.1), end),
        "clause (2): 2.1° is outside, so it builds — the threshold is the guide's 2°",
    );

    // The `0°` arm: `mid` on the chord line *outside* the segment, so the two axes are
    // near-*parallel*. The rescue owns that case; the builder must not also claim it.
    let outside = start + chord * 2.0;
    assert!(
        near_parallel_directions(outside - start, outside - end),
        "fixture sanity: the rescue's two-sided band does read this cell as degenerate",
    );
    assert!(
        cell_frame_is_wellformed(start, outside, end),
        "clause (2) has no 0° arm — the builder still builds this frame",
    );
}

/// BEVEL.md §8.2.2.1's **rung 2** and its two mirrors. When the meet misses the
/// endpoint by an ulp the frame *is* built, with one near-zero column, and the cell map
/// `origin + u·a_axis + v·b_axis` collapses onto the same chord — but parameterised by
/// the cross-section's own spacing, and running from whichever endpoint the surviving
/// column leaves standing:
///
/// ```text
///   mid coincides with `end`   → b_axis ≈ 0, origin ≈ start
///                              → sample_k = start + x_k · (end − start)
///   mid coincides with `start` → a_axis ≈ 0, origin ≈ end
///                              → sample_k = end   + y_k · (start − end)
/// ```
///
/// §8.2.2.1: *"These are **different points at every interior index and every
/// `segments ≥ 2`** … at `segments = 2` they sit at chord fractions `0.7071` and
/// `0.2929` … at `segments = 3` the two interior fractions are `(0.5, 0.8660)` against
/// `(0.1340, 0.5)`. … get it backwards and every rung-2 sample lands on the wrong side
/// of the chord."*
///
/// This is not the rung the canonical fixtures land on — §8.2.2.1 pins those on
/// clause (1) — but which mirror the map produces is a property of the map, so it is
/// checkable directly and cheaply.
#[test]
fn near_zero_cell_axis_takes_the_stated_rung_two_mirror() {
    let start = Vec3::new(-0.25, 0.25, 0.0);
    let end = Vec3::new(0.25, 0.0, 0.0);
    let chord = end - start;
    let nudge = |point: Vec3| Vec3::new(f32::from_bits(point.x.to_bits() + 1), point.y, point.z);

    let fraction_along_chord = |sample: Vec3| (sample - start).dot(chord) / chord.dot(chord);

    for (mid, expected, which) in [
        (
            nudge(end),
            [[0.7071_f32].as_slice(), [0.5, 0.8660].as_slice()],
            "`mid = end`: b_axis vanishes, so the chord runs from `start` at x_k",
        ),
        (
            nudge(start),
            [[0.2929_f32].as_slice(), [0.1340, 0.5].as_slice()],
            "`mid = start`: a_axis vanishes, so it runs from `end` at y_k",
        ),
    ] {
        let cell = ProfileCell::from_control_points(start, end, mid);
        assert!(
            cell.frame_is_wellformed(),
            "{which}: an ULP-scale miss builds the frame",
        );
        for (index, segments) in [2u32, 3].iter().copied().enumerate() {
            let chain = cell.chain_positions(segments, 0.5, None, ChainEmission::Raw);
            let fractions: Vec<f32> = (1..segments)
                .map(|k| fraction_along_chord(chain[k as usize]))
                .collect();
            for (got, want) in fractions.iter().zip(expected[index]) {
                assert!(
                    (got - want).abs() < 1.0e-3,
                    "{which}, segments = {segments}: expected chord fractions \
                     {:?}, got {fractions:?}",
                    expected[index],
                );
            }
        }
    }
}

/// BEVEL.md §8.9.4.1's worked junction: three beveled edges out of `(0, 0, 1)`, one
/// `SHARP` reflex sector whose boundvert `R` stays at its meet point, and two
/// `ARC`-mitered sectors whose arc ends are slid `spread = 0.3` out along their edges
/// at `offset = 0.015`. The cross-section spanning `R` therefore has one end pinned
/// near the junction and one far out along the edge it crosses, which swings the cell
/// to `1.31°` from straight — inside §8.2.2.1's `2°` band.
///
/// Both of `E`'s disk neighbours are beveled and `s == 3`, so the rescue takes tier 2,
/// whose two offset lines are **skew**. The bulge is the closest-approach foot on the
/// **first** line — the one anchored at the profile's own `start` — and the rival
/// readings sit `0.089` world units away, six times the bevel offset.
#[test]
fn wide_spread_miter_flank_cell_takes_the_start_anchored_tier2_foot() {
    let junction = Vec3::new(0.0, 0.0, 1.0);
    let (edge_left, edge_apex, edge_right) = (
        Vec3::new(-1.0, 1.0, 0.0),
        Vec3::new(0.0, 1.4, 0.0),
        Vec3::new(1.0, 1.0, 0.0),
    );
    let sharp_meet = Vec3::new(0.0, -0.0183712, 1.0183712);
    let arc_start = Vec3::new(-0.1878154, 0.2084076, 0.7974759);

    let mut cell = ProfileCell::new(
        sharp_meet,
        arc_start,
        junction,
        (edge_left - junction).normalize(),
    );
    assert!(
        (cell.mid - Vec3::new(-0.0936675, 0.0936675, 0.9063325)).length() < 1.0e-5,
        "the primary `mid` is the on-edge foot of the closest approach",
    );
    assert!(cell.collinear, "1.31° from straight is inside the 2° band");

    // Disk order is left, apex, right — so the crossed left edge's predecessor is the
    // right edge (start side) and its successor is the apex edge (end side).
    cell.rescue_degenerate_mid(&MidRescue {
        vertex: junction,
        neighbour_dirs: Some((
            (edge_right - junction).normalize(),
            (edge_apex - junction).normalize(),
        )),
    });
    assert!(!cell.collinear, "tier 2 recovers a frame");

    let on_first = Vec3::new(-0.1676101, -0.1859813, 1.1859814);
    let on_second = Vec3::new(-0.1878154, -0.2364944, 1.1152630);
    assert!(
        (cell.mid - on_first).length() < 1.0e-5,
        "the bulge is the foot on the line through `start`, got {:?}",
        cell.mid,
    );
    for rival in [on_second, (on_first + on_second) * 0.5] {
        assert!(
            (cell.mid - rival).length() > 0.04,
            "neither the second line's foot nor the skew pair's midpoint",
        );
    }
    assert!(
        (cell.projection_dir - cell.mid_plane_normal).length() < 1.0e-6,
        "a rescued profile projects along its own plane normal",
    );
}

/// BEVEL.md §8.2.2.1 `#tier2-start-anchored`: because the boundvert ring is walked in
/// one rotational sense, reflecting a junction **swaps** which boundvert is a
/// sector's `start`. The mirrored pair of cross-sections therefore builds the *same*
/// pair of skew lines and anchors on different members of it, so the emitted frames
/// are **not** mirror images — reflecting one gives the other's *second* foot.
///
/// The asymmetry is the rule. Symmetrising it (midpoint, or the second line's foot)
/// is mirror-symmetric by construction and cannot be right.
#[test]
fn mirrored_miter_flank_cells_take_different_feet_of_one_skew_pair() {
    let junction = Vec3::new(0.0, 0.0, 1.0);
    let (edge_left, edge_apex, edge_right) = (
        Vec3::new(-1.0, 1.0, 0.0),
        Vec3::new(0.0, 1.4, 0.0),
        Vec3::new(1.0, 1.0, 0.0),
    );
    let sharp_meet = Vec3::new(0.0, -0.0183712, 1.0183712);
    let arc_start = Vec3::new(-0.1878154, 0.2084076, 0.7974759);
    let arc_end = Vec3::new(0.1878154, 0.2084076, 0.7974759);
    let mirror = |point: Vec3| Vec3::new(-point.x, point.y, point.z);

    let rescued = |start: Vec3, end: Vec3, crossed: Vec3, before: Vec3, after: Vec3| {
        let mut cell = ProfileCell::new(start, end, junction, (crossed - junction).normalize());
        assert!(cell.collinear, "both flanking cells sit inside the band");
        cell.rescue_degenerate_mid(&MidRescue {
            vertex: junction,
            neighbour_dirs: Some((
                (before - junction).normalize(),
                (after - junction).normalize(),
            )),
        });
        cell.mid
    };

    // `R → A1` crosses the left edge (predecessor right, successor apex); the mirror
    // sector `B3 → R` crosses the right edge (predecessor apex, successor left).
    let forward = rescued(sharp_meet, arc_start, edge_left, edge_right, edge_apex);
    let mirrored = rescued(arc_end, sharp_meet, edge_right, edge_apex, edge_left);

    assert!(
        (mirrored - Vec3::new(0.1878154, -0.2364944, 1.1152630)).length() < 1.0e-5,
        "got {mirrored:?}",
    );
    assert!(
        (mirrored - mirror(forward)).length() > 0.04,
        "a mirror-symmetric fixture does not yield a mirror-symmetric frame",
    );
}

/// BEVEL.md §8.2.2.1's tier-2 gate reads the beveled edge's **rotational** neighbours
/// — the edges flanking it *around* the vertex — which only the corner ring carries.
/// The radial structure's own per-vertex edge cycle is linked in insertion order, so
/// on this fan it lists the beveled spokes as `60°, 0°, 120°` and would hand the gate
/// the wrong flanks entirely (firing on the 0° spoke, whose true rotational
/// predecessor is an unbeveled leg, and never on the 60° spoke, which is the one
/// genuinely flanked by two beveled edges).
#[test]
fn tier2_gate_reads_the_rotational_neighbours_not_the_edge_cycle() {
    let (mesh, centre, rim) = planar_fan(6);
    let corners = mesh.vertex_corner_ring(centre);
    let spoke = |k: usize| {
        mesh.query_edge_exists(centre, rim[k])
            .expect("spoke exists")
    };
    let beveled: Vec<EdgeKey> = (0..3).map(spoke).collect();
    let is_beveled = |candidate: EdgeKey| beveled.contains(&candidate);

    let (before, after) = mesh
        .mid_rescue(centre, spoke(1), &corners, &is_beveled)
        .neighbour_dirs
        .expect("the middle beveled spoke is flanked by two beveled spokes");
    let flanks = [before, after];
    for k in [0usize, 2] {
        let direction = mesh.vertex_position(rim[k]).normalize();
        assert!(
            flanks.iter().any(|f| (*f - direction).length() < 1.0e-5),
            "the tier-2 pair is the two rotationally adjacent beveled spokes",
        );
    }

    for k in [0usize, 2] {
        assert!(
            mesh.mid_rescue(centre, spoke(k), &corners, &is_beveled)
                .neighbour_dirs
                .is_none(),
            "an outer beveled spoke has an unbeveled rotational neighbour — tier 1 stands",
        );
    }
}

/// A **boundary** vertex's corner ring is an open fan, and the start face the walk
/// picks is arbitrary — so it can sit mid-fan. Chaining forward only would never reach
/// the faces *before* it and the ring would come back silently truncated, which drops
/// sectors (§7.6.1) and, through §8.2.2.1's tier-2 gate — which reads an edge's
/// rotational neighbours off exactly this ring — moves emitted profile positions.
#[test]
fn open_fan_corner_ring_reaches_both_sides_of_its_start_face() {
    // Every rotation of the build order — the walk's start face lands on each of the
    // fan's four faces in turn, including the two that have a face on *both* sides.
    for first_face in 0..4 {
        let (mesh, centre, rim) = open_planar_fan(5, first_face);
        let spoke = |k: usize| {
            mesh.query_edge_exists(centre, rim[k])
                .expect("spoke exists")
        };
        let corners = mesh.vertex_corner_ring(centre);

        assert_eq!(
            corners.len(),
            4,
            "one corner per incident face — the boundary fan is not truncated",
        );
        assert!(
            corners
                .windows(2)
                .all(|pair| pair[0].outgoing_edge == pair[1].incoming_edge),
            "consecutive corners share an edge",
        );
        let open_ends = [corners[0].incoming_edge, corners[3].outgoing_edge];
        assert!(
            open_ends.contains(&spoke(0)) && open_ends.contains(&spoke(4)),
            "the ring's two open ends are the fan's two boundary spokes",
        );

        // On the truncated ring the tier-2 gate answered `None` for every spoke the
        // ring had lost, silently falling back to tier 1 — an emitted-position change.
        let beveled: Vec<EdgeKey> = (1..4).map(spoke).collect();
        let is_beveled = |candidate: EdgeKey| beveled.contains(&candidate);
        assert!(
            mesh.mid_rescue(centre, spoke(2), &corners, &is_beveled)
                .neighbour_dirs
                .is_some(),
            "the middle beveled spoke of a boundary fan is flanked by two beveled spokes",
        );
        for k in [1usize, 3] {
            assert!(
                mesh.mid_rescue(centre, spoke(k), &corners, &is_beveled)
                    .neighbour_dirs
                    .is_none(),
                "an outer beveled spoke's other flank is the unbeveled boundary spoke",
            );
        }
    }
}

/// BEVEL.md §8.2.2.1 at a **planar `s == 3` junction**: every sector is collinear, so
/// every profile rides the rescue — and the two tiers land in genuinely different
/// places. The middle beveled spoke (both rotational neighbours beveled) takes tier
/// 2's meet of the two neighbours' offset lines; the outer two keep tier 1's
/// vertex-derived bulge. The tier-2 pairing is directional — the predecessor's offset
/// line runs through `start`, the successor's through `end` — and swapping it puts the
/// bulge on the far side of the chord, so all three candidates are pinned apart here.
#[test]
fn planar_junction_middle_sector_takes_the_tier2_offset_line_meet() {
    let (mut mesh, centre, rim) = planar_fan(6);
    let geom: Vec<GeomKey> = (0..3)
        .map(|k| {
            mesh.query_edge_exists(centre, rim[k])
                .expect("spoke exists")
                .into()
        })
        .collect();
    let result = mesh.bevel(
        &geom,
        BevelOptions {
            offset: 0.2,
            affect: BevelAffect::Edges,
            segments: 2,
            vmesh_method: BevelVmeshMethod::Adj,
            ..Default::default()
        },
    );
    mesh.validate_topology();

    // The 60° wedges put their boundverts `2·offset` out along the bisector (§7.6.2).
    let start = Vec3::new(0.34641016, 0.2, 0.0);
    let end = Vec3::new(0.0, 0.4, 0.0);
    // Tier 2: the 0° spoke's offset line (through `start`) meets the 120° spoke's
    // (through `end`) here — a milder bulge than the vertex, and on the vertex's side.
    let tier2 = Vec3::new(0.11547005, 0.2, 0.0);
    // The two rivals: tier 1's vertex bulge, and the same meet with the two offset
    // lines swapped onto the wrong chain ends (which bulges *away* from the vertex).
    let tier1 = Vec3::ZERO;
    let swapped = Vec3::new(0.23094011, 0.4, 0.0);

    let sample = |mid: Vec3| {
        ProfileCell::from_control_points(start, end, mid).chain_positions(
            2,
            0.5,
            None,
            ChainEmission::Resampled,
        )[1]
    };
    assert_eq!(
        result_has_vert_near(&mesh, &result, sample(tier2)),
        1,
        "the middle sector's chain vert is sampled through the tier-2 cell",
    );
    for rival in [tier1, swapped] {
        assert_eq!(
            result_has_vert_near(&mesh, &result, sample(rival)),
            0,
            "no chain vert may land on a rival `mid`",
        );
    }
}

/// A 3×3 quad grid whose centre vertex is lifted to `z = 0.6`, so all four incident
/// quads are **warped** (three corners in `z = 0`, one lifted). Beveling the centre's
/// `+x` and `+y` spokes makes it an `s == 2` junction with a single-face sector (the
/// `+x+y` quad, §7.6.2 case 1) and a two-absorbed-leg sector wrapping the other three
/// quads (§7.6.2 case 3).
///
/// `rotate_first_face` rotates the first quad's vertex cycle to start at the centre.
/// That is the *same* mesh — same faces, same winding, same normals — but it swaps
/// which of the centre's two edges that face reports first, which reverses the
/// direction the corner ring chains in. Every §7.6.2 quantity must be invariant to it.
fn lifted_centre_grid(rotate_first_face: bool) -> (LoopMesh, Vec<VertexKey>) {
    let mut mesh = LoopMesh::new();
    let verts: Vec<VertexKey> = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.6],
        [2.0, 1.0, 0.0],
        [0.0, 2.0, 0.0],
        [1.0, 2.0, 0.0],
        [2.0, 2.0, 0.0],
    ]
    .into_iter()
    .map(|[x, y, z]| mesh.create_vertex(Vec3::new(x, y, z)))
    .collect();
    let first: &[usize] = if rotate_first_face {
        &[4, 3, 0, 1]
    } else {
        &[0, 1, 4, 3]
    };
    for cycle in [first, &[1, 2, 5, 4], &[3, 4, 7, 6], &[4, 5, 8, 7]] {
        let corners: Vec<VertexKey> = cycle.iter().map(|&i| verts[i]).collect();
        mesh.face_create_verts(&corners, FaceCreateFlags::NONE);
    }
    (mesh, verts)
}

/// Bevel the lifted grid's `+x` / `+y` spokes at `offset = 0.2`, one segment.
fn bevel_lifted_centre_grid(rotate_first_face: bool) -> (LoopMesh, BevelResult) {
    let (mut mesh, verts) = lifted_centre_grid(rotate_first_face);
    let geom: Vec<GeomKey> = [5, 7]
        .into_iter()
        .map(|far| {
            mesh.query_edge_exists(verts[4], verts[far])
                .expect("spoke exists")
                .into()
        })
        .collect();
    let result = mesh.bevel(
        &geom,
        BevelOptions {
            offset: 0.2,
            affect: BevelAffect::Edges,
            segments: 1,
            vmesh_method: BevelVmeshMethod::Adj,
            ..Default::default()
        },
    );
    mesh.validate_topology();
    (mesh, result)
}

/// The §7.6.2 case-1 sector normal is the **edge-span cross product**, sign-fixed
/// against the sector face's normal — and that sign fix has to be read in the corner
/// ring's *walk* sense, because the ring's chaining direction is not canonical. Get it
/// wrong on a **warped** face (where the face's Newell normal tilts off the edge-span
/// plane, so the centroid-oriented fallback is not taken) and both offset-line
/// perpendiculars invert, which point-reflects the boundvert through the junction.
///
/// The single-face sector of the lifted grid pins it: the meet belongs at
/// `v + (0.178, 0.178, −0.213)`, *inside* the sector's quad, never at its reflection.
#[test]
fn warped_face_sector_meet_is_not_reflected_through_the_junction() {
    let junction = Vec3::new(1.0, 1.0, 0.6);
    let meet = Vec3::new(1.1778424, 1.1778424, 0.3865892);
    let reflection = junction - (meet - junction);

    for rotate_first_face in [false, true] {
        let (mesh, result) = bevel_lifted_centre_grid(rotate_first_face);
        assert_eq!(
            result_has_vert_near(&mesh, &result, meet),
            1,
            "the single-face sector's boundvert is the edge-span meet, inside its quad",
        );
        assert_eq!(
            result_has_vert_near(&mesh, &result, reflection),
            0,
            "no boundvert may land on the point reflection of the meet through `v`",
        );
    }
}

/// The §7.6.2 face-drop runs on a **case-3** sector: its provisional meet floats off
/// the surface and must be projected onto the `v`-anchored plane of the first sector
/// face whose wedge it lands in. The lifted grid's two-absorbed-leg sector pins it —
/// the walk's *first* face is rejected (its drop sweeps ~53° past its own wedge) and
/// the middle face accepted, which pulls the boundvert down off the junction's own
/// height. Leaving the provisional meet in place keeps `v`'s `z` and floats the
/// boundvert above the surface.
#[test]
fn warped_face_case3_sector_meet_drops_onto_the_middle_face() {
    let dropped = Vec3::new(0.8492861, 0.8492861, 0.5095717);
    let provisional = Vec3::new(0.8221542, 0.8221542, 0.6);

    for rotate_first_face in [false, true] {
        let (mesh, result) = bevel_lifted_centre_grid(rotate_first_face);
        assert_eq!(
            result_has_vert_near(&mesh, &result, dropped),
            1,
            "the case-3 boundvert drops onto the middle face's `v`-anchored plane",
        );
        assert_eq!(
            result_has_vert_near(&mesh, &result, provisional),
            0,
            "no boundvert may keep the un-dropped provisional meet (at `v`'s own height)",
        );
    }
}

/// Both §7.6.2 signed quantities — the case-1 sign fix and the face-drop's in-wedge
/// sweeps — are measured in the corner ring's walk sense, so neither may depend on
/// which way the ring happens to chain. Rotating one face's vertex cycle leaves the
/// mesh identical but reverses that direction; the bevel must not move.
#[test]
fn warped_face_junction_placement_is_invariant_to_the_ring_walk_direction() {
    let (forward, _) = bevel_lifted_centre_grid(false);
    let (reversed, _) = bevel_lifted_centre_grid(true);
    assert_eq!(counts(&forward), counts(&reversed));
    assert!(
        positions_multiset_eq(&forward, &reversed, 1.0e-5),
        "the warped junction's boundverts must not depend on the ring's chaining direction",
    );
}

/// BEVEL.md §8.2.2.1: tier 1 is **unconditional**, so a *mixed* sector — one of the
/// crossed edge's rotational neighbours unbeveled — keeps the vertex-derived bulge
/// even at `s == 3`. The reflex-side chain of the same planar junction pins it: its
/// cell is `start` / `end` on the two outer boundverts with the original vertex as the
/// bulge, a skewed cell whose lone interior sample the §8.2.2.1.1 resample then moves.
#[test]
fn planar_junction_outer_sector_keeps_the_tier1_vertex_bulge() {
    let (mut mesh, centre, rim) = planar_fan(6);
    let geom: Vec<GeomKey> = (0..3)
        .map(|k| {
            mesh.query_edge_exists(centre, rim[k])
                .expect("spoke exists")
                .into()
        })
        .collect();
    let result = mesh.bevel(
        &geom,
        BevelOptions {
            offset: 0.2,
            affect: BevelAffect::Edges,
            segments: 2,
            vmesh_method: BevelVmeshMethod::Adj,
            ..Default::default()
        },
    );
    mesh.validate_topology();

    // The chain crossing the 0° spoke spans the 30° boundvert and the reflex sector's
    // boundvert (`offset / sin 120°` out along the 240° bisector).
    let start = Vec3::new(0.34641016, 0.2, 0.0);
    let end = Vec3::new(-0.11547005, -0.2, 0.0);
    let expected = ProfileCell::from_control_points(start, end, Vec3::ZERO).chain_positions(
        2,
        0.5,
        None,
        ChainEmission::Resampled,
    )[1];
    assert_eq!(
        result_has_vert_near(&mesh, &result, expected),
        1,
        "a mixed sector's bulge is the original vertex — tier 1 does not gate on `s`",
    );
}
