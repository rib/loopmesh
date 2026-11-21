use super::*;

/// BEVEL.md §8.6.4 / §8.3.6: the circle shortcut (`profile = 0.5`, `r = 2`) is
/// pure angle spacing `(sin θ_k, cos θ_k)`, reproducing the §8.3.6 default
/// bit-for-bit. The pure sampler is checked directly so the reduction is pinned
/// independently of the cell mapping.
#[test]
fn superellipse_chain_circle_is_angle_spacing() {
    let chain = superellipse_chain(0.5, 2);
    assert_eq!(chain[0], (0.0, 1.0));
    assert_eq!(chain[2], (1.0, 0.0));
    let theta = std::f32::consts::FRAC_PI_2 / 2.0;
    assert_eq!(chain[1], (theta.sin(), theta.cos()));

    let chain4 = superellipse_chain(0.5, 4);
    for k in 1..4 {
        let theta = k as f32 * std::f32::consts::FRAC_PI_2 / 4.0;
        assert_eq!(chain4[k], (theta.sin(), theta.cos()));
    }
}

/// The unsnapped `r = 2·ln(2) / ln(1/profile)` of BEVEL.md §8.6.2, so the snap tests
/// below can say what the map would have returned without its snap table.
fn raw_profile_exponent(profile: f32) -> f32 {
    2.0 * std::f32::consts::LN_2 / (1.0 / profile).ln()
}

/// BEVEL.md §8.6.2 `#exponent-snap`: the whole snap table, in the order the guide
/// states it. Exactly one entry is tested on `profile` — the wide `profile ≥ 0.95`
/// square gate; the circle, line and square-in entries are all tested on the raw
/// exponent, and the circle / line windows exist so `r == 2.0` / `r == 1.0` are
/// reachable as **exact** float equalities for the `==`-keyed branches downstream.
///
/// The anchors `profile = 0.5` / `0.25` alone would not pin those two windows: in
/// f32 the raw map already lands on exactly `2.0` / `1.0` there, so both branches
/// could be deleted and the anchors would still pass. The off-anchor samples are
/// the ones that hold the windows open.
#[test]
fn profile_to_exponent_follows_the_snap_table() {
    let raw = raw_profile_exponent;

    for profile in [0.95_f32, 0.96, 0.99, 1.0] {
        assert_eq!(
            profile_to_exponent(profile),
            PROFILE_SQUARE_EXPONENT,
            "profile {profile} is inside the wide square gate",
        );
    }
    assert_ne!(
        profile_to_exponent(0.949),
        PROFILE_SQUARE_EXPONENT,
        "the square gate opens at 0.95, not below it",
    );

    assert_eq!(profile_to_exponent(0.5), 2.0, "circle is exact");
    assert_eq!(profile_to_exponent(0.25), 1.0, "line is exact");
    assert_eq!(profile_to_exponent(0.0), 0.0, "square-in sentinel");

    for (profile, special) in [(0.50001_f32, 2.0_f32), (0.2500086, 1.0)] {
        assert_ne!(
            raw(profile),
            special,
            "profile {profile} must sit off the anchor for this to test the snap",
        );
        assert!(
            (raw(profile) - special).abs() < PROFILE_EXPONENT_SNAP,
            "profile {profile} is inside the {special} window: r_raw = {}",
            raw(profile),
        );
        assert_eq!(
            profile_to_exponent(profile),
            special,
            "profile {profile} snaps onto {special} exactly",
        );
    }

    for profile in [0.1_f32, 0.3, 0.7, 0.9] {
        let exponent = profile_to_exponent(profile);
        assert_eq!(exponent, raw(profile), "profile {profile} is not snapped");
        assert!(exponent > 0.0 && exponent < PROFILE_SQUARE_EXPONENT);
    }
}

/// BEVEL.md §8.6.2 `#exponent-snap`: the square-in entry is a **razor on `r_raw`**,
/// not a band on `profile`. `r_raw = 2·ln(2) / ln(1/profile)` decays so slowly that
/// no **normal** positive `profile` reaches `1e-4` — at `profile = 1e-4` it is still
/// `≈ 0.151`, three decades above the razor, and even at the smallest normal it is
/// `≈ 0.0159` — so square-in is reached at `profile = 0` and everything normal above
/// it stays genuinely curved. (Only the subnormals below `≈2.94e-39` also reach it,
/// where `1/profile` overflows to `+inf`; the sentinel is the right answer there.)
///
/// A `profile`-side gate (`profile ≤ 1e-4 → r := 0`) would collapse this whole
/// window onto the sentinel, and with it the leg-linear chain and the (f.4b) arm of
/// the [V-ADJ] grid. The exponent equalities below are what fail if one returns.
#[test]
fn profile_to_exponent_square_in_razor_is_on_the_raw_exponent() {
    let raw = raw_profile_exponent;

    let mut previous = 0.0_f32;
    for (profile, expected) in [
        (1.0e-5_f32, 0.1204_f32),
        (3.0e-5, 0.1331),
        (5.0e-5, 0.1400),
        (9.0e-5, 0.1488),
        (1.0e-4, 0.1505),
    ] {
        let exponent = profile_to_exponent(profile);
        assert_eq!(
            exponent,
            raw(profile),
            "profile {profile} keeps its curved exponent",
        );
        assert!(
            (exponent - expected).abs() < 1.0e-3,
            "profile {profile} maps to r ≈ {expected}, got {exponent}",
        );
        assert!(exponent > previous, "the map stays monotonic in profile");
        previous = exponent;

        // The consequence downstream: a curved `r` runs the chord-equalisation
        // solver, whose lone `segments = 2` sample sits just off the origin on the
        // diagonal — where the square-in sentinel's leg-linear chain puts the sharp
        // corner `(0, 0)` itself.
        let sample = superellipse_chain(profile, 2)[1];
        assert!(
            sample.0 > 0.0 && sample.1 > 0.0,
            "profile {profile} is not the square-in corner: {sample:?}",
        );
        assert!(sample.0 < 0.05 && sample.1 < 0.05, "but is deeply concave");
    }

    assert_eq!(
        superellipse_chain(0.0, 2)[1],
        (0.0, 0.0),
        "`profile = 0` alone is the square-in corner",
    );
}

/// `profile_to_exponent` is **total**: every `f32`, in-domain or not, maps to a
/// finite exponent, so the `r`-keyed consumers (the ring sampler, the `== 1.0` /
/// `== 2.0` branches) never see a NaN. Off-domain inputs — `-0.0`, negatives, NaN —
/// take the square-in limit, which is an *implementer-invented* choice: BEVEL.md
/// §8.6.2 states no rule off `[0, 1]`, and picking one only keeps the map total.
///
/// This is a property of the exponent alone, **not** of the operator. The centre
/// fullness ([`adj_fullness`], §8.5.5.5(f.3)) evaluates its regression on the raw
/// `profile` scalar, so `profile = NaN` still yields a NaN fullness seed; nothing
/// here contains that, and containing it would mean sanitising `profile` where
/// [`BevelOptions`] is read, not here.
#[test]
fn profile_to_exponent_is_finite_off_the_unit_interval() {
    for profile in [0.0_f32, -0.0, -0.5, f32::NAN, f32::NEG_INFINITY] {
        assert_eq!(
            profile_to_exponent(profile),
            0.0,
            "profile {profile} takes the square-in limit",
        );
    }
    assert_eq!(profile_to_exponent(f32::INFINITY), PROFILE_SQUARE_EXPONENT);
    for profile in [1.0e-30_f32, 1.0e-6, 0.5, 0.94] {
        assert!(
            profile_to_exponent(profile).is_finite(),
            "profile {profile} maps to a finite exponent",
        );
    }

    // The scope of the guarantee, pinned so the docstring above cannot rot into a
    // crate-wide NaN-safety claim.
    assert!(adj_fullness(f32::NAN, 2).is_nan());
}

/// BEVEL.md §8.6.2 / §8.6.4: the flat profile (`profile = 0.25`, `r = 1`) is the
/// straight chord `u + v = 1`, so even spacing is even linear interpolation. At
/// `segments = 2` the lone interior sample is the chord midpoint `(0.5, 0.5)`.
#[test]
fn superellipse_chain_line_is_chord_midpoint() {
    let chain = superellipse_chain(0.25, 2);
    assert_eq!(chain[1], (0.5, 0.5));

    let chain4 = superellipse_chain(0.25, 4);
    for k in 0..=4 {
        let t = k as f32 / 4.0;
        assert!((chain4[k].0 - t).abs() < 1.0e-6 && (chain4[k].1 - (1.0 - t)).abs() < 1.0e-6);
    }
}

/// BEVEL.md §8.6.1 / §8.6.4: a general exponent (`profile = 0.7`, `r ≈ 3.89`)
/// runs the chord-equalisation solver. At `segments = 2` the single interior
/// sample sits on the `u = v` diagonal at `m = (1/2)^(1/r)`, which lies on the
/// superellipse `u^r + v^r = 1` and bulges past both the circle (0.5) and the
/// flat chord (0.25) samples.
#[test]
fn superellipse_chain_general_lands_on_superellipse() {
    let chain = superellipse_chain(0.7, 2);
    let (u, v) = chain[1];
    // On the u = v diagonal.
    assert!((u - v).abs() < 1.0e-6);
    // On the superellipse: u^r + v^r = 1.
    let r = profile_to_exponent(0.7);
    assert!((u.powf(r) + v.powf(r) - 1.0).abs() < 1.0e-4);
    // Bulges past the circle (0.707) and the flat chord (0.5).
    let circle_u = superellipse_chain(0.5, 2)[1].0;
    assert!(u > circle_u);
    assert!(u > 0.5);

    // The general solver equalises chords for a larger even count too.
    let chain4 = superellipse_chain(0.7, 4);
    let chords: Vec<f32> = (0..4)
        .map(|k| cell_chord(chain4[k], chain4[k + 1]))
        .collect();
    let max = chords.iter().cloned().fold(0.0_f32, f32::max);
    let min = chords.iter().cloned().fold(f32::INFINITY, f32::min);
    assert!(
        (max - min) < 1.0e-3,
        "general solver equalises chords: {chords:?}"
    );
    // Symmetric about the diagonal: sample k mirrors sample n−k.
    assert!((chain4[1].0 - chain4[3].1).abs() < 1.0e-5);
}

/// BEVEL.md §8.6.1 / §8.6.4: the square (`profile = 1.0`, `r = ∞` sentinel)
/// and square-in (`profile = 0`, `r = 0` sentinel) cross-sections are two
/// straight legs meeting at a corner on the `u = v` diagonal. Even chord
/// spacing lands each interior sample on a leg (never off the two edges), the
/// diagonal corner is a sample at even `segments`, and every chord is equal
/// length — the closed-form leg-linear shortcut, not the iterative solver.
#[test]
fn superellipse_chain_square_is_leg_linear_even() {
    // Square: legs (0,1)→(1,1)→(1,0), corner on the diagonal at (1,1).
    let chain = superellipse_chain(1.0, 4);
    assert_eq!(chain[0], (0.0, 1.0));
    assert_eq!(chain[4], (1.0, 0.0));
    // Even segments → the central sample is the diagonal corner.
    assert!((chain[2].0 - 1.0).abs() < 1.0e-6 && (chain[2].1 - 1.0).abs() < 1.0e-6);
    // Interior samples lie on the legs: the first-leg point has v == 1 (top
    // edge), the mirrored second-leg point has u == 1 (right edge).
    assert!(
        (chain[1].1 - 1.0).abs() < 1.0e-6,
        "chain[1] on top leg (v=1)"
    );
    assert!(
        (chain[3].0 - 1.0).abs() < 1.0e-6,
        "chain[3] on right leg (u=1)"
    );
    // Equal chords: each of the 4 chords is 2/segments = 0.5 long.
    for k in 0..4 {
        assert!((cell_chord(chain[k], chain[k + 1]) - 0.5).abs() < 1.0e-6);
    }

    // Square-in: legs (0,1)→(0,0)→(1,0), corner on the diagonal at (0,0).
    let chain_in = superellipse_chain(0.0, 4);
    assert!((chain_in[2].0).abs() < 1.0e-6 && (chain_in[2].1).abs() < 1.0e-6);
    assert!((chain_in[1].0).abs() < 1.0e-6, "chain[1] on left leg (u=0)");
    assert!(
        (chain_in[3].1).abs() < 1.0e-6,
        "chain[3] on bottom leg (v=0)"
    );
    for k in 0..4 {
        assert!((cell_chord(chain_in[k], chain_in[k + 1]) - 0.5).abs() < 1.0e-6);
    }
}

/// BEVEL.md §8.6.4: at **odd** `segments` no sample sits on the diagonal
/// corner; the central chord straddles it and is given a `√2/2`-weighted arc
/// share so the diagonal-crossing chord matches the leg chords in length. All
/// `segments` chords must therefore stay equal, and the two samples flanking
/// the corner sit a `√2/2`-scaled leg-chord away from it.
#[test]
fn superellipse_chain_square_is_leg_linear_odd() {
    for &profile in &[1.0_f32, 0.0_f32] {
        let chain = superellipse_chain(profile, 3);
        assert_eq!(chain[0], (0.0, 1.0));
        assert_eq!(chain[3], (1.0, 0.0));
        let chords: Vec<f32> = (0..3).map(|k| cell_chord(chain[k], chain[k + 1])).collect();
        let max = chords.iter().cloned().fold(0.0_f32, f32::max);
        let min = chords.iter().cloned().fold(f32::INFINITY, f32::min);
        assert!(
            (max - min) < 1.0e-6,
            "odd square chords equal (√2/2 straddle) for profile {profile}: {chords:?}",
        );
        // The interior samples flank (but never sit on) the diagonal corner.
        assert!((chain[1].0 - chain[1].1).abs() > 1.0e-3);
        assert!((chain[2].0 - chain[2].1).abs() > 1.0e-3);
        // Full leg length 1 = p·L + (√2/2)·L with p = 1, so L = 1/(1+√2/2).
        let expected = 1.0 / (1.0 + std::f32::consts::FRAC_1_SQRT_2);
        assert!((chords[0] - expected).abs() < 1.0e-6);
    }
}

/// BEVEL.md §8.6.1: the square-in sentinel `superellipse_v(u, 0)` is the
/// concave-notch step (`1` at `u = 0`, `0` for `u > 0`) — NOT the flat
/// `v ≡ 1` top the naive `r < 1` mirror form produced. The chain must ride
/// the two legs through the origin, never a flat top edge.
#[test]
fn superellipse_square_in_is_concave_not_flat_top() {
    assert_eq!(superellipse_v(0.0, 0.0), 1.0);
    assert_eq!(superellipse_v(0.25, 0.0), 0.0);
    assert_eq!(superellipse_v(0.5, 0.0), 0.0);
    assert_eq!(superellipse_v(1.0, 0.0), 0.0);

    // No interior sample sits on a flat v = 1 top edge (that would be square,
    // not square-in): every interior square-in sample has v < 1.
    let chain = superellipse_chain(0.0, 4);
    for k in 1..4 {
        assert!(
            chain[k].1 < 1.0 - 1.0e-6,
            "square-in sample {k} off flat top"
        );
    }
}

/// BEVEL.md §8.6.4 step 5: a **concave** exponent (`profile < 0.25`) is chord-equalised
/// on its reflection through the anti-diagonal `u + v = 1`, and the finished chain is
/// pushed back through that reflection before emission. So every emitted sample lies on
/// the true superellipse `u^r + v^r = 1` — the working curve, which does not, never
/// reaches the ring. The `segments = 4` chain is the guide's worked witness; the
/// diagonal sample is `(√profile, √profile)`, not its complement.
#[test]
fn superellipse_chain_concave_is_un_mapped_onto_the_true_curve() {
    let profile = 0.1_f32;
    let r = profile_to_exponent(profile);

    for segments in 2..=7 {
        let chain = superellipse_chain(profile, segments);
        for (k, &(u, v)) in chain.iter().enumerate() {
            assert!(
                (u.powf(r) + v.powf(r) - 1.0).abs() < 1.0e-4,
                "segments {segments} sample {k} = ({u}, {v}) is off u^r + v^r = 1",
            );
        }
        let chords: Vec<f32> = (0..segments as usize)
            .map(|k| cell_chord(chain[k], chain[k + 1]))
            .collect();
        let spread = chords.iter().cloned().fold(0.0_f32, f32::max)
            - chords.iter().cloned().fold(f32::INFINITY, f32::min);
        assert!(spread < 1.0e-4, "segments {segments} chords {chords:?}");
    }

    let chain = superellipse_chain(profile, 4);
    for (k, expected) in [
        (1, (0.09565, 0.62923)),
        (2, (0.31623, 0.31623)),
        (3, (0.62923, 0.09565)),
    ] {
        assert!(
            (chain[k].0 - expected.0).abs() < 1.0e-4 && (chain[k].1 - expected.1).abs() < 1.0e-4,
            "worked witness at k = {k}: got {:?}, want {expected:?}",
            chain[k],
        );
    }

    // The sharpest single test (§8.6.4's fingerprint): at `segments = 2` the lone
    // interior sample IS the diagonal one, so a missing un-map shows there in closed
    // form — `√profile` against its complement `1 − √profile`.
    let sharp = superellipse_chain(profile, 2)[1];
    let root = profile.sqrt();
    assert!((sharp.0 - root).abs() < 1.0e-5 && (sharp.1 - root).abs() < 1.0e-5);
    assert!(
        (sharp.0 - (1.0 - root)).abs() > 0.3,
        "not the working-curve crossing",
    );
}

/// BEVEL.md §8.6.4 (#concave-unmap-fingerprint): `A(u,v) − (u,v) = (1 − u − v)·(1,1)`, so
/// the anti-diagonal un-map fixes exactly the chord `u + v = 1` — which a concave profile
/// meets only at its two pinned corners. Hence `segments = 1` is the *only* count where
/// the un-map is invisible, and every interior sample of every longer chain sits strictly
/// inside the chord (so every one of them moves).
#[test]
fn superellipse_chain_concave_un_map_is_a_fixed_point_only_at_segments_one() {
    let profile = 0.1_f32;
    let single = superellipse_chain(profile, 1);
    assert_eq!(single, vec![(0.0, 1.0), (1.0, 0.0)]);
    for &corner in single.iter() {
        assert_eq!(anti_diagonal(corner), corner, "A fixes the ring corners");
    }

    for segments in 2..=6 {
        let chain = superellipse_chain(profile, segments);
        for k in 1..segments as usize {
            let (u, v) = chain[k];
            assert!(
                u + v < 1.0 - 1.0e-3,
                "segments {segments} sample {k} = ({u}, {v}) is off the chord, so it moves",
            );
        }
    }
}

/// BEVEL.md §8.6.1 (#not-a-branch): `1 − superellipse_v(1 − u, r)` is the anti-diagonal
/// reflection of the superellipse — a *different curve*, not a stability rewrite of it.
/// `working_v` is that curve for `r < 1` and the superellipse itself for `r > 1`.
#[test]
fn working_curve_is_the_anti_diagonal_reflection_below_one() {
    let r = profile_to_exponent(0.1);
    assert!(r < 1.0);
    let u = 0.09565_f32;
    assert!((superellipse_v(u, r) - 0.62923).abs() < 1.0e-4);
    assert!((working_v(u, r) - (1.0 - superellipse_v(1.0 - u, r))).abs() < 1.0e-6);
    // The two disagree by more than half the cell — they are not two spellings of one.
    assert!((working_v(u, r) - superellipse_v(u, r)).abs() > 0.3);

    // Above the line exponent the solver walks the superellipse itself.
    let convex = profile_to_exponent(0.7);
    assert!(convex > 1.0);
    for u in [0.1_f32, 0.4, 0.75] {
        assert_eq!(working_v(u, convex), superellipse_v(u, convex));
    }
}

/// BEVEL.md §8.5.5.5(c) (#refinement-consistent-exponents): even-**chord** spacing is
/// refinement-consistent at exactly four exponents — line, circle, square, square-in —
/// so only there does a stride of the `2m` chain equal a fresh solve at `m`. That is why
/// the [V-ADJ] cascade must stride its one doubled chain rather than resample per round,
/// and why a circular-only test can never discover the rule.
#[test]
fn even_chord_spacing_is_refinement_consistent_only_at_the_polyline_exponents() {
    let stride_matches = |profile: f32| {
        let dense = superellipse_chain(profile, 8);
        let coarse = superellipse_chain(profile, 4);
        (0..=4).all(|k| cell_chord(dense[2 * k], coarse[k]) < 1.0e-5)
    };
    for &profile in &[0.25_f32, 0.5, 1.0, 0.0] {
        assert!(
            stride_matches(profile),
            "refinement-consistent at {profile}",
        );
    }
    for &profile in &[0.3_f32, 0.7, 0.9, 0.1] {
        assert!(
            !stride_matches(profile),
            "NOT refinement-consistent at {profile}",
        );
    }
}
