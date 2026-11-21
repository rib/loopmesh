//! The superellipse profile: exponent mapping and chain spacing (BEVEL.md §8.6.2 / §8.6.4).

use super::*;

/// A large finite stand-in for the square exponent `r → ∞` (BEVEL.md §8.6.2).
/// True infinity is impractical, and `profile ≈ 0.95` already maps to `r ≈ 27` —
/// square enough to snap at sampling resolution, so the family stays bounded.
pub(super) const PROFILE_SQUARE_EXPONENT: f32 = 1.0e4;

/// Tolerance for snapping the computed exponent onto the special values that
/// carry closed-form spacing shortcuts (BEVEL.md §8.6.2).
pub(super) const PROFILE_EXPONENT_SNAP: f32 = 1.0e-4;

/// Map the user `profile` scalar (`0..1`) to the superellipse exponent `r`
/// (BEVEL.md §8.6.2): `r = 2·ln(2) / ln(1/profile)`, snapped onto the special
/// exponents whose even-arc-length spacing has a closed form (`1` line, `2`
/// circle, the square / square-in sentinels), and clamped to a finite square
/// sentinel at the top so the family stays bounded.
pub(super) fn profile_to_exponent(profile: f32) -> f32 {
    if profile >= 0.95 {
        return PROFILE_SQUARE_EXPONENT;
    }
    // The square-in razor below is on the *raw exponent*, never on `profile`: `r_raw`
    // decays so slowly that no **normal** positive `profile` trips it, so square-in is
    // reached at `profile = 0` and — via the reciprocal overflowing to `+inf`, which
    // drives `r_raw` to `0` — at the subnormals below `≈2.94e-39`, which is the same
    // answer the sentinel wants there anyway.
    //
    // `+0.0` needs no guard of its own (`1/+0 = +inf`, `ln(+inf) = +inf`, `r_raw = 0`,
    // razor fires). What the guard is actually for is everything the reciprocal path
    // turns into a NaN: `-0.0` gives `-inf` and `ln` of that is NaN, plain negatives
    // give a negative reciprocal and likewise, and a NaN `profile` propagates straight
    // through. Sending all of them to the square-in limit is an implementer choice —
    // BEVEL.md §8.6.2 states no rule off `[0, 1]` — picked so the exponent is total.
    if !(profile > 0.0) {
        return 0.0;
    }
    let exponent = 2.0 * std::f32::consts::LN_2 / (1.0 / profile).ln();
    if (exponent - 2.0).abs() < PROFILE_EXPONENT_SNAP {
        2.0
    } else if (exponent - 1.0).abs() < PROFILE_EXPONENT_SNAP {
        1.0
    } else if exponent < PROFILE_EXPONENT_SNAP {
        0.0
    } else if exponent >= PROFILE_SQUARE_EXPONENT {
        PROFILE_SQUARE_EXPONENT
    } else {
        exponent
    }
}

/// One quadrant of the superellipse `u^r + v^r = 1` solved for `v` at a given
/// `u ∈ [0, 1]` (BEVEL.md §8.6.1). Single-valued and `u ↔ v` symmetric at
/// **every** `r`; the concave band `r < 1` is no exception and gets no separate
/// branch here — `1 − superellipse_v(1 − u, r)` is a *different curve* (the
/// anti-diagonal reflection), and it lives in `working_v` below.
pub(super) fn superellipse_v(u: f32, r: f32) -> f32 {
    let u = u.clamp(0.0, 1.0);
    // Square-in sentinel (`r → 0`, BEVEL.md §8.6.1): the curve collapses to the
    // two legs `(0,1)→(0,0)→(1,0)` through the origin, so `v(u)` is the step
    // `1` at `u = 0`, `0` for any `u > 0` — the concave notch, not a flat top.
    if r <= 0.0 {
        return if u <= 0.0 { 1.0 } else { 0.0 };
    }
    (1.0 - u.powf(r)).max(0.0).powf(1.0 / r)
}

/// The curve the even-chord solver actually walks (BEVEL.md §8.6.4). For `r > 1`
/// it is the superellipse itself; for `r < 1` it is the superellipse's reflection
/// through the **anti-diagonal** `u + v = 1`, whose gradient stays inside
/// `[−1, 1]` on the solved half where the concave original's does not.
///
/// This is a change of coordinates, not a rewrite: everything the solver produces
/// for `r < 1` is in *working* coordinates and must be pushed back through
/// [`anti_diagonal`] before it is emitted.
pub(super) fn working_v(u: f32, r: f32) -> f32 {
    if r > 1.0 {
        superellipse_v(u, r)
    } else {
        1.0 - superellipse_v(1.0 - u, r)
    }
}

/// The anti-diagonal reflection `A(u, v) = (1 − v, 1 − u)` (BEVEL.md §8.6.4).
/// An isometry and its own inverse; it fixes both ring corners, so it preserves a
/// chain's index order — unlike the centre point reflection `(1 − u, 1 − v)`,
/// which lands on the same curve but reverses the chain end-for-end.
pub(super) fn anti_diagonal(point: (f32, f32)) -> (f32, f32) {
    (1.0 - point.1, 1.0 - point.0)
}

/// The straight-chord length between two cell points.
pub(super) fn cell_chord(a: (f32, f32), b: (f32, f32)) -> f32 {
    ((a.0 - b.0).powi(2) + (a.1 - b.1).powi(2)).sqrt()
}

/// Place the `segments + 1` cross-section cell coordinates `(u_k, v_k)` of the
/// `SUPERELLIPSE` profile (BEVEL.md §8.6). The ends are pinned to the cell ring
/// corners `(0, 1)` / `(1, 0)`; the interior is spaced by even chord length along
/// the curve (§8.6.4). The closed-form shortcuts fire at the special exponents
/// (circle / line); any other `r` runs the symmetric chord-equalisation solver.
/// Reduces bit-for-bit to the §8.3.6 circular default at `profile = 0.5`.
pub(super) fn superellipse_chain(profile: f32, segments: u32) -> Vec<(f32, f32)> {
    let n = segments as usize;
    let mut chain = vec![(0.0_f32, 0.0_f32); n + 1];
    chain[0] = (0.0, 1.0);
    chain[n] = (1.0, 0.0);
    if n < 2 {
        return chain;
    }
    let r = profile_to_exponent(profile);

    // Circle (`r = 2`): equal arc-length is equal angle (BEVEL.md §8.6.4 / §8.3.6).
    // Evaluated through the identical sin/cos expression as the historical default
    // path, so the chain is bit-for-bit the §8.3.6 default at `profile = 0.5`.
    if r == 2.0 {
        for k in 1..n {
            let theta = k as f32 * std::f32::consts::FRAC_PI_2 / segments as f32;
            chain[k] = (theta.sin(), theta.cos());
        }
        return chain;
    }
    // Line (`r = 1`): the chord is the curve, so even spacing is even linear
    // interpolation (BEVEL.md §8.6.4).
    if r == 1.0 {
        for k in 1..n {
            let t = k as f32 / segments as f32;
            chain[k] = (t, 1.0 - t);
        }
        return chain;
    }

    // Square (`r = ∞` sentinel) and square-in (`r = 0` sentinel): the
    // cross-section is two straight legs meeting at a corner on the `u = v`
    // diagonal, so even chord spacing is linear along each leg with the sample
    // budget split at the diagonal (BEVEL.md §8.6.4). Bypasses the iterative
    // solver — the general chord-equaliser cannot resolve the sharp corner.
    if r == PROFILE_SQUARE_EXPONENT {
        superellipse_chain_legs(true, segments, &mut chain);
        return chain;
    }
    if r == 0.0 {
        superellipse_chain_legs(false, segments, &mut chain);
        return chain;
    }

    superellipse_chain_general(r, segments, &mut chain);
    chain
}

/// Closed-form leg-linear spacing for the square (`r = ∞`) and square-in
/// (`r = 0`) sentinels (BEVEL.md §8.6.4). The curve is two length-1 legs meeting
/// at a right-angle corner on the `u = v` diagonal — square runs
/// `(0,1)→(1,1)→(1,0)`, square-in runs `(0,1)→(0,0)→(1,0)`. Samples are placed
/// linearly along the first leg (from the start corner toward the diagonal
/// corner) and mirrored across `u = v` to fill the second leg.
///
/// For **even** `segments` a sample lands exactly on the diagonal corner and
/// each leg carries `segments/2` chords of length `2/segments`. For **odd**
/// `segments` no sample sits on the corner; the central chord straddles it, and
/// to keep that diagonal-crossing chord the same length as the leg chords the
/// straddle consumes only a `√2/2`-weighted share of arc on each leg — so with
/// `p = (segments−1)/2` full leg chords, the leg-chord arc length solves
/// `p·L + (√2/2)·L = 1`, i.e. `L = 1 / (p + √2/2)`.
pub(super) fn superellipse_chain_legs(square: bool, segments: u32, chain: &mut [(f32, f32)]) {
    let n = segments as usize;
    let (start, corner) = if square {
        ((0.0_f32, 1.0_f32), (1.0_f32, 1.0_f32))
    } else {
        ((0.0_f32, 1.0_f32), (0.0_f32, 0.0_f32))
    };
    let leg = (corner.0 - start.0, corner.1 - start.1);

    let leg_chord = if n % 2 == 0 {
        2.0 / segments as f32
    } else {
        let full_chords = ((n - 1) / 2) as f32;
        1.0 / (full_chords + std::f32::consts::FRAC_1_SQRT_2)
    };

    for i in 0..=n / 2 {
        let t = (i as f32 * leg_chord).min(1.0);
        let point = (start.0 + t * leg.0, start.1 + t * leg.1);
        chain[i] = point;
        chain[n - i] = (point.1, point.0);
    }
}

/// The general even-chord spacing solver (BEVEL.md §8.6.4) for exponents without
/// a closed-form shortcut. Only the half-chain from the start corner up to — but
/// excluding — the `u = v` diagonal is solved (indices `0 ..= last_solved`); the
/// second half is its `u = v` mirror and, at even `segments`, the central sample
/// is the closed-form diagonal crossing `(m, m)`.
///
/// Every step runs in **working** coordinates ([`working_v`]): for `r > 1` those
/// are the real ones, for `r < 1` they are the anti-diagonal reflection. The
/// finished chain is pushed back through [`anti_diagonal`] in that case, as the
/// very last act — the `u = v` half-fill happens first, on working coordinates.
/// Writes the interior `(u_k, v_k)` into `chain[1 ..= segments-1]`.
pub(super) fn superellipse_chain_general(r: f32, segments: u32, chain: &mut [(f32, f32)]) {
    let n = segments as usize;
    let concave = r < 1.0;
    // The WORKING curve's `u = v` crossing. `A` fixes that diagonal setwise, so
    // un-mapping `(m, m)` lands on the real curve's crossing `(1/2)^(1/r)` either
    // way — but the solver measures its closing chord against the working one.
    let real_diagonal = 0.5_f32.powf(1.0 / r);
    let working_diagonal = if concave {
        1.0 - real_diagonal
    } else {
        real_diagonal
    };

    // Stop one short of the diagonal sample: at even `segments` index `n/2` IS
    // `(m, m)`, supplied in closed form and never solved for (BEVEL.md §8.6.4
    // step 1).
    let last_solved = n.div_ceil(2) - 1;

    // Seed: linear spacing in `u` over `[0, m]` (BEVEL.md §8.6.4 step 2).
    let mut samples: Vec<(f32, f32)> = (0..=last_solved)
        .map(|i| {
            let u = i as f32 * (2.0 * working_diagonal / segments as f32);
            (u, working_v(u, r))
        })
        .collect();
    samples[0] = (0.0, 1.0);

    // Advance from `prev` by chord-distance `target` along the working curve
    // toward `u = 1` (BEVEL.md §8.6.4 step 3). The chord grows monotonically with
    // `u`, so a bisection on `[u_prev, 1]` is robust; if `target` cannot be
    // reached the sample clamps to the end corner.
    let advance = |prev: (f32, f32), target: f32| -> (f32, f32) {
        if cell_chord((1.0, 0.0), prev) <= target {
            return (1.0, 0.0);
        }
        let mut lo = prev.0;
        let mut hi = 1.0_f32;
        for _ in 0..60 {
            let mid = 0.5 * (lo + hi);
            if cell_chord((mid, working_v(mid, r)), prev) > target {
                hi = mid;
            } else {
                lo = mid;
            }
        }
        let u = 0.5 * (lo + hi);
        (u, working_v(u, r))
    };

    // Fixed-point equalisation over the solved half, capped at 10 sweeps
    // (BEVEL.md §8.6.4 step 4). The closing term ties the last solved sample to
    // the diagonal: the whole chord to `(m, m)` at even `segments`, and at odd
    // `segments` the perpendicular distance to `u = v` — half the straddling
    // chord, so that at convergence the straddle itself matches `target_chord`.
    for _ in 0..10 {
        let last = samples[last_solved];
        let closing = if n % 2 == 0 {
            cell_chord((working_diagonal, working_diagonal), last)
        } else {
            (last.1 - last.0).abs() * std::f32::consts::FRAC_1_SQRT_2
        };
        if last_solved == 0 {
            break;
        }
        let interior: Vec<f32> = (0..last_solved)
            .map(|i| cell_chord(samples[i], samples[i + 1]))
            .collect();
        let chord_count = if n % 2 == 0 {
            last_solved as f32 + 1.0
        } else {
            last_solved as f32 + 0.5
        };
        let target_chord = (interior.iter().sum::<f32>() + closing) / chord_count;

        // The spread is taken over the INTERIOR chords only — the ones this sweep
        // measured, before the re-walk equalises them by construction — while
        // `target_chord` is the average that also carries the closing term. That
        // asymmetry is the whole content of the test (BEVEL.md §8.6.4 step 4).
        let converged = interior.iter().all(|d| (d - target_chord).abs() < 1.0e-7);
        if converged {
            break;
        }

        for i in 1..=last_solved {
            samples[i] = advance(samples[i - 1], target_chord);
        }
    }

    let mut full = vec![(0.0_f32, 0.0_f32); n + 1];
    full[..=last_solved].copy_from_slice(&samples);
    for i in 0..=last_solved {
        full[n - i] = (samples[i].1, samples[i].0);
    }
    if n % 2 == 0 {
        full[n / 2] = (working_diagonal, working_diagonal);
    }

    if concave {
        for point in full.iter_mut() {
            *point = anti_diagonal(*point);
        }
    }
    chain[1..n].copy_from_slice(&full[1..n]);
}
