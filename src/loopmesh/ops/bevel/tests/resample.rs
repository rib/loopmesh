use super::*;

/// BEVEL.md §8.2.2.1.1: the source resolution is the next power of two at or above
/// `max(4, segments)` (the fill's control cage is 2 wide and always doubles once), and
/// the resample is the exact identity when the source lands on the target — so
/// `4, 8, 16, …` emit the raw chain and nothing else does.
#[test]
fn boundary_resample_source_resolution_and_identity() {
    assert_eq!(boundary_source_resolution(2), 4);
    assert_eq!(boundary_source_resolution(3), 4);
    assert_eq!(boundary_source_resolution(4), 4);
    assert_eq!(boundary_source_resolution(5), 8);
    assert_eq!(boundary_source_resolution(8), 8);
    assert_eq!(boundary_source_resolution(9), 16);

    // The skewed arc-flanking cell of the §8.2.2.1.1 worked trace.
    let apex = Vec3::new(0.0, 0.0, 1.0);
    let rim = Vec3::new(-0.70711, 0.70711, 0.0);
    let cell = ProfileCell::new(
        Vec3::new(-0.19194, 0.46338, 0.45712),
        Vec3::new(-0.36871, 0.24371, 0.42679),
        apex,
        (rim - apex).normalize(),
    );
    assert_eq!(
        cell.chain_positions(4, 0.5, None, ChainEmission::Resampled),
        cell.chain_positions(4, 0.5, None, ChainEmission::Raw),
        "segments = 4 lands on the source resolution, so the resample is the identity",
    );
    assert_ne!(
        cell.chain_positions(3, 0.5, None, ChainEmission::Resampled),
        cell.chain_positions(3, 0.5, None, ChainEmission::Raw),
        "segments = 3 reads back off the 4-segment profile, moving the interior samples",
    );
}

/// BEVEL.md §8.9.5.4(c)/(g).3: a `spread = 0` inner miter mints all three of its arc
/// anchors on one point, and the resulting collapsed chain reaches §8.2.2.1.1's
/// zero-total guard **only at a power-of-two count**. The two-term interpolation
/// `(1 − t)·start + t·end` returns a coincident end bit-exactly only at dyadic `t`, so a
/// non-power-of-two count — exactly the set on which the [V-ADJ] down-resample runs —
/// leaves ulp-scale but strictly positive chords and an ordinary normalised table.
///
/// This is the load-bearing arithmetic in the whole operator: the one-term
/// `start + t·(end − start)` is algebraically the same and bit-inert on coincident ends,
/// and swapping it in silently rebuilds the fallback-on-both-axes reading §8.9.5.4(c)
/// retires.
#[test]
fn zero_spread_arc_chain_takes_the_fallback_table_only_at_a_power_of_two_count() {
    let meet = Vec3::new(0.2, 0.2, 0.0);
    let arc_chain = |segments: u32| arc_chain_positions(meet, meet, meet, 0.5, segments);

    assert_eq!(
        meet.lerp(meet, 1.0 / 3.0),
        meet,
        "the one-term form is bit-inert on coincident ends — that is the trap, and the \
         reason the two-term spelling is part of the contract",
    );

    for segments in [2u32, 4, 8] {
        let chain = arc_chain(segments);
        assert!(
            chain.iter().all(|&sample| sample == meet),
            "every k / {segments} is dyadic, so the whole chain is bit-exactly the meet \
             point: {chain:?}",
        );
        let mut fallback = vec![0.0; segments as usize + 1];
        *fallback.last_mut().unwrap() = 1.0;
        assert_eq!(normalised_chord_fractions(&chain), fallback);
    }

    let chain = arc_chain(3);
    assert_ne!(
        chain[1], meet,
        "`t = 1/3` is not dyadic, so the sample lands an ulp off the coincident point",
    );
    assert_eq!(
        normalised_chord_fractions(&chain),
        vec![0.0, 0.5, 1.0, 1.0],
        "§8.9.5.4(c)'s tabulated target table at this meet point",
    );

    // The source chain is materialised at the doubling resolution, which is a power of
    // two by construction — so the *pair* of tables the read-back consumes is a genuine
    // fallback against an ordinary one, and the interior lands inside the last source
    // span rather than at index 0.
    let source = arc_chain(boundary_source_resolution(3));
    assert!(source.iter().all(|&sample| sample == meet));
    assert_eq!(
        source_position_values(&chord_fraction_source_positions(&chain, &source)),
        vec![0.0, 3.5, 4.0, 4.0],
    );
    assert_ne!(
        source_position_values(&chord_fraction_source_positions(&chain, &source)),
        vec![0.0, 0.0, 0.0, 4.0],
        "the fallback-on-both-axes reading is the retired premise",
    );

    // A profile cell whose frame declines emits through the same rule — the arc is not a
    // special case of it (§8.2.2.1 clause (1)).
    let declined = ProfileCell::from_control_points(meet, meet, meet);
    assert!(declined.emits_straight_chord());
    assert_eq!(declined.raw_chain_positions(3, 0.5, None), chain);
}

/// BEVEL.md §8.2.2.1.1's zero-length-chain fallback: the guard is an **exact** test on
/// the accumulated chord length, and when it fails the table is `[0, …, 0, 1]` — not
/// left undivided, not filled uniformly, and not a signal to bail out of the resample.
///
/// The three readings the guide separates are all pinned here, because they are
/// indistinguishable on the degenerate chain's own samples (which coincide whatever the
/// table says) and only diverge once the table is consumed as a read-back
/// parametrisation.
#[test]
fn degenerate_chord_table_is_zero_until_the_last_step_not_uniform_and_not_the_raw_chain() {
    let point = Vec3::new(0.3, -0.7, 0.2);
    let degenerate = vec![point; 4];

    assert_eq!(
        normalised_chord_fractions(&degenerate),
        vec![0.0, 0.0, 0.0, 1.0],
        "a chain with zero total chord has no extent until its very last step",
    );
    assert_ne!(
        normalised_chord_fractions(&degenerate),
        vec![0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0],
        "the uniform table is a different rule, not the natural reading of this one",
    );

    // The guard is exact: a chain that is merely *short* still carries real fractions,
    // and widening the test into a length floor would claim this one too.
    let stride = 1.0e-8;
    let short = vec![
        Vec3::ZERO,
        Vec3::X * stride,
        Vec3::X * (3.0 * stride),
        Vec3::X * (4.0 * stride),
    ];
    assert_eq!(
        normalised_chord_fractions(&short),
        vec![0.0, 0.25, 0.75, 1.0],
        "the total-length guard is exact zero, not an epsilon",
    );

    // Read out as fractional source indices, the table sends every interior sample to
    // the source polyline's `start` end and only the last one to its far end.
    let source = vec![point; 5];
    assert_eq!(
        source_position_values(&chord_fraction_source_positions(&degenerate, &source)),
        vec![0.0, 0.0, 0.0, 4.0],
        "every interior query of a degenerate chain reads the source at index 0",
    );
    assert_ne!(
        source_position_values(&chord_fraction_source_positions(&degenerate, &source)),
        vec![0.0, 4.0 / 3.0, 8.0 / 3.0, 4.0],
        "uniform index fractions are the mis-reading this table exists to rule out",
    );
    // The far end is the upper-endpoint spelling: the index never reaches `4`, which is
    // why a saturation test written as `index == source.len() − 1` reports backwards.
    assert_eq!(
        *chord_fraction_source_positions(&degenerate, &source)
            .last()
            .unwrap(),
        SourcePosition {
            index: 3,
            remainder: 1.0,
        },
    );

    // On the chain's own samples the fallback is invisible — which is exactly why the
    // "return the raw chain unchanged" reading survives so easily.
    assert_eq!(chord_fraction_resample(&degenerate, &source), degenerate);
}

/// BEVEL.md §8.5.5.5(d)'s `locate(q, g)`, stated as the definition rather than as a span
/// search: the **least** position at which the table's piecewise-linear interpolant
/// reaches the query.
///
/// Every block below is a **spec fence, not a regression fence**: each pins a clause of
/// the definition that a scan-shaped spelling *can* get wrong — the plateau's index, the
/// collapsed head's missing `0/0`, the far end's spelling — but a span search can also
/// arrive at all three by other means, so passing them does not by itself distinguish
/// the two spellings. The case that does is
/// `a_hair_thin_final_chord_keeps_its_full_remainder_and_leaves_the_rim_on_the_polyline`.
///
/// All of it is asserted in the pair frame — a value-form assertion would pass on the
/// plateau case whichever index the implementation chose, since the two spellings of an
/// integral position denote the same number.
#[test]
fn locate_takes_the_least_position_and_needs_no_guard_at_a_collapsed_head() {
    // A plateau resolves to its FIRST index, and arrives as the upper endpoint of the
    // span below it.
    let plateau = [0.0, 0.25, 0.5, 0.5, 0.5, 1.0];
    assert_eq!(
        SourcePosition::locate(0.5, &plateau),
        SourcePosition {
            index: 1,
            remainder: 1.0,
        },
        "position 2, the plateau's first index — not 4, its last, and not inside it",
    );
    assert_eq!(SourcePosition::locate(0.5, &plateau).value(), 2.0);

    // `locate(0, g) = 0` on every table, including one whose first span is collapsed and
    // whose across-span quotient would be `0/0`. Nothing is divided before the query is
    // known to be the origin, so the case needs no guard — and a guard written on the
    // span width would be a tolerance in a metric this subsection does not use.
    let collapsed_head = [0.0, 0.0, 0.0, 1.0];
    assert_eq!(
        SourcePosition::locate(0.0, &collapsed_head),
        SourcePosition::ORIGIN,
    );
    let origin = SourcePosition::locate(0.0, &collapsed_head);
    assert!(origin.remainder.is_finite(), "a `0/0` would have been NaN");

    // The far end is the upper-endpoint spelling on every table, so the index never
    // reaches the axis's last sample.
    for table in [collapsed_head, [0.0, 0.25, 0.5, 1.0]] {
        let far = SourcePosition::locate(1.0, &table);
        assert_eq!(
            far,
            SourcePosition {
                index: 2,
                remainder: 1.0,
            },
        );
        assert_eq!(far.value(), 3.0);
    }
}

/// The **collapsing** family of §8.9.5.4: a chain that concentrates its length before its
/// last sample and closes with one hair-thin final chord. Its table does not saturate —
/// `g_{resolution − 1} < 1` — so the guide's `j*_rim` collapses to zero and the next
/// chain's rim stays on the profile polyline.
///
/// A division guard on the **span** breaks exactly here, and breaks by a whole index: at
/// the query `1` it forces the true remainder `1` to `0`, the reflection carries that to
/// `resolution − p + 1` instead of `0`, and the rim of a chain that should read its own
/// polyline is replanted onto an interior row of the relaxed net. A `1.19e-7` window on a
/// normalised chord fraction is a third tolerance in a third metric; there is one snap
/// rule and its unit is the remainder.
#[test]
fn a_hair_thin_final_chord_keeps_its_full_remainder_and_leaves_the_rim_on_the_polyline() {
    let resolution = 4usize;
    let segments = 3usize;
    // The final span is one f32 quantum of `1.0` wide — strictly positive, and inside
    // any absolute window an `f32::EPSILON` test would draw.
    let hair = 1.0f32 - f32::EPSILON;
    assert!(1.0 - hair > 0.0 && 1.0 - hair <= f32::EPSILON);
    let collapsing = [0.0, 0.4, 0.8, hair, 1.0];

    let far = SourcePosition::locate(1.0, &collapsing);
    assert_eq!(
        far,
        SourcePosition {
            index: 3,
            remainder: 1.0,
        },
        "span / span is exactly 1.0 however thin the span is",
    );
    assert_eq!(
        far.value(),
        resolution as f32,
        "the table does not saturate",
    );

    assert_eq!(
        ring_source_position(far, resolution),
        SourcePosition::ORIGIN,
        "reflected through the resolution, a non-saturating table lands on row 0",
    );
    assert_eq!(
        ring_source_position(
            SourcePosition {
                index: 3,
                remainder: 0.0,
            },
            resolution,
        ),
        SourcePosition::at(1),
        "the whole-index move a span-width guard would have produced instead",
    );

    let mut axis = vec![vec![SourcePosition::ORIGIN; segments + 1]; 2];
    axis[0][segments] = far;
    assert_eq!(
        adj_rim_source_row(&axis, 2, segments, resolution, 1),
        None,
        "so the successor's rim is the 1-D read-back along the profile polyline",
    );

    // And the point read at that query is the polyline's far end, not the sample below
    // it — the same defect, one level up, where it moves a rim vertex.
    let points = [
        Vec3::ZERO,
        Vec3::X,
        Vec3::X * 2.0,
        Vec3::X * 3.0,
        Vec3::X * 4.0,
    ];
    assert_eq!(
        polyline_at_chord_fraction(&points, &collapsing, 1.0),
        Vec3::X * 4.0,
    );
    assert_eq!(
        polyline_at_chord_fraction(&points, &[0.0, 0.0, 0.5, 0.75, 1.0], 0.0),
        Vec3::ZERO,
        "and the collapsed head reads the first point rather than dividing by zero",
    );
}

/// BEVEL.md §8.5.5.5(d)'s two-axis read: sub-patch `i` consumes chain `i`'s table
/// **forward** on the `k` axis and chain `i − 1`'s table **reversed** on the `j` axis.
/// The reversal is an *index* mirror, so `[0, …, 0, 1]` presents as `[1, 0, …, 0]` — the
/// degenerate chain pulls its own block onto one edge of the source grid and the next
/// block onto the opposite edge. Re-deriving fractions along a reversed polyline would
/// give `[0, 1, …, 1]` and a symmetric answer instead, which is the divergence.
#[test]
fn degenerate_chord_table_read_reversed_lands_on_the_opposite_end_of_the_source_axis() {
    let sectors = 3usize;
    let segments = 3usize;
    let resolution = boundary_source_resolution(segments as u32) as usize;
    let net_half = resolution / 2;
    let target_half = segments / 2;

    let point = Vec3::new(-0.2, 0.5, 0.1);
    let degenerate =
        chord_fraction_source_positions(&vec![point; segments + 1], &vec![point; resolution + 1]);

    // The `k` axis, forward: every interior column stands on the source grid's first
    // column, the one over the degenerate boundvert.
    for k in 0..segments {
        assert_eq!(
            degenerate[k],
            SourcePosition::ORIGIN,
            "forward interior column {k} reads index 0"
        );
    }
    assert_eq!(
        degenerate[segments].value(),
        resolution as f32,
        "the far end is the last entry",
    );

    // A net whose every slot is distinguishable, so a read can be identified by where
    // it landed rather than by re-deriving the arithmetic the fill does.
    let chains: Vec<Vec<Vec3>> = (0..sectors)
        .map(|i| {
            (0..=resolution)
                .map(|k| Vec3::new(i as f32, k as f32, 0.0))
                .collect()
        })
        .collect();
    let mut net = AdjNet::new(sectors, net_half, chains, true);
    let interior_slot = Vec3::new(9.0, 9.0, 9.0);
    for i in 0..sectors {
        for j in 1..=net_half {
            for k in 1..=net_half {
                net.set(i, j, k, interior_slot);
            }
        }
    }

    // Chain 0 is the degenerate arc. Its neighbours read at integer indices so the
    // bilinear kernel lands on a net slot exactly and the assertion can be an identity.
    let well_conditioned: Vec<SourcePosition> = (0..=segments).map(SourcePosition::at).collect();
    let axis = vec![
        degenerate.clone(),
        well_conditioned.clone(),
        well_conditioned,
    ];
    let interior = AdjInterior::Relaxed {
        net: &net,
        axis: &axis,
        sectors,
        segments,
    };

    // Sub-patch 1's `j` axis is chain 0's table read *reversed*, so its lone interior
    // row queries `resolution − axis[0][segments − 1] = resolution` — past the far end
    // of the stored ring axis. §8.5.2's backward seam rewrites that to `(0, k, 0)`:
    // sub-patch 0's rim column, not a row of sub-patch 1's block at all.
    let landed = interior.position(1, 1, 1, target_half);
    assert_eq!(
        landed,
        net.at(0, 1, 0),
        "the read is the previous rim column"
    );
    assert_eq!(
        landed,
        net.chains[sectors - 1][resolution - 1],
        "and that column is pinned from the chain before it, one step in from the end",
    );
    assert_ne!(
        landed, interior_slot,
        "truncating the query to the stored quadrant would land on the seam spoke — \
         the opposite end of the axis the rule points at",
    );

    // The other flank: sub-patch 0's own `k` axis reads the same table *forward*, so
    // its interior column stands on column `0` of its source axis — the one over the
    // degenerate boundvert — while its `j` coordinate stays well conditioned at the
    // axis midpoint. Opposite ends of the same stored column, as §8.9.5.4(c) says.
    let forward = interior.position(0, 1, 1, target_half);
    assert_eq!(forward, net.at(0, net_half, 0));
    assert_ne!(forward, landed);
}

/// How far the mesh's position set is from `x ↦ −x` invariance — the one-sided
/// Hausdorff distance to its own mirror — in the `1e-5` position quanta BEVEL.md
/// §8.9.5.4(d) states its readings in.
///
/// The inner distance is **Chebyshev**, component-wise, and the accumulation is `f64`:
/// §8.9.5.4(d)'s table is stated in canonical position quanta, which are per-component,
/// so a Euclidean inner distance reads systematically larger than the numbers there and
/// cannot be compared against them. Matching each position to its *nearest* mirror
/// partner keeps this a lower bound on any index-paired reading, so it under-reports
/// rather than inventing asymmetry.
fn worst_mirror_gap(mesh: &LoopMesh) -> f64 {
    let positions: Vec<[f64; 3]> = mesh
        .vert_keys()
        .iter()
        .map(|&v| {
            let position = mesh.vertex_position(v);
            [position.x as f64, position.y as f64, position.z as f64]
        })
        .collect();
    positions
        .iter()
        .map(|p| {
            let mirrored = [-p[0], p[1], p[2]];
            positions
                .iter()
                .map(|q| {
                    (q[0] - mirrored[0])
                        .abs()
                        .max((q[1] - mirrored[1]).abs())
                        .max((q[2] - mirrored[2]).abs())
                })
                .fold(f64::INFINITY, f64::min)
        })
        .fold(0.0f64, f64::max)
        / 1.0e-5
}

/// BEVEL.md §8.9.5.4: at `spread = 0` the inner miter's two boundverts coincide, so the
/// arc chain's cell declines and the chain is the chord's even subdivision. Only the
/// **source** table takes §8.2.2.1.1's degenerate fallback — it is materialised at a
/// power of two, where the two-term interpolation is bit-inert; the **target** table is
/// materialised at the run's own `segments`, and a down-resample runs only when that is
/// *not* a power of two, which is exactly when the chain carries ulp-scale but strictly
/// positive chords (§8.9.5.4(c)). Reading a fallback source axis against an ordinary
/// target table, forward on `k` and reversed on the neighbour's `j`, is what makes a
/// **mirror-symmetric** junction emit a mirror-*asymmetric* cap.
///
/// The scope is "does the down-resample reach an interior sample", **not** "is
/// `segments` odd" (§8.9.5.4(e)): `segments = 4` is a power of two, so gate part (i)
/// declines and nothing fires; `segments ∈ {3, 6}` are live. `6` is the discriminator
/// the guide names — even *and* live — and it is also the only station here where the
/// even-target mid-index pin of [`adj_interior_axis`] meets a degenerate table. Two
/// controls hold the fixture fixed and vary one thing each: `spread = 0.1` (same
/// miter, well-conditioned chord) and `SHARP` (same spread, no miter), the latter
/// carrying the baseline asymmetry §8.9.5.4(d) says the high segment counts have
/// independently of miters. What the live arm reads is the **excess** over both.
#[test]
fn inner_miter_arc_zero_spread_breaks_mirror_symmetry_only_where_the_resample_runs() {
    let beveled_fan = |segments: u32, spread: f32, miter_inner: BevelMiter| -> LoopMesh {
        let (mut mesh, centre, rim) = planar_fan_axis_aligned(12);
        let geom: Vec<GeomKey> = [1usize, 4, 7]
            .iter()
            .map(|&k| mesh.query_edge_exists(centre, rim[k]).unwrap().into())
            .collect();
        mesh.bevel(
            &geom,
            BevelOptions {
                offset: 0.2,
                affect: BevelAffect::Edges,
                segments,
                profile: 0.5,
                vmesh_method: BevelVmeshMethod::Adj,
                miter_inner,
                miter_outer: BevelMiter::Sharp,
                spread,
                ..Default::default()
            },
        );
        mesh.validate_topology();
        mesh
    };
    let arc_gap = |segments: u32, spread: f32| {
        worst_mirror_gap(&beveled_fan(segments, spread, BevelMiter::Arc))
    };

    // The census is untouched — this is a placement-only divergence.
    let live = beveled_fan(3, 0.0, BevelMiter::Arc);
    let control = beveled_fan(3, 0.1, BevelMiter::Arc);
    assert_eq!(counts(&live), counts(&control));

    assert!(
        worst_mirror_gap(&control) < 1.0,
        "the spread control stays mirror-exact at the f32 read-back floor, not {}",
        worst_mirror_gap(&control),
    );
    // The magnitude at `segments = 3` is pinned separately — see
    // `inner_miter_arc_zero_spread_segments3_matches_the_tabulated_mirror_gap`, which
    // is where the two-sided reading of this arm lives.
    assert!(
        worst_mirror_gap(&live) > 5.0,
        "a degenerate arc chain throws the flanking blocks to opposite ends of their \
         source axes, which is orders of magnitude past the read-back floor, not {}",
        worst_mirror_gap(&live),
    );

    // Rim-only at `segments = 2` (§8.9.5.4(e)): a degenerate rim is one repeated point
    // whatever the table says. Gate part (i) then declines outright at a power of two.
    assert!(
        arc_gap(2, 0.0) < 1.0,
        "only the rim is sampled at segments = 2, so the table is inert, not {}",
        arc_gap(2, 0.0),
    );
    assert!(
        arc_gap(4, 0.0) < 1.0,
        "no resample runs at segments = 4, so zero spread stays mirror-exact, not {}",
        arc_gap(4, 0.0),
    );

    // §8.9.5.4(e): even *and* live, and firing hardest of every count measured. Both
    // controls carry the baseline asymmetry the high counts have anyway (the `SHARP`
    // arm shows it with no miter in play at all), so what the live arm reads is the
    // excess over them. Two-sided: "asymmetric" is not the claim — the claim is
    // asymmetric by the amount a whole block thrown to the far end of its source axis
    // costs, and §8.9.5.4(d) tabulates that amount.
    let six_sharp = worst_mirror_gap(&beveled_fan(6, 0.0, BevelMiter::Sharp));
    let six_spread = arc_gap(6, 0.1);
    let six_live = arc_gap(6, 0.0);
    assert!(
        (180.0..240.0).contains(&six_sharp),
        "segments = 6 baseline (§8.9.5.4(d) reads 206.9): {six_sharp}",
    );
    assert!(
        (280.0..350.0).contains(&six_spread),
        "segments = 6 spread control (§8.9.5.4(d) reads 311.1): {six_spread}",
    );
    assert!(
        (2800.0..3350.0).contains(&six_live),
        "segments = 6 live arm (§8.9.5.4(d) reads 3072.4): {six_live}",
    );
    assert!(
        six_live > 8.0 * six_sharp.max(six_spread),
        "the excess must be an order of magnitude over both controls: \
         sharp {six_sharp}, spread {six_spread}, live {six_live}",
    );

    // `segments = 8` is a power of two, so the table is never built and the zero-spread
    // arm sits *below* both controls — the baseline asymmetry with nothing added.
    let eight_live = arc_gap(8, 0.0);
    assert!(
        eight_live < worst_mirror_gap(&beveled_fan(8, 0.0, BevelMiter::Sharp)),
        "no resample runs at segments = 8, so the live arm carries no excess: \
         {eight_live}",
    );
}

/// The `segments = 3` row of BEVEL.md §8.9.5.4(d): `729.5` quanta against a `0.04`
/// control band, on the same fixture and options as the arm above.
///
/// It is the **magnitude** that pins §8.9.5.4(c)'s two-table reading. Applying the
/// zero-total fallback to the target axis as well collapses the two carrier slots onto
/// positions whose mirror images exist elsewhere in the set, and the metric then reads
/// single digits — §8.9.5.4(d)'s box tabulates that mis-reading at ≈`9` quanta, which
/// is why this arm is two-sided and not merely "asymmetric".
#[test]
fn inner_miter_arc_zero_spread_segments3_matches_the_tabulated_mirror_gap() {
    let (mut mesh, centre, rim) = planar_fan_axis_aligned(12);
    let geom: Vec<GeomKey> = [1usize, 4, 7]
        .iter()
        .map(|&k| mesh.query_edge_exists(centre, rim[k]).unwrap().into())
        .collect();
    mesh.bevel(
        &geom,
        BevelOptions {
            offset: 0.2,
            affect: BevelAffect::Edges,
            segments: 3,
            profile: 0.5,
            vmesh_method: BevelVmeshMethod::Adj,
            miter_inner: BevelMiter::Arc,
            miter_outer: BevelMiter::Sharp,
            spread: 0.0,
            ..Default::default()
        },
    );
    mesh.validate_topology();

    let gap = worst_mirror_gap(&mesh);
    assert!(
        (600.0..900.0).contains(&gap),
        "segments = 3 live arm (§8.9.5.4(d) reads 729.5): {gap}",
    );
}

/// BEVEL.md §8.5.5.5(d), "The rim's own `j` coordinate": the emitted rim is the `j = 0`
/// row of the two-axis read and carries a source row `j*_rim = resolution −
/// locate(1, g_{i−1})` belonging to the **previous** chain. It collapses to zero — and
/// only then is the rim the 1-D read-back of §8.2.2.1.1 step 3 along the profile
/// polyline — exactly when that chain's source table does not saturate.
///
/// The three tables a `spread = 0` inner miter can build are all exercised here, because
/// two of them collapse and only the third displaces, and the pair that collapses is
/// where the tempting wrong rule ("a degenerate chain cannot hurt the rim") looks true:
///
/// - a well-conditioned chain — the ordinary case;
/// - a **fully** degenerate chain, whose `[0, …, 0, 1]` fallback still reaches `1` only
///   at the last index, so `locate(1, g) = resolution` and its successor's rim stays on
///   the polyline (§8.9.5.4(c): this is the `offset = 0.2` meet point);
/// - a **saturating** chain, which reaches `1` early because the meet point's mantissa
///   refuses the two-term interpolation at some dyadic parameter (§8.9.5.4(c)'s measured
///   `offset = 0.25` table), and whose successor's rim is read `resolution − p` rows
///   inside the net.
#[test]
fn rim_source_row_is_the_previous_chains_saturation_and_collapses_on_every_other_table() {
    let segments = 5usize;
    let resolution = boundary_source_resolution(segments as u32) as usize;
    assert_eq!(resolution, 8);

    // §8.9.5.4(c)'s measured saturating table, at the meet point an `offset = 0.25`
    // bevel of the fan below puts the miter on.
    let saturating_meet = Vec3::new(0.24999999, 0.25, 0.0);
    let saturating_source =
        arc_chain_positions(saturating_meet, saturating_meet, saturating_meet, 0.5, 8);
    assert_eq!(
        normalised_chord_fractions(&saturating_source),
        vec![0.0, 0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0, 1.0],
        "neither the fallback nor a well-formed monotone table — it saturates at 6",
    );
    let saturating = chord_fraction_source_positions(
        &arc_chain_positions(
            saturating_meet,
            saturating_meet,
            saturating_meet,
            0.5,
            segments as u32,
        ),
        &saturating_source,
    );
    assert_eq!(
        *saturating.last().unwrap(),
        SourcePosition {
            index: 5,
            remainder: 1.0,
        },
        "locate(1, g) is the plateau's FIRST index, not the last — and it arrives as the \
         upper endpoint of the span below it, so its *value* is 6 while its *index* is 5",
    );

    // The fully degenerate sibling: every source sample is bit-exactly the meet point,
    // so the table is the `[0, …, 0, 1]` fallback and `locate(1, g)` is the far end.
    let degenerate_meet = Vec3::new(0.2, 0.2, 0.0);
    let degenerate_source = arc_chain_positions(
        degenerate_meet,
        degenerate_meet,
        degenerate_meet,
        0.5,
        resolution as u32,
    );
    assert!(degenerate_source.iter().all(|&p| p == degenerate_meet));
    let degenerate = chord_fraction_source_positions(
        &arc_chain_positions(
            degenerate_meet,
            degenerate_meet,
            degenerate_meet,
            0.5,
            segments as u32,
        ),
        &degenerate_source,
    );
    assert_eq!(degenerate.last().unwrap().value(), resolution as f32);
    assert_ne!(
        degenerate.last().unwrap().index,
        resolution,
        "the index never reaches the axis's last sample, so a saturation predicate \
         written as `index == resolution` answers backwards on every non-saturating table",
    );

    let well_conditioned: Vec<SourcePosition> = (0..=segments)
        .map(|k| upper_endpoint_spelling(k as f32 * resolution as f32 / segments as f32))
        .collect();

    let axis = vec![
        saturating,
        well_conditioned.clone(),
        degenerate,
        well_conditioned,
    ];
    let sectors = axis.len();
    assert_eq!(
        adj_rim_source_row(&axis, sectors, segments, resolution, 1),
        Some(SourcePosition::at(2)),
        "the chain AFTER the saturating one reads two rows inside the net",
    );
    for i in [0usize, 2, 3] {
        assert_eq!(
            adj_rim_source_row(&axis, sectors, segments, resolution, i),
            None,
            "chain {i}'s predecessor does not saturate, so its rim stays on the polyline",
        );
    }
}

/// The other half of the rule: what a non-zero `j*_rim` actually reads. BEVEL.md
/// §8.9.5.4(c)'s zero-residual box names the two samples exactly —
/// `sample(chain 1, j* = 2, 1.6)` and `sample(chain 1, j* = 2, 3.2)` at `offset = 0.25`,
/// `segments = 5` — so this arm drives those coordinates through a net whose every slot
/// is identifiable and shows that they land on an interior **row**, not on the profile
/// polyline that row 0 stores.
#[test]
fn a_saturating_rim_reads_an_interior_net_row_not_the_profile_polyline() {
    let sectors = 3usize;
    let net_half = 4usize;
    let resolution = 2 * net_half;

    let chains: Vec<Vec<Vec3>> = (0..sectors)
        .map(|i| {
            (0..=resolution)
                .map(|k| Vec3::new(i as f32, k as f32, 0.0))
                .collect()
        })
        .collect();
    let mut net = AdjNet::new(sectors, net_half, chains, true);
    // `j < net_half` keeps every write on its own sub-patch: the `j == half` row is the
    // *previous* sub-patch's seam column, so writing it would relabel the slots this arm
    // identifies its reads by.
    for i in 0..sectors {
        for j in 1..net_half {
            for k in 1..=net_half {
                net.set(i, j, k, Vec3::new(100.0 + i as f32, j as f32, k as f32));
            }
        }
    }

    let rim_row = SourcePosition::at(2);
    for along_chain in [1.6f32, 3.2] {
        let chain = upper_endpoint_spelling(along_chain);
        let read = net.sample(1, rim_row, chain);
        let on_polyline = net.sample(1, SourcePosition::ORIGIN, chain);
        assert_eq!(
            read.x, 101.0,
            "the read landed in sub-patch 1's stored interior, {along_chain} along",
        );
        assert_eq!(read.y, rim_row.value(), "on the row `j*_rim` names");
        assert!(
            (read.z - along_chain).abs() < 1.0e-5,
            "and at the chain coordinate it was asked for: {read:?}",
        );
        assert_eq!(
            on_polyline,
            Vec3::new(1.0, along_chain, 0.0),
            "row 0 IS the profile polyline — which is what the collapsing case reads: \
             {on_polyline:?}",
        );
    }
}

/// End to end at BEVEL.md §8.9.5.4(c)'s worked station: the flat 12-spoke fan beveled on
/// `0° / 90° / 180°` at `offset = 0.25`, `segments = 5`, inner `ARC`, `spread = 0`. One
/// of the two collapsed arcs saturates its source table there, so its successor's first
/// `⌊segments/2⌋ = 2` rim samples leave the profile polyline.
///
/// The two positions a rim emitted from the profile chain alone carries are named below.
/// They are **absent** from the emitted mesh, and what stands in their place sits
/// `1191` and `835` position quanta away — the first being the figure §8.9.5.4(e)'s
/// `offset` sweep reports for this station, which is what ties the local rule to the
/// measured divergence. Nothing else moves: the census is identical to the
/// well-conditioned `spread` control's.
///
/// The *absolute* positions are loopmesh's own. A displaced rim sample is a read of the
/// relaxed net's interior, so unlike an ordinary rim vertex it inherits the §8.5.5.7
/// Tier-2 interior band — only the displacement is a hard quantity.
#[test]
fn zero_spread_arc_rim_is_replanted_where_the_previous_source_table_saturates() {
    let beveled_fan = |spread: f32| -> LoopMesh {
        let (mut mesh, centre, rim) = planar_fan_axis_aligned(12);
        let geom: Vec<GeomKey> = [1usize, 4, 7]
            .iter()
            .map(|&k| mesh.query_edge_exists(centre, rim[k]).unwrap().into())
            .collect();
        mesh.bevel(
            &geom,
            BevelOptions {
                offset: 0.25,
                affect: BevelAffect::Edges,
                segments: 5,
                profile: 0.5,
                vmesh_method: BevelVmeshMethod::Adj,
                miter_inner: BevelMiter::Arc,
                miter_outer: BevelMiter::Sharp,
                spread,
                ..Default::default()
            },
        );
        mesh.validate_topology();
        mesh
    };

    let mesh = beveled_fan(0.0);
    assert_eq!(
        counts(&mesh),
        counts(&beveled_fan(0.1)),
        "the replant is placement-only — same census as the well-conditioned control",
    );

    let nearest_quanta = |want: Vec3| -> f64 {
        mesh.vert_keys()
            .iter()
            .map(|&v| {
                let p = mesh.vertex_position(v);
                ((p.x - want.x).abs())
                    .max((p.y - want.y).abs())
                    .max((p.z - want.z).abs()) as f64
                    / 1.0e-5
            })
            .fold(f64::INFINITY, f64::min)
    };

    // Chain 1's own 1-D chord read-back, at source positions 1.6 and 3.2 — the two
    // samples §8.9.5.4(c)'s box identifies as the profile-emitting fill's extras.
    let profile_emitted = [
        (Vec3::new(0.055_179_883, 0.151_881_35, 0.0), 1191.0),
        (Vec3::new(0.159_748_9, 0.186_428_01, 0.0), 835.0),
    ];
    for (naive, displacement) in profile_emitted {
        let gap = nearest_quanta(naive);
        assert!(
            (displacement - 40.0..displacement + 40.0).contains(&gap),
            "the emitted rim sits {displacement} quanta off the profile polyline read, \
             not {gap}",
        );
    }
}

/// A grid read past `half` on **both** axes has no §8.5.2 canonical representative, and
/// [`AdjNet::sample`] resolves it rather than rejecting it — the case is reachable on
/// ordinary input (that method's note names the configuration). This arm pins the
/// resolution: [`AdjNet::at`]'s two seam rewrites compose, terminate, and land on a
/// stored interior slot, so the behaviour is a checked property of the code rather than
/// folklore. The *value* stays undefined by the guide, which pins neither the pair nor a
/// composition order; only its well-definedness is claimed here.
#[test]
fn a_grid_read_past_half_on_both_axes_composes_the_two_seam_rewrites() {
    let sectors = 5usize;
    let half = 3usize;
    let resolution = 2 * half;

    let chains: Vec<Vec<Vec3>> = (0..sectors)
        .map(|i| {
            (0..=resolution)
                .map(|k| Vec3::new(i as f32, k as f32, 0.0))
                .collect()
        })
        .collect();
    let mut net = AdjNet::new(sectors, half, chains, true);
    // Strictly interior slots only, so no write aliases a seam row and every read below
    // is identifiable by where it landed.
    for i in 0..sectors {
        for j in 1..half {
            for k in 1..half {
                net.set(i, j, k, Vec3::new(10.0 + i as f32, j as f32, k as f32));
            }
        }
    }

    let (i, j, k) = (1usize, 4usize, 5usize);
    let landed = net.at(i, j, k);
    assert_eq!(
        landed,
        net.at((i + 2) % sectors, resolution - j, resolution - k),
        "the `k` rewrite runs first and the `j` rewrite runs on its result",
    );
    assert_eq!(
        landed,
        Vec3::new(10.0 + 3.0, 2.0, 1.0),
        "and that composition names a stored interior slot of sub-patch `i + 2`",
    );
    assert_ne!(
        landed,
        net.at(i, half, half),
        "clamping both axes instead would land on the shared centre — the far end of \
         both axes the seam rules point at",
    );
}

/// The reachable configuration behind the arm above, driven end to end: a plain
/// 12-spoke fan beveled on `0° / 90° / 180°` with an inner `ARC` miter at zero spread
/// over the [V-ADJ] grid. The §8.9.5.4(c) degenerate chord tables it produces send one
/// [`AdjNet::sample`] read past `half` on **both** axes, which §8.5.2 asks an
/// implementation to reject — so this ran as a *panic in debug builds*, and the compat
/// suite runs in debug.
///
/// Nothing here is exotic: no degenerate geometry, no hand-built net, ordinary
/// parameters. So the arm is the pin that stops the rejecting guard being reintroduced —
/// the run must finish, the mesh must survive [`LoopMesh::validate_topology`], and every
/// vertex the bevel mints must be finite (the seam composition lands on a stored slot,
/// never on an uninitialised or NaN one).
#[test]
fn zero_spread_arc_miter_fan_completes_despite_reading_past_half_on_both_axes() {
    let (mut mesh, centre, rim) = planar_fan_axis_aligned(12);
    let geom: Vec<GeomKey> = [1usize, 4, 7]
        .iter()
        .map(|&k| mesh.query_edge_exists(centre, rim[k]).unwrap().into())
        .collect();
    let result = mesh.bevel(
        &geom,
        BevelOptions {
            offset: 0.25,
            affect: BevelAffect::Edges,
            segments: 6,
            profile: 0.5,
            vmesh_method: BevelVmeshMethod::Adj,
            miter_inner: BevelMiter::Arc,
            miter_outer: BevelMiter::Sharp,
            spread: 0.0,
            ..Default::default()
        },
    );
    mesh.validate_topology();

    assert!(
        !result.faces.is_empty(),
        "the bevel ran to completion and emitted its fill",
    );
    assert!(
        result
            .verts
            .iter()
            .all(|&v| mesh.vertex_position(v).is_finite()),
        "every minted vert is finite — the both-axes seam composition landed on a \
         stored slot",
    );
}

/// A cell point's coordinates `(u, v)` in the §8.2.2.1 profile cell: solve
/// `p = origin + u·a_axis + v·b_axis` in the (non-orthogonal) cell basis.
fn cell_uv(cell: &ProfileCell, position: Vec3) -> (f32, f32) {
    let origin = cell.start + cell.end - cell.mid;
    let a_axis = cell.mid - cell.start;
    let b_axis = cell.mid - cell.end;
    let offset = position - origin;
    let (aa, ab, bb) = (a_axis.dot(a_axis), a_axis.dot(b_axis), b_axis.dot(b_axis));
    let (oa, ob) = (offset.dot(a_axis), offset.dot(b_axis));
    let determinant = aa * bb - ab * ab;
    (
        (oa * bb - ob * ab) / determinant,
        (ob * aa - oa * ab) / determinant,
    )
}

/// The **cell radius** `√(u² + v²)` of a rim sample (BEVEL.md §8.2.2.1.1's read-out
/// table). For the circular default the profile curve is the unit superellipse in
/// exactly this frame, so `= 1` ⟺ the sample sits *on* the curve (it was snapped
/// back, §8.5.5.3) and `< 1` ⟺ it is the bare chord point of the resample, strictly
/// inside the curve (the general cap, which has no superellipsoid to snap onto).
fn cell_radius(cell: &ProfileCell, position: Vec3) -> f32 {
    let (u, v) = cell_uv(cell, position);
    (u * u + v * v).sqrt()
}

/// The `segments = 3` cube corner's profile cell across the beveled edge running to
/// `far`, between the two boundverts that bridge it. At a right-angled sector the
/// §7.6.2 meet is `v + offset·(d_a + d_b)`.
fn cube_corner_cell(far: Vec3, other_a: Vec3, other_b: Vec3) -> ProfileCell {
    let corner = Vec3::new(1.0, -1.0, -1.0);
    let toward = |p: Vec3| (p - corner).normalize();
    let (d_far, d_a, d_b) = (toward(far), toward(other_a), toward(other_b));
    ProfileCell::new(
        corner + 0.25 * (d_far + d_a),
        corner + 0.25 * (d_far + d_b),
        corner,
        d_far,
    )
}

/// BEVEL.md §8.2.2.1.1 (steps 1–4) / §8.5.5.3 / §8.5.5.5's three-way gate table: on
/// the **closed-form tri-corner cap** at a non-power-of-two `segments` the chord
/// resample *does* run (it is not gated on cap kind), and the fill's grid-level
/// superellipsoid snap then projects the chord point radially back **out** onto the
/// corner sphere. So the emitted rim sample keeps cell radius `1` — it satisfies the
/// §8.5.5.6 (2) membership invariant — but sits at a **shifted arc angle**,
/// ≈`1.9e-4` (≈19 quanta) from the raw even-angle sample it would carry if the
/// resample had been skipped.
#[test]
fn adj_cap_rim_tri_corner_resamples_then_snaps_onto_the_sphere() {
    let (mesh, result) = cube_corner_adj(3);
    mesh.validate_topology();

    let cell = cube_corner_cell(
        Vec3::new(-1.0, -1.0, -1.0),
        Vec3::new(1.0, 1.0, -1.0),
        Vec3::new(1.0, -1.0, 1.0),
    );
    let raw = cell.chain_positions(3, 0.5, None, ChainEmission::Raw);
    let chord = cell.chain_positions(3, 0.5, None, ChainEmission::Resampled);
    let snap =
        |p: Vec3| adj_sphere_center() + ADJ_SPHERE_RADIUS * (p - adj_sphere_center()).normalize();

    for k in 1..3 {
        let emitted = snap(chord[k]);

        assert!(
            (cell_radius(&cell, raw[k]) - 1.0).abs() < 1.0e-4,
            "the raw even-angle sample is on the profile curve by construction",
        );
        assert!(
            cell_radius(&cell, chord[k]) < 1.0 - 1.0e-3,
            "step 3's chord point lies strictly inside the curve",
        );
        assert!(
            (cell_radius(&cell, emitted) - 1.0).abs() < 1.0e-4,
            "step 4's snap puts the emitted rim sample back on the curve (cell radius 1)",
        );

        let shift = (emitted - raw[k]).length();
        assert!(
            (1.0e-4..3.0e-4).contains(&shift),
            "resample + snap is an *angular* shift of ≈19 quanta, not a radial one (got {shift})",
        );

        assert_eq!(
            result_has_vert_near(&mesh, &result, emitted),
            1,
            "the tri-corner cap emits the resampled-then-snapped rim sample",
        );
        assert_eq!(
            result_has_vert_near(&mesh, &result, raw[k]),
            0,
            "the raw even-angle sample is never emitted at a non-power-of-two segments",
        );
    }
}

/// BEVEL.md §8.2.2.1.1 step 4 / §8.5.5.5 ("The emitted boundary"): a **general** cap
/// — here the symmetric equal-offset valence-4 pyramid apex, which fails the §8.5.1
/// trigger on **valence alone** — has no superellipsoid, so the snap does not run and
/// the emitted rim is step 3's **bare chord point**: cell radius strictly `< 1`,
/// inside the profile curve. Emitting the raw even-angle chain here instead is a
/// ≈490-quantum error that every invariant of §8.5.5.7 is blind to.
#[test]
fn adj_cap_rim_general_ring_emits_the_bare_resample_unsnapped() {
    let mut mesh = LoopMesh::new();
    let [apex, b0, b1, b2, b3] = square_pyramid(&mut mesh);
    let apex_position = mesh.vertex_position(apex);
    let base = [b0, b1, b2, b3].map(|b| mesh.vertex_position(b));
    let slants = [b0, b1, b2, b3].map(|b| {
        mesh.query_edge_exists(apex, b)
            .expect("slant edge exists")
            .into()
    });
    let result = mesh.bevel(
        &slants,
        BevelOptions {
            offset: 0.3,
            affect: BevelAffect::Edges,
            segments: 3,
            vmesh_method: BevelVmeshMethod::Adj,
            ..Default::default()
        },
    );
    mesh.validate_topology();

    // §7.6.2 in a planar sector: the two offset lines meet on the angle bisector at
    // `v + offset·(d_a + d_b)/sin θ`, which reduces to the cube corner's
    // `v + offset·(d_a + d_b)` only at a right-angled sector.
    let toward = |p: Vec3| (p - apex_position).normalize();
    let meet = |p: Vec3, q: Vec3| {
        let (d_a, d_b) = (toward(p), toward(q));
        let sin_sector = d_a.cross(d_b).length();
        apex_position + 0.3 * (d_a + d_b) / sin_sector
    };
    // The chain across slant edge `apex→b0` runs between the meets of the two sectors
    // that edge bounds (`b3–b0` and `b0–b1`).
    let cell = ProfileCell::new(
        meet(base[3], base[0]),
        meet(base[0], base[1]),
        apex_position,
        toward(base[0]),
    );
    for endpoint in [cell.start, cell.end] {
        assert_eq!(
            result_has_vert_near(&mesh, &result, endpoint),
            1,
            "the §7.6.2 ring boundverts are the chain's two ends",
        );
    }

    let raw = cell.chain_positions(3, 0.5, None, ChainEmission::Raw);
    let chord = cell.chain_positions(3, 0.5, None, ChainEmission::Resampled);
    for k in 1..3 {
        assert!(
            cell_radius(&cell, chord[k]) < 1.0 - 1.0e-2,
            "the general cap's emitted rim is the bare chord point, inside the curve",
        );
        assert!(
            (chord[k] - raw[k]).length() > 1.0e-3,
            "the bare resample is ≈490 quanta off the raw even-angle sample",
        );
        assert_eq!(
            result_has_vert_near(&mesh, &result, chord[k]),
            1,
            "the general cap emits the unsnapped chord-fraction resample",
        );
        assert_eq!(
            result_has_vert_near(&mesh, &result, raw[k]),
            0,
            "no general-cap rim vert carries the raw even-angle sample",
        );
    }
}

/// BEVEL.md §8.2.2.1.1 gate part (i) / the three-way table's third row: at
/// `segments ∈ {4, 8, 16, …}` the source resolution lands on the target, no chord
/// resample runs, and the emitted rim **is** the raw profile chain — on the curve, at
/// the raw even-angle sample. The tri-corner snap still runs there; it is simply the
/// identity on the rim.
#[test]
fn adj_cap_rim_power_of_two_segments_emits_the_raw_profile_chain() {
    let (mesh, result) = cube_corner_adj(4);
    mesh.validate_topology();

    let cell = cube_corner_cell(
        Vec3::new(-1.0, -1.0, -1.0),
        Vec3::new(1.0, 1.0, -1.0),
        Vec3::new(1.0, -1.0, 1.0),
    );
    let raw = cell.chain_positions(4, 0.5, None, ChainEmission::Raw);
    for k in 1..4 {
        assert!(
            (cell_radius(&cell, raw[k]) - 1.0).abs() < 1.0e-4,
            "a power-of-two raw sample is already on the curve, so the snap is the identity",
        );
        assert_eq!(
            result_has_vert_near(&mesh, &result, raw[k]),
            1,
            "the raw chain is emitted verbatim at segments = 4",
        );
    }
}

/// BEVEL.md §8.2.2.1 / §8.3.0.1: at a valence-≥3 terminal the profile plane is
/// snapped to the triangle through the terminal vertex and the two chain
/// endpoints, flattening the far-end cross-section into the terminal's rim plane
/// (`z = 0` here) instead of bulging up along the tilted apex edge. The un-snapped
/// `start–mid–end` mid-plane lifts the interior sample `~0.035` off the rim.
#[test]
fn terminal_far_end_snaps_to_rim_plane() {
    let (mesh, result) = reflex_apex_outer_miter(BevelMiter::Sharp);
    mesh.validate_topology();

    // The two worked-check far-end interior samples land flat on the rim (z = 0).
    assert_eq!(
        result_has_vert_near(&mesh, &result, Vec3::new(0.90964, 0.94068, 0.0)),
        1,
        "the c-terminal far-end sample is snapped flat into the rim plane",
    );
    assert_eq!(
        result_has_vert_near(&mesh, &result, Vec3::new(0.0, 1.35435, 0.0)),
        1,
        "the b-terminal far-end sample is snapped flat into the rim plane",
    );
    // The un-snapped mid-plane value (lifted off the rim) must not appear.
    assert_eq!(
        result_has_vert_near(&mesh, &result, Vec3::new(0.87436, 0.90540, 0.03528)),
        0,
        "the tilted default mid-plane sample is not produced",
    );
}

/// BEVEL.md §8.9.1 / §8.9.2: the outer miter is **inert** at a junction with no
/// reflex sector — every sector of the convex cube corner spans `< 180°`, so
/// `miter_outer = Arc` declines to the sharp ring and matches the no-miter corner.
#[test]
fn outer_miter_inert_without_reflex_sector() {
    let (default_mesh, _default) = cube_corner_adj_segments2();
    let mut arc_mesh = LoopMesh::new();
    mesh_add_cube(&mut arc_mesh);
    let e01 = cube_edge(
        &arc_mesh,
        Vec3::new(1.0, -1.0, -1.0),
        Vec3::new(-1.0, -1.0, -1.0),
    );
    let e03 = cube_edge(
        &arc_mesh,
        Vec3::new(1.0, -1.0, -1.0),
        Vec3::new(1.0, 1.0, -1.0),
    );
    let e04 = cube_edge(
        &arc_mesh,
        Vec3::new(1.0, -1.0, -1.0),
        Vec3::new(1.0, -1.0, 1.0),
    );
    arc_mesh.bevel(
        &[e01.into(), e03.into(), e04.into()],
        BevelOptions {
            offset: 0.25,
            affect: BevelAffect::Edges,
            segments: 2,
            vmesh_method: BevelVmeshMethod::Adj,
            miter_outer: BevelMiter::Arc,
            ..Default::default()
        },
    );
    arc_mesh.validate_topology();
    assert_eq!(
        counts(&default_mesh),
        counts(&arc_mesh),
        "miter_outer = Arc is inert on the reflex-free cube corner",
    );
}
