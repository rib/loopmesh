use super::*;

/// The §8.5.5.8 driver's corner, beveled through a **custom** curve instead of the
/// analytic family. The `profile` scalar is still supplied — a custom bevel carries one,
/// and the whole point of the arms below is that nothing on this path may read it.
fn cube_corner_adj_custom(
    segments: u32,
    profile: f32,
    curve: &BevelCurveProfile,
) -> (LoopMesh, BevelResult) {
    let mut mesh = LoopMesh::new();
    mesh_add_cube(&mut mesh);
    let e01 = cube_edge(
        &mesh,
        Vec3::new(1.0, -1.0, -1.0),
        Vec3::new(-1.0, -1.0, -1.0),
    );
    let e03 = cube_edge(&mesh, Vec3::new(1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, -1.0));
    let e04 = cube_edge(&mesh, Vec3::new(1.0, -1.0, -1.0), Vec3::new(1.0, -1.0, 1.0));
    let result = mesh.bevel(
        &[e01.into(), e03.into(), e04.into()],
        BevelOptions {
            offset: 0.25,
            affect: BevelAffect::Edges,
            segments,
            profile,
            profile_type: BevelProfileType::Custom(curve),
            vmesh_method: BevelVmeshMethod::Adj,
            ..Default::default()
        },
    );
    (mesh, result)
}

/// BEVEL.md §8.5.5.5(f.1): **a custom profile never takes the tri-corner family at all.**
/// The predicate refuses it outright — one of the two unconditional refusals, checked
/// before any offset or dihedral is examined — and step 1 excludes it too, so the only
/// destination (f.1) leaves it is step 4's general fullness grid. Step 4's own shortcut
/// back into the family honours the same refusal, so there is no way around.
///
/// The orthonormal cube corner is the sharpest place to pin this, because it is the one
/// ring that fires *every* arm of the family: at the square-in sentinel it would take the
/// (f.4a) weld (no interior at all, at most one central triangle) and at the circle it
/// would take (f.2)'s unit-octant grid. Under a custom curve both must give way to the
/// §8.5.5.5(e) parity counts — and the two contrast arms below show they otherwise would.
#[test]
fn custom_profile_refuses_the_tri_corner_family_at_the_cube_corner() {
    let curve = BevelCurveProfile::from_points(&[
        (1.0, 0.0, BevelHandle::Vector),
        (0.5, 0.25, BevelHandle::Vector),
        (0.0, 1.0, BevelHandle::Vector),
    ]);
    for segments in [2u32, 3, 4, 5] {
        let (faces, verts) = adj_general_ring_parity_counts(3, segments);

        // The two exponents the family would otherwise claim: the square-in sentinel
        // (step 3's `EXP_SQUARE_IN` row) and the circular default (its `otherwise` arm).
        for profile in [0.0_f32, 0.5] {
            let (mesh, result) = cube_corner_adj_custom(segments, profile, &curve);
            mesh.validate_topology();
            let cap = curved_corner_grid_faces(&mesh, &result);
            assert_eq!(
                cap.len(),
                faces,
                "custom at profile {profile} takes step 4's grid \
                 (segments = {segments})",
            );
            assert_eq!(
                curved_corner_grid_verts(&mesh, &result).len(),
                verts,
                "…and its full §8.5.5.5(e) vert parity (segments = {segments})",
            );
        }

        // Non-vacuity, from both of the arms the refusal displaces. The weld's entire
        // face output is the odd parity's one central `s`-gon; the octant grid carries
        // the same *counts* as step 4's grid, so the contrast there is positional and is
        // the business of the chain arm below.
        let (welded, weld_result) = cube_corner_adj_profile(segments, 0.0);
        assert_eq!(
            curved_corner_grid_faces(&welded, &weld_result).len(),
            usize::from(segments % 2 == 1),
            "control: the analytic square-in corner welds (segments = {segments})",
        );
        assert!(
            faces > usize::from(segments % 2 == 1),
            "…and the grid the custom curve takes is a different object",
        );
    }
}

/// BEVEL.md §8.6.3 / §8.6.3.1(4).4 row 3: a [V-ADJ] junction sector's emitted rim is the
/// **custom curve's** chain, not the superellipse the `profile` slider would name. On an
/// all-vector control net at `segments = 2` the §8.2.2.1.1 read-back reduces to the
/// identity (`n ≥ E`, every sample on the straight control polyline), so the lone interior
/// sample of each ring chain is the control point itself, mapped through §8.2.2.1's cell.
///
/// Worked for the `{0,1}` chain of the §8.5.5.8 driver. Its two boundverts are the §7.6.2
/// meets `(0.75, −1, −0.75)` — the chain's `start` — and `(0.75, −0.75, −1)`, its `end`;
/// the on-edge foot is `(0.75, −1, −1)`, so the cell is `a_axis = mid − start =
/// (0, 0, −0.25)`, i.e. the `z` direction, and `b_axis = mid − end = (0, −0.25, 0)`, about
/// the origin `start + end − mid = (0.75, −0.75, −0.75)`.
///
/// The read takes §8.6.3.1(6)'s `reverse = true` branch: the orientation pass
/// ([`CustomOrientation`]) marks this sector's edge-end `is_start_side = false`, its seed
/// compare `start.z < end.z` being `−0.75 < −1`, which is false. That branch reads the chain
/// from the far end with the axis swap **dropped** — so there is no swap here, and at
/// `segments = 2` reversing a three-sample chain leaves the lone interior sample where it
/// was. The widget control point `(x, y) = (0.5, 0.25)` therefore enters the
/// cell as `(u, v) = (0.5, 0.25)` unchanged, at
/// `origin + 0.5·a_axis + 0.25·b_axis = (0.75, −0.8125, −0.875)` — where the circle would
/// put `(√2/2, √2/2)`, `(0.75, −0.92678, −0.92678)`. The two are 0.08 world apart, so this
/// separates the control net from the analytic family outright rather than by a tolerance.
#[test]
fn custom_profile_edge_junction_chain_rides_the_control_net_not_the_circle() {
    let curve = BevelCurveProfile::from_points(&[
        (1.0, 0.0, BevelHandle::Vector),
        (0.5, 0.25, BevelHandle::Vector),
        (0.0, 1.0, BevelHandle::Vector),
    ]);
    let control_point = Vec3::new(0.75, -0.8125, -0.875);
    let circle = Vec3::new(0.75, -0.92678, -0.92678);

    for profile in [0.0_f32, 0.5] {
        let (mesh, result) = cube_corner_adj_custom(2, profile, &curve);
        mesh.validate_topology();
        let ring: Vec<Vec3> = result
            .verts
            .iter()
            .map(|&vertex| mesh.vertex_position(vertex))
            .filter(|position| (position.x - 0.75).abs() <= 1.0e-4)
            .collect();
        assert_position_present(
            &ring,
            control_point,
            &format!("the `{{0,1}}` chain's custom sample at profile {profile}"),
        );
        assert!(
            ring.iter().all(|p| (*p - circle).length() > 1.0e-3),
            "no sample sits on the circular cross-section the slider names \
             (profile = {profile})",
        );
    }
}

/// BEVEL.md §8.6.3.1(4).4's **miter carve-out**, both halves at once: on a mitered ring a
/// beveled edge's **flank** cross-section rides the custom control net, while the split
/// sector's own **arc** chain stays analytic.
///
/// The carve-out is stated per *boundvert*: the operator hands a second, non-custom
/// spacing pair to every corner tagged an arc start or a patch start, and §8.6.3.1(6) A.6
/// makes exactly those corners skip the handedness pass. An inner miter tags each
/// sector's leading `B1`; its partner `B3` is an arc *end*, which §8.9.2/§8.9.3 give
/// genuine ownership of the sector's closing beveled edge, so `B3`'s flank is an ordinary
/// §8.6.3.1(6) A.0 edge-end cross-section and reads the curve.
///
/// Worked on the `{0,1}` flank of `custom_profile_edge_junction_chain_rides_the_control_net_not_the_circle`,
/// whose cell is `a_axis = (0, 0, −0.25)` / `b_axis = (0, −0.25, 0)` about the origin
/// `(0.75, −0.75, −0.75)`. The miter slides **both** of that chain's boundverts `spread`
/// along the *same* beveled edge (`B3` of one sector and `B1` of the next both ride the
/// edge they share), so the whole cell — start, end, on-edge foot, both axes — is a rigid
/// translation by `spread·(−1, 0, 0)` and every sample of it translates with it. At
/// `spread = 0.3` the widget control point `(0.5, 0.25)` therefore lands at
/// `(0.45, −0.8125, −0.875)` where the circle would put `(0.45, −0.92678, −0.92678)`.
///
/// The arc is the contrast: its `segments = 2` sample is the closed-form
/// `(2 − √2)·chord_mid + (√2 − 1)·midpoint` bulge, which for the `y = −1` sector's meet
/// `(0.75, −1, −0.75)` and its two edge directions `(−1, 0, 0)` / `(0, 0, 1)` is
/// `meet + ((2 − √2)/2)·spread·(−1, 0, 1) = (0.66213, −1, −0.66213)` — the same point
/// under both profile types **at this test's `profile = 0.5`**. It is the *curve* the
/// carve-out withholds from the arc, not the slider, so the two arms coincide because they
/// share a scalar and not because the type pins the exponent
/// ([`custom_profile_outer_miter_arc_chain_tracks_the_superellipse_slider`]).
///
/// **The other two flanks are pinned too, and they are not the first one's rotations.**
/// §8.6.3.1(6)(i) fixes each run's seed *end* from the input mesh's stored edge endpoints,
/// and this cube mints `e_x` as `v → (−1,−1,−1)` and `e_z` as `v → (1,−1,1)` — both seeding
/// **at the corner** — while `e_y` is minted `(1,1,−1) → v`, seeding at its far end. So by
/// (ii)'s compare (which ties on `e_z`'s constant-`z` sector and answers `false`) the three
/// bits are `reverse = true / false / true` over the `e_x` / `e_y` / `e_z` owners, and the
/// cap is **not** three-fold symmetric — §8.6.3.1(6)'s own worked cube corner makes the same
/// point with a `true / false / false` vector, the single difference being that its box
/// mints `e_z` from the far end. A mirrored read swaps the widget axes onto the cell, so the
/// `e_y` flank takes `(0.8125, −0.45, −0.875)` where the `e_x` and `e_z` flanks take the
/// unswapped `(0.45, −0.8125, −0.875)` / `(0.8125, −0.875, −0.45)`.
///
/// **The welded apex is pinned on both arms**, because the flank mids are not only rim
/// samples: they are the coarse control-net mids `(i, 0, 1)` the §8.5.5.5 centre stencil
/// consumes, so moving one moves the apex. Six boundverts ⇒ the general grid at valence
/// `s = 6` and §8.9.5's worked recipe — seed `centre_0` by the (b) fullness blend, run the
/// one even-`2` subdivision round, take `apex = E − γ_6·F + γ_6·centre_0` with
/// `γ_6 = 0.523423277`. Both expectations below are computed from that model, not read
/// back from the fill:
///
/// - **analytic** — §8.9.5's worked value verbatim. `fullness = 0.559` ((f.3) law **F2**'s
///   circle table at `n = 2`) gives `centre_0 = (0.8824, −0.8824, −0.8824)`, the
///   `(−1/6, +4/3, −1/6)` tap of §8.5.5.5(c) sharpens all six mids, and the apex lands at
///   `(0.859816, −0.859816, −0.859816)`.
/// - **custom** — three inputs change and nothing else. (i) The three flank mids are the
///   curve's, as above; the three arc mids are unchanged by the carve-out. (ii) The fullness
///   is law **F1**, `(1/(2n))·Σ_{i < n} (x_i + y_i) = ((1 + 0) + (0.5 + 0.25))/4 = 0.4375`,
///   consumed through (b)'s *reflected* blend `(2·centroid − v) + 2·fullness·(v − centroid)`,
///   so `centre_0 = (0.7, −0.7, −0.7)`. (iii) §8.5.5.5(c)'s "a custom profile is not tapped
///   at all" suppresses the sharpening tap for the whole ring — the carve-out's analytic
///   arcs included, the tap being keyed on the operation's profile type the way law **F1**
///   is. The apex is then `(0.707719, −0.710201, −0.712683)`, whose asymmetry is the `e_y`
///   flank's mirrored read carried through the stencil.
///
/// That second arm is the pin for both effects the curve threading brings to a *mitered*
/// ring for the first time: control-net mids read off the curve, and a tap suppression that
/// is anything but a no-op at `N_in = 2` (where only the half-chain and Jacobi scope rules
/// go inert, not the tap itself).
#[test]
fn custom_profile_miter_ring_flank_reads_the_curve_while_the_arc_stays_analytic() {
    let curve = BevelCurveProfile::from_points(&[
        (1.0, 0.0, BevelHandle::Vector),
        (0.5, 0.25, BevelHandle::Vector),
        (0.0, 1.0, BevelHandle::Vector),
    ]);
    let spread = 0.3;
    let control_point = Vec3::new(0.45, -0.8125, -0.875);
    let circle = Vec3::new(0.45, -0.926_776_7, -0.926_776_7);
    let arc_bulge = Vec3::new(0.662_132_0, -1.0, -0.662_132_0);
    let arc_start = Vec3::new(0.45, -0.75, -1.0);

    let (custom_mesh, custom_result) =
        cube_corner_adj_inner_miter_profile(Some(&curve), spread, 2, 0.5, BevelMiter::Arc);
    custom_mesh.validate_topology();
    let (analytic_mesh, analytic_result) =
        cube_corner_adj_inner_miter_profile(None, spread, 2, 0.5, BevelMiter::Arc);
    analytic_mesh.validate_topology();

    // Both arms actually built the miter — the ring carries the spread-pushed arc pair.
    for (mesh, result, arm) in [
        (&custom_mesh, &custom_result, "custom"),
        (&analytic_mesh, &analytic_result, "analytic"),
    ] {
        assert_eq!(
            result_has_vert_near(mesh, result, arc_start),
            1,
            "the {arm} arm's inner miter fired (arc start `spread` out along −x)",
        );
    }

    // The flank: the custom arm sits on the control point, the analytic arm on the circle,
    // and neither carries the other's sample anywhere in the mesh.
    assert_eq!(
        result_has_vert_near(&custom_mesh, &custom_result, control_point),
        1,
        "the mitered flank's lone interior sample is the widget control point",
    );
    assert_eq!(
        result_has_vert_near(&custom_mesh, &custom_result, circle),
        0,
        "…and nothing on that ring sits on the circular cross-section",
    );
    assert_eq!(
        result_has_vert_near(&analytic_mesh, &analytic_result, circle),
        1,
        "control: the analytic arm's same flank is the circle",
    );
    assert_eq!(
        result_has_vert_near(&analytic_mesh, &analytic_result, control_point),
        0,
        "…and never the control net",
    );

    // The carve-out: the arc-start sector's own chain is analytic under both.
    for (mesh, result, arm) in [
        (&custom_mesh, &custom_result, "custom"),
        (&analytic_mesh, &analytic_result, "analytic"),
    ] {
        assert_eq!(
            result_has_vert_near(mesh, result, arc_bulge),
            1,
            "the {arm} arm's arc keeps the closed-form √2 bulge (the miter carve-out)",
        );
    }

    // The other two flanks, read through their own §8.6.3.1(6)(i) seed ends: `e_z`'s
    // unswapped like `e_x`'s, `e_y`'s mirrored. They are the remaining two control-net mids
    // the apex below is built from.
    for (mid, edge) in [
        (Vec3::new(0.8125, -0.875, -0.45), "e_z"),
        (Vec3::new(0.8125, -0.45, -0.875), "e_y"),
    ] {
        assert_eq!(
            result_has_vert_near(&custom_mesh, &custom_result, mid),
            1,
            "the {edge} flank reads the curve through its own handedness bit",
        );
    }

    // The welded apex of the six-boundvert cap, from §8.9.5's model on both arms (the doc
    // comment carries the derivation). The custom arm is where the flank mids reach the
    // centre stencil and where §8.5.5.5(c)'s tap goes away.
    for (mesh, result, apex, arm) in [
        (
            &analytic_mesh,
            &analytic_result,
            Vec3::new(0.859_815_7, -0.859_815_7, -0.859_815_7),
            "analytic",
        ),
        (
            &custom_mesh,
            &custom_result,
            Vec3::new(0.707_718_5, -0.710_200_7, -0.712_682_9),
            "custom",
        ),
    ] {
        assert_eq!(
            result_has_vert_near(mesh, result, apex),
            1,
            "the {arm} arm's mitered cap welds its apex at {apex:?} (§8.9.5)",
        );
    }
}

/// BEVEL.md §8.6.3.1(4).4 row 1, at the one resolution where the schedule's two arms
/// disagree: the coarse control-net mid a mitered cap's centre stencil consumes is a read
/// at `m = 2`, so at `segments == 2` it is `target[1]` — the curve's own sample — and not
/// the analytic superellipse the `profile` scalar names.
///
/// Driven through `coarse_flank_mid` on the *un-translated* `{0,1}` cell of the cube
/// corner (origin `(0.75, −0.75, −0.75)`, `a_axis = (0, 0, −0.25)`, `b_axis =
/// (0, −0.25, 0)`), read **forward** — `reverse = false` swaps the widget axes, so
/// `(x, y) = (0.5, 0.25)` enters the cell as `(u, v) = (0.25, 0.5)` and lands at
/// `(0.75, −0.875, −0.8125)`, the *mirror* of the `reverse = true` reading the whole-mesh
/// arm above pins. That the two differ is the point: the bit rides the read, and a
/// generator that ignored it would mirror the cap's interior off its own rim.
///
/// The mid is also **raw**: `chain_positions`' §8.2.2.1.1 read-back is not applied, which
/// on this all-vector net is the identity anyway — so the value separates custom from
/// analytic, not resampled from raw.
#[test]
fn custom_profile_coarse_flank_mid_reads_the_curve_through_its_own_handedness() {
    let curve = BevelCurveProfile::from_points(&[
        (1.0, 0.0, BevelHandle::Vector),
        (0.5, 0.25, BevelHandle::Vector),
        (0.0, 1.0, BevelHandle::Vector),
    ]);
    let chains = CustomChains::new(&curve, 2);

    let mut mesh = LoopMesh::new();
    let start = mesh.create_vertex(Vec3::new(0.75, -1.0, -0.75));
    let end = mesh.create_vertex(Vec3::new(0.75, -0.75, -1.0));
    let apex = mesh.create_vertex(Vec3::new(1.0, -1.0, -1.0));
    let far = mesh.create_vertex(Vec3::new(-1.0, -1.0, -1.0));
    let edge = mesh.create_edge([apex, far], EdgeCreateFlags::NONE);
    let rescue = MidRescue {
        vertex: mesh.vertex_position(apex),
        neighbour_dirs: None,
    };

    let forward = mesh.coarse_flank_mid(
        start,
        end,
        edge,
        0.5,
        Some(CustomChainRead::edge_sector(&chains, false)),
        rescue,
    );
    let mirrored = mesh.coarse_flank_mid(
        start,
        end,
        edge,
        0.5,
        Some(CustomChainRead::edge_sector(&chains, true)),
        rescue,
    );
    let analytic = mesh.coarse_flank_mid(start, end, edge, 0.5, None, rescue);

    assert!(
        (forward - Vec3::new(0.75, -0.875, -0.8125)).length() < 1.0e-6,
        "the forward read swaps the widget axes onto the cell, got {forward:?}",
    );
    assert!(
        (mirrored - Vec3::new(0.75, -0.8125, -0.875)).length() < 1.0e-6,
        "the mirrored read drops the swap, got {mirrored:?}",
    );
    assert!(
        (analytic - Vec3::new(0.75, -0.926_776_7, -0.926_776_7)).length() < 1.0e-6,
        "control: without a curve the same cell samples the circle, got {analytic:?}",
    );
}

/// The two arms above, stated as the invariant behind them (BEVEL.md §8.6.3): a custom
/// profile "bypasses §8.6.2 and §8.6.4 entirely and supplies its chain directly", and
/// every rule that would otherwise consult the exponent — (f.1) steps 1 and 3, the (f.3)
/// `fullness` law, whose custom branch **F1** is tested before F2's line/circle closed
/// forms — is keyed on the *type*. The whole mesh must therefore be invariant under the
/// slider. The fixture's three beveled edges *do* run out to terminals at their far ends
/// (`(−1, −1, −1)`, `(1, 1, −1)`, `(1, −1, 1)`), so the terminal fill is live here — but the
/// operator's one genuinely `profile_type`-keyed positional rule lives there and keys on the
/// *type* too: `terminal_leg_slide_distance` takes the `√2` leg slide for **any** custom
/// profile, whatever the scalar reads. So the terminals are slider-invariant under a fixed
/// custom curve alongside everything else, rather than being absent.
///
/// Sweeping the four snapped sentinels plus a genuinely curved value makes this a real
/// gate: before the refusal landed, `profile = 0.0` welded, `0.25` went flat and `0.5`
/// built the octant grid, so three of these five arms disagreed with each other.
#[test]
fn custom_profile_output_is_independent_of_the_superellipse_slider() {
    let curve = BevelCurveProfile::from_points(&[
        (1.0, 0.0, BevelHandle::Vector),
        (0.5, 0.25, BevelHandle::Vector),
        (0.0, 1.0, BevelHandle::Vector),
    ]);
    for segments in [2u32, 3, 4] {
        let (baseline_mesh, baseline) = cube_corner_adj_custom(segments, 0.5, &curve);
        let reference: Vec<Vec3> = baseline
            .verts
            .iter()
            .map(|&vertex| baseline_mesh.vertex_position(vertex))
            .collect();
        for profile in [0.0_f32, 0.25, 0.7, 0.99] {
            let (mesh, result) = cube_corner_adj_custom(segments, profile, &curve);
            mesh.validate_topology();
            assert_eq!(
                (mesh.vertex_count(), mesh.edge_count(), mesh.face_count()),
                (
                    baseline_mesh.vertex_count(),
                    baseline_mesh.edge_count(),
                    baseline_mesh.face_count()
                ),
                "the slider cannot move a custom bevel's census \
                 (profile = {profile}, segments = {segments})",
            );
            let moved: Vec<Vec3> = result
                .verts
                .iter()
                .map(|&vertex| mesh.vertex_position(vertex))
                .collect();
            // The position check below is one-directional containment, so a `moved` list
            // that lost an entry — two output verts collapsed onto one — would pass it.
            // The census above cannot stand in: it counts the whole mesh, not how much of
            // it the bevel claims as its own output.
            assert_eq!(
                moved.len(),
                reference.len(),
                "the slider cannot change how many verts the bevel emits \
                 (profile = {profile}, segments = {segments})",
            );
            for position in &moved {
                assert!(
                    reference
                        .iter()
                        .any(|candidate| (*candidate - *position).length() <= 1.0e-5),
                    "…nor a single one of its positions ({position:?} at \
                     profile = {profile}, segments = {segments})",
                );
            }
        }
    }
}

/// The invariant above, carried onto a **mitered** junction (BEVEL.md §8.9.5 / §8.9.6).
///
/// A miter's admission is keyed on the profile **type**, never on the slider: the fill a
/// custom ring takes is (f.1) step 1's general grid whatever the slider resolves to. So a
/// miter that declined on the exponent under `profile_type = CUSTOM` would drop the `+1` /
/// `+2` boundverts of §8.9.6 and fall back to the sharp ring — a whole-mesh census
/// divergence keyed on a value the admission never reads.
///
/// **Positions are swept on one arm only, because only one of them is a law.** A `PATCH`
/// start's two legs are straight chords at every exponent, and for a stronger reason than
/// a different exponent (§8.9.5): each leg's stored midpoint *is* its own chord midpoint,
/// so its three anchors are collinear, the cell degenerates and no unit-square map exists
/// — the samples are plain interpolation whatever the exponent or the profile type says.
/// An `ARC` start's chain is the opposite: §8.6.3.1(4).4's carve-out builds its analytic
/// spacing pair "from the `profile` scalar", so the slider *does* move it under a curve,
/// and that is pinned positively by
/// [`custom_profile_outer_miter_arc_chain_tracks_the_superellipse_slider`].
///
/// Both builders are swept, because the sweep visits `profile = 0.99` — inside the square
/// sentinel, the one band where the two builders' admissions differ (the inner declines it,
/// lacking the (f.5) seeds; the outer mints them and admits). Under a **curve** neither may
/// read the slider at all, because (f.1) step 1's `not a custom profile` conjunct keeps a
/// custom ring off that lattice whatever the sentinel says.
///
/// The `Sharp` contrast is what stops the sweep passing vacuously: without it, a build in
/// which *every* arm declined would be perfectly slider-invariant too.
#[test]
fn custom_profile_miter_admission_is_independent_of_the_superellipse_slider() {
    let curve = BevelCurveProfile::from_points(&[
        (1.0, 0.0, BevelHandle::Vector),
        (0.5, 0.25, BevelHandle::Vector),
        (0.0, 1.0, BevelHandle::Vector),
    ]);
    let inner = |profile: f32, segments: u32, mode: BevelMiter| {
        cube_corner_adj_inner_miter_profile(Some(&curve), 0.3, segments, profile, mode)
    };
    let outer_arc = |profile: f32, segments: u32, mode: BevelMiter| {
        reflex_apex_outer_miter_profile(mode, Some(&curve), profile, segments)
    };
    let outer_patch = |profile: f32, segments: u32, mode: BevelMiter| {
        let mode = match mode {
            BevelMiter::Sharp => BevelMiter::Sharp,
            _ => BevelMiter::Patch,
        };
        reflex_apex_outer_miter_profile(mode, Some(&curve), profile, segments)
    };
    let builders: [(
        &str,
        &dyn Fn(f32, u32, BevelMiter) -> (LoopMesh, BevelResult),
        &[u32],
        bool,
    ); 3] = [
        ("inner arc", &inner, &[2, 3], false),
        ("outer arc", &outer_arc, &[2, 3], false),
        // `segments = 3` is the load-bearing arm here, not a duplicate of `2`: at
        // `segments ≥ 3` a custom ring reaches the general §8.5.5.5 grid, whose boundary
        // mask and relaxed net both re-read each ring chain's *generator* at doubled
        // resolutions. A patch leg's generator is the straight chord (§8.9.5), so if that
        // straightness were secretly conditioned on the profile type the divergence would
        // surface here and nowhere at `segments = 2`.
        ("outer patch", &outer_patch, &[2, 3], true),
    ];

    for (name, build, counts, positions_invariant) in builders {
        for &segments in counts {
            let (baseline_mesh, baseline) = build(0.5, segments, BevelMiter::Arc);
            let reference: Vec<Vec3> = baseline
                .verts
                .iter()
                .map(|&vertex| baseline_mesh.vertex_position(vertex))
                .collect();
            let (sharp_mesh, _sharp) = build(0.5, segments, BevelMiter::Sharp);
            assert_ne!(
                (
                    baseline_mesh.vertex_count(),
                    baseline_mesh.edge_count(),
                    baseline_mesh.face_count()
                ),
                (
                    sharp_mesh.vertex_count(),
                    sharp_mesh.edge_count(),
                    sharp_mesh.face_count()
                ),
                "the {name} baseline actually mitered (segments = {segments})",
            );

            for profile in [0.0_f32, 0.25, 0.7, 0.99] {
                let (mesh, result) = build(profile, segments, BevelMiter::Arc);
                mesh.validate_topology();
                assert_eq!(
                    (mesh.vertex_count(), mesh.edge_count(), mesh.face_count()),
                    (
                        baseline_mesh.vertex_count(),
                        baseline_mesh.edge_count(),
                        baseline_mesh.face_count()
                    ),
                    "the slider cannot decide whether a custom {name} miter fires \
                     (profile = {profile}, segments = {segments})",
                );
                let moved: Vec<Vec3> = result
                    .verts
                    .iter()
                    .map(|&vertex| mesh.vertex_position(vertex))
                    .collect();
                assert_eq!(
                    moved.len(),
                    reference.len(),
                    "…nor how many verts it emits (profile = {profile}, \
                     segments = {segments}, {name})",
                );
                if !positions_invariant {
                    continue;
                }
                for position in &moved {
                    assert!(
                        reference
                            .iter()
                            .any(|candidate| (*candidate - *position).length() <= 1.0e-5),
                        "…nor a single one of its positions ({position:?} at \
                         profile = {profile}, segments = {segments}, {name})",
                    );
                }
            }
        }
    }
}

/// BEVEL.md §8.6.3.1(4).4's miter carve-out, read literally: the second spacing pair it
/// hands an arc start is the analytic superellipse **of the `profile` scalar**, so moving
/// that slider moves a mitered sector's chain even under `profile_type = CUSTOM`. What the
/// carve-out withholds is the *curve*, not the scalar.
///
/// The companion invariant is
/// [`custom_profile_miter_admission_is_independent_of_the_superellipse_slider`]: admission
/// is still slider-blind, and a `PATCH` start's legs are still slider-flat — but for
/// §8.9.5's independent collinear-anchor reason, not because the type pins the exponent.
///
/// Pinned in closed form at `segments = 2` on the single-face reflex junction, whose arc
/// cell is symmetric, so the §8.2.2.1.1 resample is the identity there and raw = emitted.
/// §8.9.5's general reduction puts the lone interior sample at
/// `(2 − 2m)·½(B1 + B3) + (2m − 1)·midpoint` with `m` read off the *sampler*: `0.5` inside
/// the line sentinel (the plain chord midpoint, no pull), `1/√2` at the circle, `√0.7` on
/// the curved exponent and `1` inside the square sentinel — where the sample *is* the
/// stored corner the miter replaced. Four exponents, four distinct answers, all on the
/// bisector chord; a build that re-pinned the custom arc to the circle collapses them onto
/// the second row.
///
/// The `segments = 3` station is where raw and emitted part company (the resample is no
/// longer the identity at an odd count, §8.9.5.1), so it pins that the slider reaches the
/// *generator* and not merely one closed-form sample.
#[test]
fn custom_profile_outer_miter_arc_chain_tracks_the_superellipse_slider() {
    let curve = BevelCurveProfile::from_points(&[
        (1.0, 0.0, BevelHandle::Vector),
        (0.5, 0.25, BevelHandle::Vector),
        (0.0, 1.0, BevelHandle::Vector),
    ]);
    // §8.9.5.1 worked trace B's frame (offset 0.2, spread 0, ARC) — the same cell
    // `outer_miter_arc_samples_track_the_run_profile_exponent` reads the analytic family on.
    let b1 = Vec3::new(-0.51500, 0.27005, 0.72995);
    let b3 = Vec3::new(0.51500, 0.27005, 0.72995);
    let midpoint = Vec3::new(0.0, -0.24495, 1.24495);
    let round_sample = Vec3::new(0.0, 0.05673, 0.94327);
    // The frame literals above are quoted to five places, so the derived expectations carry
    // a few `1e-6` of quoting error — far inside this band, and four orders inside the
    // ≈`0.2` that separates any two of the four rows.
    let emitted_near = |mesh: &LoopMesh, result: &BevelResult, target: Vec3| {
        result
            .verts
            .iter()
            .filter(|&&vertex| (mesh.vertex_position(vertex) - target).length() < 1.0e-4)
            .count()
    };

    for (profile, expected) in [
        (0.25_f32, Vec3::new(0.0, 0.27005, 0.72995)),
        (0.5, round_sample),
        (0.7, Vec3::new(0.0, -0.07671, 1.07671)),
        (0.99, midpoint),
    ] {
        let (mesh, result) =
            reflex_apex_outer_miter_profile(BevelMiter::Arc, Some(&curve), profile, 2);
        mesh.validate_topology();
        assert_eq!(
            emitted_near(&mesh, &result, expected),
            1,
            "custom profile {profile}: the arc's lone interior sample is {expected:?}",
        );
        if profile != 0.5 {
            assert!(
                (expected - round_sample).length() > 1.0e-2,
                "profile {profile}'s answer is far from the circle's, so the absence \
                 check below has teeth",
            );
            assert_eq!(
                emitted_near(&mesh, &result, round_sample),
                0,
                "custom profile {profile}: the forced-circle sample must be gone",
            );
        }
    }

    for profile in [0.25_f32, 0.7, 0.99] {
        let (mesh, result) =
            reflex_apex_outer_miter_profile(BevelMiter::Arc, Some(&curve), profile, 3);
        mesh.validate_topology();
        let chain = emitted_arc_chain(b1, b3, midpoint, profile, 3);
        let round = emitted_arc_chain(b1, b3, midpoint, 0.5, 3);
        for k in 1..3 {
            assert!(
                (chain[k] - round[k]).length() > 1.0e-2,
                "profile {profile}, sample {k}: the two chains are genuinely apart",
            );
            assert_eq!(
                emitted_near(&mesh, &result, chain[k]),
                1,
                "custom profile {profile}, segments 3: sample {k} is {:?}",
                chain[k],
            );
            assert_eq!(
                emitted_near(&mesh, &result, round[k]),
                0,
                "custom profile {profile}, segments 3: the circle's sample {k} must be gone",
            );
        }
    }
}

/// BEVEL.md §8.6.3.1: the LINE preset resamples to the straight diagonal chord in the
/// cell frame, and a vector-handle polyline resamples to that polyline (linear per
/// segment) with an even-parameter budget split — the exact chains the two landed A/B
/// evaluation-model pins read back through the FFI.
#[test]
fn custom_profile_sample_chain_matches_the_evaluation_model() {
    // LINE preset: two auto points whose handles fall on the line → the diagonal.
    let line = BevelCurveProfile::preset_line();
    let chain = line.sample_chain(2, false);
    let expected = [(0.0, 1.0), (0.5, 0.5), (1.0, 0.0)];
    assert_eq!(chain.len(), expected.len(), "segments + 1 samples");
    for (got, want) in chain.iter().zip(expected.iter()) {
        assert!(
            (got.0 - want.0).abs() < 1.0e-6 && (got.1 - want.1).abs() < 1.0e-6,
            "line preset sample {got:?} != {want:?}",
        );
    }

    // A vector-handle polyline: the chain is the polyline itself (each segment a
    // straight chord), swapped into the cell frame `u = y, v = x`.
    let poly = BevelCurveProfile::from_points(&[
        (1.0, 0.0, BevelHandle::Vector),
        (0.5, 0.25, BevelHandle::Vector),
        (0.0, 1.0, BevelHandle::Vector),
    ]);
    let chain2 = poly.sample_chain(2, false);
    let want2 = [(0.0, 1.0), (0.25, 0.5), (1.0, 0.0)];
    assert_eq!(chain2.len(), want2.len(), "segments + 1 samples");
    for (got, want) in chain2.iter().zip(want2.iter()) {
        assert!(
            (got.0 - want.0).abs() < 1.0e-6 && (got.1 - want.1).abs() < 1.0e-6,
            "vector polyline sample {got:?} != {want:?}",
        );
    }

    // §8.6.3.1(4): at `segments = 4` the two segments split the budget evenly (2 each),
    // sampled at even parameter — the chain is the polyline sampled at its midpoints.
    let chain4 = poly.sample_chain(4, false);
    let want4 = [
        (0.0, 1.0),
        (0.125, 0.75),
        (0.25, 0.5),
        (0.625, 0.25),
        (1.0, 0.0),
    ];
    assert_eq!(chain4.len(), 5, "segments + 1 samples");
    for (got, want) in chain4.iter().zip(want4.iter()) {
        assert!(
            (got.0 - want.0).abs() < 1.0e-6 && (got.1 - want.1).abs() < 1.0e-6,
            "segments = 4 sample {got:?} != {want:?}",
        );
    }
}

/// BEVEL.md §8.6.3.1(4).1: the leftover budget goes to the **sharpest** control segments
/// first (descending `turn`), not the flattest. Pinned on the guide's mirror pair of
/// all-auto four-point nets — the mirror is what makes this conclusive: "lowest index
/// wins" would explain EARLY_TURN alone, "highest index wins" LATE_TURN alone, and the
/// inverted measure (`π − turn`) gets both backwards.
///
/// The `n = 3, 4, 5` allotments are quoted from the guide's worked table; the `n = 1`
/// pair is derived from the same section's `n < E` clause (a top-1 cut of the same
/// ranking), not quoted.
#[test]
fn custom_profile_leftover_budget_goes_to_the_sharpest_control_segment() {
    let allotments = |net: &[(f32, f32)], n: usize| -> Vec<usize> {
        let profile = BevelCurveProfile::from_points(
            &net.iter()
                .map(|&(x, y)| (x, y, BevelHandle::Auto))
                .collect::<Vec<_>>(),
        );
        let handles = profile.resolved_handles();
        distribute_budget(&profile.points, &handles, n)
    };
    let late_turn = [(1.0, 0.0), (0.85, 0.10), (0.20, 0.35), (0.0, 1.0)];
    let early_turn = [(1.0, 0.0), (0.35, 0.20), (0.10, 0.85), (0.0, 1.0)];

    assert_eq!(
        allotments(&late_turn, 3),
        vec![1, 1, 1],
        "no remainder to rank"
    );
    assert_eq!(
        allotments(&early_turn, 3),
        vec![1, 1, 1],
        "no remainder to rank"
    );

    // One leftover advance: it lands on the sharpest segment, which is the *last* one
    // for LATE_TURN — so a winner of index 0 here is the inverted measure.
    assert_eq!(allotments(&late_turn, 4), vec![1, 1, 2]);
    assert_eq!(allotments(&early_turn, 4), vec![2, 1, 1]);

    // Two leftover advances: sharpest, then next-sharpest.
    assert_eq!(allotments(&late_turn, 5), vec![1, 2, 2]);
    assert_eq!(allotments(&early_turn, 5), vec![2, 2, 1]);

    // The `n < E` branch ranks the same way: the single advance goes to the sharpest.
    assert_eq!(allotments(&late_turn, 1), vec![0, 0, 1]);
    assert_eq!(allotments(&early_turn, 1), vec![1, 0, 0]);
}

/// Assert a widget chain against a hard-coded expectation from the guide's tables.
fn assert_widget_chain(got: &[(f32, f32)], want: &[(f32, f32)], what: &str) {
    assert_eq!(
        got.len(),
        want.len(),
        "{what}: chain length {got:?} != {want:?}"
    );
    for (k, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        assert!(
            (g.0 - w.0).abs() < 1.0e-4 && (g.1 - w.1).abs() < 1.0e-4,
            "{what}: sample {k} {g:?} != {w:?} (chain {got:?})",
        );
    }
}

/// BEVEL.md §8.6.3.1(4).1's sub-budget cut ranks over **every** control segment, straight
/// ones included. On the guide's mixed net (`E = 4`, only two curved segments, turns
/// `0° / 19.04° / 2.20° / 0°`) a curved-only cut hands out just two advances at
/// `segments = 3`, and the short chain that follows becomes a ring with the wrong vertex
/// count — no panic, no signal. Chains quoted from the guide's worked table.
///
/// *Which* straight segment the `segments = 3` cut takes is settled by §8.6.3.1(4).1's
/// rule (b): the two are not tied at all — segment `0` carries a `½·2⁻²⁴` round-trip
/// residue and segment `3` scores exactly `0`, so the cut is `{0, 1, 2}` strictly. The
/// alternative `{1, 2, 3}` a `turn = 0` short-circuit produces differs at **both** ends.
#[test]
fn custom_profile_sub_budget_cut_reaches_into_straight_control_segments() {
    let mixed = BevelCurveProfile::from_points_with_handles(&[
        (1.0, 0.0, BevelHandle::Vector, BevelHandle::Vector),
        (0.80, 0.05, BevelHandle::Vector, BevelHandle::Auto),
        (0.50, 0.40, BevelHandle::Auto, BevelHandle::Auto),
        (0.15, 0.85, BevelHandle::Auto, BevelHandle::Vector),
        (0.0, 1.0, BevelHandle::Vector, BevelHandle::Vector),
    ]);

    assert_widget_chain(
        &mixed.widget_chain(1),
        &[(0.80, 0.05), (0.50, 0.40)],
        "mixed net segments = 1",
    );
    assert_widget_chain(
        &mixed.widget_chain(2),
        &[(0.80, 0.05), (0.50, 0.40), (0.15, 0.85)],
        "mixed net segments = 2",
    );
    // The under-fill case: `curved_count = 2 < 3 < E = 4`.
    let (turns, ranked) = turn_ranking(&mixed);
    assert_eq!(
        turns[0] / TURN_RESIDUE_SCALE,
        0.5,
        "segment 0's residue separates it from segment 3's exact zero: {turns:?}",
    );
    assert_eq!(turns[3], 0.0);
    assert_eq!(ranked, vec![1, 2, 0, 3]);
    assert_widget_chain(
        &mixed.widget_chain(3),
        &[(1.00, 0.00), (0.80, 0.05), (0.50, 0.40), (0.15, 0.85)],
        "mixed net segments = 3 (select {0, 1, 2}: the residue, not the tie-break)",
    );
    assert_widget_chain(
        &mixed.widget_chain(4),
        &[
            (1.00, 0.00),
            (0.80, 0.05),
            (0.50, 0.40),
            (0.15, 0.85),
            (0.00, 1.00),
        ],
        "mixed net segments = 4 (n = E, corner-to-corner)",
    );
}

/// BEVEL.md §8.6.3.1(4).3: on the sub-budget branch neither chain end is pinned, and a
/// run of skipped segments is a **jump** — the control points between two non-adjacent
/// selected segments never appear. All chains quoted from the guide's reference tables.
#[test]
fn custom_profile_sub_budget_chain_has_no_forced_endpoints() {
    let all_auto = |net: &[(f32, f32)]| {
        BevelCurveProfile::from_points(
            &net.iter()
                .map(|&(x, y)| (x, y, BevelHandle::Auto))
                .collect::<Vec<_>>(),
        )
    };
    // Turn order 0 < 1 < 2, so the top-2 cut is {1, 2} and the start corner drops out.
    let late_turn = late_turn_net();
    assert_widget_chain(
        &late_turn.widget_chain(2),
        &[(0.85, 0.10), (0.20, 0.35), (0.00, 1.00)],
        "LATE_TURN segments = 2",
    );
    assert_widget_chain(
        &late_turn.widget_chain(1),
        &[(0.20, 0.35), (0.00, 1.00)],
        "LATE_TURN segments = 1",
    );
    // The mirror: turn order 2 < 1 < 0, top-2 cut is {0, 1}, and the *end* corner drops.
    let early_turn = all_auto(&[(1.0, 0.0), (0.35, 0.20), (0.10, 0.85), (0.0, 1.0)]);
    assert_widget_chain(
        &early_turn.widget_chain(2),
        &[(1.00, 0.00), (0.35, 0.20), (0.10, 0.85)],
        "EARLY_TURN segments = 2",
    );

    // The split net: curvature on the two *outer* control segments only (turns
    // `24.70° / 0° / 0° / 35.13°`), so the top-2 cut is the non-adjacent {0, 3} and the
    // chain steps straight from P0 to P3 — P1 and P2 never appear.
    let split = split_net();
    assert_widget_chain(
        &split.widget_chain(1),
        &[(0.10, 0.60), (0.00, 1.00)],
        "split net segments = 1",
    );
    assert_widget_chain(
        &split.widget_chain(2),
        &[(1.00, 0.00), (0.10, 0.60), (0.00, 1.00)],
        "split net segments = 2 (the jump)",
    );
    assert_widget_chain(
        &split.widget_chain(4),
        &[
            (1.00, 0.00),
            (0.90, 0.40),
            (0.60, 0.55),
            (0.10, 0.60),
            (0.00, 1.00),
        ],
        "split net segments = 4 (n = E, corner-to-corner)",
    );
}

/// BEVEL.md §8.6.3.1(4).1's normalisation convention. The `turn` measure normalises both
/// tangents before reading the chord angle, and the zero vector normalises to itself, so
/// **exactly one** zero tangent scores a fixed `π/3 = 60°` — near the top of the ranking
/// — while **both** zero score `0°`, its bottom. Reading the degenerate case as `0°`
/// (an `atan2` on the raw inputs) inverts the ranking for exactly those segments.
///
/// The `60°` half runs on the supports preset's own control path, which §8.6.3.1(3)
/// gives at `points = 5` as `(1,0) · (0.85355,0.85355) · (0.5,1) · (0.5,1) · (0,1)` with
/// vector handles at index 0, 3 and 4 and auto at 1 and 2. Its duplicated point leaves a
/// zero-length segment whose arrival tangent is zero and whose departure is not; the
/// four turns are `28.88° / 28.88° / 60° / 0°`.
#[test]
fn custom_profile_degenerate_tangent_ranking_follows_the_chord_convention() {
    let arc = 0.5 + 0.5 * std::f32::consts::FRAC_1_SQRT_2;
    let supports = BevelCurveProfile::from_points_with_handles(&[
        (1.0, 0.0, BevelHandle::Vector, BevelHandle::Vector),
        (
            1.0 - 0.5 * (1.0 - std::f32::consts::FRAC_1_SQRT_2),
            arc,
            BevelHandle::Auto,
            BevelHandle::Auto,
        ),
        (0.5, 1.0, BevelHandle::Auto, BevelHandle::Auto),
        (0.5, 1.0, BevelHandle::Vector, BevelHandle::Vector),
        (0.0, 1.0, BevelHandle::Vector, BevelHandle::Vector),
    ]);

    // segments = 1: the top-1 cut is the degenerate segment 2, so the whole chain is the
    // coincident pair. Under the `0°` reading it would take segment 0 and return
    // (1, 0) (0.85355, 0.85355) instead — a completely different chain.
    assert_widget_chain(
        &supports.widget_chain(1),
        &[(0.5, 1.0), (0.5, 1.0)],
        "supports segments = 1",
    );
    // segments = 3: the end shoulder is uniquely lowest-ranked, so the cut is {0, 1, 2}
    // whatever the three above it do among themselves — the chain starts at the widget
    // corner (1, 0) but stops short of (0, 1).
    assert_widget_chain(
        &supports.widget_chain(3),
        &[(1.0, 0.0), (0.85355, 0.85355), (0.5, 1.0), (0.5, 1.0)],
        "supports segments = 3",
    );

    // Both tangents zero: a duplicated control point carrying **vector** handles on both
    // facing sides makes the departure zero as well, and the difference chord is 0. Here
    // the net has five control points, so E = 4 and the segments are 0..=3. Segment 2 is
    // the coincident one; segments 0 and 1 measure 13.28° each and segments 2 and 3
    // measure 0°. At segments = 2 the cut is therefore {0, 1} — tie-free at its boundary
    // — and the chain is the first three control points. Scoring the duplicate 60° would
    // put it in the cut and start the chain at (0.4, 0.4) instead.
    let duplicated = BevelCurveProfile::from_points_with_handles(&[
        (1.0, 0.0, BevelHandle::Vector, BevelHandle::Auto),
        (0.7, 0.1, BevelHandle::Auto, BevelHandle::Auto),
        (0.4, 0.4, BevelHandle::Auto, BevelHandle::Vector),
        (0.4, 0.4, BevelHandle::Vector, BevelHandle::Auto),
        (0.0, 1.0, BevelHandle::Auto, BevelHandle::Vector),
    ]);
    assert_widget_chain(
        &duplicated.widget_chain(2),
        &[(1.0, 0.0), (0.7, 0.1), (0.4, 0.4)],
        "duplicated-point net segments = 2",
    );
    // segments = 1 sits inside the 13.28° tie between segments 0 and 1, so only the
    // tie-independent fact is asserted: the cut does not take the coincident segment.
    let single = duplicated.widget_chain(1);
    assert_eq!(single.len(), 2);
    assert!(
        single[0] != single[1],
        "the both-zero segment must not outrank a genuinely curved one: {single:?}",
    );
}

/// BEVEL.md §8.6.3.1(4).1's **obtuse arm**. When the two endpoint tangents point more
/// than a right angle apart the turn is read off the *sum* chord as
/// `π − 2·asin(|d̂ + â| / 2)` — exactly one input negated. Negating both leaves the
/// difference chord untouched and yields `π − turn`, a clean rank inversion of exactly
/// the obtuse segments and nothing else. Every other custom-profile net in this module
/// is acute throughout and would pass either way.
///
/// The curl-back net puts its two outer points on opposite sides of the middle chord,
/// so the middle segment doubles back: the three turns are `85° / 170° / 85°`. The
/// top-1 cut is therefore the middle segment, uniquely and tie-free, and the chain is
/// its two control points. Under the `π − turn` misreading the turns read
/// `95° / 10° / 95°` and the cut takes an outer segment instead.
#[test]
fn custom_profile_obtuse_turn_outranks_its_acute_flanks() {
    let curl_back = BevelCurveProfile::from_points(&[
        (0.604_557, 0.447_914, BevelHandle::Auto),
        (0.9, 0.5, BevelHandle::Auto),
        (0.1, 0.5, BevelHandle::Auto),
        (0.395_443, 0.447_914, BevelHandle::Auto),
    ]);
    assert_widget_chain(
        &curl_back.widget_chain(1),
        &[(0.9, 0.5), (0.1, 0.5)],
        "curl-back net segments = 1 (the obtuse middle segment)",
    );
}

/// The rank key BEVEL.md §8.6.3.1(4).1 states, evaluated once per control segment: the
/// `turn` of the named segment, and the ranking the whole net induces (descending on
/// `turn`, then descending on index).
fn turn_ranking(net: &BevelCurveProfile) -> (Vec<f32>, Vec<usize>) {
    let handles = net.resolved_handles();
    let turns: Vec<f32> = (0..net.points.len() - 1)
        .map(|s| control_segment_turn(&net.points, &handles, s))
        .collect();
    let mut ranked: Vec<usize> = (0..turns.len()).collect();
    ranked.sort_by(|&a, &b| turns[b].total_cmp(&turns[a]).then(b.cmp(&a)));
    (turns, ranked)
}

/// Per-segment allotments for a net at budget `n` (BEVEL.md §8.6.3.1(4).1).
fn allotments_of(net: &BevelCurveProfile, n: usize) -> Vec<usize> {
    distribute_budget(&net.points, &net.resolved_handles(), n)
}

/// A net of `(x, y)` control points with **vector** handles on both sides of every point —
/// every segment a straight chord, so the whole ranking is rule (b)'s residues and rule
/// (a)'s index tie-break.
fn all_vector_net(net: &[(f32, f32)]) -> BevelCurveProfile {
    BevelCurveProfile::from_points(
        &net.iter()
            .map(|&(x, y)| (x, y, BevelHandle::Vector))
            .collect::<Vec<_>>(),
    )
}

/// The `2⁻²⁴` scale rule (b)'s round-trip residues live at. Dividing a residue by it is
/// exact (a power of two), so the multipliers below are asserted rather than approximated.
const TURN_RESIDUE_SCALE: f32 = 1.0 / (1u32 << 24) as f32;

/// BEVEL.md §8.6.3.1(4).1 **rule (b)**: a geometrically straight (vector–vector) control
/// segment does *not* necessarily score exactly `0`. Its two endpoint tangents are the same
/// chord reconstructed twice by two different `f32` round-trips — `(P_s + chord·fl(1/3)) −
/// P_s` and `P_{s+1} − (P_{s+1} − chord·fl(1/3))` — which need not land on the same float,
/// so the score is a cancellation residue at the `2⁻²⁴` scale. Those residues are part of
/// the rank key and usually order straight segments strictly before rule (a) is consulted.
///
/// The guide's `E = 3` all-vector net, on which a naive reading predicts a total tie:
/// residues `0`, `√2·2⁻²⁴`, `3·2⁻²⁴`, hence the strict ranking `[2, 1, 0]`. The residues
/// are asserted as exact multiples because each of the six arithmetic-spelling traps the
/// guide enumerates moves them — dividing by `3` instead of scaling by `fl(1/3)`, or
/// normalising by componentwise divide instead of reciprocal-multiply, each changes this
/// row and the ranking that falls out of it.
#[test]
fn custom_profile_straight_control_segments_score_rounding_residues_not_zero() {
    let (turns, ranked) = turn_ranking(&all_vector_e3_net());
    assert_eq!(turns[0], 0.0, "segment 0's two round-trips agree exactly");
    assert_eq!(
        turns[1] / TURN_RESIDUE_SCALE,
        std::f32::consts::SQRT_2,
        "segment 1 scores √2·2⁻²⁴ (turns = {turns:?})",
    );
    assert_eq!(
        turns[2] / TURN_RESIDUE_SCALE,
        3.0,
        "segment 2 scores 3·2⁻²⁴ (turns = {turns:?})",
    );
    assert_eq!(ranked, vec![2, 1, 0], "strict; rule (a) never consulted");

    // The guide's allotment / selection table for the same net. `n = 3` and `n = 6` are
    // exact multiples of `E` and pin nothing — the remainder is empty and any ranking
    // reproduces them — so they are here as the control, not as evidence.
    let net = all_vector_e3_net();
    assert_eq!(allotments_of(&net, 3), vec![1, 1, 1], "ranking inert");
    assert_eq!(allotments_of(&net, 6), vec![2, 2, 2], "ranking inert");
    assert_eq!(allotments_of(&net, 4), vec![1, 1, 2]);
    assert_eq!(allotments_of(&net, 5), vec![1, 2, 2]);
    assert_eq!(allotments_of(&net, 7), vec![2, 2, 3]);
    assert_eq!(allotments_of(&net, 8), vec![2, 3, 3]);

    // The `n < E` sub-chain branch, where the cut decides both of the chain's ends.
    assert_widget_chain(
        &net.widget_chain(1),
        &[(0.35, 0.40), (0.00, 1.00)],
        "E = 3 all-vector net segments = 1 (select {2})",
    );
    assert_widget_chain(
        &net.widget_chain(2),
        &[(0.80, 0.12), (0.35, 0.40), (0.00, 1.00)],
        "E = 3 all-vector net segments = 2 (select {2, 1})",
    );
    assert_widget_chain(
        &net.widget_chain(4),
        &[
            (1.00, 0.00),
            (0.80, 0.12),
            (0.35, 0.40),
            (0.175, 0.70),
            (0.00, 1.00),
        ],
        "E = 3 all-vector net segments = 4",
    );
}

/// BEVEL.md §8.6.3.1(4).1: rules (a) and (b) **interleave** within one ranking. On the
/// guide's `E = 5` all-vector net the residues are `(2⁻²⁴, 2⁻²⁴, 2.5·2⁻²⁴, 0, 0)`: segment
/// `2` leads on its residue alone, then rule (b) leaves two tied pairs that rule (a)
/// resolves by descending index — `1` before `0`, then `4` before `3`. Neither rule
/// reproduces `[2, 1, 0, 4, 3]` on its own: an all-tie reading gives `[4, 3, 2, 1, 0]` and
/// an ascending-index tie-break gives `[2, 0, 1, 3, 4]`, and both move the `n = 1` chain.
#[test]
fn custom_profile_residue_and_index_tie_break_interleave_in_one_ranking() {
    let net = all_vector_net(&[
        (1.0, 0.0),
        (0.9, 0.05),
        (0.7, 0.15),
        (0.4, 0.3),
        (0.15, 0.6),
        (0.0, 1.0),
    ]);
    let (turns, ranked) = turn_ranking(&net);
    let multiples: Vec<f32> = turns.iter().map(|t| t / TURN_RESIDUE_SCALE).collect();
    assert_eq!(
        multiples,
        vec![1.0, 1.0, 2.5, 0.0, 0.0],
        "turns = {turns:?}"
    );
    assert_eq!(ranked, vec![2, 1, 0, 4, 3]);

    assert_widget_chain(
        &net.widget_chain(1),
        &[(0.70, 0.15), (0.40, 0.30)],
        "E = 5 residue net segments = 1 (select {2})",
    );
    assert_widget_chain(
        &net.widget_chain(2),
        &[(0.90, 0.05), (0.70, 0.15), (0.40, 0.30)],
        "E = 5 residue net segments = 2 (select {1, 2})",
    );
    // The item-3 jump: segment 3 is outranked, so `P₃` never appears.
    assert_widget_chain(
        &net.widget_chain(4),
        &[
            (1.00, 0.00),
            (0.90, 0.05),
            (0.70, 0.15),
            (0.15, 0.60),
            (0.00, 1.00),
        ],
        "E = 5 residue net segments = 4 (select {0, 1, 2, 4}, jumping segment 3)",
    );
    assert_eq!(allotments_of(&net, 6), vec![1, 1, 2, 1, 1]);
    assert_eq!(allotments_of(&net, 7), vec![1, 2, 2, 1, 1]);
}

/// BEVEL.md §8.6.3.1(4).1 rule (a) acting **entirely on its own**: an axis-aligned
/// staircase has one exactly-zero chord component per segment, so both of rule (b)'s
/// round-trips normalise to the same axis unit vector bit-for-bit and every residue
/// cancels. The whole ranking is then the descending-index tie-break, and the sub-budget
/// cut is the staircase's **tail** — an ascending-index tie-break samples the other end
/// of the stairs and is visibly wrong at every ordinary segment count.
///
/// The stock staircase preset is built at `points = 17` (`E = 16`) whenever the widget has
/// never recorded a segment count, so `segments < 16` — every ordinary count — takes this
/// branch. The `segments = 4` chain is the guide's.
#[test]
fn custom_profile_all_tie_staircase_cut_is_the_tail() {
    let staircase = staircase_net(17);
    let (turns, ranked) = turn_ranking(&staircase);
    assert!(
        turns.iter().all(|&t| t == 0.0),
        "every axis-aligned tread and riser ties at exactly 0: {turns:?}",
    );
    assert_eq!(ranked, (0..16).rev().collect::<Vec<_>>(), "pure rule (a)");

    assert_widget_chain(
        &staircase.widget_chain(4),
        &[
            (0.250, 0.750),
            (0.125, 0.750),
            (0.125, 0.875),
            (0.000, 0.875),
            (0.000, 1.000),
        ],
        "E = 16 staircase segments = 4 (the tail, start corner absent)",
    );
    // At `segments >= E` only the remainder is ranked, still from the high index down.
    assert_eq!(
        allotments_of(&staircase, 18),
        {
            let mut want = vec![1usize; 16];
            want[15] = 2;
            want[14] = 2;
            want
        },
        "the two leftover advances go to the last two segments",
    );

    // Larger hand-built all-tie staircases take the same tail.
    for points in [9usize, 33, 65] {
        let net = staircase_net(points);
        let segment_count = points - 1;
        let n = segment_count / 2;
        let chain = net.widget_chain(n as u32);
        let want_first = (
            net.points[segment_count - n].x,
            net.points[segment_count - n].y,
        );
        assert!(
            (chain[0].0 - want_first.0).abs() < 1.0e-6
                && (chain[0].1 - want_first.1).abs() < 1.0e-6,
            "E = {segment_count} staircase at n = {n} must start at the tail's first \
             control point {want_first:?}, got {:?}",
            chain[0],
        );
    }
}

/// BEVEL.md §8.6.3.1(4).1's other worked ranking, the all-vector `E = 5` net of §(4).4:
/// `[3, 2, 1, 0, 4]`. It is the sharpest guard on the **reciprocal-multiply**
/// normalisation — swapping only that spelling for a componentwise divide re-ranks it to
/// `[3, 1, 2, 0, 4]`, which moves the `n = 2` sub-chain from
/// `(0.60,0.18) · (0.35,0.30) · (0.12,0.60)` to `(0.85,0.10) · (0.35,0.30) · (0.12,0.60)`.
#[test]
fn custom_profile_reciprocal_normalisation_ranks_the_stride_net() {
    let net = all_vector_net(&[
        (1.0, 0.0),
        (0.85, 0.10),
        (0.60, 0.18),
        (0.35, 0.30),
        (0.12, 0.60),
        (0.0, 1.0),
    ]);
    let (turns, ranked) = turn_ranking(&net);
    let multiples: Vec<f32> = turns.iter().map(|t| t / TURN_RESIDUE_SCALE).collect();
    assert_eq!(
        multiples,
        vec![1.0, 1.0, 1.0, std::f32::consts::SQRT_2, 0.0],
        "turns = {turns:?}",
    );
    assert_eq!(ranked, vec![3, 2, 1, 0, 4]);

    assert_widget_chain(
        &net.widget_chain(1),
        &[(0.35, 0.30), (0.12, 0.60)],
        "§(4).4 net segments = 1 (select {3})",
    );
    assert_widget_chain(
        &net.widget_chain(2),
        &[(0.60, 0.18), (0.35, 0.30), (0.12, 0.60)],
        "§(4).4 net segments = 2 (select {2, 3})",
    );
}

/// BEVEL.md §8.6.3.1(4).1 rule (a) on a **genuine, non-residue** tie reachable from a stock
/// preset. The supports preset's two arc segments flanking `P₁` are the same angle by
/// construction and the single-precision evaluation agrees too — the same `f32` bit
/// pattern, not two floats a residue apart — so (b) contributes nothing and (a) decides
/// alone. It picks the *later* segment `1`: at `segments = 2` the cut is `{2, 1}` (the
/// zero-length segment's `60°` first, then the tie-break's winner) and the chain starts at
/// the arc sample, not at the widget corner `(1, 0)` an ascending-index tie-break gives.
#[test]
fn custom_profile_supports_preset_two_segment_cut_is_the_pure_tie_break() {
    let supports = supports_preset_five_points();
    let (turns, ranked) = turn_ranking(&supports);
    assert_eq!(
        turns[0].to_bits(),
        turns[1].to_bits(),
        "the two arc segments must tie bit-for-bit, not merely print alike: {turns:?}",
    );
    assert_eq!(ranked, vec![2, 1, 0, 3]);

    assert_widget_chain(
        &supports.widget_chain(2),
        &[(0.85355, 0.85355), (0.5, 1.0), (0.5, 1.0)],
        "supports segments = 2 (the pure tie-break)",
    );
    assert!(
        (supports.widget_chain(2)[0].0 - 1.0).abs() > 0.1,
        "an ascending-index tie-break would start the chain at the widget corner (1, 0)",
    );
}

/// BEVEL.md §8.6.3.1(4).3's `n = 3` row on the **split** net, the one the guide's own
/// cross-backend arm asserts reference-side only. Segments `1` and `2` are vector–vector
/// chords that here score exactly `0` both — rule (b)'s residue vanishes on this net — so
/// rule (a) decides and the *later* segment `2` is taken. `P₂` reappears in the chain and
/// `P₁` does not, the opposite of an ascending-index tie-break.
#[test]
fn custom_profile_split_net_three_segment_cut_takes_the_later_tied_segment() {
    let split = split_net();
    let (turns, _) = turn_ranking(&split);
    assert_eq!((turns[1], turns[2]), (0.0, 0.0), "a genuine tie: {turns:?}");
    assert_widget_chain(
        &split.widget_chain(3),
        &[(1.00, 0.00), (0.60, 0.55), (0.10, 0.60), (0.00, 1.00)],
        "split net segments = 3 (select {0, 2, 3})",
    );
}

/// The guide's `E = 3` **all-vector** net `(1,0) → (0.8,0.12) → (0.35,0.4) → (0,1)`, on
/// which every segment is a straight chord and the entire chain is decided by
/// §8.6.3.1(4).1's residue ranking.
fn all_vector_e3_net() -> BevelCurveProfile {
    all_vector_net(&[(1.0, 0.0), (0.8, 0.12), (0.35, 0.4), (0.0, 1.0)])
}

/// The staircase preset's control path at a given odd `points` (BEVEL.md §8.6.3.1(3)):
/// `x_i = 1 − 2·⌊(i+1)/2⌋ / X`, `y_i = 2·⌊i/2⌋ / Y` with `X = Y = points − 1`, vector
/// handles throughout. Every tread and riser is axis-aligned.
fn staircase_net(points: usize) -> BevelCurveProfile {
    assert!(points % 2 == 1, "the odd-`points` layout only");
    let span = (points - 1) as f32;
    all_vector_net(
        &(0..points)
            .map(|i| {
                (
                    1.0 - 2.0 * ((i + 1) / 2) as f32 / span,
                    2.0 * (i / 2) as f32 / span,
                )
            })
            .collect::<Vec<_>>(),
    )
}

/// The **supports** preset at its `points = 5` layout (BEVEL.md §8.6.3.1(3)): the start
/// corner, one arc sample with auto handles, the arc's `θ = π/2` sample coinciding with the
/// vector-handled end shoulder — hence a duplicated point and a zero-length control segment
/// — and the end corner.
fn supports_preset_five_points() -> BevelCurveProfile {
    let arc = 0.5 + 0.5 * std::f32::consts::FRAC_1_SQRT_2;
    BevelCurveProfile::from_points_with_handles(&[
        (1.0, 0.0, BevelHandle::Vector, BevelHandle::Vector),
        (
            1.0 - 0.5 * (1.0 - std::f32::consts::FRAC_1_SQRT_2),
            arc,
            BevelHandle::Auto,
            BevelHandle::Auto,
        ),
        (0.5, 1.0, BevelHandle::Auto, BevelHandle::Auto),
        (0.5, 1.0, BevelHandle::Vector, BevelHandle::Vector),
        (0.0, 1.0, BevelHandle::Vector, BevelHandle::Vector),
    ])
}

/// The guide's all-auto **LATE_TURN** net (`E = 3`), the driver of §8.6.3.1(4).4's stride
/// worked example.
fn late_turn_net() -> BevelCurveProfile {
    BevelCurveProfile::from_points(&[
        (1.0, 0.0, BevelHandle::Auto),
        (0.85, 0.10, BevelHandle::Auto),
        (0.20, 0.35, BevelHandle::Auto),
        (0.0, 1.0, BevelHandle::Auto),
    ])
}

/// The guide's **split** net (`E = 4`): curvature on the two *outer* control segments only,
/// so its `segments = 2` chain is a sub-budget **jump** past `P₁` and `P₂`.
fn split_net() -> BevelCurveProfile {
    BevelCurveProfile::from_points_with_handles(&[
        (1.0, 0.0, BevelHandle::Vector, BevelHandle::Vector),
        (0.90, 0.40, BevelHandle::Auto, BevelHandle::Vector),
        (0.60, 0.55, BevelHandle::Vector, BevelHandle::Vector),
        (0.10, 0.60, BevelHandle::Vector, BevelHandle::Auto),
        (0.0, 1.0, BevelHandle::Vector, BevelHandle::Vector),
    ])
}

/// The **split** net with `P₂` moved from `(0.60, 0.55)` to `(0.30, 0.45)` and nothing else
/// changed — §8.6.3.1(4).4's second, model-free arm. `P₂` is the one control point the
/// `segments = 2` chain *jumps*, so both nets' `target` chains stay pointwise identical
/// (segments `1` and `2` are straight chords in both, scoring `turn = 0`, and the top-2 cut
/// is `{0, 3}` either way with the tie strictly inside the discarded block), while
/// `doubled[doubled_resolution / 2] = P₂` moves `0.30` in widget `x`.
fn split_net_moved_p2() -> BevelCurveProfile {
    BevelCurveProfile::from_points_with_handles(&[
        (1.0, 0.0, BevelHandle::Vector, BevelHandle::Vector),
        (0.90, 0.40, BevelHandle::Auto, BevelHandle::Vector),
        (0.30, 0.45, BevelHandle::Vector, BevelHandle::Vector),
        (0.10, 0.60, BevelHandle::Vector, BevelHandle::Auto),
        (0.0, 1.0, BevelHandle::Vector, BevelHandle::Vector),
    ])
}

/// A profile cell that embeds the widget frame isometrically, so a chain's widget
/// coordinates read straight off the emitted positions: `a_axis = mid − start` and
/// `b_axis = mid − end` are the two unit axes, the cell origin `start + end − mid` is the
/// world origin, and there is no projection. A widget sample `(x, y)` reaching the cell as
/// `(u, v) = (x, y)` therefore lands at `(y, x, 0)`. Being a similarity also makes 3-D
/// chord lengths equal widget-frame ones, which is what lets the §8.2.2.1.1 read-back's
/// guide-quoted widget numbers be asserted as positions.
fn widget_frame_cell() -> ProfileCell {
    ProfileCell::from_control_points(Vec3::X, Vec3::Y, Vec3::new(1.0, 1.0, 0.0))
}

/// BEVEL.md §8.6.3.1(4).4: a profile read at any resolution other than the operation's own
/// `segments` is an **index stride of the `doubled` chain**, never a fresh run of the
/// evaluation model at that resolution. The two are different tables because a custom
/// chain's per-`n` budget partition is recomputed from scratch and so is **not**
/// refinement-consistent — unlike the analytic superellipse's even-arc-length spacing,
/// where every other sample of the `2m` chain *is* the `m` chain and the distinction is
/// invisible.
///
/// Driven at a **non-power-of-two** `segments = 5` (`doubled_resolution = 8`), where the
/// cascade's middle round asks for `m = 4` and the coarse control net asks for `m = 2`.
/// Both rows are quoted from the guide's worked table; the `m = 4` pair differs by `0.197`
/// in widget `x` at `k = 2`, which is not a rounding difference.
#[test]
fn custom_profile_intermediate_read_strides_the_doubled_chain() {
    let late_turn = late_turn_net();
    let chains = CustomChains::new(&late_turn, 5);
    assert_eq!(chains.doubled_resolution, 8, "max(pow2 >= 5, 4)");

    assert_widget_chain(
        &chains.widget_chain_at(4),
        &[
            (1.0, 0.0),
            (0.85, 0.10),
            (0.397_32, 0.224_63),
            (0.084_97, 0.548_38),
            (0.0, 1.0),
        ],
        "the m = 4 cascade round reads doubled strided by 2",
    );
    // The chain the same curve returns when asked for `n = 4` directly — what a third
    // resample would have produced, and what the round must *not* read.
    assert_widget_chain(
        &late_turn.widget_chain(4),
        &[
            (1.0, 0.0),
            (0.85, 0.10),
            (0.200_00, 0.350_00),
            (0.061_34, 0.652_62),
            (0.0, 1.0),
        ],
        "the fresh n = 4 resample",
    );
    assert!(
        (chains.widget_chain_at(4)[2].0 - late_turn.widget_chain(4)[2].0).abs() > 0.19,
        "the strided and freshly-resampled m = 4 chains are 0.197 apart in widget x",
    );

    // The coarse control net's read (`m = 2`) strides by 4 at `segments >= 3`, landing on
    // `doubled[4]`. A fresh `n = 2` resample would instead return this net's sub-budget
    // chain, whose interior sample is `(0.20, 0.35)`.
    assert_widget_chain(
        &chains.widget_chain_at(2),
        &[(1.0, 0.0), (0.397_32, 0.224_63), (0.0, 1.0)],
        "the coarse net reads doubled strided by 4",
    );
}

/// BEVEL.md §8.6.3.1(4).4's **branch**, and the one segment count at which it is
/// observable. The stride rule's `m == segments` arm *wins*, so the [V-ADJ] coarse control
/// net — a profile read at `m = 2` (§8.5.5.5(a)) — takes `target[1]` at `segments == 2` and
/// the strided `doubled[doubled_resolution / 2]` at every `segments >= 3`. On the split net
/// those two points are `0.50` apart in widget `x`, so this is not a coincidence to be
/// collapsed: coding the strided arm unconditionally is wrong on **every** two-segment
/// custom bevel (≈1753 canonical position quanta of cap apex) and right everywhere else.
///
/// Asserted twice over: on the served chain, and on the `RingChainProfile::raw_positions(2)`
/// call the fill actually makes.
#[test]
fn custom_profile_coarse_net_mid_takes_the_target_chain_at_two_segments() {
    let split = split_net();

    let at_two = CustomChains::new(&split, 2);
    assert_eq!(at_two.doubled_resolution, 4, "the floor of 4, not segments");
    assert_widget_chain(
        &at_two.widget_chain_at(2),
        &[(1.0, 0.0), (0.10, 0.60), (0.0, 1.0)],
        "segments = 2: the m == segments arm serves the target chain",
    );
    // The rival the branch must decline, and the gap it would open.
    assert_widget_chain(
        &at_two.doubled,
        &[
            (1.0, 0.0),
            (0.90, 0.40),
            (0.60, 0.55),
            (0.10, 0.60),
            (0.0, 1.0),
        ],
        "doubled is this net's own five control points at n = 4 = E",
    );
    assert!(
        (at_two.doubled[2].0 - at_two.widget_chain_at(2)[1].0).abs() > 0.49,
        "the strided mid would sit 0.50 away in widget x: {:?} vs {:?}",
        at_two.doubled[2],
        at_two.widget_chain_at(2)[1],
    );

    // `segments = 3` shares this net's `doubled_resolution = 4`, so the *same* `m = 2` read
    // takes the other arm and lands on `doubled[2]`.
    let at_three = CustomChains::new(&split, 3);
    assert_eq!(at_three.doubled_resolution, 4);
    assert_widget_chain(
        &at_three.widget_chain_at(2),
        &[(1.0, 0.0), (0.60, 0.55), (0.0, 1.0)],
        "segments = 3: the m = 2 read strides",
    );

    // The same branch through the fill's own call. The widget-frame cell sends a chain
    // sample `(x, y)` to `(y, x, 0)`, and the §8.6.3.1(6) mirror leaves a chain's middle
    // sample where it is, so the two mids are `(0.60, 0.10, 0)` and `(0.55, 0.60, 0)`.
    let cell = widget_frame_cell();
    let mid_of = |chains: &CustomChains| {
        RingChainProfile::Cell {
            cell,
            profile: 0.5,
            custom: Some(CustomChainRead::vertex_cap(chains)),
        }
        .raw_positions(2)[1]
    };
    let expect_mid = |got: Vec3, want: Vec3, what: &str| {
        assert!(
            (got - want).length() < 1.0e-4,
            "{what}: coarse mid {got:?} != {want:?}",
        );
    };
    expect_mid(
        mid_of(&at_two),
        Vec3::new(0.60, 0.10, 0.0),
        "segments = 2 coarse mid is target[1]",
    );
    expect_mid(
        mid_of(&at_three),
        Vec3::new(0.55, 0.60, 0.0),
        "segments = 3 coarse mid is doubled[2]",
    );
}

/// BEVEL.md §8.6.3.1(4).4 row 6 / §8.2.2.1.1's **Scope**: every **non-[V-ADJ]** sector
/// emits `target[k]` **raw** at every `segments` — the poly-filled / tri-fanned terminal
/// (§8.3.0.2), the two-sector weld (§8.3.0.3), the cutoff fill (§8.3), the single-segment
/// strip (§7.6) and the terminal far-end cross-section (§8.2.2.1). Those are exactly the
/// sites that request [`ChainEmission::Raw`]; the chord read-back belongs to a [V-ADJ]
/// cap's boundary emission and to nothing else.
///
/// A rim that *equals* the raw chain proves nothing about whether the read-back ran, so the
/// driver is chosen where it moves loudly: the split net's `segments = 2` chain is a
/// sub-budget jump past two control points that `doubled` traverses, so the two chord-fraction
/// distributions disagree grossly rather than subtly. Its single interior sample sits at
/// widget `(0.10, 0.60)` raw and reads back at `(0.14628, 0.59536)` — `4.65e-2` apart.
#[test]
fn custom_profile_non_adjacency_sector_emits_the_raw_target_chain() {
    let split = split_net();
    let chains = CustomChains::new(&split, 2);
    let cell = widget_frame_cell();

    let read = CustomChainRead::vertex_cap(&chains);
    let raw = cell.chain_positions(2, 0.5, Some(read), ChainEmission::Raw);
    let emitted = cell.chain_positions(2, 0.5, Some(read), ChainEmission::Resampled);
    assert_eq!(raw.len(), 3);
    assert_eq!(emitted.len(), 3);
    for (index, (r, e)) in raw.iter().zip(emitted.iter()).enumerate() {
        if index == 0 || index == 2 {
            assert!(
                (*r - *e).length() < 1.0e-6,
                "the two chain ends are the boundverts under either emission",
            );
        }
    }
    assert!(
        (raw[1] - Vec3::new(0.60, 0.10, 0.0)).length() < 1.0e-4,
        "the raw arm is target[1] verbatim, got {:?}",
        raw[1],
    );
    assert!(
        (emitted[1] - Vec3::new(0.595_36, 0.146_28, 0.0)).length() < 1.0e-4,
        "the [V-ADJ] arm reads target's chord fraction off doubled, got {:?}",
        emitted[1],
    );
    assert!(
        (raw[1] - emitted[1]).length() > 4.5e-2,
        "the driver must actually move the sample, else the arms are indistinguishable",
    );
}

/// BEVEL.md §8.6.3.1(4).4: the emitted [V-ADJ] rim is **not a function of `target` alone**.
/// Two curves that produce the *same* chain at `segments` and different `doubled` chains
/// emit the same raw sector and a **different** cap rim — which is what pins the read-back's
/// source polyline as the `doubled` chain rather than a re-read of `target` (a read-back
/// against `target` itself would be the identity, and both arms would agree).
///
/// The guide's six-point net at `segments = 5 = E`: every control segment gets exactly one
/// advance, so the handles are irrelevant to `target` and the all-vector and all-auto nets
/// share it pointwise. Their `doubled` chains (`doubled_resolution = 8`) differ. The
/// all-vector net's rim is additionally the read-back's *identity* — every sample lies on
/// the straight control polyline, and chord length along a straight run is additive — which
/// is why every earlier all-vector fixture reported nothing. Its `doubled` **interior** is
/// not derivable (an all-vector net scores `turn = 0` on every segment, routing the whole
/// remainder through the undefined tie order), so nothing here asserts it.
#[test]
fn custom_profile_cap_rim_reads_the_doubled_chain_not_only_the_target() {
    let net = [
        (1.0, 0.0),
        (0.85, 0.10),
        (0.60, 0.18),
        (0.35, 0.30),
        (0.12, 0.60),
        (0.0, 1.0),
    ];
    let all_vector = BevelCurveProfile::from_points(
        &net.iter()
            .map(|&(x, y)| (x, y, BevelHandle::Vector))
            .collect::<Vec<_>>(),
    );
    let all_auto = BevelCurveProfile::from_points(
        &net.iter()
            .map(|&(x, y)| (x, y, BevelHandle::Auto))
            .collect::<Vec<_>>(),
    );
    let vector_chains = CustomChains::new(&all_vector, 5);
    let auto_chains = CustomChains::new(&all_auto, 5);
    assert_widget_chain(&vector_chains.target, &net, "all-vector target, n = 5 = E");
    assert_widget_chain(&auto_chains.target, &net, "all-auto target, n = 5 = E");
    assert_eq!(vector_chains.doubled_resolution, 8);

    let cell = widget_frame_cell();
    let emit = |chains: &CustomChains, emission| {
        cell.chain_positions(5, 0.5, Some(CustomChainRead::vertex_cap(chains)), emission)
    };
    let raw_of = |chains: &CustomChains| emit(chains, ChainEmission::Raw);
    let rim_of = |chains: &CustomChains| emit(chains, ChainEmission::Resampled);

    // Row 6: a non-[V-ADJ] sector reads `target` only, so the two nets are indistinguishable
    // there even though their `doubled` chains differ.
    for (index, (a, b)) in raw_of(&vector_chains)
        .iter()
        .zip(raw_of(&auto_chains).iter())
        .enumerate()
    {
        assert!(
            (*a - *b).length() < 1.0e-6,
            "raw sample {index} must not see the doubled chain: {a:?} vs {b:?}",
        );
    }

    // The piecewise-linear net's rim is the read-back's identity; the curved one's is not.
    let vector_raw = raw_of(&vector_chains);
    let vector_rim = rim_of(&vector_chains);
    for (index, (r, e)) in vector_raw.iter().zip(vector_rim.iter()).enumerate() {
        assert!(
            (*r - *e).length() < 1.0e-5,
            "an all-vector net resamples to itself at sample {index}: {r:?} vs {e:?}",
        );
    }
    let auto_rim = rim_of(&auto_chains);
    let moved = vector_rim
        .iter()
        .zip(auto_rim.iter())
        .map(|(a, b)| (*a - *b).length())
        .fold(0.0_f32, f32::max);
    assert!(
        moved > 1.0e-4,
        "the two caps share a target chain and must still emit different rims, \
         widest separation {moved}",
    );
}

/// BEVEL.md §8.6.3.1(6) the handedness rule: for an asymmetric control net the `reverse`
/// bit mirrors every interior sample across the cell diagonal `u = v` and reverses index
/// order, so the classic `0.375` ↔ `0.625` flip appears. The two endpoints stay pinned.
#[test]
fn custom_profile_reverse_bit_mirrors_the_asymmetric_chain() {
    let poly = BevelCurveProfile::from_points(&[
        (1.0, 0.0, BevelHandle::Vector),
        (0.5, 0.25, BevelHandle::Vector),
        (0.0, 1.0, BevelHandle::Vector),
    ]);

    // segments = 2: the single interior sample flips (0.25, 0.5) ↔ (0.5, 0.25).
    let forward = poly.sample_chain(2, false);
    let mirrored = poly.sample_chain(2, true);
    assert!((forward[1].0 - 0.25).abs() < 1.0e-6 && (forward[1].1 - 0.5).abs() < 1.0e-6);
    assert!((mirrored[1].0 - 0.5).abs() < 1.0e-6 && (mirrored[1].1 - 0.25).abs() < 1.0e-6);
    // Endpoints are pinned to the ring corners in both orientations.
    for k in [0usize, 2] {
        assert!((forward[k].0 - mirrored[k].0).abs() < 1.0e-6);
        assert!((forward[k].1 - mirrored[k].1).abs() < 1.0e-6);
    }

    // segments = 4: the mirror is the forward chain swapped-and-reversed.
    let forward4 = poly.sample_chain(4, false);
    let mirrored4 = poly.sample_chain(4, true);
    let want = [
        (0.0, 1.0),
        (0.25, 0.625),
        (0.5, 0.25),
        (0.75, 0.125),
        (1.0, 0.0),
    ];
    for (got, w) in mirrored4.iter().zip(want.iter()) {
        assert!(
            (got.0 - w.0).abs() < 1.0e-6 && (got.1 - w.1).abs() < 1.0e-6,
            "mirrored segments = 4 sample {got:?} != {w:?}",
        );
    }
    // The mirror really is index-reversed + diagonal-reflected forward.
    let n = forward4.len() - 1;
    for k in 0..=n {
        let (fx, fy) = forward4[n - k];
        assert!((mirrored4[k].0 - fy).abs() < 1.0e-6 && (mirrored4[k].1 - fx).abs() < 1.0e-6);
    }
}

/// BEVEL.md §8.6.3.1(6) Regime B (vertex bevels): a [V-ADJ] cap runs no orientation
/// walk, so every cap sector shares one uniform handedness. On the worked valence-3
/// cube corner `(1,-1,-1)` (offset 0.25, segments 2) the three cap sectors' single rim
/// mids must form one orbit of the corner's three-fold rotational symmetry — the
/// retracted per-sector world-`z` compare mirrored two of the three.
///
/// This test drives the **real** vertex-bevel pipeline (not a synthetic `ProfileCell`
/// with hand-picked `start`/`end`), because the real pipeline feeds each sector's
/// `start`/`end` from loopmesh's cap ring in its own rotational sense — the opposite of
/// the sense §8.6.3.1(6)'s table assumes. That winding is exactly what an isolated
/// synthetic cell with guide-ordered corners would mask, letting a unit test "pass"
/// while the real bevel diverged. Faithful to the real path, the three rim mids must
/// land the guide's validated table (chord fraction `0.375`, not the mirrored `0.625`)
/// and the welded apex must stay three-fold symmetric.
#[test]
fn custom_profile_vertex_cap_orientation_forms_symmetry_orbit() {
    let curve = BevelCurveProfile::from_points(&[
        (1.0, 0.0, BevelHandle::Vector),
        (0.5, 0.25, BevelHandle::Vector),
        (0.0, 1.0, BevelHandle::Vector),
    ]);
    let corner = Vec3::new(1.0, -1.0, -1.0);

    let mut mesh = LoopMesh::new();
    mesh_add_cube(&mut mesh);
    let corner_vert = vert_at(&mesh, corner);
    mesh.bevel(
        &[corner_vert.into()],
        BevelOptions {
            offset: 0.25,
            segments: 2,
            affect: BevelAffect::Vertices,
            profile_type: BevelProfileType::Custom(&curve),
            ..Default::default()
        },
    );
    mesh.validate_topology();

    // The three sector rim mids from §8.6.3.1(6)'s validated table — the values the
    // BMesh oracle produces and that the real pipeline must reproduce once loopmesh's
    // opposite cap-ring winding is compensated by the complementary handedness.
    let expected = [
        Vec3::new(0.8125, -1.0, -0.875), // (4,8)  across y = −1
        Vec3::new(1.0, -0.875, -0.8125), // (8,11) across x = 1
        Vec3::new(0.875, -0.8125, -1.0), // (11,4) across z = −1
    ];
    let mut rim_mids = Vec::new();
    for want in expected {
        let found = vert_near(&mesh, want, 1.0e-4).unwrap_or_else(|| {
            panic!(
                "real vertex-bevel pipeline must mint a rim mid at the guide-table \
                 position {want:?} (chord fraction 0.375); the mirrored 0.625 chord \
                 fraction means the cap landed on the complementary handedness",
            )
        });
        rim_mids.push(mesh.vertex_position(found));
    }

    // The three rim mids are a single orbit of the corner's three-fold rotation
    // (cycling edges −x → +y → +z). About `corner`, that rotation acts on a
    // displacement `d = p − corner` as `R·d = (−d.z, −d.x, d.y)`, and it must send
    // (4,8) → (11,4) → (8,11) → (4,8). Uniform handedness is exactly what makes the
    // orbit close; a per-sector orientation bit mirrors at least one sector onto the
    // flipped chord fraction and breaks it.
    let rotate = |p: Vec3| {
        let d = p - corner;
        corner + Vec3::new(-d.z, -d.x, d.y)
    };
    let orbit = [
        (rim_mids[0], rim_mids[2]), // (4,8)  → (11,4)
        (rim_mids[2], rim_mids[1]), // (11,4) → (8,11)
        (rim_mids[1], rim_mids[0]), // (8,11) → (4,8)
    ];
    for (from, to) in orbit {
        assert!(
            (rotate(from) - to).length() < 1.0e-5,
            "three-fold rotation must map {from:?} to {to:?}, got {:?}",
            rotate(from),
        );
    }

    // The welded cap apex is the unique minted vertex strictly off every cube face:
    // the boundverts and all three rim mids each keep at least one coordinate on a
    // `±1` face, only the apex leaves all three. Uniform handedness keeps it on the
    // corner's three-fold axis, i.e. its three coordinate magnitudes are equal; a
    // per-sector flip pulls it off-axis.
    let on_cube_face = |c: f32| (c.abs() - 1.0).abs() < 1.0e-4;
    let apex = mesh
        .vert_keys()
        .into_iter()
        .map(|v| mesh.vertex_position(v))
        .find(|p| !on_cube_face(p.x) && !on_cube_face(p.y) && !on_cube_face(p.z))
        .expect("the welded cap apex, the lone vertex off every cube face");
    assert!(
        (apex.x - (-apex.y)).abs() < 1.0e-5 && (apex.x - (-apex.z)).abs() < 1.0e-5,
        "the welded apex must sit on the corner's three-fold symmetry axis \
         (|x| = |y| = |z|), got {apex:?}",
    );
}

/// BEVEL.md §8.6.3.1(4).4 **row 6** and §8.6.3.1(6) **A.5**: the §8.3.0.3 two-boundvert
/// weld is a custom-chain consumer like any other ring sector. Row 6 lists "the §8.3.0.3
/// two-sector weld" among the families that read `target[k]`, **raw**, at every
/// `segments`; A.5 — *"the weld is not special"* — says a beveled-valence-2 junction is
/// the orientation walk's own unconditional cross-vertex step, so each welded half reads
/// the curve through **its own** edge-end's mark, with no weld-specific reversal.
///
/// **The fixture and the closed form the expectations come from.** The cube corner
/// `v = (1,-1,-1)` with its two `z = -1` edges beveled at `offset = 0.25` — the corner
/// the analytic weld pins already use. Its two ring corners are `P = (0.75,-0.75,-1)`
/// (the in-face meet) and `Q = (1,-1,-0.75)` (the meet on the unbeveled leg), and the
/// §8.3.0.3 weld-plane override re-planes both halves onto `plane(v, P, Q)`, which is
/// `x + y = 0`. Composing §8.2.2.1's cell map with the projection into that plane, the
/// half anchored at `P` reduces to the affine
///
/// ```text
///     (u, v)  ->  ( 0.75 + 0.25·u,  −(0.75 + 0.25·u),  −0.75 − 0.25·v )
/// ```
///
/// and the half anchored at `Q` to the same map with `u` and `v` **transposed** —
/// independently of which of the two beveled edges each half's chain crosses, since the
/// projection into the weld plane is what removes that difference. The map is checked
/// against an already-oracle-pinned value rather than merely asserted: the circular
/// chain's single interior cell sample `(√2/2, √2/2)` goes to
/// `(0.926777, −0.926777, −0.926777)`, which is what
/// `edge_bevel_two_edge_junction_weld_vert_sits_on_the_weld_plane` pins.
///
/// **Why the transposition is the whole point.** Because the two halves' maps are
/// transposes, and because §8.6.3.1(5)/(6)'s two readings of one chain
/// (`(u_t, v_t) = (y_t, x_t)` forward, `(x_(n−t), y_(n−t))` mirrored) are likewise
/// transposes of one another read from opposite ends, the two halves' samples
/// **coincide exactly** when their marks differ — which is what A.3's parity across the
/// cross-vertex step guarantees. Equal marks would leave two genuinely different points
/// and the §8.3.0.3 merge would average them into a third that lies on neither
/// cross-section; at `segments = 2` that average is `(0.84375, −0.84375, −0.84375)`,
/// asserted absent below.
///
/// **The expectations.** The all-vector net `(1,0) → (0.5,0.25) → (0,1)` has `E = 2`, so
/// §8.6.3.1(4).1's even split gives `target` = the two control chords sampled at even
/// parameter: `[(1,0), (0.5,0.25), (0,1)]` at `n = 2` and
/// `[(1,0), (0.75,0.125), (0.5,0.25), (0.25,0.625), (0,1)]` at `n = 4`. Mapping the
/// interior samples through the closed form above under each of the two readings gives
/// the two admissible position sets below — and *only* those two: which one lands is the
/// composition of the run's A.2 seed bit with which sector the ring indexes first, and
/// both compositions are covered because they exchange the two sets.
#[test]
fn custom_profile_two_boundvert_weld_reads_the_control_net() {
    let curve = BevelCurveProfile::from_points(&[
        (1.0, 0.0, BevelHandle::Vector),
        (0.5, 0.25, BevelHandle::Vector),
        (0.0, 1.0, BevelHandle::Vector),
    ]);

    // The two readings' interior weld positions, hard-coded from the closed form in the
    // doc comment above. `forward` reads the chain `(u_t, v_t) = (y_t, x_t)`; `mirrored`
    // reads it `(x_(n−t), y_(n−t))`.
    let cases: [(u32, [&[Vec3]; 2]); 2] = [
        (
            2,
            [
                &[Vec3::new(0.8125, -0.8125, -0.875)],
                &[Vec3::new(0.875, -0.875, -0.8125)],
            ],
        ),
        (
            4,
            [
                &[
                    Vec3::new(0.78125, -0.78125, -0.9375),
                    Vec3::new(0.8125, -0.8125, -0.875),
                    Vec3::new(0.90625, -0.90625, -0.8125),
                ],
                &[
                    Vec3::new(0.8125, -0.8125, -0.90625),
                    Vec3::new(0.875, -0.875, -0.8125),
                    Vec3::new(0.9375, -0.9375, -0.78125),
                ],
            ],
        ),
    ];

    for (segments, readings) in cases {
        let mut mesh = LoopMesh::new();
        mesh_add_cube(&mut mesh);
        let corner = vert_at(&mesh, Vec3::new(1.0, -1.0, -1.0));
        let geom: Vec<GeomKey> = [Vec3::new(-1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, -1.0)]
            .into_iter()
            .map(|far| {
                GeomKey::from(
                    mesh.query_edge_exists(corner, vert_at(&mesh, far))
                        .expect("the cube corner's beveled edge"),
                )
            })
            .collect();
        let result = mesh.bevel(
            &geom,
            BevelOptions {
                offset: 0.25,
                affect: BevelAffect::Edges,
                segments,
                profile_type: BevelProfileType::Custom(&curve),
                ..Default::default()
            },
        );
        mesh.validate_topology();

        // The welded chain's interior verts, isolated by the open box the chain runs
        // through: on the weld plane `x + y = 0`, strictly between the two ring corners
        // in both `x` and `z`. No other minted vert of this bevel is in it — the rest
        // live at the two beveled edges' far terminals.
        let mut welded: Vec<Vec3> = result
            .verts
            .iter()
            .map(|&v| mesh.vertex_position(v))
            .filter(|p| {
                (p.x + p.y).abs() < 1.0e-5
                    && p.x > 0.7501
                    && p.x < 0.9999
                    && p.z > -0.9999
                    && p.z < -0.7501
            })
            .collect();
        welded.sort_by(|a, b| a.x.total_cmp(&b.x));

        let matches = |want: &[Vec3]| {
            welded.len() == want.len()
                && welded
                    .iter()
                    .zip(want.iter())
                    .all(|(got, want)| (*got - *want).length() < 1.0e-6)
        };
        assert!(
            matches(readings[0]) || matches(readings[1]),
            "segments = {segments}: the weld emitted {welded:?}, which is neither \
             §8.6.3.1(5)/(6) reading of the custom `target` chain — {:?} or {:?}",
            readings[0],
            readings[1],
        );

        if segments == 2 {
            assert_eq!(
                result_has_vert_near(
                    &mesh,
                    &result,
                    Vec3::new(0.926_776_7, -0.926_776_7, -0.926_776_7),
                ),
                0,
                "the analytic circular cross-section must not survive a custom bevel",
            );
            assert_eq!(
                result_has_vert_near(&mesh, &result, Vec3::new(0.843_75, -0.843_75, -0.843_75),),
                0,
                "the two halves read through EQUAL marks would leave the merge averaging \
                 two mirrored samples into a point on neither cross-section",
            );
        }
    }
}

/// BEVEL.md §8.5.5.5(f.3) law **F1**: the custom-profile fullness is
/// `(1/(2·n))·Σ_{i=0}^{n-1}(u_i + v_i)` — the `i = n` endpoint `(1,0)` is excluded and
/// the divisor is `2·n`.
#[test]
fn custom_profile_fullness_f1_excludes_the_last_endpoint() {
    let line = BevelCurveProfile::preset_line();
    // chain = [(0,1),(0.5,0.5),(1,0)]; sum over i=0,1 of (u+v) = 1 + 1 = 2; /(2·2) = 0.5.
    assert!((CustomChains::new(&line, 2).fullness - 0.5).abs() < 1.0e-6);

    let poly = BevelCurveProfile::from_points(&[
        (1.0, 0.0, BevelHandle::Vector),
        (0.5, 0.25, BevelHandle::Vector),
        (0.0, 1.0, BevelHandle::Vector),
    ]);
    // chain = [(0,1),(0.25,0.5),(1,0)]; sum over i=0,1 = 1 + 0.75 = 1.75; /(4) = 0.4375.
    assert!((CustomChains::new(&poly, 2).fullness - 0.4375).abs() < 1.0e-6);
}

/// BEVEL.md §8.5.5.5(f.3) F1 on the **`n < E` sub-budget branch** (§8.6.3.1(4).3), the
/// only place the un-pinned `chain[0]` is observable: F1 sums `i = 0 ..= n-1` and so
/// reads `chain[0]` directly, while the ring fill that overwrites it never runs here.
///
/// The split net at `segments = 2` takes the non-adjacent {0, 3} cut, giving the widget
/// chain `(1,0) (0.10,0.60) (0,1)` — cell `(0,1) (0.60,0.10) (1,0)`. The sum over
/// `i = 0, 1` is `1.00 + 0.70 = 1.70`, so F1 = `1.70 / 4 = 0.425`. Forcing `chain[0]` to
/// the `(0, 1)` corner, as a pinned-endpoint chain would, leaves `1 + 1 = 2` and `0.5`.
#[test]
fn custom_profile_fullness_f1_reads_the_unpinned_sub_budget_chain_start() {
    let split = split_net();
    let chains = CustomChains::new(&split, 2);
    assert!(
        (chains.fullness - 0.425).abs() < 1.0e-6,
        "F1 on the sub-budget branch: got {}",
        chains.fullness,
    );
}

/// The `±1` box built face-by-face in BEVEL.md §8.6.3.1(6) A.2.1(vi)'s stated order —
/// `z = −1`, `x = −1`, `z = +1`, `x = +1`, `y = +1`, `y = −1` — so the three spokes of the
/// `(1, −1, −1)` corner take that worked example's **stored first endpoints**: the `−x`
/// spoke is minted out of the corner, and both the `+y` and `+z` spokes are minted *into*
/// it by a face that reaches them first. A.2.1(i) is structural, so a box assembled in any
/// other order is a different fixture with different (equally correct) bits.
fn mesh_add_cube_in_guide_face_order(mesh: &mut LoopMesh) -> Vec<VertexKey> {
    let coords = [
        Vec3::new(1.0, -1.0, -1.0),
        Vec3::new(-1.0, -1.0, -1.0),
        Vec3::new(-1.0, 1.0, -1.0),
        Vec3::new(1.0, 1.0, -1.0),
        Vec3::new(1.0, -1.0, 1.0),
        Vec3::new(-1.0, -1.0, 1.0),
        Vec3::new(-1.0, 1.0, 1.0),
        Vec3::new(1.0, 1.0, 1.0),
    ];
    let verts: Vec<VertexKey> = coords.iter().map(|&c| mesh.create_vertex(c)).collect();
    for ring in [
        [0usize, 1, 2, 3],
        [1, 5, 6, 2],
        [4, 7, 6, 5],
        [4, 0, 3, 7],
        [3, 2, 6, 7],
        [0, 4, 5, 1],
    ] {
        let face: Vec<VertexKey> = ring.iter().map(|&i| verts[i]).collect();
        mesh.face_create_verts(&face, FaceCreateFlags::NONE);
    }
    verts
}

/// BEVEL.md §8.6.3.1(6) A.2.1(vi), worked end to end: at the mutually-orthogonal
/// beveled-valence-3 cube corner A.1's contest admits **no** continuation (every other
/// spoke's dot is exactly `0`, and only strictly-positive dots are candidates), so by A.4
/// the three spokes are three separate runs carrying three *independent* seed
/// determinations — `reverse = true / false / false` around the ring. No single constant
/// can produce that triple, and the constant that used to be fed here made the corner cap
/// three-fold symmetric about the body diagonal, which the reference's is not.
#[test]
fn custom_profile_edge_junction_takes_three_independent_seeds_at_an_orthogonal_corner() {
    let mut mesh = LoopMesh::new();
    let verts = mesh_add_cube_in_guide_face_order(&mut mesh);
    let corner = verts[0];
    let spoke = |far: VertexKey| {
        mesh.query_edge_exists(corner, far)
            .expect("a beveled spoke")
    };
    let along_minus_x = spoke(verts[1]);
    let along_plus_y = spoke(verts[3]);
    let along_plus_z = spoke(verts[4]);
    let beveled = [along_minus_x, along_plus_y, along_plus_z];

    // A.2.1(i): the seed end is the edge's stored first endpoint, and on this fixture only
    // the `−x` spoke stores the corner first. That single fact is what the whole worked
    // example turns on.
    assert_eq!(mesh.edge_vert_keys(along_minus_x)[0], corner);
    assert_eq!(mesh.edge_vert_keys(along_plus_y)[0], verts[3]);
    assert_eq!(mesh.edge_vert_keys(along_plus_z)[0], verts[4]);

    let (orientation, set) = orientation_marks(&mesh, &beveled, 0.25);
    let is_beveled = |edge: EdgeKey| set.contains(&edge);

    // The ring the bits are stated against (A.2.1(vi)): `B₀` in `y = −1`, `B₁` in
    // `z = −1`, `B₂` in `x = 1`, with sector `B_k → B_(k+1)` crossing the spoke named.
    for (edge, start, end) in [
        (
            along_minus_x,
            Vec3::new(0.75, -1.0, -0.75),
            Vec3::new(0.75, -0.75, -1.0),
        ),
        (
            along_plus_y,
            Vec3::new(0.75, -0.75, -1.0),
            Vec3::new(1.0, -0.75, -0.75),
        ),
        (
            along_plus_z,
            Vec3::new(1.0, -0.75, -0.75),
            Vec3::new(0.75, -1.0, -0.75),
        ),
    ] {
        let ends = mesh
            .sector_ends(edge, corner, &is_beveled, 0.25)
            .expect("every beveled spoke of the corner spans a sector there");
        assert!(
            (ends.start - start).length() < 1.0e-6 && (ends.end - end).length() < 1.0e-6,
            "sector across {edge:?}: got {:?} → {:?}, want {start:?} → {end:?}",
            ends.start,
            ends.end,
        );
    }

    assert_eq!(
        [
            orientation.reverse(along_minus_x, corner),
            orientation.reverse(along_plus_y, corner),
            orientation.reverse(along_plus_z, corner),
        ],
        [true, false, false],
        "A.2.1(vi): the `−x` spoke seeds at the corner and reads `−0.75 < −1` (no); the \
         `+y` spoke seeds at its far end, reads no there and flips one hop back; the \
         `+z` spoke is z-parallel, ties at both ends, and reaches the corner flipped",
    );

    // A.1's contest, which is why these are three runs and not one.
    for edge in beveled {
        assert_eq!(
            mesh.orientation_continuation(edge, corner, &is_beveled),
            None,
            "mutually orthogonal spokes give dot 0, which is not a candidate at all",
        );
    }

    // The physical consequence: fed the asymmetric net, the corner cap's welded seed is
    // pulled off the body diagonal. One uniform bit lands it *on* the diagonal.
    let curve = asymmetric_custom_net();
    let geom: Vec<GeomKey> = beveled.iter().map(|&edge| edge.into()).collect();
    let result = mesh.bevel(
        &geom,
        BevelOptions {
            offset: 0.25,
            segments: 2,
            affect: BevelAffect::Edges,
            vmesh_method: BevelVmeshMethod::Adj,
            profile_type: BevelProfileType::Custom(&curve),
            ..Default::default()
        },
    );
    mesh.validate_topology();

    let seed = Vec3::new(0.814_764_4, -0.805_027_4, -0.824_501_3);
    let landed = result
        .verts
        .iter()
        .any(|&v| (mesh.vertex_position(v) - seed).length() < 2.0e-6);
    assert!(
        landed,
        "the cap seed must land at {seed:?}; got {:?}",
        result
            .verts
            .iter()
            .map(|&v| mesh.vertex_position(v))
            .collect::<Vec<_>>(),
    );
    assert!(
        (seed.y - seed.z).abs() > 1.0e-3,
        "and it must be off the corner's body diagonal — a three-fold symmetric cap is \
         the signature of one uniform bit",
    );
}

/// BEVEL.md §8.6.3.1(6) A.2.1(i)/(v): the seed end is **structural**. Re-minting the
/// `z`-parallel spoke out of the corner instead of into it — the only difference between
/// this box and the one above — flips that sector's bit, while the other two spokes, whose
/// seed compares do not tie, are unmoved. Nothing geometric distinguishes the two boxes.
#[test]
fn custom_profile_edge_junction_seed_end_is_the_edges_stored_first_endpoint() {
    let mut mesh = LoopMesh::new();
    mesh_add_cube(&mut mesh);
    let corner = vert_at(&mesh, Vec3::new(1.0, -1.0, -1.0));
    let spoke = |far: Vec3| {
        mesh.query_edge_exists(corner, vert_at(&mesh, far))
            .expect("a beveled spoke")
    };
    let along_minus_x = spoke(Vec3::new(-1.0, -1.0, -1.0));
    let along_plus_y = spoke(Vec3::new(1.0, 1.0, -1.0));
    let along_plus_z = spoke(Vec3::new(1.0, -1.0, 1.0));

    assert_eq!(
        mesh.edge_vert_keys(along_plus_z)[0],
        corner,
        "this builder mints the z-parallel spoke out of the corner",
    );

    let (orientation, _) =
        orientation_marks(&mesh, &[along_minus_x, along_plus_y, along_plus_z], 0.25);
    assert_eq!(
        [
            orientation.reverse(along_minus_x, corner),
            orientation.reverse(along_plus_y, corner),
            orientation.reverse(along_plus_z, corner),
        ],
        [true, false, true],
        "the z-parallel sector's compare ties at both ends, so which end seeds is the \
         whole of its answer; the other two are untouched",
    );
}

/// BEVEL.md §8.6.3 / §8.11: a `profile_type = CUSTOM` vertex bevel of a cube corner
/// (valence 3) at `segments = 2` builds the same general fullness grid a curved
/// superellipse does — the custom sampler only reshapes the boundary ring and the
/// centre fullness (law F1), never the topology. The resulting mesh differs in vertex
/// *positions* from the superellipse-circle bevel, proving the custom sampler is wired.
///
/// This net is all-vector with straight chords, so `target[1]` and
/// `doubled[doubled_resolution / 2]` are the *same point* and the apex below reads the same
/// under either arm of §8.6.3.1(4).4's stride rule. The schedule is discriminated by
/// `bevel_custom_profile_split_net_two_segment_apex_takes_the_target_chain_mid` instead.
#[test]
fn bevel_custom_profile_vertex_cube_corner_two_segments_builds_general_grid() {
    let curve = BevelCurveProfile::from_points(&[
        (1.0, 0.0, BevelHandle::Vector),
        (0.5, 0.25, BevelHandle::Vector),
        (0.0, 1.0, BevelHandle::Vector),
    ]);
    let corner_position = Vec3::new(1.0, -1.0, -1.0);

    let mut custom_mesh = LoopMesh::new();
    mesh_add_cube(&mut custom_mesh);
    let corner = vert_at(&custom_mesh, corner_position);
    let result = custom_mesh.bevel(
        &[corner.into()],
        BevelOptions {
            offset: 0.25,
            segments: 2,
            affect: BevelAffect::Vertices,
            profile_type: BevelProfileType::Custom(&curve),
            ..Default::default()
        },
    );
    custom_mesh.validate_topology();

    let (vertices, edges, faces) = counts(&custom_mesh);
    assert_eq!(
        (vertices, edges, faces),
        (14, 21, 9),
        "custom builds the general grid"
    );
    assert_eq!(
        vertices as i64 - edges as i64 + faces as i64,
        2,
        "custom corner bevel keeps the cube a closed genus-0 solid",
    );
    assert_eq!(
        result.faces.len(),
        3,
        "three cap quads, welded apex, no central polygon"
    );
    assert_eq!(
        face_size_multiset(&custom_mesh),
        vec![4, 4, 4, 4, 4, 4, 6, 6, 6],
        "three cap quads + three untouched cube quads + three grown hexagons",
    );

    // §8.5.5.5(c)/§8.5.5.7: the welded apex is the deterministic once-subdivided Sabin
    // extraordinary centre. Custom fullness F1 = 1.75/4 = 0.4375 seeds the coarse blend
    // `(2·centroid − v) + 2·F·(v − centroid)` to a coarse centre of 0.90625 on each body-
    // diagonal axis; **one** unmasked (custom chains are not sharpened) subdivision round
    // with γ_3 = 0.065247584 pulls it to 0.90251. The corner is symmetric on the body
    // diagonal, so the apex sits at `(0.90251, −0.90251, −0.90251)`.
    let apex = cube_corner_welded_apex(&custom_mesh);
    let expected_apex = Vec3::new(0.902_51, -0.902_51, -0.902_51);
    assert!(
        (apex - expected_apex).length() < 2.0e-5,
        "welded apex at the once-subdivided Sabin extraordinary centre: got {apex:?}, \
         want {expected_apex:?}",
    );

    // The custom sampler reshapes the boundary + fullness, so the cap vertices land at
    // different positions from the superellipse-circle bevel of the same corner.
    let mut circle_mesh = LoopMesh::new();
    mesh_add_cube(&mut circle_mesh);
    let circle_corner = vert_at(&circle_mesh, corner_position);
    circle_mesh.bevel(
        &[circle_corner.into()],
        BevelOptions {
            offset: 0.25,
            segments: 2,
            profile: 0.5,
            affect: BevelAffect::Vertices,
            ..Default::default()
        },
    );
    let custom_positions: Vec<Vec3> = custom_mesh
        .vert_keys()
        .into_iter()
        .map(|v| custom_mesh.vertex_position(v))
        .collect();
    let circle_positions: Vec<Vec3> = circle_mesh
        .vert_keys()
        .into_iter()
        .map(|v| circle_mesh.vertex_position(v))
        .collect();
    let differs = custom_positions.iter().any(|cp| {
        circle_positions
            .iter()
            .all(|sp| (*cp - *sp).length() > 1.0e-4)
    });
    assert!(
        differs,
        "the custom cap lands verts the superellipse-circle cap does not"
    );
}

/// The welded apex of a bevelled cube corner (BEVEL.md §8.5.5.5(e)): the one vertex on the
/// corner's body diagonal (`|x| = |y| = |z|`) strictly inside the cube — off the original
/// corner and off every cap rim vert.
fn cube_corner_welded_apex(mesh: &LoopMesh) -> Vec3 {
    mesh.vert_keys()
        .into_iter()
        .map(|v| mesh.vertex_position(v))
        .find(|p| {
            (p.x.abs() - p.y.abs()).abs() < 1.0e-3
                && (p.x.abs() - p.z.abs()).abs() < 1.0e-3
                && p.x.abs() < 0.99
        })
        .expect("the welded body-diagonal apex")
}

/// Bevels `mesh_add_cube`'s `(1, −1, −1)` corner with `curve` at `segments = 2`,
/// `offset = 0.25`, vertex mode, and returns the mesh.
fn custom_bevel_cube_corner_two_segments(curve: &BevelCurveProfile) -> LoopMesh {
    let mut mesh = LoopMesh::new();
    mesh_add_cube(&mut mesh);
    let corner = vert_at(&mesh, Vec3::new(1.0, -1.0, -1.0));
    mesh.bevel(
        &[corner.into()],
        BevelOptions {
            offset: 0.25,
            segments: 2,
            affect: BevelAffect::Vertices,
            profile_type: BevelProfileType::Custom(curve),
            ..Default::default()
        },
    );
    mesh.validate_topology();
    mesh
}

/// BEVEL.md §8.6.3.1(4).4's `segments = 2` branch, **end to end on a whole mesh**. This is
/// the one segment count at which the [V-ADJ] coarse control net's row of the attribution
/// table changes answers, and the cap apex is where that answer becomes geometry — so the
/// chain-level pin above (which stops at the served mid) is not on its own enough.
///
/// The guide works the case out in full. On the split net (`E = 4`) at `segments = 2`,
/// `doubled_resolution = 4`, the coarse mid is `target[1] = (0.10, 0.60)` and the rival
/// `doubled[doubled_resolution / 2] = (0.60, 0.55)` — `0.50` apart in widget `x`. Feeding
/// each candidate through §8.5.5.5(a) → (e) — coarse net, the custom centre-fullness blend
/// at law **F1** `= (1 + 0 + 0.10 + 0.60) / 4 = 0.425`, one doubling round's face points /
/// innermost edge points / extraordinary-centre stencil at `γ₃ = 0.065247584` with (c)'s
/// three-tap curvature *sharpening* declined (as a custom profile requires), then (d)'s
/// down-resample keeping the subdivided centre verbatim — predicts `0.899683` from
/// `target[1]` and `0.917210` from `doubled[2]`. F1 is identical under either candidate, so
/// the whole gap is attributable to the mid alone. All three sectors are cyclic images of
/// one another about the corner diagonal, so the apex is diagonal: `(a, −a, −a)`.
///
/// **How the discrimination was confirmed.** Deleting `widget_chain_at`'s
/// `requested_resolution == self.segments` arm — the strided-everywhere reading — moves this
/// apex to `0.91721` per coordinate: `1.75e-2` away, ≈1753 canonical position quanta, three
/// orders of magnitude outside the `2e-5` tolerance asserted here. That deletion was made
/// locally and this test failed with exactly that value before it was reverted.
///
/// The second, **model-free** arm rides along and needs none of the stencil: move the one
/// control point the `segments = 2` chain jumps (`P₂`) and the two nets' `target` chains stay
/// pointwise identical while `doubled[2]` moves `0.30`. So they must land the *same* apex —
/// impossible if the coarse mid came from `doubled` — and, because the emitted rim really
/// does read `doubled` (§8.2.2.1.1), *different* meshes overall.
#[test]
fn bevel_custom_profile_split_net_two_segment_apex_takes_the_target_chain_mid() {
    let split = split_net();
    let chains = CustomChains::new(&split, 2);
    assert_eq!(
        chains.doubled_resolution, 4,
        "the floor of 4, not 2·segments"
    );
    assert!(
        (chains.fullness - 0.425).abs() < 1.0e-6,
        "law F1 over target: got {}",
        chains.fullness,
    );

    let mesh = custom_bevel_cube_corner_two_segments(&split);
    let (vertices, edges, faces) = counts(&mesh);
    assert_eq!(
        (vertices, edges, faces),
        (14, 21, 9),
        "the general grid, welded apex, no central polygon",
    );

    let apex = cube_corner_welded_apex(&mesh);
    let from_target_mid = Vec3::new(0.899_683, -0.899_683, -0.899_683);
    let from_doubled_mid = Vec3::new(0.917_210, -0.917_210, -0.917_210);
    assert!(
        (apex - from_target_mid).length() < 2.0e-5,
        "the coarse mid is target[1]: apex {apex:?}, want {from_target_mid:?}",
    );
    assert!(
        (apex - from_doubled_mid).length() > 1.0e-2,
        "and it is emphatically not doubled[2]'s apex {from_doubled_mid:?}",
    );

    // The model-free arm: same target chain, `doubled[2]` moved 0.30 in widget x.
    let moved = split_net_moved_p2();
    let moved_chains = CustomChains::new(&moved, 2);
    assert_widget_chain(
        &moved_chains.target,
        &[(1.00, 0.00), (0.10, 0.60), (0.00, 1.00)],
        "moving the jumped-over P2 leaves the target chain alone",
    );
    assert!(
        (moved_chains.doubled[2].0 - chains.doubled[2].0 + 0.30).abs() < 1.0e-6,
        "doubled[2] moved 0.30 in widget x: {:?} vs {:?}",
        moved_chains.doubled[2],
        chains.doubled[2],
    );

    let moved_mesh = custom_bevel_cube_corner_two_segments(&moved);
    let moved_apex = cube_corner_welded_apex(&moved_mesh);
    assert!(
        (moved_apex - apex).length() < 2.0e-6,
        "two nets sharing a target chain share an apex: {moved_apex:?} vs {apex:?}",
    );
    // …but not a rim: §8.2.2.1.1's read-back does source the doubled chain.
    let widest = mesh
        .vert_keys()
        .into_iter()
        .map(|v| mesh.vertex_position(v))
        .map(|p| {
            moved_mesh
                .vert_keys()
                .into_iter()
                .map(|w| (moved_mesh.vertex_position(w) - p).length())
                .fold(f32::INFINITY, f32::min)
        })
        .fold(0.0_f32, f32::max);
    assert!(
        widest > 1.0e-3,
        "the two rims must differ — the emitted chain reads doubled, gap {widest}",
    );
}

/// A planar valence-4 corner whose beveled spoke's two disk-flanking legs are
/// **non-collinear** (BEVEL.md §8.3.0.2's un-fan precondition): spokes at `0°` (the
/// beveled `E`), `60°`, `180°`, and `300°`, so the flanking legs (`60°` / `300°`) span
/// a genuine `120°` and the terminal-far-end profile-plane triangle is non-degenerate.
/// Returns the mesh and the beveled spoke edge.
fn noncollinear_valence4_fan() -> (LoopMesh, EdgeKey) {
    let mut mesh = LoopMesh::new();
    let centre = mesh.create_vertex(Vec3::ZERO);
    let angles_deg = [0.0_f32, 60.0, 180.0, 300.0];
    let rim: Vec<VertexKey> = angles_deg
        .iter()
        .map(|deg| {
            let a = deg.to_radians();
            mesh.create_vertex(Vec3::new(a.cos(), a.sin(), 0.0))
        })
        .collect();
    for k in 0..rim.len() {
        let next = (k + 1) % rim.len();
        mesh.face_create_verts(&[centre, rim[k], rim[next]], FaceCreateFlags::NONE);
    }
    let beveled = mesh.query_edge_exists(centre, rim[0]).expect("the E spoke");
    (mesh, beveled)
}

/// The terminal fill faces at `centre`'s lone leg vert (BEVEL.md §8.3.0.2): the leg vert
/// is on the `180°` non-adjacent leg, slid `offset` (superellipse) or `offset·√2`
/// (custom) from `centre`. Returns the sorted sizes of the fill faces touching it.
fn leg_fill_sizes(mesh: &LoopMesh, minted: &[VertexKey], leg_x: f32) -> Vec<usize> {
    let leg = vert_near(mesh, Vec3::new(leg_x, 0.0, 0.0), 0.02).expect("the leg vert");
    let mut sizes: Vec<usize> = terminal_fill_faces(mesh, minted, leg)
        .iter()
        .map(|cycle| cycle.len())
        .collect();
    sizes.sort_unstable();
    sizes
}

/// BEVEL.md §8.3.0.2 — the custom-profile **un-fan**. At the three-boundvert (valence-4)
/// terminal, a `profile_type = CUSTOM` bevel whose lone leg vert lies in the beveled
/// edge's terminal-far-end profile plane emits the fill ring **whole** (one quad at
/// `segments = 2`) instead of fanning it into `segments` triangles. The superellipse
/// bevel of the same corner fans — so the custom face count is strictly lower. The leg
/// vert also slides `offset·√2` under a custom profile (§8.3.0.1).
#[test]
fn bevel_custom_profile_terminal_unfan_valence4_emits_whole_ring() {
    let offset = 0.25_f32;
    // Superellipse baseline: the valence-4 terminal fans into 2 triangles, leg at −offset.
    let (mut super_mesh, super_edge) = noncollinear_valence4_fan();
    let super_result = super_mesh.bevel(
        &[super_edge.into()],
        BevelOptions {
            offset,
            segments: 2,
            affect: BevelAffect::Edges,
            ..Default::default()
        },
    );
    super_mesh.validate_topology();
    assert_eq!(
        leg_fill_sizes(&super_mesh, &super_result.verts, -offset),
        vec![3, 3],
        "superellipse valence-4 terminal fans into two triangles",
    );

    // Custom (LINE preset): the un-fan fires — one whole quad — and the leg vert slides
    // `offset·√2` along the −x leg (§8.3.0.1).
    let curve = BevelCurveProfile::preset_line();
    let (mut custom_mesh, custom_edge) = noncollinear_valence4_fan();
    let custom_result = custom_mesh.bevel(
        &[custom_edge.into()],
        BevelOptions {
            offset,
            segments: 2,
            affect: BevelAffect::Edges,
            profile_type: BevelProfileType::Custom(&curve),
            ..Default::default()
        },
    );
    custom_mesh.validate_topology();
    let custom_leg_x = -offset * std::f32::consts::SQRT_2;
    assert!(
        vert_near(&custom_mesh, Vec3::new(custom_leg_x, 0.0, 0.0), 0.02).is_some(),
        "the custom leg vert slid offset·√2 along the −x leg",
    );
    assert_eq!(
        leg_fill_sizes(&custom_mesh, &custom_result.verts, custom_leg_x),
        vec![4],
        "custom valence-4 terminal emits the ring whole — one quad, the un-fan",
    );
    assert!(
        custom_mesh.face_keys().len() < super_mesh.face_keys().len(),
        "the un-fan drops the total face count below the superellipse fan",
    );
}

/// BEVEL.md §8.6.3.1(4).4 row 6 / §8.2.2.1.1's **Scope**: the terminal cross-section
/// (§8.2.2.1, §8.3.0.2) emits its chain **raw**. It is not a [V-ADJ] cap boundary, so the
/// chord read-back against the `doubled` chain must not run there. Nothing else in the
/// module pins the emission the terminal *site* requests — the chain-level tests exercise
/// both arms of `chain_positions` in isolation — so flipping that request to a resampled
/// emission would otherwise leave every custom-profile test green.
///
/// The driver is the guide's pair of nets that **share `target` and differ in `doubled`**
/// (§8.6.3.1(4).4's model-free arm): the split net, and the same net with the jumped-over
/// `P₂` moved. At `segments = 2` both `target` chains are `(1,0) (0.10,0.60) (0,1)` while
/// their `doubled` chains differ at exactly the strided index, and the guide quotes the pair
/// of *resampled* rim mids as widget `(0.14628, 0.59536)` and `(0.14036, 0.56976)` —
/// `2.6e-2` in the widest widget component, so the declined arm moves loudly.
///
/// A terminal's only curve-dependent mints are its chain interiors: the two strip verts come
/// off the offset and the flanking-leg directions, and the lone leg vert slides `offset·√2`
/// (§8.3.0.1). So the two bevels minting the *same* positions is exactly the statement that
/// both chains were emitted raw. The `late_turn_net` control — a third net whose `target` mid
/// is a genuinely different point — shows the comparison is sensitive to chain interiors at
/// all, rather than passing because nothing curve-dependent got minted.
///
/// **How the discrimination was confirmed.** Switching the terminal builder's own
/// `ChainEmission::Raw` request to `Resampled` was done locally: exactly two of the seven
/// minted positions moved — the two terminals' chain interiors, the strip and leg verts
/// untouched as claimed — and the pair separated by `1.9e-3` world, so this test failed
/// before the change was reverted.
#[test]
fn bevel_custom_profile_terminal_emits_the_raw_target_chain() {
    let split = split_net();
    let moved = split_net_moved_p2();
    assert_widget_chain(
        &CustomChains::new(&moved, 2).target,
        &[(1.00, 0.00), (0.10, 0.60), (0.00, 1.00)],
        "the two nets share their segments = 2 target chain",
    );

    // The resampled arm the terminal must decline separates the pair by 2.6e-2 widget.
    let cell = widget_frame_cell();
    let resampled_mid = |curve: &BevelCurveProfile| {
        let chains = CustomChains::new(curve, 2);
        cell.chain_positions(
            2,
            0.5,
            Some(CustomChainRead::vertex_cap(&chains)),
            ChainEmission::Resampled,
        )[1]
    };
    assert!(
        (resampled_mid(&split) - Vec3::new(0.595_36, 0.146_28, 0.0)).length() < 1.0e-4,
        "split net resampled mid: got {:?}",
        resampled_mid(&split),
    );
    assert!(
        (resampled_mid(&moved) - Vec3::new(0.569_76, 0.140_36, 0.0)).length() < 1.0e-4,
        "moved-P2 net resampled mid: got {:?}",
        resampled_mid(&moved),
    );

    let terminal_mints = |curve: &BevelCurveProfile| {
        let (mut mesh, edge) = noncollinear_valence4_fan();
        let result = mesh.bevel(
            &[edge.into()],
            BevelOptions {
                offset: 0.25,
                segments: 2,
                affect: BevelAffect::Edges,
                profile_type: BevelProfileType::Custom(curve),
                ..Default::default()
            },
        );
        mesh.validate_topology();
        let mut positions: Vec<Vec3> = result
            .verts
            .iter()
            .map(|&v| mesh.vertex_position(v))
            .collect();
        positions.sort_by(|a, b| {
            a.x.total_cmp(&b.x)
                .then(a.y.total_cmp(&b.y))
                .then(a.z.total_cmp(&b.z))
        });
        positions
    };
    let widest_gap = |left: &[Vec3], right: &[Vec3]| {
        assert_eq!(left.len(), right.len(), "both bevels mint the same count");
        left.iter()
            .zip(right.iter())
            .map(|(a, b)| (*a - *b).length())
            .fold(0.0_f32, f32::max)
    };

    let split_mints = terminal_mints(&split);
    let moved_mints = terminal_mints(&moved);
    assert!(
        widest_gap(&split_mints, &moved_mints) < 1.0e-6,
        "the terminal chains are raw target[k], so the doubled chains cannot reach them: \
         {split_mints:?} vs {moved_mints:?}",
    );

    let control = terminal_mints(&late_turn_net());
    assert!(
        widest_gap(&split_mints, &control) > 1.0e-2,
        "a net with a different target mid must move the mints — else the equality above \
         is vacuous",
    );
}

/// BEVEL.md §8.3.0.2: the un-fan requires **non-collinear** flanking legs. On the evenly
/// spaced `planar_fan(4)` the beveled spoke's two flanking legs are antiparallel, so the
/// terminal-far-end profile-plane triangle degenerates and the leg vert falls out of any
/// plane — a custom profile there still **fans**, exactly as the superellipse does. This
/// pins that the un-fan keys on the plane test, not merely on `profile_type = CUSTOM`.
#[test]
fn bevel_custom_profile_no_unfan_when_flanking_collinear() {
    let curve = BevelCurveProfile::preset_line();
    let (mut mesh, centre, rim) = planar_fan(4);
    let beveled = mesh.query_edge_exists(centre, rim[0]).expect("the E spoke");
    let result = mesh.bevel(
        &[beveled.into()],
        BevelOptions {
            offset: 0.25,
            segments: 2,
            affect: BevelAffect::Edges,
            profile_type: BevelProfileType::Custom(&curve),
            ..Default::default()
        },
    );
    mesh.validate_topology();
    assert_eq!(
        leg_fill_sizes(&mesh, &result.verts, -0.25 * std::f32::consts::SQRT_2),
        vec![3, 3],
        "collinear flanking ⇒ degenerate profile plane ⇒ the custom fill still fans",
    );
}
