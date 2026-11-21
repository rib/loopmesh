use super::*;

/// The folded T-junction of BEVEL.md §8.6.3.1(6) A.7 — three beveled edges meeting at `O`,
/// the `±x` pair folded `135°` against a tilted lower face and the branch left flat.
/// Returns the mesh and `[E_L, E_R, E_T]`.
fn folded_tee_junction() -> (LoopMesh, [EdgeKey; 3], Vec<VertexKey>) {
    folded_tee_junction_with_left_arm_at(Vec3::new(-1.0, 0.0, 0.0))
}

/// `folded_tee_junction` with `A` — `E_L`'s far endpoint — relocated, so the same
/// junction can be posed with a degenerate left arm.
fn folded_tee_junction_with_left_arm_at(
    left_arm: Vec3,
) -> (LoopMesh, [EdgeKey; 3], Vec<VertexKey>) {
    folded_tee_junction_with_branch_at(left_arm, Vec3::new(0.0, 1.0, 0.0))
}

/// `folded_tee_junction` with the beveled **branch** spoke `C` re-aimed, which is the
/// axis §8.2.2.1's worked table varies (`B = (0,1,0)`, `(1,1,0)`, `(1,4,0)`). The two
/// upper faces' outer corners ride the branch (`P = C − x̂`, `Q = C + x̂`) so both stay
/// simple convex quads in `z = 0` at every pose, and the `(0,1,0)` pose reproduces the
/// original fixture's coordinates exactly. Only the two faces' *plane* matters to the
/// meet, and that is `z = 0` throughout.
fn folded_tee_junction_with_branch_at(
    left_arm: Vec3,
    branch: Vec3,
) -> (LoopMesh, [EdgeKey; 3], Vec<VertexKey>) {
    let mut mesh = LoopMesh::new();
    let coords = [
        left_arm,                   // A
        Vec3::new(0.0, 0.0, 0.0),   // O
        Vec3::new(1.0, 0.0, 0.0),   // B
        branch,                     // C
        branch - Vec3::X,           // P
        branch + Vec3::X,           // Q
        Vec3::new(-1.0, -1.0, 1.0), // R
        Vec3::new(1.0, -1.0, 1.0),  // S
    ];
    let verts: Vec<VertexKey> = coords.iter().map(|&c| mesh.create_vertex(c)).collect();
    for ring in [vec![6usize, 7, 2, 1, 0], vec![0, 1, 3, 4], vec![1, 2, 5, 3]] {
        let face: Vec<VertexKey> = ring.iter().map(|&i| verts[i]).collect();
        mesh.face_create_verts(&face, FaceCreateFlags::NONE);
    }
    let edge = |a: usize, b: usize| {
        mesh.query_edge_exists(verts[a], verts[b])
            .expect("an interior edge")
    };
    let interior = [edge(0, 1), edge(1, 2), edge(1, 3)];
    (mesh, interior, verts)
}

/// The junction boundverts of a bevel at the folded T's `O` — the only result verts
/// that stay within half a unit of it, the ring verts being out at the terminals.
fn junction_boundverts(mesh: &LoopMesh, result: &BevelResult, junction: Vec3) -> Vec<Vec3> {
    let mut found: Vec<Vec3> = result
        .verts
        .iter()
        .map(|&v| mesh.vertex_position(v))
        .filter(|p| (*p - junction).length() < 0.5)
        .collect();
    found.sort_by(|a, b| a.x.total_cmp(&b.x).then(a.y.total_cmp(&b.y)));
    found
}

fn assert_boundverts(found: &[Vec3], expected: &[Vec3], what: &str) {
    assert_eq!(
        found.len(),
        expected.len(),
        "{what}: expected {expected:?}, got {found:?}",
    );
    for (got, want) in found.iter().zip(expected) {
        assert!(
            (*got - *want).length() < 1.0e-4,
            "{what}: expected {expected:?}, got {found:?}",
        );
    }
}

/// BEVEL.md §7.6.2 case 3, the **collinear flank-and-leg** sector. Bevel the folded
/// T's `E_L` and `E_T` — a beveled valence of 2 at `O`, but turning a right angle
/// instead of running through. The sector on the far side spans the *unbeveled*
/// `E_R`, which straddles the fold (`z = 0` faces one side, the `z = -y` pentagon the
/// other), so it is case 3 — and `E_L` is **collinear** with that leg, so `E_L`'s
/// local normal (flank × leg) is exactly zero.
///
/// That zero is load-bearing: §7.6.2's sign fix has no rescue, so the zero survives,
/// zeroes `E_L`'s perpendicular, and leaves `E_L`'s offset line running through `O`
/// along the `x` axis un-offset. It crosses `E_T`'s offset line at `(0.25, 0, 0)` — in
/// the `z = 0` plane its own line lies in. Rescuing the zero to the adjacent face's
/// normal instead pushes that base point a whole `offset` into the fold plane and
/// lands the boundvert at `(0.25, -0.13258, 0.13258)`.
///
/// The two rows are mirror images of each other and that is the point: `E_L + E_T`
/// reaches the zero through one slot of the case-3 branch and `E_R + E_T` through the
/// *other*, so a fix applied to only one of `normal_first` / `normal_second` fails
/// exactly one row. Each row therefore names its own slot and carries its own failure
/// label.
#[test]
fn junction_case3_collinear_flank_and_leg_leaves_the_offset_line_through_v() {
    for (pair, leg_index, zero_slot, expected, label) in [
        (
            [0usize, 2],
            1usize,
            [true, false],
            [Vec3::new(-0.25, 0.25, 0.0), Vec3::new(0.25, 0.0, 0.0)],
            "E_L + E_T: the collinear flank opens the sector, so the zero is reached \
             through `normal_first` and the `walk_first` corner's leg",
        ),
        (
            [1usize, 2],
            0usize,
            [false, true],
            [Vec3::new(-0.25, 0.0, 0.0), Vec3::new(0.25, 0.25, 0.0)],
            "E_R + E_T: the mirror — the collinear flank closes the sector, so the zero \
             is reached through `normal_second` and the `between_edges.last()` leg",
        ),
    ] {
        let (mut mesh, interior, verts) = folded_tee_junction();
        let junction_vert = verts[1];
        let junction = mesh.vertex_position(junction_vert);

        let beveled = [interior[pair[0]], interior[pair[1]]];
        let is_beveled = |edge: EdgeKey| beveled.contains(&edge);
        let leg = interior[leg_index];
        let sector = sector_split(
            &mesh.vertex_bevel_corner_ring(junction_vert, &is_beveled),
            &is_beveled,
        )
        .into_iter()
        .find(|s| s.corners.iter().any(|&(edge, _)| edge == leg))
        .expect("the sector spanning the unbeveled leg");
        let layout = mesh
            .sector_layout(junction_vert, junction, &sector)
            .expect("a two-flank sector has a layout");
        assert!(
            layout.non_planar,
            "{label}: fixture sanity — the leg straddles the fold, so this is case 3",
        );
        assert_eq!(
            [
                layout.normal_first == Vec3::ZERO,
                layout.normal_second == Vec3::ZERO,
            ],
            zero_slot,
            "{label}",
        );

        let result = mesh.bevel(
            &[interior[pair[0]].into(), interior[pair[1]].into()],
            BevelOptions {
                offset: 0.25,
                affect: BevelAffect::Edges,
                segments: 1,
                profile: 0.5,
                vmesh_method: BevelVmeshMethod::Adj,
                ..Default::default()
            },
        );
        let found = junction_boundverts(&mesh, &result, junction);
        assert_boundverts(&found, &expected, label);
        // The rescued-normal answer sits in the fold plane `z = -y`; the correct one
        // is in `z = 0`. Pin the plane directly so a near-miss cannot pass.
        for point in &found {
            assert!(
                point.z.abs() < 1.0e-5,
                "{label}: every boundvert of this junction stays in the z = 0 plane, \
                 got {point:?}",
            );
        }
        mesh.validate_topology();
    }
}

/// The limiting case of the one above: a case-3 sector whose in-between leg has **no
/// direction at all**, its far endpoint sitting on the junction. §7.6.2's case-3 recipe
/// crosses each flank with that leg and normalises, so a directionless leg is a zero
/// cross exactly as a collinear one is, and the shared zero-returning floor answers
/// both the same way — both local normals zero, both perpendiculars zero, both offset
/// lines through `O`, and the meet on `O` itself.
///
/// Here *both* slots go to zero at once (the sector's single leg is simultaneously the
/// `walk_first` corner's leg and `between_edges.last()`), which is what distinguishes
/// this from the collinear pair above. Substituting either flanking face's normal
/// instead pushes both base points a whole `offset` out and moves the meet off `O`.
///
/// Two poses of `A`, pinning two different things:
///
/// - **`A` exactly on `O`** — the rule pin. The leg is identically zero, so flank × leg
///   is exactly zero and the answer holds under *any* direction floor. `E_L` is a leg
///   here and not a beveled edge, so a zero-length one is legal input.
/// - **`A` a hair off `O`** — a **floor-width** pin, not a rule pin. `1e-8` sits above
///   the guide's cross floor but below `edge_direction_from`'s `f32::EPSILON` guard, so
///   the zero here comes from loopmesh's leg-direction floor being the wider of the two.
///   Unifying that floor across the operator is expected to flip this arm to a
///   well-conditioned unit local normal and a base point a full `offset` off `O`, and
///   that flip is a fix, not a regression.
#[test]
fn junction_case3_directionless_leg_zeroes_both_local_normals() {
    for (left_arm, label) in [
        (Vec3::ZERO, "A on O (rule pin)"),
        (
            Vec3::new(-1.0e-8, 0.0, 0.0),
            "A a hair off O (floor-width pin)",
        ),
    ] {
        let (mut mesh, interior, verts) = folded_tee_junction_with_left_arm_at(left_arm);
        let junction_vert = verts[1];
        let junction = mesh.vertex_position(junction_vert);

        let beveled = [interior[1], interior[2]];
        let is_beveled = |edge: EdgeKey| beveled.contains(&edge);
        let sector = sector_split(
            &mesh.vertex_bevel_corner_ring(junction_vert, &is_beveled),
            &is_beveled,
        )
        .into_iter()
        .find(|s| s.corners.iter().any(|&(edge, _)| edge == interior[0]))
        .expect("the sector spanning the degenerate leg");
        let layout = mesh
            .sector_layout(junction_vert, junction, &sector)
            .expect("a two-flank sector has a layout");

        assert!(
            layout.non_planar,
            "{label}: fixture sanity — the leg still straddles the fold, so this is \
             case 3; the two flanking faces stay well-conditioned even though the leg \
             between them does not",
        );
        assert!(
            mesh.edge_direction_from(interior[0], junction_vert, junction)
                .is_none(),
            "{label}: fixture sanity — the leg has no direction at all",
        );
        assert_eq!(
            [
                layout.normal_first,
                layout.normal_second,
                layout.perp_first,
                layout.perp_second,
            ],
            [Vec3::ZERO; 4],
            "{label}: a directionless leg zeroes both local normals, and both \
             perpendiculars with them",
        );

        let offset = 0.25_f32;
        let meet = mesh.solve_meet_point(junction_vert, junction, &sector, offset);
        assert!(
            (meet - junction).length() < 1.0e-5,
            "{label}: both offset lines run through `O` un-offset, so they cross there; \
             got {meet:?}",
        );

        // A sector whose boundvert collapses onto `v` is exactly the shape that can hand
        // the adjacent sectors coincident boundverts and zero-length strip edges.
        let result = mesh.bevel(
            &beveled.map(Into::into),
            BevelOptions {
                offset,
                affect: BevelAffect::Edges,
                segments: 1,
                profile: 0.5,
                vmesh_method: BevelVmeshMethod::Adj,
                ..Default::default()
            },
        );
        assert!(
            !result.verts.is_empty(),
            "{label}: the bevel produced a ring"
        );
        mesh.validate_topology();
    }
}

/// The control for the case above: the **collinear** pair `E_L` + `E_R` at the same
/// junction, same parameters. Both of its sectors are claimed by §7.6.2's
/// near-collinear ladder (the two flanks run straight through `O`), so neither builds
/// an offset line at all and the sign fix is never reached — the upper sector pushes
/// perpendicular in `z = 0`, the lower one in the fold plane `z = -y`.
#[test]
fn collinear_beveled_pair_at_the_folded_tee_junction_pushes_perpendicular() {
    let (mut mesh, interior, verts) = folded_tee_junction();
    let junction = mesh.vertex_position(verts[1]);
    let result = mesh.bevel(
        &[interior[0].into(), interior[1].into()],
        BevelOptions {
            offset: 0.25,
            affect: BevelAffect::Edges,
            segments: 1,
            profile: 0.5,
            vmesh_method: BevelVmeshMethod::Adj,
            ..Default::default()
        },
    );
    let found = junction_boundverts(&mesh, &result, junction);
    assert_boundverts(
        &found,
        &[
            Vec3::new(0.0, -0.176_776_7, 0.176_776_7),
            Vec3::new(0.0, 0.25, 0.0),
        ],
        "collinear beveled pair at the fold",
    );
    mesh.validate_topology();
}

/// BEVEL.md §8.6.3.1(6) A.1/A.3/A.4 on the folded T-junction. The `±x` pair is **one** run
/// through `O` — the contest accepts, dot `1` against a floored-at-zero runner-up — and its
/// four edge-ends hold one physical flank, which on this fixture (a strict low flank, the
/// same one at every end, so no tie and no `z`-twist) means each end's mark equals its own
/// local `z` compare. The branch is a **separate** run: arriving along it both other spokes
/// give dot `0`, so it never links, and its own seed compare ties.
#[test]
fn custom_profile_edge_run_holds_one_flank_through_a_folded_tee() {
    let (mesh, [left, right, branch], verts) = folded_tee_junction();
    let (orientation, set) = orientation_marks(&mesh, &[left, right, branch], 0.25);
    let is_beveled = |edge: EdgeKey| set.contains(&edge);
    let [a, o, b, c] = [verts[0], verts[1], verts[2], verts[3]];

    assert_eq!(
        mesh.orientation_continuation(left, o, &is_beveled),
        Some(right),
        "the through pair continues: dot 1 against a floored-at-zero runner-up",
    );
    assert_eq!(
        mesh.orientation_continuation(right, o, &is_beveled),
        Some(left),
        "and the relation is mutual here, so the decomposition is sweep-independent",
    );
    assert_eq!(
        mesh.orientation_continuation(branch, o, &is_beveled),
        None,
        "arriving along the branch both other spokes give dot 0 — no candidate at all",
    );

    for (edge, endpoint) in [(left, a), (left, o), (right, o), (right, b)] {
        let ends = mesh
            .sector_ends(edge, endpoint, &is_beveled, 0.25)
            .expect("every edge-end of the run spans a sector");
        assert_eq!(
            !orientation.reverse(edge, endpoint),
            ends.start.z < ends.end.z,
            "the run's mark must agree with this end's own compare on a fixture whose \
             low flank is strict and never changes sides",
        );
    }

    // A.4: the branch's two ends carry its *own* determination. Its cross-section is a
    // constant-`z` slice at both ends, so both compares tie and both answer `false`; the
    // `±x` run's orientation constrains it not at all.
    assert_eq!(
        [
            !orientation.reverse(branch, o),
            !orientation.reverse(branch, c),
        ],
        [false, false],
        "the branch seeds itself, on a tie",
    );
}

/// BEVEL.md §8.3.0.3's **asymmetric weld** worked check — one rescued curve, one
/// zero-axis chord — on the folded tee, over **all six** rows of §8.2.2.1's worked
/// table: branch `B = (0,1,0)`, `(1,1,0)` and `(1,4,0)`, each at `segments` 2 and 3.
///
/// The junction is the case §8.2.2.1 ["The other degeneracy"] exists for: the beveled
/// spoke continues straight through `O` into the **unbeveled** collinear leg, whose
/// sector is §7.6.2 case 3 and therefore puts its boundvert `B_1` **on the spoke's own
/// line**. The spoke's closest-approach `mid` is where the chord line meets that line,
/// which is `B_1` itself — one cell axis exactly zero. So the two welded profiles are
/// of different kinds and §8.3.0.3's merge does real work:
///
/// - the **branch**'s `mid` lands on the chord line strictly between the endpoints, so
///   its axes are exactly anti-parallel, the rescue fires, and tier 1 puts the bulge at
///   `v` — a curve;
/// - the **spoke**'s cell has a zero axis, so the rescue does **not** fire (§8.2.2.1:
///   a zero axis reads `60°`, outside the `2°` band), the frame builder declines, and
///   the chain is the **even straight subdivision** of `B_0`–`B_1`.
///
/// The expectation is built here from §8.2.2.1's closed form rather than from copied
/// coordinates, so it pins the *rule*: `w_t = ½·(arc_t + chord_t)` with
/// `arc_t = v + (1 − x_t)·(B_0 − v) + (1 − y_t)·(B_1 − v)` and
/// `chord_t = lerp(B_0, B_1, t / segments)`, `(x_t, y_t) = (sin θ_t, cos θ_t)`,
/// `θ_t = t · 90° / segments`.
///
/// **`segments = 3` is the discriminating count.** §8.2.2.1: *"Their two samples fit
/// **no single quarter-arc** about any apex whatever"* — solving for a common apex
/// gives `(0.01117, 0.05692, 0)` from `w_1` and `(−0.01117, 0.06807, 0)` from `w_2` —
/// so a rule fitted only to `segments = 2`, where the symmetric cross-section makes the
/// two halves collinear with each other, would not be caught.
///
/// **And `(0,1,0)` alone is not the discriminating pose.** That branch is perpendicular
/// to the spoke and its two boundverts are equidistant from `v`, so a merge that swapped
/// the two profiles' roles, or that assumed equal arm lengths, still reproduces it.
/// §8.2.2.1 says why the other two rows are in the table: they *"carry the same check
/// onto junctions whose two boundverts are at different distances from `v` and whose
/// chord is not perpendicular to either spoke."* Both the closed form **and** the
/// guide's own printed `w_t` are asserted, so the rule and its worked numbers are pinned
/// together.
#[test]
fn folded_tee_weld_merges_the_tier1_arc_with_the_zero_axis_even_chord() {
    let junction = Vec3::ZERO;

    // §8.2.2.1's worked table: branch `B`, its two boundverts, and the printed `w_t` at
    // `segments` 2 and 3.
    let table = [
        (
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(-0.25, 0.25, 0.0),
            Vec3::new(0.25, 0.0, 0.0),
            vec![Vec3::new(0.0, 0.099_11, 0.0)],
            vec![
                Vec3::new(-0.087_42, 0.145_83, 0.0),
                Vec3::new(0.087_42, 0.058_41, 0.0),
            ],
        ),
        (
            Vec3::new(1.0, 1.0, 0.0),
            Vec3::new(-0.103_55, 0.25, 0.0),
            Vec3::new(0.353_55, 0.0, 0.0),
            vec![Vec3::new(0.099_11, 0.099_11, 0.0)],
            vec![
                Vec3::new(0.022_20, 0.145_83, 0.0),
                Vec3::new(0.182_04, 0.058_41, 0.0),
            ],
        ),
        (
            Vec3::new(1.0, 4.0, 0.0),
            Vec3::new(-0.195_19, 0.25, 0.0),
            Vec3::new(0.257_69, 0.0, 0.0),
            vec![Vec3::new(0.024_78, 0.099_11, 0.0)],
            vec![
                Vec3::new(-0.053_65, 0.145_83, 0.0),
                Vec3::new(0.104_71, 0.058_41, 0.0),
            ],
        ),
    ];

    for (branch, boundvert_0, boundvert_1, worked_two, worked_three) in table {
        for (segments, worked) in [(2u32, &worked_two), (3, &worked_three)] {
            let (mut mesh, interior, verts) =
                folded_tee_junction_with_branch_at(Vec3::new(-1.0, 0.0, 0.0), branch);
            assert_eq!(
                mesh.vertex_position(verts[1]),
                junction,
                "fixture sanity: the junction is at the origin",
            );
            let result = mesh.bevel(
                &[interior[0].into(), interior[2].into()],
                BevelOptions {
                    offset: 0.25,
                    affect: BevelAffect::Edges,
                    segments,
                    profile: 0.5,
                    vmesh_method: BevelVmeshMethod::Adj,
                    ..Default::default()
                },
            );
            mesh.validate_topology();

            let mut expected = vec![boundvert_0];
            for t in 1..segments {
                let theta = (t as f32) * std::f32::consts::FRAC_PI_2 / segments as f32;
                let arc = junction
                    + (1.0 - theta.sin()) * (boundvert_0 - junction)
                    + (1.0 - theta.cos()) * (boundvert_1 - junction);
                let chord =
                    boundvert_0 + (boundvert_1 - boundvert_0) * (t as f32 / segments as f32);
                expected.push((arc + chord) * 0.5);
            }
            expected.push(boundvert_1);

            let what = format!("folded-tee weld, branch = {branch:?}, segments = {segments}");
            let interior_expected = &expected[1..expected.len() - 1];
            assert_boundverts(
                interior_expected,
                worked,
                &format!("{what}: closed form vs the guide's printed w_t"),
            );

            let found = junction_boundverts(&mesh, &result, junction);
            assert_boundverts(&found, &expected, &what);
        }
    }
}

/// The three readings §8.3.0.3's discrimination table kills at the same fixture, pinned
/// as *absences* so a regression that reproduces any of them fails loudly. The table's
/// own numbers, at `segments = 2`:
///
/// | reading | `w_1` | verdict |
/// |---|---|---|
/// | plain midpoint, zero-axis half = even subdivision (**the rule**) | `(0, 0.09911, 0)` | matches |
/// | merge exception fires, curved sample alone | `(0, 0.07322, 0)` | `2589` quanta out |
/// | one arc struck about `v` for the whole chain | `(0, 0.07322, 0)` | same point |
/// | zero-axis half sampled *through* a near-degenerate cell | `(0.05178, 0.07322, 0)` | `5789` quanta out |
///
/// Rows 2 and 3 are the same point by construction. Row 4 is the reference's own
/// **rung 2** in the `mid = end` mirror — a real behaviour elsewhere, but not here,
/// because this fixture's meet lands exactly on `B_1`.
///
/// §8.3.0.3 forecloses the one-arc reading at this fixture by a measured margin rather
/// than by fiat: *"Here `B_0 = (−0.25, 0.25, 0)` and `B_1 = (0.25, 0, 0)`, so the chord
/// sits **26.57°** off `E` — clear of the band by `26.6°`. … Bending the entry angle
/// into the band would need `mid` displaced by roughly `6.7` world units, about `27×`
/// the `offset`."* The `4.00° ≤ δ ≤ 6.00°` sliver window where one-arc-about-`v` *is*
/// the reference's answer is a sliver-family window and does not reach here.
#[test]
fn folded_tee_weld_is_not_the_one_arc_nor_the_near_degenerate_cell_reading() {
    let junction = Vec3::ZERO;
    let boundvert_0 = Vec3::new(-0.25, 0.25, 0.0);
    let boundvert_1 = Vec3::new(0.25, 0.0, 0.0);

    // Fixture sanity, on the quantity §8.3.0.3 names: the chord's angle to `E`. `E`
    // runs along `x`; anything outside `EPS_ANG` forecloses the rescue-entry rung.
    let chord_vs_edge = unit_chord_angle_between(boundvert_1 - boundvert_0, Vec3::X).to_degrees();
    assert!(
        (26.5..26.6).contains(&chord_vs_edge),
        "the chord sits 26.57° off `E`, far outside the 2° rescue-entry band: \
         {chord_vs_edge}",
    );

    let (mut mesh, interior, _) = folded_tee_junction();
    let result = mesh.bevel(
        &[interior[0].into(), interior[2].into()],
        BevelOptions {
            offset: 0.25,
            affect: BevelAffect::Edges,
            segments: 2,
            profile: 0.5,
            vmesh_method: BevelVmeshMethod::Adj,
            ..Default::default()
        },
    );
    let found = junction_boundverts(&mesh, &result, junction);

    for (wrong, why) in [
        (
            Vec3::new(0.0, 0.073_223, 0.0),
            "rows 2 and 3 — the curved sample alone, i.e. one arc struck about `v` for \
             the whole chain",
        ),
        (
            Vec3::new(0.051_777, 0.073_223, 0.0),
            "row 4 — the zero-axis half sampled *through* a near-degenerate cell, at \
             the cross-section's own spacing instead of evenly",
        ),
    ] {
        assert!(
            found.iter().all(|p| (*p - wrong).length() > 1.0e-3),
            "{why}: {found:?} must not contain {wrong:?}",
        );
    }
}

/// BEVEL.md §8.6.3.1(6) A.7's load-bearing assertion. The folded T is mirror-symmetric in
/// `x`, and a reflection carries a sector's ring `start` onto its image sector's `end`, so
/// the beveled output comes out `x`-symmetric **exactly when** the mirror-paired edge-ends
/// carry opposite marks. Handing the whole run one uniform bit mirrors one arm against the
/// other and displaces a rim sample by the full chord-fraction swap.
#[test]
fn custom_profile_edge_run_keeps_the_folded_tee_mirror_symmetric() {
    let (mut mesh, interior, _) = folded_tee_junction();
    let curve = asymmetric_custom_net();
    let geom: Vec<GeomKey> = interior.iter().map(|&edge| edge.into()).collect();
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

    let positions: Vec<Vec3> = mesh
        .vert_keys()
        .into_iter()
        .map(|v| mesh.vertex_position(v))
        .collect();
    for position in &positions {
        let image = Vec3::new(-position.x, position.y, position.z);
        let nearest = positions
            .iter()
            .map(|other| (*other - image).length())
            .fold(f32::INFINITY, f32::min);
        assert!(
            nearest < 1.0e-5,
            "{position:?} has no mirror partner (nearest {nearest})",
        );
    }

    // The symmetry alone leaves the run's *absolute* handedness unpinned (flipping the
    // whole run maps the output to itself) and is satisfied trivially by a flattened
    // cross-section, so pin the terminal's one interior rim sample outright. A.7 measures
    // it at chord fraction `0.375` from the `z = 0` corner, standing `0.0239` off the
    // chord: reverse A.2's inequality and it swaps to the complementary `0.625`.
    let terminal_rim = Vec3::new(-1.0, 0.099_111_65, 0.088_388_35);
    let samples: Vec<Vec3> = result
        .verts
        .iter()
        .map(|&v| mesh.vertex_position(v))
        .filter(|p| (p.x + 1.0).abs() < 1.0e-5)
        .collect();
    assert!(
        samples
            .iter()
            .any(|p| (*p - terminal_rim).length() < 1.0e-5),
        "the `−x` terminal's interior rim sample must land at {terminal_rim:?}; got \
         {samples:?}",
    );
    let chord_mid = Vec3::new(-1.0, 0.5 * (0.25 - 0.176_776_7), 0.5 * 0.176_776_7);
    assert!(
        (terminal_rim - chord_mid).length() > 1.0e-2,
        "and it must be well off the chord midpoint, or the symmetry assertion above is \
         vacuous",
    );
}

/// BEVEL.md §8.6.3.1(6) A.1's directedness, on the section's own counter-example: `E1`
/// arrives along `+x` at `O`, `E2` leaves along `+x` (dot `1`) and `E3` leaves `5°` off it
/// (dot `0.99619`). From `E1` the two are `0.0038` apart — inside the `1 − cos 10°` margin
/// — so the run stops; from `E2` the `5°` spoke has dot `−0.99619` and is not a candidate
/// at all, so the lone survivor clears the floored-at-zero runner-up and the link exists.
/// A pass that built an undirected graph and took components would merge the two.
#[test]
fn custom_profile_edge_run_continuation_is_directed() {
    let mut mesh = LoopMesh::new();
    let origin = mesh.create_vertex(Vec3::ZERO);
    let five_degrees = 5.0_f32.to_radians();
    let ends = [
        Vec3::new(-1.0, 0.0, 0.0),
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(five_degrees.cos(), five_degrees.sin(), 0.0),
    ];
    let spokes: Vec<EdgeKey> = ends
        .iter()
        .map(|&position| {
            let far = mesh.create_vertex(position);
            mesh.create_edge([origin, far], EdgeCreateFlags::NONE)
        })
        .collect();
    let set: HashSet<EdgeKey> = spokes.iter().copied().collect();
    let is_beveled = |edge: EdgeKey| set.contains(&edge);

    assert_eq!(
        mesh.orientation_continuation(spokes[0], origin, &is_beveled),
        None,
        "two candidates 0.0038 apart in dot are too close to call",
    );
    assert_eq!(
        mesh.orientation_continuation(spokes[1], origin, &is_beveled),
        Some(spokes[0]),
        "from the other side the 5° spoke folds back and drops out of the contest",
    );
}

/// BEVEL.md §8.6.3.1(6) A.1's beveled-valence-2 short-circuit, which the contest never
/// sees: with exactly one other **beveled** spoke at the vertex the run takes it
/// unconditionally, however sharply it folds back. This is not a fast path for a case the
/// contest would agree with — the same spoke, put up against a second beveled one, is a
/// fold-back past a right angle and so is not a candidate at all. And the valence the
/// short-circuit counts is the beveled one: a perfectly straight *unbeveled* continuation
/// sits at this vertex throughout and never enters either reading.
#[test]
fn custom_profile_edge_run_valence_two_takes_its_partner_unconditionally() {
    let mut mesh = LoopMesh::new();
    let origin = mesh.create_vertex(Vec3::ZERO);
    let mut spoke = |position: Vec3| {
        let far = mesh.create_vertex(position);
        mesh.create_edge([origin, far], EdgeCreateFlags::NONE)
    };
    let incoming = spoke(Vec3::new(-1.0, 0.0, 0.0));
    let folded_left = spoke(Vec3::new(-1.0, 1.0, 0.0).normalize());
    let folded_right = spoke(Vec3::new(-1.0, -1.0, 0.0).normalize());
    let _straight_but_unbeveled = spoke(Vec3::new(1.0, 0.0, 0.0));

    let pair: HashSet<EdgeKey> = [incoming, folded_left].into_iter().collect();
    assert_eq!(
        mesh.orientation_continuation(incoming, origin, &|edge| pair.contains(&edge)),
        Some(folded_left),
        "beveled valence 2 continues into a 135° fold-back without asking the contest",
    );

    let trio: HashSet<EdgeKey> = [incoming, folded_left, folded_right].into_iter().collect();
    assert_eq!(
        mesh.orientation_continuation(incoming, origin, &|edge| trio.contains(&edge)),
        None,
        "and from valence 3 the same fold-back is dot −0.707, not a candidate at all",
    );
}

/// BEVEL.md §8.6.3.1(6) A.1/A.3 on a **closed** run. The four `z = +1` edges of a cube form
/// a cycle — every corner has beveled valence 2, so A.1 links straight through the right
/// angle the contest would refuse — and the walk therefore returns to its seed edge-end and
/// `break`s on the revisit rather than re-deriving its bit. Nothing in the pass checks that
/// the closing link's *derived* flank agrees with the recorded one; it should, because an
/// orientable strip carries one physical flank the whole way round, but "should" is what
/// this test converts into an assertion. Every link of the cycle is checked in both
/// directions, the closing hop included.
#[test]
fn custom_profile_edge_run_closes_a_cycle_on_the_flank_it_started() {
    let mut mesh = LoopMesh::new();
    mesh_add_cube(&mut mesh);
    let ring_verts: Vec<VertexKey> = [
        Vec3::new(1.0, -1.0, 1.0),
        Vec3::new(-1.0, -1.0, 1.0),
        Vec3::new(-1.0, 1.0, 1.0),
        Vec3::new(1.0, 1.0, 1.0),
    ]
    .into_iter()
    .map(|position| vert_at(&mesh, position))
    .collect();
    let beveled: Vec<EdgeKey> = (0..4)
        .map(|i| {
            mesh.query_edge_exists(ring_verts[i], ring_verts[(i + 1) % 4])
                .expect("a top-face ring edge")
        })
        .collect();

    let (orientation, set) = orientation_marks(&mesh, &beveled, 0.25);
    let is_beveled = |edge: EdgeKey| set.contains(&edge);

    for (i, &edge) in beveled.iter().enumerate() {
        let corner = ring_verts[(i + 1) % 4];
        assert_eq!(
            mesh.orientation_continuation(edge, corner, &is_beveled),
            Some(beveled[(i + 1) % 4]),
            "the ring is one run: beveled valence 2 links through the 90° corner",
        );
    }
    for &edge in &beveled {
        for endpoint in mesh.edge_vert_keys(edge) {
            assert!(
                orientation.is_start_side.contains_key(&(edge, endpoint)),
                "the cycle's revisit break must not leave an edge-end unmarked",
            );
        }
    }

    for &edge in &beveled {
        for endpoint in mesh.edge_vert_keys(edge) {
            let bit = !orientation.reverse(edge, endpoint);
            for kind in [OrientationStep::CrossVertex, OrientationStep::SameEdge] {
                let Some(next) = mesh.orientation_step(edge, endpoint, kind, &is_beveled) else {
                    continue;
                };
                assert_eq!(
                    !orientation.reverse(next.0, next.1),
                    mesh.propagated_handedness_bit(
                        (edge, endpoint),
                        next,
                        kind,
                        bit,
                        &is_beveled,
                        0.25,
                    ),
                    "the mark recorded at {next:?} must be the one this link derives",
                );
            }
        }
    }
}
