use super::*;

/// Bevel the cube corner of `cube_corner_adj_segments2` with an inner miter
/// (BEVEL.md §8.9.3), `miter_inner = kind`, pushed out by `spread`, at `segments`.
fn cube_corner_adj_inner_miter(
    kind: BevelMiter,
    spread: f32,
    segments: u32,
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
            vmesh_method: BevelVmeshMethod::Adj,
            miter_inner: kind,
            spread,
            ..Default::default()
        },
    );
    (mesh, result)
}

/// Bevel the cube corner of `cube_corner_adj_segments2` with an inner miter
/// (BEVEL.md §8.9.3), `miter_inner = kind`, pushed out by `spread`, at `segments = 2`.
fn cube_corner_adj_segments2_inner_miter(kind: BevelMiter, spread: f32) -> (LoopMesh, BevelResult) {
    cube_corner_adj_inner_miter(kind, spread, 2)
}

/// BEVEL.md §8.9.3 / §8.9.5.1: the inner miter rebuilds each non-reflex sector
/// "exactly like an outer ARC", so it is segments-general — at odd `segments` the
/// arc's `B1 → B3` internal edge carries the full `segments + 1`-sample inscribed arc
/// and the enlarged ring is filled by the general valence-`s` [V-ADJ] grid rather than
/// declining to the sharp ring. This pins the segments = 3 cube corner: the ARC fires
/// (strictly more geometry than the sharp fallback) and the arc-END boundverts land at
/// their §8.9.4 spread-pushed positions.
#[test]
fn inner_miter_arc_odd_segments_builds_arc_not_sharp() {
    let spread = 0.3;
    let (mesh, result) = cube_corner_adj_inner_miter(BevelMiter::Arc, spread, 3);
    let (sharp_mesh, _sharp) = cube_corner_adj_inner_miter(BevelMiter::Sharp, spread, 3);
    mesh.validate_topology();

    // The ARC fired — three arc starts and their fill, strictly more geometry than
    // the declined sharp ring at the same segments.
    let (miter_v, miter_e, miter_f) = counts(&mesh);
    let (sharp_v, sharp_e, sharp_f) = counts(&sharp_mesh);
    assert!(
        miter_v > sharp_v && miter_e > sharp_e && miter_f > sharp_f,
        "the odd-segments inner miter builds the ARC, not the sharp fallback \
         (miter {:?} vs sharp {:?})",
        (miter_v, miter_e, miter_f),
        (sharp_v, sharp_e, sharp_f),
    );
    assert_eq!(
        (miter_v, miter_e, miter_f),
        (40, 66, 28),
        "segments = 3 inner-miter cube corner is 40 V / 66 E / 28 F",
    );

    // §8.9.3 / §8.9.4: the arc-END boundverts `B1` / `B3` are the chain ends (never
    // resampled), pushed `spread` out along their own beveled edge from the stored
    // meet corner. The bottom (z = -1) sector's meet is (0.75, -0.75, -1); `B1` slides
    // -x to (0.45, -0.75, -1), `B3` slides +y to (0.75, -0.45, -1). The stored corner
    // is a control point, so no vertex materialises there (§8.9.5).
    assert_eq!(
        result_has_vert_near(&mesh, &result, Vec3::new(0.45, -0.75, -1.0)),
        1,
        "arc start B1 sits `spread` out along the -x beveled edge",
    );
    assert_eq!(
        result_has_vert_near(&mesh, &result, Vec3::new(0.75, -0.45, -1.0)),
        1,
        "arc end B3 sits `spread` out along the +y beveled edge",
    );
    assert_eq!(
        result_has_vert_near(&mesh, &result, Vec3::new(0.75, -0.75, -1.0)),
        0,
        "the stored meet corner is a control point, not a materialised vertex",
    );
}

/// BEVEL.md §8.5.5.5(e) / §8.9.5.1: odd `segments` closes the enlarged inner-miter
/// ring on exactly one central `s`-gon. The cube corner splits all three sectors into
/// arc pairs, so the ring is valence-6 and odd parity closes on a single central
/// hexagon; every other fill cell is a quad. Exercised at two odd resolutions to pin
/// the general (not merely segments = 3) build.
#[test]
fn inner_miter_arc_odd_segments_closes_on_central_hexagon() {
    for segments in [3u32, 5] {
        let (mesh, result) = cube_corner_adj_inner_miter(BevelMiter::Arc, 0.3, segments);
        mesh.validate_topology();
        let hexagons = result
            .faces
            .iter()
            .filter(|&&f| mesh.face_vert_keys(f).len() == 6)
            .count();
        let non_quads = result
            .faces
            .iter()
            .filter(|&&f| mesh.face_vert_keys(f).len() != 4)
            .count();
        assert_eq!(
            (hexagons, non_quads),
            (1, 1),
            "segments = {segments} odd inner miter closes on exactly one central \
             hexagon; every other fill cell is a quad",
        );
    }
}

/// BEVEL.md §8.9.1: the reflex/non-reflex sign is a property of the *surface
/// orientation*, not of the order the fixture happened to build its faces in. This
/// cube is the same solid as `mesh_add_cube` — same winding — but its faces are
/// created in the reverse order and each face's loop cycle starts one corner along,
/// which rotates the vertex fan and moves each face's first loop. Both are inputs
/// the corner ring's rotational direction must *not* depend on: if it does, every
/// sector's `cross · n_ref` flips, the convex corner classifies as three reflex
/// sectors, and the inner miter silently declines to the sharp ring.
#[test]
fn inner_miter_arc_fires_independently_of_face_build_order() {
    let mut mesh = LoopMesh::new();

    let v0 = mesh.create_vertex(Vec3::new(1.0, -1.0, -1.0));
    let v1 = mesh.create_vertex(Vec3::new(-1.0, -1.0, -1.0));
    let v2 = mesh.create_vertex(Vec3::new(-1.0, 1.0, -1.0));
    let v3 = mesh.create_vertex(Vec3::new(1.0, 1.0, -1.0));
    let v4 = mesh.create_vertex(Vec3::new(1.0, -1.0, 1.0));
    let v5 = mesh.create_vertex(Vec3::new(-1.0, -1.0, 1.0));
    let v6 = mesh.create_vertex(Vec3::new(-1.0, 1.0, 1.0));
    let v7 = mesh.create_vertex(Vec3::new(1.0, 1.0, 1.0));

    let e0 = mesh.create_edge([v0, v1], EdgeCreateFlags::NONE);
    let e1 = mesh.create_edge([v1, v2], EdgeCreateFlags::NONE);
    let e2 = mesh.create_edge([v2, v3], EdgeCreateFlags::NONE);
    let e3 = mesh.create_edge([v3, v0], EdgeCreateFlags::NONE);
    let e4 = mesh.create_edge([v4, v5], EdgeCreateFlags::NONE);
    let e5 = mesh.create_edge([v5, v6], EdgeCreateFlags::NONE);
    let e6 = mesh.create_edge([v6, v7], EdgeCreateFlags::NONE);
    let e7 = mesh.create_edge([v7, v4], EdgeCreateFlags::NONE);
    let e8 = mesh.create_edge([v0, v4], EdgeCreateFlags::NONE);
    let e9 = mesh.create_edge([v1, v5], EdgeCreateFlags::NONE);
    let e10 = mesh.create_edge([v2, v6], EdgeCreateFlags::NONE);
    let e11 = mesh.create_edge([v3, v7], EdgeCreateFlags::NONE);

    // Same six faces, same winding, reverse creation order, each cycle rotated by one.
    mesh.face_create(&[v4, v5, v1, v0], &[e4, e9, e0, e8], FaceCreateFlags::NONE);
    mesh.face_create(
        &[v2, v6, v7, v3],
        &[e10, e6, e11, e2],
        FaceCreateFlags::NONE,
    );
    mesh.face_create(&[v0, v3, v7, v4], &[e3, e11, e7, e8], FaceCreateFlags::NONE);
    mesh.face_create(&[v4, v7, v6, v5], &[e7, e6, e5, e4], FaceCreateFlags::NONE);
    mesh.face_create(&[v5, v6, v2, v1], &[e5, e10, e1, e9], FaceCreateFlags::NONE);
    mesh.face_create(&[v1, v2, v3, v0], &[e1, e2, e3, e0], FaceCreateFlags::NONE);

    let corner = Vec3::new(1.0, -1.0, -1.0);
    let e01 = cube_edge(&mesh, corner, Vec3::new(-1.0, -1.0, -1.0));
    let e03 = cube_edge(&mesh, corner, Vec3::new(1.0, 1.0, -1.0));
    let e04 = cube_edge(&mesh, corner, Vec3::new(1.0, -1.0, 1.0));

    mesh.bevel(
        &[e01.into(), e03.into(), e04.into()],
        BevelOptions {
            offset: 0.25,
            affect: BevelAffect::Edges,
            segments: 2,
            vmesh_method: BevelVmeshMethod::Adj,
            miter_inner: BevelMiter::Arc,
            spread: 0.3,
            ..Default::default()
        },
    );

    mesh.validate_topology();
    assert_eq!(
        counts(&mesh),
        (26, 42, 18),
        "the inner miter fires on the convex corner whatever order the cube was built in",
    );
}

/// BEVEL.md §8.9.3 / §8.9.5: the inner miter rebuilds every convex non-reflex
/// sector of the cube corner as an arc start, enlarging the tri-corner ring from
/// three boundverts to six and filling it with a six-quad pinwheel — three more
/// faces than the sharp pinwheel (§8.9.6). The result stays a closed manifold and
/// places the spread-pushed arc-start boundverts the sharp fill never mints.
#[test]
fn inner_miter_arc_reshapes_cube_corner() {
    let spread = 0.3;
    let (mesh, result) = cube_corner_adj_segments2_inner_miter(BevelMiter::Arc, spread);
    let (sharp_mesh, _sharp) = cube_corner_adj_segments2();

    mesh.validate_topology();

    // §8.9.6: the inner miter adds one boundvert per non-reflex sector (three of
    // them) and so three more pinwheel quads than the sharp corner.
    let (sharp_v, sharp_e, sharp_f) = counts(&sharp_mesh);
    let (miter_v, miter_e, miter_f) = counts(&mesh);
    assert_eq!(
        (sharp_v, sharp_e, sharp_f),
        (20, 33, 15),
        "the sharp cube-corner ADJ seg=2 baseline is unchanged",
    );
    assert_eq!(
        (miter_v, miter_e, miter_f),
        (26, 42, 18),
        "inner miter grows the corner by +6 V / +9 E / +3 F over the sharp pinwheel",
    );

    // The pinwheel grows from three quads to six, all sharing one welded apex —
    // the lone new vertex incident to six fill quads (vs three when sharp).
    let apex: Vec<VertexKey> = result
        .verts
        .iter()
        .copied()
        .filter(|&v| {
            mesh.face_keys()
                .into_iter()
                .filter(|&f| mesh.face_vert_keys(f).contains(&v))
                .count()
                == 6
        })
        .collect();
    assert_eq!(
        apex.len(),
        1,
        "exactly one apex is shared by six pinwheel quads"
    );
    for &f in &mesh.face_keys() {
        if mesh.face_vert_keys(f).contains(&apex[0]) {
            assert_eq!(
                mesh.face_vert_keys(f).len(),
                4,
                "each pinwheel cell is a quad"
            );
        }
    }

    // Non-vacuous reshape: the arc-start boundverts are pushed `spread` out along
    // the beveled edges from each stored meet-point. The bottom-face (z = -1)
    // sector's §7.6.2 meet sits at offset·√2 from the corner along that face's
    // diagonal — (0.75, -0.75, -1); its arc start pushed `spread` along the -x
    // beveled edge lands at (0.45, -0.75, -1), a vertex the sharp fill never mints
    // (it keeps the single boundvert at the meet-point itself).
    let bottom_meet = Vec3::new(0.75, -0.75, -1.0);
    let arc_end_x = bottom_meet + spread * Vec3::new(-1.0, 0.0, 0.0);
    assert_eq!(
        result_has_vert_near(&mesh, &result, arc_end_x),
        1,
        "an arc boundvert sits `spread` out along the -x beveled edge",
    );
    assert!(
        !sharp_mesh
            .vert_keys()
            .into_iter()
            .any(|v| (sharp_mesh.vertex_position(v) - arc_end_x).length() < 1.0e-4),
        "the sharp corner mints no vertex at the spread-pushed arc boundvert",
    );
    // §8.9.5: the arc bulge midpoint is the inscribed-circle 45° sample, pulled in
    // a fraction `√2 − 1` from the chord midpoint toward the stored meet corner —
    // NOT left at the raw meet-point. For the bottom sector's spread = 0.3 ends
    // B1 = (0.45, -0.75, -1) and B3 = (0.75, -0.45, -1), the guide's worked check
    // lands the bulge at (0.66213, -0.66213, -1) (the y = -1 sibling sector gives
    // (0.66213, -1, -0.66213)).
    let bulge_mid = Vec3::new(0.662_132, -0.662_132, -1.0);
    assert_eq!(
        result_has_vert_near(&mesh, &result, bulge_mid),
        1,
        "the arc bulge midpoint sits at the §8.9.5 inscribed-arc sample",
    );
    assert_eq!(
        result_has_vert_near(&mesh, &result, bottom_meet),
        0,
        "the bulge midpoint is pulled in — no vertex stays at the raw meet corner",
    );
    // §8.9.5: the welded apex is the fullness-seeded, once-subdivided
    // extraordinary centre of the six-boundvert ring at valence `s = 6` — NOT a
    // circumsphere / body-diagonal fit. Seed `centre_0 = centroid + 0.559·(v −
    // centroid)` at 0.88240 per axis, pulled in to 0.859816 by one valence-6
    // subdivision round (γ_6 = 0.523423277, sharpening mask (−1/6, +4/3, −1/6)).
    let apex_position = mesh.vertex_position(apex[0]);
    let expected_apex = Vec3::new(0.859_816, -0.859_816, -0.859_816);
    assert!(
        (apex_position - expected_apex).length() < 1.0e-5,
        "the welded apex sits at the §8.5.5.5 valence-6 extraordinary centre \
         (0.859816, -0.859816, -0.859816), got {apex_position:?}",
    );
}

/// BEVEL.md §8.9.3 / §8.9.5's exponent-scope note, on the **inner** side: §8.9.3 rebuilds a
/// firing non-reflex sector "exactly like an outer `ARC`", so the sampler is
/// inner/outer-blind and the inner miter carries no exponent gate over the analytic band
/// either. The whole slider therefore moves positions only — the enlarged six-boundvert
/// ring of §8.9.6 is built at every exponent — and each sector's bulge lands at
/// §8.9.5's `(2 − 2m)·chord_mid + (2m − 1)·meet` with `m = √profile`, which on this corner
/// reduces along the sector diagonal to `0.45 + 0.3·√profile`: `0.600000` at the straight
/// exponent (the plain chord midpoint, no pull), `0.662132` at the circle, `0.700998` at
/// `0.7`.
#[test]
fn inner_miter_arc_admits_every_analytic_exponent() {
    let arc = |profile: f32, segments: u32| {
        cube_corner_adj_inner_miter_profile(None, 0.3, segments, profile, BevelMiter::Arc)
    };
    let sharp = |profile: f32, segments: u32| {
        cube_corner_adj_inner_miter_profile(None, 0.3, segments, profile, BevelMiter::Sharp)
    };

    for (segments, expected) in [(2_u32, (26_usize, 42_usize, 18_usize)), (3, (40, 66, 28))] {
        for profile in [0.25_f32, 0.5, 0.7] {
            let (mesh, _) = arc(profile, segments);
            mesh.validate_topology();
            assert_eq!(
                counts(&mesh),
                expected,
                "profile {profile}, segments {segments}: the analytic slider cannot decide \
                 whether the inner ARC fires (§8.9.3 / §8.9.6)",
            );
            assert_ne!(
                counts(&mesh),
                counts(&sharp(profile, segments).0),
                "profile {profile}, segments {segments}: …and the census it matches is the \
                 mitered one, not the declined sharp ring",
            );
        }
    }

    for (profile, diagonal) in [
        (0.25_f32, 0.600_000_f32),
        (0.5, 0.662_132),
        (0.7, 0.700_998),
    ] {
        let (mesh, result) = arc(profile, 2);
        // One arc bulge per split sector, on that sector's own face plane.
        for bulge in [
            Vec3::new(diagonal, -1.0, -diagonal),
            Vec3::new(1.0, -diagonal, -diagonal),
            Vec3::new(diagonal, -diagonal, -1.0),
        ] {
            assert_eq!(
                result_has_vert_near(&mesh, &result, bulge),
                1,
                "profile {profile}: the inner arc bulge sits at {bulge:?}",
            );
        }
        if profile != 0.5 {
            assert_eq!(
                result_has_vert_near(&mesh, &result, Vec3::new(0.662_132, -1.0, -0.662_132)),
                0,
                "profile {profile}: the forced-circle bulge must be gone",
            );
        }
    }
}

/// BEVEL.md §8.9.5's [general-exponent reduction](#miter-apex-exponent-general): the
/// mitered cap's kernel — the four averages, the `(−1/6, +4/3, −1/6)` sharpening mask, the
/// Sabin weight, the single even-`2` round — is exponent-**independent**. Exactly two
/// inputs move: the ring chains' raw coarse mids (through `m = √profile`) and the
/// §8.5.5.5(f.3) `fullness` seed. The guide's three-row table for this driver is
/// `0.733333 / 0.859816 / 0.975807`, and the two end rows are what make the point: at
/// `profile = 0.25` law **F2** returns exactly `0`, so the apex is the plain boundvert
/// centroid with no pull toward `v`; at `0.7` law **F3**'s parity fit exceeds `1`, the
/// seed sits *past* `v`, and the apex is pushed further out than the circle's. An
/// implementation that hard-codes the circle's `0.559` reproduces only the middle row.
#[test]
fn inner_miter_arc_apex_is_the_exponent_general_blend() {
    for (profile, expected) in [
        (0.25_f32, 0.733_333_f32),
        (0.5, 0.859_816),
        (0.7, 0.975_807),
    ] {
        let (mesh, result) =
            cube_corner_adj_inner_miter_profile(None, 0.3, 2, profile, BevelMiter::Arc);
        mesh.validate_topology();
        let apex: Vec<VertexKey> = result
            .verts
            .iter()
            .copied()
            .filter(|&v| {
                mesh.face_keys()
                    .into_iter()
                    .filter(|&f| mesh.face_vert_keys(f).contains(&v))
                    .count()
                    == 6
            })
            .collect();
        assert_eq!(
            apex.len(),
            1,
            "profile {profile}: the six-boundvert ring welds one apex (§8.5.5.5(e))",
        );
        let position = mesh.vertex_position(apex[0]);
        assert!(
            (position - Vec3::new(expected, -expected, -expected)).length() < 1.0e-4,
            "profile {profile}: the apex is the (f.3)-seeded extraordinary centre at \
             {expected}, got {position:?}",
        );
    }
}

/// BEVEL.md §8.9.3 / §8.9.3.2(a) gate the inner miter on three things — a non-sharp
/// `miter_inner`, the [V-ADJ] fill, and §8.9.1's per-**sector** verdict — and the `profile`
/// slider is **not** one of them. §8.6.2's `profile ≥ 0.95` sentinel and (f.1) step 1's
/// diversion of an even-`segments` square ring onto (f.5)'s parallel-offset lattice decide
/// how the **cap** is filled, never whether the boundvert ring splits. So the mitered ring
/// is built right across the sentinel band, at either parity.
///
/// Four junction shapes, because a profile-keyed hold is invisible on any single one: the
/// all-convex cube corner, the flat 12-fan's pipe ring (one exactly-straight sector held
/// back) and its reflex ring (one reflex sector held back), and the folded hex tube.
#[test]
fn inner_miter_arc_builds_across_the_square_sentinel_band() {
    for profile in [0.95_f32, 0.97, 0.99, 1.0] {
        for (segments, expected) in [(2_u32, (26, 42, 18)), (3, (40, 66, 28))] {
            let (mesh, _) =
                cube_corner_adj_inner_miter_profile(None, 0.3, segments, profile, BevelMiter::Arc);
            mesh.validate_topology();
            assert_eq!(
                counts(&mesh),
                expected,
                "cube corner, profile {profile}, segments {segments}",
            );
        }
    }
    for (spokes, label) in [
        (&[0_usize, 3, 6][..], "pipe ring 0/90/180"),
        (&[0, 3, 5][..], "reflex ring 0/90/150"),
    ] {
        for (segments, expected) in [(2_u32, (29, 51, 23)), (3, (41, 72, 32))] {
            let (mesh, _) =
                planar_fan12_inner_miter_profile(spokes, segments, 1.0, BevelMiter::Arc);
            assert_eq!(counts(&mesh), expected, "{label}, segments {segments}");
        }
    }
    for (segments, expected) in [(2_u32, (49, 90, 41)), (3, (61, 114, 53))] {
        let (mesh, _) = hex_tube_pipe_inner_miter_profile(segments, 1.0, BevelMiter::Arc);
        assert_eq!(counts(&mesh), expected, "folded tube, segments {segments}");
    }
}

/// The band's two edges, at the junction the sentinel was previously read off. §8.6.2 makes
/// `profile ≥ 0.95` **one corner**, so every slider value in it must produce the identical
/// mesh; `0.94` sits just below the snap window and is a genuinely curved exponent, so it
/// must build the same ring (the split is profile-blind) with a *different* interior — the
/// square lattice against the relaxed cap.
#[test]
fn inner_miter_arc_square_band_is_one_corner_and_ends_at_the_snap() {
    let reference = cube_corner_adj_inner_miter_profile(None, 0.3, 2, 0.95, BevelMiter::Arc).0;
    for profile in [0.97_f32, 0.99, 1.0] {
        let (mesh, _) = cube_corner_adj_inner_miter_profile(None, 0.3, 2, profile, BevelMiter::Arc);
        assert!(
            positions_multiset_eq(&mesh, &reference, 1.0e-5),
            "profile {profile} is the same corner as 0.95 (§8.6.2's wide gate)",
        );
    }
    let (curved, _) = cube_corner_adj_inner_miter_profile(None, 0.3, 2, 0.94, BevelMiter::Arc);
    assert_eq!(
        counts(&curved),
        counts(&reference),
        "the split is profile-blind, so the ring is the same either side of the snap",
    );
    assert!(
        !positions_multiset_eq(&curved, &reference, 1.0e-5),
        "…but 0.94 is a curved exponent, so its cap interior is not the square lattice",
    );
}

/// BEVEL.md §8.5.5.5(f.5): on the general-ring square grid every centreline terminates at
/// `v` (`CL_i[h] = v`), so at **even** `segments` the innermost cap vert **is** the original
/// corner, un-rounded — "that is the whole point of the square profile". Odd `segments`
/// never reaches the lattice ((f.1) step 1 wants even) and closes on the central n-gon
/// instead, so the corner does not survive there. Pinned on the mitered ring, whose
/// boundverts are the miter's `2s` and not the junction's `s`.
#[test]
fn inner_miter_square_ring_welds_the_untouched_corner_at_even_segments() {
    let corner = Vec3::new(1.0, -1.0, -1.0);
    for (segments, expected) in [(2_u32, 1_usize), (3, 0), (4, 1), (5, 0)] {
        let (mesh, result) =
            cube_corner_adj_inner_miter_profile(None, 0.3, segments, 1.0, BevelMiter::Arc);
        mesh.validate_topology();
        assert_eq!(
            result_has_vert_near(&mesh, &result, corner),
            expected,
            "segments {segments}: the square weld keeps `v` at even parity only",
        );
    }
    // Not a parity artefact: the curved neighbour below §8.6.2's snap keeps no vert there
    // at either parity, so this is the sentinel's lattice and not the general cap.
    for segments in [2_u32, 4] {
        let (mesh, result) =
            cube_corner_adj_inner_miter_profile(None, 0.3, segments, 0.94, BevelMiter::Arc);
        assert_eq!(result_has_vert_near(&mesh, &result, corner), 0);
    }
}

/// BEVEL.md §8.5.5.5(f.5)'s pass-2 **two-arc** branch — "reachable via path 2: two
/// *consecutive* non-reflex corners both firing inner miters put an arc start on each side
/// of the sector between them". The all-convex cube corner splits all three sectors, so
/// every flank sector on the enlarged ring is flanked by two arc starts and *every*
/// centreline is seeded by that branch: `C_i` is the meet of the shared beveled edge's line
/// with the line through the sector's two boundverts, hence on the edge's line through `v`.
/// At `segments = 2` the rim redirect makes each flank chain's single interior sample that
/// very seed, so the property is directly observable on the emitted mesh.
#[test]
fn inner_miter_square_flank_seeds_land_on_the_beveled_edge_lines() {
    let corner = Vec3::new(1.0, -1.0, -1.0);
    let (mesh, result) = cube_corner_adj_inner_miter_profile(None, 0.3, 2, 1.0, BevelMiter::Arc);
    mesh.validate_topology();
    // Each flank sector's two boundverts are its neighbours' `B_3` / `B_1`, each slid
    // `spread` out along *this* edge from its own meet-point, so they share this edge's
    // coordinate and their connecting line is skew to it: the branch's foot is that shared
    // coordinate, `offset + spread` out from `v`.
    let along = 0.25_f32 + 0.3;
    for axis in [-Vec3::X, Vec3::Y, Vec3::Z] {
        assert_eq!(
            result_has_vert_near(&mesh, &result, corner + along * axis),
            1,
            "the flank centreline seed sits on the beveled edge's own line, {along} out",
        );
    }
}

/// Bevel `spokes` of a flat 12-spoke fan with the given inner miter, at `offset = 0.2`
/// / `spread = 0.1` / round profile on the [V-ADJ] grid — the §8.9.8 pipe-ring frame.
fn planar_fan12_inner_miter(
    spokes: &[usize],
    segments: u32,
    inner: BevelMiter,
) -> (LoopMesh, BevelResult) {
    planar_fan12_inner_miter_profile(spokes, segments, 0.5, inner)
}

/// [`planar_fan12_inner_miter`] at an explicit `profile` exponent.
fn planar_fan12_inner_miter_profile(
    spokes: &[usize],
    segments: u32,
    profile: f32,
    inner: BevelMiter,
) -> (LoopMesh, BevelResult) {
    let (mut mesh, centre, rim) = planar_fan(12);
    let geom: Vec<GeomKey> = spokes
        .iter()
        .map(|&k| mesh.query_edge_exists(centre, rim[k]).unwrap().into())
        .collect();
    let result = mesh.bevel(
        &geom,
        BevelOptions {
            offset: 0.2,
            affect: BevelAffect::Edges,
            segments,
            profile,
            vmesh_method: BevelVmeshMethod::Adj,
            miter_inner: inner,
            miter_outer: BevelMiter::Sharp,
            spread: 0.1,
            ..Default::default()
        },
    );
    mesh.validate_topology();
    (mesh, result)
}

/// The §8.9.1 classification is **three**-way, and the flat pipe fan is the shape that
/// makes the third class observable: its two collinear beveled spokes cut the disk into
/// two `180°` sides, and the side carrying no intervening beveled spoke is one
/// **exactly-straight** sector (§8.9.8's `#miter-never-pipe`). §7.6.2's `θ_dir` band
/// claims it, so no miter fires there — while the other side, cut by the `90°` spoke
/// into the supplementary pair `(90°, 90°)`, is two ordinary non-reflex sectors.
///
/// The `0°/90°/150°` arm is the contrast: no collinear pair, so nothing is straight and
/// the `210°` gap reads reflex through the signed test.
#[test]
fn miter_sector_classification_is_three_way_on_the_pipe_and_reflex_fans() {
    let classify = |spokes: &[usize]| {
        let (mesh, centre, rim) = planar_fan(12);
        let beveled: Vec<EdgeKey> = spokes
            .iter()
            .map(|&k| mesh.query_edge_exists(centre, rim[k]).unwrap())
            .collect();
        let is_beveled = |edge: EdgeKey| beveled.contains(&edge);
        let corners = mesh.vertex_bevel_corner_ring(centre, &is_beveled);
        let sectors = sector_split(&corners, &is_beveled);
        let vertex_position = mesh.vertex_position(centre);
        sectors
            .iter()
            .map(|sector| {
                mesh.classify_miter_sector(centre, vertex_position, sector)
                    .expect("every fan sector has two non-degenerate flanks")
                    .class
            })
            .collect::<Vec<MiterSectorClass>>()
    };

    // Sector order is the disk walk from the first beveled spoke, so the verdicts are
    // asserted **in place**: for the pipe fan that is `[0→3]`, `[3→6]`, `[6→0]`, and it
    // is the last one — the bare `180°` side, the only sector spanning no beveled spoke
    // — that must be the straight one. A sorted or set-wise compare would accept the
    // same three verdicts pinned to the wrong sectors.
    assert_eq!(
        classify(&[0, 3, 6]),
        vec![
            MiterSectorClass::NonReflex,
            MiterSectorClass::NonReflex,
            MiterSectorClass::Straight,
        ],
        "the pipe fan's supplementary pair is (90°, 90°) non-reflex and its bare 180° \
         side is straight, not a third non-reflex sector",
    );
    assert_eq!(
        classify(&[0, 3, 5]),
        vec![
            MiterSectorClass::NonReflex,
            MiterSectorClass::NonReflex,
            MiterSectorClass::Reflex,
        ],
        "the 0°/90°/150° fan has no collinear pair, so nothing is straight and it is \
         the trailing 210° gap that reads reflex",
    );
}

/// BEVEL.md §8.9.3 / §8.9.3.2(a) put **no** junction-level geometry gate on the inner
/// miter — its gates are `miter_inner != SHARP`, `vmesh_method == ADJ`, and §8.9.1's
/// per-*sector* non-reflex-and-not-straight test. So the flat pipe fan, whose meet-point
/// ring is nothing like a forced tri-corner octant, still fires at `segments ≥ 2`, and
/// it fires on the **supplementary pair only** — the bare `180°` side stays sharp, so
/// the ring goes `3 → 5` and never `3 → 6` (§8.9.8's `#miter-never-pipe`).
///
/// `segments = 1` reads the split count directly: §8.9.3.2(d) makes each split cost
/// exactly `+1 V / +1 E / +0 F`. The `segments ∈ {2, 3}` rows are the recorded oracle
/// for the same ring under the [V-ADJ] grid.
#[test]
fn inner_miter_fires_on_the_pipe_ring_supplementary_pair_only() {
    let pipe = [0usize, 3, 6];

    let (sharp, _) = planar_fan12_inner_miter(&pipe, 1, BevelMiter::Sharp);
    let (arc, _) = planar_fan12_inner_miter(&pipe, 1, BevelMiter::Arc);
    let (sharp_v, sharp_e, sharp_f) = counts(&sharp);
    let (arc_v, arc_e, arc_f) = counts(&arc);
    assert_eq!(
        (
            arc_v as i64 - sharp_v as i64,
            arc_e as i64 - sharp_e as i64,
            arc_f as i64 - sharp_f as i64,
        ),
        (2, 2, 0),
        "exactly two of the three sectors split (§8.9.3.2(d) / §8.9.8): the straight \
         side is not one of them, and the count is never odd",
    );

    assert_eq!(
        counts(&planar_fan12_inner_miter(&pipe, 2, BevelMiter::Arc).0),
        (29, 51, 23),
        "the mitered pipe ring at segments = 2",
    );
    assert_eq!(
        counts(&planar_fan12_inner_miter(&pipe, 3, BevelMiter::Arc).0),
        (41, 72, 32),
        "the mitered pipe ring at segments = 3",
    );
    for segments in [2u32, 3] {
        assert_ne!(
            counts(&planar_fan12_inner_miter(&pipe, segments, BevelMiter::Arc).0),
            counts(&planar_fan12_inner_miter(&pipe, segments, BevelMiter::Sharp).0),
            "the multi-segment inner miter is not inert on an all-convex ring that \
             fails the §8.5.1 tri-corner geometry — that gate is §8.5.1's, not §8.9.3's",
        );
    }
}

/// The `s = 4` pipe cross carries **two** supplementary pairs and no bare side, so all
/// four sectors fire and the ring goes `4 → 8` (§8.9.8). Read at `segments = 1`, where
/// §8.9.3.2(d) makes the vertex delta the split count outright; the `segments ∈ {2, 3}`
/// rows are then the recorded oracle for the same ring under the [V-ADJ] grid, pinned
/// exactly against their own sharp controls.
#[test]
fn inner_miter_pipe_cross_splits_all_four_sectors() {
    let cross = [0usize, 3, 6, 9];

    let (sharp, _) = planar_fan12_inner_miter(&cross, 1, BevelMiter::Sharp);
    let (arc, _) = planar_fan12_inner_miter(&cross, 1, BevelMiter::Arc);
    assert_eq!(
        (
            counts(&arc).0 as i64 - counts(&sharp).0 as i64,
            counts(&arc).1 as i64 - counts(&sharp).1 as i64,
            counts(&arc).2 as i64 - counts(&sharp).2 as i64,
        ),
        (4, 4, 0),
        "both supplementary pairs fire, so the ring is 8 (§8.9.8's even firing count)",
    );

    // Exact censuses, not merely "differs from sharp": an over-splitting classifier —
    // the very failure mode the straight class exists to prevent — also differs from
    // sharp, so `assert_ne!` would pass on a ring of the wrong size. The sharp control
    // is pinned alongside so the pair still states the miter is *not inert* here.
    assert_eq!(
        (
            counts(&planar_fan12_inner_miter(&cross, 2, BevelMiter::Sharp).0),
            counts(&planar_fan12_inner_miter(&cross, 2, BevelMiter::Arc).0),
        ),
        ((29, 52, 24), (37, 64, 28)),
        "the pipe cross at segments = 2, sharp then mitered",
    );
    assert_eq!(
        (
            counts(&planar_fan12_inner_miter(&cross, 3, BevelMiter::Sharp).0),
            counts(&planar_fan12_inner_miter(&cross, 3, BevelMiter::Arc).0),
        ),
        ((40, 72, 33), (56, 96, 41)),
        "the pipe cross at segments = 3, sharp then mitered",
    );
}

/// A straight hexagonal tube: six-sided rings of unit radius at `z = 0 … 4`, bridged by
/// four bands of quads and left open at both ends. Every interior-ring vertex is a
/// valence-4 junction whose two axial rungs are exactly antiparallel, so beveling both
/// of them plus one circumferential rail makes a genuine `s = 3` **pipe** junction that
/// is not flat — unlike [`planar_fan`], the sector faces fold across the rails.
fn hex_tube_five_rings() -> (LoopMesh, Vec<Vec<VertexKey>>) {
    let mut mesh = LoopMesh::new();
    let rings: Vec<Vec<VertexKey>> = (0..5)
        .map(|ring| {
            (0..6)
                .map(|corner| {
                    let angle = corner as f32 / 6.0 * std::f32::consts::TAU;
                    mesh.create_vertex(Vec3::new(angle.cos(), angle.sin(), ring as f32))
                })
                .collect()
        })
        .collect();
    for ring in 0..4 {
        for corner in 0..6 {
            let next = (corner + 1) % 6;
            mesh.face_create_verts(
                &[
                    rings[ring][corner],
                    rings[ring][next],
                    rings[ring + 1][next],
                    rings[ring + 1][corner],
                ],
                FaceCreateFlags::NONE,
            );
        }
    }
    (mesh, rings)
}

/// Bevel the middle-ring pipe junction of [`hex_tube_five_rings`] — both axial rungs at
/// ring `2` / corner `0` plus the rail running to corner `1` — at `offset = 0.2` /
/// `spread = 0.1` on the [V-ADJ] grid.
fn hex_tube_pipe_inner_miter(segments: u32, inner: BevelMiter) -> (LoopMesh, BevelResult) {
    hex_tube_pipe_inner_miter_profile(segments, 0.5, inner)
}

/// [`hex_tube_pipe_inner_miter`] at an explicit `profile` exponent.
fn hex_tube_pipe_inner_miter_profile(
    segments: u32,
    profile: f32,
    inner: BevelMiter,
) -> (LoopMesh, BevelResult) {
    let (mut mesh, rings) = hex_tube_five_rings();
    let geom: Vec<GeomKey> = [
        mesh.query_edge_exists(rings[1][0], rings[2][0]),
        mesh.query_edge_exists(rings[2][0], rings[3][0]),
        mesh.query_edge_exists(rings[2][0], rings[2][1]),
    ]
    .map(|edge| {
        edge.expect("the junction's two rungs and its trailing rail")
            .into()
    })
    .to_vec();
    let result = mesh.bevel(
        &geom,
        BevelOptions {
            offset: 0.2,
            affect: BevelAffect::Edges,
            segments,
            profile,
            vmesh_method: BevelVmeshMethod::Adj,
            miter_inner: inner,
            miter_outer: BevelMiter::Sharp,
            spread: 0.1,
            ..Default::default()
        },
    );
    mesh.validate_topology();
    (mesh, result)
}

/// The same §8.9.8 pipe ring as the flat fan, but on a **folded** junction: the tube's
/// two axial rungs are antiparallel while its faces bend across the rails, so the
/// straight sector's classification cannot be an artefact of a single working plane.
/// The bare `180°` side stays sharp and only the supplementary pair fires — `3 → 5`,
/// never `3 → 6` — and the multi-segment rows are pinned exactly, so an over-splitting
/// classifier is caught by its ring size rather than merely by differing from sharp.
#[test]
fn inner_miter_fires_on_the_folded_tube_pipe_supplementary_pair_only() {
    let (sharp, _) = hex_tube_pipe_inner_miter(1, BevelMiter::Sharp);
    let (arc, _) = hex_tube_pipe_inner_miter(1, BevelMiter::Arc);
    let (sharp_v, sharp_e, sharp_f) = counts(&sharp);
    let (arc_v, arc_e, arc_f) = counts(&arc);
    assert_eq!(
        (
            arc_v as i64 - sharp_v as i64,
            arc_e as i64 - sharp_e as i64,
            arc_f as i64 - sharp_f as i64,
        ),
        (2, 2, 0),
        "two of the three sectors split at §8.9.3.2(d)'s +1 V / +1 E / +0 F each; the \
         straight side between the two rungs is not one of them",
    );

    assert_eq!(
        (
            counts(&hex_tube_pipe_inner_miter(2, BevelMiter::Sharp).0),
            counts(&hex_tube_pipe_inner_miter(2, BevelMiter::Arc).0),
        ),
        ((45, 84, 39), (49, 90, 41)),
        "the folded tube's pipe junction at segments = 2, sharp then mitered",
    );
    assert_eq!(
        (
            counts(&hex_tube_pipe_inner_miter(3, BevelMiter::Sharp).0),
            counts(&hex_tube_pipe_inner_miter(3, BevelMiter::Arc).0),
        ),
        ((53, 102, 49), (61, 114, 53)),
        "the folded tube's pipe junction at segments = 3, sharp then mitered",
    );
}

/// BEVEL.md §8.9.3: `miter_inner` distinguishes only sharp from non-sharp — the
/// `Patch` value yields the identical arc-start construction as `Arc` at an inner
/// (non-reflex) sector, so the two reshape the cube corner the same way.
#[test]
fn inner_miter_patch_equals_arc() {
    let spread = 0.1;
    let (arc_mesh, _arc) = cube_corner_adj_segments2_inner_miter(BevelMiter::Arc, spread);
    let (patch_mesh, _patch) = cube_corner_adj_segments2_inner_miter(BevelMiter::Patch, spread);
    arc_mesh.validate_topology();
    patch_mesh.validate_topology();

    // §8.9.3: at an inner (non-reflex) sector `Patch` runs the identical
    // arc-start construction as `Arc`, so the two meshes are the same up to
    // element relabelling — equal vertex-position multiset AND equal face
    // structure, not merely equal element counts.
    const TOL: f32 = 1.0e-5;
    assert!(
        positions_multiset_eq(&arc_mesh, &patch_mesh, TOL),
        "inner Patch and Arc yield identical vertex-position multisets (§8.9.3)",
    );
    assert!(
        faces_match(&arc_mesh, &patch_mesh, TOL),
        "inner Patch and Arc yield identical face structure up to relabelling (§8.9.3)",
    );
    // Complements the geometric compares: element counts also observe wire /
    // stray edges (belonging to no face) that the position/face-multiset
    // checks would miss.
    assert_eq!(
        counts(&arc_mesh),
        counts(&patch_mesh),
        "inner Patch and Arc produce the same element counts (§8.9.3)",
    );
}

/// BEVEL.md §8.9.3: `miter_inner = Sharp` (the default) leaves the cube corner
/// bit-identical to the no-miter ADJ fill, and a non-sharp inner miter genuinely
/// diverges from it — the miter is never a silent no-op when engaged.
#[test]
fn inner_miter_sharp_is_the_unmitered_corner() {
    let (default_mesh, _default) = cube_corner_adj_segments2();
    let (sharp_mesh, _sharp) = cube_corner_adj_segments2_inner_miter(BevelMiter::Sharp, 0.1);
    let (arc_mesh, _arc) = cube_corner_adj_segments2_inner_miter(BevelMiter::Arc, 0.1);

    // Explicit Sharp == the default (no-miter) corner, regardless of `spread`.
    assert_eq!(
        counts(&default_mesh),
        counts(&sharp_mesh),
        "explicit Sharp inner miter matches the default no-miter corner",
    );
    // A non-sharp inner miter diverges (more geometry).
    assert_ne!(
        counts(&sharp_mesh),
        counts(&arc_mesh),
        "an Arc inner miter is not a silent no-op",
    );
}

/// Bevel the cube corner's three edges at **one segment** (BEVEL.md §8.9.3.2) with
/// the given inner miter and fill method, `offset = 0.25`, `spread = 0.3`.
fn cube_corner_seg1_inner_miter(
    kind: BevelMiter,
    method: BevelVmeshMethod,
) -> (LoopMesh, BevelResult) {
    let mut mesh = LoopMesh::new();
    mesh_add_cube(&mut mesh);
    let corner = Vec3::new(1.0, -1.0, -1.0);
    let e01 = cube_edge(&mesh, corner, Vec3::new(-1.0, -1.0, -1.0));
    let e03 = cube_edge(&mesh, corner, Vec3::new(1.0, 1.0, -1.0));
    let e04 = cube_edge(&mesh, corner, Vec3::new(1.0, -1.0, 1.0));
    let result = mesh.bevel(
        &[e01.into(), e03.into(), e04.into()],
        BevelOptions {
            offset: 0.25,
            affect: BevelAffect::Edges,
            segments: 1,
            vmesh_method: method,
            miter_inner: kind,
            spread: 0.3,
            ..Default::default()
        },
    );
    (mesh, result)
}

/// BEVEL.md §8.9.3.2(a)/(d)/(g): the inner miter's arc-start construction has **no
/// `segments` floor**. At one segment every non-reflex, non-straight sector of the
/// cube corner still splits into its `B1` / `B3` pair, spread-pushed to the same
/// positions the `segments = 2` ring lands on; what degenerates is only the fill,
/// which is the §7.6.3 patch face over the now six-corner ring. The delta is a
/// per-sector **vertex split** — `+1 V / +1 E / +0 F` — so the face count is
/// unchanged and only the vertex/edge counts (and the cap's corner count) see it.
#[test]
fn seg1_inner_miter_arc_splits_every_cube_corner_sector() {
    let (mesh, result) = cube_corner_seg1_inner_miter(BevelMiter::Arc, BevelVmeshMethod::Adj);
    let (sharp_mesh, _sharp) =
        cube_corner_seg1_inner_miter(BevelMiter::Sharp, BevelVmeshMethod::Adj);
    mesh.validate_topology();

    assert_eq!(
        counts(&sharp_mesh),
        (13, 21, 10),
        "the sharp one-segment cube corner baseline is 13 V / 21 E / 10 F",
    );
    assert_eq!(
        counts(&mesh),
        (16, 24, 10),
        "three split sectors add 3 V / 3 E and **no** face (§8.9.3.2(d)/(g))",
    );

    // §8.9.3.2(g): the sharp ring's three meet-points are each replaced by a
    // `spread`-separated pair; no vertex is left at a meet-point, and the arc's
    // stored corner is a control point that is never sampled at one segment (c).
    for meet in [
        Vec3::new(0.75, -1.0, -0.75),
        Vec3::new(1.0, -0.75, -0.75),
        Vec3::new(0.75, -0.75, -1.0),
    ] {
        assert_eq!(
            result_has_vert_near(&mesh, &result, meet),
            0,
            "the retained meet-point {meet:?} is slid away by `spread`, not kept",
        );
    }
    let ring = [
        Vec3::new(1.0, -0.45, -0.75),
        Vec3::new(1.0, -0.75, -0.45),
        Vec3::new(0.75, -1.0, -0.45),
        Vec3::new(0.45, -1.0, -0.75),
        Vec3::new(0.45, -0.75, -1.0),
        Vec3::new(0.75, -0.45, -1.0),
    ];
    for corner in ring {
        assert_eq!(
            result_has_vert_near(&mesh, &result, corner),
            1,
            "the enlarged ring carries exactly one boundvert at {corner:?}",
        );
    }

    // §8.9.3.2(b): the cap is the single patch face over the whole ring — a hexagon,
    // not a grid — and §7.6.3 requires its normal on the junction's outward axis.
    let hexagons: Vec<FaceKey> = result
        .faces
        .iter()
        .copied()
        .filter(|&face| mesh.face_vert_keys(face).len() == 6)
        .collect();
    assert_eq!(
        hexagons.len(),
        1,
        "one hexagonal patch face closes the ring"
    );
    let cap_normal = mesh
        .face_newell_normal(hexagons[0])
        .expect("the patch face has a plane");
    assert!(
        cap_normal.dot(Vec3::new(1.0, -1.0, -1.0).normalize()) > 0.9,
        "the patch face winds outward from the corner (§7.6.3), got {cap_normal:?}",
    );
    // (d): each of the three corner-incident originals gains one corner — quad to
    // pentagon — on top of the three the far terminals already absorbed, and the face
    // count holds because the split mints no face.
    let pentagons = |m: &LoopMesh| {
        m.face_keys()
            .into_iter()
            .filter(|&face| m.face_vert_keys(face).len() == 5)
            .count()
    };
    assert_eq!(
        (pentagons(&sharp_mesh), pentagons(&mesh)),
        (3, 6),
        "the split gives each corner-incident original one more corner",
    );
}

/// BEVEL.md §8.9.3.2(b) / §8.9.5's two-halves breakdown: the `vmesh_method = ADJ`
/// requirement is a **parameter** gate, not a fill-existence one, so it does not
/// narrow at one segment even though no grid runs there. `CUTOFF` normalises both
/// miter modes to `SHARP` up front, and the one-segment `CUTOFF` output is therefore
/// the sharp mesh — bit for bit, not merely count-for-count.
#[test]
fn seg1_inner_miter_cutoff_normalises_to_sharp() {
    let (cutoff_arc, _arc) =
        cube_corner_seg1_inner_miter(BevelMiter::Arc, BevelVmeshMethod::Cutoff);
    let (cutoff_sharp, _sharp) =
        cube_corner_seg1_inner_miter(BevelMiter::Sharp, BevelVmeshMethod::Cutoff);
    cutoff_arc.validate_topology();

    const TOL: f32 = 1.0e-5;
    assert_eq!(
        counts(&cutoff_arc),
        (13, 21, 10),
        "`CUTOFF` at one segment gives the SHARP 13 / 21 / 10, not the mitered 16 / 24 / 10",
    );
    assert!(
        positions_multiset_eq(&cutoff_arc, &cutoff_sharp, TOL)
            && faces_match(&cutoff_arc, &cutoff_sharp, TOL),
        "`CUTOFF` suppresses the miter outright (§8.3.0.3), leaving the sharp mesh",
    );
}

/// BEVEL.md §8.9.3.2(a): `miter_inner` still distinguishes only sharp from non-sharp
/// at one segment — `PATCH` runs the identical arc-start construction as `ARC`, the
/// flat two-boundvert patch being an outer-only shape.
#[test]
fn seg1_inner_miter_patch_equals_arc() {
    let (arc, _a) = cube_corner_seg1_inner_miter(BevelMiter::Arc, BevelVmeshMethod::Adj);
    let (patch, _p) = cube_corner_seg1_inner_miter(BevelMiter::Patch, BevelVmeshMethod::Adj);
    const TOL: f32 = 1.0e-5;
    assert_eq!(counts(&arc), counts(&patch));
    assert!(
        positions_multiset_eq(&arc, &patch, TOL) && faces_match(&arc, &patch, TOL),
        "inner `PATCH` and `ARC` are indistinguishable at one segment (§8.9.3.2(a))",
    );
}

/// BEVEL.md §8.9.3.2(g), the reflex station: at one segment the two non-reflex
/// sectors of the reflex junction split while the reflex sector — `miter_outer`'s,
/// left `SHARP` — keeps its single meet-point untouched. Ring `3 → 5`, so the cap
/// grows from a triangle to a pentagon and the counts move `10 / 16 / 8` →
/// `12 / 18 / 8`.
#[test]
fn seg1_inner_miter_arc_splits_reflex_junction_narrow_sectors() {
    let build = |kind: BevelMiter| {
        let (mut mesh, _apex, apex_edges) = single_face_reflex_junction();
        let geom: Vec<GeomKey> = apex_edges.iter().map(|&e| e.into()).collect();
        let result = mesh.bevel(
            &geom,
            BevelOptions {
                offset: 0.2,
                affect: BevelAffect::Edges,
                segments: 1,
                vmesh_method: BevelVmeshMethod::Adj,
                miter_inner: kind,
                miter_outer: BevelMiter::Sharp,
                spread: 0.3,
                ..Default::default()
            },
        );
        (mesh, result)
    };
    let (sharp_mesh, _sharp) = build(BevelMiter::Sharp);
    let (mesh, result) = build(BevelMiter::Arc);
    mesh.validate_topology();

    assert_eq!(
        counts(&sharp_mesh),
        (10, 16, 8),
        "the sharp one-segment reflex-junction baseline is 10 V / 16 E / 8 F",
    );
    assert_eq!(
        counts(&mesh),
        (12, 18, 8),
        "two split sectors take the ring 3 → 5: +2 V / +2 E / +0 F (§8.9.3.2(g))",
    );

    // Each narrow sector's meet-point (±0.19480, 0.46937, 0.60908) splits into the
    // pair below, each exactly `spread` along its own flanking apex edge.
    for corner in [
        Vec3::new(-0.36801, 0.64257, 0.43587),
        Vec3::new(-0.19480, 0.71349, 0.43471),
        Vec3::new(0.19480, 0.71349, 0.43471),
        Vec3::new(0.36801, 0.64257, 0.43587),
    ] {
        assert_eq!(
            result_has_vert_near(&mesh, &result, corner),
            1,
            "the enlarged ring carries exactly one boundvert at {corner:?}",
        );
    }
    // The reflex sector is untouched — its meet-point is still a ring corner.
    assert_eq!(
        result_has_vert_near(&mesh, &result, Vec3::new(0.0, -0.24495, 1.24495)),
        1,
        "the sharp outer miter keeps the reflex meet-point as the fifth ring corner",
    );
    assert_eq!(
        result
            .faces
            .iter()
            .filter(|&&face| mesh.face_vert_keys(face).len() == 5)
            .count(),
        1,
        "the five-boundvert ring is closed by one pentagonal patch face",
    );
}

/// BEVEL.md §8.9.3.2(i): the enlarged patch face is still a ring-spanning fill and
/// the donor **rule** is unchanged — the rep of the ring's *first* boundvert — but the
/// donated *value* moves, and not because a different boundvert became first. The
/// split clears that same boundvert's own beveled edge, so the primitive resolves it
/// through the `first_edge` arm (`prev_face(g_0)`) where the sharp ring took the
/// own-beveled-edge arm (`prev_face(g_1)`). Painting the cube corner's three incident
/// originals with distinct slots makes the shift observable; an implementation that
/// caches the rep at ring-construction time reproduces the sharp answer here.
#[test]
fn seg1_inner_miter_cap_donor_falls_to_the_first_edge_arm() {
    let painted = || {
        let mut mesh = LoopMesh::new();
        mesh_add_cube(&mut mesh);
        for face in mesh.face_keys() {
            let corners: Vec<Vec3> = mesh
                .face_vert_keys(face)
                .into_iter()
                .map(|v| mesh.vertex_position(v))
                .collect();
            let all = |axis: usize, value: f32| corners.iter().all(|p| p[axis] == value);
            let slot = if all(2, -1.0) {
                1
            } else if all(0, 1.0) {
                2
            } else if all(1, -1.0) {
                3
            } else {
                0
            };
            mesh.set_face_material_index(face, slot);
        }
        mesh
    };
    let corner = Vec3::new(1.0, -1.0, -1.0);
    let bevel = |kind: BevelMiter| {
        let mut mesh = painted();
        let e01 = cube_edge(&mesh, corner, Vec3::new(-1.0, -1.0, -1.0));
        let e03 = cube_edge(&mesh, corner, Vec3::new(1.0, 1.0, -1.0));
        let e04 = cube_edge(&mesh, corner, Vec3::new(1.0, -1.0, 1.0));
        let result = mesh.bevel(
            &[e01.into(), e03.into(), e04.into()],
            BevelOptions {
                offset: 0.25,
                affect: BevelAffect::Edges,
                segments: 1,
                vmesh_method: BevelVmeshMethod::Adj,
                miter_inner: kind,
                spread: 0.3,
                ..Default::default()
            },
        );
        let cap = result
            .faces
            .iter()
            .copied()
            .find(|&face| {
                let sides = mesh.face_vert_keys(face).len();
                sides == if kind == BevelMiter::Sharp { 3 } else { 6 }
            })
            .expect("the junction patch face");
        mesh.face_material_index(cap)
    };

    // The two arms of `representative_face_of_boundvert`, read off the pristine ring:
    // `g_0` opens `B_0`'s sector, `g_1` closes it and is the spoke `B_0` owns while
    // the ring is sharp.
    let mesh = painted();
    let junction = vert_at(&mesh, corner);
    let beveled: Vec<EdgeKey> = [
        cube_edge(&mesh, corner, Vec3::new(-1.0, -1.0, -1.0)),
        cube_edge(&mesh, corner, Vec3::new(1.0, 1.0, -1.0)),
        cube_edge(&mesh, corner, Vec3::new(1.0, -1.0, 1.0)),
    ]
    .into();
    let is_beveled = |edge: EdgeKey| beveled.contains(&edge);
    let corners = mesh.vertex_bevel_corner_ring(junction, &is_beveled);
    let opening: Vec<EdgeKey> = sector_split(&corners, &is_beveled)
        .iter()
        .map(|sector| sector.bounding_edges[0])
        .collect();
    let slot_of = |edge: EdgeKey| {
        mesh.face_material_index(
            mesh.boundvert_representative(junction, edge)
                .expect("the spoke has an arriving face"),
        )
    };
    let own_edge_arm = slot_of(opening[1]);
    let first_edge_arm = slot_of(opening[0]);
    assert_ne!(
        own_edge_arm, first_edge_arm,
        "fixture sanity: the two arms must name different slots to be observable",
    );

    assert_eq!(
        bevel(BevelMiter::Sharp),
        own_edge_arm,
        "the sharp cap takes `prev_face(g_1)` — the boundvert's own beveled edge",
    );
    assert_eq!(
        bevel(BevelMiter::Arc),
        first_edge_arm,
        "the split demotes `B_0` to the `first_edge` arm, so the cap takes \
         `prev_face(g_0)` (§8.9.3.2(i))",
    );
    // Stated relative to the two arms, not as absolute slots: *which* boundvert the
    // ring seeds at is the §6.6.1.2 fan-anchor choice, and the (anchor) branch's
    // absolute answer is documented as non-invariant under a re-seeded ring
    // (§8.9.3.1). The arm demotion above is the part §8.9.3.2(i) actually fixes.
}

/// BEVEL.md §8.9.3.2(f) / §7.6.3's ring-size box: the "no patch face" rule keys on
/// the boundvert **ring size** `n`, never on the beveled-edge count `s`. A
/// two-beveled-edge junction has a two-boundvert ring and no cap under `SHARP`; both
/// of this shape's sectors classify non-reflex and non-straight, so both split, `n`
/// rises to 4 and a quad patch face **appears** — the one station where the
/// per-sector `+1 V / +1 E / +0 F` reading is wrong and the closed form of (e) must
/// be used instead.
#[test]
fn seg1_inner_miter_creates_the_cap_at_a_two_beveled_edge_junction() {
    let build = |kind: BevelMiter| {
        let mut mesh = LoopMesh::new();
        mesh_add_cube(&mut mesh);
        let corner = Vec3::new(1.0, -1.0, -1.0);
        let e01 = cube_edge(&mesh, corner, Vec3::new(-1.0, -1.0, -1.0));
        let e03 = cube_edge(&mesh, corner, Vec3::new(1.0, 1.0, -1.0));
        let result = mesh.bevel(
            &[e01.into(), e03.into()],
            BevelOptions {
                offset: 0.25,
                affect: BevelAffect::Edges,
                segments: 1,
                vmesh_method: BevelVmeshMethod::Adj,
                miter_inner: kind,
                spread: 0.3,
                ..Default::default()
            },
        );
        (mesh, result)
    };
    let (sharp_mesh, _sharp) = build(BevelMiter::Sharp);
    let (mesh, result) = build(BevelMiter::Arc);
    mesh.validate_topology();

    assert_eq!(
        counts(&sharp_mesh),
        (11, 17, 8),
        "the two-beveled-edge junction has no patch face when sharp (n = 2)",
    );
    assert_eq!(
        counts(&mesh),
        (13, 20, 9),
        "+2 V / +3 E / +1 F: the single shared seam edge becomes four ring edges and \
         a quad cap appears (§8.9.3.2(f))",
    );
    assert_eq!(
        result
            .faces
            .iter()
            .filter(|&&face| mesh.face_vert_keys(face).len() == 4)
            .count(),
        3,
        "the two strip quads are joined by the newly-created quad patch face",
    );
}

/// Bevel the three apex edges of `single_face_reflex_junction` (`segments = 2`,
/// `offset = 0.2`, `spread = 0.1`, ADJ, circular profile) with a chosen
/// `miter_inner` and `miter_outer = Sharp`, seaming / sharpening the named apex
/// edges. Apex edge `k` is `v–{a, b, c}[k]`, so the two non-reflex sectors are
/// flanked by `(0, 1)` and `(1, 2)` and the reflex quad by `(2, 0)`.
fn reflex_apex_inner_miter_flagged(
    miter_inner: BevelMiter,
    seamed: &[usize],
    sharped: &[usize],
    mark_seam: bool,
    mark_sharp: bool,
) -> (LoopMesh, BevelResult) {
    let (mut mesh, _apex, apex_edges) = single_face_reflex_junction();
    for &k in seamed {
        mesh.set_edge_seam(apex_edges[k], true);
    }
    for &k in sharped {
        mesh.set_edge_sharp(apex_edges[k], true);
    }
    let geom: Vec<GeomKey> = apex_edges.iter().map(|&e| e.into()).collect();
    let result = mesh.bevel(
        &geom,
        BevelOptions {
            offset: 0.2,
            affect: BevelAffect::Edges,
            segments: 2,
            vmesh_method: BevelVmeshMethod::Adj,
            miter_inner,
            miter_outer: BevelMiter::Sharp,
            spread: 0.1,
            mark_seam,
            mark_sharp,
            ..Default::default()
        },
    );
    (mesh, result)
}

/// BEVEL.md §8.9.1 / §8.9.3: the inner miter fires on **every** non-reflex sector,
/// including at a junction that also carries a reflex one — the reflex sector is
/// `miter_outer`'s, and with the outer miter `Sharp` it simply keeps its single
/// meet-point boundvert while both non-reflex sectors split into their arc pair.
/// The ring therefore grows `3 → 5` boundverts and the corner closes on the general
/// valence-5 cap: `+4 V` (two `B3` + two arc bulges), `+6 E` (each split sector's
/// two-edge fill chain plus the cap spokes), a valid closed manifold throughout.
#[test]
fn inner_miter_arc_splits_nonreflex_sectors_of_reflex_junction() {
    let (arc_mesh, arc_result) =
        reflex_apex_inner_miter_flagged(BevelMiter::Arc, &[], &[], false, false);
    arc_mesh.validate_topology();

    let (sharp_mesh, _sharp) =
        reflex_apex_inner_miter_flagged(BevelMiter::Sharp, &[], &[], false, false);
    assert_eq!(
        counts(&sharp_mesh),
        (17, 28, 13),
        "the sharp reflex-junction baseline is unchanged (§7.6.6)",
    );
    let (vertices, edges, _faces) = counts(&arc_mesh);
    assert_eq!(
        (vertices, edges),
        (21, 34),
        "the inner miter splits both non-reflex sectors: 17 → 21 V, 28 → 34 E (§8.9.6)",
    );
    assert_eq!(
        vertices as i64 - edges as i64 + arc_mesh.face_count() as i64,
        2,
        "the inner-mitered reflex junction stays a closed χ = 2 manifold",
    );

    // The enlarged ring is five boundverts (three sectors, two of them split), so
    // the even-`segments` cap welds five quads at one apex — the sole new vertex
    // with five incident faces, all quads (§8.5.5.5(e)).
    let apex: Vec<VertexKey> = arc_result
        .verts
        .iter()
        .copied()
        .filter(|&candidate| {
            let incident: Vec<FaceKey> = arc_mesh
                .face_keys()
                .into_iter()
                .filter(|&face| arc_mesh.face_vert_keys(face).contains(&candidate))
                .collect();
            incident.len() == 5
                && incident
                    .iter()
                    .all(|&face| arc_mesh.face_vert_keys(face).len() == 4)
        })
        .collect();
    assert_eq!(
        apex.len(),
        1,
        "the five-boundvert ring closes on one welded apex shared by five cap quads",
    );
}

/// BEVEL.md §6.7.5.4 case (6) / §6.7.1.1(a′): an inner-miter **fill chain is a corner
/// run**, flagged by the same unconditional per-rebuilt-face AND as any other — no
/// `miter_*` gate, no "did this sector split" gate. Seaming the two flanks of one
/// non-reflex triangle, **with `mark_seam` off**: the sharp inner miter carries only
/// mechanism (1)'s four outer longitudinals; the arc adds exactly the two fill-chain
/// edges of the sector whose two apex-incident originals are *both* seamed. The other
/// split sector — one seamed flank, one clean — keeps its fill chain clean in the
/// same mesh: the AND bites twice, with opposite answers.
#[test]
fn inner_miter_arc_fill_chain_is_a_flagged_corner_run() {
    let flanks = [0usize, 1];
    let (sharp, sharp_result) =
        reflex_apex_inner_miter_flagged(BevelMiter::Sharp, &flanks, &[], false, false);
    let (arc, arc_result) =
        reflex_apex_inner_miter_flagged(BevelMiter::Arc, &flanks, &[], false, false);

    let sharp_count = count_flagged_result_edges(&sharp, &sharp_result, |m, e| m.edge_seam(e));
    let arc_count = count_flagged_result_edges(&arc, &arc_result, |m, e| m.edge_seam(e));
    assert_eq!(
        sharp_count, 4,
        "sharp inner miter: only mechanism (1)'s two outer longitudinals per seamed strip",
    );
    assert_eq!(
        arc_count, 6,
        "the arc adds exactly the doubly-seamed sector's two fill-chain edges — and \
         leaves the singly-seamed sector's fill chain clean (§6.7.5.4 case (6))",
    );
    arc.validate_topology();
}

/// BEVEL.md §6.7.5.4 cases (5)/(6): the inner miter's extra boundvert **desynchronises**
/// mechanism (2)'s ring walk, and the desync is observable as *missing* bridge edges,
/// never misplaced ones. Same junction, same miter, same seam count — only the anchor's
/// following sector differs:
///
/// - flanks `(0, 1)`: the gap edge `2` is anchored at `1`, whose following sector is the
///   *split* `[v, b, c]`, so the span's single ring step lands on that sector's arc fill
///   chain instead of edge `2`'s strip chain. The corner-run pass runs *after* the walk
///   and *assigns*, so that sector's FALSE AND wipes the burnt step: `mark_seam` ON is
///   canonically identical to OFF (6 → 6) — the parameter is fully inert here, while the
///   same walk with a `Sharp` inner miter bridges normally (4 → 6).
/// - flanks `(1, 2)`: the span is anchored at `2`, whose following sector is the
///   *unsplit* reflex quad, so it bridges normally (6 → 8 = `m · segments`).
#[test]
fn inner_miter_arc_desynchronises_the_ring_walk() {
    let seam = |m: &LoopMesh, e: EdgeKey| m.edge_seam(e);

    // The desynced arm: the burnt step is erased, so ON ≡ OFF.
    let (arc_on, arc_on_r) =
        reflex_apex_inner_miter_flagged(BevelMiter::Arc, &[0, 1], &[], true, false);
    let (arc_off, arc_off_r) =
        reflex_apex_inner_miter_flagged(BevelMiter::Arc, &[0, 1], &[], false, false);
    assert_eq!(
        count_flagged_result_edges(&arc_off, &arc_off_r, seam),
        6,
        "mechanism (1) alone: four longitudinals + the doubly-seamed fill chain",
    );
    assert_eq!(
        count_flagged_result_edges(&arc_on, &arc_on_r, seam),
        6,
        "the desynced span burns its only step on a fill chain, which the corner-run \
         pass then clears: mark_seam is inert at this junction (§6.7.5.4 case (6))",
    );

    // The same walk over an unsplit sector bridges normally.
    let (sharp_on, sharp_on_r) =
        reflex_apex_inner_miter_flagged(BevelMiter::Sharp, &[0, 1], &[], true, false);
    assert_eq!(
        count_flagged_result_edges(&sharp_on, &sharp_on_r, seam),
        6,
        "with no split sector in the span's path the gap bridges: 4 → 6",
    );

    // The mirror configuration: the anchor's following sector is the unsplit reflex
    // quad, so the arc miter bridges normally too.
    let (mirror_on, mirror_on_r) =
        reflex_apex_inner_miter_flagged(BevelMiter::Arc, &[1, 2], &[], true, false);
    let (mirror_off, mirror_off_r) =
        reflex_apex_inner_miter_flagged(BevelMiter::Arc, &[1, 2], &[], false, false);
    assert_eq!(
        count_flagged_result_edges(&mirror_off, &mirror_off_r, seam),
        6,
        "mechanism (1) is symmetric between the two non-reflex sectors",
    );
    assert_eq!(
        count_flagged_result_edges(&mirror_on, &mirror_on_r, seam),
        8,
        "anchored at an unsplit following sector, the span lands on the gap's strip \
         chain: 6 → 8 = m · segments (§6.7.5.4 case (5))",
    );
    arc_on.validate_topology();
    mirror_on.validate_topology();
}

/// BEVEL.md §6.7.5.4 case (6), `mark_sharp` companion: the sharp bit runs the identical
/// mechanisms on an independent flag, so the inner miter's fill-chain corner run and its
/// ring-walk desync reproduce the seam counts exactly.
#[test]
fn inner_miter_arc_fill_chain_sharp_bit_matches_the_seam_counts() {
    let sharp_flag = |m: &LoopMesh, e: EdgeKey| m.edge_sharp(e);
    let (off, off_r) = reflex_apex_inner_miter_flagged(BevelMiter::Arc, &[], &[0, 1], false, false);
    let (on, on_r) = reflex_apex_inner_miter_flagged(BevelMiter::Arc, &[], &[0, 1], false, true);
    let (mirror_on, mirror_on_r) =
        reflex_apex_inner_miter_flagged(BevelMiter::Arc, &[], &[1, 2], false, true);

    assert_eq!(count_flagged_result_edges(&off, &off_r, sharp_flag), 6);
    assert_eq!(
        count_flagged_result_edges(&on, &on_r, sharp_flag),
        6,
        "mark_sharp desyncs on the inner miter exactly as mark_seam does",
    );
    assert_eq!(
        count_flagged_result_edges(&mirror_on, &mirror_on_r, sharp_flag),
        8,
        "the mirror configuration bridges on the sharp bit too",
    );
}

/// BEVEL.md §6.7.5.4 case (2) / §6.7.1.1(a′): one flagged flank alone cannot flag a
/// fill chain — the per-face AND fails — and cannot bridge either (a lone flagged disk
/// edge has no gap to delimit). So a single seamed apex edge leaves only its own strip's
/// two outer longitudinals, with `mark_seam` on or off.
#[test]
fn inner_miter_arc_single_flagged_flank_is_inert() {
    let seam = |m: &LoopMesh, e: EdgeKey| m.edge_seam(e);
    let (on, on_r) = reflex_apex_inner_miter_flagged(BevelMiter::Arc, &[0], &[], true, false);
    let (off, off_r) = reflex_apex_inner_miter_flagged(BevelMiter::Arc, &[0], &[], false, false);
    assert_eq!(
        count_flagged_result_edges(&off, &off_r, seam),
        2,
        "one seamed flank: mechanism (1)'s two longitudinals, no fill chain",
    );
    assert_eq!(
        count_flagged_result_edges(&on, &on_r, seam),
        2,
        "a lone flagged disk edge has no gap to bridge (§6.7.5.4 case (2))",
    );
}

/// BEVEL.md §7.6.1 / §8.9.2.2: a **multi-face** split sector — a non-reflex sector that
/// absorbed an unbeveled leg — re-stitches *every* one of its faces onto the arc's
/// profile-sample run, not just the first. Octagonal apex with legs `0, 2, 3` beveled
/// (leg 1 absorbed into the non-reflex `0 → 2` sector) and an inner arc miter over the
/// `Sharp` reflex sector: the junction vertex must be fully consumed — a face left
/// pointing at it would strand the about-to-be-freed apex.
#[test]
fn inner_miter_arc_multiface_sector_restitches_every_face() {
    let (mut mesh, apex, _rim, apex_leg) = octagonal_bipyramid_all_legs();
    let geom: Vec<GeomKey> = [0usize, 2, 3].iter().map(|&k| apex_leg[k].into()).collect();
    let result = mesh.bevel(
        &geom,
        BevelOptions {
            offset: 0.2,
            affect: BevelAffect::Edges,
            segments: 2,
            vmesh_method: BevelVmeshMethod::Adj,
            miter_inner: BevelMiter::Arc,
            miter_outer: BevelMiter::Sharp,
            spread: 0.1,
            ..Default::default()
        },
    );
    mesh.validate_topology();
    assert!(
        !mesh.vert_keys().contains(&apex),
        "the junction vertex is consumed — no absorbed-leg face still hangs off it",
    );
    assert!(
        !result.verts.is_empty(),
        "the inner miter engaged at the reflex-carrying octagonal apex",
    );
    let (vertices, edges, faces) = counts(&mesh);
    assert_eq!(
        vertices as i64 - edges as i64 + faces as i64,
        2,
        "the multi-face inner-mitered junction stays a closed χ = 2 manifold",
    );
}
