use super::*;

/// BEVEL.md §8.6: `profile = 0.5` (the default) reproduces the §8.3.6 circular
/// arc sample at the cutoff chain interior — no regression from routing the
/// chain through the general sampler.
#[test]
fn cutoff_segments2_profile_half_is_arc_sample() {
    let interior = cube_corner_cutoff_segments2_profile(0.5);
    let arc_point = Vec3::new(0.75, -0.926_776_7, -0.926_776_7);
    assert!((interior - arc_point).length() < 1.0e-5);
}

/// BEVEL.md §8.6.2 / §8.3.6: the flat `profile = 0.25` chain interior is the
/// straight chord midpoint, not the circular bulge — the special case where the
/// cutoff fill is genuinely [P]-only.
#[test]
fn cutoff_segments2_profile_flat_is_chord_midpoint() {
    let interior = cube_corner_cutoff_segments2_profile(0.25);
    let chord_midpoint = Vec3::new(0.75, -0.875, -0.875);
    assert!((interior - chord_midpoint).length() < 1.0e-5);
}

/// BEVEL.md §8.6.1: a non-default `profile = 0.7` bulges the chain interior past
/// both the circular (0.5) and flat (0.25) samples — the sampler honours the
/// `profile` shape control through the cell mapping, distinct in position.
#[test]
fn cutoff_segments2_profile_bulge_is_distinct() {
    let flat = cube_corner_cutoff_segments2_profile(0.25);
    let circle = cube_corner_cutoff_segments2_profile(0.5);
    let bulge = cube_corner_cutoff_segments2_profile(0.7);

    // All three lie on the y = z bulge diagonal at x = 0.75, ordered by bulge:
    // flat (−0.875) is shallowest, then circle (−0.927), then 0.7 (deeper).
    assert!(bulge.y < circle.y, "0.7 bulges past the circle sample");
    assert!(circle.y < flat.y, "the circle bulges past the flat chord");
    assert!(
        (bulge - circle).length() > 1.0e-3,
        "0.7 is a distinct position"
    );
    assert!(
        (bulge - flat).length() > 1.0e-3,
        "0.7 is not the chord midpoint"
    );
}

/// BEVEL.md §8.3.2 / §8.3.4 / §8.3.0.1: the `s = 3`, `segments = 2` cube-corner
/// cutoff junction folds the corner flat — three cut faces collapsing to one
/// apex (§8.3.4), no bottom face — while the three valence-3 terminals build no
/// fill face, their strips running through (§8.3.0.1). The result stays a closed
/// manifold.
#[test]
fn cutoff_junction_segments2_topology() {
    let (mesh, result) = cube_corner_cutoff_segments2();

    mesh.validate_topology();

    // The §8.3.4 apex collapse: all three sectors' corner verts coincide, so
    // there is one apex vert (BEVEL.md §8.3.4) — the common corner of the three
    // cut faces. Each cut face is a quad: apex + the 3-point chain (`segments + 2
    // = 4` corners, §8.3.5).
    let apex_position = Vec3::new(0.75, -0.75, -0.75);
    let apex: Vec<VertexKey> = result
        .verts
        .iter()
        .copied()
        .filter(|&v| (mesh.vertex_position(v) - apex_position).length() < 1.0e-4)
        .collect();
    assert_eq!(apex.len(), 1, "the three corner verts collapse to one apex");
    let apex = apex[0];

    let cut_faces: Vec<FaceKey> = mesh
        .face_keys()
        .into_iter()
        .filter(|&f| mesh.face_vert_keys(f).contains(&apex))
        .collect();
    assert_eq!(cut_faces.len(), 3, "three cut faces meet at the apex");
    for &face in &cut_faces {
        assert_eq!(
            mesh.face_vert_keys(face).len(),
            4,
            "collapsed cut face is a quad (apex + 3-point chain)",
        );
    }

    // The cube corner cutoff is a closed manifold: every edge radial-2.
    for face in mesh.face_keys() {
        for loop_key in mesh.face_loop_keys(face) {
            let edge = mesh.loop_edge(loop_key);
            assert_eq!(
                mesh.edge_radial_iter(edge).count(),
                2,
                "every edge is shared with exactly one other face",
            );
        }
    }

    // No bottom face beyond the three cut faces: the collapse omits it (§8.3.4).
    // The new geometry is 6 strip quads + 3 cut quads = 9 faces.
    assert_eq!(
        result.faces.len(),
        9,
        "6 strip quads + 3 cut quads, no bottom"
    );
}

/// The same cube-corner cutoff junction at an explicit `profile`, returning the
/// post-bevel mesh + result for the §8.3.3/§8.3.4 collapse-gate tests.
fn cube_corner_cutoff_segments2_mesh(profile: f32) -> (LoopMesh, BevelResult) {
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
            segments: 2,
            vmesh_method: BevelVmeshMethod::Cutoff,
            profile,
            ..Default::default()
        },
    );
    (mesh, result)
}

/// BEVEL.md §8.3.3 / §8.3.4: the 3-way collapse is suppressed for exactly the
/// straight-chord profile (`profile = 0.25`, the `r == 1` line case). The three
/// corner verts stay at their distinct boundvert-ring positions and a bottom
/// triangle closes the cut: two more verts, three more edges, one more face than
/// the curved collapse. The cut faces become pentagons (`C_k → P0 P1 P2 →
/// C_(k+1)`, §8.3.2) instead of the collapsed quads.
#[test]
fn cutoff_junction_segments2_flat_chord_does_not_collapse() {
    let (mesh, result) = cube_corner_cutoff_segments2_mesh(0.25);
    mesh.validate_topology();

    // No single welded apex: the corner ring stays open.
    let apex_position = Vec3::new(0.75, -0.75, -0.75);
    let at_apex = result
        .verts
        .iter()
        .filter(|&&v| (mesh.vertex_position(v) - apex_position).length() < 1.0e-4)
        .count();
    assert_eq!(at_apex, 0, "the flat chord does not collapse to one apex");

    // 6 strip quads + 3 cut pentagons + 1 bottom triangle (§8.3.2 uncollapsed).
    assert_eq!(
        result.faces.len(),
        10,
        "flat chord: 6 strip quads + 3 cut pentagons + 1 bottom triangle",
    );
    let (mut triangles, mut quads, mut pentagons) = (0, 0, 0);
    for &face in &result.faces {
        match mesh.face_vert_keys(face).len() {
            3 => triangles += 1,
            4 => quads += 1,
            5 => pentagons += 1,
            other => panic!("unexpected {other}-gon in the flat-chord cutoff fill"),
        }
    }
    assert_eq!(
        (triangles, quads, pentagons),
        (1, 6, 3),
        "one bottom triangle, six strip quads, three uncollapsed cut pentagons",
    );

    // The bottom triangle's three corner verts are coincident-in-position with the
    // three boundverts (the §8.3.3 down-push is ~0) yet distinct vert keys — the
    // +2 V / +3 E / +1 F the flat chord adds over the collapsed apex.
    let corner = Vec3::new(1.0, -1.0, -1.0);
    let ring_radius = 0.25 * 2.0_f32.sqrt();
    let on_ring = result
        .verts
        .iter()
        .filter(|&&v| ((mesh.vertex_position(v) - corner).length() - ring_radius).abs() < 1.0e-4)
        .count();
    assert_eq!(
        on_ring, 6,
        "three boundverts plus three coincident bottom corner verts on the ring",
    );
}

/// BEVEL.md §8.3.3 / §8.3.4: a curved profile (`profile = 0.7`, `r != 1`) fires
/// the 3-way collapse exactly as the default arc does — three cut quads welded at
/// one apex, no bottom face. The gate is binary, suppressed only for the straight
/// chord, not for any curved profile.
#[test]
fn cutoff_junction_segments2_bulge_still_collapses() {
    let (mesh, result) = cube_corner_cutoff_segments2_mesh(0.7);
    mesh.validate_topology();

    // The collapse apex is the cell-corner down-push point, profile-independent.
    let apex_position = Vec3::new(0.75, -0.75, -0.75);
    let at_apex = result
        .verts
        .iter()
        .filter(|&&v| (mesh.vertex_position(v) - apex_position).length() < 1.0e-4)
        .count();
    assert_eq!(
        at_apex, 1,
        "the curved 0.7 profile still collapses to one apex"
    );

    assert_eq!(
        result.faces.len(),
        9,
        "curved collapses: 6 strip quads + 3 cut quads, no bottom face",
    );
    for &face in &result.faces {
        assert_eq!(
            mesh.face_vert_keys(face).len(),
            4,
            "every collapsed-junction face is a quad",
        );
    }
}

/// BEVEL.md §8.3.3 / §8.3.4: the collapse gate is **binary**, not a continuous
/// bulge-height threshold. A near-chord curved profile (`profile = 0.26`, whose
/// exponent `r ≈ 1.03` is just above the line) still bulges off its chord, so it
/// **does** collapse to one apex — pinning the binary rule against any continuous
/// epsilon threshold that would have wrongly suppressed this near-straight case.
#[test]
fn cutoff_junction_segments2_near_chord_curved_collapses() {
    let (mesh, result) = cube_corner_cutoff_segments2_mesh(0.26);
    mesh.validate_topology();

    let apex_position = Vec3::new(0.75, -0.75, -0.75);
    let at_apex = result
        .verts
        .iter()
        .filter(|&&v| (mesh.vertex_position(v) - apex_position).length() < 1.0e-4)
        .count();
    assert_eq!(
        at_apex, 1,
        "the near-chord curved 0.26 profile still collapses to one apex",
    );

    assert_eq!(
        result.faces.len(),
        9,
        "near-chord curved collapses: 6 strip quads + 3 cut quads, no bottom face",
    );
    for &face in &result.faces {
        assert_eq!(
            mesh.face_vert_keys(face).len(),
            4,
            "every collapsed-junction face is a quad",
        );
    }
}

/// Bevel the four slant edges of the square pyramid's apex with `segments = 2`,
/// `vmesh_method = Cutoff`, at an explicit `profile` (BEVEL.md §8.3, the general
/// valence-`s = 4` cutoff junction). Returns the mesh, result, and apex corner.
fn pyramid_corner_cutoff_segments2(profile: f32) -> (LoopMesh, BevelResult, Vec3) {
    let mut mesh = LoopMesh::new();
    let [apex, b0, b1, b2, b3] = square_pyramid(&mut mesh);
    let corner = mesh.vertex_position(apex);
    let slants = [
        mesh.query_edge_exists(apex, b0).expect("slant {apex,0}"),
        mesh.query_edge_exists(apex, b1).expect("slant {apex,1}"),
        mesh.query_edge_exists(apex, b2).expect("slant {apex,2}"),
        mesh.query_edge_exists(apex, b3).expect("slant {apex,3}"),
    ];
    let result = mesh.bevel(
        &slants.map(Into::into),
        BevelOptions {
            offset: 0.3,
            affect: BevelAffect::Edges,
            segments: 2,
            vmesh_method: BevelVmeshMethod::Cutoff,
            profile,
            ..Default::default()
        },
    );
    (mesh, result, corner)
}

/// BEVEL.md §8.3.3 / §8.3.4: the binary down-push is set by the profile alone,
/// not the junction valence `s`. At an `s = 4` apex the straight chord
/// (`profile = 0.25`, `r == 1`) leaves every corner vert coincident-in-position
/// with its own boundvert on the undisplaced ring (four coincident pairs in the
/// apex region), while the curved profile (`profile = 0.5`) pushes each corner
/// the full cell diagonal into the solid, off the ring (no coincidences). The
/// `s >= 4` bottom face is built either way — the collapse is `s == 3`-only.
#[test]
fn cutoff_junction_valence4_straight_chord_leaves_corners_on_ring() {
    // The apex boundvert ring sits high on the tall pyramid; the base-corner
    // terminal caps sit near z = 0. This threshold isolates the apex junction.
    let apex_region = |mesh: &LoopMesh, result: &BevelResult| -> Vec<Vec3> {
        result
            .verts
            .iter()
            .map(|&v| mesh.vertex_position(v))
            .filter(|p| p.z > 0.6)
            .collect()
    };
    let coincident_pairs = |points: &[Vec3]| -> usize {
        let mut pairs = 0;
        for i in 0..points.len() {
            for j in (i + 1)..points.len() {
                if (points[i] - points[j]).length() < 1.0e-4 {
                    pairs += 1;
                }
            }
        }
        pairs
    };

    let (mesh_chord, result_chord, _) = pyramid_corner_cutoff_segments2(0.25);
    mesh_chord.validate_topology();
    let chord_points = apex_region(&mesh_chord, &result_chord);
    assert_eq!(
        coincident_pairs(&chord_points),
        4,
        "s = 4 straight chord: four corner verts stay coincident with their boundverts",
    );

    let (mesh_curved, result_curved, _) = pyramid_corner_cutoff_segments2(0.5);
    mesh_curved.validate_topology();
    let curved_points = apex_region(&mesh_curved, &result_curved);
    assert_eq!(
        coincident_pairs(&curved_points),
        0,
        "s = 4 curved: every corner is pushed off the ring into the solid",
    );

    // Both valences build the bottom face (the collapse is 3-way-only): each cut
    // is closed, so the apex region carries a distinct pushed/undisplaced ring.
    // The curved corners must land strictly below (into the solid) the chord ring.
    let chord_min_z = chord_points.iter().map(|p| p.z).fold(f32::MAX, f32::min);
    let curved_min_z = curved_points.iter().map(|p| p.z).fold(f32::MAX, f32::min);
    assert!(
        curved_min_z < chord_min_z - 1.0e-4,
        "the curved down-push drives corners below the undisplaced chord ring",
    );
}

/// A closed regular hexagonal bipyramid (BEVEL.md §8.3.3.1 worked example): a
/// top apex at `(0, 0, 1)`, a bottom apex at `(0, 0, -1)`, and six rim verts at
/// radius 1 in the `z = 0` plane at `0°, 60°, … , 300°`. Twelve triangular
/// faces close the solid. Returns `(top, [r0..r5], bottom)`.
fn hexagonal_bipyramid(mesh: &mut LoopMesh) -> (VertexKey, [VertexKey; 6], VertexKey) {
    fn add_tri(mesh: &mut LoopMesh, tri: [VertexKey; 3]) {
        let edges = [
            mesh.create_edge([tri[0], tri[1]], EdgeCreateFlags::NO_DUPLICATE),
            mesh.create_edge([tri[1], tri[2]], EdgeCreateFlags::NO_DUPLICATE),
            mesh.create_edge([tri[2], tri[0]], EdgeCreateFlags::NO_DUPLICATE),
        ];
        mesh.face_create(&tri, &edges, FaceCreateFlags::NONE);
    }

    let top = mesh.create_vertex(Vec3::new(0.0, 0.0, 1.0));
    let bottom = mesh.create_vertex(Vec3::new(0.0, 0.0, -1.0));
    let rim = std::array::from_fn(|k| {
        let angle = k as f32 * std::f32::consts::TAU / 6.0;
        mesh.create_vertex(Vec3::new(angle.cos(), angle.sin(), 0.0))
    });
    for k in 0..6 {
        let a = rim[k];
        let b = rim[(k + 1) % 6];
        add_tri(mesh, [top, a, b]);
        add_tri(mesh, [bottom, b, a]);
    }
    (top, rim, bottom)
}

/// BEVEL.md §8.3.3.1: the valence-6 closed-apex ring is the fixture that pins
/// the un-normalised `sin θ` fold-crease push. Beveling all six top spokes of a
/// regular hexagonal bipyramid at `offset = 0.25`, `segments = 2`, default round
/// profile produces a six-sector cutoff patch whose interior corner ring must
/// land at radius `0.396095`, height `0.407614` — the `+30°` boundvert's corner
/// vert at `(0.343028, 0.198047, 0.407614)`. Normalising the down direction
/// (dropping `sin θ = 0.661438`) over-pushes by `1 / sin θ ≈ 1.512×`, landing
/// the ring at `(0.313410, 0.180948, 0.377997)` instead — this test fails under
/// that bug and passes only with the raw crease.
#[test]
fn cutoff_valence6_apex_ring_carries_sin_theta_push() {
    let mut mesh = LoopMesh::new();
    let (top, rim, _bottom) = hexagonal_bipyramid(&mut mesh);
    let spokes: [EdgeKey; 6] =
        std::array::from_fn(|k| mesh.query_edge_exists(top, rim[k]).expect("top spoke"));
    let result = mesh.bevel(
        &spokes.map(Into::into),
        BevelOptions {
            offset: 0.25,
            affect: BevelAffect::Edges,
            segments: 2,
            vmesh_method: BevelVmeshMethod::Cutoff,
            ..Default::default()
        },
    );
    mesh.validate_topology();

    let expected = Vec3::new(0.343_028, 0.198_047, 0.407_614);
    let over_pushed = Vec3::new(0.313_410, 0.180_948, 0.377_997);

    let at_expected = result
        .verts
        .iter()
        .filter(|&&v| (mesh.vertex_position(v) - expected).length() < 1.0e-4)
        .count();
    assert_eq!(
        at_expected, 1,
        "the +30° corner vert lands at the sin θ-scaled ring position",
    );

    let at_over_pushed = result
        .verts
        .iter()
        .filter(|&&v| (mesh.vertex_position(v) - over_pushed).length() < 1.0e-4)
        .count();
    assert_eq!(
        at_over_pushed, 0,
        "no corner vert sits at the normalised (1/sin θ over-pushed) position",
    );

    // The whole interior ring shares the corrected radius/height, six-fold.
    let ring: Vec<Vec3> = result
        .verts
        .iter()
        .map(|&v| mesh.vertex_position(v))
        .filter(|p| {
            let radius = (p.x * p.x + p.y * p.y).sqrt();
            (radius - 0.396_095).abs() < 1.0e-4 && (p.z - 0.407_614).abs() < 1.0e-4
        })
        .collect();
    assert_eq!(
        ring.len(),
        6,
        "six corner verts form the corrected apex ring"
    );
}

/// BEVEL.md §8.3.0.2.1 / §8.5.5.5(b): the fully-surrounded valence-6 apex under
/// the **flat straight-chord** profile (`profile = 0.25`, `r = 1`) takes the
/// general ADJ fullness cap at its `fullness = 0` planar specialisation. Beveling
/// the six top spokes of a regular hexagonal bipyramid at `offset = 0.25`,
/// `segments = 2`, ADJ closes the apex with a planar hexagonal cap: one centre
/// vert pinned to the boundvert ring plane at the ring centroid `(0, 0, 0.465478)`
/// (no fullness bulge) and six vmesh quads fanning around it — not an open
/// hexagonal hole. This pins the fix for the previously-dropped flat ADJ centre.
#[test]
fn adj_flat_profile_valence6_apex_centre_and_cap() {
    let mut mesh = LoopMesh::new();
    let (top, rim, _bottom) = hexagonal_bipyramid(&mut mesh);
    let spokes: [EdgeKey; 6] =
        std::array::from_fn(|k| mesh.query_edge_exists(top, rim[k]).expect("top spoke"));
    let result = mesh.bevel(
        &spokes.map(Into::into),
        BevelOptions {
            offset: 0.25,
            affect: BevelAffect::Edges,
            segments: 2,
            vmesh_method: BevelVmeshMethod::Adj,
            profile: 0.25,
            ..Default::default()
        },
    );
    mesh.validate_topology();

    // The ring-plane centroid, retained flat (§8.3.0.2.1) — not the round profile's
    // bulged-up centre.
    let flat_centre = Vec3::new(0.0, 0.0, 0.465_478);
    let round_centre = Vec3::new(0.0, 0.0, 0.703_618);

    let centre_verts: Vec<VertexKey> = result
        .verts
        .iter()
        .copied()
        .filter(|&v| (mesh.vertex_position(v) - flat_centre).length() < 1.0e-4)
        .collect();
    assert_eq!(
        centre_verts.len(),
        1,
        "the flat ADJ apex centre lands at the ring-plane centroid (0, 0, 0.465478)",
    );
    let centre = centre_verts[0];

    let bulged = result
        .verts
        .iter()
        .filter(|&&v| (mesh.vertex_position(v) - round_centre).length() < 1.0e-4)
        .count();
    assert_eq!(
        bulged, 0,
        "the flat centre is not bulged to the round profile's fullness position",
    );

    // Six vmesh quads fan around the centre, closing the cap (no open hole).
    let cap_faces: Vec<FaceKey> = result
        .faces
        .iter()
        .copied()
        .filter(|&f| mesh.face_vert_keys(f).contains(&centre))
        .collect();
    assert_eq!(
        cap_faces.len(),
        6,
        "six vmesh quads fan around the flat apex centre"
    );
    for &f in &cap_faces {
        assert_eq!(
            mesh.face_vert_keys(f).len(),
            4,
            "each apex cap face is a quad"
        );
    }
}

/// BEVEL.md §8.3.0.2.1 regression guard: the **round** profile (`profile = 0.5`,
/// the default) apex centre stays **bulged** up off the boundvert ring plane at
/// `(0, 0, 0.703618)` (its fullness>0 extraordinary-centre position), unperturbed
/// by the flat-profile fix. The flat ring-plane centroid must *not* appear.
#[test]
fn adj_round_profile_valence6_apex_centre_unperturbed() {
    let mut mesh = LoopMesh::new();
    let (top, rim, _bottom) = hexagonal_bipyramid(&mut mesh);
    let spokes: [EdgeKey; 6] =
        std::array::from_fn(|k| mesh.query_edge_exists(top, rim[k]).expect("top spoke"));
    let result = mesh.bevel(
        &spokes.map(Into::into),
        BevelOptions {
            offset: 0.25,
            affect: BevelAffect::Edges,
            segments: 2,
            vmesh_method: BevelVmeshMethod::Adj,
            ..Default::default()
        },
    );
    mesh.validate_topology();

    let round_centre = Vec3::new(0.0, 0.0, 0.703_618);
    let flat_centre = Vec3::new(0.0, 0.0, 0.465_478);

    let bulged = result
        .verts
        .iter()
        .filter(|&&v| (mesh.vertex_position(v) - round_centre).length() < 1.0e-4)
        .count();
    assert_eq!(
        bulged, 1,
        "the round ADJ apex centre stays bulged at (0, 0, 0.703618)"
    );

    let flattened = result
        .verts
        .iter()
        .filter(|&&v| (mesh.vertex_position(v) - flat_centre).length() < 1.0e-4)
        .count();
    assert_eq!(
        flattened, 0,
        "the round centre is not pinned to the flat ring plane"
    );
}

/// BEVEL.md §8.2.2.1 / §8.3.6: the lone `segments = 2` interior chain vert is
/// the circular-arc 45° point in the profile cell — bulged toward the beveled
/// edge, **not** the straight chord midpoint. For the Front↔Bottom chain across
/// edge {0,1} it lands at (0.75, -0.927, -0.927), a radius 0.25·√2/2 from the
/// arc center, distinct from the chord midpoint (0.75, -0.875, -0.875).
#[test]
fn cutoff_segments2_interior_vert_is_arc_sample() {
    let (mesh, result) = cube_corner_cutoff_segments2();

    let arc_point = Vec3::new(0.75, -0.926_776_7, -0.926_776_7);
    let chord_midpoint = Vec3::new(0.75, -0.875, -0.875);

    let near_arc = result
        .verts
        .iter()
        .filter(|&&v| (mesh.vertex_position(v) - arc_point).length() < 1.0e-4)
        .count();
    assert_eq!(
        near_arc, 1,
        "the Front-Bottom interior chain vert sits at the arc 45° point",
    );

    let near_chord = result
        .verts
        .iter()
        .filter(|&&v| (mesh.vertex_position(v) - chord_midpoint).length() < 1.0e-4)
        .count();
    assert_eq!(
        near_chord, 0,
        "no interior chain vert sits at the straight chord midpoint (§8.3.6)",
    );

    // Every interior arc sample lies on a circle of radius offset (0.25) about
    // its cell's arc center: the chain bulges, it is not straight-subdivided.
    let arc_center = Vec3::new(0.75, -0.75, -0.75);
    let radius = (arc_point - arc_center).length();
    assert!(
        (radius - 0.25).abs() < 1.0e-4,
        "arc sample sits radius=offset from the (0,0) cell corner",
    );
}

/// BEVEL.md §8.2.1: the boundvert ring is segment-count-independent — the
/// `segments = 2` cutoff places the same three boundverts (offset·√2 from the
/// corner) as the single-segment junction.
#[test]
fn cutoff_segments2_ring_matches_single_segment() {
    let (mesh, result) = cube_corner_cutoff_segments2();
    let corner = Vec3::new(1.0, -1.0, -1.0);
    let expected = 0.25 * 2.0_f32.sqrt();

    // The three boundverts (planar 2-line meets) sit offset·√2 from the corner.
    let ring_verts = result
        .verts
        .iter()
        .filter(|&&v| ((mesh.vertex_position(v) - corner).length() - expected).abs() < 1.0e-4)
        .count();
    assert_eq!(
        ring_verts, 3,
        "three §7.6.2 boundverts, unchanged by segments"
    );

    // The junction corner and all three far endpoints are consumed.
    for removed in [
        corner,
        Vec3::new(-1.0, -1.0, -1.0),
        Vec3::new(1.0, 1.0, -1.0),
        Vec3::new(1.0, -1.0, 1.0),
    ] {
        assert!(
            !mesh
                .vert_keys()
                .into_iter()
                .any(|v| mesh.vertex_position(v) == removed),
            "original endpoint {removed:?} removed",
        );
    }
}

/// BEVEL.md §8.3.0.1: a valence-3 terminal builds no fill face — only the
/// through-strip quads. The whole result is a closed manifold (every edge
/// radial-2), confirming the terminals close by the absorb path, not a cutoff.
#[test]
fn cutoff_segments2_terminals_have_no_fill_and_strips_run_through() {
    let (mesh, _result) = cube_corner_cutoff_segments2();

    // The three beveled edges each shed `segments = 2` strip quads = 6 strip
    // quads; the junction adds 3 cut faces. Every face edge is manifold.
    for face in mesh.face_keys() {
        for loop_key in mesh.face_loop_keys(face) {
            let edge = mesh.loop_edge(loop_key);
            assert_eq!(
                mesh.edge_radial_iter(edge).count(),
                2,
                "closed manifold: every edge shared by exactly two faces",
            );
        }
    }
    mesh.validate_topology();
}
