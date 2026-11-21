use super::*;

/// A hexagonal bipyramid: 6 rim verts on the unit circle at `z = 0`, a top apex
/// `(0,0,1)` and a bottom apex `(0,0,-1)`, closed by 12 triangles. Each rim vert
/// is a valence-4 terminal (one top spoke, two rim edges, one bottom spoke).
fn hex_bipyramid() -> (LoopMesh, VertexKey, Vec<VertexKey>) {
    let mut mesh = LoopMesh::new();
    let top = mesh.create_vertex(Vec3::new(0.0, 0.0, 1.0));
    let bottom = mesh.create_vertex(Vec3::new(0.0, 0.0, -1.0));
    let mut rim = Vec::with_capacity(6);
    for i in 0..6 {
        let angle = (i as f32) * std::f32::consts::PI / 3.0;
        rim.push(mesh.create_vertex(Vec3::new(angle.cos(), angle.sin(), 0.0)));
    }
    for i in 0..6 {
        let j = (i + 1) % 6;
        mesh.face_create_verts(&[top, rim[i], rim[j]], FaceCreateFlags::NONE);
        mesh.face_create_verts(&[bottom, rim[j], rim[i]], FaceCreateFlags::NONE);
    }
    (mesh, top, rim)
}

/// Bevel every top-apex spoke of the hexagonal bipyramid at `segments = 2` under
/// `vmesh_method`, returning the beveled mesh.
fn bevel_bipyramid_top_spokes(vmesh_method: BevelVmeshMethod) -> LoopMesh {
    let (mut mesh, top, rim) = hex_bipyramid();
    let spokes: Vec<GeomKey> = rim
        .iter()
        .map(|&r| mesh.query_edge_exists(top, r).expect("top spoke").into())
        .collect();
    mesh.bevel(
        &spokes,
        BevelOptions {
            offset: 0.25,
            affect: BevelAffect::Edges,
            segments: 2,
            vmesh_method,
            ..Default::default()
        },
    );
    mesh
}

/// BEVEL.md §8.3.0.2 worked oracle: beveling the six top-apex spokes of the
/// hexagonal bipyramid at `offset = 0.25`, `segments = 2` mints the four fan
/// verts (`start`, `peak`, `end`, `leg`) at the `P = (1,0,0)` terminal and
/// consumes `P`. The plain-offset (no `√2`) leg slide is pinned explicitly.
#[test]
fn bevel_terminal_fan_valence4_positions() {
    let mesh = bevel_bipyramid_top_spokes(BevelVmeshMethod::Cutoff);
    let tol = 2.0e-3;

    // The original rim vertex is consumed.
    assert!(
        vert_near(&mesh, Vec3::new(1.0, 0.0, 0.0), 1.0e-4).is_none(),
        "the P=(1,0,0) terminal vertex is consumed by the fan close",
    );

    // The four §8.3.0.2 fan verts land at their oracle positions.
    for (label, position) in [
        ("start", Vec3::new(0.8664, 0.2315, 0.0)),
        ("peak", Vec3::new(0.9217, 0.0, 0.0)),
        ("end", Vec3::new(0.8664, -0.2315, 0.0)),
        ("leg", Vec3::new(0.8232, 0.0, -0.1768)),
    ] {
        assert!(
            vert_near(&mesh, position, tol).is_some(),
            "fan vert `{label}` at {position:?} is minted",
        );
    }
}

/// BEVEL.md §8.3.0.2: the leg vert is slid the PLAIN offset (no `√2`) along the
/// non-adjacent (bottom-apex) spoke for the default round profile. `P=(1,0,0)`,
/// bottom apex `(0,0,-1)` ⇒ direction `(-1,0,-1)/√2`, so
/// `leg = P + 0.25·dir = (0.8232, 0, -0.1768)` — a `0.25` slide, not `0.25·√2`.
#[test]
fn bevel_terminal_fan_leg_is_plain_offset() {
    let mesh = bevel_bipyramid_top_spokes(BevelVmeshMethod::Cutoff);
    let plain = vert_near(&mesh, Vec3::new(0.8232, 0.0, -0.1768), 2.0e-3);
    assert!(plain.is_some(), "leg vert slid the plain offset exists");

    // The √2-slid position must NOT be present (the flat-profile trap).
    let over = Vec3::new(0.0, 0.0, -1.0) - Vec3::new(1.0, 0.0, 0.0);
    let over_dir = over / over.length();
    let sqrt2_position = Vec3::new(1.0, 0.0, 0.0) + over_dir * (0.25 * std::f32::consts::SQRT_2);
    assert!(
        vert_near(&mesh, sqrt2_position, 1.0e-3).is_none(),
        "no leg vert at the √2-slid (flat-profile) position {sqrt2_position:?}",
    );
}

/// BEVEL.md §8.3.0.2: the valence-4 / `segments = 2` fan is the two triangles
/// `(start, peak, leg)` and `(peak, end, leg)` — a fan from the leg vert. So the
/// leg vert carries exactly two triangular incident faces (the fan), and the
/// through-chain rungs `start`/`end` are shared with the strip.
#[test]
fn bevel_terminal_fan_is_two_triangles() {
    let mesh = bevel_bipyramid_top_spokes(BevelVmeshMethod::Cutoff);
    let tol = 2.0e-3;
    let leg = vert_near(&mesh, Vec3::new(0.8232, 0.0, -0.1768), tol).expect("leg vert");
    let start = vert_near(&mesh, Vec3::new(0.8664, 0.2315, 0.0), tol).expect("start");
    let peak = vert_near(&mesh, Vec3::new(0.9217, 0.0, 0.0), tol).expect("peak");
    let end = vert_near(&mesh, Vec3::new(0.8664, -0.2315, 0.0), tol).expect("end");

    let triangles: Vec<Vec<VertexKey>> = mesh
        .vert_face_keys(leg)
        .into_iter()
        .map(|f| mesh.face_vert_keys(f))
        .filter(|verts| verts.len() == 3)
        .collect();
    assert_eq!(
        triangles.len(),
        2,
        "the leg vert fans exactly two triangles",
    );

    let has = |set: &[Vec<VertexKey>], a, b, c| {
        set.iter()
            .any(|verts| verts.contains(&a) && verts.contains(&b) && verts.contains(&c))
    };
    assert!(
        has(&triangles, leg, start, peak),
        "fan triangle (leg, start, peak) present",
    );
    assert!(
        has(&triangles, leg, peak, end),
        "fan triangle (leg, peak, end) present",
    );

    // The fan close leaves a manifold solid: no edge carries more than two
    // radial faces (a mis-wound fan triangle would duplicate an edge's use).
    for edge in mesh.edge_keys() {
        assert!(
            mesh.edge_radial_iter(edge).count() <= 2,
            "edge {edge:?} stays manifold (≤ 2 radial faces) after the fan close",
        );
    }
}

/// BEVEL.md §8.3.0.2 net budget: each valence-4 terminal contributes `+3` verts
/// (`start`, `peak`, `end`, `leg`, minus the consumed rim vertex). Across the six
/// rim terminals that is the `+18`-vert budget beyond the apex junction, and the
/// six original rim verts are all consumed.
#[test]
fn bevel_terminal_fan_consumes_rim_and_mints_budget() {
    let mesh = bevel_bipyramid_top_spokes(BevelVmeshMethod::Cutoff);

    for i in 0..6 {
        let angle = (i as f32) * std::f32::consts::PI / 3.0;
        let rim = Vec3::new(angle.cos(), angle.sin(), 0.0);
        assert!(
            vert_near(&mesh, rim, 1.0e-4).is_none(),
            "rim terminal {i} at {rim:?} is consumed",
        );
    }

    // Per rim terminal: exactly 4 fan verts within a 0.4 radius of the old rim
    // position (two flank boundverts, one peak, one leg), for `6 × 4 = 24` fan
    // verts total. (The apex-junction verts sit well inside this radius of the
    // origin, not the rim.)
    let mut fan_total = 0;
    for i in 0..6 {
        let angle = (i as f32) * std::f32::consts::PI / 3.0;
        let rim = Vec3::new(angle.cos(), angle.sin(), 0.0);
        let near = mesh
            .vert_keys()
            .into_iter()
            .filter(|&v| (mesh.vertex_position(v) - rim).length() < 0.4)
            .count();
        assert_eq!(near, 4, "rim terminal {i} mints its 4 fan verts");
        fan_total += near;
    }
    assert_eq!(fan_total, 24, "6 terminals × 4 fan verts");
}

/// BEVEL.md §8.3.0.2: the terminal fan is byte-identical under `Cutoff` and
/// `Adj` — it gates on "terminal", not `vmesh_method`. The rim-region geometry
/// (all verts / faces within `0.4` of each rim terminal) must match between the
/// two whole-mesh bevels; only the apex-junction fill differs.
#[test]
fn bevel_terminal_fan_vmesh_independent() {
    let cutoff = bevel_bipyramid_top_spokes(BevelVmeshMethod::Cutoff);
    let adj = bevel_bipyramid_top_spokes(BevelVmeshMethod::Adj);

    for i in 0..6 {
        let angle = (i as f32) * std::f32::consts::PI / 3.0;
        let rim = Vec3::new(angle.cos(), angle.sin(), 0.0);
        let rim_verts = |mesh: &LoopMesh| -> Vec<Vec3> {
            mesh.vert_keys()
                .into_iter()
                .map(|v| mesh.vertex_position(v))
                .filter(|&p| (p - rim).length() < 0.4)
                .collect()
        };
        let a = rim_verts(&cutoff);
        let b = rim_verts(&adj);
        assert_eq!(
            a.len(),
            b.len(),
            "rim {i} same fan vert count under both fills"
        );
        for target in &a {
            assert!(
                b.iter().any(|q| (*q - *target).length() < 1.0e-4),
                "rim {i} fan vert {target:?} identical under Cutoff and Adj",
            );
        }
    }
}

/// BEVEL.md §7.5 / §8.3.0.2 at `segments = 1`: a lone interior grid edge runs between
/// two **valence-4** terminals. Each mints its two strip verts (the §7.5.1 leg meet)
/// plus one on-edge vert on its single non-flanking leg, consumes the endpoint, and
/// closes with one endpoint fill face — the `(segments + 1) + (n − 3) − 1 = +2`-vert
/// budget per terminal. Whole delta: `+4 V / +7 E / +3 F` (one strip quad, two fills;
/// `E` itself dies with its endpoints).
#[test]
fn edge_bevel_valence_four_terminals_single_segment_grid_counts() {
    let (mut mesh, verts) = quad_grid_5x5();
    assert_eq!(counts(&mesh), (36, 60, 25), "the pristine grid");
    let beveled = mesh
        .query_edge_exists(verts[2][2], verts[3][2])
        .expect("the interior grid edge");

    let result = mesh.bevel(
        &[beveled.into()],
        BevelOptions {
            offset: 0.25,
            affect: BevelAffect::Edges,
            ..Default::default()
        },
    );
    mesh.validate_topology();

    assert_eq!(counts(&mesh), (40, 67, 28));
    assert_eq!(
        result.faces.len(),
        3,
        "one strip quad plus one endpoint fill per terminal",
    );
    assert!(
        winding_conflicts(&mesh).is_empty(),
        "no shared edge is traversed the same way by both its faces",
    );
    for edge in mesh.edge_keys() {
        assert!(
            mesh.edge_radial_iter(edge).count() <= 2,
            "edge {edge:?} stays manifold across the fill",
        );
    }
    for (i, j) in [(2usize, 2usize), (3, 2)] {
        assert!(
            vert_near(&mesh, Vec3::new(i as f32, j as f32, 0.0), 1.0e-4).is_none(),
            "the ({i},{j}) terminal vertex is consumed by the close",
        );
    }
}

/// The same grid, at the geometry: the `(2,2)` terminal's beveled edge runs along
/// `+x`, so its two flanking legs are the `±y` grid edges and its one **non-flanking**
/// leg is the `−x` one. The planar §7.5.1 leg meet lands the two strip verts on the
/// flanking legs' own lines at the plain offset, the on-edge vert is slid the plain
/// offset along the `−x` leg (§7.5), and the endpoint fill is the single triangle over
/// the three of them (§8.3.0.2 at `segments = 1`).
#[test]
fn edge_bevel_valence_four_terminal_single_segment_fill_is_one_triangle() {
    let (mut mesh, verts) = quad_grid_5x5();
    let beveled = mesh
        .query_edge_exists(verts[2][2], verts[3][2])
        .expect("the interior grid edge");
    mesh.bevel(
        &[beveled.into()],
        BevelOptions {
            offset: 0.25,
            affect: BevelAffect::Edges,
            ..Default::default()
        },
    );
    mesh.validate_topology();

    let tol = 1.0e-5;
    let strip_low = vert_near(&mesh, Vec3::new(2.0, 1.75, 0.0), tol).expect("−y strip vert");
    let strip_high = vert_near(&mesh, Vec3::new(2.0, 2.25, 0.0), tol).expect("+y strip vert");
    let leg = vert_near(&mesh, Vec3::new(1.75, 2.0, 0.0), tol).expect("on-edge leg vert");

    let fills: Vec<Vec<VertexKey>> = mesh
        .vert_face_keys(leg)
        .into_iter()
        .map(|face| mesh.face_vert_keys(face))
        .filter(|cycle| cycle.len() == 3)
        .collect();
    assert_eq!(fills.len(), 1, "the leg vert carries exactly one fill face");
    let fill = &fills[0];
    for vert in [strip_low, strip_high, leg] {
        assert!(
            fill.contains(&vert),
            "the fill spans both strip verts and the leg vert",
        );
    }

    // The on-edge vert is the only thing the `−x` leg's near end became: the leg keeps
    // its identity and re-points onto it (§6.5 item 1).
    let leg_edge = mesh
        .query_edge_exists(leg, verts[1][2])
        .expect("the −x leg survives, re-pointed onto its on-edge vert");
    assert_eq!(mesh.edge_radial_iter(leg_edge).count(), 2);
}

/// BEVEL.md §8.3.0.1, the concave-notch arm: at a valence-4 terminal, a superellipse
/// with `profile < 0.25` bows out toward the profile plane's far corner, so the
/// non-`E`-adjacent leg vert slides `offset·√2` — the box diagonal — instead of the
/// plain offset, keeping the fill's fan pivot outside the swept cross-section. The
/// `(2,2)` terminal's beveled edge runs `+x`, so its one non-flanking leg is `−x`, and
/// at `offset = 0.25`, `profile = 0.1` the leg vert lands `0.25·√2 ≈ 0.3536` in from
/// `(2,2)` — at `x ≈ 1.6464`, not the plain `1.75`.
#[test]
fn edge_bevel_valence_four_terminal_leg_vert_slides_root_two_for_concave_profile() {
    let (mut mesh, verts) = quad_grid_5x5();
    let beveled = mesh
        .query_edge_exists(verts[2][2], verts[3][2])
        .expect("the interior grid edge");
    mesh.bevel(
        &[beveled.into()],
        BevelOptions {
            offset: 0.25,
            profile: 0.1,
            affect: BevelAffect::Edges,
            ..Default::default()
        },
    );
    mesh.validate_topology();

    let tol = 1.0e-5;
    let slid = 2.0 - 0.25 * std::f32::consts::SQRT_2;
    assert!(
        vert_near(&mesh, Vec3::new(slid, 2.0, 0.0), tol).is_some(),
        "the concave-notch leg vert slides offset·√2 along the −x leg",
    );
    assert!(
        vert_near(&mesh, Vec3::new(1.75, 2.0, 0.0), tol).is_none(),
        "it does NOT sit at the plain-offset position",
    );
}

/// BEVEL.md §8.3.0.1, the strict `<` gate: `profile = 0.25` is the straight chord
/// (§8.6.2) and takes the **plain** offset, not `offset·√2` — the notch lengthening
/// fires only strictly below `0.25`. Same `(2,2)` valence-4 terminal, so the `−x` leg
/// vert must sit at the plain `1.75`, byte-identical to the `profile = 0.5` default.
#[test]
fn edge_bevel_valence_four_terminal_leg_vert_is_plain_at_profile_quarter() {
    let (mut mesh, verts) = quad_grid_5x5();
    let beveled = mesh
        .query_edge_exists(verts[2][2], verts[3][2])
        .expect("the interior grid edge");
    mesh.bevel(
        &[beveled.into()],
        BevelOptions {
            offset: 0.25,
            profile: 0.25,
            affect: BevelAffect::Edges,
            ..Default::default()
        },
    );
    mesh.validate_topology();

    let tol = 1.0e-5;
    assert!(
        vert_near(&mesh, Vec3::new(1.75, 2.0, 0.0), tol).is_some(),
        "profile = 0.25 is the straight chord: the leg vert takes the plain offset",
    );
    let slid = 2.0 - 0.25 * std::f32::consts::SQRT_2;
    assert!(
        vert_near(&mesh, Vec3::new(slid, 2.0, 0.0), tol).is_none(),
        "the strict `<` gate does not fire at the boundary",
    );
}

/// BEVEL.md §8.3.0.1: the leg-vert `√2` slide is a property of the fill's fan pivot,
/// not of the profile-strip subdivision, so it fires at `segments ≥ 2` exactly as at
/// `segments = 1` (the multi-segment terminal-chain builder). At `segments = 2`,
/// `offset = 0.25`, `profile = 0.1` the `(2,2)` terminal's `−x` leg vert still lands at
/// `x ≈ 1.6464`.
#[test]
fn edge_bevel_valence_four_terminal_leg_vert_slides_root_two_at_segments_two() {
    let (mut mesh, verts) = quad_grid_5x5();
    let beveled = mesh
        .query_edge_exists(verts[2][2], verts[3][2])
        .expect("the interior grid edge");
    mesh.bevel(
        &[beveled.into()],
        BevelOptions {
            offset: 0.25,
            profile: 0.1,
            segments: 2,
            affect: BevelAffect::Edges,
            ..Default::default()
        },
    );
    mesh.validate_topology();

    let tol = 1.0e-5;
    let slid = 2.0 - 0.25 * std::f32::consts::SQRT_2;
    assert!(
        vert_near(&mesh, Vec3::new(slid, 2.0, 0.0), tol).is_some(),
        "the notch slide is not gated on segment count — it fires at segments = 2",
    );
    assert!(
        vert_near(&mesh, Vec3::new(1.75, 2.0, 0.0), tol).is_none(),
        "and the leg vert is not left at the plain-offset position",
    );
}

/// BEVEL.md §7.5 / §8.3.0.2: a **boundary** valence-4 terminal is filled like any other.
/// Its legs are an **arc**, not a cycle, but the ring order §8.3.0.2 asks for is a
/// *rotational disk order* statement, and the arc carries that order already — `leg₁`
/// is the non-flanking leg following `end` around the disk, the walk continuing across
/// the fan's gap. So the ring is `[start, end, leg₁]` and the fill is its one triangle,
/// exactly as at an interior terminal; the single ring edge that spans the gap comes out
/// as a boundary edge of the fill.
///
/// At `(2,2)` of the holed grid the beveled `+x` edge is flanked by the `±y` legs, and
/// the one non-flanking leg is `−x` — a rim edge of the hole.
#[test]
fn edge_bevel_boundary_valence_four_terminal_open_fan_is_filled_from_the_arc() {
    let (mut mesh, verts) = quad_grid_5x5_holed();
    let before = counts(&mesh);
    let rim = verts[2][2];
    assert_eq!(
        mesh.iter_disk_edges(rim).count(),
        4,
        "the hole rim vert keeps its four edges",
    );
    assert_eq!(
        mesh.vert_face_keys(rim).len(),
        3,
        "but only three faces — an open disk",
    );

    let beveled = mesh
        .query_edge_exists(rim, verts[3][2])
        .expect("the +x edge off the hole rim");
    let result = mesh.bevel(
        &[beveled.into()],
        BevelOptions {
            offset: 0.25,
            affect: BevelAffect::Edges,
            ..Default::default()
        },
    );
    mesh.validate_topology();

    let (verts_after, _, faces_after) = counts(&mesh);
    assert_eq!(
        (verts_after, faces_after),
        (before.0 + 4, before.2 + 3),
        "both endpoints are valence-4 terminals: 2 verts each, one fill each, one strip",
    );
    assert!(
        vert_near(&mesh, Vec3::new(2.0, 2.0, 0.0), 1.0e-4).is_none(),
        "the rim terminal vertex is consumed by the close",
    );

    let tol = 1.0e-5;
    let start = vert_near(&mesh, Vec3::new(2.0, 2.25, 0.0), tol).expect("+y strip vert");
    let end = vert_near(&mesh, Vec3::new(2.0, 1.75, 0.0), tol).expect("−y strip vert");
    let leg = vert_near(&mesh, Vec3::new(1.75, 2.0, 0.0), tol).expect("−x leg vert");

    let fills = terminal_fill_faces(&mesh, &result.verts, leg);
    assert_eq!(fills.len(), 1, "three boundverts ⇒ one fill face");
    assert!(
        same_cyclic_order(&fills[0], &[start, end, leg]),
        "the fill is the arc's ring `[start, end, leg₁]`",
    );
    assert!(
        winding_conflicts(&mesh).is_empty(),
        "no shared edge is traversed the same way by both its faces",
    );

    // The fan's gap is the sector between the `−y` and `−x` legs (the missing quad), so
    // the ring edge `end → leg₁` spans it — and it comes out a **boundary** edge: the
    // fill adds no surface across the hole beyond its own single face. The ring's other
    // leg edge, `leg₁ → start`, crosses a surviving sector and is shared with the face
    // that absorbed both of its ends.
    let gap_edge = mesh
        .query_edge_exists(end, leg)
        .expect("the ring edge spanning the open fan's gap");
    assert_eq!(
        mesh.edge_radial_iter(gap_edge).count(),
        1,
        "the gap-spanning ring edge is a boundary edge of the fill",
    );
    let absorbed_edge = mesh
        .query_edge_exists(leg, start)
        .expect("the ring edge across the surviving sector");
    assert_eq!(mesh.edge_radial_iter(absorbed_edge).count(), 2);
}

/// The open fan's ring is walked off the corner ring at **every** `segments`, and at
/// `segments = 1` a valence-6 boundary terminal is the shape where it is emitted **whole**
/// (five boundverts, §8.3.0.2's n-gon rule) rather than fanned — so a `middle` in the
/// wrong order comes out as a self-crossing **bow-tie pentagon**, not merely a wrongly
/// shaped triangle. Every rotation of the build order is exercised: that is what moves
/// the per-vertex edge cycle's insertion order, the only other order the legs could be
/// read in.
#[test]
fn edge_bevel_open_fan_valence_six_single_segment_fill_is_one_whole_pentagon() {
    const SPOKES: usize = 6;
    const BEVELED: usize = 2;

    for first_face in 0..SPOKES - 1 {
        let (mut mesh, centre, rim) = open_planar_fan(SPOKES, first_face);
        let spoke_of = |mesh: &LoopMesh, k: usize| {
            mesh.query_edge_exists(centre, rim[k]).expect("a fan spoke")
        };
        let beveled = spoke_of(&mesh, BEVELED);

        // The rotational ring: `start` on `E`'s disk-preceding leg, `end` on its
        // following one, then on past `end` — over the arc's open gap and back to
        // `start`.
        let corners = mesh.vertex_corner_ring(centre);
        let mut disk: Vec<EdgeKey> = corners.iter().map(|c| c.incoming_edge).collect();
        disk.push(corners.last().expect("a corner").outgoing_edge);
        let seat = disk
            .iter()
            .position(|&e| e == beveled)
            .expect("E on the arc");
        let leg_index = |edge: EdgeKey| {
            (0..SPOKES)
                .find(|&k| spoke_of(&mesh, k) == edge)
                .expect("a spoke")
        };
        let ring_spokes: Vec<usize> = std::iter::once(seat + SPOKES - 1)
            .chain(seat + 1..seat + SPOKES - 1)
            .map(|step| leg_index(disk[step % SPOKES]))
            .collect();

        let result = mesh.bevel(
            &[beveled.into()],
            BevelOptions {
                offset: 0.25,
                affect: BevelAffect::Edges,
                ..Default::default()
            },
        );
        mesh.validate_topology();

        let on_spoke = |mesh: &LoopMesh, k: usize| -> VertexKey {
            let angle = std::f32::consts::PI * (k as f32) / ((SPOKES - 1) as f32);
            let direction = Vec3::new(angle.cos(), angle.sin(), 0.0);
            mesh.vert_keys()
                .into_iter()
                .find(|&v| {
                    let position = mesh.vertex_position(v);
                    let length = position.length();
                    length > 1.0e-4
                        && length < 0.9
                        && (position / length - direction).length() < 1.0e-3
                })
                .expect("the fill's boundvert on this spoke")
        };
        let ring: Vec<VertexKey> = ring_spokes.iter().map(|&k| on_spoke(&mesh, k)).collect();

        let fills = terminal_fill_faces(&mesh, &result.verts, ring[2]);
        assert_eq!(fills.len(), 1, "five boundverts ⇒ one whole n-gon");
        let fill = &fills[0];
        assert_eq!(
            fill.len(),
            5,
            "a pentagon: no interior rung at `segments = 1`"
        );
        assert!(
            same_cyclic_order(fill, &ring),
            "the face walks `start → end → leg₁ → leg₂ → leg₃` in arc order — no bow-tie",
        );

        // The one ring edge that spans the arc's gap joins the boundverts of the two
        // extreme spokes, and is a **boundary** edge: the fill lays no surface over the
        // hole beyond its own single face.
        let gap_edge = mesh
            .query_edge_exists(on_spoke(&mesh, 0), on_spoke(&mesh, SPOKES - 1))
            .expect("the ring edge spanning the open fan's gap");
        assert_eq!(
            mesh.edge_radial_iter(gap_edge).count(),
            1,
            "the gap-spanning ring edge is a boundary edge of the fill",
        );
        assert!(
            winding_conflicts(&mesh).is_empty(),
            "no shared edge is traversed the same way by both its faces",
        );
    }
}

/// The smallest mesh that carries the same corner — a `2 × 2` quad grid with one corner
/// quad (and its outer vertex) left out: 8 verts, 10 edges, 3 quads, and a centre vertex
/// of valence 4 whose fan is open. Returns the centre `(1,1)` and the `(0,1)` end of its
/// `−x` spoke, one of the centre's two manifold ones.
fn three_quad_open_fan_corner() -> (LoopMesh, VertexKey, VertexKey) {
    let mut mesh = LoopMesh::new();
    let coords = [
        (0, 0),
        (1, 0),
        (2, 0),
        (0, 1),
        (1, 1),
        (2, 1),
        (0, 2),
        (1, 2),
    ];
    let verts: Vec<VertexKey> = coords
        .iter()
        .map(|&(i, j)| mesh.create_vertex(Vec3::new(i as f32, j as f32, 0.0)))
        .collect();
    let at = |i: i32, j: i32| {
        verts[coords
            .iter()
            .position(|&c| c == (i, j))
            .expect("a grid vertex the fixture keeps")]
    };
    for (i, j) in [(0, 0), (1, 0), (0, 1)] {
        mesh.face_create_verts(
            &[at(i, j), at(i + 1, j), at(i + 1, j + 1), at(i, j + 1)],
            FaceCreateFlags::NONE,
        );
    }
    (mesh, at(1, 1), at(0, 1))
}

/// The whole shape of the open-fan terminal close, on the smallest fixture that has one:
/// `8 / 10 / 3` → `11 / 15 / 5` with face sizes `[3, 4, 4, 4, 5]`. The valence-4 centre
/// contributes two strip verts and one leg vert (minus the consumed centre) and one
/// **triangle** fill (§8.3.0.2 at three boundverts, `segments = 1`); the valence-3 far
/// end contributes its two strip verts and **no** fill (§8.3.0.1's two-boundvert ring);
/// the strip is one quad; the face between `end`'s leg and the leg vert grows by one
/// side to a pentagon (§7.4's absorb); the two faces flanking `E` swap each corner
/// 1-for-1 and stay quads.
#[test]
fn edge_bevel_open_fan_terminal_close_is_a_triangle_fill_and_a_grown_pentagon() {
    let (mut mesh, centre, spoke_end) = three_quad_open_fan_corner();
    assert_eq!(counts(&mesh), (8, 10, 3));

    let beveled = mesh
        .query_edge_exists(centre, spoke_end)
        .expect("the centre's −x spoke");
    assert_eq!(
        mesh.edge_radial_iter(beveled).count(),
        2,
        "the beveled spoke is manifold; the fan around it is not closed",
    );

    mesh.bevel(
        &[beveled.into()],
        BevelOptions {
            offset: 0.25,
            affect: BevelAffect::Edges,
            ..Default::default()
        },
    );
    mesh.validate_topology();

    assert_eq!(counts(&mesh), (11, 15, 5));
    let mut sizes: Vec<usize> = mesh
        .face_keys()
        .into_iter()
        .map(|face| mesh.face_vert_keys(face).len())
        .collect();
    sizes.sort_unstable();
    assert_eq!(sizes, vec![3, 4, 4, 4, 5]);
    assert!(
        vert_near(&mesh, Vec3::new(1.0, 1.0, 0.0), 1.0e-4).is_none(),
        "the open-fan terminal vertex is consumed",
    );
    assert!(winding_conflicts(&mesh).is_empty());
}

/// Bevel spoke 0 of a closed planar `spokes`-fan and hand back the centre terminal's
/// fill-ring **boundverts in disk order** — `start` (on `E`'s disk-preceding leg),
/// `end` (on its disk-following leg), then `leg₁ … leg_(n−3)` walking on past `end`
/// (BEVEL.md §8.3.0.2's leg-numbering convention) — together with the verts the
/// operator minted. Every boundvert of a *planar* terminal lands on its own leg's ray
/// (the §7.5.1 leg meet on the flanking legs, the plain-offset slide on the rest), so
/// each is found by the spoke it sits on.
fn bevel_planar_fan_terminal(
    spokes: usize,
    segments: u32,
) -> (LoopMesh, Vec<VertexKey>, Vec<VertexKey>) {
    let (mut mesh, centre, rim) = planar_fan(spokes);
    let spoke_of =
        |mesh: &LoopMesh, k: usize| mesh.query_edge_exists(centre, rim[k]).expect("a fan spoke");
    let beveled = spoke_of(&mesh, 0);

    let corners = mesh.vertex_corner_ring(centre);
    let disk: Vec<EdgeKey> = corners.iter().map(|c| c.incoming_edge).collect();
    assert_eq!(disk.len(), spokes, "a closed disk");
    let seat = disk
        .iter()
        .position(|&e| e == beveled)
        .expect("E on the disk");
    let leg_index = |edge: EdgeKey| {
        (0..spokes)
            .find(|&k| spoke_of(&mesh, k) == edge)
            .expect("a spoke")
    };
    let mut ring_spokes = vec![
        leg_index(disk[(seat + spokes - 1) % spokes]),
        leg_index(disk[(seat + 1) % spokes]),
    ];
    for step in 2..spokes - 1 {
        ring_spokes.push(leg_index(disk[(seat + step) % spokes]));
    }

    let result = mesh.bevel(
        &[beveled.into()],
        BevelOptions {
            offset: 0.25,
            segments,
            affect: BevelAffect::Edges,
            ..Default::default()
        },
    );
    mesh.validate_topology();

    let on_spoke = |mesh: &LoopMesh, k: usize| -> VertexKey {
        let angle = std::f32::consts::TAU * (k as f32) / (spokes as f32);
        let direction = Vec3::new(angle.cos(), angle.sin(), 0.0);
        mesh.vert_keys()
            .into_iter()
            .find(|&v| {
                let position = mesh.vertex_position(v);
                let length = position.length();
                length > 1.0e-4 && length < 0.9 && (position / length - direction).length() < 1.0e-3
            })
            .expect("the fill's boundvert on this spoke")
    };
    let ring: Vec<VertexKey> = ring_spokes.iter().map(|&k| on_spoke(&mesh, k)).collect();
    (mesh, ring, result.verts)
}

/// Does `cycle` visit `ring`'s verts in `ring`'s own cyclic order (either sense)? A
/// bow-tie n-gon has the same vert *set* and fails this.
fn same_cyclic_order(cycle: &[VertexKey], ring: &[VertexKey]) -> bool {
    if cycle.len() != ring.len() {
        return false;
    }
    let count = ring.len();
    let Some(offset) = cycle.iter().position(|vert| *vert == ring[0]) else {
        return false;
    };
    let forward = (0..count).all(|k| cycle[(offset + k) % count] == ring[k]);
    let backward = (0..count).all(|k| cycle[(offset + count - k) % count] == ring[k]);
    forward || backward
}

/// BEVEL.md §8.3.0.2's boundvert-count rule at **three** boundverts (valence 4, one leg
/// vert) and `segments = 2`: the ring `[start, peak, end, leg]` is a 4-gon and it *is*
/// fanned — from the leg vert, into `segments = 2` triangles whose one diagonal is
/// `leg–peak`. This is the row the whole-n-gon rule must **not** swallow: the same
/// 4-gon ring at a valence-5 / `segments = 1` terminal is emitted whole (below), so ring
/// size cannot be the discriminant — the boundvert count is.
#[test]
fn edge_bevel_valence_four_terminal_two_segment_fill_is_two_triangles() {
    let (mesh, ring, minted) = bevel_planar_fan_terminal(4, 2);
    let (start, end, leg) = (ring[0], ring[1], ring[2]);

    let fills = terminal_fill_faces(&mesh, &minted, leg);
    assert_eq!(
        fills.len(),
        2,
        "three boundverts ⇒ `segments` = 2 triangles"
    );
    for fill in &fills {
        assert_eq!(fill.len(), 3, "a fanned fill is triangles");
        assert!(
            fill.contains(&leg),
            "every fan triangle carries the leg vert"
        );
    }
    let peak = *fills
        .iter()
        .flatten()
        .find(|vert| ![start, end, leg].contains(vert))
        .expect("the chain's interior peak rung");
    assert!(
        mesh.query_edge_exists(leg, peak).is_some(),
        "the fan's one diagonal runs `leg–peak`",
    );
    assert!(
        winding_conflicts(&mesh).is_empty(),
        "no shared edge is traversed the same way by both its faces",
    );
}

/// BEVEL.md §8.3.0.2 at **four** boundverts (valence 5, two leg verts): the ring
/// `[start, end, leg₁, leg₂]` is emitted **whole** — one quad, no diagonal — where a
/// fan would have cut it into two triangles on `leg–start` (or `end–leg₂`). The face's
/// loop order is the ring's own disk order, so `leg₁` neighbours `end` and `leg₂`
/// neighbours `start`: number the legs the other way while emitting in the listed order
/// and the quad is a bow-tie.
#[test]
fn edge_bevel_valence_five_terminal_fill_is_one_whole_quad() {
    let (mesh, ring, minted) = bevel_planar_fan_terminal(5, 1);
    let (start, end, leg_one, leg_two) = (ring[0], ring[1], ring[2], ring[3]);

    let fills = terminal_fill_faces(&mesh, &minted, leg_one);
    assert_eq!(fills.len(), 1, "four boundverts ⇒ one whole n-gon");
    let fill = &fills[0];
    assert_eq!(fill.len(), 4, "and it is a quad, not two triangles");
    assert!(
        same_cyclic_order(fill, &ring),
        "the ring listing is the face's loop order — no bow-tie",
    );
    assert!(
        mesh.query_edge_exists(leg_one, start).is_none()
            && mesh.query_edge_exists(end, leg_two).is_none(),
        "neither diagonal is drawn",
    );
    assert!(
        winding_conflicts(&mesh).is_empty(),
        "the whole n-gon winds opposite every neighbour it shares an edge with",
    );
    for edge in mesh.edge_keys() {
        assert!(mesh.edge_radial_iter(edge).count() <= 2);
    }
}

/// The same four-boundvert rule on the `segments ≥ 2` builder, which is a **second**
/// close: at valence 5 / `segments = 2` the ring is `[start, peak, end, leg₁, leg₂]` and
/// the fill is one **pentagon**. It also pins that builder's leg ordering as
/// *rotational* — listing the legs in the disk linked-list order instead can put `leg₂`
/// between `end` and `leg₁` and emit a self-crossing n-gon.
#[test]
fn edge_bevel_valence_five_terminal_two_segment_fill_is_one_whole_polygon() {
    let (mesh, ring, minted) = bevel_planar_fan_terminal(5, 2);
    let (start, end, leg_one, leg_two) = (ring[0], ring[1], ring[2], ring[3]);

    let fills = terminal_fill_faces(&mesh, &minted, leg_one);
    assert_eq!(fills.len(), 1, "four boundverts ⇒ one whole n-gon");
    let fill = &fills[0];
    assert_eq!(
        fill.len(),
        5,
        "the ring is `(segments + 1) + (n − 3)` = 5 verts wide",
    );
    let peak = *fill
        .iter()
        .find(|vert| ![start, end, leg_one, leg_two].contains(vert))
        .expect("the chain's interior peak rung");
    assert!(
        same_cyclic_order(fill, &[start, peak, end, leg_one, leg_two]),
        "the face walks `start → peak → end → leg₁ → leg₂` — no bow-tie",
    );
    assert!(
        mesh.query_edge_exists(leg_one, peak).is_none()
            && mesh.query_edge_exists(leg_two, peak).is_none(),
        "no diagonal is drawn to the peak",
    );
    assert!(
        winding_conflicts(&mesh).is_empty(),
        "no shared edge is traversed the same way by both its faces",
    );
}

/// BEVEL.md §8.3.0.2 at **five** boundverts (valence 6, three leg verts): one pentagon,
/// no diagonals — where an unconditional fan would produce three triangles.
#[test]
fn edge_bevel_valence_six_terminal_fill_is_one_whole_pentagon() {
    let (mesh, ring, minted) = bevel_planar_fan_terminal(6, 1);

    let fills = terminal_fill_faces(&mesh, &minted, ring[2]);
    assert_eq!(fills.len(), 1, "five boundverts ⇒ one whole n-gon");
    let fill = &fills[0];
    assert_eq!(fill.len(), 5, "and it is a pentagon, not three triangles");
    assert!(
        same_cyclic_order(fill, &ring),
        "the face walks `start → end → leg₁ → leg₂ → leg₃` — no bow-tie",
    );
    for (a, b) in [(ring[0], ring[2]), (ring[0], ring[3]), (ring[1], ring[3])] {
        assert!(
            mesh.query_edge_exists(a, b).is_none(),
            "no diagonal is drawn across the pentagon",
        );
    }
    assert!(
        winding_conflicts(&mesh).is_empty(),
        "no shared edge is traversed the same way by both its faces",
    );
}

/// BEVEL.md §8.3.0.2's leg-numbering convention is a statement about the terminal's
/// **rotational** disk, and it binds on an *open* fan exactly as on a closed one: a
/// boundary terminal of valence 6 still mints three leg verts and still emits one whole
/// n-gon over them. Only the rotational walk (the corner ring) can order them — the
/// per-vertex edge cycle is linked in insertion order, and reading the legs off it emits
/// the ring with `leg₃` between `end` and `leg₁`: a self-intersecting **bow-tie**
/// hexagon, right verts and right face count, silently wrong face. Every rotation of the
/// build order is exercised, because that is exactly what moves the insertion order.
#[test]
fn edge_bevel_open_fan_valence_six_terminal_legs_are_in_rotational_order() {
    const SPOKES: usize = 6;
    const BEVELED: usize = 2;

    for first_face in 0..SPOKES - 1 {
        let (mut mesh, centre, rim) = open_planar_fan(SPOKES, first_face);
        let spoke_of = |mesh: &LoopMesh, k: usize| {
            mesh.query_edge_exists(centre, rim[k]).expect("a fan spoke")
        };
        let beveled = spoke_of(&mesh, BEVELED);
        assert_eq!(mesh.iter_disk_edges(centre).count(), SPOKES);
        assert_eq!(
            mesh.vert_face_keys(centre).len(),
            SPOKES - 1,
            "one face short of a cycle — the disk is an arc",
        );

        // The rotational ring, off the corner ring: `start` on `E`'s disk-preceding leg,
        // `end` on its following one, then the legs walked on past `end` — across the
        // arc's open gap and back up to `start`.
        let corners = mesh.vertex_corner_ring(centre);
        let mut disk: Vec<EdgeKey> = corners.iter().map(|c| c.incoming_edge).collect();
        disk.push(corners.last().expect("a corner").outgoing_edge);
        assert_eq!(disk.len(), SPOKES, "the whole arc, both open ends");
        let seat = disk
            .iter()
            .position(|&e| e == beveled)
            .expect("E on the arc");
        let leg_index = |edge: EdgeKey| {
            (0..SPOKES)
                .find(|&k| spoke_of(&mesh, k) == edge)
                .expect("a spoke")
        };
        let mut ring_spokes = vec![
            leg_index(disk[(seat + SPOKES - 1) % SPOKES]),
            leg_index(disk[(seat + 1) % SPOKES]),
        ];
        for step in 2..SPOKES - 1 {
            ring_spokes.push(leg_index(disk[(seat + step) % SPOKES]));
        }

        let result = mesh.bevel(
            &[beveled.into()],
            BevelOptions {
                offset: 0.25,
                segments: 2,
                affect: BevelAffect::Edges,
                ..Default::default()
            },
        );
        mesh.validate_topology();

        let on_spoke = |mesh: &LoopMesh, k: usize| -> VertexKey {
            let angle = std::f32::consts::PI * (k as f32) / ((SPOKES - 1) as f32);
            let direction = Vec3::new(angle.cos(), angle.sin(), 0.0);
            mesh.vert_keys()
                .into_iter()
                .find(|&v| {
                    let position = mesh.vertex_position(v);
                    let length = position.length();
                    length > 1.0e-4
                        && length < 0.9
                        && (position / length - direction).length() < 1.0e-3
                })
                .expect("the fill's boundvert on this spoke")
        };
        let ring: Vec<VertexKey> = ring_spokes.iter().map(|&k| on_spoke(&mesh, k)).collect();
        let (start, end) = (ring[0], ring[1]);

        let fills = terminal_fill_faces(&mesh, &result.verts, ring[2]);
        assert_eq!(fills.len(), 1, "five boundverts ⇒ one whole n-gon");
        let fill = &fills[0];
        assert_eq!(
            fill.len(),
            6,
            "the ring is `(segments + 1) + (n − 3)` = 6 verts wide",
        );
        let peak = *fill
            .iter()
            .find(|vert| !ring.contains(vert))
            .expect("the chain's interior peak rung");
        let expected = [start, peak, end, ring[2], ring[3], ring[4]];
        assert!(
            same_cyclic_order(fill, &expected),
            "the face walks `start → peak → end → leg₁ → leg₂ → leg₃` — no bow-tie",
        );
        assert!(
            winding_conflicts(&mesh).is_empty(),
            "no shared edge is traversed the same way by both its faces",
        );
    }
}

// ---- Custom profile (BEVEL.md §8.6.3 / §8.6.3.1 / §8.3.0.1 / §8.3.0.2) ----
