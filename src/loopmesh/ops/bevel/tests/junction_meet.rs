use super::*;

/// An irregular apex junction (BEVEL.md §8.8.5): apex `v0` fanned by four
/// quads to four spokes `v1..v4` at unequal directions and heights, so every
/// spoke is a silhouette edge (its two flanking quads are non-coplanar). The
/// four quads each span the apex plus two consecutive spokes plus one outer
/// rim vert. Beveling spokes {0,1} and {0,3} leaves spoke {0,2} as the lone
/// silhouette leg in the sector between them — the §8.8.2 firing sector.
/// Returns (mesh, apex, [v1,v2,v3,v4], beveled edges {0,1} and {0,3}).
fn irregular_apex_junction() -> (LoopMesh, VertexKey, [VertexKey; 4], [EdgeKey; 2]) {
    let mut mesh = LoopMesh::new();
    let apex = mesh.create_vertex(Vec3::new(0.0, 0.0, 0.0));
    // Spokes at unequal xy-directions and z-heights so the surface bends
    // across each (silhouette) and the {0,1}/{0,3} sector is irregular.
    let s1 = mesh.create_vertex(Vec3::new(1.0, 0.0, -0.3));
    let s2 = mesh.create_vertex(Vec3::new(0.35, 0.95, -0.7));
    let s3 = mesh.create_vertex(Vec3::new(-1.0, 0.15, -0.25));
    let s4 = mesh.create_vertex(Vec3::new(-0.25, -1.0, -0.55));
    let spokes = [s1, s2, s3, s4];
    // Outer rim verts, one between each consecutive spoke pair, pushed out and
    // down so the quads are non-planar petals.
    let outers: Vec<VertexKey> = (0..4)
        .map(|i| {
            let a = mesh.vertex_position(spokes[i]);
            let b = mesh.vertex_position(spokes[(i + 1) % 4]);
            let mid = (a + b) * 0.5;
            mesh.create_vertex(mid * 1.8 + Vec3::new(0.0, 0.0, -0.4))
        })
        .collect();

    let edge = |mesh: &mut LoopMesh, a: VertexKey, b: VertexKey| {
        mesh.create_edge([a, b], EdgeCreateFlags::NONE)
    };
    for i in 0..4 {
        let sa = spokes[i];
        let sb = spokes[(i + 1) % 4];
        let outer = outers[i];
        let e_apex_a = mesh
            .query_edge_exists(apex, sa)
            .unwrap_or_else(|| edge(&mut mesh, apex, sa));
        let e_a_outer = edge(&mut mesh, sa, outer);
        let e_outer_b = edge(&mut mesh, outer, sb);
        let e_b_apex = mesh
            .query_edge_exists(apex, sb)
            .unwrap_or_else(|| edge(&mut mesh, apex, sb));
        mesh.face_create(
            &[apex, sa, outer, sb],
            &[e_apex_a, e_a_outer, e_outer_b, e_b_apex],
            FaceCreateFlags::NONE,
        );
    }

    let e01 = mesh.query_edge_exists(apex, s1).expect("spoke {0,1}");
    let e03 = mesh.query_edge_exists(apex, s3).expect("spoke {0,3}");
    (mesh, apex, spokes, [e01, e03])
}

/// The boundvert of the firing sector (the new vert nearest spoke {0,2}'s
/// line through the apex) — the one `loop_slide` slides onto leg {0,2}.
fn firing_boundvert(mesh: &LoopMesh, result: &BevelResult, apex_pos: Vec3, leg_dir: Vec3) -> Vec3 {
    result
        .verts
        .iter()
        .map(|&v| mesh.vertex_position(v))
        .min_by(|&a, &b| {
            let da = ((a - apex_pos) - leg_dir * (a - apex_pos).dot(leg_dir)).length();
            let db = ((b - apex_pos) - leg_dir * (b - apex_pos).dot(leg_dir)).length();
            da.partial_cmp(&db).unwrap()
        })
        .expect("the junction produced boundverts")
}

/// Like `firing_boundvert`, but over a bare position list (the new verts already
/// collected): the vert with the smallest perpendicular distance to `leg_dir`'s
/// line through `apex_pos` — the boundvert that slid onto that leg.
fn firing_boundvert_from(positions: &[Vec3], apex_pos: Vec3, leg_dir: Vec3) -> Vec3 {
    let perp = |p: Vec3| {
        let rel = p - apex_pos;
        (rel - leg_dir * rel.dot(leg_dir)).length()
    };
    *positions
        .iter()
        .min_by(|&&a, &&b| perp(a).partial_cmp(&perp(b)).unwrap())
        .expect("the junction produced boundverts")
}

/// BEVEL.md §8.8: `loop_slide = true` slides the firing sector's boundvert
/// onto the silhouette leg's line, where `loop_slide = false` leaves it at the
/// §7.6.2 two-strip meet-point. At an irregular junction the two placements
/// differ (§8.8.5) while the topology is byte-identical — only one vertex
/// moves.
#[test]
fn edge_bevel_loop_slide_diverges_at_irregular_junction() {
    let offset = 0.2;

    let (mut mesh_off, apex_off, _spokes_off, [e01_off, e03_off]) = irregular_apex_junction();
    let off = mesh_off.bevel(
        &[e01_off.into(), e03_off.into()],
        BevelOptions {
            offset,
            affect: BevelAffect::Edges,
            loop_slide: false,
            ..Default::default()
        },
    );
    let counts_off = counts(&mesh_off);
    mesh_off.validate_topology();

    let (mut mesh_on, apex_on, spokes_on, [e01_on, e03_on]) = irregular_apex_junction();
    let apex_pos = mesh_on.vertex_position(apex_on);
    let leg_dir = (mesh_on.vertex_position(spokes_on[1]) - apex_pos).normalize();
    let on = mesh_on.bevel(
        &[e01_on.into(), e03_on.into()],
        BevelOptions {
            offset,
            affect: BevelAffect::Edges,
            loop_slide: true,
            ..Default::default()
        },
    );
    mesh_on.validate_topology();

    // §8.8: pure placement perturbation — no topology change.
    assert_eq!(
        counts(&mesh_on),
        counts_off,
        "loop_slide changes no topology (same V/E/F)",
    );

    // §8.8.5: the firing boundvert moves — on ≠ off at an irregular junction.
    let b_off = firing_boundvert(&mesh_off, &off, apex_pos, leg_dir);
    let b_on = firing_boundvert(&mesh_on, &on, apex_pos, leg_dir);
    assert!(
        (b_on - b_off).length() > 1.0e-3,
        "loop_slide on must move the firing boundvert (irregular junction), \
         got on={b_on:?} off={b_off:?}",
    );

    // §8.8.3: the slid boundvert lies on the leg's line through the apex, so
    // its component perpendicular to the leg direction vanishes.
    let perpendicular = ((b_on - apex_pos) - leg_dir * (b_on - apex_pos).dot(leg_dir)).length();
    assert!(
        perpendicular < 1.0e-4,
        "slid boundvert sits on the silhouette leg's line, perp={perpendicular}",
    );

    // The off boundvert (the §7.6.2 meet) floats off that line.
    let perpendicular_off =
        ((b_off - apex_pos) - leg_dir * (b_off - apex_pos).dot(leg_dir)).length();
    assert!(
        perpendicular_off > 1.0e-4,
        "meet-point boundvert floats off the leg's line, perp={perpendicular_off}",
    );
}

/// BEVEL.md §8.8.2.1 (firing-frame): the two beveled edges {0,1}/{0,3} split the
/// apex into **two** sectors, each holding a single silhouette leg ({0,2} and
/// {0,4}). With the firing predicate taken in the genuine *vertex-normal* frame
/// (`n_v`), **both** sectors qualify and slide their boundvert onto their leg —
/// the under-fire (a sector-face frame slid only the one whose flanking faces
/// aligned with that frame, flipping the reflex sign on the off-plane sector) is
/// gone. The non-planar legs make each slid boundvert land off the z = 0 plane,
/// so "on the leg" is the vanishing perpendicular-to-leg distance, not z ≈ 0.
#[test]
fn edge_bevel_loop_slide_fires_every_qualifying_sector() {
    let offset = 0.2;
    let (mut mesh, apex, spokes, [e01, e03]) = irregular_apex_junction();
    let apex_pos = mesh.vertex_position(apex);
    let leg2_dir = (mesh.vertex_position(spokes[1]) - apex_pos).normalize();
    let leg4_dir = (mesh.vertex_position(spokes[3]) - apex_pos).normalize();

    let result = mesh.bevel(
        &[e01.into(), e03.into()],
        BevelOptions {
            offset,
            affect: BevelAffect::Edges,
            loop_slide: true,
            ..Default::default()
        },
    );
    mesh.validate_topology();

    let b2 = firing_boundvert(&mesh, &result, apex_pos, leg2_dir);
    let b4 = firing_boundvert(&mesh, &result, apex_pos, leg4_dir);
    let perp = |p: Vec3, leg: Vec3| ((p - apex_pos) - leg * (p - apex_pos).dot(leg)).length();

    // Both sectors' boundverts sit on their own silhouette leg's line (§8.8.3),
    // and they are distinct verts (each slid onto a different leg).
    assert!(
        perp(b2, leg2_dir) < 1.0e-4,
        "sector {{0,2}} slid onto its leg, perp={}",
        perp(b2, leg2_dir),
    );
    assert!(
        perp(b4, leg4_dir) < 1.0e-4,
        "sector {{0,4}} also slid onto its leg, perp={}",
        perp(b4, leg4_dir),
    );
    assert!(
        (b2 - b4).length() > 1.0e-3,
        "the two firing boundverts are distinct, b2={b2:?} b4={b4:?}",
    );
}

/// BEVEL.md §8.8.4 / §8.8.5: at an **irregular** firing junction the flank-sine
/// ratio ≠ 1 redistributes the two flanking edges' §7.6.4 offsets, and the ripple
/// runs **along each flanking edge to its far end** — the terminal strip verts.
/// On `irregular_apex_junction` the two beveled edges {0,1}/{0,3} run from the
/// firing apex out to terminal endpoints (spokes s1/s3), so all four terminal
/// strip verts shift by a small width-pass ripple between `loop_slide` on and off.
/// This pins the §8.8.5 "missing-ripple" defect: leaving the termini static (their
/// spec-offset placement) is the bug, not a preserved invariant. The drift stays a
/// few thousandths — a reconciled-offset ripple, not a recipe change — so it is
/// bounded well below a gross routing-bug relocation.
#[test]
fn edge_bevel_loop_slide_ripples_terminal_strip_verts_at_irregular_junction() {
    let offset = 0.3;

    let build = |loop_slide: bool| {
        let (mut mesh, apex, _spokes, [e01, e03]) = irregular_apex_junction();
        let apex_pos = mesh.vertex_position(apex);
        let result = mesh.bevel(
            &[e01.into(), e03.into()],
            BevelOptions {
                offset,
                affect: BevelAffect::Edges,
                loop_slide,
                ..Default::default()
            },
        );
        mesh.validate_topology();
        let positions: Vec<Vec3> = result
            .verts
            .iter()
            .map(|&v| mesh.vertex_position(v))
            .collect();
        (apex_pos, positions)
    };

    let (apex_pos, off_positions) = build(false);
    let (_, on_positions) = build(true);

    // Two firing apex boundverts (near the apex) + four terminal strip verts (out
    // by the far endpoints). `result.verts` is built in the same order both runs,
    // so positions correspond index-for-index.
    assert_eq!(on_positions.len(), off_positions.len(), "same vert count");
    assert_eq!(
        on_positions.len(),
        6,
        "two apex boundverts + four terminal strip verts"
    );

    let mut apex_drift_max: f32 = 0.0;
    let mut terminal_count = 0;
    for (on, off) in on_positions.iter().zip(off_positions.iter()) {
        let drift = (*on - *off).length();
        // Apex boundverts slide onto the leg near the apex; terminal strip verts
        // sit out near the far endpoints (distance ≈ 1 from the apex).
        if (*off - apex_pos).length() > 0.5 {
            terminal_count += 1;
            assert!(
                drift > 1.0e-4,
                "terminal strip vert must ripple at an irregular firing junction \
                 (not stay at its spec-offset placement), drift={drift}",
            );
            // With the §7.6.4 spec-pull scale `t*` (not the meet-at-B `t_B`), the
            // reconciled offset no longer overshoots, so the ripple is bounded well
            // below the apex slide and a routing-bug relocation.
            assert!(
                drift < 3.0e-2,
                "terminal ripple stays a small width-pass shift, not a routing-bug \
                 relocation, drift={drift}",
            );
        } else {
            apex_drift_max = apex_drift_max.max(drift);
        }
    }
    assert_eq!(terminal_count, 4, "four terminal strip verts ripple");
    // The case-A apex slide is the larger, direct placement change (§8.8.5).
    assert!(
        apex_drift_max > 1.0e-3,
        "the firing apex boundverts slide directly (case A), drift={apex_drift_max}",
    );
}

/// Like `irregular_apex_junction`, but the firing sector's silhouette leg
/// (spoke {0,2}) lies exactly in the `z = 0` plane, so "on the leg's line"
/// means `z ≈ 0` directly. The two bounding beveled edges {0,1}/{0,3} rise out
/// of that plane at *unequal* angles (an irregular junction, §8.8.5), so the
/// §7.6.2 meet floats off `z = 0` while the §8.8.3 slide pins the boundvert back
/// onto it. Returns (mesh, apex, [v1..v4], beveled edges {0,1} and {0,3}).
fn irregular_apex_junction_planar_leg() -> (LoopMesh, VertexKey, [VertexKey; 4], [EdgeKey; 2]) {
    let mut mesh = LoopMesh::new();
    let apex = mesh.create_vertex(Vec3::new(0.0, 0.0, 0.0));
    // s2 — the silhouette leg — lies in z = 0. s1 and s3 (the beveled flanks)
    // lift out of the plane at unequal angles so the junction is irregular.
    let s1 = mesh.create_vertex(Vec3::new(1.0, -0.2, 0.3));
    let s2 = mesh.create_vertex(Vec3::new(-0.15, -1.0, 0.0));
    let s3 = mesh.create_vertex(Vec3::new(-0.8, -0.5, 0.9));
    let s4 = mesh.create_vertex(Vec3::new(0.1, 1.0, 0.5));
    let spokes = [s1, s2, s3, s4];
    let outers: Vec<VertexKey> = (0..4)
        .map(|i| {
            let a = mesh.vertex_position(spokes[i]);
            let b = mesh.vertex_position(spokes[(i + 1) % 4]);
            let mid = (a + b) * 0.5;
            mesh.create_vertex(mid * 1.8 + Vec3::new(0.0, 0.0, -0.4))
        })
        .collect();

    let edge = |mesh: &mut LoopMesh, a: VertexKey, b: VertexKey| {
        mesh.create_edge([a, b], EdgeCreateFlags::NONE)
    };
    for i in 0..4 {
        let sa = spokes[i];
        let sb = spokes[(i + 1) % 4];
        let outer = outers[i];
        let e_apex_a = mesh
            .query_edge_exists(apex, sa)
            .unwrap_or_else(|| edge(&mut mesh, apex, sa));
        let e_a_outer = edge(&mut mesh, sa, outer);
        let e_outer_b = edge(&mut mesh, outer, sb);
        let e_b_apex = mesh
            .query_edge_exists(apex, sb)
            .unwrap_or_else(|| edge(&mut mesh, apex, sb));
        mesh.face_create(
            &[apex, sa, outer, sb],
            &[e_apex_a, e_a_outer, e_outer_b, e_b_apex],
            FaceCreateFlags::NONE,
        );
    }

    let e01 = mesh.query_edge_exists(apex, s1).expect("spoke {0,1}");
    let e03 = mesh.query_edge_exists(apex, s3).expect("spoke {0,3}");
    (mesh, apex, spokes, [e01, e03])
}

/// Like `irregular_apex_junction_planar_leg`, but the beveled spokes {0,1}/{0,3}
/// are **clustered** so the {0,2} firing sector is strongly *lopsided* — leg s2
/// sits ~0.61 rad off e01 but ~1.54 rad off e03 (flank-sine ratio `r ≈ 1.73`,
/// well clear of `1`). That makes the §7.6.4 reconciliation observable at full
/// magnitude: the apex offset solves to `o₁ ≈ 0.68·O` (not the spec `O`), the
/// §8.8.3.1 coincident rebuild differs from the construct average by ~13%, and the
/// §7.6.4.3 terminal damping is a few thousandths — all far above f32 noise. Leg
/// s2 lies in `z = 0` so its slid apex keeps `z ≈ 0`. Returns
/// (mesh, apex, [s1..s4], beveled edges {0,1} and {0,3}).
fn lopsided_apex_junction() -> (LoopMesh, VertexKey, [VertexKey; 4], [EdgeKey; 2]) {
    let mut mesh = LoopMesh::new();
    let apex = mesh.create_vertex(Vec3::new(0.0, 0.0, 0.0));
    // s1/s3 are the beveled flanks, clustered into one half; s2 (the silhouette
    // leg, in z = 0) sits close to s1, so the {0,2} sector's two flank angles are
    // very unequal. s4 fills the opposite sector.
    let s1 = mesh.create_vertex(Vec3::new(1.0, 0.0, 0.35));
    let s2 = mesh.create_vertex(Vec3::new(0.866, 0.5, 0.0));
    let s3 = mesh.create_vertex(Vec3::new(-0.4226, 0.9063, 0.45));
    let s4 = mesh.create_vertex(Vec3::new(-0.4226, -0.9063, 0.30));
    let spokes = [s1, s2, s3, s4];
    let outers: Vec<VertexKey> = (0..4)
        .map(|i| {
            let a = mesh.vertex_position(spokes[i]);
            let b = mesh.vertex_position(spokes[(i + 1) % 4]);
            let mid = (a + b) * 0.5;
            mesh.create_vertex(mid * 1.8 + Vec3::new(0.0, 0.0, -0.4))
        })
        .collect();

    let edge = |mesh: &mut LoopMesh, a: VertexKey, b: VertexKey| {
        mesh.create_edge([a, b], EdgeCreateFlags::NONE)
    };
    for i in 0..4 {
        let sa = spokes[i];
        let sb = spokes[(i + 1) % 4];
        let outer = outers[i];
        let e_apex_a = mesh
            .query_edge_exists(apex, sa)
            .unwrap_or_else(|| edge(&mut mesh, apex, sa));
        let e_a_outer = edge(&mut mesh, sa, outer);
        let e_outer_b = edge(&mut mesh, outer, sb);
        let e_b_apex = mesh
            .query_edge_exists(apex, sb)
            .unwrap_or_else(|| edge(&mut mesh, apex, sb));
        mesh.face_create(
            &[apex, sa, outer, sb],
            &[e_apex_a, e_a_outer, e_outer_b, e_b_apex],
            FaceCreateFlags::NONE,
        );
    }

    let e01 = mesh.query_edge_exists(apex, s1).expect("spoke {0,1}");
    let e03 = mesh.query_edge_exists(apex, s3).expect("spoke {0,3}");
    (mesh, apex, spokes, [e01, e03])
}

/// BEVEL.md §8.8.3: the slid boundvert is built parametrically on the
/// *un-projected* leg line `v + t·d_leg`, so it lands **on** the silhouette
/// edge by construction — no face-plane / junction-normal projection drifts it
/// off. Here the firing sector's leg (spoke {0,2}) lies in `z = 0`, so its slid
/// boundvert must have `z ≈ 0`, whereas the §7.6.2 meet (loop_slide off) floats
/// off the plane at this irregular junction.
///
/// BEVEL.md §8.8.2.1 (firing-frame): the firing predicate is widened to the
/// genuine *vertex-normal* frame, so **both** qualifying single-silhouette apex
/// sectors slide onto their legs (spoke {0,2}'s sector and spoke {0,4}'s sector),
/// not just the one whose flanking faces happen to align with a sector-face frame.
/// The {0,2} slide is a large move (its §7.6.2 meet floats off z = 0); the {0,4}
/// sector is a *marginal two-petal* case-3 whose first in-between face `F1` drops
/// ~1.3° past its own wedge — an overshoot the §7.6.2 bounded-overshoot `2°` slack
/// **admits**, so `F1` is accepted and the meet floats *off* the leg. Its slide is
/// therefore a large move too. (A too-tight epsilon rejects that marginal `F1` and
/// advances to `F2`, pinning the meet near the leg boundary — the A/B suite proved
/// that `F2` placement wrong; this test now pins the widened-tolerance verdict.)
///
/// BEVEL.md §8.8.5 / §8.8.4: this is an **irregular** firing junction (the two
/// beveled edges rise out of the leg's plane at unequal angles), so the flank-sine
/// ratio ≠ 1 redistributes the flanking edges' §7.6.4 offsets and the ripple
/// reaches the four terminal strip verts at the edges' far ends — they are **not**
/// static here (the case-B recipe is flag-independent, but the reconciled offset it
/// consumes is not). So all six new verts differ between on and off: both apex
/// boundverts ({0,2} and the marginal-F1 {0,4}) and the termini, by observable
/// amounts (case A + case-B ripple).
#[test]
fn edge_bevel_loop_slide_lands_on_planar_leg_line() {
    let offset = 0.2;

    let build = |loop_slide: bool| {
        let (mut mesh, apex, spokes, [e01, e03]) = irregular_apex_junction_planar_leg();
        let apex_pos = mesh.vertex_position(apex);
        let leg2_dir = (mesh.vertex_position(spokes[1]) - apex_pos).normalize();
        let leg4_dir = (mesh.vertex_position(spokes[3]) - apex_pos).normalize();
        let result = mesh.bevel(
            &[e01.into(), e03.into()],
            BevelOptions {
                offset,
                affect: BevelAffect::Edges,
                loop_slide,
                ..Default::default()
            },
        );
        mesh.validate_topology();
        let positions: Vec<Vec3> = result
            .verts
            .iter()
            .map(|&v| mesh.vertex_position(v))
            .collect();
        let firing2 = firing_boundvert(&mesh, &result, apex_pos, leg2_dir);
        let firing4 = firing_boundvert(&mesh, &result, apex_pos, leg4_dir);
        (apex_pos, leg2_dir, leg4_dir, positions, firing2, firing4)
    };

    let (apex_pos, leg2_dir, leg4_dir, off_positions, b2_off, b4_off) = build(false);
    let (_, _, _, on_positions, b2_on, b4_on) = build(true);

    let perp = |p: Vec3, leg: Vec3| ((p - apex_pos) - leg * (p - apex_pos).dot(leg)).length();

    // §8.8.5: the {0,2} firing boundvert moves observably when the slide is
    // enabled — its leg lies in z = 0 while the §7.6.2 meet floats off z.
    assert!(
        (b2_on - b2_off).length() > 1.0e-3,
        "loop_slide on must move the {{0,2}} boundvert, on={b2_on:?} off={b2_off:?}",
    );
    // The {0,4} sector is a *marginal two-petal* case-3 (§7.6.2): its first
    // in-between face `F1`'s drop overshoots that face's wedge by only ~1.3°, an
    // overshoot the bounded-overshoot rule's `2°` slack (§7.6.2) **admits**, so
    // `F1` is accepted and the §7.6.2 meet lands at `F1`'s face-drop — which
    // floats *off* the leg line. So loop_slide moves the {0,4} boundvert
    // observably (like {0,2}), sliding the floating meet onto its leg. (A
    // too-tight epsilon — the prior `1e-4 rad` — wrongly rejects that marginal
    // `F1`, advancing to `F2`, whose drop pins near the leg boundary; the A/B
    // suite proved that `F2` placement wrong. This pins the widened-tolerance
    // `F1`-accept placement.)
    assert!(
        (b4_on - b4_off).length() > 1.0e-3,
        "loop_slide on must move the {{0,4}} boundvert, on={b4_on:?} off={b4_off:?}",
    );

    // §8.8.3: spoke {0,2}'s leg lies in z = 0, so its slid boundvert lands at
    // z ≈ 0 and exactly on the leg's line (`v + t·d_leg`, un-projected `d_leg`).
    assert!(
        b2_on.z.abs() < 1.0e-5,
        "slid {{0,2}} boundvert keeps z = 0 (leg lies in z = 0), got z={}",
        b2_on.z,
    );
    assert!(
        perp(b2_on, leg2_dir) < 1.0e-5,
        "slid {{0,2}} boundvert sits exactly on its leg's line, perp={}",
        perp(b2_on, leg2_dir),
    );

    // §8.8.2.1 (firing-frame): the second qualifying sector also slides onto its
    // own leg (spoke {0,4}) — the n_v frame fires it where a sector-face frame
    // wrongly rejected it (the under-fire).
    assert!(
        perp(b4_on, leg4_dir) < 1.0e-5,
        "slid {{0,4}} boundvert sits on its leg's line, perp={}",
        perp(b4_on, leg4_dir),
    );

    // The {0,2} §7.6.2 meet (loop_slide off) floats off its leg's z = 0 plane —
    // that is why its slide is observable (§8.8.5).
    assert!(
        b2_off.z.abs() > 1.0e-3,
        "{{0,2}} meet-point boundvert floats off z = 0, got z={}",
        b2_off.z,
    );
    // The {0,4} §7.6.2 meet (loop_slide off) is the marginal `F1` face-drop the
    // `2°` bounded-overshoot slack admits (§7.6.2), so it floats *off* the leg
    // line — that is why its slide is observable. (The too-tight prior epsilon
    // rejected `F1` and pinned the meet near the leg boundary at `F2`; the A/B
    // suite proved that placement wrong, so this now guards the `F1`-accept.)
    assert!(
        perp(b4_off, leg4_dir) > 1.0e-3,
        "{{0,4}} meet-point boundvert floats off its leg (marginal F1 drop), perp={}",
        perp(b4_off, leg4_dir),
    );

    // §8.8.5 / §8.8.4: at this irregular firing junction the slide perturbs the
    // two firing apex boundverts directly (case A) *and* ripples the §7.6.4
    // reconciled offsets out to the four terminal strip verts (case B), so all six
    // new verts drift between on and off. The termini move by a small width-pass
    // ripple — not a recipe change — so their drift is bounded well below the apex
    // slide's, distinguishing the legitimate ripple from a routing bug (§8.8.5).
    assert_eq!(on_positions.len(), off_positions.len(), "same vert count");
    assert_eq!(
        on_positions.len(),
        6,
        "two apex boundverts + four terminal strip verts"
    );
    let drifted = on_positions
        .iter()
        .zip(off_positions.iter())
        .filter(|(a, b)| (**a - **b).length() > 1.0e-5)
        .count();
    assert_eq!(
        drifted, 6,
        "both apex sectors slide (case A) and the four termini ripple (case B)",
    );
}

/// BEVEL.md §7.6.4.3–§7.6.4.5 / §8.8.3.1: the two-pass construct→reconcile→rebuild
/// pins the *magnitudes*, not just the directions. This test fails against the
/// earlier single-pass placement (apex left at the construct average of two unequal
/// spec sub-meets; terminals fed the apex offset carried unchanged) and passes once
/// both ends read the reconciled offsets.
///
/// **Coincident-sub-meet apex (§8.8.3.1).** The rebuilt apex sits at the single
/// reconciled distance `t* = O·(sin α₁ + sin α₂) / (sin²α₁ + sin²α₂)` along the leg
/// — where the two sub-meets coincide — not the construct average
/// `O·(sin α₁ + sin α₂) / (2·sin α₁·sin α₂)`.
///
/// **Damped terminal ripple (§7.6.4.3 / §7.6.4.4).** Each terminal strip vert is
/// fed the damped blend `(apex_offset + 0.04·O) / 1.04`, not the apex offset itself
/// (which overshoots by exactly `1.04×`). The terminal recipe (§7.5.1) is linear in
/// the offset it consumes, so an off-run affine probe at `O` and `2O` recovers each
/// terminal's *fed* offset; the recovered multiset must match the four damped
/// apex-offsets and be distinct from the undamped apex-offsets.
#[test]
fn edge_bevel_loop_slide_apex_and_termini_use_reconciled_offsets() {
    let o = 0.2_f32;

    let build = |loop_slide: bool, offset: f32| {
        let (mut mesh, apex, spokes, [e01, e03]) = lopsided_apex_junction();
        let apex_pos = mesh.vertex_position(apex);
        let leg2 = (mesh.vertex_position(spokes[1]) - apex_pos).normalize();
        let leg4 = (mesh.vertex_position(spokes[3]) - apex_pos).normalize();
        let edge01 = (mesh.vertex_position(spokes[0]) - apex_pos).normalize();
        let edge03 = (mesh.vertex_position(spokes[2]) - apex_pos).normalize();
        let result = mesh.bevel(
            &[e01.into(), e03.into()],
            BevelOptions {
                offset,
                affect: BevelAffect::Edges,
                loop_slide,
                ..Default::default()
            },
        );
        mesh.validate_topology();
        let positions: Vec<Vec3> = result
            .verts
            .iter()
            .map(|&v| mesh.vertex_position(v))
            .collect();
        (apex_pos, leg2, leg4, edge01, edge03, positions)
    };

    let (apex_pos, leg2, leg4, edge01, edge03, off_o) = build(false, o);
    let (_, _, _, _, _, off_2o) = build(false, 2.0 * o);
    let (_, _, _, _, _, on_o) = build(true, o);

    assert_eq!(off_o.len(), 6);
    assert_eq!(off_2o.len(), 6);
    assert_eq!(on_o.len(), 6);

    // The firing apex boundverts sit on their legs near the apex; the terminal
    // strip verts sit out by the far endpoints (≈ 1 away). Classify on the off@O
    // run; the same indices carry across runs (stable construction order).
    let sin = |u: Vec3, v: Vec3| u.cross(v).length();

    // §8.8.3.1 coincident-sub-meet apex distance, per firing sector.
    let apex_distance = |on_position: Vec3, leg: Vec3| (on_position - apex_pos).dot(leg);
    let b2_on = firing_boundvert_from(&on_o, apex_pos, leg2);
    let b4_on = firing_boundvert_from(&on_o, apex_pos, leg4);
    let t2 = apex_distance(b2_on, leg2);
    let t4 = apex_distance(b4_on, leg4);

    let s1_2 = sin(edge01, leg2);
    let s3_2 = sin(edge03, leg2);
    let s1_4 = sin(edge01, leg4);
    let s3_4 = sin(edge03, leg4);

    let expected_new = |sa: f32, sb: f32| o * (sa + sb) / (sa * sa + sb * sb);
    let expected_old = |sa: f32, sb: f32| o * (sa + sb) / (2.0 * sa * sb);

    // Both apex boundverts land at the reconciled (coincident) distance, not the
    // construct average. On the lopsided {0,2} sector the two differ by ~13%, which
    // is what fails the single-pass placement; the near-symmetric {0,4} sector
    // matches `new` too (where `new ≈ old`).
    let mut max_apex_gap: f32 = 0.0;
    for (t, sa, sb) in [(t2, s1_2, s3_2), (t4, s1_4, s3_4)] {
        let new = expected_new(sa, sb);
        let old = expected_old(sa, sb);
        assert!(
            (t - new).abs() < 1.0e-4,
            "apex slides to the reconciled coincident distance t*={new}, got {t}",
        );
        let gap = (new - old).abs();
        max_apex_gap = max_apex_gap.max(gap);
        if gap > 5.0e-4 {
            assert!(
                (t - old).abs() > (t - new).abs(),
                "apex is the rebuilt coincident point, not the construct average \
                 (t={t}, new={new}, old={old})",
            );
        }
    }
    assert!(
        max_apex_gap > 1.0e-3,
        "the lopsided sector exercises a real apex magnitude differential \
         (max_gap={max_apex_gap})",
    );

    // §7.6.4.3 terminal damping. The apex offset of each flanking edge in a sector
    // is `t* · sin(angle(edge, leg))`; the terminal far end is fed the damped blend.
    let damp = |apex_offset: f32| (apex_offset + 0.04 * o) / 1.04;
    let apex_offsets = [t2 * s1_2, t2 * s3_2, t4 * s1_4, t4 * s3_4];
    let mut damped: Vec<f32> = apex_offsets.iter().map(|&a| damp(a)).collect();
    damped.sort_by(|a, b| a.partial_cmp(b).unwrap());

    // Recover each terminal strip vert's fed offset via the off-run affine probe.
    let mut recovered: Vec<f32> = Vec::new();
    for index in 0..6 {
        if (off_o[index] - apex_pos).length() <= 0.5 {
            continue; // an apex boundvert, not a terminal
        }
        let dir = off_2o[index] - off_o[index]; // = O · (per-unit-offset slope)
        let fraction = (on_o[index] - off_o[index]).dot(dir) / dir.dot(dir);
        recovered.push(o + fraction * o);
    }
    recovered.sort_by(|a, b| a.partial_cmp(b).unwrap());
    assert_eq!(recovered.len(), 4, "four terminal strip verts");

    let mut apex_sorted = apex_offsets.to_vec();
    apex_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    // The recovered fed offsets are the damped values, edge-for-edge (damp is
    // monotone in the apex offset, so sorting all three keeps them aligned), and on
    // the lopsided sector they are distinguishably *not* the undamped apex offsets
    // the overshoot bug carried to the terminal.
    let mut max_damp_gap: f32 = 0.0;
    for (got, (want_damped, want_apex)) in
        recovered.iter().zip(damped.iter().zip(apex_sorted.iter()))
    {
        assert!(
            (got - want_damped).abs() < 5.0e-4,
            "terminal fed the §7.6.4.3 damped offset {want_damped}, got {got}",
        );
        let gap = (want_apex - want_damped).abs();
        max_damp_gap = max_damp_gap.max(gap);
        if gap > 5.0e-4 {
            assert!(
                (got - want_damped).abs() < (got - want_apex).abs(),
                "terminal is the §7.6.4.3 damped offset, not the apex offset carried \
                 unchanged (got={got}, damped={want_damped}, apex={want_apex})",
            );
        }
    }
    assert!(
        max_damp_gap > 1.0e-3,
        "the lopsided sector exercises a real terminal damping differential \
         (max_gap={max_damp_gap})",
    );
}

/// BEVEL.md §8.8.5: at a *symmetric* junction the slide is a no-op — the slid
/// point coincides with the bisector meet-point, so `loop_slide` on and off
/// produce the identical boundvert. A symmetric cube corner with one
/// unbeveled silhouette leg between the two beveled edges is the canonical
/// no-op fixture.
#[test]
fn edge_bevel_loop_slide_is_noop_at_symmetric_junction() {
    let build = |loop_slide: bool| {
        let mut mesh = LoopMesh::new();
        mesh_add_cube(&mut mesh);
        // v0 = [1,-1,-1]; bevel {0,1} and {0,4}, leaving {0,3} as the lone
        // unbeveled leg of the sector between them. The cube corner is
        // symmetric, so the slide must not move the boundvert.
        let p0 = Vec3::new(1.0, -1.0, -1.0);
        let e01 = cube_edge(&mesh, p0, Vec3::new(-1.0, -1.0, -1.0));
        let e04 = cube_edge(&mesh, p0, Vec3::new(1.0, -1.0, 1.0));
        let result = mesh.bevel(
            &[e01.into(), e04.into()],
            BevelOptions {
                offset: 0.25,
                affect: BevelAffect::Edges,
                loop_slide,
                ..Default::default()
            },
        );
        mesh.validate_topology();
        let mut positions: Vec<Vec3> = result
            .verts
            .iter()
            .map(|&v| mesh.vertex_position(v))
            .collect();
        positions.sort_by(|a, b| (a.x, a.y, a.z).partial_cmp(&(b.x, b.y, b.z)).unwrap());
        positions
    };

    let off = build(false);
    let on = build(true);
    assert_eq!(off.len(), on.len(), "same boundvert count");
    for (a, b) in off.iter().zip(on.iter()) {
        assert!(
            (*a - *b).length() < 1.0e-5,
            "symmetric junction: loop_slide is a no-op, {a:?} vs {b:?}",
        );
    }
}

/// Like `irregular_apex_junction`, but the firing sector's silhouette leg
/// (spoke {0,2}) points *outside* the angular wedge spanned by its two bounding
/// beveled edges {0,1} and {0,3}: it swings into the lower-y half-plane so the
/// `g_k → e_mid` sub-meet sweeps *backward*. The sector still has exactly one
/// silhouette leg (so §8.8.2 condition 1 fires), but one flanking sub-meet is
/// reflex (so condition 2 must reject the slide).
/// Returns (mesh, apex, [v1,v2,v3,v4], beveled edges {0,1} and {0,3}).
fn reflex_apex_junction() -> (LoopMesh, VertexKey, [VertexKey; 4], [EdgeKey; 2]) {
    let mut mesh = LoopMesh::new();
    let apex = mesh.create_vertex(Vec3::new(0.0, 0.0, 0.0));
    // s1, s3 bound the firing sector. s2 — the leg — is placed so that, viewed
    // from the junction's vertex-normal frame `n_v` (§8.8.2.1, the frame the
    // forward/reflex guard now uses), the `e_mid → g_(k+1)` sub-meet sweeps
    // *backward*: it folds reflex while the `g_k → e_mid` side stays forward. The
    // petals stay non-planar so s2 remains a silhouette leg.
    let s1 = mesh.create_vertex(Vec3::new(1.0, 0.2, -0.1));
    let s2 = mesh.create_vertex(Vec3::new(-0.2, 0.0, 0.2));
    let s3 = mesh.create_vertex(Vec3::new(0.2, 1.0, -0.1));
    let s4 = mesh.create_vertex(Vec3::new(-0.7, -0.7, -0.3));
    let spokes = [s1, s2, s3, s4];
    let outers: Vec<VertexKey> = (0..4)
        .map(|i| {
            let a = mesh.vertex_position(spokes[i]);
            let b = mesh.vertex_position(spokes[(i + 1) % 4]);
            let mid = (a + b) * 0.5;
            mesh.create_vertex(mid * 1.8 + Vec3::new(0.0, 0.0, -0.4))
        })
        .collect();

    let edge = |mesh: &mut LoopMesh, a: VertexKey, b: VertexKey| {
        mesh.create_edge([a, b], EdgeCreateFlags::NONE)
    };
    for i in 0..4 {
        let sa = spokes[i];
        let sb = spokes[(i + 1) % 4];
        let outer = outers[i];
        let e_apex_a = mesh
            .query_edge_exists(apex, sa)
            .unwrap_or_else(|| edge(&mut mesh, apex, sa));
        let e_a_outer = edge(&mut mesh, sa, outer);
        let e_outer_b = edge(&mut mesh, outer, sb);
        let e_b_apex = mesh
            .query_edge_exists(apex, sb)
            .unwrap_or_else(|| edge(&mut mesh, apex, sb));
        mesh.face_create(
            &[apex, sa, outer, sb],
            &[e_apex_a, e_a_outer, e_outer_b, e_b_apex],
            FaceCreateFlags::NONE,
        );
    }

    let e01 = mesh.query_edge_exists(apex, s1).expect("spoke {0,1}");
    let e03 = mesh.query_edge_exists(apex, s3).expect("spoke {0,3}");
    (mesh, apex, spokes, [e01, e03])
}

/// BEVEL.md §8.8.2 condition 2 (the "good offset" guard): firing requires
/// *both* that the sector has exactly one silhouette leg *and* that neither
/// flanking sub-meet folds back. When a sub-meet is reflex the slide is
/// rejected and the sector keeps its §7.6.2 meet-point — so on a reflex fixture
/// `loop_slide = true` yields the *same* firing boundvert as `loop_slide =
/// false`. (Condition 1 alone fires here; condition 2 is what vetoes the slide.)
#[test]
fn edge_bevel_loop_slide_rejected_when_flank_is_reflex() {
    let offset = 0.2;

    let build = |loop_slide: bool| {
        let (mut mesh, apex, spokes, [e01, e03]) = reflex_apex_junction();
        let apex_pos = mesh.vertex_position(apex);
        let leg_dir = (mesh.vertex_position(spokes[1]) - apex_pos).normalize();
        let result = mesh.bevel(
            &[e01.into(), e03.into()],
            BevelOptions {
                offset,
                affect: BevelAffect::Edges,
                loop_slide,
                ..Default::default()
            },
        );
        mesh.validate_topology();
        let boundvert = firing_boundvert(&mesh, &result, apex_pos, leg_dir);
        (apex_pos, leg_dir, boundvert)
    };

    let (apex_pos, leg_dir, b_off) = build(false);
    let (_, _, b_on) = build(true);

    // Condition 2 vetoes the slide: the firing boundvert must NOT move.
    assert!(
        (b_on - b_off).length() < 1.0e-5,
        "reflex flank: loop_slide must fall back to the meet-point, \
         got on={b_on:?} off={b_off:?}",
    );

    // And — the point of the guard — that retained meet-point genuinely floats
    // off the leg's line (the slide it rejected would have pinned it on-line).
    let perpendicular = ((b_on - apex_pos) - leg_dir * (b_on - apex_pos).dot(leg_dir)).length();
    assert!(
        perpendicular > 1.0e-4,
        "rejected slide keeps the off-line §7.6.2 meet, perp={perpendicular}",
    );
}

/// BEVEL.md §7.6.2: at an irregular non-planar junction the boundvert is the
/// *pairwise* meet of its two flanking edges' offset lines — a local 2-line
/// solve, never a fit over every incident face. The local solve stays bounded:
/// each boundvert lands within a small multiple of `offset` of the apex (the
/// strips meet near the original corner), where the discarded global
/// least-squares formulation drove a boundvert far out of position. This pins
/// the locality the rewrite restored, with `loop_slide` off so every boundvert
/// comes from §7.6.2.
#[test]
fn edge_bevel_meet_point_is_bounded_at_irregular_junction() {
    let offset = 0.2;
    let (mut mesh, apex, _spokes, [e01, e03]) = irregular_apex_junction();
    let apex_pos = mesh.vertex_position(apex);
    let result = mesh.bevel(
        &[e01.into(), e03.into()],
        BevelOptions {
            offset,
            affect: BevelAffect::Edges,
            loop_slide: false,
            ..Default::default()
        },
    );
    mesh.validate_topology();

    // The two junction boundverts are the new verts nearest the apex (the
    // strip verts slide out along the spokes, far from it). Both §7.6.2 meets
    // sit within a few offsets of the apex — the locality the rewrite restored.
    // The discarded global least-squares could drive a boundvert far out.
    let mut distances: Vec<f32> = result
        .verts
        .iter()
        .map(|&v| (mesh.vertex_position(v) - apex_pos).length())
        .collect();
    distances.sort_by(|a, b| a.partial_cmp(b).unwrap());
    for &distance in distances.iter().take(2) {
        assert!(
            distance < 4.0 * offset,
            "pairwise meet keeps boundvert near the surface, got {distance} \
             (apex-relative) for offset {offset}",
        );
    }
}

/// BEVEL.md §7.6.2 cases 1/2: a sector that is a single planar face (or whose
/// in-between legs are all in-plane) reduces to the exact 2-line crossing on the
/// angle bisector. At a right-angle planar corner with equal offset `t` the
/// meet sits `t / sin(45°) = t·√2` from the original corner — the closed-form
/// crossing identity, with no out-of-plane excursion.
#[test]
fn edge_bevel_meet_point_planar_sector_is_exact_bisector_crossing() {
    let mut mesh = LoopMesh::new();
    mesh_add_cube(&mut mesh);
    let corner = Vec3::new(1.0, -1.0, -1.0);
    let e01 = cube_edge(&mesh, corner, Vec3::new(-1.0, -1.0, -1.0));
    let e03 = cube_edge(&mesh, corner, Vec3::new(1.0, 1.0, -1.0));
    let e04 = cube_edge(&mesh, corner, Vec3::new(1.0, -1.0, 1.0));
    let offset = 0.25;
    let result = mesh.bevel(
        &[e01.into(), e03.into(), e04.into()],
        BevelOptions {
            offset,
            affect: BevelAffect::Edges,
            ..Default::default()
        },
    );
    mesh.validate_topology();

    let expected = offset * 2.0_f32.sqrt();
    // The patch face boundverts are the §7.6.2 meet points; strip verts sit
    // elsewhere, so filter to the triangular patch (matching the existing
    // three-edge-junction pin).
    let patch = result
        .faces
        .iter()
        .copied()
        .find(|&f| mesh.face_vert_keys(f).len() == 3)
        .expect("the s=3 junction patch is a triangle");
    for v in mesh.face_vert_keys(patch) {
        let distance = (mesh.vertex_position(v) - corner).length();
        assert!(
            (distance - expected).abs() < 1.0e-5,
            "planar meet sits offset·√2 from the corner, got {distance}",
        );
    }
}

/// Build a closed non-planar apex fan: apex `v` above an irregular rim of four
/// rim verts at differing heights, capped underneath so the apex disk closes.
/// Each fan triangle (apex, rim_i, rim_{i+1}) has a different normal, so every
/// apex edge sits between two non-coplanar faces — an out-of-plane (silhouette)
/// in-between edge whenever it lies inside a bevel sector. Returns (apex, rim).
fn irregular_apex_fan(mesh: &mut LoopMesh) -> (VertexKey, Vec<VertexKey>) {
    let apex = mesh.create_vertex(Vec3::new(0.0, 0.0, 1.0));
    // Six rim verts at deliberately irregular radii and heights, so every fan
    // triangle has a distinct normal and a sector spanning several in-between
    // edges sees genuinely different candidate faces along its walk.
    let rim_coords = [
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(0.55, 0.9, 0.4),
        Vec3::new(-0.5, 0.85, 0.15),
        Vec3::new(-1.0, 0.0, -0.1),
        Vec3::new(-0.45, -0.8, -0.3),
        Vec3::new(0.6, -0.95, 0.05),
    ];
    let rim: Vec<VertexKey> = rim_coords.iter().map(|&p| mesh.create_vertex(p)).collect();
    let n = rim.len();

    let spoke: Vec<EdgeKey> = rim
        .iter()
        .map(|&r| mesh.create_edge([apex, r], EdgeCreateFlags::NONE))
        .collect();
    let mut rim_edge = Vec::new();
    for i in 0..n {
        rim_edge.push(mesh.create_edge([rim[i], rim[(i + 1) % n]], EdgeCreateFlags::NONE));
    }
    for i in 0..n {
        let j = (i + 1) % n;
        mesh.face_create(
            &[apex, rim[i], rim[j]],
            &[spoke[i], rim_edge[i], spoke[j]],
            FaceCreateFlags::NONE,
        );
    }
    // Close the underside so the apex disk is a true closed fan.
    mesh.face_create(&rim, &rim_edge, FaceCreateFlags::NONE);
    (apex, rim)
}

/// BEVEL.md §7.6.2 (face-drop in-wedge test): at a **non-planar** junction the
/// two skew offset lines give a provisional midpoint that floats off the surface,
/// so it is dropped onto a sector face — but only onto the **first face in the
/// disk-order walk whose `v`-anchored wedge actually contains the drop**. The walk
/// starts at `g_k`'s forward face, but for a **three-petal** sector (two
/// out-of-plane legs) the midpoint sits deep in the sector interior, so `g_k`'s
/// forward-face drop sweeps well past that face's wedge and is **rejected**; the
/// walk advances and the drop lands in-wedge on the **next** face. This pins the
/// rejection: each junction boundvert lands on its sector's *second* face plane,
/// decisively off `g_k`'s forward face (the accept-first bug landed it on `g_k`'s
/// plane, ~0.05–0.25 unit away).
#[test]
fn junction_face_drop_lands_on_v_anchored_g_k_plane() {
    let mut mesh = LoopMesh::new();
    let (apex, rim) = irregular_apex_fan(&mut mesh);
    let apex_pos = mesh.vertex_position(apex);
    let n = rim.len();
    let rim_pos: Vec<Vec3> = rim.iter().map(|&r| mesh.vertex_position(r)).collect();

    // Pre-bevel normal of fan triangle (apex, rim_i, rim_{i+1}), captured before
    // the surgery rebuilds those faces. The forward face of beveled spoke `e_i`
    // (the face on the side toward the next spoke) is triangle `i`.
    let triangle_normal = |i: usize| {
        let a = rim_pos[i] - apex_pos;
        let b = rim_pos[(i + 1) % n] - apex_pos;
        a.cross(b).normalize()
    };

    // Bevel two spokes three apart: spoke 0 and spoke 3. The +Y sector walks
    // 0 → 3 (g_k = spoke 0, forward face = triangle 0, two in-between legs); the
    // −Y sector walks 3 → 0 (g_k = spoke 3, forward face = triangle 3). Both
    // sectors span out-of-plane legs, so both fire the skew face-drop.
    let spoke_0 = mesh.query_edge_exists(apex, rim[0]).expect("spoke 0");
    let spoke_3 = mesh.query_edge_exists(apex, rim[3]).expect("spoke 3");
    let offset = 0.3_f32;
    let result = mesh.bevel(
        &[spoke_0.into(), spoke_3.into()],
        BevelOptions {
            offset,
            affect: BevelAffect::Edges,
            ..Default::default()
        },
    );

    // The two junction-sector boundverts are the result verts near the apex; the
    // strip ring verts sit out at the rim ends (apex-distance > 1).
    let junction_verts: Vec<Vec3> = result
        .verts
        .iter()
        .map(|&v| mesh.vertex_position(v))
        .filter(|p| (*p - apex_pos).length() < 1.0)
        .collect();
    assert_eq!(
        junction_verts.len(),
        2,
        "two junction sectors → two near-apex boundverts",
    );

    for point in &junction_verts {
        // Bounded near the apex: a small on-surface nudge, not floated off (the
        // v-anchored drop keeps it ≈ offset·few out, never the ~unit rim depth).
        let radius = (*point - apex_pos).length();
        assert!(
            radius > 1.0e-4 && radius < 4.0 * offset,
            "junction boundvert sits just out from the apex, got {radius}",
        );
    }

    // The +Y sector walks 0 → 3: g_k = spoke 0 (forward face triangle 0), then
    // legs spoke 1, spoke 2 (forward faces triangle 1, triangle 2). Its midpoint
    // sits deep enough that triangle 0's drop is out-of-wedge, so the accepted
    // face is triangle 1. The −Y sector walks 3 → 0: g_k = spoke 3 (triangle 3),
    // accepted face triangle 4.
    let plus = *junction_verts
        .iter()
        .find(|p| p.y > 0.0)
        .expect("+Y junction boundvert");
    let minus = *junction_verts
        .iter()
        .find(|p| p.y < 0.0)
        .expect("-Y junction boundvert");

    // Each boundvert is decisively OFF its sector's g_k forward face (the rejected
    // first candidate) — the accept-first bug would land it here, at f32 round-off.
    assert!(
        (plus - apex_pos).dot(triangle_normal(0)).abs() > 0.02,
        "+Y boundvert rejected spoke-0's forward face (off its v-plane)",
    );
    assert!(
        (minus - apex_pos).dot(triangle_normal(3)).abs() > 0.02,
        "-Y boundvert rejected spoke-3's forward face (off its v-plane)",
    );

    // It lands ON the accepted in-wedge face's v-plane (the second face in each
    // walk): +Y on triangle 1, −Y on triangle 4.
    assert!(
        (plus - apex_pos).dot(triangle_normal(1)).abs() < 1.0e-5,
        "+Y boundvert on the accepted second face (triangle 1) v-plane, dist {}",
        (plus - apex_pos).dot(triangle_normal(1)).abs(),
    );
    assert!(
        (minus - apex_pos).dot(triangle_normal(4)).abs() < 1.0e-5,
        "-Y boundvert on the accepted second face (triangle 4) v-plane, dist {}",
        (minus - apex_pos).dot(triangle_normal(4)).abs(),
    );
    mesh.validate_topology();
}

/// BEVEL.md §7.6.2 worked example (three-petal in-wedge rejection). At the
/// irregular four-spoke apex `v` the three-petal complementary sector's skew
/// midpoint sits deep in the interior, so its drop onto `g_k`'s forward face
/// `F1` sweeps ~142° — well past `F1`'s ~95° wedge — and must be **rejected**;
/// the walk advances and the drop lands in-wedge on the next face `F2` (~50° ≤
/// ~85°). Getting the rejection wrong lifts the boundvert ~offset in the
/// out-of-plane coordinate (`z ≈ 0.089` instead of `z ≈ 0.009`). This pins the
/// predicate against the guide's table and the disk-order pairing it needs: the
/// reversed pair measures `F1`'s *reflex* wedge and mis-accepts the drop.
#[test]
fn three_petal_in_wedge_rejection_matches_worked_example() {
    let v = Vec3::ZERO;
    // Spoke directions (from v) of the sector's g_k, first leg, second leg.
    let g_k = Vec3::new(0.5, 2.0, 0.0).normalize();
    let leg_1 = Vec3::new(-2.0, 0.3, 0.6).normalize();
    let leg_2 = Vec3::new(-0.5, -2.0, 0.0).normalize();
    // The two candidate forward faces' v-anchored plane normals (guide table).
    let normal_f1 = Vec3::new(0.140, 0.140, 0.980).normalize();
    let normal_f2 = Vec3::new(0.154, -0.154, 0.976).normalize();
    // The skew closest-points midpoint the two case-3 offset lines produce.
    let midpoint = Vec3::new(-0.3501, -0.2948, 0.0108);

    // Drop the midpoint onto each face's v-anchored plane, then take the wedge
    // bounds from the *raw* disk-order walk edges (F1: g_k → leg_1; F2: leg_1 →
    // leg_2) with no in-plane projection, exactly as `drop_onto_sector_face`
    // does — `point_in_wedge` orients its sweeps by the face normal, absorbing
    // the bounds' small out-of-plane tilt.
    let drop = |normal: Vec3| midpoint - normal * (midpoint - v).dot(normal);
    let dir_to = |p: Vec3| (p - v).normalize();

    let drop_f1 = drop(normal_f1);
    let drop_f2 = drop(normal_f2);

    // The rejected first face lifts the boundvert far off the base plane; the
    // accepted second face barely lifts it — the ~0.08 out-of-plane error.
    assert!(
        (drop_f1.z - 0.0889).abs() < 2.0e-3,
        "F1 (rejected) drop lifts to z≈0.089, got {}",
        drop_f1.z,
    );
    assert!(
        (drop_f2.z - 0.0088).abs() < 2.0e-3,
        "F2 (accepted) drop stays near z≈0.009, got {}",
        drop_f2.z,
    );

    // F1's disk-order wedge rejects the drop (sweep past the wedge)…
    assert!(
        !point_in_wedge(g_k, leg_1, dir_to(drop_f1), normal_f1),
        "F1 forward face is out-of-wedge and must be rejected",
    );
    // …and F2's wedge accepts it (in-wedge), so F2's drop is the boundvert.
    assert!(
        point_in_wedge(leg_1, leg_2, dir_to(drop_f2), normal_f2),
        "F2 forward face is in-wedge and must be accepted",
    );

    // The disk-order pairing is load-bearing: the reversed pair (what a
    // winding-order edge lookup can yield) measures F1's *reflex* wedge and
    // wrongly accepts the out-of-wedge drop — the accept-first bug.
    assert!(
        point_in_wedge(leg_1, g_k, dir_to(drop_f1), normal_f1),
        "reversed pair measures the reflex wedge and mis-accepts F1",
    );
}

/// The **superseded** "into the face" perpendicular: the unit in-plane perpendicular
/// to `direction` about `normal`, flipped to point toward `face`'s vertex mean. This
/// is the heuristic §7.6.2's role-based cross order replaced, and it lives here — and
/// only here — as the contrast the sign pins measure the fix against. It agrees with
/// the cross order wherever the face is convex at `vertex_position` and inverts
/// wherever the face's mean falls on the sector's convex complement.
fn centroid_oriented_perpendicular(
    mesh: &LoopMesh,
    direction: Vec3,
    normal: Vec3,
    face: FaceKey,
    vertex_position: Vec3,
) -> Vec3 {
    let mut perpendicular = direction.cross(normal);
    let length = perpendicular.length();
    if length <= f32::EPSILON {
        return Vec3::ZERO;
    }
    perpendicular /= length;
    if perpendicular.dot(mesh.face_center_of_mass(face) - vertex_position) < 0.0 {
        perpendicular = -perpendicular;
    }
    perpendicular
}

/// BEVEL.md §7.6.2 case 3, reflex flanking face at the junction vertex, driven
/// through the **production path** (`vertex_corner_ring` → `sector_split` →
/// `sector_layout`) rather than a hand-built `Sector`. `v` is a boundary junction
/// whose forward flank `g_k` (`v→r`) bounds a concave arrowhead quad whose interior
/// angle at `v` is ~240° (reflex), while the far flank `g_(k+1)` (`v→b`) lifts out
/// of the arrowhead's plane so the shared leg is out-of-plane → the sector is
/// genuinely case 3. Because the barbs `r`/`leg` are far out and the notch tip is
/// close, the arrowhead's area **centroid sits inside the convex complement** of the
/// sector (below `v`, in the excluded down-cone between `g_k` and the leg). The old
/// case-3 construction oriented each perpendicular by that centroid and so pushed
/// `g_k`'s offset line *out* of the sector — a `2·offset`-scale error. The fix
/// (`normal_first × dir_first` on the walk-sense normal) pushes it in regardless of
/// convexity. This pins that with the production `point_in_wedge` predicate over the
/// sector's own reflex-face wedge: the fixed perpendicular lands **in** the wedge,
/// the centroid perpendicular lands **out**.
#[test]
fn case3_reflex_face_perp_pushes_into_sector() {
    let mut mesh = LoopMesh::new();
    // Arrowhead quad in z = 0 with `v` at the reflex notch: far barbs `r`, `leg`
    // (120° apart, opening downward) and a close tip. The two edges at `v` are
    // `v→r` (forward flank g_k) and `v→leg` (the in-between leg). Wound
    // consistently with `face_last` across the shared leg edge (opposite traversal
    // of `v↔leg`) so the disk walk's sense — and hence §7.6.2's perpendicular
    // sign — is well defined.
    let v = mesh.create_vertex(Vec3::new(0.0, 0.0, 0.0));
    let tip = mesh.create_vertex(Vec3::new(0.0, 0.5, 0.0));
    let r = mesh.create_vertex(Vec3::new(3.46, -2.0, 0.0));
    let leg = mesh.create_vertex(Vec3::new(-3.46, -2.0, 0.0));
    // The far flank g_(k+1) lifts out of z = 0, so the shared leg is out-of-plane
    // and the sector is case 3.
    let b = mesh.create_vertex(Vec3::new(-2.5, 0.5, 1.2));

    let reflex_face = mesh.face_create_verts(&[leg, v, r, tip], FaceCreateFlags::NONE);
    mesh.face_create_verts(&[v, leg, b], FaceCreateFlags::NONE);

    let edge_vr = mesh.query_edge_exists(v, r).expect("g_k edge");
    let edge_vb = mesh.query_edge_exists(v, b).expect("g_(k+1) edge");
    let v_pos = mesh.vertex_position(v);

    // The real disk walk and sector split — not a hand-assembled `Sector`.
    let is_beveled = |edge: EdgeKey| edge == edge_vr || edge == edge_vb;
    let corners = mesh.vertex_bevel_corner_ring(v, &is_beveled);
    let sector = sector_split(&corners, &is_beveled)
        .into_iter()
        .find(|s| {
            s.bounding_edges.len() == 2
                && s.bounding_edges.contains(&edge_vr)
                && s.bounding_edges.contains(&edge_vb)
        })
        .expect("the r↔b sector");
    let layout = mesh
        .sector_layout(v, v_pos, &sector)
        .expect("case-3 sector layout");
    assert!(layout.non_planar, "out-of-plane leg → case 3 sector");

    // The walk enters `reflex_face` first (through `g_k = v→r`), so it is the first
    // drop candidate; `perp_first` is the flank paired with it.
    assert_eq!(
        sector.bounding_edges[0], edge_vr,
        "the reflex flank is the sector's forward flank",
    );
    let reflex_wedge = &layout.drop_candidates[0];

    // The sector's forward face is genuinely reflex at `v`: the wedge swept from
    // `g_k` to its next edge (the leg), about the walk-sense normal, exceeds π.
    let swept = |from: Vec3, to: Vec3, normal: Vec3| -> f32 {
        let angle = from.dot(to).clamp(-1.0, 1.0).acos();
        if from.cross(to).dot(normal) < 0.0 {
            std::f32::consts::TAU - angle
        } else {
            angle
        }
    };
    let wedge_angle = swept(
        reflex_wedge.dir_walk,
        reflex_wedge.dir_next,
        reflex_wedge.normal,
    );
    assert!(
        wedge_angle > std::f32::consts::PI,
        "forward face is reflex at v (wedge {wedge_angle} rad > π)",
    );

    // The fix: `perp_first = normal_first × dir_first` lands **inside** the sector's
    // reflex-face wedge — it pushes g_k's offset line into the sector.
    assert!(
        point_in_wedge(
            reflex_wedge.dir_walk,
            reflex_wedge.dir_next,
            layout.perp_first,
            reflex_wedge.normal,
        ),
        "the fixed perpendicular pushes g_k's offset INTO the sector wedge",
    );

    // The old case-3 heuristic oriented the perpendicular by the face centroid.
    // That centroid sits in the convex complement (below `v`), so its perpendicular
    // lands **outside** the wedge — the sign bug the fix corrects.
    let centroid_perp = centroid_oriented_perpendicular(
        &mesh,
        layout.dir_first,
        layout.normal_first,
        reflex_face,
        v_pos,
    );
    assert!(
        !point_in_wedge(
            reflex_wedge.dir_walk,
            reflex_wedge.dir_next,
            centroid_perp,
            reflex_wedge.normal,
        ),
        "the centroid heuristic (old case-3) landed OUT of the sector wedge — the bug",
    );

    // Documenting the cause: the reflex face's centroid really is on the sector's
    // convex complement — the half-plane on the far side of `g_k` from `perp_first`.
    assert!(
        (mesh.face_center_of_mass(reflex_face) - v_pos).dot(layout.perp_first) < 0.0,
        "the reflex face's centroid sits on the sector's convex complement",
    );
}

/// A flat closed fan about `v = (0,0,0)` whose spokes `A = (−2,0,0)` and
/// `B = (−3,0.05,0)` leave `v` `0.955°` apart, so the sector they flank is a
/// near-zero-area sliver (`θ_dir ≈ 179.045°`) — the only doubled-back sector in the
/// fan. `C = (1,0,0)` and the unbeveled leg `D = (0,−1,0)` close it. The two sliver
/// spokes deliberately differ in **length** (`2` vs `3.00042`) as well as in bearing,
/// so sliding along the wrong one of the pair is visible twice over: in the direction,
/// and in the length §4.1's clamp is measured against.
///
/// The `0.955°` bearing gap is the interesting part of the band and not a free choice:
/// it sits *outside* a `cos θ ≥ 1 − 1e-4` reading of `EPS_ANG` (which is only `0.81°`)
/// and inside §7.6.2's stated `2°`, so the fixture is unreachable at all under the
/// narrower gate — see `junction_ladder_fires_across_the_whole_two_degree_band`.
fn sliver_sector_fan() -> (LoopMesh, VertexKey, [VertexKey; 4]) {
    let mut mesh = LoopMesh::new();
    let v = mesh.create_vertex(Vec3::ZERO);
    let a = mesh.create_vertex(Vec3::new(-2.0, 0.0, 0.0));
    let b = mesh.create_vertex(Vec3::new(-3.0, 0.05, 0.0));
    let c = mesh.create_vertex(Vec3::new(1.0, 0.0, 0.0));
    let d = mesh.create_vertex(Vec3::new(0.0, -1.0, 0.0));
    // Disk order by bearing at `v`: C (0°), B (179.045°), A (180°), D (270°).
    mesh.face_create_verts(&[v, c, b], FaceCreateFlags::NONE);
    mesh.face_create_verts(&[v, b, a], FaceCreateFlags::NONE);
    mesh.face_create_verts(&[v, a, d], FaceCreateFlags::NONE);
    mesh.face_create_verts(&[v, d, c], FaceCreateFlags::NONE);
    (mesh, v, [a, b, c, d])
}

/// BEVEL.md §7.6.2's doubled-back bullet: the slide runs along the **second** flanking
/// edge of the ordered pair — the disk successor, whose direction was taken out of `v`
/// — and *not* the first. Both halves of that are asserted, because §4.1's clamp reads
/// the second edge's far endpoint: at `offset = 0.5` the two candidates differ only in
/// direction, and at `offset = 2.5` they differ in whether the clamp fires at all
/// (`A` is `2` long, `B` is `3.00042`).
///
/// The disk walk's direction is not canonical, so the expectation is stated against
/// `bounding_edges[1]` rather than against `A` or `B` by name — which is exactly the
/// rule, and is what the guide's winding-reversal witness pins.
#[test]
fn junction_doubled_back_meet_slides_along_the_second_flanking_edge() {
    let (mesh, v, [a, b, c, _d]) = sliver_sector_fan();
    let v_pos = mesh.vertex_position(v);
    let spokes: Vec<EdgeKey> = [a, b, c]
        .iter()
        .map(|&far| mesh.query_edge_exists(v, far).expect("beveled spoke"))
        .collect();
    let is_beveled = |edge: EdgeKey| spokes.contains(&edge);

    let corners = mesh.vertex_bevel_corner_ring(v, &is_beveled);
    let edge_a = mesh.query_edge_exists(v, a).expect("spoke A");
    let edge_b = mesh.query_edge_exists(v, b).expect("spoke B");
    let sector = sector_split(&corners, &is_beveled)
        .into_iter()
        .find(|s| {
            s.bounding_edges.len() == 2
                && s.bounding_edges.contains(&edge_a)
                && s.bounding_edges.contains(&edge_b)
        })
        .expect("the A↔B sliver sector");

    let first = sector.bounding_edges[0];
    let second = sector.bounding_edges[1];
    for offset in [0.5_f32, 2.5] {
        let meet = mesh.solve_meet_point(v, v_pos, &sector, offset);
        let along_second = mesh.slide_along_incident_edge(v, second, offset);
        let along_first = mesh.slide_along_incident_edge(v, first, offset);
        assert!(
            (meet - along_second).length() < 1.0e-6,
            "at offset {offset} the doubled-back meet must slide along the second \
             flanking edge: {meet:?} vs {along_second:?}",
        );
        assert!(
            (along_second - along_first).length() > 1.0e-3,
            "fixture sanity: the two candidate edges must give different answers",
        );
    }

    // Past both spokes' lengths the clamp certainly fires, and it lands `5e-5` short
    // of the **second** edge's far endpoint. The two spokes are a whole unit apart in
    // length, so a slide clamped against the first would miss by ~1.
    let second_length = (mesh.edge_far_position(second, v) - v_pos).length();
    let first_length = (mesh.edge_far_position(first, v) - v_pos).length();
    let travelled = (mesh.solve_meet_point(v, v_pos, &sector, 5.0) - v_pos).length();
    assert!(
        (travelled - (second_length - SLIDE_CLAMP_GAP)).abs() < 1.0e-5,
        "the doubled-back slide is clamped against the *second* edge's length \
         ({second_length}), travelled {travelled}",
    );
    assert!(
        (first_length - second_length).abs() > 0.5,
        "fixture sanity: the two spokes' lengths must discriminate the clamp target",
    );
}

/// BEVEL.md §7.6.2 states one `EPS_ANG = 2°` for **both** halves of the ladder, and
/// the junction half must read that same band. Stating it as a tolerance in *cosine*
/// units — `cos θ_dir ≥ 1 − 1e-4` — is the same rule at only `0.81°`, and it hands the
/// `0.81°–2°` window to the general two-line solve. That is not a quanta-scale
/// difference: in that window the two offset lines are near-coincident, so their
/// crossing runs off to tens of units on a unit-scale fan.
///
/// `sliver_sector_fan`'s doubled-back sector sits at `θ_dir ≈ 179.045°`, i.e. `0.955°`
/// from the `180°` end — inside the guide's band and outside the cosine reading — so it
/// is exactly the window's witness.
#[test]
fn junction_ladder_fires_across_the_whole_two_degree_band() {
    let (mesh, v, [a, b, c, _d]) = sliver_sector_fan();
    let v_pos = mesh.vertex_position(v);
    let spokes: Vec<EdgeKey> = [a, b, c]
        .iter()
        .map(|&far| mesh.query_edge_exists(v, far).expect("beveled spoke"))
        .collect();
    let is_beveled = |edge: EdgeKey| spokes.contains(&edge);

    let corners = mesh.vertex_bevel_corner_ring(v, &is_beveled);
    let edge_a = mesh.query_edge_exists(v, a).expect("spoke A");
    let edge_b = mesh.query_edge_exists(v, b).expect("spoke B");
    let sector = sector_split(&corners, &is_beveled)
        .into_iter()
        .find(|s| {
            s.bounding_edges.len() == 2
                && s.bounding_edges.contains(&edge_a)
                && s.bounding_edges.contains(&edge_b)
        })
        .expect("the A↔B sliver sector");
    let layout = mesh
        .sector_layout(v, v_pos, &sector)
        .expect("a two-flank sector has a layout");

    let direction_angle = unit_chord_angle_between(-layout.dir_first, layout.dir_second);
    let from_anti_parallel = std::f32::consts::PI - direction_angle;
    assert!(
        from_anti_parallel > 0.0141 && from_anti_parallel < MEET_DIRECTION_EPS_ANGLE,
        "fixture must sit in the 0.81°–2° window, got {}° from anti-parallel",
        from_anti_parallel.to_degrees(),
    );
    // The narrower reading, spelled out: `cos θ_dir ≤ −1 + 1e-4` does not fire here.
    let cos_dir = (-layout.dir_first).dot(layout.dir_second);
    assert!(
        cos_dir > -1.0 + 1.0e-4,
        "fixture sanity: the cosine-units gate must miss this sector, cos = {cos_dir}",
    );

    let offset = 0.5_f32;
    let meet = mesh.solve_meet_point(v, v_pos, &sector, offset);
    let along_second = mesh.slide_along_incident_edge(v, sector.bounding_edges[1], offset);
    assert!(
        (meet - along_second).length() < 1.0e-6,
        "the ladder must claim the whole 2° band, got {meet:?}",
    );
    assert!(
        (meet - v_pos).length() <= offset + 1.0e-6,
        "a ladder placement never travels further than the offset, got {meet:?}",
    );

    // What the general solve does with the same sector, which is what the narrow gate
    // would have returned: two near-coincident offset lines crossing far off the fan.
    let extrapolated = match intersect_lines(
        &Line {
            point: v_pos + layout.perp_first * offset,
            direction: layout.dir_first,
        },
        &Line {
            point: v_pos + layout.perp_second * offset,
            direction: layout.dir_second,
        },
    ) {
        LineMeet::Crossing(point)
        | LineMeet::Skew {
            midpoint: point, ..
        } => point,
        LineMeet::Collinear => panic!("the near-coincident lines still solve as a crossing"),
    };
    assert!(
        (extrapolated - v_pos).length() > 10.0,
        "fixture sanity: the general solve must visibly extrapolate here, got \
         {extrapolated:?}",
    );
}

/// BEVEL.md §7.6.2's worked station, rebuilt from the guide's own description: a flat
/// closed fan about `v = (0,0,0)` whose beveled flanks `C = (2, −flank_drop, 0)` and
/// `A = (−2, 0, 0)` bound a **reflex** planar sector of normal `+z`, at a direction
/// angle of `atan(flank_drop / 2)` — `2.29061°` at `flank_drop = 0.08`, just outside
/// the `2°` ladder band, so the general two-line solve is what runs.
///
/// The reflex sector is filled by the single quad `v → C → X → A`, which makes it a
/// §7.6.2 **case-1** sector (no in-between edges); the unbeveled leg `L = (0,−1,0)`
/// splits the opposite side into two triangles so the fan is closed.
fn reflex_flank_fan(flank_drop: f32) -> (LoopMesh, VertexKey, [VertexKey; 3]) {
    let mut mesh = LoopMesh::new();
    let v = mesh.create_vertex(Vec3::ZERO);
    let c = mesh.create_vertex(Vec3::new(2.0, -flank_drop, 0.0));
    let a = mesh.create_vertex(Vec3::new(-2.0, 0.0, 0.0));
    let x = mesh.create_vertex(Vec3::new(0.0, 3.0, 0.0));
    let l = mesh.create_vertex(Vec3::new(0.0, -1.0, 0.0));
    mesh.face_create_verts(&[v, c, x, a], FaceCreateFlags::NONE);
    mesh.face_create_verts(&[v, a, l], FaceCreateFlags::NONE);
    mesh.face_create_verts(&[v, l, c], FaceCreateFlags::NONE);
    (mesh, v, [c, a, l])
}

/// The reflex (case-1, planar) sector of [`reflex_flank_fan`], with `C` first and `A`
/// second — the order §7.6.2's worked station states, `C` taken *into* `v` and `A`
/// *out* of it.
fn reflex_flank_sector(flank_drop: f32) -> (LoopMesh, VertexKey, Sector) {
    let (mesh, v, [c, a, _l]) = reflex_flank_fan(flank_drop);
    let edge_c = mesh.query_edge_exists(v, c).expect("flank C");
    let edge_a = mesh.query_edge_exists(v, a).expect("flank A");
    let is_beveled = |edge: EdgeKey| edge == edge_c || edge == edge_a;
    let corners = mesh.vertex_bevel_corner_ring(v, &is_beveled);
    let sector = sector_split(&corners, &is_beveled)
        .into_iter()
        .find(|s| s.bounding_edges == vec![edge_c, edge_a])
        .expect("the reflex C→A sector, in the guide's stated order");
    (mesh, v, sector)
}

/// The canonical comparison grid quantises positions at `1e-5` (MESH_COMPARISON.md,
/// "Tolerance is a grid, not a metric"), which is the resolution every one of
/// §7.6.2's arithmetic axes is measured in.
fn canonical_quanta(coordinate: f32) -> i64 {
    (coordinate / 1.0e-5).round() as i64
}

/// §7.6.2's worked station, evaluated from the guide's closed form rather than from
/// any backend: with `d = flank_drop`, `L = sqrt(4 + d²)` and the sector normal `+z`,
/// the two offset lines' exact crossing is
/// `x = offset · (d/L − 2(1 − 2/L)/d)`, `y = offset`, `z = 0`.
fn worked_station_crossing_x(flank_drop: f64, offset: f64) -> f64 {
    let length = (4.0 + flank_drop * flank_drop).sqrt();
    offset * (flank_drop / length - 2.0 * (1.0 - 2.0 / length) / flank_drop)
}

/// §7.6.2's three worked stations, evaluated on [`reflex_flank_fan`] rebuilt from the
/// guide's own description of them. All three sit **just outside** the `2°` ladder
/// band, where the two-line solve is ill-conditioned enough that algebraically
/// identical spellings land on different sides of the canonical grid's rounding
/// boundary — so the quantised `x` is the whole assertion.
///
/// The expected quanta are the guide's, not a measurement: `1499`, `4725`, `673`. Two
/// of the three are on the *far* side of the exact-arithmetic rounding (the closed
/// form gives `4725.505` and `673.501`), which is the guide's point that accuracy is
/// not the argument — these are simply the values the stated arithmetic produces.
#[test]
fn junction_planar_case1_meet_mirrors_the_guides_worked_stations() {
    for (flank_drop, offset, expected_quanta) in [
        (0.08_f32, 0.75_f32, 1499_i64),
        (0.317, 0.60, 4725),
        (0.077, 0.35, 673),
    ] {
        let (mesh, vertex, sector) = reflex_flank_sector(flank_drop);
        let vertex_position = mesh.vertex_position(vertex);
        let layout = mesh
            .sector_layout(vertex, vertex_position, &sector)
            .expect("a two-flank sector has a layout");

        let direction_angle = unit_chord_angle_between(-layout.dir_first, layout.dir_second);
        assert!(
            !is_degenerate_direction_angle(direction_angle),
            "fixture sanity: {}° must sit outside the ladder band, or the general \
             solve never runs",
            direction_angle.to_degrees(),
        );
        assert!(
            !layout.non_planar,
            "fixture sanity: the reflex sector must classify as planar case 1",
        );
        assert_eq!(
            layout.sector_normal,
            Vec3::Z,
            "fixture sanity: the sector plane is `+z`",
        );

        let meet = mesh.solve_meet_point(vertex, vertex_position, &sector, offset);
        assert_eq!(meet.y, offset, "the second offset line is `y = offset`");
        assert_eq!(
            meet.z, 0.0,
            "a planar `z = 0` sector cannot leave the plane"
        );
        assert_eq!(
            canonical_quanta(meet.x),
            expected_quanta,
            "flank_drop {flank_drop} at offset {offset}: got {} quanta ({meet:?})",
            f64::from(meet.x) / 1.0e-5,
        );

        let exact = worked_station_crossing_x(f64::from(flank_drop), f64::from(offset));
        assert!(
            (f64::from(meet.x) - exact).abs() < 0.25e-5,
            "the solve must still track the closed form to well under a quantum: \
             {} vs {} quanta",
            f64::from(meet.x) / 1.0e-5,
            exact / 1.0e-5,
        );
    }
}

/// The Gram `2 × 2` solve, in **both** of the variants §7.6.2's mirroring table gives
/// the solve axis its two rows for — read at the first line's parameter (the `4.2%`
/// row) and averaged over both lines' closest points (the `3.9%` row) — each shown to
/// move the boundvert a whole quantum at one station of this fixture family.
///
/// The station is `flank_drop = 0.074` at `offset = 1.0`: a direction angle of
/// `2.119°`, just outside the ladder band like the guide's own three, whose closed
/// form is `1849.367` quanta. That is `0.13` of a quantum below the `1849.5` rounding
/// boundary — the same order of margin as the guide's `1499.4005` station, which is
/// what makes the station able to say anything at all. The stated arithmetic lands
/// `1849.404` and both Gram spellings land past the tie (`1849.519` and `1849.552`).
///
/// Neither Gram variant parts *any* of the guide's own three stations: at `1499` the
/// point-on-first spelling reads `1499.479` and rounds the same way. A sweep of this
/// family (3600 stations over `flank_drop ∈ [0.07, 0.426]`, `offset ∈ [0.2, 2.15]`)
/// parts on `2.9%` for the point-on-first variant and `3.2%` for the averaged one —
/// the same order as the guide's `4.2%` / `3.9%` on its own different sweep.
#[test]
fn junction_planar_case1_meet_discriminates_both_gram_spellings() {
    let (mesh, vertex, sector) = reflex_flank_sector(0.074);
    let vertex_position = mesh.vertex_position(vertex);
    let offset = 1.0_f32;
    let layout = mesh
        .sector_layout(vertex, vertex_position, &sector)
        .expect("a two-flank sector has a layout");
    let dir_first = -(layout.far_first - vertex_position);
    let dir_second = layout.far_second - vertex_position;
    let base_first = vertex_position + layout.perp_first * offset;
    let base_second = vertex_position + layout.perp_second * offset;

    let separation = base_first - base_second;
    let along_first = dir_first.dot(dir_first);
    let alignment = dir_first.dot(dir_second);
    let along_second = dir_second.dot(dir_second);
    let first_projection = dir_first.dot(separation);
    let second_projection = dir_second.dot(separation);
    let denominator = along_first * along_second - alignment * alignment;
    let first_parameter =
        (alignment * second_projection - along_second * first_projection) / denominator;
    let second_parameter =
        (along_first * second_projection - alignment * first_projection) / denominator;
    let gram_on_first = base_first + dir_first * first_parameter;
    let gram_midpoint = (gram_on_first + (base_second + dir_second * second_parameter)) * 0.5;

    let meet = mesh.solve_meet_point(vertex, vertex_position, &sector, offset);
    assert_eq!(
        canonical_quanta(meet.x),
        1849,
        "the cross-product form lands at {} quanta",
        f64::from(meet.x) / 1.0e-5,
    );
    assert_eq!(
        canonical_quanta(gram_on_first.x),
        1850,
        "fixture sanity: the Gram point-on-first spelling must cross the tie, it is at \
         {} quanta",
        f64::from(gram_on_first.x) / 1.0e-5,
    );
    assert_eq!(
        canonical_quanta(gram_midpoint.x),
        1850,
        "fixture sanity: the Gram midpoint spelling must cross the tie, it is at {} \
         quanta",
        f64::from(gram_midpoint.x) / 1.0e-5,
    );
}

/// The two contrast spellings §7.6.2 names, each shown to *move* the boundvert at the
/// station the guide measured it at. Without this the pins above would pass for a
/// half-migrated solve that happened to agree on these three sectors.
///
/// The guide's own figures: the midpoint spelling lands on `1500` where the stated
/// arithmetic lands on `1499`, and unitising the flanking directions before building
/// the offset lines lands on `674` / `4726` where the raw ones land on `673` / `4725`.
#[test]
fn junction_planar_case1_meet_discriminates_the_carriage_and_the_solve() {
    let station = |flank_drop: f32, offset: f32| {
        let (mesh, vertex, sector) = reflex_flank_sector(flank_drop);
        let vertex_position = mesh.vertex_position(vertex);
        let layout = mesh
            .sector_layout(vertex, vertex_position, &sector)
            .expect("a two-flank sector has a layout");
        let raw_first = layout.far_first - vertex_position;
        let raw_second = layout.far_second - vertex_position;
        let first_base = vertex_position + layout.perp_first * offset;
        let second_base = vertex_position + layout.perp_second * offset;
        (
            mesh.solve_meet_point(vertex, vertex_position, &sector, offset),
            layout.sector_normal,
            raw_first,
            raw_second,
            first_base,
            second_base,
            vertex_position,
        )
    };

    // The solve axis: the Gram `2 × 2` system solved on both lines and averaged, which
    // is the same algebra and a different computation.
    let (meet, _, raw_first, raw_second, first_base, second_base, _) = station(0.08, 0.75);
    let midpoint_spelling = {
        let a = -raw_first;
        let b = raw_second;
        let separation = first_base - second_base;
        let along_first = a.dot(a);
        let alignment = a.dot(b);
        let along_second = b.dot(b);
        let first_projection = a.dot(separation);
        let second_projection = b.dot(separation);
        let denominator = along_first * along_second - alignment * alignment;
        let first_parameter =
            (alignment * second_projection - along_second * first_projection) / denominator;
        let second_parameter =
            (along_first * second_projection - alignment * first_projection) / denominator;
        ((first_base + a * first_parameter) + (second_base + b * second_parameter)) * 0.5
    };
    // Read the margins before trusting this pair. The stated arithmetic lands at
    // `1499.440` quanta, `0.060` clear of the `1499.5` boundary. The midpoint
    // spelling lands at `1499.49994` — its `f32` quotient by `1e-5` is *exactly*
    // `1499.5`, so it rounds away from zero to `1500` on a margin of `1e-4` of a
    // quantum. It is a genuine discriminator and a knife-edge one: any unrelated
    // last-bit change to the perpendiculars can flip it, and a flip here is not by
    // itself evidence of a regression. `..._discriminates_both_gram_spellings` above
    // is the robust station for this axis (margins `0.13` and `0.019`); this one is
    // kept because it is the contrast the guide itself quotes.
    assert_eq!(canonical_quanta(meet.x), 1499);
    assert_eq!(
        canonical_quanta(midpoint_spelling.x),
        1500,
        "fixture sanity: the midpoint spelling must land on the other side of the tie",
    );

    // The carriage axis: unit flanking directions into the perpendiculars *and* into
    // the offset lines' second points, with the solve untouched.
    for (flank_drop, offset, unit_quanta) in [(0.077_f32, 0.35_f32, 674_i64), (0.317, 0.60, 4726)] {
        let (meet, normal, raw_first, raw_second, _, _, vertex_position) =
            station(flank_drop, offset);
        let unit_first = raw_first.normalize();
        let unit_second = raw_second.normalize();
        let unit_carriage = intersect_lines(
            &Line {
                point: vertex_position
                    + normalise_at_direction_floor(normal.cross(unit_first)) * offset,
                direction: -unit_first,
            },
            &Line {
                point: vertex_position
                    + normalise_at_direction_floor(unit_second.cross(normal)) * offset,
                direction: unit_second,
            },
        );
        let LineMeet::Crossing(unit_meet) = unit_carriage else {
            panic!("the unit carriage still solves as a crossing");
        };
        assert_eq!(canonical_quanta(meet.x), unit_quanta - 1);
        assert_eq!(
            canonical_quanta(unit_meet.x),
            unit_quanta,
            "fixture sanity: unitising the flanks must move this station a quantum",
        );
    }
}

/// The apex of [`single_face_reflex_junction`] with all three spokes beveled, and
/// its three sectors keyed by their flank pair: `(first, second)` in §7.6.2's own
/// order, the first flank read *into* `v` and the second *out* of it.
///
/// Both convex sectors here are planar and **tilted** — which is the whole point of
/// the station. On an axis-aligned sector plane the edge span and the face's own
/// normal both collapse onto a signed unit axis and agree to the bit almost always
/// (`0.8%` of a sweep, against `17.2%` on randomly oriented planes), so a fixture
/// built in one coordinate plane is structurally blind to which of the two the
/// implementation picks.
fn tilted_apex_sectors() -> (LoopMesh, VertexKey, [(EdgeKey, EdgeKey); 3], Vec<Sector>) {
    let (mesh, apex, [va, vb, vc]) = single_face_reflex_junction();
    let is_beveled = |edge: EdgeKey| edge == va || edge == vb || edge == vc;
    let corners = mesh.vertex_bevel_corner_ring(apex, &is_beveled);
    let sectors = sector_split(&corners, &is_beveled);
    (mesh, apex, [(va, vb), (vb, vc), (vc, va)], sectors)
}

/// §7.6.2 "The edge-span normal, spelled out": on a **planar** case-1 sector the
/// sector normal is still the normalised raw edge span, sign-fixed against the
/// sector's face — the face's own normal is **not** licensed there. There is no
/// coplanarity test in this branch; the only face-normal substitution cases 1 and 2
/// allow is the shallow direction-angle exception, and this station's `143.6°` is
/// nowhere near it.
///
/// The two spellings are the same *direction* on a flat sector and different `f32`
/// vectors — the face's is summed over vertices the two flanks never touch — so the
/// divergence is **sub-quantum where it is born**: both quantise to the same `19480`
/// on the canonical `1e-5` grid. It becomes a canonical divergence only downstream,
/// through the multi-segment profile grid and the inner miter. Hence the assertions
/// are on `f32` bits.
///
/// The guide's oracle figures: the `+x` sector's boundvert at `offset = 0.2`,
/// `segments = 1` is `(0.19480430, 0.46936694, 0.60907954)`, and substituting the
/// face's own unit normal raises `x` by one ulp to `0x3E477AC8`. On the mirror `−x`
/// sector the same substitution moves `y` instead, `0x3EF050DB → 0x3EF050DC` — the
/// component it lands in follows the sector's orientation, not the rule.
#[test]
fn junction_planar_case1_sector_normal_is_the_edge_span_not_the_face_normal() {
    let (mesh, apex, [minus_x, plus_x, _reflex], sectors) = tilted_apex_sectors();
    let apex_position = mesh.vertex_position(apex);
    let offset = 0.2_f32;

    for (flanks, component, stated, face_normal_bits) in [
        (plus_x, 0_usize, 0x3E47_7AC7_u32, 0x3E47_7AC8_u32),
        (minus_x, 1, 0x3EF0_50DB, 0x3EF0_50DC),
    ] {
        let sector = sectors
            .iter()
            .find(|candidate| candidate.bounding_edges == vec![flanks.0, flanks.1])
            .expect("both convex apex sectors are bounded by two beveled spokes");
        let layout = mesh
            .sector_layout(apex, apex_position, sector)
            .expect("a two-flank sector has a layout");
        assert!(
            !layout.non_planar,
            "fixture sanity: this is a case-1 sector, no in-between legs at all",
        );
        let direction_angle = unit_chord_angle_between(-layout.dir_first, layout.dir_second);
        assert!(
            direction_angle > TERMINAL_DIRECTION_SMALL_ANGLE,
            "fixture sanity: {}° must clear the shallow exception, which is the one \
             branch that *does* take the face normal",
            direction_angle.to_degrees(),
        );

        let meet = mesh.solve_meet_point(apex, apex_position, sector, offset);
        assert_eq!(
            meet[component].to_bits(),
            stated,
            "the normalised raw edge span places the boundvert at {meet:?}",
        );

        let face_unit = orient_normal(
            mesh.face_plane_normal(sector.corners[0].1),
            layout.normal_first,
        );
        assert_ne!(
            face_unit, layout.normal_first,
            "fixture sanity: on a tilted sector plane the two spellings must part in \
             their last bits, or this station proves nothing",
        );
        let substituted = meet_with_sector_normal(&mesh, apex, sector, offset, face_unit);
        assert_eq!(
            substituted[component].to_bits(),
            face_normal_bits,
            "fixture sanity: the face-normal substitution must move this component",
        );
    }

    // The +x sector's full boundvert, all three components, as the guide states it.
    let plus_x_sector = sectors
        .iter()
        .find(|candidate| candidate.bounding_edges == vec![plus_x.0, plus_x.1])
        .expect("the +x apex sector");
    assert_eq!(
        mesh.solve_meet_point(apex, apex_position, plus_x_sector, offset),
        Vec3::new(0.194_804_3, 0.469_366_94, 0.609_079_54),
    );
}

/// §7.6.2 (c): the span is scaled to unit length **before** it becomes the sector
/// normal, and that is not cosmetic even though the perpendicular built from it is
/// normalised anyway — `cross(direction, k · span)` and `k · cross(direction, span)`
/// differ in their last bits.
///
/// And §7.6.2 (a)'s caution, in its concrete form: unitising the two flanks before
/// the span cross product is the guess an implementor reaches for first, and at this
/// station it reproduces the stated spelling **bit for bit** — so it cannot be what
/// moves anything here. Pinning that keeps the station honest about which axis is
/// doing the work.
#[test]
fn junction_planar_case1_edge_span_is_normalised_before_it_becomes_the_sector_normal() {
    let (mesh, apex, [_minus_x, plus_x, _reflex], sectors) = tilted_apex_sectors();
    let apex_position = mesh.vertex_position(apex);
    let offset = 0.2_f32;
    let sector = sectors
        .iter()
        .find(|candidate| candidate.bounding_edges == vec![plus_x.0, plus_x.1])
        .expect("the +x apex sector");
    let layout = mesh
        .sector_layout(apex, apex_position, sector)
        .expect("a two-flank sector has a layout");
    let raw_first = layout.far_first - apex_position;
    let raw_second = layout.far_second - apex_position;

    let unnormalised_span = {
        let span = raw_first.cross(raw_second);
        if span.dot(layout.normal_first) < 0.0 {
            -span
        } else {
            span
        }
    };
    assert_eq!(
        meet_with_sector_normal(&mesh, apex, sector, offset, unnormalised_span)
            .x
            .to_bits(),
        0x3E47_7AC8,
        "fixture sanity: the un-normalised span must land on the same wrong ulp as \
         the face-normal substitution",
    );

    // Same cross product, unit flanks in place of raw ones. §7.6.2 (b) makes the
    // cross *order* free, so this contrast isolates the flanks alone.
    let unit_flank_span = orient_normal(
        layout.dir_first.cross(layout.dir_second),
        layout.normal_first,
    );
    assert_eq!(
        meet_with_sector_normal(&mesh, apex, sector, offset, unit_flank_span),
        mesh.solve_meet_point(apex, apex_position, sector, offset),
        "the unit-flank span agrees bit for bit at this station",
    );
}

/// The two §7.6.2 offset lines rebuilt exactly as `solve_meet_point` builds them —
/// raw flanks, role-signed perpendiculars, second points stepped from the base — but
/// about a *substituted* sector normal, so a contrast spelling can be tested one
/// quantity deep without touching anything downstream of it.
fn meet_with_sector_normal(
    mesh: &LoopMesh,
    apex: VertexKey,
    sector: &Sector,
    offset: f32,
    normal: Vec3,
) -> Vec3 {
    let apex_position = mesh.vertex_position(apex);
    let layout = mesh
        .sector_layout(apex, apex_position, sector)
        .expect("a two-flank sector has a layout");
    let raw_first = layout.far_first - apex_position;
    let raw_second = layout.far_second - apex_position;
    let crossing = intersect_lines(
        &Line {
            point: apex_position + normalise_at_direction_floor(normal.cross(raw_first)) * offset,
            direction: apex_position - layout.far_first,
        },
        &Line {
            point: apex_position + normalise_at_direction_floor(raw_second.cross(normal)) * offset,
            direction: layout.far_second - apex_position,
        },
    );
    match crossing {
        LineMeet::Crossing(point)
        | LineMeet::Skew {
            midpoint: point, ..
        } => point,
        LineMeet::Collinear => panic!("the two offset lines cross at this station"),
    }
}

/// §7.6.2's parallel bail is an **exact** `== 0` on `|a × b|²` — no epsilon anywhere.
/// A tolerant bail would divert nearly-but-not-quite-parallel offset lines into the
/// `Collinear` recovery, which answers with the first line's *base point* rather than
/// with any crossing: a far larger error than the rounding the arithmetic is about.
#[test]
fn two_line_solve_bails_to_collinear_only_on_an_exact_zero_cross() {
    let base_first = Vec3::new(0.0, 0.25, 0.0);
    let base_second = Vec3::ZERO;
    let direction = Vec3::new(1.0, 0.0, 0.0);

    let exactly_parallel = intersect_lines(
        &Line {
            point: base_first,
            direction,
        },
        &Line {
            point: base_second,
            direction: direction * 2.0,
        },
    );
    assert!(
        matches!(exactly_parallel, LineMeet::Collinear),
        "an exactly parallel pair has no isolated crossing",
    );

    // A tenth of a milliradian of turn: `sin²θ = 1e-10`, an order of magnitude *under*
    // any determinant tolerance a solve of this shape would plausibly carry, yet the
    // cross product is not zero, so the crossing is the answer — far off though it is.
    let turn = 1.0e-5_f32;
    let nearly_parallel = intersect_lines(
        &Line {
            point: base_first,
            direction,
        },
        &Line {
            point: base_second,
            direction: Vec3::new(turn.cos(), turn.sin(), 0.0),
        },
    );
    let LineMeet::Crossing(meet) = nearly_parallel else {
        panic!("a non-zero cross product must still solve as a crossing");
    };
    assert!(
        meet.x > 1.0e4,
        "the near-parallel crossing runs far off, and that is the defined answer: \
         {meet:?}",
    );
}

/// §7.6.2's `EPS_COPLANAR` is **absolute**, and applied to the **un-normalised**
/// triple product, so what it admits scales with the flanking lengths:
/// `|coplanarity| = |dir_first|·|dir_second|·sin(θ_dir)·gap`. Scaling both lines'
/// *directions* by ten — leaving both base points, the perpendicular gap between the
/// lines and the angle between them untouched, so the geometry is literally the same
/// pair of lines — multiplies the triple product by a hundred and moves the pair
/// across the bound.
///
/// This is the axis §7.6.2 warns loudest about ("do not 'improve' it into a
/// normalised gap test"), and it is the reason a *planar* sector on a large mesh can
/// reach the skew branch on rounding noise alone.
#[test]
fn two_line_solve_crossing_bound_is_absolute_on_the_raw_triple_product() {
    let separation = 5.0e-8_f32;
    let classify = |scale: f32| {
        let first = Line {
            point: Vec3::ZERO,
            direction: Vec3::X * scale,
        };
        let second = Line {
            point: Vec3::new(0.0, 0.0, separation),
            direction: Vec3::Y * scale,
        };
        intersect_lines(&first, &second)
    };

    assert!(
        matches!(classify(1.0), LineMeet::Crossing(_)),
        "at unit direction lengths the triple product is `{separation:e}`, inside the \
         bound",
    );
    assert!(
        matches!(classify(10.0), LineMeet::Skew { .. }),
        "ten times the direction lengths is a hundred times the triple product, and \
         the same geometry classifies as skew",
    );
}

/// §7.6.2's skew answer solves nothing new: the second line is translated by the
/// common-perpendicular `gap` until the pair is coplanar, and the **crossing formula
/// is then applied unchanged**, yielding a point on the first line. That is not the
/// same computation as reading a Gram `2 × 2` system's first-line parameter, even
/// though the two are the same point in exact arithmetic — and on a shallow pair the
/// difference is thirteen canonical quanta, well past the "last bit" the crossing
/// case's contrasts live at.
#[test]
fn two_line_solve_skew_answer_is_the_translated_crossing_not_a_gram_closest_pair() {
    let line_first = Line {
        point: Vec3::new(0.0, 0.31, 0.0),
        direction: Vec3::new(2.4, 0.061, 0.0),
    };
    let line_second = Line {
        point: Vec3::new(0.0, 0.0, 0.017),
        direction: Vec3::new(-1.9, 0.1, 0.0),
    };
    let LineSolve::Skew { on_first, .. } = solve_two_lines(&line_first, &line_second) else {
        panic!("fixture sanity: the pair must classify as skew");
    };

    // §7.6.2's arithmetic, longhand: translate the second line by `gap`, re-derive
    // its two points there, and run the crossing formula on the coplanar pair.
    let by_translation = {
        let dir_first = (line_first.point + line_first.direction) - line_first.point;
        let dir_second = (line_second.point + line_second.direction) - line_second.point;
        let direction_cross = dir_first.cross(dir_second);
        let gap = direction_cross
            * ((line_first.point - line_second.point).dot(direction_cross)
                / direction_cross.dot(direction_cross));
        let moved_base = line_second.point + gap;
        let moved_direction = (line_second.point + line_second.direction + gap) - moved_base;
        let moved_cross = dir_first.cross(moved_direction);
        let along = (moved_base - line_first.point)
            .cross(moved_direction)
            .dot(moved_cross)
            / moved_cross.dot(moved_cross);
        line_first.point + dir_first * along
    };
    assert_eq!(
        on_first, by_translation,
        "the skew answer must be the translated pair's crossing, bit for bit",
    );

    // The contrast: the Gram `2 × 2` closest point on the first line, which is the
    // same point in exact arithmetic and a different one in `f32`.
    let by_gram = {
        let a = line_first.direction;
        let b = line_second.direction;
        let separation = line_first.point - line_second.point;
        let denominator = a.dot(a) * b.dot(b) - a.dot(b) * a.dot(b);
        let parameter = (a.dot(b) * b.dot(separation) - b.dot(b) * a.dot(separation)) / denominator;
        line_first.point + a * parameter
    };
    assert!(
        (on_first - by_gram).length() > 1.0e-5,
        "fixture sanity: the Gram spelling must be a measurably different answer \
         here, it is {} away",
        (on_first - by_gram).length(),
    );
}

/// §7.6.2's "Face-drop does not run" reading, on the sector class it applies to: a
/// **planar** case-1 sector that the absolute `EPS_COPLANAR` bound classifies as skew
/// takes "the closest-approach point on the **first** line — no midpoint, no
/// projection".
///
/// A planar sector reaches skew only through the previous test's axis: the sector's
/// two offset lines are exactly coplanar by construction (both perpendiculars are
/// square to the one shared normal), so the triple product is pure `f32` noise — but
/// it grows as the cube of the mesh scale, and past about `12 ×` the fixture's own
/// size it clears the bound. The gap that noise implies is far below a canonical
/// quantum (here it underflows to zero, so the two skew readings coincide bit for
/// bit and the *value* cannot discriminate them); what this pins is that the meet is
/// taken from the first-line reading and that the face-drop is gated on the sector's
/// planarity rather than on the solve's own verdict.
#[test]
fn junction_planar_sector_skew_reading_is_the_point_on_the_first_line() {
    // The guide's reflex fan, scaled up until its planar sector's triple product
    // clears the bound, and tilted off the axes so the coordinates carry the rounding
    // that produces it at all.
    let scale = 16.0_f32;
    let tilt = glam::Quat::from_axis_angle(Vec3::new(1.0, 1.0, 1.0).normalize(), 0.37);
    let mut mesh = LoopMesh::new();
    let place = |x: f32, y: f32| tilt * (Vec3::new(x, y, 0.0) * scale);
    let vertex = mesh.create_vertex(place(0.0, 0.0));
    let flank_c = mesh.create_vertex(place(2.0, -0.08));
    let flank_a = mesh.create_vertex(place(-2.0, 0.0));
    let rim = mesh.create_vertex(place(0.0, 3.0));
    let leg = mesh.create_vertex(place(0.0, -1.0));
    mesh.face_create_verts(&[vertex, flank_c, rim, flank_a], FaceCreateFlags::NONE);
    mesh.face_create_verts(&[vertex, flank_a, leg], FaceCreateFlags::NONE);
    mesh.face_create_verts(&[vertex, leg, flank_c], FaceCreateFlags::NONE);

    let edge_c = mesh.query_edge_exists(vertex, flank_c).expect("flank C");
    let edge_a = mesh.query_edge_exists(vertex, flank_a).expect("flank A");
    let is_beveled = |edge: EdgeKey| edge == edge_c || edge == edge_a;
    let corners = mesh.vertex_bevel_corner_ring(vertex, &is_beveled);
    let sector = sector_split(&corners, &is_beveled)
        .into_iter()
        .find(|s| s.bounding_edges == vec![edge_c, edge_a])
        .expect("the reflex C→A sector");

    let vertex_position = mesh.vertex_position(vertex);
    let offset = 0.75 * scale;
    let layout = mesh
        .sector_layout(vertex, vertex_position, &sector)
        .expect("a two-flank sector has a layout");
    assert!(
        !layout.non_planar,
        "fixture sanity: the sector must classify as planar case 1, so the face-drop \
         is gated off",
    );
    assert!(
        !layout.drop_candidates.is_empty(),
        "fixture sanity: the sector does offer a drop candidate, so gating the drop \
         off is a choice and not an absence",
    );

    let line_first = Line {
        point: vertex_position + layout.perp_first * offset,
        direction: vertex_position - layout.far_first,
    };
    let line_second = Line {
        point: vertex_position + layout.perp_second * offset,
        direction: layout.far_second - vertex_position,
    };
    let LineMeet::Skew { on_first, .. } = intersect_lines(&line_first, &line_second) else {
        panic!(
            "fixture sanity: at {scale}× the fixture's scale the planar sector's own \
             rounding must clear the absolute bound",
        );
    };

    assert_eq!(
        mesh.solve_meet_point(vertex, vertex_position, &sector, offset),
        on_first,
        "a planar sector's skew pair takes the point on the first line, untouched by \
         the face-drop",
    );
}

/// The crossing answer is a parameter along the **first** line, and the skew answer
/// fills the same slot: `on_second` is `on_first` carried by the common-perpendicular
/// `gap`, and the midpoint `intersect_lines` reports is `on_first − gap/2`. Swapping
/// the two lines therefore swaps which line the answer rides — it is not symmetric.
#[test]
fn two_line_solve_rides_the_first_line_and_reports_a_mutual_skew_pair() {
    let line_first = Line {
        point: Vec3::new(0.0, 0.0, 0.0),
        direction: Vec3::new(2.0, 0.3, 0.0),
    };
    let line_second = Line {
        point: Vec3::new(0.5, -1.0, 0.4),
        direction: Vec3::new(0.1, 1.7, 0.2),
    };

    let (on_first, on_second) = closest_approach(&line_first, &line_second).expect("a skew pair");
    let off_first = (on_first - line_first.point).cross(line_first.direction);
    let off_second = (on_second - line_second.point).cross(line_second.direction);
    assert!(
        off_first.length() < 1.0e-6 && off_second.length() < 1.0e-6,
        "each closest point must lie on its own line: {off_first:?} / {off_second:?}",
    );
    assert!(
        (on_first - on_second).length() > 1.0e-2,
        "fixture sanity: the pair must be genuinely skew",
    );

    let LineMeet::Skew {
        on_first: reported_on_first,
        midpoint,
    } = intersect_lines(&line_first, &line_second)
    else {
        panic!("a skew pair must report as skew");
    };
    assert_eq!(
        reported_on_first, on_first,
        "the skew outcome reports both of §7.6.2's readings, and the first-line one \
         is the same point `closest_approach` hands back",
    );
    assert!(
        (midpoint - (on_first + on_second) * 0.5).length() < 1.0e-6,
        "the reported skew point is the midpoint of the mutual pair",
    );

    let (swapped_first, _) = closest_approach(&line_second, &line_first).expect("a skew pair");
    assert!(
        (swapped_first - on_second).length() < 1.0e-5,
        "swapping the pair order rides the other line — the solve is not symmetric",
    );
}

/// §7.6.2 "The arithmetic, not just the floor": the shared normalisation guards the
/// **squared** length with a **strict** `>`, so a squared length exactly on the floor
/// takes the zero branch, and above it the unit vector is one `sqrt`, one scalar
/// reciprocal and three multiplies — not three divisions.
#[test]
fn shared_normalise_scales_by_one_reciprocal_under_a_strict_squared_floor() {
    // The two spellings differ here in the last bit of two components, which is the
    // whole of the 2.5%-of-shallow-sectors effect the guide measures.
    let discriminating = Vec3::new(0.3, 0.7, -1.1);
    let reciprocal = discriminating * (1.0 / discriminating.length_squared().sqrt());
    assert_eq!(normalise_at_direction_floor(discriminating), reciprocal);
    assert_ne!(
        reciprocal,
        discriminating / discriminating.length(),
        "fixture sanity: this vector must actually part the two spellings",
    );

    // A vector whose squared length is *exactly* the floor. The guard is `>`, so it
    // takes the zero branch; a `>=` would normalise it.
    let on_floor = Vec3::new(3.1618642e-18, 5.113861e-20, 0.0);
    assert_eq!(
        on_floor.length_squared(),
        SLIDE_DIRECTION_FLOOR_SQ,
        "fixture sanity: this vector must sit exactly on the floor",
    );
    assert_eq!(normalise_at_direction_floor(on_floor), Vec3::ZERO);
    assert_ne!(
        normalise_at_direction_floor(on_floor * 1.0001),
        Vec3::ZERO,
        "clear of the floor the normalisation is ordinary",
    );
}

/// The normalisation axis is a property of the **whole path** into the offset lines,
/// not of one helper. §7.6.2 states that every `normalise` in the document is the one
/// reciprocal-scaled operation, and its own mirroring table warns that the axes are
/// **not monotone** — a half-migration can measure worse on a given fixture than
/// doing nothing at all. So each unitisation the base points depend on is checked
/// here, on one vector chosen to part the two spellings, rather than only at the
/// perpendicular where the axis is measured.
///
/// The three above the perpendicular: a face's Newell normal (a case-1/2 sign
/// reference, and §7.5.1 regime 2's plane normal outright), an incident edge's unit
/// direction (which both sector-normal recipes cross), and the case-3 local normal's
/// own orientation step.
#[test]
fn every_normalisation_above_the_offset_lines_scales_by_one_reciprocal() {
    // One vector whose two spellings genuinely differ, reused so the test states the
    // *shape* of each site rather than three unrelated numbers.
    let parting = Vec3::new(0.3, 0.7, -1.1);
    let length = parting.length();
    let reciprocal = parting * (1.0 / length);
    assert_ne!(
        reciprocal,
        parting / length,
        "fixture sanity: this vector must actually part the two spellings",
    );

    // `edge_direction_from`, read off a single edge whose far endpoint is `parting`.
    let mut mesh = LoopMesh::new();
    let vertex = mesh.create_vertex(Vec3::ZERO);
    let far = mesh.create_vertex(parting);
    let apex = mesh.create_vertex(Vec3::new(-0.9, 0.2, 0.4));
    mesh.face_create_verts(&[vertex, far, apex], FaceCreateFlags::NONE);
    let edge = mesh.query_edge_exists(vertex, far).expect("the edge");
    assert_eq!(
        mesh.edge_direction_from(edge, vertex, Vec3::ZERO),
        Some(reciprocal),
    );

    // `face_newell_normal`, on a triangle whose Newell sum is `parting` scaled: the
    // sum is twice the triangle's area vector, so a triangle in the plane through the
    // origin square to `parting` produces a normal along it.
    let normal = mesh
        .face_newell_normal(mesh.face_keys()[0])
        .expect("normal");
    assert_eq!(
        normal,
        {
            let verts = mesh.face_vert_keys(mesh.face_keys()[0]);
            let mut sum = Vec3::ZERO;
            for i in 0..verts.len() {
                let current = mesh.vertex_position(verts[i]);
                let next = mesh.vertex_position(verts[(i + 1) % verts.len()]);
                sum.x += (current.y - next.y) * (current.z + next.z);
                sum.y += (current.z - next.z) * (current.x + next.x);
                sum.z += (current.x - next.x) * (current.y + next.y);
            }
            sum * (1.0 / sum.length())
        },
        "the Newell normal is the reciprocal-scaled sum",
    );

    // `orient_normal`, on both branches of its sign fix.
    assert_eq!(orient_normal(parting, parting), reciprocal);
    assert_eq!(orient_normal(parting, -parting), -reciprocal);
}

/// §4.1's slide primitive carries the same arithmetic: the direction is the raw edge
/// vector scaled by one reciprocal, and the distance is applied to *that*. Spelling it
/// `toward_far / length * distance` is a per-component division and a different
/// landing point in the last bit.
#[test]
fn slide_along_edge_scales_by_one_reciprocal() {
    let vertex = Vec3::ZERO;
    let far = Vec3::new(0.3, 0.7, -1.1);
    let distance = 0.0037_f32;
    let length = far.length();
    let reciprocal = vertex + far * (1.0 / length) * distance;
    assert_eq!(slide_along_edge(vertex, far, distance), reciprocal);
    assert_ne!(
        reciprocal,
        vertex + (far / length) * distance,
        "fixture sanity: this slide must actually part the two spellings",
    );
}

/// A flat closed fan about `v = (0,0,0)` whose two beveled spokes `P = (1,0,0)` and
/// `Q = (−2, 0.052, 0)` run *through* `v` to within `1.489°` of straight — inside
/// §7.6.2's `2°` band and outside a `0.81°` cosine reading of it. The unbeveled legs
/// `U = (0,1,0)` and `L = (0,−1,0)` split it into an upper and a lower sector, and the
/// two spokes differ in length (`1` vs `2.00068`) so the straight-through push's **raw**
/// direction sum leans measurably off the unit sum.
fn near_collinear_junction_fan() -> (LoopMesh, VertexKey, [VertexKey; 4]) {
    let mut mesh = LoopMesh::new();
    let v = mesh.create_vertex(Vec3::ZERO);
    let p = mesh.create_vertex(Vec3::new(1.0, 0.0, 0.0));
    let q = mesh.create_vertex(Vec3::new(-2.0, 0.052, 0.0));
    let u = mesh.create_vertex(Vec3::new(0.0, 1.0, 0.0));
    let l = mesh.create_vertex(Vec3::new(0.0, -1.0, 0.0));
    mesh.face_create_verts(&[v, p, u], FaceCreateFlags::NONE);
    mesh.face_create_verts(&[v, u, q], FaceCreateFlags::NONE);
    mesh.face_create_verts(&[v, q, l], FaceCreateFlags::NONE);
    mesh.face_create_verts(&[v, l, p], FaceCreateFlags::NONE);
    (mesh, v, [p, q, u, l])
}

/// The junction sector flanked by two beveled spokes, in the pair's disk order.
fn sector_between(
    mesh: &LoopMesh,
    v: VertexKey,
    beveled: &[EdgeKey],
    first: EdgeKey,
    second: EdgeKey,
) -> Sector {
    let is_beveled = |edge: EdgeKey| beveled.contains(&edge);
    sector_split(&mesh.vertex_bevel_corner_ring(v, &is_beveled), &is_beveled)
        .into_iter()
        .find(|s| s.bounding_edges == [first, second])
        .expect("the requested sector, in the requested disk order")
}

/// BEVEL.md §7.6.2, straight-through half: the common direction is the sum of the
/// pair's **raw**, full-length directions — which telescopes to `far_second −
/// far_first` — not of their unit versions. Summing unit vectors is the natural-looking
/// simplification and it discards the lean toward the longer flank.
///
/// The fan's spokes are `1` and `2.00068` long and `1.489°` apart, so at
/// `offset = 0.25` the raw sum lands the boundvert at `(0.0043330, 0.2499835, 0)` where
/// the unit sum gives `(0.0032494, 0.2500000, 0)` — `0.0010837` apart, **108 quanta** of
/// the canonical `1e-5` grid. Two conditions each hide the lean and this fixture avoids
/// both: equal flank lengths (the two sums then differ only by scale) and an exactly
/// zero direction angle (parallel vectors sum to one direction however they are scaled).
///
/// The same `1.489°` is inside §7.6.2's `2°` band and outside a `0.81°` cosine reading
/// of it, so a narrowed band shows up here too — as a boundvert about ten units out.
#[test]
fn junction_straight_through_push_sums_raw_directions_not_unit_ones() {
    let (mesh, v, [p, q, u, _l]) = near_collinear_junction_fan();
    let v_pos = mesh.vertex_position(v);
    let edge_p = mesh.query_edge_exists(v, p).expect("spoke P");
    let edge_q = mesh.query_edge_exists(v, q).expect("spoke Q");
    let sector = sector_between(&mesh, v, &[edge_p, edge_q], edge_p, edge_q);
    assert!(
        sector
            .corners
            .iter()
            .any(|&(edge, _)| Some(edge) == mesh.query_edge_exists(v, u)),
        "fixture sanity: the (P, Q) sector is the upper one, spanning the `U` leg",
    );

    let offset = 0.25_f32;
    let meet = mesh.solve_meet_point(v, v_pos, &sector, offset);
    let by_raw_sum = Vec3::new(0.0043330, 0.2499835, 0.0);
    let by_unit_sum = Vec3::new(0.0032494, 0.25, 0.0);
    assert!(
        (meet - by_raw_sum).length() < 1.0e-5,
        "the push must sum the raw directions, landing {by_raw_sum:?}, got {meet:?}",
    );
    assert!(
        (meet - by_unit_sum).length() > 1.0e-3,
        "the unit-vector sum is the failure mode, and it is 108 quanta away",
    );
    assert!(
        (by_raw_sum - by_unit_sum).length() > 1.0e-3,
        "fixture sanity: unequal flank lengths and a nonzero direction angle are what \
         make the two sums differ at all",
    );
}

/// BEVEL.md §7.6.2: `sector_normal` is used *"exactly as the three sources hand it
/// over — not re-normalised, not sign-fixed, not rescued"*, and the cross order
/// `common × sector_normal` **alone** points the push into the sector. Re-orienting the
/// result against a face-**centroid** reference is a different rule that agrees on
/// convex sectors and point-reflects the boundvert through `v` — `2d` away — where it
/// does not.
///
/// The discriminating fixture is a sector whose forward face is concave enough that its
/// centroid falls on the far side of the flanking spoke's line: `f0 = [v, P, (3,−3),
/// (4,−3), (4,4), U]` opens upward at `v` but averages to `(2, −1/6, 0)`. The
/// straight-through push belongs at `(0, +0.25, 0)`, in the upper sector with the `U`
/// leg; the centroid heuristic sends it to `(0, −0.25, 0)`, out of the sector entirely.
#[test]
fn junction_straight_through_push_takes_the_cross_order_not_the_face_centroid() {
    let mut mesh = LoopMesh::new();
    let v = mesh.create_vertex(Vec3::ZERO);
    let p = mesh.create_vertex(Vec3::new(1.0, 0.0, 0.0));
    let q = mesh.create_vertex(Vec3::new(-1.0, 0.0, 0.0));
    let u = mesh.create_vertex(Vec3::new(0.0, 1.0, 0.0));
    let l = mesh.create_vertex(Vec3::new(0.0, -1.0, 0.0));
    let lobe_low = mesh.create_vertex(Vec3::new(3.0, -3.0, 0.0));
    let lobe_out = mesh.create_vertex(Vec3::new(4.0, -3.0, 0.0));
    let lobe_high = mesh.create_vertex(Vec3::new(4.0, 4.0, 0.0));
    mesh.face_create_verts(
        &[v, p, lobe_low, lobe_out, lobe_high, u],
        FaceCreateFlags::NONE,
    );
    mesh.face_create_verts(&[v, u, q], FaceCreateFlags::NONE);
    mesh.face_create_verts(&[v, q, l], FaceCreateFlags::NONE);
    mesh.face_create_verts(&[v, l, p], FaceCreateFlags::NONE);

    let v_pos = mesh.vertex_position(v);
    let edge_p = mesh.query_edge_exists(v, p).expect("spoke P");
    let edge_q = mesh.query_edge_exists(v, q).expect("spoke Q");
    let sector = sector_between(&mesh, v, &[edge_p, edge_q], edge_p, edge_q);
    let layout = mesh
        .sector_layout(v, v_pos, &sector)
        .expect("a two-flank sector has a layout");

    let concave_face = sector.corners[0].1;
    assert!(
        mesh.face_center_of_mass(concave_face).y < 0.0,
        "fixture sanity: the sector's forward face must average below `v`'s upper half",
    );
    // Fixture sanity, measured on the superseded heuristic itself rather than on
    // `layout.perp_first`, which now carries the role-based cross order and so points
    // *into* the sector. The two disagree here, which is what makes the fixture
    // discriminating; at this `θ_dir = 0` the ladder returns before either is read, so
    // only `junction_planar_sector_perpendicular_takes_the_cross_order_not_the_centroid`
    // sees the difference in a meet.
    //
    // The plane normal has to come from the face, not from `layout.normal_first`: with
    // the two flanks exactly anti-parallel the edge span is zero and the sign fix hands
    // that zero straight on (it has no rescue), which is inert precisely because the
    // ladder claims this geometry before any offset line is built.
    let plane_normal = mesh
        .face_newell_normal(concave_face)
        .expect("the sector's forward face has a normal");
    let centroid_perp =
        centroid_oriented_perpendicular(&mesh, layout.dir_first, plane_normal, concave_face, v_pos);
    assert!(
        centroid_perp.y < 0.0,
        "fixture sanity: the centroid heuristic must point out of the sector, got \
         {centroid_perp:?}",
    );
    assert_eq!(
        layout.normal_first,
        Vec3::ZERO,
        "an exactly anti-parallel flank pair spans no plane, and the zero is kept",
    );
    assert!(
        plane_normal.cross(layout.dir_first).y > 0.0,
        "the offset-line perpendicular is the cross order about the walk-sense normal, \
         so it points into the sector",
    );

    let offset = 0.25_f32;
    let meet = mesh.solve_meet_point(v, v_pos, &sector, offset);
    assert!(
        (meet - Vec3::new(0.0, offset, 0.0)).length() < 1.0e-6,
        "the push follows `common × sector_normal` into the sector, got {meet:?}",
    );
    // The sign-fixed reading is the point reflection through `v`, `2 × offset` away.
    assert!(
        (meet - Vec3::new(0.0, -offset, 0.0)).length() > offset,
        "a centroid sign-fix would reflect the boundvert through `v`",
    );
    assert!(
        meet.y * mesh.vertex_position(u).y > 0.0,
        "the boundvert must land on the same side as the sector's own leg",
    );
}

/// BEVEL.md §7.6.2 *"The sense of the disk walk"*, on the **planar** (case 1/2) sector:
/// each offset line's perpendicular is a cross of the sector normal with the edge
/// direction whose sign comes from that edge's **role** — forward flank versus backward
/// flank — *"never from the face interior"*. Orienting it toward the forward face's
/// vertex mean instead agrees on a convex sector and inverts wherever that face is
/// reflex at `v`, because the mean then sits on the sector's convex complement.
///
/// The discriminating fixture is §7.6.6's **planar reflex junction**: a flat closed fan
/// whose forward face is an arrowhead with `v` at its notch (interior angle `240°`), so
/// the sector flanked by `v→r` and `v→b` sweeps `280°` through the absorbed leg while
/// the arrowhead's far barbs pull its vertex mean straight *down* into the `120°` cone
/// the sector excludes. `θ_dir = 100°` puts it well outside the `2°` ladder band, so
/// the general two-line solve actually builds both offset lines — which is exactly what
/// `junction_straight_through_push_takes_the_cross_order_not_the_face_centroid` cannot
/// see, sitting at `θ_dir = 0` where the ladder returns first.
///
/// The meet is checked against §7.6.2's own crossing identity `|B − v| = t / sin(θ/2)`
/// taken on the **reflex** bisector (§7.6.6: *"their crossing lands on the reflex (far)
/// side of `v`"*), and the centroid reading is rebuilt line-for-line so the failure mode
/// is stated as a position rather than as a sign: it lands `≈ 2 × offset` away.
#[test]
fn junction_planar_sector_perpendicular_takes_the_cross_order_not_the_centroid() {
    let mut mesh = LoopMesh::new();
    let v = mesh.create_vertex(Vec3::ZERO);
    let tip = mesh.create_vertex(Vec3::new(0.0, 0.5, 0.0));
    let r = mesh.create_vertex(Vec3::new(3.46, -2.0, 0.0));
    let leg = mesh.create_vertex(Vec3::new(-3.46, -2.0, 0.0));
    let b = mesh.create_vertex(Vec3::new(-1.0, -2.75, 0.0));
    let arrowhead = mesh.face_create_verts(&[leg, v, r, tip], FaceCreateFlags::NONE);
    mesh.face_create_verts(&[v, leg, b], FaceCreateFlags::NONE);
    mesh.face_create_verts(&[v, b, r], FaceCreateFlags::NONE);

    let v_pos = mesh.vertex_position(v);
    let edge_r = mesh.query_edge_exists(v, r).expect("flank g_k");
    let edge_b = mesh.query_edge_exists(v, b).expect("flank g_(k+1)");
    let sector = sector_between(&mesh, v, &[edge_r, edge_b], edge_r, edge_b);
    let layout = mesh
        .sector_layout(v, v_pos, &sector)
        .expect("a two-flank sector has a layout");

    assert_eq!(
        sector.corners[0].1, arrowhead,
        "fixture sanity: the sector under test is the one the arrowhead fills — a \
         wrong-sector regression must read as this, not as a sign flip",
    );
    assert!(
        !layout.non_planar,
        "fixture sanity: a flat fan is a case-1/2 sector, the branch under test",
    );
    let direction_angle = unit_chord_angle_between(-layout.dir_first, layout.dir_second);
    assert!(
        !is_degenerate_direction_angle(direction_angle),
        "fixture sanity: θ_dir = {} rad must be outside the ladder band so the general \
         offset-line solve runs",
        direction_angle,
    );

    // Fixture sanity: the forward face really is reflex at `v` and its vertex mean
    // really does fall on the sector's convex complement, so the two rules disagree.
    assert!(
        mesh.face_center_of_mass(arrowhead).y < 0.0,
        "the arrowhead's barbs pull its mean below `v`, into the excluded cone",
    );
    let centroid_perp_first = centroid_oriented_perpendicular(
        &mesh,
        layout.dir_first,
        layout.normal_first,
        arrowhead,
        v_pos,
    );
    let cross_order_perp_first = layout
        .normal_first
        .cross(layout.dir_first)
        .normalize_or_zero();
    assert!(
        centroid_perp_first.dot(cross_order_perp_first) < 0.0,
        "fixture sanity: the two rules must disagree here, got {centroid_perp_first:?} \
         against {cross_order_perp_first:?}",
    );
    assert!(
        (layout.perp_first - cross_order_perp_first).length() < 1.0e-6,
        "`g_k`'s perpendicular is the cross of the walk-sense sector normal with the \
         edge direction, not the push toward the face's mean: expected \
         {cross_order_perp_first:?}, got {:?}",
        layout.perp_first,
    );

    // §7.6.2's crossing identity on the reflex bisector: the wedge angle is the angle
    // between the two flanks as rays out of `v`, and the reflex side is the far one.
    let offset = 0.25_f32;
    let wedge = layout.dir_first.angle_between(layout.dir_second);
    let reflex_bisector = -(layout.dir_first + layout.dir_second).normalize();
    let expected = v_pos + reflex_bisector * (offset / (wedge * 0.5).sin());

    let meet = mesh.solve_meet_point(v, v_pos, &sector, offset);
    assert!(
        (meet - expected).length() < 1.0e-5,
        "the meet is the two offset lines' crossing at {expected:?}, got {meet:?}",
    );

    // The failure mode, rebuilt: swap `perp_first` for the centroid reading and the
    // crossing walks a full `2 × offset` off — the parity error this pin exists for.
    let face_last = sector.corners.last().expect("the sector's last corner").1;
    let centroid_perp_second = centroid_oriented_perpendicular(
        &mesh,
        layout.dir_second,
        layout.normal_second,
        face_last,
        v_pos,
    );
    let by_centroid = intersect_lines(
        &Line {
            point: v_pos + centroid_perp_first * offset,
            direction: layout.dir_first,
        },
        &Line {
            point: v_pos + centroid_perp_second * offset,
            direction: layout.dir_second,
        },
    );
    let LineMeet::Crossing(by_centroid) = by_centroid else {
        panic!("two coplanar offset lines at a 100° direction angle cross");
    };
    assert!(
        (by_centroid - meet).length() > 2.0 * offset - 1.0e-3,
        "the centroid heuristic lands `2 × offset` out at {by_centroid:?}",
    );
}

/// BEVEL.md §7.6.2's `#zero-area-sector-collapse`: source 1 of the ladder's normal
/// takes the sector's own face normal **directly**, so a zero-area face hands over the
/// zero vector, `common × sector_normal` is zero, the zero-returning normalisation
/// leaves it zero, and the straight-through push lands the boundvert **exactly on
/// `v`**. Substituting an axis (or the vertex normal, or a neighbour's) instead moves
/// it a whole `offset` off-axis — the failure mode this pins.
///
/// Four collinear verts on the `x` axis, closed by two zero-area triangles sharing
/// `{v, right}`: beveling `{v, right}` and `{v, left}` makes the sector between them
/// straight through (`θ_dir = 0`) with a zero-area sector face.
#[test]
fn junction_zero_area_sector_push_collapses_onto_the_vertex() {
    let mut mesh = LoopMesh::new();
    let v = mesh.create_vertex(Vec3::ZERO);
    let right = mesh.create_vertex(Vec3::new(2.0, 0.0, 0.0));
    let left = mesh.create_vertex(Vec3::new(-1.0, 0.0, 0.0));
    let far_right = mesh.create_vertex(Vec3::new(3.0, 0.0, 0.0));
    mesh.face_create_verts(&[v, right, left], FaceCreateFlags::NONE);
    mesh.face_create_verts(&[v, right, far_right], FaceCreateFlags::NONE);

    let v_pos = mesh.vertex_position(v);
    let edge_right = mesh.query_edge_exists(v, right).expect("E");
    let edge_left = mesh
        .query_edge_exists(v, left)
        .expect("the collinear spoke");
    let is_beveled = |edge: EdgeKey| edge == edge_right || edge == edge_left;

    let corners = mesh.vertex_bevel_corner_ring(v, &is_beveled);
    let sector = sector_split(&corners, &is_beveled)
        .into_iter()
        .find(|s| {
            s.bounding_edges.len() == 2
                && s.bounding_edges.contains(&edge_right)
                && s.bounding_edges.contains(&edge_left)
        })
        .expect("the straight-through sector");
    let layout = mesh
        .sector_layout(v, v_pos, &sector)
        .expect("a two-flank sector has a layout");

    assert_eq!(
        layout.sector_normal,
        Vec3::ZERO,
        "the ladder's normal is the zero-area face's own, taken directly",
    );

    let offset = 0.1;
    let meet = mesh.solve_meet_point(v, v_pos, &sector, offset);
    assert!(
        meet.is_finite(),
        "a zero normal must not produce a non-finite coordinate, got {meet:?}",
    );
    assert_eq!(
        meet, v_pos,
        "the straight-through push collapses exactly onto `v`",
    );
    // The rescued reading — an axis substituted for the degenerate face normal —
    // would land a whole `offset` away, which is what the assertion above excludes.
    assert!(
        (meet - v_pos).length() < offset * 0.5,
        "a rescued normal would push the full offset off-axis",
    );
}
