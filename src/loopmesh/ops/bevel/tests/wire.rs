use super::*;

/// A wire spoke on a terminal's disk: a face-less leg the **corner ring cannot see** —
/// the ring walks face corners, and a radial-empty edge bounds none. That is *correct*
/// (BEVEL.md §7.2.1): the wire is not on the bevel disk at all, so the ring covers the
/// whole of it. The centre's **face-bearing** valence is 3 here (an arc of three faced
/// spokes) and the wire is inert during layout; §8.3.0.4's post-pass re-points it.
fn wire_spoke_open_fan() -> (LoopMesh, VertexKey, EdgeKey, VertexKey) {
    let (mut mesh, centre, rim) = open_planar_fan(3, 0);
    let tip = mesh.create_vertex(Vec3::new(0.0, -1.0, 0.0));
    mesh.create_edge([centre, tip], EdgeCreateFlags::NONE);
    let beveled = mesh
        .query_edge_exists(centre, rim[1])
        .expect("the arc's middle spoke — the one with a leg on both sides");
    (mesh, centre, beveled, tip)
}

/// BEVEL.md §7.6.2 "Parallel and anti-parallel special cases" (the trap note):
/// beveling the **two collinear spokes** that run straight *through* the centre of
/// a flat grid is the `θ_dir ≈ 0` **parallel** config — the two edges' rays leave
/// `v` in *opposite* directions, but taken as (into `v`, out of `v`) they fold onto
/// one vector. It takes the **perpendicular push** of bullet 1: each of the two
/// half-plane sectors lands its boundvert one `offset` out **along its own
/// perpendicular `y`-leg**, at `v ± offset·ŷ`. Sliding along the common `x` edge
/// line (bullet 2's doubled-back rule) is the parity bug this pins against — it
/// would land the boundverts on the wrong axis, at `v ± offset·x̂`.
#[test]
fn edge_bevel_collinear_through_vertex_pushes_perpendicular_not_along_the_edge() {
    let mut mesh = LoopMesh::new();
    let (centre, edge_plus_x) = grid_3x3(&mut mesh, Vec3::ZERO);
    // The centre sits at (1, 1, 0); its two collinear spokes run along x.
    assert_eq!(mesh.vertex_position(centre), Vec3::new(1.0, 1.0, 0.0));
    let edge_minus_x = mesh
        .query_edge_exists(centre, vert_at(&mesh, Vec3::new(0.0, 1.0, 0.0)))
        .expect("the centre's -x edge");

    let result = mesh.bevel(
        &[GeomKey::from(edge_plus_x), GeomKey::from(edge_minus_x)],
        BevelOptions {
            offset: 0.25,
            affect: BevelAffect::Edges,
            segments: 1,
            ..Default::default()
        },
    );

    // Perpendicular push: one offset out along the y leg of each half-plane sector.
    assert_eq!(
        result_has_vert_near(&mesh, &result, Vec3::new(1.0, 1.25, 0.0)),
        1,
        "upper sector's boundvert must push +y (perpendicular to the beveled x spokes)",
    );
    assert_eq!(
        result_has_vert_near(&mesh, &result, Vec3::new(1.0, 0.75, 0.0)),
        1,
        "lower sector's boundvert must push -y",
    );
    // The doubled-back slide answer (along the common x line) must not appear.
    assert_eq!(
        result_has_vert_near(&mesh, &result, Vec3::new(1.25, 1.0, 0.0)),
        0,
        "the slide-along-the-edge answer is the anti-parallel bug, not this config",
    );
    assert_eq!(
        result_has_vert_near(&mesh, &result, Vec3::new(0.75, 1.0, 0.0)),
        0,
        "the slide-along-the-edge answer is the anti-parallel bug, not this config",
    );
}

/// The same perpendicular push holds at every segment count (BEVEL.md §7.6.2 oracle
/// pin, "the perpendicular push is what BMesh does here, at every segment count"):
/// the two boundverts are the segment chain's endpoints regardless of `segments`.
#[test]
fn edge_bevel_collinear_through_vertex_pushes_perpendicular_at_every_segment_count() {
    for segments in 1..=3 {
        let mut mesh = LoopMesh::new();
        let (centre, edge_plus_x) = grid_3x3(&mut mesh, Vec3::ZERO);
        let edge_minus_x = mesh
            .query_edge_exists(centre, vert_at(&mesh, Vec3::new(0.0, 1.0, 0.0)))
            .expect("the centre's -x edge");
        let result = mesh.bevel(
            &[GeomKey::from(edge_plus_x), GeomKey::from(edge_minus_x)],
            BevelOptions {
                offset: 0.25,
                affect: BevelAffect::Edges,
                segments,
                ..Default::default()
            },
        );
        assert_eq!(
            result_has_vert_near(&mesh, &result, Vec3::new(1.0, 1.25, 0.0)),
            1,
            "segments = {segments}: +y boundvert",
        );
        assert_eq!(
            result_has_vert_near(&mesh, &result, Vec3::new(1.0, 0.75, 0.0)),
            1,
            "segments = {segments}: -y boundvert",
        );
    }
}

/// The same patch with a face-less **wire** spoke off its centre out to `tip`:
/// 10 V / 13 E / 4 F. Returns the centre, the wire's far tip, and the beveled edge.
fn grid_3x3_with_wire(mesh: &mut LoopMesh, tip: Vec3) -> (VertexKey, VertexKey, EdgeKey) {
    let (centre, beveled) = grid_3x3(mesh, Vec3::ZERO);
    let tip = mesh.create_vertex(tip);
    mesh.create_edge([centre, tip], EdgeCreateFlags::NONE);
    (centre, tip, beveled)
}

/// BEVEL.md §7.2.1: a wire spoke does not disqualify the terminal — it is **inert**, and
/// the edge bevels exactly as it would with the spoke absent. §8.3.0.4: the wire is not
/// *untouched*, though, because the centre it hangs off is consumed — the post-pass
/// re-points it onto the ring before the loose-vertex free would have swept it away.
///
/// The load-bearing assertion is the last one: **the wire survives**, still face-less,
/// still on its own far tip, now hanging off a ring vertex. Destroying it silently —
/// mesh the caller never selected — is the failure mode this fixture has always guarded.
#[test]
fn edge_bevel_terminal_with_a_wire_spoke_bevels_around_it_and_never_destroys_the_wire() {
    for segments in [1, 2] {
        let (mut mesh, centre, beveled, tip) = wire_spoke_open_fan();
        let before = counts(&mesh);
        assert_eq!(mesh.iter_disk_edges(centre).count(), 4);
        assert_eq!(
            mesh.vertex_corner_ring(centre).len(),
            2,
            "the ring sees two sectors — the three faced spokes; the wire is not on it",
        );
        assert!(
            mesh.terminal_close_admits(centre, beveled),
            "the face-bearing valence is 3, so the terminal closes around the wire",
        );

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

        assert!(
            !result.verts.is_empty() && !result.faces.is_empty(),
            "the edge is beveled, not declined",
        );
        assert!(
            !mesh.vert_keys().contains(&centre),
            "the centre is consumed"
        );
        assert_ne!(counts(&mesh), before);

        let landings = wire_landings(&mesh, tip);
        assert_eq!(
            landings.len(),
            1,
            "the wire spoke survives as exactly one face-less edge off its own tip",
        );
        assert!(
            result
                .verts
                .iter()
                .any(|&vertex| mesh.vertex_position(vertex) == landings[0]),
            "and it hangs off a vertex of the new ring, not off the consumed centre",
        );
    }
}

/// BEVEL.md §8.3.0.4's worked oracle: bevel the centre's `+x` edge of the 3×3 grid whose
/// centre carries a wire. The wire raises no valence (§7.2.1), so the near terminal reads
/// as **face-bearing valence 4** — three boundverts, a fanned fill — and the whole answer
/// is the wire-free grid's answer **plus one vertex and one edge** (the untouched tip and
/// the re-pointed wire): `10/13/4 → 13/18/6` at `segments = 1`, with face sizes
/// `[3, 4, 4, 4, 5, 5]`; `15/22/8` at 2; `17/26/10` at 3.
///
/// Had the wire been counted as a leg, the centre would read valence 5 → four boundverts
/// → the fill switches from a fan to a whole n-gon and the ring gains a vert: 14 V and a
/// quad where the oracle has a triangle. The counts are the observable signature of the
/// wire having stayed out of the layout.
#[test]
fn edge_bevel_wire_spoke_is_inert_and_costs_exactly_one_vert_and_one_edge() {
    for (segments, expected) in [(1, (13, 18, 6)), (2, (15, 22, 8)), (3, (17, 26, 10))] {
        let mut mesh = LoopMesh::new();
        // Off-axis, leaning out over the −x leg: the §8.3.0.4 argmin discriminates.
        let (_, tip, beveled) = grid_3x3_with_wire(&mut mesh, Vec3::new(0.4, 1.0, 1.0));
        assert_eq!(
            counts(&mesh),
            (10, 13, 4),
            "the pristine grid plus its wire"
        );

        mesh.bevel(
            &[beveled.into()],
            BevelOptions {
                offset: 0.25,
                segments,
                affect: BevelAffect::Edges,
                ..Default::default()
            },
        );
        mesh.validate_topology();

        assert_eq!(counts(&mesh), expected, "at segments = {segments}");
        assert_eq!(
            wire_landings(&mesh, tip).len(),
            1,
            "the wire is conserved — neither destroyed nor doubled",
        );
        if segments == 1 {
            assert_eq!(
                face_size_multiset(&mesh),
                vec![3, 4, 4, 4, 5, 5],
                "the fan triangle, the strip quad, E's two radial quads, two pentagons",
            );
        }
    }
}

/// BEVEL.md §8.3.0.4 point 2: the wire lands on the boundvert **geometrically nearest**
/// its far tip — and the same ring hands it to a different boundvert when the tip leans
/// the other way. A discriminating (off-axis) tip, so no rounding decides anything.
///
/// Point 4: the answer is `segments`-independent. A larger `segments` moves no boundvert
/// and adds no candidate — only profile rungs, which are not boundverts.
#[test]
fn edge_bevel_wire_spoke_lands_on_the_nearest_boundvert_and_leans_with_the_tip() {
    let leaning_over_the_leg = (Vec3::new(0.4, 1.0, 1.0), Vec3::new(0.75, 1.0, 0.0));
    let leaning_over_the_flank = (Vec3::new(1.0, 1.6, 1.0), Vec3::new(1.0, 1.25, 0.0));

    for (tip, expected) in [leaning_over_the_leg, leaning_over_the_flank] {
        for segments in [1, 2, 3] {
            let mut mesh = LoopMesh::new();
            let (_, tip, beveled) = grid_3x3_with_wire(&mut mesh, tip);

            mesh.bevel(
                &[beveled.into()],
                BevelOptions {
                    offset: 0.25,
                    segments,
                    affect: BevelAffect::Edges,
                    ..Default::default()
                },
            );
            mesh.validate_topology();

            let landings = wire_landings(&mesh, tip);
            assert_eq!(landings.len(), 1);
            assert!(
                (landings[0] - expected).length() < 1.0e-5,
                "at segments = {segments} the wire wanted {expected}, landed on {}",
                landings[0],
            );
        }
    }
}

/// BEVEL.md §8.3.0.4 point 1, the sharp one: the candidate set is the ring's
/// **boundverts**, not the ring. At `segments = 2` the profile chain's interior rung —
/// the "peak" — lands on the midpoint of the `start`–`end` chord, which on this flat grid
/// is exactly where the consumed centre used to be: distance² **1.0** from an on-axis tip
/// against **1.0625** for every boundvert. The strictly nearer vertex is *on the ring*,
/// and the wire still goes to a boundvert.
///
/// The tie itself is not asserted (three boundverts equidistant; the winner is decided by
/// rounding, §8.3.0.4's warning) — only that the landing is one of the three, and never
/// the peak.
#[test]
fn edge_bevel_wire_spoke_on_axis_never_lands_on_the_nearer_chain_rung() {
    let mut mesh = LoopMesh::new();
    let (_, tip, beveled) = grid_3x3_with_wire(&mut mesh, Vec3::new(1.0, 1.0, 1.0));

    mesh.bevel(
        &[beveled.into()],
        BevelOptions {
            offset: 0.25,
            segments: 2,
            affect: BevelAffect::Edges,
            ..Default::default()
        },
    );
    mesh.validate_topology();

    let landings = wire_landings(&mesh, tip);
    assert_eq!(landings.len(), 1);
    let peak = Vec3::new(1.0, 1.0, 0.0);
    assert!(
        (landings[0] - peak).length() > 1.0e-5,
        "the wire landed on the chain rung at the consumed centre's own position",
    );
    let boundverts = [
        Vec3::new(1.0, 0.75, 0.0),
        Vec3::new(1.0, 1.25, 0.0),
        Vec3::new(0.75, 1.0, 0.0),
    ];
    assert!(
        boundverts
            .iter()
            .any(|&boundvert| (landings[0] - boundvert).length() < 1.0e-5),
        "the wire landed off the boundvert set entirely: {}",
        landings[0],
    );
}

/// BEVEL.md §7.6.1.1: the wire is absent from a **junction's** disk on exactly the same
/// terms — it bounds no sector, is absorbed into none, takes no boundvert, and does not
/// raise `s`. Bevel the grid centre's `+x` **and** `−x` edges, making the wire-carrying
/// centre a two-sector junction, and the §8.3.0.4 identity holds there too: the answer is
/// the wire-free mesh's answer plus the tip and the one re-pointed wire edge
/// (`+1 V / +1 E / +0 F`), with the wire hanging off a boundvert of the junction's ring.
#[test]
fn edge_bevel_wire_spoke_on_a_junction_disk_is_inert_and_re_points_onto_the_ring() {
    for segments in [1, 2] {
        let options = BevelOptions {
            offset: 0.25,
            segments,
            affect: BevelAffect::Edges,
            ..Default::default()
        };

        let mut wire_free = LoopMesh::new();
        let (centre, plus_x) = grid_3x3(&mut wire_free, Vec3::ZERO);
        let minus_x = wire_free
            .query_edge_exists(centre, vert_at(&wire_free, Vec3::new(0.0, 1.0, 0.0)))
            .expect("the centre's −x edge");
        wire_free.bevel(&[plus_x.into(), minus_x.into()], options.clone());
        let (verts, edges, faces) = counts(&wire_free);

        let mut mesh = LoopMesh::new();
        let (centre, tip, plus_x) = grid_3x3_with_wire(&mut mesh, Vec3::new(1.0, 1.6, 1.0));
        let minus_x = mesh
            .query_edge_exists(centre, vert_at(&mesh, Vec3::new(0.0, 1.0, 0.0)))
            .expect("the centre's −x edge");
        let result = mesh.bevel(&[plus_x.into(), minus_x.into()], options);
        mesh.validate_topology();

        assert_eq!(
            counts(&mesh),
            (verts + 1, edges + 1, faces),
            "the junction bevels as if the wire were not there, at segments = {segments}",
        );
        let landings = wire_landings(&mesh, tip);
        assert_eq!(
            landings.len(),
            1,
            "the wire survives the junction's consumption"
        );
        assert!(
            result
                .verts
                .iter()
                .any(|&vertex| mesh.vertex_position(vertex) == landings[0]),
            "and lands on a vertex of the junction's ring",
        );
    }
}

/// BEVEL.md §8.3.0.4's phase order, which is the whole reason the post-pass is a
/// **global** phase: both ends of the wire are beveled vertices, so both are consumed.
/// Interleave the pass per vertex and the first of the two sees no ring at the far end,
/// attaches the wire to a vertex about to be freed, and silently destroys it.
///
/// Two disjoint 3×3 grids joined only by a wire between their centres (18/25/8), one edge
/// beveled at each centre: **24/35/12** — exactly twice the wire-free single-grid answer
/// (12/17/6 each) plus the one conserved wire edge. Get the order wrong and the edge count
/// drops to 34 with the wire gone.
#[test]
fn edge_bevel_wire_spoke_with_both_ends_beveled_is_conserved_by_the_global_post_pass() {
    let mut mesh = LoopMesh::new();
    let (near_centre, first) = grid_3x3(&mut mesh, Vec3::ZERO);
    let (far_centre, second) = grid_3x3(&mut mesh, Vec3::new(3.0, 0.0, 0.0));
    let wire = mesh.create_edge([near_centre, far_centre], EdgeCreateFlags::NONE);
    assert_eq!(
        counts(&mesh),
        (18, 25, 8),
        "two grids joined only by the wire"
    );
    assert_eq!(mesh.edge_radial_iter(wire).count(), 0);

    mesh.bevel(
        &[first.into(), second.into()],
        BevelOptions {
            offset: 0.25,
            affect: BevelAffect::Edges,
            ..Default::default()
        },
    );
    mesh.validate_topology();

    assert_eq!(counts(&mesh), (24, 35, 12));
    let wires: Vec<EdgeKey> = mesh
        .edge_keys()
        .into_iter()
        .filter(|&edge| mesh.edge_radial_iter(edge).next().is_none())
        .collect();
    assert_eq!(
        wires.len(),
        1,
        "the wire is conserved as exactly one face-less edge, spanning the two rings",
    );
    for endpoint in mesh.edge_vert_keys(wires[0]) {
        assert!(
            !mesh.vert_face_keys(endpoint).is_empty(),
            "each end of the re-pointed wire is a live ring vertex",
        );
    }
}

/// Every face-less edge of the mesh, as an unordered pair of its endpoints' positions.
fn wire_edge_spans(mesh: &LoopMesh) -> Vec<[Vec3; 2]> {
    mesh.edge_keys()
        .into_iter()
        .filter(|&edge| mesh.edge_radial_iter(edge).next().is_none())
        .map(|edge| mesh.edge_vert_keys(edge).map(|v| mesh.vertex_position(v)))
        .collect()
}

fn spans_contain(spans: &[[Vec3; 2]], wanted: [Vec3; 2]) -> bool {
    spans.iter().any(|&[a, b]| {
        let matches = |p: Vec3, q: Vec3| (p - q).length() < 1.0e-4;
        (matches(a, wanted[0]) && matches(b, wanted[1]))
            || (matches(a, wanted[1]) && matches(b, wanted[0]))
    })
}

/// BEVEL.md §8.3.0.4's opening sentence, read as a *gate*: the post-pass exists because
/// the vertex the wire hangs off is **consumed**. A recorded ring is not that condition —
/// every beveled vertex gets one, but phase E frees only the endpoints the strip build
/// left loose, and a beveled vertex still incident to unbeveled faces is **kept**, wire
/// and all. Gate on the ring instead of on consumption and a surviving *near* end gets a
/// second copy of its wire (the original still hangs off it, `+1 E`), while a surviving
/// *far* end is worse: the wire is re-pointed onto that vertex's ring, and the vertex
/// itself — which was never going to lose it — is left wireless.
///
/// Built white-box, because the two conditions only come apart on a mesh the edge-mode
/// planner does not currently admit: it declines any endpoint whose fan it cannot close
/// whole, so every endpoint it *does* build is re-stitched off every incident face and
/// ends up loose. The state the gate guards is therefore constructed here directly —
/// `kept` carries a ring and a live face, `gone` carries a ring and nothing else.
#[test]
fn wire_post_pass_gates_on_consumption_not_on_a_recorded_ring() {
    let mut mesh = LoopMesh::new();
    let quad: Vec<VertexKey> = [
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(1.0, 1.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
    ]
    .into_iter()
    .map(|position| mesh.create_vertex(position))
    .collect();
    mesh.face_create_verts(&quad, FaceCreateFlags::NONE);
    let kept = quad[0];

    // `gone` is consumed (loose, and its whole ring is minted); `kept` is not (its quad
    // survives), yet both carry a ring, as both would after a bevel that kept one of them.
    let gone = mesh.create_vertex(Vec3::new(0.0, 0.0, 5.0));
    let near_boundvert = mesh.create_vertex(Vec3::new(0.5, 0.0, 4.5));
    let far_boundvert = mesh.create_vertex(Vec3::new(-0.5, 0.0, 5.0));
    let tip = mesh.create_vertex(Vec3::new(0.0, 0.0, -5.0));
    let spanning_wire = mesh.create_edge([gone, kept], EdgeCreateFlags::NONE);
    let kept_wire = mesh.create_edge([kept, tip], EdgeCreateFlags::NONE);

    let spokes = mesh.snapshot_wire_spokes(&[gone, kept]);
    assert_eq!(
        spokes.len(),
        3,
        "the spanning wire is visited from both ends"
    );
    let rings = BevelRings {
        wanted: true,
        rings: vec![
            BevelRing {
                vertex: gone,
                boundverts: vec![near_boundvert, far_boundvert],
            },
            BevelRing {
                vertex: kept,
                boundverts: vec![quad[1], quad[3]],
            },
        ],
    };

    let before = counts(&mesh);
    mesh.reattach_wire_spokes(&spokes, &rings);
    mesh.validate_topology();

    let (verts, edges, faces) = before;
    assert_eq!(
        counts(&mesh),
        (verts, edges + 1, faces),
        "exactly one wire is re-pointed — `kept`'s own wire is not re-created, and the \
         spanning wire is not minted twice",
    );
    assert!(
        mesh.query_edge_exists(near_boundvert, kept).is_some(),
        "the spanning wire lands on `gone`'s nearest boundvert at the consumed end, and \
         on `kept` **itself** at the surviving end — not on `kept`'s ring",
    );
    for boundvert in [quad[1], quad[3]] {
        assert!(
            mesh.query_edge_exists(near_boundvert, boundvert).is_none()
                && mesh.query_edge_exists(far_boundvert, boundvert).is_none(),
            "no wire was laid onto the surviving vertex's ring",
        );
    }
    assert!(
        mesh.query_edge_exists(kept, tip) == Some(kept_wire)
            && mesh.iter_disk_edges(tip).count() == 1,
        "the wire hanging off the surviving vertex is left exactly as it was",
    );
    assert!(
        mesh.edge_keys().contains(&spanning_wire),
        "the original is still on the doomed vertex, to die with it in phase E",
    );
}

/// BEVEL.md §8.3.0.4 point 3, the measured case: the wire is visited **once per beveled
/// endpoint**, each visit scanning the boundvert product with **its own** ring outermost.
/// When two candidate pairs tie *exactly* and differ in **both** elements, the two visits
/// break the tie toward different pairs, nothing dedups, and the wire is re-created
/// **twice** — `+1 E` over the ordinary case.
///
/// The guide's fixture: two grids at a pure-`x` shift, centres `(1,1,0)` and `(4,1,0)`,
/// beveled **towards each other** (`+x` and `−x`), so the rings are mirror images across
/// the wire's midpoint. Two flank↔flank pairs attain `d² = 9` exactly. Contrast the
/// *translate*-symmetric arrangement (both centres beveled `+x`, the test above): its tied
/// pairs **share** their far element, so both visits break toward the same near boundvert
/// and the wire collapses to one edge. Only in-crate can this be pinned: a per-wire dedup
/// would emit one edge here and no cross-backend comparison would name which pairs.
#[test]
fn edge_bevel_wire_spoke_tie_across_both_elements_re_creates_the_wire_twice() {
    let mut mesh = LoopMesh::new();
    let (near_centre, toward_far) = grid_3x3(&mut mesh, Vec3::ZERO);
    let (far_centre, _) = grid_3x3(&mut mesh, Vec3::new(3.0, 0.0, 0.0));
    let toward_near = mesh
        .query_edge_exists(far_centre, vert_at(&mesh, Vec3::new(3.0, 1.0, 0.0)))
        .expect("the far centre's −x edge, pointing back at the near grid");
    mesh.create_edge([near_centre, far_centre], EdgeCreateFlags::NONE);

    mesh.bevel(
        &[toward_far.into(), toward_near.into()],
        BevelOptions {
            offset: 0.25,
            affect: BevelAffect::Edges,
            ..Default::default()
        },
    );
    mesh.validate_topology();

    let spans = wire_edge_spans(&mesh);
    assert_eq!(
        spans.len(),
        2,
        "the tie is across both elements, so each visit takes its own pair: two wires",
    );
    assert!(
        spans_contain(
            &spans,
            [Vec3::new(1.0, 0.75, 0.0), Vec3::new(4.0, 0.75, 0.0)]
        ),
        "one visit took the −y flank pair: {spans:?}",
    );
    assert!(
        spans_contain(
            &spans,
            [Vec3::new(1.0, 1.25, 0.0), Vec3::new(4.0, 1.25, 0.0)]
        ),
        "the other took the +y flank pair: {spans:?}",
    );
}

/// BEVEL.md §8.3.0.4 point 3's *other* pin: with both ends beveled the rule is a closest
/// **pair** argmin over the full boundvert product — not the cheaper per-end argmin, which
/// asks each end for the boundvert nearest the far endpoint's **old** position. The two
/// agree whenever the rings are small against the wire, which is every ordinary fixture;
/// they part company when a ring's radius is comparable to the wire's length.
///
/// The guide's discriminating fixture: centres `(1,1,0)` and `(1.5,1.6,0)`, `offset = 0.4`,
/// beveled `+x` and `+y` (so the rings are not translates and nothing ties). The closest
/// pair is `(1.0,1.4,0) ↔ (1.1,1.6,0)` at `d² = 0.05`; the per-end rule would land the far
/// end on the *leg* vert `(1.5,1.2,0)` at `d² = 0.29`.
#[test]
fn edge_bevel_wire_spoke_takes_the_closest_boundvert_pair_not_the_per_end_argmin() {
    let mut mesh = LoopMesh::new();
    let (near_centre, along_x) = grid_3x3(&mut mesh, Vec3::ZERO);
    let (far_centre, _) = grid_3x3(&mut mesh, Vec3::new(0.5, 0.6, 0.0));
    let along_y = mesh
        .query_edge_exists(far_centre, vert_at(&mesh, Vec3::new(1.5, 2.6, 0.0)))
        .expect("the far centre's +y edge");
    mesh.create_edge([near_centre, far_centre], EdgeCreateFlags::NONE);

    mesh.bevel(
        &[along_x.into(), along_y.into()],
        BevelOptions {
            offset: 0.4,
            affect: BevelAffect::Edges,
            ..Default::default()
        },
    );
    mesh.validate_topology();

    let spans = wire_edge_spans(&mesh);
    assert_eq!(spans.len(), 1, "the pair argmin is strict: one wire");
    assert!(
        spans_contain(&spans, [Vec3::new(1.0, 1.4, 0.0), Vec3::new(1.1, 1.6, 0.0)]),
        "the wire took the closest pair, not the per-end argmin (which would have landed \
         on the far ring's leg vert at (1.5, 1.2, 0)): {spans:?}",
    );
}

/// BEVEL.md §8.3.0.4's re-point framing: the re-created wire **is** the wire — it carries
/// the original's whole per-edge block (crease, seam, sharp, every custom layer), not the
/// layer defaults of a fresh mint. A default-valued re-create is a per-edge-customdata
/// divergence, not a topological one, so it passes every count and every position check
/// this file makes and shows up only here.
#[test]
fn edge_bevel_wire_spoke_carries_its_per_edge_block_across_the_re_point() {
    let mut mesh = LoopMesh::new();
    let (centre, tip, beveled) = grid_3x3_with_wire(&mut mesh, Vec3::new(0.4, 1.0, 1.0));
    let wire = mesh
        .query_edge_exists(centre, tip)
        .expect("the centre's wire spoke");

    let tint = tint_every_edge(&mut mesh);
    let wire_tint = tint_of(&mesh, tint, wire);
    mesh.set_edge_seam(wire, true);

    mesh.bevel(
        &[beveled.into()],
        BevelOptions {
            offset: 0.25,
            affect: BevelAffect::Edges,
            ..Default::default()
        },
    );
    mesh.validate_topology();

    let re_pointed: Vec<EdgeKey> = mesh
        .iter_disk_edges(tip)
        .filter(|&edge| mesh.edge_radial_iter(edge).next().is_none())
        .collect();
    assert_eq!(re_pointed.len(), 1);
    assert_eq!(
        tint_of(&mesh, tint, re_pointed[0]),
        wire_tint,
        "the re-pointed wire kept its own custom layer value, not the layer default",
    );
    assert!(
        mesh.edge_seam(re_pointed[0]),
        "and its seam bit rode along with it",
    );
}

/// The face of `mesh` whose corners are *all* ring verts — the cap (BEVEL.md §5.1).
fn cap_face_of(mesh: &LoopMesh, ring: &[VertexKey]) -> Option<FaceKey> {
    mesh.face_keys().into_iter().find(|&face| {
        mesh.face_vert_keys(face)
            .iter()
            .all(|corner| ring.contains(corner))
    })
}

/// BEVEL.md §7.2.1's **vertex-mode box** — the documented exception to the edge-mode
/// face-bearing disk filter. Under `affect = Vertices` the disk is the *full* incident
/// edge set, so the 3×3 grid's wire-carrying centre reads as valence **5**, not 4: the
/// wire gets its own ring vert, slid along the wire itself by the offset, and it is a
/// corner of the cap like any other ring vert.
///
/// `10/13/4 → 14/18/5`, sizes `[5,5,5,5,6]`: the pentagonal cap over the five ring
/// verts, three quads grown to pentagons, and the one quad whose disk sector holds the
/// wire grown to a **hexagon** (it picks up the wire's ring vert on top of the two
/// every rebuilt face gains). Counting the wire *out* of the disk — the edge-mode rule
/// — would give four ring verts, a quad cap, no hexagon, and a wire hanging off a
/// consumed vertex.
///
/// *Which* face becomes the hexagon is a disk-order artefact and **not** an invariant
/// (§7.2.1: a wire shares no face with either disk neighbour), so nothing here pins it.
#[test]
fn vertex_bevel_wire_spoke_is_a_full_ring_member_of_the_disk() {
    let mut mesh = LoopMesh::new();
    let (centre, tip, _) = grid_3x3_with_wire(&mut mesh, Vec3::new(1.0, 1.0, 1.0));
    assert_eq!(counts(&mesh), (10, 13, 4));
    assert_eq!(
        mesh.iter_disk_edges(centre).count(),
        5,
        "the full disk — four grid legs and the wire",
    );

    let result = mesh.bevel(
        &[centre.into()],
        BevelOptions {
            offset: 0.25,
            segments: 1,
            affect: BevelAffect::Vertices,
            ..Default::default()
        },
    );
    mesh.validate_topology();

    assert_eq!(
        counts(&mesh),
        (14, 18, 5),
        "five ring verts (the wire is one of them), five chamfer edges, four rebuilt \
         faces plus the cap",
    );
    assert_eq!(face_size_multiset(&mesh), vec![5, 5, 5, 5, 6]);
    assert_eq!(
        result.verts.len(),
        5,
        "one ring vert per incident edge, the wire included",
    );
    assert!(
        !mesh.vert_keys().contains(&centre),
        "the beveled corner is consumed",
    );

    let wire_ring = Vec3::new(1.0, 1.0, 0.25);
    assert_eq!(
        wire_landings(&mesh, tip),
        vec![wire_ring],
        "the wire survives, re-pointed onto its *own* ring vert — slid along the wire \
         by the offset, and so the nearest boundvert by construction (§8.3.0.4)",
    );

    let wire_ring_vert = result
        .verts
        .iter()
        .copied()
        .find(|&vertex| mesh.vertex_position(vertex) == wire_ring)
        .expect("the wire's ring vert is one of the minted ring verts");
    let cap = cap_face_of(&mesh, &result.verts).expect("the interior fan is capped");
    assert_eq!(mesh.face_vert_keys(cap).len(), 5);
    assert!(
        mesh.face_vert_keys(cap).contains(&wire_ring_vert),
        "the wire's ring vert is a corner of the cap, not an orphan off to one side",
    );
}

/// The other half of §7.2.1's box: exactly **one** rebuilt original grows by *two*
/// sides — the one whose disk sector holds the wire — and it is that face which adopts
/// the wire's ring vert. Every other rebuilt face grows by one, per §6.1. Which face it
/// is stays unasserted (a disk-order artefact, §7.2.1); that it is exactly one, and
/// that it carries the wire's ring vert, is the invariant.
#[test]
fn vertex_bevel_exactly_one_rebuilt_face_adopts_the_wire_ring_vert() {
    let mut mesh = LoopMesh::new();
    let (centre, _, _) = grid_3x3_with_wire(&mut mesh, Vec3::new(1.0, 1.0, 1.0));

    let result = mesh.bevel(
        &[centre.into()],
        BevelOptions {
            offset: 0.25,
            segments: 1,
            affect: BevelAffect::Vertices,
            ..Default::default()
        },
    );
    mesh.validate_topology();

    let wire_ring_vert = result
        .verts
        .iter()
        .copied()
        .find(|&vertex| mesh.vertex_position(vertex) == Vec3::new(1.0, 1.0, 0.25))
        .expect("the wire's ring vert");
    let cap = cap_face_of(&mesh, &result.verts).expect("the interior fan is capped");

    let adopters: Vec<FaceKey> = mesh
        .face_keys()
        .into_iter()
        .filter(|&face| face != cap && mesh.face_vert_keys(face).contains(&wire_ring_vert))
        .collect();
    assert_eq!(
        adopters.len(),
        1,
        "one rebuilt original picks the wire's ring vert up as an extra corner",
    );
    assert_eq!(
        mesh.face_vert_keys(adopters[0]).len(),
        6,
        "so that quad grows by two sides, not one",
    );
    for face in mesh.face_keys() {
        if face != cap && face != adopters[0] {
            assert_eq!(
                mesh.face_vert_keys(face).len(),
                5,
                "every other rebuilt quad grows by exactly one (§6.1)",
            );
        }
    }
}

/// The wire's ring vert is placed by the ordinary §4 slide with the **wire itself** as
/// the slide edge, so an off-axis tip carries it off-axis too — and the whole answer is
/// the on-axis answer's shape with the ring vert moved. This is the fixture that
/// discriminates: a wire ring vert placed by anything but its own edge's direction
/// (dropped onto the grid plane, say) would land on the axis regardless of the tip.
#[test]
fn vertex_bevel_wire_ring_vert_slides_along_the_wire() {
    let mut mesh = LoopMesh::new();
    // The centre is `(1, 1, 0)`; the tip is one 3-4-5 leg away from it.
    let tip_position = Vec3::new(1.0, 1.0, 0.0) + Vec3::new(3.0, 0.0, 4.0);
    let (centre, tip, _) = grid_3x3_with_wire(&mut mesh, tip_position);

    mesh.bevel(
        &[centre.into()],
        BevelOptions {
            offset: 0.5,
            segments: 1,
            affect: BevelAffect::Vertices,
            ..Default::default()
        },
    );
    mesh.validate_topology();

    assert_eq!(counts(&mesh), (14, 18, 5));
    assert_eq!(face_size_multiset(&mesh), vec![5, 5, 5, 5, 6]);

    // 0.5 along the unit (3,0,4)/5 wire direction from the centre (1,1,0).
    let expected = Vec3::new(1.3, 1.0, 0.4);
    let landings = wire_landings(&mesh, tip);
    assert_eq!(landings.len(), 1, "the wire survives as one face-less edge");
    assert!(
        (landings[0] - expected).length() < 1e-5,
        "the wire lands on its own ring vert, slid `offset` along the wire: got {:?}",
        landings[0],
    );
}

/// A wire on an **open** fan, pointing squarely into a sector that *does* carry a face —
/// and §7.2.1's vertex-mode box still slots it into the **gap**, "regardless of where the
/// wire points geometrically". The chain of an open fan runs from one boundary spoke to the
/// other and stops, so where it runs out is the faceless gap, not a seam inside a sector.
/// So **no rebuilt face adopts it**: every rebuilt face grows by exactly one (§6.1) and the
/// wire's ring vert is a corner of the **cap** alone — which exists, an open fan of valence
/// >= 3 being capped like any other (§6.2). This is the fixture that discriminates: handing
/// the wire to a chain-end face (the *closed*-fan answer) diverges in counts and in the
/// face-size multiset, not merely in a tie.
#[test]
fn vertex_bevel_open_fan_wire_lands_in_the_gap_not_the_sector_it_points_into() {
    let (mut mesh, centre, _) = open_planar_fan(3, 0);
    // 45° — squarely inside the sector of the `+x` / `+y` triangle.
    let tip = mesh.create_vertex(Vec3::new(0.6, 0.6, 0.0));
    mesh.create_edge([centre, tip], EdgeCreateFlags::NONE);
    assert_eq!(counts(&mesh), (5, 6, 2));

    let result = mesh.bevel(
        &[centre.into()],
        BevelOptions {
            offset: 0.2,
            segments: 1,
            affect: BevelAffect::Vertices,
            ..Default::default()
        },
    );
    mesh.validate_topology();

    assert_eq!(
        counts(&mesh),
        (8, 10, 3),
        "valence 4: +3 V / +4 E / +1 F — two chamfer edges minted by the re-stitch, two \
         gap edges minted by the cap",
    );
    assert_eq!(
        face_size_multiset(&mesh),
        vec![4, 4, 4],
        "each triangle grown by one, plus one quad cap over all four ring verts",
    );

    let slide = 0.2 / 2.0f32.sqrt();
    let wire_ring = Vec3::new(slide, slide, 0.0);
    let landings = wire_landings(&mesh, tip);
    assert_eq!(landings.len(), 1, "the wire survives as one face-less edge");
    assert!(
        (landings[0] - wire_ring).length() < 1e-5,
        "re-pointed onto its own ring vert, slid along the wire: got {:?}",
        landings[0],
    );

    let wire_ring_vert = result
        .verts
        .iter()
        .copied()
        .find(|&vertex| (mesh.vertex_position(vertex) - wire_ring).length() < 1e-5)
        .expect("the wire's ring vert");
    let cap = cap_face_of(&mesh, &result.verts).expect("an open fan of valence 3+ is capped");
    let adopters: Vec<FaceKey> = mesh
        .face_keys()
        .into_iter()
        .filter(|&face| face != cap && mesh.face_vert_keys(face).contains(&wire_ring_vert))
        .collect();
    assert!(
        adopters.is_empty(),
        "no rebuilt face adopts a gap wire — the cap alone catches it",
    );
    assert!(mesh.face_vert_keys(cap).contains(&wire_ring_vert));
}

/// The same open fan with the wire pointing into the face-less **gap** instead: same slot,
/// same answer. The gap is where the chain runs out either way, so this fixture and the one
/// above agree down to the face-size multiset — the geometric bearing of the wire never
/// enters into it on an open fan.
#[test]
fn vertex_bevel_open_fan_wire_in_the_boundary_gap_is_caught_by_the_cap() {
    let (mut mesh, centre, _, tip) = wire_spoke_open_fan();
    assert_eq!(counts(&mesh), (5, 6, 2));

    let result = mesh.bevel(
        &[centre.into()],
        BevelOptions {
            offset: 0.2,
            segments: 1,
            affect: BevelAffect::Vertices,
            ..Default::default()
        },
    );
    mesh.validate_topology();

    assert_eq!(counts(&mesh), (8, 10, 3));
    assert_eq!(face_size_multiset(&mesh), vec![4, 4, 4]);
    assert!(
        !mesh.vert_keys().contains(&centre),
        "the corner is beveled, not declined — so it is consumed",
    );

    // The tip is at (0,-1,0), straight down into the gap; the ring vert is `offset`
    // along the wire from the centre at the origin.
    let wire_ring = Vec3::new(0.0, -0.2, 0.0);
    let landings = wire_landings(&mesh, tip);
    assert_eq!(landings.len(), 1, "the wire survives as one face-less edge");
    assert!(
        (landings[0] - wire_ring).length() < 1e-5,
        "re-pointed onto its own ring vert, slid along the wire: got {:?}",
        landings[0],
    );

    let wire_ring_vert = result
        .verts
        .iter()
        .copied()
        .find(|&vertex| (mesh.vertex_position(vertex) - wire_ring).length() < 1e-5)
        .expect("the gap wire's ring vert");
    let cap = cap_face_of(&mesh, &result.verts).expect("an open fan of valence 3+ is capped");
    assert_eq!(mesh.face_vert_keys(cap).len(), 4);
    assert!(mesh.face_vert_keys(cap).contains(&wire_ring_vert));
    for face in mesh.face_keys() {
        if face != cap {
            assert_eq!(
                mesh.face_vert_keys(face).len(),
                4,
                "every rebuilt face grows by exactly one — none hosts the gap wire",
            );
        }
    }
}

/// §6.2.1, the extreme corner: an **all-wire star**, face-less, three wires. Nothing about
/// vertex bevel requires a face (§2.1) and the general rule answers with no special case —
/// every step of the disk is a gap, so *no* ring edge comes from a re-stitch and the cap
/// mints all `n` of them. `4/3/0 → 6/6/1`: three ring verts, three re-pointed wire stubs,
/// three cap-minted ring edges, and one free-floating triangular cap.
#[test]
fn vertex_bevel_face_less_wire_star_is_capped_by_the_cap_alone() {
    let mut mesh = LoopMesh::new();
    let centre = mesh.create_vertex(Vec3::ZERO);
    let tips: Vec<VertexKey> = [
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(-1.0, 0.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
    ]
    .into_iter()
    .map(|position| mesh.create_vertex(position))
    .collect();
    for &tip in &tips {
        mesh.create_edge([centre, tip], EdgeCreateFlags::NONE);
    }
    assert_eq!(counts(&mesh), (4, 3, 0));

    let result = mesh.bevel(
        &[centre.into()],
        BevelOptions {
            offset: 0.25,
            segments: 1,
            affect: BevelAffect::Vertices,
            ..Default::default()
        },
    );
    mesh.validate_topology();

    assert_eq!(
        counts(&mesh),
        (6, 6, 1),
        "the face-less star is beveled like any other valence-3 corner",
    );
    assert_eq!(result.verts.len(), 3, "one ring vert per wire");
    assert_eq!(result.faces.len(), 1, "one triangular cap");
    assert_eq!(face_size_multiset(&mesh), vec![3]);
    assert!(
        !mesh.vert_keys().contains(&centre),
        "the hub is consumed like any beveled corner",
    );

    let cap = result.faces[0];
    let corners = mesh.face_vert_keys(cap);
    for expected in [
        Vec3::new(0.25, 0.0, 0.0),
        Vec3::new(-0.25, 0.0, 0.0),
        Vec3::new(0.0, 0.25, 0.0),
    ] {
        assert!(
            corners
                .iter()
                .any(|&corner| (mesh.vertex_position(corner) - expected).length() < 1e-5),
            "the cap's corners are the beveled ends of the wires: {expected:?} missing",
        );
    }
    for (&tip, expected) in tips.iter().zip([
        Vec3::new(0.25, 0.0, 0.0),
        Vec3::new(-0.25, 0.0, 0.0),
        Vec3::new(0.0, 0.25, 0.0),
    ]) {
        let landings = wire_landings(&mesh, tip);
        assert_eq!(landings.len(), 1, "each wire survives, re-pointed");
        assert!((landings[0] - expected).length() < 1e-5);
    }
}

/// §6.2.1's valence-2 half: a two-wire hub has no cap (a two-corner n-gon is not a face,
/// §2.3) — but the single ring edge is **still** minted, and here nothing else could have
/// minted it, there being no face to re-stitch. `3/2/0 → 4/3/0`: the two wires end up
/// joined by a new wire edge and the hub is gone. §6.3's carve-out reports that edge,
/// because a face-less corner's ring edge bounds nothing at all.
#[test]
fn vertex_bevel_two_wire_hub_mints_its_ring_edge_with_no_cap() {
    let mut mesh = LoopMesh::new();
    let hub = mesh.create_vertex(Vec3::ZERO);
    let east = mesh.create_vertex(Vec3::new(1.0, 0.0, 0.0));
    let north = mesh.create_vertex(Vec3::new(0.0, 1.0, 0.0));
    mesh.create_edge([hub, east], EdgeCreateFlags::NONE);
    mesh.create_edge([hub, north], EdgeCreateFlags::NONE);
    assert_eq!(counts(&mesh), (3, 2, 0));

    let result = mesh.bevel(
        &[hub.into()],
        BevelOptions {
            offset: 0.25,
            segments: 1,
            affect: BevelAffect::Vertices,
            ..Default::default()
        },
    );
    mesh.validate_topology();

    assert_eq!(counts(&mesh), (4, 3, 0), "+1 V / +1 E / +0 F");
    assert!(result.faces.is_empty(), "no cap at valence 2");
    assert_eq!(result.verts.len(), 2);
    assert!(!mesh.vert_keys().contains(&hub), "the hub is consumed");

    assert_eq!(
        result.edges.len(),
        1,
        "the face-less corner's ring edge bounds nothing and is reported (§6.3)",
    );
    let ring_edge = result.edges[0];
    assert_eq!(
        mesh.edge_radial_iter(ring_edge).count(),
        0,
        "and it is itself a wire",
    );
    let [a, b] = mesh.edge_vert_keys(ring_edge);
    let ends = [mesh.vertex_position(a), mesh.vertex_position(b)];
    assert!(ends.contains(&Vec3::new(0.25, 0.0, 0.0)));
    assert!(ends.contains(&Vec3::new(0.0, 0.25, 0.0)));
}

/// A wire whose **both** endpoints are beveled. Each end re-points the wire onto its
/// own ring vert (§8.3.0.4's re-point spelling), and the `(edge, endpoint)` guard makes
/// the two swaps land on the *same* edge — so the wire survives as exactly **one**
/// face-less edge joining the two rings' wire ring verts, not two, and not one dropped.
///
/// The far grid is deliberately **off-axis**: a wire running along `+x` would be
/// collinear with each centre's own `+x` grid leg and its ring vert would land exactly
/// on that leg's, so the position pin below could not tell the two apart and the mesh
/// would carry a zero-length chamfer edge. Off-axis, the wire's ring vert is the only
/// vert at its position.
#[test]
fn vertex_bevel_wire_beveled_at_both_ends_survives_as_one_face_less_edge() {
    let mut mesh = LoopMesh::new();
    let (near_centre, _) = grid_3x3(&mut mesh, Vec3::ZERO);
    let (far_centre, _) = grid_3x3(&mut mesh, Vec3::new(4.0, 1.5, 0.0));
    mesh.create_edge([near_centre, far_centre], EdgeCreateFlags::NONE);
    assert_eq!(counts(&mesh), (18, 25, 8));

    let offset = 0.25;
    let near_position = mesh.vertex_position(near_centre);
    let far_position = mesh.vertex_position(far_centre);
    let along = (far_position - near_position).normalize();

    mesh.bevel(
        &[near_centre.into(), far_centre.into()],
        BevelOptions {
            offset,
            segments: 1,
            affect: BevelAffect::Vertices,
            ..Default::default()
        },
    );
    mesh.validate_topology();

    assert_eq!(
        counts(&mesh),
        (26, 35, 10),
        "five ring verts and one cap per corner; the wire is not doubled",
    );

    let face_less: Vec<[Vec3; 2]> = mesh
        .edge_keys()
        .into_iter()
        .filter(|&edge| mesh.edge_radial_iter(edge).next().is_none())
        .map(|edge| {
            let [a, b] = mesh.edge_vert_keys(edge);
            [mesh.vertex_position(a), mesh.vertex_position(b)]
        })
        .collect();
    assert_eq!(face_less.len(), 1, "exactly one face-less edge survives");
    let [a, b] = face_less[0];
    let (a, b) = if a.x < b.x { (a, b) } else { (b, a) };
    let near_ring = near_position + along * offset;
    let far_ring = far_position - along * offset;
    assert!(
        (a - near_ring).length() < 1e-5 && (b - far_ring).length() < 1e-5,
        "it joins the two rings' wire ring verts, each slid `offset` along the wire: \
         got {a:?} — {b:?}, want {near_ring:?} — {far_ring:?}",
    );
    for ring in [near_ring, far_ring] {
        assert_eq!(
            mesh.vert_keys()
                .into_iter()
                .filter(|&v| (mesh.vertex_position(v) - ring).length() < 1e-5)
                .count(),
            1,
            "the wire's ring vert is the only vert there — no grid leg shares it",
        );
    }
}

/// The rotational sector read on **non-planar** geometry: a cube corner (valence 3, three
/// quads, no two coplanar) with a wire along the tangent-plane bisector of two of its
/// spokes. The wire falls squarely in that pair's sector, so the face those two spokes
/// bound hosts it — run once per sector, which is what makes this discriminating: a read
/// that fell back to the face chain's end would hand all three wires to the *same* face.
#[test]
fn vertex_bevel_wire_on_a_nonplanar_closed_fan_is_hosted_by_its_sector_face() {
    let corner_position = Vec3::new(1.0, -1.0, -1.0);
    let spokes = [
        Vec3::new(-1.0, -1.0, -1.0),
        Vec3::new(1.0, 1.0, -1.0),
        Vec3::new(1.0, -1.0, 1.0),
    ];
    for pair in [(0, 1), (0, 2), (1, 2)] {
        let (first, second) = (spokes[pair.0], spokes[pair.1]);
        let mut mesh = LoopMesh::new();
        mesh_add_cube(&mut mesh);
        let corner = vert_at(&mesh, corner_position);

        // The two spokes leave the corner at a right angle, so their sum bisects their
        // sector — and the bisector of a sector is inside it, whichever way the corner
        // folds. The face they bound is the cube face carrying both of them, whose
        // fourth vertex is the corner's diagonal across it.
        let bisector =
            (first - corner_position).normalize() + (second - corner_position).normalize();
        let tip = mesh.create_vertex(corner_position + bisector * 0.6);
        mesh.create_edge([corner, tip], EdgeCreateFlags::NONE);
        let diagonal = vert_at(
            &mesh,
            corner_position + (first - corner_position) + (second - corner_position),
        );

        let result = mesh.bevel(
            &[corner.into()],
            BevelOptions {
                offset: 0.25,
                segments: 1,
                affect: BevelAffect::Vertices,
                ..Default::default()
            },
        );
        mesh.validate_topology();

        assert_eq!(
            counts(&mesh),
            (12, 17, 7),
            "four ring verts (the wire is one), four chamfer edges, a quad cap",
        );
        assert_eq!(face_size_multiset(&mesh), vec![4, 4, 4, 4, 5, 5, 6]);

        let wire_ring = corner_position + bisector.normalize() * 0.25;
        let wire_ring_vert = result
            .verts
            .iter()
            .copied()
            .find(|&vertex| (mesh.vertex_position(vertex) - wire_ring).length() < 1e-5)
            .expect("the wire's ring vert, slid along the wire");
        let cap = cap_face_of(&mesh, &result.verts).expect("the closed fan is capped");
        let host = mesh
            .vert_face_keys(wire_ring_vert)
            .into_iter()
            .find(|&face| face != cap)
            .expect("a rebuilt original adopts the wire's ring vert");

        assert_eq!(
            mesh.face_vert_keys(host).len(),
            6,
            "the host quad grows by two sides",
        );
        assert!(
            mesh.face_vert_keys(host).contains(&diagonal),
            "and it is the face bounded by the wire's two sector spokes — pinned by \
             the corner's diagonal across it, which only that face carries",
        );
    }
}

/// Two wires in the **same** sector: both are ring members, both are hosted by the one
/// face spanning that sector (which therefore grows by *three* sides), and the host's
/// corner run threads them in the order they sweep the sector — so the rebuilt face's
/// cycle reads leading ring vert, far wire, near wire, trailing ring vert, and not the
/// disk order they happened to be created in.
#[test]
fn vertex_bevel_two_wires_in_one_sector_are_threaded_in_sweep_order() {
    let mut mesh = LoopMesh::new();
    let (centre, _) = grid_3x3(&mut mesh, Vec3::ZERO);
    let centre_position = Vec3::new(1.0, 1.0, 0.0);
    // Both inside the `+x` / `+y` quad's sector, at 26.6° and 63.4° off the `+x` leg.
    let shallow_tip = mesh.create_vertex(centre_position + Vec3::new(2.0, 1.0, 0.0));
    let steep_tip = mesh.create_vertex(centre_position + Vec3::new(1.0, 2.0, 0.0));
    mesh.create_edge([centre, shallow_tip], EdgeCreateFlags::NONE);
    mesh.create_edge([centre, steep_tip], EdgeCreateFlags::NONE);
    assert_eq!(counts(&mesh), (11, 14, 4));

    let result = mesh.bevel(
        &[centre.into()],
        BevelOptions {
            offset: 0.25,
            segments: 1,
            affect: BevelAffect::Vertices,
            ..Default::default()
        },
    );
    mesh.validate_topology();

    assert_eq!(
        counts(&mesh),
        (16, 20, 5),
        "six ring verts (four legs and both wires), six chamfer edges, a hexagonal cap",
    );
    assert_eq!(
        face_size_multiset(&mesh),
        vec![5, 5, 5, 6, 7],
        "the one face spanning both wires' sector grows by three sides",
    );

    let ring_at = |mesh: &LoopMesh, position: Vec3| {
        result
            .verts
            .iter()
            .copied()
            .find(|&vertex| (mesh.vertex_position(vertex) - position).length() < 1e-5)
            .expect("a ring vert at the slid position")
    };
    let slide = |direction: Vec3| centre_position + direction.normalize() * 0.25;
    let x_leg = ring_at(&mesh, slide(Vec3::X));
    let y_leg = ring_at(&mesh, slide(Vec3::Y));
    let shallow = ring_at(&mesh, slide(Vec3::new(2.0, 1.0, 0.0)));
    let steep = ring_at(&mesh, slide(Vec3::new(1.0, 2.0, 0.0)));

    let host = mesh
        .face_keys()
        .into_iter()
        .find(|&face| mesh.face_vert_keys(face).len() == 7)
        .expect("the host grew to a heptagon");
    let cycle = mesh.face_vert_keys(host);
    // The run enters along the face's incoming edge (the `+y` leg, the face winds
    // counter-clockwise about `+z`) and leaves along the `+x` one, sweeping the sector
    // clockwise: `+y`, steep wire, shallow wire, `+x`.
    let run = [y_leg, steep, shallow, x_leg];
    let start = cycle
        .iter()
        .position(|&vertex| vertex == run[0])
        .expect("the host's run starts at the leading ring vert");
    for (step, &expected) in run.iter().enumerate() {
        assert_eq!(
            cycle[(start + step) % cycle.len()],
            expected,
            "the host's corner run threads both wires in sweep order",
        );
    }
}
