use super::*;

/// A unit cube with the `y = +1` face's loop cycle reversed, so both of that face's
/// loops traverse each of its four edges the *same* way its neighbour does: every
/// edge of it is an inconsistently-wound manifold radial pair (§7.8). Every endpoint
/// of every such edge has **three** face-sectors, so §7.8.1's vote is decisive and
/// the minority (reversed) face is the disagreeing flank at both ends.
fn cube_with_one_reversed_face() -> (LoopMesh, EdgeKey, FaceKey) {
    let mut mesh = LoopMesh::new();
    mesh_add_cube(&mut mesh);
    let top = *mesh
        .face_keys()
        .iter()
        .find(|&&f| mesh.face_plane_normal(f).y > 0.9)
        .expect("the cube's +y face");
    mesh.euler_loop_reverse(top);
    let edge = mesh
        .query_edge_exists(
            vert_at(&mesh, Vec3::new(-1.0, 1.0, -1.0)),
            vert_at(&mesh, Vec3::new(1.0, 1.0, -1.0)),
        )
        .expect("an edge of the reversed face");
    (mesh, edge, top)
}

fn bevel_one_edge(mesh: &mut LoopMesh, edge: EdgeKey, segments: u32) -> BevelResult {
    mesh.bevel(
        &[edge.into()],
        BevelOptions {
            offset: 0.2,
            affect: BevelAffect::Edges,
            segments,
            ..Default::default()
        },
    )
}

fn verts_near(mesh: &LoopMesh, position: Vec3) -> Vec<VertexKey> {
    mesh.vert_keys()
        .into_iter()
        .filter(|&v| (mesh.vertex_position(v) - position).length() < 1.0e-4)
        .collect()
}

/// §7.8.2 / §7.8.3, decisive vote: the reversed flank's strip vert meets its leg
/// *behind* the endpoint, the §7.5.2 span clamp snaps it back, and it is minted at
/// the killed endpoint's own coordinates. The count delta is the consistent case's,
/// and the collapsed end is **one** vert element, not two coincident ones.
#[test]
fn edge_bevel_noncontiguous_flanks_collapse_onto_the_endpoints() {
    let endpoints = [Vec3::new(-1.0, 1.0, -1.0), Vec3::new(1.0, 1.0, -1.0)];
    let (mut mesh, edge, _) = cube_with_one_reversed_face();
    bevel_one_edge(&mut mesh, edge, 1);

    assert_eq!(counts(&mesh), (10, 15, 7), "the §7.8.3 count delta");
    for endpoint in endpoints {
        let coincident = verts_near(&mesh, endpoint);
        assert_eq!(
            coincident.len(),
            1,
            "exactly one fresh vert reproduces the killed endpoint {endpoint} \
             (two would be the §7.8.3 double-mint)",
        );
        // Shared by the strip and the rebuilt flank, so it is a full interior corner.
        assert_eq!(
            mesh.vert_face_keys(coincident[0]).len(),
            3,
            "the collapsed rung end is shared by every face that reaches it",
        );
    }
    mesh.validate_topology();
}

/// §7.8.3: the disagreeing flank is rebuilt with its corners unmoved, so it comes
/// out geometrically **unchanged** — the visible "bevel ate one face and left the
/// other alone". The agreeing flank does shrink, and the strip is half-width.
#[test]
fn edge_bevel_noncontiguous_collapse_lands_on_the_reversed_face() {
    let (mut mesh, edge, reversed) = cube_with_one_reversed_face();
    let before: Vec<Vec3> = mesh
        .face_vert_keys(reversed)
        .into_iter()
        .map(|v| mesh.vertex_position(v))
        .collect();
    bevel_one_edge(&mut mesh, edge, 1);

    let survivor: Vec<Vec<Vec3>> = mesh
        .face_keys()
        .into_iter()
        .map(|f| {
            mesh.face_vert_keys(f)
                .into_iter()
                .map(|v| mesh.vertex_position(v))
                .collect()
        })
        .collect();
    assert!(
        survivor
            .iter()
            .any(|cycle| cycle_matches(&before, cycle, 1.0e-4)),
        "the mis-wound flank is rebuilt at its original corners",
    );
}

/// §7.8.3 / §5.1: the strip quad's **absolute winding** on a decisive-vote mis-wound
/// pair. The labels come out mirrored there, so each flank's pair *is* a rail and both
/// rails are owned — but the two flanks traverse `E` the same way, so "wind opposite
/// the flank on rail 1" and "wind opposite the flank on rail 2" are contradictory
/// conditions. The strip is one surface and satisfies exactly one of them, and *which*
/// must be a fixed choice of flank rather than whichever rail the side labels happened
/// to emit first — otherwise the winding drifts with the endpoint's disk order, an
/// artefact §7.8.6 warns is a construction accident. The nominated flank is `E`'s
/// first radial face.
///
/// Every other §7.8.3 test compares faces with `cycle_matches`, which is blind to
/// reflection; this is the one that sees handedness.
#[test]
fn edge_bevel_decisive_vote_strip_winds_against_the_first_radial_face() {
    let (mut mesh, edge, _reversed) = cube_with_one_reversed_face();
    let radial: Vec<FaceKey> = mesh
        .edge_loop_keys(edge)
        .iter()
        .map(|&l| mesh.loop_face(l))
        .collect();
    let (anchor_flank, other_flank) = (radial[0], radial[1]);
    let result = bevel_one_edge(&mut mesh, edge, 1);
    assert_eq!(result.faces.len(), 1, "a valence-3 terminal builds no fill");
    let strip = result.faces[0];

    let traverses = |mesh: &LoopMesh, face: FaceKey, from: VertexKey, to: VertexKey| {
        let verts = mesh.face_vert_keys(face);
        (0..verts.len()).any(|i| verts[i] == from && verts[(i + 1) % verts.len()] == to)
    };
    // For each flank, how the strip runs the rail they share: `Some(true)` when the
    // flank runs it the *same* way (the §5.1 conflict), `Some(false)` when opposite.
    let shared_run = |mesh: &LoopMesh, flank: FaceKey| -> Option<bool> {
        let strip_verts = mesh.face_vert_keys(strip);
        let mut found = None;
        for index in 0..strip_verts.len() {
            let from = strip_verts[index];
            let to = strip_verts[(index + 1) % strip_verts.len()];
            if traverses(mesh, flank, from, to) {
                assert!(found.is_none(), "one shared rail per flank");
                found = Some(true);
            } else if traverses(mesh, flank, to, from) {
                assert!(found.is_none(), "one shared rail per flank");
                found = Some(false);
            }
        }
        found
    };

    assert_eq!(
        shared_run(&mesh, anchor_flank),
        Some(false),
        "the strip runs the first radial face's rail opposite that face (§5.1)",
    );
    assert_eq!(
        shared_run(&mesh, other_flank),
        Some(true),
        "the other flank's rail carries the conflict the mis-wound pair forces",
    );
    mesh.validate_topology();
}

/// §7.8.4: segments-independent. The collapse reproduces at every segment count —
/// the profile is simply laid across the half-width rung — and the counts stay the
/// consistently-wound cube's at the same `segments`.
#[test]
fn edge_bevel_noncontiguous_collapses_at_every_segment_count() {
    for segments in 1..=4 {
        let (mut mesh, edge, _) = cube_with_one_reversed_face();
        bevel_one_edge(&mut mesh, edge, segments);

        let mut consistent = LoopMesh::new();
        mesh_add_cube(&mut consistent);
        let control = consistent
            .query_edge_exists(
                vert_at(&consistent, Vec3::new(-1.0, 1.0, -1.0)),
                vert_at(&consistent, Vec3::new(1.0, 1.0, -1.0)),
            )
            .expect("the same cube edge");
        bevel_one_edge(&mut consistent, control, segments);

        assert_eq!(
            counts(&mesh),
            counts(&consistent),
            "segments {segments}: §7.8.3's identical count delta",
        );
        for endpoint in [Vec3::new(-1.0, 1.0, -1.0), Vec3::new(1.0, 1.0, -1.0)] {
            assert_eq!(
                verts_near(&mesh, endpoint).len(),
                1,
                "segments {segments}: the endpoint position is reproduced exactly once",
            );
            assert!(
                verts_near(&consistent, endpoint).is_empty(),
                "segments {segments}: the consistent contrast slides both strip verts",
            );
        }
        mesh.validate_topology();
    }
}

/// §7.8.1(a): the ballot normalisation is what makes the decisive-vote regime
/// deterministic. The disk walk seeds on an arbitrary incident face, so without it
/// a mesh whose faces were *built* in a different order would collapse the other
/// flank. Rebuilding the same cube with its faces created in every rotation must
/// still collapse the reversed one.
#[test]
fn edge_bevel_noncontiguous_collapse_is_independent_of_face_build_order() {
    for rotation in 0..6 {
        // The same geometry with its faces *created* in a rotated order, so the
        // corner ring's arbitrary seed face differs run to run.
        let mut mesh = LoopMesh::new();
        let corner: Vec<VertexKey> = [
            Vec3::new(1.0, -1.0, -1.0),
            Vec3::new(-1.0, -1.0, -1.0),
            Vec3::new(-1.0, 1.0, -1.0),
            Vec3::new(1.0, 1.0, -1.0),
            Vec3::new(1.0, -1.0, 1.0),
            Vec3::new(-1.0, -1.0, 1.0),
            Vec3::new(-1.0, 1.0, 1.0),
            Vec3::new(1.0, 1.0, 1.0),
        ]
        .into_iter()
        .map(|p| mesh.create_vertex(p))
        .collect();
        let quads = [
            [0, 1, 2, 3],
            [1, 5, 6, 2],
            [5, 4, 7, 6],
            [4, 0, 3, 7],
            [3, 2, 6, 7],
            [0, 4, 5, 1],
        ];
        for step in 0..quads.len() {
            let quad = quads[(step + rotation) % quads.len()];
            let ring: Vec<VertexKey> = quad.iter().map(|&i| corner[i]).collect();
            mesh.face_create_verts(&ring, FaceCreateFlags::NONE);
        }
        let top = *mesh
            .face_keys()
            .iter()
            .find(|&&f| mesh.face_plane_normal(f).y > 0.9)
            .expect("the cube's +y face");
        mesh.euler_loop_reverse(top);
        let before: Vec<Vec3> = mesh
            .face_vert_keys(top)
            .into_iter()
            .map(|v| mesh.vertex_position(v))
            .collect();
        let edge = mesh
            .query_edge_exists(
                vert_at(&mesh, Vec3::new(-1.0, 1.0, -1.0)),
                vert_at(&mesh, Vec3::new(1.0, 1.0, -1.0)),
            )
            .expect("an edge of the reversed face");
        bevel_one_edge(&mut mesh, edge, 1);

        assert_eq!(counts(&mesh), (10, 15, 7), "rotation {rotation}");
        let survivor: Vec<Vec<Vec3>> = mesh
            .face_keys()
            .into_iter()
            .map(|f| {
                mesh.face_vert_keys(f)
                    .into_iter()
                    .map(|v| mesh.vertex_position(v))
                    .collect()
            })
            .collect();
        assert!(
            survivor
                .iter()
                .any(|cycle| cycle_matches(&before, cycle, 1.0e-4)),
            "rotation {rotation}: the collapse still lands on the reversed face",
        );
    }
}

/// §7.8.1's **tied-vote** regime: an endpoint with exactly two face-sectors (the
/// beveled edge's own two flanks and nothing else) casts `+1` and `−1`, the vote
/// cancels, no normalisation happens, and *which* flank collapses follows the stored
/// disk order — so the two endpoints decide independently and can pick opposite
/// flanks. What is regime-independent is the **placement**: one rung end reproduces
/// the killed endpoint exactly, the other slides the offset along its own leg, and
/// the rung is half-width. That much is asserted here; the resulting rail pairing
/// (§7.8.3's twist, and the edge count that follows from it) is explicitly *not* an
/// invariant in the guide and is left unpinned.
#[test]
fn edge_bevel_noncontiguous_tied_vote_still_builds_one_sided_rungs() {
    let mut mesh = LoopMesh::new();
    let plane = |x: f32, y: f32| Vec3::new(x, y, 0.0);
    let corner: Vec<VertexKey> = [
        plane(0.0, 0.0),
        plane(1.0, 0.0),
        plane(1.0, 1.0),
        plane(0.0, 1.0),
        plane(2.0, 0.0),
        plane(2.0, 1.0),
    ]
    .into_iter()
    .map(|p| mesh.create_vertex(p))
    .collect();
    mesh.face_create_verts(
        &[corner[0], corner[1], corner[2], corner[3]],
        FaceCreateFlags::NONE,
    );
    // Both quads traverse the shared edge `1 → 2`: the inconsistently-wound pair.
    mesh.face_create_verts(
        &[corner[1], corner[2], corner[5], corner[4]],
        FaceCreateFlags::NONE,
    );
    let edge = mesh
        .query_edge_exists(corner[1], corner[2])
        .expect("the shared edge");
    bevel_one_edge(&mut mesh, edge, 1);

    assert_eq!(mesh.vertex_count(), 8);
    assert_eq!(mesh.face_count(), 3);
    for (endpoint, slid) in [
        (plane(1.0, 0.0), [plane(0.8, 0.0), plane(1.2, 0.0)]),
        (plane(1.0, 1.0), [plane(0.8, 1.0), plane(1.2, 1.0)]),
    ] {
        assert_eq!(
            verts_near(&mesh, endpoint).len(),
            1,
            "the collapsed rung end reproduces {endpoint} exactly once",
        );
        assert_eq!(
            slid.iter()
                .filter(|&&p| !verts_near(&mesh, p).is_empty())
                .count(),
            1,
            "the other rung end slid the offset onto exactly one of the two flanks",
        );
    }
    mesh.validate_topology();
}

/// Two unit quads in the `z = 0` plane sharing the edge from `(1, 0)` to `(1, 1)`,
/// the right-hand one wound so the pair is either contiguous or not (BEVEL.md §7.8:
/// a non-contiguous pair runs the shared edge the *same* way in both loop cycles).
/// Both endpoints are valence-3 boundary verts with exactly two face-sectors, so the
/// §7.8.1 ballot ties and neither endpoint's rotational order is normalised.
fn two_quad_plane(contiguous: bool) -> (LoopMesh, EdgeKey) {
    let mut mesh = LoopMesh::new();
    let plane = |x: f32, y: f32| Vec3::new(x, y, 0.0);
    let corner: Vec<VertexKey> = [
        plane(0.0, 0.0),
        plane(1.0, 0.0),
        plane(1.0, 1.0),
        plane(0.0, 1.0),
        plane(2.0, 0.0),
        plane(2.0, 1.0),
    ]
    .into_iter()
    .map(|position| mesh.create_vertex(position))
    .collect();
    mesh.face_create_verts(
        &[corner[0], corner[1], corner[2], corner[3]],
        FaceCreateFlags::NONE,
    );
    let right = if contiguous {
        [corner[1], corner[4], corner[5], corner[2]]
    } else {
        [corner[1], corner[2], corner[5], corner[4]]
    };
    mesh.face_create_verts(&right, FaceCreateFlags::NONE);
    let edge = mesh
        .query_edge_exists(corner[1], corner[2])
        .expect("the shared edge");
    (mesh, edge)
}

/// The face containing the given position — used to name a rebuilt flank by an
/// outer corner the bevel never touched.
fn face_through(mesh: &LoopMesh, position: Vec3) -> FaceKey {
    mesh.face_keys()
        .into_iter()
        .find(|&f| {
            mesh.face_vert_keys(f)
                .into_iter()
                .any(|v| (mesh.vertex_position(v) - position).length() < 1.0e-4)
        })
        .expect("a face reaches that corner")
}

/// How many edge elements the strip shares with the two rebuilt flanks.
fn strip_flank_shared_edges(mesh: &LoopMesh) -> usize {
    let left = face_through(mesh, Vec3::ZERO);
    let right = face_through(mesh, Vec3::new(2.0, 0.0, 0.0));
    let flank_edges: Vec<EdgeKey> = face_edge_keys(mesh, left)
        .into_iter()
        .chain(face_edge_keys(mesh, right))
        .collect();
    mesh.face_keys()
        .into_iter()
        .filter(|&f| f != left && f != right)
        .flat_map(|f| face_edge_keys(mesh, f))
        .filter(|edge| flank_edges.contains(edge))
        .count()
}

/// Every edge the strip shares with a named flank, as `(shared edges, edges the two
/// faces traverse in *opposite* directions)`. §5.1's winding rule makes the second
/// equal the first: a rail the re-stitched flank already owns must be walked backwards
/// by the strip. Unlike a cycle-up-to-reflection check this pins the strip's *absolute*
/// loop direction, since the flank named here is a pre-bevel face with a fixed winding.
fn strip_rail_windings(mesh: &LoopMesh) -> (usize, usize) {
    let left = face_through(mesh, Vec3::ZERO);
    let right = face_through(mesh, Vec3::new(2.0, 0.0, 0.0));
    let mut shared = 0;
    let mut opposed = 0;
    for face in mesh.face_keys() {
        if face == left || face == right {
            continue;
        }
        for edge in face_edge_keys(mesh, face) {
            let loops = mesh.edge_loop_keys(edge);
            let [one, two] = match loops[..] {
                [one, two] => [one, two],
                _ => continue,
            };
            let flank_side = [one, two]
                .into_iter()
                .find(|&l| matches!(mesh.loop_face(l), f if f == left || f == right));
            let Some(flank_loop) = flank_side else {
                continue;
            };
            let strip_loop = if flank_loop == one { two } else { one };
            if mesh.loop_face(strip_loop) != face {
                continue;
            }
            shared += 1;
            let flank_run = (
                mesh.loop_vertex(flank_loop),
                mesh.loop_vertex(mesh.loop_next(flank_loop)),
            );
            let strip_run = (
                mesh.loop_vertex(strip_loop),
                mesh.loop_vertex(mesh.loop_next(strip_loop)),
            );
            if flank_run == (strip_run.1, strip_run.0) {
                opposed += 1;
            }
        }
    }
    (shared, opposed)
}

/// §5.1 on both strip builders. `cycle_matches` accepts a reflected cycle, so the
/// rail-pairing tests leave the strip's absolute handedness free; this pins it against
/// the fixed winding of a flank that existed before the bevel. The isolated and
/// junction routes must agree — a winding anchor derived from record order rather than
/// from `E`'s own radial order is what would split them.
#[test]
fn strip_winds_opposite_its_flanks_rail_on_both_routes() {
    let (mut isolated_route, edge) = two_quad_plane(true);
    bevel_one_edge(&mut isolated_route, edge, 1);
    assert_eq!(
        strip_rail_windings(&isolated_route),
        (2, 2),
        "isolated route: both rails walked backwards by the strip",
    );

    let (mut junction_route, isolated, junction) = plane_plus_cube_junction(true);
    bevel_count_delta(&mut junction_route, &[isolated, junction[0], junction[1]]);
    assert_eq!(
        strip_rail_windings(&junction_route),
        (2, 2),
        "junction route: the same strip, wound the same way",
    );
}

/// §7.8.3's rail rule on a **contiguous** pair — the regime where the labels come
/// out mirrored, so each rebuilt flank's own two strip-side verts *are* a rail. The
/// cross-labelled pairing must reproduce the §7.1 quad exactly: one shared edge per
/// flank, and the consistent 8 V / 10 E / 3 F.
#[test]
fn edge_bevel_strip_rails_are_the_flanks_own_pairs_when_labels_are_mirrored() {
    let (mut mesh, edge) = two_quad_plane(true);
    bevel_one_edge(&mut mesh, edge, 1);

    assert_eq!(counts(&mesh), (8, 10, 3));
    assert_eq!(
        strip_flank_shared_edges(&mesh),
        2,
        "each rebuilt flank shares exactly one rail with the strip",
    );
    mesh.validate_topology();
}

/// §7.8.3's tied-vote twist. Both endpoints keep their un-normalised order and land
/// on the **same** handedness, so a flank fills the same rotational slot at both
/// ends and its pair of strip-side verts is *like*-labelled. The rails join opposite
/// labels — `P(a)—N(b)` and `N(a)—P(b)` — so neither is any flank's pair: the strip
/// is edge-disjoint from both, which is the crack, and the two rails are minted
/// fresh rather than re-used (12 E, against the 10 E a like-label pairing gives on
/// the same four verts at the same four positions).
#[test]
fn edge_bevel_tied_vote_twist_crosses_the_strips_rails() {
    let (mut mesh, edge) = two_quad_plane(false);
    bevel_one_edge(&mut mesh, edge, 1);

    assert_eq!(counts(&mesh), (8, 12, 3), "the §7.8.3 twisted count delta");
    assert_eq!(
        strip_flank_shared_edges(&mesh),
        0,
        "the crack: the strip shares no edge with either rebuilt flank",
    );

    let plane = |x: f32, y: f32| Vec3::new(x, y, 0.0);
    // `cA · cB · sB · sA`: the two collapsed rung ends are one rail, the two slid
    // ones the other. The like-label cycle `cA · sB · cB · sA` has the same four
    // verts and would pass a position-set check.
    let expected = [
        plane(1.0, 0.0),
        plane(1.0, 1.0),
        plane(1.2, 1.0),
        plane(0.8, 0.0),
    ];
    let strip: Vec<Vec3> = mesh
        .face_keys()
        .into_iter()
        .find(|&f| {
            f != face_through(&mesh, Vec3::ZERO) && f != face_through(&mesh, plane(2.0, 0.0))
        })
        .map(|f| {
            mesh.face_vert_keys(f)
                .into_iter()
                .map(|v| mesh.vertex_position(v))
                .collect()
        })
        .expect("the strip quad");
    assert!(
        cycle_matches(&expected, &strip, 1.0e-4),
        "the strip's rails join opposite side labels, not alike ones: got {strip:?}",
    );
    mesh.validate_topology();
}

/// §7.8.3's **absolute winding** on the tied-vote twist, at every `segments`. The
/// twist is the one configuration where neither outer rail belongs to a rebuilt
/// flank (the crack), so §5.1's "wind against the face across the shared edge" has
/// no input and the emitted cycle must come from the rule itself: every quad's two
/// same-endpoint sides run `N`-ward to `P`-ward (reading (1)).
///
/// On this fixture that reads off as `+x` at both ends — at `a = (1, 0)` the slid
/// vert `(0.8, 0)` is `N(a)` and the collapsed `(1, 0)` is `P(a)`; at `b = (1, 1)`
/// the collapsed `(1, 1)` is `N(b)` and the slid `(1.2, 1)` is `P(b)` — which is why
/// the labelling can be pinned through positions alone. Nothing weaker sees a
/// reversal: the strip is edge-disjoint from everything, so it stays internally
/// consistent when reversed wholesale, and any comparison blind to winding calls
/// the reversed strip equal.
///
/// `segments = 1` and `segments ≥ 2` are emitted by two different builders; the rule
/// is one rule, so both are swept here.
#[test]
fn edge_bevel_tied_vote_twist_quads_wind_n_to_p_at_both_ends() {
    for segments in 1..=4 {
        let (mut mesh, edge) = two_quad_plane(false);
        bevel_one_edge(&mut mesh, edge, segments);
        assert_plane_strip_winds_n_to_p(&mesh, segments);
        mesh.validate_topology();
    }
}

/// §7.8.3 reading (1) on the two-quad plane's strip, whichever builder emitted it:
/// every strip quad's two same-endpoint sides — the two joining rung ends at equal
/// `y` — must be walked `N`-ward to `P`-ward, which on this fixture is `+x`.
///
/// Only the planar faces are read, so the same check serves a mesh that also carries
/// a disjoint junction body: the plane sits entirely in `z = 0` and nothing the cube
/// contributes reaches it.
fn assert_plane_strip_winds_n_to_p(mesh: &LoopMesh, segments: u32) {
    let flanks = [
        face_through(mesh, Vec3::ZERO),
        face_through(mesh, Vec3::new(2.0, 0.0, 0.0)),
    ];
    let mut same_endpoint_sides = 0;
    for face in mesh.face_keys() {
        if flanks.contains(&face) {
            continue;
        }
        let cycle: Vec<Vec3> = mesh
            .face_vert_keys(face)
            .into_iter()
            .map(|v| mesh.vertex_position(v))
            .collect();
        if cycle.iter().any(|position| position.z.abs() > 1.0e-4) {
            continue;
        }
        for index in 0..cycle.len() {
            let from = cycle[index];
            let to = cycle[(index + 1) % cycle.len()];
            if (from.y - to.y).abs() > 1.0e-4 {
                continue;
            }
            same_endpoint_sides += 1;
            assert!(
                to.x > from.x + 1.0e-4,
                "at segments = {segments} a strip quad walks {from:?} → {to:?}, \
                 which runs P-ward to N-ward — the strip is reversed",
            );
        }
    }
    assert_eq!(
        same_endpoint_sides,
        2 * segments as usize,
        "every one of the {segments} strip quads contributes one side per endpoint",
    );
}

/// §7.8.3's endpoint-symmetry lemma, on the winding seed itself: relabelling `a ↔ b`
/// leaves every quad's directed cycle unchanged, so either end may anchor the
/// numbering. That is what licenses reading the seed off `b` when `a` carries no
/// label — the case a junction end at `a` puts the strip in — and it is only sound
/// if the two derivations agree wherever both labels exist, which is what this pins.
///
/// The pairing route sets `swap` from the labels themselves when both are present
/// (agreeing labels are §7.8.3's un-mirrored, tied-vote regime and need the `q` chain
/// reversed), so the `b`-anchored derivation is fed exactly the `swap` the real
/// caller would have.
///
/// This is the **only** pin on the two unlabelled-`a` arms, and deliberately a
/// function-level one: no mesh-level assertion can reach them. An unlabelled end
/// sends `build_multisegment_strip` down the rail-existence route, which reports
/// `anchored` on every mesh the public API can build — and while anchored, flipping
/// the seed flips the direction `strip_runs_with_restitched_face` reads its radial-1
/// rail in, so `reversed` flips back and the emitted cycle is byte-identical. The
/// seed only reaches the output where that repair has no anchor, which for an
/// unlabelled end means both of E's originals lost their rebuild — a state the
/// caller's own `debug_assert!` calls broken. So the arms are pinned here or nowhere.
#[test]
fn strip_winding_seed_is_the_same_from_either_endpoints_label() {
    for label_a in [SideLabel::PrevFirst, SideLabel::NextFirst] {
        for label_b in [SideLabel::PrevFirst, SideLabel::NextFirst] {
            let swap = label_a == label_b;
            let from_a = strip_winding_seed(label_a, label_b, swap);
            let from_b = strip_winding_seed(SideLabel::Unlabelled, label_b, swap);
            assert_eq!(
                from_a, from_b,
                "an unlabelled end must seed the same cycle its labelled partner does",
            );
            assert_eq!(
                from_a,
                Some(label_a == SideLabel::PrevFirst),
                "the anchor arc runs P → N exactly when its own label says so",
            );
        }
    }
    assert_eq!(
        strip_winding_seed(SideLabel::Unlabelled, SideLabel::Unlabelled, false),
        None,
        "two unlabelled ends leave the seed to the rail repair",
    );
}

/// §7.8.4 / §7.8.3's index rule at `segments = 2`: the rungs open into two profile
/// arcs and step `k` forward from `P(a)` pairs with step `segments − k` forward from
/// `P(b)`. On the twisted fixture the arcs run `0.8, 0.9, 1.0` at one end and
/// `1.0, 1.1, 1.2` at the other, so the cross pairing puts the *outermost* sample of
/// one arc in the same quad as the outermost of the other — a like-label pairing
/// would pair `0.8` with `1.0` instead.
#[test]
fn edge_bevel_tied_vote_twist_pairs_profile_samples_in_reverse() {
    let (mut mesh, edge) = two_quad_plane(false);
    bevel_one_edge(&mut mesh, edge, 2);

    let plane = |x: f32, y: f32| Vec3::new(x, y, 0.0);
    let holds = |face: FaceKey, position: Vec3| {
        mesh.face_vert_keys(face)
            .into_iter()
            .any(|v| (mesh.vertex_position(v) - position).length() < 1.0e-4)
    };
    let quad_with = |position: Vec3| {
        mesh.face_keys()
            .into_iter()
            .filter(|&f| f != face_through(&mesh, Vec3::ZERO))
            .filter(|&f| f != face_through(&mesh, plane(2.0, 0.0)))
            .find(|&f| holds(f, position))
            .unwrap_or_else(|| panic!("a strip quad reaches {position}"))
    };
    assert!(
        holds(quad_with(plane(0.8, 0.0)), plane(1.2, 1.0)),
        "sample 0 of one arc pairs with sample `segments` of the other",
    );
    assert!(
        holds(quad_with(plane(1.0, 0.0)), plane(1.0, 1.0)),
        "the two collapsed arc ends pair with each other",
    );
    assert_eq!(
        strip_flank_shared_edges(&mesh),
        0,
        "the crack survives the profile: both outer rails are cross-labelled",
    );
    mesh.validate_topology();
}

/// The two-quad plane plus a **disjoint** cube whose corner `(1, -1, -1)` carries two
/// beveled edges. Beveling all three in one call makes the run junction-bearing, which
/// is what routes the plane's isolated edge through the junction strip builder rather
/// than the isolated-edge path.
fn plane_plus_cube_junction(contiguous: bool) -> (LoopMesh, EdgeKey, [EdgeKey; 2]) {
    let (mut mesh, isolated) = two_quad_plane(contiguous);
    mesh_add_cube(&mut mesh);
    let corner = Vec3::new(1.0, -1.0, -1.0);
    let junction = [
        cube_edge(&mesh, corner, Vec3::new(-1.0, -1.0, -1.0)),
        cube_edge(&mesh, corner, Vec3::new(1.0, 1.0, -1.0)),
    ];
    (mesh, isolated, junction)
}

fn bevel_edges_at_segments(mesh: &mut LoopMesh, edges: &[EdgeKey], segments: u32) {
    let selection: Vec<GeomKey> = edges.iter().map(|&e| e.into()).collect();
    mesh.bevel(
        &selection,
        BevelOptions {
            offset: 0.2,
            affect: BevelAffect::Edges,
            segments,
            ..Default::default()
        },
    );
}

fn bevel_count_delta(mesh: &mut LoopMesh, edges: &[EdgeKey]) -> (i64, i64, i64) {
    let before = counts(mesh);
    bevel_edges_at_segments(mesh, edges, 1);
    let after = counts(mesh);
    (
        after.0 as i64 - before.0 as i64,
        after.1 as i64 - before.1 as i64,
        after.2 as i64 - before.2 as i64,
    )
}

/// §7.8.3 on the **junction** route. The rail pairing is the same rule in every
/// regime, so which builder a beveled edge happens to reach must not change it.
/// Adding a junction-bearing pair elsewhere in the same call sends the tied-vote
/// mis-wound isolated edge — both endpoints terminal — through the junction strip
/// builder; its own contribution must stay the twisted `+2 V / +5 E / +1 F`, not the
/// `+3 E` straight join that keying each rail to a flank *face* would give.
#[test]
fn junction_path_strip_rails_pair_by_side_label_not_flank_face() {
    let isolated_alone = {
        let (mut mesh, edge) = two_quad_plane(false);
        bevel_count_delta(&mut mesh, &[edge])
    };
    let junction_only = {
        let (mut mesh, _, junction) = plane_plus_cube_junction(false);
        bevel_count_delta(&mut mesh, &junction)
    };
    let (mut mesh, isolated, junction) = plane_plus_cube_junction(false);
    let together = bevel_count_delta(&mut mesh, &[isolated, junction[0], junction[1]]);

    assert_eq!(
        (
            together.0 - junction_only.0,
            together.1 - junction_only.1,
            together.2 - junction_only.2,
        ),
        isolated_alone,
        "the isolated edge's contribution must not depend on which strip builder \
         the run's junctions routed it to",
    );
    assert_eq!(
        strip_flank_shared_edges(&mesh),
        0,
        "the §7.8.3 crack survives the junction route: the twisted strip shares no \
         edge with either rebuilt flank",
    );
    assert_plane_strip_winds_n_to_p(&mesh, 1);
    mesh.validate_topology();

    // The absolute-winding rule is one rule at every `segments` (§7.8.3), and the
    // crack leaves the multi-rung strip nothing to be repaired against, so the
    // junction route has to spell the rule's own cycle from the seed alone here too.
    for segments in 2..=4 {
        let (mut mesh, isolated, junction) = plane_plus_cube_junction(false);
        bevel_edges_at_segments(&mut mesh, &[isolated, junction[0], junction[1]], segments);
        assert_eq!(
            strip_flank_shared_edges(&mesh),
            0,
            "the crack survives the profile on the junction route as well",
        );
        assert_plane_strip_winds_n_to_p(&mesh, segments);
        mesh.validate_topology();
    }
}

/// The mirrored-label control for the test above: on a contiguous pair the
/// cross-labelled rails *are* the rebuilt flanks' own pairs, so the junction route
/// must reproduce the §7.1 quad — one shared edge per flank — exactly as before.
#[test]
fn junction_path_strip_rails_are_the_flanks_own_pairs_when_labels_are_mirrored() {
    let isolated_alone = {
        let (mut mesh, edge) = two_quad_plane(true);
        bevel_count_delta(&mut mesh, &[edge])
    };
    let junction_only = {
        let (mut mesh, _, junction) = plane_plus_cube_junction(true);
        bevel_count_delta(&mut mesh, &junction)
    };
    let (mut mesh, isolated, junction) = plane_plus_cube_junction(true);
    let together = bevel_count_delta(&mut mesh, &[isolated, junction[0], junction[1]]);

    assert_eq!(
        (
            together.0 - junction_only.0,
            together.1 - junction_only.1,
            together.2 - junction_only.2,
        ),
        isolated_alone,
    );
    assert_eq!(
        strip_flank_shared_edges(&mesh),
        2,
        "each rebuilt flank still shares exactly one rail with the strip",
    );
    mesh.validate_topology();
}

// ---- BEVEL.md §7.5.1 "Three regimes, selected by the direction angle" ----
