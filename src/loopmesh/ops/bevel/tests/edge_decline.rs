use super::*;

/// The `5 × 5` grid with `(2,1)` slid onto `(2,2)`, so the `−y` **leg** of the `(2,2)`
/// terminal is zero-length while the topology stays intact. A leg with no direction has
/// no near-corner vert to slide onto, so the close at `(2,2)` declines — permanently:
/// it is a degenerate input (rubric 5), not a shape the operator grows into.
fn quad_grid_5x5_collapsed_leg() -> (LoopMesh, Vec<Vec<VertexKey>>) {
    let (mut mesh, verts) = quad_grid_5x5();
    mesh.set_vertex_position(verts[2][1], Vec3::new(2.0, 2.0, 0.0));
    (mesh, verts)
}

/// BEVEL.md §7.5 placement-before-surgery: a terminal close is all-or-nothing, so the
/// close is **probed for both endpoints before the first mint**. Beveling this edge
/// closes fine at its `(3,2)` endpoint and declines at `(2,2)`, whose `−y` leg is
/// collapsed and so has no direction to slide a near-corner vert along — and a decline
/// taken after `(3,2)` had been placed would leave that endpoint's strip and leg verts
/// minted, attached to nothing, on a mesh the pass otherwise leaves pristine. Driven
/// straight at the isolated-edge builder, past the up-front buildable-edge filter, so it
/// is the builder's own probe under test.
#[test]
fn edge_bevel_terminal_decline_mints_nothing_at_the_other_endpoint() {
    let (mut mesh, verts) = quad_grid_5x5_collapsed_leg();
    let before = counts(&mesh);
    let rim = verts[2][2];
    let interior = verts[3][2];
    let edge = mesh
        .query_edge_exists(rim, interior)
        .expect("the +x edge off the collapsed leg");
    assert!(
        !mesh.terminal_close_admits(rim, edge),
        "the collapsed leg's endpoint declines",
    );
    assert!(
        mesh.terminal_close_admits(interior, edge),
        "the other endpoint's close is admitted — it is the collapsed one that declines",
    );

    let mut templates = TemplateBank::default();
    let strip_template = templates.register(mesh.edge_flanking_face(edge));
    let fan_templates = mesh
        .edge_vert_keys(edge)
        .map(|endpoint| templates.register(mesh.vert_face_keys(endpoint).first().copied()));
    templates.seal(&mesh);

    let mut result = BevelResult::default();
    let mut rings = BevelRings::default();
    mesh.bevel_single_edge(
        edge,
        0.25,
        0.5,
        false,
        -1,
        &templates,
        strip_template,
        fan_templates,
        &mut rings,
        &mut result,
    );
    mesh.validate_topology();

    assert!(
        result.verts.is_empty() && result.faces.is_empty(),
        "the declined edge minted nothing — no stranded verts",
    );
    assert_eq!(counts(&mesh), before, "and the mesh is untouched");
    for vertex in mesh.vert_keys() {
        assert!(
            !mesh.vert_face_keys(vertex).is_empty(),
            "no vertex is left face-less",
        );
    }
}

/// The same all-or-nothing rule at `segments ≥ 2` (BEVEL.md §7.5): the multi-segment
/// orchestrator plans every terminal of the edge set **before the first mint**, exactly
/// as the single-segment one does. This edge closes fine at `(3,2)` and cannot close at
/// `(2,2)`, whose `−y` leg is collapsed — so the **whole edge** is declined. Without the
/// plan the pass would mint `(3,2)`'s chain, strand it when `(2,2)` declined, and record
/// a fill over the leg that never got a vert.
#[test]
fn edge_bevel_multi_segment_terminal_decline_is_whole_edge_and_up_front() {
    let (mut mesh, verts) = quad_grid_5x5_collapsed_leg();
    let before = counts(&mesh);
    let declined = mesh
        .query_edge_exists(verts[2][2], verts[3][2])
        .expect("the +x edge off the collapsed leg");

    let result = mesh.bevel(
        &[declined.into()],
        BevelOptions {
            offset: 0.25,
            segments: 3,
            affect: BevelAffect::Edges,
            ..Default::default()
        },
    );
    mesh.validate_topology();

    assert_eq!(
        counts(&mesh),
        before,
        "nothing was minted, nothing stitched"
    );
    assert!(result.verts.is_empty() && result.faces.is_empty());
    for vertex in mesh.vert_keys() {
        assert!(
            !mesh.vert_face_keys(vertex).is_empty(),
            "no vertex is left face-less",
        );
    }
}

/// And the `segments ≥ 2` orchestrator's plan does **not** over-decline: the open-fan
/// valence-4 terminal it has always built (its ring order comes off the same rotational
/// layout the `segments = 1` close now shares) still builds, with the fill fanned from
/// its one leg vert into `segments` triangles (§8.3.0.2's three-boundvert rule).
#[test]
fn edge_bevel_multi_segment_boundary_valence_four_terminal_open_fan_still_builds() {
    let (mut mesh, verts) = quad_grid_5x5_holed();
    let before = counts(&mesh);
    let beveled = mesh
        .query_edge_exists(verts[2][2], verts[3][2])
        .expect("the +x edge off the hole rim");

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

    assert!(counts(&mesh).2 > before.2, "the open fan is not declined");
    let leg = vert_near(&mesh, Vec3::new(1.75, 2.0, 0.0), 1.0e-5).expect("−x leg vert");
    let fills = terminal_fill_faces(&mesh, &result.verts, leg);
    assert_eq!(
        fills.len(),
        2,
        "three boundverts ⇒ `segments` fan triangles"
    );
    assert!(fills.iter().all(|fill| fill.len() == 3));
    assert!(winding_conflicts(&mesh).is_empty());
}

/// The `5 × 5` grid with `(2,2)` slid onto `(3,2)`, so the edge between them is
/// **zero-length** while the topology stays intact.
fn quad_grid_5x5_collapsed_edge() -> (LoopMesh, Vec<Vec<VertexKey>>) {
    let (mut mesh, verts) = quad_grid_5x5();
    mesh.set_vertex_position(verts[2][2], Vec3::new(3.0, 2.0, 0.0));
    (mesh, verts)
}

/// Rubric 5: a zero-length beveled edge has no offset line, so its terminal close
/// cannot place a thing. That is a degenerate **input**, not a broken invariant — the
/// operator declines it (whole, before the first mint, so its junction partner declines
/// with it) instead of half-placing a corner the strip build then panics on for want of
/// a near-corner vert.
#[test]
fn edge_bevel_declines_a_zero_length_edge_at_a_junction() {
    let (mut mesh, verts) = quad_grid_5x5_collapsed_edge();
    let before = counts(&mesh);
    let degenerate = mesh
        .query_edge_exists(verts[2][2], verts[3][2])
        .expect("the collapsed edge");
    let neighbour = mesh
        .query_edge_exists(verts[3][2], verts[4][2])
        .expect("its junction partner");

    let result = mesh.bevel(
        &[degenerate.into(), neighbour.into()],
        BevelOptions {
            offset: 0.25,
            affect: BevelAffect::Edges,
            ..Default::default()
        },
    );
    mesh.validate_topology();

    assert_eq!(counts(&mesh), before);
    assert!(result.verts.is_empty() && result.faces.is_empty());
}

/// BEVEL.md §7.5 / §8.3.0.2, the all-or-nothing rule: the endpoint fill's boundary is
/// exactly the run of verts the close placed on the endpoint's legs, so a leg the close
/// **cannot** place a vert on (here a zero-length one) leaves the fill polygon shorter
/// than the hole it must cover — it would lay a face over geometry the surviving faces
/// still hold. The endpoint is declined entire (no leg vert, no fill, no substitution),
/// and with it the edge.
#[test]
fn edge_bevel_declines_a_terminal_with_a_degenerate_leg() {
    let (mut mesh, verts) = quad_grid_5x5_collapsed_edge();
    let before = counts(&mesh);
    // `E` itself is fine; its `(3,2)` terminal's one **non-flanking** leg is the
    // collapsed edge, so that endpoint's fill has a leg it cannot place.
    let beveled = mesh
        .query_edge_exists(verts[3][2], verts[4][2])
        .expect("the sound edge next to the collapsed one");

    let result = mesh.bevel(
        &[beveled.into()],
        BevelOptions {
            offset: 0.25,
            affect: BevelAffect::Edges,
            ..Default::default()
        },
    );
    mesh.validate_topology();

    assert_eq!(counts(&mesh), before, "no half-built fill");
    assert!(result.verts.is_empty() && result.faces.is_empty());
}

// ---- Mechanism (1) on the lone (non-junction) beveled edge and its legs ----
//
// Cube corner `v0 = (1,-1,-1)`; the beveled edge `E = {v0, v1}` runs along −X at
// `y = z = −1`, flanked by the front (`z = −1`) and bottom (`y = −1`) faces. `v0`'s
// two unbeveled legs are `{v0, v3}` (to `(1,1,-1)`, in the front face) and
// `{v0, v4}` (to `(1,-1,1)`, in the bottom face); the surviving third face at `v0`
// is the `x = 1` side, which absorbs the terminal (BEVEL.md §7.5 / §8.3.0.1).
