use super::*;

/// BEVEL.md §2.1: a non-positive offset is a no-op.
#[test]
fn non_positive_offset_is_noop() {
    let mut mesh = LoopMesh::new();
    mesh_add_cube(&mut mesh);
    let before = counts(&mesh);
    let corner = mesh.vert_keys()[0];

    let result = mesh.bevel(
        &[corner.into()],
        BevelOptions {
            offset: 0.0,
            ..Default::default()
        },
    );

    assert_eq!(counts(&mesh), before, "offset <= 0 changes nothing");
    assert!(result.verts.is_empty() && result.edges.is_empty() && result.faces.is_empty());
    mesh.validate_topology();
}

/// BEVEL.md §7.4: beveling one isolated manifold cube edge ({0,1}) with
/// `affect = Edges` replaces the crease with four strip verts and one strip
/// quad, each valence-3 endpoint absorbed into its third face. Count delta
/// +2 V / +3 E / +1 F over the cube's 8 V / 12 E / 6 F.
#[test]
fn edge_bevel_isolated_cube_edge_count_delta() {
    let mut mesh = LoopMesh::new();
    mesh_add_cube(&mut mesh);
    assert_eq!(counts(&mesh), (8, 12, 6));

    let a = vert_at(&mesh, Vec3::new(1.0, -1.0, -1.0));
    let b = vert_at(&mesh, Vec3::new(-1.0, -1.0, -1.0));
    let edge = mesh.query_edge_exists(a, b).expect("edge {0,1}");

    let result = mesh.bevel(
        &[edge.into()],
        BevelOptions {
            offset: 0.25,
            affect: BevelAffect::Edges,
            ..Default::default()
        },
    );

    assert_eq!(
        counts(&mesh),
        (10, 15, 7),
        "isolated edge +2 V / +3 E / +1 F"
    );
    assert_eq!(result.verts.len(), 4, "four strip verts");
    assert_eq!(result.faces.len(), 1, "the strip quad is the only new face");

    // The strip face is a quad.
    let strip = result.faces[0];
    assert_eq!(mesh.face_vert_keys(strip).len(), 4, "strip quad");

    // Both original endpoints are removed.
    assert!(
        !mesh.vert_keys().into_iter().any(|v| v == a || v == b),
        "both beveled-edge endpoints removed",
    );
    mesh.validate_topology();
}

/// BEVEL.md §7.4: the strip quad's four corners sit a perpendicular `offset`
/// from the original edge line — two per endpoint, one into each adjacent
/// face. For edge {0,1} (the front/bottom crease along −Y/−Z) the strip verts
/// land in the front (z = −1) and bottom (y = −1) planes, `offset` off the
/// crease.
#[test]
fn edge_bevel_strip_verts_perpendicular_offset() {
    let mut mesh = LoopMesh::new();
    mesh_add_cube(&mut mesh);
    let edge = cube_edge(
        &mesh,
        Vec3::new(1.0, -1.0, -1.0),
        Vec3::new(-1.0, -1.0, -1.0),
    );

    let offset = 0.25_f32;
    let result = mesh.bevel(
        &[edge.into()],
        BevelOptions {
            offset,
            affect: BevelAffect::Edges,
            ..Default::default()
        },
    );

    // The crease line is x ∈ [−1,1], y = −1, z = −1. Each strip vert keeps the
    // endpoint's x, and slides `offset` along exactly one of −Y/−Z into a face:
    // the front face (z = −1) strip verts move in +Y (to y = −1+offset, z = −1),
    // the bottom face (y = −1) strip verts move in +Z (to y = −1, z = −1+offset).
    for &v in &result.verts {
        let p = mesh.vertex_position(v);
        assert!(p.x.abs() <= 1.0 + 1.0e-5, "strip vert keeps endpoint x");
        let into_front = (p - Vec3::new(p.x, -1.0 + offset, -1.0)).length();
        let into_bottom = (p - Vec3::new(p.x, -1.0, -1.0 + offset)).length();
        assert!(
            into_front < 1.0e-5 || into_bottom < 1.0e-5,
            "strip vert at {p:?} must lie {offset} into the front or bottom face",
        );
    }
    mesh.validate_topology();
}

/// BEVEL.md §7.4: after the bevel, the two `E`-adjacent faces (fL/fR) stay
/// quads (swapping their `E` side for the strip's long edge) and the two
/// third faces grow by one corner (quad → pentagon). The strip quad shares
/// each of its four edges with exactly one other face (radial count 2).
#[test]
fn edge_bevel_face_sizes_and_strip_manifold() {
    let mut mesh = LoopMesh::new();
    mesh_add_cube(&mut mesh);
    let edge = cube_edge(
        &mesh,
        Vec3::new(1.0, -1.0, -1.0),
        Vec3::new(-1.0, -1.0, -1.0),
    );

    let result = mesh.bevel(
        &[edge.into()],
        BevelOptions {
            offset: 0.25,
            affect: BevelAffect::Edges,
            ..Default::default()
        },
    );

    // Two pentagons (the third faces, originally quads, each +1 corner); the
    // remaining four pre-existing faces stay quads, plus the new strip quad.
    let sizes: Vec<usize> = mesh
        .face_keys()
        .into_iter()
        .map(|f| mesh.face_vert_keys(f).len())
        .collect();
    let pentagons = sizes.iter().filter(|&&n| n == 5).count();
    let quads = sizes.iter().filter(|&&n| n == 4).count();
    assert_eq!(pentagons, 2, "two third faces grew quad → pentagon");
    assert_eq!(quads, 5, "fL, fR, two untouched faces, and the strip quad");

    // Every strip-quad edge is shared with exactly one other face.
    let strip = result.faces[0];
    for loop_key in mesh.face_loop_keys(strip) {
        let strip_edge = mesh.loop_edge(loop_key);
        assert_eq!(
            mesh.edge_radial_iter(strip_edge).count(),
            2,
            "each strip edge is manifold (radial count 2)",
        );
    }
    mesh.validate_topology();
}

/// BEVEL.md §9: the §7 placement is not specific to one edge's orientation —
/// the differently-oriented edge {3,7} (the front∩... rim along +Z) yields the
/// same +2 V / +3 E / +1 F delta and strip-quad topology.
#[test]
fn edge_bevel_other_orientation_same_delta() {
    let mut mesh = LoopMesh::new();
    mesh_add_cube(&mut mesh);
    // vert 3 = (1,1,-1), vert 7 = (1,1,1).
    let edge = cube_edge(&mesh, Vec3::new(1.0, 1.0, -1.0), Vec3::new(1.0, 1.0, 1.0));

    let result = mesh.bevel(
        &[edge.into()],
        BevelOptions {
            offset: 0.25,
            affect: BevelAffect::Edges,
            ..Default::default()
        },
    );

    assert_eq!(counts(&mesh), (10, 15, 7), "same +2 V / +3 E / +1 F delta");
    assert_eq!(result.verts.len(), 4, "four strip verts");
    assert_eq!(result.faces.len(), 1, "single strip quad");
    assert_eq!(mesh.face_vert_keys(result.faces[0]).len(), 4, "strip quad");
    mesh.validate_topology();
}

/// BEVEL.md §7.5.1: at a **non-planar** valence-3 terminal each strip vert is
/// the leg meet — `E`'s offset line (offset 0.3) against the flanking unbeveled
/// leg (offset 0), built from the edge-cross local normal — **not** the §7.3
/// perpendicular-in-face slide. Two consequences the slide cannot produce and
/// this test pins: (1) every strip vert sits an exact perpendicular `offset`
/// from `E`'s original line (the offset-line property), and (2) the strip vert
/// on the `v0` corner lands at the leg-meet target, measurably off the §7.3
/// face-plane slide (~0.05 mesh unit here at offset 0.3 on the oblique legs).
#[test]
fn edge_bevel_terminal_leg_meet_on_nonplanar_corner() {
    let mut mesh = LoopMesh::new();
    let [v0, v1, _v2, _v3] = irregular_tetra(&mut mesh);
    let endpoint_a = mesh.vertex_position(v0);
    let endpoint_b = mesh.vertex_position(v1);
    let edge = mesh.query_edge_exists(v0, v1).expect("tetra edge {0,1}");
    let edge_dir = (endpoint_b - endpoint_a).normalize();

    let offset = 0.3_f32;
    let result = mesh.bevel(
        &[edge.into()],
        BevelOptions {
            offset,
            affect: BevelAffect::Edges,
            ..Default::default()
        },
    );
    assert_eq!(result.verts.len(), 4, "four terminal strip verts");

    // (1) Offset-line property: every strip vert lies an exact perpendicular
    // `offset` from `E`'s original line. The §7.3 face-plane slide also satisfies
    // this, so it is necessary-not-sufficient — assertion (2) separates them.
    for &strip in &result.verts {
        let to_strip = mesh.vertex_position(strip) - endpoint_a;
        let along = to_strip.dot(edge_dir);
        let perpendicular_distance = (to_strip - edge_dir * along).length();
        assert!(
            (perpendicular_distance - offset).abs() < 1.0e-4,
            "strip vert {perpendicular_distance} must lie offset {offset} from E's line",
        );
    }

    // (2) The §7.5.1 leg-meet targets at the `v0` corner, derived by hand from
    // the edge-cross local normal + §7.5.1's skew resolution (see the scratch
    // prototype): these are measurably off the §7.3 in-face slide, which would
    // keep `x = 0` (the slide stays in each face's plane through `v0`).
    let leg_meet_one = Vec3::new(0.05957, 0.29786, 0.03574);
    let leg_meet_two = Vec3::new(0.05262, 0.07894, 0.28943);
    for &target in &[leg_meet_one, leg_meet_two] {
        let hits = result
            .verts
            .iter()
            .filter(|&&v| (mesh.vertex_position(v) - target).length() < 1.0e-3)
            .count();
        assert_eq!(
            hits, 1,
            "a strip vert lands at the §7.5.1 leg meet {target:?}"
        );
        // The §7.3 slide keeps x = 0 at this corner; the leg meet pulls it off.
        assert!(target.x > 1.0e-2, "leg meet is off the §7.3 in-face slide");
    }
    mesh.validate_topology();
}

/// Two quads sharing the edge `a`–`b`, where **both** sectors have a near-flat corner
/// at `b`: each face's other edge at `b` leaves within `1.7°` of straight-through, so
/// both of `b`'s terminal strip verts dispatch to §7.5.1 regime 1's near-collinear
/// half. The `±0.03` splay is what makes the corners flat-but-not-exact; the patch is
/// planar in `z = 0`. `a`'s own corners stay wide (±45°), so `a` is an ordinary
/// regime-3 terminal and the two ends of `E` can be read independently.
///
/// `mis_wind_lower` reverses the lower quad's ring so both faces traverse `a`→`b`,
/// which is §7.8's inconsistent winding: the lower sector's own plane normal then
/// points to the opposite side of the surface from the upper's.
fn near_collinear_terminal_patch(mesh: &mut LoopMesh, mis_wind_lower: bool) -> [VertexKey; 2] {
    let a = mesh.create_vertex(Vec3::new(0.0, 0.0, 0.0));
    let b = mesh.create_vertex(Vec3::new(1.0, 0.0, 0.0));
    let upper_far = mesh.create_vertex(Vec3::new(2.0, 0.03, 0.0));
    let lower_far = mesh.create_vertex(Vec3::new(2.0, -0.03, 0.0));
    let upper = mesh.create_vertex(Vec3::new(1.0, 1.0, 0.0));
    let lower = mesh.create_vertex(Vec3::new(1.0, -1.0, 0.0));
    let lower_ring = if mis_wind_lower {
        [b, lower_far, lower, a]
    } else {
        [a, lower, lower_far, b]
    };
    for ring in [[a, b, upper_far, upper], lower_ring] {
        let edges = [
            mesh.create_edge([ring[0], ring[1]], EdgeCreateFlags::NO_DUPLICATE),
            mesh.create_edge([ring[1], ring[2]], EdgeCreateFlags::NO_DUPLICATE),
            mesh.create_edge([ring[2], ring[3]], EdgeCreateFlags::NO_DUPLICATE),
            mesh.create_edge([ring[3], ring[0]], EdgeCreateFlags::NO_DUPLICATE),
        ];
        mesh.face_create(&ring, &edges, FaceCreateFlags::NONE);
    }
    [a, b]
}

fn strip_verts_near(mesh: &LoopMesh, result: &BevelResult, target: Vec3) -> Vec<Vec3> {
    result
        .verts
        .iter()
        .map(|&vert| mesh.vertex_position(vert))
        .filter(|position| (*position - target).length() < 0.5)
        .collect()
}

/// BEVEL.md §7.5.1 regime 1, end to end: a near-collinear terminal's two strip verts
/// are each pushed `offset / cos(θ_dir/2)` square to the common direction, to
/// **opposite** sides of `E`. Nothing here trips the §7.5.2 clamp — on a consistently
/// wound surface the along-leg projection comes out positive for both sectors, which
/// is §7.8.2's "Panel B is the mirror of Panel A" restated for regime 1.
#[test]
fn edge_bevel_near_collinear_terminal_pushes_out_the_common_perpendicular() {
    let mut mesh = LoopMesh::new();
    let [a, b] = near_collinear_terminal_patch(&mut mesh, false);
    let endpoint = mesh.vertex_position(b);
    let edge = mesh.query_edge_exists(a, b).expect("shared edge");

    let offset = 0.25_f32;
    let result = mesh.bevel(
        &[edge.into()],
        BevelOptions {
            offset,
            affect: BevelAffect::Edges,
            ..Default::default()
        },
    );

    let at_b = strip_verts_near(&mesh, &result, endpoint);
    assert_eq!(at_b.len(), 2, "both of `b`'s strip verts, got {at_b:?}");
    for position in &at_b {
        let push = *position - endpoint;
        // Regime 1's magnitude `offset / cos(θ_dir/2)`, which at 1.7° is `offset` to
        // well inside this tolerance.
        assert!(
            (push.length() - offset).abs() < 1.0e-3,
            "strip vert must stand off `b` by ≈ {offset}, got {} at {position:?}",
            push.length(),
        );
        // Square to the common direction (≈ `E`'s own line here), not slid along a leg.
        assert!(
            push.y.abs() > 0.99 * offset,
            "the push must be the common perpendicular, got {push:?}",
        );
    }
    assert!(
        at_b[0].y * at_b[1].y < 0.0,
        "the two sectors push to opposite sides of `E`, got {at_b:?}",
    );
    mesh.validate_topology();
}

/// BEVEL.md §7.5.1: the regime-1 placement is returned **before** the §7.5.2 span
/// clamp — the pseudocode's `return` sits ahead of the clamp call — so no regime-1
/// strip vert is ever snapped onto its endpoint.
///
/// A consistently wound terminal cannot see the difference (the projection is
/// positive there either way), so the discriminating fixture is a **mis-wound** one:
/// the flipped sector normal throws regime 1's push into the other flank's half-plane
/// and its along-leg projection negative, which is precisely §7.8.2's collapse
/// trigger. Under regime 1 the collapse must not happen: both of `b`'s strip verts
/// still stand `≈ offset` off `b`, merely on the *same* side of `E` instead of
/// straddling it. The clamp itself is untouched — `a`'s wide corners are regime 3, and
/// its disagreeing flank still collapses onto `a` exactly as §7.8.2 describes.
#[test]
fn edge_bevel_near_collinear_terminal_skips_the_span_clamp_when_mis_wound() {
    let mut mesh = LoopMesh::new();
    let [a, b] = near_collinear_terminal_patch(&mut mesh, true);
    let far_endpoint = mesh.vertex_position(a);
    let endpoint = mesh.vertex_position(b);
    let edge = mesh.query_edge_exists(a, b).expect("shared edge");

    let offset = 0.25_f32;
    let result = mesh.bevel(
        &[edge.into()],
        BevelOptions {
            offset,
            affect: BevelAffect::Edges,
            ..Default::default()
        },
    );

    let at_b = strip_verts_near(&mesh, &result, endpoint);
    assert_eq!(at_b.len(), 2, "both of `b`'s strip verts, got {at_b:?}");
    for position in &at_b {
        let push = *position - endpoint;
        assert!(
            (push.length() - offset).abs() < 1.0e-3,
            "a regime-1 strip vert must never be clamped onto `b`; got {} at \
             {position:?}",
            push.length(),
        );
    }
    assert!(
        at_b[0].y * at_b[1].y > 0.0,
        "the mis-wound sector pushes into the other flank's half-plane, so both land \
         on one side of `E`, got {at_b:?}",
    );

    // The clamp is still armed everywhere §7.8.2 says it is: `a`'s corners are wide,
    // so its disagreeing flank takes the regime-3 line solve and collapses onto `a`.
    let at_a = strip_verts_near(&mesh, &result, far_endpoint);
    assert!(
        at_a.iter()
            .any(|position| (*position - far_endpoint).length() < 1.0e-6),
        "`a`'s disagreeing flank must still collapse onto `a`, got {at_a:?}",
    );
    mesh.validate_topology();
}

/// BEVEL.md §7.6.3 (`s = 2`, through junction): beveling the two cube edges
/// {0,1} and {0,3} that share vertex 0 makes vertex 0 a two-edge junction. The
/// two arriving strips share a pair of boundverts (§7.6.1) and run into each
/// other with **no** patch face (§7.6.3). Far endpoints 1 and 3 are valence-3
/// terminals. Count delta over the cube (8 V / 12 E / 6 F): +3 V / +5 E / +2 F
/// → 11 V / 17 E / 8 F.
#[test]
fn edge_bevel_two_edge_junction_no_patch() {
    let mut mesh = LoopMesh::new();
    mesh_add_cube(&mut mesh);
    assert_eq!(counts(&mesh), (8, 12, 6));

    let e01 = cube_edge(
        &mesh,
        Vec3::new(1.0, -1.0, -1.0),
        Vec3::new(-1.0, -1.0, -1.0),
    );
    let e03 = cube_edge(&mesh, Vec3::new(1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, -1.0));

    let result = mesh.bevel(
        &[e01.into(), e03.into()],
        BevelOptions {
            offset: 0.25,
            affect: BevelAffect::Edges,
            ..Default::default()
        },
    );

    assert_eq!(
        counts(&mesh),
        (11, 17, 8),
        "s=2 through-junction +3 V / +5 E / +2 F",
    );
    // Two strip quads, no patch face.
    assert_eq!(result.faces.len(), 2, "two strip faces, no s=2 patch");

    // The Front-side sector is a single planar face: its boundvert is the exact
    // 2-line meet at (1-offset, -1+offset, -1) = (0.75, -0.75, -1).
    let front_meet = Vec3::new(0.75, -0.75, -1.0);
    let near = result
        .verts
        .iter()
        .filter(|&&v| (mesh.vertex_position(v) - front_meet).length() < 1.0e-5)
        .count();
    assert_eq!(
        near, 1,
        "the planar Front sector boundvert is the exact meet"
    );

    // Vertex 0 and the two far endpoints are consumed.
    for removed in [
        Vec3::new(1.0, -1.0, -1.0),
        Vec3::new(-1.0, -1.0, -1.0),
        Vec3::new(1.0, 1.0, -1.0),
    ] {
        assert!(
            !mesh
                .vert_keys()
                .into_iter()
                .any(|v| mesh.vertex_position(v) == removed),
            "original endpoint {removed:?} removed",
        );
    }
    mesh.validate_topology();
}

/// BEVEL.md §7.6.3 (`s = 3`, true junction): beveling all three of cube vertex
/// 0's edges {0,1}, {0,3}, {0,4} makes vertex 0 a three-edge junction with no
/// unbeveled legs (the cleanest junction, §7.6.5). Its three boundverts — one
/// per planar sector face (Front, Left, Bottom) — form a ring closed by one
/// flat **patch face** (§7.6.3). Far endpoints 1, 3, 4 are valence-3 terminals.
/// Count delta over the cube: +5 V / +9 E / +4 F → 13 V / 21 E / 10 F.
#[test]
fn edge_bevel_three_edge_junction_one_patch() {
    let mut mesh = LoopMesh::new();
    mesh_add_cube(&mut mesh);
    assert_eq!(counts(&mesh), (8, 12, 6));

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
            ..Default::default()
        },
    );

    assert_eq!(
        counts(&mesh),
        (13, 21, 10),
        "s=3 junction +5 V / +9 E / +4 F",
    );
    // Three strip quads + one patch face.
    assert_eq!(result.faces.len(), 4, "three strips + one patch");

    // The patch face is the triangle over the three boundverts. Each sector is
    // a single planar cube face, so each boundvert is an exact 2-line meet at
    // distance offset·√2 from the original corner (1,-1,-1).
    let patch = result
        .faces
        .iter()
        .copied()
        .find(|&f| mesh.face_vert_keys(f).len() == 3)
        .expect("the s=3 junction patch is a triangle");
    let corner = Vec3::new(1.0, -1.0, -1.0);
    let expected = 0.25 * 2.0_f32.sqrt();
    for v in mesh.face_vert_keys(patch) {
        let distance = (mesh.vertex_position(v) - corner).length();
        assert!(
            (distance - expected).abs() < 1.0e-5,
            "patch boundvert {distance} should sit offset·√2 from the corner",
        );
    }

    // Vertex 0 and all three far endpoints are consumed.
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
    mesh.validate_topology();
}

/// BEVEL.md §7.6.3: every edge of the `s = 3` junction patch face is shared
/// with exactly one arriving strip (radial count 2) — the manifold condition
/// the patch winding (§5.1) preserves.
#[test]
fn edge_bevel_three_edge_junction_patch_manifold() {
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
            ..Default::default()
        },
    );

    let patch = result
        .faces
        .iter()
        .copied()
        .find(|&f| mesh.face_vert_keys(f).len() == 3)
        .expect("patch triangle");
    for loop_key in mesh.face_loop_keys(patch) {
        let edge = mesh.loop_edge(loop_key);
        assert_eq!(
            mesh.edge_radial_iter(edge).count(),
            2,
            "each patch edge is manifold (shared with one strip)",
        );
    }
    mesh.validate_topology();
}
