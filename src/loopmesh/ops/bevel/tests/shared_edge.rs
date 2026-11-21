use super::*;

/// Register a per-edge Float3 `tint` layer and tag the edge between `pa` and
/// `pb` with `value`, returning the layer key and that edge's key.
fn tag_edge_tint(
    mesh: &mut LoopMesh,
    pa: Vec3,
    pb: Vec3,
    value: [f32; 3],
) -> (ElementAttributeKey, EdgeKey) {
    let tint = mesh
        .register_edge_attribute(ElementAttributeDesc::numeric(
            AttributeKey::custom("tint"),
            AttrType::Vec3,
        ))
        .unwrap();
    let edge = cube_edge(mesh, pa, pb);
    mesh.edge_attribs_mut()
        .set_vec3(tint, edge.as_usize(), value);
    (tint, edge)
}

/// BEVEL.md §6.4.3: beveling the two endpoints of an interior (manifold)
/// shared edge keeps that edge in place — **same `EdgeKey`**, radial count 2,
/// and its per-edge custom-data carried over — rather than minting a fresh
/// inner segment that would default-initialise the attribute.
#[test]
fn shared_edge_manifold_keeps_identity_and_per_edge_cd() {
    let mut mesh = LoopMesh::new();
    mesh_add_cube(&mut mesh);
    let pa = Vec3::new(1.0, -1.0, -1.0);
    let pb = Vec3::new(-1.0, -1.0, -1.0);
    let (tint, shared_edge) = tag_edge_tint(&mut mesh, pa, pb, [0.42, 0.0, 0.0]);

    let a = vert_at(&mesh, pa);
    let b = vert_at(&mesh, pb);
    mesh.bevel(
        &[a.into(), b.into()],
        BevelOptions {
            offset: 0.5,
            ..Default::default()
        },
    );

    // The same edge object survives — its key is still alive — re-pointed onto
    // the two ring verts r_a = (0.5,-1,-1), r_b = (-0.5,-1,-1).
    assert!(
        mesh.edge_keys().contains(&shared_edge),
        "the surviving inner segment is the *same* edge (identity preserved)",
    );
    let r_a = vert_at(&mesh, Vec3::new(0.5, -1.0, -1.0));
    let r_b = vert_at(&mesh, Vec3::new(-0.5, -1.0, -1.0));
    assert_eq!(
        mesh.query_edge_exists(r_a, r_b),
        Some(shared_edge),
        "the original edge, re-pointed to {{r_a, r_b}}, is the inner segment",
    );
    assert_eq!(
        mesh.edge_radial_iter(shared_edge).count(),
        2,
        "an interior shared edge keeps radial count 2",
    );

    let carried = mesh
        .edge_attribs()
        .get_vec3(tint, shared_edge.as_usize())
        .unwrap();
    assert_eq!(
        carried,
        [0.42, 0.0, 0.0],
        "the surviving edge keeps its original per-edge CD",
    );
    mesh.validate_topology();
}

/// BEVEL.md §6.4.3 (boundary sub-case): when the shared edge straddled by a
/// *single* face is beveled at both endpoints, the same edge still survives
/// in place and keeps its per-edge CD — with radial count 1, the one
/// re-stitched face.
#[test]
fn shared_edge_single_face_keeps_identity_and_per_edge_cd() {
    // A flat quad; bevel two adjacent corners so the shared edge between them
    // is a boundary edge (one incident face).
    let mut mesh = LoopMesh::new();
    let positions = [
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(2.0, 0.0, 0.0),
        Vec3::new(2.0, 2.0, 0.0),
        Vec3::new(0.0, 2.0, 0.0),
    ];
    let verts: Vec<VertexKey> = positions.iter().map(|&p| mesh.create_vertex(p)).collect();
    let edges: Vec<EdgeKey> = (0..4)
        .map(|i| mesh.create_edge([verts[i], verts[(i + 1) % 4]], EdgeCreateFlags::NONE))
        .collect();
    mesh.face_create(&verts, &edges, FaceCreateFlags::NONE);

    let tint = mesh
        .register_edge_attribute(ElementAttributeDesc::numeric(
            AttributeKey::custom("tint"),
            AttrType::Vec3,
        ))
        .unwrap();
    // The shared edge is verts[0]-verts[1], a boundary edge of the single quad.
    let shared_edge = edges[0];
    mesh.edge_attribs_mut()
        .set_vec3(tint, shared_edge.as_usize(), [0.0, 0.7, 0.0]);
    assert_eq!(
        mesh.edge_radial_iter(shared_edge).count(),
        1,
        "shared edge has a single incident face before the bevel",
    );

    mesh.bevel(
        &[verts[0].into(), verts[1].into()],
        BevelOptions {
            offset: 0.5,
            ..Default::default()
        },
    );

    assert!(
        mesh.edge_keys().contains(&shared_edge),
        "the single-face shared edge survives as the same edge",
    );
    // r_a slid 0.5 from (0,0,0) toward (2,0,0) -> (0.5,0,0);
    // r_b slid 0.5 from (2,0,0) toward (0,0,0) -> (1.5,0,0).
    let r_a = vert_at(&mesh, Vec3::new(0.5, 0.0, 0.0));
    let r_b = vert_at(&mesh, Vec3::new(1.5, 0.0, 0.0));
    assert_eq!(
        mesh.query_edge_exists(r_a, r_b),
        Some(shared_edge),
        "the re-pointed edge spans the two ring verts",
    );
    assert_eq!(
        mesh.edge_radial_iter(shared_edge).count(),
        1,
        "still a single incident face after the bevel",
    );
    let carried = mesh
        .edge_attribs()
        .get_vec3(tint, shared_edge.as_usize())
        .unwrap();
    assert_eq!(
        carried,
        [0.0, 0.7, 0.0],
        "the surviving boundary edge keeps its per-edge CD",
    );
    mesh.validate_topology();
}

/// BEVEL.md §6.4.2 (clamped-collapse degenerate): even when the surviving
/// inner segment has *zero length* (the two ring verts coincident), it is
/// still a real edge — the same one — and keeps its per-edge CD.
#[test]
fn shared_edge_clamped_collapse_keeps_per_edge_cd() {
    let mut mesh = LoopMesh::new();
    mesh_add_cube(&mut mesh);
    let pa = Vec3::new(1.0, -1.0, -1.0);
    let pb = Vec3::new(-1.0, -1.0, -1.0);
    let (tint, shared_edge) = tag_edge_tint(&mut mesh, pa, pb, [0.1, 0.2, 0.3]);

    let a = vert_at(&mesh, pa);
    let b = vert_at(&mesh, pb);
    mesh.bevel(
        &[a.into(), b.into()],
        BevelOptions {
            offset: 1.5,
            clamp_overlap: true,
            ..Default::default()
        },
    );

    // The clamp collapses r_a, r_b onto the midpoint (0,-1,-1): a zero-length
    // edge, but still the *same* surviving edge with its CD intact.
    assert!(
        mesh.edge_keys().contains(&shared_edge),
        "the zero-length surviving segment is still the same edge",
    );
    let [p, q] = mesh.edge_vert_keys(shared_edge);
    let length = (mesh.vertex_position(p) - mesh.vertex_position(q)).length();
    assert!(length < 1.0e-5, "the surviving segment is zero-length");
    let carried = mesh
        .edge_attribs()
        .get_vec3(tint, shared_edge.as_usize())
        .unwrap();
    assert_eq!(
        carried,
        [0.1, 0.2, 0.3],
        "even the collapsed edge keeps its per-edge CD",
    );
    mesh.validate_topology();
}

/// BEVEL.md §6.1 / §6.4.3: a **single-beveled-endpoint** corner-adjacent edge
/// (one beveled endpoint, one not) is re-pointed in place exactly like the
/// shared edge — same `EdgeKey`, same per-edge CD — it is *not* re-minted by
/// the incident-face rebuild. Bevels only vertex 0 and checks edge {0,3}
/// (vertex 3 not beveled) survives re-pointed onto 0's ring vert.
#[test]
fn corner_adjacent_edge_single_beveled_endpoint_keeps_identity_and_per_edge_cd() {
    let mut mesh = LoopMesh::new();
    mesh_add_cube(&mut mesh);
    // Edge {0,3}: (1,-1,-1)-(1,1,-1). Only vertex 0 will be beveled.
    let p0 = Vec3::new(1.0, -1.0, -1.0);
    let p3 = Vec3::new(1.0, 1.0, -1.0);
    let (tint, adjacent_edge) = tag_edge_tint(&mut mesh, p0, p3, [0.0, 0.0, 0.9]);

    let v0 = vert_at(&mesh, p0);
    let v3 = vert_at(&mesh, p3);
    mesh.bevel(
        &[v0.into()],
        BevelOptions {
            offset: 0.5,
            ..Default::default()
        },
    );

    assert!(
        mesh.edge_keys().contains(&adjacent_edge),
        "the corner-adjacent edge survives as the *same* edge (identity preserved)",
    );
    // Vertex 0's ring vert on {0,3} slides 0.5 from (1,-1,-1) toward
    // (1,1,-1) -> (1,-0.5,-1). Vertex 3 is untouched.
    let ring = vert_at(&mesh, Vec3::new(1.0, -0.5, -1.0));
    assert!(
        mesh.vert_keys().contains(&v3),
        "the non-beveled far endpoint is untouched",
    );
    assert_eq!(
        mesh.query_edge_exists(ring, v3),
        Some(adjacent_edge),
        "the original edge, re-pointed {{0 -> r_0}}, spans the ring vert and vertex 3",
    );
    assert_eq!(
        mesh.edge_radial_iter(adjacent_edge).count(),
        2,
        "the re-pointed interior edge keeps radial count 2",
    );
    let carried = mesh
        .edge_attribs()
        .get_vec3(tint, adjacent_edge.as_usize())
        .unwrap();
    assert_eq!(
        carried,
        [0.0, 0.0, 0.9],
        "the single-beveled-endpoint edge keeps its per-edge CD",
    );
    mesh.validate_topology();
}

/// BEVEL.md §6.1 / §6.4.3: over the full shared-edge pair, *every* incident
/// edge of both beveled corners is preserved in place — the shared edge {0,1}
/// (both endpoints beveled) and each corner-adjacent edge (one beveled
/// endpoint). Only the chamfer edges between a corner's ring verts are minted
/// fresh. This pins the broad per-edge-CD survival the narrow shared-edge fix
/// did not cover.
#[test]
fn shared_edge_pair_preserves_all_incident_edges_cd() {
    let mut mesh = LoopMesh::new();
    mesh_add_cube(&mut mesh);

    let tint = mesh
        .register_edge_attribute(ElementAttributeDesc::numeric(
            AttributeKey::custom("tint"),
            AttrType::Vec3,
        ))
        .unwrap();

    // Tag every incident edge of the two beveled corners 0 and 1 with a
    // distinct value, and record (EdgeKey, ring-vert-position-after-bevel,
    // far-endpoint-position, expected-tint) for each.
    let p0 = Vec3::new(1.0, -1.0, -1.0);
    let p1 = Vec3::new(-1.0, -1.0, -1.0);

    // (near beveled vertex position, far endpoint position, tint)
    let incident = [
        // shared edge {0,1}: both endpoints beveled.
        (p0, p1, [0.42, 0.0, 0.0]),
        // vertex 0's other edges {0,3}, {0,4}: one beveled endpoint.
        (p0, Vec3::new(1.0, 1.0, -1.0), [0.1, 0.0, 0.0]),
        (p0, Vec3::new(1.0, -1.0, 1.0), [0.2, 0.0, 0.0]),
        // vertex 1's other edges {1,2}, {1,5}: one beveled endpoint.
        (p1, Vec3::new(-1.0, 1.0, -1.0), [0.3, 0.0, 0.0]),
        (p1, Vec3::new(-1.0, -1.0, 1.0), [0.4, 0.0, 0.0]),
    ];

    let mut tagged: Vec<(EdgeKey, [f32; 3])> = Vec::new();
    for &(near, far, value) in &incident {
        let edge = cube_edge(&mesh, near, far);
        mesh.edge_attribs_mut()
            .set_vec3(tint, edge.as_usize(), value);
        tagged.push((edge, value));
    }

    let v0 = vert_at(&mesh, p0);
    let v1 = vert_at(&mesh, p1);
    mesh.bevel(
        &[v0.into(), v1.into()],
        BevelOptions {
            offset: 0.5,
            ..Default::default()
        },
    );

    for &(edge, value) in &tagged {
        assert!(
            mesh.edge_keys().contains(&edge),
            "every incident edge survives with the same EdgeKey",
        );
        let carried = mesh.edge_attribs().get_vec3(tint, edge.as_usize()).unwrap();
        assert_eq!(
            carried, value,
            "incident edge {edge:?} keeps its per-edge CD across the bevel",
        );
    }
    mesh.validate_topology();
}

/// BEVEL.md §6.5.1: each freshly-minted corner-fill chamfer edge inherits the
/// per-edge CD of the original incident edge that owns its *leading* ring-vert
/// endpoint in the re-stitched face's boundary order. The discriminating pin:
/// the chamfer edge whose leading ring vert is the ring vert of the shared
/// edge {0,1} picks up {0,1}'s tint, on *both* corners (§6.5.2 — {0,1}
/// contributes a ring vert to each beveled corner). A chamfer edge leads with
/// the {0,1} ring vert in every re-stitched face where {0,1} is the incoming
/// corner edge, so the {0,1} ring vert may lead more than one chamfer edge
/// within a corner; the load-bearing fact is that the inheritance fires (an
/// edge next to {0,1}'s entry carries {0,1}'s tint), not a fresh default.
#[test]
fn corner_fill_chamfer_edge_inherits_leading_owning_edge_cd() {
    let mut mesh = LoopMesh::new();
    mesh_add_cube(&mut mesh);

    let tint = mesh
        .register_edge_attribute(ElementAttributeDesc::numeric(
            AttributeKey::custom("tint"),
            AttrType::Vec3,
        ))
        .unwrap();

    let p0 = Vec3::new(1.0, -1.0, -1.0);
    let p1 = Vec3::new(-1.0, -1.0, -1.0);
    let shared_value = [0.42, 0.0, 0.0];

    // Tag the shared edge {0,1} distinctly; the rest stay at the default so
    // only the {0,1}-leading chamfer edge can carry the marker.
    let shared_edge = cube_edge(&mesh, p0, p1);
    mesh.edge_attribs_mut()
        .set_vec3(tint, shared_edge.as_usize(), shared_value);

    let v0 = vert_at(&mesh, p0);
    let v1 = vert_at(&mesh, p1);
    let result = mesh.bevel(
        &[v0.into(), v1.into()],
        BevelOptions {
            offset: 0.5,
            ..Default::default()
        },
    );

    // The {0,1} ring vert of corner 0 slides 0.5 from (1,-1,-1) toward
    // (-1,-1,-1) -> (0.5,-1,-1); of corner 1, from (-1,-1,-1) toward
    // (1,-1,-1) -> (-0.5,-1,-1). Each corner's fill must produce exactly one
    // chamfer edge that leads with its {0,1} ring vert and so inherits
    // {0,1}'s tint.
    for ring_pos in [Vec3::new(0.5, -1.0, -1.0), Vec3::new(-0.5, -1.0, -1.0)] {
        let ring = vert_at(&mesh, ring_pos);
        let inheriting: Vec<EdgeKey> = mesh
            .edge_keys()
            .into_iter()
            .filter(|&edge| {
                let [a, b] = mesh.edge_vert_keys(edge);
                (a == ring || b == ring)
                    && mesh.edge_attribs().get_vec3(tint, edge.as_usize()) == Some(shared_value)
            })
            .collect();
        assert!(
            !inheriting.is_empty(),
            "a chamfer edge at the {{0,1}} ring vert {ring_pos:?} inherits \
             {{0,1}}'s per-edge CD rather than default-filling (§6.5.1)",
        );
        // Each inheriting edge must be a genuine chamfer edge (both endpoints
        // ring verts), not a re-pointed survivor that happens to share the tint.
        for &edge in &inheriting {
            let [a, b] = mesh.edge_vert_keys(edge);
            let other = if a == ring { b } else { a };
            assert!(
                result_has_vert_near(&mesh, &result, mesh.vertex_position(other)) > 0
                    || other == ring,
                "the inheriting edge {edge:?} is a chamfer edge between two ring verts",
            );
        }
    }
    mesh.validate_topology();
}

/// BEVEL.md §6.5.1: no corner-fill chamfer edge is left default-filled — every
/// freshly-minted chamfer edge inherits a tagged original edge's per-edge CD.
/// This is the broad closure of the 6/18 corner-fill divergence: with every
/// incident edge of both corners tagged distinctly, each chamfer edge carries
/// the value of the original edge whose ring vert leads it (never the [0,0,0]
/// default a fresh-mint would leave).
#[test]
fn corner_fill_chamfer_edges_are_never_default_filled() {
    let mut mesh = LoopMesh::new();
    mesh_add_cube(&mut mesh);

    let tint = mesh
        .register_edge_attribute(ElementAttributeDesc::numeric(
            AttributeKey::custom("tint"),
            AttrType::Vec3,
        ))
        .unwrap();

    let p0 = Vec3::new(1.0, -1.0, -1.0);
    let p1 = Vec3::new(-1.0, -1.0, -1.0);

    // Tag every incident edge of both corners distinctly (shared + four
    // corner-adjacent), each component non-zero so an inherited value is
    // always distinguishable from the [0,0,0] default.
    let incident = [
        (p0, p1, [0.42, 0.5, 0.5]),
        (p0, Vec3::new(1.0, 1.0, -1.0), [0.1, 0.5, 0.5]),
        (p0, Vec3::new(1.0, -1.0, 1.0), [0.2, 0.5, 0.5]),
        (p1, Vec3::new(-1.0, 1.0, -1.0), [0.3, 0.5, 0.5]),
        (p1, Vec3::new(-1.0, -1.0, 1.0), [0.4, 0.5, 0.5]),
    ];
    let mut tagged_values: Vec<[f32; 3]> = Vec::new();
    for &(near, far, value) in &incident {
        let edge = cube_edge(&mesh, near, far);
        mesh.edge_attribs_mut()
            .set_vec3(tint, edge.as_usize(), value);
        tagged_values.push(value);
    }

    let v0 = vert_at(&mesh, p0);
    let v1 = vert_at(&mesh, p1);
    let result = mesh.bevel(
        &[v0.into(), v1.into()],
        BevelOptions {
            offset: 0.5,
            ..Default::default()
        },
    );

    // The new edges between two ring verts of the same corner are the
    // chamfer / fill edges; each must carry a tagged value, never the default.
    let ring_verts: Vec<VertexKey> = result.verts.clone();
    let mut chamfer_count = 0;
    for &edge in &result.edges {
        let [a, b] = mesh.edge_vert_keys(edge);
        let is_chamfer = ring_verts.contains(&a) && ring_verts.contains(&b);
        if !is_chamfer {
            continue;
        }
        chamfer_count += 1;
        let carried = mesh.edge_attribs().get_vec3(tint, edge.as_usize()).unwrap();
        assert_ne!(
            carried,
            [0.0, 0.0, 0.0],
            "chamfer edge {edge:?} must inherit an owning edge's CD, not default",
        );
        assert!(
            tagged_values.contains(&carried),
            "chamfer edge {edge:?} inherits one of the tagged owning-edge values, got {carried:?}",
        );
    }
    assert!(
        chamfer_count >= 2,
        "both interior corners contribute fill chamfer edges",
    );
    mesh.validate_topology();
}

/// BEVEL.md §6.5.1 (cap-frame leading): pin the *specific* chamfer-edge -> CD
/// mapping, not just the multiset. The cap is wound opposite its incident
/// faces (§5.1), so the chamfer edge `(r_i, r_(i+1))` inherits the **outgoing**
/// edge `e_(i+1)`'s CD (the owning edge of the cap's leading endpoint), uniformly
/// for every corner. This catches the cyclic-shift permutation a per-edge-set
/// check passes spuriously through: tagging each owning edge distinctly forces
/// the test to fail under the wrong (incoming) assignment.
#[test]
fn corner_fill_chamfer_edge_mapping_follows_outgoing_owning_edge() {
    let mut mesh = LoopMesh::new();
    mesh_add_cube(&mut mesh);

    let tint = mesh
        .register_edge_attribute(ElementAttributeDesc::numeric(
            AttributeKey::custom("tint"),
            AttrType::Vec3,
        ))
        .unwrap();

    // Corner v0 = (1,-1,-1) with its three incident edges distinctly tagged.
    let p0 = Vec3::new(1.0, -1.0, -1.0);
    let p1 = Vec3::new(-1.0, -1.0, -1.0); // e0 = {v0,v1}
    let p3 = Vec3::new(1.0, 1.0, -1.0); // e3 = {v3,v0}
    let p4 = Vec3::new(1.0, -1.0, 1.0); // e8 = {v0,v4}
    let e0_value = [0.10, 0.5, 0.5];
    let e3_value = [0.20, 0.5, 0.5];
    let e8_value = [0.30, 0.5, 0.5];
    for &(near, far, value) in &[(p0, p1, e0_value), (p0, p3, e3_value), (p0, p4, e8_value)] {
        let edge = cube_edge(&mesh, near, far);
        mesh.edge_attribs_mut()
            .set_vec3(tint, edge.as_usize(), value);
    }

    let v0 = vert_at(&mesh, p0);
    mesh.bevel(
        &[v0.into()],
        BevelOptions {
            offset: 0.5,
            ..Default::default()
        },
    );

    // Ring-vert positions: r_e0 slides v0->v1, r_e3 v0->v3, r_e8 v0->v4.
    let r_e0 = Vec3::new(0.5, -1.0, -1.0);
    let r_e3 = Vec3::new(1.0, -0.5, -1.0);
    let r_e8 = Vec3::new(1.0, -1.0, -0.5);

    // The chamfer triangle of v0 is r_e3 -> r_e0 -> r_e8 (-> r_e3). Per the
    // cap-frame rule each chamfer edge carries its *outgoing* owning edge's
    // tint, which lands on the edge's r_(i+1) (trailing-in-face) endpoint:
    //   (r_e3, r_e0)  outgoing e0  -> e0_value   (Front)
    //   (r_e0, r_e8)  outgoing e8  -> e8_value   (Bottom)
    //   (r_e8, r_e3)  outgoing e3  -> e3_value   (Left)
    // The wrong (incoming) assignment would put e3/e0/e8 on these instead,
    // so each row distinguishes the two readings.
    for &(near_a, near_b, expected) in &[
        (r_e3, r_e0, e0_value),
        (r_e0, r_e8, e8_value),
        (r_e8, r_e3, e3_value),
    ] {
        let a = vert_at(&mesh, near_a);
        let b = vert_at(&mesh, near_b);
        let edge = mesh
            .query_edge_exists(a, b)
            .expect("the chamfer edge between these two ring verts exists");
        let carried = mesh.edge_attribs().get_vec3(tint, edge.as_usize()).unwrap();
        assert_eq!(
            carried, expected,
            "chamfer edge {near_a:?}-{near_b:?} must inherit its outgoing owning \
             edge's CD {expected:?} (cap-frame leading, §6.5.1), got {carried:?}",
        );
    }
    mesh.validate_topology();
}
