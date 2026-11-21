use super::*;

/// BEVEL.md §8.7.3 / §8.7.5: distinct weights on two non-adjacent corners in
/// one call produce a non-uniform result — the `0.4`-weighted corner slides
/// its ring verts only `0.4·offset`, the `1.0`-weighted corner the full
/// `offset`. v0 = (1,-1,-1) and v6 = (-1,1,1) are opposite cube corners
/// (disjoint one-rings), so each weight scales its corner independently.
#[test]
fn weighted_vertex_bevel_shrinks_one_corner() {
    let mut mesh = LoopMesh::new();
    mesh_add_cube(&mut mesh);
    let v0 = vert_at(&mesh, Vec3::new(1.0, -1.0, -1.0));
    let v6 = vert_at(&mesh, Vec3::new(-1.0, 1.0, 1.0));
    let pos0 = mesh.vertex_position(v0);
    let pos6 = mesh.vertex_position(v6);

    let offset = 0.5;
    let result = mesh.bevel(
        &[v0.into(), v6.into()],
        BevelOptions {
            offset,
            use_weights: true,
            vert_weights: &[(v0, 1.0), (v6, 0.4)],
            ..Default::default()
        },
    );

    let slide_full = nearest_ring_slide(&mesh, &result, pos0);
    let slide_weighted = nearest_ring_slide(&mesh, &result, pos6);
    assert!(
        (slide_full - offset).abs() < 1.0e-5,
        "weight 1.0 corner slides the full offset {offset}, got {slide_full}",
    );
    assert!(
        (slide_weighted - 0.4 * offset).abs() < 1.0e-5,
        "weight 0.4 corner slides 0.4·offset = {}, got {slide_weighted}",
        0.4 * offset,
    );
    assert!(
        slide_weighted < slide_full - 1.0e-5,
        "the weighted corner must be observably smaller than the full one",
    );
    mesh.validate_topology();
}

/// BEVEL.md §8.7.2 case 1 / §8.7.3: uniform weight `1.0` on every beveled
/// corner reproduces the unweighted geometry byte-for-byte (weight `1.0` is a
/// no-op prescale). Compared against the same bevel with `use_weights` off.
#[test]
fn uniform_weight_one_reproduces_unweighted() {
    let corner_slides = |use_weights: bool, weights: &[(VertexKey, f32)]| {
        let mut mesh = LoopMesh::new();
        mesh_add_cube(&mut mesh);
        let v0 = vert_at(&mesh, Vec3::new(1.0, -1.0, -1.0));
        let v6 = vert_at(&mesh, Vec3::new(-1.0, 1.0, 1.0));
        let pos0 = mesh.vertex_position(v0);
        let pos6 = mesh.vertex_position(v6);
        let result = mesh.bevel(
            &[v0.into(), v6.into()],
            BevelOptions {
                offset: 0.5,
                use_weights,
                vert_weights: weights,
                ..Default::default()
            },
        );
        let mut positions: Vec<Vec3> = result
            .verts
            .iter()
            .map(|&v| mesh.vertex_position(v))
            .collect();
        positions.sort_by(|a, b| {
            a.to_array()
                .partial_cmp(&b.to_array())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        (
            positions,
            nearest_ring_slide(&mesh, &result, pos0),
            nearest_ring_slide(&mesh, &result, pos6),
        )
    };

    // Build the weighted-1.0 call with the same fixture's keys. The keys are
    // deterministic across the two identical cube builds, so the all-`1.0`
    // weighted run and the gate-off run must coincide.
    let mut mesh = LoopMesh::new();
    mesh_add_cube(&mut mesh);
    let v0 = vert_at(&mesh, Vec3::new(1.0, -1.0, -1.0));
    let v6 = vert_at(&mesh, Vec3::new(-1.0, 1.0, 1.0));

    let (unweighted, u0, u6) = corner_slides(false, &[]);
    let (weighted, w0, w6) = corner_slides(true, &[(v0, 1.0), (v6, 1.0)]);

    assert!((u0 - 0.5).abs() < 1.0e-6 && (u6 - 0.5).abs() < 1.0e-6);
    assert!((w0 - u0).abs() < 1.0e-6 && (w6 - u6).abs() < 1.0e-6);
    assert_eq!(weighted.len(), unweighted.len());
    for (a, b) in weighted.iter().zip(&unweighted) {
        assert!(
            (*a - *b).length() < 1.0e-6,
            "all-1.0 weighted bevel must match the unweighted geometry",
        );
    }
}

/// BEVEL.md §8.7.2 case 1: with the gate off, the `vert_weights` slice is
/// ignored entirely — every corner slides the full offset even though the
/// slice would scale one of them down.
#[test]
fn weights_ignored_when_gate_off() {
    let mut mesh = LoopMesh::new();
    mesh_add_cube(&mut mesh);
    let v0 = vert_at(&mesh, Vec3::new(1.0, -1.0, -1.0));
    let v6 = vert_at(&mesh, Vec3::new(-1.0, 1.0, 1.0));
    let pos6 = mesh.vertex_position(v6);

    let result = mesh.bevel(
        &[v0.into(), v6.into()],
        BevelOptions {
            offset: 0.5,
            use_weights: false,
            vert_weights: &[(v0, 1.0), (v6, 0.4)],
            ..Default::default()
        },
    );

    let slide6 = nearest_ring_slide(&mesh, &result, pos6);
    assert!(
        (slide6 - 0.5).abs() < 1.0e-5,
        "gate off must ignore the 0.4 weight; corner should slide the full 0.5, got {slide6}",
    );
    mesh.validate_topology();
}

/// BEVEL.md §8.7.2 case 3 / §8.7.6: gate on, a beveled vertex *absent* from
/// `vert_weights` defaults to weight `0.0` — zero effective offset, no slide,
/// a clean no-op for that corner (its ring verts coincide with the original
/// corner). v0 is written `1.0` and bevels; v6 is omitted and does not.
#[test]
fn absent_weight_defaults_to_zero_no_op() {
    let mut mesh = LoopMesh::new();
    mesh_add_cube(&mut mesh);
    let v0 = vert_at(&mesh, Vec3::new(1.0, -1.0, -1.0));
    let v6 = vert_at(&mesh, Vec3::new(-1.0, 1.0, 1.0));
    let pos0 = mesh.vertex_position(v0);
    let pos6 = mesh.vertex_position(v6);

    let result = mesh.bevel(
        &[v0.into(), v6.into()],
        BevelOptions {
            offset: 0.5,
            use_weights: true,
            vert_weights: &[(v0, 1.0)], // v6 absent ⇒ weight 0.0
            ..Default::default()
        },
    );

    let slide0 = nearest_ring_slide(&mesh, &result, pos0);
    assert!(
        (slide0 - 0.5).abs() < 1.0e-5,
        "v0 weighted 1.0 slides the full offset, got {slide0}",
    );
    // v6's three ring verts all land at zero slide — coincident with the
    // original corner — so the nearest ring vert to pos6 is at distance 0.
    let slide6 = nearest_ring_slide(&mesh, &result, pos6);
    assert!(
        slide6 < 1.0e-6,
        "absent v6 defaults to weight 0.0: ring verts stay at the corner, got {slide6}",
    );
    mesh.validate_topology();
}
