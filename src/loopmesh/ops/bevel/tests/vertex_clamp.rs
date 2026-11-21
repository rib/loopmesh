use super::*;

/// BEVEL.md §4.4.3: beveling the two shared-edge endpoints of the cube with
/// `clamp_overlap = true` at `offset = 1.5` reduces the effective offset to
/// `1.0` (clamp factor `1.0 / 1.5`), so every ring vert slides exactly `1.0`
/// — not the requested `1.5`.
#[test]
fn clamp_overlap_reduces_slide_to_collision_offset() {
    let mut mesh = LoopMesh::new();
    mesh_add_cube(&mut mesh);
    // v0 = (1,-1,-1), v1 = (-1,-1,-1): the two endpoints of edge {0,1}.
    let a = vert_at(&mesh, Vec3::new(1.0, -1.0, -1.0));
    let b = vert_at(&mesh, Vec3::new(-1.0, -1.0, -1.0));
    let pos_a = mesh.vertex_position(a);
    let pos_b = mesh.vertex_position(b);

    let result = mesh.bevel(
        &[a.into(), b.into()],
        BevelOptions {
            offset: 1.5,
            clamp_overlap: true,
            ..Default::default()
        },
    );

    // Each ring vert slid from its own corner; the clamp pins the slide to
    // 1.0, the §4.4.3 collision offset, on every beveled edge.
    for &ring in &result.verts {
        let ring_position = mesh.vertex_position(ring);
        let slide = (ring_position - pos_a)
            .length()
            .min((ring_position - pos_b).length());
        assert!(
            (slide - 1.0).abs() < 1.0e-5,
            "clamped slide {slide} should be 1.0 (effective offset), not 1.5",
        );
    }
    mesh.validate_topology();
}

/// BEVEL.md §6.4.1: with `clamp_overlap = false` and `offset = 1.5`, the two
/// shared-edge ring verts each slide the full `1.5` and cross past each other:
/// `r_a` (from the `a = (1,-1,-1)` end) lands at `(-0.5,-1,-1)` and `r_b`
/// (from `b = (-1,-1,-1)`) at `(0.5,-1,-1)`. Each is placed off the *original*
/// edge — the place-before-surgery ordering.
#[test]
fn unclamped_adjacent_pair_crosses_past() {
    let mut mesh = LoopMesh::new();
    mesh_add_cube(&mut mesh);
    let a = vert_at(&mesh, Vec3::new(1.0, -1.0, -1.0));
    let b = vert_at(&mesh, Vec3::new(-1.0, -1.0, -1.0));

    let result = mesh.bevel(
        &[a.into(), b.into()],
        BevelOptions {
            offset: 1.5,
            clamp_overlap: false,
            ..Default::default()
        },
    );

    // The two ring verts on the shared edge land at the crossed-past
    // positions; each is a distinct ring vert.
    assert_eq!(
        result_has_vert_near(&mesh, &result, Vec3::new(-0.5, -1.0, -1.0)),
        1,
        "a-side ring vert at (-0.5,-1,-1)",
    );
    assert_eq!(
        result_has_vert_near(&mesh, &result, Vec3::new(0.5, -1.0, -1.0)),
        1,
        "b-side ring vert at (0.5,-1,-1)",
    );
    mesh.validate_topology();
}

/// A closed flat fan of `spokes` triangles with one rim vertex slid onto the apex,
/// so the spoke to that rim vertex is **zero-length** (§7.5.4). Returns the mesh, the
/// shared endpoint position (the origin), and the zero-length spoke edge.
fn zero_length_spoke_fan(spokes: usize, collapsed_rim: usize) -> (LoopMesh, Vec3, EdgeKey) {
    let (mut mesh, centre, rim) = planar_fan(spokes);
    mesh.set_vertex_position(rim[collapsed_rim], Vec3::ZERO);
    let spoke = mesh
        .query_edge_exists(centre, rim[collapsed_rim])
        .expect("the collapsed spoke exists");
    (mesh, Vec3::ZERO, spoke)
}

/// BEVEL.md §7.5.4: a zero-length beveled edge is **beveled, not declined**. The
/// spoke's coincident endpoints give it no direction, but it still sheds a strip and
/// grows the mesh to the §7.5.4-pinned counts (the same numbers a canonical, coincident-
/// bucketing comparison reports). An implementation that declined the edge would leave
/// the pristine 7-face fan untouched.
#[test]
fn zero_length_beveled_spoke_is_beveled_not_declined_segments1() {
    let (mut mesh, _shared, spoke) = zero_length_spoke_fan(7, 5);
    let result = mesh.bevel(
        &[spoke.into()],
        BevelOptions {
            offset: 0.2,
            affect: BevelAffect::Edges,
            segments: 1,
            ..Default::default()
        },
    );

    assert_eq!(
        counts(&mesh),
        (14, 22, 9),
        "§7.5.4 segments = 1: 14 V / 22 E / 9 F, not the pristine 8/14/7 of a decline",
    );
    // The strip was built (not skipped): a new face beyond the pristine fan's seven.
    assert!(
        result.faces.len() >= 2,
        "the zero-length edge sheds a strip and fills its valence-7 endpoint",
    );
    mesh.validate_topology();
}

#[test]
fn zero_length_beveled_spoke_is_beveled_not_declined_segments2() {
    let (mut mesh, _shared, spoke) = zero_length_spoke_fan(7, 5);
    let result = mesh.bevel(
        &[spoke.into()],
        BevelOptions {
            offset: 0.2,
            affect: BevelAffect::Edges,
            segments: 2,
            ..Default::default()
        },
    );

    assert_eq!(
        counts(&mesh),
        (16, 25, 10),
        "§7.5.4 segments = 2: 16 V / 25 E / 10 F",
    );
    assert!(result.faces.len() >= 3, "the strip and fill are built");
    mesh.validate_topology();
}

/// BEVEL.md §7.5.4: the four boundverts the zero-length edge contributes — two per
/// coincident endpoint — all pin to the shared endpoint coordinate (a zero-area strip),
/// regardless of the first/second disk-order designation of the leg meet.
#[test]
fn zero_length_beveled_spoke_boundverts_pin_to_the_shared_endpoint() {
    // §7.5.4: the four boundverts (two per coincident endpoint) always pin to the
    // shared point. At `segments = 2` the zero-area strip's two interior chain samples
    // collapse there too, since the strip's chords have zero length.
    for (segments, expected_on_shared) in [(1u32, 4usize), (2, 6)] {
        let (mut mesh, shared, spoke) = zero_length_spoke_fan(7, 5);
        let result = mesh.bevel(
            &[spoke.into()],
            BevelOptions {
                offset: 0.2,
                affect: BevelAffect::Edges,
                segments,
                ..Default::default()
            },
        );

        let on_shared = result
            .verts
            .iter()
            .filter(|&&v| (mesh.vertex_position(v) - shared).length() < 1.0e-6)
            .count();
        assert_eq!(
            on_shared, expected_on_shared,
            "segments = {segments}: the zero-length edge's boundverts pin to the shared \
             point — no NaN, no offset push",
        );
        // No minted vertex is non-finite (the zero-direction guard held).
        assert!(
            result
                .verts
                .iter()
                .all(|&v| mesh.vertex_position(v).is_finite()),
            "segments = {segments}: every minted vert is finite",
        );
    }
}

/// BEVEL.md §6.4.2: the clamped-collapse degenerate. At the exact collision
/// (`offset = 1.5` clamped to effective `1.0`, `d_a + d_b == L == 2`), the
/// two shared-edge ring verts land *coincident* at the edge midpoint
/// `(0,-1,-1)` — but remain **two distinct ring verts** joined by a
/// zero-length inner segment, with no weld / merge.
#[test]
fn clamped_collapse_is_coincident_pair_not_merge() {
    let mut mesh = LoopMesh::new();
    mesh_add_cube(&mut mesh);
    let a = vert_at(&mesh, Vec3::new(1.0, -1.0, -1.0));
    let b = vert_at(&mesh, Vec3::new(-1.0, -1.0, -1.0));

    let result = mesh.bevel(
        &[a.into(), b.into()],
        BevelOptions {
            offset: 1.5,
            clamp_overlap: true,
            ..Default::default()
        },
    );

    // Exactly two distinct ring verts coincide at the midpoint (one from each
    // corner) — not one merged vertex.
    let midpoint = Vec3::new(0.0, -1.0, -1.0);
    let coincident: Vec<VertexKey> = result
        .verts
        .iter()
        .copied()
        .filter(|&v| (mesh.vertex_position(v) - midpoint).length() < 1.0e-5)
        .collect();
    assert_eq!(
        coincident.len(),
        2,
        "two distinct ring verts coincide at the midpoint, no merge",
    );
    assert_ne!(
        coincident[0], coincident[1],
        "the pair is two distinct verts"
    );

    // They are joined by a real (zero-length) inner-segment edge.
    let segment = mesh.query_edge_exists(coincident[0], coincident[1]);
    assert!(
        segment.is_some(),
        "the surviving inner segment edge joins the coincident pair",
    );
    let length = {
        let [p, q] = mesh.edge_vert_keys(segment.unwrap());
        (mesh.vertex_position(p) - mesh.vertex_position(q)).length()
    };
    assert!(length < 1.0e-5, "the inner segment is zero-length");
    mesh.validate_topology();
}

/// BEVEL.md §6.4.3: the original shared edge survives as the inner segment
/// `r_a`–`r_b` between the two corners' chamfer edges, and (being an interior
/// manifold edge) keeps radial count 2 — both straddling faces re-stitched
/// onto it.
#[test]
fn shared_edge_survives_as_inner_segment_radial_two() {
    let mut mesh = LoopMesh::new();
    mesh_add_cube(&mut mesh);
    let a = vert_at(&mesh, Vec3::new(1.0, -1.0, -1.0));
    let b = vert_at(&mesh, Vec3::new(-1.0, -1.0, -1.0));

    let result = mesh.bevel(
        &[a.into(), b.into()],
        BevelOptions {
            offset: 0.5,
            clamp_overlap: false,
            ..Default::default()
        },
    );

    // r_a slid 0.5 from (1,-1,-1) toward (-1,-1,-1) -> (0.5,-1,-1).
    // r_b slid 0.5 from (-1,-1,-1) toward (1,-1,-1) -> (-0.5,-1,-1).
    let r_a = vert_at(&mesh, Vec3::new(0.5, -1.0, -1.0));
    let r_b = vert_at(&mesh, Vec3::new(-0.5, -1.0, -1.0));
    let segment = mesh
        .query_edge_exists(r_a, r_b)
        .expect("the shared edge survives as the r_a-r_b inner segment");

    assert_eq!(
        mesh.edge_radial_iter(segment).count(),
        2,
        "the surviving inner segment keeps radial count 2 (both straddling faces)",
    );

    // The two caps are independent (§6.4.4): the inner segment is on the
    // re-stitched faces, never on either cap, so it is not a chamfer edge.
    // Two distinct triangular caps were produced.
    assert_eq!(result.faces.len(), 2, "two independent corner caps");
    for &cap in &result.faces {
        assert_eq!(mesh.face_vert_keys(cap).len(), 3, "each cap is a triangle");
        assert!(
            !mesh.face_vert_keys(cap).contains(&r_a) || !mesh.face_vert_keys(cap).contains(&r_b),
            "no single cap spans both r_a and r_b",
        );
    }
    mesh.validate_topology();
}

/// BEVEL.md §6.1 / §2.3: an interior valence-3 cube corner is replaced by
/// three ring verts, a triangular cap, and three re-stitched faces.
/// Count delta +2 V / +3 E / +1 F.
#[test]
fn interior_closed_fan_cube_corner() {
    let mut mesh = LoopMesh::new();
    mesh_add_cube(&mut mesh);
    let before = counts(&mesh);
    assert_eq!(before, (8, 12, 6));

    // v0 = (1, -1, -1): an interior valence-3 corner (Front/Left/Bottom).
    let corner = mesh
        .vert_keys()
        .into_iter()
        .find(|&v| mesh.vertex_position(v) == Vec3::new(1.0, -1.0, -1.0))
        .expect("cube corner exists");

    let result = mesh.bevel(
        &[corner.into()],
        BevelOptions {
            offset: 0.25,
            ..Default::default()
        },
    );

    assert_eq!(counts(&mesh), (10, 15, 7), "interior +2 V / +3 E / +1 F");
    assert_eq!(result.verts.len(), 3, "three ring verts");
    assert_eq!(result.faces.len(), 1, "single triangular cap");

    // The cap is a triangle (n = 3).
    let cap = result.faces[0];
    assert_eq!(mesh.face_vert_keys(cap).len(), 3, "triangular cap");

    // The original corner is gone.
    assert!(
        !mesh.vert_keys().into_iter().any(|v| v == corner),
        "beveled vertex removed",
    );

    mesh.validate_topology();
}

// ---- §8.10.1 face_strength_mode shading post-pass ----
