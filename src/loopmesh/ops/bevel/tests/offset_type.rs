use super::*;

/// BEVEL.md §8.3.0.1: a junction-free `segments ≥ 2` edge set has no corner to
/// fold, but it is *not* inert — each isolated edge still sheds its
/// terminal-to-terminal through-strip, absorbed at both ends by the surviving
/// third face. Two disjoint cube edges at `segments = 2` each contribute
/// +4 V / +6 E / +2 F (six chain verts, two consumed endpoints; two strip quads).
#[test]
fn junction_free_multisegment_builds_through_strips() {
    let mut mesh = LoopMesh::new();
    mesh_add_cube(&mut mesh);

    // Two disjoint edges with no shared endpoint: every endpoint is terminal,
    // so there is no junction for the cutoff to fold.
    let e01 = cube_edge(
        &mesh,
        Vec3::new(1.0, -1.0, -1.0),
        Vec3::new(-1.0, -1.0, -1.0),
    );
    let e_far = cube_edge(&mesh, Vec3::new(-1.0, 1.0, 1.0), Vec3::new(1.0, 1.0, 1.0));

    let result = mesh.bevel(
        &[e01.into(), e_far.into()],
        BevelOptions {
            offset: 0.25,
            affect: BevelAffect::Edges,
            segments: 2,
            vmesh_method: BevelVmeshMethod::Cutoff,
            ..Default::default()
        },
    );

    assert_eq!(
        counts(&mesh),
        (16, 24, 10),
        "two isolated edges each shed a 2-quad through-strip",
    );
    assert_eq!(
        result.verts.len(),
        12,
        "three chain verts per terminal, four terminals",
    );
    assert_eq!(result.faces.len(), 4, "two strip quads per beveled edge");
    mesh.validate_topology();
}

/// BEVEL.md §2.1: a non-positive offset is a no-op for edge bevel too.
#[test]
fn edge_bevel_non_positive_offset_is_noop() {
    let mut mesh = LoopMesh::new();
    mesh_add_cube(&mut mesh);
    let before = counts(&mesh);
    let edge = cube_edge(
        &mesh,
        Vec3::new(1.0, -1.0, -1.0),
        Vec3::new(-1.0, -1.0, -1.0),
    );

    let result = mesh.bevel(
        &[edge.into()],
        BevelOptions {
            offset: 0.0,
            affect: BevelAffect::Edges,
            ..Default::default()
        },
    );
    assert_eq!(counts(&mesh), before, "offset <= 0 changes nothing");
    assert!(result.verts.is_empty() && result.faces.is_empty());
    mesh.validate_topology();
}

/// Bevel cube-corner vert `(1, -1, -1)` under `offset_type`, returning each
/// ring vert's slide distance from the original corner.
fn cube_corner_slides(offset: f32, offset_type: BevelOffsetType) -> Vec<f32> {
    let mut mesh = LoopMesh::new();
    mesh_add_cube(&mut mesh);
    let origin = Vec3::new(1.0, -1.0, -1.0);
    let corner = mesh
        .vert_keys()
        .into_iter()
        .find(|&v| mesh.vertex_position(v) == origin)
        .unwrap();
    let result = mesh.bevel(
        &[corner.into()],
        BevelOptions {
            offset,
            offset_type,
            ..Default::default()
        },
    );
    mesh.validate_topology();
    result
        .verts
        .iter()
        .map(|&v| (mesh.vertex_position(v) - origin).length())
        .collect()
}

/// BEVEL.md §4.3: `offset_type` changes only the slide distance, never the
/// topology. Every mode produces the same +2 V / +3 E / +1 F cube-corner
/// delta.
#[test]
fn offset_type_modes_are_topology_invariant() {
    for offset_type in [
        BevelOffsetType::Offset,
        BevelOffsetType::Width,
        BevelOffsetType::Depth,
        BevelOffsetType::Percent,
        BevelOffsetType::Absolute,
    ] {
        let mut mesh = LoopMesh::new();
        mesh_add_cube(&mut mesh);
        let corner = mesh
            .vert_keys()
            .into_iter()
            .find(|&v| mesh.vertex_position(v) == Vec3::new(1.0, -1.0, -1.0))
            .unwrap();
        mesh.bevel(
            &[corner.into()],
            BevelOptions {
                offset: 0.3,
                offset_type,
                ..Default::default()
            },
        );
        assert_eq!(
            counts(&mesh),
            (10, 15, 7),
            "{offset_type:?}: topology is mode-agnostic",
        );
    }
}

/// BEVEL.md §4.3 / §8: at the cube corner every edge's axis-relative angle is
/// `φ = arccos(1/√3) ≈ 54.74°`, so `WIDTH` slides by
/// `offset / |2·sin φ| ≈ 0.612·offset`, distinct from `OFFSET`'s `offset`.
#[test]
fn width_slide_chamfer_width_target() {
    let offset = 0.3;
    let phi = (1.0_f32 / 3.0_f32.sqrt()).acos();
    let expected = offset / (2.0 * phi.sin()).abs();
    for slide in cube_corner_slides(offset, BevelOffsetType::Width) {
        assert!(
            (slide - expected).abs() < 1.0e-5,
            "WIDTH slide {slide} should be {expected}",
        );
        assert!(
            (slide - offset).abs() > 1.0e-3,
            "WIDTH must differ from OFFSET at the cube corner",
        );
    }
}

/// BEVEL.md §4.3 / §8: `DEPTH` slides by `offset / |cos φ| ≈ 1.73·offset` at
/// the cube corner, distinct from `OFFSET`.
#[test]
fn depth_slide_perpendicular_depth_target() {
    let offset = 0.3;
    let phi = (1.0_f32 / 3.0_f32.sqrt()).acos();
    let expected = offset / phi.cos().abs();
    for slide in cube_corner_slides(offset, BevelOffsetType::Depth) {
        assert!(
            (slide - expected).abs() < 1.0e-5,
            "DEPTH slide {slide} should be {expected}",
        );
        assert!(
            (slide - offset).abs() > 1.0e-3,
            "DEPTH must differ from OFFSET at the cube corner",
        );
    }
}

/// BEVEL.md §4.3: `PERCENT` slides a fraction of each edge's length,
/// `d_i = |A_i − v| · offset / 100`. The cube's incident edges have length
/// `2.0`, so `PERCENT(25)` lands at `2.0 · 25 / 100 = 0.5` — the same as
/// `OFFSET(0.5)` — yet differs from `OFFSET(25)`.
#[test]
fn percent_slide_scales_by_edge_length() {
    let edge_length = 2.0_f32;
    let percent = 25.0_f32;
    let expected = edge_length * percent / 100.0;
    for slide in cube_corner_slides(percent, BevelOffsetType::Percent) {
        assert!(
            (slide - expected).abs() < 1.0e-5,
            "PERCENT slide {slide} should be {expected}",
        );
    }
    // Equivalence to OFFSET(0.5) is the length-scaling pin of §8.
    let offset_equivalent = cube_corner_slides(expected, BevelOffsetType::Offset);
    for slide in offset_equivalent {
        assert!((slide - expected).abs() < 1.0e-5);
    }
}

/// BEVEL.md §4.3: `ABSOLUTE` is a fixed slide equal to `offset`, numerically
/// identical to `OFFSET` at the foundational no-weights level.
#[test]
fn absolute_slide_coincides_with_offset() {
    let offset = 0.3;
    let absolute = cube_corner_slides(offset, BevelOffsetType::Absolute);
    let offset_slides = cube_corner_slides(offset, BevelOffsetType::Offset);
    assert_eq!(absolute.len(), offset_slides.len());
    for slide in &absolute {
        assert!(
            (slide - offset).abs() < 1.0e-5,
            "ABSOLUTE slide {slide} should equal offset {offset}",
        );
    }
    for (a, o) in absolute.iter().zip(&offset_slides) {
        assert!((a - o).abs() < 1.0e-6, "ABSOLUTE coincides with OFFSET");
    }
}
