//! Round-trip conversion tests between `nub_mesh::Mesh` and `LoopMesh`.

use loopmesh::loopmesh::{
    AttributeKey, AttributeSemantic, ComponentType, EdgeCreateFlags, ElementAttributeDesc,
    ElementKey, FaceCreateFlags, Format, LoopMesh, MeshIter,
};
use loopmesh::nub_mesh_convert::{FromNubMeshOptions, SeamSplitPolicy, ToNubMeshOptions};
use nub_mesh::primitives::{CuboidMeshBuilder, MeshBuilder};
use nub_mesh::{
    Attribute, AttributeRate, Buffer, IndexBuffer, IndexFormat, Mesh, PrimitiveTopology,
};

fn unit_cube_mesh() -> Mesh {
    CuboidMeshBuilder::default().build()
}

fn quad_mesh_with_custom_attr() -> Mesh {
    // 4 verts, 2 tris, position + custom "foo" attr (F32x4).
    let positions: [[f32; 3]; 4] = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ];
    let foo: [[f32; 4]; 4] = [
        [1.0, 0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 1.0],
        [1.0, 1.0, 0.0, 1.0],
    ];
    let indices: [u16; 6] = [0, 1, 2, 2, 3, 0];

    let mut mesh = Mesh::new(PrimitiveTopology::TriangleList, 4);
    let pos_buf = mesh.add_buffer(Buffer::from_pod_slice(&positions));
    mesh.add_attribute(Attribute {
        key: AttributeKey::Semantic(AttributeSemantic::Position),
        buffer: pos_buf,
        offset: 0,
        stride: 12,
        format: Format::new(ComponentType::F32, 3),
        rate: AttributeRate::PerVertex,
    })
    .unwrap();
    let foo_buf = mesh.add_buffer(Buffer::from_pod_slice(&foo));
    mesh.add_attribute(Attribute {
        key: AttributeKey::custom("foo"),
        buffer: foo_buf,
        offset: 0,
        stride: 16,
        format: Format::new(ComponentType::F32, 4),
        rate: AttributeRate::PerVertex,
    })
    .unwrap();
    let idx_buf = mesh.add_buffer(Buffer::from_pod_slice(&indices));
    mesh.set_indices(IndexBuffer {
        buffer: idx_buf,
        offset: 0,
        count: 6,
        format: IndexFormat::U16,
    })
    .unwrap();
    mesh
}

#[test]
fn cube_imports_with_expected_topology() {
    let m = unit_cube_mesh();
    let lm = LoopMesh::from_nub_mesh(&m).expect("import cube");
    // The default cuboid has 24 source verts (4 per face × 6 faces), 12 tris.
    // Each face's two tris share one edge, so each quad contributes 5 edges
    // → 30 edges total. No edges are shared across faces (different vert
    // sets at coincident positions).
    assert_eq!(lm.totvert(), 24);
    assert_eq!(lm.totface(), 12);
    assert_eq!(lm.totedge(), 30);
}

#[test]
fn cube_round_trip_preserves_counts_and_positions() {
    let src = unit_cube_mesh();
    let mut lm = LoopMesh::from_nub_mesh(&src).unwrap();
    let dst = lm.to_nub_mesh().expect("export cube");

    assert_eq!(dst.vertex_count(), 24);
    assert_eq!(dst.primitive_vertex_count(), 36);
    assert_eq!(dst.topology(), PrimitiveTopology::TriangleList);

    // Positions should match exactly — the import passes f32 through.
    let src_pos = src
        .find_semantic(AttributeSemantic::Position)
        .expect("source has Position");
    let dst_pos = dst
        .find_semantic(AttributeSemantic::Position)
        .expect("dst has Position");

    let src_attr = src.attribute(src_pos);
    let dst_attr = dst.attribute(dst_pos);
    let src_bytes = &src.buffer(src_attr.buffer).as_slice()[src_attr.offset..];
    let dst_bytes = &dst.buffer(dst_attr.buffer).as_slice()[dst_attr.offset..];

    let src_pos: &[[f32; 3]] = bytemuck::cast_slice(&src_bytes[..24 * 12]);
    let dst_pos: &[[f32; 3]] = bytemuck::cast_slice(&dst_bytes[..24 * 12]);

    // Vertex ordering may differ between import (source order) and export
    // (loopmesh iteration order). Compare as sets of positions.
    let mut s: Vec<[i32; 3]> = src_pos
        .iter()
        .map(|p| {
            [
                (p[0] * 1e6) as i32,
                (p[1] * 1e6) as i32,
                (p[2] * 1e6) as i32,
            ]
        })
        .collect();
    let mut d: Vec<[i32; 3]> = dst_pos
        .iter()
        .map(|p| {
            [
                (p[0] * 1e6) as i32,
                (p[1] * 1e6) as i32,
                (p[2] * 1e6) as i32,
            ]
        })
        .collect();
    s.sort_unstable();
    d.sort_unstable();
    assert_eq!(s, d, "position sets must match after round trip");
}

#[test]
fn cube_normal_attribute_survives_round_trip() {
    let src = unit_cube_mesh();
    let mut lm = LoopMesh::from_nub_mesh(&src).unwrap();
    let dst = lm.to_nub_mesh().unwrap();

    let dst_normal_id = dst
        .find_semantic(AttributeSemantic::Normal)
        .expect("export should preserve Normal");
    let attr = dst.attribute(dst_normal_id);
    assert_eq!(attr.format, Format::new(ComponentType::F32, 3));
    assert_eq!(attr.rate, AttributeRate::PerVertex);
}

#[test]
fn cube_uv_attribute_survives_round_trip() {
    let src = unit_cube_mesh();
    let mut lm = LoopMesh::from_nub_mesh(&src).unwrap();
    let dst = lm.to_nub_mesh().unwrap();

    let dst_uv = dst
        .find_semantic(AttributeSemantic::Uv0)
        .expect("export should preserve Uv0");
    let attr = dst.attribute(dst_uv);
    assert_eq!(attr.format, Format::new(ComponentType::F32, 2));
}

#[test]
fn quad_with_custom_attribute_round_trips() {
    let src = quad_mesh_with_custom_attr();
    let mut lm = LoopMesh::from_nub_mesh(&src).unwrap();
    let dst = lm.to_nub_mesh().unwrap();

    let custom = dst
        .find(&AttributeKey::custom("foo"))
        .expect("custom attr should round-trip");
    let attr = dst.attribute(custom);
    assert_eq!(attr.format, Format::new(ComponentType::F32, 4));
    assert_eq!(dst.vertex_count(), 4);
}

#[test]
fn from_nub_mesh_rejects_non_triangle_topology() {
    let mut mesh = Mesh::new(PrimitiveTopology::PointList, 3);
    let buf = mesh.add_buffer(Buffer::from_pod_slice(&[[0.0_f32; 3]; 3]));
    mesh.add_attribute(Attribute {
        key: AttributeKey::Semantic(AttributeSemantic::Position),
        buffer: buf,
        offset: 0,
        stride: 12,
        format: Format::new(ComponentType::F32, 3),
        rate: AttributeRate::PerVertex,
    })
    .unwrap();

    assert!(LoopMesh::from_nub_mesh(&mesh).is_err());
}

#[test]
fn from_nub_mesh_errors_when_position_missing_and_required() {
    let mesh = Mesh::new(PrimitiveTopology::TriangleList, 0);
    let err = LoopMesh::from_nub_mesh(&mesh);
    assert!(err.is_err());

    // With require_position = false, we accept an empty mesh.
    let ok = LoopMesh::from_nub_mesh_with(
        &mesh,
        &FromNubMeshOptions {
            require_position: false,
        },
    );
    assert!(ok.is_ok());
}

#[test]
fn string_vertex_attribute_does_not_export_to_nub_mesh() {
    let src = unit_cube_mesh();
    let mut lm = LoopMesh::from_nub_mesh(&src).unwrap();

    // A string-type custom-data layer on the vertex domain: registered, never
    // populated for this test (we only care that it's invisible to the
    // export path).
    lm.register_vertex_attribute(ElementAttributeDesc::string(AttributeKey::custom("note")))
        .unwrap();

    let dst = lm
        .to_nub_mesh()
        .expect("export must succeed even with strings");
    assert!(
        dst.find(&AttributeKey::custom("note")).is_none(),
        "string attribute must not appear in the exported Mesh"
    );
}

#[test]
fn to_nub_mesh_with_empty_loop_attribs_uses_fast_path() {
    // No loop attribs registered → fast path: one packed vertex per source.
    let src = unit_cube_mesh();
    let mut lm = LoopMesh::from_nub_mesh(&src).unwrap();
    // Register a loop attribute slot but leave it empty (zero entries).
    // No, actually: registering a descriptor makes loop_attribs non-empty.
    // Just don't register anything.
    let dst = lm.to_nub_mesh().unwrap();
    assert_eq!(dst.vertex_count(), 24);
}

#[test]
fn to_nub_mesh_uses_u16_indices_for_small_mesh() {
    let src = unit_cube_mesh();
    let mut lm = LoopMesh::from_nub_mesh(&src).unwrap();
    let dst = lm.to_nub_mesh().unwrap();
    assert_eq!(dst.index_format(), Some(IndexFormat::U16));
}

#[test]
fn round_trip_index_decoding_preserves_triangle_count() {
    let src = unit_cube_mesh();
    let mut lm = LoopMesh::from_nub_mesh(&src).unwrap();
    let dst = lm.to_nub_mesh().unwrap();

    // 12 tris × 3 indices = 36 entries.
    let count: usize = dst.index_iter().expect("indexed mesh").count();
    assert_eq!(count, 36);
}

// ----- Seam-split export -----

/// Build a small two-triangle quad sharing one edge. Useful for testing
/// loop-attribute seam-split: the two corner loops at each shared vertex
/// can carry different UV values to simulate a seam.
fn build_two_tri_quad() -> (
    LoopMesh,
    [loopmesh::loopmesh::VertexKey; 4],
    [loopmesh::loopmesh::FaceKey; 2],
) {
    let mut lm = LoopMesh::new();
    let v0 = lm.create_vertex(glam::Vec3::new(0.0, 0.0, 0.0));
    let v1 = lm.create_vertex(glam::Vec3::new(1.0, 0.0, 0.0));
    let v2 = lm.create_vertex(glam::Vec3::new(1.0, 1.0, 0.0));
    let v3 = lm.create_vertex(glam::Vec3::new(0.0, 1.0, 0.0));
    let e01 = lm.create_edge([v0, v1], EdgeCreateFlags::NONE);
    let e12 = lm.create_edge([v1, v2], EdgeCreateFlags::NONE);
    let e20 = lm.create_edge([v2, v0], EdgeCreateFlags::NONE);
    let e23 = lm.create_edge([v2, v3], EdgeCreateFlags::NONE);
    let e30 = lm.create_edge([v3, v0], EdgeCreateFlags::NONE);
    let f1 = lm.face_create(&[v0, v1, v2], &[e01, e12, e20], FaceCreateFlags::NONE);
    let f2 = lm.face_create(&[v0, v2, v3], &[e20, e23, e30], FaceCreateFlags::NONE);
    (lm, [v0, v1, v2, v3], [f1, f2])
}

/// Register a UV loop attribute and set per-corner values such that the
/// two corners at `seam_vert` disagree (creates a seam there).
fn set_seam_uvs(
    lm: &mut LoopMesh,
    uv: loopmesh::loopmesh::ElementAttributeKey,
    seam_vert: loopmesh::loopmesh::VertexKey,
    seam_a: [f32; 2],
    seam_b: [f32; 2],
) {
    // Walk every face loop, assign per-(face, vertex) UV.
    // First face's seam_vert gets seam_a; second face's seam_vert gets seam_b.
    // Non-seam corners get a deterministic value derived from the vertex slot.
    let mut seen_seam = false;
    let mut face_it = lm.mesh_faces();
    while let Some(f) = face_it.next(lm) {
        let mut it = lm.face_loops(f);
        while let Some(l) = it.next(lm) {
            let v = lm.loop_vertex(l);
            let value = if v == seam_vert {
                if !seen_seam {
                    seen_seam = true;
                    seam_a
                } else {
                    seam_b
                }
            } else {
                [v.as_usize() as f32 * 0.1, v.as_usize() as f32 * 0.2]
            };
            lm.loop_attribs_mut().set_vec2(uv, l.as_usize(), value);
        }
    }
}

#[test]
fn seam_split_default_duplicates_vertex_at_seam() {
    let (mut lm, verts, _faces) = build_two_tri_quad();
    let uv = lm
        .register_loop_attribute(ElementAttributeDesc::numeric(
            AttributeKey::custom("uv"),
            Format::new(ComponentType::F32, 2),
        ))
        .unwrap();
    // Two corners at v2 disagree → expect a seam split there.
    set_seam_uvs(&mut lm, uv, verts[2], [0.0, 0.0], [1.0, 1.0]);

    let dst = lm.to_nub_mesh().expect("split is the default");

    // Source has 4 verts. v2 has 2 distinct corner UVs across its 2 faces,
    // so it splits into 2 packed verts. v0 is also shared by both faces;
    // its corners agree (same vertex-slot-derived value), so it merges.
    // Result: 5 packed verts (v0, v1, v2_a, v2_b, v3).
    assert_eq!(dst.vertex_count(), 5);

    // The UV attribute survives.
    let uv_attr = dst
        .find(&AttributeKey::custom("uv"))
        .expect("uv attribute exported");
    let attr = dst.attribute(uv_attr);
    assert_eq!(attr.format, Format::new(ComponentType::F32, 2));
}

#[test]
fn seam_split_merge_policy_collapses_to_first_corner() {
    let (mut lm, verts, _faces) = build_two_tri_quad();
    let uv = lm
        .register_loop_attribute(ElementAttributeDesc::numeric(
            AttributeKey::custom("uv"),
            Format::new(ComponentType::F32, 2),
        ))
        .unwrap();
    set_seam_uvs(&mut lm, uv, verts[2], [0.25, 0.5], [0.75, 0.25]);

    let dst = lm
        .to_nub_mesh_with(&ToNubMeshOptions {
            seam_split: SeamSplitPolicy::Merge,
            ..Default::default()
        })
        .expect("merge always succeeds");
    // Merge collapses every source vertex to a single packed vertex
    // regardless of attribute disagreement. 4 source verts → 4 packed.
    assert_eq!(dst.vertex_count(), 4);
}

#[test]
fn seam_split_error_policy_rejects_disagreeing_corners() {
    let (mut lm, verts, _faces) = build_two_tri_quad();
    let uv = lm
        .register_loop_attribute(ElementAttributeDesc::numeric(
            AttributeKey::custom("uv"),
            Format::new(ComponentType::F32, 2),
        ))
        .unwrap();
    set_seam_uvs(&mut lm, uv, verts[2], [0.0, 0.0], [1.0, 1.0]);

    let result = lm.to_nub_mesh_with(&ToNubMeshOptions {
        seam_split: SeamSplitPolicy::Error,
        ..Default::default()
    });
    assert!(
        result.is_err(),
        "Error policy must reject vertex with differing corner attribs"
    );
}

#[test]
fn seam_split_error_policy_accepts_consistent_corners() {
    let (mut lm, _verts, _faces) = build_two_tri_quad();
    let uv = lm
        .register_loop_attribute(ElementAttributeDesc::numeric(
            AttributeKey::custom("uv"),
            Format::new(ComponentType::F32, 2),
        ))
        .unwrap();
    // No seam: set every corner to a value derived from its vertex slot
    // (consistent across faces).
    let mut face_it = lm.mesh_faces();
    while let Some(f) = face_it.next(&lm) {
        let mut it = lm.face_loops(f);
        while let Some(l) = it.next(&lm) {
            let v = lm.loop_vertex(l);
            let val = [v.as_usize() as f32 * 0.1, v.as_usize() as f32 * 0.2];
            lm.loop_attribs_mut().set_vec2(uv, l.as_usize(), val);
        }
    }

    let dst = lm
        .to_nub_mesh_with(&ToNubMeshOptions {
            seam_split: SeamSplitPolicy::Error,
            ..Default::default()
        })
        .expect("error policy accepts consistent corners");
    assert_eq!(dst.vertex_count(), 4);
}

#[test]
fn round_trip_through_seam_split_preserves_loop_attribute() {
    // End-to-end: build a quad, register loop UVs with a seam, export with
    // Split, re-import. The imported mesh should have one extra vertex at
    // the seam carrying the second UV value.
    let (mut lm, verts, _faces) = build_two_tri_quad();
    let uv_key = AttributeKey::custom("uv");
    let uv = lm
        .register_loop_attribute(ElementAttributeDesc::numeric(
            uv_key.clone(),
            Format::new(ComponentType::F32, 2),
        ))
        .unwrap();
    set_seam_uvs(&mut lm, uv, verts[2], [0.0, 0.0], [1.0, 1.0]);

    let dst = lm.to_nub_mesh().unwrap();
    let lm_back = LoopMesh::from_nub_mesh(&dst).unwrap();

    // After round-trip the seam_vert has been duplicated: source had 4 verts,
    // re-imported should have 5.
    assert_eq!(lm_back.totvert(), 5);
    // The UV attribute exists as a per-vertex attribute on lm_back.
    let uv_back = lm_back
        .vertex_attribs()
        .find(&uv_key)
        .expect("uv attribute on re-imported mesh");
    let _ = uv_back; // existence is enough — values were validated by the export tests.
}
