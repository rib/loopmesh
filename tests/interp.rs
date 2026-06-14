//! Customdata interpolation tests.
//!
//! Each test registers a known-value attribute, runs a mid-level op, and
//! asserts the new element's attribute matches the interp semantics
//! documented in `radial-mesh-guides/BMESH.md`:
//!
//! - `edge_split(e, near, fac)` — new vertex attrib = `(1-fac)*near + fac*far`;
//!   new edge attrib = source edge's value (copy).
//! - `face_split(f, v1, v2)` — new face attrib = source face's value.
//! - `face_poke(f)` — centre vertex attrib = uniform average of corners;
//!   each spawned face attrib = source face's value.
//! - `euler_vert_separate(v)` — each new vertex attrib = source vertex's
//!   value (copy).
//! - `edge_collapse(e, v_kill)` — survivor vertex attrib = `0.5*v_keep +
//!   0.5*v_kill`.
//! - `euler_edge_separate(e, l)` — new edge attrib = source edge's value.

use glam::Vec3;
use loopmesh::loopmesh::{
    AttrType, AttributeKey, EdgeCreateFlags, ElementAttributeDesc, ElementKey, FaceCreateFlags,
    InterpPolicy, LoopMesh, MeshIter,
};

// ----- Small helpers -----

fn make_quad(
    mesh: &mut LoopMesh,
) -> (
    [loopmesh::loopmesh::VertexKey; 4],
    loopmesh::loopmesh::FaceKey,
    [loopmesh::loopmesh::EdgeKey; 4],
) {
    let v0 = mesh.create_vertex(Vec3::new(0.0, 0.0, 0.0));
    let v1 = mesh.create_vertex(Vec3::new(1.0, 0.0, 0.0));
    let v2 = mesh.create_vertex(Vec3::new(1.0, 1.0, 0.0));
    let v3 = mesh.create_vertex(Vec3::new(0.0, 1.0, 0.0));
    let e0 = mesh.create_edge([v0, v1], EdgeCreateFlags::NONE);
    let e1 = mesh.create_edge([v1, v2], EdgeCreateFlags::NONE);
    let e2 = mesh.create_edge([v2, v3], EdgeCreateFlags::NONE);
    let e3 = mesh.create_edge([v3, v0], EdgeCreateFlags::NONE);
    let f = mesh.face_create(&[v0, v1, v2, v3], &[e0, e1, e2, e3], FaceCreateFlags::NONE);
    ([v0, v1, v2, v3], f, [e0, e1, e2, e3])
}

fn vec3_close(a: [f32; 3], b: [f32; 3]) -> bool {
    (a[0] - b[0]).abs() < 1e-5 && (a[1] - b[1]).abs() < 1e-5 && (a[2] - b[2]).abs() < 1e-5
}

// ----- Tests -----

#[test]
fn edge_split_interpolates_vertex_attrib_linearly() {
    let mut mesh = LoopMesh::new();
    let v0 = mesh.create_vertex(Vec3::ZERO);
    let v1 = mesh.create_vertex(Vec3::X);
    let e = mesh.create_edge([v0, v1], EdgeCreateFlags::NONE);

    let color_key = mesh
        .register_vertex_attribute(ElementAttributeDesc::numeric(
            AttributeKey::custom("color"),
            AttrType::Vec3,
        ))
        .unwrap();
    mesh.vertex_attribs_mut()
        .set_vec3(color_key, v0.as_usize(), [1.0, 0.0, 0.0]);
    mesh.vertex_attribs_mut()
        .set_vec3(color_key, v1.as_usize(), [0.0, 1.0, 0.0]);

    let (v_new, _e_new) = mesh.edge_split(e, v0, 0.25);
    // fac=0.25, near=v0 → expect 0.75*v0 + 0.25*v1 = [0.75, 0.25, 0.0]
    let got = mesh
        .vertex_attribs()
        .get_vec3(color_key, v_new.as_usize())
        .unwrap();
    assert!(
        vec3_close(got, [0.75, 0.25, 0.0]),
        "got {got:?}, expected [0.75, 0.25, 0.0]"
    );
}

#[test]
fn edge_split_at_midpoint_averages() {
    let mut mesh = LoopMesh::new();
    let v0 = mesh.create_vertex(Vec3::ZERO);
    let v1 = mesh.create_vertex(Vec3::X);
    let e = mesh.create_edge([v0, v1], EdgeCreateFlags::NONE);

    let color = mesh
        .register_vertex_attribute(ElementAttributeDesc::numeric(
            AttributeKey::custom("color"),
            AttrType::Vec4,
        ))
        .unwrap();
    mesh.vertex_attribs_mut()
        .set_vec4(color, v0.as_usize(), [4.0, 0.0, 0.0, 0.0]);
    mesh.vertex_attribs_mut()
        .set_vec4(color, v1.as_usize(), [0.0, 8.0, 0.0, 2.0]);

    let (v_new, _e_new) = mesh.edge_split(e, v0, 0.5);
    let got = mesh
        .vertex_attribs()
        .get_vec4(color, v_new.as_usize())
        .unwrap();
    let expected = [2.0_f32, 4.0, 0.0, 1.0];
    for i in 0..4 {
        assert!(
            (got[i] - expected[i]).abs() < 1e-5,
            "lane {i}: got {} expected {}",
            got[i],
            expected[i]
        );
    }
}

#[test]
fn edge_split_copies_edge_attrib_to_new_edge() {
    let mut mesh = LoopMesh::new();
    let v0 = mesh.create_vertex(Vec3::ZERO);
    let v1 = mesh.create_vertex(Vec3::X);
    let e = mesh.create_edge([v0, v1], EdgeCreateFlags::NONE);

    let crease = mesh
        .register_edge_attribute(ElementAttributeDesc::numeric(
            AttributeKey::custom("crease"),
            AttrType::F32,
        ))
        .unwrap();
    mesh.edge_attribs_mut().set_f32(crease, e.as_usize(), 0.75);

    let (_v_new, e_new) = mesh.edge_split(e, v0, 0.5);

    // Both halves should carry the source crease (e is reused for the
    // near half; e_new is the far half).
    let near_crease = mesh.edge_attribs().get_f32(crease, e.as_usize()).unwrap();
    let far_crease = mesh
        .edge_attribs()
        .get_f32(crease, e_new.as_usize())
        .unwrap();
    assert!((near_crease - 0.75).abs() < 1e-6);
    assert!((far_crease - 0.75).abs() < 1e-6);
}

#[test]
fn face_split_copies_face_attrib_to_new_face() {
    let mut mesh = LoopMesh::new();
    let ([v0, _v1, v2, _v3], f, _es) = make_quad(&mut mesh);

    // Use a Vec3 attrib so we can read it back through the public
    // typed setter/getter. (U32 buckets exist but the public typed API for
    // U32 set/get is intentionally not surfaced yet — interp policy for
    // ints is exercised via the integer path inside ElementAttributes.)
    let mat_color = mesh
        .register_face_attribute(ElementAttributeDesc::numeric(
            AttributeKey::custom("material_color"),
            AttrType::Vec3,
        ))
        .unwrap();
    mesh.face_attribs_mut()
        .set_vec3(mat_color, f.as_usize(), [0.25, 0.5, 0.75]);

    let (new_face, _new_edge) = mesh.face_split(f, v0, v2).unwrap();
    let got = mesh
        .face_attribs()
        .get_vec3(mat_color, new_face.as_usize())
        .unwrap();
    assert!(vec3_close(got, [0.25, 0.5, 0.75]), "got {got:?}");
}

#[test]
fn face_poke_centre_vertex_is_uniform_average() {
    let mut mesh = LoopMesh::new();
    let (verts, f, _) = make_quad(&mut mesh);

    let color = mesh
        .register_vertex_attribute(ElementAttributeDesc::numeric(
            AttributeKey::custom("color"),
            AttrType::Vec3,
        ))
        .unwrap();
    // Set the four corners to RGBA-like values; the centre should land on
    // the uniform mean.
    mesh.vertex_attribs_mut()
        .set_vec3(color, verts[0].as_usize(), [1.0, 0.0, 0.0]);
    mesh.vertex_attribs_mut()
        .set_vec3(color, verts[1].as_usize(), [0.0, 1.0, 0.0]);
    mesh.vertex_attribs_mut()
        .set_vec3(color, verts[2].as_usize(), [0.0, 0.0, 1.0]);
    mesh.vertex_attribs_mut()
        .set_vec3(color, verts[3].as_usize(), [1.0, 1.0, 1.0]);

    let v_centre = mesh.face_poke(f);
    let got = mesh
        .vertex_attribs()
        .get_vec3(color, v_centre.as_usize())
        .unwrap();
    let expected = [
        (1.0 + 0.0 + 0.0 + 1.0) / 4.0,
        (0.0 + 1.0 + 0.0 + 1.0) / 4.0,
        (0.0 + 0.0 + 1.0 + 1.0) / 4.0,
    ];
    assert!(
        vec3_close(got, expected),
        "got {got:?} expected {expected:?}"
    );
}

#[test]
fn face_poke_spawned_faces_inherit_source_attrib() {
    let mut mesh = LoopMesh::new();
    let (_verts, f, _) = make_quad(&mut mesh);

    let mat_color = mesh
        .register_face_attribute(ElementAttributeDesc::numeric(
            AttributeKey::custom("material_color"),
            AttrType::Vec3,
        ))
        .unwrap();
    mesh.face_attribs_mut()
        .set_vec3(mat_color, f.as_usize(), [0.1, 0.2, 0.3]);

    let _ = mesh.face_poke(f);

    // After poke, the original face is killed and 4 new triangle faces exist.
    // Every spawned face should carry the source colour.
    let face_keys = mesh.face_keys();
    assert_eq!(face_keys.len(), 4);
    for fk in face_keys {
        let got = mesh
            .face_attribs()
            .get_vec3(mat_color, fk.as_usize())
            .unwrap();
        assert!(
            vec3_close(got, [0.1, 0.2, 0.3]),
            "face {fk:?}: got {got:?}, expected source color"
        );
    }
}

#[test]
fn vert_separate_copies_source_attrib_to_new_verts() {
    // Bowtie: two triangles sharing one vertex. vert_separate at the shared
    // corner produces two verts each carrying the source's attrib.
    let mut mesh = LoopMesh::new();
    let v_share = mesh.create_vertex(Vec3::ZERO);
    let va = mesh.create_vertex(Vec3::new(-1.0, -1.0, 0.0));
    let vb = mesh.create_vertex(Vec3::new(-1.0, 1.0, 0.0));
    let vc = mesh.create_vertex(Vec3::new(1.0, -1.0, 0.0));
    let vd = mesh.create_vertex(Vec3::new(1.0, 1.0, 0.0));
    let e0 = mesh.create_edge([v_share, va], EdgeCreateFlags::NONE);
    let e1 = mesh.create_edge([va, vb], EdgeCreateFlags::NONE);
    let e2 = mesh.create_edge([vb, v_share], EdgeCreateFlags::NONE);
    let e3 = mesh.create_edge([v_share, vc], EdgeCreateFlags::NONE);
    let e4 = mesh.create_edge([vc, vd], EdgeCreateFlags::NONE);
    let e5 = mesh.create_edge([vd, v_share], EdgeCreateFlags::NONE);
    let _f0 = mesh.face_create(&[v_share, va, vb], &[e0, e1, e2], FaceCreateFlags::NONE);
    let _f1 = mesh.face_create(&[v_share, vc, vd], &[e3, e4, e5], FaceCreateFlags::NONE);

    let color = mesh
        .register_vertex_attribute(ElementAttributeDesc::numeric(
            AttributeKey::custom("color"),
            AttrType::Vec3,
        ))
        .unwrap();
    mesh.vertex_attribs_mut()
        .set_vec3(color, v_share.as_usize(), [0.4, 0.5, 0.6]);

    let result = mesh.euler_vert_separate(v_share);
    assert_eq!(result.len(), 2, "bowtie should separate into 2 fans");

    for v in &result {
        let got = mesh.vertex_attribs().get_vec3(color, v.as_usize()).unwrap();
        assert!(
            vec3_close(got, [0.4, 0.5, 0.6]),
            "vert {v:?}: got {got:?}, expected source color"
        );
    }
}

#[test]
fn edge_collapse_blends_survivor_vertex_50_50() {
    let mut mesh = LoopMesh::new();
    let v0 = mesh.create_vertex(Vec3::ZERO);
    let v1 = mesh.create_vertex(Vec3::X);
    let e = mesh.create_edge([v0, v1], EdgeCreateFlags::NONE);

    let color = mesh
        .register_vertex_attribute(ElementAttributeDesc::numeric(
            AttributeKey::custom("color"),
            AttrType::Vec3,
        ))
        .unwrap();
    mesh.vertex_attribs_mut()
        .set_vec3(color, v0.as_usize(), [1.0, 0.0, 0.0]);
    mesh.vertex_attribs_mut()
        .set_vec3(color, v1.as_usize(), [0.0, 1.0, 0.0]);

    // Collapse v1 onto v0 → survivor is v0.
    let survivor = mesh.edge_collapse(e, v1).unwrap();
    assert_eq!(survivor, v0);

    let got = mesh
        .vertex_attribs()
        .get_vec3(color, survivor.as_usize())
        .unwrap();
    assert!(
        vec3_close(got, [0.5, 0.5, 0.0]),
        "got {got:?}, expected midpoint blend"
    );
}

#[test]
fn no_interp_policy_leaves_destination_unchanged() {
    let mut mesh = LoopMesh::new();
    let v0 = mesh.create_vertex(Vec3::ZERO);
    let v1 = mesh.create_vertex(Vec3::X);
    let e = mesh.create_edge([v0, v1], EdgeCreateFlags::NONE);

    let opaque = mesh
        .register_vertex_attribute(
            ElementAttributeDesc::numeric(AttributeKey::custom("opaque_id"), AttrType::F32)
                .with_interp(InterpPolicy::NoInterp),
        )
        .unwrap();
    mesh.vertex_attribs_mut()
        .set_f32(opaque, v0.as_usize(), 42.0);
    mesh.vertex_attribs_mut()
        .set_f32(opaque, v1.as_usize(), 99.0);

    let (v_new, _) = mesh.edge_split(e, v0, 0.5);
    // NoInterp must leave the default-initialised value (0.0) in place.
    let got = mesh
        .vertex_attribs()
        .get_f32(opaque, v_new.as_usize())
        .unwrap();
    assert_eq!(got, 0.0);
}

#[test]
fn string_attribute_copy_first_on_edge_split() {
    let mut mesh = LoopMesh::new();
    let v0 = mesh.create_vertex(Vec3::ZERO);
    let v1 = mesh.create_vertex(Vec3::X);
    let e = mesh.create_edge([v0, v1], EdgeCreateFlags::NONE);

    let note = mesh
        .register_vertex_attribute(ElementAttributeDesc::string(AttributeKey::custom("note")))
        .unwrap();
    mesh.vertex_attribs_mut()
        .set_string(note, v0.as_usize(), Some("from-v0".to_string()));
    mesh.vertex_attribs_mut()
        .set_string(note, v1.as_usize(), Some("from-v1".to_string()));

    let (v_new, _) = mesh.edge_split(e, v0, 0.5);
    // CopyFirst → sources[0] is v0 (the near endpoint at weight 1-fac).
    let got = mesh
        .vertex_attribs()
        .get_string(note, v_new.as_usize())
        .map(|s| s.to_string());
    assert_eq!(got.as_deref(), Some("from-v0"));
}

// ----- Loop-attribute interpolation -----

#[test]
fn edge_split_interpolates_loop_uv_linearly() {
    // Single quad with a UV-shaped loop attribute on each corner. Split one
    // edge near the v0 endpoint and verify the new loop's UV blends between
    // the two corner loops on either side of the split.
    let mut mesh = LoopMesh::new();
    let (verts, _f, edges) = make_quad(&mut mesh);

    let uv = mesh
        .register_loop_attribute(ElementAttributeDesc::numeric(
            AttributeKey::custom("uv"),
            AttrType::Vec2,
        ))
        .unwrap();

    // Find each face loop and set a known UV. The four corners get u=v_index.
    // We need keys for the four perimeter loops to read back later.
    let mut corner_loops = std::collections::HashMap::new();
    let mut it = mesh.face_loops(_f);
    while let Some(l) = it.next(&mesh) {
        let v = mesh.loop_vertex(l);
        corner_loops.insert(v, l);
    }
    mesh.loop_attribs_mut()
        .set_vec2(uv, corner_loops[&verts[0]].as_usize(), [0.0, 0.0]);
    mesh.loop_attribs_mut()
        .set_vec2(uv, corner_loops[&verts[1]].as_usize(), [1.0, 0.0]);
    mesh.loop_attribs_mut()
        .set_vec2(uv, corner_loops[&verts[2]].as_usize(), [1.0, 1.0]);
    mesh.loop_attribs_mut()
        .set_vec2(uv, corner_loops[&verts[3]].as_usize(), [0.0, 1.0]);

    // Split edges[0] (v0→v1) at fac=0.25 from v0.
    let (v_new, _) = mesh.edge_split(edges[0], verts[0], 0.25);

    // The new loop sits at v_new on the face. Find it.
    let mut new_loop = None;
    let mut it = mesh.face_loops(_f);
    while let Some(l) = it.next(&mesh) {
        if mesh.loop_vertex(l) == v_new {
            new_loop = Some(l);
            break;
        }
    }
    let new_loop = new_loop.expect("new loop at v_new");
    let got = mesh
        .loop_attribs()
        .get_vec2(uv, new_loop.as_usize())
        .unwrap();
    // Expected: linear blend at fac=0.25 between v0's UV (0,0) and v1's (1,0)
    // → (0.25, 0.0).
    assert!(
        (got[0] - 0.25).abs() < 1e-5 && got[1].abs() < 1e-5,
        "got {got:?}, expected [0.25, 0.0]"
    );
}

#[test]
fn face_split_copies_loop_uv_from_corner_mates() {
    // The two new loops on the diagonal copy their UVs from the existing
    // same-vertex loops (the corner mates on the opposite halves).
    let mut mesh = LoopMesh::new();
    let ([v0, _v1, v2, _v3], f, _edges) = make_quad(&mut mesh);

    let uv = mesh
        .register_loop_attribute(ElementAttributeDesc::numeric(
            AttributeKey::custom("uv"),
            AttrType::Vec2,
        ))
        .unwrap();
    let mut corner_loops = std::collections::HashMap::new();
    let mut it = mesh.face_loops(f);
    while let Some(l) = it.next(&mesh) {
        corner_loops.insert(mesh.loop_vertex(l), l);
    }
    mesh.loop_attribs_mut()
        .set_vec2(uv, corner_loops[&v0].as_usize(), [0.5, 0.5]);
    mesh.loop_attribs_mut()
        .set_vec2(uv, corner_loops[&v2].as_usize(), [0.75, 0.25]);
    let v0_uv = [0.5_f32, 0.5];
    let v2_uv = [0.75_f32, 0.25];

    let (new_face, _new_edge) = mesh.face_split(f, v0, v2).unwrap();

    // Each half of the split face has one loop at v0 and one at v2. The new
    // loops on the diagonal copied their UVs from the original corner mates,
    // so the v0-side loops carry v0's UV and the v2-side loops carry v2's.
    for &fk in &[f, new_face] {
        let mut it = mesh.face_loops(fk);
        while let Some(l) = it.next(&mesh) {
            let v = mesh.loop_vertex(l);
            let got = mesh.loop_attribs().get_vec2(uv, l.as_usize()).unwrap();
            if v == v0 {
                assert!(
                    (got[0] - v0_uv[0]).abs() < 1e-5 && (got[1] - v0_uv[1]).abs() < 1e-5,
                    "face {fk:?} v0 loop: got {got:?}, expected {v0_uv:?}"
                );
            } else if v == v2 {
                assert!(
                    (got[0] - v2_uv[0]).abs() < 1e-5 && (got[1] - v2_uv[1]).abs() < 1e-5,
                    "face {fk:?} v2 loop: got {got:?}, expected {v2_uv:?}"
                );
            }
        }
    }
}

#[test]
fn face_poke_centre_loops_average_corner_uvs() {
    // face_poke creates N triangles. Each triangle's perimeter loops copy
    // from the original corner loops at the same vertex; each triangle's
    // centre-loop is the uniform average of all original corner loops.
    let mut mesh = LoopMesh::new();
    let (verts, f, _edges) = make_quad(&mut mesh);

    let uv = mesh
        .register_loop_attribute(ElementAttributeDesc::numeric(
            AttributeKey::custom("uv"),
            AttrType::Vec2,
        ))
        .unwrap();
    let mut corner_uvs = std::collections::HashMap::new();
    let mut it = mesh.face_loops(f);
    while let Some(l) = it.next(&mesh) {
        let v = mesh.loop_vertex(l);
        // Assign a known per-vertex UV: u = v_index, v = 2*v_index.
        let i = verts.iter().position(|&x| x == v).unwrap() as f32;
        let uv_val = [i, 2.0 * i];
        mesh.loop_attribs_mut().set_vec2(uv, l.as_usize(), uv_val);
        corner_uvs.insert(v, uv_val);
    }
    // Expected centre-loop UV is the uniform average over the four corners.
    let avg = {
        let mut s = [0.0_f32; 2];
        for &v in &verts {
            let uv_val = corner_uvs[&v];
            s[0] += uv_val[0];
            s[1] += uv_val[1];
        }
        [s[0] / verts.len() as f32, s[1] / verts.len() as f32]
    };

    let v_centre = mesh.face_poke(f);

    // For every spawned face, find its loop at v_centre and verify the UV
    // matches the corner average. Also verify the perimeter loops still
    // carry their corner-mate UVs.
    let face_keys = mesh.face_keys();
    assert_eq!(face_keys.len(), 4);
    for fk in face_keys {
        let mut it = mesh.face_loops(fk);
        while let Some(l) = it.next(&mesh) {
            let v = mesh.loop_vertex(l);
            let got = mesh.loop_attribs().get_vec2(uv, l.as_usize()).unwrap();
            if v == v_centre {
                assert!(
                    (got[0] - avg[0]).abs() < 1e-5 && (got[1] - avg[1]).abs() < 1e-5,
                    "centre loop on {fk:?}: got {got:?}, expected average {avg:?}"
                );
            } else {
                let want = corner_uvs[&v];
                assert!(
                    (got[0] - want[0]).abs() < 1e-5 && (got[1] - want[1]).abs() < 1e-5,
                    "perim loop {fk:?} at {v:?}: got {got:?}, expected {want:?}"
                );
            }
        }
    }
}

#[test]
fn loop_interp_is_skipped_when_no_loop_attribs() {
    // Sanity: edge_split / face_split / face_poke must still work and
    // produce the same topology as before when no loop attribs exist.
    // (The fast-path skips the snapshot+interp pass entirely.)
    let mut mesh = LoopMesh::new();
    let (verts, f, edges) = make_quad(&mut mesh);

    let (v_new, _e_new) = mesh.edge_split(edges[0], verts[0], 0.25);
    assert_eq!(mesh.vertex_count(), 5);
    assert_eq!(mesh.vertex_position(v_new).x - 0.25 < 1e-5, true);

    // face_split still works.
    let (_new_face, _new_edge) = mesh.face_split(f, v_new, verts[2]).unwrap();
    assert_eq!(mesh.face_count(), 2);
}
