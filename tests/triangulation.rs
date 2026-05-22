//! Property tests for the new ear-clip / quad-flip triangulation path
//! used by `to_nub_mesh` with `TriangulationPolicy::EarClip`.
//!
//! These tests exercise the LoopMesh-side integration:
//! [`triangulate_corners::corner_triangulation`] in particular, plus a
//! couple of end-to-end smoke tests through `to_nub_mesh`.
//!
//! A future A/B test against `bmesh-sys::BM_mesh_calc_tessellation`
//! would require a small FFI addition in `bmesh-sys/shim/`. That's
//! tracked as a parity gap rather than a regression in this milestone.

use glam::Vec3;
use loopmesh::loopmesh::{
    EdgeCreateFlags, FaceCreateFlags, FaceKey, LoopKey, LoopMesh, MeshIter, VertexKey,
};
use loopmesh::nub_mesh_convert::{SeamSplitPolicy, ToNubMeshOptions, TriangulationPolicy};
use loopmesh::triangulate_corners::corner_triangulation;
use nub_mesh::PrimitiveTopology;

// ============================== Helpers ==============================

/// Build a single-face LoopMesh from a list of 3D positions taken in
/// cycle order. Vertices and edges are created implicitly.
fn build_single_face_mesh(positions: &[[f32; 3]]) -> (LoopMesh, FaceKey, Vec<VertexKey>) {
    let mut lm = LoopMesh::new();
    let verts: Vec<VertexKey> = positions
        .iter()
        .map(|p| lm.create_vertex(Vec3::new(p[0], p[1], p[2])))
        .collect();
    let n = verts.len();
    let edges: Vec<_> = (0..n)
        .map(|i| lm.create_edge([verts[i], verts[(i + 1) % n]], EdgeCreateFlags::NONE))
        .collect();
    let f = lm.face_create(&verts, &edges, FaceCreateFlags::NONE);
    (lm, f, verts)
}

fn loops_of_face(mesh: &LoopMesh, face: FaceKey) -> Vec<LoopKey> {
    let mut out = Vec::new();
    let mut it = mesh.face_loops(face);
    while let Some(l) = it.next(mesh) {
        out.push(l);
    }
    out
}

fn signed_area(coords: &[[f32; 2]]) -> f32 {
    let mut area = 0.0f32;
    let n = coords.len();
    for i in 0..n {
        let a = coords[i];
        let b = coords[(i + 1) % n];
        area += a[0] * b[1] - b[0] * a[1];
    }
    area * 0.5
}

/// Assert every input loop appears in at least one triangle.
fn assert_covers_all_loops(loops: &[LoopKey], tris: &[[LoopKey; 3]]) {
    for &expected in loops {
        let found = tris
            .iter()
            .any(|t| t[0] == expected || t[1] == expected || t[2] == expected);
        assert!(found, "loop {:?} missing from triangulation", expected);
    }
}

/// Assert every triangle has strictly positive area when its vertex
/// positions are pulled from the LoopMesh and projected to the XY
/// plane (these tests all use faces with normal along ±Z).
fn assert_all_triangles_nondegenerate(mesh: &LoopMesh, tris: &[[LoopKey; 3]]) {
    for &[a, b, c] in tris {
        let pa = mesh.vertex_position(mesh.loop_vertex(a));
        let pb = mesh.vertex_position(mesh.loop_vertex(b));
        let pc = mesh.vertex_position(mesh.loop_vertex(c));
        let area = signed_area(&[[pa.x, pa.y], [pb.x, pb.y], [pc.x, pc.y]]).abs();
        assert!(
            area > 1e-6,
            "triangle ({a:?}, {b:?}, {c:?}) is degenerate (|area| = {area})",
        );
    }
}

fn triangulate(mesh: &LoopMesh, face: FaceKey, policy: TriangulationPolicy) -> Vec<[LoopKey; 3]> {
    let mut tri_buf = Vec::new();
    let mut scratch_loops = Vec::new();
    let mut scratch_proj = Vec::new();
    corner_triangulation(
        mesh,
        face,
        policy,
        &mut scratch_loops,
        &mut scratch_proj,
        &mut tri_buf,
    );
    tri_buf
}

// ============================== Triangle ==============================

#[test]
fn triangle_passes_through_in_cycle_order() {
    let (lm, f, _verts) =
        build_single_face_mesh(&[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0, 0.0]]);
    let loops = loops_of_face(&lm, f);
    let tris = triangulate(&lm, f, TriangulationPolicy::EarClip);
    assert_eq!(tris.len(), 1);
    assert_eq!(tris[0], [loops[0], loops[1], loops[2]]);
}

// ============================== Quads ==============================

#[test]
fn convex_quad_uses_default_diagonal() {
    // Unit CCW quad in XY plane (normal = +Z). Convex → flip check
    // produces a *negative* dot of the two crosses, so the default
    // 0-2 diagonal is used.
    let (lm, f, _verts) = build_single_face_mesh(&[
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ]);
    let loops = loops_of_face(&lm, f);
    let tris = triangulate(&lm, f, TriangulationPolicy::EarClip);
    assert_eq!(tris.len(), 2);
    assert_eq!(tris[0], [loops[0], loops[1], loops[2]]);
    assert_eq!(tris[1], [loops[0], loops[2], loops[3]]);
    assert_covers_all_loops(&loops, &tris);
    assert_all_triangles_nondegenerate(&lm, &tris);
}

#[test]
fn concave_quad_rotates_diagonal_to_avoid_flip() {
    // CCW "shark fin": v1 and v3 both lie *above* the v0-v2 line
    // (which runs along the x-axis), so the 0-2 diagonal would cut
    // outside the polygon. The fast flip check (§5.1) must detect
    // this and rotate to the 1-3 diagonal. Output triples per §5.2
    // are (loop0, loop1, loop3) and (loop1, loop2, loop3).
    let (lm, f, _verts) = build_single_face_mesh(&[
        [0.0, 0.0, 0.0],
        [1.0, 0.5, 0.0],
        [2.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
    ]);
    let loops = loops_of_face(&lm, f);
    let tris = triangulate(&lm, f, TriangulationPolicy::EarClip);
    assert_eq!(tris.len(), 2);
    assert_eq!(
        tris[0],
        [loops[0], loops[1], loops[3]],
        "concave quad should rotate diagonal — got {tris:?}"
    );
    assert_eq!(tris[1], [loops[1], loops[2], loops[3]]);
    assert_covers_all_loops(&loops, &tris);
    assert_all_triangles_nondegenerate(&lm, &tris);
}

#[test]
fn fan_policy_preserves_old_behaviour_on_quad() {
    let (lm, f, _verts) = build_single_face_mesh(&[
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ]);
    let loops = loops_of_face(&lm, f);
    let tris = triangulate(&lm, f, TriangulationPolicy::Fan);
    assert_eq!(tris.len(), 2);
    assert_eq!(tris[0], [loops[0], loops[1], loops[2]]);
    assert_eq!(tris[1], [loops[0], loops[2], loops[3]]);
}

// ============================== N-gons ==============================

#[test]
fn convex_pentagon_emits_three_tris() {
    let (lm, f, _verts) = build_single_face_mesh(&[
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.5, 1.0, 0.0],
        [0.5, 1.6, 0.0],
        [-0.5, 1.0, 0.0],
    ]);
    let loops = loops_of_face(&lm, f);
    let tris = triangulate(&lm, f, TriangulationPolicy::EarClip);
    assert_eq!(tris.len(), 3);
    assert_covers_all_loops(&loops, &tris);
    assert_all_triangles_nondegenerate(&lm, &tris);
}

#[test]
fn concave_pentagon_emits_three_tris_no_overlap() {
    // CCW Pac-Man pentagon — vertex 2 is pushed inward enough to
    // create a strong reflex angle. Fan from v0 would slice through
    // empty space; ear-clip must respect the boundary.
    let (lm, f, _verts) = build_single_face_mesh(&[
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [2.0, 2.0, 0.0],
        [0.0, 2.0, 0.0],
    ]);
    let loops = loops_of_face(&lm, f);
    let tris = triangulate(&lm, f, TriangulationPolicy::EarClip);
    assert_eq!(tris.len(), 3);
    assert_covers_all_loops(&loops, &tris);
    assert_all_triangles_nondegenerate(&lm, &tris);
}

#[test]
fn regular_hexagon_emits_four_tris() {
    let n = 6;
    let positions: Vec<[f32; 3]> = (0..n)
        .map(|i| {
            let theta = (i as f32) * (std::f32::consts::TAU / n as f32);
            [theta.cos(), theta.sin(), 0.0]
        })
        .collect();
    let (lm, f, _verts) = build_single_face_mesh(&positions);
    let loops = loops_of_face(&lm, f);
    let tris = triangulate(&lm, f, TriangulationPolicy::EarClip);
    assert_eq!(tris.len(), 4);
    assert_covers_all_loops(&loops, &tris);
    assert_all_triangles_nondegenerate(&lm, &tris);
}

#[test]
fn regular_heptagon_emits_five_tris() {
    let n = 7;
    let positions: Vec<[f32; 3]> = (0..n)
        .map(|i| {
            let theta = (i as f32) * (std::f32::consts::TAU / n as f32);
            [theta.cos(), theta.sin(), 0.0]
        })
        .collect();
    let (lm, f, _verts) = build_single_face_mesh(&positions);
    let loops = loops_of_face(&lm, f);
    let tris = triangulate(&lm, f, TriangulationPolicy::EarClip);
    assert_eq!(tris.len(), 5);
    assert_covers_all_loops(&loops, &tris);
    assert_all_triangles_nondegenerate(&lm, &tris);
}

// ============================== End-to-end via to_nub_mesh ==============================

#[test]
fn to_nub_mesh_ear_clip_concave_pentagon_emits_three_indexed_tris() {
    let (mut lm, _f, _verts) = build_single_face_mesh(&[
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [2.0, 2.0, 0.0],
        [0.0, 2.0, 0.0],
    ]);
    let nm = lm
        .to_nub_mesh_with(&ToNubMeshOptions {
            seam_split: SeamSplitPolicy::Split,
            triangulation: TriangulationPolicy::EarClip,
            ..Default::default()
        })
        .expect("export");
    assert_eq!(nm.vertex_count(), 5);
    let idx = nm.indices().expect("indices present");
    // 3 triangles × 3 indices each = 9.
    assert_eq!(idx.count, 9);
}

#[test]
fn to_nub_mesh_fan_policy_concave_pentagon_still_emits_three_tris() {
    // Even though Fan produces geometrically incorrect tris on this
    // input, the index buffer still has the same shape — same number
    // of output indices. This test pins the contract that Fan remains
    // a valid emitter, just not a parity-correct one.
    let (mut lm, _f, _verts) = build_single_face_mesh(&[
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [2.0, 2.0, 0.0],
        [0.0, 2.0, 0.0],
    ]);
    let nm = lm
        .to_nub_mesh_with(&ToNubMeshOptions {
            seam_split: SeamSplitPolicy::Split,
            triangulation: TriangulationPolicy::Fan,
            ..Default::default()
        })
        .expect("export");
    assert_eq!(nm.vertex_count(), 5);
    let idx = nm.indices().expect("indices present");
    assert_eq!(idx.count, 9);
}

// ============================== LineList ==============================

/// Build the standard CCW unit quad with its four boundary edges.
fn build_unit_quad() -> LoopMesh {
    let (lm, _f, _verts) = build_single_face_mesh(&[
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ]);
    lm
}

#[test]
fn line_list_quad_emits_four_lines() {
    let mut lm = build_unit_quad();
    let nm = lm
        .to_nub_mesh_with(&ToNubMeshOptions {
            topology: PrimitiveTopology::LineList,
            ..Default::default()
        })
        .expect("export");
    assert_eq!(nm.topology(), PrimitiveTopology::LineList);
    assert_eq!(nm.vertex_count(), 4);
    let idx = nm.indices().expect("indices present");
    // One line per edge × 2 indices each = 8.
    assert_eq!(idx.count, 8);
}

// ============================== PointList ==============================

#[test]
fn point_list_quad_emits_four_points() {
    let mut lm = build_unit_quad();
    let nm = lm
        .to_nub_mesh_with(&ToNubMeshOptions {
            topology: PrimitiveTopology::PointList,
            ..Default::default()
        })
        .expect("export");
    assert_eq!(nm.topology(), PrimitiveTopology::PointList);
    assert_eq!(nm.vertex_count(), 4);
    let idx = nm.indices().expect("indices present");
    assert_eq!(idx.count, 4);
}

#[test]
fn line_list_two_quads_shared_edge_emits_seven_lines() {
    // Two quads sharing one edge: 7 edges total → 14 line indices.
    let mut lm = LoopMesh::new();
    let v0 = lm.create_vertex(Vec3::new(0.0, 0.0, 0.0));
    let v1 = lm.create_vertex(Vec3::new(1.0, 0.0, 0.0));
    let v2 = lm.create_vertex(Vec3::new(1.0, 1.0, 0.0));
    let v3 = lm.create_vertex(Vec3::new(0.0, 1.0, 0.0));
    let v4 = lm.create_vertex(Vec3::new(2.0, 0.0, 0.0));
    let v5 = lm.create_vertex(Vec3::new(2.0, 1.0, 0.0));
    let e01 = lm.create_edge([v0, v1], EdgeCreateFlags::NONE);
    let e12 = lm.create_edge([v1, v2], EdgeCreateFlags::NONE);
    let e23 = lm.create_edge([v2, v3], EdgeCreateFlags::NONE);
    let e30 = lm.create_edge([v3, v0], EdgeCreateFlags::NONE);
    let e14 = lm.create_edge([v1, v4], EdgeCreateFlags::NONE);
    let e45 = lm.create_edge([v4, v5], EdgeCreateFlags::NONE);
    let e52 = lm.create_edge([v5, v2], EdgeCreateFlags::NONE);
    lm.face_create(
        &[v0, v1, v2, v3],
        &[e01, e12, e23, e30],
        FaceCreateFlags::NONE,
    );
    lm.face_create(
        &[v1, v4, v5, v2],
        &[e14, e45, e52, e12],
        FaceCreateFlags::NONE,
    );

    let nm = lm
        .to_nub_mesh_with(&ToNubMeshOptions {
            topology: PrimitiveTopology::LineList,
            ..Default::default()
        })
        .expect("export");
    assert_eq!(nm.vertex_count(), 6);
    let idx = nm.indices().expect("indices present");
    assert_eq!(idx.count, 7 * 2);
}
