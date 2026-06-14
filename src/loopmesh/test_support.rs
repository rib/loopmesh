//! Shared fixtures for the unit tests in the `loopmesh` child modules —
//! `euler` / `mid` / `ops` (operators) and `radial` / `normals` / `index`
//! (kernel). Compiled only under `#[cfg(test)]`.

use glam::Vec3;

use crate::loopmesh::*;

/// Build a single n-gon face from a ring of positions, creating the
/// perimeter verts and edges. Returns the face key.
pub(crate) fn create_face_from_verts(mesh: &mut LoopMesh, coords: &[Vec3]) -> FaceKey {
    let mut verts = vec![];
    for v in coords {
        verts.push(mesh.create_vertex(*v));
    }
    let mut edges = vec![];
    for i in 0..(verts.len() - 1) {
        edges.push(mesh.create_edge([verts[i], verts[i + 1]], EdgeCreateFlags::NONE));
    }
    edges.push(mesh.create_edge([verts[verts.len() - 1], verts[0]], EdgeCreateFlags::NONE));

    mesh.face_create(&verts, &edges, FaceCreateFlags::NONE)
}

/// Build a unit cube (8 verts, 12 edges, 6 quad faces). Used by tests that
/// need a closed manifold solid.
pub(crate) fn mesh_add_cube(mesh: &mut LoopMesh) {
    let v0 = mesh.create_vertex(Vec3::new(1.0, -1.0, -1.0));
    let v1 = mesh.create_vertex(Vec3::new(-1.0, -1.0, -1.0));
    let v2 = mesh.create_vertex(Vec3::new(-1.0, 1.0, -1.0));
    let v3 = mesh.create_vertex(Vec3::new(1.0, 1.0, -1.0));
    let v4 = mesh.create_vertex(Vec3::new(1.0, -1.0, 1.0));
    let v5 = mesh.create_vertex(Vec3::new(-1.0, -1.0, 1.0));
    let v6 = mesh.create_vertex(Vec3::new(-1.0, 1.0, 1.0));
    let v7 = mesh.create_vertex(Vec3::new(1.0, 1.0, 1.0));

    let e0 = mesh.create_edge([v0, v1], EdgeCreateFlags::NONE);
    let e1 = mesh.create_edge([v1, v2], EdgeCreateFlags::NONE);
    let e2 = mesh.create_edge([v2, v3], EdgeCreateFlags::NONE);
    let e3 = mesh.create_edge([v3, v0], EdgeCreateFlags::NONE);

    let e4 = mesh.create_edge([v4, v5], EdgeCreateFlags::NONE);
    let e5 = mesh.create_edge([v5, v6], EdgeCreateFlags::NONE);
    let e6 = mesh.create_edge([v6, v7], EdgeCreateFlags::NONE);
    let e7 = mesh.create_edge([v7, v4], EdgeCreateFlags::NONE);

    let e8 = mesh.create_edge([v0, v4], EdgeCreateFlags::NONE);
    let e9 = mesh.create_edge([v1, v5], EdgeCreateFlags::NONE);
    let e10 = mesh.create_edge([v2, v6], EdgeCreateFlags::NONE);
    let e11 = mesh.create_edge([v3, v7], EdgeCreateFlags::NONE);

    mesh.face_create(&[v0, v1, v2, v3], &[e0, e1, e2, e3], FaceCreateFlags::NONE); // Front
    mesh.face_create(&[v1, v5, v6, v2], &[e9, e5, e10, e1], FaceCreateFlags::NONE); // Right
    mesh.face_create(&[v5, v4, v7, v6], &[e4, e7, e6, e5], FaceCreateFlags::NONE); // Back
    mesh.face_create(&[v4, v0, v3, v7], &[e8, e3, e11, e7], FaceCreateFlags::NONE); // Left
    mesh.face_create(
        &[v3, v2, v6, v7],
        &[e2, e10, e6, e11],
        FaceCreateFlags::NONE,
    ); // Top
    mesh.face_create(&[v0, v4, v5, v1], &[e8, e4, e9, e0], FaceCreateFlags::NONE);
    // Bottom
}
