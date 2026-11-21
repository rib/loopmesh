use super::*;

/// The new cap face produced by beveling cube-corner vert 0 (interior
/// valence-3 → one triangular cap, the single new face). Located by the
/// result buffer.
fn beveled_corner_cap(mesh: &LoopMesh, result: &BevelResult) -> FaceKey {
    assert_eq!(
        result.faces.len(),
        1,
        "an interior valence-3 corner bevel creates exactly one cap face"
    );
    let _ = mesh;
    result.faces[0]
}

/// BEVEL.md §6.6 stage 2: `material >= 0` stamps that slot uniformly on the new cap
/// (over the block stage 1 inherited), while the surviving re-stitched faces keep
/// their own slot (§6.6.2).
#[test]
fn material_explicit_stamps_new_cap_and_leaves_survivors() {
    let mut mesh = LoopMesh::new();
    mesh_add_cube(&mut mesh);
    let v0 = vert_at(&mesh, Vec3::new(1.0, -1.0, -1.0));

    let result = mesh.bevel(
        &[v0.into()],
        BevelOptions {
            offset: 0.25,
            material: 3,
            ..Default::default()
        },
    );

    let cap = beveled_corner_cap(&mesh, &result);
    assert_eq!(
        mesh.face_material_index(cap),
        3,
        "explicit material=3 stamps the new cap (§6.6)"
    );

    // Every surviving (non-cap) face keeps the cube's default slot 0 — never
    // re-materialised (§6.6.2).
    for face in mesh.face_keys() {
        if face == cap {
            continue;
        }
        assert_eq!(
            mesh.face_material_index(face),
            0,
            "surviving face {face:?} keeps its own slot 0 (§6.6.2), not the stamp"
        );
    }
    mesh.validate_topology();
}

/// BEVEL.md §6.6.2 invariant: with `material >= 0` the multiset of NEW-face
/// material slots is `{material}` (one per new face) regardless of segment
/// count — pinned here for the single-cap case at a non-zero slot distinct
/// from the surviving faces.
#[test]
fn material_explicit_new_face_multiset_is_the_stamp() {
    let mut mesh = LoopMesh::new();
    mesh_add_cube(&mut mesh);
    let v0 = vert_at(&mesh, Vec3::new(1.0, -1.0, -1.0));

    let result = mesh.bevel(
        &[v0.into()],
        BevelOptions {
            offset: 0.25,
            material: 5,
            ..Default::default()
        },
    );

    let new_slots: Vec<i32> = result
        .faces
        .iter()
        .map(|&f| mesh.face_material_index(f))
        .collect();
    assert_eq!(new_slots, vec![5], "new faces all carry the stamped slot 5");
}

/// BEVEL.md §6.6.1 (stage 1) with no stage-2 stamp: at `material < 0` the cap's slot
/// is whatever it inherited from a representative *incident face of `v`* — never a
/// global default and never slot 0 unless an adjacent face genuinely uses slot 0.
/// Decorating every cube face with a distinct non-zero slot makes the donor
/// observable: the cap's slot is one of vert 0's incident faces' decorated slots.
#[test]
fn material_inherit_takes_adjacent_face_slot() {
    let mut mesh = LoopMesh::new();
    mesh_add_cube(&mut mesh);

    // Decorate the six faces with distinct non-zero slots {1..=6}.
    let faces: Vec<FaceKey> = mesh.face_keys();
    for (index, &face) in faces.iter().enumerate() {
        mesh.set_face_material_index(face, (index + 1) as i32);
    }

    let v0 = vert_at(&mesh, Vec3::new(1.0, -1.0, -1.0));
    // The slots of vert 0's incident faces, captured pristine — the §6.6.1
    // candidate representatives the cap may inherit from.
    let incident_slots: Vec<i32> = mesh
        .vert_face_keys(v0)
        .iter()
        .map(|&f| mesh.face_material_index(f))
        .collect();

    let result = mesh.bevel(
        &[v0.into()],
        BevelOptions {
            offset: 0.25,
            material: -1,
            ..Default::default()
        },
    );

    let cap = beveled_corner_cap(&mesh, &result);
    let cap_slot = mesh.face_material_index(cap);
    assert_ne!(
        cap_slot, 0,
        "the inherited cap slot is never the default 0 (§6.6.1)"
    );
    assert!(
        incident_slots.contains(&cap_slot),
        "the cap inherits a representative incident face of v's slot ({incident_slots:?}), got {cap_slot}"
    );
    mesh.validate_topology();
}

/// BEVEL.md §6.6: on a uniformly-materialed input the `-1` inherit yields a
/// uniformly-materialed result — the cap takes the one shared slot, so the
/// per-face tie-break (§6.6.1) is unobservable, the common case the scaffold
/// drives.
#[test]
fn material_inherit_uniform_input_yields_uniform_result() {
    let mut mesh = LoopMesh::new();
    mesh_add_cube(&mut mesh);
    for face in mesh.face_keys() {
        mesh.set_face_material_index(face, 4);
    }
    let v0 = vert_at(&mesh, Vec3::new(1.0, -1.0, -1.0));

    let result = mesh.bevel(
        &[v0.into()],
        BevelOptions {
            offset: 0.25,
            material: -1,
            ..Default::default()
        },
    );

    for face in mesh.face_keys() {
        assert_eq!(
            mesh.face_material_index(face),
            4,
            "uniform slot-4 input → uniform slot-4 result under -1 inherit (§6.6)"
        );
    }
    let _ = result;
    mesh.validate_topology();
}

/// The default `BevelOptions.material` is `-1` (no stage-2 stamp): a default-options
/// bevel of a uniformly slot-2 cube leaves the cap at the inherited slot 2,
/// confirming the no-stamp path is the default and matches the explicit `-1` path.
#[test]
fn material_default_is_inherit() {
    let mut mesh = LoopMesh::new();
    mesh_add_cube(&mut mesh);
    for face in mesh.face_keys() {
        mesh.set_face_material_index(face, 2);
    }
    let v0 = vert_at(&mesh, Vec3::new(1.0, -1.0, -1.0));

    let result = mesh.bevel(
        &[v0.into()],
        BevelOptions {
            offset: 0.25,
            ..Default::default()
        },
    );

    let cap = beveled_corner_cap(&mesh, &result);
    assert_eq!(
        mesh.face_material_index(cap),
        2,
        "default material is -1 (inherit), so the cap takes the adjacent slot 2"
    );
}

// ---- mark_seam / mark_sharp (BEVEL.md §6.7) ----
