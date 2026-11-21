use super::*;

/// Bevel the interior valence-3 cube corner `(1,-1,-1)` at `offset = 0.25`
/// with the given strength mode, returning the mesh, the result, and the cap
/// face (`result.faces[0]`, the single triangular corner-cap).
fn beveled_cube_corner_strength(mode: BevelFaceStrengthMode) -> (LoopMesh, BevelResult, FaceKey) {
    let mut mesh = LoopMesh::new();
    mesh_add_cube(&mut mesh);
    let corner = vert_at(&mesh, Vec3::new(1.0, -1.0, -1.0));
    let result = mesh.bevel(
        &[corner.into()],
        BevelOptions {
            offset: 0.25,
            face_strength_mode: mode,
            ..Default::default()
        },
    );
    let cap = result.faces[0];
    (mesh, result, cap)
}

/// Histogram of the strength layer over every live face, or `None` when the
/// named layer was never created (the `None` mode).
fn strength_histogram(mesh: &LoopMesh) -> Option<std::collections::BTreeMap<i32, usize>> {
    let layer = mesh
        .face_attribs()
        .find(&AttributeKey::custom(FACE_STRENGTH_LAYER))?;
    let mut histogram = std::collections::BTreeMap::new();
    for face in mesh.face_keys() {
        let value = mesh
            .face_attribs()
            .get_i32(layer, face.as_usize())
            .expect("every face has a slot in the strength layer");
        *histogram.entry(value).or_insert(0) += 1;
    }
    Some(histogram)
}

fn strength_of(mesh: &LoopMesh, face: FaceKey) -> i32 {
    let layer = mesh
        .face_attribs()
        .find(&AttributeKey::custom(FACE_STRENGTH_LAYER))
        .expect("strength layer present");
    mesh.face_attribs()
        .get_i32(layer, face.as_usize())
        .expect("face has a strength slot")
}

/// BEVEL.md §8.10.1 `ALL`: every face is stamped — the corner-cap weak, and
/// all six originals (three reconstructed + three untouched) strong.
#[test]
fn face_strength_all_stamps_every_face_cube_corner() {
    let (mesh, _result, cap) = beveled_cube_corner_strength(BevelFaceStrengthMode::All);

    assert_eq!(strength_of(&mesh, cap), FACE_STRENGTH_WEAK, "cap is weak");
    let histogram = strength_histogram(&mesh).expect("layer created");
    assert_eq!(
        histogram,
        std::collections::BTreeMap::from([(FACE_STRENGTH_WEAK, 1), (FACE_STRENGTH_STRONG, 6)]),
        "ALL: one weak cap + six strong originals, every face stamped",
    );
    mesh.validate_topology();
}

/// BEVEL.md §8.10.1 `NEW`: only the newly-created cap is stamped (weak); the
/// reconstructed and untouched originals stay at the medium default (`0`).
#[test]
fn face_strength_new_stamps_only_new_faces_cube_corner() {
    let (mesh, _result, cap) = beveled_cube_corner_strength(BevelFaceStrengthMode::New);

    assert_eq!(strength_of(&mesh, cap), FACE_STRENGTH_WEAK, "cap is weak");
    let histogram = strength_histogram(&mesh).expect("layer created");
    assert_eq!(
        histogram,
        std::collections::BTreeMap::from([(FACE_STRENGTH_WEAK, 1), (FACE_STRENGTH_MEDIUM, 6)]),
        "NEW: only the cap stamped weak; six originals left at the medium default",
    );
    mesh.validate_topology();
}

/// BEVEL.md §8.10.1 `AFFECTED`: the cap (weak) plus the three reconstructed
/// originals (strong) are stamped; the three untouched originals stay at
/// default. Distinct from both `NEW` and `ALL`.
#[test]
fn face_strength_affected_stamps_reconstructed_not_untouched_cube_corner() {
    let (mesh, _result, cap) = beveled_cube_corner_strength(BevelFaceStrengthMode::Affected);

    assert_eq!(strength_of(&mesh, cap), FACE_STRENGTH_WEAK, "cap is weak");
    let histogram = strength_histogram(&mesh).expect("layer created");
    assert_eq!(
        histogram,
        std::collections::BTreeMap::from([
            (FACE_STRENGTH_WEAK, 1),
            (FACE_STRENGTH_MEDIUM, 3),
            (FACE_STRENGTH_STRONG, 3),
        ]),
        "AFFECTED: cap weak, three reconstructed strong, three untouched at default",
    );
    mesh.validate_topology();
}

/// BEVEL.md §8.10.1 `None`: the pass does not run — the named layer is never
/// created, so existing callers see no change.
#[test]
fn face_strength_none_writes_no_layer_cube_corner() {
    let (mesh, result, _cap) = beveled_cube_corner_strength(BevelFaceStrengthMode::None);

    assert_eq!(result.faces.len(), 1, "still a single cap built");
    assert!(
        strength_histogram(&mesh).is_none(),
        "None mode creates no strength layer at all",
    );
    mesh.validate_topology();
}

/// The three modes the A/B scaffolds cross-pin (`ALL` / `NEW` / `None`) each
/// produce a distinct captured layer — the axis is non-vacuous.
#[test]
fn face_strength_all_new_none_are_distinct_cube_corner() {
    let (all_mesh, ..) = beveled_cube_corner_strength(BevelFaceStrengthMode::All);
    let (new_mesh, ..) = beveled_cube_corner_strength(BevelFaceStrengthMode::New);
    let (none_mesh, ..) = beveled_cube_corner_strength(BevelFaceStrengthMode::None);

    let all = strength_histogram(&all_mesh);
    let new = strength_histogram(&new_mesh);
    let none = strength_histogram(&none_mesh);

    assert!(all.is_some() && new.is_some() && none.is_none());
    assert_ne!(all, new, "ALL and NEW stamp different layers");
}

// ---- §8.10.2 harden_normals shading post-pass ----

// The three cube-face normals incident to the corner `(1,-1,-1)`: front
// `z=-1 → -Z`, bottom `y=-1 → -Y`, right `x=1 → +X`.
#[allow(non_snake_case)]
fn CORNER_FRONT_N() -> Vec3 {
    Vec3::new(0.0, 0.0, -1.0)
}

#[allow(non_snake_case)]
fn CORNER_BOTTOM_N() -> Vec3 {
    Vec3::new(0.0, -1.0, 0.0)
}

#[allow(non_snake_case)]
fn CORNER_RIGHT_N() -> Vec3 {
    Vec3::new(1.0, 0.0, 0.0)
}

fn approx_eq(a: Vec3, b: Vec3) -> bool {
    (a - b).length() < 1.0e-5
}

fn count_matching(normals: &[Vec3], target: Vec3) -> usize {
    normals.iter().filter(|n| approx_eq(**n, target)).count()
}

fn hardened_loop_normals(mesh: &LoopMesh, face: FaceKey) -> Vec<Vec3> {
    mesh.face_loop_keys(face)
        .into_iter()
        .map(|l| {
            mesh.loop_hardened_normal(l)
                .expect("every loop of a new bevel face is hardened")
        })
        .collect()
}

/// BEVEL.md §8.10.2 chamfer-strip → reconstructed-original branch. Beveling the
/// isolated cube edge `{0,1}` (crease between front `-Z` and bottom `-Y`) writes
/// each strip loop the flat normal of the reconstructed flanking face across its
/// *long* edge — so the four strip loops carry two `-Z` and two `-Y`, matching
/// the original surface at each seam. This is emphatically **not** the strip's
/// own flat normal (`≈ (0,-1,-1)/√2`), which a naive flat-everywhere writeback
/// would stamp on all four.
#[test]
fn harden_edge_strip_matches_reconstructed_flanks() {
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
            harden_normals: true,
            ..Default::default()
        },
    );

    let strip = result.faces[0];
    let strip_flat = mesh.face_normal(strip);
    let normals = hardened_loop_normals(&mesh, strip);

    assert_eq!(normals.len(), 4, "strip quad has four loops");
    assert_eq!(
        count_matching(&normals, CORNER_FRONT_N()),
        2,
        "two strip loops match the reconstructed front face (-Z)",
    );
    assert_eq!(
        count_matching(&normals, CORNER_BOTTOM_N()),
        2,
        "two strip loops match the reconstructed bottom face (-Y)",
    );
    // The rule genuinely differs from a flat-everywhere writeback: no loop
    // carries the strip's own diagonal facing normal.
    for n in &normals {
        assert!(
            (*n - strip_flat).length() > 0.1,
            "hardened loop normal {n:?} must not be the strip's own flat normal {strip_flat:?}",
        );
    }
    mesh.validate_topology();
}

/// BEVEL.md §8.10.2 corner-cap → reconstructed-original branch. Beveling the
/// valence-3 cube corner `(1,-1,-1)` writes each of the triangular cap's three
/// loops the flat normal of the reconstructed cube face it seams against, so the
/// three loops carry exactly the three incident axis normals `{-Z, -Y, +X}` —
/// one per fan sector — never the cap's own diagonal flat normal `(1,-1,-1)/√3`.
#[test]
fn harden_corner_cap_matches_reconstructed_faces() {
    let mut mesh = LoopMesh::new();
    mesh_add_cube(&mut mesh);
    let corner = vert_at(&mesh, Vec3::new(1.0, -1.0, -1.0));

    let result = mesh.bevel(
        &[corner.into()],
        BevelOptions {
            offset: 0.25,
            harden_normals: true,
            ..Default::default()
        },
    );

    let cap = result.faces[0];
    let cap_flat = mesh.face_normal(cap);
    let normals = hardened_loop_normals(&mesh, cap);

    assert_eq!(normals.len(), 3, "triangular cap has three loops");
    // The three loops seam against the three distinct incident cube faces, so
    // their hardened normals are the three axis normals as a set — a bijection
    // onto the fan sectors, independent of the two-reconstructed tie-break.
    assert_eq!(count_matching(&normals, CORNER_FRONT_N()), 1, "one loop -Z");
    assert_eq!(
        count_matching(&normals, CORNER_BOTTOM_N()),
        1,
        "one loop -Y"
    );
    assert_eq!(count_matching(&normals, CORNER_RIGHT_N()), 1, "one loop +X");
    // None is the cap's own diagonal flat normal (the naive writeback).
    for n in &normals {
        assert!(
            (*n - cap_flat).length() > 0.1,
            "hardened cap loop {n:?} must not be the cap's own flat normal {cap_flat:?}",
        );
    }
    mesh.validate_topology();
}

/// BEVEL.md §8.10.2 pre-pass: the reconstructed-original ↔ new-bevel seam edges
/// (here the cap's three chamfer edges) are marked **sharp** before the
/// writeback. A cube edge untouched by the bevel stays smooth.
#[test]
fn harden_marks_reconstructed_seam_edges_sharp() {
    let mut mesh = LoopMesh::new();
    mesh_add_cube(&mut mesh);
    let corner = vert_at(&mesh, Vec3::new(1.0, -1.0, -1.0));

    let result = mesh.bevel(
        &[corner.into()],
        BevelOptions {
            offset: 0.25,
            harden_normals: true,
            ..Default::default()
        },
    );

    let cap = result.faces[0];
    for loop_key in mesh.face_loop_keys(cap) {
        let edge = mesh.loop_edge(loop_key);
        assert!(
            mesh.edge_sharp(edge),
            "cap chamfer edge {edge:?} is a reconstructed↔new seam and must be sharp",
        );
    }

    // An untouched cube edge (the far top crease) stays smooth.
    let untouched = cube_edge(&mesh, Vec3::new(-1.0, 1.0, 1.0), Vec3::new(1.0, 1.0, 1.0));
    assert!(
        !mesh.edge_sharp(untouched),
        "an edge the bevel never touched must not be marked sharp",
    );
    mesh.validate_topology();
}

/// BEVEL.md §8.10.2 gate + scope: `harden_normals = false` writes no per-loop
/// layer at all, and with it on only the *new* bevel faces' loops are written —
/// an untouched original face's loops stay unwritten.
#[test]
fn harden_gate_and_scope() {
    // Gate off: no layer, no writeback.
    let mut off = LoopMesh::new();
    mesh_add_cube(&mut off);
    let corner_off = vert_at(&off, Vec3::new(1.0, -1.0, -1.0));
    let result_off = off.bevel(
        &[corner_off.into()],
        BevelOptions {
            offset: 0.25,
            ..Default::default()
        },
    );
    let cap_off = result_off.faces[0];
    for loop_key in off.face_loop_keys(cap_off) {
        assert!(
            off.loop_hardened_normal(loop_key).is_none(),
            "harden_normals off writes no per-loop normal",
        );
    }

    // Gate on: new-face loops written, untouched-original loops untouched.
    let mut on = LoopMesh::new();
    mesh_add_cube(&mut on);
    on.calc_face_normals();
    let top_face_before = on
        .face_keys()
        .into_iter()
        .find(|&f| approx_eq(on.face_normal(f), Vec3::new(0.0, 0.0, 1.0)))
        .expect("cube top face +Z");
    let corner_on = vert_at(&on, Vec3::new(1.0, -1.0, -1.0));
    let result_on = on.bevel(
        &[corner_on.into()],
        BevelOptions {
            offset: 0.25,
            harden_normals: true,
            ..Default::default()
        },
    );
    let cap_on = result_on.faces[0];
    for loop_key in on.face_loop_keys(cap_on) {
        assert!(
            on.loop_hardened_normal(loop_key).is_some(),
            "a new bevel face's loop is hardened",
        );
    }
    // The +Z top face is untouched by a bevel of the -Z/-Y/+X corner.
    for loop_key in on.face_loop_keys(top_face_before) {
        assert!(
            on.loop_hardened_normal(loop_key).is_none(),
            "an untouched original face's loop is never hardened",
        );
    }
}

/// BEVEL.md §8.10.2 leaf independence: `harden_normals` and `face_strength_mode`
/// touch disjoint customdata (per-loop normals vs a per-face integer) and both
/// take effect when requested together.
#[test]
fn harden_independent_of_face_strength_mode() {
    let mut mesh = LoopMesh::new();
    mesh_add_cube(&mut mesh);
    let corner = vert_at(&mesh, Vec3::new(1.0, -1.0, -1.0));

    let result = mesh.bevel(
        &[corner.into()],
        BevelOptions {
            offset: 0.25,
            harden_normals: true,
            face_strength_mode: BevelFaceStrengthMode::New,
            ..Default::default()
        },
    );

    let cap = result.faces[0];
    // The face-strength pass ran: the cap is weak.
    assert_eq!(strength_of(&mesh, cap), FACE_STRENGTH_WEAK);
    // The harden pass ran: the cap's loops carry the incident cube-face normals.
    let normals = hardened_loop_normals(&mesh, cap);
    assert_eq!(count_matching(&normals, CORNER_FRONT_N()), 1);
    assert_eq!(count_matching(&normals, CORNER_BOTTOM_N()), 1);
    assert_eq!(count_matching(&normals, CORNER_RIGHT_N()), 1);
    mesh.validate_topology();
}
