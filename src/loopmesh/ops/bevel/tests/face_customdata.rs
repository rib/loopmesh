use super::*;

/// Which of the cube's six axis-aligned planes every vert of `face` sits in,
/// if any. A rebuilt original face keeps this signature — its new corners are
/// ring / strip verts slid *along the cube's own edges*, which lie in the same
/// planes — while every face the bevel mints (cap, chamfer strip) spans a
/// corner or an edge and so belongs to none of them.
fn cube_plane_of(mesh: &LoopMesh, face: FaceKey) -> Option<usize> {
    let verts = mesh.face_vert_keys(face);
    (0..6).find(|&plane| {
        let axis = plane / 2;
        let sign = if plane % 2 == 0 { 1.0 } else { -1.0 };
        verts
            .iter()
            .all(|&v| (mesh.vertex_position(v)[axis] - sign).abs() < 1.0e-4)
    })
}

fn plane_tint(plane: usize) -> [f32; 3] {
    [plane as f32 + 1.0, 0.25, 0.5]
}

/// Both restitch paths (BEVEL.md §6.1's vertex-bevel rebuild and §7.4's
/// consumed-endpoint rebuild) kill each affected face and re-create it over its
/// new cycle. The replacement routinely lands on the dead face's *recycled* slab
/// key, whose attribute block is default-stamped at mint — so a face's own
/// customdata survives only because the block is captured before the kill and
/// written back after. Tint every cube face by its plane, bevel, and require
/// every surviving original to still read its own tint.
fn assert_bevel_carries_face_cd(affect: BevelAffect, geom: impl Fn(&LoopMesh) -> Vec<GeomKey>) {
    let mut mesh = LoopMesh::new();
    mesh_add_cube(&mut mesh);
    let tint = mesh
        .register_face_attribute(ElementAttributeDesc::numeric(
            AttributeKey::custom("f_tint"),
            AttrType::Vec3,
        ))
        .unwrap();
    for face in mesh.face_keys() {
        let plane = cube_plane_of(&mesh, face).expect("a cube face lies in an axis plane");
        mesh.face_attribs_mut()
            .set_vec3(tint, face.as_usize(), plane_tint(plane));
    }
    let faces_before = mesh.face_count();

    let geom = geom(&mesh);
    let result = mesh.bevel(
        &geom,
        BevelOptions {
            offset: 0.25,
            affect,
            ..Default::default()
        },
    );
    mesh.validate_topology();
    assert!(!result.faces.is_empty(), "the bevel built no new faces");
    assert!(mesh.face_count() > faces_before);

    // The rebuild really did recycle: it killed faces and the replacements
    // landed back in the freed keys, leaving the key space dense. Without a
    // recycle every restored face would be a fresh key whose block no mint had
    // stamped, and the assertions below would pass even with no restore at all.
    let live: Vec<usize> = mesh.face_keys().iter().map(|f| f.as_usize()).collect();
    assert_eq!(
        live.iter().max().copied().expect("faces exist") + 1,
        live.len(),
        "face keys are not dense — no killed face's key was recycled, so this \
         test cannot observe the restore it exists to pin",
    );

    // Only the survivors are pinned here: a *new* bevel face lies in no cube plane
    // and inherits its representative original's block instead (§6.6.1), which the
    // `face_cd_*_inherits_*` tests below pin.
    for face in mesh.face_keys() {
        let Some(plane) = cube_plane_of(&mesh, face) else {
            continue;
        };
        assert_eq!(
            mesh.face_attribs().get_vec3(tint, face.as_usize()),
            Some(plane_tint(plane)),
            "face {face:?} lost its per-face customdata across the rebuild",
        );
    }
}

#[test]
fn vertex_bevel_carries_per_face_customdata_across_rebuild() {
    assert_bevel_carries_face_cd(BevelAffect::Vertices, |mesh| {
        vec![vert_at(mesh, Vec3::new(1.0, 1.0, 1.0)).into()]
    });
}

#[test]
fn edge_bevel_carries_per_face_customdata_across_restitch() {
    assert_bevel_carries_face_cd(BevelAffect::Edges, |mesh| {
        let corner = vert_at(mesh, Vec3::new(1.0, 1.0, 1.0));
        vec![mesh.vert_edge_keys(corner)[0].into()]
    });
}

// ---- Per-face customdata on the NEW faces (§6.6.1's whole-block inherit) ----

/// Decorate every face with a per-face tint layer *and* a distinct material slot, so
/// a new face's inherit can be read as a coherent `(slot, tint)` pair drawn from one
/// single donor face rather than assembled field-by-field. Slots start at **1**, so
/// the layer default `(0, [0, 0, 0])` — what a new face lands on when its donor is
/// lost — is no donor's block and cannot pass a membership assertion.
fn tint_and_slot_faces(mesh: &mut LoopMesh) -> ElementAttributeKey {
    let tint = mesh
        .register_face_attribute(ElementAttributeDesc::numeric(
            AttributeKey::custom("f_tint"),
            AttrType::Vec3,
        ))
        .unwrap();
    for (index, face) in mesh.face_keys().into_iter().enumerate() {
        let slot = index as i32 + 1;
        let value = slot as f32;
        mesh.face_attribs_mut().set_vec3(
            tint,
            face.as_usize(),
            [value, 10.0 * value, 100.0 * value],
        );
        mesh.set_face_material_index(face, slot);
    }
    tint
}

/// Each face's per-face block read as one coherent `(material slot, tint)` value.
fn face_blocks(
    mesh: &LoopMesh,
    tint: ElementAttributeKey,
    faces: &[FaceKey],
) -> Vec<(i32, [f32; 3])> {
    faces
        .iter()
        .map(|&face| {
            (
                mesh.face_material_index(face),
                mesh.face_attribs().get_vec3(tint, face.as_usize()).unwrap(),
            )
        })
        .collect()
}

/// BEVEL.md §6.6.1: every face the bevel **created** carries one donor's *coherent*
/// whole block — its slot and its custom layer from the same original face — never
/// the layer default and never a field-by-field mix. Which donor wins is
/// disk-anchor-dependent (§6.6.1.2), so donor **membership** is what is pinned, not
/// equality.
fn assert_new_faces_inherit_from(
    mesh: &LoopMesh,
    tint: ElementAttributeKey,
    new_faces: &[FaceKey],
    donors: &[(i32, [f32; 3])],
    site: &str,
) {
    assert!(!new_faces.is_empty(), "{site}: the bevel built no new face");
    for (&face, inherited) in new_faces.iter().zip(face_blocks(mesh, tint, new_faces)) {
        assert!(
            donors.contains(&inherited),
            "{site}: new face {face:?} carries {inherited:?}, which is no donor's whole \
             block {donors:?} — the layer default is (0, [0.0, 0.0, 0.0])",
        );
    }
}

/// The per-face block a **template-less** mint lands on (BEVEL.md §6.6.1.3): the layer
/// defaults, material slot `0` included. `tint_and_slot_faces` starts its slots at 1,
/// so no original carries this block and it is the unambiguous fingerprint of a face
/// the fill handed no donor.
const TEMPLATE_LESS_BLOCK: (i32, [f32; 3]) = (0, [0.0, 0.0, 0.0]);

/// Every new face carries either one donor's **whole** block (§6.6.1, stage 1) or —
/// for the fills that supply no donor at all (§6.6.1.3) — the layer defaults; exactly
/// `template_less` of them are the latter. Which donor wins is disk-anchor-dependent
/// (§6.6.1.2), so donor **membership** is what is pinned, not equality.
fn assert_new_face_blocks(
    mesh: &LoopMesh,
    tint: ElementAttributeKey,
    new_faces: &[FaceKey],
    donors: &[(i32, [f32; 3])],
    template_less: usize,
    site: &str,
) {
    assert!(!new_faces.is_empty(), "{site}: the bevel built no new face");
    let mut defaults = 0;
    for (&face, block) in new_faces.iter().zip(face_blocks(mesh, tint, new_faces)) {
        if block == TEMPLATE_LESS_BLOCK {
            defaults += 1;
            continue;
        }
        assert!(
            donors.contains(&block),
            "{site}: new face {face:?} carries {block:?}, which is neither a donor's whole \
             block {donors:?} nor the layer default {TEMPLATE_LESS_BLOCK:?}",
        );
    }
    assert_eq!(
        defaults, template_less,
        "{site}: {defaults} new face(s) landed on the layer default, expected \
         {template_less} template-less mint(s) (§6.6.1.3)",
    );
}

/// The beveled solid is still closed. A junction whose fill face never got built reads
/// here as a hole, so this is what keeps the donor assertions non-vacuous at the fill
/// sites: it proves the fill faces are in `result.faces` at all.
fn assert_no_boundary_edges(mesh: &LoopMesh, site: &str) {
    let open = mesh
        .edge_keys()
        .iter()
        .filter(|&&edge| mesh.edge_radial_iter(edge).count() != 2)
        .count();
    assert_eq!(
        open, 0,
        "{site}: {open} boundary edge(s) — the corner is unfilled"
    );
}

/// The three cube edges meeting at the `(1, 1, 1)` corner, as bevel input, plus the
/// coherent blocks of the three original faces meeting there — the donor set every
/// new face of that junction (strip, patch, or fill) must draw from (§6.6.1).
fn cube_corner_junction(
    mesh: &LoopMesh,
    tint: ElementAttributeKey,
) -> (Vec<GeomKey>, Vec<(i32, [f32; 3])>) {
    let corner = vert_at(mesh, Vec3::new(1.0, 1.0, 1.0));
    let donors = face_blocks(mesh, tint, &mesh.vert_face_keys(corner));
    assert_eq!(donors.len(), 3, "a cube corner has three incident faces");
    let spokes = mesh
        .vert_edge_keys(corner)
        .into_iter()
        .map(GeomKey::from)
        .collect();
    (spokes, donors)
}

/// BEVEL.md §6.6.1: the vertex-bevel **cap** inherits the *whole* per-face attribute
/// block of a representative incident face of `v` — not the material slot alone. The
/// inherit is stage 1 and runs at every `material`; `material < 0` is used here only
/// so no stage-2 stamp masks the inherited slot. Which incident face wins is
/// disk-anchor-dependent
/// (§6.6.1.2), so what is pinned is the contract: the cap's block is *some* incident
/// original's, taken coherently (slot and custom layer from the same donor), and
/// never the layer default.
#[test]
fn face_cd_cap_inherits_whole_block_of_a_representative_incident_face() {
    let mut mesh = LoopMesh::new();
    mesh_add_cube(&mut mesh);
    let tint = tint_and_slot_faces(&mut mesh);

    let corner = vert_at(&mesh, Vec3::new(1.0, 1.0, 1.0));
    let candidates: Vec<(i32, [f32; 3])> = mesh
        .vert_face_keys(corner)
        .iter()
        .map(|&face| {
            (
                mesh.face_material_index(face),
                mesh.face_attribs().get_vec3(tint, face.as_usize()).unwrap(),
            )
        })
        .collect();

    let result = mesh.bevel(
        &[corner.into()],
        BevelOptions {
            offset: 0.25,
            material: -1,
            ..Default::default()
        },
    );
    let cap = *result
        .faces
        .iter()
        .find(|&&face| mesh.face_vert_keys(face).len() == 3)
        .expect("the closed-fan corner is capped by one triangle");

    let inherited = (
        mesh.face_material_index(cap),
        mesh.face_attribs().get_vec3(tint, cap.as_usize()).unwrap(),
    );
    assert!(
        candidates.contains(&inherited),
        "the cap must carry one incident original's whole block {candidates:?}, got \
         {inherited:?} (the layer default would be (0, [0,0,0]))",
    );
}

// ---- The interpolation-seam override on the ring-spanning fills (§6.6.1.4) ----

/// The half-disc fan's `k`-th triangle, found by its rim pair rather than by creation
/// order, so a decoration keyed on it says nothing about how the mesh was built.
fn open_fan_face(mesh: &LoopMesh, centre: VertexKey, rim: &[VertexKey], k: usize) -> FaceKey {
    mesh.vert_face_keys(centre)
        .into_iter()
        .find(|&face| {
            let verts = mesh.face_vert_keys(face);
            verts.contains(&rim[k]) && verts.contains(&rim[k + 1])
        })
        .expect("the fan's k-th triangle")
}

/// BEVEL.md §6.6.1.4 condition (1): a leg whose fan walk resolves **no arriving face or
/// no departing face** is an interpolation seam *unconditionally* — no customdata layer
/// is consulted, so it fires on the plainest untextured mesh — and every mesh-boundary
/// leg satisfies it. The open fan's apex therefore does **not** take §6.6.1's positional
/// donor for its ring-spanning terminal fill: it takes the tie-break winner, which here
/// (no UV layer ⇒ key 1 inert; nothing selected ⇒ key 2 tied) is the **lowest material
/// slot** among the candidates.
///
/// The candidate set is every leg's own reduce, so the winner can be — and at some
/// rotations is — an **end face of the fan that does not flank the beveled spoke at
/// all**. Rotating the decoration moves the winner across all three faces, which is
/// what no positional rule (and no beveled-leg-flanks-only rule) can follow.
#[test]
fn face_cd_open_fan_apex_fill_takes_the_seam_tie_break_donor_not_the_positional_one() {
    const SLOTS: [i32; 3] = [1, 5, 3];

    for rotation in 0..3 {
        let (mut mesh, centre, rim) = open_planar_fan(4, 0);
        for k in 0..3 {
            let face = open_fan_face(&mesh, centre, &rim, k);
            mesh.set_face_material_index(face, SLOTS[(k + rotation) % 3]);
        }
        let beveled = mesh
            .query_edge_exists(centre, rim[2])
            .expect("an interior spoke of the open fan");

        let result = mesh.bevel(
            &[beveled.into()],
            BevelOptions {
                offset: 0.2,
                affect: BevelAffect::Edges,
                ..Default::default()
            },
        );
        mesh.validate_topology();

        let fill = *result
            .faces
            .iter()
            .find(|&&face| mesh.face_vert_keys(face).len() == 3)
            .expect("the valence-4 apex mints one triangular terminal fill");
        assert_eq!(
            mesh.face_material_index(fill),
            1,
            "rotation {rotation}: the apex's two boundary legs seam it, so the fill's \
             donor is the tie-break's lowest-slot incident face — not the positional pick",
        );
    }
}

/// The vertex-bevel **cap** at a boundary corner is a ring-spanning fill too, and the
/// corner's two boundary legs seam it (§6.6.1.4 condition (1)), so its donor is the
/// tie-break winner over *every* leg's reduce — the lowest-slot incident face. Swapping
/// the two quads' slots moves the donor from one to the other; a positional pick would
/// stay on the same quad.
#[test]
fn face_cd_boundary_corner_cap_takes_the_seam_tie_break_donor_not_the_positional_one() {
    for swapped in [false, true] {
        let (mut mesh, _shared) = two_quad_strip();
        let corner = vert_at(&mesh, Vec3::new(0.0, -1.0, 0.0));
        let left = vert_at(&mesh, Vec3::new(-2.0, -1.0, 0.0));
        for face in mesh.vert_face_keys(corner) {
            let is_left = mesh.face_vert_keys(face).contains(&left);
            let slot = match (is_left, swapped) {
                (true, false) | (false, true) => 5,
                _ => 2,
            };
            mesh.set_face_material_index(face, slot);
        }

        let result = mesh.bevel(
            &[corner.into()],
            BevelOptions {
                offset: 0.2,
                ..Default::default()
            },
        );
        mesh.validate_topology();

        assert_eq!(result.faces.len(), 1, "the corner mints exactly its cap");
        assert_eq!(
            mesh.face_material_index(result.faces[0]),
            2,
            "swapped = {swapped}: the boundary corner is seamed, so the cap's donor is \
             the lower-slot quad whichever side it sits on",
        );
    }
}

/// The seam test of §6.6.1.4 condition (1) is about the **fan walk**, not about the
/// mesh boundary: a **wire** leg resolves neither an arriving nor a departing face, so
/// it seams a corner whose face fan is otherwise closed and fully interior. The cap's
/// donor there is the tie-break winner (lowest slot), not the positional pick — which
/// an implementation that shortcut condition (1) to "is this corner on the mesh
/// boundary?" would get wrong.
#[test]
fn face_cd_wire_leg_seams_a_closed_fan_and_routes_the_cap_through_the_tie_break() {
    let (mut mesh, centre, _rim) = planar_fan(4);
    let incident = mesh.vert_face_keys(centre);
    for (index, &face) in incident.iter().enumerate() {
        mesh.set_face_material_index(face, [7, 1, 4, 6][index]);
    }
    let tip = mesh.create_vertex(Vec3::new(0.0, 0.0, 1.0));
    mesh.create_edge([centre, tip], EdgeCreateFlags::NONE);

    let result = mesh.bevel(
        &[centre.into()],
        BevelOptions {
            offset: 0.2,
            ..Default::default()
        },
    );
    mesh.validate_topology();

    assert_eq!(
        result.faces.len(),
        1,
        "the wired corner mints exactly its cap"
    );
    assert_eq!(
        mesh.face_material_index(result.faces[0]),
        1,
        "the wire leg seams an interior closed fan, so the cap takes the tie-break's \
         lowest-slot incident face and not the positional pick (slot 7)",
    );
}

/// The converse, and the guard against applying the override wholesale: a **seam-free**
/// interior corner keeps §6.6.1's positional donor (§6.6.1.2's fan-anchor pick). The
/// decoration puts the lowest slot somewhere other than the positional face, so a cap
/// that ran the tie-break here would come out at slot 1 instead.
#[test]
fn face_cd_seam_free_interior_corner_keeps_the_positional_donor() {
    let mut mesh = LoopMesh::new();
    mesh_add_cube(&mut mesh);
    let corner = vert_at(&mesh, Vec3::new(1.0, 1.0, 1.0));
    let incident = mesh.vert_face_keys(corner);
    for (index, &face) in incident.iter().enumerate() {
        mesh.set_face_material_index(face, [7, 1, 4][index]);
    }

    let result = mesh.bevel(
        &[corner.into()],
        BevelOptions {
            offset: 0.25,
            ..Default::default()
        },
    );
    mesh.validate_topology();

    let cap = *result
        .faces
        .iter()
        .find(|&&face| mesh.face_vert_keys(face).len() == 3)
        .expect("the closed-fan corner is capped by one triangle");
    assert_eq!(
        mesh.face_material_index(cap),
        7,
        "no leg of a cube corner is a seam, so the cap keeps the positional donor",
    );
}

/// A **fully interior** valence-4 closed-fan corner seamed by §6.6.1.4 **condition (2)**:
/// one incident quad is painted apart on a blendable loop layer (a colour), so the two
/// edges bordering it disagree on that layer and are interpolation seams — even though
/// every leg resolves both flanking faces, so condition (1) never fires. The cap is a
/// ring-spanning fill, so a seam routes its donor through the §6.6.1.4 tie-break, which
/// on this float2-free, unselected mesh reduces (key 1 inert, key 2 tied) to the
/// **lowest incident material slot** over *all four* incident quads — not the positional
/// fan-anchor face, which is decorated away from that slot so the two picks separate.
#[test]
fn face_cd_interior_grid_corner_condition_2_seam_routes_cap_through_tie_break() {
    // Discover which incident quad the seam-free positional rule would hand the cap, by
    // decorating with identity slots — so the seamed layout below can place the global
    // minimum on a *different* quad and the two rules cannot agree by coincidence.
    let positional_face = {
        let mut mesh = LoopMesh::new();
        let (centre, _) = grid_3x3(&mut mesh, Vec3::ZERO);
        for (index, face) in mesh.face_keys().into_iter().enumerate() {
            mesh.set_face_material_index(face, index as i32);
        }
        let result = mesh.bevel(
            &[centre.into()],
            BevelOptions {
                offset: 0.2,
                ..Default::default()
            },
        );
        mesh.face_material_index(result.faces[0]) as usize
    };

    // Slots with a unique global minimum of 1, kept off the positional face.
    let mut slots = [7_i32, 5, 9, 3];
    let min_face = (positional_face + 1) % 4;
    slots[min_face] = 1;
    assert_ne!(
        slots[positional_face], 1,
        "the low slot must sit off the positional face for the test to separate the rules",
    );

    let mut mesh = LoopMesh::new();
    let (centre, _) = grid_3x3(&mut mesh, Vec3::ZERO);
    let faces = mesh.face_keys();
    for (&face, &slot) in faces.iter().zip(&slots) {
        mesh.set_face_material_index(face, slot);
    }
    // Paint the positional face's quad apart on a colour layer — a blendable, non-float2
    // discontinuity, so key 1 stays inert and only the seam routing is exercised.
    let color = mesh
        .register_loop_attribute(ElementAttributeDesc::color(
            AttributeKey::custom("color"),
            AttrType::Vec4,
        ))
        .expect("float RGBA is a supported loop attribute format");
    for &face in &faces {
        let value = if face == faces[positional_face] {
            [1.0, 0.0, 0.0, 1.0]
        } else {
            [0.0, 0.0, 0.0, 1.0]
        };
        for loop_key in mesh.face_loop_keys(face) {
            mesh.loop_attribs_mut()
                .set_vec4(color, loop_key.as_usize(), value);
        }
    }

    let result = mesh.bevel(
        &[centre.into()],
        BevelOptions {
            offset: 0.2,
            ..Default::default()
        },
    );
    mesh.validate_topology();

    assert_eq!(
        result.faces.len(),
        1,
        "the interior valence-4 corner mints exactly its cap",
    );
    assert_eq!(
        mesh.face_material_index(result.faces[0]),
        1,
        "the colour discontinuity seams the interior corner by condition (2), so the cap's \
         donor is the tie-break's lowest incident slot — not the positional pick (slot {})",
        slots[positional_face],
    );
}

/// BEVEL.md §6.6.1.4's candidate-set construction has two arms, and this pins the one a
/// naive "every incident leg" implementation gets wrong: with **more than one beveled
/// leg** at the corner, only the **beveled** legs contribute a candidate, so a face that
/// flanks no beveled leg is genuinely excluded — even when it carries the corner's
/// globally lowest slot. (§6.6.1.4's "where the collapse stops": the global-argmin
/// corollary holds on the `all_legs` arm only.)
///
/// The fixture is an open fan's boundary apex — seamed by condition (1), so the override
/// fires — with **three** interior spokes beveled. Three is the smallest beveled-leg
/// count that exhibits the arm at all: at two the junction ring carries two boundverts
/// and welds (§8.3.0.3), minting no ring-spanning fill whose donor could be read.
#[test]
fn face_cd_boundary_junction_tie_break_ranges_over_the_beveled_legs_only() {
    // Fan faces `f_k` span `rim[k] → rim[k+1]`; legs `L_0` and `L_5` are the boundary
    // ends. Beveling `L_1..L_3` leaves `f_4` flanking unbeveled legs only, and `f_4`
    // holds the corner's lowest slot.
    const SLOTS: [i32; 5] = [3, 5, 4, 6, 0];

    let (mut mesh, centre, rim) = open_planar_fan(6, 0);
    for k in 0..5 {
        let face = open_fan_face(&mesh, centre, &rim, k);
        mesh.set_face_material_index(face, SLOTS[k]);
    }
    let beveled: Vec<GeomKey> = (1..=3)
        .map(|k| {
            mesh.query_edge_exists(centre, rim[k])
                .expect("an interior spoke of the open fan")
                .into()
        })
        .collect();

    let result = mesh.bevel(
        &beveled,
        BevelOptions {
            offset: 0.2,
            affect: BevelAffect::Edges,
            ..Default::default()
        },
    );
    mesh.validate_topology();

    let patch = *result
        .faces
        .iter()
        .find(|&&face| mesh.face_vert_keys(face).len() == 3)
        .expect("a three-boundvert junction ring is capped by one patch triangle");
    assert_eq!(
        mesh.face_material_index(patch),
        3,
        "the candidate set is the three beveled legs' reduces — slots 3 and 4 — so the \
         patch takes slot 3; slot 0 belongs to the face flanking only unbeveled legs and \
         must not donate",
    );
}

/// The **fourth** ring-spanning fill family of §6.6.1: the odd-`segments` [V-ADJ] centre
/// n-gon (§8.5.5.5(e)). Its corners span the whole boundvert ring, so — unlike every
/// other face of the grid, which is per-sector — it goes through §6.6.1.4's tie-break at
/// a seam-carrying junction instead of taking the positional rep.
///
/// The decoration is chosen against the corner's *own* positional face, so the assertion
/// separates all three answers: the positional pick (highest slot), the beveled-legs
/// tie-break (the winner), and the all-legs tie-break (the unbeveled-only face's slot 0).
#[test]
fn face_cd_boundary_junction_adj_centre_ngon_takes_the_seam_tie_break_donor() {
    let (mut mesh, centre, rim) = open_planar_fan(6, 0);

    // The face §6.6.1's positional rule would hand the fill — decorated *away* from the
    // tie-break winner so the two picks cannot be confused.
    let positional = mesh.vert_face_keys(centre)[0];
    for k in 0..5 {
        let face = open_fan_face(&mesh, centre, &rim, k);
        let slot = if face == positional {
            9
        } else if k == 4 {
            0 // flanks unbeveled legs only: a candidate on the all-legs arm alone
        } else {
            (k as i32) + 1
        };
        mesh.set_face_material_index(face, slot);
    }
    let expected = (1..=3)
        .flat_map(|k| {
            let leg = mesh
                .query_edge_exists(centre, rim[k])
                .expect("an interior spoke of the open fan");
            mesh.edge_face_keys(leg)
        })
        .map(|face| mesh.face_material_index(face))
        .min()
        .expect("the beveled legs have flanking faces");

    let beveled: Vec<GeomKey> = (1..=3)
        .map(|k| mesh.query_edge_exists(centre, rim[k]).unwrap().into())
        .collect();
    let result = mesh.bevel(
        &beveled,
        BevelOptions {
            offset: 0.2,
            segments: 3,
            affect: BevelAffect::Edges,
            vmesh_method: BevelVmeshMethod::Adj,
            ..Default::default()
        },
    );
    mesh.validate_topology();

    let centre_ngon: Vec<FaceKey> = result
        .faces
        .iter()
        .copied()
        .filter(|&face| mesh.face_vert_keys(face).len() == 3)
        .collect();
    assert_eq!(
        centre_ngon.len(),
        1,
        "a three-sector ring at odd `segments` closes on exactly one centre triangle",
    );
    assert_eq!(
        mesh.face_material_index(centre_ngon[0]),
        expected,
        "the centre n-gon is ring-spanning, so it takes the tie-break winner over the \
         beveled legs — not the positional rep (slot 9) the rest of the grid carries, and \
         not the unbeveled-only face (slot 0)",
    );
}

/// A two-quad plane carrying a UV layer, its two quads decorated with slots 5 / 2 (the
/// `swapped` run exchanges them) and unwrapped either as **one** island (`islands = 1`:
/// UVs read straight off the positions, so both quads agree along the shared edge) or as
/// **two** (`islands = 2`: the right quad's `u` is shifted, so the shared edge is a
/// UV-island boundary). Every face's UV image has real area either way — the `degenerate_uv_image`
/// filter of §6.6.1.4 is not what these tests are about.
fn uv_decorated_two_quad_plane(islands: u32, swapped: bool) -> (LoopMesh, VertexKey) {
    let (mut mesh, _shared) = two_quad_strip();
    let uv = mesh
        .register_loop_attribute(ElementAttributeDesc::uv(AttributeKey::custom("uv")))
        .expect("Vec2 is a supported loop attribute format");

    let corner = vert_at(&mesh, Vec3::new(0.0, -1.0, 0.0));
    let left = vert_at(&mesh, Vec3::new(-2.0, -1.0, 0.0));
    for face in mesh.face_keys() {
        let is_left = mesh.face_vert_keys(face).contains(&left);
        let slot = match (is_left, swapped) {
            (true, false) | (false, true) => 5,
            _ => 2,
        };
        mesh.set_face_material_index(face, slot);

        let shift = if islands == 2 && !is_left { 0.5 } else { 0.0 };
        for loop_key in mesh.face_loop_keys(face) {
            let position = mesh.vertex_position(mesh.loop_vertex(loop_key));
            let value = [(position.x + 2.0) / 4.0 + shift, (position.y + 1.0) / 2.0];
            mesh.loop_attribs_mut()
                .set_vec2(uv, loop_key.as_usize(), value);
        }
    }
    (mesh, corner)
}

/// Key 1 of §6.6.1.4's tie-break — the interpolation-component id (§6.6.1.4.1) — being
/// **live** is not the same as it **separating**. On a single-UV-island mesh every
/// candidate sits in one component, key 1 ties for all of them, and the tie-break must
/// still reduce through keys 2–4. So the textured boundary corner behaves exactly as the
/// untextured one: the cap follows the **lower slot** whichever quad holds it.
#[test]
fn face_cd_uv_single_island_at_odd_segments_still_reduces_to_the_lower_keys() {
    for swapped in [false, true] {
        let (mut mesh, corner) = uv_decorated_two_quad_plane(1, swapped);

        let result = mesh.bevel(
            &[corner.into()],
            BevelOptions {
                offset: 0.2,
                ..Default::default()
            },
        );
        mesh.validate_topology();

        assert_eq!(result.faces.len(), 1, "the corner mints exactly its cap");
        assert_eq!(
            mesh.face_material_index(result.faces[0]),
            2,
            "swapped = {swapped}: one UV island ⇒ key 1 ties across the candidates, so the \
             seamed corner's cap takes the tie-break's lower slot — declining here would \
             pin the cap to the positional quad instead",
        );
    }
}

/// The **straddling** regime: candidates in two different interpolation components. Key 1
/// separates them and keeps the **lowest-numbered** one, and §6.6.1.4.1 numbers components
/// by the enumeration index of their first face — so the cap tracks the quad that holds
/// the mesh's first face, whichever slot that quad happens to carry. (Both quads are
/// coplanar here, so the top/bottom label swaps are no-ops and the numbering is pure
/// flood-fill order.)
#[test]
fn face_cd_uv_straddling_candidates_at_odd_segments_take_the_lowest_numbered_component() {
    let mut slots = Vec::new();
    for swapped in [false, true] {
        let (mut mesh, corner) = uv_decorated_two_quad_plane(2, swapped);
        let first_slot = mesh.face_material_index(mesh.face_keys()[0]);

        let result = mesh.bevel(
            &[corner.into()],
            BevelOptions {
                offset: 0.2,
                ..Default::default()
            },
        );
        mesh.validate_topology();

        assert_eq!(result.faces.len(), 1, "the corner mints exactly its cap");
        let cap_slot = mesh.face_material_index(result.faces[0]);
        assert_eq!(
            cap_slot, first_slot,
            "swapped = {swapped}: the shared edge is a UV-island boundary, so the two \
             candidates sit in different components and key 1 keeps the component seeded \
             by the first enumerated face",
        );
        slots.push(cap_slot);
    }
    assert_ne!(
        slots[0], slots[1],
        "swapping the decoration must move the slot, or the assertion above would be \
         satisfied by a rule that simply always picks the lower slot",
    );
}

/// An open fan of three coplanar triangles about a boundary apex, materials in fan order,
/// with one face's UV image translated away so the partition splits **lopsidedly**: two
/// faces in the low-numbered component, one alone in the other.
fn uv_split_open_fan(slots: [i32; 3], detached: usize) -> (LoopMesh, VertexKey) {
    let (mut mesh, centre, rim) = open_planar_fan(4, 0);
    let uv = mesh
        .register_loop_attribute(ElementAttributeDesc::uv(AttributeKey::custom("uv")))
        .expect("Vec2 is a supported loop attribute format");

    for k in 0..3 {
        let face = open_fan_face(&mesh, centre, &rim, k);
        mesh.set_face_material_index(face, slots[k]);
        let shift = if k == detached { 10.0 } else { 0.0 };
        for loop_key in mesh.face_loop_keys(face) {
            let position = mesh.vertex_position(mesh.loop_vertex(loop_key));
            mesh.loop_attribs_mut().set_vec2(
                uv,
                loop_key.as_usize(),
                [position.x + shift, position.y],
            );
        }
    }
    (mesh, centre)
}

/// §6.6.1.4.1's numbering, asserted on the partition itself: the component seeded by the
/// **earlier** face in enumeration order gets the lower id, and it is the one holding
/// *two* of the three faces here.
#[test]
fn interp_component_numbering_follows_face_enumeration_order() {
    let (mesh, _apex) = uv_split_open_fan([3, 5, 1], 2);
    let bank = TemplateBank::new(&mesh, 1);
    let partition = bank
        .interp_components()
        .expect("a UV layer at odd `segments`");

    let faces = mesh.face_keys();
    let ids: Vec<u32> = faces
        .iter()
        .map(|&face| {
            partition
                .component_of(face)
                .expect("a pristine face is labelled")
        })
        .collect();

    assert_eq!(ids[0], ids[1], "the two contiguous faces flood together");
    assert_ne!(
        ids[1], ids[2],
        "the translated UV image seams the third face off"
    );
    assert!(
        ids[0] < ids[2],
        "the component seeded first in enumeration order takes the lower id, got {ids:?}",
    );
}

/// The case the earlier decline got wrong: candidates **straddling** components where the
/// lowest-numbered component still holds **more than one** of them. Key 1 narrows to that
/// component and keys 2–4 then decide *within* it — here key 3, the lower material slot.
///
/// The keying is deliberately lopsided so all three readings differ: the global lowest
/// slot (1) sits alone in the *high* component, and the low component holds slots 3 and 5.
/// A rule that ignored the split would answer 1; one that declined to the positional donor
/// would answer 5; §6.6.1.4.1 answers **3**.
#[test]
fn face_cd_lopsided_uv_split_runs_keys_two_to_four_inside_the_lowest_component() {
    let (mut mesh, apex) = uv_split_open_fan([3, 5, 1], 2);

    let result = mesh.bevel(
        &[apex.into()],
        BevelOptions {
            offset: 0.2,
            ..Default::default()
        },
    );
    mesh.validate_topology();

    assert_eq!(
        result.faces.len(),
        1,
        "the open fan's apex mints exactly its cap"
    );
    assert_eq!(
        mesh.face_material_index(result.faces[0]),
        3,
        "key 1 keeps the two-face low component; key 3 then picks the lower slot in it",
    );
}

/// §6.6.1.4.1's *partial* canonicalisation, on the only shape that exercises it: three
/// components at three distinct heights. Read the steps literally — they swap against the
/// components of the first two **faces in enumeration order**, not against the literal ids
/// `0` and `1`, and they demonstrably do *not* end with "top = 0, bottom = 1".
#[test]
fn interp_component_numbering_applies_the_partial_top_bottom_swaps() {
    let mut mesh = LoopMesh::new();
    mesh.register_loop_attribute(ElementAttributeDesc::uv(AttributeKey::custom("uv")))
        .expect("Vec2 is a supported loop attribute format");

    // Disjoint quads: no shared edge, so each is its own component whatever the UVs are.
    for height in [0.0_f32, 2.0, -1.0] {
        let corners: Vec<VertexKey> = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
            .into_iter()
            .map(|(x, y)| mesh.create_vertex(Vec3::new(x + height * 4.0, y, height)))
            .collect();
        mesh.face_create_verts(&corners, FaceCreateFlags::NONE);
    }

    let bank = TemplateBank::new(&mesh, 1);
    let partition = bank
        .interp_components()
        .expect("a UV layer at odd `segments`");
    let ids: Vec<u32> = mesh
        .face_keys()
        .iter()
        .map(|&face| {
            partition
                .component_of(face)
                .expect("a pristine face is labelled")
        })
        .collect();

    // Flood fill labels them 0 / 1 / 2. Top is face 1 (id 1), bottom is face 2 (id 2).
    // Swap A exchanges face 0's component (0) with top (1) → 1 / 0 / 2. Swap B exchanges
    // face 1's *current* component (0) with bottom (2) → 1 / 2 / 0.
    assert_eq!(
        ids,
        vec![1, 2, 0],
        "the topmost face ends on id 2, not 0 — the canonicalisation is partial and the \
         artefact is what has to be reproduced",
    );
}

/// §6.6.1.4.1's partition is built **only** when key 1 is live — a float2 layer present
/// *and* `segments` odd — and never otherwise; and when it is built it is the loop-CD
/// equivalence relation, which the island count of the fixture decides.
#[test]
fn interp_component_partition_is_built_only_where_key_one_is_live() {
    let (untextured, _shared) = two_quad_strip();
    assert!(
        TemplateBank::new(&untextured, 1)
            .interp_components()
            .is_none(),
        "no float2 loop layer ⇒ key 1 is inert ⇒ no partition at any `segments`",
    );

    let (textured, _corner) = uv_decorated_two_quad_plane(1, false);
    assert!(
        TemplateBank::new(&textured, 2)
            .interp_components()
            .is_none(),
        "an even `segments` ⇒ key 1 is inert even with a UV layer ⇒ no partition",
    );

    let single_island = TemplateBank::new(&textured, 1);
    let partition = single_island
        .interp_components()
        .expect("a UV layer at odd `segments` ⇒ key 1 is live ⇒ the partition is built");
    assert!(
        partition.all_share_component(&textured.face_keys()),
        "a continuous unwrap joins the two quads across the shared edge: one component",
    );

    let (two_islands, _corner) = uv_decorated_two_quad_plane(2, false);
    let split = TemplateBank::new(&two_islands, 1);
    assert!(
        !split
            .interp_components()
            .expect("live")
            .all_share_component(&two_islands.face_keys()),
        "the shifted unwrap makes the shared edge a UV-island boundary: two components",
    );
}

/// The single-UV-island two-quad plane of [`uv_decorated_two_quad_plane`] plus one more
/// loop layer, with the left quad's corners and the right quad's handed back separately so
/// a test can make the shared edge discontinuous on that layer **alone**.
fn two_quad_plane_with_extra_loop_layer(
    desc: ElementAttributeDesc,
) -> (LoopMesh, ElementAttributeKey, Vec<LoopKey>, Vec<LoopKey>) {
    let (mut mesh, _corner) = uv_decorated_two_quad_plane(1, false);
    let layer = mesh
        .register_loop_attribute(desc)
        .expect("a supported loop attribute format");

    let left_vert = vert_at(&mesh, Vec3::new(-2.0, -1.0, 0.0));
    let (mut left, mut right) = (Vec::new(), Vec::new());
    for face in mesh.face_keys() {
        let corners = if mesh.face_vert_keys(face).contains(&left_vert) {
            &mut left
        } else {
            &mut right
        };
        corners.extend(mesh.face_loop_keys(face));
    }
    (mesh, layer, left, right)
}

fn quads_share_an_interp_component(mesh: &LoopMesh) -> bool {
    TemplateBank::new(mesh, 1)
        .interp_components()
        .expect("a float2 loop layer at odd `segments` ⇒ the partition is built")
        .all_share_component(&mesh.face_keys())
}

/// BEVEL.md §6.6.1.4.1's asymmetry, in the direction the partition's *edges* run: they are
/// read from the **full blendable set** (§6.6.1.4.0(a)), so a discontinuity in either colour
/// kind splits components even though no float2 value disagrees anywhere.
#[test]
fn interp_component_edges_split_on_a_colour_only_discontinuity() {
    let (mut mesh, color, left, right) = two_quad_plane_with_extra_loop_layer(
        ElementAttributeDesc::color(AttributeKey::custom("color"), AttrType::Vec4),
    );
    for &corner in left.iter().chain(&right) {
        mesh.loop_attribs_mut()
            .set_vec4(color, corner.as_usize(), [0.25, 0.5, 0.75, 1.0]);
    }
    assert!(
        quads_share_an_interp_component(&mesh),
        "an agreeing float colour layer leaves the continuous unwrap's single component alone",
    );

    for &corner in &right {
        mesh.loop_attribs_mut()
            .set_vec4(color, corner.as_usize(), [0.25, 0.5, 0.75, 0.0]);
    }
    assert!(
        !quads_share_an_interp_component(&mesh),
        "float RGBA is blendable, so a colour-only discontinuity seams the shared edge and \
         separates the quads",
    );

    let (mut mesh, byte_color, _left, right) = two_quad_plane_with_extra_loop_layer(
        ElementAttributeDesc::color(AttributeKey::custom("byte_color"), AttrType::Vec4U8),
    );
    for &corner in &right {
        mesh.loop_attribs_mut()
            .set_vec4u8(byte_color, corner.as_usize(), [0, 0, 0, 1]);
    }
    assert!(
        !quads_share_an_interp_component(&mesh),
        "byte colour is compared on the raw 0..=255 scale, so its tolerance is vacuous and a \
         one-step difference already separates the quads",
    );
}

/// §6.6.1.4.0(b): the compare is tolerant, absolute, and on the **summed squared difference
/// over the whole vector** — so a disagreement can be reached either by one large component
/// gap or by accumulating small ones, and sub-threshold paint noise is not a seam.
#[test]
fn blendable_compare_is_tolerant_and_whole_vector() {
    let (mut mesh, auxiliary, _left, right) = two_quad_plane_with_extra_loop_layer(
        ElementAttributeDesc::numeric(AttributeKey::custom("original_space"), AttrType::Vec2),
    );
    for &corner in &right {
        mesh.loop_attribs_mut()
            .set_vec2(auxiliary, corner.as_usize(), [2e-3, 0.0]);
    }
    assert!(
        quads_share_an_interp_component(&mesh),
        "summed squared difference 4e-6 is strictly below the 2-float threshold 1e-5: one \
         component, where an exact compare would over-seam",
    );

    for &corner in &right {
        mesh.loop_attribs_mut()
            .set_vec2(auxiliary, corner.as_usize(), [4e-3, 0.0]);
    }
    assert!(
        !quads_share_an_interp_component(&mesh),
        "1.6e-5 exceeds 1e-5: a genuine disagreement",
    );

    let (mut mesh, color, left, right) = two_quad_plane_with_extra_loop_layer(
        ElementAttributeDesc::color(AttributeKey::custom("color"), AttrType::Vec4),
    );
    for &corner in left.iter().chain(&right) {
        mesh.loop_attribs_mut()
            .set_vec4(color, corner.as_usize(), [0.5; 4]);
    }
    for &corner in &right {
        mesh.loop_attribs_mut()
            .set_vec4(color, corner.as_usize(), [0.52, 0.5, 0.5, 0.5]);
    }
    assert!(
        quads_share_an_interp_component(&mesh),
        "a 2e-2 gap in one colour component sums to 4e-4, below the colour threshold 1e-3",
    );

    for &corner in &right {
        mesh.loop_attribs_mut()
            .set_vec4(color, corner.as_usize(), [0.52; 4]);
    }
    assert!(
        !quads_share_an_interp_component(&mesh),
        "the same 2e-2 gap on all four components sums to 1.6e-3, which exceeds 1e-3 — the \
         test is whole-vector, not per-component",
    );
}

/// §6.6.1.4.0(a)'s closed list is keyed on layer **type**, and "float-valued" is not the
/// key: a scalar float and a 3-vector float loop layer are skipped, as are the integer and
/// boolean families. A discontinuity in any of them is invisible to the partition.
#[test]
fn interp_component_edges_never_read_a_non_blendable_layer() {
    for (desc, paint) in [
        (
            ElementAttributeDesc::numeric(AttributeKey::custom("mask"), AttrType::F32),
            &(|mesh: &mut LoopMesh, layer, corner: LoopKey| {
                mesh.loop_attribs_mut()
                    .set_f32(layer, corner.as_usize(), 7.0);
            }) as &dyn Fn(&mut LoopMesh, ElementAttributeKey, LoopKey),
        ),
        (
            ElementAttributeDesc::numeric(AttributeKey::custom("tangent"), AttrType::Vec3),
            &|mesh, layer, corner| {
                mesh.loop_attribs_mut()
                    .set_vec3(layer, corner.as_usize(), [7.0, 7.0, 7.0]);
            },
        ),
        (
            ElementAttributeDesc::numeric(AttributeKey::custom("group"), AttrType::I32),
            &|mesh, layer, corner| {
                mesh.loop_attribs_mut().set_i32(layer, corner.as_usize(), 7);
            },
        ),
        (
            ElementAttributeDesc::numeric(AttributeKey::custom("pinned"), AttrType::Bool),
            &|mesh, layer, corner| {
                mesh.loop_attribs_mut()
                    .set_bool(layer, corner.as_usize(), true);
            },
        ),
    ] {
        let name = desc.key.clone();
        let (mut mesh, layer, _left, right) = two_quad_plane_with_extra_loop_layer(desc);
        for &corner in &right {
            paint(&mut mesh, layer, corner);
        }
        assert!(
            quads_share_an_interp_component(&mesh),
            "{name:?} is not a blendable type, so however discontinuous it is the partition \
             never reads it and the quads stay in one component",
        );
    }
}

/// §6.6.1.4.0(b)'s NaN corollary: the threshold test is a strict `<`, NaN propagates out of
/// the difference-and-square chain, so a NaN component is a **disagreement** — including
/// against the bit-identical NaN on the other side.
#[test]
fn a_nan_in_a_blendable_layer_seams_the_edge_against_itself() {
    let (mut mesh, auxiliary, left, right) = two_quad_plane_with_extra_loop_layer(
        ElementAttributeDesc::numeric(AttributeKey::custom("original_space"), AttrType::Vec2),
    );
    for &corner in left.iter().chain(&right) {
        mesh.loop_attribs_mut()
            .set_vec2(auxiliary, corner.as_usize(), [f32::NAN, 0.0]);
    }
    assert!(
        !quads_share_an_interp_component(&mesh),
        "both sides hold the same NaN and the edge is still a seam",
    );
}

/// The other half of §6.6.1.4.1's asymmetry: the **arming** gate stays the one-row float2
/// *inventory* test, so on a mesh whose only blendable layers are colours no partition is
/// built at all and key 1 stays inert however discontinuous the colour is.
#[test]
fn a_colour_only_mesh_never_arms_the_partition() {
    let (mut mesh, _shared) = two_quad_strip();
    let color = mesh
        .register_loop_attribute(ElementAttributeDesc::color(
            AttributeKey::custom("color"),
            AttrType::Vec4,
        ))
        .expect("Vec4 is a supported loop attribute format");

    let left_vert = vert_at(&mesh, Vec3::new(-2.0, -1.0, 0.0));
    for face in mesh.face_keys() {
        if mesh.face_vert_keys(face).contains(&left_vert) {
            continue;
        }
        for corner in mesh.face_loop_keys(face) {
            mesh.loop_attribs_mut()
                .set_vec4(color, corner.as_usize(), [1.0; 4]);
        }
    }

    assert!(
        TemplateBank::new(&mesh, 1).interp_components().is_none(),
        "no float2 loop layer ⇒ nothing arms the partition, colours notwithstanding",
    );
}

/// BEVEL.md §6.6.1: the same whole-block inherit on the edge-bevel **strip** quad,
/// whose representative is one of the beveled edge's two flanking faces.
#[test]
fn face_cd_strip_inherits_whole_block_of_a_flanking_face() {
    let mut mesh = LoopMesh::new();
    mesh_add_cube(&mut mesh);
    let tint = tint_and_slot_faces(&mut mesh);

    let edge = cube_edge(&mesh, Vec3::new(1.0, 1.0, 1.0), Vec3::new(-1.0, 1.0, 1.0));
    let flanking: Vec<(i32, [f32; 3])> = mesh
        .edge_face_keys(edge)
        .iter()
        .map(|&face| {
            (
                mesh.face_material_index(face),
                mesh.face_attribs().get_vec3(tint, face.as_usize()).unwrap(),
            )
        })
        .collect();
    assert_eq!(flanking.len(), 2, "a cube edge is manifold");

    let result = mesh.bevel(
        &[edge.into()],
        BevelOptions {
            offset: 0.25,
            affect: BevelAffect::Edges,
            material: -1,
            ..Default::default()
        },
    );
    let strip = *result
        .faces
        .first()
        .expect("the isolated edge bevel builds its strip quad");

    let inherited = (
        mesh.face_material_index(strip),
        mesh.face_attribs()
            .get_vec3(tint, strip.as_usize())
            .unwrap(),
    );
    assert!(
        flanking.contains(&inherited),
        "the strip must carry one flanking original's whole block {flanking:?}, got \
         {inherited:?}",
    );
}

/// BEVEL.md §6.6.2's multi-segment profile-strip row (§8.2.2): the run of `segments`
/// quads splits its stage-1 donor about the strip's **centre line** — the quads on the
/// near flank's side inherit the near flanking face's whole block, the quads strictly
/// past it the far flank's. At odd `segments` the quad that *straddles* the centre line
/// belongs to neither half and takes the **near** flank — the same donor the
/// `segments = 1` quad takes, not the far one its past-the-midpoint position suggests.
///
/// A cube edge terminates at two valence-3 corners, which shed no fill (§8.3.0.1), so
/// the run *is* the whole result: at `segments = 2` the blocks read near, far; at
/// `segments = 3`, near, near, far.
#[test]
fn face_cd_multisegment_strip_donor_splits_about_the_centre_line() {
    for segments in [2_u32, 3] {
        let mut mesh = LoopMesh::new();
        mesh_add_cube(&mut mesh);
        let tint = tint_and_slot_faces(&mut mesh);

        let edge = cube_edge(&mesh, Vec3::new(1.0, 1.0, 1.0), Vec3::new(-1.0, 1.0, 1.0));
        let (near_face, far_face) = mesh.edge_flanking_faces(edge);
        let near_face = near_face.expect("a cube edge has a near flank");
        let far_face = far_face.expect("a cube edge has a far flank");
        assert_ne!(near_face, far_face, "a cube edge is manifold");

        let block_of = |mesh: &LoopMesh, face: FaceKey| {
            (
                mesh.face_material_index(face),
                mesh.face_attribs().get_vec3(tint, face.as_usize()).unwrap(),
            )
        };
        let near_block = block_of(&mesh, near_face);
        let far_block = block_of(&mesh, far_face);
        let near_center = mesh.face_center_of_mass(near_face);

        let result = mesh.bevel(
            &[edge.into()],
            BevelOptions {
                offset: 0.25,
                affect: BevelAffect::Edges,
                segments,
                material: -1,
                ..Default::default()
            },
        );
        assert_eq!(
            result.faces.len(),
            segments as usize,
            "a cube edge's two valence-3 terminals add no fill, so the run is the result",
        );

        // Ordered by how far across the chamfer they sit: the profile is monotone from
        // the near flank to the far one, so this is the centre-line order.
        let mut strip = result.faces.clone();
        strip.sort_by(|&left, &right| {
            let left_span = (mesh.face_center_of_mass(left) - near_center).length();
            let right_span = (mesh.face_center_of_mass(right) - near_center).length();
            left_span.partial_cmp(&right_span).unwrap()
        });
        let blocks: Vec<(i32, [f32; 3])> =
            strip.iter().map(|&face| block_of(&mesh, face)).collect();

        let near_quads = (segments as usize + 1) / 2;
        let expected: Vec<(i32, [f32; 3])> = (0..segments as usize)
            .map(|quad| {
                if quad < near_quads {
                    near_block
                } else {
                    far_block
                }
            })
            .collect();
        assert_eq!(
            blocks, expected,
            "at segments = {segments} the strip must inherit near ×{near_quads} then far \
             for the rest (near {near_block:?}, far {far_block:?})",
        );
    }
}

/// One beveled cube edge whose two flanking originals carry deliberately *inverted*
/// material slots — the near flank the higher one — so §6.6.1.4's key 3 (lowest slot
/// donates) picks the **far** flank and any re-route is visible against the positional
/// answer. Returns the edge and the two flanks' whole blocks, near first.
fn inverted_slot_strip_fixture(
    mesh: &mut LoopMesh,
    tint: ElementAttributeKey,
) -> (EdgeKey, (i32, [f32; 3]), (i32, [f32; 3])) {
    let edge = cube_edge(mesh, Vec3::new(1.0, 1.0, 1.0), Vec3::new(-1.0, 1.0, 1.0));
    let (near_face, far_face) = mesh.edge_flanking_faces(edge);
    let near_face = near_face.expect("a cube edge has a near flank");
    let far_face = far_face.expect("a cube edge has a far flank");
    assert_ne!(near_face, far_face, "a cube edge is manifold");
    mesh.set_face_material_index(near_face, 7);
    mesh.set_face_material_index(far_face, 2);

    let block_of = |mesh: &LoopMesh, face: FaceKey| {
        (
            mesh.face_material_index(face),
            mesh.face_attribs().get_vec3(tint, face.as_usize()).unwrap(),
        )
    };
    (edge, block_of(mesh, near_face), block_of(mesh, far_face))
}

/// The strip's quads' per-face blocks in centre-line order — the near flank's side
/// first. The profile is monotone across the chamfer, so distance from the near flank's
/// centre is that order.
fn strip_blocks_in_centre_line_order(
    mesh: &LoopMesh,
    result: &BevelResult,
    near_center: Vec3,
    tint: ElementAttributeKey,
) -> Vec<(i32, [f32; 3])> {
    let mut strip = result.faces.clone();
    strip.sort_by(|&left, &right| {
        let left_span = (mesh.face_center_of_mass(left) - near_center).length();
        let right_span = (mesh.face_center_of_mass(right) - near_center).length();
        left_span.partial_cmp(&right_span).unwrap()
    });
    strip
        .iter()
        .map(|&face| {
            (
                mesh.face_material_index(face),
                mesh.face_attribs().get_vec3(tint, face.as_usize()).unwrap(),
            )
        })
        .collect()
}

/// BEVEL.md §6.6.2's strip row + §6.6.1.4: at odd `segments` the quad that straddles
/// the centre line takes the near flank **only while the beveled edge is seam-free**.
/// At a seam its donor is the tie-break winner between the edge's two flanking faces —
/// here the far flank, which carries the lower material slot (key 3).
///
/// The seam is a **float-RGBA-only** loop-layer discontinuity, which is the configuration
/// that leaves the rule fully decided: float RGBA is blendable (§6.6.1.4.0(a)) so it
/// seams the edge by condition (2), yet key 1 (§6.6.1.4.1) stays inert because the mesh
/// carries no float2 layer to arm the partition. Only the centre quad moves — its two
/// neighbours keep their positional halves.
#[test]
fn face_cd_straddle_quad_takes_the_tie_break_winner_at_a_seamed_edge() {
    let mut mesh = LoopMesh::new();
    mesh_add_cube(&mut mesh);
    let tint = tint_and_slot_faces(&mut mesh);
    let (edge, near_block, far_block) = inverted_slot_strip_fixture(&mut mesh, tint);

    let color = mesh
        .register_loop_attribute(ElementAttributeDesc::color(
            AttributeKey::custom("l_color"),
            AttrType::Vec4,
        ))
        .unwrap();
    let near_face = mesh.edge_flanking_faces(edge).0.unwrap();
    let near_center = mesh.face_center_of_mass(near_face);
    for corner in mesh.face_loop_keys(near_face) {
        mesh.loop_attribs_mut()
            .set_vec4(color, corner.as_usize(), [1.0, 0.0, 0.0, 1.0]);
    }

    let result = mesh.bevel(
        &[edge.into()],
        BevelOptions {
            offset: 0.25,
            affect: BevelAffect::Edges,
            segments: 3,
            material: -1,
            ..Default::default()
        },
    );
    let blocks = strip_blocks_in_centre_line_order(&mesh, &result, near_center, tint);
    assert_eq!(
        blocks,
        vec![near_block, far_block, far_block],
        "the seamed edge's centre quad must take the tie-break winner (the lower slot, \
         {far_block:?}), leaving its neighbours on near {near_block:?} / far",
    );
}

/// The seam-free regression the rule above must not swallow: the very same fixture,
/// with the loop colour layer *present but contiguous* across the beveled edge. No
/// seam, so §6.6.1.4 never fires and the straddle quad stays on the **near** flank —
/// even though the far flank would win the tie-break outright.
#[test]
fn face_cd_straddle_quad_keeps_the_near_flank_when_the_edge_is_contiguous() {
    let mut mesh = LoopMesh::new();
    mesh_add_cube(&mut mesh);
    let tint = tint_and_slot_faces(&mut mesh);
    let (edge, near_block, far_block) = inverted_slot_strip_fixture(&mut mesh, tint);

    mesh.register_loop_attribute(ElementAttributeDesc::color(
        AttributeKey::custom("l_color"),
        AttrType::Vec4,
    ))
    .unwrap();
    let near_center = mesh.face_center_of_mass(mesh.edge_flanking_faces(edge).0.unwrap());

    let result = mesh.bevel(
        &[edge.into()],
        BevelOptions {
            offset: 0.25,
            affect: BevelAffect::Edges,
            segments: 3,
            material: -1,
            ..Default::default()
        },
    );
    let blocks = strip_blocks_in_centre_line_order(&mesh, &result, near_center, tint);
    assert_eq!(
        blocks,
        vec![near_block, near_block, far_block],
        "a contiguous edge keeps the positional split — near, near, far",
    );
}

/// §6.6.2's strip-row scoping: at **even** `segments` no quad straddles the centre
/// line and **no** strip face's donor is re-routed, however seamed the beveled edge.
/// Each half simply keeps its own flanking face.
#[test]
fn face_cd_even_segments_strip_ignores_the_edge_seam() {
    let mut mesh = LoopMesh::new();
    mesh_add_cube(&mut mesh);
    let tint = tint_and_slot_faces(&mut mesh);
    let (edge, near_block, far_block) = inverted_slot_strip_fixture(&mut mesh, tint);

    let color = mesh
        .register_loop_attribute(ElementAttributeDesc::color(
            AttributeKey::custom("l_color"),
            AttrType::Vec4,
        ))
        .unwrap();
    let near_face = mesh.edge_flanking_faces(edge).0.unwrap();
    let near_center = mesh.face_center_of_mass(near_face);
    for corner in mesh.face_loop_keys(near_face) {
        mesh.loop_attribs_mut()
            .set_vec4(color, corner.as_usize(), [1.0, 0.0, 0.0, 1.0]);
    }

    let result = mesh.bevel(
        &[edge.into()],
        BevelOptions {
            offset: 0.25,
            affect: BevelAffect::Edges,
            segments: 2,
            material: -1,
            ..Default::default()
        },
    );
    let blocks = strip_blocks_in_centre_line_order(&mesh, &result, near_center, tint);
    assert_eq!(
        blocks,
        vec![near_block, far_block],
        "an even run keeps both halves positional at a seam",
    );
}

/// §6.6.1.4.1 at the straddle quad: with a float2 layer present at odd `segments` key 1
/// is **live**, and when the two candidates sit in *different* interpolation components —
/// as they do at a UV-island boundary, which is also what seams the edge — key 1 alone
/// decides, keeping the **lower-numbered** flank. The expectation is read off the
/// partition rather than hard-coded, because the numbering is construction-order
/// dependent and hard-coding it would pin the cube's face order, not the rule.
#[test]
fn face_cd_straddle_quad_takes_the_lowest_numbered_component_across_uv_components() {
    let mut mesh = LoopMesh::new();
    mesh_add_cube(&mut mesh);
    let tint = tint_and_slot_faces(&mut mesh);
    let (edge, near_block, far_block) = inverted_slot_strip_fixture(&mut mesh, tint);

    let uv = mesh
        .register_loop_attribute(ElementAttributeDesc::uv(AttributeKey::custom("l_uv")))
        .unwrap();
    let near_face = mesh.edge_flanking_faces(edge).0.unwrap();
    let near_center = mesh.face_center_of_mass(near_face);
    for corner in mesh.face_loop_keys(near_face) {
        mesh.loop_attribs_mut()
            .set_vec2(uv, corner.as_usize(), [5.0, 5.0]);
    }

    let far_face = mesh.edge_flanking_faces(edge).1.unwrap();
    let bank = TemplateBank::new(&mesh, 3);
    let partition = bank
        .interp_components()
        .expect("a float2 layer at odd `segments`");
    let (near_component, far_component) = (
        partition.component_of(near_face).unwrap(),
        partition.component_of(far_face).unwrap(),
    );
    assert_ne!(
        near_component, far_component,
        "the painted flank is a UV island of its own, so key 1 is separating here",
    );
    let straddle_block = if near_component < far_component {
        near_block
    } else {
        far_block
    };

    let result = mesh.bevel(
        &[edge.into()],
        BevelOptions {
            offset: 0.25,
            affect: BevelAffect::Edges,
            segments: 3,
            material: -1,
            ..Default::default()
        },
    );
    let blocks = strip_blocks_in_centre_line_order(&mesh, &result, near_center, tint);
    assert_eq!(
        blocks,
        vec![near_block, straddle_block, far_block],
        "the straddle quad follows key 1 to the lower-numbered component; the quads off \
         the centre line stay positional",
    );
}

/// An open two-quad plane whose second quad is wound **against** the first, so both
/// flanks of the shared edge traverse it the same way. Returns the shared edge.
fn noncontiguous_two_quad_plane() -> (LoopMesh, EdgeKey) {
    let mut mesh = LoopMesh::new();
    let corner = |x: f32, y: f32| Vec3::new(x, y, 0.0);
    let a = mesh.create_vertex(corner(-2.0, -1.0));
    let b = mesh.create_vertex(corner(0.0, -1.0));
    let c = mesh.create_vertex(corner(2.0, -1.0));
    let d = mesh.create_vertex(corner(2.0, 1.0));
    let e = mesh.create_vertex(corner(0.0, 1.0));
    let f = mesh.create_vertex(corner(-2.0, 1.0));
    mesh.face_create_verts(&[a, b, e, f], FaceCreateFlags::NONE);
    // Reversed against the first quad, so both flanks traverse `b → e`.
    mesh.face_create_verts(&[e, d, c, b], FaceCreateFlags::NONE);
    let edge = mesh.query_edge_exists(b, e).expect("the shared edge");
    (mesh, edge)
}

/// §6.6.1.4 **condition (1)** at the edge-level site, pinned at the *predicate* rather
/// than through a donor observation: the condition reads the edge's own two
/// flanking-face slots, and fires exactly when one is empty.
///
/// Naming the fixture precisely matters here, because the obvious one is dead. An
/// interior edge with boundary *endpoints* (a two-quad strip's shared edge) fills both
/// its slots and is **not** a condition-(1) seam; it is the endpoint's boundary *legs*
/// that are seams, which is the corner-level form. The edge that fires is the boundary
/// edge itself, and it fires with no loop layer on the mesh at all.
#[test]
fn face_cd_an_empty_flank_slot_is_a_condition_one_seam_with_no_loop_layer() {
    let mut mesh = LoopMesh::new();
    let corner = |x: f32, y: f32| Vec3::new(x, y, 0.0);
    let a = mesh.create_vertex(corner(-2.0, -1.0));
    let b = mesh.create_vertex(corner(0.0, -1.0));
    let c = mesh.create_vertex(corner(2.0, -1.0));
    let d = mesh.create_vertex(corner(2.0, 1.0));
    let e = mesh.create_vertex(corner(0.0, 1.0));
    let f = mesh.create_vertex(corner(-2.0, 1.0));
    mesh.face_create_verts(&[a, b, e, f], FaceCreateFlags::NONE);
    mesh.face_create_verts(&[b, c, d, e], FaceCreateFlags::NONE);
    assert!(
        mesh.loop_attribs().is_empty(),
        "the fixture must carry no loop layer, or condition (2) is in play",
    );

    let boundary = mesh.query_edge_exists(a, b).expect("a boundary edge");
    assert!(
        mesh.edge_is_interp_seam(boundary),
        "a boundary edge leaves one flank slot empty — condition (1), layer-free",
    );

    let interior = mesh.query_edge_exists(b, e).expect("the shared edge");
    assert!(
        !mesh.edge_is_interp_seam(interior),
        "an interior edge with boundary endpoints fills both its own slots, and with \
         zero loop layers condition (2) short-circuits to contiguous",
    );
}

/// §6.6.1.4 condition (2)'s **zero-loop-layer short-circuit**, and the classification
/// that rides on it: inconsistent winding is a condition-(2) trigger, not a
/// condition-(1) one. Winding plays no part in filling the edge's flank slots, so an
/// inconsistently wound *manifold* edge fills both — condition (1) cannot fire — and on
/// an untextured mesh condition (2) short-circuits before the corner pairing is even
/// attempted. The edge is therefore **not a seam**, no tie-break runs, and the straddle
/// quad keeps the positional near flank.
#[test]
fn face_cd_inconsistently_wound_flanks_are_not_a_seam_with_no_loop_layer() {
    let (mesh, edge) = noncontiguous_two_quad_plane();
    assert!(
        mesh.loop_attribs().is_empty(),
        "the fixture must carry no loop layer, or the short-circuit is not under test",
    );
    assert!(
        !mesh.edge_is_interp_seam(edge),
        "condition (2) short-circuits to contiguous on a zero-loop-layer mesh, and \
         inconsistent winding is not a condition-(1) trigger",
    );

    let (near_face, far_face) = mesh.edge_flanking_faces(edge);
    assert_eq!(
        mesh.straddle_quad_donor(edge, near_face, far_face, 3, None),
        near_face,
        "an unseamed edge keeps §6.6.1's positional donor at every odd `segments`",
    );
}

/// The other side of the same short-circuit: add **one** loop layer — even one whose
/// type defines no blend at all — and condition (2) is evaluated. The corner pairing an
/// inconsistently wound radial pair makes impossible then fails, which is a seam on that
/// ground alone, whatever any layer holds.
#[test]
fn face_cd_inconsistently_wound_flanks_are_a_seam_once_any_loop_layer_exists() {
    let (mut mesh, edge) = noncontiguous_two_quad_plane();
    mesh.register_loop_attribute(ElementAttributeDesc::string(AttributeKey::custom(
        "l_label",
    )))
    .expect("a loop layer with no interpolation math");

    assert!(
        mesh.edge_is_interp_seam(edge),
        "the flanks' corners cannot be paired across an inconsistently wound edge, so \
         the layers-agree test reports seam once any loop layer is present",
    );
}

/// A non-manifold "book" of three triangular pages hinged on the spine `v0–v1`, every
/// page wound `[v0, v1, tip]`, so the spine's radial cycle has valence 3. Returns the
/// spine.
fn book_spine_three_pages() -> (LoopMesh, EdgeKey) {
    let mut mesh = LoopMesh::new();
    let v0 = mesh.create_vertex(Vec3::new(0.0, 0.0, 0.0));
    let v1 = mesh.create_vertex(Vec3::new(0.0, 1.0, 0.0));
    for page in 0..3 {
        let tip = mesh.create_vertex(Vec3::new(1.0 + page as f32, 0.5, 0.0));
        mesh.face_create_verts(&[v0, v1, tip], FaceCreateFlags::NONE);
    }
    let spine = mesh.query_edge_exists(v0, v1).expect("the spine edge");
    (mesh, spine)
}

/// §6.6.1.4 condition (1)'s third bullet: an edge with **three or more** faces does not
/// "miss" a face, so the test falls through to condition (2) — which, with zero loop
/// layers, short-circuits to contiguous. A predicate that asked "is there a face arriving
/// at each endpoint" through the fan would resolve no arriving face at `v0` on this
/// fixture and misreport a condition-(1) seam on an untextured mesh.
#[test]
fn face_cd_a_radial_three_edge_is_not_a_condition_one_seam_with_no_loop_layer() {
    let (mesh, spine) = book_spine_three_pages();
    assert_eq!(
        mesh.iter_radial_loops(spine).count(),
        3,
        "the fixture must be radial-3, or the fall-through is not under test",
    );
    assert!(
        mesh.loop_attribs().is_empty(),
        "the fixture must carry no loop layer, or condition (2) is in play",
    );
    assert!(
        !mesh.edge_is_interp_seam(spine),
        "a radial-3 edge misses no face, so condition (1) cannot fire and the layer-free \
         mesh short-circuits to contiguous",
    );
}

/// The other half of §6.6.1.4's corollary — a fan-resolved non-manifold edge is a seam
/// **iff** the mesh has at least one loop layer. The same spine that reads contiguous
/// untextured becomes a seam once any loop layer exists, because the corner pairing
/// condition (2) opens with rejects the third radial loop. Both states are asserted here,
/// so the seam is attributable to the layer rather than to the topology alone.
#[test]
fn face_cd_a_radial_three_edge_is_a_seam_once_any_loop_layer_exists() {
    let (mut mesh, spine) = book_spine_three_pages();
    assert!(
        !mesh.edge_is_interp_seam(spine),
        "the fixture starts contiguous, or the layer is not what flips it",
    );

    mesh.register_loop_attribute(ElementAttributeDesc::string(AttributeKey::custom(
        "l_label",
    )))
    .expect("a loop layer with no interpolation math");

    assert!(
        mesh.edge_is_interp_seam(spine),
        "a third radial loop makes the corner pairing impossible, which is a seam on that \
         ground alone",
    );
}

/// §6.6.1.4 scopes the strip override to the odd-`segments` centre quad **and the lone
/// quad at `segments = 1`**. The single quad is a straddle quad, so at a seamed edge it
/// takes the tie-break winner — here the far flank, on key 3 — rather than the near
/// flank the positional rule alone would give it.
#[test]
fn face_cd_single_segment_strip_quad_takes_the_tie_break_winner_at_a_seamed_edge() {
    let mut mesh = LoopMesh::new();
    mesh_add_cube(&mut mesh);
    let tint = tint_and_slot_faces(&mut mesh);
    let (edge, _near_block, far_block) = inverted_slot_strip_fixture(&mut mesh, tint);

    let color = mesh
        .register_loop_attribute(ElementAttributeDesc::color(
            AttributeKey::custom("l_color"),
            AttrType::Vec4,
        ))
        .unwrap();
    let near_face = mesh.edge_flanking_faces(edge).0.unwrap();
    let near_center = mesh.face_center_of_mass(near_face);
    for corner in mesh.face_loop_keys(near_face) {
        mesh.loop_attribs_mut()
            .set_vec4(color, corner.as_usize(), [1.0, 0.0, 0.0, 1.0]);
    }

    let result = mesh.bevel(
        &[edge.into()],
        BevelOptions {
            offset: 0.25,
            affect: BevelAffect::Edges,
            segments: 1,
            material: -1,
            ..Default::default()
        },
    );
    let blocks = strip_blocks_in_centre_line_order(&mesh, &result, near_center, tint);
    assert_eq!(
        blocks,
        vec![far_block],
        "the lone `segments = 1` quad is a straddle quad and re-routes at a seam",
    );
}

/// The regression the `segments = 1` re-route must not swallow: on a **contiguous**
/// edge the straddle donor answers the near flank, which is exactly the positional
/// donor that path took before the override reached it. Same fixture, colour layer
/// present but agreeing across the edge.
#[test]
fn face_cd_single_segment_strip_quad_keeps_the_near_flank_when_the_edge_is_contiguous() {
    let mut mesh = LoopMesh::new();
    mesh_add_cube(&mut mesh);
    let tint = tint_and_slot_faces(&mut mesh);
    let (edge, near_block, _far_block) = inverted_slot_strip_fixture(&mut mesh, tint);

    mesh.register_loop_attribute(ElementAttributeDesc::color(
        AttributeKey::custom("l_color"),
        AttrType::Vec4,
    ))
    .unwrap();
    let near_center = mesh.face_center_of_mass(mesh.edge_flanking_faces(edge).0.unwrap());

    let result = mesh.bevel(
        &[edge.into()],
        BevelOptions {
            offset: 0.25,
            affect: BevelAffect::Edges,
            segments: 1,
            material: -1,
            ..Default::default()
        },
    );
    let blocks = strip_blocks_in_centre_line_order(&mesh, &result, near_center, tint);
    assert_eq!(
        blocks,
        vec![near_block],
        "a contiguous edge keeps the positional near flank at `segments = 1`",
    );
}

/// §6.6.1.4.0(a)'s closed list is keyed on layer **type**, never on the crate's own
/// interpolation policy — and the byte-RGBA row is where the two readings come apart in
/// the *under*-seaming direction. A four-byte colour layer declares
/// [`InterpPolicy::NoInterp`] here, yet the guide puts it squarely in the blendable set,
/// so a byte-colour-only discontinuity **is** a condition-(2) seam and the straddle quad
/// re-routes to the tie-break winner. A policy-derived membership test would leave the
/// strip positional.
#[test]
fn face_cd_a_byte_colour_only_discontinuity_seams_the_edge() {
    let mut mesh = LoopMesh::new();
    mesh_add_cube(&mut mesh);
    let tint = tint_and_slot_faces(&mut mesh);
    let (edge, near_block, far_block) = inverted_slot_strip_fixture(&mut mesh, tint);

    let opaque = mesh
        .register_loop_attribute(ElementAttributeDesc::numeric(
            AttributeKey::custom("l_byte_color"),
            AttrType::Vec4U8,
        ))
        .unwrap();
    assert_eq!(
        mesh.loop_attribs()
            .descriptor(opaque)
            .expect("the layer was just registered")
            .interp,
        InterpPolicy::NoInterp,
        "the fixture is only sharp while the byte layer declares no interpolation",
    );
    let near_face = mesh.edge_flanking_faces(edge).0.unwrap();
    let near_center = mesh.face_center_of_mass(near_face);
    assert!(
        !mesh.edge_is_interp_seam(edge),
        "an agreeing byte colour layer leaves the edge contiguous",
    );
    for corner in mesh.face_loop_keys(near_face) {
        mesh.loop_attribs_mut()
            .set_vec4u8(opaque, corner.as_usize(), [255, 0, 0, 255]);
    }
    assert!(
        mesh.edge_is_interp_seam(edge),
        "byte RGBA is blendable by type, so the discontinuity seams the edge",
    );

    let result = mesh.bevel(
        &[edge.into()],
        BevelOptions {
            offset: 0.25,
            affect: BevelAffect::Edges,
            segments: 3,
            material: -1,
            ..Default::default()
        },
    );
    let blocks = strip_blocks_in_centre_line_order(&mesh, &result, near_center, tint);
    assert_eq!(
        blocks,
        vec![near_block, far_block, far_block],
        "the seamed edge's centre quad takes the tie-break winner {far_block:?}",
    );
}

/// The *over*-seaming direction at the same call site: the four families §6.6.1.4.0(a)
/// leaves off the closed list — scalar float, 3-vector float, 32-bit integer, boolean —
/// are never read, however discontinuous. The 32-bit integer and the boolean are the
/// sharp pair: both **are** interpolable (they define a rounded sum and a disjunction),
/// so an "interpolable ⇒ consulted" membership test would seam here and re-route the
/// centre quad. The strip must stay positional: near, near, far.
#[test]
fn face_cd_a_non_blendable_loop_layer_discontinuity_never_seams_the_edge() {
    type Paint = dyn Fn(&mut LoopMesh, ElementAttributeKey, LoopKey);
    let cases: [(&str, AttrType, &Paint); 4] = [
        (
            "l_scalar",
            AttrType::F32,
            &|mesh: &mut LoopMesh, layer, corner: LoopKey| {
                mesh.loop_attribs_mut()
                    .set_f32(layer, corner.as_usize(), 7.0);
            },
        ),
        ("l_tangent", AttrType::Vec3, &|mesh, layer, corner| {
            mesh.loop_attribs_mut()
                .set_vec3(layer, corner.as_usize(), [7.0, 7.0, 7.0]);
        }),
        ("l_group", AttrType::I32, &|mesh, layer, corner| {
            mesh.loop_attribs_mut().set_i32(layer, corner.as_usize(), 7);
        }),
        ("l_mask", AttrType::Bool, &|mesh, layer, corner| {
            mesh.loop_attribs_mut()
                .set_bool(layer, corner.as_usize(), true);
        }),
    ];

    for (name, attr_type, paint) in cases {
        let mut mesh = LoopMesh::new();
        mesh_add_cube(&mut mesh);
        let tint = tint_and_slot_faces(&mut mesh);
        let (edge, near_block, far_block) = inverted_slot_strip_fixture(&mut mesh, tint);

        let layer = mesh
            .register_loop_attribute(ElementAttributeDesc::numeric(
                AttributeKey::custom(name),
                attr_type,
            ))
            .unwrap();
        let near_face = mesh.edge_flanking_faces(edge).0.unwrap();
        let near_center = mesh.face_center_of_mass(near_face);
        for corner in mesh.face_loop_keys(near_face) {
            paint(&mut mesh, layer, corner);
        }
        assert!(
            !mesh.edge_is_interp_seam(edge),
            "{name} is not a blendable type, so condition (2) never reads it",
        );

        let result = mesh.bevel(
            &[edge.into()],
            BevelOptions {
                offset: 0.25,
                affect: BevelAffect::Edges,
                segments: 3,
                material: -1,
                ..Default::default()
            },
        );
        let blocks = strip_blocks_in_centre_line_order(&mesh, &result, near_center, tint);
        assert_eq!(
            blocks,
            vec![near_block, near_block, far_block],
            "{name}: an unseamed edge keeps the positional split",
        );
    }
}

/// §6.6.1.4.0(b) at the **edge-level** call site: the compare is the tolerant summed
/// squared difference, not exact equality. A sub-threshold float2 gap across the beveled
/// edge leaves it contiguous where an exact compare would seam it; widening the same gap
/// past the threshold does seam it. Both call sites of condition (2) answer alike,
/// asserted here on one mesh.
#[test]
fn face_cd_a_sub_tolerance_float2_gap_does_not_seam_the_edge() {
    for (gap, expected_seam) in [(1e-3_f32, false), (4e-3_f32, true)] {
        let mut mesh = LoopMesh::new();
        mesh_add_cube(&mut mesh);
        let tint = tint_and_slot_faces(&mut mesh);
        let (edge, _near_block, _far_block) = inverted_slot_strip_fixture(&mut mesh, tint);

        let uv = mesh
            .register_loop_attribute(ElementAttributeDesc::uv(AttributeKey::custom("uv")))
            .unwrap();
        let near_face = mesh.edge_flanking_faces(edge).0.unwrap();
        for corner in mesh.face_loop_keys(near_face) {
            mesh.loop_attribs_mut()
                .set_vec2(uv, corner.as_usize(), [gap, 0.0]);
        }

        assert_eq!(
            mesh.edge_is_interp_seam(edge),
            expected_seam,
            "gap {gap}: summed squared difference against the float2 threshold 1e-5",
        );

        let far_face = mesh.edge_flanking_faces(edge).1.unwrap();
        let flanks_share = TemplateBank::new(&mesh, 3)
            .interp_components()
            .expect("a float2 layer at odd `segments` arms the partition")
            .all_share_component(&[near_face, far_face]);
        assert_eq!(
            flanks_share, !expected_seam,
            "gap {gap}: the partition's edges run the same predicate as the seam test",
        );
    }
}

/// §6.6.1.4.0(a)'s float2 arming gate is a **type** test over the layer inventory, with
/// no role predicate anywhere in it. Positive direction: a float2 loop layer that is
/// *not* roled as a UV map still arms §6.6.1.4.1's partition. Gating on the active-UV
/// query instead would leave this case armed only through that query's fallback branch,
/// and would silently un-arm if the query ever required the role.
#[test]
fn a_non_uv_roled_float2_layer_arms_the_partition() {
    let (mut mesh, _shared) = two_quad_strip();
    let layer = mesh
        .register_loop_attribute(ElementAttributeDesc::numeric(
            AttributeKey::custom("original_space"),
            AttrType::Vec2,
        ))
        .expect("Vec2 is a supported loop attribute format");
    assert_ne!(
        mesh.loop_attribs()
            .descriptor(layer)
            .expect("the layer was just registered")
            .role,
        AttrRole::Uv,
        "the fixture only tests the type gate while the layer carries no UV role",
    );

    assert!(
        InterpComponents::live(&mesh, 3),
        "a float2 loop layer arms key 1 at odd `segments` whatever its role",
    );
    assert!(
        TemplateBank::new(&mesh, 3).interp_components().is_some(),
        "the armed gate builds the partition",
    );
}

/// BEVEL.md §6.6, both stages at once: an explicit `material >= 0` overwrites the
/// **slot field alone** of an inherited block. The stamp is not an alternative to the
/// inherit — the inherit runs first, unconditionally — so every new face comes out
/// carrying the stamped slot *over its donor's other layers*, never over a default
/// block.
#[test]
fn face_cd_explicit_material_overwrites_the_slot_of_an_inherited_block() {
    let mut mesh = LoopMesh::new();
    mesh_add_cube(&mut mesh);
    let tint = tint_and_slot_faces(&mut mesh);

    let corner = vert_at(&mesh, Vec3::new(1.0, 1.0, 1.0));
    let candidate_tints: Vec<[f32; 3]> = mesh
        .vert_face_keys(corner)
        .iter()
        .map(|&face| mesh.face_attribs().get_vec3(tint, face.as_usize()).unwrap())
        .collect();

    let result = mesh.bevel(
        &[corner.into()],
        BevelOptions {
            offset: 0.25,
            material: 3,
            ..Default::default()
        },
    );

    for (&face, (slot, value)) in result
        .faces
        .iter()
        .zip(face_blocks(&mesh, tint, &result.faces))
    {
        assert_eq!(
            slot, 3,
            "stage 2 stamps the slot of every new face, donor slots notwithstanding",
        );
        assert!(
            candidate_tints.contains(&value),
            "new face {face:?} carries tint {value:?}: the stamp replaced the whole \
             inherited block instead of its slot field — stage 1 must run at every \
             value of `material` (§6.6). Candidates: {candidate_tints:?}",
        );
    }
}

/// BEVEL.md §6.6.2 invariant #1 — the sharpest test of the two-stage rule: bevel the
/// same input at `material = -1` and at `material = k >= 0`; the new faces' per-face
/// blocks must agree on **every layer except the slot**. A "stamp instead of inherit"
/// implementation zeroes the non-slot layers under `k` and fails here, while passing
/// any slot-only check.
///
/// Run over the vertex cap, an edge strip + junction fill (donor-fed), and the cutoff
/// fill (template-less — whose non-slot layers are the *defaults* under both signs,
/// which is still agreement).
#[test]
fn face_cd_non_slot_layers_are_material_independent() {
    fn new_face_tints(
        spokes: fn(&LoopMesh) -> Vec<GeomKey>,
        options: BevelOptions<'static>,
    ) -> Vec<[u32; 3]> {
        let mut mesh = LoopMesh::new();
        mesh_add_cube(&mut mesh);
        let tint = tint_and_slot_faces(&mut mesh);
        let input = spokes(&mesh);
        let result = mesh.bevel(&input, options);
        mesh.validate_topology();
        let mut tints: Vec<[u32; 3]> = result
            .faces
            .iter()
            .map(|&face| {
                let value = mesh.face_attribs().get_vec3(tint, face.as_usize()).unwrap();
                [value[0].to_bits(), value[1].to_bits(), value[2].to_bits()]
            })
            .collect();
        tints.sort_unstable();
        tints
    }

    let corner_vert: fn(&LoopMesh) -> Vec<GeomKey> =
        |mesh| vec![vert_at(mesh, Vec3::new(1.0, 1.0, 1.0)).into()];
    let corner_spokes: fn(&LoopMesh) -> Vec<GeomKey> = |mesh| {
        mesh.vert_edge_keys(vert_at(mesh, Vec3::new(1.0, 1.0, 1.0)))
            .into_iter()
            .map(GeomKey::from)
            .collect()
    };

    let cases: [(&str, fn(&LoopMesh) -> Vec<GeomKey>, BevelOptions<'static>); 3] = [
        (
            "vertex cap",
            corner_vert,
            BevelOptions {
                offset: 0.25,
                ..Default::default()
            },
        ),
        (
            "edge strips + junction patch",
            corner_spokes,
            BevelOptions {
                offset: 0.25,
                affect: BevelAffect::Edges,
                ..Default::default()
            },
        ),
        (
            "cutoff fill",
            corner_spokes,
            BevelOptions {
                offset: 0.25,
                affect: BevelAffect::Edges,
                segments: 2,
                vmesh_method: BevelVmeshMethod::Cutoff,
                ..Default::default()
            },
        ),
    ];

    for (site, spokes, options) in cases {
        let inherited = new_face_tints(
            spokes,
            BevelOptions {
                material: -1,
                ..options
            },
        );
        let stamped = new_face_tints(
            spokes,
            BevelOptions {
                material: 6,
                ..options
            },
        );
        assert_eq!(
            inherited, stamped,
            "{site}: the new faces' non-slot per-face layers must not depend on \
             `material` (§6.6.2 invariant #1)",
        );
    }
}

/// BEVEL.md §6.6.1 / §6.6.2: the single-segment **junction patch** (§7.6.3) inherits
/// the whole block of a junction-adjacent original, like every other new face of that
/// junction. The donor must be captured while the junction still *has* faces — the
/// re-stitch consumes the vertex, so a source read after it names nothing at all and
/// the patch would silently fall back to the layer defaults.
#[test]
fn face_cd_junction_patch_inherits_whole_block_of_a_junction_face() {
    let mut mesh = LoopMesh::new();
    mesh_add_cube(&mut mesh);
    let tint = tint_and_slot_faces(&mut mesh);
    let (spokes, donors) = cube_corner_junction(&mesh, tint);

    let result = mesh.bevel(
        &spokes,
        BevelOptions {
            offset: 0.25,
            affect: BevelAffect::Edges,
            material: -1,
            ..Default::default()
        },
    );
    mesh.validate_topology();
    assert_no_boundary_edges(&mesh, "s = 3 junction patch");

    // At `segments = 1` every strip is a quad over four distinct near corners, so the
    // run's one triangle is the ring's patch face.
    let patch = *result
        .faces
        .iter()
        .find(|&&face| mesh.face_vert_keys(face).len() == 3)
        .expect("the s = 3 junction is capped by one triangular patch");
    assert_new_faces_inherit_from(&mesh, tint, &[patch], &donors, "the junction patch");
    assert_new_faces_inherit_from(&mesh, tint, &result.faces, &donors, "s = 3 junction");
}

/// BEVEL.md §6.6.1.3: the **cutoff fill is the one fill that supplies no donor**, so
/// its cut faces mint template-less — the layer defaults, slot `0` — while every other
/// new face of the same junction (the strips, whose donor is the flanking face) still
/// inherits a whole block. The junction's originals are right there and the ADJ fill
/// at the same corner uses them; the cutoff fill simply does not pass one.
///
/// The cube corner's three cut faces are the whole template-less set here: the curved
/// default profile pushes the three corner verts onto one apex (§8.3.4), so no bottom
/// face is built. `face_cd_cutoff_bottom_face_is_template_less_too` covers the bottom.
#[test]
fn face_cd_cutoff_cut_faces_are_template_less() {
    let mut mesh = LoopMesh::new();
    mesh_add_cube(&mut mesh);
    let tint = tint_and_slot_faces(&mut mesh);
    let (spokes, donors) = cube_corner_junction(&mesh, tint);

    let result = mesh.bevel(
        &spokes,
        BevelOptions {
            offset: 0.25,
            affect: BevelAffect::Edges,
            segments: 2,
            vmesh_method: BevelVmeshMethod::Cutoff,
            material: -1,
            ..Default::default()
        },
    );
    mesh.validate_topology();
    assert_no_boundary_edges(&mesh, "cutoff fill");
    assert_new_face_blocks(&mesh, tint, &result.faces, &donors, 3, "cutoff fill");
}

/// BEVEL.md §6.6.1.3: the cutoff fill's **bottom face** over the corner ring is
/// template-less as well, not just the cut faces. A valence-4 junction does not
/// collapse (§8.3.4 is a 3-way rule), so its ring is closed by a real bottom face:
/// four sectors but **five** default-block new faces is what proves the bottom face is
/// among them.
#[test]
fn face_cd_cutoff_bottom_face_is_template_less_too() {
    let mut mesh = LoopMesh::new();
    let (top, rim, _bottom) = square_bipyramid(&mut mesh);
    let tint = tint_and_slot_faces(&mut mesh);
    let donors = face_blocks(&mesh, tint, &mesh.face_keys());

    let spokes: Vec<GeomKey> = rim
        .iter()
        .map(|&r| mesh.query_edge_exists(top, r).expect("top spoke").into())
        .collect();
    let result = mesh.bevel(
        &spokes,
        BevelOptions {
            offset: 0.25,
            affect: BevelAffect::Edges,
            segments: 2,
            vmesh_method: BevelVmeshMethod::Cutoff,
            material: -1,
            ..Default::default()
        },
    );
    mesh.validate_topology();
    assert_no_boundary_edges(&mesh, "cutoff apex");
    assert_new_face_blocks(&mesh, tint, &result.faces, &donors, 5, "cutoff apex");
}

/// BEVEL.md §6.6.1.3's bottom row: stage 2 still runs on a template-less face. Under
/// `material >= 0` the cutoff fill's faces come out at the stamped slot over a
/// **default** block — the cutoff does not "ignore `material`" — while the donor-fed
/// faces of the same run come out at the stamped slot over their **donor's** block.
#[test]
fn face_cd_template_less_fill_takes_the_stamp_over_a_default_block() {
    let mut mesh = LoopMesh::new();
    mesh_add_cube(&mut mesh);
    let tint = tint_and_slot_faces(&mut mesh);
    let (spokes, donors) = cube_corner_junction(&mesh, tint);
    let donor_tints: Vec<[f32; 3]> = donors.iter().map(|&(_, tint)| tint).collect();

    let result = mesh.bevel(
        &spokes,
        BevelOptions {
            offset: 0.25,
            affect: BevelAffect::Edges,
            segments: 2,
            vmesh_method: BevelVmeshMethod::Cutoff,
            material: 7,
            ..Default::default()
        },
    );
    mesh.validate_topology();

    let mut template_less = 0;
    for (&face, (slot, value)) in result
        .faces
        .iter()
        .zip(face_blocks(&mesh, tint, &result.faces))
    {
        assert_eq!(
            slot, 7,
            "stage 2 stamps every new face's slot, face {face:?}"
        );
        if value == TEMPLATE_LESS_BLOCK.1 {
            template_less += 1;
        } else {
            assert!(
                donor_tints.contains(&value),
                "donor-fed face {face:?} kept a foreign tint {value:?} under the stamp",
            );
        }
    }
    assert_eq!(
        template_less, 3,
        "the three cut faces carry the stamped slot over a *default* block (§6.6.1.3)",
    );
}

/// BEVEL.md §6.6.3 over the **[V-ADJ] fill** (§8.5): the `segments = 2` cube-corner
/// pinwheel and the `segments = 3` unit-octant grid alike. Every grid face is a new
/// face, so every one of them inherits a junction-adjacent original's whole block.
#[test]
fn face_cd_adj_fill_inherits_whole_block_of_a_junction_face() {
    for segments in [2, 3] {
        let mut mesh = LoopMesh::new();
        mesh_add_cube(&mut mesh);
        let tint = tint_and_slot_faces(&mut mesh);
        let (spokes, donors) = cube_corner_junction(&mesh, tint);

        let result = mesh.bevel(
            &spokes,
            BevelOptions {
                offset: 0.25,
                affect: BevelAffect::Edges,
                segments,
                vmesh_method: BevelVmeshMethod::Adj,
                material: -1,
                ..Default::default()
            },
        );
        mesh.validate_topology();
        let site = format!("ADJ fill, segments = {segments}");
        assert_no_boundary_edges(&mesh, &site);
        assert_new_faces_inherit_from(&mesh, tint, &result.faces, &donors, &site);
    }
}

/// BEVEL.md §6.6.1 at the closed solid's two remaining fill sites: the **general
/// valence-`s` fill** over the beveled apex (valence 4, so neither the tri-corner snap
/// nor the cutoff's collapsed apex) and every rim vertex's **valence-4 terminal
/// poly-fan** (§8.3.0.2), whose donor is likewise registered before the re-stitch
/// consumes the terminal. Run under both `vmesh_method`s, since the fan is
/// vmesh-independent. Donor *locality* is pinned by the cube-corner tests above; what
/// this pins is that no fill or fan face falls back to the layer default.
#[test]
fn face_cd_general_fill_and_terminal_fan_inherit_original_blocks() {
    for vmesh_method in [BevelVmeshMethod::Cutoff, BevelVmeshMethod::Adj] {
        let mut mesh = LoopMesh::new();
        let (top, rim, _bottom) = square_bipyramid(&mut mesh);
        let tint = tint_and_slot_faces(&mut mesh);
        let donors = face_blocks(&mesh, tint, &mesh.face_keys());

        let spokes: Vec<GeomKey> = rim
            .iter()
            .map(|&r| mesh.query_edge_exists(top, r).expect("top spoke").into())
            .collect();
        let result = mesh.bevel(
            &spokes,
            BevelOptions {
                offset: 0.25,
                affect: BevelAffect::Edges,
                segments: 2,
                vmesh_method,
                material: -1,
                ..Default::default()
            },
        );
        mesh.validate_topology();
        let site = format!("apex fill + terminal fans, {vmesh_method:?}");
        assert_no_boundary_edges(&mesh, &site);
        // The cutoff apex mints its four cut faces and its bottom face template-less
        // (§6.6.1.3); every other new face of either run — fans included — inherits.
        let template_less = match vmesh_method {
            BevelVmeshMethod::Cutoff => 5,
            BevelVmeshMethod::Adj => 0,
        };
        assert_new_face_blocks(&mesh, tint, &result.faces, &donors, template_less, &site);
    }
}

/// BEVEL.md §6.6.1 at the **mitered** junction (§8.9.5.3): the enlarged ring's [V-ADJ]
/// fill is new geometry like any other, and its donor is registered when the ring is
/// placed. The `Sharp` run alongside is what proves the arc miter actually engaged —
/// it splits the reflex sector, so it mints strictly more verts than the sharp ring —
/// without which this would be pinning the plain general fill under a miter's name.
#[test]
fn face_cd_miter_ring_fill_inherits_whole_block_of_a_junction_face() {
    let miter_bevel = |kind: BevelMiter| {
        let (mut mesh, apex, [va, vb, vc]) = single_face_reflex_junction();
        let tint = tint_and_slot_faces(&mut mesh);
        let donors = face_blocks(&mesh, tint, &mesh.vert_face_keys(apex));
        let result = mesh.bevel(
            &[va.into(), vb.into(), vc.into()],
            BevelOptions {
                offset: 0.2,
                affect: BevelAffect::Edges,
                segments: 2,
                vmesh_method: BevelVmeshMethod::Adj,
                miter_outer: kind,
                material: -1,
                ..Default::default()
            },
        );
        (mesh, tint, donors, result)
    };

    let (_, _, _, sharp) = miter_bevel(BevelMiter::Sharp);
    let (mesh, tint, donors, result) = miter_bevel(BevelMiter::Arc);
    assert_eq!(donors.len(), 3, "the reflex apex has three incident faces");
    assert!(
        result.verts.len() > sharp.verts.len(),
        "the arc miter did not engage — splitting the reflex sector mints more ring \
         verts than the sharp ring ({} vs {})",
        result.verts.len(),
        sharp.verts.len(),
    );

    mesh.validate_topology();
    assert_no_boundary_edges(&mesh, "outer-miter ring fill");
    assert_new_faces_inherit_from(&mesh, tint, &result.faces, &donors, "outer-miter ring fill");
}

// ---- §5.2: per-vertex customdata on the new verts ----
