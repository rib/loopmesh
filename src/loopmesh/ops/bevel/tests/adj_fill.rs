use super::*;

/// Sanity: the closed-form Sabin weight reproduces the guide's tabulated constants
/// (BEVEL.md §8.5.5.5(c)) at the valences the fill pins — including the valence-4
/// exact `0.25` the outer-miter `s = 3` cap uses.
#[test]
fn sabin_gamma_matches_guide_table() {
    assert!((sabin_gamma_formula(3) - 0.065_247_584).abs() < 1.0e-6);
    assert!((sabin_gamma_formula(4) - 0.25).abs() < 1.0e-6);
    assert!((sabin_gamma_formula(5) - 0.401_983_447).abs() < 1.0e-6);
    assert!((sabin_gamma_formula(6) - 0.523_423_277).abs() < 1.0e-6);
    assert_eq!(sabin_gamma(4), 0.25);
}

/// BEVEL.md §8.5.2 / §8.5.3: the `s = 3`, `segments = 2` cube-corner ADJ junction
/// rounds the corner with a three-quad pinwheel meeting at one welded central
/// apex — no bottom face. The whole cube grows 8 V / 12 E / 6 F to 20 V / 33 E /
/// 15 F (§8.5.3), and the result stays a closed manifold.
#[test]
fn adj_junction_segments2_topology() {
    let (mesh, result) = cube_corner_adj_segments2();

    mesh.validate_topology();

    assert_eq!(
        counts(&mesh),
        (20, 33, 15),
        "cube-corner ADJ seg=2 grows the cube to 20 V / 33 E / 15 F (§8.5.3)",
    );

    // §8.5.2 closed-form apex: the body-diagonal direction snapped onto the
    // rounded surface (the sphere through the ring + chain verts, here centre
    // (0.75,-0.75,-0.75), radius = offset 0.25). The apex is that sphere's point
    // nearest the corner along the body diagonal: centre + offset·(1,-1,-1)/√3,
    // i.e. ≈ (0.894, -0.894, -0.894) — on the dome, not pushed into the solid.
    let sphere_center = Vec3::new(0.75, -0.75, -0.75);
    let apex_position = sphere_center + 0.25 * Vec3::new(1.0, -1.0, -1.0) / 3.0_f32.sqrt();
    let apex: Vec<VertexKey> = result
        .verts
        .iter()
        .copied()
        .filter(|&v| (mesh.vertex_position(v) - apex_position).length() < 1.0e-4)
        .collect();
    assert_eq!(
        apex.len(),
        1,
        "the three pinwheel apex corners weld to one apex"
    );
    let apex = apex[0];

    // §8.5.3: exactly three quad faces meet at the apex (the pinwheel).
    let pinwheel: Vec<FaceKey> = mesh
        .face_keys()
        .into_iter()
        .filter(|&f| mesh.face_vert_keys(f).contains(&apex))
        .collect();
    assert_eq!(pinwheel.len(), 3, "three pinwheel quads share the apex");
    for &face in &pinwheel {
        assert_eq!(
            mesh.face_vert_keys(face).len(),
            4,
            "each pinwheel cell is a quad (§8.5.3)",
        );
    }

    // Closed manifold: every edge radial-2. The new geometry is 6 strip quads +
    // 3 pinwheel quads = 9 new faces; with the cube's 6 originals that is 15.
    for face in mesh.face_keys() {
        for loop_key in mesh.face_loop_keys(face) {
            let edge = mesh.loop_edge(loop_key);
            assert_eq!(
                mesh.edge_radial_iter(edge).count(),
                2,
                "every edge is shared with exactly one other face",
            );
        }
    }
    assert_eq!(
        result.faces.len(),
        9,
        "6 strip quads + 3 pinwheel quads, no bottom"
    );
}

/// BEVEL.md §8.5.3: the ADJ apex sits *on* the rounded surface (the body-diagonal
/// point snapped onto the dome, §8.5.2), distinct from the cutoff apex which is
/// pushed *down into* the solid (§8.3.4). The two fills share the same boundvert
/// ring and chains but place the central vert at different points.
#[test]
fn adj_apex_differs_from_cutoff_apex() {
    let (adj_mesh, adj_result) = cube_corner_adj_segments2();
    let (cut_mesh, _cut_result) = cube_corner_cutoff_segments2();

    let sphere_center = Vec3::new(0.75, -0.75, -0.75);
    let adj_apex = sphere_center + 0.25 * Vec3::new(1.0, -1.0, -1.0) / 3.0_f32.sqrt();
    let cutoff_apex = Vec3::new(0.75, -0.75, -0.75);

    // The ADJ apex lands on the rounded surface (snapped body-diagonal point)...
    assert_eq!(
        adj_result
            .verts
            .iter()
            .filter(|&&v| (adj_mesh.vertex_position(v) - adj_apex).length() < 1.0e-4)
            .count(),
        1,
    );
    // ...which is a different position bucket from the cutoff apex.
    assert!(
        (adj_apex - cutoff_apex).length() > 1.0e-2,
        "ADJ apex (rounded surface) differs from cutoff apex (pushed down)",
    );
    // And no ADJ vert lands at the cutoff apex.
    assert_eq!(
        adj_result
            .verts
            .iter()
            .filter(|&&v| (adj_mesh.vertex_position(v) - cutoff_apex).length() < 1.0e-4)
            .count(),
        0,
    );
    // Sanity: the cutoff fixture still has its own apex where expected.
    assert!(cut_mesh
        .vert_keys()
        .into_iter()
        .any(|v| (cut_mesh.vertex_position(v) - cutoff_apex).length() < 1.0e-4));
}

/// BEVEL.md §8.5.2: the ADJ junction reuses the same segment-count-independent
/// boundvert ring (§8.2.1) and the same arc-sampled interior chain verts
/// (§8.3.6) as cutoff — only the junction fill differs. The three boundverts sit
/// offset·√2 from the corner and the interior chain verts sit at the arc 45°
/// point (radius offset from the per-cell arc center), exactly as cutoff.
#[test]
fn adj_segments2_ring_and_chain_match_cutoff() {
    let (mesh, result) = cube_corner_adj_segments2();
    let corner = Vec3::new(1.0, -1.0, -1.0);
    let expected = 0.25 * 2.0_f32.sqrt();

    let ring_verts = result
        .verts
        .iter()
        .filter(|&&v| ((mesh.vertex_position(v) - corner).length() - expected).abs() < 1.0e-4)
        .count();
    assert_eq!(
        ring_verts, 3,
        "three §7.6.2 boundverts, unchanged by vmesh method"
    );

    // The Front-Bottom interior chain vert across edge {0,1} is the arc 45° point
    // (§8.3.6), shared with cutoff — not the straight chord midpoint.
    let arc_point = Vec3::new(0.75, -0.926_776_7, -0.926_776_7);
    let chord_midpoint = Vec3::new(0.75, -0.875, -0.875);
    assert_eq!(
        result_has_vert_near(&mesh, &result, arc_point),
        1,
        "interior chain vert is the arc 45° sample, shared with cutoff",
    );
    assert_eq!(
        result_has_vert_near(&mesh, &result, chord_midpoint),
        0,
        "no interior chain vert at the straight chord midpoint",
    );
}

/// Paint the three faces meeting cube corner `(1,-1,-1)` with distinct material
/// slots (Front=1, Left=2, Bottom=3) so each of the corner's three beveled legs is
/// genuinely two-material and each sector's **own** boundvert rep is observable in
/// the fill (BEVEL.md §6.6.2's [V-ADJ] grid row). Returns the corner and the three
/// beveled-leg edges, plus the per-leg **own-rep material** (`boundvert_representative`
/// read on the pristine mesh, exactly as the fill's donor capture does).
fn two_material_corner0_fan(mesh: &mut LoopMesh) -> (VertexKey, [EdgeKey; 3], [i32; 3]) {
    let v0 = vert_at(mesh, Vec3::new(1.0, -1.0, -1.0));
    let mut slot = 1;
    for face in mesh.face_keys() {
        if mesh.face_vert_keys(face).contains(&v0) {
            mesh.set_face_material_index(face, slot);
            slot += 1;
        }
    }
    let legs = [
        cube_edge(
            mesh,
            Vec3::new(1.0, -1.0, -1.0),
            Vec3::new(-1.0, -1.0, -1.0),
        ),
        cube_edge(mesh, Vec3::new(1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, -1.0)),
        cube_edge(mesh, Vec3::new(1.0, -1.0, -1.0), Vec3::new(1.0, -1.0, 1.0)),
    ];
    let own_rep =
        legs.map(|edge| mesh.face_material_index(mesh.boundvert_representative(v0, edge).unwrap()));
    (v0, legs, own_rep)
}

/// BEVEL.md §6.6.2's [V-ADJ] grid row (edge bevel): each sector's fill faces inherit
/// their whole per-face block — material slot included — from **that sector's own**
/// boundvert rep, not from one corner-wide donor. At **even** `segments` there is no
/// centre column, so the whole grid is per-sector: the three `segments = 2` pinwheel
/// cells split three-ways across the corner's three own reps. A regression to one
/// corner-wide donor collapses them onto a single slot.
#[test]
fn edge_bevel_junction_grid_donor_is_per_sector_own_rep_segments_2() {
    let mut mesh = LoopMesh::new();
    mesh_add_cube(&mut mesh);
    let (_, legs, own_rep) = two_material_corner0_fan(&mut mesh);

    let mut expected = own_rep.to_vec();
    expected.sort_unstable();
    assert!(
        expected.first() != expected.last(),
        "fixture sanity: the three own reps must span ≥2 slots to discriminate a collapse",
    );

    let result = mesh.bevel(
        &[legs[0].into(), legs[1].into(), legs[2].into()],
        BevelOptions {
            offset: 0.25,
            affect: BevelAffect::Edges,
            segments: 2,
            vmesh_method: BevelVmeshMethod::Adj,
            ..Default::default()
        },
    );
    mesh.validate_topology();

    let fill = adj_patch_faces(&mesh, &result);
    let mut fill_slots: Vec<i32> = fill.iter().map(|&f| mesh.face_material_index(f)).collect();
    fill_slots.sort_unstable();
    assert_eq!(
        fill_slots, expected,
        "the three welded-apex cells carry the three sectors' own reps — a corner-wide \
         donor would give one slot repeated three times",
    );
}

/// BEVEL.md §6.6.2's odd-`segments` centre-column row (edge bevel): the sector's centre
/// column — the bridge strip meeting the centre n-gon — takes the sector's **own** rep
/// when its boundvert has a non-seam beveled edge, exactly as the sector's non-centre
/// grid does. On the seam-free two-material cube corner every leg qualifies, so at
/// `segments = 3` all six sector quads (one interior + one centre-column bridge per
/// sector) carry that sector's own rep — each own rep twice. A regression that left the
/// centre column on a corner-wide donor, or ran the tie-break on a non-seam edge, would
/// desync the pairing.
#[test]
fn edge_bevel_junction_centre_column_donor_is_own_rep_segments_3() {
    let mut mesh = LoopMesh::new();
    mesh_add_cube(&mut mesh);
    let (_, legs, own_rep) = two_material_corner0_fan(&mut mesh);

    let mut expected: Vec<i32> = own_rep.iter().flat_map(|&m| [m, m]).collect();
    expected.sort_unstable();
    assert!(
        expected.first() != expected.last(),
        "fixture sanity: the own reps must span ≥2 slots to discriminate a collapse",
    );

    let result = mesh.bevel(
        &[legs[0].into(), legs[1].into(), legs[2].into()],
        BevelOptions {
            offset: 0.25,
            affect: BevelAffect::Edges,
            segments: 3,
            vmesh_method: BevelVmeshMethod::Adj,
            ..Default::default()
        },
    );
    mesh.validate_topology();

    // The odd grid is six sector quads plus one ring-spanning centre triangle; the
    // triangle takes the centre n-gon's own (ring-spanning) donor, so it is set aside.
    let mut sector_slots: Vec<i32> = adj_patch_faces(&mesh, &result)
        .into_iter()
        .filter(|&f| mesh.face_vert_keys(f).len() == 4)
        .map(|f| mesh.face_material_index(f))
        .collect();
    assert_eq!(
        sector_slots.len(),
        6,
        "six sector quads at s=3 (interior + bridge)"
    );
    sector_slots.sort_unstable();
    assert_eq!(
        sector_slots, expected,
        "each sector's interior cell and centre-column bridge both carry its own rep",
    );
}

/// The **per-sector direction** of the edge-bevel grid donor, which the two multiset
/// tests above cannot see: a cyclic shift of the sectors' own reps keeps the multiset
/// but swaps which sector receives which donor. Pin it geometrically — a boundvert-
/// anchored grid cell physically replaces sector `i`'s original wedge, so it and the
/// strip quads flanking that wedge (on B_i's side) both inherit the **same** wedge face
/// and must carry the same material. B_i *opens* sector `i` along `opening_edges[i]` but
/// *starts* the beveled leg `opening_edges[i+1]` (§6.6.1's "a boundvert that starts a
/// beveled leg takes that leg's edge"), so its own rep is `prev_face(opening_edges[i+1])`
/// = that wedge face. Reading the arriving leg instead rotates every grid cell's donor
/// off its strips and this contiguity breaks (BEVEL.md §6.6.2 [V-ADJ] grid row).
///
/// The centre-column bridge and the centre n-gon are **excluded** (they anchor on no
/// boundvert): the bridge follows the own-rep rule read at the corner, which by design
/// can differ from the straddle strip quad's stored-first-endpoint donor (§6.6.2's
/// separate strip / centre-column rows), so bridge↔strip contiguity is not expected.
fn assert_cap_strip_material_contiguous(mesh: &LoopMesh, result: &BevelResult) {
    let cap: std::collections::HashSet<FaceKey> =
        adj_patch_faces(mesh, result).into_iter().collect();
    assert!(
        !cap.is_empty(),
        "the junction cap must have grid faces to check"
    );
    let corner = Vec3::new(1.0, -1.0, -1.0);
    let ring_radius = 0.25 * 2.0_f32.sqrt();
    let boundverts: std::collections::HashSet<VertexKey> = result
        .verts
        .iter()
        .copied()
        .filter(|&v| ((mesh.vertex_position(v) - corner).length() - ring_radius).abs() < 1.0e-4)
        .collect();
    assert_eq!(
        boundverts.len(),
        3,
        "three junction boundverts anchor the grid cells"
    );
    let mut checked = 0usize;
    for &f in &cap {
        // Only the boundvert-anchored grid / pinwheel cells replace an original wedge
        // and border its strips; skip the centre-column bridge and the centre n-gon.
        if !mesh
            .face_vert_keys(f)
            .iter()
            .any(|v| boundverts.contains(v))
        {
            continue;
        }
        let fmat = mesh.face_material_index(f);
        for loop_key in mesh.face_loop_keys(f) {
            let edge = mesh.loop_edge(loop_key);
            for other in mesh.iter_radial_loops(edge) {
                let g = mesh.loop_face(other);
                // cap↔cap rim edges (apex spokes, interior grid, seam column) join two
                // sectors' distinct donors; only cap↔strip edges are contiguity edges.
                if g == f || cap.contains(&g) {
                    continue;
                }
                assert_eq!(
                    fmat,
                    mesh.face_material_index(g),
                    "a boundvert-anchored grid cell and its flanking strip inherit the \
                     same wedge face's material (BEVEL.md §6.6.2 [V-ADJ] grid row, edge \
                     mode) — a rotated sector→own-edge map breaks this",
                );
                checked += 1;
            }
        }
    }
    assert!(
        checked >= 3,
        "each of the ≥3 sectors' grid cell borders a strip — the check must run",
    );
}

/// BEVEL.md §6.6.2's [V-ADJ] grid row (edge bevel), **per-sector direction** at even
/// `segments`: each `segments = 2` pinwheel cap cell borders the two strips flanking
/// its own sector, and must inherit *that* sector's wedge face — not the previous
/// sector's (the off-by-one this pins, which the multiset test cannot).
#[test]
fn edge_bevel_junction_grid_donor_direction_segments_2() {
    let mut mesh = LoopMesh::new();
    mesh_add_cube(&mut mesh);
    let (_, legs, own_rep) = two_material_corner0_fan(&mut mesh);
    let mut sorted = own_rep.to_vec();
    sorted.sort_unstable();
    assert!(
        sorted.first() != sorted.last(),
        "fixture sanity: the own reps must span ≥2 slots to discriminate a rotation",
    );

    let result = mesh.bevel(
        &[legs[0].into(), legs[1].into(), legs[2].into()],
        BevelOptions {
            offset: 0.25,
            affect: BevelAffect::Edges,
            segments: 2,
            vmesh_method: BevelVmeshMethod::Adj,
            ..Default::default()
        },
    );
    assert_cap_strip_material_contiguous(&mesh, &result);
}

/// BEVEL.md §6.6.2's [V-ADJ] grid + centre-column rows (edge bevel), **per-sector
/// direction** at odd `segments`: the interior cell *and* the centre-column bridge of
/// each sector border that sector's strips, so both must inherit its own wedge face.
/// Pins the direction of the own-vs-next map the centre-column donor also keys on.
#[test]
fn edge_bevel_junction_grid_donor_direction_segments_3() {
    let mut mesh = LoopMesh::new();
    mesh_add_cube(&mut mesh);
    let (_, legs, own_rep) = two_material_corner0_fan(&mut mesh);
    let mut sorted = own_rep.to_vec();
    sorted.sort_unstable();
    assert!(
        sorted.first() != sorted.last(),
        "fixture sanity: the own reps must span ≥2 slots to discriminate a rotation",
    );

    let result = mesh.bevel(
        &[legs[0].into(), legs[1].into(), legs[2].into()],
        BevelOptions {
            offset: 0.25,
            affect: BevelAffect::Edges,
            segments: 3,
            vmesh_method: BevelVmeshMethod::Adj,
            ..Default::default()
        },
    );
    assert_cap_strip_material_contiguous(&mesh, &result);
}

/// BEVEL.md §6.6.2's [V-ADJ] grid row (**vertex** bevel, the off-by-one arm): each
/// sector's grid takes the **next** boundvert's rep. A cyclic shift of the sectors'
/// own reps has the same multiset, so this even-`segments` case pins that the fill is
/// per-sector distributed (not collapsed onto one corner-wide donor) and carries the
/// corner's rep multiset — the seg-3 direction test
/// (`vertex_bevel_multiseg_fill_material_distributes_per_sector`) pins the off-by-one
/// itself.
#[test]
fn vertex_bevel_junction_grid_donor_is_per_sector_segments_2() {
    let mut mesh = LoopMesh::new();
    mesh_add_cube(&mut mesh);
    let (v0, _, own_rep) = two_material_corner0_fan(&mut mesh);

    let mut expected = own_rep.to_vec();
    expected.sort_unstable();
    assert!(
        expected.first() != expected.last(),
        "fixture sanity: the reps must span ≥2 slots to discriminate a collapse",
    );

    let result = mesh.bevel(
        &[v0.into()],
        BevelOptions {
            offset: 0.25,
            affect: BevelAffect::Vertices,
            segments: 2,
            vmesh_method: BevelVmeshMethod::Adj,
            ..Default::default()
        },
    );
    mesh.validate_topology();

    // Vertex bevel mints no strips, so every new face is a cap-grid fill face.
    let mut fill_slots: Vec<i32> = result
        .faces
        .iter()
        .map(|&f| mesh.face_material_index(f))
        .collect();
    fill_slots.sort_unstable();
    assert_eq!(
        fill_slots, expected,
        "the three welded-apex cells distribute across the sectors' reps",
    );
}
