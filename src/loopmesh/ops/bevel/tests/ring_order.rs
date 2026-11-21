use super::*;

/// The §7.6.1.2 worked example's slot array, spoke by spoke, on the closed 12-fan the
/// guide states it for.
///
/// `planar_fan(12)` **is** that fixture — twelve triangles `[hub, rim_k, rim_(k+1)]`
/// created in `k` order — so the hub's stored disk order is `2, 1, 3, 4, … 12` (each
/// face-create mints its closing edge first, which attaches spoke `2` ahead of spoke
/// `1`). Spoke numbers below are the guide's, one-based: spoke `j` is `rim[j - 1]`.
///
/// Both arms pin step 1's seed *and* step 2's reversal, which no other test here can
/// separate: they differ only in which spoke the selection makes the seed, and the
/// arrays they produce are a whole ring step apart.
#[test]
fn bevel_ring_slot_array_seeds_on_the_stored_order_and_reverses_by_the_ballot() {
    let (mesh, centre, rim) = planar_fan(12);
    let spoke_edge = |spoke: usize| {
        mesh.query_edge_exists(centre, rim[spoke - 1])
            .expect("spoke exists")
    };
    let spoke_number = |edge: EdgeKey| (1..=12).find(|&spoke| spoke_edge(spoke) == edge);
    let slot_array = |beveled: &[usize]| -> Vec<usize> {
        let selected: Vec<EdgeKey> = beveled.iter().map(|&s| spoke_edge(s)).collect();
        let is_beveled = |edge: EdgeKey| selected.contains(&edge);
        mesh.vertex_bevel_corner_ring(centre, &is_beveled)
            .iter()
            .map(|corner| spoke_number(corner.incoming_edge).expect("a hub spoke"))
            .collect()
    };

    assert_eq!(
        mesh.iter_disk_edges(centre)
            .map(|edge| spoke_number(edge).expect("a hub spoke"))
            .collect::<Vec<usize>>(),
        vec![2, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        "the fixture's stored disk order is the one the worked example is derived on",
    );

    // Seed 2: its backward fan neighbour (spoke 1) precedes its forward one (spoke 3) in
    // the stored order, so the walk runs clockwise, the ballot reverses it, and the seed
    // lands in the array's last slot.
    assert_eq!(
        slot_array(&[1, 2, 7, 8]),
        vec![3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 1, 2],
    );
    // Seed 1 (spoke 2 unbeveled): its forward neighbour (spoke 2) precedes its backward
    // one (spoke 12), so the walk runs counter-clockwise, is *not* reversed, and the seed
    // stays in slot 0.
    assert_eq!(
        slot_array(&[1, 3, 7, 9]),
        vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    );
}

/// BEVEL.md §7.6.1.2 step 3 on the fifteen two-collinear-pair rings of the closed 12-fan:
/// the origin scan starts at the slot array's **last** slot and wraps, so `g_0` is not
/// simply the lowest-numbered beveled spoke (which misses `1/2/7/8` and `1/6/7/12`) and
/// not the first beveled spoke from slot 0 either.
///
/// This is the guide's fingerprint table: a correct implementation matches all fifteen.
/// Both `g_0` and `g_1` are pinned because `g_1` is the pipe axis §8.3.4.1 condition 2
/// returns, and on these rings a one-step rotation of the origin swaps the two candidate
/// axes without disturbing anything else.
#[test]
fn bevel_ring_origin_scans_from_the_slot_arrays_last_slot() {
    // (beveled diameter pair `{a, b, a+6, b+6}`, expected `g_0`, expected `g_1`).
    let expected = [
        ((1, 2), 2, 7),
        ((1, 3), 1, 3),
        ((1, 4), 1, 4),
        ((1, 5), 1, 5),
        ((1, 6), 12, 1),
        ((2, 3), 2, 3),
        ((2, 4), 2, 4),
        ((2, 5), 2, 5),
        ((2, 6), 2, 6),
        ((3, 4), 3, 4),
        ((3, 5), 3, 5),
        ((3, 6), 3, 6),
        ((4, 5), 4, 5),
        ((4, 6), 4, 6),
        ((5, 6), 5, 6),
    ];
    for ((a, b), origin_spoke, axis_spoke) in expected {
        let (mesh, centre, rim) = planar_fan(12);
        let spoke_edge = |spoke: usize| {
            mesh.query_edge_exists(centre, rim[spoke - 1])
                .expect("spoke exists")
        };
        let selected: Vec<EdgeKey> = [a, b, a + 6, b + 6].map(spoke_edge).into();
        let is_beveled = |edge: EdgeKey| selected.contains(&edge);
        let corners = mesh.vertex_bevel_corner_ring(centre, &is_beveled);
        let sectors = sector_split(&corners, &is_beveled);
        assert_eq!(sectors.len(), 4, "one sector per beveled spoke");
        let spoke_number = |edge: EdgeKey| (1..=12).find(|&spoke| spoke_edge(spoke) == edge);
        assert_eq!(
            (
                spoke_number(sectors[0].bounding_edges[0]),
                spoke_number(sectors[1].bounding_edges[0]),
            ),
            (Some(origin_spoke), Some(axis_spoke)),
            "ring {a}/{b}/{}/{}: g_0 opens B_0's sector and g_1 is the pipe axis",
            a + 6,
            b + 6,
        );
    }
}

/// BEVEL.md §7.6.1.2's second worked example — the half-disc, where step 1's radial-1
/// clause *outranks* its beveled clause and so seeds on a spoke that is not in the
/// selection at all.
///
/// `open_planar_fan(7, 0)` is that fixture: hub, seven spokes over 180°, six triangles
/// created in order. Both fan ends are radial-1, so the seed is the later of them in the
/// stored order and the array reverses to run from the other end. Bevelling spokes
/// `2 / 4 / 6` then leaves the origin scan genuinely traversing the array — the last slot
/// and slot 0 are both unbeveled fan ends — instead of resolving at or beside the seed.
#[test]
fn bevel_ring_origin_seed_prefers_a_radial_one_spoke_over_a_beveled_one_half_disc() {
    let (mesh, centre, rim) = open_planar_fan(7, 0);
    let spoke_edge = |spoke: usize| {
        mesh.query_edge_exists(centre, rim[spoke - 1])
            .expect("spoke exists")
    };
    let selected: Vec<EdgeKey> = [2, 4, 6].map(spoke_edge).into();
    let is_beveled = |edge: EdgeKey| selected.contains(&edge);
    let corners = mesh.vertex_bevel_corner_ring(centre, &is_beveled);
    let spoke_number = |edge: EdgeKey| (1..=7).find(|&spoke| spoke_edge(spoke) == edge);

    assert_eq!(
        corners
            .iter()
            .map(|corner| spoke_number(corner.incoming_edge).expect("a hub spoke"))
            .collect::<Vec<usize>>(),
        vec![1, 2, 3, 4, 5, 6],
        "the array runs from the end opposite the seed; its trailing slot is spoke 7",
    );
    assert_eq!(
        spoke_number(corners[5].outgoing_edge),
        Some(7),
        "spoke 7 is the array's last slot and carries no corner of its own",
    );

    let sectors = sector_split(&corners, &is_beveled);
    assert_eq!(
        sectors.len(),
        3,
        "one boundvert per beveled spoke, the gap-spanning sector included",
    );
    assert_eq!(
        (
            spoke_number(sectors[0].bounding_edges[0]),
            spoke_number(sectors[1].bounding_edges[0]),
        ),
        (Some(2), Some(4)),
        "both fan ends are unbeveled, so the scan falls through them to spoke 2",
    );
}

/// BEVEL.md §7.6.1.2's **vertex-bevel reduction**: the spoke array's slot 0 — which
/// §6.6.1.1 calls the fan's head edge `e[0]` — on the same closed 12-fan the section's
/// worked example is stated for.
///
/// Derived, not observed. Vertex mode drops step 3 and drops step 1's beveled clause, so
/// the seed is the **disk anchor**, spoke `2`. Its backward fan neighbour (spoke `1`)
/// precedes its forward one (spoke `3`) in the stored order `2, 1, 3, … 12`, so step 2's
/// walk runs clockwise, the aggregate ballot reverses it, and the seed lands in the
/// array's *last* slot — putting its cyclic successor, spoke `3`, in slot 0. The array
/// then reads `3, 4, … 12, 1, 2`, which is the same array the edge-bevel form builds for
/// the `1/2/7/8` ring above (both seed on spoke 2); only the origin step differs, and
/// vertex mode has none.
#[test]
fn vertex_mode_disk_order_starts_at_the_ring_origin_not_the_walks_own_seed() {
    let (mesh, centre, rim) = planar_fan(12);
    let spoke_edge = |spoke: usize| {
        mesh.query_edge_exists(centre, rim[spoke - 1])
            .expect("spoke exists")
    };
    let spoke_number = |edge: EdgeKey| (1..=12).find(|&spoke| spoke_edge(spoke) == edge);
    let incident: Vec<EdgeKey> = mesh.iter_disk_edges(centre).collect();

    assert_eq!(
        mesh.vertex_mode_disk_order(centre, &incident, &[])
            .iter()
            .map(|&edge| spoke_number(edge).expect("a hub spoke"))
            .collect::<Vec<usize>>(),
        vec![3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 1, 2],
    );
}

/// BEVEL.md §6.6.2's **vertex-bevel cap** row: the cap inherits `prev_face` of the fan's
/// **head edge**, and §7.6.1.2's vertex-bevel reduction is what picks that edge.
///
/// The discriminator is that `prev_face(e[0])` is **not** the corner's first incident
/// face: both are drawn from the head spoke's two flanking faces, and on this fixture
/// they land on opposite ones, so a positional pick off the disk stamps a different
/// material slot. Every leg here is a radial-2 interior manifold edge and the mesh
/// carries no loop layer, so the corner is seam-free (a material boundary is no
/// interpolation seam) and the anchor branch is the one under test.
#[test]
fn vertex_cap_inherits_the_ring_origins_boundvert_rep_not_the_disk_heads() {
    // `planar_fan(12)` with spoke `12` attached **before** any face is created, so the
    // hub's anchor is spoke 12 rather than the spoke a face-create would mint first. The
    // stored order is then `12, 2, 1, 3, 4, … 11`, in which the anchor's *forward* fan
    // neighbour (spoke 1, rank 2) precedes its backward one (spoke 11, rank 10): step 2's
    // walk therefore runs with the fan, the array is unreversed, and the head edge is the
    // anchor itself. That is the arm where the head's `prev_face` and the disk head's own
    // first radial face part company.
    let mut mesh = LoopMesh::new();
    let centre = mesh.create_vertex(Vec3::ZERO);
    let rim: Vec<VertexKey> = (0..12)
        .map(|k| {
            let angle = std::f32::consts::TAU * (k as f32) / 12.0;
            mesh.create_vertex(Vec3::new(angle.cos(), angle.sin(), 0.0))
        })
        .collect();
    mesh.create_edge([centre, rim[11]], EdgeCreateFlags::NONE);
    for k in 0..12 {
        mesh.face_create_verts(&[centre, rim[k], rim[(k + 1) % 12]], FaceCreateFlags::NONE);
    }
    for (index, face) in mesh.face_keys().into_iter().enumerate() {
        mesh.set_face_material_index(face, index as i32 + 1);
    }
    let head = mesh
        .query_edge_exists(centre, rim[11])
        .expect("spoke 12 exists");
    let origin_rep = mesh
        .boundvert_representative(centre, head)
        .expect("the head edge's arriving face");
    let disk_head_pick = mesh
        .first_incident_face(centre)
        .expect("the disk head's first face");
    assert!(
        !mesh.corner_is_seam_carrying(centre),
        "an interior closed manifold fan raises no interpolation seam",
    );
    assert_ne!(
        origin_rep, disk_head_pick,
        "fixture sanity: the ring origin's rep and the disk head's face must differ",
    );
    let expected = mesh.face_material_index(origin_rep);

    let result = mesh.bevel(
        &[centre.into()],
        BevelOptions {
            offset: 0.2,
            segments: 1,
            affect: BevelAffect::Vertices,
            ..Default::default()
        },
    );

    let cap = result
        .faces
        .iter()
        .copied()
        .find(|&face| mesh.face_vert_keys(face).len() == 12)
        .expect("the twelve-corner cap n-gon");
    assert_eq!(
        mesh.face_material_index(cap),
        expected,
        "the cap takes prev_face of the §7.6.1.2 head edge, not of the disk anchor",
    );
}

/// BEVEL.md §6.6.2's **junction patch** row: the patch inherits the representative face
/// of the ring's *first* boundvert, and `B_0`'s own spoke is the one that **closes** its
/// sector (§7.6.1.2 fact 3) — `g_1`, not the origin spoke `g_0`.
///
/// On the 12-fan's `1/2/7/8` ring the guide's fingerprint table gives `g_0 = 2` and
/// `g_1 = 7`, so the donor is the face arriving along spoke 7. Reading `g_0` instead
/// would rotate the donor one ring step; reading the disk head would leave the ring out
/// of it altogether. All three are distinct faces here.
#[test]
fn junction_patch_inherits_the_rep_of_the_spoke_that_closes_b_zeros_sector() {
    let (mut mesh, centre, rim) = planar_fan(12);
    for (index, face) in mesh.face_keys().into_iter().enumerate() {
        mesh.set_face_material_index(face, index as i32 + 1);
    }
    let spoke_edge = |mesh: &LoopMesh, spoke: usize| {
        mesh.query_edge_exists(centre, rim[spoke - 1])
            .expect("spoke exists")
    };
    let selected: Vec<EdgeKey> = [1, 2, 7, 8].map(|s| spoke_edge(&mesh, s)).into();
    let is_beveled = |edge: EdgeKey| selected.contains(&edge);

    let corners = mesh.vertex_bevel_corner_ring(centre, &is_beveled);
    let sectors = sector_split(&corners, &is_beveled);
    let opening_edges: Vec<EdgeKey> = sectors
        .iter()
        .map(|sector| sector.bounding_edges[0])
        .collect();
    assert_eq!(
        opening_edges[1],
        spoke_edge(&mesh, 7),
        "fixture sanity: `g_1` is spoke 7 on this ring",
    );
    let expected = mesh.face_material_index(
        mesh.ring_first_boundvert_rep(centre, &opening_edges)
            .expect("B_0's own spoke has an arriving face"),
    );
    assert_ne!(
        expected,
        mesh.face_material_index(
            mesh.boundvert_representative(centre, opening_edges[0])
                .expect("g_0's arriving face"),
        ),
        "fixture sanity: reading `g_0` instead of `g_1` must land a different face",
    );
    assert_ne!(
        expected,
        mesh.face_material_index(mesh.first_incident_face(centre).expect("a disk head face")),
        "fixture sanity: the disk head's face is a third, distinct face",
    );

    let beveled: Vec<GeomKey> = selected.iter().map(|&edge| edge.into()).collect();
    let result = mesh.bevel(
        &beveled,
        BevelOptions {
            offset: 0.1,
            segments: 1,
            affect: BevelAffect::Edges,
            ..Default::default()
        },
    );

    // The patch is the one new face whose corners are *all* boundverts of this junction:
    // every strip quad reaches down its spoke to the rim, so half its corners sit a unit
    // away from the hub while the patch's four stay within an offset of it.
    let patch = result
        .faces
        .iter()
        .copied()
        .find(|&face| {
            mesh.face_vert_keys(face)
                .iter()
                .all(|&vert| mesh.vertex_position(vert).length() < 0.5)
        })
        .expect("the four-boundvert junction patch");
    assert_eq!(
        mesh.face_material_index(patch),
        expected,
        "the patch takes prev_face of `g_1`, the spoke closing `B_0`'s sector",
    );
}

/// BEVEL.md §6.6.2's **terminal fill** row under §7.6.1.2's `s == 1` scope bullet: a
/// terminal's single sector *closes* at the one beveled spoke, so `B_0`'s own spoke is
/// that spoke itself and the fill inherits `prev_face(E)`. Anchor-free — a seed cannot
/// move an origin with one beveled spoke to land on — but not what a positional pick off
/// the disk anchor answers.
///
/// Beveling spoke `1` of the interior 12-fan hub makes it a valence-12 terminal, whose
/// fill exists (`n − 1 = 11` boundverts) and whose donor is the face arriving along
/// spoke 1: the triangle between spokes `12` and `1`.
#[test]
fn terminal_fill_inherits_prev_face_of_its_one_beveled_spoke() {
    let (mut mesh, centre, rim) = planar_fan(12);
    for (index, face) in mesh.face_keys().into_iter().enumerate() {
        mesh.set_face_material_index(face, index as i32 + 1);
    }
    let beveled_spoke = mesh
        .query_edge_exists(centre, rim[0])
        .expect("spoke 1 exists");
    let expected = mesh.face_material_index(
        mesh.ring_first_boundvert_rep(centre, &[beveled_spoke])
            .expect("the beveled spoke's arriving face"),
    );
    assert_ne!(
        expected,
        mesh.face_material_index(mesh.first_incident_face(centre).expect("a disk head face")),
        "fixture sanity: prev_face(E) is not the disk head's face",
    );

    let result = mesh.bevel(
        &[beveled_spoke.into()],
        BevelOptions {
            offset: 0.15,
            segments: 1,
            affect: BevelAffect::Edges,
            ..Default::default()
        },
    );

    let fill = result
        .faces
        .iter()
        .copied()
        .find(|&face| mesh.face_vert_keys(face).len() == 11)
        .expect("the eleven-corner terminal fill");
    assert_eq!(
        mesh.face_material_index(fill),
        expected,
        "the terminal fill takes prev_face of its one beveled spoke",
    );
}
