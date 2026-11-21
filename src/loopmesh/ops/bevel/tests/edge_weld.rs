use super::*;

/// BEVEL.md §5.1 / §8.3.0: the multi-segment strip and the cap must join the
/// re-stitched originals with consistent loop traversal, so the whole beveled
/// cube corner stays globally winding-consistent at every segment count. Here
/// every strip end is a junction.
#[test]
fn edge_bevel_cube_corner_winding_consistent_all_segments() {
    assert_cube_corner_bevel_sound(3, 1..=4, &[BevelVmeshMethod::Adj, BevelVmeshMethod::Cutoff]);
}

/// BEVEL.md §5.1 / §8.3.0.1: a lone beveled cube edge — *both* strip ends are
/// valence-3 terminals, so the outer-rail winding anchor governs a strip that
/// no junction cap touches.
#[test]
fn edge_bevel_single_cube_edge_winding_consistent_all_segments() {
    assert_cube_corner_bevel_sound(1, 1..=4, &[BevelVmeshMethod::Adj, BevelVmeshMethod::Cutoff]);
}

/// BEVEL.md §5.1: the mixed case — two beveled edges at one corner give each
/// strip a junction end and a terminal end, so the strip's two outer rails are
/// anchored by faces the *junction* and the *terminal* close rebuilt.
#[test]
fn edge_bevel_cube_two_edges_winding_consistent_all_segments() {
    assert_cube_corner_bevel_sound(2, 1..=4, &[BevelVmeshMethod::Adj]);
}

/// The same mixed case under `Cutoff`. A 2-sector junction's two chains span the
/// *same* boundvert pair `{B_0, B_1}`, so BEVEL.md §8.3.0.3 welds them into one and
/// gives the corner **no** fill face under either method — a cut face per "sector"
/// here would emit one chain's fill twice (the second copy reflected, traversing
/// every shared edge the same way as the first) and the other's never.
#[test]
fn edge_bevel_cube_two_edges_cutoff_winding_consistent_all_segments() {
    assert_cube_corner_bevel_sound(2, 2..=4, &[BevelVmeshMethod::Cutoff]);
}

/// BEVEL.md §8.3.0.3: the welded two-sector junction is **closed** with zero fill
/// faces. The single chain `[B_0, w_1 … w_(segments−1), B_1]` is the shared terminal
/// rung of both arriving strips, so every chain edge lands on exactly two faces (one
/// quad per strip). Minting a chain per beveled edge instead would leave a
/// `2 × segments`-edge hole — which disturbs neither the winding nor the volume
/// assertion, so only a closedness check sees it.
#[test]
fn edge_bevel_cube_two_edges_adj_junction_is_closed() {
    for segments in 2..=4 {
        let mut mesh = LoopMesh::new();
        mesh_add_cube(&mut mesh);
        let corner = vert_at(&mesh, Vec3::new(1.0, -1.0, -1.0));
        let geom: Vec<GeomKey> = mesh
            .vert_edge_keys(corner)
            .into_iter()
            .take(2)
            .map(GeomKey::from)
            .collect();
        mesh.bevel(
            &geom,
            BevelOptions {
                offset: 0.25,
                affect: BevelAffect::Edges,
                segments,
                vmesh_method: BevelVmeshMethod::Adj,
                ..Default::default()
            },
        );
        let open = mesh
            .edge_keys()
            .iter()
            .filter(|&&edge| mesh.edge_radial_iter(edge).count() != 2)
            .count();
        assert_eq!(
            open, 0,
            "segments = {segments}: {open} boundary edge(s) — the beveled solid is open",
        );
    }
}

/// The §8.3.0.3 worked fixture: bevel the `−x` and `+y` edges at the cube corner
/// `(1, −1, −1)`, leaving the `+z` edge as the unbeveled leg of the sector that holds
/// it. Both beveled edges' far ends are valence-3 terminals.
fn bevel_two_edge_corner(segments: u32, vmesh_method: BevelVmeshMethod) -> (LoopMesh, BevelResult) {
    let mut mesh = LoopMesh::new();
    mesh_add_cube(&mut mesh);
    let corner = vert_at(&mesh, Vec3::new(1.0, -1.0, -1.0));
    let geom: Vec<GeomKey> = [Vec3::new(-1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, -1.0)]
        .into_iter()
        .map(|far| {
            let edge = mesh
                .query_edge_exists(corner, vert_at(&mesh, far))
                .expect("the cube corner's beveled edge");
            GeomKey::from(edge)
        })
        .collect();
    let result = mesh.bevel(
        &geom,
        BevelOptions {
            offset: 0.25,
            affect: BevelAffect::Edges,
            segments,
            vmesh_method,
            ..Default::default()
        },
    );
    (mesh, result)
}

/// BEVEL.md §8.3.0.3: the welded junction mints `segments − 1` interior verts
/// **once, not per sector**, and contributes **no face at all** — so the beveled
/// solid lands the oracle counts, and its only non-quads are the two *terminal*
/// far-end faces absorbing their through-chains (§8.3.0.1).
#[test]
fn edge_bevel_two_edge_junction_welds_one_chain_at_oracle_counts() {
    for (segments, expected, absorbing_face) in [
        (2u32, (14usize, 22usize, 10usize), 6usize),
        (3, (17, 27, 12), 7),
    ] {
        for vmesh_method in [BevelVmeshMethod::Adj, BevelVmeshMethod::Cutoff] {
            let (mesh, _) = bevel_two_edge_corner(segments, vmesh_method);
            assert_eq!(
                counts(&mesh),
                expected,
                "segments = {segments}, {vmesh_method:?}: V/E/F",
            );
            let mut sizes: Vec<usize> = mesh
                .face_keys()
                .iter()
                .map(|&face| mesh.face_vert_keys(face).len())
                .collect();
            sizes.sort_unstable();
            let absorbing = sizes.iter().filter(|&&n| n == absorbing_face).count();
            let quads = sizes.iter().filter(|&&n| n == 4).count();
            assert_eq!(
                (quads, absorbing),
                (expected.2 - 2, 2),
                "segments = {segments}, {vmesh_method:?}: face sizes {sizes:?} — expected \
                 only quads plus the two terminal absorbing faces",
            );
        }
    }
}

/// BEVEL.md §8.3.0.3 (the weld-plane override): the two chains are sampled in
/// *different* planes by default and must be re-planed onto plane(`v`, `B_0`, `B_1`)
/// before they can be identified. The override is load-bearing: averaging the
/// un-overridden samples would land the interior vert at `(0.9009, −0.9009, −0.9268)`,
/// which is not where the corner's cross-section runs.
#[test]
fn edge_bevel_two_edge_junction_weld_vert_sits_on_the_weld_plane() {
    let (mesh, result) = bevel_two_edge_corner(2, BevelVmeshMethod::Adj);
    assert_eq!(
        result_has_vert_near(&mesh, &result, Vec3::new(0.75, -0.75, -1.0)),
        1,
        "B_0 — the in-face meet of the two beveled edges' offset lines",
    );
    assert_eq!(
        result_has_vert_near(&mesh, &result, Vec3::new(1.0, -1.0, -0.75)),
        1,
        "B_1 — the meet in the sector holding the leg, which lands on the leg",
    );
    assert_eq!(
        result_has_vert_near(
            &mesh,
            &result,
            Vec3::new(0.926_776_7, -0.926_776_7, -0.926_776_7)
        ),
        1,
        "the single welded interior vert, sampled in the weld plane x + y = 0",
    );
    assert_eq!(
        result_has_vert_near(
            &mesh,
            &result,
            Vec3::new(0.900_888, -0.900_888, -0.926_776_7)
        ),
        0,
        "the midpoint of the two un-overridden samples is not the answer",
    );
}

/// BEVEL.md §8.3.0.3: at `s == 2` the fill dispatch keys on the boundvert ring count,
/// not on `vmesh_method` — both methods fill nothing — so `Cutoff` and `Adj` produce
/// identical meshes, vertex for vertex. (They diverge at `s == 3`, where the ring
/// really does bound a region: §8.5's rounded apex vs §8.3's flat cut corner.)
#[test]
fn edge_bevel_two_edge_junction_cutoff_matches_adj() {
    for segments in 2..=4 {
        let (cutoff, _) = bevel_two_edge_corner(segments, BevelVmeshMethod::Cutoff);
        let (adj, _) = bevel_two_edge_corner(segments, BevelVmeshMethod::Adj);
        assert!(
            positions_multiset_eq(&cutoff, &adj, 1.0e-5),
            "segments = {segments}: the two methods disagree on vertex positions",
        );
        assert!(
            faces_match(&cutoff, &adj, 1.0e-5),
            "segments = {segments}: the two methods disagree on faces",
        );
    }
}

/// BEVEL.md §8.3.0.3's strip-rung counts: the welded chain is the shared terminal rung
/// of *both* strips, so each chain edge carries one quad from each (radial 2), and an
/// interior weld vert has 4 faces and 4 edges — the two chain edges plus one
/// longitudinal rung into each strip. A per-sector chain instead would leave the
/// interior verts at radial 1 on their own side.
#[test]
fn edge_bevel_two_edge_junction_weld_vert_is_shared_by_both_strips() {
    let (mesh, result) = bevel_two_edge_corner(2, BevelVmeshMethod::Adj);
    let weld = result
        .verts
        .iter()
        .copied()
        .find(|&v| {
            (mesh.vertex_position(v) - Vec3::new(0.926_776_7, -0.926_776_7, -0.926_776_7)).length()
                < 1.0e-5
        })
        .expect("the welded interior chain vert");
    assert_eq!(
        mesh.vert_face_keys(weld).len(),
        4,
        "the weld vert takes two quads from each arriving strip",
    );
    let disk: Vec<EdgeKey> = mesh.iter_disk_edges(weld).collect();
    assert_eq!(
        disk.len(),
        4,
        "two chain edges plus one longitudinal rung edge into each strip",
    );
    for edge in disk {
        assert_eq!(
            mesh.edge_radial_iter(edge).count(),
            2,
            "every edge at the weld vert is manifold",
        );
    }
}

/// BEVEL.md §8.3.0.3: `s == 2 ⇒ n == 2` **regardless of the vertex's valence** — the
/// unbeveled legs are absorbed into their sector's boundvert and never add a ring
/// corner. The octahedron apex is valence 4 with its two beveled edges *opposite* in
/// the disk (the "cross"), and it welds and closes exactly like the valence-3 corner,
/// with no fill face under either method.
#[test]
fn edge_bevel_cross_weld_valence_four_apex_is_closed() {
    for segments in 2..=3 {
        for vmesh_method in [BevelVmeshMethod::Adj, BevelVmeshMethod::Cutoff] {
            let (mesh, result) = bevel_octahedron_cross(segments, vmesh_method, false, false);
            assert_bevel_sound(
                &mesh,
                &result,
                &format!("cross weld, segments = {segments}"),
            );
            let open = mesh
                .edge_keys()
                .iter()
                .filter(|&&edge| mesh.edge_radial_iter(edge).count() != 2)
                .count();
            assert_eq!(
                open, 0,
                "segments = {segments}, {vmesh_method:?}: {open} boundary edge(s) — the \
                 valence-4 cross weld is open",
            );
        }
    }
    for segments in 2..=3 {
        let (cutoff, _) = bevel_octahedron_cross(segments, BevelVmeshMethod::Cutoff, false, false);
        let (adj, _) = bevel_octahedron_cross(segments, BevelVmeshMethod::Adj, false, false);
        assert!(
            positions_multiset_eq(&cutoff, &adj, 1.0e-5) && faces_match(&cutoff, &adj, 1.0e-5),
            "segments = {segments}: the two methods disagree at the valence-4 weld",
        );
    }
}

/// The **collinear control** for §8.2.2.1's zero-cell-axis rule: the flat-grid cross
/// weld, where the tier-3 collapse fires for *both* profiles and must stay exactly
/// where it was.
///
/// §8.3.0.3's oracle: *"on a flat grid, bevel the two **opposite** spokes of an interior
/// valence-4 vertex … `v` then lies exactly on the `B_0`–`B_1` chord, both profiles
/// collapse, and the welded chain is the plain even subdivision of the straight chord:
/// with `v = (1, 1, 0)`, `offset = 0.25`, `B_0 = (1, 1.25, 0)`, `B_1 = (1, 0.75, 0)`,
/// the interiors are `(1, 1, 0)` at `segments = 2`; `(1, 1.08333, 0)` and
/// `(1, 0.91667, 0)` at `3`; and `(1, 1.125, 0)`, `(1, 1, 0)`, `(1, 0.875, 0)` at `4` —
/// **identical at `profile = 0.3`, `0.5` and `0.8`**, which is the signature of a
/// collapsed profile: the cross-section exponent stops mattering."*
///
/// This is the arm the zero-axis case must **not** be conflated with. Both reach the
/// even chord, but by different routes and carrying different flags — here the profiles
/// are flagged linear and their projection setup is zeroed, and §8.3.0.3 contrasts the
/// two directly: *"Contrast the zero-axis case at a clear apex, where only **one** half
/// of the merge is exponent-independent."*
#[test]
fn edge_bevel_collinear_cross_weld_is_the_exponent_independent_even_chord() {
    let build_grid = || {
        let mut grid = LoopMesh::new();
        let grid_verts: Vec<VertexKey> = (0..3)
            .flat_map(|row| (0..3).map(move |col| (row, col)))
            .map(|(row, col)| grid.create_vertex(Vec3::new(col as f32, row as f32, 0.0)))
            .collect();
        for row in 0..2usize {
            for col in 0..2usize {
                let ring = [
                    grid_verts[row * 3 + col],
                    grid_verts[row * 3 + col + 1],
                    grid_verts[(row + 1) * 3 + col + 1],
                    grid_verts[(row + 1) * 3 + col],
                ];
                grid.face_create_verts(&ring, FaceCreateFlags::NONE);
            }
        }
        (grid, grid_verts)
    };
    let junction = Vec3::new(1.0, 1.0, 0.0);
    let boundvert_0 = Vec3::new(1.0, 1.25, 0.0);
    let boundvert_1 = Vec3::new(1.0, 0.75, 0.0);

    // Fixture sanity, stated on the *inputs* §8.3.0.3 names rather than on the emitted
    // chain: `v` lies exactly on the `B_0`–`B_1` chord — its two cell axes are exactly
    // anti-parallel — which is what collapses **both** profiles.
    assert!(
        near_collinear_control_points(boundvert_0, junction, boundvert_1)
            && (junction - (boundvert_0 + boundvert_1) * 0.5).length() < 1.0e-6,
        "fixture sanity: `v` is the chord midpoint, so both profiles collapse",
    );

    for (segments, interiors) in [
        (2u32, vec![1.0_f32]),
        (3, vec![0.916_666_7, 1.083_333_3]),
        (4, vec![0.875, 1.0, 1.125]),
    ] {
        for profile in [0.3_f32, 0.5, 0.8] {
            let (mut mesh, grid_verts) = build_grid();
            let west = mesh
                .query_edge_exists(grid_verts[4], grid_verts[3])
                .expect("the −x spoke");
            let east = mesh
                .query_edge_exists(grid_verts[4], grid_verts[5])
                .expect("the +x spoke");
            let result = mesh.bevel(
                &[west.into(), east.into()],
                BevelOptions {
                    offset: 0.25,
                    affect: BevelAffect::Edges,
                    segments,
                    profile,
                    vmesh_method: BevelVmeshMethod::Adj,
                    ..Default::default()
                },
            );
            let mut chain: Vec<f32> = result
                .verts
                .iter()
                .map(|&v| mesh.vertex_position(v))
                .filter(|p| (p.x - 1.0).abs() < 1.0e-4 && (p.y - 1.0).abs() < 0.4)
                .map(|p| p.y)
                .collect();
            chain.sort_by(f32::total_cmp);
            let mut expected = vec![boundvert_1.y];
            expected.extend_from_slice(&interiors);
            expected.push(boundvert_0.y);
            assert_eq!(
                chain.len(),
                expected.len(),
                "segments = {segments}, profile = {profile}: welded chain {chain:?}",
            );
            for (got, want) in chain.iter().zip(&expected) {
                assert!(
                    (got - want).abs() < 1.0e-4,
                    "segments = {segments}, profile = {profile}: expected {expected:?}, \
                     got {chain:?} — a collapsed profile's chain is the even chord at \
                     every exponent",
                );
            }
        }
    }
}

/// BEVEL.md §8.3.0.3's attribute note: at the **cross** weld the chain reads as the
/// continuation of the two crossing legs, so its edges take those legs' flags — the
/// seam only crosses if it crosses on **both** sides, and the shading stays smooth
/// unless **both** sides are sharp. Both reconciliations are computed from both legs,
/// so they are order-independent even though the two donations collide.
#[test]
fn edge_bevel_cross_weld_chain_reconciles_the_two_crossing_legs() {
    for (both_legs, expected) in [(true, true), (false, false)] {
        let (mesh, result) = bevel_octahedron_cross(2, BevelVmeshMethod::Adj, both_legs, true);
        // The fixture is mirror-symmetric in both `x` (swapping the two beveled edges)
        // and `y` (swapping the two boundverts), so the whole welded chain lies on
        // `x = 0` and its lone interior vert on the `y = 0` axis with it. The strips'
        // longitudinal rungs are the only other edges at that vert, and they run off
        // along `±x`.
        let on_weld_plane = |v: VertexKey| mesh.vertex_position(v).x.abs() < 1.0e-4;
        let weld = result
            .verts
            .iter()
            .copied()
            .find(|&v| on_weld_plane(v) && mesh.vertex_position(v).y.abs() < 1.0e-4)
            .expect("the welded interior chain vert");
        let chain_edges: Vec<EdgeKey> = mesh
            .iter_disk_edges(weld)
            .filter(|&edge| {
                let [a, b] = mesh.edge_vert_keys(edge);
                on_weld_plane(if a == weld { b } else { a })
            })
            .collect();
        assert_eq!(
            chain_edges.len(),
            2,
            "the weld vert owns the two chain edges"
        );
        for edge in chain_edges {
            assert_eq!(
                mesh.edge_seam(edge),
                expected,
                "the seam crosses the weld only when both crossing legs carry it",
            );
            assert_eq!(
                mesh.edge_sharp(edge),
                expected,
                "the weld stays smooth unless both crossing legs are sharp",
            );
        }
    }
}

/// BEVEL.md §8.3.0.3's attribute note, the ordering half: the copy runs once per
/// beveled edge, each donating from **that edge's own** disk-predecessor leg, and both
/// runs write the *same* chain edges. There is no unique donor — the strip built
/// **last** supplies the surviving per-edge customdata. Invisible in the seam / smooth
/// bits (their reconciliation is symmetric); visible the moment the two legs differ in
/// any other per-edge attribute.
#[test]
fn edge_bevel_cross_weld_chain_takes_the_last_built_strips_donor_leg() {
    let mut mesh = LoopMesh::new();
    mesh_add_octahedron(&mut mesh);
    let apex = vert_at(&mesh, Vec3::new(0.0, 0.0, 1.0));
    let edge_to = |mesh: &LoopMesh, position: Vec3| {
        mesh.query_edge_exists(apex, vert_at(mesh, position))
            .expect("the apex's edge")
    };
    let tint = mesh
        .register_edge_attribute(ElementAttributeDesc::numeric(
            AttributeKey::custom("tint"),
            AttrType::Vec3,
        ))
        .unwrap();
    let legs = [
        edge_to(&mesh, Vec3::new(0.0, 1.0, 0.0)),
        edge_to(&mesh, Vec3::new(0.0, -1.0, 0.0)),
    ];
    mesh.edge_attribs_mut()
        .set_vec3(tint, legs[0].as_usize(), [1.0, 0.0, 0.0]);
    mesh.edge_attribs_mut()
        .set_vec3(tint, legs[1].as_usize(), [0.0, 1.0, 0.0]);

    // The two donations collide on the same chain edges and the later prevails; the
    // operator walks its beveled edges in the **mesh's** edge order, so the survivor
    // is the disk-predecessor leg of the higher-keyed spoke — taken in the §7.6.2 walk
    // sense, never off the insertion-ordered disk cycle.
    let beveled = [
        edge_to(&mesh, Vec3::new(1.0, 0.0, 0.0)),
        edge_to(&mesh, Vec3::new(-1.0, 0.0, 0.0)),
    ];
    let last = *beveled
        .iter()
        .max_by_key(|edge| edge.as_usize())
        .expect("two beveled edges");
    let donor = walk_predecessor(&mesh, apex, last);
    assert!(
        legs.contains(&donor),
        "the donor is one of the crossing legs"
    );
    let expected = mesh
        .edge_attribs()
        .get_vec3(tint, donor.as_usize())
        .unwrap();

    let result = mesh.bevel(
        &beveled.map(GeomKey::from),
        BevelOptions {
            offset: 0.25,
            affect: BevelAffect::Edges,
            segments: 2,
            vmesh_method: BevelVmeshMethod::Adj,
            ..Default::default()
        },
    );

    let on_weld_plane = |v: VertexKey| mesh.vertex_position(v).x.abs() < 1.0e-4;
    let weld = result
        .verts
        .iter()
        .copied()
        .find(|&v| on_weld_plane(v) && mesh.vertex_position(v).y.abs() < 1.0e-4)
        .expect("the welded interior chain vert");
    let chain_edges: Vec<EdgeKey> = mesh
        .iter_disk_edges(weld)
        .filter(|&edge| {
            let [a, b] = mesh.edge_vert_keys(edge);
            on_weld_plane(if a == weld { b } else { a })
        })
        .collect();
    assert_eq!(
        chain_edges.len(),
        2,
        "the weld vert owns the two chain edges"
    );
    for edge in chain_edges {
        assert_eq!(
            mesh.edge_attribs().get_vec3(tint, edge.as_usize()).unwrap(),
            expected,
            "the welded chain carries the last-built strip's donor leg's per-edge CD",
        );
    }
}

// ---- §7.7 per-edge customdata on an edge bevel ----
