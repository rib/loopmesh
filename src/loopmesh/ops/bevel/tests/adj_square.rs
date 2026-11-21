use super::*;

/// The square-in dispatch splits on the **ring**, not on the exponent (BEVEL.md
/// §8.5.5.5(f.1) steps 3 and 4): a valence-3 corner the tri-corner test fires on takes the
/// (f.4a) degenerate weld — identified leg chains, no interior grid, at most one central
/// `s`-gon — while every other ring takes step 4's grid. The weld is a different object
/// from the grid (it *deletes* the ring quads the grid builds), so routing one to the
/// other is a whole-mesh count divergence, not a placement one.
///
/// Pins the boundary from both sides: the orthonormal cube corner welds, and the tripod
/// cone's apex — same valence, tri-corner test refused — fills with §8.5.5.5(e)'s counts.
#[test]
fn adj_square_in_weld_owns_the_tri_corner_ring_and_the_grid_owns_the_rest() {
    for segments in [2u32, 3] {
        let (mesh, result) = cube_corner_adj_profile(segments, 0.0);
        mesh.validate_topology();
        assert_eq!(
            curved_corner_grid_faces(&mesh, &result).len(),
            usize::from(segments % 2 == 1),
            "the weld's entire face output is the odd parity's one central `s`-gon \
             (segments = {segments})",
        );

        let (cone_mesh, cone) = tripod_cone_adj(segments, 0.0);
        cone_mesh.validate_topology();
        // Topological cap selection, not a ball around the corner: the negative
        // square-in fullness places the interior *past* `v`, so a radius tuned to the
        // rounded cap's extent would drop the very faces this is counting.
        let (cap, _) = pyramid_apex_cap(&cone_mesh, &cone);
        assert_eq!(
            cap.len(),
            adj_general_ring_parity_counts(3, segments).0,
            "the tri-corner-refused ring takes step 4's grid (segments = {segments})",
        );
    }
}

/// BEVEL.md §8.5.5.5(f.4a): the weld emits **no ring quads at all**, and the parity of
/// §8.5.5.5(e) decides whether it emits anything. Even `segments` — the innermost sample
/// of every leg is the octant origin, the legs meet at a point, and the fill is complete
/// with **zero** faces. Odd `segments` — no sample reaches the origin, so `s` distinct
/// innermost verts are closed by exactly **one** central `s`-gon, a triangle at `s = 3`.
///
/// The ring's vert count is the identification made visible: `s` boundverts plus `s`
/// legs of `⌊n/2⌋` verts, the innermost of which collapse onto one shared vertex at even
/// `n`. An implementation that merely placed the two sectors' samples coincidently would
/// carry `s·(n−1)` chain verts here instead.
#[test]
fn adj_square_in_weld_closes_the_corner_with_at_most_one_central_ngon() {
    for segments in [2u32, 3, 4, 5] {
        let (mesh, result) = cube_corner_adj_profile(segments, 0.0);
        mesh.validate_topology();
        let half = segments as usize / 2;
        let even = segments % 2 == 0;

        let cap = curved_corner_grid_faces(&mesh, &result);
        assert_eq!(
            cap.len(),
            usize::from(!even),
            "no ring quads; one central `s`-gon iff odd (segments = {segments})",
        );
        if let Some(&face) = cap.first() {
            assert_eq!(
                mesh.face_vert_keys(face).len(),
                3,
                "the central `s`-gon is a triangle at the cube corner (segments = \
                 {segments})",
            );
        }

        let ring = square_in_weld_ring(&mesh, &result);
        let expected = if even { 3 * half + 1 } else { 3 * (half + 1) };
        assert_eq!(
            ring.len(),
            expected,
            "welded ring verts at segments = {segments}",
        );

        // Welded, not merely coincident: every ring edge lands on exactly two faces, so
        // the corner is closed even where the fill emitted nothing.
        for &vertex in &ring {
            for edge in mesh.iter_disk_edges(vertex) {
                assert_eq!(
                    mesh.edge_face_keys(edge).len(),
                    2,
                    "the welded ring leaves no boundary (segments = {segments})",
                );
            }
        }

        let origin = square_in_weld_vert_at(&mesh, &result, square_in_weld_origin());
        assert_eq!(
            origin.is_some(),
            even,
            "only even `n` lands a sample on the octant origin (segments = {segments})",
        );

        // Whole-mesh counts, which is where the weld is actually visible: fewer verts
        // than an un-welded ring (pairwise identification) and, at odd `n`, one more
        // face (the corner closes). The cube starts at 8 / 12 / 6 and stays closed, so
        // V − E + F = 2 throughout.
        let counts = (mesh.vertex_count(), mesh.edge_count(), mesh.face_count());
        let expected = match segments {
            2 => (17, 27, 12),
            3 => (22, 36, 16),
            4 => (26, 42, 18),
            _ => (31, 51, 22),
        };
        assert_eq!(counts, expected, "whole mesh at segments = {segments}");
        assert_eq!(
            counts.0 + counts.2 - counts.1,
            2,
            "the beveled cube stays closed (segments = {segments})",
        );
    }
}

/// The cube's shell turned inside out: the same eight verts and six quads, every ring
/// walked backwards. Geometrically identical, so every position-based predicate — the
/// tri-corner meet geometry, the octant frame, the `r = 0` weld's leg layout — decides
/// exactly as it does on [`mesh_add_cube`]; only the faces' sense differs. It is the
/// in-crate stand-in for an inward-wound shell (a cup seen from inside).
fn mesh_add_inward_wound_cube(mesh: &mut LoopMesh) {
    let coords = [
        Vec3::new(1.0, -1.0, -1.0),
        Vec3::new(-1.0, -1.0, -1.0),
        Vec3::new(-1.0, 1.0, -1.0),
        Vec3::new(1.0, 1.0, -1.0),
        Vec3::new(1.0, -1.0, 1.0),
        Vec3::new(-1.0, -1.0, 1.0),
        Vec3::new(-1.0, 1.0, 1.0),
        Vec3::new(1.0, 1.0, 1.0),
    ];
    let verts: Vec<VertexKey> = coords.iter().map(|&c| mesh.create_vertex(c)).collect();
    for ring in [
        [3usize, 2, 1, 0],
        [2, 6, 5, 1],
        [6, 7, 4, 5],
        [7, 3, 0, 4],
        [7, 6, 2, 3],
        [1, 5, 4, 0],
    ] {
        let face: Vec<VertexKey> = ring.iter().map(|&i| verts[i]).collect();
        mesh.face_create_verts(&face, FaceCreateFlags::NONE);
    }
}

/// [`cube_corner_adj_profile`]'s driver on the inward-wound shell: the same corner, the
/// same three beveled spokes, the same [V-ADJ] dispatch — only the incident faces' sense
/// is flipped, so the cap has to come out flipped with them.
fn inward_wound_corner_adj_profile(segments: u32, profile: f32) -> (LoopMesh, BevelResult) {
    let mut mesh = LoopMesh::new();
    mesh_add_inward_wound_cube(&mut mesh);
    let e01 = cube_edge(
        &mesh,
        Vec3::new(1.0, -1.0, -1.0),
        Vec3::new(-1.0, -1.0, -1.0),
    );
    let e03 = cube_edge(&mesh, Vec3::new(1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, -1.0));
    let e04 = cube_edge(&mesh, Vec3::new(1.0, -1.0, -1.0), Vec3::new(1.0, -1.0, 1.0));
    let result = mesh.bevel(
        &[e01.into(), e03.into(), e04.into()],
        BevelOptions {
            offset: 0.25,
            affect: BevelAffect::Edges,
            segments,
            profile,
            vmesh_method: BevelVmeshMethod::Adj,
            ..Default::default()
        },
    );
    (mesh, result)
}

/// BEVEL.md §5.1: the §8.5.5.5(f.4a) weld's centre `s`-gon is wound by the **manifold
/// condition**, not by facing away from a point inside the corner.
///
/// The `s`-gon's own edges are rim chain edges — the weld identifies `chain_i[⌊n/2⌋+1]`
/// with `chain_(i+1)[⌊n/2⌋]`, so the step from one innermost leg vert to the next is
/// chain `i`'s step across the turn — and each already carries the strip that arrived
/// along that chain. So the same one-bit rule the grid cap uses applies verbatim, and it
/// is the *only* rule that survives an inward-wound shell: an outward test against the
/// notch bottom answers the same way whichever sense the surrounding faces run in, which
/// on this shell is the wrong way round and leaves the `s`-gon's three edges traversed
/// twice in the same direction.
///
/// The upright cube is the control: identical face output, and it is wound *outward*
/// there. Odd `segments` only — even parity welds at the octant origin and emits nothing.
#[test]
fn adj_square_in_weld_ngon_winds_against_an_inward_wound_shell() {
    for segments in [3u32, 5, 7] {
        let (mesh, result) = inward_wound_corner_adj_profile(segments, 0.0);
        mesh.validate_topology();

        let cap = curved_corner_grid_faces(&mesh, &result);
        assert_eq!(
            cap.len(),
            1,
            "the weld's whole face output is one central `s`-gon (segments = {segments})",
        );
        assert_eq!(
            mesh.face_vert_keys(cap[0]).len(),
            3,
            "a triangle at the cube corner (segments = {segments})",
        );

        let conflicts = winding_conflicts(&mesh);
        assert!(
            conflicts.is_empty(),
            "segments = {segments}: {} shared edge(s) traversed the same way by both \
             adjacent faces",
            conflicts.len(),
        );
        assert!(
            signed_volume(&mesh) < 0.0,
            "segments = {segments}: the shell is still inside out — the cap followed the \
             input's sense, it did not correct it",
        );

        // The control, and the sharper statement: the same corner on the upright cube
        // emits the same triangle over the same rim, walked the *other* way.
        let (upright, upright_result) = cube_corner_adj_profile(segments, 0.0);
        upright.validate_topology();
        let upright_cap = curved_corner_grid_faces(&upright, &upright_result);
        assert_eq!(upright_cap.len(), 1);
        let ring_positions = |mesh: &LoopMesh, face: FaceKey| -> Vec<Vec3> {
            mesh.face_vert_keys(face)
                .iter()
                .map(|&v| mesh.vertex_position(v))
                .collect()
        };
        let newell = |ring: &[Vec3]| -> Vec3 {
            let centroid = ring.iter().sum::<Vec3>() / ring.len() as f32;
            (0..ring.len()).fold(Vec3::ZERO, |total, i| {
                total + (ring[i] - centroid).cross(ring[(i + 1) % ring.len()] - centroid)
            })
        };
        let inward = newell(&ring_positions(&mesh, cap[0]));
        let outward = newell(&ring_positions(&upright, upright_cap[0]));
        assert!(
            inward.dot(outward) < 0.0,
            "segments = {segments}: the two shells' `s`-gons wind oppositely",
        );
        assert!(
            outward.dot(Vec3::new(1.0, -1.0, -1.0)) > 0.0,
            "segments = {segments}: the upright shell's `s`-gon faces out of the cube",
        );
    }
}

/// The general [V-ADJ] grid cap obeys the same rule on the same shell — the aggregate is
/// a cap-wide bit, so flipping the input's sense flips the whole cap and nothing else.
/// Sweeps both parities and both sides of the §8.5.5.5(f.1) dispatch (`0.25` is the
/// straight chord's forced pinwheel at `segments = 2`).
#[test]
fn adj_grid_cap_winds_against_an_inward_wound_shell() {
    for profile in [0.25f32, 0.5, 0.7] {
        for segments in 2..=5 {
            let (mesh, result) = inward_wound_corner_adj_profile(segments, profile);
            let case = format!("inward-wound cap, segments = {segments}, profile = {profile}");
            mesh.validate_topology();
            assert!(!result.faces.is_empty(), "{case}: bevel produced no faces");
            let conflicts = winding_conflicts(&mesh);
            assert!(
                conflicts.is_empty(),
                "{case}: {} shared edge(s) traversed the same way by both adjacent faces",
                conflicts.len(),
            );
            assert!(
                signed_volume(&mesh) < 0.0,
                "{case}: the shell is still inside out",
            );
        }
    }
}

/// The cube with **one** face — the `z = −1` front — wound backwards, the rest left
/// alone: an inconsistently wound input (BEVEL.md §7.8), and the smallest mesh on which
/// a cap's rim edges genuinely disagree about which way the cap should run.
fn mesh_add_cube_with_front_inverted(mesh: &mut LoopMesh) {
    let coords = [
        Vec3::new(1.0, -1.0, -1.0),
        Vec3::new(-1.0, -1.0, -1.0),
        Vec3::new(-1.0, 1.0, -1.0),
        Vec3::new(1.0, 1.0, -1.0),
        Vec3::new(1.0, -1.0, 1.0),
        Vec3::new(-1.0, -1.0, 1.0),
        Vec3::new(-1.0, 1.0, 1.0),
        Vec3::new(1.0, 1.0, 1.0),
    ];
    let verts: Vec<VertexKey> = coords.iter().map(|&c| mesh.create_vertex(c)).collect();
    for ring in [
        [3usize, 2, 1, 0],
        [1, 5, 6, 2],
        [5, 4, 7, 6],
        [4, 0, 3, 7],
        [3, 2, 6, 7],
        [0, 4, 5, 1],
    ] {
        let face: Vec<VertexKey> = ring.iter().map(|&i| verts[i]).collect();
        mesh.face_create_verts(&face, FaceCreateFlags::NONE);
    }
}

/// The aggregate on a real inconsistently wound input (BEVEL.md §7.8), which is the case
/// that decides between "first-found plus a unanimity assertion" and "aggregate plus a
/// tie-break": beveling the three spokes of a corner **incident** to the inverted face
/// gives the cap three sectors' worth of rim edges that do **not** agree, 2 against 1.
///
/// No cap winding can satisfy all of them, so the test is which ones it dissatisfies: the
/// aggregate takes the majority and leaves exactly the **one dissenting sector's** rim in
/// conflict — that sector's strip is the one that inherited the inverted face's sense — so
/// one whole chain's worth (`segments` edges). A minority choice would leave twice that.
///
/// **The profile loop sweeps `r`, not the dispatch.** This corner reaches the *general*
/// grid at every one of these profiles, the square-in sentinel included: §8.5.1's band
/// refuses a junction with a dissenting face, so `r = 0` lands on (f.4b) rather than on
/// (f.4a)'s weld (`adj_square_in_weld_declines_at_a_miswound_junction`). What the loop does
/// vary is the fill — the §8.5.3 pinwheel at `segments = 2` and the subdivided grid above
/// it — and [`LoopMesh::debug_assert_rim_agrees_within_each_sector`] is live throughout.
/// Everywhere else in the suite that guard holds trivially, on winding-consistent input;
/// here it is deliberately pointed at the §7.8 collapse-at-a-junction regime — the corner
/// is incident to the inverted face, so two of its three strips collapse a flank (§7.8.1) —
/// and it still holds, non-vacuously: two sectors vote unanimously one way and the third
/// unanimously the other, so the guard passes on a 2:1 rim that a unanimity assertion over
/// the *whole* rim would have fired on, on a mesh the operator is meant to survive.
///
/// The cap's **internal** edges carry the other half of the claim: whatever the rim made
/// the cap choose, no two cap faces disagree with each other. That is what "one bit"
/// means, and it is what a per-face probe breaks — such a probe reverses the dissenting
/// sector's cells alone, which cracks the cap along its seams while leaving the rim tally
/// below unchanged.
#[test]
fn adj_cap_winding_takes_the_majority_on_an_inconsistently_wound_input() {
    for profile in [0.0f32, 0.5, 0.75] {
        for segments in [2u32, 3, 5] {
            let case = format!("segments = {segments}, profile = {profile}");
            let mut mesh = LoopMesh::new();
            mesh_add_cube_with_front_inverted(&mut mesh);
            assert_eq!(
                winding_conflicts(&mesh).len(),
                4,
                "the input starts with the inverted face's four edges in conflict",
            );
            let e01 = cube_edge(
                &mesh,
                Vec3::new(1.0, -1.0, -1.0),
                Vec3::new(-1.0, -1.0, -1.0),
            );
            let e03 = cube_edge(&mesh, Vec3::new(1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, -1.0));
            let e04 = cube_edge(&mesh, Vec3::new(1.0, -1.0, -1.0), Vec3::new(1.0, -1.0, 1.0));
            let result = mesh.bevel(
                &[e01.into(), e03.into(), e04.into()],
                BevelOptions {
                    offset: 0.25,
                    affect: BevelAffect::Edges,
                    segments,
                    profile,
                    vmesh_method: BevelVmeshMethod::Adj,
                    ..Default::default()
                },
            );
            mesh.validate_topology();

            let cap = curved_corner_grid_faces(&mesh, &result);
            assert!(!cap.is_empty(), "{case}: the corner emitted no cap");
            if segments == 2 {
                assert_eq!(cap.len(), 3, "{case}: the forced §8.5.3 pinwheel");
            }

            // The cap's rim is every cap edge that also carries a non-cap face — the
            // arriving strips' terminal chains, `segments` of them per sector.
            let conflicts = winding_conflicts(&mesh);
            let mut rim = 0usize;
            let mut dissenting = 0usize;
            for &face in &cap {
                for loop_key in mesh.face_loop_keys(face) {
                    let edge = mesh.loop_edge(loop_key);
                    let from = mesh.loop_vertex(loop_key);
                    let to = mesh.loop_vertex(mesh.loop_next(loop_key));
                    let conflicting =
                        conflicts.contains(&(from, to)) || conflicts.contains(&(to, from));
                    let internal = mesh
                        .edge_face_keys(edge)
                        .iter()
                        .all(|neighbour| cap.contains(neighbour));
                    if internal {
                        // One bit for the whole cap: whatever the rim made it choose,
                        // siblings never disagree with each other. A per-face probe would
                        // flip the dissenting sector's cells alone and crack the cap here
                        // while leaving the rim tally unchanged.
                        assert!(
                            !conflicting,
                            "{case}: two cap faces disagree across their shared edge",
                        );
                        continue;
                    }
                    rim += 1;
                    dissenting += usize::from(conflicting);
                }
            }
            let per_sector = segments as usize;
            assert_eq!(rim, 3 * per_sector, "{case}: three sectors' rim edges");
            assert_eq!(
                dissenting, per_sector,
                "{case}: the cap sides with the two sectors that agree, leaving only the \
                 dissenting sector's rim in conflict",
            );
        }
    }
}

/// Bevel the three spokes of one corner of the front-inverted cube at the square-in
/// sentinel, and report the shell's face count and its cap.
fn miswound_cube_corner_square_in(corner: Vec3, segments: u32) -> (usize, usize) {
    let mut mesh = LoopMesh::new();
    mesh_add_cube_with_front_inverted(&mut mesh);
    let vertex = vert_at(&mesh, corner);
    let geom: Vec<GeomKey> = mesh
        .vert_edge_keys(vertex)
        .into_iter()
        .map(GeomKey::from)
        .collect();
    let result = mesh.bevel(
        &geom,
        BevelOptions {
            offset: 0.25,
            affect: BevelAffect::Edges,
            segments,
            profile: 0.0,
            vmesh_method: BevelVmeshMethod::Adj,
            ..Default::default()
        },
    );
    mesh.validate_topology();
    // The cap is every emitted face lying wholly inside the beveled octant — the same
    // reading `curved_corner_grid_faces` takes, at whichever corner this run beveled.
    // The radius is the octant's own half-diagonal (`offset · √3`, the centre-to-deleted-
    // junction-vertex distance) with slack, so a cap vert sitting exactly on the corner
    // is not decided by float noise.
    let octant_centre = corner * 0.75;
    let octant_radius = 0.25 * 3.0_f32.sqrt() * 1.01;
    let cap = result
        .faces
        .iter()
        .filter(|&&face| {
            mesh.face_vert_keys(face)
                .iter()
                .all(|&v| (mesh.vertex_position(v) - octant_centre).length() <= octant_radius)
        })
        .count();
    (mesh.face_count(), cap)
}

/// BEVEL.md §8.5.5.5(f.1) step 3 admits a corner only where §8.5.1's tri-corner test
/// *fires*, and that test's dihedral band is a **signed** sum — so the one face whose
/// winding dissents from its neighbours (§7.8.1(b)) folds the two edges it flanks the
/// other way and drops the total to `π/2`, a whole `π` short of the `3·π/2` the band
/// centres on. Each of the four corners of the reversed face is therefore *refused*, and a
/// refusal routes the junction to step 4's general fullness grid — (f.4b) — even though its
/// boundvert ring is a perfect octant by position and the position-only meet geometry
/// cannot tell it from the four clear corners, which still take (f.4a)'s weld.
///
/// The census separates the two fills without reading a coordinate: the weld emits no ring
/// quads and at most one centre `s`-gon, so it lands far below the grid's count.
#[test]
fn adj_square_in_weld_declines_at_a_miswound_junction() {
    let straddling = [
        Vec3::new(1.0, -1.0, -1.0),
        Vec3::new(-1.0, -1.0, -1.0),
        Vec3::new(-1.0, 1.0, -1.0),
        Vec3::new(1.0, 1.0, -1.0),
    ];
    let clear = [
        Vec3::new(1.0, -1.0, 1.0),
        Vec3::new(-1.0, -1.0, 1.0),
        Vec3::new(-1.0, 1.0, 1.0),
        Vec3::new(1.0, 1.0, 1.0),
    ];
    // Both censuses are **measured**, not derived: they are the shell's face counts at
    // `segments = 2..=7` for the general fullness grid and for the weld respectively.
    // A successor changing a fill's face output should re-measure them, not "correct"
    // them toward a formula.
    let grid_census = [15usize, 22, 30, 40, 51, 64];
    let weld_census = [12usize, 16, 18, 22, 24, 28];

    for (index, segments) in (2u32..=7).enumerate() {
        for corner in straddling {
            let (faces, cap) = miswound_cube_corner_square_in(corner, segments);
            assert_eq!(
                faces, grid_census[index],
                "corner {corner:?}, segments = {segments}: the general grid's census",
            );
            assert!(
                cap > 1,
                "corner {corner:?}, segments = {segments}: a whole grid, not the weld's \
                 lone `s`-gon (cap held {cap} face(s))",
            );
        }
        for corner in clear {
            let (faces, cap) = miswound_cube_corner_square_in(corner, segments);
            assert_eq!(
                faces, weld_census[index],
                "corner {corner:?}, segments = {segments}: clear of the reversed face, so \
                 the weld still fires",
            );
            assert_eq!(
                cap,
                usize::from(segments % 2 == 1),
                "corner {corner:?}, segments = {segments}: the weld's whole face output is \
                 one centre `s`-gon at odd parity and nothing at even",
            );
        }
    }
}

/// The §8.5.1 band's total-angle deviation itself, corner by corner, which is what the
/// dispatch above reads. Three claims:
///
/// - a dissenting face costs the sum `π` — its two edges fold against the other one
///   (`π/2 − π/2 − π/2`), so the deviation is `π`, outside both the `π/4` smooth window
///   and the `π/16` square one;
/// - the four corners clear of it read `0` — the band is not a blanket decline on a shell
///   that happens to carry a reversed face somewhere;
/// - **turning the whole shell inside out changes nothing.** Inverting every face reverses
///   the disk walk and negates every term, and the two shells carry identical local data at
///   a corner, so only the sum's magnitude is a property of the corner. This is the claim a
///   naive convex-positive sum gets wrong, refusing an inward-wound cup outright.
#[test]
fn tri_corner_band_refuses_a_junction_with_a_dissenting_face() {
    let quarter = std::f32::consts::FRAC_PI_4;
    let band_deviation = |build: fn(&mut LoopMesh), corner: Vec3| -> f32 {
        let mut mesh = LoopMesh::new();
        build(&mut mesh);
        let vertex = vert_at(&mesh, corner);
        let spokes = mesh.vert_edge_keys(vertex);
        let corners = mesh.vertex_bevel_corner_ring(vertex, &|edge| spokes.contains(&edge));
        match mesh.junction_dihedral_band(vertex, &corners) {
            DihedralBand::Deviation(deviation) => deviation,
            other => panic!("corner {corner:?}: a closed fan measures a band, got {other:?}"),
        }
    };

    for corner in [
        Vec3::new(1.0, -1.0, -1.0),
        Vec3::new(-1.0, -1.0, -1.0),
        Vec3::new(-1.0, 1.0, -1.0),
        Vec3::new(1.0, 1.0, -1.0),
    ] {
        let deviation = band_deviation(mesh_add_cube_with_front_inverted, corner);
        assert!(
            (deviation - std::f32::consts::PI).abs() <= 1.0e-4,
            "corner {corner:?}: the dissenting face costs the signed sum a whole π \
             (deviation was {deviation})",
        );
        assert!(
            deviation > quarter,
            "corner {corner:?}: outside the smooth band"
        );
    }

    for corner in [
        Vec3::new(1.0, -1.0, 1.0),
        Vec3::new(-1.0, -1.0, 1.0),
        Vec3::new(-1.0, 1.0, 1.0),
        Vec3::new(1.0, 1.0, 1.0),
    ] {
        for build in [
            mesh_add_cube_with_front_inverted as fn(&mut LoopMesh),
            mesh_add_inward_wound_cube,
            mesh_add_cube,
        ] {
            let deviation = band_deviation(build, corner);
            assert!(
                deviation <= 1.0e-4,
                "corner {corner:?}: an orthonormal corner whose faces agree sums to 3·π/2 \
                 whichever way the shell as a whole is wound (deviation was {deviation})",
            );
        }
    }
}

/// Two valence-4 corners that carry the **same four fold magnitudes** — three right angles
/// and one flat splitting edge — built as a matched pair: the convex octant whose flat face
/// is cut in two by a diagonal, and the inner corner of a square notch, where the vertical
/// fold creases the other way.
///
/// `concave` picks the second. Both are closed fans of four faces, consistently wound
/// outward (so nothing dissents, §7.8.1(b)), and both give `edgecount = 4` with exactly one
/// near-flat edge — the `edgecount − 3` the §8.5.1 band asks for.
fn matched_fold_sign_junction(concave: bool) -> (LoopMesh, VertexKey) {
    let mut mesh = LoopMesh::new();
    let apex = mesh.create_vertex(Vec3::ZERO);
    let up = mesh.create_vertex(Vec3::new(0.0, 0.0, 1.0));
    let along_x = mesh.create_vertex(Vec3::new(1.0, 0.0, 0.0));
    let along_y = mesh.create_vertex(Vec3::new(0.0, 1.0, 0.0));
    // The flat edge splitting the apex's horizontal face: inside the 90° wedge for the
    // octant, inside the 270° one for the notch (whose horizontal face is reflex at the
    // apex, so it is split to keep every face corner convex and the fixtures comparable).
    let split = mesh.create_vertex(if concave {
        Vec3::new(-1.0, -1.0, 0.0)
    } else {
        Vec3::new(1.0, 1.0, 0.0)
    });
    let x_rise = mesh.create_vertex(Vec3::new(1.0, 0.0, 1.0));
    let y_rise = mesh.create_vertex(Vec3::new(0.0, 1.0, 1.0));

    if concave {
        mesh.face_create_verts(&[apex, up, x_rise, along_x], FaceCreateFlags::NONE);
        mesh.face_create_verts(&[apex, along_x, split], FaceCreateFlags::NONE);
        mesh.face_create_verts(&[apex, split, along_y], FaceCreateFlags::NONE);
        mesh.face_create_verts(&[apex, along_y, y_rise, up], FaceCreateFlags::NONE);
    } else {
        mesh.face_create_verts(&[apex, along_x, x_rise, up], FaceCreateFlags::NONE);
        mesh.face_create_verts(&[apex, split, along_x], FaceCreateFlags::NONE);
        mesh.face_create_verts(&[apex, along_y, split], FaceCreateFlags::NONE);
        mesh.face_create_verts(&[apex, up, y_rise, along_y], FaceCreateFlags::NONE);
    }
    (mesh, apex)
}

/// BEVEL.md §8.5.1: the band sums the dihedrals **signed**, and a sum of magnitudes is not
/// the same test. The matched pair above differ in exactly one thing — whether the vertical
/// fold creases outward or inward — and their four fold magnitudes are identical
/// (`π/2, π/2, π/2, 0`), so a magnitude sum reads `3·π/2` for both and admits both. The
/// signed sum separates them: the octant keeps `|Σ| = 3·π/2` (deviation `0`, admitted at
/// both windows) while the notch's concave fold subtracts, leaving `|Σ| = π/2` — a whole
/// `π` off target, refused. §8.5.1's own worked note calls this out: a corner carrying a
/// concave leg is the case where the magnitude sum goes wrong.
#[test]
fn tri_corner_band_reads_a_concave_fold_against_its_convex_neighbours() {
    let band = |concave: bool| -> DihedralBand {
        let (mesh, apex) = matched_fold_sign_junction(concave);
        let spokes = mesh.vert_edge_keys(apex);
        let corners = mesh.vertex_bevel_corner_ring(apex, &|edge| spokes.contains(&edge));
        mesh.junction_dihedral_band(apex, &corners)
    };
    let deviation = |concave: bool| -> f32 {
        match band(concave) {
            DihedralBand::Deviation(deviation) => deviation,
            other => panic!("concave = {concave}: a closed fan measures a band, got {other:?}"),
        }
    };

    let convex = deviation(false);
    assert!(
        convex <= 1.0e-4,
        "three convex right angles and a flat splitting edge sum to 3·π/2 (deviation was \
         {convex})",
    );
    let notch = deviation(true);
    assert!(
        (notch - std::f32::consts::PI).abs() <= 1.0e-4,
        "the same four magnitudes with one fold concave sum to π/2, a whole π off target \
         (deviation was {notch})",
    );

    // And the refusal is what the tri-corner gate sees, at either window.
    let apex_position = Vec3::ZERO;
    let octant_ring = [
        Vec3::new(1.0, 1.0, 0.0),
        Vec3::new(0.0, 1.0, 1.0),
        Vec3::new(1.0, 0.0, 1.0),
    ];
    for tolerance in [TRI_CORNER_BAND_SMOOTH, TRI_CORNER_BAND_SQUARE] {
        assert!(
            tri_corner_shape_admits(apex_position, &octant_ring, band(false), tolerance),
            "the octant corner is admitted at tolerance {tolerance}",
        );
        assert!(
            !tri_corner_shape_admits(apex_position, &octant_ring, band(true), tolerance),
            "the notch corner is not, octant-shaped ring or no",
        );
    }
}

/// BEVEL.md §5.1's manifold condition read over a whole rim: the cap's winding bit is an
/// **aggregate** of the per-rim-edge conditions, settled on ties by §7.6.1.2's convention
/// (a tie does nothing) — the tie-break is all that is borrowed from there.
///
/// Pins each arm on a bare triangular rim whose three edges are given faces by hand:
/// unanimity either way, a 2:1 split that a first-found probe would get backwards
/// whichever edge it reached first, the tie (which does **not** reverse), the face-less
/// star (nothing to be manifold against, so the walk stands), and both radial-2 arms — the
/// consistently wound pair that cancels to an abstention, and the **inconsistently** wound
/// pair that keeps both its votes and so outweighs a clean edge on its own. That last arm
/// is what pins the per-loop weighting as a decision: it reverses under per-loop counting
/// (`−2 + 1 = −1`) and does not under per-rim-edge counting (`−1 + 1 = 0`).
#[test]
fn patch_rim_winding_is_an_aggregate_over_the_whole_rim() {
    // `votes[i]` is the sense the face on rim edge `i` walks it: `Some(true)` the same
    // way the cap would (so the cap must reverse), `Some(false)` the other way (keep),
    // `None` no face at all. `doubled` adds a second face to edge 0: `Some(true)` wound
    // consistently with the first, `Some(false)` wound the same way as it (a conflict).
    let build = |votes: [Option<bool>; 3], doubled: Option<bool>| -> (LoopMesh, [VertexKey; 3]) {
        let mut mesh = LoopMesh::new();
        let rim = [
            mesh.create_vertex(Vec3::new(1.0, 0.0, 0.0)),
            mesh.create_vertex(Vec3::new(0.0, 1.0, 0.0)),
            mesh.create_vertex(Vec3::new(0.0, 0.0, 1.0)),
        ];
        for (index, vote) in votes.iter().enumerate() {
            let Some(same_way) = *vote else { continue };
            let from = rim[index];
            let to = rim[(index + 1) % 3];
            let outer = mesh.create_vertex(Vec3::new(index as f32, -1.0, 0.0));
            let ring = if same_way {
                [outer, from, to]
            } else {
                [outer, to, from]
            };
            mesh.face_create_verts(&ring, FaceCreateFlags::NONE);
            if index == 0 {
                if let Some(consistent) = doubled {
                    let opposite = mesh.create_vertex(Vec3::new(0.0, -2.0, 0.0));
                    let ring = if same_way == consistent {
                        [opposite, to, from]
                    } else {
                        [opposite, from, to]
                    };
                    mesh.face_create_verts(&ring, FaceCreateFlags::NONE);
                }
            }
        }
        (mesh, rim)
    };
    let verdict = |votes: [Option<bool>; 3], doubled: Option<bool>| -> bool {
        let (mesh, rim) = build(votes, doubled);
        mesh.patch_rim_needs_reversing((0..3).map(|i| (rim[i], rim[(i + 1) % 3])))
    };

    let keep = Some(false);
    let flip = Some(true);
    assert!(!verdict([keep, keep, keep], None), "unanimous keep");
    assert!(verdict([flip, flip, flip], None), "unanimous reverse");
    assert!(
        verdict([keep, flip, flip], None),
        "the majority decides, not the first rim edge reached",
    );
    assert!(
        !verdict([flip, keep, keep], None),
        "and the same the other way round",
    );
    assert!(
        !verdict([flip, keep, None], None),
        "an even split ties, and a tie does not reverse",
    );
    assert!(
        !verdict([None, None, None], None),
        "a face-less rim leaves the walk standing",
    );
    assert!(
        !verdict([flip, keep, None], Some(true)),
        "a rim edge already at radial 2 and internally consistent abstains — doubling \
         edge 0 must not give it a second, decisive vote",
    );
    assert!(
        verdict([flip, keep, None], Some(false)),
        "a rim edge already at radial 2 and internally *inconsistent* keeps both votes: \
         per-loop weighting is the decision, so it outweighs the one clean edge",
    );
}

/// BEVEL.md §8.5.5.5(f.4a)'s welding rule — `(i, 0, k)` **is the same vertex as**
/// `(i−1, 0, n−k)` for `k <= ⌊n/2⌋` — read off the finished mesh.
///
/// A leg vert therefore belongs to *both* strips that meet at its boundvert (four quads,
/// two per strip) rather than to one, and at even `n` the identification closes
/// transitively over all `s` legs: the octant origin is one vertex carrying `2·s` edges
/// and `2·s` faces. Two coincident verts would each carry half of that.
#[test]
fn adj_square_in_weld_identifies_consecutive_sectors_chains() {
    let (mesh, result) = cube_corner_adj_profile(4, 0.0);
    mesh.validate_topology();

    let origin = square_in_weld_vert_at(&mesh, &result, square_in_weld_origin())
        .expect("even `n` welds every leg's innermost sample at the octant origin");
    assert_eq!(
        mesh.iter_disk_edges(origin).count(),
        6,
        "the octant origin carries one leg edge and one strip rail per sector",
    );
    assert_eq!(
        mesh.vert_face_keys(origin).len(),
        6,
        "two quads from each of the three arriving strips",
    );

    let step = square_in_weld_step(4);
    for boundvert in square_in_weld_boundverts() {
        let leg = square_in_weld_origin() + (1.0 - step) * (boundvert - square_in_weld_origin());
        let leg = square_in_weld_vert_at(&mesh, &result, leg)
            .expect("one mid-leg vert per boundvert, not two coincident ones");
        assert_eq!(
            mesh.vert_face_keys(leg).len(),
            4,
            "a mid-leg vert takes two quads from each of the two strips meeting at its \
             boundvert",
        );
    }
}

/// BEVEL.md §8.5.5.5(f.4a)'s chain layout, quoted: sector `i`'s first half is
/// `(1 − k·b)·e_i` for `0 <= k <= ⌊n/2⌋`, with `b = 2/n` at even `n` and
/// `b = 2/(2·⌊n/2⌋ + √2)` at odd `n`. Under `unit_to_corner` the unit axis point `e_i`
/// is `B_i` and the octant origin is the corner-surface centre, so each leg is the
/// straight run `B_i → c` sampled at those parameters.
///
/// The odd step is strictly smaller than the even one (`√2 > 2·½`), which is what leaves
/// room for the single corner-cutting chord across the turn — so no odd chain is every
/// other sample of the doubled even one.
#[test]
fn adj_square_in_weld_leg_spacing_is_the_closing_chord_step() {
    for segments in [2u32, 3, 4, 5] {
        let step = square_in_weld_step(segments);
        let half = segments as usize / 2;
        let (mesh, result) = cube_corner_adj_profile(segments, 0.0);
        let positions: Vec<Vec3> = result
            .verts
            .iter()
            .map(|&v| mesh.vertex_position(v))
            .collect();
        for boundvert in square_in_weld_boundverts() {
            for k in 1..=half {
                let expected = square_in_weld_origin()
                    + (1.0 - k as f32 * step) * (boundvert - square_in_weld_origin());
                assert_position_present(
                    &positions,
                    expected,
                    &format!("leg sample k = {k} at segments = {segments}"),
                );
            }
        }
        if segments % 2 == 1 {
            assert!(
                step < 2.0 / segments as f32,
                "the odd step is the tighter one (segments = {segments})",
            );
        }
    }
}

/// Both **square** sentinels are refused *by the octant grid* — a square exponent must
/// never reach it, since `r = 1e4` makes branch R's powers numerically dead ((f.2)'s snap
/// note) — and take closed forms of their own from the same tri-corner branch: the
/// square exponent fills with (f.5)'s three flat unit squares, square-in with (f.4a)'s
/// weld, whose entire face output at odd `segments` is one central `s`-gon.
#[test]
fn adj_cube_corner_square_exponents_route_off_the_octant_grid() {
    assert!(
        !adj_octant_exponent(profile_to_exponent(0.99)),
        "square never takes the octant grid"
    );
    assert!(
        !adj_octant_exponent(profile_to_exponent(0.0)),
        "square-in never takes the octant grid"
    );

    // §8.5.5.5(f.5): the square profile at a valence-3 cube corner, odd `segments`, builds
    // the three flat unit squares.
    let (square_mesh, square) = cube_corner_adj_profile(3, 0.99);
    square_mesh.validate_topology();
    assert!(
        !curved_corner_grid_faces(&square_mesh, &square).is_empty(),
        "the square profile builds a three-flat-squares cap",
    );

    // §8.5.5.5(f.4a): square-in welds. No ring quads, so the 3·3 strip quads plus the one
    // central triangle are the whole result.
    let (square_in_mesh, square_in) = cube_corner_adj_profile(3, 0.0);
    square_in_mesh.validate_topology();
    assert_eq!(
        curved_corner_grid_faces(&square_in_mesh, &square_in).len(),
        1,
        "square-in's fill is the central `s`-gon alone",
    );
    assert_eq!(
        square_in.faces.len(),
        10,
        "square-in: all 3·3 beveled-edge strip quads plus the central triangle",
    );
}

/// §8.5.5.5(f.5), the **tri-corner** square grid: at a valence-3 cube corner with **odd**
/// `segments` the fill is three flat unit squares (closed form, no subdivision, no snap),
/// stopping at `(n−1)/n < 1` and leaving the central-triangle hole the odd centre closes.
/// At `segments = 3` that is a `6 quad + 1 triangle` cap (§8.5.5.5(e)'s odd parity: no
/// welded apex, one central `s`-gon).
#[test]
fn adj_square_tricorner_odd_segments_three_flat_squares() {
    let (mesh, result) = cube_corner_adj_profile(3, 0.99);
    mesh.validate_topology();

    let cap = curved_corner_grid_faces(&mesh, &result);
    assert_eq!(
        cap.len(),
        7,
        "3 interior + 3 bridge quads + 1 central triangle"
    );

    let mut quads = 0usize;
    let mut triangles = 0usize;
    for &face in &cap {
        match mesh.face_vert_keys(face).len() {
            4 => quads += 1,
            3 => triangles += 1,
            other => panic!("unexpected {other}-gon in the three-flat-squares cap"),
        }
    }
    assert_eq!(
        (quads, triangles),
        (6, 1),
        "odd-`n` centre closes on one triangle"
    );

    // Odd `segments` weld **no** apex: the original corner `v = (1, −1, −1)` is not carried
    // in as a grid vertex (the three squares stop short of it).
    let corner = Vec3::new(1.0, -1.0, -1.0);
    let welds_v = result
        .verts
        .iter()
        .any(|&v| (mesh.vertex_position(v) - corner).length() < 1.0e-5);
    assert!(
        !welds_v,
        "odd `segments` leave the central triangle, not a welded `v`"
    );
}

/// §8.5.5.5(f.5): a **valence-3** square corner with **even** `segments` is claimed by the
/// parallel-offset ring ((f.1) step 1), *not* the three flat squares — its even-`n` weld is
/// unreachable. The parallel-offset fill welds the original corner `v` at the exact centre
/// (un-rounded — the whole point of the square profile). At `segments = 2` the cap is three
/// quads meeting at that welded `v`.
#[test]
fn adj_square_ring_valence3_even_segments_welds_v_at_centre() {
    let (mesh, result) = cube_corner_adj_profile(2, 0.99);
    mesh.validate_topology();

    let cap = curved_corner_grid_faces(&mesh, &result);
    assert_eq!(
        cap.len(),
        3,
        "3 sector quads welded at the centre (even, no central face)"
    );
    assert!(
        cap.iter().all(|&f| mesh.face_vert_keys(f).len() == 4),
        "every parallel-offset sector cell is a quad",
    );

    // `v` survives as the exact innermost welded vert.
    let corner = Vec3::new(1.0, -1.0, -1.0);
    let welds_v = result
        .verts
        .iter()
        .any(|&v| (mesh.vertex_position(v) - corner).length() < 1.0e-6);
    assert!(
        welds_v,
        "the parallel-offset grid welds `v` un-rounded at the centre"
    );
}

/// §8.5.5.5(f.5), the **general-ring** parallel-offset grid: a valence-4 (`s ≥ 3`) apex at
/// a square profile with **even** `segments`. Each sector is the flat quad patch
/// `B_i, C_i, v, C_{i−1}`, its interior a lattice of offset-line meets; the original corner
/// `v` welds at the exact centre. The topology is the ordinary even-`segments` grid:
/// `s · (segments/2)²` quad cells, one welded apex, no central face.
#[test]
fn adj_square_ring_general_valence4_even_segments_parallel_offset() {
    let apex_position = Vec3::new(0.0, 0.0, 1.4);
    let cap_faces = |mesh: &LoopMesh, result: &BevelResult| -> Vec<FaceKey> {
        result
            .faces
            .iter()
            .copied()
            .filter(|&f| {
                mesh.face_vert_keys(f)
                    .iter()
                    .all(|&v| mesh.vertex_position(v).z > 0.9)
            })
            .collect()
    };
    let welds_v = |mesh: &LoopMesh, result: &BevelResult| -> bool {
        result
            .verts
            .iter()
            .any(|&v| (mesh.vertex_position(v) - apex_position).length() < 1.0e-6)
    };

    for (segments, expected_quads) in [(2_u32, 4_usize), (4, 16)] {
        let (mesh, result, _) = pyramid_apex_adj(segments, 0.97);
        mesh.validate_topology();

        let cap = cap_faces(&mesh, &result);
        assert_eq!(
            cap.len(),
            expected_quads,
            "segments {segments}: s·(segments/2)² = 4·{} parallel-offset cells",
            (segments / 2) * (segments / 2),
        );
        assert!(
            cap.iter().all(|&f| mesh.face_vert_keys(f).len() == 4),
            "segments {segments}: every parallel-offset sector cell is a quad",
        );
        assert!(
            welds_v(&mesh, &result),
            "segments {segments}: `v` survives as the exact welded innermost vert",
        );
    }
}

/// §8.5.5.5(f.5), the **rim run to the seed**: the general-ring square grid's two straight
/// rim rows run `B_i → C_i` and `B_i → C_{i−1}` to the line-line seed `C_i` **on the shared
/// beveled edge** — not along the profile-fold chain, whose fold midpoint folds *off* the
/// edge. On the symmetric pyramid apex the seed `C_i` and the whole seam centreline `CL_i`
/// (which runs `C_i → v`) therefore lie exactly on their slant edge's line through `v`. This
/// pins the rim redirect: every fold-rim vert (`C_i`) and every non-apex seam vert
/// (`CL_i[1..h]`) must sit on a slant line — `sectors · h` of them in all — whereas the
/// pre-redirect fold-chain rim left the `C_i` verts off every slant.
#[test]
fn adj_square_ring_general_ring_rim_runs_to_the_seed_on_the_beveled_edge() {
    let apex = Vec3::new(0.0, 0.0, 1.4);
    let bases = [
        Vec3::new(1.0, 1.0, 0.0),
        Vec3::new(-1.0, 1.0, 0.0),
        Vec3::new(-1.0, -1.0, 0.0),
        Vec3::new(1.0, -1.0, 0.0),
    ];
    let on_a_slant = |p: Vec3| -> bool {
        if (p - apex).length() < 1.0e-4 {
            return false; // the welded apex sits on every slant trivially — exclude it
        }
        bases
            .iter()
            .any(|&b| (p - apex).cross(b - apex).length() < 1.0e-3)
    };

    for segments in [2_u32, 4] {
        let half = (segments / 2) as usize;
        let (mesh, result, _) = pyramid_apex_adj(segments, 0.97);

        let on_slant = result
            .verts
            .iter()
            .filter(|&&v| mesh.vertex_position(v).z > 0.9)
            .filter(|&&v| on_a_slant(mesh.vertex_position(v)))
            .count();
        assert_eq!(
            on_slant,
            4 * half,
            "segments {segments}: the {} fold-rim + seam verts must lie on their slant \
             edge lines (the seed run), not on the off-edge profile fold",
            4 * half,
        );
    }
}

/// §8.5.5.5(f.5), the **centreline seed**: at a square profile the general-ring grid's
/// `C_i` is the exact line-line meet **on the shared beveled edge**, resolved by the
/// boundvert-granularity seeding pass — *not* the profile chain's midpoint, which folds
/// off the edge. This pins the pass's indexing (`edge_out(i) = opening_edges[i + 1]`),
/// its foot-on-the-edge-line construction, and its four-fold symmetry on a synthetic
/// convex valence-4 apex, independent of the fill's chain machinery.
#[test]
fn square_ring_centreline_seed_lands_on_the_beveled_edge_line() {
    let mut mesh = LoopMesh::new();
    let vertex = mesh.create_vertex(Vec3::ZERO);

    // Four convex beveled spokes splaying down-and-out, four-fold symmetric about z.
    let directions: Vec<Vec3> = [
        Vec3::new(1.0, 1.0, -1.0),
        Vec3::new(-1.0, 1.0, -1.0),
        Vec3::new(-1.0, -1.0, -1.0),
        Vec3::new(1.0, -1.0, -1.0),
    ]
    .into_iter()
    .map(|d| d.normalize())
    .collect();
    // Tips two units out; boundverts half a unit out on each adjacent-spoke bisector.
    let tips: Vec<Vec3> = directions.iter().map(|&d| d * 2.0).collect();
    let boundverts: Vec<Vec3> = (0..4)
        .map(|i| 0.5 * (directions[i] + directions[(i + 1) % 4]).normalize())
        .collect();

    let cut = JunctionCut {
        vertex,
        boundverts: Vec::new(),
        opening_edges: Vec::new(),
        outward: Vec3::ZERO,
        central_representative: FaceTemplate::default(),
        sector_representatives: Vec::new(),
        centre_column_representatives: Vec::new(),
        dihedral_band: DihedralBand::Unmeasurable,
        pipe_override: false,
        square_in_weld: false,
        opening_edge_directions: directions.clone(),
        opening_edge_tips: tips,
        // Every sector of this convex apex classifies non-reflex, so every boundvert is a
        // case-3 proposer.
        boundvert_non_reflex: vec![true; 4],
    };

    let seeds = square_ring_centreline_seeds(
        &cut.opening_edge_directions,
        &cut.opening_edge_tips,
        &boundverts,
        &cut.boundvert_non_reflex,
        Vec3::ZERO,
        &[],
    );
    assert_eq!(seeds.len(), 4);

    for i in 0..4 {
        let edge_out = directions[(i + 1) % 4];
        // Defining (f.5) property: the seed lies on edge_out(i)'s line through `v` — a
        // chain-midpoint seed would sit off the edge, failing this.
        assert!(
            seeds[i].cross(edge_out).length() < 1.0e-5,
            "seed {i} must lie on its shared beveled edge line",
        );
        // It is a genuine interior fold, swept down the spoke from `v` (not back at `v`).
        assert!(
            seeds[i].dot(edge_out) > 1.0e-4,
            "seed {i} sweeps out along the edge, away from `v`",
        );
    }

    // Four-fold symmetry: equal reach from `v` and equal height.
    let reach0 = seeds[0].length();
    for i in 1..4 {
        assert!(
            (seeds[i].length() - reach0).abs() < 1.0e-5,
            "symmetric reach from `v`",
        );
        assert!((seeds[i].z - seeds[0].z).abs() < 1.0e-5, "symmetric height",);
    }
}

/// §8.5.5.5(f.5) **pass-2 fallback**: a centreline both of whose flanking parallel-offset
/// proposals degenerate — the companion line runs (anti)parallel to the shared edge, so its
/// foot is at infinity — is left unseeded by pass 1 and then seeded by projecting the
/// sector's *later* boundvert onto the shared beveled-edge segment. A **straight sector**
/// reaches pass 2 twice over — §8.9.1 classifies it out of case 3, and its companion line
/// would be (anti)parallel to the edge anyway. Here `B_1` and `B_2` are straight (edges
/// collinear along ±x), which strands only `CL_1`; every other centreline keeps a live
/// proposer.
#[test]
fn square_ring_centreline_seed_straight_sector_falls_to_pass_two() {
    let axis = Vec3::new(1.0, 0.0, 0.0);
    // Edges 1..=3 collinear (±x): `B_1` (edges +x, −x) and `B_2` (edges −x, +x) are
    // straight, so `CL_1`'s two proposers — companions parallel to its shared edge — both
    // vanish.
    let directions = vec![Vec3::new(1.0, 1.0, -1.0).normalize(), axis, -axis, axis];
    let tips: Vec<Vec3> = directions.iter().map(|&d| d * 2.0).collect();
    let boundverts = vec![
        Vec3::new(0.4, 0.3, -0.3),
        Vec3::new(0.3, 0.1, 0.0),
        Vec3::new(-0.9, -0.1, 0.0),
        Vec3::new(0.5, -0.3, -0.2),
    ];
    let seeds = square_ring_centreline_seeds(
        &directions,
        &tips,
        &boundverts,
        &[true, false, false, true],
        Vec3::ZERO,
        &[],
    );

    // CL_1 (shared edge directions[2] = −x) is the pass-2 seed: the closest point of the
    // segment (`v` → tips[2]) to the later boundvert `B_2`.
    let expected = closest_point_on_segment(boundverts[2], Vec3::ZERO, tips[2]);
    assert!(
        (seeds[1] - expected).length() < 1.0e-6,
        "a straight sector strands CL_1 onto pass 2's edge-segment projection",
    );
    // The straight sector genuinely reached pass 2 (its seed is the segment projection, not
    // the `v` apex the projection would clamp to if `B_2` sat on the wrong side).
    assert!(
        (seeds[1] - Vec3::ZERO).length() > 0.1,
        "the pass-2 projection lands in the segment interior, not clamped to v",
    );
    // Every seed still lands on its own shared edge line.
    for i in 0..4 {
        let edge_out = directions[(i + 1) % 4];
        assert!(
            seeds[i].cross(edge_out).length() < 1.0e-5,
            "seed {i} lies on its shared beveled edge line",
        );
    }
}

/// §8.5.5.5(f.5) arbitration, the **near-flat divergence**: on a strongly-oblique ring the
/// two proposals a centreline receives sit far apart along the shared edge, because a
/// planar disk sector of swept angle `θ` puts its boundvert's feet at `offset / sin θ` and
/// that parameter diverges at **both** ends of the range. So it is the *near-flat*
/// boundvert that contributes the **far** proposal and its neighbour the near one, and
/// closer-to-`v` keeps the neighbour's. Every sector here is non-reflex (the widest spans
/// `175°`), so every boundvert passes case 3's gate and both proposals per centreline are
/// generated — dropping either misplaces the seed by a whole fraction of the spoke.
#[test]
fn square_ring_centreline_seed_near_flat_keeps_near_vertex_proposal() {
    let vertex = Vec3::ZERO;
    // Three spokes bunched near 0° plus one at 215°, all tilted down: sectors of 20°, 20°,
    // 175° and 145° — the widest is near-flat but still comfortably non-reflex.
    let zc = -0.3_f32;
    let azimuths = [0.0_f32, 20.0, 40.0, 215.0];
    let directions: Vec<Vec3> = azimuths
        .iter()
        .map(|&a| {
            let r = a.to_radians();
            Vec3::new(r.cos(), r.sin(), zc).normalize()
        })
        .collect();
    let tips: Vec<Vec3> = directions.iter().map(|&d| d * 2.0).collect();
    let offset = 0.3_f32;
    let boundverts: Vec<Vec3> = (0..4)
        .map(|i| {
            let da = directions[i];
            let db = directions[(i + 1) % 4];
            vertex + offset * (da + db) / da.cross(db).length()
        })
        .collect();

    let seeds =
        square_ring_centreline_seeds(&directions, &tips, &boundverts, &[true; 4], vertex, &[]);

    let mut saw_divergent_spoke = false;
    for i in 0..4 {
        let edge_out = directions[(i + 1) % 4];
        // The two flanking proposals for CL_i.
        let from_i = line_line_nearest_on_first(vertex, edge_out, boundverts[i], directions[i]);
        let from_next = line_line_nearest_on_first(
            vertex,
            edge_out,
            boundverts[(i + 1) % 4],
            directions[(i + 2) % 4],
        );
        let proposals: Vec<Vec3> = [from_i, from_next].into_iter().flatten().collect();
        let nearer = proposals
            .iter()
            .copied()
            .min_by(|a, b| {
                (*a - vertex)
                    .length_squared()
                    .partial_cmp(&(*b - vertex).length_squared())
                    .unwrap()
            })
            .expect("each centreline has at least one flanking proposal");
        assert!(
            (seeds[i] - nearer).length() < 1.0e-5,
            "seed {i} keeps the near-`v` flanking proposal, not the farther one",
        );
        // At the near-flat spoke the two proposals sit far apart, the far one contributed
        // by the very-open boundvert.
        if proposals.len() == 2 && (proposals[0] - proposals[1]).length() > 0.3 {
            saw_divergent_spoke = true;
            let farther = proposals
                .iter()
                .copied()
                .max_by(|a, b| {
                    (*a - vertex)
                        .length_squared()
                        .partial_cmp(&(*b - vertex).length_squared())
                        .unwrap()
                })
                .unwrap();
            assert!(
                (seeds[i] - farther).length() > 0.3,
                "the near-flat seed is the NEAR proposal, not the far one",
            );
        }
    }
    assert!(
        saw_divergent_spoke,
        "the fixture must exercise the near-flat divergent-proposal regime",
    );
}

/// §8.5.5.5(f.5) **case 4, the classification gate**: a reflex-classified boundvert
/// proposes nothing, for either of its two centrelines.
///
/// Flat fan, spokes at `0° / 30° / 60°`, `offset = 0.2` — sector 2 sweeps `300°` and is
/// reflex. Its meet-point's case-3 feet are perfectly well *defined*: each sits at
/// `offset / sin 300° = −0.2309` along its bounding edge, i.e. **behind** `v`, off the
/// beveled-edge segment. And `|−0.2309| < 0.4`, the parameter its non-reflex neighbours'
/// feet sit at (`offset / sin 30°`), so closer-to-`v` would hand *both* of those
/// centrelines the behind-`v` point. Only gating on the classification before the foot is
/// computed keeps them in front of the junction.
#[test]
fn square_ring_centreline_seeds_reflex_boundvert_proposes_nothing() {
    let vertex = Vec3::ZERO;
    let spoke = |degrees: f32| {
        let r = degrees.to_radians();
        Vec3::new(r.cos(), r.sin(), 0.0)
    };
    let directions = vec![spoke(0.0), spoke(30.0), spoke(60.0)];
    let tips: Vec<Vec3> = directions.iter().map(|&d| d * 2.0).collect();
    let offset = 0.2_f32;
    // The two non-reflex meets sit on the inner bisector; the reflex meet sits on the
    // *opposite* bisector, inside its own `300°` wedge.
    let meet = |i: usize, reflex: bool| {
        let da = directions[i];
        let db = directions[(i + 1) % 3];
        let sign = if reflex { -1.0 } else { 1.0 };
        sign * offset * (da + db) / da.cross(db).length()
    };
    let boundverts = vec![meet(0, false), meet(1, false), meet(2, true)];

    let gated = square_ring_centreline_seeds(
        &directions,
        &tips,
        &boundverts,
        &[true, true, false],
        vertex,
        &[],
    );
    // Both of the reflex boundvert's centrelines fall to its non-reflex neighbour's foot,
    // at `offset / sin 30° = 0.4` along the shared edge — in front of `v`.
    for (sector, edge) in [(1usize, directions[2]), (2usize, directions[0])] {
        assert!(
            gated[sector].cross(edge).length() < 1.0e-5,
            "seed {sector} stays on its shared beveled edge line",
        );
        assert!(
            (gated[sector] - 0.4 * edge).length() < 1.0e-5,
            "seed {sector} is the non-reflex neighbour's foot, in front of the junction",
        );
    }

    // The gate, not the geometry, is what does this: re-run with the reflex boundvert
    // classified non-reflex and both seeds jump behind `v`.
    let ungated =
        square_ring_centreline_seeds(&directions, &tips, &boundverts, &[true; 3], vertex, &[]);
    for (sector, edge) in [(1usize, directions[2]), (2usize, directions[0])] {
        assert!(
            (ungated[sector] + 0.23094 * edge).length() < 1.0e-4,
            "without the gate the reflex foot at offset/sin 300° wins sector {sector}",
        );
        assert!(
            ungated[sector].dot(edge) < 0.0,
            "…and it sits behind the junction, off the beveled-edge segment",
        );
    }
}

/// §8.5.5.5(f.5) case 4 → **pass 2** on an inner-mitered ring: when the reflex boundvert's
/// two centrelines have no other proposer — both neighbours were swallowed by firing inner
/// miters — the gate leaves them to pass 2's projection onto the beveled-edge **segment**,
/// which is clamped and therefore in front of `v`. This is the shape of the measured
/// divergence: the reflex proposal there was unopposed, so suppressing it is the whole of
/// the fix.
#[test]
fn square_ring_centreline_seeds_reflex_boundvert_unopposed_falls_to_pass_two() {
    let vertex = Vec3::ZERO;
    let spoke = |degrees: f32| {
        let r = degrees.to_radians();
        Vec3::new(r.cos(), r.sin(), 0.0)
    };
    let (d0, d1, d2) = (spoke(0.0), spoke(30.0), spoke(60.0));
    // The enlarged ring: both non-reflex sectors split into an arc pair, the reflex sector
    // keeps its single meet-point. `edge_in(i)` is the edge shared with the *previous* ring
    // boundvert, so the two internal arc edges read `ZERO`.
    let directions = vec![d0, Vec3::ZERO, d1, Vec3::ZERO, d2];
    let tips = vec![d0 * 2.0, Vec3::ZERO, d1 * 2.0, Vec3::ZERO, d2 * 2.0];
    let reflex_meet = -0.2 * (d2 + d0) / d2.cross(d0).length();
    let boundverts = vec![0.5 * d0, 0.5 * d1, 0.55 * d1, 0.55 * d2, reflex_meet];
    let arc_patch = vec![
        ArcPatchStart {
            index: 0,
            mode: BevelMiter::Arc,
            arc_middle: 0.2 * (d0 + d1),
        },
        ArcPatchStart {
            index: 2,
            mode: BevelMiter::Arc,
            arc_middle: 0.2 * (d1 + d2),
        },
    ];

    let seeds = square_ring_centreline_seeds(
        &directions,
        &tips,
        &boundverts,
        &[true, true, true, true, false],
        vertex,
        &arc_patch,
    );

    // Sector 3 rides `d2` and is preceded by an arc start, so pass 2 projects the *earlier*
    // boundvert; sector 4 rides `d0` and is followed by one, so it projects the *later*.
    let trailing = closest_point_on_segment(boundverts[3], vertex, tips[4]);
    let leading = closest_point_on_segment(boundverts[0], vertex, tips[0]);
    assert!(
        (seeds[3] - trailing).length() < 1.0e-6,
        "the trailing centreline takes pass 2's earlier-boundvert projection",
    );
    assert!(
        (seeds[4] - leading).length() < 1.0e-6,
        "…and the leading one pass 2's later-boundvert projection",
    );
    for sector in [3usize, 4] {
        let edge = directions[(sector + 1) % 5];
        assert!(
            seeds[sector].dot(edge) > 0.0,
            "seed {sector} stays in front of `v`, on the beveled-edge segment",
        );
    }
}

/// §8.5.5.5(f.5) arbitration on an **ordinary convex ring**: the two flanking proposals a
/// centreline receives nearly coincide, so the closer-to-`v` tiebreak is unobservable and
/// the seed matches either. This is the counter-case the near-flat fix must not regress.
#[test]
fn square_ring_centreline_seed_convex_ring_proposals_coincide() {
    let vertex = Vec3::ZERO;
    // A regular, gently-tilted valence-4 ring: every sector convex and near-symmetric.
    let azimuths = [45.0_f32, 135.0, 225.0, 315.0];
    let directions: Vec<Vec3> = azimuths
        .iter()
        .map(|&a| {
            let r = a.to_radians();
            Vec3::new(r.cos(), r.sin(), -0.4).normalize()
        })
        .collect();
    let tips: Vec<Vec3> = directions.iter().map(|&d| d * 2.0).collect();
    let offset = 0.3_f32;
    let boundverts: Vec<Vec3> = (0..4)
        .map(|i| {
            let da = directions[i];
            let db = directions[(i + 1) % 4];
            vertex + offset * (da + db) / da.cross(db).length()
        })
        .collect();
    let seeds =
        square_ring_centreline_seeds(&directions, &tips, &boundverts, &[true; 4], vertex, &[]);

    for i in 0..4 {
        let edge_out = directions[(i + 1) % 4];
        let from_i = line_line_nearest_on_first(vertex, edge_out, boundverts[i], directions[i])
            .expect("convex proposal from B_i");
        let from_next = line_line_nearest_on_first(
            vertex,
            edge_out,
            boundverts[(i + 1) % 4],
            directions[(i + 2) % 4],
        )
        .expect("convex proposal from B_{i+1}");
        assert!(
            (from_i - from_next).length() < 1.0e-3,
            "on a regular convex ring the two flanking proposals coincide (tiebreak inert)",
        );
        // With the two proposals coincident, the arbitration survivor is that shared value:
        // pin the actual seed, not merely the proposals' agreement.
        assert!(
            (seeds[i] - from_i).length() < 1.0e-3,
            "seed {i} lands on the (coincident) flanking parallel-offset proposal",
        );
        assert!(
            seeds[i].cross(edge_out).length() < 1.0e-5,
            "seed {i} lies on its shared beveled edge line",
        );
    }
}

/// BEVEL.md §8.5.5.5(f.5), the **parallel-but-distinct** interior branch: as a sector
/// opens toward flat, the `k`-th and `j`-th offset lines rotate to a shared direction
/// while their base points stay on two *different* rim rows — parallel yet never
/// coincident. The lattice must **route on the directions alone** and interpolate
/// `(i, 0, k)` toward `CL_{i−1}[k]` by `j/h`, *not* fall through to the line solver's
/// skew / degenerate fallback (which — when the solve turns singular — returns the
/// midpoint of the two rim **base** points, off the patch by `|v − B_i|/4`).
///
/// Construction: at the centre vert `(1, 1)` of an `n = 4` patch the two offset lines
/// are exactly parallel iff `v − B_i ∥ C_i − C_{i−1}` (each is a midline of the patch
/// quad `B_i, C_i, v, C_{i−1}`). We pin that config and assert the interpolation, and
/// that it is strictly distinct from the base-midpoint the old coincident-only routing
/// produced.
#[test]
fn square_grid_interior_parallel_offset_lines_interpolate_not_base_midpoint() {
    let vertex = Vec3::ZERO;
    let b0 = Vec3::new(1.0, 0.0, 1.0);
    let c0 = Vec3::new(0.6, 0.3, 0.6);
    let c2 = Vec3::new(0.1, 0.3, 0.1); // c0 - c2 = (0.5,0,0.5) ∥ v - b0 = (-1,0,-1)
                                       // sectors 1 and the rest are unread for sector 0's (1, 1) meet; fill with plausibles.
    let starts = [
        (b0, c0),
        (Vec3::new(-1.0, 1.0, 0.0), Vec3::new(-0.3, 0.4, 0.0)),
        (Vec3::new(0.0, -1.0, 1.0), c2),
    ];
    let interior = AdjInterior::Parallel {
        starts: &starts,
        vertex,
    };
    let placed = interior.position(0, 1, 1, 2);

    // Interpolation target: mid of rim row `(0,0,1)` and centreline `CL_2[1]`.
    let row0_k = 0.5 * (b0 + c0);
    let cl_prev_k = 0.5 * c2; // C_2 + (1/2)(v − C_2), v = 0
    let interp = 0.5 * (row0_k + cl_prev_k);
    assert!(
        (placed - interp).length() < 1.0e-5,
        "parallel offset lines interpolate the rim row toward the centreline by j/h",
    );

    // The pre-fix coincident-only routing fell to the solver's base-point midpoint.
    let col0_j = 0.5 * (b0 + c2);
    let base_midpoint = 0.5 * (row0_k + col0_j);
    assert!(
        (placed - base_midpoint).length() > 0.3,
        "the interpolation is the |v − B_i|/4 correction away from the base midpoint",
    );
}

/// BEVEL.md §8.5.5.5(f.5): a **clean-crossing** interior meet is unchanged by the
/// parallel routing — a mildly-oblique sector keeps the two offset families crossing, so
/// the interior vert is the genuine line-line meet, which at `(1, 1)` bisects both lines.
#[test]
fn square_grid_interior_clean_crossing_takes_the_line_meet() {
    let vertex = Vec3::ZERO;
    // A right-angled sector patch in the z = 0 plane: B at the corner, C_i / C_{i−1}
    // along the two axes, v at the far corner — the offset lines cross at 90°.
    let b0 = Vec3::new(1.0, 1.0, 0.0);
    let c0 = Vec3::new(0.0, 1.0, 0.0);
    let c2 = Vec3::new(1.0, 0.0, 0.0);
    let starts = [
        (b0, c0),
        (Vec3::new(-1.0, -1.0, 0.0), Vec3::new(-0.5, 0.0, 0.0)),
        (Vec3::new(0.0, -1.0, 0.0), c2),
    ];
    let interior = AdjInterior::Parallel {
        starts: &starts,
        vertex,
    };
    let placed = interior.position(0, 1, 1, 2);
    // Midlines of the unit square `B=(1,1), C_i=(0,1), v=(0,0), C_{i−1}=(1,0)` cross at
    // the centroid `(0.5, 0.5)`.
    assert!(
        (placed - Vec3::new(0.5, 0.5, 0.0)).length() < 1.0e-5,
        "a clean crossing lands at the patch centroid, not an interpolation artefact",
    );
}

/// BEVEL.md §8.5.5.5(f.5) seed pass on a **strongly-oblique** (near-flat, bunched)
/// valence-4 ring: every sector's centreline is still seeded by a **pass-1 sharp** foot
/// on its shared beveled edge (the closer-to-`v` of its two bounding boundverts'
/// proposals) — no sector falls through to pass 2. This pins that obliquity alone does
/// not divert the seed pass off the parallel-offset construction: every sector here spans
/// well under `180°`, so §8.9.1's verdict is non-reflex throughout and case 3's gate is
/// open at every boundvert.
#[test]
fn square_ring_centreline_seeds_oblique_ring_stay_pass_one_sharp() {
    let vertex = Vec3::ZERO;
    // Three spokes bunched near 0° plus one at 200°, all splaying down (near-flat apex):
    // a strongly-oblique ring whose interior meets reach the near-parallel regime. Sectors
    // of 20°, 20°, 160° and 160° — bunched, but none of them near the sign boundary.
    let zc = -0.3_f32;
    let azimuths = [0.0_f32, 20.0, 40.0, 200.0];
    let directions: Vec<Vec3> = azimuths
        .iter()
        .map(|&a| {
            let r = a.to_radians();
            Vec3::new(r.cos(), r.sin(), zc).normalize()
        })
        .collect();
    let tips: Vec<Vec3> = directions.iter().map(|&d| d * 2.0).collect();
    let offset = 0.3_f32;
    let boundverts: Vec<Vec3> = (0..4)
        .map(|i| {
            let da = directions[i];
            let db = directions[(i + 1) % 4];
            vertex + offset * (da + db) / da.cross(db).length()
        })
        .collect();
    let seeds =
        square_ring_centreline_seeds(&directions, &tips, &boundverts, &[true; 4], vertex, &[]);

    for i in 0..4 {
        let edge_out = directions[(i + 1) % 4];
        assert!(
            seeds[i].cross(edge_out).length() < 1.0e-5,
            "seed {i} lies on its shared beveled edge line",
        );
        // The two pass-1 sharp proposals for CL_i: from B_i (via its edge_in) and from
        // B_{i+1} (via its edge_out); the closer-to-`v` one is the seed.
        let from_i = line_line_nearest_on_first(vertex, edge_out, boundverts[i], directions[i]);
        let from_next = line_line_nearest_on_first(
            vertex,
            edge_out,
            boundverts[(i + 1) % 4],
            directions[(i + 2) % 4],
        );
        let closer = [from_i, from_next]
            .into_iter()
            .flatten()
            .min_by(|a, b| a.length_squared().partial_cmp(&b.length_squared()).unwrap())
            .expect("a convex sector has at least one pass-1 proposal");
        assert!(
            (seeds[i] - closer).length() < 1.0e-5,
            "seed {i} is the closer-to-v pass-1 sharp foot, not a pass-2 projection",
        );
    }
}

/// B1 regression (BEVEL.md §8.5.5.5(f.2)): the unit-octant frame's conditioning test is
/// **scale-free**. The frame's three axes are the corner's boundvert spokes, so its raw
/// determinant is `±offset³` at the symmetric cube corner — an absolute threshold on it
/// would refuse an ordinary narrow bevel and leave the junction an unfilled hole. A
/// `1e-4` offset is four orders of magnitude below the fixtures' width and still a
/// perfectly conditioned corner: every fill path that rides the octant frame (the
/// circular and curved `segments ≥ 3` grids, and the non-circular `segments = 2` apex)
/// must build its cap there exactly as at `offset = 0.25`.
#[test]
fn adj_cube_corner_narrow_offset_still_builds_the_octant_cap() {
    let offset = 1.0e-4_f32;

    for (segments, profile, cap_faces) in [
        (3_u32, 0.5_f32, 7_usize), // circular grid: 6 seam quads + 1 central triangle
        (3, 0.7, 7),               // curved grid: same topology, bulged
        (4, 0.7, 12),              // curved grid, even parity: welded apex, 3·(4/2)² quads
        (2, 0.25, 3),              // flat chamfer apex: the three-quad pinwheel
    ] {
        let (mesh, result) = cube_corner_adj_profile_offset(segments, profile, offset);
        mesh.validate_topology();
        assert_eq!(
            corner_grid_faces_at_offset(&mesh, &result, offset).len(),
            cap_faces,
            "segments {segments} / profile {profile}: the cap builds at a narrow offset",
        );
    }

    // The frame is refused only for a genuinely degenerate *shape*, never for a small
    // one: a flattened (coplanar-spoke) corner still declines.
    let corner = Vec3::new(1.0, -1.0, -1.0);
    assert!(
        UnitOctant::from_corner(
            [
                corner + Vec3::new(offset, 0.0, 0.0),
                corner + Vec3::new(0.0, offset, 0.0),
                corner + Vec3::new(offset, offset, 0.0),
            ],
            corner,
        )
        .is_none(),
        "coplanar spokes are still singular, at any scale",
    );
    assert!(
        UnitOctant::from_corner(
            [
                corner + Vec3::new(offset, 0.0, 0.0),
                corner + Vec3::new(0.0, offset, 0.0),
                corner + Vec3::new(0.0, 0.0, offset),
            ],
            corner,
        )
        .is_some(),
        "an orthonormal corner is well conditioned at any scale",
    );
}

/// §8.9.1 reflex reference: the signed test's `n_ref` walks a forward-face → back-face →
/// `n_v` fallback chain. Pins each rung: the forward (first-corner) face wins when usable,
/// a degenerate forward face falls through to the back (last-corner) face, and an empty
/// sector drops to the junction vertex normal.
#[test]
fn reflex_reference_normal_walks_forward_back_nv_chain() {
    // Forward (first-corner) face wins when it is non-degenerate.
    {
        let mut mesh = LoopMesh::new();
        let v = mesh.create_vertex(Vec3::ZERO);
        let a = mesh.create_vertex(Vec3::new(1.0, 0.0, 0.0));
        let b = mesh.create_vertex(Vec3::new(0.0, 1.0, 0.0));
        let c = mesh.create_vertex(Vec3::new(0.0, 1.0, 1.0));
        let e_va = mesh.create_edge([v, a], EdgeCreateFlags::NONE);
        let e_ab = mesh.create_edge([a, b], EdgeCreateFlags::NONE);
        let e_bv = mesh.create_edge([b, v], EdgeCreateFlags::NONE);
        let forward = mesh.face_create(&[v, a, b], &[e_va, e_ab, e_bv], FaceCreateFlags::NONE);
        let e_ac = mesh.create_edge([a, c], EdgeCreateFlags::NONE);
        let e_cv = mesh.create_edge([c, v], EdgeCreateFlags::NONE);
        let back = mesh.face_create(&[v, a, c], &[e_va, e_ac, e_cv], FaceCreateFlags::NONE);
        let sector = Sector {
            bounding_edges: vec![e_va, e_bv],
            corners: vec![(e_va, forward), (e_ac, back)],
        };
        let reference = mesh.reflex_reference_normal(&sector, v);
        assert!(reference.abs_diff_eq(mesh.face_plane_normal(forward), 1.0e-6));
        assert!(!reference.abs_diff_eq(mesh.face_plane_normal(back), 1.0e-6));
    }

    // A degenerate forward face falls through to the back (last-corner) face.
    {
        let mut mesh = LoopMesh::new();
        let v = mesh.create_vertex(Vec3::ZERO);
        let d1 = mesh.create_vertex(Vec3::new(1.0, 0.0, 0.0));
        let d2 = mesh.create_vertex(Vec3::new(2.0, 0.0, 0.0));
        let e_vd1 = mesh.create_edge([v, d1], EdgeCreateFlags::NONE);
        let e_d1d2 = mesh.create_edge([d1, d2], EdgeCreateFlags::NONE);
        let e_d2v = mesh.create_edge([d2, v], EdgeCreateFlags::NONE);
        // Collinear corners — zero Newell area, so this face yields no usable normal.
        let degenerate =
            mesh.face_create(&[v, d1, d2], &[e_vd1, e_d1d2, e_d2v], FaceCreateFlags::NONE);
        let b = mesh.create_vertex(Vec3::new(0.0, 1.0, 0.0));
        let e_d1b = mesh.create_edge([d1, b], EdgeCreateFlags::NONE);
        let e_bv = mesh.create_edge([b, v], EdgeCreateFlags::NONE);
        let back = mesh.face_create(&[v, d1, b], &[e_vd1, e_d1b, e_bv], FaceCreateFlags::NONE);
        assert!(mesh.face_newell_normal(degenerate).is_none());
        let sector = Sector {
            bounding_edges: vec![e_vd1, e_bv],
            corners: vec![(e_vd1, degenerate), (e_d1b, back)],
        };
        let reference = mesh.reflex_reference_normal(&sector, v);
        assert!(reference.abs_diff_eq(mesh.face_plane_normal(back), 1.0e-6));
    }

    // An empty sector drops to the junction vertex normal `n_v`.
    {
        let mut mesh = LoopMesh::new();
        let v = mesh.create_vertex(Vec3::ZERO);
        let a = mesh.create_vertex(Vec3::new(1.0, 0.0, 0.0));
        let b = mesh.create_vertex(Vec3::new(0.0, 1.0, 0.0));
        let e_va = mesh.create_edge([v, a], EdgeCreateFlags::NONE);
        let e_ab = mesh.create_edge([a, b], EdgeCreateFlags::NONE);
        let e_bv = mesh.create_edge([b, v], EdgeCreateFlags::NONE);
        mesh.face_create(&[v, a, b], &[e_va, e_ab, e_bv], FaceCreateFlags::NONE);
        let sector = Sector {
            bounding_edges: vec![e_va, e_bv],
            corners: Vec::new(),
        };
        let reference = mesh.reflex_reference_normal(&sector, v);
        assert!(reference.abs_diff_eq(mesh.junction_outward_normal(v), 1.0e-6));
    }
}

/// B2 (BEVEL.md §8.5.5.5(f.2)): the flat chamfer (`r = 1`) is a member of the tri-corner
/// family, so a `segments = 2` cube corner at `profile = 0.25` takes the forced pinwheel
/// — *not* the general ring's `fullness = 0` planar cap — and its apex is the octant
/// closed form `3^(−1/1)·(1,1,1)`, i.e. the boundvert centroid on the chamfer plane.
/// Pinwheel and general grid now orient by the **same** §5.1 rule — one aggregate bit off
/// the rim they weld to — but they reach it down different dispatch arms and place their
/// apex differently, so the reroute is still worth showing winding-neutral as well as
/// position-neutral. The reference is the **circular**
/// `segments = 2` corner, whose pinwheel is the strictly-pinned forced case (§8.5.4):
/// the flat corner emits the same three faces over the same rim, traversed in the same
/// direction, differing only in where the apex sits.
#[test]
fn adj_cube_corner_flat_chamfer_segments_2_pinwheel() {
    let (mesh, result) = cube_corner_adj_profile(2, 0.25);
    mesh.validate_topology();
    assert_eq!(
        profile_to_exponent(0.25),
        1.0,
        "profile 0.25 is the straight chord"
    );

    let cap = curved_corner_grid_faces(&mesh, &result);
    assert_eq!(
        cap.len(),
        3,
        "the three-quad pinwheel (§8.5.3), no central face"
    );
    for &face in &cap {
        assert_eq!(
            mesh.face_vert_keys(face).len(),
            4,
            "each pinwheel cell is a quad"
        );
    }

    // The apex: `unit_to_corner`'s image of `3^(−1) ·(1,1,1)` is the mean of the three
    // boundverts, each the in-face meet point one offset from both of its face's beveled
    // edges (so `offset·√2` from the corner along a face diagonal, invariant 1).
    let corner = Vec3::new(1.0, -1.0, -1.0);
    let boundverts = [
        corner + Vec3::new(-0.25, 0.25, 0.0),
        corner + Vec3::new(-0.25, 0.0, 0.25),
        corner + Vec3::new(0.0, 0.25, 0.25),
    ];
    let centroid = boundverts.iter().fold(Vec3::ZERO, |acc, &b| acc + b) / 3.0;
    let apex: Vec<VertexKey> = result
        .verts
        .iter()
        .copied()
        .filter(|&v| (mesh.vertex_position(v) - centroid).length() < 1.0e-5)
        .collect();
    assert_eq!(apex.len(), 1, "one welded apex, at the boundvert centroid");
    assert_eq!(
        mesh.face_keys()
            .into_iter()
            .filter(|&f| mesh.face_vert_keys(f).contains(&apex[0]))
            .count(),
        3,
        "the three pinwheel quads share it",
    );

    for &face in &cap {
        for loop_key in mesh.face_loop_keys(face) {
            assert_eq!(
                mesh.edge_loop_keys(mesh.loop_edge(loop_key)).len(),
                2,
                "every cap edge is radial-2 — the cap shares the rim, it does not re-mint it",
            );
        }
    }

    // Winding: identical to the circular corner's, face for face. Both runs build the
    // same rim in the same order, so a cap face's *cyclic key sequence* is its winding;
    // rotating each to start at its apex makes the three comparable as a set.
    let cap_winding = |mesh: &LoopMesh, faces: &[FaceKey], apex: VertexKey| {
        let mut wound: Vec<Vec<VertexKey>> = faces
            .iter()
            .map(|&face| {
                let verts = mesh.face_vert_keys(face);
                let at = verts.iter().position(|&v| v == apex).expect("apex corner");
                verts
                    .iter()
                    .cycle()
                    .skip(at)
                    .take(verts.len())
                    .copied()
                    .collect()
            })
            .collect();
        wound.sort_by_key(|face| face[1].as_usize());
        wound
    };

    let (round_mesh, round_result) = cube_corner_adj_profile(2, 0.5);
    let round_cap = curved_corner_grid_faces(&round_mesh, &round_result);
    let round_apex = round_result
        .verts
        .iter()
        .copied()
        .find(|&v| (round_mesh.vertex_position(v) - adj_apex_position()).length() < 1.0e-4)
        .expect("the circular corner's sphere-snapped apex");
    assert_eq!(
        cap_winding(&mesh, &cap, apex[0]),
        cap_winding(&round_mesh, &round_cap, round_apex),
        "the flat chamfer's pinwheel winds exactly as the pinned circular one",
    );
}

/// BEVEL.md §8.5.5.5(f.3), the `fullness` law. The circular table runs to
/// `segments = 11` and **stops**: at 12 and beyond even the circular profile falls
/// through to the regression F3, which lands *below* every table entry from 5 up. The
/// law genuinely steps down there — an implementation that extrapolates the table's
/// trend ("large odd counts settle toward ≈0.647") is ≈0.07 of fullness off. F3 is a
/// regression in the raw `profile` scalar, not in the exponent, and it is free to
/// exceed `1` on a bulging profile.
#[test]
fn adj_fullness_law() {
    // F2: the straight chord is exactly zero, and the circular table is exact.
    assert_eq!(adj_fullness(0.25, 4), 0.0);
    for (segments, expected) in [
        (2_u32, 0.559_f32),
        (3, 0.642),
        (4, 0.551),
        (5, 0.646),
        (6, 0.624),
        (7, 0.646),
        (8, 0.619),
        (9, 0.647),
        (10, 0.639),
        (11, 0.647),
    ] {
        assert_eq!(
            adj_fullness(0.5, segments),
            expected,
            "circle at n = {segments}"
        );
    }

    // F3: the table's end is a step *down*, not an asymptote.
    assert!((adj_fullness(0.5, 12) - 0.598664).abs() < 1.0e-5);
    assert!((adj_fullness(0.5, 13) - 0.577726).abs() < 1.0e-5);
    assert!(
        adj_fullness(0.5, 13) < adj_fullness(0.5, 11),
        "the circular law steps down past the table, it does not settle upward",
    );

    // F3 on the curved band, in the raw `profile` scalar. `fullness > 1` is normal on a
    // bulging profile: the fit seeds the centre *past* the original corner `v`.
    assert!((adj_fullness(0.7, 3) - 1.048906).abs() < 1.0e-5);
    assert!((adj_fullness(0.7, 4) - 1.088808).abs() < 1.0e-5);
    assert!((adj_fullness(0.3, 3) - 0.103506).abs() < 1.0e-5);
}

/// BEVEL.md §8.5.5.5(f.3): the extended table is not just tabulated, it is *applied*.
/// At `segments = 8` the general valence-4 cap seeds its centre with the table's
/// `0.619` — the entry a table truncated at 6 could not have — and the extrapolated
/// `0.63` an implementation would otherwise reach for lands the relaxed centre
/// somewhere else entirely.
#[test]
fn adj_fullness_table_past_six_is_applied_to_the_cap() {
    let mut mesh = LoopMesh::new();
    let [apex, b0, b1, b2, b3] = square_pyramid(&mut mesh);
    let apex_position = mesh.vertex_position(apex);
    let base: Vec<Vec3> = [b0, b1, b2, b3]
        .iter()
        .map(|&b| mesh.vertex_position(b))
        .collect();
    let slants: Vec<GeomKey> = [b0, b1, b2, b3]
        .iter()
        .map(|&b| mesh.query_edge_exists(apex, b).expect("slant edge").into())
        .collect();
    let result = mesh.bevel(
        &slants,
        BevelOptions {
            offset: 0.3,
            affect: BevelAffect::Edges,
            segments: 8,
            vmesh_method: BevelVmeshMethod::Adj,
            ..Default::default()
        },
    );
    mesh.validate_topology();

    let (boundverts, profiles) = pyramid_apex_ring(apex_position, &base, 0.3, 0.5);
    assert_eq!(adj_fullness(0.5, 8), 0.619);
    let tabulated = predicted_cap_interior(apex_position, &boundverts, &profiles, 8, 0.619);
    let extrapolated = predicted_cap_interior(apex_position, &boundverts, &profiles, 8, 0.63);
    let centre = tabulated[tabulated.len() - 1];
    let wrong_centre = extrapolated[extrapolated.len() - 1];
    assert!(
        (centre - wrong_centre).length() > 1.0e-4,
        "the two seeds are distinguishable at the emitted centre",
    );
    assert_eq!(
        result_has_vert_near(&mesh, &result, centre),
        1,
        "the cap centre is the net seeded with the table's n = 8 entry",
    );
    assert_eq!(
        result_has_vert_near(&mesh, &result, wrong_centre),
        0,
        "and not the net seeded with the truncated table's extrapolation",
    );
}

/// BEVEL.md §8.5.5.5: a two-edge (`s = 2`) ADJ junction takes no pinwheel fill — the
/// §8.5 forced case needs `s ≥ 3`, so the through-strips still build (exactly as cutoff)
/// with no central apex. The vertex-mode flat line beveled at the tail is *no longer* a
/// stub — per §8.5.5.5(f.3) law F2 it builds the general fullness grid (planar cap).
#[test]
fn adj_s2_non_forced_junction_has_no_pinwheel_fill() {
    let mut mesh = LoopMesh::new();
    mesh_add_cube(&mut mesh);
    let e01 = cube_edge(
        &mesh,
        Vec3::new(1.0, -1.0, -1.0),
        Vec3::new(-1.0, -1.0, -1.0),
    );
    let e03 = cube_edge(&mesh, Vec3::new(1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, -1.0));

    // s = 2 junction at v0, ADJ: no pinwheel fill (the general grid is a stub),
    // but the through-strip geometry still builds and the mesh stays manifold.
    let result = mesh.bevel(
        &[e01.into(), e03.into()],
        BevelOptions {
            offset: 0.25,
            affect: BevelAffect::Edges,
            segments: 2,
            vmesh_method: BevelVmeshMethod::Adj,
            ..Default::default()
        },
    );
    mesh.validate_topology();
    // No welded central apex: with s = 2 the §8.5 forced case does not apply.
    // The apex would have sat along the corner's interior body diagonal; assert
    // there is no extra fan-center vert by checking the new faces are only the
    // through-strip quads (4 = 2 edges × 2 segments), no pinwheel quad.
    assert_eq!(
        result.faces.len(),
        4,
        "s=2 ADJ junction: through-strips only, no pinwheel fill (§8.5.5)",
    );

    // §8.5.5.5(f.3) law F2: the flat-line (`profile = 0.25`, `r == 1.0`) vertex bevel
    // now builds via the general fullness grid — the untouched opposite cube corner
    // (valence 3) at `segments = 2` grows by one general even grid (+6 V / +9 E / +3 F).
    let (bv, be, bf) = counts(&mesh);
    let v_far = vert_at(&mesh, Vec3::new(-1.0, 1.0, 1.0));
    mesh.bevel(
        &[v_far.into()],
        BevelOptions {
            offset: 0.25,
            affect: BevelAffect::Vertices,
            segments: 2,
            profile: 0.25,
            vmesh_method: BevelVmeshMethod::Adj,
            ..Default::default()
        },
    );
    mesh.validate_topology();
    let (av, ae, af) = counts(&mesh);
    assert_eq!(
        (av - bv, ae - be, af - bf),
        (6, 9, 3),
        "flat-line vertex multiseg builds the general even grid (§8.5.5.5(f.3) F2)",
    );
}
