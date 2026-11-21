use super::*;

/// BEVEL.md §8.5.1: the forced-tri-corner trigger predicate read off the boundvert
/// ring. The symmetric equal-offset orthonormal cube corner fires it at every **smooth**
/// exponent (§8.5.5.5(f.1) step 3); an unequal-offset corner, a non-right-angle corner,
/// and the two square exponents all decline.
#[test]
fn adj_tri_corner_trigger_predicate() {
    let mut mesh = LoopMesh::new();
    let junction = mesh.create_vertex(Vec3::ZERO);
    let make_cut = |mesh: &mut LoopMesh, spokes: [Vec3; 3]| JunctionCut {
        vertex: junction,
        boundverts: spokes.iter().map(|&s| mesh.create_vertex(s)).collect(),
        opening_edges: Vec::new(),
        outward: Vec3::ZERO,
        central_representative: FaceTemplate::default(),
        sector_representatives: Vec::new(),
        centre_column_representatives: Vec::new(),
        // The synthetic cuts carry no real corner, so the band verdict is inert here;
        // the geometry test alone decides these assertions.
        dihedral_band: DihedralBand::Unmeasurable,
        pipe_override: false,
        square_in_weld: false,
        opening_edge_directions: Vec::new(),
        opening_edge_tips: Vec::new(),
        boundvert_non_reflex: Vec::new(),
    };

    // Orthonormal equal-offset cube corner: meet-point spokes `d_i + d_j` over the
    // three axis directions — equal length (√2) and pairwise 60° (cos ½).
    let cube = make_cut(
        &mut mesh,
        [
            Vec3::new(1.0, 1.0, 0.0),
            Vec3::new(0.0, 1.0, 1.0),
            Vec3::new(1.0, 0.0, 1.0),
        ],
    );
    assert!(
        mesh.adj_tri_corner_trigger(&cube, 0.5, false),
        "default-profile cube corner fires the §8.5.1 trigger",
    );
    // §8.5.5.5(f.1) step 3: the *smooth* member of the tri-corner family is every
    // exponent that is neither square nor square-in — the circle and the whole curved
    // band alike, both of which the unit-octant grid of (f.2) builds.
    assert!(
        mesh.adj_tri_corner_trigger(&cube, 0.7, false),
        "a curved (bulging) profile fires the tri-corner family (§8.5.5.5(f.2))",
    );
    assert!(
        mesh.adj_tri_corner_trigger(&cube, 0.3, false),
        "a curved (near-linear) profile fires it too",
    );
    assert!(
        mesh.adj_tri_corner_trigger(&cube, 0.99, false),
        "the square exponent now fires the tri-corner family — the three flat unit squares \
         of §8.5.5.5(f.5) (odd `segments`; even is claimed by the parallel-offset ring)",
    );
    assert!(
        !mesh.adj_tri_corner_trigger(&cube, 0.0, false),
        "square-in takes the degenerate weld (§8.5.5.5(f.4)), still declined here",
    );
    // The straight chord is *not* an exception: (f.1) step 3's smooth arm is every `r`
    // bar the two squares, and (f.2)'s own apex table lists `r = 1` (the flat chamfer
    // plane) as a member. It builds the octant grid like any other smooth exponent.
    assert!(
        mesh.adj_tri_corner_trigger(&cube, 0.25, false),
        "the straight chord (`r = 1`) is a smooth member too (§8.5.5.5(f.1) step 3)",
    );
    // §8.5.5.5(f.1)'s three-valued predicate: a custom profile is an unconditional
    // *refusal*, checked before any geometry — so the very corner every smooth exponent
    // above fires on is declined at each of them once the curve is in play.
    for profile in [0.0_f32, 0.25, 0.3, 0.5, 0.7, 0.99] {
        assert!(
            !mesh.adj_tri_corner_trigger(&cube, profile, true),
            "a custom profile is refused outright at profile {profile} \
             (§8.5.5.5(f.1) step 1 / the three-valued predicate)",
        );
    }

    // Unequal offset: one spoke is twice as long, so the lengths differ.
    let unequal = make_cut(
        &mut mesh,
        [
            Vec3::new(2.0, 2.0, 0.0),
            Vec3::new(0.0, 1.0, 1.0),
            Vec3::new(1.0, 0.0, 1.0),
        ],
    );
    assert!(
        !mesh.adj_tri_corner_trigger(&unequal, 0.5, false),
        "unequal offsets decline the §8.5.1 trigger",
    );

    // Non-right-angle corner: equal-length spokes 120° apart (cos −½, not ½).
    let flat = make_cut(
        &mut mesh,
        [
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(-0.5, 0.866_025_4, 0.0),
            Vec3::new(-0.5, -0.866_025_4, 0.0),
        ],
    );
    assert!(
        !mesh.adj_tri_corner_trigger(&flat, 0.5, false),
        "non-right-angle corner declines the §8.5.1 trigger",
    );
}

/// BEVEL.md §7.6.6 / §8.5.1: a **valence-4** apex whose wide (> 180°) reflex sector
/// **absorbs one folded unbeveled leg**. The three beveled spokes sit on the unit rim at
/// azimuth 20° / 90° / 160° (x-mirror-symmetric about the yz-plane), the apex rides the
/// z-axis, and the leg folds down the −y side to `z = −0.6` — so the corner is x-mirror
/// symmetric and clears the §8.5.1 dihedral band (`edgecount = 4`, boundvert ring `3`).
fn absorbed_leg_reflex_junction() -> (LoopMesh, VertexKey, [EdgeKey; 3]) {
    let mut mesh = LoopMesh::new();
    let degrees = std::f32::consts::PI / 180.0;
    let rim = |azimuth: f32| Vec3::new((azimuth * degrees).cos(), (azimuth * degrees).sin(), 0.0);
    let v = mesh.create_vertex(Vec3::new(0.0, 0.0, 0.7));
    let a = mesh.create_vertex(rim(20.0));
    let b = mesh.create_vertex(rim(90.0));
    let c = mesh.create_vertex(rim(160.0));
    let leg = mesh.create_vertex(Vec3::new(0.0, -1.0, -0.6));

    // A closed fan of four faces around `v`, wound so every normal points +z (outward);
    // the leg sits between the two faces flanking the reflex gap (c → leg → a).
    mesh.face_create_verts(&[v, a, b], FaceCreateFlags::NONE);
    mesh.face_create_verts(&[v, b, c], FaceCreateFlags::NONE);
    mesh.face_create_verts(&[v, c, leg], FaceCreateFlags::NONE);
    mesh.face_create_verts(&[v, leg, a], FaceCreateFlags::NONE);

    let va = mesh.query_edge_exists(v, a).expect("va");
    let vb = mesh.query_edge_exists(v, b).expect("vb");
    let vc = mesh.query_edge_exists(v, c).expect("vc");
    (mesh, v, [va, vb, vc])
}

/// BEVEL.md §8.5.1 / §7.6.6: the reflex-agnostic dihedral-band gate. It admits the
/// absorbed-leg reflex junction and the convex cube corner, and — via the `edgecount − 3`
/// near-flat count clause — refuses the single-face valence-3 reflex apex, which stays
/// on the general fill. That refusal is *final*, not a fall-through to the position-only
/// stand-in: the band was measurable and said no.
#[test]
fn dihedral_band_admits_absorbed_leg_reflex_junction() {
    let (mesh, apex, _) = absorbed_leg_reflex_junction();
    let corners = mesh.vertex_corner_ring(apex);
    assert!(
        mesh.junction_dihedral_band_admits(apex, &corners),
        "the absorbed-leg reflex junction clears the §8.5.1 dihedral band",
    );

    let mut cube = LoopMesh::new();
    mesh_add_cube(&mut cube);
    let corner = vert_at(&cube, Vec3::new(1.0, -1.0, -1.0));
    let cube_corners = cube.vertex_corner_ring(corner);
    assert!(
        cube.junction_dihedral_band_admits(corner, &cube_corners),
        "the convex cube corner clears the band (its signed dihedrals sum to 3·π/2 in \
         magnitude)",
    );

    let (reflex_mesh, reflex_apex, _) = single_face_reflex_junction();
    let reflex_corners = reflex_mesh.vertex_corner_ring(reflex_apex);
    assert_eq!(
        reflex_mesh.junction_dihedral_band(reflex_apex, &reflex_corners),
        DihedralBand::Refused,
        "the single-face reflex apex fails the near-flat count clause — a hard refusal, \
         not an unmeasurable band, so no ring shape can rescue it",
    );
    // A perfect octant ring — the very shape the stand-in admits — handed to the refused
    // corner: the refusal still wins, which is the whole point of the three-valued verdict.
    let apex_position = reflex_mesh.vertex_position(reflex_apex);
    let octant_ring = [
        apex_position + Vec3::new(1.0, 1.0, 0.0),
        apex_position + Vec3::new(0.0, 1.0, 1.0),
        apex_position + Vec3::new(1.0, 0.0, 1.0),
    ];
    assert!(
        tri_corner_meet_geometry(apex_position, &octant_ring),
        "the stand-in reads this ring as an octant",
    );
    assert!(
        !tri_corner_shape_admits(
            apex_position,
            &octant_ring,
            reflex_mesh.junction_dihedral_band(reflex_apex, &reflex_corners),
            TRI_CORNER_BAND_SMOOTH,
        ),
        "yet a refused corner is not bought back by its ring's shape",
    );
}

/// BEVEL.md §7.6.6 / §8.5.1 / §8.2.2.1.1: at `segments = 2` the absorbed-leg reflex
/// junction routes to the **cube-corner cap**, not the general grid. Every cap vert is
/// the four-point `unit_to_corner` map's image of a unit-sphere point, so in octant
/// coordinates it rides the unit sphere; the two reflex-flank rim samples share (y, z)
/// — the map's signature, not a shared cross-section plane.
#[test]
fn absorbed_leg_reflex_junction_takes_cube_corner_cap_segments2() {
    let (mut mesh, _apex, [va, vb, vc]) = absorbed_leg_reflex_junction();
    let apex_position = Vec3::new(0.0, 0.0, 0.7);
    let result = mesh.bevel(
        &[va.into(), vb.into(), vc.into()],
        BevelOptions {
            offset: 0.2,
            affect: BevelAffect::Edges,
            segments: 2,
            vmesh_method: BevelVmeshMethod::Adj,
            ..Default::default()
        },
    );
    mesh.validate_topology();

    // §7.6.6: an open test mesh (the dangling folded leg leaves one boundary loop),
    // so the correct open-mesh characteristic is χ = 1, not a dropped-cap χ = 1 hole.
    let (vertices, edges, faces) = counts(&mesh);
    assert_eq!(
        vertices as i64 - edges as i64 + faces as i64,
        1,
        "the closed cube-corner cap keeps the open fixture at χ = 1",
    );

    // The apex cap is the tri-corner pinwheel: one welded centre where three cap quads
    // meet (the same topology the general fill would give — the divergence is position).
    let hubs: Vec<VertexKey> = result
        .verts
        .iter()
        .copied()
        .filter(|&candidate| {
            let incident: Vec<FaceKey> = mesh
                .face_keys()
                .into_iter()
                .filter(|&face| mesh.face_vert_keys(face).contains(&candidate))
                .collect();
            incident.len() == 3
                && incident
                    .iter()
                    .all(|&face| mesh.face_vert_keys(face).len() == 4)
        })
        .collect();
    assert_eq!(
        hubs.len(),
        1,
        "the reflex junction closes on one welded apex"
    );
    let apex = hubs[0];

    let cap_faces: Vec<FaceKey> = mesh
        .face_keys()
        .into_iter()
        .filter(|&face| mesh.face_vert_keys(face).contains(&apex))
        .collect();
    assert_eq!(
        cap_faces.len(),
        3,
        "three cap quads meet at the welded apex"
    );

    let mut cap_verts: Vec<VertexKey> = Vec::new();
    for &face in &cap_faces {
        for vert in mesh.face_vert_keys(face) {
            if !cap_verts.contains(&vert) {
                cap_verts.push(vert);
            }
        }
    }
    assert_eq!(
        cap_verts.len(),
        7,
        "cap = apex + 3 rim samples + 3 boundverts"
    );

    // Rim samples are the apex's neighbours; the boundverts sit diagonal from it.
    let rim: Vec<VertexKey> = cap_verts
        .iter()
        .copied()
        .filter(|&vert| vert != apex && mesh.query_edge_exists(vert, apex).is_some())
        .collect();
    assert_eq!(
        rim.len(),
        3,
        "the apex touches exactly its three rim samples"
    );
    let boundverts: Vec<VertexKey> = cap_verts
        .iter()
        .copied()
        .filter(|&vert| vert != apex && !rim.contains(&vert))
        .collect();
    assert_eq!(boundverts.len(), 3);

    // §8.5.1: the four-point map determined by (B_0, B_1, B_2, v). For an oblique corner
    // its image of the unit sphere is an ellipsoid, so the cube-corner test is
    // membership of the *unit* sphere in octant coordinates, not a world-frame sphere.
    // The membership is invariant to the boundvert labelling (a signed axis permutation).
    let octant = UnitOctant::from_corner(
        [
            mesh.vertex_position(boundverts[0]),
            mesh.vertex_position(boundverts[1]),
            mesh.vertex_position(boundverts[2]),
        ],
        apex_position,
    )
    .expect("the reflex corner's four-point map is non-degenerate");
    for &vert in &cap_verts {
        let octant_norm = octant.to_octant(mesh.vertex_position(vert)).length();
        assert!(
            (octant_norm - 1.0).abs() <= 2.0e-3,
            "cap vert rides the unit-octant sphere (‖octant‖ = {octant_norm}) — the \
             cube-corner map, not the general fill's off-surface fullness centre",
        );
    }

    // §7.6.6: the two reflex-flank rim samples are images of ‖e_i + e_{i+1}‖ under one
    // affine map, so on this x-mirror corner they share (y, z) and differ only in world-x.
    let rim_positions: Vec<Vec3> = rim.iter().map(|&vert| mesh.vertex_position(vert)).collect();
    let mut flank_pair_shares_yz = false;
    for i in 0..3 {
        for j in (i + 1)..3 {
            let same_yz = (rim_positions[i].y - rim_positions[j].y).abs() <= 2.0e-3
                && (rim_positions[i].z - rim_positions[j].z).abs() <= 2.0e-3;
            let distinct_x = (rim_positions[i].x - rim_positions[j].x).abs() > 2.0e-3;
            if same_yz && distinct_x {
                flank_pair_shares_yz = true;
            }
        }
    }
    assert!(
        flank_pair_shares_yz,
        "the two reflex-flank rim samples share (y, z) — the cube-corner map signature",
    );
}

/// The §8.5.5.8 driver's octant coordinates of a world point: `c = (0.75, −0.75, −0.75)`
/// is the corner frame's origin and `offset = 0.25` its unit. Written out from the
/// guide's own axis relabelling rather than through the fill's map, so the check is
/// independent of the code under test.
fn cube_corner_octant(world: Vec3) -> Vec3 {
    let c = Vec3::new(0.75, -0.75, -0.75);
    Vec3::new(
        (c.z - world.z) / 0.25,
        (world.x - c.x) / 0.25,
        (c.y - world.y) / 0.25,
    )
}

/// The inverse: the world point of an octant coordinate on this driver.
fn cube_corner_world(unit: Vec3) -> Vec3 {
    let c = Vec3::new(0.75, -0.75, -0.75);
    Vec3::new(
        c.x + 0.25 * unit.y,
        c.y - 0.25 * unit.z,
        c.z - 0.25 * unit.x,
    )
}

/// BEVEL.md §8.5.5.8, the worked oracle: the curved-profile cube corner at
/// `segments = 3`, `profile = 0.7`. The exponent `r = 3.886716…` sits strictly between
/// the circle and square gates of the §8.6.2 snap, so it is genuinely curved: there is
/// no sphere to fit and no closed form anywhere in the construction. The whole grid is
/// built in the unit octant (§8.5.5.5(f.2)) — coarse net with the `1.4` pre-scaled
/// centre, one doubling round to 4, the resample back to 3, then the clamped `r`-norm
/// snap of every vert — and mapped to the corner once at the end.
///
/// Twelve corner-grid verts at the oracle's own coordinates: three boundverts, six
/// chain interiors (the permutations of `(0, 0.583876, 0.966646)`) and the three verts
/// of the central triangle (the cyclic images of `(0.923438, 0.595228, 0.595228)`),
/// closed by one triangle and six quads — the odd-`segments` parity of §8.5.5.4.
#[test]
fn adj_cube_corner_curved_profile_segments_3_matches_worked_oracle() {
    let (mesh, result) = cube_corner_adj_profile(3, 0.7);
    mesh.validate_topology();
    let verts = curved_corner_grid_verts(&mesh, &result);
    assert_eq!(verts.len(), 12, "twelve corner-grid verts (§8.5.5.8)");

    for boundvert in [
        Vec3::new(0.75, -0.75, -1.0),
        Vec3::new(1.0, -0.75, -0.75),
        Vec3::new(0.75, -1.0, -0.75),
    ] {
        assert_position_present(&verts, boundvert, "boundvert");
    }
    for chain_interior in [
        Vec3::new(0.991662, -0.75, -0.895969),
        Vec3::new(0.895969, -0.75, -0.991662),
        Vec3::new(0.991662, -0.895969, -0.75),
        Vec3::new(0.895969, -0.991662, -0.75),
        Vec3::new(0.75, -0.991662, -0.895969),
        Vec3::new(0.75, -0.895969, -0.991662),
    ] {
        assert_position_present(&verts, chain_interior, "chain interior");
    }
    for central in [
        Vec3::new(0.898807, -0.898807, -0.980859),
        Vec3::new(0.980859, -0.898807, -0.898807),
        Vec3::new(0.898807, -0.980859, -0.898807),
    ] {
        assert_position_present(&verts, central, "central-triangle vert");
    }

    // §8.5.5.8's own self-check: every corner-grid vert lies on the unit superellipsoid
    // in octant coordinates. A grid that skips or mis-orders the snap fails this on the
    // interior even when its rim looks right.
    let exponent = profile_to_exponent(0.7);
    for &position in &verts {
        let unit = cube_corner_octant(position);
        let norm = unit.x.powf(exponent) + unit.y.powf(exponent) + unit.z.powf(exponent);
        assert!(
            (norm - 1.0).abs() < 1.0e-3,
            "corner-grid vert on the unit superellipsoid: {position:?} → {unit:?}",
        );
    }

    let faces = curved_corner_grid_faces(&mesh, &result);
    let triangles = faces
        .iter()
        .filter(|&&f| mesh.face_vert_keys(f).len() == 3)
        .count();
    let quads = faces
        .iter()
        .filter(|&&f| mesh.face_vert_keys(f).len() == 4)
        .count();
    assert_eq!(
        faces.len(),
        7,
        "one central triangle + six quads (§8.5.5.8)"
    );
    assert_eq!(triangles, 1);
    assert_eq!(quads, 6);
}

/// BEVEL.md §8.5.5.5(f.2), the PRESCALE: the coarse centre is seeded at
/// `√⅓·(1,1,1)·1.4` for every exponent in the bulging band (`r > 1.5` — which the
/// circular exponent is inside, not just the curved ones). It is *invisible in the
/// apex* (the snap is a radial projection, and radial projection is scale-invariant)
/// and *visible one ring in*, so the §8.5.5.8 central triangle is the only place it can
/// be seen — and it is what this pins. Dropping the pre-scale moves those three verts
/// to the cyclic images of octant `(0.936247, 0.570578, 0.570578)`, ≈`0.006` world
/// units off the oracle: far outside the position bucket, and not a Tier-2 free choice.
#[test]
fn adj_cube_corner_curved_profile_prescale_places_the_central_triangle() {
    let (mesh, result) = cube_corner_adj_profile(3, 0.7);
    let verts = curved_corner_grid_verts(&mesh, &result);

    for oracle in [
        Vec3::new(0.923438, 0.595228, 0.595228),
        Vec3::new(0.595228, 0.923438, 0.595228),
        Vec3::new(0.595228, 0.595228, 0.923438),
    ] {
        assert_position_present(
            &verts,
            cube_corner_world(oracle),
            "the 1.4-pre-scaled central-triangle vert",
        );
    }
    for unscaled in [
        Vec3::new(0.936247, 0.570578, 0.570578),
        Vec3::new(0.570578, 0.936247, 0.570578),
        Vec3::new(0.570578, 0.570578, 0.936247),
    ] {
        let world = cube_corner_world(unscaled);
        assert_eq!(
            verts
                .iter()
                .filter(|&&p| (p - world).length() < 1.0e-4)
                .count(),
            0,
            "the un-pre-scaled centre is never emitted",
        );
    }
}

/// A vertex bevel of the cube's `(1, −1, −1)` corner through the [V-ADJ] general grid —
/// the §8.5.5.5(c) symmetry driver. Every result vert is cap.
fn cube_corner_vertex_cap(segments: u32, profile: f32) -> Vec<Vec3> {
    let mut mesh = LoopMesh::new();
    mesh_add_cube(&mut mesh);
    let corner = vert_at(&mesh, Vec3::new(1.0, -1.0, -1.0));
    let result = mesh.bevel(
        &[corner.into()],
        BevelOptions {
            offset: 0.25,
            affect: BevelAffect::Vertices,
            segments,
            profile,
            vmesh_method: BevelVmeshMethod::Adj,
            ..Default::default()
        },
    );
    result
        .verts
        .iter()
        .map(|&v| mesh.vertex_position(v))
        .collect()
}

/// How far the vert set falls short of being closed under `map`: the worst distance from
/// an image back to the nearest member of the set.
fn symmetry_defect(verts: &[Vec3], map: impl Fn(Vec3) -> Vec3) -> f32 {
    verts
        .iter()
        .map(|&p| {
            let image = map(p);
            verts
                .iter()
                .map(|&other| (other - image).length())
                .fold(f32::INFINITY, f32::min)
        })
        .fold(0.0_f32, f32::max)
}

/// The 3-fold rotation cycling the corner's three axes `+x → −y → −z → +x`, and the
/// reflection swapping two of them. Both are exact symmetries of the *input* cube.
fn corner_rotation(p: Vec3) -> Vec3 {
    Vec3::new(-p.z, -p.x, p.y)
}

fn corner_mirror(p: Vec3) -> Vec3 {
    Vec3::new(p.x, p.z, p.y)
}

/// BEVEL.md §8.5.5.5(c) (#half-chain-mask): the `(−1/6, +4/3, −1/6)` sharpening tap
/// prepares only chain indices `1 ..= N_in/2`. The rest of the chain is the *next*
/// sector's `k = 0` column storage (`(i, 0, k) ≡ (i+1, N_in − k, 0)`), which the mask
/// pass never writes. The consequence is a grid property, not a curve one, and it is
/// the fastest observable there is: the cap of a perfectly symmetric corner comes out
/// rotationally symmetric but **not** mirror-symmetric.
///
/// The distinction is inert at `N_in = 2` — the coarse chain's lone interior sample sits
/// exactly at `k = N_in/2` — so the first round is symmetric and the asymmetry first
/// bites in the second (`N_in = 4`, where `k = 3` escapes): invisible at `segments ≤ 4`,
/// live from `segments = 5`.
#[test]
fn adj_cap_half_chain_mask_leaves_the_cap_rotational_but_not_mirror_symmetric() {
    // One round (N_in = 2): the mask covers the whole chain either way.
    for profile in [0.5_f32, 0.7] {
        let verts = cube_corner_vertex_cap(4, profile);
        assert!(symmetry_defect(&verts, corner_rotation) < 1.0e-5);
        assert!(
            symmetry_defect(&verts, corner_mirror) < 1.0e-5,
            "segments = 4 is inside the mask's inert range",
        );
    }

    // Two rounds (segments = 8): `k = 3` of the second round's input chain escapes.
    let verts = cube_corner_vertex_cap(8, 0.5);
    assert!(
        symmetry_defect(&verts, corner_rotation) < 1.0e-5,
        "the cap stays rotationally symmetric",
    );
    assert!(
        symmetry_defect(&verts, corner_mirror) > 1.0e-3,
        "the cap is NOT mirror-symmetric; masking the whole chain would make it so",
    );

    // The guide's witness vertex is present and its mirror image is absent.
    let witness = Vec3::new(0.97929, -0.98941, -0.87331);
    let nearest = |target: Vec3| {
        verts
            .iter()
            .map(|&p| (p - target).length())
            .fold(f32::INFINITY, f32::min)
    };
    assert!(nearest(witness) < 1.0e-4, "witness present");
    assert!(
        nearest(corner_mirror(witness)) > 1.0e-3,
        "its mirror absent",
    );
}

/// BEVEL.md §8.5.5.5(c): at the three **polyline** exponents the half-vs-whole masking
/// choice is inert — at the line every second difference vanishes so the tap itself is
/// zero, and at square / square-in the one bending sample sits at `k = N_in/2` and so is
/// inside the masked half at every (always even) cascade resolution. The cap is
/// therefore fully mirror-symmetric there even at `segments = 8`.
///
/// This is not licence to skip the tap at those exponents: the square chains' corner
/// sample *is* moved by it.
#[test]
fn adj_cap_half_chain_mask_is_inert_at_the_polyline_exponents() {
    for profile in [0.0_f32, 0.25, 1.0] {
        let verts = cube_corner_vertex_cap(8, profile);
        assert!(symmetry_defect(&verts, corner_rotation) < 1.0e-5);
        assert!(
            symmetry_defect(&verts, corner_mirror) < 1.0e-5,
            "polyline exponent {profile} keeps the cap mirror-symmetric",
        );
    }
}

/// BEVEL.md §8.5.5.8's second companion driver: `profile = 0.3` (`r = 1.151433`) at
/// `segments = 3`. The exponent lands in the **middle** band of the PRESCALE rule
/// (`0.75 ≤ r ≤ 1.5`), so the coarse centre carries **no** pre-scale — the bracket that
/// proves the `1.4` is banded on `r` rather than applied unconditionally. Twelve verts
/// again: the chain interiors are the permutations of `(0, 0.375476, 0.711987)` and the
/// central triangle the cyclic images of `(0.517127, 0.316617, 0.316617)`.
#[test]
fn adj_cube_corner_curved_profile_segments_3_middle_band_no_prescale() {
    let (mesh, result) = cube_corner_adj_profile(3, 0.3);
    mesh.validate_topology();
    let verts = curved_corner_grid_verts(&mesh, &result);
    assert_eq!(verts.len(), 12);

    assert_eq!(adj_octant_prescale(profile_to_exponent(0.3), 3), 1.0);
    for chain_interior in [
        Vec3::new(0.0, 0.375476, 0.711987),
        Vec3::new(0.0, 0.711987, 0.375476),
        Vec3::new(0.375476, 0.0, 0.711987),
        Vec3::new(0.711987, 0.0, 0.375476),
        Vec3::new(0.375476, 0.711987, 0.0),
        Vec3::new(0.711987, 0.375476, 0.0),
    ] {
        assert_position_present(&verts, cube_corner_world(chain_interior), "chain interior");
    }
    for central in [
        Vec3::new(0.517127, 0.316617, 0.316617),
        Vec3::new(0.316617, 0.517127, 0.316617),
        Vec3::new(0.316617, 0.316617, 0.517127),
    ] {
        assert_position_present(&verts, cube_corner_world(central), "central-triangle vert");
    }

    let faces = curved_corner_grid_faces(&mesh, &result);
    assert_eq!(faces.len(), 7, "one central triangle + six quads");
}

/// BEVEL.md §8.5.5.5(f.2), the closed-form apex: on **even** `segments` the cube corner
/// welds one apex vertex, and it is a pure function of the exponent —
/// `3^(−1/r)·(1,1,1)` in octant coordinates, for every `r`. At `profile = 0.7` that is
/// `0.753778·(1,1,1)`, world `(0.938444, −0.938444, −0.938444)`, and it is the *same*
/// point at `segments = 2` (where the coarse net is the target grid, no subdivision
/// runs, and the pre-scale is guarded off) and at `segments = 4` (where a doubling
/// round and the `1.4` pre-scale both run, and the radial snap makes them invisible in
/// the apex — which is exactly why the apex must not be used to "verify" the
/// pre-scale).
#[test]
fn adj_cube_corner_curved_profile_even_apex_is_the_exponent_closed_form() {
    let exponent = profile_to_exponent(0.7);
    let apex = cube_corner_world(Vec3::splat(3.0_f32.powf(-1.0 / exponent)));
    assert!((apex - Vec3::new(0.938444, -0.938444, -0.938444)).length() < 1.0e-5);
    assert_eq!(adj_octant_prescale(exponent, 2), 1.0, "the seg ≤ 2 guard");
    assert_eq!(adj_octant_prescale(exponent, 4), 1.4, "the bulging band");

    // `segments = 2`: the three-quad pinwheel of §8.5.3, its apex the octant closed
    // form rather than a circumsphere fit (a curved corner has no sphere at all).
    let (mesh, result) = cube_corner_adj_profile(2, 0.7);
    mesh.validate_topology();
    let verts = curved_corner_grid_verts(&mesh, &result);
    assert_eq!(verts.len(), 7, "3 boundverts + 3 chain mids + the apex");
    assert_position_present(&verts, apex, "the seg = 2 welded apex");
    assert_eq!(
        curved_corner_grid_faces(&mesh, &result).len(),
        3,
        "the three-quad pinwheel",
    );

    // `segments = 4`: the same apex, now the welded centre of the subdivided grid, with
    // 3·(segments/2)² = 12 quads and no central face.
    let (mesh, result) = cube_corner_adj_profile(4, 0.7);
    mesh.validate_topology();
    let verts = curved_corner_grid_verts(&mesh, &result);
    assert_position_present(&verts, apex, "the seg = 4 welded apex");
    let faces = curved_corner_grid_faces(&mesh, &result);
    assert_eq!(faces.len(), 12, "3·(segments/2)² sub-patch quads");
    for &face in &faces {
        assert_eq!(
            mesh.face_vert_keys(face).len(),
            4,
            "no central face at even parity"
        );
    }
}

/// BEVEL.md §8.5.5.5(f.2)'s snap, and why it is three branches rather than one formula.
/// The circular exponent normalises **unclamped**, so a vert the smoothing pushed a hair
/// past a sector seam keeps its sign and stays a hair outside; a curved exponent must
/// clamp into the octant first, because a negative base raised to a fractional power is
/// `NaN`. Clamping the circular branch too would move those verts, and a curved branch
/// without the clamp produces nothing at all.
#[test]
fn adj_octant_snap_branches_on_the_exponent() {
    let overshoot = Vec3::new(0.9, 0.5, -0.02);

    let circular = snap_to_unit_superellipsoid(overshoot, 2.0, false);
    assert!((circular.length() - 1.0).abs() < 1.0e-6, "unit-norm");
    assert!(
        circular.z < 0.0,
        "the circular branch keeps the negative component"
    );

    let curved = snap_to_unit_superellipsoid(overshoot, profile_to_exponent(0.7), false);
    assert!(
        curved.is_finite(),
        "the curved branch clamps rather than NaN-ing"
    );
    assert_eq!(curved.z, 0.0, "and folds the overshoot onto the seam plane");
    let exponent = profile_to_exponent(0.7);
    let norm = curved.x.powf(exponent) + curved.y.powf(exponent) + curved.z.powf(exponent);
    assert!((norm - 1.0).abs() < 1.0e-5, "on the unit superellipsoid");

    // A point already on the surface is a fixed point of either branch.
    let on_surface = Vec3::splat(3.0_f32.powf(-1.0 / exponent));
    assert!(
        (snap_to_unit_superellipsoid(on_surface, exponent, false) - on_surface).length() < 1.0e-6
    );
}

/// BEVEL.md §8.5.5.5(f.2) **branch S** — the sentinels' axis snap, which is not a norm and
/// is not a limit of branch R.
///
/// The rule is stated in the *clamped unit square*: clamp both in-plane coordinates into
/// `[0, 1]`, then push onto the **nearer** of the two walls the sentinel names — `u == 1` /
/// `v == 1` at `square` (the profile bulges out to the cell's `(1, 1)` corner), `u == 0` /
/// `v == 0` at `square-in` (it bows away, onto the cell edges through the `(0, 0)` corner).
/// `land_on_corner` overrides that choice and lands on **both** walls at once; without it a
/// pipe's square cross-section folds into a wedge.
///
/// Two things the branch must *not* do, both checked here: it must not reach for the
/// `r`-norm (at `r = 0` that diverges and collapses the whole cross-section to a point, and
/// at `r = 1e4` every `a^r` overflows or flushes in `f32`), and it must leave the other two
/// branches' `land_on_corner` handling alone — the flag exists for this branch only.
#[test]
fn superellipsoid_snap_sentinel_branch_pushes_onto_the_nearer_wall() {
    let square = PROFILE_SQUARE_EXPONENT;
    let square_in = 0.0_f32;
    let snap = |u: f32, v: f32, exponent: f32, corner: bool| {
        snap_to_unit_superellipsoid(Vec3::new(u, v, 0.0), exponent, corner)
    };

    // `square`: the nearer of `u == 1` / `v == 1`.
    assert_eq!(snap(0.8, 0.3, square, false), Vec3::new(1.0, 0.3, 0.0));
    assert_eq!(snap(0.3, 0.8, square, false), Vec3::new(0.3, 1.0, 0.0));
    // `square-in`: the nearer of `u == 0` / `v == 0` — the other pair, not the same one.
    assert_eq!(snap(0.3, 0.8, square_in, false), Vec3::new(0.0, 0.8, 0.0));
    assert_eq!(snap(0.8, 0.3, square_in, false), Vec3::new(0.8, 0.0, 0.0));

    // `land_on_corner` takes both walls, at either sentinel.
    assert_eq!(snap(0.8, 0.3, square, true), Vec3::new(1.0, 1.0, 0.0));
    assert_eq!(snap(0.3, 0.8, square_in, true), Vec3::ZERO);

    // The clamp runs first, so a coordinate outside the cell is folded onto the square's
    // own edge before the walls are compared — never raised to a power.
    assert_eq!(snap(-0.3, 0.4, square, false), Vec3::new(0.0, 1.0, 0.0));
    assert_eq!(snap(-0.3, 0.4, square_in, false), Vec3::new(0.0, 0.4, 0.0));
    assert_eq!(snap(0.5, 1.6, square, false), Vec3::new(0.5, 1.0, 0.0));
    assert_eq!(snap(0.5, 1.6, square_in, false), Vec3::new(0.0, 1.0, 0.0));

    // A point already on a wall is a fixed point, so the snap is safe to apply to a rim
    // sample the chain sampler already put there.
    for (u, v, exponent) in [(1.0, 0.4, square), (0.0, 0.4, square_in)] {
        assert_eq!(snap(u, v, exponent, false), Vec3::new(u, v, 0.0));
    }

    // The other two branches ignore the flag entirely.
    let off_surface = Vec3::new(0.9, 0.5, 0.2);
    for exponent in [2.0_f32, profile_to_exponent(0.7), 1.0] {
        assert_eq!(
            snap_to_unit_superellipsoid(off_surface, exponent, true),
            snap_to_unit_superellipsoid(off_surface, exponent, false),
            "`land_on_corner` is branch S's alone (r = {exponent})",
        );
    }
}

/// BEVEL.md §8.5.5.5(f.1) step 3 + (f.2): the **flat chamfer** (`profile = 0.25`,
/// `r = 1`) is a member of the tri-corner family's smooth arm, not an exception to it —
/// the arm is "every `r` that is neither `EXP_SQUARE` nor `EXP_SQUARE_IN`", and (f.2)'s
/// closed-form apex table lists `r = 1.000000 -> 0.333333·(1,1,1)` explicitly. So a flat
/// cube corner **builds the octant grid**: branch R of the snap degenerates to a 1-norm
/// radial projection, whose unit surface is the plane `x + y + z = 1` — the chamfer
/// plane through the three boundverts — and whose apex is the boundvert centroid.
///
/// The pin that matters is the *interior*: a planar bilinear cap would also land the
/// apex on the centroid (which is why the apex alone proves nothing here), but its ring
/// of interior verts is a different set of points on the same plane. The octant grid
/// runs the coarse net (pre-scale `1.0`, the near-linear band), one doubling round and
/// the resample, then projects — so every cap vert lies on the plane, and the cap is the
/// odd-parity twelve-vert / seven-face grid of §8.5.5.4, not a fan.
#[test]
fn adj_cube_corner_flat_chamfer_builds_the_octant_grid() {
    assert_eq!(
        profile_to_exponent(0.25),
        1.0,
        "the §8.6.2 snap makes `r == 1` exact"
    );
    assert_eq!(
        adj_octant_prescale(1.0, 3),
        1.0,
        "`r = 1` sits in the near-linear PRESCALE band (0.75 ..= 1.5)",
    );

    let (mesh, result) = cube_corner_adj_profile(3, 0.25);
    mesh.validate_topology();
    let verts = curved_corner_grid_verts(&mesh, &result);
    assert_eq!(
        verts.len(),
        12,
        "3 boundverts + 6 chain interiors + 3 central-triangle verts"
    );

    // The whole cap — rim and interior alike — lands on the unit 1-norm surface, i.e.
    // the flat chamfer plane. This is what routing `r = 1` through branch R *means*.
    for &world in &verts {
        let unit = cube_corner_octant(world);
        assert!(
            (unit.x + unit.y + unit.z - 1.0).abs() < 1.0e-4,
            "cap vert {unit:?} is off the chamfer plane x + y + z = 1",
        );
    }

    let faces = curved_corner_grid_faces(&mesh, &result);
    assert_eq!(
        faces.len(),
        7,
        "one central triangle + six quads (§8.5.5.4 odd parity)"
    );

    // Even parity: the welded apex is (f.2)'s `3^(−1/1)·(1,1,1)` — the boundvert
    // centroid — and the cap is the twelve-quad grid with no central face.
    let (mesh, result) = cube_corner_adj_profile(4, 0.25);
    mesh.validate_topology();
    let verts = curved_corner_grid_verts(&mesh, &result);
    assert_position_present(
        &verts,
        cube_corner_world(Vec3::splat(1.0 / 3.0)),
        "the flat chamfer's welded apex",
    );
    let faces = curved_corner_grid_faces(&mesh, &result);
    assert_eq!(faces.len(), 12, "3·(segments/2)² sub-patch quads");
    for &face in &faces {
        assert_eq!(
            mesh.face_vert_keys(face).len(),
            4,
            "no central face at even parity"
        );
    }
}

/// BEVEL.md §8.5.5.5(f.2)'s PRESCALE, third band: a **curved-in** exponent (`r < 0.75`)
/// seeds the coarse centre *hollow*, at `0.6·√⅓·(1,1,1)`, rather than bulged. The band
/// is keyed on `r`, not on `profile`, so `profile = 0.1` (`r ≈ 0.602`) is the bracket
/// below the near-linear band that the `0.3` / `0.7` drivers bracket from above. The
/// seed is invisible in the apex — the snap is radial — so the observable is that the
/// grid still closes and still lands every vert on the unit superellipsoid.
#[test]
fn adj_octant_hollow_band_prescale_and_membership() {
    let exponent = profile_to_exponent(0.1);
    assert!(
        exponent < 0.75 && exponent > 0.0,
        "`profile = 0.1` is genuinely curved-in"
    );
    assert_eq!(adj_octant_prescale(exponent, 3), 0.6, "the hollow band");
    assert_eq!(
        adj_octant_prescale(exponent, 2),
        1.0,
        "still guarded off at seg ≤ 2"
    );

    let (mesh, result) = cube_corner_adj_profile(3, 0.1);
    mesh.validate_topology();
    let verts = curved_corner_grid_verts(&mesh, &result);
    assert_eq!(verts.len(), 12);
    for &world in &verts {
        let unit = cube_corner_octant(world);
        let norm = unit.x.powf(exponent) + unit.y.powf(exponent) + unit.z.powf(exponent);
        assert!(
            (norm - 1.0).abs() < 1.0e-3,
            "cap vert {unit:?} is off the unit superellipsoid (r-norm {norm})",
        );
    }
    assert_eq!(curved_corner_grid_faces(&mesh, &result).len(), 7);
}

/// The cap's genuinely **interior** verts: those on no ring chain, i.e. those whose every
/// incident face is a cap face. The rim — the boundverts and their [S] chain samples — is
/// shared with the arriving strips and the rebuilt originals, and it moves with `profile`
/// through the §8.6.1 cross-section for reasons that have nothing to do with `fullness`.
/// Only these verts are placed by the (f.3) centre seed and the relaxation it drives.
fn pyramid_apex_cap_interior(mesh: &LoopMesh, result: &BevelResult) -> Vec<VertexKey> {
    let (faces, verts) = pyramid_apex_cap(mesh, result);
    let cap: std::collections::HashSet<FaceKey> = faces.into_iter().collect();
    verts
        .into_iter()
        .filter(|&vertex| {
            mesh.vert_face_keys(vertex)
                .iter()
                .all(|face| cap.contains(face))
        })
        .collect()
}

/// BEVEL.md §8.5.5.5(f.1) **step 4**: the general (non-tri-corner) ring's fullness grid
/// carries **no exponent gate**. Every smooth exponent — the circle, the straight chord,
/// and the whole genuinely curved band on either side of them — builds the same
/// (a)–(e) grid; the exponent enters only through the [S]-sampled boundary chains and
/// through the (f.3) `fullness` centre seed, never through a snap or a closed form. So
/// the valence-4 pyramid apex fills its cap at a *curved* `r` exactly as it does at the
/// circular default, with the same topology.
///
/// The counts are §8.5.5.5(e)'s parity law read over `s` sectors, with
/// `half = segments / 2`:
///
/// ```text
///    even segments (welded apex)      faces  s·half²
///                                     verts  s·(half+1)² − s·(half+1) + 1
///    odd segments (central s-gon)     faces  s·half² + s·half + 1
///                                     verts  s·(half+1)²
/// ```
///
/// The even law's `−s·(half+1) + 1` is the `s` shared spokes running from each ring
/// chain's midpoint to the one welded centre; the odd cap shares nothing, its sectors
/// being bridged by seam strips and closed by a single central `s`-gon (a quad at
/// `s = 4`, which is why the whole cap is quads at either parity here).
#[test]
fn adj_general_ring_curved_exponent_fills_the_step4_fullness_grid() {
    for profile in [0.1_f32, 0.7] {
        let exponent = profile_to_exponent(profile);
        assert!(
            exponent != 1.0
                && exponent != 2.0
                && exponent != 0.0
                && exponent != PROFILE_SQUARE_EXPONENT,
            "profile {profile} is a genuinely curved exponent (r = {exponent})",
        );
    }

    for segments in [2u32, 3, 4, 5] {
        let sectors = pyramid_apex_adj(segments, 0.5).2;
        let half = segments as usize / 2;
        let (expected_faces, expected_verts) = if segments % 2 == 0 {
            (
                sectors * half * half,
                sectors * (half + 1) * (half + 1) - sectors * (half + 1) + 1,
            )
        } else {
            (
                sectors * half * half + sectors * half + 1,
                sectors * (half + 1) * (half + 1),
            )
        };

        // The concave (`r < 1`) and convex (`r > 2`) bands, and the circular default as
        // the control: all three build the same grid.
        for profile in [0.1_f32, 0.5, 0.7] {
            let (mesh, result, _) = pyramid_apex_adj(segments, profile);
            mesh.validate_topology();
            let (faces, verts) = pyramid_apex_cap(&mesh, &result);
            assert_eq!(
                faces.len(),
                expected_faces,
                "cap face count at segments = {segments}, profile = {profile}",
            );
            assert_eq!(
                verts.len(),
                expected_verts,
                "cap vert count at segments = {segments}, profile = {profile}",
            );
            for &face in &faces {
                assert_eq!(
                    mesh.face_vert_keys(face).len(),
                    4,
                    "every valence-4 cap face is a quad — the odd parity's central \
                     `s`-gon included (segments = {segments}, profile = {profile})",
                );
            }
        }
    }
}

/// BEVEL.md §8.5.5.5(b) + (f.3): on the general ring the *only* thing the exponent does
/// to the cap's **interior** is set `fullness`, and `fullness` is a signed blend of the
/// boundvert centroid toward the original corner `v`. Its **sign** is observable as the
/// side on which the interior leaves the cap's own **rim** — the boundverts and the [S]
/// chain samples between them:
///
/// - `profile = 0.25` (`r = 1`, law **F2**'s exact `0`) → rim and interior alike lie in
///   the boundvert ring plane, which is the planarity witness the other arms lean on;
/// - `profile < 0.25` (`r < 1`, law **F3** turning negative) → every interior vert drops
///   strictly **below the lowest rim vert**;
/// - `profile > 0.25` (`r > 1`, positive fullness) → every interior vert rises strictly
///   **above the highest rim vert**, and the top keeps rising with `profile`.
///
/// Measured against the rim, not against the flat ring plane, and on the interior verts
/// only. The rim itself moves with `profile` through the §8.6.1 cross-section for reasons
/// that have nothing to do with `fullness` — a concave rim sags below the boundvert plane
/// on its own — so "the cap left the ring plane" is no evidence at all that `fullness` was
/// consulted. Overshooting the *whole rim* is: with the seed pinned at `0` the relaxation
/// settles the interior **between** the rim's extremes. That is what makes the concave arm
/// here the one that catches a `[0, 1]` clamp of law F3 — the exact failure (f.3) warns
/// about, since it states outright that `fullness` is not confined to that interval.
#[test]
fn adj_general_ring_curved_fullness_sign_drives_the_cap_interior_past_its_rim() {
    // The cap's rim and interior `z`, split apart.
    let sample = |segments: u32, profile: f32| -> (Vec<f32>, Vec<f32>) {
        let (mesh, result, _) = pyramid_apex_adj(segments, profile);
        mesh.validate_topology();
        let (_, cap) = pyramid_apex_cap(&mesh, &result);
        let interior: std::collections::HashSet<VertexKey> =
            pyramid_apex_cap_interior(&mesh, &result)
                .into_iter()
                .collect();
        assert!(!interior.is_empty(), "the cap has an interior to measure");
        let mut rim_z = Vec::new();
        let mut interior_z = Vec::new();
        for vertex in cap {
            let z = mesh.vertex_position(vertex).z;
            if interior.contains(&vertex) {
                interior_z.push(z);
            } else {
                rim_z.push(z);
            }
        }
        assert!(!rim_z.is_empty(), "the cap has a rim to measure against");
        (rim_z, interior_z)
    };
    let low = |zs: &[f32]| zs.iter().copied().fold(f32::MAX, f32::min);
    let high = |zs: &[f32]| zs.iter().copied().fold(f32::MIN, f32::max);

    for segments in [3u32, 4] {
        assert_eq!(adj_fullness(0.25, segments), 0.0);
        let (flat_rim, flat_interior) = sample(segments, 0.25);
        let flat: Vec<f32> = flat_rim
            .iter()
            .chain(flat_interior.iter())
            .copied()
            .collect();
        assert!(
            high(&flat) - low(&flat) < 1.0e-5,
            "the `r = 1` cap is planar, rim and interior alike (segments = {segments})",
        );

        assert!(
            adj_fullness(0.1, segments) < 0.0,
            "F3 turns negative below the chamfer (segments = {segments})",
        );
        let (rim, interior) = sample(segments, 0.1);
        assert!(
            high(&interior) < low(&rim) - 1.0e-3,
            "the concave interior sinks past the whole rim, not merely with it \
             (interior ≤ {}, rim ≥ {}, segments = {segments})",
            high(&interior),
            low(&rim),
        );

        assert!(
            adj_fullness(0.7, segments) > 1.0,
            "F3 runs past 1 on a bulging profile (segments = {segments})",
        );
        let (rim, interior) = sample(segments, 0.7);
        assert!(
            low(&interior) > high(&rim) + 1.0e-3,
            "the convex interior bulges past the whole rim toward `v` \
             (interior ≥ {}, rim ≤ {}, segments = {segments})",
            low(&interior),
            high(&rim),
        );

        // …and the bulge grows with the profile right across the curved band, the
        // circular default sitting inside it rather than bounding it.
        let tops: Vec<f32> = [0.3_f32, 0.5, 0.7]
            .into_iter()
            .map(|profile| high(&sample(segments, profile).1))
            .collect();
        assert!(
            tops.windows(2).all(|pair| pair[0] < pair[1]),
            "interior height rises with `profile` (segments = {segments}, tops {tops:?})",
        );
    }
}

/// BEVEL.md §8.5.5.5(f.1) step 4 gates on **no** exponent, and the **square** sentinel is
/// not an exception at a general ring. Step 1 claims the square ring only at *even*
/// `segments` (the (f.5) parallel-offset lattice) and step 3 claims the valence-3 square
/// (the three flat unit squares), so **odd** `segments` on a general ring reaches step 4 —
/// and (f.5) says so from its own side: its lattice "is not reached on odd `segments` at
/// all", which leaves the ordinary (a)–(e) grid as the only construction that owns it.
///
/// Both parities therefore land §8.5.5.5(e)'s counts, and the even arm is the control that
/// attributes the odd arm to step 4 rather than to the exponent: the two routes are
/// different constructions (parallel offsets vs fullness relaxation) that agree on
/// topology and differ only in placement.
#[test]
fn adj_general_ring_square_exponent_odd_segments_fills_the_step4_grid() {
    for profile in [0.95_f32, 0.97] {
        assert_eq!(
            profile_to_exponent(profile),
            PROFILE_SQUARE_EXPONENT,
            "profile {profile} snaps to the square sentinel (§8.6.2)",
        );
        // Odd: the new step-4 arm. Even: step 1's parallel-offset ring, unchanged.
        for segments in [2u32, 3, 4, 5, 7] {
            let (mesh, result, sectors) = pyramid_apex_adj(segments, profile);
            mesh.validate_topology();
            let (faces, verts) = pyramid_apex_cap(&mesh, &result);
            assert_eq!(
                (faces.len(), verts.len()),
                adj_general_ring_parity_counts(sectors, segments),
                "cap counts at segments = {segments}, profile = {profile}",
            );
            for &face in &faces {
                assert_eq!(
                    mesh.face_vert_keys(face).len(),
                    4,
                    "every valence-4 cap face is a quad, the odd centre's `s`-gon included \
                     (segments = {segments}, profile = {profile})",
                );
            }
        }
    }
}

/// BEVEL.md §8.5.5.5(f.4b): **square-in** on a ring the (f.4a) weld does not own is the
/// ordinary general grid — same topology as every other exponent's — with the concave
/// profile entering through two knobs only: the boundary ring is sampled as the two
/// straight legs (§8.6.4), and the centre is seeded by law **F3** at `profile = 0`, which
/// is **negative** and so places the coarse centre on the far side of the boundvert
/// centroid from `v`.
///
/// The sign is what makes the cap a notch, and it is observable the same way the concave
/// band's is: the interior leaves the cap's whole **rim** behind, rather than settling
/// between its extremes as a zeroed (or clamped) seed would.
#[test]
fn adj_general_ring_square_in_fills_the_f4b_grid() {
    assert_eq!(
        profile_to_exponent(0.0),
        0.0,
        "square-in is the `r = 0` sentinel"
    );
    for segments in [2u32, 3, 4, 5] {
        assert!(
            adj_fullness(0.0, segments) < 0.0,
            "law F3 at `profile = 0` is negative (segments = {segments})",
        );
        let (mesh, result, sectors) = pyramid_apex_adj(segments, 0.0);
        mesh.validate_topology();
        let (faces, verts) = pyramid_apex_cap(&mesh, &result);
        assert_eq!(
            (faces.len(), verts.len()),
            adj_general_ring_parity_counts(sectors, segments),
            "cap counts at segments = {segments}",
        );

        let interior: std::collections::HashSet<VertexKey> =
            pyramid_apex_cap_interior(&mesh, &result)
                .into_iter()
                .collect();
        assert!(!interior.is_empty(), "the cap has an interior to measure");
        let (mut rim_low, mut interior_high) = (f32::MAX, f32::MIN);
        for vertex in verts {
            let z = mesh.vertex_position(vertex).z;
            if interior.contains(&vertex) {
                interior_high = interior_high.max(z);
            } else {
                rim_low = rim_low.min(z);
            }
        }
        assert!(
            interior_high < rim_low - 1.0e-3,
            "the negative fullness notches the whole interior below the rim \
             (interior ≤ {interior_high}, rim ≥ {rim_low}, segments = {segments})",
        );
    }
}

/// The §8.5.5.5(f.4a) weld's leg frame has to be **total**. It is resolved at layout time,
/// where the chains are welded and the legs are mapped through it, so a decline there is
/// unrecoverable: the `s` legs would meet nowhere and the corner would be left open in a
/// closed mesh.
///
/// `unit_octant_origin` is the frame's translation alone — an affine combination of the
/// four points — so it agrees with the conditioned frame's origin wherever that frame
/// exists and still answers where `UnitOctant::from_corner` refuses. That refusal is about
/// invertibility, a question the weld never asks: it only maps *forward*.
#[test]
fn unit_octant_origin_is_total_where_the_frame_is_not() {
    let vertex = Vec3::new(1.0, -1.0, -1.0);
    let ring = square_in_weld_boundverts();
    let frame = UnitOctant::from_corner(ring, vertex).expect("the driver's corner is sound");
    assert!(
        (frame.translation - unit_octant_origin(ring, vertex)).length() <= 1.0e-6,
        "the translation is the frame's origin",
    );
    assert!(
        (unit_octant_origin(ring, vertex) - square_in_weld_origin()).length() <= 1.0e-6,
        "and it is the driver's notch bottom `c`",
    );

    // Two coincident boundverts make two of the frame's three columns equal, so the
    // normalised determinant is zero and the frame is refused.
    let collapsed = [ring[0], ring[0], ring[2]];
    assert!(
        UnitOctant::from_corner(collapsed, vertex).is_none(),
        "the singular frame declines",
    );
    assert!(
        unit_octant_origin(collapsed, vertex).is_finite(),
        "the anchor does not",
    );
}

/// The §8.5.5.8 driver's corner carrying one extra **in-plane passenger** edge: the same
/// cube, the same three beveled spokes, but the `y = −1` face split along its `v → v_5`
/// diagonal so the junction's edge count is four while its ring is still three sectors
/// (the passenger is absorbed by the sector it runs through).
///
/// The split is exactly coplanar, so the passenger's dihedral is zero and the §8.5.1 band
/// counts it as the one near-flat edge `edgecount − 3` allows at valence 4 — the corner is
/// *not* refused. Neither is its shape: the three boundverts and `v` are where they were,
/// so the position-only meet geometry fires too. What the corner is, in §8.5.5.5(f.1)'s
/// three-valued reading, is **doesn't apply**.
fn cube_corner_in_plane_passenger_adj(segments: u32, profile: f32) -> (LoopMesh, BevelResult) {
    let mut mesh = LoopMesh::new();
    let corners = [
        Vec3::new(1.0, -1.0, -1.0),
        Vec3::new(-1.0, -1.0, -1.0),
        Vec3::new(-1.0, 1.0, -1.0),
        Vec3::new(1.0, 1.0, -1.0),
        Vec3::new(1.0, -1.0, 1.0),
        Vec3::new(-1.0, -1.0, 1.0),
        Vec3::new(-1.0, 1.0, 1.0),
        Vec3::new(1.0, 1.0, 1.0),
    ];
    let vert: Vec<VertexKey> = corners.iter().map(|&p| mesh.create_vertex(p)).collect();
    let mut edge_of = |mesh: &mut LoopMesh, a: usize, b: usize| {
        mesh.create_edge([vert[a], vert[b]], EdgeCreateFlags::NONE)
    };
    let e01 = edge_of(&mut mesh, 0, 1);
    let e12 = edge_of(&mut mesh, 1, 2);
    let e23 = edge_of(&mut mesh, 2, 3);
    let e30 = edge_of(&mut mesh, 3, 0);
    let e45 = edge_of(&mut mesh, 4, 5);
    let e56 = edge_of(&mut mesh, 5, 6);
    let e67 = edge_of(&mut mesh, 6, 7);
    let e74 = edge_of(&mut mesh, 7, 4);
    let e04 = edge_of(&mut mesh, 0, 4);
    let e15 = edge_of(&mut mesh, 1, 5);
    let e26 = edge_of(&mut mesh, 2, 6);
    let e37 = edge_of(&mut mesh, 3, 7);
    let passenger = edge_of(&mut mesh, 0, 5);

    let mut face_of = |mesh: &mut LoopMesh, ring: &[usize], edges: &[EdgeKey]| {
        let ring: Vec<VertexKey> = ring.iter().map(|&i| vert[i]).collect();
        mesh.face_create(&ring, edges, FaceCreateFlags::NONE);
    };
    face_of(&mut mesh, &[0, 1, 2, 3], &[e01, e12, e23, e30]);
    face_of(&mut mesh, &[1, 5, 6, 2], &[e15, e56, e26, e12]);
    face_of(&mut mesh, &[5, 4, 7, 6], &[e45, e74, e67, e56]);
    face_of(&mut mesh, &[4, 0, 3, 7], &[e04, e30, e37, e74]);
    face_of(&mut mesh, &[3, 2, 6, 7], &[e23, e26, e67, e37]);
    // The `y = −1` face, split along the diagonal out of the beveled corner.
    face_of(&mut mesh, &[0, 4, 5], &[e04, e45, passenger]);
    face_of(&mut mesh, &[0, 5, 1], &[passenger, e15, e01]);

    let result = mesh.bevel(
        &[e01.into(), e30.into(), e04.into()],
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

/// The driver's cube under a shear along `x` by `y`, beveled at the same corner: a
/// valence-3 junction whose three spokes are no longer mutually orthonormal.
///
/// The shear is linear, so every face stays a planar parallelogram and the solid stays
/// closed — but the corner's three sector angles now differ, so its boundverts sit at
/// unequal spoke lengths and the position-only meet geometry declines it. The signed
/// dihedrals still sum well inside the band, so §8.5.1 admits it anyway: this is the
/// **oblique** member of §7.6.6 that the band exists for, and the one place the weld's
/// two sectors genuinely disagree about where a shared leg vertex lies.
fn sheared_cube_corner_adj(
    segments: u32,
    profile: f32,
    shear: f32,
) -> (LoopMesh, BevelResult, Vec3) {
    let mut mesh = LoopMesh::new();
    let place = |x: f32, y: f32, z: f32| Vec3::new(x + shear * y, y, z);
    let corners = [
        place(1.0, -1.0, -1.0),
        place(-1.0, -1.0, -1.0),
        place(-1.0, 1.0, -1.0),
        place(1.0, 1.0, -1.0),
        place(1.0, -1.0, 1.0),
        place(-1.0, -1.0, 1.0),
        place(-1.0, 1.0, 1.0),
        place(1.0, 1.0, 1.0),
    ];
    let vert: Vec<VertexKey> = corners.iter().map(|&p| mesh.create_vertex(p)).collect();
    let mut edge_of = |mesh: &mut LoopMesh, a: usize, b: usize| {
        mesh.create_edge([vert[a], vert[b]], EdgeCreateFlags::NONE)
    };
    let e01 = edge_of(&mut mesh, 0, 1);
    let e12 = edge_of(&mut mesh, 1, 2);
    let e23 = edge_of(&mut mesh, 2, 3);
    let e30 = edge_of(&mut mesh, 3, 0);
    let e45 = edge_of(&mut mesh, 4, 5);
    let e56 = edge_of(&mut mesh, 5, 6);
    let e67 = edge_of(&mut mesh, 6, 7);
    let e74 = edge_of(&mut mesh, 7, 4);
    let e04 = edge_of(&mut mesh, 0, 4);
    let e15 = edge_of(&mut mesh, 1, 5);
    let e26 = edge_of(&mut mesh, 2, 6);
    let e37 = edge_of(&mut mesh, 3, 7);

    let mut face_of = |mesh: &mut LoopMesh, ring: &[usize], edges: &[EdgeKey]| {
        let ring: Vec<VertexKey> = ring.iter().map(|&i| vert[i]).collect();
        mesh.face_create(&ring, edges, FaceCreateFlags::NONE);
    };
    face_of(&mut mesh, &[0, 1, 2, 3], &[e01, e12, e23, e30]);
    face_of(&mut mesh, &[1, 5, 6, 2], &[e15, e56, e26, e12]);
    face_of(&mut mesh, &[5, 4, 7, 6], &[e45, e74, e67, e56]);
    face_of(&mut mesh, &[4, 0, 3, 7], &[e04, e30, e37, e74]);
    face_of(&mut mesh, &[3, 2, 6, 7], &[e23, e26, e67, e37]);
    face_of(&mut mesh, &[0, 4, 5, 1], &[e04, e45, e15, e01]);

    let result = mesh.bevel(
        &[e01.into(), e30.into(), e04.into()],
        BevelOptions {
            offset: 0.25,
            affect: BevelAffect::Edges,
            segments,
            profile,
            vmesh_method: BevelVmeshMethod::Adj,
            ..Default::default()
        },
    );
    (mesh, result, corners[0])
}

/// BEVEL.md §8.5.5.5(f.1)'s third subtlety, on the corner that can actually see it: the
/// (f.4a) layout is built **in octant coordinates and affinely mapped to the corner once**,
/// like any other tri-corner grid. So every leg sample lies exactly on the straight run
/// `c + (1 − k·b)·(B_i − c)`, where `c` is the frame origin `(B_0 + B_1 + B_2 − v)/2`.
///
/// On the symmetric cube corner that is indistinguishable from reading each sector's own
/// 3-D cross-section and merging the two views of a shared leg vertex — the two agree
/// there algebraically. On the **oblique** ring the band deliberately admits (§7.6.6) they
/// do not: the two sectors' notch bottoms are different points, and a merge of them lands
/// on neither leg. This is that ring, and it is pinned to the octant model exactly.
#[test]
fn adj_square_in_weld_legs_are_the_octant_layout_on_an_oblique_ring() {
    for segments in [3u32, 4] {
        let (mesh, result, vertex) = sheared_cube_corner_adj(segments, 0.0, 0.25);
        mesh.validate_topology();

        // The corner's own verts: the ring is the only new geometry within half a bevel
        // width of `v`. Its boundverts are the three that still lie on an original face
        // plane — one per sector — while every leg runs off its plane toward `c`. Spoke
        // *length* would not separate them here: that is precisely what the shear breaks.
        let nearby: Vec<Vec3> = result
            .verts
            .iter()
            .map(|&v| mesh.vertex_position(v))
            .filter(|&p| (p - vertex).length() <= 0.5)
            .collect();
        let on_plane = |normal: Vec3, offset: f32| -> Vec3 {
            let hits: Vec<Vec3> = nearby
                .iter()
                .copied()
                .filter(|&p| (normal.dot(p) - offset).abs() <= 1.0e-4)
                .collect();
            assert_eq!(hits.len(), 1, "one boundvert per original face plane");
            hits[0]
        };
        let ring = [
            on_plane(Vec3::Z, -1.0),
            on_plane(Vec3::Y, -1.0),
            on_plane(Vec3::new(1.0, -0.25, 0.0), 1.0),
        ];

        assert!(
            !tri_corner_meet_geometry(vertex, &ring),
            "the oblique corner is admitted by the band alone (segments = {segments})",
        );

        let centre = (ring[0] + ring[1] + ring[2] - vertex) * 0.5;
        let step = square_in_weld_step(segments);
        for boundvert in ring {
            for k in 1..=(segments as usize / 2) {
                assert_position_present(
                    &nearby,
                    centre + (1.0 - k as f32 * step) * (boundvert - centre),
                    &format!("octant leg sample k = {k} at segments = {segments}"),
                );
            }
        }

        assert_eq!(
            result
                .faces
                .iter()
                .filter(|&&f| {
                    mesh.face_vert_keys(f)
                        .iter()
                        .all(|&v| (mesh.vertex_position(v) - vertex).length() <= 0.5)
                })
                .count(),
            usize::from(segments % 2 == 1),
            "the oblique ring still welds (segments = {segments})",
        );
    }
}

/// BEVEL.md §8.5.5.5(f.1)'s three-valued tri-corner predicate, on the answer that is
/// easiest to get backwards: step 3 needs the predicate to **fire**, and *doesn't apply*
/// is a different answer that reaches the tri-corner family only through step 4's
/// shortcut — whose own gate excludes `EXP_SQUARE_IN`. So a valence-**4** junction with
/// three beveled spokes and one in-plane passenger takes step 4's general grid at
/// square-in, not the (f.4a) weld, even though its ring is a perfect three-sector octant
/// and the band admits it.
///
/// The two arms are the same corner, the same exponent and the same offsets, differing
/// only by one coplanar diagonal — and they land different *counts*, which is what makes
/// the distinction observable rather than cosmetic.
#[test]
fn adj_square_in_in_plane_passenger_takes_the_grid_not_the_weld() {
    for segments in [2u32, 3] {
        let (welded, weld_result) = cube_corner_adj_profile(segments, 0.0);
        welded.validate_topology();
        assert_eq!(
            curved_corner_grid_faces(&welded, &weld_result).len(),
            usize::from(segments % 2 == 1),
            "control: valence 3 fires, so the weld owns the corner (segments = {segments})",
        );

        let (mesh, result) = cube_corner_in_plane_passenger_adj(segments, 0.0);
        mesh.validate_topology();
        let cap = curved_corner_grid_faces(&mesh, &result);
        assert_eq!(
            cap.len(),
            adj_general_ring_parity_counts(3, segments).0,
            "the passenger makes the predicate *not apply*, so step 4's grid fills the \
             corner (segments = {segments})",
        );
        assert!(
            mesh.vertex_count() > welded.vertex_count(),
            "the grid keeps the ring quads the weld deletes (segments = {segments})",
        );
    }
}
