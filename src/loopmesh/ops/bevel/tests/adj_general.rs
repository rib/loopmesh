use super::*;

/// BEVEL.md §8.5.5.4 (even parity): `segments = 4` at the cube corner welds the
/// three sub-patches to one apex vertex on the body diagonal, with no central
/// face — the corner grid is 3·(segments/2)² = 12 quads.
#[test]
fn adj_cube_corner_segments_4_welded_apex() {
    let (mesh, result) = cube_corner_adj(4);
    adj_assert_invariants(&mesh, &result);

    // Invariant 4 (§8.5.5.4): the single welded apex on the body diagonal.
    let apex: Vec<VertexKey> = result
        .verts
        .iter()
        .copied()
        .filter(|&v| (mesh.vertex_position(v) - adj_apex_position()).length() < 1.0e-4)
        .collect();
    assert_eq!(apex.len(), 1, "even segments weld to one apex vertex");
    let apex = apex[0];
    let apex_faces: Vec<FaceKey> = mesh
        .face_keys()
        .into_iter()
        .filter(|&f| mesh.face_vert_keys(f).contains(&apex))
        .collect();
    assert_eq!(
        apex_faces.len(),
        3,
        "the three innermost cells meet the apex directly (no central face)",
    );

    // Invariant 3 (§8.5.5.6): 12 all-quad sub-patch faces.
    let patch_faces = adj_patch_faces(&mesh, &result);
    assert_eq!(
        patch_faces.len(),
        12,
        "3·(segments/2)² = 12 sub-patch quads"
    );
    for &f in &patch_faces {
        assert_eq!(
            mesh.face_vert_keys(f).len(),
            4,
            "every corner-grid face is a quad at even parity",
        );
    }
}

/// BEVEL.md §8.5.5.4 (odd parity): `segments = 3` at the cube corner places no
/// vertex on the body diagonal; the three sub-patches are bridged by seam quads
/// and closed by exactly one central `s`-gon (s = 3) triangle — 6 quads + 1
/// triangle.
#[test]
fn adj_cube_corner_segments_3_central_ngon() {
    let (mesh, result) = cube_corner_adj(3);
    adj_assert_invariants(&mesh, &result);

    // Invariant 4 (§8.5.5.4): no apex vertex on the body diagonal.
    assert_eq!(
        result
            .verts
            .iter()
            .filter(|&&v| (mesh.vertex_position(v) - adj_apex_position()).length() < 1.0e-4)
            .count(),
        0,
        "odd segments place no vertex on the body diagonal",
    );

    // Invariant 3/4: quad-dominant with exactly one central triangle.
    let patch_faces = adj_patch_faces(&mesh, &result);
    let triangles = patch_faces
        .iter()
        .filter(|&&f| mesh.face_vert_keys(f).len() == 3)
        .count();
    let quads = patch_faces
        .iter()
        .filter(|&&f| mesh.face_vert_keys(f).len() == 4)
        .count();
    assert_eq!(
        triangles, 1,
        "exactly one central s-gon (triangle) at odd parity"
    );
    assert_eq!(quads, 6, "3 corner sub-patch quads + 3 seam quads");
    assert_eq!(patch_faces.len(), 7);
}

/// BEVEL.md §8.5.5.4: the load-bearing parity contrast — an even target closes
/// the corner on a welded apex vertex and no central face, an odd target on a
/// central `s`-gon face and no apex vertex.
#[test]
fn adj_cube_corner_odd_even_central_parity_differs() {
    let (odd_mesh, odd) = cube_corner_adj(3);
    let (even_mesh, even) = cube_corner_adj(4);
    let apex_pos = adj_apex_position();

    let apex_count = |mesh: &LoopMesh, result: &BevelResult| {
        result
            .verts
            .iter()
            .filter(|&&v| (mesh.vertex_position(v) - apex_pos).length() < 1.0e-4)
            .count()
    };
    assert_eq!(
        apex_count(&even_mesh, &even),
        1,
        "even → welded apex vertex"
    );
    assert_eq!(apex_count(&odd_mesh, &odd), 0, "odd → no apex vertex");

    let central_triangles = |mesh: &LoopMesh, result: &BevelResult| {
        adj_patch_faces(mesh, result)
            .into_iter()
            .filter(|&f| mesh.face_vert_keys(f).len() == 3)
            .count()
    };
    assert_eq!(
        central_triangles(&odd_mesh, &odd),
        1,
        "odd → one central triangle"
    );
    assert_eq!(
        central_triangles(&even_mesh, &even),
        0,
        "even → no central face"
    );
}

/// Bevel the three oblique legs of the irregular tetrahedron's `v0` corner with
/// `segments`, default circular profile, `vmesh_method = ADJ` (BEVEL.md
/// §8.5.5.5). `v0` is a non-orthonormal valence-3 junction, so this declines the
/// §8.5.1 cube-corner snap and routes into the general non-symmetric
/// fullness-relaxation grid. Returns the mesh, result, and the original corner.
fn oblique_tetra_corner_adj(segments: u32) -> (LoopMesh, BevelResult, Vec3) {
    let mut mesh = LoopMesh::new();
    let [v0, v1, v2, v3] = irregular_tetra(&mut mesh);
    let corner = mesh.vertex_position(v0);
    let e01 = mesh.query_edge_exists(v0, v1).expect("tetra spoke {0,1}");
    let e02 = mesh.query_edge_exists(v0, v2).expect("tetra spoke {0,2}");
    let e03 = mesh.query_edge_exists(v0, v3).expect("tetra spoke {0,3}");
    let result = mesh.bevel(
        &[e01.into(), e02.into(), e03.into()],
        BevelOptions {
            offset: 0.35,
            affect: BevelAffect::Edges,
            segments,
            vmesh_method: BevelVmeshMethod::Adj,
            ..Default::default()
        },
    );
    (mesh, result, corner)
}

/// The corner-grid (cap) faces at the oblique junction, isolated by the
/// §8.5.5.7 "within a few `offset` of the corner" rule: every cap vert sits
/// within ~`offset·√2` of the original corner `v`, while the strips run the full
/// leg length (≥ 2.3 here) away from it, so a fixed `0.9` ball captures the cap
/// and excludes the strips. (No sphere — superellipsoid membership is dropped.)
fn oblique_patch_faces(mesh: &LoopMesh, result: &BevelResult, corner: Vec3) -> Vec<FaceKey> {
    result
        .faces
        .iter()
        .copied()
        .filter(|&f| {
            mesh.face_vert_keys(f)
                .iter()
                .all(|&v| (mesh.vertex_position(v) - corner).length() <= 0.9)
        })
        .collect()
}

/// Cap vertices keyed to how many cap (corner-grid) faces each touches. The
/// welded centre vertex (even parity) is the lone cap vert in exactly `sectors`
/// innermost cells (incidence 3 at `s = 3`); the odd parity has no such vert.
fn oblique_cap_incidence(
    mesh: &LoopMesh,
    patch: &[FaceKey],
) -> std::collections::HashMap<VertexKey, usize> {
    let mut incidence = std::collections::HashMap::new();
    for &f in patch {
        for v in mesh.face_vert_keys(f) {
            *incidence.entry(v).or_insert(0) += 1;
        }
    }
    incidence
}

/// The number of welded centre vertices in the cap: cap verts touching exactly
/// `sectors` (= 3) faces — the §8.5.5.5(e) even-parity signature (one), absent at
/// odd parity (zero, where the centre is a face not a vertex).
fn oblique_welded_centre_count(mesh: &LoopMesh, patch: &[FaceKey]) -> usize {
    oblique_cap_incidence(mesh, patch)
        .values()
        .filter(|&&c| c == 3)
        .count()
}

/// Whether the cap is **not** a cube-corner sphere (§8.5.5.7: superellipsoid
/// membership is dropped). Fits a sphere to four spread cap verts and reports the
/// max deviation of the rest; a true §8.5.1 snap would leave every cap vert on
/// one sphere of radius `offset` (deviation ≈ 0), so a large deviation is the
/// positive signal that the general fullness path produced the cap.
fn oblique_cap_off_sphere_deviation(mesh: &LoopMesh, patch: &[FaceKey]) -> f32 {
    let positions: Vec<Vec3> = oblique_cap_incidence(mesh, patch)
        .keys()
        .map(|&v| mesh.vertex_position(v))
        .collect();
    if positions.len() < 4 {
        return 0.0;
    }
    let quad = [
        positions[0],
        positions[positions.len() / 3],
        positions[2 * positions.len() / 3],
        positions[positions.len() - 1],
    ];
    match circumsphere(quad) {
        Some((c, r)) => positions
            .iter()
            .map(|p| ((*p - c).length() - r).abs())
            .fold(0.0_f32, f32::max),
        None => f32::INFINITY,
    }
}

/// Every face edge of a closed solid is shared with exactly one other face.
fn oblique_assert_manifold(mesh: &LoopMesh) {
    mesh.validate_topology();
    for face in mesh.face_keys() {
        for loop_key in mesh.face_loop_keys(face) {
            assert_eq!(
                mesh.edge_radial_iter(mesh.loop_edge(loop_key)).count(),
                2,
                "every edge is radial-2 (closed manifold)",
            );
        }
    }
}

/// BEVEL.md §8.5.5.5 / §8.5.5.7 (odd parity): the general non-symmetric fill at
/// `segments = 3` over an oblique valence-3 junction closes the centre with one
/// central `s`-gon (s = 3 → triangle) and **no** welded centre vertex — 6 quads
/// + 1 triangle, exactly the odd-parity histogram of the cube corner but with
/// the cap placed off the cube sphere by the §8.5.5.5(b) fullness blend.
#[test]
fn adj_nonsymmetric_corner_segments_3_central_ngon() {
    let (mesh, result, corner) = oblique_tetra_corner_adj(3);
    oblique_assert_manifold(&mesh);

    let patch = oblique_patch_faces(&mesh, &result, corner);
    assert!(
        !patch.is_empty(),
        "the oblique junction builds a corner-grid fill"
    );

    let triangles = patch
        .iter()
        .filter(|&&f| mesh.face_vert_keys(f).len() == 3)
        .count();
    let quads = patch
        .iter()
        .filter(|&&f| mesh.face_vert_keys(f).len() == 4)
        .count();
    assert_eq!(
        triangles, 1,
        "odd parity → exactly one central s-gon (triangle)"
    );
    assert_eq!(quads, 6, "3 corner sub-patch quads + 3 seam quads");
    assert_eq!(patch.len(), 7);

    assert_eq!(
        oblique_welded_centre_count(&mesh, &patch),
        0,
        "odd parity → no welded centre vertex (the centre is a face)",
    );

    // §8.5.5.7: the cap is NOT the cube-corner sphere — the general fullness
    // path fired, not the §8.5.1 snap.
    assert!(
        oblique_cap_off_sphere_deviation(&mesh, &patch) > 0.02,
        "the non-symmetric cap must not lie on a single offset-radius sphere",
    );
}

/// BEVEL.md §8.5.5.5 / §8.5.5.7 (even parity): the general non-symmetric fill at
/// `segments = 4` over the same oblique junction welds one shared centre vertex
/// (the §8.5.5.5(b) fullness centre) with **no** central face — 12 all-quad
/// sub-patch cells, the even-parity histogram of the cube corner placed off the
/// cube sphere.
#[test]
fn adj_nonsymmetric_corner_segments_4_welded_apex() {
    let (mesh, result, corner) = oblique_tetra_corner_adj(4);
    oblique_assert_manifold(&mesh);

    let patch = oblique_patch_faces(&mesh, &result, corner);
    assert_eq!(patch.len(), 12, "3·(segments/2)² = 12 sub-patch quads");
    for &f in &patch {
        assert_eq!(
            mesh.face_vert_keys(f).len(),
            4,
            "every cap face is a quad at even parity (no central s-gon)",
        );
    }

    assert_eq!(
        oblique_welded_centre_count(&mesh, &patch),
        1,
        "even parity → exactly one welded centre vertex shared by the s sectors",
    );

    assert!(
        oblique_cap_off_sphere_deviation(&mesh, &patch) > 0.02,
        "the non-symmetric cap must not lie on a single offset-radius sphere",
    );
}

/// BEVEL.md §8.5.5.5(e): the load-bearing parity contrast at the *general*
/// (non-symmetric) corner — odd closes on a central triangle and no welded
/// vertex; even welds a centre vertex and emits no central face. Both are the
/// general fullness cap (off the cube sphere), and the two patches are
/// observably distinct.
#[test]
fn adj_nonsymmetric_corner_odd_even_central_parity_differs() {
    let (odd_mesh, odd, odd_corner) = oblique_tetra_corner_adj(3);
    let (even_mesh, even, even_corner) = oblique_tetra_corner_adj(4);
    let odd_patch = oblique_patch_faces(&odd_mesh, &odd, odd_corner);
    let even_patch = oblique_patch_faces(&even_mesh, &even, even_corner);

    let triangles = |mesh: &LoopMesh, patch: &[FaceKey]| {
        patch
            .iter()
            .filter(|&&f| mesh.face_vert_keys(f).len() == 3)
            .count()
    };
    assert_eq!(
        triangles(&odd_mesh, &odd_patch),
        1,
        "segments = 3 → central triangle"
    );
    assert_eq!(
        triangles(&even_mesh, &even_patch),
        0,
        "segments = 4 → no central face"
    );

    assert_eq!(
        oblique_welded_centre_count(&odd_mesh, &odd_patch),
        0,
        "odd → no welded vertex"
    );
    assert_eq!(
        oblique_welded_centre_count(&even_mesh, &even_patch),
        1,
        "even → welded vertex"
    );

    // Both are the general (non-sphere) fullness cap.
    assert!(oblique_cap_off_sphere_deviation(&odd_mesh, &odd_patch) > 0.02);
    assert!(oblique_cap_off_sphere_deviation(&even_mesh, &even_patch) > 0.02);

    // The two parities are observably distinct patches (different cap topology).
    assert_ne!(
        oblique_cap_incidence(&odd_mesh, &odd_patch).len(),
        oblique_cap_incidence(&even_mesh, &even_patch).len(),
        "odd (central n-gon) and even (welded vertex) caps differ in vert count",
    );
}

/// A regular `sides`-gon pyramid: a valence-`sides` apex with `sides` slant edges, all
/// sectors congruent and all offsets equal. Beveling every slant edge presents the apex
/// as a symmetric **general** ring (it fails the §8.5.1 tri-corner trigger on valence
/// alone for `sides != 3`), so its cap is the §8.5.5.5 fill. Returns `(apex, base ring
/// in disk order)`.
fn regular_pyramid(mesh: &mut LoopMesh, sides: usize) -> (VertexKey, Vec<VertexKey>) {
    let apex = mesh.create_vertex(Vec3::new(0.0, 0.0, 1.4));
    let base: Vec<VertexKey> = (0..sides)
        .map(|k| {
            let angle = std::f32::consts::TAU * k as f32 / sides as f32;
            mesh.create_vertex(Vec3::new(angle.cos(), angle.sin(), 0.0))
        })
        .collect();
    for k in 0..sides {
        let ring = [apex, base[k], base[(k + 1) % sides]];
        let edges = [
            mesh.create_edge([ring[0], ring[1]], EdgeCreateFlags::NO_DUPLICATE),
            mesh.create_edge([ring[1], ring[2]], EdgeCreateFlags::NO_DUPLICATE),
            mesh.create_edge([ring[2], ring[0]], EdgeCreateFlags::NO_DUPLICATE),
        ];
        mesh.face_create(&ring, &edges, FaceCreateFlags::NONE);
    }
    let mut bottom: Vec<VertexKey> = base.clone();
    bottom.reverse();
    let bottom_edges: Vec<EdgeKey> = (0..sides)
        .map(|k| {
            mesh.create_edge(
                [bottom[k], bottom[(k + 1) % sides]],
                EdgeCreateFlags::NO_DUPLICATE,
            )
        })
        .collect();
    mesh.face_create(&bottom, &bottom_edges, FaceCreateFlags::NONE);
    (apex, base)
}

/// BEVEL.md §8.5.5.5(c)/(d) at a **power-of-two** target: `segments = 4` needs exactly
/// one doubling round, lands on the target resolution, and therefore runs **no**
/// interior resample — so the cap's interior is the subdivision-relaxation net's own
/// verts, deterministically. This pins the whole stencil at the symmetric valence-4
/// pyramid apex (a general ring: it fails §8.5.1 on valence alone): the face / edge /
/// ordinary-vertex masks, the Sabin extraordinary centre, and the boundary re-pin.
#[test]
fn adj_general_cap_valence4_segments4_interior_is_the_relaxed_net() {
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
            segments: 4,
            vmesh_method: BevelVmeshMethod::Adj,
            ..Default::default()
        },
    );
    mesh.validate_topology();

    let (boundverts, profiles) = pyramid_apex_ring(apex_position, &base, 0.3, 0.5);

    // Gate part (i) of §8.2.2.1.1: at `segments ∈ {4, 8, …}` the source resolution *is*
    // the target, so the read-back is the identity — nothing about the interior here is
    // the Tier-2 kernel's doing.
    let axis = adj_interior_axis(&profiles, 4, 4);
    for chain in &axis {
        for (k, &position) in chain.iter().enumerate() {
            assert_eq!(
                position,
                SourcePosition::at(k),
                "no resample runs at a power-of-two target",
            );
        }
    }

    // 4 sector-interior verts + 4 shared spokes + 1 welded centre.
    let predicted = predicted_cap_interior(
        apex_position,
        &boundverts,
        &profiles,
        4,
        adj_fullness(0.5, 4),
    );
    assert_eq!(predicted.len(), 9);
    for position in &predicted {
        assert_eq!(
            result_has_vert_near(&mesh, &result, *position),
            1,
            "the cap interior is the relaxed net's vert at {position:?}",
        );
    }

    // §8.5.5.5(b)/(c): the stencil *relaxes* the fullness seed — it does not emit it.
    // A bilinear blend of the sub-patch corners would leave the centre on the seed.
    let centroid = boundverts.iter().fold(Vec3::ZERO, |acc, &b| acc + b) / 4.0;
    let seed = centroid + adj_fullness(0.5, 4) * (apex_position - centroid);
    let centre = predicted[predicted.len() - 1];
    assert!(
        (centre - seed).length() > 1.0e-3,
        "the extraordinary-centre stencil pulls the centre off its fullness seed",
    );
    assert!(
        (centre - centroid).length() < (seed - centroid).length(),
        "and pulls it *in*, toward the subdivision-limit position",
    );
    assert_eq!(
        result_has_vert_near(&mesh, &result, seed),
        0,
        "the raw fullness seed is never emitted",
    );
}

/// A valence-3 cap ring whose three cross-section cells are **skewed**: each cell's
/// bulge control point sits `bulge_along` of the way along its chord rather than at the
/// midpoint, so `|mid − start| ≠ |mid − end|` and the chain's chord table is not
/// symmetric under reversal. `bulge_along = 0.5` is the mirror-symmetric control — the
/// three boundverts are equidistant from the origin, so the bulge then leaves the cell
/// on its perpendicular bisector.
///
/// Skew is a property of the **cell**, and the cell's two legs are `|mid − start|` and
/// `|mid − end|` — nothing about the ring's regularity decides it. A vertex-mode cap
/// builds each sector's cell as `from_control_points(B_i, B_{i+1}, v)` (§8.11.2 puts
/// `mid` at the corner vertex), so it is skewed exactly when the corner's ring slid its
/// two flanking edges by different distances. §4's ring slides *every* incident edge by
/// `offset`, which is why an unclamped vertex bevel is mirror-symmetric by construction
/// and quiet here (§8.5.5.5(d)'s vertex-mode bullet); §4.1's edge-length clamp, [W]
/// weights and the non-`OFFSET` `offset_type`s each break that and put a real skew on the
/// cell. This fixture reaches the same state directly, by moving the bulge control point
/// off the chord midpoint.
fn skewed_cap_ring(bulge_along: f32, profile: f32) -> (Vec<Vec3>, Vec<RingChainProfile<'static>>) {
    let boundverts = vec![Vec3::X, Vec3::Y, Vec3::Z];
    let profiles = (0..3)
        .map(|i| {
            let start = boundverts[i];
            let end = boundverts[(i + 1) % 3];
            let bulge = 0.3 * (start + end).normalize();
            RingChainProfile::Cell {
                cell: ProfileCell::from_control_points(
                    start,
                    end,
                    start + bulge_along * (end - start) + bulge,
                ),
                profile,
                custom: None,
            }
        })
        .collect();
    (boundverts, profiles)
}

/// The relaxed net a [`skewed_cap_ring`] descends to at `segments` (§8.5.5.5(b)/(c)),
/// seeded from the ring's own fullness centre toward the corner `(1, 1, 1)`.
fn skewed_cap_net(
    boundverts: &[Vec3],
    profiles: &[RingChainProfile],
    segments: u32,
    profile: f32,
) -> AdjNet {
    let mids: Vec<Vec3> = profiles.iter().map(|p| p.raw_positions(2)[1]).collect();
    let centroid = boundverts.iter().fold(Vec3::ZERO, |acc, &b| acc + b) / 3.0;
    let seed = centroid + adj_fullness(profile, segments) * (Vec3::ONE - centroid);
    adj_relaxed_net(boundverts, &mids, seed, profiles, segments)
}

/// Every **index-canonical** interior triple of an even-`segments` valence-3 cap
/// (§8.5.2): the `j == half` row belongs to the previous sub-patch's seam column, and
/// all `s` copies of `(half, half)` are the one welded centre.
fn canonical_interior_triples(sectors: usize, half: usize) -> Vec<(usize, usize, usize)> {
    let mut triples = Vec::new();
    for i in 0..sectors {
        for j in 1..=half {
            for k in 1..=half {
                if j < half || (k == half && i == 0) {
                    triples.push((i, j, k));
                }
            }
        }
    }
    triples
}

/// BEVEL.md §8.5.5.5(d), the ["even-`segments` mid index"] rule: at an even target the
/// mid index `half` is read off the chord table like every other index. On a skewed cell
/// that fraction is **not** the dense mid, and the whole table — mid included — is the
/// plain §8.2.2.1.1 read-back parametrisation.
///
/// Two controls the rule is the identity on, and which are why forcing the mid entry to
/// `resolution / 2` survives most fixtures: a mirror-symmetric cell, where the target's
/// `f_half` and the source's `g_{N/2}` are both `1/2`, and a power-of-two target, where
/// source and target tables are the same table. The second control reads
/// [`chord_fraction_source_positions`] directly rather than through [`adj_interior_axis`], whose
/// `resolution == segments` arm returns the identity table without consulting either
/// chain — that arm is pinned separately by
/// `adj_general_cap_valence4_segments4_interior_is_the_relaxed_net`.
#[test]
fn adj_even_mid_index_is_read_off_the_chord_table_not_pinned_to_the_dense_mid() {
    let (_, skewed) = skewed_cap_ring(0.9, 0.7);
    for segments in [6_u32, 10] {
        let resolution = boundary_source_resolution(segments);
        let half = (segments / 2) as usize;
        let axis = adj_interior_axis(&skewed, segments, resolution);
        for (i, profile) in skewed.iter().enumerate() {
            let table = chord_fraction_source_positions(
                &profile.raw_positions(segments),
                &profile.raw_positions(resolution),
            );
            assert_eq!(
                axis[i], table,
                "every index of chain {i} is the chord-fraction read-back, mid included",
            );
            // Measured `6.19e-3` at `segments = 6` and `5.45e-3` at `10`, so the margin
            // is ~6×. It also sits three orders of magnitude above the `1e-6` window
            // [`SourcePosition::snapped`] may erase, which is what keeps that snap from
            // quietly reinstating the forbidden pin *on this ring* — the window's safety
            // in general rests on the bound on its positional consequence, not on a
            // margin read off one fixture.
            assert!(
                (axis[i][half].value() - resolution as f32 / 2.0).abs() > 1.0e-3,
                "a skewed cell's mid chord fraction is provably not the dense mid: \
                 {:?} vs {}",
                axis[i][half],
                resolution as f32 / 2.0,
            );
        }
    }

    let (_, symmetric) = skewed_cap_ring(0.5, 0.7);
    for segments in [6_u32, 10, 12, 20] {
        let resolution = boundary_source_resolution(segments);
        let half = (segments / 2) as usize;
        let dense_mid = resolution as f32 / 2.0;
        for chain in adj_interior_axis(&symmetric, segments, resolution) {
            // The dense mid to within the ulp or two the two accumulations cost. Whether
            // the snap then restores it *exactly* is a question about the remainder, not
            // about this distance, and
            // `adj_symmetric_mid_source_position_snaps_where_the_remainder_is_under_the_window`
            // is where it is asked in the right unit.
            assert!(
                (chain[half].value() - dense_mid).abs() <= dense_mid * f32::EPSILON,
                "a mirror-symmetric cell reads the dense mid anyway: {:?} vs {dense_mid}",
                chain[half],
            );
        }
    }

    for profile in skewed.iter() {
        let table = chord_fraction_source_positions(
            &profile.raw_positions(8),
            &profile.raw_positions(boundary_source_resolution(8)),
        );
        for (k, &position) in table.iter().enumerate() {
            assert_eq!(
                position.value(),
                k as f32,
                "at a power-of-two target the source and target tables are one table",
            );
        }
    }
}

/// BEVEL.md §8.5.5.5(d), ["what to measure, and in what unit"]: the `1e-6` tolerance is
/// tested on the **span remainder**, so the measurement is a direct read of
/// [`chord_fraction_source_positions`]'s second return value — never a miss recovered
/// from a position, which has already lost it.
///
/// Measured on the mirror-symmetric [`skewed_cap_ring`], all three chains agreeing at
/// every count:
///
/// ```text
///   segments        6        10        12        16        20        24        34        40
///   N               8        16        16        16        32        32        64        64
///   index           3         7         8         7        16        15        32        32
///   remainder     1.0       1.0   8.15e-7       1.0   1.62e-6       1.0   4.85e-6   1.62e-6
///   k snaps?        —         —       YES         —        no         —        no        no
/// ```
///
/// Two readings of that table are worth spelling out, because both were got wrong from a
/// **collapsed** `index as f32 + remainder` measurement of the same fixture (which reads
/// `+0, +0, +1, +0, +1, +0, +1, +0` ulp of the dense mid — this test's stations reproduce
/// that fingerprint exactly, which is what makes them the same measurement re-taken in
/// the right unit):
///
/// - At `6, 10, 16, 24` the entry is **already exactly the dense mid**, in `locate`'s
///   upper-endpoint spelling `(N/2 − 1, 1)`. Its remainder is `1.0`, so the chain axis's
///   snap does **not** fire — and nothing is lost by that, because at a remainder of `1`
///   the kernel's two `lo` terms carry the exact factor `0` and the read is already the
///   pure copy of column `N/2` that a snap would have produced. The collapsed reading's
///   `+0` was decoded as "a tiny remainder, therefore snapped"; the truth is "no
///   residue at all, therefore no epsilon was ever consulted".
/// - At `40` the remainder is `1.62e-6` and the reference **declines**, yet the collapsed
///   reading is `+0`: one ulp of position `32` is `3.81e-6`, so the sum rounds the
///   remainder away and *manufactures* an exact hit, emitting a pure copy where a blend
///   with weight `1.62e-6` on column `33` is owed. That is the divergence the pair
///   representation exists to remove, and it is invisible to any position-space
///   measurement.
///
/// `12` is the station that decides the residue this snap was adopted for — the
/// mirror-symmetric cell whose innermost seam entries otherwise tip a position bucket —
/// and under the remainder rule it still snaps, at `8.15e-7` against `1e-6`.
#[test]
fn adj_symmetric_mid_source_position_snaps_where_the_remainder_is_under_the_window() {
    let (_, symmetric) = skewed_cap_ring(0.5, 0.7);
    // (segments, span index, remainder, does the chain axis snap, collapsed reading in
    // ulp of the dense mid — the last carried only as a cross-check that this is the
    // same measurement in a unit that cannot decide the question).
    let stations: [(u32, usize, f32, bool, i64); 8] = [
        (6, 3, 1.0, false, 0),
        (10, 7, 1.0, false, 0),
        (12, 8, 8.145_89e-7, true, 1),
        (16, 7, 1.0, false, 0),
        (20, 16, 1.620_055e-6, false, 1),
        (24, 15, 1.0, false, 0),
        (34, 32, 4.853_12e-6, false, 1),
        (40, 32, 1.617_707e-6, false, 0),
    ];
    let mut snapped_somewhere = false;
    let mut declined_somewhere = false;
    for (segments, index, remainder, snaps, collapsed_ulp) in stations {
        let resolution = boundary_source_resolution(segments);
        let half = (segments / 2) as usize;
        let dense_mid = resolution as f32 / 2.0;
        for profile in symmetric.iter() {
            // Straight off the producer, so nothing downstream can be masking the unit.
            let table = chord_fraction_source_positions(
                &profile.raw_positions(segments),
                &profile.raw_positions(resolution),
            );
            let position = table[half];
            assert_eq!(
                position.index, index,
                "segments {segments}: the mid entry's span index",
            );
            assert!(
                (position.remainder - remainder).abs() < 1.0e-12,
                "segments {segments}: the mid entry's remainder is {:e}, not {remainder:e}",
                position.remainder,
            );
            // A cross-check that the same measurement is being re-taken in the right
            // unit — not a parity quantity. It pins OUR chord accumulation's own
            // fingerprint, so a future mismatch here reads as "our accumulation moved",
            // never as "the reference moved".
            assert_eq!(
                position.value().to_bits() as i64 - dense_mid.to_bits() as i64,
                collapsed_ulp,
                "segments {segments}: the collapsed fingerprint",
            );

            // The chain axis: one-sided, on the remainder.
            assert_eq!(
                position.remainder < SOURCE_POSITION_INTEGRAL_EPSILON,
                snaps,
                "segments {segments}: the chain axis's snap decision",
            );
            if snaps {
                assert_eq!(position.snapped(), SourcePosition::at(index));
                assert_eq!(position.snapped().value(), dense_mid);
            } else {
                // Where the remainder is `1` the declined snap costs nothing: the read
                // is already the pure copy of column `index + 1` a snap would give.
                assert_eq!(position.snapped(), position);
            }

            // The ring axis: two-sided, on the PRE-reflection remainder. It fires
            // wherever the chain axis does, and additionally at the four counts whose
            // remainder is `1.0` — where it raises the position to the integer that pair
            // already denotes, so the composite is the identity in value.
            let two_sided =
                position.remainder.min(1.0 - position.remainder) < SOURCE_POSITION_INTEGRAL_EPSILON;
            assert_eq!(two_sided, snaps || position.remainder == 1.0);
            let ring = ring_source_position(position, resolution as usize);
            if two_sided {
                assert_eq!(
                    ring,
                    SourcePosition::at(resolution as usize / 2),
                    "segments {segments}: the reflection's own fixed point is integral, \
                     so both applications land on it exactly",
                );
            } else {
                assert_ne!(ring.remainder, 0.0, "segments {segments}");
            }

            snapped_somewhere |= snaps;
            declined_somewhere |= !two_sided;
        }
    }
    assert!(
        snapped_somewhere && declined_somewhere,
        "the sweep must contain a station on each side of the window, or it pins nothing",
    );
}

/// The snap's **call sites**, not its rule. [`SourcePosition::snapped`] is applied where
/// the two-axis kernel *consumes* the table rather than inside the shared producer
/// (which cannot carry it — the two axes want different sidedness), so pinning
/// [`SourcePosition::snapped`] in isolation leaves every consumer free to drop it.
/// This arm drives the three chain-axis consumers instead:
/// [`AdjInterior::Relaxed`], [`AdjInterior::Octant`] and
/// [`LoopMesh::replant_rim_off_the_saturating_net`].
///
/// `segments = 12` is the one station of the [`skewed_cap_ring`] sweep whose mid entry
/// carries a residue the window actually catches (`8.15e-7` — see
/// [`adj_symmetric_mid_source_position_snaps_where_the_remainder_is_under_the_window`]).
/// At every other count the entry is already integral or already the upper-endpoint
/// spelling of the dense mid, and a dropped snap is silent there — which is why an
/// interior test at an arbitrary count pins nothing about the call sites.
///
/// The net is synthetic so the leak is unmistakable rather than sub-ulp: every slot off
/// dense column `N/2` carries a `1e6` marker in `z`, which a dropped snap's `8.15e-7`
/// weight turns into a `~0.8` contamination of a coordinate that must read an exact `0`.
/// The `assert_ne!` beside each pin is what makes it a pin: it shows the fixture can see
/// the mutation the `assert_eq!` claims to exclude.
#[test]
fn the_chain_axis_snap_is_applied_at_every_kernel_call_site() {
    let (_, symmetric) = skewed_cap_ring(0.5, 0.7);
    let sectors = symmetric.len();
    let segments = 12usize;
    let resolution = boundary_source_resolution(segments as u32) as usize;
    let net_half = resolution / 2;
    let target_half = segments / 2;
    assert_eq!((resolution, net_half, target_half), (16, 8, 6));

    let axis = adj_interior_axis(&symmetric, segments as u32, resolution as u32);
    for chain in axis.iter() {
        let mid = chain[target_half];
        assert_eq!(mid.index, net_half, "the mid entry's span index");
        assert_ne!(mid.remainder, 0.0, "a residue the snap has to remove");
        assert!(
            mid.remainder < SOURCE_POSITION_INTEGRAL_EPSILON,
            "and one the window catches: {:e}",
            mid.remainder,
        );
    }

    // `z` is the marker: zero exactly on dense column `N/2`, enormous everywhere else.
    // The seam identifies `(i, half, k)` with `(i − 1, k, half)`, so the rule has to be
    // stated over *both* coordinates to stay consistent under it.
    let off_column = 1.0e6;
    let marker = |j: usize, k: usize| {
        let on_dense_mid = j == net_half || k == net_half;
        Vec3::new(
            j as f32,
            k as f32,
            if on_dense_mid { 0.0 } else { off_column },
        )
    };
    let chains: Vec<Vec<Vec3>> = (0..sectors)
        .map(|_| (0..=resolution).map(|k| marker(0, k)).collect())
        .collect();
    let mut net = AdjNet::new(sectors, net_half, chains, true);
    for i in 0..sectors {
        for j in 1..net_half {
            for k in 1..=net_half {
                net.set(i, j, k, marker(j, k));
            }
        }
    }

    let (i, j, k) = (0usize, 2usize, target_half);
    let previous = (i + sectors - 1) % sectors;
    let ring = ring_source_position(axis[previous][segments - j], resolution);
    let on_dense_column = net.sample(i, ring, SourcePosition::at(net_half));
    let leaked = net.sample(i, ring, axis[i][k]);
    assert_eq!(on_dense_column.z, 0.0);
    assert_ne!(
        leaked, on_dense_column,
        "the fixture cannot see a dropped snap, so nothing below pins one",
    );

    let relaxed = AdjInterior::Relaxed {
        net: &net,
        axis: &axis,
        sectors,
        segments,
    };
    assert_eq!(
        relaxed.position(i, j, k, target_half),
        on_dense_column,
        "the interior read of the mid column stays on dense column N/2",
    );

    let exponent = 3.0f32;
    let octant = UnitOctant::from_corner([Vec3::X, Vec3::Y, Vec3::Z], Vec3::ONE)
        .expect("the unit cube corner is a well-conditioned octant frame");
    let expected_octant = octant.to_world(snap_to_unit_superellipsoid(
        on_dense_column,
        exponent,
        false,
    ));
    assert_ne!(
        octant.to_world(snap_to_unit_superellipsoid(leaked, exponent, false)),
        expected_octant,
        "the surface snap must not wash the leak out before it can be seen",
    );
    let octant_interior = AdjInterior::Octant {
        net: &net,
        axis: &axis,
        sectors,
        segments,
        exponent,
        octant,
    };
    assert_eq!(
        octant_interior.position(i, j, k, target_half),
        expected_octant,
        "the octant read of the mid column stays on dense column N/2 too",
    );

    // The saturating-rim replant is the third consumer. Its `j` row comes from the
    // *previous* chain's last entry, so splice one that saturates; its `k` coordinate is
    // the same mid entry, and at `k == target_half` both remainders are then zero, so the
    // replanted vert must be a verbatim net slot.
    let mut saturating = axis.clone();
    for chain in saturating.iter_mut() {
        chain[segments] = SourcePosition {
            index: resolution - 5,
            remainder: 1.0,
        };
    }
    let rim_row = adj_rim_source_row(&saturating, sectors, segments, resolution, i)
        .expect("the spliced table saturates, so the rim leaves the profile polyline");
    assert_eq!(rim_row, SourcePosition::at(4));

    let mut mesh = LoopMesh::new();
    let chain_of: Vec<Vec<VertexKey>> = (0..sectors)
        .map(|_| {
            (0..=segments)
                .map(|_| mesh.create_vertex(Vec3::ZERO))
                .collect()
        })
        .collect();
    mesh.replant_rim_off_the_saturating_net(
        &chain_of,
        &net,
        &saturating,
        sectors,
        segments,
        target_half,
    );
    assert_eq!(
        mesh.vertex_position(chain_of[i][k]),
        marker(4, net_half),
        "the replanted rim sample is the verbatim slot at (row 4, dense column N/2)",
    );
    assert_ne!(
        net.sample(i, rim_row, saturating[i][k]),
        marker(4, net_half),
        "and the fixture can see that call site's snap go missing too",
    );
}

/// BEVEL.md §8.5.5.5(d), ["the tolerance's *metric*"] Consequence 1(b): the window is
/// **index-independent**, which is a property only the `(span index, remainder)` pair
/// can have. A collapsed `index as f32 + remainder` spelling quantises the remainder by
/// half an ulp of the *position* before the comparison runs, buying an effective
/// threshold that wobbles with the index — `+43%` on `[8, 16)`, `−5%` on `[16, 32)`,
/// `+281%` on `[64, 128)` — and additionally manufacturing exact hits.
///
/// Each remainder below is chosen to sit on the far side of the *collapsed* threshold
/// from the specified one, so the assertion fails under a collapse and passes only under
/// the pair.
#[test]
fn source_position_snap_window_does_not_depend_on_the_span_index() {
    // Declined by the spec at every index; the collapsed spelling's `1.43e-6` effective
    // threshold on `[8, 16)` would take it.
    for index in [0usize, 1, 4, 8, 15, 32, 64, 127] {
        let position = SourcePosition {
            index,
            remainder: 1.2e-6,
        };
        assert_eq!(position.snapped(), position, "declined at index {index}");
    }
    // Snapped by the spec at every index; the collapsed spelling's `9.54e-7` effective
    // threshold on `[16, 32)` would decline it.
    for index in [0usize, 1, 4, 8, 16, 31, 64, 127] {
        let position = SourcePosition {
            index,
            remainder: 9.8e-7,
        };
        assert_eq!(
            position.snapped(),
            SourcePosition::at(index),
            "snapped at index {index}",
        );
    }
    // Declined by the spec; the collapsed spelling's `3.81e-6` threshold on `[64, 128)`
    // would take it.
    let wide = SourcePosition {
        index: 64,
        remainder: 3.0e-6,
    };
    assert_eq!(wide.snapped(), wide);

    // The manufactured exact hit, end to end through the kernel. At index `32` one ulp of
    // the position is `3.81e-6`, so a remainder of `1.5e-6` rounds away in a collapsed
    // sum and the read becomes a pure copy of column `32` — with no epsilon test
    // involved. Carried as a pair it is a genuine two-column blend, which is what the
    // reference emits.
    let sectors = 3usize;
    let net_half = 32usize;
    // Column `33` is the only one carrying a marker, so the blend's weight on it is read
    // off directly. Numbering the columns instead would hide the effect: at a magnitude
    // of `32` a `1.5e-6` displacement is below the coordinate's own ulp, which is the
    // guide's reason this divergence has no A/B witness.
    let chains: Vec<Vec<Vec3>> = (0..sectors)
        .map(|i| {
            (0..=2 * net_half)
                .map(|k| Vec3::new(i as f32, f32::from(k == 33), 0.0))
                .collect()
        })
        .collect();
    let net = AdjNet::new(sectors, net_half, chains, true);
    let manufactured = SourcePosition {
        index: 32,
        remainder: 1.5e-6,
    };
    assert_ne!(manufactured.snapped().remainder, 0.0);
    let blended = net.sample(0, SourcePosition::ORIGIN, manufactured.snapped());
    let copied = net.sample(0, SourcePosition::ORIGIN, SourcePosition::at(32));
    assert_ne!(
        blended, copied,
        "the declined snap must leave a weight on the next column, not a pure copy",
    );
    assert_eq!(copied.y, 0.0);
    assert_eq!(blended.y, 1.5e-6);
}

/// BEVEL.md §8.5.5.5(d), ["the tolerance's *metric*"] Consequence 2: the **sidedness** is
/// part of the rule. The chain (`k`) axis snaps from above only — it has no reflection,
/// so nothing supplies a from-below arm — while the ring (`j`) axis is two-sided on its
/// pre-reflection pair, because the same one rule is applied on both sides of the
/// reflection.
///
/// Widening the chain axis to a symmetric window is the single most likely way to
/// implement this subsection wrong while believing it has been implemented, so the
/// from-below arm is asserted absent rather than merely not exercised.
#[test]
fn source_position_snap_is_one_sided_on_the_chain_axis_and_two_sided_on_the_ring_axis() {
    let resolution = 16usize;
    let just_above = SourcePosition {
        index: 8,
        remainder: 5.0e-7,
    };
    let just_below = SourcePosition {
        index: 7,
        remainder: 1.0 - 5.0e-7,
    };

    // Chain axis: takes the one from above, declines the one from below — and what it
    // leaves behind is a genuine two-column blend, weighted by the surviving remainder.
    assert_eq!(just_above.snapped(), SourcePosition::at(8));
    assert_eq!(just_below.snapped(), just_below);
    assert_ne!(just_below.snapped().remainder, 0.0);

    // Ring axis: both, and both land on the same reflected integer.
    assert_eq!(
        ring_source_position(just_above, resolution),
        SourcePosition::at(8),
    );
    assert_eq!(
        ring_source_position(just_below, resolution),
        SourcePosition::at(8),
    );

    // The two windows meet at the integer rather than being disjoint, and the meeting
    // point is the reflection's own fixed point — reached on every mirror-symmetric
    // cell, in either spelling. Both applications are the identity there.
    assert_eq!(
        ring_source_position(SourcePosition::at(8), resolution),
        SourcePosition::at(8),
    );
    assert_eq!(
        ring_source_position(
            SourcePosition {
                index: 7,
                remainder: 1.0,
            },
            resolution,
        ),
        SourcePosition::at(8),
    );

    // A genuinely fractional position passes through both axes untouched, and the ring
    // axis's reflection of it is exact index arithmetic on the pair rather than a
    // subtraction of a position float.
    let fractional = SourcePosition {
        index: 6,
        remainder: 0.5,
    };
    assert_eq!(fractional.snapped(), fractional);
    assert_eq!(
        ring_source_position(fractional, resolution),
        SourcePosition {
            index: 9,
            remainder: 0.5,
        },
    );

    // The ring axis can never carry a remainder of `1`: the pass-through arm emits
    // `1 − remainder_prev` and both snapped arms emit `0`, and `remainder_prev == 0`
    // takes a snapped arm by definition. The chain axis carries one routinely.
    for remainder in [0.0f32, 1.0, 0.25, 5.0e-7, 1.0 - 5.0e-7] {
        let reflected = ring_source_position(
            SourcePosition {
                index: 3,
                remainder,
            },
            resolution,
        );
        assert_ne!(reflected.remainder, 1.0, "remainder {remainder}");
    }

    // The guard in the other direction: whatever the window's unit and sidedness, it
    // must not reach a genuine skew and reinstate the forbidden dense-mid pin.
    let (_, skewed) = skewed_cap_ring(0.9, 0.7);
    for segments in [6_u32, 10, 12, 16, 20, 24, 34, 40] {
        let source_resolution = boundary_source_resolution(segments);
        let half = (segments / 2) as usize;
        for profile in skewed.iter() {
            let table = chord_fraction_source_positions(
                &profile.raw_positions(segments),
                &profile.raw_positions(source_resolution),
            );
            assert_eq!(
                table[half].snapped(),
                table[half],
                "segments {segments}: the snap never reaches a skewed cell's mid entry",
            );
            // The ring axis's composite is the bare reflection there too — asserted on
            // the pair, because the pair form is *exact* index arithmetic where the
            // position-space `resolution − value` it would otherwise be compared against
            // costs an ulp.
            assert_eq!(
                ring_source_position(table[half], source_resolution as usize),
                SourcePosition {
                    index: source_resolution as usize - table[half].index - 1,
                    remainder: 1.0 - table[half].remainder,
                },
            );
        }
    }
}

/// BEVEL.md §8.5.5.5(d): pinning the read-back tables' mid entry to the dense mid — the
/// rule the guide writes out as *"WRONG. Do not do this."* — displaces exactly the
/// **seam columns** `(i, j, half)` for `j = 1 … half − 1`, i.e. `s · (half − 1)` cap
/// verts, with the welded centre excluded because the centre is placed by its own rule
/// and not through the tables.
///
/// The two arms run the real fill kernel twice over one net, differing only in the
/// tables, so the displaced set is attributable to nothing else. Both quiet cases from
/// the guide are here as controls: a power-of-two target and a mirror-symmetric cell.
#[test]
fn adj_even_mid_pin_would_displace_exactly_the_seam_columns() {
    let sectors = 3;
    let displaced_by_pinning = |boundverts: &[Vec3], profiles: &[RingChainProfile], segments| {
        let seg = segments as usize;
        let half = seg / 2;
        let net = skewed_cap_net(boundverts, profiles, segments, 0.7);
        let resolution = 2 * net.half as u32;
        let axis = adj_interior_axis(profiles, segments, resolution);
        let mut pinned = axis.clone();
        for chain in pinned.iter_mut() {
            chain[half] = SourcePosition::at(resolution as usize / 2);
        }
        let read_back = AdjInterior::Relaxed {
            net: &net,
            axis: &axis,
            sectors,
            segments: seg,
        };
        let mid_pinned = AdjInterior::Relaxed {
            net: &net,
            axis: &pinned,
            sectors,
            segments: seg,
        };
        let mut moved = Vec::new();
        let mut worst = 0.0_f32;
        for (i, j, k) in canonical_interior_triples(sectors, half) {
            let gap =
                (read_back.position(i, j, k, half) - mid_pinned.position(i, j, k, half)).length();
            worst = worst.max(gap);
            if gap > 1.0e-6 {
                moved.push((i, j, k));
            }
        }
        (moved, worst)
    };

    let (skewed_ring, skewed) = skewed_cap_ring(0.9, 0.7);
    for segments in [6_u32, 10] {
        let half = (segments / 2) as usize;
        let (moved, worst) = displaced_by_pinning(&skewed_ring, &skewed, segments);
        let expected: Vec<(usize, usize, usize)> = (0..sectors)
            .flat_map(|i| (1..half).map(move |j| (i, j, half)))
            .collect();
        assert_eq!(
            moved, expected,
            "only the seam columns move, and the welded centre is not among them",
        );
        assert_eq!(moved.len(), sectors * (half - 1));
        assert!(
            worst > 1.0e-4,
            "the displacement is real at segments = {segments}, not float noise: {worst}",
        );
    }

    let (moved, _) = displaced_by_pinning(&skewed_ring, &skewed, 8);
    assert!(
        moved.is_empty(),
        "at a power-of-two target the pin is the identity: {moved:?}",
    );

    let (symmetric_ring, symmetric) = skewed_cap_ring(0.5, 0.7);
    for segments in [6_u32, 12] {
        let (moved, _) = displaced_by_pinning(&symmetric_ring, &symmetric, segments);
        assert!(
            moved.is_empty(),
            "on a mirror-symmetric cell the pin is the identity: {moved:?}",
        );
    }
}

/// BEVEL.md §8.5.5.5(d), "What *is* pinned": the even welded centre `(i, half, half)` is
/// the dense net's own centre vertex taken verbatim — its own rule, not a consequence of
/// where the two flanking chord tables land. Its observable signature is that the centre
/// is keyed to the **source** resolution `N` and not to `segments`: it holds while
/// `segments` sweeps a whole power-of-two band and steps only when the band does.
#[test]
fn adj_even_welded_centre_is_the_dense_net_centre_keyed_to_the_source_resolution() {
    let (boundverts, profiles) = skewed_cap_ring(0.9, 0.7);
    let centre_at = |segments: u32| {
        let seg = segments as usize;
        let half = seg / 2;
        let net = skewed_cap_net(&boundverts, &profiles, segments, 0.7);
        let axis = adj_interior_axis(&profiles, segments, 2 * net.half as u32);
        let interior = AdjInterior::Relaxed {
            net: &net,
            axis: &axis,
            sectors: 3,
            segments: seg,
        };
        let emitted = interior.position(0, half, half, half);
        assert_eq!(
            emitted,
            net.at(0, net.half, net.half),
            "the welded centre is the dense net's centre verbatim",
        );
        emitted
    };

    // Within a band the centre still creeps by a few `1e-5` position quanta, because the
    // §8.5.5.5(b) fullness *seed* the net descends from carries a (near-nil) segment-count
    // term; the band step is two orders of magnitude larger.
    let low_band = [centre_at(6), centre_at(8)];
    let high_band = [centre_at(10), centre_at(12), centre_at(14), centre_at(16)];
    for band in [&low_band[..], &high_band[..]] {
        for centre in band {
            assert!(
                (*centre - band[0]).length() < 5.0e-5,
                "the centre holds across a whole source-resolution band: {centre:?} vs {:?}",
                band[0],
            );
        }
    }
    assert!(
        (low_band[0] - high_band[0]).length() > 1.0e-3,
        "and steps when the band does: {:?} vs {:?}",
        low_band[0],
        high_band[0],
    );
}

/// BEVEL.md §8.5.5.5(e) / §8.5.5.7 invariant 2 at valence 5, where the central `s`-gon
/// is a pentagon and the parity is therefore visible in the face-side histogram: even
/// `segments` weld one centre vertex and close no central face; odd `segments` weld
/// none and close exactly one pentagon. Both caps' interiors are the same relaxed net,
/// read back at their own target fractions.
#[test]
fn adj_general_cap_valence5_even_odd_centre_parity() {
    for segments in [4_u32, 3] {
        let mut mesh = LoopMesh::new();
        let (apex, base) = regular_pyramid(&mut mesh, 5);
        let apex_position = mesh.vertex_position(apex);
        let base_positions: Vec<Vec3> = base.iter().map(|&b| mesh.vertex_position(b)).collect();
        let slants: Vec<GeomKey> = base
            .iter()
            .map(|&b| mesh.query_edge_exists(apex, b).expect("slant edge").into())
            .collect();
        let result = mesh.bevel(
            &slants,
            BevelOptions {
                offset: 0.3,
                affect: BevelAffect::Edges,
                segments,
                vmesh_method: BevelVmeshMethod::Adj,
                ..Default::default()
            },
        );
        mesh.validate_topology();

        let (boundverts, profiles) = pyramid_apex_ring(apex_position, &base_positions, 0.3, 0.5);
        let predicted = predicted_cap_interior(
            apex_position,
            &boundverts,
            &profiles,
            segments,
            adj_fullness(0.5, segments),
        );
        for position in &predicted {
            assert_eq!(
                result_has_vert_near(&mesh, &result, *position),
                1,
                "segments {segments}: the cap interior is the relaxed net at {position:?}",
            );
        }

        let centroid = boundverts.iter().fold(Vec3::ZERO, |acc, &b| acc + b) / 5.0;
        let on_axis = |p: Vec3| (p - centroid).cross(apex_position - centroid).length() < 1.0e-5;
        let axial: Vec<VertexKey> = result
            .verts
            .iter()
            .copied()
            .filter(|&v| on_axis(mesh.vertex_position(v)))
            .collect();
        // At odd `segments` the only all-interior face is the central `s`-gon; at even
        // `segments` the innermost quad of every sector is all-interior too, so the
        // parity is read off the welded centre instead (§8.5.5.7 invariant 2).
        let central_faces: Vec<usize> = result
            .faces
            .iter()
            .filter(|&&face| {
                mesh.face_vert_keys(face).iter().all(|&v| {
                    predicted
                        .iter()
                        .any(|&p| (mesh.vertex_position(v) - p).length() < 1.0e-5)
                })
            })
            .map(|&face| mesh.face_vert_keys(face).len())
            .collect();

        if segments == 4 {
            // 5 sector-interior + 5 shared spokes + 1 welded centre.
            assert_eq!(predicted.len(), 11);
            assert_eq!(
                axial.len(),
                1,
                "even parity welds exactly one centre vertex on the cap axis",
            );
            let centre_faces = mesh.vert_face_keys(axial[0]);
            assert_eq!(
                centre_faces.len(),
                5,
                "the welded centre is the extraordinary point: one quad per sector fans to it",
            );
            for face in centre_faces {
                assert_eq!(
                    mesh.face_vert_keys(face).len(),
                    4,
                    "even parity closes no central face — every grid face stays a quad",
                );
            }
        } else {
            // 5 distinct central-ring verts, none of them on the axis.
            assert_eq!(predicted.len(), 5);
            assert!(
                axial.is_empty(),
                "odd parity lands no sample on the cap axis"
            );
            assert_eq!(
                central_faces,
                vec![5],
                "odd parity closes on exactly one central pentagon",
            );
        }
    }
}

/// BEVEL.md §8.5.5.5(c): the `(−1/6, +4/3, −1/6)` three-tap masks the **stencil input**
/// only. The net a round stores keeps its boundary exactly on the raw profile at that
/// round's own doubled resolution (the Dirichlet condition), so the mask never leaks
/// into the boundary the grid carries — and hence never into the rim the fill emits.
#[test]
fn adj_net_round_repins_the_raw_boundary_and_masks_only_the_input() {
    let mut mesh = LoopMesh::new();
    let (apex, base) = regular_pyramid(&mut mesh, 5);
    let apex_position = mesh.vertex_position(apex);
    let base_positions: Vec<Vec3> = base.iter().map(|&b| mesh.vertex_position(b)).collect();
    let (boundverts, profiles) = pyramid_apex_ring(apex_position, &base_positions, 0.3, 0.5);
    let mids: Vec<Vec3> = profiles.iter().map(|p| p.raw_positions(2)[1]).collect();
    let centroid = boundverts.iter().fold(Vec3::ZERO, |acc, &b| acc + b) / 5.0;
    let seed = centroid + adj_fullness(0.5, 4) * (apex_position - centroid);

    let coarse = AdjNet::coarse(&boundverts, &mids, seed, true);
    // The sharpening pulls each interior chain sample a sixth *away* from the mean of
    // its neighbours, so it moves *outward* from the chord — a curvature sharpening,
    // never the `(1/6, 2/3, 1/6)` smoothing average.
    let sharp = coarse.sharpened();
    for i in 0..5 {
        let raw = coarse.chains[i][1];
        let masked = sharp.chains[i][1];
        let chord_mid = 0.5 * (boundverts[i] + boundverts[(i + 1) % 5]);
        assert!(
            (masked - raw).length() > 1.0e-4,
            "the mask moves the stencil input"
        );
        assert!(
            (masked - chord_mid).length() > (raw - chord_mid).length(),
            "and moves it away from the chord — sharpening, not smoothing",
        );
        assert_eq!(
            coarse.at(i, 0, 1),
            raw,
            "the stored net keeps the raw sample"
        );
    }

    let doubled: Vec<Vec<Vec3>> = profiles.iter().map(|p| p.raw_positions(4)).collect();
    let next = coarse.subdivide(doubled.clone(), sabin_gamma(5));
    for i in 0..5 {
        for k in 0..=2 {
            assert_eq!(
                next.at(i, 0, k),
                doubled[i][k],
                "the round re-pins its boundary on the raw profile at the doubled resolution",
            );
        }
    }
}

/// Bevel the four slant edges of the square pyramid's apex with `segments`,
/// default circular profile, `vmesh_method = ADJ` (BEVEL.md §8.5.5.5(f), the
/// general-valence `s = 4` cap). Returns the mesh, result, and the apex corner.
fn pyramid_corner_adj(segments: u32) -> (LoopMesh, BevelResult, Vec3) {
    let mut mesh = LoopMesh::new();
    let [apex, b0, b1, b2, b3] = square_pyramid(&mut mesh);
    let corner = mesh.vertex_position(apex);
    let slants = [
        mesh.query_edge_exists(apex, b0).expect("slant {apex,0}"),
        mesh.query_edge_exists(apex, b1).expect("slant {apex,1}"),
        mesh.query_edge_exists(apex, b2).expect("slant {apex,2}"),
        mesh.query_edge_exists(apex, b3).expect("slant {apex,3}"),
    ];
    let result = mesh.bevel(
        &slants.map(Into::into),
        BevelOptions {
            offset: 0.3,
            affect: BevelAffect::Edges,
            segments,
            vmesh_method: BevelVmeshMethod::Adj,
            ..Default::default()
        },
    );
    (mesh, result, corner)
}

/// BEVEL.md §8.3.0.2: beveling one apex's spokes on a **closed solid** makes each
/// rim vert a valence-4 terminal whose one beveled edge (the top spoke) gets the
/// poly-fan close — two E-flanking boundverts, one interior peak, one leg vert on
/// the non-adjacent (bottom) spoke — and the **original rim vertex is consumed**.
/// The beveled junction apex is likewise consumed; the unbeveled opposite apex
/// survives. This is the closed-solid regression guard for the fan close.
#[test]
fn multisegment_closed_bipyramid_apex_fan_consumes_terminals() {
    let mut mesh = LoopMesh::new();
    let (top, rim, bottom) = square_bipyramid(&mut mesh);
    assert_eq!(counts(&mesh), (6, 12, 8), "octahedron 6 V / 12 E / 8 F");
    let rim_positions: Vec<Vec3> = rim.iter().map(|&r| mesh.vertex_position(r)).collect();

    let spokes: Vec<GeomKey> = rim
        .iter()
        .map(|&r| mesh.query_edge_exists(top, r).expect("top spoke").into())
        .collect();

    // `segments >= 2` routes through `bevel_edges_cutoff_multisegment`; the apex
    // is a junction (4 beveled edges), each rim vert a valence-4 terminal.
    let _ = mesh.bevel(
        &spokes,
        BevelOptions {
            offset: 0.25,
            affect: BevelAffect::Edges,
            segments: 2,
            vmesh_method: BevelVmeshMethod::Cutoff,
            ..Default::default()
        },
    );

    assert!(
        !mesh.vert_keys().contains(&top),
        "the beveled junction apex is consumed (its disk was fully re-stitched)",
    );
    assert!(
        mesh.vert_keys().contains(&bottom),
        "the unbeveled opposite apex survives untouched",
    );
    for (k, &r) in rim.iter().enumerate() {
        assert!(
            !mesh.vert_keys().contains(&r),
            "rim terminal {k} is consumed by the valence-4 fan close",
        );
        // The leg vert sits the plain offset along the bottom spoke from the old
        // rim position (§8.3.0.2).
        let bottom_dir = (mesh.vertex_position(bottom) - rim_positions[k]).normalize();
        let leg = rim_positions[k] + bottom_dir * 0.25;
        assert!(
            mesh.vert_keys()
                .into_iter()
                .any(|v| (mesh.vertex_position(v) - leg).length() < 2.0e-3),
            "rim terminal {k}'s leg vert at {leg:?} is minted on the bottom spoke",
        );
    }
}

/// The `segments = 2` ADJ variant of the same closed-solid apex bevel: the
/// terminal fan is vmesh-independent (§8.3.0.2), so ADJ consumes the rim
/// terminals with the same fan close as Cutoff.
#[test]
fn multisegment_adj_closed_bipyramid_apex_fan_consumes_terminals() {
    let mut mesh = LoopMesh::new();
    let (top, rim, _bottom) = square_bipyramid(&mut mesh);

    let spokes: Vec<GeomKey> = rim
        .iter()
        .map(|&r| mesh.query_edge_exists(top, r).expect("top spoke").into())
        .collect();

    let _ = mesh.bevel(
        &spokes,
        BevelOptions {
            offset: 0.25,
            affect: BevelAffect::Edges,
            segments: 2,
            vmesh_method: BevelVmeshMethod::Adj,
            ..Default::default()
        },
    );

    for (k, &r) in rim.iter().enumerate() {
        assert!(
            !mesh.vert_keys().contains(&r),
            "rim terminal {k} is consumed by the valence-4 fan close (ADJ)",
        );
    }
}

/// The cap's symmetry axis (BEVEL.md §8.5.5.7): the line through the original
/// corner `v` and the centroid of the cap verts. At a symmetric valence-4 corner
/// the welded apex (even parity) lies on it, while the central `s`-gon (odd
/// parity) straddles it with no vert on the axis.
fn cap_symmetry_axis(mesh: &LoopMesh, patch: &[FaceKey], corner: Vec3) -> (Vec3, Vec3) {
    let incidence = oblique_cap_incidence(mesh, patch);
    let centroid = incidence
        .keys()
        .fold(Vec3::ZERO, |acc, &v| acc + mesh.vertex_position(v))
        / incidence.len().max(1) as f32;
    (corner, (centroid - corner).normalize_or_zero())
}

/// Perpendicular distance of `point` from the line `base + t·direction`.
fn distance_to_axis(point: Vec3, base: Vec3, direction: Vec3) -> f32 {
    let offset = point - base;
    (offset - direction * offset.dot(direction)).length()
}

/// Count cap verts lying on the symmetry axis (BEVEL.md §8.5.5.7): one for the
/// even-parity welded apex, zero for odd parity (where the centre is a face). At
/// `s = 4` the central `s`-gon is itself a quad, so parity is read here off the
/// axis rather than from the face-side histogram.
fn cap_on_axis_vert_count(mesh: &LoopMesh, patch: &[FaceKey], corner: Vec3) -> usize {
    let (base, direction) = cap_symmetry_axis(mesh, patch, corner);
    oblique_cap_incidence(mesh, patch)
        .keys()
        .filter(|&&v| distance_to_axis(mesh.vertex_position(v), base, direction) < 0.02)
        .count()
}

/// Count cap faces centred on the symmetry axis (BEVEL.md §8.5.5.7): one central
/// `s`-gon for odd parity, zero for even parity (where the centre is a vertex).
fn cap_on_axis_face_count(mesh: &LoopMesh, patch: &[FaceKey], corner: Vec3) -> usize {
    let (base, direction) = cap_symmetry_axis(mesh, patch, corner);
    patch
        .iter()
        .filter(|&&f| {
            let verts = mesh.face_vert_keys(f);
            let centroid = verts
                .iter()
                .fold(Vec3::ZERO, |acc, &v| acc + mesh.vertex_position(v))
                / verts.len() as f32;
            distance_to_axis(centroid, base, direction) < 0.02
        })
        .count()
}

/// BEVEL.md §8.5.5.5(f) / §8.5.5.7 (odd parity, valence 4): the general fill at
/// `segments = 3` over the symmetric valence-4 apex is a topologically regular
/// 4×4 quad grid — 16 cap verts, 9 quad faces — whose centre is a central quad
/// straddling the symmetry axis with **no** welded vertex on the axis.
#[test]
fn adj_general_valence4_corner_segments_3_central_ngon() {
    let (mesh, result, corner) = pyramid_corner_adj(3);
    oblique_assert_manifold(&mesh);

    let patch = oblique_patch_faces(&mesh, &result, corner);
    assert_eq!(
        patch.len(),
        9,
        "(segments+1)² regular grid → 4×4 = 9 cap quads"
    );
    for &f in &patch {
        assert_eq!(
            mesh.face_vert_keys(f).len(),
            4,
            "every valence-4 cap face is a quad (the central s-gon is itself a quad)",
        );
    }
    assert_eq!(
        oblique_cap_incidence(&mesh, &patch).len(),
        16,
        "4×4 grid → 16 cap verts"
    );

    assert_eq!(
        cap_on_axis_vert_count(&mesh, &patch, corner),
        0,
        "odd parity → no welded vertex on the symmetry axis",
    );
    assert_eq!(
        cap_on_axis_face_count(&mesh, &patch, corner),
        1,
        "odd parity → one central quad centred on the symmetry axis",
    );
}

/// BEVEL.md §8.5.5.5(f) / §8.5.5.7 (even parity, valence 4): the general fill at
/// `segments = 4` over the symmetric valence-4 apex is a topologically regular
/// 5×5 quad grid — 25 cap verts, 16 quad faces — that welds **one** shared apex
/// vertex on the symmetry axis with no central face.
#[test]
fn adj_general_valence4_corner_segments_4_welded_apex() {
    let (mesh, result, corner) = pyramid_corner_adj(4);
    oblique_assert_manifold(&mesh);

    let patch = oblique_patch_faces(&mesh, &result, corner);
    assert_eq!(
        patch.len(),
        16,
        "(segments+1)² regular grid → 5×5 = 16 cap quads"
    );
    for &f in &patch {
        assert_eq!(
            mesh.face_vert_keys(f).len(),
            4,
            "every cap face is a quad at even parity"
        );
    }
    assert_eq!(
        oblique_cap_incidence(&mesh, &patch).len(),
        25,
        "5×5 grid → 25 cap verts"
    );

    assert_eq!(
        cap_on_axis_vert_count(&mesh, &patch, corner),
        1,
        "even parity → one welded apex vertex on the symmetry axis",
    );
    assert_eq!(
        cap_on_axis_face_count(&mesh, &patch, corner),
        0,
        "even parity → no cap face centred on the symmetry axis",
    );
}

/// BEVEL.md §8.5.5.5(e), valence 4: the load-bearing parity contrast read off the
/// symmetry axis (the §8.5.5.7 valence-4 reading, since the central `s`-gon is a
/// quad invisible to the face-side histogram). Odd straddles the axis with a
/// central quad and no on-axis vertex; even welds one on-axis apex vertex and no
/// central face. The two caps differ in vert count (16 vs 25).
#[test]
fn adj_general_valence4_corner_odd_even_central_parity_differs() {
    let (odd_mesh, odd, odd_corner) = pyramid_corner_adj(3);
    let (even_mesh, even, even_corner) = pyramid_corner_adj(4);
    let odd_patch = oblique_patch_faces(&odd_mesh, &odd, odd_corner);
    let even_patch = oblique_patch_faces(&even_mesh, &even, even_corner);

    assert_eq!(
        cap_on_axis_vert_count(&odd_mesh, &odd_patch, odd_corner),
        0,
        "odd → no on-axis vertex"
    );
    assert_eq!(
        cap_on_axis_vert_count(&even_mesh, &even_patch, even_corner),
        1,
        "even → one on-axis vertex"
    );
    assert_eq!(
        cap_on_axis_face_count(&odd_mesh, &odd_patch, odd_corner),
        1,
        "odd → one on-axis face"
    );
    assert_eq!(
        cap_on_axis_face_count(&even_mesh, &even_patch, even_corner),
        0,
        "even → no on-axis face"
    );

    assert_ne!(
        oblique_cap_incidence(&odd_mesh, &odd_patch).len(),
        oblique_cap_incidence(&even_mesh, &even_patch).len(),
        "odd (16) and even (25) valence-4 caps differ in vert count",
    );
}

/// A closed pentagonal pyramid: a five-way apex over a regular pentagon base,
/// five outward-wound side triangles and one base pentagon. The apex is a
/// **symmetric valence-5 junction** — beveling its five slant edges drives the
/// general valence-`s` (`s = 5`) adjacency-grid cap (BEVEL.md §8.5.5.5), whose
/// odd central `s`-gon is a **pentagon** directly visible in the face histogram
/// (§8.5.5.7) — the sharpest pin on the now-live higher-valence path, needing no
/// symmetry-axis reading. Returns `[apex, b0, b1, b2, b3, b4]`.
fn pentagonal_pyramid(mesh: &mut LoopMesh) -> [VertexKey; 6] {
    let apex = mesh.create_vertex(Vec3::new(0.0, 0.0, 1.4));
    let radius = 1.4_f32;
    let mut base = [apex; 5];
    for k in 0..5 {
        let angle = std::f32::consts::FRAC_PI_2 + k as f32 * std::f32::consts::TAU / 5.0;
        base[k] = mesh.create_vertex(Vec3::new(radius * angle.cos(), radius * angle.sin(), 0.0));
    }
    for k in 0..5 {
        let ring = [apex, base[k], base[(k + 1) % 5]];
        let edges = [
            mesh.create_edge([ring[0], ring[1]], EdgeCreateFlags::NO_DUPLICATE),
            mesh.create_edge([ring[1], ring[2]], EdgeCreateFlags::NO_DUPLICATE),
            mesh.create_edge([ring[2], ring[0]], EdgeCreateFlags::NO_DUPLICATE),
        ];
        mesh.face_create(&ring, &edges, FaceCreateFlags::NONE);
    }
    // Base pentagon, wound opposite the apex fan (downward / outward).
    let base_ring = [base[0], base[4], base[3], base[2], base[1]];
    let base_edges = [
        mesh.create_edge([base_ring[0], base_ring[1]], EdgeCreateFlags::NO_DUPLICATE),
        mesh.create_edge([base_ring[1], base_ring[2]], EdgeCreateFlags::NO_DUPLICATE),
        mesh.create_edge([base_ring[2], base_ring[3]], EdgeCreateFlags::NO_DUPLICATE),
        mesh.create_edge([base_ring[3], base_ring[4]], EdgeCreateFlags::NO_DUPLICATE),
        mesh.create_edge([base_ring[4], base_ring[0]], EdgeCreateFlags::NO_DUPLICATE),
    ];
    mesh.face_create(&base_ring, &base_edges, FaceCreateFlags::NONE);
    [apex, base[0], base[1], base[2], base[3], base[4]]
}

/// Bevel the five slant edges of the pentagonal pyramid's apex with `segments`,
/// default circular profile, `vmesh_method = ADJ` (BEVEL.md §8.5.5.5(f), the
/// general-valence `s = 5` cap). Returns the mesh, result, and the apex corner.
fn pentagon_corner_adj(segments: u32) -> (LoopMesh, BevelResult, Vec3) {
    let mut mesh = LoopMesh::new();
    let [apex, b0, b1, b2, b3, b4] = pentagonal_pyramid(&mut mesh);
    let corner = mesh.vertex_position(apex);
    let slants = [
        mesh.query_edge_exists(apex, b0).expect("slant {apex,0}"),
        mesh.query_edge_exists(apex, b1).expect("slant {apex,1}"),
        mesh.query_edge_exists(apex, b2).expect("slant {apex,2}"),
        mesh.query_edge_exists(apex, b3).expect("slant {apex,3}"),
        mesh.query_edge_exists(apex, b4).expect("slant {apex,4}"),
    ];
    let result = mesh.bevel(
        &slants.map(Into::into),
        BevelOptions {
            offset: 0.3,
            affect: BevelAffect::Edges,
            segments,
            vmesh_method: BevelVmeshMethod::Adj,
            ..Default::default()
        },
    );
    (mesh, result, corner)
}

/// The number of welded centre vertices in a valence-`sectors` cap: cap verts
/// touching exactly `sectors` faces — the §8.5.5.5(e) even-parity signature
/// (one), absent at odd parity (zero). Clean for `s = 3` and `s ≥ 5`, where the
/// welded centre's incidence `s` is distinct from the regular interior incidence
/// 4; at `s = 4` the two collide, which is why valence-4 parity is read off the
/// symmetry axis instead (§8.5.5.7).
fn cap_welded_centre_count(mesh: &LoopMesh, patch: &[FaceKey], sectors: usize) -> usize {
    oblique_cap_incidence(mesh, patch)
        .values()
        .filter(|&&c| c == sectors)
        .count()
}

/// BEVEL.md §8.5.5.5(e) / §8.5.5.7 (odd parity, valence 5): the general fill at
/// `segments = 3` over the symmetric valence-5 apex closes the centre with one
/// central `s`-gon — a **pentagon**, directly visible in the face histogram (the
/// reason `s = 5` pins the now-live higher-valence path more sharply than the
/// `s = 4` quad-grid case). 10 quads + 1 pentagon, and no welded centre vertex.
#[test]
fn adj_general_valence5_corner_segments_3_central_ngon() {
    let (mesh, result, corner) = pentagon_corner_adj(3);
    oblique_assert_manifold(&mesh);

    let patch = oblique_patch_faces(&mesh, &result, corner);
    let pentagons = patch
        .iter()
        .filter(|&&f| mesh.face_vert_keys(f).len() == 5)
        .count();
    let quads = patch
        .iter()
        .filter(|&&f| mesh.face_vert_keys(f).len() == 4)
        .count();
    assert_eq!(
        pentagons, 1,
        "odd parity → exactly one central s-gon (pentagon)"
    );
    assert_eq!(quads, 10, "5 corner sub-patch quads + 5 seam quads");
    assert_eq!(patch.len(), 11);

    assert_eq!(
        cap_welded_centre_count(&mesh, &patch, 5),
        0,
        "odd parity → no welded centre vertex (the centre is a pentagon face)",
    );
}

/// BEVEL.md §8.5.5.5(e) / §8.5.5.7 (even parity, valence 5): the general fill at
/// `segments = 4` over the same valence-5 apex welds one shared centre vertex
/// (incidence 5, distinct from the regular interior 4) with **no** central face
/// — 20 all-quad sub-patch cells, the even-parity histogram at `s = 5`.
#[test]
fn adj_general_valence5_corner_segments_4_welded_apex() {
    let (mesh, result, corner) = pentagon_corner_adj(4);
    oblique_assert_manifold(&mesh);

    let patch = oblique_patch_faces(&mesh, &result, corner);
    assert_eq!(
        patch.len(),
        20,
        "s·(segments/2)² = 5·4 = 20 sub-patch quads"
    );
    for &f in &patch {
        assert_eq!(
            mesh.face_vert_keys(f).len(),
            4,
            "every cap face is a quad at even parity (no central s-gon)",
        );
    }

    assert_eq!(
        cap_welded_centre_count(&mesh, &patch, 5),
        1,
        "even parity → exactly one welded centre vertex shared by the 5 sectors",
    );
}

/// BEVEL.md §8.5.5.5(e), valence 5: the load-bearing parity contrast read
/// straight off the face histogram (the central `s`-gon is a pentagon, not a
/// quad, so no symmetry-axis trick is needed). Odd closes on a central pentagon
/// and no welded vertex; even welds one centre vertex and emits no central face.
#[test]
fn adj_general_valence5_corner_odd_even_central_parity_differs() {
    let (odd_mesh, odd, odd_corner) = pentagon_corner_adj(3);
    let (even_mesh, even, even_corner) = pentagon_corner_adj(4);
    let odd_patch = oblique_patch_faces(&odd_mesh, &odd, odd_corner);
    let even_patch = oblique_patch_faces(&even_mesh, &even, even_corner);

    let pentagons = |mesh: &LoopMesh, patch: &[FaceKey]| {
        patch
            .iter()
            .filter(|&&f| mesh.face_vert_keys(f).len() == 5)
            .count()
    };
    assert_eq!(
        pentagons(&odd_mesh, &odd_patch),
        1,
        "segments = 3 → central pentagon"
    );
    assert_eq!(
        pentagons(&even_mesh, &even_patch),
        0,
        "segments = 4 → no central face"
    );

    assert_eq!(
        cap_welded_centre_count(&odd_mesh, &odd_patch, 5),
        0,
        "odd → no welded vertex"
    );
    assert_eq!(
        cap_welded_centre_count(&even_mesh, &even_patch, 5),
        1,
        "even → welded vertex"
    );

    // The two parities are observably distinct caps (different vert counts).
    assert_ne!(
        oblique_cap_incidence(&odd_mesh, &odd_patch).len(),
        oblique_cap_incidence(&even_mesh, &even_patch).len(),
        "odd (central pentagon) and even (welded vertex) caps differ in vert count",
    );
}
