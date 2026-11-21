use super::*;

/// How many edges of the *post-bevel* mesh carry `tint`.
fn tint_count(mesh: &LoopMesh, layer: ElementAttributeKey, tint: [f32; 3]) -> usize {
    mesh.edge_keys()
        .into_iter()
        .filter(|&edge| tint_of(mesh, layer, edge) == tint)
        .count()
}

/// BEVEL.md §7.7.2: a beveled edge's whole per-edge block lands on **exactly two**
/// edges — its strip's two *outer* longitudinal edges (profile index `0` and index
/// `segments`), one per rebuilt adjacent face. The count is two at every `segments`;
/// the `segments − 1` interior longitudinal lines descend from no original and keep
/// the layer default. An implementation that flagged every longitudinal line would
/// give `segments + 1` here.
#[test]
fn edge_bevel_copies_the_edge_block_onto_exactly_two_outer_longitudinal_edges() {
    for segments in 1..=3 {
        for method in [BevelVmeshMethod::Adj, BevelVmeshMethod::Cutoff] {
            let mut mesh = LoopMesh::new();
            mesh_add_cube(&mut mesh);
            let layer = tint_every_edge(&mut mesh);
            let beveled = mesh
                .query_edge_exists(
                    vert_at(&mesh, Vec3::new(1.0, -1.0, -1.0)),
                    vert_at(&mesh, Vec3::new(-1.0, -1.0, -1.0)),
                )
                .expect("the cube edge");
            let tint = tint_of(&mesh, layer, beveled);

            mesh.bevel(
                &[GeomKey::from(beveled)],
                BevelOptions {
                    offset: 0.25,
                    affect: BevelAffect::Edges,
                    segments,
                    vmesh_method: method,
                    ..Default::default()
                },
            );

            assert_eq!(
                tint_count(&mesh, layer, tint),
                2,
                "segments = {segments}, {method:?}: the beveled edge's block belongs on \
                 its two outer longitudinal edges and nowhere else",
            );
        }
    }
}

/// BEVEL.md §7.7.3 regime 1 + §7.7.4: at a valence-3 terminal the absorbing face's
/// **winding-outgoing** leg donates its whole block to the far-end chain — so that
/// leg's tint ends up on `segments + 1` edges (the `segments` chain edges plus its own
/// re-pointed self) and the other leg's on 1.
///
/// The discriminating half is that the donor **flips between the beveled edge's two
/// endpoints**: `fL` and `fR` traverse `E` in opposite directions, so the face that
/// leaves each endpoint along `E` is `fL` at one end and `fR` at the other, and the
/// donated leg is the one *that* face shares. The two donors therefore lie on
/// **opposite sides** of `E` — which rules out every "whichever leg is on the left /
/// right" reading.
#[test]
fn edge_bevel_terminal_chain_takes_the_absorbing_faces_winding_outgoing_leg() {
    for segments in 1..=3 {
        let mut mesh = LoopMesh::new();
        mesh_add_cube(&mut mesh);
        let layer = tint_every_edge(&mut mesh);
        let [near, far] = [
            vert_at(&mesh, Vec3::new(1.0, -1.0, -1.0)),
            vert_at(&mesh, Vec3::new(-1.0, -1.0, -1.0)),
        ];
        let beveled = mesh.query_edge_exists(near, far).expect("the cube edge");
        let side_faces: Vec<FaceKey> = mesh
            .edge_radial_iter(beveled)
            .map(|loop_key| mesh.loop_face(loop_key))
            .collect();

        // Per endpoint: the absorbing face (the one that is neither `fL` nor `fR`) and
        // the leg it leaves the corner along.
        let mut donors: Vec<(VertexKey, EdgeKey)> = Vec::new();
        for endpoint in [near, far] {
            let absorbing = mesh
                .vert_face_keys(endpoint)
                .into_iter()
                .find(|face| !side_faces.contains(face))
                .expect("a valence-3 terminal has a third face");
            donors.push((endpoint, winding_outgoing(&mesh, absorbing, endpoint)));
        }

        // The two donors sit on opposite sides of `E`: one lies in `fL`, one in `fR`.
        let donor_sides: Vec<FaceKey> = donors
            .iter()
            .map(|&(_, leg)| {
                *side_faces
                    .iter()
                    .find(|&&face| {
                        mesh.edge_radial_iter(leg)
                            .any(|loop_key| mesh.loop_face(loop_key) == face)
                    })
                    .expect("a terminal's donor leg also borders one of E's side faces")
            })
            .collect();
        assert_ne!(
            donor_sides[0], donor_sides[1],
            "the donor flips between E's two endpoints — it is not a fixed side of E",
        );

        let expected: Vec<[f32; 3]> = donors
            .iter()
            .map(|&(_, leg)| tint_of(&mesh, layer, leg))
            .collect();
        let others: Vec<[f32; 3]> = [near, far]
            .iter()
            .zip(&donors)
            .map(|(&endpoint, &(_, donor))| {
                let other = mesh
                    .iter_disk_edges(endpoint)
                    .find(|&leg| leg != beveled && leg != donor)
                    .expect("a valence-3 terminal has a second leg");
                tint_of(&mesh, layer, other)
            })
            .collect();

        mesh.bevel(
            &[GeomKey::from(beveled)],
            BevelOptions {
                offset: 0.25,
                affect: BevelAffect::Edges,
                segments,
                ..Default::default()
            },
        );

        for (index, tint) in expected.iter().enumerate() {
            assert_eq!(
                tint_count(&mesh, layer, *tint),
                segments as usize + 1,
                "segments = {segments}: endpoint {index}'s donor leg's block belongs on \
                 the far-end chain's {segments} edges plus its own re-pointed self",
            );
        }
        for (index, tint) in others.iter().enumerate() {
            assert_eq!(
                tint_count(&mesh, layer, *tint),
                1,
                "segments = {segments}: endpoint {index}'s non-donor leg keeps its block \
                 to itself",
            );
        }
    }
}

/// BEVEL.md §7.7.3 regime 2, the first negative: at a terminal of valence **≥ 4** the
/// far-end chain borders the *new* fan fill rather than a rebuilt original face, so it
/// is no corner run and inherits **nothing**. What does inherit is each rebuilt
/// leg-face's one-edge corner run.
///
/// Every octahedron vertex is valence 4, so a lone beveled edge lands two such
/// terminals. Per endpoint, exactly **five** edges then carry a leg's block: the three
/// re-pointed legs themselves plus the two rebuilt leg-faces' corner-run edges — a
/// count **independent of `segments`**, which is precisely what a rule stated as "the
/// absorbed leg's block lands on the far-end chain" would break (it would grow with
/// `segments`).
#[test]
fn edge_bevel_valence_four_terminal_far_end_chain_inherits_nothing() {
    for segments in 2..=4 {
        let mut mesh = LoopMesh::new();
        mesh_add_octahedron(&mut mesh);
        let layer = tint_every_edge(&mut mesh);
        let [near, far] = [
            vert_at(&mesh, Vec3::new(0.0, 0.0, 1.0)),
            vert_at(&mesh, Vec3::new(1.0, 0.0, 0.0)),
        ];
        let beveled = mesh.query_edge_exists(near, far).expect("the apex spoke");
        let legs_of = |mesh: &LoopMesh, endpoint: VertexKey| -> Vec<[f32; 3]> {
            mesh.iter_disk_edges(endpoint)
                .filter(|&edge| edge != beveled)
                .map(|leg| tint_of(mesh, layer, leg))
                .collect()
        };
        let legs = [legs_of(&mesh, near), legs_of(&mesh, far)];

        mesh.bevel(
            &[GeomKey::from(beveled)],
            BevelOptions {
                offset: 0.2,
                affect: BevelAffect::Edges,
                segments,
                ..Default::default()
            },
        );

        for (index, endpoint_legs) in legs.iter().enumerate() {
            assert_eq!(
                endpoint_legs.len(),
                3,
                "an octahedron terminal has three legs"
            );
            let carried: usize = endpoint_legs
                .iter()
                .map(|&tint| tint_count(&mesh, layer, tint))
                .sum();
            assert_eq!(
                carried, 5,
                "segments = {segments}, endpoint {index}: three re-pointed legs plus the \
                 two rebuilt leg-faces' corner-run edges — the far-end chain takes none \
                 of them, so this cannot grow with `segments`",
            );
        }
    }
}

/// BEVEL.md §7.7.3, the second negative: a **boundary** terminal has no absorbing
/// third face, so no corner run and no donation — the far-end chain takes the layer
/// default and every leg keeps its block to itself. Only the beveled edge's own two
/// outer longitudinal edges inherit anything at all.
#[test]
fn edge_bevel_boundary_terminal_inherits_nothing() {
    for segments in 1..=3 {
        let (mut mesh, beveled) = two_quad_strip();
        let layer = tint_every_edge(&mut mesh);
        let beveled_tint = tint_of(&mesh, layer, beveled);
        let leg_tints: Vec<[f32; 3]> = mesh
            .edge_vert_keys(beveled)
            .into_iter()
            .flat_map(|endpoint| mesh.iter_disk_edges(endpoint).collect::<Vec<_>>())
            .filter(|&edge| edge != beveled)
            .map(|leg| tint_of(&mesh, layer, leg))
            .collect();

        mesh.bevel(
            &[GeomKey::from(beveled)],
            BevelOptions {
                offset: 0.25,
                affect: BevelAffect::Edges,
                segments,
                ..Default::default()
            },
        );

        assert_eq!(
            tint_count(&mesh, layer, beveled_tint),
            2,
            "segments = {segments}: the strip's two outer longitudinal edges still inherit",
        );
        for tint in &leg_tints {
            assert_eq!(
                tint_count(&mesh, layer, *tint),
                1,
                "segments = {segments}: a boundary terminal's leg donates to nothing — \
                 there is no rebuilt third face to carry a corner run",
            );
        }
    }
}

/// BEVEL.md §8.3.0.3's attribute note **and its scope box**: the cross weld's donation
/// governs `segments = 1` exactly as it governs `segments ≥ 2`. At `segments = 1` the
/// welded chain degenerates to the single chamfer edge between the junction's two
/// boundverts, and it takes the crossing leg's whole block just the same — so the
/// donor's tint lands on `segments + 1` edges (the `segments` chain edges plus the
/// re-pointed donor leg) at every `segments`, with no special case at either end of
/// the range. An implementation that wires the copy into its multi-segment chain
/// builder alone passes every `segments ≥ 2` test and drops this.
///
/// Which leg survives is fixed by the **mesh's own edge order**, not the caller's
/// input order, so the same two spokes handed over in either order donate the same
/// leg.
///
/// The square pyramid is the fixture the rule needs: its apex is valence 4 (so the two
/// beveled spokes can be *opposite* in the disk) while its base corners are valence-3
/// terminals, so the whole bevel is buildable at `segments = 1`. The octahedron's
/// terminals are valence 4 and are declined there.
#[test]
fn edge_bevel_cross_weld_donates_the_crossing_leg_at_every_segment_count() {
    for segments in 1..=3 {
        for input_order in [false, true] {
            let mut mesh = LoopMesh::new();
            let verts = square_pyramid(&mut mesh);
            let layer = tint_every_edge(&mut mesh);
            let apex = verts[0];
            let spoke = |mesh: &LoopMesh, base: usize| {
                mesh.query_edge_exists(apex, verts[base])
                    .expect("the apex spoke")
            };
            // `verts[1..]` is the base ring in rotational order, so `b0` / `b2` are the
            // two spokes **opposite** each other in the apex disk — the cross.
            let mut beveled = [spoke(&mesh, 1), spoke(&mesh, 3)];

            // The surviving donation is the last one written, and the operator walks its
            // beveled edges in the mesh's edge order — so the donor is the
            // disk-predecessor leg of the *higher-keyed* beveled edge.
            let last = *beveled
                .iter()
                .max_by_key(|edge| edge.as_usize())
                .expect("two beveled edges");
            let donor = walk_predecessor(&mesh, apex, last);
            let other = mesh
                .iter_disk_edges(apex)
                .find(|&leg| !beveled.contains(&leg) && leg != donor)
                .expect("the second crossing leg");
            assert_ne!(donor, other);
            let donor_tint = tint_of(&mesh, layer, donor);
            let other_tint = tint_of(&mesh, layer, other);

            if input_order {
                beveled.reverse();
            }
            mesh.bevel(
                &beveled.map(GeomKey::from),
                BevelOptions {
                    offset: 0.2,
                    affect: BevelAffect::Edges,
                    segments,
                    vmesh_method: BevelVmeshMethod::Adj,
                    ..Default::default()
                },
            );

            assert_eq!(
                tint_count(&mesh, layer, donor_tint),
                segments as usize + 1,
                "segments = {segments}, reversed input = {input_order}: the crossing leg's \
                 block belongs on the welded chain's {segments} edge(s) plus its own \
                 re-pointed self",
            );
            assert_eq!(
                tint_count(&mesh, layer, other_tint),
                1,
                "segments = {segments}, reversed input = {input_order}: the losing donor's \
                 block stays on its own re-pointed leg alone",
            );
        }
    }
}
