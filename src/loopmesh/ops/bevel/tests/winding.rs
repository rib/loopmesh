use super::*;

/// BEVEL.md §8.2.2.1.1's Scope blockquote: the chord-fraction resample is a property
/// of a [V-ADJ] **cap's** boundary emission, and the cutoff fill is not a cap — it
/// consumes the coarse chain verts directly, so its chains are emitted **raw**. Every
/// cutoff cell the suite had was symmetric, where the resample is the identity and the
/// two readings coincide; the planar junction's outer sectors are skewed, and there it
/// is worth ~25 position quanta.
#[test]
fn cutoff_junction_chain_is_emitted_raw() {
    let (mut mesh, centre, rim) = planar_fan(6);
    let geom: Vec<GeomKey> = (0..3)
        .map(|k| {
            mesh.query_edge_exists(centre, rim[k])
                .expect("spoke exists")
                .into()
        })
        .collect();
    let result = mesh.bevel(
        &geom,
        BevelOptions {
            offset: 0.2,
            affect: BevelAffect::Edges,
            segments: 2,
            vmesh_method: BevelVmeshMethod::Cutoff,
            ..Default::default()
        },
    );
    mesh.validate_topology();

    // The skewed outer (tier-1) cell of the planar junction: legs 0.4 and 0.23.
    let cell = ProfileCell::from_control_points(
        Vec3::new(0.34641016, 0.2, 0.0),
        Vec3::new(-0.11547005, -0.2, 0.0),
        Vec3::ZERO,
    );
    let raw = cell.chain_positions(2, 0.5, None, ChainEmission::Raw)[1];
    let resampled = cell.chain_positions(2, 0.5, None, ChainEmission::Resampled)[1];
    assert!(
        (raw - resampled).length() > 1.0e-4,
        "the cell must be skewed enough for the two emissions to be distinguishable",
    );
    assert_eq!(result_has_vert_near(&mesh, &result, raw), 1);
    assert_eq!(result_has_vert_near(&mesh, &result, resampled), 0);
}

/// BEVEL.md §8.3.0.2 / §8.5.5.5: a valence-4 junction (and, at the far ends,
/// valence-4 terminals taking the poly-fan close) must wind as consistently as
/// the valence-3 cube corner.
#[test]
fn edge_bevel_valence_four_junction_winding_consistent_all_segments() {
    assert_octahedron_junction_bevel_sound(2..=3, &[BevelVmeshMethod::Adj]);
}

/// The same octahedron at `segments = 1`, where it is the beveled edges' *far*
/// (terminal) ends, not the apex junction, that carry the valence-4 close: each takes
/// the §7.5 on-edge vert on its one non-flanking leg and the §8.3.0.2 endpoint fill,
/// which at `segments = 1` is one triangle.
#[test]
fn edge_bevel_valence_four_terminal_single_segment_winding_consistent() {
    assert_octahedron_junction_bevel_sound(
        1..=1,
        &[BevelVmeshMethod::Adj, BevelVmeshMethod::Cutoff],
    );
}

/// BEVEL.md §8.3.0 / §6.7.5.2: the strip's rung join is decided by the re-stitched
/// outer rail that already spans the two chains' outer samples — never by a length
/// compare, which goes ill-conditioned as the offset shrinks (the aligned and
/// swapped rail sums then differ only at second order in offset / |E|) and twists
/// the strip. Asserted at the mechanism: on a closed solid every rail the strip
/// lays down must end radial-2, which a twisted strip cannot manage — it abandons
/// the re-stitched outer rails at radial-1 and mints crossing ones in their place.
#[test]
fn edge_bevel_narrow_offset_strip_is_not_twisted() {
    let both = [BevelVmeshMethod::Adj, BevelVmeshMethod::Cutoff];
    for offset in [1.0e-4f32, 1.0e-2] {
        // 1 = both strip ends terminal, 3 = both ends junctions. The mixed 2-edge
        // corner is left out: its 2-sector junction is short a fill face at *any*
        // offset (the pin below), so its result is not closed to begin with.
        for (edge_count, methods) in [(1, &both[..]), (3, &both[..])] {
            for segments in 2..=4 {
                for &vmesh_method in methods {
                    let mut mesh = LoopMesh::new();
                    mesh_add_cube(&mut mesh);
                    let corner = vert_at(&mesh, Vec3::new(1.0, -1.0, -1.0));
                    let geom: Vec<GeomKey> = mesh
                        .vert_edge_keys(corner)
                        .into_iter()
                        .take(edge_count)
                        .map(GeomKey::from)
                        .collect();

                    let result = mesh.bevel(
                        &geom,
                        BevelOptions {
                            offset,
                            affect: BevelAffect::Edges,
                            segments,
                            vmesh_method,
                            ..Default::default()
                        },
                    );

                    let case = format!(
                        "{edge_count} edge(s), offset = {offset}, segments = {segments}, \
                         {vmesh_method:?}"
                    );
                    assert!(!result.faces.is_empty(), "{case}: bevel produced no faces");
                    mesh.validate_topology();
                    for &edge in mesh.edge_keys().iter() {
                        let radial = mesh.edge_radial_iter(edge).count();
                        assert_eq!(
                            radial, 2,
                            "{case}: an edge ended radial-{radial} — the strip crossed E \
                             instead of running along it",
                        );
                    }
                }
            }
        }
    }
}

/// BEVEL.md §5.1 / §8.5.3: the [V-ADJ] cap's winding is the manifold condition against
/// the rim it welds to, so it must not depend on any *quantity* the profile scales. The
/// concave band (`profile < 0.25`, exponent `r < 1`) is where a geometric outward test
/// anchored on the fullness blend inverts — the blend's own sign flips there — and takes
/// the whole cap with it. Swept over the full valence-4 junction, where every sector is
/// a general grid cell rather than a closed-form octant.
#[test]
fn adj_cap_winding_survives_the_concave_profile_band() {
    for profile in [0.5f32, 0.3, 0.26, 0.255, 0.252, 0.2, 0.1, 0.02] {
        for segments in 2..=5 {
            let mut mesh = LoopMesh::new();
            mesh_add_octahedron(&mut mesh);
            let apex = vert_at(&mesh, Vec3::new(0.0, 0.0, 1.0));
            let geom: Vec<GeomKey> = mesh
                .vert_edge_keys(apex)
                .into_iter()
                .map(GeomKey::from)
                .collect();
            assert_eq!(geom.len(), 4, "the octahedron apex has valence 4");
            let result = mesh.bevel(
                &geom,
                BevelOptions {
                    offset: 0.2,
                    affect: BevelAffect::Edges,
                    segments,
                    profile,
                    vmesh_method: BevelVmeshMethod::Adj,
                    ..Default::default()
                },
            );
            assert_bevel_sound(
                &mesh,
                &result,
                &format!("valence-4 adj cap, segments = {segments}, profile = {profile}"),
            );
        }
    }
}

/// The same rule at a **vertex** bevel, whose multi-segment cap enters the general
/// [V-ADJ] fill unconditionally (BEVEL.md §8.11.3 — no closed-form octant path), so it is
/// a second, independent way into the grid emitter.
#[test]
fn vertex_bevel_adj_cap_winding_survives_the_concave_profile_band() {
    for profile in [0.5f32, 0.252, 0.15] {
        for segments in 2..=4 {
            let mut mesh = LoopMesh::new();
            mesh_add_cube(&mut mesh);
            let corner = vert_at(&mesh, Vec3::new(1.0, -1.0, -1.0));
            let result = mesh.bevel(
                &[corner.into()],
                BevelOptions {
                    offset: 0.3,
                    affect: BevelAffect::Vertices,
                    segments,
                    profile,
                    vmesh_method: BevelVmeshMethod::Adj,
                    ..Default::default()
                },
            );
            assert_bevel_sound(
                &mesh,
                &result,
                &format!("vertex adj cap, segments = {segments}, profile = {profile}"),
            );
        }
    }
}

/// BEVEL.md §5.1: the cap is **one** cycle wound with the disk order, so its consistency
/// cannot be decided face by face against a point. A reflex junction's boundvert ring is
/// non-planar — the reflex boundvert sits far off the plane the other two span — so no
/// single point lies below every cell at once, and a per-cell test splits the cap: the
/// cells over the convex boundverts flip while the reflex one stays put. The rim's radial
/// count is the invariant that catches it.
#[test]
fn adj_cap_winding_is_consistent_over_a_reflex_ring() {
    for profile in [0.25f32, 0.2, 0.5, 0.6, 0.75, 0.9] {
        for segments in 2..=5 {
            let (mut mesh, _apex, spokes) = single_face_reflex_junction();
            let geom: Vec<GeomKey> = spokes.into_iter().map(GeomKey::from).collect();
            let result = mesh.bevel(
                &geom,
                BevelOptions {
                    offset: 0.2,
                    affect: BevelAffect::Edges,
                    segments,
                    profile,
                    vmesh_method: BevelVmeshMethod::Adj,
                    ..Default::default()
                },
            );

            let case = format!("reflex adj cap, segments = {segments}, profile = {profile}");
            mesh.validate_topology();
            assert!(!result.faces.is_empty(), "{case}: bevel produced no faces");
            let conflicts = winding_conflicts(&mesh);
            assert!(
                conflicts.is_empty(),
                "{case}: {} shared edge(s) traversed the same way by both adjacent faces",
                conflicts.len(),
            );
        }
    }
}

/// Bevel two of the four edges at the octahedron apex — a valence-4 junction at
/// one end of each strip, a valence-4 terminal (the poly-fan close, §8.3.0.2) at
/// the other — and assert every result is sound.
fn assert_octahedron_junction_bevel_sound(
    segment_range: std::ops::RangeInclusive<u32>,
    methods: &[BevelVmeshMethod],
) {
    for segments in segment_range {
        for &vmesh_method in methods {
            for profile in [0.5f32, 0.3] {
                let mut mesh = LoopMesh::new();
                mesh_add_octahedron(&mut mesh);
                let apex = vert_at(&mesh, Vec3::new(0.0, 0.0, 1.0));
                let incident = mesh.vert_edge_keys(apex);
                assert_eq!(incident.len(), 4, "the octahedron apex has valence 4");
                let geom: Vec<GeomKey> = incident.into_iter().take(2).map(GeomKey::from).collect();

                let result = mesh.bevel(
                    &geom,
                    BevelOptions {
                        offset: 0.2,
                        affect: BevelAffect::Edges,
                        segments,
                        profile,
                        vmesh_method,
                        ..Default::default()
                    },
                );

                let case = format!(
                    "valence-4 junction, segments = {segments}, profile = {profile}, \
                     {vmesh_method:?}"
                );
                assert_bevel_sound(&mesh, &result, &case);
                assert!(
                    result.faces.len() >= 2 * segments as usize,
                    "{case}: {} new face(s) — expected at least one strip quad per \
                     segment per beveled edge",
                    result.faces.len(),
                );
            }
        }
    }
}
