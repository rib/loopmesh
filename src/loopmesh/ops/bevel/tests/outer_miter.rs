use super::*;

/// BEVEL.md §7.6.6: the base **sharp** bevel (no miter engaged) of a reflex
/// valence-3 junction still closes. The far-side reflex boundvert makes the
/// boundvert ring oblique, so the closed-form tri-corner snap declines and the
/// cap routes to the general §8.5.5.5 adjacency grid, which fills the junction
/// with a centre vertex + one quad per sector even at `segments = 2` rather than
/// leaving the reflex gap an open hole. The load-bearing signal is the Euler
/// characteristic: a dropped reflex cap reads as χ = 1 (one boundary loop); the
/// closed junction is χ = 2.
#[test]
fn base_sharp_reflex_junction_closes_at_segments2() {
    let (mesh, result) = reflex_apex_outer_miter(BevelMiter::Sharp);
    mesh.validate_topology();

    // §7.6.6 oracle: 17 V / 28 E / 13 F, a closed manifold.
    let (vertices, edges, faces) = counts(&mesh);
    assert_eq!(
        (vertices, edges, faces),
        (17, 28, 13),
        "sharp reflex junction total element counts (§7.6.6)",
    );
    assert_eq!(
        vertices as i64 - edges as i64 + faces as i64,
        2,
        "the closed sharp reflex bevel is χ = 2, not the χ = 1 open hole (§7.6.6)",
    );

    // The apex cap is the §8.5.5.5(a) three-quad pinwheel: one welded centre
    // vertex where exactly three cap faces meet, every one a quad. The centre is
    // the sole new (result) vertex with exactly three incident faces that are all
    // quads — the terminal-endpoint verts also carry three faces but include the
    // absorbing triangles, and the chain / boundvert samples carry four. Its exact
    // position is a Tier-2 interior (§8.5.5.7) that still trails the deferred
    // reflex-boundvert positioning residual, so it is not pinned here.
    let centre_hubs: Vec<VertexKey> = result
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
        centre_hubs.len(),
        1,
        "the reflex junction closes on exactly one welded centre vertex (§8.5.5.5(a))",
    );
    let centre = centre_hubs[0];

    let cap_faces: Vec<FaceKey> = mesh
        .face_keys()
        .into_iter()
        .filter(|&face| mesh.face_vert_keys(face).contains(&centre))
        .collect();
    assert_eq!(
        cap_faces.len(),
        3,
        "three cap quads meet at the welded centre (§8.5.5.5(a))",
    );
    for face in cap_faces {
        assert_eq!(
            mesh.face_vert_keys(face).len(),
            4,
            "each reflex-junction cap face is a quad",
        );
    }
}

/// BEVEL.md §8.9.2 / §8.9.5: `miter_outer = Arc` at a reflex `s = 3` junction rounds
/// the one reflex sector — it adds the arc-end boundvert `B3` and the arc bulge,
/// enlarging the ring from three to four boundverts and filling it with the general
/// valence-4 cap. The `Sharp` value adds none of that, so `Arc` is not inert: the
/// two meshes differ and `Arc` carries strictly more geometry, while staying a valid
/// closed manifold.
#[test]
fn outer_miter_arc_reshapes_reflex_junction() {
    let (arc_mesh, _arc) = reflex_apex_outer_miter(BevelMiter::Arc);
    arc_mesh.validate_topology();

    let (sharp_mesh, _sharp) = reflex_apex_outer_miter(BevelMiter::Sharp);
    assert_ne!(
        counts(&arc_mesh),
        counts(&sharp_mesh),
        "an ARC outer miter is not a silent no-op at a reflex junction",
    );
    assert!(
        arc_mesh.vertex_count() > sharp_mesh.vertex_count(),
        "the ARC outer miter adds the arc pair + bulge + cap the sharp corner omits",
    );
    assert!(
        arc_mesh.face_count() > sharp_mesh.face_count(),
        "the ARC outer miter fills the reflex corner the sharp seg=2 ring leaves open",
    );
}

/// BEVEL.md §8.9.2 / §8.9.6: outer `Patch` is the distinct **flat two-segment strip**
/// at the reflex sector — it promotes the meet corner to the middle boundvert `B2`,
/// giving the ring `B1 → B2 → B3` (`+2` boundverts vs the sharp baseline, one more
/// than `Arc`'s `+1`). So `Patch` is neither the sharp ring nor the arc round-over:
/// it carries strictly more geometry than both, stays a valid closed manifold, and
/// its result is distinct from each.
#[test]
fn outer_miter_patch_builds_flat_strip() {
    let (patch_mesh, _patch) = reflex_apex_outer_miter(BevelMiter::Patch);
    patch_mesh.validate_topology();
    let (sharp_mesh, _sharp) = reflex_apex_outer_miter(BevelMiter::Sharp);
    let (arc_mesh, _arc) = reflex_apex_outer_miter(BevelMiter::Arc);

    // The closed miter stays a manifold: χ = 2.
    let (pv, pe, pf) = counts(&patch_mesh);
    assert_eq!(
        pv as i64 - pe as i64 + pf as i64,
        2,
        "the outer Patch miter closes to a χ = 2 manifold (§8.9.2)",
    );

    // §8.9.6: Patch is a distinct construction from both the sharp ring and the arc.
    assert_ne!(
        counts(&patch_mesh),
        counts(&sharp_mesh),
        "outer Patch is not the sharp ring: it adds the flat B1→B2→B3 strip (§8.9.2)",
    );
    assert_ne!(
        counts(&patch_mesh),
        counts(&arc_mesh),
        "outer Patch is not mis-served as the Arc round-over (§8.9.2)",
    );

    // §8.9.6: Patch adds `+2` reflex-ring boundverts, Arc `+1`, so Patch carries the
    // most geometry of the three at the reflex junction.
    assert!(
        patch_mesh.vertex_count() > arc_mesh.vertex_count()
            && arc_mesh.vertex_count() > sharp_mesh.vertex_count(),
        "outer Patch (+2) > Arc (+1) > Sharp (+0) in reflex-ring boundverts (§8.9.6)",
    );
    assert!(
        patch_mesh.face_count() > arc_mesh.face_count()
            && arc_mesh.face_count() > sharp_mesh.face_count(),
        "the enlarged Patch ring tiles into more cap faces than Arc or Sharp (§8.9.6)",
    );
}

/// BEVEL.md §8.9.2 / §8.9.5: the worked single-face `PATCH`. `B1` / `B3` land at the
/// **same** repositioned arc-ends the `Arc` path produces (the neighbour-plane
/// reposition is mode-independent), the middle boundvert `B2` stays at the retained
/// §7.6.2 meet corner (the unmoved tent apex), and the two flat chord samples sit at
/// the straight segment midpoints `½(B1+B2)` / `½(B2+B3)` — with **no** corner-ward
/// pull. In particular the arc's pulled-in bulge sample does **not** appear.
#[test]
fn outer_miter_patch_flat_chord_placement() {
    // §8.9.1 classification + §8.9.2 reposition, replayed off the pristine mesh.
    let (b1, midpoint, b3) = single_face_reflex_patch_boundverts();
    let mid12 = 0.5 * (b1 + midpoint);
    let mid23 = 0.5 * (midpoint + b3);

    // The pulled-in arc bulge sample the `Arc` path would emit — Patch must not.
    let sqrt2 = std::f32::consts::SQRT_2;
    let arc_bulge = (2.0 - sqrt2) * (0.5 * (b1 + b3)) + (sqrt2 - 1.0) * midpoint;

    let (mesh, result) = reflex_apex_outer_miter_segments(BevelMiter::Patch, 2);
    mesh.validate_topology();

    assert_eq!(
        result_has_vert_near(&mesh, &result, b1),
        1,
        "B1 end is minted"
    );
    assert_eq!(
        result_has_vert_near(&mesh, &result, b3),
        1,
        "B3 end is minted"
    );
    assert_eq!(
        result_has_vert_near(&mesh, &result, midpoint),
        1,
        "B2 stays at the retained §7.6.2 meet corner (the unmoved tent apex)",
    );
    assert_eq!(
        result_has_vert_near(&mesh, &result, mid12),
        1,
        "the B1→B2 chord sample sits at its straight segment midpoint (§8.9.5)",
    );
    assert_eq!(
        result_has_vert_near(&mesh, &result, mid23),
        1,
        "the B2→B3 chord sample sits at its straight segment midpoint (§8.9.5)",
    );
    assert_eq!(
        result_has_vert_near(&mesh, &result, arc_bulge),
        0,
        "no vertex sits at the arc's pulled-in bulge — the Patch chords are flat (§8.9.5)",
    );
}

/// Re-derive the single-face reflex junction's three outer-`PATCH` ring boundverts
/// `(B1, B2, B3)` independently of the operator, by replaying the §8.9.1 signed reflex
/// classification and the §8.9.2 neighbour-plane reposition. `B2` is the retained
/// §7.6.2 meet-point (unmoved); `B1` / `B3` are it slid back along their own flanking
/// beveled edge until level with the neighbouring sector's boundvert. None of the three
/// depends on `segments` (§8.9.5.1's closing line), which is what the caller asserts.
fn single_face_reflex_patch_boundverts() -> (Vec3, Vec3, Vec3) {
    let offset = 0.2;
    let (mesh, apex, [va, vb, vc]) = single_face_reflex_junction();
    let apex_position = mesh.vertex_position(apex);
    let is_beveled = |edge: EdgeKey| edge == va || edge == vb || edge == vc;
    let corners = mesh.vertex_bevel_corner_ring(apex, &is_beveled);
    let sectors = sector_split(&corners, &is_beveled);
    let sector_count = sectors.len();
    let mut meets: Vec<Vec3> = Vec::new();
    let mut dir_open: Vec<Vec3> = Vec::new();
    let mut dir_close: Vec<Vec3> = Vec::new();
    let mut reflex: Vec<bool> = Vec::new();
    for sector in &sectors {
        let opening = sector.bounding_edges[0];
        let closing = sector.bounding_edges[1];
        let face = sector.corners[0].1;
        let d_open = mesh
            .edge_direction_from(opening, apex, apex_position)
            .unwrap();
        let d_close = mesh
            .edge_direction_from(closing, apex, apex_position)
            .unwrap();
        let cross = (-d_open).cross(-d_close);
        reflex.push(cross.dot(mesh.face_plane_normal(face)) < 0.0);
        meets.push(mesh.solve_meet_point(apex, apex_position, sector, offset));
        dir_open.push(d_open);
        dir_close.push(d_close);
    }
    let r = reflex.iter().position(|&x| x).expect("one reflex sector");
    let midpoint = meets[r];
    let predecessor = meets[(r + sector_count - 1) % sector_count];
    let successor = meets[(r + 1) % sector_count];
    let b1 = midpoint + (predecessor - midpoint).dot(dir_open[r]) * dir_open[r];
    let b3 = midpoint + (successor - midpoint).dot(dir_close[r]) * dir_close[r];
    (b1, midpoint, b3)
}

/// BEVEL.md §8.9.2 / §8.9.6: the outer `PATCH` miter has **no `segments` floor** — its
/// only gates are the `s ≥ 3` beveled-edge count and the §8.9.1 firing margin. So the
/// `+2`-boundvert ring is built at every `segments ≥ 2`, and the enlarged ring re-tiles
/// through the same §8.5.5.5 general grid the sharp and `ARC` rings ride.
///
/// The totals are not a closed form at `segments ≥ 2` (§8.9.6), so they are pinned
/// against the reference figures. They are internally consistent: each extra ring sector
/// contributes the same `(ΔV, ΔE, ΔF)` at a given `segments`, so `PATCH` (`+2` sectors
/// over sharp) sits exactly twice as far from sharp as `ARC` (`+1`) does — the second
/// block asserts that, which is what fails loudly if `PATCH` silently falls back to the
/// sharp ring at any count.
#[test]
fn outer_miter_patch_counts_are_segments_general() {
    for (segments, expected) in [
        (2_u32, (21_usize, 34_usize, 15_usize)),
        (3, (33, 55, 24)),
        (4, (47, 81, 36)),
        (5, (64, 112, 50)),
    ] {
        let (patch_mesh, _) = reflex_apex_outer_miter_segments(BevelMiter::Patch, segments);
        patch_mesh.validate_topology();
        let patch = counts(&patch_mesh);
        assert_eq!(
            patch, expected,
            "segments {segments}: the outer PATCH miter's element totals (§8.9.6)",
        );
        assert_eq!(
            patch.0 as i64 - patch.1 as i64 + patch.2 as i64,
            2,
            "segments {segments}: the outer PATCH miter closes to a χ = 2 manifold",
        );

        let (sharp_mesh, _) = reflex_apex_outer_miter_segments(BevelMiter::Sharp, segments);
        let (arc_mesh, _) = reflex_apex_outer_miter_segments(BevelMiter::Arc, segments);
        let sharp = counts(&sharp_mesh);
        let arc = counts(&arc_mesh);
        assert_ne!(
            patch, sharp,
            "segments {segments}: PATCH must not fall back to the sharp ring",
        );
        // §8.9.6: `ARC` is `+1` boundvert, `PATCH` `+2`, over the same sharp baseline —
        // and the general grid's per-sector contribution is uniform around the ring.
        assert_eq!(
            (patch.0 - sharp.0, patch.1 - sharp.1, patch.2 - sharp.2),
            (
                2 * (arc.0 - sharp.0),
                2 * (arc.1 - sharp.1),
                2 * (arc.2 - sharp.2)
            ),
            "segments {segments}: PATCH's `+2` boundverts cost twice ARC's `+1` (§8.9.6)",
        );
    }
}

/// BEVEL.md §8.9.2 / §8.9.5 / §8.9.2.1: the outer `PATCH` chord legs at general
/// `segments`. Everything asserted here is derived from the guide, not from a count
/// table:
///
/// - The two ends `B1` / `B3` are the §8.9.2 neighbour-plane reposition of the retained
///   §7.6.2 meet-point, and that reposition is **`segments`-independent** (§8.9.5.1's
///   closing line) and **mode-independent** (§8.9.2's worked `PATCH` paragraph: `PATCH`
///   reuses the identical end reposition `ARC` performs). So the same three positions
///   are re-derived once, off the pristine mesh, and must appear at every `segments`.
/// - `B2` is **not** repositioned — it stays at the stored meet-point (§8.9.2's tent).
/// - Each chord leg carries `segments − 1` **shoulder** samples "evenly spaced along the
///   straight line between its two endpoints" (§8.9.5), i.e. shoulder `j` of the leading
///   leg is exactly `B1 + (j / segments)·(B2 − B1)` — the equal-step ladder §8.9.2.1's
///   worked trace reads off the leg (`z: 0.457 → 0.785 → 1.112`, equal steps).
/// - No sample is pulled toward the corner: the `ARC` bulge position (a fraction
///   `√2 − 1` off the chord midpoint toward the stored corner, §8.9.5) is absent.
#[test]
fn outer_miter_patch_chord_shoulders_are_evenly_spaced_at_every_segments() {
    let (b1, b2, b3) = single_face_reflex_patch_boundverts();
    let sqrt2 = std::f32::consts::SQRT_2;
    let arc_bulge = (2.0 - sqrt2) * (0.5 * (b1 + b3)) + (sqrt2 - 1.0) * b2;

    for segments in [2_u32, 3, 4, 5] {
        let (mesh, result) = reflex_apex_outer_miter_segments(BevelMiter::Patch, segments);
        mesh.validate_topology();

        for (label, position) in [("B1", b1), ("B2", b2), ("B3", b3)] {
            assert_eq!(
                result_has_vert_near(&mesh, &result, position),
                1,
                "segments {segments}: {label} is the segments-independent ring boundvert",
            );
        }
        for j in 1..segments {
            let fraction = j as f32 / segments as f32;
            assert_eq!(
                result_has_vert_near(&mesh, &result, b1 + fraction * (b2 - b1)),
                1,
                "segments {segments}: leading-leg shoulder {j} sits at its even step",
            );
            assert_eq!(
                result_has_vert_near(&mesh, &result, b2 + fraction * (b3 - b2)),
                1,
                "segments {segments}: trailing-leg shoulder {j} sits at its even step",
            );
        }
        assert_eq!(
            result_has_vert_near(&mesh, &result, arc_bulge),
            0,
            "segments {segments}: the PATCH chords are flat — no corner-ward pull (§8.9.5)",
        );
    }
}

/// BEVEL.md §8.9.2.1 at general `segments`: the multi-face reflex `PATCH` closure keeps
/// its shape as the legs subdivide. Every reflex face still fans to the single hub `B2`
/// (identical to the `SHARP` outer join), the interior faces stay **triangles** at every
/// `segments` (their rim side is never subdivided), and the two flank faces open into
/// `(segments + 3)`-gons — a pentagon at `segments = 2`, a hexagon at `3`, a heptagon at
/// `4` — because the flank's whole patch leg (`B_flank`, its `segments − 1` shoulders,
/// `B2`) replaces the single apex corner of a triangle.
///
/// The `segments = 3` arm additionally pins the leading flank's **exact corner run**, not
/// just its size: §8.9.2.1 splices the leg `B1 → SH_1 → SH_2 → B2` in that order, with the
/// flank boundvert `B1` on the *beveled* side and the hub against the absorbed leg's rim
/// vert. A leg spliced in reverse keeps the face's size, the histogram and χ intact, so
/// only the ordered walk catches it.
#[test]
fn multiface_patch_flank_faces_grow_with_segments() {
    let (b1_pos, b2_pos, _) = octagonal_patch_boundvert_positions();

    for segments in [2_u32, 3, 4] {
        let (mut mesh, _apex, rim, [va, vb, vc]) = octagonal_bipyramid();
        let result = mesh.bevel(
            &[va.into(), vb.into(), vc.into()],
            BevelOptions {
                offset: 0.2,
                affect: BevelAffect::Edges,
                segments,
                vmesh_method: BevelVmeshMethod::Adj,
                miter_outer: BevelMiter::Patch,
                ..Default::default()
            },
        );
        mesh.validate_topology();
        let (v, e, f) = counts(&mesh);
        assert_eq!(
            v as i64 - e as i64 + f as i64,
            2,
            "segments {segments}: the multi-face PATCH miter closes to χ = 2",
        );

        let hub = result
            .verts
            .iter()
            .copied()
            .find(|&x| (mesh.vertex_position(x) - b2_pos).length() < 1.0e-4)
            .expect("the hub B2 stays at the retained meet-point at every segments");

        // The six reflex faces are those touching an absorbed leg's rim vert.
        let absorbed = [rim[3], rim[4], rim[5], rim[6], rim[7]];
        let mut sizes: Vec<usize> = Vec::new();
        for face in mesh.face_keys() {
            let face_verts = mesh.face_vert_keys(face);
            if !face_verts.iter().any(|x| absorbed.contains(x)) {
                continue;
            }
            if !face_verts.contains(&hub) {
                continue;
            }
            sizes.push(face_verts.len());
        }
        sizes.sort_unstable();
        let mut expected = vec![3; 4];
        expected.push(segments as usize + 3);
        expected.push(segments as usize + 3);
        expected.sort_unstable();
        assert_eq!(
            sizes, expected,
            "segments {segments}: four hub triangles + two `(segments + 3)`-gon flanks \
             (§8.9.2.1)",
        );

        if segments != 3 {
            continue;
        }
        // §8.9.2.1 / §8.9.5: the leading leg's ordered samples `B1, SH_1, SH_2, B2`, each at
        // its even step along the straight chord.
        let sample_at = |position: Vec3| {
            result
                .verts
                .iter()
                .copied()
                .find(|&x| (mesh.vertex_position(x) - position).length() < 1.0e-4)
                .expect("every leading-leg sample is minted")
        };
        let leading_leg: Vec<VertexKey> = (0..=segments)
            .map(|step| sample_at(b1_pos + (step as f32 / segments as f32) * (b2_pos - b1_pos)))
            .collect();
        let flank = mesh
            .face_keys()
            .into_iter()
            .find(|&face| {
                let face_verts = mesh.face_vert_keys(face);
                face_verts.len() == segments as usize + 3
                    && face_verts.contains(&leading_leg[0])
                    && face_verts.iter().any(|x| absorbed.contains(x))
            })
            .expect("the leading flank threads B1, its leg, the hub and an absorbed rim vert");
        let face_verts = mesh.face_vert_keys(flank);
        let size = face_verts.len();
        let start = face_verts
            .iter()
            .position(|&x| x == leading_leg[0])
            .expect("B1 sits on the flank face");
        // The run is contiguous and in leg order in *one* of the two walk directions (which
        // one depends only on the face's winding); the vertex one step past the hub is the
        // absorbed leg's rim vert, which is what fixes the leg's direction rather than just
        // its vertex set.
        let walk = |forward: bool, step: usize| {
            let offset = if forward { step } else { size - step };
            face_verts[(start + offset) % size]
        };
        let ordered = |forward: bool| {
            (0..=segments as usize).all(|step| walk(forward, step) == leading_leg[step])
                && absorbed.contains(&walk(forward, segments as usize + 1))
        };
        assert!(
            ordered(true) || ordered(false),
            "segments 3: the leading flank splices its leg as B1 → SH_1 → SH_2 → B2 with the \
             hub against the absorbed leg (§8.9.2.1)",
        );
    }
}

// Replicate the §8.9.1 classification / §8.9.2 reposition on a pristine octagonal
// bipyramid to recover the reflex sector's three patch boundvert positions
// `(B1, B2, B3)` independently of the operator.
fn octagonal_patch_boundvert_positions() -> (Vec3, Vec3, Vec3) {
    let offset = 0.2;
    let (mesh, apex, _rim, [va, vb, vc]) = octagonal_bipyramid();
    let apex_position = mesh.vertex_position(apex);
    let is_beveled = |edge: EdgeKey| edge == va || edge == vb || edge == vc;
    let corners = mesh.vertex_bevel_corner_ring(apex, &is_beveled);
    let sectors = sector_split(&corners, &is_beveled);
    let sector_count = sectors.len();
    let mut meets: Vec<Vec3> = Vec::new();
    let mut dir_open: Vec<Vec3> = Vec::new();
    let mut dir_close: Vec<Vec3> = Vec::new();
    let mut reflex: Vec<bool> = Vec::new();
    for sector in &sectors {
        let opening = sector.bounding_edges[0];
        let closing = sector.bounding_edges[1];
        let face = sector.corners[0].1;
        let d_open = mesh
            .edge_direction_from(opening, apex, apex_position)
            .unwrap();
        let d_close = mesh
            .edge_direction_from(closing, apex, apex_position)
            .unwrap();
        let cross = (-d_open).cross(-d_close);
        reflex.push(cross.dot(mesh.face_plane_normal(face)) < 0.0);
        meets.push(mesh.solve_meet_point(apex, apex_position, sector, offset));
        dir_open.push(d_open);
        dir_close.push(d_close);
    }
    assert_eq!(
        reflex.iter().filter(|&&x| x).count(),
        1,
        "the octagonal bipyramid presents exactly one reflex sector",
    );
    let r = reflex.iter().position(|&x| x).unwrap();
    let midpoint = meets[r];
    let predecessor = meets[(r + sector_count - 1) % sector_count];
    let successor = meets[(r + 1) % sector_count];
    let b1 = midpoint + (predecessor - midpoint).dot(dir_open[r]) * dir_open[r];
    let b3 = midpoint + (successor - midpoint).dot(dir_close[r]) * dir_close[r];
    (b1, midpoint, b3)
}

/// BEVEL.md §8.9.2.1: the multi-face reflex outer-`PATCH` re-stitch. Beveling the
/// three bunched apex legs of an octagonal bipyramid leaves a `270°` reflex sector
/// that absorbs the five unbeveled legs `A–R4 … A–R8`. `PATCH` materialises the
/// three ring boundverts `B1 → B2 → B3` (`B2` the retained lifted meet-point) and
/// closes the reflex hole by a **fan to the single hub `B2`** — every one of the six
/// reflex faces carries `B2`, exactly as the default `SHARP` outer join does. Its
/// sole departure from `SHARP` is at the two **flank faces**, which open into
/// pentagons along their subdivided patch legs (`B2 → shoulder → B_flank`); the four
/// interior faces stay bare hub triangles `[B2, R_k, R_{k+1}]`. The mesh stays a
/// closed `χ = 2` manifold. (This replaces a withdrawn run-tiling reading — see the
/// contrast note in §8.9.2.1.)
#[test]
fn multiface_patch_fans_reflex_hole_to_hub() {
    let (b1_pos, b2_pos, b3_pos) = octagonal_patch_boundvert_positions();

    let (mut mesh, _apex, rim, [va, vb, vc]) = octagonal_bipyramid();
    let result = mesh.bevel(
        &[va.into(), vb.into(), vc.into()],
        BevelOptions {
            offset: 0.2,
            affect: BevelAffect::Edges,
            segments: 2,
            vmesh_method: BevelVmeshMethod::Adj,
            miter_outer: BevelMiter::Patch,
            ..Default::default()
        },
    );
    mesh.validate_topology();

    let (v, e, f) = counts(&mesh);
    assert_eq!(
        v as i64 - e as i64 + f as i64,
        2,
        "the multi-face PATCH miter closes to a χ = 2 manifold (§8.9.2.1)",
    );

    // All three patch boundverts are minted; `B2` stays at the retained meet-point.
    let find = |p: Vec3| {
        result
            .verts
            .iter()
            .copied()
            .find(|&x| (mesh.vertex_position(x) - p).length() < 1.0e-4)
    };
    let b1 = find(b1_pos).expect("B1 minted");
    let b2 = find(b2_pos).expect("B2 minted at the retained meet-point");
    let b3 = find(b3_pos).expect("B3 minted");
    let boundverts = [b1, b2, b3];

    // The reflex faces are exactly those touching an absorbed leg's rim vert
    // (`R4 … R8`) and a patch boundvert. Classify each by which of `{B1, B2, B3}` it
    // threads: an interior fan face threads only the hub `B2` (stays a triangle); a
    // flank face threads the hub plus its flank boundvert (a pentagon carrying the
    // grid shoulder).
    let absorbed = [rim[3], rim[4], rim[5], rim[6], rim[7]];
    let mut interior_hub_tris = 0usize;
    let mut flank_pentagons: Vec<Vec<usize>> = Vec::new();
    let mut reflex_face_count = 0usize;
    for face in mesh.face_keys() {
        let face_verts = mesh.face_vert_keys(face);
        if !face_verts.iter().any(|x| absorbed.contains(x)) {
            continue;
        }
        let threaded: Vec<usize> = boundverts
            .iter()
            .enumerate()
            .filter(|(_, &b)| face_verts.contains(&b))
            .map(|(i, _)| i)
            .collect();
        if threaded.is_empty() {
            continue;
        }
        reflex_face_count += 1;
        // §8.9.2.1: every reflex face closes at the hub `B2` (index 1).
        assert!(
            threaded.contains(&1),
            "every reflex face carries the hub B2 (§8.9.2.1)",
        );
        match threaded.len() {
            1 => {
                assert_eq!(
                    face_verts.len(),
                    3,
                    "an interior fan face is a bare hub triangle [B2, R_k, R_k+1]",
                );
                interior_hub_tris += 1;
            }
            2 => {
                assert_eq!(
                    face_verts.len(),
                    5,
                    "a flank face opens into a pentagon threading its patch leg shoulder",
                );
                flank_pentagons.push(threaded);
            }
            n => panic!("a reflex face threads {n} boundverts; only the hub + one flank"),
        }
    }

    // §8.9.2.1 worked trace: six reflex faces = four interior hub triangles + two
    // flank pentagons. All carry the hub `B2`; only the flanks reach `B1` / `B3`.
    assert_eq!(reflex_face_count, 6, "six reflex faces wrap the 270° gap");
    assert_eq!(
        interior_hub_tris, 4,
        "four interior faces fan to the hub as triangles"
    );
    assert_eq!(
        flank_pentagons.len(),
        2,
        "two flank faces open into pentagons"
    );

    // One flank pentagon threads {B1, B2}, the other {B2, B3}; `B1` and `B3` each
    // appear in exactly that one reflex face (their own flank), never in the fan.
    let mut flanks = flank_pentagons.clone();
    for flank in flanks.iter_mut() {
        flank.sort_unstable();
    }
    flanks.sort();
    assert_eq!(
        flanks,
        vec![vec![0, 1], vec![1, 2]],
        "the two flank pentagons are the leading B1-B2 and trailing B2-B3 legs (§8.9.2.1)",
    );

    // The miter is not inert: it differs from the sharp baseline.
    let (mut sharp_mesh, _apex, _rim, [sa, sb, sc]) = octagonal_bipyramid();
    let _ = sharp_mesh.bevel(
        &[sa.into(), sb.into(), sc.into()],
        BevelOptions {
            offset: 0.2,
            affect: BevelAffect::Edges,
            segments: 2,
            vmesh_method: BevelVmeshMethod::Adj,
            miter_outer: BevelMiter::Sharp,
            ..Default::default()
        },
    );
    assert_ne!(
        counts(&mesh),
        counts(&sharp_mesh),
        "multi-face PATCH carries more geometry than the sharp ring",
    );
}

/// BEVEL.md §8.5.5.5(f.5) **case 2 (arc start)**: the fired-reflex leading boundvert
/// `B_1` seeds its one centreline from the arc's stored §7.6.2 middle control point.
///
/// The sector the arc **skips past** (the sector `B_3` opens) is not separately
/// suppressed: only the *consumed* arc-end boundvert `B_3` is held out of pass 1, while
/// that sector's trailing bounding boundvert is an ordinary sharp boundvert and — per the
/// arbitration rule — still emits its parallel-offset seed on the shared edge. The
/// closer-to-`v` winner is that pass-1 foot, **not** the pass-2 earlier-boundvert
/// projection (which is only the fallback for a *non-sharp* trailing boundvert). Here the
/// arc end `B_3` is slid inward off its owned edge, so the two disagree and the pass-1
/// foot is the (closer) survivor.
#[test]
fn square_seeds_arc_start_trailing_sector_takes_pass_one_sharp_seed() {
    let vertex = Vec3::ZERO;
    // Ring `[B_1, B_3, sharp2, sharp3]`, arc sector at index 0; ring edge 0 (`B_1 → B_3`)
    // is the internal arc, so `edge_in(B_3) = directions[1] = ZERO`.
    let e_lead = Vec3::new(0.0, 1.0, -1.0).normalize(); // B_1's flank
    let e_trail = Vec3::new(1.0, 0.0, -1.0).normalize(); // the edge `sharp2` shares with `B_3`
    let e_third = Vec3::new(-1.0, -1.0, -1.0).normalize();
    let directions = vec![e_lead, Vec3::ZERO, e_trail, e_third];
    let tips = vec![e_lead * 2.0, Vec3::ZERO, e_trail * 2.0, e_third * 2.0];
    let b1 = e_lead * 0.5;
    // `B_3` slid inward off `e_trail` (the §8.9.2 neighbour-plane reposition) so the
    // pass-1 sharp seed and the pass-2 `B_3` projection are distinguishable.
    let b3 = e_trail * 0.5 + Vec3::new(-0.05, 0.0, -0.05);
    let sharp2 = (e_trail + e_third).normalize() * 0.5;
    let sharp3 = (e_third + e_lead).normalize() * 0.5;
    let boundverts = vec![b1, b3, sharp2, sharp3];

    let arc_middle = Vec3::new(0.0, 0.4, 0.6);
    let arc_patch = ArcPatchStart {
        index: 0,
        mode: BevelMiter::Arc,
        arc_middle,
    };
    let seeds = square_ring_centreline_seeds(
        &directions,
        &tips,
        &boundverts,
        // The miter's own two ring slots are skipped as start / consumed either way; the
        // two sharp meet-points classify non-reflex and propose.
        &[false, false, true, true],
        vertex,
        std::slice::from_ref(&arc_patch),
    );
    assert_eq!(seeds.len(), 4);
    // Case 2: the arc sector's centreline starts at the stored middle control point.
    assert!(
        (seeds[0] - arc_middle).length() < 1.0e-6,
        "arc sector seeds from the stored §7.6.2 meet-point",
    );
    // Sector after the arc (index 1): the trailing sharp boundvert `sharp2` proposes the
    // parallel-offset foot on the shared edge `e_trail` via its `edge_in`.
    let expected = line_line_nearest_on_first(vertex, e_trail, sharp2, e_third)
        .expect("sharp2 proposes a foot on e_trail's line");
    assert!(
        (seeds[1] - expected).length() < 1.0e-6,
        "the sector after the arc takes the trailing sharp boundvert's pass-1 seed",
    );
    // It is the closer-to-`v` survivor, strictly nearer than the pass-2 `B_3` projection
    // fallback would have placed it (that fallback only fires for a non-sharp trailing bv).
    let pass_two_fallback = closest_point_on_segment(b3, vertex, tips[2]);
    assert!(
        seeds[1].length() < pass_two_fallback.length() - 1.0e-4,
        "the pass-1 sharp seed is closer to v than the pass-2 B_3 projection",
    );
}

/// BEVEL.md §8.5.5.5(f.5) **case 1 (patch start)**: `B_1` seeds *two* centrelines from the
/// plain boundvert-pair midpoints straddling the hub `B_2`, and consumes `B_2`, `B_3`.
#[test]
fn square_seeds_patch_start_case_one_midpoints() {
    let vertex = Vec3::ZERO;
    // Ring `[B_1, B_2, B_3, sharp3, sharp4]`, patch at index 0; ring edges 0 and 1 are the
    // internal patch chords, so `directions[1] = directions[2] = ZERO`.
    let e_lead = Vec3::new(0.0, 1.0, -1.0).normalize();
    let e_trail = Vec3::new(1.0, 0.0, -1.0).normalize();
    let e_a = Vec3::new(-1.0, -1.0, -1.0).normalize();
    let e_b = Vec3::new(-1.0, 1.0, -1.0).normalize();
    let directions = vec![e_lead, Vec3::ZERO, Vec3::ZERO, e_trail, e_a];
    let tips = vec![
        e_lead * 2.0,
        Vec3::ZERO,
        Vec3::ZERO,
        e_trail * 2.0,
        e_a * 2.0,
    ];
    let b1 = e_lead * 0.5;
    let b2 = Vec3::new(-0.1, -0.1, 0.6); // the retained lifted meet-point (hub)
    let b3 = e_trail * 0.5;
    let sharp3 = (e_a + e_b).normalize() * 0.5;
    let sharp4 = (e_b + e_lead).normalize() * 0.5;
    let boundverts = vec![b1, b2, b3, sharp3, sharp4];

    let arc_patch = ArcPatchStart {
        index: 0,
        mode: BevelMiter::Patch,
        arc_middle: Vec3::ZERO, // unread for Patch
    };
    let seeds = square_ring_centreline_seeds(
        &directions,
        &tips,
        &boundverts,
        &[false, false, false, true, true],
        vertex,
        std::slice::from_ref(&arc_patch),
    );
    assert_eq!(seeds.len(), 5);
    assert!(
        (seeds[0] - 0.5 * (b1 + b2)).length() < 1.0e-6,
        "patch centreline 0 is mid(B_1, B_2)",
    );
    assert!(
        (seeds[1] - 0.5 * (b2 + b3)).length() < 1.0e-6,
        "patch centreline 1 is mid(B_2, B_3)",
    );
}

/// BEVEL.md §8.5.5.5(f.5) / §8.9.2: the **square-profile** outer `ARC` miter. Beveling the
/// three bunched apex legs of an octagonal bipyramid at `profile = 0.97` (square) fires the
/// outer miter over the `270°` reflex gap and fills the enlarged `s = 4` ring with the
/// parallel-offset grid — so the original apex `v = (0,0,1)` survives **un-rounded** at the
/// weld, unlike the circular miter (whose bulge pulls the corner in to `z ≈ 0.73`).
#[test]
fn square_outer_arc_welds_apex_and_differs_from_circular() {
    let build = |profile: f32, miter: BevelMiter| {
        let (mut mesh, _apex, _rim, [va, vb, vc]) = octagonal_bipyramid();
        mesh.bevel(
            &[va.into(), vb.into(), vc.into()],
            BevelOptions {
                offset: 0.2,
                affect: BevelAffect::Edges,
                segments: 2,
                profile,
                vmesh_method: BevelVmeshMethod::Adj,
                miter_outer: miter,
                ..Default::default()
            },
        );
        mesh
    };

    let apex = Vec3::new(0.0, 0.0, 1.0);
    let has_vert_at = |mesh: &LoopMesh, p: Vec3| {
        mesh.vert_keys()
            .into_iter()
            .any(|v| (mesh.vertex_position(v) - p).length() < 1.0e-4)
    };

    let square = build(0.97, BevelMiter::Arc);
    square.validate_topology();
    let (v, e, f) = counts(&square);
    assert_eq!(
        v as i64 - e as i64 + f as i64,
        2,
        "the square outer ARC miter closes to a χ = 2 manifold",
    );
    assert!(
        has_vert_at(&square, apex),
        "the square profile welds the un-rounded apex v at the grid centre (§8.5.5.5(f.5))",
    );

    // The circular ARC rounds the corner in — no vertex survives at the apex.
    let circular = build(0.5, BevelMiter::Arc);
    assert!(
        !has_vert_at(&circular, apex),
        "the circular ARC bulge pulls the corner off the apex",
    );

    // The miter is not inert: the square outer ARC differs from the square sharp ring.
    let sharp = build(0.97, BevelMiter::Sharp);
    assert_ne!(
        counts(&square),
        counts(&sharp),
        "the square outer ARC adds ring geometry the sharp ring lacks",
    );
}

/// BEVEL.md §8.5.5.5(f.5) / §8.9.2.1: the **square-profile** outer `PATCH` miter. The patch
/// promotes the retained meet corner `B_2` to a ring hub and seeds its two centrelines from
/// the boundvert-pair midpoints; the enlarged `s = 5` ring is filled by the parallel-offset
/// grid, welding the apex `v` at the centre, and the reflex faces still close as the four
/// hub triangles + two `(segments + 3)`-gon flanks of §8.9.2.1.
///
/// Swept over **every even `segments`** the dispatch admits, not just `2`. §8.5.5.5(f.5)'s
/// general-ring square grid is stated for "valence `s`, `≥ 3` beveled edges, **even**
/// `segments` only" with `h = ⌊n/2⌋` steps — there is no upper count in the gate — and case
/// 1's two patch seeds `C_i = mid(B_i, B_{i+1})` / `C_{i+1} = mid(B_{i+1}, B_{i+2})` read no
/// `segments` at all. So the whole even family is one construction, and the two `h`-step rim
/// rows `B_i → C_i` / `B_i → C_{i−1}` legitimately **overwrite** the flat chord shoulders the
/// miter placed: on a square ring the rim rows *are* the profile. At `segments = 4` each
/// patch leg's five slots are exactly the two rows meeting at the shared `C`, which is what
/// makes the overwrite total rather than partial.
#[test]
fn square_outer_patch_welds_apex_and_keeps_hub_fan() {
    for segments in [2_u32, 4] {
        let (mut mesh, _apex, rim, [va, vb, vc]) = octagonal_bipyramid();
        mesh.bevel(
            &[va.into(), vb.into(), vc.into()],
            BevelOptions {
                offset: 0.2,
                affect: BevelAffect::Edges,
                segments,
                profile: 0.97,
                vmesh_method: BevelVmeshMethod::Adj,
                miter_outer: BevelMiter::Patch,
                ..Default::default()
            },
        );
        mesh.validate_topology();

        let (v, e, f) = counts(&mesh);
        assert_eq!(
            v as i64 - e as i64 + f as i64,
            2,
            "segments {segments}: the square outer PATCH miter closes to a χ = 2 manifold",
        );

        let apex = Vec3::new(0.0, 0.0, 1.0);
        assert!(
            mesh.vert_keys()
                .into_iter()
                .any(|v| (mesh.vertex_position(v) - apex).length() < 1.0e-4),
            "segments {segments}: the square PATCH welds the un-rounded apex v at the grid \
             centre (§8.5.5.5(f.5))",
        );

        // §8.9.2.1 fan topology is preserved at the square profile: six reflex faces wrapping
        // the absorbed legs, four bare triangles + two `(segments + 3)`-gons. The reflex faces
        // are the apex-side faces touching an absorbed rim vert; exclude the base-pyramid
        // faces (which also touch `R4 … R8`) by dropping any face reaching the base vertex
        // `(0,0,-1)`.
        let is_base =
            |v: VertexKey| (mesh.vertex_position(v) - Vec3::new(0.0, 0.0, -1.0)).length() < 1.0e-4;
        let absorbed = [rim[3], rim[4], rim[5], rim[6], rim[7]];
        let mut sizes: Vec<usize> = Vec::new();
        for face in mesh.face_keys() {
            let verts = mesh.face_vert_keys(face);
            if !verts.iter().any(|x| absorbed.contains(x)) || verts.iter().any(|&x| is_base(x)) {
                continue;
            }
            sizes.push(verts.len());
        }
        sizes.sort_unstable();
        let mut expected = vec![3usize; 4];
        expected.push(segments as usize + 3);
        expected.push(segments as usize + 3);
        expected.sort_unstable();
        assert_eq!(
            sizes, expected,
            "segments {segments}: four interior hub triangles + two `(segments + 3)`-gon \
             flanks wrap the 270° gap (§8.9.2.1)",
        );
    }
}

/// The apex junction's **cap faces** on the `octagonal_bipyramid` driver: the new faces
/// lying wholly above the rim plane. Each of the three profile strips spans a single cell
/// from the cap's ring chain down to its rim vert at `z = 0`, and every re-stitched reflex
/// face reaches one of the absorbed rim verts, so the cut isolates the cap exactly.
fn bipyramid_apex_cap_faces(mesh: &LoopMesh, result: &BevelResult) -> Vec<FaceKey> {
    result
        .faces
        .iter()
        .copied()
        .filter(|&face| {
            mesh.face_vert_keys(face)
                .iter()
                .all(|&vertex| mesh.vertex_position(vertex).z > 0.4)
        })
        .collect()
}

/// The cap disk's **innermost ring** (BEVEL.md §8.5.5.5(e)): peel inward from the cap's
/// boundary chain — the verts it shares with the profile strips — and keep the last ring
/// the peel reaches. At odd `segments` that ring is the central `(ring valence)`-gon's seam
/// verts, so the closing polygon is identified **topologically**, without reading any
/// position (which §8.5.5.5(f.5) warns can corroborate a construction but not attribute
/// one). At even parity the same peel lands on the single welded apex.
fn cap_innermost_ring(mesh: &LoopMesh, cap: &[FaceKey]) -> Vec<VertexKey> {
    let mut cap_verts: Vec<VertexKey> = Vec::new();
    for &face in cap {
        for vertex in mesh.face_vert_keys(face) {
            if !cap_verts.contains(&vertex) {
                cap_verts.push(vertex);
            }
        }
    }
    let mut ring: Vec<VertexKey> = cap_verts
        .iter()
        .copied()
        .filter(|&vertex| {
            !mesh
                .vert_face_keys(vertex)
                .iter()
                .all(|face| cap.contains(face))
        })
        .collect();
    let mut peeled = ring.clone();
    loop {
        let mut next: Vec<VertexKey> = Vec::new();
        for &vertex in &ring {
            for neighbour in mesh.vert_neighbor_keys(vertex) {
                if cap_verts.contains(&neighbour)
                    && !peeled.contains(&neighbour)
                    && !next.contains(&neighbour)
                {
                    next.push(neighbour);
                }
            }
        }
        if next.is_empty() {
            return ring;
        }
        peeled.extend(next.iter().copied());
        ring = next;
    }
}

/// BEVEL.md §8.5.5.5(f.1) step 1 / §8.5.5.5(f.5) / §8.9.5.3: the outer miter **is admitted**
/// at the square sentinel with **odd** `segments`, and what it then builds is step 4's
/// general fullness grid — not the (f.5) parallel-offset lattice.
///
/// Step 1's claim carries an even-`segments` conjunct, and (f.5) says the same from its own
/// side: its lattice "is not reached on odd `segments` at all". So at an odd count a square
/// ring never reaches that lattice, and there is nothing for the miter's admission to
/// decline — the enlarged ring falls through to the same grid the circular exponent's miter
/// already rides, closing on §8.5.5.5(e)'s central `(s + 1)`-gon.
///
/// The pins, in the order they narrow the claim:
///
/// - the miter **fires** — the square ring is not the sharp ring. This is the regression the
///   test exists for: an admission gate carrying step 1's parity conjunct returned the sharp
///   ring verbatim at every odd count;
/// - `PATCH`'s `+2` boundverts cost exactly twice `ARC`'s `+1` over the shared sharp baseline
///   (§8.9.6), so neither mode half-fell-back;
/// - the square cap's element census equals the **circular** cap's while its positions do
///   not: at an odd count the exponent is a curve shape only, exactly as (f.5)'s witness box
///   reports ("the vertex totals are identical across all three profiles");
/// - the cap is **built** as §8.5.5.5(e)'s odd-parity grid: `valence` sub-patch blocks of
///   `h²` quads + `valence` seam bridges of `h` quads + **one** central `valence`-gon, with
///   `h = (segments − 1) / 2` — the `h = 1`, valence-4 instance of which is §8.9.5.3's worked
///   `4 + 4 + 1 = 9`-face cap. The closing polygon is found by peeling the cap disk inward,
///   *not* by reading a position: at odd `segments` **neither** (f.5) construction welds, so
///   "no vert at `v`" excludes nothing that odd parity does not already exclude, and (f.5)'s
///   reachability box warns that an apex footprint corroborates a construction rather than
///   attributing one. That test is kept below as corroboration, no longer as the attribution;
/// - `PATCH`'s §8.9.2.1 re-stitch survives the widening: the absorbed-leg fan still closes
///   as four hub triangles + two `(segments + 3)`-gons, the same shape
///   `square_outer_patch_welds_apex_and_keeps_hub_fan` pins at even counts.
///
/// Swept across the whole sentinel band (§8.6.2), because it is the snap — not one slider
/// value — that used to trip the gate.
#[test]
fn square_outer_miter_admits_odd_segments_on_the_step4_grid() {
    let build = |miter: BevelMiter, profile: f32, segments: u32| {
        let (mut mesh, _apex, rim, [va, vb, vc]) = octagonal_bipyramid();
        let result = mesh.bevel(
            &[va.into(), vb.into(), vc.into()],
            BevelOptions {
                offset: 0.2,
                affect: BevelAffect::Edges,
                segments,
                profile,
                vmesh_method: BevelVmeshMethod::Adj,
                miter_outer: miter,
                ..Default::default()
            },
        );
        (mesh, result, rim)
    };
    let apex = Vec3::new(0.0, 0.0, 1.0);

    // §8.9.6: the enlarged ring's totals are not a closed form, so the two odd counts are
    // pinned against the reference figures — `(ARC, PATCH)` per `segments`.
    for (segments, expected) in [
        (3_u32, ((37_usize, 78_usize, 43_usize), (41, 84, 45))),
        (5, ((63, 132, 71), (72, 147, 77))),
    ] {
        let (sharp_mesh, _, _) = build(BevelMiter::Sharp, 0.97, segments);
        let sharp = counts(&sharp_mesh);
        let (circular_mesh, circular_result, _) = build(BevelMiter::Arc, 0.5, segments);
        let circular = counts(&circular_mesh);
        let circular_positions: Vec<Vec3> = circular_result
            .verts
            .iter()
            .map(|&vertex| circular_mesh.vertex_position(vertex))
            .collect();

        for profile in [0.95_f32, 0.97, 0.99, 1.0] {
            assert_eq!(
                profile_to_exponent(profile),
                PROFILE_SQUARE_EXPONENT,
                "profile {profile} snaps to the square sentinel (§8.6.2)",
            );
            let (arc_mesh, arc_result, _) = build(BevelMiter::Arc, profile, segments);
            let (patch_mesh, patch_result, patch_rim) = build(BevelMiter::Patch, profile, segments);
            arc_mesh.validate_topology();
            patch_mesh.validate_topology();
            let arc = counts(&arc_mesh);
            let patch = counts(&patch_mesh);
            assert_eq!(
                (arc, patch),
                expected,
                "segments {segments}, profile {profile}: the odd square miter's element \
                 totals (§8.9.6)",
            );

            // §8.9.2 / §8.9.6: `ARC` adds one boundvert to the `s = 3` sharp ring and
            // `PATCH` two, so the fill sees a valence-4 and a valence-5 ring respectively —
            // and §8.5.5.5(e) closes each on a central polygon of exactly that valence.
            for (name, mesh, result, census, valence) in [
                ("ARC", &arc_mesh, &arc_result, arc, 4_usize),
                ("PATCH", &patch_mesh, &patch_result, patch, 5),
            ] {
                assert_eq!(
                    census.0 as i64 - census.1 as i64 + census.2 as i64,
                    2,
                    "segments {segments}, profile {profile}: the square outer {name} miter \
                     closes to a χ = 2 manifold",
                );
                assert_ne!(
                    census, sharp,
                    "segments {segments}, profile {profile}: the square outer {name} miter \
                     must not decline to the sharp ring",
                );
                assert!(
                    !mesh
                        .vert_keys()
                        .into_iter()
                        .any(|v| (mesh.vertex_position(v) - apex).length() < 1.0e-4),
                    "segments {segments}, profile {profile}: odd parity closes on \
                     §8.5.5.5(e)'s central polygon, so (f.5)'s even-only weld at `v` is \
                     absent ({name})",
                );

                let cap = bipyramid_apex_cap_faces(mesh, result);
                let steps = (segments as usize - 1) / 2;
                assert_eq!(
                    cap.len(),
                    valence * steps * steps + valence * steps + 1,
                    "segments {segments}, profile {profile}: the {name} cap is §8.5.5.5(e)'s \
                     odd grid — {valence} sub-patch blocks of {steps}² + {valence} seam \
                     bridges of {steps} + the central {valence}-gon",
                );
                let innermost = cap_innermost_ring(mesh, &cap);
                assert_eq!(
                    innermost.len(),
                    valence,
                    "segments {segments}, profile {profile}: the {name} cap's innermost ring \
                     is the {valence} seam verts of §8.5.5.5(e)",
                );
                let closing: Vec<FaceKey> = cap
                    .iter()
                    .copied()
                    .filter(|&face| {
                        mesh.face_vert_keys(face)
                            .iter()
                            .all(|vertex| innermost.contains(vertex))
                    })
                    .collect();
                assert_eq!(
                    closing.len(),
                    1,
                    "segments {segments}, profile {profile}: odd parity closes the {name} cap \
                     on exactly one central face",
                );
                assert_eq!(
                    mesh.face_vert_keys(closing[0]).len(),
                    valence,
                    "segments {segments}, profile {profile}: that central face is the \
                     `(ring valence)`-gon of §8.5.5.5(e) ({name})",
                );
            }

            // §8.9.2.1: the odd `PATCH` re-stitch is untouched by the widening — the five
            // absorbed legs still fan to the hub `B_2` as four bare triangles flanked by two
            // `(segments + 3)`-gons. Same isolation as the even-count sibling test: the
            // apex-side faces touching an absorbed rim vert, dropping the base pyramid's.
            let is_base = |v: VertexKey| {
                (patch_mesh.vertex_position(v) - Vec3::new(0.0, 0.0, -1.0)).length() < 1.0e-4
            };
            let absorbed = [
                patch_rim[3],
                patch_rim[4],
                patch_rim[5],
                patch_rim[6],
                patch_rim[7],
            ];
            let mut sizes: Vec<usize> = Vec::new();
            for face in patch_mesh.face_keys() {
                let verts = patch_mesh.face_vert_keys(face);
                if !verts.iter().any(|x| absorbed.contains(x)) || verts.iter().any(|&x| is_base(x))
                {
                    continue;
                }
                sizes.push(verts.len());
            }
            sizes.sort_unstable();
            let mut expected_sizes = vec![3_usize; 4];
            expected_sizes.push(segments as usize + 3);
            expected_sizes.push(segments as usize + 3);
            expected_sizes.sort_unstable();
            assert_eq!(
                sizes, expected_sizes,
                "segments {segments}, profile {profile}: four hub triangles + two \
                 `(segments + 3)`-gons still wrap the 270° gap (§8.9.2.1)",
            );

            assert_eq!(
                (patch.0 - sharp.0, patch.1 - sharp.1, patch.2 - sharp.2),
                (
                    2 * (arc.0 - sharp.0),
                    2 * (arc.1 - sharp.1),
                    2 * (arc.2 - sharp.2)
                ),
                "segments {segments}, profile {profile}: PATCH's `+2` boundverts cost twice \
                 ARC's `+1` (§8.9.6)",
            );
            assert_eq!(
                arc, circular,
                "segments {segments}, profile {profile}: at odd `segments` the square ring \
                 rides the same step-4 grid the circle does, so the census is unchanged",
            );
            assert!(
                arc_result.verts.iter().any(|&vertex| {
                    let position = arc_mesh.vertex_position(vertex);
                    !circular_positions
                        .iter()
                        .any(|candidate| (*candidate - position).length() <= 1.0e-4)
                }),
                "segments {segments}, profile {profile}: the exponent still shapes the ring \
                 chains, so the square cap is not the circular one",
            );
        }
    }
}

/// BEVEL.md §8.5.5.5(f.5): the convex sector **trailing** a fired outer miter keeps its
/// pass-1 parallel-offset centreline seed. On the octagonal bipyramid arc/patch bevel the
/// reflex sector opens over the `270°` gap and the miter's arc/patch end `B_3` lands on the
/// beveled edge to `R1` (the `y = 0` symmetry edge). That sector's trailing bounding
/// boundvert is an ordinary sharp meet-point, so the sector's centreline is seeded by its
/// parallel-offset foot — the same `(0.271, 0, 0.729)` the sharp ring produces — and the
/// `3`-fold-symmetric convex inner ring is intact. It must **not** collapse to the pass-2
/// `B_3`-projection `(0.503, 0, 0.497)`, which is what over-suppressing the trailing sharp
/// proposal (reserving the whole sector for pass 2) would place there.
#[test]
fn square_outer_miter_trailing_convex_sector_keeps_pass_one_seed() {
    let build = |miter: BevelMiter| {
        let (mut mesh, _apex, _rim, [va, vb, vc]) = octagonal_bipyramid();
        mesh.bevel(
            &[va.into(), vb.into(), vc.into()],
            BevelOptions {
                offset: 0.2,
                affect: BevelAffect::Edges,
                segments: 2,
                profile: 0.97,
                vmesh_method: BevelVmeshMethod::Adj,
                miter_outer: miter,
                ..Default::default()
            },
        );
        mesh
    };
    let has_vert_at = |mesh: &LoopMesh, p: Vec3| {
        mesh.vert_keys()
            .into_iter()
            .any(|v| (mesh.vertex_position(v) - p).length() < 1.0e-3)
    };

    // The three convex inner-ring centreline seeds, one per beveled edge — a mirror-
    // symmetric set about the `x = y` plane through the middle beveled edge (`R2` / 45°).
    let on_r1 = Vec3::new(0.2714, 0.0, 0.7286); // y = 0 edge, the regressed one
    let on_r2 = Vec3::new(0.0, 0.2714, 0.7286); // its mirror image across x = y
    let on_r3 = Vec3::new(0.1919, 0.1919, 0.7286); // the symmetry-axis edge
    let over_suppressed = Vec3::new(0.5031, 0.0, 0.4969); // the pass-2 B_3 projection

    for miter in [BevelMiter::Arc, BevelMiter::Patch] {
        let mesh = build(miter);
        mesh.validate_topology();
        assert!(
            has_vert_at(&mesh, on_r1),
            "{miter:?}: the y=0 trailing convex sector keeps its pass-1 seed",
        );
        assert!(
            has_vert_at(&mesh, on_r2) && has_vert_at(&mesh, on_r3),
            "{miter:?}: the other two convex centreline seeds are intact",
        );
        assert!(
            !has_vert_at(&mesh, over_suppressed),
            "{miter:?}: no vertex at the over-suppressed pass-2 B_3 projection",
        );
    }
}

/// BEVEL.md §8.9.2.2 worked trace: the multi-face reflex outer-`ARC` re-stitch at
/// `segments = 2`. Beveling the three bunched apex legs `A–R1 … A–R3` of the octagonal
/// bipyramid leaves the `270°` reflex sector absorbing the five unbeveled legs
/// `A–R4 … A–R8` (an **odd** count). `ARC` materialises the three arc samples
/// `s_0 = B1`, `s_1 = bulge`, `s_2 = B3` — with **no** `B2` hub — and **tiles** the six
/// reflex faces along that sample run: the odd middle leg `A–R6` pins the interior
/// sample `s_1`, so the two faces on either side of it (`[A,R5,R6]`, `[A,R6,R7]`) grow
/// to quads that straddle `s_1`, while the two ends and the two flanks stay triangles.
/// No reflex face fans to a single hub, and no vertex remains at the raw meet corner.
#[test]
fn multiface_arc_tiles_reflex_hole_run() {
    let (s0, s1, s2) = octagonal_arc_samples(&[0, 1, 2]);
    // The raw §7.6.2 meet corner PATCH would promote to the hub B2 — ARC keeps it only
    // as a control point and mints no vertex there.
    let (_, meet, _) = octagonal_patch_boundvert_positions();

    let (mut mesh, _apex, rim, [va, vb, vc]) = octagonal_bipyramid();
    let result = mesh.bevel(
        &[va.into(), vb.into(), vc.into()],
        BevelOptions {
            offset: 0.2,
            affect: BevelAffect::Edges,
            segments: 2,
            vmesh_method: BevelVmeshMethod::Adj,
            miter_outer: BevelMiter::Arc,
            ..Default::default()
        },
    );
    mesh.validate_topology();

    let (v, e, f) = counts(&mesh);
    assert_eq!(
        v as i64 - e as i64 + f as i64,
        2,
        "ARC miter closes to a χ = 2 manifold"
    );

    let find = |p: Vec3| {
        result
            .verts
            .iter()
            .copied()
            .find(|&x| (mesh.vertex_position(x) - p).length() < 1.0e-4)
    };
    let sample = [
        find(s0).expect("s_0 = B1 minted"),
        find(s1).expect("s_1 interior bulge minted (always materialised, §8.9.2.2)"),
        find(s2).expect("s_2 = B3 minted"),
    ];
    assert_eq!(
        result_has_vert_near(&mesh, &result, meet),
        0,
        "ARC keeps the meet corner only as a control point — no ring hub B2 (§8.9.2.2)",
    );

    // Classify every reflex face (one touching an absorbed rim vert R4 … R8) by the
    // arc samples it threads and its size. §8.9.2.2 pins the exact multiset: two
    // `{s_0}` triangles, one `{s_0,s_1}` quad, one `{s_1,s_2}` quad, two `{s_2}`
    // triangles — the run-tiling signature (no hub, the odd middle leg splits the run
    // at s_1).
    let absorbed = [rim[3], rim[4], rim[5], rim[6], rim[7]];
    let mut observed: Vec<(Vec<usize>, usize)> = Vec::new();
    for face in mesh.face_keys() {
        let face_verts = mesh.face_vert_keys(face);
        if !face_verts.iter().any(|x| absorbed.contains(x)) {
            continue;
        }
        let threaded: Vec<usize> = sample
            .iter()
            .enumerate()
            .filter(|(_, &s)| face_verts.contains(&s))
            .map(|(i, _)| i)
            .collect();
        // Base-side pyramid faces touch the absorbed rims too but thread no arc
        // sample and are not re-stitched; only reflex miter faces reach the samples.
        if threaded.is_empty() {
            continue;
        }
        observed.push((threaded, face_verts.len()));
    }
    observed.sort();
    let mut expected = vec![
        (vec![0], 3),
        (vec![0], 3),
        (vec![0, 1], 4),
        (vec![1, 2], 4),
        (vec![2], 3),
        (vec![2], 3),
    ];
    expected.sort();
    assert_eq!(
        observed, expected,
        "six reflex faces tile the arc run: 4 triangles + 2 straddle quads (§8.9.2.2)",
    );

    // The interior sample s_1 is threaded by exactly the two straddle quads — one per
    // side of the odd middle leg — never fanned to a hub.
    let s1_faces = mesh
        .face_keys()
        .into_iter()
        .filter(|&face| mesh.face_vert_keys(face).contains(&sample[1]))
        .count();
    assert!(
        s1_faces >= 2,
        "the interior bulge s_1 is a real arc vertex the straddle quads thread",
    );
}

/// BEVEL.md §8.9.2.2 even-leg cross-check: beveling **four** apex legs `A–R1 … A–R4`
/// of the octagonal bipyramid leaves the `225°` reflex sector absorbing **four**
/// unbeveled legs `A–R5 … A–R8` (an **even** count), assigned `A–R5→0, A–R6→0,
/// A–R7→2, A–R8→2` — no leg on the interior sample `s_1`. This is the case the
/// withdrawn "arc-end-only" reading claimed adds no interior vertex; the pinned rule
/// **does** materialise `s_1` and threads the single wide face `[A,R6,R7]` through it
/// as a **pentagon** `[B_3, s_1, B_1, R6, R7]`, while the other reflex faces stay
/// triangles. So the even-leg closure is one pentagon plus four triangles — not
/// arc-end-only.
#[test]
fn multiface_arc_even_legs_thread_pentagon() {
    let (s0, s1, s2) = octagonal_arc_samples(&[0, 1, 2, 3]);

    // §8.9.2 / §8.9.2.2: the arc ends inherit the reflex sector's meet-point, which
    // for this **even**-absorbed-leg sector is the two flanking edges' offset-line
    // crossing **dropped onto the straddle face** `[A,R6,R7]` (the §7.6.2 face-drop
    // fires on the case-3 crossing, not just a skew midpoint). The neighbour-plane
    // reposition then lands `B1` / `B3` here, and the §8.9.5 bulge pulls `s_1` in —
    // matching the cross-backend oracle. (The odd-leg sector's crossing bisector
    // lands on a rim vertex, so its drop rejects every face and the meet stays
    // lifted; only the even sector drops. These pins are the discriminator.)
    let oracle = [
        Vec3::new(-0.36871, 0.24371, 0.42679),
        Vec3::new(0.00368, 0.00888, 0.62133),
        Vec3::new(0.43305, -0.08839, 0.42679),
    ];
    for (got, want) in [s0, s1, s2].iter().zip(oracle.iter()) {
        assert!(
            (*got - *want).length() < 1.0e-4,
            "even-leg arc sample {got:?} matches oracle {want:?} (§8.9.2.2 reflex drop)",
        );
    }

    let (mut mesh, _apex, rim, apex_leg) = octagonal_bipyramid_all_legs();
    let beveled: Vec<GeomKey> = (0..4).map(|k| apex_leg[k].into()).collect();
    let result = mesh.bevel(
        &beveled,
        BevelOptions {
            offset: 0.2,
            affect: BevelAffect::Edges,
            segments: 2,
            vmesh_method: BevelVmeshMethod::Adj,
            miter_outer: BevelMiter::Arc,
            ..Default::default()
        },
    );
    mesh.validate_topology();

    let (v, e, f) = counts(&mesh);
    assert_eq!(
        v as i64 - e as i64 + f as i64,
        2,
        "ARC miter closes to a χ = 2 manifold"
    );

    let find = |p: Vec3| {
        result
            .verts
            .iter()
            .copied()
            .find(|&x| (mesh.vertex_position(x) - p).length() < 1.0e-4)
    };
    let sample = [
        find(s0).expect("s_0 = B1 minted"),
        find(s1).expect("s_1 interior bulge minted even with no leg on it (§8.9.2.2)"),
        find(s2).expect("s_2 = B3 minted"),
    ];

    // The absorbed rims are R5 … R8; classify reflex faces by threaded samples + size.
    let absorbed = [rim[4], rim[5], rim[6], rim[7]];
    let mut observed: Vec<(Vec<usize>, usize)> = Vec::new();
    for face in mesh.face_keys() {
        let face_verts = mesh.face_vert_keys(face);
        if !face_verts.iter().any(|x| absorbed.contains(x)) {
            continue;
        }
        let threaded: Vec<usize> = sample
            .iter()
            .enumerate()
            .filter(|(_, &s)| face_verts.contains(&s))
            .map(|(i, _)| i)
            .collect();
        // Base-side pyramid faces touch the absorbed rims too but thread no arc
        // sample and are not re-stitched; only reflex miter faces reach the samples.
        if threaded.is_empty() {
            continue;
        }
        observed.push((threaded, face_verts.len()));
    }
    observed.sort();
    // Faces: [A,R5,R6]→{s_0} tri, [A,R6,R7]→{s_0,s_1,s_2} pent (threads un-pinned s_1),
    // [A,R7,R8]→{s_2} tri, plus the two flank triangles [A,R4,R5]→{s_0}, [A,R8,R1]→{s_2}.
    let mut expected = vec![
        (vec![0], 3),
        (vec![0], 3),
        (vec![0, 1, 2], 5),
        (vec![2], 3),
        (vec![2], 3),
    ];
    expected.sort();
    assert_eq!(
        observed, expected,
        "even-leg ARC: one wide pentagon threads s_1 + four triangles (§8.9.2.2)",
    );
}

/// Re-derive a reflex sector's `segments + 1` **emitted** `ARC` sample positions on the
/// pristine octagonal bipyramid, generalising `octagonal_arc_samples` to any `segments`
/// (BEVEL.md §8.9.5.1): the two arc ends `s_0 = B1` / `s_segments = B3` (the §8.9.2
/// reposition), the raw interior samples `s_k = P(cos θ_k, sin θ_k)`,
/// `θ_k = 90°·(1 − k / segments)` on the inscribed circular profile, then the
/// §8.2.2.1.1 chord-fraction resample of that chain against the profile materialised
/// at `N = next power of two ≥ max(4, segments)` — the arc is a [V-ADJ] cap ring chain
/// like any other. Returns the ordered chain `s_0 … s_segments`.
fn octagonal_arc_chain(beveled: &[usize], segments: u32) -> Vec<Vec3> {
    let offset = 0.2;
    let (mesh, apex, _rim, apex_leg) = octagonal_bipyramid_all_legs();
    let beveled_edges: Vec<EdgeKey> = beveled.iter().map(|&k| apex_leg[k]).collect();
    let is_beveled = |edge: EdgeKey| beveled_edges.contains(&edge);
    let apex_position = mesh.vertex_position(apex);
    let corners = mesh.vertex_bevel_corner_ring(apex, &is_beveled);
    let sectors = sector_split(&corners, &is_beveled);
    let sector_count = sectors.len();
    let mut meets: Vec<Vec3> = Vec::new();
    let mut dir_open: Vec<Vec3> = Vec::new();
    let mut dir_close: Vec<Vec3> = Vec::new();
    let mut reflex: Vec<bool> = Vec::new();
    for sector in &sectors {
        let opening = sector.bounding_edges[0];
        let closing = sector.bounding_edges[1];
        let face = sector.corners[0].1;
        let d_open = mesh
            .edge_direction_from(opening, apex, apex_position)
            .unwrap();
        let d_close = mesh
            .edge_direction_from(closing, apex, apex_position)
            .unwrap();
        let cross = (-d_open).cross(-d_close);
        reflex.push(cross.dot(mesh.face_plane_normal(face)) < 0.0);
        meets.push(mesh.solve_meet_point(apex, apex_position, sector, offset));
        dir_open.push(d_open);
        dir_close.push(d_close);
    }
    let r = reflex.iter().position(|&x| x).unwrap();
    let midpoint = meets[r];
    let predecessor = meets[(r + sector_count - 1) % sector_count];
    let successor = meets[(r + 1) % sector_count];
    let b1 = midpoint + (predecessor - midpoint).dot(dir_open[r]) * dir_open[r];
    let b3 = midpoint + (successor - midpoint).dot(dir_close[r]) * dir_close[r];
    let a_axis = midpoint - b1;
    let b_axis = midpoint - b3;
    let inner = b1 + b3 - midpoint;
    let polyline = |count: u32| -> Vec<Vec3> {
        (0..=count)
            .map(|k| {
                if k == 0 {
                    b1
                } else if k == count {
                    b3
                } else {
                    let theta = std::f32::consts::FRAC_PI_2 * (1.0 - k as f32 / count as f32);
                    a_axis * theta.cos() + b_axis * theta.sin() + inner
                }
            })
            .collect()
    };
    let raw = polyline(segments);
    let source_segments = boundary_source_resolution(segments);
    if source_segments == segments {
        return raw;
    }
    chord_fraction_resample(&raw, &polyline(source_segments))
}

/// Classify a beveled octagonal-bipyramid result's reflex-hole faces (those touching
/// an absorbed rim vert and threading at least one arc sample) by the sorted list of
/// arc-sample indices each threads and its vertex count. This is the run-tiling
/// signature the §8.9.2.2 / §8.9.2.3 face rebuild pins.
fn classify_reflex_arc_faces(
    mesh: &LoopMesh,
    result: &BevelResult,
    absorbed: &[VertexKey],
    chain: &[Vec3],
) -> Vec<(Vec<usize>, usize)> {
    let sample: Vec<VertexKey> = chain
        .iter()
        .map(|&p| {
            result
                .verts
                .iter()
                .copied()
                .find(|&x| (mesh.vertex_position(x) - p).length() < 1.0e-4)
                .unwrap_or_else(|| panic!("arc sample {p:?} minted"))
        })
        .collect();
    let mut observed: Vec<(Vec<usize>, usize)> = Vec::new();
    for face in mesh.face_keys() {
        let face_verts = mesh.face_vert_keys(face);
        if !face_verts.iter().any(|x| absorbed.contains(x)) {
            continue;
        }
        let threaded: Vec<usize> = sample
            .iter()
            .enumerate()
            .filter(|(_, &s)| face_verts.contains(&s))
            .map(|(i, _)| i)
            .collect();
        if threaded.is_empty() {
            continue;
        }
        observed.push((threaded, face_verts.len()));
    }
    observed.sort();
    observed
}

/// BEVEL.md §8.9.2.3 / §8.9.5.1 worked trace: the multi-face reflex outer-`ARC`
/// re-stitch at **odd `segments = 3`**. Beveling the three bunched apex legs
/// `A–R1 … A–R3` of the octagonal bipyramid leaves the `270°` reflex sector absorbing
/// the five legs `A–R4 … A–R8` (odd). The arc now carries **four** samples `s_0 … s_3`
/// (no on-bisector midpoint sample); the odd middle leg `A–R6` pins the **floor** index
/// `3/2 = 1`, splitting the run asymmetrically: the leading step face is a quad
/// (run `0→1`) and the trailing a pentagon (run `1→3`). Result **V/E/F = 37/78/43**,
/// matching the guide's oracle read-back.
#[test]
fn multiface_arc_odd_segments3_tiles_run() {
    let chain = octagonal_arc_chain(&[0, 1, 2], 3);
    let (mut mesh, _apex, rim, [va, vb, vc]) = octagonal_bipyramid();
    let result = mesh.bevel(
        &[va.into(), vb.into(), vc.into()],
        BevelOptions {
            offset: 0.2,
            affect: BevelAffect::Edges,
            segments: 3,
            vmesh_method: BevelVmeshMethod::Adj,
            miter_outer: BevelMiter::Arc,
            ..Default::default()
        },
    );
    mesh.validate_topology();

    let (v, e, f) = counts(&mesh);
    assert_eq!(
        v as i64 - e as i64 + f as i64,
        2,
        "ARC miter closes to a χ = 2 manifold"
    );
    assert_eq!(
        (v, e, f),
        (37, 78, 43),
        "odd segments = 3 octagonal ARC matches the §8.9.2.3 oracle V/E/F",
    );

    let absorbed = [rim[3], rim[4], rim[5], rim[6], rim[7]];
    let observed = classify_reflex_arc_faces(&mesh, &result, &absorbed, &chain);
    let mut expected = vec![
        (vec![0], 3),       // leading flank [B1, T3, R4]
        (vec![0], 3),       // [B1, R4, R5]              run 0→0
        (vec![0, 1], 4),    // [s1, B1, R5, R6]          run 0→1 (leading step, quad)
        (vec![1, 2, 3], 5), // [B3, s2, s1, R6, R7]      run 1→3 (trailing step, pentagon)
        (vec![3], 3),       // [B3, R7, R8]              run 3→3
        (vec![3], 3),       // trailing flank [B3, R8, T1]
    ];
    expected.sort();
    assert_eq!(
        observed, expected,
        "odd segments = 3 ARC tiles the arc run with a floor-index asymmetric split (§8.9.2.3)",
    );
}

/// BEVEL.md §8.9.5.1 / §8.9.5.3 power-of-two cross-check: the multi-face reflex
/// outer-`ARC` re-stitch at **even `segments = 4`**. Same three-leg octagonal fan;
/// the arc carries **five** samples `s_0 … s_4` placed exactly at the even-angle
/// circle points (no non-power-of-two resample). The odd five-leg assignment
/// `R4→0, R5→0, R6→2, R7→4, R8→4` makes the two step faces pentagons threading their
/// un-pinned intermediate samples (`[s2,s1,B1,R5,R6]` run 0→2, `[B3,s3,s2,R6,R7]` run
/// 2→4). The `(s + 1) = 4`-boundvert cap is the even valence-4 grid — a welded
/// extraordinary-centre apex, **no** central face (§8.9.5.3).
#[test]
fn multiface_arc_even_segments4_tiles_run() {
    let chain = octagonal_arc_chain(&[0, 1, 2], 4);
    let (mut mesh, _apex, rim, [va, vb, vc]) = octagonal_bipyramid();
    let result = mesh.bevel(
        &[va.into(), vb.into(), vc.into()],
        BevelOptions {
            offset: 0.2,
            affect: BevelAffect::Edges,
            segments: 4,
            vmesh_method: BevelVmeshMethod::Adj,
            miter_outer: BevelMiter::Arc,
            ..Default::default()
        },
    );
    mesh.validate_topology();

    let (v, e, f) = counts(&mesh);
    assert_eq!(
        v as i64 - e as i64 + f as i64,
        2,
        "ARC miter closes to a χ = 2 manifold"
    );

    let absorbed = [rim[3], rim[4], rim[5], rim[6], rim[7]];
    let observed = classify_reflex_arc_faces(&mesh, &result, &absorbed, &chain);
    let mut expected = vec![
        (vec![0], 3),       // leading flank [B1, T3, R4]
        (vec![0], 3),       // [B1, R4, R5]                run 0→0
        (vec![0, 1, 2], 5), // [s2, s1, B1, R5, R6]        run 0→2 (leading step, pentagon)
        (vec![2, 3, 4], 5), // [B3, s3, s2, R6, R7]        run 2→4 (trailing step, pentagon)
        (vec![4], 3),       // [B3, R7, R8]                run 4→4
        (vec![4], 3),       // trailing flank [B3, R8, T1]
    ];
    expected.sort();
    assert_eq!(
        observed, expected,
        "even segments = 4 ARC tiles the arc run with two power-of-two-exact step pentagons \
         (§8.9.2.2 / §8.9.5.1)",
    );

    // §8.9.5.3 even parity: the cap welds one shared apex, so on the reflex-hole side
    // exactly one interior arc sample (the on-bisector 45° bulge s_2) is threaded by
    // both step faces — the single shared sample of the two runs.
    let mid_sample = result
        .verts
        .iter()
        .copied()
        .find(|&x| (mesh.vertex_position(x) - chain[2]).length() < 1.0e-4)
        .expect("on-bisector bulge s_2 minted");
    let absorbed_threads = mesh
        .face_keys()
        .into_iter()
        .filter(|&face| {
            let fv = mesh.face_vert_keys(face);
            fv.contains(&mid_sample) && fv.iter().any(|x| absorbed.contains(x))
        })
        .count();
    assert_eq!(
        absorbed_threads, 2,
        "the shared middle sample s_2 is threaded by exactly the two step faces (§8.9.2.2)",
    );
}

/// BEVEL.md §8.2.2.1.1: on the skewed arc-flanking cross-section cell (the reflex
/// sector's leading beveled edge `A–R4` of the octagonal-bipyramid ARC fixture,
/// `offset = 0.2`, `segments = 2`), the single interior sample is the chord-fraction
/// resample of the doubled (4-segment) profile, not the naive 45° cell point. Pins
/// the guide's oracle-verified worked trace.
#[test]
fn segment2_flanking_resample_matches_worked_trace() {
    // Leading flanking beveled edge `A–R4`: apex `A`, rim `R4` (BEVEL.md §8.2.2.1.1).
    let apex = Vec3::new(0.0, 0.0, 1.0);
    let rim = Vec3::new(-0.70711, 0.70711, 0.0);
    let edge_dir = (rim - apex).normalize();

    let start = Vec3::new(-0.19194, 0.46338, 0.45712); // shoulder meet of A–R3 / A–R4
    let end = Vec3::new(-0.36871, 0.24371, 0.42679); // inward-repositioned B_1

    let cell = ProfileCell::new(start, end, apex, edge_dir);

    // The on-edge foot: unequal legs (0.200 vs 0.133) — a genuinely skewed cell.
    assert!(
        (cell.mid - Vec3::new(-0.35577, 0.35577, 0.49687)).length() < 1.0e-4,
        "on-edge foot mid = {:?}",
        cell.mid,
    );
    let leg_start = (cell.mid - start).length();
    let leg_end = (cell.mid - end).length();
    assert!(
        (leg_start - 0.200).abs() < 2.0e-3 && (leg_end - 0.133).abs() < 2.0e-3,
        "unequal-offset legs {leg_start} / {leg_end}",
    );

    // The naive 45° sample (the wrong, symmetric-cell value the resample corrects).
    let naive = cell.sample(
        std::f32::consts::FRAC_PI_4.sin(),
        std::f32::consts::FRAC_PI_4.cos(),
    );
    assert!(
        (naive - Vec3::new(-0.31157, 0.35446, 0.46470)).length() < 1.0e-4,
        "naive 45° sample = {naive:?}",
    );

    // The §8.2.2.1.1 chord-fraction resample lands on BMesh's flanking vertex.
    let resampled = cell.segment2_boundary_resample(0.5, None);
    assert!(
        (resampled - Vec3::new(-0.31188, 0.35399, 0.46459)).length() < 1.0e-4,
        "resampled flanking sample = {resampled:?}",
    );
    // It genuinely moves off the naive point (~0.0005 world-units, ~50 quanta).
    let shift = (resampled - naive).length();
    assert!(
        shift > 3.0e-4 && shift < 1.0e-3,
        "resample shifts the sample ~0.0005 wu off the naive point (got {shift})",
    );
}

/// BEVEL.md §8.2.2.1.1: the resample is the **identity** on a symmetric
/// (equal-offset) cell — `f = 0.5` lands exactly on `Q_2 = P_1` — so applying it
/// unconditionally at `segments = 2` cannot regress the equal-offset strips.
#[test]
fn segment2_resample_is_identity_on_symmetric_cell() {
    // A symmetric cell: `start` / `end` mirror across the z-axis edge, so both cell
    // legs equal `1.0` and the foot lands at `(0, 0, 1)`.
    let start = Vec3::new(1.0, 0.0, 1.0);
    let end = Vec3::new(0.0, 1.0, 1.0);
    let cell = ProfileCell::new(start, end, Vec3::ZERO, Vec3::Z);

    assert!(
        ((cell.mid - start).length() - (cell.mid - end).length()).abs() < 1.0e-6,
        "constructed cell is symmetric",
    );

    let naive = cell.sample(
        std::f32::consts::FRAC_PI_4.sin(),
        std::f32::consts::FRAC_PI_4.cos(),
    );
    let resampled = cell.segment2_boundary_resample(0.5, None);
    assert!(
        (resampled - naive).length() < 1.0e-6,
        "symmetric-cell resample is the identity ({resampled:?} vs {naive:?})",
    );
}

/// BEVEL.md §8.2.2.1.1 / §6.7.5.4: the seg=2 resample fires for every cross-section
/// that bounds a [V-ADJ] adjacency fill (`resample_seg2 = true`) but is inert at a
/// terminal far end (`false`), which has no fill and keeps the naive profile chain.
/// On the skewed arc-flanking cell the two paths must disagree by the resample shift.
#[test]
fn chain_positions_resamples_skewed_cell_only_off_terminal() {
    let apex = Vec3::new(0.0, 0.0, 1.0);
    let rim = Vec3::new(-0.70711, 0.70711, 0.0);
    let edge_dir = (rim - apex).normalize();
    let start = Vec3::new(-0.19194, 0.46338, 0.45712);
    let end = Vec3::new(-0.36871, 0.24371, 0.42679);
    let cell = ProfileCell::new(start, end, apex, edge_dir);

    let filled = cell.chain_positions(2, 0.5, None, ChainEmission::Resampled);
    let terminal = cell.chain_positions(2, 0.5, None, ChainEmission::Raw);

    // Endpoints are identical; only the single interior sample can differ.
    assert_eq!(filled.len(), 3);
    assert_eq!(terminal.len(), 3);
    assert!((filled[0] - terminal[0]).length() < 1.0e-6);
    assert!((filled[2] - terminal[2]).length() < 1.0e-6);

    // The fill path lands on the resampled sample; the terminal path stays naive.
    let coarse = superellipse_chain(0.5, 2);
    let naive = cell.sample(coarse[1].0, coarse[1].1);
    assert!((filled[1] - cell.segment2_boundary_resample(0.5, None)).length() < 1.0e-6);
    assert!(
        (terminal[1] - naive).length() < 1.0e-6,
        "terminal far-end interior stays the naive profile-chain sample",
    );
    // On this skewed cell the two paths genuinely disagree (past the quantum).
    assert!((filled[1] - terminal[1]).length() > 3.0e-4);
}

/// BEVEL.md §8.9.2 / §8.9.5: pin the outer-miter construction against a faithful
/// re-derivation from the pristine mesh. Replicating the §8.9.1 classification and
/// §8.9.2 reposition here, the arc-end boundverts `B1` / `B3` each land level
/// (perpendicular-to-the-edge) with their neighbouring sharp meet-point, and the
/// §8.9.5 arc bulge is pulled *in* from the chord midpoint toward the stored meet
/// corner by exactly `√2 − 1`. All three verts appear in the beveled result.
#[test]
fn outer_miter_arc_boundvert_and_bulge_placement() {
    let offset = 0.2;
    let (mut mesh, apex, [va, vb, vc]) = single_face_reflex_junction();
    let apex_position = mesh.vertex_position(apex);
    let is_beveled = |edge: EdgeKey| edge == va || edge == vb || edge == vc;

    // Replicate the §8.9.1 read-only classification exactly as the operator does.
    let corners = mesh.vertex_bevel_corner_ring(apex, &is_beveled);
    let sectors = sector_split(&corners, &is_beveled);
    let sector_count = sectors.len();
    let mut meets: Vec<Vec3> = Vec::new();
    let mut dir_open: Vec<Vec3> = Vec::new();
    let mut dir_close: Vec<Vec3> = Vec::new();
    let mut reflex: Vec<bool> = Vec::new();
    for sector in &sectors {
        let opening = sector.bounding_edges[0];
        let closing = sector.bounding_edges[1];
        let face = sector.corners[0].1;
        let d_open = mesh
            .edge_direction_from(opening, apex, apex_position)
            .unwrap();
        let d_close = mesh
            .edge_direction_from(closing, apex, apex_position)
            .unwrap();
        let cross = (-d_open).cross(-d_close);
        reflex.push(cross.dot(mesh.face_plane_normal(face)) < 0.0);
        meets.push(mesh.solve_meet_point(apex, apex_position, sector, offset));
        dir_open.push(d_open);
        dir_close.push(d_close);
    }
    assert_eq!(
        reflex.iter().filter(|&&x| x).count(),
        1,
        "the fixture presents exactly one reflex sector",
    );
    let r = reflex.iter().position(|&x| x).unwrap();

    let midpoint = meets[r];
    let predecessor = meets[(r + sector_count - 1) % sector_count];
    let successor = meets[(r + 1) % sector_count];
    let d_open = dir_open[r];
    let d_close = dir_close[r];
    let b1 = midpoint + (predecessor - midpoint).dot(d_open) * d_open;
    let b3 = midpoint + (successor - midpoint).dot(d_close) * d_close;

    // §8.9.2: each arc end ends up perpendicular-to-its-edge (level) with the
    // neighbour, and genuinely slid off the meet-point.
    assert!(
        (b1 - predecessor).dot(d_open).abs() < 1.0e-5,
        "B1 is repositioned level (perpendicular) to the preceding boundvert",
    );
    assert!(
        (b3 - successor).dot(d_close).abs() < 1.0e-5,
        "B3 is repositioned level (perpendicular) to the following boundvert",
    );
    assert!(
        (b1 - midpoint).length() > 1.0e-3 && (b3 - midpoint).length() > 1.0e-3,
        "the reposition genuinely slides the arc ends off the shared meet-point",
    );

    // §8.9.5: the bulge is pulled in from the chord midpoint toward the stored corner.
    let sqrt2 = std::f32::consts::SQRT_2;
    let chord_mid = 0.5 * (b1 + b3);
    let bulge = (2.0 - sqrt2) * chord_mid + (sqrt2 - 1.0) * midpoint;
    assert!(
        (bulge - chord_mid).length() < (midpoint - chord_mid).length(),
        "the bulge is pulled in from the stored corner, not left at it",
    );

    let result = mesh.bevel(
        &[va.into(), vb.into(), vc.into()],
        BevelOptions {
            offset,
            affect: BevelAffect::Edges,
            segments: 2,
            vmesh_method: BevelVmeshMethod::Adj,
            miter_outer: BevelMiter::Arc,
            ..Default::default()
        },
    );
    mesh.validate_topology();

    assert_eq!(
        result_has_vert_near(&mesh, &result, b1),
        1,
        "B1 arc end is minted"
    );
    assert_eq!(
        result_has_vert_near(&mesh, &result, b3),
        1,
        "B3 arc end is minted"
    );
    assert_eq!(
        result_has_vert_near(&mesh, &result, bulge),
        1,
        "the arc bulge sits at the §8.9.5 inscribed-arc sample",
    );
    assert_eq!(
        result_has_vert_near(&mesh, &result, midpoint),
        0,
        "no vertex stays at the raw meet corner (the arc ends slid off it)",
    );
}

/// Decode a world point back into the §8.9.5 unit-square arc frame:
/// `P(u, v) = (midpoint − b1)·u + (midpoint − b3)·v + (b1 + b3 − midpoint)`. The two
/// axes are independent but **not** orthonormal (the reflex cell's included angle is
/// `60°` on the octagonal fan, `70.529°` on the single-face one), so the inverse is the
/// 2×2 Gram solve, not a pair of dot products.
fn arc_frame_uv(point: Vec3, b1: Vec3, b3: Vec3, midpoint: Vec3) -> (f32, f32) {
    let a_axis = midpoint - b1;
    let b_axis = midpoint - b3;
    let offset = point - (b1 + b3 - midpoint);
    let aa = a_axis.dot(a_axis);
    let bb = b_axis.dot(b_axis);
    let ab = a_axis.dot(b_axis);
    let determinant = aa * bb - ab * ab;
    assert!(
        determinant.abs() > 1.0e-9,
        "the arc cell has independent axes"
    );
    let (da, db) = (a_axis.dot(offset), b_axis.dot(offset));
    (
        (bb * da - ab * db) / determinant,
        (aa * db - ab * da) / determinant,
    )
}

/// BEVEL.md §8.9.5's exponent-scope rule: the miter arc is sampled by the **run's own**
/// `profile`-derived §8.6.4 spacing table, not by a forced circle. The circle-valued
/// exponent an arc-start boundvert records never selects a sampler, so the emitted
/// chain moves right across the family.
///
/// Read back in the §8.9.5 unit-square frame on the `70.529°` cell of the single-face
/// reflex junction, the guide publishes all four rows at `segments = 3`; a regression
/// that re-forced the circle would collapse every row onto the `profile = 0.5` one.
#[test]
fn outer_miter_arc_samples_track_the_run_profile_exponent() {
    // BEVEL.md §8.9.5.1 worked trace B's frame (offset 0.2, spread 0, ARC).
    let b1 = Vec3::new(-0.51500, 0.27005, 0.72995);
    let b3 = Vec3::new(0.51500, 0.27005, 0.72995);
    let midpoint = Vec3::new(0.0, -0.24495, 1.24495);

    // profile → the emitted `(u, v)` of `s_1` at `segments = 3`.
    let published = [
        (0.25_f32, (0.33333_f32, 0.66667_f32)), // EXP_LINE — the plain chord B1 → B3
        (0.5, (0.49429, 0.84931)),              // EXP_CIRCLE
        (0.7, (0.58451, 0.93227)),              // curved, and neither neighbour
        (1.0, (0.71010, 1.00000)),              // EXP_SQUARE — on the cell's side
    ];
    for (profile, expected) in published {
        let chain = emitted_arc_chain(b1, b3, midpoint, profile, 3);
        assert_eq!(chain.len(), 4, "segments + 1 samples");
        assert!(
            (chain[0] - b1).length() < 1.0e-6 && (chain[3] - b3).length() < 1.0e-6,
            "the arc ends are the two repositioned boundverts at every exponent",
        );
        let (u, v) = arc_frame_uv(chain[1], b1, b3, midpoint);
        assert!(
            (u - expected.0).abs() < 1.0e-4 && (v - expected.1).abs() < 1.0e-4,
            "profile {profile}: emitted s_1 at ({u}, {v}), guide says {expected:?}",
        );
        // Mirror symmetry holds at every exponent (§8.9.5.1).
        let (mirror_u, mirror_v) = arc_frame_uv(chain[2], b1, b3, midpoint);
        assert!(
            (mirror_u - expected.1).abs() < 1.0e-4 && (mirror_v - expected.0).abs() < 1.0e-4,
            "profile {profile}: s_2 is s_1's u ↔ v mirror",
        );
    }

    // The whole `profile ≥ 0.95` sentinel band is one corner (§8.6.2), so keying on
    // `profile == 1.0` would miss most of it.
    for profile in [0.95_f32, 0.97, 0.99, 1.0] {
        let (u, v) = arc_frame_uv(
            emitted_arc_chain(b1, b3, midpoint, profile, 3)[1],
            b1,
            b3,
            midpoint,
        );
        assert!(
            (u - 0.710098).abs() < 1.0e-4 && (v - 1.0).abs() < 1.0e-4,
            "profile {profile} is inside the square sentinel: got ({u}, {v})",
        );
    }
}

/// BEVEL.md §8.9.5.1's square-sentinel law. At `profile ≥ 0.95` the raw chain runs
/// along the two cell **sides** through the stored corner, the source polyline is that
/// same two-sided path, and the resample degenerates to "walk `B_1 → midpoint → B_3` by
/// arc length". At **even** `segments` a raw sample already sits on the corner and the
/// walk is the identity; at **odd** `segments` the target's straddling chord cuts the
/// corner, so every sample is pushed outward toward it.
///
/// The published `u_1` is **cell-shape dependent** — `0.738797` on the octagonal fan's
/// `60°` cell, `0.710098` on the single-face fan's `70.529°` one — so any per-`segments`
/// lookup table is wrong.
#[test]
fn outer_miter_arc_square_sentinel_walks_the_cell_corner_path() {
    // Trace A — octagonal_bipyramid, included angle 60°.
    let (a_b1, a_b3, a_mid) = (
        Vec3::new(-0.19194, 0.46338, 0.45712),
        Vec3::new(0.46338, -0.19194, 0.45712),
        Vec3::new(-0.19194, -0.19194, 1.11243),
    );
    // Trace B — single_face_reflex_junction, included angle 70.529°.
    let (b_b1, b_b3, b_mid) = (
        Vec3::new(-0.51500, 0.27005, 0.72995),
        Vec3::new(0.51500, 0.27005, 0.72995),
        Vec3::new(0.0, -0.24495, 1.24495),
    );

    let u_of = |chain: &[Vec3], k: usize, b1, b3, mid| arc_frame_uv(chain[k], b1, b3, mid).0;

    let a3 = emitted_arc_chain(a_b1, a_b3, a_mid, 1.0, 3);
    let b3_chain = emitted_arc_chain(b_b1, b_b3, b_mid, 1.0, 3);
    let ua = u_of(&a3, 1, a_b1, a_b3, a_mid);
    let ub = u_of(&b3_chain, 1, b_b1, b_b3, b_mid);
    assert!(
        (ua - 0.738797).abs() < 1.0e-4,
        "60° cell, n = 3: u_1 = {ua}"
    );
    assert!(
        (ub - 0.710098).abs() < 1.0e-4,
        "70.529° cell, n = 3: u_1 = {ub}"
    );
    assert!(
        (ua - ub).abs() > 1.0e-2,
        "the same segments and the same exponent give different samples on the two cells",
    );
    // Both are pushed outward past the raw square sample `b = 0.585786`, and neither
    // reaches the corner.
    for u in [ua, ub] {
        assert!(
            u > 0.585786 && u < 1.0,
            "odd n pushes the sample toward the corner: {u}"
        );
    }

    let a5 = emitted_arc_chain(a_b1, a_b3, a_mid, 1.0, 5);
    let b5 = emitted_arc_chain(b_b1, b_b3, b_mid, 1.0, 5);
    for (u, expected) in [
        (u_of(&a5, 1, a_b1, a_b3, a_mid), 0.424886_f32),
        (u_of(&a5, 2, a_b1, a_b3, a_mid), 0.849772),
        (u_of(&b5, 1, b_b1, b_b3, b_mid), 0.415240),
        (u_of(&b5, 2, b_b1, b_b3, b_mid), 0.830481),
    ] {
        assert!(
            (u - expected).abs() < 1.0e-4,
            "n = 5: got {u}, guide says {expected}"
        );
    }

    // Even `segments`: the resample is the identity, so `s_{n/2}` lands *on* the
    // stored corner — the un-mitered meet-point the arc replaced (§8.9.5.1).
    for segments in [2_u32, 4] {
        let chain = emitted_arc_chain(b_b1, b_b3, b_mid, 1.0, segments);
        assert!(
            (chain[segments as usize / 2] - b_mid).length() < 1.0e-5,
            "square sentinel, segments {segments}: the central sample is the stored corner",
        );
    }
}

/// BEVEL.md §8.9.5.1 worked trace B, end to end through the operator: the single-face
/// reflex junction's outer `ARC` miter at the square sentinel and odd `segments` emits
/// the corner-path chain, not the circular one. The `segments = 3` circular chain sits
/// ≈25 800 quanta away in the max-norm, so this fails loudly if the arc is re-pinned to
/// the circle.
#[test]
fn outer_miter_arc_square_sentinel_emits_the_oracle_chain() {
    for profile in [0.95_f32, 0.97, 0.99, 1.0] {
        let (mut mesh, _apex, [va, vb, vc]) = single_face_reflex_junction();
        let result = mesh.bevel(
            &[va.into(), vb.into(), vc.into()],
            BevelOptions {
                offset: 0.2,
                affect: BevelAffect::Edges,
                segments: 3,
                profile,
                vmesh_method: BevelVmeshMethod::Adj,
                miter_outer: BevelMiter::Arc,
                ..Default::default()
            },
        );
        mesh.validate_topology();

        for sample in [
            Vec3::new(-0.14930, -0.09565, 1.09565),
            Vec3::new(0.14930, -0.09565, 1.09565),
        ] {
            assert_eq!(
                result_has_vert_near(&mesh, &result, sample),
                1,
                "profile {profile}, segments 3: the arc sample {sample:?} is emitted",
            );
        }
        // The circle's own resampled chain at this driver is a different point
        // entirely and must not appear — this is the regression the fix closes.
        let circular = emitted_arc_chain(
            Vec3::new(-0.51500, 0.27005, 0.72995),
            Vec3::new(0.51500, 0.27005, 0.72995),
            Vec3::new(0.0, -0.24495, 1.24495),
            0.5,
            3,
        );
        for sample in &circular[1..3] {
            assert!(
                (*sample - Vec3::new(-0.14930, -0.09565, 1.09565)).length() > 1.0e-2,
                "the two chains are genuinely far apart, so the check below has teeth",
            );
            assert_eq!(
                result_has_vert_near(&mesh, &result, *sample),
                0,
                "profile {profile}: the forced-circle sample {sample:?} must be gone",
            );
        }
    }

    let (mut mesh, _apex, [va, vb, vc]) = single_face_reflex_junction();
    let result = mesh.bevel(
        &[va.into(), vb.into(), vc.into()],
        BevelOptions {
            offset: 0.2,
            affect: BevelAffect::Edges,
            segments: 5,
            profile: 1.0,
            vmesh_method: BevelVmeshMethod::Adj,
            miter_outer: BevelMiter::Arc,
            ..Default::default()
        },
    );
    mesh.validate_topology();
    for sample in [
        Vec3::new(-0.30115, 0.05620, 0.94380),
        Vec3::new(-0.08730, -0.15765, 1.15765),
        Vec3::new(0.08730, -0.15765, 1.15765),
        Vec3::new(0.30115, 0.05620, 0.94380),
    ] {
        assert_eq!(
            result_has_vert_near(&mesh, &result, sample),
            1,
            "segments 5: the arc sample {sample:?} is emitted",
        );
    }
}

/// BEVEL.md §8.9.2 / §8.9.5's exponent-scope note: the outer `ARC` miter carries **no
/// exponent gate**. Its admission is the `s ≥ 3` beveled-edge count and the §8.9.1 firing
/// margin, so a curved or straight analytic profile builds the same enlarged
/// `(s + 1)`-boundvert ring the circle does — §8.9.6's `+1` — and only the chain's
/// *positions* move with the slider (§8.3.0.2.1's "the profile value is structurally
/// inert" carried onto the miter). Declining there hands back the sharp ring, which is a
/// whole-mesh census divergence, not a placement one.
///
/// The positions are §8.9.5's general reduction `(2 − 2m)·½(B_1 + B_3) + (2m − 1)·midpoint`
/// with `m = √profile` read off the sampler, on this fan's symmetric arc cell where the
/// §8.2.2.1.1 resample is the identity at `segments = 2`: the plain chord midpoint at the
/// straight exponent, the `√2 − 1` bulge at the circle, and fuller than the circle at
/// `0.7`. The `segments = 3` station is where the resample stops being the identity, so it
/// pins the *generator* rather than one closed-form sample.
#[test]
fn outer_miter_arc_admits_every_analytic_exponent() {
    // §8.9.5.1 worked trace B's frame (offset 0.2, spread 0, ARC).
    let b1 = Vec3::new(-0.51500, 0.27005, 0.72995);
    let b3 = Vec3::new(0.51500, 0.27005, 0.72995);
    let midpoint = Vec3::new(0.0, -0.24495, 1.24495);
    // The frame literals are quoted to five places, so derived expectations carry a few
    // `1e-6` of quoting error — far inside this band and orders inside the ≈0.2 that
    // separates any two rows.
    let emitted_near = |mesh: &LoopMesh, result: &BevelResult, target: Vec3| {
        result
            .verts
            .iter()
            .filter(|&&vertex| (mesh.vertex_position(vertex) - target).length() < 1.0e-4)
            .count()
    };

    for segments in [2_u32, 3] {
        let (circular_mesh, _) =
            reflex_apex_outer_miter_profile(BevelMiter::Arc, None, 0.5, segments);
        let (sharp_mesh, _) =
            reflex_apex_outer_miter_profile(BevelMiter::Sharp, None, 0.5, segments);
        for profile in [0.25_f32, 0.7] {
            let (mesh, _) =
                reflex_apex_outer_miter_profile(BevelMiter::Arc, None, profile, segments);
            mesh.validate_topology();
            assert_eq!(
                counts(&mesh),
                counts(&circular_mesh),
                "profile {profile}, segments {segments}: the analytic slider cannot decide \
                 whether the outer ARC fires (§8.9.2)",
            );
            assert_ne!(
                counts(&mesh),
                counts(&sharp_mesh),
                "profile {profile}, segments {segments}: …and the census it matches is the \
                 mitered one, not the declined sharp ring",
            );
        }
    }

    // §8.9.6: the miter is `+1` boundvert on the one reflex sector at either parity.
    assert_eq!(
        counts(&reflex_apex_outer_miter_profile(BevelMiter::Arc, None, 0.7, 2).0),
        (19, 31, 14),
    );
    assert_eq!(
        counts(&reflex_apex_outer_miter_profile(BevelMiter::Arc, None, 0.7, 3).0),
        (29, 49, 22),
    );

    // §8.9.5's `bulge_mid(r)`: three exponents, three distinct points on the bisector.
    let round_sample = Vec3::new(0.0, 0.05673, 0.94327);
    for (profile, expected) in [
        (0.25_f32, Vec3::new(0.0, 0.270048, 0.729952)),
        (0.5, round_sample),
        (0.7, Vec3::new(0.0, -0.07671, 1.07671)),
    ] {
        let (mesh, result) = reflex_apex_outer_miter_profile(BevelMiter::Arc, None, profile, 2);
        assert_eq!(
            emitted_near(&mesh, &result, expected),
            1,
            "profile {profile}: the arc's lone interior sample is {expected:?}",
        );
        if profile != 0.5 {
            assert_eq!(
                emitted_near(&mesh, &result, round_sample),
                0,
                "profile {profile}: the forced-circle sample must be gone",
            );
        }
    }

    // §8.9.5.1 at an odd count: the emitted pair is the chord-fraction resample of the
    // run's own chain, and the guide's `(u, v) = (0.58451, 0.93227)` maps to these two.
    let (mesh, result) = reflex_apex_outer_miter_profile(BevelMiter::Arc, None, 0.7, 3);
    mesh.validate_topology();
    for sample in [
        Vec3::new(-0.179093, 0.003909, 0.996091),
        Vec3::new(0.179093, 0.003909, 0.996091),
    ] {
        assert_eq!(
            emitted_near(&mesh, &result, sample),
            1,
            "profile 0.7, segments 3: the arc sample {sample:?} is emitted",
        );
        let (u, v) = arc_frame_uv(sample, b1, b3, midpoint);
        assert!(
            ((u.max(v)) - 0.93227).abs() < 1.0e-4 && ((u.min(v)) - 0.58451).abs() < 1.0e-4,
            "…and decodes to the guide's ({u}, {v}) in the §8.9.5 arc frame",
        );
    }
    for sample in &emitted_arc_chain(b1, b3, midpoint, 0.5, 3)[1..3] {
        assert_eq!(
            emitted_near(&mesh, &result, *sample),
            0,
            "profile 0.7, segments 3: the circle's sample {sample:?} must be gone",
        );
    }
}

/// BEVEL.md §7.6.2 case 1: the meet-point of a **non-planar single-face reflex
/// sector** is built from the **edge-span normal** (the cross of the two flanking
/// beveled-edge directions), sign-fixed by the face normal — not the reflex
/// face's own tilted area normal, which its far vertex `w` drags off the
/// edge-span plane and which would collapse the meet toward the rim. The two
/// planar narrow sectors are untouched by the choice (their edges lie in the
/// face, so edge-span and face normal coincide).
#[test]
fn reflex_sector_meet_uses_edge_span_normal() {
    let offset = 0.2;
    let (mesh, apex, [va, vb, vc]) = single_face_reflex_junction();
    let apex_position = mesh.vertex_position(apex);
    let is_beveled = |edge: EdgeKey| edge == va || edge == vb || edge == vc;
    let corners = mesh.vertex_bevel_corner_ring(apex, &is_beveled);
    let sectors = sector_split(&corners, &is_beveled);

    let mut reflex_meet: Option<Vec3> = None;
    let mut narrow_meets: Vec<Vec3> = Vec::new();
    for sector in &sectors {
        let meet = mesh.solve_meet_point(apex, apex_position, sector, offset);
        let opening = sector.bounding_edges[0];
        let closing = sector.bounding_edges[1];
        let face = sector.corners[0].1;
        let d_open = mesh
            .edge_direction_from(opening, apex, apex_position)
            .unwrap();
        let d_close = mesh
            .edge_direction_from(closing, apex, apex_position)
            .unwrap();
        let reflex = (-d_open).cross(-d_close).dot(mesh.face_plane_normal(face)) < 0.0;
        if reflex {
            reflex_meet = Some(meet);
        } else {
            narrow_meets.push(meet);
        }
    }

    let reflex_meet = reflex_meet.expect("the fixture has exactly one reflex sector");
    assert!(
        (reflex_meet - Vec3::new(0.0, -0.24495, 1.24495)).length() < 1.0e-4,
        "reflex meet uses the edge-span normal (got {reflex_meet:?})",
    );
    // The face-normal (buggy) value sits ~0.3 unit short along the edge.
    assert!(
        (reflex_meet - Vec3::new(0.0, -0.21381, 0.94655)).length() > 0.2,
        "reflex meet is not the collapsed face-normal placement",
    );
    assert_eq!(narrow_meets.len(), 2, "two planar narrow sectors");
    for meet in narrow_meets {
        assert!(
            (meet.y - 0.46937).abs() < 1.0e-4 && (meet.z - 0.60908).abs() < 1.0e-4,
            "planar narrow-sector meets are unchanged by the edge-span choice (got {meet:?})",
        );
    }
}

/// BEVEL.md §8.9.2: with the §7.6.2 meet-point corrected, the outer-miter arc-end
/// boundvert lands at the live oracle position `(0.515, 0.270, 0.730)` — the
/// reposition slides the arc start back along its apex edge until level with the
/// preceding narrow-sector boundvert.
#[test]
fn reflex_outer_miter_arc_end_lands_at_oracle() {
    let (mesh, result) = reflex_apex_outer_miter(BevelMiter::Arc);
    mesh.validate_topology();
    assert_eq!(
        result_has_vert_near(&mesh, &result, Vec3::new(0.515, 0.27005, 0.72995)),
        1,
        "the arc-end boundvert sits at the corrected §8.9.2 oracle position",
    );
}

/// BEVEL.md §8.9.5.3 / §8.5.5.5(d): the outer-`ARC` cap over the **symmetric**
/// three-edge octagonal fan welds its valence-4 apex at the oracle-verified
/// `(0.14539, 0.14539, 0.66793)`. Because this ring is congruent, each flanking
/// cell's §8.2.2.1.1 resample is the identity, so the raw coarse mids equal the
/// emitted rim mids and the raw-mid centre stencil is a **bit-identical no-op**
/// here — the load-bearing guarantee that the raw-coarse-mid fix leaves the blessed
/// symmetric / B2 path unperturbed while landing exactly on the oracle apex.
/// BEVEL.md §8.9.5.3 / §8.5.5.5(c)–(e): the **odd**-`segments` outer-`ARC` cap over the
/// same three-edge octagonal fan closes on a central `(s + 1)`-gon — a quad at this
/// valence-4 ring — whose four verts the guide pins by oracle read-back. Reaching them
/// exercises the whole stencil end to end: the fullness-seeded coarse net, one
/// subdivision round to `N = 4` (raw coarse mids in, sharpened boundary as the stencil
/// input), and the §8.5.5.5(d) read-back to `segments = 3` — which, because `3` is not
/// a power of two, genuinely runs the interior resample.
#[test]
fn outer_miter_odd_cap_central_ngon_matches_oracle() {
    let (mut mesh, _apex, _rim, [va, vb, vc]) = octagonal_bipyramid();
    let result = mesh.bevel(
        &[va.into(), vb.into(), vc.into()],
        BevelOptions {
            offset: 0.2,
            affect: BevelAffect::Edges,
            segments: 3,
            vmesh_method: BevelVmeshMethod::Adj,
            miter_outer: BevelMiter::Arc,
            ..Default::default()
        },
    );
    mesh.validate_topology();

    let central = [
        Vec3::new(0.146_23, 0.285_17, 0.593_51),
        Vec3::new(0.023_81, 0.198_35, 0.662_73),
        Vec3::new(0.198_35, 0.023_81, 0.662_73),
        Vec3::new(0.285_17, 0.146_23, 0.593_51),
    ];
    for expected in central {
        assert_eq!(
            result_has_vert_near(&mesh, &result, expected),
            1,
            "the odd cap's central {expected:?} is the §8.9.5.3 oracle vert",
        );
    }

    // §8.5.5.5(e): exactly one central `s`-gon face, and its verts are those four.
    let closing: Vec<FaceKey> = result
        .faces
        .iter()
        .copied()
        .filter(|&face| {
            mesh.face_vert_keys(face).iter().all(|&v| {
                central
                    .iter()
                    .any(|&c| (mesh.vertex_position(v) - c).length() < 1.0e-5)
            })
        })
        .collect();
    assert_eq!(
        closing.len(),
        1,
        "odd parity closes on exactly one central face"
    );
    assert_eq!(
        mesh.face_vert_keys(closing[0]).len(),
        4,
        "the central face of a valence-4 ring is a quad",
    );

    // The reflex arc's own rim samples are untouched by the interior machinery — they
    // stay the §8.9.5.1 chord-fraction read-back (§8.2.2.1.1, "the emitted boundary").
    for arc_sample in [
        Vec3::new(-0.091_67, 0.137_19, 0.683_04),
        Vec3::new(0.137_19, -0.091_67, 0.683_04),
    ] {
        assert_eq!(
            result_has_vert_near(&mesh, &result, arc_sample),
            1,
            "the emitted rim is still the resampled arc, not a read-back of the grid",
        );
    }
}

#[test]
fn outer_miter_symmetric_fan_apex_matches_oracle() {
    let (mut mesh, _apex, _rim, [va, vb, vc]) = octagonal_bipyramid();
    let result = mesh.bevel(
        &[va.into(), vb.into(), vc.into()],
        BevelOptions {
            offset: 0.2,
            affect: BevelAffect::Edges,
            segments: 2,
            vmesh_method: BevelVmeshMethod::Adj,
            miter_outer: BevelMiter::Arc,
            ..Default::default()
        },
    );
    mesh.validate_topology();
    assert_eq!(
        result_has_vert_near(&mesh, &result, Vec3::new(0.145_385, 0.145_385, 0.667_927)),
        1,
        "the symmetric outer-miter fan welds its apex at the §8.9.5.3 oracle centre",
    );
}

/// BEVEL.md §8.9.5.3: the once-subdivided valence-4 extraordinary-centre stencil
/// collapses to the fixed affine blend `apex = 0.5·mean_mid + 0.186·mean_b +
/// 0.314·v` in its coarse inputs (fullness `0.559`, `γ_4 = 0.25`, sharpening mask
/// `(−1/6, +4/3, −1/6)`). Pinning the kernel to this closed form is the proof that
/// it is the **faithful** coarse-net reconstruction over the *true* ring adjacency —
/// not a symmetry-only contraction — and that exactly **half** of any mid-selection
/// error passes to the apex, which is why feeding the resampled rim mids in place of
/// the raw coarse mids is a correctness bug, not a Tier-2 rounding.
#[test]
fn extraordinary_centre_is_affine_blend_of_raw_coarse_inputs() {
    // An asymmetric (non-congruent) valence-4 ring so the identity is not trivially
    // masked by symmetry.
    let v = Vec3::new(0.2, -0.1, 1.3);
    let boundverts = [
        Vec3::new(0.7, 0.1, 0.6),
        Vec3::new(0.2, 0.8, 0.5),
        Vec3::new(-0.6, 0.3, 0.55),
        Vec3::new(-0.1, -0.7, 0.62),
    ];
    let mids = [
        Vec3::new(0.55, 0.5, 0.75),
        Vec3::new(-0.25, 0.62, 0.72),
        Vec3::new(-0.42, -0.28, 0.7),
        Vec3::new(0.34, -0.35, 0.71),
    ];
    let apex = adj_even_extraordinary_centre(v, &boundverts, &mids, sabin_gamma(4));
    let mean_mid = mids.iter().fold(Vec3::ZERO, |acc, &m| acc + m) / 4.0;
    let mean_b = boundverts.iter().fold(Vec3::ZERO, |acc, &b| acc + b) / 4.0;
    let affine = 0.5 * mean_mid + 0.185_5625 * mean_b + 0.314_4375 * v;
    assert!(
        (apex - affine).length() < 1.0e-6,
        "the valence-4 centre stencil equals the §8.9.5.3 affine blend (got {apex:?} vs {affine:?})",
    );

    // Half of any mid-selection error reaches the apex: perturbing one mid by `d`
    // moves the apex by exactly `0.5 · d / 4`.
    let mut perturbed = mids;
    let bump = Vec3::new(0.04, -0.03, 0.02);
    perturbed[0] += bump;
    let moved = adj_even_extraordinary_centre(v, &boundverts, &perturbed, sabin_gamma(4));
    assert!(
        (moved - apex - 0.5 * bump / 4.0).length() < 1.0e-6,
        "a resampled-vs-raw mid difference propagates to the apex at coefficient 0.5",
    );
}

/// BEVEL.md §8.5.5.5(d) / §8.9.5.3: `coarse_flank_mid` reads the **pre-resample**
/// coarse `(i, 0, 1)` profile point — the naive cell sample the §8.2.2.1.1
/// chord-fraction resample would correct. On the skewed arc-flanking cell it differs
/// from the emitted (resampled) rim mid; on a symmetric equal-offset cell the two
/// coincide (the resample is the identity), which is why the raw-mid centre feed is
/// a no-op on congruent rings and only bites on the asymmetric ones.
#[test]
fn coarse_flank_mid_is_the_raw_pre_resample_sample() {
    // The proven skewed arc-flanking cell of `segment2_flanking_resample_matches_worked_trace`
    // (unequal legs 0.200 / 0.133), rebuilt through vertex keys / an edge so it drives
    // the method: `ea` is the crossed edge's apex, `eb` its rim end.
    let mut mesh = LoopMesh::new();
    let start = mesh.create_vertex(Vec3::new(-0.19194, 0.46338, 0.45712));
    let end = mesh.create_vertex(Vec3::new(-0.36871, 0.24371, 0.42679));
    let ep0 = mesh.create_vertex(Vec3::new(0.0, 0.0, 1.0));
    let ep1 = mesh.create_vertex(Vec3::new(-0.70711, 0.70711, 0.0));
    let edge = mesh.create_edge([ep0, ep1], EdgeCreateFlags::NONE);

    // The cell is not collinear, so the §8.2.2.1 rescue is inert here; the apex
    // stands in for the corner vertex.
    let rescue = MidRescue {
        vertex: mesh.vertex_position(ep0),
        neighbour_dirs: None,
    };
    let raw = mesh.coarse_flank_mid(start, end, edge, 0.5, None, rescue);
    let edge_dir = (mesh.vertex_position(ep1) - mesh.vertex_position(ep0)).normalize_or_zero();
    let cell = ProfileCell::new(
        mesh.vertex_position(start),
        mesh.vertex_position(end),
        mesh.vertex_position(ep0),
        edge_dir,
    );
    let naive = cell.chain_positions(2, 0.5, None, ChainEmission::Raw)[1];
    let emitted = cell.chain_positions(2, 0.5, None, ChainEmission::Resampled)[1];
    assert!(
        (raw - naive).length() < 1.0e-6,
        "coarse_flank_mid is the non-resampled naive coarse mid",
    );
    assert!(
        (raw - emitted).length() > 1.0e-4,
        "on the skewed flanking cell the raw coarse mid differs from the emitted resampled rim mid",
    );
}

/// BEVEL.md §8.9.5.3 / §8.5.5.7: an **asymmetric** outer-`ARC` cap — non-contiguous
/// beveled legs, a convex-side absorbed leg, and a wide reflex arc — still closes on
/// a single welded valence-4 apex, and that apex is the deterministic raw-coarse-mid
/// extraordinary centre (fed loopmesh's own boundverts / coarse mids). The pin guards
/// the asymmetric-ring centre against re-regressing onto the emitted resampled mids.
#[test]
fn outer_miter_absorbed_leg_asymmetric_cap_welds_raw_mid_apex() {
    let (mut mesh, _apex, _rim, legs) = octagonal_bipyramid_all_legs();
    let result = mesh.bevel(
        &[legs[1].into(), legs[3].into(), legs[4].into()],
        BevelOptions {
            offset: 0.2,
            affect: BevelAffect::Edges,
            segments: 2,
            vmesh_method: BevelVmeshMethod::Adj,
            miter_outer: BevelMiter::Arc,
            ..Default::default()
        },
    );
    mesh.validate_topology();
    // The reflex arc raises the ring to valence 4 and the even cap welds one apex.
    assert_eq!(
        result_has_vert_near(&mesh, &result, Vec3::new(-0.115_204, 0.068_405, 0.735_506)),
        1,
        "the asymmetric absorbed-leg cap welds its apex at the raw-mid extraordinary centre",
    );
}

/// BEVEL.md §8.2.2.1.1 / §8.9.5.1: the reflex **arc** is a ring-edge chain of the
/// [V-ADJ] cap like any other, so its emitted rim samples are the chord-fraction
/// resample of the raw arc against the profile at `N = next pow2 ≥ max(4, segments)` —
/// never the raw `P(45°)` bulge. On this absorbed-leg driver the arc cell is grossly
/// skewed (legs `0.664` / `0.280`) and the raw bulge misses the emitted sample by
/// ≈`0.008` world-units (≈803 position quanta), so emitting the raw point is a hard
/// divergence, not a rounding.
///
/// The apex assertion is the paired guard: the rim is resampled, but the cap **centre**
/// stencil still consumes the *raw* coarse `(i, 0, 1)` bulge (§8.5.5.5(d) / §8.9.5.3).
/// Aliasing the two — feeding the centre the emitted rim vertex — moves the apex, so
/// this pin fails the moment the raw/emitted split is collapsed.
///
/// (This fixture's octagon is the guide driver's rotated 45° about `z` — its rim
/// numbering starts one leg over — so every worked coordinate appears here rotated;
/// the arc cell, its legs, and every displacement are congruent.)
#[test]
fn multiface_reflex_absorbed_leg_arc_rim_mid_is_the_chord_resample() {
    let (mut mesh, _apex, _rim, legs) = octagonal_bipyramid_all_legs();
    let result = mesh.bevel(
        &[legs[1].into(), legs[3].into(), legs[4].into()],
        BevelOptions {
            offset: 0.2,
            affect: BevelAffect::Edges,
            segments: 2,
            vmesh_method: BevelVmeshMethod::Adj,
            miter_outer: BevelMiter::Arc,
            ..Default::default()
        },
    );
    mesh.validate_topology();

    let emitted = Vec3::new(-0.052_441, -0.045_032, 0.702_732);
    let raw_bulge = Vec3::new(-0.059_892, -0.047_334, 0.700_830);
    assert_eq!(
        result_has_vert_near(&mesh, &result, emitted),
        1,
        "the arc rim mid is the §8.2.2.1.1 chord-fraction resample",
    );
    assert_eq!(
        result_has_vert_near(&mesh, &result, raw_bulge),
        0,
        "the raw 45° bulge is never emitted as a rim vertex on a skewed arc cell",
    );
    assert!(
        (emitted - raw_bulge).length() > 0.007,
        "the raw-vs-emitted arc gap is the ≈803-quantum divergence, not a rounding",
    );
    assert_eq!(
        result_has_vert_near(&mesh, &result, Vec3::new(-0.115_204, 0.068_405, 0.735_506)),
        1,
        "the apex still welds at the raw-mid extraordinary centre — the centre reads the raw bulge, the rim the resampled one",
    );
}

/// BEVEL.md §8.2.2.1.1 (gate part (i)) / §8.9.5.1: at a **non-power-of-two**
/// `segments` the source resolution is `N = next pow2 ≥ max(4, segments)` — `4` at
/// `segments = 3`, *not* `2·segments` — so every interior arc sample is displaced off
/// the raw profile. The gate is `segments`-first: not a skew test, and not a
/// `segments = 2` special case.
#[test]
fn multiface_reflex_absorbed_leg_arc_segments3_samples_are_resampled() {
    let (mut mesh, _apex, _rim, legs) = octagonal_bipyramid_all_legs();
    let result = mesh.bevel(
        &[legs[1].into(), legs[3].into(), legs[4].into()],
        BevelOptions {
            offset: 0.2,
            affect: BevelAffect::Edges,
            segments: 3,
            vmesh_method: BevelVmeshMethod::Adj,
            miter_outer: BevelMiter::Arc,
            ..Default::default()
        },
    );
    mesh.validate_topology();

    for emitted in [
        Vec3::new(-0.175_752, -0.066_595, 0.631_479),
        Vec3::new(0.046_653, -0.014_437, 0.727_964),
    ] {
        assert_eq!(
            result_has_vert_near(&mesh, &result, emitted),
            1,
            "the odd-segments arc sample is the chord-fraction resample of the 4-segment profile",
        );
    }
    for raw in [
        Vec3::new(-0.179_437, -0.069_608, 0.635_060),
        Vec3::new(0.043_777, -0.018_307, 0.734_420),
    ] {
        assert_eq!(
            result_has_vert_near(&mesh, &result, raw),
            0,
            "the raw even-angle arc sample is not emitted at a non-power-of-two segments",
        );
    }
}
