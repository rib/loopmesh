use super::*;

/// BEVEL.md §6.6.1.4.0(c), condition (1)'s winding box: the flanking-face slots are
/// filled by pure incidence, so the seam predicate counts incident faces and is blind
/// to winding. Two incident faces fill both slots however they are wound; fewer than
/// two genuinely leaves a side empty.
#[test]
fn leg_missing_flanking_face_is_winding_blind() {
    let mut mesh = LoopMesh::new();
    let a = mesh.create_vertex(Vec3::ZERO);
    let b = mesh.create_vertex(Vec3::new(1.0, 0.0, 0.0));
    let c = mesh.create_vertex(Vec3::new(0.5, 1.0, 0.0));
    let d = mesh.create_vertex(Vec3::new(0.5, -1.0, 0.0));

    // Consistently wound radial-2: `[a,b,c]` traverses a→b, `[b,a,d]` traverses b→a.
    mesh.face_create_verts(&[a, b, c], FaceCreateFlags::NONE);
    mesh.face_create_verts(&[b, a, d], FaceCreateFlags::NONE);
    let consistent = mesh.query_edge_exists(a, b).expect("shared edge");
    assert!(
        !mesh.leg_missing_flanking_face(consistent),
        "a consistently wound radial-2 leg fills both slots"
    );

    // Inconsistently wound radial-2: both faces traverse the shared edge the *same*
    // way. The winding-sensitive resolution reports an empty slot here; incidence does
    // not — two faces still flank the edge.
    let mut mis = LoopMesh::new();
    let p = mis.create_vertex(Vec3::ZERO);
    let q = mis.create_vertex(Vec3::new(1.0, 0.0, 0.0));
    let r = mis.create_vertex(Vec3::new(0.5, 1.0, 0.0));
    let s = mis.create_vertex(Vec3::new(0.5, -1.0, 0.0));
    mis.face_create_verts(&[p, q, r], FaceCreateFlags::NONE);
    mis.face_create_verts(&[p, q, s], FaceCreateFlags::NONE); // same p→q traversal
    let inconsistent = mis.query_edge_exists(p, q).expect("shared edge");
    assert!(
        !mis.leg_missing_flanking_face(inconsistent),
        "an inconsistently wound radial-2 leg still flanks two faces — not a seam"
    );

    // Boundary radial-1: one face only.
    let mut bnd = LoopMesh::new();
    let u = bnd.create_vertex(Vec3::ZERO);
    let v = bnd.create_vertex(Vec3::new(1.0, 0.0, 0.0));
    let w = bnd.create_vertex(Vec3::new(0.5, 1.0, 0.0));
    bnd.face_create_verts(&[u, v, w], FaceCreateFlags::NONE);
    let boundary = bnd.query_edge_exists(u, v).expect("edge");
    assert!(
        bnd.leg_missing_flanking_face(boundary),
        "a boundary leg has one empty side"
    );

    // Wire radial-0: no faces at all.
    let mut wire = LoopMesh::new();
    let x = wire.create_vertex(Vec3::ZERO);
    let y = wire.create_vertex(Vec3::new(1.0, 0.0, 0.0));
    let spoke = wire.create_edge([x, y], EdgeCreateFlags::NONE);
    assert!(
        wire.leg_missing_flanking_face(spoke),
        "a wire leg flanks nothing"
    );
}

/// A closed corner whose legs are all radial-2 but two of them are inconsistently
/// wound: §6.6.1.4 condition (1) is winding-blind (§6.6.1.4.0(c)), so the corner is
/// **not** seam-carrying and [`ring_spanning_donor`](LoopMesh::ring_spanning_donor)
/// keeps the positional pick rather than the tie-break winner.
#[test]
fn ring_spanning_donor_condition_one_is_winding_blind_at_the_corner() {
    let mut mesh = LoopMesh::new();
    // `v0` sits well below the rim so it is every face's z-extreme; each face's
    // bounding-box-centre z is then set by the higher of its two rim verts, which lets
    // the geometry alone pick the tie-break winner (§6.6.1.4 key 4).
    let v0 = mesh.create_vertex(Vec3::new(0.0, 0.0, -100.0));
    let r0 = mesh.create_vertex(Vec3::new(1.0, 0.0, 10.0));
    let r1 = mesh.create_vertex(Vec3::new(-0.5, 0.87, 20.0));
    let r2 = mesh.create_vertex(Vec3::new(-0.5, -0.87, 30.0));

    let f1 = mesh.face_create_verts(&[v0, r0, r1], FaceCreateFlags::NONE);
    let f2 = mesh.face_create_verts(&[v0, r1, r2], FaceCreateFlags::NONE);
    // Flipped from the consistent `[v0, r2, r0]`: this makes legs `{v0,r0}` and
    // `{v0,r2}` inconsistently wound while every leg stays radial-2.
    let f3 = mesh.face_create_verts(&[v0, r0, r2], FaceCreateFlags::NONE);

    let positional = mesh.first_incident_face(v0);
    let winner = mesh.donor_tie_break(&[f1, f2, f3], None);
    assert_ne!(
        winner, positional,
        "fixture sanity: the seam path's tie-break winner must differ from positional"
    );

    let donor = mesh.ring_spanning_donor(v0, &|_| false, BevelAffect::Vertices, 1, None);
    assert_eq!(
        donor, positional,
        "an inconsistently wound radial-2 corner is not a condition-(1) seam: \
         the donor is the positional pick, not the tie-break winner"
    );
}

/// The contrast: a genuine boundary leg (radial-1) *is* a condition-(1) seam, so the
/// same fill routes its donor through the tie-break winner, not the positional pick.
#[test]
fn ring_spanning_donor_boundary_leg_is_a_condition_one_seam() {
    let mut mesh = LoopMesh::new();
    let v0 = mesh.create_vertex(Vec3::new(0.0, 0.0, -100.0));
    let r0 = mesh.create_vertex(Vec3::new(1.0, 0.0, 10.0));
    let r1 = mesh.create_vertex(Vec3::new(0.0, 1.0, 20.0));
    let r2 = mesh.create_vertex(Vec3::new(-1.0, 0.0, 30.0));

    // Open fan: `{v0,r0}` and `{v0,r2}` are boundary legs, `{v0,r1}` is radial-2.
    mesh.face_create_verts(&[v0, r0, r1], FaceCreateFlags::NONE);
    mesh.face_create_verts(&[v0, r1, r2], FaceCreateFlags::NONE);

    let positional = mesh.first_incident_face(v0);
    let winner = mesh.donor_tie_break(&mesh.face_keys(), None);
    assert_ne!(
        winner, positional,
        "fixture sanity: winner must differ from positional to discriminate the paths"
    );

    let donor = mesh.ring_spanning_donor(v0, &|_| false, BevelAffect::Vertices, 1, None);
    assert_eq!(
        donor, winner,
        "a boundary leg is a condition-(1) seam: the donor is the tie-break winner"
    );
}

/// BEVEL.md §6.6.2's odd-`segments` centre-column donor selection, in isolation. The
/// column takes the sector's **own** rep only when its boundvert has a beveled edge
/// that is not an interpolation seam; otherwise — a seamed edge, or an unbeveled leg —
/// it takes the §6.6.1.4 tie-break between the own and the **next** boundvert's reps.
#[test]
fn centre_column_donor_switches_between_own_rep_and_the_tie_break() {
    let mut mesh = LoopMesh::new();
    let v0 = mesh.create_vertex(Vec3::new(0.0, 0.0, -100.0));
    let r0 = mesh.create_vertex(Vec3::new(1.0, 0.0, 10.0));
    let r1 = mesh.create_vertex(Vec3::new(0.0, 1.0, 20.0));
    let r2 = mesh.create_vertex(Vec3::new(-1.0, 0.0, 30.0));

    // Open fan: `{v0,r0}` and `{v0,r2}` are boundary (radial-1) legs — interpolation
    // seams by condition (1) — while `{v0,r1}` is a radial-2 manifold leg.
    let own = mesh.face_create_verts(&[v0, r0, r1], FaceCreateFlags::NONE);
    let next = mesh.face_create_verts(&[v0, r1, r2], FaceCreateFlags::NONE);
    // Slot key (§6.6.1.4 key 3) decides the tie-break: the lower slot donates, so with
    // `own` the higher slot the tie-break winner is `next`, distinct from the own rep.
    mesh.set_face_material_index(own, 5);
    mesh.set_face_material_index(next, 2);

    let seamed = mesh.query_edge_exists(v0, r0).unwrap();
    let manifold = mesh.query_edge_exists(v0, r1).unwrap();
    assert!(mesh.edge_is_interp_seam(seamed), "boundary leg is a seam");
    assert!(
        !mesh.edge_is_interp_seam(manifold),
        "manifold leg is not a seam"
    );

    let tie_break = mesh.donor_tie_break(&[own, next], None);
    assert_eq!(
        tie_break,
        Some(next),
        "fixture sanity: the lower slot wins the tie-break"
    );

    // Non-seam beveled edge: the sector's own rep.
    assert_eq!(
        mesh.centre_column_donor(manifold, Some(own), Some(next), true, None),
        Some(own),
        "a non-seam beveled boundvert keeps its own rep",
    );
    // Seamed beveled edge: the tie-break winner, not the own rep.
    assert_eq!(
        mesh.centre_column_donor(seamed, Some(own), Some(next), true, None),
        Some(next),
        "a seamed beveled boundvert routes through the §6.6.1.4 tie-break",
    );
    // Unbeveled boundvert (no beveled edge of its own): the tie-break, seam or not.
    assert_eq!(
        mesh.centre_column_donor(manifold, Some(own), Some(next), false, None),
        Some(next),
        "an unbeveled / miter-split boundvert takes the tie-break unconditionally",
    );
}

/// A fan around `v0` whose three legs flank two distinct-slot faces, plus the two edge
/// handles a miter ring's `cross_edges` names on either side of an edge-less boundvert.
/// `own` (slot 5) departs along `{v0,r0}`; `next` (slot 2) arrives along `{v0,r2}` — so
/// an edge-less slot flanked by `Some({v0,r0})` / `Some({v0,r2})` tie-breaks between them.
fn miter_centre_column_fan() -> (LoopMesh, VertexKey, EdgeKey, EdgeKey, FaceKey, FaceKey) {
    let mut mesh = LoopMesh::new();
    let v0 = mesh.create_vertex(Vec3::new(0.0, 0.0, -100.0));
    let r0 = mesh.create_vertex(Vec3::new(1.0, 0.0, 10.0));
    let r1 = mesh.create_vertex(Vec3::new(0.0, 1.0, 20.0));
    let r2 = mesh.create_vertex(Vec3::new(-1.0, 0.0, 30.0));
    let own = mesh.face_create_verts(&[v0, r0, r1], FaceCreateFlags::NONE);
    let next = mesh.face_create_verts(&[v0, r1, r2], FaceCreateFlags::NONE);
    mesh.set_face_material_index(own, 5);
    mesh.set_face_material_index(next, 2);
    let e_prev = mesh.query_edge_exists(v0, r0).unwrap();
    let e_next = mesh.query_edge_exists(v0, r2).unwrap();
    (mesh, v0, e_prev, e_next, own, next)
}

/// BEVEL.md §6.6.2 / §8.9.3: on a miter cap's enlarged ring, both a **seamed beveled**
/// sector's centre column and an **edge-less** (miter-split) boundvert's centre column
/// route through the §6.6.1.4 tie-break — the beveled sector because its boundary leg is
/// a condition-(1) seam, the edge-less boundvert unconditionally. Neither takes a
/// ring-wide uniform donor. Pins the direction: the lower-slot face of {this boundvert's
/// own rep, the next boundvert's rep} donates.
#[test]
fn miter_ring_seamed_beveled_and_edge_less_centre_columns_take_the_tie_break() {
    let (mesh, v0, e_prev, e_next, own, next) = miter_centre_column_fan();
    // Both beveled legs `{v0,r0}` and `{v0,r2}` are boundary edges — §6.6.1.4
    // condition-(1) seams — so every sector's centre column takes the tie-break.
    let cross_edges = [Some(e_prev), None, Some(e_next)];
    assert!(
        mesh.edge_is_interp_seam(e_prev),
        "the beveled leg `{{v0,r0}}` is a seam"
    );
    assert!(
        mesh.edge_is_interp_seam(e_next),
        "the beveled leg `{{v0,r2}}` is a seam"
    );
    let donors = mesh.miter_ring_centre_column_donors(v0, &cross_edges, None);

    assert_eq!(donors.len(), 3);
    // Edge-less sector 1: tie-break between its own rep (`own`, 5) and the next
    // boundvert's rep (`next`, 2) → the lower slot `next`.
    assert_eq!(
        donors[1],
        Some(next),
        "the edge-less boundvert's centre column takes the §6.6.1.4 tie-break (lower slot)",
    );
    // Seamed beveled sector 2: its own rep is `next` (2), its next boundvert (wrap to
    // sector 0) reps `own` (5); the seam routes it through the tie-break, which picks the
    // lower slot `next`. A non-seam edge would keep the own rep `next` here too, but the
    // discriminating fact is that the *ring-wide* uniform donor `own` is **not** taken.
    assert_eq!(
        donors[2],
        Some(next),
        "a seamed beveled sector routes through the §6.6.1.4 tie-break, not a uniform donor",
    );
    assert_ne!(
        donors[2], donors[0],
        "sector 2's centre column is not sector 0's donor"
    );
    // Seamed beveled sector 0: own rep `own` (5); its next boundvert (edge-less sector 1)
    // also reps `own`, so the tie-break's only candidate is `own` — it resolves there via
    // the tie-break, not by keeping a ring-wide uniform positional donor.
    assert_eq!(
        donors[0],
        Some(own),
        "sector 0's tie-break has the single candidate `own`"
    );
}

/// The direction of the tie-break is the §6.6.1.4 material-slot key: the **lower** slot
/// donates. Swapping which flank carries the lower slot swaps the edge-less centre
/// column's donor, proving the pick is geometric/material-directed, not a fixed side.
#[test]
fn miter_ring_edge_less_centre_column_follows_the_material_slot_direction() {
    let (mut mesh, v0, e_prev, e_next, own, next) = miter_centre_column_fan();
    let cross_edges = [Some(e_prev), None, Some(e_next)];

    // As built: `own` = slot 5, `next` = slot 2 → the lower slot `next` wins.
    let donors = mesh.miter_ring_centre_column_donors(v0, &cross_edges, None);
    assert_eq!(donors[1], Some(next), "lower slot (`next` = 2) donates");

    // Flip the slots: now `own` is the lower slot, so it wins the same edge-less column.
    mesh.set_face_material_index(own, 2);
    mesh.set_face_material_index(next, 5);
    let donors = mesh.miter_ring_centre_column_donors(v0, &cross_edges, None);
    assert_eq!(
        donors[1],
        Some(own),
        "after the swap the lower slot (`own` = 2) donates — the pick is slot-directed",
    );
}

/// BEVEL.md §6.6.2's grid row: a miter cap's enlarged-ring **non-centre** grid faces take
/// each sector's **own** boundvert rep — a per-sector positional donor, never one uniform
/// rep shared across the ring. No seam gate, no tie-break enters (that is the centre
/// column's business). Pins the direction so a two-material split-sector junction cannot
/// silently collapse back to a single donor.
#[test]
fn miter_ring_sector_donors_are_per_sector_positional_own_reps() {
    let (mesh, v0, e_prev, e_next, own, next) = miter_centre_column_fan();
    // Ring slot 1 is the edge-less (arc) sector; slots 0 / 2 own the beveled legs
    // `{v0,r0}` (→ `own`, slot 5) and `{v0,r2}` (→ `next`, slot 2).
    let cross_edges = [Some(e_prev), None, Some(e_next)];
    let donors = mesh.miter_ring_sector_donors(v0, &cross_edges);

    assert_eq!(donors.len(), 3);
    // Beveled sector 0: its own boundvert rep along `e_prev` — `own`.
    assert_eq!(
        donors[0],
        Some(own),
        "a beveled sector takes `boundvert_representative` of its own cross edge",
    );
    // Beveled sector 2: its own boundvert rep along `e_next` — `next`, *distinct* from
    // sector 0's donor. This is the per-sector fact the uniform `vec![rep; n]` erased.
    assert_eq!(
        donors[2],
        Some(next),
        "a distinct beveled sector takes its own (distinct) rep, not sector 0's",
    );
    assert_ne!(
        donors[0], donors[2],
        "sectors with different own faces get different donors — the divergence the fix closes",
    );
    // Arc-start sector 1 (§8.9.5.3): its leading flank `{v0,r0}` is an **open-fan**
    // (boundary) leg here, so `arriving_face`/`prev_face` is empty and §6.6.1's `first_edge`
    // arm falls through to the departing face `own`. (On a manifold leading flank — the real
    // reflex-junction case — the primary `arriving_face`/neighbour branch wins instead; that
    // is pinned by `miter_ring_arc_start_sector_donor_is_the_neighbour_across_the_leading_flank`.)
    assert_eq!(
        donors[1],
        Some(own),
        "an arc-start sector's leading-flank rep falls to the departing face on an open fan",
    );
}

/// A **closed** three-face fan around `v0` whose middle legs are radial-2 (manifold), plus
/// the two edge handles a miter ring names on either side of an arc-*start* boundvert. This
/// is the real reflex-junction shape (unlike [`miter_centre_column_fan`], whose legs are
/// open-fan boundaries): the leading flank `g_k = {v0,r1}` has a face on **both** fan sides,
/// so `arriving_face(g_k)` (the neighbour `fN`) and `departing_face(g_k)` (the own/reflex
/// `fR`) are distinct — the asymmetry §8.9.5.3 turns on. Slots: `fN`=1 (neighbour across
/// `g_k`), `fR`=2 (reflex / own sector), `fT`=3 (across the trailing flank).
fn miter_arc_start_manifold_fan() -> (
    LoopMesh,
    VertexKey,
    EdgeKey,
    EdgeKey,
    FaceKey,
    FaceKey,
    FaceKey,
) {
    let mut mesh = LoopMesh::new();
    let v0 = mesh.create_vertex(Vec3::new(0.0, 0.0, -100.0));
    let r0 = mesh.create_vertex(Vec3::new(1.0, 0.0, 10.0));
    let r1 = mesh.create_vertex(Vec3::new(0.5, 1.0, 20.0));
    let r2 = mesh.create_vertex(Vec3::new(-0.5, 1.0, 30.0));
    let r3 = mesh.create_vertex(Vec3::new(-1.0, 0.0, 40.0));
    let f_n = mesh.face_create_verts(&[v0, r0, r1], FaceCreateFlags::NONE);
    let f_r = mesh.face_create_verts(&[v0, r1, r2], FaceCreateFlags::NONE);
    let f_t = mesh.face_create_verts(&[v0, r2, r3], FaceCreateFlags::NONE);
    mesh.set_face_material_index(f_n, 1);
    mesh.set_face_material_index(f_r, 2);
    mesh.set_face_material_index(f_t, 3);
    let g_k = mesh.query_edge_exists(v0, r1).unwrap(); // leading flank of the arc start
    let g_k1 = mesh.query_edge_exists(v0, r2).unwrap(); // trailing flank
    (mesh, v0, g_k, g_k1, f_n, f_r, f_t)
}

/// BEVEL.md §8.9.5.3 / §6.6.1's `first_edge` arm — the corrected **asymmetric** rule: an
/// arc-*start* boundvert `B_1` keeps its **leading** flank edge `g_k` as a plain leg and
/// takes `prev_face(g_k)` = the **non-reflex neighbour across `g_k`** (`arriving_face`), NOT
/// `departing_face(g_k)` (the reflex / own-sector face). On a manifold `g_k` the two are
/// distinct, and the old edge-less reading (departing face) landed the reflex block on the
/// wrong arc end. This is the row-1042 divergence: loopmesh gave own/reflex, BMesh the
/// neighbour.
#[test]
fn miter_ring_arc_start_sector_donor_is_the_neighbour_across_the_leading_flank() {
    let (mesh, v0, g_k, g_k1, f_n, f_r, _f_t) = miter_arc_start_manifold_fan();

    // Fixture sanity: `g_k` is a manifold non-seam leg whose two fan sides are distinct.
    assert!(
        !mesh.edge_is_interp_seam(g_k),
        "the leading flank is a manifold non-seam leg"
    );
    assert_eq!(
        mesh.arriving_face(v0, g_k),
        Some(f_n),
        "prev_face(g_k) is the neighbour fN across the leading flank",
    );
    assert_eq!(
        mesh.departing_face(v0, g_k),
        Some(f_r),
        "next_face(g_k) is the reflex / own sector fR — the wrong donor",
    );

    // Arc start at ring index 1: leading flank `g_k`, trailing `g_(k+1)`.
    let cross_edges = [Some(g_k), None, Some(g_k1)];
    let donors = mesh.miter_ring_sector_donors(v0, &cross_edges);
    assert_eq!(
        donors[1],
        Some(f_n),
        "the arc-start sector inherits the neighbour across g_k (prev_face), not the reflex face",
    );
    assert_ne!(
        donors[1],
        Some(f_r),
        "the reflex block must NOT reach the arc-start sub-patch (§8.9.5.3 asymmetry)",
    );
}

/// BEVEL.md §8.9.5.3 odd-`segments` bullet: the arc-*start* `B_1`'s centre column feeds **its
/// own** rep — the neighbour across `g_k` (`prev_face`, the `first_edge` arm) — into the
/// §6.6.1.4 tie-break against the **next** boundvert's rep. With the neighbour on the lower
/// slot the tie-break returns it, which the old departing-face reading (whose own candidate
/// was the reflex face) could never produce.
#[test]
fn miter_ring_arc_start_centre_column_tie_breaks_from_the_neighbour_rep() {
    let (mesh, v0, g_k, g_k1, f_n, f_r, _f_t) = miter_arc_start_manifold_fan();
    let cross_edges = [Some(g_k), None, Some(g_k1)];

    // The centre column's own candidate is the neighbour `fN` (slot 1); the next boundvert
    // (arc end `B_3`) reps `fR` (slot 2, = prev_face(g_(k+1)) = the reflex face). The tie-break
    // picks the lower slot `fN`. The old own candidate would have been `fR`, giving `fR`.
    assert_eq!(
        mesh.boundvert_representative(v0, g_k1),
        Some(f_r),
        "the next boundvert (arc end) reps the reflex face fR",
    );
    let donors = mesh.miter_ring_centre_column_donors(v0, &cross_edges, None);
    assert_eq!(
        donors[1],
        Some(f_n),
        "the arc-start centre column tie-breaks between the neighbour fN and the reflex fR → fN",
    );
    assert_ne!(
        donors[1],
        Some(f_r),
        "the reflex face is the next rep, not the arc-start's own"
    );
}

/// The edge-less sector's non-centre grid donor is the *positional* own-rep primitive —
/// **not** the §6.6.1.4 tie-break the centre column of that same sector takes. Contrasts
/// the two slots on one fixture: on a seamed edge-less sector they diverge.
#[test]
fn miter_ring_edge_less_sector_donor_is_positional_not_the_centre_column_tie_break() {
    let (mesh, v0, e_prev, e_next, own, next) = miter_centre_column_fan();
    let cross_edges = [Some(e_prev), None, Some(e_next)];

    // Non-centre grid faces: positional own-rep = departing along `e_prev` = `own` (5).
    let sector = mesh.miter_ring_sector_donors(v0, &cross_edges);
    assert_eq!(
        sector[1],
        Some(own),
        "grid faces keep the positional own-rep"
    );

    // The centre column of the same edge-less sector routes through the tie-break, which
    // the fan resolves to the lower-slot `next` (2) — a strictly different donor.
    let centre = mesh.miter_ring_centre_column_donors(v0, &cross_edges, None);
    assert_eq!(
        centre[1],
        Some(next),
        "the centre column takes the tie-break winner"
    );
    assert_ne!(
        sector[1], centre[1],
        "grid vs centre-column donors are distinct slots on a seamed edge-less sector",
    );
}

/// BEVEL.md §6.6.2's beveled arm: a sector whose beveled edge is **not** an interpolation
/// seam keeps that sector's **own** rep — the seam gate short-circuits before the
/// §6.6.1.4 tie-break. Built with an interior (radial-2, untextured) beveled edge so the
/// edge is a non-seam, and with the next boundvert carrying a lower material slot so the
/// tie-break *would* pick a different face — proving the own rep, not the tie-break
/// winner, is what a non-seam beveled centre column takes.
#[test]
fn miter_ring_non_seam_beveled_centre_column_keeps_its_own_rep() {
    let mut mesh = LoopMesh::new();
    let v0 = mesh.create_vertex(Vec3::new(0.0, 0.0, -100.0));
    let a = mesh.create_vertex(Vec3::new(1.0, 0.0, 10.0));
    let b = mesh.create_vertex(Vec3::new(0.0, 1.0, 20.0));
    let c = mesh.create_vertex(Vec3::new(-1.0, 0.0, 30.0));
    // `{v0,b}` is shared by both faces — an interior radial-2 leg; with no loop layers it
    // is a non-seam. `{v0,c}` is a boundary leg of the next sector.
    let own = mesh.face_create_verts(&[v0, a, b], FaceCreateFlags::NONE);
    let next = mesh.face_create_verts(&[v0, b, c], FaceCreateFlags::NONE);
    mesh.set_face_material_index(own, 7);
    mesh.set_face_material_index(next, 3);

    let e_mid = mesh.query_edge_exists(v0, b).unwrap();
    let e_next = mesh.query_edge_exists(v0, c).unwrap();
    assert!(
        !mesh.edge_is_interp_seam(e_mid),
        "an interior untextured beveled edge is not a seam"
    );

    // Sector 0's own rep is `own` (7); the next boundvert reps `next` (3), so a tie-break
    // would pick the lower slot `next` — distinct from the own rep the non-seam gate keeps.
    assert_eq!(
        mesh.boundvert_representative(v0, e_mid),
        Some(own),
        "sector 0's own rep is `own`"
    );
    assert_eq!(
        mesh.donor_tie_break(&[own, next], None),
        Some(next),
        "fixture sanity: the tie-break would pick the lower-slot `next`",
    );

    let cross_edges = [Some(e_mid), Some(e_next)];
    let donors = mesh.miter_ring_centre_column_donors(v0, &cross_edges, None);
    assert_eq!(
        donors[0],
        Some(own),
        "a non-seam beveled sector's centre column keeps its own rep, not the tie-break winner",
    );
}

/// BEVEL.md §8.9.3.1: the inner-miter arc *end* `B_3` **owns** its trailing beveled edge
/// (its `Some` cross-edge), exactly like the outer miter's reflex-owning `B_3`, so on a
/// non-seam edge its centre column keeps that sector's **own** rep — it is **not** routed
/// through the §6.6.1.4 tie-break. Only the edge-less arc *start* `B_1` (a `None`
/// cross-edge) tie-breaks. This pins the corrected rule that overturns the earlier
/// far-flank premise (which wrongly read the inner `B_3` as edge-less): the next boundvert
/// carries a lower material slot, so a stray tie-break *would* pick a different face — the
/// own rep proves the tie-break did not fire.
#[test]
fn miter_ring_inner_arc_end_owns_its_edge_and_keeps_its_own_rep() {
    let mut mesh = LoopMesh::new();
    let v0 = mesh.create_vertex(Vec3::new(0.0, 0.0, -100.0));
    let a = mesh.create_vertex(Vec3::new(1.0, 0.0, 10.0));
    let b = mesh.create_vertex(Vec3::new(0.0, 1.0, 20.0));
    let c = mesh.create_vertex(Vec3::new(-1.0, 0.0, 30.0));
    let own = mesh.face_create_verts(&[v0, a, b], FaceCreateFlags::NONE);
    let next = mesh.face_create_verts(&[v0, b, c], FaceCreateFlags::NONE);
    mesh.set_face_material_index(own, 7);
    mesh.set_face_material_index(next, 3);

    let e_mid = mesh.query_edge_exists(v0, b).unwrap();
    let e_next = mesh.query_edge_exists(v0, c).unwrap();
    assert!(
        !mesh.edge_is_interp_seam(e_mid),
        "the beveled leg is a non-seam"
    );
    assert_eq!(
        mesh.donor_tie_break(&[own, next], None),
        Some(next),
        "fixture sanity: a stray tie-break would pick the lower-slot `next`",
    );

    // The inner `B_3` (a `Some` cross-edge that owns its trailing beveled edge) keeps its
    // own rep `own` (7) — the tie-break never fires on the non-seam edge, so the mirror
    // sectors of a two-material split junction legitimately differ (§8.9.3.1).
    let cross_edges = [Some(e_mid), Some(e_next)];
    let donors = mesh.miter_ring_centre_column_donors(v0, &cross_edges, None);
    assert_eq!(
        donors[0],
        Some(own),
        "the inner arc-end `B_3` owns its edge, so its non-seam centre column keeps its own rep",
    );
}

/// BEVEL.md §8.9.3.1 / §8.9.5.3 centre-n-gon row (and the §6.6.1.2 anchor row): a miter
/// junction's odd-`segments` centre n-gon takes the **enlarged miter ring's first
/// boundvert's** rep — the §6.6.1.2 fan anchor — on a **seam-free** junction. A material
/// boundary raises no interpolation seam, so a closed manifold fan whose faces carry
/// distinct slots stays on this branch: the centre polygon follows the enlarged ring's
/// seed, which **differs** from `ring_spanning_donor`'s raw-disk `first_incident_face`
/// (the two ring seedings diverge on a material-straddling corner — the whole point of
/// routing the outer miter through this donor rather than `ring_spanning_donor`).
#[test]
fn miter_centre_ngon_donor_material_only_junction_takes_the_enlarged_ring_anchor() {
    let mut mesh = LoopMesh::new();
    let v0 = mesh.create_vertex(Vec3::new(0.0, 0.0, -10.0));
    let r0 = mesh.create_vertex(Vec3::new(1.0, 0.0, 0.0));
    let r1 = mesh.create_vertex(Vec3::new(-0.5, 0.87, 0.0));
    let r2 = mesh.create_vertex(Vec3::new(-0.5, -0.87, 0.0));
    // Closed umbrella: every leg radial-2 manifold, no loop layers — so seam-free even
    // though the three faces carry distinct material slots.
    let f0 = mesh.face_create_verts(&[v0, r0, r1], FaceCreateFlags::NONE);
    let f1 = mesh.face_create_verts(&[v0, r1, r2], FaceCreateFlags::NONE);
    let f2 = mesh.face_create_verts(&[v0, r2, r0], FaceCreateFlags::NONE);
    mesh.set_face_material_index(f0, 1);
    mesh.set_face_material_index(f1, 2);
    mesh.set_face_material_index(f2, 3);
    assert!(
        !mesh.corner_is_seam_carrying(v0),
        "a closed manifold untextured fan raises no interpolation seam (material alone is none)",
    );

    let positional = mesh.first_incident_face(v0);
    let e0 = mesh.query_edge_exists(v0, r0).unwrap();
    let e1 = mesh.query_edge_exists(v0, r1).unwrap();
    let e2 = mesh.query_edge_exists(v0, r2).unwrap();
    // Seed the enlarged ring at a boundvert whose own rep differs from the raw disk head —
    // the §6.6.1.2 divergence the two donor paths disagree on. On a 3-distinct-face fan
    // such a leg always exists (the three legs' own reps are three distinct faces).
    let first_cross = [e0, e1, e2]
        .into_iter()
        .find(|&e| mesh.boundvert_representative(v0, e) != positional)
        .expect("some leg's own rep differs from the disk head on a 3-material fan");
    let rest: Vec<EdgeKey> = [e0, e1, e2]
        .into_iter()
        .filter(|&e| e != first_cross)
        .collect();
    let cross_edges = [Some(first_cross), Some(rest[0]), Some(rest[1])];
    let sector_donors = mesh.miter_ring_sector_donors(v0, &cross_edges);
    let anchor = sector_donors.first().copied().flatten();
    assert_ne!(
        anchor, positional,
        "fixture sanity: the enlarged-ring anchor diverges from the raw disk head",
    );

    let donor = mesh.miter_centre_ngon_donor(v0, &|_| true, BevelAffect::Edges, 3, anchor, None);
    assert_eq!(
        donor, anchor,
        "a seam-free material-only junction's centre n-gon takes the enlarged-ring anchor",
    );
    assert_eq!(
        donor, sector_donors[0],
        "the anchor is the ring's first boundvert's rep"
    );
    assert_ne!(
        donor, positional,
        "and it is NOT ring_spanning_donor's raw-disk `first_incident_face`",
    );
}

/// BEVEL.md §8.9.3.1 / §8.9.5.3 centre-n-gon row: on a **seamed** junction the centre
/// n-gon abandons the §6.6.1.2 enlarged-ring anchor and takes the junction-wide §6.6.1.4
/// tie-break — exactly `ring_spanning_donor`'s seamed branch. Built on the open-fan
/// (boundary-leg) fixture, whose legs are condition-(1) seams, and passed a deliberately
/// wrong anchor to prove the seamed branch drops it.
#[test]
fn miter_centre_ngon_donor_seamed_junction_routes_to_the_tie_break_winner() {
    let (mesh, v0, _e_prev, _e_next, own, next) = miter_centre_column_fan();
    assert!(
        mesh.corner_is_seam_carrying(v0),
        "the boundary-leg fan is seam-carrying (§6.6.1.4 condition 1)",
    );

    let winner = mesh.ring_spanning_donor(v0, &|_| true, BevelAffect::Edges, 3, None);
    assert_eq!(
        winner,
        Some(next),
        "fixture sanity: the junction-wide tie-break picks the lower-slot `next`",
    );

    // A deliberately-wrong anchor the seamed branch must ignore.
    let bogus_anchor = Some(own);
    let donor =
        mesh.miter_centre_ngon_donor(v0, &|_| true, BevelAffect::Edges, 3, bogus_anchor, None);
    assert_eq!(
        donor, winner,
        "a seamed junction routes the centre n-gon to the tie-break winner",
    );
    assert_ne!(
        donor, bogus_anchor,
        "the enlarged-ring anchor is dropped once the junction carries a seam",
    );
}
