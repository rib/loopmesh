use super::super::edge_flags::{DiskEdge, JunctionDisk};
use super::*;

/// The chamfer edge between the ring verts at `near_a` and `near_b`, by
/// position. Panics if either ring vert or the edge between them is absent.
fn chamfer_edge(mesh: &LoopMesh, near_a: Vec3, near_b: Vec3) -> EdgeKey {
    let a = vert_at(mesh, near_a);
    let b = vert_at(mesh, near_b);
    mesh.query_edge_exists(a, b)
        .expect("the chamfer edge between these two ring verts exists")
}

/// BEVEL.md §6.7.1.1: a corner segment (chamfer edge) carries the inherited
/// seam flag **only when both flanking owning edges are seamed**. A *single*
/// seamed incident edge of a beveled corner therefore flags **no** chamfer
/// edge — the contiguity AND suppresses the dead-end spur the plain §6.5.1 copy
/// would have left on the two chamfers touching that edge's ring vert.
#[test]
fn single_seamed_incident_edge_flags_no_chamfer() {
    let mut mesh = LoopMesh::new();
    mesh_add_cube(&mut mesh);

    // Seam only e0 = {v0, v1} of corner v0 = (1,-1,-1).
    let p0 = Vec3::new(1.0, -1.0, -1.0);
    let e0 = cube_edge(&mesh, p0, Vec3::new(-1.0, -1.0, -1.0));
    mesh.set_edge_seam(e0, true);

    let v0 = vert_at(&mesh, p0);
    mesh.bevel(
        &[v0.into()],
        BevelOptions {
            offset: 0.5,
            ..Default::default()
        },
    );

    // Ring verts: r_e0 = (0.5,-1,-1), r_e3 = (1,-0.5,-1), r_e8 = (1,-1,-0.5).
    let r_e0 = Vec3::new(0.5, -1.0, -1.0);
    let r_e3 = Vec3::new(1.0, -0.5, -1.0);
    let r_e8 = Vec3::new(1.0, -1.0, -0.5);

    // Every chamfer has exactly one seamed owner, so none is seamed (§6.7.1.1).
    for &(a, b) in &[(r_e0, r_e3), (r_e3, r_e8), (r_e8, r_e0)] {
        let chamfer = chamfer_edge(&mesh, a, b);
        assert!(
            !mesh.edge_seam(chamfer),
            "chamfer {a:?}-{b:?} has one seamed owner only; the §6.7.1.1 AND \
             suppresses the seam spur",
        );
    }

    // The surviving re-pointed e0 keeps its own seam (mechanism (1) item 1).
    let r_e0_vert = vert_at(&mesh, r_e0);
    assert!(
        mesh.edge_seam(e0) && mesh.edge_keys().contains(&e0),
        "the surviving re-pointed e0 keeps its seam",
    );
    assert!(
        mesh.edge_vert_keys(e0).contains(&r_e0_vert),
        "e0 is re-pointed onto its ring vert r_e0",
    );
    mesh.validate_topology();
}

/// BEVEL.md §6.7.1.1: two adjacent seamed incident edges flag the **one**
/// chamfer edge running directly between their two ring verts (its two flanking
/// owners are both seamed), and **only** that one — the other two chamfers each
/// have an unseamed owner and stay unseamed. The sharp bit follows the
/// identical AND on its own bit.
#[test]
fn two_adjacent_seamed_incident_edges_flag_their_one_chamfer() {
    let mut mesh = LoopMesh::new();
    mesh_add_cube(&mut mesh);

    let p0 = Vec3::new(1.0, -1.0, -1.0);
    let e0 = cube_edge(&mesh, p0, Vec3::new(-1.0, -1.0, -1.0)); // -> r_e0
    let e3 = cube_edge(&mesh, p0, Vec3::new(1.0, 1.0, -1.0)); // -> r_e3
                                                              // Seam e0 + e3; sharp the same pair to cover the sharp bit identically.
    mesh.set_edge_seam(e0, true);
    mesh.set_edge_seam(e3, true);
    mesh.set_edge_sharp(e0, true);
    mesh.set_edge_sharp(e3, true);

    let v0 = vert_at(&mesh, p0);
    mesh.bevel(
        &[v0.into()],
        BevelOptions {
            offset: 0.5,
            ..Default::default()
        },
    );

    let r_e0 = Vec3::new(0.5, -1.0, -1.0);
    let r_e3 = Vec3::new(1.0, -0.5, -1.0);
    let r_e8 = Vec3::new(1.0, -1.0, -0.5);

    // The chamfer between r_e0 and r_e3 has both owners (e0, e3) seamed/sharp.
    let between = chamfer_edge(&mesh, r_e0, r_e3);
    assert!(mesh.edge_seam(between), "the (r_e0,r_e3) chamfer is seamed");
    assert!(mesh.edge_sharp(between), "the (r_e0,r_e3) chamfer is sharp");

    // The other two chamfers each have one unseamed owner (e8) and stay clear.
    for &(a, b) in &[(r_e3, r_e8), (r_e8, r_e0)] {
        let chamfer = chamfer_edge(&mesh, a, b);
        assert!(
            !mesh.edge_seam(chamfer),
            "chamfer {a:?}-{b:?} has an unseamed owner; not seamed (§6.7.1.1)",
        );
        assert!(
            !mesh.edge_sharp(chamfer),
            "chamfer {a:?}-{b:?} has a smooth owner; not sharp (§6.7.1.1)",
        );
    }
    mesh.validate_topology();
}

/// BEVEL.md §6.7.1.1: all three incident edges seamed flags **every** chamfer
/// (each chamfer's two owners are both seamed) — the cap's ring is a continuous
/// seam loop.
#[test]
fn all_incident_edges_seamed_flag_every_chamfer() {
    let mut mesh = LoopMesh::new();
    mesh_add_cube(&mut mesh);

    let p0 = Vec3::new(1.0, -1.0, -1.0);
    for far in [
        Vec3::new(-1.0, -1.0, -1.0),
        Vec3::new(1.0, 1.0, -1.0),
        Vec3::new(1.0, -1.0, 1.0),
    ] {
        let edge = cube_edge(&mesh, p0, far);
        mesh.set_edge_seam(edge, true);
    }

    let v0 = vert_at(&mesh, p0);
    mesh.bevel(
        &[v0.into()],
        BevelOptions {
            offset: 0.5,
            ..Default::default()
        },
    );

    let r_e0 = Vec3::new(0.5, -1.0, -1.0);
    let r_e3 = Vec3::new(1.0, -0.5, -1.0);
    let r_e8 = Vec3::new(1.0, -1.0, -0.5);
    for &(a, b) in &[(r_e0, r_e3), (r_e3, r_e8), (r_e8, r_e0)] {
        let chamfer = chamfer_edge(&mesh, a, b);
        assert!(
            mesh.edge_seam(chamfer),
            "all owners seamed -> chamfer {a:?}-{b:?} is seamed (§6.7.1.1)",
        );
    }
    mesh.validate_topology();
}

/// BEVEL.md §6.7.2: `mark_seam` / `mark_sharp` are **inert** for vertex bevel —
/// no two beveled edges meet at the corner, so mechanism (2) never fires. The
/// parameter-ON run must produce the same topology *and* the same per-edge seam
/// / sharp flags as the parameter-OFF run; the flags ride mechanism (1)
/// identically either way.
#[test]
fn vertex_bevel_mark_seam_sharp_is_inert() {
    // Seam/sharp two adjacent incident edges so mechanism (1) *does* flag a
    // chamfer — the inertness claim must hold even when flags are present.
    let build = |mark: bool| -> Vec<((u32, u32, u32), (u32, u32, u32), bool, bool)> {
        let mut mesh = LoopMesh::new();
        mesh_add_cube(&mut mesh);
        let p0 = Vec3::new(1.0, -1.0, -1.0);
        let e0 = cube_edge(&mesh, p0, Vec3::new(-1.0, -1.0, -1.0));
        let e3 = cube_edge(&mesh, p0, Vec3::new(1.0, 1.0, -1.0));
        mesh.set_edge_seam(e0, true);
        mesh.set_edge_seam(e3, true);
        mesh.set_edge_sharp(e0, true);
        mesh.set_edge_sharp(e3, true);

        let v0 = vert_at(&mesh, p0);
        mesh.bevel(
            &[v0.into()],
            BevelOptions {
                offset: 0.5,
                mark_seam: mark,
                mark_sharp: mark,
                ..Default::default()
            },
        );

        // Read every edge's (endpoints, seam, sharp) so on/off are comparable
        // by geometry (keys differ run-to-run; positions are stable). Positions
        // are bit-cast to ordered integer keys for a stable sort.
        let key = |p: Vec3| (p.x.to_bits(), p.y.to_bits(), p.z.to_bits());
        let mut flags: Vec<((u32, u32, u32), (u32, u32, u32), bool, bool)> = mesh
            .edge_keys()
            .into_iter()
            .map(|e| {
                let [a, b] = mesh.edge_vert_keys(e);
                let mut ka = key(mesh.vertex_position(a));
                let mut kb = key(mesh.vertex_position(b));
                if kb < ka {
                    std::mem::swap(&mut ka, &mut kb);
                }
                (ka, kb, mesh.edge_seam(e), mesh.edge_sharp(e))
            })
            .collect();
        flags.sort();
        flags
    };

    let on = build(true);
    let off = build(false);
    assert_eq!(
        on, off,
        "mark_seam / mark_sharp are inert for vertex bevel (§6.7.2): on == off",
    );
}

/// BEVEL.md §6.7.3: `mark_seam` / `mark_sharp` are inert for an isolated single
/// beveled edge too — each endpoint sees one beveled edge, failing mechanism
/// (2)'s "two or more" precondition. Parameter-ON topology + flags must equal
/// parameter-OFF.
#[test]
fn isolated_edge_bevel_mark_seam_is_inert() {
    let build = |mark: bool| -> Vec<((u32, u32, u32), (u32, u32, u32), bool)> {
        let mut mesh = LoopMesh::new();
        mesh_add_cube(&mut mesh);
        let edge = cube_edge(
            &mesh,
            Vec3::new(1.0, -1.0, -1.0),
            Vec3::new(-1.0, -1.0, -1.0),
        );
        mesh.set_edge_seam(edge, true);

        mesh.bevel(
            &[edge.into()],
            BevelOptions {
                offset: 0.25,
                affect: BevelAffect::Edges,
                mark_seam: mark,
                ..Default::default()
            },
        );

        let key = |p: Vec3| (p.x.to_bits(), p.y.to_bits(), p.z.to_bits());
        let mut flags: Vec<((u32, u32, u32), (u32, u32, u32), bool)> = mesh
            .edge_keys()
            .into_iter()
            .map(|e| {
                let [a, b] = mesh.edge_vert_keys(e);
                let mut ka = key(mesh.vertex_position(a));
                let mut kb = key(mesh.vertex_position(b));
                if kb < ka {
                    std::mem::swap(&mut ka, &mut kb);
                }
                (ka, kb, mesh.edge_seam(e))
            })
            .collect();
        flags.sort();
        flags
    };

    assert_eq!(
        build(true),
        build(false),
        "mark_seam is inert for an isolated single beveled edge (§6.7.3)",
    );
}

/// Round a position to a stable integer key for set membership (the bevel
/// positions here are exact multiples of 0.25 up to f32 noise).
fn pos_key(p: Vec3) -> (i32, i32, i32) {
    let q = |x: f32| (x * 4.0).round() as i32;
    (q(p.x), q(p.y), q(p.z))
}

/// An undirected edge as the sorted pair of its endpoint position keys.
fn edge_pos_pair(mesh: &LoopMesh, edge: EdgeKey) -> ((i32, i32, i32), (i32, i32, i32)) {
    let [a, b] = mesh.edge_vert_keys(edge);
    let mut ka = pos_key(mesh.vertex_position(a));
    let mut kb = pos_key(mesh.vertex_position(b));
    if kb < ka {
        std::mem::swap(&mut ka, &mut kb);
    }
    (ka, kb)
}

/// The set of NEW (result) edges carrying `flag` (seam or sharp), as sorted
/// endpoint position-key pairs — an ordering-independent fingerprint.
fn flagged_new_edge_set(
    mesh: &LoopMesh,
    result: &BevelResult,
    flag: impl Fn(&LoopMesh, EdgeKey) -> bool,
) -> std::collections::BTreeSet<((i32, i32, i32), (i32, i32, i32))> {
    result
        .edges
        .iter()
        .filter(|&&e| mesh.edge_keys().contains(&e) && flag(mesh, e))
        .map(|&e| edge_pos_pair(mesh, e))
        .collect()
}

/// The three-edge cube-corner junction the bridging arm pins (BEVEL.md §6.7.4):
/// v0 = [1,-1,-1], edges e01/e03/e04 beveled at offset 0.25, with the two
/// flanking edges e01 (toward [-1,-1,-1]) and e04 (toward [1,-1,1]) flagged and
/// e03 (toward [1,1,-1]) the unflagged gap. Returns the beveled mesh + result.
fn bevel_three_edge_junction_with_flag(
    set_flag: impl Fn(&mut LoopMesh, EdgeKey),
    options: BevelOptions,
) -> (LoopMesh, BevelResult) {
    let mut mesh = LoopMesh::new();
    mesh_add_cube(&mut mesh);
    let p0 = Vec3::new(1.0, -1.0, -1.0);
    let e01 = cube_edge(&mesh, p0, Vec3::new(-1.0, -1.0, -1.0));
    let e03 = cube_edge(&mesh, p0, Vec3::new(1.0, 1.0, -1.0));
    let e04 = cube_edge(&mesh, p0, Vec3::new(1.0, -1.0, 1.0));
    set_flag(&mut mesh, e01);
    set_flag(&mut mesh, e04);
    let result = mesh.bevel(&[e01.into(), e03.into(), e04.into()], options);
    (mesh, result)
}

type EdgeKeySet = std::collections::BTreeSet<((i32, i32, i32), (i32, i32, i32))>;

struct JunctionExpectedSets {
    /// All three patch ring segments (the full boundvert ring).
    ring: EdgeKeySet,
    /// The single direct patch segment spanning the unseamed e03 gap edge.
    gap_segment: EdgeKeySet,
    /// The four parallel strip long edges of the seamed e01 / e04 (mechanism 1).
    parallel: EdgeKeySet,
}

/// The cube corner v0's boundvert ring (the three patch ring positions at
/// offset 0.25), the single direct patch segment spanning the e03 gap, and the
/// four parallel strip long edges of the two seamed beveled edges e01 / e04.
/// Position keys are in 0.25 units (`pos_key`).
fn junction_expected_sets() -> JunctionExpectedSets {
    let k = |x: f32, y: f32, z: f32| pos_key(Vec3::new(x, y, z));
    let pair = |a: (i32, i32, i32), b: (i32, i32, i32)| {
        if a <= b {
            (a, b)
        } else {
            (b, a)
        }
    };
    // Boundverts of v0's junction at offset 0.25.
    let b0 = k(0.75, -0.75, -1.0);
    let b1 = k(0.75, -1.0, -0.75);
    let b2 = k(1.0, -0.75, -0.75);
    // The three patch ring segments.
    let ring: std::collections::BTreeSet<_> = [pair(b0, b1), pair(b0, b2), pair(b1, b2)]
        .into_iter()
        .collect();
    // The §6.7.4 single direct patch segment spanning the unseamed gap edge e03:
    // the edge joining e03's two flanking boundverts (both carry y=-0.75, the
    // slide along e03), i.e. [0.75,-0.75,-1] and [1,-0.75,-0.75].
    let gap_segment: std::collections::BTreeSet<_> = [pair(b0, b2)].into_iter().collect();
    // The four parallel strip long edges of e01 (to its two v1-strip verts) and
    // e04 (to its two v4-strip verts) — mechanism (1).
    let parallel: std::collections::BTreeSet<_> = [
        pair(b0, k(-1.0, -0.75, -1.0)),
        pair(b1, k(-1.0, -1.0, -0.75)),
        pair(b1, k(0.75, -1.0, 1.0)),
        pair(b2, k(1.0, -0.75, 1.0)),
    ]
    .into_iter()
    .collect();
    JunctionExpectedSets {
        ring,
        gap_segment,
        parallel,
    }
}

/// BEVEL.md §6.7.1 mechanism (1) is UNCONDITIONAL — with `mark_seam = false`
/// the two seamed beveled edges still seam their four parallel strip long edges
/// (the edges descending from their re-stitched faces). The patch ring segments
/// stay unseamed (no parameter-gated bridging). This pins the unconditional
/// inheritance path independently of the parameter.
#[test]
fn junction_mechanism_one_inherits_with_mark_seam_off() {
    let (mesh, result) = bevel_three_edge_junction_with_flag(
        |m, e| m.set_edge_seam(e, true),
        BevelOptions {
            offset: 0.25,
            affect: BevelAffect::Edges,
            mark_seam: false,
            ..Default::default()
        },
    );
    let sets = junction_expected_sets();
    let seamed = flagged_new_edge_set(&mesh, &result, |m, e| m.edge_seam(e));
    assert_eq!(
        seamed, sets.parallel,
        "mark_seam=false still seams exactly the four parallel strip long edges \
         of e01/e04 via mechanism (1) (§6.7.1); no patch ring segment is seamed",
    );
    assert!(
        seamed.is_disjoint(&sets.ring),
        "no patch ring (bridging) segment may be seamed with mark_seam off",
    );
    mesh.validate_topology();
}

/// BEVEL.md §6.7.4 mechanism (2): with `mark_seam = true` the seam bridges the
/// e03 gap by the **single** direct patch segment spanning e03 (the edge joining
/// e03's two flanking boundverts) — *not* the full ring. The other two patch
/// sides (spanning the seamed e01 / e04, where their strips already meet) stay
/// unflagged. Pins the exact bridged set = four mechanism-(1) parallel edges +
/// the one gap segment, and asserts the over-bridging full-ring result is wrong.
#[test]
fn junction_mechanism_two_bridges_single_gap_segment_with_mark_seam_on() {
    let (mesh, result) = bevel_three_edge_junction_with_flag(
        |m, e| m.set_edge_seam(e, true),
        BevelOptions {
            offset: 0.25,
            affect: BevelAffect::Edges,
            mark_seam: true,
            ..Default::default()
        },
    );
    let sets = junction_expected_sets();
    let mut expected = sets.parallel.clone();
    expected.extend(sets.gap_segment.iter().copied());
    let seamed = flagged_new_edge_set(&mesh, &result, |m, e| m.edge_seam(e));
    assert_eq!(
        seamed, expected,
        "mark_seam=true seams the four mechanism-(1) parallel edges PLUS the \
         single direct patch segment spanning the unseamed e03 gap (§6.7.4) — \
         NOT the full ring",
    );
    // The two patch sides spanning the seamed e01 / e04 stay unflagged.
    let other_ring: std::collections::BTreeSet<_> =
        sets.ring.difference(&sets.gap_segment).copied().collect();
    assert!(
        seamed.is_disjoint(&other_ring),
        "the patch sides spanning the seamed edges' own sides are not bridged",
    );
    mesh.validate_topology();
}

/// BEVEL.md §6.7.4: `mark_sharp` runs the identical single-gap-segment bridging
/// on the sharp bit, independently of `mark_seam`. On bridges the one gap segment
/// (+ four parallel); off bridges nothing (only the four parallel, mechanism 1).
#[test]
fn junction_mark_sharp_bridges_single_gap_segment_and_inherits_unconditionally() {
    let sets = junction_expected_sets();

    let (mesh_off, result_off) = bevel_three_edge_junction_with_flag(
        |m, e| m.set_edge_sharp(e, true),
        BevelOptions {
            offset: 0.25,
            affect: BevelAffect::Edges,
            mark_sharp: false,
            ..Default::default()
        },
    );
    let sharp_off = flagged_new_edge_set(&mesh_off, &result_off, |m, e| m.edge_sharp(e));
    assert_eq!(
        sharp_off, sets.parallel,
        "mark_sharp=false still sharpens the four parallel strip long edges \
         (mechanism (1), §6.7.1), no ring segment",
    );

    let (mesh_on, result_on) = bevel_three_edge_junction_with_flag(
        |m, e| m.set_edge_sharp(e, true),
        BevelOptions {
            offset: 0.25,
            affect: BevelAffect::Edges,
            mark_sharp: true,
            ..Default::default()
        },
    );
    let mut expected = sets.parallel.clone();
    expected.extend(sets.gap_segment.iter().copied());
    let sharp_on = flagged_new_edge_set(&mesh_on, &result_on, |m, e| m.edge_sharp(e));
    assert_eq!(
        sharp_on, expected,
        "mark_sharp=true bridges the single gap segment on the sharp bit (§6.7.4)",
    );
    mesh_off.validate_topology();
    mesh_on.validate_topology();
}

/// Drive `bridge_junction_flags` directly over a synthetic `s`-edge junction:
/// build a ring of `s` boundverts joined by `s` patch ring segments, plus `s`
/// stub "beveled edges" `g_0..g_(s-1)` whose seamed state is `seamed[k]`
/// (`opening_edges[k] = g_k`, so segment `(B_(k-1), B_k)` is crossed by `g_k`).
/// Returns the set of ring segment indices `m` (segment `(B_(m-1), B_m)`) that
/// end up seamed. This exercises the §6.7.4 gap rule independent of the full
/// junction surgery (which the ≥4-edge regime does not yet drive end-to-end).
fn bridge_synthetic_junction(seamed: &[bool]) -> std::collections::BTreeSet<usize> {
    let s = seamed.len();
    let mut mesh = LoopMesh::new();
    // Ring boundverts on a unit circle, plus the ring edges between consecutive
    // boundverts; opening_edges are separate stub edges to detached stub verts.
    let mut boundverts = Vec::with_capacity(s);
    for k in 0..s {
        let theta = std::f32::consts::TAU * (k as f32) / (s as f32);
        boundverts.push(mesh.create_vertex(Vec3::new(theta.cos(), theta.sin(), 0.0)));
    }
    for k in 0..s {
        mesh.create_edge(
            [boundverts[k], boundverts[(k + 1) % s]],
            EdgeCreateFlags::NONE,
        );
    }
    // opening_edges[k] = g_k: a stub edge whose seam state encodes `seamed[k]`.
    let mut opening_edges = Vec::with_capacity(s);
    for k in 0..s {
        let stub = mesh.create_vertex(Vec3::new(2.0 + k as f32, 0.0, 0.0));
        let g = mesh.create_edge([boundverts[k], stub], EdgeCreateFlags::NONE);
        opening_edges.push(g);
    }

    // A leg-free disk in the same rotational order as the ring: every entry beveled,
    // so the walk's counting and delimiting tests coincide (§6.7.5.3 step 1).
    let disk = JunctionDisk {
        vertex: boundverts[0],
        edges: (0..s)
            .map(|k| DiskEdge {
                edge: opening_edges[k],
                beveled: true,
                seam: seamed[k],
                sharp: false,
                ring_vertex: None,
            })
            .collect(),
    };
    let patch = JunctionPatch {
        vertex: boundverts[0],
        boundverts: boundverts.clone(),
        opening_edges: opening_edges.into_iter().map(Some).collect(),
        representative: FaceTemplate::default(),
    };
    mesh.bridge_junction_flags(&patch.ring_chains(), &disk, true, false);

    // Read back which ring segment (B_(m-1), B_m) ended up seamed.
    (0..s)
        .filter(|&m| {
            let a = boundverts[(m + s - 1) % s];
            let b = boundverts[m];
            mesh.query_edge_exists(a, b)
                .map(|e| mesh.edge_seam(e))
                .unwrap_or(false)
        })
        .collect()
}

/// BEVEL.md §6.7.4 ≥4-edge generalization. Each case pins which patch ring
/// segments the gap rule bridges (segment `m` spans gap-edge `g_m`).
#[test]
fn junction_gap_rule_generalizes_beyond_three_edges() {
    // Two OPPOSITE seamed edges (g0,g2) at a valence-4 junction enclose two
    // separate length-1 gaps (g1 and g3); BOTH are bridged (the seam closes into
    // a loop). Segments spanning g1 and g3 are seamed; nothing else.
    assert_eq!(
        bridge_synthetic_junction(&[true, false, true, false]),
        [1, 3].into_iter().collect(),
        "two opposite seamed edges bridge BOTH length-1 gaps, not the full ring",
    );

    // Two ADJACENT seamed edges (g0,g1): the short side between them (the sector
    // boundvert they share) is a length-0 gap and bridges nothing; the far
    // length-2 gap (g2,g3) is bridged by its two consecutive direct segments.
    assert_eq!(
        bridge_synthetic_junction(&[true, true, false, false]),
        [2, 3].into_iter().collect(),
        "two adjacent seamed edges bridge only the far length-2 gap (2 segments)",
    );

    // THREE seamed edges (one unseamed g3): a single length-1 gap; only the
    // segment spanning g3 is bridged.
    assert_eq!(
        bridge_synthetic_junction(&[true, true, true, false]),
        [3].into_iter().collect(),
        "three seamed edges bridge the one length-1 gap",
    );

    // The 3-edge cube-corner case: g1 the unseamed gap flanked by g0,g2 seamed —
    // exactly the segment spanning g1 is bridged (the single direct segment, NOT
    // the full ring).
    assert_eq!(
        bridge_synthetic_junction(&[true, false, true]),
        [1].into_iter().collect(),
        "single-gap 3-edge junction bridges exactly one segment (no over-bridge)",
    );

    // No gap: a lone seamed edge bridges nothing.
    assert!(bridge_synthetic_junction(&[true, false, false]).is_empty());
    // No gap: all seamed bridges nothing (the strips already carry the seam).
    assert!(bridge_synthetic_junction(&[true, true, true]).is_empty());
    // No gap: none seamed bridges nothing.
    assert!(bridge_synthetic_junction(&[false, false, false]).is_empty());
}

// ---- mark_seam / mark_sharp, multi-segment (BEVEL.md §6.7.5) ----

/// The same three-edge cube-corner junction as `bevel_three_edge_junction_with_flag`
/// but at `segments ≥ 2` with the ADJ rounded fill (BEVEL.md §6.7.5): v0 = [1,-1,-1],
/// e01/e03/e04 beveled at offset 0.25, e01 (toward [-1,-1,-1]) and e04 (toward
/// [1,-1,1]) flagged, e03 (toward [1,1,-1]) the unflagged gap between them.
fn bevel_three_edge_junction_multisegment_with_flag(
    set_flag: impl Fn(&mut LoopMesh, EdgeKey),
    segments: u32,
    mark_seam: bool,
    mark_sharp: bool,
) -> (LoopMesh, BevelResult) {
    let mut mesh = LoopMesh::new();
    mesh_add_cube(&mut mesh);
    let p0 = Vec3::new(1.0, -1.0, -1.0);
    let e01 = cube_edge(&mesh, p0, Vec3::new(-1.0, -1.0, -1.0));
    let e03 = cube_edge(&mesh, p0, Vec3::new(1.0, 1.0, -1.0));
    let e04 = cube_edge(&mesh, p0, Vec3::new(1.0, -1.0, 1.0));
    set_flag(&mut mesh, e01);
    set_flag(&mut mesh, e04);
    let result = mesh.bevel(
        &[e01.into(), e03.into(), e04.into()],
        BevelOptions {
            offset: 0.25,
            affect: BevelAffect::Edges,
            segments,
            vmesh_method: BevelVmeshMethod::Adj,
            mark_seam,
            mark_sharp,
            ..Default::default()
        },
    );
    (mesh, result)
}

/// The two outermost-ring cross-section edges bridging the unseamed e03 gap at
/// `segments = 2` (BEVEL.md §6.7.5.1 / §6.7.5.3): the run `(b0, M)`, `(M, b2)`
/// where b0 / b2 are e03's two flanking boundverts and M is the profile chain's
/// single interior arc sample on the corner sphere (centre (0.75,-0.75,-0.75),
/// radius 0.25). Both the arc midpoint and the chord midpoint fall in the same
/// `pos_key` (0.25-grid) bucket, so the bridge set is robust to the exact profile.
fn e03_gap_chain_segments2() -> EdgeKeySet {
    let k = |x: f32, y: f32, z: f32| pos_key(Vec3::new(x, y, z));
    let pair = |a: (i32, i32, i32), b: (i32, i32, i32)| if a <= b { (a, b) } else { (b, a) };
    let b0 = k(0.75, -0.75, -1.0);
    let b2 = k(1.0, -0.75, -0.75);
    let center = Vec3::new(0.75, -0.75, -0.75);
    let mid = center + 0.25 * Vec3::new(1.0, 0.0, -1.0).normalize();
    let m = pos_key(mid);
    [pair(b0, m), pair(m, b2)].into_iter().collect()
}

/// BEVEL.md §6.7.5.2 mechanism (1) at `segments = 2` is UNCONDITIONAL: with
/// `mark_seam = false` each seamed beveled edge still stamps its flag onto its
/// strip's **two outer longitudinal edges** (index 0 and index `segments`) — the
/// same four strip-boundary edges as the single-segment case, since the interior
/// longitudinal line (index 1) is never inherited. No cross-section chain edge is
/// flagged (no parameter-gated bridging).
#[test]
fn multisegment_mechanism_one_flags_two_outer_longitudinal_edges() {
    let (mesh, result) = bevel_three_edge_junction_multisegment_with_flag(
        |m, e| m.set_edge_seam(e, true),
        2,
        false,
        false,
    );
    let sets = junction_expected_sets();
    let seamed = flagged_new_edge_set(&mesh, &result, |m, e| m.edge_seam(e));
    assert_eq!(
        seamed, sets.parallel,
        "mechanism (1) flags exactly the two outer longitudinal edges of each \
         seamed strip (four total across e01/e04, §6.7.5.2) — the interior \
         longitudinal line and every cross-section chain edge stay unflagged",
    );
    assert!(
        seamed.is_disjoint(&e03_gap_chain_segments2()),
        "no cross-section bridge chain is flagged with mark_seam off",
    );
    mesh.validate_topology();
}

/// BEVEL.md §6.7.5.3 mechanism (2) at `segments = 2`: with `mark_seam = true` the
/// e03 gap is bridged by its whole outermost-ring profile chain — a **run of two**
/// cross-section edges (`segments = 2`), not the single patch segment of the
/// `segments = 1` case. The flagged set is exactly the four mechanism-(1)
/// longitudinal edges plus that two-edge chain; the seamed edges' own chains and
/// every interior vmesh-ring edge stay unflagged.
#[test]
fn multisegment_mechanism_two_bridges_gap_as_run_of_segments_edges() {
    let (mesh, result) = bevel_three_edge_junction_multisegment_with_flag(
        |m, e| m.set_edge_seam(e, true),
        2,
        true,
        false,
    );
    let sets = junction_expected_sets();
    let bridge = e03_gap_chain_segments2();
    let mut expected = sets.parallel.clone();
    expected.extend(bridge.iter().copied());
    let seamed = flagged_new_edge_set(&mesh, &result, |m, e| m.edge_seam(e));
    assert_eq!(
        seamed, expected,
        "mark_seam=true seams the four mechanism-(1) longitudinal edges PLUS the \
         two-edge outermost-ring chain spanning the e03 gap (§6.7.5.3) — a run of \
         `segments` edges, not one",
    );
    assert_eq!(
        bridge.len(),
        2,
        "the length-1 gap bridge is a run of segments=2 edges"
    );
    mesh.validate_topology();
}

/// BEVEL.md §6.7.5.3: `mark_sharp` runs the identical run-bridge on the sharp bit,
/// independent of `mark_seam`. Off flags only the four mechanism-(1) longitudinal
/// edges (unconditional); on adds the two-edge e03 gap chain.
#[test]
fn multisegment_mark_sharp_bridges_gap_run_independently() {
    let sets = junction_expected_sets();
    let bridge = e03_gap_chain_segments2();

    let (mesh_off, result_off) = bevel_three_edge_junction_multisegment_with_flag(
        |m, e| m.set_edge_sharp(e, true),
        2,
        false,
        false,
    );
    let sharp_off = flagged_new_edge_set(&mesh_off, &result_off, |m, e| m.edge_sharp(e));
    assert_eq!(
        sharp_off, sets.parallel,
        "mark_sharp=false still sharpens the two outer longitudinal edges of each \
         seamed strip (mechanism (1), §6.7.5.2), no chain",
    );

    let (mesh_on, result_on) = bevel_three_edge_junction_multisegment_with_flag(
        |m, e| m.set_edge_sharp(e, true),
        2,
        false,
        true,
    );
    let mut expected = sets.parallel.clone();
    expected.extend(bridge.iter().copied());
    let sharp_on = flagged_new_edge_set(&mesh_on, &result_on, |m, e| m.edge_sharp(e));
    assert_eq!(
        sharp_on, expected,
        "mark_sharp=true bridges the two-edge e03 gap chain on the sharp bit (§6.7.5.3)",
    );
    mesh_off.validate_topology();
    mesh_on.validate_topology();
}

/// BEVEL.md §6.7.5.4: the terminal far ends are INERT at `segments ≥ 2` — only
/// mechanism (1)'s two outer longitudinal edges of the through-strip carry the
/// flag; the mirrored far-end cross-section chain is **not** bridged, even with
/// `mark_seam` on. e01 / e04 run into valence-3 terminals; their far-end
/// cross-section chains (between the two terminal strip verts through the terminal
/// interior sample) must stay unflagged.
#[test]
fn multisegment_terminal_far_end_is_inert() {
    let (mesh, result) = bevel_three_edge_junction_multisegment_with_flag(
        |m, e| m.set_edge_seam(e, true),
        2,
        true,
        false,
    );
    let seamed = flagged_new_edge_set(&mesh, &result, |m, e| m.edge_seam(e));

    // e01's terminal at [-1,-1,-1]: strip verts [-1,-0.75,-1] and [-1,-1,-0.75],
    // with the terminal cross-section chain running between them (through a single
    // interior sample). e04's terminal at [1,-1,1]: strip verts [0.75,-1,1] and
    // [1,-0.75,1]. None of these far-end chain edges may be flagged.
    let k = |x: f32, y: f32, z: f32| pos_key(Vec3::new(x, y, z));
    let pair = |a: (i32, i32, i32), b: (i32, i32, i32)| if a <= b { (a, b) } else { (b, a) };
    let terminal_chain = |p: Vec3, q: Vec3| -> EdgeKeySet {
        let mid = 0.5 * (p + q);
        let m = pos_key(mid);
        let pk = pos_key(p);
        let qk = pos_key(q);
        [pair(pk, m), pair(m, qk)].into_iter().collect()
    };
    let e01_terminal = terminal_chain(Vec3::new(-1.0, -0.75, -1.0), Vec3::new(-1.0, -1.0, -0.75));
    let e04_terminal = terminal_chain(Vec3::new(0.75, -1.0, 1.0), Vec3::new(1.0, -0.75, 1.0));
    assert!(
        seamed.is_disjoint(&e01_terminal),
        "e01's far-end cross-section chain is not bridged at the terminal (§6.7.5.4)",
    );
    assert!(
        seamed.is_disjoint(&e04_terminal),
        "e04's far-end cross-section chain is not bridged at the terminal (§6.7.5.4)",
    );
    mesh.validate_topology();
}

/// Bevel a bunch of the octagonal bipyramid's apex legs (`segments = 2`,
/// `offset = 0.2`, `vmesh_method = ADJ`, circular profile) with a chosen
/// `miter_outer`, seaming / sharpening the named apex-leg indices and passing
/// the corresponding `mark_seam` / `mark_sharp`. `beveled` / `seamed` /
/// `sharped` are apex-leg indices (`k` = leg `A–R_(k+1)`, so scaffold edge
/// `{0, k+1}` = `apex_leg[k]`). Drives BEVEL.md §6.7.5.4 case (4).
fn bevel_octagonal_reflex_flagged(
    beveled: &[usize],
    seamed: &[usize],
    sharped: &[usize],
    miter_outer: BevelMiter,
    mark_seam: bool,
    mark_sharp: bool,
) -> (LoopMesh, BevelResult) {
    let (mut mesh, _apex, _rim, apex_leg) = octagonal_bipyramid_all_legs();
    for &k in seamed {
        mesh.set_edge_seam(apex_leg[k], true);
    }
    for &k in sharped {
        mesh.set_edge_sharp(apex_leg[k], true);
    }
    let geom: Vec<GeomKey> = beveled.iter().map(|&k| apex_leg[k].into()).collect();
    let result = mesh.bevel(
        &geom,
        BevelOptions {
            offset: 0.2,
            affect: BevelAffect::Edges,
            segments: 2,
            vmesh_method: BevelVmeshMethod::Adj,
            miter_outer,
            mark_seam,
            mark_sharp,
            ..Default::default()
        },
    );
    (mesh, result)
}

/// BEVEL.md §6.7.5.4 case (4): a **multi-face reflex outer `ARC` miter** does not
/// desync the mechanism-(2) bunch-side bridge. Octagonal apex, legs `A–R1..A–R4`
/// beveled (a `< 180°` bunch, `> 180°` reflex sector absorbing legs 5..8); the two
/// flanking legs `A–R1` / `A–R4` seamed, the interior `A–R2` / `A–R3` the length-2
/// gap. `mark_seam` ON must bridge that gap as its own `2 gap edges × 2 segments =
/// 4`-edge run on the bunch side, exactly as at a miter-free junction — the outer
/// miter's inserted reflex-arc boundverts fall outside every bridge span.
#[test]
fn multiface_reflex_arc_seam_gap_bridge_contiguous() {
    let beveled = [0, 1, 2, 3];
    let flanks = [0, 3];
    let (on, on_r) =
        bevel_octagonal_reflex_flagged(&beveled, &flanks, &[], BevelMiter::Arc, true, false);
    let (off, off_r) =
        bevel_octagonal_reflex_flagged(&beveled, &flanks, &[], BevelMiter::Arc, false, false);
    let (bare, bare_r) =
        bevel_octagonal_reflex_flagged(&beveled, &[], &[], BevelMiter::Arc, true, false);

    let on_count = count_flagged_result_edges(&on, &on_r, |m, e| m.edge_seam(e));
    let off_count = count_flagged_result_edges(&off, &off_r, |m, e| m.edge_seam(e));
    let bare_count = count_flagged_result_edges(&bare, &bare_r, |m, e| m.edge_seam(e));

    assert_eq!(
        on_count - off_count,
        4,
        "mark_seam must add exactly the interior gap's `2 × 2 = 4`-edge bunch-side \
         run across a multi-face reflex ARC miter (§6.7.5.4 case (4))",
    );
    assert!(
        on_count > off_count,
        "the bridge must be non-vacuous (on differs from off)"
    );
    assert_eq!(
        bare_count, 0,
        "no seamed input ⇒ no gap and no mechanism-(1) flag: the ARC fill stays inert",
    );
    on.validate_topology();
    off.validate_topology();
}

/// BEVEL.md §6.7.5.4 case (4): the `mark_sharp` companion of the ARC contiguous
/// gap bridge — the run-bridge fires on the sharp bit independently and the
/// multi-face reflex ARC miter leaves it inert on the wide arc.
#[test]
fn multiface_reflex_arc_sharp_gap_bridge_contiguous() {
    let beveled = [0, 1, 2, 3];
    let flanks = [0, 3];
    let (on, on_r) =
        bevel_octagonal_reflex_flagged(&beveled, &[], &flanks, BevelMiter::Arc, false, true);
    let (off, off_r) =
        bevel_octagonal_reflex_flagged(&beveled, &[], &flanks, BevelMiter::Arc, false, false);

    let on_count = count_flagged_result_edges(&on, &on_r, |m, e| m.edge_sharp(e));
    let off_count = count_flagged_result_edges(&off, &off_r, |m, e| m.edge_sharp(e));
    assert_eq!(
        on_count - off_count,
        4,
        "mark_sharp must add exactly the interior gap's `2 × 2 = 4`-edge bunch-side \
         run across a multi-face reflex ARC miter (§6.7.5.4 case (4))",
    );
    on.validate_topology();
    off.validate_topology();
}

/// BEVEL.md §6.7.5.4 case (4): the mechanism-(2) walk STEPS OVER an absorbed
/// unbeveled leg. Octagonal apex, legs `A–R1`, `A–R3`, `A–R4` beveled with `A–R2`
/// unbeveled (absorbed into the convex `A–R1`–`A–R3` sector); `A–R1` / `A–R4`
/// seamed, the interior `A–R3` the length-1 gap. The bridge from `A–R1` reaches
/// the gap `A–R3` skipping the absorbed `A–R2` (unbeveled legs are not counted
/// toward gap length) and flags `A–R3`'s own `1 × 2 = 2`-edge run and nothing more.
#[test]
fn multiface_reflex_arc_seam_gap_bridge_absorbed_leg() {
    let beveled = [0, 2, 3];
    let flanks = [0, 3];
    let (on, on_r) =
        bevel_octagonal_reflex_flagged(&beveled, &flanks, &[], BevelMiter::Arc, true, false);
    let (off, off_r) =
        bevel_octagonal_reflex_flagged(&beveled, &flanks, &[], BevelMiter::Arc, false, false);

    let on_count = count_flagged_result_edges(&on, &on_r, |m, e| m.edge_seam(e));
    let off_count = count_flagged_result_edges(&off, &off_r, |m, e| m.edge_seam(e));
    assert_eq!(
        on_count - off_count,
        2,
        "mark_seam must bridge the length-1 gap over the absorbed leg as its own \
         `1 × 2 = 2`-edge run and nothing more (§6.7.5.4 case (4))",
    );
    on.validate_topology();
    off.validate_topology();
}

/// BEVEL.md §6.7.5.4 case (4): the strongest desync check — a multi-face reflex
/// outer `PATCH` miter inserts the MOST boundverts on the wide reflex arc (the
/// patch-middle plus patch-end, §8.9.6), yet the bunch-side contiguous gap still
/// bridges to exactly `2 × 2 = 4` edges and no patch fill edge is flagged.
#[test]
fn multiface_reflex_patch_seam_gap_bridge_contiguous() {
    let beveled = [0, 1, 2, 3];
    let flanks = [0, 3];
    let (on, on_r) =
        bevel_octagonal_reflex_flagged(&beveled, &flanks, &[], BevelMiter::Patch, true, false);
    let (off, off_r) =
        bevel_octagonal_reflex_flagged(&beveled, &flanks, &[], BevelMiter::Patch, false, false);

    let on_count = count_flagged_result_edges(&on, &on_r, |m, e| m.edge_seam(e));
    let off_count = count_flagged_result_edges(&off, &off_r, |m, e| m.edge_seam(e));
    assert_eq!(
        on_count - off_count,
        4,
        "mark_seam must add exactly `2 × 2 = 4` bunch-side seam edges under a \
         multi-face reflex PATCH miter and flag no patch fill edge (§6.7.5.4 case (4))",
    );
    on.validate_topology();
    off.validate_topology();
}

/// BEVEL.md §6.7.5.4 case (4): the degenerate baseline — a multi-face reflex outer
/// `SHARP` miter inserts NO reflex-arc boundverts. The bunch-side contiguous gap
/// bridges to `2 × 2 = 4` edges through the plain cutoff / adj junction fill path.
#[test]
fn multiface_reflex_sharp_miter_seam_gap_bridge_contiguous() {
    let beveled = [0, 1, 2, 3];
    let flanks = [0, 3];
    let (on, on_r) =
        bevel_octagonal_reflex_flagged(&beveled, &flanks, &[], BevelMiter::Sharp, true, false);
    let (off, off_r) =
        bevel_octagonal_reflex_flagged(&beveled, &flanks, &[], BevelMiter::Sharp, false, false);

    let on_count = count_flagged_result_edges(&on, &on_r, |m, e| m.edge_seam(e));
    let off_count = count_flagged_result_edges(&off, &off_r, |m, e| m.edge_seam(e));
    assert_eq!(
        on_count - off_count,
        4,
        "mark_seam must add exactly `2 × 2 = 4` bunch-side seam edges under a \
         multi-face reflex SHARP miter (§6.7.5.4 case (4))",
    );
    on.validate_topology();
    off.validate_topology();
}

/// BEVEL.md §6.7.5.4 case (5), the **long-span desync**: mechanism (2) counts a gap
/// in beveled edges but steps it out in **ring boundverts**, so a span whose ring
/// path enters a **split** sector burns a step on that sector's fill chain and comes
/// up short at its far end.
///
/// Same octagonal apex and same four beveled legs as case (4)'s arms, with the flags
/// moved to the **interior** pair: legs `A–R2` / `A–R3` seamed, so the gap is
/// `{A–R4, A–R1}` — length `m = 2`, straddling the reflex sector — and the span's
/// anchor `A–R3` never touches that sector. `SHARP` leaves the sector unsplit and the
/// walk stays in registration: the full `m · segments = 4`-edge bridge lands. `ARC`
/// splits it, so of the span's two chain-steps one lands on the gap's own strip chain
/// and the other is burned on the reflex fill chain — whose per-face AND is FALSE
/// (both flanks unflagged), so the corner-run pass clears it again. Net: **half** the
/// idealised bridge, and never a flagged edge in the wrong place.
#[test]
fn multiface_reflex_arc_desyncs_long_span_while_sharp_bridges_it() {
    let beveled = [0, 1, 2, 3];
    let interior = [1, 2];

    let (arc_on, arc_on_r) =
        bevel_octagonal_reflex_flagged(&beveled, &interior, &[], BevelMiter::Arc, true, false);
    let (arc_off, arc_off_r) =
        bevel_octagonal_reflex_flagged(&beveled, &interior, &[], BevelMiter::Arc, false, false);
    let (sharp_on, sharp_on_r) =
        bevel_octagonal_reflex_flagged(&beveled, &interior, &[], BevelMiter::Sharp, true, false);
    let (sharp_off, sharp_off_r) =
        bevel_octagonal_reflex_flagged(&beveled, &interior, &[], BevelMiter::Sharp, false, false);

    let seam = |m: &LoopMesh, e: EdgeKey| m.edge_seam(e);
    let arc_on_count = count_flagged_result_edges(&arc_on, &arc_on_r, seam);
    let arc_off_count = count_flagged_result_edges(&arc_off, &arc_off_r, seam);
    let sharp_on_count = count_flagged_result_edges(&sharp_on, &sharp_on_r, seam);
    let sharp_off_count = count_flagged_result_edges(&sharp_off, &sharp_off_r, seam);

    assert_eq!(
        arc_off_count, sharp_off_count,
        "the two runs' OFF baselines are mechanism (1)'s longitudinals alone, so the \
         boundvert split must not move them",
    );
    assert_eq!(
        arc_off_count, 4,
        "OFF: the two seamed strips' `2 × segments = 4` outer longitudinal edges \
         (§6.7.5.2), and no fill edge (the reflex flanks are unflagged)",
    );
    assert_eq!(
        sharp_on_count - sharp_off_count,
        4,
        "SHARP: the sector is unsplit, so the `m = 2` span steps the two gap edges' \
         own chains — the full `m · segments = 4`-edge bridge (§6.7.5.4 case (5))",
    );
    assert_eq!(
        arc_on_count - arc_off_count,
        2,
        "ARC: the split sector costs the span one of its two steps, so only the first \
         gap edge's `segments = 2` chain is bridged and the far one is left unflagged \
         (§6.7.5.4 case (5))",
    );

    // The desync is *missing* bridge edges, never misplaced ones: the walk's burned
    // step lands on the reflex fill chain, which is that sector's rebuilt face's
    // corner run, and the corner-run pass then clears it. So ON adds edges to OFF and
    // never moves one — the OFF set survives intact inside the ON set.
    let arc_on_flagged = flagged_new_edge_set(&arc_on, &arc_on_r, seam);
    let arc_off_flagged = flagged_new_edge_set(&arc_off, &arc_off_r, seam);
    assert!(
        arc_off_flagged.is_subset(&arc_on_flagged),
        "the desynced span must not disturb mechanism (1)'s longitudinal edges",
    );
    assert_eq!(
        arc_on_flagged.len() - arc_off_flagged.len(),
        2,
        "ON adds exactly the one bridged chain, and nothing on the fill chain",
    );

    arc_on.validate_topology();
    sharp_on.validate_topology();
}

/// The `mark_sharp` companion of the long-span desync (BEVEL.md §6.7.5.4 case (5)):
/// the desynchronisation is a property of the ring walk, not of the bit it carries.
#[test]
fn multiface_reflex_arc_desyncs_long_span_on_the_sharp_bit() {
    let beveled = [0, 1, 2, 3];
    let interior = [1, 2];

    let (arc_on, arc_on_r) =
        bevel_octagonal_reflex_flagged(&beveled, &[], &interior, BevelMiter::Arc, false, true);
    let (arc_off, arc_off_r) =
        bevel_octagonal_reflex_flagged(&beveled, &[], &interior, BevelMiter::Arc, false, false);
    let (sharp_on, sharp_on_r) =
        bevel_octagonal_reflex_flagged(&beveled, &[], &interior, BevelMiter::Sharp, false, true);
    let (sharp_off, sharp_off_r) =
        bevel_octagonal_reflex_flagged(&beveled, &[], &interior, BevelMiter::Sharp, false, false);

    let sharp_bit = |m: &LoopMesh, e: EdgeKey| m.edge_sharp(e);
    let arc_delta = count_flagged_result_edges(&arc_on, &arc_on_r, sharp_bit)
        - count_flagged_result_edges(&arc_off, &arc_off_r, sharp_bit);
    let sharp_delta = count_flagged_result_edges(&sharp_on, &sharp_on_r, sharp_bit)
        - count_flagged_result_edges(&sharp_off, &sharp_off_r, sharp_bit);

    assert_eq!(
        sharp_delta, 4,
        "SHARP outer miter: the `m = 2` span bridges in full on the sharp bit",
    );
    assert_eq!(
        arc_delta, 2,
        "ARC outer miter: the split sector desyncs the same span on the sharp bit too",
    );
    arc_on.validate_topology();
}

// ---- BEVEL.md §6.7.5.4 case (4b): the **flagged unbeveled leg** ----
//
// The octagonal apex with `g_1 … g_4` = apex legs 0..=3 beveled and the `> 180°`
// reflex sector absorbing the four unbeveled legs `L_5 … L_8` = apex legs 4..=7.
// Whole-mesh flag counts (the guide's read-back table): a surviving flagged leg
// contributes 1 of its own, each flagged beveled edge 2 outer longitudinal edges,
// mechanism (1)'s corner-run AND whatever fill run a doubly-flagged rebuilt face
// owns, and mechanism (2) the bridge spans.

/// Mechanism (2), the **delimiting** half: a flagged leg ends the run that reaches
/// it, so a junction whose only flagged *beveled* edge is `g_1` — which alone would
/// abort on the walk's termination clause — bridges after all once `L_6` is flagged
/// too. `m = 3` (`g_2, g_3, g_4`), so the bridge lays `m · segments = 6` edges. The
/// span runs entirely on the bunch side, so all three outer-miter modes agree.
#[test]
fn flagged_absorbed_leg_anchors_bridge_span_octagonal() {
    let beveled = [0, 1, 2, 3];
    let flags = [0, 5]; // g_1 and the absorbed leg L_6
    for miter in [BevelMiter::Sharp, BevelMiter::Arc, BevelMiter::Patch] {
        let (on, _) = bevel_octagonal_reflex_flagged(&beveled, &flags, &[], miter, true, false);
        let (off, _) = bevel_octagonal_reflex_flagged(&beveled, &flags, &[], miter, false, false);
        assert_eq!(
            seamed_edge_count(&off),
            3,
            "{miter:?}: OFF is mechanism (1) alone — g_1's two longitudinals plus the \
             surviving flagged leg",
        );
        assert_eq!(
            seamed_edge_count(&on),
            9,
            "{miter:?}: the flagged leg delimits g_1's run at m = 3, so the bridge lays \
             m · segments = 6 edges (§6.7.5.4 case (4b))",
        );
        on.validate_topology();
    }
}

/// The `mark_sharp` companion: the leg's delimiting is a property of the walk, not
/// of the bit it carries.
#[test]
fn flagged_absorbed_leg_anchors_bridge_span_on_the_sharp_bit_octagonal() {
    let beveled = [0, 1, 2, 3];
    let flags = [0, 5];
    let (on, _) =
        bevel_octagonal_reflex_flagged(&beveled, &[], &flags, BevelMiter::Arc, false, true);
    let (off, _) =
        bevel_octagonal_reflex_flagged(&beveled, &[], &flags, BevelMiter::Arc, false, false);
    let count = |mesh: &LoopMesh| sharp_edges(mesh).len();
    assert_eq!(
        count(&off),
        3,
        "OFF: two longitudinals plus the surviving flagged leg"
    );
    assert_eq!(
        count(&on),
        9,
        "ON: the leg-delimited m = 3 span bridges six edges"
    );
    on.validate_topology();
}

/// The contrast that makes the two bridge arms attributable to the **leg** rather
/// than to the fixture: leave the leg unflagged and the lone flagged beveled edge
/// (either `g_1` or `g_4`) finds no second flagged disk edge, so the walk's
/// termination clause aborts and nothing is bridged — ON ≡ OFF at 2 edges.
#[test]
fn unflagged_absorbed_leg_leaves_lone_flagged_edge_inert_octagonal() {
    let beveled = [0, 1, 2, 3];
    for lone in [0usize, 3] {
        for miter in [BevelMiter::Sharp, BevelMiter::Arc, BevelMiter::Patch] {
            let (on, _) =
                bevel_octagonal_reflex_flagged(&beveled, &[lone], &[], miter, true, false);
            let (off, _) =
                bevel_octagonal_reflex_flagged(&beveled, &[lone], &[], miter, false, false);
            assert_eq!(
                seamed_edge_count(&on),
                2,
                "{miter:?}: a lone flagged beveled edge bridges nothing (§6.7.5.4 case (5)'s \
                 termination clause)",
            );
            assert_eq!(
                seamed_edge_count(&off),
                seamed_edge_count(&on),
                "{miter:?}: with no second flagged disk edge the parameter is inert",
            );
        }
    }
}

/// Mechanism (2), the **anchoring** half, and case (5)'s desync reached from a leg
/// anchor. Flag `g_4` and `L_6`: `g_4`'s run hits `L_6` at once (`m = 0`), so the
/// junction's only span is anchored at the **leg**, laid from its ring vertex, with
/// `m = 3` (`g_1, g_2, g_3`). Under `SHARP` the leg's ring vertex is the sector's
/// single boundvert and the walk is in registration (`+6`); under `ARC` / `PATCH`
/// it is the split sector's *leading* ring vertex, so the span's first step is burnt
/// on the fill chain — whose per-face AND is FALSE — and only two chains survive
/// (`+4`).
#[test]
fn flagged_leg_anchored_span_desyncs_at_split_sector_octagonal() {
    let beveled = [0, 1, 2, 3];
    let flags = [3, 5]; // g_4 and the absorbed leg L_6
    for (miter, expected) in [
        (BevelMiter::Sharp, 9usize),
        (BevelMiter::Arc, 7),
        (BevelMiter::Patch, 7),
    ] {
        let (on, _) = bevel_octagonal_reflex_flagged(&beveled, &flags, &[], miter, true, false);
        let (off, _) = bevel_octagonal_reflex_flagged(&beveled, &flags, &[], miter, false, false);
        assert_eq!(
            seamed_edge_count(&off),
            3,
            "{miter:?}: OFF is g_4's two longitudinals plus the surviving flagged leg",
        );
        assert_eq!(
            seamed_edge_count(&on),
            expected,
            "{miter:?}: the leg-anchored span desyncs iff its ring path enters a split \
             sector (§6.7.5.4 case (4b) / case (5))",
        );
        on.validate_topology();
    }
}

/// **KNOWN GAP (BEVEL.md §6.7.5.4 case (4b) pins no partial-chain span).** A flagged
/// leg anchors its span at its ring vertex (case (4b)), and the walk lays whole ring
/// **chains** from there — so a leg whose ring vertex is an *interior* sample of a
/// chain begins no chain and lays **nothing at all**, even though it still delimits.
///
/// This test establishes that the shape is **reachable**, not hypothetical. Bevel
/// three of the octagonal apex's eight legs (`g_1 … g_3`) and the `> 180°` reflex
/// sector absorbs **five** legs — an *odd* count, so §8.9.2's assignment puts the
/// middle one (`L_6`) on profile-sample index `segments / 2`: at `segments = 2` the
/// arc's lone interior bulge `s_1`, asserted below against an independent re-derivation
/// of the arc samples. Flag `g_3` and `L_6`: `g_3`'s run hits `L_6` at once (`m = 0`),
/// so the junction's only span is the leg-anchored one, of length `m = 2` (`g_1, g_2`).
///
/// The three outer-miter modes then separate the cause cleanly:
///
/// - `SHARP` — the sector is unsplit, the leg's ring vertex is its single boundvert
///   (a chain start), and the span lands in full: `m · segments = 4` edges.
/// - `PATCH` — the leg's ring vertex is the hub `B2`, still a chain start, so the span
///   *starts*; its first step is then burnt on a patch fill chain and cleared by the
///   corner-run pass — the ordinary case-(5) desync, `+2`.
/// - `ARC` — the ring vertex is `s_1`, **interior** to the reflex arc chain. No chain
///   begins there, so loopmesh drops the whole span: ON ≡ OFF. This is the divergence
///   candidate; the guide does not say what a partial-chain span should lay.
#[test]
fn flagged_middle_absorbed_leg_on_interior_arc_sample_lays_no_span_octagonal() {
    let beveled = [0, 1, 2];

    // The middle absorbed leg really does land on the arc's interior bulge sample.
    let (s0, s1, s2) = octagonal_arc_samples(&beveled);
    let (mut mesh, _apex, rim, apex_leg) = octagonal_bipyramid_all_legs();
    let geom: Vec<GeomKey> = beveled.iter().map(|&k| apex_leg[k].into()).collect();
    mesh.bevel(
        &geom,
        BevelOptions {
            offset: 0.2,
            affect: BevelAffect::Edges,
            segments: 2,
            vmesh_method: BevelVmeshMethod::Adj,
            miter_outer: BevelMiter::Arc,
            ..Default::default()
        },
    );
    let [end_a, end_b] = mesh.edge_vert_keys(apex_leg[5]);
    let landing = if end_b == rim[5] { end_a } else { end_b };
    let landing_position = mesh.vertex_position(landing);
    assert!(
        (landing_position - s1).length() < 1e-4,
        "the middle absorbed leg re-points onto the arc's interior sample s_1 \
         (§8.9.2's `segments / 2` assignment), not onto either arc end",
    );
    assert!((landing_position - s0).length() > 1e-3);
    assert!((landing_position - s2).length() > 1e-3);

    let flags = [2, 5]; // g_3 and the middle absorbed leg L_6
    for (miter, expected) in [
        (BevelMiter::Sharp, 7usize),
        (BevelMiter::Patch, 5),
        (BevelMiter::Arc, 3),
    ] {
        let (on, _) = bevel_octagonal_reflex_flagged(&beveled, &flags, &[], miter, true, false);
        let (off, _) = bevel_octagonal_reflex_flagged(&beveled, &flags, &[], miter, false, false);
        assert_eq!(
            seamed_edge_count(&off),
            3,
            "{miter:?}: OFF is g_3's two longitudinals plus the surviving flagged leg",
        );
        assert_eq!(
            seamed_edge_count(&on),
            expected,
            "{miter:?}: the leg-anchored m = 2 span lands in full when its ring vertex \
             starts a chain, and vanishes when the vertex is interior to one",
        );
        on.validate_topology();
    }
}

/// Mechanism (1) through a **leg pair**: under `ARC` the fan face bounded by `L_6`
/// and `L_7` owns the whole fill run (their ring vertices are the sector's two
/// boundverts), so flagging both legs passes that face's corner-run AND and the
/// entire `segments`-edge fill chain lights up — with `mark_seam` **off**. Under
/// `PATCH` the same two legs share the hub, their fan face's run is empty, and no
/// fill edge is flagged: the mirror-negative that shows the rule is the face
/// partition and not the fill.
#[test]
fn flagged_leg_pair_flags_arc_fill_run_octagonal() {
    let beveled = [0, 1, 2, 3];
    let flags = [0, 3, 5, 6]; // g_1, g_4, L_6, L_7
    let (arc_on, _) =
        bevel_octagonal_reflex_flagged(&beveled, &flags, &[], BevelMiter::Arc, true, false);
    let (arc_off, _) =
        bevel_octagonal_reflex_flagged(&beveled, &flags, &[], BevelMiter::Arc, false, false);
    let (patch_on, _) =
        bevel_octagonal_reflex_flagged(&beveled, &flags, &[], BevelMiter::Patch, true, false);
    let (patch_off, _) =
        bevel_octagonal_reflex_flagged(&beveled, &flags, &[], BevelMiter::Patch, false, false);

    assert_eq!(
        seamed_edge_count(&arc_off),
        8,
        "ARC / OFF: 4 longitudinals + 2 surviving legs + the 2-edge fill run the \
         leg-bounded fan face's AND passes on (§6.7.5.4 case (4b))",
    );
    assert_eq!(
        seamed_edge_count(&arc_on),
        12,
        "ARC / ON adds the two bridge spans"
    );
    assert_eq!(
        seamed_edge_count(&patch_off),
        6,
        "PATCH / OFF: no fill edge — both legs share the hub, so their fan face's run \
         is empty",
    );
    assert_eq!(
        seamed_edge_count(&patch_on),
        10,
        "PATCH / ON adds the same two bridge spans"
    );
    arc_off.validate_topology();
    patch_off.validate_topology();
}

/// Mechanism (1) through a **flank + its nearest leg**: under `PATCH` the leading
/// flank face is bounded by `g_4` and `L_5`, and its corner run is that flank's
/// patch leg (`B_1 → B_2`), so flagging both lights those `segments` edges with the
/// parameter off. Under `ARC` the same flags light nothing — flank and leg share the
/// end sample `B_1`, so that face's run is empty.
#[test]
fn flagged_leg_and_flank_flags_patch_leg_octagonal() {
    let beveled = [0, 1, 2, 3];
    let flags = [0, 3, 4]; // g_1, g_4, L_5
    let (patch_on, _) =
        bevel_octagonal_reflex_flagged(&beveled, &flags, &[], BevelMiter::Patch, true, false);
    let (patch_off, _) =
        bevel_octagonal_reflex_flagged(&beveled, &flags, &[], BevelMiter::Patch, false, false);
    let (arc_on, _) =
        bevel_octagonal_reflex_flagged(&beveled, &flags, &[], BevelMiter::Arc, true, false);
    let (arc_off, _) =
        bevel_octagonal_reflex_flagged(&beveled, &flags, &[], BevelMiter::Arc, false, false);

    assert_eq!(
        seamed_edge_count(&patch_off),
        7,
        "PATCH / OFF: 4 longitudinals + 1 surviving leg + the flank face's 2-edge patch \
         leg (§6.7.5.4 case (4b))",
    );
    assert_eq!(
        seamed_edge_count(&patch_on),
        11,
        "PATCH / ON adds the two bridge spans"
    );
    assert_eq!(
        seamed_edge_count(&arc_off),
        5,
        "ARC / OFF: no fill edge — the flank face's run collapses to the shared end \
         sample B_1",
    );
    assert_eq!(
        seamed_edge_count(&arc_on),
        9,
        "ARC / ON adds the same two bridge spans"
    );
    patch_off.validate_topology();
    arc_off.validate_topology();
}

/// BEVEL.md §6.7.5.4 (terminal bullet), the `segments ≥ 2` lift: the valence-3
/// terminal's absorbing face runs the beveled edge's **far-end cross-section chain**
/// as its corner run, against the terminal's two unbeveled legs. Both legs flagged ⇒
/// **every** edge of the `segments`-edge chain rides the flag, so the terminal
/// contributes `2 + segments`. Unconditional: identical with `mark_seam` on and off.
#[test]
fn terminal_far_end_chain_flagged_when_both_legs_flagged_multisegment() {
    for segments in [2u32, 3] {
        for mark_seam in [false, true] {
            let mut mesh = LoopMesh::new();
            mesh_add_cube(&mut mesh);
            let beveled = cube_edge(
                &mesh,
                Vec3::new(1.0, -1.0, -1.0),
                Vec3::new(-1.0, -1.0, -1.0),
            );
            let front_leg = cube_edge(&mesh, Vec3::new(1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, -1.0));
            let bottom_leg =
                cube_edge(&mesh, Vec3::new(1.0, -1.0, -1.0), Vec3::new(1.0, -1.0, 1.0));
            mesh.set_edge_seam(front_leg, true);
            mesh.set_edge_seam(bottom_leg, true);
            mesh.set_edge_sharp(front_leg, true);
            mesh.set_edge_sharp(bottom_leg, true);

            mesh.bevel(
                &[beveled.into()],
                BevelOptions {
                    offset: 0.25,
                    segments,
                    affect: BevelAffect::Edges,
                    mark_seam,
                    mark_sharp: mark_seam,
                    ..Default::default()
                },
            );
            mesh.validate_topology();

            assert_eq!(
                seamed_edge_count(&mesh),
                2 + segments as usize,
                "segments={segments}, mark_seam={mark_seam}: the two surviving legs plus \
                 every edge of the far-end cross-section chain (§6.7.5.4 terminal bullet)",
            );
            assert_eq!(
                sharp_edges(&mesh).len(),
                2 + segments as usize,
                "segments={segments}: the sharp bit runs the identical AND",
            );
        }
    }
}

/// The mandatory AND-vs-OR contrast for the row above: **one** flagged leg leaves the
/// far-end chain clean at every `segments` (`1 + 1 = 2 + segments`, not `2`).
#[test]
fn terminal_far_end_chain_clean_when_one_leg_flagged_multisegment() {
    for segments in [2u32, 3] {
        let mut mesh = LoopMesh::new();
        mesh_add_cube(&mut mesh);
        let beveled = cube_edge(
            &mesh,
            Vec3::new(1.0, -1.0, -1.0),
            Vec3::new(-1.0, -1.0, -1.0),
        );
        let bottom_leg = cube_edge(&mesh, Vec3::new(1.0, -1.0, -1.0), Vec3::new(1.0, -1.0, 1.0));
        mesh.set_edge_seam(bottom_leg, true);

        mesh.bevel(
            &[beveled.into()],
            BevelOptions {
                offset: 0.25,
                segments,
                affect: BevelAffect::Edges,
                mark_seam: true,
                ..Default::default()
            },
        );
        mesh.validate_topology();

        assert_eq!(
            seamed_edges(&mesh),
            vec![bottom_leg],
            "segments={segments}: one flagged leg ⇒ the corner-run AND fails and the chain \
             stays clean",
        );
    }
}

/// Bevel the three apex edges of `single_face_reflex_junction` (`segments = 2`,
/// `offset = 0.2`, `vmesh_method = ADJ`) with the named apex edges seamed. The
/// reflex sector is the wide quad `v–c–w–a`, so its two `v`-incident originals are
/// the flanking beveled edges `va` (index 0) and `vc` (index 2); `vb` (index 1) is
/// the narrow (bunch-side) edge between them. Drives BEVEL.md §6.7.5.4 cases
/// (2)–(3).
fn bevel_single_face_reflex_seamed(
    seamed: &[usize],
    miter_outer: BevelMiter,
    mark_seam: bool,
) -> LoopMesh {
    let (mut mesh, _apex, apex_edges) = single_face_reflex_junction();
    for &k in seamed {
        mesh.set_edge_seam(apex_edges[k], true);
    }
    mesh.bevel(
        &apex_edges.map(GeomKey::from),
        BevelOptions {
            offset: 0.2,
            affect: BevelAffect::Edges,
            segments: 2,
            vmesh_method: BevelVmeshMethod::Adj,
            miter_outer,
            mark_seam,
            ..Default::default()
        },
    );
    mesh
}

/// Every live seamed edge of the mesh.
fn seamed_edge_count(mesh: &LoopMesh) -> usize {
    mesh.edge_keys()
        .into_iter()
        .filter(|&e| mesh.edge_seam(e))
        .count()
}

/// BEVEL.md §6.7.5.4 case (3): at a **single-face** reflex `ARC` miter the reflex
/// sector's one wide face has the miter's fill chain as its **corner run**, and its
/// two `v`-incident originals are the two flanking beveled edges. Both flanks
/// seamed ⇒ the §6.7.1.1 AND passes ⇒ **every** fill-chain edge is seamed. With
/// `mark_seam` ON the junction carries `4` outer longitudinal (mechanism (1)) + `2`
/// gap-bridge (mechanism (2), `1 × segments` across the unseamed `vb`) + `2`
/// arc-fill (the corner run) = **8** seam edges.
#[test]
fn single_face_reflex_arc_fill_chain_flagged_when_both_flanks_seamed() {
    let mesh = bevel_single_face_reflex_seamed(&[0, 2], BevelMiter::Arc, true);
    assert_eq!(
        seamed_edge_count(&mesh),
        8,
        "4 outer longitudinal + 2 gap-bridge + 2 arc-fill seam edges (§6.7.5.4 case (3))",
    );
    mesh.validate_topology();
}

/// BEVEL.md §6.7.5.4 case (3): the corner-run AND is **mechanism (1)** and so is
/// *not* gated by `mark_seam`. Turning the parameter off removes only the two
/// gap-bridge edges across the unseamed narrow edge — the four outer longitudinal
/// edges and the **two arc-fill edges** stay seamed (`8 − 2 = 6`). This is the
/// sharpest available demonstration that the fill chain is not parameter-gated.
#[test]
fn single_face_reflex_arc_fill_chain_fires_with_mark_seam_off() {
    let on = bevel_single_face_reflex_seamed(&[0, 2], BevelMiter::Arc, true);
    let off = bevel_single_face_reflex_seamed(&[0, 2], BevelMiter::Arc, false);
    assert_eq!(
        seamed_edge_count(&on) - seamed_edge_count(&off),
        2,
        "mark_seam adds only the `1 × segments = 2`-edge bunch-side gap bridge",
    );
    assert_eq!(
        seamed_edge_count(&off),
        6,
        "with the parameter OFF mechanism (1) still flags the 4 outer longitudinal \
         edges AND the 2 arc-fill corner-run edges (§6.7.5.4 case (3))",
    );
    off.validate_topology();
}

/// BEVEL.md §6.7.5.4 case (2): a **single** seamed edge into the same reflex `ARC`
/// miter junction leaves both mechanisms inert on the fill. Mechanism (2) never
/// runs (a lone flagged edge finds no second one, so there is no gap); the corner-run
/// AND fails (only one of the reflex face's two flanking originals is seamed). The
/// seam runs up the seamed strip's two outer longitudinal edges and stops — 2 seam
/// edges, and identical with `mark_seam` off.
#[test]
fn single_seamed_edge_into_reflex_arc_miter_is_inert() {
    let on = bevel_single_face_reflex_seamed(&[0], BevelMiter::Arc, true);
    let off = bevel_single_face_reflex_seamed(&[0], BevelMiter::Arc, false);
    assert_eq!(
        seamed_edge_count(&on),
        2,
        "one seamed flank flags only its strip's two outer longitudinal edges: the \
         arc fill's corner-run AND fails and no gap exists (§6.7.5.4 case (2))",
    );
    assert_eq!(
        seamed_edge_count(&off),
        seamed_edge_count(&on),
        "with one seamed edge the parameter is inert (§6.7.5.4 case (2))",
    );
    on.validate_topology();
}

/// BEVEL.md §6.7.1.1(a): the face-provenance half of the corner-segment test. An
/// **all-seamed** three-edge junction must flag **no** patch ring segment — every
/// patch side lies between the patch fill and a strip, both new, so it is not a
/// corner segment however its two boundverts trace back. An implementation that
/// tests only endpoint ownership over-flags the whole boundvert ring here.
#[test]
fn all_seamed_junction_does_not_overflag_the_patch_ring() {
    let mut mesh = LoopMesh::new();
    mesh_add_cube(&mut mesh);
    let p0 = Vec3::new(1.0, -1.0, -1.0);
    let e01 = cube_edge(&mesh, p0, Vec3::new(-1.0, -1.0, -1.0));
    let e03 = cube_edge(&mesh, p0, Vec3::new(1.0, 1.0, -1.0));
    let e04 = cube_edge(&mesh, p0, Vec3::new(1.0, -1.0, 1.0));
    for edge in [e01, e03, e04] {
        mesh.set_edge_seam(edge, true);
    }
    let result = mesh.bevel(
        &[e01.into(), e03.into(), e04.into()],
        BevelOptions {
            offset: 0.25,
            affect: BevelAffect::Edges,
            mark_seam: true,
            ..Default::default()
        },
    );

    let sets = junction_expected_sets();
    let seamed = flagged_new_edge_set(&mesh, &result, |m, e| m.edge_seam(e));
    assert!(
        seamed.is_disjoint(&sets.ring),
        "an all-seamed junction bridges nothing: no patch ring segment is a corner \
         segment (§6.7.1.1(a) / §6.7.4)",
    );
    assert!(
        !seamed.is_empty(),
        "the seamed beveled edges still flag their strip long edges (mechanism (1))",
    );
    mesh.validate_topology();
}

fn seamed_edges(mesh: &LoopMesh) -> Vec<EdgeKey> {
    mesh.edge_keys()
        .into_iter()
        .filter(|&e| mesh.edge_seam(e))
        .collect()
}

fn sharp_edges(mesh: &LoopMesh) -> Vec<EdgeKey> {
    mesh.edge_keys()
        .into_iter()
        .filter(|&e| mesh.edge_sharp(e))
        .collect()
}

fn flag_vert_at(mesh: &LoopMesh, position: Vec3) -> VertexKey {
    mesh.vert_keys()
        .into_iter()
        .find(|&v| (mesh.vertex_position(v) - position).length() < 1.0e-3)
        .unwrap_or_else(|| panic!("a vert sits at {position:?}"))
}

/// BEVEL.md §6.7.5.2: a beveled edge's flag rides **both** outer longitudinal
/// edges of its strip — the two the re-stitched adjacent faces rebuilt along `E`'s
/// original sides — at *every* `segments`, junction or not. Mechanism (1) is
/// unconditional, so this fires with `mark_seam` / `mark_sharp` off (they are inert
/// for a lone edge, §6.7.3). The end chords across the strip descend from no
/// original edge and stay clean.
#[test]
fn lone_beveled_edge_flag_rides_both_outer_longitudinal_edges() {
    for segments in [1u32, 2, 3] {
        let mut mesh = LoopMesh::new();
        mesh_add_cube(&mut mesh);
        let beveled = cube_edge(
            &mesh,
            Vec3::new(1.0, -1.0, -1.0),
            Vec3::new(-1.0, -1.0, -1.0),
        );
        mesh.set_edge_seam(beveled, true);
        mesh.set_edge_sharp(beveled, true);

        mesh.bevel(
            &[beveled.into()],
            BevelOptions {
                offset: 0.25,
                segments,
                affect: BevelAffect::Edges,
                ..Default::default()
            },
        );
        mesh.validate_topology();

        let seams = seamed_edges(&mesh);
        assert_eq!(
            seams.len(),
            2,
            "segments={segments}: both outer longitudinal edges seamed, nothing else",
        );
        assert_eq!(
            sharp_edges(&mesh).len(),
            2,
            "segments={segments}: the sharp bit runs the identical copy",
        );
        for seam in seams {
            let [p, q] = mesh.edge_vert_keys(seam);
            let span = (mesh.vertex_position(p).x - mesh.vertex_position(q).x).abs();
            assert!(
                span > 1.9,
                "segments={segments}: a flagged edge runs the strip's whole length, not across it",
            );
        }
    }
}

/// BEVEL.md §6.5 item 1 / §6.7.1 item 1: an unbeveled leg *survives* an edge bevel
/// — it is re-pointed onto its near-corner vert, keeping the same `EdgeKey` and so
/// its whole per-edge attribute block, seam / sharp bits included. Pinned on the
/// terminal path (a lone beveled edge) at every `segments`.
#[test]
fn unbeveled_leg_keeps_its_flags_and_identity_across_a_terminal() {
    for segments in [1u32, 2, 3] {
        let mut mesh = LoopMesh::new();
        mesh_add_cube(&mut mesh);
        let beveled = cube_edge(
            &mesh,
            Vec3::new(1.0, -1.0, -1.0),
            Vec3::new(-1.0, -1.0, -1.0),
        );
        let leg = cube_edge(&mesh, Vec3::new(1.0, -1.0, -1.0), Vec3::new(1.0, -1.0, 1.0));
        let untouched = cube_edge(&mesh, Vec3::new(-1.0, 1.0, -1.0), Vec3::new(-1.0, 1.0, 1.0));
        mesh.set_edge_seam(leg, true);
        mesh.set_edge_sharp(leg, true);

        mesh.bevel(
            &[beveled.into()],
            BevelOptions {
                offset: 0.25,
                segments,
                affect: BevelAffect::Edges,
                ..Default::default()
            },
        );
        mesh.validate_topology();

        assert_eq!(
            seamed_edges(&mesh),
            vec![leg],
            "segments={segments}: the seam stays on the same leg edge, and spreads nowhere",
        );
        assert_eq!(
            sharp_edges(&mesh),
            vec![leg],
            "segments={segments}: same for sharp",
        );

        let far = flag_vert_at(&mesh, Vec3::new(1.0, -1.0, 1.0));
        let near = flag_vert_at(&mesh, Vec3::new(1.0, -1.0, -0.75));
        assert_eq!(
            mesh.query_edge_exists(near, far),
            Some(leg),
            "segments={segments}: the leg is re-pointed onto the bottom-side strip vert",
        );
        assert!(
            !mesh.edge_seam(untouched) && !mesh.edge_sharp(untouched),
            "segments={segments}: a far edge is untouched",
        );
    }
}

/// BEVEL.md §6.7.1.1(a′) / §6.7.5.4 (terminal bullet): the terminal's absorbing
/// face runs its far-end cross-section chain against the terminal's **two unbeveled
/// legs** — the chain is flagged iff *both* legs are flagged. Positive arm: both
/// legs seamed, so the absorbed cross edge joins them, giving three seamed edges.
/// Unconditional — `mark_seam` is off here.
#[test]
fn terminal_absorbed_corner_flags_cross_edge_when_both_legs_flagged() {
    let mut mesh = LoopMesh::new();
    mesh_add_cube(&mut mesh);
    let beveled = cube_edge(
        &mesh,
        Vec3::new(1.0, -1.0, -1.0),
        Vec3::new(-1.0, -1.0, -1.0),
    );
    let front_leg = cube_edge(&mesh, Vec3::new(1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, -1.0));
    let bottom_leg = cube_edge(&mesh, Vec3::new(1.0, -1.0, -1.0), Vec3::new(1.0, -1.0, 1.0));
    mesh.set_edge_seam(front_leg, true);
    mesh.set_edge_seam(bottom_leg, true);

    mesh.bevel(
        &[beveled.into()],
        BevelOptions {
            offset: 0.25,
            affect: BevelAffect::Edges,
            ..Default::default()
        },
    );
    mesh.validate_topology();

    let cross = mesh
        .query_edge_exists(
            flag_vert_at(&mesh, Vec3::new(1.0, -0.75, -1.0)),
            flag_vert_at(&mesh, Vec3::new(1.0, -1.0, -0.75)),
        )
        .expect("the absorbing face's corner run is a single cross edge");
    assert!(mesh.edge_seam(cross), "both legs flagged ⇒ the AND passes");
    assert_eq!(
        seamed_edges(&mesh).len(),
        3,
        "the two re-pointed legs plus the one absorbed cross edge",
    );
}

/// The negative arm of the same AND (BEVEL.md §6.7.1.1(b)): **one** flagged leg
/// leaves the terminal's absorbed cross edge clean — the corner run is stamped from
/// one flanking owner and suppressed unless the other agrees. This is what
/// distinguishes the contiguity AND from a raw leading-owner copy.
#[test]
fn terminal_absorbed_corner_leaves_cross_edge_clean_when_one_leg_flagged() {
    let mut mesh = LoopMesh::new();
    mesh_add_cube(&mut mesh);
    let beveled = cube_edge(
        &mesh,
        Vec3::new(1.0, -1.0, -1.0),
        Vec3::new(-1.0, -1.0, -1.0),
    );
    let bottom_leg = cube_edge(&mesh, Vec3::new(1.0, -1.0, -1.0), Vec3::new(1.0, -1.0, 1.0));
    mesh.set_edge_seam(bottom_leg, true);

    mesh.bevel(
        &[beveled.into()],
        BevelOptions {
            offset: 0.25,
            affect: BevelAffect::Edges,
            ..Default::default()
        },
    );
    mesh.validate_topology();

    let cross = mesh
        .query_edge_exists(
            flag_vert_at(&mesh, Vec3::new(1.0, -0.75, -1.0)),
            flag_vert_at(&mesh, Vec3::new(1.0, -1.0, -0.75)),
        )
        .expect("the absorbing face's corner run is a single cross edge");
    assert!(!mesh.edge_seam(cross), "one flagged leg ⇒ the AND fails");
    assert_eq!(
        seamed_edges(&mesh),
        vec![bottom_leg],
        "the flagged leg alone stays flagged; nothing new is",
    );
}

/// The same leg preservation on the **junction** path (BEVEL.md §7.6.1: an absorbed
/// leg is re-pointed onto its sector's boundvert). Bevel two edges of cube corner
/// `v0` and flag its one remaining leg: the leg survives the junction re-stitch with
/// its flags, and flags nothing else — the beveled edges carry no flag here, so
/// mechanism (1)'s longitudinal half has nothing to copy and mechanism (2) has no
/// gap to bridge.
#[test]
fn unbeveled_leg_keeps_its_flags_across_a_junction() {
    for segments in [1u32, 2] {
        let mut mesh = LoopMesh::new();
        mesh_add_cube(&mut mesh);
        let first = cube_edge(
            &mesh,
            Vec3::new(1.0, -1.0, -1.0),
            Vec3::new(-1.0, -1.0, -1.0),
        );
        let second = cube_edge(&mesh, Vec3::new(1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, -1.0));
        let leg = cube_edge(&mesh, Vec3::new(1.0, -1.0, -1.0), Vec3::new(1.0, -1.0, 1.0));
        mesh.set_edge_seam(leg, true);
        mesh.set_edge_sharp(leg, true);

        mesh.bevel(
            &[first.into(), second.into()],
            BevelOptions {
                offset: 0.25,
                segments,
                affect: BevelAffect::Edges,
                mark_seam: true,
                mark_sharp: true,
                ..Default::default()
            },
        );
        mesh.validate_topology();

        assert_eq!(
            seamed_edges(&mesh),
            vec![leg],
            "segments={segments}: the absorbed leg keeps its seam and spreads none",
        );
        assert_eq!(
            sharp_edges(&mesh),
            vec![leg],
            "segments={segments}: same for sharp",
        );
        let far = flag_vert_at(&mesh, Vec3::new(1.0, -1.0, 1.0));
        let [p, q] = mesh.edge_vert_keys(leg);
        assert!(
            p == far || q == far,
            "segments={segments}: the leg still runs to its far endpoint",
        );
    }
}

// ---- The valence-≥4 terminal: its *middle* leg (the one flanking neither radial
// face of the beveled edge) is the leg BEVEL.md §7.5 gives an on-edge vert to. The
// bipyramid's rim verts are valence-4, so beveling one rim edge puts a valence-4
// terminal at each of its ends: at `R1` the flanking legs are `A–R1` / `B–R1` and
// the middle leg is the next rim edge `R1–R2`.

fn bipyramid_rim_edge(mesh: &LoopMesh, rim: &[VertexKey; 8], k: usize) -> EdgeKey {
    mesh.query_edge_exists(rim[k], rim[(k + 1) % 8])
        .expect("the bipyramid's rim edge exists")
}

/// BEVEL.md §8.3.0.2: at `segments ≥ 2` the valence-≥4 terminal builds its poly-fan,
/// minting an on-edge vert per non-flanking leg — so the middle leg has a landing to
/// be re-pointed onto and survives with its `EdgeKey`, and so its whole per-edge
/// attribute block (§6.5 item 1).
#[test]
fn terminal_valence_four_middle_leg_keeps_its_flags_and_identity() {
    for segments in [2u32, 3] {
        let (mut mesh, _apex, rim, _apex_leg) = octagonal_bipyramid_all_legs();
        let beveled = bipyramid_rim_edge(&mesh, &rim, 0);
        let middle = bipyramid_rim_edge(&mesh, &rim, 1);
        mesh.set_edge_seam(middle, true);
        mesh.set_edge_sharp(middle, true);
        let offset = 0.2;
        let leg_direction =
            (mesh.vertex_position(rim[2]) - mesh.vertex_position(rim[1])).normalize();
        let landing = mesh.vertex_position(rim[1]) + leg_direction * offset;

        mesh.bevel(
            &[beveled.into()],
            BevelOptions {
                offset,
                segments,
                affect: BevelAffect::Edges,
                ..Default::default()
            },
        );
        mesh.validate_topology();

        assert_eq!(
            seamed_edges(&mesh),
            vec![middle],
            "segments={segments}: the middle leg keeps its seam, and nothing else gains one",
        );
        assert_eq!(
            sharp_edges(&mesh),
            vec![middle],
            "segments={segments}: same for sharp",
        );
        let [p, q] = mesh.edge_vert_keys(middle);
        assert!(
            p == rim[2] || q == rim[2],
            "segments={segments}: the leg still runs to its far rim vert",
        );
        let near = if p == rim[2] { q } else { p };
        assert!(
            (mesh.vertex_position(near) - landing).length() < 1.0e-4,
            "segments={segments}: the leg is re-pointed onto its poly-fan on-edge vert",
        );
    }
}

/// The `segments = 1` counterpart (BEVEL.md §7.5): the single-segment close mints the
/// same on-edge vert on the middle leg, so the leg survives with its identity — same
/// key, same seam / sharp — re-pointed onto that vert exactly as at `segments ≥ 2`.
/// The endpoint fill there is a single triangle (`E`'s through-chain is just its two
/// flanking rungs), so the whole bevel is one strip quad plus two fills.
#[test]
fn terminal_valence_four_at_segments_one_repoints_the_middle_leg() {
    let (mut mesh, _apex, rim, _apex_leg) = octagonal_bipyramid_all_legs();
    let beveled = bipyramid_rim_edge(&mesh, &rim, 0);
    let middle = bipyramid_rim_edge(&mesh, &rim, 1);
    mesh.set_edge_seam(middle, true);
    mesh.set_edge_sharp(middle, true);
    let offset = 0.2;
    let leg_direction = (mesh.vertex_position(rim[2]) - mesh.vertex_position(rim[1])).normalize();
    let landing = mesh.vertex_position(rim[1]) + leg_direction * offset;

    let result = mesh.bevel(
        &[beveled.into()],
        BevelOptions {
            offset,
            affect: BevelAffect::Edges,
            ..Default::default()
        },
    );
    mesh.validate_topology();

    assert_eq!(
        result.faces.len(),
        3,
        "one strip quad plus one endpoint fill per valence-4 terminal",
    );
    assert_eq!(
        seamed_edges(&mesh),
        vec![middle],
        "the middle leg keeps its seam, and nothing else gains one",
    );
    assert_eq!(sharp_edges(&mesh), vec![middle], "same for sharp");
    let [p, q] = mesh.edge_vert_keys(middle);
    assert!(
        p == rim[2] || q == rim[2],
        "the leg still runs to its far rim vert",
    );
    let near = if p == rim[2] { q } else { p };
    assert!(
        (mesh.vertex_position(near) - landing).length() < 1.0e-4,
        "the leg is re-pointed onto its on-edge vert, slid the plain offset",
    );
}

// ---- Per-face customdata across the face rebuild ----
