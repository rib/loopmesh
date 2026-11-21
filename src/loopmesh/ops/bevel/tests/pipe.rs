use super::*;

/// BEVEL.md §8.3.4.1: a 4-way anti-parallel cross ("plus", `s = 4`) at `segments = 2`
/// takes the pipe override — the 4 cut pentagons + bottom quad are discarded and the ring
/// is re-tiled as 4 quads around a single welded hub. The connected fill is the
/// from-scratch ADJ grid (29 V / 52 E / 24 F, 12 tris + 12 quads, no pentagons); on top of
/// it the override mints `(segments − 1)·s = 1·4 = 4` loose valence-0 leftover verts, so
/// the whole mesh reads **33 V / 52 E / 24 F**.
#[test]
fn cutoff_pipe_override_cross_segments2_is_all_quads() {
    let (mesh, _result) = bevel_flat_fan_cutoff(&[1, 4, 7, 10], 0.2, 2);

    assert_eq!(
        mesh.vert_keys().len(),
        33,
        "vertex count = 29 ADJ + 4 loose"
    );
    assert_eq!(mesh.edge_keys().len(), 52, "edge count");
    assert_eq!(mesh.face_keys().len(), 24, "face count");
    assert_eq!(
        loose_vertex_count(&mesh),
        4,
        "1·4 loose valence-0 leftover verts",
    );

    let histogram = face_side_histogram(&mesh);
    assert_eq!(histogram.get(&3).copied().unwrap_or(0), 12, "12 tris");
    assert_eq!(histogram.get(&4).copied().unwrap_or(0), 12, "12 quads");
    assert_eq!(
        histogram.get(&5).copied().unwrap_or(0),
        0,
        "the pipe override leaves no cut pentagons",
    );
}

/// BEVEL.md §8.3.4.1: the same 4-way cross at `segments = 3` — the extra grid row makes
/// it a full 3×3 quad block over the ring. The connected fill is the from-scratch ADJ grid
/// (40 V / 72 E / 33 F, 12 tris + 21 quads, no hexagons); the override mints
/// `(segments − 1)·s = 2·4 = 8` loose leftover verts, so the whole mesh reads
/// **48 V / 72 E / 33 F**.
#[test]
fn cutoff_pipe_override_cross_segments3_is_full_grid() {
    let (mesh, _result) = bevel_flat_fan_cutoff(&[1, 4, 7, 10], 0.2, 3);

    assert_eq!(
        mesh.vert_keys().len(),
        48,
        "vertex count = 40 ADJ + 8 loose"
    );
    assert_eq!(mesh.edge_keys().len(), 72, "edge count");
    assert_eq!(mesh.face_keys().len(), 33, "face count");
    assert_eq!(
        loose_vertex_count(&mesh),
        8,
        "2·4 loose valence-0 leftover verts",
    );

    let histogram = face_side_histogram(&mesh);
    assert_eq!(histogram.get(&3).copied().unwrap_or(0), 12, "12 tris");
    assert_eq!(histogram.get(&4).copied().unwrap_or(0), 21, "21 quads");
    assert_eq!(
        histogram.get(&6).copied().unwrap_or(0),
        0,
        "no cut hexagons survive the override",
    );

    // §8.3.4.1: at `segments = 3` the fan cell's arc is asymmetric, so the resampled
    // rim diverges from the raw cutoff samples. The connected fill must ride the
    // resampled arc (from-scratch ADJ), leaving the raw samples only on the loose
    // leftovers — no connected rim node may coincide with a loose sample.
    assert_eq!(
        connected_verts_on_loose_sample(&mesh),
        0,
        "the resampled connected rim moved off the raw arc samples the loose verts carry",
    );
}

/// BEVEL.md §8.3.4.1 condition 1: `s ≥ 5` keeps the ordinary cut faces even when an
/// anti-parallel opposite pair is present. Beveling spokes `0, 1, 2, 3, 6` gives an
/// `s = 5` ring with the collinear pair `0`/`6`, yet the fill must stay the n-gon cut
/// faces (a pentagon at `segments = 2`), not the all-quad grid.
#[test]
fn cutoff_pipe_override_declines_at_valence_five() {
    let (mesh, _result) = bevel_flat_fan_cutoff(&[0, 1, 2, 3, 6], 0.2, 2);

    let histogram = face_side_histogram(&mesh);
    assert!(
        histogram.get(&5).copied().unwrap_or(0) > 0,
        "an `s = 5` anti-parallel ring keeps its cut pentagons: {histogram:?}",
    );
    assert_eq!(
        loose_vertex_count(&mesh),
        0,
        "an ordinary (non-override) cutoff ring mints no loose leftover verts",
    );
}

/// BEVEL.md §8.3.4.1 condition 2: a `s = 4` ring with **no** anti-parallel opposite pair
/// (four *adjacent* spokes) does not take the override — the ordinary cutoff n-gon fill
/// (pentagons at `segments = 2`) stands, output-identical to before the rule existed.
#[test]
fn cutoff_pipe_override_declines_without_antiparallel_pair() {
    let (mesh, _result) = bevel_flat_fan_cutoff(&[0, 1, 2, 3], 0.2, 2);

    let histogram = face_side_histogram(&mesh);
    assert!(
        histogram.get(&5).copied().unwrap_or(0) > 0,
        "a non-anti-parallel `s = 4` ring keeps its cut pentagons: {histogram:?}",
    );
    assert_eq!(
        loose_vertex_count(&mesh),
        0,
        "an ordinary (non-override) cutoff ring mints no loose leftover verts",
    );
}

/// BEVEL.md §8.3.4.1 condition 2 is a **ring-distance-2** test — "among three *consecutive*
/// ring members, the first and third have collinear edges" — so at `s = 4` a pair that is
/// anti-parallel but ring-**adjacent** does not name a pipe, and the ordinary cut faces
/// stand.
///
/// Spokes `0 / 1 / 2 / 6` of the 12-fan sit at `−30° / 0° / 30° / 150°`: the only
/// anti-parallel pair is `−30°`/`150°`, which is adjacent in the ring (the other three
/// spokes are all on one side of it). The two distance-2 pairs — `−30°/30°` and `0°/150°` —
/// are not collinear, so the junction declines.
///
/// This is a **count-bearing** distinction, not a placement one: the override would route
/// this junction to the adjacency grid *plus* `(segments − 1)·s` loose leftovers, so
/// `V`, `E` and `F` all move with the verdict. The sibling test above declines for a
/// different reason (no anti-parallel pair at all), which is why it cannot see this.
#[test]
fn cutoff_pipe_override_declines_on_a_ring_adjacent_pair() {
    let (mesh, _result) = bevel_flat_fan_cutoff(&[0, 1, 2, 6], 0.2, 2);

    let histogram = face_side_histogram(&mesh);
    assert!(
        histogram.get(&5).copied().unwrap_or(0) > 0,
        "a ring-adjacent anti-parallel `s = 4` ring keeps its cut pentagons: {histogram:?}",
    );
    assert_eq!(
        loose_vertex_count(&mesh),
        0,
        "an ordinary (non-override) cutoff ring mints no loose leftover verts",
    );

    // Non-vacuity: the *same* fan, the same spoke count, the same offset and segments —
    // with the anti-parallel pair moved to ring distance 2 — does take the override, so
    // the decline above is the ring distance and nothing else.
    let (retiled, _result) = bevel_flat_fan_cutoff(&[0, 1, 6, 7], 0.2, 2);
    assert_eq!(
        face_side_histogram(&retiled).get(&5).copied().unwrap_or(0),
        0,
        "the distance-2 pair re-tiles the ring as the all-quad grid",
    );
    assert_eq!(
        loose_vertex_count(&retiled),
        4,
        "`(segments − 1)·s = 1·4` loose leftover verts on the re-tiled ring",
    );
}

/// The flat 12-spoke fan with rim vert `4` lifted out of plane by `lift`, which tilts the
/// two triangles that vert bounds and nothing else. Beveled as the 3-way pipe `[0, 3, 6]`
/// — spokes at `0° / 90° / 180°`, the sole collinear pair running along `±x` — so no
/// beveled spoke moves and §8.3.4.1's conditions 1 and 2 hold exactly at every `lift`,
/// leaving condition 3's dot magnitude the only quantity in motion.
///
/// That magnitude moves at **`2·lift`**: the steepest tilted face is the triangle
/// spanning rim verts `3` (at `90°`, so its `y` component is `1`) and `4`, whose raw
/// Newell normal picks up an axis component of `lift` over a `sin 30° = ½` area scale.
fn bevel_tilted_fan_pipe_s3(lift: f32, offset: f32, segments: u32) -> (LoopMesh, f32) {
    let (mut mesh, centre, rim) = planar_fan(12);
    mesh.set_vertex_position(rim[4], mesh.vertex_position(rim[4]) + Vec3::Z * lift);

    let spokes: Vec<EdgeKey> = [0usize, 3, 6]
        .into_iter()
        .map(|k| {
            mesh.query_edge_exists(centre, rim[k])
                .expect("spoke exists")
        })
        .collect();
    let axis = mesh
        .edge_direction_from(spokes[0], centre, mesh.vertex_position(centre))
        .expect("the pipe axis");
    let out_of_plane = mesh
        .vert_face_keys(centre)
        .into_iter()
        .map(|face| mesh.face_plane_normal(face).dot(axis).abs())
        .fold(0.0_f32, f32::max);

    let geom: Vec<GeomKey> = spokes.into_iter().map(GeomKey::from).collect();
    mesh.bevel(
        &geom,
        BevelOptions {
            offset,
            affect: BevelAffect::Edges,
            segments,
            vmesh_method: BevelVmeshMethod::Cutoff,
            ..Default::default()
        },
    );
    mesh.validate_topology();
    (mesh, out_of_plane)
}

/// BEVEL.md §8.3.4.1 condition 3: the fan-parallelism window is a **dot-product
/// magnitude** against `1e-4` (≈ `0.0057°` off perpendicular), *not* an angle, and
/// emphatically not condition 2's `EPS_ANG = 2°`.
///
/// The tilted fan isolates it: conditions 1 and 2 are satisfied exactly at every lift
/// (no beveled spoke moves), so the verdict turns on the fan's largest
/// `|dot(face normal, pipe axis)|` alone, which the fixture drives linearly at `2·lift`.
/// The override's signature is its `(segments − 1)·s` loose leftover verts, which no
/// ordinary cutoff ring mints.
///
/// The last row is the one that measures the constant rather than merely the shape of
/// the comparison: at `|dot| = 0.02` the fan is a visible `1.15°` off prismatic, well
/// outside `1e-4` — yet still *inside* a `sin 2°` window, so reusing condition 2's
/// tolerance here fires the override on it.
#[test]
fn cutoff_pipe_override_condition3_window_is_a_dot_magnitude_not_an_angle() {
    let condition_2_window_as_a_dot = (2.0_f32.to_radians()).sin();
    // (lift, the `|dot|` it drives, whether §8.3.4.1 admits).
    let window: [(f32, bool); 4] = [
        (0.0, true),
        (4.0e-5, true),
        (6.0e-5, false),
        (1.0e-2, false),
    ];
    for (lift, admits) in window {
        let (mesh, out_of_plane) = bevel_tilted_fan_pipe_s3(lift, 0.25, 2);
        // `2·lift` is the first-order figure; unitising the tilted normal costs a
        // `1/√(1 + 4·lift²)` the fixture is not trying to pin.
        assert!(
            (out_of_plane - 2.0 * lift).abs() < 1.0e-3 * (2.0 * lift) + 1.0e-9,
            "fixture arithmetic: lift {lift} drives |dot| = {out_of_plane}, wanted {}",
            2.0 * lift,
        );
        assert_eq!(
            out_of_plane <= PIPE_PERPENDICULAR_DOT_EPS,
            admits,
            "fixture sanity: |dot| = {out_of_plane} sits on the wrong side of the window",
        );
        assert!(
            out_of_plane < condition_2_window_as_a_dot,
            "every row stays inside condition 2's window, so only condition 3 decides",
        );
        assert_eq!(
            loose_vertex_count(&mesh) == 3,
            admits,
            "lift {lift} (|dot| = {out_of_plane}): the override's `(segments − 1)·s = 3` \
             loose leftover verts must appear iff the fan is within `1e-4` of prismatic",
        );
    }
}

/// BEVEL.md §8.3.4.1 condition 2 / §8.5.5.5(f.6): *which* pair a two-pair ring names.
///
/// A flat 4-way cross carries two anti-parallel pairs, and the guide's worked (f.6) case
/// pins the answer for the fan whose ring walk starts at the `0°` spoke: "the ring walk
/// names the `90°/270°` pair as the axis". Under §8.2.2's association — boundvert `B_k`'s
/// own profile chain is `B_k → B_(k+1)`, crossing the edge that *closes* sector `k` —
/// boundvert `0` tests spokes `1` and `3`, which is that pair. Picking the other one yields
/// the same cap rotated a quarter turn, and the only thing that catches it is a positional
/// figure, so the choice is pinned here directly rather than through a cap fingerprint.
#[test]
fn pipe_boundvert_spokes_names_the_ring_walks_pair() {
    // The ring walk's directions on the worked fan: `0° / 90° / 180° / 270°`.
    let cross = [Vec3::X, Vec3::Y, -Vec3::X, -Vec3::Y];
    assert_eq!(
        pipe_boundvert_spokes(&cross),
        Some((1, 3)),
        "the two-pair ring names the `90°/270°` pair, so the pipe axis is `y`",
    );
    // Valence 3: one pair plus a transverse passenger, and every rotation of the ring names
    // the same axis line — at `s = 3` the distance-2 scan and an any-pair scan coincide.
    for rotation in 0..3 {
        let mut ring = [Vec3::X, -Vec3::X, Vec3::Y];
        ring.rotate_left(rotation);
        let (own, opposite) = pipe_boundvert_spokes(&ring).expect("the collinear pair");
        assert!(
            ring[own].dot(ring[opposite]) < -0.99,
            "rotation {rotation} named a non-collinear pair",
        );
    }
    // Ring-adjacent anti-parallel spokes at `s = 4` are not a pipe (see the override test):
    // `0°/180°` are adjacent in this ring, and neither distance-2 pair is collinear.
    let adjacent = [0.0_f32, 30.0, 60.0, 180.0].map(|degrees| {
        let radians = degrees.to_radians();
        Vec3::new(radians.cos(), radians.sin(), 0.0)
    });
    assert_eq!(pipe_boundvert_spokes(&adjacent), None);
}

/// A flat 3×3 quad grid (9 verts, 4 quads). The centre vertex (index 4) has four faced
/// spokes; beveling its left (→3), right (→5) and up (→7) spokes plants a `s = 3` pipe:
/// left/right collinear through the centre, up transverse, every incident face in the
/// `z = 0` plane. Returns `(mesh, result)`.
fn bevel_flat_grid_pipe_s3(offset: f32, segments: u32) -> (LoopMesh, BevelResult) {
    let mut mesh = LoopMesh::new();
    let verts: Vec<VertexKey> = [
        [0.0, 0.0],
        [1.0, 0.0],
        [2.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [2.0, 1.0],
        [0.0, 2.0],
        [1.0, 2.0],
        [2.0, 2.0],
    ]
    .into_iter()
    .map(|[x, y]| mesh.create_vertex(Vec3::new(x, y, 0.0)))
    .collect();
    for cycle in [[0, 1, 4, 3], [1, 2, 5, 4], [3, 4, 7, 6], [4, 5, 8, 7]] {
        let corners: Vec<VertexKey> = cycle.iter().map(|&i| verts[i]).collect();
        mesh.face_create_verts(&corners, FaceCreateFlags::NONE);
    }
    let geom: Vec<GeomKey> = [3usize, 5, 7]
        .into_iter()
        .map(|far| {
            mesh.query_edge_exists(verts[4], verts[far])
                .expect("spoke exists")
                .into()
        })
        .collect();
    let result = mesh.bevel(
        &geom,
        BevelOptions {
            offset,
            affect: BevelAffect::Edges,
            segments,
            vmesh_method: BevelVmeshMethod::Cutoff,
            ..Default::default()
        },
    );
    mesh.validate_topology();
    (mesh, result)
}

/// BEVEL.md §8.3.4.1: the 3-way pipe (`s = 3`, one collinear pair + one transverse spoke)
/// at `segments = 2` takes the override — an `s`-quad wheel around one hub, all quads, no
/// cut pentagons. On the **bare** grid the connected fill is the from-scratch ADJ grid
/// (21 V / 33 E / 13 F); the override mints `(segments − 1)·s = 1·3 = 3` loose leftover
/// verts, so the whole mesh reads **24 V / 33 E / 13 F**. (The §8.3.0.4 wire-decorated
/// witness adds a further +1 V/+1 E for the re-attached wire spoke — orthogonal to the
/// pipe surplus and not exercised here.)
#[test]
fn cutoff_pipe_override_three_way_segments2_is_all_quads() {
    let (mesh, _result) = bevel_flat_grid_pipe_s3(0.25, 2);

    assert_eq!(
        mesh.vert_keys().len(),
        24,
        "vertex count = 21 ADJ + 3 loose"
    );
    assert_eq!(mesh.edge_keys().len(), 33, "edge count");
    assert_eq!(mesh.face_keys().len(), 13, "face count");
    assert_eq!(
        loose_vertex_count(&mesh),
        3,
        "1·3 loose valence-0 leftover verts",
    );

    let histogram = face_side_histogram(&mesh);
    assert!(
        histogram.keys().all(|&sides| sides == 4),
        "the whole grid re-tile is all quads: {histogram:?}",
    );

    // §8.3.4.1: the two collinear (left/right) chains are skewed by the transverse
    // spoke, so their resampled rim nodes must sit **off** their raw cutoff samples —
    // only the transverse (up) chain is symmetric about the grid's mirror axis, where
    // raw and resampled coincide. So exactly its `segments − 1` rim nodes land on a
    // loose sample; the `2·(segments − 1)` skewed rim nodes have all moved off. A
    // regression that fed the raw cutoff chain into the connected fill would instead
    // leave every rim node (`3·(segments − 1)` of them) on a loose sample.
    assert_eq!(
        connected_verts_on_loose_sample(&mesh),
        1,
        "only the symmetric transverse chain's `segments − 1 = 1` rim node stays on the \
         raw samples; the two skewed chains' rim nodes moved to the resampled arc",
    );
}

/// BEVEL.md §8.3.4.1: the same 3-way pipe at `segments = 3` — the odd-parity central cap
/// appears as one central triangle plus 6 quads over the ring (1 tri + 19 quads whole
/// mesh, 20 F). The connected fill is the from-scratch ADJ grid (29 V / 48 E / 20 F);
/// the override mints `(segments − 1)·s = 2·3 = 6` loose leftover verts, so the whole
/// mesh reads **35 V / 48 E / 20 F**, and — the cell being skewed — the resampled rim
/// sits off the raw arc samples the loose verts carry.
#[test]
fn cutoff_pipe_override_three_way_segments3_is_grid_with_central_cap() {
    let (mesh, _result) = bevel_flat_grid_pipe_s3(0.25, 3);

    assert_eq!(
        mesh.vert_keys().len(),
        35,
        "vertex count = 29 ADJ + 6 loose"
    );
    assert_eq!(mesh.edge_keys().len(), 48, "edge count");
    assert_eq!(mesh.face_keys().len(), 20, "face count");
    assert_eq!(
        loose_vertex_count(&mesh),
        6,
        "2·3 loose valence-0 leftover verts",
    );

    let histogram = face_side_histogram(&mesh);
    assert_eq!(
        histogram.get(&3).copied().unwrap_or(0),
        1,
        "one §8.5 odd-parity central-cap triangle",
    );
    assert_eq!(histogram.get(&4).copied().unwrap_or(0), 19, "19 quads");
    // §8.3.4.1: only the symmetric transverse chain's `segments − 1 = 2` rim nodes stay
    // on their raw samples; the two skewed collinear chains' `2·(segments − 1) = 4` rim
    // nodes moved to the resampled arc. A raw-fed regression would leave all 6 on a
    // loose sample.
    assert_eq!(
        connected_verts_on_loose_sample(&mesh),
        2,
        "the two skewed chains' rim nodes moved off the raw arc samples; only the \
         symmetric transverse chain's two rim nodes stay coincident",
    );
}

/// A genuinely-3-D pipe junction (BEVEL.md §8.5.5.5(f.6)): a `3 × 3` grid folded into a
/// **tent** — the middle row lifted to `z = 1`, the outer rows at `z = 0` — so the four
/// quads tilt into a prism whose cross-section is a real (non-collinear) roof, constant
/// along the ridge. The ridge runs `x` through the centre vertex 4 = (1, 1, 1); every
/// incident face's plane contains the `x` direction, so all normals are ⊥ `x` (the pipe
/// face-normal condition). Beveling the centre's `±x` spokes (to 3 / 5, collinear) plus
/// one (`s = 3`) or both (`s = 4`) transverse spokes (to 7 / 1) plants the pipe.
fn tented_grid_pipe(
    beveled_spokes: &[usize],
    offset: f32,
    segments: u32,
    method: BevelVmeshMethod,
) -> (LoopMesh, BevelResult) {
    tented_grid_pipe_profile(beveled_spokes, offset, segments, method, 0.5)
}

/// The same tented pipe at a chosen `profile`. The circular default (`0.5`) is what the
/// filled-pipe tests use; a curved exponent (`0.7`) is how the **general** ring is driven
/// out of the fill's scope while leaving the pipe predicate itself untouched.
fn tented_grid_pipe_profile(
    beveled_spokes: &[usize],
    offset: f32,
    segments: u32,
    method: BevelVmeshMethod,
    profile: f32,
) -> (LoopMesh, BevelResult) {
    let mut mesh = LoopMesh::new();
    let verts: Vec<VertexKey> = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [0.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
        [2.0, 1.0, 1.0],
        [0.0, 2.0, 0.0],
        [1.0, 2.0, 0.0],
        [2.0, 2.0, 0.0],
    ]
    .into_iter()
    .map(|[x, y, z]| mesh.create_vertex(Vec3::new(x, y, z)))
    .collect();
    for cycle in [[0, 1, 4, 3], [1, 2, 5, 4], [3, 4, 7, 6], [4, 5, 8, 7]] {
        let corners: Vec<VertexKey> = cycle.iter().map(|&i| verts[i]).collect();
        mesh.face_create_verts(&corners, FaceCreateFlags::NONE);
    }
    let geom: Vec<GeomKey> = beveled_spokes
        .iter()
        .map(|&far| {
            mesh.query_edge_exists(verts[4], verts[far])
                .expect("spoke exists")
                .into()
        })
        .collect();
    let result = mesh.bevel(
        &geom,
        BevelOptions {
            offset,
            affect: BevelAffect::Edges,
            segments,
            vmesh_method: method,
            profile,
            ..Default::default()
        },
    );
    mesh.validate_topology();
    (mesh, result)
}

/// The perpendicular distance from the tented-grid pipe's constant cross-section
/// **cylinder** axis (§8.5.5.5(f.6)). The ridge runs `x` through the junction (1,1,1);
/// the symmetric skirts make the circular (`offset`-radius) cross-section a quarter
/// circle whose centre sits on the mirror plane `y = 1` at `z = 1 − offset·√2`, so a
/// vert lands **on the cross-section** iff this equals `offset`. The rounded base-grid
/// cap does not — that is what the replant corrects.
fn tented_pipe_cyl_distance(position: Vec3, offset: f32) -> f32 {
    let axis_z = 1.0 - offset * std::f32::consts::SQRT_2;
    ((position.y - 1.0).powi(2) + (position.z - axis_z).powi(2)).sqrt()
}

/// The bevel-created vert closest to the pipe junction (1,1,1): the innermost cap vert
/// — the welded / central apex (`segments = 2`) or a central `s`-gon vert
/// (`segments = 3`) — which the §8.5.5.5(f.6) replant snaps onto the cross-section.
fn innermost_cap_vert(mesh: &LoopMesh, result: &BevelResult) -> Vec3 {
    result
        .verts
        .iter()
        .map(|&v| mesh.vertex_position(v))
        .min_by(|a, b| {
            let da = (*a - Vec3::new(1.0, 1.0, 1.0)).length();
            let db = (*b - Vec3::new(1.0, 1.0, 1.0)).length();
            da.partial_cmp(&db).unwrap()
        })
        .expect("the pipe fill created cap verts")
}

/// §8.5.5.5(f.6): the replant snaps an interior grid vert onto its station's profile
/// curve — a point off the curve is projected on, the station (position along the pipe
/// axis) is preserved, and a point already on the curve is a fixed point (idempotent).
#[test]
fn pipe_replant_snaps_interior_onto_cross_section() {
    // A unit quarter-circle cross-section in the `y–z` plane; pipe axis `x`.
    let cell = ProfileCell::from_control_points(
        Vec3::new(0.0, 1.0, 0.0),
        Vec3::new(0.0, 0.0, 1.0),
        Vec3::new(0.0, 1.0, 1.0),
    );
    // A point off the arc at station `x = 5`, on the `(0,1,1)` diagonal.
    let replanted = replant(Vec3::new(5.0, 2.0, 2.0), Vec3::X, &cell, 2.0);
    let expected = Vec3::new(
        5.0,
        std::f32::consts::FRAC_1_SQRT_2,
        std::f32::consts::FRAC_1_SQRT_2,
    );
    assert!(
        (replanted - expected).length() < 1.0e-5,
        "off-curve vert snaps onto the arc at its own station: {replanted:?}",
    );
    // On the unit circle in the cross-section plane, at the station it was carried to.
    assert!(
        (replanted.x - 5.0).abs() < 1.0e-5,
        "station (pipe-axis coord) preserved"
    );
    let radius = (replanted.y.powi(2) + replanted.z.powi(2)).sqrt();
    assert!(
        (radius - 1.0).abs() < 1.0e-5,
        "lands on the unit-radius cross-section circle",
    );
    // Idempotent: re-planting a vert already on the curve does not move it.
    let again = replant(replanted, Vec3::X, &cell, 2.0);
    assert!(
        (again - replanted).length() < 1.0e-6,
        "replant is idempotent on the curve"
    );
}

/// §8.5.5.5(f.6): the two degenerate cross-sections collapse to a nearest-point rule
/// rather than asserting — a point cross-section (`start' == end'`) captures every vert,
/// and a chord cross-section (collinear projected control points, e.g. a flat-chamfer
/// pipe) captures the nearest point of the chord.
#[test]
fn pipe_replant_degenerate_point_and_chord_fallbacks() {
    // `start` / `end` differ only along the pipe axis, so both project to one station
    // point — the cross-section is a single point.
    let point_cell = ProfileCell::from_control_points(
        Vec3::new(0.0, 1.0, 0.0),
        Vec3::new(1.0, 1.0, 0.0),
        Vec3::new(0.0, 2.0, 0.0),
    );
    let collapsed = replant(Vec3::new(4.0, 9.0, 9.0), Vec3::X, &point_cell, 2.0);
    assert!(
        (collapsed - Vec3::new(4.0, 1.0, 0.0)).length() < 1.0e-5,
        "a point cross-section captures the vert at its station: {collapsed:?}",
    );
    // Collinear control points: the cross-section is the chord `start'–end'`.
    let chord_cell = ProfileCell::from_control_points(
        Vec3::new(0.0, 1.0, 0.0),
        Vec3::new(0.0, 3.0, 0.0),
        Vec3::new(0.0, 2.0, 0.0),
    );
    let onto_chord = replant(Vec3::new(0.0, 5.0, 4.0), Vec3::X, &chord_cell, 2.0);
    assert!(
        (onto_chord - Vec3::new(0.0, 3.0, 0.0)).length() < 1.0e-5,
        "a chord cross-section clamps the vert onto the segment end: {onto_chord:?}",
    );
}

/// The §8.5.5.5(f.6) planar junction's cross-section: a chord across the pipe edge in the
/// junction plane, with the junction vertex as the profile's `mid` (§8.2.2.1's tier-1
/// rescue). The pipe axis is `y`, so the clamped transverse coordinate is `x` and the
/// chord spans `[−offset, +offset]` — the frame (f.6)'s worked case states its figures in.
fn planar_pipe_cell(offset: f32) -> ProfileCell {
    ProfileCell::from_control_points(
        Vec3::new(offset, offset, 0.0),
        Vec3::new(-offset, offset, 0.0),
        Vec3::ZERO,
    )
}

/// §8.5.5.5(f.6) step 4's second fallback at a **planar** junction, which is where it is
/// the ordinary case rather than an error path: the profile plane contains the pipe axis,
/// so the projected control triple is collinear at every station and every vert replants
/// onto the closed chord `start'–end'`.
///
/// In the station frame that separates into three independent statements — the along-axis
/// coordinate is untouched (the station plane moves *with* the vert), the out-of-plane
/// component is dropped, and the transverse one is clamped into the chord's span. The
/// profile here is emphatically **not** flagged linear (its own unprojected triple spans a
/// cell: `mid` is the junction vertex, an offset clear of the chord), so an implementation
/// that reaches the chord only through that flag declines every clamp it owes.
#[test]
fn pipe_replant_planar_junction_clamps_transverse_and_keeps_the_station() {
    let offset = 0.2;
    let cell = planar_pipe_cell(offset);
    assert!(
        !cell.collinear,
        "the planar junction's profile is a genuine curve, not a flagged-linear chord",
    );

    for exponent in [
        0.0_f32,
        profile_to_exponent(0.1),
        2.0,
        PROFILE_SQUARE_EXPONENT,
    ] {
        // Outside the band: transverse clamped to the near end, station preserved.
        let outside = Vec3::new(-2.0 * offset, -0.0667, 0.0);
        let clamped = replant(outside, Vec3::Y, &cell, exponent);
        assert!(
            (clamped - Vec3::new(-offset, -0.0667, 0.0)).length() < 1.0e-6,
            "an out-of-band vert clamps to the chord end at its own station: {clamped:?}",
        );
        // Inside the band: the nearest point of the segment is the vert's own transverse
        // coordinate, so the clamp is the identity there.
        let inside = Vec3::new(0.5 * offset, 0.42, 0.0);
        let unmoved = replant(inside, Vec3::Y, &cell, exponent);
        assert!(
            (unmoved - inside).length() < 1.0e-6,
            "an in-band vert is left where the base grid put it: {unmoved:?}",
        );
        // Off the junction plane: the out-of-plane component is dropped (the chord lies in
        // the plane, so its nearest point does), while the station survives.
        let lifted = Vec3::new(0.5 * offset, 0.42, 0.05);
        let dropped = replant(lifted, Vec3::Y, &cell, exponent);
        assert!(
            (dropped - inside).length() < 1.0e-6,
            "the out-of-plane component is dropped, the station kept: {dropped:?}",
        );
    }
}

/// §8.5.5.5(f.6): the fallback is *the nearest point of the closed segment*, so the two
/// chord bounds are never assumed ordered — which end carries the larger transverse
/// coordinate is decided by the ring walk's direction, and a `clamp(x, lo, hi)` handed
/// `lo > hi` panics in Rust. Reversing the cell's two ring corners must therefore return
/// the same point, not a panic and not the other end.
#[test]
fn pipe_replant_chord_clamp_is_orientation_free() {
    let offset = 0.2;
    let forward = planar_pipe_cell(offset);
    let reversed = ProfileCell::from_control_points(forward.end, forward.start, forward.mid);
    for transverse in [-3.0_f32, -1.0, -0.5, 0.0, 0.25, 1.0, 7.0] {
        let base = Vec3::new(transverse * offset, 0.13, 0.0);
        let along_walk = replant(base, Vec3::Y, &forward, 0.0);
        let against_walk = replant(base, Vec3::Y, &reversed, 0.0);
        assert!(
            (along_walk - against_walk).length() < 1.0e-6,
            "the closed-segment clamp does not depend on endpoint order at \
             transverse {transverse}: {along_walk:?} vs {against_walk:?}",
        );
        assert!(
            along_walk.x.abs() <= offset + 1.0e-6,
            "the clamp lands inside the chord span at transverse {transverse}: \
             {along_walk:?}",
        );
    }
}

/// §8.5.5.5(f.6): the chord fallback has to survive a collinear cross-section whose
/// collinearity is only *exact in the real numbers* — the resolution half of the
/// degeneracy floor (`REPLANT_INPUT_ULP_RATIO`), which the shape half cannot supply.
///
/// The sibling fallback test above uses small integer coordinates, where the three control
/// points are collinear bit-for-bit and a shape-only floor would do. A real flat-chamfer
/// pipe is not so lucky: the control points arrive as `f32` **world** positions, so a `mid`
/// lying exactly on the `start`–`end` chord in the reals still stores a sagitta of a few
/// ulps of its own coordinate. Against the shape half alone that noise reads as a curve —
/// at world `10³` and `offset = 10⁻²` it stands up to ~33× `REPLANT_CELL_SHAPE_RATIO ·
/// extent` — and the replant builds a sliver cell out of storage rounding and takes the
/// curved branch on a profile the guide calls a chord.
#[test]
fn pipe_replant_takes_the_chord_on_a_collinear_cell_stored_in_world_coordinates() {
    let offset = 0.01_f32;
    let mut worst_sagitta_ratio = 0.0_f64;
    for step in 0..8 {
        // Far from the origin relative to `offset`, and off every coordinate direction, so
        // that laying the three points down in `f32` genuinely rounds.
        let hub = Vec3::new(
            1013.0 + 7.3 * step as f32,
            -742.0 - 3.1 * step as f32,
            311.0 + 1.7 * step as f32,
        );
        let along = Vec3::new(
            0.61 + 0.09 * step as f32,
            0.77 - 0.05 * step as f32,
            0.19 + 0.13 * step as f32,
        )
        .normalize();
        let start = hub;
        let end = hub + 2.0 * offset * along;
        // Collinear in the reals — `mid` is a point *of* the chord — and stored in `f32`.
        let mid = start + (0.37 + 0.05 * step as f32) * (end - start);
        let cell = ProfileCell::from_control_points(start, end, mid);

        // How far the stored `mid` misses the stored chord, against the shape-only floor
        // that would have to judge it: this is the term under test.
        let span = (end - start).as_dvec3();
        let sagitta = (mid - start).as_dvec3().cross(span).length() / span.length();
        let shape_floor = REPLANT_CELL_SHAPE_RATIO
            * (mid - start)
                .as_dvec3()
                .length()
                .max((mid - end).as_dvec3().length());
        worst_sagitta_ratio = worst_sagitta_ratio.max(sagitta / shape_floor);

        for exponent in [2.0_f32, profile_to_exponent(0.02)] {
            // A probe well off the chord, at a station away from the control points' own.
            let axis = Vec3::new(0.2, -0.14, 0.97).normalize();
            let sideways = along.cross(axis).normalize();
            let base = mid + 4.0 * offset * axis + 0.8 * offset * sideways;

            let replanted = replant(base, axis, &cell, exponent);
            let projected_start = start - (start - base).dot(axis) * axis;
            let projected_end = end - (end - base).dot(axis) * axis;
            let chord = closest_point_on_segment(base, projected_start, projected_end);
            // The two branches are told apart by *where along* the chord the answer sits,
            // not by its distance from the chord line: the sliver cell's two axes are
            // near-antiparallel, so even the curved branch's answer is within a quantum of
            // the line, but its `(u, v)` solve divides the probe's off-axis component by a
            // near-zero determinant and lands it up to two thirds of a cell away along it.
            // The 0.05-cell threshold is ~8 `f32` world quanta here — four times the
            // read-back rounding the chord answer itself carries, and well inside the
            // departure the curved branch produces.
            let departure = (replanted - chord).length() / offset;
            assert!(
                departure < 0.05,
                "the replant slid {departure} cell units along a collinear cross-section at \
                 world magnitude {} — the stored-sagitta noise was read as a curve",
                hub.abs().max_element(),
            );
        }
    }
    assert!(
        worst_sagitta_ratio > 3.0,
        "no cell in the sweep stores a sagitta clear of the shape-only floor (worst \
         {worst_sagitta_ratio}× it), so the chord assertions above would hold on the shape \
         half alone and this test pins nothing",
    );
}

/// §8.5.5.5(f.6.2): a vert coincident with a projected control point must replant onto
/// that cell **corner**, not near it. Its exact cell coordinates are `(1, 0)`, and branch
/// R's `r`-norm amplifies whatever residue the cell-coordinate solve leaves in the
/// coordinate that should be zero by `ε^r` — a violent *expansion* at a concave exponent,
/// so a single-precision solve's few ulps become a percent-scale inward shrink. The
/// corner-coincident verts are the ones the replant is the identity on, so the whole
/// sweep of stations along the axis-parallel line through `end` must come back unmoved —
/// and, centred, **bit-for-bit** unmoved. Every cell in the sweep is rotated by its own
/// angle, so this is the exactness claim of the axis-aligned companion test at arbitrary
/// cell orientation: the solve reads `u = determinant/determinant` and
/// `v = aa·ab − ab·aa`, both exact for any orientation once `end'` is the origin.
///
/// The identity claim needs a companion, because a corner-coincident `base` lies on the
/// projected chord and is therefore a fixed point of all three degenerate exits too — the
/// claim would read as satisfied with the curved branch never entered. Every exit returns
/// a point *of* that chord, so a second probe on the same cell landing clear of it is the
/// proof that the superellipse snap is what held the corner.
#[test]
fn pipe_replant_holds_corner_coincident_verts_at_a_deep_concave_exponent() {
    let exponent = profile_to_exponent(0.02);
    let offset = 0.2_f32;
    // A cross-section in the `z = 0` plane with the pipe along `z`, so every station
    // projection is an exact translation and the *only* rounding left is the solve's own
    // `start + end - mid` cancellation — the channel (f.6.2) names. The control points sit
    // far from the origin relative to `offset`, which is what makes that cancellation bite.
    let mut worst_chord_clearance = f32::INFINITY;
    for step in 0..16 {
        let corner = Vec3::new(1.3 + 0.37 * step as f32, -0.45 - 0.11 * step as f32, 0.0);
        let sweep = 0.19 * step as f32;
        let start = corner + offset * Vec3::new(sweep.cos(), sweep.sin(), 0.0);
        let end = corner + offset * Vec3::new(sweep.sin(), -sweep.cos(), 0.0);
        let cell = ProfileCell::from_control_points(start, end, corner);
        for station_step in 0..4 {
            // On the axis-parallel line through `end`: the cross-section is constant along
            // the pipe, so `end` projects onto this vert at *every* station, whose cell
            // coordinates are therefore the `(1, 0)` corner exactly.
            let station = -0.75 + 0.5 * station_step as f32;
            let base = Vec3::new(end.x, end.y, station);
            assert_eq!(
                replant(base, Vec3::Z, &cell, exponent),
                base,
                "corner-coincident replant moved a rotated cell's corner (step {step}, \
                 station {station})",
            );

            // Same cell, same station, but the `(1, 1)` corner — off the chord, so only the
            // curved branch can produce its answer. A concave exponent pulls it well onto
            // the `origin` side of `start'–end'` (`2^(-1/r)` along each axis, about half an
            // offset clear at this profile); every degenerate exit would leave it *on* the
            // chord.
            let off_corner = Vec3::new(corner.x, corner.y, station);
            let bulged = replant(off_corner, Vec3::Z, &cell, exponent);
            let projected_start = Vec3::new(start.x, start.y, station);
            let projected_end = Vec3::new(end.x, end.y, station);
            let chord_point = closest_point_on_segment(bulged, projected_start, projected_end);
            worst_chord_clearance =
                worst_chord_clearance.min((bulged - chord_point).length() / offset);
        }
    }
    assert!(
        worst_chord_clearance > 0.25,
        "the curved branch never ran — the off-corner probe stayed within \
         {worst_chord_clearance} cell units of the chord that every degenerate exit returns, \
         so the corner-coincident claim above would hold vacuously",
    );
}

/// A right-angle cross-section cell of extent `offset`, planted at `hub` in the `z = 0`
/// plane with the pipe running along `z`. Its `(1, 1)` cell corner is `hub` itself, so a
/// probe there is the one point every degenerate exit would answer differently: the curve
/// carries it well clear of the `start'–end'` chord, every fallback leaves it on the chord.
fn right_angle_cross_section(hub: Vec3, offset: f32) -> ProfileCell {
    ProfileCell::from_control_points(hub + offset * Vec3::X, hub + offset * Vec3::Y, hub)
}

/// §8.5.5.5(f.6): the replant's degeneracy exits are **scale-free**, so a pipe keeps a
/// genuinely curved cross-section at any bevel offset.
///
/// The cell-coordinate solve's determinant is `|a_axis × b_axis|²` (Lagrange), a **length⁴**
/// quantity: a right-angle cell of extent `L` reads `L⁴`. An absolute floor on it is
/// therefore a floor on the *offset* — an `f32::EPSILON` one fires for every `L` below
/// `(1.19e-7)^(1/4) ≈ 0.0186` — and every vert of such a bevel would take the chord
/// fallback, collapsing the profile to a flat chamfer with no diagnostic. Judged on the
/// cell's own `sin θ` instead, the same junction takes the same branch at every scale and
/// the replanted shape is exactly similar.
#[test]
fn pipe_replant_cross_section_stays_curved_at_a_small_offset() {
    let hub = Vec3::new(1.1, -0.7, 0.0);
    for exponent in [2.0_f32, profile_to_exponent(0.02)] {
        let mut reference_shape = None;
        for offset in [1.0_f32, 0.2, 0.01, 0.001] {
            let cell = right_angle_cross_section(hub, offset);
            let station = 0.35;
            let base = Vec3::new(hub.x, hub.y, station);
            let replanted = replant(base, Vec3::Z, &cell, exponent);

            let projected_start = Vec3::new(cell.start.x, cell.start.y, station);
            let projected_end = Vec3::new(cell.end.x, cell.end.y, station);
            let chord = closest_point_on_segment(replanted, projected_start, projected_end);
            let clearance = (replanted - chord).length() / offset;
            assert!(
                clearance > 0.2,
                "at offset {offset} (exponent {exponent}) the replant answered \
                 {clearance} cell units off the chord — the curved branch declined and \
                 the cross-section collapsed to a flat chamfer",
            );

            // Similar at every scale: the cell-relative answer is offset-independent.
            let shape = (replanted - base) / offset;
            match reference_shape {
                None => reference_shape = Some(shape),
                Some(reference) => assert!(
                    (shape - reference).length() < 1.0e-3,
                    "the replanted shape is not scale-free: {shape:?} at offset {offset} \
                     against {reference:?} at offset 1",
                ),
            }
        }
    }
}

/// §8.5.5.5(f.6.2): with the cell solve **centred** on the vert being replanted, the
/// corner-coincident class is not merely "within a quantum" of the identity — it *is* the
/// identity, bit for bit, at every world magnitude.
///
/// A vert on the axis-parallel line through a projected control point has exact cell
/// coordinates `(1, 0)`, and the guide says the replant leaves it where it is. Taken off
/// the world origin the solve reaches that `0` by cancelling world-magnitude terms
/// (`base − (start' + end' − mid')`), so it arrives as a solve residue that the `r`-norm
/// then expands by `ε^r`; taken off `base`, every term is cell-scale, the projected control
/// point lands at the origin exactly, and the solve returns `1` and `0` with no residue at
/// all. The world-magnitude ladder is the point of the test: the uncentred form drifts with
/// it, the centred form does not.
///
/// The exactness is **centring's** doing and this test pins centring, not precision — it
/// passes unchanged with the routine's `f64` widening removed. The widening is kept for the
/// separate reason the routine's doc gives, and no test pins that.
#[test]
fn pipe_replant_is_the_exact_identity_on_corner_coincident_verts() {
    let exponent = profile_to_exponent(0.02);
    let offset = 0.2_f32;
    for world in [1.3_f32, 50.0, 1000.0] {
        let hub = Vec3::new(world, -0.6 * world, 0.0);
        let cell = right_angle_cross_section(hub, offset);
        for station_step in 0..4 {
            let station = -0.75 + 0.5 * station_step as f32;
            // The axis-parallel line through `end`: the cross-section is constant along the
            // pipe, so `end` projects onto this vert at every station.
            let base = Vec3::new(cell.end.x, cell.end.y, station);
            assert_eq!(
                replant(base, Vec3::Z, &cell, exponent),
                base,
                "corner-coincident vert moved at world magnitude {world}, station {station}",
            );
            // Same for the other projected control point.
            let base = Vec3::new(cell.start.x, cell.start.y, station);
            assert_eq!(
                replant(base, Vec3::Z, &cell, exponent),
                base,
                "corner-coincident vert moved at world magnitude {world}, station {station}",
            );
            // Non-vacuity: the same cell at the same station moves an off-corner vert.
            let off_corner = Vec3::new(hub.x, hub.y, station);
            let bulged = replant(off_corner, Vec3::Z, &cell, exponent);
            assert!(
                (bulged - off_corner).length() > 0.1 * offset,
                "the replant is the identity everywhere — the corner claim is vacuous",
            );
        }
    }
}

/// The §8.5.5.5(f.6) planar cross-section, tilted out of the pipe axis' plane by
/// `wobble` — a fraction of the cell's own extent. At `wobble = 0` the profile plane
/// contains the axis (`y`) and the projected triple is collinear; raising it lifts `mid`
/// off that plane and re-inflates the projected cell, sagitta `wobble · offset`.
fn near_planar_pipe_cell(offset: f32, wobble: f32) -> ProfileCell {
    ProfileCell::from_control_points(
        Vec3::new(offset, offset, 0.0),
        Vec3::new(-offset, offset, 0.0),
        Vec3::new(0.0, 0.0, wobble * offset),
    )
}

/// §8.5.5.5(f.6): the criterion separating the replant's placements is the *local* one the
/// guide states — the projected triple is collinear exactly when the profile plane through
/// `start`, `mid`, `end` is parallel to the pipe direction. A planar junction's cell takes
/// the chord, a tube's takes the cell path, and a cross-section that projects to a point
/// takes the point collapse.
#[test]
fn pipe_station_placement_reads_the_cell_against_the_axis() {
    assert_eq!(
        pipe_station_placement(Vec3::Y, &planar_pipe_cell(0.2)),
        StationPlacement::Chord,
        "the planar junction's profile plane contains the pipe axis",
    );
    assert_eq!(
        pipe_station_placement(
            Vec3::Z,
            &right_angle_cross_section(Vec3::new(1.1, -0.7, 0.0), 0.2)
        ),
        StationPlacement::Cell,
        "a tube's cross-section survives projection along its axis",
    );
    // `start` and `end` differ only along the axis, so both project onto one station point.
    let point_cell = ProfileCell::from_control_points(
        Vec3::new(0.0, 1.0, 0.0),
        Vec3::new(1.0, 1.0, 0.0),
        Vec3::new(0.0, 2.0, 0.0),
    );
    assert_eq!(
        pipe_station_placement(Vec3::X, &point_cell),
        StationPlacement::Point,
    );
}

/// §8.5.5.5(f.6): the placement verdict is **scale-free**, so the same junction shape takes
/// the same branch at every bevel offset. The determinant it judges is a length⁴ quantity,
/// so an absolute floor on it would be a floor on the *offset* — every small bevel would
/// read as a flat chamfer, and every large flat one as a tube.
#[test]
fn pipe_station_placement_is_scale_free() {
    for scale in [0.001_f32, 0.01, 0.1, 1.0, 12.0, 100.0, 1.0e3, 1.0e4] {
        let bias = Vec3::new(0.3 * scale, -0.2 * scale, 0.0);
        let planar = ProfileCell::from_control_points(
            Vec3::new(scale, scale, 0.0) + bias,
            Vec3::new(-scale, scale, 0.0) + bias,
            bias,
        );
        assert_eq!(
            pipe_station_placement(Vec3::Y, &planar),
            StationPlacement::Chord,
            "a flat junction keeps the chord clamp at scale {scale}",
        );
        assert_eq!(
            pipe_station_placement(Vec3::Z, &right_angle_cross_section(bias, scale)),
            StationPlacement::Cell,
            "a genuine tube keeps the cell path at scale {scale}",
        );
    }
}

/// §8.5.5.5(f.6): the verdict the (f.1) step-4 sentinel reads and the branch the replant
/// takes are **one** predicate, so they cannot disagree on a near-planar junction.
///
/// This is the seam that a second, junction-wide planarity fit opened. Its tolerance was an
/// order of magnitude looser than the replant's own degeneracy floor, so a junction wobbling
/// out of plane by (say) `3e-4` of its size was classified planar — the sentinel let a
/// square-in grid through — while every station's own test found a cell and ran the snap.
///
/// The ladder straddles the floor (`REPLANT_CELL_SHAPE_RATIO`, `1e-4` of the cell's extent).
/// The probe sits **inside** the chord's span and off the square-in walls, which is the only
/// place the two branches disagree at `r = 0`: the chord is the identity there, while
/// (f.2)'s branch S pushes the probe onto the nearer of the two cell edges through the
/// `(0, 0)` corner. Outside the span they agree by construction — those walls end at the
/// chord's own endpoints — so a probe far out of the band cannot see the seam at all.
///
/// The probe's cell coordinates are `(u, v) = (0.25, 0.75)`: `u` is the nearer wall, so
/// branch S sends it to `(0, 0.75)`, a quarter of the way from `start'` toward the corner,
/// at transverse `0.15` against the chord's identity `0.10`.
#[test]
fn pipe_station_placement_is_the_branch_the_replant_takes() {
    let offset = 0.2;
    let probe = Vec3::new(0.5 * offset, -0.0667, 0.0);
    let mut seen_chord = false;
    let mut seen_cell = false;
    for wobble in [0.0_f32, 1.0e-6, 1.0e-5, 3.0e-4, 1.0e-3, 3.0e-3, 1.0e-2, 0.1] {
        let cell = near_planar_pipe_cell(offset, wobble);
        let placement = pipe_station_placement(Vec3::Y, &cell);
        let replanted = replant(probe, Vec3::Y, &cell, 0.0);
        match placement {
            StationPlacement::Chord => {
                seen_chord = true;
                assert!(
                    (replanted.x - 0.5 * offset).abs() < 1.0e-6,
                    "wobble {wobble} was classified chord but the replant moved a probe \
                     already inside the chord's span: {replanted:?}",
                );
            }
            StationPlacement::Cell => {
                seen_cell = true;
                assert!(
                    (replanted.x - 0.75 * offset).abs() < 1.0e-6,
                    "wobble {wobble} was classified cell but the replant did not take \
                     branch S's nearer square-in wall: {replanted:?}",
                );
            }
            StationPlacement::Point => panic!("the cross-section spans the band"),
        }
        // The station survives either way, so it cannot be what tells the branches apart.
        assert!(
            (replanted.y - probe.y).abs() < 1.0e-6,
            "station preserved at {wobble}"
        );
    }
    assert!(
        seen_chord && seen_cell,
        "the wobble ladder must straddle the floor, or the agreement claim is vacuous",
    );
    // The seam the old two-predicate shape opened, named: a junction well inside a `1e-3`
    // relative planarity tolerance is *not* inside the replant's own, and the sentinel now
    // sees exactly that.
    assert_eq!(
        pipe_station_placement(Vec3::Y, &near_planar_pipe_cell(offset, 3.0e-4)),
        StationPlacement::Cell,
    );
}

/// BEVEL.md §8.5.5.5(f.6) vs §8.3.4.1: a **flat/planar** opposite-spokes junction trips
/// the §8.3.4.1 pipe predicate and does replant, but at the default circular profile its
/// whole cross-section stays inside the `±offset` band, so the chord clamp is the identity
/// and the base grid stands. The flat valence-4 fan is 4-fold symmetric about the origin,
/// so its interior verts stay in `z = 0` and their centroid sits on the origin; the
/// regression this pins dragged every interior vert onto a single boundvert chord,
/// clustering them off-centre. It is deliberately *not* the pin for whether the clamp
/// runs — at this profile it cannot see the difference
/// ([`planar_pipe_circular_clamp_is_the_identity_and_square_in_is_not`]).
#[test]
fn planar_pipe_override_leaves_the_flat_grid_unmoved() {
    let offset = 0.2;
    for (method, segments) in [
        (BevelVmeshMethod::Adj, 2),
        (BevelVmeshMethod::Adj, 3),
        (BevelVmeshMethod::Cutoff, 2),
        (BevelVmeshMethod::Cutoff, 3),
    ] {
        let (mut mesh, centre, rim) = planar_fan(12);
        // Two anti-parallel pairs (0°/180°, 90°/270°): a planar valence-4 pipe.
        let geom: Vec<GeomKey> = [0usize, 3, 6, 9]
            .into_iter()
            .map(|k| {
                mesh.query_edge_exists(centre, rim[k])
                    .expect("spoke")
                    .into()
            })
            .collect();
        let result = mesh.bevel(
            &geom,
            BevelOptions {
                offset,
                affect: BevelAffect::Edges,
                segments,
                vmesh_method: method,
                ..Default::default()
            },
        );
        mesh.validate_topology();
        let interior: Vec<Vec3> = result
            .verts
            .iter()
            .map(|&v| mesh.vertex_position(v))
            .collect();
        assert!(!interior.is_empty(), "the fill created interior verts");
        let centroid = interior.iter().sum::<Vec3>() / interior.len() as f32;
        assert!(
            centroid.length() < 1.0e-4,
            "flat pipe leaves the symmetric grid centred on the origin ({method:?}, seg {segments}): {centroid:?}",
        );
        assert!(
            interior.iter().all(|p| p.z.abs() < 1.0e-4),
            "flat pipe keeps every interior vert in the fan plane ({method:?}, seg {segments})",
        );
    }
}

/// §8.5.5.5(f.6): a **valence-3** tube junction (octant tri-corner base grid) at
/// `segments = 2` and `3`, ADJ fill. The base grid is re-tiled by the cross-section
/// replant, so the innermost cap vert lands exactly on the constant tube cross-section
/// (a rounded base cap would sit off it). Counts and the odd-`segments` central triangle
/// pin the base-grid topology.
#[test]
fn pipe_adj_three_way_tube_retiles_onto_cross_section() {
    let offset = 0.2;

    let (mesh, result) = tented_grid_pipe(&[3, 5, 7], offset, 2, BevelVmeshMethod::Adj);
    assert_eq!(mesh.vert_keys().len(), 21, "V");
    assert_eq!(mesh.edge_keys().len(), 33, "E");
    assert_eq!(mesh.face_keys().len(), 13, "F");
    let apex = innermost_cap_vert(&mesh, &result);
    assert!(
        (tented_pipe_cyl_distance(apex, offset) - offset).abs() < 1.0e-4,
        "the seg-2 pipe apex is re-planted onto the tube cross-section, not the rounded cap",
    );

    let (mesh, result) = tented_grid_pipe(&[3, 5, 7], offset, 3, BevelVmeshMethod::Adj);
    assert_eq!(mesh.vert_keys().len(), 29, "V");
    assert_eq!(mesh.edge_keys().len(), 48, "E");
    assert_eq!(mesh.face_keys().len(), 20, "F");
    let histogram = face_side_histogram(&mesh);
    assert_eq!(
        histogram.get(&3).copied().unwrap_or(0),
        1,
        "the valence-3 odd-`segments` base grid closes on one central triangle",
    );
    let central = innermost_cap_vert(&mesh, &result);
    assert!(
        (tented_pipe_cyl_distance(central, offset) - offset).abs() < 1.0e-4,
        "the seg-3 central-triangle vert is re-planted onto the tube cross-section",
    );
}

/// §8.5.5.5(f.6): a **valence-4** tube junction (general base grid) at `segments = 2`
/// and `3`, ADJ fill. Unlike the valence-3 case its odd-`segments` centre is a central
/// **quad** (`s = 4`), not a triangle — the general-grid vs octant-grid base
/// distinction — yet the same replant snaps the cap onto the constant cross-section.
#[test]
fn pipe_adj_four_way_tube_retiles_onto_cross_section() {
    let offset = 0.2;

    let (mesh, result) = tented_grid_pipe(&[3, 5, 7, 1], offset, 2, BevelVmeshMethod::Adj);
    assert_eq!(mesh.vert_keys().len(), 25, "V");
    assert_eq!(mesh.edge_keys().len(), 40, "E");
    assert_eq!(mesh.face_keys().len(), 16, "F");
    let apex = innermost_cap_vert(&mesh, &result);
    assert!(
        (tented_pipe_cyl_distance(apex, offset) - offset).abs() < 1.0e-4,
        "the seg-2 valence-4 welded apex is re-planted onto the tube cross-section",
    );

    let (mesh, result) = tented_grid_pipe(&[3, 5, 7, 1], offset, 3, BevelVmeshMethod::Adj);
    assert_eq!(mesh.vert_keys().len(), 36, "V");
    assert_eq!(mesh.edge_keys().len(), 60, "E");
    assert_eq!(mesh.face_keys().len(), 25, "F");
    let histogram = face_side_histogram(&mesh);
    assert_eq!(
        histogram.get(&3).copied().unwrap_or(0),
        0,
        "the valence-4 base grid closes on a central quad, so it emits no triangle",
    );
    let central = innermost_cap_vert(&mesh, &result);
    assert!(
        (tented_pipe_cyl_distance(central, offset) - offset).abs() < 1.0e-4,
        "the seg-3 valence-4 central vert is re-planted onto the tube cross-section",
    );
}

/// §8.5.5.5(f.6.1)'s `land_on_corner` rule, as the slot predicate it is stated as: for
/// **even** `segments`, the canonical slots with `k = ⌊n/2⌋` that either sit in one of the
/// two pipe boundverts' sectors or sit at `(i = 0, j = ⌊n/2⌋)` — the welded apex. Odd
/// `segments` set it nowhere.
///
/// The three neighbouring readings all differ on this table: gating on `j` instead of `k`
/// swaps the seam column for the ring row, dropping the parity test fires the rule on the
/// odd cap's central `s`-gon corners, and dropping the sector test admits the whole
/// `k = ⌊n/2⌋` column instead of two sectors' worth of it.
#[test]
fn pipe_replant_lands_on_corner_is_the_even_half_slot_rule() {
    // Sectors 0 and 2 are the pipe boundverts' (a valence-4 ring with the collinear pair
    // opposite each other); 1 and 3 are transverse.
    let pipe = (0usize, 2usize);

    // Odd `segments` never set it, anywhere on the net.
    for segments in [3usize, 5, 7] {
        let half = segments / 2;
        for sector in 0..4 {
            for ring in 0..=half {
                for along in 0..=half {
                    assert!(
                        !pipe_replant_lands_on_corner((sector, ring, along), segments, pipe),
                        "odd `segments` set the flag at ({sector}, {ring}, {along})",
                    );
                }
            }
        }
    }

    for segments in [2usize, 4, 6, 8] {
        let half = segments / 2;
        for sector in 0..4 {
            for ring in 1..=half {
                for along in 0..=half {
                    let expected = along == half
                        && (sector == pipe.0 || sector == pipe.1 || (sector == 0 && ring == half));
                    assert_eq!(
                        pipe_replant_lands_on_corner((sector, ring, along), segments, pipe),
                        expected,
                        "the flag at ({sector}, {ring}, {along}), segments = {segments}",
                    );
                }
            }
        }
        // The apex `(0, h, h)` is in, whichever sector the pipe boundverts fell on...
        assert!(pipe_replant_lands_on_corner(
            (0, half, half),
            segments,
            (1, 3)
        ));
        // ...and a transverse sector's own `k = h` column is out.
        if half >= 2 {
            assert!(!pipe_replant_lands_on_corner((1, 1, half), segments, pipe));
            assert!(pipe_replant_lands_on_corner((2, 1, half), segments, pipe));
        }
    }
}

/// §8.5.5.5(f.6) step 3 at the two sentinels: the replant resolves each vert through
/// (f.2)'s **branch S**, so a 3-D tube's cap lands on the square cross-section's walls —
/// and on its corner where `land_on_corner` says so.
///
/// The cell is the quarter cross-section of `pipe_replant_snaps_interior_onto_cross_section`
/// (`start = (0,1,0)`, `end = (0,0,1)`, `mid = (0,1,1)`, pipe axis `x`), whose cell frame is
/// exactly the `y–z` unit square: `(1,1)` is `mid`, `(1,0)` is `end`, `(0,1)` is `start` and
/// `(0,0)` is the corner `start + end − mid`, the origin. The probe at cell coordinates
/// `(u, v) = (0.3, 0.8)` therefore has a hand-checkable image under each of the four
/// combinations, and the station `x = 5` survives all four.
#[test]
fn pipe_replant_sentinel_cross_section_lands_on_the_cell_walls() {
    let cell = ProfileCell::from_control_points(
        Vec3::new(0.0, 1.0, 0.0),
        Vec3::new(0.0, 0.0, 1.0),
        Vec3::new(0.0, 1.0, 1.0),
    );
    let probe = Vec3::new(5.0, 0.8, 0.3);
    assert_eq!(
        pipe_station_placement(Vec3::X, &cell),
        StationPlacement::Cell,
        "the tube's projected control triple spans a cell",
    );
    let replant = |exponent: f32, corner: bool| {
        pipe_cross_section_replant(
            probe,
            Vec3::X,
            &cell,
            StationPlacement::Cell,
            exponent,
            corner,
        )
    };
    let square = PROFILE_SQUARE_EXPONENT;
    let square_in = 0.0_f32;

    // `square`: `v = 0.8` is the nearer of the two outer walls, so the probe rides out to
    // `v = 1` — the cell edge `start–mid` — keeping `u`.
    let wall = replant(square, false);
    assert!(
        (wall - Vec3::new(5.0, 1.0, 0.3)).length() < 1.0e-6,
        "square pushes onto the nearer outer wall: {wall:?}",
    );
    // ...and `land_on_corner` takes it all the way to the cell's `(1, 1)` corner, `mid`.
    let corner = replant(square, true);
    assert!(
        (corner - Vec3::new(5.0, 1.0, 1.0)).length() < 1.0e-6,
        "square + `land_on_corner` lands on `mid`: {corner:?}",
    );

    // `square-in`: the walls are the other pair, so `u = 0.3` is the nearer one and the
    // probe drops onto the cell edge `corner–start`.
    let inner_wall = replant(square_in, false);
    assert!(
        (inner_wall - Vec3::new(5.0, 0.8, 0.0)).length() < 1.0e-6,
        "square-in pushes onto the nearer inner wall: {inner_wall:?}",
    );
    // ...and the override lands on `start + end − mid`, here the cross-section's own origin.
    let inner_corner = replant(square_in, true);
    assert!(
        (inner_corner - Vec3::new(5.0, 0.0, 0.0)).length() < 1.0e-6,
        "square-in + `land_on_corner` lands on `start + end − mid`: {inner_corner:?}",
    );

    // Not a norm: the `r`-norm reading would send both sentinels to the same place (the
    // `r = 0` one to the origin for *every* vert), and the four images above are distinct.
    for (left, right) in [
        (wall, corner),
        (wall, inner_wall),
        (corner, inner_corner),
        (inner_wall, inner_corner),
    ] {
        assert!((left - right).length() > 0.1, "{left:?} vs {right:?}");
    }
}

/// §8.5.5.5(f.6.1): the replant's rim window is the strict **second half** of every
/// ring chain, `⌊n/2⌋ + 1 ..= n − 1`, with the same index range at both parities — the
/// guide's derived table. Empty at `segments <= 2`; never an endpoint (those are the
/// boundverts); never the even-`segments` chain midpoint `n/2` (that slot sits at
/// `j = 0`, outside the canonical `j >= 1` block).
#[test]
fn pipe_replant_rim_window_is_the_second_half_of_every_chain() {
    let table: [(usize, &[usize]); 9] = [
        (1, &[]),
        (2, &[]),
        (3, &[2]),
        (4, &[3]),
        (5, &[3, 4]),
        (6, &[4, 5]),
        (7, &[4, 5, 6]),
        (8, &[5, 6, 7]),
        (9, &[5, 6, 7, 8]),
    ];
    for (segments, expected) in table {
        let window: Vec<usize> = pipe_replant_rim_window(segments).collect();
        assert_eq!(
            window, expected,
            "the (f.6.1) rim window at segments = {segments}",
        );
        assert_eq!(
            window.len(),
            (segments + 1) / 2 - 1,
            "per-chain count is ⌈n/2⌉ − 1 at segments = {segments}",
        );
    }
}

/// §8.5.5.5(f.6.1): the window is not an independent rule — it is the image of §8.5.2's
/// canonical `j >= 1` block under the net's own `k = 0` seam alias
/// `(i, j, 0) == (i − 1, 0, n − j)`, the very index `adj_grid_vert` resolves a `k = 0`
/// slot through. Re-deriving it from the canonical `j` range at both parities is what
/// keeps the two readings from drifting apart.
#[test]
fn pipe_replant_rim_window_is_the_canonical_j_block_under_the_seam_alias() {
    for segments in 2..=9usize {
        let half = segments / 2;
        // Odd `n`: `j` runs `1 ..= ⌊n/2⌋`. Even `n`: the `j = n/2` row is a relabelled
        // neighbour (only its welded centre is canonical), so `j` runs `1 ..= n/2 − 1`.
        let last_j = if segments % 2 == 0 { half - 1 } else { half };
        let mut reached: Vec<usize> = (1..=last_j).map(|j| segments - j).collect();
        reached.sort_unstable();
        assert_eq!(
            reached,
            pipe_replant_rim_window(segments).collect::<Vec<_>>(),
            "the rim slots the canonical `j >= 1` block aliases onto at segments = {segments}",
        );
    }
}

/// The ring-chain samples sitting at one pipe station of the valence-4 tented pipe,
/// ordered along the chain's `y` extent. Both the transverse chains' far-end verts and
/// the flat skirt share the station plane, so the cross-section cylinder filters them
/// out — every vert of the axial chain lies within the tube radius, the rest do not.
fn tented_pipe_station_samples(mesh: &LoopMesh, station_x: f32, offset: f32) -> Vec<Vec3> {
    let mut samples: Vec<Vec3> = mesh
        .vert_keys()
        .into_iter()
        .map(|vertex| mesh.vertex_position(vertex))
        .filter(|position| (position.x - station_x).abs() < 1.0e-5)
        .filter(|position| tented_pipe_cyl_distance(*position, offset) <= offset + 1.0e-5)
        .collect();
    samples.sort_by(|a, b| a.y.partial_cmp(&b.y).expect("finite positions"));
    samples
}

/// §8.5.5.5(f.6.1): the replant moves exactly one of each mirror pair `(k, n − k)`, so a
/// symmetric tube's pipe cap is **not** mirror-symmetric — the count of stranded verts
/// within the replant's footprint (the two pipe stations, which is all the replant can
/// reach, being a projection *into* each vert's own station plane) is exactly
/// `4·(⌈n/2⌉ − 1)`.
///
/// Two independent exemptions make it vanish: **(A)** at `segments = 2` the window is
/// empty, and **(B)** at `segments ∈ {4, 8, …}` the rim is raw, so the replant is the
/// identity on it. Neither implies the other. An implementation that replants the whole
/// rim — or none of it — reports zero breaks at *every* count and fails this ladder.
#[test]
fn pipe_replant_window_breaks_the_tube_mirror_by_four_per_moved_sample() {
    let offset = 0.2;
    for segments in 2..=9u32 {
        let (mesh, _result) =
            tented_grid_pipe(&[3, 5, 7, 1], offset, segments, BevelVmeshMethod::Adj);
        let station: Vec<Vec3> = [1.0 - offset, 1.0 + offset]
            .into_iter()
            .flat_map(|station_x| tented_pipe_station_samples(&mesh, station_x, offset))
            .collect();
        // Three rows of this ladder expect zero breaks, so pin the footprint itself:
        // both stations must hold their whole pipe-edge chain, or a fixture that stopped
        // producing a cap (or a filter that stopped matching) would satisfy them vacuously.
        assert_eq!(
            station.len(),
            2 * (segments as usize + 1),
            "both pipe stations hold their whole chain at segments = {segments}",
        );
        let stranded = station
            .iter()
            .filter(|position| {
                let mirrored = Vec3::new(position.x, 2.0 - position.y, position.z);
                !station
                    .iter()
                    .any(|other| (*other - mirrored).length() < 1.0e-5)
            })
            .count();
        let raw_rim = segments >= 4 && segments.is_power_of_two();
        let expected = if raw_rim {
            0
        } else {
            4 * ((segments as usize + 1) / 2 - 1)
        };
        assert_eq!(
            stranded, expected,
            "mirror breaks within the replant footprint at segments = {segments}",
        );
    }
}

/// §8.5.5.5(f.6.1): the `land_on_corner` override, read off the finished mesh.
///
/// At a **square-in** exponent the profile degenerates onto the two cell edges through
/// §8.2.2.1's `(0, 0)` corner `start + end − mid`, so branch S's two walls are those edges
/// and their meeting point is that corner. On the tented tube the corner has a name: the
/// two cross-section legs are the offset skirts, `mid` is the ridge point, and
/// `start' + end' − mid'` is the **cross-section cylinder's own axis** — the ridge point
/// reflected through the chord's midpoint, at `offset·√2` below it. So a slot that lands on
/// the corner lands *exactly on the tube axis*, and a slot that lands on a wall does not.
///
/// The rule is `k = ⌊n/2⌋` **and even `segments`**, in one of the two pipe boundverts'
/// sectors or at the welded apex `(0, ⌊n/2⌋, ⌊n/2⌋)`. Counting the canonical interior slots
/// that satisfy it gives `2·(⌊n/2⌋ − 1) + 1` — two sectors' worth of the `k = ⌊n/2⌋` column
/// (`j = 1 … ⌊n/2⌋ − 1`) plus the apex — and **zero** at odd `segments`. That ladder is what
/// this asserts, and it separates the flag from every neighbouring reading: computing it at
/// both parities gives `⌊n/2⌋`-many extra hits at odd counts, dropping the sector test gives
/// `s·(⌊n/2⌋ − 1) + 1`, and dropping the flag altogether gives none at all.
///
/// The count is taken **strictly inside the two pipe boundverts' stations**, because the
/// chains crossing the pipe edges put their own `n/2` sample on the corner at those two
/// stations at every even count, replant or no replant, and the tube's two far terminal
/// caps sit on the axis as well.
#[test]
fn pipe_square_in_tube_lands_the_corner_slots_on_the_cross_section_axis() {
    let offset = 0.2;
    // (spokes, the pipe boundverts' own station offset from the junction).
    for (spokes, station) in [(&[3usize, 5, 7][..], 0.1_f32), (&[3, 5, 7, 1][..], 0.2)] {
        for segments in 2..=8u32 {
            let (mesh, result) =
                tented_grid_pipe_profile(spokes, offset, segments, BevelVmeshMethod::Adj, 0.0);
            let on_axis: Vec<Vec3> = result
                .verts
                .iter()
                .map(|&vertex| mesh.vertex_position(vertex))
                .filter(|position| (position.x - 1.0).abs() < station - 1.0e-4)
                .filter(|position| tented_pipe_cyl_distance(*position, offset) < 1.0e-5)
                .collect();
            let half = segments as usize / 2;
            let expected = if segments % 2 == 0 {
                2 * (half - 1) + 1
            } else {
                0
            };
            assert_eq!(
                on_axis.len(),
                expected,
                "corner-override slots on the tube axis (spokes {spokes:?}, \
                 segments = {segments})",
            );
            // *Which* two sectors, not merely how many. The two pipe boundverts' `k = h`
            // columns run **along** the tube, so their slots sit at distinct stations
            // straddling the junction, and only the welded apex is at the junction's own.
            // Naming the transverse sectors instead — the same count, one ring step away —
            // collapses every one of them onto the junction's station.
            let at_junction_station = on_axis
                .iter()
                .filter(|position| (position.x - 1.0).abs() < 1.0e-5)
                .count();
            assert_eq!(
                at_junction_station,
                expected.min(1),
                "only the welded apex sits at the junction's own station \
                 (spokes {spokes:?}, segments = {segments})",
            );
            for position in &on_axis {
                let mirrored = Vec3::new(2.0 - position.x, position.y, position.z);
                assert!(
                    on_axis
                        .iter()
                        .any(|other| (*other - mirrored).length() < 1.0e-5),
                    "the corner slots straddle the junction symmetrically; {position:?} \
                     has no partner (spokes {spokes:?}, segments = {segments})",
                );
            }
        }
    }
}

/// §8.5.5.5(f.1) step 1 vs step 2 — the dispatch order, as a predicate.
///
/// A square profile at a pipe corner satisfies **both** tests, and the square grid wins: a
/// junction step 1 claims never enters step 2, so no (f.6) post-pass runs over its
/// parallel-offset grid. The gate is the exponent, at least three beveled edges (which,
/// with no outer miter in scope, is the boundvert count) and an even `segments ≥ 2`.
///
/// This is pinned as a predicate rather than off a mesh on purpose. Branch S is very nearly
/// **idempotent** on the (f.5) lattice — that grid already lies on the square cross-section,
/// and the corner slots already sit at the cell's `(1, 1)` corner — so suppressing the
/// post-pass moves the tented tube's verts by roughly one ulp. The ordering is real and the
/// guide states it; it is simply not observable in a position comparison here.
#[test]
fn square_ring_outranks_the_pipe_test_at_an_even_segment_count() {
    let square = PROFILE_SQUARE_EXPONENT;
    let square_in = 0.0_f32;
    let circle = 2.0_f32;
    assert!(adj_square_ring_claims(square, 3, 2, false));
    assert!(adj_square_ring_claims(square, 4, 8, false));
    // Odd `segments` falls through to the pipe, and so does the square-in sentinel and
    // every smooth exponent.
    assert!(!adj_square_ring_claims(square, 4, 3, false));
    assert!(!adj_square_ring_claims(square, 4, 0, false));
    assert!(!adj_square_ring_claims(square_in, 4, 4, false));
    assert!(!adj_square_ring_claims(circle, 4, 4, false));
    // A two-boundvert ring is not the (f.5) ring's, whatever the parity.
    assert!(!adj_square_ring_claims(square, 2, 4, false));
    // Step 1's own `not a custom profile` conjunct: a custom bevel still carries a
    // `profile` slider, so a caller that resolves the exponent from it can hand this gate
    // the square sentinel — and every such ring falls through to step 4.
    assert!(!adj_square_ring_claims(square, 3, 2, true));
    assert!(!adj_square_ring_claims(square, 4, 8, true));

    // The grid the claim routes to survives: (f.5)'s even-`n` parallel-offset ring welds its
    // innermost vert at the corner `v` exactly, which the tented tube's junction puts at
    // `(1, 1, 1)`.
    for segments in [2u32, 4, 6] {
        let (mesh, _result) =
            tented_grid_pipe_profile(&[3, 5, 7, 1], 0.2, segments, BevelVmeshMethod::Adj, 0.97);
        assert!(
            mesh_holds_position(&mesh, Vec3::new(1.0, 1.0, 1.0)),
            "the (f.5) ring's even-`n` weld keeps `v` at segments = {segments}",
        );
    }
}

/// §8.5.5.5(f.1) step 2 + (f.2) branch S: the genuine 3-D tube **fills** at both square
/// sentinels. It used to decline — the base refused to hand a grid to a replant that had no
/// sentinel branch to resolve it with — and branch S is what removes that refusal.
///
/// The census is the statement: step 4's grid (and step 1's ring, and the tri-corner
/// squares) are exponent-generic in their counts, so a filled sentinel matches the circular
/// control exactly, `V` for `V`. Positions are not: the sentinel caps are a different shape
/// from the circular one and from each other, which is what keeps the census claim from
/// being satisfiable by a backend that silently fell back to the circle.
///
/// The planar §8.3.4.1 pipe is carried alongside as the contrast that always held: it takes
/// (f.6) step 4's chord clamp at every station, never reaches branch S, and keeps step 4's
/// grid at every exponent.
#[test]
fn pipe_sentinel_tube_retiles_instead_of_declining() {
    let offset = 0.2;
    for sentinel in [0.0_f32, 0.97] {
        for spokes in [&[3usize, 5, 7][..], &[3, 5, 7, 1][..]] {
            for segments in 2..=8u32 {
                let (control, _) =
                    tented_grid_pipe_profile(spokes, offset, segments, BevelVmeshMethod::Adj, 0.5);
                let (tube, _) = tented_grid_pipe_profile(
                    spokes,
                    offset,
                    segments,
                    BevelVmeshMethod::Adj,
                    sentinel,
                );
                assert_eq!(
                    (
                        tube.vert_keys().len(),
                        tube.edge_keys().len(),
                        tube.face_keys().len(),
                    ),
                    (
                        control.vert_keys().len(),
                        control.edge_keys().len(),
                        control.face_keys().len(),
                    ),
                    "the 3-D tube fills at profile {sentinel} (spokes {spokes:?}, \
                     segments = {segments})",
                );
                // Non-vacuity: the sentinel cap is a different object from the circular one.
                let moved = tube
                    .vert_keys()
                    .into_iter()
                    .any(|vertex| !mesh_holds_position(&control, tube.vertex_position(vertex)));
                assert!(
                    moved,
                    "the sentinel cap differs from the circular control at profile \
                     {sentinel} (spokes {spokes:?}, segments = {segments})",
                );
            }
        }
    }

    // The planar pipe never reached branch S and never declined; it still matches.
    for segments in [3u32, 5] {
        for sentinel in [0.0_f32, 0.97] {
            for spokes in [&[1usize, 4, 7, 10][..], &[1, 7, 4][..]] {
                let control = planar_fan_pipe_profile(spokes, offset, segments, 0.5);
                let filled = planar_fan_pipe_profile(spokes, offset, segments, sentinel);
                assert_eq!(
                    (
                        filled.vert_keys().len(),
                        filled.edge_keys().len(),
                        filled.face_keys().len(),
                    ),
                    (
                        control.vert_keys().len(),
                        control.edge_keys().len(),
                        control.face_keys().len(),
                    ),
                    "the planar pipe keeps step 4's grid at profile {sentinel} \
                     (spokes {spokes:?}, segments = {segments})",
                );
            }
        }
    }
}

/// A planar §8.3.4.1 pipe: the flat `spokes`-triangle fan with an anti-parallel opposite
/// pair (and, at four spokes, two of them) beveled. Same predicate as the tented tube,
/// but the whole junction lies in `z = 0`, so its profile plane contains the pipe axis
/// and the (f.6) replant is the chord clamp at every station rather than the cell path.
fn planar_fan_pipe_profile(
    beveled_spokes: &[usize],
    offset: f32,
    segments: u32,
    profile: f32,
) -> LoopMesh {
    let (mut mesh, centre, rim) = planar_fan_axis_aligned(12);
    let geom: Vec<GeomKey> = beveled_spokes
        .iter()
        .map(|&k| {
            mesh.query_edge_exists(centre, rim[k])
                .expect("spoke")
                .into()
        })
        .collect();
    mesh.bevel(
        &geom,
        BevelOptions {
            offset,
            affect: BevelAffect::Edges,
            segments,
            vmesh_method: BevelVmeshMethod::Adj,
            profile,
            ..Default::default()
        },
    );
    mesh.validate_topology();
    mesh
}

/// Whether `mesh` carries a vertex at `position`, to the `1e-5` position quantum.
fn mesh_holds_position(mesh: &LoopMesh, position: Vec3) -> bool {
    mesh.vert_keys().into_iter().any(|vertex| {
        (mesh.vertex_position(vertex) - position)
            .abs()
            .max_element()
            < 1.0e-5
    })
}

/// §8.5.5.5(f.6)'s worked planar case, at its own figures. A flat 12-spoke fan beveled on
/// four (valence 4) or three (valence 3) spokes at `offset = 0.2`, `profile = 0`
/// (square-in), read in the frame the guide states: the pipe axis is a coordinate
/// direction, and the clamped transverse coordinate is the other one.
///
/// At a square-in exponent the profile degenerates onto the two cell edges through
/// §8.2.2.1's `(0, 0)` corner `start + end − mid`, which — the rescue having put `mid` at
/// the junction vertex — sits at `±2·offset` transverse on the two chains crossing the
/// **transverse** beveled edges. So the base grid leaves slots outside the `±offset`
/// band, and the replant clamps them back to the band edge *exactly* while the along-axis
/// coordinate survives to the last float. That one-coordinate-agrees-one-does-not shape is
/// the whole fingerprint of getting this wrong.
///
/// The clamp is gated by [(f.6.1)]'s replant window like the cell path, so at
/// `segments = 3` only **one slot per chain** moves: of the four out-of-band slots, two
/// come back and two stay put, leaving the cap *less* symmetric than the fixture. A
/// whole-grid clamp moves all four and looks tidier.
///
/// **Where each row's numbers come from.** Row 1 is the guide's own worked figure. The
/// other two are the same construction re-run, not readings off the implementation:
///
/// - The chain crossing a transverse edge runs between the two boundverts flanking that
///   edge; at square-in the profile degenerates onto the two cell edges through the `(0,0)`
///   corner `start + end − mid`, which — the rescue having put `mid` at the junction vertex
///   — is the junction vertex reflected through the chord's midpoint. On the valence-4 fan
///   that corner is at `(∓0.4, 0)`, on the valence-3 fan at `(0, +0.4)`: `±2·offset`
///   transverse either way.
/// - The `[S]` sampler walks that two-leg polyline uniformly, so sample `k` of `n` sits at
///   cell coordinate `u = 1 − 2k/n` along the first leg. `n = 3` puts sample 1 at `u = 1/3`
///   (the guide's own row), `n = 4` puts it at `u = 1/2` — i.e. at the leg's midpoint,
///   `corner + ½·(start − corner)`, which for the valence-4 fan is `(−0.3, −0.1)`.
/// - The clamp then replaces the transverse coordinate by the nearest end of `[−0.2, 0.2]`
///   and leaves the along-axis one alone, which is every row's second column.
///
/// *Which* of the corner's two legs the (f.6.1) window reaches is the ring walk's, and only
/// that is a snapshot — the valence-4 rows pin the unreached leg explicitly below.
#[test]
fn planar_pipe_square_in_clamps_the_replant_window() {
    let offset = 0.2;
    // (spokes, segments, out-of-band slot, its clamped image). Valence 4 names the
    // `90°/270°` pair as the axis, so its transverse coordinate is `x`; valence 3 has only
    // the `0°/180°` pair, so its transverse coordinate is `y`.
    let worked: [(&[usize], u32, Vec3, Vec3); 3] = [
        (
            &[1, 4, 7, 10],
            3,
            Vec3::new(-1.0 / 3.0, -1.0 / 15.0, 0.0),
            Vec3::new(-offset, -1.0 / 15.0, 0.0),
        ),
        (
            &[1, 4, 7, 10],
            4,
            Vec3::new(-0.3, -0.1, 0.0),
            Vec3::new(-offset, -0.1, 0.0),
        ),
        (
            &[1, 7, 4],
            3,
            Vec3::new(-1.0 / 15.0, 1.0 / 3.0, 0.0),
            Vec3::new(-1.0 / 15.0, offset, 0.0),
        ),
    ];
    for (spokes, segments, out_of_band, clamped) in worked {
        let mesh = planar_fan_pipe_profile(spokes, offset, segments, 0.0);
        assert!(
            mesh_holds_position(&mesh, clamped),
            "the replant clamped {out_of_band:?} onto the chord at {clamped:?} \
             (spokes {spokes:?}, segments = {segments})",
        );
        assert!(
            !mesh_holds_position(&mesh, out_of_band),
            "no vert is left outside the ±offset band at {out_of_band:?} \
             (spokes {spokes:?}, segments = {segments})",
        );
    }

    // The window's other half: at valence 4, `segments = 3`, the four out-of-band slots are
    // a single orbit of the fixture's own symmetry, and the window `{2}` admits one per
    // chain — so the two the walk does *not* reach must still be sitting outside the band.
    let mesh = planar_fan_pipe_profile(&[1, 4, 7, 10], offset, 3, 0.0);
    for stranded in [
        Vec3::new(1.0 / 3.0, -1.0 / 15.0, 0.0),
        Vec3::new(-1.0 / 3.0, 1.0 / 15.0, 0.0),
    ] {
        assert!(
            mesh_holds_position(&mesh, stranded),
            "the replant window leaves {stranded:?} outside the band — clamping the whole \
             grid instead would return all four slots and over-symmetrise the cap",
        );
    }
}

/// §8.5.5.5(f.6): at a **circular** profile the planar junction's clamp is the identity —
/// the cross-section never leaves the `±offset` band, so every slot's nearest point on the
/// chord is the slot itself — while at a **concave** one it moves real verts. The
/// observable is the fixture's own 4-fold symmetry: the circular cap keeps it, the
/// square-in cap does not, because the (f.6.1) window admits one of each mirror pair.
///
/// This is also why a check run at `profile = 0.5` cannot tell "the clamp ran" from "the
/// clamp was skipped", and is the wrong place to pin the planar replant.
#[test]
fn planar_pipe_circular_clamp_is_the_identity_and_square_in_is_not() {
    let offset = 0.2;
    let quarter_turn = |position: Vec3| Vec3::new(-position.y, position.x, position.z);
    let symmetric = |mesh: &LoopMesh| {
        mesh.vert_keys()
            .into_iter()
            .all(|vertex| mesh_holds_position(mesh, quarter_turn(mesh.vertex_position(vertex))))
    };
    for segments in [3u32, 5] {
        assert!(
            symmetric(&planar_fan_pipe_profile(
                &[1, 4, 7, 10],
                offset,
                segments,
                0.5
            )),
            "the circular planar pipe's clamp is inert, so the cap keeps the fixture's \
             4-fold symmetry (segments = {segments})",
        );
        assert!(
            !symmetric(&planar_fan_pipe_profile(
                &[1, 4, 7, 10],
                offset,
                segments,
                0.0
            )),
            "the square-in planar pipe's clamp reaches one of each mirror pair, so the cap \
             breaks that symmetry (segments = {segments})",
        );
    }
}

/// §8.5.5.5(f.6.1): the rim window and the net's `k = 0` column must name the **same
/// verts**. The window is the image of §8.5.2's canonical `j >= 1` block under the seam
/// alias `(i, j, 0) == (i − 1, 0, n − j)`, so the set of keys `adj_grid_vert` hands out
/// for `k = 0` over that block has to equal the set the replant's window walks — pinning
/// the derivation against the code that *resolves* a slot, not against itself.
///
/// Set equality is the right statement (and all this can see): the footprint is a union
/// over sub-patches, so which *sub-patch* reaches a given chain is not observable here,
/// while the alias index — the half of the chain that moves — is.
#[test]
fn pipe_replant_rim_window_touches_exactly_the_net_k0_column() {
    for segments in 2..=9usize {
        for sectors in [3usize, 4] {
            let mut mesh = LoopMesh::new();
            // One distinct vert per chain sample, so key equality is index equality.
            let chain_of: Vec<Vec<VertexKey>> = (0..sectors)
                .map(|i| {
                    (0..=segments)
                        .map(|k| mesh.create_vertex(Vec3::new(i as f32, k as f32, 0.0)))
                        .collect()
                })
                .collect();

            let half = segments / 2;
            let even = segments % 2 == 0;
            // §8.5.2's canonical `j` range: `1 ..= ⌊n/2⌋` at odd `n`, and `1 ..= n/2 − 1`
            // at even `n` (the `j = n/2` row is a relabelled neighbour there).
            let last_j = if even { half - 1 } else { half };
            let interior = AdjInterior::Blend { corners: &[] };
            let mut grid = std::collections::HashMap::new();
            let mut result = BevelResult::default();
            let mut emitter: std::collections::HashSet<VertexKey> =
                std::collections::HashSet::new();
            for i in 0..sectors {
                for j in 1..=last_j {
                    emitter.insert(mesh.adj_grid_vert(
                        i,
                        j,
                        0,
                        half,
                        segments,
                        sectors,
                        even,
                        &chain_of,
                        &interior,
                        None,
                        None,
                        &mut grid,
                        &mut AdjNetSlots::off(),
                        &mut result,
                    ));
                }
            }

            let mut replanted: std::collections::HashSet<VertexKey> =
                std::collections::HashSet::new();
            for chain in &chain_of {
                for sample in pipe_replant_rim_window(segments) {
                    replanted.insert(chain[sample]);
                }
            }

            assert_eq!(
                emitter.len(),
                sectors * last_j,
                "the canonical `k = 0` block has one distinct vert per (sector, j) \
                 (segments = {segments}, sectors = {sectors})",
            );
            assert_eq!(
                emitter, replanted,
                "the replant window walks the net's `k = 0` column exactly \
                 (segments = {segments}, sectors = {sectors})",
            );
        }
    }
}

/// The replant reads its samples one at a time rather than materialising the sector's
/// oriented chain, which is only sound if the two agree index-for-index from **either**
/// end — the `k = 0` column is walked from the far boundvert, so the reversed sense is
/// the one that carries the §8.5.5.5(f.6.1) window.
#[test]
fn oriented_sample_indexes_the_oriented_chain() {
    let mut mesh = LoopMesh::new();
    let verts: Vec<VertexKey> = (0..5)
        .map(|i| mesh.create_vertex(Vec3::new(i as f32, 0.0, 0.0)))
        .collect();
    let edge = mesh.create_edge([verts[0], verts[4]], EdgeCreateFlags::NONE);
    let chain = EdgeEndChain {
        edge,
        endpoint: verts[0],
        start: verts[0],
        interior: verts[1..4].to_vec(),
        end: verts[4],
        side_label: SideLabel::Unlabelled,
        cell: None,
        custom_reverse: EdgeEndChain::UNMARKED_DEFAULT_REVERSE,
    };
    for first in [verts[0], verts[4]] {
        let oriented = chain.oriented_from(first);
        for (index, &expected) in oriented.iter().enumerate() {
            assert_eq!(
                chain.oriented_sample(first, index),
                Some(expected),
                "sample {index} of the walk from {first:?}",
            );
        }
        assert_eq!(chain.oriented_sample(first, oriented.len()), None);
    }
}

/// §8.5.5.5(f.6.1): *which* half moves. On each pipe-edge chain the replanted samples
/// are a contiguous run touching one boundvert — the chain's far end in the net's ring
/// order — while the other half keeps the chord point §8.2.2.1.1's read-back left it
/// strictly inside the cross-section. The ring order walks *around* the junction, so the
/// two pipe-edge chains are traversed in opposite senses and the two stations replant
/// **opposite** ends: that is the axial mirror break, and it is why replanting the whole
/// rim (every sample on the curve) or none of it (none on the curve) both diverge.
///
/// Odd `segments` only, so that neither exemption applies and the chain has no midpoint
/// sample to sit outside the window while still being a fixed point of the read-back.
#[test]
fn pipe_replant_window_moves_one_contiguous_end_of_each_pipe_edge_chain() {
    let offset = 0.2;
    for segments in [3u32, 5, 7, 9] {
        let seg = segments as usize;
        let (mesh, _result) =
            tented_grid_pipe(&[3, 5, 7, 1], offset, segments, BevelVmeshMethod::Adj);
        let mut high_y_end_moved = Vec::with_capacity(2);
        for station_x in [1.0 - offset, 1.0 + offset] {
            let samples = tented_pipe_station_samples(&mesh, station_x, offset);
            assert_eq!(
                samples.len(),
                seg + 1,
                "the station holds the whole pipe-edge chain (segments = {segments})",
            );
            let on_curve: Vec<bool> = samples
                .iter()
                .map(|position| {
                    (tented_pipe_cyl_distance(*position, offset) - offset).abs() < 1.0e-5
                })
                .collect();
            assert!(
                on_curve[0] && on_curve[seg],
                "both boundverts sit on the cross-section (segments = {segments})",
            );
            let interior = &on_curve[1..seg];
            assert_eq!(
                interior.iter().filter(|moved| **moved).count(),
                (seg + 1) / 2 - 1,
                "⌈n/2⌉ − 1 samples replanted onto the cross-section (segments = {segments})",
            );
            let moved_at_high_y = interior[interior.len() - 1];
            // One run touching an end ⟺ the flags are monotone in walk order.
            let contiguous = interior
                .windows(2)
                .all(|pair| pair[0] == pair[1] || pair[1] == moved_at_high_y);
            assert!(
                contiguous,
                "the replanted samples form one run touching a boundvert \
                 (segments = {segments}, station x = {station_x}): {interior:?}",
            );
            high_y_end_moved.push(moved_at_high_y);
        }
        assert_ne!(
            high_y_end_moved[0], high_y_end_moved[1],
            "the two pipe-edge chains are walked in opposite senses, so they replant \
             opposite ends (segments = {segments})",
        );
    }
}

/// §8.3.4.1 + §8.5.5.5(f.6): the CUTOFF pipe override delegates its connected fill to
/// the ADJ path, so the cross-section replant runs there too — the connected cap lands
/// on the tube cross-section, and the override still mints its `(segments − 1)·s` loose
/// leftovers on top (counts = the ADJ fill plus the loose verts).
#[test]
fn pipe_cutoff_override_tube_retiles_like_adj_plus_loose() {
    let offset = 0.2;

    let (mesh, result) = tented_grid_pipe(&[3, 5, 7], offset, 3, BevelVmeshMethod::Cutoff);
    assert_eq!(mesh.vert_keys().len(), 35, "V = 29 ADJ + 6 loose");
    assert_eq!(mesh.edge_keys().len(), 48, "E matches ADJ");
    assert_eq!(mesh.face_keys().len(), 20, "F matches ADJ");
    assert_eq!(
        loose_vertex_count(&mesh),
        6,
        "(segments − 1)·s = 2·3 loose leftovers"
    );
    // The connected cap vert (loose leftovers are valence-0, so the innermost *connected*
    // vert is the cap) sits on the tube cross-section — the replant reached the CUTOFF path.
    let connected_cap = result
        .verts
        .iter()
        .copied()
        .filter(|&v| mesh.iter_disk_edges(v).next().is_some())
        .map(|v| mesh.vertex_position(v))
        .min_by(|a, b| {
            let da = (*a - Vec3::new(1.0, 1.0, 1.0)).length();
            let db = (*b - Vec3::new(1.0, 1.0, 1.0)).length();
            da.partial_cmp(&db).unwrap()
        })
        .expect("connected cap vert");
    assert!(
        (tented_pipe_cyl_distance(connected_cap, offset) - offset).abs() < 1.0e-4,
        "the CUTOFF-override connected cap is re-planted onto the tube cross-section",
    );

    let (mesh, _r) = tented_grid_pipe(&[3, 5, 7, 1], offset, 2, BevelVmeshMethod::Cutoff);
    assert_eq!(mesh.vert_keys().len(), 29, "valence-4 V = 25 ADJ + 4 loose");
    assert_eq!(
        loose_vertex_count(&mesh),
        4,
        "(segments − 1)·s = 1·4 loose leftovers"
    );
}
