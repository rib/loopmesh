use super::*;

/// BEVEL.md §4: ring verts slide `offset` along each incident edge toward the
/// far endpoint. A larger offset slides them proportionally further.
#[test]
fn ring_verts_slide_proportionally() {
    let make = |offset: f32| {
        let mut mesh = LoopMesh::new();
        mesh_add_cube(&mut mesh);
        let corner = mesh
            .vert_keys()
            .into_iter()
            .find(|&v| mesh.vertex_position(v) == Vec3::new(1.0, -1.0, -1.0))
            .unwrap();
        let result = mesh.bevel(
            &[corner.into()],
            BevelOptions {
                offset,
                ..Default::default()
            },
        );
        // Distance of each ring vert from the (known) original corner.
        let origin = Vec3::new(1.0, -1.0, -1.0);
        result
            .verts
            .iter()
            .map(|&v| (mesh.vertex_position(v) - origin).length())
            .collect::<Vec<_>>()
    };

    let small = make(0.25);
    let large = make(0.5);
    for distance in &small {
        assert!(
            (distance - 0.25).abs() < 1.0e-5,
            "small slide is the offset"
        );
    }
    for distance in &large {
        assert!((distance - 0.5).abs() < 1.0e-5, "large slide is the offset");
    }
}

/// BEVEL.md §6.2 / §2.3: a boundary valence-2 corner of a flat grid quad
/// becomes an open chamfer — two ring verts, one chamfer edge, no cap.
/// Count delta +1 V / +1 E / +0 F.
#[test]
fn boundary_open_fan_grid_corner() {
    let mut mesh = LoopMesh::new();
    let positions = [
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(1.0, 1.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
    ];
    let verts: Vec<VertexKey> = positions.iter().map(|&p| mesh.create_vertex(p)).collect();
    let edges: Vec<EdgeKey> = (0..4)
        .map(|i| mesh.create_edge([verts[i], verts[(i + 1) % 4]], EdgeCreateFlags::NONE))
        .collect();
    mesh.face_create(&verts, &edges, FaceCreateFlags::NONE);

    let before = counts(&mesh);
    assert_eq!(before, (4, 4, 1));

    let corner = verts[0];
    let result = mesh.bevel(
        &[corner.into()],
        BevelOptions {
            offset: 0.25,
            ..Default::default()
        },
    );

    assert_eq!(counts(&mesh), (5, 5, 1), "boundary +1 V / +1 E / +0 F");
    assert_eq!(result.verts.len(), 2, "two ring verts");
    assert!(result.faces.is_empty(), "no cap on an open fan");

    // The single original face is now a pentagon (quad's corner cut off).
    let face = mesh.face_keys()[0];
    assert_eq!(mesh.face_vert_keys(face).len(), 5, "corner cut into two");

    mesh.validate_topology();
}

/// BEVEL.md §6.2: the open fan **is still capped** from valence 3 up. The ring is the disk
/// *cycle*, and a cycle has no ends — the gap merely means one ring edge had no incident
/// face to have minted it during the re-stitch, so the cap mints that one itself. Counts are
/// the interior counts: `+(n−1)` V / `+n` E / `+1` F.
///
/// The valence-2 grid corner above is the *one* case where "no cap on a boundary" is true
/// (a two-corner n-gon is not a face), which is exactly why it is not a general rule.
#[test]
fn boundary_open_fan_valence_three_is_still_capped() {
    let (mut mesh, centre, _) = open_planar_fan(3, 0);
    assert_eq!(counts(&mesh), (4, 5, 2));

    let result = mesh.bevel(
        &[centre.into()],
        BevelOptions {
            offset: 0.25,
            segments: 1,
            affect: BevelAffect::Vertices,
            ..Default::default()
        },
    );
    mesh.validate_topology();

    assert_eq!(counts(&mesh), (6, 8, 3), "boundary +2 V / +3 E / +1 F");
    assert_eq!(result.verts.len(), 3, "three ring verts");
    assert_eq!(result.faces.len(), 1, "one triangular cap");
    assert_eq!(
        face_size_multiset(&mesh),
        vec![3, 4, 4],
        "both triangles grown by one, plus the cap over all three ring verts",
    );

    let cap = result.faces[0];
    assert_eq!(mesh.face_vert_keys(cap).len(), 3);

    // §6.2: the two chamfer edges keep radial count 2 (their face plus the cap); the gap
    // edge the cap minted has radial count 1 — a new boundary edge, which is what keeps a
    // boundary corner on the boundary.
    let mut radial_counts: Vec<usize> = mesh
        .face_loop_keys(cap)
        .into_iter()
        .map(|loop_key| mesh.edge_radial_iter(mesh.loop_edge(loop_key)).count())
        .collect();
    radial_counts.sort_unstable();
    assert_eq!(radial_counts, vec![1, 2, 2], "one gap edge, two chamfers");
}

/// BEVEL.md §6.2's multi-gap row, measured against the reference: a **bow-tie** corner — two
/// triangles meeting at one shared vertex and sharing no edge. Manifoldness is an edge-mode
/// precondition (§2.1), so the corner is eligible, and its two disjoint face chains are
/// concatenated into a single disk cycle whose between-chain steps are the gaps. Valence 4,
/// `s = 2` faced steps and 2 gap steps: `5/6/2 → 8/10/3`, sizes `[4, 4, 4]` — **one** quad
/// cap over all four ring verts, closing both gaps, and not a cap per fan.
#[test]
fn vertex_bevel_multi_gap_bowtie_corner_gets_one_quad_cap() {
    let mut mesh = LoopMesh::new();
    let centre = mesh.create_vertex(Vec3::ZERO);
    let east = mesh.create_vertex(Vec3::new(1.0, 0.0, 0.0));
    let north_east = mesh.create_vertex(Vec3::new(1.0, 1.0, 0.0));
    let west = mesh.create_vertex(Vec3::new(-1.0, 0.0, 0.0));
    let south_west = mesh.create_vertex(Vec3::new(-1.0, -1.0, 0.0));
    mesh.face_create_verts(&[centre, east, north_east], FaceCreateFlags::NONE);
    mesh.face_create_verts(&[centre, west, south_west], FaceCreateFlags::NONE);
    assert_eq!(counts(&mesh), (5, 6, 2));

    let result = mesh.bevel(
        &[centre.into()],
        BevelOptions {
            offset: 0.25,
            segments: 1,
            affect: BevelAffect::Vertices,
            ..Default::default()
        },
    );
    mesh.validate_topology();

    assert_eq!(counts(&mesh), (8, 10, 3), "+3 V / +4 E / +1 F");
    assert_eq!(
        result.faces.len(),
        1,
        "one cap over the whole ring, not one per fan"
    );
    assert_eq!(face_size_multiset(&mesh), vec![4, 4, 4]);

    let cap = result.faces[0];
    assert_eq!(
        mesh.face_vert_keys(cap).len(),
        4,
        "the cap spans all four ring verts",
    );
    for &ring_vertex in &result.verts {
        assert!(
            mesh.face_vert_keys(cap).contains(&ring_vertex),
            "every ring vert is a corner of the one cap",
        );
    }
    let mut radial_counts: Vec<usize> = mesh
        .face_loop_keys(cap)
        .into_iter()
        .map(|loop_key| mesh.edge_radial_iter(mesh.loop_edge(loop_key)).count())
        .collect();
    radial_counts.sort_unstable();
    assert_eq!(
        radial_counts,
        vec![1, 1, 2, 2],
        "two gap edges minted by the cap, two chamfers minted by the re-stitch",
    );
}

/// BEVEL.md §8.11 Stage [VV]: the two-quad-plane valence-3 corner at `segments = 3`
/// builds the real vmesh patch — the §4 slide ring (3 boundverts), three `mid = v`
/// profile chains (2 interior samples each), and the [V-ADJ] general odd-`segments`
/// grid (3 sub-patch quads + 3 seam quads + 1 central triangle). `6/7/2 → 17/25/9`,
/// the shape the reference produces. The chain interior verts are shared between the
/// cap and the two incident faces (grown to 7-gons); the third sector is the boundary
/// gap, whose chain the cap mints alone.
#[test]
fn vertex_bevel_multiseg_two_quad_plane_valence_three_builds_the_vmesh_patch() {
    let (mut mesh, _) = two_quad_strip();
    let corner = vert_at(&mesh, Vec3::new(0.0, -1.0, 0.0));
    assert_eq!(counts(&mesh), (6, 7, 2));

    let result = mesh.bevel(
        &[corner.into()],
        BevelOptions {
            offset: 0.25,
            segments: 3,
            affect: BevelAffect::Vertices,
            ..Default::default()
        },
    );
    mesh.validate_topology();

    assert_eq!(
        counts(&mesh),
        (17, 25, 9),
        "§8.11 Stage [VV]: 6/7/2 → 17/25/9"
    );
    // 3 ring + 3·2 chain interior + 3 patch interior.
    assert_eq!(
        result.verts.len(),
        12,
        "3 ring + 6 chain + 3 patch interior"
    );
    // 3 sub-patch quads + 3 odd seam quads + 1 central triangle.
    assert_eq!(result.faces.len(), 7, "the [V-ADJ] odd-segments grid");
    assert_eq!(
        face_size_multiset(&mesh),
        vec![3, 4, 4, 4, 4, 4, 4, 7, 7],
        "seven grid faces (a central triangle + six quads) and two grown 7-gons",
    );
}

/// BEVEL.md §6.6.1's per-sector grid rule (affect == VERTICES): each sector's
/// [V-ADJ] fill faces inherit their whole per-face block — material slot included —
/// from the **next** boundvert's representative, distributing the donor across the
/// corner's incident faces. On the material-straddling two-quad plane (quad A =
/// slot 5 on x<0, quad B = slot 2 on x>0) the valence-3 shared-edge corner at
/// `segments = 3` splits its fill: the two sector faces fed by the quad-A-side
/// boundvert land on slot 5, the rest (and the ring-spanning centre n-gon) on
/// slot 2 — so the whole mesh carries three slot-5 faces and six slot-2 faces. A
/// regression to one corner-wide donor collapses every fill face onto a single
/// slot and gives {5: 1, 2: 8}.
#[test]
fn vertex_bevel_multiseg_fill_material_distributes_per_sector() {
    let (mut mesh, _) = two_quad_strip();
    let quad_a_marker = vert_at(&mesh, Vec3::new(-2.0, -1.0, 0.0));
    let quad_b_marker = vert_at(&mesh, Vec3::new(2.0, -1.0, 0.0));
    for face in mesh.face_keys() {
        let verts = mesh.face_vert_keys(face);
        if verts.contains(&quad_a_marker) {
            mesh.set_face_material_index(face, 5);
        } else if verts.contains(&quad_b_marker) {
            mesh.set_face_material_index(face, 2);
        }
    }

    let corner = vert_at(&mesh, Vec3::new(0.0, -1.0, 0.0));
    let result = mesh.bevel(
        &[corner.into()],
        BevelOptions {
            offset: 0.25,
            segments: 3,
            affect: BevelAffect::Vertices,
            ..Default::default()
        },
    );
    mesh.validate_topology();

    let tally = |materials: &[i32]| -> (usize, usize) {
        (
            materials.iter().filter(|&&m| m == 5).count(),
            materials.iter().filter(|&&m| m == 2).count(),
        )
    };

    // The two grown originals keep their own slots (A=5, B=2); the seven fill faces
    // split 2/5 across the two donors — matching the reference's per-sector
    // distribution, whose material multiset is three 5s and six 2s.
    let all: Vec<i32> = mesh
        .face_keys()
        .into_iter()
        .map(|face| mesh.face_material_index(face))
        .collect();
    assert_eq!(all.len(), 9);
    assert_eq!(
        tally(&all),
        (3, 6),
        "three slot-5 faces and six slot-2 — the reference's per-sector distribution",
    );

    let fill: Vec<i32> = result
        .faces
        .iter()
        .map(|&face| mesh.face_material_index(face))
        .collect();
    assert_eq!(fill.len(), 7);
    assert_eq!(
        tally(&fill),
        (2, 5),
        "the two quad-A-side sector fills inherit slot 5; the rest and the centre \
         n-gon take slot 2 — a corner-wide donor would give (0, 7)",
    );
}

/// BEVEL.md §8.11.2 [LOAD-BEARING]: the profile is `mid = v`, `projection_dir = 0` but
/// **not** flagged linear, so the interior samples bulge *toward the corner vertex* and
/// do not collapse onto the straight `B_k → B_(k+1)` chord. Measured as an aggregate:
/// the centroid of the fill verts (chain interior + patch interior) is displaced from
/// the ring-vert centroid **toward `v`**. A tier-3 flat-facet regression would leave the
/// fill in the ring plane and fail this.
#[test]
fn vertex_bevel_multiseg_profile_bulges_toward_the_corner_vertex() {
    // A cube corner, where all three sectors are genuinely non-planar (unlike the
    // flat 180° gap of the two-quad plane, whose gap chord is collinear with `v`).
    let mut mesh = LoopMesh::new();
    mesh_add_cube(&mut mesh);
    let corner_position = Vec3::new(1.0, -1.0, -1.0);
    let corner = vert_at(&mesh, corner_position);
    let fars: Vec<Vec3> = [
        Vec3::new(-1.0, -1.0, -1.0),
        Vec3::new(1.0, 1.0, -1.0),
        Vec3::new(1.0, -1.0, 1.0),
    ]
    .to_vec();

    let result = mesh.bevel(
        &[corner.into()],
        BevelOptions {
            offset: 0.25,
            segments: 3,
            affect: BevelAffect::Vertices,
            ..Default::default()
        },
    );

    // A ring vert lies on one of the three rays from `v` toward a far endpoint (the §4
    // slide); every other new vert is fill (chain / patch interior).
    let on_slide_ray = |p: Vec3| -> bool {
        fars.iter().any(|&far| {
            let direction = (far - corner_position).normalize();
            let along = (p - corner_position).dot(direction);
            along > 1.0e-4 && ((p - corner_position) - along * direction).length() < 1.0e-4
        })
    };
    let mut ring_sum = Vec3::ZERO;
    let mut ring_count = 0.0;
    let mut fill_sum = Vec3::ZERO;
    let mut fill_count = 0.0;
    for &vertex in &result.verts {
        let position = mesh.vertex_position(vertex);
        if on_slide_ray(position) {
            ring_sum += position;
            ring_count += 1.0;
        } else {
            fill_sum += position;
            fill_count += 1.0;
        }
    }
    assert_eq!(ring_count as usize, 3, "three §4 slide-ring boundverts");
    assert_eq!(
        fill_count as usize, 9,
        "six chain + three patch interior verts"
    );

    let ring_centroid = ring_sum / ring_count;
    let fill_centroid = fill_sum / fill_count;
    let bulge = fill_centroid - ring_centroid;
    let toward_vertex = corner_position - ring_centroid;
    assert!(
        bulge.dot(toward_vertex) > 0.0,
        "§8.11.2: the profile chains and cap bulge toward the corner vertex, not flat",
    );
}

/// BEVEL.md §5.2.4.1: the even-`segments` vertex-mode **welded apex** takes the
/// edge-linear snap blend `cd(v) + t·(cd(far) − cd(v))` on the last-visited sector's
/// boundvert edge — the predecessor of the ring's start (the last boundvert in the
/// disk-ordered ring). It is *not* left at the layer defaults the way the other
/// patch-interior verts are. On the guide's load-bearing cube-corner arm (the
/// valence-3 corner `v = (1, −1, −1)` at `offset = 0.25`, `segments = 2`, per-vertex
/// tint `0.11` on `v` and `{0.19, 0.05, 0.23}` on its neighbours toward
/// `(1,−1,1)/(1,1,−1)/(−1,−1,−1)`) the three candidate blends are `0.112525` /
/// `0.108738` / `0.111683` and the winner is fixed by the **ring's start**, which
/// §7.6.1.2 reads off the corner's *stored* attachment order. This fixture pre-creates
/// its edges, so `v`'s anchor is the `(−1,−1,−1)` spoke and the origin lands there,
/// putting the `(1,−1,1)` spoke last: `0.11 + 0.02104·(0.19 − 0.11) = 0.111683`. A cube
/// whose edges are *minted by the face creates* anchors on the `(1,1,−1)` spoke instead
/// and lands on the guide's own `0.108738` — same rule, different construction.
/// `NoInterp` layers keep `cd(v)` (the §5.2.1 step-1 mint baseline, BMESH.md §10.1.4);
/// untouched verts keep their block.
#[test]
fn vertex_bevel_multiseg_welded_apex_takes_the_edge_linear_snap_blend() {
    let mut mesh = LoopMesh::new();
    mesh_add_cube(&mut mesh);
    let tint = mesh
        .register_vertex_attribute(ElementAttributeDesc::numeric(
            AttributeKey::custom("tint"),
            AttrType::F32,
        ))
        .unwrap();
    // A second float layer forced to NoInterp: the blend must never write it, so the
    // apex keeps its minted `cd(v)` baseline rather than a blended or default value.
    let seed = mesh
        .register_vertex_attribute(
            ElementAttributeDesc::numeric(AttributeKey::custom("seed"), AttrType::F32)
                .with_interp(InterpPolicy::NoInterp),
        )
        .unwrap();

    let corner_position = Vec3::new(1.0, -1.0, -1.0);
    let corner_tint = 0.11_f32;
    // Neighbour edge → tint, matching the guide's worked cube arm.
    let neighbours = [
        (Vec3::new(1.0, -1.0, 1.0), 0.19_f32),
        (Vec3::new(1.0, 1.0, -1.0), 0.05_f32),
        (Vec3::new(-1.0, -1.0, -1.0), 0.23_f32),
    ];

    // Give every vertex a distinct tint so an untouched vert cannot alias a blended
    // one; the corner and its three neighbours carry the guide's values.
    for (index, v) in mesh.vert_keys().into_iter().enumerate() {
        mesh.vertex_attribs_mut()
            .set_f32(tint, v.as_usize(), 0.5 + index as f32);
        mesh.vertex_attribs_mut().set_f32(seed, v.as_usize(), 9.0);
    }
    let corner = vert_at(&mesh, corner_position);
    mesh.vertex_attribs_mut()
        .set_f32(tint, corner.as_usize(), corner_tint);
    mesh.vertex_attribs_mut()
        .set_f32(seed, corner.as_usize(), 7.0);
    for (pos, value) in neighbours {
        let v = vert_at(&mesh, pos);
        mesh.vertex_attribs_mut().set_f32(tint, v.as_usize(), value);
    }
    // A vertex the bevel never touches, to pin that its block survives.
    let untouched = vert_at(&mesh, Vec3::new(-1.0, 1.0, 1.0));
    let untouched_tint = mesh
        .vertex_attribs()
        .get_f32(tint, untouched.as_usize())
        .unwrap();

    let result = mesh.bevel(
        &[corner.into()],
        BevelOptions {
            offset: 0.25,
            segments: 2,
            affect: BevelAffect::Vertices,
            ..Default::default()
        },
    );

    // The welded apex is the lone new vert on the corner's body diagonal: all three
    // of its offsets from `v` are non-zero and equal (every ring / profile vert has
    // one zero offset component).
    let apex = result
        .verts
        .iter()
        .copied()
        .find(|&v| {
            let d = mesh.vertex_position(v) - corner_position;
            d.x.abs() > 1.0e-4
                && (d.x.abs() - d.y.abs()).abs() < 1.0e-4
                && (d.x.abs() - d.z.abs()).abs() < 1.0e-4
        })
        .expect("the welded apex on the body diagonal");
    let apex_position = mesh.vertex_position(apex);
    let apex_tint = mesh
        .vertex_attribs()
        .get_f32(tint, apex.as_usize())
        .unwrap();

    // Every candidate is the edge-linear blend at the apex's clamped projection `t`
    // onto that neighbour edge; `t` is the same for all three (the apex is on the
    // diagonal), so the candidates differ only in `cd(far)`.
    let candidate = |far_position: Vec3, far_tint: f32| -> f32 {
        let axis = far_position - corner_position;
        let t =
            ((apex_position - corner_position).dot(axis) / axis.length_squared()).clamp(0.0, 1.0);
        corner_tint + t * (far_tint - corner_tint)
    };
    let candidates: Vec<f32> = neighbours
        .iter()
        .map(|&(pos, value)| candidate(pos, value))
        .collect();

    // The apex is one of the three edge-linear candidates — never the layer default
    // (0.0) it used to mint from, and never a four-corner face mean.
    assert!(
        candidates.iter().any(|&c| (c - apex_tint).abs() < 1.0e-5),
        "apex tint {apex_tint} must equal one edge-linear candidate {candidates:?}",
    );
    assert!(apex_tint.abs() > 1.0e-4, "apex is not the layer default");

    // Construction-specific lock (NOT a cross-backend BMesh-parity claim). Per
    // BEVEL.md §5.2.4.1 the apex snap-edge choice is *not* isomorphism-invariant: it is
    // a deterministic function of the ring's start, which §7.6.1.2 reads off the
    // corner's stored attachment order — a construction artefact. `mesh_add_cube`
    // pre-creates every edge, so `v`'s anchor is the `(−1,−1,−1)` spoke; the fan is
    // closed and wound with the walk, so the array is unreversed and the anchor is
    // slot 0, which makes the `(1,−1,1)` spoke the ring's last boundvert and
    // `cd(far) = 0.19` the apex's donor. On a cube whose edges are minted by the face
    // creates (closing-edge-first, so the anchor is the `(1,1,−1)` spoke) the same rule
    // reverses the array and lands the guide's own `0.108738`.
    assert!(
        (apex_tint - 0.111683).abs() < 1.0e-4,
        "apex tint {apex_tint} matches the §7.6.1.2 ring origin's snap-to-{{v,4}} value \
         0.111683 on this pre-created-edge cube construction",
    );

    // §10.1.4: the NoInterp layer is never blended — the apex keeps `cd(v)`'s seed.
    let apex_seed = mesh
        .vertex_attribs()
        .get_f32(seed, apex.as_usize())
        .unwrap();
    assert_eq!(apex_seed, 7.0, "NoInterp layer keeps the minted cd(v)");

    // An untouched vertex is undisturbed.
    assert_eq!(
        mesh.vertex_attribs().get_f32(tint, untouched.as_usize()),
        Some(untouched_tint),
        "a vertex the bevel never touches keeps its block",
    );
}

/// BEVEL.md §5.2.4.2: at even `segments = 4` the vertex-mode [V-ADJ] fill mints
/// **more** interior verts than the welded apex, and every one of them is minted from
/// the beveled corner `v` (§5.2.1 step 1) and blended — never left at the layer
/// defaults the pre-fix code produced. On the valence-3 corner (`s = 3`, `h = 2`) the
/// fill has seven interior verts: three sector-private diagonal verts `(i, 1, 1)`
/// (hard-pinnable — one writing sector, so order-independent — each the edge-linear
/// snap `cd(v) + t·(cd(far) − cd(v))` onto its own boundvert edge `e_i`), three
/// `j = h` seam verts and the one welded apex (both order-dependent, blended from a
/// face `f2` / the last-visited sector's edge). This pins the mint-from-`v` baseline
/// on all seven (a `NoInterp` layer keeps `cd(v)`) and the exact edge-linear value on
/// the three private diagonal verts.
#[test]
fn vertex_bevel_seg4_chain_interior_verts_mint_from_v_and_blend() {
    let mut mesh = LoopMesh::new();
    mesh_add_cube(&mut mesh);
    let tint = mesh
        .register_vertex_attribute(ElementAttributeDesc::numeric(
            AttributeKey::custom("tint"),
            AttrType::F32,
        ))
        .unwrap();
    // A NoInterp layer: the blend must never write it, so every interior vert keeps
    // its minted `cd(v)` baseline (7.0) — the sharp discriminator against the pre-fix
    // `create_vertex` mint, which left the layer default (0.0).
    let seed = mesh
        .register_vertex_attribute(
            ElementAttributeDesc::numeric(AttributeKey::custom("seed"), AttrType::F32)
                .with_interp(InterpPolicy::NoInterp),
        )
        .unwrap();

    let corner_position = Vec3::new(1.0, -1.0, -1.0);
    let corner_tint = 0.11_f32;
    let neighbours = [
        (Vec3::new(1.0, -1.0, 1.0), 0.19_f32),
        (Vec3::new(1.0, 1.0, -1.0), 0.05_f32),
        (Vec3::new(-1.0, -1.0, -1.0), 0.23_f32),
    ];

    // Distinct tints everywhere so no untouched vert can alias a blended one; `v` and
    // its three neighbours carry the guide values, and the `seed` layer marks `v`.
    for (index, v) in mesh.vert_keys().into_iter().enumerate() {
        mesh.vertex_attribs_mut()
            .set_f32(tint, v.as_usize(), 0.5 + index as f32);
        mesh.vertex_attribs_mut().set_f32(seed, v.as_usize(), 9.0);
    }
    let corner = vert_at(&mesh, corner_position);
    mesh.vertex_attribs_mut()
        .set_f32(tint, corner.as_usize(), corner_tint);
    mesh.vertex_attribs_mut()
        .set_f32(seed, corner.as_usize(), 7.0);
    for (pos, value) in neighbours {
        let v = vert_at(&mesh, pos);
        mesh.vertex_attribs_mut().set_f32(tint, v.as_usize(), value);
    }

    let result = mesh.bevel(
        &[corner.into()],
        BevelOptions {
            offset: 0.25,
            segments: 4,
            affect: BevelAffect::Vertices,
            ..Default::default()
        },
    );
    mesh.validate_topology();

    // The cap's own interior verts are the new verts strictly inside the corner octant
    // — off all three incident face planes (`x = 1`, `y = -1`, `z = -1`). Boundverts
    // sit on original edges and profile-chain verts lie in an incident face plane, so
    // both keep a coordinate pinned to the plane value and are excluded.
    const EPS: f32 = 1.0e-3;
    let interior: Vec<VertexKey> = result
        .verts
        .iter()
        .copied()
        .filter(|&v| {
            let p = mesh.vertex_position(v);
            p.x < 1.0 - EPS && p.y > -1.0 + EPS && p.z > -1.0 + EPS
        })
        .collect();
    assert_eq!(
        interior.len(),
        7,
        "§5.2.4.2: the valence-3 seg=4 cap has 7 interior verts (3 private + 3 seam + apex)",
    );

    // Every interior vert is minted from `v`: its NoInterp `seed` keeps `cd(v)`'s 7.0
    // rather than the layer default. This is the core of the fix.
    for &v in &interior {
        let s = mesh.vertex_attribs().get_f32(seed, v.as_usize()).unwrap();
        assert_eq!(
            s, 7.0,
            "interior vert {v:?} must mint from v (NoInterp seed keeps cd(v)), not the default",
        );
        let t = mesh.vertex_attribs().get_f32(tint, v.as_usize()).unwrap();
        assert!(
            t.abs() > 0.01,
            "interior vert {v:?} tint {t} must be a blend, never the layer default 0.0",
        );
    }

    // The three sector-private diagonal verts `(i, 1, 1)` are the interior verts that
    // are (a) NOT on the body diagonal (that is the apex) and (b) equal to the
    // edge-linear snap onto one of `v`'s three incident edges, at their own clamped
    // projection `t`. They are isomorphism-invariant, so the value is pinned exactly.
    let edge_linear_candidate = |vert_position: Vec3| -> Vec<f32> {
        neighbours
            .iter()
            .map(|&(far_position, far_tint)| {
                let axis = far_position - corner_position;
                let t = ((vert_position - corner_position).dot(axis) / axis.length_squared())
                    .clamp(0.0, 1.0);
                corner_tint + t * (far_tint - corner_tint)
            })
            .collect()
    };
    let is_body_diagonal = |p: Vec3| {
        let d = p - corner_position;
        (d.x.abs() - d.y.abs()).abs() < 1.0e-4 && (d.x.abs() - d.z.abs()).abs() < 1.0e-4
    };

    let mut private_diagonal = 0;
    for &v in &interior {
        let p = mesh.vertex_position(v);
        if is_body_diagonal(p) {
            continue; // the welded apex, order-dependent — not pinned here
        }
        let t = mesh.vertex_attribs().get_f32(tint, v.as_usize()).unwrap();
        if edge_linear_candidate(p)
            .iter()
            .any(|&c| (c - t).abs() < 1.0e-5)
        {
            private_diagonal += 1;
        }
    }
    assert_eq!(
        private_diagonal, 3,
        "§5.2.4.2: exactly the three sector-private diagonal verts take the edge-linear \
         snap onto their own boundvert edge (the seam verts blend from a face, the apex \
         is on the body diagonal)",
    );
}

/// BEVEL.md §5.2.4.3 (no-seam branch): at **odd** `segments` the [V-ADJ] fill closes on a
/// central `s`-gon whose `s` corner verts `(i, h, h)` are minted from the beveled corner
/// `v` (§5.2.1 step-1 baseline, so a `NoInterp` layer keeps `cd(v)`) and then blended. On a
/// **not-seam-carrying** corner — the fully-interior cube corner — each centre corner takes
/// the plain §10.1 interior mean-value blend of boundvert `i`'s **own** representative face
/// (no snap). At `segments = 3` (`h = 1`) the valence-3 cap's only interior verts *are*
/// those three centre corners, so this pins them all: the `NoInterp` layer keeps `cd(v)`,
/// and the interpolable layer is a genuine blend — no longer the bare `cd(v)` baseline the
/// pre-blend code left, but a convex combination inside the original-tint hull.
#[test]
fn vertex_bevel_seg3_centre_ngon_corners_blend_from_own_rep() {
    let mut mesh = LoopMesh::new();
    mesh_add_cube(&mut mesh);
    // An interpolable layer: the centre corners must now carry the own-rep interior blend,
    // never the bare `cd(v)` baseline nor the 0.0 layer default.
    let tint = mesh
        .register_vertex_attribute(ElementAttributeDesc::numeric(
            AttributeKey::custom("tint"),
            AttrType::F32,
        ))
        .unwrap();
    // A NoInterp layer: it can only ever carry a value that was copied, never blended, so it
    // is the sharp discriminator that the centre corners still mint from `v`.
    let seed = mesh
        .register_vertex_attribute(
            ElementAttributeDesc::numeric(AttributeKey::custom("seed"), AttrType::F32)
                .with_interp(InterpPolicy::NoInterp),
        )
        .unwrap();

    let corner_position = Vec3::new(1.0, -1.0, -1.0);
    let corner_tint = 0.11_f32;
    let neighbours = [
        (Vec3::new(1.0, -1.0, 1.0), 0.19_f32),
        (Vec3::new(1.0, 1.0, -1.0), 0.05_f32),
        (Vec3::new(-1.0, -1.0, -1.0), 0.23_f32),
    ];

    for (index, v) in mesh.vert_keys().into_iter().enumerate() {
        mesh.vertex_attribs_mut()
            .set_f32(tint, v.as_usize(), 0.5 + index as f32);
        mesh.vertex_attribs_mut().set_f32(seed, v.as_usize(), 9.0);
    }
    let corner = vert_at(&mesh, corner_position);
    mesh.vertex_attribs_mut()
        .set_f32(tint, corner.as_usize(), corner_tint);
    mesh.vertex_attribs_mut()
        .set_f32(seed, corner.as_usize(), 7.0);
    for (pos, value) in neighbours {
        let v = vert_at(&mesh, pos);
        mesh.vertex_attribs_mut().set_f32(tint, v.as_usize(), value);
    }

    // The original-tint hull the convex own-rep blend must land inside.
    let (tint_min, tint_max) =
        mesh.vert_keys()
            .into_iter()
            .fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), v| {
                let t = mesh.vertex_attribs().get_f32(tint, v.as_usize()).unwrap();
                (lo.min(t), hi.max(t))
            });

    let result = mesh.bevel(
        &[corner.into()],
        BevelOptions {
            offset: 0.25,
            segments: 3,
            affect: BevelAffect::Vertices,
            ..Default::default()
        },
    );
    mesh.validate_topology();

    // Interior verts: strictly inside the corner octant, off all three incident face planes.
    // At `segments = 3` these are exactly the three centre n-gon corners `(i, 1, 1)`.
    const EPS: f32 = 1.0e-3;
    let interior: Vec<VertexKey> = result
        .verts
        .iter()
        .copied()
        .filter(|&v| {
            let p = mesh.vertex_position(v);
            p.x < 1.0 - EPS && p.y > -1.0 + EPS && p.z > -1.0 + EPS
        })
        .collect();
    assert_eq!(
        interior.len(),
        3,
        "the odd seg=3 valence-3 cap's interior verts are exactly the 3 centre n-gon corners",
    );

    for &v in &interior {
        // §5.2.1 step 1: the NoInterp `seed` keeps `cd(v)`'s 7.0 — the blend never writes it.
        let s = mesh.vertex_attribs().get_f32(seed, v.as_usize()).unwrap();
        assert_eq!(
            s, 7.0,
            "centre corner {v:?} must mint from v (NoInterp seed keeps cd(v))",
        );
        // §5.2.4.3: the interpolable layer is now the own-rep interior blend — a real blend,
        // never the bare `cd(v)` baseline the pre-blend code left, and inside the tint hull.
        let t = mesh.vertex_attribs().get_f32(tint, v.as_usize()).unwrap();
        assert!(
            (t - corner_tint).abs() > 1.0e-4,
            "centre corner {v:?} tint {t} must be the own-rep blend, not the cd(v) baseline \
             {corner_tint}",
        );
        assert!(
            t >= tint_min - 1.0e-4 && t <= tint_max + 1.0e-4,
            "centre corner {v:?} tint {t} must lie inside the original-tint hull \
             [{tint_min}, {tint_max}]",
        );
    }
}

/// BEVEL.md §8.5.5.5(f.4b): vertex-mode square-in (`profile = 0`) does **not** weld —
/// it builds the ordinary general adjacency grid, square-in shaped, so the **odd**
/// `segments = 3` cube corner has the full interior ring of quads closed by one central
/// triangle (the §8.5.5.4 odd parity), identical topology to the circular grid. Total
/// `8/12/6 → 19/30/13`, a closed genus-0 solid: 3 rebuilt 7-gons + 3 untouched cube
/// quads + 6 cap quads + 1 central triangle. The square-in shape shows only in
/// *positions*: the negative fullness (§8.5.5.5(f.3) F3 at `profile = 0`) pushes the
/// centre **inward**, away from the corner vertex — the concave notch.
#[test]
fn vertex_bevel_multiseg_square_in_odd_segments_central_triangle_cube_corner() {
    let mut mesh = LoopMesh::new();
    mesh_add_cube(&mut mesh);
    let corner_position = Vec3::new(1.0, -1.0, -1.0);
    let corner = vert_at(&mesh, corner_position);
    assert_eq!(counts(&mesh), (8, 12, 6));

    let result = mesh.bevel(
        &[corner.into()],
        BevelOptions {
            offset: 0.25,
            segments: 3,
            profile: 0.0,
            affect: BevelAffect::Vertices,
            ..Default::default()
        },
    );
    mesh.validate_topology();

    let (vertices, edges, faces) = counts(&mesh);
    assert_eq!(
        (vertices, edges, faces),
        (19, 30, 13),
        "§8.5.5.5(f.4b) odd grid"
    );
    assert_eq!(
        vertices as i64 - edges as i64 + faces as i64,
        2,
        "square-in odd corner bevel keeps the cube a closed genus-0 solid",
    );

    // The general grid keeps the full interior ring of quads: 6 cap quads + exactly one
    // central triangle (the §8.5.5.4 odd parity), not the weld's lone triangle.
    assert_eq!(result.faces.len(), 7, "6 ring quads + 1 central triangle");
    let central: Vec<FaceKey> = result
        .faces
        .iter()
        .copied()
        .filter(|&face| mesh.face_vert_keys(face).len() == 3)
        .collect();
    assert_eq!(
        central.len(),
        1,
        "one central triangle closes the innermost ring"
    );

    // 3 boundverts + 6 chain interior (2 per sector) + 3 central-ring verts.
    assert_eq!(
        result.verts.len(),
        12,
        "3 boundverts + 6 chain mids + 3 central verts"
    );

    assert_eq!(
        face_size_multiset(&mesh),
        vec![3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 7, 7, 7],
        "1 central triangle + 9 quads (3 cube + 6 cap) + 3 caved 7-gons",
    );

    // §8.5.5.5(f.4b): the negative square-in fullness seeds the centre inward. Every
    // central-triangle vertex sits on the *far* side of the ring centroid from the
    // corner — the concave notch, the opposite of the circular cap's outward bulge.
    let ring_centroid = {
        let boundverts: Vec<VertexKey> = result
            .verts
            .iter()
            .copied()
            .filter(|&v| {
                result
                    .faces
                    .iter()
                    .filter(|&&f| mesh.face_vert_keys(f).contains(&v))
                    .count()
                    == 1
            })
            .collect();
        assert_eq!(boundverts.len(), 3, "three slide-ring boundverts");
        boundverts
            .iter()
            .fold(Vec3::ZERO, |acc, &v| acc + mesh.vertex_position(v))
            / 3.0
    };
    let toward_corner = corner_position - ring_centroid;
    for &tip in &mesh.face_vert_keys(central[0]) {
        let offset = mesh.vertex_position(tip) - ring_centroid;
        assert!(
            offset.dot(toward_corner) < 0.0,
            "the central triangle is notched inward (negative fullness), not bulged toward v",
        );
    }
}

/// BEVEL.md §8.5.5.5(f.4b): vertex-mode square-in (`profile = 0`) does **not** weld — the
/// **even** `segments = 2` cube corner takes the general grid with a single welded apex
/// (the §8.5.5.4 even parity) plus the full ring of cap quads, identical topology to the
/// circular grid. Total `8/12/6 → 14/21/9`, a closed solid: 3 rebuilt hexagons + 3
/// untouched cube quads + 3 cap quads. Square-in enters only through positions: the
/// negative fullness pushes the apex **inward**, away from the corner (the concave notch).
#[test]
fn vertex_bevel_multiseg_square_in_even_segments_welded_apex_cube_corner() {
    let mut mesh = LoopMesh::new();
    mesh_add_cube(&mut mesh);
    let corner_position = Vec3::new(1.0, -1.0, -1.0);
    let corner = vert_at(&mesh, corner_position);
    assert_eq!(counts(&mesh), (8, 12, 6));

    let result = mesh.bevel(
        &[corner.into()],
        BevelOptions {
            offset: 0.25,
            segments: 2,
            profile: 0.0,
            affect: BevelAffect::Vertices,
            ..Default::default()
        },
    );
    mesh.validate_topology();

    let (vertices, edges, faces) = counts(&mesh);
    assert_eq!(
        (vertices, edges, faces),
        (14, 21, 9),
        "§8.5.5.5(f.4b) even grid"
    );
    assert_eq!(
        vertices as i64 - edges as i64 + faces as i64,
        2,
        "square-in even corner bevel keeps the cube a closed genus-0 solid",
    );

    // Even parity welds one apex and mints no central face, but the ring of cap quads is
    // present (that is exactly what the weld would delete): three cap quads.
    assert_eq!(
        result.faces.len(),
        3,
        "three cap quads, no central polygon (welded apex)"
    );

    // 3 boundverts + 3 chain interior (one per sector) + 1 welded apex.
    assert_eq!(result.verts.len(), 7, "3 ring + 3 chain mid + 1 apex");

    assert_eq!(
        face_size_multiset(&mesh),
        vec![4, 4, 4, 4, 4, 4, 6, 6, 6],
        "3 cap quads + 3 untouched cube quads + 3 caved hexagons",
    );

    // Classify cap verts by how many cap quads they touch: boundverts (1), chain mids
    // (2), the welded apex (3).
    let cap_membership = |vertex: VertexKey| -> usize {
        result
            .faces
            .iter()
            .filter(|&&face| mesh.face_vert_keys(face).contains(&vertex))
            .count()
    };
    let ring_verts: Vec<VertexKey> = result
        .verts
        .iter()
        .copied()
        .filter(|&vertex| cap_membership(vertex) == 1)
        .collect();
    assert_eq!(ring_verts.len(), 3, "three slide-ring boundverts");
    let ring_centroid = ring_verts
        .iter()
        .fold(Vec3::ZERO, |acc, &v| acc + mesh.vertex_position(v))
        / 3.0;
    let toward_corner = corner_position - ring_centroid;

    // §8.5.5.5(f.4b): the negative square-in fullness pushes the apex inward, away from
    // the corner — the opposite of the circular even cap, whose apex bulges toward `v`.
    let apex = result
        .verts
        .iter()
        .copied()
        .find(|&vertex| cap_membership(vertex) == 3)
        .expect("a single welded apex shared by all three cap quads");
    let apex_offset = mesh.vertex_position(apex) - ring_centroid;
    assert!(
        apex_offset.dot(toward_corner) < 0.0,
        "the welded apex is notched inward (negative fullness), not bulged toward v",
    );
}

/// Bevel the `(1,-1,-1)` cube corner in vertex mode at `profile` / even `segments` and
/// return the signed distance the even-parity welded apex (§8.5.5.5(e)) sits from the
/// slide-ring centroid *toward* the original corner `v` (positive = bulged toward `v`).
/// Panics unless the cap is the welded-apex grid, so pass even `segments` only.
fn cube_corner_welded_apex_bulge(profile: f32, segments: u32) -> f32 {
    let mut mesh = LoopMesh::new();
    mesh_add_cube(&mut mesh);
    let corner_position = Vec3::new(1.0, -1.0, -1.0);
    let corner = vert_at(&mesh, corner_position);
    let result = mesh.bevel(
        &[corner.into()],
        BevelOptions {
            offset: 0.25,
            segments,
            profile,
            affect: BevelAffect::Vertices,
            ..Default::default()
        },
    );
    mesh.validate_topology();
    let cap_membership = |vertex: VertexKey| -> usize {
        result
            .faces
            .iter()
            .filter(|&&face| mesh.face_vert_keys(face).contains(&vertex))
            .count()
    };
    let ring_verts: Vec<VertexKey> = result
        .verts
        .iter()
        .copied()
        .filter(|&vertex| cap_membership(vertex) == 1)
        .collect();
    assert_eq!(ring_verts.len(), 3, "three slide-ring boundverts");
    let ring_centroid = ring_verts
        .iter()
        .fold(Vec3::ZERO, |acc, &v| acc + mesh.vertex_position(v))
        / 3.0;
    let toward_corner = (corner_position - ring_centroid).normalize();
    let apex = result
        .verts
        .iter()
        .copied()
        .find(|&vertex| cap_membership(vertex) == 3)
        .expect("a single welded apex shared by all three cap quads");
    (mesh.vertex_position(apex) - ring_centroid).dot(toward_corner)
}

/// BEVEL.md §8.5.5.5(f.1) step 4 / (f.2)/(f.3): a genuinely curved exponent
/// (`profile = 0.7`, `r ≈ 3.887`) is no longer declined in vertex mode — it routes to
/// the *same* general fullness grid as the circular default, differing only in the
/// boundary-ring sampler (§8.6) and the centre fullness (§8.5.5.5(f.3)). The **even**
/// `segments = 2` cube corner therefore builds the full coarse net: 3 slide-ring
/// boundverts, 3 chain mids, 1 welded apex, `8/12/6 → 14/21/9`, still a closed solid.
#[test]
fn vertex_bevel_multiseg_curved_profile_even_cube_corner_builds_general_grid() {
    let mut mesh = LoopMesh::new();
    mesh_add_cube(&mut mesh);
    let corner_position = Vec3::new(1.0, -1.0, -1.0);
    let corner = vert_at(&mesh, corner_position);
    assert_eq!(counts(&mesh), (8, 12, 6));

    let result = mesh.bevel(
        &[corner.into()],
        BevelOptions {
            offset: 0.25,
            segments: 2,
            profile: 0.7,
            affect: BevelAffect::Vertices,
            ..Default::default()
        },
    );
    mesh.validate_topology();

    let (vertices, edges, faces) = counts(&mesh);
    assert_eq!(
        (vertices, edges, faces),
        (14, 21, 9),
        "curved even grid builds"
    );
    assert_eq!(
        vertices as i64 - edges as i64 + faces as i64,
        2,
        "curved corner bevel keeps the cube a closed genus-0 solid",
    );
    assert_eq!(
        result.faces.len(),
        3,
        "three cap quads, welded apex, no central polygon"
    );
    assert_eq!(result.verts.len(), 7, "3 ring + 3 chain mid + 1 apex");
    assert_eq!(
        face_size_multiset(&mesh),
        vec![4, 4, 4, 4, 4, 4, 6, 6, 6],
        "three cap quads + three untouched cube quads + three grown hexagons",
    );

    // The fullness-seeded apex bulges toward the corner (fullness ≈1.089 at profile 0.7,
    // §8.5.5.5(f.3) F3), not onto the flat ring plane.
    let cap_membership = |vertex: VertexKey| -> usize {
        result
            .faces
            .iter()
            .filter(|&&face| mesh.face_vert_keys(face).contains(&vertex))
            .count()
    };
    let ring_centroid = {
        let ring: Vec<VertexKey> = result
            .verts
            .iter()
            .copied()
            .filter(|&v| cap_membership(v) == 1)
            .collect();
        assert_eq!(ring.len(), 3, "three slide-ring boundverts");
        ring.iter()
            .fold(Vec3::ZERO, |acc, &v| acc + mesh.vertex_position(v))
            / 3.0
    };
    let apex = result
        .verts
        .iter()
        .copied()
        .find(|&v| cap_membership(v) == 3)
        .expect("a single welded apex shared by all three cap quads");
    assert!(
        (mesh.vertex_position(apex) - ring_centroid).dot(corner_position - ring_centroid) > 0.0,
        "curved welded apex bulges toward the corner (fullness-seeded, not flat)",
    );
}

/// BEVEL.md §8.5.5.5(f.3): the general grid's centre is seeded by the profile-dependent
/// `fullness` scalar, so a curved corner's apex sits at a *different* position from the
/// circular default. F3 at `profile = 0.7` even gives ≈1.089 versus the circular table's
/// 0.559 at `segments = 2`, so the curved apex bulges strictly further toward `v`.
#[test]
fn vertex_bevel_multiseg_curved_apex_uses_profile_dependent_fullness_distinct_from_circular() {
    let circular = cube_corner_welded_apex_bulge(0.5, 2);
    let curved = cube_corner_welded_apex_bulge(0.7, 2);
    assert!(circular > 0.0, "circular apex bulges toward v");
    assert!(curved > 0.0, "curved apex bulges toward v");
    assert!(
        curved > circular + 1.0e-3,
        "curved profile 0.7 (fullness ≈1.089) bulges further than circular (0.559): \
         {curved} vs {circular}",
    );
    assert!(
        (adj_fullness(0.7, 2) - adj_fullness(0.5, 2)).abs() > 1.0e-2,
        "the centre seed reads the profile-dependent fullness, not the circular table",
    );
}

/// BEVEL.md §8.5.5.5(f.1)/(f.3) + (e): a curved exponent at **odd** `segments = 3` builds
/// the full interior ring of quads closed by one central triangle (the §8.5.5.4 odd
/// parity), `8/12/6 → 19/30/13`. The positive fullness (`≈1.049` at profile 0.7, F3 odd)
/// bulges the central triangle toward the corner — the convex cap, opposite the
/// square-in notch.
#[test]
fn vertex_bevel_multiseg_curved_profile_odd_cube_corner_central_triangle() {
    let mut mesh = LoopMesh::new();
    mesh_add_cube(&mut mesh);
    let corner_position = Vec3::new(1.0, -1.0, -1.0);
    let corner = vert_at(&mesh, corner_position);
    assert_eq!(counts(&mesh), (8, 12, 6));

    let result = mesh.bevel(
        &[corner.into()],
        BevelOptions {
            offset: 0.25,
            segments: 3,
            profile: 0.7,
            affect: BevelAffect::Vertices,
            ..Default::default()
        },
    );
    mesh.validate_topology();

    assert_eq!(counts(&mesh), (19, 30, 13), "curved odd grid builds");
    assert_eq!(result.faces.len(), 7, "6 ring quads + 1 central triangle");
    let central: Vec<FaceKey> = result
        .faces
        .iter()
        .copied()
        .filter(|&face| mesh.face_vert_keys(face).len() == 3)
        .collect();
    assert_eq!(
        central.len(),
        1,
        "one central triangle closes the innermost ring"
    );

    let ring_centroid = {
        let boundverts: Vec<VertexKey> = result
            .verts
            .iter()
            .copied()
            .filter(|&v| {
                result
                    .faces
                    .iter()
                    .filter(|&&f| mesh.face_vert_keys(f).contains(&v))
                    .count()
                    == 1
            })
            .collect();
        assert_eq!(boundverts.len(), 3, "three slide-ring boundverts");
        boundverts
            .iter()
            .fold(Vec3::ZERO, |acc, &v| acc + mesh.vertex_position(v))
            / 3.0
    };
    let toward_corner = corner_position - ring_centroid;
    for &tip in &mesh.face_vert_keys(central[0]) {
        assert!(
            (mesh.vertex_position(tip) - ring_centroid).dot(toward_corner) > 0.0,
            "the curved central triangle bulges toward v (positive fullness), not inward",
        );
    }
}

/// BEVEL.md §8.5.5.5(f.1): the **square** exponent (`profile = 1.0`, `r = ∞`) in vertex
/// mode never reaches the edge-only parallel-offset grid (step 1) — it falls to the same
/// general fullness grid as every other exponent. The even `segments = 2` cube corner
/// builds `8/12/6 → 14/21/9` with a welded apex, and its fullness (F3 even ≈1.824,
/// §8.5.5.5(f.3)) overshoots the circular seed, so the square apex bulges strictly
/// further toward `v` than the circular one.
#[test]
fn vertex_bevel_multiseg_square_profile_even_cube_corner_builds_and_overshoots() {
    let mut mesh = LoopMesh::new();
    mesh_add_cube(&mut mesh);
    let corner_position = Vec3::new(1.0, -1.0, -1.0);
    let corner = vert_at(&mesh, corner_position);
    assert_eq!(counts(&mesh), (8, 12, 6));

    let result = mesh.bevel(
        &[corner.into()],
        BevelOptions {
            offset: 0.25,
            segments: 2,
            profile: 1.0,
            affect: BevelAffect::Vertices,
            ..Default::default()
        },
    );
    mesh.validate_topology();

    assert_eq!(counts(&mesh), (14, 21, 9), "square even grid builds");
    assert_eq!(result.faces.len(), 3, "three cap quads, welded apex");
    assert_eq!(result.verts.len(), 7, "3 ring + 3 chain mid + 1 apex");
    assert_eq!(
        profile_to_exponent(1.0),
        PROFILE_SQUARE_EXPONENT,
        "profile 1.0 snaps to the square sentinel exponent",
    );

    let square = cube_corner_welded_apex_bulge(1.0, 2);
    let circular = cube_corner_welded_apex_bulge(0.5, 2);
    assert!(square > 0.0, "square apex bulges toward v");
    assert!(
        square > circular + 1.0e-3,
        "square profile (fullness ≈1.824) overshoots the circular seed (0.559): \
         {square} vs {circular}",
    );
}

/// BEVEL.md §8.5.5.5(f.3) law **F2**: the **line / flat chamfer** (`profile = 0.25`,
/// `r == 1.0`) in vertex mode is no longer declined. It is not a distinct construction —
/// it is the *same* general fullness grid with `fullness = 0`, so the even `segments = 2`
/// cube corner builds the identical coarse-net topology as the curved / circular grids
/// (`8/12/6 → 14/21/9`, three cap quads + welded apex), differing only in that its apex
/// is pinned to the boundvert-ring **plane** at the centroid (no bulge toward `v`).
#[test]
fn vertex_bevel_multiseg_flat_line_builds_planar_cap_distinct_from_curved() {
    assert_eq!(
        profile_to_exponent(0.25),
        1.0,
        "profile 0.25 snaps to the line/flat-chamfer exponent r = 1",
    );

    let mut mesh = LoopMesh::new();
    mesh_add_cube(&mut mesh);
    let corner = vert_at(&mesh, Vec3::new(1.0, -1.0, -1.0));
    assert_eq!(counts(&mesh), (8, 12, 6));

    let result = mesh.bevel(
        &[corner.into()],
        BevelOptions {
            offset: 0.25,
            segments: 2,
            profile: 0.25,
            affect: BevelAffect::Vertices,
            ..Default::default()
        },
    );
    mesh.validate_topology();

    // Same topology class as the curved / circular even grid, and as the edge-mode flat
    // line (`emit_general_adj_fill` with `flat = true`): `s` cap quads fanning a single
    // welded apex, one closed genus-0 solid.
    assert_eq!(
        counts(&mesh),
        (14, 21, 9),
        "flat line builds the general even grid"
    );
    assert_eq!(result.faces.len(), 3, "three cap quads + welded apex");
    assert_eq!(result.verts.len(), 7, "3 ring + 3 chain mid + 1 apex");
    assert_eq!(
        face_size_multiset(&mesh),
        vec![4, 4, 4, 4, 4, 4, 6, 6, 6],
        "flat line's cap class matches the curved even grid",
    );

    // F2: `fullness = 0` planar cap — the apex sits ON the boundvert-ring plane at the
    // centroid, whereas every curved / square / circular profile bulges it toward `v`.
    let flat = cube_corner_welded_apex_bulge(0.25, 2);
    let curved = cube_corner_welded_apex_bulge(0.7, 2);
    let circular = cube_corner_welded_apex_bulge(0.5, 2);
    assert!(
        flat.abs() < 1.0e-4,
        "the flat-line apex is planar (fullness 0, on the ring centroid): {flat}",
    );
    assert!(
        curved > flat + 1.0e-3 && circular > flat + 1.0e-3,
        "curved ({curved}) and circular ({circular}) bulge strictly past the flat cap ({flat})",
    );
    assert_eq!(adj_fullness(0.25, 2), 0.0, "F2: r == 1.0 ⇒ fullness = 0");
}

/// BEVEL.md §8.5.5.5(e) / §8.5.5.5(f.3) law **F2**: the straight-chord (`r = 1`) cap at
/// an **odd** segment count does not weld to an apex — it closes on a central `s`-gon
/// whose ring is a *determined* quantity. The flat cap is `fullness = 0`, so the whole
/// cap stays in the boundvert-ring plane; but the in-plane spread of the central ring is
/// **not** a free choice — it descends through the same subdivision-limit relaxation the
/// curved caps use, seeded at the ring-plane centroid. This pins the valence-3 cube
/// corner at `segments = 3`: the central triangle sits at the relaxed-net radius
/// (`0.043943`), strictly outside the ad-hoc `gamma = 1/(half + 2)` spread the old flat
/// bilinear blend produced (`0.039284`).
#[test]
fn vertex_bevel_multiseg_flat_line_odd_segments_valence3_interior_descends_from_relaxed_net() {
    let mut mesh = LoopMesh::new();
    mesh_add_cube(&mut mesh);
    let corner = vert_at(&mesh, Vec3::new(1.0, -1.0, -1.0));
    let result = mesh.bevel(
        &[corner.into()],
        BevelOptions {
            offset: 0.25,
            segments: 3,
            profile: 0.25,
            affect: BevelAffect::Vertices,
            ..Default::default()
        },
    );
    mesh.validate_topology();

    let (radius, _centroid, max_plane) = flat_cap_probe(&mesh, &result, 3);
    assert!(
        max_plane < 1.0e-4,
        "the flat (fullness 0) odd cap is planar — every cap vert on the ring plane: {max_plane}",
    );
    assert!(
        (radius - 0.043_943).abs() < 5.0e-4,
        "the central triangle descends to the determined relaxed-net radius 0.043943: {radius}",
    );
    assert!(
        radius > 0.041_5,
        "the descended ring is strictly outside the ad-hoc 1/(half+2) spread (0.039284): {radius}",
    );
}

/// BEVEL.md §8.5.5.5(e) at valence 4: the same determined odd-centre descent on a
/// symmetric square-pyramid apex. The flat cap stays planar and the central quad's ring
/// lands at the relaxed-net radius (`0.058926`), not the ad-hoc gamma spread.
#[test]
fn vertex_bevel_multiseg_flat_line_odd_segments_valence4_interior_descends_from_relaxed_net() {
    let (mut mesh, apex) = closed_pyramid(4);
    let result = mesh.bevel(
        &[apex.into()],
        BevelOptions {
            offset: 0.25,
            segments: 3,
            profile: 0.25,
            affect: BevelAffect::Vertices,
            ..Default::default()
        },
    );
    mesh.validate_topology();

    let (radius, _centroid, max_plane) = flat_cap_probe(&mesh, &result, 4);
    assert!(
        max_plane < 1.0e-4,
        "the flat valence-4 odd cap is planar: {max_plane}",
    );
    assert!(
        (radius - 0.058_926).abs() < 5.0e-4,
        "the central quad descends to the determined relaxed-net radius 0.058926: {radius}",
    );
}

/// BEVEL.md §8.5.5.5(b): curved and flat caps stay **distinct** — the flat (`fullness =
/// 0`) odd cap is planar, whereas the curved profile (`profile = 0.7`) bulges its cap
/// strictly out of the boundvert-ring plane. Guards against the odd-centre descent
/// unification collapsing the flat cap onto the curved one.
#[test]
fn vertex_bevel_multiseg_flat_vs_curved_odd_cap_stays_distinct() {
    let build = |profile: f32| {
        let mut mesh = LoopMesh::new();
        mesh_add_cube(&mut mesh);
        let corner = vert_at(&mesh, Vec3::new(1.0, -1.0, -1.0));
        let result = mesh.bevel(
            &[corner.into()],
            BevelOptions {
                offset: 0.25,
                segments: 3,
                profile,
                affect: BevelAffect::Vertices,
                ..Default::default()
            },
        );
        cap_max_plane_dev(&mesh, &result)
    };
    let flat = build(0.25);
    let curved = build(0.7);
    assert!(flat < 1.0e-4, "flat odd cap is planar: {flat}");
    assert!(
        curved > 0.02,
        "curved odd cap bulges out of the ring plane, distinct from the flat cap: {curved}",
    );
}

/// BEVEL.md §8.5.5.5(e) / §8.5.5.5(f.3) law **F2**: the straight-chord (`r = 1`) cap at
/// an **even** `segments ≥ 4` count welds a single apex and mints no central face (even
/// parity), *and* — unlike `segments = 2`, which has no interior ring — it now carries a
/// multi-ring interior grid whose verts are a **determined** quantity. That interior must
/// descend through the same subdivision-limit relaxation the odd and curved caps use
/// (`fullness = 0` seeded at the ring-plane centroid), **not** the ad-hoc planar bilinear
/// blend the `segments = 2` fast path uses. This pins the valence-3 cube corner at
/// `segments = 4`: the three apex-ring interior (sector-diagonal) verts descend to the
/// relaxed-net radius, measurably **off** the bilinear-blend prediction `(R_b + R_m)/4`
/// (the blend the old even-`≥4` path emitted), while the whole cap stays planar.
#[test]
fn vertex_bevel_multiseg_flat_line_even_seg4_valence3_interior_descends_from_relaxed_net() {
    let mut mesh = LoopMesh::new();
    mesh_add_cube(&mut mesh);
    let corner = vert_at(&mesh, Vec3::new(1.0, -1.0, -1.0));
    let result = mesh.bevel(
        &[corner.into()],
        BevelOptions {
            offset: 0.25,
            segments: 4,
            profile: 0.25,
            affect: BevelAffect::Vertices,
            ..Default::default()
        },
    );
    mesh.validate_topology();

    let cap_membership = |vertex: VertexKey| -> usize {
        result
            .faces
            .iter()
            .filter(|&&face| mesh.face_vert_keys(face).contains(&vertex))
            .count()
    };

    // Even parity: one welded apex (shared by all `s` sectors), no central polygon, every
    // cap face a quad. Same *topology* the old bilinear path emitted — the divergence is
    // purely positional, so it must be probed positionally below.
    assert!(
        result
            .faces
            .iter()
            .all(|&f| mesh.face_vert_keys(f).len() == 4),
        "even seg-4 flat cap is all quads (welded apex, no central polygon)",
    );
    let apex = result
        .verts
        .iter()
        .copied()
        .find(|&v| cap_membership(v) == 3)
        .expect("a single welded apex shared by all three cap quads");
    let center = mesh.vertex_position(apex);

    // The straight-chord (`r = 1`) rim is a set of `s` chords, so a chain's boundverts sit
    // at the **max** rim radius and its chord midpoints at the **min** — an intrinsic way
    // to read `R_b` and `R_m` off the cap without reconstructing sector adjacency.
    let rim: Vec<Vec3> = result
        .verts
        .iter()
        .copied()
        .filter(|&v| cap_membership(v) == 2)
        .map(|v| mesh.vertex_position(v))
        .collect();
    let radius = |p: Vec3| (p - center).length();
    let r_b = rim.iter().copied().map(radius).fold(0.0_f32, f32::max);
    let r_m = rim
        .iter()
        .copied()
        .map(radius)
        .fold(f32::INFINITY, f32::min);
    let bilinear_diagonal = (r_b + r_m) / 4.0;

    // Boundvert directions (the max-radius rim verts) — the apex-ring diagonal verts point
    // along them; the seam verts point along the chord-mid directions instead.
    let boundvert_dirs: Vec<Vec3> = rim
        .iter()
        .copied()
        .filter(|&p| radius(p) > 0.9 * r_b)
        .map(|p| (p - center).normalize())
        .collect();

    let mut diagonal_radii: Vec<f32> = Vec::new();
    let mut max_plane = 0.0_f32;
    let (_bv, _c, normal) = cap_ring_plane(&mesh, &result);
    for &v in &result.verts {
        let p = mesh.vertex_position(v);
        max_plane = max_plane.max((p - center).dot(normal).abs());
        if cap_membership(v) == 4 {
            let dir = (p - center).normalize();
            let align = boundvert_dirs
                .iter()
                .map(|&b| dir.dot(b))
                .fold(f32::NEG_INFINITY, f32::max);
            if align > 0.85 {
                diagonal_radii.push(radius(p));
            }
        }
    }
    assert_eq!(
        diagonal_radii.len(),
        3,
        "three apex-ring diagonal verts (one per sector)"
    );
    let diagonal = diagonal_radii.iter().sum::<f32>() / 3.0;

    assert!(
        max_plane < 1.0e-4,
        "the flat (fullness 0) even cap is planar — every cap vert on the ring plane: {max_plane}",
    );
    assert!(
        (diagonal - 0.076_547).abs() < 5.0e-4,
        "the apex-ring diagonal descends to the determined relaxed-net radius 0.076547: {diagonal}",
    );
    // The descent lands measurably off the ad-hoc bilinear blend (`(R_b + R_m)/4`) the old
    // even-`≥4` path emitted — the whole point of routing this parity through the relaxed
    // net. (Descended radius 0.0765 vs bilinear 0.0593: ~0.017 wu, ~1700 position quanta.)
    assert!(
        (diagonal - bilinear_diagonal).abs() > 1.0e-2,
        "the descended diagonal ({diagonal}) is distinct from the bilinear blend ({bilinear_diagonal})",
    );
}

// The `sectors` boundverts of a bevel cap (§4 slide-ring meet-points), their centroid,
// and the boundvert-ring plane normal. Boundverts are the cap verts each touching exactly
// one cap face.
fn cap_ring_plane(mesh: &LoopMesh, result: &BevelResult) -> (Vec<Vec3>, Vec3, Vec3) {
    let cap_membership = |vertex: VertexKey| -> usize {
        result
            .faces
            .iter()
            .filter(|&&face| mesh.face_vert_keys(face).contains(&vertex))
            .count()
    };
    let boundverts: Vec<Vec3> = result
        .verts
        .iter()
        .copied()
        .filter(|&v| cap_membership(v) == 1)
        .map(|v| mesh.vertex_position(v))
        .collect();
    let centroid = boundverts.iter().fold(Vec3::ZERO, |a, &b| a + b) / boundverts.len() as f32;
    let normal = ((boundverts[1] - boundverts[0]).cross(boundverts[2] - boundverts[0])).normalize();
    (boundverts, centroid, normal)
}

// Max out-of-plane deviation of any cap vert from the boundvert-ring plane.
fn cap_max_plane_dev(mesh: &LoopMesh, result: &BevelResult) -> f32 {
    let (_boundverts, centroid, normal) = cap_ring_plane(mesh, result);
    result
        .faces
        .iter()
        .flat_map(|&f| mesh.face_vert_keys(f))
        .map(|v| (mesh.vertex_position(v) - centroid).dot(normal).abs())
        .fold(0.0_f32, f32::max)
}

// Probe a flat odd-segments cap: returns (central s-gon ring radius, ring centroid,
// max out-of-plane deviation of any cap vert from the boundvert-ring plane).
fn flat_cap_probe(mesh: &LoopMesh, result: &BevelResult, sectors: usize) -> (f32, Vec3, f32) {
    let (_boundverts, centroid, _normal) = cap_ring_plane(mesh, result);
    let max_plane = cap_max_plane_dev(mesh, result);
    // The central s-gon: the cap face whose centroid is nearest the ring centroid.
    let central = result
        .faces
        .iter()
        .copied()
        .min_by(|&a, &b| {
            let da = face_centroid(mesh, a).distance(centroid);
            let db = face_centroid(mesh, b).distance(centroid);
            da.partial_cmp(&db).unwrap()
        })
        .unwrap();
    let verts = mesh.face_vert_keys(central);
    assert_eq!(
        verts.len(),
        sectors,
        "odd central s-gon has {sectors} sides"
    );
    let radius = verts
        .iter()
        .map(|&v| mesh.vertex_position(v).distance(centroid))
        .sum::<f32>()
        / verts.len() as f32;
    (radius, centroid, max_plane)
}

fn face_centroid(mesh: &LoopMesh, face: FaceKey) -> Vec3 {
    let verts = mesh.face_vert_keys(face);
    verts
        .iter()
        .fold(Vec3::ZERO, |a, &v| a + mesh.vertex_position(v))
        / verts.len() as f32
}

/// BEVEL.md §8.11.4 + §8.5.5.5(f.1): the curved exponent builds the general grid at
/// **general valence** too, not only the valence-3 cube corner. A valence-4 pyramid apex
/// beveled at `profile = 0.7`, `segments = 2` builds the uniform coarse net — 4 slide-ring
/// boundverts, 4 chain mids, 1 welded apex, 4 cap quads — with the apex bulged toward the
/// original apex by the profile-dependent fullness.
#[test]
fn vertex_bevel_multiseg_curved_profile_pyramid_apex_general_valence() {
    let sectors = 4;
    let (mut mesh, apex) = closed_pyramid(sectors);
    let apex_position = mesh.vertex_position(apex);

    let result = mesh.bevel(
        &[apex.into()],
        BevelOptions {
            offset: 0.25,
            segments: 2,
            profile: 0.7,
            affect: BevelAffect::Vertices,
            ..Default::default()
        },
    );
    mesh.validate_topology();

    assert_eq!(
        result.verts.len(),
        2 * sectors + 1,
        "s slide-ring boundverts + s chain mids + 1 welded apex",
    );
    assert_eq!(
        result.faces.len(),
        sectors,
        "one cap quad per sector, welded apex"
    );

    let cap_membership = |vertex: VertexKey| -> usize {
        result
            .faces
            .iter()
            .filter(|&&face| mesh.face_vert_keys(face).contains(&vertex))
            .count()
    };
    let ring: Vec<VertexKey> = result
        .verts
        .iter()
        .copied()
        .filter(|&v| cap_membership(v) == 1)
        .collect();
    assert_eq!(ring.len(), sectors, "s slide-ring boundverts");
    let ring_centroid = ring
        .iter()
        .fold(Vec3::ZERO, |acc, &v| acc + mesh.vertex_position(v))
        / sectors as f32;
    let welded_apex = result
        .verts
        .iter()
        .copied()
        .find(|&v| cap_membership(v) == sectors)
        .expect("a single welded apex shared by all cap quads");
    assert!(
        (mesh.vertex_position(welded_apex) - ring_centroid).dot(apex_position - ring_centroid)
            > 0.0,
        "the curved general-valence apex bulges toward the original apex (fullness-seeded)",
    );
}

/// BEVEL.md §8.11.3: an interior valence-3 corner (a cube corner, all three sectors
/// faced, no gap) routes to the **same** general [V-ADJ] grid — vertex mode never takes
/// the cube-corner octant fast path. `segments = 3` on a closed cube keeps it closed
/// (Euler `V − E + F = 2`): `8/12/6 → 19/30/13`.
#[test]
fn vertex_bevel_multiseg_cube_corner_valence_three_routes_to_the_general_grid() {
    let mut mesh = LoopMesh::new();
    mesh_add_cube(&mut mesh);
    let corner = vert_at(&mesh, Vec3::new(1.0, -1.0, -1.0));
    assert_eq!(counts(&mesh), (8, 12, 6));

    mesh.bevel(
        &[corner.into()],
        BevelOptions {
            offset: 0.25,
            segments: 3,
            affect: BevelAffect::Vertices,
            ..Default::default()
        },
    );
    mesh.validate_topology();

    let (vertices, edges, faces) = counts(&mesh);
    assert_eq!((vertices, edges, faces), (19, 30, 13));
    assert_eq!(
        vertices as i64 - edges as i64 + faces as i64,
        2,
        "beveling one corner of a closed cube keeps it a closed genus-0 solid",
    );
}

/// BEVEL.md §8.11.3(c) at `segments = 2`: the valence-3 cube corner does **not** take
/// the tri-corner octant sphere-snap (that path is gated on `affect == EDGES`) — it
/// routes to the *same* general-fullness grid (§8.5.5.5) as the higher counts. Even
/// parity (§8.5.5.5(e)) welds a single central apex and mints **no** central face, so
/// the coarse net is: 3 §4 slide-ring boundverts, 3 `mid = v` chain interiors (one
/// interior sample per sector, §8.11.2), and 1 welded apex. `8/12/6 → 14/21/9`, still a
/// closed genus-0 solid: 3 rebuilt hexagons + 3 untouched cube quads + 3 cap quads.
#[test]
fn vertex_bevel_multiseg_cube_corner_seg2_routes_to_the_general_grid_with_a_welded_apex() {
    let mut mesh = LoopMesh::new();
    mesh_add_cube(&mut mesh);
    let corner_position = Vec3::new(1.0, -1.0, -1.0);
    let corner = vert_at(&mesh, corner_position);
    assert_eq!(counts(&mesh), (8, 12, 6));

    let result = mesh.bevel(
        &[corner.into()],
        BevelOptions {
            offset: 0.25,
            segments: 2,
            affect: BevelAffect::Vertices,
            ..Default::default()
        },
    );
    mesh.validate_topology();

    let (vertices, edges, faces) = counts(&mesh);
    assert_eq!(
        (vertices, edges, faces),
        (14, 21, 9),
        "§8.11 Stage [VV] at seg=2"
    );
    assert_eq!(
        vertices as i64 - edges as i64 + faces as i64,
        2,
        "seg=2 corner bevel keeps the cube a closed genus-0 solid",
    );
    // 3 ring + 3 chain interior (one per sector, mid = v) + 1 welded apex.
    assert_eq!(result.verts.len(), 7, "3 ring + 3 chain mid + 1 apex");
    // Even parity welds the apex and mints no central face: three cap quads only.
    assert_eq!(
        result.faces.len(),
        3,
        "three cap quads, no central polygon (welded apex)"
    );
    assert_eq!(
        face_size_multiset(&mesh),
        vec![4, 4, 4, 4, 4, 4, 6, 6, 6],
        "three cap quads + three untouched cube quads + three grown hexagons",
    );

    // Position spot-check (§8.11.2 / §8.5.5.5(b)): the coarse net bulges toward the
    // corner vertex, it does not collapse into the ring plane. Within the cap faces each
    // boundvert sits in exactly one quad, each chain mid in two, and the welded apex in
    // all three — an order-independent way to pick the three kinds apart.
    let cap_membership = |vertex: VertexKey| -> usize {
        result
            .faces
            .iter()
            .filter(|&&face| mesh.face_vert_keys(face).contains(&vertex))
            .count()
    };
    let ring_verts: Vec<VertexKey> = result
        .verts
        .iter()
        .copied()
        .filter(|&vertex| cap_membership(vertex) == 1)
        .collect();
    assert_eq!(ring_verts.len(), 3, "three §4 slide-ring boundverts");
    let ring_centroid = ring_verts
        .iter()
        .fold(Vec3::ZERO, |acc, &v| acc + mesh.vertex_position(v))
        / 3.0;
    let toward_corner = corner_position - ring_centroid;

    let apex = result
        .verts
        .iter()
        .copied()
        .find(|&vertex| cap_membership(vertex) == 3)
        .expect("a single welded apex shared by all three cap quads");
    let apex_offset = mesh.vertex_position(apex) - ring_centroid;
    assert!(
        apex_offset.dot(toward_corner) > 0.0,
        "the welded apex bulges toward the corner vertex (fullness-seeded, not flat)",
    );

    // Each chain mid is shared by exactly two cap quads and likewise bulges toward `v`.
    let chain_mids: Vec<VertexKey> = result
        .verts
        .iter()
        .copied()
        .filter(|&vertex| cap_membership(vertex) == 2)
        .collect();
    assert_eq!(
        chain_mids.len(),
        3,
        "three sector chain mids, one interior sample each"
    );
    for mid in chain_mids {
        let offset = mesh.vertex_position(mid) - ring_centroid;
        assert!(
            offset.dot(toward_corner) > 0.0,
            "each `mid = v` chain sample bulges toward the corner, not onto the flat chord",
        );
    }
}

/// BEVEL.md §8.11.4 at general valence: the vertex-mode `segments = 2` cap is
/// **uniform across valence** and enters the §8.5.5.5 general-fullness grid
/// unconditionally (§8.11.3(c) — no valence-3 fast path on the vertex side). Beveling
/// a pyramid's valence-`s` apex therefore builds the full coarse net: `s` §4 slide-ring
/// boundverts, one `mid = v` chain interior per sector (§8.11.2), and one even-parity
/// welded apex (§8.5.5.5(e)), with `s` cap quads and **no** central polygon. Verifies
/// the ring / chain / apex placement and the fullness bulge toward the corner.
fn assert_pyramid_apex_seg2_coarse_net(base_count: usize) {
    let sectors = base_count;
    let (mut mesh, apex) = closed_pyramid(base_count);
    let apex_position = mesh.vertex_position(apex);

    let result = mesh.bevel(
        &[apex.into()],
        BevelOptions {
            offset: 0.25,
            segments: 2,
            affect: BevelAffect::Vertices,
            ..Default::default()
        },
    );
    mesh.validate_topology();

    // `s` ring boundverts + one `mid = v` chain interior per sector (seg = 2) + one
    // welded apex.
    assert_eq!(
        result.verts.len(),
        2 * sectors + 1,
        "s slide-ring boundverts + s chain mids + 1 welded apex",
    );
    // Even parity welds the apex and mints no central face: one cap quad per sector.
    assert_eq!(
        result.faces.len(),
        sectors,
        "one cap quad per sector, no central polygon (welded apex)",
    );

    let (vertices, edges, faces) = counts(&mesh);
    assert_eq!(
        vertices as i64 - edges as i64 + faces as i64,
        2,
        "beveling the apex keeps the pyramid a closed genus-0 solid",
    );

    // Classify the cap verts by how many cap quads they touch: a ring boundvert sits in
    // exactly one, a chain mid in the two adjacent, the welded apex in all `s` — an
    // order-independent way to pick the three kinds apart at any valence.
    let cap_membership = |vertex: VertexKey| -> usize {
        result
            .faces
            .iter()
            .filter(|&&face| mesh.face_vert_keys(face).contains(&vertex))
            .count()
    };
    let ring_verts: Vec<VertexKey> = result
        .verts
        .iter()
        .copied()
        .filter(|&vertex| cap_membership(vertex) == 1)
        .collect();
    assert_eq!(
        ring_verts.len(),
        sectors,
        "one §4 slide-ring boundvert per incident edge"
    );
    let chain_mids: Vec<VertexKey> = result
        .verts
        .iter()
        .copied()
        .filter(|&vertex| cap_membership(vertex) == 2)
        .collect();
    assert_eq!(
        chain_mids.len(),
        sectors,
        "one chain mid per sector, shared by two cap quads"
    );
    let apex_verts: Vec<VertexKey> = result
        .verts
        .iter()
        .copied()
        .filter(|&vertex| cap_membership(vertex) == sectors)
        .collect();
    assert_eq!(
        apex_verts.len(),
        1,
        "a single welded apex shared by every cap quad"
    );

    // §8.5.5.5(b): the coarse net bulges toward the corner vertex, it does not collapse
    // into the ring plane.
    let ring_centroid = ring_verts
        .iter()
        .fold(Vec3::ZERO, |acc, &v| acc + mesh.vertex_position(v))
        / sectors as f32;
    let toward_corner = apex_position - ring_centroid;
    let apex_offset = mesh.vertex_position(apex_verts[0]) - ring_centroid;
    assert!(
        apex_offset.dot(toward_corner) > 0.0,
        "the welded apex bulges toward the corner vertex (fullness-seeded, not flat)",
    );
    for &mid in &chain_mids {
        let offset = mesh.vertex_position(mid) - ring_centroid;
        assert!(
            offset.dot(toward_corner) > 0.0,
            "each `mid = v` chain sample bulges toward the corner, not onto the flat chord",
        );
    }

    // Each ring boundvert sits *on* its own incident edge, at slide distance `offset`
    // from `v` toward the far endpoint (§4 / §8.11.1) — displaced from the corner.
    for &ring in &ring_verts {
        assert!(
            (mesh.vertex_position(ring) - apex_position).length() > 1.0e-4,
            "a slide-ring boundvert is displaced from the corner along its edge",
        );
    }
}

/// Valence 4: the square-pyramid apex. `5/8/5 → 13/20/9`: four grown side pentagons +
/// one untouched base quad + four cap quads.
#[test]
fn vertex_bevel_multiseg_valence_four_square_pyramid_seg2_builds_the_coarse_net() {
    assert_pyramid_apex_seg2_coarse_net(4);

    let (mut mesh, apex) = closed_pyramid(4);
    mesh.bevel(
        &[apex.into()],
        BevelOptions {
            offset: 0.25,
            segments: 2,
            affect: BevelAffect::Vertices,
            ..Default::default()
        },
    );
    assert_eq!(counts(&mesh), (13, 20, 9));
    assert_eq!(
        face_size_multiset(&mesh),
        vec![4, 4, 4, 4, 4, 5, 5, 5, 5],
        "base quad + four cap quads + four grown side pentagons",
    );
}

/// Valence 5: the pentagonal-pyramid apex. `6/10/6 → 16/25/11`: five grown side
/// pentagons + one untouched base pentagon + five cap quads.
#[test]
fn vertex_bevel_multiseg_valence_five_pentagonal_pyramid_seg2_builds_the_coarse_net() {
    assert_pyramid_apex_seg2_coarse_net(5);

    let (mut mesh, apex) = closed_pyramid(5);
    mesh.bevel(
        &[apex.into()],
        BevelOptions {
            offset: 0.25,
            segments: 2,
            affect: BevelAffect::Vertices,
            ..Default::default()
        },
    );
    assert_eq!(counts(&mesh), (16, 25, 11));
    assert_eq!(
        face_size_multiset(&mesh),
        vec![4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5],
        "base pentagon + five cap quads + five grown side pentagons",
    );
}

/// The gate is tight: a single segment and a valence below 3 decline the multi-segment
/// grid rather than mis-building. Every SUPERELLIPSE exponent — circle, curved, square,
/// square-in, and the flat line (`profile = 0.25`, §8.5.5.5(f.3) law F2) — is now *in*
/// scope at `segments ≥ 2`, routing to the same general-fullness grid; only custom
/// profiles (unreachable here) and `segments < 2` / valence `< 3` fall out.
#[test]
fn vertex_bevel_multiseg_out_of_scope_cases_decline_to_no_op() {
    // Single segment — the multi-segment grid needs `segments ≥ 2`; a single-segment
    // vertex bevel takes the foundational path instead, so `bevel_vertices_multiseg`
    // itself declines here.
    let (mut mesh, _) = two_quad_strip();
    let corner = vert_at(&mesh, Vec3::new(0.0, -1.0, 0.0));
    assert!(
        mesh.bevel_vertices_multiseg(
            &[corner.into()],
            &BevelOptions {
                offset: 0.25,
                segments: 1,
                profile: 0.25,
                affect: BevelAffect::Vertices,
                ..Default::default()
            },
        )
        .is_none(),
        "single-segment multiseg still declines",
    );

    // Valence 2 — the general grid needs at least three sectors to close (§8.11.4 /
    // §2.3: a two-corner cap is no face). The `sectors < 3` floor still declines.
    let (mut mesh, centre, _) = open_planar_fan(2, 0);
    let before = counts(&mesh);
    mesh.bevel(
        &[centre.into()],
        BevelOptions {
            offset: 0.25,
            segments: 3,
            affect: BevelAffect::Vertices,
            ..Default::default()
        },
    );
    assert_eq!(counts(&mesh), before, "valence < 3 still no-ops");
}

/// BEVEL.md §7.2.1's worked open-fan wire: a single quad with a wire hanging off one corner
/// into the void. Valence 3, one incident face — `5/5/1 → 7/8/2`, sizes `[3, 5]`: the quad
/// grown by one to a pentagon and a **triangle cap** over the three ring verts (two faced
/// spokes and the wire).
#[test]
fn vertex_bevel_quad_with_a_wire_off_one_corner_caps_the_open_fan() {
    let mut mesh = LoopMesh::new();
    let corners: Vec<VertexKey> = [
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(1.0, 1.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
    ]
    .into_iter()
    .map(|position| mesh.create_vertex(position))
    .collect();
    mesh.face_create_verts(&corners, FaceCreateFlags::NONE);
    let tip = mesh.create_vertex(Vec3::new(2.0, -1.0, 0.0));
    mesh.create_edge([corners[1], tip], EdgeCreateFlags::NONE);
    assert_eq!(counts(&mesh), (5, 5, 1));

    let result = mesh.bevel(
        &[corners[1].into()],
        BevelOptions {
            offset: 0.25,
            segments: 1,
            affect: BevelAffect::Vertices,
            ..Default::default()
        },
    );
    mesh.validate_topology();

    assert_eq!(counts(&mesh), (7, 8, 2), "+2 V / +3 E / +1 F");
    assert_eq!(face_size_multiset(&mesh), vec![3, 5]);
    assert_eq!(result.faces.len(), 1);
    assert_eq!(
        mesh.face_vert_keys(result.faces[0]).len(),
        3,
        "the cap is a triangle over the two faced spokes' ring verts and the wire's",
    );
    assert_eq!(
        wire_landings(&mesh, tip).len(),
        1,
        "the wire survives, re-pointed onto its own ring vert before the corner is freed",
    );
}

/// BEVEL.md §5.1's winding rule, and the one fact no count, face-size or canonical-topology
/// assertion can see: **an inward cap has the same incidences as an outward one.** The cap
/// traverses the ring *in* the disk order, which is equivalently "opposite every rebuilt face
/// on their shared chamfer edge" — so on the outward-wound cube its Newell normal points
/// **out**, along the beveled corner's own outward diagonal. Winding the ring the other way
/// mints an inward-facing cap that every other assertion in this file would wave through.
#[test]
fn vertex_bevel_cap_normal_points_out_of_the_cube_corner() {
    let mut mesh = LoopMesh::new();
    mesh_add_cube(&mut mesh);
    let corner = vert_at(&mesh, Vec3::new(1.0, -1.0, -1.0));
    let outward = mesh.vertex_position(corner).normalize();

    let result = mesh.bevel(
        &[corner.into()],
        BevelOptions {
            offset: 0.25,
            segments: 1,
            affect: BevelAffect::Vertices,
            ..Default::default()
        },
    );
    mesh.validate_topology();

    assert_eq!(result.faces.len(), 1, "one triangular cap");
    let cap = result.faces[0];
    let normal = mesh.face_plane_normal(cap);
    assert!(
        normal.dot(outward) > 0.99,
        "the cap faces the way the removed corner's surrounding faces faced: got {normal:?}",
    );

    // The other half of the same rule: each chamfer edge is walked in *opposite* directions
    // by the cap and by the rebuilt face sharing it — the manifold condition that leaves
    // every one of them at a radial count of exactly 2.
    for cap_loop in mesh.face_loop_keys(cap) {
        let edge = mesh.loop_edge(cap_loop);
        let radial: Vec<LoopKey> = mesh.edge_loop_keys(edge);
        assert_eq!(radial.len(), 2, "a closed fan shares every ring edge");
        let cap_from = mesh.loop_vertex(cap_loop);
        let cap_to = mesh.loop_vertex(mesh.loop_next(cap_loop));
        let face_loop = radial
            .into_iter()
            .find(|&other| other != cap_loop)
            .expect("the rebuilt face's loop on the shared chamfer edge");
        let face_from = mesh.loop_vertex(face_loop);
        let face_to = mesh.loop_vertex(mesh.loop_next(face_loop));
        assert!(
            face_from == cap_to && face_to == cap_from,
            "the rebuilt face runs the chamfer edge descending, the cap ascending",
        );
    }
}
