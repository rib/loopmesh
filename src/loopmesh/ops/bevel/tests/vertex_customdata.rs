use super::*;

const HEAT: &str = "heat";

const SERIAL: &str = "serial";

/// A cube decorated per vertex with two layers: an interpolable float `heat`, and
/// a `serial` the blend never touches (its layer type defines no interpolation).
/// `heat` pins the §5.2.2 blend; `serial` pins the §5.2.1 **step 1** mint, because
/// it can only reach a new vertex by being *copied from the beveled corner* — a
/// default-initialised mint leaves it at `0`.
fn decorated_cube() -> (LoopMesh, ElementAttributeKey, ElementAttributeKey) {
    let mut mesh = LoopMesh::new();
    mesh_add_cube(&mut mesh);
    let heat = mesh
        .register_vertex_attribute(ElementAttributeDesc::numeric(
            AttributeKey::custom(HEAT),
            AttrType::F32,
        ))
        .expect("F32 is a supported vertex attribute format");
    let serial = mesh
        .register_vertex_attribute(
            ElementAttributeDesc::numeric(AttributeKey::custom(SERIAL), AttrType::I32)
                .with_interp(InterpPolicy::NoInterp),
        )
        .expect("I32 is a supported vertex attribute format");

    // A value per corner that no two corners share and that is not affine in the
    // position, so a mis-picked donor or a collapsed weight vector shows up.
    for (index, vertex) in mesh.vert_keys().into_iter().enumerate() {
        let p = mesh.vertex_position(vertex);
        let value = 10.0 + 3.0 * p.x + 5.0 * p.y + 11.0 * p.z + 2.0 * p.x * p.y * p.z;
        mesh.vertex_attribs_mut()
            .set_f32(heat, vertex.as_usize(), value);
        mesh.vertex_attribs_mut()
            .set_i32(serial, vertex.as_usize(), index as i32 + 1);
    }
    (mesh, heat, serial)
}

fn heat_of(mesh: &LoopMesh, layer: ElementAttributeKey, vertex: VertexKey) -> f32 {
    mesh.vertex_attribs()
        .get_f32(layer, vertex.as_usize())
        .expect("the heat layer is registered")
}

fn serial_of(mesh: &LoopMesh, layer: ElementAttributeKey, vertex: VertexKey) -> i32 {
    mesh.vertex_attribs()
        .get_i32(layer, vertex.as_usize())
        .expect("the serial layer is registered")
}

/// The realised slide fraction `t = |r − v| / |far − v|` (§5.2.2), read back off
/// the *positions* — never off the offset parameter.
fn realised_fraction(mesh: &LoopMesh, new_vertex: VertexKey, corner: Vec3, far: Vec3) -> f32 {
    let p = mesh.vertex_position(new_vertex);
    (p - corner).length() / (far - corner).length()
}

/// BEVEL.md §5.2.2: a single-segment ring vert lies on its original edge, so every
/// donor that writes it has that edge on its boundary and the blend collapses to
/// the edge-linear closed form `(1 − t)·cd(v) + t·cd(far)` at the *realised*
/// fraction. On the cube at `offset = 0.25`, `t = 0.125`.
#[test]
fn vertex_bevel_ring_verts_blend_edge_linearly_from_the_beveled_corner() {
    let (mut mesh, heat, _serial) = decorated_cube();
    let corner = vert_at(&mesh, Vec3::new(1.0, -1.0, -1.0));
    let corner_position = mesh.vertex_position(corner);
    let corner_heat = heat_of(&mesh, heat, corner);

    // The three far endpoints and their values, read pre-bevel.
    let legs: Vec<(Vec3, f32)> = mesh
        .iter_disk_edges(corner)
        .collect::<Vec<_>>()
        .into_iter()
        .map(|edge| {
            let [a, b] = mesh.edge_vert_keys(edge);
            let far = if a == corner { b } else { a };
            (mesh.vertex_position(far), heat_of(&mesh, heat, far))
        })
        .collect();
    assert_eq!(legs.len(), 3);

    let result = mesh.bevel(
        &[corner.into()],
        BevelOptions {
            offset: 0.25,
            ..Default::default()
        },
    );
    assert_eq!(result.verts.len(), 3);

    for &ring_vertex in &result.verts {
        let position = mesh.vertex_position(ring_vertex);
        let (far_position, far_heat) = *legs
            .iter()
            .min_by(|(a, _), (b, _)| {
                let da = (position - corner_position)
                    .normalize()
                    .dot((*a - corner_position).normalize());
                let db = (position - corner_position)
                    .normalize()
                    .dot((*b - corner_position).normalize());
                db.partial_cmp(&da).expect("finite")
            })
            .expect("the ring vert sits on one of the three legs");

        let t = realised_fraction(&mesh, ring_vertex, corner_position, far_position);
        assert!((t - 0.125).abs() < 1.0e-5, "realised fraction {t}");

        let expected = (1.0 - t) * corner_heat + t * far_heat;
        let actual = heat_of(&mesh, heat, ring_vertex);
        assert!(
            (actual - expected).abs() < 1.0e-4,
            "ring vert heat {actual} != edge-linear {expected} (corner {corner_heat}, far \
             {far_heat}, t {t})",
        );
        // The failure mode the whole §5.2.1 fact-list exists to prevent: a donor
        // read post-re-stitch, or collapsed to one face, gives the corner's block
        // back verbatim.
        assert!(
            (actual - corner_heat).abs() > 1.0e-3,
            "ring vert took cd(v) verbatim — the donor was not the pre-bevel polygon",
        );
    }

    mesh.validate_topology();
}

/// BEVEL.md §5.2.2 / §4.1: an `offset` at or past the edge length clamps the slide,
/// so the ring vert lands **exactly on** `far` — the realised fraction is `t = 1`
/// and the weight vector is an on-corner identity at `far` (BMESH.md §10.1.3).
///
/// This is where a block copy and a blend part company. The interpolable layer must
/// come out as `far`'s value (`t = 1` in the closed form). The **no-interp** layer
/// must come out as `cd(v)`'s: BMESH.md §10.1.4 says the blend does not touch such
/// a layer — it neither synthesises nor copies a source value — so what survives is
/// what the §5.2.1 step-1 mint put there, the beveled corner's own serial. An
/// implementation that short-circuits the on-corner case to a whole-block copy
/// stamps `far`'s serial here instead, and nothing else in the suite sees it.
#[test]
fn vertex_bevel_clamped_ring_vert_takes_the_far_corner_but_keeps_the_corner_no_interp() {
    let (mut mesh, heat, serial) = decorated_cube();
    let corner = vert_at(&mesh, Vec3::new(1.0, -1.0, -1.0));
    let corner_position = mesh.vertex_position(corner);
    let corner_serial = serial_of(&mesh, serial, corner);

    let legs: Vec<(Vec3, f32, i32)> = mesh
        .iter_disk_edges(corner)
        .collect::<Vec<_>>()
        .into_iter()
        .map(|edge| {
            let [a, b] = mesh.edge_vert_keys(edge);
            let far = if a == corner { b } else { a };
            (
                mesh.vertex_position(far),
                heat_of(&mesh, heat, far),
                serial_of(&mesh, serial, far),
            )
        })
        .collect();
    assert_eq!(legs.len(), 3);

    // Cube edge length 2; the slide clamps at the edge's far end (§4.1).
    let result = mesh.bevel(
        &[corner.into()],
        BevelOptions {
            offset: 2.0,
            ..Default::default()
        },
    );
    assert_eq!(result.verts.len(), 3);

    for &ring_vertex in &result.verts {
        let position = mesh.vertex_position(ring_vertex);
        let (far_position, far_heat, far_serial) = *legs
            .iter()
            .min_by(|(a, _, _), (b, _, _)| {
                let da = (position - *a).length();
                let db = (position - *b).length();
                da.partial_cmp(&db).expect("finite")
            })
            .expect("the ring vert sits on one of the three legs");
        assert_ne!(far_serial, corner_serial, "the fixture separates the two");

        let t = realised_fraction(&mesh, ring_vertex, corner_position, far_position);
        assert!((t - 1.0).abs() < 1.0e-5, "the slide did not clamp: t = {t}");

        let actual_heat = heat_of(&mesh, heat, ring_vertex);
        assert!(
            (actual_heat - far_heat).abs() < 1.0e-4,
            "clamped ring vert heat {actual_heat} != cd(far) {far_heat} at t = 1",
        );
        assert_eq!(
            serial_of(&mesh, serial, ring_vertex),
            corner_serial,
            "the on-corner blend copied the donor corner's whole block — a no-interp \
             layer must keep the value the mint from `v` gave it",
        );
    }

    mesh.validate_topology();
}

/// BEVEL.md §5.2.1 step 1: the mint is a **copy of the beveled corner's block**,
/// not the layer defaults. A layer whose type defines no interpolation is left
/// untouched by the step-2 blend (BMESH.md §10.1.4), so it is exactly where the
/// mint's baseline shows through: every new vertex must carry `v`'s serial.
#[test]
fn vertex_bevel_mints_the_new_verts_from_the_corner_not_the_defaults() {
    let (mut mesh, _heat, serial) = decorated_cube();
    let corner = vert_at(&mesh, Vec3::new(1.0, -1.0, -1.0));
    let corner_serial = serial_of(&mesh, serial, corner);
    assert_ne!(corner_serial, 0, "the corner carries a non-default serial");

    let result = mesh.bevel(
        &[corner.into()],
        BevelOptions {
            offset: 0.25,
            ..Default::default()
        },
    );

    for &ring_vertex in &result.verts {
        assert_eq!(
            serial_of(&mesh, serial, ring_vertex),
            corner_serial,
            "a minted vertex kept the layer default — it was not created from the corner",
        );
    }
}

/// BEVEL.md §5.2.1 step 1 / BMESH.md §13: the example copy carries every header
/// flag except the selection bit — the **tag** among them — so the operator clears
/// the tag on each vertex it mints. Everything else (here, the hidden bit) rides
/// along.
#[test]
fn vertex_bevel_mints_untagged_but_otherwise_flagged_like_the_corner() {
    let (mut mesh, _heat, _serial) = decorated_cube();
    let corner = vert_at(&mesh, Vec3::new(1.0, -1.0, -1.0));
    mesh.set_vert_tag(corner, true);
    mesh.set_vert_hidden(corner, true);

    let result = mesh.bevel(
        &[corner.into()],
        BevelOptions {
            offset: 0.25,
            ..Default::default()
        },
    );

    for &ring_vertex in &result.verts {
        assert!(
            !mesh.vert_tag(ring_vertex),
            "a ring vert was minted tagged — a tag-keyed teardown sweep would destroy it",
        );
        assert!(
            mesh.is_vert_hidden(ring_vertex),
            "the example copy must carry the corner's other header flags",
        );
    }
}

/// BEVEL.md §5.2.1 property (3): a re-stitched original blends **all** its corners,
/// the untouched ones included — and those sample at their own donor corner, take
/// weight `1` by the on-corner short-circuit, and keep their block exactly. An
/// implementation that lets the smooth formula run instead passes every ring-vert
/// assertion and silently drifts every untouched vertex of every rebuilt face.
#[test]
fn vertex_bevel_untouched_verts_of_a_rebuilt_face_keep_their_block_exactly() {
    let (mut mesh, heat, serial) = decorated_cube();
    let corner = vert_at(&mesh, Vec3::new(1.0, -1.0, -1.0));
    let before: Vec<(VertexKey, f32, i32)> = mesh
        .vert_keys()
        .into_iter()
        .filter(|&v| v != corner)
        .map(|v| (v, heat_of(&mesh, heat, v), serial_of(&mesh, serial, v)))
        .collect();

    mesh.bevel(
        &[corner.into()],
        BevelOptions {
            offset: 0.25,
            ..Default::default()
        },
    );

    for (vertex, expected_heat, expected_serial) in before {
        assert_eq!(
            heat_of(&mesh, heat, vertex),
            expected_heat,
            "an untouched vertex drifted — the on-corner short-circuit is not exact",
        );
        assert_eq!(serial_of(&mesh, serial, vertex), expected_serial);
    }
}

/// BEVEL.md §2.1 / §6.3: a selected vertex the operator declines to bevel — here an
/// isolated vert, and a vert with a single incident edge — must **survive**. The
/// teardown frees exactly the corners that carry a bevel record; an implementation
/// that sweeps a tag instead deletes geometry it never touched.
#[test]
fn vertex_bevel_leaves_an_ineligible_vertex_alone() {
    let (mut mesh, heat, _serial) = decorated_cube();
    let corner = vert_at(&mesh, Vec3::new(1.0, -1.0, -1.0));

    let isolated = mesh.create_vertex(Vec3::new(5.0, 0.0, 0.0));
    let stub_far = mesh.create_vertex(Vec3::new(5.0, 2.0, 0.0));
    let single_edge = mesh.create_vertex(Vec3::new(5.0, 1.0, 0.0));
    mesh.create_edge([single_edge, stub_far], EdgeCreateFlags::NONE);
    for (index, &vertex) in [isolated, single_edge, stub_far].iter().enumerate() {
        mesh.vertex_attribs_mut()
            .set_f32(heat, vertex.as_usize(), 100.0 + index as f32);
    }

    mesh.bevel(
        &[corner.into(), isolated.into(), single_edge.into()],
        BevelOptions {
            offset: 0.25,
            ..Default::default()
        },
    );

    let live = mesh.vert_keys();
    assert!(live.contains(&isolated), "the isolated vertex was deleted");
    assert!(
        live.contains(&single_edge),
        "the single-edge vertex was deleted",
    );
    assert_eq!(heat_of(&mesh, heat, isolated), 100.0);
    assert_eq!(heat_of(&mesh, heat, single_edge), 101.0);
    assert!(
        !live.contains(&corner),
        "the eligible corner is still beveled away",
    );

    mesh.validate_topology();
}

/// BEVEL.md §5.2.3: at an edge bevel the owning edge of a strip vert is the **leg**
/// it was slid onto, never the beveled edge. Bevelling one cube edge `E = (v, w)`,
/// each strip vert at `v` blends `cd(v)` with the far end of the *non-`E`* edge it
/// sits on; `E`'s own far endpoint appears in no strip vert's blend at that end.
#[test]
fn edge_bevel_strip_verts_blend_along_their_owning_leg() {
    let (mut mesh, heat, serial) = decorated_cube();
    let near = vert_at(&mesh, Vec3::new(1.0, -1.0, -1.0));
    let far_end = vert_at(&mesh, Vec3::new(1.0, -1.0, 1.0));
    let beveled = mesh
        .query_edge_exists(near, far_end)
        .expect("the cube has this edge");

    // The two legs of `near` other than `E`, and their far endpoints' values.
    let legs: Vec<(Vec3, f32)> = mesh
        .iter_disk_edges(near)
        .collect::<Vec<_>>()
        .into_iter()
        .filter(|&edge| edge != beveled)
        .map(|edge| {
            let [a, b] = mesh.edge_vert_keys(edge);
            let leg_far = if a == near { b } else { a };
            (mesh.vertex_position(leg_far), heat_of(&mesh, heat, leg_far))
        })
        .collect();
    assert_eq!(legs.len(), 2);

    let near_position = mesh.vertex_position(near);
    let near_heat = heat_of(&mesh, heat, near);
    let near_serial = serial_of(&mesh, serial, near);
    let beveled_far_heat = heat_of(&mesh, heat, far_end);

    let result = mesh.bevel(
        &[beveled.into()],
        BevelOptions {
            offset: 0.25,
            affect: BevelAffect::Edges,
            ..Default::default()
        },
    );
    assert_eq!(result.verts.len(), 4, "two strip verts per endpoint");

    let at_near: Vec<VertexKey> = result
        .verts
        .iter()
        .copied()
        .filter(|&v| (mesh.vertex_position(v) - near_position).length() < 1.0)
        .collect();
    assert_eq!(at_near.len(), 2, "two strip verts at the near endpoint");

    for &strip_vertex in &at_near {
        let position = mesh.vertex_position(strip_vertex);
        let (leg_far_position, leg_far_heat) = *legs
            .iter()
            .min_by(|(a, _), (b, _)| {
                let da = (position - *a).length();
                let db = (position - *b).length();
                da.partial_cmp(&db).expect("finite")
            })
            .expect("the strip vert sits on one of the two legs");

        let t = realised_fraction(&mesh, strip_vertex, near_position, leg_far_position);
        let expected = (1.0 - t) * near_heat + t * leg_far_heat;
        let actual = heat_of(&mesh, heat, strip_vertex);
        assert!(
            (actual - expected).abs() < 1.0e-4,
            "strip vert heat {actual} != leg-linear {expected} (endpoint {near_heat}, leg far \
             {leg_far_heat}, t {t}); the beveled edge's far end reads {beveled_far_heat} and \
             must not appear at all",
        );
        assert_eq!(
            serial_of(&mesh, serial, strip_vertex),
            near_serial,
            "the strip vert was not minted from its own endpoint",
        );
    }

    mesh.validate_topology();
}

// ---- §5.2.5: per-loop customdata across the chamfer ----

const SHADE: &str = "shade";

/// A cube whose every face-corner carries a value that is affine in the corner's
/// **position** and offset by a code for the **face** it belongs to. The face code
/// is what makes the layer genuinely per-loop: the four loops meeting at one cube
/// vertex hold four different values, so a blend that read the wrong face's corners
/// — or read the corner's *vertex* data instead — lands on a value this fixture can
/// name.
fn loop_decorated_cube() -> (LoopMesh, ElementAttributeKey) {
    let mut mesh = LoopMesh::new();
    mesh_add_cube(&mut mesh);
    let shade = mesh
        .register_loop_attribute(ElementAttributeDesc::numeric(
            AttributeKey::custom(SHADE),
            AttrType::F32,
        ))
        .expect("F32 is a supported loop attribute format");

    for face in mesh.face_keys() {
        let code = face_code(mesh.face_plane_normal(face));
        for corner in mesh.face_loop_keys(face) {
            let value = loop_field(code, mesh.vertex_position(mesh.loop_vertex(corner)));
            mesh.loop_attribs_mut()
                .set_f32(shade, corner.as_usize(), value);
        }
    }
    (mesh, shade)
}

/// A cube whose per-loop value depends on the corner's **position alone** — every
/// face meeting at a vertex holds the *same* value there, so by BEVEL.md §6.6.1.4 no
/// edge of it is an interpolation seam: condition (1) cannot fire on a closed cube,
/// and condition (2)'s contiguity test agrees at both endpoints of every edge. A
/// beveled corner of this cube is therefore **not seam-carrying**, and its cap takes
/// §5.2.1.1's positional per-boundvert donor branch — the one that is implemented.
///
/// [`loop_decorated_cube`]'s face code, by contrast, makes *every* cube edge a seam
/// by condition (2), which routes its caps down the override branch. That is fine for
/// the re-stitched-original tests (whose donor is the face itself either way) but not
/// for a cap test.
fn loop_continuous_cube() -> (LoopMesh, ElementAttributeKey) {
    let mut mesh = LoopMesh::new();
    mesh_add_cube(&mut mesh);
    let shade = mesh
        .register_loop_attribute(ElementAttributeDesc::numeric(
            AttributeKey::custom(SHADE),
            AttrType::F32,
        ))
        .expect("F32 is a supported loop attribute format");

    for face in mesh.face_keys() {
        for corner in mesh.face_loop_keys(face) {
            let value = continuous_loop_field(mesh.vertex_position(mesh.loop_vertex(corner)));
            mesh.loop_attribs_mut()
                .set_f32(shade, corner.as_usize(), value);
        }
    }
    (mesh, shade)
}

/// Non-zero at every cube corner and at every ring vert a `0.25` bevel of one, so an
/// unblended corner (which reads the layer default `0`) is never mistaken for a
/// blended one.
fn continuous_loop_field(position: Vec3) -> f32 {
    7.0 + 3.0 * position.x + 5.0 * position.y + 11.0 * position.z
}

/// One of the cube's six axis-aligned face directions, as a small integer. A
/// rebuilt face keeps its plane, so this identifies an original face *after* the
/// re-stitch has given it a new key.
fn face_code(normal: Vec3) -> i32 {
    let axis = if normal.x.abs() > normal.y.abs() && normal.x.abs() > normal.z.abs() {
        0
    } else if normal.y.abs() > normal.z.abs() {
        1
    } else {
        2
    };
    let sign = if normal[axis] > 0.0 { 1 } else { 0 };
    (axis * 2 + sign) as i32
}

fn loop_field(face_code: i32, position: Vec3) -> f32 {
    100.0 * (face_code as f32 + 1.0) + 3.0 * position.x + 5.0 * position.y + 11.0 * position.z
}

fn shade_of(mesh: &LoopMesh, layer: ElementAttributeKey, corner: LoopKey) -> f32 {
    mesh.loop_attribs()
        .get_f32(layer, corner.as_usize())
        .expect("the shade layer is registered")
}

/// The loop of `face` at `vertex`.
fn corner_of(mesh: &LoopMesh, face: FaceKey, vertex: VertexKey) -> LoopKey {
    mesh.face_loop_keys(face)
        .into_iter()
        .find(|&corner| mesh.loop_vertex(corner) == vertex)
        .expect("the face has a corner at this vertex")
}

fn face_with_normal(mesh: &LoopMesh, code: i32) -> FaceKey {
    mesh.face_keys()
        .into_iter()
        .find(|&face| face_code(mesh.face_plane_normal(face)) == code)
        .expect("the cube has a face in this plane")
}

/// BEVEL.md §5.2.5 + §5.2.2: the per-loop blend is the same call as the per-vertex
/// one — same donor, same weights — so a re-stitched original's new corner at a ring
/// vert takes **that face's own pre-bevel corner loops**, collapsed by the on-edge
/// short-circuit to the edge-linear form on the leg the ring vert sits on. The
/// donated values are the *face's* corner values, not the beveled vertex's, and they
/// are certainly not the layer default.
#[test]
fn vertex_bevel_rebuilt_face_corners_blend_the_loops_across_the_chamfer() {
    let (mut mesh, shade) = loop_decorated_cube();
    let corner = vert_at(&mesh, Vec3::new(1.0, -1.0, -1.0));
    let corner_position = mesh.vertex_position(corner);

    // The three faces at the corner, and each face's two legs at it, read pre-bevel.
    let incident: Vec<(i32, Vec<(Vec3, f32)>, f32)> = mesh
        .vert_face_keys(corner)
        .into_iter()
        .map(|face| {
            let code = face_code(mesh.face_plane_normal(face));
            let corner_shade = shade_of(&mesh, shade, corner_of(&mesh, face, corner));
            let legs = mesh
                .face_vert_keys(face)
                .into_iter()
                .filter(|&w| w != corner && mesh.query_edge_exists(corner, w).is_some())
                .map(|far| {
                    (
                        mesh.vertex_position(far),
                        shade_of(&mesh, shade, corner_of(&mesh, face, far)),
                    )
                })
                .collect::<Vec<_>>();
            assert_eq!(legs.len(), 2, "a cube face meets two of the corner's legs");
            (code, legs, corner_shade)
        })
        .collect();
    assert_eq!(incident.len(), 3);

    mesh.bevel(
        &[corner.into()],
        BevelOptions {
            offset: 0.25,
            ..Default::default()
        },
    );

    for (code, legs, corner_shade) in incident {
        let face = face_with_normal(&mesh, code);
        // The rebuilt face's two new corners: the ring verts, each on one of the
        // face's two legs at the beveled corner.
        let new_corners: Vec<LoopKey> = mesh
            .face_loop_keys(face)
            .into_iter()
            .filter(|&loop_key| {
                (mesh.vertex_position(mesh.loop_vertex(loop_key)) - corner_position).length() < 0.5
            })
            .collect();
        assert_eq!(new_corners.len(), 2, "the corner run has two ring verts");

        for new_corner in new_corners {
            let position = mesh.vertex_position(mesh.loop_vertex(new_corner));
            let &(far_position, far_shade) = legs
                .iter()
                .min_by(|(a, _), (b, _)| {
                    let da = (position - corner_position)
                        .normalize()
                        .dot((*a - corner_position).normalize());
                    let db = (position - corner_position)
                        .normalize()
                        .dot((*b - corner_position).normalize());
                    db.partial_cmp(&da).expect("finite")
                })
                .expect("the ring vert sits on one of the face's two legs");

            let t =
                (position - corner_position).length() / (far_position - corner_position).length();
            assert!((t - 0.125).abs() < 1.0e-5, "realised fraction {t}");

            let expected = (1.0 - t) * corner_shade + t * far_shade;
            let actual = shade_of(&mesh, shade, new_corner);
            assert!(
                (actual - expected).abs() < 1.0e-3,
                "rebuilt corner shade {actual} != edge-linear {expected} on face {code} \
                 (corner loop {corner_shade}, far loop {far_shade}, t {t})",
            );
            assert!(
                actual.abs() > 1.0e-3,
                "the rebuilt corner read the layer default — no per-loop blend ran",
            );
        }
    }

    mesh.validate_topology();
}

/// BEVEL.md §5.2.1 property (3), per-loop half: a re-stitched original blends **all**
/// its corners, and a corner sampled at its own donor corner takes weight `1` — so
/// the surviving corners of a rebuilt face must come back holding the loop values the
/// killed face carried there. The face's loops are destroyed with it, so a missing
/// blend here shows up as the layer default, not as drift.
#[test]
fn vertex_bevel_untouched_corners_of_a_rebuilt_face_keep_their_loop_block() {
    let (mut mesh, shade) = loop_decorated_cube();
    let corner = vert_at(&mesh, Vec3::new(1.0, -1.0, -1.0));

    mesh.bevel(
        &[corner.into()],
        BevelOptions {
            offset: 0.25,
            ..Default::default()
        },
    );

    for face in mesh.face_keys() {
        let code = face_code(mesh.face_plane_normal(face));
        for loop_key in mesh.face_loop_keys(face) {
            let position = mesh.vertex_position(mesh.loop_vertex(loop_key));
            // Only the original (un-slid) cube corners: the cube's own verts are the
            // only positions with all-unit coordinates.
            if position.abs().min_element() < 0.99 {
                continue;
            }
            let expected = loop_field(code, position);
            let actual = shade_of(&mesh, shade, loop_key);
            assert!(
                (actual - expected).abs() < 1.0e-3,
                "an untouched corner of a rebuilt face lost its loop value: {actual} != \
                 {expected}",
            );
        }
    }
}

/// BEVEL.md §5.2.1.1 + §5.2.5: the cap is the **only** face that owns its corners'
/// loops, so its blend decides them outright. Its donor list is per corner —
/// boundvert `i` takes the face whose winding *arrives* at `v` along `e_i` — and the
/// ring vert lies on `e_i`, which is that donor's boundary edge, so the value is the
/// edge-linear one **in that donor face's loops**. Collapse the list to one face and
/// the corner on the leg that face does not hold projects onto `v` itself, taking
/// `v`'s value instead of the edge-linear one.
///
/// Pins the **non-seam-carrying** branch, and so runs on the seam-free cube: a corner
/// with any seamed incident leg takes a §6.6.1.4 override face plus a weight-snap
/// edge instead, and that branch is unimplemented. Rewrite this test when it lands.
#[test]
fn vertex_bevel_cap_corners_blend_from_their_own_boundvert_donor() {
    let (mut mesh, shade) = loop_continuous_cube();
    let corner = vert_at(&mesh, Vec3::new(1.0, -1.0, -1.0));
    let corner_position = mesh.vertex_position(corner);

    // Per leg: the far endpoint, and the arriving face's loop values at both ends of
    // that leg — the donor the cap's corner on this leg must use.
    let legs: Vec<(Vec3, f32, f32)> = mesh
        .iter_disk_edges(corner)
        .collect::<Vec<_>>()
        .into_iter()
        .map(|edge| {
            let [a, b] = mesh.edge_vert_keys(edge);
            let far = if a == corner { b } else { a };
            let arriving = mesh
                .vert_face_keys(corner)
                .into_iter()
                .find(|&face| {
                    let cycle = mesh.face_vert_keys(face);
                    let index = cycle
                        .iter()
                        .position(|&w| w == corner)
                        .expect("an incident face has the corner");
                    let previous = cycle[(index + cycle.len() - 1) % cycle.len()];
                    mesh.query_edge_exists(previous, corner) == Some(edge)
                })
                .expect("a closed fan has an arriving face on every leg");
            (
                mesh.vertex_position(far),
                shade_of(&mesh, shade, corner_of(&mesh, arriving, corner)),
                shade_of(&mesh, shade, corner_of(&mesh, arriving, far)),
            )
        })
        .collect();
    assert_eq!(legs.len(), 3);

    let result = mesh.bevel(
        &[corner.into()],
        BevelOptions {
            offset: 0.25,
            ..Default::default()
        },
    );
    assert_eq!(result.faces.len(), 1, "one cap");
    let cap = result.faces[0];

    for cap_corner in mesh.face_loop_keys(cap) {
        let position = mesh.vertex_position(mesh.loop_vertex(cap_corner));
        let &(far_position, donor_corner_shade, donor_far_shade) = legs
            .iter()
            .min_by(|(a, _, _), (b, _, _)| {
                let da = (position - corner_position)
                    .normalize()
                    .dot((*a - corner_position).normalize());
                let db = (position - corner_position)
                    .normalize()
                    .dot((*b - corner_position).normalize());
                db.partial_cmp(&da).expect("finite")
            })
            .expect("the cap corner is a ring vert on one of the three legs");

        let t = (position - corner_position).length() / (far_position - corner_position).length();
        let expected = (1.0 - t) * donor_corner_shade + t * donor_far_shade;
        let actual = shade_of(&mesh, shade, cap_corner);
        assert!(
            (actual - expected).abs() < 1.0e-3,
            "cap corner shade {actual} != the arriving donor's edge-linear {expected} \
             (donor loops {donor_corner_shade} / {donor_far_shade}, t {t})",
        );
        assert!(
            actual.abs() > 1.0e-3,
            "the cap corner read the layer default — no per-loop blend ran on the cap",
        );
    }

    mesh.validate_topology();
}

// ---- §5.2.1.1 seam-carrying cap override donor + §5.2.6 UV-merge post-pass ----

/// A boundary valence-3 corner (`b` of [`two_quad_strip`], three spokes, two faces, an open
/// fan) carrying one float2 UV loop layer. `uv(is_left, position)` decorates each loop; the
/// left quad takes `left_slot`, the right `right_slot`. Beveled at `offset = 0.4`, so each
/// ring vert slides `0.4` along its leg (the ring vert on the `b → c` leg lands at
/// `(0.4, -1, 0)`). Returns the beveled corner and the UV layer key.
fn boundary_corner_uv_plane(
    uv: impl Fn(bool, Vec3) -> [f32; 2],
    left_slot: i32,
    right_slot: i32,
) -> (LoopMesh, VertexKey, ElementAttributeKey) {
    let (mut mesh, _shared) = two_quad_strip();
    let key = mesh
        .register_loop_attribute(ElementAttributeDesc::uv(AttributeKey::custom("uv")))
        .expect("Vec2 is a supported loop attribute format");
    let corner = vert_at(&mesh, Vec3::new(0.0, -1.0, 0.0));
    let left = vert_at(&mesh, Vec3::new(-2.0, -1.0, 0.0));
    for face in mesh.face_keys() {
        let is_left = mesh.face_vert_keys(face).contains(&left);
        mesh.set_face_material_index(face, if is_left { left_slot } else { right_slot });
        for loop_key in mesh.face_loop_keys(face) {
            let position = mesh.vertex_position(mesh.loop_vertex(loop_key));
            mesh.loop_attribs_mut()
                .set_vec2(key, loop_key.as_usize(), uv(is_left, position));
        }
    }
    (mesh, corner, key)
}

/// The ring vert on the `b → c` leg — the shared corner of the cap and the re-stitched
/// right pentagon in these fixtures (world `(0.4, -1, 0)` at `offset = 0.4`).
fn right_leg_ring_vert(mesh: &LoopMesh, result: &BevelResult) -> VertexKey {
    *result
        .verts
        .iter()
        .find(|&&v| {
            let p = mesh.vertex_position(v);
            p.x > 0.1 && (p.y + 1.0).abs() < 1.0e-4
        })
        .expect("the ring vert on the b→c leg")
}

fn uv_of(mesh: &LoopMesh, layer: ElementAttributeKey, loop_key: LoopKey) -> [f32; 2] {
    mesh.loop_attribs()
        .get_vec2(layer, loop_key.as_usize())
        .expect("the UV layer is registered")
}

/// BEVEL.md §5.2.1.1 + §5.2.1.1.1: a boundary corner is seam-carrying (condition (1)), so
/// its cap takes the **single override donor face** (the tie-break winner, here the lower-
/// slot left quad) with the per-corner weight-snap — **not** the per-boundvert positional
/// donor. The `b → c` ring vert projects outside the left quad, so it snaps onto one of the
/// left quad's two edges at `b`; both nearest points are `b` itself, so the sample lands on
/// the corner and the blend reads the left quad's `b`-loop verbatim: `[0.5, 0]`. Two UV
/// islands here (the right quad's `u` is shifted), so the post-pass leaves the cap loop a
/// singleton and the snapped value stands. The positional donor (the right quad) would give
/// `[1.1, 0]`, so the two branches are cleanly distinguished.
#[test]
fn vertex_bevel_seam_carrying_cap_takes_override_donor_with_weight_snap() {
    let (mut mesh, corner, uv) = boundary_corner_uv_plane(
        |is_left, position| {
            let shift = if is_left { 0.0 } else { 0.5 };
            [(position.x + 2.0) / 4.0 + shift, (position.y + 1.0) / 2.0]
        },
        1,
        2,
    );

    let result = mesh.bevel(
        &[corner.into()],
        BevelOptions {
            offset: 0.4,
            ..Default::default()
        },
    );
    mesh.validate_topology();

    assert_eq!(result.faces.len(), 1, "one cap");
    let cap = result.faces[0];
    let ring_vertex = right_leg_ring_vert(&mesh, &result);
    let cap_loop = corner_of(&mesh, cap, ring_vertex);
    let value = uv_of(&mesh, uv, cap_loop);

    assert!(
        (value[0] - 0.5).abs() < 1.0e-5 && value[1].abs() < 1.0e-5,
        "seam-carrying cap corner took {value:?}, not the override donor's snapped [0.5, 0] \
         (the positional right-quad donor would give [1.1, 0])",
    );
}

/// BEVEL.md §5.2.6's worked value on the two-quad plane. Single UV island (continuous
/// `u = (x + 2) / 4`, `v = (y + 1) / 2`), left quad the lower-slot override donor. At the
/// `b → c` ring vert the cap loop weight-snaps onto the left quad's edge at `b` → `[0.5, 0]`,
/// while the re-stitched right pentagon blends affinely on its own `b → c` edge at
/// `t = 0.2` → `[0.6, 0]`. Same beveled vertex, single island ⇒ one bucket ⇒ mean
/// **`[0.55, 0]`** written back to *every* loop in the bucket.
#[test]
fn vertex_bevel_uv_merge_reconciles_cap_and_restitched_to_the_mean() {
    let (mut mesh, corner, uv) = boundary_corner_uv_plane(
        |_is_left, position| [(position.x + 2.0) / 4.0, (position.y + 1.0) / 2.0],
        1,
        2,
    );

    let result = mesh.bevel(
        &[corner.into()],
        BevelOptions {
            offset: 0.4,
            ..Default::default()
        },
    );
    mesh.validate_topology();

    let ring_vertex = right_leg_ring_vert(&mesh, &result);
    // Every loop at the ring vert — the cap loop and the re-stitched right-pentagon loop —
    // must carry the reconciled mean.
    let loops_here: Vec<LoopKey> = mesh
        .vert_face_keys(ring_vertex)
        .into_iter()
        .map(|face| corner_of(&mesh, face, ring_vertex))
        .collect();
    assert!(
        loops_here.len() >= 2,
        "the ring vert hosts a cap and a re-stitched loop"
    );
    for loop_key in loops_here {
        let value = uv_of(&mesh, uv, loop_key);
        assert!(
            (value[0] - 0.55).abs() < 1.0e-5 && value[1].abs() < 1.0e-5,
            "post-pass wrote {value:?}, not the bucket mean [0.55, 0]",
        );
    }
}

/// BEVEL.md §5.2.6's second worked value, `[1.5625, 2.140625]`, engineered on the same
/// boundary corner: the left quad's `b`-loop (the override donor, read at the snapped
/// corner) is `[1.5, 2.125]`, and the right quad's `b`/`c` loops are set so the re-stitched
/// affine blend at `t = 0.2` lands on `[1.625, 2.15625]`. Both quads agree at `b` and at
/// `e`, so the shared edge is UV-contiguous (single island) and the two loops bucket
/// together; the mean is `[1.5625, 2.140625]`.
#[test]
fn vertex_bevel_uv_merge_reconciles_to_the_grid_worked_value() {
    let (mut mesh, corner, uv) = boundary_corner_uv_plane(
        |_is_left, position| {
            // Keyed on vertex position so both quads agree at the shared verts b and e.
            match (position.x, position.y) {
                (x, y) if x == 0.0 && y == -1.0 => [1.5, 2.125], // b
                (x, y) if x == 0.0 && y == 1.0 => [1.5, 3.125],  // e (shared, agree)
                (x, y) if x == 2.0 && y == -1.0 => [2.125, 2.28125], // c
                _ => [0.0, 0.0],                                 // a, d, f (unused)
            }
        },
        1,
        2,
    );

    let result = mesh.bevel(
        &[corner.into()],
        BevelOptions {
            offset: 0.4,
            ..Default::default()
        },
    );
    mesh.validate_topology();

    let ring_vertex = right_leg_ring_vert(&mesh, &result);
    let cap = result.faces[0];
    let cap_loop = corner_of(&mesh, cap, ring_vertex);
    let value = uv_of(&mesh, uv, cap_loop);
    assert!(
        (value[0] - 1.5625).abs() < 1.0e-4 && (value[1] - 2.140625).abs() < 1.0e-4,
        "post-pass wrote {value:?}, not the bucket mean [1.5625, 2.140625]",
    );
}

/// The regression guard: a **non-seam-carrying** corner keeps §5.2.1.1's per-boundvert
/// positional donor, and the §5.2.6 post-pass must not corrupt it. A cube corner with a
/// *continuous* float2 layer (each face agrees with its neighbours at every shared vertex)
/// is seam-free — condition (1) cannot fire on a closed cube, condition (2) agrees on every
/// edge — so each cap corner blends edge-linearly from its own boundvert's arriving face.
/// The cap and re-stitched loops already agree there (continuous field), so the post-pass
/// averages equal values and changes nothing.
#[test]
fn vertex_bevel_non_seam_cap_keeps_positional_donor_after_post_pass() {
    let mut mesh = LoopMesh::new();
    mesh_add_cube(&mut mesh);
    let uv = mesh
        .register_loop_attribute(ElementAttributeDesc::uv(AttributeKey::custom("uv")))
        .expect("Vec2 is a supported loop attribute format");
    let field = |p: Vec3| [3.0 * p.x + 5.0 * p.y + 7.0, 2.0 * p.x - 4.0 * p.z + 9.0];
    for face in mesh.face_keys() {
        for loop_key in mesh.face_loop_keys(face) {
            let position = mesh.vertex_position(mesh.loop_vertex(loop_key));
            mesh.loop_attribs_mut()
                .set_vec2(uv, loop_key.as_usize(), field(position));
        }
    }

    let corner = vert_at(&mesh, Vec3::new(1.0, -1.0, -1.0));
    let corner_position = mesh.vertex_position(corner);
    let result = mesh.bevel(
        &[corner.into()],
        BevelOptions {
            offset: 0.25,
            ..Default::default()
        },
    );
    mesh.validate_topology();

    assert_eq!(result.faces.len(), 1, "one cap");
    let cap = result.faces[0];
    for cap_corner in mesh.face_loop_keys(cap) {
        let ring_vertex = mesh.loop_vertex(cap_corner);
        let position = mesh.vertex_position(ring_vertex);
        // A seam-free cap corner blends edge-linearly from its arriving face, which for a
        // continuous field reproduces the field at the ring vert's own position.
        let expected = field(position);
        let value = uv_of(&mesh, uv, cap_corner);
        assert!(
            (value[0] - expected[0]).abs() < 1.0e-3 && (value[1] - expected[1]).abs() < 1.0e-3,
            "non-seam cap corner took {value:?}, not the positional edge-linear \
             {expected:?} at {position:?}",
        );
        // ...and distinct from the beveled corner's own value, so a value that silently
        // collapsed to cd(v) would be caught.
        let at_corner = field(corner_position);
        assert!(
            (value[0] - at_corner[0]).abs() > 1.0e-4 || (value[1] - at_corner[1]).abs() > 1.0e-4,
            "the cap corner collapsed to the beveled vertex's value {at_corner:?}",
        );
    }
}

/// BEVEL.md §5.2.3 + §5.2.5: at an edge bevel the strip verts sit on the **legs**, so
/// the rebuilt face's corner at a strip vert blends that face's loops along the leg —
/// the beveled edge's own far endpoint appears in no such blend at that end.
#[test]
fn edge_bevel_rebuilt_face_corners_blend_the_loops_along_their_leg() {
    let (mut mesh, shade) = loop_decorated_cube();
    let near = vert_at(&mesh, Vec3::new(1.0, -1.0, -1.0));
    let far_end = vert_at(&mesh, Vec3::new(1.0, -1.0, 1.0));
    let beveled = mesh
        .query_edge_exists(near, far_end)
        .expect("the cube has this edge");
    let near_position = mesh.vertex_position(near);

    // The one face at `near` that the beveled edge does *not* bound: its two corners
    // at `near`'s legs are both re-stitched, and neither leg is the beveled edge.
    let flanking = mesh.edge_face_keys(beveled);
    let side_face = mesh
        .vert_face_keys(near)
        .into_iter()
        .find(|face| !flanking.contains(face))
        .expect("the cube corner has a third face");
    let side_code = face_code(mesh.face_plane_normal(side_face));
    let near_shade = shade_of(&mesh, shade, corner_of(&mesh, side_face, near));
    let legs: Vec<(Vec3, f32)> = mesh
        .face_vert_keys(side_face)
        .into_iter()
        .filter(|&w| w != near && mesh.query_edge_exists(near, w).is_some())
        .map(|far| {
            (
                mesh.vertex_position(far),
                shade_of(&mesh, shade, corner_of(&mesh, side_face, far)),
            )
        })
        .collect();
    assert_eq!(legs.len(), 2);

    mesh.bevel(
        &[beveled.into()],
        BevelOptions {
            offset: 0.25,
            affect: BevelAffect::Edges,
            ..Default::default()
        },
    );

    let rebuilt = face_with_normal(&mesh, side_code);
    let strip_corners: Vec<LoopKey> = mesh
        .face_loop_keys(rebuilt)
        .into_iter()
        .filter(|&loop_key| {
            let position = mesh.vertex_position(mesh.loop_vertex(loop_key));
            (position - near_position).length() > 1.0e-4
                && (position - near_position).length() < 0.5
        })
        .collect();
    assert_eq!(
        strip_corners.len(),
        2,
        "the face opposite the beveled edge holds both of `near`'s legs, so its corner \
         at `near` is replaced by both strip verts",
    );

    for strip_corner in strip_corners {
        let position = mesh.vertex_position(mesh.loop_vertex(strip_corner));
        let &(leg_far_position, leg_far_shade) = legs
            .iter()
            .min_by(|(a, _), (b, _)| {
                let da = (position - *a).length();
                let db = (position - *b).length();
                da.partial_cmp(&db).expect("finite")
            })
            .expect("the strip vert sits on one of the face's legs");

        let t = (position - near_position).length() / (leg_far_position - near_position).length();
        let expected = (1.0 - t) * near_shade + t * leg_far_shade;
        let actual = shade_of(&mesh, shade, strip_corner);
        assert!(
            (actual - expected).abs() < 1.0e-3,
            "strip corner shade {actual} != leg-linear {expected} (endpoint loop \
             {near_shade}, leg-far loop {leg_far_shade}, t {t})",
        );
    }

    mesh.validate_topology();
}

/// BEVEL.md §5.2.5 / §5.2.1.1 third bullet: the §7 single-segment strip quad owns its
/// four loops, so its per-corner blend decides them outright — nothing else writes
/// them, so an unblended corner reads the layer **default**. On a position-contiguous
/// layer every donor face reproduces `f(position)` exactly (§10.1.3's on-edge
/// short-circuit at each strip vert's leg), so the arriving value is the affine field at
/// the vert's own position — and the discriminating fact is that it is **not** the
/// default, which is what today's stub left there.
#[test]
fn edge_bevel_strip_quad_loops_blend_from_their_flank() {
    let (mut mesh, shade) = loop_continuous_cube();
    let near = vert_at(&mesh, Vec3::new(1.0, -1.0, -1.0));
    let far_end = vert_at(&mesh, Vec3::new(1.0, -1.0, 1.0));
    let beveled = mesh
        .query_edge_exists(near, far_end)
        .expect("the cube has this edge");

    let result = mesh.bevel(
        &[beveled.into()],
        BevelOptions {
            offset: 0.25,
            affect: BevelAffect::Edges,
            ..Default::default()
        },
    );
    assert_eq!(
        result.faces.len(),
        1,
        "a valence-3 isolated edge mints only the strip quad",
    );
    let strip = result.faces[0];
    assert_eq!(mesh.face_loop_keys(strip).len(), 4, "the strip is a quad");

    for corner in mesh.face_loop_keys(strip) {
        let position = mesh.vertex_position(mesh.loop_vertex(corner));
        let expected = continuous_loop_field(position);
        let actual = shade_of(&mesh, shade, corner);
        assert!(
            (actual - expected).abs() < 1.0e-3,
            "strip quad loop {actual} != the flank's blend {expected} at {position:?}",
        );
        assert!(
            actual.abs() > 1.0e-3,
            "the strip quad loop read the layer default — no per-loop blend ran on the \
             minted strip",
        );
    }

    mesh.validate_topology();
}

/// BEVEL.md §6.6.2's strip row + §5.2.3: the strip quad's per-loop donor list is per
/// corner, near/near/far/far — each strip vert blends **its own flank's** loops, the
/// flank whose leg it sits on. On the per-face-coded cube that flank's value carries the
/// face's own code, so the strip quad's loop at a strip vert must **agree** with the
/// re-stitched flank's loop that meets it across the shared chamfer edge, and disagree
/// with the other flank's code. A default reading (0) or a swapped near/far mapping fails
/// on a nameable code offset.
#[test]
fn edge_bevel_strip_quad_loops_agree_with_their_rebuilt_flank() {
    let (mut mesh, shade) = loop_decorated_cube();
    let near = vert_at(&mesh, Vec3::new(1.0, -1.0, -1.0));
    let far_end = vert_at(&mesh, Vec3::new(1.0, -1.0, 1.0));
    let beveled = mesh
        .query_edge_exists(near, far_end)
        .expect("the cube has this edge");
    // `E`'s two flanking faces, by plane code — they are re-stitched (keep their plane)
    // and are the near / far flank donors the strip quad's loops must track.
    let flank_codes: Vec<i32> = mesh
        .edge_face_keys(beveled)
        .into_iter()
        .map(|face| face_code(mesh.face_plane_normal(face)))
        .collect();
    assert_eq!(flank_codes.len(), 2);

    let result = mesh.bevel(
        &[beveled.into()],
        BevelOptions {
            offset: 0.25,
            affect: BevelAffect::Edges,
            ..Default::default()
        },
    );
    assert_eq!(result.faces.len(), 1);
    let strip = result.faces[0];

    for corner in mesh.face_loop_keys(strip) {
        let vertex = mesh.loop_vertex(corner);
        let position = mesh.vertex_position(vertex);
        // The flank this strip vert sits on is the one re-stitched face, among `E`'s two,
        // that still has a corner here (the vert is on that flank's leg only).
        let flank = flank_codes
            .iter()
            .copied()
            .find(|&code| {
                let face = face_with_normal(&mesh, code);
                mesh.face_loop_keys(face)
                    .into_iter()
                    .any(|other| mesh.loop_vertex(other) == vertex)
            })
            .expect("a strip vert lies on exactly one flank's leg");
        let flank_face = face_with_normal(&mesh, flank);
        let flank_corner = corner_of(&mesh, flank_face, vertex);
        let expected = shade_of(&mesh, shade, flank_corner);
        let actual = shade_of(&mesh, shade, corner);
        assert!(
            (actual - expected).abs() < 1.0e-3,
            "strip quad loop {actual} does not agree with its flank (code {flank}) loop \
             {expected} at {position:?} — a default or a swapped near/far donor",
        );
        let other_code = *flank_codes
            .iter()
            .find(|&&c| c != flank)
            .expect("two flanks");
        assert!(
            (actual - 100.0 * (other_code as f32 + 1.0)).abs() > 50.0,
            "strip quad loop {actual} carries the *other* flank's code {other_code} — \
             the near/far mapping is swapped",
        );
    }

    mesh.validate_topology();
}

// ---- BEVEL.md §7.8: inconsistently-wound flanks — the one-sided strip ----
