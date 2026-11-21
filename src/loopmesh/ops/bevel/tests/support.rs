//! Fixtures and assertions shared by more than one bevel test module.

use super::*;

pub(super) fn counts(mesh: &LoopMesh) -> (usize, usize, usize) {
    (mesh.vertex_count(), mesh.edge_count(), mesh.face_count())
}

/// A flat `spokes`-triangle fan around a centre vertex at the origin, rim at unit
/// radius: a **planar** junction, so every beveled-edge sector's `start` / `mid` /
/// `end` comes out collinear and every profile rides the §8.2.2.1 rescue.
pub(super) fn planar_fan(spokes: usize) -> (LoopMesh, VertexKey, Vec<VertexKey>) {
    let mut mesh = LoopMesh::new();
    let centre = mesh.create_vertex(Vec3::ZERO);
    let rim: Vec<VertexKey> = (0..spokes)
        .map(|k| {
            let angle = std::f32::consts::TAU * (k as f32) / (spokes as f32);
            mesh.create_vertex(Vec3::new(angle.cos(), angle.sin(), 0.0))
        })
        .collect();
    for k in 0..spokes {
        let next = (k + 1) % spokes;
        mesh.face_create_verts(&[centre, rim[k], rim[next]], FaceCreateFlags::NONE);
    }
    (mesh, centre, rim)
}

/// [`planar_fan`] rotated so that spoke `1` sits on `+x`: the frame §8.5.5.5(f.6)'s
/// worked planar-pipe case states its figures in. At `spokes = 12` the beveled set
/// `1 / 4 / 7 / 10` is then the four spokes at `0° / 90° / 180° / 270°`, so the pipe
/// axis and the clamped transverse direction are coordinate directions and the guide's
/// numbers can be read off a position directly.
pub(super) fn planar_fan_axis_aligned(spokes: usize) -> (LoopMesh, VertexKey, Vec<VertexKey>) {
    let mut mesh = LoopMesh::new();
    let centre = mesh.create_vertex(Vec3::ZERO);
    let rim: Vec<VertexKey> = (0..spokes)
        .map(|k| {
            let angle = std::f32::consts::TAU * (k as f32 - 1.0) / (spokes as f32);
            mesh.create_vertex(Vec3::new(angle.cos(), angle.sin(), 0.0))
        })
        .collect();
    for k in 0..spokes {
        let next = (k + 1) % spokes;
        mesh.face_create_verts(&[centre, rim[k], rim[next]], FaceCreateFlags::NONE);
    }
    (mesh, centre, rim)
}

/// A half-disc fan: `spokes` rim verts spread over 180°, so the centre is a
/// **boundary** vertex with `spokes - 1` faces and an *open* corner ring whose two
/// ends are the extreme spokes. Faces are created starting at `first_face` and wrapping,
/// which shifts which face the ring walk starts from — the same mesh either way.
pub(super) fn open_planar_fan(
    spokes: usize,
    first_face: usize,
) -> (LoopMesh, VertexKey, Vec<VertexKey>) {
    let mut mesh = LoopMesh::new();
    let centre = mesh.create_vertex(Vec3::ZERO);
    let rim: Vec<VertexKey> = (0..spokes)
        .map(|k| {
            let angle = std::f32::consts::PI * (k as f32) / ((spokes - 1) as f32);
            mesh.create_vertex(Vec3::new(angle.cos(), angle.sin(), 0.0))
        })
        .collect();
    let faces = spokes - 1;
    for step in 0..faces {
        let k = (first_face + step) % faces;
        mesh.face_create_verts(&[centre, rim[k], rim[k + 1]], FaceCreateFlags::NONE);
    }
    (mesh, centre, rim)
}

/// A closed `base_count`-gon pyramid: an apex at `(0, 0, 1)`, a regular base
/// polygon of unit radius in the `z = 0` plane, `base_count` triangular side faces
/// forming a **closed fan** at the apex (disk valence `base_count`), and one base
/// n-gon closing the solid. A genuinely non-flat general-valence corner for the
/// §8.11 vertex-mode multiseg cap — unlike `planar_fan`, the fan bulges out of the
/// ring plane. Returns the mesh and the apex vertex.
pub(super) fn closed_pyramid(base_count: usize) -> (LoopMesh, VertexKey) {
    let mut mesh = LoopMesh::new();
    let apex = mesh.create_vertex(Vec3::new(0.0, 0.0, 1.0));
    let base: Vec<VertexKey> = (0..base_count)
        .map(|k| {
            let angle = std::f32::consts::TAU * (k as f32) / (base_count as f32);
            mesh.create_vertex(Vec3::new(angle.cos(), angle.sin(), 0.0))
        })
        .collect();
    for k in 0..base_count {
        let next = (k + 1) % base_count;
        mesh.face_create_verts(&[apex, base[k], base[next]], FaceCreateFlags::NONE);
    }
    // Base n-gon wound opposite the sides so every shared edge is traversed both
    // ways — the solid comes out outward-consistent.
    let base_ring: Vec<VertexKey> = base.iter().copied().rev().collect();
    mesh.face_create_verts(&base_ring, FaceCreateFlags::NONE);
    (mesh, apex)
}

pub(super) fn vert_at(mesh: &LoopMesh, position: Vec3) -> VertexKey {
    mesh.vert_keys()
        .into_iter()
        .find(|&v| mesh.vertex_position(v) == position)
        .expect("vertex at position exists")
}

/// A ring vert is "at" `position` within tolerance; returns whether any
/// new vert from the result landed there.
pub(super) fn result_has_vert_near(mesh: &LoopMesh, result: &BevelResult, position: Vec3) -> usize {
    result
        .verts
        .iter()
        .filter(|&&v| (mesh.vertex_position(v) - position).length() < 1.0e-5)
        .count()
}

/// True when two meshes carry the same set of vertex positions (as a
/// multiset), matched greedily within `tol`. Position-keyed rather than
/// key-keyed because two independent builds may assign different internal
/// element keys while producing identical geometry.
pub(super) fn positions_multiset_eq(a: &LoopMesh, b: &LoopMesh, tol: f32) -> bool {
    let positions_a: Vec<Vec3> = a
        .vert_keys()
        .iter()
        .map(|&v| a.vertex_position(v))
        .collect();
    let mut positions_b: Vec<Vec3> = b
        .vert_keys()
        .iter()
        .map(|&v| b.vertex_position(v))
        .collect();
    if positions_a.len() != positions_b.len() {
        return false;
    }
    for target in &positions_a {
        match positions_b
            .iter()
            .position(|q| (*q - *target).length() < tol)
        {
            Some(i) => {
                positions_b.swap_remove(i);
            }
            None => return false,
        }
    }
    true
}

/// Two position cycles are equal up to rotation and reflection (a face is a
/// cycle, so both are free), matched pointwise within `tol`.
pub(super) fn cycle_matches(a: &[Vec3], b: &[Vec3], tol: f32) -> bool {
    let n = a.len();
    if b.len() != n || n == 0 {
        return b.len() == n;
    }
    for &reversed in &[false, true] {
        let candidate: Vec<Vec3> = if reversed {
            b.iter().rev().copied().collect()
        } else {
            b.to_vec()
        };
        for start in 0..n {
            if (0..n).all(|i| (a[i] - candidate[(start + i) % n]).length() < tol) {
                return true;
            }
        }
    }
    false
}

/// True when two meshes carry the same set of faces (as a multiset), each
/// face compared as its position cycle up to rotation/reflection within
/// `tol`. Discriminates real geometry, not just per-face vertex counts.
pub(super) fn faces_match(a: &LoopMesh, b: &LoopMesh, tol: f32) -> bool {
    let face_cycle = |mesh: &LoopMesh, f: FaceKey| -> Vec<Vec3> {
        mesh.face_vert_keys(f)
            .iter()
            .map(|&v| mesh.vertex_position(v))
            .collect()
    };
    let faces_a: Vec<Vec<Vec3>> = a.face_keys().iter().map(|&f| face_cycle(a, f)).collect();
    let mut faces_b: Vec<Vec<Vec3>> = b.face_keys().iter().map(|&f| face_cycle(b, f)).collect();
    if faces_a.len() != faces_b.len() {
        return false;
    }
    for face in &faces_a {
        match faces_b
            .iter()
            .position(|other| cycle_matches(face, other, tol))
        {
            Some(i) => {
                faces_b.swap_remove(i);
            }
            None => return false,
        }
    }
    true
}

/// BEVEL.md §5.1: every interior manifold edge must be traversed in
/// *opposite* directions by its two adjacent faces. Returns the directed
/// vertex pairs that two faces traverse the same way (a winding conflict) —
/// empty for a globally winding-consistent mesh.
pub(super) fn winding_conflicts(mesh: &LoopMesh) -> Vec<(VertexKey, VertexKey)> {
    let mut directed: Vec<(VertexKey, VertexKey)> = Vec::new();
    for &face in mesh.face_keys().iter() {
        let verts = mesh.face_vert_keys(face);
        for index in 0..verts.len() {
            directed.push((verts[index], verts[(index + 1) % verts.len()]));
        }
    }
    let mut conflicts: Vec<(VertexKey, VertexKey)> = Vec::new();
    for (index, &pair) in directed.iter().enumerate() {
        if directed[index + 1..].contains(&pair) {
            conflicts.push(pair);
        }
    }
    conflicts
}

/// A unit octahedron (6 verts, 8 triangles), every vertex valence 4 and every
/// face wound outward — the closed solid the valence-≥4 junction / terminal-fan
/// paths need.
pub(super) fn mesh_add_octahedron(mesh: &mut LoopMesh) {
    let px = mesh.create_vertex(Vec3::new(1.0, 0.0, 0.0));
    let py = mesh.create_vertex(Vec3::new(0.0, 1.0, 0.0));
    let nx = mesh.create_vertex(Vec3::new(-1.0, 0.0, 0.0));
    let ny = mesh.create_vertex(Vec3::new(0.0, -1.0, 0.0));
    let apex = mesh.create_vertex(Vec3::new(0.0, 0.0, 1.0));
    let nadir = mesh.create_vertex(Vec3::new(0.0, 0.0, -1.0));
    for ring in [[px, py], [py, nx], [nx, ny], [ny, px]] {
        mesh.face_create_verts(&[ring[0], ring[1], apex], FaceCreateFlags::NONE);
        mesh.face_create_verts(&[ring[1], ring[0], nadir], FaceCreateFlags::NONE);
    }
}

/// Signed volume of a closed mesh, as the divergence-theorem fan sum over its
/// faces. Positive exactly when the faces wind outward, so it catches a result
/// that is winding-*consistent* yet globally inside out.
pub(super) fn signed_volume(mesh: &LoopMesh) -> f32 {
    let mut total = 0.0;
    for &face in mesh.face_keys().iter() {
        let verts = mesh.face_vert_keys(face);
        let apex = mesh.vertex_position(verts[0]);
        for index in 1..verts.len().saturating_sub(1) {
            let b = mesh.vertex_position(verts[index]);
            let c = mesh.vertex_position(verts[index + 1]);
            total += apex.dot(b.cross(c)) / 6.0;
        }
    }
    total
}

/// A beveled closed solid must stay a valid, non-degenerate, winding-consistent,
/// outward-wound mesh (BEVEL.md §5.1).
pub(super) fn assert_bevel_sound(mesh: &LoopMesh, result: &BevelResult, case: &str) {
    mesh.validate_topology();
    assert!(!result.faces.is_empty(), "{case}: bevel produced no faces");
    let conflicts = winding_conflicts(mesh);
    assert!(
        conflicts.is_empty(),
        "{case}: {} shared edge(s) traversed the same way by both adjacent faces",
        conflicts.len(),
    );
    let volume = signed_volume(mesh);
    assert!(
        volume > 0.0,
        "{case}: signed volume {volume} — the solid is consistent but inside out",
    );
}

/// Bevel the given edges of a cube corner, sweeping segments / vmesh method /
/// profile, and assert every result is sound. `edge_count` names how many of the
/// corner's three incident edges to bevel: 3 = every strip end is a junction,
/// 2 = each strip has one junction end and one terminal end, 1 = both ends
/// terminal.
pub(super) fn assert_cube_corner_bevel_sound(
    edge_count: usize,
    segment_range: std::ops::RangeInclusive<u32>,
    methods: &[BevelVmeshMethod],
) {
    for segments in segment_range {
        for &vmesh_method in methods {
            for profile in [0.5f32, 0.25, 0.85] {
                let mut mesh = LoopMesh::new();
                mesh_add_cube(&mut mesh);
                let corner = vert_at(&mesh, Vec3::new(1.0, -1.0, -1.0));
                let incident = mesh.vert_edge_keys(corner);
                assert_eq!(
                    incident.len(),
                    3,
                    "the cube corner has three incident edges"
                );
                let geom: Vec<GeomKey> = incident
                    .into_iter()
                    .take(edge_count)
                    .map(GeomKey::from)
                    .collect();

                let result = mesh.bevel(
                    &geom,
                    BevelOptions {
                        offset: 0.25,
                        affect: BevelAffect::Edges,
                        segments,
                        profile,
                        vmesh_method,
                        ..Default::default()
                    },
                );

                let case = format!(
                    "{edge_count} edge(s), segments = {segments}, profile = {profile}, \
                     {vmesh_method:?}"
                );
                assert_bevel_sound(&mesh, &result, &case);
                assert!(
                    result.faces.len() >= edge_count * segments as usize,
                    "{case}: {} new face(s) — expected at least one strip quad per \
                     segment per beveled edge",
                    result.faces.len(),
                );
            }
        }
    }
}

pub(super) const TINT: &str = "tint";

/// Register the per-edge `tint` layer and give **every** edge of `mesh` a distinct
/// value, so any donor mis-pick shows up as a wrong colour rather than a coincidence.
/// The tint of edge `e` is `(k+1, 0, 0)` for the `k`-th edge; the layer default
/// `(0, 0, 0)` is therefore carried by no original at all and marks "inherited
/// nothing".
pub(super) fn tint_every_edge(mesh: &mut LoopMesh) -> ElementAttributeKey {
    let layer = mesh
        .register_edge_attribute(ElementAttributeDesc::numeric(
            AttributeKey::custom(TINT),
            AttrType::Vec3,
        ))
        .expect("Vec3 is a supported edge attribute format");
    for (index, edge) in mesh.edge_keys().into_iter().enumerate() {
        mesh.edge_attribs_mut()
            .set_vec3(layer, edge.as_usize(), [index as f32 + 1.0, 0.0, 0.0]);
    }
    layer
}

pub(super) fn tint_of(mesh: &LoopMesh, layer: ElementAttributeKey, edge: EdgeKey) -> [f32; 3] {
    mesh.edge_attribs()
        .get_vec3(layer, edge.as_usize())
        .unwrap_or([0.0, 0.0, 0.0])
}

/// The **winding-outgoing** edge of `face` at `vertex` (BEVEL.md §7.6.2's disk-walk
/// sense): the edge `face`'s corner at `vertex` *leaves* along. Re-derived here from
/// the face's own vertex cycle, independently of the operator's own corner reads, so
/// a test asserting the donor is not merely confirming the implementation to itself.
pub(super) fn winding_outgoing(mesh: &LoopMesh, face: FaceKey, vertex: VertexKey) -> EdgeKey {
    let cycle = mesh.face_vert_keys(face);
    let index = cycle
        .iter()
        .position(|&w| w == vertex)
        .expect("the face has this corner");
    let next = cycle[(index + 1) % cycle.len()];
    mesh.query_edge_exists(vertex, next)
        .expect("a face corner's outgoing edge exists")
}

/// The predecessor of `edge` in `vertex`'s disk **walk** (BEVEL.md §7.6.2): the walk
/// reaches `edge` by crossing the face whose corner at `vertex` *arrives* along it, so
/// the predecessor is that face's winding-outgoing edge. Re-derived from the face
/// cycles, as above.
pub(super) fn walk_predecessor(mesh: &LoopMesh, vertex: VertexKey, edge: EdgeKey) -> EdgeKey {
    for face in mesh.vert_face_keys(vertex) {
        let cycle = mesh.face_vert_keys(face);
        let index = cycle
            .iter()
            .position(|&w| w == vertex)
            .expect("an incident face has this corner");
        let previous = cycle[(index + cycle.len() - 1) % cycle.len()];
        let incoming = mesh
            .query_edge_exists(previous, vertex)
            .expect("a face corner's incoming edge exists");
        if incoming == edge {
            return winding_outgoing(mesh, face, vertex);
        }
    }
    panic!("no incident face arrives at {vertex:?} along {edge:?}");
}

/// Two quads sharing one edge. The shared edge's two endpoints are valence-3 vertices
/// **on the mesh boundary**: their legs are radial-1 boundary edges and there is no
/// third face to absorb the corner — §7.7.3's boundary-terminal negative.
pub(super) fn two_quad_strip() -> (LoopMesh, EdgeKey) {
    let mut mesh = LoopMesh::new();
    let corner = |x: f32, y: f32| Vec3::new(x, y, 0.0);
    let a = mesh.create_vertex(corner(-2.0, -1.0));
    let b = mesh.create_vertex(corner(0.0, -1.0));
    let c = mesh.create_vertex(corner(2.0, -1.0));
    let d = mesh.create_vertex(corner(2.0, 1.0));
    let e = mesh.create_vertex(corner(0.0, 1.0));
    let f = mesh.create_vertex(corner(-2.0, 1.0));
    mesh.face_create_verts(&[a, b, e, f], FaceCreateFlags::NONE);
    mesh.face_create_verts(&[b, c, d, e], FaceCreateFlags::NONE);
    let shared = mesh.query_edge_exists(b, e).expect("the shared edge");
    (mesh, shared)
}

/// The §8.3.0.3 cross-weld fixture: bevel two **opposite** edges at the octahedron's
/// valence-4 apex (`+x` and `−x`), leaving the `±y` edges as the crossing legs.
/// `seam_both` flags both legs; otherwise only the first is flagged — the case the
/// reconciliation has to clear.
pub(super) fn bevel_octahedron_cross(
    segments: u32,
    vmesh_method: BevelVmeshMethod,
    both_legs: bool,
    flag_legs: bool,
) -> (LoopMesh, BevelResult) {
    let mut mesh = LoopMesh::new();
    mesh_add_octahedron(&mut mesh);
    let apex = vert_at(&mesh, Vec3::new(0.0, 0.0, 1.0));
    let edge_to = |mesh: &LoopMesh, position: Vec3| {
        mesh.query_edge_exists(apex, vert_at(mesh, position))
            .expect("the apex's edge")
    };
    if flag_legs {
        let first_leg = edge_to(&mesh, Vec3::new(0.0, 1.0, 0.0));
        let second_leg = edge_to(&mesh, Vec3::new(0.0, -1.0, 0.0));
        mesh.set_edge_seam(first_leg, true);
        mesh.set_edge_sharp(first_leg, true);
        mesh.set_edge_seam(second_leg, both_legs);
        mesh.set_edge_sharp(second_leg, both_legs);
    }
    let geom: Vec<GeomKey> = [Vec3::new(1.0, 0.0, 0.0), Vec3::new(-1.0, 0.0, 0.0)]
        .into_iter()
        .map(|position| GeomKey::from(edge_to(&mesh, position)))
        .collect();
    let result = mesh.bevel(
        &geom,
        BevelOptions {
            offset: 0.25,
            affect: BevelAffect::Edges,
            segments,
            vmesh_method,
            ..Default::default()
        },
    );
    (mesh, result)
}

/// The cube edge between the verts at `pa` and `pb`.
pub(super) fn cube_edge(mesh: &LoopMesh, pa: Vec3, pb: Vec3) -> EdgeKey {
    let a = vert_at(mesh, pa);
    let b = vert_at(mesh, pb);
    mesh.query_edge_exists(a, b).expect("cube edge exists")
}

/// An irregular tetrahedron: a manifold solid whose every vertex is a
/// **non-planar valence-3 terminal** with oblique legs. Beveling one edge
/// exercises the §7.5.1 terminal placement away from the planar/square cube
/// special case the other edge-bevel tests pin.
pub(super) fn irregular_tetra(mesh: &mut LoopMesh) -> [VertexKey; 4] {
    let v0 = mesh.create_vertex(Vec3::new(0.0, 0.0, 0.0));
    let v1 = mesh.create_vertex(Vec3::new(3.0, 0.0, 0.0));
    let v2 = mesh.create_vertex(Vec3::new(0.5, 2.5, 0.3));
    let v3 = mesh.create_vertex(Vec3::new(0.4, 0.6, 2.2));
    // Outward-wound faces (each triangle's normal points away from the centroid).
    let faces = [[v0, v2, v1], [v0, v1, v3], [v0, v3, v2], [v1, v2, v3]];
    for ring in faces {
        let edges = [
            mesh.create_edge([ring[0], ring[1]], EdgeCreateFlags::NO_DUPLICATE),
            mesh.create_edge([ring[1], ring[2]], EdgeCreateFlags::NO_DUPLICATE),
            mesh.create_edge([ring[2], ring[0]], EdgeCreateFlags::NO_DUPLICATE),
        ];
        mesh.face_create(&ring, &edges, FaceCreateFlags::NONE);
    }
    [v0, v1, v2, v3]
}

/// Build the cube-corner `s = 3` cutoff junction at `segments = 2` and return
/// the mesh + result for the cutoff tests below (BEVEL.md §8.2 / §8.3). Vertex
/// 0 = (1,-1,-1) is the junction; its three edges to 1/3/4 are beveled.
pub(super) fn cube_corner_cutoff_segments2() -> (LoopMesh, BevelResult) {
    let mut mesh = LoopMesh::new();
    mesh_add_cube(&mut mesh);
    let e01 = cube_edge(
        &mesh,
        Vec3::new(1.0, -1.0, -1.0),
        Vec3::new(-1.0, -1.0, -1.0),
    );
    let e03 = cube_edge(&mesh, Vec3::new(1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, -1.0));
    let e04 = cube_edge(&mesh, Vec3::new(1.0, -1.0, -1.0), Vec3::new(1.0, -1.0, 1.0));

    let result = mesh.bevel(
        &[e01.into(), e03.into(), e04.into()],
        BevelOptions {
            offset: 0.25,
            affect: BevelAffect::Edges,
            segments: 2,
            vmesh_method: BevelVmeshMethod::Cutoff,
            ..Default::default()
        },
    );
    (mesh, result)
}

/// The same cube-corner `s = 3` cutoff junction at `segments = 2` but with an
/// explicit `profile` (BEVEL.md §8.6), for the profile-sampling tests. Returns
/// the lone Front↔Bottom interior chain vert's position across edge {0,1}.
pub(super) fn cube_corner_cutoff_segments2_profile(profile: f32) -> Vec3 {
    let mut mesh = LoopMesh::new();
    mesh_add_cube(&mut mesh);
    let e01 = cube_edge(
        &mesh,
        Vec3::new(1.0, -1.0, -1.0),
        Vec3::new(-1.0, -1.0, -1.0),
    );
    let e03 = cube_edge(&mesh, Vec3::new(1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, -1.0));
    let e04 = cube_edge(&mesh, Vec3::new(1.0, -1.0, -1.0), Vec3::new(1.0, -1.0, 1.0));

    let result = mesh.bevel(
        &[e01.into(), e03.into(), e04.into()],
        BevelOptions {
            offset: 0.25,
            affect: BevelAffect::Edges,
            segments: 2,
            vmesh_method: BevelVmeshMethod::Cutoff,
            profile,
            ..Default::default()
        },
    );

    // The Front↔Bottom interior chain vert sits on edge {0,1}'s profile cell:
    // u = 0.75 (the boundvert plane), v sweeping the bulge toward (0.75,-0.75,
    // -0.75). It is the one new vert at x = 0.75 with y == z below the arc
    // center, i.e. the interior sample of that sector's chain.
    let interior: Vec<Vec3> = result
        .verts
        .iter()
        .map(|&v| mesh.vertex_position(v))
        .filter(|p| {
            (p.x - 0.75).abs() < 1.0e-4 && (p.y - p.z).abs() < 1.0e-4 && p.y < -0.8 && p.y > -1.0
        })
        .collect();
    assert_eq!(
        interior.len(),
        1,
        "exactly one Front-Bottom interior chain vert for profile {profile}",
    );
    interior[0]
}

/// Bevel a chosen set of a flat 12-spoke fan's spokes with the CUTOFF method
/// (BEVEL.md §8.3.4.1 fixtures). The centre is the multi-edge junction; each beveled
/// spoke's rim end is a valence-3 terminal. Spokes `k` and `k + 6` are collinear
/// through the centre, so beveling an opposite pair plants the anti-parallel pipe.
pub(super) fn bevel_flat_fan_cutoff(
    beveled_spokes: &[usize],
    offset: f32,
    segments: u32,
) -> (LoopMesh, BevelResult) {
    let (mut mesh, centre, rim) = planar_fan(12);
    let geom: Vec<GeomKey> = beveled_spokes
        .iter()
        .map(|&k| {
            mesh.query_edge_exists(centre, rim[k])
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

/// Face-side counts of the whole mesh, keyed by corner count (3 = tri, 4 = quad, …).
pub(super) fn face_side_histogram(mesh: &LoopMesh) -> std::collections::BTreeMap<usize, usize> {
    let mut histogram = std::collections::BTreeMap::new();
    for face in mesh.face_keys() {
        *histogram.entry(mesh.face_len(face)).or_insert(0usize) += 1;
    }
    histogram
}

/// The number of **loose, valence-0** verts in the mesh: in no edge and no face.
/// The §8.3.4.1 pipe override leaves exactly `(segments − 1) · s` of these.
pub(super) fn loose_vertex_count(mesh: &LoopMesh) -> usize {
    mesh.vert_keys()
        .into_iter()
        .filter(|&v| mesh.iter_disk_edges(v).next().is_none())
        .count()
}

/// The number of **connected** verts (in at least one edge) that sit — within the
/// coincident-vertex tolerance — on top of a loose valence-0 leftover's position.
/// BEVEL.md §8.3.4.1: the loose leftovers carry the **raw** cutoff-arc samples, while
/// the connected fill must be byte-identical to a from-scratch ADJ bevel — its rim on
/// the **resampled** arc. On a **skewed** pipe cell raw and resampled diverge, so the
/// correct override lands **zero** connected verts on a loose sample. A regression that
/// fed the raw cutoff chain into the connected fill would drop every rim node back onto
/// its own raw loose sample and drive this count up — which is exactly what the skewed
/// `s = 3` grid and the `s = 4` fan at `segments ≥ 3` catch.
pub(super) fn connected_verts_on_loose_sample(mesh: &LoopMesh) -> usize {
    let loose: Vec<Vec3> = mesh
        .vert_keys()
        .into_iter()
        .filter(|&v| mesh.iter_disk_edges(v).next().is_none())
        .map(|v| mesh.vertex_position(v))
        .collect();
    mesh.vert_keys()
        .into_iter()
        .filter(|&v| mesh.iter_disk_edges(v).next().is_some())
        .filter(|&v| {
            let position = mesh.vertex_position(v);
            loose
                .iter()
                .any(|&sample| (position - sample).length() < 1.0e-4)
        })
        .count()
}

/// [`pipe_cross_section_replant`] with the junction's placement resolved off the cell and
/// the axis, exactly as [`LoopMesh::pipe_cross_section`] resolves it once per junction
/// before walking the domain.
pub(super) fn replant(base: Vec3, pipe_axis: Vec3, cell: &ProfileCell, exponent: f32) -> Vec3 {
    pipe_cross_section_replant(
        base,
        pipe_axis,
        cell,
        pipe_station_placement(pipe_axis, cell),
        exponent,
        false,
    )
}

/// Build the cube-corner `s = 3` ADJ junction at `segments = 2` (BEVEL.md §8.5).
/// Vertex 0 = (1,-1,-1) is the junction; its three edges to 1/3/4 are beveled
/// with the default rounded adjacency-grid fill.
pub(super) fn cube_corner_adj_segments2() -> (LoopMesh, BevelResult) {
    let mut mesh = LoopMesh::new();
    mesh_add_cube(&mut mesh);
    let e01 = cube_edge(
        &mesh,
        Vec3::new(1.0, -1.0, -1.0),
        Vec3::new(-1.0, -1.0, -1.0),
    );
    let e03 = cube_edge(&mesh, Vec3::new(1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, -1.0));
    let e04 = cube_edge(&mesh, Vec3::new(1.0, -1.0, -1.0), Vec3::new(1.0, -1.0, 1.0));

    let result = mesh.bevel(
        &[e01.into(), e03.into(), e04.into()],
        BevelOptions {
            offset: 0.25,
            affect: BevelAffect::Edges,
            segments: 2,
            vmesh_method: BevelVmeshMethod::Adj,
            ..Default::default()
        },
    );
    (mesh, result)
}

/// A closed manifold whose interior apex `v` has a **single-face reflex sector**
/// (BEVEL.md §8.9.1): three beveled apex edges (`v-A`, `v-B`, `v-C`) bunched to one
/// side, leaving a comfortably-wide (`> 180°`) reflex gap spanned by the single quad
/// face `(v, C, W, A)`. The apex is lifted off the base plane so the fan genuinely
/// folds in 3-D (the signed reflex test needs a real vertex-normal frame, §8.9.1),
/// and the two narrow top sectors (`(v, A, B)`, `(v, B, C)`) stay non-reflex.
/// Returns `(mesh, apex, [v-A, v-B, v-C])`.
/// A closed octagonal bipyramid (BEVEL.md §8.9.2.1 worked driver): top apex `A`
/// at `(0, 0, 1)`, bottom apex `B` at `(0, 0, -1)`, rim octagon `R1 … R8` at
/// radius 1 in `z = 0` (`R_k` at `45·(k−1)°`). Beveling the three apex legs
/// `A–R1`, `A–R2`, `A–R3` bunches the junction to one side, so the `270°` reflex
/// gap `R3 → R4 → … → R8 → R1` is a **multi-face** sector absorbing the five
/// unbeveled legs `A–R4 … A–R8`. Returns `(mesh, apex A, rim [R1 … R8],
/// [A-R1, A-R2, A-R3])`.
pub(super) fn octagonal_bipyramid() -> (LoopMesh, VertexKey, [VertexKey; 8], [EdgeKey; 3]) {
    let (mesh, apex, rim, apex_leg) = octagonal_bipyramid_all_legs();
    (mesh, apex, rim, [apex_leg[0], apex_leg[1], apex_leg[2]])
}

/// The same octagonal bipyramid, exposing **all eight** apex legs so a caller can
/// bevel any prefix (e.g. four legs for the even-absorbed-leg `ARC` cross-check of
/// §8.9.2.2). Returns `(mesh, apex A, rim [R1 … R8], apex legs [A-R1 … A-R8])`.
pub(super) fn octagonal_bipyramid_all_legs() -> (LoopMesh, VertexKey, [VertexKey; 8], [EdgeKey; 8])
{
    let mut mesh = LoopMesh::new();
    let apex = mesh.create_vertex(Vec3::new(0.0, 0.0, 1.0));
    let base = mesh.create_vertex(Vec3::new(0.0, 0.0, -1.0));
    let rim: Vec<VertexKey> = (0..8)
        .map(|k| {
            let angle = std::f32::consts::FRAC_PI_4 * k as f32;
            mesh.create_vertex(Vec3::new(angle.cos(), angle.sin(), 0.0))
        })
        .collect();

    let apex_leg: Vec<EdgeKey> = (0..8)
        .map(|k| mesh.create_edge([apex, rim[k]], EdgeCreateFlags::NONE))
        .collect();
    let base_leg: Vec<EdgeKey> = (0..8)
        .map(|k| mesh.create_edge([base, rim[k]], EdgeCreateFlags::NONE))
        .collect();
    let rim_edge: Vec<EdgeKey> = (0..8)
        .map(|k| mesh.create_edge([rim[k], rim[(k + 1) % 8]], EdgeCreateFlags::NONE))
        .collect();

    for k in 0..8 {
        let next = (k + 1) % 8;
        mesh.face_create(
            &[apex, rim[k], rim[next]],
            &[apex_leg[k], rim_edge[k], apex_leg[next]],
            FaceCreateFlags::NONE,
        );
        mesh.face_create(
            &[base, rim[next], rim[k]],
            &[base_leg[next], rim_edge[k], base_leg[k]],
            FaceCreateFlags::NONE,
        );
    }

    let rim_array: [VertexKey; 8] = rim.try_into().expect("eight rim verts");
    let leg_array: [EdgeKey; 8] = apex_leg.try_into().expect("eight apex legs");
    (mesh, apex, rim_array, leg_array)
}

/// Re-derive a reflex sector's three `segments = 2` `ARC` sample positions on the
/// pristine octagonal bipyramid, independently of the operator: the two arc ends
/// `s_0 = B1` / `s_2 = B3` (the §8.9.2 neighbour-plane reposition of the sector's
/// meet-point along its two flanking edges) and the interior bulge `s_1` (pulled in
/// `√2 − 1` from the chord midpoint toward the stored meet corner, §8.9.5). `beveled`
/// names which apex-leg indices are beveled. Returns `(s_0, s_1, s_2)`.
pub(super) fn octagonal_arc_samples(beveled: &[usize]) -> (Vec3, Vec3, Vec3) {
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
    assert_eq!(
        reflex.iter().filter(|&&x| x).count(),
        1,
        "the bunched octagonal bevel presents exactly one reflex sector",
    );
    let r = reflex.iter().position(|&x| x).unwrap();
    let midpoint = meets[r];
    let predecessor = meets[(r + sector_count - 1) % sector_count];
    let successor = meets[(r + 1) % sector_count];
    let b1 = midpoint + (predecessor - midpoint).dot(dir_open[r]) * dir_open[r];
    let b3 = midpoint + (successor - midpoint).dot(dir_close[r]) * dir_close[r];
    let sqrt2 = std::f32::consts::SQRT_2;
    let bulge = (2.0 - sqrt2) * (0.5 * (b1 + b3)) + (sqrt2 - 1.0) * midpoint;
    (b1, bulge, b3)
}

pub(super) fn single_face_reflex_junction() -> (LoopMesh, VertexKey, [EdgeKey; 3]) {
    let mut mesh = LoopMesh::new();
    let v = mesh.create_vertex(Vec3::new(0.0, 0.0, 1.0));
    let a = mesh.create_vertex(Vec3::new(-1.0, 1.0, 0.0));
    let b = mesh.create_vertex(Vec3::new(0.0, 1.4, 0.0));
    let c = mesh.create_vertex(Vec3::new(1.0, 1.0, 0.0));
    let w = mesh.create_vertex(Vec3::new(0.0, -1.5, 0.0));

    let edge = |mesh: &mut LoopMesh, p: VertexKey, q: VertexKey| {
        mesh.create_edge([p, q], EdgeCreateFlags::NONE)
    };
    let va = edge(&mut mesh, v, a);
    let vb = edge(&mut mesh, v, b);
    let vc = edge(&mut mesh, v, c);
    let ab = edge(&mut mesh, a, b);
    let bc = edge(&mut mesh, b, c);
    let cw = edge(&mut mesh, c, w);
    let wa = edge(&mut mesh, w, a);

    mesh.face_create(&[v, a, b], &[va, ab, vb], FaceCreateFlags::NONE);
    mesh.face_create(&[v, b, c], &[vb, bc, vc], FaceCreateFlags::NONE);
    mesh.face_create(&[v, c, w, a], &[vc, cw, wa, va], FaceCreateFlags::NONE);
    mesh.face_create(&[a, w, c, b], &[wa, cw, bc, ab], FaceCreateFlags::NONE);

    (mesh, v, [va, vb, vc])
}

/// Bevel the three apex edges of `single_face_reflex_junction` with the given outer
/// miter, `segments = 2`, default circular profile, `vmesh_method = ADJ`.
pub(super) fn reflex_apex_outer_miter(kind: BevelMiter) -> (LoopMesh, BevelResult) {
    reflex_apex_outer_miter_segments(kind, 2)
}

/// Bevel the three apex edges of `single_face_reflex_junction` at an arbitrary
/// `segments` with the given outer miter, `offset = 0.2`, default circular profile,
/// `vmesh_method = ADJ`, `miter_inner = Sharp`, `spread = 0`.
pub(super) fn reflex_apex_outer_miter_segments(
    kind: BevelMiter,
    segments: u32,
) -> (LoopMesh, BevelResult) {
    let (mut mesh, _apex, [va, vb, vc]) = single_face_reflex_junction();
    let result = mesh.bevel(
        &[va.into(), vb.into(), vc.into()],
        BevelOptions {
            offset: 0.2,
            affect: BevelAffect::Edges,
            segments,
            vmesh_method: BevelVmeshMethod::Adj,
            miter_outer: kind,
            ..Default::default()
        },
    );
    (mesh, result)
}

/// The **emitted** arc chain of BEVEL.md §8.9.5.1: the raw chain at the run's own
/// §8.6.4 spacing, pushed through the §8.2.2.1.1 chord-fraction resample against the
/// profile materialised at the source resolution — exactly what the outer-miter
/// builder does with it.
pub(super) fn emitted_arc_chain(
    b1: Vec3,
    b3: Vec3,
    midpoint: Vec3,
    profile: f32,
    segments: u32,
) -> Vec<Vec3> {
    let arc = RingChainProfile::Arc {
        b1,
        b3,
        midpoint,
        profile,
    };
    let raw = arc.raw_positions(segments);
    let source_segments = boundary_source_resolution(segments);
    if source_segments == segments {
        raw
    } else {
        let source = arc.raw_positions(source_segments);
        chord_fraction_resample(&raw, &source)
    }
}

/// Bevel the same symmetric equal-offset cube corner with `segments` segments,
/// default circular profile, `vmesh_method = ADJ` (BEVEL.md §8.5.5).
pub(super) fn cube_corner_adj(segments: u32) -> (LoopMesh, BevelResult) {
    let mut mesh = LoopMesh::new();
    mesh_add_cube(&mut mesh);
    let e01 = cube_edge(
        &mesh,
        Vec3::new(1.0, -1.0, -1.0),
        Vec3::new(-1.0, -1.0, -1.0),
    );
    let e03 = cube_edge(&mesh, Vec3::new(1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, -1.0));
    let e04 = cube_edge(&mesh, Vec3::new(1.0, -1.0, -1.0), Vec3::new(1.0, -1.0, 1.0));
    let result = mesh.bevel(
        &[e01.into(), e03.into(), e04.into()],
        BevelOptions {
            offset: 0.25,
            affect: BevelAffect::Edges,
            segments,
            vmesh_method: BevelVmeshMethod::Adj,
            ..Default::default()
        },
    );
    (mesh, result)
}

/// The corner sphere (centre `c`, radius `R = offset`) the cube-corner patch
/// lies on (§8.5.5.3), and a "within a few `offset` of `c`" membership test that
/// isolates the corner-grid verts/faces from the per-edge strips (§8.5.5.6).
pub(super) fn adj_sphere_center() -> Vec3 {
    Vec3::new(0.75, -0.75, -0.75)
}

pub(super) const ADJ_SPHERE_RADIUS: f32 = 0.25;

pub(super) fn adj_in_corner_ball(mesh: &LoopMesh, vertex: VertexKey) -> bool {
    (mesh.vertex_position(vertex) - adj_sphere_center()).length() <= ADJ_SPHERE_RADIUS * 1.2
}

pub(super) fn adj_patch_faces(mesh: &LoopMesh, result: &BevelResult) -> Vec<FaceKey> {
    result
        .faces
        .iter()
        .copied()
        .filter(|&f| {
            mesh.face_vert_keys(f)
                .iter()
                .all(|&v| adj_in_corner_ball(mesh, v))
        })
        .collect()
}

pub(super) fn adj_apex_position() -> Vec3 {
    adj_sphere_center() + ADJ_SPHERE_RADIUS * Vec3::new(1.0, -1.0, -1.0) / 3.0_f32.sqrt()
}

pub(super) fn adj_assert_invariants(mesh: &LoopMesh, result: &BevelResult) {
    mesh.validate_topology();

    // Invariant 1 (§8.5.5.6): the three [P] boundverts, segment-count-independent
    // (offset·√2 from the corner).
    let corner = Vec3::new(1.0, -1.0, -1.0);
    let ring_radius = ADJ_SPHERE_RADIUS * 2.0_f32.sqrt();
    let ring = result
        .verts
        .iter()
        .filter(|&&v| ((mesh.vertex_position(v) - corner).length() - ring_radius).abs() < 1.0e-4)
        .count();
    assert_eq!(ring, 3, "three §7.6.2 boundverts (invariant 1)");

    // Invariant 2 (§8.5.5.6): every corner-region grid vert lies on the corner
    // sphere; the strip verts sit several offset away and are excluded.
    for &v in &result.verts {
        let p = mesh.vertex_position(v);
        if (p - adj_sphere_center()).length() <= ADJ_SPHERE_RADIUS * 1.2 {
            assert!(
                ((p - adj_sphere_center()).length() - ADJ_SPHERE_RADIUS).abs() < 1.0e-4,
                "corner-grid vert lies on the superellipsoid (invariant 2)",
            );
        }
    }

    // Closed manifold: every face edge is shared with exactly one other face.
    for face in mesh.face_keys() {
        for loop_key in mesh.face_loop_keys(face) {
            assert_eq!(
                mesh.edge_radial_iter(mesh.loop_edge(loop_key)).count(),
                2,
                "every edge is radial-2",
            );
        }
    }
}

/// A closed square pyramid: a four-way apex over a square base, four outward-wound
/// side triangles and one base quad. The apex is a **symmetric valence-4 junction**
/// — beveling its four slant edges drives the general valence-`s` (`s = 4`)
/// adjacency-grid cap (BEVEL.md §8.5.5.5). Returns `[apex, b0, b1, b2, b3]`.
pub(super) fn square_pyramid(mesh: &mut LoopMesh) -> [VertexKey; 5] {
    let apex = mesh.create_vertex(Vec3::new(0.0, 0.0, 1.4));
    let b0 = mesh.create_vertex(Vec3::new(1.0, 1.0, 0.0));
    let b1 = mesh.create_vertex(Vec3::new(-1.0, 1.0, 0.0));
    let b2 = mesh.create_vertex(Vec3::new(-1.0, -1.0, 0.0));
    let b3 = mesh.create_vertex(Vec3::new(1.0, -1.0, 0.0));
    let triangles = [
        [apex, b0, b1],
        [apex, b1, b2],
        [apex, b2, b3],
        [apex, b3, b0],
    ];
    for ring in triangles {
        let edges = [
            mesh.create_edge([ring[0], ring[1]], EdgeCreateFlags::NO_DUPLICATE),
            mesh.create_edge([ring[1], ring[2]], EdgeCreateFlags::NO_DUPLICATE),
            mesh.create_edge([ring[2], ring[0]], EdgeCreateFlags::NO_DUPLICATE),
        ];
        mesh.face_create(&ring, &edges, FaceCreateFlags::NONE);
    }
    // Base quad, wound downward (outward), reusing the four base edges.
    let base = [b0, b3, b2, b1];
    let base_edges = [
        mesh.create_edge([base[0], base[1]], EdgeCreateFlags::NO_DUPLICATE),
        mesh.create_edge([base[1], base[2]], EdgeCreateFlags::NO_DUPLICATE),
        mesh.create_edge([base[2], base[3]], EdgeCreateFlags::NO_DUPLICATE),
        mesh.create_edge([base[3], base[0]], EdgeCreateFlags::NO_DUPLICATE),
    ];
    mesh.face_create(&base, &base_edges, FaceCreateFlags::NONE);
    [apex, b0, b1, b2, b3]
}

/// The apex's §7.6.2 boundvert ring and per-ring-edge chain generators, re-derived from
/// the *sector* geometry alone — independently of the fill — so a test can predict the
/// §8.5.5.5 cap. `boundverts[i]` is the meet of the sector between slant edges `i` and
/// `i + 1`; ring edge `i` then spans `boundverts[i] → boundverts[i + 1]` across slant
/// edge `i + 1`, which is that chain's cross-section cell.
pub(super) fn pyramid_apex_ring(
    apex: Vec3,
    base: &[Vec3],
    offset: f32,
    profile: f32,
) -> (Vec<Vec3>, Vec<RingChainProfile<'static>>) {
    let sides = base.len();
    let toward = |p: Vec3| (p - apex).normalize();
    // §7.6.2 in a planar sector: the two offset lines meet on the bisector at
    // `v + offset·(d_a + d_b)/sin θ`.
    let meet = |p: Vec3, q: Vec3| {
        let (d_a, d_b) = (toward(p), toward(q));
        apex + offset * (d_a + d_b) / d_a.cross(d_b).length()
    };
    let boundverts: Vec<Vec3> = (0..sides)
        .map(|i| meet(base[i], base[(i + 1) % sides]))
        .collect();
    let profiles: Vec<RingChainProfile> = (0..sides)
        .map(|i| {
            let crossed = base[(i + 1) % sides];
            RingChainProfile::Cell {
                cell: ProfileCell::new(
                    boundverts[i],
                    boundverts[(i + 1) % sides],
                    apex,
                    toward(crossed),
                ),
                profile,
                custom: None,
            }
        })
        .collect();
    (boundverts, profiles)
}

/// Every canonical **interior** grid vert the §8.5.5.5 cap must emit, predicted from
/// the ring alone: seed the fullness centre (b), relax the coarse net (c), and read it
/// back at the target's profile-arc fractions (d).
pub(super) fn predicted_cap_interior(
    apex: Vec3,
    boundverts: &[Vec3],
    profiles: &[RingChainProfile],
    segments: u32,
    fullness: f32,
) -> Vec<Vec3> {
    let sectors = boundverts.len();
    let seg = segments as usize;
    let half = seg / 2;
    let even = seg % 2 == 0;
    let mids: Vec<Vec3> = profiles.iter().map(|p| p.raw_positions(2)[1]).collect();
    let centroid = boundverts.iter().fold(Vec3::ZERO, |acc, &b| acc + b) / sectors as f32;
    let seed = centroid + fullness * (apex - centroid);
    let net = adj_relaxed_net(boundverts, &mids, seed, profiles, segments);
    let axis = adj_interior_axis(profiles, segments, 2 * net.half as u32);
    let interior = AdjInterior::Relaxed {
        net: &net,
        axis: &axis,
        sectors,
        segments: seg,
    };

    let mut predicted = Vec::new();
    for i in 0..sectors {
        for j in 1..=half {
            for k in 1..=half {
                // Even `segments` weld the `j == half` row onto the previous sub-patch's
                // spoke and all the `(half, half)` corners onto one centre; odd keeps
                // every `(i, j, k)` its own.
                if even && j == half && (k == half && i != 0 || k < half) {
                    continue;
                }
                predicted.push(interior.position(i, j, k, half));
            }
        }
    }
    predicted
}

/// The values a run of [`SourcePosition`]s denotes. For assertions whose claim is
/// genuinely about the position — never for one about the tolerance, whose unit is the
/// remainder and whose whole point is that the value has already lost it.
pub(super) fn source_position_values(positions: &[SourcePosition]) -> Vec<f32> {
    positions.iter().map(|p| p.value()).collect()
}

/// A synthetic axis entry in [`chord_fraction_source_positions`]'s own spelling: an
/// integral position `p > 0` arrives as the **upper** endpoint `(p − 1, 1)` of the span
/// below it, because a query landing on a source sample divides a span by itself. Tests
/// that hand-build an axis must use this spelling or they exercise a pair the producer
/// never emits — and, at the far end, the only spelling available.
pub(super) fn upper_endpoint_spelling(value: f32) -> SourcePosition {
    let floor = value.floor();
    if value == floor && value > 0.0 {
        SourcePosition {
            index: floor as usize - 1,
            remainder: 1.0,
        }
    } else {
        SourcePosition {
            index: floor as usize,
            remainder: value - floor,
        }
    }
}

/// A closed square bipyramid (octahedron): two apexes over a square rim, all
/// eight triangular faces present. Beveling one apex's four spokes leaves the
/// rim verts as valence-4 **terminals** still incident to the opposite apex's
/// (unbeveled) faces — the closed-solid case §5.1.1's loose-free premise does
/// not cover. Returns `(top_apex, [r0, r1, r2, r3], bottom_apex)`.
pub(super) fn square_bipyramid(mesh: &mut LoopMesh) -> (VertexKey, [VertexKey; 4], VertexKey) {
    fn add_tri(mesh: &mut LoopMesh, tri: [VertexKey; 3]) {
        let edges = [
            mesh.create_edge([tri[0], tri[1]], EdgeCreateFlags::NO_DUPLICATE),
            mesh.create_edge([tri[1], tri[2]], EdgeCreateFlags::NO_DUPLICATE),
            mesh.create_edge([tri[2], tri[0]], EdgeCreateFlags::NO_DUPLICATE),
        ];
        mesh.face_create(&tri, &edges, FaceCreateFlags::NONE);
    }

    let top = mesh.create_vertex(Vec3::new(0.0, 0.0, 1.0));
    let bottom = mesh.create_vertex(Vec3::new(0.0, 0.0, -1.0));
    let rim = [
        mesh.create_vertex(Vec3::new(1.0, 0.0, 0.0)),
        mesh.create_vertex(Vec3::new(0.0, 1.0, 0.0)),
        mesh.create_vertex(Vec3::new(-1.0, 0.0, 0.0)),
        mesh.create_vertex(Vec3::new(0.0, -1.0, 0.0)),
    ];
    for k in 0..4 {
        let a = rim[k];
        let b = rim[(k + 1) % 4];
        add_tri(mesh, [top, a, b]);
        add_tri(mesh, [bottom, b, a]);
    }
    (top, rim, bottom)
}

/// Bevel the §8.5.5.8 worked-oracle driver: the same symmetric equal-offset cube
/// corner at `offset = 0.25`, with an explicit `profile` (BEVEL.md §8.5.5.5(f.2)).
pub(super) fn cube_corner_adj_profile(segments: u32, profile: f32) -> (LoopMesh, BevelResult) {
    cube_corner_adj_profile_offset(segments, profile, 0.25)
}

/// The same driver at an arbitrary bevel width. The corner's shape — and so every
/// tri-corner predicate that decides the fill path — is offset-independent; only the
/// scale changes.
pub(super) fn cube_corner_adj_profile_offset(
    segments: u32,
    profile: f32,
    offset: f32,
) -> (LoopMesh, BevelResult) {
    let mut mesh = LoopMesh::new();
    mesh_add_cube(&mut mesh);
    let e01 = cube_edge(
        &mesh,
        Vec3::new(1.0, -1.0, -1.0),
        Vec3::new(-1.0, -1.0, -1.0),
    );
    let e03 = cube_edge(&mesh, Vec3::new(1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, -1.0));
    let e04 = cube_edge(&mesh, Vec3::new(1.0, -1.0, -1.0), Vec3::new(1.0, -1.0, 1.0));
    let result = mesh.bevel(
        &[e01.into(), e03.into(), e04.into()],
        BevelOptions {
            offset,
            affect: BevelAffect::Edges,
            segments,
            profile,
            vmesh_method: BevelVmeshMethod::Adj,
            ..Default::default()
        },
    );
    (mesh, result)
}

/// The driver's corner-grid (cap) faces at an arbitrary bevel width: the `offset`-scaled
/// form of [`curved_corner_grid_faces`]. The frame origin `c` sits one `offset` inside
/// the corner along each axis, and the whole cap lies within `offset·√3` of it.
pub(super) fn corner_grid_faces_at_offset(
    mesh: &LoopMesh,
    result: &BevelResult,
    offset: f32,
) -> Vec<FaceKey> {
    let centre = Vec3::new(1.0, -1.0, -1.0) + offset * Vec3::new(-1.0, 1.0, 1.0);
    let reach = offset * 3.0_f32.sqrt() * 1.05;
    result
        .faces
        .iter()
        .copied()
        .filter(|&f| {
            mesh.face_vert_keys(f)
                .iter()
                .all(|&v| (mesh.vertex_position(v) - centre).length() <= reach)
        })
        .collect()
}

/// The corner-grid (cap) verts of a curved-profile cube corner: every vert of the
/// result within `offset·√3` of the corner frame's origin `c`. A curved cap bulges past
/// the circular radius (the apex sits at `3^(−1/r)·√3` octant units out), so the
/// circular `1.2·R` ball of `adj_in_corner_ball` would miss it; the nearest strip
/// cross-section is a whole cube edge away, so this stays unambiguous.
pub(super) fn curved_corner_grid_verts(mesh: &LoopMesh, result: &BevelResult) -> Vec<Vec3> {
    result
        .verts
        .iter()
        .map(|&v| mesh.vertex_position(v))
        .filter(|&p| (p - Vec3::new(0.75, -0.75, -0.75)).length() <= 0.25 * 3.0_f32.sqrt())
        .collect()
}

pub(super) fn curved_corner_grid_faces(mesh: &LoopMesh, result: &BevelResult) -> Vec<FaceKey> {
    result
        .faces
        .iter()
        .copied()
        .filter(|&f| {
            mesh.face_vert_keys(f).iter().all(|&v| {
                (mesh.vertex_position(v) - Vec3::new(0.75, -0.75, -0.75)).length()
                    <= 0.25 * 3.0_f32.sqrt()
            })
        })
        .collect()
}

pub(super) fn assert_position_present(verts: &[Vec3], expected: Vec3, what: &str) {
    let hits = verts
        .iter()
        .filter(|&&p| (p - expected).length() < 1.0e-4)
        .count();
    assert_eq!(hits, 1, "{what} at {expected:?} (got {verts:?})");
}

/// Bevel the four slant edges of the square pyramid's apex with the ADJ vmesh at an
/// explicit `segments` / `profile`: a **general, non-tri-corner** ring (valence `s = 4`),
/// so the tri-corner family of §8.5.5.5(f.1) step 3 never applies to it. The third
/// return is the apex junction's boundvert count — one per beveled edge, with no outer
/// miter in scope — which is the `s` of §8.5.5.5(e)'s parity law.
pub(super) fn pyramid_apex_adj(segments: u32, profile: f32) -> (LoopMesh, BevelResult, usize) {
    let mut mesh = LoopMesh::new();
    let [apex, b0, b1, b2, b3] = square_pyramid(&mut mesh);
    let slants = [
        mesh.query_edge_exists(apex, b0).expect("slant 0"),
        mesh.query_edge_exists(apex, b1).expect("slant 1"),
        mesh.query_edge_exists(apex, b2).expect("slant 2"),
        mesh.query_edge_exists(apex, b3).expect("slant 3"),
    ];
    let result = mesh.bevel(
        &slants.map(Into::into),
        BevelOptions {
            offset: 0.3,
            affect: BevelAffect::Edges,
            segments,
            profile,
            vmesh_method: BevelVmeshMethod::Adj,
            ..Default::default()
        },
    );
    (mesh, result, slants.len())
}

/// The cap the pyramid apex's [V-ADJ] fill emitted, selected **topologically**: a result
/// face every edge of which is shared only with other result faces. Each arriving strip
/// lands its terminal cross-section *on* the rebuilt originals, so every strip quad has
/// an edge against surviving geometry; the cap's own edges are its rim chains (shared
/// with the strips) and its interior. No world-space threshold, so a taller pyramid or a
/// larger offset cannot silently drop cap faces from the filter.
pub(super) fn pyramid_apex_cap(
    mesh: &LoopMesh,
    result: &BevelResult,
) -> (Vec<FaceKey>, Vec<VertexKey>) {
    let new_faces: std::collections::HashSet<FaceKey> = result.faces.iter().copied().collect();
    let faces: Vec<FaceKey> = result
        .faces
        .iter()
        .copied()
        .filter(|&face| {
            mesh.face_edge_keys(face).iter().all(|&edge| {
                mesh.edge_face_keys(edge)
                    .iter()
                    .all(|side| new_faces.contains(side))
            })
        })
        .collect();
    let mut seen: std::collections::HashSet<VertexKey> = std::collections::HashSet::new();
    let mut verts = Vec::new();
    for &face in &faces {
        for vertex in mesh.face_vert_keys(face) {
            if seen.insert(vertex) {
                verts.push(vertex);
            }
        }
    }
    (faces, verts)
}

/// Bevel the three slant edges of a regular triangular cone's apex. A **valence-3** ring
/// the §8.5.1 tri-corner test **refuses**: the three legs meet at a shallow, far-from-
/// orthonormal corner whose signed dihedral sum sits well outside the `±π/4` window
/// around `3·π/2`, so §8.5.5.5(f.1) sends it to step 4's general grid — including its
/// square-in arm, which a tri-corner-eligible ring would owe the (f.4a) weld instead.
pub(super) fn tripod_cone_adj(segments: u32, profile: f32) -> (LoopMesh, BevelResult) {
    let mut mesh = LoopMesh::new();
    let apex = mesh.create_vertex(Vec3::new(0.0, 0.0, 2.0));
    let base: Vec<VertexKey> = [
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(-0.5, 0.866, 0.0),
        Vec3::new(-0.5, -0.866, 0.0),
    ]
    .into_iter()
    .map(|position| mesh.create_vertex(position))
    .collect();
    for i in 0..3 {
        mesh.face_create_verts(&[apex, base[i], base[(i + 1) % 3]], FaceCreateFlags::NONE);
    }
    mesh.face_create_verts(&[base[0], base[2], base[1]], FaceCreateFlags::NONE);
    let slants: Vec<GeomKey> = base
        .iter()
        .map(|&b| mesh.query_edge_exists(apex, b).expect("slant edge").into())
        .collect();
    let result = mesh.bevel(
        &slants,
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

/// §8.5.5.5(e)'s parity law over `s` sectors — the count the general grid produces at
/// `segments` — as `(faces, verts)` of the whole cap.
pub(super) fn adj_general_ring_parity_counts(sectors: usize, segments: u32) -> (usize, usize) {
    let half = segments as usize / 2;
    if segments % 2 == 0 {
        (
            sectors * half * half,
            sectors * (half + 1) * (half + 1) - sectors * (half + 1) + 1,
        )
    } else {
        (
            sectors * half * half + sectors * half + 1,
            sectors * (half + 1) * (half + 1),
        )
    }
}

/// The world position of the cube-corner driver's octant origin — the image of the
/// unit octant's origin under `unit_to_corner`, and so the notch bottom every
/// §8.5.5.5(f.4a) leg walks down toward. It is the corner-surface centre `c`, not `v`.
pub(super) fn square_in_weld_origin() -> Vec3 {
    Vec3::new(0.75, -0.75, -0.75)
}

/// The cube-corner driver's three boundverts, in no particular ring order: the §7.6.2
/// meet points of the three beveled spokes at `offset = 0.25`.
pub(super) fn square_in_weld_boundverts() -> [Vec3; 3] {
    [
        Vec3::new(0.75, -0.75, -1.0),
        Vec3::new(1.0, -0.75, -0.75),
        Vec3::new(0.75, -1.0, -0.75),
    ]
}

/// The junction's whole welded ring: every result vert inside the corner ball, which at
/// a square-in weld is the `s` boundverts plus their inward legs and nothing else — the
/// weld has no interior.
pub(super) fn square_in_weld_ring(mesh: &LoopMesh, result: &BevelResult) -> Vec<VertexKey> {
    result
        .verts
        .iter()
        .copied()
        .filter(|&v| (mesh.vertex_position(v) - square_in_weld_origin()).length() <= 0.25 + 1.0e-4)
        .collect()
}

pub(super) fn square_in_weld_vert_at(
    mesh: &LoopMesh,
    result: &BevelResult,
    position: Vec3,
) -> Option<VertexKey> {
    let mut hits = result
        .verts
        .iter()
        .copied()
        .filter(|&v| (mesh.vertex_position(v) - position).length() < 1.0e-4);
    let first = hits.next();
    assert!(hits.next().is_none(), "at most one vertex at {position:?}");
    first
}

/// The step BEVEL.md §8.5.5.5(f.4a) prints, quoted rather than re-derived: `b = 2/n` at
/// even `n`, `b = 2/(2·⌊n/2⌋ + √2)` at odd `n` — the closed form of §8.6.4 step 3's
/// closing-chord term on two straight unit legs.
pub(super) fn square_in_weld_step(segments: u32) -> f32 {
    let half = (segments / 2) as f32;
    if segments % 2 == 0 {
        2.0 / segments as f32
    } else {
        2.0 / (2.0 * half + std::f32::consts::SQRT_2)
    }
}

/// The §8.5.5.8 driver's corner with an **inner miter** (BEVEL.md §8.9.3) pushed out by
/// `spread`, through the custom curve when one is given and the analytic family when not.
/// The two arms share every other input, so any positional difference between them is the
/// curve's.
///
/// `profile` is a parameter because the miter's admission gate is keyed on the profile
/// *type*, never on the slider — sweeping the slider under a curve must not decide whether
/// the miter fires
/// ([`custom_profile_miter_admission_is_independent_of_the_superellipse_slider`]) — while
/// the arc chain's *placement* does move with it (§8.6.3.1(4).4's carve-out withholds the
/// curve, not the scalar).
/// `miter_inner` is one so the same fixture supplies the sharp contrast.
pub(super) fn cube_corner_adj_inner_miter_profile(
    curve: Option<&BevelCurveProfile>,
    spread: f32,
    segments: u32,
    profile: f32,
    miter_inner: BevelMiter,
) -> (LoopMesh, BevelResult) {
    let mut mesh = LoopMesh::new();
    mesh_add_cube(&mut mesh);
    let e01 = cube_edge(
        &mesh,
        Vec3::new(1.0, -1.0, -1.0),
        Vec3::new(-1.0, -1.0, -1.0),
    );
    let e03 = cube_edge(&mesh, Vec3::new(1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, -1.0));
    let e04 = cube_edge(&mesh, Vec3::new(1.0, -1.0, -1.0), Vec3::new(1.0, -1.0, 1.0));
    let result = mesh.bevel(
        &[e01.into(), e03.into(), e04.into()],
        BevelOptions {
            offset: 0.25,
            affect: BevelAffect::Edges,
            segments,
            profile,
            profile_type: match curve {
                Some(curve) => BevelProfileType::Custom(curve),
                None => BevelProfileType::Superellipse,
            },
            vmesh_method: BevelVmeshMethod::Adj,
            miter_inner,
            spread,
            ..Default::default()
        },
    );
    (mesh, result)
}

/// The single-face reflex junction of [`single_face_reflex_junction`] under an **outer**
/// miter, through the custom curve when one is given. Same shape as
/// [`cube_corner_adj_inner_miter_profile`]: the slider is swept against it because a
/// miter's *admission* never reads the slider (§8.9.6), even though — under `PATCH`
/// alone — its emitted chain does not read it either (§8.9.5's straight legs).
pub(super) fn reflex_apex_outer_miter_profile(
    kind: BevelMiter,
    curve: Option<&BevelCurveProfile>,
    profile: f32,
    segments: u32,
) -> (LoopMesh, BevelResult) {
    let (mut mesh, _apex, [va, vb, vc]) = single_face_reflex_junction();
    let result = mesh.bevel(
        &[va.into(), vb.into(), vc.into()],
        BevelOptions {
            offset: 0.2,
            affect: BevelAffect::Edges,
            segments,
            profile,
            profile_type: match curve {
                Some(curve) => BevelProfileType::Custom(curve),
                None => BevelProfileType::Superellipse,
            },
            vmesh_method: BevelVmeshMethod::Adj,
            miter_outer: kind,
            ..Default::default()
        },
    );
    (mesh, result)
}

/// Count the still-live newly minted edges carrying `flag`. Mechanism (1) fires
/// unconditionally, so the ON−OFF delta of this over `mark_seam` isolates the
/// mechanism-(2) bunch-side bridge run (§6.7.5.3).
pub(super) fn count_flagged_result_edges(
    mesh: &LoopMesh,
    result: &BevelResult,
    flag: impl Fn(&LoopMesh, EdgeKey) -> bool,
) -> usize {
    result
        .edges
        .iter()
        .filter(|&&e| mesh.edge_keys().contains(&e) && flag(mesh, e))
        .count()
}

/// The minimum ring slide of a corner: distance from the original corner
/// position to the nearest of `result`'s new ring verts. For a single corner
/// at `offset` every ring vert slides exactly `offset` (§4); the *minimum* is
/// robust against the other corner's ring verts in a two-corner bevel.
pub(super) fn nearest_ring_slide(mesh: &LoopMesh, result: &BevelResult, corner: Vec3) -> f32 {
    result
        .verts
        .iter()
        .map(|&v| (mesh.vertex_position(v) - corner).length())
        .fold(f32::INFINITY, f32::min)
}

pub(super) fn vert_near(mesh: &LoopMesh, position: Vec3, tol: f32) -> Option<VertexKey> {
    mesh.vert_keys()
        .into_iter()
        .find(|&v| (mesh.vertex_position(v) - position).length() < tol)
}

/// A `5 × 5`-quad grid in the `z = 0` plane (`36 V / 60 E / 25 F`), vert `(i, j)` at
/// `(i, j, 0)`. Every vert with `1 ≤ i, j ≤ 4` is interior, so an edge between two of
/// them is a **lone** beveled edge whose two endpoints are both valence-4 terminals.
pub(super) fn quad_grid_5x5() -> (LoopMesh, Vec<Vec<VertexKey>>) {
    let mut mesh = LoopMesh::new();
    let verts: Vec<Vec<VertexKey>> = (0..6)
        .map(|i| {
            (0..6)
                .map(|j| mesh.create_vertex(Vec3::new(i as f32, j as f32, 0.0)))
                .collect()
        })
        .collect();
    for i in 0..5 {
        for j in 0..5 {
            mesh.face_create_verts(
                &[
                    verts[i][j],
                    verts[i + 1][j],
                    verts[i + 1][j + 1],
                    verts[i][j + 1],
                ],
                FaceCreateFlags::NONE,
            );
        }
    }
    (mesh, verts)
}

/// A hexagonal corner of radius `0.4` at height `z`. The radius makes every
/// circumferential edge exactly `0.4` long, which is the leg length BEVEL.md
/// §8.3.0.1's worked slide-clamp table is quoted on.
pub(super) fn hex_tube_corner(corner: usize, z: f32) -> Vec3 {
    let angle = corner as f32 * std::f32::consts::FRAC_PI_3;
    Vec3::new(0.4 * angle.cos(), 0.4 * angle.sin(), z)
}

/// A closed hexagonal tube of radius `0.4`, three rings at `z = 0, 1, 2`, side
/// faces only. Every middle-ring vertex is a face-bearing **valence-4**
/// terminal — two circumferential spokes, two axial — so beveling one middle
/// circumferential edge gives each of its endpoints exactly one non-flanking
/// leg (the next circumferential edge round), which is the leg vert §8.3.0.1
/// lengthens by `√2` and §4.1 then clamps.
pub(super) fn hex_tube_radius_two_fifths() -> (LoopMesh, Vec<Vec<VertexKey>>) {
    let mut mesh = LoopMesh::new();
    let rings: Vec<Vec<VertexKey>> = (0..3)
        .map(|ring| {
            (0..6)
                .map(|corner| mesh.create_vertex(hex_tube_corner(corner, ring as f32)))
                .collect()
        })
        .collect();
    for ring in 0..2 {
        for corner in 0..6 {
            let next = (corner + 1) % 6;
            mesh.face_create_verts(
                &[
                    rings[ring][corner],
                    rings[ring][next],
                    rings[ring + 1][next],
                    rings[ring + 1][corner],
                ],
                FaceCreateFlags::NONE,
            );
        }
    }
    (mesh, rings)
}

/// Bevel one middle-ring circumferential edge of the hex tube and report the
/// mesh plus the `corner 1` terminal's leg geometry: that terminal's original
/// position, its non-flanking leg's far endpoint, and the result.
pub(super) fn bevel_hex_tube_middle_edge(
    offset: f32,
    profile: f32,
    segments: u32,
) -> (LoopMesh, BevelResult, Vec3, Vec3) {
    let (mut mesh, rings) = hex_tube_radius_two_fifths();
    let beveled = mesh
        .query_edge_exists(rings[1][0], rings[1][1])
        .expect("the middle-ring circumferential edge");
    let terminal = mesh.vertex_position(rings[1][1]);
    let leg_far = mesh.vertex_position(rings[1][2]);
    let result = mesh.bevel(
        &[beveled.into()],
        BevelOptions {
            offset,
            profile,
            segments,
            affect: BevelAffect::Edges,
            ..Default::default()
        },
    );
    mesh.validate_topology();
    (mesh, result, terminal, leg_far)
}

/// The `5 × 5` grid with the `(1,1)` quad left out. `(2,2)` keeps all four of its
/// edges but loses one of its four faces, so its disk is an **open arc**
/// `[boundary, E, interior, boundary]` — valence 4, three faces, and no cycle.
pub(super) fn quad_grid_5x5_holed() -> (LoopMesh, Vec<Vec<VertexKey>>) {
    let mut mesh = LoopMesh::new();
    let verts: Vec<Vec<VertexKey>> = (0..6)
        .map(|i| {
            (0..6)
                .map(|j| mesh.create_vertex(Vec3::new(i as f32, j as f32, 0.0)))
                .collect()
        })
        .collect();
    for i in 0..5 {
        for j in 0..5 {
            if (i, j) == (1, 1) {
                continue;
            }
            mesh.face_create_verts(
                &[
                    verts[i][j],
                    verts[i + 1][j],
                    verts[i + 1][j + 1],
                    verts[i][j + 1],
                ],
                FaceCreateFlags::NONE,
            );
        }
    }
    (mesh, verts)
}

/// A 3×3 **vertex** grid patch — 9 V / 12 E / 4 quads — translated by `origin`, so two of
/// them fit disjointly in one mesh. Returns its `(1,1,0)` centre (an interior valence-4
/// vertex) and the centre's `+x` edge, whose far endpoint is a valence-3 boundary vertex:
/// a terminal at both ends.
pub(super) fn grid_3x3(mesh: &mut LoopMesh, origin: Vec3) -> (VertexKey, EdgeKey) {
    let verts: Vec<Vec<VertexKey>> = (0..3)
        .map(|i| {
            (0..3)
                .map(|j| mesh.create_vertex(origin + Vec3::new(i as f32, j as f32, 0.0)))
                .collect()
        })
        .collect();
    for i in 0..2 {
        for j in 0..2 {
            mesh.face_create_verts(
                &[
                    verts[i][j],
                    verts[i + 1][j],
                    verts[i + 1][j + 1],
                    verts[i][j + 1],
                ],
                FaceCreateFlags::NONE,
            );
        }
    }
    let beveled = mesh
        .query_edge_exists(verts[1][1], verts[2][1])
        .expect("the centre's +x edge");
    (verts[1][1], beveled)
}

/// The far ends of every face-less edge still incident to `tip` — the re-attached wire
/// (BEVEL.md §8.3.0.4), read back off the result.
pub(super) fn wire_landings(mesh: &LoopMesh, tip: VertexKey) -> Vec<Vec3> {
    mesh.iter_disk_edges(tip)
        .filter(|&edge| mesh.edge_radial_iter(edge).next().is_none())
        .map(|edge| {
            let [a, b] = mesh.edge_vert_keys(edge);
            mesh.vertex_position(if a == tip { b } else { a })
        })
        .collect()
}

pub(super) fn face_size_multiset(mesh: &LoopMesh) -> Vec<usize> {
    let mut sizes: Vec<usize> = mesh
        .face_keys()
        .into_iter()
        .map(|face| mesh.face_vert_keys(face).len())
        .collect();
    sizes.sort_unstable();
    sizes
}

/// The terminal's fill faces: the new faces at a **leg vert** every one of whose verts
/// the bevel minted. The strip's faces carry no leg vert, and the surviving faces that
/// absorbed the corner still carry their own original verts, so this is exactly the
/// fill — one n-gon, or the `segments` triangles of a fan.
pub(super) fn terminal_fill_faces(
    mesh: &LoopMesh,
    minted: &[VertexKey],
    leg: VertexKey,
) -> Vec<Vec<VertexKey>> {
    mesh.vert_face_keys(leg)
        .into_iter()
        .map(|face| mesh.face_vert_keys(face))
        .filter(|cycle| cycle.iter().all(|vert| minted.contains(vert)))
        .collect()
}

/// The asymmetric all-vector control net BEVEL.md §8.6.3.1(6)'s Regime-A worked examples
/// measure with: `(1, 0) → (0.5, 0.25) → (0, 1)`. Its interior sample sits at cell
/// `(0.25, 0.5)` fed forward and at the diagonal reflection `(0.5, 0.25)` fed mirrored, so
/// every handedness slip shows as a visibly displaced rim vertex.
pub(super) fn asymmetric_custom_net() -> BevelCurveProfile {
    BevelCurveProfile::from_points(&[
        (1.0, 0.0, BevelHandle::Vector),
        (0.5, 0.25, BevelHandle::Vector),
        (0.0, 1.0, BevelHandle::Vector),
    ])
}

/// Run BEVEL.md §8.6.3.1(6)'s Regime-A orientation pass over `beveled` at `offset` and
/// hand back the marks with the membership test the walk itself used.
pub(super) fn orientation_marks(
    mesh: &LoopMesh,
    beveled: &[EdgeKey],
    offset: f32,
) -> (CustomOrientation, HashSet<EdgeKey>) {
    let set: HashSet<EdgeKey> = beveled.iter().copied().collect();
    let orientation = {
        let is_beveled = |edge: EdgeKey| set.contains(&edge);
        mesh.plan_custom_orientation(beveled, &is_beveled, offset)
    };
    (orientation, set)
}

pub(super) fn face_edge_keys(mesh: &LoopMesh, face: FaceKey) -> Vec<EdgeKey> {
    let verts = mesh.face_vert_keys(face);
    (0..verts.len())
        .map(|i| {
            mesh.query_edge_exists(verts[i], verts[(i + 1) % verts.len()])
                .expect("a face's boundary edge exists")
        })
        .collect()
}
