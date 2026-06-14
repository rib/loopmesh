use crate::loopmesh::*;

/// The geometry [`LoopMesh::bisect_edges`] created, gathered as a set
/// ([`BISECT_EDGES.md`](../../../../radial-mesh-guides/operators/BISECT_EDGES.md)
/// §1). Enumeration order is not observable for parity; treat these as a set.
///
/// Unlike [`SubdivideGeom`], there is no `faces` field: bisect-edges never
/// re-splits a face, so the only geometry it produces is the midpoint verts
/// (and the edge-halves they sit between). The face count is invariant (§4).
#[derive(Default, Clone, Debug, PartialEq, Eq)]
pub struct BisectGeom {
    /// Every midpoint vert inserted on a cut edge (§3 / §3.2). A cut edge
    /// shared by several faces still contributes only its midpoints once (§2).
    pub verts: Vec<VertexKey>,
}

impl LoopMesh {
    /// Edge-driven refinement with no per-face re-split
    /// ([`BISECT_EDGES.md`](../../../../radial-mesh-guides/operators/BISECT_EDGES.md)).
    ///
    /// Splits each edge in `edges` at `cuts` evenly-spaced midpoints — a single
    /// midpoint at the parametric centre `t = 0.5` for `cuts = 1`, or `N`
    /// midpoints at `t = i/(N+1)` for `cuts = N` (§3.1 / §3.2) — and stops
    /// there. This is exactly the per-edge midpoint phase of
    /// [`LoopMesh::subdivide_edges`] with the per-face pattern dispatch removed
    /// entirely: every touched face is left as a flat-cornered `(n + cuts)`-gon,
    /// the face count never changes (§4), and no interior edge or inner vert is
    /// ever created.
    ///
    /// This is **not** subdivide-edges with the re-split gates off: that variant
    /// still runs an unconditional two-cut-edges-per-face connect. Bisect-edges
    /// runs no connect for any face under any configuration (§2 / §7), so a face
    /// with two, three, or four cut edges keeps every midpoint dangling.
    ///
    /// A shared interior edge is cut once and its midpoints are seen by both
    /// adjacent faces' loop cycles, so neither face cracks and neither is
    /// re-split (§2). A non-manifold edge (three or more incident faces) is
    /// split exactly like a manifold one; every radial face gains the shared
    /// midpoint as a flat corner (§2 / §5). A wire edge with no incident faces
    /// is split into collinear segments with no face involved (§5).
    ///
    /// `cuts = 0` is a no-op that leaves the mesh unchanged (§5). Returns the
    /// midpoint verts created (`geom_split.out`, §1); the face set is invariant.
    pub fn bisect_edges(&mut self, edges: &[EdgeKey], cuts: u32) -> BisectGeom {
        let mut geom = BisectGeom::default();
        if cuts == 0 {
            return geom;
        }

        // The entire operation: one per-edge multi-cut split per named edge, in
        // a single forward pass. `edge_split_n` places each midpoint on the
        // chord at t = i/(N+1) (so t = 0.5 for the single cut) and applies the
        // §6 two-source-linear per-vert / per-loop / per-edge interpolation. No
        // face is ever inspected, so there is no per-face re-split to suppress —
        // there simply is none.
        for &edge in edges {
            let [near, _far] = self.edge_vert_keys(edge);
            let midpoints = self.edge_split_n(edge, near, cuts);
            geom.verts.extend(midpoints);
        }

        geom
    }
}

#[cfg(test)]
mod tests {
    use glam::Vec3;

    use super::*;
    use crate::loopmesh::test_support::create_face_from_verts;

    fn counts(mesh: &LoopMesh) -> (usize, usize, usize) {
        (mesh.vertex_count(), mesh.edge_count(), mesh.face_count())
    }

    fn unit_quad(mesh: &mut LoopMesh) -> FaceKey {
        create_face_from_verts(
            mesh,
            &[
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(2.0, 0.0, 0.0),
                Vec3::new(2.0, 2.0, 0.0),
                Vec3::new(0.0, 2.0, 0.0),
            ],
        )
    }

    fn unit_tri(mesh: &mut LoopMesh) -> FaceKey {
        create_face_from_verts(
            mesh,
            &[
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(2.0, 0.0, 0.0),
                Vec3::new(0.0, 2.0, 0.0),
            ],
        )
    }

    #[test]
    fn quad_single_edge_leaves_pentagon() {
        // §3.1 + §4 — one cut edge, no re-split: quad -> flat pentagon.
        let mut mesh = LoopMesh::new();
        let face = unit_quad(&mut mesh);
        let fv = mesh.face_vert_keys(face);
        let e01 = mesh.query_edge_exists(fv[0], fv[1]).unwrap();

        let geom = mesh.bisect_edges(&[e01], 1);

        assert_eq!(counts(&mesh), (5, 5, 1), "quad one-cut: V5 E5 F1, ΔF=0");
        assert_eq!(geom.verts.len(), 1, "one midpoint reported");
        assert_eq!(mesh.face_len(face), 5, "face is a flat-cornered pentagon");

        // Midpoint lands on the chord at t = 0.5.
        let mid = geom.verts[0];
        assert!(
            mesh.vertex_position(mid).distance(Vec3::new(1.0, 0.0, 0.0)) < 1e-6,
            "midpoint at edge centre"
        );
    }

    #[test]
    fn tri_single_edge_leaves_quad() {
        // §3.1 — touched triangle becomes a flat quad (V4 E4 F1).
        let mut mesh = LoopMesh::new();
        let face = unit_tri(&mut mesh);
        let fv = mesh.face_vert_keys(face);
        let e01 = mesh.query_edge_exists(fv[0], fv[1]).unwrap();

        mesh.bisect_edges(&[e01], 1);

        assert_eq!(counts(&mesh), (4, 4, 1), "tri one-cut: V4 E4 F1");
        assert_eq!(mesh.face_len(face), 4, "triangle becomes a flat quad");
    }

    #[test]
    fn quad_two_adjacent_edges_leaves_hexagon_no_resplit() {
        // §2 + §8 — the defining contrast with subdivide_edges: two adjacent cut
        // edges leave a hexagon, NOT a connected/split face.
        let mut mesh = LoopMesh::new();
        let face = unit_quad(&mut mesh);
        let fv = mesh.face_vert_keys(face);
        let e01 = mesh.query_edge_exists(fv[0], fv[1]).unwrap();
        let e12 = mesh.query_edge_exists(fv[1], fv[2]).unwrap();

        mesh.bisect_edges(&[e01, e12], 1);

        assert_eq!(counts(&mesh), (6, 6, 1), "two adjacent cuts: V6 E6 F1");
        assert_eq!(mesh.face_len(face), 6, "hexagon, no interior connect");
    }

    #[test]
    fn quad_all_four_edges_leaves_octagon() {
        // §2 + §4 — all four cut edges leave an octagon, no inner vert.
        let mut mesh = LoopMesh::new();
        let face = unit_quad(&mut mesh);
        let fv = mesh.face_vert_keys(face);
        let edges: Vec<_> = (0..4)
            .map(|i| mesh.query_edge_exists(fv[i], fv[(i + 1) % 4]).unwrap())
            .collect();

        mesh.bisect_edges(&edges, 1);

        assert_eq!(counts(&mesh), (8, 8, 1), "all four cuts: V8 E8 F1");
        assert_eq!(mesh.face_len(face), 8, "octagon, no inner vert");
    }

    #[test]
    fn tri_all_three_edges_leaves_hexagon() {
        // §2 — all three cut edges leave a hexagon, no inner triangle.
        let mut mesh = LoopMesh::new();
        let face = unit_tri(&mut mesh);
        let fv = mesh.face_vert_keys(face);
        let edges: Vec<_> = (0..3)
            .map(|i| mesh.query_edge_exists(fv[i], fv[(i + 1) % 3]).unwrap())
            .collect();

        mesh.bisect_edges(&edges, 1);

        assert_eq!(counts(&mesh), (6, 6, 1), "tri all-three cuts: V6 E6 F1");
        assert_eq!(mesh.face_len(face), 6, "hexagon, no 1-to-4 split");
    }

    #[test]
    fn quad_all_four_edges_cuts2_twelve_gon() {
        // §3.2 — multi-cut: N=2 per edge, face un-split -> 12-gon.
        let mut mesh = LoopMesh::new();
        let face = unit_quad(&mut mesh);
        let fv = mesh.face_vert_keys(face);
        let edges: Vec<_> = (0..4)
            .map(|i| mesh.query_edge_exists(fv[i], fv[(i + 1) % 4]).unwrap())
            .collect();

        let geom = mesh.bisect_edges(&edges, 2);

        assert_eq!(counts(&mesh), (12, 12, 1), "cuts=2 all four: V12 E12 F1");
        assert_eq!(geom.verts.len(), 8, "two midpoints per cut edge");
        assert_eq!(mesh.face_len(face), 12, "12-gon, no lattice fill");

        // The bottom edge {v0,v1} from (0,0,0) to (2,0,0) gains evenly-spaced
        // midpoints at t = 1/3 and t = 2/3 (§3.2 spacing).
        let xs: Vec<f32> = geom
            .verts
            .iter()
            .map(|&m| mesh.vertex_position(m))
            .filter(|p| p.y.abs() < 1e-6 && p.x > 0.0 && p.x < 2.0)
            .map(|p| p.x)
            .collect();
        assert_eq!(xs.len(), 2, "two midpoints on the bottom edge");
        let mut sorted = xs.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!((sorted[0] - 2.0 / 3.0).abs() < 1e-5, "first at t=1/3");
        assert!((sorted[1] - 4.0 / 3.0).abs() < 1e-5, "second at t=2/3");
    }

    /// Build a quad `{v0,v1,v2,v3}` and a triangle `{v1,v4,v2}` welded across
    /// the shared interior edge `{v1,v2}`. Returns the mesh and the shared edge.
    fn quad_and_tri_shared_edge() -> (LoopMesh, EdgeKey, FaceKey, FaceKey) {
        let mut mesh = LoopMesh::new();
        let v0 = mesh.create_vertex(Vec3::new(0.0, 0.0, 0.0));
        let v1 = mesh.create_vertex(Vec3::new(1.0, 0.0, 0.0));
        let v2 = mesh.create_vertex(Vec3::new(1.0, 1.0, 0.0));
        let v3 = mesh.create_vertex(Vec3::new(0.0, 1.0, 0.0));
        let v4 = mesh.create_vertex(Vec3::new(2.0, 0.5, 0.0));

        let e01 = mesh.create_edge([v0, v1], EdgeCreateFlags::NONE);
        let e12 = mesh.create_edge([v1, v2], EdgeCreateFlags::NONE);
        let e23 = mesh.create_edge([v2, v3], EdgeCreateFlags::NONE);
        let e30 = mesh.create_edge([v3, v0], EdgeCreateFlags::NONE);
        let e14 = mesh.create_edge([v1, v4], EdgeCreateFlags::NONE);
        let e42 = mesh.create_edge([v4, v2], EdgeCreateFlags::NONE);

        let quad = mesh.face_create(
            &[v0, v1, v2, v3],
            &[e01, e12, e23, e30],
            FaceCreateFlags::NONE,
        );
        let tri = mesh.face_create(&[v1, v4, v2], &[e14, e42, e12], FaceCreateFlags::NONE);
        (mesh, e12, quad, tri)
    }

    #[test]
    fn shared_interior_edge_midpoint_once_both_unsplit() {
        // §2 — cutting the shared edge inserts the midpoint once; both incident
        // faces grow by one flat corner and neither is re-split.
        let (mut mesh, shared, quad, tri) = quad_and_tri_shared_edge();
        assert_eq!(counts(&mesh), (5, 6, 2), "starting quad+tri fixture");

        let geom = mesh.bisect_edges(&[shared], 1);

        assert_eq!(counts(&mesh), (6, 7, 2), "one shared midpoint, ΔF=0");
        assert_eq!(geom.verts.len(), 1, "midpoint created exactly once");
        assert_eq!(mesh.face_len(quad), 5, "quad gains one flat corner");
        assert_eq!(mesh.face_len(tri), 4, "tri gains one flat corner");

        // The single midpoint is seen by both faces' loop cycles (crack-free).
        let mid = geom.verts[0];
        let incident = mesh.vert_face_keys(mid);
        assert!(
            incident.contains(&quad) && incident.contains(&tri),
            "shared by both faces"
        );
    }

    #[test]
    fn nonmanifold_edge_every_radial_face_gains_corner() {
        // §2 / §5 — a spine edge shared by three faces splits like a manifold
        // one; every radial face gains the shared midpoint as a flat corner and
        // none is re-split.
        let mut mesh = LoopMesh::new();
        let s0 = mesh.create_vertex(Vec3::new(0.0, 0.0, 0.0));
        let s1 = mesh.create_vertex(Vec3::new(0.0, 0.0, 2.0));
        // Three wing tips fanning off the spine.
        let w0 = mesh.create_vertex(Vec3::new(2.0, 0.0, 1.0));
        let w1 = mesh.create_vertex(Vec3::new(-1.0, 1.7, 1.0));
        let w2 = mesh.create_vertex(Vec3::new(-1.0, -1.7, 1.0));

        let spine = mesh.create_edge([s0, s1], EdgeCreateFlags::NONE);
        let mut faces = Vec::new();
        for &w in &[w0, w1, w2] {
            let e_s0_w = mesh.create_edge([s0, w], EdgeCreateFlags::NONE);
            let e_w_s1 = mesh.create_edge([w, s1], EdgeCreateFlags::NONE);
            faces.push(mesh.face_create(
                &[s0, w, s1],
                &[e_s0_w, e_w_s1, spine],
                FaceCreateFlags::NONE,
            ));
        }
        assert_eq!(counts(&mesh), (5, 7, 3), "starting radial-3 book");

        let geom = mesh.bisect_edges(&[spine], 1);

        assert_eq!(geom.verts.len(), 1, "spine midpoint created once");
        assert_eq!(mesh.face_count(), 3, "ΔF=0, no per-face dispatch");
        let mid = geom.verts[0];
        for &f in &faces {
            assert_eq!(mesh.face_len(f), 4, "every radial face gains a flat corner");
            assert!(
                mesh.vert_face_keys(mid).contains(&f),
                "midpoint shared by face"
            );
        }
    }

    #[test]
    fn wire_edge_splits_into_two_segments() {
        // §5 — a wire edge with no incident face is split into two collinear
        // wire segments; no face is involved.
        let mut mesh = LoopMesh::new();
        let a = mesh.create_vertex(Vec3::new(0.0, 0.0, 0.0));
        let b = mesh.create_vertex(Vec3::new(2.0, 0.0, 0.0));
        let wire = mesh.create_edge([a, b], EdgeCreateFlags::NONE);

        let geom = mesh.bisect_edges(&[wire], 1);

        assert_eq!(counts(&mesh), (3, 2, 0), "wire -> two segments, no face");
        assert_eq!(geom.verts.len(), 1, "one midpoint");
        let mid = geom.verts[0];
        assert!(
            mesh.vertex_position(mid).distance(Vec3::new(1.0, 0.0, 0.0)) < 1e-6,
            "midpoint at wire centre"
        );
        assert!(
            mesh.vert_face_keys(mid).is_empty(),
            "no face touches the midpoint"
        );
    }

    #[test]
    fn cuts_zero_is_noop() {
        // §5 — cuts = 0 leaves the mesh untouched.
        let mut mesh = LoopMesh::new();
        let face = unit_quad(&mut mesh);
        let fv = mesh.face_vert_keys(face);
        let e01 = mesh.query_edge_exists(fv[0], fv[1]).unwrap();

        let geom = mesh.bisect_edges(&[e01], 0);

        assert_eq!(counts(&mesh), (4, 4, 1), "cuts=0: mesh unchanged");
        assert!(geom.verts.is_empty(), "no midpoints created");
    }

    #[test]
    fn empty_edge_set_is_noop() {
        let mut mesh = LoopMesh::new();
        unit_quad(&mut mesh);

        let geom = mesh.bisect_edges(&[], 1);

        assert_eq!(counts(&mesh), (4, 4, 1), "empty set: mesh unchanged");
        assert!(geom.verts.is_empty());
    }
}
