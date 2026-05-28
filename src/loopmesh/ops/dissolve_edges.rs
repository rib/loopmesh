use crate::loopmesh::*;

/// Parameters controlling [`LoopMesh::dissolve_edges`].
///
/// Mirrors the BMOP set-input shape described in
/// `radial-mesh-guides/operators/DISSOLVE_EDGES.md` §1. Only the
/// default-parameter path (every field at its default) is implemented
/// today; the optional pre-pass (`use_face_split`), post-pass
/// (`use_verts`), and the `angle_threshold` gate are reserved for later
/// and ignored when left at their defaults.
#[derive(Clone, Copy)]
pub struct DissolveEdgesParams {
    /// Post-pass: collapse adjacent verts that became edge-pairs after the
    /// per-edge merges (§5). Not yet implemented; must be `false`.
    pub use_verts: bool,
    /// Pre-pass: triangulate n-gon corners around verts that would become
    /// edge-pairs (§4). Not yet implemented; must be `false`.
    pub use_face_split: bool,
    /// Gate for `use_verts` (§5.2). Only consulted when `use_verts` is on.
    pub angle_threshold: f32,
}

impl Default for DissolveEdgesParams {
    fn default() -> Self {
        Self {
            use_verts: false,
            use_face_split: false,
            angle_threshold: std::f32::consts::PI,
        }
    }
}

/// Per-edge classification of an input edge (§3). Drives whether the
/// per-edge merge runs at all: only `ManifoldPair` does work; everything
/// else is a silent no-op for that edge.
enum EdgeClass {
    ManifoldPair { face_a: FaceKey, face_b: FaceKey },
    NoOp,
}

impl LoopMesh {
    /// Remove every input edge that flanks two distinct faces, merging the
    /// two faces into one.
    ///
    /// Follows the algorithm in
    /// `radial-mesh-guides/operators/DISSOLVE_EDGES.md`:
    ///
    /// 1. §3 per-edge classification: an edge is acted on only when its
    ///    radial cycle has exactly two loops belonging to two *distinct*
    ///    faces (the manifold face-pair case). Wire (radial 0), boundary
    ///    (radial 1), self-radial (radial 2, same face), and non-manifold
    ///    (radial ≥ 3) edges are silently skipped.
    /// 2. §6.4 per-edge merge: each manifold-pair edge is dissolved by
    ///    merging its two faces via the n==2 `faces_join` form, which is
    ///    the same JFKE surgery §3.1 / §6.4 names — the shared edge and its
    ///    two loops die, and the merged face's perimeter is the symmetric
    ///    difference of the two source perimeters (§3.1).
    /// 3. §6.4 / §8 multi-edge iteration: the classification is re-tested
    ///    afresh for every input edge, so a later edge sees the topology
    ///    left by earlier merges (the shared-endpoint case picks up the
    ///    post-first-merge face pair; disjoint pairs are order-independent).
    ///
    /// Only the default-parameter path is implemented. The
    /// `use_verts` / `use_face_split` / `angle_threshold` knobs of
    /// [`DissolveEdgesParams`] are reserved for later and must be left at
    /// their defaults; passing non-default values panics as an internal
    /// guard rather than silently doing the wrong thing.
    pub fn dissolve_edges(&mut self, edges: &[EdgeKey], params: DissolveEdgesParams) {
        assert!(
            !params.use_verts && !params.use_face_split,
            "dissolve_edges: only the default-parameter path is implemented",
        );

        // §6.4 — per-edge dispatch in input order. Each iteration
        // re-classifies the edge against the *current* topology so the
        // shared-endpoint case (§8) sees the previous merges.
        for &edge in edges {
            if self.edges.try_get(edge).is_none() {
                // A prior merge already consumed this edge (it was the
                // shared interior edge of a pair). Skip it silently — the
                // re-test-afresh contract treats an already-dissolved edge
                // as a no-op rather than an error.
                continue;
            }
            match self.dissolve_edges_classify(edge) {
                EdgeClass::ManifoldPair { face_a, face_b } => {
                    // n==2 `faces_join` is the JFKE merge of §3.1 / §6.4:
                    // it finds the shared edge, kills it and its two loops,
                    // and splices the two perimeters into one face.
                    let _ = self.faces_join(&[face_a, face_b]);
                }
                EdgeClass::NoOp => {}
            }
        }
    }

    /// §3 classification of a single input edge against the current
    /// topology. Returns `ManifoldPair` only when the radial cycle has
    /// exactly two loops belonging to two distinct faces; every other shape
    /// (wire, boundary, self-radial, non-manifold) is `NoOp`.
    fn dissolve_edges_classify(&self, edge: EdgeKey) -> EdgeClass {
        let mut radial = self.radial_loops(edge);
        let Some(first_loop) = radial.next(self) else {
            // Radial cycle empty: wire edge (§3 row 1).
            return EdgeClass::NoOp;
        };
        let Some(second_loop) = radial.next(self) else {
            // Radial cycle length 1: boundary edge (§3 row 2).
            return EdgeClass::NoOp;
        };
        if radial.next(self).is_some() {
            // Radial cycle length ≥ 3: non-manifold edge (§3 row 5).
            return EdgeClass::NoOp;
        }
        let face_a = self.loops.get(first_loop).face;
        let face_b = self.loops.get(second_loop).face;
        if face_a == face_b {
            // Radial cycle length 2 but both loops on the same face: a
            // self-radial fin (§3 row 4).
            return EdgeClass::NoOp;
        }
        EdgeClass::ManifoldPair { face_a, face_b }
    }
}

#[cfg(test)]
mod tests {
    use glam::Vec3;

    use crate::loopmesh::*;

    /// Build a 3x3 face grid of quads on the unit lattice. Returns
    /// `(verts[16], faces[9])` where `verts[r * 4 + c]` is the lattice
    /// point at column `c`, row `r`, and `faces[r * 3 + c]` is the quad
    /// whose lower-left corner is `verts[r * 4 + c]`. CCW winding.
    fn build_grid_3x3(mesh: &mut LoopMesh) -> (Vec<VertexKey>, Vec<FaceKey>) {
        let mut verts: Vec<VertexKey> = Vec::with_capacity(16);
        for r in 0..4 {
            for c in 0..4 {
                verts.push(mesh.create_vertex(Vec3::new(c as f32, r as f32, 0.0)));
            }
        }
        let mut faces: Vec<FaceKey> = Vec::new();
        for r in 0..3 {
            for c in 0..3 {
                let v_bl = verts[r * 4 + c];
                let v_br = verts[r * 4 + c + 1];
                let v_tr = verts[(r + 1) * 4 + c + 1];
                let v_tl = verts[(r + 1) * 4 + c];
                let e_b = mesh.create_edge([v_bl, v_br], EdgeCreateFlags::NO_DUPLICATE);
                let e_r = mesh.create_edge([v_br, v_tr], EdgeCreateFlags::NO_DUPLICATE);
                let e_t = mesh.create_edge([v_tr, v_tl], EdgeCreateFlags::NO_DUPLICATE);
                let e_l = mesh.create_edge([v_tl, v_bl], EdgeCreateFlags::NO_DUPLICATE);
                faces.push(mesh.face_create(
                    &[v_bl, v_br, v_tr, v_tl],
                    &[e_b, e_r, e_t, e_l],
                    FaceCreateFlags::NONE,
                ));
            }
        }
        (verts, faces)
    }

    /// Two triangles sharing the diagonal edge {v1, v2}. Square corners are
    /// v0=(0,0), v1=(1,0), v2=(0,1), v3=(1,1); the diagonal runs v1→v2.
    /// Triangle A = (v0, v1, v2); triangle B = (v1, v3, v2).
    fn build_two_tris_shared_diagonal(
        mesh: &mut LoopMesh,
    ) -> (Vec<VertexKey>, EdgeKey, Vec<FaceKey>) {
        let v: Vec<VertexKey> = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)]
            .into_iter()
            .map(|(x, y)| mesh.create_vertex(Vec3::new(x, y, 0.0)))
            .collect();
        let e_01 = mesh.create_edge([v[0], v[1]], EdgeCreateFlags::NO_DUPLICATE);
        let e_12 = mesh.create_edge([v[1], v[2]], EdgeCreateFlags::NO_DUPLICATE);
        let e_20 = mesh.create_edge([v[2], v[0]], EdgeCreateFlags::NO_DUPLICATE);
        let e_13 = mesh.create_edge([v[1], v[3]], EdgeCreateFlags::NO_DUPLICATE);
        let e_32 = mesh.create_edge([v[3], v[2]], EdgeCreateFlags::NO_DUPLICATE);
        let f_a = mesh.face_create(
            &[v[0], v[1], v[2]],
            &[e_01, e_12, e_20],
            FaceCreateFlags::NONE,
        );
        let f_b = mesh.face_create(
            &[v[1], v[3], v[2]],
            &[e_13, e_32, e_12],
            FaceCreateFlags::NONE,
        );
        (v, e_12, vec![f_a, f_b])
    }

    /// Single triangle (3 boundary edges) at the origin.
    fn build_triangle(mesh: &mut LoopMesh) -> (Vec<VertexKey>, Vec<EdgeKey>, FaceKey) {
        let v: Vec<VertexKey> = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]
            .into_iter()
            .map(|(x, y)| mesh.create_vertex(Vec3::new(x, y, 0.0)))
            .collect();
        let e_01 = mesh.create_edge([v[0], v[1]], EdgeCreateFlags::NO_DUPLICATE);
        let e_12 = mesh.create_edge([v[1], v[2]], EdgeCreateFlags::NO_DUPLICATE);
        let e_20 = mesh.create_edge([v[2], v[0]], EdgeCreateFlags::NO_DUPLICATE);
        let f = mesh.face_create(&[v[0], v[1], v[2]], &[e_01, e_12, e_20], FaceCreateFlags::NONE);
        (v, vec![e_01, e_12, e_20], f)
    }

    /// `n` triangles all sharing the spine edge {v0, v1} (a radial-`n`
    /// non-manifold "book"). Triangle `i` is (v0, v1, apex_i).
    fn build_nonmanifold_book(mesh: &mut LoopMesh, n: usize) -> (VertexKey, VertexKey, EdgeKey) {
        let v0 = mesh.create_vertex(Vec3::new(0.0, 0.0, 0.0));
        let v1 = mesh.create_vertex(Vec3::new(1.0, 0.0, 0.0));
        let spine = mesh.create_edge([v0, v1], EdgeCreateFlags::NO_DUPLICATE);
        for i in 0..n {
            let angle = std::f32::consts::PI * (i as f32 + 1.0) / (n as f32 + 1.0);
            let apex = mesh.create_vertex(Vec3::new(0.5, angle.cos(), angle.sin()));
            let e_1a = mesh.create_edge([v1, apex], EdgeCreateFlags::NO_DUPLICATE);
            let e_a0 = mesh.create_edge([apex, v0], EdgeCreateFlags::NO_DUPLICATE);
            mesh.face_create(&[v0, v1, apex], &[spine, e_1a, e_a0], FaceCreateFlags::NONE);
        }
        (v0, v1, spine)
    }

    #[test]
    fn dissolve_edges_interior_edge_between_two_quads() {
        // Default-parameter manifold face-pair (§3 row 3, §7 row 3): the
        // shared edge between two grid quads dissolves, merging the two
        // quads into a hexagon. -1 E, -1 F, 0 V.
        let mut mesh = LoopMesh::new();
        let (verts, _faces) = build_grid_3x3(&mut mesh);

        let v0 = mesh.totvert();
        let e0 = mesh.totedge();
        let f0 = mesh.totface();
        assert_eq!((v0, e0, f0), (16, 24, 9));

        // Vertical interior edge between faces[0] and faces[1]: verts[1]-verts[5].
        let edge = mesh
            .query_edge_exists(verts[1], verts[5])
            .expect("interior edge {1,5} present");

        mesh.dissolve_edges(&[edge], Default::default());

        assert_eq!(mesh.totvert(), v0, "0 verts removed (§7 row 3)");
        assert_eq!(mesh.totedge(), e0 - 1, "-1 edge (§7 row 3)");
        assert_eq!(mesh.totface(), f0 - 1, "-1 face (§7 row 3)");
        assert!(
            mesh.query_edge_exists(verts[1], verts[5]).is_none(),
            "the dissolved edge is gone"
        );
        // 4 + 4 corners minus the two consumed by the shared edge = 6.
        let merged = mesh
            .face_keys()
            .into_iter()
            .find(|&f| mesh.face_len(f) == 6)
            .expect("a hexagonal merged face exists");
        assert_eq!(mesh.face_len(merged), 6);
        mesh.validate_topology();
    }

    #[test]
    fn dissolve_edges_diagonal_of_two_triangles_makes_quad() {
        // Inverse-of-triangulate base case (§7 row 3 on the smallest
        // non-trivial face pair): two triangles sharing a diagonal merge
        // into a quad. Verts unchanged under default parameters.
        let mut mesh = LoopMesh::new();
        let (_verts, diagonal, _faces) = build_two_tris_shared_diagonal(&mut mesh);

        let v0 = mesh.totvert();
        let e0 = mesh.totedge();
        let f0 = mesh.totface();
        assert_eq!((v0, e0, f0), (4, 5, 2));

        mesh.dissolve_edges(&[diagonal], Default::default());

        assert_eq!(mesh.totvert(), v0, "0 verts removed");
        assert_eq!(mesh.totedge(), e0 - 1, "-1 edge (the diagonal)");
        assert_eq!(mesh.totface(), f0 - 1, "-1 face (two tris → one quad)");

        let merged = mesh.face_keys();
        assert_eq!(merged.len(), 1);
        assert_eq!(mesh.face_len(merged[0]), 4, "3 + 3 - 2 = 4 corners");
        mesh.validate_topology();
    }

    #[test]
    fn dissolve_edges_wire_edge_is_noop() {
        // §3 row 1: a wire edge (empty radial cycle) fails the face-pair
        // test; the merge is skipped and the edge survives.
        let mut mesh = LoopMesh::new();
        let (_verts, _faces) = build_grid_3x3(&mut mesh);
        // Add a free-standing wire edge between two fresh verts.
        let w0 = mesh.create_vertex(Vec3::new(10.0, 0.0, 0.0));
        let w1 = mesh.create_vertex(Vec3::new(11.0, 0.0, 0.0));
        let wire = mesh.create_edge([w0, w1], EdgeCreateFlags::NO_DUPLICATE);

        let v0 = mesh.totvert();
        let e0 = mesh.totedge();
        let f0 = mesh.totface();

        mesh.dissolve_edges(&[wire], Default::default());

        assert_eq!((mesh.totvert(), mesh.totedge(), mesh.totface()), (v0, e0, f0));
        assert!(
            mesh.query_edge_exists(w0, w1).is_some(),
            "wire edge survives (BMesh does not kill wire edges directly)"
        );
        mesh.validate_topology();
    }

    #[test]
    fn dissolve_edges_boundary_edge_is_noop() {
        // §3 row 2: a boundary edge (radial length 1) fails the face-pair
        // test; the merge is skipped and the edge survives.
        let mut mesh = LoopMesh::new();
        let (verts, edges, _f) = build_triangle(&mut mesh);

        let v0 = mesh.totvert();
        let e0 = mesh.totedge();
        let f0 = mesh.totface();

        mesh.dissolve_edges(&[edges[0]], Default::default());

        assert_eq!((mesh.totvert(), mesh.totedge(), mesh.totface()), (v0, e0, f0));
        assert!(
            mesh.query_edge_exists(verts[0], verts[1]).is_some(),
            "boundary edge survives"
        );
        mesh.validate_topology();
    }

    #[test]
    fn dissolve_edges_two_disjoint_interior_edges() {
        // Multi-edge input, disjoint pairs (§8 first bullet): two
        // non-adjacent interior edges, each between a distinct quad pair.
        // The merges are topologically independent: -2 E, -2 F, 0 V.
        let mut mesh = LoopMesh::new();
        let (verts, _faces) = build_grid_3x3(&mut mesh);

        let v0 = mesh.totvert();
        let e0 = mesh.totedge();
        let f0 = mesh.totface();

        // Edge {1,5} sits between faces[0]/faces[1] (bottom row).
        // Edge {10,14} sits between faces[6]/faces[7] (top row) — disjoint.
        let edge_a = mesh.query_edge_exists(verts[1], verts[5]).expect("edge {1,5}");
        let edge_b = mesh.query_edge_exists(verts[10], verts[14]).expect("edge {10,14}");

        mesh.dissolve_edges(&[edge_a, edge_b], Default::default());

        assert_eq!(mesh.totvert(), v0, "0 verts removed");
        assert_eq!(mesh.totedge(), e0 - 2, "-2 edges");
        assert_eq!(mesh.totface(), f0 - 2, "-2 faces");
        mesh.validate_topology();
    }

    #[test]
    fn dissolve_edges_two_disjoint_interior_edges_order_independent() {
        // Same disjoint-pair input as above but reversed order: the result
        // is identical, pinning the order-independence claim of §8.
        let mut mesh = LoopMesh::new();
        let (verts, _faces) = build_grid_3x3(&mut mesh);
        let v0 = mesh.totvert();
        let e0 = mesh.totedge();
        let f0 = mesh.totface();

        let edge_a = mesh.query_edge_exists(verts[1], verts[5]).expect("edge {1,5}");
        let edge_b = mesh.query_edge_exists(verts[10], verts[14]).expect("edge {10,14}");

        mesh.dissolve_edges(&[edge_b, edge_a], Default::default());

        assert_eq!(mesh.totvert(), v0);
        assert_eq!(mesh.totedge(), e0 - 2);
        assert_eq!(mesh.totface(), f0 - 2);
        mesh.validate_topology();
    }

    #[test]
    fn dissolve_edges_two_edges_sharing_endpoint() {
        // Multi-edge input, shared endpoint (§6.4 / §8 "two adjacent input
        // edges sharing a vert"): two interior edges meeting at the centre
        // vert of the grid. The first merge reshapes the face pair the
        // second sees; the per-edge dispatch re-classifies afresh, so both
        // merges run and consume one edge / one face each.
        let mut mesh = LoopMesh::new();
        let (verts, _faces) = build_grid_3x3(&mut mesh);
        let v0 = mesh.totvert();
        let e0 = mesh.totedge();
        let f0 = mesh.totface();

        // Centre vert is verts[5]. Two interior edges meeting at it:
        // {1,5} (between faces[0]/faces[1]) and {5,6} (between faces[1]/faces[2]).
        let edge_a = mesh.query_edge_exists(verts[1], verts[5]).expect("edge {1,5}");
        let edge_b = mesh.query_edge_exists(verts[5], verts[6]).expect("edge {5,6}");

        mesh.dissolve_edges(&[edge_a, edge_b], Default::default());

        // Two distinct face pairs each merge: faces[0]+faces[1], then the
        // result + faces[2] is NOT what happens here — edge_b's faces are
        // faces[1] (now merged with faces[0]) and faces[2]. The second
        // merge joins that merged face with faces[2]. Net over both edges:
        // -2 E, -2 F, 0 V.
        assert_eq!(mesh.totvert(), v0, "0 verts removed under default params");
        assert_eq!(mesh.totedge(), e0 - 2, "-2 edges (both dissolved)");
        assert_eq!(mesh.totface(), f0 - 2, "-2 faces (both merges ran)");
        assert!(mesh.query_edge_exists(verts[1], verts[5]).is_none());
        assert!(mesh.query_edge_exists(verts[5], verts[6]).is_none());
        mesh.validate_topology();
    }

    #[test]
    fn dissolve_edges_non_manifold_radial_3_is_noop() {
        // §3 row 5: three faces share the spine edge (radial length 3);
        // the face-pair test fails and the merge is skipped.
        let mut mesh = LoopMesh::new();
        let (s0, s1, spine) = build_nonmanifold_book(&mut mesh, 3);

        let v0 = mesh.totvert();
        let e0 = mesh.totedge();
        let f0 = mesh.totface();

        mesh.dissolve_edges(&[spine], Default::default());

        assert_eq!((mesh.totvert(), mesh.totedge(), mesh.totface()), (v0, e0, f0));
        assert!(
            mesh.query_edge_exists(s0, s1).is_some(),
            "non-manifold spine edge survives"
        );
        mesh.validate_topology();
    }

    #[test]
    fn dissolve_edges_closed_loop_boundary_perimeter_is_noop() {
        // §8 "an entire closed loop of input edges", boundary-perimeter
        // sub-case: the three edges of a lone triangle are all boundary
        // (radial length 1), so every per-edge dispatch is a no-op
        // regardless of iteration order.
        let mut mesh = LoopMesh::new();
        let (_verts, edges, _f) = build_triangle(&mut mesh);
        let v0 = mesh.totvert();
        let e0 = mesh.totedge();
        let f0 = mesh.totface();

        mesh.dissolve_edges(&edges, Default::default());

        assert_eq!((mesh.totvert(), mesh.totedge(), mesh.totface()), (v0, e0, f0));
        mesh.validate_topology();
    }

    #[test]
    fn dissolve_edges_empty_input_is_noop() {
        let mut mesh = LoopMesh::new();
        let (_verts, _faces) = build_grid_3x3(&mut mesh);
        let v0 = mesh.totvert();
        let e0 = mesh.totedge();
        let f0 = mesh.totface();

        mesh.dissolve_edges(&[], Default::default());

        assert_eq!((mesh.totvert(), mesh.totedge(), mesh.totface()), (v0, e0, f0));
        mesh.validate_topology();
    }
}
