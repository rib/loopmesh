use std::collections::HashSet;

use glam::Vec3;

use crate::loopmesh::*;

/// Strictly-positive convexity threshold (CONNECT_VERTS_CONCAVE.md §4.3): a
/// merged corner that the join would leave exactly straight (collinear) is
/// treated as not-safe-to-merge, so the diagonal is kept rather than producing
/// a degenerate 180° corner.
const CONVEX_EPS: f32 = 1e-6;

impl LoopMesh {
    /// Decompose every concave face in `faces` into convex pieces by
    /// over-cutting (full beautify triangulation) then greedily merging back
    /// across each interior diagonal whose removal keeps the result convex.
    ///
    /// Follows `radial-mesh-guides/operators/CONNECT_VERTS_CONCAVE.md`:
    ///
    /// - §3 concavity gate: faces with `face_len <= 3` and fully-convex faces
    ///   are strict no-ops; only faces that are both larger than a triangle
    ///   and carry at least one reflex corner are decomposed.
    /// - §4.1 over-cut: [`Self::face_triangulate`] with the beautify quad and
    ///   n-gon strategies — the same diagonal-only triangulation the
    ///   `triangulate` row covers.
    /// - §4.2 ordering: interior diagonals are sorted reflex-pairs-last, then
    ///   longest-first within each reflex class.
    /// - §4.3 merge-back: a single greedy sweep that joins the two triangles
    ///   across a diagonal (JFKE, via [`Self::faces_join`]) iff the merged
    ///   polygon stays convex at both diagonal endpoints.
    ///
    /// Unlike the selection-driven [`Self::connect_verts`], the caller names
    /// only faces; the operator chooses every cut itself. It never creates or
    /// destroys a vertex.
    pub fn connect_verts_concave(&mut self, faces: &[FaceKey]) {
        self.connect_verts_concave_out(faces);
    }

    /// Capturing form of [`Self::connect_verts_concave`]: returns the
    /// `edges.out` buffer (CONNECT_VERTS_CONCAVE.md §1) — the interior diagonal
    /// edges that *survive* the merge-back sweep (the edges that actually
    /// divide the convex pieces). A diagonal the sweep merges away is killed by
    /// its JFKE and drops out, so the returned set is exactly the surviving
    /// dividers. The buffer is a set; enumeration order is not observable for
    /// parity (§1, §6).
    pub fn connect_verts_concave_out(&mut self, faces: &[FaceKey]) -> Vec<EdgeKey> {
        // §3.1 — the reflex tag is a transient per-vertex working flag,
        // consumed by the §4.2 ordering. One op-flag layer for the whole pass;
        // it is re-stamped per face immediately before that face is decomposed.
        let reflex_tag = self.push_vert_op_flag_layer();

        let mut surviving_dividers: HashSet<EdgeKey> = HashSet::new();
        for &face in faces {
            self.process_concave_face(face, reflex_tag, &mut surviving_dividers);
        }

        self.pop_vert_op_flag_layer(reflex_tag);
        surviving_dividers.into_iter().collect()
    }

    /// §3 — the per-face gate plus decomposition. Returns immediately for
    /// triangles and for fully-convex faces.
    fn process_concave_face(
        &mut self,
        face: FaceKey,
        reflex_tag: crate::op_flags::VertOpFlagLayer,
        surviving_dividers: &mut HashSet<EdgeKey>,
    ) {
        if self.face_len(face) <= 3 {
            return; // §3.2 — triangles can't be concave.
        }

        // §3.1 — the face normal is the single stable reference for every
        // corner the pass examines (the gate and the §4.3 merge test both read
        // it), captured once before triangulation.
        let face_normal = self.face_compute_normal(face);

        if !self.tag_reflex_corners(face, face_normal, reflex_tag) {
            return; // §3.1 — fully convex, never triangulated.
        }

        // §4.1 — perimeter snapshot: the edges of the original face. After the
        // over-cut, the *new* interior diagonals are exactly the edges joining
        // two original-face corners that are not in this set.
        let perimeter: HashSet<EdgeKey> = self.face_edge_keys(face).into_iter().collect();
        let original_corners: HashSet<VertexKey> =
            self.face_vert_keys(face).into_iter().collect();

        // §4.1 — fully triangulate in place with the beautify triangulator.
        self.face_triangulate(face, QuadMethod::Beauty, NgonMethod::Beauty);

        let mut diagonals = self.collect_interior_diagonals(&original_corners, &perimeter);

        // §4.2 — reflex-pairs-last, then longest-first within each class.
        diagonals.sort_by(|&a, &b| {
            self.diagonal_order_key(a, reflex_tag)
                .partial_cmp(&self.diagonal_order_key(b, reflex_tag))
                .expect("connect_verts_concave: non-finite diagonal length")
        });

        // §4.3 — single greedy merge-back sweep.
        for diagonal in diagonals {
            if self.try_merge_diagonal(diagonal, face_normal) {
                continue; // merged away — `diagonal` is gone from the live mesh.
            }
            surviving_dividers.insert(diagonal);
        }
    }

    /// §3.1 — tag each reflex corner's vert in `reflex_tag` and clear the tag
    /// on convex corners. Returns true iff at least one corner was reflex (the
    /// face is concave).
    fn tag_reflex_corners(
        &mut self,
        face: FaceKey,
        face_normal: Vec3,
        reflex_tag: crate::op_flags::VertOpFlagLayer,
    ) -> bool {
        let cycle = self.face_vert_keys(face);
        let n = cycle.len();
        let mut is_concave = false;
        for i in 0..n {
            let prev = self.vertex_position(cycle[(i + n - 1) % n]);
            let here = self.vertex_position(cycle[i]);
            let next = self.vertex_position(cycle[(i + 1) % n]);
            if corner_is_convex(prev, here, next, face_normal) {
                self.vert_op_flag_clear(reflex_tag, cycle[i], 0);
            } else {
                self.vert_op_flag_set(reflex_tag, cycle[i], 0);
                is_concave = true;
            }
        }
        is_concave
    }

    /// §4.1 — the interior diagonals introduced by the over-cut: live edges
    /// both of whose endpoints are original-face corners and which were not on
    /// the original perimeter.
    fn collect_interior_diagonals(
        &self,
        original_corners: &HashSet<VertexKey>,
        perimeter: &HashSet<EdgeKey>,
    ) -> Vec<EdgeKey> {
        let mut diagonals: HashSet<EdgeKey> = HashSet::new();
        for &corner in original_corners {
            for edge in self.vert_edge_keys(corner) {
                if perimeter.contains(&edge) {
                    continue;
                }
                let [a, b] = self.edge_vertices(edge);
                if original_corners.contains(&a) && original_corners.contains(&b) {
                    diagonals.insert(edge);
                }
            }
        }
        diagonals.into_iter().collect()
    }

    /// §4.2 — the two-key sort: `(both_reflex, -len_sq)` compared
    /// lexicographically ascending → reflex pairs sort last, longest first
    /// within each reflex class.
    fn diagonal_order_key(
        &self,
        edge: EdgeKey,
        reflex_tag: crate::op_flags::VertOpFlagLayer,
    ) -> (bool, f32) {
        let [a, b] = self.edge_vertices(edge);
        let both_reflex =
            self.vert_op_flag_test(reflex_tag, a, 0) && self.vert_op_flag_test(reflex_tag, b, 0);
        let len_sq = (self.vertex_position(b) - self.vertex_position(a)).length_squared();
        (both_reflex, -len_sq)
    }

    /// §4.3 — attempt the merge-back across one diagonal. Returns true iff the
    /// join was performed (the diagonal and one of its triangles were
    /// consumed); false iff the diagonal is kept (not a clean interior edge any
    /// more, or merging would leave a reflex corner).
    fn try_merge_diagonal(&mut self, diagonal: EdgeKey, face_normal: Vec3) -> bool {
        // §4.3 — must still be a manifold radial-2 interior edge. A prior merge
        // in this sweep may have consumed one of its faces.
        let adjacent = self.edge_face_keys(diagonal);
        if adjacent.len() != 2 {
            return false;
        }
        let face_a = adjacent[0];
        let face_b = adjacent[1];
        if face_a == face_b {
            return false;
        }

        if !self.join_keeps_convex(diagonal, face_a, face_b, face_normal) {
            return false;
        }

        // JFKE — fuse the two triangles, killing the diagonal.
        self.faces_join(&[face_a, face_b])
            .expect("connect_verts_concave: faces_join failed on a radial-2 interior diagonal");
        true
    }

    /// §4.3 — would fusing the two triangles across `diagonal` keep the merged
    /// polygon convex at both diagonal endpoints? Reconstructs the merged
    /// quad's cycle and runs the §3.1 corner test (against the captured face
    /// normal) at each endpoint, so "convex" means the same sign here as in the
    /// gate.
    fn join_keeps_convex(
        &self,
        diagonal: EdgeKey,
        face_a: FaceKey,
        face_b: FaceKey,
        face_normal: Vec3,
    ) -> bool {
        let [p, q] = self.edge_vertices(diagonal);
        let tri_a = self.face_vert_keys(face_a);
        let tri_b = self.face_vert_keys(face_b);
        if tri_a.len() != 3 || tri_b.len() != 3 {
            return false;
        }
        // Triangle A's far corner is implicit in its own cycle; only B's far
        // corner is spliced in to build the merged quad.
        let Some(far_b) = third_corner(&tri_b, p, q) else {
            return false;
        };

        // Merged quad cycle: replace the diagonal step (p→q or q→p) on
        // triangle A's cycle with a detour through far_b. The result walks the
        // merged polygon in A's winding, so the §3.1 prev/next convention
        // applies directly.
        let quad = merged_quad_cycle(&tri_a, p, q, far_b);

        for &endpoint in &[p, q] {
            let i = quad
                .iter()
                .position(|&v| v == endpoint)
                .expect("connect_verts_concave: diagonal endpoint missing from merged quad");
            let n = quad.len();
            let prev = self.vertex_position(quad[(i + n - 1) % n]);
            let here = self.vertex_position(quad[i]);
            let next = self.vertex_position(quad[(i + 1) % n]);
            if !corner_is_convex(prev, here, next, face_normal) {
                return false;
            }
        }
        true
    }
}

/// §3.1 — signed convexity test. Directions leave the corner toward its cycle
/// neighbours; the corner's signed normal is `cross(to_next, to_prev)`, and the
/// corner is convex iff that agrees with the face normal beyond `CONVEX_EPS`.
/// The same factor order is used by the merge-back test (§4.3), so a convex
/// corner reliably yields the positive sign in both places.
fn corner_is_convex(prev: Vec3, here: Vec3, next: Vec3, face_normal: Vec3) -> bool {
    let to_prev = prev - here;
    let to_next = next - here;
    let corner_normal = to_next.cross(to_prev);
    corner_normal.dot(face_normal) > CONVEX_EPS
}

/// The one corner of triangle `tri` that is neither `p` nor `q`.
fn third_corner(tri: &[VertexKey], p: VertexKey, q: VertexKey) -> Option<VertexKey> {
    tri.iter().copied().find(|&v| v != p && v != q)
}

/// Build the merged quad's vertex cycle by walking triangle A's 3-cycle and,
/// at the step that traverses the shared diagonal (either p→q or q→p),
/// inserting the opposite triangle's far corner so the detour replaces the
/// diagonal. Walking A's winding keeps the cycle consistently wound.
fn merged_quad_cycle(
    tri_a: &[VertexKey],
    p: VertexKey,
    q: VertexKey,
    far_b: VertexKey,
) -> [VertexKey; 4] {
    let mut quad = [tri_a[0]; 4];
    let mut out = 0;
    for i in 0..3 {
        let cur = tri_a[i];
        let nxt = tri_a[(i + 1) % 3];
        quad[out] = cur;
        out += 1;
        let traverses_diagonal =
            (cur == p && nxt == q) || (cur == q && nxt == p);
        if traverses_diagonal {
            quad[out] = far_b;
            out += 1;
        }
    }
    debug_assert_eq!(out, 4);
    quad
}

#[cfg(test)]
mod tests {
    use glam::Vec3;

    use crate::loopmesh::test_support::create_face_from_verts;
    use crate::loopmesh::*;

    /// §5.1 — the shark-fin concave quad: `v0=(0,0)`, `v2=(2,0)` on the
    /// baseline; both `v1=(1,0.5)` and `v3=(1,1)` lifted above the `v0`-`v2`
    /// line, so corner 1 dents inward (reflex). The only legal triangulation
    /// is the 1-3 diagonal, which the merge-back must keep.
    fn concave_quad(mesh: &mut LoopMesh) -> (FaceKey, Vec<VertexKey>) {
        let coords = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.5, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
        ];
        let face = create_face_from_verts(mesh, &coords);
        let verts = mesh.face_vert_keys(face);
        (face, verts)
    }

    #[test]
    fn concave_quad_keeps_one_diagonal() {
        let mut mesh = LoopMesh::new();
        let (face, v) = concave_quad(&mut mesh);
        let v0 = mesh.totvert();
        let e0 = mesh.totedge();

        let created = mesh.connect_verts_concave_out(&[face]);

        // §5.1: 4 V (unchanged), 4 → 5 E (+1 surviving diagonal), 1 → 2 F.
        assert_eq!(mesh.totvert(), v0, "operator never creates or destroys verts");
        assert_eq!(mesh.totedge(), e0 + 1, "one surviving interior diagonal");
        assert_eq!(mesh.totface(), 2, "concave quad becomes two triangles");
        assert_eq!(created.len(), 1, "exactly one divider survives");
        // The kept diagonal is the 1-3 chord (the only legal one).
        let diag = mesh.query_edge_exists(v[1], v[3]).expect("1-3 diagonal survives");
        assert_eq!(created[0], diag);
        for f in mesh.face_keys() {
            assert_eq!(mesh.face_len(f), 3, "both survivors are triangles");
        }
        mesh.validate_topology();
    }

    #[test]
    fn convex_quad_is_noop() {
        // §5.2 — a plain convex quad: all corners convex, never triangulated.
        let mut mesh = LoopMesh::new();
        let coords = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        ];
        let face = create_face_from_verts(&mut mesh, &coords);
        let created = mesh.connect_verts_concave_out(&[face]);

        assert!(created.is_empty(), "convex quad yields no dividers");
        assert_eq!(mesh.totvert(), 4);
        assert_eq!(mesh.totedge(), 4);
        assert_eq!(mesh.totface(), 1, "convex quad untouched");
        assert_eq!(mesh.face_len(face), 4);
        mesh.validate_topology();
    }

    #[test]
    fn convex_ngon_is_noop() {
        // §5.2 scaled past the quad: a regular convex pentagon and hexagon are
        // both untouched by the concavity gate.
        for n in [5usize, 6] {
            let mut mesh = LoopMesh::new();
            let coords: Vec<Vec3> = (0..n)
                .map(|i| {
                    let theta = (i as f32) * std::f32::consts::TAU / (n as f32);
                    Vec3::new(theta.cos(), theta.sin(), 0.0)
                })
                .collect();
            let face = create_face_from_verts(&mut mesh, &coords);
            let created = mesh.connect_verts_concave_out(&[face]);

            assert!(created.is_empty(), "convex {n}-gon yields no dividers");
            assert_eq!(mesh.totface(), 1, "convex {n}-gon untouched");
            assert_eq!(mesh.face_len(face), n);
            mesh.validate_topology();
        }
    }

    #[test]
    fn triangle_is_noop() {
        // §3.2 — the `face_len <= 3` gate skips a triangle outright.
        let mut mesh = LoopMesh::new();
        let coords = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        ];
        let face = create_face_from_verts(&mut mesh, &coords);
        let created = mesh.connect_verts_concave_out(&[face]);

        assert!(created.is_empty());
        assert_eq!(mesh.totface(), 1);
        assert_eq!(mesh.face_len(face), 3);
        mesh.validate_topology();
    }

    #[test]
    fn concave_pentagon_multi_cut_convex_decomposition() {
        // §5.3 — reflex corner at index 2. The exact surviving-cut count is
        // triangulation + merge-order-dependent, so pin the structural
        // invariants the guide makes deterministic: 0 verts created, the
        // decomposition is a set of ≥2 convex pieces, and every surviving
        // piece is convex (no reflex corner remains).
        let mut mesh = LoopMesh::new();
        let coords = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
            Vec3::new(2.0, 2.0, 0.0),
            Vec3::new(0.0, 2.0, 0.0),
        ];
        let face = create_face_from_verts(&mut mesh, &coords);
        let v0 = mesh.totvert();

        let created = mesh.connect_verts_concave_out(&[face]);

        assert_eq!(mesh.totvert(), v0, "no verts created or destroyed");
        assert!(!created.is_empty(), "the reflex corner forces ≥1 surviving cut");
        let face_count = mesh.totface();
        assert!(face_count >= 2, "decomposed into ≥2 convex pieces");
        // §4 net delta: +(c-1) edges and +(c-1) faces for c pieces.
        assert_eq!(created.len(), face_count - 1, "c-1 dividers for c pieces");
        assert_eq!(mesh.totedge(), 5 + (face_count - 1), "+(c-1) interior edges");
        assert!(every_face_is_convex(&mesh), "every surviving piece is convex");
        mesh.validate_topology();
    }

    #[test]
    fn two_reflex_hexagon_decomposes_to_convex() {
        // §7 — a hexagon with two non-adjacent reflex corners, exercising the
        // §4.2 reflex-vs-reflex ordering (a single-reflex polygon can never
        // produce a both-endpoints-reflex diagonal). Pin the structural
        // invariant: every surviving piece is convex and no vert is created.
        let mut mesh = LoopMesh::new();
        // CCW hexagon with two inward notches at indices 1 and 4.
        let coords = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(2.0, 1.0, 0.0), // notch 1 (pulled inward)
            Vec3::new(4.0, 0.0, 0.0),
            Vec3::new(4.0, 3.0, 0.0),
            Vec3::new(2.0, 2.0, 0.0), // notch 4 (pulled inward)
            Vec3::new(0.0, 3.0, 0.0),
        ];
        let face = create_face_from_verts(&mut mesh, &coords);
        let v0 = mesh.totvert();

        let created = mesh.connect_verts_concave_out(&[face]);

        assert_eq!(mesh.totvert(), v0, "no verts created");
        assert!(!created.is_empty(), "two reflex corners force surviving cuts");
        let face_count = mesh.totface();
        assert!(face_count >= 2);
        assert_eq!(created.len(), face_count - 1, "c-1 dividers for c pieces");
        assert!(every_face_is_convex(&mesh), "every surviving piece is convex");
        mesh.validate_topology();
    }

    #[test]
    fn mixed_input_gates_each_face_independently() {
        // §6 — a concave quad and a convex quad in one call: the concave one
        // is decomposed, the convex one is left alone.
        let mut mesh = LoopMesh::new();
        let (concave, _cv) = concave_quad(&mut mesh);
        let convex_coords = [
            Vec3::new(5.0, 0.0, 0.0),
            Vec3::new(6.0, 0.0, 0.0),
            Vec3::new(6.0, 1.0, 0.0),
            Vec3::new(5.0, 1.0, 0.0),
        ];
        let convex = create_face_from_verts(&mut mesh, &convex_coords);

        let created = mesh.connect_verts_concave_out(&[concave, convex]);

        assert_eq!(created.len(), 1, "only the concave face is cut");
        assert_eq!(mesh.face_len(convex), 4, "convex face untouched");
        // The concave face split into two triangles → 3 faces total.
        assert_eq!(mesh.totface(), 3);
        mesh.validate_topology();
    }

    /// True iff every live face turns the same way at every corner relative to
    /// its own normal (i.e. is convex). Mirrors the §3.1 signed convexity test.
    fn every_face_is_convex(mesh: &LoopMesh) -> bool {
        for f in mesh.face_keys() {
            let cycle = mesh.face_vert_keys(f);
            let n = cycle.len();
            if n <= 3 {
                continue;
            }
            let normal = mesh.face_compute_normal(f);
            for i in 0..n {
                let prev = mesh.vertex_position(cycle[(i + n - 1) % n]);
                let here = mesh.vertex_position(cycle[i]);
                let next = mesh.vertex_position(cycle[(i + 1) % n]);
                let corner_normal = (next - here).cross(prev - here);
                // Allow exactly-straight (collinear) corners — the merge-back
                // keeps those un-merged, but a surviving piece may still carry
                // a collinear corner from the original perimeter.
                if corner_normal.dot(normal) < -1e-5 {
                    return false;
                }
            }
        }
        true
    }
}
