use glam::Vec2;

use crate::loopmesh::*;
use crate::triangulate;

/// Quad triangulation strategy for [`LoopMesh::face_triangulate`].
///
/// Algorithms in [`BMESH_POLYGON_TRIANGULATION.md`](../../../../radial-mesh-guides/BMESH_POLYGON_TRIANGULATION.md)
/// §6.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum QuadMethod {
    /// Always cut along the diagonal between the first and third loops in
    /// the face's cycle (§6.1).
    Fixed,
    /// Always cut along the diagonal between the second and fourth loops
    /// in the face's cycle (§6.1).
    Alternate,
    /// Cut along whichever diagonal is shorter; ties pick the 1–3 diagonal
    /// (§6.2).
    ShortEdge,
    /// Cut along whichever diagonal is longer; ties pick the 1–3 diagonal
    /// (§6.2).
    LongEdge,
    /// Pick whichever diagonal does not produce a flipped triangle; if
    /// both diagonals are valid, pick the one that scores better under
    /// the §8.2 area-over-perimeter rule (§6.3).
    Beauty,
}

/// N-gon triangulation strategy for [`LoopMesh::face_triangulate`].
///
/// Algorithms in [`BMESH_POLYGON_TRIANGULATION.md`](../../../../radial-mesh-guides/BMESH_POLYGON_TRIANGULATION.md)
/// §7.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum NgonMethod {
    /// Plain ear-clip (§7), no beautify refinement.
    EarClip,
    /// Ear-clip followed by the §8 area-over-perimeter diagonal-swap
    /// refinement.
    Beauty,
}

impl LoopMesh {
    /// Destructive triangulation: replace `face` with `face.len() - 2`
    /// triangles sharing the same verts. Returns immediately for triangles
    /// (already-triangulated).
    ///
    /// The source `face` is consumed: it is replaced by `n - 2` fresh
    /// triangle faces. Use [`Self::face_triangulate_collect_doubles`] when you
    /// need the duplicate-triangle report (§9.4); this entry point discards
    /// it.
    ///
    /// Face customdata: every new triangle inherits the source face's CD.
    ///
    /// Loop customdata: every new corner copies its CD from the source
    /// loop sitting at the same vertex.
    pub fn face_triangulate(
        &mut self,
        face: FaceKey,
        quad_method: QuadMethod,
        ngon_method: NgonMethod,
    ) {
        let _ = self.face_triangulate_collect_doubles(face, quad_method, ngon_method);
    }

    /// [`Self::face_triangulate`] that additionally returns the
    /// duplicate-triangle report described in
    /// [`BMESH_POLYGON_TRIANGULATION.md`](../../../../radial-mesh-guides/BMESH_POLYGON_TRIANGULATION.md)
    /// §9.4.
    ///
    /// Each new triangle is constructed with the edge-reuse surgery
    /// primitive: any side whose two verts already carry an edge (a wire
    /// diagonal, or an edge shared with a pre-existing face) reuses that
    /// edge instead of minting a parallel duplicate. After each triangle is
    /// built the operator walks the radial of its first loop's edge; if a
    /// sibling triangle on that radial closes on the same third vertex, the
    /// new triangle coincides with a pre-existing one and is appended to the
    /// returned list. The operator itself does **not** delete the
    /// duplicates — the caller decides.
    ///
    /// Returns an empty list for triangle input (`n < 4`) and for any
    /// triangulation that introduces no coincident triangle (the common
    /// case, including the wire-diagonal-reuse scenario, which reuses the
    /// wire but produces no duplicate).
    pub fn face_triangulate_collect_doubles(
        &mut self,
        face: FaceKey,
        quad_method: QuadMethod,
        ngon_method: NgonMethod,
    ) -> Vec<FaceKey> {
        let verts = self.face_vert_keys(face);
        let n = verts.len();
        if n < 4 {
            return Vec::new();
        }

        let triples: Vec<[u32; 3]> = if n == 4 {
            let use_13 = match quad_method {
                QuadMethod::Fixed => false,
                QuadMethod::Alternate => true,
                QuadMethod::ShortEdge => {
                    let d_02_sq = (self.vertex_position(verts[2]) - self.vertex_position(verts[0]))
                        .length_squared();
                    let d_13_sq = (self.vertex_position(verts[3]) - self.vertex_position(verts[1]))
                        .length_squared();
                    d_13_sq <= d_02_sq
                }
                QuadMethod::LongEdge => {
                    let d_02_sq = (self.vertex_position(verts[2]) - self.vertex_position(verts[0]))
                        .length_squared();
                    let d_13_sq = (self.vertex_position(verts[3]) - self.vertex_position(verts[1]))
                        .length_squared();
                    d_13_sq >= d_02_sq
                }
                QuadMethod::Beauty => {
                    let p0 = self.vertex_position(verts[0]);
                    let p1 = self.vertex_position(verts[1]);
                    let p2 = self.vertex_position(verts[2]);
                    let p3 = self.vertex_position(verts[3]);
                    let flip = triangulate::is_quad_flip(p0, p1, p2, p3);
                    if (flip & 1) != 0 {
                        // 0-2 flips → must use 1-3.
                        true
                    } else if (flip & 2) != 0 {
                        // 1-3 flips → must use 0-2.
                        false
                    } else {
                        // Neither flips: §6.3 tie-break keeps 0-2 only on
                        // strictly positive cost, so ties (cost == 0) and
                        // negatives both fall through to the 1-3 diagonal.
                        triangulate::beautify_quad_cost(p0, p1, p2, p3, false) <= 0.0
                    }
                }
            };
            if use_13 {
                vec![[1, 2, 3], [1, 3, 0]]
            } else {
                vec![[0, 1, 2], [0, 2, 3]]
            }
        } else {
            let normal = self.face_compute_normal(face);
            let (u, w) = triangulate::axis_dominant_basis_negated(normal);
            let proj: Vec<Vec2> = verts
                .iter()
                .map(|&v| {
                    let p = self.vertex_position(v);
                    Vec2::new(u.dot(p), w.dot(p))
                })
                .collect();

            let mut tris: Vec<[u32; 3]> = Vec::with_capacity(n - 2);
            triangulate::ear_clip(&proj, 1, &mut tris);
            debug_assert_eq!(tris.len(), n - 2);

            // NgonMethod::Beauty: the §8 diagonal-swap refinement rewrites
            // `tris` in place to the post-beautify triangulation (§8.4
            // reuses the ear-clip projection). We construct directly from
            // the final triples — no replay against the live mesh needed.
            if matches!(ngon_method, NgonMethod::Beauty) {
                triangulate::beautify_polygon(&proj, &mut tris);
            }
            tris
        };

        self.construct_triangulation(face, &verts, &triples)
    }

    /// Build the triangulation's faces with the §9.4 edge-reuse surgery and
    /// run the radial-walk duplicate check on each. `triples` indexes into
    /// `verts` (the source face's perimeter ring). The source face is killed
    /// once every triangle has been created and has inherited its CD.
    fn construct_triangulation(
        &mut self,
        source_face: FaceKey,
        verts: &[VertexKey],
        triples: &[[u32; 3]],
    ) -> Vec<FaceKey> {
        // Capture the source loop at each perimeter vertex for loop-CD
        // inheritance: every new corner copies from the same-vertex source
        // corner (§9.4). The source loops stay live until the kill below.
        let do_loop_cd = !self.loop_attribs.is_empty();
        let mut vert_src_loop: Vec<(VertexKey, LoopKey)> = Vec::new();
        if do_loop_cd {
            vert_src_loop.reserve(verts.len());
            let mut it = self.face_loops(source_face);
            while let Some(l) = it.next(self) {
                vert_src_loop.push((self.loop_vertex(l), l));
            }
        }

        let mut doubles: Vec<FaceKey> = Vec::new();
        for triple in triples {
            let tri_verts = [
                verts[triple[0] as usize],
                verts[triple[1] as usize],
                verts[triple[2] as usize],
            ];
            // Edge-reuse surgery: `face_create_verts` mints each side with
            // NO_DUPLICATE, so a pre-existing wire/shared edge between two
            // tri verts is reused rather than duplicated. FaceCreateFlags
            // stays NONE so a coincident face is genuinely created (and thus
            // discoverable by the radial walk) rather than collapsed.
            let new_face = self.face_create_verts(&tri_verts, FaceCreateFlags::NONE);
            self.face_attribs
                .copy_into(new_face.as_usize(), source_face.as_usize());

            if do_loop_cd {
                let first = self.face_first_loop(new_face);
                let mut corner = first;
                loop {
                    let v = self.loop_vertex(corner);
                    if let Some(&(_, src)) = vert_src_loop.iter().find(|(vv, _)| *vv == v) {
                        self.loop_attribs.copy_into(corner.as_usize(), src.as_usize());
                    }
                    corner = self.loop_next(corner);
                    if corner == first {
                        break;
                    }
                }
            }

            if self.triangle_coincides_on_radial(new_face) {
                doubles.push(new_face);
            }
        }

        self.face_kill(source_face);
        doubles
    }

    /// §9.4 detection check: walk the radial of `new_face`'s first loop's
    /// edge and report whether a sibling triangle closes on the same third
    /// vertex (i.e. shares all three verts with `new_face`). The walk stops
    /// at the first match.
    fn triangle_coincides_on_radial(&self, new_face: FaceKey) -> bool {
        let first_loop = self.face_first_loop(new_face);
        let first_edge = self.loop_edge(first_loop);
        // In a triangle the loop sitting on an edge has its previous loop at
        // the third (opposite) vertex.
        let third_vert_new = self.loop_vertex(self.loop_prev(first_loop));

        let mut it = self.radial_loops(first_edge);
        while let Some(sibling) = it.next(self) {
            if sibling == first_loop {
                continue;
            }
            let sibling_face = self.loop_face(sibling);
            if self.face_len(sibling_face) == 3 {
                let third_vert_sibling = self.loop_vertex(self.loop_prev(sibling));
                if third_vert_sibling == third_vert_new {
                    return true;
                }
            }
        }
        false
    }

    /// Bulk wrapper around [`Self::face_triangulate`] that filters by
    /// minimum vertex count. Iterates over `faces` and triangulates each
    /// one whose `face_len()` is at least `min_vertices`.
    ///
    /// The caller passes the explicit subset of faces to triangulate;
    /// there is no separate "tag" channel — the slice *is* the selection.
    ///
    /// Pass `min_vertices = 4` to skip triangles (which already require
    /// no work); passing `0` or `3` makes the filter a no-op for
    /// triangle inputs, where [`Self::face_triangulate`] itself
    /// short-circuits.
    pub fn mesh_triangulate(
        &mut self,
        faces: &[FaceKey],
        min_vertices: usize,
        quad_method: QuadMethod,
        ngon_method: NgonMethod,
    ) {
        for &face in faces {
            if self.face_len(face) >= min_vertices {
                self.face_triangulate(face, quad_method, ngon_method);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use glam::Vec3;

    use super::{NgonMethod, QuadMethod};
    use crate::loopmesh::test_support::create_face_from_verts;
    use crate::loopmesh::*;

    #[test]
    fn triangle_input_is_no_op() {
        let mut mesh = LoopMesh::new();
        let verts = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        ];
        let face = create_face_from_verts(&mut mesh, &verts);
        let face_count_before = mesh.face_keys().len();
        mesh.face_triangulate(face, QuadMethod::Fixed, NgonMethod::EarClip);
        assert_eq!(mesh.face_keys().len(), face_count_before);
        assert_eq!(mesh.face_len(face), 3);
    }

    #[test]
    fn quad_fixed_splits_along_first_diagonal() {
        let mut mesh = LoopMesh::new();
        let verts = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        ];
        let face = create_face_from_verts(&mut mesh, &verts);
        let fv = mesh.face_vert_keys(face);
        mesh.face_triangulate(face, QuadMethod::Fixed, NgonMethod::EarClip);
        assert_eq!(mesh.face_keys().len(), 2);
        for f in mesh.face_keys() {
            assert_eq!(mesh.face_len(f), 3);
        }
        let edges_at_v0 = mesh.vert_edge_keys(fv[0]).len();
        let edges_at_v2 = mesh.vert_edge_keys(fv[2]).len();
        assert!(edges_at_v0 >= 3 && edges_at_v2 >= 3);
    }

    #[test]
    fn pentagon_emits_three_triangles() {
        let mut mesh = LoopMesh::new();
        let verts: Vec<Vec3> = (0..5)
            .map(|i| {
                let theta = (i as f32) * std::f32::consts::TAU / 5.0;
                Vec3::new(theta.cos(), theta.sin(), 0.0)
            })
            .collect();
        let face = create_face_from_verts(&mut mesh, &verts);
        mesh.face_triangulate(face, QuadMethod::Fixed, NgonMethod::EarClip);
        assert_eq!(mesh.face_keys().len(), 3);
        for f in mesh.face_keys() {
            assert_eq!(mesh.face_len(f), 3);
        }
    }

    /// Build a unit quad `[0,1,2,3]` from explicit vertex keys so callers can
    /// share those verts with extra wire/face geometry across the diagonal.
    fn quad_verts(mesh: &mut LoopMesh) -> [VertexKey; 4] {
        [
            mesh.create_vertex(Vec3::new(0.0, 0.0, 0.0)),
            mesh.create_vertex(Vec3::new(1.0, 0.0, 0.0)),
            mesh.create_vertex(Vec3::new(1.0, 1.0, 0.0)),
            mesh.create_vertex(Vec3::new(0.0, 1.0, 0.0)),
        ]
    }

    /// §9.4 wire-edge-as-diagonal arm: a wire edge across the `{0,2}` diagonal
    /// must be *reused* by the Fixed (0-2) triangulation — promoted from a
    /// 0-loop wire to a shared internal edge — rather than duplicated, and no
    /// duplicate triangle is reported.
    #[test]
    fn quad_wire_diagonal_is_reused_no_double() {
        let mut mesh = LoopMesh::new();
        let v = quad_verts(&mut mesh);
        let face = mesh.face_create_verts(&v, FaceCreateFlags::NONE);
        // Wire diagonal across {0,2}: 4 perimeter edges + 1 wire == 5.
        mesh.create_edge([v[0], v[2]], EdgeCreateFlags::NONE);
        assert_eq!(mesh.edge_count(), 5);

        let doubles =
            mesh.face_triangulate_collect_doubles(face, QuadMethod::Fixed, NgonMethod::EarClip);

        assert!(doubles.is_empty());
        // The wire was reused, not duplicated: still 5 edges, not 6.
        assert_eq!(mesh.edge_count(), 5);
        assert_eq!(mesh.face_keys().len(), 2);
    }

    /// §9.4 coincident-tri arm: a pre-existing triangle `[0,1,2]` sharing the
    /// quad's verts means the Fixed (0-2) triangulation's first new triangle
    /// coincides with it. The radial walk must fire and report exactly that
    /// `{0,1,2}` triangle as a duplicate; the new-face count is unaffected.
    #[test]
    fn quad_coincident_triangle_detected_as_double() {
        let mut mesh = LoopMesh::new();
        let v = quad_verts(&mut mesh);
        let quad = mesh.face_create_verts(&v, FaceCreateFlags::NONE);
        // Pre-existing triangle on the same {0,1,2} verts, sharing the quad's
        // (0,1) and (1,2) perimeter edges and minting the {0,2} diagonal.
        mesh.face_create_verts(&[v[0], v[1], v[2]], FaceCreateFlags::NONE);

        let doubles =
            mesh.face_triangulate_collect_doubles(quad, QuadMethod::Fixed, NgonMethod::EarClip);

        assert_eq!(doubles.len(), 1);
        let mut triple: Vec<usize> = mesh
            .face_vert_keys(doubles[0])
            .iter()
            .map(|k| k.as_usize())
            .collect();
        triple.sort_unstable();
        let mut expected: Vec<usize> = [v[0], v[1], v[2]].iter().map(|k| k.as_usize()).collect();
        expected.sort_unstable();
        assert_eq!(triple, expected);
    }
}
