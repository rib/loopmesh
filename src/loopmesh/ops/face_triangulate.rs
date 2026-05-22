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
    /// The `FaceKey` for the source face survives the call but now refers
    /// to a triangle (one of the n - 2 new triangles); the remaining n - 3
    /// triangles are fresh faces installed by repeated [`Self::face_split`]
    /// calls along the kernel-selected diagonals.
    ///
    /// Face customdata: each new triangle inherits the source face's CD
    /// via the [`Self::face_split`] interpolation contract.
    ///
    /// Loop customdata: corners coming from the original cycle inherit
    /// their CD from the original loops; corners spawned on the new
    /// diagonals copy from the same-vertex original corner.
    pub fn face_triangulate(
        &mut self,
        face: FaceKey,
        quad_method: QuadMethod,
        ngon_method: NgonMethod,
    ) {
        let verts = self.face_vert_keys(face);
        let n = verts.len();
        if n < 4 {
            return;
        }

        if n == 4 {
            let use_13 = match quad_method {
                QuadMethod::Fixed => false,
                QuadMethod::Alternate => true,
                QuadMethod::ShortEdge => {
                    let d_02_sq = (self.vertex_position(verts[2])
                        - self.vertex_position(verts[0]))
                    .length_squared();
                    let d_13_sq = (self.vertex_position(verts[3])
                        - self.vertex_position(verts[1]))
                    .length_squared();
                    d_13_sq <= d_02_sq
                }
                QuadMethod::LongEdge => {
                    let d_02_sq = (self.vertex_position(verts[2])
                        - self.vertex_position(verts[0]))
                    .length_squared();
                    let d_13_sq = (self.vertex_position(verts[3])
                        - self.vertex_position(verts[1]))
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
            let (a, b) = if use_13 { (1, 3) } else { (0, 2) };
            // `face_split` only returns `None` for `v1 == v2` or either
            // vert not on the face. Both verts come from the face's own
            // loop cycle, so neither can fail.
            self.face_split(face, verts[a], verts[b])
                .expect("face_triangulate: face_split failed on quad");
            return;
        }

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

        // For NgonMethod::Beauty: run the §8 diagonal-swap refinement on
        // the same 2D coords used for ear-clipping — per §8.4 the
        // beautify pass reuses the ear-clip pass's projection. The swap
        // sequence is replayed against the live mesh after the ear-clip
        // surgery completes.
        let beauty_swaps = match ngon_method {
            NgonMethod::EarClip => Vec::new(),
            NgonMethod::Beauty => triangulate::beautify_polygon(&proj, &mut tris),
        };

        let mut current = face;
        // The ear-clip surgery uses the *original* ear-clip triples (before
        // beautify): each peels one ear off the surviving polygon, which is
        // the only ordering for which the diagonals fit the incremental
        // face_split pattern. We re-derive that order from a second
        // ear-clip pass into a separate buffer rather than tracking the
        // pre-beautify triples through the in-place mutation.
        let mut earclip_tris: Vec<[u32; 3]> = Vec::with_capacity(n - 2);
        if matches!(ngon_method, NgonMethod::Beauty) {
            triangulate::ear_clip(&proj, 1, &mut earclip_tris);
        }
        let triples_for_surgery = if matches!(ngon_method, NgonMethod::Beauty) {
            &earclip_tris
        } else {
            &tris
        };

        for triple in &triples_for_surgery[..triples_for_surgery.len() - 1] {
            let v_prev = verts[triple[0] as usize];
            let v_next = verts[triple[2] as usize];
            // `face_split` only returns `None` for `v_prev == v_next` or
            // either vert not on `current`. The ear-clip kernel never emits
            // a triple with `prev == next`, and by induction every un-clipped
            // vert is still a loop of `current` — earlier splits only peel
            // off one triangle each, leaving the rest of the original cycle
            // intact on the surviving polygon side.
            let (new_face, _) = self
                .face_split(current, v_prev, v_next)
                .expect("face_triangulate: face_split failed during n-gon dispatch");
            current = if self.face_len(current) > self.face_len(new_face) {
                current
            } else {
                new_face
            };
        }

        // Replay each beautify swap against the now-triangulated face.
        // `edge_rotate` joins the two triangles sharing the edge and
        // splits along the other diagonal — which is exactly the swap the
        // beautify pass recorded. `ccw` is symmetric for our purposes; we
        // pass true.
        for swap in beauty_swaps {
            let edge = self
                .query_edge_exists(verts[swap.before.0 as usize], verts[swap.before.1 as usize])
                .expect(
                    "face_triangulate: beautify swap's `before` edge missing from triangulation",
                );
            self.edge_rotate(edge, true)
                .expect("face_triangulate: edge_rotate failed for beautify swap");
        }
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
}
