use smallvec::SmallVec;

use crate::loopmesh::*;

/// Near-tangent guard for the per-corner shell term (SOLIDIFY.md §5.1, the
/// `cos < EPS ⇒ factor = 1` guard from INSET.md §8.2.3): when the smoothed
/// vertex normal is nearly tangent to a face's plane the reciprocal would
/// diverge, so the term is pinned to `1`.
const SHELL_TANGENT_EPS: f32 = 1.0e-6;

/// Fixed weight a region-boundary (open-rim) edge contributes to its
/// endpoints' smoothed-normal accumulator (SOLIDIFY.md §4.2): an open edge is
/// treated as if it met a neighbour at a right angle.
const BOUNDARY_EDGE_WEIGHT: f32 = std::f32::consts::FRAC_PI_2;

impl LoopMesh {
    /// Solidify a face region: give a selected sheet of faces volume by
    /// keeping the originals as one shell, lifting a duplicate of the region
    /// as the opposite shell, ringing the open boundary with side walls, and
    /// offsetting the duplicate along a smoothed per-vertex normal by a signed
    /// `thickness`.
    ///
    /// Structurally this is region extrude with `use_keep_orig = true`
    /// (SOLIDIFY.md §3, EXTRUDE.md §11.2) plus a preparatory winding flip of
    /// the input faces and a positional displacement of the duplicate shell —
    /// it introduces no new topology surgery of its own. The returned
    /// [`ExtrudeResult`] is the region-extrude sub-operator's own result; its
    /// `cap_faces` are the lifted duplicate shell (the analogue of the BMesh
    /// `solidify` `geom.out` slot), its `vert_map`/`edge_map` pair the kept
    /// originals with their displaced duplicates, and its `wall_faces` ring the
    /// boundary.
    ///
    /// The displacement (SOLIDIFY.md §§4–5): a high-quality smoothed normal is
    /// estimated per duplicate-region vertex from the region's own faces (per-
    /// edge angle-weighted accumulation, with open-rim edges weighted at π/2),
    /// then each duplicate vertex moves along that normal by `thickness` scaled
    /// by a corner-angle-weighted shell factor that keeps the offset surface a
    /// constant *perpendicular* distance from the original.
    ///
    /// Because the preparatory flip (SOLIDIFY.md §6) reverses the input faces
    /// before the extrude, the duplicate shell's normals point *opposite* the
    /// input faces' original facing, so a **positive** thickness pushes the
    /// duplicate to the side the input faces' backs were on (SOLIDIFY.md §5.4);
    /// a negative thickness pushes it the way the input faces originally faced.
    ///
    /// See `radial-mesh-guides/operators/SOLIDIFY.md` for the full algorithm
    /// and parity contract.
    pub fn solidify(&mut self, faces: &[FaceKey], thickness: f32) -> ExtrudeResult {
        if faces.is_empty() {
            return ExtrudeResult {
                vert_map: Vec::new(),
                edge_map: Vec::new(),
                side_edges: Vec::new(),
                wall_faces: Vec::new(),
                cap_faces: Vec::new(),
            };
        }

        // Phase 1 (SOLIDIFY.md §6): reverse the input faces' winding so the
        // duplicate shell ends up facing opposite the originals' original
        // facing. The kept originals are flipped a second time by the extrude's
        // kept-original handling, so they return to their original orientation.
        self.reverse_faces(faces, true);

        // Phase 2 (SOLIDIFY.md §3 / EXTRUDE.md §11.2): region extrude keeping
        // the originals. The duplicate shell comes back as `cap_faces`, the
        // walls as `wall_faces`, and the source/duplicate pairing as the maps.
        let result = self.extrude_region_exclude(faces, &[], true, false);

        // Phase 3 + 4 operate on the duplicate region (the lifted shell), not
        // on the kept originals.
        self.solidify_offset_duplicate(&result, thickness);

        result
    }

    /// Phases 3 and 4 (SOLIDIFY.md §§4–5): estimate a smoothed normal for every
    /// duplicate-region vertex and displace it along that normal by the
    /// shell-factor-scaled thickness.
    fn solidify_offset_duplicate(&mut self, region: &ExtrudeResult, thickness: f32) {
        let duplicate_faces = &region.cap_faces;
        if duplicate_faces.is_empty() {
            return;
        }

        // Fresh face normals for the duplicate shell drive both the smoothed
        // estimate (§4) and the shell factor (§5).
        for &face in duplicate_faces {
            self.face_calc_normal(face);
        }

        // §4.1: tag the duplicate region's verts and edges, and count per edge
        // how many region faces use it. The smoothed estimate is built only
        // from the marked region.
        let vert_watermark = self.vertices.key_watermark();
        let edge_watermark = self.edges.key_watermark();

        let mut vert_in_region: Vec<bool> = vec![false; vert_watermark];
        // Per-region-edge incident-face count: 2 ⇒ manifold interior, 1 ⇒
        // open-rim boundary, 0 or ≥3 ⇒ region-non-manifold.
        let mut edge_region_face_count: Vec<u32> = vec![0; edge_watermark];

        for &face in duplicate_faces {
            let mut loops = self.face_loops(face);
            while let Some(l) = loops.next(self) {
                let loop_data = self.loops.get(l);
                vert_in_region[loop_data.vertex.as_usize()] = true;
                edge_region_face_count[loop_data.edge.as_usize()] += 1;
            }
        }

        // §4.1: a region vertex is region-non-manifold if it touches any region
        // edge used by 0 or ≥3 region faces, or if it is non-manifold in the
        // full mesh. Such verts take the plain geometric vertex normal (§4.4
        // fallback 1) rather than the smoothed estimate.
        let mut vert_region_nonmanifold: Vec<bool> = vec![false; vert_watermark];
        for slot in 0..edge_watermark {
            let region_face_count = edge_region_face_count[slot];
            if region_face_count == 0 {
                continue;
            }
            let is_region_nonmanifold_edge = region_face_count >= 3;
            let edge = EdgeKey::from_usize(slot);
            for v in self.edge_vertices(edge) {
                // Full-mesh non-manifold: a region vert whose disk carries an
                // edge used by more than two faces anywhere in the mesh.
                if is_region_nonmanifold_edge || self.solidify_vert_is_nonmanifold(v) {
                    vert_region_nonmanifold[v.as_usize()] = true;
                }
            }
        }

        // §4.2 / §4.3: accumulate per-edge angle-weighted edge normals into
        // both endpoints. `accumulator` holds the running direction sum;
        // `had_contribution` records whether any edge contributed, so a vertex
        // whose estimate cancels can be told apart from one that never got a
        // contribution.
        let mut accumulator: Vec<Vec3> = vec![Vec3::ZERO; vert_watermark];
        let mut had_contribution: Vec<bool> = vec![false; vert_watermark];

        for slot in 0..edge_watermark {
            let region_face_count = edge_region_face_count[slot];
            if region_face_count == 0 || region_face_count >= 3 {
                continue;
            }
            let edge = EdgeKey::from_usize(slot);
            let edge_normal = if region_face_count == 2 {
                // §4.2 manifold interior edge: bisector of the two region
                // faces' normals, renormalised to length `theta` (the dihedral
                // angle), so sharper folds weigh more.
                let mut region_face_normals: SmallVec<[Vec3; 2]> = SmallVec::new();
                let mut radial = self.radial_loops(edge);
                while let Some(l) = radial.next(self) {
                    let face = self.loops.get(l).face;
                    if vert_in_region[self.loops.get(l).vertex.as_usize()]
                        && self.faces.try_get(face).is_some()
                        && duplicate_faces.contains(&face)
                    {
                        region_face_normals.push(self.faces.get(face).normal);
                    }
                }
                if region_face_normals.len() != 2 {
                    continue;
                }
                let n0 = region_face_normals[0];
                let n1 = region_face_normals[1];
                let theta = LoopMesh::acos_approx(n0.dot(n1).clamp(-1.0, 1.0));
                if theta == 0.0 {
                    // §4.2: exactly coplanar, co-oriented faces contribute
                    // nothing; their endpoints may fall back to a face normal.
                    continue;
                }
                LoopMesh::normalize_or_zero(n0 + n1) * theta
            } else {
                // §4.2 boundary (open-rim) edge: the lone region face's normal
                // scaled by the fixed π/2 weight.
                let mut region_face_normal = Vec3::ZERO;
                let mut radial = self.radial_loops(edge);
                while let Some(l) = radial.next(self) {
                    let face = self.loops.get(l).face;
                    if duplicate_faces.contains(&face) {
                        region_face_normal = self.faces.get(face).normal;
                        break;
                    }
                }
                region_face_normal * BOUNDARY_EDGE_WEIGHT
            };

            for v in self.edge_vertices(edge) {
                accumulator[v.as_usize()] += edge_normal;
                had_contribution[v.as_usize()] = true;
            }
        }

        // §4.3: resolve each region vertex's smoothed normal.
        let mut smoothed_normal: Vec<Vec3> = vec![Vec3::ZERO; vert_watermark];
        for slot in 0..vert_watermark {
            if !vert_in_region[slot] {
                continue;
            }
            let v = VertexKey::from_usize(slot);
            if vert_region_nonmanifold[slot] {
                // §4.4 fallback 1: plain geometric vertex normal over the full
                // mesh.
                self.vert_normal_update(v);
                smoothed_normal[slot] = self.vertex_normal(v);
                continue;
            }
            let unit = LoopMesh::normalize_or_zero(accumulator[slot]);
            if unit != Vec3::ZERO {
                smoothed_normal[slot] = unit;
            } else {
                // §4.4 fallback 2: the estimate cancelled to zero (a flat-but-
                // marked vertex) — take an arbitrary incident region face's
                // normal.
                let _ = had_contribution[slot];
                smoothed_normal[slot] = self.solidify_any_region_face_normal(v, duplicate_faces);
            }
        }

        // §5: displace each region vertex along its smoothed normal by
        // `thickness` scaled by the corner-angle-weighted shell factor.
        for slot in 0..vert_watermark {
            if !vert_in_region[slot] {
                continue;
            }
            let v = VertexKey::from_usize(slot);
            let n_v = smoothed_normal[slot];
            let offset_distance =
                self.solidify_vertex_offset_distance(v, n_v, duplicate_faces, thickness);
            if offset_distance != 0.0 {
                let new_position = self.vertex_position(v) + n_v * offset_distance;
                self.set_vertex_position(v, new_position);
            }
        }
    }

    /// §5.2: the corner-angle-weighted mean of the per-corner shell terms over
    /// the region faces incident to `v`. Returns `thickness * mean`, or `0`
    /// when `v` subtends no region-face corner.
    fn solidify_vertex_offset_distance(
        &self,
        v: VertexKey,
        n_v: Vec3,
        region_faces: &[FaceKey],
        thickness: f32,
    ) -> f32 {
        let apex = self.vertex_position(v);
        let mut weighted_sum = 0.0;
        let mut angle_sum = 0.0;
        let mut disk = self.disk_edges(v);
        while let Some(e) = disk.next(self) {
            let mut radial = self.radial_loops(e);
            while let Some(l) = radial.next(self) {
                let loop_data = self.loops.get(l);
                if loop_data.vertex != v {
                    continue;
                }
                let face = loop_data.face;
                if !region_faces.contains(&face) {
                    continue;
                }
                let prev_vertex = self.loops.get(self.loop_prev(l)).vertex;
                let next_vertex = self.loops.get(self.loop_next(l)).vertex;
                let in_ray =
                    LoopMesh::normalize_or_zero(self.vertices.get(prev_vertex).position - apex);
                let out_ray =
                    LoopMesh::normalize_or_zero(self.vertices.get(next_vertex).position - apex);
                let corner_angle = LoopMesh::acos_approx(in_ray.dot(out_ray).clamp(-1.0, 1.0));
                let term = solidify_shell_term(n_v, self.faces.get(face).normal);
                weighted_sum += corner_angle * term;
                angle_sum += corner_angle;
            }
        }
        if angle_sum == 0.0 {
            0.0
        } else {
            thickness * (weighted_sum / angle_sum)
        }
    }

    /// §4.4 fallback 2: the normal of any one region face incident to `v`.
    fn solidify_any_region_face_normal(&self, v: VertexKey, region_faces: &[FaceKey]) -> Vec3 {
        let mut disk = self.disk_edges(v);
        while let Some(e) = disk.next(self) {
            let mut radial = self.radial_loops(e);
            while let Some(l) = radial.next(self) {
                let loop_data = self.loops.get(l);
                if loop_data.vertex == v && region_faces.contains(&loop_data.face) {
                    return self.faces.get(loop_data.face).normal;
                }
            }
        }
        Vec3::ZERO
    }

    /// §4.1: a vertex is non-manifold in the full mesh when any of its disk
    /// edges is used by more than two faces.
    fn solidify_vert_is_nonmanifold(&self, v: VertexKey) -> bool {
        let mut disk = self.disk_edges(v);
        while let Some(e) = disk.next(self) {
            let mut face_count = 0;
            let mut radial = self.radial_loops(e);
            while radial.next(self).is_some() {
                face_count += 1;
                if face_count > 2 {
                    return true;
                }
            }
        }
        false
    }
}

/// §5.1: the per-corner shell term `1 / |dot(n_v, n_f)|`, pinned to `1` when
/// the smoothed normal is nearly tangent to the face plane.
fn solidify_shell_term(n_v: Vec3, n_f: Vec3) -> f32 {
    let cosine = n_v.dot(n_f).abs();
    if cosine < SHELL_TANGENT_EPS {
        1.0
    } else {
        1.0 / cosine
    }
}

#[cfg(test)]
mod tests {
    use glam::Vec3;

    use crate::loopmesh::test_support::create_face_from_verts;
    use crate::loopmesh::*;

    /// A unit quad lying flat in the z = 0 plane, wound so its face normal
    /// points +Z.
    fn flat_quad_plus_z(mesh: &mut LoopMesh) -> FaceKey {
        create_face_from_verts(
            mesh,
            &[
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(1.0, 1.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
            ],
        )
    }

    #[test]
    fn lone_quad_solidifies_to_closed_box() {
        // SOLIDIFY.md §2 / §7.1: an isolated flat quad becomes a closed box-ish
        // shell — kept original + duplicate cap + four walls = 6 faces, 8 verts.
        let mut mesh = LoopMesh::new();
        let face = flat_quad_plus_z(&mut mesh);

        let result = mesh.solidify(&[face], 0.5);

        assert_eq!(mesh.vertex_count(), 8, "4 original + 4 duplicate");
        assert_eq!(mesh.face_count(), 6, "1 kept + 1 duplicate + 4 walls");
        assert_eq!(result.cap_faces.len(), 1, "one duplicate-shell face");
        assert_eq!(result.wall_faces.len(), 4, "four rim walls");
        assert_eq!(result.vert_map.len(), 4);
        mesh.validate_topology();
    }

    #[test]
    fn positive_thickness_pushes_duplicate_opposite_original_facing() {
        // SOLIDIFY.md §5.4: the input quad faces +Z; the phase-1 flip makes the
        // duplicate's normal point -Z, so a positive thickness drives the
        // duplicate to z < 0 while the kept original stays at z = 0.
        let mut mesh = LoopMesh::new();
        let face = flat_quad_plus_z(&mut mesh);

        let result = mesh.solidify(&[face], 0.5);

        // The kept originals stay at z = 0.
        for &(original, _dup) in &result.vert_map {
            assert!(
                mesh.vertex_position(original).z.abs() < 1e-5,
                "kept original stays in the z = 0 plane"
            );
        }
        // Flat region ⇒ every duplicate offsets straight along the duplicate's
        // own (-Z) normal by the full thickness.
        for &(_original, dup) in &result.vert_map {
            let p = mesh.vertex_position(dup);
            assert!(
                (p.z + 0.5).abs() < 1e-4,
                "duplicate pushed to z = -0.5, got z = {}",
                p.z
            );
        }
    }

    #[test]
    fn negative_thickness_pushes_duplicate_toward_original_facing() {
        // SOLIDIFY.md §5.4: negating the thickness flips which side the
        // duplicate lands on — here toward +Z, the way the input faced.
        let mut mesh = LoopMesh::new();
        let face = flat_quad_plus_z(&mut mesh);

        let result = mesh.solidify(&[face], -0.5);

        for &(_original, dup) in &result.vert_map {
            let p = mesh.vertex_position(dup);
            assert!(
                (p.z - 0.5).abs() < 1e-4,
                "negative thickness pushes duplicate to z = +0.5, got z = {}",
                p.z
            );
        }
    }

    #[test]
    fn curved_region_offset_is_not_a_flat_translation() {
        // SOLIDIFY.md §4 / §5: two quads meeting at a 90° fold along their
        // shared edge. The shared-edge verts sit on a manifold interior edge,
        // so their smoothed normal bisects the two face normals and the shell
        // factor scales the move past the flat thickness; the four open-rim
        // verts move along their single face's normal. The result is therefore
        // not a single rigid translation — the displacement vectors differ
        // between the fold verts and the rim verts.
        let mut mesh = LoopMesh::new();
        // Quad A in z = 0 (normal +Z), Quad B folded up in the x = 1 plane
        // (normal +X), sharing the edge from v1=(1,0,0) to v2=(1,1,0).
        let v0 = mesh.create_vertex(Vec3::new(0.0, 0.0, 0.0));
        let v1 = mesh.create_vertex(Vec3::new(1.0, 0.0, 0.0));
        let v2 = mesh.create_vertex(Vec3::new(1.0, 1.0, 0.0));
        let v3 = mesh.create_vertex(Vec3::new(0.0, 1.0, 0.0));
        let v4 = mesh.create_vertex(Vec3::new(1.0, 0.0, 1.0));
        let v5 = mesh.create_vertex(Vec3::new(1.0, 1.0, 1.0));
        let e01 = mesh.create_edge([v0, v1], EdgeCreateFlags::NONE);
        let e12 = mesh.create_edge([v1, v2], EdgeCreateFlags::NONE);
        let e23 = mesh.create_edge([v2, v3], EdgeCreateFlags::NONE);
        let e30 = mesh.create_edge([v3, v0], EdgeCreateFlags::NONE);
        let e14 = mesh.create_edge([v1, v4], EdgeCreateFlags::NONE);
        let e45 = mesh.create_edge([v4, v5], EdgeCreateFlags::NONE);
        let e52 = mesh.create_edge([v5, v2], EdgeCreateFlags::NONE);
        // A: v0 v1 v2 v3 (normal +Z). B winds v1 v4 v5 v2 sharing edge e12.
        let a = mesh.face_create(&[v0, v1, v2, v3], &[e01, e12, e23, e30], FaceCreateFlags::NONE);
        let b = mesh.face_create(&[v1, v4, v5, v2], &[e14, e45, e52, e12], FaceCreateFlags::NONE);

        let result = mesh.solidify(&[a, b], 0.5);

        // Two-face region: 6 region verts duplicated, 6 walls around the L-
        // shaped open boundary, 2 duplicate caps.
        assert_eq!(result.cap_faces.len(), 2);
        assert_eq!(result.vert_map.len(), 6);
        mesh.validate_topology();

        // Collect the per-vertex displacement of each duplicate from its
        // original. They must not all be identical (a flat translation), and
        // the fold verts (shared edge) should move further than the flat
        // thickness because their bisected normal is oblique to both faces.
        let mut displacements = Vec::new();
        for &(original, dup) in &result.vert_map {
            displacements.push(mesh.vertex_position(dup) - mesh.vertex_position(original));
        }
        let first = displacements[0];
        let all_equal = displacements.iter().all(|d| (*d - first).length() < 1e-4);
        assert!(
            !all_equal,
            "a curved region must not collapse to a single rigid translation"
        );
        // At least one vertex moves further than the flat thickness (the fold
        // verts, scaled up by the shell factor 1/cos(45°) = sqrt(2)).
        let max_len = displacements
            .iter()
            .map(|d| d.length())
            .fold(0.0_f32, f32::max);
        assert!(
            max_len > 0.5 + 1e-3,
            "fold verts should overshoot the flat thickness, max move = {max_len}"
        );
    }
}
