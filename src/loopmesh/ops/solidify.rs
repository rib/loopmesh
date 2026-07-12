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

        // §4.4 fallback-1 / §7.4: the region-non-manifold fallback accumulates
        // each incident face's *stored* normal over the duplicate vert's
        // post-extrude disk, walls included. A side wall is a zero-area
        // degenerate quad at this pre-offset moment (its duplicate edge is still
        // coincident with its rim edge), so its geometric normal is undefined —
        // recomputing it would zero the wall's contribution and the fallback
        // would sum only the duplicate wings, offsetting a symmetric spine
        // wrongly. Instead each wall inherits the **stored normal of the
        // kept-original wing across its rim (base) edge**: that wing was flipped
        // in phase 1 and flipped back by extrude (§6.1), so it sits at its
        // original winding and its geometric normal is the *un-flipped* original
        // wing normal — the negation of the paired duplicate wing's normal, so
        // the (wing, wall) pair cancels on a symmetric fan.
        self.solidify_inherit_wall_normals(region);

        // §4.1: tag the duplicate region's verts and edges, and count per edge
        // how many region faces use it. The smoothed estimate is built only
        // from the marked region.
        let vert_watermark = self.vertices.key_watermark();
        let edge_watermark = self.edges.key_watermark();
        let face_watermark = self.faces.key_watermark();

        // O(1) region-face membership keyed by face watermark, built once, so
        // the radial walks below stay O(Σ degree) rather than
        // O(degree × region-size) linear scans of `duplicate_faces` (§9).
        let mut face_in_region: Vec<bool> = vec![false; face_watermark];
        for &face in duplicate_faces {
            face_in_region[face.as_usize()] = true;
        }

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
        // `vert_on_coplanar_edge` is set on both endpoints of every
        // exactly-coplanar, co-oriented interior region edge (§4.2). That flag —
        // *not* a "did the vertex receive a contribution?" tally — is the single
        // discriminator that gates fallback-2 (§4.3, §4.4, and the
        // `!had_contribution` trap note there): a zero-accumulator vertex falls
        // back to a face normal only when it is coplanar-flagged.
        let mut accumulator: Vec<Vec3> = vec![Vec3::ZERO; vert_watermark];
        let mut vert_on_coplanar_edge: Vec<bool> = vec![false; vert_watermark];

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
                        && face_in_region[face.as_usize()]
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
                    // nothing (their bisector is undefined) — but flag both
                    // endpoints as coplanar-edge endpoints so a resulting zero
                    // accumulator resolves to fallback-2 (§4.3), not to a
                    // zero-normal no-offset.
                    for v in self.edge_vertices(edge) {
                        vert_on_coplanar_edge[v.as_usize()] = true;
                    }
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
                    if face_in_region[face.as_usize()] {
                        region_face_normal = self.faces.get(face).normal;
                        break;
                    }
                }
                region_face_normal * BOUNDARY_EDGE_WEIGHT
            };

            for v in self.edge_vertices(edge) {
                accumulator[v.as_usize()] += edge_normal;
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
                // §4.4 fallback 1: the plain angle-weighted vertex normal
                // accumulated over the *duplicate* vert's post-extrude incident
                // faces — duplicate region faces **and** the side walls stitched
                // around its boundary edges (§4.3, §4.4 fallback 1, §7.4). Not
                // the input vertex's faces, not restricted to the marked region.
                // When that accumulation normalises to zero (the symmetric
                // T-junction cancellation), the vert receives NO offset: it
                // stays coincident with the kept shell. We therefore keep the
                // raw zero here rather than substituting any non-zero fallback,
                // and the §5.3 `thickness · 0` path leaves the vert in place.
                smoothed_normal[slot] = self.solidify_nonmanifold_fallback_normal(v);
                continue;
            }
            let unit = LoopMesh::normalize_or_zero(accumulator[slot]);
            if unit != Vec3::ZERO {
                smoothed_normal[slot] = unit;
            } else if vert_on_coplanar_edge[slot] {
                // §4.4 fallback 2 (narrowly gated, §4.3): the accumulator
                // cancelled to zero AND the vertex is a coplanar-edge endpoint —
                // the ordinary totally-flat interior patch, where every incident
                // region face shares one normal, so any one of them serves.
                smoothed_normal[slot] = self.solidify_any_region_face_normal(v, &face_in_region);
            }
            // else: §4.3 — a zero accumulator with no coplanar-edge flag keeps
            // the zero normal and receives no offset (§5.3's `thickness · 0`).
            // Applying fallback-2 here would be wider than the gate.
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
                self.solidify_vertex_offset_distance(v, n_v, &face_in_region, thickness);
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
        face_in_region: &[bool],
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
                if !face_in_region[face.as_usize()] {
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

    /// §4.4 fallback 1: the plain angle-weighted vertex normal for a
    /// region-non-manifold vertex, accumulated over *all* faces incident to the
    /// (duplicate) vertex in the post-extrude mesh — duplicate region faces
    /// **and** the side walls stitched around its boundary edges — using each
    /// face's current stored normal, weighted by the interior corner angle the
    /// face subtends at `v` (Thürmer–Wüthrich weighting, BMESH.md §11).
    ///
    /// Unlike the general [`Self::vert_normal_update`], this returns the **raw
    /// zero vector** when the accumulation cancels (the symmetric T-junction
    /// case, §7.4) rather than substituting `normalize(position)`. A zero normal
    /// here flows into §5.3 as `thickness · 0`, so the vert receives no offset
    /// and pins to the kept shell — which is exactly the documented behaviour.
    fn solidify_nonmanifold_fallback_normal(&self, v: VertexKey) -> Vec3 {
        let apex = self.vertices.get(v).position;
        let mut acc = Vec3::ZERO;
        let mut disk = self.disk_edges(v);
        while let Some(e) = disk.next(self) {
            let far_vertex = match self.edges.get(e).other_vertex(v) {
                Some(w) => w,
                None => continue,
            };
            let out_ray =
                LoopMesh::normalize_or_zero(self.vertices.get(far_vertex).position - apex);
            let mut radial = self.radial_loops(e);
            while let Some(l) = radial.next(self) {
                let loop_data = self.loops.get(l);
                if loop_data.vertex != v {
                    continue;
                }
                let face = loop_data.face;
                let prev_vertex = self.loops.get(self.loop_prev(l)).vertex;
                let in_ray =
                    LoopMesh::normalize_or_zero(self.vertices.get(prev_vertex).position - apex);
                let dot = in_ray.dot(out_ray).clamp(-1.0, 1.0);
                let weight = LoopMesh::acos_approx(dot);
                acc += weight * self.faces.get(face).normal;
            }
        }
        LoopMesh::normalize_or_zero(acc)
    }

    /// §4.4 fallback-1 / §7.4: give each side wall the stored normal it should
    /// inherit from the kept-original wing across its rim (base) edge.
    ///
    /// The wall is a degenerate zero-area quad at this pre-offset moment, so it
    /// has no meaningful geometric normal of its own. The kept-original wing
    /// across the wall's base edge is a real, non-degenerate face sitting at its
    /// *original* winding (flipped in phase 1, flipped back by extrude §6.1), so
    /// its geometric normal is the un-flipped original wing normal — exactly the
    /// negation of the paired duplicate wing's normal. Threading it onto the
    /// wall makes the (duplicate wing, wall) pairs at a non-manifold spine carry
    /// equal-and-opposite stored normals, so the symmetric fan cancels (§7.4).
    fn solidify_inherit_wall_normals(&mut self, region: &ExtrudeResult) {
        if region.wall_faces.is_empty() {
            return;
        }
        let vert_watermark = self.vertices.key_watermark();
        let mut vert_is_kept_original: Vec<bool> = vec![false; vert_watermark];
        for &(original, _dup) in &region.vert_map {
            vert_is_kept_original[original.as_usize()] = true;
        }

        for &wall in &region.wall_faces {
            // The wall's base (rim) edge is the one whose both endpoints are
            // kept-original verts; the other three edges touch a duplicate vert.
            let base_edge = self.face_edge_keys(wall).into_iter().find(|&e| {
                let [a, b] = self.edge_vertices(e);
                vert_is_kept_original[a.as_usize()] && vert_is_kept_original[b.as_usize()]
            });
            let base_edge = match base_edge {
                Some(e) => e,
                None => continue,
            };

            // The kept-original wing is the face across the base edge that is
            // neither the wall itself nor (defensively) another wall — i.e. the
            // surviving original face on the rim. Compute its geometric normal
            // from its real (non-degenerate) geometry and store it on the wall.
            let mut wing_normal = None;
            let mut radial = self.radial_loops(base_edge);
            while let Some(l) = radial.next(self) {
                let face = self.loops.get(l).face;
                if face == wall || region.wall_faces.contains(&face) {
                    continue;
                }
                wing_normal = Some(self.face_compute_normal(face));
                break;
            }
            if let Some(normal) = wing_normal {
                self.faces.get_mut(wall).normal = normal;
            }
        }
    }

    /// §4.4 fallback 2: the normal of any one region face incident to `v`.
    fn solidify_any_region_face_normal(&self, v: VertexKey, face_in_region: &[bool]) -> Vec3 {
        let mut disk = self.disk_edges(v);
        while let Some(e) = disk.next(self) {
            let mut radial = self.radial_loops(e);
            while let Some(l) = radial.next(self) {
                let loop_data = self.loops.get(l);
                if loop_data.vertex == v && face_in_region[loop_data.face.as_usize()] {
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
    fn flat_grid_full_region_interior_vertex_fires_fallback2() {
        // SOLIDIFY.md §4.2 / §4.3 / §4.4 (narrow gate): solidify the *whole*
        // flat 3×3 grid. An interior vertex is surrounded by four mutually
        // coplanar region faces, so every one of its incident interior edges is
        // exactly-coplanar and contributes nothing — its §4.3 accumulator
        // cancels to zero. But those coplanar edges *flag* it (§4.2), so the
        // narrow gate resolves it via fallback-2 (an arbitrary incident
        // region-face normal) and it offsets like the rest of the flat sheet.
        // A vertex that reached zero *without* being coplanar-flagged would keep
        // a zero normal and not move — the interior vertex must NOT do that.
        let mut mesh = LoopMesh::new();
        let (faces, interior) = flat_grid_3x3(&mut mesh);

        let result = mesh.solidify(&faces, 0.5);
        mesh.validate_topology();

        // Flat region ⇒ every duplicate vert (including the fallback-2 interior
        // vertex) offsets straight to z = -0.5 (duplicate normals point -Z after
        // the phase-1 flip; shell factor 1).
        let mut interior_seen = false;
        for &(original, dup) in &result.vert_map {
            let p = mesh.vertex_position(dup);
            assert!(
                (p.z + 0.5).abs() < 1e-4,
                "flat-grid duplicate should offset to z = -0.5, got z = {}",
                p.z
            );
            if original == interior {
                interior_seen = true;
            }
        }
        assert!(
            interior_seen,
            "the coplanar-flagged interior vertex must appear in the duplicate map"
        );
    }

    #[test]
    fn grid_centre_face_region_offsets_without_fallback2() {
        // SOLIDIFY.md §7.2: solidifying a *single* interior face of the 3×3 grid
        // keeps the original, duplicates it, and walls its four boundary edges.
        // Only the centre face is marked, so each of its four edges is a
        // region-boundary edge (one region face) — the accumulator is non-zero
        // and no fallback-2 path is taken. The flat single face still offsets to
        // z = -0.5. This confirms the narrow gate leaves the foundational
        // interior-region case bit-unchanged.
        let mut mesh = LoopMesh::new();
        let (faces, _interior) = flat_grid_3x3(&mut mesh);
        let centre = faces[4];

        let result = mesh.solidify(&[centre], 0.5);
        mesh.validate_topology();

        assert_eq!(result.cap_faces.len(), 1, "one duplicate centre face");
        assert_eq!(result.wall_faces.len(), 4, "four rim walls");
        for &(_original, dup) in &result.vert_map {
            let p = mesh.vertex_position(dup);
            assert!(
                (p.z + 0.5).abs() < 1e-4,
                "duplicate centre-face vert should offset to z = -0.5, got z = {}",
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
        let a = mesh.face_create(
            &[v0, v1, v2, v3],
            &[e01, e12, e23, e30],
            FaceCreateFlags::NONE,
        );
        let b = mesh.face_create(
            &[v1, v4, v5, v2],
            &[e14, e45, e52, e12],
            FaceCreateFlags::NONE,
        );

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

    #[test]
    fn three_wing_nonmanifold_spine_gets_no_offset() {
        // SOLIDIFY.md §4.4 fallback 1 / §7.4: on the symmetric three-quad
        // T-junction the spine vert's §4.4-fallback-1 normal — accumulated over
        // the duplicate vert's post-extrude incident faces (the three duplicate
        // wings AND the three side walls) — cancels to the zero vector, so the
        // duplicate spine verts receive NO offset and stay coincident with the
        // kept spine. The cancelling is wing-against-its-own-wall: two coplanar
        // wings reinforce (same stored normal), and each is annihilated by its
        // paired wall carrying the un-flipped kept-original wing normal. An
        // accumulation that omitted the walls (or used their degenerate
        // geometric normal) would sum the three wings alone to a non-zero
        // direction and offset the spine wrongly.
        let mut mesh = LoopMesh::new();
        let (faces, s0, s1) = three_wing_tjunction(&mut mesh);
        let result = mesh.solidify(&faces, 0.5);
        mesh.validate_topology();

        let mut spine_pairs = 0;
        let mut wing_offsets = 0;
        for &(original, dup) in &result.vert_map {
            let original_pos = mesh.vertex_position(original);
            let dup_pos = mesh.vertex_position(dup);
            let delta = (dup_pos - original_pos).length();
            if original == s0 || original == s1 {
                spine_pairs += 1;
                assert!(
                    delta < 1e-5,
                    "symmetric spine vert must pin to the kept shell (delta = {delta})"
                );
            } else if delta > 1e-3 {
                wing_offsets += 1;
            }
        }
        assert_eq!(spine_pairs, 2, "both spine verts present in the vert map");
        assert!(
            wing_offsets > 0,
            "wing corners away from the spine must still offset"
        );
    }

    #[test]
    fn asymmetric_nonmanifold_spine_gets_nonzero_offset() {
        // SOLIDIFY.md §4.4 fallback 1 / §7.4: the same fallback over an
        // asymmetric non-manifold fan does NOT cancel — the angle-weighted
        // normal over the duplicate spine vert's post-extrude incident faces is
        // non-zero, so the duplicate spine verts DO offset. This rules out an
        // incorrect "non-manifold ⇒ never offset" shortcut.
        let mut mesh = LoopMesh::new();
        let (faces, s0, s1) = asymmetric_fan(&mut mesh);
        let result = mesh.solidify(&faces, 0.5);
        mesh.validate_topology();

        let mut spine_pairs = 0;
        for &(original, dup) in &result.vert_map {
            if original == s0 || original == s1 {
                spine_pairs += 1;
                let delta = (mesh.vertex_position(dup) - mesh.vertex_position(original)).length();
                assert!(
                    delta > 1e-3,
                    "asymmetric spine vert must offset (delta = {delta})"
                );
            }
        }
        assert_eq!(spine_pairs, 2, "both spine verts present in the vert map");
    }

    /// A flat 3×3 grid of quad faces (4×4 verts) in the z = 0 plane, wound so
    /// every face normal points +Z. Returns the nine faces and one
    /// fully-interior vertex — surrounded by four coplanar region faces, so its
    /// §4.3 accumulator cancels to zero and it is coplanar-flagged (§4.2),
    /// exercising the narrow fallback-2 gate.
    fn flat_grid_3x3(mesh: &mut LoopMesh) -> (Vec<FaceKey>, VertexKey) {
        let mut verts = [[VertexKey::from_usize(0); 4]; 4];
        for gi in 0..4 {
            for gj in 0..4 {
                verts[gi][gj] = mesh.create_vertex(Vec3::new(gi as f32, gj as f32, 0.0));
            }
        }
        let mut faces = Vec::new();
        for gi in 0..3 {
            for gj in 0..3 {
                let ring = [
                    verts[gi][gj],
                    verts[gi + 1][gj],
                    verts[gi + 1][gj + 1],
                    verts[gi][gj + 1],
                ];
                let mut edges = [EdgeKey::from_usize(0); 4];
                for i in 0..4 {
                    edges[i] = mesh
                        .create_edge([ring[i], ring[(i + 1) % 4]], EdgeCreateFlags::NO_DUPLICATE);
                }
                faces.push(mesh.face_create(&ring, &edges, FaceCreateFlags::NONE));
            }
        }
        // verts[1][1] is a fully-interior vertex shared by four region faces.
        (faces, verts[1][1])
    }

    /// An asymmetric region-non-manifold fan: three quad wings off a shared
    /// vertical spine S0--S1, at irregular directions with no mirror symmetry.
    /// Its §4.4-fallback-1 normal does NOT cancel, so the spine verts offset.
    fn asymmetric_fan(mesh: &mut LoopMesh) -> (Vec<FaceKey>, VertexKey, VertexKey) {
        let s0 = mesh.create_vertex(Vec3::new(0.0, 0.0, 0.0));
        let s1 = mesh.create_vertex(Vec3::new(0.0, 0.0, 1.0));
        let wings: [(Vec3, bool); 3] = [
            (Vec3::new(1.0, 0.2, 0.0), true),
            (Vec3::new(0.3, 1.0, 0.0), true),
            (Vec3::new(-0.7, 0.4, 0.0), true),
        ];
        let mut faces = Vec::new();
        for (dir, forward) in wings {
            faces.push(make_wing(mesh, s0, s1, dir, forward));
        }
        (faces, s0, s1)
    }

    /// The symmetric three-quad T-junction of SOLIDIFY.md §7.4's worked
    /// example. The spine runs along `y` from `0 = (0,0,0)` to `1 = (0,1,0)`;
    /// three quad wings hang off the shared radial-3 spine edge `{0,1}`:
    ///
    /// ```text
    /// wing 0 (+x, in z=0):  0,2,3,1   (2=(1,0,0),  3=(1,1,0))   normal +z
    /// wing 1 (−x, in z=0):  0,1,5,4   (4=(−1,0,0), 5=(−1,1,0))  normal +z
    /// wing 2 (+z, in x=0):  0,1,7,6   (6=(0,0,1),  7=(0,1,1))   normal +x
    /// ```
    ///
    /// Wings 0 and 1 are coplanar in `z=0` and **reinforcing** (both +z); the
    /// cancellation that zeroes the spine offset is each wing against its own
    /// paired wall, not wing-against-wing. Returns (region faces, spine vert s0,
    /// spine vert s1). All three wings are solidified together.
    fn three_wing_tjunction(mesh: &mut LoopMesh) -> (Vec<FaceKey>, VertexKey, VertexKey) {
        let s0 = mesh.create_vertex(Vec3::new(0.0, 0.0, 0.0)); // vert 0
        let s1 = mesh.create_vertex(Vec3::new(0.0, 1.0, 0.0)); // vert 1
        let v2 = mesh.create_vertex(Vec3::new(1.0, 0.0, 0.0));
        let v3 = mesh.create_vertex(Vec3::new(1.0, 1.0, 0.0));
        let v4 = mesh.create_vertex(Vec3::new(-1.0, 0.0, 0.0));
        let v5 = mesh.create_vertex(Vec3::new(-1.0, 1.0, 0.0));
        let v6 = mesh.create_vertex(Vec3::new(0.0, 0.0, 1.0));
        let v7 = mesh.create_vertex(Vec3::new(0.0, 1.0, 1.0));
        // Exact §7.4 winding: wing 0 = 0,2,3,1 (normal +z); wing 1 = 0,1,5,4
        // (coplanar with wing 0, also +z — they reinforce); wing 2 = 0,1,7,6
        // (normal +x). The spine edge {0,1} is shared by all three (radial-3).
        let faces = vec![
            make_ring_wing(mesh, [s0, v2, v3, s1]),
            make_ring_wing(mesh, [s0, s1, v5, v4]),
            make_ring_wing(mesh, [s0, s1, v7, v6]),
        ];
        (faces, s0, s1)
    }

    /// Build one quad wing from an explicit 4-vertex ring, deduplicating shared
    /// edges (so the spine edge is shared across wings, making it radial-3).
    fn make_ring_wing(mesh: &mut LoopMesh, ring: [VertexKey; 4]) -> FaceKey {
        let mut edges = [EdgeKey::from_usize(0); 4];
        for i in 0..4 {
            edges[i] =
                mesh.create_edge([ring[i], ring[(i + 1) % 4]], EdgeCreateFlags::NO_DUPLICATE);
        }
        mesh.face_create(&ring, &edges, FaceCreateFlags::NONE)
    }

    /// Build one quad wing sharing the spine edge s0--s1. `forward` chooses the
    /// winding so callers can produce mirror-opposed coplanar wings.
    fn make_wing(
        mesh: &mut LoopMesh,
        s0: VertexKey,
        s1: VertexKey,
        dir: Vec3,
        forward: bool,
    ) -> FaceKey {
        let a = mesh.create_vertex(dir);
        let b = mesh.create_vertex(dir + Vec3::new(0.0, 0.0, 1.0));
        let ring = if forward {
            [s0, s1, b, a]
        } else {
            [s0, a, b, s1]
        };
        let mut edges = [EdgeKey::from_usize(0); 4];
        for i in 0..4 {
            edges[i] =
                mesh.create_edge([ring[i], ring[(i + 1) % 4]], EdgeCreateFlags::NO_DUPLICATE);
        }
        mesh.face_create(&ring, &edges, FaceCreateFlags::NONE)
    }
}
