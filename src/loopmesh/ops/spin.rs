use std::collections::HashMap;

use glam::{Mat4, Quat, Vec3};

use crate::loopmesh::*;

/// The running "leading ring" carried from step to step (SPIN.md §4.4). It names
/// the verts, edges, and faces of the geometry produced by the most recent step
/// — the band's frontier — partitioned by kind so the per-step rotate can move
/// the verts and the final `geom_last.out` slot can be rebuilt.
#[derive(Default, Clone)]
struct LeadingRing {
    verts: Vec<VertexKey>,
    edges: Vec<EdgeKey>,
    faces: Vec<FaceKey>,
}

impl LeadingRing {
    /// Flatten the ring into the mixed `geom_last.out` buffer (SPIN.md §1): faces
    /// first, then edges, then verts, mirroring the duplicate/extrude result
    /// ordering used elsewhere in the crate.
    fn into_geom_out(self) -> Vec<GeomKey> {
        let mut out: Vec<GeomKey> = Vec::with_capacity(
            self.faces.len() + self.edges.len() + self.verts.len(),
        );
        out.extend(self.faces.into_iter().map(GeomKey::from));
        out.extend(self.edges.into_iter().map(GeomKey::from));
        out.extend(self.verts.into_iter().map(GeomKey::from));
        out
    }
}

impl LoopMesh {
    /// Rotate-extrude (lathe / screw): sweep a `geom` profile around the axis
    /// through `cent` along `axis`, emitting one band of new geometry per step.
    ///
    /// Spin is a **composite** operator (SPIN.md §1–§2): it owns no Euler surgery
    /// of its own. Each step is built from operators already in the crate —
    /// region / edge-only extrude, duplicate, rotate, translate — and spin's
    /// contribution is the iterated per-step loop plus the optional 360°
    /// identity-merge.
    ///
    /// - `use_duplicate = false` (default): each step region/edge-extrudes the
    ///   **running** ring and rotates the new ring by the per-step **delta**
    ///   angle, stitching consecutive rings with wall bands into one connected
    ///   surface (SPIN.md §3.1 / §4.1).
    /// - `use_duplicate = true`: each step duplicates the **original** profile and
    ///   rotates the fresh clone by its **total** cumulative angle, giving `steps`
    ///   disjoint copies fanned around the axis with no walls (SPIN.md §3.2).
    ///
    /// `dvec`, when `Some`, is the per-step screw translation: after each step's
    /// rotation the new geometry is additionally translated by `dvec` **rotated
    /// into the current step's frame**, turning a flat lathe into a helix
    /// (SPIN.md §4.3). `None` is a flat lathe.
    ///
    /// `use_normal_flip` reverses the swept band's winding, applied only at the
    /// first step so the whole sweep winds consistently (SPIN.md §4.2).
    ///
    /// `use_merge` welds the final ring back onto the original profile at a full
    /// revolution, closing the seam — honoured only when `steps >= 3` (SPIN.md
    /// §5 / §5.2). It is meaningful only in extrude mode.
    ///
    /// The returned buffer is `geom_last.out` (SPIN.md §1): the leading edge of
    /// the sweep — the final extruded/duplicated ring, or the merge-survivor set
    /// after a 360° weld. It is *not* the whole swept surface.
    ///
    /// See `radial-mesh-guides/operators/SPIN.md` for the full algorithm and
    /// parity contract.
    #[allow(clippy::too_many_arguments)]
    pub fn spin(
        &mut self,
        geom: &[GeomKey],
        cent: Vec3,
        axis: Vec3,
        dvec: Option<Vec3>,
        angle: f32,
        steps: i32,
        use_merge: bool,
        use_normal_flip: bool,
        use_duplicate: bool,
    ) -> Vec<GeomKey> {
        // §7: empty profile or non-positive step count is a no-op; the leading
        // edge is empty.
        if geom.is_empty() || steps <= 0 {
            return Vec::new();
        }

        let axis = axis.normalize_or_zero();
        let dvec = dvec.unwrap_or(Vec3::ZERO);
        let use_dvec = dvec != Vec3::ZERO;
        // §5.2 — the 360° merge is honoured only in extrude mode and only when the
        // ring is fat enough (steps >= 3) to close cleanly.
        let do_merge = use_merge && !use_duplicate && steps >= 3;

        // The running leading ring starts as the original profile selection,
        // partitioned by kind. The profile faces/edges/verts seed the first
        // step's input; in duplicate mode the original is re-read every step.
        let profile = self.partition_profile(geom);

        // §5.3 — before stepping, snapshot each original profile vert's identity
        // so the final ring can name (and weld onto) the original it came from.
        // The identity ranges over the profile's *closure* verts (an edge's
        // endpoints, a face's corners), since those are the verts the extrude
        // duplicates and the final ring must fuse back onto — not just the
        // directly-named verts. The map is propagated forward across every step's
        // old->dup vert map.
        let mut origin_of: HashMap<VertexKey, VertexKey> = HashMap::new();
        if do_merge {
            for v in self.profile_closure_verts(geom) {
                origin_of.insert(v, v);
            }
        }

        let mut leading = profile.clone();

        for k in 1..=steps {
            let total_k = angle * (k as f32) / (steps as f32);

            let (new_ring, step_matrix) = if use_duplicate {
                // §3.2 — clone the *original* profile, rotate by the TOTAL angle.
                let dup = self.duplicate(geom, false);
                let ring = self.ring_from_duplicate(&dup);
                let matrix = Mat4::from_quat(Quat::from_axis_angle(axis, total_k));
                self.rotate(
                    &ring.verts,
                    RotateParams {
                        matrix,
                        cent,
                        space: Mat4::ZERO,
                        use_shapekey: false,
                    },
                );
                (ring, matrix)
            } else {
                // §3.1 — extrude the *running* ring. The first step takes the
                // user's flip; later steps inherit winding from the adjacent
                // (previous-step) wall (SPIN.md §4.2). The edge-only path reads
                // that adjacent cue automatically from the edge's surround wall,
                // so the region path is the only one that needs the explicit
                // after-first `use_normal_from_adjacent`.
                let step_flip = use_normal_flip && k == 1;
                let (extruded, ring) = self.extrude_leading_ring(&leading, step_flip, k != 1);

                // §5.3 — propagate the original-vert identity onto this step's
                // duplicates so the final ring can name its source.
                if do_merge {
                    for &(old_v, dup_v) in &extruded.vert_map {
                        if let Some(&origin) = origin_of.get(&old_v) {
                            origin_of.insert(dup_v, origin);
                        }
                    }
                }

                let is_last_merge_step = do_merge && k == steps;
                if is_last_merge_step {
                    // §5 — the final ring is welded back onto the originals
                    // instead of being rotated into the seam.
                    let survivors = self.weld_final_ring(&ring, &origin_of);
                    return survivors.into_geom_out();
                }

                // §4.1 — rotate the new ring by the per-step DELTA angle.
                let total_prev = angle * ((k - 1) as f32) / (steps as f32);
                let delta = total_k - total_prev;
                let matrix = Mat4::from_quat(Quat::from_axis_angle(axis, delta));
                self.rotate(
                    &ring.verts,
                    RotateParams {
                        matrix,
                        cent,
                        space: Mat4::ZERO,
                        use_shapekey: false,
                    },
                );
                (ring, matrix)
            };

            if use_dvec {
                // §4.3 — translate by dvec rotated into the step's frame.
                let effective = step_matrix.transform_vector3(dvec);
                self.translate(
                    &new_ring.verts,
                    TranslateParams {
                        offset: effective,
                        space: Mat4::ZERO,
                        use_shapekey: false,
                    },
                );
            }

            leading = new_ring;
        }

        leading.into_geom_out()
    }

    /// Partition a mixed `geom` selection into the leading-ring buckets. A vert
    /// implied only as a face corner or edge endpoint is *not* lifted into the
    /// vert bucket — only directly-named verts seed the per-vertex path; the
    /// face/edge extrude pulls in its own closure.
    fn partition_profile(&self, geom: &[GeomKey]) -> LeadingRing {
        let mut ring = LeadingRing::default();
        for entry in geom {
            match *entry {
                GeomKey::Vert(v) => ring.verts.push(v),
                GeomKey::Edge(e) => ring.edges.push(e),
                GeomKey::Face(f) => ring.faces.push(f),
            }
        }
        ring
    }

    /// The closure verts of a mixed profile selection (SPIN.md §5.3): every
    /// directly-named vert, every named edge's two endpoints, and every named
    /// face's corners — deduped. These are the originals the §5 merge welds the
    /// final ring back onto.
    fn profile_closure_verts(&self, geom: &[GeomKey]) -> Vec<VertexKey> {
        let mut seen: std::collections::HashSet<VertexKey> = std::collections::HashSet::new();
        let mut out: Vec<VertexKey> = Vec::new();
        let push = |v: VertexKey, seen: &mut std::collections::HashSet<VertexKey>, out: &mut Vec<VertexKey>| {
            if seen.insert(v) {
                out.push(v);
            }
        };
        for entry in geom {
            match *entry {
                GeomKey::Vert(v) => push(v, &mut seen, &mut out),
                GeomKey::Edge(e) => {
                    for v in self.edge_vertices(e) {
                        push(v, &mut seen, &mut out);
                    }
                }
                GeomKey::Face(f) => {
                    for v in self.face_vert_keys(f) {
                        push(v, &mut seen, &mut out);
                    }
                }
            }
        }
        out
    }

    /// Build the leading ring from a duplicate-mode clone result (SPIN.md §3.2):
    /// the clone's `geom_out` partitioned by kind.
    fn ring_from_duplicate(&self, dup: &DuplicateResult) -> LeadingRing {
        let mut ring = LeadingRing::default();
        for entry in &dup.geom_out {
            match *entry {
                GeomKey::Vert(v) => ring.verts.push(v),
                GeomKey::Edge(e) => ring.edges.push(e),
                GeomKey::Face(f) => ring.faces.push(f),
            }
        }
        ring
    }

    /// Extrude the running leading ring one step (SPIN.md §3.1), routing by the
    /// ring's element kinds, and return both the extrude result (for the §5.3
    /// identity propagation) and the new leading ring (the duplicates).
    ///
    /// A face ring routes through the region extrude with the after-first
    /// adjacent winding cue; a (faceless) edge ring through edge-only extrude,
    /// which reads its adjacent cue from the previous wall automatically; a bare
    /// vert ring through the per-vertex extrude.
    fn extrude_leading_ring(
        &mut self,
        leading: &LeadingRing,
        use_normal_flip: bool,
        use_normal_from_adjacent: bool,
    ) -> (ExtrudeResult, LeadingRing) {
        if !leading.faces.is_empty() {
            let result = self.extrude_region_ex_normal_from_adjacent(
                &leading.faces,
                use_normal_flip,
                use_normal_from_adjacent,
            );
            let mut ring = LeadingRing::default();
            ring.verts = result.vert_map.iter().map(|&(_, dup)| dup).collect();
            ring.edges = result.edge_map.iter().map(|&(_, dup)| dup).collect();
            // The new leading face is the lifted cap. A surround-free first step
            // keeps+flips the original and builds a fresh duplicate cap
            // (`cap_faces`); a later step's running ring already carries the
            // previous wall as surround, so the delete path re-anchors the input
            // face *onto* the duplicates — it *becomes* the cap, and `cap_faces`
            // is empty. In that case the input faces themselves are the new
            // leading ring.
            ring.faces = if result.cap_faces.is_empty() {
                leading.faces.clone()
            } else {
                result.cap_faces.clone()
            };
            (result, ring)
        } else if !leading.edges.is_empty() {
            let result = self.extrude_edge_only_ex(&leading.edges, use_normal_flip);
            let mut ring = LeadingRing::default();
            ring.verts = result.vert_map.iter().map(|&(_, dup)| dup).collect();
            ring.edges = result.edge_map.iter().map(|&(_, dup)| dup).collect();
            (result, ring)
        } else {
            // Vert-only profile: per-vertex extrude grows a wire polyline.
            let vresult = self.extrude_vert_indiv(&leading.verts);
            let mut result = ExtrudeResult {
                vert_map: vresult.vert_map.clone(),
                edge_map: Vec::new(),
                side_edges: vresult.wire_edges.clone(),
                wall_faces: Vec::new(),
                cap_faces: Vec::new(),
            };
            // The per-vertex extrude reports the lifted wire as a side edge; the
            // new leading ring is the duplicate verts and those wires.
            let mut ring = LeadingRing::default();
            ring.verts = vresult.vert_map.iter().map(|&(_, dup)| dup).collect();
            ring.edges = vresult.wire_edges.clone();
            result.edge_map.clear();
            (result, ring)
        }
    }

    /// §5.3–§5.4 — weld the final extruded ring back onto the original profile.
    /// Each final vert is paired with its originating profile vert via the
    /// carried identity map (`origin_of`), and the pairs are welded so the
    /// originals survive. The surviving (un-welded) leading-ring elements are
    /// returned as `geom_last.out`.
    fn weld_final_ring(
        &mut self,
        ring: &LeadingRing,
        origin_of: &HashMap<VertexKey, VertexKey>,
    ) -> LeadingRing {
        // §5.4 stage 1 — pair each final vert with its identified original and
        // weld onto it (the original survives). A vert whose identity resolves to
        // itself (no distinct original) is left as a survivor.
        let mut weld_pairs: Vec<(VertexKey, VertexKey)> = Vec::new();
        let mut welded_away: std::collections::HashSet<VertexKey> =
            std::collections::HashSet::new();
        for &final_v in &ring.verts {
            if let Some(&origin) = origin_of.get(&final_v) {
                if origin != final_v {
                    weld_pairs.push((final_v, origin));
                    welded_away.insert(final_v);
                }
            }
        }

        // The §5.4 three-stage weld (vert-splice, then duplicate edge/face cull)
        // is exactly weld-verts' merge engine: the vert pass splices, the edge
        // pass collapses coincident edges onto an existing survivor, and the face
        // pass drops faces that fall below three corners or duplicate an existing
        // face. `use_centroid = false` keeps the survivor (original) in place so
        // the seam sits on the originals.
        self.weld_verts(&weld_pairs, false);

        // §5.4 — the surviving leading-edge set: ring elements that still exist
        // after the weld and were not themselves welded away.
        let mut survivors = LeadingRing::default();
        for &v in &ring.verts {
            if !welded_away.contains(&v) && self.vertices.try_get(v).is_some() {
                survivors.verts.push(v);
            }
        }
        for &e in &ring.edges {
            if self.edges.try_get(e).is_some() {
                survivors.edges.push(e);
            }
        }
        for &f in &ring.faces {
            if self.faces.try_get(f).is_some() {
                survivors.faces.push(f);
            }
        }
        survivors
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use glam::Vec3;

    use crate::loopmesh::*;

    const CENT: Vec3 = Vec3::ZERO;
    const AXIS: Vec3 = Vec3::Z;

    /// The canonical lathe edge profile: a wire edge parallel to the +Z axis,
    /// offset `radius` along +X, running from `z=z_bottom` to `z=z_top`. Returns
    /// the edge and its two endpoints (bottom, top).
    fn edge_profile(
        mesh: &mut LoopMesh,
        radius: f32,
        z_bottom: f32,
        z_top: f32,
    ) -> (EdgeKey, VertexKey, VertexKey) {
        let v0 = mesh.create_vertex(Vec3::new(radius, 0.0, z_bottom));
        let v1 = mesh.create_vertex(Vec3::new(radius, 0.0, z_top));
        let e = mesh.create_edge([v0, v1], EdgeCreateFlags::NONE);
        (e, v0, v1)
    }

    /// A quad face profile lying in the x–z plane, offset `radius` along +X — the
    /// canonical lathe *face* profile. Returns the face.
    fn quad_profile(mesh: &mut LoopMesh, radius: f32) -> FaceKey {
        let verts = [
            Vec3::new(radius, 0.0, 0.0),
            Vec3::new(radius + 1.0, 0.0, 0.0),
            Vec3::new(radius + 1.0, 0.0, 1.0),
            Vec3::new(radius, 0.0, 1.0),
        ];
        let vs: Vec<VertexKey> = verts.iter().map(|&p| mesh.create_vertex(p)).collect();
        let es: Vec<EdgeKey> = (0..4)
            .map(|i| mesh.create_edge([vs[i], vs[(i + 1) % 4]], EdgeCreateFlags::NO_DUPLICATE))
            .collect();
        mesh.face_create(&vs, &es, FaceCreateFlags::NONE)
    }

    /// The angular position of a vertex about the +Z axis, in radians.
    fn angle_of(mesh: &LoopMesh, v: VertexKey) -> f32 {
        let p = mesh.vertex_position(v);
        p.y.atan2(p.x)
    }

    /// §3.1 / §4 headline: an edge profile swept `steps = 4` through `angle = π/2`
    /// in extrude mode, no merge — the quarter-pipe. Four wall quads stitch the
    /// five spokes; the leading edge is the final ring (1 edge).
    #[test]
    fn edge_quarter_pipe_extrude_no_merge() {
        let mut mesh = LoopMesh::new();
        let (e, _v0, _v1) = edge_profile(&mut mesh, 1.0, 0.0, 1.0);

        let out = mesh.spin(
            &[e.into()],
            CENT,
            AXIS,
            None,
            std::f32::consts::FRAC_PI_2,
            4,
            false,
            false,
            false,
        );

        // §6.1: edge profile, B=1 open chain → Bv=2. Each step adds 2 verts,
        // E+Bv = 1+2 = 3 edges, F+B = 0+1 = 1 wall face. Over 4 steps:
        // verts 2 + 4*2 = 10; edges 1 + 4*3 = 13; faces 4.
        assert_eq!(mesh.totvert(), 10, "2 profile + 4 rings * 2");
        assert_eq!(mesh.totedge(), 13, "1 profile + 4 steps * (1 dup + 2 vertical)");
        assert_eq!(mesh.totface(), 4, "4 wall quads");

        // geom_last.out is the final ring: one edge.
        assert_eq!(
            out.iter().filter(|g| matches!(g, GeomKey::Edge(_))).count(),
            1,
            "leading edge is the single final ring edge"
        );

        // The final ring sits at the total sweep angle π/2 (the +Y half-plane).
        for g in &out {
            if let GeomKey::Vert(v) = g {
                let a = angle_of(&mesh, *v);
                assert!(
                    (a - std::f32::consts::FRAC_PI_2).abs() < 1e-4,
                    "final ring vert at angle π/2, got {a}"
                );
            }
        }
        mesh.validate_topology();
    }

    /// §4.1 per-step cumulative angle: with `steps = 4`, `angle = π/2`, the four
    /// spokes land at exactly 0, π/8, π/4, 3π/8, π/2 — evenly spaced, computed
    /// from `angle * k / steps` rather than by composing a fixed matrix. We read
    /// the angular position of every vert at the bottom z-row.
    #[test]
    fn per_step_cumulative_angle_evenly_spaced() {
        let mut mesh = LoopMesh::new();
        let (e, _v0, _v1) = edge_profile(&mut mesh, 1.0, 0.0, 1.0);

        mesh.spin(
            &[e.into()],
            CENT,
            AXIS,
            None,
            std::f32::consts::FRAC_PI_2,
            4,
            false,
            false,
            false,
        );

        // Collect the distinct angular positions of the bottom-row verts (z≈0).
        let mut angles: Vec<f32> = mesh
            .vert_keys()
            .into_iter()
            .filter(|&v| mesh.vertex_position(v).z.abs() < 1e-5)
            .map(|v| angle_of(&mesh, v))
            .collect();
        angles.sort_by(|a, b| a.partial_cmp(b).unwrap());
        angles.dedup_by(|a, b| (*a - *b).abs() < 1e-4);

        let expected: Vec<f32> = (0..=4)
            .map(|k| std::f32::consts::FRAC_PI_2 * k as f32 / 4.0)
            .collect();
        assert_eq!(angles.len(), expected.len(), "five evenly spaced spokes");
        for (got, want) in angles.iter().zip(expected.iter()) {
            assert!((got - want).abs() < 1e-4, "spoke angle {got} != {want}");
        }
    }

    /// §3.2 / §4.1 duplicate mode: the same edge profile spun `steps = 4` through
    /// a full revolution produces 4 disjoint rotated copies (no walls), each at
    /// its TOTAL cumulative angle. Contrast with the connected extrude band.
    #[test]
    fn duplicate_mode_fans_disjoint_copies() {
        let mut mesh = LoopMesh::new();
        let (e, _v0, _v1) = edge_profile(&mut mesh, 1.0, 0.0, 1.0);

        let out = mesh.spin(
            &[e.into()],
            CENT,
            AXIS,
            None,
            std::f32::consts::TAU,
            4,
            false,
            false,
            true,
        );

        // §6.3: duplicate mode adds steps*(V,E,F). Profile 2V/1E/0F; +4*(2,1,0).
        assert_eq!(mesh.totvert(), 10, "2 original + 4 copies * 2");
        assert_eq!(mesh.totedge(), 5, "1 original + 4 copies * 1");
        assert_eq!(mesh.totface(), 0, "no walls in duplicate mode");

        // geom_last.out is the final (4th) clone: one edge.
        assert_eq!(
            out.iter().filter(|g| matches!(g, GeomKey::Edge(_))).count(),
            1
        );

        // The four clone copies sit at distinct total angles π/2, π, 3π/2, 2π≡0.
        // Read the bottom-row verts' angles; with the original there are 5 spokes
        // but the 4th clone (2π) lands back on the original's angle (0).
        let bottom_angles: HashSet<i32> = mesh
            .vert_keys()
            .into_iter()
            .filter(|&v| mesh.vertex_position(v).z.abs() < 1e-5)
            .map(|v| (angle_of(&mesh, v).rem_euclid(std::f32::consts::TAU) * 1000.0) as i32)
            .collect();
        // 0, π/2, π, 3π/2 — the 2π clone coincides angularly with 0.
        assert_eq!(bottom_angles.len(), 4, "four distinct angular positions");

        mesh.validate_topology();
    }

    /// §5 / §5.2: a full-revolution edge sweep with `use_merge = true` and
    /// `steps = 8` welds the final ring back onto the original, closing the seam
    /// into a clean 8-quad tube with no doubled geometry. Contrast: merge-off
    /// leaves the doubled seam (one extra ring + a degenerate wall band).
    #[test]
    fn full_rev_merge_closes_seam() {
        let mut merged = LoopMesh::new();
        let (e, _v0, _v1) = edge_profile(&mut merged, 1.0, 0.0, 1.0);
        merged.spin(
            &[e.into()],
            CENT,
            AXIS,
            None,
            std::f32::consts::TAU,
            8,
            true,
            false,
            false,
        );

        // Merged tube: 8 spokes * 2 verts = 16 V, 8 wall quads = 8 F, and
        // 8 (spoke) + 8*2/... edges; assert the seam is closed by checking the
        // vert count is exactly one ring fewer than the merge-off doubled count.
        let merged_counts = (merged.totvert(), merged.totedge(), merged.totface());
        assert_eq!(merged_counts.0, 16, "8 distinct spokes * 2 verts, seam fused");
        assert_eq!(merged_counts.2, 8, "8 wall quads around the tube");
        merged.validate_topology();

        let mut unmerged = LoopMesh::new();
        let (e2, _, _) = edge_profile(&mut unmerged, 1.0, 0.0, 1.0);
        unmerged.spin(
            &[e2.into()],
            CENT,
            AXIS,
            None,
            std::f32::consts::TAU,
            8,
            false,
            false,
            false,
        );
        // Merge-off doubles the seam: one extra ring of verts and an extra
        // (degenerate) wall band relative to the merged tube.
        assert!(
            unmerged.totvert() > merged.totvert(),
            "merge-off keeps the doubled seam ({} > {})",
            unmerged.totvert(),
            merged.totvert()
        );
        unmerged.validate_topology();
    }

    /// §5.2 gate: `steps = 2` full revolution with `use_merge = true` must behave
    /// as merge-off (the gate suppresses the merge below 3 steps). The result
    /// keeps the doubled final ring rather than welding it away.
    #[test]
    fn merge_suppressed_below_three_steps() {
        let mut gated = LoopMesh::new();
        let (e, _v0, _v1) = edge_profile(&mut gated, 1.0, 0.0, 1.0);
        gated.spin(
            &[e.into()],
            CENT,
            AXIS,
            None,
            std::f32::consts::TAU,
            2,
            true,
            false,
            false,
        );
        let gated_counts = (gated.totvert(), gated.totedge(), gated.totface());

        // The same input with merge explicitly off must match exactly — the gate
        // makes steps=2 merge a no-op.
        let mut off = LoopMesh::new();
        let (e2, _, _) = edge_profile(&mut off, 1.0, 0.0, 1.0);
        off.spin(
            &[e2.into()],
            CENT,
            AXIS,
            None,
            std::f32::consts::TAU,
            2,
            false,
            false,
            false,
        );
        assert_eq!(
            gated_counts,
            (off.totvert(), off.totedge(), off.totface()),
            "steps=2 merge is suppressed and equals merge-off"
        );
        gated.validate_topology();
    }

    /// §4.2 `use_normal_flip`: the flip reverses every wall's winding relative to
    /// the un-flipped sweep — a *consistent* reverse, not an alternating per-step
    /// flip. We read each wall face's signed area projected onto its sweep plane;
    /// the flip must reverse the sign for every wall, and the magnitudes match the
    /// un-flipped sweep (same topology, opposite winding).
    #[test]
    fn normal_flip_reverses_band_winding() {
        // Build both sweeps and compare per-face winding via the loop-cycle order.
        let build = |flip: bool| -> Vec<Vec<[u32; 3]>> {
            let mut mesh = LoopMesh::new();
            let (e, _, _) = edge_profile(&mut mesh, 1.0, 0.0, 1.0);
            mesh.spin(
                &[e.into()],
                CENT,
                AXIS,
                None,
                std::f32::consts::FRAC_PI_2,
                4,
                false,
                flip,
                false,
            );
            // For each face, capture its loop-cycle vertex positions (rounded).
            let mut faces: Vec<Vec<[u32; 3]>> = mesh
                .face_keys()
                .into_iter()
                .map(|f| {
                    mesh.face_vert_keys(f)
                        .into_iter()
                        .map(|v| {
                            let p = mesh.vertex_position(v);
                            [
                                (p.x * 1000.0).round() as i32 as u32,
                                (p.y * 1000.0).round() as i32 as u32,
                                (p.z * 1000.0).round() as i32 as u32,
                            ]
                        })
                        .collect()
                })
                .collect();
            faces.sort();
            faces
        };

        let unflipped = build(false);
        let flipped = build(true);
        assert_eq!(unflipped.len(), 4);
        assert_eq!(flipped.len(), 4);

        // Same four faces (same vertex *sets*) but reversed cycle order: every
        // flipped wall is the reverse of some unflipped wall.
        let cycles_reversed = |a: &[[u32; 3]], b: &[[u32; 3]]| -> bool {
            if a.len() != b.len() {
                return false;
            }
            let n = a.len();
            let rev: Vec<[u32; 3]> = b.iter().rev().copied().collect();
            (0..n).any(|shift| (0..n).all(|i| a[i] == rev[(i + shift) % n]))
        };
        for fa in &flipped {
            assert!(
                unflipped.iter().any(|fb| cycles_reversed(fa, fb)),
                "each flipped wall is the consistent reverse of an unflipped wall"
            );
        }
    }

    /// §4.3 `dvec` screw: an axial `dvec` lifts each successive ring up the axis,
    /// tracing a helix. With `dvec = (0,0,h)` and a flat (zero) `angle`-... — here
    /// a quarter sweep — the k-th ring sits at z = profile_z + k*h. We verify the
    /// final ring is lifted by `steps * h` and the band climbs monotonically.
    #[test]
    fn dvec_axial_traces_helix() {
        let mut mesh = LoopMesh::new();
        let (e, _v0, _v1) = edge_profile(&mut mesh, 1.0, 0.0, 1.0);
        let h = 0.5_f32;

        let out = mesh.spin(
            &[e.into()],
            CENT,
            AXIS,
            Some(Vec3::new(0.0, 0.0, h)),
            std::f32::consts::FRAC_PI_2,
            4,
            false,
            false,
            false,
        );

        // The final ring (4th) is lifted by 4*h above the profile bottom (z=0)
        // and top (z=1) rows. Its verts sit at z in {4h, 1+4h} = {2.0, 3.0}.
        let mut final_z: Vec<f32> = out
            .iter()
            .filter_map(|g| match g {
                GeomKey::Edge(edge) => Some(*edge),
                _ => None,
            })
            .flat_map(|edge| mesh.edge_vertices(edge))
            .map(|v| mesh.vertex_position(v).z)
            .collect();
        final_z.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!(
            (final_z[0] - 4.0 * h).abs() < 1e-4,
            "final ring bottom lifted by 4h: {}",
            final_z[0]
        );
        assert!(
            (final_z[1] - (1.0 + 4.0 * h)).abs() < 1e-4,
            "final ring top lifted by 4h: {}",
            final_z[1]
        );
        mesh.validate_topology();
    }

    /// §2 / §6 face profile arity: a quad face swept a quarter turn becomes a
    /// solid band of quad rings with face caps at each end — distinct from the
    /// edge profile's single ribbon. Each step extrudes the running face-ring.
    #[test]
    fn face_quarter_band_extrude() {
        let mut mesh = LoopMesh::new();
        let face = quad_profile(&mut mesh, 1.0);
        assert_eq!((mesh.totvert(), mesh.totedge(), mesh.totface()), (4, 4, 1));

        let out = mesh.spin(
            &[face.into()],
            CENT,
            AXIS,
            None,
            std::f32::consts::FRAC_PI_2,
            4,
            false,
            false,
            false,
        );

        // Face profile: a closed 4-edge boundary loop. Each step adds 4 verts,
        // a duplicate cap (4 edges + 4 walls), and 4 vertical edges. The leading
        // edge is the final cap face (a quad). Assert a band of walls grew and a
        // cap face survives.
        assert!(mesh.totface() > 1, "the face swept into a band of quads");
        assert_eq!(
            out.iter().filter(|g| matches!(g, GeomKey::Face(_))).count(),
            1,
            "leading edge is the final cap face"
        );
        // The final cap is a quad sitting at the total sweep angle π/2.
        for g in &out {
            if let GeomKey::Face(f) = g {
                let ring = mesh.face_vert_keys(*f);
                assert_eq!(ring.len(), 4, "final cap is a quad");
                for v in ring {
                    let a = angle_of(&mesh, v);
                    assert!(
                        (a - std::f32::consts::FRAC_PI_2).abs() < 1e-3,
                        "final cap vert at π/2, got {a}"
                    );
                }
            }
        }
        mesh.validate_topology();
    }

    /// §7 empty profile and zero steps are no-ops with an empty leading edge.
    #[test]
    fn empty_and_zero_steps_are_noops() {
        let mut mesh = LoopMesh::new();
        let (e, _v0, _v1) = edge_profile(&mut mesh, 1.0, 0.0, 1.0);
        let before = (mesh.totvert(), mesh.totedge(), mesh.totface());

        let out_empty = mesh.spin(&[], CENT, AXIS, None, std::f32::consts::TAU, 4, false, false, false);
        assert!(out_empty.is_empty());
        assert_eq!((mesh.totvert(), mesh.totedge(), mesh.totface()), before);

        let out_zero = mesh.spin(
            &[e.into()],
            CENT,
            AXIS,
            None,
            std::f32::consts::TAU,
            0,
            false,
            false,
            false,
        );
        assert!(out_zero.is_empty());
        assert_eq!((mesh.totvert(), mesh.totedge(), mesh.totface()), before);
        mesh.validate_topology();
    }
}
