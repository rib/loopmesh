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

/// The arity of the original profile (SPIN.md §2 / §6), captured once before the
/// sweep begins and held constant across every step. The per-step extrude must
/// dispatch on this rather than on the *running* leading ring: a vert profile's
/// first step emits a wire edge into the running ring, and inspecting that ring
/// on later steps would mis-route the bare-vert polyline path into the
/// edge-extrude branch (turning the intended polyline into a ribbon). §6 fixes
/// the path by the profile's `B` boundary: faces → region, edges → edge-only,
/// verts (`B = 0`) → per-vertex-extrude on *every* step.
#[derive(Clone, Copy, PartialEq, Eq)]
enum ProfileArity {
    Face,
    Edge,
    Vert,
}

impl ProfileArity {
    /// Classify the original profile selection (SPIN.md §6): a face present makes
    /// it a face sweep, else an edge an edge sweep, else a bare-vert sweep.
    fn of(profile: &LeadingRing) -> ProfileArity {
        if !profile.faces.is_empty() {
            ProfileArity::Face
        } else if !profile.edges.is_empty() {
            ProfileArity::Edge
        } else {
            ProfileArity::Vert
        }
    }
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
    /// `space`, when `Some`, is the coordinate frame in which `cent`, `axis`, the
    /// per-step rotation, and the `dvec` translation are all expressed (SPIN.md
    /// §1 / §4) — typically an object matrix. It is a **column-major** 4×4 matrix
    /// (`m[col*4 + row]`, each group of four floats a basis column, the
    /// translation in the last four). `None` is the identity / world frame and
    /// reproduces a world-space sweep byte-for-byte. `space` is threaded straight
    /// through to every per-step `rotate` / `translate`, where it drives the
    /// conjugation `space⁻¹ · M · space` (ROTATE.md §3 / TRANSLATE.md §3); it
    /// changes the frame the rotate/translate act in, not the angle schedule, the
    /// merge identity-carry, or the arity routing.
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
        space: Option<[f32; 16]>,
    ) -> Vec<GeomKey> {
        // §7 / §4.4 — the returned `geom_last.out` slot is seeded with a verbatim
        // copy of the input `geom` *before* the per-step loop, and overwritten only
        // by an ordinary step. When the loop never runs — non-positive `steps` — or
        // never emits — an empty profile — the seed survives and is passed straight
        // back. So an empty profile returns the (empty) seed, and `steps = 0`
        // returns the input profile set unchanged (count = the input buffer size,
        // not a swept ring and not zero), leaving the mesh untouched either way.
        if geom.is_empty() || steps <= 0 {
            return geom.to_vec();
        }

        // SPIN.md §1 / §4 — the coordinate frame `cent`, `axis`, the rotation, and
        // the `dvec` translation are expressed in. `None` is the world frame: the
        // all-zeros `Mat4::ZERO` sentinel the rotate/translate sub-operators read as
        // "no space, apply the matrix directly". A `Some(m)` column-major `[f32;16]`
        // is the frame they conjugate through (ROTATE.md §3 / TRANSLATE.md §3).
        let space = space.map_or(Mat4::ZERO, |m| Mat4::from_cols_array(&m));

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
        // §6 — fix the extrude path by the *original* profile arity for the whole
        // sweep, so a vert profile stays on the per-vertex-extrude polyline path
        // even after its first step seeds a wire edge into the running ring.
        let arity = ProfileArity::of(&profile);

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

        // §4.3 — the running screw-translation vector. Seeded with the input
        // `dvec` and rotated IN PLACE by each step's TOTAL-angle matrix, so the
        // translation actually applied at step `k` is the cumulative product
        // `R_total(k) · … · R_total(1) · dvec`. It is NOT recomputed from `dvec`
        // each step, and the advancing matrix is the total-angle matrix even in
        // extrude mode (where the *geometry* is instead carried by the per-step
        // delta). For an axial `dvec` every total-angle matrix fixes it, so the
        // running vector stays constant and the advance reduces to the clean
        // per-step axial rise.
        let mut dvec_running = dvec;

        for k in 1..=steps {
            let total_k = angle * (k as f32) / (steps as f32);
            // §4.3 — the TOTAL-angle matrix for this step. Drives the duplicate-mode
            // geometry rotation AND (in every mode) the `dvec` advance below.
            let total_matrix = Mat4::from_quat(Quat::from_axis_angle(axis, total_k));

            // §4.3 — advance the running screw vector in place by this step's
            // total-angle matrix, accumulating all prior steps' rotations. Done
            // once per step regardless of mode or branch so the merge step (which
            // returns early) still advances it for its penultimate-ring lift.
            if use_dvec {
                dvec_running = total_matrix.transform_vector3(dvec_running);
            }

            let new_ring = if use_duplicate {
                // §3.2 — clone the *original* profile, rotate by the TOTAL angle.
                let dup = self.duplicate(geom, false);
                let ring = self.ring_from_duplicate(&dup);
                self.rotate(
                    &ring.verts,
                    RotateParams {
                        matrix: total_matrix,
                        cent,
                        space,
                        use_shapekey: false,
                    },
                );
                ring
            } else {
                // §3.1 — extrude the *running* ring. The first step takes the
                // user's flip; later steps inherit winding from the adjacent
                // (previous-step) wall (SPIN.md §4.2). The edge-only path reads
                // that adjacent cue automatically from the edge's surround wall,
                // so the region path is the only one that needs the explicit
                // after-first `use_normal_from_adjacent`.
                let step_flip = use_normal_flip && k == 1;
                let (extruded, ring) =
                    self.extrude_leading_ring(arity, &leading, step_flip, k != 1);

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
                    // §5 / §5.5 — the final ring is welded back onto the
                    // originals instead of being rotated into the seam: the step
                    // rotation is skipped, so its frame matrix is the identity
                    // (§4.4). The weld runs on the extrude's *own* output list (a
                    // scratch set) and must NOT re-write the returned slot, which
                    // stays frozen at the penultimate ring (`leading`, the step
                    // `steps-1` ring) the weld never disturbs.
                    self.weld_final_ring(&ring, &origin_of);

                    // §4.3 / §4.4 — the `dvec` screw translate is NOT skipped on
                    // the merge step. Per the §4.4 pseudocode it is applied to
                    // whatever the returned slot currently names — and on the merge
                    // step that is the **penultimate** ring (`leading`), not the
                    // welded-away closing ring. The running screw vector was already
                    // advanced by this step's total-angle matrix above, so the lift
                    // uses the same cumulative `dvec_running` as an ordinary step
                    // would. This carries the penultimate ring up from its interior
                    // height to the full sweep rise, matching the leading edge a
                    // non-merge sweep would have returned. The welded-away closing
                    // ring is gone, so lifting it would be inert; the survivor that
                    // observes the lift is the penultimate ring.
                    if use_dvec {
                        self.translate(
                            &leading.verts,
                            TranslateParams {
                                offset: dvec_running,
                                space,
                                use_shapekey: false,
                            },
                        );
                    }
                    return leading.into_geom_out();
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
                        space,
                        use_shapekey: false,
                    },
                );
                ring
            };

            if use_dvec {
                // §4.3 — translate this step's new ring by the cumulative running
                // screw vector (already advanced by the total-angle matrix above).
                // Threading the translated ring into the next step's extrude input
                // is what builds the running *sum* of the per-step screw offsets
                // across the chain.
                self.translate(
                    &new_ring.verts,
                    TranslateParams {
                        offset: dvec_running,
                        space,
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
    /// *original profile arity* (not the running ring's kinds) and return both
    /// the extrude result (for the §5.3 identity propagation) and the new leading
    /// ring (the duplicates).
    ///
    /// A face profile routes through the region extrude with the after-first
    /// adjacent winding cue; an edge profile through edge-only extrude, which
    /// reads its adjacent cue from the previous wall automatically; a bare-vert
    /// profile through the per-vertex extrude on *every* step (§6). Dispatching on
    /// the captured arity rather than the running ring keeps the vert profile on
    /// the polyline path after its first step seeds a wire edge into that ring.
    fn extrude_leading_ring(
        &mut self,
        arity: ProfileArity,
        leading: &LeadingRing,
        use_normal_flip: bool,
        use_normal_from_adjacent: bool,
    ) -> (ExtrudeResult, LeadingRing) {
        if arity == ProfileArity::Face {
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
        } else if arity == ProfileArity::Edge {
            let result = self.extrude_edge_only_ex(&leading.edges, use_normal_flip);
            let mut ring = LeadingRing::default();
            ring.verts = result.vert_map.iter().map(|&(_, dup)| dup).collect();
            ring.edges = result.edge_map.iter().map(|&(_, dup)| dup).collect();
            (result, ring)
        } else {
            // Vert-only profile (§6, `B = 0`): per-vertex extrude grows a wire
            // polyline — one duplicate vert per step joined to the running vert
            // by a fresh wire edge, no walls, no faces.
            let vresult = self.extrude_vert_indiv(&leading.verts);
            let result = ExtrudeResult {
                vert_map: vresult.vert_map.clone(),
                edge_map: Vec::new(),
                side_edges: vresult.wire_edges.clone(),
                wall_faces: Vec::new(),
                cap_faces: Vec::new(),
            };
            // §6.4 — the new leading ring is the duplicate verts ONLY. The wire
            // edge is part of the polyline, not the leading edge: a vert
            // profile's `geom_last.out` is one profile-shaped ring (1 vert, 0
            // edges, 0 faces). Carrying the wire here would both overcount the
            // returned slot and, before the arity dispatch, mis-route later steps.
            let mut ring = LeadingRing::default();
            ring.verts = vresult.vert_map.iter().map(|&(_, dup)| dup).collect();
            (result, ring)
        }
    }

    /// §5.3–§5.4 — weld the final extruded ring back onto the original profile.
    /// Each final vert is paired with its originating profile vert via the
    /// carried identity map (`origin_of`), and the pairs are welded so the
    /// originals survive. This mutates the swept mesh (closing the seam) but does
    /// not produce the returned slot: per §5.5 the returned `geom_last.out` is the
    /// penultimate ring, which this weld never touches.
    fn weld_final_ring(
        &mut self,
        ring: &LeadingRing,
        origin_of: &HashMap<VertexKey, VertexKey>,
    ) {
        // §5.4 stage 1 — pair each final vert with its identified original and
        // weld onto it (the original survives). A vert whose identity resolves to
        // itself (no distinct original) is not a weld target.
        let mut weld_pairs: Vec<(VertexKey, VertexKey)> = Vec::new();
        for &final_v in &ring.verts {
            if let Some(&origin) = origin_of.get(&final_v) {
                if origin != final_v {
                    weld_pairs.push((final_v, origin));
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
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use glam::{Mat4, Quat, Vec3};

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
            None,
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
            None,
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
            None,
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
            None,
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
            None,
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

    /// §5.5 / §6.4 leading-edge contract: after a full-revolution merge the
    /// returned `geom_last.out` is the **penultimate** ring (the ring emitted at
    /// step `steps-1`), NOT the empty post-fuse weld remainder. For the headline
    /// edge profile that is 2 verts + 1 edge = 3 elements, sitting at the
    /// cumulative angle of step `steps-1`. Contrast: the merge-off full sweep
    /// returns the genuine *final* ring at the seam angle.
    #[test]
    fn full_rev_merge_geom_last_is_penultimate_ring() {
        let steps = 8;
        let mut merged = LoopMesh::new();
        let (e, _v0, _v1) = edge_profile(&mut merged, 1.0, 0.0, 1.0);
        let out = merged.spin(
            &[e.into()],
            CENT,
            AXIS,
            None,
            std::f32::consts::TAU,
            steps,
            true,
            false,
            false,
            None,
        );

        // The returned slot is one profile-shaped ring: 2 verts + 1 edge. Crucially
        // it is NOT empty (a port returning the emptied weld working set reports 0).
        let verts: Vec<VertexKey> = out
            .iter()
            .filter_map(|g| match g {
                GeomKey::Vert(v) => Some(*v),
                _ => None,
            })
            .collect();
        let edges: Vec<EdgeKey> = out
            .iter()
            .filter_map(|g| match g {
                GeomKey::Edge(e) => Some(*e),
                _ => None,
            })
            .collect();
        assert_eq!(out.len(), 3, "penultimate ring: 2 verts + 1 edge = 3 elements");
        assert_eq!(verts.len(), 2, "penultimate ring has 2 verts");
        assert_eq!(edges.len(), 1, "penultimate ring has 1 edge");

        // Identity: every returned element is a genuine, still-live survivor — the
        // weld at the final step touches the last ring and the seam originals, never
        // this penultimate ring.
        for &v in &verts {
            assert!(
                merged.vertices.try_get(v).is_some(),
                "returned penultimate vert is live"
            );
        }
        assert!(
            merged.edges.try_get(edges[0]).is_some(),
            "returned penultimate edge is live"
        );

        // Position: the penultimate ring sits at the cumulative angle of step
        // steps-1 = 2π * (steps-1)/steps, distinct from the seam (angle 0).
        let want = std::f32::consts::TAU * (steps - 1) as f32 / steps as f32;
        let want_wrapped = want.rem_euclid(std::f32::consts::TAU);
        for &v in &verts {
            let a = angle_of(&merged, v).rem_euclid(std::f32::consts::TAU);
            assert!(
                (a - want_wrapped).abs() < 1e-4,
                "penultimate vert at angle {want_wrapped}, got {a}"
            );
        }
        merged.validate_topology();

        // Contrast: merge-off returns the genuine final ring at the seam angle
        // (cumulative(steps) = 2π ≡ 0), still 3 elements but a different identity.
        let mut unmerged = LoopMesh::new();
        let (e2, _, _) = edge_profile(&mut unmerged, 1.0, 0.0, 1.0);
        let out_off = unmerged.spin(
            &[e2.into()],
            CENT,
            AXIS,
            None,
            std::f32::consts::TAU,
            steps,
            false,
            false,
            false,
            None,
        );
        assert_eq!(out_off.len(), 3, "merge-off final ring: 2 verts + 1 edge");
        for g in &out_off {
            if let GeomKey::Vert(v) = g {
                let a = angle_of(&unmerged, *v).rem_euclid(std::f32::consts::TAU);
                let at_seam = a.abs() < 1e-4
                    || (a - std::f32::consts::TAU).abs() < 1e-4;
                assert!(at_seam, "merge-off final ring sits at the seam angle, got {a}");
            }
        }
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
            None,
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
            None,
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
                None,
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
            None,
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

    /// §4.3 `dvec` screw, **non-axial** — the distinguishing pin. With a `dvec`
    /// carrying an in-plane component (`[0.3, 0, 0.5]` about +z) the running screw
    /// vector genuinely rotates each step, so the translation applied at step `k`
    /// is the cumulative product `R_total(k)·…·R_total(1)·dvec`, not `R_total(k)`
    /// or `R_delta(k)` on a *fresh* `dvec`. This catches the old arithmetic
    /// (`R_delta * dvec` on a fresh vector) which collapses to the same answer
    /// only for an axial `dvec`.
    ///
    /// The independent oracle simulates the exact extrude-mode threading the
    /// operator performs — each step's delta rotation acts on the *whole* prior
    /// ring (its rotated position plus its accumulated screw offset), and the
    /// step's own running-vector offset is then added — using glam matrices on the
    /// abstract profile points, with no reference to spin's internal path.
    #[test]
    fn dvec_non_axial_traces_cumulative_product() {
        let dvec = Vec3::new(0.3, 0.0, 0.5);
        let angle = std::f32::consts::FRAC_PI_2;
        let steps = 3;

        let mut mesh = LoopMesh::new();
        let (e, _v0, _v1) = edge_profile(&mut mesh, 1.0, 0.0, 1.0);
        let out = mesh.spin(
            &[e.into()],
            CENT,
            AXIS,
            Some(dvec),
            angle,
            steps,
            false,
            false,
            false,
            None,
        );
        mesh.validate_topology();

        // Independent oracle: thread two profile points through `steps` extrude
        // steps. At step k the prior ring (position + accumulated offset) is
        // carried by the DELTA rotation, then the step's running-vector offset
        // (advanced by the TOTAL-angle matrix, cumulative across steps) is added.
        let advance = |start: Vec3| -> Vec3 {
            let mut point = start;
            let mut dvec_running = dvec;
            for k in 1..=steps {
                let total_k = angle * k as f32 / steps as f32;
                let total_prev = angle * (k - 1) as f32 / steps as f32;
                let r_total = Mat4::from_quat(Quat::from_axis_angle(AXIS, total_k));
                let r_delta = Mat4::from_quat(Quat::from_axis_angle(AXIS, total_k - total_prev));
                dvec_running = r_total.transform_vector3(dvec_running);
                point = r_delta.transform_point3(point) + dvec_running;
            }
            point
        };
        let want_bottom = advance(Vec3::new(1.0, 0.0, 0.0));
        let want_top = advance(Vec3::new(1.0, 0.0, 1.0));

        let final_positions: Vec<Vec3> = out
            .iter()
            .filter_map(|g| match g {
                GeomKey::Edge(edge) => Some(*edge),
                _ => None,
            })
            .flat_map(|edge| mesh.edge_vertices(edge))
            .map(|v| mesh.vertex_position(v))
            .collect();
        assert_eq!(final_positions.len(), 2, "final ring has 2 verts");

        let hits = |target: Vec3| final_positions.iter().any(|p| p.distance(target) < 1e-4);
        assert!(
            hits(want_bottom),
            "final ring bottom at cumulative-product position {want_bottom:?}; got {final_positions:?}"
        );
        assert!(
            hits(want_top),
            "final ring top at cumulative-product position {want_top:?}; got {final_positions:?}"
        );

        // Discriminator: the old (buggy) arithmetic translated each step's ring by
        // `R_delta(k) * dvec` on a *fresh* `dvec`, accumulating a different sum.
        // Confirm that shortcut lands the final ring somewhere ELSE, so this test
        // genuinely fails against the old code (and any non-cumulative shortcut).
        let shortcut = |start: Vec3| -> Vec3 {
            let mut point = start;
            for k in 1..=steps {
                let total_k = angle * k as f32 / steps as f32;
                let total_prev = angle * (k - 1) as f32 / steps as f32;
                let r_delta = Mat4::from_quat(Quat::from_axis_angle(AXIS, total_k - total_prev));
                point = r_delta.transform_point3(point) + r_delta.transform_vector3(dvec);
            }
            point
        };
        let shortcut_bottom = shortcut(Vec3::new(1.0, 0.0, 0.0));
        assert!(
            want_bottom.distance(shortcut_bottom) > 1e-3,
            "the cumulative-product target must differ from the fresh-delta shortcut \
             (non-axial dvec is the distinguishing case); cumulative={want_bottom:?} \
             shortcut={shortcut_bottom:?}"
        );
    }

    /// §4.3 / §4.4 / §5.5 screw + full-revolution merge — the **penultimate-ring
    /// lift**. On the merge step the step matrix is the identity and the `dvec`
    /// post-step translate is applied to whatever the returned slot names. Per the
    /// §4.4 pseudocode that slot is the **penultimate** ring (the welded-away
    /// closing ring is gone, so a lift on *it* would be inert). The lift carries
    /// the penultimate ring up one extra screw step — from its interior height of
    /// `(steps-1)*h` to the full `steps*h` rise — so the returned leading edge
    /// matches the height a non-merge sweep's last ring would report.
    ///
    /// The discriminating assertion is on the returned `geom_last.out` verts: they
    /// must sit at z = steps*h (bottom) and z = 1 + steps*h (top). A port that
    /// skipped the merge-step `dvec` would under-lift them by exactly one step to
    /// z = (steps-1)*h and z = 1 + (steps-1)*h — which is the divergence the
    /// deferred A/B oracle pins. (The seam survivors and the rest of the swept
    /// surface are covered by `full_rev_merge_closes_seam`; here we target only
    /// the merge-step lift the previous fix located in the wrong place.)
    #[test]
    fn full_rev_screw_merge_lifts_penultimate_ring() {
        let steps = 4;
        let h = 0.5_f32;

        let mut mesh = LoopMesh::new();
        let (e, _v0, _v1) = edge_profile(&mut mesh, 1.0, 0.0, 1.0);
        let out = mesh.spin(
            &[e.into()],
            CENT,
            AXIS,
            Some(Vec3::new(0.0, 0.0, h)),
            std::f32::consts::TAU,
            steps,
            true,
            false,
            false,
            None,
        );
        mesh.validate_topology();

        // §5.5 leading-edge contract: the returned slot is the penultimate ring —
        // one profile-shaped ring (2 verts + 1 edge), not the empty weld remainder.
        assert_eq!(out.len(), 3, "penultimate ring: 2 verts + 1 edge");

        // The returned penultimate-ring verts: assert they were lifted by the FULL
        // steps*h screw rise (the merge-step `dvec` carried them up one extra step
        // from their interior (steps-1)*h height). This is the quantity the A/B
        // oracle disagrees on; skipping the merge-step lift would leave them at
        // (steps-1)*h / 1+(steps-1)*h and fail here.
        let mut returned_z: Vec<f32> = out
            .iter()
            .filter_map(|g| match g {
                GeomKey::Edge(edge) => Some(*edge),
                _ => None,
            })
            .flat_map(|edge| mesh.edge_vertices(edge))
            .map(|v| mesh.vertex_position(v).z)
            .collect();
        returned_z.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(returned_z.len(), 2, "penultimate ring has 2 verts");

        let full_bottom = steps as f32 * h; // 2.0
        let full_top = 1.0 + steps as f32 * h; // 3.0
        let under_bottom = (steps - 1) as f32 * h; // 1.5 — the un-lifted (buggy) height
        assert!(
            (returned_z[0] - full_bottom).abs() < 1e-4,
            "penultimate ring lifted by full steps*h={full_bottom} (not under-lifted \
             to {under_bottom}): got {}",
            returned_z[0]
        );
        assert!(
            (returned_z[1] - full_top).abs() < 1e-4,
            "penultimate ring top lifted by full steps*h: want {full_top}, got {}",
            returned_z[1]
        );

        // The seam survivors are the welded-onto originals at z = 0 and z = 1 — the
        // identity weld (§5.3) keeps them at the un-lifted originals' height, so
        // those heights still exist somewhere in the mesh and the merge closed the
        // tube (one ring fewer than the merge-off doubled seam).
        let all_z: Vec<f32> = mesh
            .vert_keys()
            .into_iter()
            .map(|v| mesh.vertex_position(v).z)
            .collect();
        assert!(
            all_z.iter().any(|z| z.abs() < 1e-4),
            "a seam survivor sits at the original bottom (z = 0): {all_z:?}"
        );
        assert!(
            all_z.iter().any(|z| (z - 1.0).abs() < 1e-4),
            "a seam survivor sits at the original top (z = 1): {all_z:?}"
        );

        let mut off = LoopMesh::new();
        let (e2, _, _) = edge_profile(&mut off, 1.0, 0.0, 1.0);
        off.spin(
            &[e2.into()],
            CENT,
            AXIS,
            Some(Vec3::new(0.0, 0.0, h)),
            std::f32::consts::TAU,
            steps,
            false,
            false,
            false,
            None,
        );
        // Merge fuses one ring of 2 verts / 1 edge onto the originals by identity.
        assert_eq!(off.totvert() - mesh.totvert(), 2, "merge fuses one ring of 2 verts");
        assert_eq!(off.totedge() - mesh.totedge(), 1, "merge fuses one ring edge");
        assert_eq!(mesh.totface(), steps as usize, "closed tube: `steps` wall bands");
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
            None,
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

    /// §6 / §6.4 vert profile arity: a lone off-axis vert swept a quarter turn
    /// (`steps = 4`, extrude mode, no merge) grows a **wire polyline**, not a
    /// ribbon. The vert profile is `B = 0`, so every step takes the per-vertex
    /// extrude path: one duplicate vert + one wire edge, no walls, no faces. Over
    /// 4 steps the polyline is `steps + 1 = 5` verts joined by `steps = 4` wire
    /// edges with 0 faces, and `geom_last.out` is the single leading vert (1
    /// element) per the §6.4 leading-edge contract. Dispatching on the running
    /// ring (rather than the captured profile arity) would let step 1's wire edge
    /// pull steps 2+ into the edge-extrude branch — the 8 V / 10 E / 3 F ribbon.
    #[test]
    fn vert_quarter_polyline_extrude_no_merge() {
        let mut mesh = LoopMesh::new();
        // A lone off-axis vert: 1 V / 0 E / 0 F.
        let v = mesh.create_vertex(Vec3::new(1.0, 0.0, 0.0));
        assert_eq!((mesh.totvert(), mesh.totedge(), mesh.totface()), (1, 0, 0));

        let out = mesh.spin(
            &[v.into()],
            CENT,
            AXIS,
            None,
            std::f32::consts::FRAC_PI_2,
            4,
            false,
            false,
            false,
            None,
        );

        // The wire polyline: steps+1 verts, steps wire edges, no faces.
        assert_eq!(mesh.totvert(), 5, "steps+1 = 5 polyline verts");
        assert_eq!(mesh.totedge(), 4, "steps = 4 wire edges joining the polyline");
        assert_eq!(mesh.totface(), 0, "vert profile grows no walls");

        // §6.4 — the leading edge is the single last extruded vert: 1 element,
        // a vert, with no edge or face in the returned slot.
        assert_eq!(out.len(), 1, "geom_last.out is one profile-shaped ring: 1 vert");
        assert!(
            matches!(out[0], GeomKey::Vert(_)),
            "the leading slot names the single last extruded vert"
        );
        assert_eq!(
            out.iter().filter(|g| matches!(g, GeomKey::Edge(_))).count(),
            0,
            "no wire edge leaks into the leading slot"
        );

        // The five spokes land at evenly spaced angles 0, π/8, π/4, 3π/8, π/2.
        let mut angles: Vec<f32> = mesh
            .vert_keys()
            .into_iter()
            .map(|v| angle_of(&mesh, v))
            .collect();
        angles.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let expected: Vec<f32> = (0..=4)
            .map(|k| std::f32::consts::FRAC_PI_2 * k as f32 / 4.0)
            .collect();
        assert_eq!(angles.len(), expected.len(), "five evenly spaced polyline verts");
        for (got, want) in angles.iter().zip(expected.iter()) {
            assert!((got - want).abs() < 1e-4, "polyline vert angle {got} != {want}");
        }

        // The leading vert sits at the total sweep angle π/2.
        if let GeomKey::Vert(lv) = out[0] {
            let a = angle_of(&mesh, lv);
            assert!(
                (a - std::f32::consts::FRAC_PI_2).abs() < 1e-4,
                "leading vert at angle π/2, got {a}"
            );
        }
        mesh.validate_topology();
    }

    /// §7 `steps = 0`: the per-step loop never runs, so the mesh is left exactly
    /// untouched. The returned `geom_last.out` is NOT empty: it is the seed — a
    /// verbatim copy of the input `geom` — passed straight back (§4.4 seeds the
    /// slot with `p.geom.clone()` before the loop). For a single-edge profile
    /// handed in as one `geom` element the returned count is `1` (that input
    /// edge), not `0` and not a swept ring.
    #[test]
    fn spin_steps_zero_leaves_profile_untouched() {
        let mut mesh = LoopMesh::new();
        let (e, v0, v1) = edge_profile(&mut mesh, 1.0, 0.0, 1.0);
        let before = (mesh.totvert(), mesh.totedge(), mesh.totface());

        let out = mesh.spin(
            &[e.into()],
            CENT,
            AXIS,
            None,
            std::f32::consts::FRAC_PI_2,
            0,
            false,
            false,
            false,
            None,
        );

        // The mesh is untouched — no swept geometry at all.
        assert_eq!(
            (mesh.totvert(), mesh.totedge(), mesh.totface()),
            before,
            "steps=0 leaves the mesh exactly as supplied"
        );

        // The returned slot is the seeded input `geom` passed straight back: the
        // one input edge (count 1, NOT the swept-ring count and NOT 0).
        assert_eq!(out.len(), 1, "geom_last.out is the input geom buffer (count 1)");
        assert_eq!(out[0], e.into(), "the returned element is the verbatim input edge");

        // The input edge and its endpoints are unchanged and still live.
        assert!(mesh.edges.try_get(e).is_some(), "input edge survives untouched");
        assert_eq!(mesh.vertex_position(v0), Vec3::new(1.0, 0.0, 0.0));
        assert_eq!(mesh.vertex_position(v1), Vec3::new(1.0, 0.0, 1.0));
        mesh.validate_topology();
    }

    /// §7 `angle = 0`, `steps > 0`: every step rotates by `angle / steps = 0`, a
    /// well-defined identity rotation. The extrude loop still runs in full — it
    /// emits `steps` genuine rings, but each lands **coincident** with the profile
    /// and every other ring (a connected band folded flat onto the profile). The
    /// rings being coincident does not shrink the slot: `geom_last.out` holds the
    /// last (coincident) ring, count = one profile-shaped ring `V + E = 2 + 1 = 3`
    /// for the headline edge profile. Stacked verts are disambiguated by incidence,
    /// not position.
    #[test]
    fn spin_angle_zero_stacks_coincident_rings() {
        let mut mesh = LoopMesh::new();
        let (e, _v0, _v1) = edge_profile(&mut mesh, 1.0, 0.0, 1.0);

        let out = mesh.spin(
            &[e.into()],
            CENT,
            AXIS,
            None,
            0.0,
            4,
            false,
            false,
            false,
            None,
        );

        // The full loop ran: 4 coincident rings stitched by (zero-length) walls.
        // Same connectivity as the non-degenerate quarter-pipe (§6.1): 10 V, 13 E,
        // 4 F — only the positions collapse onto the profile.
        assert_eq!(mesh.totvert(), 10, "steps=4 rings emitted, no merge");
        assert_eq!(mesh.totedge(), 13);
        assert_eq!(mesh.totface(), 4);

        // The swept verts all sit at the profile's angular position (angle 0,
        // i.e. the +X half-plane) — the zero rotation never moves them off it.
        for v in mesh.vert_keys() {
            let p = mesh.vertex_position(v);
            assert!(
                p.y.abs() < 1e-5 && p.x > 0.0,
                "every ring is coincident with the profile (on +X, y≈0): {p:?}"
            );
        }

        // §7 leading-edge count: the last (coincident) ring is one profile-shaped
        // ring — 2 verts + 1 edge = 3, not collapsed by the coincidence.
        assert_eq!(out.len(), 3, "last ring V+E = 2+1 = 3 (coincidence does not shrink it)");
        assert_eq!(
            out.iter().filter(|g| matches!(g, GeomKey::Vert(_))).count(),
            2,
            "last ring has 2 verts"
        );
        assert_eq!(
            out.iter().filter(|g| matches!(g, GeomKey::Edge(_))).count(),
            1,
            "last ring has 1 edge"
        );
        mesh.validate_topology();
    }

    /// §7 negative `angle`: sweeps the other way around the axis. The sign flows
    /// through the per-step matrices (§4.1), NOT through the winding flags — so a
    /// `-θ` sweep is the `+θ` sweep reflected across the axis: same element counts,
    /// same one-ring `geom_last.out`, the band on the opposite side with the *same*
    /// consistent (not inside-out) facing. We pin (a) the mirror geometry: verts at
    /// negated angle, (b) consistent winding: every wall is wound the same way as
    /// its `+θ` mirror, not reversed.
    #[test]
    fn spin_negative_angle_mirror_sweep_consistent_winding() {
        // Build the +θ and -θ quarter sweeps and capture each wall's loop-cycle
        // vertex positions (rounded), with the y-coordinate mirrored on the +θ
        // side so the two should match cycle-for-cycle iff winding is consistent.
        let build = |angle: f32| -> Vec<Vec<[i32; 3]>> {
            let mut mesh = LoopMesh::new();
            let (e, _, _) = edge_profile(&mut mesh, 1.0, 0.0, 1.0);
            let out = mesh.spin(
                &[e.into()],
                CENT,
                AXIS,
                None,
                angle,
                4,
                false,
                false,
                false,
                None,
            );
            // Same leading-edge contract as the positive sweep: one ring, 1 edge.
            assert_eq!(out.len(), 3, "negative-angle leading edge is one ring (V+E=3)");
            assert_eq!((mesh.totvert(), mesh.totedge(), mesh.totface()), (10, 13, 4));
            mesh.validate_topology();
            let quant = |p: Vec3| {
                [
                    (p.x * 1000.0).round() as i32,
                    (p.y * 1000.0).round() as i32,
                    (p.z * 1000.0).round() as i32,
                ]
            };
            let mut faces: Vec<Vec<[i32; 3]>> = mesh
                .face_keys()
                .into_iter()
                .map(|f| {
                    mesh.face_vert_keys(f)
                        .into_iter()
                        .map(|v| quant(mesh.vertex_position(v)))
                        .collect()
                })
                .collect();
            faces.sort();
            faces
        };

        let positive = build(std::f32::consts::FRAC_PI_2);
        let negative = build(-std::f32::consts::FRAC_PI_2);

        // Mirror the positive sweep across the axis plane (negate y) and preserve
        // the loop-cycle ORDER. A consistent (mirror, not inside-out) negative
        // sweep yields the same cyclic order; an inside-out one would reverse it.
        let mut positive_mirrored: Vec<Vec<[i32; 3]>> = positive
            .iter()
            .map(|cycle| cycle.iter().map(|p| [p[0], -p[1], p[2]]).collect())
            .collect();
        positive_mirrored.sort();

        let cycles_equal_rotation = |a: &[[i32; 3]], b: &[[i32; 3]]| -> bool {
            let n = a.len();
            n == b.len() && (0..n).any(|shift| (0..n).all(|i| a[i] == b[(i + shift) % n]))
        };

        assert_eq!(negative.len(), 4, "negative sweep has 4 walls");
        for neg_cycle in &negative {
            assert!(
                positive_mirrored
                    .iter()
                    .any(|mir| cycles_equal_rotation(neg_cycle, mir)),
                "each negative wall is the y-mirror of a positive wall with the SAME \
                 cyclic winding (consistent, not inside-out)"
            );
        }
    }

    /// §7 empty `geom`: no profile to sweep. The mesh is left intact and
    /// `geom_last.out` is empty (count `0`) — the seeded-empty slot returned
    /// unchanged. Pinned on a *non-empty* mesh so the no-op leaving the rest of the
    /// mesh untouched is observable.
    #[test]
    fn spin_empty_geom_is_noop() {
        let mut mesh = LoopMesh::new();
        // A non-empty mesh whose geometry must survive untouched.
        let (_e, _v0, _v1) = edge_profile(&mut mesh, 1.0, 0.0, 1.0);
        let before = (mesh.totvert(), mesh.totedge(), mesh.totface());

        let out = mesh.spin(
            &[],
            CENT,
            AXIS,
            None,
            std::f32::consts::FRAC_PI_2,
            4,
            false,
            false,
            false,
            None,
        );

        assert!(out.is_empty(), "empty geom returns the seeded-empty slot (count 0)");
        assert_eq!(
            (mesh.totvert(), mesh.totedge(), mesh.totface()),
            before,
            "empty geom leaves the non-empty mesh untouched"
        );
        mesh.validate_topology();
    }

    /// SPIN.md §1 / §4 `space` coordinate frame: the per-step rotate (and the
    /// `dvec` translate) act in the supplied frame, driving the conjugation
    /// `space⁻¹ · M · space` (ROTATE.md §3 / TRANSLATE.md §3) rather than the
    /// bare world rotation. We sweep an edge profile a single step (`steps = 1`,
    /// `angle = π/2`, no merge) through a **non-identity** space — a rotation
    /// about X by 90° composed with a Y translation — and pin the resulting ring
    /// verts to positions hand-derived from that same conjugation formula,
    /// independently of spin's internal path.
    ///
    /// Three things are checked: (a) the swept ring lands exactly where the
    /// documented `space⁻¹ · M_pivoted · space` map sends the profile verts;
    /// (b) the non-identity-space result *differs* from the identity (`None`)
    /// sweep of the same input (non-vacuity — `space` is genuinely consulted);
    /// and (c) a `None` space reproduces the plain world sweep byte-for-byte.
    #[test]
    fn space_frame_conjugates_per_step_rotation() {
        // The space frame: Rx(90°) composed with a translation along +Y. Built as
        // a glam `Mat4` and handed to `spin` in its column-major `[f32;16]` form,
        // exactly the convention the operator documents.
        let space_mat = Mat4::from_rotation_x(std::f32::consts::FRAC_PI_2)
            * Mat4::from_translation(Vec3::new(0.0, 2.0, 0.0));
        let space_cols = space_mat.to_cols_array();

        // Single-step quarter sweep of the headline edge profile through the space
        // frame. One step means one delta rotation by the full angle and no
        // accumulation, so each new ring vert is one matrix application of the
        // effective map on its profile source.
        let angle = std::f32::consts::FRAC_PI_2;
        let p_bottom = Vec3::new(1.0, 0.0, 0.0);
        let p_top = Vec3::new(1.0, 0.0, 1.0);

        let mut mesh = LoopMesh::new();
        let (e, _v0, _v1) = edge_profile(&mut mesh, 1.0, 0.0, 1.0);
        let out = mesh.spin(
            &[e.into()],
            CENT,
            AXIS,
            None,
            angle,
            1,
            false,
            false,
            false,
            Some(space_cols),
        );

        // §6.1 single-step edge sweep: profile (2V/1E) + one ring (2V) + one wall
        // (1E dup + 2 vertical, 1 quad). geom_last.out is the single final ring edge.
        assert_eq!((mesh.totvert(), mesh.totedge(), mesh.totface()), (4, 4, 1));
        assert_eq!(
            out.iter().filter(|g| matches!(g, GeomKey::Edge(_))).count(),
            1,
            "leading edge is the single final ring edge"
        );
        mesh.validate_topology();

        // Hand-derived expected positions: the effective world map is the
        // conjugation `space⁻¹ · M_pivoted · space`, with `M_pivoted` the rotation
        // about the axis through `cent` (here cent = origin, so M_pivoted is the
        // bare Rz(angle)). This is the documented ROTATE.md §3 frame semantics,
        // computed here independently of spin's own composition.
        let m_pivoted = Mat4::from_quat(Quat::from_axis_angle(AXIS, angle));
        let effective = space_mat.inverse() * m_pivoted * space_mat;
        let want_bottom = effective.transform_point3(p_bottom);
        let want_top = effective.transform_point3(p_top);

        // Collect the swept ring verts (the non-profile verts: those that are not
        // at the two profile positions).
        let ring_positions: Vec<Vec3> = mesh
            .vert_keys()
            .into_iter()
            .map(|v| mesh.vertex_position(v))
            .filter(|p| {
                p.distance(p_bottom) > 1e-4 && p.distance(p_top) > 1e-4
            })
            .collect();
        assert_eq!(ring_positions.len(), 2, "two swept ring verts");

        let matches_expected = |target: Vec3| {
            ring_positions.iter().any(|p| p.distance(target) < 1e-4)
        };
        assert!(
            matches_expected(want_bottom),
            "swept bottom vert at the conjugated position {want_bottom:?}; got {ring_positions:?}"
        );
        assert!(
            matches_expected(want_top),
            "swept top vert at the conjugated position {want_top:?}; got {ring_positions:?}"
        );

        // Non-vacuity: the same sweep with the *identity* (None) space lands the
        // ring at the plain world rotation — a different place than the space sweep.
        let mut world = LoopMesh::new();
        let (e2, _, _) = edge_profile(&mut world, 1.0, 0.0, 1.0);
        world.spin(
            &[e2.into()],
            CENT,
            AXIS,
            None,
            angle,
            1,
            false,
            false,
            false,
            None,
        );
        let world_ring: Vec<Vec3> = world
            .vert_keys()
            .into_iter()
            .map(|v| world.vertex_position(v))
            .filter(|p| p.distance(p_bottom) > 1e-4 && p.distance(p_top) > 1e-4)
            .collect();
        // The world sweep ring sits at Rz(π/2)·p = (0,1,*) — distinct from the
        // space-frame ring above.
        let world_bottom = m_pivoted.transform_point3(p_bottom);
        assert!(
            world_ring.iter().any(|p| p.distance(world_bottom) < 1e-4),
            "world sweep ring at the bare-rotation position {world_bottom:?}; got {world_ring:?}"
        );
        assert!(
            world_ring
                .iter()
                .all(|wp| ring_positions.iter().all(|sp| wp.distance(*sp) > 1e-3)),
            "non-identity space ring differs from the identity (None) sweep ring"
        );
    }

    /// SPIN.md §4 `space` identity equivalence: a `None` space and an explicit
    /// identity-matrix space must both reproduce the plain world sweep exactly —
    /// `None` is a behavioural no-op. Re-runs the headline quarter-pipe through
    /// `None` and through `Mat4::IDENTITY` as the space and asserts identical
    /// counts and ring geometry.
    #[test]
    fn space_none_reproduces_world_sweep() {
        let angle = std::f32::consts::FRAC_PI_2;

        let run = |space: Option<[f32; 16]>| -> (Vec<Vec3>, (usize, usize, usize)) {
            let mut mesh = LoopMesh::new();
            let (e, _, _) = edge_profile(&mut mesh, 1.0, 0.0, 1.0);
            mesh.spin(
                &[e.into()],
                CENT,
                AXIS,
                None,
                angle,
                4,
                false,
                false,
                false,
                space,
            );
            let mut ps: Vec<Vec3> = mesh
                .vert_keys()
                .into_iter()
                .map(|v| mesh.vertex_position(v))
                .collect();
            ps.sort_by(|a, b| {
                (a.x, a.y, a.z)
                    .partial_cmp(&(b.x, b.y, b.z))
                    .unwrap()
            });
            (ps, (mesh.totvert(), mesh.totedge(), mesh.totface()))
        };

        let (none_ps, none_counts) = run(None);
        let (ident_ps, ident_counts) = run(Some(Mat4::IDENTITY.to_cols_array()));

        assert_eq!(none_counts, ident_counts, "identity space matches None counts");
        assert_eq!(none_ps.len(), ident_ps.len());
        for (a, b) in none_ps.iter().zip(ident_ps.iter()) {
            assert!(
                a.distance(*b) < 1e-5,
                "identity-space vert {b:?} matches None-space vert {a:?}"
            );
        }
        // Confirm this is the genuine world quarter-pipe (matches the headline
        // counts), so "reproduces world sweep" is anchored, not vacuous.
        assert_eq!(none_counts, (10, 13, 4), "world quarter-pipe counts");
    }
}
