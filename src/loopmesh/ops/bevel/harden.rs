//! The shading post-passes: face strength and harden normals (BEVEL.md §8.10).

use super::*;

impl LoopMesh {
    /// §8.10.1 shading post-pass: stamp the named weighted-normal face-strength
    /// layer per face kind, gated by `mode`. A pure attribute pass — no geometry
    /// changes. Reconstructed vs untouched originals are recovered from the
    /// pre-bevel `original_faces` / `original_verts` snapshots; new bevel faces are
    /// `new_faces` (caps split out by `cap_faces`, everything else a strip).
    pub(super) fn stamp_face_strengths(
        &mut self,
        mode: BevelFaceStrengthMode,
        original_faces: &HashSet<FaceKey>,
        original_verts: &HashSet<VertexKey>,
        new_faces: &HashSet<FaceKey>,
        cap_faces: &HashSet<FaceKey>,
    ) {
        if mode == BevelFaceStrengthMode::None {
            return;
        }

        // Add the layer if absent, initialised to the medium sentinel (`0`) on
        // every face; an already-present layer is reused and only stamped faces
        // are overwritten (§8.10.1).
        let key = AttributeKey::custom(FACE_STRENGTH_LAYER);
        let layer = match self.face_attribs().find(&key) {
            Some(existing) => existing,
            None => self
                .register_face_attribute(ElementAttributeDesc::numeric(key, AttrType::I32))
                .expect("face i32 strength layer registers"),
        };

        // Classify against immutable snapshots first, then write — the classify
        // walk borrows `self` (face cycles), the write borrows it mutably.
        let mut writes: Vec<(FaceKey, i32)> = Vec::new();
        for face in self.face_keys() {
            if let Some(value) = self.face_strength_value(
                face,
                mode,
                original_faces,
                original_verts,
                new_faces,
                cap_faces,
            ) {
                writes.push((face, value));
            }
        }
        for (face, value) in writes {
            self.face_attribs_mut()
                .set_i32(layer, face.as_usize(), value);
        }
    }

    /// The strength to write for `face` under `mode`, or `None` when the face's
    /// kind is below the mode's threshold (left at the layer default). BEVEL.md
    /// §8.10.1's kind → strength map, gated by the nested mode scale.
    pub(super) fn face_strength_value(
        &self,
        face: FaceKey,
        mode: BevelFaceStrengthMode,
        original_faces: &HashSet<FaceKey>,
        original_verts: &HashSet<VertexKey>,
        new_faces: &HashSet<FaceKey>,
        cap_faces: &HashSet<FaceKey>,
    ) -> Option<i32> {
        if new_faces.contains(&face) {
            // A newly-created bevel face: corner-cap → weak, chamfer-strip →
            // medium; both stamped from `New` upward.
            if mode < BevelFaceStrengthMode::New {
                return None;
            }
            if cap_faces.contains(&face) {
                Some(FACE_STRENGTH_WEAK)
            } else {
                Some(FACE_STRENGTH_MEDIUM)
            }
        } else {
            let untouched = original_faces.contains(&face)
                && self
                    .face_vert_keys(face)
                    .iter()
                    .all(|vertex| original_verts.contains(vertex));
            if untouched {
                // Untouched original → strong, only at `All`.
                (mode >= BevelFaceStrengthMode::All).then_some(FACE_STRENGTH_STRONG)
            } else {
                // Reconstructed original (re-stitched / rebuilt) → strong, from
                // `Affected` upward.
                (mode >= BevelFaceStrengthMode::Affected).then_some(FACE_STRENGTH_STRONG)
            }
        }
    }

    /// Record one chamfer strip's *lengthwise* edge (BEVEL.md §8.10.2) while a
    /// hardening bevel is active, so its post-pass can recover which fan step is
    /// the shading boundary. A no-op outside a hardening bevel.
    pub(super) fn record_strip_long_edge(&mut self, p: VertexKey, q: VertexKey) {
        if self.bevel_strip_long_edges.is_none() {
            return;
        }
        if let Some(edge) = self.query_edge_exists(p, q) {
            if let Some(edges) = self.bevel_strip_long_edges.as_mut() {
                edges.push(edge);
            }
        }
    }

    /// The `harden_normals` shading post-pass (BEVEL.md §8.10.2). Marks the
    /// reconstructed↔new seam edges sharp, refreshes the face / vertex normals,
    /// then writes the target **world-space** custom split normal onto every loop
    /// of a *new* bevel face (corner-cap and chamfer-strip kinds). A pure
    /// attribute pass: no geometry created, nothing moved.
    pub(super) fn harden_normals_pass(
        &mut self,
        original_faces: &HashSet<FaceKey>,
        original_verts: &HashSet<VertexKey>,
        new_faces: &HashSet<FaceKey>,
        cap_faces: &HashSet<FaceKey>,
        long_edges: &HashSet<EdgeKey>,
    ) {
        // Refresh flat face normals and smoothed vertex normals so the reconstructed
        // faces' normals and the vertices' normals are current for the writeback.
        self.mesh_normals_update();

        // Pre-pass: mark every reconstructed-original ↔ new-bevel-face seam edge
        // sharp (the §6.7 seam edges), so a downstream loop-normal-space build keeps
        // the two sides independent. Collect first (the walk borrows `self`), set
        // after.
        let mut seam_edges: Vec<EdgeKey> = Vec::new();
        for &face in new_faces {
            for loop_key in self.face_loop_keys(face) {
                let edge = self.loop_edge(loop_key);
                if let Some(neighbour) = self.radial_other_face(edge, face) {
                    if self.harden_face_kind(
                        neighbour,
                        original_faces,
                        original_verts,
                        new_faces,
                        cap_faces,
                    ) == HardenFaceKind::Reconstructed
                    {
                        seam_edges.push(edge);
                    }
                }
            }
        }
        for edge in seam_edges {
            self.set_edge_sharp(edge, true);
        }

        // Writeback: the target world normal per new-face loop. Classify first
        // (borrows `self`), then store into the per-loop layer.
        let key = AttributeKey::custom(HARDEN_LOOP_NORMAL_LAYER);
        let layer = match self.loop_attribs().find(&key) {
            Some(existing) => existing,
            None => self
                .register_loop_attribute(ElementAttributeDesc::numeric(key, AttrType::Vec3))
                .expect("per-loop Vec3 harden-normal layer registers"),
        };
        let mut writes: Vec<(LoopKey, Vec3)> = Vec::new();
        for &face in new_faces {
            for loop_key in self.face_loop_keys(face) {
                let normal = self.harden_loop_world_normal(
                    loop_key,
                    face,
                    original_faces,
                    original_verts,
                    new_faces,
                    cap_faces,
                    long_edges,
                );
                writes.push((loop_key, normal));
            }
        }
        for (loop_key, normal) in writes {
            self.loop_attribs_mut()
                .set_vec3(layer, loop_key.as_usize(), normal.into());
        }
    }

    /// The `HardenFaceKind` of `face` for the §8.10.2 fan classification, recovered
    /// from the pristine snapshots (mirrors the §8.10.1 face-strength recovery).
    pub(super) fn harden_face_kind(
        &self,
        face: FaceKey,
        original_faces: &HashSet<FaceKey>,
        original_verts: &HashSet<VertexKey>,
        new_faces: &HashSet<FaceKey>,
        cap_faces: &HashSet<FaceKey>,
    ) -> HardenFaceKind {
        if new_faces.contains(&face) {
            if cap_faces.contains(&face) {
                HardenFaceKind::Cap
            } else {
                HardenFaceKind::Strip
            }
        } else if original_faces.contains(&face)
            && self
                .face_vert_keys(face)
                .iter()
                .all(|vertex| original_verts.contains(vertex))
        {
            HardenFaceKind::Untouched
        } else {
            // A surviving re-stitched original, or a killed-and-rebuilt original
            // (fresh key not in the snapshot): both are reconstructed originals.
            HardenFaceKind::Reconstructed
        }
    }

    /// The radial face of `edge` other than `face`, or `None` at a boundary edge.
    pub(super) fn radial_other_face(&self, edge: EdgeKey, face: FaceKey) -> Option<FaceKey> {
        self.edge_face_keys(edge).into_iter().find(|&f| f != face)
    }

    /// The target **world-space** custom split normal for one loop of a new bevel
    /// face (BEVEL.md §8.10.2, step 1 of the two-step encode). The choice keys on
    /// the two fan-neighbour faces reached across the loop's two incident edges,
    /// classified by kind — matching the adjacent original surface where one
    /// exists, blending flanking chamfer strips where none does, and reading the
    /// vertex's own smoothed normal where a corner is enclosed by caps.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn harden_loop_world_normal(
        &self,
        loop_key: LoopKey,
        face: FaceKey,
        original_faces: &HashSet<FaceKey>,
        original_verts: &HashSet<VertexKey>,
        new_faces: &HashSet<FaceKey>,
        cap_faces: &HashSet<FaceKey>,
        long_edges: &HashSet<EdgeKey>,
    ) -> Vec3 {
        let vertex = self.loop_vertex(loop_key);
        // The loop sits on two edges at `vertex`: its own outgoing edge and the
        // previous loop's edge. Each faces one neighbour in the vertex fan.
        let edge_own = self.loop_edge(loop_key);
        let edge_prev = self.loop_edge(self.loop_prev(loop_key));
        let neighbour_own = self.radial_other_face(edge_own, face);
        let neighbour_prev = self.radial_other_face(edge_prev, face);

        let kind_of = |f: Option<FaceKey>| {
            f.map(|f| {
                self.harden_face_kind(f, original_faces, original_verts, new_faces, cap_faces)
            })
        };

        let own_flat = self.face_normal(face);

        match self.harden_face_kind(face, original_faces, original_verts, new_faces, cap_faces) {
            HardenFaceKind::Strip => {
                // Only the lengthwise ("long") edge is a shading boundary; pick the
                // fan neighbour across whichever incident edge was tagged long.
                let across = if long_edges.contains(&edge_own) {
                    neighbour_own
                } else if long_edges.contains(&edge_prev) {
                    neighbour_prev
                } else {
                    // No tagged long edge at this loop's vertex (unexpected for a
                    // well-formed strip): keep the strip's own flat normal.
                    return own_flat;
                };
                match across {
                    Some(nf) => match self.harden_face_kind(
                        nf,
                        original_faces,
                        original_verts,
                        new_faces,
                        cap_faces,
                    ) {
                        HardenFaceKind::Reconstructed | HardenFaceKind::Untouched => {
                            // The chamfer's long edge meets the original surface at
                            // the seam: match that reconstructed face exactly.
                            self.face_normal(nf)
                        }
                        HardenFaceKind::Strip | HardenFaceKind::Cap => {
                            // Two chamfer strips flanking the long edge: blend them
                            // smoothly along the bevel while each cross-section stays
                            // flat.
                            LoopMesh::normalize_or_zero(own_flat + self.face_normal(nf))
                        }
                    },
                    // A long edge with no fan neighbour (open boundary): the strip's
                    // own flat normal is the only defined target.
                    None => own_flat,
                }
            }
            HardenFaceKind::Cap => {
                let k_own = kind_of(neighbour_own);
                let k_prev = kind_of(neighbour_prev);
                let both_caps =
                    k_own == Some(HardenFaceKind::Cap) && k_prev == Some(HardenFaceKind::Cap);
                if both_caps {
                    // Enclosed entirely by cap faces: the corner reads as a smooth
                    // rounded point at the vertex's own smoothed normal.
                    return self.vertex_normal(vertex);
                }
                // A reconstructed / untouched original neighbour matches the original
                // surface at the corner seam. Prefer the own-edge side when both
                // sides are reconstructed (see the §8.10.2 tie-break note).
                let reconstructed = |k: Option<HardenFaceKind>| {
                    matches!(
                        k,
                        Some(HardenFaceKind::Reconstructed) | Some(HardenFaceKind::Untouched)
                    )
                };
                if reconstructed(k_own) {
                    return self.face_normal(neighbour_own.unwrap());
                }
                if reconstructed(k_prev) {
                    return self.face_normal(neighbour_prev.unwrap());
                }
                // Otherwise (a mixed cap/strip neighbourhood with no reconstructed
                // face directly adjacent): the §8.10.2 "one fan-step further" case.
                // The junction-patch corner it targets is not exercised on the
                // reachable drivers this cycle; fall back to the vertex's own
                // smoothed normal, which keeps the corner rounded.
                self.vertex_normal(vertex)
            }
            // A hardened loop only ever belongs to a new bevel face; other kinds are
            // never enqueued by the writeback.
            HardenFaceKind::Reconstructed | HardenFaceKind::Untouched => own_flat,
        }
    }

    /// The stored §8.10.2 target world custom split normal for `loop_key`, or
    /// `None` when the hardening pass did not write it (not a new bevel face's
    /// loop, or `harden_normals` was off). The value is the decoded world vector.
    ///
    /// A written normal is always unit length, so the layer's all-zero default —
    /// the value read at a loop the pass never touched, once the layer exists —
    /// reads back as `None` (unwritten), distinct from any hardened value.
    pub fn loop_hardened_normal(&self, loop_key: LoopKey) -> Option<Vec3> {
        let layer = self
            .loop_attribs()
            .find(&AttributeKey::custom(HARDEN_LOOP_NORMAL_LAYER))?;
        self.loop_attribs()
            .get_vec3(layer, loop_key.as_usize())
            .map(Vec3::from)
            .filter(|normal| *normal != Vec3::ZERO)
    }
}

/// Face-provenance kind for the §8.10.2 fan classification, re-grouped from the
/// bevel's `Cap` / `Strip` recording plus the reconstructed / untouched original
/// recovery (a compatible re-grouping of §6.6.1, per §8.10).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum HardenFaceKind {
    Cap,
    Strip,
    Reconstructed,
    Untouched,
}
