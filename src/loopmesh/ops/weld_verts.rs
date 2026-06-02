use std::collections::HashMap;

use glam::Vec3;
use smallvec::SmallVec;

use crate::loopmesh::*;

/// The corner ring of a rebuilt face: the ordered survivor verts and, between
/// each consecutive pair, the survivor edge that joins them (§5.1 / §5.2). A
/// rebuilt face is `face_create`d from these two parallel arrays.
struct RebuiltRing {
    verts: SmallVec<[VertexKey; 8]>,
    edges: SmallVec<[EdgeKey; 8]>,
    /// The original loop sitting at each surviving corner, parallel to `verts`,
    /// so the per-loop customdata of the rebuilt face can be seeded from the
    /// matching source loop (§5.4).
    source_loops: SmallVec<[LoopKey; 8]>,
    /// `true` when the rebuild walk landed on a vert already placed in the ring
    /// — a self-touch the §4 pre-split was meant to prevent but a pathological
    /// many-to-one map can still reach (§5.3 bail guard).
    has_repeated_vert: bool,
}

impl LoopMesh {
    /// Weld every source vertex onto its target: for each `(source, target)`
    /// couple the source is removed and every incident edge and face is
    /// rewritten as though it had been at the target all along.
    ///
    /// Implements `radial-mesh-guides/operators/WELD_VERTS.md` §3–§6:
    ///
    /// - **§3 Phase A** — mark every source for deletion, fold its
    ///   selection header flag onto its target (the hidden bit is *not*
    ///   force-merged), and, when `use_centroid` is set, move each target to the
    ///   centroid of itself and the sources welding onto it before any topology
    ///   surgery (§3.3). Source per-vert customdata is discarded with the
    ///   source; the target keeps its own (§5.4).
    /// - **§4 Phase B** — pre-split any face two of whose corners are about to
    ///   become the same vert *non-adjacently*: the face is chorded along the
    ///   diagonal between those two corners (SFME, [`Self::face_split`]) so the
    ///   merge happens on two well-formed sub-faces instead of one
    ///   self-touching one. A face may need several such cuts; each sub-face is
    ///   re-examined until no non-adjacent self-merge remains (§4.2).
    /// - **§5.1 edge pass** — substitute each edge's endpoints. An edge whose
    ///   two substituted endpoints coincide collapses to zero length: it is
    ///   marked with the transient collapse-bit and for deletion. An edge whose
    ///   endpoints differ is re-pointed to the survivor edge between them —
    ///   reused if one already exists, else created carrying the old edge's
    ///   attributes — and the old edge is marked for deletion (§5.4 seeds the
    ///   created survivor's customdata; reused survivors keep their own).
    /// - **§5.2 / §5.3 face pass** — every face touching a source is marked for
    ///   deletion and rebuilt around the substituted verts, dropping collapsed
    ///   edges. A face whose substituted corner count falls below 3 is dropped
    ///   without a replacement (§5.2). Otherwise the rebuilt corner ring either
    ///   adopts a pre-existing duplicate face (the no-coincident-faces
    ///   invariant, §5.3), bails on a self-touch (§5.3), or is created fresh
    ///   with the original's per-face / per-loop customdata copied across
    ///   (§5.4).
    /// - **§6 Phase D** — a single deferred sweep removes every element
    ///   carrying the delete-bit (collapsed edges, re-pointed old edges, source
    ///   verts, rebuilt/dropped original faces) in face → edge → vert order.
    ///
    /// The map is taken as a flat list of `(source, target)` couples. It is a
    /// relation, not resolved transitively: a couple whose target is itself a
    /// source elsewhere is substituted by its *direct* value exactly once
    /// (§7's garbage-in-garbage-out contract); callers that need a chained map
    /// flattened must pre-flatten it.
    pub fn weld_verts(&mut self, pairs: &[(VertexKey, VertexKey)], use_centroid: bool) {
        if pairs.is_empty() {
            return;
        }

        let mut target_of: HashMap<VertexKey, VertexKey> = HashMap::with_capacity(pairs.len());
        for &(source, target) in pairs {
            target_of.insert(source, target);
        }

        // Delete-bit (§3.1 / §6): every element flagged here is removed by the
        // single deferred sweep. The collapse-bit (§5.1) marks edges that
        // pinched to zero length so the face pass can drop their corner.
        let delete_vert = self.push_vert_op_flag_layer();
        let delete_edge = self.push_edge_op_flag_layer();
        let collapse_edge = self.push_edge_op_flag_layer();
        let delete_face = self.push_face_op_flag_layer();
        const BIT: u8 = 0;

        self.weld_phase_mark_and_resolve(&target_of, use_centroid, delete_vert, BIT);
        self.weld_phase_presplit_self_merges(&target_of);
        self.weld_phase_edge_pass(&target_of, delete_edge, collapse_edge, BIT);
        self.weld_phase_face_pass(&target_of, delete_face, collapse_edge, BIT);
        self.weld_phase_delete_tagged(delete_face, delete_edge, delete_vert, BIT);

        self.pop_face_op_flag_layer(delete_face);
        self.pop_edge_op_flag_layer(collapse_edge);
        self.pop_edge_op_flag_layer(delete_edge);
        self.pop_vert_op_flag_layer(delete_vert);
    }

    /// §3 Phase A — mark sources, fold header flags onto targets, and resolve
    /// centroid positions (when enabled) before any topology surgery.
    fn weld_phase_mark_and_resolve(
        &mut self,
        target_of: &HashMap<VertexKey, VertexKey>,
        use_centroid: bool,
        delete_vert: crate::op_flags::VertOpFlagLayer,
        bit: u8,
    ) {
        // §3.3 — sources grouped by target, gathered while marking so the
        // centroid average sees every cluster member.
        let mut clusters: HashMap<VertexKey, SmallVec<[VertexKey; 4]>> = HashMap::new();

        for (&source, &target) in target_of {
            if self.vertices.try_get(source).is_none() {
                continue;
            }
            // §3.1 — the source is removed in the final sweep.
            self.vert_op_flag_set(delete_vert, source, bit);
            // §3.2 — fold the source's selection onto the target; the hidden
            // bit is intentionally not force-merged so welding a visible vert
            // onto a hidden one does not reveal the result.
            if self.vertices.get(source).eflags.contains(ElementFlags::SELECTED)
                && self.vertices.try_get(target).is_some()
            {
                self.vertices.get_mut(target).eflags |= ElementFlags::SELECTED;
            }
            if use_centroid && source != target && self.vertices.try_get(target).is_some() {
                clusters.entry(target).or_default().push(source);
            }
        }

        // §3.3 — move each target to the average of itself and its cluster.
        if use_centroid {
            for (&target, sources) in &clusters {
                if self.vertices.try_get(target).is_none() {
                    continue;
                }
                let mut sum: Vec3 = self.vertex_position(target);
                for &source in sources {
                    sum += self.vertex_position(source);
                }
                let centroid = sum / (sources.len() + 1) as f32;
                self.set_vertex_position(target, centroid);
            }
        }
    }

    /// §4 Phase B — split every face that has a non-adjacent self-merge so the
    /// later passes never see a face that would visit one vert twice.
    fn weld_phase_presplit_self_merges(&mut self, target_of: &HashMap<VertexKey, VertexKey>) {
        // The set of faces still to examine. A fresh sub-face produced by a cut
        // is pushed back on so it is re-examined for further pinches (§4.2).
        let mut pending: SmallVec<[FaceKey; 16]> = SmallVec::new();
        let mut face_iter = MeshFaces::default();
        while let Some(face) = face_iter.next(self) {
            pending.push(face);
        }

        while let Some(face) = pending.pop() {
            if self.faces.try_get(face).is_none() {
                continue;
            }
            let Some((v_corner, v_target)) = self.weld_find_nonadjacent_self_merge(face, target_of)
            else {
                continue;
            };
            // The chord cuts `face` into two well-formed sub-faces; both are
            // re-examined for further pinches.
            if let Some((new_face, _new_edge)) = self.face_split(face, v_corner, v_target) {
                pending.push(face);
                pending.push(new_face);
            }
        }
    }

    /// §4.1 — find a corner `l` of `face` whose vert is a source and whose
    /// target `v_tar` is also a corner of `face`, non-adjacent to `l`. Returns
    /// `(l.vert, v_tar)`, the chord endpoints to split along; `None` when the
    /// face has no non-adjacent self-merge.
    fn weld_find_nonadjacent_self_merge(
        &self,
        face: FaceKey,
        target_of: &HashMap<VertexKey, VertexKey>,
    ) -> Option<(VertexKey, VertexKey)> {
        let corners: SmallVec<[(LoopKey, VertexKey); 8]> = {
            let mut out: SmallVec<[(LoopKey, VertexKey); 8]> = SmallVec::new();
            let mut loops = self.face_loops(face);
            while let Some(loop_key) = loops.next(self) {
                out.push((loop_key, self.loops.get(loop_key).vertex));
            }
            out
        };
        let count = corners.len();

        for index in 0..count {
            let (corner_loop, corner_vert) = corners[index];
            let Some(&target) = target_of.get(&corner_vert) else {
                continue;
            };
            if target == corner_vert {
                continue;
            }
            // Does the same face visit `target` at another corner, and is that
            // corner non-adjacent to `corner_loop`?
            for other_index in 0..count {
                if corners[other_index].1 != target {
                    continue;
                }
                let is_adjacent = other_index == index
                    || (index + 1) % count == other_index
                    || (other_index + 1) % count == index;
                if !is_adjacent {
                    let _ = corner_loop;
                    return Some((corner_vert, target));
                }
            }
        }
        None
    }

    /// §5.1 edge pass — substitute every edge's endpoints, collapsing edges
    /// that pinch to a point and re-pointing the rest onto a survivor edge.
    ///
    /// The walk re-visits the edges it *creates* (§5.1.1): a freshly minted
    /// survivor edge is pushed onto the work list, so a re-pointed endpoint that
    /// is itself a source (a chained map, §7) fires the rule again and the
    /// surviving edge is carried step-by-step to the chain's terminal value.
    /// Each intermediate edge along the way is delete-marked. This is *not* a
    /// transitive map lookup — it is the one-step rule iterated over the pass's
    /// own output — and it is specific to the edge pass; the §5.2 ring rebuild
    /// stays strictly direct-once.
    fn weld_phase_edge_pass(
        &mut self,
        target_of: &HashMap<VertexKey, VertexKey>,
        delete_edge: crate::op_flags::EdgeOpFlagLayer,
        collapse_edge: crate::op_flags::EdgeOpFlagLayer,
        bit: u8,
    ) {
        // Seeded with the live edge set; survivor edges created below are
        // appended so the same walk re-visits them (§5.1.1). A cursor advances
        // through the list rather than `pop`ping so the order matches a single
        // forward walk over the growing edge set.
        let mut pending: SmallVec<[EdgeKey; 32]> = collect_iter::<_, 32>(self, MeshEdges::default());
        let mut cursor = 0;
        while cursor < pending.len() {
            let edge = pending[cursor];
            cursor += 1;
            if self.edges.try_get(edge).is_none() {
                continue;
            }
            let [v1, v2] = self.edge_vert_keys(edge);
            let v1_sub = target_of.get(&v1).copied().unwrap_or(v1);
            let v2_sub = target_of.get(&v2).copied().unwrap_or(v2);

            if v1_sub == v1 && v2_sub == v2 {
                // Neither endpoint is a source — covers a terminal created edge
                // whose re-pointed endpoint was a non-source vert, which is left
                // alone and survives (§5.1.1 step 3).
                continue;
            }
            if v1_sub == v2_sub {
                // §5.1 — both endpoints became the same vert: zero-length edge.
                self.edge_op_flag_set(collapse_edge, edge, bit);
                self.edge_op_flag_set(delete_edge, edge, bit);
                continue;
            }

            // §5.1 — re-point onto the survivor edge between the substituted
            // endpoints: reuse an existing one, else create one seeded from the
            // old edge's customdata (§5.4).
            let survivor = match self.query_edge_exists(v1_sub, v2_sub) {
                Some(existing) => existing,
                None => {
                    let created = self.create_edge([v1_sub, v2_sub], EdgeCreateFlags::NONE);
                    self.edge_attribs.copy_into(created.as_usize(), edge.as_usize());
                    // §5.1.1 — re-visit the created edge: if either substituted
                    // endpoint is itself a source, the rule fires again and
                    // chain-flattens the surviving edge forward.
                    pending.push(created);
                    created
                }
            };
            // §5.4 — fold the old edge's selection onto the survivor (hidden
            // not force-merged, mirroring §3.2).
            if self.edges.get(edge).eflags.contains(ElementFlags::SELECTED) {
                self.edges.get_mut(survivor).eflags |= ElementFlags::SELECTED;
            }
            // §5.1 / §5.1.1 — the old edge's role is taken over by the survivor;
            // an intermediate created edge re-pointed again is itself marked
            // here on its re-visit.
            self.edge_op_flag_set(delete_edge, edge, bit);
        }
    }

    /// §5.2 / §5.3 face pass — rebuild every face touching a source around the
    /// substituted verts, dropping degenerate faces, adopting duplicates, and
    /// creating the survivors.
    fn weld_phase_face_pass(
        &mut self,
        target_of: &HashMap<VertexKey, VertexKey>,
        delete_face: crate::op_flags::FaceOpFlagLayer,
        collapse_edge: crate::op_flags::EdgeOpFlagLayer,
        bit: u8,
    ) {
        let faces: SmallVec<[FaceKey; 32]> = collect_iter::<_, 32>(self, MeshFaces::default());
        for face in faces {
            if self.faces.try_get(face).is_none() {
                continue;
            }
            if !self.weld_face_touches_source(face, target_of) {
                continue;
            }

            // §5.2 step 1 — the original face dies regardless of what replaces
            // it (rebuilt survivor, adopted duplicate, or nothing).
            self.face_op_flag_set(delete_face, face, bit);

            // §5.2 step 2 — substituted corner count = len − collapsed edges.
            let collapsed = self.weld_count_collapse_edges(face, collapse_edge, bit);
            if self.face_len(face).saturating_sub(collapsed) < 3 {
                continue;
            }

            // §5.2 step 3 — the substituted ring, skipping collapsed steps.
            let ring = self.weld_rebuild_ring(face, target_of, collapse_edge, bit);
            if ring.verts.len() < 3 {
                continue;
            }

            // §5.3 — adopt a pre-existing duplicate (no two faces with the same
            // corner cycle coexist).
            if let Some(existing) = self.face_exists(&ring.verts) {
                if self.faces.get(face).eflags.contains(ElementFlags::SELECTED) {
                    self.faces.get_mut(existing).eflags |= ElementFlags::SELECTED;
                }
                continue;
            }
            // §5.3 — conservative self-touch bail.
            if ring.has_repeated_vert {
                continue;
            }

            // §5.3 / §5.4 — create the replacement, seeding per-face and
            // per-loop customdata from the original face.
            let new_face = self.face_create(&ring.verts, &ring.edges, FaceCreateFlags::NONE);
            self.face_attribs.copy_into(new_face.as_usize(), face.as_usize());
            if !self.loop_attribs.is_empty() {
                self.weld_copy_surviving_loop_cd(new_face, &ring.source_loops);
            }
        }
    }

    /// True iff any corner of `face` is a source vert (§5.2 pre-scan).
    fn weld_face_touches_source(
        &self,
        face: FaceKey,
        target_of: &HashMap<VertexKey, VertexKey>,
    ) -> bool {
        let mut loops = self.face_loops(face);
        while let Some(loop_key) = loops.next(self) {
            if target_of.contains_key(&self.loops.get(loop_key).vertex) {
                return true;
            }
        }
        false
    }

    /// Count `face`'s perimeter edges carrying the collapse-bit (§5.2 step 2).
    fn weld_count_collapse_edges(
        &self,
        face: FaceKey,
        collapse_edge: crate::op_flags::EdgeOpFlagLayer,
        bit: u8,
    ) -> usize {
        let mut count = 0usize;
        let mut loops = self.face_loops(face);
        while let Some(loop_key) = loops.next(self) {
            let edge = self.loops.get(loop_key).edge;
            if !edge.is_null() && self.edge_op_flag_test(collapse_edge, edge, bit) {
                count += 1;
            }
        }
        count
    }

    /// §5.2 step 3 — build the rebuilt corner ring of `face`: walk the loop
    /// cycle, substitute each vert, and emit a corner for each step whose two
    /// consecutive substituted verts differ (a collapsed edge contributes no
    /// corner). The survivor edge joining each consecutive pair (§5.1) is
    /// resolved by lookup.
    fn weld_rebuild_ring(
        &self,
        face: FaceKey,
        target_of: &HashMap<VertexKey, VertexKey>,
        collapse_edge: crate::op_flags::EdgeOpFlagLayer,
        bit: u8,
    ) -> RebuiltRing {
        // (loop, substituted vert) for every corner.
        let mut corners: SmallVec<[(LoopKey, VertexKey); 8]> = SmallVec::new();
        let mut loops = self.face_loops(face);
        while let Some(loop_key) = loops.next(self) {
            let vert = self.loops.get(loop_key).vertex;
            let sub = target_of.get(&vert).copied().unwrap_or(vert);
            corners.push((loop_key, sub));
        }

        let mut verts: SmallVec<[VertexKey; 8]> = SmallVec::new();
        let mut source_loops: SmallVec<[LoopKey; 8]> = SmallVec::new();
        let count = corners.len();
        for index in 0..count {
            let (loop_key, sub) = corners[index];
            let next_sub = corners[(index + 1) % count].1;
            // A step onto an identical substituted vert is the collapsed edge;
            // it contributes no corner.
            if sub == next_sub {
                let _ = collapse_edge;
                let _ = bit;
                continue;
            }
            verts.push(sub);
            source_loops.push(loop_key);
        }

        // Resolve the survivor edge between each consecutive surviving vert.
        let mut edges: SmallVec<[EdgeKey; 8]> = SmallVec::new();
        let ring_len = verts.len();
        let mut has_repeated_vert = false;
        for index in 0..ring_len {
            let a = verts[index];
            let b = verts[(index + 1) % ring_len];
            match self.query_edge_exists(a, b) {
                Some(edge) => edges.push(edge),
                None => {
                    // Every surviving corner-step should have a survivor edge
                    // from the §5.1 pass; a missing one means the ring folded
                    // back on itself.
                    has_repeated_vert = true;
                    edges.push(EdgeKey::NULL);
                }
            }
        }

        // §5.3 — detect a vert reused twice in the ring.
        for index in 0..ring_len {
            for other in (index + 1)..ring_len {
                if verts[index] == verts[other] {
                    has_repeated_vert = true;
                }
            }
        }

        RebuiltRing {
            verts,
            edges,
            source_loops,
            has_repeated_vert,
        }
    }

    /// §5.4 — copy each surviving corner's per-loop customdata from the
    /// matching original loop onto the rebuilt face's loops. The rebuilt face's
    /// loops are walked in the same order the ring was built, so they line up
    /// with `source_loops` by index.
    fn weld_copy_surviving_loop_cd(&mut self, new_face: FaceKey, source_loops: &[LoopKey]) {
        let new_loops: SmallVec<[LoopKey; 8]> = {
            let mut out: SmallVec<[LoopKey; 8]> = SmallVec::new();
            let mut loops = self.face_loops(new_face);
            while let Some(loop_key) = loops.next(self) {
                out.push(loop_key);
            }
            out
        };
        for (new_loop, &source_loop) in new_loops.iter().zip(source_loops.iter()) {
            self.loop_attribs
                .copy_into(new_loop.as_usize(), source_loop.as_usize());
        }
    }

    /// §6 Phase D — the single deferred delete sweep, in face → edge → vert
    /// order so no dangling reference survives a step. Faces are killed first
    /// (which also frees their loops), then the flagged edges, then the source
    /// verts.
    fn weld_phase_delete_tagged(
        &mut self,
        delete_face: crate::op_flags::FaceOpFlagLayer,
        delete_edge: crate::op_flags::EdgeOpFlagLayer,
        delete_vert: crate::op_flags::VertOpFlagLayer,
        bit: u8,
    ) {
        let mut faces_to_kill: SmallVec<[FaceKey; 16]> = SmallVec::new();
        let mut face_iter = MeshFaces::default();
        while let Some(face) = face_iter.next(self) {
            if self.face_op_flag_test(delete_face, face, bit) {
                faces_to_kill.push(face);
            }
        }
        for face in faces_to_kill {
            if self.faces.try_get(face).is_some() {
                self.face_kill(face);
            }
        }

        let mut edges_to_kill: SmallVec<[EdgeKey; 16]> = SmallVec::new();
        let mut edge_iter = MeshEdges::default();
        while let Some(edge) = edge_iter.next(self) {
            if self.edge_op_flag_test(delete_edge, edge, bit) {
                edges_to_kill.push(edge);
            }
        }
        for edge in edges_to_kill {
            if self.edges.try_get(edge).is_some() {
                self.edge_kill(edge);
            }
        }

        let mut verts_to_kill: SmallVec<[VertexKey; 16]> = SmallVec::new();
        let mut vert_iter = MeshVerts::default();
        while let Some(vert) = vert_iter.next(self) {
            if self.vert_op_flag_test(delete_vert, vert, bit) {
                verts_to_kill.push(vert);
            }
        }
        for vert in verts_to_kill {
            if self.vertices.try_get(vert).is_some() {
                self.vertex_kill(vert);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use glam::Vec3;

    use crate::loopmesh::*;

    fn quad(mesh: &mut LoopMesh, positions: [(f32, f32); 4]) -> (Vec<VertexKey>, FaceKey) {
        let verts: Vec<VertexKey> = positions
            .into_iter()
            .map(|(x, y)| mesh.create_vertex(Vec3::new(x, y, 0.0)))
            .collect();
        let edges: Vec<EdgeKey> = (0..4)
            .map(|i| {
                mesh.create_edge(
                    [verts[i], verts[(i + 1) % 4]],
                    EdgeCreateFlags::NO_DUPLICATE,
                )
            })
            .collect();
        let face = mesh.face_create(&verts, &edges, FaceCreateFlags::NONE);
        (verts, face)
    }

    fn polygon(mesh: &mut LoopMesh, positions: &[(f32, f32)]) -> (Vec<VertexKey>, FaceKey) {
        let n = positions.len();
        let verts: Vec<VertexKey> = positions
            .iter()
            .map(|&(x, y)| mesh.create_vertex(Vec3::new(x, y, 0.0)))
            .collect();
        let edges: Vec<EdgeKey> = (0..n)
            .map(|i| {
                mesh.create_edge(
                    [verts[i], verts[(i + 1) % n]],
                    EdgeCreateFlags::NO_DUPLICATE,
                )
            })
            .collect();
        let face = mesh.face_create(&verts, &edges, FaceCreateFlags::NONE);
        (verts, face)
    }

    /// §7 empty-map fast path: nothing is marked, nothing changes.
    #[test]
    fn weld_verts_empty_map_is_noop() {
        let mut mesh = LoopMesh::new();
        let (_verts, _face) = quad(&mut mesh, [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]);
        let (v0, e0, f0) = (mesh.totvert(), mesh.totedge(), mesh.totface());

        mesh.weld_verts(&[], false);

        assert_eq!((mesh.totvert(), mesh.totedge(), mesh.totface()), (v0, e0, f0));
        mesh.validate_topology();
    }

    /// §5.2 / §7 — a triangle two of whose corners merge drops below 3 distinct
    /// corners and vanishes, leaving a single edge between the survivors.
    #[test]
    fn weld_verts_triangle_two_corners_reaped() {
        let mut mesh = LoopMesh::new();
        let (verts, _face) = polygon(&mut mesh, &[(0.0, 0.0), (1.0, 0.0), (0.5, 1.0)]);

        mesh.weld_verts(&[(verts[0], verts[1])], false);

        assert_eq!(mesh.totface(), 0, "the degenerate triangle is reaped");
        assert_eq!(mesh.totvert(), 2, "source corner removed");
        let live: std::collections::HashSet<VertexKey> = mesh.vert_keys().into_iter().collect();
        assert!(!live.contains(&verts[0]), "source vert gone");
        assert!(live.contains(&verts[1]) && live.contains(&verts[2]), "survivors stay");
        assert_eq!(mesh.totedge(), 1, "one surviving edge between the survivors");
        let endpoints: std::collections::HashSet<VertexKey> =
            mesh.edge_vert_keys(mesh.edge_keys()[0]).into_iter().collect();
        assert_eq!(endpoints, [verts[1], verts[2]].into_iter().collect());
        mesh.validate_topology();
    }

    /// §5.2 — welding two opposite corners of a quad drops the whole face: each
    /// half-triangle of the implicit diagonal collapses below 3 corners.
    #[test]
    fn weld_verts_quad_opposite_corners_face_disappears() {
        let mut mesh = LoopMesh::new();
        let (verts, _face) = quad(&mut mesh, [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]);

        mesh.weld_verts(&[(verts[0], verts[2])], false);

        assert_eq!(mesh.totface(), 0, "the quad disappears entirely");
        let live: std::collections::HashSet<VertexKey> = mesh.vert_keys().into_iter().collect();
        assert!(!live.contains(&verts[0]), "source corner gone");
        assert!(live.contains(&verts[2]), "target corner survives");
        mesh.validate_topology();
    }

    /// §4.2 pre-split pin — a pentagon welding two non-adjacent corners keeps a
    /// surviving triangle only because the chord is cut first. Without the
    /// §4 pre-split the whole pentagon would hit the §5.3 self-touch bail.
    #[test]
    fn weld_verts_pentagon_nonadjacent_self_merge_keeps_triangle() {
        let mut mesh = LoopMesh::new();
        let (verts, _face) = polygon(
            &mut mesh,
            &[(0.0, 0.0), (2.0, 0.0), (2.5, 1.5), (1.0, 2.5), (-0.5, 1.5)],
        );

        // Corners 0 and 3 are non-adjacent on the pentagon.
        mesh.weld_verts(&[(verts[0], verts[3])], false);

        assert_eq!(mesh.totface(), 1, "exactly one sub-face survives the pentagon weld");
        let surviving = mesh.face_keys()[0];
        assert_eq!(mesh.face_len(surviving), 3, "the survivor is a triangle");
        let ring: std::collections::HashSet<VertexKey> =
            mesh.face_vert_keys(surviving).into_iter().collect();
        // The surviving triangle is (1, 2, 3) — vert 0 substituted onto 3.
        assert_eq!(
            ring,
            [verts[1], verts[2], verts[3]].into_iter().collect(),
            "surviving triangle is (1,2,3)"
        );
        let live: std::collections::HashSet<VertexKey> = mesh.vert_keys().into_iter().collect();
        assert!(!live.contains(&verts[0]), "welded corner gone");
        mesh.validate_topology();
    }

    /// §5.1 / §5.2 — welding one shared-edge endpoint of two quads onto the
    /// other collapses the shared edge; each quad drops a corner to a triangle.
    #[test]
    fn weld_verts_shared_edge_endpoint_collapses() {
        // Two quads sharing the vertical edge {b, c}:
        //   left quad  (a0, b, c, a1), right quad (b, d0, d1, c).
        let mut mesh = LoopMesh::new();
        let a0 = mesh.create_vertex(Vec3::new(0.0, 0.0, 0.0));
        let b = mesh.create_vertex(Vec3::new(1.0, 0.0, 0.0));
        let c = mesh.create_vertex(Vec3::new(1.0, 1.0, 0.0));
        let a1 = mesh.create_vertex(Vec3::new(0.0, 1.0, 0.0));
        let d0 = mesh.create_vertex(Vec3::new(2.0, 0.0, 0.0));
        let d1 = mesh.create_vertex(Vec3::new(2.0, 1.0, 0.0));

        let mk = |mesh: &mut LoopMesh, vs: [VertexKey; 4]| {
            let es: Vec<EdgeKey> = (0..4)
                .map(|i| mesh.create_edge([vs[i], vs[(i + 1) % 4]], EdgeCreateFlags::NO_DUPLICATE))
                .collect();
            mesh.face_create(&vs, &es, FaceCreateFlags::NONE)
        };
        mk(&mut mesh, [a0, b, c, a1]);
        mk(&mut mesh, [b, d0, d1, c]);

        assert_eq!(mesh.totface(), 2);

        // Weld b onto c: the shared edge {b, c} collapses to nothing.
        mesh.weld_verts(&[(b, c)], false);

        let live: std::collections::HashSet<VertexKey> = mesh.vert_keys().into_iter().collect();
        assert!(!live.contains(&b), "welded vert gone");
        assert!(live.contains(&c), "target survives");
        assert_eq!(mesh.totface(), 2, "both quads survive as triangles");
        for face in mesh.face_keys() {
            assert_eq!(mesh.face_len(face), 3, "each quad lost the collapsed corner");
        }
        // The collapsed shared edge is gone: no surviving edge touches the dead
        // vert `b`, and no edge is a self-loop at `c`.
        for edge in mesh.edge_keys() {
            let [e0, e1] = mesh.edge_vert_keys(edge);
            assert!(e0 != b && e1 != b, "no surviving edge touches the welded vert");
            assert!(e0 != e1, "no surviving edge is a collapsed self-loop");
        }
        mesh.validate_topology();
    }

    /// §3 / §7 — many sources onto one target (pointmerge): a fan triangle all
    /// of whose corners merge onto one survivor degenerates and drops; the
    /// survivor stays at its own position with `use_centroid` off.
    #[test]
    fn weld_verts_pointmerge_cluster_centroid_off() {
        let mut mesh = LoopMesh::new();
        // A triangle fan: target t plus three sources around it.
        let t = mesh.create_vertex(Vec3::new(0.0, 0.0, 0.0));
        let s0 = mesh.create_vertex(Vec3::new(1.0, 0.0, 0.0));
        let s1 = mesh.create_vertex(Vec3::new(0.0, 1.0, 0.0));
        let s2 = mesh.create_vertex(Vec3::new(-1.0, -1.0, 0.0));
        // One triangle on the three sources — every corner welds onto t.
        let e0 = mesh.create_edge([s0, s1], EdgeCreateFlags::NO_DUPLICATE);
        let e1 = mesh.create_edge([s1, s2], EdgeCreateFlags::NO_DUPLICATE);
        let e2 = mesh.create_edge([s2, s0], EdgeCreateFlags::NO_DUPLICATE);
        mesh.face_create(&[s0, s1, s2], &[e0, e1, e2], FaceCreateFlags::NONE);

        let t_pos = mesh.vertex_position(t);

        mesh.weld_verts(&[(s0, t), (s1, t), (s2, t)], false);

        assert_eq!(mesh.totvert(), 1, "all sources merge onto the one target");
        assert_eq!(mesh.totface(), 0, "the fully-merged triangle drops");
        let survivor = mesh.vert_keys()[0];
        assert_eq!(survivor, t, "the target is the survivor");
        assert_eq!(
            mesh.vertex_position(t),
            t_pos,
            "with use_centroid off the survivor keeps its own position"
        );
        mesh.validate_topology();
    }

    /// §3.3 — `use_centroid` moves the survivor to the average of itself and
    /// its sources; the topology outcome is identical to the off case.
    #[test]
    fn weld_verts_pointmerge_cluster_centroid_on() {
        let mut mesh = LoopMesh::new();
        let t = mesh.create_vertex(Vec3::new(0.0, 0.0, 0.0));
        let s0 = mesh.create_vertex(Vec3::new(3.0, 0.0, 0.0));
        let s1 = mesh.create_vertex(Vec3::new(0.0, 3.0, 0.0));
        let s2 = mesh.create_vertex(Vec3::new(-3.0, -3.0, 0.0));
        let e0 = mesh.create_edge([s0, s1], EdgeCreateFlags::NO_DUPLICATE);
        let e1 = mesh.create_edge([s1, s2], EdgeCreateFlags::NO_DUPLICATE);
        let e2 = mesh.create_edge([s2, s0], EdgeCreateFlags::NO_DUPLICATE);
        mesh.face_create(&[s0, s1, s2], &[e0, e1, e2], FaceCreateFlags::NONE);

        let expected = (Vec3::ZERO
            + Vec3::new(3.0, 0.0, 0.0)
            + Vec3::new(0.0, 3.0, 0.0)
            + Vec3::new(-3.0, -3.0, 0.0))
            / 4.0;

        mesh.weld_verts(&[(s0, t), (s1, t), (s2, t)], true);

        assert_eq!(mesh.totvert(), 1, "all sources merge onto the one target");
        let survivor = mesh.vert_keys()[0];
        let pos = mesh.vertex_position(survivor);
        assert!(
            (pos - expected).length() < 1e-6,
            "survivor moved to the cluster centroid: got {pos:?}, expected {expected:?}"
        );
        mesh.validate_topology();
    }

    /// §7 — an arbitrary, disconnected map entry on a bowtie: the two outer
    /// corners of one wing share no edge, so they simply fuse with no face
    /// self-merge. The wing triangle loses one corner and is reaped.
    #[test]
    fn weld_verts_bowtie_disconnected_pair() {
        let mut mesh = LoopMesh::new();
        // Two triangles sharing only the apex `hub`.
        let hub = mesh.create_vertex(Vec3::new(0.0, 0.0, 0.0));
        let a0 = mesh.create_vertex(Vec3::new(-1.0, 1.0, 0.0));
        let a1 = mesh.create_vertex(Vec3::new(-1.0, -1.0, 0.0));
        let b0 = mesh.create_vertex(Vec3::new(1.0, 1.0, 0.0));
        let b1 = mesh.create_vertex(Vec3::new(1.0, -1.0, 0.0));

        let mk = |mesh: &mut LoopMesh, vs: [VertexKey; 3]| {
            let es: Vec<EdgeKey> = (0..3)
                .map(|i| mesh.create_edge([vs[i], vs[(i + 1) % 3]], EdgeCreateFlags::NO_DUPLICATE))
                .collect();
            mesh.face_create(&vs, &es, FaceCreateFlags::NONE)
        };
        mk(&mut mesh, [hub, a0, a1]);
        mk(&mut mesh, [hub, b0, b1]);

        // a0 (a corner of wing A) and b0 (a corner of wing B) share no edge:
        // an arbitrary, disconnected map entry. They simply fuse; no face
        // self-merges.
        let _ = a1;
        let _ = b1;
        mesh.weld_verts(&[(a0, b0)], false);

        let live: std::collections::HashSet<VertexKey> = mesh.vert_keys().into_iter().collect();
        assert!(!live.contains(&a0), "welded source gone");
        assert!(live.contains(&b0), "target survives");
        // Both triangles survive: wing A becomes (hub, b0, a1), wing B is
        // untouched (hub, b0, b1). Neither degenerates.
        assert_eq!(mesh.totface(), 2, "both wings survive");
        mesh.validate_topology();
    }

    /// §5.3 — welding so two faces' rebuilt rings coincide keeps the
    /// pre-existing face and drops the rebuilt duplicate.
    #[test]
    fn weld_verts_duplicate_face_adopted() {
        // Two triangles sharing edge {0,1}: face A = (0,1,2), face B = (0,1,3),
        // with vert 2 placed so welding 2 -> 3 rebuilds A's ring as (0,1,3),
        // which coincides with B.
        let mut mesh = LoopMesh::new();
        let v0 = mesh.create_vertex(Vec3::new(0.0, 0.0, 0.0));
        let v1 = mesh.create_vertex(Vec3::new(1.0, 0.0, 0.0));
        let v2 = mesh.create_vertex(Vec3::new(0.5, 1.0, 0.0));
        let v3 = mesh.create_vertex(Vec3::new(0.5, -1.0, 0.0));

        let mk = |mesh: &mut LoopMesh, vs: [VertexKey; 3]| {
            let es: Vec<EdgeKey> = (0..3)
                .map(|i| mesh.create_edge([vs[i], vs[(i + 1) % 3]], EdgeCreateFlags::NO_DUPLICATE))
                .collect();
            mesh.face_create(&vs, &es, FaceCreateFlags::NONE)
        };
        mk(&mut mesh, [v0, v1, v2]);
        mk(&mut mesh, [v0, v1, v3]);
        assert_eq!(mesh.totface(), 2);

        mesh.weld_verts(&[(v2, v3)], false);

        assert_eq!(mesh.totface(), 1, "the rebuilt duplicate is dropped");
        let survivor = mesh.face_keys()[0];
        let ring: std::collections::HashSet<VertexKey> =
            mesh.face_vert_keys(survivor).into_iter().collect();
        assert_eq!(ring, [v0, v1, v3].into_iter().collect(), "the surviving face is (0,1,3)");
        let live: std::collections::HashSet<VertexKey> = mesh.vert_keys().into_iter().collect();
        assert!(!live.contains(&v2), "welded vert gone");
        mesh.validate_topology();
    }

    /// §7 wire-edge case — a wire edge between a source and its own target
    /// collapses to zero length and is removed, leaving the single survivor.
    #[test]
    fn weld_verts_wire_edge_endpoint() {
        let mut mesh = LoopMesh::new();
        let v0 = mesh.create_vertex(Vec3::new(0.0, 0.0, 0.0));
        let v1 = mesh.create_vertex(Vec3::new(1.0, 0.0, 0.0));
        let _wire = mesh.create_edge([v0, v1], EdgeCreateFlags::NO_DUPLICATE);

        mesh.weld_verts(&[(v0, v1)], false);

        assert_eq!(mesh.totvert(), 1, "the collapsed wire leaves one vert");
        assert_eq!(mesh.totedge(), 0, "the collapsed wire edge is removed");
        let live: std::collections::HashSet<VertexKey> = mesh.vert_keys().into_iter().collect();
        assert!(live.contains(&v1), "the target survives");
        assert!(!live.contains(&v0), "the source is gone");
        mesh.validate_topology();
    }

    /// §5.1.1 / §7 worked example — a chained map `{0 → 1, 1 → 2}` on the
    /// pentagon. The edge pass re-visits the edges it creates and chain-flattens
    /// the perimeter edge `{4,0}` forward: `0→1` mints `{4,1}`, the re-visit
    /// sees `1` is itself a source and mints `{4,2}`, which terminates. The face
    /// rebuild stays direct-once, so its `(1,2,3,4)` ring leans on still-marked
    /// vert `1` and edges `{4,1}`/`{1,2}`, and the §6 cascade reaps it. Net:
    /// F = 0 surviving faces; 3 surviving edges including the chain-flattened
    /// `{4,2}`; 3 surviving verts `{2,3,4}`.
    #[test]
    fn weld_verts_chained_map_chain_flattens_edges_but_reaps_face() {
        let mut mesh = LoopMesh::new();
        let (verts, _face) = polygon(
            &mut mesh,
            &[(0.0, 0.0), (2.0, 0.0), (2.5, 1.5), (1.0, 2.5), (-0.5, 1.5)],
        );
        let [v0, v1, v2, v3, v4] = [verts[0], verts[1], verts[2], verts[3], verts[4]];

        // Chained map: 1 is both the value of 0's entry and the key of its own.
        mesh.weld_verts(&[(v0, v1), (v1, v2)], false);

        assert_eq!(mesh.totface(), 0, "the rebuilt face leans on marked geometry and is reaped");

        let live: std::collections::HashSet<VertexKey> = mesh.vert_keys().into_iter().collect();
        assert_eq!(mesh.totvert(), 3, "verts 0 and 1 are both removed");
        assert!(!live.contains(&v0) && !live.contains(&v1), "both chain sources gone");
        assert!(
            live.contains(&v2) && live.contains(&v3) && live.contains(&v4),
            "the non-source verts survive"
        );

        assert_eq!(mesh.totedge(), 3, "three surviving edges incl. the chain-flatten");
        let norm = |a: VertexKey, b: VertexKey| {
            if a.as_usize() <= b.as_usize() { [a, b] } else { [b, a] }
        };
        let surviving: std::collections::HashSet<[VertexKey; 2]> = mesh
            .edge_keys()
            .into_iter()
            .map(|e| {
                let [a, b] = mesh.edge_vert_keys(e);
                norm(a, b)
            })
            .collect();
        assert!(surviving.contains(&norm(v2, v3)), "ring edge {{2,3}} survives");
        assert!(surviving.contains(&norm(v3, v4)), "ring edge {{3,4}} survives");
        assert!(
            surviving.contains(&norm(v4, v2)),
            "the chain-flattened edge {{4,2}} survives — {{4,0}} re-pointed 0->1 then 1->2"
        );
        mesh.validate_topology();
    }

    /// §7 contrast — the *flattened* single-entry map `{0 → 1}` on the same
    /// pentagon. Vert `1` is now a value only, so the edge pass's re-visit of
    /// the created `{4,1}` finds `1` is not a source and stops: `{4,1}` is the
    /// terminal survivor. The rebuilt `(1,2,3,4)` ring now leans on no marked
    /// geometry, so the face survives (F = 1) with the closed ring.
    #[test]
    fn weld_verts_single_entry_map_keeps_closed_ring_face() {
        let mut mesh = LoopMesh::new();
        let (verts, _face) = polygon(
            &mut mesh,
            &[(0.0, 0.0), (2.0, 0.0), (2.5, 1.5), (1.0, 2.5), (-0.5, 1.5)],
        );
        let [v0, v1, v2, v3, v4] = [verts[0], verts[1], verts[2], verts[3], verts[4]];

        mesh.weld_verts(&[(v0, v1)], false);

        assert_eq!(mesh.totface(), 1, "the single-entry weld keeps a closed-ring face");
        let survivor = mesh.face_keys()[0];
        assert_eq!(mesh.face_len(survivor), 4, "the survivor is the (1,2,3,4) quad");
        let ring: std::collections::HashSet<VertexKey> =
            mesh.face_vert_keys(survivor).into_iter().collect();
        assert_eq!(
            ring,
            [v1, v2, v3, v4].into_iter().collect(),
            "surviving ring is (1,2,3,4)"
        );

        assert_eq!(mesh.totvert(), 4, "only vert 0 is removed");
        let live: std::collections::HashSet<VertexKey> = mesh.vert_keys().into_iter().collect();
        assert!(!live.contains(&v0), "the sole source vert is gone");
        assert_eq!(mesh.totedge(), 4, "the full ring of four edges survives");
        mesh.validate_topology();
    }

    /// §7 wire-edge re-point — a wire edge `{a, source}` is re-pointed to
    /// `{a, target}` (here created fresh, since no such edge existed).
    #[test]
    fn weld_verts_wire_edge_repointed() {
        let mut mesh = LoopMesh::new();
        let a = mesh.create_vertex(Vec3::new(0.0, 0.0, 0.0));
        let source = mesh.create_vertex(Vec3::new(1.0, 0.0, 0.0));
        let target = mesh.create_vertex(Vec3::new(2.0, 0.0, 0.0));
        let _wire = mesh.create_edge([a, source], EdgeCreateFlags::NO_DUPLICATE);

        mesh.weld_verts(&[(source, target)], false);

        assert_eq!(mesh.totvert(), 2, "source removed");
        assert_eq!(mesh.totedge(), 1, "the wire re-points onto the (a, target) edge");
        assert!(
            mesh.query_edge_exists(a, target).is_some(),
            "the re-pointed edge {{a, target}} exists"
        );
        let live: std::collections::HashSet<VertexKey> = mesh.vert_keys().into_iter().collect();
        assert!(!live.contains(&source));
        mesh.validate_topology();
    }
}
