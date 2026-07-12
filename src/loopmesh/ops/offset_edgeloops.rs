use crate::loopmesh::*;

/// The new geometry [`LoopMesh::offset_edgeloops`] inserted, gathered as a set
/// ([`OFFSET_EDGELOOPS.md`](../../../../radial-mesh-guides/operators/OFFSET_EDGELOOPS.md)
/// §1 / §3.5). Enumeration order is not observable for parity (treat as a set);
/// these are the rail-segment edges (and, under `use_cap_endpoint`, the cap
/// edges) — the inserted parallel-loop edges. The new rail verts are reachable
/// from these edges, so they are not reported separately (§3.5).
#[derive(Default, Clone, Debug, PartialEq, Eq)]
pub struct OffsetEdgeLoopsResult {
    /// `edges.out` (§1): the new edges forming the inserted parallel rails.
    pub edges: Vec<EdgeKey>,
}

// Transient op-flag bits.
const BIT_TAGGED_EDGE: u8 = 0; // a selected (rail-flanked) edge
const BIT_TAGGED_FACE: u8 = 0; // a face a rail will cross (re-split candidate)
const BIT_NEW_VERT: u8 = 0; // a vert created by this call (the §3 index-`-1` marker)

impl LoopMesh {
    /// Lay one parallel edge loop down on each flanked side of `edges`,
    /// **changing only topology**
    /// ([`OFFSET_EDGELOOPS.md`](../../../../radial-mesh-guides/operators/OFFSET_EDGELOOPS.md)).
    ///
    /// The operator splits the **transverse** edges that cross the selection —
    /// each split drops a new vert coincident with its junction vert at
    /// `t = 1.0` (no displacement, §3.3) — then re-splits the touched
    /// non-triangle faces so each flanking rail becomes a real edge loop (§3.4).
    /// `use_cap_endpoint` decides what happens at the **open ends** of the
    /// selection: `true` carries the rails around the endpoint with a short cap
    /// edge; `false` (the default) leaves the ends open, dissolving the dangling
    /// 2-valence cap vert (§5).
    ///
    /// A single invocation inserts exactly one rail per flanked side — there is
    /// no cut-count and no offset-distance slot (§1); the new loops sit
    /// coincident with the selection until a follow-up edge-slide separates
    /// them. Returns the new rail (and cap) edges as a set (§3.5).
    ///
    /// Degenerate inputs return the mesh unchanged with an empty result: an
    /// empty `edges` set, or a selection with no transverse edges (no mixed
    /// vert survives the §3.2 filter — e.g. a closed loop bounding nothing or an
    /// isolated wire run), see §6.
    pub fn offset_edgeloops(
        &mut self,
        edges: &[EdgeKey],
        use_cap_endpoint: bool,
    ) -> OffsetEdgeLoopsResult {
        let mut result = OffsetEdgeLoopsResult::default();
        if edges.is_empty() {
            return result;
        }

        let tagged_edges = self.push_edge_op_flag_layer();
        let tagged_faces = self.push_face_op_flag_layer();
        let new_verts = self.push_vert_op_flag_layer();

        // ---- Phase 1 (§3.1): tag the selection, collect candidate verts. ----
        let mut candidates: Vec<VertexKey> = Vec::new();
        for &edge in edges {
            self.edge_op_flag_set(tagged_edges, edge, BIT_TAGGED_EDGE);
        }
        for &edge in edges {
            for v in self.edge_vert_keys(edge) {
                if !candidates.contains(&v) {
                    candidates.push(v);
                }
            }
        }

        // ---- Phase 2 (§3.2): filter to mixed (junction) verts. ----
        // A mixed vert touches *both* a tagged and an untagged edge.
        let mut mixed: Vec<VertexKey> = Vec::new();
        for &v in &candidates {
            let mut sees_tagged = false;
            let mut sees_untagged = false;
            for e in self.vertex_disk_iter(v) {
                if self.edge_op_flag_test(tagged_edges, e, BIT_TAGGED_EDGE) {
                    sees_tagged = true;
                } else {
                    sees_untagged = true;
                }
            }
            if sees_tagged && sees_untagged {
                mixed.push(v);
            }
        }
        if mixed.is_empty() {
            // §3.2 / §6 early-out: nothing to flank.
            self.pop_vert_op_flag_layer(new_verts);
            self.pop_face_op_flag_layer(tagged_faces);
            self.pop_edge_op_flag_layer(tagged_edges);
            return result;
        }

        // ---- Phase 3 (§3.3): split the transverse edges. ----
        // For each mixed vert, split every untagged incident edge at t = 1.0
        // from the mixed vert (coincident), tagging the faces each crosses.
        // The tagged-edge count distinguishes through (>=2) from endpoint (==1)
        // junctions; endpoint junctions drive the §5 cap logic.
        let mut endpoint_junction: Vec<bool> = Vec::with_capacity(mixed.len());
        for &v in &mixed {
            // Snapshot the disk: SEMV mutates v's disk as it runs.
            let incident: Vec<EdgeKey> = self.vertex_disk_iter(v).collect();
            let mut tagged_count = 0u32;
            for edge in incident {
                if self.edge_op_flag_test(tagged_edges, edge, BIT_TAGGED_EDGE) {
                    tagged_count += 1;
                    continue;
                }
                // Untagged (transverse) edge: tag its faces, then split it.
                for f in self.edge_face_keys(edge) {
                    self.face_op_flag_set(tagged_faces, f, BIT_TAGGED_FACE);
                }
                // §3.3 / §7: split at `t = 1.0` from the far endpoint — i.e.
                // coincident with the mixed/selection vert `v`. In `edge_split`
                // terms `v` is the near vertex and `fac = 0.0` places the new
                // vert on top of it; the same `fac` drives the §7 degenerate
                // two-source CD blend, so the rail vert inherits `v`'s per-vert
                // customdata and the new corner loops inherit `v`'s loop CD.
                let (v_new, _e_new) = self.edge_split(edge, v, 0.0);
                self.vert_op_flag_set(new_verts, v_new, BIT_NEW_VERT);
            }
            endpoint_junction.push(tagged_count == 1);
        }

        // ---- Phase 4 (§3.4): re-split the touched faces into rails. ----
        for (idx, &v) in mixed.iter().enumerate() {
            let is_endpoint = endpoint_junction[idx];

            // Collect the loops sitting at v in each tagged, non-triangle face
            // before splitting (each split mutates the face loop cycle).
            let loops_at_v: Vec<LoopKey> = self.offset_loops_at_vertex(v);
            for l in loops_at_v {
                let face = self.loops.get(l).face;
                if !self.face_op_flag_test(tagged_faces, face, BIT_TAGGED_FACE) {
                    continue;
                }
                if self.faces.get(face).len == 3 {
                    // §3.4 triangle skip — the transverse split already placed
                    // the rail edge.
                    continue;
                }
                self.resplit_face_at_loop(
                    l,
                    face,
                    is_endpoint,
                    use_cap_endpoint,
                    new_verts,
                    tagged_edges,
                    tagged_faces,
                    &mut result,
                );
            }
        }

        // ---- Phase 5 (§5): open-end cleanup under use_cap_endpoint = false. ----
        // Gather-then-kill: dissolve every dangling 2-valence new cap vert that
        // sits off an endpoint junction, after the fan scan (never mid-iter).
        if !use_cap_endpoint {
            let mut cap_verts: Vec<(VertexKey, EdgeKey)> = Vec::new();
            for (idx, &v) in mixed.iter().enumerate() {
                if !endpoint_junction[idx] {
                    continue;
                }
                for e in self.vertex_disk_iter(v) {
                    if let Some(far) = self.edges.get(e).other_vertex(v) {
                        if self.vert_op_flag_test(new_verts, far, BIT_NEW_VERT)
                            && self.vertex_disk_count_edges_capped(far, 3) == 2
                            && !cap_verts.iter().any(|&(cv, _)| cv == far)
                        {
                            cap_verts.push((far, e));
                        }
                    }
                }
            }
            for (cap_vert, e_into) in cap_verts {
                self.euler_join_edge_kill_vert(e_into, cap_vert);
            }
        }

        self.pop_vert_op_flag_layer(new_verts);
        self.pop_face_op_flag_layer(tagged_faces);
        self.pop_edge_op_flag_layer(tagged_edges);
        result
    }

    /// Every face-corner loop anchored at `v` — one per incident face. Walks
    /// the disk and each spoke's radial fan, keeping the loop whose own vertex
    /// is `v`, deduplicated by face.
    fn offset_loops_at_vertex(&self, v: VertexKey) -> Vec<LoopKey> {
        let mut out: Vec<LoopKey> = Vec::new();
        let mut seen_faces: Vec<FaceKey> = Vec::new();
        for edge in self.vertex_disk_iter(v).collect::<Vec<_>>() {
            for l in self.edge_radial_iter(edge).collect::<Vec<_>>() {
                if self.loops.get(l).vertex != v {
                    continue;
                }
                let face = self.loops.get(l).face;
                if seen_faces.contains(&face) {
                    continue;
                }
                seen_faces.push(face);
                out.push(l);
            }
        }
        out
    }

    /// §3.4 dispatch for one (mixed-vert loop `l`, tagged face `f`) pair. `l`
    /// sits at the mixed vert. Splits `f` so the new transverse verts flanking
    /// the selection become joined by a real rail edge.
    #[allow(clippy::too_many_arguments)]
    fn resplit_face_at_loop(
        &mut self,
        l: LoopKey,
        face: FaceKey,
        is_endpoint: bool,
        use_cap_endpoint: bool,
        new_verts: crate::op_flags::VertOpFlagLayer,
        tagged_edges: crate::op_flags::EdgeOpFlagLayer,
        tagged_faces: crate::op_flags::FaceOpFlagLayer,
        result: &mut OffsetEdgeLoopsResult,
    ) {
        // Re-check the tag: an earlier mixed vert's pass may have already split
        // (and untagged) this face. Each face is re-split exactly once.
        if !self.face_op_flag_test(tagged_faces, face, BIT_TAGGED_FACE) {
            return;
        }

        let l_next = self.loops.get(l).face_loop_links.next;
        let l_prev = self.loops.get(l).face_loop_links.prev;
        let next_v = self.loops.get(l_next).vertex;
        let prev_v = self.loops.get(l_prev).vertex;
        let next_new = self.vert_op_flag_test(new_verts, next_v, BIT_NEW_VERT);
        let prev_new = self.vert_op_flag_test(new_verts, prev_v, BIT_NEW_VERT);

        if next_new && prev_new {
            // Case A (§3.4): both of `l.v`'s flanking corners are new — `l.v`'s
            // two transverse edges frame this face (the endpoint cap face). The
            // rail runs straight across from l.prev to l.next, slicing l.v off.
            // At an endpoint junction this *is* the cap split; suppress it under
            // `use_cap_endpoint = false` (§5).
            if !is_endpoint || use_cap_endpoint {
                if let Some((_nf, new_edge)) = self.face_split(face, prev_v, next_v) {
                    result.edges.push(new_edge);
                }
            }
            self.face_op_flag_clear(tagged_faces, face, BIT_TAGGED_FACE);
            return;
        }

        // Case B (§3.4): the selection runs along one whole side of this face, so
        // the rail cannot be a single straight Case-A rung — it has to thread the
        // *whole* run side, dropping a coincident rail vert at **every** run vert
        // it flanks (interior through-junctions included, NOT stepped over). The
        // two ends of the rail are the transverse-split new verts at the run's
        // endpoints (each at `t = 1.0`); the interior rail verts are minted by the
        // face split itself — it lays one new rail edge between the two end verts
        // and subdivides it once per interior run vert, placing a coincident vert
        // at each (§3.4 mechanism). Reading the rail as a single end-to-end chord
        // (zero interior verts) under-produces by one vert and one segment per
        // interior junction (§4) — the parity bug guarded here.
        //
        // Walk the face rim from the near new vert (anchor on a transverse edge),
        // away from the run, collecting the *position* of each intermediate run
        // vert until the far new vert (the back-stop) closes the chain. A single
        // walk suffices: the opposite junction's pass covers the other rim
        // direction (§3.4 note).
        //
        // Orientation: enter on whichever flanking corner is the new one. `l.next`
        // new ⇒ walk backward over `l.v` and the run arc; `l.prev` new ⇒ walk
        // forward.
        let (anchor_loop, walk_back) = if next_new {
            (l_next, true)
        } else if prev_new {
            (l_prev, false)
        } else {
            self.face_op_flag_clear(tagged_faces, face, BIT_TAGGED_FACE);
            return;
        };
        let anchor_v = self.loops.get(anchor_loop).vertex;

        // Walk the rim from the anchor new vert until the next new vert (the
        // back-stop = the far transverse-split vert that terminates the rail),
        // recording every intermediate rim vert. The two rail-edge ends are the
        // two transverse-split new verts; the interior coincident rail verts are
        // the **pure through-junction** run verts between them — a run vert whose
        // own transverse split landed on the *opposite* rail, so it carries no
        // new vert on this face's rim. A run vert that is itself adjacent to a
        // new vert on this rim is a mixed-vert endpoint whose rail vert is that
        // adjacent transverse split (a §3.4 Case-A rung end), not a through-
        // junction to thread. Concretely: thread a run vert iff *both* of its rim
        // neighbours (anchor/back-stop included) are non-new.
        let mut rim_run: Vec<VertexKey> = Vec::new();
        let mut backstop: Option<VertexKey> = None;
        let mut cursor = anchor_loop;
        loop {
            cursor = if walk_back {
                self.loops.get(cursor).face_loop_links.prev
            } else {
                self.loops.get(cursor).face_loop_links.next
            };
            let cv = self.loops.get(cursor).vertex;
            if cv == anchor_v {
                break; // wrapped the whole rim without a second new vert
            }
            if self.vert_op_flag_test(new_verts, cv, BIT_NEW_VERT) {
                backstop = Some(cv); // far transverse vert — rail terminus
                break;
            } else if self.vert_on_tagged_edge(cv, tagged_edges) {
                rim_run.push(cv); // a run vert; threading decided below
            } else {
                break; // plain boundary vert — chain ends without a far new vert
            }
        }

        // A run vert is a through-junction to thread iff neither rim neighbour is
        // new. The endpoints flanking the run sequence are the anchor and the
        // back-stop (both new), so the first and last run vert in `rim_run` each
        // touch a new vert and are skipped; only the strictly-interior run verts
        // are threaded.
        let mut interior_positions: Vec<glam::Vec3> = Vec::new();
        if rim_run.len() >= 3 {
            for &cv in &rim_run[1..rim_run.len() - 1] {
                interior_positions.push(self.vertex_position(cv));
            }
        }

        if let Some(backstop) = backstop {
            // One face split lays the whole rail edge; `face_split_n` subdivides
            // it once per interior run vert, minting a coincident rail vert at
            // each. The face being re-split is known (`face`) — thread it
            // explicitly so the split never has to re-derive a "common face"
            // (which, after the first segment cut, could pick the wrong
            // sub-face).
            if let Some((_nf, _e, segments)) =
                self.face_split_n_in(face, anchor_v, backstop, &interior_positions)
            {
                for seg in segments {
                    result.edges.push(seg);
                }
            }
        }
        self.face_op_flag_clear(tagged_faces, face, BIT_TAGGED_FACE);
    }

    /// True iff `v` is incident to any tagged (selection) edge.
    fn vert_on_tagged_edge(
        &self,
        v: VertexKey,
        tagged_edges: crate::op_flags::EdgeOpFlagLayer,
    ) -> bool {
        self.vertex_disk_iter(v)
            .any(|e| self.edge_op_flag_test(tagged_edges, e, BIT_TAGGED_EDGE))
    }

    /// Split the *known* `face` along `v1`–`v2`, subdividing the new rail edge
    /// once per `interior_positions` entry (coincident rail verts at each). The
    /// caller already holds the face being re-split, so no common-face scan is
    /// needed. Returns `(new_face, new_edge, rail_segments)` where
    /// `rail_segments` are every edge of the resulting chain (the rail edge plus
    /// each sub-segment minted by the subdivision), or `None` if the split could
    /// not run.
    fn face_split_n_in(
        &mut self,
        face: FaceKey,
        v1: VertexKey,
        v2: VertexKey,
        interior_positions: &[glam::Vec3],
    ) -> Option<(FaceKey, EdgeKey, Vec<EdgeKey>)> {
        let (new_face, new_edge, mids) = self.face_split_n(face, v1, v2, interior_positions)?;

        // Gather every segment of the chain so each is reported as a rail edge.
        // The subdivision verts (`mids`) split the original `new_edge` into
        // `mids.len() + 1` collinear segments; collect them from the chain's
        // disk fans. With no interior verts this is just `new_edge`.
        let mut segments: Vec<EdgeKey> = vec![new_edge];
        for &mid in &mids {
            for e in self.vertex_disk_iter(mid).collect::<Vec<_>>() {
                if !segments.contains(&e) {
                    segments.push(e);
                }
            }
        }
        Some((new_face, new_edge, segments))
    }
}

#[cfg(test)]
mod tests {
    use glam::Vec3;

    use super::*;
    use crate::loopmesh::test_support::create_face_from_verts;

    fn counts(mesh: &LoopMesh) -> (usize, usize, usize) {
        (mesh.vertex_count(), mesh.edge_count(), mesh.face_count())
    }

    /// Build an `rows`×`cols` grid of quad faces over a unit spacing. Returns
    /// (mesh, verts) where `verts[r][c]` is the vertex at row r, col c.
    fn grid(rows: usize, cols: usize) -> (LoopMesh, Vec<Vec<VertexKey>>) {
        let mut mesh = LoopMesh::new();
        let mut verts: Vec<Vec<VertexKey>> = Vec::with_capacity(rows + 1);
        for r in 0..=rows {
            let mut row = Vec::with_capacity(cols + 1);
            for c in 0..=cols {
                row.push(mesh.create_vertex(Vec3::new(c as f32, (rows - r) as f32, 0.0)));
            }
            verts.push(row);
        }
        for r in 0..rows {
            for c in 0..cols {
                let a = verts[r][c];
                let b = verts[r][c + 1];
                let d = verts[r + 1][c + 1];
                let e = verts[r + 1][c];
                let _ = create_face_from_verts_keys(&mut mesh, &[a, b, d, e]);
            }
        }
        (mesh, verts)
    }

    fn create_face_from_verts_keys(mesh: &mut LoopMesh, vs: &[VertexKey]) -> FaceKey {
        let n = vs.len();
        let mut edges = Vec::with_capacity(n);
        for i in 0..n {
            let a = vs[i];
            let b = vs[(i + 1) % n];
            let e = mesh
                .query_edge_exists(a, b)
                .unwrap_or_else(|| mesh.create_edge([a, b], EdgeCreateFlags::NONE));
            edges.push(e);
        }
        mesh.face_create(vs, &edges, FaceCreateFlags::NONE)
    }

    #[test]
    fn grid_interior_row_lays_two_rails() {
        // §2 / §3 headline: select the interior horizontal run of a 3x3 grid;
        // two rails appear (one above, one below). Each crossed quad splits
        // once; each transverse edge gains a coincident vert.
        let (mut mesh, v) = grid(3, 3);
        // Interior horizontal run: between rows 1 and 2, the three horizontal
        // edges {v[2][0]..v[2][3]} (row index 2 is the lower interior row).
        let run = [
            mesh.query_edge_exists(v[2][0], v[2][1]).unwrap(),
            mesh.query_edge_exists(v[2][1], v[2][2]).unwrap(),
            mesh.query_edge_exists(v[2][2], v[2][3]).unwrap(),
        ];

        let res = mesh.offset_edgeloops(&run, false);

        // The 3-edge run (verts v[2][0..3]) is flanked on both sides. Each run
        // vert's vertical transverse edge above and below is split → 8 new
        // coincident verts; the 6 crossed quads each split once → +6 faces;
        // 8 transverse-half edges + 6 rail edges → +14 edges.
        assert_eq!(counts(&mesh), (24, 38, 15), "V24 E38 F15");
        assert_eq!(res.edges.len(), 6, "two 3-segment rails");

        // Every reported rail edge is a real distinct-endpoint edge.
        for &e in &res.edges {
            let [a, b] = mesh.edge_vert_keys(e);
            assert_ne!(a, b);
        }

        // Each rail vert sits coincident with the run vert it flanks: v[2][1]
        // (an interior through junction) is matched by one rail vert above and
        // one below.
        let p = mesh.vertex_position(v[2][1]);
        let coincident = mesh
            .vert_keys()
            .into_iter()
            .filter(|&w| w != v[2][1] && mesh.vertex_position(w).distance(p) < 1e-6)
            .count();
        assert_eq!(coincident, 2, "rail verts coincident above and below");

        // Rail continuity: the 6 rail edges form two connected 3-edge paths.
        // Group rail edges by connectivity; each rail vert should have degree
        // 1 or 2 within the rail subgraph (a simple path, no branching).
        let mut rail_deg: std::collections::HashMap<VertexKey, u32> =
            std::collections::HashMap::new();
        for &e in &res.edges {
            for w in mesh.edge_vert_keys(e) {
                *rail_deg.entry(w).or_insert(0) += 1;
            }
        }
        assert!(
            rail_deg.values().all(|&d| d <= 2),
            "rails are simple paths (no branching vert)"
        );
    }

    #[test]
    fn cap_true_caps_open_end() {
        // §5: an open-ended single-edge selection. With cap = true the rails
        // are carried around the endpoint with a cap edge, so the endpoint's
        // Case-A split runs (an extra face vs cap = false).
        let (mut mesh, v) = grid(3, 3);
        // Open-ended interior run that ENDS inside the grid: one edge from an
        // interior vert v[2][1] to the boundary vert v[2][0]. v[2][0] is a
        // boundary vert -> its only tagged edge is this one (endpoint junction);
        // v[2][1] is interior (through-ish) — sees other untagged edges.
        let e = mesh.query_edge_exists(v[2][1], v[2][0]).unwrap();

        let res_cap = mesh.offset_edgeloops(&[e], true);
        let faces_cap = mesh.face_count();
        assert!(!res_cap.edges.is_empty());

        // Rebuild and run with cap = false for the contrast.
        let (mut mesh2, v2) = grid(3, 3);
        let e2 = mesh2.query_edge_exists(v2[2][1], v2[2][0]).unwrap();
        let res_open = mesh2.offset_edgeloops(&[e2], false);
        let faces_open = mesh2.face_count();
        assert!(!res_open.edges.is_empty());

        // The cap split at the endpoint adds a face that the open variant
        // suppresses, so cap = true has at least as many faces as cap = false,
        // and strictly more rail/cap edges around the endpoint.
        assert!(
            faces_cap >= faces_open,
            "cap=true keeps the endpoint cap face ({faces_cap} vs {faces_open})"
        );
        assert!(
            res_cap.edges.len() > res_open.edges.len(),
            "cap=true emits the extra cap edge ({} vs {})",
            res_cap.edges.len(),
            res_open.edges.len()
        );
    }

    #[test]
    fn cap_false_dissolves_cap_vert() {
        // §5 contrast: cap = false suppresses the endpoint split and dissolves
        // the dangling 2-valence new cap vert. The endpoint boundary vert keeps
        // a low valence (no cap edge attached around it).
        let (mut mesh, v) = grid(3, 3);
        let endpoint = v[2][0];
        let e = mesh.query_edge_exists(v[2][1], endpoint).unwrap();
        let val_before = mesh.vertex_disk_count_edges(endpoint);

        mesh.offset_edgeloops(&[e], false);

        // With the cap suppressed and cap verts dissolved, the endpoint vert's
        // valence is unchanged (no cap edge wraps around it).
        let val_after = mesh.vertex_disk_count_edges(endpoint);
        assert_eq!(
            val_before, val_after,
            "no cap edge added at the open endpoint"
        );
    }

    #[test]
    fn empty_selection_is_noop() {
        let (mut mesh, _v) = grid(3, 3);
        let before = counts(&mesh);
        let res = mesh.offset_edgeloops(&[], false);
        assert_eq!(counts(&mesh), before, "empty: mesh unchanged");
        assert!(res.edges.is_empty());
    }

    #[test]
    fn isolated_edge_no_transverse_is_noop() {
        // §6: a lone wire edge whose endpoints are degree-1 has no transverse
        // edge -> no mixed vert survives -> no-op.
        let mut mesh = LoopMesh::new();
        let a = mesh.create_vertex(Vec3::new(0.0, 0.0, 0.0));
        let b = mesh.create_vertex(Vec3::new(1.0, 0.0, 0.0));
        let wire = mesh.create_edge([a, b], EdgeCreateFlags::NONE);
        let before = counts(&mesh);
        let res = mesh.offset_edgeloops(&[wire], false);
        assert_eq!(counts(&mesh), before, "isolated edge: no-op");
        assert!(res.edges.is_empty());
    }

    #[test]
    fn closed_loop_is_cap_insensitive() {
        // §5 / §8: a closed selected loop with faces on both sides has no
        // endpoint junction, so cap = true and cap = false produce the same
        // topology. Run both on identical fixtures and compare counts.
        let res_true_counts = {
            let (mut mesh, v) = grid(5, 5);
            let loop_edges = center_loop_edges(&mesh, &v);
            mesh.offset_edgeloops(&loop_edges, true);
            counts(&mesh)
        };
        let res_false_counts = {
            let (mut mesh, v) = grid(5, 5);
            let loop_edges = center_loop_edges(&mesh, &v);
            mesh.offset_edgeloops(&loop_edges, false);
            counts(&mesh)
        };
        assert_eq!(
            res_true_counts, res_false_counts,
            "closed loop: cap flag has no effect"
        );
        // Inner + outer rail around the 4-edge loop: +8V +16E +8F, 8 rail edges.
        assert_eq!(res_true_counts, (44, 76, 33), "closed-loop offset counts");
    }

    #[test]
    fn rail_vert_inherits_selection_vert_cd() {
        // §7: each rail vert is dropped at t = 1.0 from its host edge's far
        // endpoint — coincident with the selection (mixed) vert — so it inherits
        // that vert's per-vert CD (the degenerate two-source blend). Decorate
        // every vert with a distinct scalar, offset, and assert every coincident
        // rail vert reads its selection vert's value (not default 0.0).
        let (mut mesh, v) = grid(3, 3);
        let cd = mesh
            .register_vertex_attribute(ElementAttributeDesc::numeric(
                AttributeKey::custom("vcd"),
                AttrType::F32,
            ))
            .unwrap();
        // Give every existing vert a unique nonzero value keyed off position so
        // a coincident rail vert can be matched back to its source.
        let value_of = |p: Vec3| (p.x + 1.0) * 100.0 + (p.y + 1.0);
        for &vk in &mesh.vert_keys() {
            let val = value_of(mesh.vertex_position(vk));
            mesh.vertex_attribs_mut().set_f32(cd, vk.as_usize(), val);
        }
        // Selection: the interior horizontal run (verts v[2][0..3]).
        let run = [
            mesh.query_edge_exists(v[2][0], v[2][1]).unwrap(),
            mesh.query_edge_exists(v[2][1], v[2][2]).unwrap(),
            mesh.query_edge_exists(v[2][2], v[2][3]).unwrap(),
        ];
        let pre: std::collections::HashSet<VertexKey> = mesh.vert_keys().into_iter().collect();

        mesh.offset_edgeloops(&run, false);

        // Every freshly-minted rail vert must read the CD of the existing vert it
        // landed coincident with — never default 0.0.
        let mut checked = 0;
        for &rv in &mesh.vert_keys() {
            if pre.contains(&rv) {
                continue; // existing vert, not a rail vert
            }
            let p = mesh.vertex_position(rv);
            let source = mesh
                .vert_keys()
                .into_iter()
                .find(|&w| pre.contains(&w) && mesh.vertex_position(w).distance(p) < 1e-6)
                .expect("rail vert is coincident with an existing selection vert");
            let expected = mesh
                .vertex_attribs()
                .get_f32(cd, source.as_usize())
                .unwrap();
            let got = mesh.vertex_attribs().get_f32(cd, rv.as_usize()).unwrap();
            assert!(expected.abs() > 0.0, "source value is nonzero");
            assert!(
                (got - expected).abs() < 1e-4,
                "rail vert inherits selection vert CD: expected {expected}, got {got}"
            );
            checked += 1;
        }
        assert!(checked > 0, "at least one rail vert exists to check");
    }

    #[test]
    fn rail_loop_inherits_selection_loop_uv() {
        // §7: loops gaining the new rail vert as a corner take per-loop CD
        // interpolated along their edge at t = 1.0 — i.e. the selection-vert
        // corner loop's values. Decorate every loop with a UV keyed to its own
        // vertex; after offset, each rail-vert corner loop must read the UV of
        // the coincident selection vert's corner (not default [0,0]).
        let (mut mesh, v) = grid(3, 3);
        let uv = mesh
            .register_loop_attribute(ElementAttributeDesc::numeric(
                AttributeKey::custom("uv"),
                AttrType::Vec2,
            ))
            .unwrap();
        // Key each loop's UV to its corner vertex's position so a rail-vert loop
        // can be matched to the selection-vert corner it inherited from.
        let uv_of = |p: Vec3| [p.x + 1.0, p.y + 1.0];
        for &l in &mesh.loop_keys() {
            let p = mesh.vertex_position(mesh.loop_vertex(l));
            mesh.loop_attribs_mut().set_vec2(uv, l.as_usize(), uv_of(p));
        }
        let run = [
            mesh.query_edge_exists(v[2][0], v[2][1]).unwrap(),
            mesh.query_edge_exists(v[2][1], v[2][2]).unwrap(),
            mesh.query_edge_exists(v[2][2], v[2][3]).unwrap(),
        ];
        let pre_verts: std::collections::HashSet<VertexKey> =
            mesh.vert_keys().into_iter().collect();
        let pre_loops: std::collections::HashSet<LoopKey> = mesh.loop_keys().into_iter().collect();

        mesh.offset_edgeloops(&run, false);

        // Any new loop anchored at a rail vert inherits the coincident selection
        // vert's keyed UV — the value the selection corner would carry.
        let mut checked = 0;
        for &l in &mesh.loop_keys() {
            if pre_loops.contains(&l) {
                continue;
            }
            let lv = mesh.loop_vertex(l);
            if pre_verts.contains(&lv) {
                continue; // new loop, but at an existing corner (a face-split
                          // loop at a selection vert) — not a rail-vert loop
            }
            let p = mesh.vertex_position(lv);
            let source = mesh
                .vert_keys()
                .into_iter()
                .find(|&w| pre_verts.contains(&w) && mesh.vertex_position(w).distance(p) < 1e-6)
                .expect("rail vert coincident with a selection vert");
            let expected = uv_of(mesh.vertex_position(source));
            let got = mesh.loop_attribs().get_vec2(uv, l.as_usize()).unwrap();
            assert!(
                (got[0] - expected[0]).abs() < 1e-4 && (got[1] - expected[1]).abs() < 1e-4,
                "rail loop inherits selection-vert corner UV: expected {expected:?}, got {got:?}"
            );
            checked += 1;
        }
        assert!(checked > 0, "at least one rail-vert loop exists to check");
    }

    /// The four edges bounding the centre face of a 5x5 grid.
    fn center_loop_edges(mesh: &LoopMesh, v: &[Vec<VertexKey>]) -> Vec<EdgeKey> {
        // Centre face spans rows 2..3, cols 2..3.
        let tl = v[2][2];
        let tr = v[2][3];
        let br = v[3][3];
        let bl = v[3][2];
        vec![
            mesh.query_edge_exists(tl, tr).unwrap(),
            mesh.query_edge_exists(tr, br).unwrap(),
            mesh.query_edge_exists(br, bl).unwrap(),
            mesh.query_edge_exists(bl, tl).unwrap(),
        ]
    }

    #[test]
    fn ngon_offset_side_case_b_walk() {
        // §3.4 Case B: a hexagon on the offset side whose top side is the whole
        // selection run. The lower rail must thread several new verts across the
        // hexagon rim with the backward walk. The upper side is a strip of
        // quads (Case A). Verify the operator runs and the hexagon is re-split
        // into more than one face with a real rail edge inside it.
        let mut mesh = LoopMesh::new();
        // Run verts a,b,c,d along y=0.
        let a = mesh.create_vertex(Vec3::new(0.0, 0.0, 0.0));
        let b = mesh.create_vertex(Vec3::new(1.0, 0.0, 0.0));
        let c = mesh.create_vertex(Vec3::new(2.0, 0.0, 0.0));
        let d = mesh.create_vertex(Vec3::new(3.0, 0.0, 0.0));
        // Quad row above (y=1).
        let a2 = mesh.create_vertex(Vec3::new(0.0, 1.0, 0.0));
        let b2 = mesh.create_vertex(Vec3::new(1.0, 1.0, 0.0));
        let c2 = mesh.create_vertex(Vec3::new(2.0, 1.0, 0.0));
        let d2 = mesh.create_vertex(Vec3::new(3.0, 1.0, 0.0));
        // Hexagon bottom verts (y=-1).
        let bl = mesh.create_vertex(Vec3::new(0.5, -1.0, 0.0));
        let br = mesh.create_vertex(Vec3::new(2.5, -1.0, 0.0));

        create_face_from_verts_keys(&mut mesh, &[a, b, b2, a2]);
        create_face_from_verts_keys(&mut mesh, &[b, c, c2, b2]);
        create_face_from_verts_keys(&mut mesh, &[c, d, d2, c2]);
        // Hexagon: a, b, c, d (top), br, bl (bottom) wound consistently.
        let hex = create_face_from_verts_keys(&mut mesh, &[a, b, c, d, br, bl]);

        let run = [
            mesh.query_edge_exists(a, b).unwrap(),
            mesh.query_edge_exists(b, c).unwrap(),
            mesh.query_edge_exists(c, d).unwrap(),
        ];
        let before = counts(&mesh);
        assert_eq!(before, (10, 13, 4), "starts 10V 13E 4F");
        let hex_neighbors_before = mesh.face_vert_keys(hex).len();
        assert_eq!(hex_neighbors_before, 6, "starts as a hexagon");

        let res = mesh.offset_edgeloops(&run, false);

        // §3.4 Case B postcondition: the hexagon's lower rail is a multi-segment
        // chain dropping a coincident rail vert at *every* run vert along the
        // run side — near-a → on-b → on-c → near-d (4 rail verts, 3 segments) —
        // not a single near-a → near-d chord that steps over the interior
        // through-junctions b and c. The hexagon is split exactly once (one face
        // added), but the rail threads all four run verts.
        //
        // Counts: the run verts a,b,c,d each gain an upper-rail transverse split
        // (4 new verts via the quad strip) and the hexagon side gains its own
        // 4-vert chain — near-a, near-d are the slant transverse splits, on-b,
        // on-c are minted by the hexagon face split's subdivision. 10V13E4F →
        // 18V25E8F: +8 verts, +12 edges, +4 faces (3 quad rungs + 1 hexagon
        // split). This matches the cross-backend `ngon_offset_side_strip`
        // fixture's expected post-offset counts.
        assert_eq!(
            counts(&mesh),
            (18, 25, 8),
            "V18 E25 F8 corrected Case-B chain"
        );

        // A coincident rail vert exists at each interior run vert b and c (the
        // through-junctions the chord misreading stepped over). Each interior run
        // vert is flanked by exactly one new coincident vert on the hexagon side.
        for (label, run_vert) in [("b", b), ("c", c)] {
            let p = mesh.vertex_position(run_vert);
            let coincident = mesh
                .vert_keys()
                .into_iter()
                .filter(|&w| w != run_vert && mesh.vertex_position(w).distance(p) < 1e-6)
                .count();
            assert!(
                coincident >= 1,
                "interior run vert {label} has a coincident rail vert (not chorded over)"
            );
        }

        // The lower rail is a connected chain visiting near-a, on-b, on-c,
        // near-d in turn: every reported rail edge has distinct endpoints, and
        // the rail subgraph contains a vert coincident with each run vert a..d.
        for &e in &res.edges {
            let [x, y] = mesh.edge_vert_keys(e);
            assert_ne!(x, y, "rail edge has distinct endpoints");
        }
        let rail_verts: std::collections::HashSet<VertexKey> = res
            .edges
            .iter()
            .flat_map(|&e| mesh.edge_vert_keys(e))
            .collect();
        for (label, run_vert) in [("a", a), ("b", b), ("c", c), ("d", d)] {
            let p = mesh.vertex_position(run_vert);
            let threaded = rail_verts
                .iter()
                .any(|&w| mesh.vertex_position(w).distance(p) < 1e-6);
            assert!(
                threaded,
                "rail threads a vert coincident with run vert {label}"
            );
        }
    }
}
