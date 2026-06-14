use crate::loopmesh::*;

/// Bit identifying an edge as a member of the operator's `edges` candidate set.
const IN_INPUT_BIT: u8 = 0;

/// Transient-flag bit shared by both the §5 short-edge marking and the §6
/// ear-clip diagonal marking: an edge carrying this bit is scheduled for the
/// next batched `collapse`.
const COLLAPSE_PENDING_BIT: u8 = 0;

/// Transient-flag bit set on a loop once the §6 walk has corner-tested it, so
/// a corner whose two incident edges are both in the input set is tested at
/// most once across the whole operator (each loop sits on two radial cycles).
const LOOP_TESTED_BIT: u8 = 0;

impl LoopMesh {
    /// Geometry-driven cleanup pass: collapse input edges whose length is below
    /// `dist`, and clip near-collinear loop "ears" whose collinearity error is
    /// within `dist`.
    ///
    /// Implements `radial-mesh-guides/operators/DISSOLVE_DEGENERATE.md`:
    ///
    /// - **§1 input-as-filter** — `edges` is a *candidate filter*, not a hint.
    ///   An edge not in the set is invisible to both passes regardless of its
    ///   geometry; an empty set is a complete no-op even on a degenerate mesh.
    /// - **§3 three phases** — §5 marks every input edge shorter than `dist`
    ///   and clears the loop-tested bit on each input edge's radial loops; the
    ///   §5.3 batch collapses the marked edges; §6 walks every surviving input
    ///   edge's radial cycle and clips near-collinear ears, flagging an
    ///   introduced joining edge (or, for a triangle, the opposite edge); the
    ///   §7 batch collapses the freshly-flagged set. The operator does **not**
    ///   iterate to a fixed point — see §10 for the caller-driven loop.
    /// - **§4 collinearity metric** — `|u_prev - u_next| * min(len_prev,
    ///   len_next)`, inclusive against `dist`.
    /// - **§5.1 strict length test** — squared length `< dist * dist` (strict).
    /// - **§6.2 per-shape ear surgery** — equal-length triangle flags the
    ///   opposite edge; equal-length n-gon splits the diagonal; asymmetric ears
    ///   equalise the incident edge lengths with an edge-split before the
    ///   face-split.
    ///
    /// `dist` is a single tolerance shared by both passes (world units).
    ///
    /// The §5.3 / §7 collapse batches delegate to [`Self::collapse`] with
    /// `uvs = true` (§5.2): the per-loop customdata blend is part of the
    /// intended cleanup semantics, so each batch's collapse sites have their
    /// has-math loop layers (UVs / colours) snapped to the island midpoint.
    /// Topology and survivor positions are unaffected by the flag.
    pub fn dissolve_degenerate(&mut self, edges: &[EdgeKey], dist: f32) {
        // §1 / §8 — an empty candidate set marks nothing and walks nothing.
        if edges.is_empty() {
            return;
        }

        let dist_sq = dist * dist;

        let input = self.push_edge_op_flag_layer();
        let pending = self.push_edge_op_flag_layer();
        let tested = self.push_loop_op_flag_layer();

        for &edge in edges {
            self.edge_op_flag_set(input, edge, IN_INPUT_BIT);
        }

        // §5.1 — mark short input edges and seed the §6 loop-tested bits.
        let mut any_short = false;
        for edge in self.edge_keys() {
            if self.edge_op_flag_test(input, edge, IN_INPUT_BIT)
                && self.edge_length_sq(edge) < dist_sq
            {
                self.edge_op_flag_set(pending, edge, COLLAPSE_PENDING_BIT);
                any_short = true;
            }
            // §5.1 — clear the loop-tested bit on every loop of the radial
            // cycle, independent of the length test.
            let mut radial = self.radial_loops(edge);
            while let Some(loop_key) = radial.next(self) {
                self.loop_op_flag_clear(tested, loop_key, LOOP_TESTED_BIT);
            }
        }

        // §5.3 — first collapse batch over the §5-flagged edges.
        if any_short {
            let flagged = self.collect_pending_edges(pending);
            self.collapse(&flagged, true);
        }

        // §6 — ear-clip walk over every surviving input edge.
        let mut any_ears = false;
        for edge in self.edge_keys() {
            if !self.edge_op_flag_test(input, edge, IN_INPUT_BIT) {
                continue;
            }
            if self.iter_radial_loops(edge).next().is_none() {
                continue;
            }
            if self.ear_clip_edge(edge, dist, input, pending, tested) {
                any_ears = true;
            }
        }

        // §7 — second collapse batch over the §6-flagged edges.
        if any_ears {
            let flagged = self.collect_pending_edges(pending);
            self.collapse(&flagged, true);
        }

        self.pop_loop_op_flag_layer(tested);
        self.pop_edge_op_flag_layer(pending);
        self.pop_edge_op_flag_layer(input);
    }

    /// §6.1 walk of a single edge's radial cycle. Returns whether any ear
    /// surgery fired (so a joining edge was flagged for the §7 batch). The walk
    /// resets its start loop after each successful surgery (§6.3) because the
    /// surgery lengthens `edge`'s radial cycle by one.
    fn ear_clip_edge(
        &mut self,
        edge: EdgeKey,
        dist: f32,
        input: crate::op_flags::EdgeOpFlagLayer,
        pending: crate::op_flags::EdgeOpFlagLayer,
        tested: crate::op_flags::LoopOpFlagLayer,
    ) -> bool {
        let mut any_ears = false;
        let Some(first) = self.iter_radial_loops(edge).next() else {
            return false;
        };
        let mut loop_first = first;
        let mut loop_iter = first;
        loop {
            if self.loop_op_flag_test(tested, loop_iter, LOOP_TESTED_BIT) {
                loop_iter = self.loop_radial_next(loop_iter);
                if loop_iter == loop_first {
                    break;
                }
                continue;
            }
            self.loop_op_flag_set(tested, loop_iter, LOOP_TESTED_BIT);

            if self.try_ear_surgery(loop_iter, dist, input, pending) {
                any_ears = true;
                // §6.3 — the surgery added a loop to `edge`'s radial cycle;
                // re-anchor the start loop so the termination check does not
                // fire on a stale start.
                loop_first = loop_iter;
            }

            loop_iter = self.loop_radial_next(loop_iter);
            if loop_iter == loop_first {
                break;
            }
        }
        any_ears
    }

    /// §6.1 guards + §6.2 surgery for one corner loop. Returns whether surgery
    /// fired.
    fn try_ear_surgery(
        &mut self,
        loop_key: LoopKey,
        dist: f32,
        input: crate::op_flags::EdgeOpFlagLayer,
        pending: crate::op_flags::EdgeOpFlagLayer,
    ) -> bool {
        let edge_outgoing = self.loop_edge(loop_key);
        let loop_prev = self.loop_prev(loop_key);
        let edge_incoming = self.loop_edge(loop_prev);

        // §6.1 — both incident edges of the corner must be candidates.
        if !self.edge_op_flag_test(input, edge_incoming, IN_INPUT_BIT) {
            return false;
        }
        // §6.1 — skip corners whose edges are already scheduled for collapse.
        if self.edge_op_flag_test(pending, edge_outgoing, COLLAPSE_PENDING_BIT)
            || self.edge_op_flag_test(pending, edge_incoming, COLLAPSE_PENDING_BIT)
        {
            return false;
        }

        let corner_vert = self.loop_vertex(loop_key);
        let prev_vert = self.loop_vertex(loop_prev);
        let next_vert = self.loop_vertex(self.loop_next(loop_key));

        let corner_pos = self.vertex_position(corner_vert);
        let prev_pos = self.vertex_position(prev_vert);
        let next_pos = self.vertex_position(next_vert);

        let direction_prev = prev_pos - corner_pos;
        let direction_next = next_pos - corner_pos;
        let len_prev = direction_prev.length();
        let len_next = direction_next.length();

        // A zero-length incident edge cannot form a meaningful corner; the §5
        // pass owns those. Guard against the divide-by-zero in normalisation.
        if len_prev == 0.0 || len_next == 0.0 {
            return false;
        }

        let unit_prev = direction_prev / len_prev;
        let unit_next = direction_next / len_next;
        let collinearity_error = (unit_prev - unit_next).length() * len_prev.min(len_next);

        // §4 / §6.1 — inclusive tolerance.
        if collinearity_error > dist {
            return false;
        }

        let face = self.loop_face(loop_key);
        self.apply_ear_surgery(
            face,
            loop_key,
            corner_vert,
            prev_vert,
            next_vert,
            dist,
            len_prev,
            len_next,
            pending,
        );
        true
    }

    /// §6.2 — the three per-shape ear surgeries.
    #[allow(clippy::too_many_arguments)]
    fn apply_ear_surgery(
        &mut self,
        face: FaceKey,
        loop_key: LoopKey,
        corner_vert: VertexKey,
        prev_vert: VertexKey,
        next_vert: VertexKey,
        dist: f32,
        len_prev: f32,
        len_next: f32,
        pending: crate::op_flags::EdgeOpFlagLayer,
    ) {
        if (len_prev - len_next).abs() <= dist {
            // §6.2.1 — equal-length ear.
            if self.face_len(face) == 3 {
                // Flag the edge opposite the corner (the loop's next edge
                // around the face), pinching the triangle closed in §7.
                let opposite = self.loop_edge(self.loop_next(loop_key));
                self.edge_op_flag_set(pending, opposite, COLLAPSE_PENDING_BIT);
            } else if let Some((_new_face, diagonal)) = self.face_split(face, prev_vert, next_vert)
            {
                self.edge_op_flag_set(pending, diagonal, COLLAPSE_PENDING_BIT);
            }
        } else if len_prev < len_next {
            // §6.2.2 — incoming edge shorter: equalise by splitting the
            // outgoing edge, then split the face to the new vert.
            let outgoing = self.loop_edge(loop_key);
            self.edge_split(outgoing, corner_vert, len_prev / len_next);
            // The split inserts the new corner vertex as the loop's next.
            let new_next_vert = self.loop_vertex(self.loop_next(loop_key));
            if let Some((_new_face, diagonal)) = self.face_split(face, prev_vert, new_next_vert) {
                self.edge_op_flag_set(pending, diagonal, COLLAPSE_PENDING_BIT);
            }
        } else {
            // §6.2.3 — outgoing edge shorter: equalise by splitting the
            // incoming edge, then split the face from the new vert.
            let incoming = self.loop_edge(self.loop_prev(loop_key));
            self.edge_split(incoming, corner_vert, len_next / len_prev);
            // The split inserts the new corner vertex as the loop's prev.
            let new_prev_vert = self.loop_vertex(self.loop_prev(loop_key));
            if let Some((_new_face, diagonal)) = self.face_split(face, new_prev_vert, next_vert) {
                self.edge_op_flag_set(pending, diagonal, COLLAPSE_PENDING_BIT);
            }
        }
    }

    /// §5 — squared length of an edge from its endpoint world positions.
    fn edge_length_sq(&self, edge: EdgeKey) -> f32 {
        let [a, b] = self.edge_vertices(edge);
        (self.vertex_position(a) - self.vertex_position(b)).length_squared()
    }

    /// Collect every live edge currently carrying the collapse-pending flag,
    /// then clear the flag so the next batch starts fresh.
    fn collect_pending_edges(&mut self, pending: crate::op_flags::EdgeOpFlagLayer) -> Vec<EdgeKey> {
        let mut flagged = Vec::new();
        for edge in self.edge_keys() {
            if self.edge_op_flag_test(pending, edge, COLLAPSE_PENDING_BIT) {
                flagged.push(edge);
            }
        }
        self.edge_op_flag_clear_all(pending);
        flagged
    }
}

#[cfg(test)]
mod tests {
    use glam::Vec3;

    use crate::loopmesh::*;

    /// A single quad face over four positions (winding order as given).
    fn quad(mesh: &mut LoopMesh, p: [Vec3; 4]) -> ([VertexKey; 4], FaceKey) {
        let v: Vec<VertexKey> = p.iter().map(|&x| mesh.create_vertex(x)).collect();
        let e: Vec<EdgeKey> = (0..4)
            .map(|i| mesh.create_edge([v[i], v[(i + 1) % 4]], EdgeCreateFlags::NO_DUPLICATE))
            .collect();
        let face = mesh.face_create(&v, &e, FaceCreateFlags::NONE);
        ([v[0], v[1], v[2], v[3]], face)
    }

    /// A single triangle over three positions.
    fn triangle(mesh: &mut LoopMesh, p: [Vec3; 3]) -> ([VertexKey; 3], FaceKey) {
        let v: Vec<VertexKey> = p.iter().map(|&x| mesh.create_vertex(x)).collect();
        let e: Vec<EdgeKey> = (0..3)
            .map(|i| mesh.create_edge([v[i], v[(i + 1) % 3]], EdgeCreateFlags::NO_DUPLICATE))
            .collect();
        let face = mesh.face_create(&v, &e, FaceCreateFlags::NONE);
        ([v[0], v[1], v[2]], face)
    }

    /// Two quads sharing edge `{b, c}` (a vertical-ish shared edge of the given
    /// `height` in y). Returns the six verts in `[a0, b, c, a1, d0, d1]` order
    /// (matching the collapse-test fixture) and the shared edge.
    fn two_quads_shared_edge_short_y(
        mesh: &mut LoopMesh,
        height: f32,
    ) -> ([VertexKey; 6], EdgeKey) {
        let a0 = mesh.create_vertex(Vec3::new(0.0, 0.0, 0.0));
        let b = mesh.create_vertex(Vec3::new(1.0, 0.0, 0.0));
        let c = mesh.create_vertex(Vec3::new(1.0, height, 0.0));
        let a1 = mesh.create_vertex(Vec3::new(0.0, 1.0, 0.0));
        let d0 = mesh.create_vertex(Vec3::new(2.0, 0.0, 0.0));
        let d1 = mesh.create_vertex(Vec3::new(2.0, 1.0, 0.0));

        let mk = |mesh: &mut LoopMesh, vs: [VertexKey; 4]| {
            let es: Vec<EdgeKey> = (0..4)
                .map(|i| mesh.create_edge([vs[i], vs[(i + 1) % 4]], EdgeCreateFlags::NO_DUPLICATE))
                .collect();
            mesh.face_create(&vs, &es, FaceCreateFlags::NONE);
        };
        mk(mesh, [a0, b, c, a1]);
        mk(mesh, [b, d0, d1, c]);

        let shared = mesh.query_edge_exists(b, c).expect("shared edge present");
        ([a0, b, c, a1, d0, d1], shared)
    }

    /// §8 — clean mesh, `dist = 0`: no short edges, no near-collinear ears, no
    /// collapse invocations. Mesh unchanged.
    #[test]
    fn clean_quad_dist_zero_is_noop() {
        let mut mesh = LoopMesh::new();
        let (_v, _f) = quad(
            &mut mesh,
            [
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(1.0, 1.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
            ],
        );
        let before = (mesh.totvert(), mesh.totedge(), mesh.totface());

        let edges = mesh.edge_keys();
        mesh.dissolve_degenerate(&edges, 0.0);

        assert_eq!((mesh.totvert(), mesh.totedge(), mesh.totface()), before);
        mesh.validate_topology();
    }

    /// §8 — same clean mesh, small positive `dist`: the tolerance does not
    /// produce false positives. Mesh unchanged.
    #[test]
    fn clean_quad_small_dist_is_noop() {
        let mut mesh = LoopMesh::new();
        quad(
            &mut mesh,
            [
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(1.0, 1.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
            ],
        );
        let before = (mesh.totvert(), mesh.totedge(), mesh.totface());

        let edges = mesh.edge_keys();
        mesh.dissolve_degenerate(&edges, 0.01);

        assert_eq!((mesh.totvert(), mesh.totedge(), mesh.totface()), before);
        mesh.validate_topology();
    }

    /// §5 headline — a single sub-tolerance shared edge collapses; both quads
    /// drop to triangles. Net: 6V/7E/2F → 5V/6E/2F.
    #[test]
    fn short_shared_edge_collapses_in_two_quad_strip() {
        let mut mesh = LoopMesh::new();
        let (_v, _shared) = two_quads_shared_edge_short_y(&mut mesh, 0.001);
        assert_eq!((mesh.totvert(), mesh.totedge(), mesh.totface()), (6, 7, 2));

        let edges = mesh.edge_keys();
        mesh.dissolve_degenerate(&edges, 0.01);

        assert_eq!(mesh.totvert(), 5, "the short shared pair fused to one vert");
        assert_eq!(mesh.totedge(), 6, "the short edge is reaped");
        assert_eq!(mesh.totface(), 2, "both quads survive as triangles");
        for face in mesh.face_keys() {
            assert_eq!(mesh.face_len(face), 3);
        }
        mesh.validate_topology();
    }

    /// §8 — a triangle whose three edges are all sub-tolerance collapses to a
    /// single vert.
    #[test]
    fn tiny_triangle_collapses_to_single_vert() {
        let mut mesh = LoopMesh::new();
        triangle(
            &mut mesh,
            [
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(0.002, 0.0, 0.0),
                Vec3::new(0.001, 0.002, 0.0),
            ],
        );

        let edges = mesh.edge_keys();
        mesh.dissolve_degenerate(&edges, 0.01);

        assert_eq!(mesh.totvert(), 1, "the tiny triangle fused to one vert");
        assert_eq!(mesh.totedge(), 0);
        assert_eq!(mesh.totface(), 0);
        mesh.validate_topology();
    }

    /// §6.2.1 triangle case — a triangle whose three verts are collinear (zero
    /// area) but whose edges are above tolerance. §5 spares every edge; §6
    /// fires on the doubled-back end corners and flags the opposite edge; §7
    /// collapses the triangle away.
    #[test]
    fn collinear_triangle_clips_ear() {
        let mut mesh = LoopMesh::new();
        triangle(
            &mut mesh,
            [
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(2.0, 0.0, 0.0),
            ],
        );
        assert_eq!(mesh.totface(), 1);

        let edges = mesh.edge_keys();
        mesh.dissolve_degenerate(&edges, 0.01);

        assert_eq!(mesh.totface(), 0, "the zero-area triangle is consumed");
        mesh.validate_topology();
    }

    /// §8 — a straight chain of three short collinear edges on a wire collapses
    /// to a single vert.
    #[test]
    fn chain_of_short_edges_collapses_to_single_vert() {
        let mut mesh = LoopMesh::new();
        let verts: Vec<VertexKey> = (0..4)
            .map(|i| mesh.create_vertex(Vec3::new(i as f32 * 0.001, 0.0, 0.0)))
            .collect();
        let edges: Vec<EdgeKey> = (0..3)
            .map(|i| mesh.create_edge([verts[i], verts[i + 1]], EdgeCreateFlags::NO_DUPLICATE))
            .collect();

        mesh.dissolve_degenerate(&edges, 0.01);

        assert_eq!(mesh.totvert(), 1, "the whole chain fused to one vert");
        assert_eq!(mesh.totedge(), 0);
        mesh.validate_topology();
    }

    /// §1 / §8 — an empty `edges` input is a complete no-op even with an obvious
    /// degeneracy present.
    #[test]
    fn empty_input_preserves_degeneracy() {
        let mut mesh = LoopMesh::new();
        two_quads_shared_edge_short_y(&mut mesh, 0.001);
        let before = (mesh.totvert(), mesh.totedge(), mesh.totface());

        mesh.dissolve_degenerate(&[], 0.01);

        assert_eq!((mesh.totvert(), mesh.totedge(), mesh.totface()), before);
        mesh.validate_topology();
    }

    /// §8 — a short wire edge (radial cycle empty) is flagged by the purely
    /// geometric §5 length test and collapsed.
    #[test]
    fn wire_edge_short_collapses() {
        let mut mesh = LoopMesh::new();
        let a = mesh.create_vertex(Vec3::new(0.0, 0.0, 0.0));
        let b = mesh.create_vertex(Vec3::new(0.001, 0.0, 0.0));
        let wire = mesh.create_edge([a, b], EdgeCreateFlags::NO_DUPLICATE);

        mesh.dissolve_degenerate(&[wire], 0.01);

        assert_eq!(mesh.totvert(), 1, "the wire endpoints fused");
        assert_eq!(mesh.totedge(), 0);
        assert_eq!(mesh.totface(), 0);
        mesh.validate_topology();
    }

    /// §5.1 boundary — the length test is strict `<` on squared length. At
    /// `height = 0.0099 < dist = 0.01` the shared edge collapses.
    #[test]
    fn dist_boundary_below_threshold_collapses() {
        let mut mesh = LoopMesh::new();
        two_quads_shared_edge_short_y(&mut mesh, 0.0099);

        let edges = mesh.edge_keys();
        mesh.dissolve_degenerate(&edges, 0.01);

        assert_eq!(mesh.totvert(), 5, "below-threshold edge collapses");
        assert_eq!(mesh.totedge(), 6);
        mesh.validate_topology();
    }

    /// §5.1 boundary — at `height = 0.0101 > dist = 0.01` the shared edge is
    /// spared (no collinear ears either, so the mesh is unchanged).
    #[test]
    fn dist_boundary_above_threshold_preserves() {
        let mut mesh = LoopMesh::new();
        two_quads_shared_edge_short_y(&mut mesh, 0.0101);
        let before = (mesh.totvert(), mesh.totedge(), mesh.totface());

        let edges = mesh.edge_keys();
        mesh.dissolve_degenerate(&edges, 0.01);

        assert_eq!((mesh.totvert(), mesh.totedge(), mesh.totface()), before);
        mesh.validate_topology();
    }

    /// §6.2.2 — incoming edge shorter than outgoing: a quad face with a
    /// doubled-back collinear corner where `len_prev < len_next`. The §6 surgery
    /// splits the outgoing edge to equalise, splits the face, and the §7 batch
    /// collapses the introduced diagonal, consuming the degenerate spur.
    #[test]
    fn asymmetric_ear_incoming_shorter() {
        // Quad winding [v0=(0.4,0), v1=(0,0), v2=(1.6,0), v3=(1,1)]. The corner
        // at v1 doubles back: both neighbours v0 and v2 sit at +x, so the
        // collinearity error is ~0, with len_prev=0.4 < len_next=1.6 → §6.2.2.
        let mut mesh = LoopMesh::new();
        quad(
            &mut mesh,
            [
                Vec3::new(0.4, 0.0, 0.0),
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.6, 0.0, 0.0),
                Vec3::new(1.0, 1.0, 0.0),
            ],
        );

        let edges = mesh.edge_keys();
        mesh.dissolve_degenerate(&edges, 0.01);

        // The doubled-back spur at v1 is clipped away; the face survives as a
        // triangle on the off-axis apex.
        assert_eq!(mesh.totface(), 1, "one face survives");
        let face = mesh.face_keys()[0];
        assert_eq!(mesh.face_len(face), 3, "the spur corner is gone");
        mesh.validate_topology();
    }

    /// §6.2.3 — outgoing edge shorter than incoming: mirror of the above with
    /// the corner's outgoing edge the short one (`len_next < len_prev`).
    #[test]
    fn asymmetric_ear_outgoing_shorter() {
        // Quad winding [v0=(1.6,0), v1=(0,0), v2=(0.4,0), v3=(1,1)]. The corner
        // at v1 doubles back with len_prev=1.6 > len_next=0.4 → §6.2.3.
        let mut mesh = LoopMesh::new();
        quad(
            &mut mesh,
            [
                Vec3::new(1.6, 0.0, 0.0),
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(0.4, 0.0, 0.0),
                Vec3::new(1.0, 1.0, 0.0),
            ],
        );

        let edges = mesh.edge_keys();
        mesh.dissolve_degenerate(&edges, 0.01);

        assert_eq!(mesh.totface(), 1, "one face survives");
        let face = mesh.face_keys()[0];
        assert_eq!(mesh.face_len(face), 3, "the spur corner is gone");
        mesh.validate_topology();
    }

    /// §6.3 mid-walk reset — two triangles sharing edge `{0,1}` with both apex
    /// verts placed collinearly on the x-axis, so both radial loops on the
    /// shared edge are clip-worthy doubled-back ears. The walk applies surgery
    /// on both loops of the same radial cycle without tripping the termination
    /// condition on a stale start loop.
    #[test]
    fn mid_walk_reset_two_collinear_ears_on_shared_edge() {
        let mut mesh = LoopMesh::new();
        let v0 = mesh.create_vertex(Vec3::new(0.0, 0.0, 0.0));
        let v1 = mesh.create_vertex(Vec3::new(1.0, 0.0, 0.0));
        let v2 = mesh.create_vertex(Vec3::new(-1.0, 0.0, 0.0));
        let v3 = mesh.create_vertex(Vec3::new(3.0, 0.0, 0.0));

        let mk_tri = |mesh: &mut LoopMesh, vs: [VertexKey; 3]| {
            let es: Vec<EdgeKey> = (0..3)
                .map(|i| mesh.create_edge([vs[i], vs[(i + 1) % 3]], EdgeCreateFlags::NO_DUPLICATE))
                .collect();
            mesh.face_create(&vs, &es, FaceCreateFlags::NONE);
        };
        mk_tri(&mut mesh, [v0, v1, v2]);
        mk_tri(&mut mesh, [v1, v0, v3]);
        assert_eq!(mesh.totface(), 2);

        let edges = mesh.edge_keys();
        mesh.dissolve_degenerate(&edges, 0.01);

        // Both zero-area triangles are consumed; the walk did not loop forever
        // or skip a loop after the radial cycle grew.
        assert_eq!(mesh.totface(), 0, "both collinear triangles consumed");
        mesh.validate_topology();
    }

    /// §5.1 + §8 — the input set scopes the §6 ear-clip. A collinear triangle
    /// with sparse input `{ {0,1}, {1,2} }` (sharing the corner at v1): the
    /// §4 metric returns 2.0 (a straight chain corner, not doubled-back) at v1
    /// so it never clips, and the doubled-back corners at v0/v2 are skipped
    /// because their incoming edge is not in the input set. The triangle
    /// survives intact.
    #[test]
    fn loop_tested_flag_input_filter_scopes_ear_clip() {
        let mut mesh = LoopMesh::new();
        let ([v0, v1, v2], _f) = triangle(
            &mut mesh,
            [
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(2.0, 0.0, 0.0),
            ],
        );
        let e01 = mesh.query_edge_exists(v0, v1).unwrap();
        let e12 = mesh.query_edge_exists(v1, v2).unwrap();
        let before = (mesh.totvert(), mesh.totedge(), mesh.totface());

        mesh.dissolve_degenerate(&[e01, e12], 0.01);

        assert_eq!(
            (mesh.totvert(), mesh.totedge(), mesh.totface()),
            before,
            "sparse input does not clip the unscoped corners"
        );
        mesh.validate_topology();
    }

    /// §10 — the operator does not iterate internally; a caller-driven
    /// fixed-point loop converges. Two collinear triangles sharing an edge
    /// (each pass may expose new geometry) reach a stable mesh after a bounded
    /// number of passes.
    #[test]
    fn iterative_convergence_caller_driven() {
        let mut mesh = LoopMesh::new();
        let v0 = mesh.create_vertex(Vec3::new(0.0, 0.0, 0.0));
        let v1 = mesh.create_vertex(Vec3::new(1.0, 0.0, 0.0));
        let v2 = mesh.create_vertex(Vec3::new(-1.0, 0.0, 0.0));
        let v3 = mesh.create_vertex(Vec3::new(3.0, 0.0, 0.0));
        let mk_tri = |mesh: &mut LoopMesh, vs: [VertexKey; 3]| {
            let es: Vec<EdgeKey> = (0..3)
                .map(|i| mesh.create_edge([vs[i], vs[(i + 1) % 3]], EdgeCreateFlags::NO_DUPLICATE))
                .collect();
            mesh.face_create(&vs, &es, FaceCreateFlags::NONE);
        };
        mk_tri(&mut mesh, [v0, v1, v2]);
        mk_tri(&mut mesh, [v1, v0, v3]);

        let mut iterations = 0;
        loop {
            let edge_count_before = mesh.totedge();
            let edges = mesh.edge_keys();
            mesh.dissolve_degenerate(&edges, 0.01);
            mesh.validate_topology();
            if mesh.totedge() == edge_count_before {
                break;
            }
            iterations += 1;
            assert!(iterations <= 8, "caller fixed-point loop should converge");
        }

        assert_eq!(mesh.totface(), 0, "the degenerate sliver pair is removed");
    }
}
