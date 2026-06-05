use glam::Vec3;

use super::duplicate::GeomKey;
use crate::loopmesh::*;

/// Which elements the bevel replaces (BEVEL.md §1.2 / §2.2).
///
/// [`BevelAffect::Vertices`] is the foundational single-segment vertex bevel of
/// BEVEL.md §3–§6. [`BevelAffect::Edges`] is the foundational single-segment edge
/// bevel of BEVEL.md §7 — implemented for the **isolated** single manifold edge
/// (no two beveled edges share a vertex); the multi-edge junction is a
/// documented-as-future stub (BEVEL.md §7.6).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum BevelAffect {
    Vertices,
    Edges,
}

impl Default for BevelAffect {
    fn default() -> Self {
        BevelAffect::Vertices
    }
}

/// How `offset` is interpreted (BEVEL.md §1.2 / §4 / §4.3). Each mode changes
/// only how the per-edge ring-vert slide distance `d_i` is derived from `offset`
/// and the local corner geometry; the ring / cap / re-stitch surgery downstream
/// is mode-agnostic (BEVEL.md §4.3).
///
/// All five are implemented for the foundational single-segment vertex bevel:
/// [`BevelOffsetType::Offset`] is the plain slide-along-the-edge distance
/// (`d_i = offset`, BEVEL.md §4); the other four re-measure that slide from a
/// different target quantity (BEVEL.md §4.3).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum BevelOffsetType {
    /// `d_i = offset` — distance slid along each incident edge (BEVEL.md §4).
    Offset,
    /// `d_i = offset / |2·sin φ_i|` — straight-line chamfer-width target.
    Width,
    /// `d_i = offset / |cos φ_i|` — perpendicular-depth target.
    Depth,
    /// `d_i = |A_i − v| · offset / 100` — per-edge fraction of edge length.
    Percent,
    /// `d_i = offset` — fixed slide independent of angle (= `Offset` at the
    /// foundational no-weights level, BEVEL.md §4.3).
    Absolute,
}

impl Default for BevelOffsetType {
    fn default() -> Self {
        BevelOffsetType::Offset
    }
}

/// Parameters for [`LoopMesh::bevel`]. The foundational case (BEVEL.md §3) is
/// reachable by setting only `offset` and leaving every other field at its
/// default: `affect = Vertices`, `segments = 1`, `offset_type = Offset`.
#[derive(Clone, Copy, Debug)]
pub struct BevelOptions {
    /// How much material the bevel consumes, measured per `offset_type`. A
    /// non-positive offset is a no-op (BEVEL.md §2.1).
    pub offset: f32,
    pub offset_type: BevelOffsetType,
    pub segments: u32,
    pub affect: BevelAffect,
    /// When set, the effective offset is reduced just far enough that no ring
    /// vert ever slides past a collision point on its edge, so the result never
    /// self-overlaps (BEVEL.md §4.4). A global, uniform pre-pass on the resolved
    /// slide distances; it changes no topology, only the slide magnitudes.
    pub clamp_overlap: bool,
}

impl Default for BevelOptions {
    fn default() -> Self {
        BevelOptions {
            offset: 0.0,
            offset_type: BevelOffsetType::Offset,
            segments: 1,
            affect: BevelAffect::Vertices,
            clamp_overlap: false,
        }
    }
}

/// New geometry created by the bevel, mirroring the operator's
/// `verts.out` / `edges.out` / `faces.out` slots (BEVEL.md §1.1). The buffers
/// are sets; downstream comparison ignores their enumeration order (§8).
#[derive(Default)]
pub struct BevelResult {
    pub verts: Vec<VertexKey>,
    pub edges: Vec<EdgeKey>,
    pub faces: Vec<FaceKey>,
}

/// One beveled vertex's local frame, captured against the *pristine* mesh
/// before any surgery (BEVEL.md §6.4.1: place every ring vert before cutting any
/// face, so a shared edge's original direction and length are still readable
/// when the second corner is placed). Holds the disk-ordered incident edges, the
/// ring vert minted on each (paired with that edge), the incident faces, and the
/// interior/boundary fan classification.
struct BeveledCorner {
    vertex: VertexKey,
    ring_vert_for_edge: Vec<(EdgeKey, VertexKey)>,
    incident_faces: Vec<FaceKey>,
    interior_fan: bool,
}

/// Per-incident-edge placement record gathered in the slide pre-pass: the edge,
/// its far endpoint position, and the *unclamped* resolved slide distance and
/// per-offset slide rate `k` (BEVEL.md §4.3 / §4.4.2).
struct EdgePlacement {
    edge: EdgeKey,
    far_vertex: VertexKey,
    far_position: Vec3,
    slide: f32,
    rate: f32,
}

/// One beveled vertex's placement frame, captured against the pristine mesh
/// (BEVEL.md §6.4.1): its position and the per-incident-edge placements.
struct BevelFrame {
    vertex: VertexKey,
    vertex_position: Vec3,
    placements: Vec<EdgePlacement>,
}

impl LoopMesh {
    /// Foundational single-segment vertex bevel (BEVEL.md §3–§6).
    ///
    /// For each beveled vertex `v` of valence `n`, this replaces `v` with a ring
    /// of `n` new verts — one per incident edge, slid `offset` toward the far
    /// endpoint (§4) — re-stitches each incident face's `v` corner onto the two
    /// ring verts of that face's bounding edges (§6.1), and, for an interior
    /// (closed-fan) corner, closes the ring with a single cap n-gon (§5/§5.1).
    /// A boundary (open-fan) corner gets no cap (§6.2). The original vertex is
    /// then removed as a now-loose vertex (§5.1.1).
    ///
    /// Implemented for the foundational case: `affect = Vertices`,
    /// `segments = 1`, any `offset_type` (all five modes re-measure the per-edge
    /// slide per BEVEL.md §4 / §4.3). Single-segment `affect = Edges` is supported
    /// for the foundational **isolated** manifold edge (BEVEL.md §7), measured with
    /// `offset_type = Offset`. Multiple segments, profiles, miters, overlap
    /// clamping, flag propagation, and the multi-edge junction are
    /// documented-as-future stubs (BEVEL.md §7.6 / §8) and are ignored here.
    ///
    /// A non-positive `offset`, or a beveled vertex with fewer than two incident
    /// edges, is a no-op for that vertex (BEVEL.md §2.1).
    ///
    /// See `radial-mesh-guides/operators/BEVEL.md`.
    pub fn bevel(&mut self, geom: &[GeomKey], options: BevelOptions) -> BevelResult {
        let mut result = BevelResult::default();

        if options.segments != 1 || options.offset <= 0.0 {
            return result;
        }

        if options.affect == BevelAffect::Edges {
            return self.bevel_edges(geom, options.offset);
        }

        if options.affect != BevelAffect::Vertices {
            return result;
        }

        // The set of beveled verts, de-duplicated and order-preserved. A vert is
        // beveled at most once even if it appears twice in `geom`.
        let mut beveled: Vec<VertexKey> = Vec::new();
        for entry in geom {
            if let GeomKey::Vert(vertex) = *entry {
                if !beveled.contains(&vertex) {
                    beveled.push(vertex);
                }
            }
        }

        // §6.4.1 — place-before-surgery. Two phases against the pristine mesh:
        //
        //   1. Capture each beveled vertex's local frame (disk-ordered incident
        //      edges, far positions, per-edge slide distances) while the mesh is
        //      still pristine, so an edge shared by two beveled verts is measured
        //      from its *original* direction/length at both ends. The slide
        //      pre-pass also feeds the §4.4 global `clamp_overlap` reduction.
        //   2. Mint every ring vert (§4), then run all face surgery and caps.
        //
        // Doing surgery per-vertex inline would re-point a shared edge after the
        // first corner and slide the second corner off the recreated chamfer
        // (§6.4.1), so the two phases must stay separated.
        let frames = self.gather_bevel_frames(&beveled, options.offset, options.offset_type);
        let clamp_factor = if options.clamp_overlap {
            clamp_factor(options.offset, &beveled, &frames)
        } else {
            1.0
        };

        let corners = self.place_ring_verts(&frames, clamp_factor, &mut result);

        // §6.4.3: re-stitch each incident face exactly once, swapping *every*
        // beveled corner of that face at the same time. A face straddling a
        // shared edge has two beveled corners and must not be killed twice.
        self.restitch_incident_faces(&beveled, &corners);

        // §5 / §6.2 / §6.4.4: build each corner's cap over its own ring only —
        // the caps are independent across a shared edge.
        for corner in &corners {
            if corner.interior_fan {
                if let Some(cap) = self.build_cap_face(&corner.ring_vert_for_edge) {
                    result.faces.push(cap.face);
                    result.edges.extend(cap.chamfer_edges);
                }
            } else {
                self.collect_chamfer_edges(&corner.ring_vert_for_edge, &mut result);
            }
        }

        // §5.1.1 / §6.3: every beveled vertex is now loose. Free each.
        for corner in &corners {
            self.vertex_kill(corner.vertex);
        }

        result
    }

    /// Foundational single-segment edge bevel (BEVEL.md §7).
    ///
    /// Bevels each **isolated** manifold edge in `geom`: an edge with exactly two
    /// incident faces whose endpoints are shared with no *second* beveled edge
    /// (§7.2). Each such edge `E = {a, b}` is replaced by four strip verts (two
    /// per endpoint, slid perpendicular into each adjacent face, §7.3), one strip
    /// quad `aL–bL–bR–aR` (§7.1), with each valence-3 terminal endpoint absorbed
    /// into its surviving third face (quad → pentagon, §7.4 / §7.5). The two
    /// original endpoints are then freed as loose verts (§7.1 item 5).
    ///
    /// A non-manifold edge, a non-isolated edge (sharing an endpoint with another
    /// beveled edge — the junction case), or an endpoint of valence ≠ 3 is skipped
    /// here: those are the §7.6 / §8 stubs.
    fn bevel_edges(&mut self, geom: &[GeomKey], offset: f32) -> BevelResult {
        let mut result = BevelResult::default();

        let mut beveled: Vec<EdgeKey> = Vec::new();
        for entry in geom {
            if let GeomKey::Edge(edge) = *entry {
                if !beveled.contains(&edge) {
                    beveled.push(edge);
                }
            }
        }

        // §7.2: the isolated restriction — an endpoint shared by two beveled edges
        // is a junction (§7.6), out of scope. Count endpoint incidences across the
        // selection and drop any edge whose endpoint is shared with another.
        let mut endpoint_uses: Vec<(VertexKey, usize)> = Vec::new();
        for &edge in &beveled {
            for endpoint in self.edge_vert_keys(edge) {
                match endpoint_uses.iter_mut().find(|(v, _)| *v == endpoint) {
                    Some((_, count)) => *count += 1,
                    None => endpoint_uses.push((endpoint, 1)),
                }
            }
        }
        let endpoint_shared = |vertex: VertexKey| -> bool {
            endpoint_uses
                .iter()
                .find(|(v, _)| *v == vertex)
                .map(|(_, count)| *count > 1)
                .unwrap_or(false)
        };

        for &edge in &beveled {
            let [a, b] = self.edge_vert_keys(edge);
            if endpoint_shared(a) || endpoint_shared(b) {
                continue; // junction endpoint — §7.6 stub
            }
            self.bevel_single_edge(edge, offset, &mut result);
        }

        result
    }

    /// Bevel one isolated manifold edge (BEVEL.md §7). Places the four strip verts
    /// against the pristine local geometry (§7.5 placement-before-surgery), then
    /// re-stitches every incident face and builds the strip quad.
    fn bevel_single_edge(&mut self, edge: EdgeKey, offset: f32, result: &mut BevelResult) {
        let [a, b] = self.edge_vert_keys(edge);

        // §7.1: the two radial faces of the manifold edge. A non-manifold edge
        // (radial count != 2) has no well-defined left/right strip and is skipped.
        let radial_faces: Vec<FaceKey> = self
            .edge_loop_keys(edge)
            .iter()
            .map(|&l| self.loop_face(l))
            .collect();
        if radial_faces.len() != 2 {
            return;
        }
        let face_left = radial_faces[0];
        let face_right = radial_faces[1];

        // §7.5 foundational pin: only the valence-3 terminal endpoint close is
        // documented (the surviving third face absorbs the corner). A valence ≥ 4
        // endpoint needs the on-edge-vert fill (§7.5) / junction patch (§7.6).
        let valence = |vertex: VertexKey| self.iter_disk_edges(vertex).count();
        if valence(a) != 3 || valence(b) != 3 {
            return;
        }

        let direction = {
            let segment = self.vertex_position(b) - self.vertex_position(a);
            let length = segment.length();
            if length <= f32::EPSILON {
                return; // degenerate zero-length edge
            }
            segment / length
        };

        let left_normal = self.face_plane_normal(face_left);
        let right_normal = self.face_plane_normal(face_right);
        let left_center = self.face_center_of_mass(face_left);
        let right_center = self.face_center_of_mass(face_right);

        // §7.3: strip vert = endpoint + offset · (in-face perpendicular to E),
        // the perpendicular chosen to point into the face interior (toward its
        // centroid). The face normal disambiguates the in-plane perpendicular.
        let strip_a_left =
            self.create_vertex(strip_vert_position(
                self.vertex_position(a),
                direction,
                left_normal,
                left_center,
                offset,
            ));
        let strip_a_right =
            self.create_vertex(strip_vert_position(
                self.vertex_position(a),
                direction,
                right_normal,
                right_center,
                offset,
            ));
        let strip_b_left =
            self.create_vertex(strip_vert_position(
                self.vertex_position(b),
                direction,
                left_normal,
                left_center,
                offset,
            ));
        let strip_b_right =
            self.create_vertex(strip_vert_position(
                self.vertex_position(b),
                direction,
                right_normal,
                right_center,
                offset,
            ));
        result
            .verts
            .extend([strip_a_left, strip_a_right, strip_b_left, strip_b_right]);

        // The non-`E` edge of each radial face at each endpoint. These are the
        // edges shared between a radial face and the endpoint's *third* face; the
        // re-stitch re-points them onto the strip verts, so the third face must
        // place the matching strip vert next to that edge's far end (§7.4).
        let left_other_a = self.face_other_edge_at(face_left, a, edge);
        let right_other_a = self.face_other_edge_at(face_right, a, edge);
        let left_other_b = self.face_other_edge_at(face_left, b, edge);
        let right_other_b = self.face_other_edge_at(face_right, b, edge);

        let substitutions = EdgeBevelSubstitutions {
            a,
            b,
            face_left,
            face_right,
            strip_a_left,
            strip_a_right,
            strip_b_left,
            strip_b_right,
            left_other_a,
            right_other_a,
            left_other_b,
            right_other_b,
        };

        // §7.1 item 4 / §7.4: re-stitch every face incident to a or b. fL/fR swap
        // their endpoint corners 1-for-1 (a→aL/aR, b→bL/bR), staying the same
        // size and adopting the strip's near long edge as their new `E` side. Each
        // third face replaces its single endpoint corner with the endpoint's two
        // strip verts (1-for-2), growing by one side.
        let mut incident_faces: Vec<FaceKey> = self.vert_face_keys(a);
        for face in self.vert_face_keys(b) {
            if !incident_faces.contains(&face) {
                incident_faces.push(face);
            }
        }
        for face in incident_faces {
            self.restitch_edge_bevel_face(face, &substitutions);
        }

        // §7.1 item 2 / §5.1 winding: build the strip quad over the four strip
        // verts, wound opposite to how fL traverses its new `aL–bL` long edge so
        // the quad and fL share that edge with consistent (radial-2) traversal.
        let strip = self.build_strip_quad(
            strip_a_left,
            strip_b_left,
            strip_b_right,
            strip_a_right,
        );
        result.faces.push(strip.face);
        result.edges.extend(strip.edges);

        // §7.1 item 5 / §6.3: both endpoints are now loose. Free them.
        self.vertex_kill(a);
        self.vertex_kill(b);
    }

    /// Re-stitch one face incident to a beveled edge's endpoint(s) (BEVEL.md §7.4
    /// / §7.5). The two radial faces swap each beveled-edge endpoint corner for the
    /// single matching strip vert (1-for-1); every other face swaps an endpoint
    /// corner for that endpoint's two strip verts (1-for-2), ordered so the strip
    /// vert sharing the re-pointed incoming edge stays adjacent to that edge.
    fn restitch_edge_bevel_face(&mut self, face: FaceKey, sub: &EdgeBevelSubstitutions) {
        let cycle_verts = self.face_vert_keys(face);
        let cycle_length = cycle_verts.len();

        let mut new_cycle: Vec<VertexKey> = Vec::with_capacity(cycle_length + 1);
        for (index, &w) in cycle_verts.iter().enumerate() {
            if w != sub.a && w != sub.b {
                new_cycle.push(w);
                continue;
            }

            if face == sub.face_left {
                new_cycle.push(if w == sub.a { sub.strip_a_left } else { sub.strip_b_left });
            } else if face == sub.face_right {
                new_cycle.push(if w == sub.a { sub.strip_a_right } else { sub.strip_b_right });
            } else {
                // Third face: replace the corner with both strip verts, ordered so
                // the one sharing the incoming edge sits next to `prev`.
                let prev_vertex = cycle_verts[(index + cycle_length - 1) % cycle_length];
                let incoming_edge = self
                    .query_edge_exists(prev_vertex, w)
                    .expect("third face's incoming corner edge exists");
                let (left_other, right_other, strip_left, strip_right) = if w == sub.a {
                    (sub.left_other_a, sub.right_other_a, sub.strip_a_left, sub.strip_a_right)
                } else {
                    (sub.left_other_b, sub.right_other_b, sub.strip_b_left, sub.strip_b_right)
                };
                if incoming_edge == left_other {
                    new_cycle.push(strip_left);
                    new_cycle.push(strip_right);
                } else if incoming_edge == right_other {
                    new_cycle.push(strip_right);
                    new_cycle.push(strip_left);
                } else {
                    // Defensive: an incident face that is neither radial face nor
                    // a genuine third face touching one of the shared edges should
                    // not arise for an isolated valence-3 edge.
                    new_cycle.push(strip_left);
                    new_cycle.push(strip_right);
                }
            }
        }

        self.face_kill(face);
        let new_face = self.face_create_verts(&new_cycle, FaceCreateFlags::NONE);
        self.face_attribs.copy_into(new_face.as_usize(), face.as_usize());
    }

    /// Build the strip quad over the four strip verts in the cyclic order
    /// `aL → bL → bR → aR` (BEVEL.md §7.1 item 2), reversing it when needed so the
    /// quad runs the `aL–bL` long edge opposite to the re-stitched left face's
    /// traversal — the §5.1 winding rule that keeps the shared edge radial-2.
    fn build_strip_quad(
        &mut self,
        a_left: VertexKey,
        b_left: VertexKey,
        b_right: VertexKey,
        a_right: VertexKey,
    ) -> StripQuad {
        let long_edge = self
            .query_edge_exists(a_left, b_left)
            .expect("re-stitched left face minted the strip's long edge aL–bL");
        let face_runs_forward = self
            .edge_loop_keys(long_edge)
            .first()
            .map(|&l| {
                let start = self.loop_vertex(l);
                let end = self.loop_vertex(self.loop_next(l));
                start == a_left && end == b_left
            })
            .unwrap_or(false);

        let mut ring = vec![a_left, b_left, b_right, a_right];
        if face_runs_forward {
            ring.reverse();
        }

        let face = self.face_create_verts(&ring, FaceCreateFlags::NONE);

        let mut edges: Vec<EdgeKey> = (0..ring.len())
            .map(|i| {
                let p = ring[i];
                let q = ring[(i + 1) % ring.len()];
                self.query_edge_exists(p, q).expect("strip quad edge exists")
            })
            .collect();
        edges.sort_by_key(|e| e.as_usize());
        edges.dedup();

        StripQuad { face, edges }
    }

    /// The edge of `face` incident to `vertex` that is **not** `exclude` (BEVEL.md
    /// §7.4): a face corner sits on exactly two of its edges, so excluding the
    /// beveled edge leaves the single other one.
    fn face_other_edge_at(
        &self,
        face: FaceKey,
        vertex: VertexKey,
        exclude: EdgeKey,
    ) -> EdgeKey {
        let mut found = None;
        for loop_key in self.face_loop_keys(face) {
            let loop_edge = self.loop_edge(loop_key);
            if loop_edge == exclude {
                continue;
            }
            let [p, q] = self.edge_vert_keys(loop_edge);
            if p == vertex || q == vertex {
                found = Some(loop_edge);
            }
        }
        found.expect("face has a non-excluded edge at the corner vertex")
    }

    /// A face's geometric normal, computed on the fly from its loop cycle by
    /// Newell's method (BEVEL.md §7.3 needs an in-plane perpendicular; the stored
    /// face normal may be stale during surgery, so this reads positions directly).
    fn face_plane_normal(&self, face: FaceKey) -> Vec3 {
        let verts = self.face_vert_keys(face);
        let mut normal = Vec3::ZERO;
        for i in 0..verts.len() {
            let current = self.vertex_position(verts[i]);
            let next = self.vertex_position(verts[(i + 1) % verts.len()]);
            normal.x += (current.y - next.y) * (current.z + next.z);
            normal.y += (current.z - next.z) * (current.x + next.x);
            normal.z += (current.x - next.x) * (current.y + next.y);
        }
        let length = normal.length();
        if length <= f32::EPSILON {
            Vec3::Z
        } else {
            normal / length
        }
    }

    /// A face's centroid (mean of its corner positions), used to orient the
    /// in-face strip-slide direction toward the face interior (BEVEL.md §7.3).
    fn face_center_of_mass(&self, face: FaceKey) -> Vec3 {
        let verts = self.face_vert_keys(face);
        let mut sum = Vec3::ZERO;
        for &v in &verts {
            sum += self.vertex_position(v);
        }
        sum / verts.len() as f32
    }

    /// Phase 1a (BEVEL.md §6.4.1): capture each beveled vertex's per-edge
    /// placement frame against the pristine mesh — disk-ordered incident edges,
    /// far-endpoint positions, the resolved (unclamped) slide distance and the
    /// per-offset slide rate `k` each edge contributes. A vertex with fewer than
    /// two incident edges has no corner to round (§2.1) and is dropped here.
    fn gather_bevel_frames(
        &self,
        beveled: &[VertexKey],
        offset: f32,
        offset_type: BevelOffsetType,
    ) -> Vec<BevelFrame> {
        let mut frames = Vec::with_capacity(beveled.len());
        for &vertex in beveled {
            let incident_edges: Vec<EdgeKey> = self.iter_disk_edges(vertex).collect();
            if incident_edges.len() < 2 {
                continue;
            }

            let vertex_position = self.vertex_position(vertex);
            let far_verts: Vec<VertexKey> = incident_edges
                .iter()
                .map(|&edge| {
                    let [a, b] = self.edge_vert_keys(edge);
                    if a == vertex { b } else { a }
                })
                .collect();
            let far_positions: Vec<Vec3> =
                far_verts.iter().map(|&f| self.vertex_position(f)).collect();
            let corner_axis = corner_axis(vertex_position, &far_positions);

            frames.push(BevelFrame {
                vertex,
                vertex_position,
                placements: incident_edges
                    .iter()
                    .zip(&far_verts)
                    .zip(&far_positions)
                    .map(|((&edge, &far_vertex), &far_position)| {
                        let slide = slide_distance(
                            offset,
                            offset_type,
                            vertex_position,
                            far_position,
                            corner_axis,
                        );
                        // §4.4.2: the per-offset slide rate `k = d_i / offset`,
                        // the §4.3 `offset_type` factor used by the collision
                        // clamp. Derived from the resolved slide so every mode
                        // (including the angle modes and PERCENT) feeds the same
                        // collision computation.
                        let rate = if offset != 0.0 { slide / offset } else { 0.0 };
                        EdgePlacement {
                            edge,
                            far_vertex,
                            far_position,
                            slide,
                            rate,
                        }
                    })
                    .collect(),
            });
        }
        frames
    }

    /// Phase 1b: mint every ring vert (§4) at its clamp-scaled slide position,
    /// and capture each beveled vertex's interior/boundary classification and
    /// incident-face set. No face surgery happens here — only vertex creation.
    fn place_ring_verts(
        &mut self,
        frames: &[BevelFrame],
        clamp_factor: f32,
        result: &mut BevelResult,
    ) -> Vec<BeveledCorner> {
        let mut corners = Vec::with_capacity(frames.len());
        for frame in frames {
            let incident_edges: Vec<EdgeKey> =
                frame.placements.iter().map(|p| p.edge).collect();
            let incident_faces = self.vert_face_keys(frame.vertex);
            let interior_fan =
                self.is_interior_fan(frame.vertex, &incident_edges, &incident_faces);

            let mut ring_vert_for_edge: Vec<(EdgeKey, VertexKey)> =
                Vec::with_capacity(frame.placements.len());
            for placement in &frame.placements {
                let slide = placement.slide * clamp_factor;
                let ring_position =
                    ring_vert_position(frame.vertex_position, placement.far_position, slide);
                let ring_vertex = self.create_vertex(ring_position);
                ring_vert_for_edge.push((placement.edge, ring_vertex));
                result.verts.push(ring_vertex);
            }

            corners.push(BeveledCorner {
                vertex: frame.vertex,
                ring_vert_for_edge,
                incident_faces,
                interior_fan,
            });
        }
        corners
    }

    /// Phase 2 (BEVEL.md §6.1 / §6.4.3): re-stitch every face incident to any
    /// beveled vertex, swapping *all* of that face's beveled corners at once.
    /// Each beveled corner `v` becomes the pair (ring vert of `v`'s incoming
    /// edge, ring vert of `v`'s outgoing edge), keeping the face winding; the
    /// chamfer edge between the pair is minted by the rebuild. A loop edge
    /// between two beveled corners (a shared edge `E`, §6.4.3) is re-pointed from
    /// `{a, b}` to `{r_a, r_b}` — the surviving inner segment — by virtue of both
    /// its endpoints being swapped in the same rebuild.
    fn restitch_incident_faces(&mut self, beveled: &[VertexKey], corners: &[BeveledCorner]) {
        // Collect each distinct incident face once, regardless of how many
        // beveled corners it has, so a shared-edge face is rebuilt a single time.
        let mut faces: Vec<FaceKey> = Vec::new();
        for corner in corners {
            for &face in &corner.incident_faces {
                if !faces.contains(&face) {
                    faces.push(face);
                }
            }
        }

        for face in faces {
            let cycle_verts = self.face_vert_keys(face);
            let cycle_length = cycle_verts.len();

            let mut new_cycle: Vec<VertexKey> = Vec::with_capacity(cycle_length + 1);
            for (index, &w) in cycle_verts.iter().enumerate() {
                if beveled.contains(&w) {
                    let corner = corners
                        .iter()
                        .find(|c| c.vertex == w)
                        .expect("a beveled corner vertex has a ring");
                    let prev_vertex =
                        cycle_verts[(index + cycle_length - 1) % cycle_length];
                    let next_vertex = cycle_verts[(index + 1) % cycle_length];

                    let incoming_edge = self
                        .query_edge_exists(prev_vertex, w)
                        .expect("incident face's incoming corner edge exists");
                    let outgoing_edge = self
                        .query_edge_exists(w, next_vertex)
                        .expect("incident face's outgoing corner edge exists");

                    new_cycle
                        .push(self.ring_vertex_of(&corner.ring_vert_for_edge, incoming_edge));
                    new_cycle
                        .push(self.ring_vertex_of(&corner.ring_vert_for_edge, outgoing_edge));
                } else {
                    new_cycle.push(w);
                }
            }

            self.face_kill(face);
            let new_face = self.face_create_verts(&new_cycle, FaceCreateFlags::NONE);
            self.face_attribs.copy_into(new_face.as_usize(), face.as_usize());
        }
    }

    fn ring_vertex_of(
        &self,
        ring_vert_for_edge: &[(EdgeKey, VertexKey)],
        edge: EdgeKey,
    ) -> VertexKey {
        ring_vert_for_edge
            .iter()
            .find_map(|&(e, r)| (e == edge).then_some(r))
            .expect("every incident edge has a ring vertex")
    }

    /// Classify the corner as an interior (closed) fan vs a boundary (open) fan
    /// (BEVEL.md §2.3 / §6). A corner is interior when every incident edge is
    /// manifold (exactly two incident faces) and the fan closes — the count of
    /// incident faces equals the count of incident edges. Any boundary edge
    /// (radial count below two) opens the fan.
    fn is_interior_fan(
        &self,
        vertex: VertexKey,
        incident_edges: &[EdgeKey],
        incident_faces: &[FaceKey],
    ) -> bool {
        let _ = vertex;
        let all_manifold = incident_edges
            .iter()
            .all(|&edge| self.edge_radial_iter(edge).count() == 2);
        all_manifold && incident_faces.len() == incident_edges.len()
    }

    /// Walk the disk's chamfer edges into an ordered ring and build the cap face
    /// over them, wound opposite the incident faces (BEVEL.md §5.1).
    fn build_cap_face(
        &mut self,
        ring_vert_for_edge: &[(EdgeKey, VertexKey)],
    ) -> Option<CapFace> {
        let ring_verts: Vec<VertexKey> = ring_vert_for_edge.iter().map(|&(_, r)| r).collect();

        // The chamfer edges minted by the re-stitch connect ring verts pairwise.
        // For an interior fan they form one closed cycle through all ring verts.
        // §6.4.4: only this corner's own ring verts are considered, so the
        // surviving inner segment of a shared edge (which joins a ring vert of
        // *this* corner to a ring vert of the *other* corner) is never mistaken
        // for a chamfer edge of this cap.
        let chamfer_edges = self.chamfer_edges_among(&ring_verts);
        if chamfer_edges.len() != ring_verts.len() {
            return None;
        }

        let ordered_ring = order_ring_cycle(self, &ring_verts, &chamfer_edges)?;

        // §5.1: wind the cap opposite to how the incident faces traverse the
        // chamfer edges. Take the first chamfer edge's single incident loop and
        // read its direction; if the face goes a -> b on that edge, the cap goes
        // b -> a, so reverse the ring whenever it currently runs the face's way.
        let first_a = ordered_ring[0];
        let first_b = ordered_ring[1];
        let shared_edge = self
            .query_edge_exists(first_a, first_b)
            .expect("ordered ring pair shares a chamfer edge");
        let face_runs_forward = {
            let incident_loops: Vec<LoopKey> = self.edge_loop_keys(shared_edge);
            incident_loops
                .first()
                .map(|&l| {
                    let start = self.loop_vertex(l);
                    let end = self.loop_vertex(self.loop_next(l));
                    start == first_a && end == first_b
                })
                .unwrap_or(false)
        };

        let mut cap_ring = ordered_ring;
        if face_runs_forward {
            cap_ring.reverse();
        }

        let cap_edges: Vec<EdgeKey> = (0..cap_ring.len())
            .map(|i| {
                let a = cap_ring[i];
                let b = cap_ring[(i + 1) % cap_ring.len()];
                self.query_edge_exists(a, b)
                    .expect("cap reuses the existing chamfer edge")
            })
            .collect();

        let cap = self.face_create(&cap_ring, &cap_edges, FaceCreateFlags::NONE);

        let mut unique_edges = cap_edges;
        unique_edges.sort_by_key(|e| e.as_usize());
        unique_edges.dedup();

        Some(CapFace {
            face: cap,
            chamfer_edges: unique_edges,
        })
    }

    fn collect_chamfer_edges(
        &self,
        ring_vert_for_edge: &[(EdgeKey, VertexKey)],
        result: &mut BevelResult,
    ) {
        let ring_verts: Vec<VertexKey> = ring_vert_for_edge.iter().map(|&(_, r)| r).collect();
        result.edges.extend(self.chamfer_edges_among(&ring_verts));
    }

    /// Every edge that connects two ring verts — the chamfer edges minted by the
    /// re-stitch. Excludes the original incident edges (which connect a ring vert
    /// to a far neighbour outside the ring set).
    fn chamfer_edges_among(&self, ring_verts: &[VertexKey]) -> Vec<EdgeKey> {
        let mut edges: Vec<EdgeKey> = Vec::new();
        for i in 0..ring_verts.len() {
            for j in (i + 1)..ring_verts.len() {
                if let Some(edge) = self.query_edge_exists(ring_verts[i], ring_verts[j]) {
                    if !edges.contains(&edge) {
                        edges.push(edge);
                    }
                }
            }
        }
        edges
    }
}

struct CapFace {
    face: FaceKey,
    chamfer_edges: Vec<EdgeKey>,
}

/// The new strip face and its edges (BEVEL.md §7.1 item 2).
struct StripQuad {
    face: FaceKey,
    edges: Vec<EdgeKey>,
}

/// The per-edge frame an edge bevel's re-stitch reads: the beveled edge, its two
/// endpoints, its two radial faces, the four strip verts, and each radial face's
/// non-`E` edge at each endpoint (BEVEL.md §7.4). Captured against the pristine
/// local geometry before any face is cut (§7.5).
struct EdgeBevelSubstitutions {
    a: VertexKey,
    b: VertexKey,
    face_left: FaceKey,
    face_right: FaceKey,
    strip_a_left: VertexKey,
    strip_a_right: VertexKey,
    strip_b_left: VertexKey,
    strip_b_right: VertexKey,
    left_other_a: EdgeKey,
    right_other_a: EdgeKey,
    left_other_b: EdgeKey,
    right_other_b: EdgeKey,
}

#[cfg(test)]
mod tests {
    use glam::Vec3;

    use super::*;
    use crate::loopmesh::test_support::mesh_add_cube;

    fn counts(mesh: &LoopMesh) -> (usize, usize, usize) {
        (mesh.totvert(), mesh.totedge(), mesh.totface())
    }

    fn vert_at(mesh: &LoopMesh, position: Vec3) -> VertexKey {
        mesh.vert_keys()
            .into_iter()
            .find(|&v| mesh.vertex_position(v) == position)
            .expect("vertex at position exists")
    }

    /// A ring vert is "at" `position` within tolerance; returns whether any
    /// new vert from the result landed there.
    fn result_has_vert_near(mesh: &LoopMesh, result: &BevelResult, position: Vec3) -> usize {
        result
            .verts
            .iter()
            .filter(|&&v| (mesh.vertex_position(v) - position).length() < 1.0e-5)
            .count()
    }

    /// BEVEL.md §4.4.3: beveling the two shared-edge endpoints of the cube with
    /// `clamp_overlap = true` at `offset = 1.5` reduces the effective offset to
    /// `1.0` (clamp factor `1.0 / 1.5`), so every ring vert slides exactly `1.0`
    /// — not the requested `1.5`.
    #[test]
    fn clamp_overlap_reduces_slide_to_collision_offset() {
        let mut mesh = LoopMesh::new();
        mesh_add_cube(&mut mesh);
        // v0 = (1,-1,-1), v1 = (-1,-1,-1): the two endpoints of edge {0,1}.
        let a = vert_at(&mesh, Vec3::new(1.0, -1.0, -1.0));
        let b = vert_at(&mesh, Vec3::new(-1.0, -1.0, -1.0));
        let pos_a = mesh.vertex_position(a);
        let pos_b = mesh.vertex_position(b);

        let result = mesh.bevel(
            &[a.into(), b.into()],
            BevelOptions {
                offset: 1.5,
                clamp_overlap: true,
                ..Default::default()
            },
        );

        // Each ring vert slid from its own corner; the clamp pins the slide to
        // 1.0, the §4.4.3 collision offset, on every beveled edge.
        for &ring in &result.verts {
            let ring_position = mesh.vertex_position(ring);
            let slide = (ring_position - pos_a).length().min((ring_position - pos_b).length());
            assert!(
                (slide - 1.0).abs() < 1.0e-5,
                "clamped slide {slide} should be 1.0 (effective offset), not 1.5",
            );
        }
        mesh.validate_topology();
    }

    /// BEVEL.md §6.4.1: with `clamp_overlap = false` and `offset = 1.5`, the two
    /// shared-edge ring verts each slide the full `1.5` and cross past each other:
    /// `r_a` (from the `a = (1,-1,-1)` end) lands at `(-0.5,-1,-1)` and `r_b`
    /// (from `b = (-1,-1,-1)`) at `(0.5,-1,-1)`. Each is placed off the *original*
    /// edge — the place-before-surgery ordering.
    #[test]
    fn unclamped_adjacent_pair_crosses_past() {
        let mut mesh = LoopMesh::new();
        mesh_add_cube(&mut mesh);
        let a = vert_at(&mesh, Vec3::new(1.0, -1.0, -1.0));
        let b = vert_at(&mesh, Vec3::new(-1.0, -1.0, -1.0));

        let result = mesh.bevel(
            &[a.into(), b.into()],
            BevelOptions {
                offset: 1.5,
                clamp_overlap: false,
                ..Default::default()
            },
        );

        // The two ring verts on the shared edge land at the crossed-past
        // positions; each is a distinct ring vert.
        assert_eq!(
            result_has_vert_near(&mesh, &result, Vec3::new(-0.5, -1.0, -1.0)),
            1,
            "a-side ring vert at (-0.5,-1,-1)",
        );
        assert_eq!(
            result_has_vert_near(&mesh, &result, Vec3::new(0.5, -1.0, -1.0)),
            1,
            "b-side ring vert at (0.5,-1,-1)",
        );
        mesh.validate_topology();
    }

    /// BEVEL.md §6.4.2: the clamped-collapse degenerate. At the exact collision
    /// (`offset = 1.5` clamped to effective `1.0`, `d_a + d_b == L == 2`), the
    /// two shared-edge ring verts land *coincident* at the edge midpoint
    /// `(0,-1,-1)` — but remain **two distinct ring verts** joined by a
    /// zero-length inner segment, with no weld / merge.
    #[test]
    fn clamped_collapse_is_coincident_pair_not_merge() {
        let mut mesh = LoopMesh::new();
        mesh_add_cube(&mut mesh);
        let a = vert_at(&mesh, Vec3::new(1.0, -1.0, -1.0));
        let b = vert_at(&mesh, Vec3::new(-1.0, -1.0, -1.0));

        let result = mesh.bevel(
            &[a.into(), b.into()],
            BevelOptions {
                offset: 1.5,
                clamp_overlap: true,
                ..Default::default()
            },
        );

        // Exactly two distinct ring verts coincide at the midpoint (one from each
        // corner) — not one merged vertex.
        let midpoint = Vec3::new(0.0, -1.0, -1.0);
        let coincident: Vec<VertexKey> = result
            .verts
            .iter()
            .copied()
            .filter(|&v| (mesh.vertex_position(v) - midpoint).length() < 1.0e-5)
            .collect();
        assert_eq!(
            coincident.len(),
            2,
            "two distinct ring verts coincide at the midpoint, no merge",
        );
        assert_ne!(coincident[0], coincident[1], "the pair is two distinct verts");

        // They are joined by a real (zero-length) inner-segment edge.
        let segment = mesh.query_edge_exists(coincident[0], coincident[1]);
        assert!(
            segment.is_some(),
            "the surviving inner segment edge joins the coincident pair",
        );
        let length = {
            let [p, q] = mesh.edge_vert_keys(segment.unwrap());
            (mesh.vertex_position(p) - mesh.vertex_position(q)).length()
        };
        assert!(length < 1.0e-5, "the inner segment is zero-length");
        mesh.validate_topology();
    }

    /// BEVEL.md §6.4.3: the original shared edge survives as the inner segment
    /// `r_a`–`r_b` between the two corners' chamfer edges, and (being an interior
    /// manifold edge) keeps radial count 2 — both straddling faces re-stitched
    /// onto it.
    #[test]
    fn shared_edge_survives_as_inner_segment_radial_two() {
        let mut mesh = LoopMesh::new();
        mesh_add_cube(&mut mesh);
        let a = vert_at(&mesh, Vec3::new(1.0, -1.0, -1.0));
        let b = vert_at(&mesh, Vec3::new(-1.0, -1.0, -1.0));

        let result = mesh.bevel(
            &[a.into(), b.into()],
            BevelOptions {
                offset: 0.5,
                clamp_overlap: false,
                ..Default::default()
            },
        );

        // r_a slid 0.5 from (1,-1,-1) toward (-1,-1,-1) -> (0.5,-1,-1).
        // r_b slid 0.5 from (-1,-1,-1) toward (1,-1,-1) -> (-0.5,-1,-1).
        let r_a = vert_at(&mesh, Vec3::new(0.5, -1.0, -1.0));
        let r_b = vert_at(&mesh, Vec3::new(-0.5, -1.0, -1.0));
        let segment = mesh
            .query_edge_exists(r_a, r_b)
            .expect("the shared edge survives as the r_a-r_b inner segment");

        assert_eq!(
            mesh.edge_radial_iter(segment).count(),
            2,
            "the surviving inner segment keeps radial count 2 (both straddling faces)",
        );

        // The two caps are independent (§6.4.4): the inner segment is on the
        // re-stitched faces, never on either cap, so it is not a chamfer edge.
        // Two distinct triangular caps were produced.
        assert_eq!(result.faces.len(), 2, "two independent corner caps");
        for &cap in &result.faces {
            assert_eq!(mesh.face_vert_keys(cap).len(), 3, "each cap is a triangle");
            assert!(
                !mesh.face_vert_keys(cap).contains(&r_a)
                    || !mesh.face_vert_keys(cap).contains(&r_b),
                "no single cap spans both r_a and r_b",
            );
        }
        mesh.validate_topology();
    }

    /// BEVEL.md §6.1 / §2.3: an interior valence-3 cube corner is replaced by
    /// three ring verts, a triangular cap, and three re-stitched faces.
    /// Count delta +2 V / +3 E / +1 F.
    #[test]
    fn interior_closed_fan_cube_corner() {
        let mut mesh = LoopMesh::new();
        mesh_add_cube(&mut mesh);
        let before = counts(&mesh);
        assert_eq!(before, (8, 12, 6));

        // v0 = (1, -1, -1): an interior valence-3 corner (Front/Left/Bottom).
        let corner = mesh
            .vert_keys()
            .into_iter()
            .find(|&v| mesh.vertex_position(v) == Vec3::new(1.0, -1.0, -1.0))
            .expect("cube corner exists");

        let result = mesh.bevel(
            &[corner.into()],
            BevelOptions {
                offset: 0.25,
                ..Default::default()
            },
        );

        assert_eq!(counts(&mesh), (10, 15, 7), "interior +2 V / +3 E / +1 F");
        assert_eq!(result.verts.len(), 3, "three ring verts");
        assert_eq!(result.faces.len(), 1, "single triangular cap");

        // The cap is a triangle (n = 3).
        let cap = result.faces[0];
        assert_eq!(mesh.face_vert_keys(cap).len(), 3, "triangular cap");

        // The original corner is gone.
        assert!(
            !mesh
                .vert_keys()
                .into_iter()
                .any(|v| v == corner),
            "beveled vertex removed",
        );

        mesh.validate_topology();
    }

    /// BEVEL.md §4: ring verts slide `offset` along each incident edge toward the
    /// far endpoint. A larger offset slides them proportionally further.
    #[test]
    fn ring_verts_slide_proportionally() {
        let make = |offset: f32| {
            let mut mesh = LoopMesh::new();
            mesh_add_cube(&mut mesh);
            let corner = mesh
                .vert_keys()
                .into_iter()
                .find(|&v| mesh.vertex_position(v) == Vec3::new(1.0, -1.0, -1.0))
                .unwrap();
            let result = mesh.bevel(
                &[corner.into()],
                BevelOptions {
                    offset,
                    ..Default::default()
                },
            );
            // Distance of each ring vert from the (known) original corner.
            let origin = Vec3::new(1.0, -1.0, -1.0);
            result
                .verts
                .iter()
                .map(|&v| (mesh.vertex_position(v) - origin).length())
                .collect::<Vec<_>>()
        };

        let small = make(0.25);
        let large = make(0.5);
        for distance in &small {
            assert!((distance - 0.25).abs() < 1.0e-5, "small slide is the offset");
        }
        for distance in &large {
            assert!((distance - 0.5).abs() < 1.0e-5, "large slide is the offset");
        }
    }

    /// BEVEL.md §6.2 / §2.3: a boundary valence-2 corner of a flat grid quad
    /// becomes an open chamfer — two ring verts, one chamfer edge, no cap.
    /// Count delta +1 V / +1 E / +0 F.
    #[test]
    fn boundary_open_fan_grid_corner() {
        let mut mesh = LoopMesh::new();
        let positions = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        ];
        let verts: Vec<VertexKey> = positions.iter().map(|&p| mesh.create_vertex(p)).collect();
        let edges: Vec<EdgeKey> = (0..4)
            .map(|i| mesh.create_edge([verts[i], verts[(i + 1) % 4]], EdgeCreateFlags::NONE))
            .collect();
        mesh.face_create(&verts, &edges, FaceCreateFlags::NONE);

        let before = counts(&mesh);
        assert_eq!(before, (4, 4, 1));

        let corner = verts[0];
        let result = mesh.bevel(
            &[corner.into()],
            BevelOptions {
                offset: 0.25,
                ..Default::default()
            },
        );

        assert_eq!(counts(&mesh), (5, 5, 1), "boundary +1 V / +1 E / +0 F");
        assert_eq!(result.verts.len(), 2, "two ring verts");
        assert!(result.faces.is_empty(), "no cap on an open fan");

        // The single original face is now a pentagon (quad's corner cut off).
        let face = mesh.face_keys()[0];
        assert_eq!(mesh.face_vert_keys(face).len(), 5, "corner cut into two");

        mesh.validate_topology();
    }

    /// BEVEL.md §2.1: a non-positive offset is a no-op.
    #[test]
    fn non_positive_offset_is_noop() {
        let mut mesh = LoopMesh::new();
        mesh_add_cube(&mut mesh);
        let before = counts(&mesh);
        let corner = mesh.vert_keys()[0];

        let result = mesh.bevel(
            &[corner.into()],
            BevelOptions {
                offset: 0.0,
                ..Default::default()
            },
        );

        assert_eq!(counts(&mesh), before, "offset <= 0 changes nothing");
        assert!(result.verts.is_empty() && result.edges.is_empty() && result.faces.is_empty());
        mesh.validate_topology();
    }

    /// BEVEL.md §7.6 / §8: multi-segment is an unimplemented stub — a no-op, not a
    /// silent wrong result. (Edge bevel `affect = Edges` *is* implemented for the
    /// isolated single edge, §7; see the edge-bevel tests below.)
    #[test]
    fn unsupported_modes_are_noops() {
        let mut mesh = LoopMesh::new();
        mesh_add_cube(&mut mesh);
        let before = counts(&mesh);
        let corner = mesh.vert_keys()[0];

        let multi_segment = mesh.bevel(
            &[corner.into()],
            BevelOptions {
                offset: 0.25,
                segments: 2,
                ..Default::default()
            },
        );
        assert_eq!(counts(&mesh), before);
        assert!(multi_segment.verts.is_empty());
        mesh.validate_topology();
    }

    /// The cube edge between the verts at `pa` and `pb`.
    fn cube_edge(mesh: &LoopMesh, pa: Vec3, pb: Vec3) -> EdgeKey {
        let a = vert_at(mesh, pa);
        let b = vert_at(mesh, pb);
        mesh.query_edge_exists(a, b).expect("cube edge exists")
    }

    /// BEVEL.md §7.4: beveling one isolated manifold cube edge ({0,1}) with
    /// `affect = Edges` replaces the crease with four strip verts and one strip
    /// quad, each valence-3 endpoint absorbed into its third face. Count delta
    /// +2 V / +3 E / +1 F over the cube's 8 V / 12 E / 6 F.
    #[test]
    fn edge_bevel_isolated_cube_edge_count_delta() {
        let mut mesh = LoopMesh::new();
        mesh_add_cube(&mut mesh);
        assert_eq!(counts(&mesh), (8, 12, 6));

        let a = vert_at(&mesh, Vec3::new(1.0, -1.0, -1.0));
        let b = vert_at(&mesh, Vec3::new(-1.0, -1.0, -1.0));
        let edge = mesh.query_edge_exists(a, b).expect("edge {0,1}");

        let result = mesh.bevel(
            &[edge.into()],
            BevelOptions {
                offset: 0.25,
                affect: BevelAffect::Edges,
                ..Default::default()
            },
        );

        assert_eq!(counts(&mesh), (10, 15, 7), "isolated edge +2 V / +3 E / +1 F");
        assert_eq!(result.verts.len(), 4, "four strip verts");
        assert_eq!(result.faces.len(), 1, "the strip quad is the only new face");

        // The strip face is a quad.
        let strip = result.faces[0];
        assert_eq!(mesh.face_vert_keys(strip).len(), 4, "strip quad");

        // Both original endpoints are removed.
        assert!(
            !mesh.vert_keys().into_iter().any(|v| v == a || v == b),
            "both beveled-edge endpoints removed",
        );
        mesh.validate_topology();
    }

    /// BEVEL.md §7.4: the strip quad's four corners sit a perpendicular `offset`
    /// from the original edge line — two per endpoint, one into each adjacent
    /// face. For edge {0,1} (the front/bottom crease along −Y/−Z) the strip verts
    /// land in the front (z = −1) and bottom (y = −1) planes, `offset` off the
    /// crease.
    #[test]
    fn edge_bevel_strip_verts_perpendicular_offset() {
        let mut mesh = LoopMesh::new();
        mesh_add_cube(&mut mesh);
        let edge = cube_edge(&mesh, Vec3::new(1.0, -1.0, -1.0), Vec3::new(-1.0, -1.0, -1.0));

        let offset = 0.25_f32;
        let result = mesh.bevel(
            &[edge.into()],
            BevelOptions {
                offset,
                affect: BevelAffect::Edges,
                ..Default::default()
            },
        );

        // The crease line is x ∈ [−1,1], y = −1, z = −1. Each strip vert keeps the
        // endpoint's x, and slides `offset` along exactly one of −Y/−Z into a face:
        // the front face (z = −1) strip verts move in +Y (to y = −1+offset, z = −1),
        // the bottom face (y = −1) strip verts move in +Z (to y = −1, z = −1+offset).
        for &v in &result.verts {
            let p = mesh.vertex_position(v);
            assert!(p.x.abs() <= 1.0 + 1.0e-5, "strip vert keeps endpoint x");
            let into_front = (p - Vec3::new(p.x, -1.0 + offset, -1.0)).length();
            let into_bottom = (p - Vec3::new(p.x, -1.0, -1.0 + offset)).length();
            assert!(
                into_front < 1.0e-5 || into_bottom < 1.0e-5,
                "strip vert at {p:?} must lie {offset} into the front or bottom face",
            );
        }
        mesh.validate_topology();
    }

    /// BEVEL.md §7.4: after the bevel, the two `E`-adjacent faces (fL/fR) stay
    /// quads (swapping their `E` side for the strip's long edge) and the two
    /// third faces grow by one corner (quad → pentagon). The strip quad shares
    /// each of its four edges with exactly one other face (radial count 2).
    #[test]
    fn edge_bevel_face_sizes_and_strip_manifold() {
        let mut mesh = LoopMesh::new();
        mesh_add_cube(&mut mesh);
        let edge = cube_edge(&mesh, Vec3::new(1.0, -1.0, -1.0), Vec3::new(-1.0, -1.0, -1.0));

        let result = mesh.bevel(
            &[edge.into()],
            BevelOptions {
                offset: 0.25,
                affect: BevelAffect::Edges,
                ..Default::default()
            },
        );

        // Two pentagons (the third faces, originally quads, each +1 corner); the
        // remaining four pre-existing faces stay quads, plus the new strip quad.
        let sizes: Vec<usize> = mesh
            .face_keys()
            .into_iter()
            .map(|f| mesh.face_vert_keys(f).len())
            .collect();
        let pentagons = sizes.iter().filter(|&&n| n == 5).count();
        let quads = sizes.iter().filter(|&&n| n == 4).count();
        assert_eq!(pentagons, 2, "two third faces grew quad → pentagon");
        assert_eq!(quads, 5, "fL, fR, two untouched faces, and the strip quad");

        // Every strip-quad edge is shared with exactly one other face.
        let strip = result.faces[0];
        for loop_key in mesh.face_loop_keys(strip) {
            let strip_edge = mesh.loop_edge(loop_key);
            assert_eq!(
                mesh.edge_radial_iter(strip_edge).count(),
                2,
                "each strip edge is manifold (radial count 2)",
            );
        }
        mesh.validate_topology();
    }

    /// BEVEL.md §9: the §7 placement is not specific to one edge's orientation —
    /// the differently-oriented edge {3,7} (the front∩... rim along +Z) yields the
    /// same +2 V / +3 E / +1 F delta and strip-quad topology.
    #[test]
    fn edge_bevel_other_orientation_same_delta() {
        let mut mesh = LoopMesh::new();
        mesh_add_cube(&mut mesh);
        // vert 3 = (1,1,-1), vert 7 = (1,1,1).
        let edge = cube_edge(&mesh, Vec3::new(1.0, 1.0, -1.0), Vec3::new(1.0, 1.0, 1.0));

        let result = mesh.bevel(
            &[edge.into()],
            BevelOptions {
                offset: 0.25,
                affect: BevelAffect::Edges,
                ..Default::default()
            },
        );

        assert_eq!(counts(&mesh), (10, 15, 7), "same +2 V / +3 E / +1 F delta");
        assert_eq!(result.verts.len(), 4, "four strip verts");
        assert_eq!(result.faces.len(), 1, "single strip quad");
        assert_eq!(mesh.face_vert_keys(result.faces[0]).len(), 4, "strip quad");
        mesh.validate_topology();
    }

    /// BEVEL.md §2.1: a non-positive offset is a no-op for edge bevel too.
    #[test]
    fn edge_bevel_non_positive_offset_is_noop() {
        let mut mesh = LoopMesh::new();
        mesh_add_cube(&mut mesh);
        let before = counts(&mesh);
        let edge = cube_edge(&mesh, Vec3::new(1.0, -1.0, -1.0), Vec3::new(-1.0, -1.0, -1.0));

        let result = mesh.bevel(
            &[edge.into()],
            BevelOptions {
                offset: 0.0,
                affect: BevelAffect::Edges,
                ..Default::default()
            },
        );
        assert_eq!(counts(&mesh), before, "offset <= 0 changes nothing");
        assert!(result.verts.is_empty() && result.faces.is_empty());
        mesh.validate_topology();
    }

    /// Bevel cube-corner vert `(1, -1, -1)` under `offset_type`, returning each
    /// ring vert's slide distance from the original corner.
    fn cube_corner_slides(offset: f32, offset_type: BevelOffsetType) -> Vec<f32> {
        let mut mesh = LoopMesh::new();
        mesh_add_cube(&mut mesh);
        let origin = Vec3::new(1.0, -1.0, -1.0);
        let corner = mesh
            .vert_keys()
            .into_iter()
            .find(|&v| mesh.vertex_position(v) == origin)
            .unwrap();
        let result = mesh.bevel(
            &[corner.into()],
            BevelOptions {
                offset,
                offset_type,
                ..Default::default()
            },
        );
        mesh.validate_topology();
        result
            .verts
            .iter()
            .map(|&v| (mesh.vertex_position(v) - origin).length())
            .collect()
    }

    /// BEVEL.md §4.3: `offset_type` changes only the slide distance, never the
    /// topology. Every mode produces the same +2 V / +3 E / +1 F cube-corner
    /// delta.
    #[test]
    fn offset_type_modes_are_topology_invariant() {
        for offset_type in [
            BevelOffsetType::Offset,
            BevelOffsetType::Width,
            BevelOffsetType::Depth,
            BevelOffsetType::Percent,
            BevelOffsetType::Absolute,
        ] {
            let mut mesh = LoopMesh::new();
            mesh_add_cube(&mut mesh);
            let corner = mesh
                .vert_keys()
                .into_iter()
                .find(|&v| mesh.vertex_position(v) == Vec3::new(1.0, -1.0, -1.0))
                .unwrap();
            mesh.bevel(
                &[corner.into()],
                BevelOptions {
                    offset: 0.3,
                    offset_type,
                    ..Default::default()
                },
            );
            assert_eq!(
                counts(&mesh),
                (10, 15, 7),
                "{offset_type:?}: topology is mode-agnostic",
            );
        }
    }

    /// BEVEL.md §4.3 / §8: at the cube corner every edge's axis-relative angle is
    /// `φ = arccos(1/√3) ≈ 54.74°`, so `WIDTH` slides by
    /// `offset / |2·sin φ| ≈ 0.612·offset`, distinct from `OFFSET`'s `offset`.
    #[test]
    fn width_slide_chamfer_width_target() {
        let offset = 0.3;
        let phi = (1.0_f32 / 3.0_f32.sqrt()).acos();
        let expected = offset / (2.0 * phi.sin()).abs();
        for slide in cube_corner_slides(offset, BevelOffsetType::Width) {
            assert!(
                (slide - expected).abs() < 1.0e-5,
                "WIDTH slide {slide} should be {expected}",
            );
            assert!(
                (slide - offset).abs() > 1.0e-3,
                "WIDTH must differ from OFFSET at the cube corner",
            );
        }
    }

    /// BEVEL.md §4.3 / §8: `DEPTH` slides by `offset / |cos φ| ≈ 1.73·offset` at
    /// the cube corner, distinct from `OFFSET`.
    #[test]
    fn depth_slide_perpendicular_depth_target() {
        let offset = 0.3;
        let phi = (1.0_f32 / 3.0_f32.sqrt()).acos();
        let expected = offset / phi.cos().abs();
        for slide in cube_corner_slides(offset, BevelOffsetType::Depth) {
            assert!(
                (slide - expected).abs() < 1.0e-5,
                "DEPTH slide {slide} should be {expected}",
            );
            assert!(
                (slide - offset).abs() > 1.0e-3,
                "DEPTH must differ from OFFSET at the cube corner",
            );
        }
    }

    /// BEVEL.md §4.3: `PERCENT` slides a fraction of each edge's length,
    /// `d_i = |A_i − v| · offset / 100`. The cube's incident edges have length
    /// `2.0`, so `PERCENT(25)` lands at `2.0 · 25 / 100 = 0.5` — the same as
    /// `OFFSET(0.5)` — yet differs from `OFFSET(25)`.
    #[test]
    fn percent_slide_scales_by_edge_length() {
        let edge_length = 2.0_f32;
        let percent = 25.0_f32;
        let expected = edge_length * percent / 100.0;
        for slide in cube_corner_slides(percent, BevelOffsetType::Percent) {
            assert!(
                (slide - expected).abs() < 1.0e-5,
                "PERCENT slide {slide} should be {expected}",
            );
        }
        // Equivalence to OFFSET(0.5) is the length-scaling pin of §8.
        let offset_equivalent = cube_corner_slides(expected, BevelOffsetType::Offset);
        for slide in offset_equivalent {
            assert!((slide - expected).abs() < 1.0e-5);
        }
    }

    /// BEVEL.md §4.3: `ABSOLUTE` is a fixed slide equal to `offset`, numerically
    /// identical to `OFFSET` at the foundational no-weights level.
    #[test]
    fn absolute_slide_coincides_with_offset() {
        let offset = 0.3;
        let absolute = cube_corner_slides(offset, BevelOffsetType::Absolute);
        let offset_slides = cube_corner_slides(offset, BevelOffsetType::Offset);
        assert_eq!(absolute.len(), offset_slides.len());
        for slide in &absolute {
            assert!(
                (slide - offset).abs() < 1.0e-5,
                "ABSOLUTE slide {slide} should equal offset {offset}",
            );
        }
        for (a, o) in absolute.iter().zip(&offset_slides) {
            assert!((a - o).abs() < 1.0e-6, "ABSOLUTE coincides with OFFSET");
        }
    }
}

fn ring_vert_position(vertex: Vec3, far: Vec3, offset: f32) -> Vec3 {
    // §4: slide `offset` along the edge toward the far endpoint. §4.1: guard the
    // degenerate overshoot — a non-positive or near-zero edge length cannot host
    // a slide, so the ring vert stays at the original corner.
    let direction = far - vertex;
    let length = direction.length();
    if length <= f32::EPSILON {
        return vertex;
    }
    let clamped = offset.min(length);
    vertex + (direction / length) * clamped
}

/// Place one edge-bevel strip vert (BEVEL.md §7.3): slide `offset` from the
/// endpoint, perpendicular to the beveled edge and in the adjacent face's plane,
/// into the face interior. `face_normal` fixes the in-plane perpendicular
/// (`cross(direction, normal)`); `face_center` disambiguates its sign so the
/// vert moves toward the face's interior, away from the crease.
fn strip_vert_position(
    endpoint: Vec3,
    direction: Vec3,
    face_normal: Vec3,
    face_center: Vec3,
    offset: f32,
) -> Vec3 {
    let mut into_face = direction.cross(face_normal);
    let length = into_face.length();
    if length <= f32::EPSILON {
        return endpoint; // degenerate: edge parallel to the normal
    }
    into_face /= length;
    // Orient toward the face interior: the centroid lies on the interior side.
    if into_face.dot(face_center - endpoint) < 0.0 {
        into_face = -into_face;
    }
    endpoint + into_face * offset
}

/// The corner axis `Σ_i normalize(v − A_i)` (BEVEL.md §4.3): the un-normalised
/// sum of the inward edge directions. Only its direction matters — it is the
/// reference each edge's angle `φ_i` is measured against.
fn corner_axis(vertex: Vec3, far_positions: &[Vec3]) -> Vec3 {
    far_positions.iter().fold(Vec3::ZERO, |axis, &far| {
        let inward = vertex - far;
        let length = inward.length();
        if length <= f32::EPSILON {
            axis
        } else {
            axis + inward / length
        }
    })
}

/// The largest offset a single beveled edge can tolerate before its two ring
/// verts collide (BEVEL.md §4.4.2): `L / (k_a + k_b)`, where `k_a`, `k_b` are the
/// per-offset slide rates at the two ends (`k_b = 0` when the far end is not
/// beveled). When neither end slides, the edge never limits.
fn edge_collision_offset(length: f32, rate_near: f32, rate_far: f32) -> f32 {
    let rate_sum = rate_near + rate_far;
    if rate_sum <= 0.0 {
        f32::INFINITY
    } else {
        length / rate_sum
    }
}

/// The §4.4 global clamp factor: gather every beveled edge's collision offset,
/// take the smallest, and shrink everything by `min_collision / requested` if
/// that minimum falls below the requested offset (BEVEL.md §4.4.2). A shared
/// edge contributes both endpoints' slide rates; an edge beveled at one end only
/// contributes its near rate (far rate 0). Returns a factor in `(0, 1]`: every
/// beveled vertex's resolved slide is multiplied by it uniformly (§4.4.3).
fn clamp_factor(requested_offset: f32, beveled: &[VertexKey], frames: &[BevelFrame]) -> f32 {
    let mut min_collision = f32::INFINITY;
    for frame in frames {
        for placement in &frame.placements {
            let length = (placement.far_position - frame.vertex_position).length();
            let rate_near = placement.rate;
            // §4.4.2: a two-ended collision. If the far endpoint is also beveled,
            // look up *its* slide rate on this same edge so both ends' slides are
            // summed; otherwise the far rate is 0 and the edge collides only at
            // its own far endpoint.
            let rate_far = if beveled.contains(&placement.far_vertex) {
                far_end_rate(frames, placement.edge, frame.vertex)
            } else {
                0.0
            };
            let collision = edge_collision_offset(length, rate_near, rate_far);
            if collision < min_collision {
                min_collision = collision;
            }
        }
    }

    if min_collision < requested_offset && requested_offset > 0.0 {
        min_collision / requested_offset
    } else {
        1.0
    }
}

/// The slide rate at the far end of `edge`: the rate the beveled frame whose
/// vertex is *not* `near_vertex` assigns to `edge`. Returns 0 if no other frame
/// bevels `edge` (BEVEL.md §4.4.2).
fn far_end_rate(frames: &[BevelFrame], edge: EdgeKey, near_vertex: VertexKey) -> f32 {
    for frame in frames {
        if frame.vertex == near_vertex {
            continue;
        }
        for placement in &frame.placements {
            if placement.edge == edge {
                return placement.rate;
            }
        }
    }
    0.0
}

/// The smallest slide a near-degenerate angle mode is allowed to produce
/// (BEVEL.md §4.3): a hundredth of `offset`, the established fallback when the
/// `WIDTH` / `DEPTH` denominator underflows rather than dividing by ~0.
const DEGENERATE_SLIDE_FRACTION: f32 = 0.01;

/// Per-edge slide distance `d_i` for a single-segment vertex bevel (BEVEL.md
/// §4 / §4.3). Each `offset_type` interprets `offset` as a different physical
/// quantity and solves for the slide that realises it; the surgery downstream
/// is mode-agnostic.
fn slide_distance(
    offset: f32,
    offset_type: BevelOffsetType,
    vertex: Vec3,
    far: Vec3,
    corner_axis: Vec3,
) -> f32 {
    match offset_type {
        // §4 / §4.3: plain and fixed slide — `d_i = offset`. The two coincide at
        // the foundational no-weights level (§4.3).
        BevelOffsetType::Offset | BevelOffsetType::Absolute => offset,

        // §4.3: a per-edge fraction of that edge's own length, no angle term.
        BevelOffsetType::Percent => (far - vertex).length() * offset / 100.0,

        // §4.3 angle modes: re-measure against `φ_i`, the angle between the
        // edge's inward direction and the corner axis.
        BevelOffsetType::Width => {
            let phi = inward_axis_angle(vertex, far, corner_axis);
            let denominator = (2.0 * phi.sin()).abs();
            if denominator <= f32::EPSILON {
                offset * DEGENERATE_SLIDE_FRACTION
            } else {
                offset / denominator
            }
        }
        BevelOffsetType::Depth => {
            let phi = inward_axis_angle(vertex, far, corner_axis);
            let denominator = phi.cos().abs();
            if denominator <= f32::EPSILON {
                offset * DEGENERATE_SLIDE_FRACTION
            } else {
                offset / denominator
            }
        }
    }
}

/// `φ_i` — the angle in `0..π` between edge `e_i`'s inward direction
/// `normalize(v − A_i)` and the corner axis (BEVEL.md §4.3).
fn inward_axis_angle(vertex: Vec3, far: Vec3, corner_axis: Vec3) -> f32 {
    let inward = vertex - far;
    let inward_length = inward.length();
    let axis_length = corner_axis.length();
    if inward_length <= f32::EPSILON || axis_length <= f32::EPSILON {
        return 0.0;
    }
    let cosine = (inward / inward_length).dot(corner_axis / axis_length);
    cosine.clamp(-1.0, 1.0).acos()
}

/// Order the ring verts into a cycle by following the chamfer edges (BEVEL.md
/// §5.1 — the disk order). Starts from `ring_verts[0]` and walks the chamfer
/// adjacency; returns `None` if the chamfer edges do not form a single closed
/// cycle over all ring verts (an unexpected non-manifold corner).
fn order_ring_cycle(
    mesh: &LoopMesh,
    ring_verts: &[VertexKey],
    chamfer_edges: &[EdgeKey],
) -> Option<Vec<VertexKey>> {
    let mut neighbours: Vec<(VertexKey, VertexKey)> = Vec::with_capacity(chamfer_edges.len());
    for &edge in chamfer_edges {
        let [a, b] = mesh.edge_vert_keys(edge);
        neighbours.push((a, b));
    }

    let adjacent = |current: VertexKey, skip: VertexKey| -> Option<VertexKey> {
        for &(a, b) in &neighbours {
            if a == current && b != skip {
                return Some(b);
            }
            if b == current && a != skip {
                return Some(a);
            }
        }
        None
    };

    let start = ring_verts[0];
    let mut ordered = vec![start];
    // First step: pick either chamfer neighbour of `start`.
    let mut previous = start;
    let mut current = neighbours
        .iter()
        .find_map(|&(a, b)| {
            if a == start {
                Some(b)
            } else if b == start {
                Some(a)
            } else {
                None
            }
        })?;

    while current != start {
        ordered.push(current);
        let next = adjacent(current, previous)?;
        previous = current;
        current = next;
        if ordered.len() > ring_verts.len() {
            return None;
        }
    }

    if ordered.len() == ring_verts.len() {
        Some(ordered)
    } else {
        None
    }
}
