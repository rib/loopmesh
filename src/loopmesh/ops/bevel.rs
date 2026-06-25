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
    /// Material slot stamped on every face the bevel **creates** (BEVEL.md §6.6).
    /// A leaf, per-face parameter that changes no topology and no placement:
    ///
    /// - `material >= 0` — stamp that slot uniformly on every new bevel face
    ///   (cap, strip quad, junction patch).
    /// - `material < 0` (canonically `-1`) — each new face inherits the material
    ///   of a representative adjacent original face chosen per new face (the cap
    ///   from an incident face of `v`, a strip from a flanking face, a patch from
    ///   a junction-adjacent face, §6.6.1).
    ///
    /// Surviving re-stitched / re-pointed original faces are never re-materialised
    /// by either regime (§6.6.2). Defaults to `-1` (inherit).
    pub material: i32,
    /// Propagate the **seam** edge flag across a multi-edge junction's bridging
    /// segments (BEVEL.md §6.7 / §6.7.4). A leaf, flag-only parameter that changes
    /// no topology and no placement.
    ///
    /// It gates **only** mechanism (2) of §6.7.1: the contiguity-bridging of the
    /// seam across the junction's bridging segments when seamed and non-seamed
    /// beveled edges are interleaved around the same junction. It is **inert** for
    /// vertex bevel (§6.7.2) and isolated single-edge bevel (§6.7.3), where the
    /// flag rides only the unconditional §6.5-style per-edge attribute copy
    /// (mechanism (1)), corrected by the §6.7.1.1 corner-segment contiguity AND.
    /// Defaults to `false`.
    pub mark_seam: bool,
    /// The same propagation for the **sharp** (not-smooth) edge flag (BEVEL.md
    /// §6.7). Runs the identical junction bridging on the sharp bit, independently
    /// of `mark_seam`. Defaults to `false`.
    pub mark_sharp: bool,
}

impl Default for BevelOptions {
    fn default() -> Self {
        BevelOptions {
            offset: 0.0,
            offset_type: BevelOffsetType::Offset,
            segments: 1,
            affect: BevelAffect::Vertices,
            clamp_overlap: false,
            material: -1,
            mark_seam: false,
            mark_sharp: false,
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
    /// The material slot of the representative incident face used by the cap's
    /// `material < 0` inherit (BEVEL.md §6.6.1: the cap inherits an incident face
    /// of `v`). Captured against the pristine mesh in `place_ring_verts` before
    /// those faces are killed and rebuilt, so it survives the surgery.
    representative_material: i32,
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
            return self.bevel_edges(geom, &options);
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
                    // §6.6: the cap is a new face — stamp `material` (>= 0) or
                    // inherit the representative incident face's slot (< 0).
                    self.assign_bevel_face_material(
                        cap.face,
                        options.material,
                        corner.representative_material,
                    );
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
    /// Bevels each manifold edge in `geom`. An edge `E = {a, b}` is replaced by
    /// four strip verts (two per endpoint, slid perpendicular into each adjacent
    /// face, §7.3) and one strip quad `aL–bL–bR–aR` (§7.1). How each endpoint is
    /// closed depends on how many beveled edges meet there:
    ///
    /// - **Terminal endpoint** (`s == 1`, exactly one beveled edge): the two
    ///   strip verts are absorbed by the surviving valence-3 third face (quad →
    ///   pentagon, §7.4 / §7.5).
    /// - **Junction endpoint** (`s ≥ 2`, two or more beveled edges meeting): the
    ///   arriving strips share a ring of `s` **boundverts** (one per disk sector
    ///   between consecutive beveled edges, §7.6.1), each placed by the
    ///   constrained meet of the two adjacent strips' offset lines (§7.6.2). A
    ///   `s ≥ 3` junction caps the ring with one flat **patch face** (§7.6.3); a
    ///   `s == 2` "through" junction adds no patch face.
    ///
    /// Single segment / symmetric `offset` only. A non-manifold edge or an
    /// endpoint whose terminal close needs valence ≥ 4 (the on-edge-vert fill of
    /// §7.5) is skipped — those remain §7.5 / §8 stubs.
    fn bevel_edges(&mut self, geom: &[GeomKey], options: &BevelOptions) -> BevelResult {
        let offset = options.offset;
        let material = options.material;
        let mut result = BevelResult::default();

        let mut beveled: Vec<EdgeKey> = Vec::new();
        for entry in geom {
            if let GeomKey::Edge(edge) = *entry {
                if !beveled.contains(&edge) {
                    beveled.push(edge);
                }
            }
        }

        // Count how many beveled edges meet at each endpoint: `s == 1` is a
        // terminal close (§7.5), `s ≥ 2` a junction (§7.6).
        let mut beveled_at: Vec<(VertexKey, usize)> = Vec::new();
        for &edge in &beveled {
            for endpoint in self.edge_vert_keys(edge) {
                match beveled_at.iter_mut().find(|(v, _)| *v == endpoint) {
                    Some((_, count)) => *count += 1,
                    None => beveled_at.push((endpoint, 1)),
                }
            }
        }
        // The set of endpoints that are junctions (≥ 2 beveled edges meeting).
        let junctions: Vec<VertexKey> = beveled_at
            .iter()
            .filter(|(_, count)| *count >= 2)
            .map(|(v, _)| *v)
            .collect();

        if junctions.is_empty() {
            // No shared endpoint: every edge is isolated, every endpoint terminal
            // — the foundational §7 path, unchanged.
            for &edge in &beveled {
                self.bevel_single_edge(edge, offset, material, &mut result);
            }
            return result;
        }

        self.bevel_edges_with_junctions(&beveled, &junctions, options, &mut result);
        result
    }

    /// Bevel one isolated manifold edge (BEVEL.md §7). Places the four strip verts
    /// against the pristine local geometry (§7.5 placement-before-surgery), then
    /// re-stitches every incident face and builds the strip quad.
    fn bevel_single_edge(
        &mut self,
        edge: EdgeKey,
        offset: f32,
        material: i32,
        result: &mut BevelResult,
    ) {
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

        // §6.6.1: the strip quad's `-1` inherit source is a flanking face (fL/fR)
        // of the beveled edge. Capture it pristine before the surgery.
        let representative_material = self.face_material_index(face_left);

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
        let strip_a_left = self.create_vertex(strip_vert_position(
            self.vertex_position(a),
            direction,
            left_normal,
            left_center,
            offset,
        ));
        let strip_a_right = self.create_vertex(strip_vert_position(
            self.vertex_position(a),
            direction,
            right_normal,
            right_center,
            offset,
        ));
        let strip_b_left = self.create_vertex(strip_vert_position(
            self.vertex_position(b),
            direction,
            left_normal,
            left_center,
            offset,
        ));
        let strip_b_right = self.create_vertex(strip_vert_position(
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
        let strip = self.build_strip_quad(strip_a_left, strip_b_left, strip_b_right, strip_a_right);
        self.assign_bevel_face_material(strip.face, material, representative_material);
        result.faces.push(strip.face);
        result.edges.extend(strip.edges);

        // §7.1 item 5 / §6.3: both endpoints are now loose. Free them.
        self.free_loose_vertex(a);
        self.free_loose_vertex(b);
    }

    /// Free a beveled endpoint after re-stitch (BEVEL.md §5.1.1 / §6.3). The
    /// re-stitch rebuilds every incident face off the endpoint, leaving only
    /// **loose** incident edges (orphaned legs with no remaining loop) on its disk;
    /// this guards that no edge still on the disk carries a face before draining
    /// the disk and freeing the vertex. A surviving *surface* edge (one still on a
    /// face) means the re-stitch left the endpoint attached to live geometry — the
    /// invariant the old blind `vertex_kill` relied on, now checked rather than
    /// assumed.
    fn free_loose_vertex(&mut self, vertex: VertexKey) {
        for edge in self.iter_disk_edges(vertex).collect::<Vec<_>>() {
            assert!(
                self.edge_radial_iter(edge).next().is_none(),
                "free_loose_vertex: endpoint {vertex:?} still on a face via edge {edge:?}",
            );
        }
        self.vertex_kill(vertex);
    }

    /// Edge bevel where two or more beveled edges meet at one or more **junction**
    /// vertices (BEVEL.md §7.6). Places every near-corner vert (boundverts at
    /// junctions, strip verts at terminal endpoints) and far-corner strip vert
    /// against the *pristine* mesh (§7.5 placement-before-surgery), then re-stitches
    /// every incident face, builds the per-edge strip quads and the per-junction
    /// patch faces, and frees the consumed original endpoints.
    fn bevel_edges_with_junctions(
        &mut self,
        beveled: &[EdgeKey],
        junctions: &[VertexKey],
        options: &BevelOptions,
        result: &mut BevelResult,
    ) {
        let offset = options.offset;
        let material = options.material;
        // Drop any non-manifold beveled edge (§2.1): a junction strip needs each
        // arriving edge to carry exactly two faces.
        let mut beveled_manifold: Vec<EdgeKey> = Vec::new();
        for &edge in beveled {
            if self.edge_radial_iter(edge).count() == 2 {
                beveled_manifold.push(edge);
            }
        }
        if beveled_manifold.is_empty() {
            return;
        }

        // §6.6.1: capture the `-1` inherit source for each new face against the
        // pristine mesh, before re-stitch kills the adjacent originals. A strip's
        // representative is one of its beveled edge's two flanking faces; a
        // junction patch's representative is a face incident to that junction
        // vertex. Captured now, applied to the new faces built in phases 3/4.
        let strip_material: Vec<(EdgeKey, i32)> = beveled_manifold
            .iter()
            .map(|&edge| {
                let representative = self
                    .edge_loop_keys(edge)
                    .first()
                    .map(|&l| self.face_material_index(self.loop_face(l)))
                    .unwrap_or(0);
                (edge, representative)
            })
            .collect();
        // §6.7.4: capture each beveled edge's seam / sharp state against the
        // pristine mesh, keyed by edge, for the junction bridging pass. The
        // beveled edges survive (re-pointed) so their flags persist, but a pristine
        // snapshot keeps the bridging decision independent of surgery order.
        let edge_seam_state: Vec<(EdgeKey, bool, bool)> = beveled_manifold
            .iter()
            .map(|&edge| (edge, self.edge_seam(edge), self.edge_sharp(edge)))
            .collect();
        let junction_material: Vec<(VertexKey, i32)> = junctions
            .iter()
            .map(|&vertex| {
                let representative = self
                    .vert_face_keys(vertex)
                    .first()
                    .map(|&face| self.face_material_index(face))
                    .unwrap_or(0);
                (vertex, representative)
            })
            .collect();

        let is_beveled = |edge: EdgeKey| beveled_manifold.contains(&edge);

        // The original endpoints to consume: every endpoint of a beveled edge.
        let mut endpoints: Vec<VertexKey> = Vec::new();
        for &edge in &beveled_manifold {
            for endpoint in self.edge_vert_keys(edge) {
                if !endpoints.contains(&endpoint) {
                    endpoints.push(endpoint);
                }
            }
        }

        // Phase 1 (pristine mesh): for every (beveled edge, endpoint, adjacent
        // face) corner, decide which *near-corner vertex* the strip lands on. At a
        // terminal endpoint each face-side gets its own fresh strip vert; at a
        // junction the two strip verts flanking a shared face collapse to that
        // sector's single boundvert (§7.6.1). `corner_vert` maps
        // (edge, endpoint, face) → the vertex the strip corner uses.
        let mut corner_vert: Vec<CornerVert> = Vec::new();
        let mut substitutions: Vec<FaceSubstitution> = Vec::new();
        let mut junction_patches: Vec<JunctionPatch> = Vec::new();

        for &endpoint in &endpoints {
            if junctions.contains(&endpoint) {
                self.place_junction_boundverts(
                    endpoint,
                    &is_beveled,
                    offset,
                    &mut corner_vert,
                    &mut substitutions,
                    &mut junction_patches,
                    result,
                );
            } else {
                self.place_terminal_strip_verts(
                    endpoint,
                    &beveled_manifold,
                    offset,
                    &mut corner_vert,
                    &mut substitutions,
                    result,
                );
            }
        }

        // Phase 2: re-stitch every incident face, swapping each consumed endpoint
        // corner for the ordered near-corner vert(s) recorded for that (vertex,
        // face) corner (§7.4 / §7.6.1). A terminal third face swaps its corner for
        // two strip verts (1-for-2); a strip's radial face and every junction
        // sector face swap for one (1-for-1).
        let mut incident_faces: Vec<FaceKey> = Vec::new();
        for &endpoint in &endpoints {
            for face in self.vert_face_keys(endpoint) {
                if !incident_faces.contains(&face) {
                    incident_faces.push(face);
                }
            }
        }
        for face in incident_faces {
            self.restitch_junction_face(face, &endpoints, &substitutions);
        }

        // Phase 3: one strip quad per beveled edge over its four near/far corner
        // verts (§7.1 item 2). §6.6: stamp / inherit each new strip's material.
        for &edge in &beveled_manifold {
            let representative = strip_material
                .iter()
                .find_map(|&(e, m)| (e == edge).then_some(m))
                .unwrap_or(0);
            self.build_junction_strip(edge, &corner_vert, material, representative, result);
        }

        // Phase 4: cap each `s ≥ 3` junction with one flat patch face over its
        // boundvert ring (§7.6.3). `s == 2` adds no patch. §6.6: stamp / inherit
        // each new patch's material from a junction-adjacent original face.
        for patch in &junction_patches {
            if patch.boundverts.len() >= 3 {
                let representative = junction_material
                    .iter()
                    .find_map(|&(v, m)| (v == patch.vertex).then_some(m))
                    .unwrap_or(0);
                self.build_patch_face(&patch.boundverts, material, representative, result);
            }
        }

        // Phase 4.25 (§6.7.1 mechanism (1), UNCONDITIONAL — not gated by the
        // parameters): each beveled edge's two strip long edges descend from that
        // edge's re-stitched radial faces, so they inherit its seam / sharp exactly
        // as a re-pointed survivor would (§6.7.1 item 1 / §6.7.3). A seam on an
        // arriving beveled edge therefore reappears on the strip edges descending
        // from it whether or not `mark_seam` is set. The perpendicular near edges
        // (the patch ring segments) are *not* descendants of any single edge — they
        // are the parameter-gated bridging segments of phase 4.5 below.
        self.inherit_strip_long_edge_flags(&beveled_manifold, &corner_vert, &edge_seam_state);

        // Phase 4.5 (§6.7.4): the `mark_seam` / `mark_sharp` junction bridging.
        // Mechanism (2): bridge the seam / sharp flag across each junction's patch
        // ring segments so the flag runs continuously around the junction, joining
        // the strip long edges of consecutive seamed beveled edges.
        if options.mark_seam || options.mark_sharp {
            for patch in &junction_patches {
                self.bridge_junction_flags(
                    patch,
                    &edge_seam_state,
                    options.mark_seam,
                    options.mark_sharp,
                );
            }
        }

        // Phase 5: every consumed endpoint is now loose. Free each (§5.1.1).
        for &endpoint in &endpoints {
            self.free_loose_vertex(endpoint);
        }
    }

    /// Place the boundverts of one junction vertex (BEVEL.md §7.6.1 / §7.6.2).
    ///
    /// Walks the faces around `vertex` to recover the rotational sector structure,
    /// groups them into sectors delimited by consecutive beveled edges, places one
    /// boundvert per sector at the constrained meet of the two flanking strips'
    /// offset lines, and records (edge, vertex, face) → boundvert for every corner
    /// in the sector. The ordered boundvert ring is stored for the patch face.
    fn place_junction_boundverts(
        &mut self,
        vertex: VertexKey,
        is_beveled: &dyn Fn(EdgeKey) -> bool,
        offset: f32,
        corner_vert: &mut Vec<CornerVert>,
        substitutions: &mut Vec<FaceSubstitution>,
        junction_patches: &mut Vec<JunctionPatch>,
        result: &mut BevelResult,
    ) {
        // Rotational corner ring: each incident face contributes the pair of its
        // two edges meeting at `vertex`. Chain them edge-to-edge into the disk
        // walk order around the vertex.
        let corners = self.vertex_corner_ring(vertex);

        // Split the ring into sectors at beveled edges. A sector runs from one
        // beveled edge, through any unbeveled edges and their faces, to the next
        // beveled edge. Each sector yields one boundvert (§7.6.1).
        let sectors = sector_split(&corners, is_beveled);

        let vertex_position = self.vertex_position(vertex);
        let mut boundverts: Vec<VertexKey> = Vec::with_capacity(sectors.len());
        let mut opening_edges: Vec<EdgeKey> = Vec::with_capacity(sectors.len());

        for sector in &sectors {
            let boundvert_position = self.solve_meet_point(vertex, vertex_position, sector, offset);
            let boundvert = self.create_vertex(boundvert_position);
            result.verts.push(boundvert);
            boundverts.push(boundvert);
            // `bounding_edges[0]` is the beveled edge that opens this sector
            // (sector_split records the opening edge first) — `g_k` for `B_k`.
            opening_edges.push(sector.bounding_edges[0]);

            // Every face in this sector terminates on this single boundvert: the
            // two bounding beveled edges' strips and any absorbed unbeveled leg's
            // faces all join here (§7.6.1). Record (vertex, face) → boundvert for
            // each distinct face so the re-stitch collapses each sector corner to
            // one vert, and keep the per-edge record for the strip builder.
            for &(edge, face) in &sector.corners {
                set_face_substitution(substitutions, vertex, face, vec![boundvert]);
                if is_beveled(edge) {
                    corner_vert.push(CornerVert {
                        edge,
                        vertex,
                        face,
                        near_vert: boundvert,
                    });
                }
            }
        }

        junction_patches.push(JunctionPatch {
            vertex,
            boundverts,
            opening_edges,
        });
    }

    /// Place the two strip verts of a terminal endpoint (BEVEL.md §7.3 / §7.4 /
    /// §7.5): one per adjacent face of the single beveled edge there, slid
    /// perpendicular into the face. Each strip's radial face swaps its corner for
    /// the one strip vert on its side (1-for-1); the surviving **third** face(s)
    /// between the unbeveled legs swap their corner for *both* strip verts ordered
    /// so the strip vert sharing the incoming leg stays adjacent to `prev`
    /// (1-for-2, the absorbed valence-3 corner of §7.4 / §7.5).
    fn place_terminal_strip_verts(
        &mut self,
        vertex: VertexKey,
        beveled: &[EdgeKey],
        offset: f32,
        corner_vert: &mut Vec<CornerVert>,
        substitutions: &mut Vec<FaceSubstitution>,
        result: &mut BevelResult,
    ) {
        // The single beveled edge at this terminal endpoint.
        let edge = self
            .iter_disk_edges(vertex)
            .find(|&e| beveled.contains(&e))
            .expect("a terminal endpoint has its one beveled edge");

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

        let [a, b] = self.edge_vert_keys(edge);
        let far = if a == vertex { b } else { a };
        let direction = {
            let segment = self.vertex_position(far) - self.vertex_position(vertex);
            let length = segment.length();
            if length <= f32::EPSILON {
                return;
            }
            segment / length
        };

        // §7.5 foundational pin: only the valence-3 terminal close is implemented
        // — the surviving third face absorbs the corner. Valence ≥ 4 needs the
        // on-edge-vert fill (§7.5), still a stub; skip it.
        if self.iter_disk_edges(vertex).count() != 3 {
            return;
        }

        let strip_for_face = |this: &mut Self, face: FaceKey, result: &mut BevelResult| {
            let normal = this.face_plane_normal(face);
            let center = this.face_center_of_mass(face);
            let position = strip_vert_position(
                this.vertex_position(vertex),
                direction,
                normal,
                center,
                offset,
            );
            let strip_vert = this.create_vertex(position);
            result.verts.push(strip_vert);
            strip_vert
        };

        let strip_left = strip_for_face(self, face_left, result);
        let strip_right = strip_for_face(self, face_right, result);

        // 1-for-1 on each radial face, plus the per-edge record for the strip.
        for &(face, strip_vert) in &[(face_left, strip_left), (face_right, strip_right)] {
            set_face_substitution(substitutions, vertex, face, vec![strip_vert]);
            corner_vert.push(CornerVert {
                edge,
                vertex,
                face,
                near_vert: strip_vert,
            });
        }

        // The third face(s): each swaps the corner for both strip verts, ordered so
        // the one sharing the incoming leg sits next to `prev` (§7.4).
        let left_other = self.face_other_edge_at(face_left, vertex, edge);
        let right_other = self.face_other_edge_at(face_right, vertex, edge);
        for face in self.vert_face_keys(vertex) {
            if face == face_left || face == face_right {
                continue;
            }
            let ordered = self.order_third_face_strip(
                face,
                vertex,
                left_other,
                right_other,
                strip_left,
                strip_right,
            );
            set_face_substitution(substitutions, vertex, face, ordered);
        }
    }

    /// Order a terminal third face's two strip verts so the one sharing the
    /// face's incoming leg at `vertex` comes first (BEVEL.md §7.4). The third face
    /// shares the unbeveled leg `left_other` with the strip's left face and
    /// `right_other` with its right face.
    fn order_third_face_strip(
        &self,
        face: FaceKey,
        vertex: VertexKey,
        left_other: EdgeKey,
        right_other: EdgeKey,
        strip_left: VertexKey,
        strip_right: VertexKey,
    ) -> Vec<VertexKey> {
        let cycle_verts = self.face_vert_keys(face);
        let cycle_length = cycle_verts.len();
        let index = cycle_verts
            .iter()
            .position(|&w| w == vertex)
            .expect("third face has the endpoint as a corner");
        let prev_vertex = cycle_verts[(index + cycle_length - 1) % cycle_length];
        let incoming_edge = self
            .query_edge_exists(prev_vertex, vertex)
            .expect("third face's incoming corner edge exists");
        if incoming_edge == left_other {
            vec![strip_left, strip_right]
        } else if incoming_edge == right_other {
            vec![strip_right, strip_left]
        } else {
            vec![strip_left, strip_right]
        }
    }

    /// The rotational ring of (edge-pair, face) corners around `vertex`: each
    /// incident face contributes its two edges meeting at `vertex`. Returned as a
    /// cyclic sequence of `VertexCorner { incoming_edge, face, outgoing_edge }`
    /// chained so consecutive corners share an edge — the disk walk order.
    fn vertex_corner_ring(&self, vertex: VertexKey) -> Vec<VertexCorner> {
        // Per incident face: its two edges at `vertex`.
        let mut face_edges: Vec<(FaceKey, EdgeKey, EdgeKey)> = Vec::new();
        for face in self.vert_face_keys(vertex) {
            let mut at_vertex: Vec<EdgeKey> = Vec::new();
            for loop_key in self.face_loop_keys(face) {
                let edge = self.loop_edge(loop_key);
                let [p, q] = self.edge_vert_keys(edge);
                if (p == vertex || q == vertex) && !at_vertex.contains(&edge) {
                    at_vertex.push(edge);
                }
            }
            if at_vertex.len() == 2 {
                face_edges.push((face, at_vertex[0], at_vertex[1]));
            }
        }

        // Chain faces edge-to-edge. Start anywhere; follow the shared edge into
        // the next face until the ring closes.
        let mut ring: Vec<VertexCorner> = Vec::with_capacity(face_edges.len());
        if face_edges.is_empty() {
            return ring;
        }

        let mut used = vec![false; face_edges.len()];
        let (start_face, start_e0, start_e1) = face_edges[0];
        used[0] = true;
        ring.push(VertexCorner {
            incoming_edge: start_e0,
            face: start_face,
            outgoing_edge: start_e1,
        });
        let mut frontier_edge = start_e1;

        while ring.len() < face_edges.len() {
            let next = face_edges.iter().enumerate().find(|(index, (_, e0, e1))| {
                !used[*index] && (*e0 == frontier_edge || *e1 == frontier_edge)
            });
            match next {
                Some((index, &(face, e0, e1))) => {
                    used[index] = true;
                    let outgoing = if e0 == frontier_edge { e1 } else { e0 };
                    ring.push(VertexCorner {
                        incoming_edge: frontier_edge,
                        face,
                        outgoing_edge: outgoing,
                    });
                    frontier_edge = outgoing;
                }
                None => break, // open fan or non-manifold — stop chaining
            }
        }

        ring
    }

    /// Solve a boundvert's constrained meet point (BEVEL.md §7.6.2). Assembles the
    /// offset-line rows (one per face in the sector, "stay at perpendicular
    /// `offset` from my bounding beveled edge in my plane") plus, for a single
    /// planar face, an embedding row (membership in the face plane, zero
    /// out-of-plane rise), and solves the normal equations `AᵀA·x = Aᵀc`. The
    /// least-squares form subsumes the exact 2-line intersection of the planar
    /// case.
    fn solve_meet_point(
        &self,
        vertex: VertexKey,
        vertex_position: Vec3,
        sector: &Sector,
        offset: f32,
    ) -> Vec3 {
        // Each face in the sector that is bounded by a beveled edge contributes an
        // offset row: the unit in-plane perpendicular to that edge, pointing into
        // the face, with target `offset` (measured relative to `vertex`).
        let mut rows: Vec<(Vec3, f32)> = Vec::new();
        let mut sector_normal = Vec3::ZERO;

        for &(edge, face) in &sector.corners {
            if !sector.bounding_edges.contains(&edge) {
                continue;
            }
            let [a, b] = self.edge_vert_keys(edge);
            let far = if a == vertex { b } else { a };
            let direction = {
                let segment = self.vertex_position(far) - vertex_position;
                let length = segment.length();
                if length <= f32::EPSILON {
                    continue;
                }
                segment / length
            };
            let normal = self.face_plane_normal(face);
            let center = self.face_center_of_mass(face);
            let mut into_face = direction.cross(normal);
            let into_len = into_face.length();
            if into_len <= f32::EPSILON {
                continue;
            }
            into_face /= into_len;
            if into_face.dot(center - vertex_position) < 0.0 {
                into_face = -into_face;
            }
            rows.push((into_face, offset));
            sector_normal += normal;
        }

        // Embedding row for a planar (single-face) sector: pin membership in the
        // face plane (no out-of-plane rise for a flat chamfer). Without it two
        // coplanar offset rows leave the meet free along a whole line (§7.6.2).
        if sector.corners.len() == 1 && sector_normal.length() > f32::EPSILON {
            rows.push((sector_normal.normalize(), 0.0));
        }

        solve_normal_equations(&rows, vertex_position).unwrap_or(vertex_position)
    }

    /// Re-stitch one face incident to a consumed endpoint (BEVEL.md §7.4 /
    /// §7.6.1). Each consumed-endpoint corner of `face` is replaced by the ordered
    /// near-corner vert(s) recorded for that (vertex, face) corner: one vert for a
    /// strip's radial face or a junction sector face (1-for-1), two for a terminal
    /// third face (1-for-2). A corner with no recorded substitution is left as-is.
    fn restitch_junction_face(
        &mut self,
        face: FaceKey,
        endpoints: &[VertexKey],
        substitutions: &[FaceSubstitution],
    ) {
        let cycle_verts = self.face_vert_keys(face);

        let mut new_cycle: Vec<VertexKey> = Vec::with_capacity(cycle_verts.len() + 1);
        let mut changed = false;
        for &w in &cycle_verts {
            if !endpoints.contains(&w) {
                new_cycle.push(w);
                continue;
            }
            match face_substitution(substitutions, w, face) {
                Some(replacement) => {
                    new_cycle.extend_from_slice(replacement);
                    changed = true;
                }
                None => new_cycle.push(w),
            }
        }

        if !changed {
            return;
        }

        self.face_kill(face);
        let new_face = self.face_create_verts(&new_cycle, FaceCreateFlags::NONE);
        self.face_attribs
            .copy_into(new_face.as_usize(), face.as_usize());
    }

    /// Build one beveled edge's strip quad over its four near/far corner verts
    /// (BEVEL.md §7.1 item 2). The four corners are the edge's two near corners
    /// (one per adjacent face, at each endpoint) — at a junction these are shared
    /// boundverts, at a terminal they are the endpoint's fresh strip verts. Wound
    /// opposite the left face's traversal so each strip edge stays radial-2 (§5.1).
    fn build_junction_strip(
        &mut self,
        edge: EdgeKey,
        corner_vert: &[CornerVert],
        material: i32,
        representative: i32,
        result: &mut BevelResult,
    ) {
        let [a, b] = self.edge_vert_keys(edge);

        // The edge's two radial faces give its left/right sides. The strip quad is
        // a-left, b-left, b-right, a-right.
        let radial_faces: Vec<FaceKey> = {
            let mut faces = Vec::new();
            for record in corner_vert {
                if record.edge == edge && !faces.contains(&record.face) {
                    faces.push(record.face);
                }
            }
            faces
        };
        if radial_faces.len() != 2 {
            return; // a non-manifold edge produced no strip
        }
        let face_left = radial_faces[0];
        let face_right = radial_faces[1];

        let a_left = near_corner(corner_vert, edge, a, face_left);
        let a_right = near_corner(corner_vert, edge, a, face_right);
        let b_left = near_corner(corner_vert, edge, b, face_left);
        let b_right = near_corner(corner_vert, edge, b, face_right);

        // A through-junction (s == 2) collapses an edge's two near corners on a
        // side into one boundvert; the strip quad can then degenerate. Skip a strip
        // whose four corners are not four distinct verts in a proper ring.
        let ring = [a_left, b_left, b_right, a_right];
        if ring_has_repeat(&ring) {
            // The strip ran into a coincident boundvert pair; emit a triangle by
            // dropping the duplicate so the strip still closes (§7.6.3 seam).
            self.build_collapsed_strip(&ring, material, representative, result);
            return;
        }

        // Adopt the left face's near long edge orientation: the re-stitched left
        // face traverses a_left–b_left, so wind the quad the opposite way.
        let mut quad = ring.to_vec();
        if let Some(long_edge) = self.query_edge_exists(a_left, b_left) {
            let face_runs_forward = self
                .edge_loop_keys(long_edge)
                .first()
                .map(|&l| {
                    let start = self.loop_vertex(l);
                    let end = self.loop_vertex(self.loop_next(l));
                    start == a_left && end == b_left
                })
                .unwrap_or(false);
            if face_runs_forward {
                quad.reverse();
            }
        }

        let face = self.face_create_verts(&quad, FaceCreateFlags::NONE);
        self.assign_bevel_face_material(face, material, representative);
        result.faces.push(face);
        for i in 0..quad.len() {
            let p = quad[i];
            let q = quad[(i + 1) % quad.len()];
            if let Some(strip_edge) = self.query_edge_exists(p, q) {
                if !result.edges.contains(&strip_edge) {
                    result.edges.push(strip_edge);
                }
            }
        }
    }

    /// Build a degenerate strip whose ring has a repeated vert (a through-junction
    /// seam where two near corners coincide, §7.6.3): drop the duplicate and build
    /// the surviving triangle, or skip entirely if fewer than three distinct verts.
    fn build_collapsed_strip(
        &mut self,
        ring: &[VertexKey; 4],
        material: i32,
        representative: i32,
        result: &mut BevelResult,
    ) {
        let mut distinct: Vec<VertexKey> = Vec::with_capacity(4);
        for &v in ring {
            if !distinct.contains(&v) {
                distinct.push(v);
            }
        }
        if distinct.len() < 3 {
            return;
        }
        let face = self.face_create_verts(&distinct, FaceCreateFlags::NONE);
        self.assign_bevel_face_material(face, material, representative);
        result.faces.push(face);
        for i in 0..distinct.len() {
            let p = distinct[i];
            let q = distinct[(i + 1) % distinct.len()];
            if let Some(edge) = self.query_edge_exists(p, q) {
                if !result.edges.contains(&edge) {
                    result.edges.push(edge);
                }
            }
        }
    }

    /// Cap a junction's boundvert ring with one flat patch face (BEVEL.md §7.6.3).
    /// The patch adopts the strip edges already minted between consecutive
    /// boundverts (`B_(k-1)`–`B_k` is the near edge of edge `g_k`'s strip), wound
    /// opposite the arriving strips so each patch edge stays radial-2.
    fn build_patch_face(
        &mut self,
        boundverts: &[VertexKey],
        material: i32,
        representative: i32,
        result: &mut BevelResult,
    ) {
        // The patch boundary is the boundvert ring, already in sector order. Wind
        // it opposite the strip face traversal on the first shared edge.
        let first_a = boundverts[0];
        let first_b = boundverts[1];
        let mut ring = boundverts.to_vec();
        if let Some(shared_edge) = self.query_edge_exists(first_a, first_b) {
            let strip_runs_forward = self
                .edge_loop_keys(shared_edge)
                .first()
                .map(|&l| {
                    let start = self.loop_vertex(l);
                    let end = self.loop_vertex(self.loop_next(l));
                    start == first_a && end == first_b
                })
                .unwrap_or(false);
            if strip_runs_forward {
                ring.reverse();
            }
        }

        let face = self.face_create_verts(&ring, FaceCreateFlags::NONE);
        self.assign_bevel_face_material(face, material, representative);
        result.faces.push(face);
        for i in 0..ring.len() {
            let p = ring[i];
            let q = ring[(i + 1) % ring.len()];
            if let Some(edge) = self.query_edge_exists(p, q) {
                if !result.edges.contains(&edge) {
                    result.edges.push(edge);
                }
            }
        }
    }

    /// Inherit each beveled edge's seam / sharp onto its two strip long edges
    /// (BEVEL.md §6.7.1 mechanism (1), unconditional).
    ///
    /// A beveled edge `g = (a, b)` sheds a strip running *along* `g`. The two edges
    /// of that strip parallel to `g` — one in each of `g`'s two re-stitched radial
    /// faces, between the near corner at `a` and the near corner at `b` — are the
    /// edges that descend from `g`'s re-stitched faces (the new `g`-side of each
    /// face, §7.1 / §7.6.1). They inherit `g`'s per-edge attribute block, seam and
    /// sharp bits included, exactly as a re-pointed survivor keeps its own
    /// (§6.7.1 item 1). The perpendicular strip rails — the patch ring segments —
    /// are *not* descendants of `g` and are left to the parameter-gated bridging.
    ///
    /// Unconditional: runs whether or not `mark_seam` / `mark_sharp` is set; the
    /// flag is read from the *pristine* `edge_seam_state` snapshot.
    fn inherit_strip_long_edge_flags(
        &mut self,
        beveled: &[EdgeKey],
        corner_vert: &[CornerVert],
        edge_seam_state: &[(EdgeKey, bool, bool)],
    ) {
        for &edge in beveled {
            let Some(&(_, seam, sharp)) = edge_seam_state.iter().find(|&&(e, ..)| e == edge) else {
                continue;
            };
            if !seam && !sharp {
                continue;
            }
            let [a, b] = self.edge_vert_keys(edge);
            // One long edge per radial face of the strip: the near corner at `a` to
            // the near corner at `b` on that face.
            let mut faces: Vec<FaceKey> = Vec::new();
            for record in corner_vert {
                if record.edge == edge && !faces.contains(&record.face) {
                    faces.push(record.face);
                }
            }
            for face in faces {
                let near_a = near_corner(corner_vert, edge, a, face);
                let near_b = near_corner(corner_vert, edge, b, face);
                if near_a == near_b {
                    continue; // a collapsed through-junction side, no long edge
                }
                if let Some(long_edge) = self.query_edge_exists(near_a, near_b) {
                    if seam {
                        self.set_edge_seam(long_edge, true);
                    }
                    if sharp {
                        self.set_edge_sharp(long_edge, true);
                    }
                }
            }
        }
    }

    /// Bridge the seam / sharp flag across one junction's patch ring segments
    /// (BEVEL.md §6.7.4 — mechanism (2), gated by `mark_seam` / `mark_sharp`).
    ///
    /// The junction's boundverts `B_0, …, B_(s-1)` are in disk-sector order, with
    /// `opening_edges[k]` the beveled edge `g_k` opening sector `k` (boundvert
    /// `B_k` lives in that sector). The patch ring segment between consecutive
    /// boundverts `(B_(m-1), B_m)` is crossed by the beveled edge `g_m` that
    /// separates the two adjacent sectors. So each patch segment maps one-to-one to
    /// a beveled edge it crosses — this `segment ↔ crossing-edge` bijection is the
    /// ordering-robust frame the bridging rides.
    ///
    /// The rule flags the **single direct patch segment spanning each beveled edge
    /// that lies in a *gap*** — a maximal run of consecutive *unseamed* beveled
    /// edges flanked on **both** sides by a seamed beveled edge — and flags nothing
    /// else. It does **not** route the seam the long way around the ring, and a
    /// seamed run with no enclosing gap (a lone seamed edge, or all edges seamed)
    /// bridges nothing. Two opposite seamed edges at a ≥4-edge junction enclose two
    /// separate length-1 gaps and so bridge both; two adjacent seamed edges share a
    /// length-0 gap (no bridge) and bridge only the far length-2 gap.
    ///
    /// `seam` and `sharp` run the identical walk on independent bits.
    fn bridge_junction_flags(
        &mut self,
        patch: &JunctionPatch,
        edge_seam_state: &[(EdgeKey, bool, bool)],
        seam: bool,
        sharp: bool,
    ) {
        let count = patch.boundverts.len();
        if count < 2 {
            return;
        }

        let flag_of = |edge: EdgeKey, want_seam: bool| -> bool {
            edge_seam_state
                .iter()
                .find_map(|&(e, s, h)| (e == edge).then_some(if want_seam { s } else { h }))
                .unwrap_or(false)
        };

        // `segment_edge(m)` returns the patch segment `(B_(m-1), B_m)`, crossed by
        // the beveled edge `g_m = opening_edges[m]`.
        let segment_edge = |m: usize, this: &Self| -> Option<EdgeKey> {
            let a = patch.boundverts[(m + count - 1) % count];
            let b = patch.boundverts[m];
            this.query_edge_exists(a, b)
        };

        let mut bridge = |want_seam: bool, this: &mut Self| {
            let seamed = |k: usize| flag_of(patch.opening_edges[k], want_seam);

            let seamed_count = (0..count).filter(|&k| seamed(k)).count();
            // No gap exists unless seamed and unseamed beveled edges interleave: a
            // gap needs a seamed edge on each side, so at least two seamed edges and
            // at least one unseamed edge.
            if seamed_count < 2 || seamed_count == count {
                return;
            }

            // With at least one seamed edge on the cyclic disk, every maximal
            // unseamed run is flanked by a seamed edge on both sides — so every
            // unseamed beveled edge is a gap edge. Flag the single direct patch
            // segment spanning each.
            for m in 0..count {
                if seamed(m) {
                    continue;
                }
                if let Some(segment) = segment_edge(m, this) {
                    if want_seam {
                        this.set_edge_seam(segment, true);
                    } else {
                        this.set_edge_sharp(segment, true);
                    }
                }
            }
        };

        if seam {
            bridge(true, self);
        }
        if sharp {
            bridge(false, self);
        }
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
                new_cycle.push(if w == sub.a {
                    sub.strip_a_left
                } else {
                    sub.strip_b_left
                });
            } else if face == sub.face_right {
                new_cycle.push(if w == sub.a {
                    sub.strip_a_right
                } else {
                    sub.strip_b_right
                });
            } else {
                // Third face: replace the corner with both strip verts, ordered so
                // the one sharing the incoming edge sits next to `prev`.
                let prev_vertex = cycle_verts[(index + cycle_length - 1) % cycle_length];
                let incoming_edge = self
                    .query_edge_exists(prev_vertex, w)
                    .expect("third face's incoming corner edge exists");
                let (left_other, right_other, strip_left, strip_right) = if w == sub.a {
                    (
                        sub.left_other_a,
                        sub.right_other_a,
                        sub.strip_a_left,
                        sub.strip_a_right,
                    )
                } else {
                    (
                        sub.left_other_b,
                        sub.right_other_b,
                        sub.strip_b_left,
                        sub.strip_b_right,
                    )
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
        self.face_attribs
            .copy_into(new_face.as_usize(), face.as_usize());
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
                self.query_edge_exists(p, q)
                    .expect("strip quad edge exists")
            })
            .collect();
        edges.sort_by_key(|e| e.as_usize());
        edges.dedup();

        StripQuad { face, edges }
    }

    /// The edge of `face` incident to `vertex` that is **not** `exclude` (BEVEL.md
    /// §7.4): a face corner sits on exactly two of its edges, so excluding the
    /// beveled edge leaves the single other one.
    fn face_other_edge_at(&self, face: FaceKey, vertex: VertexKey, exclude: EdgeKey) -> EdgeKey {
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
                    if a == vertex {
                        b
                    } else {
                        a
                    }
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
            let incident_edges: Vec<EdgeKey> = frame.placements.iter().map(|p| p.edge).collect();
            let incident_faces = self.vert_face_keys(frame.vertex);
            let interior_fan = self.is_interior_fan(frame.vertex, &incident_edges, &incident_faces);

            // §6.6.1: the cap's `-1` inherit source is a representative incident
            // face of `v`. Capture its material slot now, against the pristine
            // mesh, before the incident faces are killed / rebuilt.
            let representative_material = incident_faces
                .first()
                .map(|&face| self.face_material_index(face))
                .unwrap_or(0);

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
                representative_material,
            });
        }
        corners
    }

    /// Phase 2 (BEVEL.md §6.1 / §6.4.3): re-stitch every face incident to any
    /// beveled vertex, swapping *all* of that face's beveled corners at once.
    /// Each beveled corner `v` becomes the pair (ring vert of `v`'s incoming
    /// edge, ring vert of `v`'s outgoing edge), keeping the face winding; the
    /// only freshly minted edge is the chamfer edge between the pair.
    ///
    /// **Every incident edge of a beveled corner survives in place.** §6.1 says a
    /// re-stitched face's two corner edges become `prev — r_i` (the original
    /// incoming edge `e_i`) and `r_(i+1) — next` (the original outgoing edge
    /// `e_(i+1)`) — *the same edges re-pointed*, not new ones — and §6.4.3 states
    /// the general preservation principle: a surviving edge "keeps the original
    /// edge's identity (and any per-edge attributes), simply re-pointed". So each
    /// of a corner `v`'s incident edges `e` has its `v`-endpoint re-pointed to
    /// `e`'s ring vert for `v`. An edge with **one** beveled endpoint is swapped
    /// once (`v → r_v`); a **shared** edge with both endpoints beveled is swapped
    /// once from each corner (`a → r_a`, then `b → r_b`), so the both-beveled case
    /// is just the general re-point applied twice. That up-front pass
    /// (`repoint_incident_edges`) uses the in-place re-point primitive (no
    /// kill/recreate, so each edge keeps the **same `EdgeKey` and `edge_attribs`
    /// slot**). The subsequent face rebuild then *adopts* every re-pointed edge
    /// (the face builder reuses an existing edge rather than minting a duplicate,
    /// BEVEL.md §6.1 / §6.4.3) and mints only the chamfer edges fresh. An interior
    /// shared edge ends with radial count 2 — both straddling faces rebuilt onto
    /// the same surviving edge.
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

        // Compute every face's rebuilt vertex cycle against the *current* mesh,
        // before any re-point or kill mutates the disks. Each face's cycle and
        // the edge lookups it depends on (which ring vert each beveled corner's
        // incoming / outgoing edge carries) are read while the original incident
        // edges still connect their original endpoints.
        // Each freshly-minted chamfer edge and the original incident edge whose
        // per-edge CD it inherits (BEVEL.md §6.5.1). §6.5.1's "leading endpoint"
        // is the chamfer edge's first endpoint in the walk of the face that *owns*
        // the edge for inheritance — and that is the **cap** (§5.1), not the
        // incident surface face. The cap is wound *opposite* its incident faces
        // (§5.1), so the chamfer edge `(r_i, r_(i+1))` — which in incident face
        // `f_i`'s rebuilt boundary runs `r_i → r_(i+1)` (leading `r_i`, owned by
        // the incoming edge `e_i`) — is traversed `r_(i+1) → r_i` by the cap.
        // Its cap-leading endpoint is therefore the *trailing* ring vert `r_(i+1)`
        // of the incident-face walk, whose owning edge is the **outgoing** edge
        // `e_(i+1)`. Recording the outgoing edge makes the inheritance the
        // canonical form reads land on §6.5.1's leading endpoint for *every*
        // corner uniformly, regardless of the corner's disk handedness. The owning
        // EdgeKey is recorded against the original mesh and survives the in-place
        // re-point below (same key / same CD slot), so it is still readable after
        // the rebuild mints the chamfer edge.
        let mut chamfer_cd: Vec<ChamferInheritance> = Vec::new();

        let mut rebuilds: Vec<(FaceKey, Vec<VertexKey>)> = Vec::with_capacity(faces.len());
        for &face in &faces {
            let cycle_verts = self.face_vert_keys(face);
            let cycle_length = cycle_verts.len();

            let mut new_cycle: Vec<VertexKey> = Vec::with_capacity(cycle_length + 1);
            for (index, &w) in cycle_verts.iter().enumerate() {
                if beveled.contains(&w) {
                    let corner = corners
                        .iter()
                        .find(|c| c.vertex == w)
                        .expect("a beveled corner vertex has a ring");
                    let prev_vertex = cycle_verts[(index + cycle_length - 1) % cycle_length];
                    let next_vertex = cycle_verts[(index + 1) % cycle_length];

                    let incoming_edge = self
                        .query_edge_exists(prev_vertex, w)
                        .expect("incident face's incoming corner edge exists");
                    let outgoing_edge = self
                        .query_edge_exists(w, next_vertex)
                        .expect("incident face's outgoing corner edge exists");

                    let leading_ring = self.ring_vertex_of(&corner.ring_vert_for_edge, incoming_edge);
                    let trailing_ring =
                        self.ring_vertex_of(&corner.ring_vert_for_edge, outgoing_edge);

                    // §6.5.1 (cap-frame leading): the chamfer edge
                    // (leading_ring, trailing_ring) inherits the per-edge CD of the
                    // **outgoing** original edge — the owning edge of the cap's
                    // leading endpoint (`trailing_ring` in this incident-face walk).
                    //
                    // §6.7.1.1: its seam / sharp bit, however, is the *logical AND*
                    // of the two flanking owning edges — the leading owner
                    // `incoming_edge` (`e_i`) and the trailing owner `outgoing_edge`
                    // (`e_(i+1)`) — so a single seamed incident edge leaves no
                    // dead-end spur on the cap. Record both owners.
                    chamfer_cd.push(ChamferInheritance {
                        leading_ring,
                        trailing_ring,
                        cd_owner: outgoing_edge,
                        leading_owner: incoming_edge,
                        trailing_owner: outgoing_edge,
                    });

                    new_cycle.push(leading_ring);
                    new_cycle.push(trailing_ring);
                } else {
                    new_cycle.push(w);
                }
            }
            rebuilds.push((face, new_cycle));
        }

        // §6.1 / §6.4.3: re-point every incident edge of every beveled corner in
        // place *after* the cycles are captured but *before* any face is killed,
        // preserving each edge's identity / per-edge CD. After this every
        // corner-adjacent edge connects its ring vert(s); the face rebuild below
        // reuses each one (the face builder's NO_DUPLICATE reuse) rather than
        // minting it fresh, leaving only the chamfer edges newly minted.
        let _ = beveled;
        self.repoint_incident_edges(corners);

        for (face, new_cycle) in rebuilds {
            self.face_kill(face);
            let new_face = self.face_create_verts(&new_cycle, FaceCreateFlags::NONE);
            self.face_attribs
                .copy_into(new_face.as_usize(), face.as_usize());
        }

        // §6.5.1: copy each owning (outgoing) edge's per-edge CD onto the chamfer
        // edge it owns. Run after every face is rebuilt so the chamfer edge exists
        // (the cap later *adopts* this same edge, §5.1, so no separate cap step is
        // needed). The chamfer edge joins two ring verts of one corner — never a
        // re-pointed surviving edge — so copying here cannot clobber a re-pointed
        // edge's own preserved CD.
        for inheritance in chamfer_cd {
            let Some(chamfer_edge) =
                self.query_edge_exists(inheritance.leading_ring, inheritance.trailing_ring)
            else {
                continue;
            };
            self.edge_attribs
                .copy_into(chamfer_edge.as_usize(), inheritance.cd_owner.as_usize());

            // §6.7.1.1: the whole-block copy above carried the trailing owner's
            // seam / sharp bit (mechanism (1)'s raw §6.5.1 copy). The corner
            // segment keeps the flag only when *both* flanking owners carry it, so
            // re-derive each bit as the AND of the leading and trailing owners,
            // overriding whatever the block copy left. Unconditional — not gated by
            // `mark_seam` / `mark_sharp` (§6.7.1 mechanism (1)).
            let seam = self.edge_seam(inheritance.leading_owner)
                && self.edge_seam(inheritance.trailing_owner);
            let sharp = self.edge_sharp(inheritance.leading_owner)
                && self.edge_sharp(inheritance.trailing_owner);
            self.set_edge_seam(chamfer_edge, seam);
            self.set_edge_sharp(chamfer_edge, sharp);
        }
    }

    /// Re-point every incident edge of every beveled corner in place
    /// (BEVEL.md §6.1 / §6.4.3).
    ///
    /// For a beveled corner `v` and one of its incident edges `e` carrying ring
    /// vert `r`, this swaps `e`'s `v`-endpoint to `r` with the in-place edge
    /// re-point primitive — the same `EdgeKey`, the same `edge_attribs` slot,
    /// just a new endpoint — so the re-pointed edge *is* the original edge with
    /// its identity and per-edge CD preserved. The re-point also patches `e`'s
    /// radial loops; those loops belong to faces about to be killed and rebuilt,
    /// so the transient face-cycle inconsistency it leaves is immediately
    /// discarded.
    ///
    /// An edge with a **single** beveled endpoint is re-pointed exactly once
    /// (`v → r`). A **shared** edge `E = {a, b}` whose *both* endpoints are
    /// beveled is reached once from `a`'s ring and once from `b`'s ring, so the
    /// loop below swaps it `a → r_a` when processing `a` and `b → r_b` when
    /// processing `b`, leaving `E` as the surviving inner segment `{r_a, r_b}`
    /// (§6.4.3) — the both-beveled case is the general re-point applied twice. The
    /// `(edge, endpoint)` guard ensures each endpoint of each edge is swapped at
    /// most once even if a corner's ring lists the same edge twice (a degenerate
    /// self-edge cannot arise here, but the guard keeps the pass idempotent).
    fn repoint_incident_edges(&mut self, corners: &[BeveledCorner]) {
        let mut done: Vec<(EdgeKey, VertexKey)> = Vec::new();
        for corner in corners {
            let v = corner.vertex;
            for &(edge, ring) in &corner.ring_vert_for_edge {
                if done.contains(&(edge, v)) {
                    continue;
                }
                // Precondition of the re-point primitive: the new vert (the ring
                // vert) must not already be on the edge. Ring verts are freshly
                // minted, so this holds for both the single- and both-beveled
                // cases.
                self.edge_replace_vertex_raw(edge, v, ring);
                done.push((edge, v));
            }
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
    fn build_cap_face(&mut self, ring_vert_for_edge: &[(EdgeKey, VertexKey)]) -> Option<CapFace> {
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

    /// Assign the `material` slot of one **new** bevel face (BEVEL.md §6.6).
    ///
    /// - `material >= 0`: stamp that slot uniformly (§6.6 regime 1).
    /// - `material < 0`: inherit `representative` — the material slot of a
    ///   representative adjacent original face chosen per new face (§6.6.1: cap ←
    ///   incident face of `v`, strip ← flanking face, patch ← junction-adjacent
    ///   face). The representative slot is captured against the pristine mesh by
    ///   the caller before the adjacent faces are killed / rebuilt.
    ///
    /// §6.6.1 specifies inheriting the whole face-attribute block; `material_index`
    /// is the only per-face attribute the parity comparison observes, so the
    /// inherit is realized by copying that slot (the remaining per-face CD axes are
    /// the §5.2 follow-up). Surviving re-stitched faces are never routed here, so
    /// they keep their own material (§6.6.2).
    fn assign_bevel_face_material(&mut self, face: FaceKey, material: i32, representative: i32) {
        let slot = if material >= 0 { material } else { representative };
        self.set_face_material_index(face, slot);
    }
}

struct CapFace {
    face: FaceKey,
    chamfer_edges: Vec<EdgeKey>,
}

/// How one freshly-minted chamfer edge `(leading_ring, trailing_ring)` inherits
/// its per-edge attributes from the original incident edges of the beveled corner.
///
/// - `cd_owner` is the owning edge whose whole per-edge attribute block the
///   chamfer copies (BEVEL.md §6.5.1, cap-frame leading = the outgoing edge).
/// - `leading_owner` / `trailing_owner` are the two original incident edges
///   flanking the corner segment (the leading owner `e_i` of `leading_ring` and
///   the trailing owner `e_(i+1)` of `trailing_ring`). The chamfer's seam / sharp
///   bit is the AND of these two (BEVEL.md §6.7.1.1), overriding the block copy.
struct ChamferInheritance {
    leading_ring: VertexKey,
    trailing_ring: VertexKey,
    cd_owner: EdgeKey,
    leading_owner: EdgeKey,
    trailing_owner: EdgeKey,
}

/// One (beveled edge, endpoint, adjacent face) corner of an edge-bevel junction
/// group and the **near-corner vert** its strip lands on there (BEVEL.md §7.6.1):
/// a fresh strip vert at a terminal endpoint, or a shared sector boundvert at a
/// junction.
struct CornerVert {
    edge: EdgeKey,
    vertex: VertexKey,
    face: FaceKey,
    near_vert: VertexKey,
}

/// One junction vertex's ordered boundvert ring (BEVEL.md §7.6.1), capped by the
/// patch face when `s ≥ 3` (§7.6.3). `vertex` is the junction the patch is built
/// around, used to look up its `-1` inherit material source (§6.6.1).
struct JunctionPatch {
    vertex: VertexKey,
    boundverts: Vec<VertexKey>,
    /// Per boundvert, the beveled edge that *opens* that sector in disk order
    /// (`g_k` for boundvert `B_k`, BEVEL.md §7.6.1). The patch edge between
    /// consecutive boundverts `(B_(k-1), B_k)` is crossed by `opening_edges[k]`, so
    /// this maps each junction bridging segment to the beveled edge it bridges,
    /// for the §6.7.4 seam / sharp bridging.
    opening_edges: Vec<EdgeKey>,
}

/// What a consumed endpoint corner becomes on one specific face (BEVEL.md §7.4 /
/// §7.6.1): the ordered near-corner vert(s) the re-stitch substitutes for the
/// single corner vertex `vertex` on `face`. One vert for a 1-for-1 swap (a strip's
/// radial face or a junction sector face), two for a terminal third face's
/// 1-for-2 absorbed corner.
struct FaceSubstitution {
    vertex: VertexKey,
    face: FaceKey,
    verts: Vec<VertexKey>,
}

/// Record (or overwrite) the ordered substitution for `(vertex, face)`.
fn set_face_substitution(
    substitutions: &mut Vec<FaceSubstitution>,
    vertex: VertexKey,
    face: FaceKey,
    verts: Vec<VertexKey>,
) {
    match substitutions
        .iter_mut()
        .find(|s| s.vertex == vertex && s.face == face)
    {
        Some(existing) => existing.verts = verts,
        None => substitutions.push(FaceSubstitution {
            vertex,
            face,
            verts,
        }),
    }
}

/// The ordered substitution recorded for `(vertex, face)`, if any.
fn face_substitution(
    substitutions: &[FaceSubstitution],
    vertex: VertexKey,
    face: FaceKey,
) -> Option<&[VertexKey]> {
    substitutions
        .iter()
        .find(|s| s.vertex == vertex && s.face == face)
        .map(|s| s.verts.as_slice())
}

/// One corner of the rotational ring around a vertex: an incident face flanked by
/// its two edges meeting at that vertex (BEVEL.md §7.6.1 disk walk). Consecutive
/// ring entries share an edge.
#[derive(Clone, Copy)]
struct VertexCorner {
    incoming_edge: EdgeKey,
    face: FaceKey,
    outgoing_edge: EdgeKey,
}

/// One angular sector of a junction (BEVEL.md §7.6.1): the run of faces between
/// two consecutive beveled edges around the disk. `bounding_edges` are the two
/// beveled edges delimiting it (the offset-line sources for the meet solve);
/// `corners` are every (edge, face) pair inside it.
struct Sector {
    bounding_edges: Vec<EdgeKey>,
    corners: Vec<(EdgeKey, FaceKey)>,
}

/// The near-corner vert recorded for the (edge, vertex, face) corner.
fn near_corner(
    corner_vert: &[CornerVert],
    edge: EdgeKey,
    vertex: VertexKey,
    face: FaceKey,
) -> VertexKey {
    corner_vert
        .iter()
        .find(|c| c.edge == edge && c.vertex == vertex && c.face == face)
        .map(|c| c.near_vert)
        .expect("every beveled corner has a recorded near-corner vert")
}

/// True if `ring` contains the same vertex twice (a collapsed strip side, §7.6.3).
fn ring_has_repeat(ring: &[VertexKey]) -> bool {
    for i in 0..ring.len() {
        for j in (i + 1)..ring.len() {
            if ring[i] == ring[j] {
                return true;
            }
        }
    }
    false
}

/// Split a vertex's rotational corner ring into sectors at the beveled edges
/// (BEVEL.md §7.6.1). Each sector runs from one beveled edge, through any unbeveled
/// edges/faces, to the next beveled edge, and yields one boundvert. Returns one
/// `Sector` per gap between consecutive beveled edges in disk order.
fn sector_split(corners: &[VertexCorner], is_beveled: &dyn Fn(EdgeKey) -> bool) -> Vec<Sector> {
    if corners.is_empty() {
        return Vec::new();
    }

    // Find a starting corner whose incoming edge is beveled, so the first sector
    // opens on a beveled edge. (A closed junction always has ≥ 2 beveled edges.)
    let start = corners
        .iter()
        .position(|c| is_beveled(c.incoming_edge))
        .unwrap_or(0);
    let count = corners.len();

    let mut sectors: Vec<Sector> = Vec::new();
    let mut current = Sector {
        bounding_edges: vec![corners[start].incoming_edge],
        corners: Vec::new(),
    };

    for step in 0..count {
        let corner = corners[(start + step) % count];
        current.corners.push((corner.incoming_edge, corner.face));
        // A different (outgoing) beveled edge closes the current sector and opens
        // the next.
        if is_beveled(corner.outgoing_edge) {
            current.corners.push((corner.outgoing_edge, corner.face));
            if !current.bounding_edges.contains(&corner.outgoing_edge) {
                current.bounding_edges.push(corner.outgoing_edge);
            }
            sectors.push(current);
            current = Sector {
                bounding_edges: vec![corner.outgoing_edge],
                corners: Vec::new(),
            };
        }
    }

    sectors
}

/// Solve the small constrained meet system by the normal equations
/// `AᵀA·x = Aᵀc` (BEVEL.md §7.6.2). Each row is `(direction, target)` meaning
/// `dot(x − anchor, direction) = target`. Returns the solution as
/// `anchor + offset`, or `None` if `AᵀA` is singular and no minimum-norm fallback
/// is available.
fn solve_normal_equations(rows: &[(Vec3, f32)], anchor: Vec3) -> Option<Vec3> {
    if rows.is_empty() {
        return None;
    }

    // Build the symmetric 3×3 normal matrix M = AᵀA and rhs = Aᵀc, with x measured
    // relative to `anchor` (so the right-hand targets are already relative).
    let mut m = [[0.0f32; 3]; 3];
    let mut rhs = [0.0f32; 3];
    for &(dir, target) in rows {
        let d = [dir.x, dir.y, dir.z];
        for i in 0..3 {
            for j in 0..3 {
                m[i][j] += d[i] * d[j];
            }
            rhs[i] += d[i] * target;
        }
    }

    if let Some(solution) = solve_3x3(&m, &rhs) {
        return Some(anchor + Vec3::new(solution[0], solution[1], solution[2]));
    }

    // Rank-deficient AᵀA (§7.6.2 degeneracy): fall back to the minimum-norm point,
    // which anchored at the vertex is the solution nearest the vertex. A
    // Tikhonov-regularised solve (M + εI) recovers exactly that as ε → 0.
    let epsilon = 1.0e-6;
    let mut regularised = m;
    for i in 0..3 {
        regularised[i][i] += epsilon;
    }
    solve_3x3(&regularised, &rhs).map(|s| anchor + Vec3::new(s[0], s[1], s[2]))
}

/// Solve a 3×3 linear system by Cramer's rule; `None` when the matrix is singular.
fn solve_3x3(m: &[[f32; 3]; 3], rhs: &[f32; 3]) -> Option<[f32; 3]> {
    let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    if det.abs() <= 1.0e-12 {
        return None;
    }
    let mut solution = [0.0f32; 3];
    for column in 0..3 {
        let mut replaced = *m;
        for row in 0..3 {
            replaced[row][column] = rhs[row];
        }
        let det_c = replaced[0][0]
            * (replaced[1][1] * replaced[2][2] - replaced[1][2] * replaced[2][1])
            - replaced[0][1] * (replaced[1][0] * replaced[2][2] - replaced[1][2] * replaced[2][0])
            + replaced[0][2] * (replaced[1][0] * replaced[2][1] - replaced[1][1] * replaced[2][0]);
        solution[column] = det_c / det;
    }
    Some(solution)
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
        (mesh.vertex_count(), mesh.edge_count(), mesh.face_count())
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
            let slide = (ring_position - pos_a)
                .length()
                .min((ring_position - pos_b).length());
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
        assert_ne!(
            coincident[0], coincident[1],
            "the pair is two distinct verts"
        );

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
            !mesh.vert_keys().into_iter().any(|v| v == corner),
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
            assert!(
                (distance - 0.25).abs() < 1.0e-5,
                "small slide is the offset"
            );
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

        assert_eq!(
            counts(&mesh),
            (10, 15, 7),
            "isolated edge +2 V / +3 E / +1 F"
        );
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
        let edge = cube_edge(
            &mesh,
            Vec3::new(1.0, -1.0, -1.0),
            Vec3::new(-1.0, -1.0, -1.0),
        );

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
        let edge = cube_edge(
            &mesh,
            Vec3::new(1.0, -1.0, -1.0),
            Vec3::new(-1.0, -1.0, -1.0),
        );

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

    /// BEVEL.md §7.6.3 (`s = 2`, through junction): beveling the two cube edges
    /// {0,1} and {0,3} that share vertex 0 makes vertex 0 a two-edge junction. The
    /// two arriving strips share a pair of boundverts (§7.6.1) and run into each
    /// other with **no** patch face (§7.6.3). Far endpoints 1 and 3 are valence-3
    /// terminals. Count delta over the cube (8 V / 12 E / 6 F): +3 V / +5 E / +2 F
    /// → 11 V / 17 E / 8 F.
    #[test]
    fn edge_bevel_two_edge_junction_no_patch() {
        let mut mesh = LoopMesh::new();
        mesh_add_cube(&mut mesh);
        assert_eq!(counts(&mesh), (8, 12, 6));

        let e01 = cube_edge(
            &mesh,
            Vec3::new(1.0, -1.0, -1.0),
            Vec3::new(-1.0, -1.0, -1.0),
        );
        let e03 = cube_edge(&mesh, Vec3::new(1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, -1.0));

        let result = mesh.bevel(
            &[e01.into(), e03.into()],
            BevelOptions {
                offset: 0.25,
                affect: BevelAffect::Edges,
                ..Default::default()
            },
        );

        assert_eq!(
            counts(&mesh),
            (11, 17, 8),
            "s=2 through-junction +3 V / +5 E / +2 F",
        );
        // Two strip quads, no patch face.
        assert_eq!(result.faces.len(), 2, "two strip faces, no s=2 patch");

        // The Front-side sector is a single planar face: its boundvert is the exact
        // 2-line meet at (1-offset, -1+offset, -1) = (0.75, -0.75, -1).
        let front_meet = Vec3::new(0.75, -0.75, -1.0);
        let near = result
            .verts
            .iter()
            .filter(|&&v| (mesh.vertex_position(v) - front_meet).length() < 1.0e-5)
            .count();
        assert_eq!(
            near, 1,
            "the planar Front sector boundvert is the exact meet"
        );

        // Vertex 0 and the two far endpoints are consumed.
        for removed in [
            Vec3::new(1.0, -1.0, -1.0),
            Vec3::new(-1.0, -1.0, -1.0),
            Vec3::new(1.0, 1.0, -1.0),
        ] {
            assert!(
                !mesh
                    .vert_keys()
                    .into_iter()
                    .any(|v| mesh.vertex_position(v) == removed),
                "original endpoint {removed:?} removed",
            );
        }
        mesh.validate_topology();
    }

    /// BEVEL.md §7.6.3 (`s = 3`, true junction): beveling all three of cube vertex
    /// 0's edges {0,1}, {0,3}, {0,4} makes vertex 0 a three-edge junction with no
    /// unbeveled legs (the cleanest junction, §7.6.5). Its three boundverts — one
    /// per planar sector face (Front, Left, Bottom) — form a ring closed by one
    /// flat **patch face** (§7.6.3). Far endpoints 1, 3, 4 are valence-3 terminals.
    /// Count delta over the cube: +5 V / +9 E / +4 F → 13 V / 21 E / 10 F.
    #[test]
    fn edge_bevel_three_edge_junction_one_patch() {
        let mut mesh = LoopMesh::new();
        mesh_add_cube(&mut mesh);
        assert_eq!(counts(&mesh), (8, 12, 6));

        let e01 = cube_edge(
            &mesh,
            Vec3::new(1.0, -1.0, -1.0),
            Vec3::new(-1.0, -1.0, -1.0),
        );
        let e03 = cube_edge(&mesh, Vec3::new(1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, -1.0));
        let e04 = cube_edge(&mesh, Vec3::new(1.0, -1.0, -1.0), Vec3::new(1.0, -1.0, 1.0));

        let result = mesh.bevel(
            &[e01.into(), e03.into(), e04.into()],
            BevelOptions {
                offset: 0.25,
                affect: BevelAffect::Edges,
                ..Default::default()
            },
        );

        assert_eq!(
            counts(&mesh),
            (13, 21, 10),
            "s=3 junction +5 V / +9 E / +4 F",
        );
        // Three strip quads + one patch face.
        assert_eq!(result.faces.len(), 4, "three strips + one patch");

        // The patch face is the triangle over the three boundverts. Each sector is
        // a single planar cube face, so each boundvert is an exact 2-line meet at
        // distance offset·√2 from the original corner (1,-1,-1).
        let patch = result
            .faces
            .iter()
            .copied()
            .find(|&f| mesh.face_vert_keys(f).len() == 3)
            .expect("the s=3 junction patch is a triangle");
        let corner = Vec3::new(1.0, -1.0, -1.0);
        let expected = 0.25 * 2.0_f32.sqrt();
        for v in mesh.face_vert_keys(patch) {
            let distance = (mesh.vertex_position(v) - corner).length();
            assert!(
                (distance - expected).abs() < 1.0e-5,
                "patch boundvert {distance} should sit offset·√2 from the corner",
            );
        }

        // Vertex 0 and all three far endpoints are consumed.
        for removed in [
            corner,
            Vec3::new(-1.0, -1.0, -1.0),
            Vec3::new(1.0, 1.0, -1.0),
            Vec3::new(1.0, -1.0, 1.0),
        ] {
            assert!(
                !mesh
                    .vert_keys()
                    .into_iter()
                    .any(|v| mesh.vertex_position(v) == removed),
                "original endpoint {removed:?} removed",
            );
        }
        mesh.validate_topology();
    }

    /// BEVEL.md §7.6.3: every edge of the `s = 3` junction patch face is shared
    /// with exactly one arriving strip (radial count 2) — the manifold condition
    /// the patch winding (§5.1) preserves.
    #[test]
    fn edge_bevel_three_edge_junction_patch_manifold() {
        let mut mesh = LoopMesh::new();
        mesh_add_cube(&mut mesh);
        let e01 = cube_edge(
            &mesh,
            Vec3::new(1.0, -1.0, -1.0),
            Vec3::new(-1.0, -1.0, -1.0),
        );
        let e03 = cube_edge(&mesh, Vec3::new(1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, -1.0));
        let e04 = cube_edge(&mesh, Vec3::new(1.0, -1.0, -1.0), Vec3::new(1.0, -1.0, 1.0));

        let result = mesh.bevel(
            &[e01.into(), e03.into(), e04.into()],
            BevelOptions {
                offset: 0.25,
                affect: BevelAffect::Edges,
                ..Default::default()
            },
        );

        let patch = result
            .faces
            .iter()
            .copied()
            .find(|&f| mesh.face_vert_keys(f).len() == 3)
            .expect("patch triangle");
        for loop_key in mesh.face_loop_keys(patch) {
            let edge = mesh.loop_edge(loop_key);
            assert_eq!(
                mesh.edge_radial_iter(edge).count(),
                2,
                "each patch edge is manifold (shared with one strip)",
            );
        }
        mesh.validate_topology();
    }

    /// BEVEL.md §2.1: a non-positive offset is a no-op for edge bevel too.
    #[test]
    fn edge_bevel_non_positive_offset_is_noop() {
        let mut mesh = LoopMesh::new();
        mesh_add_cube(&mut mesh);
        let before = counts(&mesh);
        let edge = cube_edge(
            &mesh,
            Vec3::new(1.0, -1.0, -1.0),
            Vec3::new(-1.0, -1.0, -1.0),
        );

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

    /// Register a per-edge Float3 `tint` layer and tag the edge between `pa` and
    /// `pb` with `value`, returning the layer key and that edge's key.
    fn tag_edge_tint(
        mesh: &mut LoopMesh,
        pa: Vec3,
        pb: Vec3,
        value: [f32; 3],
    ) -> (ElementAttributeKey, EdgeKey) {
        let tint = mesh
            .register_edge_attribute(ElementAttributeDesc::numeric(
                AttributeKey::custom("tint"),
                AttrType::Vec3,
            ))
            .unwrap();
        let edge = cube_edge(mesh, pa, pb);
        mesh.edge_attribs_mut()
            .set_vec3(tint, edge.as_usize(), value);
        (tint, edge)
    }

    /// BEVEL.md §6.4.3: beveling the two endpoints of an interior (manifold)
    /// shared edge keeps that edge in place — **same `EdgeKey`**, radial count 2,
    /// and its per-edge custom-data carried over — rather than minting a fresh
    /// inner segment that would default-initialise the attribute.
    #[test]
    fn shared_edge_manifold_keeps_identity_and_per_edge_cd() {
        let mut mesh = LoopMesh::new();
        mesh_add_cube(&mut mesh);
        let pa = Vec3::new(1.0, -1.0, -1.0);
        let pb = Vec3::new(-1.0, -1.0, -1.0);
        let (tint, shared_edge) = tag_edge_tint(&mut mesh, pa, pb, [0.42, 0.0, 0.0]);

        let a = vert_at(&mesh, pa);
        let b = vert_at(&mesh, pb);
        mesh.bevel(
            &[a.into(), b.into()],
            BevelOptions {
                offset: 0.5,
                ..Default::default()
            },
        );

        // The same edge object survives — its key is still alive — re-pointed onto
        // the two ring verts r_a = (0.5,-1,-1), r_b = (-0.5,-1,-1).
        assert!(
            mesh.edge_keys().contains(&shared_edge),
            "the surviving inner segment is the *same* edge (identity preserved)",
        );
        let r_a = vert_at(&mesh, Vec3::new(0.5, -1.0, -1.0));
        let r_b = vert_at(&mesh, Vec3::new(-0.5, -1.0, -1.0));
        assert_eq!(
            mesh.query_edge_exists(r_a, r_b),
            Some(shared_edge),
            "the original edge, re-pointed to {{r_a, r_b}}, is the inner segment",
        );
        assert_eq!(
            mesh.edge_radial_iter(shared_edge).count(),
            2,
            "an interior shared edge keeps radial count 2",
        );

        let carried = mesh
            .edge_attribs()
            .get_vec3(tint, shared_edge.as_usize())
            .unwrap();
        assert_eq!(
            carried, [0.42, 0.0, 0.0],
            "the surviving edge keeps its original per-edge CD",
        );
        mesh.validate_topology();
    }

    /// BEVEL.md §6.4.3 (boundary sub-case): when the shared edge straddled by a
    /// *single* face is beveled at both endpoints, the same edge still survives
    /// in place and keeps its per-edge CD — with radial count 1, the one
    /// re-stitched face.
    #[test]
    fn shared_edge_single_face_keeps_identity_and_per_edge_cd() {
        // A flat quad; bevel two adjacent corners so the shared edge between them
        // is a boundary edge (one incident face).
        let mut mesh = LoopMesh::new();
        let positions = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::new(2.0, 2.0, 0.0),
            Vec3::new(0.0, 2.0, 0.0),
        ];
        let verts: Vec<VertexKey> = positions.iter().map(|&p| mesh.create_vertex(p)).collect();
        let edges: Vec<EdgeKey> = (0..4)
            .map(|i| mesh.create_edge([verts[i], verts[(i + 1) % 4]], EdgeCreateFlags::NONE))
            .collect();
        mesh.face_create(&verts, &edges, FaceCreateFlags::NONE);

        let tint = mesh
            .register_edge_attribute(ElementAttributeDesc::numeric(
                AttributeKey::custom("tint"),
                AttrType::Vec3,
            ))
            .unwrap();
        // The shared edge is verts[0]-verts[1], a boundary edge of the single quad.
        let shared_edge = edges[0];
        mesh.edge_attribs_mut()
            .set_vec3(tint, shared_edge.as_usize(), [0.0, 0.7, 0.0]);
        assert_eq!(
            mesh.edge_radial_iter(shared_edge).count(),
            1,
            "shared edge has a single incident face before the bevel",
        );

        mesh.bevel(
            &[verts[0].into(), verts[1].into()],
            BevelOptions {
                offset: 0.5,
                ..Default::default()
            },
        );

        assert!(
            mesh.edge_keys().contains(&shared_edge),
            "the single-face shared edge survives as the same edge",
        );
        // r_a slid 0.5 from (0,0,0) toward (2,0,0) -> (0.5,0,0);
        // r_b slid 0.5 from (2,0,0) toward (0,0,0) -> (1.5,0,0).
        let r_a = vert_at(&mesh, Vec3::new(0.5, 0.0, 0.0));
        let r_b = vert_at(&mesh, Vec3::new(1.5, 0.0, 0.0));
        assert_eq!(
            mesh.query_edge_exists(r_a, r_b),
            Some(shared_edge),
            "the re-pointed edge spans the two ring verts",
        );
        assert_eq!(
            mesh.edge_radial_iter(shared_edge).count(),
            1,
            "still a single incident face after the bevel",
        );
        let carried = mesh
            .edge_attribs()
            .get_vec3(tint, shared_edge.as_usize())
            .unwrap();
        assert_eq!(
            carried, [0.0, 0.7, 0.0],
            "the surviving boundary edge keeps its per-edge CD",
        );
        mesh.validate_topology();
    }

    /// BEVEL.md §6.4.2 (clamped-collapse degenerate): even when the surviving
    /// inner segment has *zero length* (the two ring verts coincident), it is
    /// still a real edge — the same one — and keeps its per-edge CD.
    #[test]
    fn shared_edge_clamped_collapse_keeps_per_edge_cd() {
        let mut mesh = LoopMesh::new();
        mesh_add_cube(&mut mesh);
        let pa = Vec3::new(1.0, -1.0, -1.0);
        let pb = Vec3::new(-1.0, -1.0, -1.0);
        let (tint, shared_edge) = tag_edge_tint(&mut mesh, pa, pb, [0.1, 0.2, 0.3]);

        let a = vert_at(&mesh, pa);
        let b = vert_at(&mesh, pb);
        mesh.bevel(
            &[a.into(), b.into()],
            BevelOptions {
                offset: 1.5,
                clamp_overlap: true,
                ..Default::default()
            },
        );

        // The clamp collapses r_a, r_b onto the midpoint (0,-1,-1): a zero-length
        // edge, but still the *same* surviving edge with its CD intact.
        assert!(
            mesh.edge_keys().contains(&shared_edge),
            "the zero-length surviving segment is still the same edge",
        );
        let [p, q] = mesh.edge_vert_keys(shared_edge);
        let length = (mesh.vertex_position(p) - mesh.vertex_position(q)).length();
        assert!(length < 1.0e-5, "the surviving segment is zero-length");
        let carried = mesh
            .edge_attribs()
            .get_vec3(tint, shared_edge.as_usize())
            .unwrap();
        assert_eq!(
            carried, [0.1, 0.2, 0.3],
            "even the collapsed edge keeps its per-edge CD",
        );
        mesh.validate_topology();
    }

    /// BEVEL.md §6.1 / §6.4.3: a **single-beveled-endpoint** corner-adjacent edge
    /// (one beveled endpoint, one not) is re-pointed in place exactly like the
    /// shared edge — same `EdgeKey`, same per-edge CD — it is *not* re-minted by
    /// the incident-face rebuild. Bevels only vertex 0 and checks edge {0,3}
    /// (vertex 3 not beveled) survives re-pointed onto 0's ring vert.
    #[test]
    fn corner_adjacent_edge_single_beveled_endpoint_keeps_identity_and_per_edge_cd() {
        let mut mesh = LoopMesh::new();
        mesh_add_cube(&mut mesh);
        // Edge {0,3}: (1,-1,-1)-(1,1,-1). Only vertex 0 will be beveled.
        let p0 = Vec3::new(1.0, -1.0, -1.0);
        let p3 = Vec3::new(1.0, 1.0, -1.0);
        let (tint, adjacent_edge) = tag_edge_tint(&mut mesh, p0, p3, [0.0, 0.0, 0.9]);

        let v0 = vert_at(&mesh, p0);
        let v3 = vert_at(&mesh, p3);
        mesh.bevel(
            &[v0.into()],
            BevelOptions {
                offset: 0.5,
                ..Default::default()
            },
        );

        assert!(
            mesh.edge_keys().contains(&adjacent_edge),
            "the corner-adjacent edge survives as the *same* edge (identity preserved)",
        );
        // Vertex 0's ring vert on {0,3} slides 0.5 from (1,-1,-1) toward
        // (1,1,-1) -> (1,-0.5,-1). Vertex 3 is untouched.
        let ring = vert_at(&mesh, Vec3::new(1.0, -0.5, -1.0));
        assert!(
            mesh.vert_keys().contains(&v3),
            "the non-beveled far endpoint is untouched",
        );
        assert_eq!(
            mesh.query_edge_exists(ring, v3),
            Some(adjacent_edge),
            "the original edge, re-pointed {{0 -> r_0}}, spans the ring vert and vertex 3",
        );
        assert_eq!(
            mesh.edge_radial_iter(adjacent_edge).count(),
            2,
            "the re-pointed interior edge keeps radial count 2",
        );
        let carried = mesh
            .edge_attribs()
            .get_vec3(tint, adjacent_edge.as_usize())
            .unwrap();
        assert_eq!(
            carried, [0.0, 0.0, 0.9],
            "the single-beveled-endpoint edge keeps its per-edge CD",
        );
        mesh.validate_topology();
    }

    /// BEVEL.md §6.1 / §6.4.3: over the full shared-edge pair, *every* incident
    /// edge of both beveled corners is preserved in place — the shared edge {0,1}
    /// (both endpoints beveled) and each corner-adjacent edge (one beveled
    /// endpoint). Only the chamfer edges between a corner's ring verts are minted
    /// fresh. This pins the broad per-edge-CD survival the narrow shared-edge fix
    /// did not cover.
    #[test]
    fn shared_edge_pair_preserves_all_incident_edges_cd() {
        let mut mesh = LoopMesh::new();
        mesh_add_cube(&mut mesh);

        let tint = mesh
            .register_edge_attribute(ElementAttributeDesc::numeric(
                AttributeKey::custom("tint"),
                AttrType::Vec3,
            ))
            .unwrap();

        // Tag every incident edge of the two beveled corners 0 and 1 with a
        // distinct value, and record (EdgeKey, ring-vert-position-after-bevel,
        // far-endpoint-position, expected-tint) for each.
        let p0 = Vec3::new(1.0, -1.0, -1.0);
        let p1 = Vec3::new(-1.0, -1.0, -1.0);

        // (near beveled vertex position, far endpoint position, tint)
        let incident = [
            // shared edge {0,1}: both endpoints beveled.
            (p0, p1, [0.42, 0.0, 0.0]),
            // vertex 0's other edges {0,3}, {0,4}: one beveled endpoint.
            (p0, Vec3::new(1.0, 1.0, -1.0), [0.1, 0.0, 0.0]),
            (p0, Vec3::new(1.0, -1.0, 1.0), [0.2, 0.0, 0.0]),
            // vertex 1's other edges {1,2}, {1,5}: one beveled endpoint.
            (p1, Vec3::new(-1.0, 1.0, -1.0), [0.3, 0.0, 0.0]),
            (p1, Vec3::new(-1.0, -1.0, 1.0), [0.4, 0.0, 0.0]),
        ];

        let mut tagged: Vec<(EdgeKey, [f32; 3])> = Vec::new();
        for &(near, far, value) in &incident {
            let edge = cube_edge(&mesh, near, far);
            mesh.edge_attribs_mut().set_vec3(tint, edge.as_usize(), value);
            tagged.push((edge, value));
        }

        let v0 = vert_at(&mesh, p0);
        let v1 = vert_at(&mesh, p1);
        mesh.bevel(
            &[v0.into(), v1.into()],
            BevelOptions {
                offset: 0.5,
                ..Default::default()
            },
        );

        for &(edge, value) in &tagged {
            assert!(
                mesh.edge_keys().contains(&edge),
                "every incident edge survives with the same EdgeKey",
            );
            let carried = mesh.edge_attribs().get_vec3(tint, edge.as_usize()).unwrap();
            assert_eq!(
                carried, value,
                "incident edge {edge:?} keeps its per-edge CD across the bevel",
            );
        }
        mesh.validate_topology();
    }

    /// BEVEL.md §6.5.1: each freshly-minted corner-fill chamfer edge inherits the
    /// per-edge CD of the original incident edge that owns its *leading* ring-vert
    /// endpoint in the re-stitched face's boundary order. The discriminating pin:
    /// the chamfer edge whose leading ring vert is the ring vert of the shared
    /// edge {0,1} picks up {0,1}'s tint, on *both* corners (§6.5.2 — {0,1}
    /// contributes a ring vert to each beveled corner). A chamfer edge leads with
    /// the {0,1} ring vert in every re-stitched face where {0,1} is the incoming
    /// corner edge, so the {0,1} ring vert may lead more than one chamfer edge
    /// within a corner; the load-bearing fact is that the inheritance fires (an
    /// edge next to {0,1}'s entry carries {0,1}'s tint), not a fresh default.
    #[test]
    fn corner_fill_chamfer_edge_inherits_leading_owning_edge_cd() {
        let mut mesh = LoopMesh::new();
        mesh_add_cube(&mut mesh);

        let tint = mesh
            .register_edge_attribute(ElementAttributeDesc::numeric(
                AttributeKey::custom("tint"),
                AttrType::Vec3,
            ))
            .unwrap();

        let p0 = Vec3::new(1.0, -1.0, -1.0);
        let p1 = Vec3::new(-1.0, -1.0, -1.0);
        let shared_value = [0.42, 0.0, 0.0];

        // Tag the shared edge {0,1} distinctly; the rest stay at the default so
        // only the {0,1}-leading chamfer edge can carry the marker.
        let shared_edge = cube_edge(&mesh, p0, p1);
        mesh.edge_attribs_mut()
            .set_vec3(tint, shared_edge.as_usize(), shared_value);

        let v0 = vert_at(&mesh, p0);
        let v1 = vert_at(&mesh, p1);
        let result = mesh.bevel(
            &[v0.into(), v1.into()],
            BevelOptions {
                offset: 0.5,
                ..Default::default()
            },
        );

        // The {0,1} ring vert of corner 0 slides 0.5 from (1,-1,-1) toward
        // (-1,-1,-1) -> (0.5,-1,-1); of corner 1, from (-1,-1,-1) toward
        // (1,-1,-1) -> (-0.5,-1,-1). Each corner's fill must produce exactly one
        // chamfer edge that leads with its {0,1} ring vert and so inherits
        // {0,1}'s tint.
        for ring_pos in [Vec3::new(0.5, -1.0, -1.0), Vec3::new(-0.5, -1.0, -1.0)] {
            let ring = vert_at(&mesh, ring_pos);
            let inheriting: Vec<EdgeKey> = mesh
                .edge_keys()
                .into_iter()
                .filter(|&edge| {
                    let [a, b] = mesh.edge_vert_keys(edge);
                    (a == ring || b == ring)
                        && mesh.edge_attribs().get_vec3(tint, edge.as_usize())
                            == Some(shared_value)
                })
                .collect();
            assert!(
                !inheriting.is_empty(),
                "a chamfer edge at the {{0,1}} ring vert {ring_pos:?} inherits \
                 {{0,1}}'s per-edge CD rather than default-filling (§6.5.1)",
            );
            // Each inheriting edge must be a genuine chamfer edge (both endpoints
            // ring verts), not a re-pointed survivor that happens to share the tint.
            for &edge in &inheriting {
                let [a, b] = mesh.edge_vert_keys(edge);
                let other = if a == ring { b } else { a };
                assert!(
                    result_has_vert_near(&mesh, &result, mesh.vertex_position(other)) > 0
                        || other == ring,
                    "the inheriting edge {edge:?} is a chamfer edge between two ring verts",
                );
            }
        }
        mesh.validate_topology();
    }

    /// BEVEL.md §6.5.1: no corner-fill chamfer edge is left default-filled — every
    /// freshly-minted chamfer edge inherits a tagged original edge's per-edge CD.
    /// This is the broad closure of the 6/18 corner-fill divergence: with every
    /// incident edge of both corners tagged distinctly, each chamfer edge carries
    /// the value of the original edge whose ring vert leads it (never the [0,0,0]
    /// default a fresh-mint would leave).
    #[test]
    fn corner_fill_chamfer_edges_are_never_default_filled() {
        let mut mesh = LoopMesh::new();
        mesh_add_cube(&mut mesh);

        let tint = mesh
            .register_edge_attribute(ElementAttributeDesc::numeric(
                AttributeKey::custom("tint"),
                AttrType::Vec3,
            ))
            .unwrap();

        let p0 = Vec3::new(1.0, -1.0, -1.0);
        let p1 = Vec3::new(-1.0, -1.0, -1.0);

        // Tag every incident edge of both corners distinctly (shared + four
        // corner-adjacent), each component non-zero so an inherited value is
        // always distinguishable from the [0,0,0] default.
        let incident = [
            (p0, p1, [0.42, 0.5, 0.5]),
            (p0, Vec3::new(1.0, 1.0, -1.0), [0.1, 0.5, 0.5]),
            (p0, Vec3::new(1.0, -1.0, 1.0), [0.2, 0.5, 0.5]),
            (p1, Vec3::new(-1.0, 1.0, -1.0), [0.3, 0.5, 0.5]),
            (p1, Vec3::new(-1.0, -1.0, 1.0), [0.4, 0.5, 0.5]),
        ];
        let mut tagged_values: Vec<[f32; 3]> = Vec::new();
        for &(near, far, value) in &incident {
            let edge = cube_edge(&mesh, near, far);
            mesh.edge_attribs_mut().set_vec3(tint, edge.as_usize(), value);
            tagged_values.push(value);
        }

        let v0 = vert_at(&mesh, p0);
        let v1 = vert_at(&mesh, p1);
        let result = mesh.bevel(
            &[v0.into(), v1.into()],
            BevelOptions {
                offset: 0.5,
                ..Default::default()
            },
        );

        // The new edges between two ring verts of the same corner are the
        // chamfer / fill edges; each must carry a tagged value, never the default.
        let ring_verts: Vec<VertexKey> = result.verts.clone();
        let mut chamfer_count = 0;
        for &edge in &result.edges {
            let [a, b] = mesh.edge_vert_keys(edge);
            let is_chamfer = ring_verts.contains(&a) && ring_verts.contains(&b);
            if !is_chamfer {
                continue;
            }
            chamfer_count += 1;
            let carried = mesh.edge_attribs().get_vec3(tint, edge.as_usize()).unwrap();
            assert_ne!(
                carried, [0.0, 0.0, 0.0],
                "chamfer edge {edge:?} must inherit an owning edge's CD, not default",
            );
            assert!(
                tagged_values.contains(&carried),
                "chamfer edge {edge:?} inherits one of the tagged owning-edge values, got {carried:?}",
            );
        }
        assert!(
            chamfer_count >= 2,
            "both interior corners contribute fill chamfer edges",
        );
        mesh.validate_topology();
    }

    /// BEVEL.md §6.5.1 (cap-frame leading): pin the *specific* chamfer-edge -> CD
    /// mapping, not just the multiset. The cap is wound opposite its incident
    /// faces (§5.1), so the chamfer edge `(r_i, r_(i+1))` inherits the **outgoing**
    /// edge `e_(i+1)`'s CD (the owning edge of the cap's leading endpoint), uniformly
    /// for every corner. This catches the cyclic-shift permutation a per-edge-set
    /// check passes spuriously through: tagging each owning edge distinctly forces
    /// the test to fail under the wrong (incoming) assignment.
    #[test]
    fn corner_fill_chamfer_edge_mapping_follows_outgoing_owning_edge() {
        let mut mesh = LoopMesh::new();
        mesh_add_cube(&mut mesh);

        let tint = mesh
            .register_edge_attribute(ElementAttributeDesc::numeric(
                AttributeKey::custom("tint"),
                AttrType::Vec3,
            ))
            .unwrap();

        // Corner v0 = (1,-1,-1) with its three incident edges distinctly tagged.
        let p0 = Vec3::new(1.0, -1.0, -1.0);
        let p1 = Vec3::new(-1.0, -1.0, -1.0); // e0 = {v0,v1}
        let p3 = Vec3::new(1.0, 1.0, -1.0); // e3 = {v3,v0}
        let p4 = Vec3::new(1.0, -1.0, 1.0); // e8 = {v0,v4}
        let e0_value = [0.10, 0.5, 0.5];
        let e3_value = [0.20, 0.5, 0.5];
        let e8_value = [0.30, 0.5, 0.5];
        for &(near, far, value) in &[
            (p0, p1, e0_value),
            (p0, p3, e3_value),
            (p0, p4, e8_value),
        ] {
            let edge = cube_edge(&mesh, near, far);
            mesh.edge_attribs_mut().set_vec3(tint, edge.as_usize(), value);
        }

        let v0 = vert_at(&mesh, p0);
        mesh.bevel(
            &[v0.into()],
            BevelOptions {
                offset: 0.5,
                ..Default::default()
            },
        );

        // Ring-vert positions: r_e0 slides v0->v1, r_e3 v0->v3, r_e8 v0->v4.
        let r_e0 = Vec3::new(0.5, -1.0, -1.0);
        let r_e3 = Vec3::new(1.0, -0.5, -1.0);
        let r_e8 = Vec3::new(1.0, -1.0, -0.5);

        // The chamfer triangle of v0 is r_e3 -> r_e0 -> r_e8 (-> r_e3). Per the
        // cap-frame rule each chamfer edge carries its *outgoing* owning edge's
        // tint, which lands on the edge's r_(i+1) (trailing-in-face) endpoint:
        //   (r_e3, r_e0)  outgoing e0  -> e0_value   (Front)
        //   (r_e0, r_e8)  outgoing e8  -> e8_value   (Bottom)
        //   (r_e8, r_e3)  outgoing e3  -> e3_value   (Left)
        // The wrong (incoming) assignment would put e3/e0/e8 on these instead,
        // so each row distinguishes the two readings.
        for &(near_a, near_b, expected) in &[
            (r_e3, r_e0, e0_value),
            (r_e0, r_e8, e8_value),
            (r_e8, r_e3, e3_value),
        ] {
            let a = vert_at(&mesh, near_a);
            let b = vert_at(&mesh, near_b);
            let edge = mesh
                .query_edge_exists(a, b)
                .expect("the chamfer edge between these two ring verts exists");
            let carried = mesh.edge_attribs().get_vec3(tint, edge.as_usize()).unwrap();
            assert_eq!(
                carried, expected,
                "chamfer edge {near_a:?}-{near_b:?} must inherit its outgoing owning \
                 edge's CD {expected:?} (cap-frame leading, §6.5.1), got {carried:?}",
            );
        }
        mesh.validate_topology();
    }

    /// The new cap face produced by beveling cube-corner vert 0 (interior
    /// valence-3 → one triangular cap, the single new face). Located by the
    /// result buffer.
    fn beveled_corner_cap(mesh: &LoopMesh, result: &BevelResult) -> FaceKey {
        assert_eq!(
            result.faces.len(),
            1,
            "an interior valence-3 corner bevel creates exactly one cap face"
        );
        let _ = mesh;
        result.faces[0]
    }

    /// BEVEL.md §6.6 regime 1: `material >= 0` stamps that slot uniformly on the
    /// new cap, while the surviving re-stitched faces keep their own slot (§6.6.2).
    #[test]
    fn material_explicit_stamps_new_cap_and_leaves_survivors() {
        let mut mesh = LoopMesh::new();
        mesh_add_cube(&mut mesh);
        let v0 = vert_at(&mesh, Vec3::new(1.0, -1.0, -1.0));

        let result = mesh.bevel(
            &[v0.into()],
            BevelOptions {
                offset: 0.25,
                material: 3,
                ..Default::default()
            },
        );

        let cap = beveled_corner_cap(&mesh, &result);
        assert_eq!(
            mesh.face_material_index(cap),
            3,
            "explicit material=3 stamps the new cap (§6.6)"
        );

        // Every surviving (non-cap) face keeps the cube's default slot 0 — never
        // re-materialised (§6.6.2).
        for face in mesh.face_keys() {
            if face == cap {
                continue;
            }
            assert_eq!(
                mesh.face_material_index(face),
                0,
                "surviving face {face:?} keeps its own slot 0 (§6.6.2), not the stamp"
            );
        }
        mesh.validate_topology();
    }

    /// BEVEL.md §6.6.2 invariant: with `material >= 0` the multiset of NEW-face
    /// material slots is `{material}` (one per new face) regardless of segment
    /// count — pinned here for the single-cap case at a non-zero slot distinct
    /// from the surviving faces.
    #[test]
    fn material_explicit_new_face_multiset_is_the_stamp() {
        let mut mesh = LoopMesh::new();
        mesh_add_cube(&mut mesh);
        let v0 = vert_at(&mesh, Vec3::new(1.0, -1.0, -1.0));

        let result = mesh.bevel(
            &[v0.into()],
            BevelOptions {
                offset: 0.25,
                material: 5,
                ..Default::default()
            },
        );

        let new_slots: Vec<i32> = result
            .faces
            .iter()
            .map(|&f| mesh.face_material_index(f))
            .collect();
        assert_eq!(new_slots, vec![5], "new faces all carry the stamped slot 5");
    }

    /// BEVEL.md §6.6 regime 2 / §6.6.1: `material < 0` makes the new cap inherit a
    /// representative *incident face of `v`*'s slot — never a global default and
    /// never slot 0 unless an adjacent face genuinely uses slot 0. Decorating
    /// every cube face with a distinct non-zero slot makes the inherit source
    /// observable: the cap's slot is one of vert 0's incident faces' decorated
    /// slots.
    #[test]
    fn material_inherit_takes_adjacent_face_slot() {
        let mut mesh = LoopMesh::new();
        mesh_add_cube(&mut mesh);

        // Decorate the six faces with distinct non-zero slots {1..=6}.
        let faces: Vec<FaceKey> = mesh.face_keys();
        for (index, &face) in faces.iter().enumerate() {
            mesh.set_face_material_index(face, (index + 1) as i32);
        }

        let v0 = vert_at(&mesh, Vec3::new(1.0, -1.0, -1.0));
        // The slots of vert 0's incident faces, captured pristine — the §6.6.1
        // candidate representatives the cap may inherit from.
        let incident_slots: Vec<i32> = mesh
            .vert_face_keys(v0)
            .iter()
            .map(|&f| mesh.face_material_index(f))
            .collect();

        let result = mesh.bevel(
            &[v0.into()],
            BevelOptions {
                offset: 0.25,
                material: -1,
                ..Default::default()
            },
        );

        let cap = beveled_corner_cap(&mesh, &result);
        let cap_slot = mesh.face_material_index(cap);
        assert_ne!(cap_slot, 0, "the inherited cap slot is never the default 0 (§6.6.1)");
        assert!(
            incident_slots.contains(&cap_slot),
            "the cap inherits a representative incident face of v's slot ({incident_slots:?}), got {cap_slot}"
        );
        mesh.validate_topology();
    }

    /// BEVEL.md §6.6: on a uniformly-materialed input the `-1` inherit yields a
    /// uniformly-materialed result — the cap takes the one shared slot, so the
    /// per-face tie-break (§6.6.1) is unobservable, the common case the scaffold
    /// drives.
    #[test]
    fn material_inherit_uniform_input_yields_uniform_result() {
        let mut mesh = LoopMesh::new();
        mesh_add_cube(&mut mesh);
        for face in mesh.face_keys() {
            mesh.set_face_material_index(face, 4);
        }
        let v0 = vert_at(&mesh, Vec3::new(1.0, -1.0, -1.0));

        let result = mesh.bevel(
            &[v0.into()],
            BevelOptions {
                offset: 0.25,
                material: -1,
                ..Default::default()
            },
        );

        for face in mesh.face_keys() {
            assert_eq!(
                mesh.face_material_index(face),
                4,
                "uniform slot-4 input → uniform slot-4 result under -1 inherit (§6.6)"
            );
        }
        let _ = result;
        mesh.validate_topology();
    }

    /// The default `BevelOptions.material` is `-1` (inherit): a default-options
    /// bevel of a uniformly slot-2 cube leaves the cap at slot 2, confirming the
    /// inherit regime is the default and matches the explicit `-1` path.
    #[test]
    fn material_default_is_inherit() {
        let mut mesh = LoopMesh::new();
        mesh_add_cube(&mut mesh);
        for face in mesh.face_keys() {
            mesh.set_face_material_index(face, 2);
        }
        let v0 = vert_at(&mesh, Vec3::new(1.0, -1.0, -1.0));

        let result = mesh.bevel(
            &[v0.into()],
            BevelOptions {
                offset: 0.25,
                ..Default::default()
            },
        );

        let cap = beveled_corner_cap(&mesh, &result);
        assert_eq!(
            mesh.face_material_index(cap),
            2,
            "default material is -1 (inherit), so the cap takes the adjacent slot 2"
        );
    }

    // ---- mark_seam / mark_sharp (BEVEL.md §6.7) ----

    /// The chamfer edge between the ring verts at `near_a` and `near_b`, by
    /// position. Panics if either ring vert or the edge between them is absent.
    fn chamfer_edge(mesh: &LoopMesh, near_a: Vec3, near_b: Vec3) -> EdgeKey {
        let a = vert_at(mesh, near_a);
        let b = vert_at(mesh, near_b);
        mesh.query_edge_exists(a, b)
            .expect("the chamfer edge between these two ring verts exists")
    }

    /// BEVEL.md §6.7.1.1: a corner segment (chamfer edge) carries the inherited
    /// seam flag **only when both flanking owning edges are seamed**. A *single*
    /// seamed incident edge of a beveled corner therefore flags **no** chamfer
    /// edge — the contiguity AND suppresses the dead-end spur the plain §6.5.1 copy
    /// would have left on the two chamfers touching that edge's ring vert.
    #[test]
    fn single_seamed_incident_edge_flags_no_chamfer() {
        let mut mesh = LoopMesh::new();
        mesh_add_cube(&mut mesh);

        // Seam only e0 = {v0, v1} of corner v0 = (1,-1,-1).
        let p0 = Vec3::new(1.0, -1.0, -1.0);
        let e0 = cube_edge(&mesh, p0, Vec3::new(-1.0, -1.0, -1.0));
        mesh.set_edge_seam(e0, true);

        let v0 = vert_at(&mesh, p0);
        mesh.bevel(
            &[v0.into()],
            BevelOptions {
                offset: 0.5,
                ..Default::default()
            },
        );

        // Ring verts: r_e0 = (0.5,-1,-1), r_e3 = (1,-0.5,-1), r_e8 = (1,-1,-0.5).
        let r_e0 = Vec3::new(0.5, -1.0, -1.0);
        let r_e3 = Vec3::new(1.0, -0.5, -1.0);
        let r_e8 = Vec3::new(1.0, -1.0, -0.5);

        // Every chamfer has exactly one seamed owner, so none is seamed (§6.7.1.1).
        for &(a, b) in &[(r_e0, r_e3), (r_e3, r_e8), (r_e8, r_e0)] {
            let chamfer = chamfer_edge(&mesh, a, b);
            assert!(
                !mesh.edge_seam(chamfer),
                "chamfer {a:?}-{b:?} has one seamed owner only; the §6.7.1.1 AND \
                 suppresses the seam spur",
            );
        }

        // The surviving re-pointed e0 keeps its own seam (mechanism (1) item 1).
        let r_e0_vert = vert_at(&mesh, r_e0);
        assert!(
            mesh.edge_seam(e0) && mesh.edge_keys().contains(&e0),
            "the surviving re-pointed e0 keeps its seam",
        );
        assert!(
            mesh.edge_vert_keys(e0).contains(&r_e0_vert),
            "e0 is re-pointed onto its ring vert r_e0",
        );
        mesh.validate_topology();
    }

    /// BEVEL.md §6.7.1.1: two adjacent seamed incident edges flag the **one**
    /// chamfer edge running directly between their two ring verts (its two flanking
    /// owners are both seamed), and **only** that one — the other two chamfers each
    /// have an unseamed owner and stay unseamed. The sharp bit follows the
    /// identical AND on its own bit.
    #[test]
    fn two_adjacent_seamed_incident_edges_flag_their_one_chamfer() {
        let mut mesh = LoopMesh::new();
        mesh_add_cube(&mut mesh);

        let p0 = Vec3::new(1.0, -1.0, -1.0);
        let e0 = cube_edge(&mesh, p0, Vec3::new(-1.0, -1.0, -1.0)); // -> r_e0
        let e3 = cube_edge(&mesh, p0, Vec3::new(1.0, 1.0, -1.0)); // -> r_e3
        // Seam e0 + e3; sharp the same pair to cover the sharp bit identically.
        mesh.set_edge_seam(e0, true);
        mesh.set_edge_seam(e3, true);
        mesh.set_edge_sharp(e0, true);
        mesh.set_edge_sharp(e3, true);

        let v0 = vert_at(&mesh, p0);
        mesh.bevel(
            &[v0.into()],
            BevelOptions {
                offset: 0.5,
                ..Default::default()
            },
        );

        let r_e0 = Vec3::new(0.5, -1.0, -1.0);
        let r_e3 = Vec3::new(1.0, -0.5, -1.0);
        let r_e8 = Vec3::new(1.0, -1.0, -0.5);

        // The chamfer between r_e0 and r_e3 has both owners (e0, e3) seamed/sharp.
        let between = chamfer_edge(&mesh, r_e0, r_e3);
        assert!(mesh.edge_seam(between), "the (r_e0,r_e3) chamfer is seamed");
        assert!(mesh.edge_sharp(between), "the (r_e0,r_e3) chamfer is sharp");

        // The other two chamfers each have one unseamed owner (e8) and stay clear.
        for &(a, b) in &[(r_e3, r_e8), (r_e8, r_e0)] {
            let chamfer = chamfer_edge(&mesh, a, b);
            assert!(
                !mesh.edge_seam(chamfer),
                "chamfer {a:?}-{b:?} has an unseamed owner; not seamed (§6.7.1.1)",
            );
            assert!(
                !mesh.edge_sharp(chamfer),
                "chamfer {a:?}-{b:?} has a smooth owner; not sharp (§6.7.1.1)",
            );
        }
        mesh.validate_topology();
    }

    /// BEVEL.md §6.7.1.1: all three incident edges seamed flags **every** chamfer
    /// (each chamfer's two owners are both seamed) — the cap's ring is a continuous
    /// seam loop.
    #[test]
    fn all_incident_edges_seamed_flag_every_chamfer() {
        let mut mesh = LoopMesh::new();
        mesh_add_cube(&mut mesh);

        let p0 = Vec3::new(1.0, -1.0, -1.0);
        for far in [
            Vec3::new(-1.0, -1.0, -1.0),
            Vec3::new(1.0, 1.0, -1.0),
            Vec3::new(1.0, -1.0, 1.0),
        ] {
            let edge = cube_edge(&mesh, p0, far);
            mesh.set_edge_seam(edge, true);
        }

        let v0 = vert_at(&mesh, p0);
        mesh.bevel(
            &[v0.into()],
            BevelOptions {
                offset: 0.5,
                ..Default::default()
            },
        );

        let r_e0 = Vec3::new(0.5, -1.0, -1.0);
        let r_e3 = Vec3::new(1.0, -0.5, -1.0);
        let r_e8 = Vec3::new(1.0, -1.0, -0.5);
        for &(a, b) in &[(r_e0, r_e3), (r_e3, r_e8), (r_e8, r_e0)] {
            let chamfer = chamfer_edge(&mesh, a, b);
            assert!(
                mesh.edge_seam(chamfer),
                "all owners seamed -> chamfer {a:?}-{b:?} is seamed (§6.7.1.1)",
            );
        }
        mesh.validate_topology();
    }

    /// BEVEL.md §6.7.2: `mark_seam` / `mark_sharp` are **inert** for vertex bevel —
    /// no two beveled edges meet at the corner, so mechanism (2) never fires. The
    /// parameter-ON run must produce the same topology *and* the same per-edge seam
    /// / sharp flags as the parameter-OFF run; the flags ride mechanism (1)
    /// identically either way.
    #[test]
    fn vertex_bevel_mark_seam_sharp_is_inert() {
        // Seam/sharp two adjacent incident edges so mechanism (1) *does* flag a
        // chamfer — the inertness claim must hold even when flags are present.
        let build = |mark: bool| -> Vec<((u32, u32, u32), (u32, u32, u32), bool, bool)> {
            let mut mesh = LoopMesh::new();
            mesh_add_cube(&mut mesh);
            let p0 = Vec3::new(1.0, -1.0, -1.0);
            let e0 = cube_edge(&mesh, p0, Vec3::new(-1.0, -1.0, -1.0));
            let e3 = cube_edge(&mesh, p0, Vec3::new(1.0, 1.0, -1.0));
            mesh.set_edge_seam(e0, true);
            mesh.set_edge_seam(e3, true);
            mesh.set_edge_sharp(e0, true);
            mesh.set_edge_sharp(e3, true);

            let v0 = vert_at(&mesh, p0);
            mesh.bevel(
                &[v0.into()],
                BevelOptions {
                    offset: 0.5,
                    mark_seam: mark,
                    mark_sharp: mark,
                    ..Default::default()
                },
            );

            // Read every edge's (endpoints, seam, sharp) so on/off are comparable
            // by geometry (keys differ run-to-run; positions are stable). Positions
            // are bit-cast to ordered integer keys for a stable sort.
            let key = |p: Vec3| (p.x.to_bits(), p.y.to_bits(), p.z.to_bits());
            let mut flags: Vec<((u32, u32, u32), (u32, u32, u32), bool, bool)> = mesh
                .edge_keys()
                .into_iter()
                .map(|e| {
                    let [a, b] = mesh.edge_vert_keys(e);
                    let mut ka = key(mesh.vertex_position(a));
                    let mut kb = key(mesh.vertex_position(b));
                    if kb < ka {
                        std::mem::swap(&mut ka, &mut kb);
                    }
                    (ka, kb, mesh.edge_seam(e), mesh.edge_sharp(e))
                })
                .collect();
            flags.sort();
            flags
        };

        let on = build(true);
        let off = build(false);
        assert_eq!(
            on, off,
            "mark_seam / mark_sharp are inert for vertex bevel (§6.7.2): on == off",
        );
    }

    /// BEVEL.md §6.7.3: `mark_seam` / `mark_sharp` are inert for an isolated single
    /// beveled edge too — each endpoint sees one beveled edge, failing mechanism
    /// (2)'s "two or more" precondition. Parameter-ON topology + flags must equal
    /// parameter-OFF.
    #[test]
    fn isolated_edge_bevel_mark_seam_is_inert() {
        let build = |mark: bool| -> Vec<((u32, u32, u32), (u32, u32, u32), bool)> {
            let mut mesh = LoopMesh::new();
            mesh_add_cube(&mut mesh);
            let edge = cube_edge(
                &mesh,
                Vec3::new(1.0, -1.0, -1.0),
                Vec3::new(-1.0, -1.0, -1.0),
            );
            mesh.set_edge_seam(edge, true);

            mesh.bevel(
                &[edge.into()],
                BevelOptions {
                    offset: 0.25,
                    affect: BevelAffect::Edges,
                    mark_seam: mark,
                    ..Default::default()
                },
            );

            let key = |p: Vec3| (p.x.to_bits(), p.y.to_bits(), p.z.to_bits());
            let mut flags: Vec<((u32, u32, u32), (u32, u32, u32), bool)> = mesh
                .edge_keys()
                .into_iter()
                .map(|e| {
                    let [a, b] = mesh.edge_vert_keys(e);
                    let mut ka = key(mesh.vertex_position(a));
                    let mut kb = key(mesh.vertex_position(b));
                    if kb < ka {
                        std::mem::swap(&mut ka, &mut kb);
                    }
                    (ka, kb, mesh.edge_seam(e))
                })
                .collect();
            flags.sort();
            flags
        };

        assert_eq!(
            build(true),
            build(false),
            "mark_seam is inert for an isolated single beveled edge (§6.7.3)",
        );
    }

    /// Round a position to a stable integer key for set membership (the bevel
    /// positions here are exact multiples of 0.25 up to f32 noise).
    fn pos_key(p: Vec3) -> (i32, i32, i32) {
        let q = |x: f32| (x * 4.0).round() as i32;
        (q(p.x), q(p.y), q(p.z))
    }

    /// An undirected edge as the sorted pair of its endpoint position keys.
    fn edge_pos_pair(
        mesh: &LoopMesh,
        edge: EdgeKey,
    ) -> ((i32, i32, i32), (i32, i32, i32)) {
        let [a, b] = mesh.edge_vert_keys(edge);
        let mut ka = pos_key(mesh.vertex_position(a));
        let mut kb = pos_key(mesh.vertex_position(b));
        if kb < ka {
            std::mem::swap(&mut ka, &mut kb);
        }
        (ka, kb)
    }

    /// The set of NEW (result) edges carrying `flag` (seam or sharp), as sorted
    /// endpoint position-key pairs — an ordering-independent fingerprint.
    fn flagged_new_edge_set(
        mesh: &LoopMesh,
        result: &BevelResult,
        flag: impl Fn(&LoopMesh, EdgeKey) -> bool,
    ) -> std::collections::BTreeSet<((i32, i32, i32), (i32, i32, i32))> {
        result
            .edges
            .iter()
            .filter(|&&e| mesh.edge_keys().contains(&e) && flag(mesh, e))
            .map(|&e| edge_pos_pair(mesh, e))
            .collect()
    }

    /// The three-edge cube-corner junction the bridging arm pins (BEVEL.md §6.7.4):
    /// v0 = [1,-1,-1], edges e01/e03/e04 beveled at offset 0.25, with the two
    /// flanking edges e01 (toward [-1,-1,-1]) and e04 (toward [1,-1,1]) flagged and
    /// e03 (toward [1,1,-1]) the unflagged gap. Returns the beveled mesh + result.
    fn bevel_three_edge_junction_with_flag(
        set_flag: impl Fn(&mut LoopMesh, EdgeKey),
        options: BevelOptions,
    ) -> (LoopMesh, BevelResult) {
        let mut mesh = LoopMesh::new();
        mesh_add_cube(&mut mesh);
        let p0 = Vec3::new(1.0, -1.0, -1.0);
        let e01 = cube_edge(&mesh, p0, Vec3::new(-1.0, -1.0, -1.0));
        let e03 = cube_edge(&mesh, p0, Vec3::new(1.0, 1.0, -1.0));
        let e04 = cube_edge(&mesh, p0, Vec3::new(1.0, -1.0, 1.0));
        set_flag(&mut mesh, e01);
        set_flag(&mut mesh, e04);
        let result = mesh.bevel(&[e01.into(), e03.into(), e04.into()], options);
        (mesh, result)
    }

    type EdgeKeySet = std::collections::BTreeSet<((i32, i32, i32), (i32, i32, i32))>;

    struct JunctionExpectedSets {
        /// All three patch ring segments (the full boundvert ring).
        ring: EdgeKeySet,
        /// The single direct patch segment spanning the unseamed e03 gap edge.
        gap_segment: EdgeKeySet,
        /// The four parallel strip long edges of the seamed e01 / e04 (mechanism 1).
        parallel: EdgeKeySet,
    }

    /// The cube corner v0's boundvert ring (the three patch ring positions at
    /// offset 0.25), the single direct patch segment spanning the e03 gap, and the
    /// four parallel strip long edges of the two seamed beveled edges e01 / e04.
    /// Position keys are in 0.25 units (`pos_key`).
    fn junction_expected_sets() -> JunctionExpectedSets {
        let k = |x: f32, y: f32, z: f32| pos_key(Vec3::new(x, y, z));
        let pair = |a: (i32, i32, i32), b: (i32, i32, i32)| {
            if a <= b {
                (a, b)
            } else {
                (b, a)
            }
        };
        // Boundverts of v0's junction at offset 0.25.
        let b0 = k(0.75, -0.75, -1.0);
        let b1 = k(0.75, -1.0, -0.75);
        let b2 = k(1.0, -0.75, -0.75);
        // The three patch ring segments.
        let ring: std::collections::BTreeSet<_> =
            [pair(b0, b1), pair(b0, b2), pair(b1, b2)].into_iter().collect();
        // The §6.7.4 single direct patch segment spanning the unseamed gap edge e03:
        // the edge joining e03's two flanking boundverts (both carry y=-0.75, the
        // slide along e03), i.e. [0.75,-0.75,-1] and [1,-0.75,-0.75].
        let gap_segment: std::collections::BTreeSet<_> = [pair(b0, b2)].into_iter().collect();
        // The four parallel strip long edges of e01 (to its two v1-strip verts) and
        // e04 (to its two v4-strip verts) — mechanism (1).
        let parallel: std::collections::BTreeSet<_> = [
            pair(b0, k(-1.0, -0.75, -1.0)),
            pair(b1, k(-1.0, -1.0, -0.75)),
            pair(b1, k(0.75, -1.0, 1.0)),
            pair(b2, k(1.0, -0.75, 1.0)),
        ]
        .into_iter()
        .collect();
        JunctionExpectedSets {
            ring,
            gap_segment,
            parallel,
        }
    }

    /// BEVEL.md §6.7.1 mechanism (1) is UNCONDITIONAL — with `mark_seam = false`
    /// the two seamed beveled edges still seam their four parallel strip long edges
    /// (the edges descending from their re-stitched faces). The patch ring segments
    /// stay unseamed (no parameter-gated bridging). This pins the unconditional
    /// inheritance path independently of the parameter.
    #[test]
    fn junction_mechanism_one_inherits_with_mark_seam_off() {
        let (mesh, result) = bevel_three_edge_junction_with_flag(
            |m, e| m.set_edge_seam(e, true),
            BevelOptions {
                offset: 0.25,
                affect: BevelAffect::Edges,
                mark_seam: false,
                ..Default::default()
            },
        );
        let sets = junction_expected_sets();
        let seamed = flagged_new_edge_set(&mesh, &result, |m, e| m.edge_seam(e));
        assert_eq!(
            seamed, sets.parallel,
            "mark_seam=false still seams exactly the four parallel strip long edges \
             of e01/e04 via mechanism (1) (§6.7.1); no patch ring segment is seamed",
        );
        assert!(
            seamed.is_disjoint(&sets.ring),
            "no patch ring (bridging) segment may be seamed with mark_seam off",
        );
        mesh.validate_topology();
    }

    /// BEVEL.md §6.7.4 mechanism (2): with `mark_seam = true` the seam bridges the
    /// e03 gap by the **single** direct patch segment spanning e03 (the edge joining
    /// e03's two flanking boundverts) — *not* the full ring. The other two patch
    /// sides (spanning the seamed e01 / e04, where their strips already meet) stay
    /// unflagged. Pins the exact bridged set = four mechanism-(1) parallel edges +
    /// the one gap segment, and asserts the over-bridging full-ring result is wrong.
    #[test]
    fn junction_mechanism_two_bridges_single_gap_segment_with_mark_seam_on() {
        let (mesh, result) = bevel_three_edge_junction_with_flag(
            |m, e| m.set_edge_seam(e, true),
            BevelOptions {
                offset: 0.25,
                affect: BevelAffect::Edges,
                mark_seam: true,
                ..Default::default()
            },
        );
        let sets = junction_expected_sets();
        let mut expected = sets.parallel.clone();
        expected.extend(sets.gap_segment.iter().copied());
        let seamed = flagged_new_edge_set(&mesh, &result, |m, e| m.edge_seam(e));
        assert_eq!(
            seamed, expected,
            "mark_seam=true seams the four mechanism-(1) parallel edges PLUS the \
             single direct patch segment spanning the unseamed e03 gap (§6.7.4) — \
             NOT the full ring",
        );
        // The two patch sides spanning the seamed e01 / e04 stay unflagged.
        let other_ring: std::collections::BTreeSet<_> =
            sets.ring.difference(&sets.gap_segment).copied().collect();
        assert!(
            seamed.is_disjoint(&other_ring),
            "the patch sides spanning the seamed edges' own sides are not bridged",
        );
        mesh.validate_topology();
    }

    /// BEVEL.md §6.7.4: `mark_sharp` runs the identical single-gap-segment bridging
    /// on the sharp bit, independently of `mark_seam`. On bridges the one gap segment
    /// (+ four parallel); off bridges nothing (only the four parallel, mechanism 1).
    #[test]
    fn junction_mark_sharp_bridges_single_gap_segment_and_inherits_unconditionally() {
        let sets = junction_expected_sets();

        let (mesh_off, result_off) = bevel_three_edge_junction_with_flag(
            |m, e| m.set_edge_sharp(e, true),
            BevelOptions {
                offset: 0.25,
                affect: BevelAffect::Edges,
                mark_sharp: false,
                ..Default::default()
            },
        );
        let sharp_off = flagged_new_edge_set(&mesh_off, &result_off, |m, e| m.edge_sharp(e));
        assert_eq!(
            sharp_off, sets.parallel,
            "mark_sharp=false still sharpens the four parallel strip long edges \
             (mechanism (1), §6.7.1), no ring segment",
        );

        let (mesh_on, result_on) = bevel_three_edge_junction_with_flag(
            |m, e| m.set_edge_sharp(e, true),
            BevelOptions {
                offset: 0.25,
                affect: BevelAffect::Edges,
                mark_sharp: true,
                ..Default::default()
            },
        );
        let mut expected = sets.parallel.clone();
        expected.extend(sets.gap_segment.iter().copied());
        let sharp_on = flagged_new_edge_set(&mesh_on, &result_on, |m, e| m.edge_sharp(e));
        assert_eq!(
            sharp_on, expected,
            "mark_sharp=true bridges the single gap segment on the sharp bit (§6.7.4)",
        );
        mesh_off.validate_topology();
        mesh_on.validate_topology();
    }

    /// Drive `bridge_junction_flags` directly over a synthetic `s`-edge junction:
    /// build a ring of `s` boundverts joined by `s` patch ring segments, plus `s`
    /// stub "beveled edges" `g_0..g_(s-1)` whose seamed state is `seamed[k]`
    /// (`opening_edges[k] = g_k`, so segment `(B_(k-1), B_k)` is crossed by `g_k`).
    /// Returns the set of ring segment indices `m` (segment `(B_(m-1), B_m)`) that
    /// end up seamed. This exercises the §6.7.4 gap rule independent of the full
    /// junction surgery (which the ≥4-edge regime does not yet drive end-to-end).
    fn bridge_synthetic_junction(seamed: &[bool]) -> std::collections::BTreeSet<usize> {
        let s = seamed.len();
        let mut mesh = LoopMesh::new();
        // Ring boundverts on a unit circle, plus the ring edges between consecutive
        // boundverts; opening_edges are separate stub edges to detached stub verts.
        let mut boundverts = Vec::with_capacity(s);
        for k in 0..s {
            let theta = std::f32::consts::TAU * (k as f32) / (s as f32);
            boundverts.push(mesh.create_vertex(Vec3::new(theta.cos(), theta.sin(), 0.0)));
        }
        for k in 0..s {
            mesh.create_edge(
                [boundverts[k], boundverts[(k + 1) % s]],
                EdgeCreateFlags::NONE,
            );
        }
        // opening_edges[k] = g_k: a stub edge whose seam state encodes `seamed[k]`.
        let mut opening_edges = Vec::with_capacity(s);
        let mut edge_seam_state = Vec::with_capacity(s);
        for k in 0..s {
            let stub = mesh.create_vertex(Vec3::new(2.0 + k as f32, 0.0, 0.0));
            let g = mesh.create_edge([boundverts[k], stub], EdgeCreateFlags::NONE);
            opening_edges.push(g);
            edge_seam_state.push((g, seamed[k], false));
        }

        let patch = JunctionPatch {
            vertex: boundverts[0],
            boundverts: boundverts.clone(),
            opening_edges,
        };
        mesh.bridge_junction_flags(&patch, &edge_seam_state, true, false);

        // Read back which ring segment (B_(m-1), B_m) ended up seamed.
        (0..s)
            .filter(|&m| {
                let a = boundverts[(m + s - 1) % s];
                let b = boundverts[m];
                mesh.query_edge_exists(a, b)
                    .map(|e| mesh.edge_seam(e))
                    .unwrap_or(false)
            })
            .collect()
    }

    /// BEVEL.md §6.7.4 ≥4-edge generalization. Each case pins which patch ring
    /// segments the gap rule bridges (segment `m` spans gap-edge `g_m`).
    #[test]
    fn junction_gap_rule_generalizes_beyond_three_edges() {
        // Two OPPOSITE seamed edges (g0,g2) at a valence-4 junction enclose two
        // separate length-1 gaps (g1 and g3); BOTH are bridged (the seam closes into
        // a loop). Segments spanning g1 and g3 are seamed; nothing else.
        assert_eq!(
            bridge_synthetic_junction(&[true, false, true, false]),
            [1, 3].into_iter().collect(),
            "two opposite seamed edges bridge BOTH length-1 gaps, not the full ring",
        );

        // Two ADJACENT seamed edges (g0,g1): the short side between them (the sector
        // boundvert they share) is a length-0 gap and bridges nothing; the far
        // length-2 gap (g2,g3) is bridged by its two consecutive direct segments.
        assert_eq!(
            bridge_synthetic_junction(&[true, true, false, false]),
            [2, 3].into_iter().collect(),
            "two adjacent seamed edges bridge only the far length-2 gap (2 segments)",
        );

        // THREE seamed edges (one unseamed g3): a single length-1 gap; only the
        // segment spanning g3 is bridged.
        assert_eq!(
            bridge_synthetic_junction(&[true, true, true, false]),
            [3].into_iter().collect(),
            "three seamed edges bridge the one length-1 gap",
        );

        // The 3-edge cube-corner case: g1 the unseamed gap flanked by g0,g2 seamed —
        // exactly the segment spanning g1 is bridged (the single direct segment, NOT
        // the full ring).
        assert_eq!(
            bridge_synthetic_junction(&[true, false, true]),
            [1].into_iter().collect(),
            "single-gap 3-edge junction bridges exactly one segment (no over-bridge)",
        );

        // No gap: a lone seamed edge bridges nothing.
        assert!(bridge_synthetic_junction(&[true, false, false]).is_empty());
        // No gap: all seamed bridges nothing (the strips already carry the seam).
        assert!(bridge_synthetic_junction(&[true, true, true]).is_empty());
        // No gap: none seamed bridges nothing.
        assert!(bridge_synthetic_junction(&[false, false, false]).is_empty());
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
    let mut current = neighbours.iter().find_map(|&(a, b)| {
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
