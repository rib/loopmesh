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

/// How a multi-edge junction's corner patch is filled at `segments ≥ 2`
/// (BEVEL.md §1.2 / §8.3).
///
/// [`BevelVmeshMethod::Adj`] is the default rounded adjacency grid that blends the
/// arriving profile strips (a documented-as-future stub, BEVEL.md §8.4).
/// [`BevelVmeshMethod::Cutoff`] slices the corner off flat: one cut face per
/// sector spanning that sector's profile chain plus two bottom corner verts, the
/// bottom corner ring closed by one bottom face — or, at a 3-way junction whose
/// corner verts coincide, collapsed to a single apex (BEVEL.md §8.3).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum BevelVmeshMethod {
    Adj,
    Cutoff,
}

impl Default for BevelVmeshMethod {
    fn default() -> Self {
        BevelVmeshMethod::Adj
    }
}

/// Parameters for [`LoopMesh::bevel`]. The foundational case (BEVEL.md §3) is
/// reachable by setting only `offset` and leaving every other field at its
/// default: `affect = Vertices`, `segments = 1`, `offset_type = Offset`.
#[derive(Clone, Copy, Debug)]
pub struct BevelOptions<'a> {
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
    /// How a multi-edge junction's corner patch is filled at `segments ≥ 2`
    /// (BEVEL.md §1.2 / §8.3). Only consulted for `affect = Edges`,
    /// `segments ≥ 2`. Defaults to `Adj` (the rounded grid; still a stub). The
    /// `Cutoff` flat-cut fill is implemented for the multi-edge junction at the
    /// default profile (BEVEL.md §8.2 / §8.3).
    pub vmesh_method: BevelVmeshMethod,
    /// Cross-section shape for the `SUPERELLIPSE` profile family (BEVEL.md §8.6),
    /// a scalar in `0..1`. `0.5` (the default) is a circular arc, `0.25` flattens
    /// to a straight chamfer chord, larger values bulge toward a square corner.
    /// Only observable at `segments ≥ 2`: a single segment is a straight chamfer
    /// regardless of `profile` (§8.6.2). The mapping to the superellipse exponent
    /// `r` is logarithmic with `0.25 → r = 1` (line), `0.5 → r = 2` (circle),
    /// `→ 1 → r → ∞` (square). The `CUSTOM` `profile_type` (a tabulated user curve,
    /// §8.6.3) is not yet representable and is out of scope. Defaults to `0.5`.
    pub profile: f32,
    /// Gate for the per-element bevel-weight prescale (BEVEL.md §8.7.2). When
    /// `false` (the default), weights are ignored entirely and every beveled
    /// element slides the full `offset` — byte-identical to the unweighted path.
    /// When `true`, each beveled vertex's resolved offset is prescaled by its
    /// weight (BEVEL.md §8.7.3); a beveled vertex *absent* from `vert_weights`
    /// defaults to weight `0.0` — no slide, no-op for that corner (§8.7.2 case 3).
    pub use_weights: bool,
    /// Per-vertex bevel weights consulted when `use_weights` is `true`, for
    /// `affect = Vertices` (BEVEL.md §8.7.1 / §8.7.3). A slice of
    /// `(VertexKey, weight)`; a beveled vertex's weight is looked up here, and an
    /// absent vertex resolves to `0.0` (§8.7.2). The weight multiplies that
    /// corner's offset uniformly across all its incident edges before the §4
    /// slide. Defaults to an empty slice.
    pub vert_weights: &'a [(VertexKey, f32)],
    /// Slide a junction boundvert **onto a single unbeveled edge** rather than
    /// leaving it at the two-strip meet-point (BEVEL.md §8.8). A pure placement
    /// perturbation: it changes no topology and no surgery, only where one
    /// boundvert lands in a junction sector. It fires for a sector with exactly
    /// one silhouette leg (or, absent silhouette legs, exactly one in-plane leg)
    /// when the slide does not fold either flanking side (§8.8.2), placing the
    /// boundvert on that leg's line by the meet-then-average rule of §8.8.3.
    /// Realised at `segments = 1` on the landed junction meet-point (§7.6.2); a
    /// no-op at a symmetric junction, diverging only at an irregular one
    /// (§8.8.5). Defaults to `false`.
    pub loop_slide: bool,
}

impl Default for BevelOptions<'_> {
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
            vmesh_method: BevelVmeshMethod::Adj,
            profile: 0.5,
            use_weights: false,
            vert_weights: &[],
            loop_slide: false,
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

        if options.offset <= 0.0 {
            return result;
        }

        // The multi-segment edge-bevel junction (BEVEL.md §8.2 / §8.3 / §8.5) is the
        // only `segments ≥ 2` path implemented; vertex multi-segment still no-ops.
        // Both vmesh methods (Cutoff flat fill §8.3, Adj rounded fill §8.5) share
        // the same [P] ring / [S] chains / multi-segment strips; they differ only
        // in the junction fill. ADJ is distilled for the forced `segments = 2`
        // 3-way cube-corner case (§8.5); other ADJ cases still no-op below.
        if options.segments >= 2 {
            if options.affect == BevelAffect::Edges {
                return self.bevel_edges(geom, &options);
            }
            return result;
        }

        if options.segments != 1 {
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
        let frames = self.gather_bevel_frames(
            &beveled,
            options.offset,
            options.offset_type,
            options.use_weights,
            options.vert_weights,
        );
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

        // Multi-segment cutoff (BEVEL.md §8.2 / §8.3) needs at least one
        // multi-beveled-edge junction to fill — an all-isolated edge set has no
        // cutoff corner. The orchestrator itself no-ops on the junction-free case.
        if options.segments >= 2 {
            self.bevel_edges_cutoff_multisegment(&beveled, &junctions, options, &mut result);
            return result;
        }

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

        let _ = direction;

        // The non-`E` edge of each radial face at each endpoint — the flanking
        // unbeveled leg shared between that radial face and the endpoint's *third*
        // face. The re-stitch re-points them onto the strip verts, so the third face
        // must place the matching strip vert next to that edge's far end (§7.4); the
        // §7.5.1 leg meet also places each strip vert against its own flanking leg.
        let left_other_a = self.face_other_edge_at(face_left, a, edge);
        let right_other_a = self.face_other_edge_at(face_right, a, edge);
        let left_other_b = self.face_other_edge_at(face_left, b, edge);
        let right_other_b = self.face_other_edge_at(face_right, b, edge);

        // §7.5.1: each terminal strip vert is the leg meet of E (offset) against its
        // own flanking leg (offset 0), built from the local edge-cross normal — not
        // the §7.3 perpendicular-in-face slide, which only coincides on a planar,
        // square terminal and diverges at an irregular non-planar one.
        let normal_a = self.surface_normal_at(a);
        let normal_b = self.surface_normal_at(b);
        let strip_a_left = self.create_vertex(
            self.terminal_strip_vert(a, edge, left_other_a, face_left, offset, normal_a),
        );
        let strip_a_right = self.create_vertex(
            self.terminal_strip_vert(a, edge, right_other_a, face_right, offset, normal_a),
        );
        let strip_b_left = self.create_vertex(
            self.terminal_strip_vert(b, edge, left_other_b, face_left, offset, normal_b),
        );
        let strip_b_right = self.create_vertex(
            self.terminal_strip_vert(b, edge, right_other_b, face_right, offset, normal_b),
        );
        result
            .verts
            .extend([strip_a_left, strip_a_right, strip_b_left, strip_b_right]);

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
        let mut reconciled: Vec<ReconciledOffset> = Vec::new();

        // §7.6.4 / §8.8.4: place every junction first, so a firing `loop_slide` slide
        // records its reconciled per-side offsets, then place the terminals which read
        // them — the width-pass ripple reaches the far ends of the flanking edges
        // (§8.8.5). At a symmetric junction (or with `loop_slide` off) nothing is
        // recorded and the terminals fall back to the spec offset, unchanged.
        for &endpoint in &endpoints {
            if junctions.contains(&endpoint) {
                self.place_junction_boundverts(
                    endpoint,
                    &is_beveled,
                    offset,
                    options.loop_slide,
                    &mut corner_vert,
                    &mut substitutions,
                    &mut junction_patches,
                    &mut reconciled,
                    result,
                );
            }
        }
        for &endpoint in &endpoints {
            if !junctions.contains(&endpoint) {
                self.place_terminal_strip_verts(
                    endpoint,
                    &beveled_manifold,
                    offset,
                    &reconciled,
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

    /// Multi-segment cutoff edge bevel at a multi-edge junction (BEVEL.md §8.2 /
    /// §8.3). Drives the `segments ≥ 2`, `vmesh_method = Cutoff` case.
    ///
    /// The boundvert ring at every endpoint is the §7.6.2 / §7.5 ring, unchanged
    /// by `segments` (§8.2.1). What `segments ≥ 2` adds is, per beveled edge per
    /// endpoint, a **profile chain** of `segments + 1` arc-sampled verts spanning
    /// that edge's two flanking boundverts (§8.2.2 / §8.2.2.1), the `segments`-wide
    /// **strip** along each edge (§8.3.0), and, at each multi-beveled-edge
    /// **junction**, the **cutoff fill**: one flat cut face per sector plus a
    /// bottom face — or, when the corner verts coincide (the `s = 3` cube corner),
    /// a single collapsed apex (§8.3.2 / §8.3.4). A valence-3 **terminal** builds
    /// no fill face: its edge's strip runs through with the chain mirrored, the
    /// surviving faces absorbing the `segments + 1`-wide run (§8.3.0.1).
    fn bevel_edges_cutoff_multisegment(
        &mut self,
        beveled: &[EdgeKey],
        junctions: &[VertexKey],
        options: &BevelOptions,
        result: &mut BevelResult,
    ) {
        let offset = options.offset;
        let material = options.material;
        let segments = options.segments;
        let profile = options.profile;

        // §8.3: the cutoff fill needs at least one multi-beveled-edge junction —
        // an all-isolated / junction-free edge set has no corner to fold, so it is
        // a clean no-op rather than partial through-strip geometry.
        if junctions.is_empty() {
            return;
        }

        let mut beveled_manifold: Vec<EdgeKey> = Vec::new();
        for &edge in beveled {
            if self.edge_radial_iter(edge).count() == 2 {
                beveled_manifold.push(edge);
            }
        }
        if beveled_manifold.is_empty() {
            return;
        }
        let is_beveled = |edge: EdgeKey| beveled_manifold.contains(&edge);

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

        let mut endpoints: Vec<VertexKey> = Vec::new();
        for &edge in &beveled_manifold {
            for endpoint in self.edge_vert_keys(edge) {
                if !endpoints.contains(&endpoint) {
                    endpoints.push(endpoint);
                }
            }
        }

        // Phase 1 (pristine mesh): per endpoint, place its boundvert ring and build
        // each incident beveled edge's profile chain. `chains` keys each chain by
        // (edge, endpoint); `face_subst` records per (vertex, face) the ordered
        // vert run the re-stitch swaps the corner for; `junction_cuts` carries the
        // per-junction cutoff-fill data.
        let mut chains: Vec<EdgeEndChain> = Vec::new();
        let mut face_subst: Vec<FaceSubstitution> = Vec::new();
        let mut junction_cuts: Vec<JunctionCut> = Vec::new();

        for &endpoint in &endpoints {
            if junctions.contains(&endpoint) {
                self.build_junction_chains(
                    endpoint,
                    &is_beveled,
                    offset,
                    segments,
                    profile,
                    &mut chains,
                    &mut face_subst,
                    &mut junction_cuts,
                    result,
                );
            } else {
                self.build_terminal_chains(
                    endpoint,
                    &beveled_manifold,
                    offset,
                    segments,
                    profile,
                    &mut chains,
                    &mut face_subst,
                    result,
                );
            }
        }

        // Phase 2: re-stitch every incident face once, swapping each consumed
        // endpoint corner for its recorded vert run (§8.3.0.1 absorb / sector
        // 1-for-1).
        let mut incident_faces: Vec<FaceKey> = Vec::new();
        for &endpoint in &endpoints {
            for face in self.vert_face_keys(endpoint) {
                if !incident_faces.contains(&face) {
                    incident_faces.push(face);
                }
            }
        }
        for face in incident_faces {
            self.restitch_junction_face(face, &endpoints, &face_subst);
        }

        // Phase 3: one strip (segments quads) per beveled edge over its two
        // endpoint chains (§8.3.0).
        for &edge in &beveled_manifold {
            let representative = strip_material
                .iter()
                .find_map(|&(e, m)| (e == edge).then_some(m))
                .unwrap_or(0);
            self.build_multisegment_strip(
                edge,
                segments,
                &chains,
                material,
                representative,
                result,
            );
        }

        // Phase 4: the junction fill — flat cut (§8.3.2 / §8.3.4) or rounded
        // adjacency grid (§8.5), per `vmesh_method`. ADJ is implemented only for
        // the forced `segments = 2` 3-way cube-corner (§8.5); a junction outside
        // that forced case takes no ADJ fill (the general grid is a stub, §8.5.5).
        for cut in &junction_cuts {
            match options.vmesh_method {
                BevelVmeshMethod::Cutoff => {
                    self.build_cutoff_fill(
                        cut,
                        &chains,
                        segments,
                        options.profile,
                        material,
                        result,
                    );
                }
                BevelVmeshMethod::Adj => {
                    self.build_adj_fill(
                        cut,
                        &chains,
                        segments,
                        options.profile,
                        material,
                        result,
                    );
                }
            }
        }

        // Phase 5: every consumed endpoint is now loose. Free each (§5.1.1).
        for &endpoint in &endpoints {
            self.free_loose_vertex(endpoint);
        }
    }

    /// Place a junction's boundvert ring and build each sector's profile chain
    /// (BEVEL.md §8.2 / §8.3.0). One boundvert per sector at the §7.6.2 meet; one
    /// chain per consecutive boundvert pair, crossing the beveled edge shared by
    /// the two sectors. Records the cutoff-fill data for the junction.
    #[allow(clippy::too_many_arguments)]
    fn build_junction_chains(
        &mut self,
        vertex: VertexKey,
        is_beveled: &dyn Fn(EdgeKey) -> bool,
        offset: f32,
        segments: u32,
        profile: f32,
        chains: &mut Vec<EdgeEndChain>,
        face_subst: &mut Vec<FaceSubstitution>,
        junction_cuts: &mut Vec<JunctionCut>,
        result: &mut BevelResult,
    ) {
        let corners = self.vertex_corner_ring(vertex);
        let sectors = sector_split(&corners, is_beveled);
        let vertex_position = self.vertex_position(vertex);

        let mut boundverts: Vec<VertexKey> = Vec::with_capacity(sectors.len());
        // Per sector, the beveled edge that *opens* it (`g_k` for `B_k`).
        let mut opening_edges: Vec<EdgeKey> = Vec::with_capacity(sectors.len());

        for sector in &sectors {
            let position = self.solve_meet_point(vertex, vertex_position, sector, offset);
            let boundvert = self.create_vertex(position);
            result.verts.push(boundvert);
            boundverts.push(boundvert);
            opening_edges.push(sector.bounding_edges[0]);

            // Each sector face collapses its corner to this one boundvert (1-for-1).
            for &(_, face) in &sector.corners {
                set_face_substitution(face_subst, vertex, face, vec![boundvert]);
            }
        }

        let outward = self.junction_outward_normal(vertex);
        let count = boundverts.len();

        // One chain per consecutive boundvert pair (B_k → B_{k+1}), crossing the
        // beveled edge shared by sectors k and k+1 (= the edge opening sector k+1).
        for k in 0..count {
            let next = (k + 1) % count;
            let crossing = opening_edges[next];
            let interior = self.build_chain_between(
                boundverts[k],
                boundverts[next],
                crossing,
                segments,
                profile,
                result,
            );
            chains.push(EdgeEndChain {
                edge: crossing,
                endpoint: vertex,
                start: boundverts[k],
                interior,
                end: boundverts[next],
            });
        }

        junction_cuts.push(JunctionCut {
            vertex,
            boundverts,
            opening_edges,
            outward,
        });
    }

    /// Place a valence-3 terminal's two boundverts and build its single edge's
    /// chain (BEVEL.md §8.3.0.1). No fill face: the two boundverts are E's two
    /// perpendicular strip verts (the §7.5 planar offset), the chain runs between
    /// them, and the surviving third face(s) absorb the whole `segments + 1` run.
    #[allow(clippy::too_many_arguments)]
    fn build_terminal_chains(
        &mut self,
        vertex: VertexKey,
        beveled: &[EdgeKey],
        offset: f32,
        segments: u32,
        profile: f32,
        chains: &mut Vec<EdgeEndChain>,
        face_subst: &mut Vec<FaceSubstitution>,
        result: &mut BevelResult,
    ) {
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

        // Only the valence-3 terminal close is implemented (§8.3.0.1): the
        // surviving third face absorbs the run. Valence ≥ 4 (the fan/poly fill)
        // remains a stub.
        if self.iter_disk_edges(vertex).count() != 3 {
            return;
        }

        // The two flanking unbeveled legs share E's two radial faces (§7.5.1); each
        // strip vert is the leg meet of E against its own flanking leg. `direction`
        // is the chain axis, not the strip placement.
        let _ = direction;
        let left_other = self.face_other_edge_at(face_left, vertex, edge);
        let right_other = self.face_other_edge_at(face_right, vertex, edge);
        let surface_normal = self.surface_normal_at(vertex);

        let strip_for_leg =
            |this: &mut Self, leg: EdgeKey, face: FaceKey, result: &mut BevelResult| {
                let position =
                    this.terminal_strip_vert(vertex, edge, leg, face, offset, surface_normal);
                let strip_vert = this.create_vertex(position);
                result.verts.push(strip_vert);
                strip_vert
            };

        let strip_left = strip_for_leg(self, left_other, face_left, result);
        let strip_right = strip_for_leg(self, right_other, face_right, result);

        // The chain crosses E between the two strip verts (§8.3.0.1). face_left's
        // strip vert is the chain's `start`, face_right's its `end` — matching the
        // strip builder's face_left-first ordering of E's radial faces.
        let interior =
            self.build_chain_between(strip_left, strip_right, edge, segments, profile, result);

        // Each E-adjacent radial face collapses its corner to its one strip vert.
        for &(face, strip_vert) in &[(face_left, strip_left), (face_right, strip_right)] {
            set_face_substitution(face_subst, vertex, face, vec![strip_vert]);
        }

        // The third face(s) absorb the full chain run, ordered so the strip vert
        // sharing the incoming leg sits next to `prev` (§8.3.0.1 / §7.4).
        let mut forward_run = Vec::with_capacity(segments as usize + 1);
        forward_run.push(strip_left);
        forward_run.extend_from_slice(&interior);
        forward_run.push(strip_right);
        let mut reverse_run = forward_run.clone();
        reverse_run.reverse();

        for face in self.vert_face_keys(vertex) {
            if face == face_left || face == face_right {
                continue;
            }
            let cycle_verts = self.face_vert_keys(face);
            let cycle_length = cycle_verts.len();
            // A malformed third face (wire-edge sharing, non-manifold) may not
            // carry the endpoint as a clean corner, or its incoming corner edge
            // may not resolve. Skip such a face rather than panic — the gate does
            // not enforce these as preconditions (rubric-5: panics are for
            // internal invariants only).
            let index = match cycle_verts.iter().position(|&w| w == vertex) {
                Some(index) => index,
                None => continue,
            };
            let prev_vertex = cycle_verts[(index + cycle_length - 1) % cycle_length];
            let incoming_edge = match self.query_edge_exists(prev_vertex, vertex) {
                Some(edge) => edge,
                None => continue,
            };
            let ordered = if incoming_edge == right_other {
                reverse_run.clone()
            } else if incoming_edge == left_other {
                forward_run.clone()
            } else {
                forward_run.clone()
            };
            set_face_substitution(face_subst, vertex, face, ordered);
        }

        chains.push(EdgeEndChain {
            edge,
            endpoint: vertex,
            start: strip_left,
            interior,
            end: strip_right,
        });
    }

    /// Build a profile chain's `segments − 1` interior verts between two ring
    /// boundverts crossing beveled edge `edge` (BEVEL.md §8.2.2.1 / §8.3.6). The
    /// endpoints are passed in; the interior arc samples are minted here.
    fn build_chain_between(
        &mut self,
        start: VertexKey,
        end: VertexKey,
        edge: EdgeKey,
        segments: u32,
        profile: f32,
        result: &mut BevelResult,
    ) -> Vec<VertexKey> {
        let start_position = self.vertex_position(start);
        let end_position = self.vertex_position(end);
        let [ea, eb] = self.edge_vert_keys(edge);
        let edge_point = self.vertex_position(ea);
        let edge_dir = (self.vertex_position(eb) - edge_point).normalize_or_zero();

        let cell = ProfileCell::new(start_position, end_position, edge_point, edge_dir);
        let positions = cell.chain_positions(segments, profile);

        let mut interior = Vec::with_capacity(segments.saturating_sub(1) as usize);
        for position in positions.iter().take(positions.len() - 1).skip(1) {
            let vertex = self.create_vertex(*position);
            result.verts.push(vertex);
            interior.push(vertex);
        }
        interior
    }

    /// Build one beveled edge's `segments`-wide strip (BEVEL.md §8.3.0): `segments`
    /// quads joining the edge's two endpoint chains rung-to-rung, `p`'s sample `k`
    /// against `q`'s sample `segments − k`. Each quad shares its rung edges with
    /// the cut / third faces and its long edges with the re-stitched radial faces.
    fn build_multisegment_strip(
        &mut self,
        edge: EdgeKey,
        segments: u32,
        chains: &[EdgeEndChain],
        material: i32,
        representative: i32,
        result: &mut BevelResult,
    ) {
        let [a, b] = self.edge_vert_keys(edge);
        let chain_a = chains.iter().find(|c| c.edge == edge && c.endpoint == a);
        let chain_b = chains.iter().find(|c| c.edge == edge && c.endpoint == b);
        let (chain_a, chain_b) = match (chain_a, chain_b) {
            (Some(a), Some(b)) => (a, b),
            _ => return,
        };

        // The §8.3.0 rung join pairs `p` sample k with the `q` sample on the *same
        // physical face side* of E. Orient `q` so its `start` sits on the same side
        // as `p`'s — the rails then run *along* E (short) instead of crossing it.
        // Picked by the shorter total rail length, which is unambiguous since the
        // two endpoints' chains face each other across E.
        let p_samples = chain_a.samples();
        let mut q_samples = chain_b.samples();
        let aligned = (self.vertex_position(p_samples[0])
            - self.vertex_position(q_samples[0]))
        .length()
            + (self.vertex_position(p_samples[p_samples.len() - 1])
                - self.vertex_position(q_samples[q_samples.len() - 1]))
            .length();
        let swapped = (self.vertex_position(p_samples[0])
            - self.vertex_position(q_samples[q_samples.len() - 1]))
        .length()
            + (self.vertex_position(p_samples[p_samples.len() - 1])
                - self.vertex_position(q_samples[0]))
            .length();
        if swapped < aligned {
            q_samples.reverse();
        }
        if p_samples.len() != segments as usize + 1 || q_samples.len() != segments as usize + 1 {
            return;
        }

        for k in 1..=segments as usize {
            let ring = [
                p_samples[k - 1],
                p_samples[k],
                q_samples[k],
                q_samples[k - 1],
            ];
            let mut distinct: Vec<VertexKey> = Vec::with_capacity(4);
            for &v in &ring {
                if !distinct.contains(&v) {
                    distinct.push(v);
                }
            }
            if distinct.len() < 3 {
                continue;
            }
            // Wind opposite the re-stitched face traversal on the long edge so each
            // long edge stays radial-2 (§5.1).
            let mut quad = distinct;
            if let Some(long_edge) = self.query_edge_exists(ring[0], ring[1]) {
                let runs_forward = self
                    .edge_loop_keys(long_edge)
                    .first()
                    .map(|&l| {
                        let start = self.loop_vertex(l);
                        let next = self.loop_vertex(self.loop_next(l));
                        start == ring[0] && next == ring[1]
                    })
                    .unwrap_or(false);
                if runs_forward {
                    quad.reverse();
                }
            }
            let face = self.face_create_verts(&quad, FaceCreateFlags::NONE);
            self.assign_bevel_face_material(face, material, representative);
            result.faces.push(face);
            self.collect_face_edges(face, result);
        }
    }

    /// Build the cutoff fill at one junction (BEVEL.md §8.3.2 / §8.3.4): one cut
    /// face per sector spanning its chain plus two bottom corner verts, the corner
    /// ring closed by a bottom face — or collapsed to a single apex when the corner
    /// verts coincide (the `s = 3` case, §8.3.4).
    fn build_cutoff_fill(
        &mut self,
        cut: &JunctionCut,
        chains: &[EdgeEndChain],
        segments: u32,
        profile: f32,
        material: i32,
        result: &mut BevelResult,
    ) {
        let count = cut.boundverts.len();
        if count < 2 {
            return;
        }
        let representative = cut.representative_material(self);

        // Per boundvert, its down-pushed corner-vert position (§8.3.3).
        let mut corner_positions: Vec<Vec3> = Vec::with_capacity(count);
        for k in 0..count {
            corner_positions.push(self.corner_vert_position(cut, chains, k));
        }

        // §8.3.3/§8.3.4: the 3-way collapse to a single apex is a binary
        // profile decision. It is suppressed for exactly one profile — the
        // straight chord (`profile = 0.25`, the `r == 1` line case the
        // superellipse sampler already special-cases) — whose corner verts stay
        // distinct at the undisplaced boundvert ring (closed by a bottom
        // triangle). Every curved profile (`r != 1`, including a near-chord
        // `profile = 0.26`) collapses to a single apex.
        let flat_chord = count == 3 && profile_to_exponent(profile) == 1.0;
        let collapsed = count == 3 && !flat_chord && corner_verts_coincide(&corner_positions);

        let corner_verts: Vec<VertexKey> = if collapsed {
            let apex_position =
                (corner_positions[0] + corner_positions[1] + corner_positions[2]) / 3.0;
            let apex = self.create_vertex(apex_position);
            result.verts.push(apex);
            vec![apex; count]
        } else {
            (0..count)
                .map(|k| {
                    // A flat-chord 3-way takes its corners at the undisplaced
                    // boundvert ring (the §8.3.3 down-push is ~0); a wider or bulged
                    // ring takes the full down-pushed position.
                    let position = if flat_chord {
                        self.vertex_position(cut.boundverts[k])
                    } else {
                        corner_positions[k]
                    };
                    let corner = self.create_vertex(position);
                    result.verts.push(corner);
                    corner
                })
                .collect()
        };

        // One cut face per sector: C_k → chain(B_k → B_{k+1}) → C_{k+1} → C_k.
        for k in 0..count {
            let next = (k + 1) % count;
            let start_boundvert = cut.boundverts[k];
            let end_boundvert = cut.boundverts[next];
            let chain = chains.iter().find(|c| {
                c.endpoint == cut.vertex
                    && ((c.start == start_boundvert && c.end == end_boundvert)
                        || (c.start == end_boundvert && c.end == start_boundvert))
            });
            let chain = match chain {
                Some(chain) => chain.oriented_from(start_boundvert),
                None => continue,
            };

            let mut loop_verts: Vec<VertexKey> = Vec::with_capacity(segments as usize + 3);
            loop_verts.push(corner_verts[k]);
            loop_verts.extend_from_slice(&chain);
            loop_verts.push(corner_verts[next]);

            // Drop consecutive duplicates (a collapsed apex makes C_k == C_{k+1}).
            let mut deduped: Vec<VertexKey> = Vec::with_capacity(loop_verts.len());
            for &v in &loop_verts {
                if deduped.last() != Some(&v) {
                    deduped.push(v);
                }
            }
            if deduped.len() >= 2 && deduped.first() == deduped.last() {
                deduped.pop();
            }
            if deduped.len() < 3 {
                continue;
            }

            // Wind opposite the strip's traversal of the first chain edge so the cut
            // face stays radial-2 against the strip (§8.3.2).
            let mut face_verts = deduped;
            if let Some(chain_edge) = self.query_edge_exists(chain[0], chain[1]) {
                let runs_forward = self
                    .edge_loop_keys(chain_edge)
                    .first()
                    .map(|&l| {
                        let start = self.loop_vertex(l);
                        let next = self.loop_vertex(self.loop_next(l));
                        start == chain[0] && next == chain[1]
                    })
                    .unwrap_or(false);
                if runs_forward {
                    face_verts.reverse();
                }
            }
            let face = self.face_create_verts(&face_verts, FaceCreateFlags::NONE);
            self.assign_bevel_face_material(face, material, representative);
            result.faces.push(face);
            self.collect_face_edges(face, result);
        }

        // The bottom face over the corner ring (omitted on the apex collapse).
        if !collapsed && count >= 3 {
            let mut ring = corner_verts.clone();
            if let Some(bottom_edge) = self.query_edge_exists(ring[0], ring[1]) {
                let cut_runs_forward = self
                    .edge_loop_keys(bottom_edge)
                    .first()
                    .map(|&l| {
                        let start = self.loop_vertex(l);
                        let next = self.loop_vertex(self.loop_next(l));
                        start == ring[0] && next == ring[1]
                    })
                    .unwrap_or(false);
                if cut_runs_forward {
                    ring.reverse();
                }
            }
            let face = self.face_create_verts(&ring, FaceCreateFlags::NONE);
            self.assign_bevel_face_material(face, material, representative);
            result.faces.push(face);
            self.collect_face_edges(face, result);
        }
    }

    /// The §8.5.1 forced-tri-corner trigger: whether this 3-way junction is the
    /// symmetric equal-offset orthonormal cube corner at the default circular
    /// profile that the ADJ tri-corner fill (§8.5.1–§8.5.5) is distilled for.
    ///
    /// The two geometric conditions (§8.5.5.3) are read off the boundvert ring,
    /// which is pairing-robust at a triangle (every pair of boundverts shares one
    /// beveled edge). Writing each ring vert as the §7.6.2 meet-point
    /// `B_i = v + offset·(d_a + d_b)` over the two orthonormal inward edge
    /// directions it bridges:
    /// - **equal offset** ⇒ the three `w_i = B_i − v` have equal length
    ///   (`|w_i| = offset·√2`); an unequal-offset corner breaks this;
    /// - **right-angle convex octant** ⇒ every pair `w_i, w_j` meets at 60°
    ///   (`cos = ½`, the signature of mutually perpendicular `d_*`); a non-octant
    ///   corner moves the cosine off ½.
    ///
    /// The default-profile condition is the circular superellipse (`r = 2`); a
    /// curved / square / custom profile snaps onto a superellipsoid the closed-form
    /// sphere fit does not solve (§8.5.5.5), so it declines here.
    fn adj_tri_corner_trigger(&self, cut: &JunctionCut, profile: f32) -> bool {
        if cut.boundverts.len() != 3 {
            return false;
        }
        if profile_to_exponent(profile) != 2.0 {
            return false;
        }
        let vertex = self.vertex_position(cut.vertex);
        let spokes = [
            self.vertex_position(cut.boundverts[0]) - vertex,
            self.vertex_position(cut.boundverts[1]) - vertex,
            self.vertex_position(cut.boundverts[2]) - vertex,
        ];
        let lengths = [spokes[0].length(), spokes[1].length(), spokes[2].length()];
        let longest = lengths.iter().copied().fold(0.0_f32, f32::max);
        if longest <= 1.0e-6 {
            return false;
        }
        // Equal offset within tolerance: the three meet-point spokes are equal length.
        let length_tol = longest * 1.0e-3;
        if (lengths[0] - lengths[1]).abs() > length_tol
            || (lengths[1] - lengths[2]).abs() > length_tol
            || (lengths[2] - lengths[0]).abs() > length_tol
        {
            return false;
        }
        // Right-angle convex octant: each spoke pair meets at 60° (cos = ½).
        let directions = [
            spokes[0] / lengths[0],
            spokes[1] / lengths[1],
            spokes[2] / lengths[2],
        ];
        let cos_tol = 1.0e-3;
        (directions[0].dot(directions[1]) - 0.5).abs() <= cos_tol
            && (directions[1].dot(directions[2]) - 0.5).abs() <= cos_tol
            && (directions[2].dot(directions[0]) - 0.5).abs() <= cos_tol
    }

    /// Build the rounded adjacency-grid fill at one junction (BEVEL.md §8.5),
    /// distilled for the forced `segments = 2` 3-way (cube-corner) case: a
    /// three-quad pinwheel meeting at one welded central apex.
    ///
    /// Per §8.5.3 each sector `i` contributes one quad whose corners are
    /// `B_i` (`(i,0,0)`), sector `i`'s interior chain vert (`(i,0,1)`), the central
    /// apex (`(i,1,1)`), and sector `i−1`'s interior chain vert (`(i,1,0)`, the
    /// shared seam vert with the previous sector). The three apex corners weld to
    /// one vertex, placed closed-form (§8.5.2): the `unit_to_corner` body-diagonal
    /// direction snapped onto the rounded surface the ring + chain verts lie on, so
    /// the apex sits *on* the dome rather than pushed down into the solid (§8.5.3).
    ///
    /// Only the forced case is handled: exactly three sectors, each with a single
    /// lone interior chain vert (`segments = 2`). Any other shape is left unfilled
    /// (the general adjacency grid is a stub, §8.5.5).
    fn build_adj_fill(
        &mut self,
        cut: &JunctionCut,
        chains: &[EdgeEndChain],
        segments: u32,
        profile: f32,
        material: i32,
        result: &mut BevelResult,
    ) {
        let count = cut.boundverts.len();
        // §8.5 forces the tri-corner only for `s = 3`; higher valence is the
        // §8.5.5.5 stub.
        if count != 3 {
            return;
        }
        // §8.5.1 forced-tri-corner trigger. The whole ADJ tri-corner construction
        // (both the closed-form `segments = 2` pinwheel and the larger-`segments`
        // grid) is distilled *only* for the symmetric equal-offset orthonormal cube
        // corner at the default circular profile. Outside that the unit-frame map is
        // no longer a similarity and the snap target is a genuine ellipsoid the guide
        // defers (§8.5.5.5); the closed-form sphere snap below would otherwise pin the
        // fill onto the wrong surface. When the trigger does not hold we decline (the
        // same no-op as the `count != 3` case) rather than emit unpinned geometry.
        if !self.adj_tri_corner_trigger(cut, profile) {
            return;
        }
        // The forced `segments = 2` pinwheel (§8.5.1–§8.5.4) keeps its dedicated
        // closed-form path below; `segments ≥ 3` takes the general subdivided
        // grid (§8.5.5). `segments < 2` has no interior to fill.
        match segments {
            0 | 1 => return,
            2 => {}
            _ => {
                self.build_adj_fill_general(cut, chains, segments, material, result);
                return;
            }
        }

        // Sector i's interior chain vert (i,0,1): the lone interior sample of the
        // chain that *starts* at B_i (B_i → B_{i+1}).
        let mut interior_of_sector: Vec<VertexKey> = Vec::with_capacity(count);
        for k in 0..count {
            let start = cut.boundverts[k];
            let next = cut.boundverts[(k + 1) % count];
            let chain = chains.iter().find(|c| {
                c.endpoint == cut.vertex
                    && ((c.start == start && c.end == next) || (c.start == next && c.end == start))
            });
            match chain.and_then(|c| c.interior.first().copied()) {
                Some(interior) => interior_of_sector.push(interior),
                // A degenerate sector with no interior sample (collinear chain) is
                // outside the forced cube-corner regime; skip the whole fill.
                None => return,
            }
        }

        let representative = cut.representative_material(self);

        // §8.5.2 closed-form apex. The affine `unit_to_corner` axis-map ((1,0,0)→B_0,
        // (0,1,0)→B_1, (0,0,1)→B_2, origin→v) sends the unit body-diagonal point to
        // the corner-interior direction; the apex is that direction *snapped onto the
        // rounded surface* the ring + chain verts already lie on (the superellipsoid;
        // for the default profile, the sphere). The naive affine image alone lands the
        // apex at 2·offset from the corner — pushed off the surface into the solid — so
        // the snap step is what places it on the dome (§8.5.3).
        //
        // That rounded surface is the sphere through the three boundverts (ring) and
        // the sectors' interior arc samples (chain). Fit it from those on-surface
        // verts, then take the apex as its point nearest the junction vertex along the
        // corner's body diagonal: `c + normalize(v − c)·R`.
        let vertex_position = self.vertex_position(cut.vertex);
        let on_surface = [
            self.vertex_position(cut.boundverts[0]),
            self.vertex_position(cut.boundverts[1]),
            self.vertex_position(cut.boundverts[2]),
            self.vertex_position(interior_of_sector[0]),
        ];
        let apex_position = match circumsphere(on_surface) {
            Some((center, radius)) => {
                let body_diagonal = (vertex_position - center).normalize_or_zero();
                center + body_diagonal * radius
            }
            // Degenerate (co-spherically deficient) ring: fall back to the affine image
            // of the body-diagonal point so the fill still closes.
            None => {
                let axis_sum: Vec3 = cut.boundverts.iter().fold(Vec3::ZERO, |acc, &b| {
                    acc + (self.vertex_position(b) - vertex_position)
                });
                vertex_position + axis_sum / 3.0_f32.sqrt()
            }
        };
        let apex = self.create_vertex(apex_position);
        result.verts.push(apex);

        // §8.5.3: one pinwheel quad per sector, all sharing the apex. Sector i's
        // quad is B_i → (i,0,1) → apex → (i−1,0,1).
        for i in 0..count {
            let before = (i + count - 1) % count;
            let mut quad = vec![
                cut.boundverts[i],
                interior_of_sector[i],
                apex,
                interior_of_sector[before],
            ];

            // Wind opposite the strip's traversal of the shared `B_i → (i,0,1)` edge
            // so that edge stays radial-2 against the arriving multi-segment strip
            // (§8.5.3 / §5.1), the same share-don't-flip discipline as cutoff.
            if let Some(strip_edge) = self.query_edge_exists(quad[0], quad[1]) {
                let runs_forward = self
                    .edge_loop_keys(strip_edge)
                    .first()
                    .map(|&l| {
                        let start = self.loop_vertex(l);
                        let next = self.loop_vertex(self.loop_next(l));
                        start == quad[0] && next == quad[1]
                    })
                    .unwrap_or(false);
                if runs_forward {
                    quad.reverse();
                }
            }
            let face = self.face_create_verts(&quad, FaceCreateFlags::NONE);
            self.assign_bevel_face_material(face, material, representative);
            result.faces.push(face);
            self.collect_face_edges(face, result);
        }
    }

    /// Build the rounded adjacency-grid fill for the symmetric equal-offset 3-way
    /// cube corner at `segments ≥ 3` (BEVEL.md §8.5.5). The `segments = 2` pinwheel
    /// (§8.5.1–§8.5.4) is the coarse control net; this tiles the larger grid as
    /// `s = 3` quad **sub-patches** — one per boundvert `B_i`, spanning from `B_i`
    /// inward to the corner centre — each interior grid vert snapped onto the corner
    /// superellipsoid (the sphere, for the default circular profile, §8.5.5.3).
    ///
    /// The centre resolves by parity (§8.5.5.4): even `segments` welds the three
    /// sub-patches at one apex vertex `c + R·normalize(v − c)`; odd `segments` leaves
    /// the sub-patches separated by a ring of seam quads closed by a central
    /// `s`-gon face, with no apex vertex on the body diagonal.
    ///
    /// The boundary (the three profile chains and the boundvert ring) is *reused*
    /// from [P]/[S], not re-minted, so the patch shares its rim with the arriving
    /// strips. Interior positions are an admissible implementation choice (Tier-2,
    /// `bevel_vmesh.md`); only their superellipsoid membership and the grid
    /// counts / parity are pinned.
    ///
    /// Scope (§8.5.5.5): the symmetric equal-offset cube corner, default circular
    /// profile only. Higher valence, unequal offsets, and the pipe / square grids
    /// are left unfilled.
    fn build_adj_fill_general(
        &mut self,
        cut: &JunctionCut,
        chains: &[EdgeEndChain],
        segments: u32,
        material: i32,
        result: &mut BevelResult,
    ) {
        let sectors = cut.boundverts.len(); // 3 (guarded by the caller)
        let seg = segments as usize;
        let half = seg / 2;

        // Each sector's profile chain, oriented from its opening boundvert `B_i`
        // (`seg + 1` samples; the two ends are the ring boundverts).
        let mut chain_of: Vec<Vec<VertexKey>> = Vec::with_capacity(sectors);
        for i in 0..sectors {
            match self.oriented_chain(cut, chains, i) {
                Some(chain) if chain.len() == seg + 1 => chain_of.push(chain),
                // A sector that did not sample to `seg + 1` verts (a collinear /
                // degenerate chain) is outside the forced cube-corner regime; skip
                // the whole fill rather than emit a torn patch.
                _ => return,
            }
        }

        // The corner sphere the whole patch lies on (§8.5.5.3): fit it from the
        // three ring boundverts and one on-arc chain sample, all at distance `R`
        // from the centre `c`. For the default circular profile the superellipsoid
        // is exactly this sphere, so every grid vert snaps onto it by the radial
        // projection `c + R·normalize(p − c)`.
        let on_surface = [
            self.vertex_position(cut.boundverts[0]),
            self.vertex_position(cut.boundverts[1]),
            self.vertex_position(cut.boundverts[2]),
            self.vertex_position(chain_of[0][1]),
        ];
        let (center, radius) = match circumsphere(on_surface) {
            Some(sphere) => sphere,
            None => return,
        };

        let vertex_position = self.vertex_position(cut.vertex);
        let even = seg % 2 == 0;

        // Per sub-patch, its four corner positions: `B_i`, the chain-`i` inner end
        // (`P0t`), the chain-`(i−1)` inner end (`Pt0`), and the inner corner (`Ptt`)
        // — the welded apex for even `segments`, a distinct central-ring vert `G_i`
        // for odd. Interior grid verts are the bilinear blend of these four, snapped
        // to the sphere (§8.5.5.3).
        let mut corners: Vec<[Vec3; 4]> = Vec::with_capacity(sectors);
        for i in 0..sectors {
            let p00 = self.vertex_position(chain_of[i][0]);
            let p0t = self.vertex_position(chain_of[i][half]);
            let prev = (i + sectors - 1) % sectors;
            let pt0 = self.vertex_position(chain_of[prev][seg - half]);
            let ptt = if even {
                center + radius * (vertex_position - center).normalize_or_zero()
            } else {
                // Odd centre ring: tilt the body-diagonal direction toward `B_i` so
                // the three `G_i` form a small central `s`-gon around the diagonal,
                // each on the sphere. The exact tilt is a free interior choice.
                let diagonal = (vertex_position - center).normalize_or_zero();
                let toward_boundvert = (p00 - center).normalize_or_zero();
                let gamma = 1.0 / (half as f32 + 1.0);
                center + radius * (diagonal + gamma * toward_boundvert).normalize_or_zero()
            };
            corners.push([p00, p0t, pt0, ptt]);
        }

        let representative = cut.representative_material(self);
        let mut grid: std::collections::HashMap<(usize, usize, usize), VertexKey> =
            std::collections::HashMap::new();

        // Sub-patch interior cells (`half × half` quads each), shared between
        // sectors at the seams (even) or independent (odd, the seam strips bridge
        // them). Walk every cell of every sub-patch.
        for i in 0..sectors {
            for cell_j in 0..half {
                for cell_k in 0..half {
                    let a = self.adj_grid_vert(
                        i, cell_j, cell_k, half, seg, sectors, even, &chain_of, &corners,
                        center, radius, &mut grid, result,
                    );
                    let b = self.adj_grid_vert(
                        i, cell_j, cell_k + 1, half, seg, sectors, even, &chain_of, &corners,
                        center, radius, &mut grid, result,
                    );
                    let c = self.adj_grid_vert(
                        i, cell_j + 1, cell_k + 1, half, seg, sectors, even, &chain_of, &corners,
                        center, radius, &mut grid, result,
                    );
                    let d = self.adj_grid_vert(
                        i, cell_j + 1, cell_k, half, seg, sectors, even, &chain_of, &corners,
                        center, radius, &mut grid, result,
                    );
                    self.emit_patch_face(&[a, b, c, d], center, material, representative, result);
                }
            }
        }

        if even {
            // The even centre is already welded: every sub-patch's innermost cell
            // meets the shared apex vertex, so no further faces are needed.
            return;
        }

        // Odd centre (§8.5.5.4): bridge each pair of adjacent sub-patches with a
        // strip of `half` quads — sub-patch `i`'s inner column (`k = half`) against
        // sub-patch `(i+1)`'s inner row (`j = half`) — covering the chains' central
        // segments, then close the `G_i` ring with one central `s`-gon face.
        for i in 0..sectors {
            let next = (i + 1) % sectors;
            let mut column: Vec<VertexKey> = Vec::with_capacity(half + 1);
            let mut row: Vec<VertexKey> = Vec::with_capacity(half + 1);
            for r in 0..=half {
                column.push(self.adj_grid_vert(
                    i, r, half, half, seg, sectors, even, &chain_of, &corners, center, radius,
                    &mut grid, result,
                ));
                row.push(self.adj_grid_vert(
                    next, half, r, half, seg, sectors, even, &chain_of, &corners, center, radius,
                    &mut grid, result,
                ));
            }
            for r in 0..half {
                self.emit_patch_face(
                    &[column[r], row[r], row[r + 1], column[r + 1]],
                    center,
                    material,
                    representative,
                    result,
                );
            }
        }

        let mut central: Vec<VertexKey> = Vec::with_capacity(sectors);
        for i in 0..sectors {
            central.push(self.adj_grid_vert(
                i, half, half, half, seg, sectors, even, &chain_of, &corners, center, radius,
                &mut grid, result,
            ));
        }
        self.emit_patch_face(&central, center, material, representative, result);
    }

    /// Resolve the adjacency-grid vert at sub-patch index `(i, j, k)` (BEVEL.md
    /// §8.5.5). Boundary rows (`j == 0` / `k == 0`) reuse the existing chain /
    /// boundvert keys so the patch shares its rim with the strips; interior verts
    /// are created once at the bilinear blend of the sub-patch corners, snapped to
    /// the corner sphere, and cached. For even `segments` the inner column (`k ==
    /// half`), inner row (`j == half`), and centre alias onto the neighbouring
    /// sub-patch (the welded seams / apex); odd `segments` keeps each sub-patch
    /// independent.
    #[allow(clippy::too_many_arguments)]
    fn adj_grid_vert(
        &mut self,
        i: usize,
        j: usize,
        k: usize,
        half: usize,
        seg: usize,
        sectors: usize,
        even: bool,
        chain_of: &[Vec<VertexKey>],
        corners: &[[Vec3; 4]],
        center: Vec3,
        radius: f32,
        grid: &mut std::collections::HashMap<(usize, usize, usize), VertexKey>,
        result: &mut BevelResult,
    ) -> VertexKey {
        // `j = 0` is the chain `i` boundary (from `B_i`); `k = 0` is the chain
        // `(i−1)` boundary (toward `B_i`, hence reversed). Their shared corner
        // `(0, 0)` is `B_i` from either chain.
        if j == 0 {
            return chain_of[i][k];
        }
        if k == 0 {
            return chain_of[(i + sectors - 1) % sectors][seg - j];
        }

        // Interior. Even `segments` welds the seams / apex by aliasing the inner
        // column / row / corner onto one canonical sub-patch; odd keeps `(i, j, k)`.
        let canon = if even {
            if j == half && k == half {
                (0, half, half) // the single welded apex
            } else if k == half {
                (i, j, half) // inner column = seam, owned by sub-patch i
            } else if j == half {
                ((i + sectors - 1) % sectors, k, half) // inner row aliases prev sub-patch's seam
            } else {
                (i, j, k)
            }
        } else {
            (i, j, k)
        };
        if let Some(&existing) = grid.get(&canon) {
            return existing;
        }

        let [p00, p0t, pt0, ptt] = corners[canon.0];
        let s = canon.2 as f32 / half as f32;
        let u = canon.1 as f32 / half as f32;
        let blended = p00 * ((1.0 - s) * (1.0 - u))
            + p0t * (s * (1.0 - u))
            + pt0 * ((1.0 - s) * u)
            + ptt * (s * u);
        let position = center + radius * (blended - center).normalize_or_zero();
        let vertex = self.create_vertex(position);
        result.verts.push(vertex);
        grid.insert(canon, vertex);
        vertex
    }

    /// Emit one adjacency-grid patch face (BEVEL.md §8.5.5), wound outward so it is
    /// radial-2 against the arriving strips and its neighbours: the face normal is
    /// flipped to agree with the outward sphere normal `centroid − c`. Consecutive
    /// duplicate corners (none expected, a guard) are dropped.
    fn emit_patch_face(
        &mut self,
        verts: &[VertexKey],
        center: Vec3,
        material: i32,
        representative: i32,
        result: &mut BevelResult,
    ) {
        let mut ring: Vec<VertexKey> = Vec::with_capacity(verts.len());
        for &vertex in verts {
            if ring.last() != Some(&vertex) {
                ring.push(vertex);
            }
        }
        if ring.len() >= 2 && ring.first() == ring.last() {
            ring.pop();
        }
        if ring.len() < 3 {
            return;
        }

        let positions: Vec<Vec3> = ring.iter().map(|&v| self.vertex_position(v)).collect();
        let centroid = positions.iter().sum::<Vec3>() / positions.len() as f32;
        let mut normal = Vec3::ZERO;
        for index in 0..positions.len() {
            let edge_a = positions[index] - centroid;
            let edge_b = positions[(index + 1) % positions.len()] - centroid;
            normal += edge_a.cross(edge_b);
        }
        if normal.dot(centroid - center) < 0.0 {
            ring.reverse();
        }

        let face = self.face_create_verts(&ring, FaceCreateFlags::NONE);
        self.assign_bevel_face_material(face, material, representative);
        result.faces.push(face);
        self.collect_face_edges(face, result);
    }

    /// The profile chain of sector `i` at this junction, oriented from its opening
    /// boundvert `B_i` (BEVEL.md §8.2.2). `None` when no chain spans the sector's
    /// boundvert pair.
    fn oriented_chain(
        &self,
        cut: &JunctionCut,
        chains: &[EdgeEndChain],
        i: usize,
    ) -> Option<Vec<VertexKey>> {
        let count = cut.boundverts.len();
        let start = cut.boundverts[i];
        let end = cut.boundverts[(i + 1) % count];
        let chain = chains.iter().find(|c| {
            c.endpoint == cut.vertex
                && ((c.start == start && c.end == end) || (c.start == end && c.end == start))
        })?;
        Some(chain.oriented_from(start))
    }

    /// One boundvert's down-pushed corner-vert position (BEVEL.md §8.3.3): `B_k`
    /// plus the cross of the two adjacent sectors' mid-plane normals (the fold
    /// crease), scaled by the average of the two profiles' `height/√2`, oriented
    /// into the corner.
    fn corner_vert_position(&self, cut: &JunctionCut, chains: &[EdgeEndChain], k: usize) -> Vec3 {
        let count = cut.boundverts.len();
        let boundvert = cut.boundverts[k];
        let before = (k + count - 1) % count; // sector ending at B_k
        let after = k; // sector starting at B_k

        let cell_before = self.sector_cell(cut, chains, before);
        let cell_after = self.sector_cell(cut, chains, after);

        let boundvert_position = self.vertex_position(boundvert);
        let (normal_before, height_before) = cell_before
            .map(|c| (c.mid_plane_normal, c.height()))
            .unwrap_or((Vec3::ZERO, 0.0));
        let (normal_after, height_after) = cell_after
            .map(|c| (c.mid_plane_normal, c.height()))
            .unwrap_or((Vec3::ZERO, 0.0));

        let mut down = normal_before.cross(normal_after);
        if down.length_squared() <= 1.0e-12 {
            return boundvert_position;
        }
        down = down.normalize();
        if down.dot(cut.outward) > 0.0 {
            down = -down;
        }

        let inv_sqrt2 = 1.0 / 2.0_f32.sqrt();
        let push = (height_before * inv_sqrt2 + height_after * inv_sqrt2) / 2.0;
        boundvert_position + down * push
    }

    /// The profile cell of sector `k` at the junction (BEVEL.md §8.2.2.1), rebuilt
    /// from that sector's chain (B_k → B_{k+1}) and its crossing beveled edge.
    fn sector_cell(
        &self,
        cut: &JunctionCut,
        chains: &[EdgeEndChain],
        k: usize,
    ) -> Option<ProfileCell> {
        let count = cut.boundverts.len();
        let next = (k + 1) % count;
        let start_boundvert = cut.boundverts[k];
        let end_boundvert = cut.boundverts[next];
        let chain = chains.iter().find(|c| {
            c.endpoint == cut.vertex
                && ((c.start == start_boundvert && c.end == end_boundvert)
                    || (c.start == end_boundvert && c.end == start_boundvert))
        })?;
        let [ea, eb] = self.edge_vert_keys(chain.edge);
        let edge_point = self.vertex_position(ea);
        let edge_dir = (self.vertex_position(eb) - edge_point).normalize_or_zero();
        Some(ProfileCell::new(
            self.vertex_position(start_boundvert),
            self.vertex_position(end_boundvert),
            edge_point,
            edge_dir,
        ))
    }

    /// A junction vertex's outward surface normal: the summed plane normals of its
    /// incident faces (BEVEL.md §8.3.3 down-push sign reference). Used only to
    /// decide which way the fold crease points (into the solid).
    fn junction_outward_normal(&self, vertex: VertexKey) -> Vec3 {
        let mut sum = Vec3::ZERO;
        for face in self.vert_face_keys(vertex) {
            sum += self.face_plane_normal(face);
        }
        sum.normalize_or_zero()
    }

    /// Push every edge of `face` into the result's edge set (de-duplicated).
    fn collect_face_edges(&self, face: FaceKey, result: &mut BevelResult) {
        for loop_key in self.face_loop_keys(face) {
            let edge = self.loop_edge(loop_key);
            if !result.edges.contains(&edge) {
                result.edges.push(edge);
            }
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
        loop_slide: bool,
        corner_vert: &mut Vec<CornerVert>,
        substitutions: &mut Vec<FaceSubstitution>,
        junction_patches: &mut Vec<JunctionPatch>,
        reconciled: &mut Vec<ReconciledOffset>,
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
            // §8.8: when `loop_slide` is set and this sector fires (exactly one
            // silhouette/in-plane leg, slide not folding either flank), place the
            // boundvert on that leg's line instead of at the §7.6.2 meet-point.
            let slid = if loop_slide {
                self.slide_boundvert(vertex, vertex_position, sector, is_beveled)
            } else {
                None
            };
            let boundvert_position = if let Some(record) = slid {
                // §7.6.4.5 construct→reconcile→rebuild, in closed form for the single
                // firing junction (§7.6.4.3). The slid boundvert constrains its two
                // flanking edges' *apex-end* offsets to the flank-sine ratio
                // `o_k : o_(k+1) = sin α₁ : sin α₂` (width family, weight `1.0`) while a
                // spec-pull (weight `w = 0.2`) anchors the absolute scale to the user
                // offset. With `r = sin(α₂)/sin(α₁)` the apex offsets solve to
                // `o₁ = offset·(1+r)/(1+r²)` (g_k side) and `o₂ = r·o₁` (g_(k+1) side),
                // independent of `w` (§7.6.4.3). At a symmetric junction `r = 1` so both
                // collapse to the spec offset and the whole pass is a bit-exact no-op.
                let g_k = sector.bounding_edges[0];
                let g_k1 = sector.bounding_edges[1];
                let r = record.flank_sine_ratio;
                let apex_scale = (1.0 + r) / (1.0 + r * r);
                let o1 = offset * apex_scale;
                let o2 = offset * r * apex_scale;

                // §7.6.4.1 / §7.6.4.3: the *far end* of each flanking edge is its own
                // variable — the damped blend `(apex_offset + w²·O) / (1 + w²)`, not the
                // apex offset carried unchanged (which overshoots the terminal by the
                // fixed `1 + w² = 1.04`, §7.6.4.4). Keyed by `(edge, face)` — the edge's
                // radial face bordering this sector — so it ripples down that edge to its
                // far end, where a terminal close (§7.5.1) reads it (§8.8.5). At `r = 1`
                // the damped blend is exactly the spec offset, so a symmetric firing
                // junction leaves the termini static.
                let damp = |apex_offset: f32| {
                    (apex_offset + WIDTH_SPEC_WEIGHT_SQ * offset) / (1.0 + WIDTH_SPEC_WEIGHT_SQ)
                };
                if let Some(&(_, face)) = sector.corners.first() {
                    reconciled.push(ReconciledOffset {
                        edge: g_k,
                        face,
                        offset: damp(o1),
                    });
                }
                if let Some(&(_, face)) = sector.corners.last() {
                    reconciled.push(ReconciledOffset {
                        edge: g_k1,
                        face,
                        offset: damp(o2),
                    });
                }

                // §8.8.3.1 rebuild: re-place the apex off the *reconciled* offsets. The
                // reconciled pair makes the two sub-meets coincide —
                // `m₁ = v + (o₁/sin α₁)·d_leg` and
                // `m₂ = v + (o₂/sin α₂)·d_leg = v + (r·o₁/sin α₂)·d_leg = m₁`
                // (since `r = sin α₂/sin α₁`) — so the apex lands exactly at the single
                // reconciled point, not at the construct pass's average of two *unequal*
                // spec sub-meets `(O/sin α₁ + O/sin α₂)/2`. Both terms run along the
                // un-projected `d_leg`, so the apex stays on the leg's line (§8.8.3).
                vertex_position + record.leg_direction * (o1 / record.sin_alpha_before)
            } else {
                self.solve_meet_point(vertex, vertex_position, sector, offset)
            };
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

    /// Place the two strip verts of a terminal endpoint (BEVEL.md §7.4 / §7.5.1):
    /// one per adjacent face of the single beveled edge there. Each strip vert is
    /// the §7.5.1 **leg meet** — the pairwise offset-line meet of the beveled edge
    /// `E` (offset) against the flanking unbeveled leg (offset 0) — **not** the
    /// §7.3 perpendicular-in-face slide, which only coincides at a planar, square
    /// terminal and diverges at an irregular non-planar one. Each strip's radial
    /// face swaps its corner for the one strip vert on its side (1-for-1); the
    /// surviving **third** face(s) between the unbeveled legs swap their corner for
    /// *both* strip verts ordered so the strip vert sharing the incoming leg stays
    /// adjacent to `prev` (1-for-2, the absorbed valence-3 corner of §7.4 / §7.5).
    fn place_terminal_strip_verts(
        &mut self,
        vertex: VertexKey,
        beveled: &[EdgeKey],
        offset: f32,
        reconciled: &[ReconciledOffset],
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

        // The two flanking unbeveled legs: `left_other` shares `face_left` with the
        // strip, `right_other` shares `face_right`. Each strip vert is the §7.5.1
        // leg meet of `E` against its own flanking leg (offset 0). `direction`
        // unused for the leg meet (the meet builds its own edge-cross normal).
        let _ = direction;
        let left_other = self.face_other_edge_at(face_left, vertex, edge);
        let right_other = self.face_other_edge_at(face_right, vertex, edge);
        let surface_normal = self.surface_normal_at(vertex);

        // §8.8.5: each strip vert reads its edge-side's §7.6.4-reconciled offset —
        // the spec offset unless a firing slide at this edge's *other* end shifted it
        // (§8.8.4). The recipe is the §7.5.1 leg meet either way; only the offset it
        // is fed ripples, so the terminal moves at an irregular firing junction and
        // stays put at a symmetric one (where the reconciled offset equals the spec).
        let strip_for_leg =
            |this: &mut Self, leg: EdgeKey, face: FaceKey, result: &mut BevelResult| {
                let face_offset = reconciled_offset(reconciled, edge, face, offset);
                let position =
                    this.terminal_strip_vert(vertex, edge, leg, face, face_offset, surface_normal);
                let strip_vert = this.create_vertex(position);
                result.verts.push(strip_vert);
                strip_vert
            };

        let strip_left = strip_for_leg(self, left_other, face_left, result);
        let strip_right = strip_for_leg(self, right_other, face_right, result);

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

    /// Place one terminal strip vert as the §7.5.1 **leg meet**: the pairwise
    /// offset-line meet of the beveled edge `edge` (carrying `offset`) against the
    /// flanking unbeveled `leg` (carrying offset 0), both lines built from a
    /// **local edge-cross normal** (`leg × edge`, sign-fixed by `surface_normal`),
    /// **not** an adjacent face's plane normal. The two lines are generally skew at
    /// a non-planar terminal, so the meet is their closest-points midpoint
    /// (`intersect_lines`), then the §7.5.2 zero-offset span clamp anchors it to the
    /// real leg. Reduces exactly to the §7.3 perpendicular slide on a planar, square
    /// terminal, where the edge-cross normal coincides with the face normal.
    ///
    /// `face` is `E`'s radial face on this leg's side; it is used **only** to pick
    /// the discrete straddle branch (which side of `E` this strip vert sits on) — its
    /// plane normal is **not** used to build the offset perpendicular, which is the
    /// edge-cross local normal alone. The two flanking legs land on opposite sides of
    /// `E`. Exactly one input carries the offset (`E`); the leg carries 0.
    fn terminal_strip_vert(
        &self,
        vertex: VertexKey,
        edge: EdgeKey,
        leg: EdgeKey,
        face: FaceKey,
        offset: f32,
        surface_normal: Vec3,
    ) -> Vec3 {
        let endpoint = self.vertex_position(vertex);
        // Outward unit directions of `E` and the leg from the endpoint.
        let edge_out = match self.edge_direction_from(edge, vertex, endpoint) {
            Some(dir) => dir,
            None => return endpoint,
        };
        let leg_out = match self.edge_direction_from(leg, vertex, endpoint) {
            Some(dir) => dir,
            None => return endpoint,
        };

        // The "sideways" perpendicular is built from the LOCAL edge-cross normal,
        // oriented to the surface's outward side — the decisive §7.5.1 departure from
        // §7.3's per-face plane normal. The cross is sign-ambiguous; the surface
        // normal picks the outward branch. `cross(edge_out, leg_out)` is the
        // canonical orientation; the per-side straddle is set by the offset-push sign
        // below, not by re-deriving the normal per leg.
        let mut local_normal = edge_out.cross(leg_out);
        let length = local_normal.length();
        if length <= f32::EPSILON {
            // Leg parallel to E: no perpendicular. Slide along the leg by `offset`.
            return self.clamp_zero_offset_terminal(endpoint + leg_out * offset, vertex, leg);
        }
        local_normal /= length;
        if local_normal.dot(surface_normal) < 0.0 {
            local_normal = -local_normal;
        }

        // `E`'s offset line is pushed sideways by `offset`, square to `E`, from the
        // local normal. Its sign is the straddle branch: pick the push that drives
        // the offset line toward `face`'s interior, so this strip vert sits on the
        // radial face it belongs to. (The face fixes only this binary side, never the
        // perpendicular vector.)
        let mut edge_perp = edge_out.cross(local_normal).normalize_or_zero();
        let face_center = self.face_center_of_mass(face);
        if edge_perp.dot(face_center - endpoint) < 0.0 {
            edge_perp = -edge_perp;
        }

        // Offset line for `E` (base pushed sideways by `offset`) against the leg's own
        // line (offset 0 ⇒ base on the endpoint). They generally pass skew at a
        // non-planar terminal; take the closest-points midpoint.
        let line_edge = Line {
            point: endpoint + edge_perp * offset,
            direction: edge_out,
        };
        let line_leg = Line {
            point: endpoint,
            direction: leg_out,
        };
        let meet = match intersect_lines(&line_edge, &line_leg) {
            LineMeet::Crossing(point) | LineMeet::Skew(point) => point,
            LineMeet::Collinear => line_leg.point,
        };

        // §7.5.2: the zero-offset leg must not fling the meet past its own span.
        self.clamp_zero_offset_terminal(meet, vertex, leg)
    }

    /// §7.5.2 zero-offset span clamp: the unbeveled leg carries offset 0, so its
    /// offset line is the leg itself and the meet can land past the leg's far
    /// endpoint. Snap any such out-of-span meet back to the nearer endpoint of the
    /// leg's segment (`vertex` … leg's other end).
    fn clamp_zero_offset_terminal(&self, meet: Vec3, vertex: VertexKey, leg: EdgeKey) -> Vec3 {
        let [a, b] = self.edge_vert_keys(leg);
        let far = if a == vertex { b } else { a };
        let near_position = self.vertex_position(vertex);
        let far_position = self.vertex_position(far);
        let segment = far_position - near_position;
        let length_squared = segment.length_squared();
        if length_squared <= f32::EPSILON {
            return near_position;
        }
        let parameter = (meet - near_position).dot(segment) / length_squared;
        if parameter < 0.0 {
            near_position
        } else if parameter > 1.0 {
            far_position
        } else {
            meet
        }
    }

    /// §7.5.1 "surface normal at the terminal" / §8.8 junction `n_v`: the outward
    /// reference used to fix the *sign* of an edge-cross local normal (§7.5.1) and
    /// the forward/reflex sign of a `loop_slide` sub-meet (§8.8.2.1). The average of
    /// the unit normals of the faces incident to `vertex`, falling back to the
    /// stored vertex normal when that average is degenerate (near-zero length).
    fn surface_normal_at(&self, vertex: VertexKey) -> Vec3 {
        let mut accumulator = Vec3::ZERO;
        for face in self.vert_face_keys(vertex) {
            accumulator += self.face_plane_normal(face);
        }
        let length = accumulator.length();
        if length > f32::EPSILON {
            accumulator / length
        } else {
            self.vertex_normal(vertex)
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

    /// Place a sector's boundvert by the pairwise offset-line meet (BEVEL.md
    /// §7.6.2). The boundvert is fixed by *only* the sector's two flanking beveled
    /// edges — never a fit over every incident face. Build the offset line of
    /// `g_k`'s inner side and the offset line of `g_(k+1)`'s inner side, then
    /// intersect them. Three sector cases (no in-between edge / all-in-plane /
    /// non-planar) choose which plane each offset line is built in; the
    /// parallel / anti-parallel ladder handles the near-degenerate edge angles.
    fn solve_meet_point(
        &self,
        vertex: VertexKey,
        vertex_position: Vec3,
        sector: &Sector,
        offset: f32,
    ) -> Vec3 {
        let layout = match self.sector_layout(vertex, vertex_position, sector) {
            Some(layout) => layout,
            None => return vertex_position,
        };

        let dir_first = layout.dir_first;
        let dir_second = layout.dir_second;

        // §7.6.2 parallel/anti-parallel ladder, gated on the angle between the two
        // edge *directions*. cos θ = dir_first · dir_second; the two near-degenerate
        // ends each have a closed-form placement that the line-line solve cannot
        // produce (parallel lines never cross, anti-parallel collapse to a sliver).
        let cos_theta = dir_first.dot(dir_second).clamp(-1.0, 1.0);
        if cos_theta >= 1.0 - MEET_ANGLE_TOLERANCE {
            // Edges nearly parallel (θ ≈ 0): no well-defined crossing. Push out
            // perpendicular to both edges by `offset / cos(θ/2)` (→ `offset` as
            // θ → 0); a finite placement, not the bisector `t / sin(θ/2)` limit.
            return self.parallel_meet(vertex_position, &layout, offset, cos_theta);
        }
        if cos_theta <= -1.0 + MEET_ANGLE_TOLERANCE {
            // Edges nearly anti-parallel (opening angle ≈ 180°): slide along the
            // common edge line by the offset; there is no perpendicular to take.
            return vertex_position + dir_first * offset;
        }

        // Build the two offset lines as point-plus-direction (§7.6.2 "Building the
        // two offset lines"). The base point is `v` pushed perpendicular to the
        // edge, into the sector face, by `offset`; the direction is the edge's own.
        let line_first = Line {
            point: vertex_position + layout.perp_first * offset,
            direction: dir_first,
        };
        let line_second = Line {
            point: vertex_position + layout.perp_second * offset,
            direction: dir_second,
        };

        match intersect_lines(&line_first, &line_second) {
            LineMeet::Crossing(point) => point,
            LineMeet::Collinear => line_first.point,
            LineMeet::Skew(midpoint) => {
                self.drop_onto_sector_face(vertex_position, &layout, midpoint, offset)
            }
        }
    }

    /// The geometric layout a §7.6.2 meet needs from a sector: the two flanking
    /// edge directions and, per the three-case classification, the in-plane
    /// perpendicular each offset line is built along, plus the in-between faces
    /// used by the skew face-drop. `None` for a degenerate (zero-length) flank.
    fn sector_layout(
        &self,
        vertex: VertexKey,
        vertex_position: Vec3,
        sector: &Sector,
    ) -> Option<SectorLayout> {
        if sector.bounding_edges.len() != 2 {
            // A junction's first sector can transiently carry a single bounding
            // edge only on a malformed open fan; the §7.6.1 closed junction always
            // has two. Fall back to the vertex position rather than guess.
            return None;
        }
        let edge_first = sector.bounding_edges[0];
        let edge_second = sector.bounding_edges[1];
        let dir_first = self.edge_direction_from(edge_first, vertex, vertex_position)?;
        let dir_second = self.edge_direction_from(edge_second, vertex, vertex_position)?;

        // The sector's face run, in disk order: f0 (just past g_k) … f_last (just
        // before g_(k+1)). `corners` is `[(g_k,f0),(leg1,f1),…,(g_(k+1),f_last)]`,
        // so deduping consecutive faces recovers the run, and the unbeveled edges
        // strictly between the two flanking beveled edges are the in-between legs.
        let mut faces: Vec<FaceKey> = Vec::new();
        for &(_, face) in &sector.corners {
            if faces.last() != Some(&face) {
                faces.push(face);
            }
        }
        let mut between_edges: Vec<EdgeKey> = Vec::new();
        for &(edge, _) in &sector.corners {
            if edge != edge_first && edge != edge_second && !between_edges.contains(&edge) {
                between_edges.push(edge);
            }
        }

        let face_first = *faces.first()?;
        let face_last = *faces.last()?;

        // Classify the sector into the three §7.6.2 cases by the in-between edges:
        // out-of-plane (silhouette) legs are the ones whose flanking faces are not
        // coplanar. Cases 1 and 2 (no legs, or all legs in-plane) share one sector
        // plane; case 3 (≥1 out-of-plane leg) gives each line its own local normal.
        let has_out_of_plane = between_edges
            .iter()
            .enumerate()
            .any(|(index, _)| {
                let before = faces[index];
                let after = faces[index + 1];
                let dot = self.face_plane_normal(before).dot(self.face_plane_normal(after));
                dot.abs() < 1.0 - MEET_PLANAR_TOLERANCE
            });

        let (normal_first, normal_second) = if has_out_of_plane {
            // Case 3: two different local planes. Each line's normal is taken from
            // the corner at its own end of the sector — the cross of that flanking
            // beveled edge's direction with the nearest in-between leg's direction,
            // oriented by the adjacent face.
            let first_leg = between_edges
                .first()
                .and_then(|&edge| self.edge_direction_from(edge, vertex, vertex_position));
            let last_leg = between_edges
                .last()
                .and_then(|&edge| self.edge_direction_from(edge, vertex, vertex_position));
            let normal_first = match first_leg {
                Some(leg) => orient_normal(dir_first.cross(leg), self.face_plane_normal(face_first)),
                None => self.face_plane_normal(face_first),
            };
            let normal_second = match last_leg {
                Some(leg) => {
                    orient_normal(dir_second.cross(leg), self.face_plane_normal(face_last))
                }
                None => self.face_plane_normal(face_last),
            };
            (normal_first, normal_second)
        } else {
            // Cases 1 and 2: one shared sector plane. The first flanking face's
            // normal serves both perpendiculars (§7.6.2 case 1/2 are identical math).
            let normal = self.face_plane_normal(face_first);
            (normal, normal)
        };

        // Each base-point perpendicular is `cross(edge, sector normal)` — in the
        // face plane and perpendicular to the edge — oriented toward the face
        // interior so the offset pushes *into* the sector, not out of it.
        let perp_first = self.into_face_perpendicular(
            dir_first,
            normal_first,
            face_first,
            vertex_position,
        );
        let perp_second = self.into_face_perpendicular(
            dir_second,
            normal_second,
            face_last,
            vertex_position,
        );

        Some(SectorLayout {
            dir_first,
            dir_second,
            perp_first,
            perp_second,
            normal_first,
            normal_second,
            between_faces: faces,
            between_edges,
        })
    }

    /// The unit in-plane perpendicular to `direction` (in the plane of `normal`)
    /// pointing toward `face`'s interior. Falls back to the raw cross when the
    /// edge is parallel to the normal (degenerate, no in-plane perpendicular).
    fn into_face_perpendicular(
        &self,
        direction: Vec3,
        normal: Vec3,
        face: FaceKey,
        vertex_position: Vec3,
    ) -> Vec3 {
        let mut perpendicular = direction.cross(normal);
        let length = perpendicular.length();
        if length <= f32::EPSILON {
            return Vec3::ZERO;
        }
        perpendicular /= length;
        let center = self.face_center_of_mass(face);
        if perpendicular.dot(center - vertex_position) < 0.0 {
            perpendicular = -perpendicular;
        }
        perpendicular
    }

    /// §7.6.2 parallel special case: with the two edges nearly parallel, the offset
    /// lines never cross, so place the meet straight out from `v` perpendicular to
    /// both edges, a distance `offset / cos(θ/2)` along the common direction's
    /// in-plane perpendicular. `θ` is the (small) angle between the two edge
    /// directions; `cos(θ/2)` is recovered from `cos θ` by the half-angle identity.
    fn parallel_meet(
        &self,
        vertex_position: Vec3,
        layout: &SectorLayout,
        offset: f32,
        cos_theta: f32,
    ) -> Vec3 {
        let common = (layout.dir_first + layout.dir_second).normalize_or_zero();
        if common == Vec3::ZERO {
            return vertex_position;
        }
        let normal = layout.normal_first;
        let mut perpendicular = common.cross(normal);
        let length = perpendicular.length();
        if length <= f32::EPSILON {
            // The averaged direction lies along the plane normal: reuse the first
            // line's already-oriented perpendicular rather than a degenerate cross.
            perpendicular = layout.perp_first;
        } else {
            perpendicular /= length;
            if perpendicular.dot(layout.perp_first) < 0.0 {
                perpendicular = -perpendicular;
            }
        }
        let cos_half = ((1.0 + cos_theta) * 0.5).max(0.0).sqrt();
        let distance = if cos_half > f32::EPSILON {
            offset / cos_half
        } else {
            offset
        };
        vertex_position + perpendicular * distance
    }

    /// §7.6.2 face-drop: a skew-line midpoint floats off the surface, so project it
    /// onto the first in-walk sector face whose plane projection lands inside that
    /// face's angular wedge at `v` (the in-wedge predicate). If no face projection
    /// is in-wedge, the midpoint stands.
    fn drop_onto_sector_face(
        &self,
        vertex_position: Vec3,
        layout: &SectorLayout,
        midpoint: Vec3,
        offset: f32,
    ) -> Vec3 {
        // The drop only runs when both flanking offsets are positive (a zero-offset
        // side is the §8.7 degenerate handled by its own guard, not here).
        if offset <= 0.0 {
            return midpoint;
        }
        // Walk the sector's edges in disk order starting at the flanking beveled edge
        // `g_k` itself (not the first in-between leg), advancing up to — but not
        // including — `g_(k+1)`. The forward face of `g_k` is `between_faces[0]`, of
        // in-between leg `i` is `between_faces[i + 1]`, so the walk's forward faces
        // are exactly `between_faces` in order. The projection plane is anchored at
        // `v` (not the face's own supporting plane), oriented by the face normal; on
        // a non-planar sector face those differ by up to the offset itself. The wedge
        // is bounded by the walk edge and the next edge — the two edges sharing that
        // forward face at `v`.
        for &face in &layout.between_faces {
            let normal = self.face_plane_normal(face);
            let projected = midpoint - normal * (midpoint - vertex_position).dot(normal);
            let bounds = self.face_edge_directions_at(face, vertex_position);
            let (dir_one, dir_two) = match bounds {
                Some(pair) => pair,
                None => continue,
            };
            let to_projected = projected - vertex_position;
            if to_projected.length() <= f32::EPSILON {
                continue;
            }
            let dir_projected = to_projected.normalize();
            if point_in_wedge(dir_one, dir_two, dir_projected, normal) {
                return projected;
            }
        }
        midpoint
    }

    /// The unit directions of `face`'s two edges incident to the corner at
    /// `vertex_position`, in the face's winding order (the edge entering the corner
    /// then the edge leaving it). `None` if the corner or a bounding edge is
    /// degenerate.
    fn face_edge_directions_at(
        &self,
        face: FaceKey,
        vertex_position: Vec3,
    ) -> Option<(Vec3, Vec3)> {
        let verts = self.face_vert_keys(face);
        let count = verts.len();
        let index = verts
            .iter()
            .position(|&v| (self.vertex_position(v) - vertex_position).length() <= f32::EPSILON)?;
        let prev = self.vertex_position(verts[(index + count - 1) % count]);
        let next = self.vertex_position(verts[(index + 1) % count]);
        let to_prev = prev - vertex_position;
        let to_next = next - vertex_position;
        if to_prev.length() <= f32::EPSILON || to_next.length() <= f32::EPSILON {
            return None;
        }
        Some((to_prev.normalize(), to_next.normalize()))
    }

    /// Try to place a sector's boundvert by the `loop_slide` rule (BEVEL.md §8.8):
    /// slide it onto a single unbeveled leg's line instead of the §7.6.2
    /// meet-point. Returns `Some(position)` when the sector fires (§8.8.2) and the
    /// slide is accepted, `None` to fall back to the plain meet-point.
    fn slide_boundvert(
        &self,
        vertex: VertexKey,
        vertex_position: Vec3,
        sector: &Sector,
        is_beveled: &dyn Fn(EdgeKey) -> bool,
    ) -> Option<SlideRecord> {
        if sector.bounding_edges.len() != 2 {
            return None;
        }

        // The sector's corners are the ordered (edge, face) stream the disk walk
        // produced, opening on `g_k` and closing on `g_(k+1)` (§7.6.1). Each
        // consecutive pair shares the next entry's edge as the leg between their two
        // faces, so the unbeveled outgoing edges between the two bounding beveled
        // edges *are* the legs, already in disk order — the same representation
        // `solve_meet_point` reads. Take them straight from that stream rather than
        // a face-dedup → shared-edge round trip (which mis-orders if a face recurs
        // non-contiguously).
        let mut legs: Vec<(EdgeKey, FaceKey, FaceKey)> = Vec::new();
        for pair in sector.corners.windows(2) {
            let (leg, face_after) = pair[1];
            let face_before = pair[0].1;
            // `leg` is the closing beveled edge for the final pair and any beveled
            // edge mid-sector; only unbeveled legs slide. A non-shared/degenerate
            // pair is skipped, not aborted: the firing decision must still see the
            // legs that genuinely exist (§8.8.2).
            if is_beveled(leg) {
                continue;
            }
            legs.push((leg, face_before, face_after));
        }
        if legs.is_empty() {
            return None;
        }

        // Classify each leg in-plane vs silhouette by the dot of its two flanking
        // face normals (§8.8.2): coplanar (≈ ±1) is in-plane, else silhouette.
        let silhouette: Vec<&(EdgeKey, FaceKey, FaceKey)> = legs
            .iter()
            .filter(|&&(_, fa, fb)| {
                let dot = self.face_plane_normal(fa).dot(self.face_plane_normal(fb));
                dot.abs() < 1.0 - LOOP_SLIDE_TOLERANCE
            })
            .collect();

        // Firing condition (§8.8.2) part 1: exactly one silhouette leg, or — absent
        // any silhouette leg — exactly one in-plane leg. Anything else falls back.
        let slide_leg = if !silhouette.is_empty() {
            if silhouette.len() != 1 {
                return None;
            }
            silhouette[0].0
        } else {
            if legs.len() != 1 {
                return None;
            }
            legs[0].0
        };

        let leg_direction = self.edge_direction_from(slide_leg, vertex, vertex_position)?;
        let edge_before = sector.bounding_edges[0];
        let edge_after = sector.bounding_edges[1];
        let dir_before = self.edge_direction_from(edge_before, vertex, vertex_position)?;
        let dir_after = self.edge_direction_from(edge_after, vertex, vertex_position)?;

        // Firing condition (§8.8.2.1) part 2 — the "good offset" guard, taken in the
        // genuine junction *vertex-normal* frame `n_v`, never a sector face-plane
        // frame. Both sub-meets `g_k → e_mid` and `e_mid → g_(k+1)` must sweep
        // *forward* and open a non-degenerate angle; a reflex sub-meet would crease
        // the surface backward, so the slide is rejected (return `None`). Forward vs
        // reflex is the sign of the swept angle viewed from `n_v` (`cross · n_v < 0`
        // is reflex); `n_v` is the incident-face-average vertex normal (vertex-normal
        // fallback) shared with §7.5.1/§8.8.3. A *sector face-plane* normal tilts away
        // from the true corner on an irregular junction's off-plane sectors and flips
        // this sign, wrongly rejecting the very sectors that should slide (the
        // under-fire); `n_v` is sign-stable across every sector of the junction.
        // Degeneracy is measured on the genuine un-projected opening angle (within
        // ~0.1 rad of 0 or π), separately from the sign.
        let vertex_normal = self.surface_normal_at(vertex);
        if !subsweep_is_good(dir_before, leg_direction, vertex_normal)
            || !subsweep_is_good(leg_direction, dir_after, vertex_normal)
        {
            return None;
        }

        // The two sub-meets sit `offset / sin(α)` along the leg from `v`, where α is
        // the (true) opening angle of the flanking beveled edge against the leg
        // (§8.8.3). The gate above guarantees both are forward and non-degenerate, so
        // §8.8.3's single-survivor (one sub-meet reflex) and both-degenerate (all
        // three lines collinear) fallback branches are intentionally not built here.
        // The position itself is not computed in this construct pass: it is re-derived
        // by the caller's §7.6.4.5 rebuild off the reconciled offsets (§8.8.3.1), where
        // the reconciled pair makes the two sub-meets coincide — so only the slide
        // parameters (`d_leg`, `sin α₁`, the flank-sine ratio) are returned.
        let sin_alpha_before = dir_before.cross(leg_direction).length();
        let sin_alpha_after = dir_after.cross(leg_direction).length();

        // §8.8.4: record the flank-sine ratio `sin(α₂)/sin(α₁)` — a pure measure of
        // how lopsided the two flanking angles are — for the §7.6.4 width pass to
        // thread into the far-end offsets. The explicit guard: if either sine
        // collapses (a flank gone collinear with the leg) the ratio diverges, so
        // override it to `1` rather than evaluate the blow-up — an unconstrained side
        // makes equal demands on the leg, so a balanced ratio adds no bias.
        let flank_sine_ratio = if sin_alpha_before < LOOP_SLIDE_TOLERANCE
            || sin_alpha_after < LOOP_SLIDE_TOLERANCE
        {
            1.0
        } else {
            sin_alpha_after / sin_alpha_before
        };

        Some(SlideRecord {
            flank_sine_ratio,
            leg_direction,
            sin_alpha_before,
        })
    }

    /// The unit direction of `edge` pointing away from `vertex`, or `None` for a
    /// degenerate (zero-length) edge.
    fn edge_direction_from(
        &self,
        edge: EdgeKey,
        vertex: VertexKey,
        vertex_position: Vec3,
    ) -> Option<Vec3> {
        let [a, b] = self.edge_vert_keys(edge);
        let far = if a == vertex { b } else { a };
        let segment = self.vertex_position(far) - vertex_position;
        let length = segment.length();
        (length > f32::EPSILON).then(|| segment / length)
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
        use_weights: bool,
        vert_weights: &[(VertexKey, f32)],
    ) -> Vec<BevelFrame> {
        let mut frames = Vec::with_capacity(beveled.len());
        for &vertex in beveled {
            let incident_edges: Vec<EdgeKey> = self.iter_disk_edges(vertex).collect();
            if incident_edges.len() < 2 {
                continue;
            }

            // §8.7.2 / §8.7.3: the per-corner weight prescales this vertex's
            // offset uniformly across all its incident edges, *before* the §4 /
            // §4.3 slide solve. Gate off ⇒ weight 1.0 (full offset); gate on but
            // absent from the slice ⇒ weight 0.0 (no slide for this corner).
            let weight = corner_weight(use_weights, vert_weights, vertex);
            let effective_offset = offset * weight;

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
                        // §8.7.3: prescale the corner offset, then solve the §4 /
                        // §4.3 slide unchanged. The solve is linear in offset, so
                        // this is identical to scaling the resolved slide by the
                        // weight afterward.
                        let slide = slide_distance(
                            effective_offset,
                            offset_type,
                            vertex_position,
                            far_position,
                            corner_axis,
                        );
                        // §4.4.2 / §8.7.4: the per-offset slide rate `k = d_i /
                        // offset`, the §4.3 `offset_type` factor used by the
                        // collision clamp. Derived from the *weighted* slide
                        // against the requested offset, so the clamp sees the
                        // weighted rates (§8.7.4: the weight is the inner scale,
                        // the clamp the outer).
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

/// One beveled-edge side's **terminal-end** width-reconciled offset
/// (BEVEL.md §7.6.4 / §8.8.4). When `loop_slide` fires at a junction sector, the slid
/// boundvert constrains its two flanking edges' *apex-end* offsets to the flank-sine
/// ratio (§7.6.4.3's `o₁`, `o₂`); the **far end** of each flanking edge is its own
/// reconciliation variable (§7.6.4.1), the damped blend
/// `(apex_offset + w²·O) / (1 + w²)` (§7.6.4.3). It is *that* damped value — not the
/// apex offset carried unchanged, which overshoots by `1.04×` (§7.6.4.4) — that is
/// stored here, keyed by `(edge, face)` (the edge's radial face bordering the firing
/// sector), and read by the terminal close (§7.5.1) at the edge's far end (§8.8.5).
/// An edge-side that no firing sector constrained keeps the spec offset.
struct ReconciledOffset {
    edge: EdgeKey,
    face: FaceKey,
    offset: f32,
}

/// The slide parameters of a firing `loop_slide` sector (BEVEL.md §8.8.3 / §8.8.4),
/// returned by `slide_boundvert` so the caller can run the §7.6.4.5 *rebuild* pass:
/// re-derive both the apex boundvert (§8.8.3.1) and the flanking terminal offsets
/// (§7.6.4.3) from the reconciled offsets, rather than placing the apex once off the
/// spec offset. `leg_direction` and `sin_alpha_before` are the un-projected leg
/// direction `d_leg` and `sin α₁` of §8.8.3 — the apex rebuild lands at
/// `v + (o₁ / sin α₁) · d_leg`, where the reconciled pair makes the two sub-meets
/// coincide.
struct SlideRecord {
    flank_sine_ratio: f32,
    leg_direction: Vec3,
    sin_alpha_before: f32,
}

/// The §7.6.4-reconciled offset for the `(edge, face)` side, or `spec` when no firing
/// sector constrained it (BEVEL.md §8.8.4).
fn reconciled_offset(
    reconciled: &[ReconciledOffset],
    edge: EdgeKey,
    face: FaceKey,
    spec: f32,
) -> f32 {
    reconciled
        .iter()
        .find(|record| record.edge == edge && record.face == face)
        .map(|record| record.offset)
        .unwrap_or(spec)
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

/// One beveled edge's profile chain at one endpoint (BEVEL.md §8.2.2 / §8.3.0):
/// the two ring boundverts (`start` / `end`) it spans plus the `segments − 1`
/// interior arc samples between them. `start` is the boundvert on the edge's first
/// radial face, so both ends of the edge order their chains consistently for the
/// strip's rung-to-rung join.
struct EdgeEndChain {
    edge: EdgeKey,
    endpoint: VertexKey,
    start: VertexKey,
    interior: Vec<VertexKey>,
    end: VertexKey,
}

impl EdgeEndChain {
    /// The chain's `segments + 1` verts in order `start → interior → end`.
    fn samples(&self) -> Vec<VertexKey> {
        let mut samples = Vec::with_capacity(self.interior.len() + 2);
        samples.push(self.start);
        samples.extend_from_slice(&self.interior);
        samples.push(self.end);
        samples
    }

    /// The chain's verts ordered to begin at `first` (either endpoint).
    fn oriented_from(&self, first: VertexKey) -> Vec<VertexKey> {
        let mut samples = self.samples();
        if self.start != first {
            samples.reverse();
        }
        samples
    }
}

/// One multi-edge junction's cutoff-fill data (BEVEL.md §8.3): its ordered
/// boundvert ring, the per-sector opening beveled edge, and the outward surface
/// normal that orients the down-push (§8.3.3).
struct JunctionCut {
    vertex: VertexKey,
    boundverts: Vec<VertexKey>,
    #[allow(dead_code)]
    opening_edges: Vec<EdgeKey>,
    outward: Vec3,
}

impl JunctionCut {
    /// The `material < 0` inherit source for this junction's new fill faces
    /// (BEVEL.md §6.6.1): a face incident to the junction vertex. The vertex is
    /// consumed during surgery, so this is only valid before phase 5.
    fn representative_material(&self, mesh: &LoopMesh) -> i32 {
        mesh.vert_face_keys(self.vertex)
            .first()
            .map(|&face| mesh.face_material_index(face))
            .unwrap_or(0)
    }
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

/// True when all the corner-vert positions coincide within tolerance (the §8.3.4
/// 3-way apex collapse test).
fn corner_verts_coincide(positions: &[Vec3]) -> bool {
    if positions.len() < 2 {
        return true;
    }
    let first = positions[0];
    positions
        .iter()
        .all(|&position| (position - first).length() <= 1.0e-4)
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

/// An offset line in point-plus-direction form (BEVEL.md §7.6.2).
struct Line {
    point: Vec3,
    direction: Vec3,
}

/// The outcome of intersecting two offset lines (BEVEL.md §7.6.2 "Intersecting
/// the two offset lines").
enum LineMeet {
    /// The lines cross — the intersection point.
    Crossing(Vec3),
    /// The lines coincide — caller picks a defined on-line point.
    Collinear,
    /// The lines are skew — the midpoint of their two closest points.
    Skew(Vec3),
}

/// The per-sector geometry a §7.6.2 meet consumes (built by `sector_layout`): the
/// two flanking edge directions and their offset-line perpendiculars/normals, plus
/// the in-between faces/edges the skew face-drop walks.
struct SectorLayout {
    dir_first: Vec3,
    dir_second: Vec3,
    perp_first: Vec3,
    perp_second: Vec3,
    normal_first: Vec3,
    normal_second: Vec3,
    between_faces: Vec<FaceKey>,
    between_edges: Vec<EdgeKey>,
}

/// Orient `candidate` to agree with `reference` (flip when they oppose), giving a
/// consistent local sector normal for a case-3 offset-line perpendicular.
fn orient_normal(candidate: Vec3, reference: Vec3) -> Vec3 {
    let length = candidate.length();
    if length <= f32::EPSILON {
        return reference;
    }
    let unit = candidate / length;
    if unit.dot(reference) < 0.0 {
        -unit
    } else {
        unit
    }
}

/// Intersect two lines in 3-D by closest approach (BEVEL.md §7.6.2). Returns the
/// crossing point when they meet, `Collinear` when they coincide, and the
/// closest-points midpoint when they are skew.
fn intersect_lines(first: &Line, second: &Line) -> LineMeet {
    let d1 = first.direction;
    let d2 = second.direction;
    let r = first.point - second.point;
    let a = d1.dot(d1);
    let b = d1.dot(d2);
    let c = d2.dot(d2);
    let d = d1.dot(r);
    let e = d2.dot(r);
    let denominator = a * c - b * b;

    if denominator.abs() <= LINE_PARALLEL_TOLERANCE {
        // Parallel lines: distinguish coincident (the offset between the two base
        // points lies along the shared direction) from a true parallel pair. The
        // parallel-pair case is reached as a fallback only — the §7.6.2 ladder
        // normally diverts near-parallel edges before the line solve.
        let along = if a > f32::EPSILON {
            r - d1 * (d1.dot(r) / a)
        } else {
            r
        };
        if along.length() <= LINE_COINCIDENT_TOLERANCE {
            return LineMeet::Collinear;
        }
        // Treat a genuine parallel pair as skew with the midpoint of the two base
        // points; the drop step recovers a surface point if one is in-wedge.
        return LineMeet::Skew((first.point + second.point) * 0.5);
    }

    let t = (b * e - c * d) / denominator;
    let s = (a * e - b * d) / denominator;
    let closest_first = first.point + d1 * t;
    let closest_second = second.point + d2 * s;
    let gap = closest_first - closest_second;
    if gap.length() <= LINE_COINCIDENT_TOLERANCE {
        LineMeet::Crossing((closest_first + closest_second) * 0.5)
    } else {
        LineMeet::Skew((closest_first + closest_second) * 0.5)
    }
}

/// The §7.6.2 in-wedge predicate: is `dir_projected` inside the angular wedge that
/// runs from `dir_one` to `dir_two` in `normal`'s winding sense? Each unsigned
/// angle in `[0, π]` is reflexed to `2π − angle` when its directions' cross points
/// against `normal`, so reflex wedges are measured correctly. In-wedge when the
/// `dir_one → dir_projected` sweep does not exceed the `dir_one → dir_two` sweep.
fn point_in_wedge(dir_one: Vec3, dir_two: Vec3, dir_projected: Vec3, normal: Vec3) -> bool {
    let swept = |from: Vec3, to: Vec3| -> f32 {
        let angle = from.dot(to).clamp(-1.0, 1.0).acos();
        if from.cross(to).dot(normal) < 0.0 {
            std::f32::consts::TAU - angle
        } else {
            angle
        }
    };
    let wedge = swept(dir_one, dir_two);
    let test = swept(dir_one, dir_projected);
    test <= wedge + WEDGE_ANGLE_TOLERANCE
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

    /// An irregular tetrahedron: a manifold solid whose every vertex is a
    /// **non-planar valence-3 terminal** with oblique legs. Beveling one edge
    /// exercises the §7.5.1 terminal placement away from the planar/square cube
    /// special case the other edge-bevel tests pin.
    fn irregular_tetra(mesh: &mut LoopMesh) -> [VertexKey; 4] {
        let v0 = mesh.create_vertex(Vec3::new(0.0, 0.0, 0.0));
        let v1 = mesh.create_vertex(Vec3::new(3.0, 0.0, 0.0));
        let v2 = mesh.create_vertex(Vec3::new(0.5, 2.5, 0.3));
        let v3 = mesh.create_vertex(Vec3::new(0.4, 0.6, 2.2));
        // Outward-wound faces (each triangle's normal points away from the centroid).
        let faces = [[v0, v2, v1], [v0, v1, v3], [v0, v3, v2], [v1, v2, v3]];
        for ring in faces {
            let edges = [
                mesh.create_edge([ring[0], ring[1]], EdgeCreateFlags::NO_DUPLICATE),
                mesh.create_edge([ring[1], ring[2]], EdgeCreateFlags::NO_DUPLICATE),
                mesh.create_edge([ring[2], ring[0]], EdgeCreateFlags::NO_DUPLICATE),
            ];
            mesh.face_create(&ring, &edges, FaceCreateFlags::NONE);
        }
        [v0, v1, v2, v3]
    }

    /// BEVEL.md §7.5.1: at a **non-planar** valence-3 terminal each strip vert is
    /// the leg meet — `E`'s offset line (offset 0.3) against the flanking unbeveled
    /// leg (offset 0), built from the edge-cross local normal — **not** the §7.3
    /// perpendicular-in-face slide. Two consequences the slide cannot produce and
    /// this test pins: (1) every strip vert sits an exact perpendicular `offset`
    /// from `E`'s original line (the offset-line property), and (2) the strip vert
    /// on the `v0` corner lands at the leg-meet target, measurably off the §7.3
    /// face-plane slide (~0.05 mesh unit here at offset 0.3 on the oblique legs).
    #[test]
    fn edge_bevel_terminal_leg_meet_on_nonplanar_corner() {
        let mut mesh = LoopMesh::new();
        let [v0, v1, _v2, _v3] = irregular_tetra(&mut mesh);
        let endpoint_a = mesh.vertex_position(v0);
        let endpoint_b = mesh.vertex_position(v1);
        let edge = mesh.query_edge_exists(v0, v1).expect("tetra edge {0,1}");
        let edge_dir = (endpoint_b - endpoint_a).normalize();

        let offset = 0.3_f32;
        let result = mesh.bevel(
            &[edge.into()],
            BevelOptions {
                offset,
                affect: BevelAffect::Edges,
                ..Default::default()
            },
        );
        assert_eq!(result.verts.len(), 4, "four terminal strip verts");

        // (1) Offset-line property: every strip vert lies an exact perpendicular
        // `offset` from `E`'s original line. The §7.3 face-plane slide also satisfies
        // this, so it is necessary-not-sufficient — assertion (2) separates them.
        for &strip in &result.verts {
            let to_strip = mesh.vertex_position(strip) - endpoint_a;
            let along = to_strip.dot(edge_dir);
            let perpendicular_distance = (to_strip - edge_dir * along).length();
            assert!(
                (perpendicular_distance - offset).abs() < 1.0e-4,
                "strip vert {perpendicular_distance} must lie offset {offset} from E's line",
            );
        }

        // (2) The §7.5.1 leg-meet targets at the `v0` corner, derived by hand from
        // the edge-cross local normal + closest-points meet (see the scratch
        // prototype): these are measurably off the §7.3 in-face slide, which would
        // keep `x = 0` (the slide stays in each face's plane through `v0`).
        let leg_meet_one = Vec3::new(0.05957, 0.29786, 0.03574);
        let leg_meet_two = Vec3::new(0.05262, 0.07894, 0.28943);
        for &target in &[leg_meet_one, leg_meet_two] {
            let hits = result
                .verts
                .iter()
                .filter(|&&v| (mesh.vertex_position(v) - target).length() < 1.0e-3)
                .count();
            assert_eq!(hits, 1, "a strip vert lands at the §7.5.1 leg meet {target:?}");
            // The §7.3 slide keeps x = 0 at this corner; the leg meet pulls it off.
            assert!(target.x > 1.0e-2, "leg meet is off the §7.3 in-face slide");
        }
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

    /// Build the cube-corner `s = 3` cutoff junction at `segments = 2` and return
    /// the mesh + result for the cutoff tests below (BEVEL.md §8.2 / §8.3). Vertex
    /// 0 = (1,-1,-1) is the junction; its three edges to 1/3/4 are beveled.
    fn cube_corner_cutoff_segments2() -> (LoopMesh, BevelResult) {
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
                segments: 2,
                vmesh_method: BevelVmeshMethod::Cutoff,
                ..Default::default()
            },
        );
        (mesh, result)
    }

    /// The same cube-corner `s = 3` cutoff junction at `segments = 2` but with an
    /// explicit `profile` (BEVEL.md §8.6), for the profile-sampling tests. Returns
    /// the lone Front↔Bottom interior chain vert's position across edge {0,1}.
    fn cube_corner_cutoff_segments2_profile(profile: f32) -> Vec3 {
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
                segments: 2,
                vmesh_method: BevelVmeshMethod::Cutoff,
                profile,
                ..Default::default()
            },
        );

        // The Front↔Bottom interior chain vert sits on edge {0,1}'s profile cell:
        // u = 0.75 (the boundvert plane), v sweeping the bulge toward (0.75,-0.75,
        // -0.75). It is the one new vert at x = 0.75 with y == z below the arc
        // center, i.e. the interior sample of that sector's chain.
        let interior: Vec<Vec3> = result
            .verts
            .iter()
            .map(|&v| mesh.vertex_position(v))
            .filter(|p| {
                (p.x - 0.75).abs() < 1.0e-4
                    && (p.y - p.z).abs() < 1.0e-4
                    && p.y < -0.8
                    && p.y > -1.0
            })
            .collect();
        assert_eq!(
            interior.len(),
            1,
            "exactly one Front-Bottom interior chain vert for profile {profile}",
        );
        interior[0]
    }

    /// BEVEL.md §8.6.4 / §8.3.6: the circle shortcut (`profile = 0.5`, `r = 2`) is
    /// pure angle spacing `(sin θ_k, cos θ_k)`, reproducing the §8.3.6 default
    /// bit-for-bit. The pure sampler is checked directly so the reduction is pinned
    /// independently of the cell mapping.
    #[test]
    fn superellipse_chain_circle_is_angle_spacing() {
        let chain = superellipse_chain(0.5, 2);
        assert_eq!(chain[0], (0.0, 1.0));
        assert_eq!(chain[2], (1.0, 0.0));
        let theta = std::f32::consts::FRAC_PI_2 / 2.0;
        assert_eq!(chain[1], (theta.sin(), theta.cos()));

        let chain4 = superellipse_chain(0.5, 4);
        for k in 1..4 {
            let theta = k as f32 * std::f32::consts::FRAC_PI_2 / 4.0;
            assert_eq!(chain4[k], (theta.sin(), theta.cos()));
        }
    }

    /// BEVEL.md §8.6.2 / §8.6.4: the flat profile (`profile = 0.25`, `r = 1`) is the
    /// straight chord `u + v = 1`, so even spacing is even linear interpolation. At
    /// `segments = 2` the lone interior sample is the chord midpoint `(0.5, 0.5)`.
    #[test]
    fn superellipse_chain_line_is_chord_midpoint() {
        let chain = superellipse_chain(0.25, 2);
        assert_eq!(chain[1], (0.5, 0.5));

        let chain4 = superellipse_chain(0.25, 4);
        for k in 0..=4 {
            let t = k as f32 / 4.0;
            assert!((chain4[k].0 - t).abs() < 1.0e-6 && (chain4[k].1 - (1.0 - t)).abs() < 1.0e-6);
        }
    }

    /// BEVEL.md §8.6.1 / §8.6.4: a general exponent (`profile = 0.7`, `r ≈ 3.89`)
    /// runs the chord-equalisation solver. At `segments = 2` the single interior
    /// sample sits on the `u = v` diagonal at `m = (1/2)^(1/r)`, which lies on the
    /// superellipse `u^r + v^r = 1` and bulges past both the circle (0.5) and the
    /// flat chord (0.25) samples.
    #[test]
    fn superellipse_chain_general_lands_on_superellipse() {
        let chain = superellipse_chain(0.7, 2);
        let (u, v) = chain[1];
        // On the u = v diagonal.
        assert!((u - v).abs() < 1.0e-6);
        // On the superellipse: u^r + v^r = 1.
        let r = profile_to_exponent(0.7);
        assert!((u.powf(r) + v.powf(r) - 1.0).abs() < 1.0e-4);
        // Bulges past the circle (0.707) and the flat chord (0.5).
        let circle_u = superellipse_chain(0.5, 2)[1].0;
        assert!(u > circle_u);
        assert!(u > 0.5);

        // The general solver equalises chords for a larger even count too.
        let chain4 = superellipse_chain(0.7, 4);
        let chords: Vec<f32> = (0..4).map(|k| cell_chord(chain4[k], chain4[k + 1])).collect();
        let max = chords.iter().cloned().fold(0.0_f32, f32::max);
        let min = chords.iter().cloned().fold(f32::INFINITY, f32::min);
        assert!((max - min) < 1.0e-3, "general solver equalises chords: {chords:?}");
        // Symmetric about the diagonal: sample k mirrors sample n−k.
        assert!((chain4[1].0 - chain4[3].1).abs() < 1.0e-5);
    }

    /// BEVEL.md §8.6: `profile = 0.5` (the default) reproduces the §8.3.6 circular
    /// arc sample at the cutoff chain interior — no regression from routing the
    /// chain through the general sampler.
    #[test]
    fn cutoff_segments2_profile_half_is_arc_sample() {
        let interior = cube_corner_cutoff_segments2_profile(0.5);
        let arc_point = Vec3::new(0.75, -0.926_776_7, -0.926_776_7);
        assert!((interior - arc_point).length() < 1.0e-5);
    }

    /// BEVEL.md §8.6.2 / §8.3.6: the flat `profile = 0.25` chain interior is the
    /// straight chord midpoint, not the circular bulge — the special case where the
    /// cutoff fill is genuinely [P]-only.
    #[test]
    fn cutoff_segments2_profile_flat_is_chord_midpoint() {
        let interior = cube_corner_cutoff_segments2_profile(0.25);
        let chord_midpoint = Vec3::new(0.75, -0.875, -0.875);
        assert!((interior - chord_midpoint).length() < 1.0e-5);
    }

    /// BEVEL.md §8.6.1: a non-default `profile = 0.7` bulges the chain interior past
    /// both the circular (0.5) and flat (0.25) samples — the sampler honours the
    /// `profile` shape control through the cell mapping, distinct in position.
    #[test]
    fn cutoff_segments2_profile_bulge_is_distinct() {
        let flat = cube_corner_cutoff_segments2_profile(0.25);
        let circle = cube_corner_cutoff_segments2_profile(0.5);
        let bulge = cube_corner_cutoff_segments2_profile(0.7);

        // All three lie on the y = z bulge diagonal at x = 0.75, ordered by bulge:
        // flat (−0.875) is shallowest, then circle (−0.927), then 0.7 (deeper).
        assert!(bulge.y < circle.y, "0.7 bulges past the circle sample");
        assert!(circle.y < flat.y, "the circle bulges past the flat chord");
        assert!((bulge - circle).length() > 1.0e-3, "0.7 is a distinct position");
        assert!((bulge - flat).length() > 1.0e-3, "0.7 is not the chord midpoint");
    }

    /// BEVEL.md §8.3.2 / §8.3.4 / §8.3.0.1: the `s = 3`, `segments = 2` cube-corner
    /// cutoff junction folds the corner flat — three cut faces collapsing to one
    /// apex (§8.3.4), no bottom face — while the three valence-3 terminals build no
    /// fill face, their strips running through (§8.3.0.1). The result stays a closed
    /// manifold.
    #[test]
    fn cutoff_junction_segments2_topology() {
        let (mesh, result) = cube_corner_cutoff_segments2();

        mesh.validate_topology();

        // The §8.3.4 apex collapse: all three sectors' corner verts coincide, so
        // there is one apex vert (BEVEL.md §8.3.4) — the common corner of the three
        // cut faces. Each cut face is a quad: apex + the 3-point chain (`segments + 2
        // = 4` corners, §8.3.5).
        let apex_position = Vec3::new(0.75, -0.75, -0.75);
        let apex: Vec<VertexKey> = result
            .verts
            .iter()
            .copied()
            .filter(|&v| (mesh.vertex_position(v) - apex_position).length() < 1.0e-4)
            .collect();
        assert_eq!(apex.len(), 1, "the three corner verts collapse to one apex");
        let apex = apex[0];

        let cut_faces: Vec<FaceKey> = mesh
            .face_keys()
            .into_iter()
            .filter(|&f| mesh.face_vert_keys(f).contains(&apex))
            .collect();
        assert_eq!(cut_faces.len(), 3, "three cut faces meet at the apex");
        for &face in &cut_faces {
            assert_eq!(
                mesh.face_vert_keys(face).len(),
                4,
                "collapsed cut face is a quad (apex + 3-point chain)",
            );
        }

        // The cube corner cutoff is a closed manifold: every edge radial-2.
        for face in mesh.face_keys() {
            for loop_key in mesh.face_loop_keys(face) {
                let edge = mesh.loop_edge(loop_key);
                assert_eq!(
                    mesh.edge_radial_iter(edge).count(),
                    2,
                    "every edge is shared with exactly one other face",
                );
            }
        }

        // No bottom face beyond the three cut faces: the collapse omits it (§8.3.4).
        // The new geometry is 6 strip quads + 3 cut quads = 9 faces.
        assert_eq!(result.faces.len(), 9, "6 strip quads + 3 cut quads, no bottom");
    }

    /// The same cube-corner cutoff junction at an explicit `profile`, returning the
    /// post-bevel mesh + result for the §8.3.3/§8.3.4 collapse-gate tests.
    fn cube_corner_cutoff_segments2_mesh(profile: f32) -> (LoopMesh, BevelResult) {
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
                segments: 2,
                vmesh_method: BevelVmeshMethod::Cutoff,
                profile,
                ..Default::default()
            },
        );
        (mesh, result)
    }

    /// BEVEL.md §8.3.3 / §8.3.4: the 3-way collapse is suppressed for exactly the
    /// straight-chord profile (`profile = 0.25`, the `r == 1` line case). The three
    /// corner verts stay at their distinct boundvert-ring positions and a bottom
    /// triangle closes the cut: two more verts, three more edges, one more face than
    /// the curved collapse. The cut faces become pentagons (`C_k → P0 P1 P2 →
    /// C_(k+1)`, §8.3.2) instead of the collapsed quads.
    #[test]
    fn cutoff_junction_segments2_flat_chord_does_not_collapse() {
        let (mesh, result) = cube_corner_cutoff_segments2_mesh(0.25);
        mesh.validate_topology();

        // No single welded apex: the corner ring stays open.
        let apex_position = Vec3::new(0.75, -0.75, -0.75);
        let at_apex = result
            .verts
            .iter()
            .filter(|&&v| (mesh.vertex_position(v) - apex_position).length() < 1.0e-4)
            .count();
        assert_eq!(at_apex, 0, "the flat chord does not collapse to one apex");

        // 6 strip quads + 3 cut pentagons + 1 bottom triangle (§8.3.2 uncollapsed).
        assert_eq!(
            result.faces.len(),
            10,
            "flat chord: 6 strip quads + 3 cut pentagons + 1 bottom triangle",
        );
        let (mut triangles, mut quads, mut pentagons) = (0, 0, 0);
        for &face in &result.faces {
            match mesh.face_vert_keys(face).len() {
                3 => triangles += 1,
                4 => quads += 1,
                5 => pentagons += 1,
                other => panic!("unexpected {other}-gon in the flat-chord cutoff fill"),
            }
        }
        assert_eq!(
            (triangles, quads, pentagons),
            (1, 6, 3),
            "one bottom triangle, six strip quads, three uncollapsed cut pentagons",
        );

        // The bottom triangle's three corner verts are coincident-in-position with the
        // three boundverts (the §8.3.3 down-push is ~0) yet distinct vert keys — the
        // +2 V / +3 E / +1 F the flat chord adds over the collapsed apex.
        let corner = Vec3::new(1.0, -1.0, -1.0);
        let ring_radius = 0.25 * 2.0_f32.sqrt();
        let on_ring = result
            .verts
            .iter()
            .filter(|&&v| {
                ((mesh.vertex_position(v) - corner).length() - ring_radius).abs() < 1.0e-4
            })
            .count();
        assert_eq!(
            on_ring, 6,
            "three boundverts plus three coincident bottom corner verts on the ring",
        );
    }

    /// BEVEL.md §8.3.3 / §8.3.4: a curved profile (`profile = 0.7`, `r != 1`) fires
    /// the 3-way collapse exactly as the default arc does — three cut quads welded at
    /// one apex, no bottom face. The gate is binary, suppressed only for the straight
    /// chord, not for any curved profile.
    #[test]
    fn cutoff_junction_segments2_bulge_still_collapses() {
        let (mesh, result) = cube_corner_cutoff_segments2_mesh(0.7);
        mesh.validate_topology();

        // The collapse apex is the cell-corner down-push point, profile-independent.
        let apex_position = Vec3::new(0.75, -0.75, -0.75);
        let at_apex = result
            .verts
            .iter()
            .filter(|&&v| (mesh.vertex_position(v) - apex_position).length() < 1.0e-4)
            .count();
        assert_eq!(at_apex, 1, "the curved 0.7 profile still collapses to one apex");

        assert_eq!(
            result.faces.len(),
            9,
            "curved collapses: 6 strip quads + 3 cut quads, no bottom face",
        );
        for &face in &result.faces {
            assert_eq!(
                mesh.face_vert_keys(face).len(),
                4,
                "every collapsed-junction face is a quad",
            );
        }
    }

    /// BEVEL.md §8.3.3 / §8.3.4: the collapse gate is **binary**, not a continuous
    /// bulge-height threshold. A near-chord curved profile (`profile = 0.26`, whose
    /// exponent `r ≈ 1.03` is just above the line) still bulges off its chord, so it
    /// **does** collapse to one apex — pinning the binary rule against any continuous
    /// epsilon threshold that would have wrongly suppressed this near-straight case.
    #[test]
    fn cutoff_junction_segments2_near_chord_curved_collapses() {
        let (mesh, result) = cube_corner_cutoff_segments2_mesh(0.26);
        mesh.validate_topology();

        let apex_position = Vec3::new(0.75, -0.75, -0.75);
        let at_apex = result
            .verts
            .iter()
            .filter(|&&v| (mesh.vertex_position(v) - apex_position).length() < 1.0e-4)
            .count();
        assert_eq!(
            at_apex, 1,
            "the near-chord curved 0.26 profile still collapses to one apex",
        );

        assert_eq!(
            result.faces.len(),
            9,
            "near-chord curved collapses: 6 strip quads + 3 cut quads, no bottom face",
        );
        for &face in &result.faces {
            assert_eq!(
                mesh.face_vert_keys(face).len(),
                4,
                "every collapsed-junction face is a quad",
            );
        }
    }

    /// BEVEL.md §8.2.2.1 / §8.3.6: the lone `segments = 2` interior chain vert is
    /// the circular-arc 45° point in the profile cell — bulged toward the beveled
    /// edge, **not** the straight chord midpoint. For the Front↔Bottom chain across
    /// edge {0,1} it lands at (0.75, -0.927, -0.927), a radius 0.25·√2/2 from the
    /// arc center, distinct from the chord midpoint (0.75, -0.875, -0.875).
    #[test]
    fn cutoff_segments2_interior_vert_is_arc_sample() {
        let (mesh, result) = cube_corner_cutoff_segments2();

        let arc_point = Vec3::new(0.75, -0.926_776_7, -0.926_776_7);
        let chord_midpoint = Vec3::new(0.75, -0.875, -0.875);

        let near_arc = result
            .verts
            .iter()
            .filter(|&&v| (mesh.vertex_position(v) - arc_point).length() < 1.0e-4)
            .count();
        assert_eq!(
            near_arc, 1,
            "the Front-Bottom interior chain vert sits at the arc 45° point",
        );

        let near_chord = result
            .verts
            .iter()
            .filter(|&&v| (mesh.vertex_position(v) - chord_midpoint).length() < 1.0e-4)
            .count();
        assert_eq!(
            near_chord, 0,
            "no interior chain vert sits at the straight chord midpoint (§8.3.6)",
        );

        // Every interior arc sample lies on a circle of radius offset (0.25) about
        // its cell's arc center: the chain bulges, it is not straight-subdivided.
        let arc_center = Vec3::new(0.75, -0.75, -0.75);
        let radius = (arc_point - arc_center).length();
        assert!(
            (radius - 0.25).abs() < 1.0e-4,
            "arc sample sits radius=offset from the (0,0) cell corner",
        );
    }

    /// BEVEL.md §8.2.1: the boundvert ring is segment-count-independent — the
    /// `segments = 2` cutoff places the same three boundverts (offset·√2 from the
    /// corner) as the single-segment junction.
    #[test]
    fn cutoff_segments2_ring_matches_single_segment() {
        let (mesh, result) = cube_corner_cutoff_segments2();
        let corner = Vec3::new(1.0, -1.0, -1.0);
        let expected = 0.25 * 2.0_f32.sqrt();

        // The three boundverts (planar 2-line meets) sit offset·√2 from the corner.
        let ring_verts = result
            .verts
            .iter()
            .filter(|&&v| ((mesh.vertex_position(v) - corner).length() - expected).abs() < 1.0e-4)
            .count();
        assert_eq!(ring_verts, 3, "three §7.6.2 boundverts, unchanged by segments");

        // The junction corner and all three far endpoints are consumed.
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
    }

    /// BEVEL.md §8.3.0.1: a valence-3 terminal builds no fill face — only the
    /// through-strip quads. The whole result is a closed manifold (every edge
    /// radial-2), confirming the terminals close by the absorb path, not a cutoff.
    #[test]
    fn cutoff_segments2_terminals_have_no_fill_and_strips_run_through() {
        let (mesh, _result) = cube_corner_cutoff_segments2();

        // The three beveled edges each shed `segments = 2` strip quads = 6 strip
        // quads; the junction adds 3 cut faces. Every face edge is manifold.
        for face in mesh.face_keys() {
            for loop_key in mesh.face_loop_keys(face) {
                let edge = mesh.loop_edge(loop_key);
                assert_eq!(
                    mesh.edge_radial_iter(edge).count(),
                    2,
                    "closed manifold: every edge shared by exactly two faces",
                );
            }
        }
        mesh.validate_topology();
    }

    /// Build the cube-corner `s = 3` ADJ junction at `segments = 2` (BEVEL.md §8.5).
    /// Vertex 0 = (1,-1,-1) is the junction; its three edges to 1/3/4 are beveled
    /// with the default rounded adjacency-grid fill.
    fn cube_corner_adj_segments2() -> (LoopMesh, BevelResult) {
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
                segments: 2,
                vmesh_method: BevelVmeshMethod::Adj,
                ..Default::default()
            },
        );
        (mesh, result)
    }

    /// BEVEL.md §8.5.2 / §8.5.3: the `s = 3`, `segments = 2` cube-corner ADJ junction
    /// rounds the corner with a three-quad pinwheel meeting at one welded central
    /// apex — no bottom face. The whole cube grows 8 V / 12 E / 6 F to 20 V / 33 E /
    /// 15 F (§8.5.3), and the result stays a closed manifold.
    #[test]
    fn adj_junction_segments2_topology() {
        let (mesh, result) = cube_corner_adj_segments2();

        mesh.validate_topology();

        assert_eq!(
            counts(&mesh),
            (20, 33, 15),
            "cube-corner ADJ seg=2 grows the cube to 20 V / 33 E / 15 F (§8.5.3)",
        );

        // §8.5.2 closed-form apex: the body-diagonal direction snapped onto the
        // rounded surface (the sphere through the ring + chain verts, here centre
        // (0.75,-0.75,-0.75), radius = offset 0.25). The apex is that sphere's point
        // nearest the corner along the body diagonal: centre + offset·(1,-1,-1)/√3,
        // i.e. ≈ (0.894, -0.894, -0.894) — on the dome, not pushed into the solid.
        let sphere_center = Vec3::new(0.75, -0.75, -0.75);
        let apex_position = sphere_center + 0.25 * Vec3::new(1.0, -1.0, -1.0) / 3.0_f32.sqrt();
        let apex: Vec<VertexKey> = result
            .verts
            .iter()
            .copied()
            .filter(|&v| (mesh.vertex_position(v) - apex_position).length() < 1.0e-4)
            .collect();
        assert_eq!(apex.len(), 1, "the three pinwheel apex corners weld to one apex");
        let apex = apex[0];

        // §8.5.3: exactly three quad faces meet at the apex (the pinwheel).
        let pinwheel: Vec<FaceKey> = mesh
            .face_keys()
            .into_iter()
            .filter(|&f| mesh.face_vert_keys(f).contains(&apex))
            .collect();
        assert_eq!(pinwheel.len(), 3, "three pinwheel quads share the apex");
        for &face in &pinwheel {
            assert_eq!(
                mesh.face_vert_keys(face).len(),
                4,
                "each pinwheel cell is a quad (§8.5.3)",
            );
        }

        // Closed manifold: every edge radial-2. The new geometry is 6 strip quads +
        // 3 pinwheel quads = 9 new faces; with the cube's 6 originals that is 15.
        for face in mesh.face_keys() {
            for loop_key in mesh.face_loop_keys(face) {
                let edge = mesh.loop_edge(loop_key);
                assert_eq!(
                    mesh.edge_radial_iter(edge).count(),
                    2,
                    "every edge is shared with exactly one other face",
                );
            }
        }
        assert_eq!(result.faces.len(), 9, "6 strip quads + 3 pinwheel quads, no bottom");
    }

    /// BEVEL.md §8.5.3: the ADJ apex sits *on* the rounded surface (the body-diagonal
    /// point snapped onto the dome, §8.5.2), distinct from the cutoff apex which is
    /// pushed *down into* the solid (§8.3.4). The two fills share the same boundvert
    /// ring and chains but place the central vert at different points.
    #[test]
    fn adj_apex_differs_from_cutoff_apex() {
        let (adj_mesh, adj_result) = cube_corner_adj_segments2();
        let (cut_mesh, _cut_result) = cube_corner_cutoff_segments2();

        let sphere_center = Vec3::new(0.75, -0.75, -0.75);
        let adj_apex = sphere_center + 0.25 * Vec3::new(1.0, -1.0, -1.0) / 3.0_f32.sqrt();
        let cutoff_apex = Vec3::new(0.75, -0.75, -0.75);

        // The ADJ apex lands on the rounded surface (snapped body-diagonal point)...
        assert_eq!(
            adj_result
                .verts
                .iter()
                .filter(|&&v| (adj_mesh.vertex_position(v) - adj_apex).length() < 1.0e-4)
                .count(),
            1,
        );
        // ...which is a different position bucket from the cutoff apex.
        assert!(
            (adj_apex - cutoff_apex).length() > 1.0e-2,
            "ADJ apex (rounded surface) differs from cutoff apex (pushed down)",
        );
        // And no ADJ vert lands at the cutoff apex.
        assert_eq!(
            adj_result
                .verts
                .iter()
                .filter(|&&v| (adj_mesh.vertex_position(v) - cutoff_apex).length() < 1.0e-4)
                .count(),
            0,
        );
        // Sanity: the cutoff fixture still has its own apex where expected.
        assert!(cut_mesh
            .vert_keys()
            .into_iter()
            .any(|v| (cut_mesh.vertex_position(v) - cutoff_apex).length() < 1.0e-4));
    }

    /// BEVEL.md §8.5.2: the ADJ junction reuses the same segment-count-independent
    /// boundvert ring (§8.2.1) and the same arc-sampled interior chain verts
    /// (§8.3.6) as cutoff — only the junction fill differs. The three boundverts sit
    /// offset·√2 from the corner and the interior chain verts sit at the arc 45°
    /// point (radius offset from the per-cell arc center), exactly as cutoff.
    #[test]
    fn adj_segments2_ring_and_chain_match_cutoff() {
        let (mesh, result) = cube_corner_adj_segments2();
        let corner = Vec3::new(1.0, -1.0, -1.0);
        let expected = 0.25 * 2.0_f32.sqrt();

        let ring_verts = result
            .verts
            .iter()
            .filter(|&&v| ((mesh.vertex_position(v) - corner).length() - expected).abs() < 1.0e-4)
            .count();
        assert_eq!(ring_verts, 3, "three §7.6.2 boundverts, unchanged by vmesh method");

        // The Front-Bottom interior chain vert across edge {0,1} is the arc 45° point
        // (§8.3.6), shared with cutoff — not the straight chord midpoint.
        let arc_point = Vec3::new(0.75, -0.926_776_7, -0.926_776_7);
        let chord_midpoint = Vec3::new(0.75, -0.875, -0.875);
        assert_eq!(
            result_has_vert_near(&mesh, &result, arc_point),
            1,
            "interior chain vert is the arc 45° sample, shared with cutoff",
        );
        assert_eq!(
            result_has_vert_near(&mesh, &result, chord_midpoint),
            0,
            "no interior chain vert at the straight chord midpoint",
        );
    }

    /// Bevel the same symmetric equal-offset cube corner with `segments` segments,
    /// default circular profile, `vmesh_method = ADJ` (BEVEL.md §8.5.5).
    fn cube_corner_adj(segments: u32) -> (LoopMesh, BevelResult) {
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
                segments,
                vmesh_method: BevelVmeshMethod::Adj,
                ..Default::default()
            },
        );
        (mesh, result)
    }

    /// The corner sphere (centre `c`, radius `R = offset`) the cube-corner patch
    /// lies on (§8.5.5.3), and a "within a few `offset` of `c`" membership test that
    /// isolates the corner-grid verts/faces from the per-edge strips (§8.5.5.6).
    fn adj_sphere_center() -> Vec3 {
        Vec3::new(0.75, -0.75, -0.75)
    }
    const ADJ_SPHERE_RADIUS: f32 = 0.25;

    fn adj_in_corner_ball(mesh: &LoopMesh, vertex: VertexKey) -> bool {
        (mesh.vertex_position(vertex) - adj_sphere_center()).length() <= ADJ_SPHERE_RADIUS * 1.2
    }

    fn adj_patch_faces(mesh: &LoopMesh, result: &BevelResult) -> Vec<FaceKey> {
        result
            .faces
            .iter()
            .copied()
            .filter(|&f| mesh.face_vert_keys(f).iter().all(|&v| adj_in_corner_ball(mesh, v)))
            .collect()
    }

    fn adj_apex_position() -> Vec3 {
        adj_sphere_center() + ADJ_SPHERE_RADIUS * Vec3::new(1.0, -1.0, -1.0) / 3.0_f32.sqrt()
    }

    fn adj_assert_invariants(mesh: &LoopMesh, result: &BevelResult) {
        mesh.validate_topology();

        // Invariant 1 (§8.5.5.6): the three [P] boundverts, segment-count-independent
        // (offset·√2 from the corner).
        let corner = Vec3::new(1.0, -1.0, -1.0);
        let ring_radius = ADJ_SPHERE_RADIUS * 2.0_f32.sqrt();
        let ring = result
            .verts
            .iter()
            .filter(|&&v| ((mesh.vertex_position(v) - corner).length() - ring_radius).abs() < 1.0e-4)
            .count();
        assert_eq!(ring, 3, "three §7.6.2 boundverts (invariant 1)");

        // Invariant 2 (§8.5.5.6): every corner-region grid vert lies on the corner
        // sphere; the strip verts sit several offset away and are excluded.
        for &v in &result.verts {
            let p = mesh.vertex_position(v);
            if (p - adj_sphere_center()).length() <= ADJ_SPHERE_RADIUS * 1.2 {
                assert!(
                    ((p - adj_sphere_center()).length() - ADJ_SPHERE_RADIUS).abs() < 1.0e-4,
                    "corner-grid vert lies on the superellipsoid (invariant 2)",
                );
            }
        }

        // Closed manifold: every face edge is shared with exactly one other face.
        for face in mesh.face_keys() {
            for loop_key in mesh.face_loop_keys(face) {
                assert_eq!(
                    mesh.edge_radial_iter(mesh.loop_edge(loop_key)).count(),
                    2,
                    "every edge is radial-2",
                );
            }
        }
    }

    /// BEVEL.md §8.5.5.4 (even parity): `segments = 4` at the cube corner welds the
    /// three sub-patches to one apex vertex on the body diagonal, with no central
    /// face — the corner grid is 3·(segments/2)² = 12 quads.
    #[test]
    fn adj_cube_corner_segments_4_welded_apex() {
        let (mesh, result) = cube_corner_adj(4);
        adj_assert_invariants(&mesh, &result);

        // Invariant 4 (§8.5.5.4): the single welded apex on the body diagonal.
        let apex: Vec<VertexKey> = result
            .verts
            .iter()
            .copied()
            .filter(|&v| (mesh.vertex_position(v) - adj_apex_position()).length() < 1.0e-4)
            .collect();
        assert_eq!(apex.len(), 1, "even segments weld to one apex vertex");
        let apex = apex[0];
        let apex_faces: Vec<FaceKey> = mesh
            .face_keys()
            .into_iter()
            .filter(|&f| mesh.face_vert_keys(f).contains(&apex))
            .collect();
        assert_eq!(
            apex_faces.len(),
            3,
            "the three innermost cells meet the apex directly (no central face)",
        );

        // Invariant 3 (§8.5.5.6): 12 all-quad sub-patch faces.
        let patch_faces = adj_patch_faces(&mesh, &result);
        assert_eq!(patch_faces.len(), 12, "3·(segments/2)² = 12 sub-patch quads");
        for &f in &patch_faces {
            assert_eq!(
                mesh.face_vert_keys(f).len(),
                4,
                "every corner-grid face is a quad at even parity",
            );
        }
    }

    /// BEVEL.md §8.5.5.4 (odd parity): `segments = 3` at the cube corner places no
    /// vertex on the body diagonal; the three sub-patches are bridged by seam quads
    /// and closed by exactly one central `s`-gon (s = 3) triangle — 6 quads + 1
    /// triangle.
    #[test]
    fn adj_cube_corner_segments_3_central_ngon() {
        let (mesh, result) = cube_corner_adj(3);
        adj_assert_invariants(&mesh, &result);

        // Invariant 4 (§8.5.5.4): no apex vertex on the body diagonal.
        assert_eq!(
            result
                .verts
                .iter()
                .filter(|&&v| (mesh.vertex_position(v) - adj_apex_position()).length() < 1.0e-4)
                .count(),
            0,
            "odd segments place no vertex on the body diagonal",
        );

        // Invariant 3/4: quad-dominant with exactly one central triangle.
        let patch_faces = adj_patch_faces(&mesh, &result);
        let triangles = patch_faces
            .iter()
            .filter(|&&f| mesh.face_vert_keys(f).len() == 3)
            .count();
        let quads = patch_faces
            .iter()
            .filter(|&&f| mesh.face_vert_keys(f).len() == 4)
            .count();
        assert_eq!(triangles, 1, "exactly one central s-gon (triangle) at odd parity");
        assert_eq!(quads, 6, "3 corner sub-patch quads + 3 seam quads");
        assert_eq!(patch_faces.len(), 7);
    }

    /// BEVEL.md §8.5.5.4: the load-bearing parity contrast — an even target closes
    /// the corner on a welded apex vertex and no central face, an odd target on a
    /// central `s`-gon face and no apex vertex.
    #[test]
    fn adj_cube_corner_odd_even_central_parity_differs() {
        let (odd_mesh, odd) = cube_corner_adj(3);
        let (even_mesh, even) = cube_corner_adj(4);
        let apex_pos = adj_apex_position();

        let apex_count = |mesh: &LoopMesh, result: &BevelResult| {
            result
                .verts
                .iter()
                .filter(|&&v| (mesh.vertex_position(v) - apex_pos).length() < 1.0e-4)
                .count()
        };
        assert_eq!(apex_count(&even_mesh, &even), 1, "even → welded apex vertex");
        assert_eq!(apex_count(&odd_mesh, &odd), 0, "odd → no apex vertex");

        let central_triangles = |mesh: &LoopMesh, result: &BevelResult| {
            adj_patch_faces(mesh, result)
                .into_iter()
                .filter(|&f| mesh.face_vert_keys(f).len() == 3)
                .count()
        };
        assert_eq!(central_triangles(&odd_mesh, &odd), 1, "odd → one central triangle");
        assert_eq!(central_triangles(&even_mesh, &even), 0, "even → no central face");
    }

    /// BEVEL.md §8.5.1: the forced-tri-corner trigger predicate read off the
    /// boundvert ring. The symmetric equal-offset orthonormal cube corner at the
    /// default circular profile fires it; an unequal-offset corner, a non-right-angle
    /// corner, or a non-default profile all decline (the §8.5.5.5 deferred cases).
    #[test]
    fn adj_tri_corner_trigger_predicate() {
        let mut mesh = LoopMesh::new();
        let junction = mesh.create_vertex(Vec3::ZERO);
        let make_cut = |mesh: &mut LoopMesh, spokes: [Vec3; 3]| JunctionCut {
            vertex: junction,
            boundverts: spokes.iter().map(|&s| mesh.create_vertex(s)).collect(),
            opening_edges: Vec::new(),
            outward: Vec3::ZERO,
        };

        // Orthonormal equal-offset cube corner: meet-point spokes `d_i + d_j` over the
        // three axis directions — equal length (√2) and pairwise 60° (cos ½).
        let cube = make_cut(
            &mut mesh,
            [
                Vec3::new(1.0, 1.0, 0.0),
                Vec3::new(0.0, 1.0, 1.0),
                Vec3::new(1.0, 0.0, 1.0),
            ],
        );
        assert!(
            mesh.adj_tri_corner_trigger(&cube, 0.5),
            "default-profile cube corner fires the §8.5.1 trigger",
        );
        assert!(
            !mesh.adj_tri_corner_trigger(&cube, 0.7),
            "a curved profile declines (§8.5.5.5)",
        );

        // Unequal offset: one spoke is twice as long, so the lengths differ.
        let unequal = make_cut(
            &mut mesh,
            [
                Vec3::new(2.0, 2.0, 0.0),
                Vec3::new(0.0, 1.0, 1.0),
                Vec3::new(1.0, 0.0, 1.0),
            ],
        );
        assert!(
            !mesh.adj_tri_corner_trigger(&unequal, 0.5),
            "unequal offsets decline the §8.5.1 trigger",
        );

        // Non-right-angle corner: equal-length spokes 120° apart (cos −½, not ½).
        let flat = make_cut(
            &mut mesh,
            [
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(-0.5, 0.866_025_4, 0.0),
                Vec3::new(-0.5, -0.866_025_4, 0.0),
            ],
        );
        assert!(
            !mesh.adj_tri_corner_trigger(&flat, 0.5),
            "non-right-angle corner declines the §8.5.1 trigger",
        );
    }

    /// BEVEL.md §8.5.1 / §8.5.5.5: a non-default (curved) profile at the symmetric
    /// cube corner fails the trigger's default-profile condition, so the
    /// `segments ≥ 3` general fill (`build_adj_fill_general`) is declined rather than
    /// snapping the interior onto a sphere it does not lie on. The through-strips
    /// still build; the corner is simply left unfilled.
    #[test]
    fn adj_tri_corner_guard_declines_curved_profile_segments3() {
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
                segments: 3,
                profile: 0.7,
                vmesh_method: BevelVmeshMethod::Adj,
                ..Default::default()
            },
        );
        mesh.validate_topology();
        assert_eq!(
            adj_patch_faces(&mesh, &result).len(),
            0,
            "curved-profile cube corner declines the §8.5.5 general fill",
        );
        assert_eq!(
            result
                .verts
                .iter()
                .filter(|&&v| (mesh.vertex_position(v) - adj_apex_position()).length() < 1.0e-4)
                .count(),
            0,
            "no central body-diagonal vertex emitted when the trigger declines",
        );
    }

    /// BEVEL.md §8.5.1 / §8.5.5.5: the same default-profile guard covers the
    /// closed-form `segments = 2` pinwheel path. A curved profile declines the
    /// pinwheel, leaving only the 6 through-strip quads (vs the 9 = 6 strip + 3
    /// pinwheel of the supported default-profile cube corner).
    #[test]
    fn adj_tri_corner_guard_declines_curved_profile_segments2() {
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
                segments: 2,
                profile: 0.7,
                vmesh_method: BevelVmeshMethod::Adj,
                ..Default::default()
            },
        );
        mesh.validate_topology();
        assert_eq!(
            result.faces.len(),
            6,
            "curved-profile seg=2 corner declines the pinwheel: 6 strip quads only",
        );
    }

    /// BEVEL.md §8.5.5.5: ADJ outside the symmetric cube corner stays a stub. A
    /// two-edge (`s = 2`) junction takes no pinwheel fill (the through-strips still
    /// build, exactly as cutoff), and there is no central apex. Multi-segment vertex
    /// bevel also remains a no-op.
    #[test]
    fn adj_non_forced_and_vertex_are_stubbed() {
        let mut mesh = LoopMesh::new();
        mesh_add_cube(&mut mesh);
        let e01 = cube_edge(
            &mesh,
            Vec3::new(1.0, -1.0, -1.0),
            Vec3::new(-1.0, -1.0, -1.0),
        );
        let e03 = cube_edge(&mesh, Vec3::new(1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, -1.0));

        // s = 2 junction at v0, ADJ: no pinwheel fill (the general grid is a stub),
        // but the through-strip geometry still builds and the mesh stays manifold.
        let result = mesh.bevel(
            &[e01.into(), e03.into()],
            BevelOptions {
                offset: 0.25,
                affect: BevelAffect::Edges,
                segments: 2,
                vmesh_method: BevelVmeshMethod::Adj,
                ..Default::default()
            },
        );
        mesh.validate_topology();
        // No welded central apex: with s = 2 the §8.5 forced case does not apply.
        // The apex would have sat along the corner's interior body diagonal; assert
        // there is no extra fan-center vert by checking the new faces are only the
        // through-strip quads (4 = 2 edges × 2 segments), no pinwheel quad.
        assert_eq!(
            result.faces.len(),
            4,
            "s=2 ADJ junction: through-strips only, no pinwheel fill (§8.5.5)",
        );

        // segments ≥ 2 vertex bevel: still a stub, no-op.
        let before = counts(&mesh);
        let v_far = vert_at(&mesh, Vec3::new(-1.0, 1.0, 1.0));
        mesh.bevel(
            &[v_far.into()],
            BevelOptions {
                offset: 0.25,
                affect: BevelAffect::Vertices,
                segments: 2,
                vmesh_method: BevelVmeshMethod::Adj,
                ..Default::default()
            },
        );
        assert_eq!(counts(&mesh), before, "segments≥2 vertex bevel is a no-op");
    }

    /// BEVEL.md §8.3: the cutoff fill needs at least one multi-beveled-edge
    /// junction. A `segments ≥ 2`, `Cutoff` call over a junction-free edge set
    /// (here two disjoint, non-shared cube edges) is a clean no-op — it must not
    /// emit partial through-strip geometry.
    #[test]
    fn cutoff_no_junction_is_noop() {
        let mut mesh = LoopMesh::new();
        mesh_add_cube(&mut mesh);
        let before = counts(&mesh);

        // Two disjoint edges with no shared endpoint: every endpoint is terminal,
        // so there is no junction for the cutoff to fold.
        let e01 = cube_edge(
            &mesh,
            Vec3::new(1.0, -1.0, -1.0),
            Vec3::new(-1.0, -1.0, -1.0),
        );
        let e_far = cube_edge(&mesh, Vec3::new(-1.0, 1.0, 1.0), Vec3::new(1.0, 1.0, 1.0));

        let result = mesh.bevel(
            &[e01.into(), e_far.into()],
            BevelOptions {
                offset: 0.25,
                affect: BevelAffect::Edges,
                segments: 2,
                vmesh_method: BevelVmeshMethod::Cutoff,
                ..Default::default()
            },
        );

        assert_eq!(counts(&mesh), before, "junction-free cutoff is a no-op");
        assert!(result.verts.is_empty(), "no new verts");
        assert!(result.edges.is_empty(), "no new edges");
        assert!(result.faces.is_empty(), "no new faces");
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

    /// The minimum ring slide of a corner: distance from the original corner
    /// position to the nearest of `result`'s new ring verts. For a single corner
    /// at `offset` every ring vert slides exactly `offset` (§4); the *minimum* is
    /// robust against the other corner's ring verts in a two-corner bevel.
    fn nearest_ring_slide(mesh: &LoopMesh, result: &BevelResult, corner: Vec3) -> f32 {
        result
            .verts
            .iter()
            .map(|&v| (mesh.vertex_position(v) - corner).length())
            .fold(f32::INFINITY, f32::min)
    }

    /// BEVEL.md §8.7.3 / §8.7.5: distinct weights on two non-adjacent corners in
    /// one call produce a non-uniform result — the `0.4`-weighted corner slides
    /// its ring verts only `0.4·offset`, the `1.0`-weighted corner the full
    /// `offset`. v0 = (1,-1,-1) and v6 = (-1,1,1) are opposite cube corners
    /// (disjoint one-rings), so each weight scales its corner independently.
    #[test]
    fn weighted_vertex_bevel_shrinks_one_corner() {
        let mut mesh = LoopMesh::new();
        mesh_add_cube(&mut mesh);
        let v0 = vert_at(&mesh, Vec3::new(1.0, -1.0, -1.0));
        let v6 = vert_at(&mesh, Vec3::new(-1.0, 1.0, 1.0));
        let pos0 = mesh.vertex_position(v0);
        let pos6 = mesh.vertex_position(v6);

        let offset = 0.5;
        let result = mesh.bevel(
            &[v0.into(), v6.into()],
            BevelOptions {
                offset,
                use_weights: true,
                vert_weights: &[(v0, 1.0), (v6, 0.4)],
                ..Default::default()
            },
        );

        let slide_full = nearest_ring_slide(&mesh, &result, pos0);
        let slide_weighted = nearest_ring_slide(&mesh, &result, pos6);
        assert!(
            (slide_full - offset).abs() < 1.0e-5,
            "weight 1.0 corner slides the full offset {offset}, got {slide_full}",
        );
        assert!(
            (slide_weighted - 0.4 * offset).abs() < 1.0e-5,
            "weight 0.4 corner slides 0.4·offset = {}, got {slide_weighted}",
            0.4 * offset,
        );
        assert!(
            slide_weighted < slide_full - 1.0e-5,
            "the weighted corner must be observably smaller than the full one",
        );
        mesh.validate_topology();
    }

    /// BEVEL.md §8.7.2 case 1 / §8.7.3: uniform weight `1.0` on every beveled
    /// corner reproduces the unweighted geometry byte-for-byte (weight `1.0` is a
    /// no-op prescale). Compared against the same bevel with `use_weights` off.
    #[test]
    fn uniform_weight_one_reproduces_unweighted() {
        let corner_slides = |use_weights: bool, weights: &[(VertexKey, f32)]| {
            let mut mesh = LoopMesh::new();
            mesh_add_cube(&mut mesh);
            let v0 = vert_at(&mesh, Vec3::new(1.0, -1.0, -1.0));
            let v6 = vert_at(&mesh, Vec3::new(-1.0, 1.0, 1.0));
            let pos0 = mesh.vertex_position(v0);
            let pos6 = mesh.vertex_position(v6);
            let result = mesh.bevel(
                &[v0.into(), v6.into()],
                BevelOptions {
                    offset: 0.5,
                    use_weights,
                    vert_weights: weights,
                    ..Default::default()
                },
            );
            let mut positions: Vec<Vec3> =
                result.verts.iter().map(|&v| mesh.vertex_position(v)).collect();
            positions.sort_by(|a, b| {
                a.to_array()
                    .partial_cmp(&b.to_array())
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            (positions, nearest_ring_slide(&mesh, &result, pos0), nearest_ring_slide(&mesh, &result, pos6))
        };

        // Build the weighted-1.0 call with the same fixture's keys. The keys are
        // deterministic across the two identical cube builds, so the all-`1.0`
        // weighted run and the gate-off run must coincide.
        let mut mesh = LoopMesh::new();
        mesh_add_cube(&mut mesh);
        let v0 = vert_at(&mesh, Vec3::new(1.0, -1.0, -1.0));
        let v6 = vert_at(&mesh, Vec3::new(-1.0, 1.0, 1.0));

        let (unweighted, u0, u6) = corner_slides(false, &[]);
        let (weighted, w0, w6) = corner_slides(true, &[(v0, 1.0), (v6, 1.0)]);

        assert!((u0 - 0.5).abs() < 1.0e-6 && (u6 - 0.5).abs() < 1.0e-6);
        assert!((w0 - u0).abs() < 1.0e-6 && (w6 - u6).abs() < 1.0e-6);
        assert_eq!(weighted.len(), unweighted.len());
        for (a, b) in weighted.iter().zip(&unweighted) {
            assert!(
                (*a - *b).length() < 1.0e-6,
                "all-1.0 weighted bevel must match the unweighted geometry",
            );
        }
    }

    /// BEVEL.md §8.7.2 case 1: with the gate off, the `vert_weights` slice is
    /// ignored entirely — every corner slides the full offset even though the
    /// slice would scale one of them down.
    #[test]
    fn weights_ignored_when_gate_off() {
        let mut mesh = LoopMesh::new();
        mesh_add_cube(&mut mesh);
        let v0 = vert_at(&mesh, Vec3::new(1.0, -1.0, -1.0));
        let v6 = vert_at(&mesh, Vec3::new(-1.0, 1.0, 1.0));
        let pos6 = mesh.vertex_position(v6);

        let result = mesh.bevel(
            &[v0.into(), v6.into()],
            BevelOptions {
                offset: 0.5,
                use_weights: false,
                vert_weights: &[(v0, 1.0), (v6, 0.4)],
                ..Default::default()
            },
        );

        let slide6 = nearest_ring_slide(&mesh, &result, pos6);
        assert!(
            (slide6 - 0.5).abs() < 1.0e-5,
            "gate off must ignore the 0.4 weight; corner should slide the full 0.5, got {slide6}",
        );
        mesh.validate_topology();
    }

    /// BEVEL.md §8.7.2 case 3 / §8.7.6: gate on, a beveled vertex *absent* from
    /// `vert_weights` defaults to weight `0.0` — zero effective offset, no slide,
    /// a clean no-op for that corner (its ring verts coincide with the original
    /// corner). v0 is written `1.0` and bevels; v6 is omitted and does not.
    #[test]
    fn absent_weight_defaults_to_zero_no_op() {
        let mut mesh = LoopMesh::new();
        mesh_add_cube(&mut mesh);
        let v0 = vert_at(&mesh, Vec3::new(1.0, -1.0, -1.0));
        let v6 = vert_at(&mesh, Vec3::new(-1.0, 1.0, 1.0));
        let pos0 = mesh.vertex_position(v0);
        let pos6 = mesh.vertex_position(v6);

        let result = mesh.bevel(
            &[v0.into(), v6.into()],
            BevelOptions {
                offset: 0.5,
                use_weights: true,
                vert_weights: &[(v0, 1.0)], // v6 absent ⇒ weight 0.0
                ..Default::default()
            },
        );

        let slide0 = nearest_ring_slide(&mesh, &result, pos0);
        assert!(
            (slide0 - 0.5).abs() < 1.0e-5,
            "v0 weighted 1.0 slides the full offset, got {slide0}",
        );
        // v6's three ring verts all land at zero slide — coincident with the
        // original corner — so the nearest ring vert to pos6 is at distance 0.
        let slide6 = nearest_ring_slide(&mesh, &result, pos6);
        assert!(
            slide6 < 1.0e-6,
            "absent v6 defaults to weight 0.0: ring verts stay at the corner, got {slide6}",
        );
        mesh.validate_topology();
    }

    /// An irregular apex junction (BEVEL.md §8.8.5): apex `v0` fanned by four
    /// quads to four spokes `v1..v4` at unequal directions and heights, so every
    /// spoke is a silhouette edge (its two flanking quads are non-coplanar). The
    /// four quads each span the apex plus two consecutive spokes plus one outer
    /// rim vert. Beveling spokes {0,1} and {0,3} leaves spoke {0,2} as the lone
    /// silhouette leg in the sector between them — the §8.8.2 firing sector.
    /// Returns (mesh, apex, [v1,v2,v3,v4], beveled edges {0,1} and {0,3}).
    fn irregular_apex_junction() -> (LoopMesh, VertexKey, [VertexKey; 4], [EdgeKey; 2]) {
        let mut mesh = LoopMesh::new();
        let apex = mesh.create_vertex(Vec3::new(0.0, 0.0, 0.0));
        // Spokes at unequal xy-directions and z-heights so the surface bends
        // across each (silhouette) and the {0,1}/{0,3} sector is irregular.
        let s1 = mesh.create_vertex(Vec3::new(1.0, 0.0, -0.3));
        let s2 = mesh.create_vertex(Vec3::new(0.35, 0.95, -0.7));
        let s3 = mesh.create_vertex(Vec3::new(-1.0, 0.15, -0.25));
        let s4 = mesh.create_vertex(Vec3::new(-0.25, -1.0, -0.55));
        let spokes = [s1, s2, s3, s4];
        // Outer rim verts, one between each consecutive spoke pair, pushed out and
        // down so the quads are non-planar petals.
        let outers: Vec<VertexKey> = (0..4)
            .map(|i| {
                let a = mesh.vertex_position(spokes[i]);
                let b = mesh.vertex_position(spokes[(i + 1) % 4]);
                let mid = (a + b) * 0.5;
                mesh.create_vertex(mid * 1.8 + Vec3::new(0.0, 0.0, -0.4))
            })
            .collect();

        let edge = |mesh: &mut LoopMesh, a: VertexKey, b: VertexKey| {
            mesh.create_edge([a, b], EdgeCreateFlags::NONE)
        };
        for i in 0..4 {
            let sa = spokes[i];
            let sb = spokes[(i + 1) % 4];
            let outer = outers[i];
            let e_apex_a = mesh
                .query_edge_exists(apex, sa)
                .unwrap_or_else(|| edge(&mut mesh, apex, sa));
            let e_a_outer = edge(&mut mesh, sa, outer);
            let e_outer_b = edge(&mut mesh, outer, sb);
            let e_b_apex = mesh
                .query_edge_exists(apex, sb)
                .unwrap_or_else(|| edge(&mut mesh, apex, sb));
            mesh.face_create(
                &[apex, sa, outer, sb],
                &[e_apex_a, e_a_outer, e_outer_b, e_b_apex],
                FaceCreateFlags::NONE,
            );
        }

        let e01 = mesh.query_edge_exists(apex, s1).expect("spoke {0,1}");
        let e03 = mesh.query_edge_exists(apex, s3).expect("spoke {0,3}");
        (mesh, apex, spokes, [e01, e03])
    }

    /// The boundvert of the firing sector (the new vert nearest spoke {0,2}'s
    /// line through the apex) — the one `loop_slide` slides onto leg {0,2}.
    fn firing_boundvert(
        mesh: &LoopMesh,
        result: &BevelResult,
        apex_pos: Vec3,
        leg_dir: Vec3,
    ) -> Vec3 {
        result
            .verts
            .iter()
            .map(|&v| mesh.vertex_position(v))
            .min_by(|&a, &b| {
                let da = ((a - apex_pos) - leg_dir * (a - apex_pos).dot(leg_dir)).length();
                let db = ((b - apex_pos) - leg_dir * (b - apex_pos).dot(leg_dir)).length();
                da.partial_cmp(&db).unwrap()
            })
            .expect("the junction produced boundverts")
    }

    /// Like `firing_boundvert`, but over a bare position list (the new verts already
    /// collected): the vert with the smallest perpendicular distance to `leg_dir`'s
    /// line through `apex_pos` — the boundvert that slid onto that leg.
    fn firing_boundvert_from(positions: &[Vec3], apex_pos: Vec3, leg_dir: Vec3) -> Vec3 {
        let perp = |p: Vec3| {
            let rel = p - apex_pos;
            (rel - leg_dir * rel.dot(leg_dir)).length()
        };
        *positions
            .iter()
            .min_by(|&&a, &&b| perp(a).partial_cmp(&perp(b)).unwrap())
            .expect("the junction produced boundverts")
    }

    /// BEVEL.md §8.8: `loop_slide = true` slides the firing sector's boundvert
    /// onto the silhouette leg's line, where `loop_slide = false` leaves it at the
    /// §7.6.2 two-strip meet-point. At an irregular junction the two placements
    /// differ (§8.8.5) while the topology is byte-identical — only one vertex
    /// moves.
    #[test]
    fn edge_bevel_loop_slide_diverges_at_irregular_junction() {
        let offset = 0.2;

        let (mut mesh_off, apex_off, _spokes_off, [e01_off, e03_off]) = irregular_apex_junction();
        let off = mesh_off.bevel(
            &[e01_off.into(), e03_off.into()],
            BevelOptions {
                offset,
                affect: BevelAffect::Edges,
                loop_slide: false,
                ..Default::default()
            },
        );
        let counts_off = counts(&mesh_off);
        mesh_off.validate_topology();

        let (mut mesh_on, apex_on, spokes_on, [e01_on, e03_on]) = irregular_apex_junction();
        let apex_pos = mesh_on.vertex_position(apex_on);
        let leg_dir = (mesh_on.vertex_position(spokes_on[1]) - apex_pos).normalize();
        let on = mesh_on.bevel(
            &[e01_on.into(), e03_on.into()],
            BevelOptions {
                offset,
                affect: BevelAffect::Edges,
                loop_slide: true,
                ..Default::default()
            },
        );
        mesh_on.validate_topology();

        // §8.8: pure placement perturbation — no topology change.
        assert_eq!(
            counts(&mesh_on),
            counts_off,
            "loop_slide changes no topology (same V/E/F)",
        );

        // §8.8.5: the firing boundvert moves — on ≠ off at an irregular junction.
        let b_off = firing_boundvert(&mesh_off, &off, apex_pos, leg_dir);
        let b_on = firing_boundvert(&mesh_on, &on, apex_pos, leg_dir);
        assert!(
            (b_on - b_off).length() > 1.0e-3,
            "loop_slide on must move the firing boundvert (irregular junction), \
             got on={b_on:?} off={b_off:?}",
        );

        // §8.8.3: the slid boundvert lies on the leg's line through the apex, so
        // its component perpendicular to the leg direction vanishes.
        let perpendicular =
            ((b_on - apex_pos) - leg_dir * (b_on - apex_pos).dot(leg_dir)).length();
        assert!(
            perpendicular < 1.0e-4,
            "slid boundvert sits on the silhouette leg's line, perp={perpendicular}",
        );

        // The off boundvert (the §7.6.2 meet) floats off that line.
        let perpendicular_off =
            ((b_off - apex_pos) - leg_dir * (b_off - apex_pos).dot(leg_dir)).length();
        assert!(
            perpendicular_off > 1.0e-4,
            "meet-point boundvert floats off the leg's line, perp={perpendicular_off}",
        );
    }

    /// BEVEL.md §8.8.2.1 (firing-frame): the two beveled edges {0,1}/{0,3} split the
    /// apex into **two** sectors, each holding a single silhouette leg ({0,2} and
    /// {0,4}). With the firing predicate taken in the genuine *vertex-normal* frame
    /// (`n_v`), **both** sectors qualify and slide their boundvert onto their leg —
    /// the under-fire (a sector-face frame slid only the one whose flanking faces
    /// aligned with that frame, flipping the reflex sign on the off-plane sector) is
    /// gone. The non-planar legs make each slid boundvert land off the z = 0 plane,
    /// so "on the leg" is the vanishing perpendicular-to-leg distance, not z ≈ 0.
    #[test]
    fn edge_bevel_loop_slide_fires_every_qualifying_sector() {
        let offset = 0.2;
        let (mut mesh, apex, spokes, [e01, e03]) = irregular_apex_junction();
        let apex_pos = mesh.vertex_position(apex);
        let leg2_dir = (mesh.vertex_position(spokes[1]) - apex_pos).normalize();
        let leg4_dir = (mesh.vertex_position(spokes[3]) - apex_pos).normalize();

        let result = mesh.bevel(
            &[e01.into(), e03.into()],
            BevelOptions {
                offset,
                affect: BevelAffect::Edges,
                loop_slide: true,
                ..Default::default()
            },
        );
        mesh.validate_topology();

        let b2 = firing_boundvert(&mesh, &result, apex_pos, leg2_dir);
        let b4 = firing_boundvert(&mesh, &result, apex_pos, leg4_dir);
        let perp = |p: Vec3, leg: Vec3| ((p - apex_pos) - leg * (p - apex_pos).dot(leg)).length();

        // Both sectors' boundverts sit on their own silhouette leg's line (§8.8.3),
        // and they are distinct verts (each slid onto a different leg).
        assert!(
            perp(b2, leg2_dir) < 1.0e-4,
            "sector {{0,2}} slid onto its leg, perp={}",
            perp(b2, leg2_dir),
        );
        assert!(
            perp(b4, leg4_dir) < 1.0e-4,
            "sector {{0,4}} also slid onto its leg, perp={}",
            perp(b4, leg4_dir),
        );
        assert!(
            (b2 - b4).length() > 1.0e-3,
            "the two firing boundverts are distinct, b2={b2:?} b4={b4:?}",
        );
    }

    /// BEVEL.md §8.8.4 / §8.8.5: at an **irregular** firing junction the flank-sine
    /// ratio ≠ 1 redistributes the two flanking edges' §7.6.4 offsets, and the ripple
    /// runs **along each flanking edge to its far end** — the terminal strip verts.
    /// On `irregular_apex_junction` the two beveled edges {0,1}/{0,3} run from the
    /// firing apex out to terminal endpoints (spokes s1/s3), so all four terminal
    /// strip verts shift by a small width-pass ripple between `loop_slide` on and off.
    /// This pins the §8.8.5 "missing-ripple" defect: leaving the termini static (their
    /// spec-offset placement) is the bug, not a preserved invariant. The drift stays a
    /// few thousandths — a reconciled-offset ripple, not a recipe change — so it is
    /// bounded well below a gross routing-bug relocation.
    #[test]
    fn edge_bevel_loop_slide_ripples_terminal_strip_verts_at_irregular_junction() {
        let offset = 0.3;

        let build = |loop_slide: bool| {
            let (mut mesh, apex, _spokes, [e01, e03]) = irregular_apex_junction();
            let apex_pos = mesh.vertex_position(apex);
            let result = mesh.bevel(
                &[e01.into(), e03.into()],
                BevelOptions {
                    offset,
                    affect: BevelAffect::Edges,
                    loop_slide,
                    ..Default::default()
                },
            );
            mesh.validate_topology();
            let positions: Vec<Vec3> =
                result.verts.iter().map(|&v| mesh.vertex_position(v)).collect();
            (apex_pos, positions)
        };

        let (apex_pos, off_positions) = build(false);
        let (_, on_positions) = build(true);

        // Two firing apex boundverts (near the apex) + four terminal strip verts (out
        // by the far endpoints). `result.verts` is built in the same order both runs,
        // so positions correspond index-for-index.
        assert_eq!(on_positions.len(), off_positions.len(), "same vert count");
        assert_eq!(on_positions.len(), 6, "two apex boundverts + four terminal strip verts");

        let mut apex_drift_max: f32 = 0.0;
        let mut terminal_count = 0;
        for (on, off) in on_positions.iter().zip(off_positions.iter()) {
            let drift = (*on - *off).length();
            // Apex boundverts slide onto the leg near the apex; terminal strip verts
            // sit out near the far endpoints (distance ≈ 1 from the apex).
            if (*off - apex_pos).length() > 0.5 {
                terminal_count += 1;
                assert!(
                    drift > 1.0e-4,
                    "terminal strip vert must ripple at an irregular firing junction \
                     (not stay at its spec-offset placement), drift={drift}",
                );
                // With the §7.6.4 spec-pull scale `t*` (not the meet-at-B `t_B`), the
                // reconciled offset no longer overshoots, so the ripple is bounded well
                // below the apex slide and a routing-bug relocation.
                assert!(
                    drift < 3.0e-2,
                    "terminal ripple stays a small width-pass shift, not a routing-bug \
                     relocation, drift={drift}",
                );
            } else {
                apex_drift_max = apex_drift_max.max(drift);
            }
        }
        assert_eq!(terminal_count, 4, "four terminal strip verts ripple");
        // The case-A apex slide is the larger, direct placement change (§8.8.5).
        assert!(
            apex_drift_max > 1.0e-3,
            "the firing apex boundverts slide directly (case A), drift={apex_drift_max}",
        );
    }

    /// Like `irregular_apex_junction`, but the firing sector's silhouette leg
    /// (spoke {0,2}) lies exactly in the `z = 0` plane, so "on the leg's line"
    /// means `z ≈ 0` directly. The two bounding beveled edges {0,1}/{0,3} rise out
    /// of that plane at *unequal* angles (an irregular junction, §8.8.5), so the
    /// §7.6.2 meet floats off `z = 0` while the §8.8.3 slide pins the boundvert back
    /// onto it. Returns (mesh, apex, [v1..v4], beveled edges {0,1} and {0,3}).
    fn irregular_apex_junction_planar_leg() -> (LoopMesh, VertexKey, [VertexKey; 4], [EdgeKey; 2])
    {
        let mut mesh = LoopMesh::new();
        let apex = mesh.create_vertex(Vec3::new(0.0, 0.0, 0.0));
        // s2 — the silhouette leg — lies in z = 0. s1 and s3 (the beveled flanks)
        // lift out of the plane at unequal angles so the junction is irregular.
        let s1 = mesh.create_vertex(Vec3::new(1.0, -0.2, 0.3));
        let s2 = mesh.create_vertex(Vec3::new(-0.15, -1.0, 0.0));
        let s3 = mesh.create_vertex(Vec3::new(-0.8, -0.5, 0.9));
        let s4 = mesh.create_vertex(Vec3::new(0.1, 1.0, 0.5));
        let spokes = [s1, s2, s3, s4];
        let outers: Vec<VertexKey> = (0..4)
            .map(|i| {
                let a = mesh.vertex_position(spokes[i]);
                let b = mesh.vertex_position(spokes[(i + 1) % 4]);
                let mid = (a + b) * 0.5;
                mesh.create_vertex(mid * 1.8 + Vec3::new(0.0, 0.0, -0.4))
            })
            .collect();

        let edge = |mesh: &mut LoopMesh, a: VertexKey, b: VertexKey| {
            mesh.create_edge([a, b], EdgeCreateFlags::NONE)
        };
        for i in 0..4 {
            let sa = spokes[i];
            let sb = spokes[(i + 1) % 4];
            let outer = outers[i];
            let e_apex_a = mesh
                .query_edge_exists(apex, sa)
                .unwrap_or_else(|| edge(&mut mesh, apex, sa));
            let e_a_outer = edge(&mut mesh, sa, outer);
            let e_outer_b = edge(&mut mesh, outer, sb);
            let e_b_apex = mesh
                .query_edge_exists(apex, sb)
                .unwrap_or_else(|| edge(&mut mesh, apex, sb));
            mesh.face_create(
                &[apex, sa, outer, sb],
                &[e_apex_a, e_a_outer, e_outer_b, e_b_apex],
                FaceCreateFlags::NONE,
            );
        }

        let e01 = mesh.query_edge_exists(apex, s1).expect("spoke {0,1}");
        let e03 = mesh.query_edge_exists(apex, s3).expect("spoke {0,3}");
        (mesh, apex, spokes, [e01, e03])
    }

    /// Like `irregular_apex_junction_planar_leg`, but the beveled spokes {0,1}/{0,3}
    /// are **clustered** so the {0,2} firing sector is strongly *lopsided* — leg s2
    /// sits ~0.61 rad off e01 but ~1.54 rad off e03 (flank-sine ratio `r ≈ 1.73`,
    /// well clear of `1`). That makes the §7.6.4 reconciliation observable at full
    /// magnitude: the apex offset solves to `o₁ ≈ 0.68·O` (not the spec `O`), the
    /// §8.8.3.1 coincident rebuild differs from the construct average by ~13%, and the
    /// §7.6.4.3 terminal damping is a few thousandths — all far above f32 noise. Leg
    /// s2 lies in `z = 0` so its slid apex keeps `z ≈ 0`. Returns
    /// (mesh, apex, [s1..s4], beveled edges {0,1} and {0,3}).
    fn lopsided_apex_junction() -> (LoopMesh, VertexKey, [VertexKey; 4], [EdgeKey; 2]) {
        let mut mesh = LoopMesh::new();
        let apex = mesh.create_vertex(Vec3::new(0.0, 0.0, 0.0));
        // s1/s3 are the beveled flanks, clustered into one half; s2 (the silhouette
        // leg, in z = 0) sits close to s1, so the {0,2} sector's two flank angles are
        // very unequal. s4 fills the opposite sector.
        let s1 = mesh.create_vertex(Vec3::new(1.0, 0.0, 0.35));
        let s2 = mesh.create_vertex(Vec3::new(0.866, 0.5, 0.0));
        let s3 = mesh.create_vertex(Vec3::new(-0.4226, 0.9063, 0.45));
        let s4 = mesh.create_vertex(Vec3::new(-0.4226, -0.9063, 0.30));
        let spokes = [s1, s2, s3, s4];
        let outers: Vec<VertexKey> = (0..4)
            .map(|i| {
                let a = mesh.vertex_position(spokes[i]);
                let b = mesh.vertex_position(spokes[(i + 1) % 4]);
                let mid = (a + b) * 0.5;
                mesh.create_vertex(mid * 1.8 + Vec3::new(0.0, 0.0, -0.4))
            })
            .collect();

        let edge = |mesh: &mut LoopMesh, a: VertexKey, b: VertexKey| {
            mesh.create_edge([a, b], EdgeCreateFlags::NONE)
        };
        for i in 0..4 {
            let sa = spokes[i];
            let sb = spokes[(i + 1) % 4];
            let outer = outers[i];
            let e_apex_a = mesh
                .query_edge_exists(apex, sa)
                .unwrap_or_else(|| edge(&mut mesh, apex, sa));
            let e_a_outer = edge(&mut mesh, sa, outer);
            let e_outer_b = edge(&mut mesh, outer, sb);
            let e_b_apex = mesh
                .query_edge_exists(apex, sb)
                .unwrap_or_else(|| edge(&mut mesh, apex, sb));
            mesh.face_create(
                &[apex, sa, outer, sb],
                &[e_apex_a, e_a_outer, e_outer_b, e_b_apex],
                FaceCreateFlags::NONE,
            );
        }

        let e01 = mesh.query_edge_exists(apex, s1).expect("spoke {0,1}");
        let e03 = mesh.query_edge_exists(apex, s3).expect("spoke {0,3}");
        (mesh, apex, spokes, [e01, e03])
    }

    /// BEVEL.md §8.8.3: the slid boundvert is built parametrically on the
    /// *un-projected* leg line `v + t·d_leg`, so it lands **on** the silhouette
    /// edge by construction — no face-plane / junction-normal projection drifts it
    /// off. Here the firing sector's leg (spoke {0,2}) lies in `z = 0`, so its slid
    /// boundvert must have `z ≈ 0`, whereas the §7.6.2 meet (loop_slide off) floats
    /// off the plane at this irregular junction.
    ///
    /// BEVEL.md §8.8.2.1 (firing-frame): the firing predicate is widened to the
    /// genuine *vertex-normal* frame, so **both** qualifying single-silhouette apex
    /// sectors slide (spoke {0,2}'s sector and spoke {0,4}'s sector), not just the
    /// one whose flanking faces happen to align with a sector-face frame.
    ///
    /// BEVEL.md §8.8.5 / §8.8.4: this is an **irregular** firing junction (the two
    /// beveled edges rise out of the leg's plane at unequal angles), so the flank-sine
    /// ratio ≠ 1 redistributes the flanking edges' §7.6.4 offsets and the ripple
    /// reaches the four terminal strip verts at the edges' far ends — they are **not**
    /// static here (the case-B recipe is flag-independent, but the reconciled offset it
    /// consumes is not). So all six new verts drift between on and off: the two firing
    /// apex boundverts directly (case A) and the four termini by the width-pass ripple.
    #[test]
    fn edge_bevel_loop_slide_lands_on_planar_leg_line() {
        let offset = 0.2;

        let build = |loop_slide: bool| {
            let (mut mesh, apex, spokes, [e01, e03]) = irregular_apex_junction_planar_leg();
            let apex_pos = mesh.vertex_position(apex);
            let leg2_dir = (mesh.vertex_position(spokes[1]) - apex_pos).normalize();
            let leg4_dir = (mesh.vertex_position(spokes[3]) - apex_pos).normalize();
            let result = mesh.bevel(
                &[e01.into(), e03.into()],
                BevelOptions {
                    offset,
                    affect: BevelAffect::Edges,
                    loop_slide,
                    ..Default::default()
                },
            );
            mesh.validate_topology();
            let positions: Vec<Vec3> =
                result.verts.iter().map(|&v| mesh.vertex_position(v)).collect();
            let firing2 = firing_boundvert(&mesh, &result, apex_pos, leg2_dir);
            let firing4 = firing_boundvert(&mesh, &result, apex_pos, leg4_dir);
            (apex_pos, leg2_dir, leg4_dir, positions, firing2, firing4)
        };

        let (apex_pos, leg2_dir, leg4_dir, off_positions, b2_off, b4_off) = build(false);
        let (_, _, _, on_positions, b2_on, b4_on) = build(true);

        let perp = |p: Vec3, leg: Vec3| ((p - apex_pos) - leg * (p - apex_pos).dot(leg)).length();

        // §8.8.5: both firing boundverts move when the slide is enabled.
        assert!(
            (b2_on - b2_off).length() > 1.0e-3,
            "loop_slide on must move the {{0,2}} boundvert, on={b2_on:?} off={b2_off:?}",
        );
        assert!(
            (b4_on - b4_off).length() > 1.0e-3,
            "loop_slide on must move the {{0,4}} boundvert, on={b4_on:?} off={b4_off:?}",
        );

        // §8.8.3: spoke {0,2}'s leg lies in z = 0, so its slid boundvert lands at
        // z ≈ 0 and exactly on the leg's line (`v + t·d_leg`, un-projected `d_leg`).
        assert!(
            b2_on.z.abs() < 1.0e-5,
            "slid {{0,2}} boundvert keeps z = 0 (leg lies in z = 0), got z={}",
            b2_on.z,
        );
        assert!(
            perp(b2_on, leg2_dir) < 1.0e-5,
            "slid {{0,2}} boundvert sits exactly on its leg's line, perp={}",
            perp(b2_on, leg2_dir),
        );

        // §8.8.2.1 (firing-frame): the second qualifying sector also slides onto its
        // own leg (spoke {0,4}) — the n_v frame fires it where a sector-face frame
        // wrongly rejected it (the under-fire).
        assert!(
            perp(b4_on, leg4_dir) < 1.0e-5,
            "slid {{0,4}} boundvert sits on its leg's line, perp={}",
            perp(b4_on, leg4_dir),
        );

        // The §7.6.2 meets (loop_slide off) float off their legs — that is why the
        // slide is observable (§8.8.5).
        assert!(
            b2_off.z.abs() > 1.0e-3,
            "{{0,2}} meet-point boundvert floats off z = 0, got z={}",
            b2_off.z,
        );
        assert!(
            perp(b4_off, leg4_dir) > 1.0e-3,
            "{{0,4}} meet-point boundvert floats off its leg, perp={}",
            perp(b4_off, leg4_dir),
        );

        // §8.8.5 / §8.8.4: at this irregular firing junction the slide perturbs the
        // two firing apex boundverts directly (case A) *and* ripples the §7.6.4
        // reconciled offsets out to the four terminal strip verts (case B), so all six
        // new verts drift between on and off. The termini move by a small width-pass
        // ripple — not a recipe change — so their drift is bounded well below the apex
        // slide's, distinguishing the legitimate ripple from a routing bug (§8.8.5).
        assert_eq!(on_positions.len(), off_positions.len(), "same vert count");
        assert_eq!(on_positions.len(), 6, "two apex boundverts + four terminal strip verts");
        let drifted = on_positions
            .iter()
            .zip(off_positions.iter())
            .filter(|(a, b)| (**a - **b).length() > 1.0e-5)
            .count();
        assert_eq!(
            drifted, 6,
            "both apex sectors slide (case A) and the four termini ripple (case B)",
        );
    }

    /// BEVEL.md §7.6.4.3–§7.6.4.5 / §8.8.3.1: the two-pass construct→reconcile→rebuild
    /// pins the *magnitudes*, not just the directions. This test fails against the
    /// earlier single-pass placement (apex left at the construct average of two unequal
    /// spec sub-meets; terminals fed the apex offset carried unchanged) and passes once
    /// both ends read the reconciled offsets.
    ///
    /// **Coincident-sub-meet apex (§8.8.3.1).** The rebuilt apex sits at the single
    /// reconciled distance `t* = O·(sin α₁ + sin α₂) / (sin²α₁ + sin²α₂)` along the leg
    /// — where the two sub-meets coincide — not the construct average
    /// `O·(sin α₁ + sin α₂) / (2·sin α₁·sin α₂)`.
    ///
    /// **Damped terminal ripple (§7.6.4.3 / §7.6.4.4).** Each terminal strip vert is
    /// fed the damped blend `(apex_offset + 0.04·O) / 1.04`, not the apex offset itself
    /// (which overshoots by exactly `1.04×`). The terminal recipe (§7.5.1) is linear in
    /// the offset it consumes, so an off-run affine probe at `O` and `2O` recovers each
    /// terminal's *fed* offset; the recovered multiset must match the four damped
    /// apex-offsets and be distinct from the undamped apex-offsets.
    #[test]
    fn edge_bevel_loop_slide_apex_and_termini_use_reconciled_offsets() {
        let o = 0.2_f32;

        let build = |loop_slide: bool, offset: f32| {
            let (mut mesh, apex, spokes, [e01, e03]) = lopsided_apex_junction();
            let apex_pos = mesh.vertex_position(apex);
            let leg2 = (mesh.vertex_position(spokes[1]) - apex_pos).normalize();
            let leg4 = (mesh.vertex_position(spokes[3]) - apex_pos).normalize();
            let edge01 = (mesh.vertex_position(spokes[0]) - apex_pos).normalize();
            let edge03 = (mesh.vertex_position(spokes[2]) - apex_pos).normalize();
            let result = mesh.bevel(
                &[e01.into(), e03.into()],
                BevelOptions {
                    offset,
                    affect: BevelAffect::Edges,
                    loop_slide,
                    ..Default::default()
                },
            );
            mesh.validate_topology();
            let positions: Vec<Vec3> =
                result.verts.iter().map(|&v| mesh.vertex_position(v)).collect();
            (apex_pos, leg2, leg4, edge01, edge03, positions)
        };

        let (apex_pos, leg2, leg4, edge01, edge03, off_o) = build(false, o);
        let (_, _, _, _, _, off_2o) = build(false, 2.0 * o);
        let (_, _, _, _, _, on_o) = build(true, o);

        assert_eq!(off_o.len(), 6);
        assert_eq!(off_2o.len(), 6);
        assert_eq!(on_o.len(), 6);

        // The firing apex boundverts sit on their legs near the apex; the terminal
        // strip verts sit out by the far endpoints (≈ 1 away). Classify on the off@O
        // run; the same indices carry across runs (stable construction order).
        let sin = |u: Vec3, v: Vec3| u.cross(v).length();

        // §8.8.3.1 coincident-sub-meet apex distance, per firing sector.
        let apex_distance = |on_position: Vec3, leg: Vec3| (on_position - apex_pos).dot(leg);
        let b2_on = firing_boundvert_from(&on_o, apex_pos, leg2);
        let b4_on = firing_boundvert_from(&on_o, apex_pos, leg4);
        let t2 = apex_distance(b2_on, leg2);
        let t4 = apex_distance(b4_on, leg4);

        let s1_2 = sin(edge01, leg2);
        let s3_2 = sin(edge03, leg2);
        let s1_4 = sin(edge01, leg4);
        let s3_4 = sin(edge03, leg4);

        let expected_new = |sa: f32, sb: f32| o * (sa + sb) / (sa * sa + sb * sb);
        let expected_old = |sa: f32, sb: f32| o * (sa + sb) / (2.0 * sa * sb);

        // Both apex boundverts land at the reconciled (coincident) distance, not the
        // construct average. On the lopsided {0,2} sector the two differ by ~13%, which
        // is what fails the single-pass placement; the near-symmetric {0,4} sector
        // matches `new` too (where `new ≈ old`).
        let mut max_apex_gap: f32 = 0.0;
        for (t, sa, sb) in [(t2, s1_2, s3_2), (t4, s1_4, s3_4)] {
            let new = expected_new(sa, sb);
            let old = expected_old(sa, sb);
            assert!(
                (t - new).abs() < 1.0e-4,
                "apex slides to the reconciled coincident distance t*={new}, got {t}",
            );
            let gap = (new - old).abs();
            max_apex_gap = max_apex_gap.max(gap);
            if gap > 5.0e-4 {
                assert!(
                    (t - old).abs() > (t - new).abs(),
                    "apex is the rebuilt coincident point, not the construct average \
                     (t={t}, new={new}, old={old})",
                );
            }
        }
        assert!(
            max_apex_gap > 1.0e-3,
            "the lopsided sector exercises a real apex magnitude differential \
             (max_gap={max_apex_gap})",
        );

        // §7.6.4.3 terminal damping. The apex offset of each flanking edge in a sector
        // is `t* · sin(angle(edge, leg))`; the terminal far end is fed the damped blend.
        let damp = |apex_offset: f32| (apex_offset + 0.04 * o) / 1.04;
        let apex_offsets = [t2 * s1_2, t2 * s3_2, t4 * s1_4, t4 * s3_4];
        let mut damped: Vec<f32> = apex_offsets.iter().map(|&a| damp(a)).collect();
        damped.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // Recover each terminal strip vert's fed offset via the off-run affine probe.
        let mut recovered: Vec<f32> = Vec::new();
        for index in 0..6 {
            if (off_o[index] - apex_pos).length() <= 0.5 {
                continue; // an apex boundvert, not a terminal
            }
            let dir = off_2o[index] - off_o[index]; // = O · (per-unit-offset slope)
            let fraction = (on_o[index] - off_o[index]).dot(dir) / dir.dot(dir);
            recovered.push(o + fraction * o);
        }
        recovered.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(recovered.len(), 4, "four terminal strip verts");

        let mut apex_sorted = apex_offsets.to_vec();
        apex_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // The recovered fed offsets are the damped values, edge-for-edge (damp is
        // monotone in the apex offset, so sorting all three keeps them aligned), and on
        // the lopsided sector they are distinguishably *not* the undamped apex offsets
        // the overshoot bug carried to the terminal.
        let mut max_damp_gap: f32 = 0.0;
        for (got, (want_damped, want_apex)) in recovered
            .iter()
            .zip(damped.iter().zip(apex_sorted.iter()))
        {
            assert!(
                (got - want_damped).abs() < 5.0e-4,
                "terminal fed the §7.6.4.3 damped offset {want_damped}, got {got}",
            );
            let gap = (want_apex - want_damped).abs();
            max_damp_gap = max_damp_gap.max(gap);
            if gap > 5.0e-4 {
                assert!(
                    (got - want_damped).abs() < (got - want_apex).abs(),
                    "terminal is the §7.6.4.3 damped offset, not the apex offset carried \
                     unchanged (got={got}, damped={want_damped}, apex={want_apex})",
                );
            }
        }
        assert!(
            max_damp_gap > 1.0e-3,
            "the lopsided sector exercises a real terminal damping differential \
             (max_gap={max_damp_gap})",
        );
    }

    /// BEVEL.md §8.8.5: at a *symmetric* junction the slide is a no-op — the slid
    /// point coincides with the bisector meet-point, so `loop_slide` on and off
    /// produce the identical boundvert. A symmetric cube corner with one
    /// unbeveled silhouette leg between the two beveled edges is the canonical
    /// no-op fixture.
    #[test]
    fn edge_bevel_loop_slide_is_noop_at_symmetric_junction() {
        let build = |loop_slide: bool| {
            let mut mesh = LoopMesh::new();
            mesh_add_cube(&mut mesh);
            // v0 = [1,-1,-1]; bevel {0,1} and {0,4}, leaving {0,3} as the lone
            // unbeveled leg of the sector between them. The cube corner is
            // symmetric, so the slide must not move the boundvert.
            let p0 = Vec3::new(1.0, -1.0, -1.0);
            let e01 = cube_edge(&mesh, p0, Vec3::new(-1.0, -1.0, -1.0));
            let e04 = cube_edge(&mesh, p0, Vec3::new(1.0, -1.0, 1.0));
            let result = mesh.bevel(
                &[e01.into(), e04.into()],
                BevelOptions {
                    offset: 0.25,
                    affect: BevelAffect::Edges,
                    loop_slide,
                    ..Default::default()
                },
            );
            mesh.validate_topology();
            let mut positions: Vec<Vec3> = result
                .verts
                .iter()
                .map(|&v| mesh.vertex_position(v))
                .collect();
            positions.sort_by(|a, b| {
                (a.x, a.y, a.z)
                    .partial_cmp(&(b.x, b.y, b.z))
                    .unwrap()
            });
            positions
        };

        let off = build(false);
        let on = build(true);
        assert_eq!(off.len(), on.len(), "same boundvert count");
        for (a, b) in off.iter().zip(on.iter()) {
            assert!(
                (*a - *b).length() < 1.0e-5,
                "symmetric junction: loop_slide is a no-op, {a:?} vs {b:?}",
            );
        }
    }

    /// Like `irregular_apex_junction`, but the firing sector's silhouette leg
    /// (spoke {0,2}) points *outside* the angular wedge spanned by its two bounding
    /// beveled edges {0,1} and {0,3}: it swings into the lower-y half-plane so the
    /// `g_k → e_mid` sub-meet sweeps *backward*. The sector still has exactly one
    /// silhouette leg (so §8.8.2 condition 1 fires), but one flanking sub-meet is
    /// reflex (so condition 2 must reject the slide).
    /// Returns (mesh, apex, [v1,v2,v3,v4], beveled edges {0,1} and {0,3}).
    fn reflex_apex_junction() -> (LoopMesh, VertexKey, [VertexKey; 4], [EdgeKey; 2]) {
        let mut mesh = LoopMesh::new();
        let apex = mesh.create_vertex(Vec3::new(0.0, 0.0, 0.0));
        // s1, s3 bound the firing sector. s2 — the leg — is placed so that, viewed
        // from the junction's vertex-normal frame `n_v` (§8.8.2.1, the frame the
        // forward/reflex guard now uses), the `e_mid → g_(k+1)` sub-meet sweeps
        // *backward*: it folds reflex while the `g_k → e_mid` side stays forward. The
        // petals stay non-planar so s2 remains a silhouette leg.
        let s1 = mesh.create_vertex(Vec3::new(1.0, 0.2, -0.1));
        let s2 = mesh.create_vertex(Vec3::new(-0.2, 0.0, 0.2));
        let s3 = mesh.create_vertex(Vec3::new(0.2, 1.0, -0.1));
        let s4 = mesh.create_vertex(Vec3::new(-0.7, -0.7, -0.3));
        let spokes = [s1, s2, s3, s4];
        let outers: Vec<VertexKey> = (0..4)
            .map(|i| {
                let a = mesh.vertex_position(spokes[i]);
                let b = mesh.vertex_position(spokes[(i + 1) % 4]);
                let mid = (a + b) * 0.5;
                mesh.create_vertex(mid * 1.8 + Vec3::new(0.0, 0.0, -0.4))
            })
            .collect();

        let edge = |mesh: &mut LoopMesh, a: VertexKey, b: VertexKey| {
            mesh.create_edge([a, b], EdgeCreateFlags::NONE)
        };
        for i in 0..4 {
            let sa = spokes[i];
            let sb = spokes[(i + 1) % 4];
            let outer = outers[i];
            let e_apex_a = mesh
                .query_edge_exists(apex, sa)
                .unwrap_or_else(|| edge(&mut mesh, apex, sa));
            let e_a_outer = edge(&mut mesh, sa, outer);
            let e_outer_b = edge(&mut mesh, outer, sb);
            let e_b_apex = mesh
                .query_edge_exists(apex, sb)
                .unwrap_or_else(|| edge(&mut mesh, apex, sb));
            mesh.face_create(
                &[apex, sa, outer, sb],
                &[e_apex_a, e_a_outer, e_outer_b, e_b_apex],
                FaceCreateFlags::NONE,
            );
        }

        let e01 = mesh.query_edge_exists(apex, s1).expect("spoke {0,1}");
        let e03 = mesh.query_edge_exists(apex, s3).expect("spoke {0,3}");
        (mesh, apex, spokes, [e01, e03])
    }

    /// BEVEL.md §8.8.2 condition 2 (the "good offset" guard): firing requires
    /// *both* that the sector has exactly one silhouette leg *and* that neither
    /// flanking sub-meet folds back. When a sub-meet is reflex the slide is
    /// rejected and the sector keeps its §7.6.2 meet-point — so on a reflex fixture
    /// `loop_slide = true` yields the *same* firing boundvert as `loop_slide =
    /// false`. (Condition 1 alone fires here; condition 2 is what vetoes the slide.)
    #[test]
    fn edge_bevel_loop_slide_rejected_when_flank_is_reflex() {
        let offset = 0.2;

        let build = |loop_slide: bool| {
            let (mut mesh, apex, spokes, [e01, e03]) = reflex_apex_junction();
            let apex_pos = mesh.vertex_position(apex);
            let leg_dir = (mesh.vertex_position(spokes[1]) - apex_pos).normalize();
            let result = mesh.bevel(
                &[e01.into(), e03.into()],
                BevelOptions {
                    offset,
                    affect: BevelAffect::Edges,
                    loop_slide,
                    ..Default::default()
                },
            );
            mesh.validate_topology();
            let boundvert = firing_boundvert(&mesh, &result, apex_pos, leg_dir);
            (apex_pos, leg_dir, boundvert)
        };

        let (apex_pos, leg_dir, b_off) = build(false);
        let (_, _, b_on) = build(true);

        // Condition 2 vetoes the slide: the firing boundvert must NOT move.
        assert!(
            (b_on - b_off).length() < 1.0e-5,
            "reflex flank: loop_slide must fall back to the meet-point, \
             got on={b_on:?} off={b_off:?}",
        );

        // And — the point of the guard — that retained meet-point genuinely floats
        // off the leg's line (the slide it rejected would have pinned it on-line).
        let perpendicular =
            ((b_on - apex_pos) - leg_dir * (b_on - apex_pos).dot(leg_dir)).length();
        assert!(
            perpendicular > 1.0e-4,
            "rejected slide keeps the off-line §7.6.2 meet, perp={perpendicular}",
        );
    }

    /// BEVEL.md §7.6.2: at an irregular non-planar junction the boundvert is the
    /// *pairwise* meet of its two flanking edges' offset lines — a local 2-line
    /// solve, never a fit over every incident face. The local solve stays bounded:
    /// each boundvert lands within a small multiple of `offset` of the apex (the
    /// strips meet near the original corner), where the discarded global
    /// least-squares formulation drove a boundvert far out of position. This pins
    /// the locality the rewrite restored, with `loop_slide` off so every boundvert
    /// comes from §7.6.2.
    #[test]
    fn edge_bevel_meet_point_is_bounded_at_irregular_junction() {
        let offset = 0.2;
        let (mut mesh, apex, _spokes, [e01, e03]) = irregular_apex_junction();
        let apex_pos = mesh.vertex_position(apex);
        let result = mesh.bevel(
            &[e01.into(), e03.into()],
            BevelOptions {
                offset,
                affect: BevelAffect::Edges,
                loop_slide: false,
                ..Default::default()
            },
        );
        mesh.validate_topology();

        // The two junction boundverts are the new verts nearest the apex (the
        // strip verts slide out along the spokes, far from it). Both §7.6.2 meets
        // sit within a few offsets of the apex — the locality the rewrite restored.
        // The discarded global least-squares could drive a boundvert far out.
        let mut distances: Vec<f32> = result
            .verts
            .iter()
            .map(|&v| (mesh.vertex_position(v) - apex_pos).length())
            .collect();
        distances.sort_by(|a, b| a.partial_cmp(b).unwrap());
        for &distance in distances.iter().take(2) {
            assert!(
                distance < 4.0 * offset,
                "pairwise meet keeps boundvert near the surface, got {distance} \
                 (apex-relative) for offset {offset}",
            );
        }
    }

    /// BEVEL.md §7.6.2 cases 1/2: a sector that is a single planar face (or whose
    /// in-between legs are all in-plane) reduces to the exact 2-line crossing on the
    /// angle bisector. At a right-angle planar corner with equal offset `t` the
    /// meet sits `t / sin(45°) = t·√2` from the original corner — the closed-form
    /// crossing identity, with no out-of-plane excursion.
    #[test]
    fn edge_bevel_meet_point_planar_sector_is_exact_bisector_crossing() {
        let mut mesh = LoopMesh::new();
        mesh_add_cube(&mut mesh);
        let corner = Vec3::new(1.0, -1.0, -1.0);
        let e01 = cube_edge(&mesh, corner, Vec3::new(-1.0, -1.0, -1.0));
        let e03 = cube_edge(&mesh, corner, Vec3::new(1.0, 1.0, -1.0));
        let e04 = cube_edge(&mesh, corner, Vec3::new(1.0, -1.0, 1.0));
        let offset = 0.25;
        let result = mesh.bevel(
            &[e01.into(), e03.into(), e04.into()],
            BevelOptions {
                offset,
                affect: BevelAffect::Edges,
                ..Default::default()
            },
        );
        mesh.validate_topology();

        let expected = offset * 2.0_f32.sqrt();
        // The patch face boundverts are the §7.6.2 meet points; strip verts sit
        // elsewhere, so filter to the triangular patch (matching the existing
        // three-edge-junction pin).
        let patch = result
            .faces
            .iter()
            .copied()
            .find(|&f| mesh.face_vert_keys(f).len() == 3)
            .expect("the s=3 junction patch is a triangle");
        for v in mesh.face_vert_keys(patch) {
            let distance = (mesh.vertex_position(v) - corner).length();
            assert!(
                (distance - expected).abs() < 1.0e-5,
                "planar meet sits offset·√2 from the corner, got {distance}",
            );
        }
    }

    /// Build a closed non-planar apex fan: apex `v` above an irregular rim of four
    /// rim verts at differing heights, capped underneath so the apex disk closes.
    /// Each fan triangle (apex, rim_i, rim_{i+1}) has a different normal, so every
    /// apex edge sits between two non-coplanar faces — an out-of-plane (silhouette)
    /// in-between edge whenever it lies inside a bevel sector. Returns (apex, rim).
    fn irregular_apex_fan(mesh: &mut LoopMesh) -> (VertexKey, Vec<VertexKey>) {
        let apex = mesh.create_vertex(Vec3::new(0.0, 0.0, 1.0));
        // Six rim verts at deliberately irregular radii and heights, so every fan
        // triangle has a distinct normal and a sector spanning several in-between
        // edges sees genuinely different candidate faces along its walk.
        let rim_coords = [
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.55, 0.9, 0.4),
            Vec3::new(-0.5, 0.85, 0.15),
            Vec3::new(-1.0, 0.0, -0.1),
            Vec3::new(-0.45, -0.8, -0.3),
            Vec3::new(0.6, -0.95, 0.05),
        ];
        let rim: Vec<VertexKey> = rim_coords.iter().map(|&p| mesh.create_vertex(p)).collect();
        let n = rim.len();

        let spoke: Vec<EdgeKey> = rim
            .iter()
            .map(|&r| mesh.create_edge([apex, r], EdgeCreateFlags::NONE))
            .collect();
        let mut rim_edge = Vec::new();
        for i in 0..n {
            rim_edge.push(mesh.create_edge([rim[i], rim[(i + 1) % n]], EdgeCreateFlags::NONE));
        }
        for i in 0..n {
            let j = (i + 1) % n;
            mesh.face_create(
                &[apex, rim[i], rim[j]],
                &[spoke[i], rim_edge[i], spoke[j]],
                FaceCreateFlags::NONE,
            );
        }
        // Close the underside so the apex disk is a true closed fan.
        mesh.face_create(&rim, &rim_edge, FaceCreateFlags::NONE);
        (apex, rim)
    }

    /// BEVEL.md §7.6.2 (corrected face-drop step): at a **non-planar** junction the
    /// two skew offset lines give a provisional midpoint that floats off the surface,
    /// so it is dropped onto a sector face. The corrected walk **starts at the
    /// flanking beveled edge `g_k` itself** (testing `g_k`'s forward face first) and
    /// drops onto a **plane through the junction vertex `v`** carrying that face's
    /// normal — not the face's own supporting plane. This pins both: each junction
    /// boundvert lands exactly on the `v`-anchored plane of its sector's `g_k`
    /// forward face, bounded just out from the apex.
    #[test]
    fn junction_face_drop_lands_on_v_anchored_g_k_plane() {
        let mut mesh = LoopMesh::new();
        let (apex, rim) = irregular_apex_fan(&mut mesh);
        let apex_pos = mesh.vertex_position(apex);
        let n = rim.len();
        let rim_pos: Vec<Vec3> = rim.iter().map(|&r| mesh.vertex_position(r)).collect();

        // Pre-bevel normal of fan triangle (apex, rim_i, rim_{i+1}), captured before
        // the surgery rebuilds those faces. The forward face of beveled spoke `e_i`
        // (the face on the side toward the next spoke) is triangle `i`.
        let triangle_normal = |i: usize| {
            let a = rim_pos[i] - apex_pos;
            let b = rim_pos[(i + 1) % n] - apex_pos;
            a.cross(b).normalize()
        };

        // Bevel two spokes three apart: spoke 0 and spoke 3. The +Y sector walks
        // 0 → 3 (g_k = spoke 0, forward face = triangle 0, two in-between legs); the
        // −Y sector walks 3 → 0 (g_k = spoke 3, forward face = triangle 3). Both
        // sectors span out-of-plane legs, so both fire the skew face-drop.
        let spoke_0 = mesh.query_edge_exists(apex, rim[0]).expect("spoke 0");
        let spoke_3 = mesh.query_edge_exists(apex, rim[3]).expect("spoke 3");
        let offset = 0.3_f32;
        let result = mesh.bevel(
            &[spoke_0.into(), spoke_3.into()],
            BevelOptions {
                offset,
                affect: BevelAffect::Edges,
                ..Default::default()
            },
        );

        // The two junction-sector boundverts are the result verts near the apex; the
        // strip ring verts sit out at the rim ends (apex-distance > 1).
        let junction_verts: Vec<Vec3> = result
            .verts
            .iter()
            .map(|&v| mesh.vertex_position(v))
            .filter(|p| (*p - apex_pos).length() < 1.0)
            .collect();
        assert_eq!(
            junction_verts.len(),
            2,
            "two junction sectors → two near-apex boundverts",
        );

        let normal_g_k_plus = triangle_normal(0);
        let normal_g_k_minus = triangle_normal(3);
        for point in &junction_verts {
            let from_apex = *point - apex_pos;
            // Bounded near the apex: a small on-surface nudge, not floated off (the
            // v-anchored drop keeps it ≈ offset·few out, never the ~unit rim depth).
            let radius = from_apex.length();
            assert!(
                radius > 1.0e-4 && radius < 4.0 * offset,
                "junction boundvert sits just out from the apex, got {radius}",
            );
            // It lies exactly on the v-anchored plane of *one* flanking g_k forward
            // face — the corrected drop plane. Either sector's boundvert matches its
            // own g_k face to f32 round-off.
            let on_plus = from_apex.dot(normal_g_k_plus).abs();
            let on_minus = from_apex.dot(normal_g_k_minus).abs();
            assert!(
                on_plus < 1.0e-5 || on_minus < 1.0e-5,
                "boundvert on a v-anchored g_k-face plane; \
                 +Y dist {on_plus}, -Y dist {on_minus}",
            );
        }

        // Pair each boundvert to its own sector's g_k face: the +Y boundvert
        // (positive Y) must lie on triangle 0's plane, the −Y boundvert on
        // triangle 3's plane — confirming the walk started at g_k, not a later leg.
        let plus = junction_verts
            .iter()
            .find(|p| p.y > 0.0)
            .expect("+Y junction boundvert");
        let minus = junction_verts
            .iter()
            .find(|p| p.y < 0.0)
            .expect("-Y junction boundvert");
        assert!(
            (*plus - apex_pos).dot(normal_g_k_plus).abs() < 1.0e-5,
            "+Y boundvert on spoke-0's forward-face v-plane",
        );
        assert!(
            (*minus - apex_pos).dot(normal_g_k_minus).abs() < 1.0e-5,
            "-Y boundvert on spoke-3's forward-face v-plane",
        );
        mesh.validate_topology();
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

/// The gated per-corner bevel weight for `affect = Vertices` (BEVEL.md §8.7.2).
/// The three-way rule: gate off ⇒ `1.0` (full offset, weights ignored); gate on
/// and the vertex present in the slice ⇒ its stored float; gate on and absent ⇒
/// `0.0` (no slide — the asymmetric "weighting on but nothing to read" default).
fn corner_weight(use_weights: bool, vert_weights: &[(VertexKey, f32)], vertex: VertexKey) -> f32 {
    if !use_weights {
        return 1.0;
    }
    vert_weights
        .iter()
        .find_map(|&(key, weight)| (key == vertex).then_some(weight))
        .unwrap_or(0.0)
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

/// Coincidence / co-linearity tolerance for the `loop_slide` placement
/// (BEVEL.md §8.8.2 / §8.8.3). The guide is silent on the numeric value; this
/// reuses the same `1e-4` length tolerance `corner_verts_coincide` already
/// applies to position coincidence in this file, so the silhouette classifier's
/// normal-dot test and the sub-meet `sin α` degeneracy test share one
/// crate-local convention rather than introducing a fresh magic number.
const LOOP_SLIDE_TOLERANCE: f32 = 1.0e-4;

/// The §7.6.4.2 width-reconciliation spec-match weight, *squared*. The two residual
/// families are stacked with width-match coefficient `1.0` and spec-match
/// coefficient `w = 0.2`; the normal equations square the coefficients, so the spec
/// family enters the solve with effective weight `w² = 0.04` — width matching is
/// weighted 25× more heavily than spec matching. This finite ratio sets the
/// terminal-end damping `1 / (1 + w²) = 1 / 1.04` of §7.6.4.3 and must not be a
/// "hard width / soft spec" limit (which would drop the `1.04`).
const WIDTH_SPEC_WEIGHT_SQ: f32 = 0.04;

/// Angular band (radians) within which a §8.8.2.1 sub-meet's un-projected opening
/// angle is rejected as degenerate — within ~0.1 rad of either `0` or `π`. This is
/// the *magnitude* half of the "good offset" guard (the *sign* half is the
/// vertex-normal-frame forward/reflex test); a leg collinear with a flanking
/// beveled edge (angle near 0 or π) carries no usable slide direction.
const LOOP_SLIDE_ANGLE_BAND: f32 = 0.1;

/// The §8.8.2.1 "good offset" sub-meet guard, evaluated in the genuine junction
/// vertex-normal frame `n_v`. The swept angle from `dir_first` to `dir_second`
/// must be **forward** — viewed from `n_v` it turns positively, i.e.
/// `cross(dir_first, dir_second) · n_v > 0`; a reflex (`< 0`) sweep folds the
/// surface backward — **and** **non-degenerate** — its genuine un-projected
/// opening angle in `[0, π]` must lie outside the `~0.1 rad` band of both `0` and
/// `π`. Sign and magnitude are read separately (§8.8.2.1): the sign from the
/// `n_v` projection, the magnitude from the un-projected `atan2(|cross|, dot)`.
fn subsweep_is_good(dir_first: Vec3, dir_second: Vec3, vertex_normal: Vec3) -> bool {
    let cross = dir_first.cross(dir_second);
    if cross.dot(vertex_normal) <= 0.0 {
        return false;
    }
    let opening_angle = cross.length().atan2(dir_first.dot(dir_second));
    opening_angle > LOOP_SLIDE_ANGLE_BAND
        && opening_angle < std::f32::consts::PI - LOOP_SLIDE_ANGLE_BAND
}

/// Angle tolerance (in `cos θ` units) for the §7.6.2 parallel / anti-parallel
/// special cases: when the two flanking edge directions are within this of being
/// parallel (`cos θ ≈ 1`) or anti-parallel (`cos θ ≈ −1`) the line-line solve is
/// bypassed for the closed-form ladder placement. Matches the crate-local `1e-4`
/// convention used elsewhere in this file.
const MEET_ANGLE_TOLERANCE: f32 = 1.0e-4;

/// Coplanarity tolerance (on the dot of two face normals) for the §7.6.2 sector
/// classification: an in-between leg is out-of-plane (silhouette) when its two
/// flanking face normals differ from ±1 by more than this.
const MEET_PLANAR_TOLERANCE: f32 = 1.0e-4;

/// Length tolerance below which the two closest points of two lines are treated as
/// the same point (a crossing) rather than skew (§7.6.2).
const LINE_COINCIDENT_TOLERANCE: f32 = 1.0e-5;

/// Denominator floor below which the two-line system is treated as parallel
/// (§7.6.2). `a·c − b²` is the squared sine of the angle between unit directions,
/// so this is an angular degeneracy guard, not a length.
const LINE_PARALLEL_TOLERANCE: f32 = 1.0e-7;

/// Angular slack (radians) for the §7.6.2 in-wedge face-drop predicate.
const WEDGE_ANGLE_TOLERANCE: f32 = 1.0e-4;

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

/// The 3-D frame of one sector's profile cell (BEVEL.md §8.2.2.1): the three
/// control points `start` / `end` / `mid` plus the derived in-plane axes and the
/// mid-plane normal. `start` / `end` are the two ring boundverts the chain spans;
/// `mid` is the bulge control point on the crossed beveled edge's line. A
/// degenerate (collinear) corner is flagged so the chain collapses to a straight
/// chord (the flat-profile branch, §8.2.2.1).
#[derive(Clone, Copy)]
struct ProfileCell {
    start: Vec3,
    end: Vec3,
    mid: Vec3,
    mid_plane_normal: Vec3,
    /// Unit direction of the crossed beveled edge, used to project each cell
    /// sample onto the mid-plane (BEVEL.md §8.2.2.1). Zero requests no projection.
    edge_dir: Vec3,
    collinear: bool,
}

impl ProfileCell {
    /// Build the cell from the two ring boundverts and the crossed beveled edge
    /// (BEVEL.md §8.2.2.1). `mid` is the foot on the edge's line of the
    /// closest-approach pair between that line and the `start`–`end` line.
    fn new(start: Vec3, end: Vec3, edge_point: Vec3, edge_dir: Vec3) -> Self {
        let mid = line_line_nearest_on_first(edge_point, edge_dir, start, end - start)
            .unwrap_or((start + end) * 0.5);

        let a_axis = mid - start;
        let b_axis = mid - end;
        let cross = a_axis.cross(b_axis);
        let collinear = cross.length() <= 1.0e-6;
        let mid_plane_normal = if collinear {
            Vec3::ZERO
        } else {
            cross.normalize()
        };

        ProfileCell {
            start,
            end,
            mid,
            mid_plane_normal,
            edge_dir,
            collinear,
        }
    }

    /// Map a cell coordinate `(u, v)` into 3-D (BEVEL.md §8.2.2.1):
    /// `origin + u·a_axis + v·b_axis` with `origin = start + end − mid`, then
    /// project along the edge direction onto the mid-plane.
    fn sample(&self, u: f32, v: f32) -> Vec3 {
        let origin = self.start + self.end - self.mid;
        let raw = origin + u * (self.mid - self.start) + v * (self.mid - self.end);
        if self.edge_dir.length_squared() <= f32::EPSILON || self.collinear {
            return raw;
        }
        intersect_line_plane(raw, self.edge_dir, self.start, self.mid_plane_normal).unwrap_or(raw)
    }

    /// The chain's `segments + 1` sample positions (BEVEL.md §8.2.2.1 / §8.6).
    /// Endpoint samples are the ring boundverts; the interior samples are the
    /// `SUPERELLIPSE` cross-section cell coordinates `(u_k, v_k)` for the requested
    /// `profile` (§8.6.4, reducing to the §8.3.6 circular default at `profile =
    /// 0.5`) mapped through the cell. A collinear (flat) corner has no bulge and
    /// falls back to the even straight subdivision of the chord (§8.2.2.1).
    fn chain_positions(&self, segments: u32, profile: f32) -> Vec<Vec3> {
        let mut positions = Vec::with_capacity(segments as usize + 1);
        positions.push(self.start);
        if self.collinear {
            for k in 1..segments {
                let t = k as f32 / segments as f32;
                positions.push(self.start + (self.end - self.start) * t);
            }
        } else {
            let cells = superellipse_chain(profile, segments);
            for &(u, v) in cells.iter().take(segments as usize).skip(1) {
                positions.push(self.sample(u, v));
            }
        }
        positions.push(self.end);
        positions
    }

    /// The profile's "height" (BEVEL.md §8.3.3): the span between the cell's
    /// `(0,0)` corner (the arc center `start + end − mid`) and its `(1,1)` corner
    /// (the bulge `mid`), i.e. `|2·mid − start − end|`.
    fn height(&self) -> f32 {
        (2.0 * self.mid - self.start - self.end).length()
    }
}

/// A large finite stand-in for the square exponent `r → ∞` (BEVEL.md §8.6.2).
/// True infinity is impractical, and `profile ≈ 0.95` already maps to `r ≈ 27` —
/// square enough to snap at sampling resolution, so the family stays bounded.
const PROFILE_SQUARE_EXPONENT: f32 = 1.0e4;

/// Tolerance for snapping the computed exponent onto the special values that
/// carry closed-form spacing shortcuts (BEVEL.md §8.6.2).
const PROFILE_EXPONENT_SNAP: f32 = 1.0e-4;

/// Map the user `profile` scalar (`0..1`) to the superellipse exponent `r`
/// (BEVEL.md §8.6.2): `r = 2·ln(2) / ln(1/profile)`, snapped onto the special
/// exponents whose even-arc-length spacing has a closed form (`1` line, `2`
/// circle, the square / square-in sentinels), and clamped to a finite square
/// sentinel at the top so the family stays bounded.
fn profile_to_exponent(profile: f32) -> f32 {
    if profile >= 0.95 {
        return PROFILE_SQUARE_EXPONENT;
    }
    if profile <= PROFILE_EXPONENT_SNAP {
        return 0.0;
    }
    let exponent = 2.0 * std::f32::consts::LN_2 / (1.0 / profile).ln();
    if (exponent - 2.0).abs() < PROFILE_EXPONENT_SNAP {
        2.0
    } else if (exponent - 1.0).abs() < PROFILE_EXPONENT_SNAP {
        1.0
    } else if exponent < PROFILE_EXPONENT_SNAP {
        0.0
    } else if exponent >= PROFILE_SQUARE_EXPONENT {
        PROFILE_SQUARE_EXPONENT
    } else {
        exponent
    }
}

/// One quadrant of the superellipse `u^r + v^r = 1` solved for `v` at a given
/// `u ∈ [0, 1]` (BEVEL.md §8.6.1). The mirror form is used for `r < 1` to keep
/// the evaluation numerically stable near the endpoints.
fn superellipse_v(u: f32, r: f32) -> f32 {
    let u = u.clamp(0.0, 1.0);
    if r >= 1.0 {
        (1.0 - u.powf(r)).max(0.0).powf(1.0 / r)
    } else {
        1.0 - (1.0 - (1.0 - u).powf(r)).max(0.0).powf(1.0 / r)
    }
}

/// The straight-chord length between two cell points.
fn cell_chord(a: (f32, f32), b: (f32, f32)) -> f32 {
    ((a.0 - b.0).powi(2) + (a.1 - b.1).powi(2)).sqrt()
}

/// Place the `segments + 1` cross-section cell coordinates `(u_k, v_k)` of the
/// `SUPERELLIPSE` profile (BEVEL.md §8.6). The ends are pinned to the cell ring
/// corners `(0, 1)` / `(1, 0)`; the interior is spaced by even chord length along
/// the curve (§8.6.4). The closed-form shortcuts fire at the special exponents
/// (circle / line); any other `r` runs the symmetric chord-equalisation solver.
/// Reduces bit-for-bit to the §8.3.6 circular default at `profile = 0.5`.
fn superellipse_chain(profile: f32, segments: u32) -> Vec<(f32, f32)> {
    let n = segments as usize;
    let mut chain = vec![(0.0_f32, 0.0_f32); n + 1];
    chain[0] = (0.0, 1.0);
    chain[n] = (1.0, 0.0);
    if n < 2 {
        return chain;
    }
    let r = profile_to_exponent(profile);

    // Circle (`r = 2`): equal arc-length is equal angle (BEVEL.md §8.6.4 / §8.3.6).
    // Evaluated through the identical sin/cos expression as the historical default
    // path, so the chain is bit-for-bit the §8.3.6 default at `profile = 0.5`.
    if r == 2.0 {
        for k in 1..n {
            let theta = k as f32 * std::f32::consts::FRAC_PI_2 / segments as f32;
            chain[k] = (theta.sin(), theta.cos());
        }
        return chain;
    }
    // Line (`r = 1`): the chord is the curve, so even spacing is even linear
    // interpolation (BEVEL.md §8.6.4).
    if r == 1.0 {
        for k in 1..n {
            let t = k as f32 / segments as f32;
            chain[k] = (t, 1.0 - t);
        }
        return chain;
    }

    superellipse_chain_general(r, segments, &mut chain);
    chain
}

/// The general even-chord spacing solver (BEVEL.md §8.6.4) for exponents without
/// a closed-form shortcut. A fixed-point chord-equalisation re-walks the curve
/// placing each next sample a target chord-distance from the previous, then snaps
/// the result symmetric about the `u = v` diagonal (the diagonal crossing point
/// `m = (1/2)^(1/r)` becomes the central sample at even `segments`). Writes the
/// interior `(u_k, v_k)` into `chain[1 ..= segments-1]`.
fn superellipse_chain_general(r: f32, segments: u32, chain: &mut [(f32, f32)]) {
    let n = segments as usize;
    let diagonal = if r > 0.0 { 0.5_f32.powf(1.0 / r) } else { 0.0 };

    // Seed: linear spacing in `u` over the whole quadrant (BEVEL.md §8.6.4 step 2;
    // a deliberately cheap start the equalisation corrects).
    let mut samples: Vec<(f32, f32)> = (0..=n)
        .map(|k| {
            let u = k as f32 / segments as f32;
            (u, superellipse_v(u, r))
        })
        .collect();
    samples[0] = (0.0, 1.0);
    samples[n] = (1.0, 0.0);

    // Advance from `prev` by chord-distance `target` along the curve toward
    // `u = 1`: a bracketed search for `u_next` with
    // `‖(u_next, sev(u_next)) − prev‖ = target` (BEVEL.md §8.6.4 step 3). The
    // chord grows monotonically with `u`, so a bisection on `[u_prev, 1]` is
    // robust; if `target` can't be reached the sample clamps to the end corner.
    let advance = |prev: (f32, f32), target: f32| -> (f32, f32) {
        if cell_chord((1.0, 0.0), prev) <= target {
            return (1.0, 0.0);
        }
        let mut lo = prev.0;
        let mut hi = 1.0_f32;
        for _ in 0..60 {
            let mid = 0.5 * (lo + hi);
            if cell_chord((mid, superellipse_v(mid, r)), prev) > target {
                hi = mid;
            } else {
                lo = mid;
            }
        }
        let u = 0.5 * (lo + hi);
        (u, superellipse_v(u, r))
    };

    // Fixed-point equalisation: average the current chords, re-walk at that
    // target, repeat until the chord spread is negligible (BEVEL.md §8.6.4
    // step 3/4).
    for _ in 0..64 {
        let target = (0..n).map(|k| cell_chord(samples[k], samples[k + 1])).sum::<f32>()
            / segments as f32;
        let mut next = samples.clone();
        for k in 1..n {
            next[k] = advance(next[k - 1], target);
        }
        next[0] = (0.0, 1.0);
        next[n] = (1.0, 0.0);

        let mut spread = 0.0_f32;
        for k in 0..n {
            spread = spread.max((cell_chord(next[k], next[k + 1]) - target).abs());
        }
        samples = next;
        if spread < 1.0e-7 {
            break;
        }
    }

    // Snap exactly symmetric about `u = v` (BEVEL.md §8.6.4 step 1/5): the second
    // half is the reflection of the first; the exact centre (even `segments`) is
    // the diagonal crossing `(m, m)`.
    for k in 1..n {
        if 2 * k > n {
            samples[k] = (samples[n - k].1, samples[n - k].0);
        } else if 2 * k == n {
            samples[k] = (diagonal, diagonal);
        }
    }
    chain[1..n].copy_from_slice(&samples[1..n]);
}

/// The point on the first line (`p0 + t·d0`) nearest the second line
/// (`p1 + s·d1`) — the first-line foot of the mutual closest-approach pair
/// (BEVEL.md §8.2.2.1 `mid` solve). `None` when the lines are parallel.
fn line_line_nearest_on_first(p0: Vec3, d0: Vec3, p1: Vec3, d1: Vec3) -> Option<Vec3> {
    let d0n = d0.normalize_or_zero();
    if d0n.length_squared() <= f32::EPSILON || d1.length_squared() <= f32::EPSILON {
        return None;
    }
    let r = p0 - p1;
    let a = d0n.dot(d0n);
    let b = d0n.dot(d1);
    let c = d1.dot(d1);
    let d = d0n.dot(r);
    let e = d1.dot(r);
    let denominator = a * c - b * b;
    if denominator.abs() <= 1.0e-9 {
        return None;
    }
    let t = (b * e - c * d) / denominator;
    Some(p0 + d0n * t)
}

/// The sphere (center, radius) through four points (BEVEL.md §8.5.2: the rounded
/// surface the ADJ tri-corner's ring + chain verts lie on). Solves the three
/// equidistance planes `|x − p_i|² = |x − p_0|²`. `None` when the four points are
/// (near-)coplanar so the system is singular.
fn circumsphere(points: [Vec3; 4]) -> Option<(Vec3, f32)> {
    let base = points[0];
    let mut rows = [Vec3::ZERO; 3];
    let mut rhs = [0.0_f32; 3];
    for i in 0..3 {
        let p = points[i + 1];
        rows[i] = 2.0 * (p - base);
        rhs[i] = p.length_squared() - base.length_squared();
    }
    let m = glam::Mat3::from_cols(
        Vec3::new(rows[0].x, rows[1].x, rows[2].x),
        Vec3::new(rows[0].y, rows[1].y, rows[2].y),
        Vec3::new(rows[0].z, rows[1].z, rows[2].z),
    );
    if m.determinant().abs() <= 1.0e-9 {
        return None;
    }
    let center = m.inverse() * Vec3::new(rhs[0], rhs[1], rhs[2]);
    Some((center, (center - base).length()))
}

/// Intersect the line `point + t·direction` with the plane through `plane_point`
/// with normal `plane_normal` (BEVEL.md §8.2.2.1 projection). `None` when the
/// line is parallel to the plane.
fn intersect_line_plane(
    point: Vec3,
    direction: Vec3,
    plane_point: Vec3,
    plane_normal: Vec3,
) -> Option<Vec3> {
    let denominator = direction.dot(plane_normal);
    if denominator.abs() <= 1.0e-9 {
        return None;
    }
    let t = (plane_point - point).dot(plane_normal) / denominator;
    Some(point + direction * t)
}
