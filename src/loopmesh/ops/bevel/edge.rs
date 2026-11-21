//! Edge-mode entry and the isolated single-edge bevel (BEVEL.md §7).

use super::*;

impl LoopMesh {
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
    /// Single segment / symmetric `offset` only. A non-manifold edge, or one whose
    /// endpoint has no terminal close at all (valence < 3, §7.5), is skipped.
    pub(super) fn bevel_edges(&mut self, geom: &[GeomKey], options: &BevelOptions) -> BevelResult {
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

        // The multi-segment orchestrator (BEVEL.md §8.2 / §8.3) drives junctions and
        // terminals alike: a junction-free edge set simply has no fill to build, and
        // resolves to the §8.3.0.1 terminal-to-terminal through-strip. It plans its own
        // edge set, off the manifold subset.
        if options.segments >= 2 {
            self.bevel_edges_cutoff_multisegment(&beveled, options, &mut result);
            return result;
        }

        // §7.5: a terminal endpoint needs two flanking legs to meet its strip verts
        // against, so a valence-2 endpoint has no close at all. Decline such an edge
        // here, whole, or the strip build finds no corner to land on at that end. The plan
        // hands back the junctions (§7.6) of the surviving set along with it.
        let (beveled, junctions) = self.buildable_beveled_edges(&beveled);
        if beveled.is_empty() {
            return result;
        }

        if junctions.is_empty() {
            // No shared endpoint: every edge is isolated, every endpoint terminal
            // — the foundational §7 path, unchanged.
            //
            // §6.6.1: every strip's donor is a flanking face of its own edge. They are
            // registered and sealed up front, before the *first* edge's re-stitch — a
            // per-edge bank would re-capture every registered layer once per edge, and
            // the later edges' donors are still pristine at this point anyway.
            //
            // §6.6.1.4 scopes its override to the odd-`segments` centre quad *and* the lone
            // `segments = 1` quad, so the single quad here is a straddle quad: it goes
            // through the same donor. On a seam-free edge that answers the near flank —
            // `edge_flanking_face`'s own pick — so a contiguous edge is unaffected.
            let mut templates = TemplateBank::new(self, options.segments);
            let strip_templates: Vec<FaceTemplate> = beveled
                .iter()
                .map(|&edge| {
                    let donor = self.strip_quad_donor(edge, options.segments, &templates);
                    templates.register(donor)
                })
                .collect();
            // §6.6.1: a valence-≥4 terminal's endpoint fill (§7.5) is a ring-spanning fill —
            // it inherits from the rep face of the ring's first boundvert, or from
            // §6.6.1.4's tie-break winner when the corner carries a seam leg. At a terminal
            // (`s == 1`) §7.6.1.2's scope bullet anchors that first boundvert one step
            // differently: its sector *closes* at the single beveled spoke, so `B_0`'s own
            // spoke is the beveled edge itself. Either way the
            // donor is one of the faces the endpoint's own consumption kills. Registered for
            // **both** endpoints of every edge, whatever their valence: a donor capture
            // gated on the shape of the corner is the bug class this bank exists to kill.
            // Membership as a set: the donor's candidate gate asks it once per leg of every
            // beveled corner, so a linear scan here would multiply the edge set into the
            // per-corner path.
            let beveled_set: HashSet<EdgeKey> = beveled.iter().copied().collect();
            let terminal_templates: Vec<[FaceTemplate; 2]> = beveled
                .iter()
                .map(|&edge| {
                    self.edge_vert_keys(edge).map(|endpoint| {
                        templates.register(self.anchored_ring_spanning_donor(
                            endpoint,
                            &|leg| beveled_set.contains(&leg),
                            options.affect,
                            options.segments,
                            self.ring_first_boundvert_rep(endpoint, &[edge]),
                            templates.interp_components(),
                        ))
                    })
                })
                .collect();
            templates.seal(self);

            // §8.3.0.4 phases D / E: the wire post-pass and the loose-vertex free are
            // **global**, so both are hoisted out of the per-edge build below. A wire can
            // span two beveled vertices of two *different* edges of this same set (the
            // edges are isolated; the wire between them is not), and a per-edge free would
            // consume the first before the second's ring existed — destroying it.
            let mut endpoints: Vec<VertexKey> = Vec::new();
            for &edge in &beveled {
                for endpoint in self.edge_vert_keys(edge) {
                    if !endpoints.contains(&endpoint) {
                        endpoints.push(endpoint);
                    }
                }
            }
            let wire_spokes = self.snapshot_wire_spokes(&endpoints);

            let mut rings = BevelRings::for_spokes(&wire_spokes);
            for ((&edge, &strip_template), &fan_templates) in beveled
                .iter()
                .zip(&strip_templates)
                .zip(&terminal_templates)
            {
                self.bevel_single_edge(
                    edge,
                    offset,
                    options.profile,
                    options.profile_type.custom().is_some(),
                    material,
                    &templates,
                    strip_template,
                    fan_templates,
                    &mut rings,
                    &mut result,
                );
            }

            self.reattach_wire_spokes(&wire_spokes, &rings);
            for &endpoint in &endpoints {
                if self.vertex_is_loose(endpoint) {
                    self.free_loose_vertex(endpoint);
                }
            }
            return result;
        }

        self.bevel_edges_with_junctions(&beveled, &junctions, options, &mut result);
        result
    }

    /// The flanking original face a beveled edge's strip inherits its whole per-face
    /// block from (BEVEL.md §6.6.1: the strip takes the material of the surface it
    /// bridges — `fL` here). The inherit is stage 1 and runs at every value of
    /// `material`. `None` on a wire edge.
    pub(super) fn edge_flanking_face(&self, edge: EdgeKey) -> Option<FaceKey> {
        self.edge_loop_keys(edge)
            .first()
            .map(|&loop_key| self.loop_face(loop_key))
    }

    /// A beveled edge's two flanking originals (BEVEL.md §6.6.2, the multi-segment
    /// profile-strip row): the **near** flank is the one the `segments = 1` strip quad
    /// inherits from, the **far** flank the one the quads strictly past the strip's centre
    /// line inherit from. A wire edge has neither; a lone-faced edge reports that face as
    /// both, so a caller that cannot reach the far side still lands on an adjacent original
    /// rather than the layer defaults.
    pub(super) fn edge_flanking_faces(&self, edge: EdgeKey) -> (Option<FaceKey>, Option<FaceKey>) {
        let loops = self.edge_loop_keys(edge);
        let near = loops.first().map(|&loop_key| self.loop_face(loop_key));
        let far = loops
            .get(1)
            .map(|&loop_key| self.loop_face(loop_key))
            .or(near);
        (near, far)
    }

    /// The beveled edges the operator can actually build (BEVEL.md §7.5), at **every**
    /// `segments`: every endpoint must be either a junction (§7.6, any valence) or a
    /// terminal whose close [`terminal_close_legs`](Self::terminal_close_legs) can lay out.
    /// A **zero-length** `E` (coincident endpoints) is *not* declined at a **terminal**:
    /// §7.5.4 bevels it like any other edge, its boundverts simply collapsing onto the shared
    /// endpoint in the leg meet. A zero-length `E` keeping a **junction** endpoint is a
    /// different, undistilled case (§7.5.4 is a terminal rule) and is still declined. Dropping
    /// an edge can turn a junction into a terminal, which can drop a further edge, so the set
    /// is shrunk to a fixed point.
    ///
    /// The decline is **whole-edge and up-front**: the close is placement-before-surgery
    /// (§7.5), so an endpoint that only fails half way through would strand the verts the
    /// other end already minted, and a fill recorded over a leg the close declined would
    /// lay a face over geometry the surviving faces still cover. The terminal's admission
    /// is the same predicate at `segments ≥ 2` — the multi-segment chain build wants the
    /// same layout, the same leg directions, and the same clean face corners.
    ///
    /// Returns the surviving edges **and their junctions** — the endpoints where two or more
    /// of them meet (§7.6). The two are one answer: a junction is defined against the *kept*
    /// set (declining an edge can demote a junction back to a terminal, which can decline a
    /// further edge), so the builders' preconditions — "this endpoint is a junction, or its
    /// terminal close was admitted" — only hold if every caller reads the same set the fixed
    /// point settled on. Handing them back together is what keeps a second, drifting
    /// definition from being written at each call site.
    pub(super) fn buildable_beveled_edges(
        &self,
        beveled: &[EdgeKey],
    ) -> (Vec<EdgeKey>, Vec<VertexKey>) {
        let mut buildable: Vec<EdgeKey> = beveled.to_vec();
        loop {
            let mut beveled_degree: HashMap<VertexKey, usize> = HashMap::new();
            for &edge in &buildable {
                for endpoint in self.edge_vert_keys(edge) {
                    *beveled_degree.entry(endpoint).or_default() += 1;
                }
            }
            let kept: Vec<EdgeKey> = buildable
                .iter()
                .copied()
                .filter(|&edge| {
                    let [a, b] = self.edge_vert_keys(edge);
                    // §7.5.4: a zero-length edge bevels at a *terminal* (boundverts collapse
                    // onto the endpoint in the leg meet), but a junction endpoint of one is
                    // undistilled and still declines the whole edge.
                    let is_zero_length = (self.vertex_position(b) - self.vertex_position(a))
                        .length()
                        <= f32::EPSILON;
                    [a, b].into_iter().all(|endpoint| {
                        if beveled_degree[&endpoint] >= 2 {
                            !is_zero_length
                        } else {
                            self.terminal_close_admits(endpoint, edge)
                        }
                    })
                })
                .collect();
            if kept.len() != buildable.len() {
                buildable = kept;
                continue;
            }
            // The degrees were counted over exactly this set, so they are the fixed point's.
            // Collected off the edge list rather than the map, to keep the order the callers
            // walk the junctions in independent of the map's iteration order.
            let mut junctions: Vec<VertexKey> = Vec::new();
            for &edge in &kept {
                for endpoint in self.edge_vert_keys(edge) {
                    if beveled_degree[&endpoint] >= 2 && !junctions.contains(&endpoint) {
                        junctions.push(endpoint);
                    }
                }
            }
            return (kept, junctions);
        }
    }

    /// Bevel one isolated manifold edge (BEVEL.md §7). Places the four strip verts —
    /// and, at a valence-≥4 terminal, that endpoint's on-edge leg verts — against the
    /// pristine local geometry (§7.5 placement-before-surgery), then re-stitches every
    /// incident face, builds the strip quad, and fills each valence-≥4 endpoint (§7.5 /
    /// §8.3.0.2).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn bevel_single_edge(
        &mut self,
        edge: EdgeKey,
        offset: f32,
        profile: f32,
        custom: bool,
        material: i32,
        templates: &TemplateBank,
        strip_template: FaceTemplate,
        fan_templates: [FaceTemplate; 2],
        rings: &mut BevelRings,
        result: &mut BevelResult,
    ) {
        let representative = templates.resolve(strip_template);
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

        // §7.5: both endpoints must *have* a terminal close — two flanking legs for the
        // strip verts to meet against (valence ≥ 3). A zero-length `E` is *not* declined
        // (§7.5.4): it bevels like any other edge, its boundverts collapsing onto the
        // shared endpoint in the leg meet. Decided before the first mint: the close is
        // placement-before-surgery, so bailing after one endpoint were placed would strand
        // its verts.
        // Both endpoints are probed here, ahead of the first mint: `close_terminal_endpoint`
        // is placement-before-surgery, so declining `b` after `a` had been closed would
        // strand `a`'s verts on a mesh that is otherwise left pristine. The probe is exactly
        // the close's own precondition (the planner is pure, and the close reads its faces
        // off `E`'s radial pair).
        if !self.terminal_close_admits(a, edge) || !self.terminal_close_admits(b, edge) {
            return;
        }

        // §7.4 / §7.5: what each incident face's endpoint corner becomes. fL / fR swap
        // 1-for-1 onto their own strip vert; every other face at the endpoint absorbs the
        // corner as the run of the near-corner verts of its two legs, laid down in the
        // face's winding order (which is what lets the leg re-point read its landing vert
        // straight off it). At valence ≥ 4 the close also mints an on-edge vert per
        // non-flanking leg and records the endpoint fill built below.
        let mut substitutions: Vec<FaceSubstitution> = Vec::new();
        let mut terminal_fills: Vec<TerminalFill> = Vec::new();
        let endpoints = [a, b];
        let Some(ends_a) = self.close_terminal_endpoint(
            a,
            edge,
            face_left,
            face_right,
            offset,
            profile,
            custom,
            &[],
            fan_templates[0],
            &mut substitutions,
            &mut terminal_fills,
            rings,
            result,
        ) else {
            return;
        };
        let Some(ends_b) = self.close_terminal_endpoint(
            b,
            edge,
            face_left,
            face_right,
            offset,
            profile,
            custom,
            &[],
            fan_templates[1],
            &mut substitutions,
            &mut terminal_fills,
            rings,
            result,
        ) else {
            return;
        };

        // §7.7.3 / §6.7.1.1(a′): the third face's absorbed corner is its corner run — a
        // single cross edge here — taking the whole per-edge block of that face's
        // winding-outgoing leg, with the seam / sharp bits overridden by the AND of the
        // face's two legs. Read off the pristine mesh; applied once the re-stitch has
        // laid the run down (below). The two endpoints' donor legs sit on *opposite*
        // sides of `E` (§7.7.4), and reading each face's own winding is what delivers
        // that without a side-of-`E` special case.
        let corner_runs = self.snapshot_corner_runs(&substitutions);

        // §7.1 item 4 / §7.4: re-stitch every face incident to a or b. fL/fR swap
        // their endpoint corners 1-for-1 (a→aL/aR, b→bL/bR), staying the same
        // size and adopting the strip's near long edge as their new `E` side. Each
        // third face replaces its single endpoint corner with the endpoint's two
        // strip verts (1-for-2), growing by one side. The four unbeveled legs are
        // re-pointed in place first, so each keeps its per-edge attributes (§6.5
        // item 1).
        let mut incident_faces: Vec<FaceKey> = self.vert_face_keys(a);
        for face in self.vert_face_keys(b) {
            if !incident_faces.contains(&face) {
                incident_faces.push(face);
            }
        }
        let (rebuilt, donors) =
            self.restitch_consumed_endpoints(&incident_faces, &endpoints, &[edge], &substitutions);

        // §7.1 item 2 / §7.8.3: build the strip quad over the four rung ends, paired
        // by **side label** — a rail joins opposite labels, `P(a)—N(b)` and
        // `N(a)—P(b)`, never a flank's own two ring verts. On every mirrored-label
        // input (all consistently-wound meshes, and the decisive-vote mis-wound ones)
        // each rail *is* a rebuilt flank's own pair, so this is the §7.1 quad; under
        // §7.8.3's tied-vote twist it is the twisted quad that cracks away from both.
        // The winding anchor is `face_left`'s own pair at the two endpoints — a fixed
        // flank, read per-face (the sanctioned use of the labels), never "whichever rail
        // the labels put first". On mirrored labels that pair *is* a rail; un-mirrored it
        // is none, and the strip keeps its emission handedness.
        let strip = self.build_strip_quad(
            ends_a.prev_side,
            ends_b.next_side,
            ends_b.prev_side,
            ends_a.next_side,
            [ends_a.on_face(face_left), ends_b.on_face(face_left)],
        );
        self.assign_bevel_face_attributes(
            strip.face,
            material,
            representative,
            BevelNewFaceKind::Strip,
        );
        // §8.10.2 long-edge tagging: the strip quad's two *lengthwise* edges run
        // endpoint-to-endpoint along the beveled edge (aL–bL flanking fL, aR–bR
        // flanking fR); the a-end / b-end chords run *across* it. Only the long
        // edges separate a chamfer face from its lengthwise shading neighbour, so
        // the harden post-pass keys its average-vs-copy choice on them.
        self.record_strip_long_edge(ends_a.prev_side, ends_b.next_side);
        self.record_strip_long_edge(ends_a.next_side, ends_b.prev_side);
        result.faces.push(strip.face);
        result.edges.extend(strip.edges);

        // §7.5 / §8.3.0.2: the endpoint fill of a valence-≥4 terminal — the one face that
        // closes the gap the on-edge leg verts opened across the surviving legs. Built
        // after the strip and the re-stitched leg faces, so each triangle can wind against
        // an already-built neighbour on a shared edge. A valence-3 terminal records no
        // fill: its surviving third face absorbed the corner instead (§7.4).
        for fill in &terminal_fills {
            self.build_terminal_fill(fill, material, templates, result);
        }

        // §7.7.2 / §6.7.1 mechanism (1), UNCONDITIONAL — not gated by `mark_seam` /
        // `mark_sharp`, which are inert for an isolated edge (§6.7.3). `E`'s whole
        // per-edge block rides **both** outer long edges of its strip (aL–bL flanking
        // fL, aR–bR flanking fR): those are the two edges the re-stitched radial faces
        // rebuilt along `E`'s two original sides, so they inherit what `E` carried
        // exactly as a re-pointed survivor keeps its own — crease, bevel weight, custom
        // layers, seam and sharp alike. The two end chords run *across* the strip,
        // descend from no original edge, and stay at the layer default. `E` is still
        // alive here; it dies with its endpoints below.
        for (near_a, near_b) in [
            (ends_a.prev_side, ends_b.next_side),
            (ends_a.next_side, ends_b.prev_side),
        ] {
            if let Some(long_edge) = self.query_edge_exists(near_a, near_b) {
                self.edge_attribs
                    .copy_into(long_edge.as_usize(), edge.as_usize());
            }
        }

        // §7.7.3 / §6.7.1.1(a′) mechanism (1)'s corner-run half, UNCONDITIONAL: the
        // third face's absorbed cross edge takes that face's winding-outgoing leg's
        // block, its flag bits then re-derived as the AND of both of that terminal's
        // legs. Run after the re-stitch, when the run's verts are consecutive on the
        // rebuilt face's boundary and so the cross edge exists.
        self.apply_corner_run_attributes(&corner_runs);

        // §5.2.5 / §5.2.1.1 third bullet: the strip quad owns its four loops. Its
        // per-face block took only the near flank (above), but its per-loop blend is per
        // corner, ordered near/near/far/far (§6.6.2): the two strip verts on `face_left`'s
        // side blend that flank's pre-bevel corner loops, the two on `face_right`'s side the
        // other's — each collapsing to the leg-linear form by the same §10.1.3 short-circuit
        // as the re-stitch below. `face_left` is `E`'s stored-first radial face, so it is the
        // near flank the block already inherited.
        self.blend_strip_quad_loops(
            strip.face,
            face_left,
            face_right,
            [ends_a.on_face(face_left), ends_b.on_face(face_left)],
            [ends_a.on_face(face_right), ends_b.on_face(face_right)],
            &donors,
        );

        // §5.2.1 fact (c) / §5.2.3: the re-stitched originals blend every corner from
        // themselves, pre-bevel, and are the **last** faces to write per-vertex data —
        // so this runs after the strip build above, not inside the re-stitch. For a
        // strip vert this is the rule that matters: it sits on one of the endpoint's
        // *legs* — never on the beveled edge — and that leg is a boundary edge of the
        // very faces rebuilt around it, so the on-edge short-circuit gives
        // `(1 − t)·cd(v) + t·cd(far end of the leg)`. The beveled edge's own far
        // endpoint appears in no strip vert's blend.
        self.blend_restitched_originals(&rebuilt, &donors);

        // §7.1 item 5 / §6.3: both endpoints are now loose, but the free is the caller's
        // (§8.3.0.4 phase E) — it may not begin until the wire post-pass has run for
        // **every** beveled vertex of the whole call, not just this edge's two.
    }

    /// Mint one new bevel vertex **from the beveled corner it belongs to**
    /// (BEVEL.md §5.2.1 step 1).
    ///
    /// Two things happen, and both are normative:
    ///
    /// 1. The whole per-vertex block starts as a **copy of the corner's block**, via
    ///    the kernel's create-from-example facility (BMESH.md §13) — deliberately
    ///    overriding the crate-wide "a minted element takes the layer defaults"
    ///    guarantee. It is the baseline the §5.2.1 step-2 blend overwrites where it
    ///    runs, and the value that stands where it does not.
    /// 2. The **tag is cleared**. The example copy carries every header flag except
    ///    the selection bit — the tag among them — so a vertex minted from a tagged
    ///    corner arrives already tagged. Bevel's teardown sweeps the beveled corners
    ///    (§6.3), and any implementation that recovers that victim set from the tag
    ///    would destroy its own new geometry. The corner's *other* header bits
    ///    (hidden, and whatever a caller set) ride along by design.
    pub(super) fn mint_bevel_vertex(&mut self, position: Vec3, corner: VertexKey) -> VertexKey {
        let vertex = self.create_vertex_from_example(position, corner);
        self.set_vert_tag(vertex, false);
        vertex
    }

    /// Free a beveled endpoint after re-stitch (BEVEL.md §5.1.1 / §6.3). The
    /// re-stitch rebuilds every incident face off the endpoint, leaving only
    /// **loose** incident edges (orphaned legs with no remaining loop) on its disk;
    /// this guards that no edge still on the disk carries a face before draining
    /// the disk and freeing the vertex. A surviving *surface* edge (one still on a
    /// face) means the re-stitch left the endpoint attached to live geometry — the
    /// invariant the old blind `vertex_kill` relied on, now checked rather than
    /// assumed.
    pub(super) fn free_loose_vertex(&mut self, vertex: VertexKey) {
        for edge in self.iter_disk_edges(vertex).collect::<Vec<_>>() {
            assert!(
                self.edge_radial_iter(edge).next().is_none(),
                "free_loose_vertex: endpoint {vertex:?} still on a face via edge {edge:?}",
            );
        }
        self.vertex_kill(vertex);
    }

    /// Whether a beveled endpoint is **loose** (BEVEL.md §5.1.1): every edge on
    /// its disk is radial-empty, so no live face still references it. §5.1.1's
    /// free step assumes every incident face was re-stitched off the endpoint,
    /// which holds at a terminal whose whole fan was beveled but *not* on a
    /// closed solid, where a terminal endpoint can stay incident to unbeveled
    /// faces after the strip build. Those endpoints are kept, not freed — the
    /// condition `free_loose_vertex` asserts, read here as a gate instead.
    pub(super) fn vertex_is_loose(&self, vertex: VertexKey) -> bool {
        self.iter_disk_edges(vertex)
            .all(|edge| self.edge_radial_iter(edge).next().is_none())
    }
}
