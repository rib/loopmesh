//! The single-segment multi-edge junction (BEVEL.md §7.4 / §7.6): boundvert placement,
//! endpoint re-stitch, strips and the patch face.

use super::*;

impl LoopMesh {
    /// Edge bevel where two or more beveled edges meet at one or more **junction**
    /// vertices (BEVEL.md §7.6). Places every near-corner vert (boundverts at
    /// junctions, strip verts at terminal endpoints) and far-corner strip vert
    /// against the *pristine* mesh (§7.5 placement-before-surgery), then re-stitches
    /// every incident face, builds the per-edge strip quads and the per-junction
    /// patch faces, and frees the consumed original endpoints.
    pub(super) fn bevel_edges_with_junctions(
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

        // §6.6.1: register the stage-1 inherit source of each new face against the pristine
        // mesh. A strip's donor is one of its beveled edge's two flanking faces; a
        // junction patch's donor is a face incident to that junction vertex (which the
        // surgery consumes, so by phase 4 it names no adjacent face at all). Each strip's
        // handle rides its edge's slot in this vec; each patch's rides the patch itself.
        // The bank is sealed once phase 1 has registered them all, before the re-stitch.
        // The strip's lone quad at `segments = 1` is §6.6.1.4's straddle quad (the section
        // scopes the override to it by name), so it takes the same donor the multi-segment
        // centre quad does — the near flank on a seam-free edge, unchanged from the
        // positional pick.
        let mut templates = TemplateBank::new(self, options.segments);
        let strip_templates: Vec<FaceTemplate> = beveled_manifold
            .iter()
            .map(|&edge| {
                let donor = self.strip_quad_donor(edge, options.segments, &templates);
                templates.register(donor)
            })
            .collect();

        // A set, not a scan: §6.6.1.4's candidate gate asks this once per leg of every
        // beveled corner, so a linear membership test multiplies the whole edge set into
        // the per-corner path.
        let beveled_set: HashSet<EdgeKey> = beveled_manifold.iter().copied().collect();
        let is_beveled = |edge: EdgeKey| beveled_set.contains(&edge);

        // The original endpoints to consume: every endpoint of a beveled edge.
        let mut endpoints: Vec<VertexKey> = Vec::new();
        for &edge in &beveled_manifold {
            for endpoint in self.edge_vert_keys(edge) {
                if !endpoints.contains(&endpoint) {
                    endpoints.push(endpoint);
                }
            }
        }

        // §7.5 placement-before-surgery, hoisted to the whole edge set: every terminal
        // close is planned here, before the first mint. The close is all-or-nothing, and
        // one taken half way — a terminal declining after its edge's other endpoint, or
        // another edge's terminal, had already been closed — would leave those verts
        // minted and stranded on a mesh no face of this pass reaches. The planner is pure,
        // and the placement below runs against the same pristine disks, so a plan taken
        // here still holds when the placement asks for it.
        let terminal_declined = endpoints.iter().any(|&endpoint| {
            if junctions.contains(&endpoint) {
                return false;
            }
            match beveled_manifold
                .iter()
                .find(|&&edge| self.edge_vert_keys(edge).contains(&endpoint))
            {
                Some(&edge) => !self.terminal_close_admits(endpoint, edge),
                None => false,
            }
        });
        if terminal_declined {
            return;
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

        // §7.2.1 / §8.3.0.4: every face-less spoke of every consumed endpoint, read while
        // the mesh still stands — the beveled edges are radial-empty too by the time the
        // post-pass runs, so the wires are only identifiable now.
        let wire_spokes = self.snapshot_wire_spokes(&endpoints);
        let mut rings = BevelRings::for_spokes(&wire_spokes);

        // §8.9.3.2(a): the inner miter's arc-start split has **no `segments` floor** — the
        // one-segment junction builds the same `B1` / `B3` pair the multi-segment ring does,
        // and only the fill spanning it degenerates to the §7.6.3 patch face. What does
        // *not* narrow at one segment is the parameter gate: `vmesh_method = Cutoff`
        // normalises both miter modes to `Sharp` before any placement runs (§8.3.0.3), at
        // every segment count.
        let inner_miter = (options.miter_inner != BevelMiter::Sharp
            && options.vmesh_method == BevelVmeshMethod::Adj)
            .then_some(Seg1InnerMiter {
                spread: options.spread,
                outer: options.miter_outer,
            });

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
                    options.affect,
                    options.segments,
                    offset,
                    options.loop_slide,
                    inner_miter,
                    &mut templates,
                    &mut corner_vert,
                    &mut substitutions,
                    &mut junction_patches,
                    &mut reconciled,
                    result,
                );
            }
        }
        // §8.3.0.3's attribute note, at `segments = 1`: a two-boundvert junction ring is a
        // weld here too — its "chain" is the single chamfer edge joining `B_0` to `B_1`,
        // minted by the two arriving strips. A **cross** weld (valence 4, the two beveled
        // edges opposite in the disk) donates its crossing legs' per-edge blocks onto that
        // one edge exactly as it does onto the `segments − 1`-interior chain at higher
        // segment counts: the donation belongs to the weld, not to the chain builder.
        // Classified now, while the disks are still pristine; applied once phase 3 has
        // laid the chamfer edge down.
        let mut cross_welds: Vec<CrossWeld> = Vec::new();
        for patch in &junction_patches {
            if patch.boundverts.len() != 2 {
                continue;
            }
            if let Some(cross) = self.cross_weld(
                patch.vertex,
                patch.boundverts[0],
                patch.boundverts[1],
                &[],
                &is_beveled,
            ) {
                cross_welds.push(cross);
            }
        }

        // §7.5: a valence-≥4 terminal records an endpoint fill here and builds it in
        // phase 4b, once its strip and re-stitched leg faces exist to wind against.
        let mut terminal_fills: Vec<TerminalFill> = Vec::new();
        for &endpoint in &endpoints {
            if !junctions.contains(&endpoint) {
                self.place_terminal_strip_verts(
                    endpoint,
                    &beveled_set,
                    options.affect,
                    options.segments,
                    offset,
                    options.profile,
                    options.profile_type.custom().is_some(),
                    &reconciled,
                    &mut templates,
                    &mut corner_vert,
                    &mut substitutions,
                    &mut terminal_fills,
                    &mut rings,
                    result,
                );
            }
        }

        // §8.3.0.4: a junction's ring is its sector boundverts, `B_0` first — the head the
        // post-pass's strict-`<` scan starts from.
        for patch in &junction_patches {
            rings.record(patch.vertex, || patch.boundverts.clone());
        }

        // §6.6.1: every donor is registered; capture them all in one snapshot while the
        // mesh is still pristine. The re-stitch below kills and rebuilds the donors, and a
        // rebuilt face recycles its slot default-stamped.
        templates.seal(self);

        // §6.7.1.1(a′): the corner runs, read off the pristine mesh (see the
        // multi-segment path). At `segments = 1` the only non-empty run is a terminal
        // third face's 1-for-2 absorbed corner, whose two flanking originals are the
        // terminal's unbeveled legs; every junction sector face's run is a single
        // boundvert and flags nothing.
        let corner_runs = self.snapshot_corner_runs(&substitutions);

        // §6.7.5.3 step 1: each junction's whole disk — legs included — as mechanism
        // (2)'s walk reads it. Captured pristine, while the legs still meet the corner.
        // Mechanism (2) is the *only* reader, so a default bevel pays nothing for it.
        let junction_disks = if options.mark_seam || options.mark_sharp {
            self.snapshot_junction_disks(&junctions, &beveled_manifold, &substitutions)
        } else {
            Vec::new()
        };

        // Phase 2: re-stitch every incident face, swapping each consumed endpoint
        // corner for the ordered near-corner vert(s) recorded for that (vertex,
        // face) corner (§7.4 / §7.6.1). A terminal third face swaps its corner for
        // two strip verts (1-for-2); a strip's radial face and every junction
        // sector face swap for one (1-for-1). Every unbeveled leg is re-pointed in
        // place onto its near-corner vert first, so it survives the re-stitch with
        // its per-edge attributes — seam / sharp included (§6.5 item 1).
        let mut incident_faces: Vec<FaceKey> = Vec::new();
        for &endpoint in &endpoints {
            for face in self.vert_face_keys(endpoint) {
                if !incident_faces.contains(&face) {
                    incident_faces.push(face);
                }
            }
        }
        // Each beveled edge's own radial face order, read while `E` still has its loops.
        // This — not the order the corner records happened to be pushed in — nominates the
        // strip's winding anchor, so an edge winds the same way whichever route (isolated
        // or junction) closed its endpoints. The re-stitch below strips `E` of its loops,
        // hence the capture here rather than inside the strip builder.
        let strip_flanks: Vec<Option<[FaceKey; 2]>> = beveled_manifold
            .iter()
            .map(|&edge| {
                let faces: Vec<FaceKey> = self
                    .edge_loop_keys(edge)
                    .iter()
                    .map(|&l| self.loop_face(l))
                    .collect();
                match faces[..] {
                    [left, right] => Some([left, right]),
                    _ => None,
                }
            })
            .collect();

        let (rebuilt, donors) = self.restitch_consumed_endpoints(
            &incident_faces,
            &endpoints,
            &beveled_manifold,
            &substitutions,
        );

        // Phase 3: one strip quad per beveled edge over its four near/far corner
        // verts (§7.1 item 2). §6.6: each new strip inherits its donor's whole block,
        // then takes the `material` stamp over the slot if >= 0.
        for ((&edge, &representative), &flanks) in beveled_manifold
            .iter()
            .zip(&strip_templates)
            .zip(&strip_flanks)
        {
            self.build_junction_strip(
                edge,
                flanks,
                &corner_vert,
                material,
                templates.resolve(representative),
                result,
            );
        }

        // Phase 3b (§8.3.0.3 attribute note): the cross weld's donation, on the single
        // chamfer edge its `segments = 1` chain degenerates to.
        for cross in &cross_welds {
            self.apply_cross_weld_attributes(cross);
        }

        // Phase 4: cap each junction whose **ring size** is `n ≥ 3` with one flat patch
        // face over its boundvert ring (§7.6.3). The gate is the ring, never the
        // beveled-edge count: a firing inner miter parts the two, and a two-beveled-edge
        // junction whose sectors both split gains a patch face it had none of when sharp
        // (§8.9.3.2(f)). §6.6: each new patch inherits a junction-adjacent original's
        // whole block, then takes the stamp over the slot.
        for patch in &junction_patches {
            if patch.boundverts.len() >= 3 {
                self.build_patch_face(
                    &patch.boundverts,
                    material,
                    templates.resolve(patch.representative),
                    result,
                );
            }
        }

        // Phase 4b (§7.5 / §8.3.0.2): each valence-≥4 terminal's endpoint fill — the one
        // face closing the gap its on-edge leg verts opened across the surviving legs.
        // Gated on "terminal", never on the fill method (§8.3.0.2), and built after the
        // strips and the re-stitched leg faces so each triangle winds against an
        // already-built neighbour. A valence-3 terminal has no fill: its surviving third
        // face absorbed the corner instead (§7.4).
        for fill in &terminal_fills {
            self.build_terminal_fill(fill, material, &templates, result);
        }

        // Phase 4.25 (§7.7.2 / §6.7.1 mechanism (1), UNCONDITIONAL — not gated by the
        // parameters): each beveled edge's two strip long edges descend from that
        // edge's re-stitched radial faces, so they inherit its whole per-edge block —
        // crease, bevel weight, custom layers, seam and sharp — exactly as a re-pointed
        // survivor keeps its own (§6.7.1 item 1 / §6.7.3). The perpendicular near edges
        // (the patch ring segments) are *not* descendants of any single edge — they
        // are the parameter-gated bridging segments of phase 4.5 below.
        self.inherit_strip_long_edge_attributes(&beveled_manifold, &corner_vert);

        // Phase 4.25b (§7.7.3 / §6.7.1.1(a′) mechanism (1)'s corner-run half,
        // UNCONDITIONAL): the same per-face copy the multi-segment path runs — the run
        // takes its face's winding-outgoing original's block, its two flag bits then
        // re-derived as the contiguity AND. Here it bites only on a terminal's absorbed
        // corner, whose donor is one of the terminal's two unbeveled legs.
        self.apply_corner_run_attributes(&corner_runs);

        // Phase 4.3 (§5.2.1 fact (c) / §5.2.3): the re-stitched originals blend every
        // corner from their own pre-bevel polygon, and they are the **last** faces to
        // write per-vertex data — hence after every fill above.
        self.blend_restitched_originals(&rebuilt, &donors);

        // Phase 4.5 (§6.7.4): the `mark_seam` / `mark_sharp` junction bridging.
        // Mechanism (2): bridge the seam / sharp flag across each junction's patch
        // ring segments so the flag runs continuously around the junction, joining
        // the strip long edges of consecutive seamed beveled edges. The `segments = 1`
        // ring is one direct patch segment per sector, so the walk's chain steps and
        // its beveled-edge count are always in registration here.
        if options.mark_seam || options.mark_sharp {
            for patch in &junction_patches {
                let Some(disk) = junction_disks
                    .iter()
                    .find(|disk| disk.vertex == patch.vertex)
                else {
                    continue;
                };
                let ring = patch.ring_chains();
                self.bridge_junction_flags(&ring, disk, options.mark_seam, options.mark_sharp);
            }
        }

        // Phase D (§8.3.0.4): every ring is final, and no endpoint has been freed yet —
        // the one window in which a wire spoke can be re-pointed onto the ring of the
        // vertex it hung off (and, when its far end was beveled too, onto that ring as
        // well). Runs before phase 5 below, which would otherwise take the wire with it.
        self.reattach_wire_spokes(&wire_spokes, &rings);

        // Phase 5 (§8.3.0.4 phase E): every consumed endpoint is now loose. Free each
        // (§5.1.1).
        for &endpoint in &endpoints {
            self.free_loose_vertex(endpoint);
        }
    }

    /// Push every edge of `face` into the result's edge set (de-duplicated).
    pub(super) fn collect_face_edges(&self, face: FaceKey, result: &mut BevelResult) {
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
    ///
    /// With `inner_miter` set, every sector §8.9.3.2(a) admits is **split** instead: its
    /// meet-point boundvert is retained as the arc start `B1` and a second boundvert —
    /// the arc end `B3` — is minted on the trailing flank, the two then pushed `spread`
    /// apart along their own flanking beveled edges (§8.9.3 / §8.9.4). At one segment the
    /// arc between them carries no interior sample, so the ring gains exactly one vertex
    /// and one chord edge per split sector and the cap keeps its face count (§8.9.3.2(d)).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn place_junction_boundverts(
        &mut self,
        vertex: VertexKey,
        is_beveled: &dyn Fn(EdgeKey) -> bool,
        affect: BevelAffect,
        segments: u32,
        offset: f32,
        loop_slide: bool,
        inner_miter: Option<Seg1InnerMiter>,
        templates: &mut TemplateBank,
        corner_vert: &mut Vec<CornerVert>,
        substitutions: &mut Vec<FaceSubstitution>,
        junction_patches: &mut Vec<JunctionPatch>,
        reconciled: &mut Vec<ReconciledOffset>,
        result: &mut BevelResult,
    ) {
        // Rotational corner ring: each incident face contributes the pair of its
        // two edges meeting at `vertex`. Chain them edge-to-edge into the disk
        // walk order around the vertex.
        let corners = self.vertex_bevel_corner_ring(vertex, is_beveled);

        // Split the ring into sectors at beveled edges. A sector runs from one
        // beveled edge, through any unbeveled edges and their faces, to the next
        // beveled edge. Each sector yields one boundvert (§7.6.1).
        let sectors = sector_split(&corners, is_beveled);

        let vertex_position = self.vertex_position(vertex);

        // §8.9.3.2(a): classify the miter read-only, before the first mint, so a junction
        // the miter declines is placed exactly as it was without one.
        let arc_split =
            self.seg1_inner_miter_splits(vertex, vertex_position, &sectors, inner_miter);
        let spread = inner_miter.map_or(0.0, |miter| miter.spread);

        let mut boundverts: Vec<VertexKey> = Vec::with_capacity(sectors.len());
        // Per **ring position**, the beveled edge opening the sector piece there. A split
        // sector's arc end takes `None`: the ring edge arriving at it is the miter chord,
        // which crosses no beveled edge at all (§8.9.3.2(c), the §6.7.5.4 case (6)
        // desynchronisation).
        let mut opening_edges: Vec<Option<EdgeKey>> = Vec::with_capacity(sectors.len());
        // Per **sector**, the same edge with no miter slots — what the §6.6.1 ring-anchor
        // donor is stated on.
        let mut sector_opening_edges: Vec<EdgeKey> = Vec::with_capacity(sectors.len());

        for (sector_index, sector) in sectors.iter().enumerate() {
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
            // `bounding_edges[0]` is the beveled edge that opens this sector
            // (sector_split records the opening edge first) — `g_k` for `B_k`.
            sector_opening_edges.push(sector.bounding_edges[0]);

            let Some((dir_open, dir_close)) = arc_split[sector_index] else {
                // §5.2.1 step 1: minted from the junction corner it rings.
                let boundvert = self.mint_bevel_vertex(boundvert_position, vertex);
                result.verts.push(boundvert);
                boundverts.push(boundvert);
                opening_edges.push(Some(sector.bounding_edges[0]));

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
                            side_label: SideLabel::Unlabelled,
                        });
                    }
                }
                continue;
            };

            // §8.9.3 / §8.9.4: the retained meet-point boundvert becomes the arc start,
            // the fresh one on the trailing flank the arc end, and each is pushed `spread`
            // out along its **own** flanking beveled edge from the meet. The slide is
            // `segments`-independent (§8.9.3.2(a)), so these land where the multi-segment
            // ring's arc ends land. Both are minted from the corner exactly as the sharp
            // boundvert is.
            let arc_start = self.mint_bevel_vertex(boundvert_position + dir_open * spread, vertex);
            let arc_end = self.mint_bevel_vertex(boundvert_position + dir_close * spread, vertex);
            result.verts.push(arc_start);
            result.verts.push(arc_end);
            boundverts.push(arc_start);
            boundverts.push(arc_end);
            opening_edges.push(Some(sector.bounding_edges[0]));
            opening_edges.push(None);

            // §8.9.3.1: the leading flank stays with the arc start and the trailing flank
            // is handed to the arc end, so each flanking strip meets the ring on its own
            // side of the chord.
            let (leading_edge, leading_face) = sector.corners[0];
            let &(trailing_edge, trailing_face) = sector
                .corners
                .last()
                .expect("a split sector is bounded by two beveled edges, hence two corners");
            corner_vert.push(CornerVert {
                edge: leading_edge,
                vertex,
                face: leading_face,
                near_vert: arc_start,
                side_label: SideLabel::Unlabelled,
            });
            corner_vert.push(CornerVert {
                edge: trailing_edge,
                vertex,
                face: trailing_face,
                near_vert: arc_end,
                side_label: SideLabel::Unlabelled,
            });

            // §8.9.3.2(d): the split is a vertex split — the sector's own face simply
            // gains a corner, laid down as the pair in the face's own winding. A sector
            // that absorbed unbeveled legs (§7.6.1) fans several faces onto the pair by
            // §8.9.2's profile-sample index rule, which at one segment resolves every leg
            // to one arc end or the other (there is no interior sample to land on).
            let mut sector_faces: Vec<FaceKey> = Vec::with_capacity(sector.corners.len());
            for &(_, face) in &sector.corners {
                if !sector_faces.contains(&face) {
                    sector_faces.push(face);
                }
            }
            match sector_faces[..] {
                [face] => {
                    let run = match self.face_corner_edges(face, vertex) {
                        Some((incoming, _)) if incoming == leading_edge => vec![arc_start, arc_end],
                        _ => vec![arc_end, arc_start],
                    };
                    set_face_substitution(substitutions, vertex, face, run);
                }
                _ => self.restitch_multiface_arc_sector(
                    vertex,
                    sector,
                    &[arc_start, arc_end],
                    1,
                    substitutions,
                ),
            }
        }

        // §6.6.1: the patch is a ring-spanning fill, so its inherit source is the ring's
        // first boundvert's rep — `prev_face` of the spoke that *closes* `B_0`'s sector,
        // read off this junction's own §7.6.1.2-anchored ring — or §6.6.1.4's tie-break
        // winner over the beveled legs when the junction carries a seam leg. Registered
        // while the junction still has its faces — the re-stitch consumes the vertex.
        //
        // §8.9.3.2(i): a split first sector does **not** change which boundvert is first —
        // it is the same object — but it clears that boundvert's own beveled edge, so the
        // rep resolves through the `first_edge` arm (`prev_face(g_0)`) instead of the
        // own-beveled-edge arm (`prev_face(g_1)`). Read here, after the split: caching it
        // while the ring is built reproduces the sharp answer under a firing miter.
        let first_boundvert_rep = match arc_split.first().copied().flatten() {
            Some(_) => self.boundvert_representative(vertex, sector_opening_edges[0]),
            None => self.ring_first_boundvert_rep(vertex, &sector_opening_edges),
        };
        let representative = templates.register(self.anchored_ring_spanning_donor(
            vertex,
            is_beveled,
            affect,
            segments,
            first_boundvert_rep,
            templates.interp_components(),
        ));

        junction_patches.push(JunctionPatch {
            vertex,
            boundverts,
            opening_edges,
            representative,
        });
    }

    /// Re-stitch every face incident to a consumed endpoint (BEVEL.md §7.4 /
    /// §7.6.1), preserving each surviving unbeveled leg's identity (§6.5 item 1).
    ///
    /// Three steps that cannot interleave:
    ///
    /// 1. Read each incident face's rebuilt vertex cycle off the *current* mesh —
    ///    every consumed-endpoint corner replaced by the ordered near-corner vert(s)
    ///    recorded for that (vertex, face) corner: one vert for a strip's radial face
    ///    or a miter-free junction sector face (1-for-1), a whole run for a terminal's
    ///    absorbing face or a mitered sector (1-for-many). A corner with no recorded
    ///    substitution is left as-is, and a face with no substituted corner is not
    ///    rebuilt at all.
    /// 2. Re-point every unbeveled **leg** in place onto the near-corner vert it lands
    ///    on. The leg then *is* the original edge — same `EdgeKey`, same `edge_attribs`
    ///    slot — so its per-edge customdata and its seam / sharp bits survive the bevel
    ///    (§6.5 item 1 / §6.7.1 item 1), and step 3 reuses it rather than minting a
    ///    blank edge in its place.
    /// 3. Kill each face and re-create it over its rebuilt cycle, carrying the face's
    ///    own attributes across.
    ///
    /// Step 2 patches the leg's radial loops as well as its disk, so the cycles of
    /// step 1 must already be in hand when it runs.
    ///
    /// Returns the rebuilt originals paired with the pre-bevel donor bank they blend
    /// from. The blend itself is the caller's last act (§5.2.1 fact (c): the
    /// re-stitched originals are the last faces to write per-vertex data), so it
    /// deliberately does **not** run here — the caller's strip / fill builds come in
    /// between.
    pub(super) fn restitch_consumed_endpoints(
        &mut self,
        incident_faces: &[FaceKey],
        endpoints: &[VertexKey],
        beveled: &[EdgeKey],
        substitutions: &[FaceSubstitution],
    ) -> (Vec<RebuiltOriginal>, DonorBank) {
        let mut rebuilds: Vec<(FaceKey, Vec<VertexKey>)> = Vec::with_capacity(incident_faces.len());
        for &face in incident_faces {
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

            if changed {
                rebuilds.push((face, new_cycle));
            }
        }

        // §5.2.1 fact (a): capture each rebuilt original as a **pre-bevel** donor
        // polygon before the leg re-point rewrites its corners. Its own corners
        // (the consumed endpoint among them, at its original position and with its
        // original block) are what the per-corner blend below resamples.
        let donor_faces: Vec<FaceKey> = rebuilds.iter().map(|(face, _)| *face).collect();
        let bank = self.capture_donor_bank(&donor_faces);

        self.repoint_unbeveled_legs(endpoints, beveled, substitutions, &rebuilds);

        // Capture every rebuilt face's block in one snapshot, before any kill:
        // a rebuilt face may recycle a killed face's slab slot, and a minted
        // element's attribute block is default-stamped, so the dead face's slot
        // is no longer a source once the loop below is under way.
        let carried_slots: Vec<usize> = rebuilds.iter().map(|(f, _)| f.as_usize()).collect();
        let carried = self.face_attribs.snapshot_slots(&carried_slots);

        let mut rebuilt: Vec<RebuiltOriginal> = Vec::with_capacity(rebuilds.len());
        for (index, (face, new_cycle)) in rebuilds.into_iter().enumerate() {
            let donor = bank.index_of(face);
            debug_assert!(
                donor.is_some() || bank.is_empty(),
                "a re-stitched original was not captured as a donor pre-bevel",
            );
            self.face_kill(face);
            let new_face = self.face_create_verts(&new_cycle, FaceCreateFlags::NONE);
            self.face_attribs
                .restore_slot(new_face.as_usize(), &carried, index);
            rebuilt.push(RebuiltOriginal {
                face: new_face,
                donor,
            });
        }

        (rebuilt, bank)
    }

    /// Re-point each consumed endpoint's unbeveled legs onto their near-corner verts
    /// (BEVEL.md §6.5 item 1 / §6.7.1 item 1) — the edge-bevel analogue of the vertex
    /// bevel's [`LoopMesh::repoint_incident_edges`].
    ///
    /// A leg's landing vert is read straight off the substitution runs, per the
    /// [`FaceSubstitution`] run-order contract: a face lays its corner run down in its
    /// own winding order, so the corner's *incoming* edge meets the run's first vert
    /// and its *outgoing* edge the run's last. A leg is therefore re-pointed only when
    /// **every** face on its radial fan has a recorded run and all of them agree on the
    /// landing vert — the manifold case. Anything else (a face the placement pass
    /// declined to substitute, a leg whose two sides disagree) declines the re-point and
    /// leaves the leg to be rebuilt blank, which is no worse than not running this pass
    /// at all.
    ///
    /// That "every radial face has a run" condition is doing two jobs. Besides pinning
    /// the landing vert, it is the **topological safety property**: `rebuilds` holds
    /// exactly the faces with a run, so re-pointing only fully-covered legs guarantees
    /// every face whose loops this pass mutates is killed and re-created straight after
    /// (see [`FaceSubstitution`]'s safety invariant). The `debug_assert!` below is that
    /// property, checked.
    ///
    /// The absorbed-leg case falls out of the same read: several legs inside one
    /// junction sector all land on that sector's single boundvert, so they are all
    /// re-pointed onto it and keep their own attributes.
    ///
    /// Scope: a leg whose fan is only *partly* covered keeps the consumed endpoint and
    /// is rebuilt blank, losing its per-edge attributes. Every path that consumes an
    /// endpoint covers that endpoint's whole fan — a terminal close places a near-corner
    /// vert on **every** leg of its disk, whatever the valence (the two flanking legs get
    /// the strip verts, the rest an on-edge vert each, §7.5) — so a partial cover only
    /// arises on a malformed corner the placement pass declined outright.
    pub(super) fn repoint_unbeveled_legs(
        &mut self,
        endpoints: &[VertexKey],
        beveled: &[EdgeKey],
        substitutions: &[FaceSubstitution],
        rebuilds: &[(FaceKey, Vec<VertexKey>)],
    ) {
        let mut repoints: Vec<(EdgeKey, VertexKey, VertexKey)> = Vec::new();

        for &vertex in endpoints {
            for leg in self.iter_disk_edges(vertex) {
                if beveled.contains(&leg) {
                    continue;
                }
                if let Some(vert) = self.leg_ring_vertex(vertex, leg, substitutions) {
                    debug_assert!(
                        self.edge_radial_iter(leg)
                            .map(|loop_key| self.loop_face(loop_key))
                            .all(|face| rebuilds.iter().any(|(rebuilt, _)| *rebuilt == face)),
                        "re-pointing leg {leg:?} would leave a live face with a rewritten corner loop",
                    );
                    if !self.edge_vert_keys(leg).contains(&vert) {
                        repoints.push((leg, vertex, vert));
                    }
                }
            }
        }

        for (leg, vertex, landing) in repoints {
            self.edge_replace_vertex_raw(leg, vertex, landing);
        }
    }

    /// Build one beveled edge's strip quad over its four near/far corner verts
    /// (BEVEL.md §7.1 item 2). The four corners are the edge's two near corners
    /// (one per adjacent face, at each endpoint) — at a junction these are shared
    /// boundverts, at a terminal they are the endpoint's fresh strip verts. Wound
    /// opposite the left face's traversal so each strip edge stays radial-2 (§5.1).
    pub(super) fn build_junction_strip(
        &mut self,
        edge: EdgeKey,
        flanks: Option<[FaceKey; 2]>,
        corner_vert: &[CornerVert],
        material: i32,
        representative: FaceTemplateRef<'_>,
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
        // The left/right nomination comes from `E`'s own pre-stitch radial order, never
        // from the order the corner records happened to be pushed in: the winding anchor
        // below must not depend on the sector/push route, or the same edge would wind
        // opposite ways depending on how its endpoints were closed.
        let Some([face_left, face_right]) = flanks else {
            return;
        };
        debug_assert!(
            radial_faces.contains(&face_left) && radial_faces.contains(&face_right),
            "the recorded corner faces are the beveled edge's own radial pair",
        );

        // A corner the placement pass declined (a degenerate endpoint) has no near vert,
        // so there is no strip to build here — decline rather than assert.
        let (Some(a_left), Some(a_right), Some(b_left), Some(b_right)) = (
            near_corner(corner_vert, edge, a, face_left),
            near_corner(corner_vert, edge, a, face_right),
            near_corner(corner_vert, edge, b, face_left),
            near_corner(corner_vert, edge, b, face_right),
        ) else {
            return;
        };

        // §7.8.3: pair the rails by side label when both endpoints carry one. On mirrored
        // labels this is the same cycle the face-keyed order below produces; under the
        // tied-vote twist it is the twisted quad, and the face-keyed order is the guide's
        // explicitly named wrong route (the like-label pairing).
        if let Some((prev_a, next_a)) = labelled_rung_ends(corner_vert, edge, a) {
            if let Some((prev_b, next_b)) = labelled_rung_ends(corner_vert, edge, b) {
                let ring = [prev_a, next_b, prev_b, next_a];
                if ring_has_repeat(&ring) {
                    self.build_collapsed_strip(&ring, material, representative, result);
                    return;
                }
                // The winding anchor is a *fixed* flank's own pair — read per-face, the
                // sanctioned use of the labels — so the emitted handedness never varies
                // with which rail the disk order happened to label first.
                let strip = self.build_strip_quad(prev_a, next_b, prev_b, next_a, [a_left, b_left]);
                self.assign_bevel_face_attributes(
                    strip.face,
                    material,
                    representative,
                    BevelNewFaceKind::Strip,
                );
                result.faces.push(strip.face);
                for strip_edge in strip.edges {
                    if !result.edges.contains(&strip_edge) {
                        result.edges.push(strip_edge);
                    }
                }
                return;
            }
        }

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
        self.assign_bevel_face_attributes(face, material, representative, BevelNewFaceKind::Strip);
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
    pub(super) fn build_collapsed_strip(
        &mut self,
        ring: &[VertexKey; 4],
        material: i32,
        representative: FaceTemplateRef<'_>,
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
        self.assign_bevel_face_attributes(face, material, representative, BevelNewFaceKind::Strip);
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
    pub(super) fn build_patch_face(
        &mut self,
        boundverts: &[VertexKey],
        material: i32,
        representative: FaceTemplateRef<'_>,
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
        self.assign_bevel_face_attributes(face, material, representative, BevelNewFaceKind::Cap);
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
}

/// One (beveled edge, endpoint, adjacent face) corner of an edge-bevel junction
/// group and the **near-corner vert** its strip lands on there (BEVEL.md §7.6.1):
/// a fresh strip vert at a terminal endpoint, or a shared sector boundvert at a
/// junction.
pub(super) struct CornerVert {
    pub(super) edge: EdgeKey,
    pub(super) vertex: VertexKey,
    pub(super) face: FaceKey,
    pub(super) near_vert: VertexKey,
    /// Which of `vertex`'s two BEVEL.md §7.8.3 side labels `near_vert` carries:
    /// `PrevFirst` for `P(vertex)`, `NextFirst` for `N(vertex)`.
    ///
    /// `Unlabelled` at a junction endpoint — not because no label exists there (§7.8.6's
    /// junction bullet scopes out the *collapse* and the ring it produces, explicitly not
    /// the `P` / `N` labels, which §8.3.0 reads at every beveled endpoint of any valence)
    /// but because the junction ring builders here do not record which of `vertex`'s two
    /// rotational slots each boundvert filled. Where that leaves the rail pairing naming
    /// the side by flank face instead, it is on §7.8.3's explicitly wrong route; see
    /// [`labelled_strip_rails`].
    pub(super) side_label: SideLabel,
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
pub(super) struct ReconciledOffset {
    pub(super) edge: EdgeKey,
    pub(super) face: FaceKey,
    pub(super) offset: f32,
}

/// The §7.6.4-reconciled offset for the `(edge, face)` side, or `spec` when no firing
/// sector constrained it (BEVEL.md §8.8.4).
pub(super) fn reconciled_offset(
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
/// patch face when the **ring size** `n ≥ 3` (§7.6.3 — keyed on `n`, never on the
/// beveled-edge count `s`, which a firing miter parts company with). `vertex` is the
/// junction the patch is built around, used to look up its stage-1 inherit donor
/// (§6.6.1).
pub(super) struct JunctionPatch {
    pub(super) vertex: VertexKey,
    pub(super) boundverts: Vec<VertexKey>,
    /// Per boundvert, the beveled edge that *opens* that sector in disk order
    /// (`g_k` for boundvert `B_k`, BEVEL.md §7.6.1). The patch edge between
    /// consecutive boundverts `(B_(k-1), B_k)` is crossed by `opening_edges[k]`, so
    /// this maps each junction bridging segment to the beveled edge it bridges,
    /// for the §6.7.4 seam / sharp bridging.
    ///
    /// `None` on a split sector's **arc end** (§8.9.3.2): the ring edge arriving there is
    /// the miter chord, which crosses no beveled edge — the §6.7.5.4 case (6)
    /// desynchronisation, in its one-segment form.
    pub(super) opening_edges: Vec<Option<EdgeKey>>,
    /// The stage-1 inherit donor for this junction's patch face (BEVEL.md §6.6.1, read
    /// at every value of `material`): a face incident to the junction vertex, registered
    /// while the ring is placed — the vertex is consumed by the re-stitch, so by patch
    /// time it names no adjacent face at all.
    pub(super) representative: FaceTemplate,
}

impl JunctionPatch {
    /// The junction's boundvert ring as the §6.7.4 walk's chain sequence. At
    /// `segments = 1` a "chain" is the single direct patch segment `B_i → B_(i+1)` — the
    /// degenerate case of the §6.7.5.1 profile chain, so one walk serves both regimes.
    /// A miter-free sector's segment is crossed by the beveled edge opening the next
    /// sector; a split sector contributes one further segment — the miter chord — that no
    /// beveled edge crosses, so a span stepping through it comes up one chain short at its
    /// far end (§6.7.5.4 case (6)).
    pub(super) fn ring_chains(&self) -> Vec<RingChain> {
        // One opening-edge slot per ring position (§7.6.1, plus §8.9.3.2's arc ends) —
        // this is what makes the `opening_edges` index below in-range everywhere.
        debug_assert_eq!(self.boundverts.len(), self.opening_edges.len());
        let count = self.boundverts.len();
        (0..count)
            .map(|i| RingChain {
                cross_edge: self.opening_edges[(i + 1) % count],
                samples: vec![self.boundverts[i], self.boundverts[(i + 1) % count]],
            })
            .collect()
    }
}

/// What a consumed endpoint corner becomes on one specific face (BEVEL.md §7.4 /
/// §7.6.1): the ordered near-corner vert(s) the re-stitch substitutes for the
/// single corner vertex `vertex` on `face`. One vert for a 1-for-1 swap (a strip's
/// radial face or a junction sector face), two or more for an absorbed corner run
/// (a terminal third face, a junction's multi-face sector, a `segments ≥ 2` chain).
///
/// **Run-order contract — every producer must honour it.** A run is recorded in
/// `face`'s own winding order: the corner's **incoming** edge (the one arriving at
/// `vertex` in the face's loop traversal) meets `verts.first()`, and its **outgoing**
/// edge meets `verts.last()`. `repoint_unbeveled_legs` reads each leg's landing vert
/// straight off that ordering, so a producer that lays a run down backwards silently
/// re-points the legs onto the wrong near-corner verts.
///
/// **Safety invariant — a recorded run is a promise the face gets rebuilt.** A face
/// with a run always substitutes at least one corner, so `restitch_consumed_endpoints`
/// always sees `changed` for it and rebuilds it. `repoint_unbeveled_legs` relies on
/// that: it re-points a leg's disk *and radial loops* onto the landing vert, which
/// deliberately leaves each corner loop inconsistent (the loop's `vertex` is rewritten
/// while its `edge` still names the consumed endpoint) — harmless only because
/// `face_kill` keys on `loop.edge` and every such face is killed moments later. A
/// producer that recorded a run for a face which then failed the `changed` test would
/// strand a live, corrupted face; the `debug_assert!` in `repoint_unbeveled_legs`
/// guards exactly that. A face a producer *declines* to substitute is fine — the
/// decline propagates to the legs, which then keep the consumed endpoint too.
pub(super) struct FaceSubstitution {
    pub(super) vertex: VertexKey,
    pub(super) face: FaceKey,
    pub(super) verts: Vec<VertexKey>,
}

/// Record (or overwrite) the ordered substitution for `(vertex, face)`.
pub(super) fn set_face_substitution(
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
pub(super) fn face_substitution(
    substitutions: &[FaceSubstitution],
    vertex: VertexKey,
    face: FaceKey,
) -> Option<&[VertexKey]> {
    substitutions
        .iter()
        .find(|s| s.vertex == vertex && s.face == face)
        .map(|s| s.verts.as_slice())
}

/// The near-corner vert recorded for the (edge, vertex, face) corner. `None` when the
/// placement pass **declined** that corner (a degenerate endpoint it could not close):
/// the strip that would have landed on it is then declined too, rather than the corner
/// being an invariant the build asserts (rubric 5 — a degenerate input, not a broken
/// invariant).
pub(super) fn near_corner(
    corner_vert: &[CornerVert],
    edge: EdgeKey,
    vertex: VertexKey,
    face: FaceKey,
) -> Option<VertexKey> {
    corner_vert
        .iter()
        .find(|c| c.edge == edge && c.vertex == vertex && c.face == face)
        .map(|c| c.near_vert)
}

/// True if `ring` contains the same vertex twice (a collapsed strip side, §7.6.3).
pub(super) fn ring_has_repeat(ring: &[VertexKey]) -> bool {
    for i in 0..ring.len() {
        for j in (i + 1)..ring.len() {
            if ring[i] == ring[j] {
                return true;
            }
        }
    }
    false
}

/// The §7.6.4.2 width-reconciliation spec-match weight, *squared*. The two residual
/// families are stacked with width-match coefficient `1.0` and spec-match
/// coefficient `w = 0.2`; the normal equations square the coefficients, so the spec
/// family enters the solve with effective weight `w² = 0.04` — width matching is
/// weighted 25× more heavily than spec matching. This finite ratio sets the
/// terminal-end damping `1 / (1 + w²) = 1 / 1.04` of §7.6.4.3 and must not be a
/// "hard width / soft spec" limit (which would drop the `1.04`).
pub(super) const WIDTH_SPEC_WEIGHT_SQ: f32 = 0.04;
