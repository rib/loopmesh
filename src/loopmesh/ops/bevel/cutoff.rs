//! The multi-segment edge bevel and its cutoff fill (BEVEL.md §8.2 / §8.3).

use super::*;

impl LoopMesh {
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
    pub(super) fn bevel_edges_cutoff_multisegment(
        &mut self,
        beveled: &[EdgeKey],
        options: &BevelOptions,
        result: &mut BevelResult,
    ) {
        let offset = options.offset;
        let material = options.material;
        let segments = options.segments;
        let profile = options.profile;
        // §8.6.3.1(4).4: the custom curve's `target` and `doubled` chains, materialised once
        // for the whole operation. Every downstream read strides one of them.
        let custom_chains = options
            .profile_type
            .custom()
            .map(|curve| CustomChains::new(curve, segments));
        let custom = custom_chains.as_ref();

        let mut manifold: Vec<EdgeKey> = Vec::new();
        for &edge in beveled {
            if self.edge_radial_iter(edge).count() == 2 {
                manifold.push(edge);
            }
        }

        // §7.5 placement-before-surgery, hoisted to the whole edge set — the same plan the
        // `segments = 1` path takes, for the same reason. `build_terminal_chains` is
        // all-or-nothing, so a terminal declining *after* another endpoint (this edge's own
        // far end, or another edge's) had already minted its chain would leave those verts
        // stranded on a mesh no face of this pass reaches. The planner is pure and reads the
        // same pristine disks phase 1 does, so a plan taken here still holds when phase 1
        // asks for it — which is why the builders below `expect` rather than skip.
        // The plan's junctions (§7.6) are the ones the surviving set defines — declining an
        // edge can demote a junction back to a terminal — and every `expect` below rests on
        // reading exactly that set, not a second copy of the predicate.
        let (beveled_manifold, junctions) = self.buildable_beveled_edges(&manifold);
        if beveled_manifold.is_empty() {
            return;
        }
        // A set, not a scan: §6.6.1.4's candidate gate asks this once per leg of every
        // beveled corner, so a linear membership test multiplies the whole edge set into
        // the per-corner path.
        let beveled_set: HashSet<EdgeKey> = beveled_manifold.iter().copied().collect();
        let is_beveled = |edge: EdgeKey| beveled_set.contains(&edge);

        // §6.6.1 / §6.6.2: a multi-segment strip has **two** stage-1 donors, not one — its
        // beveled edge's near and far flanking faces, split about the strip's centre line.
        // Both are registered here, against the pristine mesh; the split's orientation is
        // resolved once phase 1 has built the chains. Phase 1 registers the remaining donors
        // (junction fills, miter rings, terminal fans) as it walks the pristine endpoints;
        // the bank is sealed once, before the re-stitch.
        let mut templates = TemplateBank::new(self, options.segments);
        let mut strip_donors: Vec<StripDonors> = beveled_manifold
            .iter()
            .map(|&edge| {
                let (near_face, far_face) = self.edge_flanking_faces(edge);
                let straddle_face = self.straddle_quad_donor(
                    edge,
                    near_face,
                    far_face,
                    options.segments,
                    templates.interp_components(),
                );
                let near = templates.register(near_face);
                let far = templates.register(far_face);
                // The straddle donor is one of the two flanks whenever the override declines
                // (and at even `segments`, where the handle goes unread), so reuse the handle
                // rather than banking a third copy of the same block.
                let straddle = if straddle_face == near_face {
                    near
                } else if straddle_face == far_face {
                    far
                } else {
                    templates.register(straddle_face)
                };
                StripDonors {
                    near,
                    far,
                    straddle,
                    near_face,
                    near_at_chain_start: true,
                }
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
        let mut miter_ring_junctions: Vec<MiterRingJunction> = Vec::new();
        let mut terminal_fills: Vec<TerminalFill> = Vec::new();
        let mut cross_welds: Vec<CrossWeld> = Vec::new();

        // §7.2.1 / §8.3.0.4: the face-less spokes of every consumed endpoint, read pristine.
        let wire_spokes = self.snapshot_wire_spokes(&endpoints);
        let mut rings = BevelRings::for_spokes(&wire_spokes);

        // §8.2.2.1.1: the chord-fraction resample is a property of the [V-ADJ] cap's
        // *boundary emission*, so it fires on every ring-edge chain of a junction the
        // adjacency grid will own. The cutoff fill (§8.3, the rule's Scope blockquote)
        // is not a [V-ADJ] cap and consumes the coarse chain verts directly.
        let junction_emission = match options.vmesh_method {
            BevelVmeshMethod::Adj => ChainEmission::Resampled,
            BevelVmeshMethod::Cutoff => ChainEmission::Raw,
        };

        // §8.6.3.1(6) Regime A: decompose the beveled set into runs and mark every edge-end
        // with its handedness, *before* the first chain is minted — a run's bit is anchored at
        // its seed edge-end, which is generally at some other endpoint than the one being
        // built and whose ring therefore does not exist yet. The pass reads the pristine
        // corners only, so it can place the seed sector's two ring corners itself. Skipped
        // outright without a curve: every analytic cross-section is diagonal-symmetric, so the
        // bit is unobservable and the walk would be pure cost.
        let orientation = match custom {
            Some(_) => self.plan_custom_orientation(&beveled_manifold, &is_beveled, offset),
            None => CustomOrientation::default(),
        };

        for &endpoint in &endpoints {
            if junctions.contains(&endpoint) {
                // §8.9.3: the inner miter splits every non-reflex sector into an arc.
                // It engages only for `vmesh_method = Adj` with a non-sharp inner
                // miter; `build_junction_inner_miter` itself declines (without
                // mutating) any junction outside its scope, falling back to the sharp
                // boundvert ring below.
                let inner_miter = options.miter_inner != BevelMiter::Sharp
                    && options.vmesh_method == BevelVmeshMethod::Adj
                    && self.build_junction_inner_miter(
                        endpoint,
                        &is_beveled,
                        options.affect,
                        offset,
                        segments,
                        profile,
                        custom,
                        &orientation,
                        options.spread,
                        options.miter_outer,
                        &mut templates,
                        &mut chains,
                        &mut face_subst,
                        &mut miter_ring_junctions,
                        result,
                    );
                // §8.9.2: the outer miter reshapes the one reflex sector of an `s ≥ 3`
                // junction. Like the inner miter it classifies read-only and declines
                // (without mutating) any junction that is not a single-face reflex
                // sector past the firing margin, falling back to the sharp ring below.
                // `Arc` and `Patch` engage here (`Sharp` never does): they add `+1` /
                // `+2` reflex-ring boundverts respectively (§8.9.6). The dispatch gates
                // strictly on the enum value so `Patch` never silently serves the arc
                // round-over and vice versa.
                let outer_miter = !inner_miter
                    && matches!(options.miter_outer, BevelMiter::Arc | BevelMiter::Patch)
                    && options.vmesh_method == BevelVmeshMethod::Adj
                    && self.build_junction_outer_miter(
                        endpoint,
                        &is_beveled,
                        options.affect,
                        offset,
                        segments,
                        profile,
                        custom,
                        &orientation,
                        options.miter_outer,
                        &mut templates,
                        &mut chains,
                        &mut face_subst,
                        &mut miter_ring_junctions,
                        result,
                    );
                if !inner_miter && !outer_miter {
                    self.build_junction_chains(
                        endpoint,
                        &is_beveled,
                        options.affect,
                        offset,
                        segments,
                        profile,
                        custom,
                        &orientation,
                        junction_emission,
                        &mut templates,
                        &mut chains,
                        &mut face_subst,
                        &mut junction_cuts,
                        &mut cross_welds,
                        result,
                    );
                }
            } else {
                self.build_terminal_chains(
                    endpoint,
                    &beveled_set,
                    options.affect,
                    offset,
                    segments,
                    profile,
                    custom,
                    &orientation,
                    &mut templates,
                    &mut chains,
                    &mut face_subst,
                    &mut terminal_fills,
                    &mut rings,
                    result,
                );
            }
        }

        // §6.6.2: orient each strip's centre-line donor split now, while the flanking
        // originals still stand and still own the corners their boundverts descend from.
        for (&edge, donors) in beveled_manifold.iter().zip(&mut strip_donors) {
            let anchor = self.edge_vert_keys(edge)[0];
            donors.near_at_chain_start =
                near_flank_is_at_chain_start(anchor, edge, donors.near_face, &chains, &face_subst);
        }

        // §8.3.0.4: each junction's ring is its sector boundverts (a mitered junction's is
        // its enlarged ring's). The chains' *interior* rungs are ring verts but not
        // boundverts, so they are not candidates — which is what makes the wire's landing
        // independent of `segments`.
        for cut in &junction_cuts {
            rings.record(cut.vertex, || cut.boundverts.clone());
        }
        for miter in &miter_ring_junctions {
            rings.record(miter.vertex, || miter.boundverts.clone());
        }

        // §6.6.1: phase 1 has registered every donor — strips, junction fills, miter
        // rings, terminal fans — against the pristine mesh. Capture them all in one
        // snapshot before the re-stitch below kills and rebuilds them.
        templates.seal(self);

        // §6.7.1.1(a′): read every rebuilt original face's corner run — and the AND of
        // its two endpoint-incident original edges' flags — off the *pristine* mesh,
        // while those originals still meet at the corner. Applied once the re-stitch
        // below has laid the runs down as real face boundary edges (phase 4.25b).
        let corner_runs = self.snapshot_corner_runs(&face_subst);

        // §6.7.5.3 step 1: each junction's whole disk — every edge, beveled or not,
        // in rotational order, with the ring vertex each unbeveled leg re-points onto
        // (§6.7.5.4 case (4b)). Read pristine, for the same reason the corner runs are.
        // Mechanism (2) is the *only* reader, so a default bevel pays nothing for it.
        let junction_disks = if options.mark_seam || options.mark_sharp {
            self.snapshot_junction_disks(&junctions, &beveled_manifold, &face_subst)
        } else {
            Vec::new()
        };

        // Phase 2: re-stitch every incident face once, swapping each consumed
        // endpoint corner for its recorded vert run (§8.3.0.1 absorb / sector
        // 1-for-1), after re-pointing every unbeveled leg onto its near-corner vert
        // so the leg keeps its per-edge attributes across the bevel (§6.5 item 1).
        let mut incident_faces: Vec<FaceKey> = Vec::new();
        for &endpoint in &endpoints {
            for face in self.vert_face_keys(endpoint) {
                if !incident_faces.contains(&face) {
                    incident_faces.push(face);
                }
            }
        }
        let (rebuilt, donors) = self.restitch_consumed_endpoints(
            &incident_faces,
            &endpoints,
            &beveled_manifold,
            &face_subst,
        );

        // Phase 3: one strip (segments quads) per beveled edge over its two
        // endpoint chains (§8.3.0), its quads' donors split about the centre line (§6.6.2).
        for (&edge, donors) in beveled_manifold.iter().zip(&strip_donors) {
            let resolved = StripDonorRefs {
                near: templates.resolve(donors.near),
                far: templates.resolve(donors.far),
                straddle: templates.resolve(donors.straddle),
                near_at_chain_start: donors.near_at_chain_start,
            };
            self.build_multisegment_strip(edge, segments, &chains, material, resolved, result);
        }

        // Phase 3b (§8.3.0.3 attribute note): at a **cross** weld — a valence-4 welded
        // junction whose two beveled edges are opposite in the disk — the welded chain
        // reads as the continuation of the two crossing legs, so its edges take those
        // legs' per-edge attributes rather than fresh ones. Runs here, in strip order,
        // once the chain's edges exist: the two donations write the same edges and the
        // later one prevails.
        for cross in &cross_welds {
            self.apply_cross_weld_attributes(cross);
        }

        // Phase 4: the junction fill — flat cut (§8.3.2 / §8.3.4) or rounded
        // adjacency grid (§8.5), per `vmesh_method`. ADJ fills the cube corner (any
        // `segments`) and the general valence-`s` corner at every exponent (§8.5.5.5, and
        // (f.1) step 4's ungated fullness grid). §8.5.5.5(f.4a)'s square-in weld is filled
        // here too, and is the one variant that tiles no net: its chains were *identified* at
        // layout time, so all that is left is the odd-`segments` central `s`-gon.
        for cut in &junction_cuts {
            // §8.3.0.3: the fill kind is chosen by the ring's boundvert count, not by the
            // beveled-edge count and not by `vmesh_method`. A two-boundvert ring bounds a
            // zero-area 2-gon — its two arcs are one welded chain and both strips already
            // terminate on it — so **no** fill face is correct here, under either method.
            if cut.boundverts.len() < 3 {
                continue;
            }
            match options.vmesh_method {
                // §8.3.4.1: a cutoff ring that clears the pipe-override gate abandons its cut
                // faces and takes the all-quad adjacency grid instead. The connected fill is
                // **byte-identical to a from-scratch ADJ bevel** — same topology *and*
                // positions, because condition 3 forces a locally flat neighbourhood — so the
                // connected half is a plain delegation to `build_adj_fill`. On top of it the
                // override also leaves `(segments − 1) · s` **loose, valence-0 leftover verts**:
                // the cutoff pass sampled each beveled boundvert's profile-arc interior points
                // as real verts before the pipe test redirected the fill, and the ADJ builder
                // rebuilt its own arc and orphaned them. So V = V(ADJ) + (segments − 1)·s while
                // E and F match ADJ exactly.
                BevelVmeshMethod::Cutoff if cut.pipe_override => {
                    self.build_adj_fill(
                        cut,
                        &chains,
                        segments,
                        options.profile,
                        custom,
                        material,
                        &templates,
                        result,
                    );
                    self.mint_pipe_override_loose_verts(
                        cut,
                        &chains,
                        segments,
                        options.profile,
                        result,
                    );
                }
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
                        custom,
                        material,
                        &templates,
                        result,
                    );
                }
            }
        }

        // Phase 4d (§8.3.0.2): each valence-≥4 terminal takes its poly-fan — the
        // through-chain plus the leg verts, triangulated from a leg vert. The fan is
        // vmesh-independent (identical under Cutoff and Adj), so it is built for
        // every terminal regardless of `vmesh_method`, after the strip (phase 3) and
        // re-stitched leg faces (phase 2) exist so each triangle can wind against a
        // built neighbour.
        for fill in &terminal_fills {
            self.build_terminal_fill(fill, material, &templates, result);
        }

        // Phase 4b (§8.9.2 / §8.9.3 / §8.9.5): every mitered junction — inner, outer, or
        // both — takes the general valence-`s` [V-ADJ] fill over its enlarged ring. A
        // miter is a ring perturbation, so the fill needs no miter-specific step: each
        // split sector simply contributes an extra boundvert and an internal (fill)
        // chain the cap tiles like any cross-section (§8.9.5.3).
        for ring in &miter_ring_junctions {
            self.build_adj_fill_miter_ring(ring, material, &templates, result);
        }

        // Phase 4.25 (§7.7.2 / §6.7.5.2 mechanism (1), UNCONDITIONAL): each beveled edge
        // copies its whole per-edge block onto its strip's **two outer longitudinal
        // edges** (profile index 0 and index `segments`), the strip boundaries the
        // re-stitched adjacent faces rebuilt along the edge's two original sides. The
        // count is two independent of `segments`; the `segments − 1` interior
        // longitudinal lines and every interior vmesh-ring edge descend from no original
        // and keep the layer default. This holds at a junction end, a terminal end, and
        // between.
        self.inherit_multisegment_strip_long_edge_attributes(&beveled_manifold, &chains);

        // Phase 4.5 (§6.7.5.3 mechanism (2), gated by `mark_seam` / `mark_sharp`):
        // bridge each junction's gap as a run of outermost-ring chains, counting the
        // gap's length in **beveled edges** but stepping it out in **ring boundverts**
        // (§6.7.5.4 case (5)). Both the plain cutoff / adj junction fills
        // (`junction_cuts`, one boundvert per sector — count and steps in registration)
        // and the **mitered** junctions (`miter_ring_junctions`, each of whose split
        // sectors carries `+1` / `+2` boundverts and so costs a crossing span extra
        // steps — outer and inner miters alike, §6.7.5.4 case (6)) run the same walk
        // over their own ring. Terminals stay inert.
        //
        // This runs **before** the corner-run pass below, which assigns: a step burned
        // on a split sector's fill chain is a step written here and cleared there.
        if options.mark_seam || options.mark_sharp {
            for cut in &junction_cuts {
                let Some(disk) = junction_disks.iter().find(|disk| disk.vertex == cut.vertex)
                else {
                    continue;
                };
                let ring = cut.ring_chains(&chains);
                self.bridge_junction_flags(&ring, disk, options.mark_seam, options.mark_sharp);
            }
            for miter in &miter_ring_junctions {
                let Some(disk) = junction_disks
                    .iter()
                    .find(|disk| disk.vertex == miter.vertex)
                else {
                    continue;
                };
                let ring = miter.ring_chains();
                self.bridge_junction_flags(&ring, disk, options.mark_seam, options.mark_sharp);
            }
        }

        // Phase 4.6 (§7.7.3 / §6.7.1.1(a′) mechanism (1)'s corner-run half,
        // UNCONDITIONAL): every rebuilt original face's corner run takes the whole
        // per-edge block of that face's **winding-outgoing** original edge at the
        // endpoint, then re-derives the flag bits as the AND of that face's two
        // endpoint-incident originals. One pass, no case analysis: a miter-free sector's
        // run is a lone boundvert and does nothing, a mitered sector's run is the miter
        // fill chain, a multi-face reflex sector's absorbed legs break the AND on every
        // face that owns a fill edge, and a valence-3 terminal's absorbing face runs the
        // far-end cross-section chain off the leg it leaves the corner along. A
        // valence-≥4 terminal's far-end chain borders the *new* fan fill rather than a
        // rebuilt face, so it is no corner run and inherits nothing — the copy lands on
        // the rebuilt leg faces' runs instead, and it falls out of face provenance rather
        // than a special case (§7.7.3 regime 2). The pass **assigns**, so a FALSE AND
        // also *clears* — including any mark a desynced phase-4.5 span left on a fill
        // chain.
        self.apply_corner_run_attributes(&corner_runs);

        // Phase 4.7 (§5.2.1 fact (c)): the re-stitched originals blend every corner
        // from their own pre-bevel polygon, **last** — after every strip, fill and fan
        // built above, whose own per-vertex writes (the §5.2.4 profile / interior verts,
        // once they land) must not overwrite them.
        self.blend_restitched_originals(&rebuilt, &donors);

        // Phase D (§8.3.0.4): re-point every wire spoke onto the finished ring of the
        // vertex it hangs off — after every ring is final, before any endpoint is freed.
        // The attachment set is the ring's boundverts, so this is the same pass at every
        // `segments`: the extra profile rungs a larger `segments` mints are ring verts, not
        // boundverts, and never candidates.
        self.reattach_wire_spokes(&wire_spokes, &rings);

        // Phase 5 (§8.3.0.4 phase E): free each consumed endpoint that the strip build
        // actually left loose (§5.1.1). On a closed solid a terminal endpoint can remain
        // incident to unbeveled faces (its disk still carries live radial faces) — that
        // vert is kept, matching the reference operator, rather than force-freed.
        for &endpoint in &endpoints {
            if self.vertex_is_loose(endpoint) {
                self.free_loose_vertex(endpoint);
            }
        }
    }

    /// Build the cutoff fill at one junction (BEVEL.md §8.3.2 / §8.3.4): one cut
    /// face per sector spanning its chain plus two bottom corner verts, the corner
    /// ring closed by a bottom face — or collapsed to a single apex when the corner
    /// verts coincide (the `s = 3` case, §8.3.4).
    ///
    /// This is the operator's one **template-less** fill (§6.6.1.3): neither the cut
    /// faces nor the bottom face are handed a representative, so both land on the
    /// per-face layer defaults (slot `0`) even though the junction's originals are
    /// right there — the §6.6 stamp still applies on top. The junction's registered
    /// donor exists (the ADJ fill at the same junction uses it) and is deliberately
    /// not read here.
    pub(super) fn build_cutoff_fill(
        &mut self,
        cut: &JunctionCut,
        chains: &[EdgeEndChain],
        segments: u32,
        profile: f32,
        material: i32,
        result: &mut BevelResult,
    ) {
        let count = cut.boundverts.len();
        // §8.3.0.3: an `n == 2` ring is the welded chain, which no method fills.
        if count < 3 {
            return;
        }
        let representative = FaceTemplateRef::default();

        // Per boundvert, its down-pushed corner-vert position (§8.3.3). The push
        // is binary in the profile: full cell diagonal for a curved profile, zero
        // for the straight chord (which leaves the corner on the undisplaced ring).
        let mut corner_positions: Vec<Vec3> = Vec::with_capacity(count);
        for k in 0..count {
            corner_positions.push(self.corner_vert_position(cut, chains, k, profile));
        }

        // §8.3.4: the 3-way collapse to a single apex reads off the plain geometric
        // coincidence of the (already profile-aware) corner positions. A curved
        // profile pushes all three onto one point → collapse; the straight chord
        // leaves them at their distinct ring positions → suppressed and the bottom
        // triangle is built. For `s >= 4` the corners generally do not coincide, so
        // the bottom n-gon is built regardless.
        let collapsed = count == 3 && corner_verts_coincide(&corner_positions);

        let corner_verts: Vec<VertexKey> = if collapsed {
            let apex_position =
                (corner_positions[0] + corner_positions[1] + corner_positions[2]) / 3.0;
            // §5.2.4: not yet minted from the corner.
            let apex = self.create_vertex(apex_position);
            result.verts.push(apex);
            vec![apex; count]
        } else {
            (0..count)
                .map(|k| {
                    // §5.2.4: not yet minted from the corner.
                    let corner = self.create_vertex(corner_positions[k]);
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
            self.assign_bevel_face_attributes(
                face,
                material,
                representative,
                BevelNewFaceKind::Cap,
            );
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
            self.assign_bevel_face_attributes(
                face,
                material,
                representative,
                BevelNewFaceKind::Cap,
            );
            result.faces.push(face);
            self.collect_face_edges(face, result);
        }
    }
}

/// One multi-edge junction's cutoff-fill data (BEVEL.md §8.3): its ordered
/// boundvert ring, the per-sector opening beveled edge, and the outward surface
/// normal that orients the down-push (§8.3.3).
pub(super) struct JunctionCut {
    pub(super) vertex: VertexKey,
    pub(super) boundverts: Vec<VertexKey>,
    /// Per sector `k` (disk order), the beveled edge `g_k` that opens it (BEVEL.md
    /// §7.6.1). The outermost-ring chain crossing `g_m` spans boundverts `B_(m−1)`
    /// and `B_m`; this maps each junction bridging chain to the beveled edge it
    /// spans, for the §6.7.5.3 multi-segment seam / sharp bridging.
    pub(super) opening_edges: Vec<EdgeKey>,
    pub(super) outward: Vec3,
    /// The donor of the **odd-`segments` centre n-gon** alone (BEVEL.md §8.5.5.5(e)). That
    /// one face of the grid is a **ring-spanning** fill — its corners are the whole
    /// boundvert ring — so §6.6.1 pins it to the ring's first boundvert's rep, displaced by
    /// §6.6.1.4's tie-break at a seam-carrying corner. Every other grid face is per-sector
    /// and keeps `sector_representatives`.
    pub(super) central_representative: FaceTemplate,
    /// Per sector `i` (disk order, `boundverts[i]`'s), the [V-ADJ] grid donor for that
    /// sector's non-centre faces (BEVEL.md §6.6.2's grid row, edge bevel): the sector's
    /// **own** boundvert rep, `boundvert_representative` of `opening_edges[i]`. Registered
    /// against the pristine mesh alongside the ring-spanning donors, so it survives the
    /// re-stitch that consumes the junction vertex.
    pub(super) sector_representatives: Vec<FaceTemplate>,
    /// Per sector `i`, the donor of that sector's **centre column** at odd `segments` (the
    /// bridge strip meeting the centre n-gon, BEVEL.md §6.6.2 / §8.5): the sector's own rep
    /// when `opening_edges[i]` is a non-seam beveled edge, else the §6.6.1.4 tie-break
    /// between this and the next boundvert's reps. Unread at even `segments` (no centre
    /// column) and under vertex bevel (the caller supplies its own next-boundvert reps).
    pub(super) centre_column_representatives: Vec<FaceTemplate>,
    /// The original corner's §8.5.1 **dihedral-band** verdict (BEVEL.md §8.5.1 / §7.6.6), in
    /// its three values: the band's raw **total-angle deviation** from `3·π/2`, a hard
    /// **refusal** (an edge folded past `3·π/4`, or a near-flat count other than
    /// `edgecount − 3`), or **unmeasurable** (no closed fan to fold across — an open fan or a
    /// wire), where the position-only meet geometry stands in. Captured here — before the
    /// re-stitch consumes the vertex — because by fill time the corner's edges and faces are
    /// gone.
    ///
    /// The deviation is carried raw rather than pre-thresholded because the window is the
    /// *caller's*: it admits the tri-corner cube-corner cap for an **oblique / absorbed-leg /
    /// reflex** corner that the position-only meet-geometry test (equal spokes, 60° octant)
    /// would decline — the reflex-agnostic split of §7.6.6 — and a **square** exponent tests
    /// it against the tighter [`TRI_CORNER_BAND_SQUARE`] window the three-flat-squares grid
    /// demands (§8.5.5.5(f.1)).
    pub(super) dihedral_band: DihedralBand,
    /// Whether this junction takes the §8.3.4.1 **pipe override**: a `CUTOFF` ring of 3 or 4
    /// boundverts carrying an exactly anti-parallel opposite pair of beveled spokes (two
    /// spokes collinear *through* `v`) in a neighbourhood whose incident faces are all
    /// parallel to that pipe axis. When set, the cutoff fill is abandoned and the ring takes
    /// the all-quad adjacency grid (§8.5) over the *cutoff*-configured profile chains instead
    /// of the per-sector cut faces + bottom n-gon. Read from the *original* corner — the
    /// beveled-edge directions and incident-face normals are gone by fill time — so it is
    /// captured here alongside `dihedral_band`.
    pub(super) pipe_override: bool,
    /// Set when this junction's chains were laid as the §8.5.5.5(f.4a) **square-in weld**
    /// (`build_square_in_weld_chains`): edge bevel, the `r = 0` sentinel, no pipe, and a
    /// tri-corner ring the predicate *fires* on (valence 3, not merely three boundverts).
    /// The layout is a phase-1 decision — the strips are built over the chains, so the
    /// pairwise identification has to precede them — and the fill's only remaining job is the
    /// odd-`segments` central `s`-gon, so both halves read the one verdict rather than
    /// re-deriving it and risking disagreement. The layout's frame origin `c` does *not* ride
    /// along: the fill maps nothing and orients the `s`-gon off its own rim.
    pub(super) square_in_weld: bool,
    /// Per sector `k` (disk order), the **pristine** unit direction of `opening_edges[k]`
    /// out of `vertex` (BEVEL.md §8.5.5.5(f.5)). The square parallel-offset grid's centreline
    /// seeds are line-line meets on these beveled edges, but the edges themselves are consumed
    /// by the re-stitch — so their geometry is captured here, like `pipe_override`'s. `ZERO`
    /// for a degenerate (zero-length) edge. Empty on synthetic cuts that never fill.
    pub(super) opening_edge_directions: Vec<Vec3>,
    /// Per sector `k`, the pristine world position of `opening_edges[k]`'s **far** endpoint
    /// (the beveled-edge tip away from `vertex`). Pairs with `vertex` to give the beveled-edge
    /// **segment** that §8.5.5.5(f.5)'s pass-2 seed projects the later boundvert onto.
    pub(super) opening_edge_tips: Vec<Vec3>,
    /// Per boundvert `i`, whether §8.9.1's three-way signed test read boundvert `i`'s **own**
    /// disk sector (the wedge at `vertex` between `edge_in(i)` and `edge_out(i)`) as
    /// **non-reflex and not straight** — the entry condition of §8.5.5.5(f.5)'s case 3. A
    /// `false` covers all three of case 4's classes alike (reflex, straight, degenerate),
    /// which is exactly the granularity the seeding gate needs.
    ///
    /// Captured here, off the pristine corner, because the frame the verdict needs — the
    /// incident-face normal chain — is consumed by the re-stitch, and a flattened
    /// working-plane substitute misreads near-flat corners (§8.9.1's consequence 1).
    pub(super) boundvert_non_reflex: Vec<bool>,
}

impl JunctionCut {
    /// Resolve the per-sector [V-ADJ] grid donors (BEVEL.md §6.6.2's grid row) against the
    /// sealed template bank — one per sector, in `boundverts` / `chain_of` order.
    pub(super) fn resolve_sector_representatives<'a>(
        &self,
        templates: &'a TemplateBank,
    ) -> Vec<FaceTemplateRef<'a>> {
        self.sector_representatives
            .iter()
            .map(|&template| templates.resolve(template))
            .collect()
    }

    /// Resolve the per-sector odd-`segments` centre-column donors (BEVEL.md §6.6.2 / §8.5)
    /// against the sealed bank. Unread at even `segments` (no centre column).
    pub(super) fn resolve_centre_column_representatives<'a>(
        &self,
        templates: &'a TemplateBank,
    ) -> Vec<FaceTemplateRef<'a>> {
        self.centre_column_representatives
            .iter()
            .map(|&template| templates.resolve(template))
            .collect()
    }

    /// The junction's outermost ring as the §6.7.5.3 walk's chain sequence. Every
    /// sector carries exactly one boundvert here (no miter splits it), so every ring
    /// chain crosses a beveled edge and the walk's count and its steps stay in
    /// registration: `B_i → B_(i+1)` is crossed by `g_(i+1)`.
    pub(super) fn ring_chains(&self, chains: &[EdgeEndChain]) -> Vec<RingChain> {
        debug_assert_eq!(self.boundverts.len(), self.opening_edges.len());
        let count = self.boundverts.len();
        (0..count)
            .map(|i| {
                let cross_edge = self.opening_edges[(i + 1) % count];
                let samples = chains
                    .iter()
                    .find(|c| c.edge == cross_edge && c.endpoint == self.vertex)
                    .map(EdgeEndChain::samples)
                    .unwrap_or_default();
                RingChain {
                    cross_edge: Some(cross_edge),
                    samples,
                }
            })
            .collect()
    }
}

/// True when all the corner-vert positions coincide within tolerance (the §8.3.4
/// 3-way apex collapse test).
pub(super) fn corner_verts_coincide(positions: &[Vec3]) -> bool {
    if positions.len() < 2 {
        return true;
    }
    let first = positions[0];
    positions
        .iter()
        .all(|&position| (position - first).length() <= 1.0e-4)
}
