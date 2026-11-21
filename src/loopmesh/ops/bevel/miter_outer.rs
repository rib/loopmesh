//! The outer miter and its multi-face re-stitch (BEVEL.md §8.9.2).

use super::*;

impl LoopMesh {
    /// Build a junction's **outer-miter** boundvert ring for its one reflex sector,
    /// riding the general valence-`s` [V-ADJ] fill (BEVEL.md §8.9.2 / §8.9.5).
    ///
    /// The reflex sector's §7.6.2 meet-point becomes an **arc start** `B1`; one new
    /// boundvert `B3` takes over the closing beveled edge's side (`ARC`, §8.9.2). Both
    /// are slid **back along their own beveled edges** until each lands level with the
    /// neighbouring boundvert (the neighbour-plane reposition), while the *original*
    /// meet-point is kept as the arc's stored midpoint — the §8.9.5 circumscribing-
    /// square corner that fixes the bulge sample (pulled in a fraction `√2 − 1` from the
    /// chord midpoint toward that corner). Every other (non-reflex) sector keeps its
    /// single sharp meet-point, so the junction presents an `(s + 1)`-boundvert ring
    /// the general grid tiles.
    ///
    /// Scoped to `s ≥ 3` at every profile: both `ARC` (§8.9.5.1–§8.9.5.3) and `PATCH`
    /// (§8.9.2's `s − 1` shoulders per chord leg) are segments- and exponent-general. Both
    /// engage at a
    /// **single-face** reflex sector (no absorbed unbeveled legs); a **multi-face**
    /// reflex sector (absorbed legs) re-stitches for `PATCH` by a fan to the hub `B2`
    /// (§8.9.2.1, `restitch_multiface_patch_reflex`) and for `ARC` by tiling the arc's
    /// profile-sample run at either parity (§8.9.2.2 / §8.9.2.3,
    /// `restitch_multiface_arc_sector`). All classification runs read-only first; the
    /// method returns `false` **without mutating** for any junction outside those cases
    /// (no reflex sector — the miter is inert — more than one reflex sector, `s < 3`, or
    /// `segments < 2`), so the caller falls back to the sharp ring.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn build_junction_outer_miter<'c>(
        &mut self,
        vertex: VertexKey,
        is_beveled: &dyn Fn(EdgeKey) -> bool,
        affect: BevelAffect,
        offset: f32,
        segments: u32,
        profile: f32,
        custom: Option<&'c CustomChains<'c>>,
        orientation: &CustomOrientation,
        mode: BevelMiter,
        templates: &mut TemplateBank,
        chains: &mut Vec<EdgeEndChain>,
        face_subst: &mut Vec<FaceSubstitution>,
        miter_ring_junctions: &mut Vec<MiterRingJunction<'c>>,
        result: &mut BevelResult,
    ) -> bool {
        // The outer miter carries **no exponent gate**: §8.9.2 / §8.9.6 gate it on the `s ≥ 3`
        // beveled-edge count and the §8.9.1 firing margin, nothing else. §8.9.5's
        // exponent-scope note makes the arc chain read the run's own §8.6.4 spacing table at
        // every `r` — `profile = 0.25` collapses it to the plain chord, the square sentinel
        // walks it through the stored corner — and §8.9.5.3 hands the enlarged ring to
        // §8.5.5.5's ordinary (f.1) dispatch with no miter-specific step, so *which* grid
        // fills the cap is that dispatch's business and never this gate's. The only floor is
        // §8.9.5's: the miter rides the [V-ADJ] grid, hence `segments ≥ 2`.
        //
        // In particular the parallel-offset lattice of (f.5) is claimed by step 1 only at the
        // square exponent with **even** `segments` and no custom curve, and
        // `build_adj_fill_miter_ring` asks step 1's predicate itself
        // (`adj_square_ring_claims`) over the enlarged ring — with the (f.5) seed inputs this
        // builder mints below. Everything else lands on step 4's general fullness grid, whose
        // `fullness` is (f.3)'s law at the run's own `profile`.
        if segments < 2 {
            return false;
        }
        let corners = self.vertex_bevel_corner_ring(vertex, is_beveled);
        let sectors = sector_split(&corners, is_beveled);
        let sector_count = sectors.len();
        // §8.9.2: fewer than three beveled edges forces `SHARP`.
        if sector_count < 3 {
            return false;
        }
        let vertex_position = self.vertex_position(vertex);

        // Read-only classification (§8.9.1): per sector solve the meet-point and the two
        // flanking edge directions, and take the three-way straight / reflex / non-reflex
        // verdict.
        struct SectorPlan {
            meet: Vec3,
            dir_open: Vec3,
            dir_close: Vec3,
            class: MiterSectorClass,
        }
        let mut plans: Vec<SectorPlan> = Vec::with_capacity(sector_count);
        for sector in &sectors {
            if sector.corners.is_empty() {
                return false;
            }
            let Some(verdict) = self.classify_miter_sector(vertex, vertex_position, sector) else {
                return false;
            };
            let meet = self.solve_meet_point(vertex, vertex_position, sector, offset);
            plans.push(SectorPlan {
                meet,
                dir_open: verdict.dir_open,
                dir_close: verdict.dir_close,
                class: verdict.class,
            });
        }

        // §8.9.1 / §8.9.2: exactly one reflex sector fires the outer miter. Zero ⇒ the
        // miter is inert (no reflex corner); fall back to the sharp ring. More than one
        // ⇒ the signed test contradicts the "at most one reflex sector" invariant, so
        // decline defensively rather than build an ill-defined ring.
        let reflex_indices: Vec<usize> = (0..sector_count)
            .filter(|&k| plans[k].class == MiterSectorClass::Reflex)
            .collect();
        if reflex_indices.len() != 1 {
            return false;
        }
        let r = reflex_indices[0];

        // A **single-face** reflex sector (no absorbed unbeveled legs) re-stitches its
        // one face onto the whole run like a non-miter sector (§7.6.3). A **multi-face**
        // reflex sector absorbs one or more unbeveled legs: `Patch` fans them to the hub
        // `B2` (§8.9.2.1); `Arc` tiles them along the arc's profile-sample run — the
        // even/odd run-split re-stitch of §8.9.2.2 / §8.9.2.3, segments-general. Only
        // `Sharp` never reaches here, so no mode is declined on the multi-face path.
        let mut reflex_faces: Vec<FaceKey> = Vec::new();
        for &(_, face) in &sectors[r].corners {
            if !reflex_faces.contains(&face) {
                reflex_faces.push(face);
            }
        }

        // Every check has passed — mutate.
        let midpoint = plans[r].meet;
        let predecessor = plans[(r + sector_count - 1) % sector_count].meet;
        let successor = plans[(r + 1) % sector_count].meet;
        let dir_open = plans[r].dir_open; // e_miter — the reflex sector's opening edge
        let dir_close = plans[r].dir_close; // the reflex sector's closing edge

        // §8.9.2 neighbour-plane reposition: slide `B1` along `e_miter` until level with
        // the preceding boundvert, and `B3` along the closing edge until level with the
        // following boundvert. Each is the projection of a neighbour's plane back along
        // the slide direction; the line runs *along* that same direction so it never
        // sits parallel to the plane (no degenerate `offset / (segments / 2)` fallback).
        let b1_position = midpoint + (predecessor - midpoint).dot(dir_open) * dir_open;
        let b3_position = midpoint + (successor - midpoint).dot(dir_close) * dir_close;

        // §5.2.4: not yet minted from the corner.
        let b1 = self.create_vertex(b1_position);
        let b3 = self.create_vertex(b3_position);
        result.verts.push(b1);
        result.verts.push(b3);

        // The reflex sector's interior ring differs by mode (§8.9.2 / §8.9.6):
        //   - `Arc` adds no interior ring boundvert; the single `B1 → B3` internal edge
        //     carries the `segments + 1`-sample arc chain `s_0 … s_segments` (§8.9.5.1) —
        //     `s_0 = B1`, `s_segments = B3`, each interior sample at the run's own §8.6.4
        //     even-chord parameter `(u_k, v_k)` in the arc's unit-square frame. At the round
        //     profile and `segments = 2` the lone interior sample is the `√2 − 1` bulge.
        //   - `Patch` promotes the retained meet corner to the middle boundvert `B2`, so
        //     the ring gains `B1 → B2 → B3`; the two internal edges carry flat chord
        //     samples evenly spaced along their straight leg — no corner-ward bulge, so
        //     the tent apex `B2` stays at the meet while its two ends slid back (§8.9.5).
        //     Segments-general: each leg carries `segments − 1` shoulder samples
        //     (§8.9.2.1), one at the leg midpoint when `segments = 2`.
        // `reflex_ring` is the sector's ordered ring boundverts; `internal_chains` holds
        // one full `k + 1`-sample chain per internal (`None` cross-edge) edge between them
        // (oriented along the ring); `raw_internal_mids` holds each of those chains' *raw*
        // coarse `(i, 0, 1)` control point — the pre-resample profile mid the centre
        // stencil consumes (§8.5.5.5(d) / §8.9.5.3), which is **not** the emitted rim
        // sample; `forward_run` is the reflex face's rebuilt boundary threading `B1 … B3`
        // in sector order.
        let (reflex_ring, internal_chains, raw_internal_mids, internal_profiles, forward_run): (
            Vec<VertexKey>,
            Vec<Vec<VertexKey>>,
            Vec<Vec3>,
            Vec<RingChainProfile>,
            Vec<VertexKey>,
        ) = match mode {
            BevelMiter::Arc => {
                // §8.9.5.1: the reflex chain is the run's own cross-section quadrant sampled
                // in the (non-orthonormal) unit-square frame of `B1` / `B3` / the stored
                // midpoint — the §8.6.4 even-chord table at the `profile`-derived exponent,
                // *not* a forced circle (§8.9.5's exponent-scope note). At the square sentinel
                // the raw chain therefore walks the two cell sides through the stored midpoint.
                //
                // Non-custom even under `profile_type = CUSTOM`, by §8.6.3.1(4).4's miter
                // carve-out: `B1` is the arc start, and an arc / patch start is handed a
                // second, analytic spacing pair and skips §8.6.3.1(6)'s handedness pass
                // outright (A.6). Matching behaviour, not an omission. That pair is built
                // "from the `profile` scalar", so the carve-out withholds the **curve** and
                // not the slider — a `CUSTOM` run's miter chain moves with `profile`.
                let arc = RingChainProfile::Arc {
                    b1: b1_position,
                    b3: b3_position,
                    midpoint,
                    profile,
                };
                let raw_bulge = arc.raw_positions(2)[1];

                // §8.2.2.1.1 / §8.9.5.2: the arc is a ring-edge chain of the [V-ADJ] cap
                // like any other, so its *emitted* rim samples are the chord-fraction
                // resample of the raw arc against the profile materialised at the source
                // resolution. On an asymmetric (absorbed-leg) reflex fan the arc cell is
                // the most skewed in the cap and the shift is large. The centre stencil
                // keeps reading `raw_bulge` (below) — the two roles are distinct values.
                let raw_chain = arc.raw_positions(segments);
                let source_segments = boundary_source_resolution(segments);
                let emitted = if source_segments == segments {
                    raw_chain
                } else {
                    let source = arc.raw_positions(source_segments);
                    chord_fraction_resample(&raw_chain, &source)
                };

                let mut samples = Vec::with_capacity(segments as usize + 1);
                samples.push(b1);
                for position in emitted.iter().take(segments as usize).skip(1) {
                    // §5.2.4: not yet minted from the corner.
                    let sample = self.create_vertex(*position);
                    result.verts.push(sample);
                    samples.push(sample);
                }
                samples.push(b3);
                (
                    vec![b1, b3],
                    vec![samples.clone()],
                    vec![raw_bulge],
                    vec![arc],
                    samples,
                )
            }
            BevelMiter::Patch => {
                // §5.2.4: not yet minted from the corner.
                let b2 = self.create_vertex(midpoint);
                result.verts.push(b2);
                // §8.9.2 / §8.9.5 build **both** chord legs with straight profiles: the
                // exponent is not forced to a circle and no bulge is applied, so each leg's
                // interior samples are evenly spaced along the line between its ends. Straight
                // under a custom curve too, and not because of any boundvert's tag — `B2 → B3`
                // is not `B1`'s leg and `B2` carries no tag at all. That construction is blind
                // to the profile type.
                let legs = [
                    RingChainProfile::Chord {
                        start: b1_position,
                        end: midpoint,
                    },
                    RingChainProfile::Chord {
                        start: midpoint,
                        end: b3_position,
                    },
                ];
                let leg_ends = [(b1, b2), (b2, b3)];
                // §8.9.2.1: at `segments = s` each leg carries `s − 1` shoulder samples.
                // The §8.2.2.1.1 chord-fraction resample every ring chain runs through is
                // the **identity** here whatever `segments` is: even spacing along a
                // straight line *is* even chord spacing, and the source polyline is that
                // same line, so each target fraction reads back its own point. So raw and
                // emitted coincide on a chord — unlike the arc, where they part company at
                // every non-power-of-two count (§8.9.5.1).
                let [leading_leg, trailing_leg]: [Vec<VertexKey>; 2] =
                    std::array::from_fn(|index| {
                        let (leg_start, leg_end) = leg_ends[index];
                        let positions = legs[index].raw_positions(segments);
                        let mut chain = Vec::with_capacity(segments as usize + 1);
                        chain.push(leg_start);
                        for position in positions.iter().take(segments as usize).skip(1) {
                            // §5.2.4: not yet minted from the corner.
                            let shoulder = self.create_vertex(*position);
                            result.verts.push(shoulder);
                            chain.push(shoulder);
                        }
                        chain.push(leg_end);
                        chain
                    });
                // `[B1, shoulders…, B2, shoulders…, B3]` — the hub `B2` at index `segments`.
                let mut run = Vec::with_capacity(2 * segments as usize + 1);
                run.extend_from_slice(&leading_leg);
                run.extend_from_slice(&trailing_leg[1..]);
                // §8.5.5.5(d) / §8.9.5.3: the centre stencil eats each ring chain's raw
                // *coarse* `(i, 0, 1)` control point — the leg's own midpoint, whatever
                // resolution the emitted chain was built at. Only the `segments = 2`
                // stencil consumes them (see `raw_coarse_mids` below), and there the coarse
                // resolution *is* `segments`, so this repeats the mid already emitted rather
                // than recovering a distinct one — as it must, a chord being resample-inert.
                let raw_mids = legs.iter().map(|leg| leg.raw_positions(2)[1]).collect();
                (
                    vec![b1, b2, b3],
                    vec![leading_leg, trailing_leg],
                    raw_mids,
                    legs.to_vec(),
                    run,
                )
            }
            BevelMiter::Sharp => {
                unreachable!("the outer-miter dispatch calls this only for Arc / Patch (§8.9.2)")
            }
        };

        // The non-reflex sectors keep their single sharp meet-point boundvert.
        let mut sharp: Vec<Option<VertexKey>> = vec![None; sector_count];
        for (k, plan) in plans.iter().enumerate() {
            if k == r {
                continue;
            }
            // §5.2.4: not yet minted from the corner.
            let boundvert = self.create_vertex(plan.meet);
            result.verts.push(boundvert);
            sharp[k] = Some(boundvert);
        }

        // Re-stitch: each non-reflex sector face collapses to its single boundvert
        // (1-for-1); the reflex sector's one face re-stitches onto the whole
        // `forward_run`, ordered to the face's traversal through `v`. That run's length is
        // both mode- and count-dependent — `segments + 1` slots under `Arc` (the sample
        // chain `B1 … B3`), `2·segments + 1` under `Patch` (the two chord legs spliced at
        // the hub `B2`) — so the familiar 1-for-3 is only the `Arc`, `segments = 2` corner
        // of the rule, not the rule.
        for (k, sector) in sectors.iter().enumerate() {
            if k == r {
                continue;
            }
            let boundvert = sharp[k].expect("non-reflex sector has a sharp boundvert");
            for &(_, face) in &sector.corners {
                set_face_substitution(face_subst, vertex, face, vec![boundvert]);
            }
        }
        if reflex_faces.len() == 1 {
            // Single-face reflex sector: the one face threads the whole run in its own
            // winding (forward when it enters the apex from the sector's opening edge).
            let reflex_face = reflex_faces[0];
            let run = match self.face_corner_edges(reflex_face, vertex) {
                Some((incoming, _)) if incoming == sectors[r].bounding_edges[0] => {
                    forward_run.clone()
                }
                _ => {
                    let mut reversed = forward_run.clone();
                    reversed.reverse();
                    reversed
                }
            };
            set_face_substitution(face_subst, vertex, reflex_face, run);
        } else {
            // Multi-face reflex sector: rebuild each absorbed-leg face by mode.
            match mode {
                // §8.9.2.1: `Patch` fans every reflex face to the hub `B2` (identical to
                // the SHARP outer join), opening only the two flank faces, each of which
                // grows to a `(segments + 3)`-gon along its patch leg. `forward_run` is the
                // patch chain `[B1, shoulders…, B2, shoulders…, B3]`, so the hub sits at
                // index `segments`.
                BevelMiter::Patch => {
                    self.restitch_multiface_patch_reflex(
                        vertex,
                        &sectors[r],
                        &forward_run,
                        segments,
                        face_subst,
                    );
                }
                // §8.9.2.2: `Arc` tiles the reflex faces along the arc sample run — no
                // hub. `forward_run` is the sample chain `[s_0 = B1, s_1 = bulge, …,
                // s_segments = B3]`, so each face threads the inclusive run between its two
                // legs' assigned sample indices.
                BevelMiter::Arc => {
                    self.restitch_multiface_arc_sector(
                        vertex,
                        &sectors[r],
                        &forward_run,
                        segments,
                        face_subst,
                    );
                }
                BevelMiter::Sharp => {
                    unreachable!("the outer-miter dispatch calls this only for Arc / Patch")
                }
            }
        }

        // Build the enlarged ring in disk order together with the crossing beveled edge
        // (or the arc) on each ring edge. `cross_edge[i]` is `Some` for a strip
        // `ring[i] → ring[i+1]`, `None` for the internal `B1 → B3` arc.
        let mut ring: Vec<VertexKey> = Vec::with_capacity(sector_count + 1);
        let mut cross_edge: Vec<Option<EdgeKey>> = Vec::with_capacity(sector_count + 1);
        // The enlarged-ring index of the reflex sector's leading boundvert `B_1` — the
        // §8.5.5.5(f.5) arc/patch start (case 1/2 classification), captured as it is pushed.
        let mut b1_ring_index = 0usize;
        // §8.5.5.5(f.5) case 3's per-boundvert entry condition, carried in ring order.
        let mut boundvert_non_reflex: Vec<bool> = Vec::with_capacity(sector_count + 2);
        for k in 0..sector_count {
            let next_opening = sectors[(k + 1) % sector_count].bounding_edges[0];
            if k == r {
                b1_ring_index = ring.len();
                for (j, &ring_vert) in reflex_ring.iter().enumerate() {
                    ring.push(ring_vert);
                    boundvert_non_reflex.push(false);
                    if j + 1 < reflex_ring.len() {
                        cross_edge.push(None);
                    } else {
                        cross_edge.push(Some(next_opening));
                    }
                }
            } else {
                ring.push(sharp[k].expect("non-reflex sector has a sharp boundvert"));
                boundvert_non_reflex.push(plans[k].class == MiterSectorClass::NonReflex);
                cross_edge.push(Some(next_opening));
            }
        }

        // The full `segments + 1`-sample chain on each ring edge (oriented `ring[i] →
        // ring[i+1]`): the strip cross-section across the shared beveled edge (§8.9.5.2),
        // or the reflex sector's arc chain between `B1` and `B3` (§8.9.5.1). The general
        // valence-`(s + 1)` cap fill (§8.9.5.3) tiles these heterogeneous chains as the
        // §8.5.5.5 grid, so each must carry its full profile, not just one interior mid.
        let mut internal_chains = internal_chains.into_iter();
        let mut internal_profiles = internal_profiles.into_iter();
        let mut ring_chains: Vec<Vec<VertexKey>> = Vec::with_capacity(ring.len());
        let mut ring_profiles: Vec<RingChainProfile> = Vec::with_capacity(ring.len());
        for i in 0..ring.len() {
            let start = ring[i];
            let end = ring[(i + 1) % ring.len()];
            match cross_edge[i] {
                None => {
                    let chain = internal_chains
                        .next()
                        .expect("every internal reflex edge has a precreated sample chain");
                    ring_chains.push(chain);
                    ring_profiles.push(
                        internal_profiles
                            .next()
                            .expect("every internal reflex edge has a precreated profile"),
                    );
                }
                Some(edge) => {
                    let rescue = self.mid_rescue(vertex, edge, &corners, is_beveled);
                    // §8.6.3.1(4).4 / §8.6.3.1(6) A.0: this ring edge is a beveled edge's
                    // cross-section, owned by the corner it opens at — a plain sharp
                    // meet-point, or the reflex sector's arc *end* `B3`, which §8.9.2 gives
                    // genuine ownership of the closing beveled edge while clearing the arc
                    // start `B1`'s own-edge slot. The carve-out tags only arc / patch
                    // *starts*, so every flank here reads the curve with A.3's mark.
                    let reverse = orientation.reverse(edge, vertex);
                    let read = custom.map(|chains| CustomChainRead::edge_sector(chains, reverse));
                    ring_profiles
                        .push(self.ring_flank_profile(start, end, edge, profile, read, rescue));
                    // §8.9.5.2: each flanking beveled-edge cross-section carries `segments
                    // − 1` interior samples. The reflex sector's flanking pair runs to an
                    // inward-repositioned arc-start boundvert, making the cell skewed; at
                    // `segments = 2` its lone interior sample is the §8.2.2.1.1 chord-
                    // fraction resample (identity on the symmetric interior sector strips).
                    let (interior, cell) = self.build_chain_between(
                        start,
                        end,
                        edge,
                        segments,
                        profile,
                        read,
                        rescue,
                        None,
                        ChainEmission::Resampled,
                        result,
                    );
                    // A circular cross-section always samples to `segments − 1` interior
                    // verts; a short chain here would be an internal-invariant violation,
                    // and we are already past the read-only classification pass (verts /
                    // substitutions are minted), so a graceful decline is impossible
                    // without orphaning geometry.
                    // Unreachable: the sample count is geometry-independent — the profile
                    // sampler emits exactly `segments + 1` positions (both the collinear
                    // and the arc branch), so `interior` is always `segments − 1` long. The
                    // guard is a pure invariant assertion, never an expected-failure path.
                    if interior.len() != segments as usize - 1 {
                        panic!("cross-section must sample to segments−1 interior verts");
                    }
                    let mut chain = Vec::with_capacity(segments as usize + 1);
                    chain.push(start);
                    chain.extend_from_slice(&interior);
                    chain.push(end);
                    ring_chains.push(chain);
                    chains.push(EdgeEndChain {
                        edge,
                        endpoint: vertex,
                        start,
                        interior,
                        end,
                        side_label: SideLabel::Unlabelled,
                        cell: Some(cell),
                        custom_reverse: reverse,
                    });
                }
            }
        }

        // §8.5.5.5(d) / §8.9.5.3: the extraordinary-centre stencil must consume each ring
        // chain's *raw coarse* `(i, 0, 1)` profile mid, never the emitted rim mid. Every
        // chain — the reflex arc included — emits the §8.2.2.1.1 chord-fraction resample,
        // so on a skewed cell the two values differ and reading the rim vertex back would
        // skew the centre. Flank mids come from the pre-resample cell sample; the internal
        // (arc / patch-chord) mids were captured raw when the chain was minted. Only the
        // `segments = 2` centre stencil consumes these, so populate them only there.
        let mut internal_index = 0usize;
        let raw_coarse_mids: Vec<Vec3> = if segments == 2 {
            (0..ring.len())
                .map(|i| match cross_edge[i] {
                    None => {
                        let mid = raw_internal_mids[internal_index];
                        internal_index += 1;
                        mid
                    }
                    Some(edge) => self.coarse_flank_mid(
                        ring[i],
                        ring[(i + 1) % ring.len()],
                        edge,
                        profile,
                        custom.map(|chains| {
                            CustomChainRead::edge_sector(chains, orientation.reverse(edge, vertex))
                        }),
                        self.mid_rescue(vertex, edge, &corners, is_beveled),
                    ),
                })
                .collect()
        } else {
            Vec::new()
        };

        // §6.6.2 grid row: each non-centre grid face takes its own sector's positional donor,
        // not one uniform rep — a two-material split-sector junction diverges otherwise.
        let sector_donors = self.miter_ring_sector_donors(vertex, &cross_edge);
        let sector_representatives: Vec<FaceTemplate> = sector_donors
            .iter()
            .map(|&donor| templates.register(donor))
            .collect();
        // §8.9.5.3: the outer-miter cap is the §8.5.5.5 general grid over the enlarged
        // `(s + 1)`-boundvert ring, so its odd-`segments` centre n-gon (§8.5.5.5(e)) takes the
        // **enlarged miter ring's first boundvert's** rep — the §6.6.1.2 fan anchor of *this*
        // ring, not the raw disk's `first_incident_face` — on a seam-free junction (a material
        // boundary raises no interpolation seam), and only a genuine seam routes it to the
        // junction-wide §6.6.1.4 tie-break. Aligned-by-construction with the inner miter, which
        // splits its own sectors but feeds the same enlarged-ring grid.
        let central_representative = templates.register(self.miter_centre_ngon_donor(
            vertex,
            is_beveled,
            affect,
            segments,
            sector_donors.first().copied().flatten(),
            templates.interp_components(),
        ));
        // §6.6.2 / §8.9.3.1: at odd `segments` each sector's centre column resolves its own
        // donor — a seam-gated own-rep-vs-tie-break for a beveled sector (its `Some`
        // cross-edge, including the reflex arc *end* `B_3`, which owns the trailing beveled
        // edge), the §6.6.1.4 tie-break unconditionally for the reflex arc's edge-less *start*
        // `B_1` (its `None` cross-edge). Computed pristine here.
        let centre_column_donors = self.miter_ring_centre_column_donors(
            vertex,
            &cross_edge,
            templates.interp_components(),
        );
        let centre_column_representatives: Vec<FaceTemplate> = centre_column_donors
            .into_iter()
            .map(|donor| templates.register(donor))
            .collect();

        // §8.5.5.5(f.5) seed inputs for the square parallel-offset fill. The seeding pass is
        // over ring boundverts: `edge_in(i)` — the beveled edge boundvert `i` shares with its
        // *predecessor* — is `cross_edge[i − 1]` (the edge that ring edge crosses), and it is
        // `None` on the reflex sector's internal arc/patch edge, giving `Vec3::ZERO`. Only the
        // square profile reads these; the circular outer miter takes the general relaxed cap.
        let ring_len = ring.len();
        let opening_edge_directions: Vec<Vec3> = (0..ring_len)
            .map(|i| match cross_edge[(i + ring_len - 1) % ring_len] {
                Some(edge) => self
                    .edge_direction_from(edge, vertex, vertex_position)
                    .unwrap_or(Vec3::ZERO),
                None => Vec3::ZERO,
            })
            .collect();
        let opening_edge_tips: Vec<Vec3> = (0..ring_len)
            .map(|i| match cross_edge[(i + ring_len - 1) % ring_len] {
                Some(edge) => {
                    let [a, b] = self.edge_vert_keys(edge);
                    let far = if a == vertex { b } else { a };
                    self.vertex_position(far)
                }
                None => Vec3::ZERO,
            })
            .collect();
        // Case 1/2 classification: the fired-reflex leading `B_1`, its mode, and — for `ARC` —
        // the stored §7.6.2 middle control point (`midpoint`, the pre-reposition meet-point).
        let arc_patch_starts = vec![(b1_ring_index, mode, midpoint)];

        miter_ring_junctions.push(MiterRingJunction {
            vertex,
            boundverts: ring,
            chains: ring_chains,
            raw_coarse_mids,
            profiles: ring_profiles,
            cross_edges: cross_edge,
            segments,
            profile,
            custom,
            sector_representatives,
            central_representative,
            centre_column_representatives,
            opening_edge_directions,
            opening_edge_tips,
            boundvert_non_reflex,
            arc_patch_starts,
        });
        true
    }

    /// Re-stitch a **multi-face** reflex sector for the outer `PATCH` miter
    /// (BEVEL.md §8.9.2.1). The sector's `> 180°` gap absorbs one or more unbeveled
    /// legs; `patch_chain` is the three ring boundverts with each leg's `segments − 1` flat
    /// chord samples (the grid **shoulders**) interleaved — `[B1, SH_lead…, B2, SH_trail…,
    /// B3]`, so the hub `B2` sits at index `segments`.
    ///
    /// `PATCH` closes the reflex hole by a **fan to the single hub `B2`**, not by tiling
    /// the ring run. `B2` is the retained §7.6.2 meet-point (the lifted `SHARP` V-join
    /// crossing), and **every** reflex face closes at it — the interior fan is identical
    /// to the default `SHARP` outer join. `PATCH`'s sole departure is at the sector's two
    /// **flank faces** (the ones bounded by a flanking *beveled* edge): there the sharp
    /// apex corner opens along that flank's subdivided patch leg (hub → shoulder →
    /// flank boundvert), growing the triangle to an `(s + 3)`-gon (a pentagon at
    /// `segments = 2`). The interior faces stay bare hub triangles `[B2, R_k, R_{k+1}]`.
    /// There is no per-leg index assignment and no quad-growth.
    pub(super) fn restitch_multiface_patch_reflex(
        &self,
        vertex: VertexKey,
        reflex_sector: &Sector,
        patch_chain: &[VertexKey],
        segments: u32,
        face_subst: &mut Vec<FaceSubstitution>,
    ) {
        // The sector's legs in disk order (leading flank, absorbed legs, trailing
        // flank) and the face bounded by each consecutive leg pair. `corners` lists
        // one `(incoming_leg, face)` per face in order, plus one trailing duplicate for
        // the closing beveled edge — so the legs are every corner's edge and the face
        // between `legs[i]` and `legs[i + 1]` is `corners[i].1`.
        let legs: Vec<EdgeKey> = reflex_sector
            .corners
            .iter()
            .map(|&(edge, _)| edge)
            .collect();
        let leg_count = legs.len();
        // A multi-face reflex sector is two flanking beveled edges plus ≥ 1 absorbed
        // leg; anything shorter is single-face and never reaches here. `patch_chain` is
        // the two `segments`-edge legs spliced at the hub, so `2·segments + 1` slots.
        let seg = segments as usize;
        if leg_count < 3 || patch_chain.len() != 2 * seg + 1 {
            return;
        }
        let hub = patch_chain[seg]; // B2
        let leading_flank = 0;
        let trailing_flank = leg_count - 2;

        for face_slot in 0..leg_count - 1 {
            let face = reflex_sector.corners[face_slot].1;
            // The apex-corner run, ordered from the face's disk-*leading* leg side to its
            // trailing side. Interior faces collapse the apex to the lone hub; the two
            // flank faces splice that flank's whole patch leg (`B_flank → shoulders… →
            // hub`) between the hub and the flank's reflex boundvert, growing the triangle
            // to an `(segments + 3)`-gon (§8.9.2.1).
            let forward: Vec<VertexKey> = if face_slot == leading_flank {
                // Leading flank (disk-leading leg is the flank beveled edge): the flank
                // boundvert B1 sits on the leading side, the hub on the trailing side.
                patch_chain[0..=seg].to_vec() // [B1, SH_lead…, B2]
            } else if face_slot == trailing_flank {
                // Trailing flank (disk-trailing leg is the flank beveled edge): the hub
                // sits on the leading side, the flank boundvert B3 on the trailing side.
                patch_chain[seg..=2 * seg].to_vec() // [B2, SH_trail…, B3]
            } else {
                vec![hub]
            };
            // Orient the run to the face's own winding: forward when the face enters the
            // apex from its disk-leading leg, reversed otherwise (§7.6.3 convention).
            let run = match self.face_corner_edges(face, vertex) {
                Some((incoming, _)) if incoming == legs[face_slot] => forward,
                _ => {
                    let mut reversed = forward;
                    reversed.reverse();
                    reversed
                }
            };
            set_face_substitution(face_subst, vertex, face, run);
        }
    }

    /// Re-stitch a **multi-face** arc-split sector (BEVEL.md §8.9.2.2, **even
    /// `segments`**) — the outer `ARC` miter's reflex sector, or an inner-mitered
    /// non-reflex sector that absorbed unbeveled legs, since §8.9.3 builds the identical
    /// arc there. Unlike `PATCH`, an arc has **no** `B2`
    /// ring hub: the arc between `B1` and `B3` is realised as its full profile — the
    /// sample chain `[s_0 = B1, s_1, …, s_segments = B3]` (`sample_chain`, all interior
    /// samples materialised whether or not a leg lands on one) — and the sector's faces
    /// **tile contiguous runs** of that chain rather than fanning to a point.
    ///
    /// Each absorbed leg gets a profile-sample index by the §8.9.2 assignment (walking
    /// the absorbed legs in disk order: first half → `0`, second half → `segments`, and
    /// — for an odd absorbed-leg count — the single middle leg → `segments / 2`); the two
    /// flanking beveled edges take the arc ends (`0` for the leading flank, `segments` for
    /// the trailing). The indices are non-decreasing in disk order. Each sector face,
    /// bounded by two legs with indices `i ≤ j`, then rebuilds by substituting its apex
    /// corner with the inclusive sample run `s_i … s_j` — a single sample (triangle) when
    /// `i = j`, one extra vertex per interior sample the run crosses (a face straddling an
    /// un-pinned interior sample threads it rather than skipping it). A flank face's rim
    /// vertex on the beveled side is replaced by that edge's terminal boundvert through the
    /// ordinary beveled-edge restitch — not here — exactly as for `PATCH`.
    pub(super) fn restitch_multiface_arc_sector(
        &self,
        vertex: VertexKey,
        reflex_sector: &Sector,
        sample_chain: &[VertexKey],
        segments: u32,
        face_subst: &mut Vec<FaceSubstitution>,
    ) {
        // The sector's legs in disk order (leading flank, absorbed legs, trailing flank)
        // and the face bounded by each consecutive leg pair, `corners[i].1` between
        // `legs[i]` and `legs[i + 1]` — the same layout `restitch_multiface_patch_reflex`
        // consumes.
        let legs: Vec<EdgeKey> = reflex_sector
            .corners
            .iter()
            .map(|&(edge, _)| edge)
            .collect();
        let leg_count = legs.len();
        let seg = segments as usize;
        // A multi-face reflex sector is two flanking beveled edges plus ≥ 1 absorbed leg;
        // anything shorter is single-face and never reaches here. The sample chain is the
        // arc's `segments + 1` profile vertices `s_0 … s_segments`.
        if leg_count < 3 || sample_chain.len() != seg + 1 {
            return;
        }

        // §8.9.2 profile-sample index per leg. Flanks anchor the two arc ends; absorbed
        // legs split first-half → 0 / (odd) middle → segments/2 / second-half → segments.
        let absorbed_count = leg_count - 2;
        let first_half = absorbed_count / 2;
        let has_middle = absorbed_count % 2 == 1;
        let mut leg_sample: Vec<usize> = vec![0; leg_count];
        leg_sample[leg_count - 1] = seg; // trailing flank → B3 end
        for absorbed in 0..absorbed_count {
            let leg = absorbed + 1; // absorbed legs occupy legs[1 ..= leg_count - 2]
            leg_sample[leg] = if absorbed < first_half {
                0
            } else if has_middle && absorbed == first_half {
                seg / 2
            } else {
                seg
            };
        }

        for face_slot in 0..leg_count - 1 {
            let face = reflex_sector.corners[face_slot].1;
            // The face's earlier (disk-leading) leg carries the lower index `i`, the later
            // leg the higher `j`; the run replacing the apex is the inclusive `s_i … s_j`.
            let i = leg_sample[face_slot];
            let j = leg_sample[face_slot + 1];
            // Ascending run `[s_i … s_j]`, the order used when the face enters the apex
            // from its disk-leading leg (so `s_i` lands next to that leg's rim vertex).
            let forward: Vec<VertexKey> = sample_chain[i..=j].to_vec();
            let run = match self.face_corner_edges(face, vertex) {
                Some((incoming, _)) if incoming == legs[face_slot] => forward,
                _ => {
                    let mut reversed = forward;
                    reversed.reverse();
                    reversed
                }
            };
            set_face_substitution(face_subst, vertex, face, run);
        }
    }
}
