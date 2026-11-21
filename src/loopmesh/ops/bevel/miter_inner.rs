//! The inner miter (BEVEL.md §8.9.3).

use super::*;

impl LoopMesh {
    /// Build a junction's **inner-miter** boundvert ring and its cross-section chains
    /// (BEVEL.md §8.9.3 / §8.9.5).
    ///
    /// Every **non-reflex and not straight** sector is rebuilt as an **arc start**
    /// (§8.9.3): its §7.6.2
    /// meet-point becomes the stored **arc midpoint**, and the sector gains two ring
    /// boundverts — `B1` keeping the opening edge's side and a new `B3` taking over the
    /// closing edge's side. Both are pushed `spread` away from the junction along their
    /// own beveled edge from that stored midpoint (§8.9.4), and the sector's rebuilt
    /// face(s) re-stitch onto the arc run `B1 → bulge → B3`. `miter_inner` only
    /// distinguishes sharp from non-sharp (§8.9.3), so `Arc` and `Patch` build the same
    /// arc here; the caller gates on that.
    ///
    /// A junction may also carry **one reflex sector** (§8.9.1). That sector belongs to
    /// `miter_outer`, so this builder handles it only when the outer miter is `Sharp`:
    /// the reflex sector then keeps its single sharp meet-point boundvert and the ring is
    /// the mixed `s + (split sectors)` one the general valence-`s` cap fills. With a
    /// non-sharp `miter_outer` the whole junction is declined here and
    /// [`Self::build_junction_outer_miter`] owns it.
    ///
    /// The arc build is segments-general (§8.9.5.1): each qualifying sector's `B1 → B3`
    /// internal edge carries the `segments + 1`-sample inscribed arc, so odd `segments`
    /// closes the enlarged ring rather than declining. The inner miter splits **every**
    /// qualifying sector, so the central polygon's valence equals the **boundvert-ring
    /// valence** — one boundvert per unsplit sector plus two per split one, i.e. `2s` when
    /// every sector qualifies and one fewer per sector held back as a sharp reflex or a
    /// straight meet-point — **not** the outer miter's `s + 1` (§8.9.5.3 splits exactly one
    /// sector). A pipe junction is the shape that makes the straight class visible: its
    /// bare `180°` side is one exactly-straight sector, so an `s = 3` pipe ring goes to
    /// **5**, never 4 (§8.9.8's `#miter-never-pipe` derivation).
    ///
    /// The whole analytic band admits, the square sentinel included: §8.9.3 / §8.9.3.2(a)
    /// license no `profile` gate on the split, and the enlarged ring carries the §8.5.5.5(f.5)
    /// seed inputs so a square even-`segments` cap takes the parallel-offset lattice rather
    /// than declining.
    ///
    /// All classification runs read-only first; the method returns `false` **without
    /// mutating** for any junction outside its scope (`segments < 2`, fewer than three
    /// sectors, no qualifying sector at all (the miter is then inert and the enlarged ring
    /// *is* the sharp ring), more than one reflex sector, or a reflex sector the outer
    /// miter owns), so the caller falls back to the sharp ring.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn build_junction_inner_miter<'c>(
        &mut self,
        vertex: VertexKey,
        is_beveled: &dyn Fn(EdgeKey) -> bool,
        affect: BevelAffect,
        offset: f32,
        segments: u32,
        profile: f32,
        custom: Option<&'c CustomChains<'c>>,
        orientation: &CustomOrientation,
        spread: f32,
        miter_outer: BevelMiter,
        templates: &mut TemplateBank,
        chains: &mut Vec<EdgeEndChain>,
        face_subst: &mut Vec<FaceSubstitution>,
        miter_ring_junctions: &mut Vec<MiterRingJunction<'c>>,
        result: &mut BevelResult,
    ) -> bool {
        // §8.9.5: the inner miter rides the [V-ADJ] grid, so `segments ≥ 2`. The arc build is
        // segments-general and exponent-general (§8.9.5.1) — the same sampling the outer `ARC`
        // uses (§8.9.3 rebuilds the inner miter "exactly like an outer ARC") — so odd
        // `segments` closes on the enlarged boundvert ring (valence `2s` when every sector
        // fires, one fewer for each sector held back as a sharp reflex or a straight
        // meet-point — not the outer miter's `s + 1`) rather than declining to the sharp ring.
        //
        // The whole **analytic** band is admitted. §8.9.5's exponent-scope note makes the arc
        // chain read the run's own §8.6.4 spacing table at every `r`, and §8.9.3 rebuilds a
        // firing non-reflex sector "exactly like an outer ARC", so the sampler is
        // inner/outer-blind; §8.9.5's general-exponent reduction then shows the cap kernel
        // (the four averages, the sharpening mask, the Sabin weight, the single even-`2`
        // round) is exponent-independent — only the raw coarse mids and the (f.3) `fullness`
        // move, and both are already general here.
        //
        // The **square sentinel** is admitted too. §8.9.3 / §8.9.3.2(a) gate the inner miter on
        // three things only — a non-sharp `miter_inner`, the [V-ADJ] fill, and §8.9.1's
        // per-sector verdict — and the `profile` slider is not one of them. (f.1) step 1's
        // square-ring diversion changes how the **cap** is filled (the (f.5) planar
        // parallel-offset lattice), not whether the ring splits, and (f.5) case 2 reads an
        // *inner* arc start's stored midpoint exactly as it reads an outer one's. The seed
        // inputs are minted below, so the enlarged ring routes to that lattice at even
        // `segments` and to the general grid at odd, like any other ring.
        if segments < 2 {
            return false;
        }
        let corners = self.vertex_bevel_corner_ring(vertex, is_beveled);
        let sectors = sector_split(&corners, is_beveled);
        let sector_count = sectors.len();
        if sector_count < 3 {
            return false;
        }
        let vertex_position = self.vertex_position(vertex);

        // Read-only classification pass (§8.9.1): solve each sector's meet-point and its
        // two flanking edge directions, and take the three-way straight / reflex /
        // non-reflex verdict.
        struct SectorPlan {
            opening: EdgeKey,
            closing: EdgeKey,
            meet: Vec3,
            dir_open: Vec3,
            dir_close: Vec3,
            class: MiterSectorClass,
        }
        let mut plans: Vec<SectorPlan> = Vec::with_capacity(sector_count);
        for sector in &sectors {
            let (Some(&opening), Some(&closing)) =
                (sector.bounding_edges.first(), sector.bounding_edges.get(1))
            else {
                return false;
            };
            if sector.corners.is_empty() {
                return false;
            }
            let Some(verdict) = self.classify_miter_sector(vertex, vertex_position, sector) else {
                return false;
            };
            let meet = self.solve_meet_point(vertex, vertex_position, sector, offset);
            plans.push(SectorPlan {
                opening,
                closing,
                meet,
                dir_open: verdict.dir_open,
                dir_close: verdict.dir_close,
                class: verdict.class,
            });
        }

        // §8.9.1: a junction has at most one reflex sector. Two or more contradicts that
        // invariant, so decline rather than build an ill-defined ring.
        let reflex_count = plans
            .iter()
            .filter(|plan| plan.class == MiterSectorClass::Reflex)
            .count();
        match reflex_count {
            // §8.9.3 / §8.9.3.2(a) license **no** junction-level geometry gate on the
            // all-convex ring: the inner miter has no beveled-edge-count floor, no
            // `segments` floor, and no admission test beyond the per-sector
            // non-reflex-and-not-straight predicate.
            0 => {}
            // The reflex sector is `miter_outer`'s. Only when the outer miter is `Sharp`
            // does it stay a plain meet-point this builder can carry; otherwise the outer
            // builder owns the whole ring.
            1 => {
                if miter_outer != BevelMiter::Sharp {
                    return false;
                }
            }
            _ => return false,
        }

        // No sector qualifies, so the miter is inert here (§8.9.1's straight bullet) and
        // the enlarged ring *is* the sharp ring. Hand it back to the ordinary sharp path
        // rather than rebuild an identical ring through the miter machinery.
        if plans
            .iter()
            .all(|plan| plan.class != MiterSectorClass::NonReflex)
        {
            return false;
        }

        // Mutation pass: every check has passed. Each non-reflex, non-straight sector
        // splits into its arc pair; a straight sector and the reflex sector (if any) each
        // keep their single sharp meet-point.
        // Per sector: its ring boundverts in ring order (one when sharp, `B1`/`B3` when
        // split), the split sector's internal arc chain (`segments + 1` samples,
        // §8.9.5.1), and that arc's raw coarse mid. The arc cell is always symmetric
        // (`|meet − B1| = |meet − B3| = spread`), but symmetry buys the §8.2.2.1.1 boundary
        // resample = identity only at `segments = 2`: §8.9.5.2 warns a symmetric cell's
        // interior samples still move at non-power-of-two `segments`, so the resample runs
        // unconditionally (§8.9.5) and the raw coarse mid is captured only for the
        // `segments = 2` centre stencil.
        let mut sector_ring: Vec<Vec<VertexKey>> = Vec::with_capacity(sector_count);
        let mut sector_arc: Vec<Option<Vec<VertexKey>>> = Vec::with_capacity(sector_count);
        let mut sector_arc_profile: Vec<Option<RingChainProfile>> =
            Vec::with_capacity(sector_count);
        let mut sector_bulge: Vec<Option<Vec3>> = Vec::with_capacity(sector_count);
        for plan in &plans {
            if plan.class != MiterSectorClass::NonReflex {
                // §5.2.4: not yet minted from the corner.
                let boundvert = self.create_vertex(plan.meet);
                result.verts.push(boundvert);
                sector_ring.push(vec![boundvert]);
                sector_arc.push(None);
                sector_arc_profile.push(None);
                sector_bulge.push(None);
                continue;
            }
            // §8.9.3 / §8.9.4: `B1` and `B3` are pushed `spread` out along their own
            // beveled edge from the stored meet-point.
            let b1_position = plan.meet + plan.dir_open * spread;
            let b3_position = plan.meet + plan.dir_close * spread;
            // §8.9.3 / §8.9.5.1: the inner arc is rebuilt exactly like the outer `ARC` —
            // the run's own cross-section quadrant in the arc's (non-orthonormal)
            // unit-square frame of `B1` / `B3` / the stored meet-corner, sampled by the
            // §8.6.4 even-chord table and segments-general. The stored meet-point is the
            // circumscribing square's outer corner, not generally a point on the curve; the
            // `segments = 2` interior sample is `(2 − 2m)·chord_mid + (2m − 1)·meet` with
            // `m = √profile` read off the sampler — the `√2 − 1` bulge only at the circle.
            //
            // Analytic under a **custom** curve too, and that is the answer, not a gap:
            // §8.6.3.1(4).4's miter carve-out hands every arc-start / patch-start boundvert a
            // second, non-custom spacing pair, and §8.6.3.1(6) A.6 makes those the one site
            // that never reads the orientation walk's mark. `B1` is that arc start; threading
            // the curve in here would diverge. Its ring *successor* `B3` is not tagged, owns
            // the sector's closing beveled edge, and does read the curve — below. What the
            // carve-out withholds is the **curve**, not the slider: the second pair is the
            // analytic superellipse at the run's own `profile`, so the scalar still moves this
            // chain under `CUSTOM` exactly as it does under the analytic family.
            let arc = RingChainProfile::Arc {
                b1: b1_position,
                b3: b3_position,
                midpoint: plan.meet,
                profile,
            };
            // §8.5.5.5(d) / §8.9.5.3: the extraordinary-centre stencil consumes the arc's
            // raw coarse `(i, 0, 1)` mid, never the emitted rim mid. The inner arc cell is
            // symmetric, so its raw bulge is both; captured for the `segments = 2` centre.
            let raw_bulge = arc.raw_positions(2)[1];
            // §8.2.2.1.1 / §8.9.5.2: the arc is a ring-edge chain like any other, so its
            // emitted rim samples are the chord-fraction resample of the raw arc against the
            // profile at the source resolution. Symmetry makes this the identity only at
            // `segments = 2` (and powers of two); at odd `segments` even a symmetric cell's
            // interior samples move (§8.9.5.1 / §8.9.5.2), so the resample is routed
            // unconditionally precisely because odd `segments` displaces symmetric cells —
            // "do not gate this regime on skew".
            let raw_chain = arc.raw_positions(segments);
            let source_segments = boundary_source_resolution(segments);
            let emitted = if source_segments == segments {
                raw_chain
            } else {
                let source = arc.raw_positions(source_segments);
                chord_fraction_resample(&raw_chain, &source)
            };
            sector_arc_profile.push(Some(arc));
            // §5.2.4: not yet minted from the corner.
            let b1 = self.create_vertex(b1_position);
            result.verts.push(b1);
            let mut samples = Vec::with_capacity(segments as usize + 1);
            samples.push(b1);
            for position in emitted.iter().take(segments as usize).skip(1) {
                // §5.2.4: not yet minted from the corner.
                let sample = self.create_vertex(*position);
                result.verts.push(sample);
                samples.push(sample);
            }
            let b3 = self.create_vertex(b3_position);
            result.verts.push(b3);
            samples.push(b3);
            sector_ring.push(vec![b1, b3]);
            sector_arc.push(Some(samples));
            sector_bulge.push(Some(raw_bulge));
        }

        // Re-stitch each sector's face(s). A sharp sector collapses its corner onto the
        // lone boundvert (1-for-1). A split sector's single face threads the whole arc run
        // in its own winding; a split sector that absorbed unbeveled legs (§7.6.1) is a
        // **multi-face** sector, and its faces tile contiguous runs of the arc's profile
        // samples by the §8.9.2.2 assignment — the same rule the outer arc uses, since
        // §8.9.3 builds the identical arc.
        for (k, sector) in sectors.iter().enumerate() {
            match &sector_arc[k] {
                None => {
                    let boundvert = sector_ring[k][0];
                    for &(_, face) in &sector.corners {
                        set_face_substitution(face_subst, vertex, face, vec![boundvert]);
                    }
                }
                Some(arc) => {
                    let mut faces: Vec<FaceKey> = Vec::new();
                    for &(_, face) in &sector.corners {
                        if !faces.contains(&face) {
                            faces.push(face);
                        }
                    }
                    if faces.len() == 1 {
                        let face = faces[0];
                        let run = match self.face_corner_edges(face, vertex) {
                            Some((incoming, _)) if incoming == plans[k].opening => arc.clone(),
                            _ => {
                                let mut reversed = arc.clone();
                                reversed.reverse();
                                reversed
                            }
                        };
                        set_face_substitution(face_subst, vertex, face, run);
                    } else {
                        self.restitch_multiface_arc_sector(
                            vertex, sector, arc, segments, face_subst,
                        );
                    }
                }
            }
        }

        // The enlarged ring in disk order, with the beveled edge each ring edge crosses:
        // `None` on a split sector's internal arc edge (which crosses none — this is what
        // desynchronises the §6.7.5.3 walk, case (6)), `Some` on each strip cross-section.
        let mut ring: Vec<VertexKey> = Vec::with_capacity(sector_count + 2);
        let mut cross_edge: Vec<Option<EdgeKey>> = Vec::with_capacity(sector_count + 2);
        // §8.5.5.5(f.5) path-2 tagging: every split sector's leading `B_1` is an arc start,
        // seeding its own centreline from the sector's stored §7.6.2 meet-point (the
        // pre-reposition value — the `spread` slide moved `B_1` / `B_3`, never this). Unlike
        // the outer miter's single tag there may be `m` of them on one ring.
        let mut arc_patch_starts: Vec<(usize, BevelMiter, Vec3)> = Vec::new();
        let mut boundvert_non_reflex: Vec<bool> = Vec::with_capacity(sector_count + 2);
        // §8.9.3.1: a split sector's trailing boundvert `B_3` is the arc *end* — it **owns**
        // its trailing beveled edge (the sector's closing edge), so it takes a `Some`
        // cross-edge exactly like the outer miter's reflex-owning `B_3` and a lone sharp
        // meet-point. Only the arc *start* `B_1`, whose internal arc edge crosses no beveled
        // edge, is edge-less (`None`).
        for k in 0..sector_count {
            let boundverts = &sector_ring[k];
            for (slot, &boundvert) in boundverts.iter().enumerate() {
                if slot == 0 && boundverts.len() == 2 {
                    arc_patch_starts.push((ring.len(), BevelMiter::Arc, plans[k].meet));
                }
                ring.push(boundvert);
                boundvert_non_reflex.push(plans[k].class == MiterSectorClass::NonReflex);
                if slot + 1 < boundverts.len() {
                    cross_edge.push(None);
                } else {
                    // The sector's closing beveled edge opens the next sector, so it is
                    // the edge the strip cross-section leaving this sector crosses.
                    cross_edge.push(Some(plans[k].closing));
                }
            }
        }

        // One full `segments + 1`-sample chain per ring edge (oriented `ring[i] →
        // ring[i+1]`): the pre-minted arc chain on an internal edge, or the strip
        // cross-section across the shared beveled edge (§8.9.5.2), which is also
        // registered as the beveled edge's chain so the strip build joins onto it.
        let ring_count = ring.len();
        let mut arc_chains = sector_arc.iter().flatten();
        let mut arc_profiles = sector_arc_profile.iter().flatten();
        let mut ring_chains: Vec<Vec<VertexKey>> = Vec::with_capacity(ring_count);
        let mut ring_profiles: Vec<RingChainProfile> = Vec::with_capacity(ring_count);
        for i in 0..ring_count {
            let start = ring[i];
            let end = ring[(i + 1) % ring_count];
            match cross_edge[i] {
                None => {
                    let chain = arc_chains
                        .next()
                        .expect("every internal ring edge has a pre-minted arc chain");
                    ring_chains.push(chain.clone());
                    ring_profiles.push(
                        *arc_profiles
                            .next()
                            .expect("every internal ring edge has a pre-minted arc profile"),
                    );
                }
                Some(edge) => {
                    let rescue = self.mid_rescue(vertex, edge, &corners, is_beveled);
                    // §8.6.3.1(4).4: a ring flank is the cross-section of a beveled edge, owned
                    // by the ring corner it opens at (§8.6.3.1(6) A.0) — here the arc *end*
                    // `B3`, or a sharp reflex meet-point. Neither is tagged an arc / patch
                    // start, so neither takes the miter carve-out: the flank rides the custom
                    // curve, through A.3's mark on *this* edge-end like any ordinary sector.
                    let reverse = orientation.reverse(edge, vertex);
                    let read = custom.map(|chains| CustomChainRead::edge_sector(chains, reverse));
                    ring_profiles
                        .push(self.ring_flank_profile(start, end, edge, profile, read, rescue));
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
                    // Unreachable: the profile sampler emits exactly `segments + 1`
                    // positions regardless of the cell's geometry, so a short chain would
                    // be an internal-invariant violation — and we are past the read-only
                    // classification pass, so a graceful decline here would orphan the
                    // verts and substitutions already minted.
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

        // §8.5.5.5(d) / §8.9.5.3: the extraordinary-centre stencil consumes each ring
        // chain's *raw coarse* profile mid, never the emitted (resampled) rim mid — the
        // arc-flanking cross-sections of a spread-pushed ring are skewed cells, where the
        // two differ. The arc's own mid is symmetric, so its raw bulge is both. Only the
        // `segments = 2` centre stencil consumes these (every other resolution reads the
        // mids straight off each ring edge's generator profile), so populate them only
        // there — mirroring the outer miter.
        let mut bulges = sector_bulge.iter().flatten();
        let raw_coarse_mids: Vec<Vec3> = if segments == 2 {
            (0..ring_count)
                .map(|i| match cross_edge[i] {
                    None => *bulges
                        .next()
                        .expect("every internal ring edge has a pre-minted raw bulge"),
                    Some(edge) => self.coarse_flank_mid(
                        ring[i],
                        ring[(i + 1) % ring_count],
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
        // not one uniform rep — a two-material split-sector junction diverges otherwise. The
        // split sector's flank quads therefore stay asymmetric: mirror sectors painted from
        // different slots donate their own (§8.9.3.1).
        let sector_donors = self.miter_ring_sector_donors(vertex, &cross_edge);
        let sector_representatives: Vec<FaceTemplate> = sector_donors
            .iter()
            .map(|&donor| templates.register(donor))
            .collect();
        // §8.9.3.1: the odd-`segments` centre n-gon takes the **enlarged miter ring's first
        // boundvert's** rep — the §6.6.1.2 fan anchor of *this* ring, not the raw disk's — on a
        // seam-free junction (a material boundary raises no interpolation seam), and only a
        // genuine seam routes it to the junction-wide §6.6.1.4 tie-break.
        let central_representative = templates.register(self.miter_centre_ngon_donor(
            vertex,
            is_beveled,
            affect,
            segments,
            sector_donors.first().copied().flatten(),
            templates.interp_components(),
        ));
        // §6.6.2 / §8.9.3.1: at odd `segments` each sector's centre column resolves its own
        // donor — a seam-gated own-rep-vs-tie-break for a beveled sector (its `Some` cross-edge,
        // including the inner arc *end* `B_3`, which owns its trailing beveled edge), the
        // §6.6.1.4 tie-break unconditionally for the edge-less arc *start* `B_1` (its `None`
        // cross-edge). Routed identically to the outer miter. Computed pristine here so it
        // survives the re-stitch.
        let centre_column_donors = self.miter_ring_centre_column_donors(
            vertex,
            &cross_edge,
            templates.interp_components(),
        );
        let centre_column_representatives: Vec<FaceTemplate> = centre_column_donors
            .into_iter()
            .map(|donor| templates.register(donor))
            .collect();

        // §8.5.5.5(f.5) seed inputs. The seeding pass is over ring boundverts: `edge_in(i)` —
        // the beveled edge boundvert `i` shares with its *predecessor* — is `cross_edge[i − 1]`,
        // and is `None` on a split sector's internal arc edge, giving `Vec3::ZERO`. Identical
        // to the outer miter's; only which sectors split differs.
        let opening_edge_directions: Vec<Vec3> = (0..ring_count)
            .map(|i| match cross_edge[(i + ring_count - 1) % ring_count] {
                Some(edge) => self
                    .edge_direction_from(edge, vertex, vertex_position)
                    .unwrap_or(Vec3::ZERO),
                None => Vec3::ZERO,
            })
            .collect();
        let opening_edge_tips: Vec<Vec3> = (0..ring_count)
            .map(|i| match cross_edge[(i + ring_count - 1) % ring_count] {
                Some(edge) => {
                    let [a, b] = self.edge_vert_keys(edge);
                    let far = if a == vertex { b } else { a };
                    self.vertex_position(far)
                }
                None => Vec3::ZERO,
            })
            .collect();

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
}
