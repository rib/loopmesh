//! The square-profile [V-ADJ] grids (BEVEL.md §8.5.5.5(f.4a) / (f.5)).

use super::*;

impl LoopMesh {
    /// Build the **general-ring** square [V-ADJ] fill — the planar parallel-offset grid of
    /// BEVEL.md §8.5.5.5(f.5) (`≥ 3` beveled edges, **even** `segments`, square profile).
    ///
    /// This is not a smooth cap but the intersection pattern of parallel offset lines. Each
    /// grid sector `i` is the flat quad patch `B_i, C_i, v, C_{i−1}`: two of its sides are the
    /// straight rim rows (from `B_i` out to the two centreline starts), the far two are the
    /// centrelines `CL_i` / `CL_{i−1}` swept in to `v`, and its interior is a lattice of
    /// line-line meets between the two families of offset lines. The original corner `v`
    /// **survives un-rounded** as the exact innermost (welded) vertex — the whole point of a
    /// square profile.
    ///
    /// **Centreline start `C_i`.** The (f.5) seeding pass (a pass over boundverts, resolved by
    /// `square_ring_centreline_seeds`) lands each `C_i` at an exact **line-line meet on the
    /// shared beveled edge** — *not* at the sector chain's midpoint. The chain midpoint is the
    /// square cross-section's fold on the profile *rim*, which is off the beveled edge; the
    /// real seed is the point on the edge where the two offset-line families cross. `C_i`
    /// feeds both the interior lattice (`AdjInterior::Parallel`) **and** the grid's two
    /// straight rim rows: (f.5) pins those rows as straight runs `(i, 0, k) = B_i + (k/h)·(C_i
    /// − B_i)` and `(i, j, 0) = B_i + (j/h)·(C_{i−1} − B_i)`, landing on the seed rather than
    /// following the reused profile-fold chain. So the rim chain verts keep their keys — the
    /// cap still shares its rim with the arriving strips, and `adj_grid_vert` still returns
    /// those keys for `j == 0` / `k == 0` — but their fold-chain **positions are overwritten**
    /// onto the seed runs here, before the grid is emitted. The arc-start / patch-start seeds
    /// (cases 1 & 2 of (f.5), which need the square-profile outer miter to fire) are out of
    /// scope here; reflex / straight sectors take pass 2's projection.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn build_adj_fill_square_ring(
        &mut self,
        cut: &JunctionCut,
        chains: &[EdgeEndChain],
        segments: u32,
        profile: f32,
        material: i32,
        templates: &TemplateBank,
        slots: &mut AdjNetSlots,
        result: &mut BevelResult,
    ) -> bool {
        let sectors = cut.boundverts.len();
        let seg = segments as usize;
        let half = seg / 2;
        if sectors < 3 || seg < 2 || seg % 2 != 0 {
            return false;
        }

        let mut chain_of: Vec<Vec<VertexKey>> = Vec::with_capacity(sectors);
        for i in 0..sectors {
            match self.oriented_chain(cut, chains, i) {
                Some(chain) if chain.len() == seg + 1 => chain_of.push(chain),
                // A sector that did not sample to `seg + 1` verts (a collinear / degenerate
                // chain) is outside scope; skip the whole fill rather than emit a torn patch.
                _ => return false,
            }
        }

        let vertex_position = self.vertex_position(cut.vertex);
        // Per sector: (`B_i`, `C_i`). `B_i` is the sector's opening boundvert; `C_i` is the
        // §8.5.5.5(f.5) centreline start, the exact line-line meet on the shared beveled edge
        // (see the doc note) — resolved by the seeding pass, not read off the rim.
        let boundvert_positions: Vec<Vec3> = (0..sectors)
            .map(|i| self.vertex_position(chain_of[i][0]))
            .collect();
        let seeds = square_ring_centreline_seeds(
            &cut.opening_edge_directions,
            &cut.opening_edge_tips,
            &boundvert_positions,
            &cut.boundvert_non_reflex,
            vertex_position,
            // The plain junction cut carries no miter, so no arc/patch start — reflex
            // sectors take pass 2 (cases 1 & 2 are the miter fill's, below).
            &[],
        );
        let starts: Vec<(Vec3, Vec3)> = boundvert_positions.iter().copied().zip(seeds).collect();

        // §8.5.5.5(f.5): the grid's two straight rim rows run `B_i → C_i` and `B_i → C_{i−1}`
        // as straight segments to the line-line seed, *not* along the reused profile-fold
        // chain (whose fold midpoint sits off the beveled edge). The rim verts keep their keys
        // (`adj_grid_vert` still returns `chain_of[i][k]` / `chain_of[i−1][seg−j]` for the
        // boundary), so the cap stays welded to the arriving strips — only their positions are
        // moved onto the seed runs. The interior lattice is already cast from these same runs.
        // Each rim row covers half a chain: sector `i`'s `j = 0` row lands `chain_of[i][0..=h]`
        // and its `k = 0` column lands `chain_of[i−1][h..=seg]`; the shared fold `chain_of[i][h]
        // = C_i` is written from both sides to the same seed, and the boundverts (`step = 0`)
        // are left untouched.
        for i in 0..sectors {
            let (b_i, c_i) = starts[i];
            let previous = (i + sectors - 1) % sectors;
            let c_prev = starts[previous].1;
            for step in 1..=half {
                let t = step as f32 / half as f32;
                // `j = 0` rim row: `(i, 0, step) = B_i + t·(C_i − B_i)`.
                self.set_vertex_position(chain_of[i][step], b_i + t * (c_i - b_i));
                // `k = 0` rim column: `(i, step, 0) = B_i + t·(C_{i−1} − B_i)`, whose vert is
                // the previous sector's chain read back from `B_i` (`chain_of[i−1][seg−step]`).
                self.set_vertex_position(chain_of[previous][seg - step], b_i + t * (c_prev - b_i));
            }
        }

        let sector_representatives = cut.resolve_sector_representatives(templates);
        let centre_column_representatives = cut.resolve_centre_column_representatives(templates);
        let central_representative = templates.resolve(cut.central_representative);
        self.emit_adj_grid(
            sectors,
            seg,
            half,
            true, // even `segments` — the innermost vert welds at `v`
            &chain_of,
            &AdjInterior::Parallel {
                starts: &starts,
                vertex: vertex_position,
            },
            material,
            &sector_representatives,
            &centre_column_representatives,
            central_representative,
            None,
            None,
            slots,
            result,
        );
        true
    }

    /// Close a §8.5.5.5(f.4a) square-in weld — the **entire** face output of a square-in
    /// tri-corner, which is at most one polygon.
    ///
    /// The chains were already laid as `s` welded legs at layout time
    /// ([`Self::build_square_in_weld_chains`]), so there is no interior grid and no ring
    /// quad to emit here: the arriving strips already meet along every leg edge. What is
    /// left is the parity of §8.5.5.5(e), read off the legs:
    ///
    /// - **even** `segments`: the innermost sample of every leg is the octant origin, one
    ///   shared vertex, so the legs meet at a point and the corner is already complete —
    ///   **zero** faces;
    /// - **odd** `segments`: no sample reaches the origin, so the `s` innermost leg verts
    ///   `(i, 0, ⌊n/2⌋)` are distinct and exactly **one** central `s`-gon closes the hole.
    ///
    /// The `s`-gon's **winding** is §5.1's manifold condition, exactly as the grid cap's is,
    /// because its own edges are rim chain edges: the weld's identification makes
    /// `chain_i[⌊n/2⌋ + 1]` the same vertex as `chain_(i+1)[⌊n/2⌋]`, so the step from
    /// `central[i]` to `central[i+1]` *is* chain `i`'s ascending step across the turn — the
    /// corner-cutting chord, carrying the one strip face that arrived along that chain. So the
    /// ring hands straight to [`Self::patch_rim_needs_reversing`], with no interior point to
    /// orient against and nothing geometric left in the decision.
    ///
    /// This fill emits **no** interior net, so it cannot leave a net slot to replant and
    /// answers `false` to its caller's question. (A pipe junction cannot reach this row
    /// anyway — (f.1) tests the pipe one step earlier.)
    pub(super) fn build_adj_fill_square_in_weld(
        &mut self,
        cut: &JunctionCut,
        chains: &[EdgeEndChain],
        segments: u32,
        material: i32,
        templates: &TemplateBank,
        result: &mut BevelResult,
    ) -> bool {
        let sectors = cut.boundverts.len();
        let seg = segments as usize;
        if seg < 2 || seg % 2 == 0 {
            return false;
        }
        let half = seg / 2;

        // Invariant, not a guard on input: the same phase that decided this junction welds
        // laid one chain per sector, each holding one vert per segment boundary.
        let mut central: Vec<VertexKey> = Vec::with_capacity(sectors);
        for i in 0..sectors {
            let chain = self
                .oriented_chain(cut, chains, i)
                .expect("a welded junction has one chain per sector");
            assert_eq!(
                chain.len(),
                seg + 1,
                "a welded chain holds one vert per segment boundary",
            );
            central.push(chain[half]);
        }
        // Re-read the step across the turn only where the assertion that consumes it is
        // compiled; a release build must not carry a vector for a debug-only reader.
        if cfg!(debug_assertions) {
            for i in 0..sectors {
                let across_the_turn = self
                    .sector_chain(cut, chains, i)
                    .and_then(|chain| chain.oriented_sample(cut.boundverts[i], half + 1));
                debug_assert_eq!(
                    across_the_turn,
                    Some(central[(i + 1) % sectors]),
                    "the weld identifies chain {i}'s step across the turn with the next sector's \
                     innermost leg vert — which is what makes the `s`-gon's edges rim edges",
                );
            }
        }

        let reverse = self.patch_rim_needs_reversing(
            (0..sectors).map(|i| (central[i], central[(i + 1) % sectors])),
        );

        // §6.6.1: the centre `s`-gon is a ring-spanning face — its corners span the whole
        // boundvert ring — so it takes the ring-spanning donor, exactly as the general grid's
        // odd centre does.
        let central_representative = templates.resolve(cut.central_representative);
        self.emit_patch_face(&central, reverse, material, central_representative, result);
        false
    }
}

/// Resolve the §8.5.5.5(f.5) **centreline seeds** `C_i` for a general-ring square
/// parallel-offset grid — one point per grid sector, each the exact line-line meet the
/// seeding pass lands on, *not* the profile chain's midpoint (which lies off the beveled
/// edge). The seeds feed only the interior lattice; the rim stays the reused chain.
///
/// The pass is over **boundverts**, not sectors (BEVEL.md §8.5.5.5(f.5)): `edge_in(i)` is
/// `opening_edges[i]` and `edge_out(i)` is `opening_edges[(i + 1) % s]`, the edge shared with
/// the next boundvert, along which sector `i`'s centreline `CL_i` rides.
///
/// - **Pass 1, case 3 (parallel-offset proposals).** A boundvert `B_i` that a patch/arc did
///   not swallow proposes a candidate for **both** adjacent centrelines: for `CL_i` (the sector
///   after it), the foot — on `edge_out(i)`'s own line — of the line through `B_i` parallel to
///   `edge_in(i)`; for `CL_{i−1}` (the sector before it), the foot on `edge_in(i)`'s line of
///   the line through `B_i` parallel to `edge_out(i)`. A (anti)parallel companion proposes
///   nothing — its foot is at infinity — so a boundvert may propose for one centreline and not
///   the other. Each centreline is meant to receive one proposal from *each* of its two
///   flanking boundverts: the foot's parameter along the edge is `offset / sin θ` on a planar
///   disk sector of swept angle `θ`, which diverges at **both** ends of the range, so on a
///   near-flat sector's centreline the two proposals sit far apart and dropping either
///   misplaces the seed by a whole fraction of the spoke.
/// - **Pass 1, case 4 — the gate (`non_reflex`).** The proposal runs only where §8.9.1's
///   three-way signed test read `B_i`'s **own** disk sector — the wedge at `v` between
///   `edge_in(i)` and `edge_out(i)` — as non-reflex and not straight. Reflex, straight and
///   degenerate boundverts propose nothing and leave their centrelines to pass 2. This is a
///   **gate on the classification, evaluated before any foot is computed**, not a filter on
///   the computed foot: at a reflex corner the foot is perfectly well defined but sits at
///   `offset / sin θ < 0`, i.e. *behind* `v` and off the beveled-edge segment, and — being
///   near `v` — would win the arbitration below and drag the whole centreline through the
///   junction. The verdict is the caller's (`boundvert_non_reflex`) because it must be read in
///   §8.9.1's genuine incident-face frame off the pristine corner; a flattened working-plane
///   substitute misreads near-flat corners as reflex and strands their centrelines instead.
/// - **Arbitration.** A centreline can receive a proposal from each of its two bounding
///   boundverts (two points on the same edge line); the one **closer to `v`** wins. A
///   centreline receiving only one proposal takes it however far out it lands.
/// - **Pass 2 (no-arc branch).** A centreline left unseeded — a sector both of whose flanking
///   boundverts fell to case 4, or whose companion lines were (anti)parallel — projects the
///   sector's **later** boundvert onto its shared beveled-edge **segment** (`v → edge tip`),
///   clamped to the segment; with no shared edge it falls to the midpoint. The clamp is what
///   keeps a reflex sector's centreline in front of `v`.
///
/// `arc_patch` carries the enlarged miter ring's arc-or-patch starts (§8.5.5.5(f.5) cases 1
/// & 2). There are **two tagging paths** and only the first is unique per ring: the *outer*
/// miter tags at most one (§8.9.1's single reflex sector), while the *inner* miter tags one
/// per firing non-reflex corner, so a ring can carry `m` arc starts. The cases apply **per
/// start** — the starts do not interact except where two are adjacent, which is what makes
/// pass 2's two-arc branch live.
///
/// - **Case 1 (`Patch`).** `B_1` seeds **two** centrelines from plain boundvert-pair
///   midpoints — `C_{B_1} = mid(B_1, B_2)`, `C_{B_2} = mid(B_2, B_3)` — and the pass skips
///   over `B_2`, `B_3` (no sharp evaluation), resuming at `B_4`. Outer-only: `PATCH` and
///   `ARC` coincide at an inner corner (§8.9.3).
/// - **Case 2 (`Arc`).** `B_1` seeds **one** centreline from the arc's stored §7.6.2 middle
///   control point; the pass skips `B_3` and resumes at `B_2` (`= B_1 + 1` on the ring).
///
/// Both seeds enter the same closer-to-`v` arbitration as a sharp proposal. Pass 2 then
/// gains the arc-aware branches: the sector *immediately after* an arc (predecessor `= B_1`)
/// projects its **earlier** boundvert onto the shared edge, and a sector flanked by **two**
/// arc starts meets the shared edge's line with the line through its two boundverts.
pub(super) fn square_ring_centreline_seeds(
    directions: &[Vec3],
    tips: &[Vec3],
    boundverts: &[Vec3],
    non_reflex: &[bool],
    vertex: Vec3,
    arc_patch: &[ArcPatchStart],
) -> Vec<Vec3> {
    let sectors = boundverts.len();
    debug_assert_eq!(non_reflex.len(), sectors);
    let mut seeds: Vec<Option<Vec3>> = vec![None; sectors];
    // Boundverts a patch / arc swallowed — they get no sharp (case-3) evaluation in pass 1.
    // This is the whole of the skip-ahead (§8.5.5.5(f.5) case 1/2 "the pass resumes at
    // `B_{i+2}` / `B_{i+3}`"): only the *consumed boundverts* are held out of the case-3
    // proposal loop. The sector `B_3` opens is **not** separately suppressed — its trailing
    // bounding boundvert still proposes for that sector under the usual arbitration. Pass 2's
    // arc branch is the *fallback* for when that trailing boundvert proposes nothing.
    let mut consumed = vec![false; sectors];

    // Cases 1 & 2 (§8.5.5.5(f.5)): each tagged leading boundvert `B_1` seeds its arc/patch
    // centrelines first, so they take part in pass 1's closer-to-`v` arbitration. A ring
    // carries at most one *outer* start but any number of *inner* ones, and the cases apply
    // per start.
    let is_arc_start = |index: usize| {
        arc_patch
            .iter()
            .any(|start| start.mode == BevelMiter::Arc && start.index == index)
    };
    for start in arc_patch {
        let b1 = start.index;
        let b2 = (b1 + 1) % sectors;
        let b3 = (b1 + 2) % sectors;
        match start.mode {
            BevelMiter::Patch => {
                seeds[b1] = Some(0.5 * (boundverts[b1] + boundverts[b2]));
                seeds[b2] = Some(0.5 * (boundverts[b2] + boundverts[b3]));
                consumed[b2] = true;
                consumed[b3] = true;
            }
            BevelMiter::Arc => {
                // The arc's two-vert ring is `B_1 → B_3`, so `B_3` is `b1 + 1` on the ring.
                seeds[b1] = Some(start.arc_middle);
                consumed[b2] = true;
            }
            BevelMiter::Sharp => {}
        }
    }

    // Pass 1 — parallel-offset proposals (case 3), under case 4's classification gate. The
    // arc/patch start and every boundvert a patch or arc consumed are skipped, so they never
    // propose. Of the rest, only a boundvert whose own disk sector came back **non-reflex and
    // not straight** proposes — for *both* of its adjacent centrelines, with the closer-to-`v`
    // arbitration keeping the survivor.
    //
    // Two reasons the gate must precede the foot and not filter it. A reflex corner's foot is
    // well defined but lands at a negative parameter, behind `v` and off the beveled-edge
    // segment; "closer to `v`" does not exclude it, so it would win. And the two readings
    // ("skip reflex" vs "reject a negative parameter") only coincide on a planar junction —
    // the classification is the one that holds off-plane. A boundvert whose companion line is
    // (anti)parallel to the edge proposes nothing on that side either — the foot is at
    // infinity — and either suppression drops the centreline to pass 2.
    for i in 0..sectors {
        if consumed[i] {
            continue;
        }
        if arc_patch.iter().any(|start| start.index == i) {
            continue;
        }
        if !non_reflex[i] {
            continue;
        }
        let edge_in = directions[i];
        let edge_out = directions[(i + 1) % sectors];
        let boundvert = boundverts[i];
        if let Some(candidate) = line_line_nearest_on_first(vertex, edge_out, boundvert, edge_in) {
            seed_closer_to_vertex(&mut seeds[i], candidate, vertex);
        }
        let previous = (i + sectors - 1) % sectors;
        if let Some(candidate) = line_line_nearest_on_first(vertex, edge_in, boundvert, edge_out) {
            seed_closer_to_vertex(&mut seeds[previous], candidate, vertex);
        }
    }

    // Pass 2 — fill the still-unseeded centrelines from the sector's shared edge segment.
    for i in 0..sectors {
        if seeds[i].is_some() {
            continue;
        }
        let next = (i + 1) % sectors;
        let previous = (i + sectors - 1) % sectors;
        let prev_is_arc = is_arc_start(previous);
        let next_is_arc = is_arc_start(next);
        let seed = if directions[next] == Vec3::ZERO {
            // No shared beveled edge to project onto.
            0.5 * (boundverts[i] + boundverts[next])
        } else if prev_is_arc && next_is_arc {
            // A sector flanked by two arcs. The *outer* miter cannot supply this (§8.9.1
            // allows one reflex sector), but two adjacent firing **inner** miters do — the
            // sector between them runs from the first arc's partner `B_3` to the second
            // arc's tagged `B_1`. Meet E's line with the line through the two boundverts:
            // a **skew** pair still answers with the closest-approach point *on E's line*
            // (the same site rule as case 3 — never the midpoint of the closest pair), and
            // only an (anti)parallel pair, which has no such point, falls to the midpoint.
            line_line_nearest_on_first(
                vertex,
                directions[next],
                boundverts[i],
                boundverts[next] - boundverts[i],
            )
            .unwrap_or_else(|| 0.5 * (boundverts[i] + boundverts[next]))
        } else if prev_is_arc {
            // The sector immediately after the arc: project the *earlier* boundvert (`B_3`).
            closest_point_on_segment(boundverts[i], vertex, tips[next])
        } else {
            // No-arc / general case: project the *later* boundvert.
            closest_point_on_segment(boundverts[next], vertex, tips[next])
        };
        seeds[i] = Some(seed);
    }

    seeds
        .into_iter()
        .enumerate()
        .map(|(i, seed)| {
            seed.unwrap_or_else(|| 0.5 * (boundverts[i] + boundverts[(i + 1) % sectors]))
        })
        .collect()
}

/// Keep the centreline proposal closer to `v` (BEVEL.md §8.5.5.5(f.5) arbitration): the
/// two bounding boundverts of a sector each propose a point on the shared edge line, and the
/// survivor is whichever is nearer the corner.
pub(super) fn seed_closer_to_vertex(slot: &mut Option<Vec3>, candidate: Vec3, vertex: Vec3) {
    match slot {
        None => *slot = Some(candidate),
        Some(existing) => {
            if (candidate - vertex).length_squared() < (*existing - vertex).length_squared() {
                *existing = candidate;
            }
        }
    }
}

/// BEVEL.md §8.5.5.5(f.4a)'s leg step `b`, quoted: `2/n` at even `n`, `2/(2·⌊n/2⌋ + √2)` at
/// odd `n` — the closed-form solution of §8.6.4 step 3's closing-chord term on two straight
/// unit legs. The odd step is the tighter one (`√2 > 2·½`), which is what leaves room for the
/// single corner-cutting chord across the turn.
pub(super) fn square_in_weld_leg_step(segments: u32) -> f32 {
    let half = (segments / 2) as f32;
    if segments % 2 == 0 {
        2.0 / segments as f32
    } else {
        2.0 / (2.0 * half + std::f32::consts::SQRT_2)
    }
}

/// Whether §8.5.5.5(f.1) **step 1** claims this junction for (f.5)'s planar parallel-offset
/// ring grid: the `square` sentinel, **not a custom profile**, at least three beveled edges,
/// and an even `segments ≥ 2`.
///
/// Step 1 outranks step 2, and the ordering is part of the dispatch — "a square profile at a
/// pipe corner satisfies both the square and pipe tests, **and the square grid wins**". So this
/// is asked three times: by [`LoopMesh::build_adj_fill_base`] to route the grid, by
/// [`LoopMesh::build_adj_fill`] to suppress the (f.6) replant that would otherwise post-pass it,
/// and by [`LoopMesh::build_adj_fill_miter_ring`] to route the mitered ring's own fill.
/// A junction the ring claims never enters step 2 at all.
///
/// `custom` is step 1's own `not a custom profile` conjunct, and it is not the same test as the
/// tri-corner predicate's refusal even though both exclude the same profiles: a custom curve has
/// no superellipse exponent at all, so it can never *reach* the square sentinel by value — the
/// conjunct is what keeps a caller that resolves `exponent` from the `profile` slider (which a
/// custom bevel still carries) from routing the ring here on a scalar the curve does not use.
///
/// The count is the **ring's**, not the junction's beveled-edge count: they coincide at an
/// unmitered junction, and a miter only raises the ring (§8.9.6), so the `≥ 3` floor reads the
/// same way at either caller.
pub(super) fn adj_square_ring_claims(
    exponent: f32,
    boundverts: usize,
    segments: u32,
    custom: bool,
) -> bool {
    exponent == PROFILE_SQUARE_EXPONENT
        && !custom
        && boundverts >= 3
        && segments >= 2
        && segments % 2 == 0
}
