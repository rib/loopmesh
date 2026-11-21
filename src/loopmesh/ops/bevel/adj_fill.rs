//! The rounded [V-ADJ] junction fill: dispatch and the general fill (BEVEL.md §8.5).

use super::*;

impl LoopMesh {
    /// The rounded adjacency-grid fill at one junction, with the §8.5.5.5(f.6) **pipe**
    /// cross-section replant applied as a post-pass (BEVEL.md §8.5 / §8.5.5.5(f.6)).
    ///
    /// A *pipe* junction runs straight **through** `v` — two beveled edges are collinear,
    /// so the corner is a section of tube (a constant cross-section) rather than a rounded
    /// corner. Its fill is a two-step post-pass: build the ordinary base grid — the same
    /// octant / general dispatch this junction would take had the pipe test not fired
    /// ([`build_adj_fill_base`], step 1) — then re-plant the net's index-canonical `j >= 1`
    /// slots onto that constant cross-section (step 2). That slot block is *not* the same
    /// set as "the verts this fill allocated": through the net's seam aliasing it also
    /// reaches the second half of every ring chain ([`Self::replant_pipe_rim_window`]).
    /// Every non-pipe junction runs the base grid alone.
    ///
    /// Step 2's domain being the net's slots also decides what happens when step 1 declines:
    /// no net, no slots, so **neither** replant half runs. That case is not exotic — a
    /// degenerate chain, a singular corner map, or the square-in weld's ring all reach it —
    /// and the rim half is the one that matters there, since those verts are shared with the
    /// arriving strips and moving them with no cap to justify it distorts the junction.
    ///
    /// The pipe predicate is the §8.3.4.1 one captured on the cut (`pipe_override`), and so
    /// is the axis; the cross-section shape is the pipe boundvert's own profile, read back
    /// off its chain.
    ///
    /// **One junction the pipe test does not get.** (f.1) tests its four steps *in order*, and
    /// the order is load-bearing: a square profile at a pipe corner satisfies both the square
    /// and the pipe test, and the square grid wins. So step 1's parallel-offset ring
    /// ([`adj_square_ring_claims`]) is asked here, before the pipe, and a junction it claims
    /// never enters step 2 at all — its (f.5) grid stands with no post-pass over it. That
    /// verdict is computed once, up front, rather than left to the step-1 branch inside the
    /// base dispatch, so the replant and the routing cannot disagree about which grid is being
    /// built.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn build_adj_fill(
        &mut self,
        cut: &JunctionCut,
        chains: &[EdgeEndChain],
        segments: u32,
        profile: f32,
        custom: Option<&CustomChains>,
        material: i32,
        templates: &TemplateBank,
        result: &mut BevelResult,
    ) {
        // §8.5.5.5(f.6): a pipe junction re-plants its interior grid verts once the base
        // grid exists. The predicate rides the cut (same test as the §8.3.4.1 cutoff
        // override); the axis + profile are recovered below.
        //
        // A **planar** §8.3.4.1 junction — the flat anti-parallel / wire-disk configuration
        // the CUTOFF override was built for — replants like any other pipe: its profile plane
        // contains the pipe axis, so the projected control triple is collinear at every
        // station and every vert in the domain takes step 4's chord clamp. That clamp is the
        // identity on a circular profile (the cross-section never leaves the band) and moves
        // real verts on a concave one, so "flat ⇒ the replant is inert" holds only for the
        // former and is not a reason to decline.
        //
        // (f.1) step 1 is asked first and outranks the pipe test: a square profile at a pipe
        // corner satisfies both, and the square grid wins, so the (f.5) parallel-offset ring is
        // never post-passed. One verdict serves both the routing and the replant.
        let exponent = profile_to_exponent(profile);
        let pipe = if cut.pipe_override
            && !adj_square_ring_claims(exponent, cut.boundverts.len(), segments, custom.is_some())
        {
            self.pipe_cross_section(cut, chains)
        } else {
            None
        };
        // Only the base fill's own interior grid verts are appended to `result.verts` — the
        // rim (boundvert ring + chain samples) is reused from the strips and never pushed
        // here (§8.5.5). So every vert added from this point is a §8.5.2 index-canonical
        // slot with `j >= 1` *and* `k >= 1`, already deduplicated by the grid's canon map.
        // That is only half the replant's domain: the canonical block's `k = 0` column is a
        // ring chain, reached below.
        //
        // The slot walk (f.6.1) asks an implementation to drive the replant off is recorded
        // only when one will actually run, so every other junction pays a branch per interior
        // vert and no allocation.
        let interior_start = result.verts.len();
        let mut slots = match pipe {
            Some(_) => AdjNetSlots::recording(),
            None => AdjNetSlots::off(),
        };
        let net_emitted = self.build_adj_fill_base(
            cut,
            chains,
            pipe.is_some(),
            segments,
            profile,
            custom,
            material,
            templates,
            &mut slots,
            result,
        );
        let Some(pipe) = pipe else {
            return;
        };
        // Step 2 is a post-pass over **the net's** canonical slots, so a junction whose base
        // dispatch declined to tile anything (an out-of-scope exponent / ring, a degenerate
        // chain, a singular corner map) has no slots for it to walk — and in particular no
        // licence to move the rim, which at that point is still the strips' own §8.2.2.1.1
        // read-back with no cap to justify displacing it. Gating on step 1's outcome is what
        // makes the ordering rule structural instead of a comment.
        if !net_emitted {
            return;
        }
        debug_assert_eq!(
            slots.len(),
            result.verts.len() - interior_start,
            "the slot walk and the minted interior verts must stay in step",
        );
        for index in interior_start..result.verts.len() {
            let vertex = result.verts[index];
            let base = self.vertex_position(vertex);
            // §8.5.5.5(f.6.1): the `land_on_corner` override is a property of the vert's
            // canonical net slot, so it needs the walk's own `(i, j, k)` and not the mint
            // order it happens to arrive in.
            let land_on_corner = slots.slot(index - interior_start).is_some_and(|slot| {
                pipe_replant_lands_on_corner(slot, segments as usize, pipe.sectors)
            });
            let replanted = pipe_cross_section_replant(
                base,
                pipe.axis,
                &pipe.cell,
                pipe.placement,
                exponent,
                land_on_corner,
            );
            self.set_vertex_position(vertex, replanted);
        }
        self.replant_pipe_rim_window(cut, chains, segments as usize, &pipe, exponent);
    }

    /// Dispatch the rounded adjacency-grid fill at one junction (BEVEL.md §8.5).
    ///
    /// Three paths share the valence-`s` grid machinery, picked by the corner's
    /// geometry:
    ///
    /// - **Forced `segments = 2` 3-way cube corner (§8.5.1–§8.5.4):** a three-quad
    ///   pinwheel meeting at one closed-form welded central apex, snapped onto the
    ///   corner sphere (§8.5.2). Combinatorially forced ⇒ the only strict-isomorphism
    ///   path; handled inline below.
    /// - **Larger-`segments` 3-way cube corner (§8.5.5.1–§8.5.5.4 / §8.5.5.5(f.2)):** the
    ///   unit-octant grid (`build_adj_fill_octant`) — coarse net, subdivision, resample
    ///   and superellipsoid snap all in octant coordinates, mapped to the corner once at
    ///   the end. Runs for the circular default and for every genuinely curved exponent;
    ///   the circular sphere-fit is the special case of it, not the other way round.
    /// - **General valence-`s` corner (§8.5.5.5):** any junction without the
    ///   valence-3 orthonormal sphere — `s ≠ 3`, oblique, or irregular-ring — takes
    ///   the centre-fullness cap (`build_adj_fill_general_nonsymmetric`), which is
    ///   valence-generic. The even/odd centre parity (§8.5.5.5(e)) decides between a
    ///   welded centre vertex and a central `s`-gon for every `s`.
    ///
    /// Scope: every exponent at the general ring — §8.5.5.5(f.1) step 4 has no exponent
    /// gate, so the two square sentinels take that grid there as well (a square exponent
    /// on **odd** `segments`, which (f.5)'s lattice never sees, and square-in on any ring
    /// the (f.4a) weld does not own). The square grids of (f.5) keep their own routes
    /// (step 1's parallel-offset ring, the tri-corner three squares), and square-in at a
    /// corner the tri-corner predicate *fires* on takes (f.4a)'s degenerate weld from the
    /// step-4 branch below — a fill that emits at most one face and no net at all.
    /// `pipe_claims_junction` never coincides with that weld, because (f.1) tests the pipe at
    /// step 2, one step *before* the row the weld belongs to.
    /// The **pipe** grid (§8.5.5.5(f.6)) *is* built here — its base grid is this very
    /// dispatch, including the tri-corner shortcut, so a valence-3 pipe's base is the octant
    /// grid or (f.5)'s three squares rather than the fullness grid — with the cross-section
    /// replant applied by the [`Self::build_adj_fill`] wrapper as a post-pass at **every**
    /// exponent, the two sentinels included: the wrapper resolves each vert through (f.2)'s
    /// branch S there.
    ///
    /// Returns whether this fill left **interior net slots to replant**. Every out-of-scope
    /// ring, degenerate chain and singular corner map below leaves the junction untouched and
    /// answers `false`; so does the (f.4a) weld, which does emit a face but tiles no net. The
    /// pipe post-pass keys on that: its domain is the net's slots, so no slots means nothing
    /// to walk.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn build_adj_fill_base(
        &mut self,
        cut: &JunctionCut,
        chains: &[EdgeEndChain],
        pipe_claims_junction: bool,
        segments: u32,
        profile: f32,
        custom: Option<&CustomChains>,
        material: i32,
        templates: &TemplateBank,
        slots: &mut AdjNetSlots,
        result: &mut BevelResult,
    ) -> bool {
        let count = cut.boundverts.len();
        // The cube-corner closed-form sphere snap (§8.5.5.3) exists only at a
        // valence-3 orthonormal equal-offset corner; a valence-`s` (`s ≠ 3`)
        // junction has no such sphere and takes the general fullness cap below
        // (§8.5.5.5(e)/(f)), routed once the tri-corner trigger declines. Fewer than
        // three boundverts is not a fillable corner.
        if count < 3 {
            return false;
        }
        // §8.5.5.5(f.1) step 3: the symmetric equal-offset *orthonormal* cube corner at a
        // smooth exponent — every `r` bar the two squares, the flat chamfer (`r = 1`)
        // included — takes the tri-corner family: the `segments = 2` pinwheel below and the
        // unit-octant grid at `segments ≥ 3`. Every other junction — a valence-`s` (`s ≠ 3`)
        // corner, or an oblique / irregular-ring valence-3 corner (§7.6.6, e.g. a reflex
        // sector whose dihedral fold fails the tri-corner test) — routes to the general
        // fullness-relaxation grid of (a)–(e), which loopmesh fills at **every** exponent:
        // the circle (`r = 2`), the straight chord (`r = 1`, the `fullness = 0` planar
        // specialisation, §8.3.0.2.1), the whole genuinely curved band on either side of
        // them, and — step 4 gating on no exponent at all — the two square sentinels.
        // The **pipe** grid (§8.5.5.5(f.6)) is built too — its base grid is this dispatch,
        // with the cross-section replant applied as a post-pass. The one ring that takes
        // neither grid is square-in where the tri-corner predicate *fires*: it owes (f.4a)'s
        // weld, which has no interior at all; see step 4 below.
        // The general fill runs even at `segments = 2`: it subdivides its coarse net once
        // before resampling (§8.5.5.5(d)), so a declined-tri-corner `segments = 2` ring
        // gets the centre-vertex pinwheel cap rather than an unfilled hole. Only
        // `segments < 2` has no interior grid to fill.
        let exponent = profile_to_exponent(profile);

        // §8.5.5.5(f.1) step 1 — tested first, before the tri-corner gate, and the order is
        // load-bearing: a **square** exponent with `≥ 3` beveled edges and **even**
        // `segments` takes the planar parallel-offset ring grid (f.5), and step 1 claims
        // *every* such ring — a valence-3 square+even ring never reaches the three flat
        // squares below (its even-`n` weld is unreachable, (f.5)'s blockquote). With no outer
        // miter in scope the boundvert count equals the beveled-edge count, so it gates the
        // ring here. Odd `segments` and custom profiles fall through — the latter through
        // step 1's own `not a custom profile` conjunct, which the gate now carries.
        if adj_square_ring_claims(exponent, count, segments, custom.is_some()) {
            return self.build_adj_fill_square_ring(
                cut, chains, segments, profile, material, templates, slots, result,
            );
        }

        if !self.adj_tri_corner_trigger(cut, profile, custom.is_some()) {
            // §8.5.5.5(f.1) step 4 carries **no exponent gate**: the circle, the straight
            // chord, and the whole genuinely curved band between and beyond them all take
            // the general fullness grid of (a)–(e). The exponent enters that grid only
            // twice, and both entries are already exponent-generic: through the [S]-sampled
            // boundary chains the stencil re-pins against, and through the (f.3) `fullness`
            // centre seed. There is no snap on this fill at all, so nothing here needs a
            // closed form.
            //
            // "No exponent gate" is meant literally, and the two **square** sentinels
            // arrive here too. Step 1 above consumed square + `count ≥ 3` + even
            // `segments` (the (f.5) parallel-offset ring) and the tri-corner trigger
            // consumed the valence-3 square (the (f.5) three flat squares), so what
            // reaches this point at a square exponent is **odd** `segments` on a
            // non-tri-corner ring — which (f.5) itself says its lattice never sees
            // ("not reached on odd `segments` at all") and which therefore belongs to
            // step 4 like any other ring. Square-in reaches it whenever the corner is
            // not the (f.4a) weld's, and (f.4b) states that grid outright: the ordinary
            // (a)–(e) topology with the ring sampled as the concave profile and the
            // centre seeded by law F3's **negative** fullness. Both fall out of the two
            // exponent-generic entries above; neither needs a construction of its own.
            //
            // One ring is not step 4's at all: square-in at a corner the tri-corner family
            // owns — step 3's `EXP_SQUARE_IN` row, the (f.4a) degenerate weld, whose sectors'
            // chains are *identified* pairwise and which has no interior grid whatsoever. That
            // is a different object from this one (it deletes the ring quads this builds), and
            // the trigger cannot report it because it refuses square-in outright — hence the
            // verdict riding the cut from chain-layout time, where the identification was made.
            //
            // A **pipe** junction never carries that verdict, and that is dispatch order
            // rather than leniency: step 2 is tested before step 3, so a pipe never reaches the
            // weld's row at all. What it builds instead is the grid step 4 would have built
            // *including step 4's own tri-corner shortcut* — and that shortcut excludes
            // square-in explicitly, so a valence-3 square-in tube lands on the ordinary
            // (a)–(e) grid and is then re-planted onto its constant cross-section.
            //
            // A pipe at either **square** sentinel is likewise not declined: the replant
            // now has the branch it needs — (f.2)'s branch S, the axis push in the clamped unit
            // square plus the `land_on_corner` override keyed on the net's own `(i, j, k)`
            // slots — where before it would have resolved through a divergent `r`-norm.
            //
            // A **custom** profile reaches this point at *every* ring, orthonormal cube corner
            // included: the tri-corner predicate refuses it outright and step 1 excludes it, so
            // (f.1) leaves it exactly one destination. Its `r` is whatever the untouched
            // `profile` slider happens to resolve to and is read by nothing here — the curve
            // supplies the boundary chains and law F1 supplies the centre seed.
            debug_assert!(
                !(cut.square_in_weld && pipe_claims_junction),
                "(f.1) tests the pipe one step before the weld's row",
            );
            if cut.square_in_weld {
                return self.build_adj_fill_square_in_weld(
                    cut, chains, segments, material, templates, result,
                );
            }
            match segments {
                0 | 1 => return false,
                _ => {
                    return self.build_adj_fill_general_nonsymmetric(
                        cut, chains, segments, profile, custom, material, templates, slots, result,
                    );
                }
            }
        }
        // The forced `segments = 2` pinwheel (§8.5.1–§8.5.4) keeps its dedicated
        // closed-form path below — at `segments = 2` the coarse net *is* the target grid,
        // so no subdivision round runs on this path at all (§8.5.5.5(f.2), the PRESCALE
        // guard). `segments ≥ 3` takes the unit-octant subdivided grid.
        // `segments < 2` has no interior to fill.
        match segments {
            0 | 1 => return false,
            2 => {}
            _ => {
                // §8.5.5.5(f.5): the tri-corner **square** grid is three flat unit squares
                // (closed form, no subdivision, no snap). It is reached only on **odd**
                // `segments` — even+`s≥3` was already claimed by step 1's parallel-offset ring
                // above — so it always leaves the central-triangle hole the odd centre fills.
                if exponent == PROFILE_SQUARE_EXPONENT {
                    return self.build_adj_fill_square_tricorner(
                        cut, chains, segments, profile, material, templates, slots, result,
                    );
                }
                return self.build_adj_fill_octant(
                    cut, chains, segments, profile, material, templates, slots, result,
                );
            }
        }

        // Sector i's interior chain vert (i,0,1): the lone interior sample of the
        // chain that *starts* at B_i (B_i → B_{i+1}).
        let mut interior_of_sector: Vec<VertexKey> = Vec::with_capacity(count);
        for k in 0..count {
            let chain = self.sector_chain(cut, chains, k);
            match chain.and_then(|c| c.interior.first().copied()) {
                Some(interior) => interior_of_sector.push(interior),
                // A degenerate sector with no interior sample (collinear chain) is
                // outside the forced cube-corner regime; skip the whole fill.
                None => return false,
            }
        }

        let vertex_position = self.vertex_position(cut.vertex);
        let boundvert_positions = [
            self.vertex_position(cut.boundverts[0]),
            self.vertex_position(cut.boundverts[1]),
            self.vertex_position(cut.boundverts[2]),
        ];

        // §8.5.1 / §8.2.2.1.1: a corner admitted only by the dihedral band (oblique /
        // absorbed-leg / reflex, §7.6.6) is **not** symmetric, so its rim + apex are not the
        // profile-cell / circumsphere placement the symmetric octant sub-case shares with the
        // arriving strips. The cube-corner map (§8.5.2's four-point `unit_to_corner`) +
        // superellipsoid snap place every cap vert instead: sector `k`'s lone interior sample
        // rides the unit arc `‖e_k + e_{k+1}‖`, so the two reflex-flank samples share a world
        // coordinate pair — the map's signature (§7.6.6), not the profile-cell plane. The
        // symmetric corner keeps its exact existing placement below (the octant arc point and
        // the circumsphere apex coincide there, §8.5.2), so nothing on that path moves.
        const UNIT_AXES: [Vec3; 3] = [Vec3::X, Vec3::Y, Vec3::Z];
        let oblique_apex = if tri_corner_meet_geometry(vertex_position, &boundvert_positions) {
            None
        } else {
            match self.unit_octant(cut) {
                Some(octant) => {
                    for k in 0..count {
                        let arc_mid = (UNIT_AXES[k] + UNIT_AXES[(k + 1) % count]).normalize();
                        let placed =
                            octant.to_world(snap_to_unit_superellipsoid(arc_mid, exponent, false));
                        self.set_vertex_position(interior_of_sector[k], placed);
                    }
                    Some(octant.to_world(snap_to_unit_superellipsoid(
                        Vec3::splat((1.0_f32 / 3.0).sqrt()),
                        exponent,
                        false,
                    )))
                }
                // A degenerate four-point map cannot build the cube-corner cap; fall back
                // to the symmetric apex computation below so the fill still closes.
                None => None,
            }
        };

        let sector_representatives: Vec<FaceTemplateRef<'_>> = cut
            .sector_representatives
            .iter()
            .map(|&template| templates.resolve(template))
            .collect();

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
        //
        // At any **non-circular** smooth exponent there is no such sphere (§8.5.5.5(f.2)) —
        // that includes the flat chamfer `r = 1`, whose surface is a plane — so the apex is
        // the octant closed form instead: the body-diagonal point of the unit octant snapped
        // onto the unit superellipsoid is `3^(−1/r)·(1,1,1)` for every `r`, and
        // `unit_to_corner` carries it to the corner (at `r = 1` that is the boundvert
        // centroid, on the chamfer plane). At `r = 2` the two agree — `3^(−1/2)` *is*
        // `c + R·normalize(v − c)` read in octant coordinates — and the circular corner keeps
        // the sphere fit, which is the strictly-pinned forced case (§8.5.4).
        let apex_position = if let Some(position) = oblique_apex {
            position
        } else if exponent != 2.0 {
            match self.unit_octant(cut) {
                Some(octant) => octant.to_world(Vec3::splat(3.0_f32.powf(-1.0 / exponent))),
                None => return false,
            }
        } else {
            let on_surface = [
                boundvert_positions[0],
                boundvert_positions[1],
                boundvert_positions[2],
                self.vertex_position(interior_of_sector[0]),
            ];
            match circumsphere(on_surface) {
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
            }
        };
        // §5.2.4: not yet minted from the corner.
        let apex = self.create_vertex(apex_position);
        result.verts.push(apex);
        // The pinwheel's one interior vert is the net's welded apex `(0, 1, 1)` — the
        // `segments = 2` instance of §8.5.2's canonical `(0, h, h)`. Recording it keeps the
        // slot walk aligned with `result.verts` on this closed-form path too.
        slots.record((0, 1, 1));

        // The pinwheel's rim is the general grid's rim at `seg = 2` and nothing else. Sector
        // `k`'s chain is `B_k → (k,0,1) → B_(k+1)`; quad `k` walks its **first** step as the
        // cell's `j = 0` row (`(k,0,0) → (k,0,1)`) and quad `k+1` walks its **second** as the
        // cell's closing `k = 0` column step (`(k+1,1,0) → (k+1,0,0)`, which is `(k,0,1) →
        // B_(k+1)`). Between them the `2·count` chain edges are each walked exactly once and
        // **ascending** — the disk order of §5.1, the same one [`Self::adj_grid_needs_reversing`]
        // hands the subdivided grid. So the whole pinwheel takes **one** aggregate §5.1 bit,
        // read here before any cap face exists so that no quad's vote can be polluted by a
        // sibling's, rather than a per-quad probe that a §7.8 input could split.
        let pinwheel_rim_edge = |i: usize, k: usize| {
            if k == 0 {
                (cut.boundverts[i], interior_of_sector[i])
            } else {
                (interior_of_sector[i], cut.boundverts[(i + 1) % count])
            }
        };
        self.debug_assert_rim_agrees_within_each_sector(count, 2, pinwheel_rim_edge);
        let reverse = self.patch_rim_needs_reversing(
            (0..count).flat_map(|i| (0..2).map(move |k| pinwheel_rim_edge(i, k))),
        );

        // §8.5.3: one pinwheel quad per sector, all sharing the apex. Sector i's
        // quad is B_i → (i,0,1) → apex → (i−1,0,1).
        for i in 0..count {
            let before = (i + count - 1) % count;
            let quad = [
                cut.boundverts[i],
                interior_of_sector[i],
                apex,
                interior_of_sector[before],
            ];
            // §6.6.2's grid row: this `segments = 2` pinwheel quad is sub-patch `i`'s lone
            // welded cell, so it takes sector `i`'s **own** boundvert rep — not one
            // corner-wide donor. Even `segments` has no centre column to re-scope.
            self.emit_patch_face(&quad, reverse, material, sector_representatives[i], result);
        }
        true
    }

    /// Emit one **general valence-`s` [V-ADJ] adjacency fill** (BEVEL.md §8.5.5.5) over a
    /// prepared ring of per-edge profile chains. Each `chain_of[i]` is the `segments +
    /// 1`-sample chain on ring edge `i` (oriented `B_i → B_{i+1}`); the two chain ends are
    /// the ring boundverts. `profiles[i]` is that ring edge's *generator* — the fill
    /// re-samples it at each subdivision round's own doubled resolution (§8.5.5.5(c)) and,
    /// at the coarse resolution, reads its raw `(i, 0, 1)` control point. `flat` selects
    /// the `r = 1` planar specialisation (fullness zeroed, centre on the ring-plane
    /// centroid, §8.3.0.2.1): only the **even** cap at `segments = 2` — a bare welded apex
    /// with no interior ring — takes the closed-form bilinear blend. Every other flat cap
    /// (all odd, and even `segments ≥ 4`, whose interior grid is a determined quantity)
    /// descends through the same subdivision-limit relaxation the curved caps use, only with
    /// the `fullness = 0` seed (§8.5.5.5(e) / (f.3) law **F2**). This is the shared
    /// core of the general non-symmetric junction fill and the miter cap fills (§8.9.5.3):
    /// the only per-caller difference is where the chains come from.
    ///
    /// Returns whether a grid was tiled; a malformed ring (too few sectors, a chain that
    /// did not sample to `segments + 1`) leaves the junction untouched and reports `false`,
    /// which is what lets the §8.5.5.5(f.6) pipe post-pass tell "there is a net" from
    /// "there is not".
    #[allow(clippy::too_many_arguments)]
    pub(super) fn emit_general_adj_fill(
        &mut self,
        vertex: VertexKey,
        chain_of: &[Vec<VertexKey>],
        profiles: &[RingChainProfile],
        segments: u32,
        profile: f32,
        custom: Option<&CustomChains>,
        flat: bool,
        raw_coarse_mids: Option<&[Vec3]>,
        material: i32,
        sector_representatives: &[FaceTemplateRef<'_>],
        centre_column_representatives: &[FaceTemplateRef<'_>],
        central_representative: FaceTemplateRef<'_>,
        apex_blend: Option<&ApexBlend>,
        interior_blend: Option<&AdjInteriorBlend<'_>>,
        slots: &mut AdjNetSlots,
        result: &mut BevelResult,
    ) -> bool {
        let sectors = chain_of.len();
        let seg = segments as usize;
        let half = seg / 2;
        let even = seg % 2 == 0;
        if sectors < 3
            || profiles.len() != sectors
            || sector_representatives.len() != sectors
            || centre_column_representatives.len() != sectors
        {
            return false;
        }
        for chain in chain_of {
            // A chain that did not sample to `seg + 1` verts is outside scope; skip the
            // whole fill rather than emit a torn patch.
            if chain.len() != seg + 1 {
                return false;
            }
        }

        let vertex_position = self.vertex_position(vertex);

        // §8.5.5.5(b): the centre-fullness blend. `centroid` is the mean of the `s`
        // boundverts (the `(i, 0, 0)` ring corners); the fullness seed is pulled a
        // `fullness` fraction back toward the original corner `v`. The flat straight-chord
        // profile is the `fullness = 0` planar specialisation (§8.3.0.2.1).
        let boundverts: Vec<Vec3> = chain_of
            .iter()
            .map(|chain| self.vertex_position(chain[0]))
            .collect();
        let centroid = boundverts.iter().fold(Vec3::ZERO, |acc, &b| acc + b) / sectors as f32;
        let fullness = if flat {
            0.0
        } else if let Some(chains) = custom {
            chains.fullness
        } else {
            adj_fullness(profile, segments)
        };
        // §8.5.5.5(b): a **custom** profile consumes its fullness *differently* — the centre
        // is seeded from `v` reflected through the centroid, then moved back toward `v`:
        // `(2·centroid − v) + 2·fullness·(v − centroid)`, because a tabulated curve has no
        // analytic sphere to fit (law F1). Every analytic exponent uses the plain blend
        // `centroid + fullness·(v − centroid)`.
        let fullness_seed = if custom.is_some() {
            (2.0 * centroid - vertex_position) + 2.0 * fullness * (vertex_position - centroid)
        } else {
            centroid + fullness * (vertex_position - centroid)
        };

        if flat && even && seg == 2 {
            // §8.3.0.2.1: the straight-chord **even** cap at `segments = 2` welds at the
            // ring-plane centroid (`fullness = 0` ⇒ `fullness_seed == centroid`) with **no
            // interior ring** — each sub-patch is a single quad meeting the shared apex — so
            // the sub-patch bilinear blend *is* the fill: there is nothing to relax and no
            // grid to descend. From `segments ≥ 4` an interior grid exists, and its verts are
            // a *determined* quantity (§8.5.5.5(e) / (f.3) law **F2**) that must descend
            // through the same subdivision-limit relaxation the odd and curved caps use — not
            // a planar bilinear patch — so those parities fall through to the relaxed net
            // below (still with `fullness_seed == centroid`).
            let mut corners: Vec<[Vec3; 4]> = Vec::with_capacity(sectors);
            for i in 0..sectors {
                let p00 = boundverts[i];
                let p0t = self.vertex_position(chain_of[i][half]);
                let prev = (i + sectors - 1) % sectors;
                let pt0 = self.vertex_position(chain_of[prev][seg - half]);
                corners.push([p00, p0t, pt0, fullness_seed]);
            }
            self.emit_adj_grid(
                sectors,
                seg,
                half,
                even,
                chain_of,
                &AdjInterior::Blend { corners: &corners },
                material,
                sector_representatives,
                centre_column_representatives,
                central_representative,
                apex_blend,
                interior_blend,
                slots,
                result,
            );
            return true;
        }

        // §8.5.5.5(e) / §8.5.5.5(f.3) law **F2**: the straight-chord cap that reaches here —
        // every **odd** cap, and the **even** cap at `segments ≥ 4` — has an interior that is
        // a *determined* quantity, not a free interior choice. The odd cap closes on a central
        // `s`-gon; the even cap welds a single apex and mints no central face (the parity is
        // resolved by `emit_adj_grid`'s `even` branch, identical to the curved caps). Either
        // way the interior descends from the fullness centre (here the ring-plane centroid,
        // `fullness = 0`) through the very same subdivision-limit relaxation the curved caps
        // use; only the `fullness = 0` seed differs. So the flat cap folds onto the general
        // relaxed-net construction below with `fullness_seed == centroid` — it is not a
        // distinct planar bilinear blend.

        // §8.5.5.5(a) and the "the centre stencil consumes the *raw coarse* profile mids"
        // rule: the coarse net's `(i, 0, 1)` control points are each ring edge's own
        // profile sampled at the coarse `segments = 2` resolution — **never** the emitted
        // (§8.2.2.1.1-resampled) rim mids, which are a strictly later, emitted-only step.
        // A caller that already captured them hands them over; otherwise they come
        // straight off the ring edge's generator.
        let mids: Vec<Vec3> = (0..sectors)
            .map(|i| match raw_coarse_mids {
                Some(raw) if raw.len() == sectors => raw[i],
                _ => profiles[i].raw_positions(2)[1],
            })
            .collect();

        // §8.5.5.5(c)/(d): double the fullness-seeded coarse net — at least once, even at
        // `segments = 2` — until the working resolution reaches the target, then read the
        // grid back at the target sample fractions. The rim is the net's `j = 0` row of that
        // same read, which normally reproduces the already-emitted chain and sometimes does
        // not (below).
        let net = adj_relaxed_net(&boundverts, &mids, fullness_seed, profiles, segments);
        let axis = adj_interior_axis(profiles, segments, 2 * net.half as u32);
        self.replant_rim_off_the_saturating_net(chain_of, &net, &axis, sectors, seg, half);

        self.emit_adj_grid(
            sectors,
            seg,
            half,
            even,
            chain_of,
            &AdjInterior::Relaxed {
                net: &net,
                axis: &axis,
                sectors,
                segments: seg,
            },
            material,
            sector_representatives,
            centre_column_representatives,
            central_representative,
            apex_blend,
            interior_blend,
            slots,
            result,
        );
        true
    }

    /// Re-place the rim samples that the §8.5.5.5(d) two-axis read does **not** put on the
    /// profile polyline — the `⌊segments/2⌋` per chain whose predecessor's source table
    /// saturates ([`adj_rim_source_row`]).
    ///
    /// **Which samples can move.** The resample computes sub-patch `i`'s block over
    /// `j = 0 … ⌈segments/2⌉ − 1`, `k = 0 … ⌊segments/2⌋`, so chain `i`'s rim is produced by
    /// **two** sub-patches: `k = 1 … half` is block `i`'s own `j = 0` row and carries the
    /// foreign `j*_rim`, while `k = half + 1 … segments − 1` is stored as block `i + 1`'s
    /// `k = 0` column, whose `j` coordinate comes from chain `i`'s own tables and so
    /// reproduces §8.2.2.1.1 step 3's 1-D read-back algebraically at every count, whatever
    /// any table does. One chain's two rim halves are therefore computed against two
    /// different chains' tables and need not be reflections of each other — that asymmetry
    /// is structural, not miter-specific.
    ///
    /// **Why the collapsing case is left alone.** At `j*_rim = 0` the kernel lerps along
    /// source row `0`, and source row `0` *is* the profile polyline, so the read is the same
    /// point the chain builders' 1-D read-back already emitted; re-deriving it through the
    /// grid would move every well-conditioned rim in the crate by a float representation's
    /// worth for no parity gain. The two expressions are the same rule, and
    /// [`adj_rim_source_row`] is where the guide's collapse condition is tested.
    ///
    /// **Re-placing rather than minting is a scheduling compromise.** §8.5.5.5(d) mints the
    /// rim *from* the net, which it can do because the net exists first; here the rim verts
    /// were minted with the ring, since they are shared with the beveled edge's
    /// cross-section and the strips are stitched before any cap. Moving the already-minted
    /// keys leaves the topology, the census and every rim-sharing strip identical — what it
    /// does not reach is anything position-keyed that a caller derived from a rim vert
    /// before the cap ran.
    pub(super) fn replant_rim_off_the_saturating_net(
        &mut self,
        chain_of: &[Vec<VertexKey>],
        net: &AdjNet,
        axis: &[Vec<SourcePosition>],
        sectors: usize,
        seg: usize,
        half: usize,
    ) {
        if seg < 2 {
            return;
        }
        let resolution = 2 * net.half;
        for i in 0..sectors {
            let Some(rim_row) = adj_rim_source_row(axis, sectors, seg, resolution, i) else {
                continue;
            };
            for k in 1..=half.min(seg - 1) {
                let position = net.sample(i, rim_row, axis[i][k].snapped());
                self.set_vertex_position(chain_of[i][k], position);
            }
        }
    }

    /// The (incoming, outgoing) edges of `face` meeting at `vertex` in the face's
    /// loop traversal order: the edge arriving at `vertex` and the edge leaving it.
    pub(super) fn face_corner_edges(
        &self,
        face: FaceKey,
        vertex: VertexKey,
    ) -> Option<(EdgeKey, EdgeKey)> {
        self.iter_face_loops(face)
            .find(|&loop_key| self.loop_vertex(loop_key) == vertex)
            .map(|loop_key| {
                (
                    self.loop_edge(self.loop_prev(loop_key)),
                    self.loop_edge(loop_key),
                )
            })
    }

    /// Build the rounded adjacency-grid fill for a **general non-symmetric**
    /// (oblique / irregular-ring) valence-`s` junction at any **smooth** exponent and
    /// equal offsets (BEVEL.md §8.5.5.5). Where the cube corner leaned
    /// on a closed-form sphere snap (§8.5.5.3, only an orthonormal equal-offset
    /// corner has a sphere of radius `offset`), this corner has no such sphere, so
    /// the cap is placed by two general mechanisms instead:
    ///
    /// - **centre-fullness blend (§8.5.5.5(b)):** the central control vertex is
    ///   `centroid + fullness·(v − centroid)`, blending the boundvert centroid
    ///   toward the original corner `v`. `fullness` is a profile/segment fit scalar
    ///   (the circular default tabulated near `0.55–0.65`; the (f.3) law **F3**
    ///   regression everywhere else, signed and unbounded), not a per-vertex solve;
    /// - **subdivision-relaxation interior (§8.5.5.5(c)):** the coarse `segments = 2`
    ///   control net is refined by repeated cubic (Catmull–Clark, Sabin-modified)
    ///   rounds — at least one, even at `segments = 2` — each re-pinning the chain
    ///   boundary from the profile at that round's own doubled resolution; a
    ///   non-power-of-two target then reads the dense net back at its own profile-arc
    ///   fractions (§8.5.5.5(d)). The rim is never re-derived: it stays the §8.2.2.1.1
    ///   read-back the strips already carry.
    ///
    /// The grid **topology** (vert / face counts, per-face histogram, even/odd
    /// centre parity, §8.5.5.5(e)) is identical to the cube-corner grid — only the
    /// vertex *positions* differ (no sphere). Even `segments` weld one shared centre
    /// vertex (the once-subdivided extraordinary centre, pinned strictly) with no
    /// central face; odd `segments` close the `s` sectors with one central `s`-gon.
    /// Only the (d) read-back kernel is a free choice (Tier-2, §8.5.5.7); the cap
    /// pointedly does **not** lie on the cube-corner sphere — the positive signal
    /// that the general relaxation path, not the §8.5.1 snap, produced it.
    ///
    /// Scope (§8.5.5.5): irregular rings, oblique corners, valence `s`, every exponent
    /// (§8.5.5.5(f.1) step 4 gates on none — the two square sentinels included, since
    /// their closed forms belong to the *tri-corner* and even-`segments` routes, not to
    /// this one), equal offsets. At square-in this is step 4's grid unchanged: same
    /// topology, ring sampled as the two concave legs, centre seeded by law **F3**'s
    /// negative fullness. ((f.4b) spells that same construction out, but for *vertex* mode,
    /// where it is the only square-in route; this builder serves both modes, and in edge
    /// mode it is reached whenever the (f.4a) weld's ring test declines.)
    #[allow(clippy::too_many_arguments)]
    pub(super) fn build_adj_fill_general_nonsymmetric(
        &mut self,
        cut: &JunctionCut,
        chains: &[EdgeEndChain],
        segments: u32,
        profile: f32,
        custom: Option<&CustomChains>,
        material: i32,
        templates: &TemplateBank,
        slots: &mut AdjNetSlots,
        result: &mut BevelResult,
    ) -> bool {
        let sectors = cut.boundverts.len();
        let seg = segments as usize;

        // Each sector's profile chain, oriented from its opening boundvert `B_i`
        // (`seg + 1` samples; the two ends are the ring boundverts). Reused from
        // [P]/[S], not re-minted, so the rim coincides with the arriving strips.
        let mut chain_of: Vec<Vec<VertexKey>> = Vec::with_capacity(sectors);
        for i in 0..sectors {
            match self.oriented_chain(cut, chains, i) {
                Some(chain) if chain.len() == seg + 1 => chain_of.push(chain),
                // A sector that did not sample to `seg + 1` verts (a collinear /
                // degenerate chain) is outside scope; skip the whole fill rather
                // than emit a torn patch.
                _ => return false,
            }
        }

        // Each sector's ring-edge generator (§8.5.5.5(a)): the cross-section cell of the
        // beveled edge that sector's chain crosses. The stencil re-samples it at every
        // round's doubled resolution and reads its raw coarse mid off it.
        let mut profiles: Vec<RingChainProfile> = Vec::with_capacity(sectors);
        for i in 0..sectors {
            match self.sector_chain(cut, chains, i) {
                // §8.6.3.1(6): each sector re-reads the curve through the handedness bit its
                // own chain was emitted with — siblings of one ring routinely differ.
                Some(chain) => match chain.cell {
                    Some(cell) => profiles.push(RingChainProfile::Cell {
                        cell,
                        profile,
                        custom: custom.map(|chains| {
                            CustomChainRead::edge_sector(chains, chain.custom_reverse)
                        }),
                    }),
                    None => return false,
                },
                None => return false,
            }
        }

        // §8.5.5.5(b) / §8.3.0.2.1: the flat straight-chord profile (`r = 1`) is the
        // `fullness = 0` planar specialisation — the cap lies in the boundvert ring
        // plane with no bulge toward the corner `v` and no extraordinary-centre
        // relaxation. Every other *smooth* exponent keeps the (f.3) fullness seed — the
        // table is consulted only for the circular `r = 2` at `segments ≤ 11`, everything
        // else is law F3's regression in the raw `profile` scalar — and relaxes. Law **F1**
        // (custom) is tested *before* F2, so a custom curve never takes the flat branch even
        // when the untouched `profile` slider happens to sit at the line exponent — the curve,
        // not the slider, is the cross-section.
        let flat = profile_to_exponent(profile) == 1.0 && custom.is_none();
        let sector_representatives = cut.resolve_sector_representatives(templates);
        let centre_column_representatives = cut.resolve_centre_column_representatives(templates);
        let central_representative = templates.resolve(cut.central_representative);
        self.emit_general_adj_fill(
            cut.vertex,
            &chain_of,
            &profiles,
            segments,
            profile,
            custom,
            flat,
            None,
            material,
            &sector_representatives,
            &centre_column_representatives,
            central_representative,
            // §5.2.4.1 / §5.2.4.2 are vertex-mode-only rules; edge-mode junction fills
            // leave the welded apex and the chain-interior verts at their §5.2.4 minted
            // defaults.
            None,
            None,
            slots,
            result,
        )
    }
}
