//! Miter classification, mitered ring data and mitered [V-ADJ] fills (BEVEL.md §8.9).

use super::*;

impl LoopMesh {
    /// Build a mitered junction's [V-ADJ] fill (BEVEL.md §8.9.2 / §8.9.3 / §8.9.5.3): the
    /// enlarged ring is filled by the **general valence-`s` adjacency grid** of §8.5.5.5 —
    /// the same centre-fullness / subdivision-relaxation machinery the general
    /// non-symmetric junction rides — over the miter's heterogeneous ring (unsplit sharp
    /// meet-points interleaved with each split sector's arc / patch chain). The two chain
    /// kinds (a split sector's fill chain, §8.9.5.1, and each beveled edge's cross-section,
    /// §8.9.5.2) are interchangeable `chain_of` inputs, so the fill needs no miter-specific
    /// step — inner and outer miters differ only in *which* sectors split. The parity of
    /// `segments` decides the centre by §8.5.5.5(e) — even welds one extraordinary-centre
    /// apex (no central face), odd closes on a central `s`-gon.
    pub(super) fn build_adj_fill_miter_ring(
        &mut self,
        omj: &MiterRingJunction,
        material: i32,
        templates: &TemplateBank,
        result: &mut BevelResult,
    ) {
        let count = omj.boundverts.len();
        // A miter splits at least one sector of an `s ≥ 3` ring, so the enlarged ring
        // always carries `≥ 4` boundverts.
        if count < 4
            || omj.chains.len() != count
            || omj.profiles.len() != count
            || omj.sector_representatives.len() != count
            || omj.centre_column_representatives.len() != count
        {
            return;
        }
        // §8.5.5.5(f.5): at a **square** profile the enlarged ring is filled by the
        // parallel-offset grid (even `segments` only), not the general relaxed cap — the
        // arc/patch start seeds one centreline off the stored meet-point / boundvert-pair
        // midpoints and the original corner `v` survives un-rounded at the weld.
        //
        // The dispatch predicate is (f.1) step 1's, asked over the *enlarged* ring —
        // [`adj_square_ring_claims`] reads a boundvert count, which a miter only raises. Both
        // miters mint the (f.5) seed inputs, so the length checks are structural guards, not a
        // route: an inner-mitered square ring reaches the lattice exactly as an outer one does.
        if adj_square_ring_claims(
            profile_to_exponent(omj.profile),
            count,
            omj.segments,
            omj.custom.is_some(),
        ) && omj.opening_edge_directions.len() == count
            && omj.opening_edge_tips.len() == count
            && omj.boundvert_non_reflex.len() == count
        {
            self.build_adj_fill_miter_square_ring(omj, material, templates, result);
            return;
        }
        let sector_representatives = omj.resolve_sector_representatives(templates);
        let centre_column_representatives = omj.resolve_centre_column_representatives(templates);
        let central_representative = templates.resolve(omj.central_representative);
        // §8.9.5.3: the mitered ring takes the ordinary (f.1) dispatch with no miter-specific
        // step, so it reads the flat cap off (f.3) law **F2** exactly as a plain ring does —
        // `r == 1` (the mitered chain there *is* the flat chord, §8.9.5) seeds `fullness = 0`.
        // Law F1 is tested before F2, so a custom curve is never the flat cap whatever the
        // untouched slider reads.
        let flat = profile_to_exponent(omj.profile) == 1.0 && omj.custom.is_none();
        // §8.9.5.3: feed the centre stencil the raw coarse control-net mids (populated
        // at `segments = 2`), so the resampled arc-flanking rim mids never skew the apex.
        let raw_coarse_mids = if omj.raw_coarse_mids.len() == count {
            Some(omj.raw_coarse_mids.as_slice())
        } else {
            None
        };
        self.emit_general_adj_fill(
            omj.vertex,
            &omj.chains,
            &omj.profiles,
            omj.segments,
            omj.profile,
            // §8.5.5.5(f.3): a mitered cap is a [V-ADJ] cap like any other, so a custom
            // profile puts its centre on law **F1** and the reflected-seed blend of
            // §8.5.5.5(b). The carve-out is about which *cross-sections* read the curve, not
            // about the cap's centre.
            omj.custom,
            flat,
            raw_coarse_mids,
            material,
            &sector_representatives,
            // §8.9.3 / §6.6.2: at odd `segments` the centre column of an edge-less
            // (miter-split) boundvert takes the §6.6.1.4 tie-break donor; every other
            // centre column keeps the sector's positional donor. Even `segments` reads
            // none of these (no centre column).
            &centre_column_representatives,
            central_representative,
            // §5.2.4.1 / §5.2.4.2 are vertex-mode-only rules; edge-mode junction fills
            // leave the welded apex and the chain-interior verts at their §5.2.4 minted
            // defaults.
            None,
            None,
            &mut AdjNetSlots::off(),
            result,
        );
    }

    /// Fill a **square-profile** mitered junction with the §8.5.5.5(f.5) parallel-offset grid
    /// over the enlarged ring — the miter-ring analogue of [`Self::build_adj_fill_square_ring`].
    ///
    /// The enlarged ring carries its miter's arc/patch starts (§8.5.5.5(f.5) cases 1 & 2): for
    /// `ARC` each tagged leading `B_1` seeds its one centreline from the stored §7.6.2 meet
    /// point (`omj.arc_patch_starts`' middle control point); for `PATCH`, `B_1` seeds two
    /// centrelines from the boundvert-pair midpoints straddling `B_2`. The outer miter tags one
    /// start, the inner miter one per firing sector. Every other centreline comes from the
    /// sharp / pass-2 machinery, identical to the plain square ring. The two straight rim rows
    /// are then redirected onto the seed runs and `v` survives at the weld.
    pub(super) fn build_adj_fill_miter_square_ring(
        &mut self,
        omj: &MiterRingJunction,
        material: i32,
        templates: &TemplateBank,
        result: &mut BevelResult,
    ) {
        let sectors = omj.boundverts.len();
        let seg = omj.segments as usize;
        let half = seg / 2;
        if sectors < 4 || seg < 2 || seg % 2 != 0 {
            return;
        }
        let chain_of = &omj.chains;
        for chain in chain_of {
            if chain.len() != seg + 1 {
                return;
            }
        }

        let vertex_position = self.vertex_position(omj.vertex);
        let boundvert_positions: Vec<Vec3> = (0..sectors)
            .map(|i| self.vertex_position(chain_of[i][0]))
            .collect();
        let arc_patch: Vec<ArcPatchStart> = omj
            .arc_patch_starts
            .iter()
            .map(|&(index, mode, arc_middle)| ArcPatchStart {
                index,
                mode,
                arc_middle,
            })
            .collect();
        let seeds = square_ring_centreline_seeds(
            &omj.opening_edge_directions,
            &omj.opening_edge_tips,
            &boundvert_positions,
            &omj.boundvert_non_reflex,
            vertex_position,
            &arc_patch,
        );
        let starts: Vec<(Vec3, Vec3)> = boundvert_positions.iter().copied().zip(seeds).collect();

        // §8.5.5.5(f.5): redirect the grid's two straight rim rows onto the seed runs
        // `B_i → C_i` / `B_i → C_{i−1}` (the rim verts keep their keys — the cap stays welded
        // to the arriving strips — only their positions move). Identical to the plain square
        // ring; the arc sector's fold lands on the stored meet-point (`C_arc`) by construction.
        for i in 0..sectors {
            let (b_i, c_i) = starts[i];
            let previous = (i + sectors - 1) % sectors;
            let c_prev = starts[previous].1;
            for step in 1..=half {
                let t = step as f32 / half as f32;
                self.set_vertex_position(chain_of[i][step], b_i + t * (c_i - b_i));
                self.set_vertex_position(chain_of[previous][seg - step], b_i + t * (c_prev - b_i));
            }
        }

        let sector_representatives = omj.resolve_sector_representatives(templates);
        let centre_column_representatives = omj.resolve_centre_column_representatives(templates);
        let central_representative = templates.resolve(omj.central_representative);
        self.emit_adj_grid(
            sectors,
            seg,
            half,
            true, // even `segments` — the innermost vert welds at `v`
            chain_of,
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
            // A miter ring is not a §8.3.4.1 pipe, so no post-pass reads the slot walk.
            &mut AdjNetSlots::off(),
            result,
        );
    }

    /// The §8.9.1 reference normal `n_ref` for a sector's signed reflex test. The chain:
    /// the **forward** incident face just past `g_k` — the sector's first-corner face
    /// (§7.6.1: the walk enters the sector on `g_k`, so its first corner's face is the
    /// one just past `g_k`) — falling back to the **back** incident face just before
    /// `g_(k+1)` (its last-corner face) when the forward face's plane is degenerate, and
    /// finally to the junction vertex normal `n_v` when neither incident face is usable.
    pub(super) fn reflex_reference_normal(&self, sector: &Sector, vertex: VertexKey) -> Vec3 {
        for corner in [sector.corners.first(), sector.corners.last()] {
            if let Some(&(_, face)) = corner {
                if let Some(normal) = self.face_newell_normal(face) {
                    if normal.is_finite() {
                        return normal;
                    }
                }
            }
        }
        self.junction_outward_normal(vertex)
    }

    /// §8.9.1's three-way sector classification, with the two flanking edge directions
    /// (both taken **out of** `vertex`) the miter builders slide along.
    ///
    /// The **straight** band is asked first, and that order is the guide's: §8.9.8's
    /// pipe-ring derivation has the straight branch claim a sector "before any reflex
    /// classification is asked for", which is what keeps an exactly-`180°` side of a pipe
    /// junction out of *both* miters. Straightness is §7.6.2's near-collinear band on the
    /// direction angle `θ_dir` — first flank taken **into** `v`, second **out of** it, so
    /// `θ_dir ≈ 0` is the straight-through run and `θ_dir ≈ 180°` the double-back — and
    /// the band is symmetric at both ends, so the supplementary pair a beveled spoke cuts
    /// a `180°` side into is always classified together.
    ///
    /// Otherwise the winding-invariant signed test decides: `cross · n_ref < 0` is the
    /// wide (reflex) sector, in the genuine incident-face frame and never a flattened
    /// working-plane one (§8.9.1's consequence 1 — no flat swept-angle substitute, no
    /// fitted angular cutoff).
    ///
    /// `None` when the sector carries fewer than two bounding edges (the malformed open
    /// fan [`Self::sector_layout`] also declines) or when either flank is degenerate.
    pub(super) fn classify_miter_sector(
        &self,
        vertex: VertexKey,
        vertex_position: Vec3,
        sector: &Sector,
    ) -> Option<MiterSectorVerdict> {
        let dir_open =
            self.edge_direction_from(*sector.bounding_edges.first()?, vertex, vertex_position)?;
        let dir_close =
            self.edge_direction_from(*sector.bounding_edges.get(1)?, vertex, vertex_position)?;
        // Load-bearing coupling: these are bit-identical to the arguments `solve_meet_point`
        // feeds its own §7.6.2 degenerate-ladder gate, which is *why* a `Straight` verdict
        // guarantees the sector's meet-point took that ladder's (anti-)parallel branch —
        // exactly what §8.9.1's straight bullet asks of a sector no miter may fire on. Any
        // refactor that lets the two angles drift apart breaks the guarantee silently.
        let class = if is_degenerate_direction_angle(unit_chord_angle_between(-dir_open, dir_close))
        {
            MiterSectorClass::Straight
        } else if (-dir_open)
            .cross(-dir_close)
            .dot(self.reflex_reference_normal(sector, vertex))
            < 0.0
        {
            MiterSectorClass::Reflex
        } else {
            MiterSectorClass::NonReflex
        };
        Some(MiterSectorVerdict {
            dir_open,
            dir_close,
            class,
        })
    }

    /// Which sectors the one-segment inner miter splits into an arc pair, and the two
    /// flanking directions each split sector's `spread` slide runs along (BEVEL.md
    /// §8.9.3.2(a)). `None` in a slot means that sector keeps its single sharp
    /// meet-point.
    ///
    /// The predicate is **non-reflex AND not straight**, not merely "non-reflex":
    /// §8.9.1's classification is three-way, and a straight sector — the flanks running
    /// through the vertex, or doubling back on it — fires no miter at all. Both halves
    /// are the same tests the sharp junction already runs: the winding-invariant signed
    /// reflex sign `cross · n_ref < 0` in the genuine incident-face frame, and §7.6.2's
    /// near-collinear direction-angle band.
    ///
    /// The whole junction declines (every slot `None`) when it presents more than one
    /// reflex sector — §8.9.1 admits at most one, so a second means the frame is not to
    /// be trusted — or when a fired reflex sector is `miter_outer`'s and that miter is
    /// non-sharp: the outer miter has no one-segment build here, and half-mitering a
    /// junction is worse than leaving it sharp.
    pub(super) fn seg1_inner_miter_splits(
        &self,
        vertex: VertexKey,
        vertex_position: Vec3,
        sectors: &[Sector],
        miter: Option<Seg1InnerMiter>,
    ) -> Vec<Option<(Vec3, Vec3)>> {
        let Some(miter) = miter else {
            return vec![None; sectors.len()];
        };
        let mut split: Vec<Option<(Vec3, Vec3)>> = Vec::with_capacity(sectors.len());
        let mut reflex_count = 0usize;
        for sector in sectors {
            if sector.corners.len() < 2 {
                split.push(None);
                continue;
            }
            let Some(verdict) = self.classify_miter_sector(vertex, vertex_position, sector) else {
                split.push(None);
                continue;
            };
            match verdict.class {
                MiterSectorClass::Reflex => {
                    reflex_count += 1;
                    split.push(None);
                }
                MiterSectorClass::Straight => split.push(None),
                MiterSectorClass::NonReflex => {
                    split.push(Some((verdict.dir_open, verdict.dir_close)))
                }
            }
        }
        if reflex_count > 1 || (reflex_count == 1 && miter.outer != BevelMiter::Sharp) {
            return vec![None; sectors.len()];
        }
        split
    }
}

/// The one-segment inner miter's parameters (BEVEL.md §8.9.3.2), resolved by the caller.
/// `Some` only when `miter_inner` is non-`Sharp` **and** `vmesh_method = Adj`: `Cutoff`
/// normalises both miter modes to `Sharp` up front (§8.3.0.3), a parameter gate that
/// carries no `segments` qualifier and so does not narrow at one segment.
///
/// `miter_inner` distinguishes only sharp from non-sharp (§8.9.3), so `Arc` and `Patch`
/// are the same construction here and the mode itself need not be carried.
#[derive(Clone, Copy)]
pub(super) struct Seg1InnerMiter {
    /// §8.9.4: how far each arc end is pushed out along its own flanking beveled edge.
    pub(super) spread: f32,
    /// The junction's `miter_outer`, read only to decline a junction whose reflex sector
    /// belongs to an outer miter this path does not build.
    pub(super) outer: BevelMiter,
}

/// One **mitered** junction's ring data (BEVEL.md §8.9.2 / §8.9.3 / §8.9.5), consumed
/// by the general valence-`s` [V-ADJ] fill. `boundverts` is the enlarged ring in disk
/// order — one sharp meet-point per unsplit sector, plus the arc pair `B1`/`B3` (or
/// `PATCH`'s `B1`/`B2`/`B3`) of every **split** sector, whichever miter split it — and
/// `chains[i]` is the full `segments + 1`-sample profile chain on the ring edge from
/// `boundverts[i]` to `boundverts[(i+1) % len]` (oriented that way): a strip
/// cross-section across the shared beveled edge (§8.9.5.2), or a split sector's own
/// fill chain `s_0 … s_segments` (§8.9.5.1). `segments` is the fill width.
pub(super) struct MiterRingJunction<'c> {
    pub(super) vertex: VertexKey,
    pub(super) boundverts: Vec<VertexKey>,
    pub(super) chains: Vec<Vec<VertexKey>>,
    /// Per ring edge (same order as `chains`), the **raw coarse control-net mid** —
    /// the sector's single coarse `(i, 0, 1)` profile point at the `segments = 2`
    /// resolution, sampled *before* the §8.2.2.1.1 chord-fraction resample skews the
    /// arc-flanking cross-sections (BEVEL.md §8.5.5.5(d) / §8.9.5.3). This is what the
    /// even-`segments = 2` extraordinary-centre stencil consumes; feeding it the
    /// emitted (resampled) rim mids instead is the diagnosed multiface-reflex apex
    /// bug. Populated only at `segments = 2` (the only width the centre stencil fires
    /// at); empty otherwise.
    pub(super) raw_coarse_mids: Vec<Vec3>,
    /// Per ring edge (same order as `chains`), that edge's profile **generator**
    /// (BEVEL.md §8.5.5.5(a)): the §8.5.5.5(c) subdivision-relaxation stencil re-pins its
    /// boundary at every round's own doubled resolution, so it needs each chain's profile,
    /// not just the emitted rim verts. Heterogeneous by construction — a split sector
    /// contributes an arc (or two patch chords) where the rest contribute cross-section
    /// cells — and the fill never tests which is which (§8.9.5.3).
    pub(super) profiles: Vec<RingChainProfile<'c>>,
    /// Per ring edge (same order as `chains`), the beveled edge that ring edge's
    /// cross-section crosses — `None` on a split sector's **internal** (arc /
    /// patch-chord) edges, which cross no beveled edge. This is what makes the split
    /// sector cost the §6.7.5.3 walk its extra ring step(s): the walk counts only the
    /// `Some` entries but steps over every entry (§6.7.5.4 case (5)).
    pub(super) cross_edges: Vec<Option<EdgeKey>>,
    pub(super) segments: u32,
    /// The junction's `profile` scalar. The miter builders admit the whole analytic band, the
    /// square sentinel included, so the fill reads the run's exponent from here rather than
    /// assuming one — it is what routes the enlarged ring between (f.5)'s lattice and the
    /// general relaxed cap, and what seeds that cap's `fullness`.
    pub(super) profile: f32,
    /// The operation's custom curve, or `None` for the analytic family. Carried because the
    /// cap's `fullness` is law **F1** whenever the profile is custom (BEVEL.md §8.5.5.5(f.3),
    /// which keys on the profile *type*, is computed once per operation and knows nothing of
    /// miters) — the per-edge generators cannot stand in for it, since §8.6.3.1(4).4's miter
    /// carve-out leaves the arc / patch chains analytic on a ring whose flanks ride the curve.
    pub(super) custom: Option<&'c CustomChains<'c>>,
    /// Per sector `i` (same order as `boundverts` / `chains`), the [V-ADJ] grid donor of
    /// that sector's **non-centre** faces (BEVEL.md §6.6.2's grid row, edge bevel): the
    /// sector's **own** boundvert rep — `boundvert_representative` of `cross_edges[i]` for a
    /// beveled sector, or §6.6.1's edge-less-boundvert primitive for a miter-split
    /// (edge-less) one whose `cross_edges` entry is `None`. Captured pristine (the re-stitch
    /// consumes the junction vertex), exactly as [`JunctionCut::sector_representatives`] is.
    /// The centre column is a distinct slot (`centre_column_representatives`), which is the
    /// only grid face that ever re-routes off this positional donor (§6.6.2).
    pub(super) sector_representatives: Vec<FaceTemplate>,
    /// The donor of the miter cap's **odd-`segments` centre n-gon** — the same
    /// ring-spanning fill [`JunctionCut::central_representative`] carries, over the miter's
    /// enlarged ring.
    pub(super) central_representative: FaceTemplate,
    /// Per boundvert (same order as `boundverts`), the stage-1 donor of that sector's
    /// **centre column** at odd `segments` — the `k == segments/2` bridge strip meeting the
    /// centre n-gon (BEVEL.md §6.6.2's [V-ADJ] grid row / §8.9.3). A boundvert that owns a
    /// beveled edge takes that sector's **own** rep when the edge is a non-seam, else the
    /// §6.6.1.4 tie-break between its own rep and the next boundvert's rep; an **edge-less**
    /// miter-split boundvert — the arc's far-flank partner, which owns no beveled edge —
    /// takes that tie-break *unconditionally* (no seam gate). No ring-wide uniform donor
    /// enters. Unread at even `segments`, where the cap has no centre column.
    pub(super) centre_column_representatives: Vec<FaceTemplate>,
    /// Per **enlarged-ring boundvert** `i`, the direction (out of `vertex`) of the beveled
    /// edge shared with the *previous* ring boundvert — `edge_in(i)` in the §8.5.5.5(f.5)
    /// seeding pass, `Vec3::ZERO` where that ring edge is an internal arc/patch chain that
    /// crosses no beveled edge. Populated by both miters: (f.1) step 1 gates on the exponent
    /// and the *ring* count, neither of which distinguishes inner from outer (§8.9.3 licenses
    /// no profile gate of its own).
    pub(super) opening_edge_directions: Vec<Vec3>,
    /// Per enlarged-ring boundvert `i`, the far tip of that same `edge_in(i)` (for pass 2's
    /// segment projection). Same population rule as `opening_edge_directions`.
    pub(super) opening_edge_tips: Vec<Vec3>,
    /// Per enlarged-ring boundvert `i`, whether §8.9.1's signed test read that boundvert's own
    /// sector as **non-reflex and not straight** — §8.5.5.5(f.5) case 3's entry condition, and
    /// the whole of the gate: a `false` slot proposes nothing and drops its centrelines to
    /// pass 2 (case 4). The outer miter *opens* its reflex sector into arc/patch boundverts, so
    /// those slots are skipped as a start or as consumed either way; the inner miter leaves a
    /// `miter_outer = Sharp` reflex sector as a plain meet-point, and that is the slot this
    /// diverts.
    pub(super) boundvert_non_reflex: Vec<bool>,
    /// The enlarged ring's arc/patch starts (§8.5.5.5(f.5) cases 1 & 2), each the ring index
    /// of a tagged leading boundvert `B_1`, its miter mode (`Arc` / `Patch`), and the arc's
    /// stored §7.6.2 **middle control point** (the pre-reposition meet-point — case 2's
    /// `C_i`; unread for `Patch`). The **outer** miter tags at most one (§8.9.1's single
    /// reflex sector); the **inner** miter tags one per firing non-reflex corner, so this
    /// carries `m` entries there. Empty when no miter fired.
    pub(super) arc_patch_starts: Vec<(usize, BevelMiter, Vec3)>,
}

/// The §8.5.5.5(f.5) arc/patch seed-input for the square parallel-offset grid: which
/// enlarged-ring boundvert is the fired-reflex leading `B_1`, its miter mode, and the arc's
/// stored §7.6.2 middle control point (case 2's centreline seed; unread for `Patch`).
pub(super) struct ArcPatchStart {
    pub(super) index: usize,
    pub(super) mode: BevelMiter,
    pub(super) arc_middle: Vec3,
}

impl MiterRingJunction<'_> {
    /// The junction's enlarged outermost ring as the §6.7.5.3 walk's chain sequence.
    /// Each split sector contributes `p − 1` fill chains (`p = 2` for an arc, `3` for the
    /// outer `PATCH`) that no beveled edge crosses, so a span stepping through it comes up
    /// that many chains short at its far end — outer and inner miters alike (§6.7.5.4
    /// cases (5) and (6)).
    pub(super) fn ring_chains(&self) -> Vec<RingChain> {
        self.cross_edges
            .iter()
            .zip(self.chains.iter())
            .map(|(&cross_edge, samples)| RingChain {
                cross_edge,
                samples: samples.clone(),
            })
            .collect()
    }

    /// Resolve the per-sector [V-ADJ] grid donors (BEVEL.md §6.6.2's grid row) against the
    /// sealed template bank — one per sector, in `boundverts` / `chains` order. Mirrors
    /// [`JunctionCut::resolve_sector_representatives`] over the miter's enlarged ring.
    pub(super) fn resolve_sector_representatives<'a>(
        &self,
        templates: &'a TemplateBank,
    ) -> Vec<FaceTemplateRef<'a>> {
        self.sector_representatives
            .iter()
            .map(|&template| templates.resolve(template))
            .collect()
    }

    /// Resolve the per-boundvert odd-`segments` centre-column donors (BEVEL.md §6.6.2 /
    /// §8.9.3) against the sealed bank. Unread at even `segments` (no centre column).
    pub(super) fn resolve_centre_column_representatives<'a>(
        &self,
        templates: &'a TemplateBank,
    ) -> Vec<FaceTemplateRef<'a>> {
        self.centre_column_representatives
            .iter()
            .map(|&template| templates.resolve(template))
            .collect()
    }
}

/// §8.9.1's sector classification, which is **three**-way and not a reflex boolean: a
/// sector whose two flanking beveled edges run nearly collinear fires *no* miter at all,
/// and collapsing that class into `NonReflex` over-splits every straight sector in the
/// mesh (§8.9.3 / §8.9.3.2(a)).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum MiterSectorClass {
    /// The flanks run straight through `v` (`θ_dir ≈ 0`) or double back on it
    /// (`θ_dir ≈ 180°`). The sector keeps its single §7.6.2 meet-point.
    Straight,
    /// The wide (`> 180°`) sector — `miter_outer`'s.
    Reflex,
    /// The narrow sector — `miter_inner`'s.
    NonReflex,
}

/// One sector's §8.9.1 verdict: its class plus the two unit flank directions taken **out
/// of** the junction vertex, which every miter builder then slides its boundverts along.
pub(super) struct MiterSectorVerdict {
    pub(super) dir_open: Vec3,
    pub(super) dir_close: Vec3,
    pub(super) class: MiterSectorClass,
}

/// The `segments + 1` raw samples of a miter arc (BEVEL.md §8.9.5 / §8.9.5.1) in the
/// arc's (generally non-orthonormal) unit-square frame: `a_axis = midpoint − b1`,
/// `b_axis = midpoint − b3`, inner corner `b1 + b3 − midpoint`, sample `k` at
/// `P(u_k, v_k) = a_axis·u_k + b_axis·v_k + inner_corner`, so `(0, 1)` recovers `b1`,
/// `(1, 0)` recovers `b3` and `(1, 1)` the stored midpoint.
///
/// The `(u_k, v_k)` are the run's **own** [S] even-chord spacing table (§8.6.4) at the
/// `profile`-derived exponent (§8.6.2) — the very table every beveled edge's
/// cross-section reads. There is no arc-specific sampler and no arc-specific spacing
/// table: §8.9.5's exponent-scope note is explicit that the circle-valued exponent an
/// arc-start boundvert *records* never selects a sampler, so an arc is circular only when
/// the run is. At `profile = 0.25` the table degenerates to the plain chord `b1 → b3`; at
/// the square sentinel it runs along the two cell sides *through* the stored midpoint; at
/// `profile = 0.5` it is the inscribed quarter circle whose `segments = 2` interior sample
/// is the closed-form `(2−√2)·chord_mid + (√2−1)·midpoint` bulge.
pub(super) fn arc_chain_positions(
    b1: Vec3,
    b3: Vec3,
    midpoint: Vec3,
    profile: f32,
    segments: u32,
) -> Vec<Vec3> {
    let a_axis = midpoint - b1;
    let b_axis = midpoint - b3;
    // §8.2.2.1 clause (1), the exact zero-axis arm, tested before anything is normalised:
    // an inner miter at `spread = 0` mints both its ends on the retained meet point, so all
    // three of the arc's anchors are one point and there is no frame to sample through
    // (§8.9.5.4(b)). What the chain emits then is the chord's even subdivision — and at
    // this count it is the *arithmetic* of that subdivision, not only its algebra, that the
    // fill downstream reads (§8.9.5.4(c)).
    if a_axis == Vec3::ZERO || b_axis == Vec3::ZERO {
        return (0..=segments)
            .map(|k| collapsed_chain_sample(b1, b3, k, segments))
            .collect();
    }
    let inner_corner = b1 + b3 - midpoint;
    superellipse_chain(profile, segments)
        .into_iter()
        .enumerate()
        .map(|(k, (u, v))| {
            if k == 0 {
                b1
            } else if k == segments as usize {
                b3
            } else {
                a_axis * u + b_axis * v + inner_corner
            }
        })
        .collect()
}
