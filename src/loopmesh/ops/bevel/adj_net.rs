//! The [V-ADJ] cap control net and its relaxation (BEVEL.md §8.5.5.5).

use super::*;

/// The canonical §8.5.2 slot `(i, j, k)` of every interior grid vert a [V-ADJ] fill mints, in
/// mint order — the net's own index vocabulary, recorded so the §8.5.5.5(f.6) pipe post-pass
/// can drive itself off the slot walk rather than off "the vertices this fill allocated"
/// ((f.6.1), "what an implementation must do"). The distinction has teeth: `land_on_corner` is
/// stated over `(i, j, k)` and is unrecoverable from a vertex key.
///
/// Recording is **off** unless a replant will actually run, so every other junction pays one
/// branch per interior vert and no allocation.
#[derive(Default)]
pub(super) struct AdjNetSlots {
    pub(super) walk: Option<Vec<(usize, usize, usize)>>,
}

impl AdjNetSlots {
    pub(super) fn off() -> Self {
        Self::default()
    }

    pub(super) fn recording() -> Self {
        Self {
            walk: Some(Vec::new()),
        }
    }

    pub(super) fn record(&mut self, slot: (usize, usize, usize)) {
        if let Some(walk) = &mut self.walk {
            walk.push(slot);
        }
    }

    pub(super) fn len(&self) -> usize {
        self.walk.as_ref().map_or(0, Vec::len)
    }

    pub(super) fn slot(&self, index: usize) -> Option<(usize, usize, usize)> {
        self.walk.as_ref()?.get(index).copied()
    }
}

/// The generating profile of one ring edge of a [V-ADJ] cap (BEVEL.md §8.5.5.5(a)),
/// re-samplable at any resolution. The subdivision-relaxation stencil re-pins its
/// boundary at every round's own doubled resolution (§8.5.5.5(c)), so the fill needs each
/// chain's *generator*, not merely its emitted rim verts: a beveled edge's cross-section
/// cell (§8.2.2.1), a reflex miter arc in its unit-square frame (§8.9.5.1), or a straight
/// patch chord (§8.9.5). The three kinds are interchangeable `chain_of` inputs — the fill
/// never tests which one it holds (§8.9.5.3).
#[derive(Clone, Copy)]
pub(super) enum RingChainProfile<'c> {
    Cell {
        cell: ProfileCell,
        profile: f32,
        /// The custom curve's two chains, their read schedule (BEVEL.md §8.6.3 /
        /// §8.6.3.1(4).4) and the sector's §8.6.3.1(6) handedness bit, or `None` for the
        /// analytic superellipse family. The bit must be the one the *emitted* chain was read
        /// through: the stencil re-pins its boundary off this generator, so a generator that
        /// mirrors the rim tears the cap's interior away from its own edge.
        custom: Option<CustomChainRead<'c>>,
    },
    Arc {
        b1: Vec3,
        b3: Vec3,
        midpoint: Vec3,
        /// The operation-wide `profile` scalar (§8.6.2). An arc start records the circle
        /// exponent, but that copy never selects a spacing table — the arc reads the run's
        /// own §8.6.4 table like every other chain (§8.9.5's exponent-scope note).
        profile: f32,
    },
    Chord {
        start: Vec3,
        end: Vec3,
    },
}

impl RingChainProfile<'_> {
    /// The chain's **raw** (un-resampled) `segments + 1` samples, oriented from the ring
    /// edge's opening boundvert. Serves three distinct roles, all raw: the §8.2.2.1.1
    /// *source* polyline, the stencil's pinned Dirichlet boundary at each round's
    /// resolution, and — at `segments = 2` — the raw coarse `(i, 0, 1)` control point the
    /// extraordinary-centre stencil consumes (§8.5.5.5, "the centre stencil consumes the
    /// raw coarse profile mids").
    ///
    /// The argument is a *requested resolution*, and for a custom curve the three roles are
    /// served by §8.6.3.1(4).4's schedule rather than by three fresh resamples: the
    /// intermediate cascade rounds and the `m = 2` coarse read stride the doubled chain,
    /// while a read at the operation's own count returns the target chain.
    pub(super) fn raw_positions(&self, segments: u32) -> Vec<Vec3> {
        match *self {
            RingChainProfile::Cell {
                cell,
                profile,
                custom,
            } => cell.raw_chain_positions(segments, profile, custom),
            RingChainProfile::Arc {
                b1,
                b3,
                midpoint,
                profile,
            } => arc_chain_positions(b1, b3, midpoint, profile, segments),
            RingChainProfile::Chord { start, end } => (0..=segments)
                .map(|k| start + (end - start) * (k as f32 / segments as f32))
                .collect(),
        }
    }

    /// Whether this ring edge samples a **custom** curve (§8.6.3). A custom chain "is not
    /// masked at all" by the §8.5.5.5(c) sharpening tap, so a cap built over custom chains
    /// relaxes its interior against the verbatim boundary.
    pub(super) fn is_custom(&self) -> bool {
        matches!(
            self,
            RingChainProfile::Cell {
                custom: Some(_),
                ..
            }
        )
    }
}

/// The [V-ADJ] cap's quad control net at one working resolution (BEVEL.md §8.5.5.5(a)):
/// `sectors` sub-patches, each a `(half + 1) × (half + 1)` grid of `(j, k)` samples, over
/// a chain resolution of `N = 2·half` per ring edge. Sub-patch `i`'s `j = 0` row is
/// chain `i` walked from `B_i`; its `k = 0` column is chain `i − 1` walked back to `B_i`;
/// its `k = half` column is the radial spoke it **shares** with sub-patch `i + 1`'s
/// `j = half` row; and `(half, half)` is the one shared centre. So the net is a closed
/// quad mesh whose only boundary is the `sectors` profile chains and whose only
/// non-valence-4 interior vertex is that centre — which is exactly what makes the
/// §8.5.5.5(c) stencil a plain Catmull–Clark round plus one Sabin extraordinary point.
#[derive(Clone)]
pub(super) struct AdjNet {
    pub(super) sectors: usize,
    pub(super) half: usize,
    /// The `sectors` raw profile chains at this resolution (`2·half + 1` samples each),
    /// oriented `B_i → B_{i+1}` — the net's Dirichlet boundary (§8.5.5.5(c)).
    pub(super) chains: Vec<Vec<Vec3>>,
    /// Whether the stencil's boundary-input copy is passed through the `(−1/6, +4/3, −1/6)`
    /// sharpening three-tap (§8.5.5.5(c)). The analytic superellipse chains are masked; a
    /// **custom** profile "is not masked at all" (§8.5.5.5(c)), so its chain feeds the
    /// interior stencil verbatim — masking a concave custom mid would drag the welded apex
    /// too deep into the solid.
    pub(super) mask_boundary: bool,
    pub(super) data: Vec<Vec3>,
}

impl AdjNet {
    pub(super) fn new(
        sectors: usize,
        half: usize,
        chains: Vec<Vec<Vec3>>,
        mask_boundary: bool,
    ) -> AdjNet {
        let stride = (half + 1) * (half + 1);
        let mut net = AdjNet {
            sectors,
            half,
            chains,
            mask_boundary,
            data: vec![Vec3::ZERO; sectors * stride],
        };
        net.pin_boundary();
        net
    }

    /// The coarse `segments = 2` control net of §8.5.5.5(a)/(b): the `s` boundverts, each
    /// ring edge's single **raw** mid-chain sample, and the fullness-seeded centre.
    pub(super) fn coarse(
        boundverts: &[Vec3],
        mids: &[Vec3],
        centre: Vec3,
        mask_boundary: bool,
    ) -> AdjNet {
        let sectors = boundverts.len();
        let chains = (0..sectors)
            .map(|i| vec![boundverts[i], mids[i], boundverts[(i + 1) % sectors]])
            .collect();
        let mut net = AdjNet::new(sectors, 1, chains, mask_boundary);
        net.set(0, 1, 1, centre);
        net
    }

    /// The owning slot of `(i, j, k)`: the `j == half` row is the *previous* sub-patch's
    /// `k == half` spoke, and every sub-patch's `(half, half)` is the one shared centre.
    pub(super) fn canonical(&self, i: usize, j: usize, k: usize) -> usize {
        let half = self.half;
        let (i, j, k) = if j == half && k == half {
            (0, half, half)
        } else if j == half {
            ((i + self.sectors - 1) % self.sectors, k, half)
        } else {
            (i % self.sectors, j, k)
        };
        i * (half + 1) * (half + 1) + j * (half + 1) + k
    }

    /// Read `(i, j, k)`, following the seam when an index runs one past `half`: sub-patch
    /// `i`'s chart continues into `i + 1` as `(i, j, half + t) ≡ (i + 1, half − t, j)`
    /// (and symmetrically backward across `j`). That is what lets the valence-4 vertex
    /// mask be written once and still be correct on the shared spokes.
    ///
    /// Both rewrites may fire on one query (see [`AdjNet::sample`]); the pair composes and
    /// terminates. What the chart genuinely cannot express is an index outside
    /// `0 ..= 2·half`: the forward rewrite would reflect it below zero. The rewrites
    /// preserve that bound, so asserting it on entry covers the recursion too.
    pub(super) fn at(&self, i: usize, j: usize, k: usize) -> Vec3 {
        let half = self.half;
        debug_assert!(
            j <= 2 * half && k <= 2 * half,
            "grid index outside the `[0, resolution]` chart",
        );
        if k > half {
            return self.at((i + 1) % self.sectors, half - (k - half), j);
        }
        if j > half {
            return self.at((i + self.sectors - 1) % self.sectors, k, half - (j - half));
        }
        self.data[self.canonical(i, j, k)]
    }

    pub(super) fn set(&mut self, i: usize, j: usize, k: usize, position: Vec3) {
        let slot = self.canonical(i, j, k);
        self.data[slot] = position;
    }

    pub(super) fn pin_boundary(&mut self) {
        let resolution = 2 * self.half;
        for i in 0..self.sectors {
            for k in 0..=self.half {
                let position = self.chains[i][k];
                self.set(i, 0, k, position);
            }
            let previous = (i + self.sectors - 1) % self.sectors;
            for j in 0..=self.half {
                let position = self.chains[previous][resolution - j];
                self.set(i, j, 0, position);
            }
        }
    }

    /// The stencil's *input* copy of the net (BEVEL.md §8.5.5.5(c)): the boundary chains
    /// passed through the `(−1/6, +4/3, −1/6)` three-tap. The mask **sharpens** — the
    /// discrete Laplacian is subtracted, so each sample is pulled a sixth *away* from the
    /// mean of its two chain neighbours. It touches only what the interior relaxes
    /// against; the boundary the net stores, and the rim the fill emits, are never masked.
    ///
    /// **The tap reaches only the FIRST HALF of each chain** — indices `1 ..= half`, where
    /// `half = N_in/2` at this round's *input* resolution (§8.5.5.5(c)). That bound is a
    /// property of the net's **storage**, not of the profile: only `k ≤ half` lives in
    /// sector `i`'s own `j = 0` row, while `k > half` is the same slot as sector `i + 1`'s
    /// `k = 0` column (`(i, 0, k) ≡ (i+1, N_in − k, 0)`), which this pass never writes. So
    /// the finished cap is rotationally symmetric but **not** mirror-symmetric, even on a
    /// perfectly symmetric corner — masking the whole chain gives a prettier, wrong answer.
    /// Inert at `half == 1`, where the lone interior sample already sits at `k = half`.
    ///
    /// Every tap reads the round's unmodified input chain and writes elsewhere (a Jacobi
    /// sweep): with the masked range a contiguous run, an ascending in-place update would
    /// compound the sharpening along it.
    pub(super) fn sharpened(&self) -> AdjNet {
        // §8.5.5.5(c): "A custom profile is not masked at all" — the chain feeds the
        // interior stencil verbatim, so the sharpening three-tap is suppressed.
        if !self.mask_boundary {
            return self.clone();
        }
        let half = self.half;
        let mut input = self.clone();
        for chain in input.chains.iter_mut() {
            let raw = chain.clone();
            for k in 1..=half {
                chain[k] = raw[k] - (1.0 / 6.0) * (raw[k - 1] + raw[k + 1] - 2.0 * raw[k]);
            }
        }
        input.pin_boundary();
        input
    }

    /// One cubic (Catmull–Clark, Sabin-modified) subdivision round (BEVEL.md §8.5.5.5(c)):
    /// face points, interior edge points, the valence-4 vertex mask `E − γ·F + γ·v`, and
    /// the extraordinary centre `E − γ_s·F + γ_s·centre` — all read off the *sharpened*
    /// boundary — then the doubled boundary re-pinned from `next_chains`, the raw profile
    /// samples at the new resolution. A single deterministic pass; no iteration.
    pub(super) fn subdivide(&self, next_chains: Vec<Vec<Vec3>>, gamma_s: f32) -> AdjNet {
        let sectors = self.sectors;
        let half = self.half;
        let input = self.sharpened();
        let mut next = AdjNet::new(sectors, 2 * half, next_chains, self.mask_boundary);

        for i in 0..sectors {
            for j in 0..half {
                for k in 0..half {
                    let face = 0.25
                        * (input.at(i, j, k)
                            + input.at(i, j, k + 1)
                            + input.at(i, j + 1, k)
                            + input.at(i, j + 1, k + 1));
                    next.set(i, 2 * j + 1, 2 * k + 1, face);
                }
            }
        }

        // Interior edge points. The chain-direction edges of the `j = 0` row and the
        // spoke-direction edges of the `k = 0` column are the net's *boundary* edges —
        // they carry no edge point, because the whole doubled boundary is re-pinned from
        // the profile instead. Every other edge is interior and flanked by two faces; the
        // `k == half` spoke edges take their second face from across the seam, which
        // `at`'s seam continuation supplies.
        for i in 0..sectors {
            for j in 1..half {
                for k in 0..half {
                    let point = 0.25
                        * (input.at(i, j, k)
                            + input.at(i, j, k + 1)
                            + next.at(i, 2 * j - 1, 2 * k + 1)
                            + next.at(i, 2 * j + 1, 2 * k + 1));
                    next.set(i, 2 * j, 2 * k + 1, point);
                }
            }
            for j in 0..half {
                for k in 1..=half {
                    let point = 0.25
                        * (input.at(i, j, k)
                            + input.at(i, j + 1, k)
                            + next.at(i, 2 * j + 1, 2 * k - 1)
                            + next.at(i, 2 * j + 1, 2 * k + 1));
                    next.set(i, 2 * j + 1, 2 * k, point);
                }
            }
        }

        // Ordinary (valence-4) interior verts, the shared spokes (`k == half`) included.
        for i in 0..sectors {
            for j in 1..half {
                for k in 1..=half {
                    let old = input.at(i, j, k);
                    let edges = 0.25
                        * (next.at(i, 2 * j - 1, 2 * k)
                            + next.at(i, 2 * j + 1, 2 * k)
                            + next.at(i, 2 * j, 2 * k - 1)
                            + next.at(i, 2 * j, 2 * k + 1));
                    let faces = 0.25
                        * (next.at(i, 2 * j - 1, 2 * k - 1)
                            + next.at(i, 2 * j + 1, 2 * k - 1)
                            + next.at(i, 2 * j - 1, 2 * k + 1)
                            + next.at(i, 2 * j + 1, 2 * k + 1));
                    next.set(i, 2 * j, 2 * k, edges - 0.25 * faces + 0.25 * old);
                }
            }
        }

        // The extraordinary centre (valence `sectors`): the mean of the `s` innermost edge
        // points, less `γ_s` times the mean of the `s` faces around it, plus `γ_s` times
        // its old position.
        let mean_edge = (0..sectors).fold(Vec3::ZERO, |acc, i| {
            acc + next.at(i, 2 * half - 1, 2 * half)
        }) / sectors as f32;
        let mean_face = (0..sectors).fold(Vec3::ZERO, |acc, i| {
            acc + next.at(i, 2 * half - 1, 2 * half - 1)
        }) / sectors as f32;
        let centre = self.at(0, half, half);
        next.set(
            0,
            2 * half,
            2 * half,
            mean_edge - gamma_s * mean_face + gamma_s * centre,
        );
        next
    }

    /// Read sub-patch `i` at fractional grid coordinates — the §8.5.5.5(d) resample
    /// kernel, bilinear within the source quad.
    ///
    /// The domain is the **whole** chart on both axes, `[0, resolution]` with
    /// `resolution = 2·half`, not the stored quadrant `[0, half]`, and a *well-conditioned*
    /// read-back leaves the quadrant too. At an even target the seam column's own chord
    /// fraction is read like every other index (§8.5.5.5(d)), and on a skewed cell it lands
    /// a fraction of a cell past the dense mid — the second-order gap between a chord table
    /// taken at `segments` and one taken at `resolution`; that is a `k`-axis over-run,
    /// resolved by §8.5.2's **forward** seam rule. A **degenerate** chain's `[0, …, 0, 1]`
    /// table is the extreme of the same thing on the other axis: it sends the neighbouring
    /// sub-patch's `j` coordinate to `resolution` outright (§8.9.5.4(c)), which is the
    /// **backward** rule. Clamping in either case would land on the seam spoke — the
    /// opposite end of the very axis the rule points at — so the cell corners are handed to
    /// [`AdjNet::at`], which carries both rewrites.
    ///
    /// A triple past `half` on **both** axes at once is **resolved here, not rejected**,
    /// and no guard stands against it. §8.5.2 asks for the opposite — it guards its backward
    /// rule with `k ≤ h` and says to reject a triple that has left the domain on both axes,
    /// on the reading that "in practice that triple never arises". That reading is wrong for
    /// this kernel. The two over-run mechanisms are **independent per-chain conditions**, so
    /// nothing about the construction rules the pair out: a degenerate *source* table drives
    /// the `k` axis to `resolution` at ordinary non-centre triples (measured `k = 8.0`,
    /// `j = 1.33` at `half = 4` inside the `inner_miter_arc_zero_spread_*` fixtures), a
    /// degenerate *target* table drives the `j` axis there
    /// (`degenerate_chord_table_read_reversed_*`), and a ring carrying one on chain `i` and
    /// the other on chain `i − 1` over-runs both. Ordinary input reaches it:
    /// `planar_fan_axis_aligned(12)` beveled on its `0° / 90° / 180°` spokes at
    /// `offset = 0.25`, `segments = 6`, `profile = 0.5`, `ADJ`, inner `ARC`, outer `SHARP`,
    /// `spread = 0` does, as do neighbouring offsets and counts
    /// (`zero_spread_arc_miter_fan_completes_*` is the pin). Rejecting would abort a bevel a
    /// caller may legitimately ask for — and abort it in *debug* builds, which is what the
    /// parity suite runs.
    ///
    /// So the cell corners go to [`AdjNet::at`] like any others, and its two seam rewrites
    /// compose in succession, `(i, j, k) → (i + 1, resolution − k, j) →
    /// (i + 2, resolution − j, resolution − k)`, which terminates on a stored slot and stays
    /// in bounds. That value is well defined but **undefined-*by-the-guide***: §8.5.2 neither
    /// defines the pair nor says which axis composes first, so nothing here claims the
    /// composition is the parity-correct one — only that it is total, deterministic, and
    /// in-bounds (`a_grid_read_past_half_on_both_axes_composes_the_two_seam_rewrites` pins
    /// it). Clamping instead would be the worse answer: it lands on the seam spoke, the far
    /// end of both axes the rules point at.
    ///
    /// The real invariant is the chart bound, `0 ≤ j, k ≤ resolution`, and it is upheld by
    /// the **arithmetic**, with no guard standing for it. A chain position carries
    /// `index ≤ resolution − 1` by [`chord_fraction_source_positions`]'s contract, and a ring
    /// position reaches `index == resolution` only through [`SourcePosition::reflected`]'s
    /// zero-remainder arm, where `hi` collapses back onto `lo`. A normalisation written for
    /// the far edge could therefore never fire; what it *would* do is discard the remainder
    /// of a genuinely out-of-chart position and rewrite it into a plausible-looking one,
    /// hiding it from [`AdjNet::at`]'s own bound assertion, which already covers both
    /// coordinates and both `hi` lines.
    ///
    /// **Both coordinates arrive as [`SourcePosition`] pairs and are consumed as pairs**: the
    /// pair *is* the `⌊pos⌋` / `pos − ⌊pos⌋` split the kernel needs, so there is no floor to
    /// take and — decisively — no `index as f32 + remainder` sum to round the weights
    /// through. Two consequences remove special cases rather than adding them. The far edge
    /// needs no clamp: a chain position of `resolution` arrives as `(resolution − 1, 1)`, and
    /// a ring position of `resolution` as `(resolution, 0)` whose zero remainder collapses
    /// `hi` back onto `lo`, so neither steps past the net's last line. And a zero remainder
    /// needs no branch: its two `hi` terms carry the exact factor `0`, so the sum collapses
    /// to a 1-D blend along the surviving line, and to a verbatim copy when both vanish. The
    /// mirror-image caution matters as much: a remainder of exactly `1` is **routine**, so
    /// nothing here may assert it away — at `remainder == 1` the two `lo` terms carry the
    /// exact factor `0` and the read is the `hi` line, which is what the `(lo + 1, 0)`
    /// spelling would have produced.
    pub(super) fn sample(&self, i: usize, ring: SourcePosition, chain: SourcePosition) -> Vec3 {
        let ring_hi = ring.index + usize::from(ring.remainder > 0.0);
        let chain_hi = chain.index + usize::from(chain.remainder > 0.0);
        let tj = ring.remainder;
        let tk = chain.remainder;
        self.at(i, ring.index, chain.index) * ((1.0 - tj) * (1.0 - tk))
            + self.at(i, ring.index, chain_hi) * ((1.0 - tj) * tk)
            + self.at(i, ring_hi, chain.index) * (tj * (1.0 - tk))
            + self.at(i, ring_hi, chain_hi) * (tj * tk)
    }

    /// The even target's welded centre `(i, half, half)`: the dense net's own centre vertex,
    /// taken **verbatim**, as its own rule rather than as a consequence of where the two
    /// flanking chord tables happen to land (§8.5.5.5(d), "What *is* pinned"). It is a
    /// statement about this **one** vertex — generalising it to the coordinate `half`, by
    /// pinning the mid entry of the read-back tables, is what displaces the seam columns.
    ///
    /// The observable consequence is that the even welded centre is a function of the
    /// **source** resolution alone: it holds constant while `segments` sweeps a whole
    /// power-of-two band and steps only when the band does.
    ///
    /// `None` at an odd target, where the grid closes on a central `s`-gon and there is no
    /// welded vertex for the rule to bite on, and at every non-centre index.
    ///
    /// `target_half` is the **target** grid's half-width `segments / 2`, which `(j, k)` are
    /// indices into; `self.half` is the **source** net's, which the returned vertex is named
    /// by. Confusing the two is the whole defect this rule exists to rule out, so they do not
    /// share a name.
    pub(super) fn welded_centre(
        &self,
        segments: usize,
        j: usize,
        k: usize,
        target_half: usize,
    ) -> Option<Vec3> {
        debug_assert_eq!(target_half, segments / 2);
        if segments % 2 != 0 || j != target_half || k != target_half {
            return None;
        }
        Some(self.at(0, self.half, self.half))
    }
}

/// The §8.5.5.5(d) read-back parametrisation, one map per ring chain: where target sample
/// `k` of that chain sits on the dense net's `resolution`-wide axis, as a fractional
/// index. It is the *same* chord-fraction reparametrisation the emitted rim uses
/// (§8.2.2.1.1), so a target grid column stands over the dense grid exactly where its own
/// rim sample stands — the interior meets the rim it is stitched to.
///
/// **Every index is read off the table, an even target's mid `half` included**
/// (§8.5.5.5(d), "the even-`segments` mid index"). Forcing that one entry to the dense mid
/// `resolution / 2` — so that the two sub-patches sharing the coordinate `half` are
/// "guaranteed to agree" — is the fill's classic mis-derivation. There is nothing to
/// reconcile: the grid is evaluated only at index-canonical triples (§8.5.2), so the shared
/// row has exactly one owner and one evaluation. Pinning it instead stands the seam column
/// over a different point of the dense polyline from the rim vertex it is stitched to,
/// which displaces `sectors · (half − 1)` cap verts on any skewed cell (it is the identity
/// on a mirror-symmetric one, at a power-of-two target, and at the line exponent — which is
/// why it hides). The one quantity that *is* pinned is the welded centre, and it is pinned
/// as its own rule ([`AdjNet::welded_centre`]), not through this table.
pub(super) fn adj_interior_axis(
    profiles: &[RingChainProfile],
    segments: u32,
    resolution: u32,
) -> Vec<Vec<SourcePosition>> {
    let seg = segments as usize;
    profiles
        .iter()
        .map(|profile| {
            if resolution == segments {
                (0..=seg).map(SourcePosition::at).collect()
            } else {
                chord_fraction_source_positions(
                    &profile.raw_positions(segments),
                    &profile.raw_positions(resolution),
                )
            }
        })
        .collect()
}

/// The **rim** row's own `j` source coordinate, `j*_rim = resolution − locate(1, g_{i−1})`
/// (BEVEL.md §8.5.5.5(d), "The rim's own `j` coordinate"), or `None` when it comes out zero.
///
/// The emitted rim `(i, 0, k)` is the `j = 0` row of the same two-axis read the interior
/// takes, not a standalone 1-D rule, so it carries a `j` coordinate like every other row.
/// Substituting `j = 0` makes the `j`-axis query `f^{(i−1)}_{segments}`, which is `1` for
/// every normalised table — so the coordinate is a quantity of the **previous** chain with
/// nothing of chain `i` in it, and `locate(1, g)` is the last entry of that chain's index
/// table.
///
/// `locate(1, g)` is `resolution` for any table whose final source chord survived
/// normalisation, which is every well-conditioned chain and also the fully degenerate
/// `[0, …, 0, 1]` fallback — hence the `None`, and hence the rim's usual collapse to the
/// pure `k`-axis read-back of §8.2.2.1.1 step 3 along source row `0`, which *is* the profile
/// polyline. That collapse is a **theorem about the table**, not a rule of the fill: a
/// **saturating** table — one that reaches `1` at some `p < resolution` and plateaus, which a
/// `spread = 0` inner-miter arc produces whenever the meet point's mantissa refuses the
/// two-term interpolation at some dyadic parameter (§8.9.5.4(c)) — leaves
/// `j*_rim = resolution − p > 0`, and then the next chain's rim is lerped along an interior
/// row of the relaxed net instead.
///
/// The chain that is degenerate is never the chain that moves: a collapsed chain's own rim is
/// one repeated point either way, and it is its **successor** around the ring that reads the
/// foreign row.
pub(super) fn adj_rim_source_row(
    axis: &[Vec<SourcePosition>],
    sectors: usize,
    segments: usize,
    resolution: usize,
    i: usize,
) -> Option<SourcePosition> {
    let previous = (i + sectors - 1) % sectors;
    // The rim is the `j = 0` row of the same two-axis read, so it goes through the ring
    // axis's own composite rather than a subtraction of its own. That composite is what
    // makes the saturation test right: at the query `1` the across-span quotient is
    // `span / span == 1.0` exactly, so the pair is the *upper*-endpoint spelling
    // `(p − 1, 1)` and its **index never reaches `resolution`** — comparing the index
    // against `resolution` would answer "saturating" on every well-conditioned table. The
    // reflection carries `(p − 1, 1)` to `(resolution − p, 0)`, and *that* being the origin
    // is exactly `g_{resolution − 1} < 1`, the guide's collapse condition.
    let row = ring_source_position(axis[previous][segments], resolution);
    (row != SourcePosition::ORIGIN).then_some(row)
}

/// Relax a general valence-`s` cap's coarse control net up to the working resolution the
/// target `segments` needs (BEVEL.md §8.5.5.5(c)): seed the fullness centre, then double
/// — **at least once, even at `segments = 2`** — until the chain resolution reaches
/// `next power of two ≥ max(4, segments)`, re-pinning each round's boundary from the ring
/// chains' own profiles at that round's resolution.
///
/// **Each round's boundary is an index stride of the ONE chain built at
/// `doubled_resolution`, never a fresh sampler call at the round's own resolution**
/// (§8.5.5.5(c) / §8.6.3.1(4).4). Even-chord spacing is refinement-consistent at exactly
/// four exponents — line, circle, square, square-in — so on any other exponent the two
/// readings are different chains, and only the stride is the operator's. Every cascade
/// resolution divides `doubled_resolution`, so the stride is exact and nothing is
/// interpolated.
pub(super) fn adj_relaxed_net(
    boundverts: &[Vec3],
    mids: &[Vec3],
    centre: Vec3,
    profiles: &[RingChainProfile],
    segments: u32,
) -> AdjNet {
    let gamma_s = sabin_gamma(boundverts.len());
    let target = boundary_source_resolution(segments);
    // §8.5.5.5(c): a custom-profile ring feeds its chains to the interior stencil verbatim
    // — no sharpening tap. The analytic superellipse chains are masked.
    //
    // A **mitered** ring reaches `false` here too, and it is the one ring whose two chain kinds
    // disagree: under a custom bevel its arc / patch chains stay analytic (§8.6.3.1(4).4's
    // carve-out) while its flanks read the curve. `any`, not `all`, because §8.5.5.5(c) states
    // the suppression of the profile *type* ("a custom profile is not tapped at all") the way
    // (f.3)'s law **F1** does — a per-operation key, not a per-chain switch — so the carve-out's
    // analytic arcs ride the ring's answer. Not a no-op at `N_in = 2`: the lone interior sample
    // *is* tapped there (only the half-chain and Jacobi scope rules go inert), so the answer
    // moves the welded apex.
    let mask_boundary = !profiles.iter().any(RingChainProfile::is_custom);
    let mut net = AdjNet::coarse(boundverts, mids, centre, mask_boundary);
    let source: Vec<Vec<Vec3>> = profiles
        .iter()
        .map(|profile| profile.raw_positions(target))
        .collect();
    while (2 * net.half as u32) < target {
        let resolution = 4 * net.half as u32;
        let stride = (target / resolution) as usize;
        let next_chains = source
            .iter()
            .map(|chain| {
                (0..=resolution as usize)
                    .map(|k| chain[k * stride])
                    .collect()
            })
            .collect();
        net = net.subdivide(next_chains, gamma_s);
    }
    net
}

/// The exact valence-6 Sabin extraordinary-point smoothing weight `γ_6` (BEVEL.md
/// §8.5.5.5(c)): the real root `x` of `x³ + (4k² − 3)·x − 2k = 0` at `k = cos(π/6)`,
/// fed through `γ_s = (k·x + 2k² − 1)/(x²·(k·x + 1))`. It is load-bearing to `1e-5`
/// for the inner-miter apex (§8.9.5); the rounded `0.523` overshoots the position
/// bucket by `≈4e-5`.
pub(super) const SABIN_GAMMA_6: f32 = 0.523_423_277;

/// The Sabin / Levin extraordinary-centre smoothing weight `γ_s` for valence `s`
/// (BEVEL.md §8.5.5.5(c)): from the real root `x` of `x³ + (4k² − 3)·x − 2k = 0` with
/// `k = cos(π/s)`, `γ_s = (k·x + 2k² − 1) / (x²·(k·x + 1))`. The load-bearing low
/// valences are pinned to the guide's tabulated constants (the fill's A/B compare is
/// sensitive to `γ_6` at the sixth digit); higher valences use the closed-form root
/// solved in `f64` (`sabin_gamma_formula`).
pub(super) fn sabin_gamma(valence: usize) -> f32 {
    match valence {
        3 => 0.065_247_584,
        4 => 0.25,
        5 => 0.401_983_447,
        6 => SABIN_GAMMA_6,
        _ => sabin_gamma_formula(valence) as f32,
    }
}

/// The closed-form Sabin weight `γ_s` in `f64` (BEVEL.md §8.5.5.5(c)). Solves the
/// depressed cubic `x³ + p·x + q = 0` (`p = 4k² − 3`, `q = −2k`, `k = cos(π/s)`) by
/// Cardano when it has a single real root (`p ≥ 0`) and by the trigonometric form
/// (taking the largest, tangent-continuity root) otherwise.
pub(super) fn sabin_gamma_formula(valence: usize) -> f64 {
    let k = (std::f64::consts::PI / valence as f64).cos();
    let p = 4.0 * k * k - 3.0;
    let q = -2.0 * k;
    let discriminant = (q / 2.0).powi(2) + (p / 3.0).powi(3);
    let x = if discriminant > 0.0 {
        let sqrt_d = discriminant.sqrt();
        (-q / 2.0 + sqrt_d).cbrt() + (-q / 2.0 - sqrt_d).cbrt()
    } else {
        let radius = 2.0 * (-p / 3.0).sqrt();
        let theta = ((3.0 * q) / (2.0 * p) * (-3.0 / p).sqrt()).acos();
        radius * (theta / 3.0).cos()
    };
    (k * x + 2.0 * k * k - 1.0) / (x * x * (k * x + 1.0))
}

/// The circular exponent's tabulated `fullness`, indexed by `segments` (BEVEL.md
/// §8.5.5.5(f.3), law **F2**). It runs to `segments = 11` and **stops** — past that even
/// the circular profile falls through to the regression **F3**. The table does not rise
/// monotonically: it oscillates with parity, even counts sitting below the odd counts that
/// bracket them, because an even cap closes on a welded apex and an odd one on a central
/// polygon (§8.5.5.5(e)) — which is why it is pinned directly rather than fitted.
pub(super) const ADJ_FULLNESS_CIRCLE: [f32; 12] = [
    0.0, 0.0, 0.559, 0.642, 0.551, 0.646, 0.624, 0.646, 0.619, 0.647, 0.639, 0.647,
];

/// The §8.5.5.5(b)/(f.3) centre-fullness fit scalar: how far the general ring's cap bulges
/// from the boundvert centroid toward the original corner `v`. It is a fit constant of the
/// profile and the segment count (separately for even and odd), not a per-vertex solve.
///
/// Three branches, tested in order — laws **F2** then **F3**. Law **F1** is never one of
/// them: a custom profile's fullness is [`custom_fullness_f1`] over its own chain, and the
/// general fill selects that *instead of* calling this function at all:
///
/// - the straight chord (`r = 1`) is exactly `0` — the cap is planar and its centre sits on
///   the boundvert centroid;
/// - the circular exponent is the tabulated [`ADJ_FULLNESS_CIRCLE`], for `1 ≤ n ≤ 11`;
/// - **everything else** — a curved exponent, the square exponents, *and the circular
///   exponent at `n ≥ 12`* — is an empirical linear regression in the **raw `profile`
///   scalar** (not in `r`), with a near-nil segment-count term and a separate even/odd fit.
///
/// Two things the law makes unavoidable. It **steps down** at the table's end rather than
/// asymptoting — `n = 11` (circle) is `0.647`, `n = 13` is `0.5777` — so the table must not
/// be extrapolated. And `fullness > 1` is normal on a bulging profile (`profile > 0.5`): the
/// fit runs past `1`, seeding the centre *beyond* `v`, and clamping it to `[0, 1]` "for
/// safety" would silently flatten every bulging cap.
///
/// The centre it seeds is Tier-2 (non-unique, §8.5.5.7). `profile = 0.5, segments = 2 →
/// 0.559` is the value the inner-miter apex of §8.9.5 consumes.
pub(super) fn adj_fullness(profile: f32, segments: u32) -> f32 {
    let exponent = profile_to_exponent(profile);
    if exponent == 1.0 {
        return 0.0;
    }
    if exponent == 2.0 && (segments as usize) < ADJ_FULLNESS_CIRCLE.len() {
        return ADJ_FULLNESS_CIRCLE[segments as usize];
    }
    let count = segments as f32;
    if segments % 2 == 0 {
        2.4506 * profile - 0.000_003_0 * count - 0.6266
    } else {
        2.3635 * profile + 0.000_152 * count - 0.6060
    }
}

/// The even-`segments = 2` extraordinary-centre apex of a valence-`s` adjacency grid
/// (BEVEL.md §8.5.5.5(b)–(d)), in closed form. `boundverts` are the `s` ring
/// meet-points in ring order and `mids[i]` is the single mid-chain sample of the
/// sector between `boundverts[i]` and `boundverts[(i+1) % s]`; `gamma_s` is the Sabin
/// weight for valence `s`. Seeds the fullness centre, runs one subdivision round
/// (boundary sharpening `(−1/6, +4/3, −1/6)`, per-sector face/edge points), and pulls
/// the seed in through the valence-`s` extraordinary-centre stencil `apex = E −
/// γ_s·F + γ_s·centre_0`. The even resample keeps this subdivided centre verbatim.
pub(super) fn adj_even_extraordinary_centre(
    vertex: Vec3,
    boundverts: &[Vec3],
    mids: &[Vec3],
    gamma_s: f32,
) -> Vec3 {
    let sectors = boundverts.len();
    let centroid = boundverts.iter().fold(Vec3::ZERO, |acc, &b| acc + b) / sectors as f32;
    let centre_0 = centroid + adj_fullness(0.5, 2) * (vertex - centroid);

    // The centre after one round depends only on the coarse net (the round's re-pinned
    // doubled boundary is a Dirichlet condition the *next* round reads, never this one),
    // so any well-formed doubled chain serves here.
    let doubled: Vec<Vec<Vec3>> = (0..sectors)
        .map(|i| {
            RingChainProfile::Chord {
                start: boundverts[i],
                end: boundverts[(i + 1) % sectors],
            }
            .raw_positions(4)
        })
        .collect();
    // This closed form serves the analytic superellipse caps only; the sharpening tap runs.
    AdjNet::coarse(boundverts, mids, centre_0, true)
        .subdivide(doubled, gamma_s)
        .at(0, 2, 2)
}
