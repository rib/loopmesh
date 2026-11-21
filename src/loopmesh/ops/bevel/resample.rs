//! Chord-fraction resampling and source positions (BEVEL.md §8.2.2.1.1).

use super::*;

/// The source resolution of the §8.2.2.1.1 boundary resample: the round-over fill's
/// control cage is always 2 segments wide and always doubles at least once, so the
/// profile is materialised at `next power of two ≥ max(4, segments)` — **not** at
/// `2·segments`. It equals `segments` exactly at `4, 8, 16, …`, where no down-resample
/// runs and the emitted chain is the raw chain.
///
/// This is also §8.6.3.1(4).4's `doubled_resolution`, and the formula **already carries**
/// the floor of `4` — a custom profile must not apply it a second time.
pub(super) fn boundary_source_resolution(segments: u32) -> u32 {
    let mut resolution = 4;
    while resolution < segments {
        resolution *= 2;
    }
    resolution
}

/// The normalised chord fractions of a polyline (`0` at the first point, `1` at the
/// last).
///
/// The guard on the accumulated length is **exact** (BEVEL.md §8.2.2.1.1): a chain whose
/// samples all coincide has total `0`, and its table is neither left undivided, nor filled
/// uniformly, nor a signal to skip the resample. Only the *last* entry is set —
/// `[0, 0, …, 0, 1]` — so every interior query reads the source polyline at its `start`
/// end. Widening the test into a length floor is the larger divergence, because it changes
/// the answer for a band of chains whose fractions are perfectly well defined (§8.9.5.4(f));
/// the singularity is removable, not thresholded.
///
/// The table is deliberately **asymmetric under reversal**, and it *is* read both ways:
/// §8.5.5.5(d) reads one sector's table forward on the `k` axis and the neighbouring
/// sector's reversed on the `j` axis. Reversal is an **index** mirror, so `[0, …, 0, 1]`
/// presents as `[1, 0, …, 0]` — the two blocks flanking a degenerate chain are pulled to
/// opposite ends of their source axes.
///
/// **Accumulate in `f32`, take the square root per chord, and test the total exactly.**
/// Widening any of the three changes the table of a chain whose chords are ulp-scale, and
/// a `spread = 0` inner miter builds exactly such a chain on its target axis whenever the
/// resample runs (§8.9.5.4(c)/(g).3). That table — not the fallback — is what makes such a
/// miter emit a mirror-*asymmetric* cap on a mirror-symmetric junction.
pub(super) fn normalised_chord_fractions(points: &[Vec3]) -> Vec<f32> {
    let mut fractions = Vec::with_capacity(points.len());
    let mut travelled = 0.0;
    fractions.push(0.0);
    for pair in points.windows(2) {
        travelled += (pair[1] - pair[0]).length();
        fractions.push(travelled);
    }
    // `travelled > 0.0` is also false for NaN, and the fallback below would then dress a
    // broken chain up as a plausible degenerate one. A non-finite accumulated chord length
    // is an upstream invariant break, so say so rather than absorbing it.
    assert!(
        travelled.is_finite(),
        "chord-length accumulation over a profile chain must stay finite",
    );
    if travelled > 0.0 {
        for fraction in fractions.iter_mut() {
            *fraction /= travelled;
        }
    } else {
        for fraction in fractions.iter_mut() {
            *fraction = 0.0;
        }
        if let Some(last) = fractions.last_mut() {
            *last = 1.0;
        }
    }
    fractions
}

/// Read a polyline at a normalised chord fraction: [`SourcePosition::locate`] gives the
/// span and how far across it the fraction sits, and the point is the linear blend of that
/// span's two ends.
///
/// `points` must hold at least two samples, and `fractions` must be that same polyline's
/// [`normalised_chord_fractions`] table, so the located span index always addresses a real
/// pair of ends. Every caller materialises its source at `segments + 1` samples with
/// `segments >= 4`, so the floor states intent rather than naming a case to handle.
///
/// The span and remainder are derived by the same definition-first route the interior
/// read-back uses, not by a span search of this function's own, so **the located pair**
/// agrees between the two — the emitted rim and the interior positions that meet it are the
/// same query answered at two levels. What each does with the pair afterwards differs: no
/// snap happens here, whereas the interior read-back applies [`SourcePosition::snapped`]
/// first (and [`ring_source_position`] applies it twice).
pub(super) fn polyline_at_chord_fraction(
    points: &[Vec3],
    fractions: &[f32],
    fraction: f32,
) -> Vec3 {
    debug_assert_eq!(
        points.len(),
        fractions.len(),
        "a chord-fraction table must carry one entry per polyline sample",
    );
    let position = SourcePosition::locate(fraction, fractions);
    let start = points[position.index];
    let end = points[position.index + 1];
    start + (end - start) * position.remainder
}

/// A position on a source axis, as the `(span index, remainder)` **pair** BEVEL.md
/// §8.5.5.5(d) requires — never as the single float `index + remainder` denotes.
///
/// `remainder` is how far across span `index` the position sits, in units of one source
/// span, so it lies in `[0, 1]` **closed at both ends**. A remainder of exactly `1` is
/// routine rather than exceptional: any target fraction landing on a source sample produces
/// one, and the rim's `query = 1` does it deliberately. There is deliberately no
/// `remainder < 1` invariant here to assert.
///
/// **The pair is the specification, not an optimisation of one.** The
/// [`SOURCE_POSITION_INTEGRAL_EPSILON`] tolerance is measured on `remainder`, which is a
/// full-precision `f32` whichever span it came from. Collapsing to `index as f32 +
/// remainder` quantises it by half an ulp of the *position*, which grows with the index, so
/// the collapsed spelling silently acquires an **index-dependent** window — `+43%` at index
/// `8`, `−5%` at `16`, `+91%` at `32`, `+281%` at `64` — and, worse, *manufactures* exact
/// hits, where the sum rounds a remainder away before any tolerance is consulted and the
/// read emits a pure copy where a bilinear blend was owed. [`SourcePosition::value`]
/// therefore exists for value-form identities and for nothing else; it is never the input to
/// a tolerance test or to the kernel's weights.
///
/// An integral position `p` has two spellings, `(p, 0)` and `(p − 1, 1)`, and both occur:
/// [`chord_fraction_source_positions`] reaches an integer as `(p − 1, 1)` — its index never
/// reaches the axis's last sample at all — while [`SourcePosition::reflected`] always emits
/// `(p, 0)`. Neither [`SourcePosition::snapped`] nor [`AdjNet::sample`] can tell them apart.
/// The corollary is the one predicate that is easy to write backwards: **an index compared
/// against the axis length is false on every table**, so "does this table saturate" must be
/// asked of the pair as a whole (`index == last − 1 && remainder == 1` means it does not) or
/// of the reflected pair's being the origin, never of `index == last`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct SourcePosition {
    pub(super) index: usize,
    pub(super) remainder: f32,
}

impl SourcePosition {
    pub(super) const ORIGIN: Self = Self::at(0);

    pub(super) const fn at(index: usize) -> Self {
        Self {
            index,
            remainder: 0.0,
        }
    }

    /// The real number the pair denotes, for value-form identities only — never for the
    /// epsilon test and never for the kernel. See the type's own docs for why.
    pub(super) fn value(self) -> f32 {
        self.index as f32 + self.remainder
    }

    /// BEVEL.md §8.5.5.5(d)'s `locate(q, g)`: the **least** position at which `cumulative`'s
    /// piecewise-linear interpolant reaches `query`.
    ///
    /// `cumulative` is non-decreasing, starts at `0` and ends at `1`; `query` lies in
    /// `[0, 1]`. "Least" is the word doing the work — it resolves a plateau onto the
    /// plateau's *first* index, which is the whole of the saturating / degenerate
    /// distinction §8.9.5.4 turns on. "Find the span containing the query" is not a
    /// definition: on a table with ties it names several spans and picks none.
    ///
    /// Stating it as the definition rather than as a scan retires two guards a span search
    /// needs and this does not:
    ///
    /// * **No `0/0`.** `locate(0, g)` is the origin on *every* table, including one whose
    ///   head is collapsed (`g_1 = 0`), because nothing has been divided at that point.
    ///   That is the one case that made a division guard look load-bearing.
    /// * **No division guard.** The partition point itself gives
    ///   `cumulative[span_start] < query <= cumulative[reached]`, so it is the *selected
    ///   span* — not merely some entry below the reached index — that has strictly positive
    ///   width, and it has it on **every** table rather than only on well-conditioned ones.
    ///   The difference of two distinct `f32`s is correctly rounded, so it cannot underflow
    ///   to zero either. A guard here would
    ///   have to name a window on the *span* — a third tolerance in a third metric, in the
    ///   subsection whose thesis is that there is exactly one snap rule, on the remainder —
    ///   and it would not be value-neutral: at `query = 1` on a chain that concentrates its
    ///   length early and closes with one hair-thin final chord, forcing a true remainder of
    ///   `1` to `0` is a **whole-index** move, which [`Self::reflected`] carries into the
    ///   ring axis and which decides whether a rim row is read back at all.
    ///
    /// The returned index lies in `0 ..= cumulative.len() − 2`, so the axis's far end
    /// arrives as `(last − 1, 1)` and never as `(last, 0)`.
    pub(super) fn locate(query: f32, cumulative: &[f32]) -> Self {
        let reached = cumulative.partition_point(|&fraction| fraction < query);
        // The table ends at `1` and no query exceeds it, so some entry always reaches the
        // query. Assert rather than clamp: arriving here means a caller broke
        // normalisation, and a clamp would hide that behind a plausible-looking position.
        debug_assert!(
            reached < cumulative.len(),
            "a normalised chord table must reach every query asked of it",
        );
        if reached == 0 {
            return Self::ORIGIN;
        }
        let span_start = reached - 1;
        Self {
            index: span_start,
            remainder: (query - cumulative[span_start])
                / (cumulative[reached] - cumulative[span_start]),
        }
    }

    /// The one snap rule of §8.5.5.5(d): a *downward* attraction on the **remainder**.
    ///
    /// One unit of remainder is one source span, so the window is absolute in that unit —
    /// it does not scale with the position's magnitude, with the source resolution, with
    /// `segments`, with the profile exponent, or with the chain's length, and it is not a
    /// fixed ulp count. The reference applies exactly this rule once on the chain axis and
    /// twice on the ring axis, straddling the reflection; nothing else in the fill snaps.
    pub(super) fn snapped(self) -> Self {
        if self.remainder < SOURCE_POSITION_INTEGRAL_EPSILON {
            Self::at(self.index)
        } else {
            self
        }
    }

    /// `x ↦ resolution − x`, done on the pair so it never subtracts a position float.
    pub(super) fn reflected(self, resolution: usize) -> Self {
        if self.remainder == 0.0 {
            Self::at(resolution - self.index)
        } else {
            Self {
                index: resolution - self.index - 1,
                remainder: 1.0 - self.remainder,
            }
        }
    }
}

/// How far a source position may sit past a whole source sample and still be treated as
/// landing **on** it (BEVEL.md §8.5.5.5(d), "The tolerance's *metric*").
///
/// The unit is the **span remainder** — the second component of a [`SourcePosition`] — one
/// unit of which is one source span. Measured there the window has no binade structure at
/// all: it is a full-resolution `1e-6` of a span at every index, with no collapse and no
/// crossover. A fixed-ulp-of-*position* window is a genuinely different rule that crosses
/// this one at source position `4` and disagrees with it in opposite directions on either
/// side — narrower below (it cannot reach a position a hair above `0` at all, because it
/// scales with what it measures) and wider above, by `7.6×` at the dense mid of a net with
/// `N = 32` and `15×` at `N = 64`.
///
/// **What it is for.** Where the read-back is *mathematically* exact — a mirror-symmetric
/// cell's mid entry is the standing case, `f_h = 1/2` on the target polyline and
/// `g_{N/2} = 1/2` on the source one — the two halves of `1/2` are accumulated over
/// different chains and divided by different totals, so the quotient can land beside the
/// integer instead of on it, which is enough to tip the innermost seam entries into the
/// neighbouring position bucket. Measured on the mirror-symmetric [`skewed_cap_ring`], all
/// chains agreeing:
///
/// ```text
///   segments        6        10        12        16        20        24        34        40
///   N               8        16        16        16        32        32        64        64
///   index           3         7         8         7        16        15        32        32
///   remainder     1.0       1.0   8.15e-7       1.0   1.62e-6       1.0   4.85e-6   1.62e-6
///   snaps?          —         —       YES         —        no         —        no        no
/// ```
///
/// Half the counts carry **no residue at all**: the entry is exactly the dense mid in the
/// upper-endpoint spelling `(N/2 − 1, 1)`, where the kernel already emits the pure copy a
/// snap would produce, so no epsilon is consulted. `12` is the count the window exists for.
/// `40` is the one that shows why it must be measured on the remainder: collapsed to a
/// position it reads as an *exact* hit, because a `1.62e-6` remainder is under half an ulp
/// of position `32` — a pure copy manufactured by the sum, with no tolerance involved.
pub(super) const SOURCE_POSITION_INTEGRAL_EPSILON: f32 = 1.0e-6;

/// The **ring (`j`) axis's** source position: the *previous* chain's own position, snapped,
/// reflected through `resolution`, snapped again (BEVEL.md §8.5.5.5(d),
/// "The `1e-6` integer snap, pinned in one frame").
///
/// The two applications are the *same* [`SourcePosition::snapped`] the chain axis applies
/// once, in **application** order — the inner one before the reflection, on the previous
/// chain's own axis, the outer one after it. Because the reflection turns "just above an
/// integer" into "just below" one, the composite is **two-sided in the position frame**: it
/// fires exactly when `min(remainder, 1 − remainder) < 1e-6` on the *pre-reflection* pair,
/// and moves that position to the nearer integer.
///
/// The two windows meet at the integer rather than being disjoint, and that meeting point is
/// reachable: `resolution / 2` is the reflection's own fixed point, is integral on every net,
/// and is exactly what a mirror-symmetric cell lands on. There both applications are the
/// identity.
///
/// Testing symmetrically on the *post*-reflection position instead would be a different rule
/// — the reflection inverts the two ends, so the position the inner application attracts is
/// the **upper** endpoint of its span, not the lower one.
pub(super) fn ring_source_position(previous: SourcePosition, resolution: usize) -> SourcePosition {
    previous.snapped().reflected(resolution).snapped()
}

/// Where each target sample of the §8.2.2.1.1 resample lands on the source polyline,
/// as a **source-axis position** (`0 … source.len() − 1`) rather than a point. Same
/// chord-fraction rule, read out one level up: `chord_fraction_resample` is this map
/// composed with a lerp along the source span.
///
/// Each entry is [`SourcePosition::locate`] — the guide's `locate(q, g)` — asked of the
/// source table at that target sample's own chord fraction. The definition, its plateau
/// rule and the two guards it retires all live there; this function only supplies the pair
/// of tables and the query order.
///
/// This is what the §8.5.5.5(d) interior read-back parametrises against ("profile-arc
/// fractions"): the target grid's `k` axis *is* its chain, so a target column sits over
/// the dense grid at exactly the fractional column its rim sample sits at. That keeps the
/// interior consistent with the rim it meets, which uniform index fractions would not.
///
/// A **degenerate** chain — every sample the same point — carries
/// [`normalised_chord_fractions`]'s `[0, …, 0, 1]` and so maps to
/// `[0, …, 0, source.len() − 1]`. Uniform index fractions here would be a divergence, not a
/// harmless bail-out: this is the one place the degenerate table is *observable*, since the
/// resampled chain's own samples coincide whatever the table says (§8.9.5.4(g)).
///
/// The `spread = 0` inner-miter arc reaches that state on the **source** side only. Its
/// target chain is materialised at the run's own `segments`, and a resample runs at all
/// only when that is not a power of two — precisely when the collapsed chain's two-term
/// interpolation leaves ulp-scale, strictly positive chords and the zero-total guard does
/// *not* fire (§8.9.5.4(c)). So the usual query here is a fallback source table read
/// against an ordinary target one, which lands each interior sample at
/// `source.len() − 1 + fraction` — inside the last source span, where the §8.5.2 seam
/// rules take it into a neighbouring sector's slots.
///
/// **No snap happens here.** One table is consumed by *both* axes of the §8.5.5.5(d)
/// kernel under **different** sidedness — the chain axis reads it forward and snaps from
/// above only, the ring axis reads the *previous* chain's entry, snaps, reflects and snaps
/// again — so the tolerance cannot live in the shared producer without widening one axis or
/// narrowing the other. It lives at the two consumption sites,
/// [`SourcePosition::snapped`] and [`ring_source_position`].
///
/// The returned index lies in `0 ..= last − 1`, so the axis's far end arrives as
/// `(last − 1, 1.0)` and **never** as `(last, 0.0)`.
pub(super) fn chord_fraction_source_positions(
    raw: &[Vec3],
    source: &[Vec3],
) -> Vec<SourcePosition> {
    let target_fractions = normalised_chord_fractions(raw);
    let source_fractions = normalised_chord_fractions(source);
    target_fractions
        .iter()
        .map(|&fraction| SourcePosition::locate(fraction, &source_fractions))
        .collect()
}

/// The §8.2.2.1.1 chord-fraction resample: each raw target sample carries a chord
/// fraction along its **own** polyline, and the emitted sample is the point at that
/// same fraction along the **source** polyline (the profile materialised at the
/// doubling resolution). The two endpoints are fixed, so only the interior samples can
/// move; it is the identity wherever the source polyline already contains the raw
/// sample at the same fraction, so it is safe to run on every chain unconditionally.
pub(super) fn chord_fraction_resample(raw: &[Vec3], source: &[Vec3]) -> Vec<Vec3> {
    let target_fractions = normalised_chord_fractions(raw);
    let source_fractions = normalised_chord_fractions(source);
    let mut emitted = Vec::with_capacity(raw.len());
    emitted.push(raw[0]);
    for k in 1..raw.len() - 1 {
        emitted.push(polyline_at_chord_fraction(
            source,
            &source_fractions,
            target_fractions[k],
        ));
    }
    emitted.push(raw[raw.len() - 1]);
    emitted
}
