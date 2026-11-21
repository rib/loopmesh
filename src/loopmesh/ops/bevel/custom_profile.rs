//! Custom profile curves: handles, sampling and budget distribution (BEVEL.md §8.6.3 /
//! §8.6.3.1).

use super::*;

impl BevelCurveProfile {
    /// Build a custom profile from a list of `(x, y, handle)` control points in the
    /// widget frame; the given handle kind is applied to both sides of each point. The
    /// list runs corner-to-corner (default start `(1, 0)`, end `(0, 1)`). `Free` /
    /// `Aligned` fall back to `Auto` here, as in [`Self::from_points_with_handles`].
    pub fn from_points(points: &[(f32, f32, BevelHandle)]) -> Self {
        BevelCurveProfile {
            points: points
                .iter()
                .map(|&(x, y, handle)| BevelCurvePoint {
                    x,
                    y,
                    h1: handle,
                    h2: handle,
                    h1_loc: None,
                    h2_loc: None,
                })
                .collect(),
        }
    }

    /// Build a custom profile whose control points carry a **different handle kind per
    /// side** — `(x, y, arriving, leaving)`, the arriving handle governing the segment that
    /// ends at the point and the leaving one the segment that starts there (BEVEL.md
    /// §8.6.3.1(1)). A segment counts as curved when *either* of its two bordering handles
    /// is non-vector, so per-side kinds are the only way to build a net that mixes straight
    /// and curved control segments.
    ///
    /// `Free` / `Aligned` carry an **explicit** handle location in the reference authoring
    /// model, and this constructor has nowhere to put one: it supplies no location, so
    /// §8.6.3.1(2)'s fallback fires and both kinds evaluate as `Auto`. They still differ
    /// from `Vector` for the curved/straight classification of §8.6.3.1(4), which reads the
    /// kind and not the location.
    pub fn from_points_with_handles(points: &[(f32, f32, BevelHandle, BevelHandle)]) -> Self {
        BevelCurveProfile {
            points: points
                .iter()
                .map(|&(x, y, arriving, leaving)| BevelCurvePoint {
                    x,
                    y,
                    h1: arriving,
                    h2: leaving,
                    h1_loc: None,
                    h2_loc: None,
                })
                .collect(),
        }
    }

    /// The **line** preset (BEVEL.md §8.6.3.1(3)): two auto points `(1, 0)` and `(0, 1)`.
    /// With the phantom-endpoint rule the auto handles fall on the line, so the evaluated
    /// segment is the straight diagonal chord — identical to the superellipse `r = 1`
    /// (line) case.
    pub fn preset_line() -> Self {
        Self::from_points(&[(1.0, 0.0, BevelHandle::Auto), (0.0, 1.0, BevelHandle::Auto)])
    }

    /// The custom profile's sample chain in the profile **cell** frame `(u_k, v_k)`
    /// (BEVEL.md §8.6.3.1): resample the control-point Bézier at even parameter (step 4),
    /// then apply the fixed axis swap `u = y`, `v = x` (step 5). `reverse` is the
    /// §8.6.3.1(6) handedness bit (see `widget_chain_to_cell`); it is a no-op on a
    /// diagonal-symmetric chain and only observable for an asymmetric control net.
    ///
    /// The ends are the cell ring corners `(0, 1)` / `(1, 0)` **only when** the chain runs
    /// corner-to-corner, i.e. on the `segments >= E` branch; below the control-segment
    /// count the sub-chain of §8.6.3.1(4).3 starts and ends wherever the surviving
    /// segments do, and nothing pins them. The ring fill overwrites index `0` and the last
    /// index with the sector's boundverts either way, so only the interior samples reach
    /// geometry.
    ///
    /// Public for an out-of-tree parity harness that pins the chain directly; no in-tree
    /// caller needs it outside this module's own tests.
    pub fn sample_chain(&self, segments: u32, reverse: bool) -> Vec<(f32, f32)> {
        widget_chain_to_cell(&self.widget_chain(segments), reverse)
    }

    /// The evaluated Bézier sample chain in the widget's `(x, y)` frame (BEVEL.md
    /// §8.6.3.1(2)/(4)): resolve every point's handle positions, split the `segments`
    /// budget across the control-point segments, and sample each segment at even Bézier
    /// parameter. The result has `max(segments, 1) + 1` entries — a zero-segment request
    /// still yields the two-sample chord, since there is no chain shorter than one advance.
    ///
    /// The chain runs from the first control point to the last **only when every segment
    /// receives an advance** — the `segments >= E` branch — and that shape is a
    /// consequence of the fill, never a forced overwrite. Below the control-segment count
    /// §8.6.3.1(4).3's sub-chain fires: each selected segment holds exactly one advance,
    /// every sample is a control point, neither end is pinned, and a run of skipped
    /// segments is a jump rather than a coarse walk.
    ///
    /// This is the whole evaluation model, and §8.6.3.1(4).4 caps a bevel at **two** runs of
    /// it: [`CustomChains::new`] calls here once at the operation's own `segments` and once at
    /// `doubled_resolution`, and every read at any other resolution is a stride of the second
    /// chain. So a *read site* calling this method would be the third run the schedule
    /// forbids — go through `CustomChains` instead. Also public for an out-of-tree parity
    /// harness that pins the chain directly.
    pub fn widget_chain(&self, segments: u32) -> Vec<(f32, f32)> {
        let n = segments.max(1) as usize;
        let points = &self.points;
        // A degenerate (< 2 point) definition has no segment to sample: fall back to the
        // straight diagonal, the line preset's shape.
        if points.len() < 2 {
            return (0..=n)
                .map(|k| {
                    let t = k as f32 / n as f32;
                    (1.0 - t, t)
                })
                .collect();
        }

        let handles = self.resolved_handles();
        let segment_count = points.len() - 1;
        let budgets = distribute_budget(points, &handles, n);

        let mut chain: Vec<(f32, f32)> = Vec::with_capacity(n + 1);
        for seg in 0..segment_count {
            let m = budgets[seg];
            if m == 0 {
                continue;
            }
            let p0 = (points[seg].x, points[seg].y);
            let c1 = handles[seg].right;
            let c2 = handles[seg + 1].left;
            let p3 = (points[seg + 1].x, points[seg + 1].y);
            for step in 0..=m {
                let t = step as f32 / m as f32;
                let sample = cubic_bezier(p0, c1, c2, p3, t);
                match chain.last_mut() {
                    // §8.6.3.1(4).2: consecutive contributions share one chain slot, and
                    // the incoming `t = 0` sample overwrites the outgoing `t = 1` one
                    // unconditionally — curve adjacency is never consulted, which is what
                    // turns a skipped run into a jump on the sub-budget branch.
                    Some(shared) if step == 0 => *shared = sample,
                    _ => chain.push(sample),
                }
            }
        }
        debug_assert_eq!(chain.len(), n + 1);
        chain
    }

    /// Resolve every control point's left / right handle location (BEVEL.md §8.6.3.1(2)):
    /// vector handles at one-third toward the relevant neighbour, auto handles from the
    /// smooth-tangent rule, with phantom-reflected neighbours at the two endpoints.
    pub(super) fn resolved_handles(&self) -> Vec<ResolvedHandles> {
        let points = &self.points;
        let count = points.len();
        (0..count)
            .map(|i| {
                let p = (points[i].x, points[i].y);
                let prev = if i == 0 {
                    // Phantom reflection of the real neighbour through the point.
                    let next = (points[1].x, points[1].y);
                    (2.0 * p.0 - next.0, 2.0 * p.1 - next.1)
                } else {
                    (points[i - 1].x, points[i - 1].y)
                };
                let next = if i + 1 == count {
                    let real_prev = (points[i - 1].x, points[i - 1].y);
                    (2.0 * p.0 - real_prev.0, 2.0 * p.1 - real_prev.1)
                } else {
                    (points[i + 1].x, points[i + 1].y)
                };
                let left = resolve_handle_side(points[i].h1, points[i].h1_loc, p, prev, next, true);
                let right =
                    resolve_handle_side(points[i].h2, points[i].h2_loc, p, prev, next, false);
                ResolvedHandles { left, right }
            })
            .collect()
    }
}

/// The custom-profile fullness law **F1** (BEVEL.md §8.5.5.5(f.3)) over a widget-frame chain:
/// `(1 / (2·n)) · Σ_{i=0}^{n-1} (x_i + y_i)`. The sum **includes** index `0` and **excludes**
/// index `n`, and the divisor is `2·n`, not `2·(n+1)`.
///
/// F1 is defined over the **widget** chain, unmirrored — attribution row 5 / §8.5.5.5(f.3) —
/// which is what this takes. The §8.6.3.1(5) axis swap is harmless either way, since `x + y`
/// is symmetric in its two arguments. The §8.6.3.1(6) mirror is *not*: it would be harmless
/// only on the `n ≥ E` corner-to-corner branch, where the two endpoints the mirror exchanges
/// both carry `x + y = 1` and the sum's asymmetric treatment of them cannot bite. On the
/// `n < E` sub-chain (§8.6.3.1(4).3) the ends need not be the widget corners at all — the
/// LATE_TURN net's `n = 2` chain ends carry `0.95` and `1.0` — so applying the mirror there
/// *would* move F1. Hence: read the widget chain, never a mirrored one.
///
/// The chain is the one at the operation's own `segments` — §8.6.3.1(4).4 row 5 — never a
/// strided read of the doubled chain.
pub(super) fn custom_fullness_f1(chain: &[(f32, f32)], segments: u32) -> f32 {
    let n = segments.max(1) as usize;
    debug_assert_eq!(
        chain.len(),
        n + 1,
        "F1 sums the chain built at its own `segments`",
    );
    let sum: f32 = chain.iter().take(n).map(|&(x, y)| x + y).sum();
    sum / (2.0 * n as f32)
}

/// The **two** custom sample chains a bevel materialises per profile, plus the schedule that
/// decides which of them every downstream read is served (BEVEL.md §8.6.3.1(4).4).
///
/// The evaluation model of §8.6.3.1(1)–(4).3 runs exactly **twice** per operation — once at
/// the operation's own `segments` (`target`) and once at `doubled_resolution` (`doubled`) —
/// and every read at any *other* resolution `m` is an **index stride** of `doubled`, never a
/// third run of the model at `m`. The [V-ADJ] cascade visits `2 → 4 → … →
/// doubled_resolution` (§8.5.5.5(c)) and the coarse control net is a read at `m = 2`, so the
/// distinction is reached on most segment counts.
///
/// The distinction is **not** custom-only. Even-**chord** spacing is refinement-consistent
/// at exactly four exponents — line, circle, square, square-in — so an analytic chain at any
/// other exponent is two different chains under the two readings too (§8.5.5.5(c)); a
/// circular-only suite cannot see it. A custom chain is worse still: §8.6.3.1(4).1's budget
/// is an integer partition over control segments recomputed from scratch for each `n`, so
/// halving `n` can move samples onto entirely different control segments.
///
/// Both chains are held in the **widget** frame. The §8.6.3.1(5) axis swap and the (6)
/// handedness mirror are per-read cell-mapping decisions, while the stride indexes the
/// profile's *own* resolution — so the stride must be taken first, on widget indices.
pub(super) struct CustomChains<'c> {
    pub(super) curve: &'c BevelCurveProfile,
    /// The chain at `n = segments`.
    pub(super) target: Vec<(f32, f32)>,
    /// The chain at `n = doubled_resolution`: the source polyline of every strided read and
    /// of the §8.2.2.1.1 chord read-back.
    pub(super) doubled: Vec<(f32, f32)>,
    pub(super) segments: u32,
    pub(super) doubled_resolution: u32,
    /// Law **F1** over `target`, cached: it depends only on the curve and the operation's
    /// `segments`, both fixed for the whole operation (§8.5.5.5(f.3)).
    pub(super) fullness: f32,
}

impl<'c> CustomChains<'c> {
    pub(super) fn new(curve: &'c BevelCurveProfile, segments: u32) -> Self {
        let doubled_resolution = boundary_source_resolution(segments);
        let target = curve.widget_chain(segments);
        // The two counts coincide at an exact power of two `≥ 4` and **nowhere else** — in
        // particular not at `segments = 2`, where the floor of `4` leaves a two-segment custom
        // bevel carrying two genuinely different tables.
        let doubled = if doubled_resolution == segments {
            target.clone()
        } else {
            curve.widget_chain(doubled_resolution)
        };
        let fullness = custom_fullness_f1(&target, segments);
        CustomChains {
            curve,
            target,
            doubled,
            segments,
            doubled_resolution,
            fullness,
        }
    }

    /// The widget-frame chain a read at `requested_resolution` is served, per §8.6.3.1(4).4's
    /// stride rule: `target[k]` when the request equals the operation's own `segments`, else
    /// `doubled[k·(doubled_resolution / requested_resolution)]`.
    ///
    /// **The `requested_resolution == segments` arm wins, and that priority is load-bearing at
    /// exactly one count.** The [V-ADJ] coarse control net is a read at resolution `2`
    /// (§8.5.5.5(a)), so at `segments == 2` it must be served `target[1]` — *not*
    /// `doubled[doubled_resolution / 2]`. The two expressions denote **different points**
    /// there: on the guide's split net they sit `0.5` apart in widget `x`, worth ≈1753
    /// canonical position quanta of cap apex. At every `segments ≥ 3` the same read takes the
    /// stride arm, and at an exact power of two `≥ 4` both arms are one table, so the ordering
    /// cannot be observed at all. Coding the stride arm unconditionally is therefore wrong on
    /// *every* two-segment custom bevel and right everywhere else — which is precisely the
    /// shape of defect a suite built on `segments ∈ {4, 8}` cannot see.
    pub(super) fn widget_chain_at(&self, requested_resolution: u32) -> Vec<(f32, f32)> {
        if requested_resolution == self.segments {
            return self.target.clone();
        }
        let stride = (self.doubled_resolution / requested_resolution.max(1)) as usize;
        if stride == 0 || stride * requested_resolution as usize != self.doubled_resolution as usize
        {
            // Every resolution the operator asks for is a power of two `≤ doubled_resolution`,
            // so the stride divides exactly and no interpolation ever happens here.
            debug_assert!(
                false,
                "resolution {requested_resolution} is off the doubling schedule of {}",
                self.doubled_resolution,
            );
            return self.curve.widget_chain(requested_resolution);
        }
        (0..=requested_resolution as usize)
            .map(|k| self.doubled[k * stride])
            .collect()
    }

    /// The chain a read at `requested_resolution` is served, mapped into the profile cell's
    /// `(u, v)` (§8.6.3.1(5)) with the §8.6.3.1(6) handedness bit applied.
    pub(super) fn chain_at(&self, requested_resolution: u32, reverse: bool) -> Vec<(f32, f32)> {
        widget_chain_to_cell(&self.widget_chain_at(requested_resolution), reverse)
    }
}

/// A control point's resolved left/right Bézier handle locations in the widget frame
/// (BEVEL.md §8.6.3.1(2)).
#[derive(Clone, Copy)]
pub(super) struct ResolvedHandles {
    pub(super) left: (f32, f32),
    pub(super) right: (f32, f32),
}

/// The fixed auto-handle reach tuning factor (BEVEL.md §8.6.3.1(2)) — reproduce verbatim
/// for parity; it has no simpler closed form.
pub(super) const CURVE_AUTO_HANDLE_FACTOR: f32 = 2.5614;

/// The vector handle's one-third reach as a **rounded `f32` constant** (BEVEL.md
/// §8.6.3.1(4).1, rule (b)). Scaling by `fl(1/3)` and dividing by `3` differ by an ulp on
/// some inputs, and the difference is not cosmetic: the turn measure reads a vector-handled
/// segment's two tangents as the *same chord reconstructed twice*, so its score is a
/// cancellation residue at the `2⁻²⁴` scale and every spelling choice on the way there moves
/// it. Dividing re-ranks the guide's `E = 5` residue net from `[2, 1, 0, 4, 3]` to
/// `[3, 2, 1, 0, 4]`, which lands its `n = 1` sub-chain on a different pair of control points.
pub(super) const CURVE_VECTOR_HANDLE_THIRD: f32 = 1.0 / 3.0;

/// Resolve one side of a control point's Bézier handle (BEVEL.md §8.6.3.1(2)). `is_left`
/// selects the arriving (`h1`, toward `prev`) vs the leaving (`h2`, toward `next`) side.
pub(super) fn resolve_handle_side(
    kind: BevelHandle,
    explicit: Option<(f32, f32)>,
    p: (f32, f32),
    prev: (f32, f32),
    next: (f32, f32),
    is_left: bool,
) -> (f32, f32) {
    match kind {
        BevelHandle::Free | BevelHandle::Aligned => {
            explicit.unwrap_or_else(|| auto_handle(p, prev, next, is_left))
        }
        BevelHandle::Vector => {
            let neighbour = if is_left { prev } else { next };
            (
                p.0 + (neighbour.0 - p.0) * CURVE_VECTOR_HANDLE_THIRD,
                p.1 + (neighbour.1 - p.1) * CURVE_VECTOR_HANDLE_THIRD,
            )
        }
        BevelHandle::Auto => auto_handle(p, prev, next, is_left),
    }
}

/// The auto (smooth-tangent) handle location (BEVEL.md §8.6.3.1(2)): `a = P − P_prev`,
/// `b = P_next − P` with each length floored to `1` when zero, `t = a/|a| + b/|b|`,
/// `L = |t|·2.5614`; the left handle is `P − t·(|a|/L)`, the right `P + t·(|b|/L)`. A zero
/// `L` (antiparallel unit tangents) leaves the handle on the point.
pub(super) fn auto_handle(
    p: (f32, f32),
    prev: (f32, f32),
    next: (f32, f32),
    is_left: bool,
) -> (f32, f32) {
    let a = (p.0 - prev.0, p.1 - prev.1);
    let b = (next.0 - p.0, next.1 - p.1);
    let mut len_a = (a.0 * a.0 + a.1 * a.1).sqrt();
    let mut len_b = (b.0 * b.0 + b.1 * b.1).sqrt();
    if len_a == 0.0 {
        len_a = 1.0;
    }
    if len_b == 0.0 {
        len_b = 1.0;
    }
    let t = (a.0 / len_a + b.0 / len_b, a.1 / len_a + b.1 / len_b);
    let l = (t.0 * t.0 + t.1 * t.1).sqrt() * CURVE_AUTO_HANDLE_FACTOR;
    if l == 0.0 {
        return p;
    }
    if is_left {
        (p.0 - t.0 * (len_a / l), p.1 - t.1 * (len_a / l))
    } else {
        (p.0 + t.0 * (len_b / l), p.1 + t.1 * (len_b / l))
    }
}

/// Evaluate a cubic Bézier at parameter `t` (BEVEL.md §8.6.3.1(4)).
pub(super) fn cubic_bezier(
    p0: (f32, f32),
    c1: (f32, f32),
    c2: (f32, f32),
    p3: (f32, f32),
    t: f32,
) -> (f32, f32) {
    let u = 1.0 - t;
    let w0 = u * u * u;
    let w1 = 3.0 * u * u * t;
    let w2 = 3.0 * u * t * t;
    let w3 = t * t * t;
    (
        w0 * p0.0 + w1 * c1.0 + w2 * c2.0 + w3 * p3.0,
        w0 * p0.1 + w1 * c1.1 + w2 * c2.1 + w3 * p3.1,
    )
}

/// A custom curve's sample chains bundled with the §8.6.3.1(6) handedness bit the reading
/// sector feeds them through. The two travel together because a chain read without its bit is
/// only half a cross-section: the bit picks which physical ring corner the asymmetric shape's
/// start ties to, and reading it from the wrong regime mirrors every interior sample.
#[derive(Clone, Copy)]
pub(super) struct CustomChainRead<'c> {
    pub(super) chains: &'c CustomChains<'c>,
    /// `reverse = !is_start_side` — see [`widget_chain_to_cell`].
    pub(super) reverse: bool,
}

impl<'c> CustomChainRead<'c> {
    /// The §8.6.3.1(6) **Regime B** read: a vertex-bevel cap sector, which runs no orientation
    /// walk and so takes the whole cap's one uniform bit.
    pub(super) fn vertex_cap(chains: &'c CustomChains<'c>) -> Self {
        CustomChainRead {
            chains,
            reverse: vertex_cap_reverse_bit(),
        }
    }

    /// The §8.6.3.1(6) **Regime A** read: an edge-mode sector, whose bit is the orientation
    /// pass's mark on the edge-end it spans.
    pub(super) fn edge_sector(chains: &'c CustomChains<'c>, reverse: bool) -> Self {
        CustomChainRead { chains, reverse }
    }
}

/// Map a widget-frame `(x, y)` chain into the profile cell frame (BEVEL.md §8.6.3.1(5)/(6)).
///
/// The `reverse` handedness bit (§8.6.3.1(6)) selects which physical ring corner an
/// asymmetric profile's shape-start ties to. The mapping itself pins nothing: it relabels
/// axes and index order, so whatever the chain's two ends were in the widget frame they
/// stay (the ring corners only when the chain ran corner-to-corner). The *ring fill*
/// overwrites both ends with the sector's boundverts regardless.
/// - `reverse = false` (forward): sample `k` → `(u_k, v_k) = (y_k, x_k)`, the plain axis
///   swap — shape-start (widget `(1, 0)`) ties to `B_k`.
/// - `reverse = true` (mirrored): sample `k` → `(u_k, v_k) = (x_(N−k), y_(N−k))`, i.e. the
///   chain read from the far end with the swap **dropped** — reflects every interior sample
///   across the cell diagonal `u = v` and reverses index order, tying shape-start to
///   `B_(k+1)`. On a diagonal-symmetric chain this equals the forward mapping.
pub(super) fn widget_chain_to_cell(chain: &[(f32, f32)], reverse: bool) -> Vec<(f32, f32)> {
    if reverse {
        chain.iter().rev().map(|&(x, y)| (x, y)).collect()
    } else {
        chain.iter().map(|&(x, y)| (y, x)).collect()
    }
}

/// One control segment's **turn magnitude** — the rank key of BEVEL.md §8.6.3.1(4).1.
///
/// Both tangents point the way the curve travels along the segment: the departure runs from
/// the start point to its *right* handle, the arrival runs from the end point's *left* handle
/// **into** the end point. Taken that way a straight chord scores `≈ 0` and a sharper bend
/// scores larger, up to `π`; heading the arrival at the handle instead yields exactly
/// `π − turn` on every segment — a clean rank inversion with no numerical noise to hint at it.
///
/// The measure normalises **first** and then reads the angle off a chord between the two unit
/// vectors, splitting on the sign of their dot with the boundary inclusive on the acute arm.
/// That is not an equivalent restatement of `acos(dot/(|d|·|a|))`: a zero-length tangent
/// normalises to the zero vector, which lands on `dot = 0` and so takes the acute arm, giving
/// the difference chord the surviving unit vector's own length. **Exactly one** zero tangent
/// therefore scores a fixed `π/3` — near the *top* of the ranking, not the bottom — while
/// both zero score `0`.
///
/// **Every arithmetic step here is part of the rule** rather than an implementation detail
/// (§8.6.3.1(4).1, rule (b)). On a vector-handled segment the two tangents are the *same*
/// chord reconstructed twice by two different round-trips, so the score is a cancellation
/// residue at the `2⁻²⁴ ≈ 5.96e-8` radian scale rather than an exact zero — and those residues
/// order geometrically-identical straight segments strictly, deciding the allotment before the
/// index tie-break is ever consulted. So: read the reconstructed handle *positions* and
/// difference them back out (never short-circuit a straight segment to `0`, never simplify the
/// round-trip algebraically — the simplification is exactly what cancels the residue), take
/// each length as `sqrt(dot)` in `f32`, and normalise by a single `f32` reciprocal multiplied
/// into both components. A componentwise divide is algebraically identical and numerically is
/// not: it re-ranks the guide's §(4).4 net from `[3, 2, 1, 0, 4]` to `[3, 1, 2, 0, 4]` and
/// moves that net's sub-chain. Widening to `f64` does not make the residues vanish either —
/// the same cancellation happens at `2⁻⁵³` and induces a different order once ties are broken.
pub(super) fn control_segment_turn(
    points: &[BevelCurvePoint],
    handles: &[ResolvedHandles],
    s: usize,
) -> f32 {
    let start = (points[s].x, points[s].y);
    let end = (points[s + 1].x, points[s + 1].y);
    let departure = (handles[s].right.0 - start.0, handles[s].right.1 - start.1);
    let arrival = (end.0 - handles[s + 1].left.0, end.1 - handles[s + 1].left.1);
    let unit = |v: (f32, f32)| {
        let length = (v.0 * v.0 + v.1 * v.1).sqrt();
        if length == 0.0 {
            (0.0, 0.0)
        } else {
            let reciprocal = 1.0 / length;
            (v.0 * reciprocal, v.1 * reciprocal)
        }
    };
    let departure_unit = unit(departure);
    let arrival_unit = unit(arrival);
    let dot = departure_unit.0 * arrival_unit.0 + departure_unit.1 * arrival_unit.1;
    let half_chord = |chord: (f32, f32)| {
        (0.5 * (chord.0 * chord.0 + chord.1 * chord.1).sqrt())
            .clamp(-1.0, 1.0)
            .asin()
    };
    if dot >= 0.0 {
        2.0 * half_chord((
            departure_unit.0 - arrival_unit.0,
            departure_unit.1 - arrival_unit.1,
        ))
    } else {
        std::f32::consts::PI
            - 2.0
                * half_chord((
                    departure_unit.0 + arrival_unit.0,
                    departure_unit.1 + arrival_unit.1,
                ))
    }
}

/// Distribute a chain budget of `n` interior advances across the control-point segments
/// (BEVEL.md §8.6.3.1(4), straight-edge sampling **off**). Returns the per-segment sample
/// count `m_i` (each segment sampled at `m_i + 1` Bézier points), with `Σ m_i = n`.
pub(super) fn distribute_budget(
    points: &[BevelCurvePoint],
    handles: &[ResolvedHandles],
    n: usize,
) -> Vec<usize> {
    let segment_count = points.len() - 1;
    // A segment is curved when either bordering handle is not a vector handle.
    let curved: Vec<bool> = (0..segment_count)
        .map(|s| points[s].h2 != BevelHandle::Vector || points[s + 1].h1 != BevelHandle::Vector)
        .collect();
    let mut curved_count = curved.iter().filter(|&&c| c).count();
    let all_curved = curved_count == 0;
    if all_curved {
        curved_count = segment_count;
    }
    let is_curved = |s: usize| all_curved || curved[s];

    let turn = |s: usize| control_segment_turn(points, handles, s);

    // Every segment ranked sharpest-first, scored once rather than once per comparison. The
    // key is the pair `(turn, index)` compared **descending on both** (§8.6.3.1(4).1, rule
    // (a)): among segments whose `turn` compares equal the one *later* along the curve takes
    // its advance first. Descending, not ascending — the "obvious" stable descending-`turn`
    // sort leaves ascending index among equal scorers and diverges on the first genuine tie,
    // which is a mainstream path rather than a corner: a staircase net's every segment is
    // axis-aligned, so both round-trips of rule (b) normalise to the same axis unit vector
    // bit-for-bit and the whole ranking is the tie-break's.
    //
    // `total_cmp` is safe because `turn` is never `-0.0`: both arms return `2·asin` of a
    // non-negative chord length (or π minus such a term), and `asin(+0) = +0`.
    let mut ranked: Vec<(f32, usize)> = (0..segment_count).map(|s| (turn(s), s)).collect();
    ranked.sort_by(|&(turn_a, a), &(turn_b, b)| turn_b.total_cmp(&turn_a).then(b.cmp(&a)));

    let mut budget = vec![0usize; segment_count];
    if n >= segment_count {
        let straight_count = segment_count - curved_count;
        let remaining = n.saturating_sub(straight_count);
        let per = remaining / curved_count;
        let leftover = remaining % curved_count;
        for s in 0..segment_count {
            budget[s] = if is_curved(s) { per } else { 1 };
        }
        // The remainder is what the even split left over *among the curved segments*, so
        // only they are offered it; the straight ones are already at their reserved one.
        // (An all-vector net has no curved segment at all, and §8.6.3.1(4).1's "if none are
        // curved, treat all as curved" makes the filter inert there — which is the branch the
        // guide's worked residue nets exercise.)
        for &(_, s) in ranked.iter().filter(|&&(_, s)| is_curved(s)).take(leftover) {
            budget[s] += 1;
        }
    } else {
        // §8.6.3.1(4).1's sub-budget branch: a plain top-`n` cut of the same ranking over
        // **every** segment, straight ones included, with no reservation and exactly one
        // advance each. Restricting the cut to the curved segments hands out fewer than `n`
        // advances once `curved_count < n`, and the short chain that follows becomes a ring
        // with the wrong vertex count — silently.
        for &(_, s) in ranked.iter().take(n) {
            budget[s] = 1;
        }
    }
    budget
}
