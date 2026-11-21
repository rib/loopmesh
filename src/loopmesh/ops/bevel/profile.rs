//! Profile cells and the degenerate-frame predicates (BEVEL.md §8.2.2.1).

use super::*;

/// Degeneracy floor for the §8.2.2.1 terminal-far-end plane override: below this the
/// `terminal–start–end` triangle is treated as collinear (no well-formed normal) or
/// its normal as parallel to the edge (degenerate projection), so the default
/// mid-plane is kept instead of snapping.
pub(super) const TERMINAL_PLANE_TOLERANCE: f32 = 1.0e-4;

/// Cross-product length below which the §8.3.0.3 weld plane — and each welded profile's
/// existing projection plane — is treated as degenerate.
///
/// A profile **cell's** own degeneracy is *not* this test: §8.2.2.1 fixes that as an
/// angle band, [`near_parallel_directions`], because an area floor mis-reads the cell in
/// both directions.
pub(super) const COLLINEAR_MID_TOLERANCE: f32 = 1.0e-6;

/// (Anti)parallel tolerance on two unit plane **normals** for the §8.3.0.3 weld
/// override's per-profile coplanarity skip: a profile whose projection plane already is
/// the weld plane keeps its own normal.
pub(super) const COPLANAR_NORMAL_TOLERANCE: f32 = 1.0e-4;

/// BEVEL.md §8.6.3.1's `angle_between`, the operator-wide measure: **normalise first**,
/// then read the angle off a *chord* between the two unit vectors, branching on the sign
/// of their dot with the `dot = 0` boundary taken down the **acute** arm.
///
/// The branch is not a tie-break detail — it is the whole of the degenerate answer, and
/// the degenerate answer decides a branch in two separate places. §8.6.3.1: *"a zero
/// input gives `dot(d̂, â) = 0`, which takes the **acute** arm. With one input zero the
/// difference chord is the surviving unit vector's own length, `1`: `turn = 2·asin(1/2)
/// = π/3 = 60°` — a fixed constant, independent of the other input."* Both inputs zero
/// gives a zero chord and so `0°`, which is a *different* answer; §8.6.3.1 warns against
/// generalising the `60°` past the exactly-one-zero case.
///
/// Two spellings a reader reaches for first are wrong at exactly that input, and
/// §8.6.3.1 names both: `atan2(|a × b|, a · b)` on the raw inputs returns `0°`, and
/// `acos(dot)` after normalising returns `90°`. The `atan2` one is the load-bearing
/// misreading, because `0°` falls *inside* §8.2.2.1's `2°` band — see
/// [`near_parallel_directions`], whose zero-axis case §8.2.2.1 ("The other degeneracy")
/// turns on this `60°`.
///
/// Carry **one** of these. This is it; the custom-profile segment ranking of
/// §8.6.3.1(4).1 spells the same convention over its own 2-D control-net tangents.
pub(super) fn unit_chord_angle_between(first: Vec3, second: Vec3) -> f32 {
    let first_unit = normalise_at_direction_floor(first);
    let second_unit = normalise_at_direction_floor(second);
    let chord_angle = |chord: Vec3| 2.0 * (0.5 * chord.length()).clamp(-1.0, 1.0).asin();
    if first_unit.dot(second_unit) >= 0.0 {
        chord_angle(first_unit - second_unit)
    } else {
        std::f32::consts::PI - chord_angle(first_unit + second_unit)
    }
}

/// BEVEL.md §8.2.2.1's degeneracy predicate, on **normalised** directions: two
/// directions are near-parallel iff the angle between them is within `EPS_ANG = 2°` of
/// `0` or of `π` — the same angular ladder, at the same tolerance, §7.6.2 takes.
///
/// This is emphatically **not** a floor on the raw cross-product length. That reading
/// measures an *area*, so it is scale-dependent and wrong in both directions: far too
/// tight for a merely near-straight cell with long legs (a `1.31°` cell on legs of
/// `≈ 0.18` carries a raw cross of `7.7e-4`, three orders above a `1e-6` floor, so an
/// area-gated cell never rescues where it must), and far too loose for a small one (legs
/// of `1e-3` clear a `1e-6` floor at any angle at all). `2°` is wide enough to be a
/// *behavioural* boundary between two visibly different constructions rather than a
/// round-off guard, so neither end of that error is noise.
///
/// **A zero input is not "parallel to" anything.** §8.2.2.1 ("The other degeneracy"):
/// *"A zero axis normalises to zero, and a zero direction is not 'collinear with'
/// anything: fed to `directions_near_collinear` it comes out **60°** from `0°` and
/// `120°` from `180°`, so the predicate answers **false** and the rescue is skipped."*
/// That verdict is scoped to *this* predicate — the degenerate-`mid` rescue's **entry**
/// test — and it is what keeps a cell with a zero axis out of the rescue ladder
/// entirely. Catching such a cell is [`cell_frame_is_wellformed`]'s job, on a differently-shaped test.
pub(super) fn near_parallel_directions(first: Vec3, second: Vec3) -> bool {
    is_degenerate_direction_angle(unit_chord_angle_between(first, second))
}

/// BEVEL.md §8.2.2.1's **cell frame builder** and its own well-formedness test — a
/// separate, later step from the degenerate-`mid` rescue, and deliberately *not* a second
/// copy of the rescue's ladder. §8.2.2.1: *"it is shorter, asymmetric, and asks 'can this
/// parallelogram be built at all'."* Two clauses, in this order:
///
/// 1. **Either axis exactly zero, component-wise** — *"Not a length floor and not an
///    angle band."* It runs first, before any normalisation, because it exists for the one
///    case the rescue's normalised angular predicate structurally cannot see (a zero axis
///    reads `60°` there, so nothing upstream has certified such a cell), and because a
///    length floor here would start swallowing thin-but-valid cells. §8.2.2.1 calls this
///    *"the one place in the cell machinery where a tolerance would be actively wrong"*,
///    and pins the resulting ULP-scale discontinuity as the reference's own.
/// 2. **The two normalised axes within `EPS_ANG` of *anti*-parallel** — the `180°` end and
///    only that end. *"The `0°` arm is absent on purpose: a `0°` pair means `mid` sits on
///    the chord line outside the segment, which the rescue upstream has already caught."*
///    This clause is what turns away a tier-3 chord that reached the builder (its two axes
///    are exactly anti-parallel about the chord midpoint), so the collapsed and the
///    declined routes agree with no special case.
///
/// `false` means the chain emits the **even straight subdivision** of its chord instead of
/// a cell sample. Under clause (1) that carries *none* of tier 3's other consequences — no
/// linear flag, no zeroed projection direction — which is precisely what §8.3.0.3's merge
/// rule reads.
///
/// It is a predicate and not a frame *factory* on purpose: the cell map itself is spelled
/// once, in [`ProfileCell::sample`]. Handing the axes back here would give the map two
/// spellings that must agree for every cell the builder accepts.
pub(super) fn cell_frame_is_wellformed(start: Vec3, mid: Vec3, end: Vec3) -> bool {
    let a_axis = mid - start;
    let b_axis = mid - end;
    if a_axis == Vec3::ZERO || b_axis == Vec3::ZERO {
        return false;
    }
    unit_chord_angle_between(a_axis, b_axis) <= std::f32::consts::PI - MEET_DIRECTION_EPS_ANGLE
}

/// The interior samples of a chain whose cell frame **declined** (BEVEL.md §8.2.2.1
/// clause (1)) — the even straight subdivision of the chord — written in the **two-term**
/// form `(1 − t)·start + t·end`, evaluated componentwise in `f32`.
///
/// The spelling is part of the contract here, not a style choice (BEVEL.md §8.9.5.4(g).3).
/// The algebraically equivalent one-term `start + t·(end − start)` returns `start`
/// bit-exactly whenever the two ends **coincide**; this form returns the ends exactly only
/// at dyadic `t`, and otherwise lands an ulp off. That single ulp decides whether the
/// chain's own chord table is §8.2.2.1.1's zero-total fallback or the ordinary
/// normalisation of ulp-scale chords — and those two tables park the neighbouring sector's
/// interior block on different lines of the mesh (§8.9.5.4(c)). A chain materialised at a
/// power-of-two count takes the fallback either way; one materialised at a
/// non-power-of-two count does not, which is exactly the count set on which the [V-ADJ]
/// down-resample runs.
///
/// `Vec3::lerp` and any fused-multiply-add contraction *are* the one-term form. Do not
/// route this through either, and do not widen any of it to `f64`.
pub(super) fn collapsed_chain_sample(start: Vec3, end: Vec3, k: u32, n: u32) -> Vec3 {
    if k == 0 {
        return start;
    }
    if k == n {
        return end;
    }
    let t = k as f32 / n as f32;
    let s = 1.0 - t;
    Vec3::new(
        s * start.x + t * end.x,
        s * start.y + t * end.y,
        s * start.z + t * end.z,
    )
}

/// BEVEL.md §8.2.2.1's `near_collinear` on a profile cell's three control points, with
/// `apex` — the cell's `mid` — as the corner: the cell's two in-plane axes are
/// `apex − start` and `apex − end`, so the cell is degenerate exactly when those two are
/// [near-parallel](near_parallel_directions).
pub(super) fn near_collinear_control_points(start: Vec3, apex: Vec3, end: Vec3) -> bool {
    near_parallel_directions(apex - start, apex - end)
}

/// The 3-D frame of one sector's profile cell (BEVEL.md §8.2.2.1): the three
/// control points `start` / `end` / `mid` plus the derived in-plane axes and the
/// mid-plane normal. `start` / `end` are the two ring boundverts the chain spans;
/// `mid` is the bulge control point on the crossed beveled edge's line. A
/// degenerate (collinear) corner is flagged so the chain collapses to a straight
/// chord (the flat-profile branch, §8.2.2.1).
#[derive(Clone, Copy)]
pub(super) struct ProfileCell {
    pub(super) start: Vec3,
    pub(super) end: Vec3,
    pub(super) mid: Vec3,
    pub(super) mid_plane_normal: Vec3,
    /// The direction each cell sample is cast along to land on the profile plane
    /// (BEVEL.md §8.2.2.1). Initialised to the crossed beveled edge's unit direction,
    /// but **not synonymous with it**: the degenerate-`mid` rescue re-points it at the
    /// rescued plane's normal, and a tier-3 collapse zeroes it. Zero is the
    /// machine-checkable signature of a collapsed (straight-chord) profile — every
    /// downstream override keys on it — and requests no projection.
    pub(super) projection_dir: Vec3,
    pub(super) collinear: bool,
}

impl ProfileCell {
    /// Build the cell from the two ring boundverts and the crossed beveled edge
    /// (BEVEL.md §8.2.2.1). `mid` is the foot on the edge's line of the
    /// closest-approach pair between that line and the `start`–`end` line.
    pub(super) fn new(start: Vec3, end: Vec3, edge_point: Vec3, edge_dir: Vec3) -> Self {
        let mid = line_line_nearest_on_first(edge_point, edge_dir, start, end - start)
            .unwrap_or((start + end) * 0.5);

        let a_axis = mid - start;
        let b_axis = mid - end;
        let collinear = near_parallel_directions(a_axis, b_axis);
        // A zero axis leaves the cross exactly zero (§8.2.2.1: "The only frame field that
        // *does* degenerate is `mid_plane_normal` … and is therefore **zero**"), and the
        // rescue-entry predicate above reads that cell as `60°`, i.e. not collinear — so
        // the zero branch cannot be keyed on `collinear` alone.
        let mid_plane_normal = if collinear {
            Vec3::ZERO
        } else {
            a_axis.cross(b_axis).normalize_or_zero()
        };

        ProfileCell {
            start,
            end,
            mid,
            mid_plane_normal,
            projection_dir: edge_dir,
            collinear,
        }
    }

    /// The §8.2.2.1 degenerate-`mid` rescue — two tiers, then collapse. Runs as part of
    /// the frame derivation of every beveled-edge sector's cell (any `s`, any
    /// `segments`), and only bites when the closest-approach `mid` came out collinear
    /// with `start`/`end` — a planar corner's normal state, not a pathology.
    ///
    /// Tier 1 (unconditional) takes the bulge from the original vertex itself, which
    /// lies in the plane the corner's unbeveled legs span and off the `start`–`end`
    /// chord, so a planar corner still bevels as a *curve* and not a facet. Tier 2
    /// (only when both of the crossed edge's disk-neighbours are themselves beveled and
    /// the vertex carries `s ≥ 3` beveled edges — then there is no leg plane to stay in)
    /// takes the bulge from the meet of those two neighbours' offset lines instead, read
    /// **start-anchored**: the closest-approach point on the line through this profile's
    /// own `start`. Tier 3 — a still-collinear triangle, or a failed tier-2 meet — is the
    /// straight chord.
    ///
    /// "Collinear" throughout is §8.2.2.1's `EPS_ANG = 2°` band on the cell's two
    /// normalised axes, not an area floor: at a wide-`spread` inner miter the flanking
    /// cells sit a degree or so off straight while still carrying plenty of raw area, and
    /// they must rescue.
    ///
    /// A successful rescue re-planes the cell: the projection direction switches from the
    /// edge direction to the rescued plane's own normal, so the samples drop orthogonally
    /// into that plane rather than being cast along the edge. The plane's anchor stays
    /// `start`.
    pub(super) fn rescue_degenerate_mid(&mut self, rescue: &MidRescue) {
        if !self.collinear {
            return;
        }

        let mut mid = rescue.vertex;
        if let Some((direction_before, direction_after)) = rescue.neighbour_dirs {
            if near_parallel_directions(direction_before, direction_after) {
                self.collapse();
                return;
            }
            let meet = intersect_lines(
                &Line {
                    point: self.start,
                    direction: direction_before,
                },
                &Line {
                    point: self.end,
                    direction: direction_after,
                },
            );
            match meet {
                // The two lines are generally **skew**, and the bulge is the
                // closest-approach point on the **first** of them — the one anchored at
                // this profile's own `start`, along the crossed edge's disk-predecessor.
                // Not the second line's foot, and not the pair's midpoint. Both of those
                // readings are mirror-symmetric by construction, and the correct answer is
                // not: reflecting a junction reverses the ring's rotational sense, which
                // swaps which boundvert is `start`, so a sector and its mirror build the
                // same pair of skew lines and take *different* feet. The resulting
                // asymmetry on a symmetric fixture is the rule, not a defect in it.
                LineMeet::Crossing(point)
                | LineMeet::Skew {
                    on_first: point, ..
                } => mid = point,
                LineMeet::Collinear => {
                    self.collapse();
                    return;
                }
            }
        }

        if near_collinear_control_points(self.start, mid, self.end) {
            self.collapse();
            return;
        }
        let cross = (mid - self.start).cross(mid - self.end);
        // `cross` is exactly zero when the rescued `mid` lands *on* `start` or `end` — a
        // zero cell axis, which the re-check above reads as `60°` (§8.2.2.1, "The other
        // degeneracy") and therefore does not collapse. The ladder is then followed
        // literally: the rescue re-planes onto its own plane normal, which here is zero,
        // and `collinear` stays unset because no tier-3 ran. So such a cell reads
        // `emits_straight_chord` (its frame declines) while `is_tier3_collapsed` reads
        // **false** — the only state in this type where those two disagree. Both routes
        // emit the even chord, and §8.3.0.3's merge exception is a no-op at `s == 2`, so
        // the disagreement is not observable in the emitted positions.
        self.mid = mid;
        self.mid_plane_normal = cross.normalize_or_zero();
        self.projection_dir = self.mid_plane_normal;
        self.collinear = false;
    }

    /// Whether this profile went through the §8.2.2.1 **tier-3 collapse** — asked as the
    /// whole bundle that section gives the collapse, not as the chord *shape* alone.
    /// §8.2.2.1: *"the profile is flagged **linear**, ... `plane_point`,
    /// `mid_plane_normal` and `projection_dir` are all **zeroed**. '`projection_dir` is
    /// zero' is thus the machine-checkable signature of a collapsed profile."*
    ///
    /// Neither half alone is that signature for a cell built through
    /// [`LoopMesh::derive_profile_cell`]. [`ProfileCell::collinear`] is set the moment the
    /// closest-approach `mid` comes out degenerate, *before* the rescue ladder has decided
    /// between a recovered frame and a chord; and a zero `projection_dir` alone is also
    /// what a §7.5.4 zero-length beveled edge hands a perfectly uncollapsed cell. Taken
    /// together they are exactly "[`ProfileCell::collapse`] ran".
    ///
    /// The distinction matters because §8.3.0.3's merge exception is scoped to *"§8.2.2.1's
    /// tier-3 collapse"* by name — not to every cell that emits a straight chord.
    pub(super) fn is_tier3_collapsed(&self) -> bool {
        self.collinear && self.projection_dir.length_squared() <= f32::EPSILON
    }

    /// Whether [`cell_frame_is_wellformed`] accepts this cell's parallelogram (BEVEL.md
    /// §8.2.2.1, "The other degeneracy").
    pub(super) fn frame_is_wellformed(&self) -> bool {
        cell_frame_is_wellformed(self.start, self.mid, self.end)
    }

    /// Whether the chain through this cell is the **even straight subdivision**
    /// `lerp(start, end, k / segments)` rather than a cross-section sample.
    ///
    /// Two independent routes reach it, and BEVEL.md §8.3.0.3 turns on their being
    /// distinguishable: the §8.2.2.1 **tier-3 collapse** (which also flags the profile
    /// linear and zeroes its projection setup), and the frame builder **declining**
    /// (§8.2.2.1: *"the same points a tier-3 collapse emits, but reached by a different
    /// route and carrying **none** of tier 3's other consequences"*). Only the first is
    /// [`ProfileCell::is_tier3_collapsed`].
    pub(super) fn emits_straight_chord(&self) -> bool {
        self.collinear || !self.frame_is_wellformed()
    }

    /// The §8.2.2.1 tier-3 collapse: the profile is a straight chord, so `mid` is the
    /// plain chord midpoint and the projection setup is dropped wholesale.
    pub(super) fn collapse(&mut self) {
        self.mid = (self.start + self.end) * 0.5;
        self.mid_plane_normal = Vec3::ZERO;
        self.projection_dir = Vec3::ZERO;
        self.collinear = true;
    }

    /// The §8.3.0.3 weld-plane override for the two profiles of an `n == 2` junction
    /// ring: replace **only** each profile's plane normal with the normal of the single
    /// plane through the original vertex and the two boundverts, so the two chains —
    /// which span the same boundvert pair in opposite directions — sample to the *same*
    /// points and can be welded vertex-for-vertex.
    ///
    /// Each profile keeps its own anchor (its own chain `start`) and its own projection
    /// direction. The override is abandoned outright when either profile has collapsed
    /// to a chord (no projection direction to re-plane), when the weld plane itself
    /// degenerates, or when *both* profiles' existing projection planes degenerate; with
    /// exactly one degenerate the override still proceeds and that profile *is*
    /// re-planed. A profile whose projection plane already **is** the weld plane keeps
    /// its own normal — a per-profile skip, unlike the abandonment arms.
    pub(super) fn override_weld_plane(
        first: &mut ProfileCell,
        second: &mut ProfileCell,
        vertex: Vec3,
    ) {
        if first.projection_dir.length_squared() <= f32::EPSILON
            || second.projection_dir.length_squared() <= f32::EPSILON
        {
            return;
        }

        let to_first = vertex - first.start;
        let to_second = vertex - second.start;
        let weld = to_first.cross(to_second);
        if weld.length() <= COLLINEAR_MID_TOLERANCE {
            return;
        }
        let weld_plane_normal = weld.normalize();

        let existing_plane_normal = |cell: &ProfileCell, to_boundvert: Vec3| -> Option<Vec3> {
            let normal = to_boundvert.cross(cell.projection_dir);
            (normal.length() > COLLINEAR_MID_TOLERANCE).then(|| normal.normalize())
        };
        let normal_first = existing_plane_normal(first, to_first);
        let normal_second = existing_plane_normal(second, to_second);
        if normal_first.is_none() && normal_second.is_none() {
            return;
        }

        for (cell, existing) in [(first, normal_first), (second, normal_second)] {
            // A profile already coplanar with the weld plane keeps its own normal; a
            // degenerate `existing` cannot be (anti)parallel to anything, so it re-planes.
            let coplanar = existing
                .map(|normal| {
                    weld_plane_normal.dot(normal).abs() >= 1.0 - COPLANAR_NORMAL_TOLERANCE
                })
                .unwrap_or(false);
            if !coplanar {
                cell.mid_plane_normal = weld_plane_normal;
            }
        }
    }

    /// Build the cell from its three control points directly, with **no** beveled-edge
    /// projection (BEVEL.md §8.2.2.1: the projection exists to land a world cross-section
    /// on the crossed edge's mid-plane, and a cell whose control points already lie in the
    /// plane they belong to has nothing to project). This is what the unit-octant grid of
    /// §8.5.5.5(f.2) samples its sector arcs from: `start` / `end` are two octant axis
    /// points and `mid` their unit-square corner.
    pub(super) fn from_control_points(start: Vec3, end: Vec3, mid: Vec3) -> Self {
        let cross = (mid - start).cross(mid - end);
        let collinear = near_collinear_control_points(start, mid, end);
        ProfileCell {
            start,
            end,
            mid,
            mid_plane_normal: if collinear {
                Vec3::ZERO
            } else {
                cross.normalize_or_zero()
            },
            projection_dir: Vec3::ZERO,
            collinear,
        }
    }

    /// §8.2.2.1 terminal-far-end plane override: replace the default `start–mid–end`
    /// mid-plane normal with the normal of the triangle through the terminal vertex
    /// and the two chain endpoints, so the projection lands the far-end cross-section
    /// flat in the terminal's rim plane instead of bulging up along the tilted edge.
    /// Applied only when the triangle is well-formed (`terminal`, `start`, `end` not
    /// collinear) and the resulting normal is not parallel to the projection edge
    /// direction (which would make the projection degenerate); otherwise the default
    /// mid-plane stands. The plane's sign is irrelevant — it defines a plane, not a
    /// half-space — so no orientation is needed.
    pub(super) fn snap_terminal_plane(&mut self, terminal: Vec3) {
        // §8.2.2.1 precondition 1: a collapsed profile has no projection to re-plane.
        if self.projection_dir.length_squared() <= f32::EPSILON {
            return;
        }
        let to_start = (terminal - self.start).normalize_or_zero();
        let to_end = (terminal - self.end).normalize_or_zero();
        let candidate = to_start.cross(to_end);
        let length = candidate.length();
        if length <= TERMINAL_PLANE_TOLERANCE {
            return;
        }
        let candidate = candidate / length;
        if candidate.dot(self.projection_dir).abs() >= 1.0 - TERMINAL_PLANE_TOLERANCE {
            return;
        }
        self.mid_plane_normal = candidate;
    }

    /// Map a cell coordinate `(u, v)` into 3-D (BEVEL.md §8.2.2.1):
    /// `origin + u·a_axis + v·b_axis` with `origin = start + end − mid`, then
    /// project along the edge direction onto the mid-plane.
    pub(super) fn sample(&self, u: f32, v: f32) -> Vec3 {
        let origin = self.start + self.end - self.mid;
        let raw = origin + u * (self.mid - self.start) + v * (self.mid - self.end);
        if self.projection_dir.length_squared() <= f32::EPSILON || self.collinear {
            return raw;
        }
        intersect_line_plane(raw, self.projection_dir, self.start, self.mid_plane_normal)
            .unwrap_or(raw)
    }

    /// The chain's `segments + 1` sample positions (BEVEL.md §8.2.2.1 / §8.6).
    /// Endpoint samples are the ring boundverts; the interior samples are the
    /// `SUPERELLIPSE` cross-section cell coordinates `(u_k, v_k)` for the requested
    /// `profile` (§8.6.4, reducing to the §8.3.6 circular default at `profile =
    /// 0.5`) mapped through the cell. A collinear (flat) corner has no bulge and
    /// falls back to the even straight subdivision of the chord (§8.2.2.1).
    pub(super) fn chain_positions(
        &self,
        segments: u32,
        profile: f32,
        custom: Option<CustomChainRead<'_>>,
        emission: ChainEmission,
    ) -> Vec<Vec3> {
        let raw = self.raw_chain_positions(segments, profile, custom);
        if self.emits_straight_chord() {
            return raw;
        }
        if emission == ChainEmission::Raw {
            return raw;
        }
        // The source polyline is the profile at `doubled_resolution` — for a custom curve
        // literally §8.6.3.1(4).4's `doubled` chain, which the stride rule hands back at
        // stride 1 rather than re-running the evaluation model a third time.
        let source_segments = boundary_source_resolution(segments);
        if source_segments == segments {
            return raw;
        }
        let source = self.raw_chain_positions(source_segments, profile, custom);
        chord_fraction_resample(&raw, &source)
    }

    /// The chain's raw (un-resampled) `segments + 1` profile samples: the two ring
    /// boundverts and the cross-section cell coordinates between them. This is the
    /// coarse control chain the fill's centre stencil reads (§8.5.5.5(d)); the emitted
    /// rim is `chain_positions`' resample of it (§8.2.2.1.1).
    ///
    /// `segments` here is a *requested resolution*, not necessarily the operation's own: the
    /// [V-ADJ] cascade asks for `2, 4, 8, …`. A custom profile therefore answers through
    /// §8.6.3.1(4).4's schedule (`CustomChains::chain_at`), which knows the operation's count
    /// and so can tell the `m == segments` arm from the strided one.
    pub(super) fn raw_chain_positions(
        &self,
        segments: u32,
        profile: f32,
        custom: Option<CustomChainRead<'_>>,
    ) -> Vec<Vec3> {
        let mut positions = Vec::with_capacity(segments as usize + 1);
        positions.push(self.start);
        if self.emits_straight_chord() {
            for k in 1..segments {
                positions.push(collapsed_chain_sample(self.start, self.end, k, segments));
            }
        } else {
            // §8.6.3: a custom profile supplies its `(u, v)` chain directly (§8.6.3.1),
            // bypassing the superellipse exponent / even-arc-length solve entirely.
            let cells = match custom {
                // §8.6.3.1(6): the handedness bit rides the read, resolved by the caller —
                // Regime A's per-edge-end walk mark ([`CustomOrientation`]) at an edge-mode
                // sector, Regime B's uniform cap constant ([`vertex_cap_reverse_bit`]) at a
                // vertex-bevel cap.
                Some(read) => read.chains.chain_at(segments, read.reverse),
                None => superellipse_chain(profile, segments),
            };
            debug_assert_eq!(
                cells.len(),
                segments as usize + 1,
                "a chain holds one sample per segment boundary",
            );
            for &(u, v) in cells.iter().take(segments as usize).skip(1) {
                positions.push(self.sample(u, v));
            }
        }
        positions.push(self.end);
        positions
    }

    /// §8.2.2.1.1 at `segments = 2`: the chord-fraction resample of the single interior
    /// sample against the doubled (4-segment) profile. Identity on a symmetric cell
    /// (`f = 0.5` lands on `Q_2`); the skew pushes `f` toward the shorter leg.
    pub(super) fn segment2_boundary_resample(
        &self,
        profile: f32,
        custom: Option<CustomChainRead<'_>>,
    ) -> Vec3 {
        self.chain_positions(2, profile, custom, ChainEmission::Resampled)[1]
    }

    /// The profile's "height" (BEVEL.md §8.3.3): the span between the cell's
    /// `(0,0)` corner (the arc center `start + end − mid`) and its `(1,1)` corner
    /// (the bulge `mid`), i.e. `|2·mid − start − end|`.
    pub(super) fn height(&self) -> f32 {
        (2.0 * self.mid - self.start - self.end).length()
    }
}

/// How a cross-section chain emits its samples (BEVEL.md §8.2.2.1.1).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum ChainEmission {
    /// The raw profile chain. Emitted by everything §8.2.2.1.1's Scope blockquote puts
    /// outside the boundary resample: the **cutoff** fill (§8.3), which consumes the
    /// coarse chain verts directly, and the terminal far end (§6.7.5.4), which has no
    /// adjacency fill behind it. The resample is a [V-ADJ] *cap* property, and neither of
    /// those is a cap — the distinction is invisible on a symmetric cell, where the
    /// resample is the identity, and worth ≈25 quanta on a skewed one.
    Raw,
    /// The chord-fraction resample (§8.2.2.1.1 steps 1–3), run at every `segments` on every
    /// **sampled** ring-edge chain of a [V-ADJ] cap — the plain junction ring, the outer
    /// miter's arc, its flanking cross-sections, and its interior-sector strips (§8.9.5.1 /
    /// §8.9.5.2). No chain-kind test and no skew test: steps 1–3 are safe to run
    /// unconditionally, returning the sample unchanged wherever the gate says it should not
    /// move. Whether the emitted sample is then projected back onto the profile curve is a
    /// *grid-level* decision the fill makes (step 4), not a chain-level one.
    ///
    /// "Sampled" is the scope, and §8.5.5.5(f.4a)'s square-in weld is the one cap outside it:
    /// its ring is not sampled from a cross-section at all but stated in closed form at `n`,
    /// so there is no source polyline to read chord fractions off — see
    /// [`LoopMesh::build_square_in_weld_chains`].
    Resampled,
}
