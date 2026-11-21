//! The bevel operator (BEVEL.md): its public options and result types, and the
//! [`LoopMesh::bevel`] entry point that dispatches to the build stages in the submodules.

use std::collections::{HashMap, HashSet};

use glam::{DVec3, Mat3, Vec2, Vec3};

use super::duplicate::GeomKey;
use super::mean_value::{mean_value_weights, PlaneProjection};
use crate::loopmesh::*;

mod adj_fill;
mod adj_grid;
mod adj_net;
mod adj_square;
mod adj_tri_corner;
mod chain;
mod corner_ring;
mod custom_orientation;
mod custom_profile;
mod cutoff;
mod donor;
mod edge;
mod edge_flags;
mod geometry;
mod harden;
mod interp;
mod junction;
mod meet;
mod miter;
mod miter_inner;
mod miter_outer;
mod offset;
mod pipe;
mod profile;
mod resample;
mod slide;
mod strip;
mod superellipse;
mod terminal;
mod terminal_meet;
#[cfg(test)]
mod tests;
mod vertex;
mod wire;

use adj_grid::*;
use adj_net::*;
use adj_square::*;
use adj_tri_corner::*;
use chain::*;
use corner_ring::*;
use custom_orientation::*;
use custom_profile::*;
use cutoff::*;
use geometry::*;
use interp::*;
use junction::*;
use meet::*;
use miter::*;
use offset::*;
use pipe::*;
use profile::*;
use resample::*;
use slide::*;
use strip::*;
use superellipse::*;
use terminal::*;
use terminal_meet::*;
use vertex::*;
use wire::*;

/// Which elements the bevel replaces (BEVEL.md §1.2 / §2.2).
///
/// [`BevelAffect::Vertices`] is the foundational single-segment vertex bevel of
/// BEVEL.md §3–§6. [`BevelAffect::Edges`] is the foundational single-segment edge
/// bevel of BEVEL.md §7 — implemented for the **isolated** single manifold edge
/// (no two beveled edges share a vertex); the multi-edge junction is a
/// documented-as-future stub (BEVEL.md §7.6).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum BevelAffect {
    Vertices,
    Edges,
}

impl Default for BevelAffect {
    fn default() -> Self {
        BevelAffect::Vertices
    }
}

/// How `offset` is interpreted (BEVEL.md §1.2 / §4 / §4.3). Each mode changes
/// only how the per-edge ring-vert slide distance `d_i` is derived from `offset`
/// and the local corner geometry; the ring / cap / re-stitch surgery downstream
/// is mode-agnostic (BEVEL.md §4.3).
///
/// All five are implemented for the foundational single-segment vertex bevel:
/// [`BevelOffsetType::Offset`] is the plain slide-along-the-edge distance
/// (`d_i = offset`, BEVEL.md §4); the other four re-measure that slide from a
/// different target quantity (BEVEL.md §4.3).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum BevelOffsetType {
    /// `d_i = offset` — distance slid along each incident edge (BEVEL.md §4).
    Offset,
    /// `d_i = offset / |2·sin φ_i|` — straight-line chamfer-width target.
    Width,
    /// `d_i = offset / |cos φ_i|` — perpendicular-depth target.
    Depth,
    /// `d_i = |A_i − v| · offset / 100` — per-edge fraction of edge length.
    Percent,
    /// `d_i = offset` — fixed slide independent of angle (= `Offset` at the
    /// foundational no-weights level, BEVEL.md §4.3).
    Absolute,
}

impl Default for BevelOffsetType {
    fn default() -> Self {
        BevelOffsetType::Offset
    }
}

/// How a multi-edge junction's corner patch is filled at `segments ≥ 2`
/// (BEVEL.md §1.2 / §8.3).
///
/// [`BevelVmeshMethod::Adj`] is the default rounded adjacency grid that blends the
/// arriving profile strips (a documented-as-future stub, BEVEL.md §8.4).
/// [`BevelVmeshMethod::Cutoff`] slices the corner off flat: one cut face per
/// sector spanning that sector's profile chain plus two bottom corner verts, the
/// bottom corner ring closed by one bottom face — or, at a 3-way junction whose
/// corner verts coincide, collapsed to a single apex (BEVEL.md §8.3).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum BevelVmeshMethod {
    Adj,
    Cutoff,
}

impl Default for BevelVmeshMethod {
    fn default() -> Self {
        BevelVmeshMethod::Adj
    }
}

/// How a junction's corner is reshaped when two beveled edges meet at an extreme
/// angle (BEVEL.md §1.2 / §8.9). `Sharp` is the default V-join the foundational
/// junction already builds; `Patch` and `Arc` replace that single point with a
/// small fill pushed out from the junction.
///
/// At a **non-reflex** (inner) sector the two non-sharp modes coincide — the inner
/// miter is always an arc start (§8.9.3), so `Patch` and `Arc` produce identical
/// inner geometry. The flat two-boundvert `Patch` strip is an **outer**-only
/// feature (§8.9.2).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum BevelMiter {
    Sharp,
    Patch,
    Arc,
}

impl Default for BevelMiter {
    fn default() -> Self {
        BevelMiter::Sharp
    }
}

/// Which face kinds the weighted-normal strength post-pass stamps (BEVEL.md
/// §8.10.1). The four modes form a nested inclusion — each stamps a superset of
/// the previous — so they compare as an ordered scale
/// `None < New < Affected < All` (the declaration order below).
///
/// - [`None`](Self::None) — the pass does not run; no layer is created.
/// - [`New`](Self::New) — stamp only the newly-created bevel faces (corner-cap →
///   weak, chamfer-strip → medium). Reconstructed / untouched originals stay at
///   the layer default (`0`).
/// - [`Affected`](Self::Affected) — additionally stamp reconstructed originals
///   (the faces the bevel re-stitched but did not create) strong.
/// - [`All`](Self::All) — additionally stamp untouched originals strong too, so
///   every face carries an explicit strength.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Default)]
pub enum BevelFaceStrengthMode {
    #[default]
    None,
    New,
    Affected,
    All,
}

/// Name of the face-domain 32-bit-integer customdata layer the strength
/// post-pass writes, shared by convention with the downstream weighted-normal
/// consumer so producer and consumer agree (BEVEL.md §8.10.1).
const FACE_STRENGTH_LAYER: &str = "__mod_weightednormals_faceweight";

/// The three strength sentinels stamped by mode (BEVEL.md §8.10.1). These exact
/// integers are part of the observable layer, not merely their ordering.
const FACE_STRENGTH_WEAK: i32 = -16384;

const FACE_STRENGTH_MEDIUM: i32 = 0;

const FACE_STRENGTH_STRONG: i32 = 16384;

/// Name of the per-loop `Vec3` customdata layer the §8.10.2 hardening post-pass
/// writes: the target **world-space** custom split normal for each loop of a new
/// bevel face. This is the decoded world vector (step 1 of §8.10.2); the
/// loop-normal-*space* encode/store is a later cycle, so loopmesh pins the
/// firewall-safe decoded value directly.
const HARDEN_LOOP_NORMAL_LAYER: &str = "__bevel_harden_loop_normal";

/// Provenance kind recorded for a **new** bevel face at creation time so the
/// §8.10.1 post-pass can key its strength on it. The junction vmesh patch and the
/// vertex-bevel cap fold into [`Cap`](Self::Cap) (weak); the profile / chamfer
/// strips fold into [`Strip`](Self::Strip) (medium). Reconstructed and untouched
/// originals are recovered by the post-pass from the pre-bevel face/vertex
/// snapshots, not tagged here.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum BevelNewFaceKind {
    Cap,
    Strip,
}

/// Which cross-section family a multi-segment bevel rounds toward (BEVEL.md §8.6.3).
///
/// [`Superellipse`](Self::Superellipse) is the analytic `u^r + v^r = 1` family the
/// `profile` scalar governs (§8.6.1). [`Custom`](Self::Custom) samples the chain from a
/// user-authored [`BevelCurveProfile`] instead — a parameter-spaced Bézier resample of a
/// handle-typed control-point list (§8.6.3.1), **not** the even-arc-length superellipse
/// solve. The type (not the `profile` value) also drives the terminal `√2` leg slide and
/// the three-boundvert un-fan (§8.3.0.1 / §8.3.0.2).
#[derive(Clone, Copy, Debug)]
pub enum BevelProfileType<'c> {
    Superellipse,
    Custom(&'c BevelCurveProfile),
}

impl Default for BevelProfileType<'_> {
    fn default() -> Self {
        BevelProfileType::Superellipse
    }
}

impl<'c> BevelProfileType<'c> {
    /// The custom curve when this is [`Custom`](Self::Custom), else `None`.
    fn custom(self) -> Option<&'c BevelCurveProfile> {
        match self {
            BevelProfileType::Custom(curve) => Some(curve),
            BevelProfileType::Superellipse => None,
        }
    }
}

/// A Bézier handle kind for a custom-profile control point (BEVEL.md §8.6.3.1(1)),
/// reusing the standard curve-widget vocabulary.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum BevelHandle {
    /// One-third of the way toward the neighbouring control point — a straight segment.
    Vector,
    /// A smooth tangent derived from the two neighbours (`t = a/|a| + b/|b|`).
    Auto,
    /// An explicit, independently-placed handle location.
    Free,
    /// An explicit handle whose opposite side is forced collinear through the point.
    Aligned,
}

/// One control point of a [`BevelCurveProfile`] (BEVEL.md §8.6.3.1(1)): a position in the
/// widget's `(x, y)` space, a per-side handle kind for the arriving (`h1`) and leaving
/// (`h2`) segment, and — for the free / aligned kinds — an explicit handle location per
/// side.
#[derive(Clone, Copy, Debug)]
struct BevelCurvePoint {
    x: f32,
    y: f32,
    h1: BevelHandle,
    h2: BevelHandle,
    h1_loc: Option<(f32, f32)>,
    h2_loc: Option<(f32, f32)>,
}

/// A user-authored custom bevel profile curve (BEVEL.md §8.6.3 / §8.6.3.1): an ordered
/// list of control points in the widget's unit-square `(x, y)` space, with the default
/// start at `(1, 0)` and end at `(0, 1)`. The chain a multi-segment bevel samples from it
/// is a **parameter-spaced** per-segment cubic-Bézier resample (§8.6.3.1(2)/(4)), mapped
/// into the profile cell by the fixed axis swap `u = y`, `v = x` (§8.6.3.1(5)).
#[derive(Clone, Debug)]
pub struct BevelCurveProfile {
    points: Vec<BevelCurvePoint>,
}

/// Parameters for [`LoopMesh::bevel`]. The foundational case (BEVEL.md §3) is
/// reachable by setting only `offset` and leaving every other field at its
/// default: `affect = Vertices`, `segments = 1`, `offset_type = Offset`.
#[derive(Clone, Copy, Debug)]
pub struct BevelOptions<'a> {
    /// How much material the bevel consumes, measured per `offset_type`. A
    /// non-positive offset is a no-op (BEVEL.md §2.1).
    pub offset: f32,
    pub offset_type: BevelOffsetType,
    pub segments: u32,
    pub affect: BevelAffect,
    /// When set, the effective offset is reduced just far enough that no ring
    /// vert ever slides past a collision point on its edge, so the result never
    /// self-overlaps (BEVEL.md §4.4). A global, uniform pre-pass on the resolved
    /// slide distances; it changes no topology, only the slide magnitudes.
    pub clamp_overlap: bool,
    /// Material slot **overridden** on every face the bevel creates (BEVEL.md §6.6).
    /// A leaf, per-face parameter that changes no topology and no placement.
    ///
    /// Not a two-regime switch: every new face is minted in two independent stages.
    /// Stage 1 copies the *whole* per-face attribute block of a representative
    /// original face whenever the fill supplies one (§6.6.1) — **unconditionally on
    /// `material`** — or lands on the layer defaults when it supplies none (§6.6.1.3).
    /// Stage 2 then, and only when `material >= 0`, overwrites the **material-slot
    /// field alone**; every other per-face layer keeps whatever stage 1 left there.
    ///
    /// So a donor-fed face at `material = 3` carries slot `3` *and the donor's other
    /// layers*; a template-less one carries slot `3` over a default block.
    ///
    /// Surviving re-stitched / re-pointed original faces are never re-materialised at
    /// any value (§6.6.2). Defaults to `-1` (no stamp: the inherited slot survives).
    pub material: i32,
    /// Propagate the **seam** edge flag across a multi-edge junction's bridging
    /// segments (BEVEL.md §6.7 / §6.7.4). A leaf, flag-only parameter that changes
    /// no topology and no placement.
    ///
    /// It gates **only** mechanism (2) of §6.7.1: the contiguity-bridging of the
    /// seam across the junction's bridging segments when seamed and non-seamed
    /// beveled edges are interleaved around the same junction. It is **inert** for
    /// vertex bevel (§6.7.2) and isolated single-edge bevel (§6.7.3), where the
    /// flag rides only the unconditional §6.5-style per-edge attribute copy
    /// (mechanism (1)), corrected by the §6.7.1.1 corner-segment contiguity AND.
    /// Defaults to `false`.
    pub mark_seam: bool,
    /// The same propagation for the **sharp** (not-smooth) edge flag (BEVEL.md
    /// §6.7). Runs the identical junction bridging on the sharp bit, independently
    /// of `mark_seam`. Defaults to `false`.
    pub mark_sharp: bool,
    /// How a multi-edge junction's corner patch is filled at `segments ≥ 2`
    /// (BEVEL.md §1.2 / §8.3). Only consulted for `affect = Edges`,
    /// `segments ≥ 2`. Defaults to `Adj` (the rounded grid), implemented at every
    /// smooth exponent for the cube corner (any `segments`) and the general
    /// valence-`s` corner (BEVEL.md §8.5 / §8.5.5.5). The `Cutoff`
    /// flat-cut fill is implemented for the multi-edge junction at the default
    /// profile (BEVEL.md §8.2 / §8.3).
    pub vmesh_method: BevelVmeshMethod,
    /// Cross-section shape for the `SUPERELLIPSE` profile family (BEVEL.md §8.6),
    /// a scalar in `0..1`. `0.5` (the default) is a circular arc, `0.25` flattens
    /// to a straight chamfer chord, larger values bulge toward a square corner.
    /// Only observable at `segments ≥ 2`: a single segment is a straight chamfer
    /// regardless of `profile` (§8.6.2). The mapping to the superellipse exponent
    /// `r` is logarithmic with `0.25 → r = 1` (line), `0.5 → r = 2` (circle),
    /// `→ 1 → r → ∞` (square). Ignored when `profile_type` is
    /// [`Custom`](BevelProfileType::Custom). Defaults to `0.5`.
    pub profile: f32,
    /// Which cross-section family the profile is sampled from (BEVEL.md §8.6.3):
    /// [`Superellipse`](BevelProfileType::Superellipse) (the `profile`-scalar analytic
    /// family, the default) or [`Custom`](BevelProfileType::Custom) (a user-authored
    /// [`BevelCurveProfile`]). The type also drives the terminal `√2` leg slide and the
    /// three-boundvert un-fan (§8.3.0.1 / §8.3.0.2). Only observable at `segments ≥ 2`.
    pub profile_type: BevelProfileType<'a>,
    /// Gate for the per-element bevel-weight prescale (BEVEL.md §8.7.2). When
    /// `false` (the default), weights are ignored entirely and every beveled
    /// element slides the full `offset` — byte-identical to the unweighted path.
    /// When `true`, each beveled vertex's resolved offset is prescaled by its
    /// weight (BEVEL.md §8.7.3); a beveled vertex *absent* from `vert_weights`
    /// defaults to weight `0.0` — no slide, no-op for that corner (§8.7.2 case 3).
    pub use_weights: bool,
    /// Per-vertex bevel weights consulted when `use_weights` is `true`, for
    /// `affect = Vertices` (BEVEL.md §8.7.1 / §8.7.3). A slice of
    /// `(VertexKey, weight)`; a beveled vertex's weight is looked up here, and an
    /// absent vertex resolves to `0.0` (§8.7.2). The weight multiplies that
    /// corner's offset uniformly across all its incident edges before the §4
    /// slide. Defaults to an empty slice.
    pub vert_weights: &'a [(VertexKey, f32)],
    /// Slide a junction boundvert **onto a single unbeveled edge** rather than
    /// leaving it at the two-strip meet-point (BEVEL.md §8.8). A pure placement
    /// perturbation: it changes no topology and no surgery, only where one
    /// boundvert lands in a junction sector. It fires for a sector with exactly
    /// one silhouette leg (or, absent silhouette legs, exactly one in-plane leg)
    /// when the slide does not fold either flanking side (§8.8.2), placing the
    /// boundvert on that leg's line by the meet-then-average rule of §8.8.3.
    /// Realised at `segments = 1` on the landed junction meet-point (§7.6.2); a
    /// no-op at a symmetric junction, diverging only at an irregular one
    /// (§8.8.5). Defaults to `false`.
    pub loop_slide: bool,
    /// How a junction's corner is reshaped at a **non-reflex** (inner) sector
    /// (BEVEL.md §8.9.3). `Sharp` (the default) keeps the foundational V-join; a
    /// non-`Sharp` value rebuilds every convex non-reflex sector as an **arc
    /// start** — one extra boundvert on the far flank, the original meet-point
    /// stored as the arc midpoint, and a circular miter profile the [V-ADJ] grid
    /// spans (§8.9.5). `Patch` and `Arc` produce identical inner geometry
    /// (§8.9.3). Engages on the [V-ADJ] fill (`vmesh_method = Adj`, `segments = 2`)
    /// at a convex tri-corner and at a junction whose one reflex sector the outer
    /// miter leaves `Sharp` (there the reflex sector keeps its meet-point and only
    /// the non-reflex sectors split); other junctions decline to `Sharp`.
    pub miter_inner: BevelMiter,
    /// How a junction's corner is reshaped at a **reflex** (outer) sector
    /// (BEVEL.md §8.9.2). `Sharp` (the default) keeps the foundational V-join; `Arc`
    /// and `Patch` rebuild the one reflex sector of an `s ≥ 3` junction past the
    /// §8.9.1 firing margin. Engages on the [V-ADJ] fill; other junctions decline to
    /// `Sharp`.
    pub miter_outer: BevelMiter,
    /// How far an inner `Arc` miter is pushed away from the junction along each
    /// beveled edge (BEVEL.md §8.9.4). Applies **only** to the inner miter; the
    /// outer miter ignores it. At `spread = 0` the two arc ends stay coincident
    /// with the meet-point and the arc collapses to a sliver, so a meaningful
    /// inner miter wants `spread > 0`. Defaults to `0.0`.
    pub spread: f32,
    /// Weighted-normal face-strength post-pass (BEVEL.md §8.10.1). A pure
    /// attribute pass that runs after the whole bevel is built: it stamps a
    /// per-face-kind strength sentinel (corner-cap → weak `-16384`, chamfer-strip
    /// → medium `0`, reconstructed / untouched original → strong `+16384`) into a
    /// named face-`Int32` layer (`__mod_weightednormals_faceweight`), gated by the
    /// nested [`BevelFaceStrengthMode`] scale. Creates and moves no geometry.
    /// Defaults to [`BevelFaceStrengthMode::None`] — the layer is not written at
    /// all, so existing callers are unaffected.
    pub face_strength_mode: BevelFaceStrengthMode,
    /// Hard-normal shading post-pass (BEVEL.md §8.10.2). A pure attribute pass
    /// that runs after the whole bevel is built: it marks the reconstructed↔new
    /// seam edges sharp, refreshes the face / vertex normals, then writes a
    /// per-loop **custom split normal** onto every loop of a *new* bevel face
    /// (corner-cap and chamfer-strip kinds) so the flat bevel reads as a crisp
    /// transition while the surrounding original surface keeps its smooth
    /// shading. Creates and moves no geometry. Gated on a nonzero offset; a leaf
    /// independent of `face_strength_mode` (either / both / neither may run).
    /// Defaults to `false` — the per-loop normal layer is not written at all, so
    /// existing callers are unaffected.
    pub harden_normals: bool,
}

impl Default for BevelOptions<'_> {
    fn default() -> Self {
        BevelOptions {
            offset: 0.0,
            offset_type: BevelOffsetType::Offset,
            segments: 1,
            affect: BevelAffect::Vertices,
            clamp_overlap: false,
            material: -1,
            mark_seam: false,
            mark_sharp: false,
            vmesh_method: BevelVmeshMethod::Adj,
            profile: 0.5,
            profile_type: BevelProfileType::Superellipse,
            use_weights: false,
            vert_weights: &[],
            loop_slide: false,
            miter_inner: BevelMiter::Sharp,
            miter_outer: BevelMiter::Sharp,
            spread: 0.0,
            face_strength_mode: BevelFaceStrengthMode::None,
            harden_normals: false,
        }
    }
}

/// New geometry created by the bevel, mirroring the operator's
/// `verts.out` / `edges.out` / `faces.out` slots (BEVEL.md §1.1). The buffers
/// are sets; downstream comparison ignores their enumeration order (§8).
#[derive(Default)]
pub struct BevelResult {
    pub verts: Vec<VertexKey>,
    pub edges: Vec<EdgeKey>,
    pub faces: Vec<FaceKey>,
}

impl LoopMesh {
    /// Foundational single-segment vertex bevel (BEVEL.md §3–§6).
    ///
    /// For each beveled vertex `v` of valence `n` — the **full** incident-edge count, wire
    /// spokes included (§7.2.1's vertex-mode box) — this replaces `v` with a ring of `n` new
    /// verts, one per incident edge, slid `offset` toward the far endpoint (§4); re-stitches
    /// each incident face's `v` corner onto the ring verts of the disk steps it spans (§6.1);
    /// and closes the ring with a single cap n-gon (§5 / §5.1). The ring is the disk **cycle**,
    /// so the cap is minted for *every* corner of valence >= 3 — interior, boundary, bow-tie and
    /// face-less wire star alike (§6.2, §6.2.1) — and the count delta is a function of `n`
    /// alone: `+(n−1)` V / `+n` E / `+1` F. Valence 2 is the one special case (§2.3): a
    /// two-corner n-gon is not a face, so the corner takes its single ring edge and no cap. The
    /// original vertex is then removed as a now-loose vertex (§5.1.1).
    ///
    /// Implemented for the foundational case: `affect = Vertices`,
    /// `segments = 1`, any `offset_type` (all five modes re-measure the per-edge
    /// slide per BEVEL.md §4 / §4.3). Single-segment `affect = Edges` is supported
    /// for the foundational **isolated** manifold edge (BEVEL.md §7), measured with
    /// `offset_type = Offset`. Multiple segments, profiles, miters, overlap
    /// clamping, flag propagation, and the multi-edge junction are
    /// documented-as-future stubs (BEVEL.md §7.6 / §8) and are ignored here.
    ///
    /// A non-positive `offset`, or a beveled vertex with fewer than two incident
    /// edges, is a no-op for that vertex (BEVEL.md §2.1).
    ///
    /// After the topological build, the `face_strength_mode` shading post-pass
    /// (BEVEL.md §8.10.1) runs when the mode is not
    /// [`BevelFaceStrengthMode::None`]: a pure attribute pass stamping the named
    /// weighted-normal strength layer per face kind. It touches no geometry.
    ///
    /// See `radial-mesh-guides/operators/BEVEL.md`.
    pub fn bevel(&mut self, geom: &[GeomKey], options: BevelOptions) -> BevelResult {
        let mode = options.face_strength_mode;
        let harden = options.harden_normals;

        // Neither shading post-pass wants provenance tracking — take the bare
        // topological path with no snapshot / recording overhead.
        if mode == BevelFaceStrengthMode::None && !harden {
            return self.bevel_dispatch(geom, &options);
        }

        // §8.10.1 / §8.10.2 recover reconstructed vs untouched originals from the
        // *pristine* face / vertex sets: a surviving original that gained a ring /
        // strip vert (or that was killed and rebuilt into a fresh key) is
        // reconstructed; everything still keyed and all-original-verts is
        // untouched. New bevel faces are recorded live (cap vs strip) into
        // `bevel_cap_faces`, and the chamfer strips' long edges (needed by the
        // §8.10.2 fan step) into `bevel_strip_long_edges`.
        let original_faces: HashSet<FaceKey> = self.face_keys().into_iter().collect();
        let original_verts: HashSet<VertexKey> = self.vert_keys().into_iter().collect();

        self.bevel_cap_faces = Some(Vec::new());
        self.bevel_strip_long_edges = Some(Vec::new());
        let result = self.bevel_dispatch(geom, &options);
        let cap_faces: HashSet<FaceKey> = self
            .bevel_cap_faces
            .take()
            .unwrap_or_default()
            .into_iter()
            .collect();
        let long_edges: HashSet<EdgeKey> = self
            .bevel_strip_long_edges
            .take()
            .unwrap_or_default()
            .into_iter()
            .collect();
        let new_faces: HashSet<FaceKey> = result.faces.iter().copied().collect();

        if mode != BevelFaceStrengthMode::None {
            self.stamp_face_strengths(
                mode,
                &original_faces,
                &original_verts,
                &new_faces,
                &cap_faces,
            );
        }
        // §8.10.2 is a leaf independent of §8.10.1: it runs whenever the caller
        // requested it and the bevel actually built new geometry (nonzero offset).
        if harden && !new_faces.is_empty() {
            self.harden_normals_pass(
                &original_faces,
                &original_verts,
                &new_faces,
                &cap_faces,
                &long_edges,
            );
        }

        result
    }

    /// The topological bevel build (BEVEL.md §3–§8), independent of the shading
    /// post-passes. Returns the created geometry; the public [`bevel`](Self::bevel)
    /// wraps this with the `face_strength_mode` post-pass (§8.10.1).
    fn bevel_dispatch(&mut self, geom: &[GeomKey], options: &BevelOptions) -> BevelResult {
        let result = BevelResult::default();

        if options.offset <= 0.0 {
            return result;
        }

        // The multi-segment edge-bevel junction (BEVEL.md §8.2 / §8.3 / §8.5) and the
        // general-valence vertex-mode cap (§8.11) are the `segments ≥ 2` paths
        // implemented. Both vmesh methods (Cutoff flat fill §8.3, Adj rounded fill §8.5)
        // share the same [P] ring / [S] chains / multi-segment strips; they differ only
        // in the junction fill. ADJ is distilled for the general non-symmetric /
        // general-valence case (§8.5.5.5).
        if options.segments >= 2 {
            if options.affect == BevelAffect::Edges {
                return self.bevel_edges(geom, options);
            }
            // Vertex-mode multi-segment (BEVEL.md §8.11, Stage [VV]). The
            // general-valence (`s ≥ 3`) / `segments ≥ 2` corner builds a real vmesh patch
            // for the circle, the curved band, both square sentinels, and — per §8.5.5.5(f.3)
            // law **F2** — the line / flat-chamfer profile, which is the *same* general fill
            // with `fullness = 0` (a planar cap) rather than a distinct construction. Only
            // valence < 3, wire spokes, and miters still decline to the historical no-op below.
            if options.affect == BevelAffect::Vertices {
                // §8.5.5.5(f.1): vertex mode never welds and never takes the tri-corner
                // family — the tri-corner gate refuses `affect == VERTICES` outright
                // (§8.5.1) — so every exponent, the flat line included, falls to the general
                // fullness grid, differing only in the boundary-ring sampler (§8.6) and the
                // centre fullness (§8.5.5.5(f.3) law **F2**: `r == 1.0` ⇒ `fullness = 0`,
                // routed through `emit_general_adj_fill`'s `flat` path).
                if let Some(built) = self.bevel_vertices_multiseg(geom, options) {
                    return built;
                }
            }
            return result;
        }

        if options.segments != 1 {
            return result;
        }

        if options.affect == BevelAffect::Edges {
            return self.bevel_edges(geom, options);
        }

        if options.affect != BevelAffect::Vertices {
            return result;
        }

        self.bevel_vertices_single_segment(geom, options)
    }
}
