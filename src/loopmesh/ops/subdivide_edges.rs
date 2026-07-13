use glam::Vec3;
use smallvec::SmallVec;

use super::fractal_noise;
use crate::loopmesh::*;

/// §4.3.1 — the flat-corner guard's collinearity tolerance: two cut edges
/// sharing a vertex are collinear there when `|cos θ| > 1 - FLAT_CORNER_COS_EPS`
/// (about half a degree either side of straight).
const FLAT_CORNER_COS_EPS: f32 = 5.0e-5;

/// Bit on the guard's transient edge op-flag layer: this edge is in the
/// operator's input set (§4.1's cut-edge count is read through it).
const CUT_EDGE: u8 = 0;

/// Bits on the guard's face op-flag layer. `FACE_SEEN` dedupes the candidate
/// walk (a face is reachable from each of its cut edges); `FLAT_CORNER` carries
/// the §4.3.1 verdict, read back by the per-face dispatch.
const FACE_SEEN: u8 = 0;
const FLAT_CORNER: u8 = 1;

/// Bits on the operator's vert op-flag layer. The two must stay **distinct**,
/// and conflating them is a live bug rather than a tidiness question (§3.1,
/// §7.0.3):
///
/// - `MIDPOINT` — a **pass-two** cut vert: one of the midpoints §3 minted on an
///   input edge. This is the predicate §7.0.3's cut-vert census keys on, and the
///   one §7.1 finds its runs from.
/// - `PASS_THREE_VERT` — an interior vert a pass-three *pattern* minted (§5.2.2's
///   and §5.4's `c`, the §9.5 lattice interiors). Such a vert can re-enter a
///   *pending* face's ring — the pattern splits an edge whose radial cycle holds
///   that face — and re-inflate a face an earlier connect had carved, back to the
///   very corner count the pending template was recorded with. The abandonment
///   gate therefore rejects any live ring carrying one: it is the direct,
///   winding-independent answer to "was I carved?", where a corner-count test
///   passes and an anchor-membership test is a coin-flip on which corner the
///   carve happened to peel.
///
/// Both verts are `geom_inner.out` members (§1), so an implementation that keys
/// the census on the *output* mark counts a pattern's `c` as a cut vert and
/// mis-passes exactly the case the census exists to catch.
const MIDPOINT: u8 = 1;
const PASS_THREE_VERT: u8 = 2;

/// Strategy for the quad-with-two-adjacent-cut-edges re-split
/// ([`SUBDIVIDE_EDGES.md`](../../../../radial-mesh-guides/operators/SUBDIVIDE_EDGES.md)
/// §5.2). Only [`QuadCornerType::StraightCut`] is part of the core single-cut
/// surface; the other three lay down distinct interior geometry and are tracked
/// as a separate parity row (§9.3).
///
/// The discriminants follow the documented enum order (§5.2): `INNERVERT = 0`,
/// `PATH = 1`, `FAN = 2`, `STRAIGHT_CUT = 3` — the default is the *last* value.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum QuadCornerType {
    InnerVert = 0,
    Path = 1,
    Fan = 2,
    StraightCut = 3,
}

impl Default for QuadCornerType {
    fn default() -> Self {
        QuadCornerType::StraightCut
    }
}

/// Falloff curve applied to the `smooth` offset
/// ([`SUBDIVIDE_EDGES.md`](../../../../radial-mesh-guides/operators/SUBDIVIDE_EDGES.md)
/// §9.1.3 step 1). Shapes the per-midpoint offset magnitude as a function of
/// the cut parameter `t` via a tent-and-curve. `Linear` is special-cased: it
/// bypasses the tent entirely and pins the falloff base at `1.0` (the full
/// offset, on-target at every midpoint); every other curve maps `[0,1] → [0,1]`
/// monotonically and adds `1.0` to its tent value, so a centre cut overshoots
/// `co_curved` ~2× (§9.1.3).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum SmoothFalloff {
    /// Smoothstep `3v² − 2v³` — the default gentle S-curve taper.
    Smooth,
    /// `sqrt(2v − v²)` — rounder, fuller near the ends.
    Sphere,
    /// `sqrt(v)` — fast rise, full early.
    Root,
    /// `v²` — slow rise, pinched.
    Sharp,
    /// Bypassed — the falloff base is held at `1.0`, a uniform full offset with
    /// no taper (the only curve that lands a centre cut exactly on `co_curved`).
    Linear,
    /// `v·(2 − v)` — similar to `Sphere`, slightly fuller.
    InverseSquare,
}

impl Default for SmoothFalloff {
    fn default() -> Self {
        SmoothFalloff::Smooth
    }
}

impl SmoothFalloff {
    /// The §9.1.3 step-1 falloff base for a cut parameter `t`. `Linear` returns
    /// `1.0` outright (the tent is skipped); every other curve shapes the tent
    /// value `tent = 1 − 2·|0.5 − t|` and adds the `1.0 +` base.
    fn base(self, t: f32) -> f32 {
        if self == SmoothFalloff::Linear {
            return 1.0;
        }
        let tent = 1.0 - 2.0 * (0.5 - t).abs();
        1.0 + self.curve(tent)
    }

    /// The §9.1.3 shaping curve `falloff_curve(val)`, each `[0,1] → [0,1]`,
    /// monotonic, fixing `0 ↦ 0` and `1 ↦ 1`.
    fn curve(self, v: f32) -> f32 {
        match self {
            SmoothFalloff::Smooth => v * v * (3.0 - 2.0 * v),
            SmoothFalloff::Sphere => (2.0 * v - v * v).max(0.0).sqrt(),
            SmoothFalloff::Root => v.max(0.0).sqrt(),
            SmoothFalloff::Sharp => v * v,
            SmoothFalloff::Linear => v,
            SmoothFalloff::InverseSquare => v * (2.0 - v),
        }
    }
}

/// Parameters for [`LoopMesh::subdivide_edges`].
///
/// Mirrors the BMOP set-input shape described in
/// [`SUBDIVIDE_EDGES.md`](../../../../radial-mesh-guides/operators/SUBDIVIDE_EDGES.md)
/// §1. The default value is the standard-subdivide core configuration: a single
/// midpoint cut per edge (`cuts = 1`), no smoothing or fractal displacement, the
/// per-face re-split patterns enabled (`use_grid_fill` + `use_single_edge` on),
/// and `STRAIGHT_CUT` corners — exactly what this operator implements today.
///
/// Turning a gate **off** does not decline the operator: the §3 midpoints are
/// still inserted, but the corresponding per-face re-split is suppressed and the
/// touched face is left as a flat-cornered n-gon (§1 pattern-gating note /
/// §9.3). Multi-cut (`cuts >= 2`, §3.3 / §9.5) is supported: the lattice fills
/// honour `use_grid_fill` / `use_single_edge` / `use_only_quads` the same way
/// the single-cut patterns do. The deferred knobs (smooth/fractal/sphere) are
/// not yet part of the surface and are at their no-op defaults.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SubdivideParams {
    /// Number of midpoint verts to insert per edge (§1 / §3.3). `1` is the
    /// single-cut core; `N >= 2` inserts `N` evenly-spaced verts per edge and
    /// fills each touched face with the §9.5 lattice.
    pub cuts: u32,
    /// Enables the one-cut re-split patterns (§5.1 / §6.1). On for the core.
    pub use_single_edge: bool,
    /// Enables the all-edges re-split patterns (§5.4 / §6.3). On for the core.
    pub use_grid_fill: bool,
    /// Corner strategy for the quad-two-adjacent case (§5.2). Core uses
    /// [`QuadCornerType::StraightCut`].
    pub quad_corner_type: QuadCornerType,
    /// Restrict the per-face re-split to quad faces (§9.3). Off by default. When
    /// on, triangles and n-gons keep their inserted midpoints but are left as
    /// flat-cornered n-gons — only quads re-split. Used by loop-cut.
    pub use_only_quads: bool,
    /// Catmull–Clark smoothing offset (§9.1). The master gain on the
    /// off-chord displacement: `0.0` (default) leaves every midpoint exactly on
    /// its §3 chord position, and the offset scales linearly in this value.
    pub smooth: f32,
    /// The falloff curve applied to the `smooth` offset (§9.1.3 step 1). No
    /// effect when `smooth == 0`.
    pub smooth_falloff: SmoothFalloff,
    /// Maintain an even offset *height* (rather than a fraction) when smoothing
    /// by multiplying in the §9.1.3 step-2 shell factor. No effect when
    /// `smooth == 0` or on a flat edge (parallel endpoint normals).
    pub use_smooth_even: bool,
    /// Project each inserted vert radially onto the origin-centred sphere of
    /// radius `smooth` (§9.4). The dome / to-sphere placement rule: every
    /// inserted vert (rim midpoint or interior/lattice node) is snapped to
    /// `normalize(co_chord) * smooth`; pre-existing corners are never moved.
    /// Mutually exclusive with the §9.1 `smooth` off-chord offset — when set,
    /// the §9.1 machinery is skipped and `smooth` is overloaded as the sphere
    /// radius (§9.4.3). `smooth_falloff` / `use_smooth_even` become inert.
    pub use_sphere: bool,
    /// Random displacement magnitude (§9.2). The master gain on the per-vert
    /// fractal jitter: `0.0` (default) is the no-op core (every inserted vert
    /// stays exactly where §9.1 / §9.4 / the chord placed it). A non-zero value
    /// roughens the subdivision — each inserted vert is pushed by a
    /// deterministic turbulence-noise displacement scaled by `fractal` and the
    /// cut edge's length (§9.2.3). Applied **last**, after the §9.4 sphere /
    /// §9.1 smooth / chord branch (§9.2.6), so `use_sphere` + `fractal`
    /// composes into a roughened dome.
    pub fractal: f32,
    /// Fraction of the fractal displacement constrained to the surface normal
    /// (§9.2.4), in `[0, 1]`. Scales **only** the two tangential displacement
    /// components by `(1 - along_normal)`; the normal-direction component is
    /// untouched. `0.0` (default) is a free 3-D jitter; `1.0` moves each vert
    /// only along the averaged endpoint normal (displacement-mapping style). No
    /// effect when `fractal == 0`.
    pub along_normal: f32,
    /// RNG seed for the fractal displacement (§9.2.1). Consumed exactly once,
    /// up front, to derive the fixed per-operator offset that shifts the noise
    /// sample domain — so two runs with the same `seed` produce bit-identical
    /// displacements regardless of vert count or visit order. No effect when
    /// `fractal == 0`.
    pub seed: i32,
}

impl Default for SubdivideParams {
    fn default() -> Self {
        Self {
            cuts: 1,
            use_single_edge: true,
            use_grid_fill: true,
            quad_corner_type: QuadCornerType::StraightCut,
            use_only_quads: false,
            smooth: 0.0,
            smooth_falloff: SmoothFalloff::Smooth,
            use_smooth_even: false,
            use_sphere: false,
            fractal: 0.0,
            along_normal: 0.0,
            seed: 0,
        }
    }
}

impl SubdivideParams {
    /// Set the number of midpoint verts inserted per cut edge (§1 / §3.3).
    /// Defaults to `1` (the single-cut core). `N >= 2` evenly spaces `N`
    /// midpoints at `t = i/(N+1)` along each cut edge and fills each touched
    /// face with the §9.5 lattice (the fully-cut quad `(N+1)x(N+1)` grid, the
    /// fully-cut triangle lattice, and the partial-cut fans). `cuts(0)` is a
    /// no-op cut count and leaves the mesh untouched.
    #[must_use]
    pub fn cuts(mut self, cuts: u32) -> Self {
        self.cuts = cuts;
        self
    }

    /// Toggle the one-cut re-split patterns (§5.1 / §6.1). On by default. With
    /// this off, a face with a single cut edge keeps its inserted midpoint but
    /// is left as a flat-cornered n-gon — the §5.1 / §6.1 surgery is skipped
    /// (§1 pattern-gating note / §9.3).
    #[must_use]
    pub fn use_single_edge(mut self, enabled: bool) -> Self {
        self.use_single_edge = enabled;
        self
    }

    /// Toggle the all-edges re-split patterns (§5.4 / §6.3). On by default.
    /// With this off, a fully-cut quad or triangle keeps its rim midpoints but
    /// is left as a flat-cornered n-gon — the §5.4 / §6.3 surgery is skipped
    /// (§1 pattern-gating note / §9.3).
    #[must_use]
    pub fn use_grid_fill(mut self, enabled: bool) -> Self {
        self.use_grid_fill = enabled;
        self
    }

    /// Select the corner strategy for the quad-two-adjacent case (§5.2).
    /// Defaults to [`QuadCornerType::StraightCut`] (the core single corner cut,
    /// §5.2.1). The three non-default strategies lay down distinct interior
    /// geometry: `INNERVERT` (§5.2.2) drops an inner vert on the corner cut and
    /// spokes it to the far corner, `PATH` (§5.2.3) chains the corner cut with a
    /// `v0`→`v2` second cut, and `FAN` (§5.2.4) spokes both rim midpoints to the
    /// far corner. Only the quad-two-**adjacent**-edges case consults this knob;
    /// every other re-split pattern is unaffected.
    #[must_use]
    pub fn quad_corner_type(mut self, corner_type: QuadCornerType) -> Self {
        self.quad_corner_type = corner_type;
        self
    }

    /// Toggle the quad-only re-split restriction (§9.3). Off by default. With
    /// this on, only quad faces re-split; a touched triangle or n-gon keeps its
    /// inserted midpoint(s) but is left as a flat-cornered n-gon — the §6 /
    /// §5.3 / §7 surgery is skipped. The gate filters the per-face dispatch
    /// only; every cut edge still gains its midpoint in §3.
    #[must_use]
    pub fn use_only_quads(mut self, enabled: bool) -> Self {
        self.use_only_quads = enabled;
        self
    }

    /// Set the Catmull–Clark smoothing offset (§9.1). Defaults to `0.0` (the
    /// core: every midpoint stays on its §3 chord). A non-zero value pushes each
    /// inserted midpoint off the chord toward the normal-guided curved target
    /// (§9.1.2), scaled by the falloff curve and (optionally) the shell factor
    /// (§9.1.3). The displacement is purely positional — topology and per-vert /
    /// per-loop customdata are unchanged (§9.1.4).
    #[must_use]
    pub fn smooth(mut self, smooth: f32) -> Self {
        self.smooth = smooth;
        self
    }

    /// Select the falloff curve applied to the `smooth` offset (§9.1.3 step 1).
    /// Defaults to [`SmoothFalloff::Smooth`]. No effect when `smooth == 0`. At a
    /// single centre cut every non-linear curve overshoots `co_curved` ~2×;
    /// only [`SmoothFalloff::Linear`] lands the midpoint on-target.
    #[must_use]
    pub fn smooth_falloff(mut self, falloff: SmoothFalloff) -> Self {
        self.smooth_falloff = falloff;
        self
    }

    /// Toggle the even-offset shell factor (§9.1.3 step 2). Off by default. When
    /// on, the `smooth` offset is multiplied by `1 / cos(half-angle between the
    /// endpoint normals)` so the bulge keeps a consistent *height* across edges
    /// of differing crease angle. A no-op when `smooth == 0` or on a flat edge
    /// (parallel normals give a shell factor of `1.0`).
    #[must_use]
    pub fn use_smooth_even(mut self, enabled: bool) -> Self {
        self.use_smooth_even = enabled;
        self
    }

    /// Toggle the dome / to-sphere projection (§9.4). Off by default. When on,
    /// every inserted vert (rim midpoint and interior/lattice node) is snapped
    /// radially onto the origin-centred sphere whose radius is the `smooth`
    /// parameter: `co_sphere = normalize(co_chord) * smooth`. Pre-existing
    /// corners are never moved (§9.4.2). This is **mutually exclusive** with the
    /// §9.1 off-chord offset and wins — the §9.1 machinery is skipped and
    /// `smooth_falloff` / `use_smooth_even` become inert (§9.4.3); `smooth` is
    /// reused as the sphere radius. Feed a coarse cage whose corners already sit
    /// on the sphere of radius `smooth` and every cut bulges outward onto it.
    #[must_use]
    pub fn use_sphere(mut self, enabled: bool) -> Self {
        self.use_sphere = enabled;
        self
    }

    /// Set the fractal displacement magnitude (§9.2). Defaults to `0.0` (the
    /// no-op core: every inserted vert keeps its §9.1 / §9.4 / chord position).
    /// A non-zero value pushes each inserted vert by a deterministic
    /// turbulence-noise displacement scaled by this value and the cut edge's
    /// length (§9.2.3), applied last (§9.2.6). The jitter is reproducible for a
    /// fixed [`Self::seed`].
    #[must_use]
    pub fn fractal(mut self, fractal: f32) -> Self {
        self.fractal = fractal;
        self
    }

    /// Bias the fractal jitter onto the surface normal (§9.2.4). Defaults to
    /// `0.0` (free 3-D displacement). Scales **only** the two tangential
    /// components by `(1 - along_normal)`; at `1.0` the vert moves only along
    /// the averaged endpoint normal. No effect when `fractal == 0`.
    #[must_use]
    pub fn along_normal(mut self, along_normal: f32) -> Self {
        self.along_normal = along_normal;
        self
    }

    /// Set the fractal RNG seed (§9.2.1). Defaults to `0`. Consumed exactly
    /// once to derive the fixed per-operator offset, so the jitter is fully
    /// reproducible for a fixed seed. No effect when `fractal == 0`.
    #[must_use]
    pub fn seed(mut self, seed: i32) -> Self {
        self.seed = seed;
        self
    }
}

/// Outcome of [`LoopMesh::subdivide_edges`]. Distinguishes a real run from a
/// declined-because-unsupported configuration, so a caller can detect a
/// deferred-parameter request without inspecting the mesh.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SubdivideResult {
    /// The operator ran. Carries the new geometry the cut created.
    Ran(SubdivideGeom),
    /// The parameter tuple is outside the implemented surface (a §9 deferred
    /// parameter — smooth/fractal/sphere at a non-default value). The mesh was
    /// left untouched.
    Unsupported,
}

/// The geometry [`LoopMesh::subdivide_edges`] created, gathered as sets
/// ([`SUBDIVIDE_EDGES.md`](../../../../radial-mesh-guides/operators/SUBDIVIDE_EDGES.md)
/// §1). Enumeration order is not observable for parity; treat every field as a
/// set (per [`MESH_COMPARISON.md`](../../../../radial-mesh-guides/MESH_COMPARISON.md)).
///
/// The operator framework emits three output buffers (§1). They are recoverable
/// here as three views over the captured geometry:
///
/// - [`Self::geom_split`] (`geom_split.out`) — the new midpoint verts and the
///   edge-halves produced by splitting the input edges.
/// - [`Self::geom_inner`] (`geom_inner.out`) — the inner population (§1): **every
///   inserted midpoint vert** (all rim midpoints are inner), plus the interior
///   geometry of the *grid-style* fills only — the §5.4 quad plus-split, the
///   §6.3 tri 1-to-4, the §9.5 lattices, and the §5.2.2 INNERVERT corner vert
///   `c` — marking their interior verts, interior edges, and fragment faces
///   inner. The corner-cut / fan / path patterns (§5.1, §5.2 straight/path/fan,
///   §5.3, §6.1, §6.2) contribute **only** their rim midpoints, never their
///   connecting edges or fragment faces.
/// - [`Self::geom`] (`geom.out`) — the union of the split, inner, and per-face
///   re-split marks (§1). A re-split face appears here as the **single**
///   re-split-marked original face, *not* as all its fragments; a grid fill's
///   fragments enter only through the inner mark. So `geom.out`'s face count is
///   the inner fragment count plus one re-split mark per re-split face — far
///   fewer than the raw fragment total for the non-grid patterns.
///
/// The [`Self::verts`] / [`Self::faces`] fields are the `geom.out` vert and
/// face sets; [`Self::edges`] completes that union with the `geom.out` edge set.
#[derive(Default, Clone, Debug, PartialEq, Eq)]
pub struct SubdivideGeom {
    /// `geom.out` verts (§1): split ∪ inner — every midpoint vert inserted on a
    /// cut edge (§3) plus any interior vert a grid fill added (the §5.4 quad
    /// centre, the §5.2.2 INNERVERT corner, the §9.5 lattice interior).
    pub verts: Vec<VertexKey>,
    /// `geom.out` edges (§1): split ∪ inner — the edge-halves of the split input
    /// edges (§3) and the interior edges laid by the *grid* fills (§5.4 / §6.3 /
    /// §9.5) and the two-edge fallback. The connecting edges of the non-grid
    /// patterns are in *no* mark-set and never appear here.
    pub edges: Vec<EdgeKey>,
    /// `geom.out` faces (§1): one re-split mark per re-split input face plus the
    /// inner fragment faces of the grid fills. **Not** every fragment — a
    /// non-grid re-split face contributes a single face (the re-split-marked
    /// original), not all its fragments.
    pub faces: Vec<FaceKey>,
    /// `geom_split.out` verts: the new midpoint verts produced by splitting the
    /// input edges (§3). A subset of [`Self::verts`].
    pub split_verts: Vec<VertexKey>,
    /// `geom_split.out` edges: the edge-halves produced by splitting the input
    /// edges (§3). A subset of [`Self::edges`].
    pub split_edges: Vec<EdgeKey>,
    /// `geom_inner.out` verts (§1): **every** inserted midpoint vert (all rim
    /// midpoints are inner) plus the interior vert(s) a grid fill added (§5.4
    /// quad centre, §5.2.2 INNERVERT corner, §9.5 lattice interior). A subset of
    /// [`Self::verts`].
    pub inner_verts: Vec<VertexKey>,
    /// `geom_inner.out` edges (§1): the interior edges a *grid* fill (§5.4 / §6.3
    /// / §9.5) or the two-edge fallback laid down inside a face. The non-grid
    /// patterns' connecting edges are *not* inner. A subset of [`Self::edges`].
    pub inner_edges: Vec<EdgeKey>,
    /// `geom_inner.out` faces (§1): the new fragment faces of a *grid* fill
    /// (§5.4 / §6.3 / §9.5), minus the one fragment that keeps the original
    /// face's identity (the re-split mark). The non-grid patterns mark no
    /// fragment inner. A subset of [`Self::faces`].
    pub inner_faces: Vec<FaceKey>,
}

/// A read-only view over one of [`SubdivideGeom`]'s three output slots (§1).
/// Carries the slot's vert / edge / face sets by reference; enumeration order is
/// not observable, so callers treat each as a set.
#[derive(Copy, Clone, Debug)]
pub struct SubdivideSlot<'a> {
    pub verts: &'a [VertexKey],
    pub edges: &'a [EdgeKey],
    pub faces: &'a [FaceKey],
}

impl SubdivideGeom {
    /// `geom_split.out` (§1): the midpoint verts and edge-halves from splitting
    /// the input edges. No faces (the split itself creates none).
    pub fn geom_split(&self) -> SubdivideSlot<'_> {
        SubdivideSlot {
            verts: &self.split_verts,
            edges: &self.split_edges,
            faces: &[],
        }
    }

    /// `geom_inner.out` (§1): every inserted midpoint vert plus the interior
    /// verts/edges/faces of the grid-style fills (and the §5.2.2 INNERVERT
    /// corner vert).
    pub fn geom_inner(&self) -> SubdivideSlot<'_> {
        SubdivideSlot {
            verts: &self.inner_verts,
            edges: &self.inner_edges,
            faces: &self.inner_faces,
        }
    }

    /// `geom.out` (§1): the headline union — every vert, edge, and face the
    /// operator created or replaced.
    pub fn geom(&self) -> SubdivideSlot<'_> {
        SubdivideSlot {
            verts: &self.verts,
            edges: &self.edges,
            faces: &self.faces,
        }
    }

    /// Record a midpoint vert and the two edge-halves of the just-split input
    /// edge into `geom.out`, `geom_split.out`, and `geom_inner.out` (§3).
    /// `original_half` is the input-edge handle that survives the split (the
    /// near-side half), `new_half` the freshly-created far-side half. Every rim
    /// midpoint is inner (§1), so the vert lands in `geom_inner.out` too.
    fn record_split(&mut self, midpoint: VertexKey, original_half: EdgeKey, new_half: EdgeKey) {
        self.verts.push(midpoint);
        self.split_verts.push(midpoint);
        self.inner_verts.push(midpoint);
        self.push_split_edge(original_half);
        self.push_split_edge(new_half);
    }

    /// Record a rim midpoint vert into `geom.out`, `geom_split.out`, and
    /// `geom_inner.out` (§3). The multi-cut split path records its segment edges
    /// separately; this captures only the vert (which is split *and* inner, §1).
    fn record_split_vert(&mut self, midpoint: VertexKey) {
        self.verts.push(midpoint);
        self.split_verts.push(midpoint);
        self.inner_verts.push(midpoint);
    }

    fn push_split_edge(&mut self, edge: EdgeKey) {
        if !self.split_edges.contains(&edge) {
            self.split_edges.push(edge);
        }
        if !self.edges.contains(&edge) {
            self.edges.push(edge);
        }
    }

    /// Record an inner edge a per-face re-split pattern laid down into both
    /// `geom.out` and `geom_inner.out` (§5 / §6 / §9.5).
    fn record_inner_edge(&mut self, edge: EdgeKey) {
        if !self.inner_edges.contains(&edge) {
            self.inner_edges.push(edge);
        }
        if !self.edges.contains(&edge) {
            self.edges.push(edge);
        }
    }

    /// Record an inner vert a per-face pattern added into both `geom.out` and
    /// `geom_inner.out` (§5.4 / §5.2.2 / §9.5).
    fn record_inner_vert(&mut self, vert: VertexKey) {
        self.verts.push(vert);
        self.inner_verts.push(vert);
    }

    /// Record the single re-split mark for one re-split input face into
    /// `geom.out` (§1). A re-split face contributes exactly one face to
    /// `geom.out` — the original handle the dispatcher flagged — regardless of
    /// how many fragments the surgery produced. Non-grid patterns call this for
    /// their original face and record no fragment faces at all.
    fn record_resplit_face(&mut self, face: FaceKey) {
        if !self.faces.contains(&face) {
            self.faces.push(face);
        }
    }

    /// Re-point a re-split mark from a face the surgery killed onto the fragment
    /// that stands in for it (§9.5's lattice fills rebuild their face rather than
    /// splitting it). The mark itself is not revoked — only its carrier moves.
    fn replace_resplit_face(&mut self, dead: FaceKey, standin: FaceKey) {
        // Only a *recorded* face reaches a lattice fill, and the dispatch scan
        // marks every face it records (§4.4), so the mark is always there to
        // move. Its absence would mean the record / mark invariant had broken —
        // an internal-invariant guard, not a reachable state to paper over.
        match self.faces.iter_mut().find(|f| **f == dead) {
            Some(slot) => *slot = standin,
            None => debug_assert!(false, "a re-split face carries its geom.out mark"),
        }
    }

    /// Record a grid-fill fragment face into both `geom.out` and
    /// `geom_inner.out` (§5.4 / §6.3 / §9.5). Only the grid-style fills mark
    /// their fragments inner; one fragment per fill keeps the original face's
    /// identity and is recorded via [`Self::record_resplit_face`] instead.
    fn record_inner_face(&mut self, face: FaceKey) {
        if !self.inner_faces.contains(&face) {
            self.inner_faces.push(face);
        }
        if !self.faces.contains(&face) {
            self.faces.push(face);
        }
    }
}

/// §4.4 — one entry of the dispatch scan's record, taken on the face's
/// **pre-cut** ring. The pattern itself is re-matched against the live ring at
/// apply time (which is equivalent for a face that has not been carved, and for
/// one that has, the abandonment test below fires first); what pass one has to
/// carry forward is the ring shape the pattern was matched *for*.
#[derive(Copy, Clone, Debug)]
struct FaceRecord {
    face: FaceKey,
    /// Corner count before any edge was cut (the template's own size for a
    /// template face; any `n >= 3` for a fallback face).
    pre_len: usize,
    /// How many of the face's ring edges are input edges — the **true** count.
    /// The flat-corner guard gates the record *decision*, never this value:
    /// feeding a guard-zeroed count to `expected_len` would abandon a
    /// flat-cornered face's perfectly good template (§4.4).
    cut_count: usize,
    /// §7.0.3 — the anchor corner: the pre-cut corner immediately preceding the
    /// face's first cut edge in loop order. An original rim corner, so the §3
    /// splits leave it intact, and it is on the ring iff the ring is still the
    /// one the pattern was matched against.
    anchor: VertexKey,
    /// §7.1 — the fallback's dropped-connect verdict: do this face's two cut
    /// edges have a **second common face**? Decided here, on the *input*
    /// topology, and uniform across the face's `N` connects by construction
    /// (they all sit on the same two cut edges). Always `false` when the face is
    /// not a two-cut face, which is the only kind that can read it.
    strip_connects_dropped: bool,
}

impl FaceRecord {
    /// §7.0.3 — the ring the apply pass expects to find. Each of the face's cut
    /// edges contributed `cuts` fresh rim corners in pass two.
    fn expected_len(&self, cuts: u32) -> usize {
        self.pre_len + cuts as usize * self.cut_count
    }
}

/// §7.0.3 — what the abandonment gate needs to know about a re-derived rim,
/// read in one walk without decomposing it into the §9.5 lattice's sides.
struct RimCensus {
    corner_count: usize,
    run_count: usize,
    /// Every cut-vert run is exactly `cuts` long.
    runs_uniform: bool,
    has_anchor: bool,
}

/// §7.1 — a two-cut face's rim, read as the strip sees it: the two cut-vert runs
/// in loop order, and whether the sides carrying them share a corner.
struct TwoCutRuns {
    run_a: SmallVec<[VertexKey; 4]>,
    run_b: SmallVec<[VertexKey; 4]>,
    adjacent: bool,
}

impl LoopMesh {
    /// Edge-driven, face-aware refinement
    /// ([`SUBDIVIDE_EDGES.md`](../../../../radial-mesh-guides/operators/SUBDIVIDE_EDGES.md)).
    ///
    /// Splits every edge in `edges` at its midpoint (§3), then re-splits each
    /// face touched by a cut according to how many of its edges were cut (§4):
    /// the quad patterns of §5 and the triangle patterns of §6, in the
    /// standard-subdivide configuration (`use_single_edge` + `use_grid_fill`
    /// on, `STRAIGHT_CUT`). A shared interior edge is cut once and its midpoint
    /// shared by both faces, so adjacent re-splits stay crack-free (§2).
    ///
    /// Only the core single-cut configuration is implemented; a deferred
    /// parameter (e.g. `cuts != 1`) makes the operator decline with
    /// [`SubdivideResult::Unsupported`] without touching the mesh (§9).
    pub fn subdivide_edges(
        &mut self,
        edges: &[EdgeKey],
        params: SubdivideParams,
    ) -> SubdivideResult {
        if params.cuts == 0 {
            return SubdivideResult::Ran(SubdivideGeom::default());
        }
        if params.cuts >= 2 {
            return self.subdivide_edges_multicut(edges, params);
        }

        let mut geom = SubdivideGeom::default();

        // §9.1.1 / §9.2.3 — both the off-chord offset and the fractal basis read
        // the cut edge's endpoint vertex normals, which must be current before
        // placement. Freshen all cached normals once up front; the chord-only
        // core (`smooth == 0 && fractal == 0`) skips this and stays bit-identical
        // to the no-displacement path.
        if params.smooth != 0.0 || params.fractal != 0.0 {
            self.mesh_normals_update();
        }

        // §9.2.1 — the seed is consumed exactly once, up front, into a fixed
        // per-operator offset shared by every inserted vert. `None` when fractal
        // is off, leaving the core untouched.
        let fractal_offset = self.fractal_offset(&params);

        // §4.3.1 — the flat-corner guard reads the *input* edge directions, so it
        // must be decided before the cuts rewrite them. Its radial walk doubles as
        // the enumeration of the faces a cut edge touches, which pass one consumes,
        // and its cut-edge layer is exactly the one pass one keys on (§4.1).
        let (guard_layer, cut_layer, touched) = self.push_flat_corner_guard(edges);

        // ---- Pass one: the dispatch scan (§4.4) ----
        // Match and record every face *before* any edge is cut, in mesh face
        // order, stamping the §1 re-split mark as we go.
        let records = self.dispatch_scan(cut_layer, touched, guard_layer, &params, &mut geom);
        self.pop_edge_op_flag_layer(cut_layer);

        // A vert op-flag layer tags the verts the operator mints: the §3 pass-two
        // midpoints (`MIDPOINT`) and the pass-three pattern interiors
        // (`PASS_THREE_VERT`). Pass three reads both — the first to tell original
        // corners from inserted midpoints on a face's ring (§4.1), the second to
        // recognise a ring an earlier pattern re-inflated (§7.0.3).
        let midpoint_layer = self.push_vert_op_flag_layer();

        // §9.2.3 deferred write — the fractal jitter does not mutate `v.co` in
        // place; it stages displaced positions (computed from each vert's
        // pristine position) here, copied back only after every cut edge has been
        // processed. This is what makes a shared corner a last-writer overwrite,
        // not an accumulation (§9.2.3.3). Empty/unused when fractal is off.
        let mut fractal_scratch: std::collections::HashMap<VertexKey, Vec3> =
            std::collections::HashMap::new();

        // ---- Pass two: per-edge midpoint insertion (§3) ----
        // Each input edge is split once at t = 0.5. The edge_split wrapper does
        // the position lerp and the per-vert / per-edge / per-loop linear
        // interpolation (§7.2 along-edge blend) for us.
        for &edge in edges {
            let [near, _far] = self.edge_vert_keys(edge);
            // §9.1 — snapshot the endpoint positions and vertex normals *before*
            // the split (the split rewrites the edge's far endpoint). `t = 0.5`
            // is the §3.2 directional cut parameter, measured from `near`. The
            // §9.1 offset is skipped under `use_sphere` (§9.4.3, sphere wins).
            let placement = (!params.use_sphere && params.smooth != 0.0)
                .then(|| self.smooth_placement(edge, near, 0.5, &params));
            // §9.2.3 — snapshot the cut edge's fractal basis (endpoint positions,
            // length, and angle-weighted endpoint normals) before the split
            // rewrites the far endpoint.
            let fractal_ctx = fractal_offset.map(|_| self.fractal_edge_context(edge, near));
            let (m, e_new) = self.edge_split(edge, near, 0.5);
            if let Some(co_final) = placement {
                self.set_vertex_position(m, co_final);
                self.vert_normal_update(m);
            }
            // §9.4.1 — radial snap onto the origin-centred sphere of radius
            // `smooth`, taken from the inserted midpoint's chord position.
            self.maybe_sphere_project(m, &params);
            // §9.2.3 / §9.2.6 — fractal is the last placement step. It stages the
            // displaced midpoint *and* the cut edge's two endpoints (§9.2.3.3)
            // into the deferred scratch store rather than writing in place.
            if let (Some(offset), Some(ctx)) = (fractal_offset, &fractal_ctx) {
                self.stage_fractal(m, offset, ctx, &params, &mut fractal_scratch);
            }
            self.vert_op_flag_set(midpoint_layer, m, MIDPOINT);
            // §1 — the midpoint and the two edge-halves (`edge` survives as the
            // near-side half, `e_new` is the far-side half) populate geom_split.
            geom.record_split(m, edge, e_new);
        }

        // §9.2.3 — copy the staged displacements back into the mesh now that
        // every cut edge has been processed (the deferred write).
        for (v, co) in &fractal_scratch {
            self.set_vertex_position(*v, *co);
        }

        // ---- Pass three: apply the recorded patterns (§4.4) ----
        // Last-recorded-first. One face's connect can carve another (§7.0.1), so
        // the order is observable in the output mesh, not just in slot ordering.
        for record in records.iter().rev() {
            self.resplit_face(record, midpoint_layer, guard_layer, &params, &mut geom);
        }

        self.pop_face_op_flag_layer(guard_layer);
        self.pop_vert_op_flag_layer(midpoint_layer);
        SubdivideResult::Ran(geom)
    }

    /// Read the §9.1 smooth-placement context for the cut edge `e = {near, far}`
    /// — the two endpoint positions and **angle-weighted** vertex normals
    /// (§9.1.5). `near` is the edge's first endpoint `v_a`, so the directional
    /// cut parameter `t` (§3.2 / §3.3) is measured from it.
    ///
    /// The endpoint normals are computed here rather than read from the cached
    /// vertex-normal field because the cached normals are area-weighted, whereas
    /// the §9.1.2 arc demands the angle-weighted (Thürmer–Wüthrich) convention
    /// (§9.1.5). Face normals must be current (the caller freshens them via
    /// [`Self::mesh_normals_update`] before the cut).
    fn smooth_endpoint_context(&self, edge: EdgeKey, near: VertexKey) -> SmoothEndpointContext {
        let far = self
            .edges
            .get(edge)
            .other_vertex(near)
            .expect("near_vertex must belong to edge");
        SmoothEndpointContext {
            co_a: self.vertex_position(near),
            co_b: self.vertex_position(far),
            no_a: self.angle_weighted_vertex_normal(near),
            no_b: self.angle_weighted_vertex_normal(far),
        }
    }

    /// §9.1.5 — the angle-weighted (Thürmer–Wüthrich) vertex normal: each
    /// incident face contributes its **unit** face normal scaled by the interior
    /// corner angle the vertex subtends in that face, summed and renormalised.
    /// This is the convention the §9.1.2 smooth arc reads; it is *not* the
    /// area-weighted [`Self::vertex_normal`] cache. Assumes face normals are
    /// current. Falls back to `Vec3::ZERO` for a vertex with no incident faces.
    fn angle_weighted_vertex_normal(&self, vertex: VertexKey) -> Vec3 {
        let vertex_co = self.vertex_position(vertex);
        let mut accumulated = Vec3::ZERO;

        let mut disk = self.disk_edges(vertex);
        while let Some(edge) = disk.next(self) {
            let mut radial = self.radial_loops(edge);
            while let Some(corner) = radial.next(self) {
                // Only count the corner whose vertex is `vertex` (each incident
                // face has exactly one such loop). The radial walk visits both
                // sides of every incident edge, so guard on the corner vertex.
                if self.loops.get(corner).vertex != vertex {
                    continue;
                }
                let prev_co = self.vertex_position(self.loops.get(self.loop_prev(corner)).vertex);
                let next_co = self.vertex_position(self.loops.get(self.loop_next(corner)).vertex);
                let to_prev = (prev_co - vertex_co).normalize_or_zero();
                let to_next = (next_co - vertex_co).normalize_or_zero();
                let angle = to_prev.dot(to_next).clamp(-1.0, 1.0).acos();
                let face_normal = self.face_normal(self.loops.get(corner).face);
                accumulated += face_normal * angle;
            }
        }

        if accumulated.length_squared() > 0.0 {
            accumulated.normalize()
        } else {
            Vec3::ZERO
        }
    }

    /// The §9.1 off-chord position for a single midpoint cut at parameter `t`
    /// on edge `e = {near, far}`. Convenience wrapper over
    /// [`Self::smooth_endpoint_context`] + [`SmoothEndpointContext::placement`].
    fn smooth_placement(
        &self,
        edge: EdgeKey,
        near: VertexKey,
        t: f32,
        params: &SubdivideParams,
    ) -> Vec3 {
        self.smooth_endpoint_context(edge, near)
            .placement(t, params)
    }

    /// §9.4.1 — when `use_sphere` is set, snap the inserted vert `v` radially
    /// onto the origin-centred sphere of radius `smooth`, reading the direction
    /// from `v`'s *current* (chord / centroid / lattice) position. A no-op when
    /// the flag is clear, so the chord and §9.1-smooth paths are untouched. Only
    /// the verts the operator *creates* are ever passed here; pre-existing
    /// corners are never projected (§9.4.2).
    fn maybe_sphere_project(&mut self, v: VertexKey, params: &SubdivideParams) {
        if params.use_sphere {
            let co_sphere = sphere_project(self.vertex_position(v), params.smooth);
            self.set_vertex_position(v, co_sphere);
        }
    }

    /// §9.2.1 — derive the fixed per-operator fractal offset from the seed, by
    /// drawing three floats from the hashed RNG and scaling each by `200.0`. The
    /// RNG is consumed *exactly* here and nowhere else, so the whole pass shares
    /// one offset and a fixed seed is reproducible. `None` when `fractal == 0`,
    /// so the core never advances the RNG.
    fn fractal_offset(&self, params: &SubdivideParams) -> Option<Vec3> {
        if params.fractal == 0.0 {
            return None;
        }
        let mut rng = fractal_noise::SeededRng::new_hashed(params.seed as u32);
        Some(Vec3::new(
            rng.next_float() * 200.0,
            rng.next_float() * 200.0,
            rng.next_float() * 200.0,
        ))
    }

    /// §9.2.3 — snapshot the cut edge's fractal basis before the split: the
    /// endpoint positions, the edge length (the displacement scales with it),
    /// and the **angle-weighted** endpoint vertex normals (the §9.1.5 convention,
    /// shared with the smooth path; assumes face normals are current). `near` is
    /// the edge's first endpoint `v_a`.
    fn fractal_edge_context(&self, edge: EdgeKey, near: VertexKey) -> FractalEdgeContext {
        let far = self
            .edges
            .get(edge)
            .other_vertex(near)
            .expect("near_vertex must belong to edge");
        let co_a = self.vertex_position(near);
        let co_b = self.vertex_position(far);
        FractalEdgeContext {
            edge_len: (co_a - co_b).length(),
            no_a: self.angle_weighted_vertex_normal(near),
            no_b: self.angle_weighted_vertex_normal(far),
            va: near,
            co_a,
            vb: far,
            co_b,
        }
    }

    /// §9.2.3 / §9.2.3.3 — compute the fractal displacement for every vert of one
    /// cut edge (the inserted midpoint `m` plus the two pre-existing endpoints)
    /// and stage the results into the deferred scratch store. Each vert's noise
    /// sample reads its **pristine** position — for `m` the freshly-placed
    /// (chord / smooth / sphere) position, for an endpoint its captured input
    /// position — so a corner shared by several cut edges is recomputed from its
    /// unchanged input and **overwritten** by each incident edge (last-writer
    /// wins, §9.2.3.3); the scratch is copied back to the mesh only after every
    /// cut edge is processed.
    fn stage_fractal(
        &self,
        m: VertexKey,
        offset: Vec3,
        ctx: &FractalEdgeContext,
        params: &SubdivideParams,
        scratch: &mut std::collections::HashMap<VertexKey, Vec3>,
    ) {
        // The inserted midpoint reads its own freshly-placed position; the two
        // endpoints read their pristine input positions captured in `ctx`.
        scratch.insert(
            m,
            fractal_displace(self.vertex_position(m), offset, ctx, params),
        );
        scratch.insert(ctx.va, fractal_displace(ctx.co_a, offset, ctx, params));
        scratch.insert(ctx.vb, fractal_displace(ctx.co_b, offset, ctx, params));
    }

    /// §4.3.1 — the flat-corner guard. Decide, for every face a cut edge
    /// touches, whether its generic two-edge fallback (§7.1) is suppressed:
    /// exactly two of the face's edges are in the input set, those two edges
    /// **share a vertex**, and they are **collinear** there. Such a face is
    /// treated as having no cut edges at all — it is not re-split, contributes
    /// no re-split mark to `geom.out`, and keeps its inserted midpoints as flat
    /// rim corners.
    ///
    /// The verdict is written into the `FLAT_CORNER` bit of a face op-flag
    /// layer, returned for the dispatch to read back and the caller to pop. It is
    /// taken on the *input* geometry, before any cut is inserted, which is what
    /// makes it independent of `cuts` (§9.5.3.5).
    ///
    /// The radial walk that enumerates the guard's candidates is the same walk
    /// the dispatch scan needs (§4.4 pass one: every face a cut edge touches), so
    /// it is done **once** and the deduped candidate set handed back with the
    /// layer — as is the cut-edge layer, which is the very membership test §4.1
    /// keys on. `FACE_SEEN` is the dedupe — a face is reachable from each of its
    /// cut edges — which keeps the enumeration linear in the touched faces rather
    /// than quadratic. The caller pops the cut layer once the scan has read it.
    fn push_flat_corner_guard(
        &mut self,
        edges: &[EdgeKey],
    ) -> (
        crate::op_flags::FaceOpFlagLayer,
        crate::op_flags::EdgeOpFlagLayer,
        Vec<FaceKey>,
    ) {
        let cut_layer = self.push_edge_op_flag_layer();
        for &edge in edges {
            self.edge_op_flag_set(cut_layer, edge, CUT_EDGE);
        }

        let guard_layer = self.push_face_op_flag_layer();
        let mut touched: Vec<FaceKey> = Vec::new();
        for &edge in edges {
            let mut radial = self.radial_loops(edge);
            while let Some(l) = radial.next(self) {
                let face = self.loops.get(l).face;
                if self.face_op_flag_test(guard_layer, face, FACE_SEEN) {
                    continue;
                }
                self.face_op_flag_set(guard_layer, face, FACE_SEEN);
                touched.push(face);
                if let Some([a, b]) = self.face_cut_edge_pair(face, cut_layer) {
                    if self.cut_edges_form_flat_corner(a, b) {
                        self.face_op_flag_set(guard_layer, face, FLAT_CORNER);
                    }
                }
            }
        }

        (guard_layer, cut_layer, touched)
    }

    /// The face's two cut edges (§4.1), or `None` unless it has **exactly** two.
    /// Only the two-cut faces can reach the fallback, so any other count leaves
    /// the guard with nothing to decide.
    fn face_cut_edge_pair(
        &self,
        face: FaceKey,
        cut_layer: crate::op_flags::EdgeOpFlagLayer,
    ) -> Option<[EdgeKey; 2]> {
        let mut cut_edges: SmallVec<[EdgeKey; 2]> = SmallVec::new();
        let mut it = self.face_loops(face);
        while let Some(l) = it.next(self) {
            let edge = self.loops.get(l).edge;
            if edge.is_null() {
                continue;
            }
            if self.edge_op_flag_test(cut_layer, edge, CUT_EDGE) {
                if cut_edges.len() == 2 {
                    return None;
                }
                cut_edges.push(edge);
            }
        }
        (cut_edges.len() == 2).then(|| [cut_edges[0], cut_edges[1]])
    }

    /// True when `a` and `b` share a vertex and are collinear there, to within
    /// the §4.3.1 tolerance. The cosine is taken **unsigned**, so this catches
    /// both the ~180° straight corner (the edges continue each other) and the
    /// ~0° spike (they fold back onto one another).
    fn cut_edges_form_flat_corner(&self, a: EdgeKey, b: EdgeKey) -> bool {
        let [a0, a1] = self.edge_vert_keys(a);
        let [b0, b1] = self.edge_vert_keys(b);
        let shares_vertex = a0 == b0 || a0 == b1 || a1 == b0 || a1 == b1;
        if !shares_vertex {
            return false;
        }
        // Orientation is irrelevant to an unsigned cosine, so the raw edge
        // directions serve without re-rooting them at the shared vertex.
        let u = self.vertex_position(a1) - self.vertex_position(a0);
        let w = self.vertex_position(b1) - self.vertex_position(b0);
        let (u_len, w_len) = (u.length(), w.length());
        if u_len == 0.0 || w_len == 0.0 {
            return false;
        }
        let cos = (u / u_len).dot(w / w_len);
        cos.abs() > 1.0 - FLAT_CORNER_COS_EPS
    }

    /// §4.4 pass one — the dispatch scan. Walks every touched face **in mesh
    /// face order**, *before* any edge is cut, decides whether the dispatcher
    /// records it, and stamps the §1 re-split mark on each face it does. The mark
    /// means "recorded", not "re-split": pass three may abandon the template
    /// (§7.0.3), drop every one of its connects (§7.0.2), or find no pattern at
    /// all (a one-cut pentagon, §4.3), and the mark stands regardless. The only
    /// thing that keeps it off a cut face is the flat-corner guard, and only on a
    /// face no template claimed.
    ///
    /// The *pattern gates* — `use_single_edge`, `use_grid_fill`, `use_only_quads`
    /// — are all deliberately absent here. They decide whether a matched pattern
    /// fires in pass three, never whether the dispatcher records the face: §4.4's
    /// record rule reads "a template matched it, or its guard-adjusted cut count
    /// is non-zero", and a gated-off face still satisfies it. A gate that
    /// suppressed the *record* would also suppress the §1 re-split mark, which is
    /// A/B-visible in `geom.out`'s slot cardinalities.
    fn dispatch_scan(
        &mut self,
        cut_layer: crate::op_flags::EdgeOpFlagLayer,
        mut touched: Vec<FaceKey>,
        guard_layer: crate::op_flags::FaceOpFlagLayer,
        params: &SubdivideParams,
        geom: &mut SubdivideGeom,
    ) -> Vec<FaceRecord> {
        // Mesh face order. The scan order is unobservable in itself — it matters
        // only through pass three's reversal (§4.4) — but the *reversal* is not,
        // so the order it reverses has to be the mesh's own.
        touched.sort_unstable_by_key(|f| f.as_usize());

        let mut records: Vec<FaceRecord> = Vec::with_capacity(touched.len());
        for face in touched {
            let mut cut_edges: SmallVec<[EdgeKey; 4]> = SmallVec::new();
            let mut pre_len = 0usize;
            // §7.0.3 — the anchor is the corner just *before* the first cut edge:
            // a loop's vertex is the tail of the loop's edge, so the loop that
            // carries the first cut edge carries the anchor too.
            let mut anchor: Option<VertexKey> = None;
            let mut it = self.face_loops(face);
            while let Some(l) = it.next(self) {
                pre_len += 1;
                let loop_ref = self.loops.get(l);
                let (vertex, edge) = (loop_ref.vertex, loop_ref.edge);
                if !edge.is_null() && self.edge_op_flag_test(cut_layer, edge, CUT_EDGE) {
                    anchor.get_or_insert(vertex);
                    cut_edges.push(edge);
                }
            }
            let Some(anchor) = anchor else {
                continue;
            };

            // §4.3.1 / §4.4 — the guard's zeroed cut count gates the *fallback*
            // record and nothing else. Template matching keys on the cut-edge
            // mask, which the guard never touches, so a guard-hit face that
            // matches a template is recorded, marked, and re-split as usual.
            let templated = self.template_matches(pre_len, &cut_edges, params);
            if !templated && self.face_op_flag_test(guard_layer, face, FLAT_CORNER) {
                continue;
            }

            // §7.1 — the dropped-connect precondition, decided here for the same
            // reason as the §4.3.1 guard beside it: it is a function of the
            // *input topology*, and only pass one still sees that topology. Both
            // of the face's cut edges are still whole edges now (from `cuts = 2`
            // up they are not, by apply time), and asking the live mesh at apply
            // time asks a different question — an earlier-applied face's
            // fragments carry cut verts too, so a face that shares both cut edges
            // can be re-split out of the answer before the fallback gets its turn.
            let strip_connects_dropped = cut_edges.len() == 2
                && self.cut_edges_have_a_second_common_face(face, cut_edges[0], cut_edges[1]);

            geom.record_resplit_face(face);
            records.push(FaceRecord {
                face,
                pre_len,
                cut_count: cut_edges.len(),
                anchor,
                strip_connects_dropped,
            });
        }

        records
    }

    /// §4.1 — does a dedicated §5 / §6 / §9.5 template claim this face? Keyed on
    /// the pre-cut ring's corner count and cut-edge **mask** (for the ambiguous
    /// quad-two-edge row, whether the two cut edges are adjacent, and which
    /// corner strategy is in force). The pattern gates (`use_single_edge`,
    /// `use_grid_fill`, `use_only_quads`) are deliberately not consulted: they
    /// decide whether a matched pattern *fires*, not whether a template claims
    /// the face — and hence not whether the dispatcher records it. The
    /// `quad_corner_type` strategy *is* consulted, because it does not gate a
    /// template, it chooses one (and `STRAIGHT_CUT` chooses none, dropping the
    /// row onto the fallback the guard can reach).
    fn template_matches(
        &self,
        pre_len: usize,
        cut_edges: &[EdgeKey],
        params: &SubdivideParams,
    ) -> bool {
        match (pre_len, cut_edges.len()) {
            (3, 1) | (3, 3) | (4, 1) | (4, 3) | (4, 4) => true,
            (4, 2) => {
                params.quad_corner_type != QuadCornerType::StraightCut
                    && self.edges_share_vertex(cut_edges[0], cut_edges[1])
            }
            _ => false,
        }
    }

    fn edges_share_vertex(&self, a: EdgeKey, b: EdgeKey) -> bool {
        let [a0, a1] = self.edge_vert_keys(a);
        let [b0, b1] = self.edge_vert_keys(b);
        a0 == b0 || a0 == b1 || a1 == b0 || a1 == b1
    }

    /// Dispatch and apply the §5 / §6 re-split pattern for one recorded face.
    fn resplit_face(
        &mut self,
        record: &FaceRecord,
        layer: crate::op_flags::VertOpFlagLayer,
        guard_layer: crate::op_flags::FaceOpFlagLayer,
        params: &SubdivideParams,
        geom: &mut SubdivideGeom,
    ) {
        let face = record.face;

        // §7.0.3 — abandonment, first half. An earlier-applied template's connect
        // may have selected this face as its carrier and carved it down; the ring
        // the pattern was matched for is then gone, and the template lays
        // *nothing* rather than indexing a ring it does not describe.
        if self.face_len(face) != record.expected_len(1) {
            return;
        }

        // §7.0.3 — the pair list is re-derived here, from the ring that exists at
        // *this* moment, never snapshotted once the cuts are in. Walk the loop
        // cycle once, recording each corner's vert and whether it is a
        // freshly-inserted midpoint, in cyclic order (§4.1).
        let mut ring: SmallVec<[(VertexKey, bool); 12]> = SmallVec::new();
        let mut it = self.face_loops(face);
        let mut has_anchor = false;
        let mut re_inflated = false;
        while let Some(l) = it.next(self) {
            let v = self.loops.get(l).vertex;
            let is_mid = self.vert_op_flag_test(layer, v, MIDPOINT);
            has_anchor |= v == record.anchor;
            re_inflated |= self.vert_op_flag_test(layer, v, PASS_THREE_VERT);
            ring.push((v, is_mid));
        }

        let cut_count = ring.iter().filter(|(_, m)| *m).count();
        let corner_count = ring.len() - cut_count; // original (non-midpoint) corners

        // §7.0.3 — abandonment, second half, and it is *not* redundant. The
        // length test alone would be, if corner counts only ever fell: a carved
        // face's fragments are both strictly smaller than their parent. But two
        // templates (§5.2.2's inner vert and §5.4's grid centre) split an edge in
        // pass **three**, on the chord their own first connect just laid — and
        // that chord's carrier can be a *pending* face, which is thereby carved
        // and then handed a corner back by the very vert the split mints. Such a
        // face lands exactly back on the length it was recorded with, carrying a
        // ring the pattern was never matched for.
        //
        // `re_inflated` is what closes that channel, and it is the only clause
        // that closes it *soundly*: the census counts only pass-two midpoints, so
        // the re-inflating vert reads as an original corner and the census passes;
        // the anchor clause depends on *which* corner the carve happened to peel,
        // which is an artefact of the victim's stored ring start (§4.2 says the
        // dispatcher must not depend on that), so the same configuration wound the
        // other way slips through it. The pass-three tag asks the question
        // directly — "is a vert no pass-two cut minted sitting on my ring?" — and
        // is winding-independent. All four clauses are cheap; the ring walk above
        // is happening anyway.
        if cut_count != record.cut_count
            || corner_count != record.pre_len
            || !has_anchor
            || re_inflated
        {
            return;
        }

        // §9.3 — `use_only_quads` gates the *pattern*, not the record: the face
        // keeps the re-split mark the dispatch scan stamped on it (§4.4) and
        // simply keeps its midpoints as flat rim corners.
        if params.use_only_quads && corner_count != 4 {
            return;
        }

        match (corner_count, cut_count) {
            (3, 1) => {
                if params.use_single_edge {
                    self.tri_one_cut(&ring);
                }
            }
            (3, 3) => {
                if params.use_grid_fill {
                    self.tri_three_cut(&ring, geom);
                }
            }
            (4, 1) => {
                if params.use_single_edge {
                    self.quad_one_cut(&ring);
                }
            }
            (4, 3) => {
                // §5.3 — quad, three edges cut. The two corner triangles plus a
                // pentagon-diagonal cut; no gating flag (neither one-cut nor
                // all-cut).
                self.quad_three_cut(&ring);
            }
            (4, 4) => {
                if params.use_grid_fill {
                    self.quad_four_cut(face, &ring, layer, params, geom);
                }
            }
            // §5.2 adjacent case, non-default strategy — the only two-cut row a
            // *template* claims. The strategy chooses the interior geometry laid
            // down once the two rim midpoints exist; all three share the §5.2.1
            // shared-corner anchor and far corner `v3`, and none of them is
            // guarded (§4.3.1 bites the fallback only). `STRAIGHT_CUT` is the
            // *absence* of a corner template, so it drops through to the strip.
            (4, 2)
                if params.quad_corner_type != QuadCornerType::StraightCut
                    && Self::cut_runs_are_adjacent(&ring) =>
            {
                match params.quad_corner_type {
                    QuadCornerType::InnerVert => {
                        self.quad_two_adjacent_inner_vert(&ring, layer, params, geom)
                    }
                    QuadCornerType::Path => self.quad_two_adjacent_path(&ring),
                    QuadCornerType::Fan => self.quad_two_adjacent_fan(&ring),
                    QuadCornerType::StraightCut => unreachable!("guarded by the match arm"),
                }
            }
            // §4.3 / §7.1 — every other two-cut face, whatever its size, is the
            // generic strip's: the two-opposite-edge quad, the `STRAIGHT_CUT`
            // two-adjacent quad, the two-edge triangle, the two-edge n-gon.
            //
            // §4.3.1 is enforced upstream, in the dispatch scan, which refuses to
            // *record* a flat-cornered face no template claimed — so a guarded
            // face never reaches pass three at all (and carries no re-split mark).
            (_, 2) => {
                debug_assert!(
                    !self.face_op_flag_test(guard_layer, face, FLAT_CORNER),
                    "a guarded template-less face is never recorded (§4.3.1)"
                );
                self.two_edge_strip(face, &ring, record.strip_connects_dropped, geom);
            }
            // §4.3 — the fallback is *not* a catch-all. A template-less face with
            // any other cut count (a one-cut or three-cut n-gon, say) is never
            // re-split: it keeps its rim cut verts as flat corners and stays one
            // larger n-gon.
            _ => {}
        }
    }

    /// Rotate `ring` so the first entry is a midpoint corner, returning the
    /// rotated vertex sequence and the per-position midpoint mask. The §4.2
    /// dispatch is rotation-agnostic; this normalises the start corner so each
    /// pattern can be phrased against a fixed layout.
    fn rotate_to_first_midpoint(ring: &[(VertexKey, bool)]) -> (SmallVec<[VertexKey; 12]>, usize) {
        let start = ring.iter().position(|(_, m)| *m).expect("at least one cut");
        let n = ring.len();
        let mut verts: SmallVec<[VertexKey; 12]> = SmallVec::new();
        for i in 0..n {
            verts.push(ring[(start + i) % n].0);
        }
        (verts, start)
    }

    /// §7.0 — the face a template connect targets: the one with the **fewest
    /// corners** among all faces carrying both endpoints in their loop cycle.
    ///
    /// Within one face's re-split this is exact — as the face is carved up, the
    /// fragment still holding both endpoints is always the smallest that does —
    /// which is what makes each pattern's pair list order-insensitive. Across
    /// faces it is a heuristic: two distinct faces can share two edges and both
    /// carry the pair (manifold-legal, §7.1), and then the smallest simply wins,
    /// with no appeal to which face the template was dispatched for.
    ///
    /// The scan is over the faces around the **first-named** endpoint, in the
    /// vertex face-corner fan order (RADIAL_EDGE_STRUCTURE.md §6.5), and the
    /// comparison is **strict**, so a corner-count tie goes to whichever tied face
    /// the fan yields first. Both halves of that are load-bearing, and the fan
    /// order is not a free choice: it is corner-edge-major / radial-minor, it
    /// yields each incident face exactly once, and its group sequence starts at
    /// the *first face's* corner edge rather than at the disk-seed edge. A
    /// disk-major walk over every radial loop is a different order — it can
    /// surface a face at the edge it *arrives* on, a position the fan never yields
    /// it at — and the two tie choices produce different meshes (§7.0.1, §7.0.3).
    /// The readings coincide only at a valence-2 hub (a cut midpoint of a manifold
    /// interior edge), so nothing but a valence-≥3 first-named endpoint separates
    /// them — and §5.2.2 / §5.2.3 / §5.4 all name one.
    ///
    /// The scan is also deliberately blind to whether the winner can actually be
    /// split (§7.0.2 decides that, afterwards, and a failure there is terminal for
    /// the connect).
    ///
    /// `None` when no face carries both — such a connect is a silent no-op.
    fn smallest_face_with_both(&self, a: VertexKey, b: VertexKey) -> Option<FaceKey> {
        let mut best: Option<(FaceKey, usize)> = None;
        let mut fan = self.vert_fan_loops(a);
        while let Some(corner) = fan.next(self) {
            let candidate = self.loops.get(corner).face;
            let corners = self.face_len(candidate);
            // The O(1) size test comes first so a candidate that cannot beat the
            // incumbent never pays for the membership walk — and so a tie leaves
            // the incumbent standing (first minimum wins).
            if best.is_some_and(|(_, best_corners)| corners >= best_corners) {
                continue;
            }
            if self.face_has_vert(candidate, b) {
                best = Some((candidate, corners));
            }
        }
        best.map(|(face, _)| face)
    }

    fn face_has_vert(&self, face: FaceKey, v: VertexKey) -> bool {
        let mut it = self.face_loops(face);
        while let Some(l) = it.next(self) {
            if self.loops.get(l).vertex == v {
                return true;
            }
        }
        false
    }

    /// §7.0 — lay one template connect. Every pair in the §5 / §6 / §9.5 pattern
    /// lists is laid through here: it is resolved to a carrier by
    /// [`Self::smallest_face_with_both`] (§7.0.1) and split there.
    ///
    /// There are exactly **two** ways the connect evaporates, and both are silent
    /// no-ops — no edge, no fragment, no slot entry, no error (§7.0.2):
    ///
    /// - **No carrier.** No face at all holds both endpoints.
    /// - **Unsplittable carrier.** The fewest-corners carrier holds the endpoints
    ///   as adjacent corners, so the split would carve off a degenerate 2-gon —
    ///   the SFME primitive itself refuses (RADIAL_EDGE_STRUCTURE.md §8.3) and its
    ///   `None` is this no-op. There is **no retry** on the next-smallest carrier — selection is
    ///   adjacency-blind by design (§7.0.1), and a connect dies here even when a
    ///   larger carrier holds the pair on a perfectly good diagonal. Re-targeting
    ///   would lay an edge the operator does not lay.
    ///
    /// The §7.1 two-edge fallback deliberately does **not** use this primitive —
    /// it splits the parent face handle directly, and it substitutes its own,
    /// stricter precondition (two cut edges with a *second* common face drop all
    /// `N` of its connects) for the no-op rules above. That precondition is
    /// **not** checked here: a template connect targeted at a pair two faces both
    /// carry is resolved by the fewest-corners heuristic, which guesses where the
    /// fallback declines.
    fn template_connect(&mut self, a: VertexKey, b: VertexKey) -> Option<(FaceKey, EdgeKey)> {
        let target = self.smallest_face_with_both(a, b)?;
        self.face_split(target, a, b)
    }

    /// §6.1 — triangle, one edge cut. Cycle is `v0, m, v1, v2`; connect the
    /// midpoint `m` to the opposite apex `v2`. 1 tri -> 2 tris.
    fn tri_one_cut(&mut self, ring: &[(VertexKey, bool)]) {
        let (v, _) = Self::rotate_to_first_midpoint(ring);
        // After rotation: v[0] = m, v[1] = v1, v[2] = v2, v[3] = v0. The apex
        // opposite the cut edge {v0,v1} is v[2].
        let (m, apex) = (v[0], v[2]);
        // §1 / §6.1 Output-slot note: the apex-connect edge and the fragment
        // faces are *not* inner; only the rim midpoint (recorded in §3) is. The
        // face's geom.out re-split mark was stamped by the dispatch scan.
        self.template_connect(m, apex);
    }

    /// §7.1 / §9.5.3.3 — **the** generic two-edge strip: the operator's one and
    /// only fallback, for every template-less face with exactly two cut edges, at
    /// every face size and every `cuts`. Its `cuts = 1` instances are the cases
    /// §5.2 (two-opposite quad), §5.2.1 (`STRAIGHT_CUT` two-adjacent quad) and
    /// §6.2 (two-edge triangle) describe — three sightings of this one path, not
    /// three templates.
    ///
    /// `rim` is the face's post-cut loop cycle as `(vert, is_pass_two_cut_vert)`.
    /// Its cut verts form two contiguous runs, `a_1..a_N` (the one that comes
    /// first in loop order, reading from an original corner) and `b_1..b_N`. The
    /// runs face each other *antiparallel* around the rim, so they zip
    /// **outside-in** — `a_i` with `b_{N+1-i}` — for `N` connects that do not
    /// cross, and hence exactly `N + 1` fragments for every `N` and every face
    /// size. No hub, no median tie-break, no odd/even asymmetry (contrast
    /// §9.5.3.1's fan).
    ///
    /// Every connect is a split of the **parent face handle itself** — the
    /// fallback deliberately does *not* go through the §7.0 fewest-corners
    /// carrier search, which ranges over the whole mesh and would substitute a
    /// neighbour's fragment for the parent (§7.1). That is sound because the
    /// outside-in pairing leaves the parent holding the *inner band* after each
    /// split: the split retains the input face on the `l_a -> l_b` side
    /// (RADIAL_EDGE_STRUCTURE.md §8.3) and `a_i` is issued as `l_a`, so the side
    /// the parent keeps is the one still carrying every remaining pair. Swap the
    /// two arguments and the parent keeps the outer band, where the remaining
    /// pairs are not.
    ///
    /// Caller obligations, all settled before the call: the §4.3.1 flat-corner
    /// guard, the §7.0.3 abandonment gate, the dispatch that establishes this
    /// face is template-less with `cut_count = 2`, and `connects_dropped` — the
    /// §7.1 precondition, which pass one decided on the input topology
    /// ([`Self::cut_edges_have_a_second_common_face`]). The precondition is the
    /// fallback's own substitute for §7.0's no-op-when-absent rule, and it takes
    /// **all `N`** connects or none: every candidate pair has one endpoint on
    /// each of the same two cut edges, so it cannot differ between the pairs, and
    /// a partially stripped face is not a reachable outcome of the rule.
    fn two_edge_strip(
        &mut self,
        face: FaceKey,
        rim: &[(VertexKey, bool)],
        connects_dropped: bool,
        geom: &mut SubdivideGeom,
    ) {
        if connects_dropped {
            return;
        }

        let Some(runs) = Self::two_cut_runs(rim) else {
            debug_assert!(false, "the strip fires only on a face with two cut-vert runs");
            return;
        };
        debug_assert_eq!(
            runs.run_a.len(),
            runs.run_b.len(),
            "both cut edges carry the same number of cut verts"
        );
        let n = runs.run_a.len();

        for i in 0..n {
            let (cut_vert_a, cut_vert_b) = (runs.run_a[i], runs.run_b[n - 1 - i]);
            let Some((_fragment, connect)) = self.face_split(face, cut_vert_a, cut_vert_b) else {
                // Not a documented no-op on this path (contrast `template_connect`,
                // §7.0.2): the parent keeps the inner band, on which every
                // remaining pair is a diagonal. A refusal here means the retention
                // convention or the argument order broke, and the face would come
                // out partially stripped — an outcome §7.1 says is unreachable.
                debug_assert!(false, "the parent's inner band holds every remaining pair non-adjacent");
                return;
            };
            // §1 rule 3 / §7.1 / §9.5.3.3 Output-slot note: the strip is the
            // one non-grid path that marks a connecting *edge* inner. Its
            // fragment faces are not inner, and the face's re-split mark was
            // stamped once, by the dispatch scan. So a lone two-cut face
            // yields geom_inner.out = {2N verts, N edges, 0 faces}.
            geom.record_inner_edge(connect);
        }
    }

    /// §7.1 — the fallback's dropped-connect precondition, in the reduced form the
    /// guide derives. A connect is laid only when its endpoints are corners of **no
    /// face other than** the one being re-split; a cut vert of edge `e` is a corner
    /// of exactly the faces in `e`'s radial cycle, so on the pre-cut mesh the whole
    /// question is: *do the face's two cut edges have a common face other than this
    /// one?* One answer settles all `N` of the face's connects, because their
    /// endpoints always sit on those same two edges.
    ///
    /// The reduction is a statement about the **input topology**, which is why it
    /// is asked in pass one and not from the strip. The two forms agree only until
    /// pass three starts cutting: from then on an earlier-applied face's fragments
    /// carry cut verts of their own, and a neighbour that shares both cut edges can
    /// have re-split itself out of any single fragment holding the pair — the live
    /// query would then say "lay it" where the reduction says "drop it".
    ///
    /// Two faces sharing two edges is manifold-legal (the valence-2 pillow corner),
    /// and both of them drop: the rule is deliberately not a tiebreak, so a pair
    /// neither face can claim unambiguously is claimed by neither. The §7.0 carrier
    /// search *does* guess in the same situation — which is exactly why the
    /// fallback must not be routed through it.
    fn cut_edges_have_a_second_common_face(
        &self,
        face: FaceKey,
        cut_edge_a: EdgeKey,
        cut_edge_b: EdgeKey,
    ) -> bool {
        let mut radial_a = self.radial_loops(cut_edge_a);
        while let Some(loop_a) = radial_a.next(self) {
            let candidate = self.loops.get(loop_a).face;
            if candidate == face {
                continue;
            }
            let mut radial_b = self.radial_loops(cut_edge_b);
            while let Some(loop_b) = radial_b.next(self) {
                if self.loops.get(loop_b).face == candidate {
                    return true;
                }
            }
        }
        false
    }

    /// Locate the two contiguous cut-vert runs of a two-cut face on its post-cut
    /// `rim`, in loop order from an original corner, and report whether the two
    /// runs are **adjacent** — separated by a single corner on one side, i.e. the
    /// two cut edges share it.
    ///
    /// `None` when the rim carries anything other than exactly two runs (an
    /// all-cut-vert ring has no corner to root at, and a ring pass three has
    /// carved may hold any number of runs — §7.0.3's abandonment gate is what
    /// decides such a face, so this is not an invariant to assert on).
    fn two_cut_runs(rim: &[(VertexKey, bool)]) -> Option<TwoCutRuns> {
        let n = rim.len();
        let start = rim.iter().position(|(_, is_cut)| !*is_cut)?;

        let mut runs: SmallVec<[SmallVec<[VertexKey; 4]>; 2]> = SmallVec::new();
        let mut corners = 0usize;
        let mut corners_between = 0usize;
        let mut in_run = false;
        for i in 0..n {
            let (vert, is_cut) = rim[(start + i) % n];
            if is_cut {
                if !in_run {
                    if runs.len() == 2 {
                        return None;
                    }
                    runs.push(SmallVec::new());
                    in_run = true;
                }
                let last = runs.len() - 1;
                runs[last].push(vert);
            } else {
                in_run = false;
                corners += 1;
                // The rim is rooted at a corner, so a corner counted while exactly
                // one run is open lies strictly between the two runs.
                if runs.len() == 1 {
                    corners_between += 1;
                }
            }
        }
        if runs.len() != 2 {
            return None;
        }

        let run_b = runs.pop().expect("two runs");
        let run_a = runs.pop().expect("two runs");
        let adjacent = corners_between == 1 || corners - corners_between == 1;
        Some(TwoCutRuns {
            run_a,
            run_b,
            adjacent,
        })
    }

    /// §5.2 / §5.2.1 / §6.2 — are the face's two cut runs on **adjacent** sides
    /// (sharing a corner between them)? The two-cut quad's ambiguous row is the
    /// only place this matters: the `quad_corner_type` templates claim the
    /// adjacent case, while the opposite case is always the §7.1 fallback. Both
    /// dispatchers ask it in this one spelling — on a `(4, 2)` face "adjacent" and
    /// "not opposite" are the same question.
    fn cut_runs_are_adjacent(rim: &[(VertexKey, bool)]) -> bool {
        Self::two_cut_runs(rim).is_some_and(|runs| runs.adjacent)
    }

    /// §6.3 — triangle, three edges cut. Cycle is `v0, m01, v1, m12, v2, m20`;
    /// connect the three midpoints pairwise into the centre triangle.
    /// 1 tri -> 4 tris.
    fn tri_three_cut(&mut self, ring: &[(VertexKey, bool)], geom: &mut SubdivideGeom) {
        let (v, _) = Self::rotate_to_first_midpoint(ring);
        // v = [m01, v1, m12, v2, m20, v0].
        let (m01, m12, m20) = (v[0], v[2], v[4]);

        // The §6.3 pair list, verbatim and in order. The endpoint order within a
        // pair is normative (§7.0.1): the first-named vert is the one whose fan
        // the carrier search scans and whose loop the split retains the parent on.
        //
        // m20 -> m12 peels the corner triangle {m12, v2, m20}.
        let peel_a = self.template_connect(m20, m12);
        // m20 -> m01 peels {m20, v0, m01} from the remaining pentagon.
        let peel_b = self.template_connect(m20, m01);
        // m12 -> m01 closes the centre triangle, peeling {m01, v1, m12}.
        let peel_c = self.template_connect(m12, m01);

        // §1 / §6.3 Output-slot note: this is a grid-style fill — its three
        // connecting edges and its new fragment faces are inner. The original
        // `face` keeps its identity (the re-split mark, stamped by the dispatch
        // scan), so it goes to geom.out only; the peeled corner fragments are
        // inner faces. A connect §7.0 dropped created nothing, so it contributes
        // to no slot.
        for (corner, inner) in [peel_a, peel_b, peel_c].into_iter().flatten() {
            geom.record_inner_face(corner);
            geom.record_inner_edge(inner);
        }
    }

    /// §5.1 — quad, one edge cut. Cycle is `v0, m, v1, v2, v3`; connect `m` into
    /// the interior to the two opposite corners `v3` and `v2`. 1 quad -> 3
    /// triangular fragments.
    fn quad_one_cut(&mut self, ring: &[(VertexKey, bool)]) {
        let (v, _) = Self::rotate_to_first_midpoint(ring);
        // v = [m, v1, v2, v3, v0]. The cut edge was {v0, v1}; the two interior
        // targets are v3 = v[3] and v2 = v[2].
        let (m, v2, v3) = (v[0], v[2], v[3]);
        // m -> v3 peels {v0, m, v3}; m -> v2 splits the remaining quad
        // {m, v1, v2, v3}. Both go through the §7.0 primitive: neither pair is
        // guaranteed to be carried by the dispatched face alone (a neighbour
        // sharing two edges can carry it too, §7.1), and the fewest-corners rule
        // decides. Either can also be dropped outright (§7.0.2), leaving the face
        // with fewer than three fragments — which is correct, not a failure.
        //
        // §1 / §5.1 Output-slot note: the two connecting edges (m->v3, m->v2) and
        // the three fragments are *not* inner; only the rim midpoint (recorded in
        // §3) is. The face's geom.out re-split mark was stamped by the dispatch
        // scan and stands whatever these connects do.
        self.template_connect(m, v3);
        self.template_connect(m, v2);
    }

    /// §5.2 — discriminate the two-cut quad's sub-case. After the §3 midpoint
    /// insertion a quad with two cut edges is a six-corner ring whose two
    /// midpoint corners sit either diametrically opposite (the §5.2 *opposite*
    /// loop-cut case) or two apart on the short side (the §5.2 *adjacent*
    /// corner-cut case). The cut edges share a quad corner iff their midpoints
    /// are *not* diametrically opposite in the ring.
    /// Anchor the §5.2 adjacent two-cut quad against its shared corner. The
    /// post-§3 hexagon ring reads `v0, m01, v1, m12, v2, v3` from some start;
    /// rotate so the start corner is the rim midpoint `m01` whose loop successor
    /// is the shared corner `v1`, with the second midpoint `m12` two corners on.
    /// Returns the six corners in that canonical order: `[m01, v1, m12, v2, v3,
    /// v0]`. Shared by all four `quad_corner_type` strategies (§5.2.1–§5.2.4).
    fn adjacent_two_cut_corners(ring: &[(VertexKey, bool)]) -> [VertexKey; 6] {
        let n = ring.len(); // 6
        let mids: SmallVec<[usize; 2]> = ring
            .iter()
            .enumerate()
            .filter(|(_, (_, m))| *m)
            .map(|(i, _)| i)
            .collect();
        debug_assert_eq!(mids.len(), 2, "two-cut quad has exactly two midpoints");
        // Start at the midpoint that has the other exactly two corners ahead, so
        // the shared corner v1 sits between them on the forward arc.
        let start = if (mids[1] - mids[0]) == 2 {
            mids[0]
        } else {
            mids[1]
        };
        let at = |off: usize| ring[(start + off) % n].0;
        debug_assert!(
            !ring[(start + 1) % n].1,
            "shared corner is original, not a midpoint"
        );
        // ring order from m01: m01, v1, m12, v2, v3, v0.
        [at(0), at(1), at(2), at(3), at(4), at(5)]
    }

    /// §5.2.2 — quad, two **adjacent** edges cut, `INNERVERT`. Lay the
    /// `m01`→`m12` shared-corner cut, drop an inner vert `c` at that cut's
    /// midpoint, then spoke `c`→`v3` (the far corner). The corner triangle
    /// `{m01, v1, m12}` gains `c` as a flat rim corner (a 4-gon `{m01, v1, m12,
    /// c}`); the pentagon splits along `c–v3` into two quads `{v0, m01, c, v3}`
    /// and `{v3, c, m12, v2}`. The only adjacent strategy with an inner vert.
    /// 1 quad -> 3 fragments. V 7 / E 9 / F 3.
    fn quad_two_adjacent_inner_vert(
        &mut self,
        ring: &[(VertexKey, bool)],
        layer: crate::op_flags::VertOpFlagLayer,
        params: &SubdivideParams,
        geom: &mut SubdivideGeom,
    ) {
        let [m01, _v1, m12, _v2, v3, _v0] = Self::adjacent_two_cut_corners(ring);

        // 1. Corner cut m01 -> m12, carving the corner triangle off and leaving
        //    the pentagon (the STRAIGHT_CUT intermediate). §1 / §5.2 Output-slot
        //    note: INNERVERT is non-grid for edges/faces — its connecting edges
        //    and fragments are *not* inner; only its three verts (the two rim
        //    midpoints + `c`) are. The face's re-split mark was stamped by the
        //    dispatch scan.
        //
        //    §7.0.2 — the corner cut can be dropped: the fewest-corners carrier of
        //    the pair need not be the dispatched face (an earlier-applied face's
        //    surgery can have left a smaller fragment holding both midpoints, and
        //    holding them *adjacent*). Then there is no edge to hang the inner vert
        //    on, the rest of the pattern has nothing to continue from, and laying
        //    nothing is the documented outcome — not an error.
        let Some((_new_face, corner_cut_edge)) = self.template_connect(m01, m12) else {
            return;
        };

        // 2. Inner vert c at the midpoint of the m01-m12 cut. The edge_split
        //    wrapper places c at t = 0.5 and applies the §7.2 two-source blend
        //    of the cut's endpoints (m01, m12) — exactly the documented inner-
        //    vert rule, *not* the §5.4 face-wide centroid. `c` is the lone
        //    non-grid interior vert that is inner (§1); its two interior
        //    half-edges are *not* inner.
        let (c, _c_half) = self.edge_split(corner_cut_edge, m01, 0.5);
        // §7.0.3 — this split adds a flat corner to *every* face on the corner-cut
        // edge, and one of them can be a face whose own template is still pending
        // (the carrier search is not confined to this quad). Tagging `c` is what
        // lets that face see the re-inflation for what it is.
        self.vert_op_flag_set(layer, c, PASS_THREE_VERT);
        // §9.4.2 — the operator-created inner vert is snapped onto the sphere
        // under `use_sphere`, outward from its chord (corner-cut midpoint).
        self.maybe_sphere_project(c, params);
        geom.record_inner_vert(c);

        // 3. Spoke c -> v3, splitting the pentagon into the two filling quads.
        //    §7.0 can have routed the corner cut into a neighbour, in which case
        //    `c` sits there and no face need carry {c, v3} at all — a no-op, not
        //    an error.
        self.template_connect(c, v3);
    }

    /// §5.2.3 — quad, two **adjacent** edges cut, `PATH`. Lay the `m01`→`m12`
    /// shared-corner cut, then chain a second `v0`→`v2` cut across the remaining
    /// pentagon. Yields the corner triangle `{m01, v1, m12}`, a central quad
    /// `{v0, m01, m12, v2}`, and the far-corner triangle `{v0, v2, v3}`. No
    /// inner vert. 1 quad -> 3 fragments. V 6 / E 8 / F 3.
    fn quad_two_adjacent_path(&mut self, ring: &[(VertexKey, bool)]) {
        let [m01, _v1, m12, v2, _v3, v0] = Self::adjacent_two_cut_corners(ring);

        // 1. Corner cut m01 -> m12. §1 / §5.2 Output-slot note: PATH is
        //    non-grid — its two cuts and three fragments are *not* inner; only
        //    the two rim midpoints (recorded in §3) are.
        self.template_connect(m01, m12);

        // 2. Chained v0 -> v2 cut across the pentagon {v0, m01, m12, v2, v3},
        //    splitting it into the central quad {v0, m01, m12, v2} and the far-
        //    corner triangle {v0, v2, v3}. Chained: if §7.0 routed the corner cut
        //    into a neighbour, the pentagon this cut wants may never exist, and
        //    the pair then resolves elsewhere or nowhere (a no-op, §7.0).
        self.template_connect(v0, v2);
    }

    /// §5.2.4 — quad, two **adjacent** edges cut, `FAN`. Spoke both rim
    /// midpoints to the far corner `v3`: `m01`→`v3` then `m12`→`v3`. Yields the
    /// triangle `{v3, v0, m01}`, a central quad `{v3, m01, v1, m12}` (the only
    /// fragment holding the shared corner `v1`), and the triangle `{v3, m12,
    /// v2}`. `v3` becomes a degree-4 fan hub. No inner vert. 1 quad -> 3
    /// fragments. V 6 / E 8 / F 3.
    fn quad_two_adjacent_fan(&mut self, ring: &[(VertexKey, bool)]) {
        let [m01, _v1, m12, _v2, v3, _v0] = Self::adjacent_two_cut_corners(ring);

        // 1. First spoke m01 -> v3, carving the triangle {v3, v0, m01}.
        self.template_connect(m01, v3);
        // 2. Second spoke m12 -> v3, carving the triangle {v3, m12, v2}, leaving
        //    the central quad {v3, m01, v1, m12} between the two spokes.
        //
        // §1 / §5.2 Output-slot note: FAN is non-grid — the two spokes and the
        // three fragments are *not* inner; only the two rim midpoints (recorded
        // in §3) are.
        self.template_connect(m12, v3);
    }

    /// §5.3 — quad, three edges cut. Three of the four edges carry a midpoint
    /// (`m01`, `m12`, `m23`; edge `{v3,v0}` uncut). After the §3 insertion the
    /// loop cycle is a heptagon `v0, m01, v1, m12, v2, m23, v3` with the three
    /// midpoints two apart on the cut side and the uncut edge spanning `v3→v0`.
    ///
    /// The pattern peels the two corner triangles flanking the *middle*
    /// midpoint `m12` (`m12`→`m01` and `m12`→`m23`), then splits the remaining
    /// pentagon `{v0, m01, m12, m23, v3}` along its `m01`→`m23` diagonal. The
    /// result is three triangles (`{m01,v1,m12}`, `{m12,v2,m23}`,
    /// `{m01,m12,m23}`) plus one quad (`{v0,m01,m23,v3}`). No inner vert.
    /// 1 quad -> 4 fragments.
    fn quad_three_cut(&mut self, ring: &[(VertexKey, bool)]) {
        let n = ring.len(); // 7
        let mids: SmallVec<[usize; 3]> = ring
            .iter()
            .enumerate()
            .filter(|(_, (_, m))| *m)
            .map(|(i, _)| i)
            .collect();
        debug_assert_eq!(mids.len(), 3, "three-cut quad has exactly three midpoints");

        // The middle midpoint is the one with a midpoint two corners away on
        // *both* sides (the other two are the "end" midpoints, each adjacent to
        // the uncut edge). Identify it, then name the three in cut order.
        //
        // §7.0.2 / §7.0.3 — the search can come up empty on a ring the abandonment
        // gate let through but whose midpoint *spacing* no longer matches the
        // template's, so it is not an invariant to assert on: laying nothing is
        // the specified outcome, and this operator must not panic on a reachable
        // configuration.
        let is_mid = |i: usize| ring[i % n].1;
        let Some(&middle) = mids
            .iter()
            .find(|&&i| is_mid((i + 2) % n) && is_mid((i + n - 2) % n))
        else {
            return;
        };

        let m12 = ring[middle].0;
        let m01 = ring[(middle + n - 2) % n].0; // end midpoint trailing m12
        let m23 = ring[(middle + 2) % n].0; // end midpoint leading m12

        // The §5.3 pair list, verbatim and in order — note the middle midpoint
        // `m12` is named *second* in both of the pairs it appears in, and the
        // endpoint order is normative (§7.0.1).
        //
        // m01 -> m12 peels the corner triangle {m01, v1, m12}.
        self.template_connect(m01, m12);
        // m23 -> m12 peels the corner triangle {m12, v2, m23}.
        self.template_connect(m23, m12);
        // m01 -> m23 splits the remaining pentagon {v0, m01, m12, m23, v3} into
        // the centre triangle {m01, m12, m23} and the quad {v0, m01, m23, v3}.
        // Each pair is routed by §7.0 in its own right; one resolving into a
        // neighbour (or nowhere) does not stop the others.
        //
        // §1 — the three-edge quad is *not* a grid fill (§1 lists §5.3 among the
        // non-grid patterns): its three connecting edges and four fragments are
        // not inner; only the three rim midpoints (recorded in §3) are.
        self.template_connect(m01, m23);
    }

    /// §5.4 — quad, all four edges cut. Cycle is
    /// `v0, m01, v1, m12, v2, m23, v3, m30`; insert one inner vert `c` at the
    /// centroid and connect each rim midpoint to it (the "plus" split).
    /// 1 quad -> 2x2 grid (4 quads).
    fn quad_four_cut(
        &mut self,
        face: FaceKey,
        ring: &[(VertexKey, bool)],
        layer: crate::op_flags::VertOpFlagLayer,
        params: &SubdivideParams,
        geom: &mut SubdivideGeom,
    ) {
        let (v, _) = Self::rotate_to_first_midpoint(ring);
        // v = [m01, v1, m12, v2, m23, v3, m30, v0].
        let (m01, m12, m23, m30) = (v[0], v[2], v[4], v[6]);
        let (c0, c1, c2, c3) = (v[7], v[1], v[3], v[5]); // original corners v0..v3

        // The §5.4 pair list: the m01 - m23 bar first, then the centre vert `c`
        // born by splitting it, then c - m30 and m12 - c (note `c` is named first
        // in one and second in the other; the order is normative, §7.0.1).
        //
        // §7.0.2 — the bar can be dropped, the same way §5.2.2's corner cut can:
        // the carrier is whichever face in the *mesh* holds the pair with fewest
        // corners, and a fragment left by an earlier-applied face's surgery can
        // hold the two opposite midpoints adjacent. Without the bar there is no
        // centre vert and no grid, so lay nothing rather than fabricate one.
        let Some((_hex_b, diag)) = self.template_connect(m01, m23) else {
            return;
        };
        let mids = self.edge_split_n(diag, m01, 1);
        let c = mids[0];
        // §7.0.3 — the bar split adds a flat corner to every face on the bar, and
        // the bar's carrier can be a *pending* face rather than a fragment of this
        // quad. Tag `c` so that face's abandonment gate can see it.
        self.vert_op_flag_set(layer, c, PASS_THREE_VERT);

        // §7.2 — the inner vert is a face-wide blend of the four original
        // corners (equal weights for a quad), not the along-edge blend the
        // edge_split wrapper applied. Override its position and per-vert
        // customdata accordingly. (Loop CD on c's corners is left as the
        // along-diagonal blend, which is the same family of two-source lerp;
        // pinning the per-corner face average is the natural follow-up.)
        let centroid = (self.vertex_position(c0)
            + self.vertex_position(c1)
            + self.vertex_position(c2)
            + self.vertex_position(c3))
            * 0.25;
        self.set_vertex_position(c, centroid);
        let w = 0.25_f32;
        let sources = [
            (w, c0.as_usize()),
            (w, c1.as_usize()),
            (w, c2.as_usize()),
            (w, c3.as_usize()),
        ];
        self.vertex_attribs.interpolate_into(c.as_usize(), &sources);

        // §9.4.4 — under `use_sphere`, the inner vert is snapped onto the sphere
        // outward from its centroid chord position (the rim midpoints were
        // already snapped in phase one). Attribute interpolation is unchanged.
        self.maybe_sphere_project(c, params);

        // Connect the remaining two rim midpoints to c, finishing the four
        // spokes (m01-c and m23-c already exist as the halved diagonal). If §7.0
        // routed the diagonal into a neighbour, `c` was born there and these
        // pairs may have no carrier — a no-op, not an error (§7.0). The inner
        // slots below are read back off `c`'s actual incidence, so they stay
        // truthful either way.
        self.template_connect(c, m30);
        self.template_connect(m12, c);

        // §1 / §5.4 Output-slot note: this grid fill marks its interior
        // geometry inner — `c`, the four "plus" spokes, and the new fragment
        // faces. One fragment keeps `face`'s identity (the re-split mark, stamped
        // by the dispatch scan) and is in geom.out only; the rest are inner. The
        // four cells are exactly c's incident faces.
        geom.record_inner_vert(c);
        for f in self.vert_face_keys(c) {
            if f != face {
                geom.record_inner_face(f);
            }
        }
        // The four "plus" spokes are exactly c's incident edges (the halved
        // diagonal m01-c / c-m23 plus the two new c-m12 / c-m30 splits); each is
        // an inner edge (§1 geom_inner).
        let mut disk = self.disk_edges(c);
        while let Some(spoke) = disk.next(self) {
            geom.record_inner_edge(spoke);
        }
    }

    // ---- Multi-cut (`cuts >= 2`, §3.3 / §9.5) ----------------------------

    /// Multi-cut driver (§3.3 + §9.5). Inserts `N` evenly-spaced midpoints on
    /// each cut edge (§3.3, crack-free / shared-once on interior edges), then
    /// fills each touched face with its §9.5 lattice: the fully-cut quad
    /// `(N+1)x(N+1)` grid (§9.5.1), the fully-cut triangle lattice (§9.5.2), and
    /// the partial-cut fans (§9.5.3). The per-face pattern is keyed on the same
    /// `(corner count, cut-side count)` pair as the single-cut dispatch (§4.1);
    /// `N` scales the *depth* of each pattern, not which pattern fires.
    fn subdivide_edges_multicut(
        &mut self,
        edges: &[EdgeKey],
        params: SubdivideParams,
    ) -> SubdivideResult {
        let cuts = params.cuts;
        let mut geom = SubdivideGeom::default();

        // §9.1.1 / §9.2.3 — endpoint vertex normals must be current before any
        // off-chord or fractal placement. Freshen once; the chord-only path
        // (`smooth == 0 && fractal == 0`) keeps the core untouched.
        if params.smooth != 0.0 || params.fractal != 0.0 {
            self.mesh_normals_update();
        }

        // §9.2.1 — the seed is consumed exactly once, shared by every inserted
        // vert. `None` when fractal is off.
        let fractal_offset = self.fractal_offset(&params);

        // §9.5.3.5 — the flat-corner guard is `cuts`-independent: it reads the
        // *input* edge directions, so it is taken exactly as at `cuts = 1`,
        // before the cuts rewrite them.
        let (guard_layer, cut_layer, touched) = self.push_flat_corner_guard(edges);

        // ---- Pass one: the dispatch scan (§4.4) ----
        let records = self.dispatch_scan(cut_layer, touched, guard_layer, &params, &mut geom);
        self.pop_edge_op_flag_layer(cut_layer);

        let midpoint_layer = self.push_vert_op_flag_layer();

        // §9.2.3 deferred write — staged fractal displacements, copied back only
        // after every cut edge is processed (§9.2.3.3 last-writer overwrite).
        let mut fractal_scratch: std::collections::HashMap<VertexKey, Vec3> =
            std::collections::HashMap::new();

        // Pass two: per-edge N-evenly-spaced split (§3.3). Each input edge is
        // split into N+1 segments by inserting N midpoints at t = i/(N+1). A
        // shared interior edge is cut once, so both incident faces see the same
        // midpoint run — crack-free (§9.5.4).
        for &edge in edges {
            let [near, far] = self.edge_vert_keys(edge);
            // §9.1 — capture the smooth endpoint context before the split. Each
            // of the N midpoints lands at the directional cut parameter
            // t = i/(N+1) (§3.3), measured from `near`.
            // §9.4.3 — the §9.1 offset is skipped under `use_sphere` (sphere wins).
            let smooth_ctx = (!params.use_sphere && params.smooth != 0.0)
                .then(|| self.smooth_endpoint_context(edge, near));
            // §9.2.3 — snapshot the fractal basis before the split (it reads the
            // same pre-split endpoint positions / normals).
            let fractal_ctx = fractal_offset.map(|_| self.fractal_edge_context(edge, near));
            let mids = self.edge_split_n(edge, near, cuts);
            if let Some(ctx) = &smooth_ctx {
                let denom = cuts as f32 + 1.0;
                for (i, &m) in mids.iter().enumerate() {
                    let t = (i as f32 + 1.0) / denom;
                    let co_final = ctx.placement(t, &params);
                    self.set_vertex_position(m, co_final);
                    self.vert_normal_update(m);
                }
            }
            for &m in &mids {
                // §9.4.1 — radial snap each rim midpoint from its chord position.
                self.maybe_sphere_project(m, &params);
                // §9.2.3 / §9.2.6 — fractal is the last placement step. Stages
                // each rim midpoint and the cut edge's two endpoints (§9.2.3.3)
                // into the deferred scratch store.
                if let (Some(offset), Some(ctx)) = (fractal_offset, &fractal_ctx) {
                    self.stage_fractal(m, offset, ctx, &params, &mut fractal_scratch);
                }
                self.vert_op_flag_set(midpoint_layer, m, MIDPOINT);
                // §1 — every rim midpoint is both a split vert and an inner vert.
                geom.record_split_vert(m);
            }
            // §1 — the N+1 edge-halves are the segments of the now-subdivided
            // input edge: the consecutive pairs of [near, mids.., far].
            let mut prev = near;
            for &m in &mids {
                if let Some(half) = self.query_edge_exists(prev, m) {
                    geom.push_split_edge(half);
                }
                prev = m;
            }
            if let Some(half) = self.query_edge_exists(prev, far) {
                geom.push_split_edge(half);
            }
        }

        // §9.2.3 — copy the staged displacements back now that every cut edge
        // has been processed (the deferred write).
        for (v, co) in &fractal_scratch {
            self.set_vertex_position(*v, *co);
        }

        // ---- Pass three: apply the recorded §9.5 patterns, last-recorded-first
        // (§4.4 / §7.0.3(a)). ----
        for record in records.iter().rev() {
            self.resplit_face_multicut(
                record,
                midpoint_layer,
                guard_layer,
                cuts,
                &params,
                &mut geom,
            );
        }

        self.pop_vert_op_flag_layer(midpoint_layer);
        self.pop_face_op_flag_layer(guard_layer);
        SubdivideResult::Ran(geom)
    }

    /// Decompose a touched face's post-§3.3 rim — `(vert, is_cut_vert)` in loop
    /// order — into its original corners and, for each side, the ordered midpoint
    /// run between consecutive corners (in face-traversal order). A side with an
    /// empty run was not cut. The §9.5 lattice patterns index these; the §7.1
    /// strip does not (it reads its two runs off the rim directly).
    ///
    /// `None` for an all-midpoint ring, which has no corner to rotate to. That is
    /// not an invariant to assert on: pass three's own surgery can leave a face
    /// with a ring the dispatcher never saw (§7.0.3), and the abandonment gate —
    /// which consumes this decomposition — is what decides such a face lays
    /// nothing. Panicking here would turn a specified no-op into a crash.
    fn corners_and_sides(
        rim: &[(VertexKey, bool)],
    ) -> Option<(Vec<VertexKey>, Vec<Vec<VertexKey>>)> {
        // Rotate so the rim starts at an original corner.
        let start = rim.iter().position(|(_, m)| !*m)?;
        let n = rim.len();

        let mut corners: Vec<VertexKey> = Vec::new();
        let mut sides: Vec<Vec<VertexKey>> = Vec::new();
        let mut current_run: Vec<VertexKey> = Vec::new();
        for i in 0..n {
            let (v, is_mid) = rim[(start + i) % n];
            if is_mid {
                current_run.push(v);
            } else {
                if !corners.is_empty() {
                    sides.push(std::mem::take(&mut current_run));
                }
                corners.push(v);
            }
        }
        // The trailing run closes the cycle back to corners[0].
        sides.push(std::mem::take(&mut current_run));
        debug_assert_eq!(sides.len(), corners.len());
        Some((corners, sides))
    }

    /// §7.0.3 — the abandonment gate's reading of a re-derived rim, taken in one
    /// allocation-free walk rooted at an original corner: how many original
    /// corners the ring carries, how many contiguous cut-vert runs, whether every
    /// run is exactly `cuts` long, and whether the recorded anchor is still a
    /// corner.
    ///
    /// `None` for an all-cut-vert ring — no corner to root at. That is not an
    /// invariant to assert on: pass three's own surgery can leave a face with a
    /// ring the dispatcher never saw, and the gate is what decides such a face
    /// lays nothing.
    fn rim_census(rim: &[(VertexKey, bool)], anchor: VertexKey, cuts: usize) -> Option<RimCensus> {
        let n = rim.len();
        let start = rim.iter().position(|(_, is_cut)| !*is_cut)?;

        let mut census = RimCensus {
            corner_count: 0,
            run_count: 0,
            runs_uniform: true,
            has_anchor: false,
        };
        let mut run_len = 0usize;
        for i in 0..n {
            let (vert, is_cut) = rim[(start + i) % n];
            if is_cut {
                if run_len == 0 {
                    census.run_count += 1;
                }
                run_len += 1;
            } else {
                if run_len != 0 {
                    census.runs_uniform &= run_len == cuts;
                    run_len = 0;
                }
                census.corner_count += 1;
                census.has_anchor |= vert == anchor;
            }
        }
        // Rooted at a corner, so a run still open here is the one that closes the
        // cycle back to it.
        if run_len != 0 {
            census.runs_uniform &= run_len == cuts;
        }
        Some(census)
    }

    fn resplit_face_multicut(
        &mut self,
        record: &FaceRecord,
        layer: crate::op_flags::VertOpFlagLayer,
        guard_layer: crate::op_flags::FaceOpFlagLayer,
        cuts: u32,
        params: &SubdivideParams,
        geom: &mut SubdivideGeom,
    ) {
        let face = record.face;

        // §7.0.3 / §7.1 — abandonment, on the same three clauses as the single-cut
        // path, and settled in full *before* anything reads the ring's shape. A
        // lattice face is exposed to carving exactly as a template face is: it
        // shares the reverse apply order, and the carrier search ranges over the
        // whole mesh, so an earlier-applied connect can have carved it before its
        // turn — and a pass-three inner vert can then re-inflate it back to the
        // recorded length. Every unwrap and every fixed index below is only sound
        // on a ring that survived this gate.
        if self.face_len(face) != record.expected_len(cuts) {
            return;
        }

        let mut rim: SmallVec<[(VertexKey, bool); 12]> = SmallVec::new();
        let mut re_inflated = false;
        let mut it = self.face_loops(face);
        while let Some(l) = it.next(self) {
            let v = self.loops.get(l).vertex;
            re_inflated |= self.vert_op_flag_test(layer, v, PASS_THREE_VERT);
            rim.push((v, self.vert_op_flag_test(layer, v, MIDPOINT)));
        }
        if re_inflated {
            return;
        }

        // §7.0.3's strict census: the cut verts must sit in exactly `cut_count`
        // contiguous runs of exactly `cuts`, on a ring of `pre_len` original
        // corners still carrying the recorded anchor. Anything else is a ring the
        // pattern was never matched for. It is read straight off the rim, so the
        // strip — which reads its two runs off the rim as well — never pays for
        // the §9.5 lattice decomposition it does not use.
        //
        // §7.1 exempts a *fallback* record from the anchor clause — it locates its
        // runs on the re-derived ring, never by rotating to an anchor, so the
        // anchor is not a thing it depends on. loopmesh keeps the clause on that
        // path too (the record carries an anchor regardless): it is the strictly
        // stronger test, and any ring it rejects has already lost a corner the
        // scan saw, which the length and census clauses reject anyway.
        let Some(census) = Self::rim_census(&rim, record.anchor, cuts as usize) else {
            return;
        };
        if census.corner_count != record.pre_len
            || census.run_count != record.cut_count
            || !census.runs_uniform
            || !census.has_anchor
        {
            return;
        }

        // §9.3 — `use_only_quads` gates the *pattern*, not the record (§4.4): the
        // face keeps its re-split mark and its rim cut verts, and stays flat.
        if params.use_only_quads && census.corner_count != 4 {
            return;
        }

        match (census.corner_count, census.run_count) {
            // §5.2.2–§5.2.4 — the `quad_corner_type` templates claim the
            // two-adjacent quad away from the fallback. Their multi-cut arms are
            // their own parity row and are not implemented, so such a face stays
            // flat rather than being handed to the strip, which is not its
            // pattern. A template is never guarded (§4.3.1), hence no verdict here.
            (4, 2)
                if params.quad_corner_type != QuadCornerType::StraightCut
                    && Self::cut_runs_are_adjacent(&rim) =>
            {
                return;
            }
            (_, 2) => {
                // §9.5.3.3 / §7.1 — the generic two-edge strip at `cuts = N`, and
                // the same code path as at `cuts = 1`: it is face-size agnostic
                // and `N`-agnostic, zipping the two runs outside-in into `N`
                // connects and `N + 1` fragments (or, when §7.1's precondition
                // drops them, none at all and one fragment).
                //
                // §9.5.3.5 pins the guard as `cuts`-independent — it reads the
                // *input* edge directions — so a face the dispatch scan refused to
                // record at `cuts = 1` is refused at every `N`.
                debug_assert!(
                    !self.face_op_flag_test(guard_layer, face, FLAT_CORNER),
                    "a guarded template-less face is never recorded (§4.3.1)"
                );
                self.two_edge_strip(face, &rim, record.strip_connects_dropped, geom);
                return;
            }
            _ => {}
        }

        let Some((corners, sides)) = Self::corners_and_sides(&rim) else {
            debug_assert!(false, "the census already found a corner to root at");
            return;
        };

        let n = cuts as usize;
        match (census.corner_count, census.run_count) {
            (4, 4) => {
                if params.use_grid_fill {
                    self.quad_grid_fill(face, &corners, &sides, n, layer, params, geom);
                }
            }
            (3, 3) => {
                if params.use_grid_fill {
                    self.tri_lattice_fill(face, &corners, &sides, n, layer, params, geom);
                }
            }
            (4, 1) => {
                if params.use_single_edge {
                    self.quad_one_edge_fan(&corners, &sides, n);
                }
            }
            (3, 1) => {
                if params.use_single_edge {
                    self.tri_one_edge_fan(&corners, &sides);
                }
            }
            // The three-edge quad template (§9.5.3.4) is still a gap, as is any
            // remaining template-less cut count — an n-gon with one, three, or
            // more cut edges is never re-split at any N (§4.3). These faces keep
            // their §3.3 rim cut verts and stay flat-cornered.
            _ => {}
        }
    }

    /// Create a lattice cell face from a vert ring, reusing existing boundary
    /// edges and creating interior edges once. The new face inherits the
    /// source face's per-face customdata (§7.2 one-source copy). Each cell edge
    /// is recorded into `geom.out`; any cell edge that is not a phase-one
    /// edge-half (so it lies strictly inside the source face) is also an inner
    /// edge (§1 geom_inner).
    fn create_lattice_cell(
        &mut self,
        ring: &[VertexKey],
        src_face: usize,
        geom: &mut SubdivideGeom,
    ) -> FaceKey {
        let count = ring.len();
        let mut cell_edges: SmallVec<[EdgeKey; 4]> = SmallVec::new();
        for i in 0..count {
            let a = ring[i];
            let b = ring[(i + 1) % count];
            let e = self.create_edge([a, b], EdgeCreateFlags::NO_DUPLICATE);
            if !geom.edges.contains(&e) {
                geom.edges.push(e);
            }
            // A cell edge that is not a rim edge-half (captured in phase one) is
            // interior to the source face — an inner edge.
            if !geom.split_edges.contains(&e) && !geom.inner_edges.contains(&e) {
                geom.inner_edges.push(e);
            }
            cell_edges.push(e);
        }
        let new_face = self.face_create(ring, &cell_edges, FaceCreateFlags::NONE);
        self.face_attribs.copy_into(new_face.as_usize(), src_face);
        new_face
    }

    /// §9.5.1 — fully-cut quad, the `(N+1)x(N+1)` grid. `corners` is
    /// `[v0, v1, v2, v3]` in loop order; `sides[k]` is the midpoint run on the
    /// edge `corners[k]->corners[k+1]`. Builds the node grid (rim midpoints +
    /// `N*N` interior bilinear verts), kills the n-gon, and emits `(N+1)^2`
    /// quad cells.
    fn quad_grid_fill(
        &mut self,
        face: FaceKey,
        corners: &[VertexKey],
        sides: &[Vec<VertexKey>],
        n: usize,
        layer: crate::op_flags::VertOpFlagLayer,
        params: &SubdivideParams,
        geom: &mut SubdivideGeom,
    ) {
        let (v0, v1, v2, v3) = (corners[0], corners[1], corners[2], corners[3]);
        let p = [
            self.vertex_position(v0),
            self.vertex_position(v1),
            self.vertex_position(v2),
            self.vertex_position(v3),
        ];

        // node[row][col], row,col in 0..=n+1. col increases v0->v1, row
        // increases v0->v3 (so v2 is at (n+1, n+1)).
        let dim = n + 2;
        let mut node = vec![vec![VertexKey::from_usize(0); dim]; dim];
        node[0][0] = v0;
        node[0][n + 1] = v1;
        node[n + 1][n + 1] = v2;
        node[n + 1][0] = v3;
        for c in 1..=n {
            node[0][c] = sides[0][c - 1]; // bottom: v0->v1
            node[n + 1][c] = sides[2][n - c]; // top: v2->v3, reversed
        }
        for r in 1..=n {
            node[r][n + 1] = sides[1][r - 1]; // right: v1->v2
            node[r][0] = sides[3][n - r]; // left: v3->v0, reversed
        }

        // Interior bilinear verts (§9.5.1). u = col/(N+1), v = row/(N+1).
        for r in 1..=n {
            for c in 1..=n {
                let u = c as f32 / (n as f32 + 1.0);
                let v = r as f32 / (n as f32 + 1.0);
                let pos = p[0] * ((1.0 - u) * (1.0 - v))
                    + p[1] * (u * (1.0 - v))
                    + p[2] * (u * v)
                    + p[3] * ((1.0 - u) * v);
                let vk = self.create_vertex(pos);
                let sources = [
                    ((1.0 - u) * (1.0 - v), v0.as_usize()),
                    (u * (1.0 - v), v1.as_usize()),
                    (u * v, v2.as_usize()),
                    ((1.0 - u) * v, v3.as_usize()),
                ];
                self.vertex_attribs
                    .interpolate_into(vk.as_usize(), &sources);
                // §9.4.4 — snap each interior lattice node onto the sphere.
                self.maybe_sphere_project(vk, params);
                // §7.0.3 — a pass-three-born vert, like the §5.2.2 / §5.4 inner
                // verts. The lattice rebuilds its face rather than splitting an
                // edge, so this one cannot re-inflate a neighbour today; tagging it
                // keeps the predicate meaning "minted by pass three", which is what
                // the gate reads.
                self.vert_op_flag_set(layer, vk, PASS_THREE_VERT);
                node[r][c] = vk;
                geom.record_inner_vert(vk);
            }
        }

        let src_face = face.as_usize();
        self.face_kill(face);
        // §1 / §5.4 Output-slot note: the lattice fill is grid-style — its
        // interior verts/edges and its fragment faces are inner. The original
        // face was killed and rebuilt rather than split, so its re-split mark
        // (stamped in the dispatch scan) moves onto the cell that stands in for
        // it; the rest are inner faces.
        let mut first = true;
        for r in 0..=n {
            for c in 0..=n {
                let cell = [
                    node[r][c],
                    node[r][c + 1],
                    node[r + 1][c + 1],
                    node[r + 1][c],
                ];
                let f = self.create_lattice_cell(&cell, src_face, geom);
                if first {
                    geom.replace_resplit_face(face, f);
                    first = false;
                } else {
                    geom.record_inner_face(f);
                }
            }
        }
    }

    /// §9.5.2 — fully-cut triangle, the `(N+1)^2`-cell lattice. `corners` is
    /// `[v0, v1, v2]`; `sides[k]` is the midpoint run on `corners[k]->
    /// corners[k+1]`. Builds the barycentric node grid (`(N-1)N/2` interior
    /// verts), kills the n-gon, and emits `(N+1)^2` small triangles.
    fn tri_lattice_fill(
        &mut self,
        face: FaceKey,
        corners: &[VertexKey],
        sides: &[Vec<VertexKey>],
        n: usize,
        layer: crate::op_flags::VertOpFlagLayer,
        params: &SubdivideParams,
        geom: &mut SubdivideGeom,
    ) {
        let (v0, v1, v2) = (corners[0], corners[1], corners[2]);
        let p = [
            self.vertex_position(v0),
            self.vertex_position(v1),
            self.vertex_position(v2),
        ];

        // rows[s] holds the node run at lattice row s (s = 0 on edge v0->v1,
        // s = n+1 at apex v2). Row s has (n+2-s) nodes, left->right.
        let mut rows: Vec<Vec<VertexKey>> = Vec::with_capacity(n + 2);
        for s in 0..=(n + 1) {
            let width = n + 1 - s; // last index in the row
            let mut row: Vec<VertexKey> = Vec::with_capacity(width + 1);
            for k in 0..=width {
                let vk = if s == 0 && k == 0 {
                    v0
                } else if s == 0 && k == n + 1 {
                    v1
                } else if s == n + 1 {
                    v2
                } else if s == 0 {
                    sides[0][k - 1] // bottom: v0->v1
                } else if k == 0 {
                    sides[2][n - s] // left edge v2->v0, reversed
                } else if k == width {
                    sides[1][s - 1] // right edge v1->v2
                } else {
                    // Interior barycentric vert (§9.5.2).
                    let w2 = s as f32 / (n as f32 + 1.0);
                    let w1 = k as f32 / (n as f32 + 1.0);
                    let w0 = 1.0 - w1 - w2;
                    let pos = p[0] * w0 + p[1] * w1 + p[2] * w2;
                    let vk = self.create_vertex(pos);
                    let sources = [
                        (w0, v0.as_usize()),
                        (w1, v1.as_usize()),
                        (w2, v2.as_usize()),
                    ];
                    self.vertex_attribs
                        .interpolate_into(vk.as_usize(), &sources);
                    // §9.4.4 — snap each interior lattice node onto the sphere.
                    self.maybe_sphere_project(vk, params);
                    // §7.0.3 — pass-three-born (see the quad grid's node loop).
                    self.vert_op_flag_set(layer, vk, PASS_THREE_VERT);
                    geom.record_inner_vert(vk);
                    vk
                };
                row.push(vk);
            }
            rows.push(row);
        }

        let src_face = face.as_usize();
        self.face_kill(face);
        // §1 / §6.3 Output-slot note: grid-style fill — interior geometry and
        // fragments are inner; the original face was killed and rebuilt, so its
        // re-split mark moves onto the cell standing in for it.
        let mut first = true;
        let mut record = |this: &mut Self, cell: &[VertexKey], geom: &mut SubdivideGeom| {
            let f = this.create_lattice_cell(cell, src_face, geom);
            if first {
                geom.replace_resplit_face(face, f);
                first = false;
            } else {
                geom.record_inner_face(f);
            }
        };
        for s in 0..=n {
            let upper = n - s; // top index of the down-triangle run
                               // Up triangles.
            for k in 0..=upper {
                let cell = [rows[s][k], rows[s][k + 1], rows[s + 1][k]];
                record(self, &cell, geom);
            }
            // Down triangles.
            for k in 0..upper {
                let cell = [rows[s][k + 1], rows[s + 1][k + 1], rows[s + 1][k]];
                record(self, &cell, geom);
            }
        }
    }

    /// §9.5.3.1 — one-edge quad fan (the half-split two-hub fan). The single cut
    /// edge `v0->v1` carries the `N` cut verts `m_1..m_N`; each connects to the
    /// hub on its own half of the cut edge — `v3` (the corner adjacent to `v0`)
    /// when `2i <= N + 1`, `v2` (adjacent to `v1`) when `2i >= N + 1`. The two
    /// tests deliberately overlap on the median vert (`2i == N + 1`, which exists
    /// iff `N` is odd), which therefore double-hubs. Fragments are hence
    /// `N + 1 + [N odd]`, not `N + 1`: 3/3/5/5/7 at `N = 1..5`.
    fn quad_one_edge_fan(&mut self, corners: &[VertexKey], sides: &[Vec<VertexKey>], n: usize) {
        let cut = sides
            .iter()
            .position(|s| !s.is_empty())
            .expect("one cut side");
        // Rotate corners so the cut side is corners[0]->corners[1].
        let v0 = corners[cut];
        let v1 = corners[(cut + 1) % 4];
        let v2 = corners[(cut + 2) % 4];
        let v3 = corners[(cut + 3) % 4];
        let run = &sides[cut];
        let _ = (v0, v1);

        // §9.5.3.1 half-split rule. `position` is 1-based along the run from v0
        // toward v1; the `<=` / `>=` pair is not a partition — it overlaps on the
        // median (2*position == n + 1), which lays *both* connects.
        //
        // §1 / §5.1 Output-slot note: the one-edge fan is non-grid — its
        // connecting edges and fragments are *not* inner; only the rim midpoints
        // (recorded in §3) are. The face's re-split mark was stamped by the
        // dispatch scan and does not depend on a spoke having landed.
        for (zero_based, &m) in run.iter().enumerate() {
            let position = zero_based + 1;
            if 2 * position <= n + 1 {
                self.template_connect(m, v3);
            }
            if 2 * position >= n + 1 {
                self.template_connect(m, v2);
            }
        }
    }

    /// §9.5.3.2 — one-edge triangle fan (single-hub). The apex is the only corner
    /// off the cut edge, so there is no hub choice and no median tie-break: every
    /// one of the `N` cut verts connects to it, giving `N + 1` triangles. This is
    /// the fan whose count genuinely is `N + 1` — unlike the quad's (§9.5.3.1).
    fn tri_one_edge_fan(&mut self, corners: &[VertexKey], sides: &[Vec<VertexKey>]) {
        let cut = sides
            .iter()
            .position(|s| !s.is_empty())
            .expect("one cut side");
        let apex = corners[(cut + 2) % 3];
        let run = sides[cut].clone();

        // §1 / §6.1 Output-slot note: the one-edge tri fan is non-grid — its
        // apex-connect edges and fragments are *not* inner; only the rim
        // midpoints (recorded in §3) are. The re-split mark was stamped by the
        // dispatch scan.
        for m in run {
            self.template_connect(m, apex);
        }
    }
}

/// The §9.1 smooth-placement inputs for one cut edge: the endpoint positions
/// and (current) vertex normals. Pure geometry — independent of the mesh, so
/// the offset can be evaluated after the edge is split without re-reading the
/// rewritten edge.
#[derive(Clone, Copy, Debug)]
struct SmoothEndpointContext {
    co_a: Vec3,
    co_b: Vec3,
    no_a: Vec3,
    no_b: Vec3,
}

/// A near-zero edge direction guard. Below this the edge is degenerate and the
/// offset collapses to the chord.
const SMOOTH_DIR_EPS: f32 = 1.0e-9;

impl SmoothEndpointContext {
    /// The final §9.1 placement at cut parameter `t`: lerp from the chord
    /// position `co_flat` toward the curved target `co_curved` (§9.1.2) by the
    /// assembled `factor` (§9.1.3). At `smooth == 0`, or where the endpoint
    /// normals are parallel and perpendicular to the edge, this returns exactly
    /// `co_flat` — the §3 chord midpoint.
    fn placement(&self, t: f32, params: &SubdivideParams) -> Vec3 {
        let co_flat = self.co_a.lerp(self.co_b, t);
        let co_curved = self.curved_target(t);

        // §9.1.3 — factor = falloff_base · [shell_factor] · smooth.
        let mut factor = params.smooth_falloff.base(t);
        if params.use_smooth_even {
            factor *= shell_factor(self.no_a, self.no_b);
        }
        factor *= params.smooth;

        co_flat.lerp(co_curved, factor)
    }

    /// §9.1.2 — the curved target: the dual-blended circular arc. Two one-sided
    /// arcs are built, each keyed to a *single* endpoint's normal (the far end
    /// honours that same normal's reflection across the chord), then blended at
    /// `t`. Each arc collapses to the straight chord lerp when its keyed normal
    /// coincides with its own reflection (a chord-perpendicular normal — flat),
    /// so a uniformly flat edge returns `co_flat` and `smooth` is a no-op there.
    fn curved_target(&self, t: f32) -> Vec3 {
        let chord = self.co_a - self.co_b;
        if chord.length_squared() < SMOOTH_DIR_EPS {
            return self.co_a.lerp(self.co_b, t);
        }
        // §9.1.2 — the single sign convention `chord_dir = normalize(co_a - co_b)`
        // flows through every reflection and every arc.
        let chord_dir = chord.normalize();

        // Start-keyed arc: driven entirely by `no_a`; its own reflection across
        // the chord supplies the far-end normal.
        let reflected_a = reflect(self.no_a, chord_dir);
        let toward_start = if normal_coincides_with_reflection(self.no_a, reflected_a) {
            self.co_a.lerp(self.co_b, t)
        } else {
            arc_point(self.co_a, self.no_a, self.co_b, reflected_a, chord_dir, t)
        };

        // End-keyed arc: driven entirely by `no_b`.
        let reflected_b = reflect(self.no_b, chord_dir);
        let toward_end = if normal_coincides_with_reflection(self.no_b, reflected_b) {
            self.co_a.lerp(self.co_b, t)
        } else {
            arc_point(self.co_a, reflected_b, self.co_b, self.no_b, chord_dir, t)
        };

        toward_start.lerp(toward_end, t)
    }
}

/// §9.4.1 — the dome / to-sphere radial snap: keep an inserted vert's direction
/// from the origin and reset its distance to `radius`, i.e.
/// `co_sphere = normalize(co_chord) * radius`, origin-centred. A pure radial
/// function of the chord position and the radius — no normal, no tangent, no arc
/// (§9.4.5). A vert whose chord position is at the origin has no defined radial
/// direction (§9.4.1 degeneracy); `normalize_or_zero` leaves it at the origin in
/// that case, which the documented usage avoids by keeping the origin off every
/// cut chord and lattice node.
fn sphere_project(co_chord: Vec3, radius: f32) -> Vec3 {
    co_chord.normalize_or_zero() * radius
}

/// The §9.2.3 fractal basis for one cut edge `{v_a, v_b}`: the edge length (the
/// displacement scales with it), the two endpoint vertex normals (the
/// displacement frame is built from their plain, *un-renormalised* average), and
/// the two pre-existing endpoint verts (which are jittered alongside the
/// inserted midpoint, §9.2.3.3). Captured before the split since the split
/// rewrites the far endpoint.
#[derive(Clone, Copy, Debug)]
struct FractalEdgeContext {
    edge_len: f32,
    no_a: Vec3,
    no_b: Vec3,
    /// The cut edge's first endpoint `v_a` and its pristine input position.
    va: VertexKey,
    co_a: Vec3,
    /// The cut edge's second endpoint `v_b` and its pristine input position.
    vb: VertexKey,
    co_b: Vec3,
}

/// §9.2.3 — the per-vert fractal displacement. `co` is the vert's current
/// (chord / smooth / sphere) position; `offset` is the §9.2.1 per-operator
/// constant. Returns the displaced position. A pure function — no mesh access.
fn fractal_displace(
    co: Vec3,
    offset: Vec3,
    ctx: &FractalEdgeContext,
    params: &SubdivideParams,
) -> Vec3 {
    // 1. Magnitude scales with the cut edge's length (§9.2.3 step 1).
    let fac = params.fractal * ctx.edge_len;

    // 2. Sample point: shift by the per-operator offset, scale by the fixed
    //    frequency constant 10.0 (§9.2.3 step 2).
    let sample = (co + offset) * 10.0;

    // 3. Three decorrelated turbulence components, each recentred to ±0.5 and
    //    scaled by `fac`. The three coordinate orderings are load-bearing
    //    (§9.2.3 step 3): (x,y,z), (y,x,z), (y,z,x).
    let t0 = fractal_noise::turbulence(sample.x, sample.y, sample.z) - 0.5;
    let t1 = fractal_noise::turbulence(sample.y, sample.x, sample.z) - 0.5;
    let t2 = fractal_noise::turbulence(sample.y, sample.z, sample.x) - 0.5;
    let tvec = Vec3::new(fac * t0, fac * t1, fac * t2);

    // 4. Apply along the frame derived from the plain (un-renormalised) average
    //    endpoint normal plus the §9.2.3.4 closed-form tangents (§9.2.3 step 4).
    //    `along_normal` attenuates *only* the two tangential components (§9.2.4).
    let normal = 0.5 * (ctx.no_a + ctx.no_b);
    let (base1, base2) = tangent_frame(normal);
    let tangential = 1.0 - params.along_normal;

    co + normal * tvec.x + base1 * (tvec.y * tangential) + base2 * (tvec.z * tangential)
}

/// §9.2.3.4 — the closed-form tangent frame of the (un-renormalised) averaged
/// endpoint `normal = (nx, ny, nz)`. Returns `(base1, base2)` where `base1` is
/// the **unit** in-plane perpendicular to `normal`'s xy-projection and `base2`
/// is `normal × base1`, whose length equals `|normal|` (it is **not**
/// re-normalised — that length attenuates the third displacement component on a
/// non-flat edge exactly as the normal component is attenuated).
///
/// Anchored on the squared length of the xy-projection `f = nx² + ny²`. When the
/// projection vanishes (`normal` along ±z, `f ≤ f32::EPSILON`) the frame falls
/// back to the axis-aligned pair `base1 = (±1, 0, 0)` (x-sign from `nz`),
/// `base2 = (0, 1, 0)`.
fn tangent_frame(normal: Vec3) -> (Vec3, Vec3) {
    let (nx, ny, nz) = (normal.x, normal.y, normal.z);
    let xy_proj_sq = nx * nx + ny * ny;

    if xy_proj_sq > f32::EPSILON {
        let inv_xy_len = 1.0 / xy_proj_sq.sqrt();
        // Unit in-plane perpendicular to the xy-projection (`z = 0`).
        let base1 = Vec3::new(ny * inv_xy_len, -nx * inv_xy_len, 0.0);
        // The substituted closed form of `normal × base1`; length is |normal|.
        let base2 = Vec3::new(
            nx * nz * inv_xy_len,
            ny * nz * inv_xy_len,
            -xy_proj_sq.sqrt(),
        );
        (base1, base2)
    } else {
        // Projection vanishes: `normal` points (nearly) along ±z.
        let base1 = Vec3::new(if nz < 0.0 { -1.0 } else { 1.0 }, 0.0, 0.0);
        let base2 = Vec3::new(0.0, 1.0, 0.0);
        (base1, base2)
    }
}

/// Reflect `n` across the plane whose normal is `dir` (a unit chord direction):
/// `n − 2·(n·dir)·dir`. Used by §9.1.2 to build each one-sided arc's far-end
/// normal.
fn reflect(n: Vec3, dir: Vec3) -> Vec3 {
    n - 2.0 * n.dot(dir) * dir
}

/// §9.1.2 degeneracy guard: a (unit) normal coincides with its reflection when
/// it already lies in the chord-perpendicular plane — reflecting changes
/// nothing, so the arc has no bow on that side and collapses to the chord. The
/// test is a squared-distance threshold of `1e-5` between the normal and its
/// reflection.
fn normal_coincides_with_reflection(normal: Vec3, reflected: Vec3) -> bool {
    (normal - reflected).length_squared() < 1.0e-5
}

/// §9.1.2 — a point on the single circular arc at parameter `t`. The arc passes
/// through `start` and `end` and meets the surface tangentially at each end
/// (its radius there is perpendicular to that end's respected normal). The
/// radius *direction* is swept by great-arc (slerp) interpolation; the radius
/// *length* is the straight linear blend `lerp(radius_at_start, radius_at_end,
/// t)`. Degeneracies (normals parallel to the chord, no finite centre, or
/// diametrically opposite radii) fall back to the straight chord lerp.
fn arc_point(
    start: Vec3,
    normal_at_start: Vec3,
    end: Vec3,
    normal_at_end: Vec3,
    chord_dir: Vec3,
    t: f32,
) -> Vec3 {
    // Arc plane: perpendicular to the chord and to the shared "up". The average
    // here is one endpoint's normal with that same normal's reflection (per the
    // dual-blend caller), so it is exactly the chord-perpendicular part of that
    // one endpoint's normal.
    let average_normal = (normal_at_start + normal_at_end).normalize_or_zero();
    let arc_plane_normal = average_normal.cross(chord_dir);
    if arc_plane_normal.length_squared() < SMOOTH_DIR_EPS {
        return start.lerp(end, t);
    }
    let arc_plane_normal = arc_plane_normal.normalize();

    // Centre = the point equidistant from the endpoints whose radius to each end
    // is perpendicular to that end's normal, confined to the arc plane. Three
    // planes intersect to fix it:
    //   - radius at `start` ⊥ normal_at_start  (in-plane part of that normal)
    //   - radius at `end`   ⊥ normal_at_end    (in-plane part of that normal)
    //   - centre lies in the arc plane itself.
    let in_plane_start = project_out(arc_plane_normal.cross(normal_at_start), arc_plane_normal);
    let in_plane_end = project_out(arc_plane_normal.cross(normal_at_end), arc_plane_normal);
    let arc_center = match intersect_three_planes(
        (start, in_plane_start),
        (end, in_plane_end),
        (start, arc_plane_normal),
    ) {
        Some(c) => c,
        None => return start.lerp(end, t),
    };

    let radius_vec_start = start - arc_center;
    let radius_vec_end = end - arc_center;
    let radius_at_start = radius_vec_start.length();
    let radius_at_end = radius_vec_end.length();
    if radius_at_start < SMOOTH_DIR_EPS || radius_at_end < SMOOTH_DIR_EPS {
        return start.lerp(end, t);
    }
    let unit_radius_start = radius_vec_start / radius_at_start;
    let unit_radius_end = radius_vec_end / radius_at_end;

    // Sweep DIRECTION by great-arc interpolation; blend RADIUS LINEARLY.
    match slerp_unit(unit_radius_start, unit_radius_end, t) {
        Some(swept_direction) => {
            let radius = radius_at_start + (radius_at_end - radius_at_start) * t;
            arc_center + swept_direction * radius
        }
        // Diametrically opposite radii: no unique sweep, go straight.
        None => start.lerp(end, t),
    }
}

/// Remove the `axis`-aligned component of `v` (project `v` onto the plane whose
/// normal is the unit-ish `axis`).
fn project_out(v: Vec3, axis: Vec3) -> Vec3 {
    let axis2 = axis.length_squared();
    if axis2 < SMOOTH_DIR_EPS {
        return v;
    }
    v - axis * (v.dot(axis) / axis2)
}

/// Intersect three planes, each given as `(point_on_plane, plane_normal)`.
/// Returns the unique intersection point, or `None` when the normals are
/// near-coplanar (degenerate system).
fn intersect_three_planes(
    plane_a: (Vec3, Vec3),
    plane_b: (Vec3, Vec3),
    plane_c: (Vec3, Vec3),
) -> Option<Vec3> {
    let (pa, na) = plane_a;
    let (pb, nb) = plane_b;
    let (pc, nc) = plane_c;
    // Plane i: n_i · x = n_i · p_i. Solve [n_a; n_b; n_c] x = d by Cramer.
    let da = na.dot(pa);
    let db = nb.dot(pb);
    let dc = nc.dot(pc);
    let denom = na.dot(nb.cross(nc));
    if denom.abs() < SMOOTH_DIR_EPS {
        return None;
    }
    let numerator = da * nb.cross(nc) + db * nc.cross(na) + dc * na.cross(nb);
    Some(numerator / denom)
}

/// Great-arc (spherical linear) interpolation between two unit vectors. Returns
/// `None` when the two are diametrically opposite (no unique shortest arc).
fn slerp_unit(from: Vec3, to: Vec3, t: f32) -> Option<Vec3> {
    let cosine = from.dot(to).clamp(-1.0, 1.0);
    if cosine < -1.0 + 1.0e-6 {
        return None;
    }
    if cosine > 1.0 - 1.0e-6 {
        // Nearly identical: plain lerp avoids the small-angle division blow-up.
        return Some((from + (to - from) * t).normalize_or_zero());
    }
    let angle = cosine.acos();
    let sin_angle = angle.sin();
    let w_from = ((1.0 - t) * angle).sin() / sin_angle;
    let w_to = (t * angle).sin() / sin_angle;
    Some((from * w_from + to * w_to).normalize_or_zero())
}

/// §9.1.3 step 2 shell factor: `1 / cos(half-angle between na and nb)`. `1.0`
/// for parallel normals (a flat edge — `use_smooth_even` is a no-op there),
/// growing as the normals diverge so the bulge keeps a consistent *height*.
fn shell_factor(na: Vec3, nb: Vec3) -> f32 {
    let a = na.normalize_or_zero();
    let b = nb.normalize_or_zero();
    if a.length_squared() < 0.5 || b.length_squared() < 0.5 {
        return 1.0;
    }
    // The averaged endpoint normal foreshortens by cos(half-angle); the shell
    // factor is its reciprocal. half = normalize((a + b)/2); cos = a·half.
    let half = (a + b).normalize_or_zero();
    let cosine = a.dot(half);
    if cosine.abs() < 1.0e-6 {
        1.0
    } else {
        1.0 / cosine.abs()
    }
}

#[cfg(test)]
mod tests {
    use glam::Vec3;

    use super::*;
    use crate::loopmesh::test_support::create_face_from_verts;

    fn unit_quad(mesh: &mut LoopMesh) -> FaceKey {
        create_face_from_verts(
            mesh,
            &[
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(2.0, 0.0, 0.0),
                Vec3::new(2.0, 2.0, 0.0),
                Vec3::new(0.0, 2.0, 0.0),
            ],
        )
    }

    fn unit_tri(mesh: &mut LoopMesh) -> FaceKey {
        create_face_from_verts(
            mesh,
            &[
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(2.0, 0.0, 0.0),
                Vec3::new(0.0, 2.0, 0.0),
            ],
        )
    }

    fn counts(mesh: &LoopMesh) -> (usize, usize, usize) {
        (mesh.vertex_count(), mesh.edge_count(), mesh.face_count())
    }

    /// The actual re-split *fragment* faces in the mesh. §1's `geom.out.faces`
    /// reports only the one re-split-marked original per re-split face (not the
    /// fragments), so a test that checks fragment shapes must walk the topology.
    /// These fixtures start from a single isolated face, so after the op every
    /// face in the mesh is one of that face's fragments.
    fn resplit_fragments(mesh: &LoopMesh, _geom: &SubdivideGeom) -> Vec<FaceKey> {
        mesh.face_keys()
    }

    /// §7.0 / §7.1 — a quad `(a, b, c, d)` and a pentagon `(a, b, c, e, f)`
    /// sharing **two** edges (`{a,b}` and `{b,c}`), closing around the valence-2
    /// vertex `b` like the corner of a pillow. Edge- and vertex-manifold, and the
    /// one shape where a template connect's endpoint pair is carried by a face
    /// *other* than the one the template was dispatched for. Returns
    /// `(mesh, [a, b, c, d, e, f], e_ab, e_bc)`.
    fn quad_pentagon_shared_two_edges() -> (LoopMesh, [VertexKey; 6], EdgeKey, EdgeKey) {
        let mut mesh = LoopMesh::new();
        let a = mesh.create_vertex(Vec3::new(0.0, 0.0, 0.0));
        let b = mesh.create_vertex(Vec3::new(1.0, 0.0, 0.0));
        let c = mesh.create_vertex(Vec3::new(1.0, 1.0, 0.0));
        let d = mesh.create_vertex(Vec3::new(0.0, 1.0, 0.0));
        let e = mesh.create_vertex(Vec3::new(1.5, 0.5, 1.0));
        let f = mesh.create_vertex(Vec3::new(-0.5, -0.5, 1.0));

        let e_ab = mesh.create_edge([a, b], EdgeCreateFlags::NONE);
        let e_bc = mesh.create_edge([b, c], EdgeCreateFlags::NONE);
        let e_cd = mesh.create_edge([c, d], EdgeCreateFlags::NONE);
        let e_da = mesh.create_edge([d, a], EdgeCreateFlags::NONE);
        let e_af = mesh.create_edge([a, f], EdgeCreateFlags::NONE);
        let e_fe = mesh.create_edge([f, e], EdgeCreateFlags::NONE);
        let e_ec = mesh.create_edge([e, c], EdgeCreateFlags::NONE);

        mesh.face_create(
            &[a, b, c, d],
            &[e_ab, e_bc, e_cd, e_da],
            FaceCreateFlags::NONE,
        );
        // Wound c -> b -> a so both shared edges are traversed opposite to the
        // quad, keeping every radial cycle at length 2.
        mesh.face_create(
            &[c, b, a, f, e],
            &[e_bc, e_ab, e_af, e_fe, e_ec],
            FaceCreateFlags::NONE,
        );
        (mesh, [a, b, c, d, e, f], e_ab, e_bc)
    }

    /// §7.0 / §7.1 — a quad `(a, b, c, d)` and a triangle `(a, b, c)` sharing
    /// **two** edges (`{a,b}` and `{b,c}`). Like the pillow above, but the
    /// neighbour is *strictly smaller* than the parent on both sides, so the
    /// fewest-corners rule would pull each face's shared-corner cut into the
    /// other one — which is exactly what the §7.1 fallback rows must not do.
    /// Returns `(mesh, [a, b, c, d], e_ab, e_bc)`.
    fn quad_triangle_shared_two_edges() -> (LoopMesh, [VertexKey; 4], EdgeKey, EdgeKey) {
        let mut mesh = LoopMesh::new();
        let a = mesh.create_vertex(Vec3::new(0.0, 0.0, 0.0));
        let b = mesh.create_vertex(Vec3::new(1.0, 0.0, 0.0));
        let c = mesh.create_vertex(Vec3::new(1.0, 1.0, 0.0));
        let d = mesh.create_vertex(Vec3::new(0.0, 1.0, 0.0));

        let e_ab = mesh.create_edge([a, b], EdgeCreateFlags::NONE);
        let e_bc = mesh.create_edge([b, c], EdgeCreateFlags::NONE);
        let e_cd = mesh.create_edge([c, d], EdgeCreateFlags::NONE);
        let e_da = mesh.create_edge([d, a], EdgeCreateFlags::NONE);
        let e_ca = mesh.create_edge([c, a], EdgeCreateFlags::NONE);

        mesh.face_create(
            &[a, b, c, d],
            &[e_ab, e_bc, e_cd, e_da],
            FaceCreateFlags::NONE,
        );
        // Wound c -> b -> a so both shared edges are traversed opposite to the
        // quad, keeping every radial cycle at length 2.
        mesh.face_create(&[c, b, a], &[e_bc, e_ab, e_ca], FaceCreateFlags::NONE);
        (mesh, [a, b, c, d], e_ab, e_bc)
    }

    /// The same pillow, built in the **other face order**: the triangle is face 0
    /// and the quad face 1. Identical geometry, identical winding, identical
    /// shared edges — only the creation order differs, which (pass three being the
    /// reverse of mesh face order, §7.0.3(a)) makes the *quad* apply first and the
    /// *triangle* the pending face whose ring can be carved out from under it.
    /// Returns `(mesh, [a, b, c, d], e_ab, e_bc)`.
    fn triangle_quad_shared_two_edges() -> (LoopMesh, [VertexKey; 4], EdgeKey, EdgeKey) {
        let mut mesh = LoopMesh::new();
        let a = mesh.create_vertex(Vec3::new(0.0, 0.0, 0.0));
        let b = mesh.create_vertex(Vec3::new(1.0, 0.0, 0.0));
        let c = mesh.create_vertex(Vec3::new(1.0, 1.0, 0.0));
        let d = mesh.create_vertex(Vec3::new(0.0, 1.0, 0.0));

        let e_ab = mesh.create_edge([a, b], EdgeCreateFlags::NONE);
        let e_bc = mesh.create_edge([b, c], EdgeCreateFlags::NONE);
        let e_cd = mesh.create_edge([c, d], EdgeCreateFlags::NONE);
        let e_da = mesh.create_edge([d, a], EdgeCreateFlags::NONE);
        let e_ca = mesh.create_edge([c, a], EdgeCreateFlags::NONE);

        mesh.face_create(&[c, b, a], &[e_bc, e_ab, e_ca], FaceCreateFlags::NONE);
        mesh.face_create(
            &[a, b, c, d],
            &[e_ab, e_bc, e_cd, e_da],
            FaceCreateFlags::NONE,
        );
        (mesh, [a, b, c, d], e_ab, e_bc)
    }

    /// §7.0.2 / §5.4 — a quad `A = (v0, v1, v2, v3)` (face 0) and a second quad
    /// `B = (v0, v1, v3, v2)` (face 1) over the **same four verts**, sharing
    /// exactly the two edges `{v0,v1}` and `{v2,v3}` — which are *opposite* in
    /// both rings — and joining them the other way round through two private
    /// edges of its own. Edge-manifold throughout (every radial cycle is 2).
    ///
    /// The shape exists to give a §5.4 grid fill a neighbour that lays a **direct**
    /// `m01 - m23` chord: `B`'s two cut edges are opposite, so it takes the §5.2
    /// loop-cut fallback and splits itself along exactly the pair `A`'s bar will
    /// nominate — leaving two 4-ring fragments that hold that pair *adjacent*.
    /// Returns `(mesh, [v0, v1, v2, v3])`.
    fn quad_quad_shared_two_opposite_edges() -> (LoopMesh, [VertexKey; 4]) {
        let mut mesh = LoopMesh::new();
        let v0 = mesh.create_vertex(Vec3::new(0.0, 0.0, 0.0));
        let v1 = mesh.create_vertex(Vec3::new(2.0, 0.0, 0.0));
        let v2 = mesh.create_vertex(Vec3::new(2.0, 2.0, 0.0));
        let v3 = mesh.create_vertex(Vec3::new(0.0, 2.0, 0.0));

        let e01 = mesh.create_edge([v0, v1], EdgeCreateFlags::NONE);
        let e12 = mesh.create_edge([v1, v2], EdgeCreateFlags::NONE);
        let e23 = mesh.create_edge([v2, v3], EdgeCreateFlags::NONE);
        let e30 = mesh.create_edge([v3, v0], EdgeCreateFlags::NONE);
        let e13 = mesh.create_edge([v1, v3], EdgeCreateFlags::NONE);
        let e20 = mesh.create_edge([v2, v0], EdgeCreateFlags::NONE);

        mesh.face_create(
            &[v0, v1, v2, v3],
            &[e01, e12, e23, e30],
            FaceCreateFlags::NONE,
        );
        mesh.face_create(
            &[v0, v1, v3, v2],
            &[e01, e13, e23, e20],
            FaceCreateFlags::NONE,
        );
        (mesh, [v0, v1, v2, v3])
    }

    /// §7.0.3 — the same pillow corner with a **quad** in place of the triangle,
    /// so the two faces have *equal* corner counts and a template connect they
    /// both carry falls to the tie-break. `A = (v0, v1, v2, v3)` is face 0,
    /// `B = (v1, v0, v4, v2)` face 1; they share `{v0,v1}` and `{v1,v2}`, and `v1`
    /// is an interior vertex of valence 2. Returns `(mesh, [v0..v4], e01)`.
    fn quad_quad_shared_two_edges() -> (LoopMesh, [VertexKey; 5], EdgeKey) {
        let mut mesh = LoopMesh::new();
        let v0 = mesh.create_vertex(Vec3::new(0.0, 1.0, 0.0));
        let v1 = mesh.create_vertex(Vec3::new(0.2, 0.0, 0.0));
        let v2 = mesh.create_vertex(Vec3::new(0.0, -1.0, 0.0));
        let v3 = mesh.create_vertex(Vec3::new(-1.0, 0.0, 0.0));
        let v4 = mesh.create_vertex(Vec3::new(1.0, 0.0, 0.0));

        let e01 = mesh.create_edge([v0, v1], EdgeCreateFlags::NONE);
        let e12 = mesh.create_edge([v1, v2], EdgeCreateFlags::NONE);
        let e23 = mesh.create_edge([v2, v3], EdgeCreateFlags::NONE);
        let e30 = mesh.create_edge([v3, v0], EdgeCreateFlags::NONE);
        let e04 = mesh.create_edge([v0, v4], EdgeCreateFlags::NONE);
        let e42 = mesh.create_edge([v4, v2], EdgeCreateFlags::NONE);

        mesh.face_create(
            &[v0, v1, v2, v3],
            &[e01, e12, e23, e30],
            FaceCreateFlags::NONE,
        );
        // Wound v1 -> v0 so both shared edges are traversed opposite to A's,
        // keeping every radial cycle at length 2.
        mesh.face_create(
            &[v1, v0, v4, v2],
            &[e01, e04, e42, e12],
            FaceCreateFlags::NONE,
        );
        (mesh, [v0, v1, v2, v3, v4], e01)
    }

    /// A dihedral fold of two quads hinged about the edge `{v1, v2}` (§9.1.5).
    /// Quad A lies flat in `z = 0`; quad B folds up by `angle` about the hinge.
    /// The hinge verts `v1`/`v2` get bisector normals (divergent from the flat
    /// face normal), so the edge `{v0, v1}` — flat endpoint `v0`, hinge endpoint
    /// `v1` — has the divergent endpoint normals §9.1 needs to bow a midpoint
    /// off the chord. Returns `(mesh, [v0,v1,v2,v3,v4,v5], e01)`.
    fn folded_two_quads(angle: f32) -> (LoopMesh, [VertexKey; 6], EdgeKey) {
        let mut mesh = LoopMesh::new();
        let (c, s) = (angle.cos(), angle.sin());
        let v0 = mesh.create_vertex(Vec3::new(0.0, 0.0, 0.0));
        let v1 = mesh.create_vertex(Vec3::new(1.0, 0.0, 0.0));
        let v2 = mesh.create_vertex(Vec3::new(1.0, 1.0, 0.0));
        let v3 = mesh.create_vertex(Vec3::new(0.0, 1.0, 0.0));
        // Quad B folds about the hinge {v1, v2} (the y-axis line at x = 1).
        let v4 = mesh.create_vertex(Vec3::new(1.0 + c, 0.0, s));
        let v5 = mesh.create_vertex(Vec3::new(1.0 + c, 1.0, s));

        let e01 = mesh.create_edge([v0, v1], EdgeCreateFlags::NONE);
        let e12 = mesh.create_edge([v1, v2], EdgeCreateFlags::NONE);
        let e23 = mesh.create_edge([v2, v3], EdgeCreateFlags::NONE);
        let e30 = mesh.create_edge([v3, v0], EdgeCreateFlags::NONE);
        let e14 = mesh.create_edge([v1, v4], EdgeCreateFlags::NONE);
        let e45 = mesh.create_edge([v4, v5], EdgeCreateFlags::NONE);
        let e52 = mesh.create_edge([v5, v2], EdgeCreateFlags::NONE);

        mesh.face_create(
            &[v0, v1, v2, v3],
            &[e01, e12, e23, e30],
            FaceCreateFlags::NONE,
        );
        mesh.face_create(
            &[v1, v4, v5, v2],
            &[e14, e45, e52, e12],
            FaceCreateFlags::NONE,
        );
        (mesh, [v0, v1, v2, v3, v4, v5], e01)
    }

    fn midpoint_pos_after_cut(mesh: &LoopMesh, geom: &SubdivideGeom) -> Vec3 {
        // The single cut edge inserts exactly one midpoint; it is the first
        // vert recorded in geom.verts (§3 phase one runs before the re-split).
        mesh.vertex_position(geom.verts[0])
    }

    #[test]
    fn quad_single_edge_three_fragments() {
        // §5.1 — V 4->5, E 4->7, 1 quad -> 3 faces.
        let mut mesh = LoopMesh::new();
        let face = unit_quad(&mut mesh);
        let fv = mesh.face_vert_keys(face);
        let e01 = mesh.query_edge_exists(fv[0], fv[1]).unwrap();

        let res = mesh.subdivide_edges(&[e01], SubdivideParams::default());
        assert!(matches!(res, SubdivideResult::Ran(_)));

        let (v, e, f) = counts(&mesh);
        assert_eq!((v, e, f), (5, 7, 3), "quad one-cut deltas");
    }

    #[test]
    fn quad_all_four_edges_plus_split() {
        // §5.4 — V 4->9 (4 rim mids + 1 centre), E 4->12, 1 quad -> 4 quads.
        let mut mesh = LoopMesh::new();
        let face = unit_quad(&mut mesh);
        let fv = mesh.face_vert_keys(face);
        let edges: Vec<_> = (0..4)
            .map(|i| mesh.query_edge_exists(fv[i], fv[(i + 1) % 4]).unwrap())
            .collect();

        let res = mesh.subdivide_edges(&edges, SubdivideParams::default());
        let geom = match res {
            SubdivideResult::Ran(g) => g,
            _ => panic!("expected Ran"),
        };

        let (v, e, f) = counts(&mesh);
        assert_eq!((v, e, f), (9, 12, 4), "quad four-cut deltas");

        // The centre vert lands at the quad centroid (1, 1, 0).
        let centre = *geom.verts.last().unwrap();
        assert!(
            mesh.vertex_position(centre)
                .distance(Vec3::new(1.0, 1.0, 0.0))
                < 1e-5,
            "centre vert at centroid"
        );
        // Every resulting face is a quad.
        for f in &geom.faces {
            assert_eq!(mesh.face_len(*f), 4, "plus-split yields quads");
        }
    }

    #[test]
    fn tri_single_edge_two_tris() {
        // §6.1 — V 3->4, E 3->5, 1 tri -> 2 tris.
        let mut mesh = LoopMesh::new();
        let face = unit_tri(&mut mesh);
        let fv = mesh.face_vert_keys(face);
        let e01 = mesh.query_edge_exists(fv[0], fv[1]).unwrap();

        mesh.subdivide_edges(&[e01], SubdivideParams::default());

        let (v, e, f) = counts(&mesh);
        assert_eq!((v, e, f), (4, 5, 2), "tri one-cut deltas");
    }

    #[test]
    fn tri_three_edges_one_to_four() {
        // §6.3 — V 3->6, E 3->9, 1 tri -> 4 tris.
        let mut mesh = LoopMesh::new();
        let face = unit_tri(&mut mesh);
        let fv = mesh.face_vert_keys(face);
        let edges: Vec<_> = (0..3)
            .map(|i| mesh.query_edge_exists(fv[i], fv[(i + 1) % 3]).unwrap())
            .collect();

        let res = mesh.subdivide_edges(&edges, SubdivideParams::default());
        let geom = match res {
            SubdivideResult::Ran(g) => g,
            _ => panic!("expected Ran"),
        };

        let (v, e, f) = counts(&mesh);
        assert_eq!((v, e, f), (6, 9, 4), "tri three-cut deltas");
        for f in &geom.faces {
            assert_eq!(mesh.face_len(*f), 3, "1-to-4 yields triangles");
        }
    }

    #[test]
    fn shared_edge_crack_free() {
        // §2 / §4 — two quads sharing edge {v1,v2}; cut the shared edge only.
        // The midpoint is created once (V 6->7), each quad re-splits via §5.1
        // into 3 fragments (2 faces -> 6), and they stay welded across the
        // shared midpoint. Edge delta: 7 -> 12 (one halved shared edge +1, and
        // 2 interior connects per quad = +4).
        let mut mesh = LoopMesh::new();
        let v0 = mesh.create_vertex(Vec3::new(0.0, 0.0, 0.0));
        let v1 = mesh.create_vertex(Vec3::new(1.0, 0.0, 0.0));
        let v2 = mesh.create_vertex(Vec3::new(1.0, 1.0, 0.0));
        let v3 = mesh.create_vertex(Vec3::new(0.0, 1.0, 0.0));
        let v4 = mesh.create_vertex(Vec3::new(2.0, 0.0, 0.0));
        let v5 = mesh.create_vertex(Vec3::new(2.0, 1.0, 0.0));

        let e01 = mesh.create_edge([v0, v1], EdgeCreateFlags::NONE);
        let e12 = mesh.create_edge([v1, v2], EdgeCreateFlags::NONE);
        let e23 = mesh.create_edge([v2, v3], EdgeCreateFlags::NONE);
        let e30 = mesh.create_edge([v3, v0], EdgeCreateFlags::NONE);
        let e14 = mesh.create_edge([v1, v4], EdgeCreateFlags::NONE);
        let e45 = mesh.create_edge([v4, v5], EdgeCreateFlags::NONE);
        let e52 = mesh.create_edge([v5, v2], EdgeCreateFlags::NONE);

        mesh.face_create(
            &[v0, v1, v2, v3],
            &[e01, e12, e23, e30],
            FaceCreateFlags::NONE,
        );
        mesh.face_create(
            &[v1, v4, v5, v2],
            &[e14, e45, e52, e12],
            FaceCreateFlags::NONE,
        );

        let (v, e, f) = counts(&mesh);
        assert_eq!((v, e, f), (6, 7, 2), "starting two-quad fixture");

        mesh.subdivide_edges(&[e12], SubdivideParams::default());

        let (v, e, f) = counts(&mesh);
        assert_eq!(v, 7, "shared midpoint created once");
        assert_eq!(f, 6, "each quad -> 3 fragments");
        assert_eq!(e, 12, "shared halved + 2 interior connects per quad");

        // Crack-free: exactly one vertex sits at the shared-edge midpoint, and
        // it is shared by both fans. Its disk is degree 6: the two halves of the
        // cut shared edge, plus the two §5.1 interior connects each quad draws
        // from the midpoint to its two opposite corners.
        let mid = mesh
            .vert_keys()
            .into_iter()
            .find(|&vk| mesh.vertex_position(vk).distance(Vec3::new(1.0, 0.5, 0.0)) < 1e-5)
            .expect("shared midpoint exists");
        assert_eq!(
            mesh.vertex_disk_count_edges(mid),
            6,
            "shared midpoint welds both quads"
        );
    }

    #[test]
    fn quad_two_opposite_edges_loop_cut() {
        // §5.2 opposite — V 4->6 (2 rim midpoints), E 4->7 (2 halved + 1
        // connecting cut), 1 quad -> 2 quads.
        let mut mesh = LoopMesh::new();
        let face = unit_quad(&mut mesh);
        let fv = mesh.face_vert_keys(face);
        let e01 = mesh.query_edge_exists(fv[0], fv[1]).unwrap();
        let e23 = mesh.query_edge_exists(fv[2], fv[3]).unwrap();

        let res = mesh.subdivide_edges(&[e01, e23], SubdivideParams::default());
        let geom = match res {
            SubdivideResult::Ran(g) => g,
            _ => panic!("expected Ran"),
        };

        let (v, e, f) = counts(&mesh);
        assert_eq!((v, e, f), (6, 7, 2), "quad opposite two-cut deltas");

        // Both resulting faces are quads (a straight loop cut, no triangles).
        let frags = resplit_fragments(&mesh, &geom);
        assert_eq!(frags.len(), 2, "loop cut yields two fragments");
        for f in &frags {
            assert_eq!(mesh.face_len(*f), 4, "loop cut yields quads");
        }
        // geom.out reports a single re-split mark, not both fragments (§1).
        assert_eq!(geom.faces.len(), 1, "geom.out: one re-split mark");

        // The two rim midpoints land at the edge midpoints, and a connecting
        // edge joins them straight across.
        let m01 = geom.verts[0];
        let m23 = geom.verts[1];
        assert!(
            mesh.vertex_position(m01).distance(Vec3::new(1.0, 0.0, 0.0)) < 1e-5,
            "m01 at midpoint of bottom edge"
        );
        assert!(
            mesh.vertex_position(m23).distance(Vec3::new(1.0, 2.0, 0.0)) < 1e-5,
            "m23 at midpoint of top edge"
        );
        assert!(
            mesh.query_edge_exists(m01, m23).is_some(),
            "connecting cut joins the two opposite midpoints"
        );
    }

    #[test]
    fn quad_two_adjacent_edges_straight_cut() {
        // §5.2.1 adjacent STRAIGHT_CUT (default): the two cut edges share
        // corner v1. After inserting both rim midpoints, a single m01->m12
        // corner cut carves the shared corner off as a triangle and leaves a
        // pentagon. V 4->6 (2 rim midpoints), E 4->7 (2 halved + 1 corner cut),
        // 1 quad -> 2 fragments.
        let mut mesh = LoopMesh::new();
        let face = unit_quad(&mut mesh);
        let fv = mesh.face_vert_keys(face);
        let [v0, v1, v2, _v3] = [fv[0], fv[1], fv[2], fv[3]];
        let e01 = mesh.query_edge_exists(v0, v1).unwrap();
        let e12 = mesh.query_edge_exists(v1, v2).unwrap();

        let res = mesh.subdivide_edges(&[e01, e12], SubdivideParams::default());
        let geom = match res {
            SubdivideResult::Ran(g) => g,
            _ => panic!("expected Ran"),
        };

        let (v, e, f) = counts(&mesh);
        assert_eq!(
            (v, e, f),
            (6, 7, 2),
            "adjacent two-cut STRAIGHT_CUT: +2V / +3E / +1F"
        );

        // The two rim midpoints land at the cut-edge midpoints.
        let m01 = mesh
            .vert_keys()
            .into_iter()
            .find(|&vk| mesh.vertex_position(vk).distance(Vec3::new(1.0, 0.0, 0.0)) < 1e-5)
            .expect("m01 at bottom-edge midpoint");
        let m12 = mesh
            .vert_keys()
            .into_iter()
            .find(|&vk| mesh.vertex_position(vk).distance(Vec3::new(2.0, 1.0, 0.0)) < 1e-5)
            .expect("m12 at right-edge midpoint");

        // The connecting cut joins the two adjacent-edge midpoints.
        assert!(
            mesh.query_edge_exists(m01, m12).is_some(),
            "connecting cut joins m01 and m12"
        );

        // Exactly one triangle and one pentagon fragment.
        let frags = resplit_fragments(&mesh, &geom);
        let mut lens: Vec<usize> = frags.iter().map(|&f| mesh.face_len(f)).collect();
        lens.sort_unstable();
        assert_eq!(lens, vec![3, 5], "1 triangle + 1 pentagon");

        // The triangle is the shared-corner cut-off {m01, v1, m12}; v1 is on it.
        let tri = frags
            .iter()
            .copied()
            .find(|&f| mesh.face_len(f) == 3)
            .unwrap();
        let tri_verts = mesh.face_vert_keys(tri);
        assert!(
            tri_verts.contains(&v1),
            "triangle holds the shared corner v1"
        );
        assert!(tri_verts.contains(&m01) && tri_verts.contains(&m12));

        // The pentagon does *not* contain the shared corner v1.
        let pent = frags
            .iter()
            .copied()
            .find(|&f| mesh.face_len(f) == 5)
            .unwrap();
        assert!(
            !mesh.face_vert_keys(pent).contains(&v1),
            "pentagon excludes the shared corner v1"
        );
    }

    /// A quad with a **straight** (180°) rim corner at `v1`: `(0,0)`, `(1,0)`,
    /// `(2,0)`, `(1,1)`. The two edges flanking `v1` are collinear, so cutting
    /// both takes the §4.3.1 flat-corner guard's branch.
    fn flat_corner_quad(mesh: &mut LoopMesh) -> FaceKey {
        create_face_from_verts(
            mesh,
            &[
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(2.0, 0.0, 0.0),
                Vec3::new(1.0, 1.0, 0.0),
            ],
        )
    }

    #[test]
    fn flat_corner_guard_suppresses_two_adjacent_fallback() {
        // §4.3.1 — the two cut edges share the straight corner v1 and are
        // collinear there, so the STRAIGHT_CUT fallback is suppressed: the two
        // midpoints land on the rim (§3) but no corner cut is laid. One
        // flat-cornered hexagon: V 6 / E 6 / F 1, and no re-split mark.
        let mut mesh = LoopMesh::new();
        let face = flat_corner_quad(&mut mesh);
        let fv = mesh.face_vert_keys(face);
        let [v0, v1, v2] = [fv[0], fv[1], fv[2]];
        let e01 = mesh.query_edge_exists(v0, v1).unwrap();
        let e12 = mesh.query_edge_exists(v1, v2).unwrap();

        let res = mesh.subdivide_edges(&[e01, e12], SubdivideParams::default());
        let geom = match res {
            SubdivideResult::Ran(g) => g,
            _ => panic!("expected Ran"),
        };

        assert_eq!(
            counts(&mesh),
            (6, 6, 1),
            "flat corner: midpoints inserted, face left flat"
        );
        let m01 = find_vert_at(&mesh, Vec3::new(0.5, 0.0, 0.0));
        let m12 = find_vert_at(&mesh, Vec3::new(1.5, 0.0, 0.0));
        assert!(
            mesh.query_edge_exists(m01, m12).is_none(),
            "no connecting cut across the straight corner"
        );
        assert!(
            geom.geom().faces.is_empty(),
            "a guarded face contributes no re-split mark"
        );
    }

    #[test]
    fn flat_corner_guard_does_not_bite_opposite_cut_edges() {
        // §4.3.1 contrast — the same fixture with two *opposite* edges cut. They
        // share no vertex, so the guard cannot apply and the strip fires:
        // V 6 / E 7 / F 2.
        let mut mesh = LoopMesh::new();
        let face = flat_corner_quad(&mut mesh);
        let fv = mesh.face_vert_keys(face);
        let [v0, v1, v2, v3] = [fv[0], fv[1], fv[2], fv[3]];
        let e01 = mesh.query_edge_exists(v0, v1).unwrap();
        let e23 = mesh.query_edge_exists(v2, v3).unwrap();

        let res = mesh.subdivide_edges(&[e01, e23], SubdivideParams::default());
        assert!(matches!(res, SubdivideResult::Ran(_)));
        assert_eq!(
            counts(&mesh),
            (6, 7, 2),
            "opposite cut edges share no vertex: the strip fires"
        );
    }

    #[test]
    fn flat_corner_guard_spares_the_corner_templates() {
        // §4.3.1 — the guard is consulted only on the fallback path. The same
        // flat-cornered quad under INNERVERT matches a template and re-splits
        // normally: V 7 / E 9 / F 3, as on a square quad.
        let mut mesh = LoopMesh::new();
        let face = flat_corner_quad(&mut mesh);
        let fv = mesh.face_vert_keys(face);
        let e01 = mesh.query_edge_exists(fv[0], fv[1]).unwrap();
        let e12 = mesh.query_edge_exists(fv[1], fv[2]).unwrap();

        let res = mesh.subdivide_edges(
            &[e01, e12],
            SubdivideParams::default().quad_corner_type(QuadCornerType::InnerVert),
        );
        assert!(matches!(res, SubdivideResult::Ran(_)));
        assert_eq!(
            counts(&mesh),
            (7, 9, 3),
            "a template row re-splits across a straight corner"
        );
    }

    #[test]
    fn flat_corner_guard_catches_the_spike_corner() {
        // §4.3.1 — the cosine is unsigned, so a ~0° spike corner (the two cut
        // edges folding back onto one another) is guarded exactly like the ~180°
        // straight corner. Here the corner at v1 folds back along the x axis.
        let mut mesh = LoopMesh::new();
        let face = create_face_from_verts(
            &mut mesh,
            &[
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(2.0, 0.0, 0.0),
                Vec3::new(0.5, 0.0, 0.0),
                Vec3::new(1.0, 1.0, 0.0),
            ],
        );
        let fv = mesh.face_vert_keys(face);
        let e01 = mesh.query_edge_exists(fv[0], fv[1]).unwrap();
        let e12 = mesh.query_edge_exists(fv[1], fv[2]).unwrap();

        let res = mesh.subdivide_edges(&[e01, e12], SubdivideParams::default());
        assert!(matches!(res, SubdivideResult::Ran(_)));
        assert_eq!(
            counts(&mesh),
            (6, 6, 1),
            "spike corner is guarded like the straight corner"
        );
    }

    /// A quad whose corner at `v1` deviates from straight by `off_straight`
    /// radians: `(0,0)`, `(1,0)`, `v1 + (cos θ, sin θ)`, `(1,1)`. At `θ = 0` this
    /// is exactly [`flat_corner_quad`]; the two edges flanking `v1` have an
    /// unsigned cosine of `cos θ`, so `θ` is precisely the guard's input.
    fn near_flat_corner_quad(mesh: &mut LoopMesh, off_straight: f32) -> FaceKey {
        create_face_from_verts(
            mesh,
            &[
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(1.0 + off_straight.cos(), off_straight.sin(), 0.0),
                Vec3::new(1.0, 1.0, 0.0),
            ],
        )
    }

    /// Cut the two edges flanking the near-straight corner `v1` and report the
    /// post-op counts.
    fn cut_near_flat_corner(off_straight_degrees: f32) -> (usize, usize, usize) {
        let mut mesh = LoopMesh::new();
        let face = near_flat_corner_quad(&mut mesh, off_straight_degrees.to_radians());
        let fv = mesh.face_vert_keys(face);
        let e01 = mesh.query_edge_exists(fv[0], fv[1]).unwrap();
        let e12 = mesh.query_edge_exists(fv[1], fv[2]).unwrap();
        mesh.subdivide_edges(&[e01, e12], SubdivideParams::default());
        counts(&mesh)
    }

    #[test]
    fn flat_corner_guard_tolerance_brackets_half_a_degree() {
        // §4.3.1 — the threshold is |cos θ| > 1 − 5·10⁻⁵, i.e. collinear to
        // within ~0.57°. Bracket it: a corner 0.1° off straight is *inside* the
        // tolerance and guarded (left flat, V 6 / E 6 / F 1); one 5° off is
        // outside it and re-splits through the STRAIGHT_CUT fallback
        // (V 6 / E 7 / F 2). A tolerance wrong by orders of magnitude — or one
        // stated in radians — fails one of the two.
        assert_eq!(
            cut_near_flat_corner(0.1),
            (6, 6, 1),
            "0.1° off straight is within the collinearity tolerance: guarded"
        );
        assert_eq!(
            cut_near_flat_corner(5.0),
            (6, 7, 2),
            "5° off straight is a real corner: the fallback fires"
        );
    }

    #[test]
    fn flat_corner_guard_ignores_a_zero_length_cut_edge() {
        // §4.3.1 — a zero-length cut edge has no direction, so no cosine can be
        // taken and the guard declines to fire (its early return). The face is
        // template-less with two cut edges, so the fallback runs as normal: the
        // corner cut lands and the quad becomes two fragments (V 6 / E 7 / F 2).
        // Here `v1` and `v2` are coincident, so `{v1, v2}` is degenerate.
        let mut mesh = LoopMesh::new();
        let face = create_face_from_verts(
            &mut mesh,
            &[
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(1.0, 1.0, 0.0),
            ],
        );
        let fv = mesh.face_vert_keys(face);
        let e01 = mesh.query_edge_exists(fv[0], fv[1]).unwrap();
        let e12 = mesh.query_edge_exists(fv[1], fv[2]).unwrap();

        let res = mesh.subdivide_edges(&[e01, e12], SubdivideParams::default());
        let geom = match res {
            SubdivideResult::Ran(g) => g,
            _ => panic!("expected Ran"),
        };
        assert_eq!(
            counts(&mesh),
            (6, 7, 2),
            "a degenerate cut edge is not collinear with anything: unguarded"
        );
        assert_eq!(
            geom.geom().faces.len(),
            1,
            "the fallback fired, so the face carries its re-split mark"
        );
    }

    #[test]
    fn flat_corner_guard_is_cuts_independent() {
        // §9.5.3.5 — the guard reads the *input* edge directions, so its verdict
        // is the same at every N: the flat-cornered quad under STRAIGHT_CUT is
        // left un-re-split at `cuts = 2` exactly as at `cuts = 1`. Rim: 4 corners
        // + 2 runs of 2 midpoints = V 8; each cut edge becomes 3 segments, so
        // E 8; the face is never re-split, so F 1 and no re-split mark.
        //
        // (The multi-cut two-edge strip that this suppresses is a separate parity
        // row and does not exist yet, so an *unguarded* two-adjacent quad also
        // comes out flat today — this pins the guard's verdict now so the strip
        // lands already-guarded.)
        let mut mesh = LoopMesh::new();
        let face = flat_corner_quad(&mut mesh);
        let fv = mesh.face_vert_keys(face);
        let e01 = mesh.query_edge_exists(fv[0], fv[1]).unwrap();
        let e12 = mesh.query_edge_exists(fv[1], fv[2]).unwrap();

        let res = mesh.subdivide_edges(&[e01, e12], SubdivideParams::default().cuts(2));
        let geom = match res {
            SubdivideResult::Ran(g) => g,
            _ => panic!("expected Ran"),
        };
        assert_eq!(
            counts(&mesh),
            (8, 8, 1),
            "guarded at cuts = 2, as at cuts = 1"
        );
        assert!(
            geom.geom().faces.is_empty(),
            "a guarded face contributes no re-split mark at any N"
        );
    }

    #[test]
    fn flat_corner_guard_spares_the_multicut_templates() {
        // §4.3.1 / §9.5.3.5 — threading the guard through the multi-cut dispatch
        // must not reach the template rows. The same flat-cornered quad with all
        // four edges cut at `cuts = 2` is a fully-cut face — a template, never
        // guarded — so it grid-fills as usual: a 4x4 lattice, V 16 / E 24 / F 9.
        let mut mesh = LoopMesh::new();
        let face = flat_corner_quad(&mut mesh);
        let edges = mesh.face_edge_keys(face);

        let res = mesh.subdivide_edges(&edges, SubdivideParams::default().cuts(2));
        assert!(matches!(res, SubdivideResult::Ran(_)));
        assert_eq!(
            counts(&mesh),
            (16, 24, 9),
            "a template row grid-fills across a straight corner at cuts = 2"
        );
    }

    /// Cut the two adjacent edges `{v0,v1}` + `{v1,v2}` of a fresh unit quad
    /// under the given corner strategy. Returns the mesh, the geom, and the four
    /// original corner keys `[v0, v1, v2, v3]`.
    fn adjacent_two_cut(corner_type: QuadCornerType) -> (LoopMesh, SubdivideGeom, [VertexKey; 4]) {
        let mut mesh = LoopMesh::new();
        let face = unit_quad(&mut mesh);
        let fv = mesh.face_vert_keys(face);
        let corners = [fv[0], fv[1], fv[2], fv[3]];
        let e01 = mesh.query_edge_exists(corners[0], corners[1]).unwrap();
        let e12 = mesh.query_edge_exists(corners[1], corners[2]).unwrap();
        let res = mesh.subdivide_edges(
            &[e01, e12],
            SubdivideParams::default().quad_corner_type(corner_type),
        );
        let geom = match res {
            SubdivideResult::Ran(g) => g,
            _ => panic!("expected Ran"),
        };
        (mesh, geom, corners)
    }

    fn find_vert_at(mesh: &LoopMesh, p: Vec3) -> VertexKey {
        mesh.vert_keys()
            .into_iter()
            .find(|&vk| mesh.vertex_position(vk).distance(p) < 1e-5)
            .unwrap_or_else(|| panic!("no vert at {p:?}"))
    }

    #[test]
    fn quad_two_adjacent_inner_vert_strategy() {
        // §5.2.2 INNERVERT — the m01->m12 corner cut, an inner vert c at its
        // midpoint, then c->v3. The only adjacent strategy with an inner vert.
        // V 7 / E 9 / F 3.
        let (mesh, geom, [_v0, _v1, _v2, _v3]) = adjacent_two_cut(QuadCornerType::InnerVert);
        let (v, e, f) = counts(&mesh);
        assert_eq!((v, e, f), (7, 9, 3), "INNERVERT: +3V / +5E / +2F");

        // The inner vert c is the geom's last reported vert (after the two rim
        // midpoints), and it sits at the midpoint of the m01-m12 cut: m01 =
        // (1,0), m12 = (2,1) => c = (1.5, 0.5).
        assert_eq!(geom.verts.len(), 3, "two rim midpoints + one inner vert");
        let c = *geom.verts.last().unwrap();
        assert!(
            mesh.vertex_position(c).distance(Vec3::new(1.5, 0.5, 0.0)) < 1e-5,
            "inner vert at the corner-cut midpoint (two-source blend of m01/m12)"
        );

        // Fragment shapes: one 4-gon (the shared-corner ring {m01,v1,m12,c})
        // plus two quads {v0,m01,c,v3} and {v3,c,m12,v2}.
        let frags = resplit_fragments(&mesh, &geom);
        let mut lens: Vec<usize> = frags.iter().map(|&f| mesh.face_len(f)).collect();
        lens.sort_unstable();
        assert_eq!(lens, vec![4, 4, 4], "4-gon + 2 quads");

        // c is spoked to v3: the c-v3 edge exists, and c has the two filling
        // quads plus the corner 4-gon around it (degree 3 in its disk: the
        // halved corner cut x2 + the c-v3 spoke).
        let v3 = find_vert_at(&mesh, Vec3::new(0.0, 2.0, 0.0));
        assert!(
            mesh.query_edge_exists(c, v3).is_some(),
            "c spokes to far corner v3"
        );
        assert_eq!(
            mesh.vertex_disk_count_edges(c),
            3,
            "c disk: two cut halves + spoke"
        );
    }

    #[test]
    fn quad_two_adjacent_path_strategy() {
        // §5.2.3 PATH — the m01->m12 corner cut chained with a v0->v2 second
        // cut. No inner vert. V 6 / E 8 / F 3: corner tri + central quad + far
        // tri.
        let (mesh, geom, [v0, v1, v2, v3]) = adjacent_two_cut(QuadCornerType::Path);
        let (v, e, f) = counts(&mesh);
        assert_eq!((v, e, f), (6, 8, 3), "PATH: +2V / +4E / +2F");
        assert_eq!(geom.verts.len(), 2, "two rim midpoints, no inner vert");

        let m01 = find_vert_at(&mesh, Vec3::new(1.0, 0.0, 0.0));
        let m12 = find_vert_at(&mesh, Vec3::new(2.0, 1.0, 0.0));
        assert!(
            mesh.query_edge_exists(m01, m12).is_some(),
            "m01-m12 corner cut"
        );
        assert!(
            mesh.query_edge_exists(v0, v2).is_some(),
            "v0-v2 chained cut"
        );

        // Fragment shapes: two triangles + one quad.
        let frags = resplit_fragments(&mesh, &geom);
        let mut lens: Vec<usize> = frags.iter().map(|&f| mesh.face_len(f)).collect();
        lens.sort_unstable();
        assert_eq!(lens, vec![3, 3, 4], "corner tri + central quad + far tri");

        // The corner triangle {m01, v1, m12} holds the shared corner v1.
        let corner_tri = frags
            .iter()
            .copied()
            .find(|&fk| mesh.face_len(fk) == 3 && mesh.face_vert_keys(fk).contains(&v1))
            .expect("corner triangle holds v1");
        let cv = mesh.face_vert_keys(corner_tri);
        assert!(cv.contains(&m01) && cv.contains(&m12));

        // The far-corner triangle {v0, v2, v3} holds v3 and both uncut corners.
        let far_tri = frags
            .iter()
            .copied()
            .find(|&fk| mesh.face_len(fk) == 3 && mesh.face_vert_keys(fk).contains(&v3))
            .expect("far triangle holds v3");
        let fvv = mesh.face_vert_keys(far_tri);
        assert!(
            fvv.contains(&v0) && fvv.contains(&v2),
            "far tri is {{v0,v2,v3}}"
        );
    }

    #[test]
    fn quad_two_adjacent_fan_strategy() {
        // §5.2.4 FAN — spoke both rim midpoints to the far corner v3. No inner
        // vert. V 6 / E 8 / F 3: tri + central quad (holds v1) + tri, with v3 a
        // degree-4 fan hub.
        let (mesh, geom, [v0, v1, v2, v3]) = adjacent_two_cut(QuadCornerType::Fan);
        let (v, e, f) = counts(&mesh);
        assert_eq!((v, e, f), (6, 8, 3), "FAN: +2V / +4E / +2F");
        assert_eq!(geom.verts.len(), 2, "two rim midpoints, no inner vert");

        let m01 = find_vert_at(&mesh, Vec3::new(1.0, 0.0, 0.0));
        let m12 = find_vert_at(&mesh, Vec3::new(2.0, 1.0, 0.0));
        // Both spokes terminate at v3; v3 never gets cut across, so there is no
        // m01-m12 corner cut.
        assert!(mesh.query_edge_exists(m01, v3).is_some(), "m01->v3 spoke");
        assert!(mesh.query_edge_exists(m12, v3).is_some(), "m12->v3 spoke");
        assert!(
            mesh.query_edge_exists(m01, m12).is_none(),
            "FAN never cuts across the shared corner"
        );

        // Fragment shapes: two triangles + one central quad.
        let frags = resplit_fragments(&mesh, &geom);
        let mut lens: Vec<usize> = frags.iter().map(|&f| mesh.face_len(f)).collect();
        lens.sort_unstable();
        assert_eq!(lens, vec![3, 3, 4], "tri + central quad + tri");

        // The central quad {v3, m01, v1, m12} is the only fragment holding the
        // shared corner v1.
        let central = frags
            .iter()
            .copied()
            .find(|&fk| mesh.face_len(fk) == 4)
            .expect("central quad");
        let qv = mesh.face_vert_keys(central);
        assert!(qv.contains(&v1), "central quad holds the shared corner v1");
        assert!(
            qv.contains(&v3) && qv.contains(&m01) && qv.contains(&m12),
            "central quad is {{v3, m01, v1, m12}}"
        );

        // v1 appears on exactly one fragment (the central quad); the two flank
        // triangles hold the uncut corners v0 / v2.
        let holds_v1 = frags
            .iter()
            .filter(|&&fk| mesh.face_vert_keys(fk).contains(&v1))
            .count();
        assert_eq!(holds_v1, 1, "FAN keeps v1 on a single fragment");

        // v3 is the fan hub: its disk gains both spokes on top of the two uncut
        // edges {v2,v3} and {v3,v0}.
        assert_eq!(
            mesh.vertex_disk_count_edges(v3),
            4,
            "v3 is a degree-4 fan hub"
        );
        let _ = (v0, v2);
    }

    #[test]
    fn quad_corner_type_default_is_straight_cut() {
        // The setter default and SubdivideParams::default() must agree, leaving
        // the core STRAIGHT_CUT outcome unchanged (1 triangle + 1 pentagon).
        assert_eq!(
            SubdivideParams::default().quad_corner_type,
            QuadCornerType::StraightCut
        );
        let (mesh, geom, _) = adjacent_two_cut(QuadCornerType::StraightCut);
        let (v, e, f) = counts(&mesh);
        assert_eq!((v, e, f), (6, 7, 2), "STRAIGHT_CUT unchanged via setter");
        let frags = resplit_fragments(&mesh, &geom);
        let mut lens: Vec<usize> = frags.iter().map(|&f| mesh.face_len(f)).collect();
        lens.sort_unstable();
        assert_eq!(lens, vec![3, 5], "1 triangle + 1 pentagon");
    }

    #[test]
    fn quad_three_edges_mixed_resplit() {
        // §5.3 — three of four edges cut ({v0,v1},{v1,v2},{v2,v3}); {v3,v0}
        // whole. V 4->7 (3 rim midpoints), E 4->10 (3 halved + 3 connecting
        // cuts), 1 quad -> 4 fragments (3 triangles + 1 quad).
        let mut mesh = LoopMesh::new();
        let face = unit_quad(&mut mesh);
        let fv = mesh.face_vert_keys(face);
        let [v0, v1, v2, v3] = [fv[0], fv[1], fv[2], fv[3]];
        let e01 = mesh.query_edge_exists(v0, v1).unwrap();
        let e12 = mesh.query_edge_exists(v1, v2).unwrap();
        let e23 = mesh.query_edge_exists(v2, v3).unwrap();

        let res = mesh.subdivide_edges(&[e01, e12, e23], SubdivideParams::default());
        let geom = match res {
            SubdivideResult::Ran(g) => g,
            _ => panic!("expected Ran"),
        };

        let (v, e, f) = counts(&mesh);
        assert_eq!((v, e, f), (7, 10, 4), "quad three-cut deltas");

        // Fragment shapes: three triangles + one quad.
        let frags = resplit_fragments(&mesh, &geom);
        let mut lens: Vec<usize> = frags.iter().map(|&f| mesh.face_len(f)).collect();
        lens.sort_unstable();
        assert_eq!(lens, vec![3, 3, 3, 4], "3 triangles + 1 quad");

        // The three rim midpoints land at their cut-edge midpoints.
        let m01 = mesh
            .vert_keys()
            .into_iter()
            .find(|&vk| mesh.vertex_position(vk).distance(Vec3::new(1.0, 0.0, 0.0)) < 1e-5)
            .expect("m01");
        let m12 = mesh
            .vert_keys()
            .into_iter()
            .find(|&vk| mesh.vertex_position(vk).distance(Vec3::new(2.0, 1.0, 0.0)) < 1e-5)
            .expect("m12");
        let m23 = mesh
            .vert_keys()
            .into_iter()
            .find(|&vk| mesh.vertex_position(vk).distance(Vec3::new(1.0, 2.0, 0.0)) < 1e-5)
            .expect("m23");

        // The middle midpoint m12 fans to both end midpoints; m01-m23 is the
        // pentagon diagonal. Those three connecting edges exist.
        assert!(
            mesh.query_edge_exists(m12, m01).is_some(),
            "m12-m01 corner cut"
        );
        assert!(
            mesh.query_edge_exists(m12, m23).is_some(),
            "m12-m23 corner cut"
        );
        assert!(
            mesh.query_edge_exists(m01, m23).is_some(),
            "m01-m23 pentagon diagonal"
        );

        // The lone quad fragment is {v0, m01, m23, v3} and holds the two corners
        // of the uncut edge {v3,v0}.
        let quad = frags
            .iter()
            .copied()
            .find(|&f| mesh.face_len(f) == 4)
            .unwrap();
        let qv = mesh.face_vert_keys(quad);
        assert!(
            qv.contains(&v0) && qv.contains(&v3),
            "quad spans the uncut edge"
        );
        assert!(qv.contains(&m01) && qv.contains(&m23));
        assert!(
            !qv.contains(&v1) && !qv.contains(&v2),
            "quad excludes the cut corners"
        );
    }

    #[test]
    fn tri_two_edges_corner_and_quad() {
        // §6.2 — two of three edges cut ({v0,v1},{v1,v2}); {v2,v0} whole. The
        // two cut edges share corner v1. A single m01->m12 cut peels the corner
        // triangle and leaves a quad. V 3->5 (2 rim midpoints), E 3->6 (2 halved
        // + 1 corner cut), 1 tri -> 2 fragments.
        let mut mesh = LoopMesh::new();
        let face = unit_tri(&mut mesh);
        let fv = mesh.face_vert_keys(face);
        let [v0, v1, v2] = [fv[0], fv[1], fv[2]];
        let e01 = mesh.query_edge_exists(v0, v1).unwrap();
        let e12 = mesh.query_edge_exists(v1, v2).unwrap();

        let res = mesh.subdivide_edges(&[e01, e12], SubdivideParams::default());
        let geom = match res {
            SubdivideResult::Ran(g) => g,
            _ => panic!("expected Ran"),
        };

        let (v, e, f) = counts(&mesh);
        assert_eq!((v, e, f), (5, 6, 2), "tri two-cut deltas");

        // Fragment shapes: one triangle + one quad.
        let frags = resplit_fragments(&mesh, &geom);
        let mut lens: Vec<usize> = frags.iter().map(|&f| mesh.face_len(f)).collect();
        lens.sort_unstable();
        assert_eq!(lens, vec![3, 4], "1 triangle + 1 quad");

        let m01 = mesh
            .vert_keys()
            .into_iter()
            .find(|&vk| mesh.vertex_position(vk).distance(Vec3::new(1.0, 0.0, 0.0)) < 1e-5)
            .expect("m01");
        let m12 = mesh
            .vert_keys()
            .into_iter()
            .find(|&vk| mesh.vertex_position(vk).distance(Vec3::new(1.0, 1.0, 0.0)) < 1e-5)
            .expect("m12");
        assert!(
            mesh.query_edge_exists(m01, m12).is_some(),
            "m01-m12 corner cut"
        );

        // The triangle is the cut-off shared corner {m01, v1, m12}; v1 is on it.
        let tri = frags
            .iter()
            .copied()
            .find(|&f| mesh.face_len(f) == 3)
            .unwrap();
        let tv = mesh.face_vert_keys(tri);
        assert!(tv.contains(&v1), "triangle holds the shared corner v1");
        assert!(tv.contains(&m01) && tv.contains(&m12));

        // The quad {v0, m01, m12, v2} excludes the shared corner v1.
        let quad = frags
            .iter()
            .copied()
            .find(|&f| mesh.face_len(f) == 4)
            .unwrap();
        let qv = mesh.face_vert_keys(quad);
        assert!(!qv.contains(&v1), "quad excludes the shared corner v1");
        assert!(qv.contains(&v0) && qv.contains(&v2));
    }

    #[test]
    fn cut_runs_adjacency_discriminator() {
        // §5.2 / §7.1, white-box — the one predicate both dispatchers ask, on
        // rims the two callers hand it. `v(i)` stands for a distinct corner and
        // `m(i)` for a cut vert; the rim is `(vert, is_cut_vert)` in loop order.
        let v = |i: usize| (VertexKey::from_usize(i), false);
        let m = |i: usize| (VertexKey::from_usize(i), true);

        // Two-cut quad at cuts = 1: adjacent midpoints sit 2 apart, opposite 3.
        assert!(LoopMesh::cut_runs_are_adjacent(&[
            v(0),
            m(1),
            v(2),
            m(3),
            v(4),
            v(5)
        ]));
        assert!(!LoopMesh::cut_runs_are_adjacent(&[
            v(0),
            m(1),
            v(2),
            v(3),
            m(4),
            v(5)
        ]));

        // The same rim wound so the first run wraps the ring start — the
        // predicate roots itself at an original corner, so it must not care.
        assert!(LoopMesh::cut_runs_are_adjacent(&[
            m(3),
            v(4),
            v(5),
            v(0),
            m(1),
            v(2)
        ]));

        // Multi-cut runs, N = 2: the quad's two runs of two, adjacent (sharing
        // corner v2) then opposite.
        assert!(LoopMesh::cut_runs_are_adjacent(&[
            v(0),
            m(1),
            m(2),
            v(3),
            m(4),
            m(5),
            v(6),
            v(7)
        ]));
        assert!(!LoopMesh::cut_runs_are_adjacent(&[
            v(0),
            m(1),
            m(2),
            v(3),
            v(4),
            m(5),
            m(6),
            v(7)
        ]));

        // A pentagon's two cut runs, two corners apart on both arcs: neither
        // side is adjacent, and the strip claims it either way.
        assert!(!LoopMesh::cut_runs_are_adjacent(&[
            v(0),
            m(1),
            v(2),
            v(3),
            m(4),
            v(5),
            v(6)
        ]));
    }

    #[test]
    fn cuts_zero_is_noop() {
        let mut mesh = LoopMesh::new();
        let face = unit_quad(&mut mesh);
        let fv = mesh.face_vert_keys(face);
        let e01 = mesh.query_edge_exists(fv[0], fv[1]).unwrap();
        let before = counts(&mesh);
        let res = mesh.subdivide_edges(&[e01], SubdivideParams::default().cuts(0));
        assert!(matches!(res, SubdivideResult::Ran(_)));
        assert_eq!(counts(&mesh), before, "cuts = 0 leaves the mesh untouched");
    }

    #[test]
    fn use_single_edge_off_quad_stays_pentagon() {
        // §1 pattern-gating / §9.3 — with use_single_edge off, the §5.1 one-cut
        // quad re-split is suppressed: the midpoint is inserted but not connected
        // into the interior. The quad stays a flat-cornered pentagon.
        // V 4->5, E 4->5, F unchanged (1).
        let mut mesh = LoopMesh::new();
        let face = unit_quad(&mut mesh);
        let fv = mesh.face_vert_keys(face);
        let e01 = mesh.query_edge_exists(fv[0], fv[1]).unwrap();

        let res = mesh.subdivide_edges(&[e01], SubdivideParams::default().use_single_edge(false));
        assert!(matches!(res, SubdivideResult::Ran(_)));

        let (v, e, f) = counts(&mesh);
        assert_eq!((v, e, f), (5, 5, 1), "single-edge quad off: flat 5-gon");
        assert_eq!(mesh.face_len(face), 5, "the face is a flat pentagon");
    }

    #[test]
    fn use_single_edge_off_tri_stays_quad() {
        // §1 pattern-gating / §9.3 — with use_single_edge off, the §6.1 one-cut
        // tri re-split is suppressed. The triangle stays a flat 4-gon.
        // V 3->4, E 3->4, F unchanged (1).
        let mut mesh = LoopMesh::new();
        let face = unit_tri(&mut mesh);
        let fv = mesh.face_vert_keys(face);
        let e01 = mesh.query_edge_exists(fv[0], fv[1]).unwrap();

        let res = mesh.subdivide_edges(&[e01], SubdivideParams::default().use_single_edge(false));
        assert!(matches!(res, SubdivideResult::Ran(_)));

        let (v, e, f) = counts(&mesh);
        assert_eq!((v, e, f), (4, 4, 1), "single-edge tri off: flat 4-gon");
        assert_eq!(mesh.face_len(face), 4, "the face is a flat 4-gon");
    }

    #[test]
    fn use_grid_fill_off_quad_stays_octagon() {
        // §1 pattern-gating / §9.3 — with use_grid_fill off, the §5.4 all-edges
        // quad plus-split is suppressed: the four rim midpoints are inserted but
        // no inner vert and no spokes. The quad stays a flat 8-gon.
        // V 4->8, E 4->8, F unchanged (1).
        let mut mesh = LoopMesh::new();
        let face = unit_quad(&mut mesh);
        let fv = mesh.face_vert_keys(face);
        let edges: Vec<_> = (0..4)
            .map(|i| mesh.query_edge_exists(fv[i], fv[(i + 1) % 4]).unwrap())
            .collect();

        let res = mesh.subdivide_edges(&edges, SubdivideParams::default().use_grid_fill(false));
        assert!(matches!(res, SubdivideResult::Ran(_)));

        let (v, e, f) = counts(&mesh);
        assert_eq!((v, e, f), (8, 8, 1), "grid-fill quad off: flat 8-gon");
        assert_eq!(mesh.face_len(face), 8, "the face is a flat 8-gon");
    }

    #[test]
    fn use_grid_fill_off_tri_stays_hexagon() {
        // §1 pattern-gating / §9.3 — with use_grid_fill off, the §6.3 all-edges
        // tri 1-to-4 split is suppressed. The triangle stays a flat 6-gon.
        // V 3->6, E 3->6, F unchanged (1).
        let mut mesh = LoopMesh::new();
        let face = unit_tri(&mut mesh);
        let fv = mesh.face_vert_keys(face);
        let edges: Vec<_> = (0..3)
            .map(|i| mesh.query_edge_exists(fv[i], fv[(i + 1) % 3]).unwrap())
            .collect();

        let res = mesh.subdivide_edges(&edges, SubdivideParams::default().use_grid_fill(false));
        assert!(matches!(res, SubdivideResult::Ran(_)));

        let (v, e, f) = counts(&mesh);
        assert_eq!((v, e, f), (6, 6, 1), "grid-fill tri off: flat 6-gon");
        assert_eq!(mesh.face_len(face), 6, "the face is a flat 6-gon");
    }

    #[test]
    fn gates_default_on_still_resplit() {
        // The off-case toggles must not change the default: with default params
        // (both gates on), the one-cut quad still re-splits into 3 fragments and
        // the all-edges quad still plus-splits into 4 — confirming the seven core
        // rows stay green. (Spot-check both gate-governed cases.)
        let mut one_cut = LoopMesh::new();
        let f1 = unit_quad(&mut one_cut);
        let fv = one_cut.face_vert_keys(f1);
        let e01 = one_cut.query_edge_exists(fv[0], fv[1]).unwrap();
        one_cut.subdivide_edges(&[e01], SubdivideParams::default());
        assert_eq!(
            counts(&one_cut),
            (5, 7, 3),
            "default one-cut quad re-splits"
        );

        let mut all_cut = LoopMesh::new();
        let f2 = unit_quad(&mut all_cut);
        let fv = all_cut.face_vert_keys(f2);
        let edges: Vec<_> = (0..4)
            .map(|i| all_cut.query_edge_exists(fv[i], fv[(i + 1) % 4]).unwrap())
            .collect();
        all_cut.subdivide_edges(&edges, SubdivideParams::default());
        assert_eq!(
            counts(&all_cut),
            (9, 12, 4),
            "default four-cut quad plus-splits"
        );
    }

    /// Build a quad `{v0,v1,v2,v3}` and a triangle `{v1,v4,v2}` welded across the
    /// shared interior edge `{v1,v2}`. Returns the mesh and the shared edge.
    fn quad_and_tri_shared_edge() -> (LoopMesh, EdgeKey) {
        let mut mesh = LoopMesh::new();
        let v0 = mesh.create_vertex(Vec3::new(0.0, 0.0, 0.0));
        let v1 = mesh.create_vertex(Vec3::new(1.0, 0.0, 0.0));
        let v2 = mesh.create_vertex(Vec3::new(1.0, 1.0, 0.0));
        let v3 = mesh.create_vertex(Vec3::new(0.0, 1.0, 0.0));
        let v4 = mesh.create_vertex(Vec3::new(2.0, 0.5, 0.0));

        let e01 = mesh.create_edge([v0, v1], EdgeCreateFlags::NONE);
        let e12 = mesh.create_edge([v1, v2], EdgeCreateFlags::NONE);
        let e23 = mesh.create_edge([v2, v3], EdgeCreateFlags::NONE);
        let e30 = mesh.create_edge([v3, v0], EdgeCreateFlags::NONE);
        let e14 = mesh.create_edge([v1, v4], EdgeCreateFlags::NONE);
        let e42 = mesh.create_edge([v4, v2], EdgeCreateFlags::NONE);

        mesh.face_create(
            &[v0, v1, v2, v3],
            &[e01, e12, e23, e30],
            FaceCreateFlags::NONE,
        );
        mesh.face_create(&[v1, v4, v2], &[e14, e42, e12], FaceCreateFlags::NONE);
        (mesh, e12)
    }

    #[test]
    fn use_only_quads_on_skips_shared_triangle() {
        // §9.3 — quad + triangle sharing the cut edge {v1,v2}, gate ON. The §3
        // midpoint is inserted once on the shared edge (V 5->6). The quad
        // re-splits via §5.1 (-> 3 fragments); the triangle is dropped from the
        // dispatch and left flat (its midpoint a 180° rim corner). 2 faces -> 4.
        let (mut mesh, shared) = quad_and_tri_shared_edge();
        assert_eq!(counts(&mesh), (5, 6, 2), "starting quad+tri fixture");

        let geom = match mesh
            .subdivide_edges(&[shared], SubdivideParams::default().use_only_quads(true))
        {
            SubdivideResult::Ran(g) => g,
            _ => panic!("subdivide ran"),
        };

        // §4.4 — the gate suppresses the *pattern*, not the *record*: a one-cut
        // triangle matches a template (§6.1), so the dispatcher records it and
        // stamps its §1 re-split mark whatever the gate then does with it. (This
        // is the same reasoning that keeps a gated-off `use_single_edge` quad
        // marked — a gate is not a dispatch key.)
        assert_eq!(
            geom.geom().faces.len(),
            2,
            "gate ON: the triangle is still recorded and marked, only its pattern is suppressed"
        );

        let (v, e, f) = counts(&mesh);
        assert_eq!(
            (v, f),
            (6, 4),
            "gate ON: shared midpoint + quad->3, tri flat"
        );

        // The triangle that shared the edge is now a flat 4-gon (midpoint
        // inserted as a rim corner, no interior connection).
        let mid = mesh
            .vert_keys()
            .into_iter()
            .find(|&vk| mesh.vertex_position(vk).distance(Vec3::new(1.0, 0.5, 0.0)) < 1e-5)
            .expect("shared midpoint exists");
        // The non-quad face incident to the midpoint keeps all its original
        // corners plus the new midpoint: a flat 4-gon, not re-split.
        let flat = mesh
            .vert_face_keys(mid)
            .into_iter()
            .find(|&fk| {
                mesh.face_len(fk) == 4 && {
                    // the flat 4-gon is the only midpoint-incident face that is NOT
                    // a quad-fragment: it holds the apex v4 (at x=2).
                    mesh.face_vert_keys(fk)
                        .iter()
                        .any(|&vk| mesh.vertex_position(vk).x > 1.5)
                }
            })
            .expect("flat 4-gon on the triangle side");
        assert_eq!(mesh.face_len(flat), 4, "triangle left flat (4-gon)");
        // edge count is consistent: shared edge halved (+1), quad's two §5.1
        // interior connects (+2). 6 -> 9.
        assert_eq!(e, 9, "gate ON edge delta: shared halved + 2 quad connects");
    }

    #[test]
    fn use_only_quads_off_resplits_shared_triangle() {
        // §9.3 default (gate OFF) — same fixture, same cut. Both faces re-split
        // off the shared midpoint: the quad via §5.1 (-> 3) and the triangle via
        // §6.1 connect-to-apex (-> 2). 2 faces -> 5; the triangle side produces
        // one more fragment than the gate-ON outcome.
        let (mut mesh, shared) = quad_and_tri_shared_edge();

        let res = mesh.subdivide_edges(&[shared], SubdivideParams::default());
        assert!(matches!(res, SubdivideResult::Ran(_)));

        let (v, e, f) = counts(&mesh);
        assert_eq!((v, f), (6, 5), "gate OFF: both faces re-split");
        // shared edge halved (+1), quad's two §5.1 connects (+2), triangle's one
        // §6.1 connect (+1). 6 -> 10.
        assert_eq!(e, 10, "gate OFF edge delta vs gate ON adds the tri connect");

        // The triangle side is re-split: the midpoint connects to the apex v4,
        // so no flat 4-gon remains on that side. The midpoint's disk picks up
        // the §6.1 connect in addition to the two halves + two quad connects.
        let mid = mesh
            .vert_keys()
            .into_iter()
            .find(|&vk| mesh.vertex_position(vk).distance(Vec3::new(1.0, 0.5, 0.0)) < 1e-5)
            .expect("shared midpoint exists");
        let apex = mesh
            .vert_keys()
            .into_iter()
            .find(|&vk| mesh.vertex_position(vk).distance(Vec3::new(2.0, 0.5, 0.0)) < 1e-5)
            .expect("triangle apex v4");
        assert!(
            mesh.query_edge_exists(mid, apex).is_some(),
            "gate OFF: triangle re-splits, midpoint connects to apex"
        );
    }

    #[test]
    fn use_only_quads_on_passes_quad_only_mesh() {
        // The gate must not change behaviour when every touched face is a quad.
        // A lone quad with one cut edge still re-splits via §5.1 under the gate.
        let mut mesh = LoopMesh::new();
        let face = unit_quad(&mut mesh);
        let fv = mesh.face_vert_keys(face);
        let e01 = mesh.query_edge_exists(fv[0], fv[1]).unwrap();

        mesh.subdivide_edges(&[e01], SubdivideParams::default().use_only_quads(true));
        assert_eq!(
            counts(&mesh),
            (5, 7, 3),
            "gate ON, all-quad: quad still re-splits"
        );
    }

    #[test]
    fn use_only_quads_on_lone_triangle_stays_flat() {
        // A lone triangle with one cut edge under the gate keeps its midpoint but
        // is left flat (a 4-gon) — the §6.1 apex connect is skipped. V 3->4,
        // E 3->4, F unchanged (1).
        let mut mesh = LoopMesh::new();
        let face = unit_tri(&mut mesh);
        let fv = mesh.face_vert_keys(face);
        let e01 = mesh.query_edge_exists(fv[0], fv[1]).unwrap();

        mesh.subdivide_edges(&[e01], SubdivideParams::default().use_only_quads(true));
        assert_eq!(counts(&mesh), (4, 4, 1), "gate ON, lone tri: flat 4-gon");
        assert_eq!(mesh.face_len(face), 4, "triangle left flat");
    }

    #[test]
    fn linear_uv_midpoint_is_edge_average() {
        // §7.2 — the per-vert linear blend at the midpoint. Decorate verts with
        // a scalar "u" coordinate and check the midpoint reads the average of
        // its edge endpoints.
        let mut mesh = LoopMesh::new();
        let u = mesh
            .register_vertex_attribute(ElementAttributeDesc::numeric(
                AttributeKey::custom("u"),
                AttrType::F32,
            ))
            .unwrap();
        let face = unit_quad(&mut mesh);
        let fv = mesh.face_vert_keys(face);
        // Assign u = x-coordinate of each corner: v0=0, v1=2, v2=2, v3=0.
        for &v in &fv {
            let x = mesh.vertex_position(v).x;
            mesh.vertex_attribs_mut().set_f32(u, v.as_usize(), x);
        }
        let e01 = mesh.query_edge_exists(fv[0], fv[1]).unwrap();
        let res = mesh.subdivide_edges(&[e01], SubdivideParams::default());
        let geom = match res {
            SubdivideResult::Ran(g) => g,
            _ => panic!(),
        };
        let m = geom.verts[0];
        // Midpoint of {v0 (u=0), v1 (u=2)} should read u = 1.
        let got = mesh.vertex_attribs().get_f32(u, m.as_usize()).unwrap();
        assert!(
            (got - 1.0).abs() < 1e-5,
            "midpoint u expected 1.0, got {got}"
        );
    }

    // ---- Multi-cut (`cuts >= 2`, §3.3 / §9.5) ----------------------------

    fn quad_all_edges(mesh: &mut LoopMesh, face: FaceKey) -> Vec<EdgeKey> {
        let fv = mesh.face_vert_keys(face);
        (0..4)
            .map(|i| mesh.query_edge_exists(fv[i], fv[(i + 1) % 4]).unwrap())
            .collect()
    }

    fn tri_all_edges(mesh: &mut LoopMesh, face: FaceKey) -> Vec<EdgeKey> {
        let fv = mesh.face_vert_keys(face);
        (0..3)
            .map(|i| mesh.query_edge_exists(fv[i], fv[(i + 1) % 3]).unwrap())
            .collect()
    }

    #[test]
    fn multicut_quad_grid_cuts2() {
        // §9.5.1 — fully-cut quad at cuts = 2 → 3x3 grid. V 16 / E 24 / F 9.
        let mut mesh = LoopMesh::new();
        let face = unit_quad(&mut mesh);
        let edges = quad_all_edges(&mut mesh, face);

        let res = mesh.subdivide_edges(&edges, SubdivideParams::default().cuts(2));
        let geom = match res {
            SubdivideResult::Ran(g) => g,
            _ => panic!("expected Ran"),
        };
        assert_eq!(counts(&mesh), (16, 24, 9), "quad grid cuts=2 deltas");
        for &f in &geom.faces {
            assert_eq!(mesh.face_len(f), 4, "grid cells are quads");
        }
        // The four interior verts land at the bilinear lattice nodes of the
        // [0,2]^2 quad: (2/3, 2/3), (4/3, 2/3), (2/3, 4/3), (4/3, 4/3).
        for &(x, y) in &[(2.0 / 3.0, 2.0 / 3.0), (4.0 / 3.0, 2.0 / 3.0)] {
            assert!(
                mesh.vert_keys()
                    .into_iter()
                    .any(|vk| mesh.vertex_position(vk).distance(Vec3::new(x, y, 0.0)) < 1e-5),
                "interior lattice vert at ({x}, {y})"
            );
        }
    }

    #[test]
    fn multicut_quad_grid_cuts3() {
        // §9.5.1 — fully-cut quad at cuts = 3 → 4x4 grid. V 25 / E 40 / F 16.
        let mut mesh = LoopMesh::new();
        let face = unit_quad(&mut mesh);
        let edges = quad_all_edges(&mut mesh, face);

        let res = mesh.subdivide_edges(&edges, SubdivideParams::default().cuts(3));
        let geom = match res {
            SubdivideResult::Ran(g) => g,
            _ => panic!("expected Ran"),
        };
        assert_eq!(counts(&mesh), (25, 40, 16), "quad grid cuts=3 deltas");
        assert_eq!(geom.faces.len(), 16, "16 grid cells");
        for &f in &geom.faces {
            assert_eq!(mesh.face_len(f), 4, "grid cells are quads");
        }
    }

    #[test]
    fn multicut_tri_lattice_cuts2() {
        // §9.5.2 — fully-cut triangle at cuts = 2 → 9 small triangles, one
        // interior vert. V 10 / E 18 / F 9.
        let mut mesh = LoopMesh::new();
        let face = unit_tri(&mut mesh);
        let edges = tri_all_edges(&mut mesh, face);

        let res = mesh.subdivide_edges(&edges, SubdivideParams::default().cuts(2));
        let geom = match res {
            SubdivideResult::Ran(g) => g,
            _ => panic!("expected Ran"),
        };
        assert_eq!(counts(&mesh), (10, 18, 9), "tri lattice cuts=2 deltas");
        for &f in &geom.faces {
            assert_eq!(mesh.face_len(f), 3, "lattice cells are triangles");
        }
        // The single interior vert is the barycentric centroid of the tri
        // {(0,0),(2,0),(0,2)} at (1/3, 1/3) weights → (2/3, 2/3).
        let interior: Vec<_> = geom
            .verts
            .iter()
            .copied()
            .filter(|&vk| {
                mesh.vertex_position(vk)
                    .distance(Vec3::new(2.0 / 3.0, 2.0 / 3.0, 0.0))
                    < 1e-5
            })
            .collect();
        assert_eq!(interior.len(), 1, "exactly one interior barycentric vert");
    }

    #[test]
    fn multicut_tri_lattice_cuts3() {
        // §9.5.2 — fully-cut triangle at cuts = 3 → 16 triangles, three
        // interior verts. V 15 / E 30 / F 16.
        let mut mesh = LoopMesh::new();
        let face = unit_tri(&mut mesh);
        let edges = tri_all_edges(&mut mesh, face);

        let res = mesh.subdivide_edges(&edges, SubdivideParams::default().cuts(3));
        let geom = match res {
            SubdivideResult::Ran(g) => g,
            _ => panic!("expected Ran"),
        };
        assert_eq!(counts(&mesh), (15, 30, 16), "tri lattice cuts=3 deltas");
        assert_eq!(geom.faces.len(), 16, "16 lattice cells");
        for &f in &geom.faces {
            assert_eq!(mesh.face_len(f), 3, "lattice cells are triangles");
        }
    }

    #[test]
    fn multicut_quad_one_edge_fan_cuts2() {
        // §9.5.3 — one-edge quad at cuts = 2 → fan of 3 fragments.
        // V 6 / E 8 / F 3.
        let mut mesh = LoopMesh::new();
        let face = unit_quad(&mut mesh);
        let fv = mesh.face_vert_keys(face);
        let e01 = mesh.query_edge_exists(fv[0], fv[1]).unwrap();

        let res = mesh.subdivide_edges(&[e01], SubdivideParams::default().cuts(2));
        assert!(matches!(res, SubdivideResult::Ran(_)));
        assert_eq!(counts(&mesh), (6, 8, 3), "one-edge quad fan cuts=2 deltas");
        // The two midpoints land at t = 1/3 and 2/3 along {v0,v1} = [(0,0),(2,0)].
        for &x in &[2.0 / 3.0, 4.0 / 3.0] {
            assert!(
                mesh.vert_keys()
                    .into_iter()
                    .any(|vk| mesh.vertex_position(vk).distance(Vec3::new(x, 0.0, 0.0)) < 1e-5),
                "rim midpoint at t along the cut edge ({x}, 0)"
            );
        }
    }

    /// Run the §9.5.3.1 fan on a lone quad with only `{v0,v1}` cut, and report
    /// `(V, E, F)`, the sorted fragment-shape multiset, and the run of cut verts
    /// in `v0 -> v1` order alongside the two hubs.
    fn quad_one_edge_fan_at(
        cuts: u32,
    ) -> (
        LoopMesh,
        (usize, usize, usize),
        Vec<usize>,
        Vec<VertexKey>,
        VertexKey,
        VertexKey,
    ) {
        let mut mesh = LoopMesh::new();
        let face = unit_quad(&mut mesh);
        let fv = mesh.face_vert_keys(face);
        let (v2, v3) = (fv[2], fv[3]);
        let e01 = mesh.query_edge_exists(fv[0], fv[1]).unwrap();

        let res = mesh.subdivide_edges(&[e01], SubdivideParams::default().cuts(cuts));
        assert!(matches!(res, SubdivideResult::Ran(_)), "cuts={cuts}");

        let n = cuts as usize;
        let run: Vec<VertexKey> = (1..=n)
            .map(|i| {
                let x = 2.0 * i as f32 / (n as f32 + 1.0);
                find_vert_at(&mesh, Vec3::new(x, 0.0, 0.0))
            })
            .collect();
        let mut shapes: Vec<usize> = mesh.face_keys().iter().map(|&f| mesh.face_len(f)).collect();
        shapes.sort_unstable();
        let vef = counts(&mesh);
        (mesh, vef, shapes, run, v2, v3)
    }

    #[test]
    fn multicut_quad_one_edge_fan_half_split_n1_to_n5() {
        // §9.5.3.1 — the fan's fragment count is N + 1 + [N odd] = 2*ceil(N/2)+1,
        // and the middle fragment is a quad exactly when N is even. Counts alone
        // do not discriminate a wrong hub assignment, so the connect edges
        // themselves are pinned below.
        let expected: [((usize, usize, usize), &[usize]); 5] = [
            ((5, 7, 3), &[3, 3, 3]),
            ((6, 8, 3), &[3, 3, 4]),
            ((7, 11, 5), &[3, 3, 3, 3, 3]),
            ((8, 12, 5), &[3, 3, 3, 3, 4]),
            ((9, 15, 7), &[3, 3, 3, 3, 3, 3, 3]),
        ];
        for (index, (want_counts, want_shapes)) in expected.iter().enumerate() {
            let n = index + 1;
            let (mesh, got_counts, got_shapes, run, v2, v3) = quad_one_edge_fan_at(n as u32);
            assert_eq!(got_counts, *want_counts, "V/E/F at cuts={n}");
            assert_eq!(got_shapes, *want_shapes, "fragment shapes at cuts={n}");

            // The half-split rule, edge by edge: m_i hubs on v3 iff 2i <= n+1 and
            // on v2 iff 2i >= n+1 — the median (odd n) doing both.
            for (zero_based, &m) in run.iter().enumerate() {
                let position = zero_based + 1;
                assert_eq!(
                    mesh.query_edge_exists(m, v3).is_some(),
                    2 * position <= n + 1,
                    "cuts={n}: m_{position} -> v3 connect"
                );
                assert_eq!(
                    mesh.query_edge_exists(m, v2).is_some(),
                    2 * position >= n + 1,
                    "cuts={n}: m_{position} -> v2 connect"
                );
            }
        }
    }

    #[test]
    fn multicut_quad_one_edge_fan_odd_median_double_hubs() {
        // §9.5.3.1 at N = 3 — the discriminating row. The median m_2 lays two
        // connects, giving the middle triangle {m_2, v2, v3}; the single-hub
        // reading would give V 7 / E 10 / F 4 and a middle quad instead.
        let (mesh, got_counts, shapes, run, v2, v3) = quad_one_edge_fan_at(3);
        assert_eq!(got_counts, (7, 11, 5), "N=3 fan is 5 fragments, 11 edges");
        assert_eq!(shapes, vec![3, 3, 3, 3, 3], "odd N fans to all triangles");

        let median = run[1];
        assert!(
            mesh.query_edge_exists(median, v3).is_some()
                && mesh.query_edge_exists(median, v2).is_some(),
            "the median cut vert connects to BOTH hubs"
        );
        // ... and the middle triangle {m_2, v2, v3} closes on the uncut rim edge.
        assert!(mesh.query_edge_exists(v3, v2).is_some(), "uncut rim edge");
    }

    #[test]
    fn multicut_quad_one_edge_fan_even_middle_quad() {
        // §9.5.3.1 at N = 4 — no median, so nothing crosses the middle span and
        // the fragment between m_2's and m_3's strokes is left whole as a quad.
        let (mesh, got_counts, shapes, run, v2, v3) = quad_one_edge_fan_at(4);
        assert_eq!(got_counts, (8, 12, 5), "N=4 fan is 5 fragments, 12 edges");
        assert_eq!(shapes, vec![3, 3, 3, 3, 4], "even N leaves a middle quad");
        for (zero_based, &m) in run.iter().enumerate() {
            let hubs_on_v3 = mesh.query_edge_exists(m, v3).is_some();
            let hubs_on_v2 = mesh.query_edge_exists(m, v2).is_some();
            assert!(
                hubs_on_v3 ^ hubs_on_v2,
                "even N: m_{} hubs on exactly one corner",
                zero_based + 1
            );
        }
    }

    #[test]
    fn multicut_tri_one_edge_fan_is_single_hub_n3() {
        // §9.5.3.2 — the triangle fan has no hub choice and no median tie-break:
        // N + 1 fragments for every N. V 6 / E 9 / F 4 at N = 3. This is the
        // contrast that must NOT be "fixed" to match the quad's §9.5.3.1 rule.
        let mut mesh = LoopMesh::new();
        let face = unit_tri(&mut mesh);
        let fv = mesh.face_vert_keys(face);
        let apex = fv[2];
        let e01 = mesh.query_edge_exists(fv[0], fv[1]).unwrap();

        let res = mesh.subdivide_edges(&[e01], SubdivideParams::default().cuts(3));
        assert!(matches!(res, SubdivideResult::Ran(_)));
        assert_eq!(counts(&mesh), (6, 9, 4), "tri fan cuts=3 is N+1 fragments");
        for f in mesh.face_keys() {
            assert_eq!(mesh.face_len(f), 3, "tri fan fragments are triangles");
        }
        for i in 1..=3 {
            let m = find_vert_at(&mesh, Vec3::new(2.0 * i as f32 / 4.0, 0.0, 0.0));
            assert!(
                mesh.query_edge_exists(m, apex).is_some(),
                "cut vert {i} spokes to the apex"
            );
        }
    }

    #[test]
    fn multicut_tri_one_edge_fan_cuts2() {
        // §9.5.3 — one-edge triangle at cuts = 2 → fan of 3 triangles.
        // V 5 / E 7 / F 3.
        let mut mesh = LoopMesh::new();
        let face = unit_tri(&mut mesh);
        let fv = mesh.face_vert_keys(face);
        let e01 = mesh.query_edge_exists(fv[0], fv[1]).unwrap();

        let res = mesh.subdivide_edges(&[e01], SubdivideParams::default().cuts(2));
        let geom = match res {
            SubdivideResult::Ran(g) => g,
            _ => panic!("expected Ran"),
        };
        assert_eq!(counts(&mesh), (5, 7, 3), "one-edge tri fan cuts=2 deltas");
        for &f in &geom.faces {
            assert_eq!(mesh.face_len(f), 3, "fan fragments are triangles");
        }
        // Both midpoints connect to the opposite apex v2 = (0,2).
        let apex = fv[2];
        for &x in &[2.0 / 3.0, 4.0 / 3.0] {
            let m = find_vert_at(&mesh, Vec3::new(x, 0.0, 0.0));
            assert!(
                mesh.query_edge_exists(m, apex).is_some(),
                "midpoint at ({x},0) connects to the apex"
            );
        }
    }

    #[test]
    fn multicut_shared_edge_two_quads_cuts2_crack_free() {
        // §9.5.4 — two quads sharing edge {v1,v2}, that edge cut at cuts = 2.
        // The 2 midpoints are created once and shared by both quads; each quad
        // re-splits off the shared run. V 8 / E 13 / F 6.
        let mut mesh = LoopMesh::new();
        let v0 = mesh.create_vertex(Vec3::new(0.0, 0.0, 0.0));
        let v1 = mesh.create_vertex(Vec3::new(1.0, 0.0, 0.0));
        let v2 = mesh.create_vertex(Vec3::new(1.0, 1.0, 0.0));
        let v3 = mesh.create_vertex(Vec3::new(0.0, 1.0, 0.0));
        let v4 = mesh.create_vertex(Vec3::new(2.0, 0.0, 0.0));
        let v5 = mesh.create_vertex(Vec3::new(2.0, 1.0, 0.0));

        let e01 = mesh.create_edge([v0, v1], EdgeCreateFlags::NONE);
        let e12 = mesh.create_edge([v1, v2], EdgeCreateFlags::NONE);
        let e23 = mesh.create_edge([v2, v3], EdgeCreateFlags::NONE);
        let e30 = mesh.create_edge([v3, v0], EdgeCreateFlags::NONE);
        let e14 = mesh.create_edge([v1, v4], EdgeCreateFlags::NONE);
        let e45 = mesh.create_edge([v4, v5], EdgeCreateFlags::NONE);
        let e52 = mesh.create_edge([v5, v2], EdgeCreateFlags::NONE);

        mesh.face_create(
            &[v0, v1, v2, v3],
            &[e01, e12, e23, e30],
            FaceCreateFlags::NONE,
        );
        mesh.face_create(
            &[v1, v4, v5, v2],
            &[e14, e45, e52, e12],
            FaceCreateFlags::NONE,
        );
        assert_eq!(counts(&mesh), (6, 7, 2), "starting two-quad fixture");

        mesh.subdivide_edges(&[e12], SubdivideParams::default().cuts(2));
        assert_eq!(
            counts(&mesh),
            (8, 13, 6),
            "shared-edge cuts=2 crack-free deltas"
        );

        // Each shared midpoint sits on the shared edge and welds both quads:
        // its disk has the two halves of the (now thrice-split) shared edge
        // plus one fan connect from each side. Both midpoints are degree 4.
        for &y in &[1.0 / 3.0, 2.0 / 3.0] {
            let m = find_vert_at(&mesh, Vec3::new(1.0, y, 0.0));
            assert_eq!(
                mesh.vertex_disk_count_edges(m),
                4,
                "shared midpoint at (1,{y}) welds both quads (degree 4)"
            );
        }
    }

    #[test]
    fn multicut_use_only_quads_skips_triangle() {
        // §9.3 at multi-cut — quad + triangle sharing the cut edge, gate ON at
        // cuts = 2. The 2 shared midpoints are inserted once; the quad fans, the
        // triangle is left flat.
        let (mut mesh, shared) = quad_and_tri_shared_edge();
        let res = mesh.subdivide_edges(
            &[shared],
            SubdivideParams::default().cuts(2).use_only_quads(true),
        );
        assert!(matches!(res, SubdivideResult::Ran(_)));
        // 5 -> 7 verts (2 shared midpoints), quad -> 3 fans + flat triangle.
        let (v, _e, f) = counts(&mesh);
        assert_eq!(v, 7, "two shared midpoints inserted once");
        assert_eq!(f, 4, "quad fans to 3, triangle left flat (1)");
    }

    // ---- §9.1 smooth displacement ----------------------------------------

    use std::f32::consts::FRAC_PI_3;

    #[test]
    fn smooth_displaces_midpoint_off_chord_on_fold() {
        // §9.1.2 / §9.1.4 — on a 60° fold, cutting the edge {v0, v1} (flat
        // endpoint v0, hinge endpoint v1 with a divergent bisector normal) with
        // smooth = 1.0 pushes the midpoint off the chord. Topology is unchanged
        // (the §9.1.4 invariance): the displacement only moves the inserted vert.
        let (mut mesh, _v, e01) = folded_two_quads(FRAC_PI_3);
        let before = counts(&mesh);

        let res = mesh.subdivide_edges(&[e01], SubdivideParams::default().smooth(1.0));
        let geom = match res {
            SubdivideResult::Ran(g) => g,
            _ => panic!("expected Ran"),
        };

        // Same element deltas as a smooth = 0 single-edge quad cut (§9.1.4):
        // the cut quad re-splits into 3, the flat neighbour is untouched.
        let after = counts(&mesh);
        assert_eq!(
            (after.0 - before.0, after.2 - before.2),
            (1, 2),
            "smooth changes positions, not topology (§9.1.4)"
        );

        // The midpoint moved off the chord midpoint (0.5, 0, 0).
        let chord_mid = Vec3::new(0.5, 0.0, 0.0);
        let placed = midpoint_pos_after_cut(&mesh, &geom);
        assert!(
            placed.distance(chord_mid) > 1.0e-4,
            "smooth bows the midpoint off the chord (got {placed:?})"
        );
    }

    #[test]
    fn smooth_zero_keeps_chord_midpoint() {
        // The smooth = 0 contrast: identical fixture, identical topology, but the
        // midpoint stays exactly on the chord (bit-identical to the core path).
        let (mut mesh, _v, e01) = folded_two_quads(FRAC_PI_3);
        let res = mesh.subdivide_edges(&[e01], SubdivideParams::default().smooth(0.0));
        let geom = match res {
            SubdivideResult::Ran(g) => g,
            _ => panic!("expected Ran"),
        };
        let placed = midpoint_pos_after_cut(&mesh, &geom);
        assert!(
            placed.distance(Vec3::new(0.5, 0.0, 0.0)) < 1.0e-6,
            "smooth = 0 leaves the midpoint on the chord (got {placed:?})"
        );
    }

    #[test]
    fn smooth_falloff_linear_vs_smooth_overshoot() {
        // §9.1.3 step 1 — at a single centre cut LINEAR pins factor = 1.0
        // (lands on co_curved) while the non-linear SMOOTH curve reaches
        // factor = 2.0 (overshoots co_curved ~2×). The two therefore land the
        // midpoint at distinct positions, with SMOOTH twice as far off the chord
        // as LINEAR along the same bow direction.
        let chord_mid = Vec3::new(0.5, 0.0, 0.0);

        let (mut lin_mesh, _v, e_lin) = folded_two_quads(FRAC_PI_3);
        let lin_geom = match lin_mesh.subdivide_edges(
            &[e_lin],
            SubdivideParams::default()
                .smooth(1.0)
                .smooth_falloff(SmoothFalloff::Linear),
        ) {
            SubdivideResult::Ran(g) => g,
            _ => panic!("expected Ran"),
        };
        let lin = midpoint_pos_after_cut(&lin_mesh, &lin_geom);

        let (mut sm_mesh, _v, e_sm) = folded_two_quads(FRAC_PI_3);
        let sm_geom = match sm_mesh.subdivide_edges(
            &[e_sm],
            SubdivideParams::default()
                .smooth(1.0)
                .smooth_falloff(SmoothFalloff::Smooth),
        ) {
            SubdivideResult::Ran(g) => g,
            _ => panic!("expected Ran"),
        };
        let sm = midpoint_pos_after_cut(&sm_mesh, &sm_geom);

        let off_lin = lin - chord_mid;
        let off_sm = sm - chord_mid;
        assert!(off_lin.length() > 1.0e-4, "LINEAR still displaces");
        assert!(
            lin.distance(sm) > 1.0e-4,
            "LINEAR and SMOOTH land in different positions at a centre cut"
        );
        // SMOOTH overshoots ~2× along the same direction as LINEAR.
        assert!(
            (off_sm.length() / off_lin.length() - 2.0).abs() < 1.0e-3,
            "SMOOTH centre-cut offset is ~2× LINEAR's (factor 2.0 vs 1.0): ratio {}",
            off_sm.length() / off_lin.length()
        );
    }

    #[test]
    fn smooth_circular_arc_lands_exact_position_on_fold() {
        // §9.1.2 — the dual-blend circular arc lands a *specific* point, derived
        // from the angle-weighted endpoint normals (§9.1.5). On the 60° fold,
        // cutting {v0, v1} with smooth = 1.0 and the LINEAR falloff (factor = 1.0,
        // no overshoot, no shell):
        //
        //   - v0 touches only the flat quad, so its angle-weighted normal is
        //     (0, 0, 1); its one-sided arc is chord-perpendicular and collapses
        //     to the straight lerp (toward_start = chord midpoint).
        //   - v1 is shared by the flat quad (normal (0,0,1)) and the 60°-folded
        //     quad (normal (-sin60, 0, cos60)); equal 90° corner angles make its
        //     angle-weighted normal their bisector, (-0.5, 0, sin60) normalised
        //     = (-0.5, 0, 0.8660) — i.e. 30° off vertical toward -x.
        //   - The v1-keyed circular arc through (0,0,0) and (1,0,0), tangent to
        //     that normal, is a unit-radius circle centred at (0.5, 0, sin60).
        //     Its symmetric midpoint is (0.5, 0, sin60 - 1) = (0.5, 0, -0.13397).
        //   - co_curved = lerp(chord_mid, that, 0.5) = (0.5, 0, -0.0669873).
        //
        // The bow is purely in -z (no x component): the old cubic-Hermite
        // construction tilted the path in x via the tangent-plane projection and
        // would land elsewhere, so this exact-position pin guards the §9.1.2
        // arc against regressing to the chord-bowing form.
        let (mut mesh, _v, e01) = folded_two_quads(FRAC_PI_3);
        let res = mesh.subdivide_edges(
            &[e01],
            SubdivideParams::default()
                .smooth(1.0)
                .smooth_falloff(SmoothFalloff::Linear),
        );
        let geom = match res {
            SubdivideResult::Ran(g) => g,
            _ => panic!("expected Ran"),
        };
        let placed = midpoint_pos_after_cut(&mesh, &geom);

        let sin60 = FRAC_PI_3.sin();
        let expected = Vec3::new(0.5, 0.0, (sin60 - 1.0) * 0.5);
        assert!(
            placed.distance(expected) < 1.0e-5,
            "circular-arc midpoint {placed:?} != expected {expected:?}"
        );
        // The bow is purely off the surface (no chord-tangent drift): x and y
        // stay on the chord midplane, only z moves.
        assert!(
            (placed.x - 0.5).abs() < 1.0e-6 && placed.y.abs() < 1.0e-6,
            "arc midpoint stays on the symmetric chord midplane (got {placed:?})"
        );
    }

    #[test]
    fn smooth_uses_angle_weighted_normals() {
        // §9.1.5 — the smooth arc reads angle-weighted (not area-weighted)
        // endpoint normals. On a fixture where the two weightings disagree, the
        // bow direction differs. Build a fan around a hub vertex whose incident
        // faces have very different areas but equal corner angles at the hub: the
        // angle-weighted normal is the equal-weight average of the face normals,
        // whereas an area-weighted normal would be dominated by the large face.
        //
        // Concretely two right triangles share the hub edge and fold about it;
        // one fold quad is stretched far along its free direction (large area)
        // while keeping the same 90° hub corner. The angle-weighted hub normal
        // bisects the fold regardless of the stretch, so the placed midpoint is
        // independent of the stretch factor — a property area-weighting breaks.
        let placed_for = |stretch: f32| -> Vec3 {
            let mut mesh = LoopMesh::new();
            let s = FRAC_PI_3.sin();
            let c = FRAC_PI_3.cos();
            // Flat quad A in z = 0, square.
            let v0 = mesh.create_vertex(Vec3::new(0.0, 0.0, 0.0));
            let v1 = mesh.create_vertex(Vec3::new(1.0, 0.0, 0.0));
            let v2 = mesh.create_vertex(Vec3::new(1.0, 1.0, 0.0));
            let v3 = mesh.create_vertex(Vec3::new(0.0, 1.0, 0.0));
            // Folded quad B about the hinge {v1, v2}; its free side is stretched
            // by `stretch` (changing area, not the 90° corner angles at v1/v2).
            let v4 = mesh.create_vertex(Vec3::new(1.0 + c * stretch, 0.0, s * stretch));
            let v5 = mesh.create_vertex(Vec3::new(1.0 + c * stretch, 1.0, s * stretch));
            let e01 = mesh.create_edge([v0, v1], EdgeCreateFlags::NONE);
            let e12 = mesh.create_edge([v1, v2], EdgeCreateFlags::NONE);
            let e23 = mesh.create_edge([v2, v3], EdgeCreateFlags::NONE);
            let e30 = mesh.create_edge([v3, v0], EdgeCreateFlags::NONE);
            let e14 = mesh.create_edge([v1, v4], EdgeCreateFlags::NONE);
            let e45 = mesh.create_edge([v4, v5], EdgeCreateFlags::NONE);
            let e52 = mesh.create_edge([v5, v2], EdgeCreateFlags::NONE);
            mesh.face_create(
                &[v0, v1, v2, v3],
                &[e01, e12, e23, e30],
                FaceCreateFlags::NONE,
            );
            mesh.face_create(
                &[v1, v4, v5, v2],
                &[e14, e45, e52, e12],
                FaceCreateFlags::NONE,
            );
            let res = mesh.subdivide_edges(
                &[e01],
                SubdivideParams::default()
                    .smooth(1.0)
                    .smooth_falloff(SmoothFalloff::Linear),
            );
            let geom = match res {
                SubdivideResult::Ran(g) => g,
                _ => panic!("expected Ran"),
            };
            midpoint_pos_after_cut(&mesh, &geom)
        };

        // The hub corner angle is unchanged by the stretch, so angle-weighting
        // gives the same hub normal and therefore the same bow at both stretches.
        let near = placed_for(1.0);
        let far = placed_for(4.0);
        assert!(
            near.distance(far) < 1.0e-5,
            "angle-weighted normals make the bow stretch-independent: \
             {near:?} vs {far:?} (area-weighting would differ)"
        );
    }

    #[test]
    fn use_smooth_even_scales_offset_up() {
        // §9.1.3 step 2 — on the non-parallel fold edge the shell factor is > 1,
        // so use_smooth_even moves the midpoint further off the chord than the
        // flag-off run (same LINEAR falloff, same smooth gain).
        let chord_mid = Vec3::new(0.5, 0.0, 0.0);

        let (mut off_mesh, _v, e_off) = folded_two_quads(FRAC_PI_3);
        let off_geom = match off_mesh.subdivide_edges(
            &[e_off],
            SubdivideParams::default()
                .smooth(1.0)
                .smooth_falloff(SmoothFalloff::Linear)
                .use_smooth_even(false),
        ) {
            SubdivideResult::Ran(g) => g,
            _ => panic!("expected Ran"),
        };
        let off = midpoint_pos_after_cut(&off_mesh, &off_geom);

        let (mut on_mesh, _v, e_on) = folded_two_quads(FRAC_PI_3);
        let on_geom = match on_mesh.subdivide_edges(
            &[e_on],
            SubdivideParams::default()
                .smooth(1.0)
                .smooth_falloff(SmoothFalloff::Linear)
                .use_smooth_even(true),
        ) {
            SubdivideResult::Ran(g) => g,
            _ => panic!("expected Ran"),
        };
        let on = midpoint_pos_after_cut(&on_mesh, &on_geom);

        let d_off = off.distance(chord_mid);
        let d_on = on.distance(chord_mid);
        assert!(d_off > 1.0e-4, "flag-off run still displaces");
        assert!(
            d_on > d_off * (1.0 + 1.0e-3),
            "use_smooth_even shell factor (>1) pushes the midpoint further \
             (on {d_on} vs off {d_off})"
        );
    }

    // ---- §9.4 use_sphere (dome / to-sphere projection) -------------------

    /// A cage quad sitting on the cube face `x = 1`: its four corners
    /// `(1,±1,±1)` each lie on the origin-centred sphere of radius √3 (the
    /// §9.4 "corners already on the hull" cage). Loop order
    /// `[(1,1,1),(1,1,-1),(1,-1,-1),(1,-1,1)]`, so the edge `{v0,v1}` runs
    /// `(1,1,1)→(1,1,-1)` with chord midpoint `(1,1,0)` (radius √2). Returns the
    /// mesh and the face; the sphere radius is √3.
    fn sphere_cage_quad(mesh: &mut LoopMesh) -> FaceKey {
        create_face_from_verts(
            mesh,
            &[
                Vec3::new(1.0, 1.0, 1.0),
                Vec3::new(1.0, 1.0, -1.0),
                Vec3::new(1.0, -1.0, -1.0),
                Vec3::new(1.0, -1.0, 1.0),
            ],
        )
    }

    const SPHERE_R: f32 = 1.732_050_8; // √3, the cage-corner radius

    #[test]
    fn use_sphere_projects_single_midpoint() {
        // §9.4.1 — cut one cage edge with use_sphere and smooth = √3. The §3
        // chord midpoint (1,1,0) (radius √2) is snapped radially out to radius
        // √3 (no normals, no arc), while the four corners stay put.
        let mut mesh = LoopMesh::new();
        let face = sphere_cage_quad(&mut mesh);
        let fv = mesh.face_vert_keys(face);
        let e01 = mesh.query_edge_exists(fv[0], fv[1]).unwrap();

        let res = mesh.subdivide_edges(
            &[e01],
            SubdivideParams::default().smooth(SPHERE_R).use_sphere(true),
        );
        let geom = match res {
            SubdivideResult::Ran(g) => g,
            _ => panic!("expected Ran"),
        };

        // §9.4 topology-invariance: same fragment graph as the chord run (§5.1
        // one-edge quad re-split, 1 quad -> 3 faces).
        let (v, e, f) = counts(&mesh);
        assert_eq!(
            (v, e, f),
            (5, 7, 3),
            "use_sphere is topology-invariant (§9.4)"
        );

        // The midpoint is the single inserted vert. Its direction is the chord
        // midpoint (1,1,0) and its distance from the origin is now exactly √3.
        let m = geom.verts[0];
        let pos = mesh.vertex_position(m);
        assert!(
            (pos.length() - SPHERE_R).abs() < 1e-5,
            "midpoint snapped to radius √3, got {}",
            pos.length()
        );
        // Direction preserved: (1,1,0) normalised, scaled to √3 -> (1.2247..,
        // 1.2247.., 0).
        let expected = Vec3::new(1.0, 1.0, 0.0).normalize() * SPHERE_R;
        assert!(
            pos.distance(expected) < 1e-5,
            "projected along the chord radial"
        );

        // §9.4.2 — pre-existing corners are never moved. The corner at (1,1,1)
        // is still exactly there (the rule never touches originals).
        let corner = find_vert_at(&mesh, Vec3::new(1.0, 1.0, 1.0));
        assert!(
            mesh.vertex_position(corner)
                .distance(Vec3::new(1.0, 1.0, 1.0))
                < 1e-9,
            "original corner unmoved"
        );
    }

    #[test]
    fn use_sphere_off_keeps_chord_midpoint() {
        // §9.4.3 contrast — with use_sphere off (and no smooth offset) the same
        // cut leaves the midpoint on its chord at (1,1,0), radius √2.
        let mut mesh = LoopMesh::new();
        let face = sphere_cage_quad(&mut mesh);
        let fv = mesh.face_vert_keys(face);
        let e01 = mesh.query_edge_exists(fv[0], fv[1]).unwrap();

        let res = mesh.subdivide_edges(&[e01], SubdivideParams::default().use_sphere(false));
        let geom = match res {
            SubdivideResult::Ran(g) => g,
            _ => panic!("expected Ran"),
        };

        let m = geom.verts[0];
        let pos = mesh.vertex_position(m);
        assert!(
            pos.distance(Vec3::new(1.0, 1.0, 0.0)) < 1e-5,
            "chord midpoint stays at (1,1,0)"
        );
        assert!(
            (pos.length() - std::f32::consts::SQRT_2).abs() < 1e-5,
            "chord midpoint is at radius √2, not √3"
        );
    }

    #[test]
    fn use_sphere_skips_smooth_offset() {
        // §9.4.3 — use_sphere and the §9.1 smooth offset are mutually exclusive;
        // sphere wins. A non-flat fold edge would bow under §9.1, but with
        // use_sphere on the midpoint must land on the sphere of radius `smooth`
        // (the radial snap), NOT on the §9.1 curved target.
        let (mut mesh, _v, e) = folded_two_quads(FRAC_PI_3);
        let res = mesh.subdivide_edges(
            &[e],
            SubdivideParams::default()
                .smooth(SPHERE_R)
                .smooth_falloff(SmoothFalloff::Linear)
                .use_sphere(true),
        );
        let geom = match res {
            SubdivideResult::Ran(g) => g,
            _ => panic!("expected Ran"),
        };
        let pos = midpoint_pos_after_cut(&mesh, &geom);
        // The radial snap pins the midpoint exactly on the sphere of radius √3.
        assert!(
            (pos.length() - SPHERE_R).abs() < 1e-5,
            "use_sphere snaps to the radius, ignoring the §9.1 offset (r = {})",
            pos.length()
        );
    }

    #[test]
    fn use_sphere_all_four_edges_dome() {
        // §9.4.4 + §5.4 — cut all four cage edges with use_sphere and smooth =
        // √3. All four rim midpoints (chord radius √2) and the §5.4 inner vert
        // (chord = face centroid (1,0,0), radius 1) snap onto the sphere of
        // radius √3. Topology is the unchanged §5.4 plus-split (2x2 grid).
        let mut mesh = LoopMesh::new();
        let face = sphere_cage_quad(&mut mesh);
        let fv = mesh.face_vert_keys(face);
        let edges: Vec<_> = (0..4)
            .map(|i| mesh.query_edge_exists(fv[i], fv[(i + 1) % 4]).unwrap())
            .collect();

        let res = mesh.subdivide_edges(
            &edges,
            SubdivideParams::default().smooth(SPHERE_R).use_sphere(true),
        );
        let geom = match res {
            SubdivideResult::Ran(g) => g,
            _ => panic!("expected Ran"),
        };

        let (v, e, f) = counts(&mesh);
        assert_eq!(
            (v, e, f),
            (9, 12, 4),
            "use_sphere four-cut is the §5.4 plus-split"
        );

        // Every inserted vert (4 rim midpoints + 1 inner) lands on the sphere.
        assert_eq!(geom.verts.len(), 5, "4 rim midpoints + 1 inner vert");
        for &iv in &geom.verts {
            let r = mesh.vertex_position(iv).length();
            assert!(
                (r - SPHERE_R).abs() < 1e-5,
                "inserted vert projected onto the sphere (r = {r})"
            );
        }

        // The inner vert (geom's last) is the centroid (1,0,0) snapped to
        // (√3,0,0): its direction is +x and its radius is √3.
        let inner = *geom.verts.last().unwrap();
        let inner_pos = mesh.vertex_position(inner);
        assert!(
            inner_pos.distance(Vec3::new(SPHERE_R, 0.0, 0.0)) < 1e-5,
            "inner vert snapped outward from the centroid along +x, got {inner_pos:?}"
        );

        // The four corners are untouched (still at radius √3 where they began).
        let corner = find_vert_at(&mesh, Vec3::new(1.0, 1.0, 1.0));
        assert!(
            mesh.vertex_position(corner)
                .distance(Vec3::new(1.0, 1.0, 1.0))
                < 1e-9
        );
    }

    #[test]
    fn use_sphere_multicut_lattice_dome() {
        // §9.4.4 — multi-cut composes with the radial snap: cut all four cage
        // edges at cuts = 2 (the §9.5.1 3x3 grid). Every inserted vert (8 rim
        // midpoints + 4 interior bilinear nodes) is snapped onto the sphere.
        let mut mesh = LoopMesh::new();
        let face = sphere_cage_quad(&mut mesh);
        let fv = mesh.face_vert_keys(face);
        let edges: Vec<_> = (0..4)
            .map(|i| mesh.query_edge_exists(fv[i], fv[(i + 1) % 4]).unwrap())
            .collect();

        let res = mesh.subdivide_edges(
            &edges,
            SubdivideParams::default()
                .cuts(2)
                .smooth(SPHERE_R)
                .use_sphere(true),
        );
        let geom = match res {
            SubdivideResult::Ran(g) => g,
            _ => panic!("expected Ran"),
        };

        // §9.5.1 3x3 grid topology is unchanged by use_sphere: V 16 / E 24 / F 9.
        let (v, e, f) = counts(&mesh);
        assert_eq!(
            (v, e, f),
            (16, 24, 9),
            "cuts=2 grid topology unchanged (§9.5.1)"
        );

        // 8 rim midpoints + 4 interior lattice nodes, all on the sphere.
        assert_eq!(geom.verts.len(), 12, "8 rim midpoints + 4 interior nodes");
        for &iv in &geom.verts {
            let r = mesh.vertex_position(iv).length();
            assert!(
                (r - SPHERE_R).abs() < 1e-5,
                "lattice node projected onto the sphere (r = {r})"
            );
        }
    }

    // ---- Output slots (§1) ------------------------------------------------

    /// Assert a vert/edge slice has no duplicate handles (the slots are sets,
    /// so length-as-membership assertions below must rest on no double-counts).
    fn no_dup<K: Copy + PartialEq + std::fmt::Debug>(items: &[K], what: &str) {
        for (i, &a) in items.iter().enumerate() {
            for &b in &items[i + 1..] {
                assert_ne!(a, b, "{what} has a duplicate handle: {a:?}");
            }
        }
    }

    #[test]
    fn slots_quad_single_edge() {
        // §5.1 / §1.1 — one cut edge of a quad. The §5.1 corner-cut pattern is
        // non-grid, so it marks *none* of its connecting edges or fragment faces
        // inner; its only inner contribution is the rim midpoint. The pinned
        // cardinalities (§1.1): split {1V, 2E, 0F}, inner {1V, 0E, 0F},
        // geom {1V, 2E, 1F} — the geom face count is 1, not 3.
        let mut mesh = LoopMesh::new();
        let face = unit_quad(&mut mesh);
        let fv = mesh.face_vert_keys(face);
        let e01 = mesh.query_edge_exists(fv[0], fv[1]).unwrap();

        let geom = match mesh.subdivide_edges(&[e01], SubdivideParams::default()) {
            SubdivideResult::Ran(g) => g,
            _ => panic!("expected Ran"),
        };

        // geom_split: the one midpoint + its two edge-halves (§3).
        let split = geom.geom_split();
        assert_eq!(split.verts.len(), 1, "one midpoint in geom_split");
        assert_eq!(split.edges.len(), 2, "two edge-halves in geom_split");
        assert!(split.faces.is_empty(), "the split creates no faces");
        no_dup(split.edges, "geom_split.edges");
        // The two halves are exactly the segments of the cut edge {v0, v1}.
        let m = split.verts[0];
        assert!(
            split.edges.iter().all(|&e| {
                let [a, b] = mesh.edge_vert_keys(e);
                a == m || b == m
            }),
            "each edge-half is incident to the midpoint"
        );

        // geom_inner: only the rim midpoint is inner; the §5.1 connecting edges
        // and fragments are *not* (§1.1 inner = {1V, 0E, 0F}).
        let inner = geom.geom_inner();
        assert_eq!(inner.verts.len(), 1, "the rim midpoint is inner");
        assert_eq!(inner.verts[0], m, "the inner vert is the rim midpoint");
        assert!(
            inner.edges.is_empty(),
            "§5.1 marks no connecting edge inner"
        );
        assert!(inner.faces.is_empty(), "§5.1 marks no fragment face inner");

        // geom (union): split ∪ inner ∪ {one re-split-marked original face}.
        // The two connecting edges (m→v3, m→v2) are in *no* mark-set, so the
        // union edge count is just the 2 split-halves (§1.1 geom = {1V, 2E, 1F}).
        let g = geom.geom();
        assert_eq!(g.verts.len(), 1, "geom union: one created vert");
        assert_eq!(
            g.faces.len(),
            1,
            "geom union: 1 re-split mark, NOT 3 fragments"
        );
        assert_eq!(g.faces[0], face, "the re-split mark is the original face");
        assert_eq!(g.edges.len(), 2, "geom union: just the 2 split-halves");
        no_dup(g.edges, "geom.edges");
        // geom_split alone covers geom's edge set for this non-grid pattern.
        for &e in split.edges {
            assert!(g.edges.contains(&e), "split edge is in the union");
        }
    }

    #[test]
    fn slots_quad_four_edges_plus_split() {
        // §5.4 / §1 — the headline four-cut quad. This is a grid-style fill, so
        // it marks its interior geometry inner. geom_split: 4 midpoints + 8
        // edge-halves. geom_inner: 4 rim midpoints + the centre inner vert (5V),
        // the 4 "plus" spokes (4E), and the fragment faces minus the one
        // re-split-marked original (3F). geom: split ∪ inner ∪ the re-split mark.
        let mut mesh = LoopMesh::new();
        let face = unit_quad(&mut mesh);
        let edges = quad_all_edges(&mut mesh, face);

        let geom = match mesh.subdivide_edges(&edges, SubdivideParams::default()) {
            SubdivideResult::Ran(g) => g,
            _ => panic!("expected Ran"),
        };

        let split = geom.geom_split();
        assert_eq!(split.verts.len(), 4, "4 rim midpoints in geom_split");
        assert_eq!(split.edges.len(), 8, "8 edge-halves in geom_split");
        no_dup(split.verts, "geom_split.verts");
        no_dup(split.edges, "geom_split.edges");

        let inner = geom.geom_inner();
        // §1 — all 4 rim midpoints are inner, plus the centre vert (5 total).
        assert_eq!(inner.verts.len(), 5, "4 rim midpoints + 1 centre vert");
        no_dup(inner.verts, "geom_inner.verts");
        assert_eq!(inner.edges.len(), 4, "the 4 plus-split spokes");
        no_dup(inner.edges, "geom_inner.edges");
        // The grid fill marks all but one of its 4 fragments inner (the one that
        // keeps the original face's identity carries the re-split mark instead).
        assert_eq!(
            inner.faces.len(),
            3,
            "4 fragments minus 1 re-split original"
        );
        no_dup(inner.faces, "geom_inner.faces");
        assert!(
            !inner.faces.contains(&face),
            "the original re-split face is not an inner face"
        );
        // The centre vert is the centroid (1, 1, 0); each inner edge meets it.
        let c = inner
            .verts
            .iter()
            .copied()
            .find(|&v| mesh.vertex_position(v).distance(Vec3::new(1.0, 1.0, 0.0)) < 1e-5)
            .expect("centre vert at the centroid");
        for &e in inner.edges {
            let [a, b] = mesh.edge_vert_keys(e);
            assert!(a == c || b == c, "each spoke meets the inner vert");
        }

        let g = geom.geom();
        assert_eq!(g.verts.len(), 5, "geom union: 4 mid + 1 inner");
        assert_eq!(g.edges.len(), 12, "geom union: 8 split + 4 inner");
        assert_eq!(g.faces.len(), 4, "geom union: 3 inner + 1 re-split mark");
        no_dup(g.edges, "geom.edges");
        no_dup(g.verts, "geom.verts");
        no_dup(g.faces, "geom.faces");
        assert!(g.faces.contains(&face), "the re-split mark is in the union");
        assert_eq!(counts(&mesh), (9, 12, 4), "four-cut quad deltas");
        // The midpoints are shared between geom_split and geom_inner, so the
        // union vert count is split (4) + just the centre (1) = 5, and edges
        // partition cleanly (split halves vs. inner spokes are disjoint).
        assert_eq!(
            split.edges.len() + inner.edges.len(),
            g.edges.len(),
            "split+inner edges partition the union"
        );
    }

    #[test]
    fn slots_multicut_quad_grid_cuts2() {
        // §9.5.1 / §1.1 — fully-cut quad at cuts = 2 (3x3 grid). The pinned
        // §1.1 cardinalities: split {8V, 12E}, inner {12V, 12E, 8F}, geom
        // {12V, 24E, 9F}. inner verts = 8 rim midpoints (all inner) + 4 interior
        // lattice nodes; inner faces = the 9 cells minus the one re-split mark.
        let mut mesh = LoopMesh::new();
        let face = unit_quad(&mut mesh);
        let edges = quad_all_edges(&mut mesh, face);

        let geom = match mesh.subdivide_edges(&edges, SubdivideParams::default().cuts(2)) {
            SubdivideResult::Ran(g) => g,
            _ => panic!("expected Ran"),
        };

        let split = geom.geom_split();
        assert_eq!(split.verts.len(), 8, "2 midpoints per edge x 4 edges");
        assert_eq!(split.edges.len(), 12, "3 segments per edge x 4 edges");
        no_dup(split.verts, "geom_split.verts");
        no_dup(split.edges, "geom_split.edges");

        let inner = geom.geom_inner();
        // §1 — 8 rim midpoints are inner, plus the 4 interior lattice nodes.
        assert_eq!(inner.verts.len(), 12, "8 rim midpoints + 4 interior nodes");
        no_dup(inner.verts, "geom_inner.verts");
        assert_eq!(inner.edges.len(), 12, "the 12 interior lattice edges");
        no_dup(inner.edges, "geom_inner.edges");
        assert_eq!(inner.faces.len(), 8, "9 cells minus 1 re-split original");
        no_dup(inner.faces, "geom_inner.faces");
        // No inner edge is one of the rim edge-halves (disjoint slots).
        for &e in inner.edges {
            assert!(
                !split.edges.contains(&e),
                "inner lattice edge is not a rim edge-half"
            );
        }

        let g = geom.geom();
        assert_eq!(g.verts.len(), 12, "geom union: 8 rim + 4 interior verts");
        assert_eq!(g.faces.len(), 9, "geom union: the 3x3 cells");
        no_dup(g.verts, "geom.verts");
        no_dup(g.edges, "geom.edges");
        no_dup(g.faces, "geom.faces");
        // Union edges = the 12 rim halves + every interior lattice edge; the
        // total matches the final mesh edge count (no edge survives uncut).
        assert_eq!(g.edges.len(), 24, "geom union covers all 24 mesh edges");
        assert_eq!(split.edges.len() + inner.edges.len(), g.edges.len());
        assert_eq!(counts(&mesh), (16, 24, 9), "cuts=2 grid deltas");
    }

    // ---- §9.2 fractal / along_normal / seed ----

    /// Cut edge `{0,1}` of the flat `unit_quad` with the given fractal params and
    /// return `(mesh, geom, midpoint_position)`. The cut edge runs from
    /// `(0,0,0)` to `(2,0,0)`; its chord midpoint is `(1,0,0)`.
    fn fractal_cut(params: SubdivideParams) -> (LoopMesh, SubdivideGeom, Vec3) {
        let mut mesh = LoopMesh::new();
        let face = unit_quad(&mut mesh);
        let fv = mesh.face_vert_keys(face);
        let e01 = mesh.query_edge_exists(fv[0], fv[1]).unwrap();
        let geom = match mesh.subdivide_edges(&[e01], params) {
            SubdivideResult::Ran(g) => g,
            _ => panic!("expected Ran"),
        };
        let pos = mesh.vertex_position(geom.verts[0]);
        (mesh, geom, pos)
    }

    #[test]
    fn fractal_zero_keeps_chord_midpoint() {
        // §9.2 / §9.2.6 — fractal = 0 is the no-op core: the midpoint stays on
        // the chord at (1,0,0) and the §5.1 one-edge re-split is unchanged.
        let (mesh, _geom, pos) = fractal_cut(SubdivideParams::default());
        assert!(
            pos.distance(Vec3::new(1.0, 0.0, 0.0)) < 1e-6,
            "fractal = 0 keeps the chord midpoint, got {pos:?}"
        );
        assert_eq!(counts(&mesh), (5, 7, 3), "core single-edge quad re-split");
    }

    #[test]
    fn fractal_displaces_midpoint_deterministically() {
        // §9.2.3 — a non-zero fractal pushes the midpoint off the chord. The
        // displacement is a deterministic function of seed + position (§9.2.5):
        // it matches the §9.2 arithmetic computed independently, and is
        // topology-invariant (counts unchanged vs the fractal = 0 core).
        let params = SubdivideParams::default().fractal(1.0).seed(1);
        let (mesh, _geom, pos) = fractal_cut(params);

        assert!(
            pos.distance(Vec3::new(1.0, 0.0, 0.0)) > 1e-4,
            "fractal displaces the midpoint off the chord, got {pos:?}"
        );
        assert_eq!(
            counts(&mesh),
            (5, 7, 3),
            "fractal is topology-invariant (§9.2)"
        );

        // Self-check the exact §9.2 placement. On the flat quad both endpoints'
        // angle-weighted normals are +Z, so the basis matches a known offset and
        // the in-crate noise port pins the position to the bit.
        let offset = {
            let mut rng = fractal_noise::SeededRng::new_hashed(1);
            Vec3::new(
                rng.next_float() * 200.0,
                rng.next_float() * 200.0,
                rng.next_float() * 200.0,
            )
        };
        let ctx = FractalEdgeContext {
            edge_len: 2.0,
            no_a: Vec3::Z,
            no_b: Vec3::Z,
            // The endpoint fields are inert for this `fractal_displace` self-check
            // (it reads only edge_len / no_a / no_b); placeholder keys/positions.
            va: VertexKey(0),
            co_a: Vec3::new(0.0, 0.0, 0.0),
            vb: VertexKey(1),
            co_b: Vec3::new(2.0, 0.0, 0.0),
        };
        let expected = fractal_displace(Vec3::new(1.0, 0.0, 0.0), offset, &ctx, &params);
        assert!(
            pos.distance(expected) < 1e-6,
            "midpoint matches the §9.2 arithmetic exactly: got {pos:?}, want {expected:?}"
        );
    }

    #[test]
    fn fractal_same_seed_reproducible_different_seed_differs() {
        // §9.2.1 — the RNG is consumed once up front, so a fixed seed is fully
        // reproducible while a different seed samples a different noise region.
        let (_m1, _g1, a) = fractal_cut(SubdivideParams::default().fractal(1.0).seed(1));
        let (_m2, _g2, b) = fractal_cut(SubdivideParams::default().fractal(1.0).seed(1));
        let (_m3, _g3, c) = fractal_cut(SubdivideParams::default().fractal(1.0).seed(2));

        assert!(a.distance(b) < 1e-9, "same seed -> identical position");
        assert!(a.distance(c) > 1e-5, "different seed -> different position");
    }

    #[test]
    fn fractal_along_normal_constrains_to_normal_axis() {
        // §9.2.4 — along_normal = 1.0 zeroes the two tangential components, so
        // the midpoint moves only along the averaged endpoint normal (here +Z,
        // off the flat z = 0 quad). along_normal = 0.0 is the free 3-D jitter,
        // which also moves in the in-plane (x, y) directions.
        let (_mn, _gn, normal_only) = fractal_cut(
            SubdivideParams::default()
                .fractal(1.0)
                .along_normal(1.0)
                .seed(1),
        );
        let (_mf, _gf, free_3d) = fractal_cut(
            SubdivideParams::default()
                .fractal(1.0)
                .along_normal(0.0)
                .seed(1),
        );

        // normal-only: the chord midpoint (1,0,0) moves purely in z.
        assert!(
            (normal_only.x - 1.0).abs() < 1e-6 && normal_only.y.abs() < 1e-6,
            "along_normal = 1 keeps x,y on the chord, got {normal_only:?}"
        );
        assert!(
            normal_only.z.abs() > 1e-5,
            "along_normal = 1 still displaces along the normal (z), got {normal_only:?}"
        );

        // free-3D moves off the chord in the tangent plane too, so it differs
        // from the normal-only run.
        assert!(
            free_3d.distance(normal_only) > 1e-5,
            "along_normal = 0 free-3D differs from normal-only"
        );
        // The z (normal) component is shared — along_normal only attenuates the
        // tangential push, never the normal one (§9.2.4).
        assert!(
            (free_3d.z - normal_only.z).abs() < 1e-6,
            "the normal-direction component is untouched by along_normal"
        );
    }

    #[test]
    fn use_sphere_then_fractal_composes() {
        // §9.2.6 — fractal is applied *after* the §9.4 sphere snap, so the two
        // compose into a roughened dome (they are not mutually exclusive). The
        // displaced vert is off both the chord and the clean sphere point.
        let mut mesh = LoopMesh::new();
        let face = sphere_cage_quad(&mut mesh);
        let fv = mesh.face_vert_keys(face);
        let e01 = mesh.query_edge_exists(fv[0], fv[1]).unwrap();

        let params = SubdivideParams::default()
            .smooth(SPHERE_R)
            .use_sphere(true)
            .fractal(1.0)
            .seed(1);
        let geom = match mesh.subdivide_edges(&[e01], params) {
            SubdivideResult::Ran(g) => g,
            _ => panic!("expected Ran"),
        };
        let pos = mesh.vertex_position(geom.verts[0]);

        // The clean sphere point (no fractal) is (1,1,0) normalised to radius √3.
        let sphere_point = Vec3::new(1.0, 1.0, 0.0).normalize() * SPHERE_R;
        assert!(
            pos.distance(sphere_point) > 1e-4,
            "fractal displaces off the sphere point, got {pos:?}"
        );
        assert_eq!(
            counts(&mesh),
            (5, 7, 3),
            "sphere + fractal is topology-invariant"
        );

        // Self-check: the placement is exactly the §9.2 displacement applied to
        // the sphere point (the base position fractal reads, §9.2.3 step 2 /
        // §9.2.6). The cut edge's basis (length, angle-weighted endpoint normals)
        // is read from a clean reference mesh *before* the re-split moves things,
        // matching what the operator captures pre-split.
        let offset = {
            let mut rng = fractal_noise::SeededRng::new_hashed(1);
            Vec3::new(
                rng.next_float() * 200.0,
                rng.next_float() * 200.0,
                rng.next_float() * 200.0,
            )
        };
        let mut reference = LoopMesh::new();
        let ref_face = sphere_cage_quad(&mut reference);
        let ref_fv = reference.face_vert_keys(ref_face);
        reference.mesh_normals_update();
        let ctx = FractalEdgeContext {
            edge_len: (reference.vertex_position(ref_fv[0]) - reference.vertex_position(ref_fv[1]))
                .length(),
            no_a: reference.angle_weighted_vertex_normal(ref_fv[0]),
            no_b: reference.angle_weighted_vertex_normal(ref_fv[1]),
            // Inert for this midpoint self-check (only edge_len / normals read).
            va: ref_fv[0],
            co_a: reference.vertex_position(ref_fv[0]),
            vb: ref_fv[1],
            co_b: reference.vertex_position(ref_fv[1]),
        };
        let expected = fractal_displace(sphere_point, offset, &ctx, &params);
        assert!(
            pos.distance(expected) < 1e-5,
            "sphere+fractal matches §9.2 arithmetic: got {pos:?}, want {expected:?}"
        );
    }

    #[test]
    fn fractal_displaces_cut_edge_endpoints() {
        // §9.2.3.3 — the displacement runs on the cut edge's two pre-existing
        // ENDPOINT verts, not only the inserted midpoint. Cut edge {0,1} runs
        // from (0,0,0) to (2,0,0); after a non-zero fractal both corners must
        // have moved off their input positions. The far corners (2,3) of the
        // quad are NOT on the cut edge, so they stay put.
        let mut mesh = LoopMesh::new();
        let face = unit_quad(&mut mesh);
        let fv = mesh.face_vert_keys(face);
        let (v0, v1, v2, v3) = (fv[0], fv[1], fv[2], fv[3]);
        let e01 = mesh.query_edge_exists(v0, v1).unwrap();
        let params = SubdivideParams::default().fractal(1.0).seed(1);
        match mesh.subdivide_edges(&[e01], params) {
            SubdivideResult::Ran(_) => {}
            _ => panic!("expected Ran"),
        }

        // The two cut-edge endpoints are jittered off their input positions.
        assert!(
            mesh.vertex_position(v0).distance(Vec3::new(0.0, 0.0, 0.0)) > 1e-4,
            "cut-edge endpoint v0 displaced, got {:?}",
            mesh.vertex_position(v0)
        );
        assert!(
            mesh.vertex_position(v1).distance(Vec3::new(2.0, 0.0, 0.0)) > 1e-4,
            "cut-edge endpoint v1 displaced, got {:?}",
            mesh.vertex_position(v1)
        );
        // Corners off the cut edge are untouched.
        assert!(
            mesh.vertex_position(v2).distance(Vec3::new(2.0, 2.0, 0.0)) < 1e-9,
            "non-cut corner v2 stays put"
        );
        assert!(
            mesh.vertex_position(v3).distance(Vec3::new(0.0, 2.0, 0.0)) < 1e-9,
            "non-cut corner v3 stays put"
        );

        // The endpoints read their OWN pristine input position (§9.2.3.3): the
        // displacement reproduces fractal_displace evaluated at (0,0,0) / (2,0,0)
        // with the cut edge's frame (flat +z, edge_len 2, factor inert).
        let offset = {
            let mut rng = fractal_noise::SeededRng::new_hashed(1);
            Vec3::new(
                rng.next_float() * 200.0,
                rng.next_float() * 200.0,
                rng.next_float() * 200.0,
            )
        };
        let ctx = FractalEdgeContext {
            edge_len: 2.0,
            no_a: Vec3::Z,
            no_b: Vec3::Z,
            va: v0,
            co_a: Vec3::new(0.0, 0.0, 0.0),
            vb: v1,
            co_b: Vec3::new(2.0, 0.0, 0.0),
        };
        let want_v0 = fractal_displace(Vec3::new(0.0, 0.0, 0.0), offset, &ctx, &params);
        let want_v1 = fractal_displace(Vec3::new(2.0, 0.0, 0.0), offset, &ctx, &params);
        assert!(
            mesh.vertex_position(v0).distance(want_v0) < 1e-6,
            "v0 matches the §9.2.3.3 endpoint arithmetic"
        );
        assert!(
            mesh.vertex_position(v1).distance(want_v1) < 1e-6,
            "v1 matches the §9.2.3.3 endpoint arithmetic"
        );
    }

    #[test]
    fn fractal_zero_leaves_endpoints_unmoved() {
        // §9.2.3.3 contrast — with fractal = 0 the deferred displacement never
        // runs, so the cut edge's endpoints stay exactly on their input
        // positions and the core stays bit-identical (no endpoint move).
        let mut mesh = LoopMesh::new();
        let face = unit_quad(&mut mesh);
        let fv = mesh.face_vert_keys(face);
        let (v0, v1) = (fv[0], fv[1]);
        let e01 = mesh.query_edge_exists(v0, v1).unwrap();
        match mesh.subdivide_edges(&[e01], SubdivideParams::default()) {
            SubdivideResult::Ran(_) => {}
            _ => panic!("expected Ran"),
        }
        assert!(
            mesh.vertex_position(v0).distance(Vec3::new(0.0, 0.0, 0.0)) < 1e-9,
            "fractal = 0 leaves endpoint v0 on its input position"
        );
        assert!(
            mesh.vertex_position(v1).distance(Vec3::new(2.0, 0.0, 0.0)) < 1e-9,
            "fractal = 0 leaves endpoint v1 on its input position"
        );
    }

    #[test]
    fn fractal_shared_endpoint_last_writer_wins() {
        // §9.2.3.3 overwrite-not-accumulate — on the flat equal-length quad a
        // corner shared by two cut edges of identical length and identical +z
        // averaged normal collapses to a single application (the last-writer
        // ambiguity vanishes). Cut two edges sharing corner v1; v1's displaced
        // position must equal a single fractal_displace from its pristine input,
        // NOT twice the offset (which an accumulating port would produce).
        let mut mesh = LoopMesh::new();
        let face = unit_quad(&mut mesh);
        let fv = mesh.face_vert_keys(face);
        let (v0, v1, v2) = (fv[0], fv[1], fv[2]);
        let e01 = mesh.query_edge_exists(v0, v1).unwrap();
        let e12 = mesh.query_edge_exists(v1, v2).unwrap();
        let params = SubdivideParams::default().fractal(1.0).seed(3);
        match mesh.subdivide_edges(&[e01, e12], params) {
            SubdivideResult::Ran(_) => {}
            _ => panic!("expected Ran"),
        }

        let offset = {
            let mut rng = fractal_noise::SeededRng::new_hashed(3);
            Vec3::new(
                rng.next_float() * 200.0,
                rng.next_float() * 200.0,
                rng.next_float() * 200.0,
            )
        };
        // Both incident edges have length 2, +z normal: the frame is identical,
        // so the single-application result is the same regardless of writer.
        let ctx = FractalEdgeContext {
            edge_len: 2.0,
            no_a: Vec3::Z,
            no_b: Vec3::Z,
            va: v1,
            co_a: Vec3::new(2.0, 0.0, 0.0),
            vb: v2,
            co_b: Vec3::new(2.0, 2.0, 0.0),
        };
        let single = fractal_displace(Vec3::new(2.0, 0.0, 0.0), offset, &ctx, &params);
        assert!(
            mesh.vertex_position(v1).distance(single) < 1e-6,
            "shared corner v1 is a single (last-writer) application, not accumulated: \
             got {:?}, want {single:?}",
            mesh.vertex_position(v1)
        );
    }

    #[test]
    fn tangent_frame_closed_form_and_degenerate_branch() {
        // §9.2.3.4 — the exact closed-form tangent basis.

        // Non-degenerate: a tilted normal with a non-trivial xy-projection.
        let normal = Vec3::new(0.6, 0.8, 0.5);
        let (base1, base2) = tangent_frame(normal);
        let f = normal.x * normal.x + normal.y * normal.y; // 0.36 + 0.64 = 1.0
        let d = 1.0 / f.sqrt();
        let want_base1 = Vec3::new(normal.y * d, -normal.x * d, 0.0);
        let want_base2 = Vec3::new(normal.x * normal.z * d, normal.y * normal.z * d, -f.sqrt());
        assert!(base1.distance(want_base1) < 1e-6, "base1 closed form");
        assert!(base2.distance(want_base2) < 1e-6, "base2 closed form");
        // base1 is UNIT; base2 has length |normal| (NOT renormalised).
        assert!((base1.length() - 1.0).abs() < 1e-6, "base1 is unit");
        assert!(
            (base2.length() - normal.length()).abs() < 1e-5,
            "base2 length == |normal|, got {} vs {}",
            base2.length(),
            normal.length()
        );
        // base1 lies in the xy-plane (z = 0).
        assert!(base1.z.abs() < 1e-7, "base1 is in the xy-plane");

        // Degenerate +z: f <= EPSILON => base1 = (1,0,0), base2 = (0,1,0).
        let (b1, b2) = tangent_frame(Vec3::Z);
        assert!(
            b1.distance(Vec3::new(1.0, 0.0, 0.0)) < 1e-9,
            "+z base1 = +x"
        );
        assert!(
            b2.distance(Vec3::new(0.0, 1.0, 0.0)) < 1e-9,
            "+z base2 = +y"
        );

        // Degenerate -z: the x-sign of base1 flips with nz < 0.
        let (b1n, b2n) = tangent_frame(Vec3::new(0.0, 0.0, -1.0));
        assert!(
            b1n.distance(Vec3::new(-1.0, 0.0, 0.0)) < 1e-9,
            "-z base1 = -x"
        );
        assert!(
            b2n.distance(Vec3::new(0.0, 1.0, 0.0)) < 1e-9,
            "-z base2 = +y"
        );
    }

    #[test]
    fn connect_targeting_picks_the_fewest_corners_face() {
        // §7.0 — the primitive itself. `a` and `c` are carried by *both* faces of
        // the pillow; the 4-corner quad beats the 5-corner pentagon regardless of
        // disk order. A pair no face carries resolves to nothing, and laying it is
        // a silent no-op rather than an error.
        let (mut mesh, [a, _b, c, d, e, _f], _e_ab, _e_bc) = quad_pentagon_shared_two_edges();

        let target = mesh
            .smallest_face_with_both(a, c)
            .expect("both faces carry the pair");
        assert_eq!(mesh.face_len(target), 4, "the quad wins on corner count");

        assert!(
            mesh.smallest_face_with_both(d, e).is_none(),
            "no face carries a quad-only and a pentagon-only corner"
        );
        let before = counts(&mesh);
        assert!(mesh.template_connect(d, e).is_none(), "untargeted connect");
        assert_eq!(
            counts(&mesh),
            before,
            "the no-op creates no edge, no fragment"
        );
    }

    #[test]
    fn shared_two_edges_single_cut_template_targets_smaller_face() {
        // §7.0 — cut only the shared edge {a, b}. The quad takes the §5.1 one-edge
        // template; its second connect `m — c` is carried by the quad fragment
        // {m, b, c, d} (4 corners) *and* by the pentagon (6 corners after the cut).
        // Fewest corners wins, so both connects stay inside the quad and the
        // pentagon is left flat (it has one cut edge and no template).
        // V 6->7, E 7->10, F 2->4.
        let (mut mesh, [_a, b, c, _d, _e, _f], e_ab, _e_bc) = quad_pentagon_shared_two_edges();

        let res = mesh.subdivide_edges(&[e_ab], SubdivideParams::default());
        let geom = match res {
            SubdivideResult::Ran(g) => g,
            _ => panic!("subdivide ran"),
        };
        assert_eq!(
            counts(&mesh),
            (7, 10, 4),
            "the quad splits in three; the pentagon does not split"
        );

        let m = geom.verts[0];
        let mut shapes: Vec<usize> = mesh.face_keys().iter().map(|&f| mesh.face_len(f)).collect();
        shapes.sort_unstable();
        assert_eq!(
            shapes,
            vec![3, 3, 3, 6],
            "three quad fragments + the intact 6-ring pentagon"
        );

        // The discriminator: `m — c` must separate two *quad* fragments. Targeting
        // the first face carrying the pair instead lays it inside the pentagon,
        // which then re-enters the dispatch as a 4-corner ring and over-splits.
        let mc = mesh
            .query_edge_exists(m, c)
            .expect("the m - c connect was laid");
        let mut radial: Vec<usize> = mesh
            .edge_face_keys(mc)
            .iter()
            .map(|&f| mesh.face_len(f))
            .collect();
        radial.sort_unstable();
        assert_eq!(
            radial,
            vec![3, 3],
            "m - c splits the quad, not the pentagon"
        );

        let pentagon = mesh
            .face_keys()
            .into_iter()
            .find(|&f| mesh.face_len(f) == 6)
            .expect("the pentagon survives");
        let pentagon_verts = mesh.face_vert_keys(pentagon);
        assert!(
            pentagon_verts.contains(&m) && pentagon_verts.contains(&b),
            "the pentagon keeps the shared cut's midpoint on its rim, un-connected"
        );
    }

    #[test]
    fn two_edge_fallback_drops_all_its_connects_when_a_second_face_carries_the_pair() {
        // §7.1's dropped-connect precondition, on its smallest witness. Both faces
        // of this pillow are two-edge fallback rows (§5.2.1 for the quad, §6.2 for
        // the triangle) and *both* cut edges bound both faces — so on either face,
        // the endpoints of the candidate connect are corners of a face other than
        // the one being re-split. Both faces therefore drop, symmetrically: the
        // rule is not a tiebreak, and the mesh gains two rim midpoints and **not
        // one connecting edge**. V 6 / E 7 / F 2.
        //
        // Skip the precondition and each parent splits itself instead (V 6 / E 9 /
        // F 4) — which also mints small fragments that go on to win carrier
        // contests they should never have entered (the two tests below).
        let (mut mesh, _corners, e_ab, e_bc) = quad_triangle_shared_two_edges();

        let geom = match mesh.subdivide_edges(&[e_ab, e_bc], SubdivideParams::default()) {
            SubdivideResult::Ran(g) => g,
            _ => panic!("subdivide ran"),
        };

        assert_eq!(counts(&mesh), (6, 7, 2), "two rim midpoints, no connect");

        let (m_ab, m_bc) = (geom.verts[0], geom.verts[1]);
        assert!(
            mesh.query_edge_exists(m_ab, m_bc).is_none(),
            "neither face may claim the pair the other also carries"
        );

        let mut shapes: Vec<usize> = mesh.face_keys().iter().map(|&f| mesh.face_len(f)).collect();
        shapes.sort_unstable();
        assert_eq!(
            shapes,
            vec![5, 6],
            "both parents keep their rings, cut verts flat on the rim"
        );

        assert_eq!(
            geom.faces.len(),
            2,
            "geom.out: both were recorded in pass one, and a drop revokes no mark"
        );
        assert_eq!(
            geom.geom_inner().edges.len(),
            0,
            "a dropped connect is not laid, so it contributes to no slot"
        );
    }

    #[test]
    fn dropped_connect_precondition_holds_for_every_cut_at_higher_n() {
        // §7.1 — the verdict is uniform across a face's `N` candidate pairs (they
        // all have one endpoint on each of the same two cut edges), so a face lays
        // all `N` connects or none. At `cuts = 3` on the quad/pentagon pillow that
        // is 3 + 3 connects dropped: F stays 2, and a partially stripped face
        // (which §7.1 says is unreachable) would show up as any other face count.
        let (mut mesh, _corners, e_ab, e_bc) = quad_pentagon_shared_two_edges();

        let geom = match mesh.subdivide_edges(&[e_ab, e_bc], SubdivideParams::default().cuts(3)) {
            SubdivideResult::Ran(g) => g,
            _ => panic!("subdivide ran"),
        };

        assert_eq!(
            counts(&mesh),
            (6 + 6, 7 + 6, 2),
            "six rim cut verts, both cut edges quartered, and no connect at all"
        );
        assert_eq!(
            geom.geom_inner().edges.len(),
            0,
            "all-or-nothing: not one of the six candidate connects is laid"
        );
    }

    #[test]
    fn template_connect_drops_an_adjacent_pair_rather_than_mint_a_two_gon() {
        // §7.0.2 — the project's 2-gon witness. Cut only the shared edge {a, b}.
        // Three connects are nominated: the triangle's `m01 - c` (it is face 1, so
        // it applies *first*), the quad's `m01 - d`, and the quad's `m01 - c`. The
        // last one's fewest-corners carrier is one of the two triangles the first
        // connect just made, in which `m01` and `c` are **adjacent** — joined by
        // the very edge that connect laid. The split is refused: no edge, no
        // fragment, no retry on the larger (and perfectly splittable) 4-corner
        // carrier that also holds the pair. V 5 / E 8 / F 4.
        let (mut mesh, [_a, _b, c, _d], e_ab, _e_bc) = quad_triangle_shared_two_edges();

        let geom = match mesh.subdivide_edges(&[e_ab], SubdivideParams::default()) {
            SubdivideResult::Ran(g) => g,
            _ => panic!("subdivide ran"),
        };

        assert_eq!(
            counts(&mesh),
            (5, 8, 4),
            "three connects nominated, two laid, one dropped by adjacency"
        );
        assert!(
            mesh.face_keys().iter().all(|&f| mesh.face_len(f) >= 3),
            "the dropped connect must not mint a degenerate 2-gon"
        );

        // The pair itself is joined by exactly one edge — the one the triangle's
        // connect laid. A second (duplicate) edge over {m01, c} is the signature
        // of an implementation that split the adjacent-holding carrier anyway.
        let m01 = geom.verts[0];
        let mut spans = 0;
        let mut disk = mesh.disk_edges(m01);
        while let Some(edge) = disk.next(&mesh) {
            if mesh.edge_vert_keys(edge).contains(&c) {
                spans += 1;
            }
        }
        assert_eq!(spans, 1, "one m01 - c edge, laid by the triangle");

        let mut shapes: Vec<usize> = mesh.face_keys().iter().map(|&f| mesh.face_len(f)).collect();
        shapes.sort_unstable();
        assert_eq!(shapes, vec![3, 3, 3, 4], "no 2-gon among the fragments");
    }

    #[test]
    fn carved_away_template_is_abandoned_wholesale() {
        // §7.0.3 — the equal-corner-count tie. Both 5-rings carry `B`'s first
        // chord `m01 - v2`, the tie falls to the face the fan of `m01` yields
        // first (`A`, attached to the cut edge first), and `B`'s own connect
        // therefore carves its *neighbour* down to the triangle {m01, v1, v2}.
        // When `A`'s template finally runs, the ring it re-reads is a 3-ring its
        // 5-ring pattern was never matched for: the template lays **nothing** —
        // not even the `m01 - v3` chord, whose carrier at that moment would hold
        // it on a perfectly splittable diagonal.
        //
        // Three rules have to be right to land V 6 / E 9 / F 4, and each failure
        // gives E 10 / F 5 instead: the reverse apply order, the tie-break, and
        // abandonment (a snapshotted pair list lays `m01 - v3`).
        let (mut mesh, [v0, v1, v2, v3, v4], e01) = quad_quad_shared_two_edges();

        let geom = match mesh.subdivide_edges(&[e01], SubdivideParams::default()) {
            SubdivideResult::Ran(g) => g,
            _ => panic!("subdivide ran"),
        };
        let m = geom.verts[0];

        assert_eq!(
            counts(&mesh),
            (6, 9, 4),
            "only two of the four nominated connects exist"
        );

        // The fragment set is the load-bearing assertion: all three failure modes
        // share the same V/E/F *and* the same face-shape multiset.
        let sorted = |vs: &[VertexKey]| {
            let mut ids: Vec<usize> = vs.iter().map(|v| v.as_usize()).collect();
            ids.sort_unstable();
            ids
        };
        let mut fragments: Vec<Vec<usize>> = mesh
            .face_keys()
            .into_iter()
            .map(|f| sorted(&mesh.face_vert_keys(f)))
            .collect();
        fragments.sort_unstable();

        let mut expected: Vec<Vec<usize>> = vec![
            sorted(&[m, v1, v2]),     // what `A` shrank to
            sorted(&[v0, m, v2, v3]), // the fragment `B`'s chord carved off `A`
            sorted(&[m, v0, v4]),     // `B`'s second chord
            sorted(&[v1, m, v4, v2]), // what `B` kept
        ];
        expected.sort_unstable();
        assert_eq!(fragments, expected, "the pinned fragment set");

        // `m01 - v3` was never nominated, so no edge spans that pair.
        assert!(
            mesh.query_edge_exists(m, v3).is_none(),
            "the abandoned template lays nothing at all"
        );
    }

    #[test]
    fn resplit_mark_is_stamped_on_record_not_on_work_laid() {
        // §1 / §4.4 — the mark means "the dispatcher recorded this face", never
        // "this face was re-split", and it is never revoked. Two faces exercise
        // the two ways that comes apart, and both are pinned in `geom.out`.
        //
        // (a) An **abandoned** template. On the tie fixture, `A`'s template lays
        // nothing (it was carved to a triangle before its turn) — and still
        // carries its mark, alongside `B`, which did split.
        let (mut mesh, _corners, e01) = quad_quad_shared_two_edges();
        let geom = match mesh.subdivide_edges(&[e01], SubdivideParams::default()) {
            SubdivideResult::Ran(g) => g,
            _ => panic!("subdivide ran"),
        };
        assert_eq!(
            geom.geom().faces.len(),
            2,
            "both 5-rings were recorded, so both are marked — even the abandoned one"
        );

        // (b) A **template-less** face the fallback then declines: the pentagon of
        // the pillow has one cut edge, and no pattern claims an n-gon at any cut
        // count other than two. It is recorded (its cut count is non-zero), so it
        // is marked, and it stays a flat-cornered 6-gon.
        let (mut mesh, _corners, e_ab, _e_bc) = quad_pentagon_shared_two_edges();
        let geom = match mesh.subdivide_edges(&[e_ab], SubdivideParams::default()) {
            SubdivideResult::Ran(g) => g,
            _ => panic!("subdivide ran"),
        };
        let pentagon = mesh
            .face_keys()
            .into_iter()
            .find(|&f| mesh.face_len(f) == 6)
            .expect("the pentagon survives, un-re-split");
        assert!(
            geom.geom().faces.contains(&pentagon),
            "'marked' and 're-split' come apart: a one-cut n-gon is marked and left flat"
        );
        assert_eq!(geom.geom().faces.len(), 2, "one mark per recorded face");
    }

    #[test]
    fn flat_corner_guard_keeps_the_mark_off_the_fallback_only() {
        // §1 / §4.3.1 — the one thing that keeps the mark off a cut face is the
        // guard, and only on a face no template claimed. The same flat-cornered
        // quad under INNERVERT matches a template, so it is recorded, marked, and
        // re-split (V 7 / E 9 / F 3) — with `c` landing exactly on `v1` and the
        // corner fragment zero-area, which is the correct pinned output (§8).
        let mut mesh = LoopMesh::new();
        let face = flat_corner_quad(&mut mesh);
        let fv = mesh.face_vert_keys(face);
        let e01 = mesh.query_edge_exists(fv[0], fv[1]).unwrap();
        let e12 = mesh.query_edge_exists(fv[1], fv[2]).unwrap();

        let geom = match mesh.subdivide_edges(
            &[e01, e12],
            SubdivideParams::default().quad_corner_type(QuadCornerType::InnerVert),
        ) {
            SubdivideResult::Ran(g) => g,
            _ => panic!("subdivide ran"),
        };
        assert_eq!(
            counts(&mesh),
            (7, 9, 3),
            "the guard never gates a template row"
        );
        assert_eq!(
            geom.geom().faces.len(),
            1,
            "a guard-hit *template* face is recorded and marked all the same"
        );
    }

    #[test]
    fn dropped_connect_precondition_is_upstream_of_a_template_carrier_contest() {
        // §7.1's box — the precondition reaches beyond the fallback's own fragment
        // count. Cut both shared edges of the pillow under INNERVERT. The triangle
        // (face 1, so applied first, §7.0.3(a)) is a §6.2 fallback and **drops**:
        // its two cut edges have a second common face, the quad. It therefore
        // stays a whole 5-ring, and mints no small fragments.
        //
        // The quad's INNERVERT template then nominates the same `m_ab - m_bc` pair.
        // §7.0.1's search is mesh-wide and adjacency-blind, so it takes the
        // fewest-corners carrier — the untouched triangle (5 corners against the
        // quad's own 6) — which holds the pair non-adjacent, on either side of `b`.
        // The corner cut lands **in the neighbour**, `c` is born there, and the
        // quad's own ring is never touched: its `c - v3` spoke has no carrier at
        // all (nothing holds both `c` and `d`), so it is a §7.0.2 no-op and the
        // quad comes out a whole hexagon. V 7 / E 9 / F 3.
        //
        // Had the triangle split itself, that same contest would have been decided
        // by *its* 3-ring fragment instead — a different mesh. Which is exactly why
        // §7.1 says to implement the precondition before chasing a carrier
        // divergence: a missing precondition looks like a mis-targeted template.
        let (mut mesh, [_a, b, _c, d], e_ab, e_bc) = quad_triangle_shared_two_edges();

        let geom = match mesh.subdivide_edges(
            &[e_ab, e_bc],
            SubdivideParams::default().quad_corner_type(QuadCornerType::InnerVert),
        ) {
            SubdivideResult::Ran(g) => g,
            _ => panic!("subdivide ran"),
        };

        assert_eq!(
            counts(&mesh),
            (7, 9, 3),
            "two rim midpoints plus `c`; the corner cut halved by `c`; the triangle split"
        );

        let mut shapes: Vec<usize> = mesh.face_keys().iter().map(|&f| mesh.face_len(f)).collect();
        shapes.sort_unstable();
        assert_eq!(
            shapes,
            vec![4, 5, 6],
            "the quad stays a whole hexagon; the carved triangle is the corner quad + a 5-ring"
        );

        // `c` was born on the corner cut, so no *direct* m_ab - m_bc edge survives,
        // and `c` is a corner of the neighbour's fragments — never of the quad.
        let (m_ab, m_bc) = (geom.verts[0], geom.verts[1]);
        assert!(
            mesh.query_edge_exists(m_ab, m_bc).is_none(),
            "the corner cut was split by `c`"
        );
        assert_eq!(
            geom.inner_verts.len(),
            3,
            "two rim midpoints and the one inner vert `c`"
        );
        let c_inner = geom.inner_verts[2];
        assert!(
            mesh.query_edge_exists(c_inner, d).is_none(),
            "the far-corner spoke has no carrier: no face holds both `c` and `v3`"
        );

        let hexagon = mesh
            .face_keys()
            .into_iter()
            .find(|&f| mesh.face_len(f) == 6)
            .expect("the quad survives as a hexagon");
        let hexagon_verts = mesh.face_vert_keys(hexagon);
        assert!(
            hexagon_verts.contains(&b) && !hexagon_verts.contains(&c_inner),
            "the template's own face was never carved: `c` is a corner of the neighbour's \
             fragments alone"
        );

        assert_eq!(
            geom.faces.len(),
            2,
            "both faces were recorded in pass one, and the drop revokes no mark"
        );
    }

    #[test]
    fn re_inflated_pending_face_is_abandoned_not_re_split() {
        // §7.0.3 — the re-inflation channel, and the *only* clause that closes it.
        // The mirror of the test above: the same pillow, but the triangle is face
        // 0, so pass three (reverse order) applies the **quad** first and leaves
        // the triangle pending.
        //
        // The quad's INNERVERT corner cut `m_ab - m_bc` finds its fewest-corners
        // carrier in the triangle (a 5-ring, against the quad's own 6-ring) and
        // carves it: the triangle retains (m_ab, a, c, m_bc) — 4 corners, with the
        // original corner `b` peeled off into the new fragment. Step 2 then splits
        // that chord to give birth to `c_in`, which rejoins the triangle's ring.
        //
        // The triangle is now back at 5 corners, and every quantity pass one
        // recorded about it still checks out: length 5 = 3 + 1x2 = expected_len;
        // pass-two midpoint census 2 = its cut count; original-corner count 3 =
        // its pre_len. The anchor survives too — the carve peeled `b`, not the
        // corner the scan happened to anchor on, and *which* corner that is falls
        // out of the face's stored ring start, so the anchor clause is a coin-flip
        // on winding here. Only "my ring carries a vert pass three minted" tells
        // the truth: the triangle abandons, and lays nothing.
        //
        // A second `m_ab - m_bc` edge (laid straight across the m_ab - c_in - m_bc
        // chain, on a ring the §6.2 pattern was never matched for) is the exact
        // signature of the gate that misses this. V 7 / E 9 / F 3.
        let (mut mesh, [_a, _b, _c, _d], e_ab, e_bc) = triangle_quad_shared_two_edges();

        let geom = match mesh.subdivide_edges(
            &[e_ab, e_bc],
            SubdivideParams::default().quad_corner_type(QuadCornerType::InnerVert),
        ) {
            SubdivideResult::Ran(g) => g,
            _ => panic!("subdivide ran"),
        };

        assert_eq!(
            counts(&mesh),
            (7, 9, 3),
            "the quad's corner cut + its inner vert, and nothing from the triangle"
        );

        let (m_ab, m_bc) = (geom.verts[0], geom.verts[1]);
        assert!(
            mesh.query_edge_exists(m_ab, m_bc).is_none(),
            "the corner cut was split by `c`, so no direct m_ab - m_bc edge may exist: \
             one here is the abandoned template laying its pattern anyway"
        );

        let mut shapes: Vec<usize> = mesh.face_keys().iter().map(|&f| mesh.face_len(f)).collect();
        shapes.sort_unstable();
        assert_eq!(
            shapes,
            vec![4, 5, 6],
            "the quad stays a whole hexagon, the carved triangle a re-inflated 5-ring, \
             and the peeled corner a 4-ring — no spurious fragment"
        );

        assert_eq!(
            geom.inner_verts.len(),
            3,
            "two rim midpoints and the one inner vert `c`"
        );
        assert_eq!(
            geom.faces.len(),
            2,
            "both faces were recorded, and abandonment revokes no mark"
        );
    }

    #[test]
    fn grid_fill_bar_lands_in_the_neighbour_the_precondition_left_whole() {
        // §7.1's box / §5.4 — the same lesson as the INNERVERT pillow, on the grid
        // fill's *bar*. `A` (face 0) has all four edges cut and is a §5.4 grid
        // fill. `B` (face 1) shares exactly two of those cut edges, opposite in its
        // own ring, so it is a §5.2 fallback — and it **drops**: both its cut edges
        // bound `A` as well. Applying first (pass three is reversed), it lays
        // nothing and stays a whole 6-ring.
        //
        // `A`'s bar `m01 - m23` then goes to the fewest-corners carrier, which is
        // that intact 6-ring `B` (6 < A's own 8), and it holds the pair on a
        // diagonal: the bar is laid **in the neighbour**, the centre vert `c` is
        // born there, and `A`'s two remaining spokes have no carrier (nothing holds
        // both `c` and `m12` / `m30`), so they are §7.0.2 no-ops. `A` therefore
        // comes out a whole octagon with no grid at all. V 9 / E 12 / F 3.
        let (mut mesh, [v0, v1, v2, v3]) = quad_quad_shared_two_opposite_edges();
        let a = mesh.face_keys()[0];
        let b = mesh.face_keys()[1];
        let edges = [
            mesh.query_edge_exists(v0, v1).unwrap(),
            mesh.query_edge_exists(v1, v2).unwrap(),
            mesh.query_edge_exists(v2, v3).unwrap(),
            mesh.query_edge_exists(v3, v0).unwrap(),
        ];

        let geom = match mesh.subdivide_edges(&edges, SubdivideParams::default()) {
            SubdivideResult::Ran(g) => g,
            _ => panic!("subdivide ran"),
        };

        assert_eq!(
            counts(&mesh),
            (9, 12, 3),
            "four rim midpoints, the centre vert, and the halved bar — all inside B"
        );
        assert_eq!(
            mesh.face_len(a),
            8,
            "A keeps its whole octagon ring: its bar was laid in the neighbour"
        );
        assert_eq!(
            geom.inner_verts.len(),
            5,
            "four rim midpoints plus the one centre vert the bar's split minted"
        );
        let centre = geom.inner_verts[4];
        assert!(
            !mesh.face_vert_keys(a).contains(&centre),
            "the centre vert is a corner of the neighbour's fragments, never of A"
        );
        assert!(
            mesh.face_keys().iter().all(|&f| mesh.face_len(f) >= 3),
            "no degenerate 2-gon anywhere"
        );
        assert!(
            geom.faces.contains(&a) && geom.faces.contains(&b),
            "both faces were recorded — dropping every connect revokes no re-split mark (§4.4)"
        );
        assert_eq!(
            geom.faces.len(),
            3,
            "the two re-split marks, plus the one fragment the bar minted inside B \
             (§1: a grid fill's fragments are inner faces; the one that kept B's handle \
             is already the mark)"
        );
    }

    /// §7.1 — the shape that arbitrates *when* the dropped-connect precondition is
    /// decided. `G` is a quad `(v0, v1, v2, v3)` with all four edges cut, so it is
    /// a §5.4 grid fill. `F` is a **7-gon** sharing exactly `G`'s two *opposite*
    /// edges `{v0,v1}` and `{v2,v3}`, its other five being its own — so `F` has
    /// `cut_count = 2`, matches no template, and is the §7.1 fallback. Both shared
    /// edges keep radial length 2; nothing here is degenerate.
    ///
    /// The 7-gon is the load-bearing choice. Post-cut `F` carries 9 corners against
    /// `G`'s 8, so `G` wins the carrier contest for its own `m01 - m23` bar
    /// (§7.0.1, strictly fewest corners) and `F` is never carved — while `G`'s grid
    /// leaves **no** fragment holding both `m01` and `m23`. The precondition's two
    /// readings therefore come apart: on the *input topology* (§7.1's reduction)
    /// `F`'s two cut edges have a second common face, namely `G`, so `F` drops. Ask
    /// the *live* mesh at apply time instead and the answer flips with the apply
    /// schedule — pass three applies in reverse face-index order (§4.4), so
    /// whichever face was created second goes first.
    ///
    /// `create_grid_first` flips exactly that. Returns `(mesh, [v0..v3], f)`.
    fn grid_quad_and_ngon_on_two_opposite_edges(
        create_grid_first: bool,
    ) -> (LoopMesh, [VertexKey; 4], FaceKey) {
        let mut mesh = LoopMesh::new();
        let v0 = mesh.create_vertex(Vec3::new(0.0, 0.0, 0.0));
        let v1 = mesh.create_vertex(Vec3::new(2.0, 0.0, 0.0));
        let v2 = mesh.create_vertex(Vec3::new(2.0, 2.0, 0.0));
        let v3 = mesh.create_vertex(Vec3::new(0.0, 2.0, 0.0));
        // The 7-gon's own three corners, folded off the quad's plane so it reads as
        // a pocket hanging from the two shared edges.
        let p = mesh.create_vertex(Vec3::new(3.5, -0.5, -1.5));
        let q = mesh.create_vertex(Vec3::new(3.5, 2.5, -1.5));
        let r = mesh.create_vertex(Vec3::new(-1.5, 1.0, -1.5));

        let e01 = mesh.create_edge([v0, v1], EdgeCreateFlags::NONE);
        let e12 = mesh.create_edge([v1, v2], EdgeCreateFlags::NONE);
        let e23 = mesh.create_edge([v2, v3], EdgeCreateFlags::NONE);
        let e30 = mesh.create_edge([v3, v0], EdgeCreateFlags::NONE);
        let e2q = mesh.create_edge([v2, q], EdgeCreateFlags::NONE);
        let eqp = mesh.create_edge([q, p], EdgeCreateFlags::NONE);
        let ep1 = mesh.create_edge([p, v1], EdgeCreateFlags::NONE);
        let e0r = mesh.create_edge([v0, r], EdgeCreateFlags::NONE);
        let er3 = mesh.create_edge([r, v3], EdgeCreateFlags::NONE);

        // `F` walks the two shared edges against `G`'s winding, so the pair is
        // consistently wound and every radial cycle stays length 2.
        let f_verts = [v3, v2, q, p, v1, v0, r];
        let f_edges = [e23, e2q, eqp, ep1, e01, e0r, er3];
        let g_verts = [v0, v1, v2, v3];
        let g_edges = [e01, e12, e23, e30];

        let f = if create_grid_first {
            mesh.face_create(&g_verts, &g_edges, FaceCreateFlags::NONE);
            mesh.face_create(&f_verts, &f_edges, FaceCreateFlags::NONE)
        } else {
            let f = mesh.face_create(&f_verts, &f_edges, FaceCreateFlags::NONE);
            mesh.face_create(&g_verts, &g_edges, FaceCreateFlags::NONE);
            f
        };
        (mesh, [v0, v1, v2, v3], f)
    }

    #[test]
    fn the_strip_precondition_is_a_function_of_the_input_not_the_apply_order() {
        // §7.1 — the precondition reduces to "the face's two cut edges have a
        // second common face", a property of the mesh the operator was *handed*.
        // `F` shares both of its cut edges with `G`, so `F` drops its connect and
        // stays whole — whichever order the two faces were created in, and hence
        // whichever order pass three applies them in.
        //
        // Both orders: `G` grid-fills into 4 quads (its bar's carrier is itself, 8
        // corners against `F`'s 9), `F` stays a single 9-ring.
        //   V 12 = 4 quad corners + 3 of the 7-gon's own + 4 rim midpoints + centre
        //   E 17 = 8 halved rim + 5 private + the halved bar + 2 spokes
        //   F  5 = the grid's 4 + the whole 7-gon
        for create_grid_first in [true, false] {
            let (mut mesh, [v0, v1, v2, v3], f) =
                grid_quad_and_ngon_on_two_opposite_edges(create_grid_first);
            let edges = [
                mesh.query_edge_exists(v0, v1).unwrap(),
                mesh.query_edge_exists(v1, v2).unwrap(),
                mesh.query_edge_exists(v2, v3).unwrap(),
                mesh.query_edge_exists(v3, v0).unwrap(),
            ];

            let geom = match mesh.subdivide_edges(&edges, SubdivideParams::default()) {
                SubdivideResult::Ran(g) => g,
                _ => panic!("subdivide ran"),
            };

            assert_eq!(
                counts(&mesh),
                (12, 17, 5),
                "create_grid_first = {create_grid_first}: the grid's 4 fragments and \
                 the whole 7-gon — the strip laid no chord"
            );
            assert_eq!(
                mesh.face_len(f),
                9,
                "create_grid_first = {create_grid_first}: the 7-gon keeps its whole \
                 ring, its two rim midpoints as flat corners"
            );
            assert_eq!(
                geom.inner_edges.len(),
                4,
                "create_grid_first = {create_grid_first}: the grid's four spokes and \
                 nothing from the dropped strip"
            );
            assert!(
                geom.faces.contains(&f),
                "create_grid_first = {create_grid_first}: the dropped 7-gon keeps its \
                 re-split mark — a drop revokes no mark (§4.4)"
            );
            assert_eq!(
                geom.faces.len(),
                5,
                "create_grid_first = {create_grid_first}: the two re-split marks plus \
                 the three grid fragments that did not keep the quad's handle (§1)"
            );

            let mut shapes: Vec<usize> = mesh.face_keys().iter().map(|&f| mesh.face_len(f)).collect();
            shapes.sort_unstable();
            assert_eq!(
                shapes,
                vec![4, 4, 4, 4, 9],
                "create_grid_first = {create_grid_first}: a 7-gon split into 2 here is \
                 the strip reading the live mesh instead of the input topology"
            );
        }
    }

    /// A convex pentagon, wide enough that no rim corner is anywhere near
    /// straight (so the §4.3.1 guard never fires on it).
    fn unit_pentagon(mesh: &mut LoopMesh) -> FaceKey {
        create_face_from_verts(
            mesh,
            &[
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(2.0, 0.0, 0.0),
                Vec3::new(3.0, 2.0, 0.0),
                Vec3::new(1.0, 3.0, 0.0),
                Vec3::new(-1.0, 2.0, 0.0),
            ],
        )
    }

    #[test]
    fn ngon_two_edges_strip_at_single_cut() {
        // §7.1 / §4.3 — an n-gon has no template at any cut count, so a pentagon
        // with exactly two cut edges is the generic strip's, and at `cuts = 1` the
        // strip is one connect: V 5->7 (2 rim cut verts), E 5->8 (2 halved + 1
        // connect), 1 pentagon -> 2 fragments.
        let mut mesh = LoopMesh::new();
        let face = unit_pentagon(&mut mesh);
        let fv = mesh.face_vert_keys(face);
        let e01 = mesh.query_edge_exists(fv[0], fv[1]).unwrap();
        let e23 = mesh.query_edge_exists(fv[2], fv[3]).unwrap();

        let geom = match mesh.subdivide_edges(&[e01, e23], SubdivideParams::default()) {
            SubdivideResult::Ran(g) => g,
            _ => panic!("subdivide ran"),
        };

        assert_eq!(counts(&mesh), (7, 8, 2), "pentagon, two cut edges: N + 1 = 2");

        let m01 = find_vert_at(&mesh, Vec3::new(1.0, 0.0, 0.0));
        let m23 = find_vert_at(&mesh, Vec3::new(2.0, 2.5, 0.0));
        assert!(
            mesh.query_edge_exists(m01, m23).is_some(),
            "the strip's lone connect joins the two cut verts"
        );

        let mut shapes: Vec<usize> = mesh.face_keys().iter().map(|&f| mesh.face_len(f)).collect();
        shapes.sort_unstable();
        assert_eq!(shapes, vec![4, 5], "the connect bands the rim into 4-gon + 5-gon");

        // §7.1 output-slot note — the fallback is the one non-grid path that
        // inner-marks a connecting edge.
        assert_eq!(
            (
                geom.geom_inner().verts.len(),
                geom.geom_inner().edges.len(),
                geom.geom_inner().faces.len()
            ),
            (2, 1, 0),
            "geom_inner.out is 2N verts / N edges / 0 faces at N = 1"
        );
    }

    #[test]
    fn ngon_one_edge_is_never_re_split() {
        // §4.3 — the corollary that keeps the fallback honest: it fires at
        // `cut_count = 2` and nowhere else. A pentagon with *one* cut edge (at any
        // `cuts`) is left as a single flat-cornered n-gon, never fanned.
        for cuts in [1, 3] {
            let mut mesh = LoopMesh::new();
            let face = unit_pentagon(&mut mesh);
            let fv = mesh.face_vert_keys(face);
            let e01 = mesh.query_edge_exists(fv[0], fv[1]).unwrap();

            mesh.subdivide_edges(&[e01], SubdivideParams::default().cuts(cuts));

            let n = cuts as usize;
            assert_eq!(
                counts(&mesh),
                (5 + n, 5 + n, 1),
                "one-cut n-gon stays flat at cuts = {cuts}"
            );
        }
    }

    #[test]
    fn quad_two_opposite_edges_cuts3_strip() {
        // §9.5.3.3 — the strip at `cuts = 3`: three connects, four fragments, and
        // no odd/even asymmetry. V 4->10, E 4->13 (8 segments + 3 connects), F 4.
        let mut mesh = LoopMesh::new();
        let face = unit_quad(&mut mesh);
        let fv = mesh.face_vert_keys(face);
        let e01 = mesh.query_edge_exists(fv[0], fv[1]).unwrap();
        let e23 = mesh.query_edge_exists(fv[2], fv[3]).unwrap();

        let geom = match mesh.subdivide_edges(&[e01, e23], SubdivideParams::default().cuts(3)) {
            SubdivideResult::Ran(g) => g,
            _ => panic!("subdivide ran"),
        };

        assert_eq!(counts(&mesh), (10, 13, 4), "two-opposite quad at N = 3");

        // The outside-in zip is the whole content of the pattern, and this is what
        // discriminates it: the three connects run *straight across* the quad,
        // pairing each bottom cut vert with the top one directly above it. A
        // same-index zip would cross them into an hourglass instead.
        for x in [0.5f32, 1.0, 1.5] {
            let bottom = find_vert_at(&mesh, Vec3::new(x, 0.0, 0.0));
            let top = find_vert_at(&mesh, Vec3::new(x, 2.0, 0.0));
            assert!(
                mesh.query_edge_exists(bottom, top).is_some(),
                "the strip pairs the runs outside-in, so the connect at x = {x} is vertical"
            );
        }

        assert!(
            mesh.face_keys().iter().all(|&f| mesh.face_len(f) == 4),
            "four quad bands"
        );
        assert_eq!(
            (
                geom.geom_inner().verts.len(),
                geom.geom_inner().edges.len(),
                geom.geom_inner().faces.len()
            ),
            (6, 3, 0),
            "geom_inner.out is 2N verts / N edges / 0 faces at N = 3"
        );
        assert_eq!(geom.faces.len(), 1, "geom.out: one re-split mark");
    }

    #[test]
    fn tri_two_edges_cuts3_strip() {
        // §9.5.3.3 — the same strip on a triangle, whose two cut edges *share* a
        // corner: the runs still zip outside-in, giving N + 1 = 4 fragments (a
        // corner triangle plus three bands). V 3->9, E 3->12, F 4.
        let mut mesh = LoopMesh::new();
        let face = unit_tri(&mut mesh);
        let fv = mesh.face_vert_keys(face);
        let e01 = mesh.query_edge_exists(fv[0], fv[1]).unwrap();
        let e12 = mesh.query_edge_exists(fv[1], fv[2]).unwrap();

        let geom = match mesh.subdivide_edges(&[e01, e12], SubdivideParams::default().cuts(3)) {
            SubdivideResult::Ran(g) => g,
            _ => panic!("subdivide ran"),
        };

        assert_eq!(counts(&mesh), (9, 12, 4), "two-edge triangle at N = 3");

        // The innermost connect walls the shared corner v1 off as a triangle; the
        // other two band the rest. Outside-in pairing means the cut vert nearest
        // v1 on one edge joins the one nearest v1 on the other.
        let mut shapes: Vec<usize> = mesh.face_keys().iter().map(|&f| mesh.face_len(f)).collect();
        shapes.sort_unstable();
        assert_eq!(shapes, vec![3, 4, 4, 4], "corner triangle + three bands");

        assert_eq!(
            (
                geom.geom_inner().verts.len(),
                geom.geom_inner().edges.len(),
                geom.geom_inner().faces.len()
            ),
            (6, 3, 0),
            "the strip inner-marks each connect it lays"
        );
    }

    #[test]
    fn quad_two_adjacent_straight_cut_cuts2_strip() {
        // §9.5.3.3 — the default `STRAIGHT_CUT` two-adjacent quad is the same
        // fallback at `cuts = 2`: V 8 / E 10 / F 3, two connects, three fragments.
        // The guard's verdict is threaded through this row, so a *flat*-cornered
        // quad in the same configuration stays whole (the contrast below).
        let mut mesh = LoopMesh::new();
        let face = unit_quad(&mut mesh);
        let fv = mesh.face_vert_keys(face);
        let e01 = mesh.query_edge_exists(fv[0], fv[1]).unwrap();
        let e12 = mesh.query_edge_exists(fv[1], fv[2]).unwrap();

        mesh.subdivide_edges(&[e01, e12], SubdivideParams::default().cuts(2));
        assert_eq!(counts(&mesh), (8, 10, 3), "adjacent STRAIGHT_CUT at N = 2");

        let mut mesh = LoopMesh::new();
        let face = flat_corner_quad(&mut mesh);
        let fv = mesh.face_vert_keys(face);
        let e01 = mesh.query_edge_exists(fv[0], fv[1]).unwrap();
        let e12 = mesh.query_edge_exists(fv[1], fv[2]).unwrap();

        let geom = match mesh.subdivide_edges(&[e01, e12], SubdivideParams::default().cuts(2)) {
            SubdivideResult::Ran(g) => g,
            _ => panic!("subdivide ran"),
        };
        assert_eq!(
            counts(&mesh),
            (8, 8, 1),
            "the guard is cuts-independent: the flat-cornered strip lays nothing"
        );
        // §4.3.1 / §4.4 — and the limb that actually carries that behaviour is the
        // *dispatch scan*, which refuses to record a guarded template-less face at
        // all. The absent re-split mark is what pins it: the apply pass's own
        // `flat_corner` read is belt-and-braces on a face that never gets there.
        assert!(
            geom.geom().faces.is_empty(),
            "a guarded fallback face is never recorded, so it carries no re-split mark"
        );
        assert!(
            !geom.geom().faces.contains(&face),
            "the guarded face itself is absent from geom.out"
        );
    }

    #[test]
    fn ngon_two_edges_strip_at_multicut() {
        // §7.1 / §9.5.3.3 — the row's own intersection: an n-gon (no template at
        // any cut count) at `cuts >= 2`. The pentagon's two cut edges give two runs
        // of N, zipped outside-in into N connects and N + 1 fragments — the strip
        // is face-size *and* N agnostic.
        for cuts in [2u32, 3] {
            let n = cuts as usize;
            let mut mesh = LoopMesh::new();
            let face = unit_pentagon(&mut mesh);
            let fv = mesh.face_vert_keys(face);
            let e01 = mesh.query_edge_exists(fv[0], fv[1]).unwrap();
            let e23 = mesh.query_edge_exists(fv[2], fv[3]).unwrap();

            let geom = match mesh.subdivide_edges(&[e01, e23], SubdivideParams::default().cuts(cuts))
            {
                SubdivideResult::Ran(g) => g,
                _ => panic!("subdivide ran"),
            };

            assert_eq!(
                counts(&mesh),
                (5 + 2 * n, 5 + 2 * n + n, n + 1),
                "pentagon, two cut edges at cuts = {cuts}: N + 1 fragments"
            );
            assert_eq!(
                (
                    geom.geom_inner().verts.len(),
                    geom.geom_inner().edges.len(),
                    geom.geom_inner().faces.len()
                ),
                (2 * n, n, 0),
                "geom_inner.out is 2N verts / N edges / 0 faces"
            );
            assert_eq!(geom.faces.len(), 1, "geom.out: one re-split mark");
        }

        // The zip's discriminator at N = 2, read off the geometry: the pentagon's
        // cut edges run x: 0->2 (y = 0) and x: 3->1 (y = 2 -> 3). Outside-in pairs
        // the cut vert nearest v0 on the first run with the one nearest v3 on the
        // second, so the two connects do not cross.
        let mut mesh = LoopMesh::new();
        let face = unit_pentagon(&mut mesh);
        let fv = mesh.face_vert_keys(face);
        let e01 = mesh.query_edge_exists(fv[0], fv[1]).unwrap();
        let e23 = mesh.query_edge_exists(fv[2], fv[3]).unwrap();
        mesh.subdivide_edges(&[e01, e23], SubdivideParams::default().cuts(2));

        let near_v0 = find_vert_at(&mesh, Vec3::new(2.0 / 3.0, 0.0, 0.0));
        let near_v1 = find_vert_at(&mesh, Vec3::new(4.0 / 3.0, 0.0, 0.0));
        let near_v2 = find_vert_at(&mesh, Vec3::new(3.0 - 2.0 / 3.0, 2.0 + 1.0 / 3.0, 0.0));
        let near_v3 = find_vert_at(&mesh, Vec3::new(3.0 - 4.0 / 3.0, 2.0 + 2.0 / 3.0, 0.0));
        assert!(
            mesh.query_edge_exists(near_v0, near_v3).is_some()
                && mesh.query_edge_exists(near_v1, near_v2).is_some(),
            "the runs zip outside-in; a same-index zip would cross the two connects"
        );
    }

    #[test]
    fn corner_strategy_claims_only_the_adjacent_two_cut_quad_at_multicut() {
        // §5.2 / §9.5.3.3 — the `quad_corner_type` templates claim the *adjacent*
        // two-cut quad and nothing else, and the adjacency predicate is what draws
        // the line. Their multi-cut arms are a separate parity row and not yet
        // implemented, so a claimed face stays flat at `cuts = 2`...
        for corner_type in [
            QuadCornerType::Fan,
            QuadCornerType::Path,
            QuadCornerType::InnerVert,
        ] {
            let mut mesh = LoopMesh::new();
            let face = unit_quad(&mut mesh);
            let fv = mesh.face_vert_keys(face);
            let e01 = mesh.query_edge_exists(fv[0], fv[1]).unwrap();
            let e12 = mesh.query_edge_exists(fv[1], fv[2]).unwrap();

            mesh.subdivide_edges(
                &[e01, e12],
                SubdivideParams::default()
                    .cuts(2)
                    .quad_corner_type(corner_type),
            );
            assert_eq!(
                counts(&mesh),
                (8, 8, 1),
                "{corner_type:?} claims the adjacent two-cut quad, so the strip must not \
                 take it: the face stays a flat-cornered octagon"
            );

            // ...while the *opposite* two-cut quad is claimed by no strategy at any
            // cut count, so it drops onto the strip regardless: two connects, three
            // fragments. A predicate that answered "adjacent" unconditionally would
            // leave this face flat too.
            let mut mesh = LoopMesh::new();
            let face = unit_quad(&mut mesh);
            let fv = mesh.face_vert_keys(face);
            let e01 = mesh.query_edge_exists(fv[0], fv[1]).unwrap();
            let e23 = mesh.query_edge_exists(fv[2], fv[3]).unwrap();

            mesh.subdivide_edges(
                &[e01, e23],
                SubdivideParams::default()
                    .cuts(2)
                    .quad_corner_type(corner_type),
            );
            assert_eq!(
                counts(&mesh),
                (8, 10, 3),
                "{corner_type:?} claims nothing here: the opposite two-cut quad is the \
                 strip's at N = 2"
            );
        }
    }

    #[test]
    fn flat_corner_verdict_is_stamped_on_the_face() {
        // §4.3.1, white-box — the guard's verdict layer itself, independent of any
        // `cuts` value or dispatch consumer. `push_flat_corner_guard` decides on
        // the *input* edge directions, and the FLAT_CORNER bit is what the
        // fallback rows read.
        let straight = |mesh: &mut LoopMesh, edges: &[EdgeKey], face: FaceKey| -> bool {
            let (guard_layer, cut_layer, touched) = mesh.push_flat_corner_guard(edges);
            let verdict = mesh.face_op_flag_test(guard_layer, face, FLAT_CORNER);
            assert!(touched.contains(&face), "the guard enumerates the cut faces");
            mesh.pop_edge_op_flag_layer(cut_layer);
            mesh.pop_face_op_flag_layer(guard_layer);
            verdict
        };

        // The two edges flanking the straight (180°) corner v1: collinear, guarded.
        let mut mesh = LoopMesh::new();
        let face = flat_corner_quad(&mut mesh);
        let fv = mesh.face_vert_keys(face);
        let e01 = mesh.query_edge_exists(fv[0], fv[1]).unwrap();
        let e12 = mesh.query_edge_exists(fv[1], fv[2]).unwrap();
        let e23 = mesh.query_edge_exists(fv[2], fv[3]).unwrap();
        assert!(
            straight(&mut mesh, &[e01, e12], face),
            "collinear cut edges sharing the straight corner are flagged"
        );
        // Opposite cut edges share no vertex, so no corner exists to be straight.
        assert!(
            !straight(&mut mesh, &[e01, e23], face),
            "opposite cut edges never trip the guard"
        );

        // 5° off straight is a real corner, inside the same face shape.
        let mut mesh = LoopMesh::new();
        let face = near_flat_corner_quad(&mut mesh, 5.0f32.to_radians());
        let fv = mesh.face_vert_keys(face);
        let e01 = mesh.query_edge_exists(fv[0], fv[1]).unwrap();
        let e12 = mesh.query_edge_exists(fv[1], fv[2]).unwrap();
        assert!(
            !straight(&mut mesh, &[e01, e12], face),
            "5° off straight is outside the collinearity tolerance"
        );
    }

}
