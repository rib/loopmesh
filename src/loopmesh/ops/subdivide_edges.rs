use glam::Vec3;
use smallvec::SmallVec;

use crate::loopmesh::*;

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
/// - [`Self::geom_inner`] (`geom_inner.out`) — the geometry created *inside* a
///   re-split face: the inner edges the pattern added, plus (for the four-cut
///   quad §5.4, the §5.2.2 INNERVERT corner, and the multi-cut lattices §9.5)
///   the inner vert(s).
/// - [`Self::geom`] (`geom.out`) — the headline union: every vert, edge, and
///   face the operator created or replaced. The original re-split faces keep
///   their handle and also appear here.
///
/// The [`Self::verts`] / [`Self::faces`] fields are the `geom.out` vert and
/// face sets (their long-standing meaning); [`Self::edges`] completes that
/// union with the `geom.out` edge set.
#[derive(Default, Clone, Debug, PartialEq, Eq)]
pub struct SubdivideGeom {
    /// `geom.out` verts: every midpoint vert inserted on a cut edge (§3), plus
    /// any inner vert a per-face pattern added (the §5.4 quad centre, the
    /// §5.2.2 INNERVERT corner, the §9.5 lattice interior).
    pub verts: Vec<VertexKey>,
    /// `geom.out` edges: every edge the operator created — the edge-halves of
    /// the split input edges (§3) and the inner edges laid by the re-split
    /// patterns (§5 / §6 / §9.5).
    pub edges: Vec<EdgeKey>,
    /// `geom.out` faces: every face produced by a re-split (§5 / §6 / §9.5). The
    /// original faces that were re-split keep their handle and also appear here.
    pub faces: Vec<FaceKey>,
    /// `geom_split.out` verts: the new midpoint verts produced by splitting the
    /// input edges (§3). A subset of [`Self::verts`].
    pub split_verts: Vec<VertexKey>,
    /// `geom_split.out` edges: the edge-halves produced by splitting the input
    /// edges (§3). A subset of [`Self::edges`].
    pub split_edges: Vec<EdgeKey>,
    /// `geom_inner.out` verts: the inner vert(s) a per-face pattern added (§5.4
    /// quad centre, §5.2.2 INNERVERT corner, §9.5 lattice interior). A subset of
    /// [`Self::verts`].
    pub inner_verts: Vec<VertexKey>,
    /// `geom_inner.out` edges: the inner edges a per-face re-split pattern added
    /// inside a face (§5 / §6 / §9.5). A subset of [`Self::edges`].
    pub inner_edges: Vec<EdgeKey>,
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

    /// `geom_inner.out` (§1): the inner edges (and inner vert, where a pattern
    /// adds one) created *inside* a re-split face. No faces.
    pub fn geom_inner(&self) -> SubdivideSlot<'_> {
        SubdivideSlot {
            verts: &self.inner_verts,
            edges: &self.inner_edges,
            faces: &[],
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
    /// edge into both `geom.out` and `geom_split.out` (§3). `original_half` is
    /// the input-edge handle that survives the split (the near-side half),
    /// `new_half` the freshly-created far-side half.
    fn record_split(
        &mut self,
        midpoint: VertexKey,
        original_half: EdgeKey,
        new_half: EdgeKey,
    ) {
        self.verts.push(midpoint);
        self.split_verts.push(midpoint);
        self.push_split_edge(original_half);
        self.push_split_edge(new_half);
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

        // §9.1.1 — the off-chord offset reads the cut edge's endpoint vertex
        // normals, which must be current before placement. Freshen all cached
        // normals once up front; the chord-only core (`smooth == 0`) skips this
        // and stays bit-identical to the no-displacement path.
        if params.smooth != 0.0 {
            self.mesh_normals_update();
        }

        // A vert op-flag layer tags the freshly-inserted midpoints so phase two
        // can tell original corners from inserted midpoints when it walks a
        // face's loop cycle (§3.1 / §4.1).
        let midpoint_layer = self.push_vert_op_flag_layer();
        const MIDPOINT: u8 = 1;

        // ---- Phase one: per-edge midpoint insertion (§3) ----
        // Each input edge is split once at t = 0.5. The edge_split wrapper does
        // the position lerp and the per-vert / per-edge / per-loop linear
        // interpolation (§7.2 along-edge blend) for us.
        let mut touched_faces: Vec<FaceKey> = Vec::new();
        for &edge in edges {
            let [near, _far] = self.edge_vert_keys(edge);
            // §9.1 — snapshot the endpoint positions and vertex normals *before*
            // the split (the split rewrites the edge's far endpoint). `t = 0.5`
            // is the §3.2 directional cut parameter, measured from `near`. The
            // §9.1 offset is skipped under `use_sphere` (§9.4.3, sphere wins).
            let placement = (!params.use_sphere && params.smooth != 0.0)
                .then(|| self.smooth_placement(edge, near, 0.5, &params));
            let (m, e_new) = self.edge_split(edge, near, 0.5);
            if let Some(co_final) = placement {
                self.set_vertex_position(m, co_final);
                self.vert_normal_update(m);
            }
            // §9.4.1 — radial snap onto the origin-centred sphere of radius
            // `smooth`, taken from the inserted midpoint's chord position.
            self.maybe_sphere_project(m, &params);
            self.vert_op_flag_set(midpoint_layer, m, MIDPOINT);
            // §1 — the midpoint and the two edge-halves (`edge` survives as the
            // near-side half, `e_new` is the far-side half) populate geom_split.
            geom.record_split(m, edge, e_new);
            for f in self.vert_face_keys(m) {
                if !touched_faces.contains(&f) {
                    touched_faces.push(f);
                }
            }
        }

        // ---- Phase two: per-face pattern dispatch (§4) ----
        for face in touched_faces {
            self.resplit_face(face, midpoint_layer, MIDPOINT, &params, &mut geom);
        }

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

    /// Dispatch and apply the §5 / §6 re-split pattern for one touched face.
    fn resplit_face(
        &mut self,
        face: FaceKey,
        layer: crate::op_flags::VertOpFlagLayer,
        midpoint_flag: u8,
        params: &SubdivideParams,
        geom: &mut SubdivideGeom,
    ) {
        // Walk the loop cycle once, recording each corner's vert and whether it
        // is a freshly-inserted midpoint, in cyclic order (§4.1).
        let mut ring: SmallVec<[(VertexKey, bool); 12]> = SmallVec::new();
        let mut it = self.face_loops(face);
        while let Some(l) = it.next(self) {
            let v = self.loops.get(l).vertex;
            let is_mid = self.vert_op_flag_test(layer, v, midpoint_flag);
            ring.push((v, is_mid));
        }

        let cut_count = ring.iter().filter(|(_, m)| *m).count();
        let corner_count = ring.len() - cut_count; // original (non-midpoint) corners

        if cut_count == 0 {
            return; // never visited
        }

        // §9.3 — with use_only_quads on, the per-face dispatch is restricted to
        // quads. A touched triangle or n-gon keeps its §3 midpoint(s) but is
        // left flat (a dangling 180° rim corner, no interior connection). The
        // gate filters this pass only; §3 already inserted the midpoint.
        if params.use_only_quads && corner_count != 4 {
            return;
        }

        match (corner_count, cut_count) {
            (3, 1) => {
                if params.use_single_edge {
                    self.tri_one_cut(face, &ring, geom);
                }
            }
            (3, 2) => {
                // §6.2 — triangle, two edges cut. A single shared-corner cut
                // peels a small triangle and leaves a quad. Always fires (no
                // gating flag — it is neither a one-cut nor an all-cut case).
                self.tri_two_cut(face, &ring, geom);
            }
            (3, 3) => {
                if params.use_grid_fill {
                    self.tri_three_cut(face, &ring, geom);
                }
            }
            (4, 1) => {
                if params.use_single_edge {
                    self.quad_one_cut(face, &ring, geom);
                }
            }
            (4, 3) => {
                // §5.3 — quad, three edges cut. The two corner triangles plus a
                // pentagon-diagonal cut; no gating flag (neither one-cut nor
                // all-cut).
                self.quad_three_cut(face, &ring, geom);
            }
            (4, 4) => {
                if params.use_grid_fill {
                    self.quad_four_cut(face, &ring, params, geom);
                }
            }
            (4, 2) if Self::quad_two_cut_is_opposite(&ring) => {
                // §5.2 opposite case — the straight loop cut. Always fires
                // (no strategy choice, no gating flag).
                self.quad_two_opposite_cut(face, &ring, geom);
            }
            (4, 2) => {
                // §5.2 adjacent case. The strategy chooses the interior
                // geometry laid down once the two rim midpoints exist; all four
                // share the §5.2.1 shared-corner anchor and far corner `v3`.
                match params.quad_corner_type {
                    QuadCornerType::StraightCut => {
                        self.quad_two_adjacent_straight_cut(face, &ring, geom)
                    }
                    QuadCornerType::InnerVert => {
                        self.quad_two_adjacent_inner_vert(face, &ring, params, geom)
                    }
                    QuadCornerType::Path => self.quad_two_adjacent_path(face, &ring, geom),
                    QuadCornerType::Fan => self.quad_two_adjacent_fan(face, &ring, geom),
                }
            }
            // Three-edge quads, the n-gon generic, and the deferred corner
            // strategies are out of the core's pinned scope (§4.3 / §9); leave
            // the flat-cornered face untouched.
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

    /// The sub-face of the just-split region that contains both `a` and `b`.
    /// After [`Self::face_split`] keeps the input face as one half and returns
    /// the other, the next split must target whichever half still holds the
    /// pair it connects.
    fn face_with_both(&self, a: VertexKey, b: VertexKey) -> FaceKey {
        for f in self.vert_face_keys(a) {
            if self.face_vert_keys(f).contains(&b) {
                return f;
            }
        }
        panic!("subdivide_edges: no face shares the split pair");
    }

    /// §6.1 — triangle, one edge cut. Cycle is `v0, m, v1, v2`; connect the
    /// midpoint `m` to the opposite apex `v2`. 1 tri -> 2 tris.
    fn tri_one_cut(&mut self, face: FaceKey, ring: &[(VertexKey, bool)], geom: &mut SubdivideGeom) {
        let (v, _) = Self::rotate_to_first_midpoint(ring);
        // After rotation: v[0] = m, v[1] = v1, v[2] = v2, v[3] = v0. The apex
        // opposite the cut edge {v0,v1} is v[2].
        let (m, apex) = (v[0], v[2]);
        if let Some((new_face, inner)) = self.face_split(face, m, apex) {
            geom.faces.push(face);
            geom.faces.push(new_face);
            geom.record_inner_edge(inner);
        }
    }

    /// §6.2 — triangle, two edges cut. The two cut edges share corner `v1`
    /// (`{v0,v1}` and `{v1,v2}`), midpoints `m01` and `m12`. After the §3
    /// insertion the loop cycle is a pentagon `v0, m01, v1, m12, v2`. A single
    /// face-split connects `m01` → `m12`, carving the shared corner off as a
    /// small triangle `{m01, v1, m12}` and leaving a quad `{v0, m01, m12, v2}`.
    /// 1 tri -> 2 fragments (1 triangle + 1 quad).
    fn tri_two_cut(&mut self, face: FaceKey, ring: &[(VertexKey, bool)], geom: &mut SubdivideGeom) {
        // The two midpoints are the only two cut corners and always sit two
        // apart in the five-corner ring (the shared corner v1 between them).
        let n = ring.len(); // 5
        let mids: SmallVec<[usize; 2]> = ring
            .iter()
            .enumerate()
            .filter(|(_, (_, m))| *m)
            .map(|(i, _)| i)
            .collect();
        debug_assert_eq!(mids.len(), 2, "two-cut tri has exactly two midpoints");
        // Start at the midpoint that has the other exactly two corners ahead,
        // so v1 (the shared corner) sits between them on the forward arc.
        let start = if (mids[1] - mids[0]) == 2 {
            mids[0]
        } else {
            mids[1]
        };
        let m01 = ring[start].0;
        let m12 = ring[(start + 2) % n].0;
        debug_assert!(
            !ring[(start + 1) % n].1,
            "shared corner is original, not a midpoint"
        );

        if let Some((new_face, inner)) = self.face_split(face, m01, m12) {
            geom.faces.push(face);
            geom.faces.push(new_face);
            geom.record_inner_edge(inner);
        }
    }

    /// §6.3 — triangle, three edges cut. Cycle is `v0, m01, v1, m12, v2, m20`;
    /// connect the three midpoints pairwise into the centre triangle.
    /// 1 tri -> 4 tris.
    fn tri_three_cut(
        &mut self,
        face: FaceKey,
        ring: &[(VertexKey, bool)],
        geom: &mut SubdivideGeom,
    ) {
        let (v, _) = Self::rotate_to_first_midpoint(ring);
        // v = [m01, v1, m12, v2, m20, v0].
        let (m01, m12, m20) = (v[0], v[2], v[4]);

        // m01 -> m12 peels the corner triangle {m01, v1, m12}.
        let f1 = self.face_with_both(m01, m12);
        let (corner_a, inner_a) = self.face_split(f1, m01, m12).expect("m01-m12 split");
        // m12 -> m20 peels {m12, v2, m20} from the remaining pentagon.
        let f2 = self.face_with_both(m12, m20);
        let (corner_b, inner_b) = self.face_split(f2, m12, m20).expect("m12-m20 split");
        // m20 -> m01 closes the centre triangle, peeling {m20, v0, m01}.
        let f3 = self.face_with_both(m20, m01);
        let (corner_c, inner_c) = self.face_split(f3, m20, m01).expect("m20-m01 split");

        geom.faces.push(face);
        geom.faces.push(corner_a);
        geom.faces.push(corner_b);
        geom.faces.push(corner_c);
        geom.record_inner_edge(inner_a);
        geom.record_inner_edge(inner_b);
        geom.record_inner_edge(inner_c);
    }

    /// §5.1 — quad, one edge cut. Cycle is `v0, m, v1, v2, v3`; connect `m` into
    /// the interior to the two opposite corners `v3` and `v2`. 1 quad -> 3
    /// triangular fragments.
    fn quad_one_cut(
        &mut self,
        face: FaceKey,
        ring: &[(VertexKey, bool)],
        geom: &mut SubdivideGeom,
    ) {
        let (v, _) = Self::rotate_to_first_midpoint(ring);
        // v = [m, v1, v2, v3, v0]. The cut edge was {v0, v1}; the two interior
        // targets are v3 = v[3] and v2 = v[2].
        let (m, v2, v3) = (v[0], v[2], v[3]);
        // m -> v3 peels {v0, m, v3}.
        let (frag_a, inner_a) = self.face_split(face, m, v3).expect("m-v3 split");
        // m -> v2 splits the remaining quad {m, v1, v2, v3}.
        let f2 = self.face_with_both(m, v2);
        let (frag_b, inner_b) = self.face_split(f2, m, v2).expect("m-v2 split");

        geom.faces.push(face);
        geom.faces.push(frag_a);
        geom.faces.push(frag_b);
        geom.record_inner_edge(inner_a);
        geom.record_inner_edge(inner_b);
    }

    /// §5.2 — discriminate the two-cut quad's sub-case. After the §3 midpoint
    /// insertion a quad with two cut edges is a six-corner ring whose two
    /// midpoint corners sit either diametrically opposite (the §5.2 *opposite*
    /// loop-cut case) or two apart on the short side (the §5.2 *adjacent*
    /// corner-cut case). The cut edges share a quad corner iff their midpoints
    /// are *not* diametrically opposite in the ring.
    fn quad_two_cut_is_opposite(ring: &[(VertexKey, bool)]) -> bool {
        let n = ring.len(); // 6 for a two-cut quad
        let mids: SmallVec<[usize; 2]> = ring
            .iter()
            .enumerate()
            .filter(|(_, (_, m))| *m)
            .map(|(i, _)| i)
            .collect();
        debug_assert_eq!(mids.len(), 2, "two-cut quad has exactly two midpoints");
        let gap = mids[1] - mids[0];
        gap == n / 2
    }

    /// §5.2 opposite case — the straight loop cut. The two cut edges are
    /// opposite (`{v0,v1}` and `{v2,v3}`), midpoints `m01` and `m23`. After the
    /// §3 insertion the loop cycle is `v0, m01, v1, v2, m23, v3`; connect the
    /// two midpoints straight across, splitting the quad into two quads. No
    /// inner vert, no strategy choice. 1 quad -> 2 quads.
    fn quad_two_opposite_cut(
        &mut self,
        face: FaceKey,
        ring: &[(VertexKey, bool)],
        geom: &mut SubdivideGeom,
    ) {
        let (v, _) = Self::rotate_to_first_midpoint(ring);
        // v = [m01, v1, v2, m23, v3, v0]; the two opposite midpoints are v[0]
        // and v[3] (diametrically opposite in the six-corner ring).
        let (m01, m23) = (v[0], v[3]);
        if let Some((new_face, inner)) = self.face_split(face, m01, m23) {
            geom.faces.push(face);
            geom.faces.push(new_face);
            geom.record_inner_edge(inner);
        }
    }

    /// §5.2.1 — quad, two **adjacent** edges cut, default `STRAIGHT_CUT`. The
    /// two cut edges share corner `v1` (`{v0,v1}` and `{v1,v2}`), midpoints
    /// `m01` and `m12`. After the §3 insertion the loop cycle is a hexagon
    /// `v0, m01, v1, m12, v2, v3` with `m01` and `m12` two corners apart on the
    /// short arc and `v1` sitting between them. A single face-split connects
    /// `m01` → `m12`, carving the shared corner off as a triangle `{m01, v1,
    /// m12}` and leaving a pentagon `{v0, m01, m12, v2, v3}`. No inner vert.
    /// 1 quad -> 2 fragments (1 triangle + 1 pentagon).
    fn quad_two_adjacent_straight_cut(
        &mut self,
        face: FaceKey,
        ring: &[(VertexKey, bool)],
        geom: &mut SubdivideGeom,
    ) {
        // Rotate so the start corner is the midpoint that opens the short arc:
        // the one whose loop successor is the shared corner v1, with the second
        // midpoint two corners further on. The two midpoints sit two apart on
        // one side of the six-corner ring; pick the start so v[2] is the other
        // midpoint (i.e. the two-corner gap runs forward from the start).
        let n = ring.len(); // 6
        let mids: SmallVec<[usize; 2]> = ring
            .iter()
            .enumerate()
            .filter(|(_, (_, m))| *m)
            .map(|(i, _)| i)
            .collect();
        debug_assert_eq!(mids.len(), 2, "two-cut quad has exactly two midpoints");
        // Forward gap from mids[0] to mids[1]; the start is whichever midpoint
        // has the other exactly two corners ahead of it.
        let start = if (mids[1] - mids[0]) == 2 {
            mids[0]
        } else {
            mids[1]
        };

        let m01 = ring[start].0;
        let m12 = ring[(start + 2) % n].0;
        debug_assert!(
            ring[(start + 1) % n].0 != m01,
            "shared corner between midpoints"
        );

        // m01 -> m12 carves the corner triangle off the hexagon.
        if let Some((new_face, inner)) = self.face_split(face, m01, m12) {
            geom.faces.push(face);
            geom.faces.push(new_face);
            geom.record_inner_edge(inner);
        }
    }

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
        face: FaceKey,
        ring: &[(VertexKey, bool)],
        params: &SubdivideParams,
        geom: &mut SubdivideGeom,
    ) {
        let [m01, _v1, m12, _v2, v3, _v0] = Self::adjacent_two_cut_corners(ring);

        // 1. Corner cut m01 -> m12, carving the corner triangle off and leaving
        //    the pentagon (the STRAIGHT_CUT intermediate). The corner cut is an
        //    inner edge (§1 geom_inner).
        let (new_face, corner_cut_edge) = self.face_split(face, m01, m12).expect("m01-m12 split");
        geom.faces.push(face);
        geom.faces.push(new_face);

        // 2. Inner vert c at the midpoint of the m01-m12 cut. The edge_split
        //    wrapper places c at t = 0.5 and applies the §7.2 two-source blend
        //    of the cut's endpoints (m01, m12) — exactly the documented inner-
        //    vert rule, *not* the §5.4 face-wide centroid. The split leaves
        //    `corner_cut_edge` as the m01-c half and yields `c_half` (c-m12);
        //    both are inner edges, as is c itself an inner vert.
        let (c, c_half) = self.edge_split(corner_cut_edge, m01, 0.5);
        // §9.4.2 — the operator-created inner vert is snapped onto the sphere
        // under `use_sphere`, outward from its chord (corner-cut midpoint).
        self.maybe_sphere_project(c, params);
        geom.record_inner_vert(c);
        geom.record_inner_edge(corner_cut_edge);
        geom.record_inner_edge(c_half);

        // 3. Spoke c -> v3, splitting the pentagon into the two filling quads.
        let pent = self.face_with_both(c, v3);
        let (quad_b, spoke) = self.face_split(pent, c, v3).expect("c-v3 spoke");
        geom.faces.push(quad_b);
        geom.record_inner_edge(spoke);
    }

    /// §5.2.3 — quad, two **adjacent** edges cut, `PATH`. Lay the `m01`→`m12`
    /// shared-corner cut, then chain a second `v0`→`v2` cut across the remaining
    /// pentagon. Yields the corner triangle `{m01, v1, m12}`, a central quad
    /// `{v0, m01, m12, v2}`, and the far-corner triangle `{v0, v2, v3}`. No
    /// inner vert. 1 quad -> 3 fragments. V 6 / E 8 / F 3.
    fn quad_two_adjacent_path(
        &mut self,
        face: FaceKey,
        ring: &[(VertexKey, bool)],
        geom: &mut SubdivideGeom,
    ) {
        let [m01, _v1, m12, v2, _v3, v0] = Self::adjacent_two_cut_corners(ring);

        // 1. Corner cut m01 -> m12.
        let (new_face, corner_cut) = self.face_split(face, m01, m12).expect("m01-m12 split");
        geom.faces.push(face);
        geom.faces.push(new_face);
        geom.record_inner_edge(corner_cut);

        // 2. Chained v0 -> v2 cut across the pentagon {v0, m01, m12, v2, v3},
        //    splitting it into the central quad {v0, m01, m12, v2} and the far-
        //    corner triangle {v0, v2, v3}.
        let pent = self.face_with_both(v0, v2);
        let (tri_far, chain_cut) = self.face_split(pent, v0, v2).expect("v0-v2 chained cut");
        geom.faces.push(tri_far);
        geom.record_inner_edge(chain_cut);
    }

    /// §5.2.4 — quad, two **adjacent** edges cut, `FAN`. Spoke both rim
    /// midpoints to the far corner `v3`: `m01`→`v3` then `m12`→`v3`. Yields the
    /// triangle `{v3, v0, m01}`, a central quad `{v3, m01, v1, m12}` (the only
    /// fragment holding the shared corner `v1`), and the triangle `{v3, m12,
    /// v2}`. `v3` becomes a degree-4 fan hub. No inner vert. 1 quad -> 3
    /// fragments. V 6 / E 8 / F 3.
    fn quad_two_adjacent_fan(
        &mut self,
        face: FaceKey,
        ring: &[(VertexKey, bool)],
        geom: &mut SubdivideGeom,
    ) {
        let [m01, _v1, m12, _v2, v3, _v0] = Self::adjacent_two_cut_corners(ring);

        // 1. First spoke m01 -> v3, carving the triangle {v3, v0, m01}.
        let f1 = self.face_with_both(m01, v3);
        let (tri_a, spoke_a) = self.face_split(f1, m01, v3).expect("m01-v3 spoke");
        // 2. Second spoke m12 -> v3, carving the triangle {v3, m12, v2}, leaving
        //    the central quad {v3, m01, v1, m12} between the two spokes.
        let f2 = self.face_with_both(m12, v3);
        let (tri_b, spoke_b) = self.face_split(f2, m12, v3).expect("m12-v3 spoke");

        geom.faces.push(face);
        geom.faces.push(tri_a);
        geom.faces.push(tri_b);
        geom.record_inner_edge(spoke_a);
        geom.record_inner_edge(spoke_b);
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
    fn quad_three_cut(
        &mut self,
        face: FaceKey,
        ring: &[(VertexKey, bool)],
        geom: &mut SubdivideGeom,
    ) {
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
        let is_mid = |i: usize| ring[i % n].1;
        let middle = *mids
            .iter()
            .find(|&&i| is_mid((i + 2) % n) && is_mid((i + n - 2) % n))
            .expect("three-cut quad has a middle midpoint");

        let m12 = ring[middle].0;
        let m01 = ring[(middle + n - 2) % n].0; // end midpoint trailing m12
        let m23 = ring[(middle + 2) % n].0; // end midpoint leading m12

        // m12 -> m01 peels the corner triangle {m01, v1, m12}.
        let f1 = self.face_with_both(m12, m01);
        let (corner_a, inner_a) = self.face_split(f1, m12, m01).expect("m12-m01 split");
        // m12 -> m23 peels the corner triangle {m12, v2, m23}.
        let f2 = self.face_with_both(m12, m23);
        let (corner_b, inner_b) = self.face_split(f2, m12, m23).expect("m12-m23 split");
        // m01 -> m23 splits the remaining pentagon {v0, m01, m12, m23, v3} into
        // the centre triangle {m01, m12, m23} and the quad {v0, m01, m23, v3}.
        let f3 = self.face_with_both(m01, m23);
        let (centre, inner_c) = self.face_split(f3, m01, m23).expect("m01-m23 split");

        geom.faces.push(face);
        geom.faces.push(corner_a);
        geom.faces.push(corner_b);
        geom.faces.push(centre);
        geom.record_inner_edge(inner_a);
        geom.record_inner_edge(inner_b);
        geom.record_inner_edge(inner_c);
    }

    /// §5.4 — quad, all four edges cut. Cycle is
    /// `v0, m01, v1, m12, v2, m23, v3, m30`; insert one inner vert `c` at the
    /// centroid and connect each rim midpoint to it (the "plus" split).
    /// 1 quad -> 2x2 grid (4 quads).
    fn quad_four_cut(
        &mut self,
        face: FaceKey,
        ring: &[(VertexKey, bool)],
        params: &SubdivideParams,
        geom: &mut SubdivideGeom,
    ) {
        let (v, _) = Self::rotate_to_first_midpoint(ring);
        // v = [m01, v1, m12, v2, m23, v3, m30, v0].
        let (m01, m12, m23, m30) = (v[0], v[2], v[4], v[6]);
        let (c0, c1, c2, c3) = (v[7], v[1], v[3], v[5]); // original corners v0..v3

        // Cut the octagon m01 -> m23 across, then split that diagonal at its
        // midpoint to introduce the centre vert c. The diagonal joins two
        // opposite rim midpoints, so its midpoint is the quad centroid.
        let (_hex_b, diag) = self.face_split(face, m01, m23).expect("m01-m23 split");
        let mids = self.edge_split_n(diag, m01, 1);
        let c = mids[0];

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
        // spokes (m01-c and m23-c already exist as the halved diagonal).
        let f_m12 = self.face_with_both(c, m12);
        let (q_a, _) = self.face_split(f_m12, c, m12).expect("c-m12 split");
        let f_m30 = self.face_with_both(c, m30);
        let (q_b, _) = self.face_split(f_m30, c, m30).expect("c-m30 split");

        geom.record_inner_vert(c);
        geom.faces.push(face);
        geom.faces.push(q_a);
        geom.faces.push(q_b);
        // The fourth quad is whichever half the last two splits left as `face`'s
        // sibling; it is already in the mesh and reachable, but to report the
        // full fragment set we collect every face now incident to c.
        for f in self.vert_face_keys(c) {
            if !geom.faces.contains(&f) {
                geom.faces.push(f);
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

        // §9.1.1 — endpoint vertex normals must be current before any off-chord
        // placement. Freshen once; `smooth == 0` keeps the chord-only path.
        if params.smooth != 0.0 {
            self.mesh_normals_update();
        }

        let midpoint_layer = self.push_vert_op_flag_layer();
        const MIDPOINT: u8 = 1;

        // Phase one: per-edge N-evenly-spaced split (§3.3). Each input edge is
        // split into N+1 segments by inserting N midpoints at t = i/(N+1). A
        // shared interior edge is cut once, so both incident faces see the same
        // midpoint run — crack-free (§9.5.4).
        let mut touched_faces: Vec<FaceKey> = Vec::new();
        for &edge in edges {
            let [near, far] = self.edge_vert_keys(edge);
            // §9.1 — capture the smooth endpoint context before the split. Each
            // of the N midpoints lands at the directional cut parameter
            // t = i/(N+1) (§3.3), measured from `near`.
            // §9.4.3 — the §9.1 offset is skipped under `use_sphere` (sphere wins).
            let smooth_ctx = (!params.use_sphere && params.smooth != 0.0)
                .then(|| self.smooth_endpoint_context(edge, near));
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
                self.vert_op_flag_set(midpoint_layer, m, MIDPOINT);
                geom.verts.push(m);
                geom.split_verts.push(m);
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
            // Any midpoint's incident faces are touched; pick the first to
            // discover them (all midpoints of an edge touch the same faces).
            if let Some(&m) = mids.first() {
                for f in self.vert_face_keys(m) {
                    if !touched_faces.contains(&f) {
                        touched_faces.push(f);
                    }
                }
            }
        }

        // Phase two: per-face §9.5 lattice fill.
        for face in touched_faces {
            self.resplit_face_multicut(face, midpoint_layer, MIDPOINT, cuts, &params, &mut geom);
        }

        self.pop_vert_op_flag_layer(midpoint_layer);
        SubdivideResult::Ran(geom)
    }

    /// Decompose a touched face's post-§3.3 loop cycle into its original
    /// corners and, for each side, the ordered midpoint run between consecutive
    /// corners (in face-traversal order). A side with an empty run was not cut.
    fn multicut_corners_and_sides(
        &self,
        face: FaceKey,
        layer: crate::op_flags::VertOpFlagLayer,
        midpoint_flag: u8,
    ) -> (Vec<VertexKey>, Vec<Vec<VertexKey>>) {
        let mut rim: Vec<(VertexKey, bool)> = Vec::new();
        let mut it = self.face_loops(face);
        while let Some(l) = it.next(self) {
            let v = self.loops.get(l).vertex;
            let is_mid = self.vert_op_flag_test(layer, v, midpoint_flag);
            rim.push((v, is_mid));
        }

        // Rotate so the rim starts at an original corner.
        let start = rim
            .iter()
            .position(|(_, m)| !*m)
            .expect("a touched face has at least one original corner");
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
        (corners, sides)
    }

    fn resplit_face_multicut(
        &mut self,
        face: FaceKey,
        layer: crate::op_flags::VertOpFlagLayer,
        midpoint_flag: u8,
        cuts: u32,
        params: &SubdivideParams,
        geom: &mut SubdivideGeom,
    ) {
        let (corners, sides) = self.multicut_corners_and_sides(face, layer, midpoint_flag);
        let corner_count = corners.len();
        let cut_sides = sides.iter().filter(|s| !s.is_empty()).count();
        if cut_sides == 0 {
            return;
        }

        // §9.3 — restrict re-split to quads when use_only_quads is on. A
        // non-quad keeps its §3.3 midpoints but is left flat.
        if params.use_only_quads && corner_count != 4 {
            return;
        }

        let n = cuts as usize;
        match (corner_count, cut_sides) {
            (4, 4) => {
                if params.use_grid_fill {
                    self.quad_grid_fill(face, &corners, &sides, n, params, geom);
                }
            }
            (3, 3) => {
                if params.use_grid_fill {
                    self.tri_lattice_fill(face, &corners, &sides, n, params, geom);
                }
            }
            (4, 1) => {
                if params.use_single_edge {
                    self.quad_one_edge_fan(face, &corners, &sides, n, geom);
                }
            }
            (3, 1) => {
                if params.use_single_edge {
                    self.tri_one_edge_fan(face, &corners, &sides, geom);
                }
            }
            // Other multi-cut configurations (partial quad two/three-edge,
            // triangle two-edge, n-gons) are §9.5.3 / §4.3 generic-fan cases
            // not yet pinned at multi-cut; leave the flat-cornered face.
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
                node[r][c] = vk;
                geom.record_inner_vert(vk);
            }
        }

        let src_face = face.as_usize();
        self.face_kill(face);
        for r in 0..=n {
            for c in 0..=n {
                let cell = [
                    node[r][c],
                    node[r][c + 1],
                    node[r + 1][c + 1],
                    node[r + 1][c],
                ];
                let f = self.create_lattice_cell(&cell, src_face, geom);
                geom.faces.push(f);
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
                    geom.record_inner_vert(vk);
                    vk
                };
                row.push(vk);
            }
            rows.push(row);
        }

        let src_face = face.as_usize();
        self.face_kill(face);
        for s in 0..=n {
            let upper = n - s; // top index of the down-triangle run
                               // Up triangles.
            for k in 0..=upper {
                let cell = [rows[s][k], rows[s][k + 1], rows[s + 1][k]];
                let f = self.create_lattice_cell(&cell, src_face, geom);
                geom.faces.push(f);
            }
            // Down triangles.
            for k in 0..upper {
                let cell = [rows[s][k + 1], rows[s + 1][k + 1], rows[s + 1][k]];
                let f = self.create_lattice_cell(&cell, src_face, geom);
                geom.faces.push(f);
            }
        }
    }

    /// §9.5.3 — one-edge quad fan. The single cut edge `corners[0]->corners[1]`
    /// carries `N` midpoints; each connects into the interior toward the two
    /// opposite corners (the lower half toward `v3`, the upper half toward
    /// `v2`), fanning the quad into `N + 1` fragments.
    fn quad_one_edge_fan(
        &mut self,
        face: FaceKey,
        corners: &[VertexKey],
        sides: &[Vec<VertexKey>],
        n: usize,
        geom: &mut SubdivideGeom,
    ) {
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

        geom.faces.push(face);
        // Lower half (parameter < 0.5) fans to v3; upper half to v2.
        let split = (n + 1) / 2;
        for i in 0..n {
            let m = run[i];
            let target = if i < split { v3 } else { v2 };
            let sub = self.face_with_both(m, target);
            if let Some((new_face, inner)) = self.face_split(sub, m, target) {
                geom.faces.push(new_face);
                geom.record_inner_edge(inner);
            }
        }
        // Collect any final residual fragment(s) reachable from the midpoints.
        for &m in run {
            for f in self.vert_face_keys(m) {
                if !geom.faces.contains(&f) {
                    geom.faces.push(f);
                }
            }
        }
    }

    /// §9.5.3 — one-edge triangle fan. The single cut edge `corners[0]->
    /// corners[1]` carries `N` midpoints; each connects to the opposite apex
    /// `corners[2]`, fanning the triangle into `N + 1` triangles.
    fn tri_one_edge_fan(
        &mut self,
        face: FaceKey,
        corners: &[VertexKey],
        sides: &[Vec<VertexKey>],
        geom: &mut SubdivideGeom,
    ) {
        let cut = sides
            .iter()
            .position(|s| !s.is_empty())
            .expect("one cut side");
        let apex = corners[(cut + 2) % 3];
        let run = sides[cut].clone();

        geom.faces.push(face);
        for m in run {
            let sub = self.face_with_both(m, apex);
            if let Some((new_face, inner)) = self.face_split(sub, m, apex) {
                geom.faces.push(new_face);
                geom.record_inner_edge(inner);
            }
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
        for f in &geom.faces {
            assert_eq!(mesh.face_len(*f), 4, "loop cut yields quads");
        }

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
        let mut lens: Vec<usize> = geom.faces.iter().map(|&f| mesh.face_len(f)).collect();
        lens.sort_unstable();
        assert_eq!(lens, vec![3, 5], "1 triangle + 1 pentagon");

        // The triangle is the shared-corner cut-off {m01, v1, m12}; v1 is on it.
        let tri = geom
            .faces
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
        let pent = geom
            .faces
            .iter()
            .copied()
            .find(|&f| mesh.face_len(f) == 5)
            .unwrap();
        assert!(
            !mesh.face_vert_keys(pent).contains(&v1),
            "pentagon excludes the shared corner v1"
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
        let mut lens: Vec<usize> = geom.faces.iter().map(|&f| mesh.face_len(f)).collect();
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
        let mut lens: Vec<usize> = geom.faces.iter().map(|&f| mesh.face_len(f)).collect();
        lens.sort_unstable();
        assert_eq!(lens, vec![3, 3, 4], "corner tri + central quad + far tri");

        // The corner triangle {m01, v1, m12} holds the shared corner v1.
        let corner_tri = geom
            .faces
            .iter()
            .copied()
            .find(|&fk| mesh.face_len(fk) == 3 && mesh.face_vert_keys(fk).contains(&v1))
            .expect("corner triangle holds v1");
        let cv = mesh.face_vert_keys(corner_tri);
        assert!(cv.contains(&m01) && cv.contains(&m12));

        // The far-corner triangle {v0, v2, v3} holds v3 and both uncut corners.
        let far_tri = geom
            .faces
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
        let mut lens: Vec<usize> = geom.faces.iter().map(|&f| mesh.face_len(f)).collect();
        lens.sort_unstable();
        assert_eq!(lens, vec![3, 3, 4], "tri + central quad + tri");

        // The central quad {v3, m01, v1, m12} is the only fragment holding the
        // shared corner v1.
        let central = geom
            .faces
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
        let holds_v1 = geom
            .faces
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
        let mut lens: Vec<usize> = geom.faces.iter().map(|&f| mesh.face_len(f)).collect();
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
        let mut lens: Vec<usize> = geom.faces.iter().map(|&f| mesh.face_len(f)).collect();
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
        let quad = geom
            .faces
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
        let mut lens: Vec<usize> = geom.faces.iter().map(|&f| mesh.face_len(f)).collect();
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
        let tri = geom
            .faces
            .iter()
            .copied()
            .find(|&f| mesh.face_len(f) == 3)
            .unwrap();
        let tv = mesh.face_vert_keys(tri);
        assert!(tv.contains(&v1), "triangle holds the shared corner v1");
        assert!(tv.contains(&m01) && tv.contains(&m12));

        // The quad {v0, m01, m12, v2} excludes the shared corner v1.
        let quad = geom
            .faces
            .iter()
            .copied()
            .find(|&f| mesh.face_len(f) == 4)
            .unwrap();
        let qv = mesh.face_vert_keys(quad);
        assert!(!qv.contains(&v1), "quad excludes the shared corner v1");
        assert!(qv.contains(&v0) && qv.contains(&v2));
    }

    #[test]
    fn quad_two_cut_opposite_discriminator() {
        // Direct check of the opposite-vs-adjacent discriminator on the
        // post-insertion six-corner ring. Opposite midpoints sit 3 apart;
        // adjacent midpoints sit 2 apart.
        let dummy = VertexKey::from_usize(0);
        let opposite = [
            (dummy, true),
            (dummy, false),
            (dummy, false),
            (dummy, true),
            (dummy, false),
            (dummy, false),
        ];
        let adjacent = [
            (dummy, true),
            (dummy, false),
            (dummy, true),
            (dummy, false),
            (dummy, false),
            (dummy, false),
        ];
        assert!(LoopMesh::quad_two_cut_is_opposite(&opposite));
        assert!(!LoopMesh::quad_two_cut_is_opposite(&adjacent));
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

        let res = mesh.subdivide_edges(&[shared], SubdivideParams::default().use_only_quads(true));
        assert!(matches!(res, SubdivideResult::Ran(_)));

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
        // §5.1 / §1 — one cut edge of a quad. geom_split: 1 midpoint + the 2
        // edge-halves of the split input edge. geom_inner: the 2 interior
        // connecting edges (m→v3, m→v2), no inner vert. geom: the union.
        let mut mesh = LoopMesh::new();
        let face = unit_quad(&mut mesh);
        let fv = mesh.face_vert_keys(face);
        let e01 = mesh.query_edge_exists(fv[0], fv[1]).unwrap();

        let geom = match mesh.subdivide_edges(&[e01], SubdivideParams::default()) {
            SubdivideResult::Ran(g) => g,
            _ => panic!("expected Ran"),
        };

        // geom_split.
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

        // geom_inner.
        let inner = geom.geom_inner();
        assert!(inner.verts.is_empty(), "no inner vert for the one-cut quad");
        assert_eq!(inner.edges.len(), 2, "two interior connecting edges");
        no_dup(inner.edges, "geom_inner.edges");

        // geom (union). 1 vert created, 3 edges created (2 halves + ... the two
        // inner connects are extra, the surviving half overlaps); E went 4→7 so
        // 3 new + 1 surviving half = the 4 created-or-replaced edges that the
        // operator touched (2 split + 2 inner).
        let g = geom.geom();
        assert_eq!(g.verts.len(), 1, "geom union: one created vert");
        assert_eq!(g.faces.len(), 3, "geom union: 3 fragment faces");
        // Union edges = the 2 split-halves + 2 inner edges (disjoint sets here).
        assert_eq!(g.edges.len(), 4, "geom union: 2 split + 2 inner edges");
        no_dup(g.edges, "geom.edges");
        // geom_split and geom_inner partition geom's edge set for this pattern.
        for &e in split.edges {
            assert!(g.edges.contains(&e), "split edge is in the union");
        }
        for &e in inner.edges {
            assert!(g.edges.contains(&e), "inner edge is in the union");
        }
    }

    #[test]
    fn slots_quad_four_edges_plus_split() {
        // §5.4 / §1 — the headline four-cut quad. geom_split: 4 midpoints + 8
        // edge-halves (each of the 4 input edges → 2). geom_inner: the centre
        // inner vert + the 4 "plus" spokes. geom: the union (V 5 / E 12 / F 4).
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
        assert_eq!(inner.verts.len(), 1, "the centre inner vert");
        assert_eq!(inner.edges.len(), 4, "the 4 plus-split spokes");
        no_dup(inner.edges, "geom_inner.edges");
        // The inner vert is the centroid (1, 1, 0); each inner edge meets it.
        let c = inner.verts[0];
        assert!(
            mesh.vertex_position(c).distance(Vec3::new(1.0, 1.0, 0.0)) < 1e-5,
            "inner vert at the centroid"
        );
        for &e in inner.edges {
            let [a, b] = mesh.edge_vert_keys(e);
            assert!(a == c || b == c, "each spoke meets the inner vert");
        }

        let g = geom.geom();
        assert_eq!(g.verts.len(), 5, "geom union: 4 mid + 1 inner");
        assert_eq!(g.edges.len(), 12, "geom union: 8 split + 4 inner");
        assert_eq!(g.faces.len(), 4, "geom union: the 2x2 grid faces");
        no_dup(g.edges, "geom.edges");
        no_dup(g.verts, "geom.verts");
        // The union vert/edge sets equal the final mesh totals minus the
        // pre-existing 4 corners / 0 reused-whole edges.
        assert_eq!(counts(&mesh), (9, 12, 4), "four-cut quad deltas");
        // geom_split ∪ geom_inner = geom for verts and edges (disjoint here).
        assert_eq!(
            split.verts.len() + inner.verts.len(),
            g.verts.len(),
            "split+inner verts partition the union"
        );
        assert_eq!(
            split.edges.len() + inner.edges.len(),
            g.edges.len(),
            "split+inner edges partition the union"
        );
    }

    #[test]
    fn slots_multicut_quad_grid_cuts2() {
        // §9.5.1 / §1 — fully-cut quad at cuts = 2 (3x3 grid). geom_split: 8
        // rim midpoints + 16 edge-halves (each of the 4 input edges → 3
        // segments). geom_inner: the 4 interior lattice verts + the inner
        // lattice edges; geom: the union (V 16 / E 24 / F 9).
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
        assert_eq!(inner.verts.len(), 4, "the 4 interior lattice nodes");
        no_dup(inner.verts, "geom_inner.verts");
        no_dup(inner.edges, "geom_inner.edges");
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
        // Union edges = the 12 rim halves + every interior lattice edge; the
        // total matches the final mesh edge count (no edge survives uncut).
        assert_eq!(g.edges.len(), 24, "geom union covers all 24 mesh edges");
        assert_eq!(split.edges.len() + inner.edges.len(), g.edges.len());
        assert_eq!(counts(&mesh), (16, 24, 9), "cuts=2 grid deltas");
    }
}
