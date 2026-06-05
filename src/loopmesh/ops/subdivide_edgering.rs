use std::collections::HashMap;
use std::collections::HashSet;

use glam::{Mat3, Quat, Vec3};

use crate::loopmesh::*;

/// Placement of the new ring verts across the strip
/// ([`SUBDIVIDE_EDGERING.md`](../../../../radial-mesh-guides/operators/SUBDIVIDE_EDGERING.md)
/// §5 / §6.2).
///
/// All three modes share the §3–§4 / §6.1 topology and differ **only** in where
/// each inserted vert lands (§6.2 is positions-only):
///
/// - [`RingInterp::Linear`] (§5 / §6.1.4) — the straight chord point of each
///   rung at `t_j`. The flat baseline.
/// - [`RingInterp::Path`] (§6.2.2) — carries each inserted ring's cross-section
///   along the strip's centre spline, rotating it by a minimum-twist frame
///   sweep. Diverges from `Linear` as soon as the strip bends along its length,
///   **including at `smooth = 0`** (the divergence is the rim-frame rotation,
///   not a bulge).
/// - [`RingInterp::Surface`] (§6.2.3) — pulls each inserted ring onto its own
///   cubic meeting the two rims tangentially. At `smooth = 0` it stays on the
///   chord *line* but re-spaces interior verts by smoothstep `B(t)=t²(3−2t)`
///   (exact `Linear` only for `cuts = 1`, where `B(1/2)=1/2`); the off-chord bow
///   scales with `smooth`.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum RingInterp {
    Linear = 0,
    Path = 1,
    Surface = 2,
}

impl Default for RingInterp {
    fn default() -> Self {
        RingInterp::Linear
    }
}

/// Parameters for [`LoopMesh::subdivide_edgering`].
///
/// Mirrors the BMOP parameter surface
/// ([`SUBDIVIDE_EDGERING.md`](../../../../radial-mesh-guides/operators/SUBDIVIDE_EDGERING.md)
/// §1). The default is the foundational core: a single mid-ring cut
/// (`cuts = 1`), `LINEAR` placement, no smoothing, and the flat profile — every
/// new ring vert lands on the straight chord midpoint of its rung (§5).
///
/// `cuts >= 2` (§6.1) and `interp_mode` PATH/SURFACE (§6.2) are honoured. The
/// `smooth` bulge and the profile parameters (§6.3) are surveyed there and
/// tracked as a separate parity row; `smooth` participates in the §6.2 spline
/// handles (so non-zero values bend the PATH/SURFACE splines), but the §6.3
/// profile-curve amplitude is not yet applied.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SubdivideEdgeRingParams {
    /// Number of new parallel rings inserted between the strip's rim rings
    /// (§1 / §6.1). The core supports `1` (a single midpoint ring).
    pub cuts: u32,
    /// Placement of the new ring verts (§5 / §6.2). LINEAR, PATH and SURFACE
    /// are all honoured.
    pub interp_mode: RingInterp,
    /// Spline-handle / bulge factor (§6.2 / §6.3). At `0.0` PATH still rotates
    /// the cross-section along the centre line and SURFACE still smoothstep-spaces
    /// interior verts (both stay on the chord *line*); non-zero values bend the
    /// PATH/SURFACE splines off the chord. No effect under LINEAR.
    pub smooth: f32,
    /// Profile amplitude (§6.3). No-op in the core (`0.0`).
    pub profile_shape_factor: f32,
}

impl Default for SubdivideEdgeRingParams {
    fn default() -> Self {
        SubdivideEdgeRingParams {
            cuts: 1,
            interp_mode: RingInterp::Linear,
            smooth: 0.0,
            profile_shape_factor: 0.0,
        }
    }
}

/// The geometry [`LoopMesh::subdivide_edgering`] created, gathered as a set
/// ([`SUBDIVIDE_EDGERING.md`](../../../../radial-mesh-guides/operators/SUBDIVIDE_EDGERING.md)
/// §1 / §4). Enumeration order is not observable for parity; treat as a set.
#[derive(Default, Clone, Debug, PartialEq, Eq)]
pub struct SubdivideEdgeRingResult {
    /// The new ring verts — one chord midpoint per cut rung (§3 / §5).
    pub verts: Vec<VertexKey>,
    /// The new fill faces (`faces.out`, §4): each strip quad split into an
    /// A-side and a B-side quad.
    pub faces: Vec<FaceKey>,
}

/// A strip quad of the resolved ring: a quad face flanked by two opposite rungs
/// (§2 / §4). Captured before phase one so the face identity survives the rung
/// splits (the split-edge-make-vert Euler op only adds rim corners — it never
/// retires the face).
///
/// The four corner verts are recorded in the quad's loop-cycle order with the
/// rung pair at `(rung_a_near → rung_a_far)` and `(rung_b_near → rung_b_far)`,
/// where `rung_a_near` and `rung_b_near` are the two endpoints joined by one
/// rail (the "A-rail" side). Phase two aligns each rung's midpoint run to this
/// near→far direction so level `j` on `rung_a` pairs one-for-one with level `j`
/// on `rung_b` (§6.1.1 "the directional parameter is the load-bearing detail").
struct StripQuad {
    face: FaceKey,
    rung_a: EdgeKey,
    rung_b: EdgeKey,
    /// Endpoint of `rung_a` on the shared rail with `rung_b_near`.
    rung_a_near: VertexKey,
    /// Endpoint of `rung_b` rail-joined to `rung_a_near` — the level-0 mate.
    rung_b_near: VertexKey,
    /// Far endpoint of `rung_a` (the B-rail corner). Phase two uses it to tell
    /// the deeper (B-rail-side) fragment from the freshly peeled row.
    rung_a_far: VertexKey,
}

/// Phase-one record for one rung: the new midpoints in `near → far` order
/// (§3.3 directional), together with the `near` endpoint they were measured
/// from. Phase two reverses this run when a quad reads the rung from its far
/// endpoint so the level index stays a consistent cross-strip position.
struct RungCuts {
    near: VertexKey,
    /// `midpoints[j]` is the level-`(j+1)` cut vert (`t = (j+1)/(N+1)` from
    /// `near`); `len() == N`.
    midpoints: Vec<VertexKey>,
}

impl RungCuts {
    /// The midpoint run as seen from `from_near`, i.e. level 0..N in cross-strip
    /// order when read starting at `from_near`. Reverses the stored run when the
    /// quad reads this rung from the far endpoint.
    fn aligned(&self, from_near: VertexKey) -> Vec<VertexKey> {
        if from_near == self.near {
            self.midpoints.clone()
        } else {
            self.midpoints.iter().rev().copied().collect()
        }
    }
}

/// Per-strip data the §6.2 non-linear placement needs, resolved from the
/// pre-split topology. `rungs` maps each rung to its `(rail-A endpoint, rail-B
/// endpoint)`; the inserted verts are repositioned between them per `mode`.
struct RingPlacement {
    rungs: HashMap<EdgeKey, (VertexKey, VertexKey)>,
    mode: RingPlacementMode,
}

enum RingPlacementMode {
    /// §6.2.2 PATH: the per-strip centre-spline setup.
    Path(PathSetup),
    /// §6.2.3 SURFACE: a snapshot of each rim vert's surrounding-surface
    /// tangent, taken before any split (the §6.2.3 feedback hazard).
    Surface(HashMap<VertexKey, Vec3>),
}

/// §6.2.1 shared setup for PATH: the two rim-ring centres (the centre-path
/// endpoints) and their reconciled plane normals (the spline end tangents).
struct PathSetup {
    center_a: Vec3,
    center_b: Vec3,
    normal_a: Vec3,
    normal_b: Vec3,
}

/// Two-colour union-find over rim membership: rungs' endpoints split across two
/// rails (§6.2.1). `same` merges two verts onto one rail; `opposite` records
/// that two verts sit on different rails. After all strip quads are folded in,
/// `on_anchor_side` reports whether a vert shares the anchor's rail.
struct RailColoring {
    parent: HashMap<VertexKey, VertexKey>,
    /// Parity of a vert relative to its component root: `false` = same rail as
    /// root, `true` = opposite rail.
    parity: HashMap<VertexKey, bool>,
}

impl RailColoring {
    fn new() -> Self {
        RailColoring {
            parent: HashMap::new(),
            parity: HashMap::new(),
        }
    }

    fn ensure(&mut self, v: VertexKey) {
        self.parent.entry(v).or_insert(v);
        self.parity.entry(v).or_insert(false);
    }

    /// Returns `(root, parity-to-root)` with path compression.
    fn find(&mut self, v: VertexKey) -> (VertexKey, bool) {
        self.ensure(v);
        let p = self.parent[&v];
        if p == v {
            return (v, false);
        }
        let (root, par_p) = self.find(p);
        let par_v = par_p ^ self.parity[&v];
        self.parent.insert(v, root);
        self.parity.insert(v, par_v);
        (root, par_v)
    }

    fn union(&mut self, a: VertexKey, b: VertexKey, want_opposite: bool) {
        let (ra, pa) = self.find(a);
        let (rb, pb) = self.find(b);
        if ra == rb {
            return;
        }
        // Re-root rb under ra. Its parity relative to ra must make
        // parity(a) ^ parity(b) == want_opposite hold.
        let new_parity = pa ^ pb ^ want_opposite;
        self.parent.insert(rb, ra);
        self.parity.insert(rb, new_parity);
    }

    fn same(&mut self, a: VertexKey, b: VertexKey) {
        self.union(a, b, false);
    }

    fn opposite(&mut self, a: VertexKey, b: VertexKey) {
        self.union(a, b, true);
    }

    /// Whether `v` is on the same rail as `anchor` (both must be in the same
    /// connected component — true for any strip resolved from a contiguous quad
    /// set).
    fn on_anchor_side(&mut self, v: VertexKey, anchor: VertexKey) -> bool {
        let (rv, pv) = self.find(v);
        let (ra, pa) = self.find(anchor);
        debug_assert_eq!(rv, ra, "rim verts of one strip share a component");
        pv == pa
    }
}

/// The shared Bézier handle length for the §6.2 splines (§6.2.1.1). Scales the
/// off-chord bow, not its direction (multiplied by `smooth` at the call site, so
/// it never affects the `smooth = 0` geometry). Generic in its two anchor points
/// `a`, `b` and two **unit** end directions `dir_a`, `dir_b` (each already
/// leaning the way its handle will): PATH feeds it the rim centres + reconciled
/// normals, SURFACE the cut ring's rim verts + surface tangents.
///
/// `H = (L_proj / 2) · fac`:
///
/// - `fac = 4/3` (the quarter-circle handle length) when the two directions are
///   not opposed (`dot ≥ 0`), eased linearly down toward `3/4` as they oppose
///   (`dot → −1`) via `t = 1 + dot`, so a tight fold does not overshoot.
/// - `L_proj` is the chord `a → b` measured **only in the plane the two
///   directions span** — the component along their cross axis (the fold's hinge)
///   is discarded, so `H` tracks the bow rather than the span. When the
///   directions are parallel there is no bend plane; fall back to the full chord
///   length.
fn handle_length(a: Vec3, b: Vec3, dir_a: Vec3, dir_b: Vec3) -> f32 {
    let dot = dir_a.dot(dir_b);
    let mut fac = 4.0 / 3.0;
    if dot < 0.0 {
        let t = 1.0 + dot;
        fac = fac * t + 0.75 * (1.0 - t);
    }

    let chord = b - a;
    let axis = dir_a.cross(dir_b);
    let len = if axis.length_squared() > 1e-12 {
        // Perpendicular distance from `b` to the line through `a` along `axis`:
        // the chord with its hinge-axis component removed.
        let axis = axis.normalize();
        (chord - chord.dot(axis) * axis).length()
    } else {
        chord.length()
    };

    (len * 0.5) * fac
}

/// Edge-length-weighted centre of a rim ring (§6.2.1). `ring` holds the ring's
/// verts in loop order (treated as closed). Each vert is weighted by the sum of
/// the lengths of its two incident ring edges, so longer spans pull the centre
/// toward them and an unevenly-spaced ring is not biased toward its densely
/// sampled side. Reduces to the plain mean only for an equal-edge ring.
fn rim_center(ring: &[Vec3]) -> Vec3 {
    if ring.is_empty() {
        return Vec3::ZERO;
    }
    let mut weighted_sum = Vec3::ZERO;
    let mut total_weight = 0.0f32;
    let n = ring.len();
    for i in 0..n {
        let prev = ring[(i + n - 1) % n];
        let curr = ring[i];
        let next = ring[(i + 1) % n];
        let weight = (curr - prev).length() + (next - curr).length();
        weighted_sum += weight * curr;
        total_weight += weight;
    }
    if total_weight > 0.0 {
        weighted_sum / total_weight
    } else {
        ring.iter().fold(Vec3::ZERO, |a, &v| a + v) / n as f32
    }
}

/// Area-weighted (Newell) ring normal of a rim ring (§6.2.1). `ring` holds the
/// ring's verts in loop order; the ring is treated as a *closed* polygon (the
/// last vert wraps back to the first). Each consecutive pair contributes the
/// Newell edge term — the 3D shoelace / signed-projected-area contribution — so
/// each edge weighs in proportion to the area it sweeps and the result stays
/// robust on a non-planar (twisted) ring rather than snapping to a least-variance
/// axis. The accumulated vector's sign follows the ring's winding; the caller's
/// reconciliation fixes the two rims' signs relative to each other. Returns
/// `None` when the ring is degenerate/collinear (near-zero swept area), letting
/// the caller substitute the centre-aligned fallback.
fn newell_ring_normal(ring: &[Vec3]) -> Option<Vec3> {
    if ring.len() < 3 {
        return None;
    }
    let mut no = Vec3::ZERO;
    for i in 0..ring.len() {
        let v_prev = ring[i];
        let v_curr = ring[(i + 1) % ring.len()];
        no.x += (v_prev.y - v_curr.y) * (v_prev.z + v_curr.z);
        no.y += (v_prev.z - v_curr.z) * (v_prev.x + v_curr.x);
        no.z += (v_prev.x - v_curr.x) * (v_prev.y + v_curr.y);
    }
    let n = no.normalize_or_zero();
    if n.length_squared() < 0.5 {
        None
    } else {
        Some(n)
    }
}

/// Fallback rim normal for a degenerate/collinear ring (§6.2.1): the ring sweeps
/// no area, so the Newell construction is undefined. Build an axis in the same
/// edge-summed spirit — for each ring edge, accumulate the component of that
/// edge's perpendicular lying in the plane normal to the centre-to-centre
/// `axis` (`(edge × axis) × edge`, summed over the closed ring), then normalise.
/// Falls back to the bare `axis` direction if even that accumulation vanishes
/// (a ring collinear *with* the axis).
fn centre_aligned_fallback_normal(ring: &[Vec3], axis: Vec3) -> Vec3 {
    let mut accum = Vec3::ZERO;
    if ring.len() >= 2 {
        for i in 0..ring.len() {
            let edge = ring[(i + 1) % ring.len()] - ring[i];
            accum += edge.cross(axis).cross(edge);
        }
    }
    let n = accum.normalize_or_zero();
    if n.length_squared() < 0.5 {
        axis.normalize_or_zero()
    } else {
        n
    }
}

/// Cubic Bézier point at `t` from four control points.
fn bezier(p0: Vec3, p1: Vec3, p2: Vec3, p3: Vec3, t: f32) -> Vec3 {
    let u = 1.0 - t;
    p0 * (u * u * u)
        + p1 * (3.0 * u * u * t)
        + p2 * (3.0 * u * t * t)
        + p3 * (t * t * t)
}

/// The local path tangent at an interior sample, the angle bisector of the two
/// incoming/outgoing segment directions (§6.2.2 step 2).
fn path_bisector(prev: Vec3, here: Vec3, next: Vec3) -> Vec3 {
    let incoming = (here - prev).normalize_or_zero();
    let outgoing = (next - here).normalize_or_zero();
    let bisector = incoming + outgoing;
    if bisector.length_squared() > 1e-12 {
        bisector.normalize()
    } else {
        outgoing
    }
}

/// Minimum-twist orientation sweep along the per-level directions (§6.2.2 step
/// 2). Frame 0 is the curve-framing seed (`seed_frame`: local +Z aligned to
/// `dir[0]`, local +Y the up reference) — **not** a free shortest-arc rotation,
/// because the up choice pins the roll about `dir[0]` and that roll is carried
/// through every level. Each subsequent frame is the previous one pre-rotated by
/// the smallest rotation carrying `dir[j-1]` onto `dir[j]` — a
/// rotation-minimizing frame, no spurious roll beyond the seed's.
fn minimum_twist_frames(dir: &[Vec3]) -> Vec<Quat> {
    let mut frames = Vec::with_capacity(dir.len());
    frames.push(seed_frame(dir[0]));
    for j in 1..dir.len() {
        let delta = rotation_between(dir[j - 1], dir[j]);
        let frame = delta * frames[j - 1];
        frames.push(frame);
    }
    frames
}

/// The curve-framing seed orientation (§6.2.2 step 2 "Seed"): the rotation that
/// aligns the frame's local **+Z axis to `dir`** while keeping its local **+Y
/// axis** as close as possible to the **up reference** (world +Y). This is the
/// align-one-axis-to-the-tangent / pick-a-second-axis-as-up convention; the up
/// choice fixes the roll about `dir`, so it must not be a free
/// shortest-arc rotation onto `dir`.
///
/// Construction: with `z = dir`, `x = normalize(up × z)`, `y = z × x`, the
/// frame is the rotation whose columns are `(x, y, z)`. When `dir` is parallel
/// to the up reference the cross product degenerates; a secondary up reference
/// (world +X) is substituted so the roll stays deterministic.
fn seed_frame(dir: Vec3) -> Quat {
    let z = dir.normalize_or_zero();
    if z.length_squared() < 0.5 {
        return Quat::IDENTITY;
    }
    let mut up = Vec3::Y;
    if z.cross(up).length_squared() < 1e-6 {
        up = Vec3::X;
    }
    let x = up.cross(z).normalize();
    let y = z.cross(x);
    Quat::from_mat3(&Mat3::from_cols(x, y, z))
}

/// The smallest rotation carrying unit `from` onto unit `to` (about their cross
/// product through the angle between them). Stable for the near-parallel and
/// near-antiparallel limits.
fn rotation_between(from: Vec3, to: Vec3) -> Quat {
    let f = from.normalize_or_zero();
    let t = to.normalize_or_zero();
    if f.length_squared() < 0.5 || t.length_squared() < 0.5 {
        return Quat::IDENTITY;
    }
    let d = f.dot(t).clamp(-1.0, 1.0);
    if d > 1.0 - 1e-6 {
        return Quat::IDENTITY;
    }
    if d < -1.0 + 1e-6 {
        // Antiparallel: rotate π about any axis orthogonal to `f`.
        let axis = f.cross(Vec3::X);
        let axis = if axis.length_squared() < 1e-6 {
            f.cross(Vec3::Y)
        } else {
            axis
        };
        return Quat::from_axis_angle(axis.normalize(), std::f32::consts::PI);
    }
    let axis = f.cross(t).normalize();
    let angle = d.acos();
    Quat::from_axis_angle(axis, angle)
}

/// A reference triangle for one level (§6.2.2 step 3): the origin and the tips
/// of two unit legs along the frame's local +X then +Y axes, rotated by `frame`
/// and anchored at `path`. The vertex order pins the handedness and axis pairing
/// of the triangle-to-triangle transport (`p0` = origin, `p1` = local-X leg,
/// `p2` = local-Y leg), so it is **not** an ad-hoc rigid reframe.
///
/// `size` is the per-level shape size `s` (§6.2.2 step 3 / §6.3): `1.0` at the
/// default, scaled by the (out-of-scope) profile falloff when set. At `s = 1`
/// every level's triangle is the same unit right-isosceles triangle merely
/// rotated and translated, so the transport reduces to a rigid motion.
struct RefTriangle {
    p0: Vec3,
    p1: Vec3,
    p2: Vec3,
    /// Unit plane normal (= the frame's local +Z up to construction sign).
    normal: Vec3,
    /// Triangle area (for the `sqrt(area)` out-of-plane scaling).
    area: f32,
}

fn reference_triangle(frame: Quat, path: Vec3, size: f32) -> RefTriangle {
    let p0 = path;
    let p1 = frame * Vec3::new(size, 0.0, 0.0) + path;
    let p2 = frame * Vec3::new(0.0, size, 0.0) + path;
    let cross = (p1 - p0).cross(p2 - p0);
    let area = 0.5 * cross.length();
    let normal = cross.normalize_or_zero();
    RefTriangle {
        p0,
        p1,
        p2,
        normal,
        area,
    }
}

/// The triangle-to-triangle affine transport `tri_transform(p, src, dst)`
/// (§6.2.2 step 3). Maps `p` from the plane/orientation of `src` into that of
/// `dst`. **Affine, not rigid**: it is the in-plane barycentric image plus the
/// out-of-plane offset scaled by `sqrt(area_dst / area_src)`.
///
/// 1. Project `p` into `src`'s plane and express the projection in barycentric
///    weights `(w0, w1, w2)` of `src`'s three vertices.
/// 2. Apply the same weights to `dst`'s vertices for the in-plane image.
/// 3. Offset along `dst`'s normal by `z · sqrt(area_dst / area_src)`, where `z`
///    is `p`'s signed height above `src`'s plane.
///
/// When `src` and `dst` are congruent (`s = 1` for all levels — the `smooth = 0`
/// default regime) the area ratio is 1 and the in-plane part is an isometry, so
/// the transport reduces to the rigid motion carrying `src`'s frame onto `dst`'s.
fn tri_transform(p: Vec3, src: &RefTriangle, dst: &RefTriangle) -> Vec3 {
    let rel = p - src.p0;
    let height = rel.dot(src.normal);
    let in_plane = rel - height * src.normal;

    // Barycentric of the in-plane point against src's two legs (e1, e2 from p0).
    let e1 = src.p1 - src.p0;
    let e2 = src.p2 - src.p0;
    let d11 = e1.dot(e1);
    let d12 = e1.dot(e2);
    let d22 = e2.dot(e2);
    let denom = d11 * d22 - d12 * d12;
    let (u, v) = if denom.abs() > 1e-12 {
        let p1c = in_plane.dot(e1);
        let p2c = in_plane.dot(e2);
        (
            (p1c * d22 - p2c * d12) / denom,
            (p2c * d11 - p1c * d12) / denom,
        )
    } else {
        (0.0, 0.0)
    };

    let image = dst.p0 + u * (dst.p1 - dst.p0) + v * (dst.p2 - dst.p0);

    let scale = if src.area > 1e-12 {
        (dst.area / src.area).sqrt()
    } else {
        1.0
    };
    image + dst.normal * (height * scale)
}

impl LoopMesh {
    /// Subdivide an edge ring
    /// ([`SUBDIVIDE_EDGERING.md`](../../../../radial-mesh-guides/operators/SUBDIVIDE_EDGERING.md)).
    ///
    /// Takes the **resolved rung set** of the ring(s) to cut — the full set of
    /// parallel edges crossing the quad strip, not a single seed edge (§1 "BMOP
    /// input"). The fill in §4 only fires on the strip quads that lie *between
    /// two named rungs*, so a higher-level "subdivide the ring through this one
    /// edge" entry point would trace the ring (§2.1) to populate this set first.
    ///
    /// For the foundational core (`cuts = 1`, `interp_mode = LINEAR`,
    /// `smooth = 0`, flat profile) the operator runs two phases:
    ///
    /// - **Phase one (§3):** split every rung once at its chord midpoint
    ///   (`t = 0.5`) via the split-edge-make-vert Euler op. The midpoints line
    ///   up into the new cut ring, born crack-free because each rung is one
    ///   shared edge seen by both its quads.
    /// - **Phase two (§4):** connect the two midpoints across each strip quad
    ///   with a single face-split (split-face-make-edge), carving each strip
    ///   quad into an A-side and a B-side quad and stitching the midpoints into
    ///   a continuous ring.
    ///
    /// Element-count deltas. For `cuts = 1` (§4.1): an open strip of `k` quads
    /// gains `k + 1` verts, `2k + 1` edges and `k` faces; a closed band gains
    /// `k` verts, `2k` edges and `k` faces. For `cuts = N` (§6.1.3) the deltas
    /// are `N` times those: open `+N·(k+1)` V / `+N·(2k+1)` E / `+N·k` F;
    /// closed `+N·k` V / `+2N·k` E / `+N·k` F. A strip of `k` quads becomes a
    /// band of `(N+1)·k` quads (`N+1` quad rows separated by the `N` new rings).
    ///
    /// Placement (§5 / §6.2) is honoured for all three `interp_mode` values:
    /// LINEAR falls out of the per-segment chord lerp; PATH and SURFACE
    /// reposition the inserted verts in a positions-only post-pass (the topology
    /// is identical across the three modes). The §6.3 profile-curve amplitude
    /// (`profile_shape_factor`) is not yet applied. `cuts` is honoured for all
    /// `N >= 1`.
    pub fn subdivide_edgering(
        &mut self,
        rungs: &[EdgeKey],
        params: SubdivideEdgeRingParams,
    ) -> SubdivideEdgeRingResult {
        let mut result = SubdivideEdgeRingResult::default();
        let cuts = params.cuts;
        if rungs.is_empty() || cuts == 0 {
            return result;
        }

        let rung_set: HashSet<EdgeKey> = rungs.iter().copied().collect();

        // Identify the strip quads before phase one. A strip quad is a quad
        // face flanked by two *opposite* rungs of the set (§2 / §4); the fill
        // fires only on those. Capturing before the splits keeps the face keys
        // and corner verts valid: split-edge-make-vert only inserts rim corners
        // (§3 / §6.1.1), it never retires the face or its original corners.
        let strip_quads = self.collect_strip_quads(&rung_set);

        // Non-linear placement (§6.2) is positions-only but reads pre-split
        // state: PATH/SURFACE need the two rim rings and (for SURFACE) the
        // surrounding-surface tangents snapshotted *before* phase one modifies
        // the band (§6.2.3 feedback hazard). Resolve the rims up front, while
        // the topology is still the original strip.
        let placement = if params.interp_mode == RingInterp::Linear {
            None
        } else {
            self.resolve_ring_placement(&strip_quads, &params)
        };

        // Phase one (§3 / §6.1.1): multi-cut each rung into `N + 1` equal
        // segments, recording the `N` midpoints in `near → far` order. The
        // directional run is the load-bearing detail (§6.1.1) — phase two
        // aligns level `j` across adjacent rungs by reading each rung from the
        // endpoint on the shared rail. `edge_split_n` runs the §3.3 directional
        // multi-cut, applying the two-source-linear vert/loop/edge blend at
        // each `t_j = j/(N+1)`. The chord positions it writes are the LINEAR
        // placement; PATH/SURFACE overwrite them below.
        let mut rung_cuts: HashMap<EdgeKey, RungCuts> = HashMap::with_capacity(rung_set.len());
        for &rung in &rung_set {
            let [near, _far] = self.edge_vert_keys(rung);
            let midpoints = self.edge_split_n(rung, near, cuts);
            result.verts.extend_from_slice(&midpoints);
            rung_cuts.insert(rung, RungCuts { near, midpoints });
        }

        // Phase two (§4 / §6.1.2): the level-aligned ladder fill. For each strip
        // quad, connect level `j` on `rung_a` to level `j` on `rung_b` with one
        // straight cross-cut (split-face-make-edge), carving the `(2N+2)`-gon
        // into `N + 1` quad rows from the A-rail toward the B-rail.
        for strip in &strip_quads {
            let levels_a = rung_cuts[&strip.rung_a].aligned(strip.rung_a_near);
            let levels_b = rung_cuts[&strip.rung_b].aligned(strip.rung_b_near);
            debug_assert_eq!(levels_a.len(), levels_b.len(), "rungs cut to same N");

            self.fill_strip_quad(
                strip.face,
                &levels_a,
                &levels_b,
                strip.rung_a_far,
                &mut result.faces,
            );
        }

        // §6.2 positions-only post-pass: overwrite the chord positions the
        // inserted verts were born with by the PATH / SURFACE placement.
        if let Some(placement) = placement {
            self.apply_ring_placement(&placement, &rung_cuts, cuts, params.smooth);
        }

        result
    }

    /// Resolve the per-strip data the §6.2 non-linear placement needs, read
    /// from the original (pre-split) topology. Returns `None` when the strip has
    /// no fillable quad (nothing to place).
    ///
    /// Two pieces: (1) the rail assignment — each rung's two endpoints split
    /// across rail A and rail B (§6.2.1) — recovered by walking the strip quads'
    /// `near~near` / `far~far` rail equivalences; and (2) the per-mode setup
    /// (PATH: rim centres + reconciled normals; SURFACE: a snapshot of every rim
    /// vert's surrounding-surface tangent, taken now to dodge the half-built-mesh
    /// feedback hazard of §6.2.3).
    fn resolve_ring_placement(
        &self,
        strip_quads: &[StripQuad],
        params: &SubdivideEdgeRingParams,
    ) -> Option<RingPlacement> {
        if strip_quads.is_empty() {
            return None;
        }

        // Rail equivalence walk. Within each strip quad, the two `near`
        // endpoints share one rail and the two `far` endpoints share the other
        // (§6.1.1 rail pairing). Union those equivalences across all quads, then
        // two-colour the components: every rung's endpoints fall in opposite
        // components, defining rail A vs rail B.
        let mut rail = RailColoring::new();
        for strip in strip_quads {
            let rung_b_far = self.edge_other_vert(strip.rung_b, strip.rung_b_near);
            rail.same(strip.rung_a_near, strip.rung_b_near);
            rail.same(strip.rung_a_far, rung_b_far);
            // The two endpoints of each rung are on opposite rails.
            rail.opposite(strip.rung_a_near, strip.rung_a_far);
            rail.opposite(strip.rung_b_near, rung_b_far);
        }

        // Per-rung (v_a, v_b): pick rail A as the colour of the first quad's
        // `rung_a_near`. Every rung contributes both its endpoints once.
        let rail_a_anchor = strip_quads[0].rung_a_near;
        let mut rungs: HashMap<EdgeKey, (VertexKey, VertexKey)> = HashMap::new();
        let mut rail_a_verts: HashSet<VertexKey> = HashSet::new();
        let mut rail_b_verts: HashSet<VertexKey> = HashSet::new();
        for strip in strip_quads {
            for &rung in &[strip.rung_a, strip.rung_b] {
                if rungs.contains_key(&rung) {
                    continue;
                }
                let [e0, e1] = self.edge_vert_keys(rung);
                let (v_a, v_b) = if rail.on_anchor_side(e0, rail_a_anchor) {
                    (e0, e1)
                } else {
                    (e1, e0)
                };
                rail_a_verts.insert(v_a);
                rail_b_verts.insert(v_b);
                rungs.insert(rung, (v_a, v_b));
            }
        }

        let mode = match params.interp_mode {
            RingInterp::Linear => return None,
            RingInterp::Path => {
                let setup = self.rim_path_setup(&rungs, &rail_a_verts);
                RingPlacementMode::Path(setup)
            }
            RingInterp::Surface => {
                let band_faces: HashSet<FaceKey> =
                    strip_quads.iter().map(|s| s.face).collect();
                let tangents =
                    self.rim_surface_tangents(&rungs, &rail_a_verts, &band_faces);
                RingPlacementMode::Surface(tangents)
            }
        };

        Some(RingPlacement { rungs, mode })
    }

    fn edge_other_vert(&self, edge: EdgeKey, vert: VertexKey) -> VertexKey {
        let [a, b] = self.edge_vert_keys(edge);
        if a == vert {
            b
        } else {
            a
        }
    }

    /// §6.2.1 PATH setup: each rim ring's edge-length-weighted centre and its
    /// area-weighted (Newell) ring normal, computed over the ring's verts walked
    /// in loop order. The two normals are then reconciled — flipped to lean
    /// toward each other along the centre-to-centre direction so they read as a
    /// Bézier's end tangents.
    fn rim_path_setup(
        &self,
        rungs: &HashMap<EdgeKey, (VertexKey, VertexKey)>,
        rail_a_verts: &HashSet<VertexKey>,
    ) -> PathSetup {
        // Order the two rim rings *consistently*: walk rail A into loop order,
        // then read rail B column-for-column off the rung pairing. Winding the
        // two rings the same way (both ride the strip in the same direction)
        // makes their raw Newell normals comparably oriented, which is what lets
        // the reconciliation step settle their relative sign even when a normal
        // lands near-perpendicular to the centre-to-centre axis (§6.2.1).
        let mate: HashMap<VertexKey, VertexKey> =
            rungs.values().map(|&(a, b)| (a, b)).collect();
        let ordered_a = self.ordered_rail(rail_a_verts);
        let ring_a: Vec<Vec3> = ordered_a.iter().map(|&v| self.vertex_position(v)).collect();
        let ring_b: Vec<Vec3> = ordered_a
            .iter()
            .map(|&v| self.vertex_position(mate[&v]))
            .collect();

        let center_a = rim_center(&ring_a);
        let center_b = rim_center(&ring_b);
        let between = center_a - center_b;

        let mut normal_a = newell_ring_normal(&ring_a)
            .unwrap_or_else(|| centre_aligned_fallback_normal(&ring_a, between));
        let mut normal_b = newell_ring_normal(&ring_b)
            .unwrap_or_else(|| centre_aligned_fallback_normal(&ring_b, between));

        // Reconcile (§6.2.1): normal_A points back toward B, normal_B toward A.
        if normal_a.dot(between) > 0.0 {
            normal_a = -normal_a;
        }
        if normal_b.dot(between) < 0.0 {
            normal_b = -normal_b;
        }

        PathSetup {
            center_a,
            center_b,
            normal_a,
            normal_b,
        }
    }

    /// Walk one rail into loop order. The rail verts are joined to each other by
    /// the mesh edges running *along* the rail (each such edge has both endpoints
    /// in `verts`); chaining those edges gives the closed (or, for an open strip,
    /// the open) sequence the §6.2.1 Newell normal and edge-length-weighted
    /// centre walk over.
    ///
    /// The start vert and winding direction are pinned by a stable key order
    /// (smallest-key endpoint of an open rail / smallest-key vert of a closed
    /// rail, stepping to the smallest-key unvisited rail neighbour) so the walk
    /// is deterministic rather than `HashSet`-iteration-order-dependent. The
    /// absolute winding direction does not affect the reconciled normal away from
    /// the perpendicular boundary (reversing it negates the raw Newell normal,
    /// which the reconciliation flip then re-aligns) and never affects the
    /// edge-length-weighted centre; pinning it only removes run-to-run jitter in
    /// the near-perpendicular tie-break.
    fn ordered_rail(&self, verts: &HashSet<VertexKey>) -> Vec<VertexKey> {
        let mut neighbours: HashMap<VertexKey, Vec<VertexKey>> = HashMap::new();
        for &v in verts {
            let entry = neighbours.entry(v).or_default();
            for edge in self.vertex_disk_iter(v).collect::<Vec<_>>() {
                let other = self.edge_other_vert(edge, v);
                if verts.contains(&other) && !entry.contains(&other) {
                    entry.push(other);
                }
            }
        }
        for adj in neighbours.values_mut() {
            adj.sort_by_key(|v| v.as_usize());
        }

        // Start at the smallest-key open end (degree-1 rail vert) if the rail is
        // an open polyline; otherwise the smallest-key vert seeds the closed walk.
        let start = verts
            .iter()
            .copied()
            .filter(|v| neighbours.get(v).map_or(0, |n| n.len()) == 1)
            .min_by_key(|v| v.as_usize())
            .or_else(|| verts.iter().copied().min_by_key(|v| v.as_usize()))
            .expect("non-empty rail ring");

        let mut ordered = Vec::with_capacity(verts.len());
        let mut visited: HashSet<VertexKey> = HashSet::new();
        let mut current = start;
        loop {
            ordered.push(current);
            visited.insert(current);
            let next = neighbours
                .get(&current)
                .and_then(|n| n.iter().copied().find(|c| !visited.contains(c)));
            match next {
                Some(v) => current = v,
                None => break,
            }
        }
        ordered
    }

    /// §6.2.3 step 1: snapshot each rim vert's surrounding-surface tangent — the
    /// unit "off-the-rim, into the surface" direction, pointing across the strip
    /// toward the other rail. It is accumulated from a per-face primitive over the
    /// rim vert's **rail edges** (the edges running *along* the strip — §2's
    /// rails — that meet the vert, not the rungs being cut), bucketed by whether
    /// each face belongs to the subdivided band (inner) or surrounds it (outer).
    ///
    /// The snapshot is taken **before** any split: once the band starts cutting,
    /// the faces around a rim vert change, so a tangent estimated mid-operation
    /// would read the half-built mesh (the §6.2.3 feedback hazard).
    fn rim_surface_tangents(
        &self,
        rungs: &HashMap<EdgeKey, (VertexKey, VertexKey)>,
        rail_a_verts: &HashSet<VertexKey>,
        band_faces: &HashSet<FaceKey>,
    ) -> HashMap<VertexKey, Vec3> {
        let rung_set: HashSet<EdgeKey> = rungs.keys().copied().collect();
        let mut tangents: HashMap<VertexKey, Vec3> = HashMap::new();
        for (&_rung, &(v_a, v_b)) in rungs {
            // Chord fallback direction: from each rim vert across the strip
            // toward the other rail. Used only when the bucketed accumulation
            // degenerates (e.g. stale/zero stored face normals, §6.2.3 trap).
            let chord_a = (self.vertex_position(v_b) - self.vertex_position(v_a))
                .normalize_or_zero();
            let chord_b = -chord_a;
            tangents.insert(
                v_a,
                self.surface_tangent_at(v_a, &rung_set, band_faces, chord_a),
            );
            tangents.insert(
                v_b,
                self.surface_tangent_at(v_b, &rung_set, band_faces, chord_b),
            );
        }
        let _ = rail_a_verts;
        tangents
    }

    /// The surrounding-surface tangent at one rim vert (§6.2.3 step 1). Walks the
    /// vert's **rail edges** — its incident edges that are not rungs and border
    /// the subdivided band — and, for each face on such an edge, accumulates the
    /// in-face edge tangent `normalize((v_first − v_second) × face_normal)` (the
    /// edge taken in the face's loop-winding order so the sign points into the
    /// face). The contributions are bucketed:
    ///
    /// - **inner** — the band faces themselves, and
    /// - **outer** — the surrounding faces outside the band.
    ///
    /// The final tangent is `normalize(−t_outer)` when any outer face is present
    /// (the negation flips it from "out of the strip" to "across toward the other
    /// rail"); otherwise — a boundary rim vert with only its band face — the
    /// normalised inner sum `normalize(t_inner)`.
    ///
    /// **Reads stored face normals** (§6.2.3 trap): in a mesh whose normals are
    /// stale/zero every primitive is zero and the accumulation collapses; the
    /// `chord_dir` fallback then keeps the handle direction defined.
    fn surface_tangent_at(
        &self,
        vert: VertexKey,
        rung_set: &HashSet<EdgeKey>,
        band_faces: &HashSet<FaceKey>,
        chord_dir: Vec3,
    ) -> Vec3 {
        let mut t_inner = Vec3::ZERO;
        let mut t_outer = Vec3::ZERO;
        let mut has_outer = false;
        for rail_edge in self.vertex_disk_iter(vert).collect::<Vec<_>>() {
            if rung_set.contains(&rail_edge) {
                continue;
            }
            // A rail edge borders the band; rungs are skipped above. Edges that
            // touch no band face are not rails of this strip — ignore them.
            let touches_band = self
                .edge_face_keys(rail_edge)
                .iter()
                .any(|f| band_faces.contains(f));
            if !touches_band {
                continue;
            }
            for loop_key in self.edge_radial_iter(rail_edge).collect::<Vec<_>>() {
                let tangent = self.in_face_edge_tangent(loop_key);
                let face = self.loop_face(loop_key);
                if band_faces.contains(&face) {
                    t_inner += tangent;
                } else {
                    t_outer += tangent;
                    has_outer = true;
                }
            }
        }

        let tan = if has_outer {
            (-t_outer).normalize_or_zero()
        } else {
            t_inner.normalize_or_zero()
        };
        if tan.length_squared() > 0.5 {
            tan
        } else {
            chord_dir
        }
    }

    /// The in-face edge tangent at one loop (§6.2.3 step 1 primitive): the unit
    /// vector in the loop's face plane, perpendicular to the loop's edge, pointing
    /// into the face interior — `normalize((v_first − v_second) × face_normal)`,
    /// with `v_first → v_second` the loop's edge in face-winding order and
    /// `face_normal` the face's **stored** normal.
    fn in_face_edge_tangent(&self, loop_key: LoopKey) -> Vec3 {
        let v_first = self.loop_vertex(loop_key);
        let v_second = self.loop_vertex(self.loop_next(loop_key));
        let edge_vec = self.vertex_position(v_first) - self.vertex_position(v_second);
        let face = self.loop_face(loop_key);
        edge_vec.cross(self.face_normal(face)).normalize_or_zero()
    }

    /// Apply the §6.2 PATH / SURFACE placement, overwriting the LINEAR chord
    /// positions the inserted verts were born with. Positions-only; the topology
    /// is untouched.
    fn apply_ring_placement(
        &mut self,
        placement: &RingPlacement,
        rung_cuts: &HashMap<EdgeKey, RungCuts>,
        cuts: u32,
        smooth: f32,
    ) {
        match &placement.mode {
            RingPlacementMode::Path(setup) => {
                self.apply_path_placement(placement, rung_cuts, setup, cuts, smooth)
            }
            RingPlacementMode::Surface(tangents) => {
                self.apply_surface_placement(placement, rung_cuts, tangents, cuts, smooth)
            }
        }
    }

    /// §6.2.2 PATH. One centre spline for the whole strip, a minimum-twist frame
    /// per level, a per-level reference triangle, and a two-sided
    /// **triangle-to-triangle** carry of each rung's rim endpoints into the level
    /// triangle (`v_A` from the A-end triangle, `v_B` from the B-end triangle),
    /// blended at the level parameter `t_i`.
    ///
    /// `resolu = N + 2` is the number of levels including the two ends; levels
    /// run `i = 0 .. resolu - 1`, with `i = 0` the A end, `i = resolu - 1` the B
    /// end, and `i = 1 .. N` the inserted interior levels.
    fn apply_path_placement(
        &mut self,
        placement: &RingPlacement,
        rung_cuts: &HashMap<EdgeKey, RungCuts>,
        setup: &PathSetup,
        cuts: u32,
        smooth: f32,
    ) {
        let levels = cuts as usize;
        let resolu = levels + 2;
        let last = resolu - 1;

        // Step 1 — the centre spline (Bézier), sampled at the `resolu` even
        // parameters. At smooth = 0 the handles collapse onto the centres → the
        // straight chord of centres, but PATH still diverges from LINEAR via the
        // frames and the triangle transport below.
        let handle_len =
            handle_length(setup.center_a, setup.center_b, setup.normal_a, setup.normal_b);
        let handle_a = setup.center_a + smooth * handle_len * setup.normal_a;
        let handle_b = setup.center_b + smooth * handle_len * setup.normal_b;
        let path: Vec<Vec3> = (0..resolu)
            .map(|i| {
                let t = i as f32 / last as f32;
                bezier(setup.center_a, handle_a, handle_b, setup.center_b, t)
            })
            .collect();

        // Step 2 — per-level directions and the minimum-twist frame sweep. The
        // end directions are the reconciled rim normals; the interior ones are
        // the path's local tangents.
        let mut dir = vec![Vec3::ZERO; resolu];
        dir[0] = setup.normal_a;
        dir[last] = -setup.normal_b;
        for i in 1..last {
            dir[i] = path_bisector(path[i - 1], path[i], path[i + 1]);
        }
        let frames = minimum_twist_frames(&dir);

        // Step 3 — one reference triangle per level (unit legs along local +X
        // then +Y at `path[i]`; `s = 1` at the default profile). The triangle
        // carries the frame's position and orientation; its vertex order pins the
        // handedness of the transport.
        let shape_size = 1.0;
        let tris: Vec<RefTriangle> = (0..resolu)
            .map(|i| reference_triangle(frames[i], path[i], shape_size))
            .collect();

        // Transport each rung's rim endpoints into each interior level's triangle
        // and blend. `v_A` is read in the A-end triangle (level 0), `v_B` in the
        // B-end triangle (level `last`), each replanted into level `i`'s triangle,
        // then lerped at `t_i = i / (resolu - 1)`.
        for (&rung, &(v_a, v_b)) in &placement.rungs {
            let pos_a = self.vertex_position(v_a);
            let pos_b = self.vertex_position(v_b);
            let inserted = rung_cuts[&rung].aligned(v_a);
            for (idx, &vert) in inserted.iter().enumerate() {
                let i = idx + 1;
                let t = i as f32 / last as f32;
                let co_a = tri_transform(pos_a, &tris[0], &tris[i]);
                let co_b = tri_transform(pos_b, &tris[last], &tris[i]);
                self.set_vertex_position(vert, co_a.lerp(co_b, t));
            }
        }
    }

    /// §6.2.3 SURFACE. Each inserted cut ring independently on its own cubic
    /// meeting both rims tangentially; no shared spline, no frame sweep.
    fn apply_surface_placement(
        &mut self,
        placement: &RingPlacement,
        rung_cuts: &HashMap<EdgeKey, RungCuts>,
        tangents: &HashMap<VertexKey, Vec3>,
        cuts: u32,
        smooth: f32,
    ) {
        let levels = cuts as usize;
        let segments = levels + 1;

        for (&rung, &(v_a, v_b)) in &placement.rungs {
            let pos_a = self.vertex_position(v_a);
            let pos_b = self.vertex_position(v_b);
            let tan_a = tangents[&v_a];
            let tan_b = tangents[&v_b];
            // At smooth = 0 the handles collapse onto the endpoints → control
            // polygon (v_a, v_a, v_b, v_b): geometrically the chord line, but
            // the forward-difference cubic samples it as smoothstep B(t), not
            // the uniform lerp, re-spacing interior verts (§6.2.3).
            let handle_len = handle_length(pos_a, pos_b, tan_a, tan_b);
            let handle_a = pos_a + smooth * handle_len * tan_a;
            let handle_b = pos_b + smooth * handle_len * tan_b;

            let inserted = rung_cuts[&rung].aligned(v_a);
            for (idx, &vert) in inserted.iter().enumerate() {
                let level = idx + 1;
                let t = level as f32 / segments as f32;
                let co = bezier(pos_a, handle_a, handle_b, pos_b, t);
                self.set_vertex_position(vert, co);
            }
        }
    }

    /// Carve one (multi-cut) strip quad into `N + 1` rows (§6.1.2). `levels_a`
    /// and `levels_b` are the `N` midpoints of the two rungs in cross-strip
    /// order (level 0 nearest the A-rail). Each level's cross-cut peels one quad
    /// row off the A-rail end; the remaining fragment — the one still carrying
    /// the deeper levels and the B-rail — is split again. `b_rail_corner` is the
    /// B-rail-side corner (`rung_a`'s far endpoint) used to tell the deeper
    /// fragment from the freshly peeled row at every level, including the last.
    /// The carved-off rows and the final B-rail remainder are pushed to
    /// `out_faces` (`N + 1` quads).
    fn fill_strip_quad(
        &mut self,
        face: FaceKey,
        levels_a: &[VertexKey],
        levels_b: &[VertexKey],
        b_rail_corner: VertexKey,
        out_faces: &mut Vec<FaceKey>,
    ) {
        let mut remaining = face;
        for (&m_a, &m_b) in levels_a.iter().zip(levels_b.iter()) {
            // face_split returns None only when a corner is absent from the
            // face — an invariant violation here, since every level vert was
            // inserted onto this quad's rim in phase one (§6.1.1). Guard it so
            // an unexpected None surfaces rather than silently dropping a row.
            let split = self.face_split(remaining, m_a, m_b);
            debug_assert!(
                split.is_some(),
                "level cross-cut corner missing from strip quad"
            );
            let Some((new_face, _new_edge)) = split else {
                return;
            };

            // One of {remaining, new_face} is the freshly peeled A-side row; the
            // other still holds the deeper levels + the B-rail. The deeper
            // fragment is the one carrying the B-rail corner; the row is the
            // other. This probe is independent of which side SFME keeps as the
            // surviving face, and works at every level (no special last-level
            // case).
            let (row, deeper) = if self.face_contains_vert(new_face, b_rail_corner) {
                (remaining, new_face)
            } else {
                (new_face, remaining)
            };
            out_faces.push(row);
            remaining = deeper;
        }
        // The final remainder is the B-rail row.
        out_faces.push(remaining);
    }

    /// Whether `vert` is a corner of `face`.
    fn face_contains_vert(&self, face: FaceKey, vert: VertexKey) -> bool {
        self.face_vert_keys(face).contains(&vert)
    }

    /// Gather the strip quads of the resolved ring (§2 / §4): the quad faces
    /// flanked by two *opposite* rungs of `rung_set`. Each such face is
    /// captured once, with the two rungs that flank it and the rail pairing of
    /// their endpoints (§6.1.1) so phase two can align cut levels.
    fn collect_strip_quads(&self, rung_set: &HashSet<EdgeKey>) -> Vec<StripQuad> {
        // Candidate faces: every face incident to a rung. A strip quad is seen
        // from both its rungs, so dedupe before classifying.
        let mut candidates: HashSet<FaceKey> = HashSet::new();
        for &rung in rung_set {
            for face in self.edge_face_keys(rung) {
                candidates.insert(face);
            }
        }

        let mut strip_quads = Vec::new();
        for face in candidates {
            if self.face_len(face) != 4 {
                continue;
            }
            // A quad's verts/edges sit at loop positions 0..4; edge `i` joins
            // vert `i` to vert `i+1`. Edge positions 0 and 2 are one opposite
            // pair, 1 and 3 the other. A strip quad is flanked by exactly one
            // opposite pair of rungs. The rail joining the two rungs' "near"
            // endpoints pins the level-0 alignment for phase two: with rungs at
            // edge 0 (verts v0→v1) and edge 2 (verts v2→v3), rail edge 3
            // (v3→v0) joins rung_a's v0 to rung_b's v3, so those are the
            // rail-mate near endpoints. (The symmetric tie-break — rungs at
            // edges 1/3 — pairs v1↔v2 the same way. Choosing edge-0/2 first when
            // both pairs are rungs is arbitrary but safe: a strip quad can be
            // flanked by only one opposite rung pair, so the branches are
            // mutually exclusive in practice and the choice never observably
            // matters.)
            let verts = self.face_vert_keys(face);
            let edges = self.face_edge_keys(face);
            debug_assert_eq!(edges.len(), 4, "quad has four edges");
            debug_assert_eq!(verts.len(), 4, "quad has four verts");
            if rung_set.contains(&edges[0]) && rung_set.contains(&edges[2]) {
                strip_quads.push(StripQuad {
                    face,
                    rung_a: edges[0],
                    rung_b: edges[2],
                    rung_a_near: verts[0],
                    rung_b_near: verts[3],
                    rung_a_far: verts[1],
                });
            } else if rung_set.contains(&edges[1]) && rung_set.contains(&edges[3]) {
                strip_quads.push(StripQuad {
                    face,
                    rung_a: edges[1],
                    rung_b: edges[3],
                    rung_a_near: verts[1],
                    rung_b_near: verts[0],
                    rung_a_far: verts[2],
                });
            }
        }
        strip_quads
    }
}

#[cfg(test)]
mod tests {
    use glam::Vec3;

    use super::*;

    fn counts(mesh: &LoopMesh) -> (usize, usize, usize) {
        (mesh.totvert(), mesh.totedge(), mesh.totface())
    }

    /// Build an open 1×k quad strip in the z = 0 plane. Bottom row verts at
    /// y = 0, top row at y = 1. Returns the mesh and the `k + 1` rung edges
    /// (bottom_i ↔ top_i), in column order.
    fn open_strip(k: usize) -> (LoopMesh, Vec<EdgeKey>) {
        let mut mesh = LoopMesh::new();
        let cols = k + 1;
        let bottom: Vec<VertexKey> = (0..cols)
            .map(|i| mesh.create_vertex(Vec3::new(i as f32, 0.0, 0.0)))
            .collect();
        let top: Vec<VertexKey> = (0..cols)
            .map(|i| mesh.create_vertex(Vec3::new(i as f32, 1.0, 0.0)))
            .collect();
        let rungs: Vec<EdgeKey> = (0..cols)
            .map(|i| mesh.create_edge([bottom[i], top[i]], EdgeCreateFlags::NONE))
            .collect();
        for q in 0..k {
            let b0 = bottom[q];
            let b1 = bottom[q + 1];
            let t1 = top[q + 1];
            let t0 = top[q];
            let e_bottom = mesh.create_edge([b0, b1], EdgeCreateFlags::NONE);
            let e_right = rungs[q + 1];
            let e_top = mesh.create_edge([t1, t0], EdgeCreateFlags::NONE);
            let e_left = rungs[q];
            mesh.face_create(
                &[b0, b1, t1, t0],
                &[e_bottom, e_right, e_top, e_left],
                FaceCreateFlags::NONE,
            );
        }
        (mesh, rungs)
    }

    /// Build a closed band of `k` quads (a prism ring). Lower ring at y = 0,
    /// upper ring at y = 1, `k` rungs, quad q wraps q → (q+1) % k.
    fn closed_band(k: usize) -> (LoopMesh, Vec<EdgeKey>) {
        let mut mesh = LoopMesh::new();
        let lower: Vec<VertexKey> = (0..k)
            .map(|i| {
                let theta = i as f32 / k as f32 * std::f32::consts::TAU;
                mesh.create_vertex(Vec3::new(theta.cos(), 0.0, theta.sin()))
            })
            .collect();
        let upper: Vec<VertexKey> = (0..k)
            .map(|i| {
                let theta = i as f32 / k as f32 * std::f32::consts::TAU;
                mesh.create_vertex(Vec3::new(theta.cos(), 1.0, theta.sin()))
            })
            .collect();
        let rungs: Vec<EdgeKey> = (0..k)
            .map(|i| mesh.create_edge([lower[i], upper[i]], EdgeCreateFlags::NONE))
            .collect();
        for q in 0..k {
            let q1 = (q + 1) % k;
            let l0 = lower[q];
            let l1 = lower[q1];
            let u1 = upper[q1];
            let u0 = upper[q];
            let e_bottom = mesh.create_edge([l0, l1], EdgeCreateFlags::NONE);
            let e_right = rungs[q1];
            let e_top = mesh.create_edge([u1, u0], EdgeCreateFlags::NONE);
            let e_left = rungs[q];
            mesh.face_create(
                &[l0, l1, u1, u0],
                &[e_bottom, e_right, e_top, e_left],
                FaceCreateFlags::NONE,
            );
        }
        (mesh, rungs)
    }

    #[test]
    fn open_strip_single_ring_deltas() {
        // §4.1 open delta (k = 3): 8 V / 10 E / 3 F → 12 V / 17 E / 6 F.
        let (mut mesh, rungs) = open_strip(3);
        assert_eq!(counts(&mesh), (8, 10, 3), "starting 1×3 open strip");

        let result = mesh.subdivide_edgering(&rungs, SubdivideEdgeRingParams::default());

        assert_eq!(counts(&mesh), (12, 17, 6), "open k=3: +4 V / +7 E / +3 F");
        assert_eq!(result.verts.len(), 4, "one midpoint per rung");
        assert_eq!(result.faces.len(), 6, "three strip quads → six fill quads");
    }

    #[test]
    fn open_strip_fill_faces_are_quads() {
        // §4 — every strip quad splits into two quads (A-side + B-side).
        let (mut mesh, rungs) = open_strip(3);
        let result = mesh.subdivide_edgering(&rungs, SubdivideEdgeRingParams::default());
        for &f in &result.faces {
            assert_eq!(mesh.face_len(f), 4, "fill fragment is a quad");
        }
    }

    #[test]
    fn open_strip_midpoints_on_chord() {
        // §5 — each new ring vert lands at the chord midpoint of its rung. The
        // rungs run y = 0 → y = 1, so every midpoint sits at y = 0.5.
        let (mut mesh, rungs) = open_strip(3);
        let result = mesh.subdivide_edgering(&rungs, SubdivideEdgeRingParams::default());
        for &m in &result.verts {
            let p = mesh.vertex_position(m);
            assert!((p.y - 0.5).abs() < 1e-6, "midpoint on the chord at t = 0.5");
            assert!(p.z.abs() < 1e-6, "stays in the strip plane");
        }
        // The four midpoints span the four rung columns x = 0..=3.
        let mut xs: Vec<f32> = result.verts.iter().map(|&m| mesh.vertex_position(m).x).collect();
        xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        for (i, x) in xs.iter().enumerate() {
            assert!((x - i as f32).abs() < 1e-6, "midpoint column {i}");
        }
    }

    #[test]
    fn closed_band_single_ring_deltas() {
        // §4.1 closed delta (k = 4): 8 V / 12 E / 4 F → 12 V / 20 E / 8 F. The
        // wrap-around connecting edge closes the new ring.
        let (mut mesh, rungs) = closed_band(4);
        assert_eq!(counts(&mesh), (8, 12, 4), "starting closed 4-quad band");

        let result = mesh.subdivide_edgering(&rungs, SubdivideEdgeRingParams::default());

        assert_eq!(counts(&mesh), (12, 20, 8), "closed k=4: +4 V / +8 E / +4 F");
        assert_eq!(result.verts.len(), 4, "one midpoint per rung");
        assert_eq!(result.faces.len(), 8, "four band quads → eight fill quads");
        for &f in &result.faces {
            assert_eq!(mesh.face_len(f), 4, "fill fragment is a quad");
        }
    }

    #[test]
    fn new_ring_is_continuous() {
        // §4 — the inserted midpoints are stitched into a continuous ring: each
        // consecutive pair of midpoints is joined by a connecting edge. For the
        // closed band that ring closes (k midpoints, k connecting edges).
        let (mut mesh, rungs) = closed_band(4);
        let result = mesh.subdivide_edgering(&rungs, SubdivideEdgeRingParams::default());

        let mut connect_edges = 0;
        for &m in &result.verts {
            for e in mesh.vertex_disk_iter(m).collect::<Vec<_>>() {
                let [a, b] = mesh.edge_vert_keys(e);
                let other = if a == m { b } else { a };
                if result.verts.contains(&other) {
                    connect_edges += 1;
                }
            }
        }
        // Each connecting edge is counted from both endpoints.
        assert_eq!(connect_edges / 2, 4, "closed new ring has k connecting edges");
    }

    #[test]
    fn empty_rung_set_is_noop() {
        let (mut mesh, _rungs) = open_strip(3);
        let result = mesh.subdivide_edgering(&[], SubdivideEdgeRingParams::default());
        assert_eq!(counts(&mesh), (8, 10, 3), "empty set: mesh unchanged");
        assert!(result.verts.is_empty() && result.faces.is_empty());
    }

    fn cuts(n: u32) -> SubdivideEdgeRingParams {
        SubdivideEdgeRingParams {
            cuts: n,
            ..Default::default()
        }
    }

    #[test]
    fn open_strip_cuts2_deltas() {
        // §6.1.3 open (k = 3, N = 2): 2× the §4.1 open deltas.
        // 8 V / 10 E / 3 F → 16 V / 24 E / 9 F (+8 V / +14 E / +6 F).
        let (mut mesh, rungs) = open_strip(3);
        assert_eq!(counts(&mesh), (8, 10, 3));

        let result = mesh.subdivide_edgering(&rungs, cuts(2));

        assert_eq!(counts(&mesh), (16, 24, 9), "open k=3, N=2");
        assert_eq!(result.verts.len(), 8, "N midpoints per rung: 2 × 4 rungs");
        // (N+1)·k fill quads = 3 × 3 = 9.
        assert_eq!(result.faces.len(), 9, "(N+1)·k fill quads");
        for &f in &result.faces {
            assert_eq!(mesh.face_len(f), 4, "every fill fragment is a quad");
        }
    }

    #[test]
    fn open_strip_cuts3_deltas() {
        // §6.1.3 open (k = 3, N = 3): 8 V / 10 E / 3 F → 20 V / 31 E / 12 F.
        let (mut mesh, rungs) = open_strip(3);
        let result = mesh.subdivide_edgering(&rungs, cuts(3));
        assert_eq!(counts(&mesh), (20, 31, 12), "open k=3, N=3");
        assert_eq!(result.verts.len(), 12, "3 × 4 rungs");
        assert_eq!(result.faces.len(), 12, "(N+1)·k = 4 × 3");
    }

    #[test]
    fn open_strip_cuts2_rings_on_chord() {
        // §6.1.4 — the N interior chord points land at t = 1/3 and t = 2/3 along
        // each rung. Rungs run y = 0 → y = 1, so the cut verts split into a ring
        // at y = 1/3 and a ring at y = 2/3, each spanning the four columns.
        let (mut mesh, rungs) = open_strip(3);
        let result = mesh.subdivide_edgering(&rungs, cuts(2));

        let mut at_third = 0;
        let mut at_two_thirds = 0;
        for &m in &result.verts {
            let p = mesh.vertex_position(m);
            assert!(p.z.abs() < 1e-6, "stays in the strip plane");
            if (p.y - 1.0 / 3.0).abs() < 1e-6 {
                at_third += 1;
            } else if (p.y - 2.0 / 3.0).abs() < 1e-6 {
                at_two_thirds += 1;
            } else {
                panic!("cut vert off the expected chord rings: y = {}", p.y);
            }
        }
        assert_eq!(at_third, 4, "first ring: one vert per rung column");
        assert_eq!(at_two_thirds, 4, "second ring: one vert per rung column");
    }

    #[test]
    fn closed_band_cuts3_deltas() {
        // §6.1.3 closed (k = 4, N = 3): 2×§4.1-closed scaled by N.
        // 8 V / 12 E / 4 F → 20 V / 36 E / 16 F (+12 V / +24 E / +12 F).
        let (mut mesh, rungs) = closed_band(4);
        assert_eq!(counts(&mesh), (8, 12, 4));

        let result = mesh.subdivide_edgering(&rungs, cuts(3));

        assert_eq!(counts(&mesh), (20, 36, 16), "closed k=4, N=3");
        assert_eq!(result.verts.len(), 12, "N × k = 3 × 4");
        assert_eq!(result.faces.len(), 16, "(N+1)·k = 4 × 4");
        for &f in &result.faces {
            assert_eq!(mesh.face_len(f), 4, "every fill fragment is a quad");
        }
    }

    #[test]
    fn closed_band_cuts2_rings_continuous() {
        // §6.1.2 — the N midpoint families stitch into N continuous closed
        // rings, one per cut level. Each ring's connecting edges join two cut
        // verts at the *same* level (same chord fraction, hence same y here);
        // inter-level rung sub-edges join cut verts at *different* levels and
        // are not ring edges. For N = 2 closed rings over k = 4: each ring has
        // k = 4 connecting edges. Each ring vertex must have degree 2 within
        // its own level (a closed loop), confirming continuity.
        let (mut mesh, rungs) = closed_band(4);
        let result = mesh.subdivide_edgering(&rungs, cuts(2));

        let level_of = |y: f32| -> Option<u8> {
            if (y - 1.0 / 3.0).abs() < 1e-6 {
                Some(0)
            } else if (y - 2.0 / 3.0).abs() < 1e-6 {
                Some(1)
            } else {
                None
            }
        };

        let mut ring_edges = [0usize; 2];
        let mut same_level_degree: HashMap<VertexKey, usize> = HashMap::new();
        for &m in &result.verts {
            let m_level = level_of(mesh.vertex_position(m).y).expect("cut vert on a ring");
            for e in mesh.vertex_disk_iter(m).collect::<Vec<_>>() {
                let [a, b] = mesh.edge_vert_keys(e);
                let other = if a == m { b } else { a };
                if let Some(other_level) = level_of(mesh.vertex_position(other).y) {
                    if other_level == m_level {
                        ring_edges[m_level as usize] += 1;
                        *same_level_degree.entry(m).or_default() += 1;
                    }
                }
            }
        }
        assert_eq!(ring_edges[0] / 2, 4, "ring 0 has k connecting edges");
        assert_eq!(ring_edges[1] / 2, 4, "ring 1 has k connecting edges");
        for (&_v, &deg) in &same_level_degree {
            assert_eq!(deg, 2, "each ring vert has degree 2 (closed loop)");
        }
    }

    #[test]
    fn cuts1_matches_core_path() {
        // N = 1 must collapse onto the §4 single-cut fill: open k = 3 lands the
        // same 12 V / 17 E / 6 F as the default-params core.
        let (mut mesh, rungs) = open_strip(3);
        let result = mesh.subdivide_edgering(&rungs, cuts(1));
        assert_eq!(counts(&mesh), (12, 17, 6), "N=1 ≡ core single cut");
        assert_eq!(result.verts.len(), 4);
        assert_eq!(result.faces.len(), 6);
    }

    #[test]
    fn cuts_zero_is_noop() {
        let (mut mesh, rungs) = open_strip(3);
        let result = mesh.subdivide_edgering(&rungs, cuts(0));
        assert_eq!(counts(&mesh), (8, 10, 3), "cuts = 0: mesh unchanged");
        assert!(result.verts.is_empty() && result.faces.is_empty());
    }

    // ----- §6.2 non-linear placement (PATH / SURFACE) -----

    fn params(interp: RingInterp, cuts: u32, smooth: f32) -> SubdivideEdgeRingParams {
        SubdivideEdgeRingParams {
            cuts,
            interp_mode: interp,
            smooth,
            profile_shape_factor: 0.0,
        }
    }

    /// Build an open 1×k strip twisted about its length (x) axis: rung `i` is
    /// rotated by `i · step` radians, so the two rails spiral into non-parallel
    /// planes (the §7 PATH fixture condition). The rails are also non-collinear
    /// (each spans 2D), so each rim has a real plane normal rather than the
    /// §6.2.1 centre-to-centre fallback.
    fn twisted_strip(k: usize, step: f32) -> (LoopMesh, Vec<EdgeKey>) {
        let mut mesh = LoopMesh::new();
        let cols = k + 1;
        // Rung axis starts along +y; rotate it about x by i·step, around the
        // strip centreline at y = 0.5.
        let mut rail_a = Vec::with_capacity(cols);
        let mut rail_b = Vec::with_capacity(cols);
        for i in 0..cols {
            let a = i as f32 * step;
            let (s, c) = a.sin_cos();
            // half-rung offset from the centreline, rotated about x.
            let off = Vec3::new(0.0, 0.5 * c, 0.5 * s);
            let center = Vec3::new(i as f32, 0.5, 0.0);
            rail_a.push(mesh.create_vertex(center - off));
            rail_b.push(mesh.create_vertex(center + off));
        }
        let rungs: Vec<EdgeKey> = (0..cols)
            .map(|i| mesh.create_edge([rail_a[i], rail_b[i]], EdgeCreateFlags::NONE))
            .collect();
        for q in 0..k {
            let a0 = rail_a[q];
            let a1 = rail_a[q + 1];
            let b1 = rail_b[q + 1];
            let b0 = rail_b[q];
            let e_bottom = mesh.create_edge([a0, a1], EdgeCreateFlags::NONE);
            let e_right = rungs[q + 1];
            let e_top = mesh.create_edge([b1, b0], EdgeCreateFlags::NONE);
            let e_left = rungs[q];
            mesh.face_create(
                &[a0, a1, b1, b0],
                &[e_bottom, e_right, e_top, e_left],
                FaceCreateFlags::NONE,
            );
        }
        (mesh, rungs)
    }

    /// Snapshot the cut-vert positions of a subdivide-edgering run keyed by the
    /// rounded `(x)` rung column so PATH/SURFACE and LINEAR runs of the *same*
    /// fixture can be compared cut-for-cut.
    fn cut_positions(mesh: &LoopMesh, verts: &[VertexKey]) -> Vec<Vec3> {
        let mut ps: Vec<Vec3> = verts.iter().map(|&v| mesh.vertex_position(v)).collect();
        ps.sort_by(|a, b| {
            (a.x, a.y, a.z)
                .partial_cmp(&(b.x, b.y, b.z))
                .unwrap()
        });
        ps
    }

    #[test]
    fn path_equals_linear_on_flat_strip() {
        // §6.2.4 — on a non-bending strip every minimum-twist frame is identical
        // (rim normals equal, path tangent agrees), so PATH coincides with
        // LINEAR. The flat open strip is the one case where PATH is vacuous.
        let (mut path_mesh, path_rungs) = open_strip(3);
        let path = path_mesh.subdivide_edgering(&path_rungs, params(RingInterp::Path, 1, 0.0));

        let (mut lin_mesh, lin_rungs) = open_strip(3);
        let lin = lin_mesh.subdivide_edgering(&lin_rungs, params(RingInterp::Linear, 1, 0.0));

        let pp = cut_positions(&path_mesh, &path.verts);
        let lp = cut_positions(&lin_mesh, &lin.verts);
        assert_eq!(pp.len(), lp.len());
        for (a, b) in pp.iter().zip(lp.iter()) {
            assert!(
                a.distance(*b) < 1e-5,
                "PATH must match LINEAR on a flat strip: {a} vs {b}"
            );
        }
    }

    #[test]
    fn path_diverges_off_chord_at_smooth_zero_on_twist() {
        // §6.2.2 — PATH diverges from LINEAR as soon as the strip bends/twists
        // along its length, *even at smooth = 0* (the divergence is the rim-frame
        // rotation, not a bulge). On a twisted strip at least one cut vert must
        // leave the LINEAR chord midpoint.
        let step = 0.5;
        let (mut path_mesh, path_rungs) = twisted_strip(4, step);
        let path = path_mesh.subdivide_edgering(&path_rungs, params(RingInterp::Path, 1, 0.0));

        let (mut lin_mesh, lin_rungs) = twisted_strip(4, step);
        let lin = lin_mesh.subdivide_edgering(&lin_rungs, params(RingInterp::Linear, 1, 0.0));

        let pp = cut_positions(&path_mesh, &path.verts);
        let lp = cut_positions(&lin_mesh, &lin.verts);
        let max_div = pp
            .iter()
            .zip(lp.iter())
            .map(|(a, b)| a.distance(*b))
            .fold(0.0_f32, f32::max);
        assert!(
            max_div > 1e-3,
            "PATH at smooth = 0 must diverge off the LINEAR chord on a twisted \
             strip (§6.2.2); max divergence was {max_div}"
        );
    }

    /// Independent, from-the-guide reimplementation of the §6.2.2 PATH placement
    /// at `smooth = 0`, `cuts = 1`: rebuild the centre spline (straight chord of
    /// centres at smooth = 0), the per-level directions, the curve-framing seed +
    /// minimum-twist sweep, the per-level unit reference triangles, and the
    /// two-sided triangle transport blended at `t = 1/2`. Computed entirely from
    /// the rim vert positions, with no call into the operator's own placement
    /// helpers — so a regression in the operator's PATH math is caught by a
    /// second, independent realisation of the same recipe.
    fn reference_path_midpoint(
        rail_a: &[Vec3],
        rail_b: &[Vec3],
        rung_a: Vec3,
        rung_b: Vec3,
    ) -> Vec3 {
        // Edge-length-weighted ring centre (§6.2.1), recomputed here from the
        // ring's loop-ordered verts independently of the operator: each vert is
        // weighted by the sum of its two incident ring edges' lengths (closed
        // ring, last wraps to first).
        let weighted_center = |vs: &[Vec3]| -> Vec3 {
            let n = vs.len();
            let mut sum = Vec3::ZERO;
            let mut total = 0.0f32;
            for i in 0..n {
                let prev = vs[(i + n - 1) % n];
                let curr = vs[i];
                let next = vs[(i + 1) % n];
                let w = (curr - prev).length() + (next - curr).length();
                sum += w * curr;
                total += w;
            }
            sum / total
        };
        let center_a = weighted_center(rail_a);
        let center_b = weighted_center(rail_b);
        let between = center_a - center_b;

        // Area-weighted Newell ring normal (§6.2.1), recomputed here independently
        // of the operator: accumulate the Newell edge term around the closed ring,
        // then normalise.
        let newell = |vs: &[Vec3]| -> Vec3 {
            let n = vs.len();
            let mut no = Vec3::ZERO;
            for i in 0..n {
                let p = vs[i];
                let q = vs[(i + 1) % n];
                no.x += (p.y - q.y) * (p.z + q.z);
                no.y += (p.z - q.z) * (p.x + q.x);
                no.z += (p.x - q.x) * (p.y + q.y);
            }
            no.normalize_or_zero()
        };
        let mut normal_a = newell(rail_a);
        let mut normal_b = newell(rail_b);
        if normal_a.dot(between) > 0.0 {
            normal_a = -normal_a;
        }
        if normal_b.dot(between) < 0.0 {
            normal_b = -normal_b;
        }

        // smooth = 0: straight chord of centres at resolu = 3 levels.
        let path = [center_a, 0.5 * (center_a + center_b), center_b];
        let dir = [normal_a, (center_b - center_a).normalize(), -normal_b];

        // Curve-framing seed (+Z to dir[0], +Y up) then minimum-twist sweep.
        let seed = |d: Vec3| -> Quat {
            let z = d.normalize();
            let up = if z.cross(Vec3::Y).length_squared() < 1e-6 {
                Vec3::X
            } else {
                Vec3::Y
            };
            let x = up.cross(z).normalize();
            let y = z.cross(x);
            Quat::from_mat3(&Mat3::from_cols(x, y, z))
        };
        let arc = |from: Vec3, to: Vec3| -> Quat {
            let f = from.normalize();
            let t = to.normalize();
            let d = f.dot(t).clamp(-1.0, 1.0);
            if d > 1.0 - 1e-6 {
                Quat::IDENTITY
            } else {
                Quat::from_axis_angle(f.cross(t).normalize(), d.acos())
            }
        };
        let frame0 = seed(dir[0]);
        let frame1 = arc(dir[0], dir[1]) * frame0;
        let frame2 = arc(dir[1], dir[2]) * frame1;
        let frames = [frame0, frame1, frame2];

        // Unit reference triangles (legs along local +X then +Y).
        let tri = |f: Quat, p: Vec3| -> [Vec3; 3] {
            [p, f * Vec3::X + p, f * Vec3::Y + p]
        };
        let tris = [
            tri(frames[0], path[0]),
            tri(frames[1], path[1]),
            tri(frames[2], path[2]),
        ];

        // Congruent unit triangles → rigid transport: barycentric image of the
        // legs is an isometry, area ratio 1. Carry via the frame change directly
        // (equivalent to tri_transform at s = 1): local = frame_src^-1·(p - p0_src),
        // image = p0_dst + frame_dst·local.
        let carry = |p: Vec3, src: usize, dst: usize| -> Vec3 {
            let local = frames[src].inverse() * (p - path[src]);
            path[dst] + frames[dst] * local
        };
        let co_a = carry(rung_a, 0, 1);
        let co_b = carry(rung_b, 2, 1);
        co_a.lerp(co_b, 0.5)
    }

    #[test]
    fn path_placement_matches_guide_recipe_at_smooth_zero() {
        // §6.2.2 — pin the *actual* PATH placement (not merely PATH ≠ LINEAR)
        // against an independent from-the-guide reimplementation of the corrected
        // recipe (curve-framing seed roll + per-level reference-triangle transport).
        // This is the in-crate analogue of the cross-backend A/B pin; it catches
        // a regression in either the seed-roll convention or the triangle map.
        let step = 0.5;
        let (mut mesh, rungs) = twisted_strip(4, step);

        // Rim verts in column order: rail A = each rung's first endpoint, rail B
        // the second (twisted_strip builds rail_a[i]/rail_b[i] in that order).
        let mut rail_a = Vec::new();
        let mut rail_b = Vec::new();
        let mut rung_pts = Vec::new();
        for &r in &rungs {
            let [a, b] = mesh.edge_vert_keys(r);
            let pa = mesh.vertex_position(a);
            let pb = mesh.vertex_position(b);
            rail_a.push(pa);
            rail_b.push(pb);
            rung_pts.push((pa, pb));
        }

        let result = mesh.subdivide_edgering(&rungs, params(RingInterp::Path, 1, 0.0));

        // Which rail the operator labels "A" is a construction choice (it walks
        // the strip quads with a union-find; §6.2.1). The §6.2.2 recipe is
        // orientation-deterministic *given* a rail labelling, so compute the
        // expected placement under both labellings and require the operator to
        // match one consistently (the cross-backend A/B test pins the exact one).
        for &(pa, pb) in &rung_pts {
            let expect_ab = reference_path_midpoint(&rail_a, &rail_b, pa, pb);
            let expect_ba = reference_path_midpoint(&rail_b, &rail_a, pb, pa);
            let chord_mid = 0.5 * (pa + pb);
            let actual = result
                .verts
                .iter()
                .map(|&v| mesh.vertex_position(v))
                .min_by(|p, q| {
                    p.distance(chord_mid)
                        .partial_cmp(&q.distance(chord_mid))
                        .unwrap()
                })
                .unwrap();
            let best = actual.distance(expect_ab).min(actual.distance(expect_ba));
            assert!(
                best < 1e-4,
                "PATH cut must match the §6.2.2 recipe (either rail labelling): \
                 expected {expect_ab} or {expect_ba}, got {actual}"
            );
        }
    }

    #[test]
    fn surface_cuts1_equals_linear_at_smooth_zero() {
        // §6.2.3 / §6.2.4 — for cuts = 1 the lone sample sits at the smoothstep
        // fixed point B(1/2) = 1/2, so SURFACE at smooth = 0 is *exactly* LINEAR
        // (the chord midpoint), even on a twisted strip.
        let step = 0.5;
        let (mut surf_mesh, surf_rungs) = twisted_strip(4, step);
        let surf =
            surf_mesh.subdivide_edgering(&surf_rungs, params(RingInterp::Surface, 1, 0.0));

        let (mut lin_mesh, lin_rungs) = twisted_strip(4, step);
        let lin = lin_mesh.subdivide_edgering(&lin_rungs, params(RingInterp::Linear, 1, 0.0));

        let sp = cut_positions(&surf_mesh, &surf.verts);
        let lp = cut_positions(&lin_mesh, &lin.verts);
        assert_eq!(sp.len(), lp.len());
        for (a, b) in sp.iter().zip(lp.iter()) {
            assert!(
                a.distance(*b) < 1e-5,
                "SURFACE cuts=1 at smooth=0 must equal LINEAR (B(1/2)=1/2): {a} vs {b}"
            );
        }
    }

    #[test]
    fn surface_cuts2_respaces_along_chord_at_smooth_zero() {
        // §6.2.3 — at smooth = 0 SURFACE stays on the chord *line* but the
        // forward-difference cubic samples it as smoothstep B(t) = t²(3−2t), not
        // the uniform lerp. So for cuts = 2 the interior verts bunch toward the
        // centre: B(1/3) ≈ 0.259 / B(2/3) ≈ 0.741, against LINEAR's 1/3 / 2/3.
        // Use the flat strip (rungs run y = 0 → y = 1) so the chord line is the
        // y axis and "on the chord line" is just "z = 0, x = column".
        let (mut surf_mesh, surf_rungs) = open_strip(3);
        let surf =
            surf_mesh.subdivide_edgering(&surf_rungs, params(RingInterp::Surface, 2, 0.0));

        let b13 = {
            let t = 1.0_f32 / 3.0;
            t * t * (3.0 - 2.0 * t)
        };
        let b23 = {
            let t = 2.0_f32 / 3.0;
            t * t * (3.0 - 2.0 * t)
        };
        assert!((b13 - 0.259).abs() < 0.01 && (b23 - 0.741).abs() < 0.01);

        let mut near = 0; // verts at smoothstep B(1/3)
        let mut far = 0; // verts at smoothstep B(2/3)
        for &v in &surf.verts {
            let p = surf_mesh.vertex_position(v);
            assert!(p.z.abs() < 1e-5, "stays on the chord line (z = 0)");
            if (p.y - b13).abs() < 1e-4 {
                near += 1;
            } else if (p.y - b23).abs() < 1e-4 {
                far += 1;
            } else {
                panic!("SURFACE cut vert not at a smoothstep position: y = {}", p.y);
            }
            // And it is *not* at the LINEAR even spacing.
            assert!(
                (p.y - 1.0 / 3.0).abs() > 1e-3 || (p.y - 2.0 / 3.0).abs() > 1e-3,
                "interior vert must be re-spaced away from the even LINEAR t"
            );
        }
        assert_eq!(near, 4, "level-1 ring at B(1/3) across four rungs");
        assert_eq!(far, 4, "level-2 ring at B(2/3) across four rungs");
    }

    #[test]
    fn surface_cuts2_diverges_from_linear_on_flat_strip() {
        // §6.2.4 — even on a flat strip SURFACE at cuts ≥ 2 is *near*, not equal,
        // to LINEAR: the smoothstep re-spacing moves the interior verts along the
        // chord. Pin the non-vacuity directly.
        let (mut surf_mesh, surf_rungs) = open_strip(3);
        let surf =
            surf_mesh.subdivide_edgering(&surf_rungs, params(RingInterp::Surface, 2, 0.0));

        let (mut lin_mesh, lin_rungs) = open_strip(3);
        let lin = lin_mesh.subdivide_edgering(&lin_rungs, params(RingInterp::Linear, 2, 0.0));

        let sp = cut_positions(&surf_mesh, &surf.verts);
        let lp = cut_positions(&lin_mesh, &lin.verts);
        let max_div = sp
            .iter()
            .zip(lp.iter())
            .map(|(a, b)| a.distance(*b))
            .fold(0.0_f32, f32::max);
        assert!(
            max_div > 1e-3,
            "SURFACE cuts=2 must re-space off LINEAR even at smooth=0 (smoothstep); \
             max divergence {max_div}"
        );
    }

    /// Build the §6.2.1.1 / §6.2.3 barrel tube: `k = 6` hexagonal rings stacked
    /// along +z at `z = 0..4` with radii `[1.0, 1.5, 1.7, 1.5, 1.0]`, the rings
    /// bridged by `4` bands of quads. The rungs are the vertical edges; the rails
    /// run around each hexagon. Returns the mesh together with the rung edges of
    /// the **interior band between ring 1 (z = 1, r = 1.5) and ring 2 (z = 2,
    /// r = 1.7)** — the band the worked SURFACE example subdivides. Those rungs'
    /// rim verts each carry a surrounding band (band 0 below ring 1, band 2 above
    /// ring 2), so the surface tangents lean radially off the chord.
    fn barrel_tube() -> (LoopMesh, Vec<EdgeKey>, Vec<VertexKey>) {
        let radii = [1.0_f32, 1.5, 1.7, 1.5, 1.0];
        let k = 6usize;
        let mut mesh = LoopMesh::new();
        // rings[z][i] vert key.
        let mut rings: Vec<Vec<VertexKey>> = Vec::with_capacity(radii.len());
        for (z, &r) in radii.iter().enumerate() {
            let ring: Vec<VertexKey> = (0..k)
                .map(|i| {
                    let theta = i as f32 / k as f32 * std::f32::consts::TAU;
                    mesh.create_vertex(Vec3::new(r * theta.cos(), r * theta.sin(), z as f32))
                })
                .collect();
            rings.push(ring);
        }
        // Vertical rung edges per level boundary: rung_edges[z][i] joins
        // rings[z][i] → rings[z+1][i].
        let mut rung_edges: Vec<Vec<EdgeKey>> = Vec::with_capacity(radii.len() - 1);
        for z in 0..radii.len() - 1 {
            let level: Vec<EdgeKey> = (0..k)
                .map(|i| mesh.create_edge([rings[z][i], rings[z + 1][i]], EdgeCreateFlags::NONE))
                .collect();
            rung_edges.push(level);
        }
        // Horizontal ring edges per ring: ring_edges[z][i] joins
        // rings[z][i] → rings[z][(i+1)%k].
        let mut ring_edges: Vec<Vec<EdgeKey>> = Vec::with_capacity(radii.len());
        for z in 0..radii.len() {
            let level: Vec<EdgeKey> = (0..k)
                .map(|i| {
                    mesh.create_edge(
                        [rings[z][i], rings[z][(i + 1) % k]],
                        EdgeCreateFlags::NONE,
                    )
                })
                .collect();
            ring_edges.push(level);
        }
        // Quad bands. Quad (z, i) has corners ring z..z+1, angle i..i+1, wound so
        // its stored normal (once refreshed) points radially outward.
        for z in 0..radii.len() - 1 {
            for i in 0..k {
                let i1 = (i + 1) % k;
                let v00 = rings[z][i];
                let v01 = rings[z][i1];
                let v11 = rings[z + 1][i1];
                let v10 = rings[z + 1][i];
                let e_bottom = ring_edges[z][i];
                let e_right = rung_edges[z][i1];
                let e_top = ring_edges[z + 1][i];
                let e_left = rung_edges[z][i];
                mesh.face_create(
                    &[v00, v01, v11, v10],
                    &[e_bottom, e_right, e_top, e_left],
                    FaceCreateFlags::NONE,
                );
            }
        }
        // Refresh stored face normals so the §6.2.3 tangent primitive reads a
        // non-degenerate normal (the stored-normals trap, §6.2.3 step 1 / §6.2.4).
        let all_faces = mesh.face_keys();
        mesh.recalc_face_normals(&all_faces, false);

        let band_rungs = rung_edges[1].clone();
        (mesh, band_rungs, rings[1].clone())
    }

    #[test]
    fn surface_smooth_barrel_matches_worked_example() {
        // §6.2.3 step 2 / §6.2.1.1 — the worked SURFACE `smooth = 0.6`, `cuts = 2`
        // barrel: the angle-0 rung (v_A = (1.5,0,1), v_B = (1.7,0,2)) lands its two
        // inserted ring verts at the *normals-current* oracle coordinates
        // (1.59948, 0, 1.30874) at t = 1/3 and (1.68442, 0, 1.68242) at t = 2/3.
        let (mut mesh, rungs, _ring1) = barrel_tube();
        let result = mesh.subdivide_edgering(&rungs, params(RingInterp::Surface, 2, 0.6));

        let target_third = Vec3::new(1.59948, 0.0, 1.30874);
        let target_two_thirds = Vec3::new(1.68442, 0.0, 1.68242);

        // The angle-0 rung's two cut verts are the ones whose (x, z) sit closest
        // to the two targets; y must stay ≈ 0 (the angle-0 plane).
        let mut nearest_third = f32::INFINITY;
        let mut nearest_two_thirds = f32::INFINITY;
        for &v in &result.verts {
            let p = mesh.vertex_position(v);
            nearest_third = nearest_third.min(p.distance(target_third));
            nearest_two_thirds = nearest_two_thirds.min(p.distance(target_two_thirds));
        }
        assert!(
            nearest_third < 1e-3,
            "SURFACE smooth=0.6 t=1/3 vert must hit the worked barrel coordinate \
             {target_third}; nearest was {nearest_third}"
        );
        assert!(
            nearest_two_thirds < 1e-3,
            "SURFACE smooth=0.6 t=2/3 vert must hit the worked barrel coordinate \
             {target_two_thirds}; nearest was {nearest_two_thirds}"
        );
    }

    #[test]
    fn surface_smooth_barrel_bows_off_chord() {
        // §6.2.3 — the `smooth = 0.6` SURFACE cut bows the angle-0 ring verts
        // *off* the chord line v_A → v_B by ≈ 0.037 (t = 1/3) and ≈ 0.047
        // (t = 2/3). Pin the off-chord magnitude directly so a regression that
        // collapses the bow (e.g. zero tangents) is caught even though it would
        // still leave SURFACE ≠ LINEAR via the smoothstep re-spacing.
        let (mut mesh, rungs, _ring1) = barrel_tube();
        let result = mesh.subdivide_edgering(&rungs, params(RingInterp::Surface, 2, 0.6));

        let v_a = Vec3::new(1.5, 0.0, 1.0);
        let v_b = Vec3::new(1.7, 0.0, 2.0);
        let chord_dir = (v_b - v_a).normalize();
        let off_chord = |p: Vec3| -> f32 {
            let rel = p - v_a;
            (rel - rel.dot(chord_dir) * chord_dir).length()
        };

        let mut max_bow = 0.0f32;
        for &v in &result.verts {
            let p = mesh.vertex_position(v);
            if p.y.abs() < 1e-3 {
                max_bow = max_bow.max(off_chord(p));
            }
        }
        assert!(
            max_bow > 0.03,
            "SURFACE smooth=0.6 must bow the angle-0 ring off the chord line \
             (≈0.037..0.049); max off-chord was {max_bow}"
        );
    }

    #[test]
    fn closed_band_path_preserves_topology() {
        // §6.2 is positions-only: PATH must leave the §4.1 closed deltas
        // untouched (only coordinates differ). The closed band's rims are real
        // circles (non-collinear), exercising the area-weighted normal path.
        let (mut mesh, rungs) = closed_band(4);
        let result = mesh.subdivide_edgering(&rungs, params(RingInterp::Path, 1, 0.0));
        assert_eq!(counts(&mesh), (12, 20, 8), "PATH keeps the LINEAR topology");
        assert_eq!(result.verts.len(), 4);
        assert_eq!(result.faces.len(), 8);
        for &f in &result.faces {
            assert_eq!(mesh.face_len(f), 4, "fill fragment still a quad");
        }
    }
}
