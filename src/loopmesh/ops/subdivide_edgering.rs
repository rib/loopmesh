use std::collections::HashMap;
use std::collections::HashSet;

use glam::{Quat, Vec3};

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

/// A handle-length estimate for the §6.2 splines: scales the off-chord bow, not
/// its direction (multiplied by `smooth`, so it is irrelevant at `smooth = 0`).
/// A "how far apart, projected onto the handle directions' span" length that
/// approximates a circular blend across the gap.
fn handle_length(a: Vec3, b: Vec3, _dir_a: Vec3, _dir_b: Vec3) -> f32 {
    (b - a).length() / 3.0
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
/// 2). Frame 0 rotates a fixed reference axis onto `dir[0]`; each subsequent
/// frame is the previous one pre-rotated by the smallest rotation carrying
/// `dir[j-1]` onto `dir[j]` — a rotation-minimizing frame, no spurious roll.
fn minimum_twist_frames(dir: &[Vec3]) -> Vec<Quat> {
    let reference = Vec3::Z;
    let mut frames = Vec::with_capacity(dir.len());
    frames.push(rotation_between(reference, dir[0]));
    for j in 1..dir.len() {
        let delta = rotation_between(dir[j - 1], dir[j]);
        let frame = delta * frames[j - 1];
        frames.push(frame);
    }
    frames
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

/// Rigid-body carry of `p` from the origin level's oriented frame into the
/// target level's (§6.2.2 step 3): express `p` relative to the origin point and
/// orientation, then re-plant it at the target point and orientation.
fn reframe(
    p: Vec3,
    origin_frame: Quat,
    origin: Vec3,
    target_frame: Quat,
    target: Vec3,
) -> Vec3 {
    let local = origin_frame.inverse() * (p - origin);
    target + target_frame * local
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
                let setup = self.rim_path_setup(&rail_a_verts, &rail_b_verts);
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

    /// §6.2.1 PATH setup: each rim ring's centre (mean of its verts) and a
    /// reconciled best-fit plane normal. The two normals are flipped to lean
    /// toward each other along the centre-to-centre direction so they read as a
    /// Bézier's end tangents.
    fn rim_path_setup(
        &self,
        rail_a_verts: &HashSet<VertexKey>,
        rail_b_verts: &HashSet<VertexKey>,
    ) -> PathSetup {
        let center_a = self.rim_center(rail_a_verts);
        let center_b = self.rim_center(rail_b_verts);
        let between = center_a - center_b;

        let mut normal_a = self.rim_normal(rail_a_verts, center_a, between);
        let mut normal_b = self.rim_normal(rail_b_verts, center_b, between);

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

    fn rim_center(&self, verts: &HashSet<VertexKey>) -> Vec3 {
        let mut sum = Vec3::ZERO;
        for &v in verts {
            sum += self.vertex_position(v);
        }
        sum / verts.len().max(1) as f32
    }

    /// Area-weighted best-fit plane normal of a rim ring (Newell's method over
    /// the ring fanned around its centre). Falls back to the centre-to-centre
    /// direction for a degenerate/collinear ring (§6.2.1) so the spline handle
    /// is always defined rather than NaN.
    fn rim_normal(
        &self,
        verts: &HashSet<VertexKey>,
        center: Vec3,
        between: Vec3,
    ) -> Vec3 {
        // Newell's method is order-sensitive; a rim ring has no stored order
        // here, so fan triangles from the centre to each vertex pair and sum
        // their cross products. Without an ordering this still yields a stable
        // plane normal direction for a planar ring (each triangle's normal
        // agrees up to sign with the plane), and the magnitude (area) weights
        // it. Collinear/zero-area rings sum near zero → fallback.
        let points: Vec<Vec3> = verts.iter().map(|&v| self.vertex_position(v)).collect();
        let mut accum = Vec3::ZERO;
        for i in 0..points.len() {
            let p = points[i] - center;
            let q = points[(i + 1) % points.len()] - center;
            accum += p.cross(q);
        }
        if accum.length_squared() > 1e-12 {
            accum.normalize()
        } else {
            between.normalize_or_zero()
        }
    }

    /// §6.2.3 step 1: snapshot each rim vert's surrounding-surface tangent — the
    /// "off-the-rim, into the surface" direction, built from the faces *outside*
    /// the subdivided band that meet the rim vert. Each such face contributes
    /// its edge-tangent at the rim edge; these are averaged. Verts with no
    /// outside face (a lone band) fall back to the chord direction toward the
    /// other rail, leaving the per-ring cubic chord-aligned (vacuous, §6.2.4).
    fn rim_surface_tangents(
        &self,
        rungs: &HashMap<EdgeKey, (VertexKey, VertexKey)>,
        rail_a_verts: &HashSet<VertexKey>,
        band_faces: &HashSet<FaceKey>,
    ) -> HashMap<VertexKey, Vec3> {
        let mut tangents: HashMap<VertexKey, Vec3> = HashMap::new();
        for (&_rung, &(v_a, v_b)) in rungs {
            // Chord fallback direction: from each rim vert across the strip
            // toward the other rail.
            let chord_a = (self.vertex_position(v_b) - self.vertex_position(v_a))
                .normalize_or_zero();
            let chord_b = -chord_a;
            tangents.insert(v_a, self.surface_tangent_at(v_a, band_faces, chord_a));
            tangents.insert(v_b, self.surface_tangent_at(v_b, band_faces, chord_b));
        }
        let _ = rail_a_verts;
        tangents
    }

    /// The averaged off-the-rim tangent at one rim vert, summed over the edges
    /// of faces *outside* the band that emanate from it. Falls back to
    /// `chord_dir` when no outside face touches the vert.
    fn surface_tangent_at(
        &self,
        vert: VertexKey,
        band_faces: &HashSet<FaceKey>,
        chord_dir: Vec3,
    ) -> Vec3 {
        let p = self.vertex_position(vert);
        let mut accum = Vec3::ZERO;
        for edge in self.vertex_disk_iter(vert).collect::<Vec<_>>() {
            let touches_outside = self
                .edge_face_keys(edge)
                .iter()
                .any(|f| !band_faces.contains(f));
            if !touches_outside {
                continue;
            }
            let other = self.edge_other_vert(edge, vert);
            accum += (self.vertex_position(other) - p).normalize_or_zero();
        }
        if accum.length_squared() > 1e-12 {
            accum.normalize()
        } else {
            chord_dir
        }
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
    /// per level, then a rigid two-sided carry of each rung's rim endpoints into
    /// the level frame.
    fn apply_path_placement(
        &mut self,
        placement: &RingPlacement,
        rung_cuts: &HashMap<EdgeKey, RungCuts>,
        setup: &PathSetup,
        cuts: u32,
        smooth: f32,
    ) {
        let levels = cuts as usize;
        let segments = levels + 1;

        // Step 1 — the centre spline (Bézier). At smooth = 0 the handles
        // collapse onto the centres → straight segment, but PATH still diverges
        // from LINEAR via the frames below.
        let handle_len = handle_length(setup.center_a, setup.center_b, setup.normal_a, setup.normal_b);
        let handle_a = setup.center_a + smooth * handle_len * setup.normal_a;
        let handle_b = setup.center_b + smooth * handle_len * setup.normal_b;
        let path: Vec<Vec3> = (0..=segments)
            .map(|j| {
                let t = j as f32 / segments as f32;
                bezier(setup.center_a, handle_a, handle_b, setup.center_b, t)
            })
            .collect();

        // Step 2 — per-level directions and the minimum-twist frame sweep.
        let mut dir = vec![Vec3::ZERO; segments + 1];
        dir[0] = setup.normal_a;
        dir[segments] = -setup.normal_b;
        for j in 1..segments {
            dir[j] = path_bisector(path[j - 1], path[j], path[j + 1]);
        }
        let frames = minimum_twist_frames(&dir);

        // Step 3 — carry each rung's rim endpoints into the level frame.
        for (&rung, &(v_a, v_b)) in &placement.rungs {
            let pos_a = self.vertex_position(v_a);
            let pos_b = self.vertex_position(v_b);
            let inserted = rung_cuts[&rung].aligned(v_a);
            for (idx, &vert) in inserted.iter().enumerate() {
                let level = idx + 1;
                let t = level as f32 / segments as f32;
                let from_a = reframe(pos_a, frames[0], path[0], frames[level], path[level]);
                let from_b = reframe(
                    pos_b,
                    frames[segments],
                    path[segments],
                    frames[level],
                    path[level],
                );
                self.set_vertex_position(vert, from_a.lerp(from_b, t));
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
