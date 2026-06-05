use std::collections::HashMap;
use std::collections::HashSet;

use crate::loopmesh::*;

/// Placement of the new ring verts across the strip
/// ([`SUBDIVIDE_EDGERING.md`](../../../../radial-mesh-guides/operators/SUBDIVIDE_EDGERING.md)
/// §5 / §6.2). Only [`RingInterp::Linear`] — the straight chord midpoint — is
/// part of the foundational core. `Path` and `Surface` reposition the inserted
/// verts without changing the §3–§4 topology and are tracked as a separate
/// parity row (§6.2).
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
/// The deferred knobs (`cuts >= 2`, `interp_mode` PATH/SURFACE, `smooth` and the
/// profile parameters) are surveyed in §6 and tracked as separate parity rows;
/// only the core configuration is honoured today.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SubdivideEdgeRingParams {
    /// Number of new parallel rings inserted between the strip's rim rings
    /// (§1 / §6.1). The core supports `1` (a single midpoint ring).
    pub cuts: u32,
    /// Placement of the new ring verts (§5 / §6.2). The core uses
    /// [`RingInterp::Linear`].
    pub interp_mode: RingInterp,
    /// Bulge factor off the flat chord (§6.3). No-op in the core (`0.0`).
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
    /// The placement-tolerant parameters (§6.2 `interp_mode` PATH/SURFACE, §6.3
    /// `smooth` and the profile) are accepted on the params struct but only
    /// their no-op core values are implemented; any other value is a parity gap
    /// tracked as a separate row, not handled here. `cuts` is honoured for all
    /// `N >= 1`.
    pub fn subdivide_edgering(
        &mut self,
        rungs: &[EdgeKey],
        params: SubdivideEdgeRingParams,
    ) -> SubdivideEdgeRingResult {
        debug_assert_eq!(params.interp_mode, RingInterp::Linear);
        debug_assert_eq!(params.smooth, 0.0);

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

        // Phase one (§3 / §6.1.1): multi-cut each rung into `N + 1` equal
        // segments, recording the `N` midpoints in `near → far` order. The
        // directional run is the load-bearing detail (§6.1.1) — phase two
        // aligns level `j` across adjacent rungs by reading each rung from the
        // endpoint on the shared rail. `edge_split_n` runs the §3.3 directional
        // multi-cut, applying the two-source-linear vert/loop/edge blend at
        // each `t_j = j/(N+1)`.
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

        result
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
}
