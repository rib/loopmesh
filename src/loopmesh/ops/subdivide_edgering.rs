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
struct StripQuad {
    face: FaceKey,
    rung_a: EdgeKey,
    rung_b: EdgeKey,
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
    /// Element-count deltas (§4.1): an open strip of `k` quads gains `k + 1`
    /// verts, `2k + 1` edges and `k` faces; a closed band of `k` quads gains
    /// `k` verts, `2k` edges and `k` faces (the wrap-around connecting edge
    /// closes the new ring).
    ///
    /// The deferred parameters (§6) are accepted on the params struct but only
    /// their no-op core values are implemented; any other value is a parity gap
    /// tracked as a separate row, not handled here.
    pub fn subdivide_edgering(
        &mut self,
        rungs: &[EdgeKey],
        params: SubdivideEdgeRingParams,
    ) -> SubdivideEdgeRingResult {
        debug_assert_eq!(params.cuts, 1, "core supports cuts = 1 only");
        debug_assert_eq!(params.interp_mode, RingInterp::Linear);
        debug_assert_eq!(params.smooth, 0.0);

        let mut result = SubdivideEdgeRingResult::default();
        if rungs.is_empty() {
            return result;
        }

        let rung_set: HashSet<EdgeKey> = rungs.iter().copied().collect();

        // Identify the strip quads before phase one. A strip quad is a quad
        // face flanked by two *opposite* rungs of the set (§2 / §4); the fill
        // fires only on those. Capturing before the splits keeps the face keys
        // valid: split-edge-make-vert only inserts rim corners (§3), it never
        // retires the face.
        let strip_quads = self.collect_strip_quads(&rung_set);

        // Phase one (§3): one chord-midpoint cut per rung. `edge_split` at
        // t = 0.5 applies the §3 two-source-linear vert/loop/edge blend.
        let mut rung_midpoint: HashMap<EdgeKey, VertexKey> = HashMap::with_capacity(rungs.len());
        for &rung in &rung_set {
            let [near, _far] = self.edge_vert_keys(rung);
            let (midpoint, _new_edge) = self.edge_split(rung, near, 0.5);
            rung_midpoint.insert(rung, midpoint);
            result.verts.push(midpoint);
        }

        // Phase two (§4): one straight-cross fill per strip quad, connecting the
        // two midpoints. face_split copies the source face's per-face customdata
        // onto the new fragment (§4) and interpolates the new diagonal's loops.
        for strip in &strip_quads {
            let m_a = rung_midpoint[&strip.rung_a];
            let m_b = rung_midpoint[&strip.rung_b];
            if let Some((new_face, _new_edge)) = self.face_split(strip.face, m_a, m_b) {
                result.faces.push(strip.face);
                result.faces.push(new_face);
            }
        }

        result
    }

    /// Gather the strip quads of the resolved ring (§2 / §4): the quad faces
    /// flanked by two *opposite* rungs of `rung_set`. Each such face is
    /// captured once, with the two rungs that flank it.
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
            // A quad's edges sit at loop positions 0..4; positions 0 and 2 are
            // one opposite pair, 1 and 3 the other. A strip quad is flanked by
            // exactly one opposite pair of rungs.
            let edges = self.face_edge_keys(face);
            debug_assert_eq!(edges.len(), 4, "quad has four edges");
            if rung_set.contains(&edges[0]) && rung_set.contains(&edges[2]) {
                strip_quads.push(StripQuad {
                    face,
                    rung_a: edges[0],
                    rung_b: edges[2],
                });
            } else if rung_set.contains(&edges[1]) && rung_set.contains(&edges[3]) {
                strip_quads.push(StripQuad {
                    face,
                    rung_a: edges[1],
                    rung_b: edges[3],
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
}
