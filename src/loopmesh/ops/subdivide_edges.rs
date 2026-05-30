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

/// Parameters for [`LoopMesh::subdivide_edges`].
///
/// Mirrors the BMOP set-input shape described in
/// [`SUBDIVIDE_EDGES.md`](../../../../radial-mesh-guides/operators/SUBDIVIDE_EDGES.md)
/// §1. The default value is the standard-subdivide core configuration: a single
/// midpoint cut per edge (`cuts = 1`), no smoothing or fractal displacement, the
/// per-face re-split patterns enabled (`use_grid_fill` + `use_single_edge` on),
/// and `STRAIGHT_CUT` corners — exactly what this operator implements today.
///
/// The deferred knobs (smooth/fractal/sphere, `cuts >= 2`, the non-default corner
/// strategies, `use_only_quads`) are not yet part of the surface; supplying a
/// value outside the core makes the operator decline (§9, see
/// [`SubdivideResult::Unsupported`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SubdivideParams {
    /// Number of midpoint verts to insert per edge (§1). Only `1` is supported.
    pub cuts: u32,
    /// Enables the one-cut re-split patterns (§5.1 / §6.1). On for the core.
    pub use_single_edge: bool,
    /// Enables the all-edges re-split patterns (§5.4 / §6.3). On for the core.
    pub use_grid_fill: bool,
    /// Corner strategy for the quad-two-adjacent case (§5.2). Core uses
    /// [`QuadCornerType::StraightCut`].
    pub quad_corner_type: QuadCornerType,
}

impl Default for SubdivideParams {
    fn default() -> Self {
        Self {
            cuts: 1,
            use_single_edge: true,
            use_grid_fill: true,
            quad_corner_type: QuadCornerType::StraightCut,
        }
    }
}

/// Outcome of [`LoopMesh::subdivide_edges`]. Distinguishes a real run from a
/// declined-because-unsupported configuration, so a caller can detect a
/// deferred-parameter request without inspecting the mesh.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SubdivideResult {
    /// The operator ran. Carries the new geometry the cut created.
    Ran(SubdivideGeom),
    /// The parameter tuple is outside the implemented core (§9 deferred
    /// parameter, e.g. `cuts != 1`). The mesh was left untouched.
    Unsupported,
}

/// The geometry [`LoopMesh::subdivide_edges`] created, gathered as sets
/// ([`SUBDIVIDE_EDGES.md`](../../../../radial-mesh-guides/operators/SUBDIVIDE_EDGES.md)
/// §1). Enumeration order is not observable for parity; treat these as sets.
#[derive(Default, Clone, Debug, PartialEq, Eq)]
pub struct SubdivideGeom {
    /// Every midpoint vert inserted on a cut edge (§3), plus any inner vert a
    /// per-face pattern added (the §5.4 quad centre).
    pub verts: Vec<VertexKey>,
    /// Every face produced by a re-split (§5 / §6). The original faces that were
    /// re-split keep their handle and also appear here.
    pub faces: Vec<FaceKey>,
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
        if params.cuts != 1 {
            return SubdivideResult::Unsupported;
        }

        let mut geom = SubdivideGeom::default();

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
            let (m, _e_new) = self.edge_split(edge, near, 0.5);
            self.vert_op_flag_set(midpoint_layer, m, MIDPOINT);
            geom.verts.push(m);
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
                    self.quad_four_cut(face, &ring, geom);
                }
            }
            (4, 2) if Self::quad_two_cut_is_opposite(&ring) => {
                // §5.2 opposite case — the straight loop cut. Always fires
                // (no strategy choice, no gating flag).
                self.quad_two_opposite_cut(face, &ring, geom);
            }
            (4, 2) => {
                // §5.2 adjacent case. The default STRAIGHT_CUT strategy lays a
                // single corner cut; the non-default strategies (PATH / FAN /
                // INNERVERT) remain deferred (§9.3) and leave the face flat.
                if params.quad_corner_type == QuadCornerType::StraightCut {
                    self.quad_two_adjacent_straight_cut(face, &ring, geom);
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
        if let Some((new_face, _)) = self.face_split(face, m, apex) {
            geom.faces.push(face);
            geom.faces.push(new_face);
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
        let start = if (mids[1] - mids[0]) == 2 { mids[0] } else { mids[1] };
        let m01 = ring[start].0;
        let m12 = ring[(start + 2) % n].0;
        debug_assert!(!ring[(start + 1) % n].1, "shared corner is original, not a midpoint");

        if let Some((new_face, _)) = self.face_split(face, m01, m12) {
            geom.faces.push(face);
            geom.faces.push(new_face);
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
        let (corner_a, _) = self.face_split(f1, m01, m12).expect("m01-m12 split");
        // m12 -> m20 peels {m12, v2, m20} from the remaining pentagon.
        let f2 = self.face_with_both(m12, m20);
        let (corner_b, _) = self.face_split(f2, m12, m20).expect("m12-m20 split");
        // m20 -> m01 closes the centre triangle, peeling {m20, v0, m01}.
        let f3 = self.face_with_both(m20, m01);
        let (corner_c, _) = self.face_split(f3, m20, m01).expect("m20-m01 split");

        geom.faces.push(face);
        geom.faces.push(corner_a);
        geom.faces.push(corner_b);
        geom.faces.push(corner_c);
    }

    /// §5.1 — quad, one edge cut. Cycle is `v0, m, v1, v2, v3`; connect `m` into
    /// the interior to the two opposite corners `v3` and `v2`. 1 quad -> 3
    /// triangular fragments.
    fn quad_one_cut(&mut self, face: FaceKey, ring: &[(VertexKey, bool)], geom: &mut SubdivideGeom) {
        let (v, _) = Self::rotate_to_first_midpoint(ring);
        // v = [m, v1, v2, v3, v0]. The cut edge was {v0, v1}; the two interior
        // targets are v3 = v[3] and v2 = v[2].
        let (m, v2, v3) = (v[0], v[2], v[3]);
        // m -> v3 peels {v0, m, v3}.
        let (frag_a, _) = self.face_split(face, m, v3).expect("m-v3 split");
        // m -> v2 splits the remaining quad {m, v1, v2, v3}.
        let f2 = self.face_with_both(m, v2);
        let (frag_b, _) = self.face_split(f2, m, v2).expect("m-v2 split");

        geom.faces.push(face);
        geom.faces.push(frag_a);
        geom.faces.push(frag_b);
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
        if let Some((new_face, _)) = self.face_split(face, m01, m23) {
            geom.faces.push(face);
            geom.faces.push(new_face);
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
        let start = if (mids[1] - mids[0]) == 2 { mids[0] } else { mids[1] };

        let m01 = ring[start].0;
        let m12 = ring[(start + 2) % n].0;
        debug_assert!(ring[(start + 1) % n].0 != m01, "shared corner between midpoints");

        // m01 -> m12 carves the corner triangle off the hexagon.
        if let Some((new_face, _)) = self.face_split(face, m01, m12) {
            geom.faces.push(face);
            geom.faces.push(new_face);
        }
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
        let (corner_a, _) = self.face_split(f1, m12, m01).expect("m12-m01 split");
        // m12 -> m23 peels the corner triangle {m12, v2, m23}.
        let f2 = self.face_with_both(m12, m23);
        let (corner_b, _) = self.face_split(f2, m12, m23).expect("m12-m23 split");
        // m01 -> m23 splits the remaining pentagon {v0, m01, m12, m23, v3} into
        // the centre triangle {m01, m12, m23} and the quad {v0, m01, m23, v3}.
        let f3 = self.face_with_both(m01, m23);
        let (centre, _) = self.face_split(f3, m01, m23).expect("m01-m23 split");

        geom.faces.push(face);
        geom.faces.push(corner_a);
        geom.faces.push(corner_b);
        geom.faces.push(centre);
    }

    /// §5.4 — quad, all four edges cut. Cycle is
    /// `v0, m01, v1, m12, v2, m23, v3, m30`; insert one inner vert `c` at the
    /// centroid and connect each rim midpoint to it (the "plus" split).
    /// 1 quad -> 2x2 grid (4 quads).
    fn quad_four_cut(
        &mut self,
        face: FaceKey,
        ring: &[(VertexKey, bool)],
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

        // Connect the remaining two rim midpoints to c, finishing the four
        // spokes (m01-c and m23-c already exist as the halved diagonal).
        let f_m12 = self.face_with_both(c, m12);
        let (q_a, _) = self.face_split(f_m12, c, m12).expect("c-m12 split");
        let f_m30 = self.face_with_both(c, m30);
        let (q_b, _) = self.face_split(f_m30, c, m30).expect("c-m30 split");

        geom.verts.push(c);
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
        (mesh.totvert(), mesh.totedge(), mesh.totface())
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
            .map(|i| {
                mesh.query_edge_exists(fv[i], fv[(i + 1) % 4]).unwrap()
            })
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
            mesh.vertex_position(centre).distance(Vec3::new(1.0, 1.0, 0.0)) < 1e-5,
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

        mesh.face_create(&[v0, v1, v2, v3], &[e01, e12, e23, e30], FaceCreateFlags::NONE);
        mesh.face_create(&[v1, v4, v5, v2], &[e14, e45, e52, e12], FaceCreateFlags::NONE);

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
        assert!(tri_verts.contains(&v1), "triangle holds the shared corner v1");
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
        assert!(mesh.query_edge_exists(m12, m01).is_some(), "m12-m01 corner cut");
        assert!(mesh.query_edge_exists(m12, m23).is_some(), "m12-m23 corner cut");
        assert!(mesh.query_edge_exists(m01, m23).is_some(), "m01-m23 pentagon diagonal");

        // The lone quad fragment is {v0, m01, m23, v3} and holds the two corners
        // of the uncut edge {v3,v0}.
        let quad = geom
            .faces
            .iter()
            .copied()
            .find(|&f| mesh.face_len(f) == 4)
            .unwrap();
        let qv = mesh.face_vert_keys(quad);
        assert!(qv.contains(&v0) && qv.contains(&v3), "quad spans the uncut edge");
        assert!(qv.contains(&m01) && qv.contains(&m23));
        assert!(!qv.contains(&v1) && !qv.contains(&v2), "quad excludes the cut corners");
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
        assert!(mesh.query_edge_exists(m01, m12).is_some(), "m01-m12 corner cut");

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
    fn unsupported_multicut_declines() {
        let mut mesh = LoopMesh::new();
        let face = unit_quad(&mut mesh);
        let fv = mesh.face_vert_keys(face);
        let e01 = mesh.query_edge_exists(fv[0], fv[1]).unwrap();
        let before = counts(&mesh);
        let res = mesh.subdivide_edges(
            &[e01],
            SubdivideParams {
                cuts: 2,
                ..Default::default()
            },
        );
        assert_eq!(res, SubdivideResult::Unsupported);
        assert_eq!(counts(&mesh), before, "declined run leaves mesh untouched");
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
                Format::new(ComponentType::F32, 1),
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
        assert!((got - 1.0).abs() < 1e-5, "midpoint u expected 1.0, got {got}");
    }
}
