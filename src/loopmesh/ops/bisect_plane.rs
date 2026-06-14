use glam::Vec3;

use crate::loopmesh::ops::GeomKey;
use crate::loopmesh::*;

/// Which side of the plane a vert lies on, by the sign of its signed
/// plane-distance against the `dist` tolerance band
/// ([`BISECT_PLANE.md`](../../../../radial-mesh-guides/operators/BISECT_PLANE.md)
/// §3).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum Side {
    Negative,
    OnPlane,
    Positive,
}

/// Parameters for [`LoopMesh::bisect_plane`]
/// ([`BISECT_PLANE.md`](../../../../radial-mesh-guides/operators/BISECT_PLANE.md)
/// §1).
///
/// The plane is fixed by a point (`plane_co`) it passes through and a normal
/// (`plane_no`); a point `p` is on the positive side when
/// `dot(p - plane_co, plane_no) > 0`. The normal need not be unit length: every
/// geometric step (the side test §3, the crossing point §4, the snap §3.1) is
/// scale-invariant in its magnitude. The only consequence of a non-unit normal
/// is that `dist` is measured in the units that normal implies — pass a unit
/// normal for a world-unit `dist`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BisectParams {
    /// A point the cutting plane passes through (§1).
    pub plane_co: Vec3,
    /// The plane normal; the positive side is the half it points into (§1).
    pub plane_no: Vec3,
    /// On-plane tolerance: a vert whose signed plane-distance has magnitude
    /// `<= dist` is classified on-plane rather than to either side (§3). `0`
    /// treats only verts exactly on the plane as on-plane.
    pub dist: f32,
    /// When set, every on-plane vert is projected exactly onto the plane so its
    /// signed distance becomes 0 (§3.1). Purely positional; never changes the
    /// classification or topology.
    pub use_snap_center: bool,
    /// Remove everything strictly on the negative side after the cut (§4.1).
    pub clear_inner: bool,
    /// Remove everything strictly on the positive side after the cut (§4.1).
    pub clear_outer: bool,
}

/// Outcome of [`LoopMesh::bisect_plane`]
/// ([`BISECT_PLANE.md`](../../../../radial-mesh-guides/operators/BISECT_PLANE.md)
/// §1).
///
/// `geom_out` is the geometry that survived and/or was created by the cut (the
/// `geom.out` slot); `seam_verts` / `seam_edges` are the on-plane elements (the
/// `geom_cut.out` slot, §5), reported as a first-class output rather than left
/// for the caller to re-derive. The seam is a subset of `geom_out`: on-plane
/// elements survive any clear.
#[derive(Debug, Default)]
pub struct BisectResult {
    /// The survivors plus the inscribed cut — the `geom.out` slot (§1).
    pub geom_out: Vec<GeomKey>,
    /// The verts lying on the plane after the cut: new crossing verts plus
    /// pre-existing on-plane verts (snapped, if `use_snap_center`). The seam
    /// part of `geom_cut.out` (§5).
    pub seam_verts: Vec<VertexKey>,
    /// The edges both of whose endpoints are seam verts: split-face seam edges
    /// plus any pre-existing edge lying in the plane. The other part of
    /// `geom_cut.out` (§5).
    pub seam_edges: Vec<EdgeKey>,
}

const ON: u8 = 0;
const POS: u8 = 1;

impl LoopMesh {
    /// Slice the tagged geometry by an infinite plane, optionally clearing one
    /// or both sides, and report the cut seam.
    ///
    /// Follows
    /// [`BISECT_PLANE.md`](../../../../radial-mesh-guides/operators/BISECT_PLANE.md)
    /// §6, composing the existing kernel splits:
    ///
    /// 1. Classify every tagged vert by signed plane distance (§3).
    /// 2. Snap on-plane verts onto the plane when `use_snap_center` (§3.1).
    /// 3. Cut every straddling tagged edge at its crossing point `t =
    ///    da/(da-db)` via [`Self::edge_split`] (§4); the new vert is on-plane.
    /// 4. Split every straddling tagged face along its two on-plane corners via
    ///    [`Self::face_split`] (§4), so no face crosses the plane afterward.
    /// 5. Clear the selected side(s): the cleared side's non-seam verts are
    ///    deleted, cascading up to their incident edges and faces; on-plane
    ///    (seam) elements survive any clear (§4.1).
    /// 6. Report the survivors (`geom.out`) and the on-plane set
    ///    (`geom_cut.out`, §5).
    ///
    /// `geom` is the mixed verts/edges/faces set to cut; only tagged edges are
    /// cut and only tagged faces split, matching the BMOP `geom` slot (§1). In
    /// normal use this is the whole mesh.
    pub fn bisect_plane(&mut self, geom: &[GeomKey], params: BisectParams) -> BisectResult {
        let normal = params.plane_no;
        let plane_co = params.plane_co;

        // Tagged element sets (§1). Verts reachable from any tagged element are
        // classified; only tagged edges cut, only tagged faces split.
        let tagged_verts = self.collect_tagged_verts(geom);
        let tagged_edges = self.collect_tagged_edges(geom);
        let tagged_faces = self.collect_tagged_faces(geom);

        // `side` carries the per-vert classification across the whole op,
        // including the new cut verts (eager layer growth keeps it valid as
        // edge_split inserts verts). ON = on-plane, POS = positive; neither bit
        // = negative.
        let side = self.push_vert_op_flag_layer();

        // Step 1 (§3) — classify each tagged vert.
        for &vert in &tagged_verts {
            let s = classify(self.vertex_position(vert), plane_co, normal, params.dist);
            self.mark_side(side, vert, s);
        }

        // Step 2 (§3.1) — snap on-plane verts exactly onto the plane.
        if params.use_snap_center {
            for &vert in &tagged_verts {
                if self.vert_op_flag_test(side, vert, ON) {
                    let snapped = snap_onto_plane(self.vertex_position(vert), plane_co, normal);
                    self.set_vertex_position(vert, snapped);
                }
            }
        }

        // Step 3 (§4) — cut every straddling tagged edge at its crossing point.
        // The split inserts a vert classified on-plane that joins the seam.
        for &edge in &tagged_edges {
            if self.edges.try_get(edge).is_none() {
                continue;
            }
            let [a, b] = self.edge_vert_keys(edge);
            if !self.straddles(side, a, b) {
                continue;
            }
            let pa = self.vertex_position(a);
            let pb = self.vertex_position(b);
            let t = crossing_t(pa, pb, plane_co, normal);
            let (cut_vert, _new_edge) = self.edge_split(edge, a, t);
            self.mark_side(side, cut_vert, Side::OnPlane);
        }

        // Step 4 (§4) — split every straddling tagged face along its seam verts.
        // Re-resolve the live face set: the input faces may have grown new
        // corners from step 3's edge splits, but no face was killed yet.
        for &face in &tagged_faces {
            if self.faces.try_get(face).is_none() {
                continue;
            }
            self.split_straddling_face(face, side);
        }

        // Step 5 (§4.1) — clear the selected side(s). The clear does NOT re-read
        // the §3 `Side`; it runs a fresh, strict, `dist`-shifted signed-distance
        // test over the ORIGINAL INPUT verts only (`tagged_verts`, snapshotted
        // before steps 3-4 created any crossing/duplicate verts). An input vert at
        // signed distance within `dist` (including exactly on the plane at
        // `dist = 0`) survives, and verts made during the cut are never deletion
        // candidates — which is why an on-plane column at `dist = 0` survives both
        // clears and the degenerate seam face is kept.
        if params.clear_inner || params.clear_outer {
            self.clear_input_verts(
                &tagged_verts,
                plane_co,
                normal,
                params.dist,
                params.clear_inner,
                params.clear_outer,
            );
        }

        // Step 6 (§5) — collect outputs from the post-cut classification.
        let result = self.collect_outputs(side);
        self.pop_vert_op_flag_layer(side);
        result
    }

    fn collect_tagged_verts(&self, geom: &[GeomKey]) -> Vec<VertexKey> {
        let mut seen: Vec<VertexKey> = Vec::new();
        let push = |v: VertexKey, seen: &mut Vec<VertexKey>| {
            if !seen.contains(&v) {
                seen.push(v);
            }
        };
        for &g in geom {
            match g {
                GeomKey::Vert(v) => push(v, &mut seen),
                GeomKey::Edge(e) => {
                    let [a, b] = self.edge_vert_keys(e);
                    push(a, &mut seen);
                    push(b, &mut seen);
                }
                GeomKey::Face(f) => {
                    for v in self.face_vert_keys(f) {
                        push(v, &mut seen);
                    }
                }
            }
        }
        seen
    }

    fn collect_tagged_edges(&self, geom: &[GeomKey]) -> Vec<EdgeKey> {
        let mut seen: Vec<EdgeKey> = Vec::new();
        let push = |e: EdgeKey, seen: &mut Vec<EdgeKey>| {
            if !seen.contains(&e) {
                seen.push(e);
            }
        };
        for &g in geom {
            match g {
                GeomKey::Edge(e) => push(e, &mut seen),
                GeomKey::Face(f) => {
                    for e in self.face_edge_keys(f) {
                        push(e, &mut seen);
                    }
                }
                GeomKey::Vert(_) => {}
            }
        }
        seen
    }

    fn collect_tagged_faces(&self, geom: &[GeomKey]) -> Vec<FaceKey> {
        let mut seen: Vec<FaceKey> = Vec::new();
        for &g in geom {
            if let GeomKey::Face(f) = g {
                if !seen.contains(&f) {
                    seen.push(f);
                }
            }
        }
        seen
    }

    fn mark_side(&mut self, layer: crate::op_flags::VertOpFlagLayer, vert: VertexKey, side: Side) {
        // Clear first so a re-classified vert lands cleanly.
        self.vert_op_flag_clear(layer, vert, ON);
        self.vert_op_flag_clear(layer, vert, POS);
        match side {
            Side::OnPlane => self.vert_op_flag_set(layer, vert, ON),
            Side::Positive => self.vert_op_flag_set(layer, vert, POS),
            Side::Negative => {}
        }
    }

    fn side_of(&self, layer: crate::op_flags::VertOpFlagLayer, vert: VertexKey) -> Side {
        if self.vert_op_flag_test(layer, vert, ON) {
            Side::OnPlane
        } else if self.vert_op_flag_test(layer, vert, POS) {
            Side::Positive
        } else {
            Side::Negative
        }
    }

    /// An edge straddles when its endpoints sit on strictly opposite sides
    /// (one Negative, one Positive). An on-plane endpoint does not straddle —
    /// it already meets the seam (§4).
    fn straddles(
        &self,
        layer: crate::op_flags::VertOpFlagLayer,
        a: VertexKey,
        b: VertexKey,
    ) -> bool {
        matches!(
            (self.side_of(layer, a), self.side_of(layer, b)),
            (Side::Negative, Side::Positive) | (Side::Positive, Side::Negative)
        )
    }

    /// §4 — split a tagged face that straddles the plane along the chord joining
    /// its two on-plane boundary corners. A face with corners on both sides has
    /// exactly two on-plane corners after step 3's edge cuts (for a convex cut);
    /// connecting them carves it into one kept sub-face per side. A face touched
    /// only along a single on-plane corner or wholly on one side does not
    /// straddle and is left whole.
    fn split_straddling_face(&mut self, face: FaceKey, side: crate::op_flags::VertOpFlagLayer) {
        let corners = self.face_vert_keys(face);
        let mut has_pos = false;
        let mut has_neg = false;
        let mut on_plane: Vec<VertexKey> = Vec::new();
        for &v in &corners {
            match self.side_of(side, v) {
                Side::Positive => has_pos = true,
                Side::Negative => has_neg = true,
                Side::OnPlane => on_plane.push(v),
            }
        }
        // Straddles iff corners land on both sides (§4). Otherwise the plane only
        // grazes the boundary — no split.
        if !(has_pos && has_neg) {
            return;
        }
        debug_assert!(
            on_plane.len() >= 2,
            "a straddling face has two on-plane corners after the edge cuts"
        );
        // For each consecutive pair of on-plane corners around the boundary that
        // brackets a same-side arc, draw one seam chord (§4 / §7 multi-crossing).
        // The common convex case has exactly two on-plane corners and one chord.
        let pairs = self.seam_chord_pairs(&corners, &on_plane, side);
        for (a, b) in pairs {
            // Re-resolve the live sub-face still holding both endpoints; an
            // earlier chord may have split this face.
            if let Some(target) = self.live_face_with_pair(face, a, b) {
                self.face_split(target, a, b);
            }
        }
    }

    /// Pair up a straddling face's on-plane corners into seam chords: each chord
    /// brackets a contiguous boundary arc lying strictly on one side, so the
    /// split lands a whole sub-face on each side (§4). For a convex cut there are
    /// two on-plane corners and a single chord; a non-convex face crossing more
    /// than twice yields one chord per crossing pair (§7).
    fn seam_chord_pairs(
        &self,
        corners: &[VertexKey],
        on_plane: &[VertexKey],
        side: crate::op_flags::VertOpFlagLayer,
    ) -> Vec<(VertexKey, VertexKey)> {
        let n = corners.len();
        // Positions of on-plane corners in boundary order.
        let mut on_positions: Vec<usize> = (0..n)
            .filter(|&i| matches!(self.side_of(side, corners[i]), Side::OnPlane))
            .collect();
        debug_assert_eq!(on_positions.len(), on_plane.len());
        let mut pairs: Vec<(VertexKey, VertexKey)> = Vec::new();
        // Walk consecutive on-plane corners; a chord between two of them is a
        // genuine interior cut only when the boundary arc strictly between them
        // is non-empty and lies entirely on one side. Adjacent on-plane corners
        // (the arc between them empty) already share a perimeter edge in the
        // plane — no chord.
        on_positions.sort_unstable();
        let m = on_positions.len();
        // Two on-plane corners (the convex cut) need a single chord: the two arcs
        // between them are the two sides, but they share the same chord. Emit it
        // once. With more than two crossings (§7) each forward arc that lies on a
        // single side gets its own chord.
        if m == 2 {
            return vec![(corners[on_positions[0]], corners[on_positions[1]])];
        }
        for k in 0..m {
            let pa = on_positions[k];
            let pb = on_positions[(k + 1) % m];
            if arc_is_single_side(corners, pa, pb, n, |v| self.side_of(side, v)) {
                pairs.push((corners[pa], corners[pb]));
            }
        }
        pairs
    }

    /// The live face still incident to both `a` and `b`. After a face_split the
    /// next chord's endpoints land on exactly one of the two sub-faces; this
    /// finds it. Falls back to the original handle when no split has happened.
    fn live_face_with_pair(
        &self,
        original: FaceKey,
        a: VertexKey,
        b: VertexKey,
    ) -> Option<FaceKey> {
        for f in self.vert_face_keys(a) {
            if self.face_vert_keys(f).contains(&b) {
                return Some(f);
            }
        }
        if self.faces.try_get(original).is_some() {
            let verts = self.face_vert_keys(original);
            if verts.contains(&a) && verts.contains(&b) {
                return Some(original);
            }
        }
        None
    }

    /// §4.1 — clear the selected side(s) by a fresh, strict, `dist`-shifted
    /// signed-distance test over the ORIGINAL INPUT verts only. This is NOT the
    /// §3 cut classification: `clear_outer` kills an input vert iff
    /// `signed_dist > dist` and `clear_inner` iff `signed_dist < -dist`, both
    /// strict. An input vert within `dist` of the plane (including exactly on it
    /// at `dist = 0`) survives both clears; verts created during the cut are not
    /// in `input_verts`, so they are never deletion candidates. Seeding the
    /// doomed verts into a vert-delete cascades up to their now-loose incident
    /// edges and faces, so the seam — and the degenerate seam face, whose
    /// surviving corners are never killed — survives any clear.
    fn clear_input_verts(
        &mut self,
        input_verts: &[VertexKey],
        plane_co: Vec3,
        plane_no: Vec3,
        dist: f32,
        clear_inner: bool,
        clear_outer: bool,
    ) {
        let mut doomed: Vec<VertexKey> = Vec::new();
        for &vert in input_verts {
            if self.vertices.try_get(vert).is_none() {
                continue;
            }
            let s = (self.vertex_position(vert) - plane_co).dot(plane_no);
            let kill = (clear_outer && s - dist > 0.0) || (clear_inner && s + dist < 0.0);
            if kill {
                doomed.push(vert);
            }
        }
        self.delete(&doomed, DeleteContext::Verts);
    }

    /// §5 — collect the survivors (`geom.out`) and the on-plane set
    /// (`geom_cut.out`). Read off the post-cut classification rather than a
    /// fresh distance test: the snap may have moved verts and the cut verts have
    /// no pre-cut identity.
    fn collect_outputs(&self, side: crate::op_flags::VertOpFlagLayer) -> BisectResult {
        let mut geom_out: Vec<GeomKey> = Vec::new();
        let mut seam_verts: Vec<VertexKey> = Vec::new();
        let mut seam_edges: Vec<EdgeKey> = Vec::new();

        let mut vit = MeshVerts::default();
        while let Some(vert) = vit.next(self) {
            geom_out.push(GeomKey::Vert(vert));
            if matches!(self.side_of(side, vert), Side::OnPlane) {
                seam_verts.push(vert);
            }
        }
        let mut eit = MeshEdges::default();
        while let Some(edge) = eit.next(self) {
            geom_out.push(GeomKey::Edge(edge));
            let [a, b] = self.edge_vert_keys(edge);
            if matches!(self.side_of(side, a), Side::OnPlane)
                && matches!(self.side_of(side, b), Side::OnPlane)
            {
                seam_edges.push(edge);
            }
        }
        let mut fit = MeshFaces::default();
        while let Some(face) = fit.next(self) {
            geom_out.push(GeomKey::Face(face));
        }

        BisectResult {
            geom_out,
            seam_verts,
            seam_edges,
        }
    }
}

/// §3 — signed plane-distance side classification against the `dist` band.
///
/// The boundary is asymmetric: the negative test is checked first and is
/// inclusive, so a vert exactly at `-dist` (and, when `dist == 0`, a vert
/// exactly on the plane) lands on the negative side, not on-plane. The on-plane
/// region is the open band `(-dist, +dist)`; at `dist == 0` that band is empty,
/// which is what drives the duplicate-column regime (§2.1).
fn classify(p: Vec3, plane_co: Vec3, plane_no: Vec3, dist: f32) -> Side {
    let signed_dist = (p - plane_co).dot(plane_no);
    if signed_dist <= -dist {
        Side::Negative
    } else if signed_dist >= dist {
        Side::Positive
    } else {
        Side::OnPlane
    }
}

/// §4 — crossing parameter where the signed distance is 0 along the segment
/// `a -> b`. Both endpoints are on opposite sides so `t` lies in `(0, 1)` and
/// the denominator is non-zero. Scale-invariant in `plane_no`'s magnitude.
fn crossing_t(a: Vec3, b: Vec3, plane_co: Vec3, plane_no: Vec3) -> f32 {
    let da = (a - plane_co).dot(plane_no);
    let db = (b - plane_co).dot(plane_no);
    da / (da - db)
}

/// §3.1 — project `p` onto the plane along the normal so its signed distance
/// becomes 0. Dividing by `dot(plane_no, plane_no)` removes exactly the normal
/// component for any magnitude (it reduces to `p - plane_no * signed_dist` for a
/// unit normal).
fn snap_onto_plane(p: Vec3, plane_co: Vec3, plane_no: Vec3) -> Vec3 {
    let signed_dist = (p - plane_co).dot(plane_no);
    p - plane_no * (signed_dist / plane_no.dot(plane_no))
}

/// True when the boundary arc strictly between corners `pa` and `pb` (forward
/// from `pa` to `pb`) is non-empty and every corner on it lies on a single side
/// (all Positive or all Negative). A chord between two on-plane corners that
/// bracket such an arc carves a whole sub-face off that side (§4).
fn arc_is_single_side(
    corners: &[VertexKey],
    pa: usize,
    pb: usize,
    n: usize,
    side_of: impl Fn(VertexKey) -> Side,
) -> bool {
    let mut saw_pos = false;
    let mut saw_neg = false;
    let mut count = 0;
    let mut i = (pa + 1) % n;
    while i != pb {
        match side_of(corners[i]) {
            Side::Positive => saw_pos = true,
            Side::Negative => saw_neg = true,
            // An on-plane corner inside the arc means this is not the tight
            // bracketing pair; the closer on-plane corner owns the chord.
            Side::OnPlane => return false,
        }
        count += 1;
        i = (i + 1) % n;
    }
    count > 0 && (saw_pos ^ saw_neg)
}

#[cfg(test)]
mod tests {
    use glam::Vec3;

    use super::*;
    use crate::loopmesh::*;

    const PLANE_NO_X: Vec3 = Vec3::X;

    fn edge_of(mesh: &mut LoopMesh, a: VertexKey, b: VertexKey) -> EdgeKey {
        mesh.query_edge_exists(a, b)
            .unwrap_or_else(|| mesh.create_edge([a, b], EdgeCreateFlags::NONE))
    }

    fn make_face(mesh: &mut LoopMesh, ring: &[VertexKey]) -> FaceKey {
        let edges: Vec<EdgeKey> = (0..ring.len())
            .map(|i| edge_of(mesh, ring[i], ring[(i + 1) % ring.len()]))
            .collect();
        mesh.face_create(ring, &edges, FaceCreateFlags::NONE)
    }

    /// A strip of `cols` quads along x, spanning rows y = 0 and y = 1. Column
    /// `c` runs x = `x_start + c`. Returns (mesh, faces, all verts row-major:
    /// bottom row then top row).
    fn strip(x_start: f32, cols: usize) -> (LoopMesh, Vec<FaceKey>, Vec<VertexKey>) {
        let mut mesh = LoopMesh::new();
        let n = cols + 1;
        let mut bottom = Vec::new();
        let mut top = Vec::new();
        for i in 0..n {
            let x = x_start + i as f32;
            bottom.push(mesh.create_vertex(Vec3::new(x, 0.0, 0.0)));
            top.push(mesh.create_vertex(Vec3::new(x, 1.0, 0.0)));
        }
        let mut faces = Vec::new();
        for c in 0..cols {
            faces.push(make_face(
                &mut mesh,
                &[bottom[c], bottom[c + 1], top[c + 1], top[c]],
            ));
        }
        let mut verts = bottom;
        verts.extend(top);
        (mesh, faces, verts)
    }

    fn whole_mesh_geom(mesh: &LoopMesh) -> Vec<GeomKey> {
        let mut geom: Vec<GeomKey> = Vec::new();
        geom.extend(mesh.vert_keys().into_iter().map(GeomKey::Vert));
        geom.extend(mesh.edge_keys().into_iter().map(GeomKey::Edge));
        geom.extend(mesh.face_keys().into_iter().map(GeomKey::Face));
        geom
    }

    fn params(dist: f32, snap: bool, inner: bool, outer: bool) -> BisectParams {
        BisectParams {
            plane_co: Vec3::ZERO,
            plane_no: PLANE_NO_X,
            dist,
            use_snap_center: snap,
            clear_inner: inner,
            clear_outer: outer,
        }
    }

    #[test]
    fn classify_and_crossing_t_are_closed_form() {
        // §3 classification at the band edges and §4 crossing parameter.
        assert_eq!(
            classify(Vec3::new(0.5, 0.0, 0.0), Vec3::ZERO, PLANE_NO_X, 0.2),
            Side::Positive
        );
        assert_eq!(
            classify(Vec3::new(-0.5, 0.0, 0.0), Vec3::ZERO, PLANE_NO_X, 0.2),
            Side::Negative
        );
        // The on-plane region is the OPEN band (-dist, +dist); the boundary is
        // asymmetric (§3). Exactly inside the band is on-plane:
        assert_eq!(
            classify(Vec3::new(0.1, 0.0, 0.0), Vec3::ZERO, PLANE_NO_X, 0.2),
            Side::OnPlane
        );
        // The upper boundary is inclusive — signed_dist == +dist is Positive:
        assert_eq!(
            classify(Vec3::new(0.2, 0.0, 0.0), Vec3::ZERO, PLANE_NO_X, 0.2),
            Side::Positive
        );
        // The negative boundary is inclusive and checked first — signed_dist ==
        // -dist is Negative:
        assert_eq!(
            classify(Vec3::new(-0.2, 0.0, 0.0), Vec3::ZERO, PLANE_NO_X, 0.2),
            Side::Negative
        );
        // At dist == 0 the open band is empty and the centre tie goes Negative —
        // a vert exactly ON the plane classifies Negative, not on-plane (§3,
        // the load-bearing tie-break behind the duplicate-column regime §2.1):
        assert_eq!(
            classify(Vec3::ZERO, Vec3::ZERO, PLANE_NO_X, 0.0),
            Side::Negative
        );
        // A strictly-positive distance still reaches Positive at dist == 0:
        assert_eq!(
            classify(Vec3::new(0.01, 0.0, 0.0), Vec3::ZERO, PLANE_NO_X, 0.0),
            Side::Positive
        );
        // Crossing at x = 0 between x = -1 and x = 3 is t = 1/4 (§4).
        let t = crossing_t(
            Vec3::new(-1.0, 0.0, 0.0),
            Vec3::new(3.0, 0.0, 0.0),
            Vec3::ZERO,
            PLANE_NO_X,
        );
        assert!((t - 0.25).abs() < 1e-6, "t = da/(da-db) = 1/4");
    }

    #[test]
    fn clear_outer_keeps_negative_half_and_reports_seam() {
        // §4 / §4.1 / §5: a strip straddling x = 0 cut with clear_outer keeps
        // every survivor at x <= 0 and reports a non-empty seam, all on x = 0.
        let (mut mesh, _faces, _verts) = strip(-2.0, 4); // columns at x = -2..2
        let geom = whole_mesh_geom(&mesh);
        let result = mesh.bisect_plane(&geom, params(0.0, true, false, true));

        assert!(!result.seam_verts.is_empty(), "non-empty seam (§5)");
        for v in mesh.vert_keys() {
            assert!(
                mesh.vertex_position(v).x <= 1e-5,
                "every survivor on the kept (x <= 0) side"
            );
        }
        // The seam is a subset of the survivors: every seam vert is live and
        // sits exactly on the plane.
        for &sv in &result.seam_verts {
            assert!(mesh.vertices.try_get(sv).is_some());
            assert!(mesh.vertex_position(sv).x.abs() < 1e-5, "seam on x = 0");
        }
        assert!(!result.seam_edges.is_empty(), "seam edges reported (§5)");
    }

    #[test]
    fn clear_inner_keeps_positive_half_differs_from_clear_outer() {
        // §4.1: clear_inner keeps x >= 0; the non-symmetric strip makes the
        // survivor differ from the clear_outer run (verts on opposite sides).
        let (mut mesh, _f, _v) = strip(-2.0, 4);
        let geom = whole_mesh_geom(&mesh);
        mesh.bisect_plane(&geom, params(0.0, true, true, false));
        for v in mesh.vert_keys() {
            assert!(
                mesh.vertex_position(v).x >= -1e-5,
                "every survivor on the kept (x >= 0) side"
            );
        }

        // Survivor vert counts differ between the two clears: the strip is not
        // symmetric about x = 0 (it spans x = -2..2 with the plane cutting
        // through the x = 0 column, but the half-counts of a 4-column strip are
        // equal here — so contrast on the bounding box instead).
        let (mut outer_mesh, _f2, _v2) = strip(-2.0, 4);
        let geom2 = whole_mesh_geom(&outer_mesh);
        outer_mesh.bisect_plane(&geom2, params(0.0, true, false, true));
        let inner_max = mesh
            .vert_keys()
            .iter()
            .map(|&v| mesh.vertex_position(v).x)
            .fold(f32::NEG_INFINITY, f32::max);
        let outer_min = outer_mesh
            .vert_keys()
            .iter()
            .map(|&v| outer_mesh.vertex_position(v).x)
            .fold(f32::INFINITY, f32::min);
        assert!(inner_max > 1.0, "clear_inner keeps the positive extent");
        assert!(outer_min < -1.0, "clear_outer keeps the negative extent");
    }

    #[test]
    fn dist_zero_duplicates_on_plane_column() {
        // §2.1 / §3 regime B / §8.2: a 4-quad strip whose middle column (x = 0)
        // lies exactly on the plane, cut at dist = 0. The on-plane verts classify
        // Negative (the centre tie-break, §3), the spanning edges to the positive
        // column straddle and cut at t = 0 (coincident copies, §4), the column is
        // DUPLICATED, and a degenerate zero-width seam face is inscribed.
        let cut = |inner: bool, outer: bool| {
            let (mut mesh, _f, _v) = strip(-2.0, 4); // columns at x = -2..2
            let geom = whole_mesh_geom(&mesh);
            mesh.bisect_plane(&geom, params(0.0, false, inner, outer));
            (mesh.vertex_count(), mesh.edge_count(), mesh.face_count())
        };

        // clear_outer keeps the negative half: two kept quads + the degenerate
        // seam quad {2, 2', 7', 7} (its two on-plane corners 2,7 sit at signed
        // distance 0, excluded by the §4.1 strict clear test; the duplicates are
        // post-cut, never deletion candidates) → 8 V / 10 E / 3 F (§2.1 / §8.2).
        assert_eq!(cut(false, true), (8, 10, 3), "clear_outer @ dist = 0");
        // clear_inner keeps the positive half: the SAME count, because the §4.1
        // clear keys off a fresh strict input-vert distance test (not the §3 cut
        // `Side`), so the degenerate seam face's corners survive this clear too →
        // also 8 V / 10 E / 3 F (§4.1 consequence 1 / §8.2). A "delete every
        // Negative vert" reading would give the wrong 6/7/2 here.
        assert_eq!(cut(true, false), (8, 10, 3), "clear_inner @ dist = 0");
        // No clear keeps the whole strip; the duplicated column splits it in two
        // plus the degenerate seam quad → 12 V / 16 E / 5 F.
        assert_eq!(cut(false, false), (12, 16, 5), "no-clear @ dist = 0");
    }

    #[test]
    fn dist_zero_clears_keep_opposite_survivors() {
        // §4.1: clear_outer and clear_inner at dist = 0 over the on-plane-column
        // strip yield the SAME element count (8/10/3) but keep survivors on
        // OPPOSITE sides — so the two results are not accidentally identical.
        let survivor_xs = |inner: bool, outer: bool| {
            let (mut mesh, _f, _v) = strip(-2.0, 4); // columns at x = -2..2
            let geom = whole_mesh_geom(&mesh);
            mesh.bisect_plane(&geom, params(0.0, false, inner, outer));
            let min = mesh
                .vert_keys()
                .iter()
                .map(|&v| mesh.vertex_position(v).x)
                .fold(f32::INFINITY, f32::min);
            let max = mesh
                .vert_keys()
                .iter()
                .map(|&v| mesh.vertex_position(v).x)
                .fold(f32::NEG_INFINITY, f32::max);
            (min, max)
        };

        let (outer_min, outer_max) = survivor_xs(false, true);
        let (inner_min, inner_max) = survivor_xs(true, false);
        // clear_outer keeps x <= 0: the negative extent reaches -2, none above 0.
        assert!(outer_min < -1.0, "clear_outer keeps the negative extent");
        assert!(
            outer_max <= 1e-5,
            "clear_outer keeps nothing strictly positive"
        );
        // clear_inner keeps x >= 0: the positive extent reaches +2, none below 0.
        assert!(inner_max > 1.0, "clear_inner keeps the positive extent");
        assert!(
            inner_min >= -1e-5,
            "clear_inner keeps nothing strictly negative"
        );
        // The two survivor x-extents differ — the clears select opposite sides.
        assert_ne!(
            (outer_min, outer_max),
            (inner_min, inner_max),
            "clears keep opposite-side survivors despite equal counts"
        );
    }

    #[test]
    fn dist_positive_keeps_on_plane_column_single() {
        // §2.1 regime A / §8.2: the same strip and plane, cut with a small
        // positive dist so the middle column falls inside the open on-plane band
        // (§3). Nothing straddles, the column stays single, no degenerate face.
        let cut = |inner: bool, outer: bool| {
            let (mut mesh, _f, _v) = strip(-2.0, 4);
            let geom = whole_mesh_geom(&mesh);
            mesh.bisect_plane(&geom, params(0.1, false, inner, outer));
            (mesh.vertex_count(), mesh.edge_count(), mesh.face_count())
        };

        // clear_outer keeps the negative half sharing the single on-plane
        // column → 6 V / 7 E / 2 F (§2.1 / §8.2), strictly fewer than dist = 0's
        // 8 V / 10 E / 3 F — the duplication is genuinely suppressed.
        assert_eq!(cut(false, true), (6, 7, 2), "clear_outer @ dist > 0");
        // clear_inner keeps the positive half, mirror count → also 6 V / 7 E /
        // 2 F. The §4.1 strict test kills the strictly-negative input verts; the
        // on-plane column (within dist) survives as the shared single column.
        assert_eq!(cut(true, false), (6, 7, 2), "clear_inner @ dist > 0");
        // No clear leaves the input unchanged (plus seam tagging): 10 V / 13 E /
        // 4 F — the four original quads, no split, no duplicate.
        assert_eq!(cut(false, false), (10, 13, 4), "no-clear @ dist > 0");
    }

    #[test]
    fn between_columns_cut_creates_seam_verts_and_splits_faces() {
        // §2 / §4: a plane through x = 0.5 falls between columns, so the middle
        // quad straddles. The cut inserts two crossing verts and splits the
        // face; clear_outer keeps the negative side.
        let (mut mesh, _faces, _verts) = strip(0.0, 2); // columns x = 0..1, 1..2
        let v_before = mesh.vertex_count();
        let geom = whole_mesh_geom(&mesh);
        let p = BisectParams {
            plane_co: Vec3::new(0.5, 0.0, 0.0),
            plane_no: PLANE_NO_X,
            dist: 0.0,
            use_snap_center: false,
            clear_inner: false,
            clear_outer: true,
        };
        let result = mesh.bisect_plane(&geom, p);

        // Two crossing verts were created on the two straddling edges (§4).
        assert!(mesh.vertex_count() < v_before, "positive side cleared");
        assert_eq!(result.seam_verts.len(), 2, "two crossing verts on x = 0.5");
        for &sv in &result.seam_verts {
            assert!(
                (mesh.vertex_position(sv).x - 0.5).abs() < 1e-5,
                "seam at x = 0.5"
            );
        }
        // The kept sub-face is a quad between x = 0 and x = 0.5.
        for v in mesh.vert_keys() {
            assert!(mesh.vertex_position(v).x <= 0.5 + 1e-5);
        }
        assert_eq!(
            result.seam_edges.len(),
            1,
            "one seam edge across the cut face"
        );
    }

    #[test]
    fn snap_center_moves_near_plane_vert_onto_plane() {
        // §3 / §3.1: a strip whose innermost kept column sits just off the plane.
        // With use_snap_center and dist > offset, that column snaps exactly onto
        // x = 0; without snap it keeps its offset position.
        let offset = 0.08_f32;
        let build = || {
            let mut mesh = LoopMesh::new();
            // Two columns: near-plane at x = +offset and far at x = +1.2, plus a
            // negative column at x = -1 so the strip straddles.
            let xs = [-1.0_f32, offset, 1.2];
            let mut bottom = Vec::new();
            let mut top = Vec::new();
            for &x in &xs {
                bottom.push(mesh.create_vertex(Vec3::new(x, 0.0, 0.0)));
                top.push(mesh.create_vertex(Vec3::new(x, 1.0, 0.0)));
            }
            for c in 0..2 {
                make_face(&mut mesh, &[bottom[c], bottom[c + 1], top[c + 1], top[c]]);
            }
            (mesh, bottom[1], top[1]) // the near-plane column verts
        };

        // Snap on, dist = 0.2 > offset, clear_inner (keep positive incl. the
        // near-plane column).
        let (mut snapped, nb, nt) = build();
        let geom = whole_mesh_geom(&snapped);
        snapped.bisect_plane(&geom, params(0.2, true, true, false));
        assert!(
            snapped.vertex_position(nb).x.abs() < 1e-6,
            "near-plane vert snapped onto x = 0"
        );
        assert!(snapped.vertex_position(nt).x.abs() < 1e-6);

        // Snap off, same dist/clear: the column keeps its offset position.
        let (mut kept, nb2, nt2) = build();
        let geom2 = whole_mesh_geom(&kept);
        kept.bisect_plane(&geom2, params(0.2, false, true, false));
        assert!(
            (kept.vertex_position(nb2).x - offset).abs() < 1e-6,
            "without snap the near-plane vert keeps its offset"
        );
        assert!((kept.vertex_position(nt2).x - offset).abs() < 1e-6);
    }

    #[test]
    fn dist_classifies_near_plane_column_on_seam_vs_zero() {
        // §3: dist = 0.2 > offset classifies the near-plane column on-seam (so
        // it is kept under clear_inner as a seam vert and not cut), while
        // dist = 0 leaves it on the positive side.
        let offset = 0.08_f32;
        let build = || {
            let mut mesh = LoopMesh::new();
            let xs = [-1.0_f32, offset, 1.2];
            let mut bottom = Vec::new();
            let mut top = Vec::new();
            for &x in &xs {
                bottom.push(mesh.create_vertex(Vec3::new(x, 0.0, 0.0)));
                top.push(mesh.create_vertex(Vec3::new(x, 1.0, 0.0)));
            }
            for c in 0..2 {
                make_face(&mut mesh, &[bottom[c], bottom[c + 1], top[c + 1], top[c]]);
            }
            (mesh, bottom[1])
        };

        // dist large: the near-plane column is on-seam, reported in the seam,
        // left at its offset (no snap).
        let (mut wide, near) = build();
        let geom = whole_mesh_geom(&wide);
        let r_wide = wide.bisect_plane(&geom, params(0.2, false, true, false));
        assert!(
            r_wide.seam_verts.contains(&near),
            "near-plane column is a seam vert under dist = 0.2"
        );
        assert!(
            wide.vertices.try_get(near).is_some(),
            "kept under clear_inner"
        );

        // dist = 0: the near-plane column is strictly positive; under
        // clear_inner it is kept, but it is NOT a seam vert.
        let (mut tight, near2) = build();
        let geom2 = whole_mesh_geom(&tight);
        let r_tight = tight.bisect_plane(&geom2, params(0.0, false, true, false));
        assert!(
            !r_tight.seam_verts.contains(&near2),
            "with dist = 0 the offset column is not on-seam"
        );
    }

    #[test]
    fn split_faces_keep_parent_winding() {
        // §4 / §8.1: cut a multi-face strip between columns so every kept face is
        // a freshly split sub-face; each sub-face must keep the parent's CCW
        // winding (normal +z). The face-split preserves orientation.
        let (mut mesh, faces, _verts) = strip(0.0, 3); // x = 0..3
                                                       // Parent faces are wound CCW in the z = 0 plane → +z normal.
        for &f in &faces {
            assert!(mesh.face_compute_normal(f).z > 0.0, "parent +z");
        }
        let geom = whole_mesh_geom(&mesh);
        let p = BisectParams {
            plane_co: Vec3::new(1.5, 0.0, 0.0),
            plane_no: PLANE_NO_X,
            dist: 0.0,
            use_snap_center: false,
            clear_inner: false,
            clear_outer: true,
        };
        mesh.bisect_plane(&geom, p);

        for f in mesh.face_keys() {
            assert!(
                mesh.face_compute_normal(f).z > 0.0,
                "every sub-face keeps the parent +z winding (no flip)"
            );
        }
    }

    #[test]
    fn empty_geom_is_noop() {
        let (mut mesh, _f, _v) = strip(-1.0, 2);
        let (v0, e0, f0) = (mesh.vertex_count(), mesh.edge_count(), mesh.face_count());
        let result = mesh.bisect_plane(&[], params(0.0, true, false, true));
        assert_eq!(
            (v0, e0, f0),
            (mesh.vertex_count(), mesh.edge_count(), mesh.face_count())
        );
        // Nothing tagged → nothing classified on-plane → empty seam.
        assert!(result.seam_verts.is_empty());
        assert!(result.seam_edges.is_empty());
    }
}
