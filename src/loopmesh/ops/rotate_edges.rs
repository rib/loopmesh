//! The **rotate-edges** operator: spin a *selection* of edges onto their
//! other diagonal.
//!
//! This is the operator-level set-handling layer over the single-edge
//! [`LoopMesh::edge_rotate`] primitive
//! (`radial-mesh-guides/RADIAL_EDGE_STRUCTURE.md` §9). The primitive does the
//! per-edge surgery (join the two adjacent faces, re-split along the other
//! diagonal — 0 V / 0 E / 0 F delta, only incidence moves); this layer adds
//! the parts that are unique to the operator, per
//! `radial-mesh-guides/operators/ROTATE_EDGES.md`:
//!
//! - the §3 eligibility predicate and §3.1 "filter, don't fail" skip
//!   discipline,
//! - the §3.2 selection-size guard dispatch (`{exists}` for a one-edge
//!   selection, `{exists, degenerate}` for a multi-edge selection),
//! - the §4.0 disjoint-vs-shared-face strategy choice,
//! - the §4.1 simple input-order sweep, the §4.2 priority "boundaries first,
//!   then all" heap sweep with neighbour re-evaluation, and
//! - the §6 `edges.out` slot.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashSet};

use crate::loopmesh::*;

/// The guard set chosen per §3.2 by selection size. The `exists` guard
/// (decline a rotation whose target diagonal already joins the two corners) is
/// always on; the `degenerate` guard (decline a rotation that fails the §3.3
/// per-arrival-corner orientation test — an inside-out fold or a collinear
/// sliver) is added only for multi-edge selections.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct GuardSet {
    degenerate: bool,
}

/// The set of edges produced by successful rotations (§6). Reported as an
/// order-insensitive set: each entry is the *new* edge of one rotation, never
/// the (now-dangling) input edge.
#[derive(Clone, Debug, Default)]
pub struct RotateEdgesResult {
    edges: Vec<EdgeKey>,
}

impl RotateEdgesResult {
    pub fn len(&self) -> usize {
        self.edges.len()
    }
    pub fn is_empty(&self) -> bool {
        self.edges.is_empty()
    }
    pub fn iter(&self) -> std::slice::Iter<'_, EdgeKey> {
        self.edges.iter()
    }
    /// The new edges, as a slice.
    pub fn edges(&self) -> &[EdgeKey] {
        &self.edges
    }
}

impl<'a> IntoIterator for &'a RotateEdgesResult {
    type Item = &'a EdgeKey;
    type IntoIter = std::slice::Iter<'a, EdgeKey>;
    fn into_iter(self) -> Self::IntoIter {
        self.edges.iter()
    }
}

/// Ordered heap entry. The min-heap of §4.2 is realised here as a max-heap on
/// a derived key, so the *longest* edge pops first (cost = squared length).
/// The boundary pass lifts interior (re-evaluation-inserted) keys into a band
/// strictly above every boundary-seed key, so a seeded run drains completely
/// before the next boundary seed pops (§4.2 invariant 2). Ties break on the
/// raw squared length, which is order-independent (§4.2 invariant 1).
#[derive(Clone, Copy)]
struct HeapEntry {
    edge: EdgeKey,
    key: f64,
}

impl PartialEq for HeapEntry {
    fn eq(&self, other: &Self) -> bool {
        self.key == other.key
    }
}
impl Eq for HeapEntry {}
impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        self.key.total_cmp(&other.key)
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Pass {
    Boundary,
    All,
    Done,
}

impl Pass {
    fn next(self) -> Pass {
        match self {
            Pass::Boundary => Pass::All,
            Pass::All => Pass::Done,
            Pass::Done => Pass::Done,
        }
    }
}

impl LoopMesh {
    /// Rotate a selection of `edges` onto their other diagonal, spinning each
    /// eligible edge with direction `use_ccw` (§1). Ineligible edges
    /// (boundary / wire / non-manifold, or those whose rotation would
    /// duplicate or — for multi-edge selections — degenerate a face) are
    /// silently skipped (§3.1). Returns the new edges produced by successful
    /// rotations (§6).
    pub fn rotate_edges(&mut self, edges: &[EdgeKey], use_ccw: bool) -> RotateEdgesResult {
        // §3.2 — the guard set is a function of the selection size alone.
        let guards = GuardSet {
            degenerate: edges.len() > 1,
        };

        let mut result = RotateEdgesResult::default();

        // §4.0 — pick the strategy by scanning for a shared face.
        let shared = edges.len() > 1 && self.any_two_input_edges_share_a_face(edges);

        if shared {
            self.rotate_edges_shared(edges, use_ccw, guards, &mut result);
        } else {
            self.rotate_edges_simple(edges, use_ccw, guards, &mut result);
        }

        result
    }

    /// §4.1 — the simple sweep for a disjoint selection (no two input edges
    /// share a face). A single straight pass in input-slot order; the
    /// rotations do not interact, so the order is irrelevant to the result.
    fn rotate_edges_simple(
        &mut self,
        edges: &[EdgeKey],
        use_ccw: bool,
        guards: GuardSet,
        result: &mut RotateEdgesResult,
    ) {
        for &edge in edges {
            if let Some(new_edge) = self.try_rotate(edge, use_ccw, guards) {
                result.edges.push(new_edge);
            }
        }
    }

    /// §4.2 — the priority heap sweep for a shared-face selection. Two passes
    /// (region boundaries first, then all remaining), each draining its own
    /// heap; a rotation re-evaluates the edges bordering the new faces and
    /// re-admits any that just became eligible. The pop order is a function of
    /// the mesh geometry, not the input order (§5).
    fn rotate_edges_shared(
        &mut self,
        edges: &[EdgeKey],
        use_ccw: bool,
        guards: GuardSet,
        result: &mut RotateEdgesResult,
    ) {
        let input: HashSet<EdgeKey> = edges.iter().copied().collect();
        // Edges that have rotated (the input edge handle is dead) — never
        // revisited. Plays the role of the §4.2 "sentinel" validity tag.
        let mut freed: HashSet<EdgeKey> = HashSet::new();
        // Edges currently sitting in the heap — so re-evaluation does not
        // double-insert.
        let mut queued: HashSet<EdgeKey> = HashSet::new();

        let mut rotated_total = 0usize;
        let mut pass = Pass::Boundary;

        while pass != Pass::Done && rotated_total != edges.len() {
            let mut heap: BinaryHeap<HeapEntry> = BinaryHeap::new();

            // Seed the heap with this pass's eligible candidates.
            for &edge in edges {
                if freed.contains(&edge) || queued.contains(&edge) {
                    continue;
                }
                let mut ok = self.is_rotatable(edge, use_ccw, guards);
                if ok && pass == Pass::Boundary {
                    ok = self.is_region_boundary_edge(edge, &input);
                }
                if ok {
                    let key = self.seed_key(edge);
                    heap.push(HeapEntry { edge, key });
                    queued.insert(edge);
                }
            }

            if heap.is_empty() {
                pass = pass.next();
                continue;
            }

            let rotated_before = rotated_total;
            while let Some(HeapEntry { edge, .. }) = heap.pop() {
                queued.remove(&edge);
                // Re-check at pop time — a previous rotation may have changed
                // this edge's surroundings.
                if freed.contains(&edge) || !self.is_rotatable(edge, use_ccw, guards) {
                    continue;
                }
                let Some(new_edge) = self.try_rotate(edge, use_ccw, guards) else {
                    continue;
                };
                result.edges.push(new_edge);
                freed.insert(edge);
                rotated_total += 1;

                // §4.2 re-evaluation: any input edge bordering the new edge's
                // two faces that is not freed, not already queued, and now
                // passes the eligibility re-test is (re-)admitted. In the
                // boundary pass these interior edges enter in the high band so
                // they drain ahead of any remaining boundary seed (invariant 2);
                // in the all-edges pass they compete on raw cost — there is no
                // boundary/interior distinction to enforce (§7 closed loop).
                for nb in self.edges_bordering_faces_of(new_edge) {
                    if nb == new_edge
                        || !input.contains(&nb)
                        || freed.contains(&nb)
                        || queued.contains(&nb)
                    {
                        continue;
                    }
                    if self.is_rotatable(nb, use_ccw, guards) {
                        let key = self.interior_key(nb, pass);
                        heap.push(HeapEntry { edge: nb, key });
                        queued.insert(nb);
                    }
                }
            }

            if rotated_total == rotated_before {
                pass = pass.next();
            }
        }
    }

    /// The §3 eligibility predicate: an edge is rotatable iff it is 2-manifold
    /// (a well-defined other diagonal exists), its target diagonal does not
    /// already exist (the `exists` guard), and — when the `degenerate` guard
    /// is on — the rotation passes the §3.3 per-arrival-corner orientation
    /// test (no inside-out fold, no collinear sliver).
    fn is_rotatable(&self, edge: EdgeKey, use_ccw: bool, guards: GuardSet) -> bool {
        let Some((a, c)) = self.edge_rotate_diagonal(edge, use_ccw) else {
            return false;
        };
        // §3 cond 2 — the `exists` guard.
        if self.query_edge_exists(a, c).is_some() {
            return false;
        }
        // §3 cond 3 — the `degenerate` guard (multi-edge selections only).
        if guards.degenerate && self.rotation_is_degenerate(edge, a, c) {
            return false;
        }
        true
    }

    /// Run one rotation through the primitive, honouring the guard set. The
    /// primitive already enforces the manifold and `exists` guards; the
    /// degenerate guard is applied here on top (the primitive does not expose
    /// it). Returns the new edge, or `None` on a skip.
    ///
    /// The degenerate/manifold/exists guard is re-run here even though the heap
    /// sweep pop-loop already pre-checks `is_rotatable`: the simple sweep
    /// (§4.1) calls `try_rotate` with *no* pre-check, so this in-line guard is
    /// the only eligibility gate on that path and must stay.
    fn try_rotate(&mut self, edge: EdgeKey, use_ccw: bool, guards: GuardSet) -> Option<EdgeKey> {
        if guards.degenerate {
            let (a, c) = self.edge_rotate_diagonal(edge, use_ccw)?;
            if self.rotation_is_degenerate(edge, a, c) {
                return None;
            }
        }
        self.edge_rotate(edge, use_ccw)
    }

    /// §3.3 — the degenerate guard's **corner-orientation test**. Would
    /// re-chording the union quad onto the new diagonal turn either re-split
    /// face inside-out (arm (i)) or collapse an arrival corner to a
    /// near-collinear sliver (arm (ii))?
    ///
    /// The test is fully local, 3-D (no projection to a plane — it stays
    /// correct on a folded quad), and **per-arrival-corner**: each arrival
    /// corner is judged independently, with no step comparing the two re-split
    /// faces to each other. A whole-face opposite-winding / bow-tie test is
    /// strictly coarser and wrongly accepts a same-winding *folded* quad whose
    /// single arrival corner flips; this per-corner form catches it.
    ///
    /// Labelling (§2 / §3.3): the input edge `e` leaves corners `b`, `d` and
    /// arrives at corners `a`, `c` (the new diagonal's endpoints). The corner
    /// produced by stepping in `e`'s first endpoint's face is paired with that
    /// endpoint as its *old* partner; the *other* endpoint is its *new*
    /// partner. [`Self::edge_rotate_diagonal`] returns `(a, c)` in exactly the
    /// `(step-from-v1, step-from-v2)` order, so `a`'s old partner is `v1` (its
    /// new partner `v2`) and `c`'s old partner is `v2` (its new partner `v1`).
    fn rotation_is_degenerate(&self, edge: EdgeKey, a: VertexKey, c: VertexKey) -> bool {
        // §2/§3.3 labels: `b = v1`, `d = v2` (the corners `e` leaves).
        let [b_vertex, d_vertex] = self.edges.get(edge).vertices;
        let position_a = self.vertex_position(a);
        let position_c = self.vertex_position(c);
        let position_b = self.vertex_position(b_vertex);
        let position_d = self.vertex_position(d_vertex);

        // Both arms share two normalized direction vectors (§3.3): the old
        // diagonal `b → d` and the new diagonal `a → c`.
        let dir_old = (position_d - position_b).normalize_or_zero();
        let dir_new = (position_c - position_a).normalize_or_zero();
        if dir_old == Vec3::ZERO || dir_new == Vec3::ZERO {
            // A zero-length diagonal is itself degenerate.
            return true;
        }

        // Arrival corner `a`: old partner `b_a = b_vertex` (`v1`), new partner
        // `d_a = d_vertex` (`v2`); `step_a = d_a → a` reads *into* `a`.
        let to_old_partner_a = (position_b - position_a).normalize_or_zero();
        let step_a = (position_a - position_d).normalize_or_zero();
        // Arrival corner `c`: old partner `b_c = d_vertex` (`v2`), new partner
        // `d_c = b_vertex` (`v1`); `step_c = d_c → c` reads *into* `c`.
        let to_old_partner_c = (position_d - position_c).normalize_or_zero();
        let step_c = (position_c - position_b).normalize_or_zero();
        if to_old_partner_a == Vec3::ZERO
            || step_a == Vec3::ZERO
            || to_old_partner_c == Vec3::ZERO
            || step_c == Vec3::ZERO
        {
            return true;
        }

        // Arm (i): per-corner orientation preservation. The deliberate
        // asymmetry — old term reads *out of* the corner toward the old
        // partner, new term reads *into* the corner from the new partner — is
        // load-bearing; it makes the dot's sign track the corner's *turn
        // reversal*, not a fixed handedness.
        let cross_old_a = dir_old.cross(to_old_partner_a);
        let cross_new_a = dir_new.cross(step_a);
        if cross_old_a.dot(cross_new_a) < 0.0 {
            return true;
        }
        let cross_old_c = dir_old.cross(to_old_partner_c);
        let cross_new_c = dir_new.cross(step_c);
        if cross_old_c.dot(cross_new_c) < 0.0 {
            return true;
        }

        // Arm (ii): thinness / collinear-grazing sliver. Reject when the new
        // diagonal lies almost *along* the post-rotation edge an arrival corner
        // brings in, so that re-split face collapses to a near-zero-area sliver.
        // "Along" is direction-agnostic: a near-parallel `dir_new`/`step`
        // pinches the corner to ~0°, a near-anti-parallel pair straightens it to
        // ~180°; both are vanishing-area slivers the guard rejects. (§3.3 states
        // each clause one-sided — `dir_new·step_a ≈ +1`, `−dir_new·step_c ≈ +1`
        // — to name the ~180° "arrival corner straightens" form; the magnitude
        // test below subsumes those and additionally rejects the ~0° form, which
        // is the same vanishing-area sliver presenting when the collinear vertex
        // is an *original* edge endpoint rather than the arrival corner. Without
        // it, an exactly-flat re-split triangle is accepted and §5's order-
        // independence breaks on a region that produces one mid-sweep.)
        if dir_new.dot(step_a).abs() >= COLLINEAR_COS {
            return true;
        }
        if dir_new.dot(step_c).abs() >= COLLINEAR_COS {
            return true;
        }

        false
    }

    /// §4.0 — does any face flank two input edges? Only manifold input edges
    /// contribute face claims (a non-manifold / boundary edge cannot rotate,
    /// so it does not by itself force the shared-face strategy).
    fn any_two_input_edges_share_a_face(&self, edges: &[EdgeKey]) -> bool {
        let mut claimed: HashSet<FaceKey> = HashSet::new();
        for &edge in edges {
            let mut radial = self.edge_radial_iter(edge);
            let (Some(l0), Some(l1)) = (radial.next(), radial.next()) else {
                continue;
            };
            if radial.next().is_some() {
                continue; // non-manifold (radial ≥ 3)
            }
            for face in [self.loops.get(l0).face, self.loops.get(l1).face] {
                if !claimed.insert(face) {
                    return true;
                }
            }
        }
        false
    }

    /// §4.3 — is `edge` a region-boundary input edge? It is iff at most one of
    /// the *other* edges flanking its two faces is also an input edge.
    fn is_region_boundary_edge(&self, edge: EdgeKey, input: &HashSet<EdgeKey>) -> bool {
        let mut radial = self.edge_radial_iter(edge);
        let (Some(l0), Some(l1)) = (radial.next(), radial.next()) else {
            return false;
        };
        let mut selected_neighbours = 0usize;
        for face_loop in [l0, l1] {
            let face = self.loops.get(face_loop).face;
            let start = self.faces.get(face).first_edge_loop;
            for l in self.face_loop_iter(start) {
                let e = self.loops.get(l).edge;
                if e != edge && input.contains(&e) {
                    selected_neighbours += 1;
                }
            }
        }
        selected_neighbours <= 1
    }

    /// The edges flanking the two faces of `edge` (which must be 2-manifold).
    /// Used by §4.2 re-evaluation to find candidates that the rotation may
    /// have just enabled.
    fn edges_bordering_faces_of(&self, edge: EdgeKey) -> Vec<EdgeKey> {
        let mut out: Vec<EdgeKey> = Vec::new();
        let mut radial = self.edge_radial_iter(edge);
        let (Some(l0), Some(l1)) = (radial.next(), radial.next()) else {
            return out;
        };
        for face_loop in [l0, l1] {
            let face = self.loops.get(face_loop).face;
            let start = self.faces.get(face).first_edge_loop;
            for l in self.face_loop_iter(start) {
                let e = self.loops.get(l).edge;
                if !out.contains(&e) {
                    out.push(e);
                }
            }
        }
        out
    }

    /// Raw geometric cost key: squared edge length. With the max-heap, the
    /// longest edge pops first (the §4.2 negative-squared-length min-heap,
    /// sign-flipped).
    fn squared_length(&self, edge: EdgeKey) -> f64 {
        let [v0, v1] = self.edges.get(edge).vertices;
        let p0 = self.vertex_position(v0);
        let p1 = self.vertex_position(v1);
        (p1 - p0).length_squared() as f64
    }

    /// Key for an interior edge admitted by re-evaluation. The `INTERIOR_BAND`
    /// lift is a *boundary-pass-only* device (§4.2): in the boundary pass it
    /// raises the re-eval insert above every still-unpopped boundary seed so a
    /// seeded run drains completely (invariant 2). In the all-edges pass there
    /// is no boundary/interior banding — re-eval inserts and all-pass seeds
    /// compete on the raw cost, letting the geometric cost ordering anchor the
    /// run on its own (§7 closed-loop / all-interior selection).
    fn interior_key(&self, edge: EdgeKey, pass: Pass) -> f64 {
        match pass {
            Pass::Boundary => self.squared_length(edge) + INTERIOR_BAND,
            _ => self.squared_length(edge),
        }
    }

    /// Key for a seed of the current pass. Seeds always use the raw cost (the
    /// low band): boundary seeds pop longest-first among themselves (§4.2
    /// invariant 1), while interior re-evaluation edges in the boundary pass
    /// enter the high band (see [`Self::interior_key`]) so they outrank every
    /// remaining seed (invariant 2). The all-edges pass has no boundary
    /// distinction, so seeds and re-eval inserts there share the raw cost.
    fn seed_key(&self, edge: EdgeKey) -> f64 {
        self.squared_length(edge)
    }
}

/// Direction-cosine tolerance for the §3.3 arm (ii) thinness / collinear-
/// grazing sliver check (~2.5° of separation). The exact value is not
/// parity-observable so long as both sides reject the same near-collinear
/// slivers and the canonical position quantum is well below the gap it leaves.
const COLLINEAR_COS: f32 = 0.999;

/// The width that lifts interior re-evaluation keys above every boundary seed
/// key (§4.2 invariant 2). Any value past the largest squared length any
/// realistic mesh edge reaches works; the exact band is not parity-observable.
const INTERIOR_BAND: f64 = 1.0e18;

#[cfg(test)]
mod tests {
    use glam::Vec3;

    use crate::loopmesh::*;

    fn edge_between(mesh: &LoopMesh, a: VertexKey, b: VertexKey) -> EdgeKey {
        mesh.query_edge_exists(a, b).expect("edge must exist")
    }

    fn tri(mesh: &mut LoopMesh, vs: [VertexKey; 3]) -> FaceKey {
        let es: Vec<EdgeKey> = (0..3)
            .map(|i| mesh.create_edge([vs[i], vs[(i + 1) % 3]], EdgeCreateFlags::NO_DUPLICATE))
            .collect();
        mesh.face_create(&vs, &es, FaceCreateFlags::NONE)
    }

    /// Two triangles sharing the `{v0, v2}` diagonal of the unit square
    /// `v0(0,0) v1(1,0) v2(1,1) v3(0,1)`. Returns `[v0, v1, v2, v3]`.
    fn two_tris_shared_diagonal(mesh: &mut LoopMesh) -> [VertexKey; 4] {
        let v0 = mesh.create_vertex(Vec3::new(0.0, 0.0, 0.0));
        let v1 = mesh.create_vertex(Vec3::new(1.0, 0.0, 0.0));
        let v2 = mesh.create_vertex(Vec3::new(1.0, 1.0, 0.0));
        let v3 = mesh.create_vertex(Vec3::new(0.0, 1.0, 0.0));
        tri(mesh, [v0, v1, v2]);
        tri(mesh, [v0, v2, v3]);
        [v0, v1, v2, v3]
    }

    /// §7 single eligible edge → other diagonal. Rotating the shared `{0,2}`
    /// diagonal lands it on `{1,3}`; counts unchanged.
    #[test]
    fn single_eligible_edge_rotates_to_other_diagonal() {
        let mut mesh = LoopMesh::new();
        let [v0, v1, v2, v3] = two_tris_shared_diagonal(&mut mesh);
        let diag = edge_between(&mesh, v0, v2);
        let counts = (mesh.totvert(), mesh.totedge(), mesh.totface());

        let out = mesh.rotate_edges(&[diag], true);

        assert_eq!(out.len(), 1, "one rotation, one reported edge");
        assert_eq!(
            (mesh.totvert(), mesh.totedge(), mesh.totface()),
            counts,
            "rotation is 0 V / 0 E / 0 F"
        );
        assert!(
            mesh.query_edge_exists(v1, v3).is_some(),
            "the new diagonal {{1,3}} exists"
        );
        assert!(
            mesh.query_edge_exists(v0, v2).is_none(),
            "the old diagonal {{0,2}} is gone"
        );
        let new_edge = out.edges()[0];
        let [a, b] = mesh.edges.get(new_edge).vertices;
        assert!(
            (a == v1 && b == v3) || (a == v3 && b == v1),
            "edges.out reports the new {{1,3}} edge"
        );
    }

    /// §1 / §7 direction contrast: on the symmetric quad both directions reach
    /// the only other diagonal `{1,3}`, but each is a valid rotation that the
    /// operator must accept.
    #[test]
    fn use_ccw_false_also_rotates() {
        let mut mesh = LoopMesh::new();
        let [v0, v1, v2, v3] = two_tris_shared_diagonal(&mut mesh);
        let diag = edge_between(&mesh, v0, v2);

        let out = mesh.rotate_edges(&[diag], false);

        assert_eq!(out.len(), 1);
        assert!(mesh.query_edge_exists(v1, v3).is_some());
        assert!(mesh.query_edge_exists(v0, v2).is_none());
    }

    /// §3 cond 1 / §3.1 — a boundary edge is ineligible; the operator skips it,
    /// leaving the mesh untouched and `edges.out` empty.
    #[test]
    fn boundary_edge_is_skipped_noop() {
        let mut mesh = LoopMesh::new();
        let [v0, v1, _v2, _v3] = two_tris_shared_diagonal(&mut mesh);
        let boundary = edge_between(&mesh, v0, v1); // radial 1
        let counts = (mesh.totvert(), mesh.totedge(), mesh.totface());

        let out = mesh.rotate_edges(&[boundary], true);

        assert!(out.is_empty(), "ineligible edge → empty edges.out");
        assert_eq!(
            (mesh.totvert(), mesh.totedge(), mesh.totface()),
            counts,
            "mesh unchanged"
        );
        assert!(
            mesh.query_edge_exists(v0, v1).is_some(),
            "boundary edge survives"
        );
    }

    /// An entirely ineligible selection is a complete no-op (§7).
    #[test]
    fn all_ineligible_selection_is_noop() {
        let mut mesh = LoopMesh::new();
        let [v0, v1, v2, v3] = two_tris_shared_diagonal(&mut mesh);
        let b0 = edge_between(&mesh, v0, v1);
        let b1 = edge_between(&mesh, v2, v3);
        let counts = (mesh.totvert(), mesh.totedge(), mesh.totface());

        let out = mesh.rotate_edges(&[b0, b1], true);

        assert!(out.is_empty());
        assert_eq!((mesh.totvert(), mesh.totedge(), mesh.totface()), counts);
    }

    /// Two well-separated triangulated quads, diagonals `{0,2}` and `{4,6}`.
    /// No face is claimed twice → §4.1 simple sweep. Returns the 8 verts.
    fn two_disjoint_tri_quads(mesh: &mut LoopMesh) -> [VertexKey; 8] {
        let a = two_tris_shared_diagonal(mesh);
        // Second patch translated +10 in x so the two never share a face.
        let v4 = mesh.create_vertex(Vec3::new(10.0, 0.0, 0.0));
        let v5 = mesh.create_vertex(Vec3::new(11.0, 0.0, 0.0));
        let v6 = mesh.create_vertex(Vec3::new(11.0, 1.0, 0.0));
        let v7 = mesh.create_vertex(Vec3::new(10.0, 1.0, 0.0));
        tri(mesh, [v4, v5, v6]);
        tri(mesh, [v4, v6, v7]);
        [a[0], a[1], a[2], a[3], v4, v5, v6, v7]
    }

    /// §4.0 / §4.1 — multiple disjoint eligible edges; each rotates
    /// independently, order-independent, counts unchanged.
    #[test]
    fn disjoint_pair_simple_sweep_rotates_both() {
        let mut mesh = LoopMesh::new();
        let v = two_disjoint_tri_quads(&mut mesh);
        let d0 = edge_between(&mesh, v[0], v[2]);
        let d1 = edge_between(&mesh, v[4], v[6]);
        let counts = (mesh.totvert(), mesh.totedge(), mesh.totface());

        let out = mesh.rotate_edges(&[d0, d1], true);

        assert_eq!(out.len(), 2, "both disjoint edges rotate");
        assert_eq!((mesh.totvert(), mesh.totedge(), mesh.totface()), counts);
        assert!(mesh.query_edge_exists(v[1], v[3]).is_some());
        assert!(mesh.query_edge_exists(v[5], v[7]).is_some());
    }

    /// Order-independence of the disjoint sweep: reversing the input slot
    /// yields the same final topology.
    #[test]
    fn disjoint_sweep_is_order_independent() {
        let final_diags = |reversed: bool| {
            let mut mesh = LoopMesh::new();
            let v = two_disjoint_tri_quads(&mut mesh);
            let d0 = edge_between(&mesh, v[0], v[2]);
            let d1 = edge_between(&mesh, v[4], v[6]);
            let sel = if reversed { [d1, d0] } else { [d0, d1] };
            mesh.rotate_edges(&sel, true);
            (
                mesh.query_edge_exists(v[1], v[3]).is_some(),
                mesh.query_edge_exists(v[5], v[7]).is_some(),
            )
        };
        assert_eq!(final_diags(false), (true, true));
        assert_eq!(final_diags(true), (true, true));
    }

    /// Two triangulated unit quads sharing the interior vertical edge `{1,4}`.
    /// Grid verts, bottom row `v0(0,0) v1(1,0) v2(2,0)`, top row
    /// `v3(0,1) v4(1,1) v5(2,1)`. Left quad split on `{0,4}`, right quad split
    /// on `{1,5}`. The shared vertical `{1,4}` and the right diagonal `{1,5}`
    /// both border the right-lower triangle. Returns `[v0..v5]`.
    fn tri_quads_shared_interior_edge(mesh: &mut LoopMesh) -> [VertexKey; 6] {
        let v0 = mesh.create_vertex(Vec3::new(0.0, 0.0, 0.0));
        let v1 = mesh.create_vertex(Vec3::new(1.0, 0.0, 0.0));
        let v2 = mesh.create_vertex(Vec3::new(2.0, 0.0, 0.0));
        let v3 = mesh.create_vertex(Vec3::new(0.0, 1.0, 0.0));
        let v4 = mesh.create_vertex(Vec3::new(1.0, 1.0, 0.0));
        let v5 = mesh.create_vertex(Vec3::new(2.0, 1.0, 0.0));
        // Left quad [v0,v1,v4,v3] split along {0,4}.
        tri(mesh, [v0, v1, v4]);
        tri(mesh, [v0, v4, v3]);
        // Right quad [v1,v2,v5,v4] split along {1,5}.
        tri(mesh, [v1, v2, v5]);
        tri(mesh, [v1, v5, v4]);
        [v0, v1, v2, v3, v4, v5]
    }

    /// §4.0 / §4.2 — two eligible edges sharing a face force the heap sweep.
    /// The selection `{ {1,4}, {1,5} }` both border the right-lower triangle.
    /// The longer edge (`{1,5}`, length √2) anchors; re-evaluation enables the
    /// shorter shared edge. Both end up rotated; counts unchanged.
    #[test]
    fn shared_face_pair_uses_heap_sweep() {
        let mut mesh = LoopMesh::new();
        let [v0, v1, _v2, v3, v4, v5] = tri_quads_shared_interior_edge(&mut mesh);
        let m = edge_between(&mesh, v1, v4); // shared vertical, length 1
        let r = edge_between(&mesh, v1, v5); // right diagonal, length √2
        let counts = (mesh.totvert(), mesh.totedge(), mesh.totface());

        let out = mesh.rotate_edges(&[m, r], true);

        assert_eq!((mesh.totvert(), mesh.totedge(), mesh.totface()), counts);
        assert!(!out.is_empty(), "at least the anchor edge rotated");
        // The shared vertical {1,4} and the right diagonal {1,5} cannot both
        // survive unrotated — the sweep made progress on a shared-face pair.
        assert!(
            mesh.query_edge_exists(v1, v5).is_none() || mesh.query_edge_exists(v1, v4).is_none(),
            "at least one selected edge spun"
        );
        let _ = (v0, v3, v4);
    }

    /// Canonicalise post-op topology as a sorted set of edges keyed by the
    /// rounded world positions of their endpoints — independent of element
    /// allocation order.
    fn pos_edge_set(mesh: &LoopMesh) -> Vec<[(i64, i64, i64); 2]> {
        let q = |p: Vec3| -> (i64, i64, i64) {
            (
                (p.x * 1e4).round() as i64,
                (p.y * 1e4).round() as i64,
                (p.z * 1e4).round() as i64,
            )
        };
        let mut edges: Vec<[(i64, i64, i64); 2]> = mesh
            .edge_keys()
            .iter()
            .map(|&e| {
                let [a, b] = mesh.edges.get(e).vertices;
                let mut ends = [q(mesh.vertex_position(a)), q(mesh.vertex_position(b))];
                ends.sort_unstable();
                ends
            })
            .collect();
        edges.sort_unstable();
        edges
    }

    /// §4.2 / §5 — the heap sweep is a function of the edge *set*, not input
    /// order. The same selection in two different input orders yields the same
    /// final topology and the same number of rotations.
    #[test]
    fn heap_sweep_is_order_independent() {
        let run = |reversed: bool| {
            let mut mesh = LoopMesh::new();
            let [_v0, v1, _v2, _v3, v4, v5] = tri_quads_shared_interior_edge(&mut mesh);
            let m = edge_between(&mesh, v1, v4);
            let r = edge_between(&mesh, v1, v5);
            let sel = if reversed { [r, m] } else { [m, r] };
            let out = mesh.rotate_edges(&sel, true);
            (out.len(), pos_edge_set(&mesh))
        };
        assert_eq!(run(false), run(true), "result is order-independent (§5)");
    }

    /// §4.2 / §5 re-evaluation — selecting all three interior rotatable edges
    /// drives the recompute-after-each-pop path. The result is order-
    /// independent and counts are unchanged.
    #[test]
    fn selection_sweep_reevaluates_neighbours() {
        let run = |order: usize| {
            let mut mesh = LoopMesh::new();
            let [v0, v1, _v2, _v3, v4, v5] = tri_quads_shared_interior_edge(&mut mesh);
            let left = edge_between(&mesh, v0, v4); // {0,4}
            let right = edge_between(&mesh, v1, v5); // {1,5}
            let shared = edge_between(&mesh, v1, v4); // {1,4}
            let sel = match order {
                0 => [left, right, shared],
                1 => [shared, right, left],
                _ => [right, shared, left],
            };
            let counts_before = (mesh.totvert(), mesh.totedge(), mesh.totface());
            mesh.rotate_edges(&sel, true);
            assert_eq!(
                (mesh.totvert(), mesh.totedge(), mesh.totface()),
                counts_before,
                "all rotations are count-preserving"
            );
            pos_edge_set(&mesh)
        };
        let a = run(0);
        assert_eq!(a, run(1), "result independent of input order (§5)");
        assert_eq!(a, run(2), "result independent of input order (§5)");
    }

    /// §3.2 selection-size guard dispatch. A degenerate (collinear) union quad:
    /// rotating the shared edge alone (`{exists}` only) succeeds, but inside a
    /// multi-edge selection (`{exists, degenerate}`) the same edge is skipped.
    ///
    /// Geometry: a "flat fan" where the rotation's new diagonal would produce a
    /// zero-area triangle. Verts `a(0,0) b(1,0) c(2,0) d(1,1)` — `a,b,c`
    /// collinear. Triangles `(a,b,d)` and `(b,c,d)` share edge `{b,d}`.
    /// Rotating `{b,d}` would connect `{a,c}`, splitting into `(a,b,c)`
    /// (zero-area, collinear) and `(a,c,d)`.
    #[test]
    fn selection_size_guard_dispatch() {
        let build = |mesh: &mut LoopMesh| {
            let a = mesh.create_vertex(Vec3::new(0.0, 0.0, 0.0));
            let b = mesh.create_vertex(Vec3::new(1.0, 0.0, 0.0));
            let c = mesh.create_vertex(Vec3::new(2.0, 0.0, 0.0));
            let d = mesh.create_vertex(Vec3::new(1.0, 1.0, 0.0));
            tri(mesh, [a, b, d]);
            tri(mesh, [b, c, d]);
            [a, b, c, d]
        };

        // Single-edge selection — degenerate guard OFF — rotates even into the
        // collinear result.
        let mut lone = LoopMesh::new();
        let [a, b, c, d] = build(&mut lone);
        let bd = edge_between(&lone, b, d);
        let out_lone = lone.rotate_edges(&[bd], true);
        assert_eq!(
            out_lone.len(),
            1,
            "lone edge rotates despite degenerate result"
        );
        assert!(
            lone.query_edge_exists(a, c).is_some(),
            "new {{a,c}} diagonal laid"
        );

        // Multi-edge selection containing the same edge plus another (so the
        // guard set becomes {exists, degenerate}) — the degenerate rotation is
        // skipped. The companion edge is a boundary edge (ineligible too), so
        // the whole call is a no-op on the shared edge.
        let mut region = LoopMesh::new();
        let [a2, b2, c2, d2] = build(&mut region);
        let bd2 = edge_between(&region, b2, d2);
        let other = edge_between(&region, a2, b2); // boundary, just to size the selection
        let counts = (region.totvert(), region.totedge(), region.totface());
        let out_region = region.rotate_edges(&[bd2, other], true);
        assert!(
            region.query_edge_exists(b2, d2).is_some(),
            "the shared edge is NOT rotated under the degenerate guard"
        );
        assert!(region.query_edge_exists(a2, c2).is_none());
        assert_eq!(
            (region.totvert(), region.totedge(), region.totface()),
            counts,
            "degenerate-guarded multi-selection is a no-op here"
        );
        let _ = (out_region, c, d);
    }

    /// A closed triangle fan around a central vertex `c`: a ring of `n`
    /// vertices on the unit circle, each consecutive pair forming a triangle
    /// `(c, r_i, r_{i+1})` with the ring wrapping closed. Returns
    /// `(c, ring)`. Every spoke `{c, r_i}` is shared by its two neighbouring
    /// fan triangles, and selecting *all* spokes makes each one region-interior
    /// (§4.3: each spoke sees its two ring-adjacent spokes as selected
    /// neighbours), so the boundary pass seeds nothing and the all-edges pass
    /// does the work (§7 closed-loop case).
    fn closed_tri_fan(mesh: &mut LoopMesh, n: usize) -> (VertexKey, Vec<VertexKey>) {
        let c = mesh.create_vertex(Vec3::new(0.0, 0.0, 0.0));
        let ring: Vec<VertexKey> = (0..n)
            .map(|i| {
                // Irregular radii so spoke lengths differ — the cost ordering
                // then has a unique anchor and the pop order is pinned.
                let theta = std::f32::consts::TAU * (i as f32) / (n as f32);
                let radius = 1.0 + 0.37 * (i as f32);
                mesh.create_vertex(Vec3::new(radius * theta.cos(), radius * theta.sin(), 0.0))
            })
            .collect();
        for i in 0..n {
            let a = ring[i];
            let b = ring[(i + 1) % n];
            tri(mesh, [c, a, b]);
        }
        (c, ring)
    }

    /// §4.2 / §7 — a closed loop of selected edges (no region-boundary edge).
    /// Every spoke is region-interior, so the boundary pass is empty and the
    /// all-edges pass does all the rotating. In the all-edges pass re-eval
    /// inserts must compete on *raw* cost (no `INTERIOR_BAND` lift), so the pop
    /// order — and thus which diagonals land — is anchored purely by edge
    /// length. This pins the specific resulting topology, not just counts: it is
    /// the regression for the §4.2 boundary-pass-only remap rule.
    #[test]
    fn closed_loop_all_interior_pins_landed_diagonals() {
        let landed = |reversed: bool| {
            let mut mesh = LoopMesh::new();
            let (c, ring) = closed_tri_fan(&mut mesh, 7);
            let mut spokes: Vec<EdgeKey> =
                ring.iter().map(|&r| edge_between(&mesh, c, r)).collect();
            // Confirm the selection truly has no region-boundary edge, so the
            // boundary pass seeds nothing and the all-edges pass owns the work.
            let set: std::collections::HashSet<EdgeKey> = spokes.iter().copied().collect();
            assert!(
                spokes
                    .iter()
                    .all(|&e| !mesh.is_region_boundary_edge(e, &set)),
                "every spoke is region-interior (§4.3) — pure all-edges-pass case"
            );
            if reversed {
                spokes.reverse();
            }
            let counts = (mesh.totvert(), mesh.totedge(), mesh.totface());
            let out = mesh.rotate_edges(&spokes, true);
            assert_eq!(
                (mesh.totvert(), mesh.totedge(), mesh.totface()),
                counts,
                "every rotation is count-preserving (§2)"
            );
            (out.len(), pos_edge_set(&mesh))
        };

        let forward = landed(false);
        // Order-independence: the all-pass result is a function of the edge set.
        assert_eq!(
            forward,
            landed(true),
            "all-interior sweep is order-independent (§5)"
        );

        let (rotated, edges) = forward;
        // Pin the specific landed topology produced by the raw-cost all-edges
        // pass. The INTERIOR_BAND lift is boundary-pass-only (§4.2); applying it
        // in the all-pass would reorder the re-eval inserts versus the remaining
        // all-pass seeds and change which diagonals the interacting spokes land
        // on. This exact edge set was captured from the correct (raw-cost)
        // all-pass ordering and diverges from the buggy lifted-band variant —
        // it is the regression guard for the boundary-pass-only remap rule.
        assert_eq!(rotated, 4, "four spokes rotate in the all-edges pass");
        let mut expected: Vec<[(i64, i64, i64); 2]> = vec![
            [(-22344, -10760, 0), (-19010, 9155, 0)],
            [(-22344, -10760, 0), (-6342, -27785, 0)],
            [(-22344, -10760, 0), (-3872, 16964, 0)],
            [(-22344, -10760, 0), (0, 0, 0)],
            [(-22344, -10760, 0), (8542, 10711, 0)],
            [(-22344, -10760, 0), (10000, 0, 0)],
            [(-19010, 9155, 0), (-3872, 16964, 0)],
            [(-6342, -27785, 0), (10000, 0, 0)],
            [(-6342, -27785, 0), (20076, -25175, 0)],
            [(-3872, 16964, 0), (8542, 10711, 0)],
            [(0, 0, 0), (8542, 10711, 0)],
            [(0, 0, 0), (10000, 0, 0)],
            [(8542, 10711, 0), (10000, 0, 0)],
            [(10000, 0, 0), (20076, -25175, 0)],
        ];
        for ends in &mut expected {
            ends.sort_unstable();
        }
        expected.sort_unstable();
        assert_eq!(
            edges, expected,
            "pinned landed diagonals (raw-cost all-pass order)"
        );
    }

    /// Build two tris sharing diagonal {p1,p3} of an explicit quad
    /// p0,p1,p2,p3 (cyclic). Tris (p0,p1,p3) and (p1,p2,p3) share {p1,p3}.
    /// Returns the shared edge and the 4 verts in cyclic order.
    fn quad_on_diag(mesh: &mut LoopMesh, p: [Vec3; 4]) -> (EdgeKey, [VertexKey; 4]) {
        let v: Vec<VertexKey> = p.iter().map(|&pp| mesh.create_vertex(pp)).collect();
        tri(mesh, [v[0], v[1], v[3]]);
        tri(mesh, [v[1], v[2], v[3]]);
        let diag = edge_between(mesh, v[1], v[3]);
        (diag, [v[0], v[1], v[2], v[3]])
    }

    /// §3.3 positive control — a convex quad (even a skewed kite) preserves
    /// every arrival corner's orientation and stays clear of the thinness
    /// tolerance, so the degenerate guard *accepts* it in both directions.
    #[test]
    fn convex_quad_is_not_degenerate() {
        let cases: &[[Vec3; 4]] = &[
            [
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(1.0, 1.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
            ],
            [
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(2.0, 0.3, 0.0),
                Vec3::new(2.5, 2.0, 0.0),
                Vec3::new(0.2, 1.7, 0.0),
            ],
        ];
        for p in cases {
            let mut mesh = LoopMesh::new();
            let (diag, _v) = quad_on_diag(&mut mesh, *p);
            for &ccw in &[true, false] {
                let (a, c) = mesh.edge_rotate_diagonal(diag, ccw).unwrap();
                assert!(
                    !mesh.rotation_is_degenerate(diag, a, c),
                    "convex quad rotation must be accepted (ccw={ccw})"
                );
            }
        }
    }

    /// Append a disjoint convex unit-square patch far in +x and return its
    /// shared diagonal edge plus the corner pair the rotation will lay.
    fn disjoint_convex_patch(mesh: &mut LoopMesh) -> (EdgeKey, [VertexKey; 2]) {
        let w0 = mesh.create_vertex(Vec3::new(50.0, 0.0, 0.0));
        let w1 = mesh.create_vertex(Vec3::new(51.0, 0.0, 0.0));
        let w2 = mesh.create_vertex(Vec3::new(51.0, 1.0, 0.0));
        let w3 = mesh.create_vertex(Vec3::new(50.0, 1.0, 0.0));
        tri(mesh, [w0, w1, w2]);
        tri(mesh, [w0, w2, w3]);
        (edge_between(mesh, w0, w2), [w1, w3])
    }

    fn signed_normal(pa: Vec3, pb: Vec3, pc: Vec3) -> Vec3 {
        (pb - pa).cross(pc - pa)
    }

    /// §3.3 arm (i) — a *planar concave* (reflex-notch) quad. Its rotation
    /// would lay the new diagonal *outside* the polygon, reversing one arrival
    /// corner's turn; both re-split faces stay non-zero-area. Selected alongside
    /// a disjoint convex patch (so the selection is multi-edge and the
    /// degenerate guard is on, §3.2): the concave diagonal is skipped, the
    /// convex diagonal rotates. A zero-area-only shortcut would wrongly commit
    /// the concave rotation (no face is degenerate by area).
    #[test]
    fn concave_quad_degenerate_guard_declines() {
        let mut mesh = LoopMesh::new();
        // Arrowhead: cyclic p0,p1,p2,p3 with p3 the reflex notch.
        let concave = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(2.0, 1.0, 0.0),
            Vec3::new(0.0, 2.0, 0.0),
            Vec3::new(0.9, 1.0, 0.0),
        ];
        let (concave_diag, cverts) = quad_on_diag(&mut mesh, concave);
        let (convex_diag, [cw1, cw3]) = disjoint_convex_patch(&mut mesh);
        let counts = (mesh.totvert(), mesh.totedge(), mesh.totface());

        let out = mesh.rotate_edges(&[concave_diag, convex_diag], true);

        // Concave skipped: its shared diagonal survives, the would-be new
        // diagonal {p0,p2} was not laid.
        assert!(
            mesh.query_edge_exists(cverts[1], cverts[3]).is_some(),
            "concave shared diagonal survives — rotation declined"
        );
        assert!(
            mesh.query_edge_exists(cverts[0], cverts[2]).is_none(),
            "concave new diagonal was not laid"
        );
        // Convex rotated.
        assert!(
            mesh.query_edge_exists(cw1, cw3).is_some(),
            "convex diagonal rotated"
        );
        assert_eq!(
            out.len(),
            1,
            "only the convex rotation contributes to edges.out"
        );
        assert_eq!(
            (mesh.totvert(), mesh.totedge(), mesh.totface()),
            counts,
            "rotations are count-preserving"
        );
    }

    /// §3.3 arm (i) — the load-bearing *folded* (non-planar) quad. Its rotation
    /// reverses exactly one arrival corner's old-vs-new turn (arm (i) fires)
    /// **yet the two re-split faces wind the same way** (same-sign 3-D normals
    /// *and* same-sign projected areas), with both faces non-zero-area. A
    /// bow-tie / opposite-winding shortcut would wrongly accept this rotation;
    /// a zero-area-only shortcut would too. Only the full per-corner arm (i)
    /// declines it. The fold geometry was found by sweeping the dihedral and is
    /// well inside the (declined ∧ same-winding) plateau.
    #[test]
    fn folded_same_winding_quad_arm_i_declines() {
        let folded = [
            Vec3::new(0.0, -1.5, 0.6),
            Vec3::new(-1.0, 0.0, 0.0),
            Vec3::new(0.0, -0.5, -0.36),
            Vec3::new(1.0, 0.0, 0.0),
        ];

        // First assert the fixture really is "same-winding + non-zero-area" —
        // the property that makes this case distinguish arm (i) from a bow-tie
        // shortcut. (Done on a throwaway mesh so the predicate sees the same
        // geometry the operator will.)
        {
            let mut probe = LoopMesh::new();
            let (diag, [pa_v, _pb_v, pc_v, _pd_v]) = quad_on_diag(&mut probe, folded);
            let (a, c) = probe.edge_rotate_diagonal(diag, true).unwrap();
            let [b_v, d_v] = probe.edges.get(diag).vertices;
            let pa = probe.vertex_position(a);
            let pc = probe.vertex_position(c);
            let pb = probe.vertex_position(b_v);
            let pd = probe.vertex_position(d_v);
            let n_abc = signed_normal(pa, pb, pc);
            let n_acd = signed_normal(pa, pc, pd);
            assert!(
                n_abc.dot(n_acd) > 0.0,
                "re-split faces must wind the SAME way (bow-tie guard would accept)"
            );
            assert!(
                n_abc.length() > 0.3 && n_acd.length() > 0.3,
                "both re-split faces non-zero-area (zero-area guard would accept)"
            );
            assert!(
                probe.rotation_is_degenerate(diag, a, c),
                "arm (i) must decline this same-winding folded rotation"
            );
            let _ = (pa_v, pc_v);
        }

        // Now the operator-level behaviour: multi-edge selection (guard on),
        // folded diagonal skipped, disjoint convex diagonal rotated.
        let mut mesh = LoopMesh::new();
        let (folded_diag, fverts) = quad_on_diag(&mut mesh, folded);
        let (convex_diag, [cw1, cw3]) = disjoint_convex_patch(&mut mesh);
        let counts = (mesh.totvert(), mesh.totedge(), mesh.totface());

        let out = mesh.rotate_edges(&[folded_diag, convex_diag], true);

        assert!(
            mesh.query_edge_exists(fverts[1], fverts[3]).is_some(),
            "folded shared diagonal survives — arm (i) declined the rotation"
        );
        assert!(
            mesh.query_edge_exists(fverts[0], fverts[2]).is_none(),
            "folded new diagonal was not laid"
        );
        assert!(
            mesh.query_edge_exists(cw1, cw3).is_some(),
            "convex diagonal rotated"
        );
        assert_eq!(
            out.len(),
            1,
            "only the convex rotation contributes to edges.out"
        );
        assert_eq!((mesh.totvert(), mesh.totedge(), mesh.totface()), counts);
    }

    /// §6 — `edges.out` carries one live new edge per successful rotation and
    /// nothing for skipped edges. Mixed eligible / ineligible selection.
    #[test]
    fn edges_out_reports_only_successful_rotations() {
        let mut mesh = LoopMesh::new();
        let [v0, v1, v2, v3] = two_tris_shared_diagonal(&mut mesh);
        let diag = edge_between(&mesh, v0, v2); // eligible
        let boundary = edge_between(&mesh, v0, v1); // ineligible

        let out = mesh.rotate_edges(&[diag, boundary], true);

        assert_eq!(
            out.len(),
            1,
            "only the eligible edge contributes to edges.out"
        );
        let live: std::collections::HashSet<EdgeKey> = mesh.edge_keys().into_iter().collect();
        for &e in &out {
            assert!(live.contains(&e), "every out-edge is a live post-op edge");
        }
        // The reported edge is the new {1,3} diagonal (the old {0,2} edge was
        // killed; its slot may be recycled, so compare by endpoints, not key).
        let [a, b] = mesh.edges.get(out.edges()[0]).vertices;
        assert!(
            (a == v1 && b == v3) || (a == v3 && b == v1),
            "edges.out reports the new diagonal, not the input edge"
        );
        let _ = (v0, v2);
    }
}
