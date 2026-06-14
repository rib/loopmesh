use crate::loopmesh::*;

/// Removal mode for [`LoopMesh::delete`].
///
/// Mirrors the seven-mode context taxonomy in
/// `radial-mesh-guides/operators/DELETE.md` §2. The context decides how the
/// seed set is expanded (downward to incident lower-dimensional elements,
/// upward to bounded higher-dimensional elements, or not at all) and which of
/// those elements are killed versus spared.
///
/// The vert-, edge-, and face-seeded modes are implemented today (§4 / §5.1 /
/// §5.2 / §6 / §7 / §7.5). The remaining mode (`tagged-only`) is documented in
/// the guide but not yet built; new variants slot in here without disturbing
/// the existing ones.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DeleteContext {
    /// Kill the seeded verts, and — entirely via the kernel vert-kill's
    /// *unconditional* upward cascade — every edge incident to a seeded vert and
    /// every face on those edges' radial cycles. Nothing incident survives. A
    /// non-seed vert the cascade leaves loose is **not** swept up: it stays as
    /// an isolated point. Seed is a vert slice. (§6.)
    Verts,
    /// Kill the seeded edges, and — via the kernel edge-kill's *unconditional*
    /// upward cascade — every face on a seed edge's radial cycle. Then a
    /// *conditional* downward sweep removes a **seed-edge endpoint** iff it is
    /// left loose (empty disk — no remaining incident edge). An endpoint a
    /// surviving edge still holds is kept (the §7.2 keep-alive), and a vert that
    /// was not a seed endpoint is never touched. Seed is an edge slice. (§7.)
    Edges,
    /// Like [`DeleteContext::Edges`], but **skip** the loose-endpoint sweep:
    /// kill the seeded edges and the faces on them, yet keep *every* vert — even
    /// a seed endpoint left with an empty disk stays as an isolated point. The
    /// single dropped sweep is the entire Edges-vs-EdgesFaces distinction. Seed
    /// is an edge slice. (§7.5.)
    EdgesFaces,
    /// Kill the seeded faces, plus any boundary edge / vert used by *no*
    /// surviving face — geometry the deleted region solely owned. Edges and
    /// verts still shared with an out-of-set face are spared. (§4.)
    Faces,
    /// Kill only the seeded face headers. The entire boundary wireframe
    /// (shared *and* solely-owned) survives. No downward cascade. (§5.1.)
    FacesOnly,
    /// Like [`DeleteContext::Faces`], but an edge that is *already* a mesh
    /// boundary edge at op time — radial cycle length 1 on the live mesh,
    /// before any deletion — is always kept-as-wire (and its verts rescued),
    /// even when the deleted region solely owns it. An interior edge that would
    /// only *become* boundary after deletion is not spared. (§5.2.)
    FacesKeepBoundary,
    /// Kill exactly the seeded elements of each type — every seeded vert, edge,
    /// and face — with **no expansion in either direction**: no conditional
    /// downward cascade (unlike [`DeleteContext::Faces`]), no loose-endpoint
    /// sweep (unlike [`DeleteContext::Edges`]), no reliance on the kernel's
    /// upward cascade to find more work (unlike [`DeleteContext::Verts`]).
    ///
    /// This is the only context whose seed is genuinely **mixed** (verts, edges
    /// and faces in one buffer — see [`MixedSeed`]). Closure is the *caller's*
    /// responsibility (§8.3): the sweep uses the plain kernel kill primitives,
    /// which still cascade **upward**, so an unclosed tag set (a tagged vert
    /// without its edges, or a tagged edge without its faces) will still lose
    /// those untagged higher-dimensional users to the kernel cascade.
    /// tagged-only declines to *add* to the kill set; it does not *protect*
    /// untagged users from the primitives' own cascade. Tagging a face without
    /// its boundary is the benign direction — face-kill never cascades
    /// downward, so the wire stays. (§8.)
    TaggedOnly,
}

/// Mixed-type seed for [`DeleteContext::TaggedOnly`].
///
/// Every other delete context is driven by a single-type seed (`&[FaceKey]` /
/// `&[VertexKey]` / `&[EdgeKey]`), but tagged-only (`DELETE.md` §8) reads all
/// three element types from one buffer: it kills exactly the verts, edges, and
/// faces it is handed, in the §3.1 face-then-edge-then-vert sweep order, with
/// no mark expansion. The three slices are borrowed for the duration of the
/// call; an empty slice simply contributes nothing of that type (a face-only
/// tagged seed, for instance, leaves `verts` and `edges` empty — §8.3 benign
/// direction).
#[derive(Clone, Copy)]
pub struct MixedSeed<'verts, 'edges, 'faces> {
    pub verts: &'verts [VertexKey],
    pub edges: &'edges [EdgeKey],
    pub faces: &'faces [FaceKey],
}

/// A seed buffer for [`LoopMesh::delete`].
///
/// The guide's `geom` buffer (`DELETE.md` §1) is a mixed verts/edges/faces set,
/// but every context except tagged-only reads only one dimension of it (§4.1 /
/// §6.1 / §7.1). The single-dimension contexts therefore take a typed slice
/// seed — `&[FaceKey]` for the face-seeded contexts, `&[VertexKey]` for the
/// verts context, `&[EdgeKey]` for the edge-seeded contexts — while
/// [`DeleteContext::TaggedOnly`] (§8) takes a [`MixedSeed`] carrying all three
/// types at once (§8.5). This trait lets the single `delete` entry accept any
/// of those seed shapes, dispatching to the right routine and guarding that the
/// context matches the seed's dimension.
pub trait DeleteSeed {
    #[doc(hidden)]
    fn delete_seeded(self, mesh: &mut LoopMesh, context: DeleteContext);
}

impl DeleteSeed for &[FaceKey] {
    fn delete_seeded(self, mesh: &mut LoopMesh, context: DeleteContext) {
        match context {
            DeleteContext::FacesOnly => mesh.delete_faces_only(self),
            DeleteContext::Faces => mesh.delete_faces(self, false),
            DeleteContext::FacesKeepBoundary => mesh.delete_faces(self, true),
            DeleteContext::Verts => {
                panic!("DeleteContext::Verts requires a vert seed, not faces")
            }
            DeleteContext::Edges | DeleteContext::EdgesFaces => {
                panic!("DeleteContext::Edges / EdgesFaces require an edge seed, not faces")
            }
            DeleteContext::TaggedOnly => {
                panic!("DeleteContext::TaggedOnly requires a MixedSeed, not a face slice")
            }
        }
    }
}

impl DeleteSeed for &[VertexKey] {
    fn delete_seeded(self, mesh: &mut LoopMesh, context: DeleteContext) {
        match context {
            DeleteContext::Verts => mesh.delete_verts(self),
            DeleteContext::TaggedOnly => {
                panic!("DeleteContext::TaggedOnly requires a MixedSeed, not a vert slice")
            }
            _ => panic!("a vert seed is only valid with DeleteContext::Verts"),
        }
    }
}

impl DeleteSeed for &[EdgeKey] {
    fn delete_seeded(self, mesh: &mut LoopMesh, context: DeleteContext) {
        match context {
            DeleteContext::Edges => mesh.delete_edges(self, true),
            DeleteContext::EdgesFaces => mesh.delete_edges(self, false),
            DeleteContext::TaggedOnly => {
                panic!("DeleteContext::TaggedOnly requires a MixedSeed, not an edge slice")
            }
            _ => panic!("an edge seed is only valid with DeleteContext::Edges / EdgesFaces"),
        }
    }
}

// Array-reference seeds (`&block`, `&[centre]`, …) unsize to the slice impls so
// callers need not spell out the `&seed[..]` coercion.
impl<const N: usize> DeleteSeed for &[FaceKey; N] {
    fn delete_seeded(self, mesh: &mut LoopMesh, context: DeleteContext) {
        (self.as_slice()).delete_seeded(mesh, context)
    }
}

impl<const N: usize> DeleteSeed for &[VertexKey; N] {
    fn delete_seeded(self, mesh: &mut LoopMesh, context: DeleteContext) {
        (self.as_slice()).delete_seeded(mesh, context)
    }
}

impl<const N: usize> DeleteSeed for &[EdgeKey; N] {
    fn delete_seeded(self, mesh: &mut LoopMesh, context: DeleteContext) {
        (self.as_slice()).delete_seeded(mesh, context)
    }
}

// `Vec`-reference seeds (the `face_vert_keys` / `face_edge_keys` results, …)
// likewise forward to the slice impls.
impl DeleteSeed for &Vec<FaceKey> {
    fn delete_seeded(self, mesh: &mut LoopMesh, context: DeleteContext) {
        (self.as_slice()).delete_seeded(mesh, context)
    }
}

impl DeleteSeed for &Vec<VertexKey> {
    fn delete_seeded(self, mesh: &mut LoopMesh, context: DeleteContext) {
        (self.as_slice()).delete_seeded(mesh, context)
    }
}

impl DeleteSeed for &Vec<EdgeKey> {
    fn delete_seeded(self, mesh: &mut LoopMesh, context: DeleteContext) {
        (self.as_slice()).delete_seeded(mesh, context)
    }
}

impl DeleteSeed for MixedSeed<'_, '_, '_> {
    fn delete_seeded(self, mesh: &mut LoopMesh, context: DeleteContext) {
        match context {
            DeleteContext::TaggedOnly => mesh.delete_tagged(self.verts, self.edges, self.faces),
            _ => panic!("a MixedSeed is only valid with DeleteContext::TaggedOnly"),
        }
    }
}

impl LoopMesh {
    /// Remove the seeded geometry according to `context`.
    ///
    /// Implements the face- and vert-seeded modes of
    /// `radial-mesh-guides/operators/DELETE.md`. The seed slice is typed to the
    /// dimension the chosen context reads (the guide's mixed `geom` buffer is
    /// read one-dimension-at-a-time per context — §4.1 / §6.1 / §7.1), so pass a
    /// `&[FaceKey]` for the face-seeded contexts, a `&[VertexKey]` for
    /// [`DeleteContext::Verts`], and a `&[EdgeKey]` for
    /// [`DeleteContext::Edges`] / [`DeleteContext::EdgesFaces`]. A mismatched
    /// seed/context pair panics (it is a caller bug, not an expected mesh state).
    ///
    /// - [`DeleteContext::Verts`] (§6) — kill the seeded verts and let the
    ///   kernel vert-kill's *unconditional* upward cascade take every incident
    ///   edge and every face on those edges. No mark phase, no downward sweep —
    ///   a non-seed vert left loose by the cascade survives as an isolated point
    ///   (the §6.2 loose-neighbour rule).
    /// - [`DeleteContext::Edges`] (§7) — kill the seeded edges and let the
    ///   kernel edge-kill's *unconditional* upward cascade take every face on
    ///   their radial cycles. Then a conditional loose-endpoint sweep removes a
    ///   seed-edge endpoint iff it is left with an empty disk; an endpoint a
    ///   surviving edge still holds is kept (the §7.2 keep-alive).
    /// - [`DeleteContext::EdgesFaces`] (§7.5) — like [`DeleteContext::Edges`]
    ///   but with the loose-endpoint sweep skipped, so every vert survives even
    ///   a seed endpoint left loose. That single dropped sweep is the whole
    ///   Edges-vs-EdgesFaces difference.
    /// - [`DeleteContext::FacesOnly`] (§5.1) — the non-cascading face sweep.
    ///   Every seeded face header and its loops are freed; every boundary edge
    ///   and vert is left standing as wire. No mark phase.
    /// - [`DeleteContext::Faces`] (§4) — the mark/unmark/sweep of §4.3. The
    ///   seeded faces die, and so does every boundary edge/vert the region
    ///   solely owned; geometry still shared with a surviving face is kept (the
    ///   §4.2 shared-boundary keep-alive).
    /// - [`DeleteContext::FacesKeepBoundary`] (§5.2) — like
    ///   [`DeleteContext::Faces`], but an edge already on the mesh boundary
    ///   (radial cycle length 1 on the live mesh) is kept-as-wire even when the
    ///   region solely owns it; an edge that would only *become* boundary after
    ///   deletion is not spared.
    ///
    /// Delete creates no replacement geometry, so there is no customdata
    /// interpolation: survivors keep their attributes byte-for-byte and killed
    /// elements' attributes are discarded with them.
    /// The mixed-seed [`DeleteContext::TaggedOnly`] mode (§8) takes a
    /// [`MixedSeed`] rather than a slice; pass
    /// `MixedSeed { verts, edges, faces }` with each field a key slice (any may
    /// be empty). It kills exactly the tagged elements of each type in the §3.1
    /// face->edge->vert order with no mark expansion of its own — closure is the
    /// caller's responsibility (§8.3).
    pub fn delete<S: DeleteSeed>(&mut self, seed: S, context: DeleteContext) {
        seed.delete_seeded(self, context);
    }

    /// §6 — kill the seeded verts and let each vert-kill's upward cascade do the
    /// rest.
    ///
    /// `vertex_kill` (DELETE.md §3) is the fully-cascading primitive: it first
    /// kills every edge on the vert's disk — each of which first kills every
    /// face on its radial cycle — then frees the vert. So a single vert sweep
    /// subsumes the §3.1 face→edge→vert ordering *inside* each kill, and there
    /// is no separate face or edge sweep (§6.3).
    ///
    /// No mark phase is needed: the cascade is unconditional, and the verts
    /// context never sweeps downward to non-seed verts the cascade leaves loose
    /// (§6.2). The sweep is over the seed slice directly, re-testing liveness so
    /// a duplicate seed entry — or a seed vert an earlier kill already freed
    /// (verts are never freed by *another* vert's cascade, but a defensive
    /// re-test keeps the loop idempotent) — is skipped.
    fn delete_verts(&mut self, verts: &[VertexKey]) {
        for &vert in verts {
            if self.vertices.try_get(vert).is_some() {
                self.vertex_kill(vert);
            }
        }
    }

    /// §7 / §7.5 — kill the seeded edges, their cascaded faces, and (when
    /// `sweep_loose_endpoints` is set) any seed endpoint left loose.
    ///
    /// `edge_kill` (DELETE.md §3) is the upward-cascading primitive: it first
    /// kills every face on the edge's radial cycle — including a non-manifold
    /// radial ≥ 3 — then unlinks the edge from its endpoints' disks and frees
    /// it. So the single edge sweep over the seed subsumes all the face kills;
    /// no separate face sweep is written (§7.3).
    ///
    /// The endpoints are first flushed into a transient mark (the only downward
    /// marking this context does, and unconditional). After the edge sweep, the
    /// conditional part runs: a marked endpoint is killed iff its disk is now
    /// empty (`is_loose()` in guide terms — no remaining incident edge). The
    /// `is_loose()` guard is the entire behavioural difference from the verts
    /// context (§7.3): it spares a seed endpoint a surviving non-seed edge still
    /// anchors (the §7.2 keep-alive), and scopes the sweep to seed endpoints
    /// only — a third-party vert is never reconsidered.
    ///
    /// `DeleteContext::EdgesFaces` (§7.5) passes `sweep_loose_endpoints = false`,
    /// dropping the loose-endpoint sweep entirely so every vert survives — even
    /// a seed endpoint left with an empty disk. That single skipped sweep is the
    /// whole Edges-vs-EdgesFaces distinction.
    fn delete_edges(&mut self, edges: &[EdgeKey], sweep_loose_endpoints: bool) {
        let doomed_edge = self.push_edge_op_flag_layer();
        let seed_endpoint = self.push_vert_op_flag_layer();

        const DOOM: u8 = 0;
        const ENDPOINT: u8 = 0;

        // Phase 1 — mark the seed edges and flush the mark down to both
        // endpoints. The endpoint mark scopes the later loose sweep to seed
        // endpoints only (§7.2 / §7.3).
        for &edge in edges {
            if self.edges.try_get(edge).is_none() {
                continue;
            }
            self.edge_op_flag_set(doomed_edge, edge, DOOM);
            let [v0, v1] = self.edge_vertices(edge);
            self.vert_op_flag_set(seed_endpoint, v0, ENDPOINT);
            self.vert_op_flag_set(seed_endpoint, v1, ENDPOINT);
        }

        // Phase 2a — sweep the edges. Each edge-kill cascades upward to the
        // faces on its radial cycle, so this single sweep performs all the face
        // kills implicitly. The mesh-wide lending iterator re-tests liveness on
        // each step, so it is kill-safe against faces an earlier kill freed.
        let mut edge_iter = MeshEdges::default();
        while let Some(edge) = edge_iter.next(self) {
            if self.edge_op_flag_test(doomed_edge, edge, DOOM) {
                self.edge_kill(edge);
            }
        }

        // Phase 2b — loose-endpoint sweep (skipped under EdgesFaces, §7.5). Kill
        // a marked seed endpoint iff its disk is now empty; a marked endpoint a
        // surviving edge still anchors is left standing (§7.2 keep-alive).
        if sweep_loose_endpoints {
            let mut vert_iter = MeshVerts::default();
            while let Some(vert) = vert_iter.next(self) {
                if self.vert_op_flag_test(seed_endpoint, vert, ENDPOINT)
                    && self.vertex_disk_iter(vert).next().is_none()
                {
                    self.vertex_kill(vert);
                }
            }
        }

        self.pop_vert_op_flag_layer(seed_endpoint);
        self.pop_edge_op_flag_layer(doomed_edge);
    }

    /// §5.1 — kill exactly the seeded faces, touch nothing else.
    ///
    /// `face_kill` (DELETE.md §3) is the non-cascading primitive: it unlinks
    /// the face's loop cycle from its boundary edges' radial cycles and frees
    /// the face header and loops, leaving every 1- and 0-cell alone. So the
    /// whole operator is one face sweep over the seed.
    fn delete_faces_only(&mut self, faces: &[FaceKey]) {
        for &face in faces {
            // Re-test liveness: a face may appear twice in the seed, or have
            // been freed by an earlier kill in this same call.
            if self.faces.try_get(face).is_some() {
                self.face_kill(face);
            }
        }
    }

    /// §4 — kill the seeded faces plus the boundary they solely owned.
    ///
    /// Three mesh-wide mark passes (§4.3) compute the closed kill set, then the
    /// face→edge→vert sweep (§3.1) frees it bottom-of-the-cascade-first so each
    /// kill in the sweep is a leaf kill.
    ///
    /// When `keep_boundary` is set this is the §5.2 `faces-keep-edges` mode: an
    /// extra unmark pass, gated on the edge's *live* radial cycle length being 1
    /// (already a mesh-boundary edge at op time), runs before phase 1c so a kept
    /// edge's verts ride along and survive when 1c flushes the keep down to them.
    /// An interior edge that would only *become* boundary after deletion is
    /// radial-2 right now, fails the test, and still dies with the region.
    fn delete_faces(&mut self, faces: &[FaceKey], keep_boundary: bool) {
        let seeded = self.push_face_op_flag_layer();
        let doomed_vert = self.push_vert_op_flag_layer();
        let doomed_edge = self.push_edge_op_flag_layer();
        let doomed_face = self.push_face_op_flag_layer();

        const SEED: u8 = 0;
        const DOOM: u8 = 0;

        // The seed faces themselves are always killed; mark them directly. A
        // separate `seeded` bit drives the §4.3 over-mark / un-doom passes,
        // since those test face membership in the input set.
        for &face in faces {
            if self.faces.try_get(face).is_some() {
                self.face_op_flag_set(seeded, face, SEED);
                self.face_op_flag_set(doomed_face, face, DOOM);
            }
        }

        // Phase 1a — over-mark: tentatively doom every boundary vert/edge of
        // every seeded face. Optimistically assume the whole boundary goes.
        let mut face_iter = MeshFaces::default();
        while let Some(face) = face_iter.next(self) {
            if !self.face_op_flag_test(seeded, face, SEED) {
                continue;
            }
            let mut loop_iter = FaceLoops::new(self, face);
            while let Some(corner) = loop_iter.next(self) {
                let vert = self.loop_vertex(corner);
                let edge = self.loop_edge(corner);
                self.vert_op_flag_set(doomed_vert, vert, DOOM);
                self.edge_op_flag_set(doomed_edge, edge, DOOM);
            }
        }

        // Phase 1b — un-doom survivors: any boundary vert/edge a NON-seeded
        // face also touches must be kept (the §4.2 shared-boundary keep-alive).
        let mut face_iter = MeshFaces::default();
        while let Some(face) = face_iter.next(self) {
            if self.face_op_flag_test(seeded, face, SEED) {
                continue;
            }
            let mut loop_iter = FaceLoops::new(self, face);
            while let Some(corner) = loop_iter.next(self) {
                let vert = self.loop_vertex(corner);
                let edge = self.loop_edge(corner);
                self.vert_op_flag_clear(doomed_vert, vert, DOOM);
                self.edge_op_flag_clear(doomed_edge, edge, DOOM);
            }
        }

        // §5.2 carve-out — un-doom edges already on the mesh boundary. An edge
        // whose *live* radial cycle (measured on the mesh as it stands, before
        // any deletion) has length 1 is pre-existing boundary/wire fringe; keep
        // it as wire even when the region solely owns it. Runs before 1c so the
        // kept edge's verts are rescued when 1c flushes surviving-edge keeps
        // down to their endpoints. Interior edges (radial-2) that would only
        // *become* boundary after the sweep stay marked and die.
        if keep_boundary {
            let mut edge_iter = MeshEdges::default();
            while let Some(edge) = edge_iter.next(self) {
                if !self.edge_op_flag_test(doomed_edge, edge, DOOM) {
                    continue;
                }
                if self.edge_radial_iter(edge).count() == 1 {
                    self.edge_op_flag_clear(doomed_edge, edge, DOOM);
                }
            }
        }

        // Phase 1c — protect verts of surviving edges: a vert on any surviving
        // (un-doomed) edge is kept, even if every face it touched is doomed.
        // This makes the vert rule "dies iff every incident *edge* dies" rather
        // than "iff every incident *face* dies" — the wire-edge keep-alive of
        // §4.4. Must run after 1b so it sees the post-keep-alive edge marks.
        let mut edge_iter = MeshEdges::default();
        while let Some(edge) = edge_iter.next(self) {
            if self.edge_op_flag_test(doomed_edge, edge, DOOM) {
                continue;
            }
            let [v0, v1] = self.edge_vertices(edge);
            self.vert_op_flag_clear(doomed_vert, v0, DOOM);
            self.vert_op_flag_clear(doomed_vert, v1, DOOM);
        }

        // Phase 2 — sweep faces, then edges, then verts (the §3.1 invariant).
        // The mesh-wide lending iterators are cursor-based and re-test liveness
        // on each step, so they are kill-safe against elements an earlier kill
        // already freed via the kernel cascade.
        let mut face_iter = MeshFaces::default();
        while let Some(face) = face_iter.next(self) {
            if self.face_op_flag_test(doomed_face, face, DOOM) {
                self.face_kill(face);
            }
        }
        let mut edge_iter = MeshEdges::default();
        while let Some(edge) = edge_iter.next(self) {
            if self.edge_op_flag_test(doomed_edge, edge, DOOM) {
                self.edge_kill(edge);
            }
        }
        let mut vert_iter = MeshVerts::default();
        while let Some(vert) = vert_iter.next(self) {
            if self.vert_op_flag_test(doomed_vert, vert, DOOM) {
                self.vertex_kill(vert);
            }
        }

        self.pop_face_op_flag_layer(doomed_face);
        self.pop_edge_op_flag_layer(doomed_edge);
        self.pop_vert_op_flag_layer(doomed_vert);
        self.pop_face_op_flag_layer(seeded);
    }

    /// §8 — kill exactly the tagged elements of each type, no mark expansion.
    ///
    /// There is no mark *computation* (§8.3): the seed already names every
    /// element to kill. The only structure is the mark-then-sweep, marking each
    /// element by its own type with no flushing down to sub-elements and no
    /// marking up to super-elements, then sweeping in the §3.1
    /// face->edge->vert order so that — *for a closed tag set* — each kill in
    /// the sweep is a leaf kill (every doomed face already gone by the time the
    /// edge sweep runs, every doomed edge gone by the vert sweep).
    ///
    /// Closure is the *caller's* responsibility. The sweep uses the plain
    /// kernel kill primitives (`face_kill`, `edge_kill`, `vertex_kill`), which
    /// always cascade **upward** (DELETE.md §3). So an unclosed tag set that
    /// tags an edge without its faces, or a vert without its edges, still loses
    /// those untagged higher-dimensional users to the kernel cascade during the
    /// later edge/vert sweep — tagged-only declines to *add* to the kill set but
    /// does not *protect* untagged users from the primitives' own cascade (§8.3).
    /// Tagging a face without its boundary is benign: face-kill never cascades
    /// downward, so the untagged wire survives.
    ///
    /// There is no loose-vert sweep and no `is_loose` guard (contrast the edges
    /// context's §7.3): a tagged vert is killed iff tagged, full stop.
    fn delete_tagged(&mut self, verts: &[VertexKey], edges: &[EdgeKey], faces: &[FaceKey]) {
        let doomed_face = self.push_face_op_flag_layer();
        let doomed_edge = self.push_edge_op_flag_layer();
        let doomed_vert = self.push_vert_op_flag_layer();

        const DOOM: u8 = 0;

        // Phase 1 — mark exactly the seeded elements, each by its own type.
        for &face in faces {
            if self.faces.try_get(face).is_some() {
                self.face_op_flag_set(doomed_face, face, DOOM);
            }
        }
        for &edge in edges {
            if self.edges.try_get(edge).is_some() {
                self.edge_op_flag_set(doomed_edge, edge, DOOM);
            }
        }
        for &vert in verts {
            if self.vertices.try_get(vert).is_some() {
                self.vert_op_flag_set(doomed_vert, vert, DOOM);
            }
        }

        // Phase 2 — sweep faces, then edges, then verts (the §3.1 invariant).
        // The mesh-wide lending iterators re-test liveness on each step, so they
        // are kill-safe against elements an earlier kill's upward cascade freed
        // (the unclosed-set case — §8.3).
        let mut face_iter = MeshFaces::default();
        while let Some(face) = face_iter.next(self) {
            if self.face_op_flag_test(doomed_face, face, DOOM) {
                self.face_kill(face);
            }
        }
        let mut edge_iter = MeshEdges::default();
        while let Some(edge) = edge_iter.next(self) {
            if self.edge_op_flag_test(doomed_edge, edge, DOOM) {
                self.edge_kill(edge);
            }
        }
        let mut vert_iter = MeshVerts::default();
        while let Some(vert) = vert_iter.next(self) {
            if self.vert_op_flag_test(doomed_vert, vert, DOOM) {
                self.vertex_kill(vert);
            }
        }

        self.pop_vert_op_flag_layer(doomed_vert);
        self.pop_edge_op_flag_layer(doomed_edge);
        self.pop_face_op_flag_layer(doomed_face);
    }
}

#[cfg(test)]
mod tests {
    use glam::Vec3;

    use crate::loopmesh::*;

    /// Build an `n × n` grid of quad faces (so `(n+1) × (n+1)` verts). Returns
    /// the faces in row-major order: face `(r, c)` is at index `r * n + c`.
    fn build_quad_grid(mesh: &mut LoopMesh, n: usize) -> Vec<FaceKey> {
        let side = n + 1;
        let mut verts = Vec::with_capacity(side * side);
        for r in 0..side {
            for c in 0..side {
                verts.push(mesh.create_vertex(Vec3::new(c as f32, r as f32, 0.0)));
            }
        }
        let vert_at = |r: usize, c: usize| verts[r * side + c];

        let mut faces = Vec::with_capacity(n * n);
        for r in 0..n {
            for c in 0..n {
                let v00 = vert_at(r, c);
                let v01 = vert_at(r, c + 1);
                let v11 = vert_at(r + 1, c + 1);
                let v10 = vert_at(r + 1, c);
                let ring = [v00, v01, v11, v10];
                let mut edges = [EdgeKey::NULL; 4];
                for i in 0..4 {
                    let a = ring[i];
                    let b = ring[(i + 1) % 4];
                    edges[i] = mesh
                        .query_edge_exists(a, b)
                        .unwrap_or_else(|| mesh.create_edge([a, b], EdgeCreateFlags::NONE));
                }
                faces.push(mesh.face_create(&ring, &edges, FaceCreateFlags::NONE));
            }
        }
        faces
    }

    /// The interior 2×2 block of a 5×5 face grid, matching DELETE.md §7's
    /// worked example. Faces `(1,1) (1,2) (2,1) (2,2)` → indices 6, 7, 11, 12.
    fn interior_2x2_block(faces: &[FaceKey]) -> [FaceKey; 4] {
        let n = 5;
        [
            faces[1 * n + 1],
            faces[1 * n + 2],
            faces[2 * n + 1],
            faces[2 * n + 2],
        ]
    }

    #[test]
    fn faces_context_kills_solely_owned_geometry() {
        let mut mesh = LoopMesh::new();
        let faces = build_quad_grid(&mut mesh, 5);
        let (v0, e0, f0) = (mesh.totvert(), mesh.totedge(), mesh.totface());

        let block = interior_2x2_block(&faces);
        mesh.delete(&block, DeleteContext::Faces);

        // DELETE.md §7: −4 F, −4 E (the four interior cross-edges), −1 V (the
        // single centre vert where the cross-edges met).
        assert_eq!(f0 - mesh.totface(), 4, "faces");
        assert_eq!(e0 - mesh.totedge(), 4, "edges");
        assert_eq!(v0 - mesh.totvert(), 1, "verts");

        // The deleted block faces are gone.
        for &f in &block {
            assert!(mesh.faces.try_get(f).is_none());
        }
    }

    #[test]
    fn faces_context_keeps_outer_ring() {
        let mut mesh = LoopMesh::new();
        let faces = build_quad_grid(&mut mesh, 5);

        // Snapshot the ring edges/verts: every edge bounding a deleted block
        // face but shared with a surrounding (out-of-block) grid face.
        let block = interior_2x2_block(&faces);
        let block_set: std::collections::HashSet<FaceKey> = block.iter().copied().collect();

        let mut ring_edges = std::collections::HashSet::new();
        let mut ring_verts = std::collections::HashSet::new();
        for &f in &block {
            for e in mesh.face_edge_keys(f) {
                // A ring (shared) edge has a face on its radial cycle that is
                // not in the block.
                let mut iter = RadialLoops::new(&mesh, e);
                let mut shared = false;
                while let Some(l) = iter.next(&mesh) {
                    if !block_set.contains(&mesh.loop_face(l)) {
                        shared = true;
                    }
                }
                if shared {
                    ring_edges.insert(e);
                    let [a, b] = mesh.edge_vertices(e);
                    ring_verts.insert(a);
                    ring_verts.insert(b);
                }
            }
        }
        assert_eq!(ring_edges.len(), 8, "block perimeter is 8 shared edges");

        mesh.delete(&block, DeleteContext::Faces);

        for e in ring_edges {
            assert!(
                mesh.edges.try_get(e).is_some(),
                "shared ring edge must survive (§4.2 keep-alive)"
            );
        }
        for v in ring_verts {
            assert!(
                mesh.vertices.try_get(v).is_some(),
                "ring vert held by a surviving edge must survive (§4.3 1c)"
            );
        }
    }

    #[test]
    fn faces_only_context_keeps_whole_wireframe() {
        let mut mesh = LoopMesh::new();
        let faces = build_quad_grid(&mut mesh, 5);
        let (v0, e0, f0) = (mesh.totvert(), mesh.totedge(), mesh.totface());

        let block = interior_2x2_block(&faces);
        mesh.delete(&block, DeleteContext::FacesOnly);

        // DELETE.md §7 contrast pin: −4 F, 0 E, 0 V. The interior cross-edges
        // and centre vert that die under Faces all survive as wire here.
        assert_eq!(f0 - mesh.totface(), 4, "faces");
        assert_eq!(e0 - mesh.totedge(), 0, "edges");
        assert_eq!(v0 - mesh.totvert(), 0, "verts");

        for &f in &block {
            assert!(mesh.faces.try_get(f).is_none());
        }
    }

    #[test]
    fn faces_context_single_quad_drops_owned_boundary() {
        // A lone quad with no neighbours: every boundary edge and vert is
        // solely owned, so Faces removes the whole quad (§7, n=1, b=0).
        let mut mesh = LoopMesh::new();
        let faces = build_quad_grid(&mut mesh, 1);

        mesh.delete(&faces, DeleteContext::Faces);

        assert_eq!(mesh.totface(), 0);
        assert_eq!(mesh.totedge(), 0);
        assert_eq!(mesh.totvert(), 0);
    }

    #[test]
    fn faces_only_single_quad_leaves_wire_loop() {
        // The §5.1 worked sentence: faces-only on a single quad leaves a
        // four-edge wire loop with four corner verts.
        let mut mesh = LoopMesh::new();
        let faces = build_quad_grid(&mut mesh, 1);

        mesh.delete(&faces, DeleteContext::FacesOnly);

        assert_eq!(mesh.totface(), 0);
        assert_eq!(mesh.totedge(), 4);
        assert_eq!(mesh.totvert(), 4);
    }

    /// Two quads sharing one edge: 6 verts, 7 edges (4 + 4 − 1 shared), 2
    /// faces. The 6 perimeter edges are each radial-1 (mesh boundary); the 1
    /// shared edge is radial-2 (interior). Returns the two faces.
    fn build_two_quads_shared_edge(mesh: &mut LoopMesh) -> [FaceKey; 2] {
        let v0 = mesh.create_vertex(Vec3::new(0.0, 0.0, 0.0));
        let v1 = mesh.create_vertex(Vec3::new(1.0, 0.0, 0.0));
        let v2 = mesh.create_vertex(Vec3::new(2.0, 0.0, 0.0));
        let v3 = mesh.create_vertex(Vec3::new(0.0, 1.0, 0.0));
        let v4 = mesh.create_vertex(Vec3::new(1.0, 1.0, 0.0));
        let v5 = mesh.create_vertex(Vec3::new(2.0, 1.0, 0.0));

        let mut make = |mesh: &mut LoopMesh, ring: [VertexKey; 4]| {
            let mut edges = [EdgeKey::NULL; 4];
            for i in 0..4 {
                let a = ring[i];
                let b = ring[(i + 1) % 4];
                edges[i] = mesh
                    .query_edge_exists(a, b)
                    .unwrap_or_else(|| mesh.create_edge([a, b], EdgeCreateFlags::NONE));
            }
            mesh.face_create(&ring, &edges, FaceCreateFlags::NONE)
        };

        let left = make(mesh, [v0, v1, v4, v3]);
        let right = make(mesh, [v1, v2, v5, v4]);
        [left, right]
    }

    #[test]
    fn keep_boundary_keeps_owned_fringe_kills_interior() {
        // §5.2 / §7: deleting both quads. Every perimeter edge is radial-1
        // (already boundary) and is kept-as-wire with its verts; the single
        // radial-2 shared edge would only *become* boundary, so it dies.
        // Surviving counts: 6 V, 6 E, 0 F.
        let mut mesh = LoopMesh::new();
        let block = build_two_quads_shared_edge(&mut mesh);

        // The shared edge (v1–v4) is radial-2 before deletion.
        let shared = mesh.query_edge_exists(
            mesh.face_vert_keys(block[0])[1],
            mesh.face_vert_keys(block[0])[2],
        );
        let shared = shared.expect("shared edge exists");
        assert_eq!(
            mesh.edge_radial_iter(shared).count(),
            2,
            "shared is interior"
        );

        mesh.delete(&block, DeleteContext::FacesKeepBoundary);

        assert_eq!(mesh.totface(), 0, "both faces gone");
        assert_eq!(mesh.totedge(), 6, "6 boundary edges kept as wire");
        assert_eq!(mesh.totvert(), 6, "all verts rescued by kept wire edges");
        assert!(
            mesh.edges.try_get(shared).is_none(),
            "radial-2 shared edge dies (only would-be boundary)"
        );
    }

    #[test]
    fn keep_boundary_contrast_with_plain_faces() {
        // Same seed under plain Faces: the region solely owns the entire two-
        // quad surface, so every edge and vert dies along with the faces.
        let mut mesh = LoopMesh::new();
        let block = build_two_quads_shared_edge(&mut mesh);

        mesh.delete(&block, DeleteContext::Faces);

        assert_eq!(mesh.totface(), 0);
        assert_eq!(
            mesh.totedge(),
            0,
            "plain Faces drops the owned boundary too"
        );
        assert_eq!(mesh.totvert(), 0);
    }

    #[test]
    fn keep_boundary_keeps_single_quad_as_wire() {
        // A lone quad: all four edges are radial-1 boundary, so keep-boundary
        // leaves the full wire loop and corners standing where plain Faces
        // (see `faces_context_single_quad_drops_owned_boundary`) wipes them.
        let mut mesh = LoopMesh::new();
        let faces = build_quad_grid(&mut mesh, 1);

        mesh.delete(&faces, DeleteContext::FacesKeepBoundary);

        assert_eq!(mesh.totface(), 0);
        assert_eq!(mesh.totedge(), 4);
        assert_eq!(mesh.totvert(), 4);
    }

    #[test]
    fn keep_boundary_interior_block_matches_plain_faces() {
        // §5.2: the interior 2×2 block's at-risk edges are its four cross-edges,
        // all radial-2 (both faces seeded) — they would only *become* boundary,
        // so keep-boundary spares none of them and matches plain Faces exactly:
        // −4 F, −4 E, −1 V (the centre vert). The block's outer ring is shared
        // (radial-2 with a surviving grid face) and kept by the §4.2 keep-alive.
        let mut mesh = LoopMesh::new();
        let faces = build_quad_grid(&mut mesh, 5);
        let (v0, e0, f0) = (mesh.totvert(), mesh.totedge(), mesh.totface());

        let block = interior_2x2_block(&faces);
        mesh.delete(&block, DeleteContext::FacesKeepBoundary);

        assert_eq!(f0 - mesh.totface(), 4, "faces");
        assert_eq!(e0 - mesh.totedge(), 4, "edges (radial-2 cross-edges die)");
        assert_eq!(v0 - mesh.totvert(), 1, "verts (centre)");
    }

    #[test]
    fn empty_seed_is_noop() {
        let mut mesh = LoopMesh::new();
        let faces = build_quad_grid(&mut mesh, 3);
        let (v0, e0, f0) = (mesh.totvert(), mesh.totedge(), mesh.totface());

        mesh.delete(&[] as &[FaceKey], DeleteContext::Faces);
        mesh.delete(&[] as &[FaceKey], DeleteContext::FacesOnly);
        mesh.delete(&[] as &[VertexKey], DeleteContext::Verts);

        assert_eq!(
            (v0, e0, f0),
            (mesh.totvert(), mesh.totedge(), mesh.totface())
        );
    }

    #[test]
    fn verts_interior_manifold_vert_punches_hole() {
        // §6.4 interior-manifold-vert case (scaffold `grid_3x3` centre vert 4 →
        // here the centre of a 2×2 grid: the lone 4-valence interior vert). The
        // four meeting faces, the four spoke edges, and the seed vert die; the
        // eight outer verts and the outer ring edges (which never touched the
        // seed) survive. Net: −1 V, −4 E, −4 F.
        let mut mesh = LoopMesh::new();
        let faces = build_quad_grid(&mut mesh, 2);
        let (v0, e0, f0) = (mesh.totvert(), mesh.totedge(), mesh.totface());

        // In a 2×2 grid (3×3 verts), the single interior vert is the v11 corner
        // of face (0,0) — the corner diagonally opposite that face's origin.
        let centre = mesh.face_vert_keys(faces[0])[2];
        mesh.delete(&[centre], DeleteContext::Verts);

        assert_eq!(f0 - mesh.totface(), 4, "all four faces die");
        assert_eq!(e0 - mesh.totedge(), 4, "four spoke edges die");
        assert_eq!(v0 - mesh.totvert(), 1, "only the seed vert dies");
        assert!(mesh.vertices.try_get(centre).is_none(), "seed vert gone");
    }

    #[test]
    fn verts_single_corner_kills_face_keeps_far_edges() {
        // §6.4 "deleting one corner of a face": seed one corner of a lone quad.
        // The two edges incident to that corner die (taking the face with
        // them), plus the seed vert. The face's two far edges and three other
        // corners survive (the far edges as wire). Net: −1 V, −2 E, −1 F →
        // remaining 3 V, 2 E, 0 F.
        let mut mesh = LoopMesh::new();
        let faces = build_quad_grid(&mut mesh, 1);
        let corner = mesh.face_vert_keys(faces[0])[0];

        mesh.delete(&[corner], DeleteContext::Verts);

        assert_eq!(mesh.totface(), 0, "the face dies through its corner");
        assert_eq!(mesh.totvert(), 3, "three far corners survive");
        assert_eq!(mesh.totedge(), 2, "two far edges survive as wire");
        assert!(mesh.vertices.try_get(corner).is_none());
    }

    #[test]
    fn verts_wire_anchor_drops_wire_keeps_far_endpoint_loose() {
        // §6.4 wire-anchor case (scaffold quad + wire {1,4}, seed vert 1). A
        // lone quad with a wire edge hanging off one corner; seed that corner.
        // The quad's two incident edges + face die, the wire edge dies (on the
        // seed's disk), and the seed vert dies — but the wire's *far* endpoint
        // is not seeded and survives as a loose point (§6.2).
        let mut mesh = LoopMesh::new();
        let faces = build_quad_grid(&mut mesh, 1);
        let corner = mesh.face_vert_keys(faces[0])[0];

        // A wire edge off `corner` to a fresh far vert.
        let far = mesh.create_vertex(Vec3::new(-1.0, -1.0, 0.0));
        mesh.create_edge([corner, far], EdgeCreateFlags::NONE);

        mesh.delete(&[corner], DeleteContext::Verts);

        assert_eq!(mesh.totface(), 0, "quad face dies");
        assert!(mesh.vertices.try_get(corner).is_none(), "seed vert gone");
        assert!(
            mesh.vertices.try_get(far).is_some(),
            "wire far endpoint survives loose (§6.2)"
        );
        // Survivors: the three far quad corners + the loose wire endpoint = 4 V;
        // the two far quad edges (wire) = 2 E; the wire edge itself is gone.
        assert_eq!(mesh.totvert(), 4, "3 quad corners + 1 loose far endpoint");
        assert_eq!(mesh.totedge(), 2, "two far quad edges survive, wire dies");
    }

    /// Two triangles sharing exactly one vert (a bowtie): tris `(a,b,hub)` and
    /// `(hub,c,d)`. Returns the shared hub vert and the four non-shared corners.
    fn build_bowtie(mesh: &mut LoopMesh) -> (VertexKey, [VertexKey; 4]) {
        let a = mesh.create_vertex(Vec3::new(-2.0, 1.0, 0.0));
        let b = mesh.create_vertex(Vec3::new(-2.0, -1.0, 0.0));
        let hub = mesh.create_vertex(Vec3::new(0.0, 0.0, 0.0));
        let c = mesh.create_vertex(Vec3::new(2.0, 1.0, 0.0));
        let d = mesh.create_vertex(Vec3::new(2.0, -1.0, 0.0));

        let mut make = |mesh: &mut LoopMesh, ring: [VertexKey; 3]| {
            let mut edges = [EdgeKey::NULL; 3];
            for i in 0..3 {
                let p = ring[i];
                let q = ring[(i + 1) % 3];
                edges[i] = mesh
                    .query_edge_exists(p, q)
                    .unwrap_or_else(|| mesh.create_edge([p, q], EdgeCreateFlags::NONE));
            }
            mesh.face_create(&ring, &edges, FaceCreateFlags::NONE);
        };
        make(mesh, [a, b, hub]);
        make(mesh, [hub, c, d]);
        (hub, [a, b, c, d])
    }

    #[test]
    fn verts_nonmanifold_hub_kills_both_fans_keeps_corners_loose() {
        // §6.4 non-manifold-hub case (scaffold `bowtie` shared corner vert 2).
        // Seeding the shared hub of a bowtie kills *both* triangles. The four
        // edges incident to the hub (two per triangle) die; each triangle's
        // *opposite* edge (a–b and c–d) does not touch the hub, so the two of
        // them survive as wire. The four non-shared corners survive as the
        // endpoints of those surviving wire edges (none were seeded — §6.2).
        // Net: −1 V (hub), −4 E (hub edges), −2 F → remaining 4 V, 2 E, 0 F.
        let mut mesh = LoopMesh::new();
        let (hub, corners) = build_bowtie(&mut mesh);

        mesh.delete(&[hub], DeleteContext::Verts);

        assert_eq!(mesh.totface(), 0, "both triangles die");
        assert_eq!(mesh.totedge(), 2, "the two hub-free opposite edges survive");
        assert_eq!(mesh.totvert(), 4, "four non-shared corners survive loose");
        assert!(mesh.vertices.try_get(hub).is_none(), "hub gone");
        for c in corners {
            assert!(
                mesh.vertices.try_get(c).is_some(),
                "non-shared corner survives as loose point (§6.2)"
            );
        }
    }

    /// An `n`-page "book": `n` triangles sharing one spine edge `(spine0,
    /// spine1)`. Each page `i` has its own apex vert. Returns the spine edge and
    /// its two endpoints.
    fn build_nonmanifold_book(mesh: &mut LoopMesh, pages: usize) -> (EdgeKey, [VertexKey; 2]) {
        let spine0 = mesh.create_vertex(Vec3::new(0.0, 0.0, 0.0));
        let spine1 = mesh.create_vertex(Vec3::new(0.0, 1.0, 0.0));
        let spine = mesh.create_edge([spine0, spine1], EdgeCreateFlags::NONE);

        for i in 0..pages {
            let apex = mesh.create_vertex(Vec3::new(1.0 + i as f32, 0.5, 0.0));
            let ring = [spine0, spine1, apex];
            let mut edges = [EdgeKey::NULL; 3];
            for j in 0..3 {
                let a = ring[j];
                let b = ring[(j + 1) % 3];
                edges[j] = mesh
                    .query_edge_exists(a, b)
                    .unwrap_or_else(|| mesh.create_edge([a, b], EdgeCreateFlags::NONE));
            }
            mesh.face_create(&ring, &edges, FaceCreateFlags::NONE);
        }
        (spine, [spine0, spine1])
    }

    #[test]
    fn edges_interior_manifold_edge_keeps_endpoints() {
        // §7.4 interior-manifold-edge case (scaffold `grid_3x3`). A radial-2 seed
        // edge shared by two faces: both faces die and the edge dies, but each
        // endpoint still has three other edges, so both survive. Net: −2 F,
        // −1 E, 0 V.
        let mut mesh = LoopMesh::new();
        let faces = build_quad_grid(&mut mesh, 3);
        let (v0, e0, f0) = (mesh.totvert(), mesh.totedge(), mesh.totface());

        // A vertical interior edge between faces (1,1) and (1,1)'s right
        // neighbour: the shared edge of faces[4] (centre) and faces[5].
        let centre = faces[1 * 3 + 1];
        let right = faces[1 * 3 + 2];
        let centre_verts = mesh.face_vert_keys(centre);
        let right_verts = mesh.face_vert_keys(right);
        // The shared edge connects the two verts common to both faces.
        let shared = centre_verts
            .iter()
            .flat_map(|&a| centre_verts.iter().map(move |&b| (a, b)))
            .filter_map(|(a, b)| mesh.query_edge_exists(a, b))
            .find(|&e| {
                let [x, y] = mesh.edge_vertices(e);
                right_verts.contains(&x) && right_verts.contains(&y)
            })
            .expect("shared interior edge exists");
        let [ea, eb] = mesh.edge_vertices(shared);

        mesh.delete(&[shared], DeleteContext::Edges);

        assert_eq!(f0 - mesh.totface(), 2, "both radial faces die");
        assert_eq!(e0 - mesh.totedge(), 1, "only the seed edge dies");
        assert_eq!(v0 - mesh.totvert(), 0, "both endpoints kept (§7.2)");
        assert!(mesh.vertices.try_get(ea).is_some());
        assert!(mesh.vertices.try_get(eb).is_some());
    }

    #[test]
    fn edges_isolated_wire_sweeps_both_endpoints() {
        // §7.4 isolated-wire case (scaffold `wire_edge_pair`). A lone wire edge
        // whose endpoints have no other edge: deleting it sweeps up both
        // endpoints, so the whole 2-vert/1-edge component vanishes.
        let mut mesh = LoopMesh::new();
        let a = mesh.create_vertex(Vec3::new(0.0, 0.0, 0.0));
        let b = mesh.create_vertex(Vec3::new(1.0, 0.0, 0.0));
        let wire = mesh.create_edge([a, b], EdgeCreateFlags::NONE);

        mesh.delete(&[wire], DeleteContext::Edges);

        assert_eq!(mesh.totedge(), 0, "wire edge gone");
        assert_eq!(mesh.totvert(), 0, "both loose endpoints swept (§7.4)");
    }

    #[test]
    fn edges_nonmanifold_spine_kills_pages_keeps_wire() {
        // §7.4 non-manifold-edge case (scaffold `nonmanifold_book(3)`). Seeding
        // the spine edge of a 3-page book kills the edge and *all three*
        // triangles (the exhaustive radial cascade). Each spine endpoint still
        // anchors three page edges, so both survive with those page edges left
        // as wire; no page edge or apex vert is a seed endpoint, so the downward
        // sweep never reaches them. Remaining: 5 V (2 spine + 3 apex), 6 E (the
        // page edges), 0 F.
        let mut mesh = LoopMesh::new();
        let (spine, [s0, s1]) = build_nonmanifold_book(&mut mesh, 3);

        mesh.delete(&[spine], DeleteContext::Edges);

        assert_eq!(
            mesh.totface(),
            0,
            "all three pages die (exhaustive cascade)"
        );
        assert_eq!(mesh.totedge(), 6, "six page edges survive as wire");
        assert_eq!(mesh.totvert(), 5, "both spine ends + three apexes survive");
        assert!(mesh.vertices.try_get(s0).is_some(), "spine end kept (§7.2)");
        assert!(mesh.vertices.try_get(s1).is_some(), "spine end kept (§7.2)");
        assert!(mesh.edges.try_get(spine).is_none(), "spine edge gone");
    }

    #[test]
    fn edges_all_quad_edges_erases_face() {
        // §7.4 all-edges-of-a-face case (scaffold `quad`). Seeding all four
        // edges of a lone quad kills the face and all four edges; every corner
        // then has an empty disk and is swept (it is a seed endpoint). The quad
        // is erased entirely.
        let mut mesh = LoopMesh::new();
        let faces = build_quad_grid(&mut mesh, 1);
        let seed = mesh.face_edge_keys(faces[0]);

        mesh.delete(&seed, DeleteContext::Edges);

        assert_eq!(mesh.totface(), 0);
        assert_eq!(mesh.totedge(), 0);
        assert_eq!(mesh.totvert(), 0, "all four loose corners swept");
    }

    #[test]
    fn edges_faces_isolated_wire_keeps_both_endpoints() {
        // §7.5 contrast pin (scaffold `wire_edge_pair`). The *same* isolated
        // wire seed as `edges_isolated_wire_sweeps_both_endpoints`, but under
        // EdgesFaces the loose-endpoint sweep is skipped, so both endpoints
        // survive as isolated points even with empty disks. The single dropped
        // sweep is the entire Edges-vs-EdgesFaces difference.
        let mut mesh = LoopMesh::new();
        let a = mesh.create_vertex(Vec3::new(0.0, 0.0, 0.0));
        let b = mesh.create_vertex(Vec3::new(1.0, 0.0, 0.0));
        let wire = mesh.create_edge([a, b], EdgeCreateFlags::NONE);

        mesh.delete(&[wire], DeleteContext::EdgesFaces);

        assert_eq!(mesh.totedge(), 0, "wire edge gone");
        assert_eq!(mesh.totvert(), 2, "both endpoints kept loose (§7.5)");
        assert!(mesh.vertices.try_get(a).is_some());
        assert!(mesh.vertices.try_get(b).is_some());
    }

    #[test]
    fn edges_empty_seed_is_noop() {
        let mut mesh = LoopMesh::new();
        let faces = build_quad_grid(&mut mesh, 3);
        let (v0, e0, f0) = (mesh.totvert(), mesh.totedge(), mesh.totface());

        mesh.delete(&[] as &[EdgeKey], DeleteContext::Edges);
        mesh.delete(&[] as &[EdgeKey], DeleteContext::EdgesFaces);
        let _ = faces;

        assert_eq!(
            (v0, e0, f0),
            (mesh.totvert(), mesh.totedge(), mesh.totface())
        );
    }

    #[test]
    fn verts_all_corners_erases_face() {
        // §6.4 "deleting every vert of a face": seeding all four corners of a
        // lone quad removes the face, all edges, and all verts — no surviving
        // loose geometry, since every would-be-loose vert was itself seeded.
        let mut mesh = LoopMesh::new();
        let faces = build_quad_grid(&mut mesh, 1);
        let corners = mesh.face_vert_keys(faces[0]);

        mesh.delete(&corners, DeleteContext::Verts);

        assert_eq!(mesh.totface(), 0);
        assert_eq!(mesh.totedge(), 0);
        assert_eq!(mesh.totvert(), 0, "mesh fully erased");
    }

    #[test]
    fn tagged_only_closed_set_removes_exactly_tagged() {
        // §8.4 closed-set case (scaffold `grid_3x3`): tag the four block faces,
        // the four spoke (disk) edges of the centre vert, and the centre vert
        // itself — the full closure. tagged-only kills exactly that set with no
        // expansion. Net: −4 F, −4 E, −1 V, leaving the eight outer ring edges
        // and verts untouched. (9 V → 8 V, 12 E → 8 E, 4 F → 0 F.)
        let mut mesh = LoopMesh::new();
        let faces = build_quad_grid(&mut mesh, 2);

        // The lone 4-valence interior vert of a 2×2 grid, and its disk edges.
        let centre = mesh.face_vert_keys(faces[0])[2];
        let spokes: Vec<EdgeKey> = mesh.vertex_disk_iter(centre).collect();
        assert_eq!(spokes.len(), 4, "centre vert has a 4-edge disk");

        let outer_ring_edges: Vec<EdgeKey> = {
            let mut all: std::collections::HashSet<EdgeKey> = std::collections::HashSet::new();
            let mut it = MeshEdges::default();
            while let Some(e) = it.next(&mesh) {
                all.insert(e);
            }
            for s in &spokes {
                all.remove(s);
            }
            all.into_iter().collect()
        };
        assert_eq!(outer_ring_edges.len(), 8, "eight outer ring edges");

        mesh.delete(
            MixedSeed {
                verts: &[centre],
                edges: &spokes,
                faces: &faces,
            },
            DeleteContext::TaggedOnly,
        );

        assert_eq!(mesh.totface(), 0, "all four tagged faces gone");
        assert_eq!(mesh.totedge(), 8, "only the four tagged spokes removed");
        assert_eq!(mesh.totvert(), 8, "only the tagged centre vert removed");
        assert!(mesh.vertices.try_get(centre).is_none(), "centre vert gone");
        for e in outer_ring_edges {
            assert!(
                mesh.edges.try_get(e).is_some(),
                "untagged outer ring edge must survive (§8.2 closed set)"
            );
        }
    }

    #[test]
    fn tagged_only_face_only_keeps_boundary_wireframe() {
        // §8.3 benign-direction case (scaffold lone `quad`): tag a single face,
        // with neither its edges nor its verts in the seed. face-kill does not
        // cascade downward (§3), so the face header goes and the entire
        // four-edge wire loop with its four corner verts stays standing.
        let mut mesh = LoopMesh::new();
        let faces = build_quad_grid(&mut mesh, 1);

        mesh.delete(
            MixedSeed {
                verts: &[],
                edges: &[],
                faces: &faces,
            },
            DeleteContext::TaggedOnly,
        );

        assert_eq!(mesh.totface(), 0, "tagged face gone");
        assert_eq!(mesh.totedge(), 4, "untagged wire loop kept (§8.3 benign)");
        assert_eq!(mesh.totvert(), 4, "untagged corners kept");
    }

    #[test]
    fn tagged_only_unclosed_vert_seed_cascades_through_kernel() {
        // §8.4 unclosed contrast (scaffold `grid_3x3`): tag only the centre vert
        // and *one* block face, leaving the spokes and the other three faces
        // untagged. The one tagged face is killed first (no cascade), then the
        // centre-vert sweep `vertex_kill`s the centre — and that cascades up its
        // disk, removing all four spokes and all four faces anyway. tagged-only
        // does not protect untagged users from the kernel cascade (§8.3): the
        // outcome matches the closed set, −4 F, −4 E, −1 V.
        let mut mesh = LoopMesh::new();
        let faces = build_quad_grid(&mut mesh, 2);
        let (v0, e0, f0) = (mesh.totvert(), mesh.totedge(), mesh.totface());

        let centre = mesh.face_vert_keys(faces[0])[2];

        mesh.delete(
            MixedSeed {
                verts: &[centre],
                edges: &[],
                faces: &faces[..1],
            },
            DeleteContext::TaggedOnly,
        );

        assert_eq!(f0 - mesh.totface(), 4, "all four faces gone via cascade");
        assert_eq!(e0 - mesh.totedge(), 4, "all four spokes gone via cascade");
        assert_eq!(
            v0 - mesh.totvert(),
            1,
            "only the tagged centre vert removed"
        );
        assert!(mesh.vertices.try_get(centre).is_none(), "centre vert gone");
    }
}
