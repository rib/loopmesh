use std::collections::HashSet;

use glam::DVec3;

use crate::loopmesh::*;

/// The `geom.out` of [`LoopMesh::convex_hull`]: every vertex, edge, and face
/// that lies **on the built hull surface** — the on-hull input vertices, the
/// hull edges, and the materialised hull faces (CONVEX_HULL.md §1, §4.3).
///
/// The `geom_interior.out` / `geom_unused.out` slots carry the full vertex,
/// edge, and face classification (CONVEX_HULL.md §4, §4.1, §4.2, §7) when mixed
/// edge/face input is supplied via [`LoopMesh::convex_hull_with_input`]; the
/// verts-only [`LoopMesh::convex_hull`] populates only the vertex half (its
/// input edge/face sets are empty). The holes slot (CONVEX_HULL.md §5) is
/// populated by the `use_existing_faces` path.
#[derive(Clone, Debug, Default)]
pub struct ConvexHullResult {
    verts: Vec<VertexKey>,
    edges: Vec<EdgeKey>,
    faces: Vec<FaceKey>,
    interior_verts: Vec<VertexKey>,
    interior_edges: Vec<EdgeKey>,
    interior_faces: Vec<FaceKey>,
    unused_verts: Vec<VertexKey>,
    unused_edges: Vec<EdgeKey>,
    unused_faces: Vec<FaceKey>,
    hole_edges: Vec<EdgeKey>,
    hole_faces: Vec<FaceKey>,
}

impl ConvexHullResult {
    /// On-hull vertices — the extremal subset of the input cloud. These are
    /// existing input verts (the operator never invents a point, CONVEX_HULL.md
    /// §2), so their positions equal the input positions exactly.
    pub fn verts(&self) -> &[VertexKey] {
        &self.verts
    }

    /// Hull edges — every undirected vertex pair bounding a hull triangle,
    /// created with the no-double discipline or reused if already present
    /// (CONVEX_HULL.md §6 step 1).
    pub fn edges(&self) -> &[EdgeKey] {
        &self.edges
    }

    /// Hull faces — one outward-wound triangle per fanned hull facet
    /// (CONVEX_HULL.md §3.1, §6 steps 2–4).
    pub fn faces(&self) -> &[FaceKey] {
        &self.faces
    }

    /// The whole `geom.out` slot as a single mixed element set
    /// (CONVEX_HULL.md §1 `geom.out`).
    pub fn geom_out(&self) -> Vec<GeomKey> {
        let mut out = Vec::with_capacity(self.verts.len() + self.edges.len() + self.faces.len());
        out.extend(self.verts.iter().copied().map(GeomKey::from));
        out.extend(self.edges.iter().copied().map(GeomKey::from));
        out.extend(self.faces.iter().copied().map(GeomKey::from));
        out
    }

    /// Interior (buried) input vertices — every input vert that is *not* a hull
    /// corner: the culled coplanar-interior point, the collinear intermediate
    /// points, a coincident duplicate's key, and any 3D buried point
    /// (CONVEX_HULL.md §4, §4.1).
    pub fn interior_verts(&self) -> &[VertexKey] {
        &self.interior_verts
    }

    /// Interior (buried) input edges — every input edge whose endpoint pair is
    /// *not* a hull edge: a chord cutting through the polytope's interior
    /// (CONVEX_HULL.md §4.2).
    pub fn interior_edges(&self) -> &[EdgeKey] {
        &self.interior_edges
    }

    /// Interior input faces — every input face except those *reused* as a hull
    /// facet under `use_existing_faces` (CONVEX_HULL.md §4.2, §5). Reuse clears
    /// the interior flag at reuse time, so a reused facet later un-holed as a
    /// patch border (and thus absent from [`Self::hole_faces`]) is still not
    /// interior.
    pub fn interior_faces(&self) -> &[FaceKey] {
        &self.interior_faces
    }

    /// The `geom_interior.out` slot as a single mixed element set: the interior
    /// verts, edges, and faces (CONVEX_HULL.md §1 `geom_interior.out`, §4.2).
    pub fn geom_interior_out(&self) -> Vec<GeomKey> {
        let mut out = Vec::with_capacity(
            self.interior_verts.len() + self.interior_edges.len() + self.interior_faces.len(),
        );
        out.extend(self.interior_verts.iter().copied().map(GeomKey::from));
        out.extend(self.interior_edges.iter().copied().map(GeomKey::from));
        out.extend(self.interior_faces.iter().copied().map(GeomKey::from));
        out
    }

    /// Interior vertices that are *unused* — safe to delete because every
    /// incident edge and face is itself an input element (CONVEX_HULL.md §7).
    pub fn unused_verts(&self) -> &[VertexKey] {
        &self.unused_verts
    }

    /// Interior edges that are *unused* — every incident face is an input face
    /// (an interior edge with no faces is trivially unused) (CONVEX_HULL.md §7).
    pub fn unused_edges(&self) -> &[EdgeKey] {
        &self.unused_edges
    }

    /// Interior faces that are *unused* — every interior face is unused, a face
    /// being a top-level element nothing else depends on (CONVEX_HULL.md §7).
    pub fn unused_faces(&self) -> &[FaceKey] {
        &self.unused_faces
    }

    /// The `geom_unused.out` slot as a single mixed element set: the unused
    /// verts, edges, and faces (CONVEX_HULL.md §1 `geom_unused.out`, §7).
    pub fn geom_unused_out(&self) -> Vec<GeomKey> {
        let mut out = Vec::with_capacity(
            self.unused_verts.len() + self.unused_edges.len() + self.unused_faces.len(),
        );
        out.extend(self.unused_verts.iter().copied().map(GeomKey::from));
        out.extend(self.unused_edges.iter().copied().map(GeomKey::from));
        out.extend(self.unused_faces.iter().copied().map(GeomKey::from));
        out
    }

    /// Reused on-hull faces tagged as **holes** under `use_existing_faces`: a
    /// pre-existing input face whose footprint coincides with a hull triangle
    /// and all of whose edges lie on the hull, minus any face that touches a
    /// boundary edge (un-holed by the cleanup pass — CONVEX_HULL.md §5). Empty
    /// when `use_existing_faces` is clear.
    pub fn hole_faces(&self) -> &[FaceKey] {
        &self.hole_faces
    }

    /// Edges all of whose (at least one) incident faces are holes — the edges
    /// of the reused patch, tagged by the cleanup pass so the holes slot
    /// reports the seam's edges as well as its faces (CONVEX_HULL.md §5).
    pub fn hole_edges(&self) -> &[EdgeKey] {
        &self.hole_edges
    }

    /// The `geom_holes.out` slot as a single mixed element set: the reused
    /// patch faces (minus border faces) plus the patch's all-hole edges
    /// (CONVEX_HULL.md §1 `geom_holes.out`, §5). Empty when `use_existing_faces`
    /// is clear.
    pub fn geom_holes_out(&self) -> Vec<GeomKey> {
        let mut out = Vec::with_capacity(self.hole_edges.len() + self.hole_faces.len());
        out.extend(self.hole_edges.iter().copied().map(GeomKey::from));
        out.extend(self.hole_faces.iter().copied().map(GeomKey::from));
        out
    }

    pub fn is_empty(&self) -> bool {
        self.verts.is_empty() && self.edges.is_empty() && self.faces.is_empty()
    }
}

impl LoopMesh {
    /// Wrap the positions of an `input` vertex set in their 3D convex hull and
    /// materialise that hull as outward-wound triangle faces
    /// (CONVEX_HULL.md §1).
    ///
    /// Only the **vertices** drive the hull cloud (CONVEX_HULL.md §1.1); every
    /// input vertex contributes its position. The hull *geometry* — which
    /// subset is extremal and how the extremal points connect into outward-wound
    /// facets — is delegated to the in-tree [`Quickhull`] provider, which hands
    /// back a neutral index-based [`HullTopology`]. This operator *consumes* that
    /// topology and performs the radial-edge surgery on top of it: it fans each
    /// facet into triangles (CONVEX_HULL.md §3.1), then, per triangle, creates or
    /// reuses the three boundary edges with a no-double discipline and finds or
    /// creates the outward-wound face (CONVEX_HULL.md §6 steps 1–2).
    ///
    /// No vertex is ever created — every hull vertex is an existing input vertex,
    /// mapped straight back from its cloud index (CONVEX_HULL.md §2). The
    /// outward winding the provider guarantees (CONVEX_HULL.md §3.2) is carried
    /// verbatim into each created face, so the hull surface has coherent outward
    /// normals.
    ///
    /// **Scope:** any vertex cloud (full-3D, coplanar, collinear, or coincident)
    /// with `use_existing_faces = false` and no input edges or faces. The result
    /// reports `geom.out` (the built on-hull surface, CONVEX_HULL.md §4.3) and
    /// the **vertex** half of the `geom_interior.out` / `geom_unused.out`
    /// classification (CONVEX_HULL.md §4, §4.1, §7): every input vert not on the
    /// hull is interior, and an interior vert is unused when all its incident
    /// elements are themselves input (verts-only ⇒ a buried vert is loose ⇒
    /// unused). A fresh hull face bordering a pre-existing non-input neighbour
    /// inherits its customdata verbatim (the example-face seed, §6.1).
    /// For mixed input (input edges/faces routed into the classification) see
    /// [`Self::convex_hull_with_input`]; for the `use_existing_faces` reuse path
    /// (§5) and the `geom_holes.out` slot see [`Self::convex_hull_with_faces`].
    ///
    /// Degenerate clouds (CONVEX_HULL.md §8) are handled, not errors:
    /// - **Coplanar** input yields a single planar facet, fanned into a flat
    ///   outward-wound triangle patch; culled interior points are absent from
    ///   the provider's on-hull set and so from `geom.out`.
    /// - **Collinear** input yields a 1D hull — two extreme endpoints, no
    ///   facet. No faces are built; the two endpoints are reported as on-hull
    ///   verts in `geom.out`.
    /// - **Coincident** duplicate input verts are deduplicated by the provider
    ///   before the hull runs, so a non-representative duplicate never appears
    ///   in the on-hull set.
    ///
    /// Returns an empty result and makes no changes when given **fewer than
    /// three** input vertices (the count guard, CONVEX_HULL.md §1.2), or when
    /// the provider returns no hull of any dimension (fewer than two distinct
    /// positions).
    pub fn convex_hull(&mut self, input: &[VertexKey]) -> ConvexHullResult {
        self.convex_hull_with_input(input, &[], &[], false)
    }

    /// Mixed-input convex hull (CONVEX_HULL.md §1): the `verts` cloud drives the
    /// hull geometry exactly as [`Self::convex_hull`], while `input_faces` are
    /// the pre-existing input faces the classification + reuse passes inspect.
    /// Only the verts contribute hull points (§1.1); `input_faces` (and the
    /// edges they imply) participate only in the reuse / holes logic.
    ///
    /// `use_existing_faces` (CONVEX_HULL.md §5): when set, a hull triangle whose
    /// footprint is entirely covered by an input face — the face spans the three
    /// triangle verts and *every* edge of that face is a hull edge — is **not**
    /// materialised; the input face is reused in place, un-marked interior, and
    /// tagged a *hole*. A cleanup pass then un-holes any hole face touching a
    /// boundary edge (an edge with fewer than two faces) and holes every edge
    /// all of whose faces are holes, so `geom_holes.out` reports the reused
    /// patch's faces and edges. When clear, every hull triangle is built
    /// unconditionally (stacking on any coincident input face, which can yield
    /// non-manifold edges — §2) and `geom_holes.out` comes back empty.
    ///
    /// `input_faces` are routed into `geom_interior.out` / `geom_unused.out`
    /// (§4.2, §7) exactly as [`Self::convex_hull_with_input`] with an empty
    /// input-edge set — every input face is interior unless reused as a hole,
    /// and an interior face is always unused. A fresh hull face bordering a
    /// pre-existing non-input neighbour inherits its customdata verbatim (the
    /// example-face seed, §6.1).
    pub fn convex_hull_with_faces(
        &mut self,
        verts: &[VertexKey],
        input_faces: &[FaceKey],
        use_existing_faces: bool,
    ) -> ConvexHullResult {
        self.convex_hull_with_input(verts, &[], input_faces, use_existing_faces)
    }

    /// Mixed-input convex hull with the full §4.2 / §7 edge + face
    /// classification: the `verts` cloud drives the hull geometry (§1.1), while
    /// `input_edges` and `input_faces` are the pre-existing input elements the
    /// classification passes inspect. Only the verts contribute hull points; the
    /// input edges/faces participate only in classification, reuse, and the
    /// unused guard.
    ///
    /// **§4.2 interior edges/faces.** After the hull triangles are known, an
    /// input **edge** whose endpoint pair is not a hull edge (a buried chord) is
    /// tagged interior, and every input **face** is tagged interior — except a
    /// face reused as a hole under `use_existing_faces` (§5), which is un-marked.
    ///
    /// **§7 unused guard (general).** An interior **vertex** is unused iff every
    /// incident edge is an input edge *and* every incident face is an input
    /// face; an interior **edge** is unused iff every incident face is an input
    /// face (no faces ⇒ trivially unused); every interior **face** is unused. A
    /// non-input edge or face leaning on an element keeps it out of
    /// `geom_unused.out` so a downstream delete can't disturb unselected
    /// geometry.
    ///
    /// `use_existing_faces` (§5) behaves exactly as
    /// [`Self::convex_hull_with_faces`]. A fresh hull face bordering a
    /// pre-existing non-input neighbour inherits its customdata verbatim (the
    /// example-face seed, §6.1).
    pub fn convex_hull_with_input(
        &mut self,
        verts: &[VertexKey],
        input_edges: &[EdgeKey],
        input_faces: &[FaceKey],
        use_existing_faces: bool,
    ) -> ConvexHullResult {
        self.convex_hull_dispatch(
            verts,
            input_edges,
            input_faces,
            use_existing_faces,
            &Quickhull,
        )
    }

    /// [`Self::convex_hull_with_input`] with the hull *primitive* supplied by the
    /// caller instead of the in-tree [`Quickhull`] default.
    ///
    /// Only the hull geometry is delegated: the provider is handed the input
    /// verts' positions as a point cloud (index `i` is `verts[i]`, §1.1) and
    /// hands back a neutral index-based [`HullTopology`]. Everything downstream —
    /// the fan (§3.1), the no-double edge/face surgery (§6), the reuse/holes pass
    /// (§5), and the interior/unused classification (§4, §7) — is the operator's
    /// own and consumes the returned topology **verbatim**: each facet's loop is
    /// fanned from *its own* first vertex, in *its own* order, with *its own*
    /// winding. Facets are neither sorted, rotated, re-merged, nor re-wound, so
    /// a provider's facet order and loop rotation are observable in the built
    /// mesh's diagonals and normals.
    ///
    /// The provider must uphold the [`ConvexHullProvider`] contract: face indices
    /// address the *input* slice, faces are CCW as seen from outside, and
    /// `vertices` is the on-hull index subset in strictly ascending order — and,
    /// with facets present, exactly the set of indices its own faces reference
    /// (the operator emits `geom.out`'s verts in that order, and treats indices
    /// absent from it as interior). Violations corrupt the output slots silently,
    /// so the contract is audited on entry by
    /// [`assert_provider_contract`], compiled only under `debug_assertions` —
    /// debug builds name the broken clause; release builds pay nothing.
    ///
    /// This is the injection point behind the prerequisite's "Pluggability" rule:
    /// the boundary is coarse (one call per whole hull), so `&dyn` dispatch costs
    /// a single virtual call per operator run, while the three default entry
    /// points reach the same body monomorphised on the ZST [`Quickhull`] and pay
    /// nothing.
    pub fn convex_hull_with_provider(
        &mut self,
        verts: &[VertexKey],
        input_edges: &[EdgeKey],
        input_faces: &[FaceKey],
        use_existing_faces: bool,
        provider: &dyn ConvexHullProvider,
    ) -> ConvexHullResult {
        self.convex_hull_dispatch(verts, input_edges, input_faces, use_existing_faces, provider)
    }

    /// The provider-generic front half: guard, cloud, hull. Kept deliberately
    /// small — it is the *only* code monomorphised per provider, so the ZST
    /// default path inlines to a direct `Quickhull` call while the operator's
    /// (large) surgery body is compiled exactly once and shared by both paths.
    fn convex_hull_dispatch<P: ConvexHullProvider + ?Sized>(
        &mut self,
        verts: &[VertexKey],
        input_edges: &[EdgeKey],
        input_faces: &[FaceKey],
        use_existing_faces: bool,
        provider: &P,
    ) -> ConvexHullResult {
        // §1.2 — minimum-input guard: a count, not a non-degeneracy test.
        if verts.len() < 3 {
            return ConvexHullResult::default();
        }

        // §1.1 — only the verts drive the cloud. Map each input vertex's
        // position into the provider's f64 cloud; the cloud index is the input
        // slot, so an on-hull index maps straight back to `verts[index]` with no
        // remapping and no positional drift (§2).
        let points: Vec<DVec3> = verts
            .iter()
            .map(|&v| self.vertex_position(v).as_dvec3())
            .collect();

        let Some(topology) = provider.convex_hull(&points) else {
            return ConvexHullResult::default();
        };

        self.convex_hull_from_topology(
            verts,
            input_edges,
            input_faces,
            use_existing_faces,
            &topology,
        )
    }

    /// The operator proper: everything downstream of the hull primitive. Takes
    /// the provider's topology as given (see [`Self::convex_hull_with_provider`]).
    fn convex_hull_from_topology(
        &mut self,
        verts: &[VertexKey],
        input_edges: &[EdgeKey],
        input_faces: &[FaceKey],
        use_existing_faces: bool,
        topology: &HullTopology,
    ) -> ConvexHullResult {
        #[cfg(debug_assertions)]
        assert_provider_contract(verts.len(), topology);

        let mut result = ConvexHullResult::default();

        // §3.1 — fan every facet into outward-wound triangles up front, keyed
        // back to the mesh's vertex handles. The reuse test (§5) and the surgery
        // (§6) both walk this flat triangle list.
        let mut triangles: Vec<[VertexKey; 3]> = Vec::new();
        for facet in &topology.faces {
            let corners = &facet.vertices;
            if corners.len() < 3 {
                continue;
            }
            let apex = verts[corners[0]];
            for window in corners[1..].windows(2) {
                triangles.push([apex, verts[window[0]], verts[window[1]]]);
            }
        }

        // §4.2 — a hull edge is any undirected vertex pair bounding some hull
        // triangle. The reuse test needs this set to decide whether *every* edge
        // of a candidate input face is a hull edge.
        let mut hull_edge_pairs: HashSet<[VertexKey; 2]> = HashSet::new();
        for tri in &triangles {
            for (a, b) in [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])] {
                hull_edge_pairs.insert(undirected(a, b));
            }
        }

        let input_face_set: HashSet<FaceKey> = input_faces.iter().copied().collect();

        let mut on_hull_verts: HashSet<VertexKey> = HashSet::new();
        let mut on_hull_edges: HashSet<EdgeKey> = HashSet::new();
        let mut seen_faces: HashSet<FaceKey> = HashSet::new();
        let mut hole_faces: HashSet<FaceKey> = HashSet::new();

        for tri in &triangles {
            let [a, b, c] = *tri;

            // §5 — under `use_existing_faces`, try to reuse an input face whose
            // footprint coincides with this triangle and all of whose edges are
            // hull edges. A reused face is skipped (no new face built), tagged a
            // hole, and its verts/edges still flagged on-hull (§6.2).
            if use_existing_faces {
                if let Some(face) =
                    self.find_reusable_face(a, b, c, &input_face_set, &hull_edge_pairs)
                {
                    hole_faces.insert(face);
                    for vert in [a, b, c] {
                        on_hull_verts.insert(vert);
                    }
                    // §6.2 — flag the reused patch's hull edges on-hull so
                    // `geom.out` stays complete across built and reused faces.
                    for edge in self.face_edge_keys(face) {
                        on_hull_edges.insert(edge);
                    }
                    continue;
                }
            }

            // §3.2 — the facet's CCW order is the face's winding, so the built
            // face's stored normal points outward. `face_create_verts` with
            // NO_DUPLICATE performs both the no-double edge creation (§6 step 1)
            // and the face find-or-create (§6 step 2) in one call.
            //
            // §6.1 — find the per-face example *before* the build, while the
            // three boundary edges may still carry only their pre-existing
            // neighbour: a freshly-minted hull edge has no other face, so the
            // example is always found across a *reused* edge that already
            // bordered existing geometry (an input face is eligible — §6.1).
            let example = self.find_example_face(a, b, c, &seen_faces, &hole_faces);

            // §6 step 2 — find-or-create. A NO_DUPLICATE build that *reuses* an
            // existing face (its footprint already spanned by a face) keeps that
            // face's own customdata live and is not re-seeded (§6 step 2); only a
            // genuinely fresh face takes the example-face seed (§6 step 3). The
            // face-count delta distinguishes the two.
            let faces_before = self.face_count();
            let face = self.face_create_verts(&[a, b, c], FaceCreateFlags::NO_DUPLICATE);
            let freshly_built = self.face_count() != faces_before;

            // §4.3 / §6 step 4 — flag the face, its edges, and its verts
            // on-hull as they are materialised.
            if seen_faces.insert(face) {
                result.faces.push(face);
                // §6 step 3 — seed the fresh face's customdata by **two distinct
                // verbatim copies on two domains** (never an interpolation): the
                // per-face layers from the single example neighbour (§6.1), and
                // the per-loop layers by the separate `BM_face_copy_shared`-equiv
                // vertex-matched write-once pass over the shared-edge neighbours
                // (§6 step 3). Only a genuinely fresh face takes the seed (a
                // reused face's customdata is already live, §6 step 2).
                if freshly_built {
                    if let Some(example) = example {
                        self.copy_example_face_data(face, example);
                    }
                    self.copy_shared_corner_data(face);
                }
            }
            for vert in [a, b, c] {
                on_hull_verts.insert(vert);
            }
            for edge in self.face_edge_keys(face) {
                on_hull_edges.insert(edge);
            }
        }

        // The set of faces *reused* as hull facets (§5 step 1) — every face
        // `find_reusable_face` matched, captured *before* the cleanup pass runs.
        // A reused face is reclassified out of interior at reuse time (§4.2:
        // "reused-and-reclassified"); the §5 cleanup may later un-hole a border
        // face, but that only clears its *hole* flag, never re-marks it interior.
        // So the §4.2 interior-face pass must exclude the whole reused set, not
        // just the post-cleanup surviving holes.
        let reused_faces: HashSet<FaceKey> = hole_faces.iter().copied().collect();

        // §5 — hole cleanup pass. A hole face touching a boundary edge (fewer
        // than two incident faces) is on the patch's border, not a true interior
        // hole, so it is un-holed. Then every edge all of whose faces are holes
        // is itself tagged a hole, so the patch's edges are reported too.
        if use_existing_faces && !hole_faces.is_empty() {
            self.cleanup_holes(&mut hole_faces, &mut result);
        }

        // §2 / §8 — the on-hull vertex set is the extremal subset of the input,
        // emitted in the provider's ascending-index order so the slot is stable.
        //
        // The emission filter (`on_hull_verts.contains`) coincides with the
        // provider's own vertex set for a full-3D hull: every vertex the provider
        // reports is a corner of some materialised triangle, so the two agree.
        // The degenerate paths populate `result.verts` directly because no facet
        // bounds the on-hull verts there: a 1D collinear hull (§8 collinear
        // bullet) builds no faces yet its two extreme endpoints are still on-hull,
        // and a coplanar flat hull with on-hull verts is covered by the facet
        // path. Coincident duplicates never reach the on-hull set — the provider
        // deduplicates coincident positions before the hull runs, so a
        // non-representative duplicate is simply absent from `topology.vertices`
        // (§8 coincident bullet) and lands in the interior/unused slots instead.
        //
        // `no_facets` distinguishes a genuinely degenerate (collinear) hull from
        // one whose facets were all reused as holes (§5): in the reuse case the
        // facets exist and bound their verts, so the `on_hull_verts.contains`
        // filter is correct and we must *not* fall into the no-facets seam.
        let no_facets = triangles.is_empty();
        for &index in &topology.vertices {
            let vert = verts[index];
            if no_facets || on_hull_verts.contains(&vert) {
                result.verts.push(vert);
                // No-op for the facet path; load-bearing for the no-faces
                // (collinear) path — seeds on-hull so the later classification
                // pass excludes these endpoints from interior.
                on_hull_verts.insert(vert);
            }
        }
        // The on-hull edges are accumulated through a HashSet (drain order is
        // nondeterministic); sort so this slot is as stable as the sibling
        // verts/interior/unused slots.
        result.edges = on_hull_edges.into_iter().collect();
        result.edges.sort_unstable_by_key(ElementKey::as_usize);

        let input_edge_set: HashSet<EdgeKey> = input_edges.iter().copied().collect();

        // §4 / §4.1 — classify input vertices. Every input vert is optimistically
        // interior; a vert that turned out to be a hull corner has its tag
        // cleared. The remaining interior verts are the buried ones: the culled
        // coplanar-interior point, collinear intermediates, a coincident
        // duplicate's key, and any 3D buried point.
        let mut seen_interior_verts: HashSet<VertexKey> = HashSet::new();
        for &vert in verts {
            if on_hull_verts.contains(&vert) || !seen_interior_verts.insert(vert) {
                continue;
            }
            result.interior_verts.push(vert);

            // §7 — an interior vertex is unused iff *every* incident edge is an
            // input edge AND *every* incident face is an input face. A non-input
            // edge or face leaning on it keeps it out of the unused slot.
            if self.vertex_incidence_all_input(vert, &input_edge_set, &input_face_set) {
                result.unused_verts.push(vert);
            }
        }

        // §4.2 — an input edge whose endpoint pair is not a hull edge is a buried
        // chord, tagged interior. §7 — that interior edge is unused iff every
        // incident face is an input face (no faces ⇒ trivially unused).
        let mut seen_interior_edges: HashSet<EdgeKey> = HashSet::new();
        for &edge in input_edges {
            let [u, v] = self.edge_vertices(edge);
            if hull_edge_pairs.contains(&undirected(u, v)) || !seen_interior_edges.insert(edge) {
                continue;
            }
            result.interior_edges.push(edge);

            if self.edge_faces_all_input(edge, &input_face_set) {
                result.unused_edges.push(edge);
            }
        }

        // §4.2 — every input face is interior except one *reused* as a hull
        // facet (§5). A reused face is reclassified out of interior at reuse
        // time, whether or not the cleanup pass later un-holed it as a border
        // face — so an un-holed border facet stays out of interior (and thus out
        // of §7 unused) even though it is absent from `result.hole_faces`. §7 —
        // every interior face is unused.
        let mut seen_interior_faces: HashSet<FaceKey> = HashSet::new();
        for &face in input_faces {
            if reused_faces.contains(&face) || !seen_interior_faces.insert(face) {
                continue;
            }
            result.interior_faces.push(face);
            result.unused_faces.push(face);
        }

        result
    }

    /// CONVEX_HULL.md §7 vertex-unused test. A loose vertex passes vacuously.
    fn vertex_incidence_all_input(
        &self,
        vertex: VertexKey,
        input_edges: &HashSet<EdgeKey>,
        input_faces: &HashSet<FaceKey>,
    ) -> bool {
        for edge in self.iter_disk_edges(vertex) {
            if !input_edges.contains(&edge) {
                return false;
            }
            for loop_key in self.iter_radial_loops(edge) {
                if !input_faces.contains(&self.loop_face(loop_key)) {
                    return false;
                }
            }
        }
        true
    }

    /// CONVEX_HULL.md §7 edge-unused test. A faceless edge passes vacuously.
    fn edge_faces_all_input(&self, edge: EdgeKey, input_faces: &HashSet<FaceKey>) -> bool {
        self.iter_radial_loops(edge)
            .all(|loop_key| input_faces.contains(&self.loop_face(loop_key)))
    }

    /// CONVEX_HULL.md §5 reuse test, per hull triangle: find an input face whose
    /// loop spans exactly the triangle's three verts and *every* edge of which
    /// is a hull edge. Returns the reusable face or `None`.
    ///
    /// The scan is anchored on the triangle's first vertex (`a`): only faces
    /// incident on `a` can contain all three verts, so the disk of `a` bounds
    /// the search.
    fn find_reusable_face(
        &self,
        a: VertexKey,
        b: VertexKey,
        c: VertexKey,
        input_faces: &HashSet<FaceKey>,
        hull_edge_pairs: &HashSet<[VertexKey; 2]>,
    ) -> Option<FaceKey> {
        let mut seen: HashSet<FaceKey> = HashSet::new();
        for edge in self.iter_disk_edges(a) {
            for loop_key in self.iter_radial_loops(edge) {
                let face = self.loop_face(loop_key);
                if !input_faces.contains(&face) || !seen.insert(face) {
                    continue;
                }

                // §5 step 1 — the face must occupy the triangle's footprint: its
                // loop spans exactly the three triangle verts (a triangular
                // face whose vertex set is {a, b, c}).
                if self.face_len(face) != 3 {
                    continue;
                }
                let face_verts = self.face_vert_keys(face);
                let covers = [b, c]
                    .iter()
                    .all(|v| face_verts.contains(v))
                    && face_verts.contains(&a);
                if !covers {
                    continue;
                }

                // §5 step 2 — a face is reusable only if it is entirely a hull
                // face: every one of its edges must be a hull edge. One interior
                // chord edge disqualifies it.
                let all_hull_edges = self.face_edge_keys(face).iter().all(|&e| {
                    let [u, v] = self.edge_vertices(e);
                    hull_edge_pairs.contains(&undirected(u, v))
                });
                if all_hull_edges {
                    return Some(face);
                }
            }
        }
        None
    }

    /// CONVEX_HULL.md §6.1 — find the **per-face** example face for a fresh hull
    /// triangle. This governs **only** the per-face copy of §6 step 3; the
    /// per-loop data is sourced separately and per-edge (see
    /// [`Self::copy_shared_corner_data`]).
    ///
    /// The criterion is broadened from "external neighbour" to **any face that is
    /// an input face OR a face not (yet) flagged as a hull-output face** — i.e.
    /// an already-present neighbour that is not one of the new hull faces this
    /// operator is building (nor a reused-as-hole facet). Input faces are
    /// **eligible**; the only exclusions are the operator's own built / reused
    /// hull-output faces. The scan walks the triangle's three edges in order
    /// `(a,b), (b,c), (c,a)`, and within each edge in radial-cycle order,
    /// returning the *first* eligible neighbour — the choice is position-dependent
    /// and affects customdata values, not incidence. A freshly-minted hull edge
    /// has no other face, so the match is always across a reused edge that
    /// already bordered existing geometry; `None` when no such neighbour exists
    /// (e.g. a hull over a pure loose-vert cloud), in which case the per-face
    /// data falls back to defaults.
    fn find_example_face(
        &self,
        a: VertexKey,
        b: VertexKey,
        c: VertexKey,
        built_faces: &HashSet<FaceKey>,
        hole_faces: &HashSet<FaceKey>,
    ) -> Option<FaceKey> {
        for (u, v) in [(a, b), (b, c), (c, a)] {
            let Some(edge) = self.query_edge_exists(u, v) else {
                continue;
            };
            for loop_key in self.iter_radial_loops(edge) {
                let face = self.loop_face(loop_key);
                if built_faces.contains(&face) || hole_faces.contains(&face) {
                    continue;
                }
                return Some(face);
            }
        }
        None
    }

    /// CONVEX_HULL.md §6 step 3, per-face half — copy `example`'s per-face
    /// customdata onto `face` *verbatim* (the `BM_elem_attrs_copy` semantics of
    /// BMESH_API.md §11, a copy and not an interpolation). This carries **only**
    /// the per-face layers; the new face's three corners still hold default loop
    /// data afterwards — the per-loop layers are filled separately by
    /// [`Self::copy_shared_corner_data`].
    fn copy_example_face_data(&mut self, face: FaceKey, example: FaceKey) {
        if !self.face_attribs.is_empty() {
            self.face_attribs
                .copy_into(face.as_usize(), example.as_usize());
        }
    }

    /// CONVEX_HULL.md §6 step 3, per-loop half — the `BM_face_copy_shared`
    /// equivalent (BMESH_API.md §11). For each of `face`'s edges that has a
    /// radially-adjacent neighbour face, copy that neighbour's two corner blocks
    /// onto `face`'s two corners at the shared edge's endpoints, **matched by
    /// vertex** (the destination corner at vertex `v` receives the neighbour's
    /// corner at the same vertex `v`), verbatim. The per-loop source is **each
    /// edge's own radial neighbour** (up to three different neighbours), *not* the
    /// per-face example. A corner whose touching edges have no other face keeps
    /// its default loop data.
    ///
    /// **Write-once per corner — first shared edge in loop-iteration order
    /// wins.** Each corner sits on two of `face`'s edges, both of which may carry
    /// a (distinct) neighbour; the walk visits the loops in loop order and a
    /// destination corner is written **at most once**, by the *first* shared edge
    /// reached that touches it. A corner already filled by an earlier edge is
    /// left untouched when a later edge's neighbour would otherwise write it.
    fn copy_shared_corner_data(&mut self, face: FaceKey) {
        if self.loop_attribs.is_empty() {
            return;
        }

        // Write-once per destination corner: a corner already supplied by an
        // earlier shared edge is not overwritten by a later edge's neighbour.
        let mut filled: HashSet<LoopKey> = HashSet::new();

        // Walk `face`'s loops in loop order (no allocation — a manual loop-link
        // walk, as the wall/cap copies in `extrude_region` do). Each loop sits on
        // one of `face`'s edges; `start` is `loop_edge(loop)`'s start corner and
        // `loop_next(start)` is its end corner.
        let first = self.face_first_loop(face);
        let mut start = first;
        loop {
            let end = self.loop_next(start);
            let edge = self.loop_edge(start);

            // The neighbour across this shared edge: the first loop in the edge's
            // radial cycle that belongs to a *different* face. A freshly-minted
            // hull edge has only `face`'s loop, so there is no neighbour.
            let mut neighbour: Option<LoopKey> = None;
            for radial in self.iter_radial_loops(edge) {
                if self.loop_face(radial) != face {
                    neighbour = Some(radial);
                    break;
                }
            }

            if let Some(neighbour_loop) = neighbour {
                // The neighbour's two corners on this edge: `neighbour_loop` at
                // its own vertex, `loop_next(neighbour_loop)` at the edge's other
                // endpoint. Match each destination corner to the neighbour corner
                // at the *same* vertex (winding-independent).
                let neighbour_next = self.loop_next(neighbour_loop);
                for &source in &[neighbour_loop, neighbour_next] {
                    let source_vertex = self.loop_vertex(source);
                    let target = if self.loop_vertex(start) == source_vertex {
                        start
                    } else if self.loop_vertex(end) == source_vertex {
                        end
                    } else {
                        continue;
                    };
                    if filled.insert(target) {
                        self.loop_attribs
                            .copy_into(target.as_usize(), source.as_usize());
                    }
                }
            }

            start = end;
            if start == first {
                break;
            }
        }
    }

    /// CONVEX_HULL.md §5 hole-cleanup pass. Mutates `hole_faces` in place
    /// (un-holing border faces) and writes the surviving hole faces + the
    /// patch's all-hole edges into `result`.
    fn cleanup_holes(&self, hole_faces: &mut HashSet<FaceKey>, result: &mut ConvexHullResult) {
        // A hole face touching a boundary edge (fewer than two incident faces)
        // is on the patch's border, not a true interior hole — un-hole it.
        let border: Vec<FaceKey> = hole_faces
            .iter()
            .copied()
            .filter(|&face| {
                self.face_edge_keys(face)
                    .iter()
                    .any(|&e| self.iter_radial_loops(e).take(2).count() < 2)
            })
            .collect();
        for face in border {
            hole_faces.remove(&face);
        }

        result.hole_faces = hole_faces.iter().copied().collect();
        result.hole_faces.sort_unstable_by_key(ElementKey::as_usize);

        // An edge all of whose faces are holes (and which has at least one face)
        // is itself a hole — the holes slot reports the patch's edges too.
        let mut hole_edges: HashSet<EdgeKey> = HashSet::new();
        for &face in &result.hole_faces {
            for edge in self.face_edge_keys(face) {
                let mut has_face = false;
                let all_holes = self.iter_radial_loops(edge).all(|l| {
                    has_face = true;
                    hole_faces.contains(&self.loop_face(l))
                });
                if has_face && all_holes {
                    hole_edges.insert(edge);
                }
            }
        }
        result.hole_edges = hole_edges.into_iter().collect();
        result.hole_edges.sort_unstable_by_key(ElementKey::as_usize);
    }
}

/// Audit of the [`ConvexHullProvider`] contract, run against whatever topology
/// the provider handed back. The whole function is `#[cfg(debug_assertions)]`
/// (and only called from behind the same gate), so its plain `assert!`s are
/// debug-only and release builds pay nothing — do not "fix" them into
/// `debug_assert!`, which would merely double-gate them.
///
/// [`Quickhull`] upholds every clause by construction; the check exists for
/// *injected* providers (a foreign hull engine behind an adapter), where a
/// contract slip would otherwise corrupt **silently**: an index the facets use
/// but `vertices` omits vanishes from every output slot, an index `vertices`
/// lists but no facet uses is handed to a downstream delete as unused, an
/// unsorted or duplicated `vertices` reorders or doubles `geom.out`, and a
/// facet-less topology is indistinguishable from the legitimate collinear hull.
/// Each message names its clause distinctly, so an adapter author reading the
/// failure knows exactly which obligation they broke.
#[cfg(debug_assertions)]
fn assert_provider_contract(cloud_len: usize, topology: &HullTopology) {
    // Clause: `vertices` is the on-hull subset of the input indices, strictly
    // ascending (which also rules out duplicates). Checked first — the facet
    // membership test below binary-searches it.
    for window in topology.vertices.windows(2) {
        assert!(
            window[0] < window[1],
            "ConvexHullProvider contract: `vertices` must be strictly ascending input \
             indices, but {} is followed by {} (a descending pair, or a duplicate that \
             would be emitted twice in geom.out)",
            window[0],
            window[1]
        );
    }
    for &index in &topology.vertices {
        assert!(
            index < cloud_len,
            "ConvexHullProvider contract: `vertices` lists out-of-range point index \
             {index}; the cloud has only {cloud_len} point(s), and `vertices` addresses \
             the input slice"
        );
    }

    for (facet_index, facet) in topology.faces.iter().enumerate() {
        assert!(
            facet.vertices.len() >= 3,
            "ConvexHullProvider contract: facet {facet_index} has too few corners ({}); a \
             hull face has at least three (a shorter loop is silently dropped from the fan)",
            facet.vertices.len()
        );
        for &index in &facet.vertices {
            assert!(
                index < cloud_len,
                "ConvexHullProvider contract: facet {facet_index} references out-of-range \
                 point index {index}; the cloud has only {cloud_len} point(s), and facet \
                 loops address the input slice"
            );
            assert!(
                topology.vertices.binary_search(&index).is_ok(),
                "ConvexHullProvider contract: facet {facet_index} references point index \
                 {index}, which `vertices` omits; `vertices` must contain every index its \
                 own facets reference (an omitted index is built into a face yet reported \
                 in no output slot)"
            );
        }
    }

    // Clause (the other half of the set equality): with facets present, `vertices`
    // is exactly the facets' index set — so no listed index may go unused. An
    // extra one is reported on-hull yet built into no face, and is then handed to
    // the reuse/delete pass as an unused vert. (Only checked with facets present:
    // the facet-less 1D seam's two endpoints legitimately belong to no face, and
    // the clause below owns that case.)
    if !topology.faces.is_empty() {
        for &index in &topology.vertices {
            assert!(
                topology
                    .faces
                    .iter()
                    .any(|facet| facet.vertices.contains(&index)),
                "ConvexHullProvider contract: `vertices` lists point index {index}, which no \
                 facet uses; with facets present `vertices` must be exactly the set of \
                 indices its own facets reference (a listed-but-unused index is reported \
                 on-hull yet built into no face)"
            );
        }
    }

    // Clause (derived): the only facet-less hull is the 1D (collinear) seam, whose
    // hull *is* its two extreme endpoints — a provider returns `None` below two
    // distinct points, and any 2D/3D cloud has at least one facet. So a facet-less
    // topology carrying anything but two verts is an adapter that lost its faces,
    // and without this check it would be taken for a valid degenerate hull: every
    // listed vert emitted on-hull, not one face built.
    if topology.faces.is_empty() {
        assert!(
            topology.vertices.len() == 2,
            "ConvexHullProvider contract: a facet-less hull is the 1D (collinear) seam and \
             must report exactly its two extreme endpoints, but `vertices` has {} \
             entries — a hull with no faces over a 2D/3D cloud means the provider lost its \
             facets",
            topology.vertices.len()
        );
    }
}

/// Order-independent key for an undirected vertex pair, so a hull edge is the
/// same whichever direction a triangle or face traverses it.
fn undirected(a: VertexKey, b: VertexKey) -> [VertexKey; 2] {
    if a.as_usize() <= b.as_usize() {
        [a, b]
    } else {
        [b, a]
    }
}

#[cfg(test)]
mod tests {
    use glam::Vec3;

    use crate::loopmesh::*;

    /// Build loose input verts from a slice of positions, returning their keys
    /// in input order.
    fn cloud(mesh: &mut LoopMesh, positions: &[Vec3]) -> Vec<VertexKey> {
        positions.iter().map(|&p| mesh.create_vertex(p)).collect()
    }

    /// The eight corners of the unit cube centred on the origin.
    fn cube_corners() -> Vec<Vec3> {
        let mut out = Vec::with_capacity(8);
        for &x in &[-1.0f32, 1.0] {
            for &y in &[-1.0f32, 1.0] {
                for &z in &[-1.0f32, 1.0] {
                    out.push(Vec3::new(x, y, z));
                }
            }
        }
        out
    }

    /// Geometric centroid of the input positions — the reference point an
    /// outward normal must point away from.
    fn centroid(mesh: &LoopMesh, verts: &[VertexKey]) -> Vec3 {
        let sum: Vec3 = verts
            .iter()
            .fold(Vec3::ZERO, |acc, &v| acc + mesh.vertex_position(v));
        sum / verts.len() as f32
    }

    /// Assert the closed-surface invariants the guide names for a fully
    /// 3D hull (CONVEX_HULL.md §9, "Full-stack invariants"): every input vert
    /// on the hull, all faces triangles, Euler V−E+F=2, and outward winding.
    fn assert_closed_hull_invariants(mesh: &mut LoopMesh, result: &ConvexHullResult, input: &[VertexKey]) {
        // Every input vertex is extremal for these clouds → on the hull.
        let on_hull: std::collections::HashSet<VertexKey> = result.verts().iter().copied().collect();
        for &v in input {
            assert!(on_hull.contains(&v), "input vertex {v:?} must be on the hull");
        }

        // All hull faces are triangles (the fan output, §3.1).
        for &face in result.faces() {
            assert_eq!(mesh.face_len(face), 3, "every hull face is a triangle");
        }

        // Closed convex triangle surface: V − E + F = 2.
        let v = result.verts().len() as i64;
        let e = result.edges().len() as i64;
        let f = result.faces().len() as i64;
        assert_eq!(v - e + f, 2, "Euler characteristic of a closed hull surface is 2");

        // Outward winding: each face normal points away from the centroid (§3.2).
        let center = centroid(mesh, input);
        let faces: Vec<FaceKey> = result.faces().to_vec();
        for face in faces {
            mesh.face_normal_update(face);
            let normal = mesh.face_normal(face);
            let face_point = mesh.vertex_position(mesh.loop_vertex(mesh.face_first_loop(face)));
            let outward = face_point - center;
            assert!(
                normal.dot(outward) > 0.0,
                "hull face {face:?} must be wound outward (normal points away from centroid)"
            );
        }
    }

    #[test]
    fn cube_cloud_builds_closed_triangulated_hull() {
        let mut mesh = LoopMesh::new();
        let input = cloud(&mut mesh, &cube_corners());
        let result = mesh.convex_hull(&input);

        assert_eq!(mesh.vertex_count(), 8, "no vertex is created");
        assert_eq!(result.verts().len(), 8, "all eight corners are extremal");
        // A cube hull is 6 quad facets, each fanned into 2 triangles → 12.
        assert_eq!(result.faces().len(), 12, "cube hull fans to 12 triangles");
        assert_eq!(result.edges().len(), 18, "12 cube edges + 6 facet diagonals");

        assert_closed_hull_invariants(&mut mesh, &result, &input);
        mesh.validate_topology();
    }

    #[test]
    fn tetrahedron_cloud_builds_four_triangles() {
        let mut mesh = LoopMesh::new();
        let input = cloud(
            &mut mesh,
            &[
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
                Vec3::new(0.0, 0.0, 1.0),
            ],
        );
        let result = mesh.convex_hull(&input);

        assert_eq!(mesh.vertex_count(), 4, "no vertex is created");
        assert_eq!(result.verts().len(), 4, "all four corners are extremal");
        assert_eq!(result.faces().len(), 4, "a tetrahedron hull is four triangles");
        assert_eq!(result.edges().len(), 6, "a tetrahedron has six edges");

        assert_closed_hull_invariants(&mut mesh, &result, &input);
        mesh.validate_topology();
    }

    #[test]
    fn interior_point_is_not_on_the_hull() {
        // Eight cube corners plus one point buried at the centre: the buried
        // point is not extremal, so it is absent from geom.out while the hull
        // is still the cube (CONVEX_HULL.md §4.1 — interior verts excluded).
        let mut mesh = LoopMesh::new();
        let mut positions = cube_corners();
        positions.push(Vec3::new(0.0, 0.0, 0.0));
        let input = cloud(&mut mesh, &positions);
        let interior = *input.last().unwrap();

        let result = mesh.convex_hull(&input);

        let on_hull: std::collections::HashSet<VertexKey> = result.verts().iter().copied().collect();
        assert!(
            !on_hull.contains(&interior),
            "the buried centre point must not be on the hull"
        );
        assert_eq!(result.verts().len(), 8, "only the eight extremal corners");
        assert_eq!(result.faces().len(), 12, "the hull is still the cube → 12 triangles");
        // The closed-hull invariants hold over the eight extremal corners.
        assert_closed_hull_invariants(&mut mesh, &result, &input[..8]);
        mesh.validate_topology();
    }

    #[test]
    fn fewer_than_three_verts_is_a_no_op() {
        // §1.2 — the count guard cancels with no changes.
        let mut mesh = LoopMesh::new();
        let input = cloud(
            &mut mesh,
            &[Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0)],
        );
        let result = mesh.convex_hull(&input);

        assert!(result.is_empty(), "fewer than three verts builds nothing");
        assert_eq!(mesh.face_count(), 0, "no face is created");
        assert_eq!(mesh.edge_count(), 0, "no edge is created");
        mesh.validate_topology();
    }

    #[test]
    fn existing_hull_edges_are_reused_not_duplicated() {
        // Wire one cube edge before hulling: the no-double discipline must reuse
        // it rather than mint a parallel edge (CONVEX_HULL.md §6 step 1 / §2).
        let mut mesh = LoopMesh::new();
        let input = cloud(&mut mesh, &cube_corners());
        // corners[0] = (-1,-1,-1), corners[1] = (-1,-1,1): a cube edge.
        let pre = mesh.create_edge([input[0], input[1]], EdgeCreateFlags::NO_DUPLICATE);

        let result = mesh.convex_hull(&input);

        assert!(
            result.edges().contains(&pre),
            "the pre-existing cube edge must be reused as a hull edge"
        );
        assert_eq!(mesh.edge_count(), 18, "no duplicate edge is minted");
        assert_eq!(result.faces().len(), 12);
        assert_closed_hull_invariants(&mut mesh, &result, &input);
        mesh.validate_topology();
    }

    #[test]
    fn coplanar_square_builds_flat_fan_with_interior_culled() {
        // Four square corners (z = 0) plus one strictly-interior centre point:
        // the hull is the flat square patch, the centre is culled
        // (CONVEX_HULL.md §8 coplanar bullet). The provider returns one planar
        // facet over the four corners, fanned into a flat outward-wound
        // triangle patch (§3.1). A strictly-interior point avoids the
        // on-boundary-edge ambiguity a denser grid would introduce.
        let mut mesh = LoopMesh::new();
        let input = cloud(
            &mut mesh,
            &[
                Vec3::new(-1.0, -1.0, 0.0),
                Vec3::new(1.0, -1.0, 0.0),
                Vec3::new(1.0, 1.0, 0.0),
                Vec3::new(-1.0, 1.0, 0.0),
                Vec3::new(0.0, 0.0, 0.0), // strictly interior — culled
            ],
        );
        let center = *input.last().unwrap();

        let result = mesh.convex_hull(&input);

        assert_eq!(mesh.vertex_count(), 5, "no vertex is created");

        let on_hull: std::collections::HashSet<VertexKey> =
            result.verts().iter().copied().collect();
        assert!(
            !on_hull.contains(&center),
            "the strictly-interior centre point is culled from the flat hull"
        );
        assert_eq!(
            result.verts().len(),
            4,
            "the four square corners are on the flat hull"
        );

        // A flat patch is a disk, not a closed surface: it is fanned into
        // triangles, every face stays in the input plane (no invented vertex).
        // A quad facet fans into 4 − 2 = 2 triangles.
        assert_eq!(result.faces().len(), 2, "the quad facet fans to two triangles");
        for &face in result.faces() {
            assert_eq!(mesh.face_len(face), 3, "the flat facet is fanned to triangles");
        }
        for &v in result.verts() {
            assert!(
                mesh.vertex_position(v).z.abs() < 1e-6,
                "flat-hull verts stay in the input plane"
            );
        }
        mesh.validate_topology();
    }

    #[test]
    fn collinear_cloud_builds_no_faces_two_endpoints_on_hull() {
        // Five points strung along the x axis: the hull is a 1D segment with two
        // extreme endpoints and no facet (CONVEX_HULL.md §8 collinear bullet).
        // The operator builds no faces; the two endpoints are on-hull verts.
        let mut mesh = LoopMesh::new();
        let input = cloud(
            &mut mesh,
            &[
                Vec3::new(-2.0, 0.0, 0.0),
                Vec3::new(-1.0, 0.0, 0.0),
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(2.0, 0.0, 0.0),
            ],
        );

        let result = mesh.convex_hull(&input);

        assert!(result.faces().is_empty(), "a 1D hull builds no faces");
        assert!(result.edges().is_empty(), "a 1D hull builds no hull edges");
        assert_eq!(mesh.face_count(), 0, "no face is materialised");
        assert_eq!(mesh.edge_count(), 0, "no edge is materialised");

        // The two extreme endpoints (x = -2 and x = 2) are the on-hull verts.
        let on_hull: std::collections::HashSet<VertexKey> =
            result.verts().iter().copied().collect();
        assert_eq!(result.verts().len(), 2, "exactly the two extreme endpoints");
        assert!(on_hull.contains(&input[0]), "x = -2 endpoint is on-hull");
        assert!(on_hull.contains(&input[4]), "x = 2 endpoint is on-hull");
        for interior in [input[1], input[2], input[3]] {
            assert!(
                !on_hull.contains(&interior),
                "the three interior collinear points are not on-hull"
            );
        }

        // The no-faces path must still populate geom.out verts directly (the
        // degenerate-path seam): the two endpoints are reported as verts even
        // though no facet bounds them.
        let geom_verts: std::collections::HashSet<VertexKey> = result
            .geom_out()
            .into_iter()
            .filter_map(|g| match g {
                GeomKey::Vert(v) => Some(v),
                _ => None,
            })
            .collect();
        assert_eq!(geom_verts.len(), 2, "geom.out reports two endpoint verts");
        assert!(geom_verts.contains(&input[0]), "x = -2 endpoint in geom.out");
        assert!(geom_verts.contains(&input[4]), "x = 2 endpoint in geom.out");
        mesh.validate_topology();
    }

    #[test]
    fn coincident_duplicate_is_deduplicated_off_the_hull() {
        // Tetrahedron corners plus an exact duplicate of corner 0. The provider
        // deduplicates coincident positions before the hull runs, so the
        // duplicate never becomes a distinct hull corner (CONVEX_HULL.md §8
        // coincident bullet). The hull is the same four-triangle tetrahedron and
        // the duplicate is absent from the on-hull set.
        let mut mesh = LoopMesh::new();
        let mut positions = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
        ];
        positions.push(positions[0]); // exact duplicate of corner 0
        let input = cloud(&mut mesh, &positions);
        let duplicate = *input.last().unwrap();

        let result = mesh.convex_hull(&input);

        assert_eq!(mesh.vertex_count(), 5, "no vertex is created");
        assert_eq!(result.faces().len(), 4, "the hull is still a four-triangle tetrahedron");

        let on_hull: std::collections::HashSet<VertexKey> =
            result.verts().iter().copied().collect();
        assert!(
            !on_hull.contains(&duplicate),
            "the non-representative duplicate is not on the hull"
        );
        // Exactly one of {corner 0, its duplicate} represents that position; the
        // on-hull set is the four distinct tetrahedron corners.
        assert_eq!(result.verts().len(), 4, "four distinct extremal corners");
        assert!(on_hull.contains(&input[0]), "the representative corner 0 is on-hull");
        mesh.validate_topology();
    }

    /// Collect the vertex keys of an interior/unused slot into a set.
    fn vert_set(keys: &[VertexKey]) -> std::collections::HashSet<VertexKey> {
        keys.iter().copied().collect()
    }

    #[test]
    fn interior_point_is_classified_interior_and_unused() {
        // Eight cube corners plus one loose buried point: the buried point is the
        // sole interior vert, and (being loose — no incident edges) it is unused
        // (CONVEX_HULL.md §4.1, §7).
        let mut mesh = LoopMesh::new();
        let mut positions = cube_corners();
        positions.push(Vec3::new(0.0, 0.0, 0.0));
        let input = cloud(&mut mesh, &positions);
        let interior = *input.last().unwrap();

        let result = mesh.convex_hull(&input);

        assert_eq!(
            result.interior_verts(),
            &[interior],
            "only the buried centre point is interior"
        );
        assert_eq!(
            result.unused_verts(),
            &[interior],
            "the loose buried point is unused (no incident geometry)"
        );

        // The on-hull corners are neither interior nor unused.
        let on_hull = vert_set(result.verts());
        for v in result.interior_verts() {
            assert!(!on_hull.contains(v), "interior verts are excluded from geom.out");
        }

        // geom_interior.out / geom_unused.out mirror the typed accessors.
        let interior_geom: std::collections::HashSet<VertexKey> = result
            .geom_interior_out()
            .into_iter()
            .filter_map(|g| match g {
                GeomKey::Vert(v) => Some(v),
                _ => None,
            })
            .collect();
        assert_eq!(interior_geom, vert_set(&[interior]));
        mesh.validate_topology();
    }

    #[test]
    fn coplanar_culled_centre_is_interior_and_unused() {
        // Four square corners + one strictly-interior centre point: the centre is
        // culled from the flat hull and must be reported interior + unused
        // (CONVEX_HULL.md §8 coplanar bullet, §4.1, §7).
        let mut mesh = LoopMesh::new();
        let input = cloud(
            &mut mesh,
            &[
                Vec3::new(-1.0, -1.0, 0.0),
                Vec3::new(1.0, -1.0, 0.0),
                Vec3::new(1.0, 1.0, 0.0),
                Vec3::new(-1.0, 1.0, 0.0),
                Vec3::new(0.0, 0.0, 0.0), // strictly interior — culled
            ],
        );
        let center = *input.last().unwrap();

        let result = mesh.convex_hull(&input);

        assert_eq!(result.interior_verts(), &[center], "the culled centre is interior");
        assert_eq!(result.unused_verts(), &[center], "the culled centre is unused");
        mesh.validate_topology();
    }

    #[test]
    fn collinear_intermediates_are_interior_and_unused() {
        // Five x-axis points: the two extreme endpoints are on-hull; the three
        // intermediate points are interior, and (loose) unused (CONVEX_HULL.md
        // §8 collinear bullet, §7).
        let mut mesh = LoopMesh::new();
        let input = cloud(
            &mut mesh,
            &[
                Vec3::new(-2.0, 0.0, 0.0),
                Vec3::new(-1.0, 0.0, 0.0),
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(2.0, 0.0, 0.0),
            ],
        );

        let result = mesh.convex_hull(&input);

        let expected = vert_set(&[input[1], input[2], input[3]]);
        assert_eq!(vert_set(result.interior_verts()), expected, "three intermediates interior");
        assert_eq!(vert_set(result.unused_verts()), expected, "loose intermediates unused");
        mesh.validate_topology();
    }

    #[test]
    fn coincident_duplicate_key_is_interior_and_unused() {
        // Tetrahedron corners + exact duplicate of corner 0: the duplicate's key
        // is buried (the provider keeps only one representative), so it is
        // interior + unused (CONVEX_HULL.md §8 coincident bullet, §4.1, §7).
        let mut mesh = LoopMesh::new();
        let mut positions = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
        ];
        positions.push(positions[0]);
        let input = cloud(&mut mesh, &positions);
        let duplicate = *input.last().unwrap();

        let result = mesh.convex_hull(&input);

        // Exactly one of {corner 0, duplicate} is the on-hull representative; the
        // other is the lone interior/unused vert.
        assert_eq!(result.interior_verts().len(), 1, "one buried duplicate key");
        let interior = result.interior_verts()[0];
        assert!(
            interior == input[0] || interior == duplicate,
            "the buried key is corner 0 or its duplicate"
        );
        assert_eq!(vert_set(result.unused_verts()), vert_set(&[interior]), "the buried key is unused");
        mesh.validate_topology();
    }

    #[test]
    fn fully_extremal_cloud_has_no_interior_verts() {
        // Tetrahedron: every input vert is a hull corner, so nothing is interior.
        let mut mesh = LoopMesh::new();
        let input = cloud(
            &mut mesh,
            &[
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
                Vec3::new(0.0, 0.0, 1.0),
            ],
        );

        let result = mesh.convex_hull(&input);

        assert!(result.interior_verts().is_empty(), "every corner is extremal");
        assert!(result.unused_verts().is_empty(), "nothing to delete");
        mesh.validate_topology();
    }

    #[test]
    fn interior_vert_with_non_input_incidence_is_not_unused() {
        // A buried point wired to a non-input vertex by a non-input edge: the
        // non-input-incidence guard (CONVEX_HULL.md §7) keeps it interior but
        // refuses to mark it unused, since deleting it would disturb the
        // unselected edge/vertex.
        let mut mesh = LoopMesh::new();
        let mut positions = cube_corners();
        positions.push(Vec3::new(0.0, 0.0, 0.0)); // buried, input
        let input = cloud(&mut mesh, &positions);
        let interior = *input.last().unwrap();

        // A non-input vertex outside the cube, wired to the buried point.
        let outsider = mesh.create_vertex(Vec3::new(5.0, 0.0, 0.0));
        mesh.create_edge([interior, outsider], EdgeCreateFlags::NO_DUPLICATE);

        let result = mesh.convex_hull(&input);

        assert_eq!(result.interior_verts(), &[interior], "the buried point is still interior");
        assert!(
            result.unused_verts().is_empty(),
            "a vert leaned on by a non-input edge is not unused"
        );
        mesh.validate_topology();
    }

    #[test]
    fn input_chord_edge_buried_in_hull_is_interior_and_unused() {
        // §4.2 / §7 — a chord edge between two cube corners that is not a hull
        // edge (a face diagonal across the cube's interior) is interior, and —
        // with no faces touching it — unused. The two on-hull endpoints stay on
        // the hull; only the chord itself is interior.
        let mut mesh = LoopMesh::new();
        let input = cloud(&mut mesh, &cube_corners());
        // corners[0] = (-1,-1,-1), corners[7] = (1,1,1): the cube's space
        // diagonal — a body chord through the interior, sharing no facet, so it
        // is never a hull triangle edge (a face diagonal could coincide with a
        // fan diagonal; the space diagonal cannot).
        let chord = mesh.create_edge([input[0], input[7]], EdgeCreateFlags::NO_DUPLICATE);

        let result = mesh.convex_hull_with_input(&input, &[chord], &[], false);

        assert_eq!(
            result.interior_edges(),
            &[chord],
            "the buried chord edge is interior"
        );
        assert_eq!(
            result.unused_edges(),
            &[chord],
            "the chord has no faces, so it is unused"
        );
        // The chord must not be reported on the hull surface.
        assert!(
            !result.edges().contains(&chord),
            "an interior chord is not a hull edge"
        );
        // geom_interior.out / geom_unused.out carry it as an edge.
        let interior_edges: std::collections::HashSet<EdgeKey> = result
            .geom_interior_out()
            .into_iter()
            .filter_map(|g| match g {
                GeomKey::Edge(e) => Some(e),
                _ => None,
            })
            .collect();
        assert!(interior_edges.contains(&chord), "geom_interior.out carries the chord");
        mesh.validate_topology();
    }

    #[test]
    fn input_face_buried_in_hull_is_interior_and_unused() {
        // §4.2 / §7 — an input face buried inside the hull (using a buried
        // interior vertex) is interior, and every interior face is unused. The
        // hull surface is built fresh over the extremal verts.
        let mut mesh = LoopMesh::new();
        let mut positions = tetra_positions();
        positions.push(Vec3::new(0.4, 0.4, 0.4)); // buried interior point
        let input = cloud(&mut mesh, &positions);
        let interior_vert = *input.last().unwrap();

        // A face leaning on the buried vertex — never a hull facet.
        let buried_face = mesh.face_create_verts(
            &[input[0], input[1], interior_vert],
            FaceCreateFlags::NO_DUPLICATE,
        );

        let result = mesh.convex_hull_with_input(&input, &[], &[buried_face], false);

        assert_eq!(
            result.interior_faces(),
            &[buried_face],
            "the buried input face is interior"
        );
        assert_eq!(
            result.unused_faces(),
            &[buried_face],
            "every interior face is unused"
        );
        mesh.validate_topology();
    }

    #[test]
    fn interior_vert_all_input_incidence_is_unused() {
        // §7 discriminating case (distinct from the loose-vert case): a buried
        // vertex wired by an *input* edge to another buried vertex, both leaning
        // only on input geometry, is unused — the general all-incident-input
        // guard passes even though the vert is not loose.
        let mut mesh = LoopMesh::new();
        let mut positions = cube_corners();
        positions.push(Vec3::new(0.0, 0.0, 0.0)); // buried A
        positions.push(Vec3::new(0.1, 0.1, 0.1)); // buried B
        let input = cloud(&mut mesh, &positions);
        let buried_a = input[8];
        let buried_b = input[9];

        // An input edge between the two buried points: its only incidence is
        // itself an input element.
        let inner = mesh.create_edge([buried_a, buried_b], EdgeCreateFlags::NO_DUPLICATE);

        let result = mesh.convex_hull_with_input(&input, &[inner], &[], false);

        let interior = vert_set(result.interior_verts());
        assert_eq!(interior, vert_set(&[buried_a, buried_b]), "both buried points interior");
        // Both are unused: each one's only incident edge is the input `inner`,
        // which has no faces — the general guard passes though neither is loose.
        let _ = inner;
        assert_eq!(
            vert_set(result.unused_verts()),
            vert_set(&[buried_a, buried_b]),
            "all-input-incidence buried verts are unused (not loose, but guarded)"
        );
        // The shared input edge is itself interior + unused (faceless).
        assert_eq!(result.interior_edges(), &[inner], "the inner edge is interior");
        assert_eq!(result.unused_edges(), &[inner], "the faceless inner edge is unused");
        mesh.validate_topology();
    }

    #[test]
    fn non_input_incidence_blocks_unused_for_vert_and_edge() {
        // §7 — the discriminating guard: a buried vertex and an input chord edge
        // are both interior, but a *non-input* face leaning on them keeps both
        // out of the unused slot, since deleting them would disturb unselected
        // geometry.
        let mut mesh = LoopMesh::new();
        let mut positions = cube_corners();
        positions.push(Vec3::new(0.0, 0.0, 0.0)); // buried interior point
        let input = cloud(&mut mesh, &positions);
        let buried = input[8];

        // An input chord edge from a cube corner to the buried point.
        let chord = mesh.create_edge([input[0], buried], EdgeCreateFlags::NO_DUPLICATE);
        // A NON-input face that uses both the buried vert and the chord edge.
        let _non_input_face = mesh.face_create_verts(
            &[input[0], buried, input[1]],
            FaceCreateFlags::NO_DUPLICATE,
        );

        // Only the chord is an input edge; the non-input face is not in the input
        // face set.
        let result = mesh.convex_hull_with_input(&input, &[chord], &[], false);

        // Both are interior.
        assert!(result.interior_verts().contains(&buried), "buried vert interior");
        assert!(result.interior_edges().contains(&chord), "chord edge interior");

        // Neither is unused: the non-input face leans on both.
        assert!(
            !result.unused_verts().contains(&buried),
            "a vert touched by a non-input face is not unused"
        );
        assert!(
            !result.unused_edges().contains(&chord),
            "an edge with a non-input face is not unused"
        );
        mesh.validate_topology();
    }

    /// The four corners of a regular-ish tetrahedron whose four facets are all
    /// triangles — the cleanest fixture for the reuse path, since a triangular
    /// hull facet has no triangulation ambiguity (the facet *is* the triangle).
    fn tetra_positions() -> Vec<Vec3> {
        vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::new(0.0, 2.0, 0.0),
            Vec3::new(0.0, 0.0, 2.0),
        ]
    }

    /// Build the four triangular faces of the tetrahedron over `verts`, each in
    /// an outward winding. Returns the created face keys. The closed surface
    /// gives every edge two incident faces, so none of the four faces is a
    /// border face — they all survive the §5 hole-cleanup pass.
    fn build_tetra_faces(mesh: &mut LoopMesh, v: &[VertexKey]) -> Vec<FaceKey> {
        let tris = [
            [v[0], v[2], v[1]],
            [v[0], v[1], v[3]],
            [v[0], v[3], v[2]],
            [v[1], v[2], v[3]],
        ];
        tris.iter()
            .map(|t| mesh.face_create_verts(t, FaceCreateFlags::NO_DUPLICATE))
            .collect()
    }

    #[test]
    fn use_existing_faces_true_reuses_facets_as_holes_no_duplicates() {
        // A fully-faced tetrahedron, hulled with use_existing_faces = true: each
        // hull facet coincides with an existing triangle whose edges are all
        // hull edges, so every facet is reused (not rebuilt) and tagged a hole
        // (CONVEX_HULL.md §5). No new face is created.
        let mut mesh = LoopMesh::new();
        let input = cloud(&mut mesh, &tetra_positions());
        let faces = build_tetra_faces(&mut mesh, &input);
        let faces_before = mesh.face_count();
        let edges_before = mesh.edge_count();

        let result = mesh.convex_hull_with_faces(&input, &faces, true);

        // No hull triangle is materialised — all four are reused.
        assert!(result.faces().is_empty(), "every facet is reused, none built");
        assert_eq!(mesh.face_count(), faces_before, "no new face is created");
        assert_eq!(mesh.edge_count(), edges_before, "no new edge is created");

        // All four pre-existing faces are reported as holes (closed surface →
        // no border face un-holed).
        let hole_faces: std::collections::HashSet<FaceKey> =
            result.hole_faces().iter().copied().collect();
        assert_eq!(hole_faces.len(), 4, "all four facets are holes");
        for &f in &faces {
            assert!(hole_faces.contains(&f), "input face {f:?} is a hole");
        }

        // The patch's edges — all six tetra edges, each with both faces holes —
        // are reported as hole edges too (§5 cleanup).
        assert_eq!(result.hole_edges().len(), 6, "all six patch edges are holes");

        // geom.out still reports the whole hull surface: the reused verts/edges
        // show up even though no face was built (§6.2).
        assert_eq!(result.verts().len(), 4, "all four corners on-hull");
        assert_eq!(result.edges().len(), 6, "all six hull edges reported");

        // geom_holes.out mirrors the typed accessors.
        let holes_geom = result.geom_holes_out();
        assert_eq!(holes_geom.len(), 10, "6 hole edges + 4 hole faces");
        mesh.validate_topology();
    }

    #[test]
    fn use_existing_faces_false_builds_duplicates_holes_empty() {
        // The contrast arm: the same fully-faced tetrahedron hulled with
        // use_existing_faces = false. Every hull triangle is materialised
        // unconditionally, stacking a second face on each existing facet's
        // footprint, and geom_holes.out comes back empty
        // (CONVEX_HULL.md §5 last ¶, §9).
        let mut mesh = LoopMesh::new();
        let input = cloud(&mut mesh, &tetra_positions());
        let faces = build_tetra_faces(&mut mesh, &input);
        let faces_before = mesh.face_count();

        let result = mesh.convex_hull_with_faces(&input, &faces, false);

        // Every hull triangle is materialised regardless of whether an input
        // face already occupies its footprint (§5 last ¶): the four facets are
        // reported as *built* output, and at least as many faces exist as before
        // (the unconditional build never removes the input faces).
        assert_eq!(result.faces().len(), 4, "four hull facets reported as built");
        assert!(
            mesh.face_count() >= faces_before,
            "the unconditional build never drops the input faces"
        );
        let _ = faces;

        // The contrast contract (the whole point of this pair): the holes slot
        // is empty when the flag is clear.
        assert!(result.hole_faces().is_empty(), "no holes when flag clear");
        assert!(result.hole_edges().is_empty(), "no hole edges when flag clear");
        assert!(result.geom_holes_out().is_empty(), "geom_holes.out empty when flag clear");
        mesh.validate_topology();
    }

    #[test]
    fn use_existing_faces_true_with_non_hull_chord_edge_is_not_reused() {
        // A reuse candidate is rejected if even one of its edges is *not* a hull
        // edge (CONVEX_HULL.md §5 step 2). Here a face buried inside the hull
        // (not on the surface) must never be reused — its footprint is not a
        // hull triangle at all, so it is simply not found, and the surface is
        // built fresh over the extremal verts.
        let mut mesh = LoopMesh::new();
        let mut positions = tetra_positions();
        positions.push(Vec3::new(0.4, 0.4, 0.4)); // buried interior point
        let input = cloud(&mut mesh, &positions);
        let interior = *input.last().unwrap();

        // A face that uses the buried interior vertex — its edges to the
        // interior point are interior chords, never hull edges.
        let buried = mesh.face_create_verts(
            &[input[0], input[1], interior],
            FaceCreateFlags::NO_DUPLICATE,
        );

        let result = mesh.convex_hull_with_faces(&input, &[buried], true);

        // The buried face is not on the hull surface, so it is not a hole.
        assert!(
            result.hole_faces().is_empty(),
            "a buried interior face is never reused as a hole"
        );
        // The hull surface is still the tetrahedron, built fresh.
        assert_eq!(result.faces().len(), 4, "the four tetra facets are built");
        // The buried point is interior.
        assert!(
            result.interior_verts().contains(&interior),
            "the buried point is interior"
        );
        mesh.validate_topology();
    }

    #[test]
    fn use_existing_faces_true_border_face_is_un_holed() {
        // §5 cleanup: a hole face touching a boundary edge (an edge with fewer
        // than two incident faces) is on the patch's border, not a true interior
        // hole, and is un-holed. A flat single-triangle hull over three coplanar
        // verts has exactly one open facet whose three edges are all boundary
        // (no second facet shares them), so the reused face — though it
        // coincided with the hull facet and was skipped — is un-holed by the
        // cleanup pass and the holes slot ends up empty.
        let mut mesh = LoopMesh::new();
        let input = cloud(
            &mut mesh,
            &[
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(2.0, 0.0, 0.0),
                Vec3::new(0.0, 2.0, 0.0),
            ],
        );
        let patch = mesh.face_create_verts(&input, FaceCreateFlags::NO_DUPLICATE);

        let result = mesh.convex_hull_with_faces(&input, &[patch], true);

        // The single facet coincides with the patch and is reused (skipped, not
        // rebuilt) — its three edges are all boundary, so cleanup un-holes it.
        assert!(
            result.hole_faces().is_empty(),
            "a border patch face is un-holed (all its edges are boundary)"
        );
        assert!(result.hole_edges().is_empty(), "no all-hole edges survive");

        // The patch's facet is still reported on the hull surface (§6.2): its
        // three verts and three edges are in geom.out even though nothing built.
        assert!(result.faces().is_empty(), "the only facet was reused, none built");
        assert_eq!(result.verts().len(), 3, "all three corners on-hull");
        assert_eq!(result.edges().len(), 3, "all three facet edges on-hull");
        mesh.validate_topology();
    }

    #[test]
    fn use_existing_faces_true_un_holed_border_face_is_not_interior_or_unused() {
        // §4.2 / §5 — a face *reused* as a hull facet is reclassified out of
        // interior at reuse time, independent of whether the §5 cleanup later
        // un-holes it as a patch border. The flat single-triangle hull reuses
        // its one coincident face (skipped, not rebuilt) and the cleanup pass
        // then un-holes it (all three edges are boundary). The un-holed face
        // must NOT fall back to interior — it is on the hull surface — and the
        // §7 every-interior-face-unused rule must NOT route it to unused.
        let mut mesh = LoopMesh::new();
        let input = cloud(
            &mut mesh,
            &[
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(2.0, 0.0, 0.0),
                Vec3::new(0.0, 2.0, 0.0),
            ],
        );
        let patch = mesh.face_create_verts(&input, FaceCreateFlags::NO_DUPLICATE);

        let result = mesh.convex_hull_with_faces(&input, &[patch], true);

        // Reused-and-then-un-holed: not a (surviving) hole, none built.
        assert!(result.hole_faces().is_empty(), "the border face is un-holed");
        assert!(result.faces().is_empty(), "the facet was reused, none built");

        // The crux: the reused face is on the hull surface, so it is neither
        // interior nor unused — reporting it would name a hull-surface face as
        // safe-to-delete.
        assert!(
            !result.interior_faces().contains(&patch),
            "a reused (then un-holed) facet is not interior"
        );
        assert!(
            !result.unused_faces().contains(&patch),
            "a reused (then un-holed) facet is not unused"
        );
        let interior_faces: std::collections::HashSet<FaceKey> = result
            .geom_interior_out()
            .into_iter()
            .filter_map(|g| match g {
                GeomKey::Face(f) => Some(f),
                _ => None,
            })
            .collect();
        assert!(
            !interior_faces.contains(&patch),
            "geom_interior.out must not carry the reused facet"
        );
        let unused_faces: std::collections::HashSet<FaceKey> = result
            .geom_unused_out()
            .into_iter()
            .filter_map(|g| match g {
                GeomKey::Face(f) => Some(f),
                _ => None,
            })
            .collect();
        assert!(
            !unused_faces.contains(&patch),
            "geom_unused.out must not carry the reused facet"
        );

        // The reused facet's verts and edges are still reported on-hull (§6.2).
        assert_eq!(result.verts().len(), 3, "all three corners on-hull");
        assert_eq!(result.edges().len(), 3, "all three facet edges on-hull");
        mesh.validate_topology();
    }

    /// Five corners of a triangular bipyramid: an equatorial triangle (verts
    /// 0,1,2) with an apex above (3) and below (4). All six facets are genuine
    /// triangles — no coplanar merge — so it isolates the reuse-build path from
    /// the facet-set wobble a cospherical cloud would add.
    fn bipyramid_positions() -> Vec<Vec3> {
        vec![
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(-0.5, 0.866_025, 0.0),
            Vec3::new(-0.5, -0.866_025, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(0.0, 0.0, -1.0),
        ]
    }

    #[test]
    fn use_existing_faces_true_partial_patch_is_deterministic_and_outward() {
        // The reuse-build path: pre-build the two upper-apex facets of a
        // bipyramid, then hull with use_existing_faces = true. The pre-built
        // facets must be reused-as-holes (reported under their original keys, no
        // duplicate face minted on the same edges), and the remaining facets must
        // be built fresh with outward winding — deterministically, not at the
        // mercy of the hash seed (CONVEX_HULL.md §5, §3.2). This pins the bug the
        // partial-patch A/B test exposed.
        let mut mesh = LoopMesh::new();
        let input = cloud(&mut mesh, &bipyramid_positions());

        // Two upper facets sharing apex 3, wound outward (apex away from the
        // bipyramid centroid at the origin).
        let pre_upper_a = mesh.face_create_verts(
            &[input[0], input[1], input[3]],
            FaceCreateFlags::NO_DUPLICATE,
        );
        let pre_upper_b = mesh.face_create_verts(
            &[input[1], input[2], input[3]],
            FaceCreateFlags::NO_DUPLICATE,
        );
        let pre_faces = [pre_upper_a, pre_upper_b];
        let faces_before = mesh.face_count();

        let result = mesh.convex_hull_with_faces(&input, &pre_faces, true);

        // The reused facets must be reported under their *original* keys — no
        // fresh key, no duplicate mint.
        let holes: std::collections::HashSet<FaceKey> =
            result.hole_faces().iter().copied().collect();
        for &pre in &pre_faces {
            assert!(
                holes.contains(&pre),
                "pre-built facet {pre:?} must be reused under its own key"
            );
        }

        // No duplicate face is minted on the reused footprints: the total face
        // count is exactly the two pre-built facets plus the built remainder.
        assert_eq!(
            mesh.face_count(),
            faces_before + result.faces().len(),
            "every new face is genuinely new — no duplicate minted over a reused facet"
        );

        // The two reused facets are not double-counted as built.
        let built: std::collections::HashSet<FaceKey> =
            result.faces().iter().copied().collect();
        for &pre in &pre_faces {
            assert!(!built.contains(&pre), "a reused facet is not also a built face");
        }

        // Every built face is outward-wound (normal points away from the
        // origin centroid), deterministically.
        let center = centroid(&mesh, &input);
        let built_faces: Vec<FaceKey> = result.faces().to_vec();
        for face in built_faces {
            mesh.face_normal_update(face);
            let normal = mesh.face_normal(face);
            let face_point =
                mesh.vertex_position(mesh.loop_vertex(mesh.face_first_loop(face)));
            assert!(
                normal.dot(face_point - center) > 0.0,
                "built hull face {face:?} must be wound outward"
            );
        }
        mesh.validate_topology();
    }

    /// Register a per-face and a per-loop i32 layer, returning their attribute
    /// handles. The convex_hull glob (`use crate::loopmesh::*`) re-exports the
    /// attribute vocabulary.
    fn register_cd_layers(mesh: &mut LoopMesh) -> (ElementAttributeKey, ElementAttributeKey) {
        let face_layer = mesh
            .register_face_attribute(ElementAttributeDesc::numeric(
                AttributeKey::custom("face_tag"),
                AttrType::I32,
            ))
            .unwrap();
        let loop_layer = mesh
            .register_loop_attribute(ElementAttributeDesc::numeric(
                AttributeKey::custom("loop_tag"),
                AttrType::I32,
            ))
            .unwrap();
        (face_layer, loop_layer)
    }

    #[test]
    fn fresh_hull_face_inherits_neighbour_customdata_verbatim() {
        // §6.1 / §6 step 3 — a fresh hull triangle that borders a pre-existing
        // neighbour face across a *reused* edge inherits that neighbour's
        // per-face customdata (the §6.1 example) AND its per-loop customdata via
        // the vertex-matched `BM_face_copy_shared` pass (a copy, not an
        // interpolation). Only the two corners on the shared edge are filled; the
        // third (unshared) corner stays at default.
        let mut mesh = LoopMesh::new();
        let (face_layer, loop_layer) = register_cd_layers(&mut mesh);

        let input = cloud(&mut mesh, &tetra_positions());
        // A non-input vertex outside the hull, sharing the hull edge
        // (input[0], input[1]) via a neighbour face. That edge bounds a tetra
        // facet, so the hull's surgery reuses it and the neighbour is found
        // across it.
        let outsider = mesh.create_vertex(Vec3::new(2.0, 2.0, 0.0));
        let neighbour =
            mesh.face_create_verts(&[input[0], input[1], outsider], FaceCreateFlags::NO_DUPLICATE);

        // Distinctive customdata on the neighbour: a face-layer value and a
        // single loop-layer value on every corner (so the per-loop copy is
        // unambiguous whichever corner is the verbatim source).
        const FACE_VALUE: i32 = 4242;
        const LOOP_VALUE: i32 = 777;
        assert!(mesh
            .face_attribs_mut()
            .set_i32(face_layer, neighbour.as_usize(), FACE_VALUE));
        for loop_key in mesh.face_loop_keys(neighbour) {
            assert!(mesh
                .loop_attribs_mut()
                .set_i32(loop_layer, loop_key.as_usize(), LOOP_VALUE));
        }

        let result = mesh.convex_hull_with_input(&input, &[], &[], false);

        // Find the built hull face spanning {input[0], input[1], input[2]} —
        // the facet that shares the reused (input[0], input[1]) edge with the
        // neighbour.
        let seeded = result
            .faces()
            .iter()
            .copied()
            .find(|&f| {
                let vs = mesh.face_vert_keys(f);
                vs.contains(&input[0]) && vs.contains(&input[1])
            })
            .expect("a hull facet must use the reused (input[0], input[1]) edge");

        // Per-face customdata copied verbatim from the example face.
        assert_eq!(
            mesh.face_attribs().get_i32(face_layer, seeded.as_usize()),
            Some(FACE_VALUE),
            "the fresh hull face inherits the neighbour's per-face customdata verbatim"
        );

        // Per-loop customdata: the two corners on the shared (input[0], input[1])
        // edge inherit the neighbour's value; the unshared corner stays default.
        for loop_key in mesh.face_loop_keys(seeded) {
            let v = mesh.loop_vertex(loop_key);
            let got = mesh.loop_attribs().get_i32(loop_layer, loop_key.as_usize());
            if v == input[0] || v == input[1] {
                assert_eq!(got, Some(LOOP_VALUE), "shared-edge corner inherits per-loop CD");
            } else {
                assert_eq!(got, Some(0), "the unshared corner keeps default per-loop CD");
            }
        }
        mesh.validate_topology();
    }

    #[test]
    fn shared_corner_data_is_vertex_matched_not_broadcast() {
        // §6 step 3 (the per-loop `BM_face_copy_shared` pass) — a neighbour with
        // **different** values on its different corners. A vertex-matched copy and
        // a broadcast-from-first-corner give different results; assert the
        // triangle's corners receive the **vertex-matched** values.
        //
        // The neighbour spans (input[0], input[1], outsider) and carries a
        // distinct per-corner value at each of input[0] / input[1]. The hull
        // facet sharing the (input[0], input[1]) edge must receive the
        // neighbour's value *at the matching vertex* on each of those two
        // corners.
        let mut mesh = LoopMesh::new();
        let (_face_layer, loop_layer) = register_cd_layers(&mut mesh);

        let input = cloud(&mut mesh, &tetra_positions());
        let outsider = mesh.create_vertex(Vec3::new(2.0, 2.0, 0.0));
        let neighbour =
            mesh.face_create_verts(&[input[0], input[1], outsider], FaceCreateFlags::NO_DUPLICATE);

        // Distinct value per neighbour corner, keyed by which vertex the corner
        // sits at.
        const VALUE_AT_0: i32 = 11;
        const VALUE_AT_1: i32 = 22;
        const VALUE_AT_OUTSIDER: i32 = 33;
        for loop_key in mesh.face_loop_keys(neighbour) {
            let v = mesh.loop_vertex(loop_key);
            let value = if v == input[0] {
                VALUE_AT_0
            } else if v == input[1] {
                VALUE_AT_1
            } else {
                VALUE_AT_OUTSIDER
            };
            assert!(mesh
                .loop_attribs_mut()
                .set_i32(loop_layer, loop_key.as_usize(), value));
        }

        let result = mesh.convex_hull_with_input(&input, &[], &[], false);

        let seeded = result
            .faces()
            .iter()
            .copied()
            .find(|&f| {
                let vs = mesh.face_vert_keys(f);
                vs.contains(&input[0]) && vs.contains(&input[1])
            })
            .expect("a hull facet must use the reused (input[0], input[1]) edge");

        // The corner at input[0] gets VALUE_AT_0, the corner at input[1] gets
        // VALUE_AT_1 — a broadcast-from-first-corner would put the *same* value on
        // both. The third (unshared) corner stays default.
        for loop_key in mesh.face_loop_keys(seeded) {
            let v = mesh.loop_vertex(loop_key);
            let got = mesh.loop_attribs().get_i32(loop_layer, loop_key.as_usize());
            if v == input[0] {
                assert_eq!(got, Some(VALUE_AT_0), "corner at input[0] gets the matching value");
            } else if v == input[1] {
                assert_eq!(got, Some(VALUE_AT_1), "corner at input[1] gets the matching value");
            } else {
                assert_eq!(got, Some(0), "the unshared corner keeps default per-loop CD");
            }
        }
        mesh.validate_topology();
    }

    #[test]
    fn shared_corner_write_once_first_edge_in_loop_order_wins() {
        // §6 step 3 write-once tie-break — a triangle corner shared by **two**
        // edges whose neighbours carry **different** loop CD. The earlier-visited
        // edge in loop-iteration order supplies that corner; the later edge's
        // write to the same corner is dropped (its other endpoint still lands).
        //
        // Fixture: a flat hull over four coplanar square corners. The square
        // facet fans into two triangles. We surround the patch with two neighbour
        // faces across two adjacent boundary edges that meet at one shared corner,
        // each carrying a distinct per-loop value — so exactly one hull-facet
        // corner is contested by two differently-attributed neighbours.
        let mut mesh = LoopMesh::new();
        let (_face_layer, loop_layer) = register_cd_layers(&mut mesh);

        // A square in the z = 0 plane.
        let input = cloud(
            &mut mesh,
            &[
                Vec3::new(-1.0, -1.0, 0.0), // 0
                Vec3::new(1.0, -1.0, 0.0),  // 1
                Vec3::new(1.0, 1.0, 0.0),   // 2
                Vec3::new(-1.0, 1.0, 0.0),  // 3
            ],
        );

        // Two neighbour faces across two square edges that meet at corner
        // input[1]: one across (input[0], input[1]), one across (input[1],
        // input[2]). Both carry a distinct per-corner value at input[1].
        let out_a = mesh.create_vertex(Vec3::new(0.0, -3.0, 0.0));
        let out_b = mesh.create_vertex(Vec3::new(3.0, 0.0, 0.0));
        let neighbour_a =
            mesh.face_create_verts(&[input[0], input[1], out_a], FaceCreateFlags::NO_DUPLICATE);
        let neighbour_b =
            mesh.face_create_verts(&[input[1], input[2], out_b], FaceCreateFlags::NO_DUPLICATE);

        const VALUE_A_AT_1: i32 = 100;
        const VALUE_B_AT_1: i32 = 200;
        for loop_key in mesh.face_loop_keys(neighbour_a) {
            if mesh.loop_vertex(loop_key) == input[1] {
                assert!(mesh
                    .loop_attribs_mut()
                    .set_i32(loop_layer, loop_key.as_usize(), VALUE_A_AT_1));
            }
        }
        for loop_key in mesh.face_loop_keys(neighbour_b) {
            if mesh.loop_vertex(loop_key) == input[1] {
                assert!(mesh
                    .loop_attribs_mut()
                    .set_i32(loop_layer, loop_key.as_usize(), VALUE_B_AT_1));
            }
        }

        let result = mesh.convex_hull(&input);

        // The hull facet that owns the contested corner input[1] sits on both the
        // (input[0], input[1]) and (input[1], input[2]) edges (a fan apex/spoke).
        let facet = result
            .faces()
            .iter()
            .copied()
            .find(|&f| {
                let vs = mesh.face_vert_keys(f);
                vs.contains(&input[0]) && vs.contains(&input[1]) && vs.contains(&input[2])
            })
            .expect("a fan triangle must span input[0], input[1], input[2]");

        // Read the contested corner's value.
        let mut contested: Option<i32> = None;
        for loop_key in mesh.face_loop_keys(facet) {
            if mesh.loop_vertex(loop_key) == input[1] {
                contested = mesh.loop_attribs().get_i32(loop_layer, loop_key.as_usize());
            }
        }
        let contested = contested.expect("corner at input[1] is present");
        assert!(
            contested == VALUE_A_AT_1 || contested == VALUE_B_AT_1,
            "the contested corner takes exactly one neighbour's value, not a blend or default"
        );

        // Decide independently which shared edge is visited **first** in the
        // facet's loop order — the write-once precedence claim. A corner is
        // written when its containing shared edge is reached; the loop walk visits
        // edges in `face_loop_keys` order, so the first of the two contested edges
        // to appear supplies the contested corner.
        let mut expected: Option<i32> = None;
        for loop_key in mesh.face_loop_keys(facet) {
            let u = mesh.loop_vertex(loop_key);
            let w = mesh.loop_vertex(mesh.loop_next(loop_key));
            let edge = std::collections::HashSet::from([u, w]);
            if edge == std::collections::HashSet::from([input[0], input[1]]) {
                expected = Some(VALUE_A_AT_1);
                break;
            }
            if edge == std::collections::HashSet::from([input[1], input[2]]) {
                expected = Some(VALUE_B_AT_1);
                break;
            }
        }
        let expected = expected.expect("the facet visits one of the two contested edges first");
        assert_eq!(
            contested, expected,
            "the first shared edge in loop order wins the contested corner — \
             the later edge's write is dropped (not last-writer-wins, not a blend)"
        );
        mesh.validate_topology();
    }

    #[test]
    fn fresh_hull_face_with_no_neighbour_gets_default_customdata() {
        // §6.1 — a hull over a pure loose-vert cloud has no non-input neighbour
        // across any edge, so every fresh face keeps DEFAULT customdata (not some
        // distinctive value). Pinned by registering a layer (default 0) and
        // asserting the hull faces stay at the default.
        let mut mesh = LoopMesh::new();
        let (face_layer, loop_layer) = register_cd_layers(&mut mesh);

        let input = cloud(&mut mesh, &tetra_positions());
        let result = mesh.convex_hull(&input);

        assert_eq!(result.faces().len(), 4, "the loose tetra cloud builds four faces");
        for &face in result.faces() {
            assert_eq!(
                mesh.face_attribs().get_i32(face_layer, face.as_usize()),
                Some(0),
                "a face with no neighbour keeps default per-face customdata"
            );
            for loop_key in mesh.face_loop_keys(face) {
                assert_eq!(
                    mesh.loop_attribs().get_i32(loop_layer, loop_key.as_usize()),
                    Some(0),
                    "a face with no neighbour keeps default per-loop customdata"
                );
            }
        }
        mesh.validate_topology();
    }

    #[test]
    fn input_neighbour_face_is_eligible_as_example() {
        // §6.1 (corrected) — an **input** face IS eligible as the per-face
        // example (the criterion accepts any face that is an input face OR not
        // yet flagged a hull-output face). Here the only neighbour across the
        // reused hull edge (input[0], input[1]) is an input face, so the fresh
        // hull facet inherits its per-face customdata (and, since the same face
        // is the shared-edge neighbour, its per-loop customdata via
        // `BM_face_copy_shared`).
        let mut mesh = LoopMesh::new();
        let (face_layer, loop_layer) = register_cd_layers(&mut mesh);

        let input = cloud(&mut mesh, &tetra_positions());
        let outsider = mesh.create_vertex(Vec3::new(2.0, 2.0, 0.0));
        // A neighbour across the hull edge (input[0], input[1]) — and it IS an
        // input face this time. The corrected criterion accepts it.
        let input_neighbour =
            mesh.face_create_verts(&[input[0], input[1], outsider], FaceCreateFlags::NO_DUPLICATE);

        const FACE_VALUE: i32 = 4242;
        const LOOP_VALUE: i32 = 777;
        assert!(mesh
            .face_attribs_mut()
            .set_i32(face_layer, input_neighbour.as_usize(), FACE_VALUE));
        for loop_key in mesh.face_loop_keys(input_neighbour) {
            assert!(mesh
                .loop_attribs_mut()
                .set_i32(loop_layer, loop_key.as_usize(), LOOP_VALUE));
        }

        // Pass the neighbour as an input face — it must still be eligible as the
        // example (input faces are NOT rejected, per the corrected §6.1).
        let result = mesh.convex_hull_with_input(&input, &[], &[input_neighbour], false);

        let candidate = result
            .faces()
            .iter()
            .copied()
            .find(|&f| {
                let vs = mesh.face_vert_keys(f);
                vs.contains(&input[0]) && vs.contains(&input[1])
            })
            .expect("a hull facet must use the (input[0], input[1]) edge");

        // Per-face customdata inherited from the input example face.
        assert_eq!(
            mesh.face_attribs().get_i32(face_layer, candidate.as_usize()),
            Some(FACE_VALUE),
            "an input neighbour IS chosen as the example face — per-face CD inherited"
        );
        // Per-loop: the two corners on the shared (input[0], input[1]) edge get
        // the neighbour's loop values; the third (unshared) corner stays default.
        for loop_key in mesh.face_loop_keys(candidate) {
            let v = mesh.loop_vertex(loop_key);
            let got = mesh.loop_attribs().get_i32(loop_layer, loop_key.as_usize());
            if v == input[0] || v == input[1] {
                assert_eq!(got, Some(LOOP_VALUE), "shared-edge corner inherits per-loop CD");
            } else {
                assert_eq!(got, Some(0), "the unshared corner keeps default per-loop CD");
            }
        }
        mesh.validate_topology();
    }

    #[test]
    fn convex_hull_default_leaves_holes_empty() {
        // The plain verts-only entrypoint delegates with use_existing_faces =
        // false, so geom_holes.out is always empty for it.
        let mut mesh = LoopMesh::new();
        let input = cloud(&mut mesh, &cube_corners());
        let result = mesh.convex_hull(&input);
        assert!(result.geom_holes_out().is_empty(), "default path has no holes");
        mesh.validate_topology();
    }

    // ---- Provider seam ----
    //
    // The operator must consume the *injected* provider's topology verbatim —
    // its facets, its loop order, its winding — and never recompute a hull of
    // its own. These tests pin that with a stub provider that hands back a
    // hand-authored topology deliberately unlike anything `Quickhull` would
    // return for the same cloud.

    /// A provider that ignores the point cloud and replays a fixed topology.
    struct FixedHull {
        vertices: Vec<usize>,
        faces: Vec<Vec<usize>>,
    }

    impl ConvexHullProvider for FixedHull {
        fn convex_hull(&self, _points: &[glam::DVec3]) -> Option<HullTopology> {
            Some(HullTopology {
                vertices: self.vertices.clone(),
                faces: self
                    .faces
                    .iter()
                    .map(|f| HullFace { vertices: f.clone() })
                    .collect(),
            })
        }
    }

    /// The six outward-wound quad facets of `cube_corners()`, each loop rotated
    /// left by `rotation` — so the fan apex (`corners[0]`) and hence each
    /// facet's diagonal change with `rotation`, while the hull itself does not.
    fn cube_quad_facets(rotation: usize) -> Vec<Vec<usize>> {
        [
            [0, 1, 3, 2],
            [4, 6, 7, 5],
            [0, 4, 5, 1],
            [2, 3, 7, 6],
            [0, 2, 6, 4],
            [1, 5, 7, 3],
        ]
        .iter()
        .map(|quad| (0..4).map(|i| quad[(i + rotation) % 4]).collect())
        .collect()
    }

    #[test]
    fn injected_provider_topology_is_used_instead_of_quickhull() {
        // The cloud is a cube (Quickhull: 8 on-hull verts, 12 triangles, 18
        // edges), but the injected provider reports only the inscribed
        // tetrahedron over corners 0/3/5/6. The operator must build *that* —
        // proving it never recomputes a hull of its own.
        let mut mesh = LoopMesh::new();
        let input = cloud(&mut mesh, &cube_corners());

        let provider = FixedHull {
            vertices: vec![0, 3, 5, 6],
            faces: vec![vec![0, 5, 3], vec![0, 6, 5], vec![0, 3, 6], vec![3, 5, 6]],
        };
        let result = mesh.convex_hull_with_provider(&input, &[], &[], false, &provider);

        assert_eq!(result.faces().len(), 4, "the provider's four facets, not 12");
        assert_eq!(result.edges().len(), 6, "the provider's tetrahedron, not 18");
        assert_eq!(
            result.verts(),
            &[input[0], input[3], input[5], input[6]],
            "geom.out's verts are the provider's on-hull set, in the contract's \
             ascending-index order"
        );

        // The four corners the provider omitted are classified interior + unused,
        // even though they are geometrically extremal.
        assert_eq!(
            vert_set(result.interior_verts()),
            vert_set(&[input[1], input[2], input[4], input[7]]),
            "corners the provider left off the hull are interior"
        );
        assert_eq!(vert_set(result.unused_verts()), vert_set(result.interior_verts()));

        // The provider's winding is carried verbatim into the built faces.
        assert_closed_hull_invariants(
            &mut mesh,
            &result,
            &[input[0], input[3], input[5], input[6]],
        );
        mesh.validate_topology();
    }

    #[test]
    fn injected_provider_loop_rotation_drives_the_fan_diagonal() {
        // Same six quad facets, two different loop rotations. The operator fans
        // each facet from *its own* first vertex, so the diagonal of every facet
        // must follow the provider's rotation — the property T2d's adapter leans
        // on when it passes the foreign engine's loops through untouched.
        let expected_diagonals = |rotation: usize| -> Vec<[usize; 2]> {
            cube_quad_facets(rotation)
                .iter()
                .map(|quad| {
                    let (a, b) = (quad[0], quad[2]);
                    if a <= b { [a, b] } else { [b, a] }
                })
                .collect()
        };

        for rotation in [0usize, 1] {
            let mut mesh = LoopMesh::new();
            let input = cloud(&mut mesh, &cube_corners());
            let provider = FixedHull {
                vertices: (0..8).collect(),
                faces: cube_quad_facets(rotation),
            };

            let result = mesh.convex_hull_with_provider(&input, &[], &[], false, &provider);

            assert_eq!(result.faces().len(), 12, "six quads fan to twelve triangles");
            assert_eq!(result.edges().len(), 18, "12 cube edges + 6 facet diagonals");

            let present = |a: usize, b: usize| mesh.query_edge_exists(input[a], input[b]).is_some();
            for [a, b] in expected_diagonals(rotation) {
                assert!(
                    present(a, b),
                    "rotation {rotation}: the facet's apex diagonal {a}-{b} must be built"
                );
            }
            // The *other* diagonal of each facet — the one a different rotation
            // would have produced — must be absent.
            for [a, b] in expected_diagonals(rotation + 1) {
                assert!(
                    !present(a, b),
                    "rotation {rotation}: diagonal {a}-{b} belongs to another rotation"
                );
            }

            assert_closed_hull_invariants(&mut mesh, &result, &input);
            mesh.validate_topology();
        }
    }

    // ---- Provider contract (debug-only guard rail) ----
    //
    // Every clause below is a silent corrupter if it slips through, so the
    // operator audits the injected topology under `debug_assert`. These tests
    // only apply where that audit is compiled in.

    #[cfg(debug_assertions)]
    fn hull_cube_with(provider: FixedHull) {
        let mut mesh = LoopMesh::new();
        let input = cloud(&mut mesh, &cube_corners());
        mesh.convex_hull_with_provider(&input, &[], &[], false, &provider);
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "which `vertices` omits")]
    fn provider_vertices_omitting_a_facet_index_is_rejected() {
        // Index 6 is built into a face yet reported in no output slot.
        hull_cube_with(FixedHull {
            vertices: vec![0, 3, 5],
            faces: vec![vec![0, 5, 3], vec![0, 6, 5], vec![0, 3, 6], vec![3, 5, 6]],
        });
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "strictly ascending")]
    fn provider_vertices_out_of_order_or_duplicated_is_rejected() {
        hull_cube_with(FixedHull {
            vertices: vec![0, 5, 3, 6],
            faces: vec![vec![0, 5, 3], vec![0, 6, 5], vec![0, 3, 6], vec![3, 5, 6]],
        });
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "`vertices` lists point index 1, which no facet uses")]
    fn provider_vertices_listing_an_unused_index_is_rejected() {
        // All eight corners reported on-hull, but only the inscribed tetrahedron
        // built — 1/2/4/7 would be emitted in geom.out yet belong to no face.
        hull_cube_with(FixedHull {
            vertices: (0..8).collect(),
            faces: vec![vec![0, 5, 3], vec![0, 6, 5], vec![0, 3, 6], vec![3, 5, 6]],
        });
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "facet 1 references out-of-range point index 9")]
    fn provider_facet_index_out_of_range_is_rejected() {
        // `vertices` itself stays in range, so the facet loop's own range check is
        // the clause under test (it runs before the superset check, which index 9
        // would also trip).
        hull_cube_with(FixedHull {
            vertices: vec![0, 3, 5],
            faces: vec![vec![0, 5, 3], vec![0, 9, 5]],
        });
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "`vertices` lists out-of-range point index 9")]
    fn provider_vertices_index_out_of_range_is_rejected() {
        hull_cube_with(FixedHull {
            vertices: vec![0, 3, 5, 9],
            faces: vec![vec![0, 5, 3], vec![0, 9, 5], vec![0, 3, 9], vec![3, 5, 9]],
        });
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "facet 1 has too few corners (2)")]
    fn provider_facet_with_fewer_than_three_corners_is_rejected() {
        // A two-corner loop fans to no triangle at all, so the facet would vanish
        // from the built hull without a word.
        hull_cube_with(FixedHull {
            vertices: vec![0, 3, 5, 6],
            faces: vec![vec![0, 5, 3], vec![0, 6], vec![0, 3, 6], vec![3, 5, 6]],
        });
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "facet-less hull is the 1D (collinear) seam")]
    fn provider_losing_its_facets_is_not_mistaken_for_a_collinear_hull() {
        // The 3D cloud's eight corners with no faces at all — the shape a raw
        // adapter takes when the foreign engine hands back zero facets. Without
        // the guard this is indistinguishable from a valid 1D hull.
        hull_cube_with(FixedHull {
            vertices: (0..8).collect(),
            faces: vec![],
        });
    }

    #[test]
    fn quickhull_provider_matches_the_default_entry_points() {
        // Delegating the default path through the seam must be behaviour-neutral:
        // an explicit `&Quickhull` injection reproduces `convex_hull` exactly.
        let mut direct = LoopMesh::new();
        let direct_input = cloud(&mut direct, &cube_corners());
        let direct_result = direct.convex_hull(&direct_input);

        let mut injected = LoopMesh::new();
        let injected_input = cloud(&mut injected, &cube_corners());
        let injected_result =
            injected.convex_hull_with_provider(&injected_input, &[], &[], false, &Quickhull);

        assert_eq!(direct_result.verts(), injected_result.verts());
        assert_eq!(direct_result.edges(), injected_result.edges());
        assert_eq!(direct_result.faces(), injected_result.faces());
        assert_eq!(direct.face_count(), injected.face_count());
        assert_eq!(direct.edge_count(), injected.edge_count());
    }

    #[test]
    fn quickhull_provider_matches_the_default_entry_points_with_existing_faces() {
        // The same delegation neutrality on the branch carrying the most state:
        // the partial-patch bipyramid with use_existing_faces = true, which
        // exercises reuse capture (§5 step 1), the hole-cleanup pass, the built
        // remainder, and the interior/unused classification over input faces.
        // `convex_hull_with_faces` and an explicit `&Quickhull` injection must
        // agree slot for slot.
        let build = |mesh: &mut LoopMesh| -> (Vec<VertexKey>, [FaceKey; 2]) {
            let input = cloud(mesh, &bipyramid_positions());
            let upper_a = mesh.face_create_verts(
                &[input[0], input[1], input[3]],
                FaceCreateFlags::NO_DUPLICATE,
            );
            let upper_b = mesh.face_create_verts(
                &[input[1], input[2], input[3]],
                FaceCreateFlags::NO_DUPLICATE,
            );
            (input, [upper_a, upper_b])
        };

        let mut direct = LoopMesh::new();
        let (direct_input, direct_faces) = build(&mut direct);
        let direct_result = direct.convex_hull_with_faces(&direct_input, &direct_faces, true);

        let mut injected = LoopMesh::new();
        let (injected_input, injected_faces) = build(&mut injected);
        let injected_result = injected.convex_hull_with_provider(
            &injected_input,
            &[],
            &injected_faces,
            true,
            &Quickhull,
        );

        assert!(!direct_result.hole_faces().is_empty(), "the fixture must reuse faces");
        assert!(!direct_result.faces().is_empty(), "the fixture must also build faces");

        assert_eq!(direct_result.verts(), injected_result.verts());
        assert_eq!(direct_result.edges(), injected_result.edges());
        assert_eq!(direct_result.faces(), injected_result.faces());
        assert_eq!(direct_result.hole_faces(), injected_result.hole_faces());
        assert_eq!(direct_result.hole_edges(), injected_result.hole_edges());
        assert_eq!(direct_result.interior_verts(), injected_result.interior_verts());
        assert_eq!(direct_result.interior_edges(), injected_result.interior_edges());
        assert_eq!(direct_result.interior_faces(), injected_result.interior_faces());
        assert_eq!(direct_result.unused_verts(), injected_result.unused_verts());
        assert_eq!(direct_result.unused_edges(), injected_result.unused_edges());
        assert_eq!(direct_result.unused_faces(), injected_result.unused_faces());
        assert_eq!(direct.face_count(), injected.face_count());
        assert_eq!(direct.edge_count(), injected.edge_count());
        direct.validate_topology();
        injected.validate_topology();
    }
}
