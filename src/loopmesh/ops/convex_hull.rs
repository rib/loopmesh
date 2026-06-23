use std::collections::HashSet;

use glam::DVec3;

use crate::loopmesh::*;

/// The `geom.out` of [`LoopMesh::convex_hull`]: every vertex, edge, and face
/// that lies **on the built hull surface** — the on-hull input vertices, the
/// hull edges, and the materialised hull faces (CONVEX_HULL.md §1, §4.3).
///
/// This cycle implements the additive surgery for a verts-only input cloud
/// with no face reuse (`use_existing_faces = false`), so `geom_out` is exactly
/// the freshly-built hull surface. The `geom_interior.out` / `geom_unused.out`
/// **vertex** classification (CONVEX_HULL.md §4, §4.1, §7) is populated; the
/// edge / face classification and the holes slot (CONVEX_HULL.md §4.2, §5)
/// remain unpopulated (mixed edge/face input is not yet accepted).
#[derive(Clone, Debug, Default)]
pub struct ConvexHullResult {
    verts: Vec<VertexKey>,
    edges: Vec<EdgeKey>,
    faces: Vec<FaceKey>,
    interior_verts: Vec<VertexKey>,
    unused_verts: Vec<VertexKey>,
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
    /// (CONVEX_HULL.md §4, §4.1). Verts-only this cycle, so input edges/faces
    /// are not classified.
    pub fn interior_verts(&self) -> &[VertexKey] {
        &self.interior_verts
    }

    /// The `geom_interior.out` slot. Verts-only this cycle: contains only the
    /// interior vertices (CONVEX_HULL.md §4); interior edge/face classification
    /// is not yet implemented.
    pub fn geom_interior_out(&self) -> Vec<GeomKey> {
        self.interior_verts.iter().copied().map(GeomKey::from).collect()
    }

    /// Interior vertices that are *unused* — safe to delete because every
    /// incident edge and face is itself an input element (CONVEX_HULL.md §7).
    /// For a verts-only cloud the buried verts are loose, so they qualify.
    pub fn unused_verts(&self) -> &[VertexKey] {
        &self.unused_verts
    }

    /// The `geom_unused.out` slot. Verts-only this cycle: contains only the
    /// unused interior vertices (CONVEX_HULL.md §7); interior edge/face
    /// classification is not yet implemented.
    pub fn geom_unused_out(&self) -> Vec<GeomKey> {
        self.unused_verts.iter().copied().map(GeomKey::from).collect()
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
    /// **Scope (this cycle):** any vertex cloud (full-3D, coplanar, collinear,
    /// or coincident) with `use_existing_faces = false` and no input faces. The
    /// result reports `geom.out` (the built on-hull surface, CONVEX_HULL.md
    /// §4.3) and the **vertex** half of the `geom_interior.out` /
    /// `geom_unused.out` classification (CONVEX_HULL.md §4, §4.1, §7): every
    /// input vert not on the hull is interior, and an interior vert is unused
    /// when all its incident elements are themselves input. Input edge/face
    /// classification (§4.2, §7) and the customdata example-face seed (§6.1) are
    /// not yet implemented. For the `use_existing_faces` reuse path (§5) and the
    /// `geom_holes.out` slot, see [`Self::convex_hull_with_faces`].
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
        self.convex_hull_with_faces(input, &[], false)
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
    /// **Deferred (still 📝):** input edge/face interior/unused classification
    /// (§4.2, §7) — `input_faces` are not yet routed into `geom_interior.out` /
    /// `geom_unused.out`, only into the reuse/holes path. The customdata
    /// example-face seed (§6.1) is also not yet implemented.
    pub fn convex_hull_with_faces(
        &mut self,
        verts: &[VertexKey],
        input_faces: &[FaceKey],
        use_existing_faces: bool,
    ) -> ConvexHullResult {
        let mut result = ConvexHullResult::default();

        // §1.2 — minimum-input guard: a count, not a non-degeneracy test.
        if verts.len() < 3 {
            return result;
        }

        // §1.1 — only the verts drive the cloud. Map each input vertex's
        // position into the provider's f64 cloud; the cloud index is the input
        // slot, so an on-hull index maps straight back to `verts[index]` with no
        // remapping and no positional drift (§2).
        let points: Vec<DVec3> = verts
            .iter()
            .map(|&v| self.vertex_position(v).as_dvec3())
            .collect();

        let Some(topology) = Quickhull.convex_hull(&points) else {
            return result;
        };

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
            let face = self.face_create_verts(&[a, b, c], FaceCreateFlags::NO_DUPLICATE);

            // §4.3 / §6 step 4 — flag the face, its edges, and its verts
            // on-hull as they are materialised.
            if seen_faces.insert(face) {
                result.faces.push(face);
            }
            for vert in [a, b, c] {
                on_hull_verts.insert(vert);
            }
            for edge in self.face_edge_keys(face) {
                on_hull_edges.insert(edge);
            }
        }

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

        // §4 / §4.1 / §7 — classify input vertices. Every input vert is
        // optimistically interior; a vert that turned out to be a hull corner has
        // its tag cleared. The remaining interior verts are the buried ones: the
        // culled coplanar-interior point, collinear intermediates, a coincident
        // duplicate's key, and any 3D buried point.
        let mut seen_interior: HashSet<VertexKey> = HashSet::new();
        for &vert in verts {
            if on_hull_verts.contains(&vert) || !seen_interior.insert(vert) {
                continue;
            }
            result.interior_verts.push(vert);

            // §7 — verts-only shortcut: a buried vert with no incident geometry
            // is unused. §7's full test ("every incident edge is an input edge
            // AND every incident face is an input face") is deferred to the
            // mixed-input cycle; for a verts-only cloud the input edge/face sets
            // are empty, so a loose buried vert is the only thing that qualifies.
            if self.vertex_is_loose(vert) {
                result.unused_verts.push(vert);
            }
        }

        result
    }

    /// Verts-only shortcut for CONVEX_HULL.md §7's unused test: a buried vert
    /// with no incident geometry is unused. §7's full rule is "every incident
    /// edge is an *input* edge AND every incident face is an *input* face"; for
    /// a verts-only cloud the input edge/face sets are empty, so the only buried
    /// vert that can pass is a loose one. The general all-incident-input test is
    /// deferred to the mixed-input cycle — do *not* read this as that guard.
    fn vertex_is_loose(&self, vertex: VertexKey) -> bool {
        self.iter_disk_edges(vertex).next().is_none()
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
    fn convex_hull_default_leaves_holes_empty() {
        // The plain verts-only entrypoint delegates with use_existing_faces =
        // false, so geom_holes.out is always empty for it.
        let mut mesh = LoopMesh::new();
        let input = cloud(&mut mesh, &cube_corners());
        let result = mesh.convex_hull(&input);
        assert!(result.geom_holes_out().is_empty(), "default path has no holes");
        mesh.validate_topology();
    }
}
