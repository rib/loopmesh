use std::collections::HashSet;

use glam::DVec3;

use crate::loopmesh::*;

/// The `geom.out` of [`LoopMesh::convex_hull`]: every vertex, edge, and face
/// that lies **on the built hull surface** — the on-hull input vertices, the
/// hull edges, and the materialised hull faces (CONVEX_HULL.md §1, §4.3).
///
/// This cycle implements the additive surgery for a non-degenerate vertex
/// cloud with no face reuse (`use_existing_faces = false`), so `geom_out` is
/// exactly the freshly-built hull surface. The interior / unused / holes
/// classification slots (CONVEX_HULL.md §4–§7) are not yet populated.
#[derive(Clone, Debug, Default)]
pub struct ConvexHullResult {
    verts: Vec<VertexKey>,
    edges: Vec<EdgeKey>,
    faces: Vec<FaceKey>,
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
    /// or coincident) with `use_existing_faces = false`. The result reports only
    /// `geom.out` — the built on-hull surface (CONVEX_HULL.md §4.3). The
    /// interior / unused / holes classification (CONVEX_HULL.md §4–§7), the
    /// `use_existing_faces` reuse path (§5), and the customdata example-face
    /// seed (§6.1) are not yet implemented.
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
        let mut result = ConvexHullResult::default();

        // §1.2 — minimum-input guard: a count, not a non-degeneracy test.
        if input.len() < 3 {
            return result;
        }

        // §1.1 — only the verts drive the cloud. Map each input vertex's
        // position into the provider's f64 cloud; the cloud index is the input
        // slot, so an on-hull index maps straight back to `input[index]` with no
        // remapping and no positional drift (§2).
        let points: Vec<DVec3> = input
            .iter()
            .map(|&v| self.vertex_position(v).as_dvec3())
            .collect();

        let Some(topology) = Quickhull.convex_hull(&points) else {
            return result;
        };

        // §3.1 — fan each facet into outward-wound triangles, then run the §6
        // surgery per triangle. `face_create_verts` with NO_DUPLICATE performs
        // both the no-double edge creation (§6 step 1) and the face
        // find-or-create (§6 step 2) in one call.
        let mut on_hull_verts: HashSet<VertexKey> = HashSet::new();
        let mut on_hull_edges: HashSet<EdgeKey> = HashSet::new();
        let mut seen_faces: HashSet<FaceKey> = HashSet::new();

        for facet in &topology.faces {
            let corners = &facet.vertices;
            if corners.len() < 3 {
                continue;
            }
            let apex = input[corners[0]];
            for window in corners[1..].windows(2) {
                let a = apex;
                let b = input[window[0]];
                let c = input[window[1]];

                // §3.2 — the facet's CCW order is the face's winding, so the
                // built face's stored normal points outward.
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
        }

        // §2 / §8 — the on-hull vertex set is the extremal subset of the input,
        // emitted in the provider's ascending-index order so the slot is stable.
        //
        // When the provider built facets (the full-3D and coplanar flat-hull
        // paths), a hull vertex is on-hull iff it was a corner of a materialised
        // triangle (§4.3 / §6 step 4). When the provider built **no** facets, the
        // cloud is a 1D collinear hull (§8 collinear bullet): its two extreme
        // endpoints are still on-hull and reported as verts even though no face
        // bounds them. Coincident duplicates never reach here — the provider
        // deduplicates coincident positions before the hull runs, so a
        // non-representative duplicate is simply absent from `topology.vertices`
        // (§8 coincident bullet) and lands in the interior/unused slots instead.
        let built_no_faces = result.faces.is_empty();
        for &index in &topology.vertices {
            let vert = input[index];
            if built_no_faces || on_hull_verts.contains(&vert) {
                result.verts.push(vert);
            }
        }
        result.edges = on_hull_edges.into_iter().collect();

        result
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
}
