use std::collections::HashMap;

use glam::{Vec2, Vec3};

use crate::loopmesh::*;
use crate::triangulate::{axis_dominant_basis_negated, ear_clip};

/// Squared-length floor below which a normal (derived corner normal, accumulated
/// fill normal, or supplied `normal`) is treated as zero. Guards the projection
/// basis and the supplied-vs-derived choice against a degenerate-length vector.
const MIN_NORMAL_LEN_SQ: f32 = 1e-12;

/// Parameters for [`LoopMesh::triangle_fill`], mirroring the operator's BMOP
/// slot knobs.
///
/// Algorithm in
/// [`TRIANGLE_FILL.md`](../../../../radial-mesh-guides/operators/TRIANGLE_FILL.md)
/// §1.
#[derive(Copy, Clone, Debug, Default)]
pub struct TriangleFillParams {
    /// When set, request the beautify edge-rotation pass after the raw fill
    /// (§6). The beautify pass itself is **not yet applied** — see
    /// [`LoopMesh::triangle_fill`] — so setting this currently produces the
    /// un-beautified fill (a valid triangulation, just not Delaunay-rotated).
    pub use_beauty: bool,
    /// When set, dissolve the fill's new interior edges into larger n-gons
    /// after the fill (§7).
    pub use_dissolve: bool,
    /// Projection-plane normal (§1/§3). `None` (the default) derives a fill
    /// normal from the input geometry (§3); a supplied non-zero value is used
    /// directly as the projection plane and skips the §4 winding vote.
    pub normal: Option<Vec3>,
}

/// The `geom.out` of [`LoopMesh::triangle_fill`]: every **new** interior edge
/// and **new** fill face the operator created (§1). Input boundary edges are
/// never reported. After a dissolve pass the surviving joined faces and
/// surviving fill edges are reported.
#[derive(Clone, Debug, Default)]
pub struct TriangleFillResult {
    edges: Vec<EdgeKey>,
    faces: Vec<FaceKey>,
}

impl TriangleFillResult {
    pub fn edges(&self) -> &[EdgeKey] {
        &self.edges
    }
    pub fn faces(&self) -> &[FaceKey] {
        &self.faces
    }
    pub fn is_empty(&self) -> bool {
        self.edges.is_empty() && self.faces.is_empty()
    }
}

impl LoopMesh {
    /// Span the empty region bounded by an undirected set of `edges` with
    /// triangles (§1). The operator never creates a vertex (every corner is
    /// an existing input endpoint, §2) and never deletes an input edge.
    ///
    /// Algorithm in
    /// [`TRIANGLE_FILL.md`](../../../../radial-mesh-guides/operators/TRIANGLE_FILL.md):
    /// the fill is a planar triangulation of the bounded region (§1.1 — the
    /// contract is *any* valid triangulation, optionally beautified), the
    /// projection plane comes from the derived or supplied normal (§3), the
    /// winding agrees with any existing boundary faces (§4), each triangle is
    /// materialised with the no-double edge discipline (§5), and — when
    /// `params.use_dissolve` is set — the new interior edges are JFKE-joined
    /// back into n-gons (§7).
    ///
    /// A supplied non-zero `params.normal` is used directly as the projection
    /// plane and skips the §4 winding vote; the fill orients to that normal.
    ///
    /// **Scope (current):** the in-scope case is a single closed loop of
    /// input edges (the editor's "fill a selected edge loop" path). An edge
    /// set that does not resolve to exactly one simple closed loop — multiple
    /// disjoint loops, an annulus/hole, an open chain, or loose edges (§8) —
    /// fills nothing and returns an empty result. The multi-region sweep is
    /// not yet implemented.
    ///
    /// **`use_beauty`:** the beautify post-pass (§6) is not yet applied — the
    /// flag is accepted but ignored, producing the un-beautified (still valid)
    /// fill rather than aborting, since `use_beauty` is a caller-settable input.
    pub fn triangle_fill(
        &mut self,
        edges: &[EdgeKey],
        params: &TriangleFillParams,
    ) -> TriangleFillResult {
        let mut result = TriangleFillResult::default();

        // §8: fewer than three edges can bound no region.
        if edges.len() < 3 {
            return result;
        }

        let Some(loop_verts) = self.triangle_fill_order_single_loop(edges) else {
            // Not a single simple closed loop — the multi-region / hole /
            // open-chain sweep (§1.1/§8) is out of scope. Fill nothing.
            return result;
        };
        let vertex_count = loop_verts.len();
        if vertex_count < 3 {
            return result;
        }

        // §3 — projection-plane normal: supplied (authoritative) or derived.
        let (fill_normal, normal_was_derived) = match params.normal {
            Some(n) if n.length_squared() > MIN_NORMAL_LEN_SQ => (n.normalize(), false),
            _ => (self.triangle_fill_derive_normal(&loop_verts), true),
        };

        // Project onto the plane and triangulate. ear_clip produces *a* valid
        // triangulation of the simple polygon, which satisfies the §1.1
        // "any valid triangulation" contract; with use_dissolve the choice is
        // erased anyway (§7/§9 strategy 1).
        let (axis_u, axis_w) = axis_dominant_basis_negated(fill_normal);
        let projected: Vec<Vec2> = loop_verts
            .iter()
            .map(|&v| {
                let p = self.vertex_position(v);
                Vec2::new(axis_u.dot(p), axis_w.dot(p))
            })
            .collect();

        let mut tris: Vec<[u32; 3]> = Vec::with_capacity(vertex_count - 2);
        ear_clip(&projected, 0, &mut tris);
        if tris.is_empty() {
            // Collinear / degenerate projection (§8): nothing to fill.
            return result;
        }

        // §4 — winding vote: only when the normal was derived and some input
        // edge already carries a face (rim-of-a-hole). The sweep is already
        // consistently wound, so the decision is global.
        let flip = normal_was_derived
            && self.triangle_fill_winding_vote(&loop_verts, &tris);

        // §5 — materialise each triangle with no-double edge creation, marking
        // the new interior diagonals for geom.out and the dissolve pass.
        let mut interior_edges: Vec<EdgeKey> = Vec::new();
        let mut interior_seen: std::collections::HashSet<EdgeKey> = std::collections::HashSet::new();
        for tri in &tris {
            let (a, b, c) = if flip {
                (tri[0], tri[2], tri[1])
            } else {
                (tri[0], tri[1], tri[2])
            };
            let va = loop_verts[a as usize];
            let vb = loop_verts[b as usize];
            let vc = loop_verts[c as usize];

            // Track which of the three edges already existed before this
            // triangle (boundary edges always do; an interior diagonal might
            // already have been minted by a previous fill triangle). Only the
            // genuinely-fresh, non-boundary diagonals are interior fill edges.
            let face = self.face_create_verts(&[va, vb, vc], FaceCreateFlags::NO_DUPLICATE);
            result.faces.push(face);

            for edge in self.face_edge_keys(face) {
                if interior_seen.contains(&edge) {
                    continue;
                }
                // A diagonal is interior iff it is not one of the input
                // boundary edges. Input edges pre-existed and must survive
                // the dissolve pass (§5/§7).
                if !edges.contains(&edge) {
                    interior_seen.insert(edge);
                    interior_edges.push(edge);
                }
            }
        }

        if params.use_dissolve {
            // Seed the survivor set with *every* fill face just created, so a
            // face whose edges are never visited as a manifold interior edge
            // (e.g. the lone triangle of a 3-edge loop) is still reported in
            // geom.out (§1/§7). `result.faces` currently holds exactly those.
            let created_faces = std::mem::take(&mut result.faces);
            self.triangle_fill_dissolve(&interior_edges, created_faces, &mut result);
        } else {
            result.edges = interior_edges;
        }

        result
    }

    /// Order an undirected edge set into a single simple closed loop's vertex
    /// cycle, or `None` if the edges do not form exactly one such loop (every
    /// vertex must have degree exactly two within the set, and the set must be
    /// a single connected cycle covering all its edges).
    fn triangle_fill_order_single_loop(&self, edges: &[EdgeKey]) -> Option<Vec<VertexKey>> {
        let mut adjacency: HashMap<VertexKey, smallvec::SmallVec<[VertexKey; 2]>> = HashMap::new();
        for &edge in edges {
            let [v0, v1] = self.edge_vert_keys(edge);
            if v0 == v1 {
                return None;
            }
            adjacency.entry(v0).or_default().push(v1);
            adjacency.entry(v1).or_default().push(v0);
        }
        // A single simple cycle has every vertex at degree exactly two.
        if adjacency.values().any(|neighbors| neighbors.len() != 2) {
            return None;
        }

        let start = *adjacency.keys().next()?;
        let mut ordered: Vec<VertexKey> = Vec::with_capacity(adjacency.len());
        let mut current = start;
        let mut previous: Option<VertexKey> = None;
        loop {
            ordered.push(current);
            let neighbors = &adjacency[&current];
            let next = if Some(neighbors[0]) == previous {
                neighbors[1]
            } else {
                neighbors[0]
            };
            previous = Some(current);
            current = next;
            if current == start {
                break;
            }
            // Unreachable safety net: a degree-2 net traversed from `start`
            // either closes back to `start` within `adjacency.len()` steps or is
            // a disjoint cycle (rejected by the post-loop coverage check). This
            // guards against an unbounded walk should that invariant ever break.
            if ordered.len() > adjacency.len() {
                return None;
            }
        }
        // The walk closed back to `start` — reject if it did not cover every
        // vertex of the net (disjoint cycles).
        if ordered.len() != adjacency.len() {
            return None;
        }
        Some(ordered)
    }

    /// Derive a robust fill normal from the loop's corner turns (§3).
    fn triangle_fill_derive_normal(&self, loop_verts: &[VertexKey]) -> Vec3 {
        let n = loop_verts.len();

        // §3 step 1-2: per-corner cross product, sign-normalised by a fixed,
        // order-independent edge-length rule so the same corner always yields
        // the same-signed normal regardless of input edge order.
        let mut corner_normals: Vec<(f32, Vec3)> = Vec::with_capacity(n);
        for i in 0..n {
            let prev = self.vertex_position(loop_verts[(i + n - 1) % n]);
            let here = self.vertex_position(loop_verts[i]);
            let next = self.vertex_position(loop_verts[(i + 1) % n]);
            let incoming = here - prev;
            let outgoing = next - here;
            let mut corner = incoming.cross(outgoing);
            // Order-independent sign rule: flip so the corner normal is
            // oriented consistently for the unordered pair {incoming, outgoing}.
            // The tie-broken convention keys off the two edge lengths.
            let incoming_len = incoming.length_squared();
            let outgoing_len = outgoing.length_squared();
            if outgoing_len < incoming_len {
                corner = -corner;
            }
            corner_normals.push((corner.length_squared(), corner));
        }

        // §3 step 3: seed from the squarest corner, align the rest.
        corner_normals.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        let mut accumulated = Vec3::ZERO;
        for (magnitude, corner) in &corner_normals {
            if *magnitude <= MIN_NORMAL_LEN_SQ {
                continue;
            }
            let oriented = if accumulated.dot(*corner) < 0.0 {
                -*corner
            } else {
                *corner
            };
            accumulated += oriented;
        }

        // §3 step 4: no two-edge corner contributed — average stored normals.
        if accumulated.length_squared() <= MIN_NORMAL_LEN_SQ {
            for &v in loop_verts {
                accumulated += self.vertex_normal(v);
            }
        }

        // §3 step 5: final degenerate guard → +Z.
        if accumulated.length_squared() <= MIN_NORMAL_LEN_SQ {
            Vec3::Z
        } else {
            accumulated.normalize()
        }
    }

    /// §4 winding vote: returns true if the whole fill should be flipped to
    /// agree with the existing surface across the shared boundary edges.
    /// Fires only when some input edge already carries exactly one face.
    fn triangle_fill_winding_vote(
        &self,
        loop_verts: &[VertexKey],
        tris: &[[u32; 3]],
    ) -> bool {
        let mut tally: i32 = 0;
        for tri in tris {
            // Each directed corner pair (a→b) of the fill triangle.
            for &(a, b) in &[
                (tri[0], tri[1]),
                (tri[1], tri[2]),
                (tri[2], tri[0]),
            ] {
                let va = loop_verts[a as usize];
                let vb = loop_verts[b as usize];
                let Some(edge) = self.query_edge_exists(va, vb) else {
                    continue;
                };
                // Only boundary input edges (exactly one existing face) vote.
                let radial = self.edge_loop_keys(edge);
                if radial.len() != 1 {
                    continue;
                }
                let neighbour_loop = radial[0];
                // The neighbour traverses the edge starting at this loop's
                // vertex. Coherent normals require the fill to traverse the
                // shared edge in the *opposite* direction to the neighbour.
                let neighbour_start = self.loop_vertex(neighbour_loop);
                if neighbour_start == va {
                    // Fill and neighbour traverse va→vb the same way →
                    // incoherent → vote to flip.
                    tally += 1;
                } else {
                    tally -= 1;
                }
            }
        }
        tally > 0
    }

    /// §7 dissolve pass: JFKE-join the two fill triangles flanking each new
    /// manifold interior edge, removing the edge; report the surviving joined
    /// faces and any surviving fill edges in `geom.out`.
    fn triangle_fill_dissolve(
        &mut self,
        interior_edges: &[EdgeKey],
        created_faces: Vec<FaceKey>,
        result: &mut TriangleFillResult,
    ) {
        // Seed with every fill face created during materialisation (§5); each
        // join removes the two consumed faces and inserts the survivor, so a
        // face never touched by an interior-edge join (e.g. the single triangle
        // of a 3-edge loop) still survives to geom.out (§1/§7).
        let mut surviving_faces: std::collections::HashSet<FaceKey> =
            created_faces.into_iter().collect();
        for &edge in interior_edges {
            // The edge may already be gone (killed when an earlier neighbour
            // edge's join removed it) — skip stale keys.
            if self.edges.try_get(edge).is_none() {
                continue;
            }
            let radial = self.edge_loop_keys(edge);
            match radial.len() {
                2 => {
                    let face_a = self.loop_face(radial[0]);
                    let face_b = self.loop_face(radial[1]);
                    if face_a == face_b {
                        // Same-face filament — joining is undefined; leave it.
                        result.edges.push(edge);
                        continue;
                    }
                    if let Some(joined) = self.faces_join(&[face_a, face_b]) {
                        surviving_faces.remove(&face_a);
                        surviving_faces.remove(&face_b);
                        surviving_faces.insert(joined);
                    } else {
                        result.edges.push(edge);
                    }
                }
                0 => {
                    // Loose interior edge (degenerate net) — kill it (§7).
                    self.edge_kill(edge);
                }
                _ => {
                    // One or three-plus faces — joining undefined; leave it.
                    result.edges.push(edge);
                }
            }
        }
        result.faces = surviving_faces.into_iter().collect();
    }
}

#[cfg(test)]
mod tests {
    use std::f32::consts::TAU;

    use glam::Vec3;

    use super::TriangleFillParams;
    use crate::loopmesh::*;

    /// Build a free wire ring of `n` verts on a unit circle in the z=0 plane,
    /// returning the ring edges in arbitrary (set) order.
    fn convex_wire_ring(mesh: &mut LoopMesh, n: usize) -> Vec<EdgeKey> {
        let verts: Vec<VertexKey> = (0..n)
            .map(|i| {
                let theta = (i as f32) * TAU / (n as f32);
                mesh.create_vertex(Vec3::new(theta.cos(), theta.sin(), 0.0))
            })
            .collect();
        (0..n)
            .map(|i| mesh.create_edge([verts[i], verts[(i + 1) % n]], EdgeCreateFlags::NO_DUPLICATE))
            .collect()
    }

    #[test]
    fn convex_hexagon_dissolve_to_single_ngon() {
        let mut mesh = LoopMesh::new();
        let edges = convex_wire_ring(&mut mesh, 6);
        let result = mesh.triangle_fill(
            &edges,
            &TriangleFillParams {
                use_beauty: false,
                use_dissolve: true,
                normal: None,
            },
        );
        assert_eq!(mesh.vertex_count(), 6, "no vertex created");
        assert_eq!(mesh.edge_count(), 6, "only boundary edges survive");
        assert_eq!(mesh.face_count(), 1, "convex loop dissolves to 1 n-gon");
        let face = mesh.face_keys()[0];
        assert_eq!(mesh.face_len(face), 6, "the n-gon spans all six corners");
        assert_eq!(
            result.faces().len(),
            1,
            "geom.out reports the single surviving dissolved n-gon"
        );
        mesh.validate_topology();
    }

    #[test]
    fn triangle_loop_dissolve_reports_single_fill_face() {
        // A 3-edge loop fills to one triangle and has *zero* interior
        // diagonals, so the dissolve pass joins nothing — the lone fill face
        // must still be reported in geom.out.
        let mut mesh = LoopMesh::new();
        let edges = convex_wire_ring(&mut mesh, 3);
        let result = mesh.triangle_fill(
            &edges,
            &TriangleFillParams {
                use_beauty: false,
                use_dissolve: true,
                normal: None,
            },
        );
        assert_eq!(mesh.vertex_count(), 3, "no vertex created");
        assert_eq!(mesh.edge_count(), 3, "only the three boundary edges");
        assert_eq!(mesh.face_count(), 1, "one fill triangle, live in the mesh");
        assert_eq!(
            result.faces().len(),
            1,
            "geom.out must report the lone fill triangle (no interior edges to join)"
        );
        mesh.validate_topology();
    }

    #[test]
    fn use_beauty_does_not_panic_and_fills() {
        // use_beauty is a caller-settable input: it must not abort. The
        // beautify pass is not yet applied, so this is just a valid fill.
        let mut mesh = LoopMesh::new();
        let edges = convex_wire_ring(&mut mesh, 6);
        let result = mesh.triangle_fill(
            &edges,
            &TriangleFillParams {
                use_beauty: true,
                use_dissolve: false,
                normal: None,
            },
        );
        assert_eq!(mesh.face_count(), 4, "hexagon → n-2 = 4 triangles");
        assert_eq!(result.faces().len(), 4);
        mesh.validate_topology();
    }

    /// Fill a fresh pentagon ring with the supplied projection normal and
    /// return the (consistent) sign of each fill face's normal along the
    /// supplied axis. The ring is rebuilt each call so vertex/edge ordering is
    /// identical across calls — only the supplied normal differs.
    fn pentagon_fill_face_signs(supplied: Vec3) -> Vec<f32> {
        let mut mesh = LoopMesh::new();
        let edges = convex_wire_ring(&mut mesh, 5);
        let result = mesh.triangle_fill(
            &edges,
            &TriangleFillParams {
                use_beauty: false,
                use_dissolve: false,
                normal: Some(supplied),
            },
        );
        assert_eq!(result.faces().len(), 3, "pentagon → n-2 = 3 triangles");
        mesh.validate_topology();
        result
            .faces()
            .iter()
            .map(|&face| {
                mesh.face_normal_update(face);
                mesh.face_normal(face).dot(supplied.normalize())
            })
            .collect()
    }

    #[test]
    fn explicit_normal_supplied_projects_and_winds_consistently() {
        // A supplied non-zero normal is authoritative (§3/§4): it is used
        // directly as the projection plane and the §4 winding vote is skipped.
        // The guide guarantees the fill is *consistently* wound (the sweep
        // emits a consistently-wound triangle set); it does not tie the face
        // normals' sign to the supplied normal's sign, so assert consistency,
        // not a particular hemisphere.
        let signs = pentagon_fill_face_signs(Vec3::new(0.0, 0.3, 1.0));
        let first = signs[0].signum();
        assert!(
            signs.iter().all(|s| s.signum() == first && s.abs() > 0.0),
            "fill must be consistently wound for a supplied normal: {signs:?}"
        );
    }

    #[test]
    fn convex_pentagon_dissolve_to_single_ngon() {
        let mut mesh = LoopMesh::new();
        let edges = convex_wire_ring(&mut mesh, 5);
        mesh.triangle_fill(
            &edges,
            &TriangleFillParams {
                use_beauty: false,
                use_dissolve: true,
                normal: None,
            },
        );
        assert_eq!(mesh.vertex_count(), 5, "no vertex created");
        assert_eq!(mesh.edge_count(), 5, "only boundary edges survive");
        assert_eq!(mesh.face_count(), 1, "odd-arity loop dissolves to 1 n-gon");
        assert_eq!(mesh.face_len(mesh.face_keys()[0]), 5);
        mesh.validate_topology();
    }

    #[test]
    fn raw_fill_emits_n_minus_two_triangles() {
        let mut mesh = LoopMesh::new();
        let n = 7;
        let edges = convex_wire_ring(&mut mesh, n);
        let result = mesh.triangle_fill(
            &edges,
            &TriangleFillParams {
                use_beauty: false,
                use_dissolve: false,
                normal: None,
            },
        );
        assert_eq!(mesh.vertex_count(), n, "no vertex created");
        assert_eq!(mesh.face_count(), n - 2, "simple n-gon → n-2 triangles");
        for face in mesh.face_keys() {
            assert_eq!(mesh.face_len(face), 3, "every fill face is a triangle");
        }
        // n boundary edges survive; the raw fill adds n-3 interior diagonals.
        assert_eq!(mesh.edge_count(), n + (n - 3));
        assert_eq!(
            result.edges().len(),
            n - 3,
            "geom.out reports only the new interior diagonals"
        );
        assert_eq!(result.faces().len(), n - 2);
        mesh.validate_topology();
    }

    #[test]
    fn fewer_than_three_edges_fills_nothing() {
        let mut mesh = LoopMesh::new();
        let v0 = mesh.create_vertex(Vec3::ZERO);
        let v1 = mesh.create_vertex(Vec3::X);
        let v2 = mesh.create_vertex(Vec3::Y);
        let e0 = mesh.create_edge([v0, v1], EdgeCreateFlags::NO_DUPLICATE);
        let e1 = mesh.create_edge([v1, v2], EdgeCreateFlags::NO_DUPLICATE);
        let result = mesh.triangle_fill(
            &[e0, e1],
            &TriangleFillParams::default(),
        );
        assert!(result.is_empty());
        assert_eq!(mesh.face_count(), 0);
    }

    #[test]
    fn open_chain_fills_nothing() {
        // Three edges that form an open path, not a closed loop.
        let mut mesh = LoopMesh::new();
        let v0 = mesh.create_vertex(Vec3::new(0.0, 0.0, 0.0));
        let v1 = mesh.create_vertex(Vec3::new(1.0, 0.0, 0.0));
        let v2 = mesh.create_vertex(Vec3::new(2.0, 0.0, 0.0));
        let v3 = mesh.create_vertex(Vec3::new(3.0, 0.0, 0.0));
        let e0 = mesh.create_edge([v0, v1], EdgeCreateFlags::NO_DUPLICATE);
        let e1 = mesh.create_edge([v1, v2], EdgeCreateFlags::NO_DUPLICATE);
        let e2 = mesh.create_edge([v2, v3], EdgeCreateFlags::NO_DUPLICATE);
        let result = mesh.triangle_fill(&[e0, e1, e2], &TriangleFillParams::default());
        assert!(result.is_empty(), "open chain bounds no region");
        assert_eq!(mesh.face_count(), 0);
    }
}
