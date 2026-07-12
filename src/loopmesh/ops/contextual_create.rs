use std::collections::HashSet;

use glam::Vec3;

use crate::loopmesh::*;

/// The two knobs of [`LoopMesh::contextual_create`], mirroring the operator's
/// BMOP slot parameters
/// ([`CONTEXTUAL_CREATE.md`](../../../../radial-mesh-guides/operators/CONTEXTUAL_CREATE.md)
/// §1). Both are stamped **only** on faces the operator makes directly — the
/// vert-cloud n-gon (§10) and the edge-net fill caps (§6); the dissolve (§7)
/// and edgeloop-fill (§8) paths inherit or leave defaults.
#[derive(Copy, Clone, Debug, Default)]
pub struct ContextualCreateParams {
    /// Material index stamped on the §6 edge-net-fill caps and the §10
    /// vert-cloud n-gon (§1).
    pub mat_nr: i32,
    /// Smooth header bit stamped on the same directly-created / edge-net-filled
    /// faces as `mat_nr` (§1).
    pub use_smooth: bool,
}

/// Output of [`LoopMesh::contextual_create`]
/// ([`CONTEXTUAL_CREATE.md`](../../../../radial-mesh-guides/operators/CONTEXTUAL_CREATE.md)
/// §1). `faces` carries the face(s) a fill / dissolve branch produced; `edges`
/// carries **only** the stand-alone edge the two-vert branch (§4) makes — edges
/// created incidentally as the sides of a new face are not reported here.
#[derive(Clone, Debug, Default)]
pub struct ContextualCreateResult {
    faces: Vec<FaceKey>,
    edges: Vec<EdgeKey>,
}

impl ContextualCreateResult {
    /// The `faces.out` slot — the created / dissolved faces (§1).
    pub fn faces(&self) -> &[FaceKey] {
        &self.faces
    }
    /// The `edges.out` slot — only the two-vert branch's stand-alone edge (§1 / §4).
    pub fn edges(&self) -> &[EdgeKey] {
        &self.edges
    }
    pub fn is_empty(&self) -> bool {
        self.faces.is_empty() && self.edges.is_empty()
    }
}

impl LoopMesh {
    /// Build the single most-appropriate piece of geometry from a mixed
    /// selection — the *make edge / face* dispatcher. It counts the selected
    /// verts / edges / faces (`totv` / `tote` / `totf`) and routes to one of
    /// several sub-behaviours, most of which **delegate** to an existing fill /
    /// dissolve operator; only the two-vert edge (§4 / §5) and the vert-cloud
    /// n-gon (§10) touch the mesh directly. The routing — the guard predicates
    /// and their order — is the whole substance of the operator.
    ///
    /// Algorithm in
    /// [`CONTEXTUAL_CREATE.md`](../../../../radial-mesh-guides/operators/CONTEXTUAL_CREATE.md):
    /// §3 counts and tags the selection, then the §1.1 cascade fires the first
    /// matching guard — §4 two-vert edge (terminal), §5 isolated-vert closure
    /// (falls through to §6), §6 edge-net prepare + fill, §7 dissolve a face
    /// region, §8 edgeloop-fill fallback, §10 the last-resort vert-cloud n-gon.
    /// The disabled select-history branch (§9) is omitted, so the cascade goes
    /// §8 → §10 directly.
    pub fn contextual_create(
        &mut self,
        geom: &[GeomKey],
        params: ContextualCreateParams,
    ) -> ContextualCreateResult {
        let mut result = ContextualCreateResult::default();

        // §3 — count and de-duplicate the selection into its three kinds. Every
        // downstream guard is a predicate over the triple (totv, tote, totf);
        // `edges` doubles as the working net the fill branches consume.
        let mut verts: Vec<VertexKey> = Vec::new();
        let mut edges: Vec<EdgeKey> = Vec::new();
        let mut faces: Vec<FaceKey> = Vec::new();
        let mut seen_verts: HashSet<VertexKey> = HashSet::new();
        let mut seen_edges: HashSet<EdgeKey> = HashSet::new();
        let mut seen_faces: HashSet<FaceKey> = HashSet::new();
        for entry in geom {
            match *entry {
                GeomKey::Vert(v) => {
                    if seen_verts.insert(v) {
                        verts.push(v);
                    }
                }
                GeomKey::Edge(e) => {
                    if seen_edges.insert(e) {
                        edges.push(e);
                    }
                }
                GeomKey::Face(f) => {
                    if seen_faces.insert(f) {
                        faces.push(f);
                    }
                }
            }
        }
        let totv = verts.len();
        // §3 — the count of *selected* edges. `edges` is later reassigned to the
        // prepared/working fill net (§6), which can be larger than the selection,
        // so §8's `tote > 2` guard must key off this dedicated counter, not
        // `edges.len()`. Its only sanctioned increment is §5's `+2` chain closure.
        let mut tote = edges.len();
        let totf = faces.len();

        // §4 — two verts make an edge. Terminal: report the (no-duplicate) edge
        // in edges.out and return; faces.out stays empty.
        if totv == 2 && tote == 0 && totf == 0 {
            let edge = self.create_edge([verts[0], verts[1]], EdgeCreateFlags::NO_DUPLICATE);
            result.edges.push(edge);
            return result;
        }

        // §5 — an isolated vert closes an open chain. The `totv == tote + 2`
        // signature is an open chain plus one free vert; when the consistency
        // scan finds exactly one free vert and two chain endpoints, fold the
        // free vert in with two edges, then **fall through** to §6. A failed
        // scan just skips the edit and lets the cascade continue.
        if totf == 0 && totv >= 4 && totv == tote + 2 {
            if let Some((free, endpoint_a, endpoint_b)) =
                self.contextual_create_chain_closure(&verts, &edges)
            {
                edges.push(self.create_edge([free, endpoint_a], EdgeCreateFlags::NO_DUPLICATE));
                edges.push(self.create_edge([free, endpoint_b], EdgeCreateFlags::NO_DUPLICATE));
                // §5 — the two chain-closure edges are the only sanctioned bump
                // to the selected-edge count the §8 guard reads.
                tote += 2;
            }
        }

        // §6 — edge-net prepare then fill. Prepare bridges open chains into a
        // closed net (a branch-rejected input reports empty, in which case the
        // fill acts on the original edges — §11); the fill caps every enclosed
        // cycle with one n-gon, stamped with the caller's knobs.
        if !edges.is_empty() {
            let prepared = self.edgenet_prepare(&edges);
            let fill_input: Vec<EdgeKey> = if prepared.is_empty() {
                edges.clone()
            } else {
                prepared.edges().to_vec()
            };
            let fill = self.edgenet_fill(
                &fill_input,
                &EdgenetFillParams {
                    mat_nr: params.mat_nr,
                    use_smooth: params.use_smooth,
                    // §6 — an effectively unbounded `sides`; edge-net fill never
                    // reads it anyway (it is inert there).
                    sides: usize::MAX,
                },
            );
            if !fill.is_empty() {
                result.faces = fill.faces().to_vec();
                return result;
            }
            // No cycle closed — the prepared net is the working set §8 inherits.
            edges = fill_input;
        }

        // §7 — dissolve a face region. Run even on a single face: a lone face
        // has no selected neighbour to merge with, so dissolve produces nothing
        // and the cascade falls through to a no-op (§11).
        if !faces.is_empty() {
            let dissolved = self.dissolve_faces(&faces, DissolveFacesOptions::default());
            if !dissolved.region.is_empty() {
                result.faces = dissolved.region;
                return result;
            }
        }

        // §8 — edgeloop-fill fallback for a closed loop around existing geometry
        // that edge-net fill's minimal-cycle walk could not close. Inherits /
        // leaves default material (the caller's knobs are *not* stamped here, §1).
        // §8 guards on `tote` (the §3 selected-edge count, +2 if §5 fired), not
        // the working net `edges`, which §6's prepare may have grown past `tote`.
        if tote > 2 {
            let filled = self.edgeloop_fill(&edges, &EdgeloopFillParams::default());
            if !filled.is_empty() {
                result.faces = filled.faces().to_vec();
                return result;
            }
        }

        // §9 — the select-history line is disabled; the cascade skips it.

        // §10 — the vert-cloud n-gon, last resort. A loose cloud of three or
        // more verts that no fill or dissolve consumed is angular-sorted about
        // its centroid and capped with one stamped n-gon.
        if totv > 2 {
            let face = self.contextual_create_vert_cloud_ngon(&verts, params);
            result.faces.push(face);
            return result;
        }

        // §11 — no branch matched (zero / one vert, and nothing else): no-op.
        result
    }

    /// §5 consistency scan. Classify each selected vert by how many **selected**
    /// edges touch it: zero → the free vert (at most one), one → an open-chain
    /// endpoint (at most two), two → an interior vert, three or more → a branch
    /// that cancels the closure. Returns the free vert and the two endpoints
    /// only when the shape is exactly one free vert plus two endpoints.
    fn contextual_create_chain_closure(
        &self,
        verts: &[VertexKey],
        edges: &[EdgeKey],
    ) -> Option<(VertexKey, VertexKey, VertexKey)> {
        let selected_edges: HashSet<EdgeKey> = edges.iter().copied().collect();
        let mut free: Option<VertexKey> = None;
        let mut endpoints: Vec<VertexKey> = Vec::with_capacity(2);
        for &vertex in verts {
            let selected_degree = self
                .iter_disk_edges(vertex)
                .filter(|edge| selected_edges.contains(edge))
                .count();
            match selected_degree {
                0 => {
                    if free.is_some() {
                        return None;
                    }
                    free = Some(vertex);
                }
                1 => {
                    if endpoints.len() == 2 {
                        return None;
                    }
                    endpoints.push(vertex);
                }
                2 => {}
                _ => return None,
            }
        }
        match (free, endpoints.as_slice()) {
            (Some(free), [endpoint_a, endpoint_b]) => Some((free, *endpoint_a, *endpoint_b)),
            _ => None,
        }
    }

    /// §10 — cap a loose vert cloud with one n-gon. Fit a plane through the
    /// verts' centroid, sort the verts by angle around the centroid in that
    /// plane (a best-effort rotational order, not a hull), build one n-gon over
    /// the sorted ring (its boundary edges no-duplicate), stamp the caller's
    /// material / smooth, and pull each corner's loop data from any existing
    /// face across a shared edge — the caller's stamp survives.
    fn contextual_create_vert_cloud_ngon(
        &mut self,
        verts: &[VertexKey],
        params: ContextualCreateParams,
    ) -> FaceKey {
        let centroid = self.contextual_create_centroid(verts);
        let normal = self.contextual_create_fit_plane_normal(verts, centroid);

        // In-plane orthonormal basis (axis_u, axis_w) for the angular sort.
        let reference = if normal.x.abs() < 0.9 {
            Vec3::X
        } else {
            Vec3::Y
        };
        let mut axis_u = (reference - normal * reference.dot(normal)).normalize_or_zero();
        if axis_u.length_squared() < 1e-12 {
            axis_u = normal.any_orthonormal_vector();
        }
        let axis_w = normal.cross(axis_u);

        let mut ring: Vec<VertexKey> = verts.to_vec();
        // Compute each vert's plane-relative angle once (O(n) atan2), keyed on a
        // monotonic total-order transform of the f32 so the stable sort keeps
        // input order on ties. Adding 0.0 folds −0.0 onto +0.0, so equal-angle
        // ties compare equal exactly as the prior `partial_cmp` did.
        ring.sort_by_cached_key(|&v| {
            let offset = self.vertex_position(v) - centroid;
            let angle = offset.dot(axis_w).atan2(offset.dot(axis_u)) + 0.0;
            let bits = angle.to_bits();
            if bits >> 31 == 1 {
                !bits
            } else {
                bits | 0x8000_0000
            }
        });

        let face = self.face_create_verts(&ring, FaceCreateFlags::NO_DUPLICATE);
        self.set_face_material_index(face, params.mat_nr);
        self.set_face_smooth(face, params.use_smooth);

        // §10 step 3 — inherit per-corner loop data only (UVs / colours); the
        // face-domain material / smooth stamp above is left intact.
        if !self.loop_attribs.is_empty() {
            let cap_set: HashSet<FaceKey> = std::iter::once(face).collect();
            self.holes_fill_copy_loop_data(face, &cap_set);
        }

        face
    }

    fn contextual_create_centroid(&self, verts: &[VertexKey]) -> Vec3 {
        let mut centroid = Vec3::ZERO;
        for &vertex in verts {
            centroid += self.vertex_position(vertex);
        }
        centroid / verts.len() as f32
    }

    /// Least-squares plane normal through `centroid`: the null direction of the
    /// centered covariance matrix, taken as the longest pairwise cross product
    /// of its rows (exact for a coplanar cloud, best-effort otherwise). The
    /// sign is not determinate — §10 compares winding orientation-tolerantly.
    fn contextual_create_fit_plane_normal(&self, verts: &[VertexKey], centroid: Vec3) -> Vec3 {
        let mut covariance = [[0.0f32; 3]; 3];
        for &vertex in verts {
            let offset = self.vertex_position(vertex) - centroid;
            let coords = [offset.x, offset.y, offset.z];
            for row in 0..3 {
                for col in 0..3 {
                    covariance[row][col] += coords[row] * coords[col];
                }
            }
        }
        let rows = [
            Vec3::from(covariance[0]),
            Vec3::from(covariance[1]),
            Vec3::from(covariance[2]),
        ];
        let candidates = [
            rows[0].cross(rows[1]),
            rows[1].cross(rows[2]),
            rows[2].cross(rows[0]),
        ];
        let mut normal = candidates[0];
        for candidate in &candidates[1..] {
            if candidate.length_squared() > normal.length_squared() {
                normal = *candidate;
            }
        }
        let normal = normal.normalize_or_zero();
        if normal.length_squared() < 1e-12 {
            Vec3::Z
        } else {
            normal
        }
    }
}

#[cfg(test)]
mod tests {
    use std::f32::consts::TAU;

    use glam::Vec3;

    use super::ContextualCreateParams;
    use crate::loopmesh::*;

    /// Build `n` verts on a circle of `radius` in the z=0 plane, returning them
    /// in angular order.
    fn ring_verts(mesh: &mut LoopMesh, n: usize, radius: f32) -> Vec<VertexKey> {
        (0..n)
            .map(|i| {
                let theta = (i as f32) * TAU / (n as f32);
                mesh.create_vertex(Vec3::new(radius * theta.cos(), radius * theta.sin(), 0.0))
            })
            .collect()
    }

    #[test]
    fn two_verts_make_one_edge() {
        // §4 — exactly two verts, nothing else: one edge, no face, reported in
        // edges.out.
        let mut mesh = LoopMesh::new();
        let a = mesh.create_vertex(Vec3::new(0.0, 0.0, 0.0));
        let b = mesh.create_vertex(Vec3::new(1.0, 0.0, 0.0));

        let out = mesh.contextual_create(&[a.into(), b.into()], ContextualCreateParams::default());

        assert_eq!(mesh.vertex_count(), 2, "no vertex created");
        assert_eq!(mesh.edge_count(), 1, "exactly one edge created");
        assert_eq!(mesh.face_count(), 0, "the two-vert branch makes no face");
        assert_eq!(
            out.edges().len(),
            1,
            "the new edge is reported in edges.out"
        );
        assert!(out.faces().is_empty(), "faces.out stays empty");
        assert!(
            mesh.query_edge_exists(a, b).is_some(),
            "the two verts are joined"
        );
        mesh.validate_topology();
    }

    #[test]
    fn already_edged_pair_reuses_edge() {
        // §4 / §11 — two verts already joined: the no-duplicate guarantee reuses
        // the edge, the mesh gains nothing, but edges.out still names it.
        let mut mesh = LoopMesh::new();
        let a = mesh.create_vertex(Vec3::new(0.0, 0.0, 0.0));
        let b = mesh.create_vertex(Vec3::new(1.0, 0.0, 0.0));
        let existing = mesh.create_edge([a, b], EdgeCreateFlags::NO_DUPLICATE);

        let out = mesh.contextual_create(&[a.into(), b.into()], ContextualCreateParams::default());

        assert_eq!(mesh.edge_count(), 1, "no parallel edge minted");
        assert_eq!(out.edges(), &[existing], "the reused edge is reported");
        mesh.validate_topology();
    }

    #[test]
    fn isolated_vert_closes_chain_and_fills() {
        // §5 → §6 — an open 3-edge chain (v0..v3) plus one free vert v4:
        // totv == tote + 2 (5 == 3 + 2). The scan folds v4 in with two edges,
        // closing a pentagon that §6 caps with one face.
        let mut mesh = LoopMesh::new();
        let v0 = mesh.create_vertex(Vec3::new(0.0, 0.0, 0.0));
        let v1 = mesh.create_vertex(Vec3::new(1.0, 0.0, 0.0));
        let v2 = mesh.create_vertex(Vec3::new(1.0, 1.0, 0.0));
        let v3 = mesh.create_vertex(Vec3::new(0.0, 1.0, 0.0));
        let v4 = mesh.create_vertex(Vec3::new(-0.5, 0.5, 0.0));
        let e0 = mesh.create_edge([v0, v1], EdgeCreateFlags::NO_DUPLICATE);
        let e1 = mesh.create_edge([v1, v2], EdgeCreateFlags::NO_DUPLICATE);
        let e2 = mesh.create_edge([v2, v3], EdgeCreateFlags::NO_DUPLICATE);

        let geom = [
            v0.into(),
            v1.into(),
            v2.into(),
            v3.into(),
            v4.into(),
            e0.into(),
            e1.into(),
            e2.into(),
        ];
        let out = mesh.contextual_create(&geom, ContextualCreateParams::default());

        assert_eq!(
            mesh.edge_count(),
            5,
            "two closure edges added to the three chain edges"
        );
        assert!(
            mesh.query_edge_exists(v4, v0).is_some(),
            "free vert wired to one endpoint"
        );
        assert!(
            mesh.query_edge_exists(v4, v3).is_some(),
            "free vert wired to the other endpoint"
        );
        assert_eq!(out.faces().len(), 1, "the closed pentagon caps to one face");
        assert_eq!(mesh.face_len(out.faces()[0]), 5, "a pentagon cap");
        assert_eq!(mesh.face_count(), 1);
        mesh.validate_topology();
    }

    #[test]
    fn closed_loop_fills_ngon() {
        // §6 — a closed pentagon wire loop caps to one n-gon; no vert or edge
        // created (the loop is reused).
        let mut mesh = LoopMesh::new();
        let verts = ring_verts(&mut mesh, 5, 1.0);
        let loop_edges: Vec<EdgeKey> = (0..5)
            .map(|i| {
                mesh.create_edge(
                    [verts[i], verts[(i + 1) % 5]],
                    EdgeCreateFlags::NO_DUPLICATE,
                )
            })
            .collect();
        let geom: Vec<GeomKey> = loop_edges.iter().map(|&e| e.into()).collect();

        let out = mesh.contextual_create(&geom, ContextualCreateParams::default());

        assert_eq!(out.faces().len(), 1, "closed loop caps to one n-gon");
        assert_eq!(mesh.vertex_count(), 5, "no vertex created");
        assert_eq!(mesh.edge_count(), 5, "no edge created (loop reused)");
        assert_eq!(mesh.face_count(), 1);
        assert_eq!(
            mesh.face_len(out.faces()[0]),
            5,
            "the cap spans all five corners"
        );
        mesh.validate_topology();
    }

    #[test]
    fn open_chain_prepare_bridges_then_fills() {
        // §6 — a single open 3-edge chain (no free vert, so §5 declines): the
        // edge-net prepare pre-pass bridges its two ends, then the fill caps the
        // closed quad. Contextual create fills it where bare edge-net fill would
        // not.
        let mut mesh = LoopMesh::new();
        let v0 = mesh.create_vertex(Vec3::new(0.0, 0.0, 0.0));
        let v1 = mesh.create_vertex(Vec3::new(1.0, 0.0, 0.0));
        let v2 = mesh.create_vertex(Vec3::new(1.0, 1.0, 0.0));
        let v3 = mesh.create_vertex(Vec3::new(0.0, 1.0, 0.0));
        let chain: Vec<GeomKey> = [
            mesh.create_edge([v0, v1], EdgeCreateFlags::NO_DUPLICATE),
            mesh.create_edge([v1, v2], EdgeCreateFlags::NO_DUPLICATE),
            mesh.create_edge([v2, v3], EdgeCreateFlags::NO_DUPLICATE),
        ]
        .iter()
        .map(|&e| e.into())
        .collect();

        let out = mesh.contextual_create(&chain, ContextualCreateParams::default());

        assert_eq!(
            out.faces().len(),
            1,
            "the prepare-bridged chain caps to one face"
        );
        assert_eq!(mesh.edge_count(), 4, "one bridge edge added by prepare");
        assert!(
            mesh.query_edge_exists(v0, v3).is_some(),
            "the chain ends are bridged"
        );
        assert_eq!(mesh.face_len(out.faces()[0]), 4, "a quad cap");
        mesh.validate_topology();
    }

    fn build_two_quads(mesh: &mut LoopMesh) -> Vec<EdgeKey> {
        let v0 = mesh.create_vertex(Vec3::new(0.0, 0.0, 0.0));
        let v1 = mesh.create_vertex(Vec3::new(1.0, 0.0, 0.0));
        let v2 = mesh.create_vertex(Vec3::new(2.0, 0.0, 0.0));
        let v3 = mesh.create_vertex(Vec3::new(0.0, 1.0, 0.0));
        let v4 = mesh.create_vertex(Vec3::new(1.0, 1.0, 0.0));
        let v5 = mesh.create_vertex(Vec3::new(2.0, 1.0, 0.0));
        mesh.face_create_verts(&[v0, v1, v4, v3], FaceCreateFlags::NO_DUPLICATE);
        mesh.face_create_verts(&[v1, v2, v5, v4], FaceCreateFlags::NO_DUPLICATE);
        [(v0, v1), (v1, v2), (v2, v5), (v5, v4), (v4, v3), (v3, v0)]
            .iter()
            .map(|&(a, b)| mesh.query_edge_exists(a, b).unwrap())
            .collect()
    }

    #[test]
    fn edgenet_declines_on_overlap_so_edgeloop_fill_is_reached() {
        // The load-bearing routing pin for §8: the enclosing hexagon of the
        // two-quad region overlaps the already-faced surface, so §6 edge-net fill
        // must **decline** (its minimal-cycle walk is blocked by the faced edges)
        // and §8 edgeloop fill must succeed — proven here on independent meshes so
        // the two sub-op calls don't share state.
        let mut net_mesh = LoopMesh::new();
        let net_ring = build_two_quads(&mut net_mesh);
        let net = net_mesh.edgenet_fill(
            &net_ring,
            &EdgenetFillParams {
                mat_nr: 0,
                use_smooth: false,
                sides: usize::MAX,
            },
        );
        assert!(
            net.is_empty(),
            "§6 edge-net fill declines on the overlapping ring"
        );

        let mut loop_mesh = LoopMesh::new();
        let loop_ring = build_two_quads(&mut loop_mesh);
        let filled = loop_mesh.edgeloop_fill(&loop_ring, &EdgeloopFillParams::default());
        assert_eq!(filled.faces().len(), 1, "§8 edgeloop fill caps the ring");
    }

    #[test]
    fn edge_loop_around_surface_edgeloop_fills_shadow_ngon() {
        // §6 declines / §8 fills — two adjacent coplanar quads share the interior
        // edge v1-v4; the six *outer* boundary edges form an enclosing hexagon
        // that overlaps the already-faced two-quad region. Edge-net fill (§6)
        // declines on that overlap, the cascade falls through §7, and edgeloop
        // fill (§8) caps the ring with one shadow n-gon. No vert / edge created;
        // +1 face.
        let mut mesh = LoopMesh::new();
        let outer_edges = build_two_quads(&mut mesh);
        assert_eq!(mesh.edge_count(), 7, "two quads share one interior edge");
        assert_eq!(mesh.face_count(), 2);

        // The six outer boundary edges (the hexagon); the interior v1-v4 is left
        // unselected.
        let geom: Vec<GeomKey> = outer_edges.iter().map(|&e| e.into()).collect();

        let out = mesh.contextual_create(&geom, ContextualCreateParams::default());

        assert_eq!(
            out.faces().len(),
            1,
            "§8 edgeloop fill caps the ring with one n-gon"
        );
        assert_eq!(
            mesh.face_len(out.faces()[0]),
            6,
            "the shadow cap spans the hexagon"
        );
        assert_eq!(mesh.vertex_count(), 6, "no vertex created");
        assert_eq!(mesh.edge_count(), 7, "no edge created (the ring is reused)");
        assert_eq!(
            mesh.face_count(),
            3,
            "the shadow n-gon is added over the two quads"
        );
        mesh.validate_topology();
    }

    #[test]
    fn single_face_is_noop() {
        // §7 / §11 — dissolve runs even on a lone face, finds no selected
        // neighbour to merge with, produces nothing, and the cascade ends a
        // no-op. The mesh is unchanged.
        let mut mesh = LoopMesh::new();
        let verts = ring_verts(&mut mesh, 4, 1.0);
        let face = mesh.face_create_verts(&verts, FaceCreateFlags::NO_DUPLICATE);

        let out = mesh.contextual_create(&[face.into()], ContextualCreateParams::default());

        assert!(out.is_empty(), "single-face dispatch is a no-op");
        assert_eq!(mesh.vertex_count(), 4, "mesh unchanged: verts");
        assert_eq!(mesh.edge_count(), 4, "mesh unchanged: edges");
        assert_eq!(
            mesh.face_count(),
            1,
            "mesh unchanged: the one face survives"
        );
        mesh.validate_topology();
    }

    #[test]
    fn vert_cloud_stamps_caller_material_and_smooth() {
        // §10 — a loose five-vert coplanar cloud caps to one n-gon with five
        // boundary edges, stamped with the caller's material / smooth.
        let mut mesh = LoopMesh::new();
        let verts = ring_verts(&mut mesh, 5, 1.0);
        let geom: Vec<GeomKey> = verts.iter().map(|&v| v.into()).collect();

        let out = mesh.contextual_create(
            &geom,
            ContextualCreateParams {
                mat_nr: 6,
                use_smooth: true,
            },
        );

        assert_eq!(out.faces().len(), 1, "the cloud caps to one n-gon");
        let cap = out.faces()[0];
        assert_eq!(mesh.face_len(cap), 5, "a pentagon over all five verts");
        assert_eq!(mesh.vertex_count(), 5, "no vertex created");
        assert_eq!(mesh.edge_count(), 5, "five boundary edges created");
        assert_eq!(mesh.face_count(), 1);
        assert_eq!(mesh.face_material_index(cap), 6, "caller material stamped");
        assert!(mesh.is_face_smooth(cap), "caller smooth stamped");
        mesh.validate_topology();
    }

    #[test]
    fn vert_cloud_zero_stamp_is_default_and_flat() {
        // §10 — the zero contrast that makes the stamp pin meaningful: mat_nr 0 /
        // use_smooth false → default material, flat.
        let mut mesh = LoopMesh::new();
        let verts = ring_verts(&mut mesh, 5, 1.0);
        let geom: Vec<GeomKey> = verts.iter().map(|&v| v.into()).collect();

        let out = mesh.contextual_create(
            &geom,
            ContextualCreateParams {
                mat_nr: 0,
                use_smooth: false,
            },
        );

        assert_eq!(out.faces().len(), 1);
        let cap = out.faces()[0];
        assert_eq!(mesh.face_material_index(cap), 0, "default material slot 0");
        assert!(!mesh.is_face_smooth(cap), "default cap is flat");
        assert_eq!(mesh.edge_count(), 5);
        mesh.validate_topology();
    }

    #[test]
    fn three_verts_make_triangle() {
        // §11 — the minimal cloud: three loose verts fall through to §10 and
        // make a triangle.
        let mut mesh = LoopMesh::new();
        let a = mesh.create_vertex(Vec3::new(0.0, 0.0, 0.0));
        let b = mesh.create_vertex(Vec3::new(1.0, 0.0, 0.0));
        let c = mesh.create_vertex(Vec3::new(0.5, 1.0, 0.0));

        let out = mesh.contextual_create(
            &[a.into(), b.into(), c.into()],
            ContextualCreateParams::default(),
        );

        assert_eq!(out.faces().len(), 1, "three verts cap to one triangle");
        assert_eq!(mesh.face_len(out.faces()[0]), 3);
        assert_eq!(mesh.edge_count(), 3, "three boundary edges");
        mesh.validate_topology();
    }

    #[test]
    fn zero_or_one_vert_is_noop() {
        // §11 — no branch precondition matches zero or one selected vert.
        let mut mesh = LoopMesh::new();
        let out_empty = mesh.contextual_create(&[], ContextualCreateParams::default());
        assert!(out_empty.is_empty(), "empty selection is a no-op");

        let a = mesh.create_vertex(Vec3::new(0.0, 0.0, 0.0));
        let out_one = mesh.contextual_create(&[a.into()], ContextualCreateParams::default());
        assert!(out_one.is_empty(), "one vert is a no-op");
        assert_eq!(mesh.vertex_count(), 1);
        assert_eq!(mesh.edge_count(), 0);
        assert_eq!(mesh.face_count(), 0);
        mesh.validate_topology();
    }
}
