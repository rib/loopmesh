use smallvec::SmallVec;

use crate::loopmesh::*;

/// Output of [`LoopMesh::extrude_region_mixed`]. A mixed-input extrude routes
/// each input element by its kind (EXTRUDE.md §1/§2), so its output is the
/// union of the three per-kind sub-operator results: the region/face pass, the
/// edge-only pass for genuinely loose input edges, and the per-vertex pass for
/// isolated input verts.
#[derive(Debug, Default)]
pub struct ExtrudeMixedResult {
    /// `(source, duplicate)` for every duplicated vertex, across all routed
    /// kinds. A vertex is duplicated at most once even when it is reachable
    /// through more than one input kind (a face's vert that is also named
    /// directly, say).
    pub vert_map: Vec<(VertexKey, VertexKey)>,
    /// `(source, duplicate)` for every duplicated edge, across the face and
    /// edge-only passes.
    pub edge_map: Vec<(EdgeKey, EdgeKey)>,
    /// Newly-created "vertical" edges connecting source verts to their
    /// duplicates around the face / edge walls.
    pub side_edges: Vec<EdgeKey>,
    /// Newly-created wall quad faces from the face pass and the edge-only pass.
    pub wall_faces: Vec<FaceKey>,
    /// Duplicate cap faces (populated only under `use_keep_orig`).
    pub cap_faces: Vec<FaceKey>,
    /// Fresh wire edges joining each isolated input vert to its duplicate
    /// (§3.5). Empty when the input named no isolated verts.
    pub wire_edges: Vec<EdgeKey>,
}

impl LoopMesh {
    /// Extrude a **mixed selection** of faces, edges, and verts in one call,
    /// routing each element by its kind (EXTRUDE.md §1, §2).
    ///
    /// The geom slot of region extrude accepts all three element kinds at once.
    /// Per §2.1 a named face *implies* its edges and verts are part of the
    /// input, and per §3.1 the closure is applied before routing. The
    /// consequence — and the load-bearing de-duplication rule of §2/§8.5 — is
    /// that an edge already on a selected face's boundary is **not** processed
    /// a second time as a free edge: it is owned by the face/region pass, which
    /// classifies it as a boundary or interior edge there. Likewise a vert
    /// already covered by an input face or input edge is **not** routed to the
    /// per-vertex pass.
    ///
    /// Routing after that closure:
    ///
    /// - **Faces** → the region pass ([`extrude_region`](Self::extrude_region)),
    ///   which lifts the selected patch, walls its open boundary, and (here)
    ///   already covers every edge and vert implied by those faces.
    /// - **Edges named directly that are not on any input face** → the
    ///   edge-only pass ([`extrude_edge_only`](Self::extrude_edge_only)): each
    ///   gets a wall quad spanning the original edge to its duplicate (§10).
    ///   A genuinely loose wire edge in this set is the §8.3 case.
    /// - **Verts named directly that are not on any input face or routed input
    ///   edge** → the per-vertex pass
    ///   ([`extrude_vert_indiv`](Self::extrude_vert_indiv)): each isolated vert
    ///   is duplicated and joined to its duplicate by a fresh wire edge (§3.5).
    ///
    /// The three passes operate on disjoint element sets after the de-dup, so
    /// composing them reproduces the single-buffer routing the guide describes.
    /// The merged result names every new element each pass produced.
    ///
    /// See `radial-mesh-guides/operators/EXTRUDE.md` §1–§2 (geom-slot routing)
    /// and §§3.5, 8.3, 8.5 for the de-duplication and isolated-element rules.
    pub fn extrude_region_mixed(
        &mut self,
        faces: &[FaceKey],
        edges: &[EdgeKey],
        verts: &[VertexKey],
    ) -> ExtrudeMixedResult {
        self.extrude_region_mixed_ex(faces, edges, verts, false, false)
    }

    /// [`extrude_region_mixed`](Self::extrude_region_mixed) with the
    /// `use_keep_orig` and `use_normal_flip` flags exposed. The flags apply to
    /// the face/region pass exactly as in
    /// [`extrude_region_exclude`](Self::extrude_region_exclude); the edge-only
    /// and per-vertex passes honour `use_normal_flip` (edge walls) and are
    /// otherwise unaffected (neither has an interior to keep).
    pub fn extrude_region_mixed_ex(
        &mut self,
        faces: &[FaceKey],
        edges: &[EdgeKey],
        verts: &[VertexKey],
        use_keep_orig: bool,
        use_normal_flip: bool,
    ) -> ExtrudeMixedResult {
        let vert_watermark = self.vertices.key_watermark();
        let edge_watermark = self.edges.key_watermark();

        // §3.1 input closure, used here only to decide routing ownership: an
        // edge or vert already reachable from a named face is owned by the
        // region pass and must not be re-extruded by the edge / vert passes.
        let mut vert_on_face: Vec<bool> = vec![false; vert_watermark];
        let mut edge_on_face: Vec<bool> = vec![false; edge_watermark];
        for &f in faces {
            let mut it = self.face_loops(f);
            while let Some(l) = it.next(self) {
                let v = self.loops.get(l).vertex;
                let e = self.loops.get(l).edge;
                vert_on_face[v.as_usize()] = true;
                edge_on_face[e.as_usize()] = true;
            }
        }

        // Edges to route to the edge-only pass: named directly, not on any
        // input face. Dedup so a doubly-named edge routes once.
        let mut edge_seen: Vec<bool> = vec![false; edge_watermark];
        let mut routed_edges: SmallVec<[EdgeKey; 16]> = SmallVec::new();
        for &e in edges {
            let slot = e.as_usize();
            if slot >= edge_watermark || edge_on_face[slot] || edge_seen[slot] {
                continue;
            }
            edge_seen[slot] = true;
            routed_edges.push(e);
        }

        // Verts owned by a routed edge are not isolated.
        let mut vert_on_routed_edge: Vec<bool> = vec![false; vert_watermark];
        for &e in &routed_edges {
            let [v0, v1] = self.edge_vertices(e);
            vert_on_routed_edge[v0.as_usize()] = true;
            vert_on_routed_edge[v1.as_usize()] = true;
        }

        // §3.5 isolated verts: named directly, on no input face and no routed
        // edge. Dedup against the prior kinds and against each other.
        let mut vert_seen: Vec<bool> = vec![false; vert_watermark];
        let mut routed_verts: SmallVec<[VertexKey; 16]> = SmallVec::new();
        for &v in verts {
            let slot = v.as_usize();
            if slot >= vert_watermark
                || vert_on_face[slot]
                || vert_on_routed_edge[slot]
                || vert_seen[slot]
            {
                continue;
            }
            vert_seen[slot] = true;
            routed_verts.push(v);
        }

        let mut out = ExtrudeMixedResult::default();

        if !faces.is_empty() {
            let region =
                self.extrude_region_exclude(faces, &[], use_keep_orig, use_normal_flip);
            out.vert_map.extend(region.vert_map);
            out.edge_map.extend(region.edge_map);
            out.side_edges.extend(region.side_edges);
            out.wall_faces.extend(region.wall_faces);
            out.cap_faces.extend(region.cap_faces);
        }

        if !routed_edges.is_empty() {
            let edge_res = self.extrude_edge_only_ex(&routed_edges, use_normal_flip);
            out.vert_map.extend(edge_res.vert_map);
            out.edge_map.extend(edge_res.edge_map);
            out.side_edges.extend(edge_res.side_edges);
            out.wall_faces.extend(edge_res.wall_faces);
            out.cap_faces.extend(edge_res.cap_faces);
        }

        if !routed_verts.is_empty() {
            let vert_res = self.extrude_vert_indiv(&routed_verts);
            out.vert_map.extend(vert_res.vert_map);
            out.wire_edges.extend(vert_res.wire_edges);
        }

        out
    }
}

#[cfg(test)]
mod tests {
    use glam::Vec3;

    use crate::loopmesh::test_support::create_face_from_verts;
    use crate::loopmesh::*;

    fn unit_quad(mesh: &mut LoopMesh) -> FaceKey {
        let verts = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        ];
        create_face_from_verts(mesh, &verts)
    }

    // The compat fixture shape: a quad face (verts 0..3), a disjoint wire edge
    // {4,5}, and a single loose vert 6.
    fn face_loose_edge_and_vert(
        mesh: &mut LoopMesh,
    ) -> (FaceKey, EdgeKey, VertexKey) {
        let face = unit_quad(mesh);
        let v4 = mesh.create_vertex(Vec3::new(3.0, 0.0, 0.0));
        let v5 = mesh.create_vertex(Vec3::new(4.0, 0.0, 0.0));
        let wire = mesh.create_edge([v4, v5], EdgeCreateFlags::NONE);
        let loose = mesh.create_vertex(Vec3::new(6.0, 6.0, 0.0));
        (face, wire, loose)
    }

    #[test]
    fn mixed_all_three_kinds_route_independently() {
        // Face → region (4 walls + cap), wire edge → edge-only (1 wall),
        // loose vert → vert-indiv (1 wire edge). The three sets are disjoint
        // so the merged result is the exact union of the three passes.
        let mut mesh = LoopMesh::new();
        let (face, wire, loose) = face_loose_edge_and_vert(&mut mesh);

        let result = mesh.extrude_region_mixed(&[face], &[wire], &[loose]);

        // 4 face verts + 2 edge endpoints + 1 loose vert duplicated.
        assert_eq!(result.vert_map.len(), 7);
        // 4 face boundary edges + 1 wire edge duplicated.
        assert_eq!(result.edge_map.len(), 5);
        // 4 walls from the face, 1 wall from the wire edge.
        assert_eq!(result.wall_faces.len(), 5);
        // 4 face verticals + 2 wire verticals.
        assert_eq!(result.side_edges.len(), 6);
        // One wire edge from the isolated vert.
        assert_eq!(result.wire_edges.len(), 1);
        // Default path: original face re-anchored as cap, so no fresh cap.
        assert_eq!(result.cap_faces.len(), 0);

        // Total faces: original quad becomes a cap (5 walls + 1 cap = 6).
        assert_eq!(mesh.totface(), 6);
        mesh.validate_topology();
    }

    #[test]
    fn mixed_face_and_edge_subset() {
        // Face + a disjoint wire edge, no loose vert. The vert list being empty
        // means no per-vertex pass runs.
        let mut mesh = LoopMesh::new();
        let (face, wire, _loose) = face_loose_edge_and_vert(&mut mesh);

        let result = mesh.extrude_region_mixed(&[face], &[wire], &[]);

        assert_eq!(result.vert_map.len(), 6, "4 face verts + 2 edge endpoints");
        assert_eq!(result.edge_map.len(), 5);
        assert_eq!(result.wall_faces.len(), 5);
        assert_eq!(result.wire_edges.len(), 0, "no isolated verts named");
        mesh.validate_topology();
    }

    #[test]
    fn mixed_edge_on_selected_face_not_double_walled() {
        // §2 / §8.5 de-dup: an edge that is part of a selected face's boundary
        // must NOT be re-routed to the edge-only pass. Naming such an edge
        // explicitly alongside its owning face yields the same result as the
        // face alone — the face/region pass owns it as a boundary edge.
        let mut mesh_named = LoopMesh::new();
        let face_named = unit_quad(&mut mesh_named);
        // A boundary edge of the face (v0–v1).
        let fv = mesh_named.face_vert_keys(face_named);
        let boundary_edge = mesh_named
            .query_edge_exists(fv[0], fv[1])
            .expect("perimeter edge v0-v1");
        let result_named =
            mesh_named.extrude_region_mixed(&[face_named], &[boundary_edge], &[]);

        let mut mesh_plain = LoopMesh::new();
        let face_plain = unit_quad(&mut mesh_plain);
        let result_plain = mesh_plain.extrude_region_mixed(&[face_plain], &[], &[]);

        // Naming the boundary edge changed nothing: same wall / edge counts.
        assert_eq!(result_named.wall_faces.len(), result_plain.wall_faces.len());
        assert_eq!(result_named.wall_faces.len(), 4, "exactly 4 walls, no extra");
        assert_eq!(result_named.edge_map.len(), result_plain.edge_map.len());
        assert_eq!(result_named.edge_map.len(), 4);
        assert_eq!(mesh_named.totface(), mesh_plain.totface());
        assert_eq!(mesh_named.totface(), 5);
        mesh_named.validate_topology();
    }

    #[test]
    fn mixed_vert_on_selected_face_not_individually_extruded() {
        // A vert that already belongs to a selected face is owned by the region
        // pass and must not be routed to the per-vertex pass (which would add a
        // stray wire edge). Naming such a vert is a no-op for the vert pass.
        let mut mesh = LoopMesh::new();
        let face = unit_quad(&mut mesh);
        let corner = mesh.face_vert_keys(face)[0];

        let result = mesh.extrude_region_mixed(&[face], &[], &[corner]);

        // No isolated wire edge: the corner is covered by the face.
        assert_eq!(result.wire_edges.len(), 0, "face-owned vert is not isolated");
        assert_eq!(result.vert_map.len(), 4, "only the 4 face verts duplicated");
        assert_eq!(result.wall_faces.len(), 4);
        mesh.validate_topology();
    }

    #[test]
    fn mixed_vert_on_routed_edge_not_individually_extruded() {
        // A vert that is an endpoint of a routed (edge-only) input edge is not
        // isolated, so naming it does not trigger a separate vert-indiv wire.
        let mut mesh = LoopMesh::new();
        let v0 = mesh.create_vertex(Vec3::new(0.0, 0.0, 0.0));
        let v1 = mesh.create_vertex(Vec3::new(1.0, 0.0, 0.0));
        let wire = mesh.create_edge([v0, v1], EdgeCreateFlags::NONE);

        let result = mesh.extrude_region_mixed(&[], &[wire], &[v0]);

        assert_eq!(result.wall_faces.len(), 1, "one wall from the wire edge");
        assert_eq!(result.wire_edges.len(), 0, "edge endpoint is not isolated");
        assert_eq!(result.vert_map.len(), 2, "both edge endpoints duplicated once");
        mesh.validate_topology();
    }

    #[test]
    fn mixed_empty_input_is_noop() {
        let mut mesh = LoopMesh::new();
        let _ = unit_quad(&mut mesh);
        let before = (mesh.totvert(), mesh.totedge(), mesh.totface());

        let result = mesh.extrude_region_mixed(&[], &[], &[]);

        assert_eq!(result.vert_map.len(), 0);
        assert_eq!(result.edge_map.len(), 0);
        assert_eq!(result.wall_faces.len(), 0);
        assert_eq!(result.wire_edges.len(), 0);
        assert_eq!((mesh.totvert(), mesh.totedge(), mesh.totface()), before);
        mesh.validate_topology();
    }
}
