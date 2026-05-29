use crate::loopmesh::*;

/// Output of [`LoopMesh::extrude_vert_indiv`]. Per-vertex extrude is purely
/// additive, so the result names only the new geometry: each input vertex's
/// coincident duplicate and the fresh wire edge that joins the pair.
#[derive(Debug, Default)]
pub struct ExtrudeVertIndivResult {
    /// `(source_vertex, duplicate)` for every input vertex. The duplicate sits
    /// at the source's position and carries a copy of its attribs.
    pub vert_map: Vec<(VertexKey, VertexKey)>,
    /// One fresh wire edge per input vertex, joining the source to its
    /// duplicate. Index-aligned with [`vert_map`](Self::vert_map).
    pub wire_edges: Vec<EdgeKey>,
}

impl LoopMesh {
    /// Extrude a set of selected vertices with no edge or face input
    /// (per-vertex / individual-vertex extrude).
    ///
    /// For each input vertex the operator creates a coincident duplicate (vertex
    /// attribs copied) and joins the original to its duplicate with a fresh wire
    /// edge. The operation is **purely additive**: no original geometry is
    /// deleted, no faces are split, and a vertex incident to a face is duplicated
    /// without disturbing that face. The result is a set of "antennae" growing
    /// from the input verts that the caller typically translates outward to grow
    /// a wire skeleton.
    ///
    /// The §11.1 wind-swap consistency tweak: when an input vertex is a *wire
    /// endpoint* — incident to exactly one edge, that edge is a wire (faceless)
    /// edge, and the vertex is the first stored endpoint of it — the new wire
    /// edge is wound `duplicate -> original` rather than `original -> duplicate`,
    /// so a chain of extrusions keeps winding outward consistently from existing
    /// wires.
    ///
    /// Duplicate vertices are deduplicated against the input: naming the same
    /// vertex twice produces a single duplicate and a single wire edge.
    ///
    /// See `radial-mesh-guides/operators/EXTRUDE.md` §11.1 for the full algorithm
    /// and parity contract. (The §11.1 skin-root and select-history bookkeeping
    /// is editor-side and outside this topology contract.)
    pub fn extrude_vert_indiv(&mut self, verts: &[VertexKey]) -> ExtrudeVertIndivResult {
        let mut result = ExtrudeVertIndivResult::default();

        let mut seen: Vec<bool> = vec![false; self.vertices.key_watermark()];
        for &source in verts {
            if seen[source.as_usize()] {
                continue;
            }
            seen[source.as_usize()] = true;

            let wind_swapped = self.is_first_wire_endpoint(source);

            let duplicate = self.create_vertex(self.vertex_position(source));
            self.vertex_attribs
                .copy_into(duplicate.as_usize(), source.as_usize());

            let endpoints = if wind_swapped {
                [duplicate, source]
            } else {
                [source, duplicate]
            };
            let wire = self.create_edge(endpoints, EdgeCreateFlags::NONE);

            result.vert_map.push((source, duplicate));
            result.wire_edges.push(wire);
        }

        result
    }

    /// A vertex qualifies for the §11.1 wind-swap when it is attached to exactly
    /// one edge, that edge carries no faces (a wire edge), and the vertex is its
    /// first stored endpoint.
    fn is_first_wire_endpoint(&self, vertex: VertexKey) -> bool {
        let mut disk = self.vertex_disk_iter(vertex);
        let Some(only_edge) = disk.next() else {
            return false;
        };
        if disk.next().is_some() {
            return false;
        }
        if self.edges.get(only_edge).radial_link.is_some() {
            return false;
        }
        self.edge_vertices(only_edge)[0] == vertex
    }
}

#[cfg(test)]
mod tests {
    use glam::Vec3;

    use crate::loopmesh::test_support::create_face_from_verts;
    use crate::loopmesh::*;

    #[test]
    fn vert_indiv_single_loose_vert() {
        // A lone vertex gains a coincident duplicate joined by a fresh wire edge.
        let mut mesh = LoopMesh::new();
        let v = mesh.create_vertex(Vec3::new(0.5, 0.5, 0.0));

        let result = mesh.extrude_vert_indiv(&[v]);

        assert_eq!(result.vert_map.len(), 1, "one duplicate");
        assert_eq!(result.wire_edges.len(), 1, "one wire edge");
        assert_eq!(mesh.totvert(), 2, "original + duplicate");
        assert_eq!(mesh.totedge(), 1, "the joining wire edge");

        let (source, duplicate) = result.vert_map[0];
        assert_eq!(source, v);
        assert_eq!(
            mesh.vertex_position(duplicate),
            mesh.vertex_position(source),
            "duplicate is coincident"
        );
        let endpoints = mesh.edge_vertices(result.wire_edges[0]);
        assert!(
            endpoints.contains(&source) && endpoints.contains(&duplicate),
            "wire edge joins the pair"
        );
        // A loose vert is not a wire endpoint (no incident edge), so the default
        // winding source -> duplicate applies.
        assert_eq!(endpoints, [source, duplicate], "default winding");
        mesh.validate_topology();
    }

    #[test]
    fn vert_indiv_multiple_loose_verts() {
        // Each input vert gets its own duplicate and wire edge, independently.
        let mut mesh = LoopMesh::new();
        let positions = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
        ];
        let verts: Vec<VertexKey> = positions
            .iter()
            .map(|&p| mesh.create_vertex(p))
            .collect();

        let result = mesh.extrude_vert_indiv(&verts);

        assert_eq!(result.vert_map.len(), 3);
        assert_eq!(result.wire_edges.len(), 3);
        assert_eq!(mesh.totvert(), 6, "3 originals + 3 duplicates");
        assert_eq!(mesh.totedge(), 3, "one wire edge per input vert");

        for (i, &(source, duplicate)) in result.vert_map.iter().enumerate() {
            assert_eq!(source, verts[i]);
            assert_eq!(mesh.vertex_position(duplicate), positions[i]);
        }
        mesh.validate_topology();
    }

    #[test]
    fn vert_indiv_duplicate_input_deduped() {
        // Naming the same vertex twice produces a single duplicate and edge.
        let mut mesh = LoopMesh::new();
        let v = mesh.create_vertex(Vec3::new(0.0, 0.0, 0.0));

        let result = mesh.extrude_vert_indiv(&[v, v]);

        assert_eq!(result.vert_map.len(), 1, "deduped to one duplicate");
        assert_eq!(result.wire_edges.len(), 1);
        assert_eq!(mesh.totvert(), 2);
        assert_eq!(mesh.totedge(), 1);
        mesh.validate_topology();
    }

    #[test]
    fn vert_indiv_vert_on_face_does_not_split() {
        // A vertex incident to a face is duplicated and wired, but the face is
        // left intact — purely additive, no split.
        let mut mesh = LoopMesh::new();
        let quad = create_face_from_verts(
            &mut mesh,
            &[
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(1.0, 1.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
            ],
        );
        let corner = mesh.face_vert_keys(quad)[0];
        let edges_before = mesh.totedge();

        let result = mesh.extrude_vert_indiv(&[corner]);

        assert_eq!(result.vert_map.len(), 1);
        assert_eq!(result.wire_edges.len(), 1);
        // The quad is untouched: still one face, still 4 verts on it.
        assert_eq!(mesh.totface(), 1, "face is not split");
        assert_eq!(mesh.face_vert_keys(quad).len(), 4, "face still a quad");
        assert!(mesh.face_keys().contains(&quad), "original face survives");
        // Additive: one new vert, one new edge.
        assert_eq!(mesh.totvert(), 5, "4 quad verts + 1 duplicate");
        assert_eq!(mesh.totedge(), edges_before + 1, "only the wire edge added");

        // The duplicate is wired to the corner but is NOT part of the face.
        let (_source, duplicate) = result.vert_map[0];
        assert!(
            !mesh.face_vert_keys(quad).contains(&duplicate),
            "duplicate is not woven into the face"
        );
        mesh.validate_topology();
    }

    #[test]
    fn vert_indiv_wire_endpoint_swaps_winding() {
        // §11.1 wind-swap: a wire endpoint that is v1 (the first stored
        // endpoint) of its sole wire edge gets its new wire wound
        // duplicate -> original; the second stored endpoint keeps the default
        // source -> duplicate winding.
        let mut mesh = LoopMesh::new();
        let a = mesh.create_vertex(Vec3::new(0.0, 0.0, 0.0));
        let b = mesh.create_vertex(Vec3::new(1.0, 0.0, 0.0));
        // Wire edge stored [a, b]: a is the first stored endpoint (v1), b the second.
        let _wire = mesh.create_edge([a, b], EdgeCreateFlags::NONE);

        let result = mesh.extrude_vert_indiv(&[a, b]);

        let new_edge_for = |source: VertexKey| -> [VertexKey; 2] {
            let index = result
                .vert_map
                .iter()
                .position(|&(s, _)| s == source)
                .unwrap();
            mesh.edge_vertices(result.wire_edges[index])
        };

        let (_s_a, dup_a) = *result.vert_map.iter().find(|&&(s, _)| s == a).unwrap();
        let (_s_b, dup_b) = *result.vert_map.iter().find(|&&(s, _)| s == b).unwrap();

        // a is the first stored endpoint (v1) of the wire -> swap -> [dup_a, a].
        assert_eq!(new_edge_for(a), [dup_a, a], "first wire endpoint swaps winding");
        // b is the second stored endpoint -> no swap -> default [b, dup_b].
        assert_eq!(new_edge_for(b), [b, dup_b], "second endpoint keeps default winding");

        mesh.validate_topology();
    }
}
