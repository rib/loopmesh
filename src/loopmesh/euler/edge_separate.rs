use crate::loopmesh::*;

impl LoopMesh {
    /// Detaches one specific loop `l_sep` from `edge`'s radial cycle and
    /// re-anchors it on a new duplicate edge between the same two vertices.
    ///
    /// Returns the new edge. Caller must ensure `loop_edge(l_sep) == Some(edge)`
    /// and the edge has more than one loop on its radial cycle (otherwise the
    /// op is a no-op and we panic to surface the misuse).
    pub fn euler_edge_separate(&mut self, edge: EdgeKey, l_sep: LoopKey) -> EdgeKey {
        debug_assert_eq!(self.loops.get(l_sep).edge, edge);
        debug_assert!(
            self.edge_radial_iter(edge).count() >= 2,
            "edge_separate on a boundary edge would be a no-op"
        );

        let verts = self.edges.get(edge).vertices;
        let e_new = self.create_edge(verts, EdgeCreateFlags::NONE);
        // Customdata: the duplicate inherits the source edge's CD.
        self.edge_attribs
            .copy_into(e_new.as_usize(), edge.as_usize());
        self.face_loop_disconnect_edge(l_sep);
        self.face_loop_connect_edge(e_new, l_sep);
        e_new
    }
}

#[cfg(test)]
mod tests {
    use crate::loopmesh::test_support::mesh_add_cube;
    use crate::loopmesh::*;

    /// edge_separate then edge_splice round-trip on a cube interior edge.
    #[test]
    fn test_edge_separate_then_splice_cube_interior() {
        let mut mesh = LoopMesh::new();
        mesh_add_cube(&mut mesh);

        // Pick any interior edge — every cube edge has radial length 2.
        let interior_edge = *mesh
            .edge_keys()
            .iter()
            .find(|&&e| mesh.edge_radial_iter(e).count() == 2)
            .expect("cube has interior edges");

        // Snapshot the cube's totals.
        let v0_before = mesh.totvert();
        let e0_before = mesh.totedge();
        let f0_before = mesh.totface();

        // Separate one loop off the edge.
        let l_sep = mesh.edge_radial_iter(interior_edge).next().unwrap();
        let e_new = mesh.euler_edge_separate(interior_edge, l_sep);

        assert_eq!(mesh.totedge(), e0_before + 1);
        assert_eq!(mesh.edge_radial_iter(interior_edge).count(), 1);
        assert_eq!(mesh.edge_radial_iter(e_new).count(), 1);

        // Splice them back together. After: cube topology restored.
        assert!(mesh.edge_splice(interior_edge, e_new));
        assert_eq!(mesh.totvert(), v0_before);
        assert_eq!(mesh.totedge(), e0_before);
        assert_eq!(mesh.totface(), f0_before);
        assert_eq!(mesh.edge_radial_iter(interior_edge).count(), 2);
    }
}
