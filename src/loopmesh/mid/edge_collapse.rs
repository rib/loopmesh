use crate::loopmesh::*;

impl LoopMesh {
    /// Collapses `edge` by merging `v_kill` onto the other endpoint. Returns
    /// the surviving vertex. Thin convenience wrapper over
    /// [`Self::euler_join_vert_kill_edge`].
    pub fn edge_collapse(&mut self, edge: EdgeKey, v_kill: VertexKey) -> Option<VertexKey> {
        // Customdata: blend the surviving vertex's CD to the 50/50 mix of the
        // two endpoints. Performed before the JVKE call so v_kill's attribute
        // slot is still live; if v_kill isn't on `edge`, JVKE rejects and the
        // interp above is a no-op overwrite of v_keep with its own value.
        if let Some(v_keep) = self.edges.get(edge).other_vertex(v_kill) {
            let sources = [(0.5, v_keep.as_usize()), (0.5, v_kill.as_usize())];
            self.vertex_attribs
                .interpolate_into(v_keep.as_usize(), &sources);
        }
        self.euler_join_vert_kill_edge(edge, v_kill)
    }

    /// Collapses `v_kill` along `e_kill` onto the edge's other endpoint. Same
    /// outcome as `edge_collapse`, but assumes `v_kill` is degree-2 and routes
    /// through JEKV rather than the general vertex-kill path.
    pub fn vert_collapse_edge(&mut self, e_kill: EdgeKey, v_kill: VertexKey) -> Option<EdgeKey> {
        // Degree-2 v_kill is JEKV's precondition.
        self.euler_join_edge_kill_vert(e_kill, v_kill)
    }
}
