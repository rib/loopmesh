use super::*;

impl LoopMesh {
    pub fn vertices_ensure_index_field(&mut self) {
        if self.flags & LoopMeshFlags::VERTICES_INDEX_DIRTY == LoopMeshFlags::NONE {
            return;
        }
        // We're about to walk every live vertex; populate the table at the
        // same time to keep both forms cheap.
        self.vertices_index.clear();
        let mut it = MeshVerts::default();
        let mut i: u32 = 0;
        while let Some(v) = it.next(self) {
            self.vertices.get_mut(v).index = i;
            self.vertices_index.push(v);
            i += 1;
        }
        self.flags &= !LoopMeshFlags::VERTICES_INDEX_DIRTY;
    }

    pub fn edges_ensure_index_field(&mut self) {
        if self.flags & LoopMeshFlags::EDGES_INDEX_DIRTY == LoopMeshFlags::NONE {
            return;
        }
        self.edges_index.clear();
        let mut it = MeshEdges::default();
        let mut i: u32 = 0;
        while let Some(e) = it.next(self) {
            self.edges.get_mut(e).index = i;
            self.edges_index.push(e);
            i += 1;
        }
        self.flags &= !LoopMeshFlags::EDGES_INDEX_DIRTY;
    }

    pub fn faces_ensure_index_field(&mut self) {
        if self.flags & LoopMeshFlags::FACES_INDEX_DIRTY == LoopMeshFlags::NONE {
            return;
        }
        self.faces_index.clear();
        let mut it = MeshFaces::default();
        let mut i: u32 = 0;
        while let Some(f) = it.next(self) {
            self.faces.get_mut(f).index = i;
            self.faces_index.push(f);
            i += 1;
        }
        self.flags &= !LoopMeshFlags::FACES_INDEX_DIRTY;
    }

    pub fn loops_ensure_index_field(&mut self) {
        if self.flags & LoopMeshFlags::LOOPS_INDEX_DIRTY == LoopMeshFlags::NONE {
            return;
        }
        self.loops_index.clear();
        let mut it = MeshLoops::default();
        let mut i: u32 = 0;
        while let Some(l) = it.next(self) {
            self.loops.get_mut(l).index = i;
            self.loops_index.push(l);
            i += 1;
        }
        self.flags &= !LoopMeshFlags::LOOPS_INDEX_DIRTY;
    }

    /// Populate `vertices_index: Vec<VertexKey>`. Implies
    /// [`vertices_ensure_index_field`].
    pub fn vertices_ensure_table(&mut self) {
        self.vertices_ensure_index_field();
    }

    pub fn edges_ensure_table(&mut self) {
        self.edges_ensure_index_field();
    }

    pub fn faces_ensure_table(&mut self) {
        self.faces_ensure_index_field();
    }

    pub fn loops_ensure_table(&mut self) {
        self.loops_ensure_index_field();
    }

    /// Convenience: ensure indices and tables are fresh on every axis.
    pub fn vertices_ensure_index(&mut self) {
        self.vertices_ensure_index_field();
    }

    pub fn edges_ensure_index(&mut self) {
        self.edges_ensure_index_field();
    }

    pub fn faces_ensure_index(&mut self) {
        self.faces_ensure_index_field();
    }

    pub fn loops_ensure_index(&mut self) {
        self.loops_ensure_index_field();
    }

    pub fn all_indices_ensure(&mut self) {
        self.vertices_ensure_index_field();
        self.edges_ensure_index_field();
        self.faces_ensure_index_field();
        self.loops_ensure_index_field();
    }

    pub fn is_vertices_index_dirty(&self) -> bool {
        self.flags & LoopMeshFlags::VERTICES_INDEX_DIRTY != LoopMeshFlags::NONE
    }

    pub fn is_edges_index_dirty(&self) -> bool {
        self.flags & LoopMeshFlags::EDGES_INDEX_DIRTY != LoopMeshFlags::NONE
    }

    pub fn is_faces_index_dirty(&self) -> bool {
        self.flags & LoopMeshFlags::FACES_INDEX_DIRTY != LoopMeshFlags::NONE
    }

    pub fn is_loops_index_dirty(&self) -> bool {
        self.flags & LoopMeshFlags::LOOPS_INDEX_DIRTY != LoopMeshFlags::NONE
    }

    /// Debug pass: assert every live element's cached `index` matches its
    /// position in the dense table, and that no dirty bit is set. Run after
    /// [`all_indices_ensure`] to verify integrity.
    ///
    /// Also compiled under `cfg(test)` so the in-crate unit tests still
    /// build when the harness is compiled with debug assertions off (as
    /// `cargo bench` does).
    #[cfg(any(test, debug_assertions))]
    pub fn validate_indices(&self) {
        assert!(
            !self.is_vertices_index_dirty(),
            "validate_indices: vertex indices are dirty; call vertices_ensure_index() first"
        );
        assert!(
            !self.is_edges_index_dirty(),
            "validate_indices: edge indices are dirty"
        );
        assert!(
            !self.is_faces_index_dirty(),
            "validate_indices: face indices are dirty"
        );
        assert!(
            !self.is_loops_index_dirty(),
            "validate_indices: loop indices are dirty"
        );

        for (i, &v) in self.vertices_index.iter().enumerate() {
            let cached = self.vertices.get(v).index as usize;
            assert_eq!(
                cached, i,
                "vertex {v:?}: cached index {cached} disagrees with table position {i}"
            );
        }
        for (i, &e) in self.edges_index.iter().enumerate() {
            let cached = self.edges.get(e).index as usize;
            assert_eq!(
                cached, i,
                "edge {e:?}: cached index {cached} disagrees with table position {i}"
            );
        }
        for (i, &f) in self.faces_index.iter().enumerate() {
            let cached = self.faces.get(f).index as usize;
            assert_eq!(
                cached, i,
                "face {f:?}: cached index {cached} disagrees with table position {i}"
            );
        }
        for (i, &l) in self.loops_index.iter().enumerate() {
            let cached = self.loops.get(l).index as usize;
            assert_eq!(
                cached, i,
                "loop {l:?}: cached index {cached} disagrees with table position {i}"
            );
        }
    }

    /// Read the cached dense index of a vertex. Caller is responsible for
    /// having called `vertices_ensure_index()` since the last mutation.
    pub fn vertex_index(&self, v: VertexKey) -> u32 {
        self.vertices.get(v).index
    }

    pub fn edge_index(&self, e: EdgeKey) -> u32 {
        self.edges.get(e).index
    }

    pub fn face_index(&self, f: FaceKey) -> u32 {
        self.faces.get(f).index
    }

    pub fn loop_index(&self, l: LoopKey) -> u32 {
        self.loops.get(l).index
    }

    /// Read the index-table slice (matches the keys' cached `index` field).
    pub fn vertex_index_table(&self) -> &[VertexKey] {
        &self.vertices_index
    }

    pub fn edge_index_table(&self) -> &[EdgeKey] {
        &self.edges_index
    }

    pub fn face_index_table(&self) -> &[FaceKey] {
        &self.faces_index
    }

    pub fn loop_index_table(&self) -> &[LoopKey] {
        &self.loops_index
    }
}
