use crate::loopmesh::*;

/// Removal mode for [`LoopMesh::delete`].
///
/// Mirrors the seven-mode context taxonomy in
/// `radial-mesh-guides/operators/DELETE.md` §2. The context decides how the
/// seed set is expanded (downward to incident lower-dimensional elements,
/// upward to bounded higher-dimensional elements, or not at all) and which of
/// those elements are killed versus spared.
///
/// Only the two face-seeded modes that differ *solely* in whether the downward
/// cascade runs are implemented today (§4 / §5.1). The remaining five modes
/// (`verts`, `edges`, `edges+faces`, `faces-keep-edges`, `tagged-only`) are
/// documented in the guide but not yet built; new variants slot in here without
/// disturbing the existing two.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DeleteContext {
    /// Kill the seeded faces, plus any boundary edge / vert used by *no*
    /// surviving face — geometry the deleted region solely owned. Edges and
    /// verts still shared with an out-of-set face are spared. (§4.)
    Faces,
    /// Kill only the seeded face headers. The entire boundary wireframe
    /// (shared *and* solely-owned) survives. No downward cascade. (§5.1.)
    FacesOnly,
}

impl LoopMesh {
    /// Remove the seeded `faces` according to `context`.
    ///
    /// Implements the face-seeded modes of
    /// `radial-mesh-guides/operators/DELETE.md`:
    ///
    /// - [`DeleteContext::FacesOnly`] (§5.1) — the non-cascading face sweep.
    ///   Every seeded face header and its loops are freed; every boundary edge
    ///   and vert is left standing as wire. No mark phase.
    /// - [`DeleteContext::Faces`] (§4) — the mark/unmark/sweep of §4.3. The
    ///   seeded faces die, and so does every boundary edge/vert the region
    ///   solely owned; geometry still shared with a surviving face is kept (the
    ///   §4.2 shared-boundary keep-alive).
    ///
    /// Delete creates no replacement geometry, so there is no customdata
    /// interpolation: survivors keep their attributes byte-for-byte and killed
    /// elements' attributes are discarded with them.
    ///
    /// The seed is a face slice because both implemented contexts are
    /// face-seeded (verts/edges in the guide's mixed `geom` buffer are ignored
    /// by these contexts — §4.1). The lower-dimension-seeded modes, when added,
    /// will take their own seed shape.
    pub fn delete(&mut self, faces: &[FaceKey], context: DeleteContext) {
        if faces.is_empty() {
            return;
        }
        match context {
            DeleteContext::FacesOnly => self.delete_faces_only(faces),
            DeleteContext::Faces => self.delete_faces(faces),
        }
    }

    /// §5.1 — kill exactly the seeded faces, touch nothing else.
    ///
    /// `face_kill` (DELETE.md §3) is the non-cascading primitive: it unlinks
    /// the face's loop cycle from its boundary edges' radial cycles and frees
    /// the face header and loops, leaving every 1- and 0-cell alone. So the
    /// whole operator is one face sweep over the seed.
    fn delete_faces_only(&mut self, faces: &[FaceKey]) {
        for &face in faces {
            // Re-test liveness: a face may appear twice in the seed, or have
            // been freed by an earlier kill in this same call.
            if self.faces.try_get(face).is_some() {
                self.face_kill(face);
            }
        }
    }

    /// §4 — kill the seeded faces plus the boundary they solely owned.
    ///
    /// Three mesh-wide mark passes (§4.3) compute the closed kill set, then the
    /// face→edge→vert sweep (§3.1) frees it bottom-of-the-cascade-first so each
    /// kill in the sweep is a leaf kill.
    fn delete_faces(&mut self, faces: &[FaceKey]) {
        let seeded = self.push_face_op_flag_layer();
        let doomed_vert = self.push_vert_op_flag_layer();
        let doomed_edge = self.push_edge_op_flag_layer();
        let doomed_face = self.push_face_op_flag_layer();

        const SEED: u8 = 0;
        const DOOM: u8 = 0;

        // The seed faces themselves are always killed; mark them directly. A
        // separate `seeded` bit drives the §4.3 over-mark / un-doom passes,
        // since those test face membership in the input set.
        for &face in faces {
            if self.faces.try_get(face).is_some() {
                self.face_op_flag_set(seeded, face, SEED);
                self.face_op_flag_set(doomed_face, face, DOOM);
            }
        }

        // Phase 1a — over-mark: tentatively doom every boundary vert/edge of
        // every seeded face. Optimistically assume the whole boundary goes.
        let mut face_iter = MeshFaces::default();
        while let Some(face) = face_iter.next(self) {
            if !self.face_op_flag_test(seeded, face, SEED) {
                continue;
            }
            let mut loop_iter = FaceLoops::new(self, face);
            while let Some(corner) = loop_iter.next(self) {
                let vert = self.loop_vertex(corner);
                let edge = self.loop_edge(corner);
                self.vert_op_flag_set(doomed_vert, vert, DOOM);
                self.edge_op_flag_set(doomed_edge, edge, DOOM);
            }
        }

        // Phase 1b — un-doom survivors: any boundary vert/edge a NON-seeded
        // face also touches must be kept (the §4.2 shared-boundary keep-alive).
        let mut face_iter = MeshFaces::default();
        while let Some(face) = face_iter.next(self) {
            if self.face_op_flag_test(seeded, face, SEED) {
                continue;
            }
            let mut loop_iter = FaceLoops::new(self, face);
            while let Some(corner) = loop_iter.next(self) {
                let vert = self.loop_vertex(corner);
                let edge = self.loop_edge(corner);
                self.vert_op_flag_clear(doomed_vert, vert, DOOM);
                self.edge_op_flag_clear(doomed_edge, edge, DOOM);
            }
        }

        // Phase 1c — protect verts of surviving edges: a vert on any surviving
        // (un-doomed) edge is kept, even if every face it touched is doomed.
        // This makes the vert rule "dies iff every incident *edge* dies" rather
        // than "iff every incident *face* dies" — the wire-edge keep-alive of
        // §4.4. Must run after 1b so it sees the post-keep-alive edge marks.
        let mut edge_iter = MeshEdges::default();
        while let Some(edge) = edge_iter.next(self) {
            if self.edge_op_flag_test(doomed_edge, edge, DOOM) {
                continue;
            }
            let [v0, v1] = self.edge_vertices(edge);
            self.vert_op_flag_clear(doomed_vert, v0, DOOM);
            self.vert_op_flag_clear(doomed_vert, v1, DOOM);
        }

        // Phase 2 — sweep faces, then edges, then verts (the §3.1 invariant).
        // The mesh-wide lending iterators are cursor-based and re-test liveness
        // on each step, so they are kill-safe against elements an earlier kill
        // already freed via the kernel cascade.
        let mut face_iter = MeshFaces::default();
        while let Some(face) = face_iter.next(self) {
            if self.face_op_flag_test(doomed_face, face, DOOM) {
                self.face_kill(face);
            }
        }
        let mut edge_iter = MeshEdges::default();
        while let Some(edge) = edge_iter.next(self) {
            if self.edge_op_flag_test(doomed_edge, edge, DOOM) {
                self.edge_kill(edge);
            }
        }
        let mut vert_iter = MeshVerts::default();
        while let Some(vert) = vert_iter.next(self) {
            if self.vert_op_flag_test(doomed_vert, vert, DOOM) {
                self.vertex_kill(vert);
            }
        }

        self.pop_face_op_flag_layer(doomed_face);
        self.pop_edge_op_flag_layer(doomed_edge);
        self.pop_vert_op_flag_layer(doomed_vert);
        self.pop_face_op_flag_layer(seeded);
    }
}

#[cfg(test)]
mod tests {
    use glam::Vec3;

    use crate::loopmesh::*;

    /// Build an `n × n` grid of quad faces (so `(n+1) × (n+1)` verts). Returns
    /// the faces in row-major order: face `(r, c)` is at index `r * n + c`.
    fn build_quad_grid(mesh: &mut LoopMesh, n: usize) -> Vec<FaceKey> {
        let side = n + 1;
        let mut verts = Vec::with_capacity(side * side);
        for r in 0..side {
            for c in 0..side {
                verts.push(mesh.create_vertex(Vec3::new(c as f32, r as f32, 0.0)));
            }
        }
        let vert_at = |r: usize, c: usize| verts[r * side + c];

        let mut faces = Vec::with_capacity(n * n);
        for r in 0..n {
            for c in 0..n {
                let v00 = vert_at(r, c);
                let v01 = vert_at(r, c + 1);
                let v11 = vert_at(r + 1, c + 1);
                let v10 = vert_at(r + 1, c);
                let ring = [v00, v01, v11, v10];
                let mut edges = [EdgeKey::NULL; 4];
                for i in 0..4 {
                    let a = ring[i];
                    let b = ring[(i + 1) % 4];
                    edges[i] = mesh
                        .query_edge_exists(a, b)
                        .unwrap_or_else(|| mesh.create_edge([a, b], EdgeCreateFlags::NONE));
                }
                faces.push(mesh.face_create(&ring, &edges, FaceCreateFlags::NONE));
            }
        }
        faces
    }

    /// The interior 2×2 block of a 5×5 face grid, matching DELETE.md §7's
    /// worked example. Faces `(1,1) (1,2) (2,1) (2,2)` → indices 6, 7, 11, 12.
    fn interior_2x2_block(faces: &[FaceKey]) -> [FaceKey; 4] {
        let n = 5;
        [
            faces[1 * n + 1],
            faces[1 * n + 2],
            faces[2 * n + 1],
            faces[2 * n + 2],
        ]
    }

    #[test]
    fn faces_context_kills_solely_owned_geometry() {
        let mut mesh = LoopMesh::new();
        let faces = build_quad_grid(&mut mesh, 5);
        let (v0, e0, f0) = (mesh.totvert(), mesh.totedge(), mesh.totface());

        let block = interior_2x2_block(&faces);
        mesh.delete(&block, DeleteContext::Faces);

        // DELETE.md §7: −4 F, −4 E (the four interior cross-edges), −1 V (the
        // single centre vert where the cross-edges met).
        assert_eq!(f0 - mesh.totface(), 4, "faces");
        assert_eq!(e0 - mesh.totedge(), 4, "edges");
        assert_eq!(v0 - mesh.totvert(), 1, "verts");

        // The deleted block faces are gone.
        for &f in &block {
            assert!(mesh.faces.try_get(f).is_none());
        }
    }

    #[test]
    fn faces_context_keeps_outer_ring() {
        let mut mesh = LoopMesh::new();
        let faces = build_quad_grid(&mut mesh, 5);

        // Snapshot the ring edges/verts: every edge bounding a deleted block
        // face but shared with a surrounding (out-of-block) grid face.
        let block = interior_2x2_block(&faces);
        let block_set: std::collections::HashSet<FaceKey> = block.iter().copied().collect();

        let mut ring_edges = std::collections::HashSet::new();
        let mut ring_verts = std::collections::HashSet::new();
        for &f in &block {
            for e in mesh.face_edge_keys(f) {
                // A ring (shared) edge has a face on its radial cycle that is
                // not in the block.
                let mut iter = RadialLoops::new(&mesh, e);
                let mut shared = false;
                while let Some(l) = iter.next(&mesh) {
                    if !block_set.contains(&mesh.loop_face(l)) {
                        shared = true;
                    }
                }
                if shared {
                    ring_edges.insert(e);
                    let [a, b] = mesh.edge_vertices(e);
                    ring_verts.insert(a);
                    ring_verts.insert(b);
                }
            }
        }
        assert_eq!(ring_edges.len(), 8, "block perimeter is 8 shared edges");

        mesh.delete(&block, DeleteContext::Faces);

        for e in ring_edges {
            assert!(
                mesh.edges.try_get(e).is_some(),
                "shared ring edge must survive (§4.2 keep-alive)"
            );
        }
        for v in ring_verts {
            assert!(
                mesh.vertices.try_get(v).is_some(),
                "ring vert held by a surviving edge must survive (§4.3 1c)"
            );
        }
    }

    #[test]
    fn faces_only_context_keeps_whole_wireframe() {
        let mut mesh = LoopMesh::new();
        let faces = build_quad_grid(&mut mesh, 5);
        let (v0, e0, f0) = (mesh.totvert(), mesh.totedge(), mesh.totface());

        let block = interior_2x2_block(&faces);
        mesh.delete(&block, DeleteContext::FacesOnly);

        // DELETE.md §7 contrast pin: −4 F, 0 E, 0 V. The interior cross-edges
        // and centre vert that die under Faces all survive as wire here.
        assert_eq!(f0 - mesh.totface(), 4, "faces");
        assert_eq!(e0 - mesh.totedge(), 0, "edges");
        assert_eq!(v0 - mesh.totvert(), 0, "verts");

        for &f in &block {
            assert!(mesh.faces.try_get(f).is_none());
        }
    }

    #[test]
    fn faces_context_single_quad_drops_owned_boundary() {
        // A lone quad with no neighbours: every boundary edge and vert is
        // solely owned, so Faces removes the whole quad (§7, n=1, b=0).
        let mut mesh = LoopMesh::new();
        let faces = build_quad_grid(&mut mesh, 1);

        mesh.delete(&faces, DeleteContext::Faces);

        assert_eq!(mesh.totface(), 0);
        assert_eq!(mesh.totedge(), 0);
        assert_eq!(mesh.totvert(), 0);
    }

    #[test]
    fn faces_only_single_quad_leaves_wire_loop() {
        // The §5.1 worked sentence: faces-only on a single quad leaves a
        // four-edge wire loop with four corner verts.
        let mut mesh = LoopMesh::new();
        let faces = build_quad_grid(&mut mesh, 1);

        mesh.delete(&faces, DeleteContext::FacesOnly);

        assert_eq!(mesh.totface(), 0);
        assert_eq!(mesh.totedge(), 4);
        assert_eq!(mesh.totvert(), 4);
    }

    #[test]
    fn empty_seed_is_noop() {
        let mut mesh = LoopMesh::new();
        let faces = build_quad_grid(&mut mesh, 3);
        let (v0, e0, f0) = (mesh.totvert(), mesh.totedge(), mesh.totface());

        mesh.delete(&[], DeleteContext::Faces);
        mesh.delete(&[], DeleteContext::FacesOnly);

        assert_eq!((v0, e0, f0), (mesh.totvert(), mesh.totedge(), mesh.totface()));
    }
}
