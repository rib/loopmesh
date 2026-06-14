use crate::loopmesh::*;

impl LoopMesh {
    /// Splits `face` along a new edge connecting `l_v1.vertex` and `l_v2.vertex`.
    /// Both loops must currently belong to `face`. Returns `(new_face, new_edge, new_loop_on_new_face)`.
    ///
    /// After the split:
    ///
    /// - `face` retains the loops between `l_v1` and `l_v2` (inclusive) walking
    ///   forwards from `l_v1`, plus a new loop sitting at `l_v2.vertex`.
    /// - The new face takes the loops between `l_v2` and `l_v1` (the other half
    ///   of the original cycle), plus a new loop sitting at `l_v1.vertex`.
    /// - The new edge connects the two split vertices and has both of the new
    ///   loops on its radial cycle.
    pub fn euler_split_face_make_edge(
        &mut self,
        face: FaceKey,
        l_v1: LoopKey,
        l_v2: LoopKey,
    ) -> (FaceKey, EdgeKey, LoopKey) {
        self.euler_split_face_make_edge_reusing(face, l_v1, l_v2, false)
    }

    /// SFME with the edge-reuse variant (RADIAL_EDGE_STRUCTURE.md §8.3.1).
    /// When `reuse_existing_edge` is set and `l_v1.vertex`–`l_v2.vertex` are
    /// already joined by an interior/wire edge off this face's loop cycle, that
    /// edge is adopted as the splitting edge instead of a fresh one being
    /// allocated (`0 E` instead of `+1 E`); otherwise the behaviour is
    /// identical to the default fresh-edge path. The reuse path can leave a
    /// duplicate sub-face — detection and removal are the caller's
    /// responsibility (§8.3.1).
    pub fn euler_split_face_make_edge_reusing(
        &mut self,
        face: FaceKey,
        l_v1: LoopKey,
        l_v2: LoopKey,
        reuse_existing_edge: bool,
    ) -> (FaceKey, EdgeKey, LoopKey) {
        debug_assert_eq!(self.loops.get(l_v1).face, face);
        debug_assert_eq!(self.loops.get(l_v2).face, face);
        debug_assert_ne!(l_v1, l_v2);

        let v1 = self.loops.get(l_v1).vertex;
        let v2 = self.loops.get(l_v2).vertex;

        // 1. Resolve the splitting edge: adopt an existing v1–v2 edge on the
        //    reuse path, otherwise allocate a fresh one.
        let new_edge = if reuse_existing_edge {
            self.query_edge_exists(v1, v2)
                .unwrap_or_else(|| self.create_edge([v1, v2], EdgeCreateFlags::NONE))
        } else {
            self.create_edge([v1, v2], EdgeCreateFlags::NONE)
        };

        // 2. Allocate the new face via direct slab access (face_create
        //    expects a verts/edges list — not applicable here).
        let face2 = self.alloc_empty_face(l_v2);

        // 3. Allocate the two new loops. `l_f1` joins face's surviving half,
        //    `l_f2` joins face2.
        let l_f1 = self.create_loop(v2, face);
        let l_f2 = self.create_loop(v1, face2);

        // 4. Splice both new loops into the face loop cycle. Capture the
        //    original prev pointers first.
        let l_v1_prev_orig = self.loops.get(l_v1).face_loop_links.prev;
        let l_v2_prev_orig = self.loops.get(l_v2).face_loop_links.prev;

        self.loops.get_mut(l_f1).face_loop_links = DoubleLinks {
            prev: l_v2_prev_orig,
            next: l_v1,
        };
        self.loops.get_mut(l_f2).face_loop_links = DoubleLinks {
            prev: l_v1_prev_orig,
            next: l_v2,
        };
        self.loops.get_mut(l_v2_prev_orig).face_loop_links.next = l_f1;
        self.loops.get_mut(l_v1_prev_orig).face_loop_links.next = l_f2;
        self.loops.get_mut(l_v1).face_loop_links.prev = l_f1;
        self.loops.get_mut(l_v2).face_loop_links.prev = l_f2;

        // 5. Hook the two new loops into new_edge's radial cycle.
        self.face_loop_connect_edge(new_edge, l_f1);
        self.face_loop_connect_edge(new_edge, l_f2);

        // 6. Resolve which side has the original first_edge_loop. Walk
        //    face's new cycle (starting from l_f1); if it contains the
        //    original first loop, face keeps it; otherwise face2 takes it.
        let original_first = self.faces.get(face).first_edge_loop;
        let face1_loops: Vec<LoopKey> = self.face_loop_iter(l_f1).collect();
        let first_on_face1 = face1_loops.contains(&original_first);

        if first_on_face1 {
            // face keeps original_first. Pick a "near" loop on face2 as its
            // first, so the two halves stay aligned.
            let orig_prev = self.loops.get(original_first).face_loop_links.prev;
            let orig_next = self.loops.get(original_first).face_loop_links.next;
            let face2_first = if orig_prev == l_f1 {
                self.loops.get(l_f2).face_loop_links.prev
            } else if orig_next == l_f1 {
                self.loops.get(l_f2).face_loop_links.next
            } else {
                l_f2
            };
            self.faces.get_mut(face2).first_edge_loop = face2_first;
        } else {
            // Original first was on face2's side. Swap them.
            self.faces.get_mut(face2).first_edge_loop = original_first;
            let orig_prev = self.loops.get(original_first).face_loop_links.prev;
            let orig_next = self.loops.get(original_first).face_loop_links.next;
            let face1_first = if orig_prev == l_f2 {
                self.loops.get(l_f1).face_loop_links.prev
            } else if orig_next == l_f2 {
                self.loops.get(l_f1).face_loop_links.next
            } else {
                l_f1
            };
            self.faces.get_mut(face).first_edge_loop = face1_first;
        }

        // 7. Re-stamp face pointers for loops that now belong to face2, and
        //    recompute both face lengths.
        let face2_first = self.faces.get(face2).first_edge_loop;
        let face2_loops: Vec<LoopKey> = self.face_loop_iter(face2_first).collect();
        for &l in &face2_loops {
            self.loops.get_mut(l).face = face2;
        }
        self.faces.get_mut(face2).len = face2_loops.len() as u32;

        let face1_first = self.faces.get(face).first_edge_loop;
        let face1_len = self.face_loop_iter(face1_first).count();
        self.faces.get_mut(face).len = face1_len as u32;

        (face2, new_edge, l_f2)
    }
}

#[cfg(test)]
mod tests {
    use glam::Vec3;

    use crate::loopmesh::test_support::create_face_from_verts;
    use crate::loopmesh::*;

    /// Split a quad along its V0-V2 diagonal: should produce two triangles
    /// that share the new edge.
    #[test]
    fn test_euler_split_face_make_edge_quad_diagonal() {
        let mut mesh = LoopMesh::new();
        let verts = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::new(2.0, 1.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        ];
        let face = create_face_from_verts(&mut mesh, &verts);
        let face_verts_orig = mesh.face_vert_keys(face);
        assert_eq!(face_verts_orig.len(), 4);
        let v0 = face_verts_orig[0];
        let v2 = face_verts_orig[2];

        // Find the two loops in face that sit at v0 and v2.
        let face_loops_orig = mesh.face_loop_keys(face);
        let l_v0 = *face_loops_orig
            .iter()
            .find(|&&l| mesh.loop_vertex(l) == v0)
            .unwrap();
        let l_v2 = *face_loops_orig
            .iter()
            .find(|&&l| mesh.loop_vertex(l) == v2)
            .unwrap();

        let pre_v = mesh.vertex_count();
        let pre_e = mesh.edge_count();
        let pre_f = mesh.face_count();

        let (face2, new_edge, _l_new) = mesh.euler_split_face_make_edge(face, l_v0, l_v2);

        assert_eq!(mesh.vertex_count(), pre_v, "no vertex change");
        assert_eq!(mesh.edge_count(), pre_e + 1, "one edge added");
        assert_eq!(mesh.face_count(), pre_f + 1, "one face added");

        // The new edge connects v0 <-> v2.
        let ne_verts = mesh.edge_vertices(new_edge);
        assert!(
            ne_verts == [v0, v2] || ne_verts == [v2, v0],
            "new edge should connect v0<->v2, got {:?}",
            ne_verts
        );

        // Each half should be a triangle.
        assert_eq!(mesh.face_len(face), 3);
        assert_eq!(mesh.face_len(face2), 3);

        // The new edge's radial cycle should contain exactly both faces.
        let radial_faces: std::collections::HashSet<FaceKey> =
            mesh.edge_face_keys(new_edge).into_iter().collect();
        assert_eq!(radial_faces.len(), 2);
        assert!(radial_faces.contains(&face));
        assert!(radial_faces.contains(&face2));

        // Together both faces still cover the original 4 vertices.
        let mut all_verts: std::collections::HashSet<VertexKey> =
            mesh.face_vert_keys(face).into_iter().collect();
        all_verts.extend(mesh.face_vert_keys(face2));
        assert_eq!(all_verts.len(), 4);
        for v in &face_verts_orig {
            assert!(all_verts.contains(v));
        }

        // Each face's edges must all reference that face on their radial.
        for f in [face, face2] {
            for e in mesh.face_edge_keys(f) {
                let rf = mesh.edge_face_keys(e);
                assert!(
                    rf.contains(&f),
                    "face {} missing from radial of edge {}",
                    f,
                    e
                );
            }
        }
    }
}
