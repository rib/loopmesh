use glam::Vec3;

use crate::loopmesh::*;

impl LoopMesh {
    /// Inserts a center vertex at the median position of `face` and
    /// fan-triangulates: each existing perimeter edge becomes the base of
    /// a triangle that includes the new center vertex. The original face
    /// is killed; `face.len()` new triangle faces are created. Returns
    /// the new center vertex.
    pub fn face_poke(&mut self, face: FaceKey) -> VertexKey {
        // Snapshot the face's perimeter (verts + their outgoing edges + the
        // original loop at each corner) in a single forward walk. SmallVec
        // keeps small fans off the heap. The original-loop snapshot serves
        // the customdata copy further below.
        let mut verts: smallvec::SmallVec<[VertexKey; 8]> = smallvec::SmallVec::new();
        let mut perim_edges: smallvec::SmallVec<[EdgeKey; 8]> = smallvec::SmallVec::new();
        let mut orig_loops: smallvec::SmallVec<[LoopKey; 8]> = smallvec::SmallVec::new();
        let mut it = self.face_loops(face);
        while let Some(l) = it.next(self) {
            let lref = self.loops.get(l);
            verts.push(lref.vertex);
            orig_loops.push(l);
            debug_assert!(
                !lref.edge.is_null(),
                "face_poke: loop on perimeter has no edge"
            );
            perim_edges.push(lref.edge);
        }
        let n = verts.len();
        debug_assert!(n >= 3, "face_poke needs at least 3 vertices");
        debug_assert_eq!(perim_edges.len(), n);

        // Median centre.
        let mut center = Vec3::ZERO;
        for &v in &verts {
            center += self.vertex_position(v);
        }
        center /= n as f32;

        // Snapshot the source face's slot — face_kill at the end recycles
        // it, but the slot's CD bucket entry stays addressable for the copy
        // calls in the new-triangle loop.
        let face_slot = face.as_usize();
        let do_loop_interp = !self.loop_attribs.is_empty();

        // NOTE: We delay face_kill until after the new triangles' loops
        // have been copy_into'd from the original loops. Killing the face
        // up front would recycle the original loop slab slots, breaking
        // the loop CD source-of-truth. The perimeter edges happily carry
        // two loops on their radial in the meantime — the original (from
        // `face`) plus the new triangle's. face_kill at the end removes
        // the originals.

        // Build the center vertex.
        let v_center = self.create_vertex(center);

        // Customdata: the centre vertex is the uniform average of every
        // corner.
        let w = 1.0_f32 / n as f32;
        let vert_sources: smallvec::SmallVec<[(f32, usize); 8]> =
            verts.iter().map(|v| (w, v.as_usize())).collect();
        self.vertex_attribs
            .interpolate_into(v_center.as_usize(), &vert_sources);

        // Pre-compute the loop-average source list (one entry per original
        // corner-loop, uniform weight). Reused per spawned triangle for the
        // centre-vertex loop's interp.
        let loop_avg_sources: smallvec::SmallVec<[(f32, usize); 8]> = if do_loop_interp {
            orig_loops.iter().map(|l| (w, l.as_usize())).collect()
        } else {
            smallvec::SmallVec::new()
        };

        // Create the N center→perimeter edges (stack-allocated for small faces).
        let mut center_edges: smallvec::SmallVec<[EdgeKey; 8]> = smallvec::SmallVec::new();
        for &v in &verts {
            center_edges.push(self.create_edge([v_center, v], EdgeCreateFlags::NONE));
        }

        // Build N triangles, each (verts[i], verts[i+1], v_center) with
        // edges (perim_edges[i], center_edges[i+1], center_edges[i]).
        // Each fan face inherits the source face's CD (any custom face
        // attribs). Loop-level CD: the two perimeter corners
        // copy from the original loops at the same vertices; the centre
        // corner takes the uniform average.
        for i in 0..n {
            let j = (i + 1) % n;
            let tri_verts = [verts[i], verts[j], v_center];
            let tri_edges = [perim_edges[i], center_edges[j], center_edges[i]];
            let f_new = self.face_create(&tri_verts, &tri_edges, FaceCreateFlags::NONE);
            self.face_attribs.copy_into(f_new.as_usize(), face_slot);

            if do_loop_interp {
                // The new face's loops sit at verts[i], verts[j], v_center
                // in that order in the loop cycle.
                let l_i = self.faces.get(f_new).first_edge_loop;
                let l_j = self.loops.get(l_i).face_loop_links.next;
                let l_c = self.loops.get(l_j).face_loop_links.next;

                self.loop_attribs
                    .copy_into(l_i.as_usize(), orig_loops[i].as_usize());
                self.loop_attribs
                    .copy_into(l_j.as_usize(), orig_loops[j].as_usize());
                self.loop_attribs
                    .interpolate_into(l_c.as_usize(), &loop_avg_sources);
            }
        }

        // Now safe to kill the original face — its loops were the source
        // for the copies above and are no longer needed.
        self.face_kill(face);

        v_center
    }
}
