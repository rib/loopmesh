use crate::loopmesh::*;

impl LoopMesh {
    /// Dissolves `v` — collapses it away by either killing it (isolated),
    /// collapsing the surrounding degree-2 chain, or merging its surrounding
    /// face fan and collapsing.
    ///
    /// Returns `true` on success. On failure (boundary, non-manifold, bowtie,
    /// or other unsupported topology) returns `false` and **the mesh is
    /// unchanged**.
    pub fn vert_dissolve(&mut self, v: VertexKey) -> bool {
        // Snapshot v's disk edges + per-edge radial counts in a single walk,
        // off the heap for typical valences.
        let edges: smallvec::SmallVec<[EdgeKey; 8]> =
            collect_iter::<_, 8>(self, self.disk_edges(v));
        let ne = edges.len();

        // Isolated vertex: just kill it.
        if ne == 0 {
            self.vertex_kill(v);
            return true;
        }

        // Wire-only vertex with one edge: kill (cascades to the edge).
        if ne == 1 {
            self.vertex_kill(v);
            return true;
        }

        // Pre-classify the topology so we never start mutating before
        // confirming the operation is supported.
        let radial_counts: smallvec::SmallVec<[usize; 8]> = edges
            .iter()
            .map(|&e| {
                let mut it = self.radial_loops(e);
                let mut c = 0usize;
                while it.next(self).is_some() {
                    c += 1;
                }
                c
            })
            .collect();
        let is_wire = radial_counts.iter().all(|&c| c == 0);

        // Degree-2 wire chain → collapse via JEKV (vert_collapse_edge).
        if ne == 2 && is_wire {
            return self.vert_collapse_edge(edges[0], v).is_some();
        }

        // Degree-2 boundary vertex on a single face → also JEKV.
        // Count unique faces incident to v.
        if ne == 2 {
            let mut faces: smallvec::SmallVec<[FaceKey; 4]> = smallvec::SmallVec::new();
            let mut disk = self.disk_edges(v);
            while let Some(e) = disk.next(self) {
                let mut rl = self.radial_loops(e);
                while let Some(l) = rl.next(self) {
                    let f = self.loops.get(l).face;
                    if !faces.contains(&f) {
                        faces.push(f);
                    }
                }
            }
            if faces.len() == 1 {
                return self.vert_collapse_edge(edges[0], v).is_some();
            }
        }

        // The "disk dissolve" algorithm works for any vertex whose disk
        // contains only boundary (radial=1) and interior (radial=2) edges,
        // as long as at least one interior edge is present to fold the
        // surrounding fan down to a degree-2 boundary collapse. This
        // includes:
        //   - the pure manifold-interior case (all radial=2)
        //   - the mixed boundary-fan case (vertex sits on the open edge of
        //     a strip of faces; some edges boundary, some interior).
        let no_non_manifold_no_wire = radial_counts.iter().all(|&c| c == 1 || c == 2);
        let has_interior_edge = radial_counts.iter().any(|&c| c == 2);
        if no_non_manifold_no_wire && has_interior_edge {
            return self.vert_dissolve_manifold_disk(v);
        }

        // Anything else (pure-boundary fan with no shared edges, non-manifold
        // edges, wire/face mix, bowtie): refuse.
        false
    }

    /// Inner: dissolve a vertex whose every incident edge is a 2-manifold
    /// interior edge. Repeatedly joins face pairs around `v` until `v` is
    /// degree-2, then collapses via JEKV. Caller must have verified the
    /// manifold precondition.
    fn vert_dissolve_manifold_disk(&mut self, v: VertexKey) -> bool {
        loop {
            // Snapshot current disk edges (off the heap for small valences).
            let edges: smallvec::SmallVec<[EdgeKey; 8]> =
                collect_iter::<_, 8>(self, self.disk_edges(v));
            if edges.len() <= 2 {
                break;
            }
            // Find any edge at v whose radial has exactly 2 faces and
            // join those two faces along it.
            let mut merged = false;
            for &e in &edges {
                let mut rl = self.radial_loops(e);
                let l0 = match rl.next(self) {
                    Some(l) => l,
                    None => continue,
                };
                let l1 = match rl.next(self) {
                    Some(l) => l,
                    None => continue,
                };
                if rl.next(self).is_some() {
                    continue;
                }
                let f0 = self.loops.get(l0).face;
                let f1 = self.loops.get(l1).face;
                if f0 != f1 {
                    if self.euler_join_face_kill_edge(f0, f1, e).is_some() {
                        merged = true;
                        break;
                    }
                }
            }
            if !merged {
                // Shouldn't happen given the precondition, but guard against
                // infinite loops in case of pathological inputs.
                return false;
            }
        }

        // v is now degree-2 (one face on either side of two co-linear edges,
        // or a single face that touches v twice). Collapse it through the
        // reaping-enabled primitive so a triangle corner is reaped to a 2-gon
        // and any coincident doubled face the collapse produces is welded away
        // (DISSOLVE_VERTS.md §6.6) — this is what closes a degenerating fan onto
        // a single clean face rather than a doubled-face remnant.
        let final_edges: smallvec::SmallVec<[EdgeKey; 2]> =
            collect_iter::<_, 2>(self, self.disk_edges(v));
        if final_edges.len() != 2 {
            return false;
        }
        let surviving_edge = self
            .vert_collapse_edge(final_edges[0], v)
            .or_else(|| self.vert_collapse_edge(final_edges[1], v));
        let Some(surviving_edge) = surviving_edge else {
            return false;
        };

        // If the surviving edge is still a 2-manifold interior edge, the two
        // faces it sits between are the residue of the original disk and
        // should be merged.
        if self.edges.try_get(surviving_edge).is_none() {
            return true;
        }
        let mut rl = self.radial_loops(surviving_edge);
        let l0 = rl.next(self);
        let l1 = rl.next(self);
        let l2 = rl.next(self);
        if let (Some(l0), Some(l1), None) = (l0, l1, l2) {
            let f0 = self.loops.get(l0).face;
            let f1 = self.loops.get(l1).face;
            if f0 != f1 {
                let _ = self.euler_join_face_kill_edge(f0, f1, surviving_edge);
            }
        }
        true
    }
}
