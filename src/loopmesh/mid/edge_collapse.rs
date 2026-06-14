use smallvec::SmallVec;

use crate::loopmesh::*;

/// Behaviour switches for the edge-collapse primitive
/// ([`LoopMesh::vert_collapse_edge_with`]).
///
/// The collapse always splices the two edges at the degree-2 vertex into one
/// and removes the vertex; these flags govern what happens to faces the
/// collapse pushes below three sides. See
/// `radial-mesh-guides/operators/DISSOLVE_VERTS.md` §6.6 for the contract.
#[derive(Clone, Copy)]
pub struct EdgeCollapseFlags {
    /// Free the collapsed vertex's slab slot. When `false` the vertex is left
    /// edge-less but alive (the caller reaps it later).
    pub delete: bool,
    /// Reap any face the side-count decrement drops below three sides (a
    /// triangle corner → 2-gon). When `false` the degenerate face is left in
    /// place, which violates the face-validity invariant.
    pub kill_degenerate_faces: bool,
    /// After reaping a degenerate face, splice the two now-parallel edges that
    /// bounded it into a single edge. Without this the mesh keeps a redundant
    /// duplicate edge between the two surviving neighbours.
    pub kill_duplicate_faces: bool,
}

impl EdgeCollapseFlags {
    /// The dissolve family's defaults: delete the vertex and fully reap any
    /// degenerate face (kill the 2-gon and splice away the duplicate edge),
    /// matching `DISSOLVE_VERTS.md` §9's phase-4 collapse call.
    pub const DISSOLVE: Self = Self {
        delete: true,
        kill_degenerate_faces: true,
        kill_duplicate_faces: true,
    };
}

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

    /// Collapses `v_kill` along `e_kill` onto the edge's other endpoint,
    /// reaping any degenerate face the collapse produces. Same outcome as
    /// `edge_collapse`, but assumes `v_kill` is degree-2 and routes through
    /// JEKV-style edge splicing.
    ///
    /// This is the cross-operator collapse primitive every dissolve mode
    /// reuses (`DISSOLVE_VERTS.md` §6.6, §12). It runs with the dissolve
    /// family's [`EdgeCollapseFlags::DISSOLVE`] defaults — the collapsed
    /// vertex is deleted and a triangle-corner collapse reaps the resulting
    /// 2-gon and splices away the duplicate edge.
    pub fn vert_collapse_edge(&mut self, e_kill: EdgeKey, v_kill: VertexKey) -> Option<EdgeKey> {
        self.vert_collapse_edge_with(e_kill, v_kill, EdgeCollapseFlags::DISSOLVE)
    }

    /// Edge-collapse primitive with explicit behaviour switches. See
    /// [`EdgeCollapseFlags`] and `DISSOLVE_VERTS.md` §6.6.
    ///
    /// Given degree-2 `v_kill` whose two incident edges are `e_kill`
    /// (`v_kill – v_target`) and `e_old` (`v_kill – v_old`), the surviving
    /// edge spans `v_target – v_old` directly. Each face that had `v_kill` as
    /// a corner loses one side; a face dropped below three sides is reaped per
    /// the flags. Returns the surviving edge, or `None` if a precondition
    /// fails.
    pub fn vert_collapse_edge_with(
        &mut self,
        e_kill: EdgeKey,
        v_kill: VertexKey,
        flags: EdgeCollapseFlags,
    ) -> Option<EdgeKey> {
        if self.edges.get(e_kill).contains(v_kill).is_none() {
            return None;
        }
        if self.vertex_disk_count_edges_capped(v_kill, 3) != 2 {
            return None;
        }

        let e_old = self
            .vertex_disk_iter(v_kill)
            .find(|&e| e != e_kill)
            .expect("disk count is 2 but no other edge found");

        let v_target = self.edges.get(e_kill).other_vertex(v_kill).unwrap();
        let v_old = self.edges.get(e_old).other_vertex(v_kill).unwrap();

        // A self-spanning result (v_target == v_old) cannot be re-pointed into
        // a valid edge — refuse rather than fabricate a self-loop.
        if v_target == v_old {
            return None;
        }

        // The faces on e_kill's radial are the faces that have v_kill as a
        // corner. Any of these whose side count is exactly 3 will bottom out
        // at a 2-gon once the corner is removed; record them so they can be
        // reaped after the splice. (No allocation for the typical boundary /
        // manifold fan radial of ≤ 2 loops.)
        let degenerate_faces: SmallVec<[FaceKey; 4]> = {
            let mut out: SmallVec<[FaceKey; 4]> = SmallVec::new();
            let mut radial = self.radial_loops(e_kill);
            while let Some(l_kill) = radial.next(self) {
                let face = self.loops.get(l_kill).face;
                if self.faces.get(face).len <= 3 && !out.contains(&face) {
                    out.push(face);
                }
            }
            out
        };

        // Migrate e_old's endpoint from v_kill to v_target. After this e_old
        // spans v_target – v_old; if a triangle corner is being collapsed it
        // is now parallel to that triangle's third edge.
        self.vertex_disk_remove_edge(v_kill, e_old);
        let slot_old = if self.edges.get(e_old).vertices[0] == v_kill {
            0
        } else {
            1
        };
        self.edges.get_mut(e_old).vertices[slot_old] = v_target;
        self.vertex_disk_append_edge(v_target, e_old);

        // Remove e_kill from v_target's disk (v_kill's disk is wiped along with the vertex).
        self.vertex_disk_remove_edge(v_target, e_kill);

        // Splice every corner loop at v_kill out of its face cycle, decrement
        // that face's side count, and unhook the loop from e_kill's radial.
        let radial_loops: SmallVec<[LoopKey; 4]> = if self.edges.get(e_kill).radial_link.is_some() {
            collect_iter::<_, 4>(self, self.radial_loops(e_kill))
        } else {
            SmallVec::new()
        };

        for &l_kill in &radial_loops {
            let l_kill_face = self.loops.get(l_kill).face;
            let l_kill_prev = self.loops.get(l_kill).face_loop_links.prev;
            let l_kill_next = self.loops.get(l_kill).face_loop_links.next;

            // If l_kill_next pointed at v_kill (Case A: l_kill.vertex was the
            // v_target side, and the loop went v_target -> v_kill -> v_old),
            // redirect to v_target. (Case B is handled implicitly via the
            // e_old endpoint mutation above.)
            if self.loops.get(l_kill_next).vertex == v_kill {
                self.loops.get_mut(l_kill_next).vertex = v_target;
            }

            self.loops.get_mut(l_kill_prev).face_loop_links.next = l_kill_next;
            self.loops.get_mut(l_kill_next).face_loop_links.prev = l_kill_prev;

            if self.faces.get(l_kill_face).first_edge_loop == l_kill {
                self.faces.get_mut(l_kill_face).first_edge_loop = l_kill_next;
            }
            self.faces.get_mut(l_kill_face).len -= 1;

            self.face_loop_disconnect_edge(l_kill);
            self.remove_loop_slab_entry(l_kill);
        }

        self.remove_edge_slab_entry(e_kill);
        if flags.delete {
            self.remove_vertex_slab_entry(v_kill);
        }

        // Reap any face the corner removal pushed below three sides. After the
        // splice above, such a face's two remaining sides are parallel edges
        // spanning v_target – v_old; killing the face leaves them as a
        // duplicate pair to be spliced into one.
        if flags.kill_degenerate_faces {
            for &face in &degenerate_faces {
                if self.faces.try_get(face).is_none() {
                    continue;
                }
                if self.faces.get(face).len >= 3 {
                    continue;
                }
                self.reap_degenerate_face(face, e_old, flags.kill_duplicate_faces);
            }
        }

        // Weld any coincident doubled face the collapse produced. When the
        // collapsed corner sat on a face that does *not* bottom out at a 2-gon
        // (it had 4+ sides, so it survives at 3+) but whose re-pointed side now
        // duplicates a pre-existing edge between the same two verts, the face
        // can land coincident with the face already on that edge — a doubled
        // face. This is the join-then-split / weld endgame of
        // `BM_vert_collapse_faces`: `kill_duplicate_faces` splices the parallel
        // edges and reaps the redundant coincident face (`DISSOLVE_VERTS.md`
        // §6.6, §8 — the single-vert-helper path that closes a degenerating fan
        // onto one clean face rather than keeping the duplicate).
        let mut surviving = e_old;
        if flags.kill_duplicate_faces && self.edges.try_get(e_old).is_some() {
            if let Some(welded) = self.weld_duplicate_edge_face(e_old, v_target, v_old) {
                surviving = welded;
            }
        }

        Some(surviving)
    }

    /// After a corner collapse, fold a re-pointed edge that now duplicates a
    /// pre-existing edge between the same two verts into one, welding any face
    /// that thereby becomes coincident with its neighbour. Reaps the redundant
    /// doubled face so a degenerating fan closes onto a single clean face
    /// (`DISSOLVE_VERTS.md` §6.6 `kill_duplicate_faces`). Returns the surviving
    /// edge after the splice (the kept sibling), or `None` if no duplicate was
    /// found and `repointed` is left as-is.
    fn weld_duplicate_edge_face(
        &mut self,
        repointed: EdgeKey,
        va: VertexKey,
        vb: VertexKey,
    ) -> Option<EdgeKey> {
        // Find a sibling edge spanning the same vertex pair as `repointed`.
        let sibling = {
            let mut found: Option<EdgeKey> = None;
            let mut disk = self.disk_edges(va);
            while let Some(edge) = disk.next(self) {
                if edge == repointed {
                    continue;
                }
                if self.edges.get(edge).other_vertex(va) == Some(vb) {
                    found = Some(edge);
                    break;
                }
            }
            found
        };
        let sibling = sibling?;

        // Splice the re-pointed duplicate onto the pre-existing sibling, keeping
        // the sibling (the older edge) as the survivor. Both edges' loops now
        // share one radial cycle.
        if !self.edge_splice(sibling, repointed) {
            return None;
        }

        // Any two faces that are now coincident — same loop vertex set, same
        // edges — are a doubled face; reap the extra one. Collect the faces on
        // the merged edge's radial cycle and compare their vertex sets.
        let radial_faces: SmallVec<[FaceKey; 4]> = {
            let mut out: SmallVec<[FaceKey; 4]> = SmallVec::new();
            let mut radial = self.radial_loops(sibling);
            while let Some(l) = radial.next(self) {
                let face = self.loops.get(l).face;
                if !out.contains(&face) {
                    out.push(face);
                }
            }
            out
        };
        for outer in 0..radial_faces.len() {
            let face_a = radial_faces[outer];
            if self.faces.try_get(face_a).is_none() {
                continue;
            }
            let set_a: SmallVec<[VertexKey; 6]> = self.face_vert_keys(face_a).into_iter().collect();
            for &face_b in radial_faces.iter().skip(outer + 1) {
                if face_b == face_a || self.faces.try_get(face_b).is_none() {
                    continue;
                }
                let set_b: SmallVec<[VertexKey; 6]> =
                    self.face_vert_keys(face_b).into_iter().collect();
                if set_a.len() == set_b.len() && set_a.iter().all(|v| set_b.contains(v)) {
                    self.face_kill(face_b);
                }
            }
        }
        Some(sibling)
    }

    /// Kill a sub-triangle face produced by an edge collapse and, when
    /// `splice_duplicate`, fold its two now-parallel boundary edges into one.
    /// `repointed_edge` is the collapse's re-pointed edge — the side that is
    /// killed by the splice so the triangle's pre-existing third edge survives
    /// (`DISSOLVE_VERTS.md` §11).
    fn reap_degenerate_face(
        &mut self,
        face: FaceKey,
        repointed_edge: EdgeKey,
        splice_duplicate: bool,
    ) {
        // Snapshot the face's remaining edges before killing it. A reaped
        // 2-gon has two loops on two distinct edges; the third (`repointed`)
        // is the duplicate side, the other is the survivor.
        let boundary_edges: SmallVec<[EdgeKey; 2]> = {
            let mut out: SmallVec<[EdgeKey; 2]> = SmallVec::new();
            let mut face_loops = self.face_loops(face);
            while let Some(l) = face_loops.next(self) {
                let edge = self.loops.get(l).edge;
                if !edge.is_null() && !out.contains(&edge) {
                    out.push(edge);
                }
            }
            out
        };

        self.face_kill(face);

        if !splice_duplicate {
            return;
        }

        // The survivor is the boundary edge that is not the re-pointed one.
        let survivor = boundary_edges
            .iter()
            .copied()
            .find(|&e| e != repointed_edge);
        if let Some(survivor) = survivor {
            if self.edges.try_get(repointed_edge).is_some()
                && self.edges.try_get(survivor).is_some()
            {
                // Keep the pre-existing third edge; the re-pointed duplicate
                // is killed by the splice.
                self.edge_splice(survivor, repointed_edge);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use glam::Vec3;

    use crate::loopmesh::*;

    /// Build an isolated triangle and return its three corner verts. `v[1]`
    /// is the corner we collapse in the degenerate-reaping tests.
    fn build_triangle(mesh: &mut LoopMesh) -> [VertexKey; 3] {
        let v: Vec<VertexKey> = [(0.0, 0.0), (1.0, 0.0), (0.5, 1.0)]
            .into_iter()
            .map(|(x, y)| mesh.create_vertex(Vec3::new(x, y, 0.0)))
            .collect();
        let e0 = mesh.create_edge([v[0], v[1]], EdgeCreateFlags::NO_DUPLICATE);
        let e1 = mesh.create_edge([v[1], v[2]], EdgeCreateFlags::NO_DUPLICATE);
        let e2 = mesh.create_edge([v[2], v[0]], EdgeCreateFlags::NO_DUPLICATE);
        mesh.face_create(&[v[0], v[1], v[2]], &[e0, e1, e2], FaceCreateFlags::NONE);
        [v[0], v[1], v[2]]
    }

    /// Collapsing the corner of a lone triangle reaps the resulting 2-gon and
    /// splices away the duplicate edge: V=3,E=3,F=1 → V=2,E=1,F=0
    /// (`DISSOLVE_VERTS.md` §6.6 triangle-corner row).
    #[test]
    fn vert_collapse_edge_triangle_corner_reaps_face() {
        let mut mesh = LoopMesh::new();
        let [v0, v1, v2] = build_triangle(&mut mesh);

        assert_eq!(mesh.vertex_count(), 3);
        assert_eq!(mesh.edge_count(), 3);
        assert_eq!(mesh.face_count(), 1);

        let e_kill = mesh.vert_edge_keys(v1)[0];
        let surviving = mesh.vert_collapse_edge(e_kill, v1);
        assert!(
            surviving.is_some(),
            "collapse succeeds on a triangle corner"
        );

        assert_eq!(mesh.vertex_count(), 2, "-1 V (collapsed corner)");
        assert_eq!(mesh.edge_count(), 1, "-2 E (spoke + spliced duplicate)");
        assert_eq!(mesh.face_count(), 0, "-1 F (the 2-gon is reaped)");

        let live: std::collections::HashSet<VertexKey> = mesh.vert_keys().into_iter().collect();
        assert!(!live.contains(&v1), "the collapsed corner is gone");
        assert!(
            live.contains(&v0) && live.contains(&v2),
            "neighbours survive"
        );

        // The single surviving edge spans the two surviving neighbours.
        let edges = mesh.edge_keys();
        assert_eq!(edges.len(), 1);
        let endpoints = mesh.edge_vert_keys(edges[0]);
        let endpoint_set: std::collections::HashSet<VertexKey> = endpoints.into_iter().collect();
        assert_eq!(
            endpoint_set,
            [v0, v2].into_iter().collect(),
            "the surviving edge connects the two neighbours"
        );
        mesh.validate_topology();
    }

    /// Collapsing the corner of a 4+-sided face leaves the face alive at one
    /// fewer side — no reaping: V=4,E=4,F=1 → V=3,E=3,F=1
    /// (`DISSOLVE_VERTS.md` §6.6 4+-sided row).
    #[test]
    fn vert_collapse_edge_quad_corner_keeps_face() {
        let mut mesh = LoopMesh::new();
        let v: Vec<VertexKey> = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
            .into_iter()
            .map(|(x, y)| mesh.create_vertex(Vec3::new(x, y, 0.0)))
            .collect();
        let e0 = mesh.create_edge([v[0], v[1]], EdgeCreateFlags::NO_DUPLICATE);
        let e1 = mesh.create_edge([v[1], v[2]], EdgeCreateFlags::NO_DUPLICATE);
        let e2 = mesh.create_edge([v[2], v[3]], EdgeCreateFlags::NO_DUPLICATE);
        let e3 = mesh.create_edge([v[3], v[0]], EdgeCreateFlags::NO_DUPLICATE);
        let face = mesh.face_create(
            &[v[0], v[1], v[2], v[3]],
            &[e0, e1, e2, e3],
            FaceCreateFlags::NONE,
        );

        assert_eq!(mesh.vertex_count(), 4);
        assert_eq!(mesh.edge_count(), 4);
        assert_eq!(mesh.face_count(), 1);

        let e_kill = mesh.vert_edge_keys(v[1])[0];
        let surviving = mesh.vert_collapse_edge(e_kill, v[1]);
        assert!(surviving.is_some(), "collapse succeeds on a quad corner");

        assert_eq!(mesh.vertex_count(), 3, "-1 V (collapsed corner)");
        assert_eq!(mesh.edge_count(), 3, "-1 E (spoke only; no duplicate)");
        assert_eq!(mesh.face_count(), 1, "the face survives as a triangle");
        assert_eq!(mesh.face_len(face), 3, "quad → triangle, still valid");

        let live: std::collections::HashSet<VertexKey> = mesh.vert_keys().into_iter().collect();
        assert!(!live.contains(&v[1]), "the collapsed corner is gone");
        mesh.validate_topology();
    }

    /// With `kill_degenerate_faces = false` the collapse decrements the side
    /// count but leaves the (invalid) 2-gon alive: V=2,E=2,F=1. Pins that the
    /// reaping is conditional on the flag.
    #[test]
    fn vert_collapse_edge_triangle_corner_without_reaping_leaves_2gon() {
        let mut mesh = LoopMesh::new();
        let [_v0, v1, _v2] = build_triangle(&mut mesh);

        let e_kill = mesh.vert_edge_keys(v1)[0];
        let flags = EdgeCollapseFlags {
            delete: true,
            kill_degenerate_faces: false,
            kill_duplicate_faces: false,
        };
        let surviving = mesh.vert_collapse_edge_with(e_kill, v1, flags);
        assert!(surviving.is_some());

        assert_eq!(mesh.vertex_count(), 2, "-1 V");
        assert_eq!(mesh.edge_count(), 2, "-1 E (spoke only; duplicate left)");
        assert_eq!(
            mesh.face_count(),
            1,
            "the 2-gon is left alive when reaping is off"
        );
    }

    /// The reaping is a property of the primitive, so the single-vert
    /// `vert_dissolve` helper reaps a triangle corner too: the same
    /// V=2,E=1,F=0 result (`DISSOLVE_VERTS.md` §6.6, §12).
    #[test]
    fn vert_dissolve_triangle_corner_reaps_face() {
        let mut mesh = LoopMesh::new();
        let [v0, v1, v2] = build_triangle(&mut mesh);

        assert!(mesh.vert_dissolve(v1), "single-vert dissolve succeeds");

        assert_eq!(mesh.vertex_count(), 2);
        assert_eq!(mesh.edge_count(), 1);
        assert_eq!(mesh.face_count(), 0);
        let live: std::collections::HashSet<VertexKey> = mesh.vert_keys().into_iter().collect();
        assert!(!live.contains(&v1));
        assert!(live.contains(&v0) && live.contains(&v2));
        mesh.validate_topology();
    }
}
