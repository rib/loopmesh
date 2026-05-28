use crate::loopmesh::*;

impl LoopMesh {
    /// Joins N faces into one by repeatedly killing edges shared between
    /// pairs of remaining faces. Returns the surviving face, or `None` if
    /// the input faces are not transitively connected by shared edges.
    ///
    /// This is the iterative-JFKE form. For multi-cycle regions where some
    /// interior shared edges become same-face filaments after the pairwise
    /// joins (e.g. a 2x2 block of quads sharing a centre vertex), the
    /// trailing cleanup below removes those filament edges and any vertex
    /// whose entire incident-edge set was interior to the merged region,
    /// matching the many-input semantics described in
    /// `radial-mesh-guides/operators/DISSOLVE_FACES.md` §4.1 steps 2 + 4.
    pub fn faces_join(&mut self, faces: &[FaceKey]) -> Option<FaceKey> {
        if faces.is_empty() {
            return None;
        }
        if faces.len() == 1 {
            return Some(faces[0]);
        }

        // Snapshot every vert on any input face's perimeter. After the merge
        // surgery any of these whose entire incident-edge set was interior
        // to the region will be a degree-zero orphan — the §4.1-step-4 kill
        // candidates.
        let mut interior_vert_candidates: smallvec::SmallVec<[VertexKey; 16]> =
            smallvec::SmallVec::new();
        for &face in faces {
            for vertex in self.face_vert_keys(face) {
                if !interior_vert_candidates.contains(&vertex) {
                    interior_vert_candidates.push(vertex);
                }
            }
        }

        let mut remaining: Vec<FaceKey> = faces.to_vec();

        let survivor = 'outer: loop {
            if remaining.len() == 1 {
                break 'outer remaining[0];
            }
            // Find any edge shared by two remaining faces (radial cycle of
            // length 2 with both faces in `remaining`).
            for (i, &fa) in remaining.iter().enumerate() {
                let edges: smallvec::SmallVec<[EdgeKey; 8]> =
                    collect_iter::<_, 8>(self, FaceLoops::new(self, fa))
                        .into_iter()
                        .map(|l| self.loops.get(l).edge)
                        .filter(|e| !e.is_null())
                        .collect();
                for &e in &edges {
                    // Read the 2-radial face pair via the lending iterator.
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
                    } // not 2-manifold
                    let f0 = self.loops.get(l0).face;
                    let f1 = self.loops.get(l1).face;
                    let other = if f0 == fa {
                        f1
                    } else if f1 == fa {
                        f0
                    } else {
                        continue;
                    };
                    if other == fa {
                        continue;
                    }
                    if let Some(j) = remaining.iter().position(|&f| f == other) {
                        // Join fa and other along e. fa survives.
                        self.euler_join_face_kill_edge(fa, other, e)?;
                        remaining.swap_remove(j);
                        let _ = i;
                        continue 'outer;
                    }
                }
            }
            // No more shared edges available — the input wasn't fully connected.
            return None;
        };

        // §4.1 step 2 (residual): pairwise JFKE leaves an interior shared
        // edge behind whenever two input faces share more than one edge —
        // the second-and-later shared edges become "filaments" with both
        // radial loops on the surviving merged face. Sweep them out.
        self.faces_join_dissolve_same_face_filaments(survivor);

        // §4.1 step 4: kill verts whose entire incident-edge set was
        // interior. After steps above, such a vert has empty disk. Any
        // candidate that retains at least one edge (boundary or external)
        // keeps a non-empty disk and must NOT be killed.
        for vertex in interior_vert_candidates {
            let Some(vertex_data) = self.vertices.try_get(vertex) else {
                continue;
            };
            if vertex_data.disk_loop_start.is_none() {
                self.vertex_kill(vertex);
            }
        }

        // §4.2 double-face detection: a pre-existing face whose perimeter
        // coincides with the merged face's perimeter wins; the merge
        // result is killed and the pre-existing face is returned as the
        // survivor (its customdata is preserved). The merged face's
        // perimeter is, by construction, a simple cycle of boundary
        // edges shared with at most one other face on each edge's
        // radial cycle. Any face that shares *every* edge of the merged
        // perimeter must appear on the radial of every such edge — so
        // checking the radial cycle of a single perimeter edge is
        // sufficient to enumerate the candidates.
        if let Some(shadow) = self.faces_join_find_coincident_face(survivor) {
            self.face_kill(survivor);
            return Some(shadow);
        }

        Some(survivor)
    }

    /// Search the radial cycle of `merged`'s first perimeter edge for a
    /// pre-existing face whose corner cycle (vertex sequence) equals
    /// `merged`'s corner cycle modulo rotation and reflection. Returns
    /// the matching face if any, else `None`.
    ///
    /// A face that shares the merged face's entire perimeter must sit on
    /// every one of those perimeter edges' radial cycles, so probing any
    /// single such cycle enumerates all candidates. The full perimeter
    /// match is then verified by comparing the candidate's vertex cycle
    /// against `merged`'s as an ordered sequence (allowing rotation /
    /// reversal).
    fn faces_join_find_coincident_face(&self, merged: FaceKey) -> Option<FaceKey> {
        let merged_perimeter = self.face_vert_keys(merged);
        if merged_perimeter.is_empty() {
            return None;
        }

        let first_loop = self.faces.get(merged).first_edge_loop;
        let probe_edge = self.loops.get(first_loop).edge;
        if probe_edge.is_null() {
            return None;
        }

        let mut radial = self.radial_loops(probe_edge);
        while let Some(loop_key) = radial.next(self) {
            let candidate = self.loops.get(loop_key).face;
            if candidate == merged {
                continue;
            }
            let candidate_perimeter = self.face_vert_keys(candidate);
            if vertex_cycles_coincide(&merged_perimeter, &candidate_perimeter) {
                return Some(candidate);
            }
        }
        None
    }

    /// Walk `face`'s loop cycle and kill every edge whose radial cycle has
    /// length 2 and *both* loops belong to `face`. Such an edge is a same-
    /// face filament — a residual interior shared edge that pairwise JFKE
    /// could not remove. Killing it splices its two loops out of the face
    /// cycle and frees the edge from its endpoints' disks.
    fn faces_join_dissolve_same_face_filaments(&mut self, face: FaceKey) {
        loop {
            let scan_edges: smallvec::SmallVec<[(LoopKey, EdgeKey); 16]> =
                collect_iter::<_, 16>(self, FaceLoops::new(self, face))
                    .into_iter()
                    .map(|l| (l, self.loops.get(l).edge))
                    .filter(|(_, e)| !e.is_null())
                    .collect();

            let mut filament: Option<(LoopKey, LoopKey, EdgeKey)> = None;
            for (loopi, edge) in scan_edges {
                let mut radial = self.radial_loops(edge);
                let r0 = match radial.next(self) {
                    Some(l) => l,
                    None => continue,
                };
                let r1 = match radial.next(self) {
                    Some(l) => l,
                    None => continue,
                };
                if radial.next(self).is_some() {
                    continue;
                }
                let f0 = self.loops.get(r0).face;
                let f1 = self.loops.get(r1).face;
                if f0 == face && f1 == face {
                    // Both radial loops belong to `face` — this is a same-
                    // face filament. `loopi` is one of (r0, r1); the other
                    // is its radial partner on `face`.
                    let other = if r0 == loopi { r1 } else { r0 };
                    filament = Some((loopi, other, edge));
                    break;
                }
            }

            let Some((l_a, l_b, edge)) = filament else {
                return;
            };
            self.faces_join_kill_filament_edge(face, l_a, l_b, edge);
        }
    }

    /// Splice the pair `(l_a, l_b)` out of `face`'s loop cycle, then kill
    /// the loops and the edge. Both loops must be on `face`, share `edge`,
    /// and be each other's radial partners (the only two loops on `edge`).
    fn faces_join_kill_filament_edge(
        &mut self,
        face: FaceKey,
        l_a: LoopKey,
        l_b: LoopKey,
        edge: EdgeKey,
    ) {
        let l_a_prev = self.loops.get(l_a).face_loop_links.prev;
        let l_a_next = self.loops.get(l_a).face_loop_links.next;
        let l_b_prev = self.loops.get(l_b).face_loop_links.prev;
        let l_b_next = self.loops.get(l_b).face_loop_links.next;

        // The two filament loops may sit adjacent on the face cycle (the
        // "spur off the perimeter" case — `l_a` immediately followed by
        // `l_b` or vice versa) or non-adjacent (the cycle pinches across
        // the filament and the splice yields two sub-cycles). The two
        // cases need different splice connectivity.
        if l_a_next == l_b {
            // ... l_a_prev -> l_a -> l_b -> l_b_next ...
            self.loops.get_mut(l_a_prev).face_loop_links.next = l_b_next;
            self.loops.get_mut(l_b_next).face_loop_links.prev = l_a_prev;
        } else if l_b_next == l_a {
            self.loops.get_mut(l_b_prev).face_loop_links.next = l_a_next;
            self.loops.get_mut(l_a_next).face_loop_links.prev = l_b_prev;
        } else {
            self.loops.get_mut(l_a_prev).face_loop_links.next = l_a_next;
            self.loops.get_mut(l_a_next).face_loop_links.prev = l_a_prev;
            self.loops.get_mut(l_b_prev).face_loop_links.next = l_b_next;
            self.loops.get_mut(l_b_next).face_loop_links.prev = l_b_prev;
        }
        self.faces.get_mut(face).len = self.faces.get(face).len.saturating_sub(2);

        // Re-anchor `first_edge_loop` if it pointed at a doomed loop. A
        // surviving neighbour of the spliced pair is the safest anchor;
        // `l_a_prev` is always live in the spur case and the non-adjacent
        // case alike.
        let anchor = self.faces.get(face).first_edge_loop;
        if anchor == l_a || anchor == l_b {
            let new_anchor = if l_a_prev != l_a && l_a_prev != l_b {
                l_a_prev
            } else {
                l_b_prev
            };
            self.faces.get_mut(face).first_edge_loop = new_anchor;
        }

        self.face_loop_disconnect_edge(l_a);
        self.face_loop_disconnect_edge(l_b);

        self.remove_loop_slab_entry(l_a);
        self.remove_loop_slab_entry(l_b);

        let edge_verts = self.edges.get(edge).vertices;
        self.vertex_disk_remove_edge(edge_verts[0], edge);
        self.vertex_disk_remove_edge(edge_verts[1], edge);
        self.remove_edge_slab_entry(edge);
    }
}

/// True iff `cycle_a` and `cycle_b` are the same cyclic vertex sequence
/// modulo rotation and reflection. Used by §4.2 double-face detection
/// to decide whether the merged face duplicates a pre-existing face's
/// perimeter.
fn vertex_cycles_coincide(cycle_a: &[VertexKey], cycle_b: &[VertexKey]) -> bool {
    let n = cycle_a.len();
    if n == 0 || n != cycle_b.len() {
        return false;
    }
    let start_indices: smallvec::SmallVec<[usize; 4]> = cycle_b
        .iter()
        .enumerate()
        .filter(|(_, v)| **v == cycle_a[0])
        .map(|(i, _)| i)
        .collect();
    for start in start_indices {
        if (0..n).all(|i| cycle_b[(start + i) % n] == cycle_a[i]) {
            return true;
        }
        if (0..n).all(|i| cycle_b[(start + n - i) % n] == cycle_a[i]) {
            return true;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use glam::Vec3;

    use crate::loopmesh::*;

    /// Build a 3x3 face grid of quads with positions on the unit lattice.
    /// `faces[r * 3 + c]` is the quad whose lower-left vertex is
    /// `verts[r * 4 + c]`.
    fn build_grid_3x3(mesh: &mut LoopMesh) -> (Vec<VertexKey>, Vec<FaceKey>) {
        let mut verts: Vec<VertexKey> = Vec::with_capacity(16);
        for r in 0..4 {
            for c in 0..4 {
                verts.push(mesh.create_vertex(Vec3::new(c as f32, r as f32, 0.0)));
            }
        }
        let mut faces: Vec<FaceKey> = Vec::new();
        for r in 0..3 {
            for c in 0..3 {
                let v_bl = verts[r * 4 + c];
                let v_br = verts[r * 4 + c + 1];
                let v_tr = verts[(r + 1) * 4 + c + 1];
                let v_tl = verts[(r + 1) * 4 + c];
                let e_b = mesh.create_edge([v_bl, v_br], EdgeCreateFlags::NO_DUPLICATE);
                let e_r = mesh.create_edge([v_br, v_tr], EdgeCreateFlags::NO_DUPLICATE);
                let e_t = mesh.create_edge([v_tr, v_tl], EdgeCreateFlags::NO_DUPLICATE);
                let e_l = mesh.create_edge([v_tl, v_bl], EdgeCreateFlags::NO_DUPLICATE);
                faces.push(mesh.face_create(
                    &[v_bl, v_br, v_tr, v_tl],
                    &[e_b, e_r, e_t, e_l],
                    FaceCreateFlags::NONE,
                ));
            }
        }
        (verts, faces)
    }

    /// §4.1 step 4: joining the four bottom-left quads of a 3x3 grid into
    /// one face must kill the centre vertex (`verts[5]`) whose four incident
    /// edges were all interior to the merged region, and remove all four
    /// interior edges. The merged face's perimeter is the 8 outer verts of
    /// the 2x2 region.
    #[test]
    fn faces_join_kills_interior_only_vertex() {
        let mut mesh = LoopMesh::new();
        let (verts, faces) = build_grid_3x3(&mut mesh);

        let totvert_before = mesh.totvert();
        let totedge_before = mesh.totedge();
        let totface_before = mesh.totface();
        let centre = verts[5];

        let merged = mesh
            .faces_join(&[faces[0], faces[1], faces[3], faces[4]])
            .expect("four-quad region joins successfully");

        assert!(
            mesh.vertices.try_get(centre).is_none(),
            "interior-only centre vert must be killed per §4.1 step 4"
        );
        assert_eq!(mesh.totvert(), totvert_before - 1, "1 vert killed");
        assert_eq!(mesh.totedge(), totedge_before - 4, "4 interior edges killed");
        assert_eq!(mesh.totface(), totface_before - 3, "4 input faces → 1 face");

        let perimeter = mesh.face_vert_keys(merged);
        assert_eq!(perimeter.len(), 8, "8-corner merged face");
        assert!(!perimeter.contains(&centre));

        mesh.validate_topology();
    }

    /// Negative side of the §4.1 step 4 guard: a region whose interior verts
    /// still have at least one *non-interior* incident edge must NOT have
    /// any verts killed. The two-quad-on-a-grid case is the canonical
    /// fixture — both endpoints of the single shared interior edge retain
    /// edges that bound the rest of the grid.
    #[test]
    fn faces_join_preserves_verts_with_surviving_edges() {
        let mut mesh = LoopMesh::new();
        let (_verts, faces) = build_grid_3x3(&mut mesh);

        let totvert_before = mesh.totvert();

        mesh.faces_join(&[faces[0], faces[1]])
            .expect("two-quad region joins successfully");

        assert_eq!(
            mesh.totvert(),
            totvert_before,
            "no verts may be killed when every region vert retains a surviving edge"
        );

        mesh.validate_topology();
    }

    /// Construct the shadow-face fixture for §4.2: two adjacent quads
    /// (`face 0` left, `face 1` right) sharing interior edge `{v1, v2}`,
    /// overlaid by a third hexagonal `shadow` face (`face 2`) whose
    /// perimeter coincides with the outer boundary of the two-quad
    /// region (`[v0, v1, v4, v5, v2, v3]`). Returns
    /// `(verts, [left_quad, right_quad, shadow])`.
    fn build_shadow_over_two_quads(
        mesh: &mut LoopMesh,
    ) -> (Vec<VertexKey>, [FaceKey; 3]) {
        let v: Vec<VertexKey> = [
            (0.0, 0.0),
            (1.0, 0.0),
            (1.0, 1.0),
            (0.0, 1.0),
            (2.0, 0.0),
            (2.0, 1.0),
        ]
        .into_iter()
        .map(|(x, y)| mesh.create_vertex(Vec3::new(x, y, 0.0)))
        .collect();

        let e_01 = mesh.create_edge([v[0], v[1]], EdgeCreateFlags::NO_DUPLICATE);
        let e_12 = mesh.create_edge([v[1], v[2]], EdgeCreateFlags::NO_DUPLICATE);
        let e_23 = mesh.create_edge([v[2], v[3]], EdgeCreateFlags::NO_DUPLICATE);
        let e_30 = mesh.create_edge([v[3], v[0]], EdgeCreateFlags::NO_DUPLICATE);
        let e_14 = mesh.create_edge([v[1], v[4]], EdgeCreateFlags::NO_DUPLICATE);
        let e_45 = mesh.create_edge([v[4], v[5]], EdgeCreateFlags::NO_DUPLICATE);
        let e_52 = mesh.create_edge([v[5], v[2]], EdgeCreateFlags::NO_DUPLICATE);

        let f_left = mesh.face_create(
            &[v[0], v[1], v[2], v[3]],
            &[e_01, e_12, e_23, e_30],
            FaceCreateFlags::NONE,
        );
        let f_right = mesh.face_create(
            &[v[1], v[4], v[5], v[2]],
            &[e_14, e_45, e_52, e_12],
            FaceCreateFlags::NONE,
        );
        let f_shadow = mesh.face_create(
            &[v[0], v[1], v[4], v[5], v[2], v[3]],
            &[e_01, e_14, e_45, e_52, e_23, e_30],
            FaceCreateFlags::NONE,
        );

        (v, [f_left, f_right, f_shadow])
    }

    /// §4.2: joining the two quads of `build_shadow_over_two_quads` would
    /// normally produce a hexagonal merged face whose perimeter coincides
    /// with the shadow's. Double-face detection must kick in: the shadow
    /// survives, the merge result is killed, and `faces_join` returns the
    /// shadow as the survivor. Final mesh: 6 V, 6 E, 1 F.
    #[test]
    fn faces_join_double_face_detection_shadow_wins() {
        let mut mesh = LoopMesh::new();
        let (verts, [f_left, f_right, f_shadow]) = build_shadow_over_two_quads(&mut mesh);

        assert_eq!(mesh.totvert(), 6);
        assert_eq!(mesh.totedge(), 7, "6 outer-rim + 1 interior v1-v2");
        assert_eq!(mesh.totface(), 3);

        let survivor = mesh
            .faces_join(&[f_left, f_right])
            .expect("two-quad region joins successfully");

        assert_eq!(
            survivor, f_shadow,
            "§4.2 step 3: the pre-existing shadow face becomes the reported survivor"
        );

        assert_eq!(mesh.totvert(), 6, "every perimeter vert retains a shadow edge");
        assert_eq!(mesh.totedge(), 6, "interior v1-v2 edge gone; outer rim survives");
        assert_eq!(
            mesh.totface(),
            1,
            "two source quads + new merged face killed; only the shadow remains"
        );

        // The surviving face is exactly the shadow, with the original
        // perimeter still anchored on the six outer-rim verts.
        let survivor_perimeter = mesh.face_vert_keys(survivor);
        assert_eq!(survivor_perimeter.len(), 6);
        let expected: std::collections::HashSet<VertexKey> = verts.iter().copied().collect();
        let got: std::collections::HashSet<VertexKey> =
            survivor_perimeter.iter().copied().collect();
        assert_eq!(got, expected);

        // The two source quads must be gone from the face table.
        let live_faces: std::collections::HashSet<FaceKey> = mesh.face_keys().into_iter().collect();
        assert!(!live_faces.contains(&f_left));
        assert!(!live_faces.contains(&f_right));
        assert!(live_faces.contains(&f_shadow));

        mesh.validate_topology();
    }

    /// Negative case for §4.2: joining two adjacent quads on a 3x3 grid
    /// has no pre-existing coincident face. The detection must NOT
    /// trigger; the merge result itself is returned as the survivor.
    #[test]
    fn faces_join_double_face_detection_does_not_fire_without_shadow() {
        let mut mesh = LoopMesh::new();
        let (_verts, faces) = build_grid_3x3(&mut mesh);

        let survivor = mesh
            .faces_join(&[faces[0], faces[1]])
            .expect("two-quad merge");

        // The survivor must be a brand-new face (the JFKE chain keeps
        // face[0]'s slot alive as the survivor; either way it is NOT some
        // *other* pre-existing face).
        let live_faces: std::collections::HashSet<FaceKey> =
            mesh.face_keys().into_iter().collect();
        assert!(live_faces.contains(&survivor));
        assert_eq!(survivor, faces[0], "JFKE preserves the first input's face slot");

        mesh.validate_topology();
    }
}
