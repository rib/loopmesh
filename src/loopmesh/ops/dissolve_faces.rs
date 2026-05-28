use smallvec::SmallVec;

use crate::loopmesh::*;

/// Parameters controlling [`LoopMesh::dissolve_faces`].
///
/// Mirrors the BMOP set-input shape described in
/// `radial-mesh-guides/operators/DISSOLVE_FACES.md` §1.
#[derive(Clone, Copy, Default)]
pub struct DissolveFacesOptions {
    /// Post-pass that collapses every vertex which was *not* an edge-pair
    /// before the operator started but became one as a consequence of the
    /// per-region merges. Documented in
    /// `radial-mesh-guides/operators/DISSOLVE_FACES.md` §6.
    ///
    /// Not yet implemented — see the matrix row
    /// `dissolve_faces` / "`use_verts = true` — newly-edge-pair vert collapses".
    pub use_verts: bool,
}

/// Output of [`LoopMesh::dissolve_faces`]. One entry per successful
/// per-region merge — the surviving merged face. Failed regions and
/// singletons contribute nothing.
///
/// Conceptually parallels the `region.out` slot described in
/// `radial-mesh-guides/operators/DISSOLVE_FACES.md` §1.
#[derive(Default)]
pub struct DissolveFacesResult {
    pub region: Vec<FaceKey>,
}

impl LoopMesh {
    /// Merge each manifoldly-connected region of `faces` into a single face.
    ///
    /// Follows the algorithm in
    /// `radial-mesh-guides/operators/DISSOLVE_FACES.md`:
    ///
    /// 1. Tag every input face (§3 walker face-mask).
    /// 2. From each unvisited input face, BFS across edges whose radial
    ///    cycle has exactly two faces *both* of which are tagged (§3 +
    ///    §3.3). Singleton regions are skipped per §3.1.
    /// 3. For each region of size ≥ 2, merge via the many-input
    ///    `faces_join` primitive (§4). The primitive kills the interior
    ///    shared edges and any vertex whose every incident edge was
    ///    interior (§4.1 steps 2 + 4).
    /// 4. Collect the surviving merged face into the result (§1
    ///    `region.out`).
    ///
    /// Currently supports the default-parameter path only — `use_verts`
    /// triggers an unimplemented panic. The double-face detection of §4.2
    /// is not yet implemented; on inputs that would hit it the merged
    /// face is reported as the survivor.
    pub fn dissolve_faces(
        &mut self,
        faces: &[FaceKey],
        options: DissolveFacesOptions,
    ) -> DissolveFacesResult {
        if options.use_verts {
            // §6 post-pass — pre-merge `!is_edge_pair` mark + post-merge
            // edge-pair collapse. Pending implementation in a future cycle;
            // see the matrix row `dissolve_faces` /
            // "`use_verts = true` — newly-edge-pair vert collapses".
            todo!(
                "dissolve_faces: use_verts post-pass not implemented — see \
                 radial-mesh-guides/operators/DISSOLVE_FACES.md §6 and the \
                 matrix row `dissolve_faces` / `use_verts = true`."
            );
        }

        let mut result = DissolveFacesResult::default();
        if faces.is_empty() {
            return result;
        }

        // §3 — face mark = "this face is in the input set". The walker
        // crosses an edge iff its radial cycle has exactly two faces and
        // both carry this mark.
        let in_input = self.push_face_op_flag_layer();
        for &f in faces {
            self.face_op_flag_set(in_input, f, 0);
        }
        // Independent "already visited" bit so that a single walker step
        // never re-seeds a face already pulled into another region's BFS.
        let visited = self.push_face_op_flag_layer();

        // Stable input-iteration order — the §11 customdata identity rule
        // ("first face in walker-emit order wins") depends on it.
        for &seed in faces {
            if self.face_op_flag_test(visited, seed, 0) {
                continue;
            }
            let region = self.dissolve_faces_walk_region(seed, in_input, visited);
            if region.len() < 2 {
                // §3.1 — singletons are skipped silently.
                continue;
            }
            // §4 — the many-input merge primitive does the rest.
            if let Some(merged) = self.faces_join(&region) {
                result.region.push(merged);
            }
            // Failed merges (§4.3) leave the region's input faces intact;
            // we contribute nothing to `region.out` for that region.
        }

        self.pop_face_op_flag_layer(visited);
        self.pop_face_op_flag_layer(in_input);

        result
    }

    /// BFS the manifold region rooted at `seed`. Visits every face
    /// reachable across an edge whose radial cycle is exactly the seed
    /// face plus one neighbour, where the neighbour also carries the
    /// `in_input` mark. Each visited face is stamped with `visited` so
    /// the caller's outer loop does not re-seed it.
    fn dissolve_faces_walk_region(
        &mut self,
        seed: FaceKey,
        in_input: crate::op_flags::FaceOpFlagLayer,
        visited: crate::op_flags::FaceOpFlagLayer,
    ) -> SmallVec<[FaceKey; 8]> {
        let mut region: SmallVec<[FaceKey; 8]> = SmallVec::new();
        let mut frontier: SmallVec<[FaceKey; 8]> = SmallVec::new();
        frontier.push(seed);
        self.face_op_flag_set(visited, seed, 0);

        while let Some(face) = frontier.pop() {
            region.push(face);

            // Snapshot the face's perimeter edges; the walker reads radial
            // cycles but does not mutate the mesh.
            let edges: SmallVec<[EdgeKey; 8]> =
                collect_iter::<_, 8>(self, FaceLoops::new(self, face))
                    .into_iter()
                    .map(|l| self.loops.get(l).edge)
                    .filter(|e| !e.is_null())
                    .collect();

            for edge in edges {
                // §3 + §3.3 — manifold gate: radial must be exactly length 2.
                let mut radial = self.radial_loops(edge);
                let l0 = match radial.next(self) {
                    Some(l) => l,
                    None => continue,
                };
                let l1 = match radial.next(self) {
                    Some(l) => l,
                    None => continue,
                };
                if radial.next(self).is_some() {
                    continue;
                }
                let f0 = self.loops.get(l0).face;
                let f1 = self.loops.get(l1).face;
                let neighbour = if f0 == face {
                    f1
                } else if f1 == face {
                    f0
                } else {
                    continue;
                };
                if neighbour == face {
                    continue;
                }
                if !self.face_op_flag_test(in_input, neighbour, 0) {
                    continue;
                }
                if self.face_op_flag_test(visited, neighbour, 0) {
                    continue;
                }
                self.face_op_flag_set(visited, neighbour, 0);
                frontier.push(neighbour);
            }
        }

        region
    }
}

#[cfg(test)]
mod tests {
    use glam::Vec3;

    use crate::loopmesh::*;

    /// Build a 3x3 face grid of quads with positions on the unit lattice.
    /// Returns `(verts[16], edges, faces[9])` where `faces[r * 3 + c]` is
    /// the quad whose lower-left vertex is `verts[r * 4 + c]`. Vertex
    /// numbering: `r * 4 + c` over rows `r in 0..4` and columns
    /// `c in 0..4`. Quad winding is CCW from above (lower-left → lower-
    /// right → upper-right → upper-left).
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

    #[test]
    fn dissolve_faces_two_adjacent_quads_merge_into_rectangle() {
        // §7 row 2: a region of n=2 with i=1 interior edge and m=0
        // interior-only verts on a grid contributes -1 F, -1 E, 0 V.
        let mut mesh = LoopMesh::new();
        let (_verts, faces) = build_grid_3x3(&mut mesh);

        let v0 = mesh.totvert();
        let e0 = mesh.totedge();
        let f0 = mesh.totface();
        assert_eq!(v0, 16);
        assert_eq!(e0, 24, "3x3 grid: 4*3 horizontal + 4*3 vertical = 24 edges");
        assert_eq!(f0, 9);

        // Dissolve the bottom-left two quads (faces[0] and faces[1]) which
        // share the vertical edge between (1,0)-(1,1).
        let result = mesh.dissolve_faces(&[faces[0], faces[1]], Default::default());

        assert_eq!(result.region.len(), 1, "one region, one successful merge");
        let merged = result.region[0];

        assert_eq!(mesh.totface(), f0 - 1, "-1 face per §7 row 2");
        assert_eq!(mesh.totedge(), e0 - 1, "-1 edge per §7 row 2");
        assert_eq!(mesh.totvert(), v0, "0 verts removed per §7 row 2");

        let perimeter = mesh.face_vert_keys(merged);
        assert_eq!(
            perimeter.len(),
            6,
            "merged face has 6 corners: 4+4 minus the two consumed by the interior edge"
        );
        mesh.validate_topology();
    }

    #[test]
    fn dissolve_faces_empty_input_is_noop() {
        let mut mesh = LoopMesh::new();
        let (_verts, _faces) = build_grid_3x3(&mut mesh);
        let v0 = mesh.totvert();
        let e0 = mesh.totedge();
        let f0 = mesh.totface();

        let result = mesh.dissolve_faces(&[], Default::default());

        assert!(result.region.is_empty());
        assert_eq!(mesh.totvert(), v0);
        assert_eq!(mesh.totedge(), e0);
        assert_eq!(mesh.totface(), f0);
        mesh.validate_topology();
    }

    #[test]
    fn dissolve_faces_single_face_input_is_noop() {
        // §3.1 — a region of size 1 is silently skipped: the walker
        // requires at least one manifold step to open a region.
        let mut mesh = LoopMesh::new();
        let (_verts, faces) = build_grid_3x3(&mut mesh);
        let v0 = mesh.totvert();
        let e0 = mesh.totedge();
        let f0 = mesh.totface();

        let result = mesh.dissolve_faces(&[faces[0]], Default::default());

        assert!(result.region.is_empty(), "singleton region contributes nothing to region.out");
        assert_eq!(mesh.totvert(), v0);
        assert_eq!(mesh.totedge(), e0);
        assert_eq!(mesh.totface(), f0);
        mesh.validate_topology();
    }

    #[test]
    fn dissolve_faces_merged_face_perimeter_keeps_outer_grid_verts() {
        // The merged face's six corners must be exactly the six outer
        // vertices of the two-quad region; the two endpoints of the
        // shared interior edge survive (they still border the rest of
        // the grid).
        let mut mesh = LoopMesh::new();
        let (verts, faces) = build_grid_3x3(&mut mesh);

        // Faces[0] perimeter = v0,v1,v5,v4. Faces[1] perimeter = v1,v2,v6,v5.
        // Shared edge sits between v1 and v5. Merged perimeter = {v0,v1,v2,v6,v5,v4}.
        let expected: std::collections::HashSet<VertexKey> = [
            verts[0], verts[1], verts[2], verts[6], verts[5], verts[4],
        ]
        .into_iter()
        .collect();

        let result = mesh.dissolve_faces(&[faces[0], faces[1]], Default::default());
        let merged = result.region[0];
        let got: std::collections::HashSet<VertexKey> = mesh.face_vert_keys(merged).into_iter().collect();
        assert_eq!(got, expected);
    }
}
