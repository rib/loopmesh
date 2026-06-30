use std::collections::HashSet;

use crate::loopmesh::*;

/// The `faces_fail.out` of [`LoopMesh::face_attribute_fill`]: the destination
/// faces the inward flood could **not** resolve — those for which no source
/// (unmarked) face was ever reachable, directly or transitively
/// ([`FACE_ATTRIBUTE_FILL.md`](../../../../radial-mesh-guides/operators/FACE_ATTRIBUTE_FILL.md)
/// §5). Membership is a property of reachability alone, independent of the
/// `use_normals` / `use_data` flags (§1.1). A failed face is left completely
/// unchanged. An all-success run leaves this empty.
#[derive(Clone, Debug, Default)]
pub struct FaceAttributeFillResult {
    faces_fail: Vec<FaceKey>,
}

impl FaceAttributeFillResult {
    pub fn faces_fail(&self) -> &[FaceKey] {
        &self.faces_fail
    }
    pub fn is_empty(&self) -> bool {
        self.faces_fail.is_empty()
    }
}

impl LoopMesh {
    /// Rewrite each **destination** face's attributes (and, with `use_normals`,
    /// its winding) by inheriting them from an adjacent face *not* in the
    /// `faces` set — a breadth-first inward flood. Changes **no topology**: only
    /// per-face data, per-corner loop data, and orientation can change.
    ///
    /// The input `faces` are destinations, **not** seeds: data flows *from* the
    /// surrounding unmarked surface *into* the selected set. This is the bare
    /// attribute-inheritance pass [`LoopMesh::holes_fill`] §6 invokes after
    /// capping a hole, exposed on its own.
    ///
    /// - `use_data` — copy the source's face-domain attributes (material index,
    ///   face custom-data, the smooth flag) and per-corner loop data across the
    ///   shared edge (§4.5). Face data is **transitive** (it reaches the centre
    ///   of a thick region through already-resolved neighbours); per-corner loop
    ///   data is **local** (interior corners of a thick region stay default).
    /// - `use_normals` — flip a destination wound *incoherently* with the source
    ///   it inherits from, so the two traverse their shared edge in opposite
    ///   senses (§4.4).
    ///
    /// Both default off and are independent; with both off the flood still runs
    /// purely as a reachability classifier and still reports `faces_fail.out`.
    ///
    /// Algorithm in
    /// [`FACE_ATTRIBUTE_FILL.md`](../../../../radial-mesh-guides/operators/FACE_ATTRIBUTE_FILL.md):
    /// the §3 destination/source tag model, the §4 breadth-first flood (§4.1
    /// frontier seeding, §4.2 ring propagation with resolve-then-promote, §4.3
    /// per-face source selection, §4.4 the winding-coherence flip, §4.5 the
    /// face + loop custom-data inheritance, §4.6 enqueue-before-mutate), and the
    /// §5 `faces_fail.out` reporting.
    pub fn face_attribute_fill(
        &mut self,
        faces: &[FaceKey],
        use_normals: bool,
        use_data: bool,
    ) -> FaceAttributeFillResult {
        let mut result = FaceAttributeFillResult::default();
        if faces.is_empty() {
            return result;
        }

        // §3 — the working tag both marks the destination set and tracks
        // "still-unresolved": resolving a face clears its tag, simultaneously
        // recording it done and promoting it to a source for its neighbours.
        let mut tagged: HashSet<FaceKey> = HashSet::with_capacity(faces.len());
        for &face in faces {
            tagged.insert(face);
        }
        let destination_count = tagged.len();

        // §4.1 — seed the frontier with every loop of a destination face whose
        // shared edge has a source on the other side. A boundary/wire edge is
        // never a frontier (no source across it). Seed in input-buffer order (not
        // `tagged`'s hash order) so a destination bordering two differently-data'd
        // sources resolves against a deterministic first source. `faces` may
        // repeat; `seen` keeps a repeated face from re-seeding its loops.
        let mut current_ring: Vec<LoopKey> = Vec::new();
        let mut next_ring: Vec<LoopKey> = Vec::new();
        let mut seen: HashSet<FaceKey> = HashSet::with_capacity(faces.len());
        for &face in faces {
            if !seen.insert(face) {
                continue;
            }
            let start = self.face_first_loop(face);
            for face_loop in self.face_loop_iter(start) {
                if self.face_attribute_fill_loop_touches_source(face_loop, &tagged) {
                    current_ring.push(face_loop);
                }
            }
        }

        // §4.2 — two-ring breadth-first propagation. A destination face is
        // resolved on the first ring at which any source becomes reachable.
        let mut resolved_count = 0usize;
        while !current_ring.is_empty() {
            for &resolving_loop in &current_ring {
                let face = self.loop_face(resolving_loop);
                // Reached from more than one direction, or already resolved.
                if !tagged.contains(&face) {
                    continue;
                }
                // Step 1 — clear the tag: the face is now done and a source.
                tagged.remove(&face);
                resolved_count += 1;

                // Step 2 — enqueue the next ring *before* any mutation (§4.6),
                // reading the face's adjacency against its original orientation
                // so a later flip cannot perturb which neighbours were scheduled.
                let start = self.face_first_loop(face);
                for other_loop in self.face_loop_iter(start) {
                    if other_loop == resolving_loop {
                        continue;
                    }
                    for radial in self.edge_radial_iter(self.loop_edge(other_loop)) {
                        if radial != other_loop && tagged.contains(&self.loop_face(radial)) {
                            next_ring.push(radial);
                        }
                    }
                }

                // Step 3 — copy from the source across the resolving edge.
                self.face_attribute_fill_resolve(
                    face,
                    resolving_loop,
                    use_normals,
                    use_data,
                    &tagged,
                );
            }
            current_ring.clear();
            std::mem::swap(&mut current_ring, &mut next_ring);
        }

        // §1 / NORMALS_CALC — refresh a representative normal on each resolved
        // (possibly flipped) destination face. Failed faces keep their normal.
        if use_normals {
            for &face in faces {
                if !tagged.contains(&face) {
                    self.face_normal_update(face);
                }
            }
        }

        // §5 — every face still tagged was never reached: a failure. Reported in
        // input order, deduplicated, only when at least one destination failed.
        if resolved_count < destination_count {
            let mut reported: HashSet<FaceKey> = HashSet::new();
            for &face in faces {
                if tagged.contains(&face) && reported.insert(face) {
                    result.faces_fail.push(face);
                }
            }
        }

        result
    }

    /// §4.1 — a loop qualifies as a frontier when the radial cycle of its edge
    /// contains at least one *untagged* (source) face. A boundary edge holds only
    /// the destination face itself, so its loop never seeds.
    fn face_attribute_fill_loop_touches_source(
        &self,
        face_loop: LoopKey,
        tagged: &HashSet<FaceKey>,
    ) -> bool {
        self.edge_radial_iter(self.loop_edge(face_loop))
            .any(|radial| !tagged.contains(&self.loop_face(radial)))
    }

    /// §4.3 / §4.4 / §4.5 — resolve `face`: walk the radial cycle of the
    /// resolving loop's edge past any still-tagged destinations to the first
    /// source, then (when enabled) copy its face-domain + per-corner loop data
    /// and make `face` coherently wound with it. The data copy precedes the
    /// winding flip (§4.6).
    fn face_attribute_fill_resolve(
        &mut self,
        face: FaceKey,
        resolving_loop: LoopKey,
        use_normals: bool,
        use_data: bool,
        tagged: &HashSet<FaceKey>,
    ) {
        // §4.3 — the first untagged face around the radial cycle is the source.
        // An already-resolved destination is untagged and is exactly what the
        // walk stops at; only still-pending destinations are skipped past.
        let mut source_loop = self.loop_radial_next(resolving_loop);
        while tagged.contains(&self.loop_face(source_loop)) {
            source_loop = self.loop_radial_next(source_loop);
        }
        let source_face = self.loop_face(source_loop);

        if use_data {
            // §4.5 — face-domain data is a straight per-face copy from the
            // source (transitive: the source may be a resolved neighbour that
            // already inherited the surround's record). Per-corner loop data is
            // pulled locally from whichever unmarked face borders each corner —
            // a resolved neighbour counts, but its own un-filled interior corners
            // carry default, so a thick region's interior loop data stays default.
            self.holes_fill_copy_face_attribs(source_face, face);
            if !self.loop_attribs().is_empty() {
                self.holes_fill_copy_loop_data(face, tagged);
            }
        }

        if use_normals {
            // §4.4 — coherent neighbours sharing a manifold edge originate their
            // loops at *different* vertices. Same origin => same traversal
            // direction => incoherent => flip the destination.
            if self.loop_vertex(resolving_loop) == self.loop_vertex(source_loop) {
                self.reverse_faces(&[face], false);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use glam::Vec3;

    use super::*;

    /// Register a Vec2 `uv` loop layer and return its key.
    fn register_uv(mesh: &mut LoopMesh) -> ElementAttributeKey {
        mesh.register_loop_attribute(ElementAttributeDesc::numeric(
            AttributeKey::custom("uv"),
            AttrType::Vec2,
        ))
        .expect("uv layer registers")
    }

    /// Build an `n`×`n` grid of unit quad cells in the z=0 plane, every cell a
    /// CCW quad (so all faces wind `+z`). Returns the mesh and an
    /// `n`-by-`n` row-major map of cell -> FaceKey.
    fn grid_faces(n: usize) -> (LoopMesh, Vec<FaceKey>) {
        let mut mesh = LoopMesh::new();
        let stride = n + 1;
        let verts: Vec<VertexKey> = (0..stride)
            .flat_map(|row| (0..stride).map(move |col| (row, col)))
            .map(|(row, col)| mesh.create_vertex(Vec3::new(col as f32, row as f32, 0.0)))
            .collect();
        let at = |row: usize, col: usize| verts[row * stride + col];
        let mut faces = Vec::with_capacity(n * n);
        for row in 0..n {
            for col in 0..n {
                let ring = [
                    at(row, col),
                    at(row, col + 1),
                    at(row + 1, col + 1),
                    at(row + 1, col),
                ];
                faces.push(mesh.face_create_verts(&ring, FaceCreateFlags::NO_DUPLICATE));
            }
        }
        (mesh, faces)
    }

    /// Stamp every non-destination face with `material`, the smooth flag, and a
    /// constant corner `uv_value` — the named destinations are left at default.
    fn decorate_surround_with(
        mesh: &mut LoopMesh,
        uv: ElementAttributeKey,
        faces: &[FaceKey],
        destinations: &[FaceKey],
        material: i32,
        uv_value: [f32; 2],
    ) {
        let dest: HashSet<FaceKey> = destinations.iter().copied().collect();
        for &face in faces {
            if dest.contains(&face) {
                continue;
            }
            mesh.set_face_material_index(face, material);
            mesh.set_face_smooth(face, true);
            for face_loop in mesh.face_loop_keys(face) {
                mesh.loop_attribs_mut()
                    .set_vec2(uv, face_loop.as_usize(), uv_value);
            }
        }
    }

    /// Default surround decoration: material 3, smooth, UV `[0.25, 0.75]`.
    fn decorate_surround(
        mesh: &mut LoopMesh,
        uv: ElementAttributeKey,
        faces: &[FaceKey],
        destinations: &[FaceKey],
    ) {
        decorate_surround_with(mesh, uv, faces, destinations, 3, [0.25, 0.75]);
    }

    #[test]
    fn use_data_floods_center_from_ring() {
        // §4 / §4.5 — a 3×3 grid, the eight ring faces decorated, the centre left
        // default and selected. The centre inherits material / smooth / corner UV
        // from its surround; every corner of the lone centre has an unmarked
        // neighbour, so loop data lands on all four corners.
        let (mut mesh, faces) = grid_faces(3);
        let uv = register_uv(&mut mesh);
        let center = faces[1 * 3 + 1];
        decorate_surround(&mut mesh, uv, &faces, &[center]);

        let result = mesh.face_attribute_fill(&[center], true, true);

        assert!(result.is_empty(), "centre is reachable, no failures");
        assert_eq!(mesh.face_material_index(center), 3, "inherits material");
        assert!(mesh.is_face_smooth(center), "inherits smooth flag");
        for center_loop in mesh.face_loop_keys(center) {
            let got = mesh.loop_attribs().get_vec2(uv, center_loop.as_usize()).unwrap();
            assert_eq!(got, [0.25, 0.75], "every centre corner inherits UV");
        }
        mesh.validate_topology();
    }

    #[test]
    fn use_data_false_leaves_attributes_default() {
        // Contrast (§1.1 / §6): same fixture, data off. The flood runs and the
        // centre is reachable (no failure), but nothing is copied.
        let (mut mesh, faces) = grid_faces(3);
        let uv = register_uv(&mut mesh);
        let center = faces[1 * 3 + 1];
        decorate_surround(&mut mesh, uv, &faces, &[center]);

        let result = mesh.face_attribute_fill(&[center], false, false);

        assert!(result.is_empty(), "centre still reachable with both flags off");
        assert_eq!(mesh.face_material_index(center), 0, "default material kept");
        assert!(!mesh.is_face_smooth(center), "default flat shading kept");
        for center_loop in mesh.face_loop_keys(center) {
            let got = mesh.loop_attribs().get_vec2(uv, center_loop.as_usize()).unwrap();
            assert_eq!(got, [0.0, 0.0], "default corner UV kept");
        }
        mesh.validate_topology();
    }

    /// Two quads sharing edge (v1, v2). `face0` is wound CCW (`+z`); `face1` is
    /// wound so that — *incoherently* — both faces traverse the shared edge the
    /// same direction (their loops on it originate at the same vertex).
    fn two_quads_incoherent() -> (LoopMesh, FaceKey, FaceKey, EdgeKey) {
        let mut mesh = LoopMesh::new();
        let v0 = mesh.create_vertex(Vec3::new(0.0, 0.0, 0.0));
        let v1 = mesh.create_vertex(Vec3::new(1.0, 0.0, 0.0));
        let v2 = mesh.create_vertex(Vec3::new(1.0, 1.0, 0.0));
        let v3 = mesh.create_vertex(Vec3::new(0.0, 1.0, 0.0));
        let v4 = mesh.create_vertex(Vec3::new(2.0, 0.0, 0.0));
        let v5 = mesh.create_vertex(Vec3::new(2.0, 1.0, 0.0));
        // face0 CCW: v0 v1 v2 v3 — its loop on (v1,v2) runs v1 -> v2.
        let face0 = mesh.face_create_verts(&[v0, v1, v2, v3], FaceCreateFlags::NO_DUPLICATE);
        // face1 CW: v1 v4 v5 v2 — its loop on (v1,v2) is v2 -> v1? Build so the
        // shared-edge loop originates at v1 (same as face0) => incoherent.
        // Order v1 v2 v5 v4 makes the (v1,v2) loop run v1 -> v2, same origin.
        let face1 = mesh.face_create_verts(&[v1, v2, v5, v4], FaceCreateFlags::NO_DUPLICATE);
        let shared = mesh.query_edge_exists(v1, v2).expect("shared edge");
        (mesh, face0, face1, shared)
    }

    #[test]
    fn use_normals_flips_incoherent_destination() {
        // §4.4 — face1 is wound incoherently with face0 (its source). With
        // use_normals the destination is flipped to wind coherently; the source
        // face0 is untouched.
        let (mut mesh, face0, face1, shared) = two_quads_incoherent();
        mesh.calc_face_normals();
        let source_normal = mesh.face_normal(face0);

        // Precondition: the two faces traverse the shared edge the same way.
        let radial = mesh.edge_loop_keys(shared);
        assert_eq!(radial.len(), 2);
        assert_eq!(
            mesh.loop_vertex(radial[0]),
            mesh.loop_vertex(radial[1]),
            "fixture is incoherent: loops share an origin vertex"
        );

        let result = mesh.face_attribute_fill(&[face1], true, false);
        assert!(result.is_empty(), "face1 has a source, resolves");

        // After the flip the two faces traverse the shared edge oppositely.
        let radial = mesh.edge_loop_keys(shared);
        assert_ne!(
            mesh.loop_vertex(radial[0]),
            mesh.loop_vertex(radial[1]),
            "destination now coherent with its source"
        );
        assert!(
            (mesh.face_normal(face0) - source_normal).length() < 1e-6,
            "source winding untouched"
        );
        mesh.validate_topology();
    }

    #[test]
    fn use_normals_false_leaves_destination_winding() {
        // Contrast: same incoherent fixture, normals off. The destination is
        // resolved (reachable) but left wound exactly as it was.
        let (mut mesh, _face0, face1, shared) = two_quads_incoherent();
        let before = mesh.face_vert_keys(face1);

        let result = mesh.face_attribute_fill(&[face1], false, false);
        assert!(result.is_empty());

        let radial = mesh.edge_loop_keys(shared);
        assert_eq!(
            mesh.loop_vertex(radial[0]),
            mesh.loop_vertex(radial[1]),
            "still incoherent: winding left alone"
        );
        assert_eq!(mesh.face_vert_keys(face1), before, "loop order unchanged");
        mesh.validate_topology();
    }

    #[test]
    fn thick_region_face_data_transitive_loop_data_local() {
        // §4.2 / §4.5 — a 5×5 grid with the inner 3×3 block selected. The rim of
        // the block resolves first (it touches the decorated frame), then the
        // central face transitively. Face data (material) reaches the centre
        // uniformly; the centre's interior loop corners stay at default.
        let (mut mesh, faces) = grid_faces(5);
        let uv = register_uv(&mut mesh);
        let at = |row: usize, col: usize| faces[row * 5 + col];
        let block: Vec<FaceKey> = (1..4).flat_map(|r| (1..4).map(move |c| (r, c))).map(|(r, c)| at(r, c)).collect();
        decorate_surround(&mut mesh, uv, &faces, &block);
        let center = at(2, 2);

        let result = mesh.face_attribute_fill(&block, true, true);
        assert!(result.is_empty(), "every block face is reachable");

        // Face data is transitive: the whole block ends uniformly materialled.
        for &face in &block {
            assert_eq!(mesh.face_material_index(face), 3, "material floods to interior");
            assert!(mesh.is_face_smooth(face), "smooth floods to interior");
        }

        // Loop data is local: the centre face has no decorated (source) neighbour
        // at any corner — all four neighbours are block faces — so every corner of
        // the centre stays at default UV.
        for center_loop in mesh.face_loop_keys(center) {
            let got = mesh.loop_attribs().get_vec2(uv, center_loop.as_usize()).unwrap();
            assert_eq!(got, [0.0, 0.0], "interior loop corner stays default");
        }

        // A rim face of the block, by contrast, has corners on the frame that did
        // inherit the surround UV — proving the loop pass is partial, not absent.
        let edge_face = at(1, 2); // top-mid of the block, shares an edge with the frame
        let inherited = mesh
            .face_loop_keys(edge_face)
            .into_iter()
            .any(|l| mesh.loop_attribs().get_vec2(uv, l.as_usize()).unwrap() == [0.25, 0.75]);
        assert!(inherited, "rim face inherits UV on its frame-touching corners");
        mesh.validate_topology();
    }

    #[test]
    fn disjoint_destinations_flood_independently() {
        // §7 — two separate 3×3 grids, each with its centre selected. Each surround
        // carries a *different* material so the no-bleed property is observable:
        // each centre must inherit only its own region's value, never the other's.
        let (mut mesh_a, faces_a) = grid_faces(3);
        let uv = register_uv(&mut mesh_a);
        let center_a = faces_a[1 * 3 + 1];
        decorate_surround_with(&mut mesh_a, uv, &faces_a, &[center_a], 3, [0.25, 0.75]);

        // Second grid offset far in x, with a distinct surround value.
        let mut faces_b = Vec::with_capacity(9);
        let stride = 4;
        let base: Vec<VertexKey> = (0..stride)
            .flat_map(|row| (0..stride).map(move |col| (row, col)))
            .map(|(row, col)| mesh_a.create_vertex(Vec3::new(10.0 + col as f32, row as f32, 0.0)))
            .collect();
        let at = |row: usize, col: usize| base[row * stride + col];
        for row in 0..3 {
            for col in 0..3 {
                let ring = [at(row, col), at(row, col + 1), at(row + 1, col + 1), at(row + 1, col)];
                faces_b.push(mesh_a.face_create_verts(&ring, FaceCreateFlags::NO_DUPLICATE));
            }
        }
        let center_b = faces_b[1 * 3 + 1];
        decorate_surround_with(&mut mesh_a, uv, &faces_b, &[center_b], 6, [0.5, 0.125]);

        let result = mesh_a.face_attribute_fill(&[center_a, center_b], true, true);
        assert!(result.is_empty(), "both centres reachable in one call");

        // Each centre inherits *its own* region's value, with no cross-region bleed.
        assert_eq!(mesh_a.face_material_index(center_a), 3, "block A inherits A's ring");
        assert_eq!(mesh_a.face_material_index(center_b), 6, "block B inherits B's ring");
        for center_loop in mesh_a.face_loop_keys(center_a) {
            let got = mesh_a.loop_attribs().get_vec2(uv, center_loop.as_usize()).unwrap();
            assert_eq!(got, [0.25, 0.75], "block A corner UV is A's, not B's");
        }
        for center_loop in mesh_a.face_loop_keys(center_b) {
            let got = mesh_a.loop_attribs().get_vec2(uv, center_loop.as_usize()).unwrap();
            assert_eq!(got, [0.5, 0.125], "block B corner UV is B's, not A's");
        }
        mesh_a.validate_topology();
    }

    #[test]
    fn whole_component_selection_all_fail() {
        // §5 — selecting every face of the grid leaves no source anywhere, so all
        // destinations fail and are returned unchanged in faces_fail.
        let (mut mesh, faces) = grid_faces(3);
        let uv = register_uv(&mut mesh);
        // Give the destinations distinctive data so we can prove they're untouched.
        for &face in &faces {
            mesh.set_face_material_index(face, 7);
            for l in mesh.face_loop_keys(face) {
                mesh.loop_attribs_mut().set_vec2(uv, l.as_usize(), [1.0, 2.0]);
            }
        }

        let result = mesh.face_attribute_fill(&faces, true, true);

        let failed: HashSet<FaceKey> = result.faces_fail().iter().copied().collect();
        let expected: HashSet<FaceKey> = faces.iter().copied().collect();
        assert_eq!(failed, expected, "every face fails — none had a source");
        for &face in &faces {
            assert_eq!(mesh.face_material_index(face), 7, "failed face unchanged");
        }
        mesh.validate_topology();
    }

    #[test]
    fn boundary_only_border_fails() {
        // §4.1 / §7 — a lone quad whose every edge is a boundary edge (radial
        // count 1) has no source across any edge. Boundary edges are never
        // frontiers, so the face is never seeded, never resolves, and comes back
        // in faces_fail unchanged — the cheapest single-face failure.
        let mut mesh = LoopMesh::new();
        let v0 = mesh.create_vertex(Vec3::new(0.0, 0.0, 0.0));
        let v1 = mesh.create_vertex(Vec3::new(1.0, 0.0, 0.0));
        let v2 = mesh.create_vertex(Vec3::new(1.0, 1.0, 0.0));
        let v3 = mesh.create_vertex(Vec3::new(0.0, 1.0, 0.0));
        let face = mesh.face_create_verts(&[v0, v1, v2, v3], FaceCreateFlags::NO_DUPLICATE);
        mesh.set_face_material_index(face, 5);

        let result = mesh.face_attribute_fill(&[face], true, true);

        assert_eq!(result.faces_fail(), [face], "boundary-only face cannot inherit");
        assert_eq!(mesh.face_material_index(face), 5, "failed face left unchanged");
        mesh.validate_topology();
    }

    #[test]
    fn empty_input_is_noop() {
        // §7 — no destinations: the flood seeds and resolves nothing, faces_fail
        // stays empty, mesh unchanged.
        let (mut mesh, faces) = grid_faces(3);
        let before: Vec<i32> = faces.iter().map(|&f| mesh.face_material_index(f)).collect();

        let result = mesh.face_attribute_fill(&[], true, true);

        assert!(result.is_empty(), "empty input yields no failures");
        let after: Vec<i32> = faces.iter().map(|&f| mesh.face_material_index(f)).collect();
        assert_eq!(before, after, "mesh untouched");
        mesh.validate_topology();
    }
}
