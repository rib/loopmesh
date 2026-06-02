//! The **split-edges** operator: peel a set of selected edges apart so the
//! faces that shared them no longer do — "open a seam".
//!
//! This is the operator-level wrapper over two separate primitives, exactly as
//! laid out in `radial-mesh-guides/operators/SPLIT_EDGES.md`:
//!
//! - **edge-separate** ([`LoopMesh::euler_edge_separate`],
//!   `SPLIT_EDGES.md` §3): peel one face off a manifold edge onto a fresh
//!   coincident duplicate edge, iterated until the edge carries one face.
//! - **vert-separate** ([`LoopMesh::euler_vert_separate`],
//!   `SPLIT_EDGES.md` §4.2): cut a vertex's disk into the connected fans that
//!   survive once the tagged edges are peeled, giving each disconnected fan its
//!   own duplicate vertex.
//!
//! The wrapper itself contributes only the tagging discipline plus the
//! `use_verts` vertex-eligibility rule (§5) — including the §5.3
//! both-endpoints-unflagged guard — and the §4.3 splice cleanup that removes
//! any accidental coincident double edge left when an endpoint stays welded.
//!
//! New duplicate edges and vertices copy their source element's attributes
//! verbatim (one source per duplicate; a copy, never an interpolation — §1).
//!
//! Distinct from the *subdivide* edge-split Euler operator (which inserts a
//! midpoint vertex) and from [`LoopMesh::split`] (which tears a selection into a
//! disjoint copy).

use smallvec::SmallVec;

use crate::loopmesh::*;

impl LoopMesh {
    /// Open a seam along `edges`, peeling apart the faces that shared them.
    ///
    /// See [`Self::split_edges_out`] for the full contract; this convenience
    /// entry point discards the `edges.out` seam buffer.
    pub fn split_edges(&mut self, edges: &[EdgeKey], verts: &[VertexKey], use_verts: bool) {
        let _ = self.split_edges_out(edges, verts, use_verts);
    }

    /// Open a seam along `edges` and return every edge along the opened seam.
    ///
    /// Implements `radial-mesh-guides/operators/SPLIT_EDGES.md`:
    ///
    /// - Each selected manifold (radial ≥ 2) edge is edge-separated (§3) down
    ///   to one face per edge: a radial-`k` edge becomes `k` coincident edges.
    /// - Vertex eligibility (§5) decides which endpoints are cut. With
    ///   `use_verts = false` (the default) every endpoint of every selected
    ///   edge is eligible; with `use_verts = true` only the vertices named in
    ///   `verts` are, except that a tagged edge whose **both** endpoints are
    ///   ineligible forces both eligible (§5.3) so the two copies never weld at
    ///   both ends.
    /// - Each eligible vertex's disk is cut into connected fans (§4.2): a disk
    ///   that splits into `f` fans yields `f` coincident vertices; a disk that
    ///   stays one fan (an unselected edge still bridges the cut, or the
    ///   endpoint was left ineligible) is **not** duplicated.
    /// - A final cleanup (§4.3) splices back any coincident double edge left
    ///   between the same vertex pair when an endpoint stayed welded.
    ///
    /// The returned buffer is the §1/§7 `edges.out` slot: **every** edge along
    /// the opened seam after separation — the live originals **plus** the
    /// coincident duplicates peeled off them. A radial-`k` split edge
    /// contributes `k` entries; a boundary/wire selected edge contributes only
    /// itself. Order is not part of the contract.
    pub fn split_edges_out(
        &mut self,
        edges: &[EdgeKey],
        verts: &[VertexKey],
        use_verts: bool,
    ) -> Vec<EdgeKey> {
        // §6 empty-input fast path.
        if edges.is_empty() {
            return Vec::new();
        }

        // Dedup the tagged edges, dropping any already-dead key, so an edge
        // named twice is only peeled once and its seam group is one run.
        let mut tagged: SmallVec<[EdgeKey; 8]> = SmallVec::new();
        for &edge in edges {
            if self.edges.try_get(edge).is_some() && !tagged.contains(&edge) {
                tagged.push(edge);
            }
        }
        if tagged.is_empty() {
            return Vec::new();
        }

        // §5 vertex eligibility. O(1) membership via an op-flag layer, mirroring
        // the tagging idiom the other set operators use.
        const BIT: u8 = 0;
        let eligible = self.push_vert_op_flag_layer();

        if use_verts {
            // §5.2 — only the named verts are eligible.
            for &v in verts {
                if self.vertices.try_get(v).is_some() {
                    self.vert_op_flag_set(eligible, v, BIT);
                }
            }
            // §5.3 — a tagged edge with neither endpoint eligible would leave a
            // coincident double edge welded at both ends; force both eligible.
            for &edge in &tagged {
                let [a, b] = self.edges.get(edge).vertices;
                if !self.vert_op_flag_test(eligible, a, BIT)
                    && !self.vert_op_flag_test(eligible, b, BIT)
                {
                    self.vert_op_flag_set(eligible, a, BIT);
                    self.vert_op_flag_set(eligible, b, BIT);
                }
            }
        } else {
            // §5.1 — every endpoint of every tagged edge is eligible.
            for &edge in &tagged {
                let [a, b] = self.edges.get(edge).vertices;
                self.vert_op_flag_set(eligible, a, BIT);
                self.vert_op_flag_set(eligible, b, BIT);
            }
        }

        // §4.1 / §3 — edge-separate every tagged manifold edge down to one face
        // per edge. Each seam group records the original plus its duplicates so
        // the §4.3 cleanup can splice within the group and §7 can report it.
        let mut seam_groups: SmallVec<[SmallVec<[EdgeKey; 4]>; 8]> = SmallVec::new();
        let mut endpoints: SmallVec<[VertexKey; 16]> = SmallVec::new();
        for &edge in &tagged {
            let [a, b] = self.edges.get(edge).vertices;
            if !endpoints.contains(&a) {
                endpoints.push(a);
            }
            if !endpoints.contains(&b) {
                endpoints.push(b);
            }

            let mut group: SmallVec<[EdgeKey; 4]> = SmallVec::new();
            group.push(edge);
            // Peel one face at a time until the edge is radial-1 (or a boundary
            // /wire edge, which supports no separation: the loop never runs).
            loop {
                let loops: SmallVec<[LoopKey; 4]> =
                    self.edge_radial_iter(edge).collect();
                if loops.len() < 2 {
                    break;
                }
                let l_sep = loops[0];
                let e_new = self.euler_edge_separate(edge, l_sep);
                group.push(e_new);
            }
            seam_groups.push(group);
        }

        // §4.2 — cut each eligible endpoint's disk into its connected fans. A
        // disk that stays one fan is a no-op inside the primitive.
        for &v in &endpoints {
            if self.vert_op_flag_test(eligible, v, BIT) && self.vertices.try_get(v).is_some() {
                let _ = self.euler_vert_separate(v);
            }
        }

        // §4.3 — splice back any coincident double edge left within a seam
        // group: two of a group's edges that ended up on the same vertex pair
        // (because the fan they belonged to was not actually separated) are a
        // redundant duplicate. Merge their radial cycles and drop the extra.
        for group in &mut seam_groups {
            let mut i = 0;
            while i < group.len() {
                let dst = group[i];
                let mut j = i + 1;
                while j < group.len() {
                    let src = group[j];
                    if self.split_edges_same_endpoints(dst, src) {
                        self.edge_splice(dst, src);
                        group.swap_remove(j);
                    } else {
                        j += 1;
                    }
                }
                i += 1;
            }
        }

        self.pop_vert_op_flag_layer(eligible);

        // §7 edges.out — every live seam edge across all groups.
        let mut out: Vec<EdgeKey> = Vec::new();
        for group in &seam_groups {
            for &edge in group {
                if self.edges.try_get(edge).is_some() {
                    out.push(edge);
                }
            }
        }
        out
    }

    /// True iff two live edges connect the identical (unordered) vertex pair —
    /// the §4.3 redundant-double-edge predicate.
    fn split_edges_same_endpoints(&self, a: EdgeKey, b: EdgeKey) -> bool {
        if self.edges.try_get(a).is_none() || self.edges.try_get(b).is_none() {
            return false;
        }
        let [a0, a1] = self.edges.get(a).vertices;
        let [b0, b1] = self.edges.get(b).vertices;
        (a0 == b0 && a1 == b1) || (a0 == b1 && a1 == b0)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use glam::Vec3;

    use crate::loopmesh::*;

    /// Two unit quads sharing the interior edge `{1, 2}`:
    ///
    /// ```text
    ///   3 ---- 2 ---- 5
    ///   |  A   |  B   |
    ///   0 ---- 1 ---- 4
    /// ```
    ///
    /// Returns `(verts[6], faces[2], shared_edge)`.
    fn build_two_quads(mesh: &mut LoopMesh) -> (Vec<VertexKey>, Vec<FaceKey>, EdgeKey) {
        let coords = [
            (0.0, 0.0),
            (1.0, 0.0),
            (1.0, 1.0),
            (0.0, 1.0),
            (2.0, 0.0),
            (2.0, 1.0),
        ];
        let verts: Vec<VertexKey> = coords
            .into_iter()
            .map(|(x, y)| mesh.create_vertex(Vec3::new(x, y, 0.0)))
            .collect();

        let mut edge = |mesh: &mut LoopMesh, a: usize, b: usize| {
            mesh.create_edge([verts[a], verts[b]], EdgeCreateFlags::NO_DUPLICATE)
        };
        // Quad A: 0,1,2,3.
        let e01 = edge(mesh, 0, 1);
        let e12 = edge(mesh, 1, 2);
        let e23 = edge(mesh, 2, 3);
        let e30 = edge(mesh, 3, 0);
        let a = mesh.face_create(
            &[verts[0], verts[1], verts[2], verts[3]],
            &[e01, e12, e23, e30],
            FaceCreateFlags::NONE,
        );
        // Quad B: 1,4,5,2 — reuses e12 as the shared edge.
        let e14 = edge(mesh, 1, 4);
        let e45 = edge(mesh, 4, 5);
        let e52 = edge(mesh, 5, 2);
        let b = mesh.face_create(
            &[verts[1], verts[4], verts[5], verts[2]],
            &[e14, e45, e52, e12],
            FaceCreateFlags::NONE,
        );
        (verts, vec![a, b], e12)
    }

    /// Book of `pages` triangles all hinged on one spine edge `{0, 1}`. Page `i`
    /// has apex vert `2 + i`. Returns `(verts, spine_edge)`.
    fn build_book(mesh: &mut LoopMesh, pages: usize) -> (Vec<VertexKey>, EdgeKey) {
        let mut verts = vec![
            mesh.create_vertex(Vec3::new(0.0, 0.0, 0.0)),
            mesh.create_vertex(Vec3::new(1.0, 0.0, 0.0)),
        ];
        let spine = mesh.create_edge([verts[0], verts[1]], EdgeCreateFlags::NO_DUPLICATE);
        for i in 0..pages {
            let angle = std::f32::consts::PI * (i as f32) / (pages as f32);
            let apex = mesh.create_vertex(Vec3::new(0.5, angle.cos(), angle.sin()));
            verts.push(apex);
            let e_a = mesh.create_edge([verts[0], apex], EdgeCreateFlags::NO_DUPLICATE);
            let e_b = mesh.create_edge([apex, verts[1]], EdgeCreateFlags::NO_DUPLICATE);
            mesh.face_create(
                &[verts[0], verts[1], apex],
                &[spine, e_b, e_a],
                FaceCreateFlags::NONE,
            );
        }
        (verts, spine)
    }

    /// 3x3 face grid (16 verts, 24 edges, 9 faces). `verts[r*4+c]`,
    /// `faces[r*3+c]`. The 4-valence centre is `verts[5]`.
    fn build_grid_3x3(mesh: &mut LoopMesh) -> (Vec<VertexKey>, Vec<FaceKey>) {
        let mut verts = Vec::with_capacity(16);
        for r in 0..4 {
            for c in 0..4 {
                verts.push(mesh.create_vertex(Vec3::new(c as f32, r as f32, 0.0)));
            }
        }
        let mut faces = Vec::new();
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

    /// Find the live edge connecting two verts, if any.
    fn edge_between(mesh: &LoopMesh, a: VertexKey, b: VertexKey) -> Option<EdgeKey> {
        mesh.vertex_disk_iter(a)
            .find(|&e| mesh.edges.get(e).contains(b).is_some())
    }

    /// SPLIT_EDGES.md §2 / §6 headline: one interior edge between two quads,
    /// `use_verts = false`. Two coincident edges, both endpoints doubled. Net
    /// +2 V / +1 E / +0 F, and the result is two disjoint quads sharing no
    /// edge and no vert.
    #[test]
    fn split_edges_manifold_single_edge_opens_seam() {
        let mut mesh = LoopMesh::new();
        let (verts, _faces, shared) = build_two_quads(&mut mesh);
        let (v0, e0, f0) = (mesh.totvert(), mesh.totedge(), mesh.totface());
        assert_eq!((v0, e0, f0), (6, 7, 2));
        assert_eq!(mesh.edge_face_keys(shared).len(), 2, "shared edge is radial-2");

        mesh.split_edges(&[shared], &[], false);

        assert_eq!(mesh.totvert(), v0 + 2, "+2 V (both endpoints doubled)");
        assert_eq!(mesh.totedge(), e0 + 1, "+1 E (one coincident duplicate)");
        assert_eq!(mesh.totface(), f0, "+0 F");

        // The two quads now share no edge and no vert: vert 1 and vert 2 each
        // split, so neither face touches the other.
        let faces = mesh.face_keys();
        assert_eq!(faces.len(), 2);
        let f_a: HashSet<VertexKey> = mesh.face_vert_keys(faces[0]).into_iter().collect();
        let f_b: HashSet<VertexKey> = mesh.face_vert_keys(faces[1]).into_iter().collect();
        assert_eq!(f_a.len(), 4);
        assert_eq!(f_b.len(), 4);
        assert!(f_a.is_disjoint(&f_b), "the two quads share no vertex");
        // The original corner verts 0 and 3 survive (quad A keeps them).
        let _ = verts;
        mesh.validate_topology();
    }

    /// SPLIT_EDGES.md §1 / §7: the `edges.out` slot reports every edge along
    /// the opened seam — the original plus its duplicate. One radial-2 split
    /// edge -> 2 live seam edges, both connecting the same coincident endpoints
    /// only after their verts split.
    #[test]
    fn split_edges_out_slot_reports_seam_originals_plus_duplicates() {
        let mut mesh = LoopMesh::new();
        let (_verts, _faces, shared) = build_two_quads(&mut mesh);

        let out = mesh.split_edges_out(&[shared], &[], false);
        assert_eq!(out.len(), 2, "seam = original + one duplicate");
        let unique: HashSet<EdgeKey> = out.iter().copied().collect();
        assert_eq!(unique.len(), 2, "the two seam edges are distinct");
        for &e in &out {
            assert!(mesh.edges.try_get(e).is_some(), "every reported edge is live");
            assert_eq!(mesh.edge_face_keys(e).len(), 1, "each seam edge carries one face");
        }
        mesh.validate_topology();
    }

    /// SPLIT_EDGES.md §5.1 / §5.2 contrast pair — the `use_verts` discriminator
    /// is the vertex count. Same fixture, same selected edge:
    ///   - `use_verts = false` (or both endpoints named) -> +2 V.
    ///   - `use_verts = true`, one endpoint named (rips half) -> +1 V; the
    ///     unnamed endpoint stays welded so the two faces hinge there.
    #[test]
    fn split_edges_use_verts_contrast_pair() {
        // Both endpoints named: fires fully, +2 V.
        {
            let mut mesh = LoopMesh::new();
            let (verts, _faces, shared) = build_two_quads(&mut mesh);
            let (v0, e0, f0) = (mesh.totvert(), mesh.totedge(), mesh.totface());
            mesh.split_edges(&[shared], &[verts[1], verts[2]], true);
            assert_eq!(mesh.totvert(), v0 + 2, "both named -> +2 V");
            assert_eq!(mesh.totedge(), e0 + 1, "+1 E");
            assert_eq!(mesh.totface(), f0, "+0 F");
            mesh.validate_topology();
        }
        // Only vert 1 named: rips half, +1 V; vert 2 stays welded.
        {
            let mut mesh = LoopMesh::new();
            let (verts, _faces, shared) = build_two_quads(&mut mesh);
            let (v0, e0, f0) = (mesh.totvert(), mesh.totedge(), mesh.totface());
            mesh.split_edges(&[shared], &[verts[1]], true);
            assert_eq!(mesh.totvert(), v0 + 1, "one named -> +1 V (rips half)");
            assert_eq!(mesh.totedge(), e0 + 1, "+1 E (the duplicate edge stays)");
            assert_eq!(mesh.totface(), f0, "+0 F");
            // Vert 2 stays single: both faces still contain it (the surface
            // hinges there), while vert 1 doubled so the faces parted at that
            // end.
            let faces = mesh.face_keys();
            let touching_v2 = faces
                .iter()
                .filter(|&&f| mesh.face_vert_keys(f).contains(&verts[2]))
                .count();
            assert_eq!(touching_v2, 2, "vert 2 stays welded — both faces still hinge on it");
            mesh.validate_topology();
        }
    }

    /// SPLIT_EDGES.md §5.3 both-endpoints-unflagged guard: `use_verts = true`
    /// with an *empty* `verts` set. A tagged edge with neither endpoint
    /// eligible would leave a coincident double edge; the guard forces both
    /// eligible, so the seam opens fully — matching the both-named case
    /// (+2 V), not a no-op and not a stranded double edge.
    #[test]
    fn split_edges_use_verts_both_unflagged_guard_forces_full_seam() {
        let mut mesh = LoopMesh::new();
        let (_verts, _faces, shared) = build_two_quads(&mut mesh);
        let (v0, e0, f0) = (mesh.totvert(), mesh.totedge(), mesh.totface());

        mesh.split_edges(&[shared], &[], true);

        assert_eq!(mesh.totvert(), v0 + 2, "guard forced both eligible -> +2 V");
        assert_eq!(mesh.totedge(), e0 + 1, "+1 E");
        assert_eq!(mesh.totface(), f0, "+0 F");
        // No stranded double edge: every live edge connects a unique vert pair.
        let mut pairs: HashSet<(VertexKey, VertexKey)> = HashSet::new();
        for e in mesh.edge_keys() {
            let [a, b] = mesh.edge_vert_keys(e);
            let key = if a.as_usize() <= b.as_usize() { (a, b) } else { (b, a) };
            assert!(pairs.insert(key), "no coincident double edge survives the guard");
        }
        mesh.validate_topology();
    }

    /// SPLIT_EDGES.md §3 / §6: the radial-3 spine of a 3-page book peels into
    /// three coincident edges (+2 E), and each spine endpoint's disk fans into
    /// three (+2 V each) -> net +4 V / +2 E. Each page detaches into a free
    /// triangle.
    #[test]
    fn split_edges_nonmanifold_spine_separates_fans() {
        let mut mesh = LoopMesh::new();
        let (_verts, spine) = build_book(&mut mesh, 3);
        let (v0, e0, f0) = (mesh.totvert(), mesh.totedge(), mesh.totface());
        assert_eq!((v0, e0, f0), (5, 7, 3));
        assert_eq!(mesh.edge_face_keys(spine).len(), 3, "spine is radial-3");

        let out = mesh.split_edges_out(&[spine], &[], false);

        assert_eq!(mesh.totvert(), v0 + 4, "+4 V (each endpoint fans into three)");
        assert_eq!(mesh.totedge(), e0 + 2, "+2 E (spine -> three coincident edges)");
        assert_eq!(mesh.totface(), f0, "+0 F (three pages survive)");
        assert_eq!(out.len(), 3, "edges.out = three coincident spine edges");

        // Every face is now a free triangle: each of its three edges is
        // radial-1.
        for f in mesh.face_keys() {
            for e in mesh.face_edge_keys(f) {
                assert_eq!(
                    mesh.edge_face_keys(e).len(),
                    1,
                    "every page edge carries exactly one face"
                );
            }
        }
        mesh.validate_topology();
    }

    /// SPLIT_EDGES.md §4.2 / §6 crossing seams at a shared interior vertex: the
    /// two collinear interior edges `{1,5}` and `{5,9}` of a 3x3 grid meet at
    /// the centre-column vert 5. Splitting both cuts vert 5's disk by both at
    /// once; its duplication count is fan-connectivity-driven. The vertical
    /// seam parts the grid's left two columns from the right column.
    #[test]
    fn split_edges_crossing_seams_at_shared_vert() {
        let mut mesh = LoopMesh::new();
        let (verts, _faces) = build_grid_3x3(&mut mesh);
        let (v0, e0, f0) = (mesh.totvert(), mesh.totedge(), mesh.totface());
        assert_eq!((v0, e0, f0), (16, 24, 9));

        // The middle vertical seam: the three stacked interior edges down
        // column 1 — {1,5}, {5,9}, {9,13} — fully separate the left block from
        // the right. Use the two that meet at interior vert 5: {1,5},{5,9}.
        let e15 = edge_between(&mesh, verts[1], verts[5]).unwrap();
        let e59 = edge_between(&mesh, verts[5], verts[9]).unwrap();
        let e9_13 = edge_between(&mesh, verts[9], verts[13]).unwrap();
        assert_eq!(mesh.edge_face_keys(e15).len(), 2, "{{1,5}} is interior");
        assert_eq!(mesh.edge_face_keys(e59).len(), 2, "{{5,9}} is interior");

        // Split the whole vertical interior seam (all three stacked edges) so
        // the grid parts cleanly into left (cols 0..2) and right (col 2..3).
        mesh.split_edges(&[e15, e59, e9_13], &[], false);

        // Three interior edges duplicated -> +3 E. The four interior verts on
        // the seam (1, 5, 9, 13) each split into two fans (left/right) -> +4 V.
        assert_eq!(mesh.totedge(), e0 + 3, "+3 E (three duplicated seam edges)");
        assert_eq!(mesh.totvert(), v0 + 4, "+4 V (each seam vert fans into two)");
        assert_eq!(mesh.totface(), f0, "+0 F");
        mesh.validate_topology();
    }

    /// SPLIT_EDGES.md §6 wire / empty no-ops.
    #[test]
    fn split_edges_wire_and_empty_are_noops() {
        // Two disjoint wire edges: nothing to peel.
        {
            let mut mesh = LoopMesh::new();
            let a = mesh.create_vertex(Vec3::new(0.0, 0.0, 0.0));
            let b = mesh.create_vertex(Vec3::new(1.0, 0.0, 0.0));
            let c = mesh.create_vertex(Vec3::new(2.0, 0.0, 0.0));
            let d = mesh.create_vertex(Vec3::new(3.0, 0.0, 0.0));
            let w0 = mesh.create_edge([a, b], EdgeCreateFlags::NO_DUPLICATE);
            let w1 = mesh.create_edge([c, d], EdgeCreateFlags::NO_DUPLICATE);
            let (v0, e0, f0) = (mesh.totvert(), mesh.totedge(), mesh.totface());

            let out = mesh.split_edges_out(&[w0, w1], &[], false);

            assert_eq!(mesh.totvert(), v0, "wire no-op: V unchanged");
            assert_eq!(mesh.totedge(), e0, "wire no-op: E unchanged");
            assert_eq!(mesh.totface(), f0, "wire no-op: F unchanged");
            // Each wire edge contributes only itself to the seam.
            assert_eq!(out.len(), 2, "edges.out = the two wire edges themselves");
            mesh.validate_topology();
        }
        // Empty input over a real mesh: whole no-op, empty edges.out.
        {
            let mut mesh = LoopMesh::new();
            let (_verts, _faces, _shared) = build_two_quads(&mut mesh);
            let (v0, e0, f0) = (mesh.totvert(), mesh.totedge(), mesh.totface());

            let out = mesh.split_edges_out(&[], &[], false);

            assert!(out.is_empty(), "empty input -> empty edges.out");
            assert_eq!(mesh.totvert(), v0);
            assert_eq!(mesh.totedge(), e0);
            assert_eq!(mesh.totface(), f0);
            mesh.validate_topology();
        }
    }
}
