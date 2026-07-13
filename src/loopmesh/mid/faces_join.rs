use crate::loopmesh::*;

/// What [`LoopMesh::faces_join_report_double`] hands back: the union the join
/// minted, plus the pre-existing face it coincides with, if any
/// (`DISSOLVE_FACES.md` §4.2). Both faces are alive on return — resolving the
/// overlap belongs to the caller.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FacesJoinOutcome {
    pub union: FaceKey,
    pub double: Option<FaceKey>,
}

impl LoopMesh {
    /// The **composed, loop-minting** face join
    /// (`RADIAL_EDGE_STRUCTURE.md` §9), specialised to a manifold pair sharing
    /// `edge`. This — not the raw Euler join
    /// ([`LoopMesh::euler_join_face_kill_edge`]) — is the merge every
    /// dissolve-family operator and `edge_rotate` are built on: it constructs
    /// the union, so every edge of the union's perimeter takes the union's loop
    /// as its radial **entry**, and the donor rules that read the entry as a
    /// face-age channel depend on it. The union keeps `donor`'s handle and
    /// per-face attributes (§9 seeds the union from the caller's first face).
    pub fn faces_join_pair(
        &mut self,
        donor: FaceKey,
        absorbed: FaceKey,
        edge: EdgeKey,
    ) -> Option<FaceKey> {
        let union = self.euler_join_face_kill_edge(donor, absorbed, edge)?;
        self.face_seat_perimeter_as_constructed(union);
        Some(union)
    }

    /// Bring `face`'s radial cycles to the state a *constructed* union leaves
    /// them in: the union's fresh loop appended to each perimeter edge (landing
    /// immediately after that edge's pre-merge entry, and becoming the entry),
    /// followed by the removal of the source face's loop from that same cycle
    /// (`RADIAL_EDGE_STRUCTURE.md` §6.5's append + removal rules, §9).
    ///
    /// Since this crate's joins re-use a source loop as the union's loop rather
    /// than minting a fresh one, "append the union's loop, then remove the
    /// source's" collapses to: **move that one loop to just after the pre-merge
    /// entry, and seat the entry on it**. The re-splice is skipped when the loop
    /// already *is* the entry — there, appending after it and then deleting it
    /// leaves the fresh loop in exactly the slot the source loop already
    /// occupies, so any churn would rotate the cycle where the reference does
    /// not (invisible below radial 3, wrong above it).
    ///
    /// Call this **once per union**, over the union's *final* perimeter. Seating
    /// each intermediate union of an n-way merge would leave loops in slots the
    /// reference never occupies (§9's "mint once").
    pub(crate) fn face_seat_perimeter_as_constructed(&mut self, face: FaceKey) {
        // Only radial links are touched, never the face-loop cycle being walked,
        // so the cursor is stable and the walk allocation-free.
        let first = self.faces.get(face).first_edge_loop;
        let mut cursor = first;
        loop {
            let next = self.loops.get(cursor).face_loop_links.next;
            let perimeter_edge = self.loops.get(cursor).edge;
            if !perimeter_edge.is_null() && self.edge_radial_entry(perimeter_edge) != Some(cursor) {
                self.face_loop_disconnect_edge(cursor);
                self.face_loop_connect_edge(perimeter_edge, cursor);
            }
            cursor = next;
            if cursor == first {
                break;
            }
        }
    }

    /// The **many-input** face join
    /// (`DISSOLVE_FACES.md` §4.1): merge a connected region of faces into one
    /// fresh union face laid over the region's boundary. Returns the merge
    /// result, or `None` if the region is rejected (§4.3).
    ///
    /// The join **never deletes an input face directly** (§4.1.2). It collects
    /// two kill-sets — interior edges nobody outside the region uses, and
    /// vertices buried entirely inside the region — constructs the union over
    /// the region's *boundary* edges, and then kills the two sets. Input faces
    /// die only as **collateral** of those kills, so an input face incident to
    /// neither a collected edge nor a collected vertex **survives** the join,
    /// overlapped by the union (§4.1.4). Callers that do not want the survivors
    /// must sweep them themselves — [`LoopMesh::dissolve_faces`] does, in its §5
    /// cleanup pass.
    ///
    /// Constructing over the boundary — rather than chaining pairwise splices
    /// through the interior edges — is load-bearing, not a style choice (§9.1):
    /// a pairwise splice across a **kept-alive** interior edge necessarily
    /// consumes both its faces, and yields a *smaller* union than the region's
    /// boundary. It is also what mints the union exactly once, so every
    /// perimeter edge takes the union's loop as its radial entry and no
    /// region-interior edge ever does (`RADIAL_EDGE_STRUCTURE.md` §9).
    pub fn faces_join(&mut self, faces: &[FaceKey]) -> Option<FaceKey> {
        if faces.len() < 2 {
            return faces.first().copied();
        }

        let (union, double, plan) = self.faces_join_build(faces, CallingForm::Resolve)?;

        // §4.2 — the *resolving* calling form: a pre-existing face carrying the
        // union's corner cycle wins, and the fresh union is killed before the
        // cascade ever runs. Callers that need to see the double instead reach
        // for [`LoopMesh::faces_join_report_double`].
        let result = match double {
            Some(shadow) => {
                self.face_kill(union);
                shadow
            }
            None => union,
        };

        self.faces_join_apply_cascade(plan);
        Some(result)
    }

    /// The **many-input** face join in its §4.2 *report-the-double* calling form:
    /// the primitive keeps both faces and hands the caller the freshly-minted
    /// union alongside any pre-existing face whose corner cycle coincides with
    /// it. Resolving the overlap is then the caller's job — as is knowing that
    /// there *was* one, which is what [`LoopMesh::dissolve_faces`] needs for its
    /// §5 survivor arithmetic.
    ///
    /// The classification, the kill-sets, the cascade and the §4.1.4 survivors
    /// are exactly [`LoopMesh::faces_join`]'s. The one divergence is the §4.1.5
    /// per-loop transfer: this form always runs it, because the union it hands
    /// back is **alive**, and a live face owes the caller initialised per-corner
    /// data even when a double exists (§9's `reusing` fork).
    pub fn faces_join_report_double(&mut self, faces: &[FaceKey]) -> Option<FacesJoinOutcome> {
        if faces.len() < 2 {
            return faces.first().copied().map(|union| FacesJoinOutcome {
                union,
                double: None,
            });
        }

        let (union, double, plan) = self.faces_join_build(faces, CallingForm::Report)?;
        self.faces_join_apply_cascade(plan);
        Some(FacesJoinOutcome { union, double })
    }

    /// §4.1.2 steps 1 + 2 — classify, chain the boundary, mint the union, run the
    /// §4.2 coincident-face probe. The cascade is *not* applied: the plan comes
    /// back un-executed so each calling form can settle the double before or
    /// after it, as its contract requires.
    fn faces_join_build(
        &mut self,
        faces: &[FaceKey],
        form: CallingForm,
    ) -> Option<(FaceKey, Option<FaceKey>, FacesJoinPlan)> {
        let region: std::collections::HashSet<FaceKey> = faces.iter().copied().collect();

        let plan = self.faces_join_collect(faces, &region)?;
        let (ring_verts, ring_edges) = self.faces_join_chain_boundary(&plan)?;

        // §4.1.2 step 2 — construct. Every perimeter edge gains a freshly-minted
        // loop appended after its pre-merge entry, which is what seats the union
        // as each perimeter edge's entry; the source loops leave those cycles
        // later, in the cascade (`RADIAL_EDGE_STRUCTURE.md` §6.5's append +
        // removal rules).
        let union = self.face_create(&ring_verts, &ring_edges, FaceCreateFlags::NONE);
        let seed_face = faces[0];
        self.face_attribs
            .copy_into(union.as_usize(), seed_face.as_usize());
        let seed_eflags = self.faces.get(seed_face).eflags;
        self.faces.get_mut(union).eflags = seed_eflags;
        self.face_normal_update(union);

        let double = self.faces_join_find_coincident_face(union);
        // §9's `reusing` fork: the per-loop transfer is skipped only where the
        // pre-existing face is *adopted* as the merge result — it already carries
        // its own per-corner data, and the union it displaces is about to die.
        // The mere existence of a double does not skip it: under the report form
        // the union stays alive, and a live face with uninitialised corners is a
        // live face with wrong corners.
        let reusing = double.is_some() && form == CallingForm::Resolve;
        if !reusing {
            self.faces_join_transfer_boundary_loop_attrs(union, &region);
        }

        Some((union, double, plan))
    }

    /// §4.1.2 steps 3 + 4 — the cascade, and the only thing in the join that
    /// deletes an input face. The two kill-sets partition the dying-edge set
    /// (§7), so no edge is killed twice: an interior edge with a buried endpoint
    /// is never collected, and dies inside that vertex's disk instead.
    ///
    /// The union is never collateral here: it is built over the region's
    /// *boundary* edges, and neither kill-set can touch one (a boundary edge
    /// carries an out-of-region face or nothing at all, so it is never a
    /// collected interior edge, and its endpoints are never region-enclosed).
    fn faces_join_apply_cascade(&mut self, plan: FacesJoinPlan) {
        for edge in plan.kill_edges {
            self.edge_kill(edge);
        }
        for vertex in plan.kill_verts {
            self.vertex_kill(vertex);
        }
    }

    /// §4.1.2 step 1 + §4.1.1 + §4.1.3 — classify every edge of every input face
    /// by its **region count** (input faces on its radial, never the radial's
    /// total length) and route the interior ones into the two kill-sets.
    /// Modifies nothing. `None` means the region carries an **over-shared** edge
    /// and the join declines (§4.3).
    fn faces_join_collect(
        &self,
        faces: &[FaceKey],
        region: &std::collections::HashSet<FaceKey>,
    ) -> Option<FacesJoinPlan> {
        let mut plan = FacesJoinPlan::default();

        for &face in faces {
            let mut face_loops = FaceLoops::new(self, face);
            while let Some(loop_key) = face_loops.next(self) {
                let edge = self.loops.get(loop_key).edge;
                if edge.is_null() {
                    continue;
                }

                let mut region_count = 0usize;
                let mut total_count = 0usize;
                let mut radial = self.radial_loops(edge);
                while let Some(radial_loop) = radial.next(self) {
                    total_count += 1;
                    if region.contains(&self.loops.get(radial_loop).face) {
                        region_count += 1;
                    }
                }

                if region_count >= 3 {
                    return None;
                }

                if region_count == 1 {
                    if plan.boundary.insert(edge) {
                        // The union's winding is seeded from the first boundary
                        // edge of the first input face that has one: the union
                        // sits where its source face sat, so it must traverse
                        // that edge the same way the source loop does.
                        if plan.winding_seed.is_none() {
                            plan.winding_seed = Some((edge, self.loops.get(loop_key).vertex));
                        }
                    }
                    continue;
                }

                let [v0, v1] = self.edges.get(edge).vertices;
                let enclosed_0 = self.vertex_is_region_enclosed(v0, region);
                let enclosed_1 = self.vertex_is_region_enclosed(v1, region);

                if !enclosed_0 && !enclosed_1 {
                    // Keep-alive (§4.1.3): an out-of-region face is on this edge
                    // too, so killing it would take that face down. Collect
                    // nothing — the join steps *around* such an edge, and the two
                    // input faces on it are consumed only if some *other* edge or
                    // vertex of theirs is collected.
                    if total_count <= 2 && !plan.kill_edges.contains(&edge) {
                        plan.kill_edges.push(edge);
                    }
                } else {
                    // The buried vertex dies instead of the edge; its kill takes
                    // the whole disk, so collecting the edge as well would be a
                    // double kill.
                    if enclosed_0 && !plan.kill_verts.contains(&v0) {
                        plan.kill_verts.push(v0);
                    }
                    if enclosed_1 && !plan.kill_verts.contains(&v1) {
                        plan.kill_verts.push(v1);
                    }
                }
            }
        }

        Some(plan)
    }

    /// §4.1.3 — "buried in the region with no line of sight out": at least one
    /// incident edge, no wire or boundary edge in the disk, and every face on
    /// every disk edge's radial in the input set.
    fn vertex_is_region_enclosed(
        &self,
        vertex: VertexKey,
        region: &std::collections::HashSet<FaceKey>,
    ) -> bool {
        let mut disk = self.disk_edges(vertex);
        let mut has_edge = false;
        while let Some(edge) = disk.next(self) {
            has_edge = true;
            let mut radial_len = 0usize;
            let mut radial = self.radial_loops(edge);
            while let Some(radial_loop) = radial.next(self) {
                radial_len += 1;
                if !region.contains(&self.loops.get(radial_loop).face) {
                    return false;
                }
            }
            if radial_len < 2 {
                return false;
            }
        }
        has_edge
    }

    /// §4.1.2 step 2 / §4.3 — sort the boundary edge set into the single simple
    /// cycle the union face is built on, wound as the region's source faces are.
    /// `None` when the boundary forks at a vertex, splits into more than one
    /// cycle (a disconnected region, or a union that would need a hole), or is
    /// empty (a region with no boundary at all).
    fn faces_join_chain_boundary(
        &self,
        plan: &FacesJoinPlan,
    ) -> Option<(
        smallvec::SmallVec<[VertexKey; 16]>,
        smallvec::SmallVec<[EdgeKey; 16]>,
    )> {
        let (start_edge, start_vertex) = plan.winding_seed?;

        let mut ring_verts: smallvec::SmallVec<[VertexKey; 16]> = smallvec::SmallVec::new();
        let mut ring_edges: smallvec::SmallVec<[EdgeKey; 16]> = smallvec::SmallVec::new();

        let mut edge = start_edge;
        let mut vertex = start_vertex;
        loop {
            if !self.vertex_has_exactly_two_boundary_edges(vertex, &plan.boundary) {
                return None;
            }
            ring_verts.push(vertex);
            ring_edges.push(edge);
            if ring_edges.len() > plan.boundary.len() {
                return None;
            }

            let next_vertex = self.edges.get(edge).other_vertex(vertex)?;
            if next_vertex == start_vertex {
                break;
            }
            edge = self.vertex_other_boundary_edge(next_vertex, edge, &plan.boundary)?;
            vertex = next_vertex;
        }

        // A short ring closed a *sub*-cycle: the region's boundary is not a
        // single simple cycle, so there is no n-gon to build (§4.3).
        if ring_edges.len() != plan.boundary.len() {
            return None;
        }

        Some((ring_verts, ring_edges))
    }

    fn vertex_has_exactly_two_boundary_edges(
        &self,
        vertex: VertexKey,
        boundary: &std::collections::HashSet<EdgeKey>,
    ) -> bool {
        let mut disk = self.disk_edges(vertex);
        let mut count = 0usize;
        while let Some(edge) = disk.next(self) {
            if boundary.contains(&edge) {
                count += 1;
                if count > 2 {
                    return false;
                }
            }
        }
        count == 2
    }

    fn vertex_other_boundary_edge(
        &self,
        vertex: VertexKey,
        exclude: EdgeKey,
        boundary: &std::collections::HashSet<EdgeKey>,
    ) -> Option<EdgeKey> {
        let mut disk = self.disk_edges(vertex);
        while let Some(edge) = disk.next(self) {
            if edge != exclude && boundary.contains(&edge) {
                return Some(edge);
            }
        }
        None
    }

    /// §4.1.4 / §4.1.5 — each boundary loop of the fresh union takes the per-loop
    /// customdata of its *source* loop: the one input-face loop on that boundary
    /// edge (unambiguous, since a boundary edge carries exactly one input face).
    /// The two loops may sit at either end of the shared edge, so the per-corner
    /// routing test picks the source corner that shares the union loop's vertex.
    fn faces_join_transfer_boundary_loop_attrs(
        &mut self,
        union: FaceKey,
        region: &std::collections::HashSet<FaceKey>,
    ) {
        let union_loops = collect_iter::<_, 16>(self, FaceLoops::new(self, union));
        for union_loop in union_loops {
            let edge = self.loops.get(union_loop).edge;
            if edge.is_null() {
                continue;
            }

            let mut source: Option<LoopKey> = None;
            let mut radial = self.radial_loops(edge);
            while let Some(radial_loop) = radial.next(self) {
                if region.contains(&self.loops.get(radial_loop).face) {
                    source = Some(radial_loop);
                    break;
                }
            }
            let Some(source) = source else { continue };

            let corner = if self.loops.get(source).vertex == self.loops.get(union_loop).vertex {
                source
            } else {
                self.loops.get(source).face_loop_links.next
            };
            self.loop_attribs
                .copy_into(union_loop.as_usize(), corner.as_usize());
        }
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
}

/// Which of §4.2's two calling forms the join is running under. The fork is
/// observable only when a double exists: `Resolve` *adopts* the pre-existing
/// face (and so skips §4.1.5's per-loop transfer into the doomed union), while
/// `Report` keeps the union alive and hands the overlap to the caller.
#[derive(Clone, Copy, PartialEq, Eq)]
enum CallingForm {
    Resolve,
    Report,
}

/// The §4.1.2-step-1 collection: what the join decided to build the union over,
/// and what it decided to kill. Nothing in it has been applied to the mesh yet.
#[derive(Default)]
struct FacesJoinPlan {
    /// Edges with exactly one input face on their radial (§4.1.1) — the union's
    /// perimeter.
    boundary: std::collections::HashSet<EdgeKey>,
    /// The boundary edge the union's winding is read off, and the vertex its
    /// source loop starts at.
    winding_seed: Option<(EdgeKey, VertexKey)>,
    kill_edges: smallvec::SmallVec<[EdgeKey; 8]>,
    kill_verts: smallvec::SmallVec<[VertexKey; 4]>,
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

        let vertex_count_before = mesh.vertex_count();
        let edge_count_before = mesh.edge_count();
        let face_count_before = mesh.face_count();
        let centre = verts[5];

        let merged = mesh
            .faces_join(&[faces[0], faces[1], faces[3], faces[4]])
            .expect("four-quad region joins successfully");

        assert!(
            mesh.vertices.try_get(centre).is_none(),
            "interior-only centre vert must be killed per §4.1 step 4"
        );
        assert_eq!(
            mesh.vertex_count(),
            vertex_count_before - 1,
            "1 vert killed"
        );
        assert_eq!(
            mesh.edge_count(),
            edge_count_before - 4,
            "4 interior edges killed"
        );
        assert_eq!(
            mesh.face_count(),
            face_count_before - 3,
            "4 input faces → 1 face"
        );

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

        let vertex_count_before = mesh.vertex_count();

        mesh.faces_join(&[faces[0], faces[1]])
            .expect("two-quad region joins successfully");

        assert_eq!(
            mesh.vertex_count(),
            vertex_count_before,
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
    fn build_shadow_over_two_quads(mesh: &mut LoopMesh) -> (Vec<VertexKey>, [FaceKey; 3]) {
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

        assert_eq!(mesh.vertex_count(), 6);
        assert_eq!(mesh.edge_count(), 7, "6 outer-rim + 1 interior v1-v2");
        assert_eq!(mesh.face_count(), 3);

        let survivor = mesh
            .faces_join(&[f_left, f_right])
            .expect("two-quad region joins successfully");

        assert_eq!(
            survivor, f_shadow,
            "§4.2 step 3: the pre-existing shadow face becomes the reported survivor"
        );

        assert_eq!(
            mesh.vertex_count(),
            6,
            "every perimeter vert retains a shadow edge"
        );
        assert_eq!(
            mesh.edge_count(),
            6,
            "interior v1-v2 edge gone; outer rim survives"
        );
        assert_eq!(
            mesh.face_count(),
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

    /// §9's `reusing` fork, on the side that only the **report** form reaches:
    /// a double exists, but the fresh union is handed back **alive**, so §4.1.5's
    /// per-loop transfer must still have run over it. Skipping it on the mere
    /// existence of a double would return a live face whose corners carry the
    /// attribute store's zero-fill instead of the source loops' values.
    #[test]
    fn faces_join_report_double_initialises_the_live_unions_corner_attrs() {
        let mut mesh = LoopMesh::new();
        let uv = mesh
            .register_loop_attribute(ElementAttributeDesc::numeric(
                AttributeKey::custom("uv"),
                AttrType::Vec2,
            ))
            .expect("uv layer registers");

        let (verts, [f_left, f_right, f_shadow]) = build_shadow_over_two_quads(&mut mesh);
        let vert_index = |key: VertexKey| verts.iter().position(|&k| k == key).unwrap() as f32;

        // Source corners carry (face_id, vertex_index); the shadow's own corners
        // carry a sentinel, so adopting *its* data would be visible too.
        for (face_id, face) in [(1.0_f32, f_left), (2.0_f32, f_right), (9.0_f32, f_shadow)] {
            for loop_key in mesh.face_loop_keys(face) {
                let tagged_vert = vert_index(mesh.loop_vertex(loop_key));
                mesh.loop_attribs_mut()
                    .set_vec2(uv, loop_key.as_usize(), [face_id, tagged_vert]);
            }
        }

        let outcome = mesh
            .faces_join_report_double(&[f_left, f_right])
            .expect("the two-quad region joins");

        assert_eq!(
            outcome.double,
            Some(f_shadow),
            "the pre-existing hexagon coincides with the union's corner cycle"
        );
        assert!(
            mesh.face_keys().contains(&outcome.union),
            "the report form keeps the fresh union alive — resolving it is the caller's job"
        );

        for loop_key in mesh.face_loop_keys(outcome.union) {
            let tag = mesh
                .loop_attribs()
                .get_vec2(uv, loop_key.as_usize())
                .unwrap();
            assert!(
                tag[0] == 1.0 || tag[0] == 2.0,
                "every corner of the reported union carries its source loop's data, \
                 not the store's zero-fill ({tag:?}) and not the double's sentinel"
            );
            assert_eq!(
                tag[1],
                vert_index(mesh.loop_vertex(loop_key)),
                "the transfer routed each source corner to the union corner on its own vertex"
            );
        }

        mesh.validate_topology();
    }

    /// The `reusing` side of the same fork: under the **resolving** form the
    /// pre-existing face is adopted as the merge result, and it keeps the
    /// per-corner data it already had — the join must not overwrite it with the
    /// source loops'.
    #[test]
    fn faces_join_resolving_form_leaves_the_adopted_doubles_corner_attrs_alone() {
        let mut mesh = LoopMesh::new();
        let uv = mesh
            .register_loop_attribute(ElementAttributeDesc::numeric(
                AttributeKey::custom("uv"),
                AttrType::Vec2,
            ))
            .expect("uv layer registers");

        let (_verts, [f_left, f_right, f_shadow]) = build_shadow_over_two_quads(&mut mesh);
        for (face_id, face) in [(1.0_f32, f_left), (2.0_f32, f_right), (9.0_f32, f_shadow)] {
            for loop_key in mesh.face_loop_keys(face) {
                mesh.loop_attribs_mut()
                    .set_vec2(uv, loop_key.as_usize(), [face_id, 0.0]);
            }
        }

        let survivor = mesh
            .faces_join(&[f_left, f_right])
            .expect("the two-quad region joins");
        assert_eq!(survivor, f_shadow);

        for loop_key in mesh.face_loop_keys(survivor) {
            let tag = mesh
                .loop_attribs()
                .get_vec2(uv, loop_key.as_usize())
                .unwrap();
            assert_eq!(
                tag[0], 9.0,
                "the adopted face is the merge result customdata and all — no transfer runs into it"
            );
        }

        mesh.validate_topology();
    }

    /// §4.1.5 same-direction (non-contiguous) winding: two coplanar quads
    /// share edge `{v1, v2}` but both traverse it `v1 -> v2` (inconsistent
    /// face orientations). The merge must still succeed and produce the
    /// same 6-corner n-gon as the opposite-direction case — the
    /// same-direction handling is purely an internal surgery concern, not
    /// an observable difference in the result.
    #[test]
    fn faces_join_merges_non_contiguous_winding_manifold_pair() {
        let mut mesh = LoopMesh::new();
        let uv = mesh
            .register_loop_attribute(ElementAttributeDesc::numeric(
                AttributeKey::custom("uv"),
                AttrType::Vec2,
            ))
            .expect("uv layer registers");

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
        let e_25 = mesh.create_edge([v[2], v[5]], EdgeCreateFlags::NO_DUPLICATE);
        let e_54 = mesh.create_edge([v[5], v[4]], EdgeCreateFlags::NO_DUPLICATE);
        let e_41 = mesh.create_edge([v[4], v[1]], EdgeCreateFlags::NO_DUPLICATE);

        // Left quad, CCW: [v0, v1, v2, v3]. Its loop on the shared edge is
        // anchored at v1 and walks v1 -> v2.
        let f_left = mesh.face_create(
            &[v[0], v[1], v[2], v[3]],
            &[e_01, e_12, e_23, e_30],
            FaceCreateFlags::NONE,
        );
        // Right quad with inconsistent winding: [v1, v2, v5, v4]. Its loop
        // on the shared edge is *also* anchored at v1 and walks v1 -> v2 —
        // the same-direction (non-contiguous) case of §4.1.5.
        let f_right = mesh.face_create(
            &[v[1], v[2], v[5], v[4]],
            &[e_12, e_25, e_54, e_41],
            FaceCreateFlags::NONE,
        );

        // Confirm the fixture is genuinely the same-direction case.
        let l_left = mesh.face_edge_share_loop(f_left, e_12).unwrap();
        let l_right = mesh.face_edge_share_loop(f_right, e_12).unwrap();
        assert_eq!(
            mesh.loops.get(l_left).vertex,
            mesh.loops.get(l_right).vertex,
            "fixture must be same-direction: both shared-edge loops start at the same vert"
        );

        // Tag every loop's UV with (face_id, vertex_index) so we can verify
        // each surviving boundary corner kept its own source attribute and
        // is anchored at the correct vertex after the splice.
        let vert_index = |key: VertexKey| v.iter().position(|&k| k == key).unwrap() as f32;
        for (face_id, face) in [(1.0_f32, f_left), (2.0_f32, f_right)] {
            for l in mesh.face_loop_keys(face) {
                let vi = vert_index(mesh.loop_vertex(l));
                mesh.loop_attribs_mut()
                    .set_vec2(uv, l.as_usize(), [face_id, vi]);
            }
        }

        let vertex_count_before = mesh.vertex_count();

        let merged = mesh
            .faces_join(&[f_left, f_right])
            .expect("non-contiguous coplanar pair must still merge per §4.1.5");

        // Topology: 6-corner n-gon on the outer rim; the shared interior
        // edge is killed; no verts lost (every rim vert keeps a rim edge).
        let perimeter = mesh.face_vert_keys(merged);
        assert_eq!(perimeter.len(), 6, "merged n-gon has 6 corners");
        let got: std::collections::HashSet<VertexKey> = perimeter.iter().copied().collect();
        let expected: std::collections::HashSet<VertexKey> = v.iter().copied().collect();
        assert_eq!(got, expected, "perimeter is the union's outer rim");
        assert_eq!(mesh.vertex_count(), vertex_count_before, "no verts killed");
        assert_eq!(
            mesh.edge_count(),
            6,
            "shared interior edge killed; 6 rim edges remain"
        );
        assert_eq!(mesh.face_count(), 1, "two quads merged into one");
        assert!(
            mesh.edge_keys().into_iter().all(|e| e != e_12),
            "shared edge gone"
        );

        // Per-corner attribute routing (§4.1.5): every surviving boundary
        // loop kept its own source-face UV and is anchored at the vertex
        // its (face_id, vertex_index) encodes — i.e. the splice did not
        // shuffle attributes onto the wrong corner.
        for l in mesh.face_loop_keys(merged) {
            let tag = mesh.loop_attribs().get_vec2(uv, l.as_usize()).unwrap();
            let recorded_vert = tag[1];
            assert_eq!(
                recorded_vert,
                vert_index(mesh.loop_vertex(l)),
                "loop attribute stayed pinned to its source corner"
            );
            assert!(
                tag[0] == 1.0 || tag[0] == 2.0,
                "each corner carries a recognised source-face tag"
            );
        }

        mesh.validate_topology();
    }

    /// Build the two-quad coplanar fixture with UVs tagging each corner's
    /// vertex position. `right_winding` chooses the right quad's winding so
    /// the caller can build either the opposite-direction (contiguous) or
    /// same-direction (non-contiguous) case across the shared edge `{v1,v2}`.
    /// Merges the pair and returns a `(world-position -> source-face UV)`
    /// map of the merged face's corners. §4.1.5 requires this map to be
    /// identical for both windings.
    fn merge_two_quads_corner_uv_map(
        right_winding: &[usize; 4],
    ) -> std::collections::HashMap<[u32; 2], [u32; 2]> {
        let mut mesh = LoopMesh::new();
        let uv = mesh
            .register_loop_attribute(ElementAttributeDesc::numeric(
                AttributeKey::custom("uv"),
                AttrType::Vec2,
            ))
            .expect("uv layer registers");

        let coords = [
            (0.0_f32, 0.0_f32),
            (1.0, 0.0),
            (1.0, 1.0),
            (0.0, 1.0),
            (2.0, 0.0),
            (2.0, 1.0),
        ];
        let v: Vec<VertexKey> = coords
            .iter()
            .map(|&(x, y)| mesh.create_vertex(Vec3::new(x, y, 0.0)))
            .collect();

        let mk = |mesh: &mut LoopMesh, a: usize, b: usize| {
            mesh.create_edge([v[a], v[b]], EdgeCreateFlags::NO_DUPLICATE)
        };
        let e_01 = mk(&mut mesh, 0, 1);
        let e_12 = mk(&mut mesh, 1, 2);
        let e_23 = mk(&mut mesh, 2, 3);
        let e_30 = mk(&mut mesh, 3, 0);

        let f_left = mesh.face_create(
            &[v[0], v[1], v[2], v[3]],
            &[e_01, e_12, e_23, e_30],
            FaceCreateFlags::NONE,
        );

        let rv = right_winding;
        let r_edges: Vec<EdgeKey> = (0..4)
            .map(|i| mk(&mut mesh, rv[i], rv[(i + 1) % 4]))
            .collect();
        let r_verts: Vec<VertexKey> = rv.iter().map(|&i| v[i]).collect();
        let f_right = mesh.face_create(&r_verts, &r_edges, FaceCreateFlags::NONE);

        for (face_id, face) in [(1.0_f32, f_left), (2.0_f32, f_right)] {
            for l in mesh.face_loop_keys(face) {
                let pos = mesh.vertex_position(mesh.loop_vertex(l));
                mesh.loop_attribs_mut()
                    .set_vec2(uv, l.as_usize(), [face_id, pos.x + 10.0 * pos.y]);
            }
        }

        let merged = mesh.faces_join(&[f_left, f_right]).expect("pair merges");

        let mut map = std::collections::HashMap::new();
        for l in mesh.face_loop_keys(merged) {
            let pos = mesh.vertex_position(mesh.loop_vertex(l));
            let uv_val = mesh.loop_attribs().get_vec2(uv, l.as_usize()).unwrap();
            map.insert(
                [pos.x.to_bits(), pos.y.to_bits()],
                [uv_val[0].to_bits(), uv_val[1].to_bits()],
            );
        }
        map
    }

    /// §4.1.5 invariant: the same-direction (non-contiguous) merge yields a
    /// merged face whose per-corner attributes are *identical* to the
    /// opposite-direction (contiguous) merge of the geometrically-equivalent
    /// pair. Opposite-direction right quad is `[v1, v4, v5, v2]` (shared
    /// edge walked v2 -> v1); same-direction is `[v1, v2, v5, v4]` (walked
    /// v1 -> v2, matching the left quad).
    #[test]
    fn faces_join_non_contiguous_matches_contiguous_corner_attributes() {
        let contiguous = merge_two_quads_corner_uv_map(&[1, 4, 5, 2]);
        let non_contiguous = merge_two_quads_corner_uv_map(&[1, 2, 5, 4]);
        assert_eq!(
            contiguous, non_contiguous,
            "§4.1.5: same-direction merge must produce the identical per-corner \
             attribute map as the opposite-direction merge"
        );
        // Sanity: both produced a 6-corner map.
        assert_eq!(contiguous.len(), 6, "merged hexagon has 6 distinct corners");
    }

    /// The union face must be aged on its own perimeter whichever pairwise path
    /// the merge took. The natural manifold case routes through the Euler
    /// primitive (which re-seats); the same-direction case routes through the
    /// local reverse-cycle splice, and it must re-seat too — otherwise "is the
    /// union the entry of its perimeter edges?" would depend on the winding of
    /// the input, and the entry is the dissolve family's face-age channel
    /// (`DISSOLVE_VERTS.md` §6.3's entry-reseat rule).
    #[test]
    fn faces_join_same_direction_path_ages_the_union_on_its_perimeter() {
        let mut mesh = LoopMesh::new();
        let v: Vec<VertexKey> = [
            (0.0, 0.0),
            (1.0, 0.0),
            (1.0, 1.0),
            (0.0, 1.0),
            (2.0, 0.0),
            (2.0, 1.0),
            (0.0, 2.0),
            (1.0, 2.0),
        ]
        .into_iter()
        .map(|(x, y)| mesh.create_vertex(Vec3::new(x, y, 0.0)))
        .collect();

        let mk = |mesh: &mut LoopMesh, ring: [usize; 4]| {
            let verts: Vec<VertexKey> = ring.iter().map(|&i| v[i]).collect();
            let edges: Vec<EdgeKey> = (0..4)
                .map(|k| {
                    mesh.create_edge(
                        [verts[k], verts[(k + 1) % 4]],
                        EdgeCreateFlags::NO_DUPLICATE,
                    )
                })
                .collect();
            mesh.face_create(&verts, &edges, FaceCreateFlags::NONE)
        };

        let f_left = mk(&mut mesh, [0, 1, 2, 3]);
        // Same-direction (non-contiguous) across the shared edge {v1, v2}: both
        // quads' loops on it start at v1, so the merge takes the splice path.
        let f_right = mk(&mut mesh, [1, 2, 5, 4]);
        // A face outside the join, on what will be a perimeter edge of the union.
        let f_above = mk(&mut mesh, [3, 2, 7, 6]);

        let shared = mesh.query_edge_exists(v[1], v[2]).expect("shared edge");
        let l_left = mesh.face_edge_share_loop(f_left, shared).unwrap();
        let l_right = mesh.face_edge_share_loop(f_right, shared).unwrap();
        assert_eq!(
            mesh.loop_vertex(l_left),
            mesh.loop_vertex(l_right),
            "fixture must exercise the same-direction merge path"
        );

        let outward = mesh.query_edge_exists(v[2], v[3]).expect("outward edge");
        assert_eq!(
            mesh.loop_face(mesh.edge_radial_entry(outward).unwrap()),
            f_above,
            "pristine entry is the last face attached to the outward edge"
        );

        let merged = mesh
            .faces_join(&[f_left, f_right])
            .expect("same-direction pair merges");

        assert_eq!(
            mesh.loop_face(mesh.edge_radial_entry(outward).unwrap()),
            merged,
            "the union is the newest face on the outward edge, as on the Euler path"
        );
        for l in mesh.face_loop_keys(merged) {
            let e = mesh.loop_edge(l);
            assert_eq!(
                mesh.loop_face(mesh.edge_radial_entry(e).unwrap()),
                merged,
                "every perimeter edge of the union enters on its loop"
            );
        }
        mesh.validate_topology();
    }

    /// `RADIAL_EDGE_STRUCTURE.md` §6.5's discriminating trace, at n = 2. A spine
    /// edge carries four faces attached `X, A, Y, Z` — the merged-away source `A`
    /// sits in the **middle**, which is what separates "append the union after
    /// the pre-merge entry" (the reference) from "seat the union in the dead
    /// source's slot" (the rule that does not exist). Merging `A` with `B` across
    /// a *different* edge leaves the spine on the union's perimeter, and the
    /// resulting traversal must be `U, X, Y, Z` — not `U, Y, Z, X`.
    #[test]
    fn faces_join_seats_the_union_where_an_append_would_not_in_the_dead_slot() {
        let mut mesh = LoopMesh::new();
        // The spine edge {s0, s1}, and the four faces hanging off it.
        let s0 = mesh.create_vertex(Vec3::new(0.0, 0.0, 0.0));
        let s1 = mesh.create_vertex(Vec3::new(0.0, 1.0, 0.0));

        let mut fin = |mesh: &mut LoopMesh, x: f32, z: f32| -> FaceKey {
            let a = mesh.create_vertex(Vec3::new(x, 0.0, z));
            let b = mesh.create_vertex(Vec3::new(x, 1.0, z));
            let ring = [s0, a, b, s1];
            let edges: Vec<EdgeKey> = (0..4)
                .map(|k| {
                    mesh.create_edge([ring[k], ring[(k + 1) % 4]], EdgeCreateFlags::NO_DUPLICATE)
                })
                .collect();
            mesh.face_create(&ring, &edges, FaceCreateFlags::NONE)
        };

        let face_x = fin(&mut mesh, 1.0, 1.0);
        // `A` is attached second — in the middle of the radial cycle.
        let face_a = fin(&mut mesh, 1.0, 0.0);
        let face_y = fin(&mut mesh, 1.0, 2.0);
        let face_z = fin(&mut mesh, 1.0, 3.0);

        let spine = mesh.query_edge_exists(s0, s1).expect("spine edge");
        assert_eq!(
            mesh.edge_face_keys(spine),
            vec![face_z, face_x, face_a, face_y],
            "pristine traversal from the entry: Z, X, A, Y"
        );

        // `B`: a quad glued to `A` along A's far edge (the one opposite the
        // spine), so the merge crosses an edge that is *not* the spine.
        let a_far = mesh
            .face_edge_keys(face_a)
            .into_iter()
            .find(|&e| {
                mesh.edge_face_keys(e).len() == 1
                    && mesh
                        .edge_vert_keys(e)
                        .iter()
                        .all(|&v| mesh.vertex_position(v).x == 1.0)
            })
            .expect("A's far edge");
        let [f0, f1] = mesh.edge_vert_keys(a_far);
        let b0 = mesh.create_vertex(Vec3::new(2.0, 0.0, 0.0));
        let b1 = mesh.create_vertex(Vec3::new(2.0, 1.0, 0.0));
        // Wind B so its loop on the shared edge opposes A's.
        let ring = if mesh.vertex_position(f0).y == 0.0 {
            [f1, f0, b0, b1]
        } else {
            [f0, f1, b0, b1]
        };
        let edges: Vec<EdgeKey> = (0..4)
            .map(|k| mesh.create_edge([ring[k], ring[(k + 1) % 4]], EdgeCreateFlags::NO_DUPLICATE))
            .collect();
        let face_b = mesh.face_create(&ring, &edges, FaceCreateFlags::NONE);

        let union = mesh
            .faces_join(&[face_a, face_b])
            .expect("A and B share exactly one edge");

        assert_eq!(
            mesh.edge_face_keys(spine),
            vec![union, face_x, face_y, face_z],
            "the union lands where an append lands — after the pre-merge entry Z — \
             not in A's dead slot (which would read U, Y, Z, X)"
        );
        mesh.validate_topology();
    }

    /// §9's "mint once": the union is constructed over the region's boundary, so
    /// a region-**interior** edge never takes the union's loop at all. The chord
    /// `c` here is shared by two region faces *and* two out-of-region fins, so
    /// it is a §4.1.3 **keep-alive** edge: nothing is collected for it, and the
    /// entry it carried before the merge is never disturbed. `C` — whose only
    /// interior edge is that chord — is therefore incident to no collected edge
    /// and no collected vertex, and **survives** the join (§4.1.4), keeping its
    /// loop on the chord.
    #[test]
    fn faces_join_never_seats_an_edge_interior_to_the_final_union() {
        let mut mesh = LoopMesh::new();
        // Chord {c0, c1}; quads B and C flank it, fins P and Q also use it.
        let c0 = mesh.create_vertex(Vec3::new(0.0, 0.0, 0.0));
        let c1 = mesh.create_vertex(Vec3::new(0.0, 1.0, 0.0));

        let mut quad = |mesh: &mut LoopMesh, ring: [VertexKey; 4]| -> FaceKey {
            let edges: Vec<EdgeKey> = (0..4)
                .map(|k| {
                    mesh.create_edge([ring[k], ring[(k + 1) % 4]], EdgeCreateFlags::NO_DUPLICATE)
                })
                .collect();
            mesh.face_create(&ring, &edges, FaceCreateFlags::NONE)
        };

        // B, on the -x side. Attached to the chord first.
        let b0 = mesh.create_vertex(Vec3::new(-1.0, 0.0, 0.0));
        let b1 = mesh.create_vertex(Vec3::new(-1.0, 1.0, 0.0));
        let face_b = quad(&mut mesh, [b0, c0, c1, b1]);
        // Fin P, out of the region.
        let p0 = mesh.create_vertex(Vec3::new(0.0, 0.0, 1.0));
        let p1 = mesh.create_vertex(Vec3::new(0.0, 1.0, 1.0));
        let fin_p = quad(&mut mesh, [c0, p0, p1, c1]);
        // C, on the +x side (opposite winding across the chord to B).
        let d0 = mesh.create_vertex(Vec3::new(1.0, 0.0, 0.0));
        let d1 = mesh.create_vertex(Vec3::new(1.0, 1.0, 0.0));
        let face_c = quad(&mut mesh, [c1, c0, d0, d1]);
        // Fin Q, out of the region — attached last, so it is the chord's entry.
        let q0 = mesh.create_vertex(Vec3::new(0.0, 0.0, 2.0));
        let q1 = mesh.create_vertex(Vec3::new(0.0, 1.0, 2.0));
        let fin_q = quad(&mut mesh, [c0, q0, q1, c1]);
        // A, glued to B's far edge — the third region face, so the merge is n = 3.
        let a0 = mesh.create_vertex(Vec3::new(-2.0, 0.0, 0.0));
        let a1 = mesh.create_vertex(Vec3::new(-2.0, 1.0, 0.0));
        let face_a = quad(&mut mesh, [a0, b0, b1, a1]);

        let chord = mesh.query_edge_exists(c0, c1).expect("chord");
        assert_eq!(
            mesh.edge_face_keys(chord),
            vec![fin_q, face_b, fin_p, face_c],
            "pristine traversal from the entry: Q, B, P, C"
        );

        let union = mesh
            .faces_join(&[face_a, face_b, face_c])
            .expect("A-B-C is edge-connected");

        assert_eq!(
            mesh.edge_face_keys(chord),
            vec![fin_q, fin_p, face_c],
            "the chord is interior to the union: it never takes the union's loop. \
             It merely loses B's (B cascade-died on the A-B seam); C survives, and \
             the survivors keep their order, read from the untouched entry Q"
        );
        assert!(
            mesh.face_keys().contains(&face_c),
            "C's only interior edge is the keep-alive chord, so no kill reaches it (§4.1.4)"
        );
        assert!(
            mesh.face_edge_keys(union).iter().all(|&e| e != chord),
            "the union spans the region's boundary, and the chord is not on it"
        );
        mesh.validate_topology();
    }

    /// The consequence the dissolve family's customdata flow rests on
    /// (`DISSOLVE_VERTS.md` §6.3): a *chain* of composed joins, each taking its
    /// donor from the entry face of the edge it joins across, carries the
    /// **first** join's per-face block to the end — after the first join the
    /// running union is the entry of every surviving perimeter edge, so it is
    /// its own donor from then on.
    #[test]
    fn faces_join_pair_chain_carries_the_first_donors_face_cd() {
        let mut mesh = LoopMesh::new();
        let strip: Vec<FaceKey> = {
            let bottom: Vec<VertexKey> = (0..=3)
                .map(|i| mesh.create_vertex(Vec3::new(i as f32, 0.0, 0.0)))
                .collect();
            let top: Vec<VertexKey> = (0..=3)
                .map(|i| mesh.create_vertex(Vec3::new(i as f32, 1.0, 0.0)))
                .collect();
            (0..3)
                .map(|i| {
                    let ring = [bottom[i], bottom[i + 1], top[i + 1], top[i]];
                    let edges: Vec<EdgeKey> = (0..4)
                        .map(|k| {
                            mesh.create_edge(
                                [ring[k], ring[(k + 1) % 4]],
                                EdgeCreateFlags::NO_DUPLICATE,
                            )
                        })
                        .collect();
                    mesh.face_create(&ring, &edges, FaceCreateFlags::NONE)
                })
                .collect()
        };
        let tag = mesh
            .register_face_attribute(ElementAttributeDesc::numeric(
                AttributeKey::custom("face_tag"),
                AttrType::Vec3,
            ))
            .unwrap();
        for (i, &face) in strip.iter().enumerate() {
            mesh.face_attribs_mut()
                .set_vec3(tag, face.as_usize(), [i as f32, 0.0, 0.0]);
        }
        let seam = |mesh: &LoopMesh, a: FaceKey, b: FaceKey| {
            mesh.face_edge_keys(a)
                .into_iter()
                .find(|&e| mesh.edge_face_keys(e).contains(&b))
                .expect("the two faces share a seam")
        };
        let seam_ab = seam(&mesh, strip[0], strip[1]);
        let seam_bc = seam(&mesh, strip[1], strip[2]);

        let donor = mesh.loop_face(mesh.edge_radial_entry(seam_ab).unwrap());
        assert_eq!(donor, strip[1], "pristine entry is the later-built face");
        let running = mesh
            .faces_join_pair(donor, strip[0], seam_ab)
            .expect("first seam joins");

        let donor = mesh.loop_face(mesh.edge_radial_entry(seam_bc).unwrap());
        assert_eq!(
            donor, running,
            "the running union is the newest face on the next seam"
        );
        let final_face = mesh
            .faces_join_pair(donor, strip[2], seam_bc)
            .expect("second seam joins");

        assert_eq!(mesh.face_count(), 1);
        assert_eq!(
            mesh.face_attribs()
                .get_vec3(tag, final_face.as_usize())
                .expect("tagged")[0]
                .round() as usize,
            1,
            "the chain ends on the first join's donor, not the pristine f2"
        );
        mesh.validate_topology();
    }

    /// §4.1.4's worked fixture — a row of three quads whose two interior edges
    /// are one genuinely manifold edge (`{1,4}`) and one radial-4 chord
    /// (`{2,3}`, also carrying two out-of-region fins). Returns
    /// `(verts, [f0, f1, f2, fin_a, fin_b])`; `f2` is the survivor-to-be.
    fn build_quad_row_with_nonmanifold_chord(
        mesh: &mut LoopMesh,
    ) -> (Vec<VertexKey>, [FaceKey; 5]) {
        let v: Vec<VertexKey> = [
            (0.0, 0.0, 0.0),  // 0
            (1.0, 0.0, 0.0),  // 1
            (2.0, 0.0, 0.0),  // 2
            (2.0, 1.0, 0.0),  // 3
            (1.0, 1.0, 0.0),  // 4
            (0.0, 1.0, 0.0),  // 5
            (3.0, 0.0, 0.0),  // 6
            (3.0, 1.0, 0.0),  // 7
            (2.0, 0.0, 1.0),  // 8
            (2.0, 1.0, 1.0),  // 9
            (2.0, 0.0, -1.0), // 10
            (2.0, 1.0, -1.0), // 11
        ]
        .into_iter()
        .map(|(x, y, z)| mesh.create_vertex(Vec3::new(x, y, z)))
        .collect();

        let mut quad = |mesh: &mut LoopMesh, ring: [usize; 4]| -> FaceKey {
            let verts: Vec<VertexKey> = ring.iter().map(|&i| v[i]).collect();
            mesh.face_create_verts(&verts, FaceCreateFlags::NONE)
        };

        let f0 = quad(mesh, [0, 1, 4, 5]);
        let f1 = quad(mesh, [1, 2, 3, 4]);
        let f2 = quad(mesh, [2, 6, 7, 3]);
        let fin_a = quad(mesh, [2, 3, 9, 8]);
        let fin_b = quad(mesh, [3, 2, 10, 11]);

        (v, [f0, f1, f2, fin_a, fin_b])
    }

    /// §4.1.4 — the join deletes input faces only by **cascade**, so a region
    /// face whose only interior edge is a kept-alive non-manifold chord sits on
    /// no collected edge and no collected vertex, and **survives**. The union is
    /// still built over the *whole* region's boundary, so it is laid across the
    /// survivor's territory.
    #[test]
    fn faces_join_leaves_the_face_whose_only_interior_edge_is_kept_alive() {
        let mut mesh = LoopMesh::new();
        let (v, [f0, f1, f2, fin_a, fin_b]) = build_quad_row_with_nonmanifold_chord(&mut mesh);

        assert_eq!(
            (mesh.vertex_count(), mesh.edge_count(), mesh.face_count()),
            (12, 16, 5)
        );
        let chord = mesh.query_edge_exists(v[2], v[3]).expect("chord");
        assert_eq!(mesh.edge_face_keys(chord).len(), 4, "the chord is radial-4");

        let union = mesh
            .faces_join(&[f0, f1, f2])
            .expect("the row is edge-connected and has no over-shared edge");

        // Only `{1,4}` is collected: it is the sole interior edge with no buried
        // endpoint AND a total face count of 2. The chord is kept alive.
        assert!(
            mesh.query_edge_exists(v[1], v[4]).is_none(),
            "the manifold interior edge is killed"
        );
        assert!(
            mesh.query_edge_exists(v[2], v[3]).is_some(),
            "the chord is kept alive — killing it would take the fins down too"
        );

        let live: std::collections::HashSet<FaceKey> = mesh.face_keys().into_iter().collect();
        assert!(
            !live.contains(&f0) && !live.contains(&f1),
            "both cascade-die on {{1,4}}"
        );
        assert!(
            live.contains(&f2),
            "f2 sits on no collected edge or vertex — it survives"
        );
        assert!(live.contains(&fin_a) && live.contains(&fin_b));

        assert_eq!(
            (mesh.vertex_count(), mesh.edge_count(), mesh.face_count()),
            (12, 15, 4),
            "§4.1.4: -1 F (two cascade-killed, one union minted), -1 E, 0 V"
        );

        // The union spans the whole region's boundary — including the survivor's
        // territory, which it overlaps.
        let perimeter = mesh.face_vert_keys(union);
        assert_eq!(perimeter.len(), 8, "the union is the row-spanning octagon");
        let expected: std::collections::HashSet<VertexKey> = v[0..8].iter().copied().collect();
        let got: std::collections::HashSet<VertexKey> = perimeter.iter().copied().collect();
        assert_eq!(got, expected);

        mesh.validate_topology();
    }

    /// §4.4 — the vertex channel, on the smallest fixture that isolates it. A
    /// quad and a triangle sharing **two** adjacent edges bury the pinch vertex
    /// between them: `kill_edges` is **empty**, `kill_verts = { pinch }`, and
    /// both shared edges die inside the pinch's disk, cascading into both input
    /// faces. The discriminating assertion is the **vertex count**: an edge-only
    /// join reaches the same face and edge counts but strands the pinch as a
    /// loose vertex.
    #[test]
    fn faces_join_consumes_the_pinch_vertex_buried_between_two_shared_edges() {
        let mut mesh = LoopMesh::new();
        let v: Vec<VertexKey> = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (1.0, 1.0)]
            .into_iter()
            .map(|(x, y)| mesh.create_vertex(Vec3::new(x, y, 0.0)))
            .collect();

        // Quad (0,1,2,3) and triangle (2,1,0) share {0,1} and {1,2}; vertex 1's
        // whole disk is exactly those two edges.
        let quad = mesh.face_create_verts(&[v[0], v[1], v[2], v[3]], FaceCreateFlags::NONE);
        let triangle = mesh.face_create_verts(&[v[2], v[1], v[0]], FaceCreateFlags::NONE);

        assert_eq!(
            (mesh.vertex_count(), mesh.edge_count(), mesh.face_count()),
            (4, 5, 2)
        );

        let union = mesh
            .faces_join(&[quad, triangle])
            .expect("the pinch region merges through the vertex channel");

        assert!(
            mesh.vertices.try_get(v[1]).is_none(),
            "the buried pinch vertex is collected and killed with its whole disk"
        );
        assert_eq!(
            (mesh.vertex_count(), mesh.edge_count(), mesh.face_count()),
            (3, 3, 1),
            "an edge-only join would land on 3 E / 1 F but strand the pinch: 4 V, not 3"
        );

        let perimeter = mesh.face_vert_keys(union);
        assert_eq!(perimeter.len(), 3, "the union is the triangle 0-2-3");
        assert!(!perimeter.contains(&v[1]));

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

        // The union is a freshly *constructed* face (§4.1.2 step 2) — not a
        // recycled input handle, and not some other pre-existing face either.
        let live_faces: std::collections::HashSet<FaceKey> = mesh.face_keys().into_iter().collect();
        assert!(live_faces.contains(&survivor));
        assert!(
            !faces.contains(&survivor),
            "the union is a fresh face, never one of the inputs"
        );
        assert!(
            !live_faces.contains(&faces[0]) && !live_faces.contains(&faces[1]),
            "both inputs cascade-died on the manifold interior edge"
        );

        mesh.validate_topology();
    }
}
