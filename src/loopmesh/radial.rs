use super::*;

impl LoopMesh {
    /// Debug-only structural integrity pass over the whole mesh. Panics on
    /// the first inconsistency it finds. Run after a non-trivial op to
    /// catch broken radial / disk / face cycles, dangling references, and
    /// loop ↔ edge / loop ↔ face mismatches.
    ///
    /// Not part of the parity contract — it inspects loopmesh's internal
    /// link invariants rather than reproducing any specific BMesh check.
    /// Safe to call on a freshly-constructed mesh; runs in O(V + E + F + L).
    ///
    /// Also compiled under `cfg(test)` so the in-crate unit tests still
    /// build when the harness is compiled with debug assertions off (as
    /// `cargo bench` does).
    #[cfg(any(test, debug_assertions))]
    pub fn validate_topology(&self) {
        const MAX_CYCLE: usize = 1_000_000;

        // ----- 1. Vertex disk cycles -----
        let v_keys: Vec<VertexKey> = self.vertices.iter().collect();
        for v in v_keys {
            let start = self.vertices.get(v).disk_loop_start;
            let Some(start) = start else { continue };
            // The disk_loop_start edge must contain this vertex.
            assert!(
                self.edges.get(start).contains(v).is_some(),
                "vertex {v:?}: disk_loop_start {start:?} doesn't contain it"
            );
            // Walk the cycle. Verify every edge claims `v` and the doubly-
            // linked invariant holds.
            let mut cur = start;
            let mut steps = 0;
            loop {
                let links = self.edges.get(cur).disk_loop_links_for_vert(v);
                let next = links.next;
                let next_links = self.edges.get(next).disk_loop_links_for_vert(v);
                assert_eq!(
                    next_links.prev, cur,
                    "vertex {v:?}: disk cycle broken at {cur:?} -> {next:?} (next.prev = {:?})",
                    next_links.prev
                );
                assert!(
                    self.edges.get(next).contains(v).is_some(),
                    "vertex {v:?}: disk cycle visits {next:?} which doesn't contain v"
                );
                cur = next;
                steps += 1;
                if cur == start {
                    break;
                }
                assert!(
                    steps < MAX_CYCLE,
                    "vertex {v:?}: disk cycle didn't close in {MAX_CYCLE} steps"
                );
            }
        }

        // ----- 2. Edge radial cycles -----
        let e_keys: Vec<EdgeKey> = self.edges.iter().collect();
        for e in e_keys {
            // Every edge's two endpoints must contain `e` in their disks
            // (otherwise the disk above wouldn't have seen it). Sanity-
            // check the back-pointer: the radial_link's loop must claim
            // this edge.
            let Some(radial_start) = self.edges.get(e).radial_link else {
                continue;
            };
            assert_eq!(
                self.loops.get(radial_start).edge,
                e,
                "edge {e:?}: radial_link {radial_start:?} doesn't claim this edge"
            );
            let mut cur = radial_start;
            let mut steps = 0;
            loop {
                let links = self.loops.get(cur).edge_radial_links;
                let next = links.next;
                assert_eq!(
                    self.loops.get(next).edge,
                    e,
                    "edge {e:?}: radial cycle visits {next:?} whose edge field disagrees"
                );
                let next_links = self.loops.get(next).edge_radial_links;
                assert_eq!(
                    next_links.prev, cur,
                    "edge {e:?}: radial cycle broken at {cur:?} -> {next:?}"
                );
                cur = next;
                steps += 1;
                if cur == radial_start {
                    break;
                }
                assert!(
                    steps < MAX_CYCLE,
                    "edge {e:?}: radial cycle didn't close in {MAX_CYCLE} steps"
                );
            }
        }

        // ----- 3. Face loop cycles + loop / edge / face cross-checks -----
        let f_keys: Vec<FaceKey> = self.faces.iter().collect();
        for f in f_keys {
            let face = self.faces.get(f);
            let start = face.first_edge_loop;
            let face_len = face.len as usize;
            assert!(
                face_len >= 1,
                "face {f:?}: len = 0 (face must have at least one loop)"
            );
            assert_eq!(
                self.loops.get(start).face,
                f,
                "face {f:?}: first_edge_loop {start:?}'s face field disagrees"
            );
            let mut cur = start;
            let mut count = 0;
            loop {
                let l = self.loops.get(cur);
                assert_eq!(
                    l.face, f,
                    "face {f:?}: loop {cur:?} in cycle claims face {:?}",
                    l.face
                );
                assert!(
                    !l.edge.is_null(),
                    "face {f:?}: loop {cur:?} has null edge (invariant violation outside SEMV)"
                );
                let e = l.edge;
                // The loop's edge must contain the loop's vertex.
                assert!(
                    self.edges.get(e).contains(l.vertex).is_some(),
                    "face {f:?} loop {cur:?}: edge {e:?} doesn't contain loop's vertex {:?}",
                    l.vertex
                );
                // The loop's edge must contain the next-loop's vertex too —
                // each loop covers the segment (vertex -> next.vertex).
                let next = l.face_loop_links.next;
                let next_vertex = self.loops.get(next).vertex;
                assert!(
                    self.edges.get(e).contains(next_vertex).is_some(),
                    "face {f:?} loop {cur:?}: edge {e:?} doesn't contain next-loop's vertex {next_vertex:?}"
                );
                // Doubly-linked face cycle.
                let next_links = self.loops.get(next).face_loop_links;
                assert_eq!(
                    next_links.prev, cur,
                    "face {f:?}: face_loop_links broken at {cur:?} -> {next:?}"
                );
                cur = next;
                count += 1;
                if cur == start {
                    break;
                }
                assert!(
                    count <= face_len,
                    "face {f:?}: face_loop cycle exceeds declared len {face_len}"
                );
            }
            assert_eq!(
                count, face_len,
                "face {f:?}: face_loop cycle length {count} disagrees with face.len {face_len}"
            );
        }

        // ----- 4. Cross-check: every live loop is reachable from its face -----
        // We've already validated each face's cycle visits the right number
        // of loops with the right `face` field. The complementary check —
        // "no orphan loops" — falls out as: live-loop-count must equal the
        // sum of face.len over all live faces.
        let total_loops_live = self.loops.iter().count();
        let total_face_loops: usize = self
            .faces
            .iter()
            .map(|f| self.faces.get(f).len as usize)
            .sum();
        assert_eq!(
            total_loops_live, total_face_loops,
            "{total_loops_live} live loops but face.len sum is {total_face_loops} — orphan loop(s)"
        );
    }

    /// Append an edge to the disk loop of this hub vertex
    /// like: bmesh_disk_edge_append
    pub(crate) fn vertex_disk_append_edge(&mut self, vertex: VertexKey, edge: EdgeKey) {
        let disk_loop_start = self.vertices.get(vertex).disk_loop_start;
        match disk_loop_start {
            Some(start_edge) => {
                let start_links = self.edges.get(start_edge).disk_loop_links_for_vert(vertex);
                let end_edge = start_links.prev;
                let end_links = self.edges.get(end_edge).disk_loop_links_for_vert(vertex);

                let (end_links, new_links, start_links) =
                    DoubleLinks::insert_link(end_links, end_edge, edge, start_links, start_edge);

                self.edges
                    .get_mut(end_edge)
                    .set_disk_loop_links_for_vert(vertex, end_links);
                self.edges
                    .get_mut(edge)
                    .set_disk_loop_links_for_vert(vertex, new_links);
                self.edges
                    .get_mut(start_edge)
                    .set_disk_loop_links_for_vert(vertex, start_links);
            }
            None => {
                // Vertex currently has an empty disk loop
                let links = DoubleLinks {
                    next: edge,
                    prev: edge,
                };
                self.edges
                    .get_mut(edge)
                    .set_disk_loop_links_for_vert(vertex, links);
                self.vertices.get_mut(vertex).disk_loop_start = Some(edge);
            }
        }
    }

    /// Removes an edge from the disk cycle of a vertex
    /// Panics if the edge isn't associated with the vertex!
    pub(crate) fn vertex_disk_remove_edge(&mut self, vertex: VertexKey, edge: EdgeKey) {
        let links_old = self.edges.get(edge).disk_loop_links_for_vert(vertex);

        // "Only edge in the disk" iff its own next is itself. (For a length-2
        // cycle the prev and next are both the OTHER edge, so comparing
        // prev != next would mis-identify it as length-1.)
        if links_old.next != edge {
            // If this isn't the last edge in the disk cycle...
            let edge_next = links_old.next;
            let edge_prev = links_old.prev;
            // NB: read edge_PREV's links into the prev-slot, and edge_NEXT's
            // links into the next-slot — earlier revisions of this code had
            // them swapped, which happened to work for length-2 cycles
            // (special case in remove_link) but produced stale links for
            // length-3+ disk cycles.
            let links_for_prev = self.edges.get(edge_prev).disk_loop_links_for_vert(vertex);
            let links_for_next = self.edges.get(edge_next).disk_loop_links_for_vert(vertex);

            let (new_prev_links, old_links, new_next_links) = DoubleLinks::remove_link(
                links_for_prev,
                edge_prev,
                edge,
                links_for_next,
                edge_next,
            );

            self.edges
                .get_mut(edge_prev)
                .set_disk_loop_links_for_vert(vertex, new_prev_links);
            self.edges
                .get_mut(edge)
                .set_disk_loop_links_for_vert(vertex, old_links);
            self.edges
                .get_mut(edge_next)
                .set_disk_loop_links_for_vert(vertex, new_next_links);

            if self.vertices.get(vertex).disk_loop_start == Some(edge) {
                self.vertices.get_mut(vertex).disk_loop_start = Some(edge_next);
            }
        } else {
            self.vertices.get_mut(vertex).disk_loop_start = None;
        }
    }

    pub(crate) fn vertex_disk_replace_vertex(
        &mut self,
        edge: EdgeKey,
        old: VertexKey,
        new: VertexKey,
    ) {
    }

    pub fn create_vertex(&mut self, position: Vec3) -> VertexKey {
        self.flags |= LoopMeshFlags::VERTICES_INDEX_DIRTY;
        let entry = self.vertices.vacant_entry();
        let vi = VertexKey::from_usize(entry.key());
        entry.insert(Vertex {
            eflags: ElementFlags::NONE,
            key: vi,
            index: 0,
            position,
            normal: Vec3::ZERO,
            disk_loop_start: None,
        });

        let key_watermark = self.vertices.key_watermark();
        self.vertex_attribs.grow_all(key_watermark);
        self.op_flags.grow_vert_layers(key_watermark);

        vi
    }

    fn vertex_data_remove(&mut self, vertex: VertexKey) {
        self.remove_vertex_slab_entry(vertex);
    }

    pub(crate) fn remove_vertex_slab_entry(&mut self, v: VertexKey) {
        self.flags |= LoopMeshFlags::VERTICES_INDEX_DIRTY;
        self.vertices_index.clear();
        self.vertices.remove(v);
    }

    pub(crate) fn remove_edge_slab_entry(&mut self, e: EdgeKey) {
        self.flags |= LoopMeshFlags::EDGES_INDEX_DIRTY;
        self.edges_index.clear();
        self.edges.remove(e);
    }

    pub(crate) fn remove_face_slab_entry(&mut self, f: FaceKey) {
        self.flags |= LoopMeshFlags::FACES_INDEX_DIRTY;
        self.faces_index.clear();
        self.faces.remove(f);
    }

    pub(crate) fn remove_loop_slab_entry(&mut self, l: LoopKey) {
        self.flags |= LoopMeshFlags::LOOPS_INDEX_DIRTY;
        self.loops_index.clear();
        self.loops.remove(l);
    }

    pub fn vertex_kill(&mut self, vertex: VertexKey) {
        // Kill every edge on the vert's disk, then free the vert. We can't hold
        // a disk iterator across the kills (it would borrow the mesh), and we
        // can't pre-capture the cycle's `next` either: each `edge_kill` shortens
        // the disk in place, so any edge key captured before a kill can name an
        // edge a later kill has already freed. Instead we always re-read the
        // live `disk_loop_start` and kill whatever it currently points at; each
        // `edge_kill` clears or advances it, so the disk drains to empty.
        while let Some(edge) = self.vertices.get(vertex).disk_loop_start {
            self.edge_kill(edge);
        }

        self.vertex_data_remove(vertex);
    }

    pub fn create_edge(&mut self, vertices: [VertexKey; 2], flags: EdgeCreateFlags) -> EdgeKey {
        if flags & EdgeCreateFlags::NO_DUPLICATE != EdgeCreateFlags::NONE {
            if let Some(edge) = self.query_edge_exists(vertices[0], vertices[1]) {
                return edge;
            }
        }
        self.flags |= LoopMeshFlags::EDGES_INDEX_DIRTY;

        let new_entry = self.edges.vacant_entry();
        let edge = EdgeKey(new_entry.key() as u32);

        let self_links = DoubleLinks {
            next: edge,
            prev: edge,
        };
        new_entry.insert(Edge {
            eflags: ElementFlags::NONE,
            key: edge,
            index: 0,
            vertices,
            vertex_disk_links: [self_links, self_links],
            radial_link: None,
        });

        let key_watermark = self.edges.key_watermark();
        self.edge_attribs.grow_all(key_watermark);
        self.op_flags.grow_edge_layers(key_watermark);

        self.vertex_disk_append_edge(vertices[0], edge);
        self.vertex_disk_append_edge(vertices[1], edge);

        edge
    }

    fn edge_data_remove(&mut self, edge: EdgeKey) {
        self.remove_edge_slab_entry(edge);
    }

    /// Removes an edge, and any associated faces (and loops) from the mesh
    pub fn edge_kill(&mut self, edge: EdgeKey) {
        // Can't use an iterator since that would take a shared reference to the mesh
        // and stop us removing the associated faces.
        // Kill every face on the edge's radial cycle. We can't pre-capture the
        // cycle's `next` loop: each `face_kill` splices its loop out of this
        // edge's radial cycle (and frees that loop), so a loop key captured
        // before a kill can name a freed loop, and the original `first` loop is
        // gone after the first kill — leaving the `cur == first` termination
        // unreachable. Instead we always re-read the live `radial_link` and kill
        // the face it currently points at; each `face_kill` clears or advances
        // it, so the radial cycle drains to empty.
        while let Some(loopi) = self.edges.get(edge).radial_link {
            let face = self.loops.get(loopi).face;
            self.face_kill(face);
        }

        let vertices = self.edges.get(edge).vertices;
        self.vertex_disk_remove_edge(vertices[0], edge);
        self.vertex_disk_remove_edge(vertices[1], edge);

        self.edge_data_remove(edge);
    }

    fn init_face_edge_loop_data(loopi: LoopKey, vertex: VertexKey, face: FaceKey) -> Loop {
        Loop {
            key: loopi,
            index: 0,
            vertex,
            edge: EdgeKey::NULL,
            face,
            edge_radial_links: DoubleLinks {
                next: loopi,
                prev: loopi,
            },
            face_loop_links: DoubleLinks {
                next: loopi,
                prev: loopi,
            },
        }
    }

    /// NB: The returned loop is disconnected/orphaned, in that it's not yet linked in to the radial cycle of
    /// and edge. Call edge_radial_append_loop to also connect the loop to an edge.
    pub fn create_loop(&mut self, vertex: VertexKey, face: FaceKey) -> LoopKey {
        self.flags |= LoopMeshFlags::LOOPS_INDEX_DIRTY;
        let new_entry = self.loops.vacant_entry();
        let loopi = LoopKey(new_entry.key() as u32);

        let loop_data = LoopMesh::init_face_edge_loop_data(loopi, vertex, face);
        new_entry.insert(loop_data);

        let key_watermark = self.loops.key_watermark();
        self.loop_attribs.grow_all(key_watermark);
        self.op_flags.grow_loop_layers(key_watermark);

        loopi
    }

    // like: bmesh_radial_loop_append
    //
    // Invariant: a loop with a non-null edge is linked into that edge's radial
    // cycle. A loop is only transiently edge-null while an Euler op rewires it.
    pub(crate) fn face_loop_connect_edge(&mut self, axis_edge: EdgeKey, loopi: LoopKey) {
        let cur = self.loops.get(loopi).edge;
        assert!(
            cur.is_null() || cur == axis_edge,
            "face_loop_connect_edge: loop is already on a different edge ({cur:?} vs {axis_edge:?})"
        );

        // Make sure we don't old Ref to axis edge, since we need to mutate later...
        let axis_radial_link = self.edges.get(axis_edge).radial_link;
        if let Some(starti) = axis_radial_link {
            let start_links = self.loops.get(starti).edge_radial_links;
            let endi = start_links.prev;
            let end_links = self.loops.get(endi).edge_radial_links;
            let (end_links, new_links, start_links) =
                DoubleLinks::insert_link(end_links, endi, loopi, start_links, starti);

            self.loops.get_mut(endi).edge_radial_links = end_links;
            self.loops.get_mut(loopi).edge_radial_links = new_links;
            self.loops.get_mut(starti).edge_radial_links = start_links;
        } else {
            self.edges.get_mut(axis_edge).radial_link = Some(loopi);
            self.loops.get_mut(loopi).edge_radial_links = DoubleLinks {
                next: loopi,
                prev: loopi,
            };
        }

        self.loops.get_mut(loopi).edge = axis_edge;
    }

    /// like: bmesh_radial_loop_remove
    pub(crate) fn face_loop_disconnect_edge(&mut self, loopi: LoopKey) {
        // Bind the edge value into a local first so subsequent get_mut
        // calls on the same loop don't conflict with an outstanding borrow.
        let axis_edge = self.loops.get(loopi).edge;
        if axis_edge.is_null() {
            return;
        }
        let old_links = self.loops.get(loopi).edge_radial_links;

        // Two cases:
        // - last loop on this edge's radial — set edge.radial_link = None.
        // - otherwise — splice out the loop. If edge.radial_link == loopi,
        //   redirect it to old_links.next.
        // Detect "only loop in the cycle" via `next == loopi` (a single-
        // element cycle self-points). A length-2 cycle has prev == next ==
        // the OTHER loop, so `prev != next` would incorrectly mark it as
        // length-1.
        if old_links.next != loopi {
            let previ = old_links.prev;
            let nexti = old_links.next;
            let prev_links = self.loops.get(previ).edge_radial_links;
            let next_links = self.loops.get(nexti).edge_radial_links;

            let (prev_links, old_links, next_links) =
                DoubleLinks::remove_link(prev_links, previ, loopi, next_links, nexti);

            self.loops.get_mut(previ).edge_radial_links = prev_links;
            self.loops.get_mut(loopi).edge_radial_links = old_links;
            self.loops.get_mut(nexti).edge_radial_links = next_links;

            if self.edges.get(axis_edge).radial_link == Some(loopi) {
                self.edges.get_mut(axis_edge).radial_link = Some(nexti);
            }
        } else {
            self.edges.get_mut(axis_edge).radial_link = None;
        }

        self.loops.get_mut(loopi).edge = EdgeKey::NULL;
    }

    pub(crate) fn face_loop_radial_len(&self, loopi: LoopKey) -> usize {
        let mut count = 0;
        for l in self.face_loop_iter(loopi) {
            count += 1;
        }
        count
    }

    /// Re-anchor a single face-loop from its current edge onto `new_edge`,
    /// leaving the face's loop cycle (face_loop_links) untouched. The new
    /// edge must connect the same two vertices the old edge did, so that
    /// the loop's `(vertex, next.vertex)` contract still holds.
    ///
    /// This is the primitive `extrude_region` uses to swap a selected
    /// face's perimeter from "shared with boundary" edges onto duplicate
    /// edges between the duplicate vertices. Unlike `euler_split_face_make_edge`
    /// it does not change the face's loop count — only which edge each
    /// loop is anchored to.
    pub(crate) fn edge_replace_in_face_loop(&mut self, loopi: LoopKey, new_edge: EdgeKey) {
        self.face_loop_disconnect_edge(loopi);
        self.face_loop_connect_edge(new_edge, loopi);
    }

    /// Create a face from a vertex ring, minting any missing perimeter edges
    /// itself. The edges are minted **closing-edge-first**: for verts
    /// `[v0, v1, .., v_{k-1}]` the closing edge `(v_{k-1}, v0)` is created first,
    /// then the leading edges `(v0, v1), (v1, v2), ..` in listed order. That
    /// order is load-bearing for parity: it fixes which incident edge each
    /// vertex sees *first*, i.e. its disk anchor, and therefore the
    /// construction-determined first loop of the vertex's fan (see
    /// `radial-mesh-guides/RADIAL_EDGE_STRUCTURE.md` §6.5 — a hub vertex listed
    /// first anchors on its closing spoke, not its leading one). A face builder
    /// that minted leading-edge-first would seat a different anchor and surface a
    /// different first corner on a shared spoke.
    ///
    /// Existing edges are reused (`NO_DUPLICATE`), so a shared spoke is minted
    /// once by the first face that needs it and merely re-adopted afterward.
    pub fn face_create_verts(&mut self, verts: &[VertexKey], flags: FaceCreateFlags) -> FaceKey {
        let len = verts.len();
        assert!(len >= 1);

        // Closing edge (v_{len-1}, v0) first, then the leading edges in order.
        // The per-vertex edge slot the face loop expects is edge i == (v_i, v_{i+1});
        // mint into a slot-indexed buffer so face_create still receives edges in
        // perimeter (loop) order while the *creation* order is closing-first.
        let mut edges: SmallVec<[EdgeKey; 8]> = smallvec::smallvec![EdgeKey::NULL; len];
        let closing = self.create_edge([verts[len - 1], verts[0]], EdgeCreateFlags::NO_DUPLICATE);
        edges[len - 1] = closing;
        for i in 0..(len - 1) {
            edges[i] = self.create_edge([verts[i], verts[i + 1]], EdgeCreateFlags::NO_DUPLICATE);
        }

        self.face_create(verts, &edges, flags)
    }

    pub fn face_create(
        &mut self,
        verts: &[VertexKey],
        edges: &[EdgeKey],
        flags: FaceCreateFlags,
    ) -> FaceKey {
        // For now we allow degenerate edges/faces, since these are low-level APIs and the more general
        // representation may help avoid special cases during non-atomic mesh edits
        assert!(!verts.is_empty() && edges.len() == verts.len());

        if flags & FaceCreateFlags::NO_DUPLICATE != FaceCreateFlags::NONE {
            if let Some(face) = self.face_exists(verts) {
                return face;
            }
        }
        self.flags |= LoopMeshFlags::FACES_INDEX_DIRTY | LoopMeshFlags::LOOPS_INDEX_DIRTY;

        let new_face_entry = self.faces.vacant_entry();
        let face = FaceKey(new_face_entry.key() as u32);

        // Note: we don't use self.create_loop because the vacant_entry is holding a mutable reference
        // to the slab and that stops us invoking methods that would take a second mutable reference
        // to the mesh
        let new_loop_entry = self.loops.vacant_entry();
        let first_edge_loop = LoopKey(new_loop_entry.key() as u32);
        new_loop_entry.insert(LoopMesh::init_face_edge_loop_data(
            first_edge_loop,
            verts[0],
            face,
        ));

        let key_watermark = self.loops.key_watermark();
        self.loop_attribs.grow_all(key_watermark);
        self.op_flags.grow_loop_layers(key_watermark);

        new_face_entry.insert(Face {
            eflags: ElementFlags::NONE,
            key: face,
            index: 0,
            len: verts.len() as u32,
            normal: Vec3::ZERO,
            first_edge_loop,
        });

        let key_watermark = self.faces.key_watermark();
        self.face_attribs.grow_all(key_watermark);
        self.op_flags.grow_face_layers(key_watermark);

        // Finish initializing the face edge loop
        self.face_loop_connect_edge(edges[0], first_edge_loop);

        // connect all the edges into the loop...
        let mut prev_loopi = first_edge_loop;
        for i in 1..verts.len() {
            let loopi = self.create_loop(verts[i], face);
            self.face_loop_connect_edge(edges[i], loopi);

            self.loops.get_mut(loopi).face_loop_links.prev = prev_loopi;
            self.loops.get_mut(prev_loopi).face_loop_links.next = loopi;
            prev_loopi = loopi;
        }

        // close the loop
        self.loops.get_mut(first_edge_loop).face_loop_links.prev = prev_loopi;
        self.loops.get_mut(prev_loopi).face_loop_links.next = first_edge_loop;

        face
    }

    /// Removes a Loop from the slab, without doing anything to verify/maintain
    /// the integrity of the mesh structure, such as unlinking from a face or
    /// radial loops (assumes caller is responsible for this)
    fn face_loop_data_remove(&mut self, loopi: LoopKey) {
        self.remove_loop_slab_entry(loopi);
    }

    /// Updates the `edge` to connect to a `new` vertex instead of some `existing` vertex.
    /// Updates all associated (radial) face loops to also reference the `new` vertex.
    ///
    /// Assumes that the existing vertex is currently associated with the edge
    /// Assumes that the new vertex is not currently associated with edge
    ///
    /// Beware: This function will lead to inconsistent face loops that need to be repaired!
    ///
    /// Loops that reference the existing vertex and a different edge will end up
    /// referencing the new vertex which probably doesn't belong to the edge associated
    /// with that loop. For example...
    ///
    /// ```pre
    ///      E1             E2           E3
    /// V1 ======= V2 ============== V3 ===== V4 : EDGES
    /// e1v1-----> e2v2------------> e3v3----->  : LOOPS
    ///
    /// might become....
    ///      E1         E2                E3
    /// V1 ======= V2 ===== V5       V3 ===== V4 : EDGES
    /// e1v1-----> e2v2---> e3v5-------------->  : LOOPS
    ///                      ^__ invalid loop
    ///
    /// Or:
    ///      E1             E2           E3
    /// V1 ======= V2 ============== V3 ===== V4 : EDGES
    /// <--------e1v2 <------------e2v3 <---e3v4 : LOOPS
    ///
    /// might become....
    ///      E1         E2                E3
    /// V1 ======= V2 ===== V5       V3 ===== V4 ...
    /// <--------e1v2 <---e2v5 <------------e3v4
    ///                                      ^
    ///                               invalid loop
    /// ```
    ///
    /// like: bmesh_edge_vert_swap (but swapped dst/src arguments)
    pub(crate) fn edge_replace_vertex_raw(
        &mut self,
        edge: EdgeKey,
        existing: VertexKey,
        new: VertexKey,
    ) {
        self.vertex_disk_remove_edge(existing, edge);

        let slot = if self.edges.get(edge).vertices[0] == existing {
            0
        } else {
            1
        };
        self.edges.get_mut(edge).vertices[slot] = new;
        self.vertex_disk_append_edge(new, edge);

        // Iterate the radial cycle of face loops for the edge and also make sure to replace
        // the same vertex for face loops associated with this edge...
        //
        if let Some(first) = self.edges.get(edge).radial_link {
            let mut cur = first;
            loop {
                let links = self.loops.get(cur).edge_radial_links;

                // For loops whose perimeter span (loop.vertex -> loop.next.vertex)
                // crosses `existing`, swap to `new`. We tolerate "neither side
                // matches" silently — this happens during multi-step Euler
                // operations (notably JVKE) where some loops were already
                // patched in an earlier step.
                if self.loops.get(cur).vertex == existing {
                    self.loops.get_mut(cur).vertex = new;
                } else {
                    let loop_next = self.loops.get(cur).face_loop_links.next;
                    if self.loops.get(loop_next).vertex == existing {
                        self.loops.get_mut(loop_next).vertex = new;
                    }
                }

                cur = links.next;
                if cur == first {
                    break;
                }
            }
        }
    }

    /// Allocates an empty face with the given placeholder first_edge_loop,
    /// resizing face attribute storage. Caller is responsible for patching
    /// the face fields (len, first_edge_loop, etc.) once topology is settled.
    pub(crate) fn alloc_empty_face(&mut self, placeholder_first_loop: LoopKey) -> FaceKey {
        self.flags |= LoopMeshFlags::FACES_INDEX_DIRTY;
        let entry = self.faces.vacant_entry();
        let key = FaceKey(entry.key() as u32);
        entry.insert(Face {
            eflags: ElementFlags::NONE,
            key,
            index: 0,
            len: 0,
            first_edge_loop: placeholder_first_loop,
            normal: Vec3::ZERO,
        });
        let key_watermark = self.faces.key_watermark();
        self.face_attribs.grow_all(key_watermark);
        self.op_flags.grow_face_layers(key_watermark);
        key
    }

    /// Merges the edge disks from a `src` vertex into a `dst` vertex and then kills the
    /// `src` vertex.
    ///
    /// The vertices must not belong to the same edge or be part of the same face.
    pub fn vertex_merge_disks(&mut self, dst: VertexKey, src: VertexKey) {
        debug_assert!(self.query_edge_exists(dst, src).is_none());
        debug_assert!(self.query_vertex_shared_face(dst, src).is_none());

        if src == dst {
            return;
        }

        loop {
            let loop_start = self.vertices.get(dst).disk_loop_start;
            match loop_start {
                Some(edge) => {
                    self.edge_replace_vertex_raw(edge, src, dst);
                }
                None => break,
            }
        }

        self.vertex_kill(src);
    }

    /// Checks whether merging the two vertices (via `vertex_merge_disks`) would result in
    /// a double edge.
    ///
    /// The vertices must not belong to the same edge or be part of the same face.
    ///
    pub fn vertex_merge_disks_check_for_double_edges(
        &mut self,
        dst: VertexKey,
        src: VertexKey,
    ) -> bool {
        debug_assert!(self.query_edge_exists(dst, src).is_none());
        debug_assert!(self.query_vertex_shared_face(dst, src).is_none());

        // FIXME: iterate the disk for each src/dst vertex just _once_ by tagging the 'other'
        // vertices during the iteration of the first vertex and then just check for the
        // flag when iterating the other vertices in the disk of the second vertex.

        for e_dst in self.vertex_disk_iter(dst) {
            let other_dst = self.edges.get(e_dst).other_vertex(dst);

            for e_src in self.vertex_disk_iter(src) {
                let other_src = self.edges.get(e_src).other_vertex(src);

                if other_src == other_dst {
                    return true;
                }
            }
        }

        false
    }

    pub fn face_kill(&mut self, face: FaceKey) {
        // NB: We can't use a regular iterator since that would hold a shared reference
        // to our mesh, stopping us from disconnecting and freeing the loops as we go...
        let first = self.faces.get(face).first_edge_loop;
        let mut cur = first;
        loop {
            let links = self.loops.get(cur).face_loop_links;

            self.face_loop_disconnect_edge(cur);
            self.face_loop_data_remove(cur);

            cur = links.next;
            if cur == first {
                break;
            }
        }

        self.remove_face_slab_entry(face);
    }
}
