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
        self.create_vertex_inner(position, None)
    }

    /// Create a vertex **from an example** (`radial-mesh-guides/BMESH.md` §13):
    /// the new vertex's whole per-vertex attribute block is a *copy of the
    /// example's* rather than the layer defaults, and the copy reaches past the
    /// block — it also carries the example's cached normal and **all its header
    /// flags except the selection bit**, so the new vertex comes out unselected
    /// but otherwise flagged exactly as the example was, **tag included**.
    ///
    /// The tag riding along has teeth: an operator whose end-of-run sweep or
    /// output harvest keys on the tag must put every element it mints from an
    /// example into the tag state that sweep expects — see [`Self::set_vert_tag`].
    /// The copy is of the whole header word, so the two *scratch* tag bits
    /// ([`ElementFlags::TAG1`] / [`ElementFlags::TAG2`], which the op-flag pool
    /// docs offer as free transient bits) ride along too: an operator marking with
    /// them while minting from examples inherits its own marks on the new elements.
    ///
    /// `example` must still be **live**: the block is read through the example's
    /// slot at mint time, so a killed (or since-recycled) key names the wrong data.
    /// An operator that kills its examples must mint before the kill, or capture
    /// their blocks first.
    ///
    /// This is the initialisation, not a post-mint patch: an operator that needs
    /// the example's block (bevel mints every new vertex from the beveled corner)
    /// cannot reach the same result by defaulting first and copying after without
    /// duplicating the layer walk.
    pub fn create_vertex_from_example(&mut self, position: Vec3, example: VertexKey) -> VertexKey {
        self.create_vertex_inner(position, Some(example))
    }

    fn create_vertex_inner(&mut self, position: Vec3, example: Option<VertexKey>) -> VertexKey {
        let (eflags, normal) = match example {
            Some(source) => {
                let source_vertex = self.vertices.get(source);
                (
                    source_vertex.eflags & !ElementFlags::SELECTED,
                    source_vertex.normal,
                )
            }
            None => (ElementFlags::NONE, Vec3::ZERO),
        };

        self.flags |= LoopMeshFlags::VERTICES_INDEX_DIRTY;
        let entry = self.vertices.vacant_entry();
        let vi = VertexKey::from_usize(entry.key());
        entry.insert(Vertex {
            eflags,
            key: vi,
            index: 0,
            position,
            normal,
            disk_loop_start: None,
        });

        let key_watermark = self.vertices.key_watermark();
        self.vertex_attribs.mint_slot(vi.as_usize(), key_watermark);
        self.op_flags.mint_vert_slot(vi.as_usize(), key_watermark);

        if let Some(source) = example {
            self.vertex_attribs
                .copy_into(vi.as_usize(), source.as_usize());
        }

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
        self.edge_attribs.mint_slot(edge.as_usize(), key_watermark);
        self.op_flags.mint_edge_slot(edge.as_usize(), key_watermark);

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
        self.loop_attribs.mint_slot(loopi.as_usize(), key_watermark);
        self.op_flags
            .mint_loop_slot(loopi.as_usize(), key_watermark);

        loopi
    }

    /// Append `loopi` to `axis_edge`'s radial cycle (§6.5's radial append): the
    /// new loop is spliced in immediately **after the current entry**, and the
    /// entry moves onto it. Under the head-anchored storage (entry =
    /// `radial_link.prev`) that is exactly "insert as the head's predecessor",
    /// which is what this does — the entry needs no explicit write.
    ///
    /// Invariant: a loop with a non-null edge is linked into that edge's radial
    /// cycle. A loop is only transiently edge-null while an Euler op rewires it.
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

    /// Remove `loopi` from its edge's radial cycle (§6.5's radial removal). The
    /// survivors keep their cyclic order; only the **entry** can move, and it
    /// moves **forward** — onto the removed loop's radial successor — exactly
    /// when the removed loop *was* the entry. Observably: the traversal after
    /// the removal is the traversal before it minus the removed loop; nothing
    /// rotates.
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
        // - otherwise — splice out the loop and re-seat the head as needed.
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

            let head = self
                .edges
                .get(axis_edge)
                .radial_link
                .expect("a loop on a radial cycle implies the edge has a head");
            if head == loopi {
                // The head left. The entry (`head.prev`) is untouched by the
                // splice, so the head only has to step forward to keep naming
                // the same cycle — the derived entry is unchanged.
                self.edges.get_mut(axis_edge).radial_link = Some(nexti);
            } else if nexti == head {
                // The loop that left WAS the entry (entry == head.prev, i.e.
                // entry.next == head). §6.5's companion re-seat for a
                // head-anchored layout: step the head forward, so the derived
                // entry lands on the *old* head — the removed entry's radial
                // successor. Leaving the head put would slide the entry
                // backwards instead, which is invisible below radial length 3
                // and wrong above it. (At length 2 the head's post-splice
                // successor is itself, so the sole survivor stays both head and
                // entry.)
                let new_head = self.loops.get(head).edge_radial_links.next;
                self.edges.get_mut(axis_edge).radial_link = Some(new_head);
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
        self.loop_attribs
            .mint_slot(first_edge_loop.as_usize(), key_watermark);
        self.op_flags
            .mint_loop_slot(first_edge_loop.as_usize(), key_watermark);

        new_face_entry.insert(Face {
            eflags: ElementFlags::NONE,
            key: face,
            index: 0,
            len: verts.len() as u32,
            normal: Vec3::ZERO,
            first_edge_loop,
        });

        let key_watermark = self.faces.key_watermark();
        self.face_attribs.mint_slot(face.as_usize(), key_watermark);
        self.op_flags.mint_face_slot(face.as_usize(), key_watermark);

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
        self.face_attribs.mint_slot(key.as_usize(), key_watermark);
        self.op_flags.mint_face_slot(key.as_usize(), key_watermark);
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

    /// Graft `donor`'s entire loop cycle onto `dst`, discarding `dst`'s
    /// previous loops and freeing the now-emptied `donor` face slab entry.
    ///
    /// `dst` keeps its own key, its slab identity, and its per-face
    /// attributes — only the loops it owns change. `donor`'s loops (and the
    /// edges they sit on) are preserved and re-parented to `dst`; `donor`'s
    /// per-face attributes are dropped along with its face slab entry. Loop-
    /// level customdata rides along on the donor loops untouched, and the
    /// cached face normal of the grafted topology is adopted too.
    ///
    /// This is the handle-preserving rebuild primitive: build a replacement
    /// topology on a throwaway `donor` face, then graft it onto a long-lived
    /// `dst` handle so callers holding `dst` keep a live key naming the new
    /// topology. `dst`'s old loops must be safe to free (their radial / disk
    /// neighbours already account for the impending removal, exactly as
    /// [`Self::face_kill`] assumes).
    ///
    /// Both faces must be live and distinct.
    pub(crate) fn face_adopt_loop_cycle(&mut self, dst: FaceKey, donor: FaceKey) {
        assert_ne!(
            dst, donor,
            "face_adopt_loop_cycle: dst and donor must differ"
        );

        // Free dst's current loop cycle, leaving its face slab entry in place.
        let dst_first = self.faces.get(dst).first_edge_loop;
        let mut cur = dst_first;
        loop {
            let links = self.loops.get(cur).face_loop_links;
            self.face_loop_disconnect_edge(cur);
            self.face_loop_data_remove(cur);
            cur = links.next;
            if cur == dst_first {
                break;
            }
        }

        // Re-parent the donor's loops onto dst (their radial / edge links are
        // untouched — only the owning face changes).
        let donor_first = self.faces.get(donor).first_edge_loop;
        let mut cur = donor_first;
        loop {
            self.loops.get_mut(cur).face = dst;
            cur = self.loops.get(cur).face_loop_links.next;
            if cur == donor_first {
                break;
            }
        }

        let donor_len = self.faces.get(donor).len;
        let donor_normal = self.faces.get(donor).normal;
        let dst_face = self.faces.get_mut(dst);
        dst_face.first_edge_loop = donor_first;
        dst_face.len = donor_len;
        dst_face.normal = donor_normal;
        self.flags |= LoopMeshFlags::LOOPS_INDEX_DIRTY;

        self.remove_face_slab_entry(donor);
    }
}

#[cfg(test)]
mod radial_entry_convention_tests {
    use crate::loopmesh::*;
    use glam::Vec3;

    /// `n` triangular pages sharing one spine edge, attached in build order:
    /// page `k` is `[spine_a, spine_b, tip_k]`. The spine's radial cycle is
    /// therefore a pristine (append-only) cycle of length `n`, with pages
    /// `L₁ … Lₙ` in attachment order. Returns `(spine_edge, pages)`.
    fn build_book(mesh: &mut LoopMesh, n: usize) -> (EdgeKey, Vec<FaceKey>) {
        let spine_a = mesh.create_vertex(Vec3::new(0.0, 0.0, 0.0));
        let spine_b = mesh.create_vertex(Vec3::new(0.0, 1.0, 0.0));
        let pages: Vec<FaceKey> = (0..n)
            .map(|i| {
                let tip = mesh.create_vertex(Vec3::new(1.0 + i as f32, 0.5, 0.0));
                mesh.face_create_verts(&[spine_a, spine_b, tip], FaceCreateFlags::NONE)
            })
            .collect();
        let spine = mesh
            .query_edge_exists(spine_a, spine_b)
            .expect("every page adopted the same spine");
        (spine, pages)
    }

    /// The §6.2 traversal of `edge`'s radial cycle, read as faces.
    fn traversal(mesh: &LoopMesh, edge: EdgeKey) -> Vec<FaceKey> {
        mesh.edge_radial_iter(edge)
            .map(|l| mesh.loop_face(l))
            .collect()
    }

    /// Add one more page to an existing spine — a fresh radial append.
    fn append_page(mesh: &mut LoopMesh, spine: EdgeKey, tip_x: f32) -> FaceKey {
        let [spine_a, spine_b] = mesh.edge_vert_keys(spine);
        let tip = mesh.create_vertex(Vec3::new(tip_x, 0.5, 0.0));
        mesh.face_create_verts(&[spine_a, spine_b, tip], FaceCreateFlags::NONE)
    }

    /// §6.2 — the radial walk starts at the **entry**, which on a pristine
    /// (append-only) cycle is the *last*-attached loop: `Lₙ, L₁, …, Lₙ₋₁`.
    /// A walk seeded from the stored head would read `L₁ … Lₙ` — the same cycle
    /// rotated by one. Radial length 3 is the shortest that separates them from
    /// a mere reversal.
    #[test]
    fn radial_traversal_starts_at_the_entry_not_the_head() {
        let mut mesh = LoopMesh::new();
        let (spine, p) = build_book(&mut mesh, 3);

        assert_eq!(
            traversal(&mesh, spine),
            vec![p[2], p[0], p[1]],
            "§6.2: forward from the entry (the last-attached page), not from the head"
        );
        assert_eq!(
            mesh.loop_face(mesh.edge_radial_entry(spine).expect("non-wire spine")),
            p[2],
        );
    }

    /// §6.5 Consequence 1 — "the traversal after a removal is the traversal
    /// before it, minus the removed loop; nothing rotates." Checked at radial
    /// length 4 for every removal position, including the **entry** itself,
    /// which is the only position that moves the pointer (forward, onto the
    /// removed loop's successor). Below length 3 every candidate re-seat rule
    /// agrees, so this has to be tested up here.
    #[test]
    fn radial_removal_leaves_the_traversal_minus_the_removed_loop() {
        // Pristine traversal is L4 L1 L2 L3; each case strikes one page out.
        let cases: [(usize, [usize; 3]); 4] = [
            (0, [3, 1, 2]), // the stored head, not the entry
            (1, [3, 0, 2]), // a middle page
            (2, [3, 0, 1]), // a middle page
            (3, [0, 1, 2]), // the ENTRY — re-seats FORWARD onto L4.next = L1
        ];
        for (killed, expected) in cases {
            let mut mesh = LoopMesh::new();
            let (spine, p) = build_book(&mut mesh, 4);
            assert_eq!(traversal(&mesh, spine), vec![p[3], p[0], p[1], p[2]]);

            mesh.face_kill(p[killed]);
            mesh.validate_topology();

            assert_eq!(
                traversal(&mesh, spine),
                expected.iter().map(|&i| p[i]).collect::<Vec<_>>(),
                "removing page {killed} must delete it from the traversal, not rotate it \
                 (a backwards re-seat of the entry would rotate the whole walk)"
            );
        }
    }

    /// §6.5's removal→append→removal trace — the case that separates the
    /// **local** rule (the entry steps one place forward along the cycle) from
    /// every rule phrased in terms of attachment order. After an entry removal
    /// the forward order is no longer attachment order, permanently, so
    /// "re-seat onto the first-attached survivor" and "onto the last-attached
    /// survivor" both part company with the model here.
    #[test]
    fn radial_entry_reseat_is_local_not_attachment_order() {
        let mut mesh = LoopMesh::new();
        let (spine, p) = build_book(&mut mesh, 4);

        // Kill the entry L4: entry goes forward onto L1.
        mesh.face_kill(p[3]);
        assert_eq!(traversal(&mesh, spine), vec![p[0], p[1], p[2]]);

        // Append L5: it splices in after the entry L1 and the entry moves onto
        // it, giving the cycle L1 → L5 → L2 → L3 and the traversal L5 L2 L3 L1.
        let p4 = append_page(&mut mesh, spine, 9.0);
        mesh.validate_topology();
        assert_eq!(
            traversal(&mesh, spine),
            vec![p4, p[1], p[2], p[0]],
            "the append lands after the entry, and the forward order has now \
             diverged from attachment order for good"
        );

        // Kill the entry L5: forward onto L5.next = L2.
        mesh.face_kill(p4);
        mesh.validate_topology();
        let after = traversal(&mesh, spine);
        assert_eq!(after, vec![p[1], p[2], p[0]], "entry := removed.next = L2");
        assert_ne!(
            after,
            vec![p[0], p[1], p[2]],
            "not the first-attached survivor"
        );
        assert_ne!(
            after,
            vec![p[2], p[0], p[1]],
            "not the last-attached survivor"
        );
    }

    /// §8.7 — the edge splice consumes `e_src` in its §6.2 traversal order and
    /// appends each loop by the ordinary radial append, so `e_dst`'s post-splice
    /// entry is the loop visited **last**. This is the guide's worked example on
    /// the nose: `e_dst` reading `[F3, F2]` and `e_src` reading `[F1, F0]` splice
    /// to `e_dst` reading `[F0, F2, F3, F1]`.
    ///
    /// A snapshot taken from `e_src`'s *head* instead would visit `F0, F1` and
    /// land `[F1, F2, F3, F0]` — same faces, same incidences, different cycle
    /// and different entry.
    #[test]
    fn edge_splice_folds_src_traversal_onto_dst_entry() {
        let mut mesh = LoopMesh::new();
        let (e_dst, p) = build_book(&mut mesh, 4);

        // Peel pages 0 and 1 (in that order) onto a parallel edge, leaving
        // e_dst reading [p3, p2] and building e_src reading [p1, p0].
        let e_src = mesh.create_edge(mesh.edge_vert_keys(e_dst), EdgeCreateFlags::NONE);
        for page in [p[0], p[1]] {
            let l = mesh
                .face_edge_share_loop(page, e_dst)
                .expect("page sits on the spine");
            mesh.face_loop_disconnect_edge(l);
            mesh.face_loop_connect_edge(e_src, l);
        }
        assert_eq!(traversal(&mesh, e_dst), vec![p[3], p[2]]);
        assert_eq!(traversal(&mesh, e_src), vec![p[1], p[0]]);

        assert!(mesh.edge_splice(e_dst, e_src));
        mesh.validate_topology();

        assert_eq!(
            traversal(&mesh, e_dst),
            vec![p[0], p[2], p[3], p[1]],
            "§8.7's closed form: fold [s₀ … sₖ₋₁] over [d₀ … dₘ₋₁] one append at a time"
        );
        assert_eq!(
            mesh.loop_face(mesh.edge_radial_entry(e_dst).expect("non-wire")),
            p[0],
            "the post-splice entry is the loop visited last",
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::loopmesh::test_support::{create_face_from_verts, mesh_add_cube};

    #[test]
    fn create_vertex_disk() {
        let mut mesh = LoopMesh::new();
        let v0 = mesh.create_vertex(Vec3::new(0.0, 0.0, 0.0));
        let v1 = mesh.create_vertex(Vec3::new(0.0, 1.0, 0.0));
        let v2 = mesh.create_vertex(Vec3::new(0.0, 0.0, 1.0));
        let edge = mesh.create_edge([v0, v1], EdgeCreateFlags::NONE);
        let edge = mesh.create_edge([v0, v2], EdgeCreateFlags::NONE);
        mesh.vertex_disk_print_edges(v0);
        assert_eq!(mesh.vertex_disk_count_edges(v0), 2);
    }

    #[test]
    fn create_cube_faces() {
        let mut mesh = LoopMesh::new();
        mesh_add_cube(&mut mesh);

        for v in mesh.vertices.iter() {
            assert_eq!(mesh.vertex_disk_count_edges(v), 3);
        }
    }

    #[test]
    fn validate_topology_passes_on_empty_mesh() {
        LoopMesh::new().validate_topology();
    }

    #[test]
    fn validate_topology_passes_on_lone_vertex() {
        let mut mesh = LoopMesh::new();
        let _ = mesh.create_vertex(Vec3::ZERO);
        mesh.validate_topology();
    }

    #[test]
    fn validate_topology_passes_on_quad() {
        let mut mesh = LoopMesh::new();
        let verts = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        ];
        let _f = create_face_from_verts(&mut mesh, &verts);
        mesh.validate_topology();
    }

    #[test]
    fn validate_topology_passes_after_edge_split() {
        let mut mesh = LoopMesh::new();
        let verts = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        ];
        let f = create_face_from_verts(&mut mesh, &verts);
        // Split one edge to exercise SEMV's transient null-edge state.
        let l = mesh.face_first_loop(f);
        let e = mesh.loop_edge(l);
        let v_near = mesh.loop_vertex(l);
        mesh.edge_split(e, v_near, 0.5);
        mesh.validate_topology();
    }

    #[test]
    fn validate_topology_passes_after_face_split_and_poke() {
        let mut mesh = LoopMesh::new();
        let verts = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        ];
        let f = create_face_from_verts(&mut mesh, &verts);
        let face_verts = mesh.face_vert_keys(f);
        let v0 = face_verts[0];
        let v2 = face_verts[2];
        let (new_face, _new_edge) = mesh.face_split(f, v0, v2).unwrap();
        mesh.validate_topology();
        // Poke one of the halves.
        mesh.face_poke(new_face);
        mesh.validate_topology();
    }
}
