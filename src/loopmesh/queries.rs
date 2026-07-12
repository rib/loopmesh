use super::*;

impl LoopMesh {
    pub fn vert_keys(&self) -> Vec<VertexKey> {
        self.vertices.iter().collect()
    }

    pub fn edge_keys(&self) -> Vec<EdgeKey> {
        self.edges.iter().collect()
    }

    pub fn face_keys(&self) -> Vec<FaceKey> {
        self.faces.iter().collect()
    }

    /// All currently-allocated loops; callers normally reach loops via faces.
    pub fn loop_keys(&self) -> Vec<LoopKey> {
        self.loops.iter().collect()
    }

    /// Refill `buf` with the keys of every live vertex (clearing first).
    /// Zero allocation if `buf` already has enough capacity.
    pub fn collect_vert_keys_into(&self, buf: &mut Vec<VertexKey>) {
        buf.clear();
        let mut it = MeshVerts::default();
        while let Some(v) = it.next(self) {
            buf.push(v);
        }
    }

    pub fn collect_edge_keys_into(&self, buf: &mut Vec<EdgeKey>) {
        buf.clear();
        let mut it = MeshEdges::default();
        while let Some(e) = it.next(self) {
            buf.push(e);
        }
    }

    pub fn collect_face_keys_into(&self, buf: &mut Vec<FaceKey>) {
        buf.clear();
        let mut it = MeshFaces::default();
        while let Some(f) = it.next(self) {
            buf.push(f);
        }
    }

    pub fn collect_loop_keys_into(&self, buf: &mut Vec<LoopKey>) {
        buf.clear();
        let mut it = MeshLoops::default();
        while let Some(l) = it.next(self) {
            buf.push(l);
        }
    }

    /// Borrow the mesh-owned vert-key scratch buffer (pre-filled with the
    /// current keys) for the duration of `f`. The buf is `mem::take`-d
    /// from the mesh, so `f` gets a fresh `&mut self` for free; the buf is
    /// restored after `f` returns. **No restoration on panic** — the buf
    /// gets reset to empty next time, which is correctness-safe.
    pub fn with_vert_keys<R>(&mut self, f: impl FnOnce(&mut LoopMesh, &[VertexKey]) -> R) -> R {
        let mut buf = std::mem::take(&mut self.vert_key_scratch);
        self.collect_vert_keys_into(&mut buf);
        let r = f(self, &buf);
        self.vert_key_scratch = buf;
        r
    }

    pub fn with_edge_keys<R>(&mut self, f: impl FnOnce(&mut LoopMesh, &[EdgeKey]) -> R) -> R {
        let mut buf = std::mem::take(&mut self.edge_key_scratch);
        self.collect_edge_keys_into(&mut buf);
        let r = f(self, &buf);
        self.edge_key_scratch = buf;
        r
    }

    pub fn with_face_keys<R>(&mut self, f: impl FnOnce(&mut LoopMesh, &[FaceKey]) -> R) -> R {
        let mut buf = std::mem::take(&mut self.face_key_scratch);
        self.collect_face_keys_into(&mut buf);
        let r = f(self, &buf);
        self.face_key_scratch = buf;
        r
    }

    pub fn with_loop_keys<R>(&mut self, f: impl FnOnce(&mut LoopMesh, &[LoopKey]) -> R) -> R {
        let mut buf = std::mem::take(&mut self.loop_key_scratch);
        self.collect_loop_keys_into(&mut buf);
        let r = f(self, &buf);
        self.loop_key_scratch = buf;
        r
    }

    pub fn vertex_count(&self) -> usize {
        self.vertices.len()
    }

    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    pub fn face_count(&self) -> usize {
        self.faces.len()
    }

    pub fn loop_count(&self) -> usize {
        self.loops.len()
    }

    /// Slab-key watermark — one past the highest slab slot ever allocated
    /// for this element type. Differs from `vertex_count()` after kills: kills
    /// reduce `vertex_count` but the watermark stays high (slot recycling is
    /// the slab's own concern). Useful for sizing per-element scratch
    /// buffers indexed by `key.as_usize()`.
    pub fn vertex_slot_count(&self) -> usize {
        self.vertices.key_watermark()
    }

    pub fn edge_slot_count(&self) -> usize {
        self.edges.key_watermark()
    }

    pub fn face_slot_count(&self) -> usize {
        self.faces.key_watermark()
    }

    pub fn loop_slot_count(&self) -> usize {
        self.loops.key_watermark()
    }

    pub fn vert_edge_keys(&self, v: VertexKey) -> Vec<EdgeKey> {
        self.vertex_disk_iter(v).collect()
    }

    /// Collects each neighbour vertex (the other endpoint of each disk edge).
    pub fn vert_neighbor_keys(&self, v: VertexKey) -> Vec<VertexKey> {
        self.vertex_disk_iter(v)
            .filter_map(|e| self.edges.get(e).other_vertex(v))
            .collect()
    }

    /// Deduplicates faces reached via multiple incident edges.
    pub fn vert_face_keys(&self, v: VertexKey) -> Vec<FaceKey> {
        let mut seen: Vec<FaceKey> = Vec::new();
        for e in self.vertex_disk_iter(v) {
            for l in self.edge_radial_iter(e) {
                let f = self.loops.get(l).face;
                if !seen.contains(&f) {
                    seen.push(f);
                }
            }
        }
        seen
    }

    pub fn edge_vert_keys(&self, e: EdgeKey) -> [VertexKey; 2] {
        self.edges.get(e).vertices
    }

    pub fn edge_loop_keys(&self, e: EdgeKey) -> Vec<LoopKey> {
        self.edge_radial_iter(e).collect()
    }

    /// One entry per loop in the radial cycle (will repeat if the same
    /// face is incident to the edge twice — non-manifold).
    pub fn edge_face_keys(&self, e: EdgeKey) -> Vec<FaceKey> {
        self.edge_radial_iter(e)
            .map(|l| self.loops.get(l).face)
            .collect()
    }

    pub fn face_loop_keys(&self, f: FaceKey) -> Vec<LoopKey> {
        let start = self.faces.get(f).first_edge_loop;
        self.face_loop_iter(start).collect()
    }

    pub fn face_vert_keys(&self, f: FaceKey) -> Vec<VertexKey> {
        let start = self.faces.get(f).first_edge_loop;
        self.face_loop_iter(start)
            .map(|l| self.loops.get(l).vertex)
            .collect()
    }

    /// On a fully-constructed mesh every face loop has an edge; the
    /// transient `EdgeKey::NULL` state inside SEMV is filtered out for
    /// safety but should never be observable here.
    pub fn face_edge_keys(&self, f: FaceKey) -> Vec<EdgeKey> {
        let start = self.faces.get(f).first_edge_loop;
        self.face_loop_iter(start)
            .map(|l| self.loops.get(l).edge)
            .filter(|e| !e.is_null())
            .collect()
    }

    pub fn vertex_position(&self, v: VertexKey) -> Vec3 {
        self.vertices.get(v).position
    }

    /// Overwrite the vertex's position.
    pub fn set_vertex_position(&mut self, v: VertexKey, pos: Vec3) {
        self.vertices.get_mut(v).position = pos;
    }

    pub fn vertex_normal(&self, v: VertexKey) -> Vec3 {
        self.vertices.get(v).normal
    }

    /// Overwrite the vertex's stored normal.
    pub fn set_vertex_normal(&mut self, v: VertexKey, n: Vec3) {
        self.vertices.get_mut(v).normal = n;
    }

    /// Set or clear the vertex's persistent HIDDEN header bit. Hidden geometry
    /// is skipped by the grow direction of `region_extend` and ignored by its
    /// shrink direction (REGION_EXTEND.md §2.3 / §5.2).
    pub fn set_vert_hidden(&mut self, v: VertexKey, hidden: bool) {
        self.vertices
            .get_mut(v)
            .eflags
            .set(ElementFlags::HIDDEN, hidden);
    }

    pub fn is_vert_hidden(&self, v: VertexKey) -> bool {
        self.vertices.get(v).eflags.contains(ElementFlags::HIDDEN)
    }

    /// Read a vertex's persistent header flags (select / hide / tag) as the
    /// typed [`ElementFlags`] view. A freshly `create_vert`'d vertex carries no
    /// header flags — the value is [`ElementFlags::empty`] (CREATE_VERT.md
    /// §2.3); the persistent select bit is a post-execution concern of the
    /// caller, never set by the bare create path. Only the all-clear
    /// (`== empty`) case is a layout-independent contract: loopmesh's bit
    /// layout is not asserted to match any other backend's, so individual set
    /// bits are not portable comparands.
    pub fn vert_flags(&self, v: VertexKey) -> ElementFlags {
        self.vertices.get(v).eflags
    }

    /// Set or clear a vertex's persistent **element tag** header bit — the
    /// caller-settable tag an operator reads as input (e.g. `beautify_fill`'s
    /// `use_restrict_tag` seam restriction, BEAUTIFY_FILL.md §6.2). It is a
    /// persistent flag the caller sets *before* the op; the op only reads it.
    ///
    /// why TAG0: of the three available header tag bits
    /// ([`ElementFlags::TAG0`]..[`TAG2`]) loopmesh fixes TAG0 as the single
    /// public "element tag" convention, so a caller and the operators it drives
    /// agree on one bit; TAG1/TAG2 stay free for transient operator scratch use.
    pub fn set_vert_tag(&mut self, v: VertexKey, tagged: bool) {
        self.vertices
            .get_mut(v)
            .eflags
            .set(ElementFlags::TAG0, tagged);
    }

    /// Read a vertex's persistent element-tag header bit (see [`Self::set_vert_tag`]).
    pub fn vert_tag(&self, v: VertexKey) -> bool {
        self.vertices.get(v).eflags.contains(ElementFlags::TAG0)
    }

    /// Set or clear the edge's persistent HIDDEN header bit.
    pub fn set_edge_hidden(&mut self, e: EdgeKey, hidden: bool) {
        self.edges
            .get_mut(e)
            .eflags
            .set(ElementFlags::HIDDEN, hidden);
    }

    pub fn is_edge_hidden(&self, e: EdgeKey) -> bool {
        self.edges.get(e).eflags.contains(ElementFlags::HIDDEN)
    }

    /// Set or clear the face's persistent HIDDEN header bit.
    pub fn set_face_hidden(&mut self, f: FaceKey, hidden: bool) {
        self.faces
            .get_mut(f)
            .eflags
            .set(ElementFlags::HIDDEN, hidden);
    }

    pub fn is_face_hidden(&self, f: FaceKey) -> bool {
        self.faces.get(f).eflags.contains(ElementFlags::HIDDEN)
    }

    /// Set or clear the face's persistent SMOOTH header bit. A face carrying the
    /// bit is smooth-shaded; cleared, it is flat (EDGELOOP_FILL.md §1/§5).
    pub fn set_face_smooth(&mut self, f: FaceKey, smooth: bool) {
        self.faces
            .get_mut(f)
            .eflags
            .set(ElementFlags::SMOOTH, smooth);
    }

    pub fn is_face_smooth(&self, f: FaceKey) -> bool {
        self.faces.get(f).eflags.contains(ElementFlags::SMOOTH)
    }

    /// Read a face's persistent header flags (select / hide / smooth / tag) as
    /// the typed [`ElementFlags`] view, mirroring [`LoopMesh::vert_flags`]. A
    /// freshly created face carries no header flags — the value is
    /// [`ElementFlags::empty`]. Only the all-clear (`== empty`) case is a
    /// layout-independent contract: loopmesh's bit layout is not asserted to
    /// match any other backend's, so individual set bits are not portable
    /// comparands — read them through the typed predicates
    /// ([`LoopMesh::is_face_smooth`], …) instead.
    pub fn face_flags(&self, f: FaceKey) -> ElementFlags {
        self.faces.get(f).eflags
    }

    pub fn edge_vertices(&self, e: EdgeKey) -> [VertexKey; 2] {
        self.edges.get(e).vertices
    }

    /// Vertex/loop count of the face.
    pub fn face_len(&self, f: FaceKey) -> usize {
        self.faces.get(f).len as usize
    }

    pub fn face_normal(&self, f: FaceKey) -> Vec3 {
        self.faces.get(f).normal
    }

    /// One loop of the face's loop cycle.
    pub fn face_first_loop(&self, f: FaceKey) -> LoopKey {
        self.faces.get(f).first_edge_loop
    }

    pub fn loop_vertex(&self, l: LoopKey) -> VertexKey {
        self.loops.get(l).vertex
    }

    /// Returns the edge this loop sits on.
    ///
    /// The returned key equals [`EdgeKey::NULL`] only in two situations:
    /// 1. Transiently inside `euler_split_edge_make_vert` while the radial
    ///    cycle is being rewoven — internal to that Euler op, never observable
    ///    at op boundaries.
    /// 2. After a manual `face_loop_disconnect_edge` call on a loop that
    ///    hasn't yet been reconnected.
    ///
    /// On a fully-constructed mesh (one that passes `validate_topology`),
    /// the returned key is always a live edge.
    pub fn loop_edge(&self, l: LoopKey) -> EdgeKey {
        self.loops.get(l).edge
    }

    /// True if `l` is currently connected to an edge (`loop_edge != NULL`).
    /// Use this in code that legitimately runs during a mid-edit transient,
    /// otherwise prefer `loop_edge` directly.
    pub fn loop_has_edge(&self, l: LoopKey) -> bool {
        !self.loops.get(l).edge.is_null()
    }

    pub fn loop_face(&self, l: LoopKey) -> FaceKey {
        self.loops.get(l).face
    }

    pub fn loop_next(&self, l: LoopKey) -> LoopKey {
        self.loops.get(l).face_loop_links.next
    }

    pub fn loop_prev(&self, l: LoopKey) -> LoopKey {
        self.loops.get(l).face_loop_links.prev
    }

    pub fn loop_radial_next(&self, l: LoopKey) -> LoopKey {
        self.loops.get(l).edge_radial_links.next
    }

    pub fn loop_radial_prev(&self, l: LoopKey) -> LoopKey {
        self.loops.get(l).edge_radial_links.prev
    }

    /// `edge`'s **radial entry** — the loop every radial traversal starts at
    /// ([`RADIAL_EDGE_STRUCTURE.md`](../../../radial-mesh-guides/RADIAL_EDGE_STRUCTURE.md)
    /// §6.2), and the only radial pick an order-sensitive consumer may use: the
    /// fan's first corner (§6.5) and the start of the walk an edge-splitting
    /// rebuild must follow (§8.1 step 4) are both defined off it.
    ///
    /// The entry is *not* "the last-attached loop" — that is a reading that
    /// holds only while the cycle has never had a loop removed. It is defined
    /// by two local rules: an append splices the new loop in immediately after
    /// the entry and re-seats the entry onto it; a removal re-seats the entry
    /// **forward**, onto the removed loop's radial successor, and only when the
    /// removed loop *was* the entry (§6.5).
    ///
    /// This crate uses the head-anchored storage of §6.5's "Implementation
    /// asymmetry" box: `radial_link` names the cycle's *head* — the loop the
    /// entry's successor-chain wraps back to — and each newly appended loop is
    /// spliced in as the head's predecessor, i.e. immediately after the entry.
    /// The entry is therefore the derived quantity `radial_link.prev`, which
    /// stays valid because [`LoopMesh::face_loop_disconnect_edge`] carries the
    /// companion re-seat (`head := head.next` on an entry removal) that box
    /// mandates. Reading `radial_link` directly yields a walk rotated by one;
    /// every order-sensitive read must go through here instead.
    pub fn edge_radial_entry(&self, edge: EdgeKey) -> Option<LoopKey> {
        let head = self.edges.get(edge).radial_link?;
        Some(self.loop_radial_prev(head))
    }

    pub fn vertex_disk_print_links(&self, vertex: VertexKey) {
        let v = &self.vertices.get(vertex);
        let start = v.disk_loop_start;
        let mut cur = start;
        let mut i = 0;
        while let Some(current) = cur {
            let e = &self.edges.get(current);
            let links = e.disk_loop_links_for_vert(vertex);
            println!(
                "edge {current}: next = {}, prev = {}",
                links.next, links.prev
            );
            cur = Some(links.next);
            if cur == start {
                break;
            }
            i += 1;
        }
    }

    pub(crate) fn vertex_disk_print_edges(&mut self, hub_vertex: VertexKey) {
        if let Some(start_edge) = self.vertices.get(hub_vertex).disk_loop_start {
            let mut i = 0;
            let mut next_edge = start_edge;
            loop {
                println!("Edge {i} = {next_edge}");
                let links = self
                    .edges
                    .get(next_edge)
                    .disk_loop_links_for_vert(hub_vertex);
                next_edge = links.next;
                if next_edge == start_edge {
                    break;
                }
                i += 1;
            }
        } else {
            println!("No disk loop for vertex");
        }
    }

    /// Iterates through all the edges in the disk loop of the given `hub_vertex` to
    /// determine if the `other_vertex` is also contained in the disk loop
    pub fn vertex_disk_contains_edge(
        &mut self,
        hub_vertex: VertexKey,
        other_vertex: VertexKey,
    ) -> Option<EdgeKey> {
        for edge in self.vertex_disk_iter(hub_vertex) {
            if self.edges.get(edge).contains(other_vertex).is_some() {
                return Some(edge);
            }
        }
        None
    }

    /// Iterates through all the edges in the disk loop of the given `hub_vertex` to
    /// count the number of edges in the loop.
    pub fn vertex_disk_count_edges(&self, hub_vertex: VertexKey) -> usize {
        let mut count = 0;
        for _ in self.vertex_disk_iter(hub_vertex) {
            count += 1;
        }
        count
    }

    /// Iterates through all the edges in the disk loop of the given `hub_vertex` to
    /// count the number of edges in the loop, up to `max_count`.
    pub(crate) fn vertex_disk_count_edges_capped(
        &self,
        hub_vertex: VertexKey,
        max_count: usize,
    ) -> usize {
        let mut count = 0;
        for _ in self.vertex_disk_iter(hub_vertex) {
            count += 1;
            if count >= max_count {
                return max_count;
            }
        }
        count
    }

    /// Validates the integrity of a disk loop around the given `hub_vertex` with respect to a
    /// known edge member and an expected edge count.
    pub(crate) fn vertex_disk_validate_edge_state(
        &mut self,
        hub_vertex: VertexKey,
        member_edge: EdgeKey,
        num_edges: usize,
    ) -> Result<()> {
        if num_edges == 0 {
            return Err(anyhow!("invalid, empty, disk loop"));
        }

        if self.edges.get(member_edge).contains(hub_vertex).is_none() {
            return Err(anyhow!(
                "invalid disk loop: member edge doesn't contain hub vertex"
            ));
        }

        let found_edges = self.vertex_disk_count_edges_capped(hub_vertex, num_edges + 1);
        if found_edges != num_edges {
            return Err(anyhow!("invalid disk loop: expected {num_edges} edges, found {found_edges} (capped search at {})", num_edges + 1));
        }

        Ok(())
    }

    pub fn query_edge_exists(&self, v0: VertexKey, v1: VertexKey) -> Option<EdgeKey> {
        if v0 == v1 {
            return None;
        }

        // iterate the disk cycle for each vertex in lockstep, looking for a reciprocal
        // pointer back to the other edge, and stop searching at the end of the
        // shortest cycle

        let start0 = self.vertices.get(v0).disk_loop_start;
        let start1 = self.vertices.get(v1).disk_loop_start;

        if start0.is_none() || start1.is_none() {
            // one or the other vertex doesn't belong to any edge...
            return None;
        }
        let start0 = start0.unwrap();
        let start1 = start1.unwrap();

        let mut next0 = start0;
        let mut next1 = start1;

        loop {
            if self.edges.get(next0).contains(v1).is_some() {
                return Some(next0);
            }
            if self.edges.get(next1).contains(v0).is_some() {
                return Some(next1);
            }

            let links0 = self.edges.get(next0).disk_loop_links_for_vert(v0);
            next0 = links0.next;
            if next0 == start0 {
                break;
            }
            let links1 = self.edges.get(next1).disk_loop_links_for_vert(v1);
            next1 = links1.next;
            if next1 == start1 {
                break;
            }
        }

        None
    }

    pub fn find_edge_duplicate(&self, edge: EdgeKey) -> Option<EdgeKey> {
        let v1 = self.edges.get(edge).vertices[0];
        let v2 = self.edges.get(edge).vertices[1];

        self.vertex_disk_iter(v1)
            .find(|&e| self.edges.get(e).contains(v2).is_some())
    }

    /// Queries whether some face already exists for the given set of vertices, with
    /// either a forwards or reverse winding order.
    pub fn face_exists(&self, verts: &[VertexKey]) -> Option<FaceKey> {
        let face_len = verts.len() as u32;
        assert!(face_len >= 1);

        // Look for another edge connected to the first vertex with any
        // associated (radial) face loops...
        for e in self.vertex_disk_iter(verts[0]) {
            for l in self.edge_radial_iter(e) {
                let radial_loop = &self.loops.get(l);
                let face = radial_loop.face;
                if radial_loop.vertex != verts[0] || self.faces.get(face).len != face_len {
                    continue;
                }

                // Anchor the comparison on `l` (the loop that actually starts at
                // `verts[0]`), not on the face's canonical `first_edge_loop`: a
                // face's stored start loop is an arbitrary rotation of its cycle,
                // so anchoring there would only match queries whose `verts[0]`
                // happens to coincide with it. Walking forward/backward from `l`
                // makes the match rotation-insensitive — any rotation of the same
                // vertex loop (forward) or its reflection (reverse) is found.

                // Try matching the vertices with a forwards winding order first...

                let iter_forward = self.face_loop_iter(l).skip(1);

                let mut i = 1;
                for fl in iter_forward {
                    if self.loops.get(fl).vertex != verts[i] {
                        break;
                    }
                    i += 1;
                }
                if i as u32 == face_len {
                    return Some(face);
                }

                // Then try matching the vertices with a reverse winding order...

                let iter_backwards = self.face_loop_reverse_iter(l).skip(1);

                i = 1;
                for fl in iter_backwards {
                    if self.loops.get(fl).vertex != verts[i] {
                        break;
                    }
                    i += 1;
                }
                if i as u32 == face_len {
                    return Some(face);
                }
            }
        }

        None
    }

    /// Returns the loop in `face`'s cycle whose edge is `e`, if any.
    pub fn face_edge_share_loop(&self, face: FaceKey, e: EdgeKey) -> Option<LoopKey> {
        let first = self.faces.get(face).first_edge_loop;
        for l in self.face_loop_iter(first) {
            if self.loops.get(l).edge == e {
                return Some(l);
            }
        }
        None
    }

    /// A face incident to both `v0` and `v1`, if any. Every face on every edge
    /// of `v0`'s disk is a candidate — including the far-side faces of a
    /// radial-≥2 edge, which a walk that only looked at one loop per edge would
    /// never see. Each edge's radial walk is seeded at the edge's **entry**
    /// ([`LoopMesh::edge_radial_entry`], §6.2), so the reported face is
    /// reproducible rather than a function of which loop the stored head names.
    pub fn query_vertex_shared_face(&self, v0: VertexKey, v1: VertexKey) -> Option<FaceKey> {
        for e0 in self.vertex_disk_iter(v0) {
            for radial_loop in self.edge_radial_iter(e0) {
                let face = self.loops.get(radial_loop).face;
                if self.face_contains_vertex(face, v1) {
                    return Some(face);
                }
            }
        }

        None
    }

    fn face_contains_vertex(&self, face: FaceKey, vertex: VertexKey) -> bool {
        let first = self.faces.get(face).first_edge_loop;
        self.face_loop_iter(first)
            .any(|l| self.loops.get(l).vertex == vertex)
    }

    /// Resolve the active UV layer: the loop-domain layer UV-aware operators
    /// target by default. Prefer the canonical primary-UV semantic — a
    /// 2-component loop layer tagged with the [`AttrRole::Uv`] role; otherwise
    /// fall back to the first registered 2-component loop layer (the
    /// established "a vec2 loop layer is a UV layer" convention). `None` when
    /// the mesh carries no UV layer at all.
    pub(crate) fn active_uv_loop_layer(&self) -> Option<ElementAttributeKey> {
        let attribs = self.loop_attribs();
        // Among 2-component loop layers, prefer those tagged with the Uv role;
        // fall back to the layout convention (first Vec2 loop layer) when none
        // is tagged.
        let is_vec2 = |key| matches!(attribs.storage_of(key), Some(AttributeStorage::Vec2(_)));
        let any_tagged = attribs
            .iter_descriptors()
            .any(|(key, d)| d.role == AttrRole::Uv && is_vec2(key));
        attribs
            .iter_descriptors()
            .filter(|(key, d)| is_vec2(*key) && (!any_tagged || d.role == AttrRole::Uv))
            .map(|(key, _)| key)
            .next()
    }
}

#[cfg(test)]
mod tests {
    use glam::Vec3;

    use crate::loopmesh::*;

    /// `face_exists` must match an existing face regardless of which of its
    /// vertices the query lists first (any rotation), and regardless of traversal
    /// direction (the reflection). The match anchors on the loop that actually
    /// starts at the query's `verts[0]`, not on the face's arbitrary stored start
    /// loop — so a find-or-create cannot miss an existing face and mint a
    /// duplicate just because the query rotated the vertex list.
    #[test]
    fn face_exists_is_rotation_and_reflection_insensitive() {
        let mut mesh = LoopMesh::new();
        let a = mesh.create_vertex(Vec3::new(0.0, 0.0, 0.0));
        let b = mesh.create_vertex(Vec3::new(1.0, 0.0, 0.0));
        let c = mesh.create_vertex(Vec3::new(0.0, 1.0, 0.0));
        let face = mesh.face_create_verts(&[a, b, c], FaceCreateFlags::NO_DUPLICATE);

        // Every rotation of the forward winding finds the same face.
        for query in [[a, b, c], [b, c, a], [c, a, b]] {
            assert_eq!(
                mesh.face_exists(&query),
                Some(face),
                "forward rotation {query:?} must find the existing face"
            );
        }
        // Every rotation of the reverse winding finds it too.
        for query in [[a, c, b], [c, b, a], [b, a, c]] {
            assert_eq!(
                mesh.face_exists(&query),
                Some(face),
                "reverse rotation {query:?} must find the existing face"
            );
        }
    }

    /// The duplicate-mint guard, end to end: `face_create_verts(NO_DUPLICATE)` on
    /// a rotated vertex list must reuse the existing face, not mint a second face
    /// on the same three edges.
    #[test]
    fn face_create_no_duplicate_reuses_rotated_face() {
        let mut mesh = LoopMesh::new();
        let a = mesh.create_vertex(Vec3::new(0.0, 0.0, 0.0));
        let b = mesh.create_vertex(Vec3::new(1.0, 0.0, 0.0));
        let c = mesh.create_vertex(Vec3::new(0.0, 1.0, 0.0));
        let face = mesh.face_create_verts(&[a, b, c], FaceCreateFlags::NO_DUPLICATE);
        let faces_before = mesh.face_count();

        let again = mesh.face_create_verts(&[b, c, a], FaceCreateFlags::NO_DUPLICATE);

        assert_eq!(again, face, "a rotated query reuses the existing face");
        assert_eq!(
            mesh.face_count(),
            faces_before,
            "no duplicate face is minted"
        );
        mesh.validate_topology();
    }
}
