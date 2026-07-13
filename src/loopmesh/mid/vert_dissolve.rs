use crate::loopmesh::*;

impl LoopMesh {
    /// Dissolves `v` — collapses it away by either killing it (isolated),
    /// collapsing the surrounding degree-2 chain, or merging its surrounding
    /// face fan and collapsing.
    ///
    /// **The return value is not "was the vertex removed"** (`DISSOLVE_VERTS.md`
    /// §6.7). On a fan it is the post-merge *split-target adjacency* query of
    /// §6.6, which is orthogonal to whether `v` survived: a boundary fan whose
    /// first face is an n-gon merges into one face, leaves `v` standing as a
    /// 2-valence corner, and still answers `true`; a boundary fan whose first
    /// face is a triangle reaches the same end state and answers `false`. A
    /// caller that needs to know whether `v` is gone must re-check `v`.
    ///
    /// A vertex whose shape is **unsupported** (non-manifold disk, wire/face
    /// mix, three or more boundary spokes, a fan with no interior spoke, a fan
    /// pinched by a face that uses one spoke twice, a fan whose faces are not
    /// consistently wound, or a *multi-fan* — faces at `v` that no chain of
    /// interior spokes connects) is refused before any
    /// mutation: `false`, mesh unchanged. Every *other* `false` is
    /// **destructive** — the fan has already been merged, wholly or down to the
    /// two faces hinged on the reserved keep spoke (§6.7's `R` table).
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

        // The disk-dissolve schedule (§6.7) is the manifold-fan branch, and the
        // manifold-vert predicate admits boundary spokes: every spoke is
        // boundary (radial-1) or interior (radial-2), at most two are boundary,
        // and at least one is interior (a fan with no interior spoke is a
        // bowtie — nothing bridges its faces). That covers
        //   - the fully-interior fan (all radial-2), and
        //   - the boundary fan (two boundary spokes bounding an open chain of
        //     faces), whose outcome §6.7 indexes by `R`, not by face count.
        let no_non_manifold_no_wire = radial_counts.iter().all(|&c| c == 1 || c == 2);
        let boundary_spokes = radial_counts.iter().filter(|&&c| c == 1).count();
        let has_interior_edge = radial_counts.iter().any(|&c| c == 2);
        // A radial-2 spoke whose two loops belong to the *same* face pinches the
        // fan: no join can consume it, so the merge walk would stall half-way and
        // report failure over an already-mutated mesh. Refuse it here, while the
        // mesh is still untouched.
        let has_pinched_spoke = edges
            .iter()
            .zip(&radial_counts)
            .any(|(&e, &count)| count == 2 && self.spoke_face_pair(e).is_none());
        if no_non_manifold_no_wire
            && boundary_spokes <= 2
            && has_interior_edge
            && !has_pinched_spoke
            && self.vert_fan_walk_covers_disk(v, &edges)
        {
            return self.vert_dissolve_manifold_disk(v);
        }

        // Anything else (pure-boundary fan with no shared edges, non-manifold
        // edges, wire/face mix, bowtie, multi-fan): refuse.
        false
    }

    /// §3's connectivity clause, decided by §3.1's **fan walk**: from the disk
    /// anchor's radial entry, step face-to-face across `v`'s interior spokes and
    /// require the walk to reach **every** face incident at `v` and **every**
    /// spoke of `v`'s disk.
    ///
    /// The spoke *counts* alone are connectivity-blind. A closed fan plus a lone
    /// face touching `v` at `v` only satisfies every count clause (all radial in
    /// {1, 2}, two boundary spokes, an interior spoke present, nothing pinched)
    /// and is still a multi-fan, which the disk-dissolve schedule must refuse —
    /// otherwise it merges a *disconnected* region and the "refused before any
    /// mutation" contract breaks.
    ///
    /// A fully-interior fan closes back on its first loop; a boundary fan is an
    /// open chain, so the walk runs both ways out of the start face and stops at
    /// each boundary spoke. Both are single fans, and both cover their disk.
    ///
    /// **Winding is part of the predicate, by decision.** The step across an
    /// interior spoke maps to the far face's corner at `v` assuming the spoke's
    /// two loops run in *opposite* directions — i.e. the fan is consistently
    /// wound. A fan that is not (a neighbour flipped relative to its peers) has
    /// no consistent join order for its spokes: every pairwise merge across a
    /// spoke would meet the shared edge in the same direction, which is not a
    /// join the disk-dissolve schedule can consume. The walk therefore fails to
    /// cover such a fan and `vert_dissolve` refuses it, untouched, rather than
    /// stalling half-way through a destructive merge. This is a refusal §3 does
    /// not name; it is chosen here, not emergent.
    fn vert_fan_walk_covers_disk(&self, v: VertexKey, disk: &[EdgeKey]) -> bool {
        let Some(start_loop) = disk
            .iter()
            .find_map(|&spoke| self.edge_radial_entry(spoke))
            .and_then(|entry| self.loop_at_vertex(entry, v))
        else {
            return false;
        };

        let mut walked_faces: smallvec::SmallVec<[FaceKey; 8]> = smallvec::SmallVec::new();
        let mut walked_spokes: smallvec::SmallVec<[EdgeKey; 8]> = smallvec::SmallVec::new();
        let mut record = |mesh: &Self, l: LoopKey| {
            let face = mesh.loop_face(l);
            if !walked_faces.contains(&face) {
                walked_faces.push(face);
            }
            for spoke in [mesh.loop_edge(l), mesh.loop_edge(mesh.loop_prev(l))] {
                if !walked_spokes.contains(&spoke) {
                    walked_spokes.push(spoke);
                }
            }
        };

        // Forward: cross the spoke that *enters* v in this face (the prev loop's
        // edge), as §3.1's walk does.
        let mut closed = false;
        let mut current = start_loop;
        for _ in 0..=disk.len() {
            record(self, current);
            let Some(next) = self.fan_step_across(self.loop_prev(current), v) else {
                break;
            };
            if next == start_loop {
                closed = true;
                break;
            }
            current = next;
        }

        // Open (boundary) fan: the forward sweep ran off one outline edge, so the
        // rest of the chain hangs off the start face's other spoke.
        if !closed {
            let mut current = start_loop;
            for _ in 0..=disk.len() {
                let Some(next) = self.fan_step_across(current, v) else {
                    break;
                };
                if next == start_loop {
                    break;
                }
                current = next;
                record(self, current);
            }
        }

        let mut faces_at_vert = smallvec::SmallVec::<[FaceKey; 8]>::new();
        for &spoke in disk {
            for face in self.spoke_faces(spoke) {
                if !faces_at_vert.contains(&face) {
                    faces_at_vert.push(face);
                }
            }
        }

        faces_at_vert.iter().all(|face| walked_faces.contains(face))
            && disk.iter().all(|spoke| walked_spokes.contains(spoke))
    }

    /// Cross the spoke `spoke_loop` sits on, into the face on its other side, and
    /// come back holding *that* face's corner at `v`. `None` when the spoke is a
    /// boundary (nothing on the far side), is radial-≥3, or flanks the same face
    /// twice — none of which the fan walk can step over.
    fn fan_step_across(&self, spoke_loop: LoopKey, v: VertexKey) -> Option<LoopKey> {
        let across = self.loop_radial_next(spoke_loop);
        if across == spoke_loop || self.loop_radial_next(across) != spoke_loop {
            return None;
        }
        if self.loop_face(across) == self.loop_face(spoke_loop) {
            return None;
        }
        self.loop_at_vertex(across, v)
    }

    /// The corner at `v` of `l`'s face, given a loop of that face that lies on an
    /// edge incident at `v` — `l` itself when it starts at `v`, otherwise its
    /// successor (which does, since the two span the edge).
    fn loop_at_vertex(&self, l: LoopKey, v: VertexKey) -> Option<LoopKey> {
        if self.loop_vertex(l) == v {
            return Some(l);
        }
        let next = self.loop_next(l);
        (self.loop_vertex(next) == v).then_some(next)
    }

    /// Inner: dissolve a vertex whose incident edges are all boundary
    /// (radial-1) or interior (radial-2). Caller must have verified the
    /// manifold precondition.
    ///
    /// A fan with a reservable *keep* spoke runs the helper's **reserved-spoke
    /// schedule** (`DISSOLVE_VERTS.md` §6.7): `base` (the disk anchor) and
    /// `keep` are held out of the merge scan, and what the collapse then finds
    /// at `v` decides the outcome. On a fully-interior fan the two reserved
    /// spokes cut the face ring into two arcs (the *halves*) that merge
    /// independently and are joined last — that schedule, not §6.3's single
    /// running-survivor chain, is what fixes the merged n-gon's donor for this
    /// entry point. On a **boundary** fan the spokes form an open chain rather
    /// than a ring, and the outcome is indexed by `R` — how many of the two
    /// reserved spokes are boundary spokes — never by the face count.
    ///
    /// A vertex with no keep spoke (every spoke shares a face with the anchor)
    /// takes §6.7's low-valence branch: join across the anchor, then collapse.
    fn vert_dissolve_manifold_disk(&mut self, v: VertexKey) -> bool {
        if let Some((base_spoke, keep_spoke)) = self.vert_dissolve_reserved_spokes(v) {
            return self.vert_dissolve_reserved_schedule(v, base_spoke, keep_spoke);
        }

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
                if f0 != f1 && self.faces_join_pair(f0, f1, e).is_some() {
                    merged = true;
                    break;
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
                self.faces_join_pair(f0, f1, surviving_edge);
            }
        }
        true
    }

    /// The two spokes the helper reserves (`DISSOLVE_VERTS.md` §6.7): the
    /// **base** (`v`'s disk anchor) and the **keep** — the first spoke *in disk
    /// order* forward from the anchor that shares no face with it. Disk order is
    /// attachment order, not the geometric ring order, so the keep spoke is read
    /// off the disk cycle and never inferred from the fan's shape.
    ///
    /// A boundary spoke is an ordinary candidate for either role: whether it
    /// lands in one is what §6.7's `R` counts, and `R` — not the fan's face
    /// count — is what decides the boundary fan's outcome. The one build order
    /// that reserves *both* boundary spokes dissolves the fan cleanly at any
    /// `k`, so a search that skipped radial-1 spokes could never reach it.
    ///
    /// `None` when no keep spoke exists (every spoke shares a face with the
    /// anchor — §6.7's low-valence / no-keep branch).
    fn vert_dissolve_reserved_spokes(&self, v: VertexKey) -> Option<(EdgeKey, EdgeKey)> {
        let mut disk = self.disk_edges(v);
        let base_spoke = disk.next(self)?;
        let base_faces = self.spoke_faces(base_spoke);
        while let Some(spoke) = disk.next(self) {
            let faces = self.spoke_faces(spoke);
            if !faces.is_empty() && !faces.iter().any(|face| base_faces.contains(face)) {
                return Some((base_spoke, spoke));
            }
        }
        None
    }

    /// §6.7 — the reserved-spoke schedule: merge every unreserved interior
    /// spoke's faces, then collapse `v` along the base spoke.
    ///
    /// Step 1's scan restarts from the anchor after every join and can only fire
    /// on a radial-2 spoke, so a boundary spoke is never consumed — it is still
    /// standing at `v` when the collapse runs *unless it was reserved*. That is
    /// the whole of `R`: the collapse is handed a 2-valence `v` only when both
    /// boundary spokes were reserved, and what it does with a `v` of any other
    /// valence depends on which of its two routes step 2 selects.
    ///
    /// The donor of every pairwise join is the **most recently created** of the
    /// two faces flanking the joined spoke, and that face is exactly the one the
    /// spoke's radial **entry** names (§6.3's entry-reseat rule) — so the donor
    /// is read straight off the entry. The entry of any edge is re-read *after*
    /// each surgery, never cached across one (§6.3, bound 3).
    ///
    /// The reading stays true across joins because the merge used here is the
    /// *composed* (loop-minting) join, whose constructed union enters every
    /// perimeter edge it kept on its own loop — including across *calls*, which
    /// the overlapping fans of a repeated coarsening rely on: a merged face that
    /// a later fan re-joins must still score as the newest face on its own
    /// perimeter, not as pristine.
    fn vert_dissolve_reserved_schedule(
        &mut self,
        v: VertexKey,
        base_spoke: EdgeKey,
        keep_spoke: EdgeKey,
    ) -> bool {
        loop {
            // Restart the scan from the anchor after every join, per §6.7 — the
            // join deletes the spoke it fires on, so the scan makes progress.
            let mut next_spoke = None;
            let mut disk = self.disk_edges(v);
            while let Some(spoke) = disk.next(self) {
                if spoke == base_spoke || spoke == keep_spoke {
                    continue;
                }
                if let Some(faces) = self.spoke_face_pair(spoke) {
                    next_spoke = Some((spoke, faces));
                    break;
                }
            }
            let Some((spoke, (donor, absorbed))) = next_spoke else {
                break;
            };
            if self.faces_join_pair(donor, absorbed, spoke).is_none() {
                return false;
            }
        }

        // Step 2 — the collapse's two routes are mutually exclusive and are
        // selected by one bit: is the base spoke a boundary edge? (§6.6, §6.7.)
        if self.edge_radial_len_capped(base_spoke, 2) == 1 {
            return self.vert_dissolve_collapse_plain(v, base_spoke);
        }
        self.vert_dissolve_collapse_joining(v, base_spoke)
    }

    /// §6.7 step 2, **base is a boundary edge** → the plain join-edge-kill-vert
    /// route (§6.6). Its precondition *is* a 2-valence `v`, so it fires only
    /// when both boundary spokes were reserved (`R = 2`) and refuses otherwise —
    /// leaving the fan merged down to the two faces hinged on the interior keep
    /// spoke, `v` standing at valence 3, and a `false` that a caller must not
    /// read as "nothing happened".
    fn vert_dissolve_collapse_plain(&mut self, v: VertexKey, base_spoke: EdgeKey) -> bool {
        let Some(surviving_edge) = self.vert_collapse_edge(base_spoke, v) else {
            return false;
        };
        if self.edges.try_get(surviving_edge).is_none() {
            return true;
        }

        // Step 3's join. On a boundary fan the surviving edge carries a single
        // face and this does not fire; on an interior fan it is the join that
        // re-merges the two halves, and the surviving edge's entry names the
        // half that finished merging last.
        if let Some((donor, absorbed)) = self.spoke_face_pair(surviving_edge) {
            self.faces_join_pair(donor, absorbed, surviving_edge);
        }
        true
    }

    /// §6.7 step 2, **base is interior** → the face-joining route (§6.6): merge
    /// *every* face still incident at `v` into one, then re-split the union
    /// between the far endpoints of `base` and of `base`'s disk successor.
    ///
    /// The merge has no valence precondition. It swallows the whole fan, and it
    /// orphans `v` only if **every** spoke of `v` was interior to the union — so
    /// on an interior fan `v` dies, while on a boundary fan its unjoined
    /// boundary spokes sit on the union's outline and `v` **survives** as a
    /// 2-valence corner. Either way the fan is now one face, and that is
    /// unconditional: nothing below can change it.
    ///
    /// What the re-split decides is therefore only the **return value**. It is an
    /// SFME, whose two target corners must not already be adjacent on the union's
    /// loop cycle; when they are, the split refuses and so does this route — over
    /// a mesh it has already merged. Both answers reach the same end state (the
    /// split-then-rejoin is a donor-neutral round trip), which is why the boolean
    /// says nothing about whether `v` is gone.
    fn vert_dissolve_collapse_joining(&mut self, v: VertexKey, base_spoke: EdgeKey) -> bool {
        // Both split targets come off the pre-merge disk — the merge destroys it.
        let Some(v_target_base) = self.edges.get(base_spoke).other_vertex(v) else {
            return false;
        };
        let Some(successor_spoke) = self.vert_disk_successor(v, base_spoke) else {
            return false;
        };
        let Some(v_target_next) = self.edges.get(successor_spoke).other_vertex(v) else {
            return false;
        };

        // The n-input merge's donor is its first face, and the vert's face
        // iteration is seeded from the disk anchor — so the donor reads as the
        // radial entry face of the base spoke, which after step 1's scan is the
        // scan survivor sitting on it, not an original fan face (§6.7).
        let Some(entry_loop) = self.edge_radial_entry(base_spoke) else {
            return false;
        };
        let mut fan_faces: smallvec::SmallVec<[FaceKey; 8]> =
            smallvec::SmallVec::from_slice(&[self.loop_face(entry_loop)]);
        let mut disk = self.disk_edges(v);
        while let Some(spoke) = disk.next(self) {
            let mut radial = self.radial_loops(spoke);
            while let Some(l) = radial.next(self) {
                let face = self.loops.get(l).face;
                if !fan_faces.contains(&face) {
                    fan_faces.push(face);
                }
            }
        }
        let Some(merged) = self.faces_join_ex(&fan_faces, FacesJoinDelete::Yes) else {
            return false;
        };

        // SFME enforces the non-adjacency precondition itself (§8.3); its
        // refusal is this route's `false`.
        let Some((_fragment, diagonal)) = self.face_split(merged, v_target_base, v_target_next)
        else {
            return false;
        };
        if let Some((donor, absorbed)) = self.spoke_face_pair(diagonal) {
            self.faces_join_pair(donor, absorbed, diagonal);
        }
        true
    }

    /// The spoke immediately after `spoke` in `v`'s disk cycle.
    fn vert_disk_successor(&self, v: VertexKey, spoke: EdgeKey) -> Option<EdgeKey> {
        let disk: smallvec::SmallVec<[EdgeKey; 8]> = collect_iter::<_, 8>(self, self.disk_edges(v));
        let at = disk.iter().position(|&e| e == spoke)?;
        (disk.len() > 1).then(|| disk[(at + 1) % disk.len()])
    }

    /// The distinct faces flanking `spoke` — none (wire), one (boundary), or
    /// two (interior). A radial-2 spoke both of whose loops belong to the same
    /// face yields that face once.
    fn spoke_faces(&self, spoke: EdgeKey) -> smallvec::SmallVec<[FaceKey; 2]> {
        let mut faces: smallvec::SmallVec<[FaceKey; 2]> = smallvec::SmallVec::new();
        let mut radial = self.radial_loops(spoke);
        while let Some(l) = radial.next(self) {
            let face = self.loops.get(l).face;
            if !faces.contains(&face) {
                faces.push(face);
            }
        }
        faces
    }

    /// `spoke`'s radial length, stopped at `cap`: wire (0), boundary (1),
    /// interior (2) without walking a non-manifold fan out.
    fn edge_radial_len_capped(&self, spoke: EdgeKey, cap: usize) -> usize {
        let mut radial = self.radial_loops(spoke);
        let mut count = 0usize;
        while count < cap && radial.next(self).is_some() {
            count += 1;
        }
        count
    }

    /// The two distinct faces flanking a radial-2 `spoke`, the edge's radial
    /// **entry** face first — i.e. `(donor, absorbed)` for a join across it,
    /// since the entry is the most recently created face on the spoke (§6.3).
    /// `None` for a wire, boundary, non-manifold, or self-flanked spoke.
    fn spoke_face_pair(&self, spoke: EdgeKey) -> Option<(FaceKey, FaceKey)> {
        let mut radial = self.radial_loops(spoke);
        let entry_loop = radial.next(self)?;
        let partner_loop = radial.next(self)?;
        if radial.next(self).is_some() {
            return None;
        }
        let entry_face = self.loops.get(entry_loop).face;
        let partner_face = self.loops.get(partner_loop).face;
        if entry_face == partner_face {
            return None;
        }
        Some((entry_face, partner_face))
    }
}

#[cfg(test)]
mod tests {
    use glam::Vec3;

    use crate::loopmesh::*;

    /// Build a face from an ordered vertex ring under the perimeter contract of
    /// `DISSOLVE_VERTS.md` §3.1: the **closing** edge `v(n-1)–v0` is created
    /// first, then the opening edges in ascending ring order. That order is what
    /// fixes each vertex's disk anchor, and every §6.7 donor trace is stated
    /// against it.
    fn face_from_ring(mesh: &mut LoopMesh, ring: &[VertexKey]) -> FaceKey {
        let n = ring.len();
        let closing = mesh.create_edge([ring[n - 1], ring[0]], EdgeCreateFlags::NO_DUPLICATE);
        let mut edges: Vec<EdgeKey> = Vec::with_capacity(n);
        for i in 0..n - 1 {
            edges.push(mesh.create_edge([ring[i], ring[i + 1]], EdgeCreateFlags::NO_DUPLICATE));
        }
        edges.push(closing);
        mesh.face_create(ring, &edges, FaceCreateFlags::NONE)
    }

    /// A closed `n`-triangle fan: apex `0`, rim `1 ..= n`, faces built in ring
    /// order `f_i = [apex, rim_i, rim_{i+1}]` (§6.7's `disk_fan(n)`).
    fn build_disk_fan(mesh: &mut LoopMesh, n: usize) -> (VertexKey, Vec<FaceKey>) {
        let apex = mesh.create_vertex(Vec3::ZERO);
        let rim: Vec<VertexKey> = (0..n)
            .map(|i| {
                let theta = std::f32::consts::TAU * (i as f32) / (n as f32);
                mesh.create_vertex(Vec3::new(theta.cos(), theta.sin(), 0.0))
            })
            .collect();
        let faces = (0..n)
            .map(|i| face_from_ring(mesh, &[apex, rim[i], rim[(i + 1) % n]]))
            .collect();
        (apex, faces)
    }

    /// The 2x2 quad grid of §6.7's first worked example: verts `0..9` row-major
    /// on a 3x3 lattice, faces `f0..f3` in row-major order, centre vert `4`.
    fn build_grid_3x3(mesh: &mut LoopMesh) -> (Vec<VertexKey>, Vec<FaceKey>) {
        let verts: Vec<VertexKey> = (0..3)
            .flat_map(|row| (0..3).map(move |col| (row, col)))
            .map(|(row, col)| mesh.create_vertex(Vec3::new(col as f32, row as f32, 0.0)))
            .collect();
        let mut faces = Vec::new();
        for row in 0..2 {
            for col in 0..2 {
                let ring = [
                    verts[row * 3 + col],
                    verts[row * 3 + col + 1],
                    verts[(row + 1) * 3 + col + 1],
                    verts[(row + 1) * 3 + col],
                ];
                faces.push(face_from_ring(mesh, &ring));
            }
        }
        (verts, faces)
    }

    /// A 3x3 quad grid on a 4x4 lattice: verts `0..15` row-major, faces
    /// `f0..f8` row-major. The interior verts 5 and 10 have overlapping fans
    /// (they share the centre quad `f4`), which is the shape a repeated
    /// coarsening walks (`UNSUBDIVIDE.md` §5.1).
    fn build_grid_4x4(mesh: &mut LoopMesh) -> (Vec<VertexKey>, Vec<FaceKey>) {
        let verts: Vec<VertexKey> = (0..4)
            .flat_map(|row| (0..4).map(move |col| (row, col)))
            .map(|(row, col)| mesh.create_vertex(Vec3::new(col as f32, row as f32, 0.0)))
            .collect();
        let mut faces = Vec::new();
        for row in 0..3 {
            for col in 0..3 {
                let ring = [
                    verts[row * 4 + col],
                    verts[row * 4 + col + 1],
                    verts[(row + 1) * 4 + col + 1],
                    verts[(row + 1) * 4 + col],
                ];
                faces.push(face_from_ring(mesh, &ring));
            }
        }
        (verts, faces)
    }

    /// The rim of an open `k`-face fan: apex plus rim verts `r0 ..= rk`, laid
    /// out on a half-circle. No faces yet — each `open_*_fan` fixture below adds
    /// them in the build order its `R` arm needs.
    fn build_open_fan_rim(mesh: &mut LoopMesh, k: usize) -> (VertexKey, Vec<VertexKey>) {
        let apex = mesh.create_vertex(Vec3::ZERO);
        let rim = (0..=k)
            .map(|i| {
                let theta = std::f32::consts::PI * (i as f32) / (k as f32);
                mesh.create_vertex(Vec3::new(theta.cos(), theta.sin(), 0.0))
            })
            .collect();
        (apex, rim)
    }

    /// §6.7's `open_disk_fan(k)`: an open `k`-triangle fan, faces
    /// `f_i = [apex, r_i, r_{i+1}]` built in index order. The apex is each ring's
    /// **seam**, so its anchor is `f0`'s closing edge `{r1, apex}` — an
    /// **interior** spoke. Faces come back in build order.
    fn build_open_disk_fan(mesh: &mut LoopMesh, k: usize) -> (VertexKey, Vec<FaceKey>) {
        let (apex, rim) = build_open_fan_rim(mesh, k);
        let faces = (0..k)
            .map(|i| face_from_ring(mesh, &[apex, rim[i], rim[i + 1]]))
            .collect();
        (apex, faces)
    }

    /// §6.7's `open_disk_fan_boundary_anchor(k)`: the same half-disc, but `f0` is
    /// *listed* `[r0, r1, apex]` — same winding, apex merely last in the ring, so
    /// by §3.1's last row the apex's anchor becomes `f0`'s closing edge
    /// `{apex, r0}`: the fan's **outermost**, boundary spoke. Faces in build order.
    fn build_open_disk_fan_boundary_anchor(
        mesh: &mut LoopMesh,
        k: usize,
    ) -> (VertexKey, Vec<FaceKey>) {
        let (apex, rim) = build_open_fan_rim(mesh, k);
        let mut faces = vec![face_from_ring(mesh, &[rim[0], rim[1], apex])];
        for i in 1..k {
            faces.push(face_from_ring(mesh, &[apex, rim[i], rim[i + 1]]));
        }
        (apex, faces)
    }

    /// §6.7's `open_disk_fan_end_faces_first(k)`: the near end face apex-last (so
    /// the anchor is the near boundary spoke), then the **far** end face — whose
    /// closing edge is the far boundary spoke — then the middle faces. The keep
    /// search therefore reaches the far boundary spoke first: `R = 2`. Faces come
    /// back in build order (`f0` near end, `f1` far end, `f2 …` the middles
    /// inwards from the near end).
    fn build_open_disk_fan_end_faces_first(
        mesh: &mut LoopMesh,
        k: usize,
    ) -> (VertexKey, Vec<FaceKey>) {
        let (apex, rim) = build_open_fan_rim(mesh, k);
        let mut faces = vec![
            face_from_ring(mesh, &[rim[0], rim[1], apex]),
            face_from_ring(mesh, &[apex, rim[k - 1], rim[k]]),
        ];
        for i in 1..k - 1 {
            faces.push(face_from_ring(mesh, &[apex, rim[i], rim[i + 1]]));
        }
        (apex, faces)
    }

    /// §6.7's `open_quad_fan(k)`: `k` quads round an apex,
    /// `f_i = [apex, r_i, o_i, r_{i+1}]`. The apex's anchor is again `f0`'s
    /// closing edge (interior) — but `f0` is an **n-gon**, so its other verts
    /// separate the two split targets on the merged outline. Faces in build order.
    fn build_open_quad_fan(mesh: &mut LoopMesh, k: usize) -> (VertexKey, Vec<FaceKey>) {
        let (apex, rim) = build_open_fan_rim(mesh, k);
        let outer: Vec<VertexKey> = (0..k)
            .map(|i| {
                let theta = std::f32::consts::PI * (i as f32 + 0.5) / (k as f32);
                mesh.create_vertex(Vec3::new(2.0 * theta.cos(), 2.0 * theta.sin(), 0.0))
            })
            .collect();
        let faces = (0..k)
            .map(|i| face_from_ring(mesh, &[apex, rim[i], outer[i], rim[i + 1]]))
            .collect();
        (apex, faces)
    }

    /// `v`'s valence.
    fn disk_valence(mesh: &LoopMesh, v: VertexKey) -> usize {
        collect_iter::<_, 8>(mesh, mesh.disk_edges(v)).len()
    }

    fn tag_faces(mesh: &mut LoopMesh, faces: &[FaceKey]) -> ElementAttributeKey {
        let tag = mesh
            .register_face_attribute(ElementAttributeDesc::numeric(
                AttributeKey::custom("face_tag"),
                AttrType::Vec3,
            ))
            .unwrap();
        for (i, &face) in faces.iter().enumerate() {
            mesh.face_attribs_mut()
                .set_vec3(tag, face.as_usize(), [i as f32, 0.0, 0.0]);
        }
        tag
    }

    /// The one surviving face with `sides` corners, and the fixture index its
    /// per-face tag names.
    fn merged_face_donor_index(mesh: &LoopMesh, tag: ElementAttributeKey, sides: usize) -> usize {
        let mut merged: Option<FaceKey> = None;
        let mut faces = MeshFaces::default();
        while let Some(face) = faces.next(mesh) {
            if mesh.face_vert_keys(face).len() == sides {
                assert!(merged.is_none(), "exactly one {sides}-corner face survives");
                merged = Some(face);
            }
        }
        let merged = merged.expect("merged face exists");
        let tagged = mesh
            .face_attribs()
            .get_vec3(tag, merged.as_usize())
            .expect("merged face carries the tag");
        tagged[0].round() as usize
    }

    /// §6.7 — the helper reserves `base = v.e` and the first spoke in **disk**
    /// order sharing no face with it. On the `grid_3x3` centre that is
    /// `{1,4}` / `{4,7}`, which is the fact the whole donor derivation rests on.
    #[test]
    fn vert_dissolve_helper_reserves_base_and_first_non_neighbour_spoke() {
        let mut mesh = LoopMesh::new();
        let (verts, _faces) = build_grid_3x3(&mut mesh);
        let centre = verts[4];

        let disk: Vec<EdgeKey> = collect_iter::<_, 8>(&mesh, mesh.disk_edges(centre)).into_vec();
        let ends = |e: EdgeKey| {
            let [a, b] = mesh.edges.get(e).vertices;
            let mut pair = [
                verts.iter().position(|&v| v == a).unwrap(),
                verts.iter().position(|&v| v == b).unwrap(),
            ];
            pair.sort();
            pair
        };
        assert_eq!(
            disk.iter().map(|&e| ends(e)).collect::<Vec<_>>(),
            vec![[1, 4], [3, 4], [4, 5], [4, 7]],
            "§3.1 disk order at the grid centre is attachment order"
        );

        let (base, keep) = mesh
            .vert_dissolve_reserved_spokes(centre)
            .expect("interior 4-fan reserves two spokes");
        assert_eq!(ends(base), [1, 4], "base is the disk anchor v.e");
        assert_eq!(
            ends(keep),
            [4, 7],
            "keep is the first spoke in disk order sharing no face with base"
        );
    }

    /// §6.7 worked example 1 — `grid_3x3` centre: halves `{f0,f2}` and
    /// `{f1,f3}`, half B (carrying f3) finishes last, so the octagon takes
    /// fixture face **3**'s per-face customdata. (Valence 4 alone cannot
    /// discriminate §6.7's schedule from §6.3's — the fans below can.)
    #[test]
    fn vert_dissolve_helper_grid_centre_merged_face_cd_donor() {
        let mut mesh = LoopMesh::new();
        let (verts, faces) = build_grid_3x3(&mut mesh);
        let tag = tag_faces(&mut mesh, &faces);

        assert!(mesh.vert_dissolve(verts[4]));

        assert_eq!(mesh.face_count(), 1);
        assert_eq!(merged_face_donor_index(&mesh, tag, 8), 3);
        mesh.validate_topology();
    }

    /// §6.7 worked example 3 — `disk_fan(5)`: half A `{f1,f2}` completes in pass
    /// 2, half B `{f3,f4,f0}` in pass 3, so the merged pentagon carries fixture
    /// face **4**. The old whole-fan chain answered face 2 here.
    #[test]
    fn vert_dissolve_helper_disk_fan_5_merged_face_cd_donor() {
        let mut mesh = LoopMesh::new();
        let (apex, faces) = build_disk_fan(&mut mesh, 5);
        let tag = tag_faces(&mut mesh, &faces);

        assert!(mesh.vert_dissolve(apex));

        assert_eq!(mesh.face_count(), 1);
        assert_eq!(mesh.vertex_count(), 5, "the apex is gone; the rim survives");
        assert_eq!(mesh.edge_count(), 5, "every spoke is gone");
        assert_eq!(merged_face_donor_index(&mesh, tag, 5), 4);
        mesh.validate_topology();
    }

    /// §6.7 worked example 2 — `disk_fan(6)`: half B is assembled out of two
    /// sub-merges (`X` in pass 1, `Y` in pass 3) and closes in pass 4 carrying
    /// `Y`'s block, so the hexagon takes fixture face **4** — neither the first
    /// join's face (f5) nor the base spoke's original entry face (f1).
    #[test]
    fn vert_dissolve_helper_disk_fan_6_merged_face_cd_donor() {
        let mut mesh = LoopMesh::new();
        let (apex, faces) = build_disk_fan(&mut mesh, 6);
        let tag = tag_faces(&mut mesh, &faces);

        assert!(mesh.vert_dissolve(apex));

        assert_eq!(mesh.face_count(), 1);
        assert_eq!(mesh.vertex_count(), 6);
        assert_eq!(mesh.edge_count(), 6);
        assert_eq!(merged_face_donor_index(&mesh, tag, 6), 4);
        mesh.validate_topology();
    }

    /// §6.7's `R` rule, arm `R = 0` / `R = 1` with **`base` interior** — the
    /// disk anchor is `f0`'s closing edge, an interior spoke. The collapse takes
    /// the face-joining route, so the whole `k`-face fan merges into one face
    /// (`k + 2` corners, this fan being all-triangles) and the apex **survives**
    /// as a 2-valence corner. `f0` is a triangle, so its third edge still joins
    /// the two split targets on the merged outline: they are adjacent, the SFME
    /// refuses, and the helper returns **false** — over a mesh it has already
    /// merged. Donor is fixture face 2 at every `k` (§6.7's trace).
    #[test]
    fn vert_dissolve_helper_boundary_fan_base_interior_merges_all_and_returns_false() {
        for k in 3..=5 {
            let mut mesh = LoopMesh::new();
            let (apex, faces) = build_open_disk_fan(&mut mesh, k);
            let tag = tag_faces(&mut mesh, &faces);

            assert!(
                !mesh.vert_dissolve(apex),
                "k = {k}: the split targets are adjacent, so the SFME refuses"
            );

            assert_eq!(mesh.face_count(), 1, "k = {k}: the whole fan merged");
            assert_eq!(
                mesh.vertex_count(),
                k + 2,
                "k = {k}: apex and rim all survive"
            );
            assert_eq!(
                mesh.edge_count(),
                k + 2,
                "k = {k}: every interior spoke went"
            );
            assert_eq!(
                disk_valence(&mesh, apex),
                2,
                "k = {k}: the apex is left a 2-valence corner of the merged face"
            );
            assert_eq!(merged_face_donor_index(&mesh, tag, k + 2), 2);
            mesh.validate_topology();
        }
    }

    /// §6.7's `R = 1` arm with **`base` boundary**: the same half-disc listed so
    /// the anchor is the outermost spoke. `keep` is interior, so the scan stalls
    /// the fan at **two faces** hinged across it, and the collapse's plain route
    /// — whose precondition is a 2-valence vert — refuses the 3-valence apex.
    /// **false**, with the fan already half-merged. Same fan, same `k`, different
    /// disk order, entirely different partial mesh from the arm above.
    #[test]
    fn vert_dissolve_helper_boundary_fan_base_boundary_keep_interior_stalls_at_two_faces() {
        for k in 3..=4 {
            let mut mesh = LoopMesh::new();
            let (apex, _faces) = build_open_disk_fan_boundary_anchor(&mut mesh, k);

            assert!(
                !mesh.vert_dissolve(apex),
                "k = {k}: the plain collapse route refuses a 3-valence vert"
            );

            assert_eq!(mesh.face_count(), 2, "k = {k}: stalled on the keep hinge");
            assert_eq!(mesh.vertex_count(), k + 2);
            assert_eq!(mesh.edge_count(), 2 * k + 1 - (k - 2));
            assert_eq!(
                disk_valence(&mesh, apex),
                3,
                "k = {k}: base, keep and the unreserved boundary spoke stand"
            );
            mesh.validate_topology();
        }
    }

    /// §6.7's `R = 2` arm: build the two **end** faces first and the keep search
    /// reserves the far boundary spoke, so every unreserved spoke is interior,
    /// the scan consumes them all, and the apex is genuinely 2-valence when the
    /// plain collapse route runs. Clean dissolve, **true** — at any `k`, which is
    /// what shows the rule is indexed by `R` and not by the face count.
    #[test]
    fn vert_dissolve_helper_boundary_fan_both_reserved_boundary_dissolves_cleanly() {
        for (k, donor) in [(3usize, 2usize), (4, 3)] {
            let mut mesh = LoopMesh::new();
            let (apex, faces) = build_open_disk_fan_end_faces_first(&mut mesh, k);
            let tag = tag_faces(&mut mesh, &faces);

            assert!(mesh.vert_dissolve(apex), "k = {k}: R = 2 dissolves cleanly");

            assert_eq!(mesh.face_count(), 1);
            assert_eq!(mesh.vertex_count(), k + 1, "k = {k}: the apex is gone");
            assert_eq!(mesh.edge_count(), k + 1);
            assert_eq!(merged_face_donor_index(&mesh, tag, k + 1), donor);
            mesh.validate_topology();
        }
    }

    /// §6.7 — the return value is **orthogonal to vert removal**. `open_quad_fan`
    /// has an interior anchor like the triangle fan above, so the fan merges whole
    /// and the apex survives; but `f0` is a quad, so `outer_0` separates the two
    /// split targets on the merged outline, the SFME fires, and the helper returns
    /// **true with the apex still in the mesh**. The union is an octagon —
    /// `Σ nᵢ − 2(k−1) = 8`, not the `k + 2` a triangle fan would give.
    #[test]
    fn vert_dissolve_helper_boundary_fan_ngon_first_face_returns_true_without_removing_the_vert() {
        let mut mesh = LoopMesh::new();
        let (apex, faces) = build_open_quad_fan(&mut mesh, 3);
        let tag = tag_faces(&mut mesh, &faces);

        assert!(mesh.vert_dissolve(apex));

        assert!(
            mesh.vertices.try_get(apex).is_some(),
            "true does not mean the vertex was dissolved"
        );
        assert_eq!(disk_valence(&mesh, apex), 2);
        assert_eq!(mesh.face_count(), 1);
        assert_eq!(mesh.vertex_count(), 8);
        assert_eq!(mesh.edge_count(), 8);
        assert_eq!(merged_face_donor_index(&mesh, tag, 8), 2);
        mesh.validate_topology();
    }

    /// The reserved spokes are what `R` counts, so pin them: a boundary spoke is
    /// an ordinary candidate for either role, and which one it lands in is fixed
    /// by **disk order** (attachment order), not by the fan's geometry.
    #[test]
    fn vert_dissolve_helper_boundary_fan_reserved_spokes_follow_the_disk_order() {
        let is_boundary = |mesh: &LoopMesh, e: EdgeKey| mesh.edge_radial_len_capped(e, 2) == 1;

        let mut mesh = LoopMesh::new();
        let (apex, _faces) = build_open_disk_fan(&mut mesh, 4);
        let (base, keep) = mesh.vert_dissolve_reserved_spokes(apex).expect("reserved");
        assert!(!is_boundary(&mesh, base), "anchor is f0's closing edge");
        assert!(!is_boundary(&mesh, keep), "R = 0 at k = 4");

        let mut mesh = LoopMesh::new();
        let (apex, _faces) = build_open_disk_fan_boundary_anchor(&mut mesh, 4);
        let (base, keep) = mesh.vert_dissolve_reserved_spokes(apex).expect("reserved");
        assert!(is_boundary(&mesh, base), "the apex is f0's last ring vert");
        assert!(!is_boundary(&mesh, keep), "R = 1, base boundary");

        let mut mesh = LoopMesh::new();
        let (apex, _faces) = build_open_disk_fan_end_faces_first(&mut mesh, 4);
        let (base, keep) = mesh.vert_dissolve_reserved_spokes(apex).expect("reserved");
        assert!(is_boundary(&mesh, base));
        assert!(
            is_boundary(&mesh, keep),
            "R = 2: the far boundary spoke attached before the keep search reached it"
        );
    }

    /// The donor's age channel is the stored radial **entry**, and it has to
    /// survive *across* calls: `unsubdivide` dissolves adjacent verts in
    /// sequence, so the second call's fan contains a face the first call merged.
    /// A merged face is the newest face on every perimeter edge it kept
    /// (§6.3's entry-reseat rule), so it wins every donor contest it enters
    /// next time — an age channel rebuilt per call would score it as pristine
    /// and read a stale build-order entry instead.
    ///
    /// On the 4x4 grid, dissolving vert 5 merges `{f0, f1, f3, f4}`; the spoke
    /// `{9,10}` of vert 10's fan then flanks that merged face and `f7`, and its
    /// *pristine* entry named `f7` (the later-built of the two). After the first
    /// dissolve it must name the merged face.
    #[test]
    fn vert_dissolve_helper_overlapping_fans_donor_reads_the_merged_face() {
        let mut mesh = LoopMesh::new();
        let (verts, faces) = build_grid_4x4(&mut mesh);
        let tag = tag_faces(&mut mesh, &faces);

        assert!(mesh.vert_dissolve(verts[5]));

        let spoke_9_10 = collect_iter::<_, 8>(&mesh, mesh.disk_edges(verts[10]))
            .into_iter()
            .find(|&e| mesh.edges.get(e).contains(verts[9]).is_some())
            .expect("{9,10} survives the first dissolve");
        let entry_face = mesh.loop_face(mesh.edge_radial_entry(spoke_9_10).expect("radial-2"));
        assert_eq!(
            mesh.face_attribs()
                .get_vec3(tag, entry_face.as_usize())
                .expect("tagged")[0]
                .round() as usize,
            4,
            "the merged face (donor f4's block) is the newest face on {{9,10}}"
        );

        assert!(mesh.vert_dissolve(verts[10]));

        assert_eq!(mesh.face_count(), 3, "the merged 12-gon plus f2 and f6");
        assert_eq!(mesh.vertex_count(), 14);
        assert_eq!(mesh.edge_count(), 16);
        assert_eq!(
            merged_face_donor_index(&mesh, tag, 12),
            8,
            "half B ({{f5, f8}}, donor f8) completes last"
        );
        mesh.validate_topology();
    }

    /// Attach a triangle that meets `v` at `v` alone (its other two verts are
    /// fresh), turning `v`'s disk into a **multi-fan**: the counts still read
    /// like a boundary fan (radial 1 or 2 everywhere, two boundary spokes, an
    /// interior spoke present, nothing pinched), but no chain of interior spokes
    /// bridges the loose triangle to the rest.
    fn attach_loose_triangle(mesh: &mut LoopMesh, v: VertexKey) -> FaceKey {
        let a = mesh.create_vertex(Vec3::new(5.0, 0.0, 0.0));
        let b = mesh.create_vertex(Vec3::new(5.0, 1.0, 0.0));
        face_from_ring(mesh, &[v, a, b])
    }

    /// §3 / §3.1 — the manifold-fan predicate is not a count: a closed fan plus a
    /// triangle touching the apex at the apex only passes every count clause and
    /// is still a multi-fan. The fan walk refuses it, and refuses it **before any
    /// mutation** — whichever of the two sub-fans the disk anchor happens to sit
    /// in (build the loose triangle after the fan, and before it).
    #[test]
    fn vert_dissolve_multi_fan_is_refused_before_any_mutation() {
        for loose_first in [false, true] {
            let mut mesh = LoopMesh::new();
            let apex = if loose_first {
                let apex = mesh.create_vertex(Vec3::ZERO);
                attach_loose_triangle(&mut mesh, apex);
                let rim: Vec<VertexKey> = (0..3)
                    .map(|i| {
                        let theta = std::f32::consts::TAU * (i as f32) / 3.0;
                        mesh.create_vertex(Vec3::new(theta.cos(), theta.sin(), 0.0))
                    })
                    .collect();
                for i in 0..3 {
                    face_from_ring(&mut mesh, &[apex, rim[i], rim[(i + 1) % 3]]);
                }
                apex
            } else {
                let (apex, _faces) = build_disk_fan(&mut mesh, 3);
                attach_loose_triangle(&mut mesh, apex);
                apex
            };

            let before = (mesh.vertex_count(), mesh.edge_count(), mesh.face_count());
            assert_eq!(before, (6, 9, 4));
            assert_eq!(disk_valence(&mesh, apex), 5);

            assert!(
                !mesh.vert_dissolve(apex),
                "loose_first = {loose_first}: a multi-fan is not a manifold fan"
            );

            assert_eq!(
                (mesh.vertex_count(), mesh.edge_count(), mesh.face_count()),
                before,
                "loose_first = {loose_first}: refusal is before any mutation"
            );
            assert_eq!(disk_valence(&mesh, apex), 5);
            mesh.validate_topology();
        }
    }

    /// Attach a closed `n`-triangle cone to an existing apex `v`, its rim lifted
    /// to `z` so the two cones of a double cone are distinguishable. Same ring
    /// order as `build_disk_fan`, so every spoke it adds is interior.
    fn attach_closed_cone(mesh: &mut LoopMesh, v: VertexKey, n: usize, z: f32) -> Vec<FaceKey> {
        let rim: Vec<VertexKey> = (0..n)
            .map(|i| {
                let theta = std::f32::consts::TAU * (i as f32) / (n as f32);
                mesh.create_vertex(Vec3::new(theta.cos(), theta.sin(), z))
            })
            .collect();
        (0..n)
            .map(|i| face_from_ring(mesh, &[v, rim[i], rim[(i + 1) % n]]))
            .collect()
    }

    /// §3 / §3.1 — the **all-interior** multi-fan: two closed cones meeting only
    /// at the apex (the pinch vertex / double cone). Unlike the loose-triangle
    /// multi-fan, this shape passes every count clause *vacuously* — every spoke
    /// is radial-2, there are **zero** boundary spokes, an interior spoke exists,
    /// nothing is pinched — so the fan walk is the only thing refusing it: it
    /// closes on the cone holding the disk anchor and never reaches the other.
    /// Built in both cone orders, so the anchor lands once in each cone.
    #[test]
    fn vert_dissolve_double_cone_is_refused_before_any_mutation() {
        for upper_first in [false, true] {
            let mut mesh = LoopMesh::new();
            let apex = mesh.create_vertex(Vec3::ZERO);
            let (first_z, second_z) = if upper_first {
                (1.0, -1.0)
            } else {
                (-1.0, 1.0)
            };
            attach_closed_cone(&mut mesh, apex, 3, first_z);
            attach_closed_cone(&mut mesh, apex, 3, second_z);

            let before = (mesh.vertex_count(), mesh.edge_count(), mesh.face_count());
            assert_eq!(before, (7, 12, 6));
            assert_eq!(disk_valence(&mesh, apex), 6);
            // The vacuity that makes this the walk's own test case: no count
            // clause objects to it.
            let disk = collect_iter::<_, 8>(&mesh, mesh.disk_edges(apex));
            for &spoke in disk.iter() {
                assert_eq!(
                    mesh.edge_radial_len_capped(spoke, 3),
                    2,
                    "upper_first = {upper_first}: every spoke of a double cone is interior"
                );
                assert!(mesh.spoke_face_pair(spoke).is_some(), "and none is pinched");
            }
            assert!(
                !mesh.vert_fan_walk_covers_disk(apex, &disk),
                "upper_first = {upper_first}: the walk covers one cone, not the disk"
            );

            assert!(
                !mesh.vert_dissolve(apex),
                "upper_first = {upper_first}: a double cone is not a manifold fan"
            );

            assert_eq!(
                (mesh.vertex_count(), mesh.edge_count(), mesh.face_count()),
                before,
                "upper_first = {upper_first}: refusal is before any mutation"
            );
            assert_eq!(disk_valence(&mesh, apex), 6);
            mesh.validate_topology();
        }
    }

    /// The other side of the same guard: the shapes §3 *does* call manifold fans
    /// must still walk their whole disk — a fully-interior fan (the walk closes),
    /// a boundary fan (the walk is open and sweeps both ways out of the start
    /// face), and the n-gon fan.
    #[test]
    fn vert_fan_walk_covers_the_disk_of_every_supported_fan() {
        let mut mesh = LoopMesh::new();
        let (apex, _faces) = build_disk_fan(&mut mesh, 5);
        let disk = collect_iter::<_, 8>(&mesh, mesh.disk_edges(apex));
        assert!(mesh.vert_fan_walk_covers_disk(apex, &disk), "interior fan");

        let mut mesh = LoopMesh::new();
        let (apex, _faces) = build_open_disk_fan(&mut mesh, 4);
        let disk = collect_iter::<_, 8>(&mesh, mesh.disk_edges(apex));
        assert!(mesh.vert_fan_walk_covers_disk(apex, &disk), "boundary fan");

        let mut mesh = LoopMesh::new();
        let (apex, _faces) = build_open_disk_fan_boundary_anchor(&mut mesh, 4);
        let disk = collect_iter::<_, 8>(&mesh, mesh.disk_edges(apex));
        assert!(
            mesh.vert_fan_walk_covers_disk(apex, &disk),
            "boundary fan whose anchor is an outline spoke — the walk starts on a \
             boundary-most face and only the reverse sweep covers it"
        );

        let mut mesh = LoopMesh::new();
        let (apex, _faces) = build_open_quad_fan(&mut mesh, 3);
        let disk = collect_iter::<_, 8>(&mesh, mesh.disk_edges(apex));
        assert!(mesh.vert_fan_walk_covers_disk(apex, &disk), "n-gon fan");
    }

    /// §3's other two non-manifold shapes stay no-ops, and stay no-ops for the
    /// count clauses' own reasons (a bowtie has four boundary spokes; a T-junction
    /// has a radial-3 spoke) — the fan walk must not have made either reachable.
    #[test]
    fn vert_dissolve_bowtie_and_t_junction_stay_no_ops() {
        let mut mesh = LoopMesh::new();
        let apex = mesh.create_vertex(Vec3::ZERO);
        attach_loose_triangle(&mut mesh, apex);
        attach_loose_triangle(&mut mesh, apex);
        let before = (mesh.vertex_count(), mesh.edge_count(), mesh.face_count());
        assert!(!mesh.vert_dissolve(apex), "bowtie");
        assert_eq!(
            (mesh.vertex_count(), mesh.edge_count(), mesh.face_count()),
            before
        );
        mesh.validate_topology();

        let mut mesh = LoopMesh::new();
        let apex = mesh.create_vertex(Vec3::ZERO);
        let far = mesh.create_vertex(Vec3::new(1.0, 0.0, 0.0));
        for i in 0..3 {
            let wing = mesh.create_vertex(Vec3::new(0.5, 1.0, i as f32));
            face_from_ring(&mut mesh, &[apex, far, wing]);
        }
        let before = (mesh.vertex_count(), mesh.edge_count(), mesh.face_count());
        assert!(!mesh.vert_dissolve(apex), "T-junction / book spine");
        assert_eq!(
            (mesh.vertex_count(), mesh.edge_count(), mesh.face_count()),
            before
        );
        mesh.validate_topology();
    }

    /// §3's edge-pair class: a 2-valence corner of a single face collapses through
    /// JEKV, and never reaches the fan walk at all.
    #[test]
    fn vert_dissolve_edge_pair_on_a_face_boundary_collapses() {
        let mut mesh = LoopMesh::new();
        let ring: Vec<VertexKey> = [
            Vec3::ZERO,
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        ]
        .into_iter()
        .map(|p| mesh.create_vertex(p))
        .collect();
        face_from_ring(&mut mesh, &ring);

        assert!(mesh.vert_dissolve(ring[1]));

        assert_eq!(mesh.vertex_count(), 3);
        assert_eq!(mesh.edge_count(), 3);
        assert_eq!(mesh.face_count(), 1);
        mesh.validate_topology();
    }

    /// §6.7's no-keep branch: at valence 3 every spoke shares a face with the
    /// anchor, so nothing is reserved and the fan collapses through the single
    /// join + reaping collapse. The halves schedule must not claim this fan.
    #[test]
    fn vert_dissolve_helper_three_triangle_fan_takes_no_keep_branch() {
        let mut mesh = LoopMesh::new();
        let (apex, _faces) = build_disk_fan(&mut mesh, 3);
        assert!(
            mesh.vert_dissolve_reserved_spokes(apex).is_none(),
            "a 3-fan has no keep spoke"
        );

        assert!(mesh.vert_dissolve(apex));

        assert_eq!(mesh.face_count(), 1);
        assert_eq!(mesh.vertex_count(), 3, "collapses to a single triangle");
        assert_eq!(mesh.edge_count(), 3);
        mesh.validate_topology();
    }
}
