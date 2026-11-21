//! Per-edge attribute and seam / sharp flag inheritance (BEVEL.md §6.7 / §7.7).

use super::*;

impl LoopMesh {
    /// Copy each beveled edge's whole per-edge attribute block onto its strip's two
    /// **outer longitudinal edges** (BEVEL.md §7.7.2; mechanism (1) of §6.7.1,
    /// unconditional).
    ///
    /// A beveled edge `g = (a, b)` sheds a strip running *along* `g`. The two edges of
    /// that strip parallel to `g` — one in each of `g`'s two re-stitched radial faces,
    /// between the near corner at `a` and the near corner at `b` — are the edges that
    /// *replace* `g` on each rebuilt face's boundary, so they inherit what `g` carried:
    /// crease, bevel weight, every custom edge layer, and — as two passengers of the
    /// same block — the seam and sharp bits. The count is two, always, independent of
    /// `segments`. The perpendicular strip rails (the patch ring segments) descend from
    /// no original and are left to the parameter-gated bridging.
    ///
    /// `g` itself is still alive here (it is consumed only when its endpoints are
    /// freed), so its block is read straight off it — no snapshot needed, nothing in the
    /// surgery writes to a beveled edge's own slot.
    ///
    /// Unconditional: runs whether or not `mark_seam` / `mark_sharp` is set.
    pub(super) fn inherit_strip_long_edge_attributes(
        &mut self,
        beveled: &[EdgeKey],
        corner_vert: &[CornerVert],
    ) {
        for &edge in beveled {
            let [a, b] = self.edge_vert_keys(edge);
            // §7.8.3: the two long edges *are* the strip's rails, so name them by side
            // label wherever both endpoints carry one. The face-keyed pairs below coincide
            // with the rails on mirrored labels, but under the tied-vote twist they are no
            // rail at all — copying `edge`'s block onto a rebuilt flank's own edge.
            if let Some(rails) = labelled_strip_rails(corner_vert, edge, a, b) {
                for (near_a, near_b) in rails {
                    if let Some(long_edge) = self.query_edge_exists(near_a, near_b) {
                        self.edge_attribs
                            .copy_into(long_edge.as_usize(), edge.as_usize());
                    }
                }
                continue;
            }
            // One long edge per radial face of the strip: the near corner at `a` to
            // the near corner at `b` on that face.
            let mut faces: Vec<FaceKey> = Vec::new();
            for record in corner_vert {
                if record.edge == edge && !faces.contains(&record.face) {
                    faces.push(record.face);
                }
            }
            for face in faces {
                let (Some(near_a), Some(near_b)) = (
                    near_corner(corner_vert, edge, a, face),
                    near_corner(corner_vert, edge, b, face),
                ) else {
                    continue; // a declined corner built no strip to inherit onto
                };
                if near_a == near_b {
                    continue; // a collapsed through-junction side, no long edge
                }
                if let Some(long_edge) = self.query_edge_exists(near_a, near_b) {
                    self.edge_attribs
                        .copy_into(long_edge.as_usize(), edge.as_usize());
                }
            }
        }
    }

    /// Snapshot mechanism (1)'s **corner runs** against the pristine mesh
    /// (BEVEL.md §6.5.1 / §7.7.3 / §6.7.1.1(a′)), before the re-stitch consumes the
    /// beveled endpoints' corners.
    ///
    /// Re-stitching a face `f` that touched a consumed endpoint `v` replaces `v`'s
    /// corner by the ordered run of outermost-ring verts recorded for `(v, f)` —
    /// that run *is* `f`'s corner run. Every edge of the run inherits the **whole
    /// per-edge attribute block** of `f`'s *winding-outgoing* original edge at `v`
    /// (the edge `f`'s corner at `v` leaves along); the run's interior verts own no
    /// original edge of their own, so every segment takes the same donor. The seam /
    /// sharp bits ride that copy as two passengers and are then overridden by the
    /// contiguity AND of `f`'s two `v`-incident originals — the run is flagged iff
    /// *both* of them are.
    ///
    /// The donor is *not* a fixed side of the beveled edge: at a valence-3 terminal it
    /// is the leg the absorbing face leaves the corner along, which flips between the
    /// beveled edge's two endpoints (§7.7.3 regime 1 — `fL` and `fR` traverse `E` in
    /// opposite directions, so the face that leaves along `E` is `fL` at one end and
    /// `fR` at the other). Reading it off the face's own winding, per face, gets that
    /// for free.
    ///
    /// Because the runs come from the face-substitution table, only faces that
    /// **existed before the bevel** are ever considered — the face-provenance half of
    /// §6.7.1.1(a). An edge whose two faces are both new (a patch side, an interior
    /// vmesh ring, a junction-end cross-section chain) belongs to no corner run and is
    /// governed by mechanism (2) alone.
    ///
    /// The pass is deliberately free of case analysis: it does not look at the miter
    /// kind, at whether the sector's boundvert was split, or at `mark_seam` /
    /// `mark_sharp`. A miter-free junction sector's run is a single boundvert (zero
    /// edges — the gate runs and does no work); a mitered sector's run is the miter's
    /// fill chain; a terminal's absorbing face's run is the beveled edge's far-end
    /// cross-section chain, whose two flanking originals are the terminal's unbeveled
    /// legs.
    pub(super) fn snapshot_corner_runs(
        &self,
        substitutions: &[FaceSubstitution],
    ) -> Vec<CornerRun> {
        let mut runs: Vec<CornerRun> = Vec::new();
        for substitution in substitutions {
            if substitution.verts.len() < 2 {
                continue; // an empty run: nothing to flag
            }
            let Some((incoming, outgoing)) =
                self.face_corner_edges(substitution.face, substitution.vertex)
            else {
                continue;
            };
            // A FALSE AND is recorded too, and matters: the pass **assigns**, so a
            // false run *clears* its corner segments (§6.7.5.4 case (5)'s write order).
            let seam = self.edge_seam(incoming) && self.edge_seam(outgoing);
            let sharp = self.edge_sharp(incoming) && self.edge_sharp(outgoing);
            runs.push(CornerRun {
                verts: substitution.verts.clone(),
                donor: outgoing,
                seam,
                sharp,
            });
        }
        runs
    }

    /// Copy each snapshotted corner run's donor block onto the rebuilt faces' corner
    /// segments, then override the two flag passengers with the contiguity AND
    /// (BEVEL.md §7.7.3 / §6.7.1.1(a′), mechanism (1) — UNCONDITIONAL).
    ///
    /// This is a **whole per-edge attribute block** copy — crease, bevel weight and
    /// every custom edge layer travel with it, not just the seam / sharp booleans. The
    /// AND then re-derives those two bits on top, so a flagged-on-one-side corner
    /// leaves no one-segment spur.
    ///
    /// Run after the re-stitch, when each run's consecutive verts are consecutive on
    /// a rebuilt face's boundary and so the corner segments exist. A run of `> 1`
    /// segment (a miter fill chain, a terminal's far-end chain) takes one donor and one
    /// AND throughout: both are evaluated once per face, not once per segment.
    ///
    /// The pass **assigns**, it does not accumulate — a face whose AND is FALSE clears
    /// its whole corner run. That is what makes a desynchronised ring walk (§6.7.5.4
    /// case (5)) observable as *missing* bridge edges rather than misplaced ones: the
    /// steps such a span burns land on a split sector's fill chain, which is that
    /// sector's rebuilt face's corner run, so this pass — which runs **after** the walk
    /// — wipes them again. Only corner segments are written; a rebuilt face's
    /// longitudinal boundary edges take their own original's block from the strip copy
    /// and are untouched here.
    ///
    /// A corner segment joins two verts of one endpoint's ring, so it is always freshly
    /// minted — a re-pointed survivor spans a ring vert and an *original* far vertex and
    /// can never be hit here, which is what makes the blind copy safe.
    pub(super) fn apply_corner_run_attributes(&mut self, runs: &[CornerRun]) {
        for run in runs {
            for pair in run.verts.windows(2) {
                let Some(segment) = self.query_edge_exists(pair[0], pair[1]) else {
                    continue;
                };
                self.edge_attribs
                    .copy_into(segment.as_usize(), run.donor.as_usize());
                self.set_edge_seam(segment, run.seam);
                self.set_edge_sharp(segment, run.sharp);
            }
        }
    }

    /// The outermost-ring vertex an unbeveled leg re-points onto (BEVEL.md §6.7.5.4
    /// case (4b)): its sector's single boundvert at a miter-free sector, the hub under
    /// `PATCH`, its assigned arc sample under `ARC`. The leg has no boundvert of its
    /// own, so this vertex is its stand-in — the survivor the re-stitch hangs it on, and
    /// the point mechanism (2)'s bridge span starts from when the leg is the anchor.
    ///
    /// Read straight off the face-substitution runs, in the run-order contract
    /// [`FaceSubstitution`] states: the leg meets its face's run at the run's first
    /// vert when the leg is that face's incoming corner edge, at its last when it is
    /// the outgoing one. Every face on the leg's radial fan must agree; anything else
    /// (a wire leg with no radial face, a face the placement pass declined to
    /// substitute, a leg whose two sides disagree) declines and returns `None`.
    pub(super) fn leg_ring_vertex(
        &self,
        vertex: VertexKey,
        leg: EdgeKey,
        substitutions: &[FaceSubstitution],
    ) -> Option<VertexKey> {
        let mut landing: Option<VertexKey> = None;
        for loop_key in self.edge_radial_iter(leg) {
            let face = self.loop_face(loop_key);
            let candidate = face_substitution(substitutions, vertex, face)
                .zip(self.face_corner_edges(face, vertex))
                .and_then(|(run, (incoming, outgoing))| {
                    if leg == incoming {
                        run.first().copied()
                    } else if leg == outgoing {
                        run.last().copied()
                    } else {
                        None
                    }
                })?;
            match landing {
                None => landing = Some(candidate),
                Some(agreed) if agreed == candidate => {}
                Some(_) => return None,
            }
        }
        landing
    }

    /// Snapshot every junction's disk as mechanism (2)'s walk reads it (BEVEL.md
    /// §6.7.5.3 step 1 / §6.7.5.4 case (4b)): **every** disk edge in rotational order,
    /// beveled or not, with its pristine seam / sharp bits and — for an unbeveled leg —
    /// the ring vertex it re-points onto.
    ///
    /// The walk asks two separate questions of each entry. *Is it flagged?* delimits,
    /// and every disk edge answers it: a flagged leg ends the run that reaches it and
    /// anchors a span of its own. *Is it beveled?* counts, and only beveled edges add
    /// to a gap's length. Keeping the two apart is the whole of the leg rule — the count
    /// is the walk's only `beveled` branch.
    ///
    /// `ring_vertex` is filled for the unbeveled legs alone, because a beveled edge's
    /// span-start vertex is the trailing end of its own ring chain and the ring is not
    /// built yet here; the walk resolves it there (see [`Self::bridge_junction_flags`]).
    ///
    /// Must be read against the pristine mesh: the re-stitch consumes the junction
    /// vertex, so its disk exists only before phase 2. Only mechanism (2) reads this, so
    /// the callers build it only when `mark_seam` / `mark_sharp` asked for it.
    pub(super) fn snapshot_junction_disks(
        &self,
        junctions: &[VertexKey],
        beveled: &[EdgeKey],
        substitutions: &[FaceSubstitution],
    ) -> Vec<JunctionDisk> {
        junctions
            .iter()
            .map(|&vertex| {
                let edges = self
                    .vertex_corner_ring(vertex)
                    .into_iter()
                    .map(|corner| {
                        let edge = corner.incoming_edge;
                        let beveled = beveled.contains(&edge);
                        DiskEdge {
                            edge,
                            beveled,
                            seam: self.edge_seam(edge),
                            sharp: self.edge_sharp(edge),
                            ring_vertex: if beveled {
                                None
                            } else {
                                self.leg_ring_vertex(vertex, edge, substitutions)
                            },
                        }
                    })
                    .collect();
                JunctionDisk { vertex, edges }
            })
            .collect()
    }

    /// Bridge the seam / sharp flag across one junction's outermost ring (BEVEL.md
    /// §6.7.4 / §6.7.5.3 — mechanism (2), the only effect `mark_seam` / `mark_sharp`
    /// gate).
    ///
    /// **The operational walk** (§6.7.5.4 case (5)): *count in beveled edges, step in
    /// ring chains.* From each flagged disk edge, scan forward in rotational order,
    /// counting the **beveled** edges among the unflagged ones passed, and stop at the
    /// next **flagged** disk edge — *any* flagged edge, beveled or not, since the
    /// delimiter test asks every entry (§6.7.5.3 step 1). Then lay the span down from
    /// the anchor's **span-start ring vertex**, flagging the next `m` outermost-ring
    /// chains: every cross-section edge of each, not just one edge of it.
    ///
    /// Three things make it come out right, and each is load-bearing:
    ///
    /// - **The termination clause.** A scan that comes back round to its own anchor
    ///   without meeting a different flagged edge records nothing — the disk's only
    ///   flagged edge has no gap to bridge. Omit it and a single-flagged-edge junction
    ///   flags nearly its whole ring.
    /// - **A leg is a first-class anchor.** Its span starts at its **ring vertex**
    ///   (§6.7.5.4 case (4b)); a beveled edge's starts at its trailing boundvert, which
    ///   is where its own ring chain ends. So a junction whose only flagged *beveled*
    ///   edge would abort on the termination clause bridges after all once a leg is
    ///   flagged too.
    /// - **Counting and stepping desynchronise.** A chain crossing no beveled edge (a
    ///   split sector's miter fill chain) is *stepped over* but never *counted*, so a
    ///   span whose ring path enters a split sector burns steps on the fill and comes up
    ///   short at its far end. Those burned marks land on the split sector's rebuilt
    ///   face's **corner run**, and the corner-run pass runs **after** this one and
    ///   *assigns*, so it clears them again: a desync is observable as *missing* bridge
    ///   edges, never as misplaced ones.
    ///
    /// Only the outermost ring is touched; the interior vmesh rings that close the
    /// corner are never flagged. `seam` and `sharp` run the identical walk on
    /// independent bits.
    pub(super) fn bridge_junction_flags(
        &mut self,
        ring: &[RingChain],
        disk: &JunctionDisk,
        seam: bool,
        sharp: bool,
    ) {
        let chain_count = ring.len();
        let disk_count = disk.edges.len();
        if chain_count < 2 || disk_count < 2 {
            return;
        }

        // The walk *counts* in disk order and *steps* in ring order, so the two orders
        // must run the same way round the junction; were they ever to disagree
        // rotationally, every span would be laid on the wrong side of its anchor.
        debug_assert!(
            cyclically_equal(
                &disk
                    .edges
                    .iter()
                    .filter(|entry| entry.beveled)
                    .map(|entry| entry.edge)
                    .collect::<Vec<_>>(),
                &ring
                    .iter()
                    .filter_map(|chain| chain.cross_edge)
                    .collect::<Vec<_>>(),
            ),
            "the disk's beveled edges and the ring's crossings must agree rotationally",
        );

        // The ring vertex a flagged disk edge's span **starts from**, asked of every
        // entry the same way (§6.7.5.4 case (4b)): a leg carries its own, snapshotted off
        // the face-substitution runs; a beveled edge's is its trailing boundvert, which
        // is exactly where its own crossing chain ends. Both are then resolved through
        // the one lookup below, so the walk never branches on `beveled` outside the count.
        let span_vertex = |entry: &DiskEdge| -> Option<VertexKey> {
            entry.ring_vertex.or_else(|| {
                ring.iter()
                    .find(|chain| chain.cross_edge == Some(entry.edge))
                    .and_then(|chain| chain.samples.last().copied())
            })
        };

        // The first ring chain the span lays down: the chain **beginning** at that
        // vertex. A leg whose ring vertex is an *interior* sample of a chain (the middle
        // leg of an odd absorbed-leg count under `ARC`, §8.9.2) begins no whole chain and
        // so lays nothing — the guide pins no partial-chain span. See
        // `flagged_middle_absorbed_leg_on_interior_arc_sample_lays_no_span_octagonal`.
        let span_start = |index: usize| -> Option<usize> {
            let vertex = span_vertex(&disk.edges[index])?;
            ring.iter()
                .position(|chain| chain.samples.first() == Some(&vertex))
        };

        let bridge = |want_seam: bool, this: &mut Self| {
            let flagged = |index: usize| {
                let entry = &disk.edges[index];
                if want_seam {
                    entry.seam
                } else {
                    entry.sharp
                }
            };

            for anchor in 0..disk_count {
                if !flagged(anchor) {
                    continue;
                }

                let mut gap_length = 0usize;
                let mut delimited_by_another_flag = false;
                for step in 1..=disk_count {
                    let index = (anchor + step) % disk_count;
                    if flagged(index) {
                        delimited_by_another_flag = index != anchor;
                        break;
                    }
                    if disk.edges[index].beveled {
                        gap_length += 1;
                    }
                }
                if !delimited_by_another_flag || gap_length == 0 {
                    continue;
                }
                let Some(start) = span_start(anchor) else {
                    continue;
                };

                for step in 0..gap_length {
                    let chain = &ring[(start + step) % chain_count];
                    for pair in chain.samples.windows(2) {
                        if let Some(cross) = this.query_edge_exists(pair[0], pair[1]) {
                            if want_seam {
                                this.set_edge_seam(cross, true);
                            } else {
                                this.set_edge_sharp(cross, true);
                            }
                        }
                    }
                }
            }
        };

        if seam {
            bridge(true, self);
        }
        if sharp {
            bridge(false, self);
        }
    }

    /// Multi-segment mechanism (1): copy each beveled edge's whole per-edge attribute
    /// block onto its strip's **two outer longitudinal edges** (BEVEL.md §7.7.2 /
    /// §6.7.5.2). The strip is `segments` quads sharing their long edges, so it carries
    /// `segments + 1` longitudinal edge-lines; only the two outer ones (profile index
    /// `0` and index `segments`) lie on the boundary the re-stitched adjacent faces
    /// rebuilt along the edge's two original sides, so only those two inherit — crease,
    /// bevel weight, custom layers, and the seam / sharp bits that ride along with them.
    /// The `segments − 1` interior longitudinal lines descend from no original and keep
    /// the layer default. This is UNCONDITIONAL — it runs whether or not `mark_seam` /
    /// `mark_sharp` is set, exactly as the single-segment inheritance (§6.7.4) does, and
    /// applies at junction, terminal, and through ends alike.
    ///
    /// The two outer longitudinal edges are the only edges joining a chain-end of
    /// the edge's endpoint-`a` profile chain to a chain-end of its endpoint-`b`
    /// chain (the interior profile samples never terminate an outer longitudinal
    /// edge, and the two endpoints' chains face each other across the edge with no
    /// other edge spanning between them). Iterating the four chain ends and keeping
    /// the pairs that resolve to a real edge therefore finds exactly those two.
    pub(super) fn inherit_multisegment_strip_long_edge_attributes(
        &mut self,
        beveled: &[EdgeKey],
        chains: &[EdgeEndChain],
    ) {
        for &edge in beveled {
            let [a, b] = self.edge_vert_keys(edge);
            let chain_a = chains.iter().find(|c| c.edge == edge && c.endpoint == a);
            let chain_b = chains.iter().find(|c| c.edge == edge && c.endpoint == b);
            let (chain_a, chain_b) = match (chain_a, chain_b) {
                (Some(x), Some(y)) => (x, y),
                _ => continue,
            };
            for &near_a in &[chain_a.start, chain_a.end] {
                for &near_b in &[chain_b.start, chain_b.end] {
                    if near_a == near_b {
                        continue;
                    }
                    if let Some(long_edge) = self.query_edge_exists(near_a, near_b) {
                        self.edge_attribs
                            .copy_into(long_edge.as_usize(), edge.as_usize());
                    }
                }
            }
        }
    }
}

/// Whether the two sequences list the same keys in the same **cyclic** order — some
/// rotation of `left` equals `right`. Used to check the disk-order / ring-order
/// agreement mechanism (2)'s walk rests on.
pub(super) fn cyclically_equal(left: &[EdgeKey], right: &[EdgeKey]) -> bool {
    if left.len() != right.len() {
        return false;
    }
    let count = left.len();
    count == 0 || (0..count).any(|shift| (0..count).all(|i| left[(i + shift) % count] == right[i]))
}

/// One junction's disk in rotational order, as mechanism (2)'s walk reads it (BEVEL.md
/// §6.7.5.3 step 1). **Every** disk edge is here, beveled or not: the delimiter test
/// asks all of them, and only the counting test asks whether an edge is beveled. An
/// unbeveled leg is therefore a first-class participant — it can end a run and anchor a
/// span — and the gap count is the walk's only `beveled` branch.
pub(super) struct JunctionDisk {
    pub(super) vertex: VertexKey,
    pub(super) edges: Vec<DiskEdge>,
}

/// One edge of a junction's disk, with the flags it carried on the pristine mesh
/// (BEVEL.md §6.7.5.4 case (4b)).
pub(super) struct DiskEdge {
    pub(super) edge: EdgeKey,
    pub(super) beveled: bool,
    pub(super) seam: bool,
    pub(super) sharp: bool,
    /// The outermost-ring vertex an unbeveled leg re-points onto — the leg's stand-in
    /// for a boundvert, and where its bridge span begins. `None` for a beveled edge
    /// (whose span starts at its trailing boundvert, which the walk reads off the ring
    /// as its crossing chain's end sample) and for a leg whose landing the placement
    /// pass declined to pin.
    pub(super) ring_vertex: Option<VertexKey>,
}

/// One rebuilt original face's **corner run** at a consumed endpoint, with the
/// per-edge attributes mechanism (1) hands it (BEVEL.md §6.5.1 / §7.7.3): the ordered
/// outermost-ring verts the face's corner became, the face's **winding-outgoing**
/// original edge at that endpoint (the whole-block donor for every edge of the run),
/// and the AND of that face's two endpoint-incident original edges' seam / sharp bits
/// — the two passengers of the block copy that mechanism (1) overrides
/// (§6.7.1.1(a′)). Snapshotted against the pristine mesh, applied after the re-stitch.
pub(super) struct CornerRun {
    pub(super) verts: Vec<VertexKey>,
    pub(super) donor: EdgeKey,
    pub(super) seam: bool,
    pub(super) sharp: bool,
}
