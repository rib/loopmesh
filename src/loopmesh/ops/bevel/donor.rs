//! Representative-face and donor selection (BEVEL.md §6.6.1 / §6.6.1.4 / §6.6.2).

use super::*;

impl LoopMesh {
    /// The face whose winding **arrives** at `vertex` along `edge` — the boundvert's
    /// representative donor of BEVEL.md §6.6.1, and the per-corner donor the
    /// ring-spanning cap fill gives corner `i` when `v` is not seam-carrying
    /// (§5.2.1.1). `None` on the first leg of an open fan, where no face arrives.
    pub(super) fn arriving_face(&self, vertex: VertexKey, edge: EdgeKey) -> Option<FaceKey> {
        // A radial loop of `edge` carries its face's traversal of the edge, running from
        // the loop's own vertex to the next one — so the face arrives at `vertex` exactly
        // when the loop's successor sits on it. Read off the radial cycle rather than off
        // the corner's face list: this runs once per leg on every beveled corner, and the
        // face-list form allocated twice per incident face to say the same thing.
        self.iter_radial_loops(edge)
            .find(|&l| self.loop_vertex(self.loop_next(l)) == vertex)
            .map(|l| self.loop_face(l))
    }

    /// The face whose winding **departs** from `vertex` along `edge` — `next_face` of
    /// BEVEL.md §6.6.1's fan primitive, and the mirror of
    /// [`arriving_face`](Self::arriving_face). `None` on the last leg of an open fan,
    /// where no face departs.
    pub(super) fn departing_face(&self, vertex: VertexKey, edge: EdgeKey) -> Option<FaceKey> {
        self.iter_radial_loops(edge)
            .find(|&l| self.loop_vertex(l) == vertex)
            .map(|l| self.loop_face(l))
    }

    /// Does `leg` genuinely lack a flanking face on one side — the **winding-blind**
    /// incidence form of BEVEL.md §6.6.1.4 condition (1) (§6.6.1.4.0(c), the winding box)?
    ///
    /// The fan's two flanking-face slots are filled by pure *incidence*: a leg resolves
    /// both slots exactly when it carries **two or more** incident faces, whatever their
    /// winding. So condition (1) fires only when the leg carries fewer than two — a
    /// mesh-boundary leg (one incident face) or a wire leg (none). An **inconsistently
    /// wound** radial-2 edge still fills both slots and is **not** a condition-(1) seam;
    /// a fan-resolved non-manifold edge (three or more faces) likewise fills both. This is
    /// the winding-blind test the winding-sensitive
    /// [`arriving_face`](Self::arriving_face) / [`departing_face`](Self::departing_face)
    /// resolution mis-answers: those report an empty slot at one endpoint of an
    /// inconsistently wound radial-2 edge and so mis-file it as a seam.
    ///
    /// The count is short-circuited at two: a high-valence non-manifold leg never walks
    /// its whole radial cycle here.
    pub(super) fn leg_missing_flanking_face(&self, leg: EdgeKey) -> bool {
        self.iter_radial_loops(leg).take(2).count() < 2
    }

    /// A boundvert's own representative face (BEVEL.md §6.6.1's
    /// `representative_face_of_boundvert`, the no-beveled-leg branch a vertex bevel
    /// takes): the face arriving at `vertex` along the boundvert's leg `edge`, else the
    /// one departing. No tie-break — §6.6.1's per-sector grid rule feeds each sector's
    /// fill faces this positional primitive directly, and the ring-spanning tie-break of
    /// §6.6.1.4 pre-empts only the centre n-gon.
    pub(super) fn boundvert_representative(
        &self,
        vertex: VertexKey,
        edge: EdgeKey,
    ) -> Option<FaceKey> {
        self.arriving_face(vertex, edge)
            .or_else(|| self.departing_face(vertex, edge))
    }

    /// The representative face of the ring's **first** boundvert `B_0` — the donor every
    /// **(anchor)** ring-spanning row of BEVEL.md §6.6.2 names (the vertex cap, the junction
    /// patch, the endpoint / terminal fill, the odd-`segments` centre n-gon).
    ///
    /// `opening_edges` holds each sector's *opening* beveled spoke `g_k` in ring order — the
    /// array [`sector_split`] hands back once it is fed the §7.6.1.2 slot array. `B_0`'s
    /// **own** spoke is its sector's *closing* one (§7.6.1.2 fact 3), so the rep is
    /// `prev_face(g_1)`, not `prev_face(g_0)`; reading the opening spoke instead takes the
    /// *previous* sector's face — an off-by-one rotation of the donor around the ring.
    ///
    /// The `1 % count` wrap is the **terminal** (`s == 1`) arm of §7.6.1.2's scope bullet
    /// rather than a bounds guard: a terminal's single sector *closes* at its one beveled
    /// spoke `E`, so `B_0`'s own spoke is `E` itself and the rep is `prev_face(E)`. That
    /// makes the terminal donor anchor-free — the seed and the last-slot scan cannot move an
    /// origin that has only one beveled spoke to land on.
    pub(super) fn ring_first_boundvert_rep(
        &self,
        vertex: VertexKey,
        opening_edges: &[EdgeKey],
    ) -> Option<FaceKey> {
        let count = opening_edges.len();
        if count == 0 {
            return None;
        }
        self.boundvert_representative(vertex, opening_edges[1 % count])
    }

    /// The corner's positional donor (BEVEL.md §6.6.1): the representative face of the
    /// ring's first boundvert, which the fan anchor picks — the first incident face the
    /// corner's disk walk reaches (§6.6.1.2's known non-invariance).
    pub(super) fn first_incident_face(&self, vertex: VertexKey) -> Option<FaceKey> {
        self.iter_disk_edges(vertex)
            .find_map(|edge| self.iter_radial_loops(edge).next())
            .map(|l| self.loop_face(l))
    }

    /// The mesh's **blendable** loop layers — BEVEL.md §6.6.1.4.0(a)'s closed four-row list,
    /// keyed on layer *type* and never on role, name or contents: the 2-float type, the
    /// original-space auxiliary (a second 2-float type this crate does not model separately,
    /// and which carries the same threshold anyway), float RGBA, and byte RGBA. Every other
    /// loop-layer type is skipped outright — a scalar float, a 3-vector float, an integer or
    /// a boolean layer is never read here however discontinuous it is.
    ///
    /// This is the set **both** call sites of condition (2) read — §6.6.1.4.1's component
    /// partition and [`edge_is_interp_seam`](Self::edge_is_interp_seam). It is deliberately
    /// **not** the set that arms the partition: arming is the narrower one-row float2
    /// *inventory* test (§6.6.1.4.0(a)), which [`has_float2_loop_layer`](Self::has_float2_loop_layer)
    /// answers.
    ///
    /// Known divergence: §6.6.1.4.0(a) distinguishes the 2-float *layer type* from the
    /// original-space auxiliary, which stores two floats under a **distinct** type and so is
    /// blendable without arming the gate. This crate has no separate storage for that
    /// auxiliary — it is `Vec2` — so a mesh whose only float2-storage layer is the auxiliary
    /// arms the gate where the reference does not. Tracked in `bmesh-parity/FOLLOWUPS.md`.
    pub(super) fn blendable_loop_layers(&self) -> Vec<BlendableLayer> {
        self.loop_attribs().blendable_layers()
    }

    /// Does the mesh's loop-layer **inventory** carry at least one layer of the 2-float type
    /// — BEVEL.md §6.6.1.4.0(a)'s float2 gate, which arms `degenerate_uv_image` and
    /// §6.6.1.4.1's whole component partition?
    ///
    /// Decided by the layer's **type alone**: not its role, not its name, not its contents.
    /// A float2 loop layer roled as anything at all arms the gate, which is why this is a
    /// distinct predicate rather than a query for the *active UV* layer — the latter's role
    /// preference is a different question and would silently un-arm the partition.
    pub(super) fn has_float2_loop_layer(&self) -> bool {
        let attribs = self.loop_attribs();
        attribs
            .iter_descriptors()
            .any(|(key, _)| matches!(attribs.storage_of(key), Some(AttributeStorage::Vec2(_))))
    }

    /// Do two corners agree on one blendable layer? The predicate itself is the class's
    /// single one — see
    /// [`ElementAttributes::slots_agree_on_blendable_layer`] — shared verbatim with `INSET.md` §6.3's corner merge, as §6.6.1.4.0(b) requires.
    pub(super) fn loops_agree_on_blendable_layer(
        &self,
        layer: BlendableLayer,
        left: LoopKey,
        right: LoopKey,
    ) -> bool {
        self.loop_attribs()
            .slots_agree_on_blendable_layer(layer, left.as_usize(), right.as_usize())
    }

    /// The **topological half** of BEVEL.md §6.6.1.4's seam condition (2): the four corners
    /// of `edge`'s two flanking faces, paired by the endpoint they share, or `None` when the
    /// edge fails the pairing outright and is a seam on that ground alone whatever any layer
    /// holds. It fails when:
    ///
    /// - the edge does not present **exactly two** loops (a boundary or wire edge flanks
    ///   nothing; a fan-resolved non-manifold edge carries three or more), or
    /// - its two loops run the **same** direction along it — inconsistently wound flanks.
    ///
    /// Consistent winding is what makes the pairing well-defined: one flank's loop and the
    /// *other* flank's `loop_next` sit on the same endpoint. Both callers of condition (2)
    /// share this walk **and** the layer set they compare the pairs on — the blendable set
    /// of §6.6.1.4.0(a), read under §6.6.1.4.0(b)'s tolerant compare. Condition (2) is one
    /// predicate; the edge-level seam test and §6.6.1.4.1's component partition are two call
    /// sites of it, differing only in the zero-loop-layer short-circuit the seam test opens
    /// with.
    pub(super) fn paired_corners(&self, edge: EdgeKey) -> Option<[(LoopKey, LoopKey); 2]> {
        let mut radial = self.iter_radial_loops(edge);
        let first = radial.next()?;
        let second = radial.next()?;
        if radial.next().is_some() {
            return None;
        }
        if self.loop_vertex(first) == self.loop_vertex(second) {
            return None;
        }
        Some([
            (first, self.loop_next(second)),
            (self.loop_next(first), second),
        ])
    }

    /// The face across `edge` from `face` when `edge` is **loop-contiguous** — the negation
    /// of BEVEL.md §6.6.1.4's seam condition (2), read over the **blendable set**
    /// (§6.6.1.4.0(a)) §6.6.1.4.1's partition is built from. `None` when the flood must not
    /// cross: either [`paired_corners`](Self::paired_corners) rejects the edge, or a
    /// blendable layer disagrees on one of the pairs.
    ///
    /// Colours are in that set, so a colour-only discontinuity does split components — on a
    /// mesh that carries a float2 layer somewhere to arm the partition at all.
    pub(super) fn interp_contiguous_neighbor(
        &self,
        face: FaceKey,
        edge: EdgeKey,
        blendable_layers: &[BlendableLayer],
    ) -> Option<FaceKey> {
        let pairs = self.paired_corners(edge)?;
        for &layer in blendable_layers {
            if pairs
                .iter()
                .any(|&(left, right)| !self.loops_agree_on_blendable_layer(layer, left, right))
            {
                return None;
            }
        }

        let (first, _) = pairs[0];
        let (_, second) = pairs[1];
        let far = if self.loop_face(first) == face {
            second
        } else {
            first
        };
        Some(self.loop_face(far))
    }

    /// The stage-1 donor of a **ring-spanning** fill at corner `vertex` — the vertex cap
    /// (BEVEL.md §5), the junction patch (§7.6.3) and the terminal / endpoint fill (§7.5),
    /// the row family whose donor §6.6.1 pins to the representative face of the ring's
    /// *first* boundvert. Two regimes:
    ///
    /// - **Seam-free corner** — every leg's fan walk resolves both flanking faces, so the
    ///   donor is the rep of the ring's *first* boundvert. That is the **ring's** question,
    ///   not the disk's, so the callers that have a ring resolve it themselves
    ///   ([`ring_first_boundvert_rep`](Self::ring_first_boundvert_rep)) and enter through
    ///   [`anchored_ring_spanning_donor`](Self::anchored_ring_spanning_donor); the disk-head
    ///   fallback kept here answers only for a caller with no ring in hand, and coincides
    ///   with the ring answer by accident rather than by rule.
    /// - **Seam-carrying corner** — some incident leg is an interpolation seam under
    ///   §6.6.1.4's corner-level test, which fires under *either* independent condition:
    ///   - **Condition (1)**: the leg genuinely lacks a flanking face on one side — an
    ///     interpolation seam *unconditionally*, purely topologically, with no customdata
    ///     layer consulted, so it fires on the plainest untextured mesh; **every**
    ///     mesh-boundary leg (and every wire leg) satisfies it. Read through the
    ///     **winding-blind** fan primitive (§6.6.1.4.0(c),
    ///     [`leg_missing_flanking_face`](Self::leg_missing_flanking_face)): an
    ///     inconsistently wound radial-2 leg fills both slots by incidence and is **not**
    ///     a seam.
    ///   - **Condition (2)**: the leg's two flanking faces disagree on a **blendable**
    ///     loop layer (§6.6.1.4.0(a)) — the UV-island / painted-apart seam. This is the
    ///     only condition that can seam a *fully interior* closed-fan corner, where every
    ///     leg resolves both flanking faces. It is read through the edge-level
    ///     [`edge_is_interp_seam`](Self::edge_is_interp_seam), which carries the
    ///     layer-tolerant compare and short-circuits to "not a seam" on a zero-loop-layer
    ///     mesh — so an untextured corner never reaches condition (2) and stays positional.
    ///
    ///   Under either condition the positional pick is displaced by the deterministic
    ///   tie-break over a positionally-named candidate set, and the fan anchor drops out
    ///   of the picture entirely.
    ///
    /// **Key 1 runs whenever it is live.** The interpolation-component id (§6.6.1.4.1) is
    /// live when a float2 loop layer is present **and** `segments` is odd, and `components`
    /// then carries the mesh-wide partition. Live is not the same as separating: candidates
    /// sitting in one component tie on key 1, so keys 2–4 decide and the answer is an
    /// isomorphism invariant — the single-UV-island mesh, the common textured case. A
    /// candidate set that **straddles** components keeps the lowest-numbered one and then
    /// runs keys 2–4 *within* it; the numbering is a construction-order artefact, so that
    /// donor is reproducible against a same-order fixture rather than invariant.
    ///
    /// (That same float2 inventory gate also arms §6.6.1.4's `degenerate_uv_image` candidate filter,
    /// which is not implemented at any `segments`: a candidate whose UV image of the corner
    /// collapses to near-zero area is kept here where the reference would drop it.)
    pub(super) fn ring_spanning_donor(
        &self,
        vertex: VertexKey,
        is_beveled: &dyn Fn(EdgeKey) -> bool,
        affect: BevelAffect,
        segments: u32,
        components: Option<&InterpComponents>,
    ) -> Option<FaceKey> {
        let positional = self.first_incident_face(vertex);

        // §6.6.1.4's corner-level seam test: a corner is seam-carrying when **any**
        // incident leg is an interpolation seam, and a leg is a seam under *either*
        // independent condition.
        //
        // - **Condition (1)** — the leg's fan walk resolves no arriving face or no
        //   departing face. Unconditional and purely topological (every mesh-boundary
        //   leg, every wire leg), read through the **winding-blind** fan primitive
        //   (§6.6.1.4.0(c)) so an inconsistently wound radial-2 leg is *not* mis-filed.
        // - **Condition (2)** — the leg's two flanking faces disagree on a **blendable**
        //   loop layer (§6.6.1.4.0(a)). This is what seams a *fully interior* closed-fan
        //   corner painted apart by a colour / UV discontinuity, where condition (1)
        //   never fires. `edge_is_interp_seam` carries the layer-tolerant compare and the
        //   zero-loop-layer short-circuit, so on an untextured mesh it returns before any
        //   allocation and only condition (1) can fire.
        //
        // The seam-free corner — the common one — keeps the positional pick, so this pass
        // runs before a single candidate is built.
        if !self.corner_is_seam_carrying(vertex) {
            return positional;
        }

        // §6.6.1.4's candidate set: every leg contributes when the corner carries exactly
        // one beveled leg (an edge bevel's terminal) or when a vertex bevel runs at odd
        // `segments`; otherwise only the beveled legs do (an edge bevel's junction).
        let beveled_leg_count = self
            .iter_disk_edges(vertex)
            .filter(|&leg| is_beveled(leg))
            .count();
        let every_leg =
            beveled_leg_count == 1 || (affect == BevelAffect::Vertices && segments % 2 == 1);
        let considers = |leg: EdgeKey| every_leg || is_beveled(leg);

        let mut candidates: Vec<FaceKey> = Vec::new();
        for leg in self.iter_disk_edges(vertex) {
            if !considers(leg) {
                continue;
            }
            // Each leg first reduces its own two flanking faces to one, by the very same
            // tie-break — this is where `prev_face` stops being privileged. A wire leg
            // reduces to nothing and contributes no candidate.
            let winner = match (
                self.arriving_face(vertex, leg),
                self.departing_face(vertex, leg),
            ) {
                (Some(arriving), Some(departing)) => {
                    if self.donor_precedes(departing, arriving, components) {
                        Some(departing)
                    } else {
                        Some(arriving)
                    }
                }
                (Some(face), None) | (None, Some(face)) => Some(face),
                (None, None) => None,
            };
            if let Some(winner) = winner {
                if !candidates.contains(&winner) {
                    candidates.push(winner);
                }
            }
        }

        self.donor_tie_break(&candidates, components).or(positional)
    }

    /// The stage-1 donor of a miter junction's odd-`segments` **centre n-gon** — the single
    /// innermost polygon closing the whole mitered junction (BEVEL.md §8.9.3.1, §6.6.2's
    /// centre-n-gon row). It is a **ring-spanning** fill, so:
    ///
    /// - **Seam-free junction** — the §6.6.1.2 fan anchor: the **enlarged miter ring's first
    ///   boundvert's** rep (`ring_first_boundvert_rep`, the caller's `sector_donors[0]`). This
    ///   is the ring's *own* anchor, **not** the raw disk's first incident face
    ///   ([`ring_spanning_donor`](Self::ring_spanning_donor)'s positional pick): the miter
    ///   enlarges the ring by splitting sectors, so the two seeds differ on a
    ///   material-straddling junction, and the centre n-gon follows the enlarged ring. Anchor
    ///   divergence carries here (§6.6.1.2): not isomorphism-invariant, but reproducible given
    ///   the same ring seeding. A **material** boundary alone keeps this branch (no seam).
    /// - **Seamed junction** — the junction-wide §6.6.1.4 tie-break over the beveled legs' reps,
    ///   which is exactly [`ring_spanning_donor`](Self::ring_spanning_donor)'s seamed branch.
    pub(super) fn miter_centre_ngon_donor(
        &self,
        vertex: VertexKey,
        is_beveled: &dyn Fn(EdgeKey) -> bool,
        affect: BevelAffect,
        segments: u32,
        ring_first_boundvert_rep: Option<FaceKey>,
        components: Option<&InterpComponents>,
    ) -> Option<FaceKey> {
        self.anchored_ring_spanning_donor(
            vertex,
            is_beveled,
            affect,
            segments,
            ring_first_boundvert_rep,
            components,
        )
    }

    /// The stage-1 donor of an **(anchor)** ring-spanning fill (BEVEL.md §6.6.2's three
    /// marked rows), in the two regimes §6.6.1.2 splits them into:
    ///
    /// - **Seam-free corner** — the positional rule: the representative face of the ring's
    ///   *first* boundvert, which the caller resolves off its own ring
    ///   ([`ring_first_boundvert_rep`](Self::ring_first_boundvert_rep)). The ring's origin is
    ///   §7.6.1.2's, so this carries §6.6.1.2's documented non-invariance — reproducible given
    ///   the same construction history, not an isomorphism invariant. A **material** boundary
    ///   raises no interpolation seam, so a plain closed manifold fan stays on this branch.
    /// - **Seam-carrying corner** — §6.6.1.4's tie-break over the positionally-named candidate
    ///   set, which is [`ring_spanning_donor`](Self::ring_spanning_donor)'s seamed branch. The
    ///   anchor drops out of the picture entirely there, so the caller's ring rep is ignored.
    ///
    /// `None` is a real answer, not a failure: a boundvert that resolves no representative
    /// feeds a **template-less** mint (§6.6.1.3). On a fan-walkable input a beveled spoke
    /// always has its arriving face, so it reaches here only from the face-less star.
    pub(super) fn anchored_ring_spanning_donor(
        &self,
        vertex: VertexKey,
        is_beveled: &dyn Fn(EdgeKey) -> bool,
        affect: BevelAffect,
        segments: u32,
        ring_first_boundvert_rep: Option<FaceKey>,
        components: Option<&InterpComponents>,
    ) -> Option<FaceKey> {
        if self.corner_is_seam_carrying(vertex) {
            self.ring_spanning_donor(vertex, is_beveled, affect, segments, components)
        } else {
            ring_first_boundvert_rep
        }
    }

    /// [`straddle_quad_donor`](Self::straddle_quad_donor) read off an edge alone — the shape
    /// the strip builders need, where the near/far split has no other use at the call site.
    pub(super) fn strip_quad_donor(
        &self,
        edge: EdgeKey,
        segments: u32,
        templates: &TemplateBank,
    ) -> Option<FaceKey> {
        let (near_face, far_face) = self.edge_flanking_faces(edge);
        self.straddle_quad_donor(
            edge,
            near_face,
            far_face,
            segments,
            templates.interp_components(),
        )
    }

    /// The stage-1 donor of the **straddle quad** — the strip quad that sits *on* the
    /// centre line at odd `segments`, and the only face of the run whose donor cannot be
    /// read off the near/far split (BEVEL.md §6.6.2's strip-row scoping, §8.2.2).
    ///
    /// Two regimes, keyed on the **edge-level** form of §6.6.1.4's seam test — is *this*
    /// beveled edge an interpolation seam — rather than on the corner-level "any incident
    /// leg" form the ring-spanning fills use:
    ///
    /// - **Seam-free edge** — the near flank, exactly as at `segments = 1`, and *not* the
    ///   far flank its past-the-midpoint position suggests.
    /// - **Seamed edge** — §6.6.1.4's tie-break, applied **once** over a candidate set that
    ///   is just the two faces the quad sits between: the beveled edge's near and far
    ///   flanks. There is no per-leg reduce here and no fan anchor; the ring-spanning
    ///   two-pass construction belongs to a different row.
    ///
    /// Key 1 (§6.6.1.4.1) is live at every odd `segments` on a mesh carrying a float2 layer
    /// and leads the chain here exactly as it does in
    /// [`ring_spanning_donor`](Self::ring_spanning_donor). §6.6.1.4.1's partition splits on
    /// exactly the condition (2) this test runs, so on a float2-bearing mesh a condition-(2)
    /// seam usually does put the two flanks in different components, and key 1 alone decides
    /// between them. With no float2 layer key 1 is inert and keys 2–4 decide.
    ///
    /// The single quad at `segments = 1` is the same row and routes through here too
    /// (§6.6.1.4's per-kind scoping names it alongside the odd-`segments` centre quad); on a
    /// seam-free edge the answer is the near flank, which is the positional donor that path
    /// always took. At even `segments` no quad straddles and this is never consulted; it
    /// still answers the near flank so the field is never a live wrong value.
    pub(super) fn straddle_quad_donor(
        &self,
        edge: EdgeKey,
        near_face: Option<FaceKey>,
        far_face: Option<FaceKey>,
        segments: u32,
        components: Option<&InterpComponents>,
    ) -> Option<FaceKey> {
        if segments % 2 == 0 || !self.edge_is_interp_seam(edge) {
            return near_face;
        }

        let mut candidates: Vec<FaceKey> = Vec::with_capacity(2);
        for face in [near_face, far_face].into_iter().flatten() {
            if !candidates.contains(&face) {
                candidates.push(face);
            }
        }

        self.donor_tie_break(&candidates, components).or(near_face)
    }

    /// The stage-1 donor of an edge-bevel [V-ADJ] sector's **centre column** — the run of
    /// grid faces at `k == segments/2` that meets the centre n-gon, which at odd `segments`
    /// is the bridge strip between two sub-patches and at even `segments` does not exist
    /// (BEVEL.md §6.6.2's [V-ADJ] grid row, §8.5).
    ///
    /// Two regimes, keyed on the sector's boundvert:
    ///
    /// - The boundvert **has a beveled edge that is not an interpolation seam** — the
    ///   sector's **own** rep, exactly as every non-centre grid face of the sector takes.
    /// - Otherwise — an unbeveled leg (§6.7.4 / §7.5), a miter-split boundvert (§8.9, which
    ///   has no beveled edge of its own), or a **seamed** beveled edge — the §6.6.1.4
    ///   tie-break over a candidate set that is just the sector's own and the **next**
    ///   boundvert's reps, applied once. No fan anchor enters either way.
    ///
    /// The edge-level seam form of [`edge_is_interp_seam`](Self::edge_is_interp_seam) is the
    /// one this keys on, as against the corner-level "any incident leg" form the
    /// ring-spanning fills use. `components` leads key 1 of the tie-break exactly as it does
    /// for the strip's straddle quad.
    pub(super) fn centre_column_donor(
        &self,
        edge: EdgeKey,
        own_face: Option<FaceKey>,
        next_face: Option<FaceKey>,
        is_beveled: bool,
        components: Option<&InterpComponents>,
    ) -> Option<FaceKey> {
        if is_beveled && !self.edge_is_interp_seam(edge) {
            return own_face;
        }

        let mut candidates: Vec<FaceKey> = Vec::with_capacity(2);
        for face in [own_face, next_face].into_iter().flatten() {
            if !candidates.contains(&face) {
                candidates.push(face);
            }
        }

        self.donor_tie_break(&candidates, components).or(own_face)
    }

    /// The per-sector stage-1 donors of a **miter cap's** enlarged ring's **non-centre**
    /// [V-ADJ] grid faces (BEVEL.md §6.6.2's grid row, edge bevel), keyed on the ring's
    /// `cross_edges`. One per sector `i` in `chains` / `boundverts` order — the sector whose
    /// sub-patch is anchored at `boundverts[i]`, whose outgoing chain crosses `cross_edges[i]`.
    ///
    /// Each sector takes its **own** boundvert's positional rep: **no** seam gate and **no**
    /// tie-break enter (§6.6.2 re-routes only the centre column, a distinct slot resolved by
    /// [`miter_ring_centre_column_donors`](Self::miter_ring_centre_column_donors)). Two
    /// regimes per sector `i`:
    ///
    /// - `cross_edges[i]` is `Some(edge)` — the boundvert owns that beveled edge; its rep is
    ///   [`boundvert_representative`](Self::boundvert_representative) of it, exactly as
    ///   [`JunctionCut`]'s per-sector grid donors resolve (§6.6.1's per-sector rule).
    /// - `cross_edges[i]` is `None` — an **arc-*start*** boundvert `B_1` (§8.9.5.3). Its own
    ///   beveled edge is cleared, but it is **not** edge-less: it keeps its **leading** flank
    ///   edge `g_k` (= `cross_edges[i-1]`, the beveled edge the arriving ring edge crosses) as
    ///   a plain leg. Its rep follows §6.6.1's `first_edge` arm — `prev_face(g_k)` =
    ///   [`arriving_face`](Self::arriving_face), the **non-reflex neighbour across `g_k`** — and
    ///   is emphatically **not** `departing_face(g_k)` (`next_face`), the reflex / own-sector
    ///   face on the opposite fan side. `g_k` is a beveled (manifold) edge, so `arriving_face`
    ///   always resolves; the departing / next-edge fallbacks serve only a degenerate open fan.
    ///   The asymmetry is the whole point: the arc *end* `B_3` (a `Some` cross-edge) owns
    ///   `g_(k+1)` and takes `prev_face(g_(k+1))` = the reflex face, so exactly **one** arc end
    ///   carries the reflex block.
    ///
    /// Read pristine (arriving / departing faces need the junction's original faces intact),
    /// like every §6.6.1 donor.
    pub(super) fn miter_ring_sector_donors(
        &self,
        vertex: VertexKey,
        cross_edges: &[Option<EdgeKey>],
    ) -> Vec<Option<FaceKey>> {
        let count = cross_edges.len();
        (0..count)
            .map(|i| match cross_edges[i] {
                Some(edge) => self.boundvert_representative(vertex, edge),
                None => {
                    // §8.9.5.3 / §6.6.1 `first_edge` arm: a `None` cross-edge boundvert is an
                    // arc-*start* `B_1` — its own beveled edge is cleared, but it is **not**
                    // edge-less: it keeps its **leading** flank edge `g_k` (= `prev_edge`, the
                    // beveled edge the ring edge arriving here crosses) as a plain leg. Its rep
                    // is `prev_face(g_k)` = `arriving_face` — the non-reflex sector across `g_k`
                    // — and emphatically **not** `departing_face(g_k)` (= `next_face`), which is
                    // the reflex / own-sector face on the opposite fan side. `g_k` is a beveled
                    // (manifold) edge, so `arriving_face` always resolves; the departing / next
                    // fallbacks only serve a degenerate open-fan `g_k`.
                    let prev_edge = cross_edges[(i + count - 1) % count];
                    let next_edge = cross_edges[(i + 1) % count];
                    prev_edge
                        .and_then(|edge| self.boundvert_representative(vertex, edge))
                        .or_else(|| next_edge.and_then(|edge| self.arriving_face(vertex, edge)))
                }
            })
            .collect()
    }

    /// The per-boundvert stage-1 donor of a **miter cap's** centre column at odd `segments`
    /// (BEVEL.md §6.6.2's [V-ADJ] grid row / §8.9.3), keyed on the enlarged ring's
    /// `cross_edges`. No ring-wide uniform donor enters: each sector resolves its own.
    ///
    /// Two regimes, per boundvert `i`:
    ///
    /// - `cross_edges[i]` is `Some(edge)` — the boundvert owns that beveled edge. Its centre
    ///   column takes the sector's **own** rep when `edge` is *not* an interpolation seam,
    ///   and the §6.6.1.4 tie-break between this and the **next** boundvert's rep when it
    ///   *is* — the seam-gated pick of [`centre_column_donor`](Self::centre_column_donor).
    ///   A boundary-touching cap has both beveled legs seamed, so its beveled sectors route
    ///   through the tie-break too.
    /// - `cross_edges[i]` is `None` — an **arc-*start*** boundvert `B_1` (§8.9.5.3). Its own
    ///   beveled edge is cleared, so its centre column takes the §6.6.1.4 tie-break between
    ///   **this** boundvert's own rep and the **next** boundvert's rep *unconditionally* —
    ///   there is no beveled edge to seam-gate on (§6.6.2's centre-column row). Its **own rep**
    ///   is the neighbour across its retained leading flank edge `g_k` (= `cross_edges[i-1]`):
    ///   `prev_face(g_k)` = [`arriving_face`](Self::arriving_face) (§6.6.1's `first_edge` arm),
    ///   **not** `departing_face(g_k)` (the reflex / own-sector face). The tie-break is a
    ///   *choice* between that neighbour rep and the next boundvert's rep (§8.9.5.3's odd-`segments`
    ///   bullet — it need not return `B_1`'s own). `g_k` and the next boundvert's edge are both
    ///   beveled, so both candidates resolve on any well-formed ring.
    ///
    /// The **next** boundvert's rep is the per-sector own rep
    /// [`miter_ring_sector_donors`](Self::miter_ring_sector_donors) computes, so a beveled
    /// sector and the edge-less arm read that candidate the same way. Read pristine
    /// (arriving / departing faces need the junction's original faces intact), like every
    /// donor of §6.6.1.
    ///
    /// Both the outer- and inner-miter arc *end* `B_3` **owns** its trailing beveled edge
    /// (§8.9.2 / §8.9.3.1), so both take a `Some` cross-edge and route the seam-gated own-rep
    /// pick — own rep when the edge is a non-seam, the §6.6.1.4 tie-break when it is. Only the
    /// edge-less arc *start* `B_1` (a `None` cross-edge) tie-breaks unconditionally. Reading the
    /// inner `B_3` as edge-less and forcing it through the tie-break inverts §8.9.3.1's rule and
    /// collapses a two-material split junction's mirror sectors onto one donor.
    pub(super) fn miter_ring_centre_column_donors(
        &self,
        vertex: VertexKey,
        cross_edges: &[Option<EdgeKey>],
        components: Option<&InterpComponents>,
    ) -> Vec<Option<FaceKey>> {
        let count = cross_edges.len();
        let sector_reps = self.miter_ring_sector_donors(vertex, cross_edges);
        (0..count)
            .map(|i| {
                let next_i = (i + 1) % count;
                match cross_edges[i] {
                    Some(edge) => {
                        // §8.9.3.1: a `Some`-cross-edge boundvert is an arc *end* `B_3` (inner
                        // or outer) that **owns** its trailing beveled edge, so its centre
                        // column keeps the seam-gated own-vs-tie-break pick — own rep on a
                        // non-seam edge, the §6.6.1.4 tie-break only when the edge is a seam.
                        self.centre_column_donor(
                            edge,
                            sector_reps[i],
                            sector_reps[next_i],
                            true,
                            components,
                        )
                    }
                    None => {
                        // §8.9.5.3 odd-`segments` bullet: the arc-*start* `B_1`'s centre
                        // column routes **its own** rep — the neighbour across the leading
                        // flank `g_k` (= `prev_face(g_k)` = `arriving_face`, §6.6.1's
                        // `first_edge` arm) — through the §6.6.1.4 tie-break against the
                        // **next** boundvert's rep. The own candidate is *not*
                        // `departing_face(g_k)` (the reflex / own-sector face).
                        let prev_edge = cross_edges[(i + count - 1) % count];
                        let next_edge = cross_edges[next_i];
                        let own = prev_edge
                            .and_then(|edge| self.boundvert_representative(vertex, edge))
                            .or_else(|| {
                                next_edge.and_then(|edge| self.arriving_face(vertex, edge))
                            });
                        let next_rep =
                            next_edge.and_then(|edge| self.boundvert_representative(vertex, edge));
                        let mut candidates: Vec<FaceKey> = Vec::with_capacity(2);
                        for face in [own, next_rep].into_iter().flatten() {
                            if !candidates.contains(&face) {
                                candidates.push(face);
                            }
                        }
                        self.donor_tie_break(&candidates, components).or(own)
                    }
                }
            })
            .collect()
    }

    /// The **edge-level** form of BEVEL.md §6.6.1.4's interpolation-seam test: is `edge`
    /// itself a seam? This is what the strip's centre quad keys on, as against the
    /// corner-level "any incident leg is a seam" form of the ring-spanning fills. It is not
    /// the seam *flag* of §6.6.7 — no header bit is read.
    ///
    /// Condition (1), purely topological and layer-free: one of the edge's own two
    /// flanking-face slots is left **empty**. Note the scope — it is the edge's own slots
    /// that are read, not the endpoint's other legs, so an interior edge with *boundary
    /// endpoints* is not a seam by it (the boundary legs are, but that is the corner-level
    /// form). What satisfies it is a boundary or wire edge outright, and nothing else here.
    ///
    /// **Winding plays no part in filling the slots** (§7.8's restatement of §6.6.1.4): a
    /// radial-2 edge whose two flanks are wound **inconsistently** still resolves a face
    /// into both slots, so condition (1) does **not** fire on it and the test falls through
    /// to condition (2) — where the failed corner pairing seams it *iff* the mesh carries a
    /// loop layer. Reading the slots through an arriving/departing walk (the corner-level
    /// fan primitive) would misattribute that case to condition (1) and seam an untextured
    /// mesh the reference leaves contiguous. A radial-3-or-more edge does not "miss" a face
    /// either, so it too falls through.
    ///
    /// §6.6.1.4 also names an edge *heading a restarted fan run* at a non-manifold corner.
    /// The slots here are read off the radial cycle rather than off a materialised fan
    /// partition, so that sub-case is **not** distinguished: such an edge reads as
    /// seam-free unless one of the causes above also holds. A similar narrowing applies at
    /// the corner-level site ([`ring_spanning_donor`](Self::ring_spanning_donor)), which
    /// reads its slots through the fan primitive because there the *leg's* two flanking
    /// slots really are the fan's, and an unfilled one is the corner-level condition (1).
    ///
    /// Condition (2) short-circuits to *not a seam* the moment the mesh carries **zero**
    /// loop layers, before any topology is examined — which is why condition (1) is the
    /// only one that can fire on an untextured mesh. That count is over **every** loop
    /// layer, blendable or not (§6.6.1.4.0(a)'s short-circuit trap): a mesh carrying only an
    /// integer flag layer does *not* take it, so a failed pairing there is still a seam
    /// though not one layer value is ever read.
    ///
    /// Past the short-circuit the edge must present exactly two loops in **consistent
    /// winding**; failing that pairing is a seam on that ground alone, whatever the values.
    /// Otherwise the two flanks' corners are compared at **both** endpoints over the
    /// **blendable set** (§6.6.1.4.0(a)) under §6.6.1.4.0(b)'s tolerant per-kind compare —
    /// the same set and the same compare §6.6.1.4.1's component partition uses, because this
    /// is one normative predicate with two call sites. Colours are in that set, so a
    /// colour-only discontinuity (byte colour included) seams the edge; a scalar float, a
    /// 3-vector, an integer or a boolean layer is not, however discontinuous.
    pub(super) fn edge_is_interp_seam(&self, edge: EdgeKey) -> bool {
        if self.iter_radial_loops(edge).take(2).count() < 2 {
            return true;
        }

        if self.loop_attribs().is_empty() {
            return false;
        }

        let Some(pairs) = self.paired_corners(edge) else {
            return true;
        };
        let blendable_layers = self.blendable_loop_layers();
        pairs.into_iter().any(|(left, right)| {
            blendable_layers
                .iter()
                .any(|&layer| !self.loops_agree_on_blendable_layer(layer, left, right))
        })
    }

    /// BEVEL.md §6.6.1.4's deterministic tie-break: the smallest candidate on the first key
    /// that separates it, and on a total tie the earliest in the order the set was built.
    pub(super) fn donor_tie_break(
        &self,
        candidates: &[FaceKey],
        components: Option<&InterpComponents>,
    ) -> Option<FaceKey> {
        candidates.iter().copied().reduce(|best, candidate| {
            if self.donor_precedes(candidate, best, components) {
                candidate
            } else {
                best
            }
        })
    }

    /// Does `candidate` beat `best` on BEVEL.md §6.6.1.4's key chain? A total tie answers
    /// `false`, which keeps the earlier candidate — the guide's own last resort.
    ///
    /// Key 1 is the interpolation-component id of §6.6.1.4.1, live only where `components`
    /// is `Some` (a float2 loop layer present *and* `segments` odd). It ties — and the chain
    /// falls through to keys 2–4 *within* the winning component — whenever the two
    /// candidates share a component, which is the single-island case, and whenever either
    /// face is one the partition never saw.
    pub(super) fn donor_precedes(
        &self,
        candidate: FaceKey,
        best: FaceKey,
        components: Option<&InterpComponents>,
    ) -> bool {
        // Key 1: lowest component id.
        if let Some(partition) = components {
            if let (Some(candidate_component), Some(best_component)) = (
                partition.component_of(candidate),
                partition.component_of(best),
            ) {
                if candidate_component != best_component {
                    return candidate_component < best_component;
                }
            }
        }

        // Key 2: selected beats unselected. This reads the face's **own** header flag, not
        // its membership in the operator's input geometry.
        let candidate_selected = self.face_flags(candidate).contains(ElementFlags::SELECTED);
        let best_selected = self.face_flags(best).contains(ElementFlags::SELECTED);
        if candidate_selected != best_selected {
            return candidate_selected;
        }

        // Key 3: lowest material slot — at a material-straddling seam the *smaller* slot
        // number donates.
        let candidate_slot = self.face_material_index(candidate);
        let best_slot = self.face_material_index(best);
        if candidate_slot != best_slot {
            return candidate_slot < best_slot;
        }

        // Key 4: lowest bounding-box-centre z, then x, then y. Compared by `total_cmp`, not
        // by `<`: §6.6.1.4's global-argmin corollary rests on the key chain being a *total*
        // order, and a partial float compare loses totality the moment a coordinate is NaN
        // — the per-leg reduce would stop being a minimum and the argmin argument with it.
        let candidate_center = self.face_bounding_box_center(candidate);
        let best_center = self.face_bounding_box_center(best);
        for (candidate_axis, best_axis) in [
            (candidate_center.z, best_center.z),
            (candidate_center.x, best_center.x),
            (candidate_center.y, best_center.y),
        ] {
            match candidate_axis.total_cmp(&best_axis) {
                std::cmp::Ordering::Less => return true,
                std::cmp::Ordering::Greater => return false,
                std::cmp::Ordering::Equal => {}
            }
        }
        false
    }
}
