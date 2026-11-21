//! Beveled-edge strips and their winding (BEVEL.md §7.1 / §7.8.3 / §8.3.0).

use super::*;

impl LoopMesh {
    /// Build one beveled edge's `segments`-wide strip (BEVEL.md §8.3.0): `segments`
    /// quads joining the edge's two endpoint chains rung-to-rung, `p`'s sample `k`
    /// against `q`'s sample `segments − k`. Each quad shares its rung edges with
    /// the cut / third faces and its long edges with the re-stitched radial faces.
    ///
    /// The run's stage-1 donors are **split about the strip's centre line** (§6.6.2): the
    /// quads on the near flank's side inherit the near flanking face's whole block, the
    /// quads strictly past it the far flanking face's.
    pub(super) fn build_multisegment_strip(
        &mut self,
        edge: EdgeKey,
        segments: u32,
        chains: &[EdgeEndChain],
        material: i32,
        donors: StripDonorRefs<'_>,
        result: &mut BevelResult,
    ) {
        let [a, b] = self.edge_vert_keys(edge);
        let chain_a = chains.iter().find(|c| c.edge == edge && c.endpoint == a);
        let chain_b = chains.iter().find(|c| c.edge == edge && c.endpoint == b);
        let (chain_a, chain_b) = match (chain_a, chain_b) {
            (Some(a), Some(b)) => (a, b),
            _ => return,
        };

        // §7.8.3's side-label rule fixes the rung join: profile step `k` forward from
        // `P(a)` pairs with step `segments − k` forward from `P(b)` — a rail always
        // joins *opposite* labels. When both ends carry a label the pairing is read
        // straight off it, pre-geometrically, and it holds in every winding regime.
        //
        // Only when an end has no label (a junction chain, ordered by its boundvert
        // ring alone) does the join fall back to naming the side *topologically*: the
        // re-stitched original on each side of E already owns the rail joining the two
        // chains' outer samples (§6.7.5.2), so that rail's mere existence names the
        // side. There is no geometric fallback: a length compare is ill-conditioned as
        // the offset shrinks (the aligned and swapped rail sums then differ only at
        // second order in offset / |E|) and would twist the strip on a narrow bevel.
        let p_samples = chain_a.samples();
        let mut q_samples = chain_b.samples();
        let q_last = q_samples.len() - 1;
        let p_last = p_samples.len() - 1;
        // Labels mirrored between the two ends is the ordinary regime; un-mirrored is
        // §7.8.3's tied-vote twist, where neither rail is any rebuilt flank's own pair.
        let labelled_swap = match (chain_a.side_label, chain_b.side_label) {
            (SideLabel::Unlabelled, _) | (_, SideLabel::Unlabelled) => None,
            (label_a, label_b) => Some(label_a == label_b),
        };
        // Labels agreeing between the two ends means they are *un*-mirrored, which by
        // §7.8.3 is precisely the tied-vote twist: the cross-labelled pairing the rule
        // demands is then nobody's rail, so the strip has no re-stitched face to wind
        // against. Read straight off the label — re-deriving it from rail existence is
        // the "by flank-face identity" route §7.8.3 excludes, and it would leave the
        // callee's assertion agreeing with this caller by construction.
        let (swap, anchored) = match labelled_swap {
            Some(labels_unmirrored) => (labels_unmirrored, !labels_unmirrored),
            None => {
                let rail_aligned = self
                    .query_edge_exists(p_samples[0], q_samples[0])
                    .or_else(|| self.query_edge_exists(p_samples[p_last], q_samples[q_last]))
                    .is_some();
                let rail_swapped = self
                    .query_edge_exists(p_samples[0], q_samples[q_last])
                    .or_else(|| self.query_edge_exists(p_samples[p_last], q_samples[0]))
                    .is_some();
                match (rail_aligned, rail_swapped) {
                    (false, true) => (true, true),
                    // Neither pairing is railed: both of E's originals lost their rebuild,
                    // so there is nothing left to pair against — the same broken invariant
                    // `strip_runs_with_restitched_face` asserts on. Keep one deterministic
                    // handedness rather than re-arming a metric compare that a narrow
                    // offset cannot resolve.
                    (false, false) => {
                        debug_assert!(
                            false,
                            "strip has no re-stitched rail to pair its chain ends"
                        );
                        (false, false)
                    }
                    // A winding-inconsistent input across E can rail both pairings; the
                    // strip is one surface and cannot honour both, so it keeps the aligned
                    // reading.
                    _ => (false, true),
                }
            }
        };
        if swap {
            q_samples.reverse();
        }
        if p_samples.len() != segments as usize + 1 || q_samples.len() != segments as usize + 1 {
            return;
        }
        // The no-regression argument for reading the cycle off the labels is that the two
        // candidate ring orders below are exact negations of each other. They are — unless
        // a vert appears on *both* endpoints' chains, where the dedup collapses both orders
        // onto the same triple and the negation stops holding. No builder can produce that
        // (the two chains sit at opposite ends of E), so keep it checkable rather than
        // assumed.
        //
        // The negation is also what makes a seed flip *inert* wherever the rail repair
        // below finds an anchor — but only **conditionally**, on that anchor being the
        // lone face on its rail (`strip_runs_with_restitched_face`'s radial-1 assert):
        // flipping the seed flips the direction the rail is read in, and a radial-1 rail
        // answers the two directions oppositely, so `reversed` flips back and the emitted
        // cycle is unchanged. On a radial-2 rail *both* directions answer yes, the
        // compensation stops holding, and a seed change would propagate to the output.
        // Unreachable today, which is why that stays an assert rather than a branch.
        debug_assert!(
            p_samples.iter().all(|p| !q_samples.contains(p)),
            "the two endpoints' chains must be vertex-disjoint",
        );

        // §7.8.3's absolute-winding rule fixes each quad's *directed* cycle with nothing
        // but the two arcs and their `P`-end indexing consulted: index each end's arc
        // forward from that end's `P` label and quad `k` is
        // `[A_(k−1), B_(s−k+1), B_(s−k), A_k]` — cross the edge first, then step *down*
        // both arcs. With the chains already paired cross-label, that is
        // `[p_(k−1), q_(k−1), q_k, p_k]` when the anchor chain runs `P → N`, and its
        // reverse when it runs `N → P`: re-indexing *both* arcs from the other label
        // reverses the cycle (relabelling the two endpoints, by contrast, leaves it
        // alone, which is why either endpoint may anchor the numbering).
        //
        // Seeding the rule's own cycle rather than repairing an arbitrary one is what
        // makes the twisted strip come out right: there the crack leaves neither outer
        // rail owned by a rebuilt flank, so the anchor below has no input and whatever
        // handedness the seed carried survives untouched.
        //
        // The rule is endpoint-symmetric — relabelling `a ↔ b` sends quad `k` to quad
        // `segments − k + 1` with its directed cycle unchanged, so either end may anchor
        // the numbering — which is what lets an unlabelled `a` end take its seed from `b`
        // instead of falling back to a handedness nothing chose.
        //
        // What that buys is bounded, and worth stating exactly: the *pairing* above is
        // still established by the rail-existence probe on such a strip — §7.8.3's
        // explicitly-excluded flank-identity route — so what the `b`-anchored seed
        // delivers is that the **winding is right whenever the pairing is right**. It
        // does not make the pairing right. Only recording the label on junction chains
        // does that.
        let anchor_chain_starts_at_prev_side =
            match strip_winding_seed(chain_a.side_label, chain_b.side_label, swap) {
                Some(seed) => seed,
                None => {
                    // Neither end carries a label because *this crate's junction chain
                    // builders record none* — a gap on our side, not a licence from the
                    // spec: §7.8.6's junction bullet scopes out the collapse and the ring
                    // it produces, not the `P` / `N` labels, which §8.3.0 reads at both
                    // ends of every multi-segment strip whatever the endpoint's valence.
                    // Until they are recorded the seed leans entirely on the rail repair
                    // below, which needs a rail to read.
                    debug_assert!(
                        anchored,
                        "an unlabelled strip end needs a re-stitched rail to wind against",
                    );
                    false
                }
            };

        // The seed is still checked against an outer longitudinal rail (§5.1 / §6.7.5.2)
        // — see `strip_runs_with_restitched_face` — because junction ends at *both* ends
        // of E leave the arcs indexed by their boundvert ring alone, with no *recorded*
        // `P` end to read. Recorded: §8.3.0 defines the label at both ends of every
        // multi-segment strip whatever the endpoint's valence, and this crate's junction
        // chain builders simply do not carry it through. A rung edge borders the *fill*,
        // not a rebuilt original, so it is no such anchor.
        let reversed = self.strip_runs_with_restitched_face(
            &p_samples,
            &q_samples,
            anchor_chain_starts_at_prev_side,
            anchored,
        );

        for k in 1..=segments as usize {
            let ring = if anchor_chain_starts_at_prev_side {
                [
                    p_samples[k - 1],
                    q_samples[k - 1],
                    q_samples[k],
                    p_samples[k],
                ]
            } else {
                [
                    p_samples[k - 1],
                    p_samples[k],
                    q_samples[k],
                    q_samples[k - 1],
                ]
            };
            let mut distinct: Vec<VertexKey> = Vec::with_capacity(4);
            for &v in &ring {
                if !distinct.contains(&v) {
                    distinct.push(v);
                }
            }
            if distinct.len() < 3 {
                continue;
            }
            let mut quad = distinct;
            if reversed {
                quad.reverse();
            }
            let face = self.face_create_verts(&quad, FaceCreateFlags::NONE);
            self.assign_bevel_face_attributes(
                face,
                material,
                donors.quad(k, segments as usize),
                BevelNewFaceKind::Strip,
            );
            result.faces.push(face);
            self.collect_face_edges(face, result);
        }
    }

    /// Whether the seeded strip order (BEVEL.md §8.3.0, §7.8.3's absolute-winding rule)
    /// would traverse an outer longitudinal rail the *same* way the re-stitched
    /// original face already traverses it — i.e. whether the strip must be reversed
    /// to keep every rail radial-2 (§5.1). `anchor_chain_starts_at_prev_side` says which
    /// cycle the caller seeded — it is set when the arc the caller anchored the numbering
    /// on runs `P → N`, giving `[p_(k−1), q_(k−1), q_k, p_k]`, and clear for its reverse.
    ///
    /// The strip's two *outer* longitudinal rails — sample 0 and sample `segments`
    /// (§6.7.5.2) — are the boundary the two re-stitched originals rebuilt along E's
    /// sides, so they are the strip's only absolute winding anchor: an interior rung
    /// or arc edge borders only new fill, and the chain's own start/end order is an
    /// artefact of which radial face of E came first. One flip decision serves the
    /// whole strip, since consecutive quads traverse their shared interior rail
    /// oppositely by construction.
    ///
    /// Under the seed `anchor_chain_starts_at_prev_side` names, quad 1 runs the sample-0
    /// rail `p_0 → q_0` and quad `segments` runs the sample-`segments` rail `q_s → p_s`
    /// (both reversed when it is clear); each must run *opposite* the re-stitched
    /// face already owning that rail, so the rail ends radial-2. The sample-0 rail
    /// decides whenever it carries a face; the sample-`segments` rail is the fallback
    /// for a collapsed chain end. Should the input mesh be winding-inconsistent across
    /// E the two rails disagree — the sample-0 rail then wins, deliberately: the strip
    /// is one surface and cannot satisfy both, so it keeps a single handedness rather
    /// than tearing.
    pub(super) fn strip_runs_with_restitched_face(
        &self,
        p_samples: &[VertexKey],
        q_samples: &[VertexKey],
        anchor_chain_starts_at_prev_side: bool,
        anchored: bool,
    ) -> bool {
        debug_assert_eq!(p_samples.len(), q_samples.len());
        debug_assert!(!p_samples.is_empty());
        let last = p_samples.len() - 1;
        let rails = if anchor_chain_starts_at_prev_side {
            [
                (p_samples[0], q_samples[0]),
                (q_samples[last], p_samples[last]),
            ]
        } else {
            [
                (q_samples[0], p_samples[0]),
                (p_samples[last], q_samples[last]),
            ]
        };
        for (from, to) in rails {
            let Some(rail) = self.query_edge_exists(from, to) else {
                continue;
            };
            let mut radial = self.edge_radial_iter(rail).peekable();
            if radial.peek().is_none() {
                continue;
            }
            // "Does *some* face run this rail `from → to`?" is only the right question
            // while the rail carries exactly the one re-stitched face: on a radial-2
            // rail one of the two faces always runs it forward, so `any` would be true
            // unconditionally and force-reverse the strip.
            debug_assert_eq!(
                self.edge_radial_iter(rail).count(),
                1,
                "the strip's winding anchor must be the lone re-stitched face on the rail",
            );
            return radial.any(|loop_key| {
                self.loop_vertex(loop_key) == from
                    && self.loop_vertex(self.loop_next(loop_key)) == to
            });
        }
        // Only edges that passed the radial-count-2 gate are beveled here, and both
        // their originals are re-stitched before any strip is built, so at least one
        // outer rail always carries a face — unless the §7.8.3 side labels came out
        // un-mirrored, where the cross-labelled rails are by construction nobody's.
        debug_assert!(
            !anchored,
            "strip has no re-stitched rail to anchor its winding"
        );
        false
    }

    /// Build the strip quad over the four rung ends in the §7.8.3 cyclic order
    /// `P(a) → N(b) → P(b) → N(a)`: sides 1→2 and 3→4 are the rails, 2→3 and 4→1 the
    /// rungs. The cycle is reversed when needed so the quad runs its first rail
    /// opposite to the re-stitched flank that already owns it — the §5.1 winding rule
    /// that keeps the shared edge radial-2.
    ///
    /// `anchor` is **one nominated flank's own pair** of rung ends — its ring vert at `a`
    /// and its ring vert at `b`. Which flank is nominated must be a fixed property of `E`,
    /// never a by-product of how the endpoints were closed: the caller always nominates
    /// `E`'s own first radial face, read before the re-stitch.
    ///
    /// The regime that makes that nomination load-bearing is the §7.8.3 **decisive-vote
    /// mis-wound** one: the labels are still mirrored, so both pairs are rails and the
    /// anchor does resolve, but the two flanks are inconsistently wound *with each other*,
    /// so each demands the opposite reversal of the emission order. Whichever flank is
    /// nominated wins, and a nomination that drifted with the record order would flip the
    /// emitted handedness for the same input.
    ///
    /// Under §7.8.3's un-mirrored labels the question does not arise at all: a
    /// like-labelled pair sits *diagonally* in `[P(a), N(b), P(b), N(a)]` — indices 0/2 or
    /// 1/3, never consecutive — so `emitted_anchor` is `None`, the anchor is irrelevant,
    /// and the emission order above stands as written. It is anchored on `E`'s first
    /// stored endpoint, which is what makes the twisted quad reproducible rather than
    /// seed-dependent.
    pub(super) fn build_strip_quad(
        &mut self,
        prev_a: VertexKey,
        next_b: VertexKey,
        prev_b: VertexKey,
        next_a: VertexKey,
        anchor: [VertexKey; 2],
    ) -> StripQuad {
        let mut ring = vec![prev_a, next_b, prev_b, next_a];

        // The nominated pair is a *side* of this cycle exactly when it is cross-labelled;
        // a like-labelled pair sits diagonally opposite and matches no consecutive pair.
        // Reading the direction out of the cycle rather than assuming rail 1 is what keeps
        // the anchor's identity fixed while its position in the emission order varies.
        let emitted_anchor = (0..ring.len()).find_map(|i| {
            let from = ring[i];
            let to = ring[(i + 1) % ring.len()];
            (anchor.contains(&from) && anchor.contains(&to) && from != to).then_some((from, to))
        });
        let anchor_rail = emitted_anchor.and_then(|(from, to)| {
            self.query_edge_exists(from, to)
                .map(|rail| (rail, from, to))
        });
        // Mirrored labels ⇒ the flank's pair is one of the strip's two rails, and the
        // re-stitched flank already owns it; un-mirrored ⇒ it is nobody's rail. Exactly
        // one rail of a pair existing is unreachable under §7.8.3.
        debug_assert_eq!(
            emitted_anchor.is_some(),
            anchor_rail.is_some(),
            "a cross-labelled anchor pair must already be railed by its re-stitched flank",
        );
        let rail_runs_forward = anchor_rail
            .and_then(|(rail, from, to)| {
                self.edge_loop_keys(rail).first().copied().map(|l| {
                    let start = self.loop_vertex(l);
                    let end = self.loop_vertex(self.loop_next(l));
                    start == from && end == to
                })
            })
            .unwrap_or(false);

        if rail_runs_forward {
            ring.reverse();
        }

        let face = self.face_create_verts(&ring, FaceCreateFlags::NONE);

        let mut edges: Vec<EdgeKey> = (0..ring.len())
            .map(|i| {
                let p = ring[i];
                let q = ring[(i + 1) % ring.len()];
                self.query_edge_exists(p, q)
                    .expect("strip quad edge exists")
            })
            .collect();
        edges.sort_by_key(|e| e.as_usize());
        edges.dedup();

        StripQuad { face, edges }
    }
}

/// Whether the near flanking face lies on the **sample-`0`** side of `edge`'s chain at
/// `anchor` — the endpoint whose chain numbers the strip's quads (BEVEL.md §8.3.0's quad
/// `k` spans that chain's samples `k − 1` and `k`). This is what orients §6.6.2's
/// centre-line donor split onto the run.
///
/// Read topologically off the pristine substitutions, not measured: the chain's two ends
/// are the boundverts the beveled edge's two flanking originals each collapse their corner
/// onto, so the flank whose substitution names an end *is* the flank that end's rail
/// borders. A geometric compare would be ill-conditioned as the offset shrinks, exactly as
/// for the rung pairing.
pub(super) fn near_flank_is_at_chain_start(
    anchor: VertexKey,
    edge: EdgeKey,
    near_face: Option<FaceKey>,
    chains: &[EdgeEndChain],
    substitutions: &[FaceSubstitution],
) -> bool {
    let Some(near_face) = near_face else {
        return true;
    };
    let Some(chain) = chains
        .iter()
        .find(|chain| chain.edge == edge && chain.endpoint == anchor)
    else {
        return true;
    };
    let Some(substitution) = substitutions
        .iter()
        .find(|s| s.vertex == anchor && s.face == near_face)
    else {
        debug_assert!(
            false,
            "a beveled edge's flanking original must have a recorded corner substitution at \
             the endpoint that anchors its strip",
        );
        return true;
    };
    if substitution.verts.contains(&chain.start) {
        return true;
    }
    if substitution.verts.contains(&chain.end) {
        return false;
    }
    debug_assert!(
        false,
        "the near flank's corner must collapse onto one of its strip chain's two ends",
    );
    true
}

/// The endpoint's two labelled rung ends as `(P(vertex), N(vertex))` (BEVEL.md §7.8.3).
/// `None` when either label is missing — a junction endpoint, or a corner the placement
/// pass declined.
pub(super) fn labelled_rung_ends(
    corner_vert: &[CornerVert],
    edge: EdgeKey,
    vertex: VertexKey,
) -> Option<(VertexKey, VertexKey)> {
    let labelled = |wanted: SideLabel| {
        corner_vert
            .iter()
            .find(|c| c.edge == edge && c.vertex == vertex && c.side_label == wanted)
            .map(|c| c.near_vert)
    };
    Some((
        labelled(SideLabel::PrevFirst)?,
        labelled(SideLabel::NextFirst)?,
    ))
}

/// The beveled edge's two **rails** on the junction path, paired by BEVEL.md §7.8.3's
/// side-label rule: a rail joins *opposite* labels, `P(a)—N(b)` and `N(a)—P(b)`, never a
/// flank's own two ring verts. `None` when *either* endpoint is unlabelled (a junction
/// end), where the pairing falls back to naming the side by flank-face identity.
///
/// That fallback is **not** justified by "the labels would have been mirrored anyway", and
/// it is not licensed by the spec either: an inconsistently-wound endpoint carrying two or
/// more beveled edges makes this return `None` for its edges, so both ends get paired by
/// flank face — §7.8.3's explicitly named wrong route. §7.8.6's *fourth* bullet ("junctions
/// are out of scope") does **not** cover it: that bullet scopes out the §7.8.2 collapse and
/// the boundvert ring it produces, and says in as many words that the `P` / `N` labels are
/// defined at every beveled endpoint of any valence — §8.3.0 restates it. So the hole is
/// ours: recording the label on junction boundverts is what closes it.
pub(super) fn labelled_strip_rails(
    corner_vert: &[CornerVert],
    edge: EdgeKey,
    a: VertexKey,
    b: VertexKey,
) -> Option<[(VertexKey, VertexKey); 2]> {
    let (prev_a, next_a) = labelled_rung_ends(corner_vert, edge, a)?;
    let (prev_b, next_b) = labelled_rung_ends(corner_vert, edge, b)?;
    Some([(prev_a, next_b), (next_a, prev_b)])
}

/// BEVEL.md §7.8.3's absolute-winding seed for one strip: whether the arc the numbering
/// is anchored on runs `P → N`, so quad `k` comes out as `[p_(k−1), q_(k−1), q_k, p_k]`
/// rather than its reverse.
///
/// **`swap` is a precondition the caller owes**, not a description of anything checkable
/// here: it must be the reversal that leaves the two chains in *cross-label*
/// correspondence, `q_(k−1)` reading the partner end's arc backwards from the `p`
/// numbering (§7.8.3, §7.8.4). Every arm's correctness rests on that, including the arms
/// that never see a second label — where a label is missing the caller derives `swap`
/// from rail existence instead, and in the doubly-unlabelled case from nothing at all, so
/// the burden of establishing the correspondence stays entirely on the caller.
///
/// The rule is **endpoint-symmetric** — relabelling `a ↔ b` sends quad `k` to quad
/// `segments − k + 1` with its directed cycle unchanged — so either end may anchor the
/// numbering and the two derivations must agree wherever both labels exist. That is what
/// lets an unlabelled `a` end read its seed off `b`: with the chains paired cross-label,
/// the seeded cycle is the rule's own exactly when the paired `q` walk starts at `N(b)`,
/// since `q_(k−1)` then reads `B_(segments−k+1)`. Which is to say the unlabelled arms buy
/// **the right winding whenever the pairing is right** — the pairing itself is the
/// caller's precondition above, and on an unlabelled end it still comes from the rail
/// probe §7.8.3 excludes.
///
/// `None` when neither end recorded a label, which is a gap in this crate's junction
/// chain builders rather than an endpoint that has no labels — see [`SideLabel`].
pub(super) fn strip_winding_seed(
    label_a: SideLabel,
    label_b: SideLabel,
    swap: bool,
) -> Option<bool> {
    match (label_a, label_b) {
        (SideLabel::PrevFirst, _) => Some(true),
        (SideLabel::NextFirst, _) => Some(false),
        (SideLabel::Unlabelled, SideLabel::PrevFirst) => Some(swap),
        (SideLabel::Unlabelled, SideLabel::NextFirst) => Some(!swap),
        (SideLabel::Unlabelled, SideLabel::Unlabelled) => None,
    }
}

/// The new strip face and its edges (BEVEL.md §7.1 item 2).
pub(super) struct StripQuad {
    pub(super) face: FaceKey,
    pub(super) edges: Vec<EdgeKey>,
}
