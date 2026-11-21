//! Wire spokes on beveled vertices (BEVEL.md §7.2.1 / §8.3.0.4).

use super::*;

impl LoopMesh {
    /// Every **wire** spoke hanging off a beveled vertex (BEVEL.md §7.2.1 / §8.3.0.4),
    /// read off the **pristine** mesh: an incident edge with no faces at all. Read here
    /// and not at the post-pass, because by then the beveled edge itself is radial-empty
    /// too — the two are only distinguishable while the mesh still stands.
    pub(super) fn snapshot_wire_spokes(&self, endpoints: &[VertexKey]) -> Vec<WireSpoke> {
        let mut spokes: Vec<WireSpoke> = Vec::new();
        for &vertex in endpoints {
            for wire in self.iter_disk_edges(vertex) {
                if self.edge_radial_iter(wire).next().is_some() {
                    continue;
                }
                let [a, b] = self.edge_vert_keys(wire);
                let far = if a == vertex { b } else { a };
                spokes.push(WireSpoke { vertex, wire, far });
            }
        }
        spokes
    }

    /// Re-attach every wire spoke of the run (BEVEL.md §8.3.0.4) — **phase D**: after every
    /// beveled vertex's ring is final, and before *any* of them is freed. Both halves of
    /// that ordering are load-bearing. A wire whose far end is beveled too has to land on
    /// **its** ring, which only exists once every ring does; and the free that follows kills
    /// the wire along with the vertex it hangs off, so a wire not re-created by then is
    /// silently destroyed — which is what a per-vertex pass does to a both-ends-beveled wire.
    ///
    /// The candidate set is the ring's **boundverts**, never the ring's whole vertex set: a
    /// profile chain's interior rung can sit strictly nearer the wire's tip than any
    /// boundvert (it lands on the consumed vertex's own old position at `segments = 2`) and
    /// is still not a candidate. That is what makes the rule `segments`-independent.
    ///
    /// The wire is visited **once per beveled endpoint**, each visit scanning the candidate
    /// product with **its own** ring outermost and keeping the first *strictly* smaller
    /// distance — so a tie goes to the earlier boundvert in ring order. Two visits that
    /// select the same pair collapse onto one edge; two visits that break an exact tie
    /// differently legitimately mint two. Dedup at the *visit*, never per wire.
    ///
    /// The re-created edge carries the original wire's whole per-edge block, so this is
    /// observationally a re-point: same endpoints-but-one, same attributes. It stays out of
    /// [`BevelResult`] — it bounds no minted bevel face, and it is not new (§8.3.0.4).
    ///
    /// The gate on both ends is **consumption**, not ring-existence. A ring is recorded for
    /// every beveled vertex, but a beveled vertex only *loses* its wire if phase E frees it,
    /// and phase E frees only the endpoints the strip build actually left loose — an endpoint
    /// still incident to unbeveled faces is kept, and its wire with it. Re-pointing there
    /// would double the wire (the original still hangs off the surviving vertex); treating a
    /// surviving *far* end as beveled is worse, since it moves the wire off a vertex that was
    /// never going to lose it. At phase D the re-stitch has run, so looseness is readable.
    pub(super) fn reattach_wire_spokes(&mut self, spokes: &[WireSpoke], rings: &BevelRings) {
        let consumed_ring = |mesh: &Self, vertex: VertexKey| {
            rings.of(vertex).filter(|_| mesh.vertex_is_loose(vertex))
        };
        // The pairs this pass has already chosen. Tracked as endpoints rather than as edge
        // keys, and never probed against the mesh: an edge that already joins the chosen
        // pair may be surviving geometry (a strip rail running parallel to the wire), and
        // folding the wire into it would both lose the wire's attributes and let the far
        // end's visit mint a duplicate.
        let mut minted: Vec<[VertexKey; 2]> = Vec::new();

        for spoke in spokes {
            let Some(ring) = consumed_ring(self, spoke.vertex) else {
                continue;
            };
            let (near_vert, far_vert) = match consumed_ring(self, spoke.far) {
                Some(far_ring) => {
                    let mut best: Option<(f32, VertexKey, VertexKey)> = None;
                    for &near in &ring.boundverts {
                        for &far in &far_ring.boundverts {
                            let distance = (self.vertex_position(near) - self.vertex_position(far))
                                .length_squared();
                            if best.is_none_or(|(shortest, _, _)| distance < shortest) {
                                best = Some((distance, near, far));
                            }
                        }
                    }
                    let (_, near, far) = best.expect("a recorded ring carries a boundvert");
                    (near, far)
                }
                None => {
                    let tip = self.vertex_position(spoke.far);
                    let mut best: Option<(f32, VertexKey)> = None;
                    for &near in &ring.boundverts {
                        let distance = (self.vertex_position(near) - tip).length_squared();
                        if best.is_none_or(|(shortest, _)| distance < shortest) {
                            best = Some((distance, near));
                        }
                    }
                    let (_, near) = best.expect("a recorded ring carries a boundvert");
                    (near, spoke.far)
                }
            };

            // The far end's own visit may have chosen this very pair already (the ordinary
            // case, where the pair argmin is strict and both visits agree): collapse the two
            // visits onto one edge. When the two visits break an exact tie differently the
            // pairs differ, nothing collapses, and the wire is legitimately minted twice.
            if minted
                .iter()
                .any(|&[a, b]| (a, b) == (near_vert, far_vert) || (a, b) == (far_vert, near_vert))
            {
                continue;
            }
            let reattached = self.create_edge([near_vert, far_vert], EdgeCreateFlags::NONE);
            // The re-created edge *is* the wire: same header word, same per-edge block —
            // crease, bevel weight, seam, sharp, every custom layer. A default-valued mint
            // here is a per-edge-customdata divergence (§7.7) that a topology-only
            // comparison would not see.
            self.edges.get_mut(reattached).eflags = self.edges.get(spoke.wire).eflags;
            self.edge_attribs
                .copy_into(reattached.as_usize(), spoke.wire.as_usize());
            minted.push([near_vert, far_vert]);
        }
    }
}

/// One beveled vertex's ring **boundverts**, in ring order (BEVEL.md §8.3.0.4 point 1):
/// `[start, end, leg₁ … leg_(n−3)]` at a terminal, `[B_0 … B_(s−1)]` at a junction. The
/// only candidates a wire spoke may re-attach to — the ring's *other* verts (a profile
/// chain's interior rungs, a fill's interior) are not boundverts and are never candidates,
/// however much nearer the wire's tip one of them may lie.
pub(super) struct BevelRing {
    pub(super) vertex: VertexKey,
    pub(super) boundverts: Vec<VertexKey>,
}

/// Collects the [`BevelRing`]s the §8.3.0.4 post-pass reads — and collects **nothing**
/// when the run carries no wire spoke at all, which is the overwhelmingly common case. A
/// ring's boundvert list is pure post-pass input: recording one per beveled vertex on a
/// wire-free bevel spends a heap allocation per beveled vertex (and a clone per junction)
/// on data nothing ever reads. The `wire_spokes` snapshot is taken before the first mint
/// on every driver, so the answer is known before any ring is built.
#[derive(Default)]
pub(super) struct BevelRings {
    pub(super) wanted: bool,
    pub(super) rings: Vec<BevelRing>,
}

impl BevelRings {
    pub(super) fn for_spokes(spokes: &[WireSpoke]) -> Self {
        Self {
            wanted: !spokes.is_empty(),
            rings: Vec::new(),
        }
    }

    /// Record one beveled vertex's ring boundverts, in ring order (the post-pass's scan
    /// origin is the ring's head: `start` at a terminal, `B_0` at a junction). The
    /// boundverts are built by the closure only when a wire spoke exists to read them.
    pub(super) fn record(
        &mut self,
        vertex: VertexKey,
        boundverts: impl FnOnce() -> Vec<VertexKey>,
    ) {
        if self.wanted {
            self.rings.push(BevelRing {
                vertex,
                boundverts: boundverts(),
            });
        }
    }

    /// The recorded ring of a beveled vertex, if it has one with any boundvert on it.
    pub(super) fn of(&self, vertex: VertexKey) -> Option<&BevelRing> {
        self.rings
            .iter()
            .find(|ring| ring.vertex == vertex && !ring.boundverts.is_empty())
    }
}

/// A **wire** spoke on a beveled vertex's disk (BEVEL.md §7.2.1): a face-less incident
/// edge. Inert during layout — it is not a leg, takes no boundvert, and does not raise the
/// valence — but not untouched, because the vertex it hangs off is consumed. Captured
/// pristine; consumed by the §8.3.0.4 post-pass.
pub(super) struct WireSpoke {
    pub(super) vertex: VertexKey,
    pub(super) wire: EdgeKey,
    pub(super) far: VertexKey,
}
