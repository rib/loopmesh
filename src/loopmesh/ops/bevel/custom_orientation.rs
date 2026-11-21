//! Custom-profile handedness across edge-ends and caps (BEVEL.md §8.6.3.1(6)).

use super::*;

impl LoopMesh {
    /// The §8.6.3.1(6) Regime-A orientation pass: mark every beveled **edge-end** with its
    /// `is_start_side` bit, so that each cross-section of a run of beveled edges ties its
    /// asymmetric profile's shape-start to the *same physical flank* of the running strip.
    ///
    /// The pass is three rules stacked:
    ///
    /// - **A.1** — beveled edge-ends are linked by two kinds of step (same-edge hop,
    ///   cross-vertex continuation). The cross-vertex link is **directed** and may exist one
    ///   way only, so a "run" is what one seeding *visits*, not a graph component: the sweep
    ///   walks the beveled edges in mesh order, seeds each unvisited one, and marks outward.
    /// - **A.2 / A.2.1** — the seed bit is a single world-`z` tie-break taken at the seed
    ///   edge-end's own two ring corners: `is_start_side = (start.z < end.z)`, strict, no
    ///   epsilon, exact equality folded into `false`. The seed *end* is the edge's **stored
    ///   first endpoint** — a structural datum, not a geometric one — and it is decisive
    ///   exactly where the compare ties, which is every edge running parallel to world `z`.
    /// - **A.3** — every other edge-end in the run presents the *same physical flank*, which
    ///   step to step means flipping the `start`-relative bit. See
    ///   [`LoopMesh::propagated_handedness_bit`] for why that is not a plain parity here.
    ///
    /// A.4 falls out rather than being coded: a junction whose spokes all fail A.1's contest
    /// leaves each spoke unvisited, so each is later seeded as its own run with its own
    /// determination. That is the mutually-orthogonal corner, where three spokes give three
    /// independent bits and no single constant can serve them.
    ///
    /// Only a custom profile reads the result — every analytic cross-section is
    /// diagonal-symmetric, so the bit is a no-op there — which is why the caller runs the pass
    /// only when a curve is in play.
    pub(super) fn plan_custom_orientation(
        &self,
        beveled: &[EdgeKey],
        is_beveled: &dyn Fn(EdgeKey) -> bool,
        offset: f32,
    ) -> CustomOrientation {
        // Mesh order, so a run whose continuation relation is one-way is decomposed the same
        // way on every run of the operator regardless of the caller's selection order.
        let mut sweep: Vec<EdgeKey> = beveled.to_vec();
        sweep.sort_unstable_by_key(|edge| edge.as_usize());

        let mut orientation = CustomOrientation::default();
        for &edge in &sweep {
            // A.2.1(i): the seed end is the edge's stored first endpoint. Read unnormalised —
            // normalising the pair (by index, by height, by valence) destroys the datum and
            // mirrors every run whose seed compare ties.
            let seed_endpoint = self.edge_vert_keys(edge)[0];
            if orientation
                .is_start_side
                .contains_key(&(edge, seed_endpoint))
            {
                continue;
            }
            let seed_bit = self.seed_handedness_bit(edge, seed_endpoint, is_beveled, offset);
            orientation
                .is_start_side
                .insert((edge, seed_endpoint), seed_bit);

            for first_step in [OrientationStep::CrossVertex, OrientationStep::SameEdge] {
                let mut kind = first_step;
                let mut node = (edge, seed_endpoint);
                let mut bit = seed_bit;
                while let Some(next) = self.orientation_step(node.0, node.1, kind, is_beveled) {
                    if orientation.is_start_side.contains_key(&next) {
                        break;
                    }
                    bit = self.propagated_handedness_bit(node, next, kind, bit, is_beveled, offset);
                    orientation.is_start_side.insert(next, bit);
                    node = next;
                    kind = kind.other();
                }
            }
        }

        // The unmarked default is `is_start_side = false` for *every* edge-end, i.e. bit-for-bit
        // the uniform constant this pass replaces — so an edge-end the sweep failed to visit
        // would reinstate the mirror bug on exactly that edge, invisibly. Pin the coverage here
        // rather than leave it as a two-loop argument. (A self-loop contributes one edge-end, not
        // two: its same-edge hop has nowhere to go.)
        debug_assert!(
            sweep.iter().all(|&edge| {
                self.edge_vert_keys(edge)
                    .into_iter()
                    .all(|endpoint| orientation.is_start_side.contains_key(&(edge, endpoint)))
            }),
            "every beveled edge-end must carry an orientation mark",
        );
        orientation
    }

    /// One step of §8.6.3.1(6) A.1's walk out of the edge-end `edge` at `endpoint`.
    pub(super) fn orientation_step(
        &self,
        edge: EdgeKey,
        endpoint: VertexKey,
        kind: OrientationStep,
        is_beveled: &dyn Fn(EdgeKey) -> bool,
    ) -> Option<(EdgeKey, VertexKey)> {
        match kind {
            OrientationStep::SameEdge => {
                let [a, b] = self.edge_vert_keys(edge);
                let far = if a == endpoint { b } else { a };
                (far != endpoint).then_some((edge, far))
            }
            OrientationStep::CrossVertex => self
                .orientation_continuation(edge, endpoint, is_beveled)
                .map(|next| (next, endpoint)),
        }
    }

    /// §8.6.3.1(6) A.1's cross-vertex contest: arriving at `endpoint` along `edge`, which
    /// other beveled edge there is the run's continuation?
    ///
    /// Two short-circuits precede the contest, both keyed on the vertex's **beveled** valence
    /// (its unbeveled edges never take part): valence 1 is a run terminal, and valence 2 takes
    /// its unique partner unconditionally — however sharply it folds back. That second case is
    /// the weld, and it carries no handedness decision of its own.
    ///
    /// From valence 3 up, the contest ranks the other beveled edges by how nearly each
    /// continues the incoming line straight through. Three properties of it are each easy to
    /// get wrong and each change the run decomposition:
    ///
    /// - only **strictly positive** dots are candidates, so a spoke folded back past a right
    ///   angle is not merely a poor continuation but no continuation at all — and if every
    ///   other spoke folds back, the run stops even at high valence;
    /// - the runner-up is **floored at zero**, so a lone candidate still faces the margin and
    ///   is rejected if it turns by more than about `89.13°`;
    /// - the margin is a closeness test on the **dot** values, not on the angles between them.
    ///
    /// The relation this builds is **directed**: `X` continuing to `Y` does not imply the
    /// reverse, because the field of contestants differs from each side.
    pub(super) fn orientation_continuation(
        &self,
        edge: EdgeKey,
        endpoint: VertexKey,
        is_beveled: &dyn Fn(EdgeKey) -> bool,
    ) -> Option<EdgeKey> {
        let mut contestants: SmallVec<[EdgeKey; 8]> = SmallVec::new();
        for candidate in self.iter_disk_edges(endpoint) {
            if candidate != edge && is_beveled(candidate) {
                contestants.push(candidate);
            }
        }
        match contestants.len() {
            0 => return None,
            1 => return Some(contestants[0]),
            _ => {}
        }

        let endpoint_position = self.vertex_position(endpoint);
        let direction_in =
            (endpoint_position - self.edge_far_position(edge, endpoint)).normalize_or_zero();
        let mut winner: Option<(EdgeKey, f32)> = None;
        let mut runner_up = 0.0_f32;
        for candidate in contestants {
            let direction_out = (self.edge_far_position(candidate, endpoint) - endpoint_position)
                .normalize_or_zero();
            let dot = direction_out.dot(direction_in);
            if dot <= 0.0 {
                continue;
            }
            match winner {
                Some((_, best)) if best >= dot => runner_up = runner_up.max(dot),
                Some((_, best)) => {
                    runner_up = runner_up.max(best);
                    winner = Some((candidate, dot));
                }
                None => winner = Some((candidate, dot)),
            }
        }
        let (best_edge, best_dot) = winner?;
        (best_dot - runner_up > orientation_continuation_margin()).then_some(best_edge)
    }

    /// §8.6.3.1(6) A.2 / A.2.1(ii)-(iii): the seed edge-end's own tie-break. Of the seed
    /// sector's two ring corners — as placed, before any vmesh construction — the one strictly
    /// lower in world `z` is the shape-start side. Single-precision, strict `<`, no epsilon, no
    /// secondary axis, no index fallback: a strictly higher `start` and an exact tie both
    /// answer `false`. The tie is not academic — a beveled edge parallel to world `z` has a
    /// constant-`z` cross-section and therefore ties at *both* its ends.
    pub(super) fn seed_handedness_bit(
        &self,
        edge: EdgeKey,
        endpoint: VertexKey,
        is_beveled: &dyn Fn(EdgeKey) -> bool,
        offset: f32,
    ) -> bool {
        match self.sector_ends(edge, endpoint, is_beveled, offset) {
            Some(ends) => ends.start.z < ends.end.z,
            None => false,
        }
    }

    /// §8.6.3.1(6) A.3: carry the run's handedness one walk step, from the edge-end `from` to
    /// the edge-end `to`.
    ///
    /// What A.3 conserves is a **physical flank** — every cross-section of the run ties its
    /// shape-start to the same side of the running strip — and the bit flips step to step only
    /// because each step swaps which of the sector's two ring corners is labelled `start`. The
    /// two step kinds realise that differently, and conflating them is the trap:
    ///
    /// - a **cross-vertex** step reads both sectors off the *same* junction ring, where the
    ///   two are consecutive (or, at a straight-through crossing of higher beveled valence,
    ///   symmetrically placed about the through-line) and so present their corners in opposite
    ///   roles. Mirror the ring and both sectors mirror together, so the swap holds whichever
    ///   sense the ring runs in: the bit simply flips;
    /// - a **same-edge** hop crosses to a ring built at the *other* endpoint, whose sense is
    ///   not fixed relative to this one. So the flank is carried by name — a beveled edge's two
    ///   ring corners sit against its two radial faces, one each — and re-read against the far
    ///   end's own labelling. A blind flip here is right only when the two ends happen to label
    ///   the same physical flank oppositely, which is exactly what a terminal ring need not do.
    pub(super) fn propagated_handedness_bit(
        &self,
        from: (EdgeKey, VertexKey),
        to: (EdgeKey, VertexKey),
        kind: OrientationStep,
        bit: bool,
        is_beveled: &dyn Fn(EdgeKey) -> bool,
        offset: f32,
    ) -> bool {
        if kind == OrientationStep::CrossVertex {
            return !bit;
        }
        let here = self.sector_ends(from.0, from.1, is_beveled, offset);
        let there = self.sector_ends(to.0, to.1, is_beveled, offset);
        // Both ends of a *buildable* beveled edge always name their flanks, so the fallback
        // below is dead. It has to be spelled out because the value it would take — the plain
        // alternation — is exactly the blind parity this function exists to reject, and reading
        // as a supported degenerate path would make silently reintroducing it a one-line change.
        //
        // At a terminal end, `terminal_sector_ends` declines only on a non-manifold radial pair
        // or an unavailable leg layout, and the edge would not be in the walked set at all
        // without both (the terminal admission is the layout plus the two-radial-face check, and
        // the set is shrunk to its fixed point before the pass runs). At a junction end,
        // `junction_sector_ends` declines only on an empty sector split or a beveled edge that
        // opens no sector; every beveled edge is manifold, so it is never a fan-end spoke and
        // therefore appears as some ring corner's outgoing edge, which is what opens a sector.
        debug_assert!(
            here.is_some() && there.is_some(),
            "a walked edge-end always names its two flank faces",
        );
        match (here, there) {
            (Some(here), Some(there)) => {
                let flank = if bit { here.start_face } else { here.end_face };
                flank == there.start_face
            }
            _ => !bit,
        }
    }

    /// The sector that spans `edge` at `endpoint`, as the orientation pass needs to see it:
    /// the chain's two ring corners `start` (the owner `B_k`) and `end` (`B_(k+1)`, §8.2.2.1)
    /// together with the **radial face of `edge` each of them sits against**. Placed exactly as
    /// the chain builders place them but without minting anything, so the pass can ask about an
    /// edge-end whose own endpoint has not been built yet.
    pub(super) fn sector_ends(
        &self,
        edge: EdgeKey,
        endpoint: VertexKey,
        is_beveled: &dyn Fn(EdgeKey) -> bool,
        offset: f32,
    ) -> Option<SectorEnds> {
        let beveled_valence = self
            .iter_disk_edges(endpoint)
            .filter(|&e| is_beveled(e))
            .count();
        if beveled_valence >= 2 {
            self.junction_sector_ends(edge, endpoint, is_beveled, offset)
        } else {
            self.terminal_sector_ends(edge, endpoint, offset)
        }
    }

    /// [`LoopMesh::sector_ends`] at a junction: the sector whose chain crosses `edge` is the
    /// one *preceding* the sector `edge` opens, so its owner is the ring's previous boundvert.
    /// Reading the opened sector's own boundvert as the owner rotates every mark one step
    /// around the ring. The two flank faces fall out of the same split: the owning sector
    /// *closes* on `edge`, so its last corner carries the face just before `edge` in the disk
    /// walk, and the opened sector's first corner carries the face just after it.
    pub(super) fn junction_sector_ends(
        &self,
        edge: EdgeKey,
        endpoint: VertexKey,
        is_beveled: &dyn Fn(EdgeKey) -> bool,
        offset: f32,
    ) -> Option<SectorEnds> {
        let corners = self.vertex_bevel_corner_ring(endpoint, is_beveled);
        let sectors = sector_split(&corners, is_beveled);
        let count = sectors.len();
        if count == 0 {
            return None;
        }
        let opened = sectors
            .iter()
            .position(|sector| sector.bounding_edges[0] == edge)?;
        let owner = (opened + count - 1) % count;
        let endpoint_position = self.vertex_position(endpoint);
        Some(SectorEnds {
            start: self.solve_meet_point(endpoint, endpoint_position, &sectors[owner], offset),
            end: self.solve_meet_point(endpoint, endpoint_position, &sectors[opened], offset),
            start_face: sectors[owner].corners.last()?.1,
            end_face: sectors[opened].corners.first()?.1,
        })
    }

    /// [`LoopMesh::sector_ends`] at a terminal, where the "ring" is the beveled edge's two
    /// strip verts (§7.5.1) and the chain runs from the first radial face's vert to the
    /// second's — the same `start`/`end` roles `build_terminal_chains` hands the cell.
    pub(super) fn terminal_sector_ends(
        &self,
        edge: EdgeKey,
        endpoint: VertexKey,
        offset: f32,
    ) -> Option<SectorEnds> {
        let radial_faces: Vec<FaceKey> = self
            .edge_loop_keys(edge)
            .iter()
            .map(|&l| self.loop_face(l))
            .collect();
        if radial_faces.len() != 2 {
            return None;
        }
        let [face_left, face_right] = [radial_faces[0], radial_faces[1]];
        let legs = self.terminal_leg_layout(endpoint, edge)?;
        let (left_leg, right_leg) = if legs.start_face == face_left {
            (legs.start_leg, legs.end_leg)
        } else {
            (legs.end_leg, legs.start_leg)
        };
        Some(SectorEnds {
            start: self.terminal_strip_vert(
                endpoint,
                edge,
                left_leg,
                face_left,
                left_leg == legs.start_leg,
                offset,
            ),
            end: self.terminal_strip_vert(
                endpoint,
                edge,
                right_leg,
                face_right,
                right_leg == legs.start_leg,
                offset,
            ),
            start_face: face_left,
            end_face: face_right,
        })
    }
}

/// The §8.6.3.1(6) **Regime B** handedness bit: the one uniform value every sector of a
/// *vertex*-bevel cap reads. It takes no argument because Regime B distinguishes none — a cap
/// has no beveled edge for the orientation walk to traverse, so §8.6.3.1(6) hands the whole cap
/// a single bit and deliberately declines to name its value: the value is winding-dependent,
/// fixed by the rotational sense in which an implementation feeds each sector's `start`/`end`
/// pair into the cell. §8.6.3.1(6)'s worked rim-mid table is stated under its own canonical
/// forward-feed winding (the `reverse = false` reading), and loopmesh builds each cap sector's
/// `start`/`end` in the index-reversed **mirror** of that order, so it needs the
/// **complementary** constant to land the same rim positions — chord fraction 0.375 along
/// `B_k → B_(k+1)`, not the mirrored 0.625. Do **not** "simplify" this to `false`: that
/// mirror-flips every cap sector onto the wrong fraction. Pinned by the in-crate
/// `custom_profile_vertex_cap_orientation_forms_symmetry_orbit` and cross-backend by
/// `bevel_custom_profile_vertex_cube_corner_two_segments`.
///
/// Edge-mode sectors do **not** read this — §8.6.3.1(6) Regime A admits no constant at all;
/// see [`CustomOrientation`].
pub(super) fn vertex_cap_reverse_bit() -> bool {
    true
}

/// One beveled edge-end's cross-section as §8.6.3.1(6)'s orientation pass reads it: the two
/// ring corners the chain runs between, and the radial face of the beveled edge each corner
/// sits against. The faces are what let a same-edge hop carry the run's *physical* flank across
/// to a ring whose rotational sense is settled independently.
#[derive(Debug)]
pub(super) struct SectorEnds {
    pub(super) start: Vec3,
    pub(super) end: Vec3,
    pub(super) start_face: FaceKey,
    pub(super) end_face: FaceKey,
}

/// §8.6.3.1(6) A.1's cross-vertex margin: two contestants closer than `1 − cos 10°` in **dot**
/// are too close to call and the run stops. The runner-up is floored at zero, so a lone
/// contestant is measured against `0` and must clear the margin on its own.
///
/// Derived, never transcribed: the predicate is a closeness test, so the value is consulted
/// exactly at its own decision boundary — the one place a hand-rounded decimal shows. The
/// difference is taken in `f64` and rounded once, because `1 − cos θ` cancels badly at small `θ`
/// and the same expression carried out in `f32` throughout lands ~25 ULP high.
pub(super) fn orientation_continuation_margin() -> f32 {
    (1.0 - (10.0_f64).to_radians().cos()) as f32
}

/// One step of §8.6.3.1(6) A.1's edge-end walk. The two kinds alternate, and each swaps which
/// physical flank the arriving sector calls `start` — but they do **not** carry A.3's handedness
/// the same way. A cross-vertex step stays on one ring, whose two sectors present their corners
/// in opposite roles whichever sense the ring runs in, so the bit simply flips; a same-edge hop
/// crosses to a ring whose sense is settled independently, so the flank has to be re-read by
/// name. See [`LoopMesh::propagated_handedness_bit`] — a blind parity on both kinds mirrors
/// every terminal.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum OrientationStep {
    /// To the other beveled edge-end at the same junction vertex — the near-straight
    /// continuation that wins A.1's contest.
    CrossVertex,
    /// To the other end of the same beveled edge.
    SameEdge,
}

impl OrientationStep {
    pub(super) fn other(self) -> Self {
        match self {
            OrientationStep::CrossVertex => OrientationStep::SameEdge,
            OrientationStep::SameEdge => OrientationStep::CrossVertex,
        }
    }
}

/// The §8.6.3.1(6) **Regime A** orientation marks: one `is_start_side` bit per **edge-end**,
/// keyed by the pair (beveled edge, the endpoint whose junction sector spans it).
///
/// The edge-end — not the edge — is the unit, because a beveled edge carries two cross-section
/// arcs, one owned by each of its end junctions, and the two *often* take opposite marks: where
/// the two ends' boundvert rings wind oppositely as seen along the edge, holding one **physical**
/// flank as the shape-start across the edge means flipping the `start`/`end`-relative bit. Often,
/// not always — the two rings' senses are settled independently, so A.3 carries the flank by name
/// rather than by parity across a same-edge hop ([`LoopMesh::propagated_handedness_bit`]).
///
/// An edge-end absent from the map owns no beveled-edge arc and keeps the pass's default,
/// `is_start_side = false` — which is the uniform constant the pass exists to replace, so the
/// pass asserts it has marked every edge-end it walked
/// ([`LoopMesh::plan_custom_orientation`]).
#[derive(Default)]
pub(super) struct CustomOrientation {
    pub(super) is_start_side: HashMap<(EdgeKey, VertexKey), bool>,
}

impl CustomOrientation {
    /// The `reverse` bit [`widget_chain_to_cell`] reads for the sector of `edge` at
    /// `endpoint`. `reverse = !is_start_side`, so an unmarked edge-end reads `true`.
    pub(super) fn reverse(&self, edge: EdgeKey, endpoint: VertexKey) -> bool {
        !self
            .is_start_side
            .get(&(edge, endpoint))
            .copied()
            .unwrap_or(false)
    }
}
