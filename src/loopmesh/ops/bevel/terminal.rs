//! Terminal endpoints of a beveled edge (BEVEL.md §7.5 / §8.3.0.1 / §8.3.0.2).

use super::*;

impl LoopMesh {
    /// Place a terminal's boundverts and build its single edge's chain (BEVEL.md
    /// §8.3.0.1 / §8.3.0.2). The two E-flanking boundverts (`start`/`end`) are E's
    /// two perpendicular strip verts (the §7.5.1 leg-meet) and the chain runs
    /// between them across E. The close then splits on terminal valence:
    ///
    /// - **valence 3** (§8.3.0.1): no fill face — the surviving third face(s) absorb
    ///   the whole `segments + 1` run through the terminal.
    /// - **valence ≥ 4** (§8.3.0.2): one **fill** — one vert slid the plain offset along
    ///   each unbeveled leg not adjacent to E, and one fill over the ring of the
    ///   through-chain plus those leg verts, in the rotational order the layout gives.
    ///   Whether that ring is fanned or emitted whole is the boundvert-count rule, and it
    ///   belongs to `build_terminal_fill`; the original terminal vertex is consumed either
    ///   way (every incident face re-stitches off it). The fill is identical under both
    ///   fill methods; it is gated on "terminal", not `vmesh_method`, and is built later
    ///   (`build_terminal_fill`) once the strip and re-stitched leg faces exist.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn build_terminal_chains(
        &mut self,
        vertex: VertexKey,
        beveled: &HashSet<EdgeKey>,
        affect: BevelAffect,
        offset: f32,
        segments: u32,
        profile: f32,
        custom: Option<&CustomChains>,
        orientation: &CustomOrientation,
        templates: &mut TemplateBank,
        chains: &mut Vec<EdgeEndChain>,
        face_subst: &mut Vec<FaceSubstitution>,
        terminal_fills: &mut Vec<TerminalFill>,
        rings: &mut BevelRings,
        result: &mut BevelResult,
    ) {
        let edge = self
            .iter_disk_edges(vertex)
            .find(|&e| beveled.contains(&e))
            .expect("a terminal endpoint has its one beveled edge");

        // The caller planned every terminal of the edge set before the first mint
        // (`buildable_beveled_edges`), so each of the three preconditions below — E manifold,
        // a layout, and that layout's flanking faces being E's radial pair — is a broken
        // invariant here rather than an input shape. Bailing on one mid-pass is what would
        // strand the verts a previously built endpoint already minted.
        let radial_faces: Vec<FaceKey> = self
            .edge_loop_keys(edge)
            .iter()
            .map(|&l| self.loop_face(l))
            .collect();
        assert_eq!(
            radial_faces.len(),
            2,
            "the plan admitted only manifold beveled edges",
        );
        let face_left = radial_faces[0];
        let face_right = radial_faces[1];

        // The rotational layout is the *one* source of truth for this terminal: the two
        // flanking legs and the faces they bound with E, and the order the rest are walked
        // in — an arc on an open fan, a cycle on a closed one, and the fill's ring order
        // either way.
        let legs = self
            .terminal_leg_layout(vertex, edge)
            .expect("the terminal plan admitted this endpoint's layout");
        assert!(
            [legs.start_face, legs.end_face] == [face_left, face_right]
                || [legs.start_face, legs.end_face] == [face_right, face_left],
            "the terminal plan admitted only layouts flanked by E's radial pair",
        );
        let starts_left = legs.start_face == face_left;
        let (left_leg, right_leg) = if starts_left {
            (legs.start_leg, legs.end_leg)
        } else {
            (legs.end_leg, legs.start_leg)
        };

        let strip_for_leg = |this: &mut Self,
                             leg: EdgeKey,
                             face: FaceKey,
                             result: &mut BevelResult| {
            let position =
                this.terminal_strip_vert(vertex, edge, leg, face, leg == legs.start_leg, offset);
            // §5.2.1 step 1: minted from the terminal endpoint it belongs to.
            let strip_vert = this.mint_bevel_vertex(position, vertex);
            result.verts.push(strip_vert);
            strip_vert
        };

        let strip_left = strip_for_leg(self, left_leg, face_left, result);
        let strip_right = strip_for_leg(self, right_leg, face_right, result);

        // The chain crosses E between the two strip verts (§8.3.0.1). face_left's
        // strip vert is the chain's `start`, face_right's its `end` — matching the
        // strip builder's face_left-first ordering of E's radial faces. This is a
        // valence-≥3 terminal (any terminal endpoint reaching this point), so the
        // §8.2.2.1 terminal-far-end plane override applies: pass the terminal vertex.
        let terminal_position = self.vertex_position(vertex);
        // A terminal endpoint carries exactly one beveled edge, so §8.2.2.1's tier-2 gate
        // (`s ≥ 3`) can never open here — no corner ring needed to find that out.
        let rescue = MidRescue::tier1_only(terminal_position);
        // §8.6.3.1(6): a terminal is a run *end*, never a branch point, but its cross-section
        // still carries the run's physical flank — and at a single-edge run it is the end that
        // takes the parity flip.
        let reverse = orientation.reverse(edge, vertex);
        let (interior, cell) = self.build_chain_between(
            strip_left,
            strip_right,
            edge,
            segments,
            profile,
            custom.map(|chains| CustomChainRead::edge_sector(chains, reverse)),
            rescue,
            Some(terminal_position),
            ChainEmission::Raw,
            result,
        );

        // Each E-adjacent radial face collapses its corner to its one strip vert.
        for &(face, strip_vert) in &[(face_left, strip_left), (face_right, strip_right)] {
            set_face_substitution(face_subst, vertex, face, vec![strip_vert]);
        }

        // The fill exists exactly when there is a leg that does not flank `E` — which is a
        // statement about the *ring* the fill's boundary is walked off (`legs.middle`), and
        // so must be tested on the ring, not on the raw disk count. The two agree only
        // because the layout demands the ring cover the whole disk; testing the disk count
        // here would take this branch with an empty ring and mint the chain's interior rungs
        // with no face to absorb them (§7.5).
        if !legs.middle.is_empty() {
            // §8.3.0.2 endpoint fill: one vert slid the plain offset along each unbeveled
            // leg NOT adjacent to E, and one fill over the through-chain plus the leg
            // verts. The original terminal vertex is consumed — every incident face
            // re-stitches off it.

            // The fill is ring-spanning, so §6.6.1's rep of the ring's first boundvert
            // donates — at a terminal the boundvert whose sector *closes* at `E`, so its own
            // spoke is `E` and its rep is `prev_face(E)` (§7.6.1.2's scope bullet). Displaced
            // by §6.6.1.4's tie-break when the terminal carries a seam leg. Registered now,
            // before the re-stitch strips the terminal of its faces.
            let representative = templates.register(self.anchored_ring_spanning_donor(
                vertex,
                &|leg| beveled.contains(&leg),
                affect,
                segments,
                self.ring_first_boundvert_rep(vertex, &[edge]),
                templates.interp_components(),
            ));

            // The fill ring walks the legs in **rotational** order (§8.3.0.2's
            // leg-numbering convention): `leg₁` is the non-flanking leg following `end`
            // around the disk, and the last one precedes `start`. That is `legs.middle`,
            // open fan or closed — the per-vertex edge cycle's insertion order would list
            // them arbitrarily and emit a self-crossing n-gon at valence ≥ 5.

            // Near-corner vert per unbeveled leg: the two E-adjacent legs collapse to
            // the two flanking boundverts; each non-adjacent leg gets its own vert
            // slid the plain offset from the terminal along that leg (§7.5).
            let mut leg_near: Vec<(EdgeKey, VertexKey)> =
                vec![(left_leg, strip_left), (right_leg, strip_right)];
            let mut leg_verts: Vec<VertexKey> = Vec::with_capacity(legs.middle.len());
            for &leg in &legs.middle {
                // §8.3.0.1: the fan pivot slides the spec offset, lengthened by `√2` for a
                // concave-notch superellipse (`profile < 0.25`) or **any** custom profile so
                // the fill never cuts into the swept cross-section; every `profile ≥ 0.25`
                // superellipse shape takes the plain offset. The rule is not
                // `segments`-gated, and neither is §4.1's guard on the resulting slide.
                let slide = terminal_leg_slide_distance(offset, profile, custom.is_some());
                let leg_position = self.slide_along_incident_edge(vertex, leg, slide);
                // §5.2.1 step 1: minted from the terminal endpoint it belongs to.
                let leg_vert = self.mint_bevel_vertex(leg_position, vertex);
                result.verts.push(leg_vert);
                leg_verts.push(leg_vert);
                leg_near.push((leg, leg_vert));
            }

            // §8.3.0.4: the boundverts, in ring order — the two flanking rungs (`start`
            // first, in the disk's sense) then the leg verts. The chain's *interior* rungs
            // are ring verts but not boundverts, so they are deliberately absent: a wire
            // spoke never attaches to one, however much nearer the tip it lies.
            rings.record(vertex, || {
                let mut boundverts = if starts_left {
                    vec![strip_left, strip_right]
                } else {
                    vec![strip_right, strip_left]
                };
                boundverts.extend_from_slice(&leg_verts);
                boundverts
            });

            // Every incident face other than E's two radial faces (already collapsed
            // to their flanking boundvert) re-stitches its corner onto the near verts
            // of its two flanking legs, ordered so the leg shared with `prev` comes
            // first (§8.3.0.2 re-stitch; the §7.4 absorb widened to the fan boundary).
            let near_of = |leg: EdgeKey| -> Option<VertexKey> {
                leg_near.iter().find(|&&(l, _)| l == leg).map(|&(_, v)| v)
            };
            for face in self.vert_face_keys(vertex) {
                if face == face_left || face == face_right {
                    continue;
                }
                // The plan admitted this terminal only if every one of its faces resolves to
                // a corner on two of its disk legs, each of which has just been given a near
                // vert — so a run that cannot be laid here is a broken invariant. Skipping
                // the face while keeping the fill would leave the fill spanning geometry the
                // skipped face still covers.
                let (incoming, outgoing) = self
                    .face_corner_edges(face, vertex)
                    .expect("the terminal plan admitted only faces with a clean corner here");
                let (Some(near_in), Some(near_out)) = (near_of(incoming), near_of(outgoing)) else {
                    panic!("the terminal plan placed a near-corner vert on every leg of this face");
                };
                let mut run = vec![near_in];
                if near_out != near_in {
                    run.push(near_out);
                }
                set_face_substitution(face_subst, vertex, face, run);
            }

            // The chain was built `face_left → face_right` (the radial order, which says
            // nothing about the disk's sense), so the ring's through-chain run is that
            // list read in the direction that starts on `start_face` — reversed when the
            // disk walk enters E from the right-hand face.
            let mut boundary = Vec::with_capacity(interior.len() + 2);
            if starts_left {
                boundary.push(strip_left);
                boundary.extend_from_slice(&interior);
                boundary.push(strip_right);
            } else {
                boundary.push(strip_right);
                boundary.extend(interior.iter().rev().copied());
                boundary.push(strip_left);
            }
            terminal_fills.push(TerminalFill {
                boundary,
                leg_verts,
                representative,
                cell: Some(cell),
                custom: custom.is_some(),
            });
        } else {
            // A two-boundvert terminal still has a ring, and a wire spoke on its disk still
            // has to land on it (§8.3.0.4).
            rings.record(vertex, || {
                if starts_left {
                    vec![strip_left, strip_right]
                } else {
                    vec![strip_right, strip_left]
                }
            });

            // §8.3.0.1 valence-3 through-strip: no fill face — the surviving third
            // face(s) absorb the full chain run, ordered so the strip vert sharing
            // the incoming leg sits next to `prev` (§7.4).
            let mut forward_run = Vec::with_capacity(segments as usize + 1);
            forward_run.push(strip_left);
            forward_run.extend_from_slice(&interior);
            forward_run.push(strip_right);
            let mut reverse_run = forward_run.clone();
            reverse_run.reverse();

            for face in self.vert_face_keys(vertex) {
                if face == face_left || face == face_right {
                    continue;
                }
                // As above: the plan admitted this terminal's every face as a clean corner
                // on two disk legs, and a third face that failed to absorb the run would
                // survive the re-stitch still pointing at the consumed endpoint.
                let (incoming_edge, _) = self
                    .face_corner_edges(face, vertex)
                    .expect("the terminal plan admitted only faces with a clean corner here");
                let ordered = if incoming_edge == right_leg {
                    reverse_run.clone()
                } else {
                    forward_run.clone()
                };
                set_face_substitution(face_subst, vertex, face, ordered);
            }
        }

        chains.push(EdgeEndChain {
            edge,
            endpoint: vertex,
            start: strip_left,
            interior,
            end: strip_right,
            // §7.8.3: `strip_left` is `P(v)` exactly when the layout's *preceding*
            // sector is the one `face_left` fills.
            side_label: if starts_left {
                SideLabel::PrevFirst
            } else {
                SideLabel::NextFirst
            },
            cell: Some(cell),
            custom_reverse: reverse,
        });
    }

    /// Build one valence-≥4 terminal's endpoint fill (BEVEL.md §8.3.0.2). The fill ring
    /// is `[boundary…, leg_verts…]` — the beveled edge's whole through-chain followed by
    /// one leg vert per non-flanking leg, in disk order — and its **boundvert count**,
    /// `2 + leg_verts.len()`, is the fill's one discriminant:
    ///
    /// - **two** (valence 3) — no fill at all; the caller records none.
    /// - **three** (valence 4, one leg vert) — the ring is **fanned** from that leg vert,
    ///   into `segments` triangles: `(leg, start, peak₁) … (leg, peak_(s−1), end)`. At
    ///   `segments = 1` the run degenerates to the ring's own single triangle.
    /// - **four or more** (valence ≥ 5) — the ring is emitted **whole**, as one n-gon,
    ///   with no interior diagonals at any `segments`.
    ///
    /// The ring's *size* is not the discriminant: the valence-4 / `segments = 2` ring and
    /// the valence-5 / `segments = 1` ring are both 4-gons, and only the first is fanned.
    /// Each face winds opposite an already-built neighbour on a shared edge, so the fill
    /// stays radial-2 against the strip and the re-stitched leg faces.
    pub(super) fn build_terminal_fill(
        &mut self,
        fill: &TerminalFill,
        material: i32,
        templates: &TemplateBank,
        result: &mut BevelResult,
    ) {
        if fill.leg_verts.is_empty() {
            return;
        }
        let representative = templates.resolve(fill.representative);
        let mut ring: Vec<VertexKey> =
            Vec::with_capacity(fill.boundary.len() + fill.leg_verts.len());
        ring.extend_from_slice(&fill.boundary);
        ring.extend_from_slice(&fill.leg_verts);
        let count = ring.len();
        if count < 3 {
            return;
        }

        let mut emit = |this: &mut Self, mut cycle: Vec<VertexKey>| {
            this.wind_fill_face_opposite_neighbor(&mut cycle);
            let face = this.face_create_verts(&cycle, FaceCreateFlags::NONE);
            this.assign_bevel_face_attributes(
                face,
                material,
                representative,
                BevelNewFaceKind::Cap,
            );
            result.faces.push(face);
            this.collect_face_edges(face, result);
        };

        if fill.leg_verts.len() >= 2 {
            emit(self, ring);
            return;
        }

        // §8.3.0.2 the one profile-keyed exception — the **custom-profile un-fan**. At the
        // three-boundvert case (valence 4, one leg vert) *and only there*, under a custom
        // profile, if the lone leg vert lies within tolerance of the beveled edge's
        // terminal-far-end profile plane, the ring is emitted **whole** instead of fanned.
        // The test is the leg vert's **squared** plane distance against the **un-squared**
        // coarse epsilon `1e-4` (an effective linear tolerance of `1e-2`) — reproduce the
        // squared-vs-linear mismatch verbatim; a squared-vs-squared or fine-epsilon test
        // flips the face count.
        if fill.custom {
            if let Some(cell) = &fill.cell {
                let normal = cell.mid_plane_normal;
                if normal.length_squared() > 0.0 {
                    let leg = self.vertex_position(fill.leg_verts[0]);
                    let distance = (leg - cell.start).dot(normal);
                    if distance * distance < 1.0e-4 {
                        emit(self, ring);
                        return;
                    }
                }
            }
        }

        // Three boundverts: fan from the ring's **last** vert — its one leg vert (§8.3.0.2's
        // pivot: never a chain rung). The run is `(leg, start, peak₁) … (leg, peak_(s−1), end)`
        // — `count − 2` = `segments` triangles, every diagonal emanating from the pivot.
        let pivot = ring[count - 1];
        for step in 0..count - 2 {
            emit(self, vec![pivot, ring[step], ring[step + 1]]);
        }
    }

    /// Reverse `cycle` if it runs the same way as an already-built neighbour on a shared
    /// edge, so the fill face stays radial-2 against the strip / re-stitched leg faces
    /// (BEVEL.md §8.3.0.2 re-stitch, the share-don't-duplicate discipline of §7.6.3).
    /// Uses the first shared edge that already carries a face loop.
    pub(super) fn wind_fill_face_opposite_neighbor(&self, cycle: &mut [VertexKey]) {
        for i in 0..cycle.len() {
            let p = cycle[i];
            let q = cycle[(i + 1) % cycle.len()];
            let edge = match self.query_edge_exists(p, q) {
                Some(edge) => edge,
                None => continue,
            };
            let first_loop = match self.edge_loop_keys(edge).first().copied() {
                Some(loop_key) => loop_key,
                None => continue,
            };
            let start = self.loop_vertex(first_loop);
            let next = self.loop_vertex(self.loop_next(first_loop));
            if start == p && next == q {
                cycle.reverse();
            }
            return;
        }
    }

    /// Place the two strip verts of a terminal endpoint on the junction path
    /// (BEVEL.md §7.5), and record the per-(edge, face) corner the strip build lands
    /// on. The close itself — placement, substitutions, and the valence-≥4 endpoint
    /// fill — is [`close_terminal_endpoint`](Self::close_terminal_endpoint), shared
    /// with the isolated-edge path. Called only for endpoints the caller's up-front plan
    /// ([`terminal_close_admits`](Self::terminal_close_admits)) already admitted.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn place_terminal_strip_verts(
        &mut self,
        vertex: VertexKey,
        beveled: &HashSet<EdgeKey>,
        affect: BevelAffect,
        segments: u32,
        offset: f32,
        profile: f32,
        custom: bool,
        reconciled: &[ReconciledOffset],
        templates: &mut TemplateBank,
        corner_vert: &mut Vec<CornerVert>,
        substitutions: &mut Vec<FaceSubstitution>,
        terminal_fills: &mut Vec<TerminalFill>,
        rings: &mut BevelRings,
        result: &mut BevelResult,
    ) {
        // The single beveled edge at this terminal endpoint.
        let edge = self
            .iter_disk_edges(vertex)
            .find(|&e| beveled.contains(&e))
            .expect("a terminal endpoint has its one beveled edge");

        let radial_faces: Vec<FaceKey> = self
            .edge_loop_keys(edge)
            .iter()
            .map(|&l| self.loop_face(l))
            .collect();
        if radial_faces.len() != 2 {
            return;
        }
        let face_left = radial_faces[0];
        let face_right = radial_faces[1];

        // §6.6.1: the endpoint fill is a ring-spanning fill — the rep of the ring's first
        // boundvert, which at a terminal is the boundvert whose sector *closes* at `edge`
        // and so reads `prev_face(edge)` (§7.6.1.2's scope bullet) — or §6.6.1.4's tie-break
        // winner over every leg when the corner carries a seam leg (a terminal has one
        // beveled leg, so every leg is a candidate).
        // Registered while the terminal still has its faces — the re-stitch consumes the
        // vertex — and whatever the valence: a donor capture gated on the shape of the
        // corner is the bug class the bank exists to kill.
        let fan_template = templates.register(self.anchored_ring_spanning_donor(
            vertex,
            &|leg| beveled.contains(&leg),
            affect,
            segments,
            self.ring_first_boundvert_rep(vertex, &[edge]),
            templates.interp_components(),
        ));

        // The caller planned every terminal of the edge set before the first mint, so a
        // decline here is a broken invariant, not an input shape — and taking it as one
        // (skipping the endpoint) is what would strand the verts already minted for the
        // rest of the pass.
        let rung_ends = self
            .close_terminal_endpoint(
                vertex,
                edge,
                face_left,
                face_right,
                offset,
                profile,
                custom,
                reconciled,
                fan_template,
                substitutions,
                terminal_fills,
                rings,
                result,
            )
            .expect("the up-front terminal plan admitted this endpoint's close");

        // §7.8.3 property 1: the label is read off the endpoint's *normalised* rotational
        // order at layout time, not recovered later from which flank touches the vert —
        // which is why it must ride the record rather than be re-derived from `face`.
        for face in [face_left, face_right] {
            corner_vert.push(CornerVert {
                edge,
                vertex,
                face,
                near_vert: rung_ends.on_face(face),
                side_label: rung_ends.label_on_face(face),
            });
        }
    }

    /// Close one terminal endpoint of a beveled edge at `segments = 1` (BEVEL.md §7.5).
    /// Returns the endpoint's two strip verts **carrying their §7.8.3 side labels** — or
    /// `None` when the corner has no close at all, **decided before the first mint** by
    /// [`terminal_close_legs`](Self::terminal_close_legs) so a declined endpoint strands
    /// nothing and records no fill.
    ///
    /// Each strip vert is the §7.5.1 **leg meet**: the pairwise offset-line meet of `E`
    /// (carrying the offset) against the flanking unbeveled leg (carrying 0) — **not**
    /// the §7.3 perpendicular-in-face slide, which only coincides at a planar, square
    /// terminal and diverges at an irregular non-planar one. The two `E`-flanking legs
    /// are `E`'s two disk neighbours, which are also the legs it shares with its own
    /// radial faces.
    ///
    /// What closes the corner then depends only on the valence (§7.5):
    ///
    /// - **valence 3** — the two legs are all there is, and the single surviving face
    ///   between them **absorbs** the corner as the 1-for-2 run of both strip verts
    ///   (§7.4). No new vert, no fill face. (On a boundary terminal there is no such face
    ///   and the corner simply ends at the rim; still no fill.)
    /// - **valence ≥ 4** — every leg *not* flanking `E` receives an **on-edge vert**,
    ///   slid the plain offset along it away from the endpoint, and the corner closes
    ///   with one **endpoint fill** face over `[start, end, leg verts…]` (§8.3.0.2's
    ///   poly-fan at `segments = 1`, where `E`'s through-chain is just its two flanking
    ///   rungs). The fill is recorded, not built: it needs the strip and the re-stitched
    ///   leg faces to wind against, so the caller builds it after both exist.
    ///
    /// Either way every face at the endpoint gets a substitution run of the near-corner
    /// verts of its two legs, laid down in the face's own winding order, and the endpoint
    /// is consumed by the re-stitch.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn close_terminal_endpoint(
        &mut self,
        vertex: VertexKey,
        edge: EdgeKey,
        face_left: FaceKey,
        face_right: FaceKey,
        offset: f32,
        profile: f32,
        custom: bool,
        reconciled: &[ReconciledOffset],
        fan_template: FaceTemplate,
        substitutions: &mut Vec<FaceSubstitution>,
        terminal_fills: &mut Vec<TerminalFill>,
        rings: &mut BevelRings,
        result: &mut BevelResult,
    ) -> Option<RungEnds> {
        let legs = self.terminal_close_legs(vertex, edge)?;
        // The two `E`-flanking legs are its disk neighbours, and `face_left` / `face_right`
        // are the two faces they bound with `E` — in the radial order, which says nothing
        // about which way round the disk runs. The fill's walk is a *disk-order* statement
        // (§8.3.0.1), so it is taken off the ring, and the radial pair is only mapped back
        // onto it for the callers that key their corners by face.
        if [legs.start_face, legs.end_face] != [face_left, face_right]
            && [legs.start_face, legs.end_face] != [face_right, face_left]
        {
            return None;
        }
        // §8.8.5: each strip vert reads its edge-side's §7.6.4-reconciled offset —
        // the spec offset unless a firing slide at this edge's *other* end shifted it
        // (§8.8.4). The recipe is the §7.5.1 leg meet either way; only the offset it
        // is fed ripples, so the terminal moves at an irregular firing junction and
        // stays put at a symmetric one (where the reconciled offset equals the spec).
        let start_leg = legs.start_leg;
        let mut strip_for_leg = |this: &mut Self, leg: EdgeKey, face: FaceKey| {
            let face_offset = reconciled_offset(reconciled, edge, face, offset);
            let position =
                this.terminal_strip_vert(vertex, edge, leg, face, leg == start_leg, face_offset);
            // §5.2.1 step 1: minted from the terminal endpoint it belongs to.
            let strip_vert = this.mint_bevel_vertex(position, vertex);
            result.verts.push(strip_vert);
            strip_vert
        };
        // §8.3.0.1: `start` is the meet against `E`'s disk-*preceding* leg (it owns the
        // chain), `end` the meet against its disk-*following* one.
        let strip_start = strip_for_leg(self, legs.start_leg, legs.start_face);
        let strip_end = strip_for_leg(self, legs.end_leg, legs.end_face);

        // 1-for-1 on each of `E`'s radial faces.
        for &(face, strip_vert) in &[(legs.start_face, strip_start), (legs.end_face, strip_end)] {
            set_face_substitution(substitutions, vertex, face, vec![strip_vert]);
        }

        // §7.5 / §8.3.0.1: one on-edge vert per leg not flanking `E`, slid along that leg
        // from the terminal. The slide reads the spec offset (no §7.6.4 reconciliation —
        // this pivot is not a strip vert), lengthened by `√2` for a concave-notch
        // superellipse (`profile < 0.25`, strict) and the plain offset for every
        // `profile ≥ 0.25` shape. `legs.middle` already runs them in the disk order that
        // continues past `end`, which is the order the fill's boundary walks them in after
        // the `end` rung.
        let leg_slide = terminal_leg_slide_distance(offset, profile, custom);
        let mut leg_near: Vec<(EdgeKey, VertexKey)> =
            vec![(legs.start_leg, strip_start), (legs.end_leg, strip_end)];
        let mut leg_verts: Vec<VertexKey> = Vec::with_capacity(legs.middle.len());
        for &leg in &legs.middle {
            // §7.5 / §4.1: a plain slide, so the lengthened distance meets the
            // shared primitive's guard — the knee sits at `|leg| / √2` on the
            // concave / custom branch, well inside the range a caller sweeps.
            let leg_position = self.slide_along_incident_edge(vertex, leg, leg_slide);
            let leg_vert = self.mint_bevel_vertex(leg_position, vertex);
            result.verts.push(leg_vert);
            leg_verts.push(leg_vert);
            leg_near.push((leg, leg_vert));
        }

        // Every remaining face at the endpoint absorbs its corner as the run of its two
        // legs' near-corner verts, first the leg it arrives on, last the leg it leaves on
        // (the [`FaceSubstitution`] run-order contract, which the leg re-point reads the
        // landing vert off). At valence 3 that run is both strip verts — the §7.4 absorbed
        // corner; at valence ≥ 4 each such face runs one strip vert (or one leg vert) to
        // the next, so the fill's boundary is exactly what is left uncovered. The plan
        // admitted this endpoint only if every one of its faces resolves to a corner on
        // two of its legs, so a run that cannot be laid here is a broken invariant, not an
        // input shape — the alternative (skip the face, keep the fill) would leave the
        // fill spanning a hole the skipped face still covers.
        for face in self.vert_face_keys(vertex) {
            if face == legs.start_face || face == legs.end_face {
                continue;
            }
            let (incoming, outgoing) = self
                .face_corner_edges(face, vertex)
                .expect("the close plan admitted only faces with a clean corner here");
            let near_of = |leg: EdgeKey| leg_near.iter().find(|&&(l, _)| l == leg).map(|&(_, v)| v);
            let (Some(near_in), Some(near_out)) = (near_of(incoming), near_of(outgoing)) else {
                panic!("the close plan placed a near-corner vert on every leg of this face");
            };
            let mut run = vec![near_in];
            if near_out != near_in {
                run.push(near_out);
            }
            set_face_substitution(substitutions, vertex, face, run);
        }

        // §8.3.0.4: the ring's boundverts, in ring order — `start`, `end`, then the leg
        // verts as the disk walk reaches them. Recorded whatever the valence (a
        // two-boundvert terminal has a ring even though it has no fill), because a wire
        // spoke on this disk must land on it.
        rings.record(vertex, || {
            let mut boundverts = vec![strip_start, strip_end];
            boundverts.extend_from_slice(&leg_verts);
            boundverts
        });

        if !leg_verts.is_empty() {
            // §8.3.0.2 at `segments = 1`: the fill ring is `E`'s two flanking rungs (the
            // whole through-chain when there are no interior profile samples) plus one leg
            // vert per non-flanking leg, walked in rotational disk order — so the ring
            // order is a statement about the disk, not about the arbitrary order of `E`'s
            // two radial faces. Whether it is fanned (three boundverts) or emitted whole
            // (four or more) is `build_terminal_fill`'s boundvert-count rule.
            terminal_fills.push(TerminalFill {
                boundary: vec![strip_start, strip_end],
                leg_verts,
                representative: fan_template,
                // §8.3.0.2: at `segments = 1` the ring is a bare triangle, so fan and whole
                // coincide — the un-fan is invisible and needs no plane test.
                cell: None,
                custom,
            });
        }

        Some(RungEnds {
            prev_side: strip_start,
            next_side: strip_end,
            prev_face: legs.start_face,
            next_face: legs.end_face,
        })
    }

    /// The **layout** half of a terminal close (BEVEL.md §7.5 / §8.3.0.1): the terminal's
    /// legs in **rotational disk order**, whatever the disk's shape. `start_leg` is the leg
    /// preceding `E` around the vertex, `end_leg` the one following it, and `middle` the
    /// legs strictly between `end_leg` and `start_leg` — walked forward from `end_leg`,
    /// which is the order the §8.3.0.2 fill's boundary visits them in after its `end` rung
    /// (its leg-numbering convention: `middle`'s first entry is `leg₁`, ring-adjacent to
    /// `end`; its last is `leg_(n−3)`, ring-adjacent to `start`).
    ///
    /// The order comes off the **corner ring**, so it is rotational on an *open* fan too —
    /// a boundary terminal's legs are an **arc**, and the walk simply continues across the
    /// gap back to the arc's other end, which is the rotational order the fill's ring wants.
    /// The per-vertex edge cycle's own order is an insertion order and would list the legs
    /// arbitrarily, which emits the fill as a bow-tie n-gon at valence ≥ 5. So every builder
    /// — the `segments = 1` close and the multi-segment chain build alike — takes its
    /// ordering from here.
    ///
    /// The valence the layout counts is the **face-bearing** one (BEVEL.md §7.2.1): the
    /// incident spokes carrying at least one face. A radial-1 boundary spoke is a full leg;
    /// a radial-0 **wire** spoke is not on the disk at all — no leg, no near-corner vert,
    /// no sector, and it does not raise `n`. The wire's whole interaction with the bevel is
    /// the §8.3.0.4 post-pass, which re-points it onto the finished ring.
    ///
    /// `None` only when there is no layout at all: face-bearing valence < 3 (nothing for the
    /// strip verts to meet against), a non-manifold `E`, an open fan whose `E` sits at an arc
    /// end and so has a leg on one side only, or a faced disk the corner ring does not cover
    /// (a malformed fan). A **zero-length** `E` is admitted (§7.5.4): it lays out like any
    /// other edge and its strip verts collapse onto the endpoint in the leg meet.
    pub(super) fn terminal_leg_layout(
        &self,
        vertex: VertexKey,
        edge: EdgeKey,
    ) -> Option<TerminalLegs> {
        // A terminal meets its strip verts against the two legs flanking `E`, so a
        // face-bearing valence of 2 has no close, and a non-manifold `E` has no two-sided
        // strip to close onto. A zero-length `E` still lays out (§7.5.4).
        let faced: Vec<EdgeKey> = self
            .iter_disk_edges(vertex)
            .filter(|&spoke| self.edge_radial_iter(spoke).next().is_some())
            .collect();
        let valence = faced.len();
        if valence < 3 || self.edge_loop_keys(edge).len() != 2 {
            return None;
        }

        let corners = self.vertex_corner_ring(vertex);
        let mut disk: Vec<EdgeKey> = corners.iter().map(|corner| corner.incoming_edge).collect();
        if let Some(last) = corners.last() {
            // An open fan's walk ends on an edge no corner arrives *on*; a closed one's
            // last outgoing edge is the first corner's incoming edge, already there.
            if !disk.contains(&last.outgoing_edge) {
                disk.push(last.outgoing_edge);
            }
        }
        // The ring is a walk over *face corners*, so it reaches exactly the faced spokes —
        // which is the whole disk the layout is denominated on (§7.2.1). A faced spoke the
        // ring still cannot arrive on means a fan the walk could not chain, and there is no
        // layout to take. Widening the walk is never the fix: a wire has no face corner to
        // reach, and it is legitimately not among the legs.
        if disk.len() != valence {
            return None;
        }
        let closed = corners.len() == valence
            && faced.iter().all(|&leg| self.edge_loop_keys(leg).len() == 2);

        let count = disk.len();
        let seat = disk.iter().position(|&e| e == edge)?;
        if !closed && (seat == 0 || seat + 1 == count) {
            return None; // an open fan's `E` must still have a leg on both sides
        }
        let start_leg = disk[(seat + count - 1) % count];
        let end_leg = disk[(seat + 1) % count];
        if start_leg == end_leg || start_leg == edge || end_leg == edge {
            return None;
        }
        // The face a flanking leg bounds with `E` is the corner between them: the ring
        // enters corner `seat` on `E` and leaves on `end_leg`, and enters the previous
        // corner on `start_leg` and leaves it on `E`.
        let start_face = corners[(seat + corners.len() - 1) % corners.len()].face;
        let end_face = corners[seat % corners.len()].face;
        if start_face == end_face {
            return None;
        }

        let middle: Vec<EdgeKey> = (2..count.saturating_sub(1))
            .map(|step| disk[(seat + step) % count])
            .collect();

        Some(TerminalLegs {
            start_leg,
            end_leg,
            start_face,
            end_face,
            middle,
            disk,
        })
    }

    /// The **admission** half of a terminal close (BEVEL.md §7.5), at every `segments`: the
    /// layout, plus the preconditions the close's all-or-nothing surgery rests on, or
    /// `None` when this endpoint has no buildable close and the whole edge must be
    /// declined. Pure: it mints nothing, so both the up-front buildability filter and the
    /// close itself can ask it, and they always agree.
    ///
    /// An **open** fan is admitted like any other: a boundary terminal's fill ring is the
    /// same rotational walk, read off the arc, and the one ring edge that spans the arc's
    /// gap simply comes out as a boundary edge of the fill (§8.3.0.2's ring listing is a
    /// disk-order statement, not a cycle-of-faces one). The layout's arc-end guard already
    /// keeps `E` itself off the rim, where it would have a leg on one side only.
    ///
    /// Beyond the layout it demands that every leg it places a vert on has a direction to
    /// place it along, and that the endpoint's faces **partition** the disk's sectors: one
    /// face per corner of the ring, each resolving to a corner on two disk legs. A face that
    /// covers no ring sector — or two faces over one — would leave the fill recorded over a
    /// boundary the substitutions never opened. One bad leg or face declines the whole
    /// endpoint.
    pub(super) fn terminal_close_legs(
        &self,
        vertex: VertexKey,
        edge: EdgeKey,
    ) -> Option<TerminalLegs> {
        let vertex_position = self.vertex_position(vertex);
        let legs = self.terminal_leg_layout(vertex, edge)?;

        for &leg in [legs.start_leg, legs.end_leg]
            .iter()
            .chain(legs.middle.iter())
        {
            self.edge_direction_from(leg, vertex, vertex_position)?;
        }
        let corners = self.vertex_corner_ring(vertex);
        let faces = self.vert_face_keys(vertex);
        if faces.len() != corners.len() {
            return None;
        }
        for face in faces {
            if corners.iter().filter(|corner| corner.face == face).count() != 1 {
                return None;
            }
            let (incoming, outgoing) = self.face_corner_edges(face, vertex)?;
            if !legs.disk.contains(&incoming)
                || !legs.disk.contains(&outgoing)
                || incoming == outgoing
            {
                return None;
            }
        }

        Some(legs)
    }

    /// Whether a terminal endpoint's close can be built at all, at **every** `segments` —
    /// the [admission](Self::terminal_close_legs) *and* the caller-side precondition every
    /// close shares: the layout's two flanking faces are exactly `E`'s radial pair, which
    /// is what lets a strip corner be keyed by face. This is the one gate both builders
    /// stand on: [`close_terminal_endpoint`](Self::close_terminal_endpoint) at `segments =
    /// 1` and [`build_terminal_chains`](Self::build_terminal_chains) above it. Asked
    /// **before the first mint** by every entry point into the close (§7.5
    /// placement-before-surgery), so a declined endpoint strands nothing.
    pub(super) fn terminal_close_admits(&self, vertex: VertexKey, edge: EdgeKey) -> bool {
        let radial: Vec<FaceKey> = self
            .edge_loop_keys(edge)
            .iter()
            .map(|&loop_key| self.loop_face(loop_key))
            .collect();
        if radial.len() != 2 {
            return false;
        }
        self.terminal_close_legs(vertex, edge).is_some_and(|legs| {
            [legs.start_face, legs.end_face] == [radial[0], radial[1]]
                || [legs.start_face, legs.end_face] == [radial[1], radial[0]]
        })
    }

    /// Place one terminal strip vert as the §7.5.1 **leg meet**: the pairwise
    /// offset-line meet of the beveled edge `edge` (carrying `offset`) against the
    /// flanking unbeveled `leg` (carrying offset 0), both lines built from a
    /// **local edge-cross normal** (`leg × edge`, sign-fixed by `surface_normal`),
    /// **not** an adjacent face's plane normal. The two lines are generally skew at
    /// a non-planar terminal, and §7.5.1 resolves that pair at the closest-approach
    /// point on the **first** line — the pair's disk predecessor (`closest_approach`) —
    /// never at the midpoint of the two. So one of the endpoint's two strip verts rides
    /// its own leg's line and the other rides `E`'s offset line. The §7.5.2 span clamp
    /// then bounds the meet's *along-leg projection* to `[0, 1]`, snapping an out-of-span
    /// meet bodily onto the nearer end of the leg segment — lateral component and all,
    /// so a clamped after-`E` vert leaves `E`'s offset line entirely. Reduces exactly to
    /// the §7.3 perpendicular slide on a planar, square terminal, where the edge-cross
    /// normal coincides with the face normal.
    ///
    /// `face` is `E`'s radial face on this leg's side — the **sector face**, whose own
    /// plane normal is the §7.5.1 orientation reference that sign-fixes the edge-cross
    /// local normal (the endpoint's vertex normal is the fallback, used only when that
    /// face yields no usable normal). It is deliberately *not* an average over the
    /// endpoint's incident faces: the averaged reading agrees on every consistently-wound
    /// mesh and disagrees exactly on the mis-wound ones of §7.8, where it cancels toward
    /// the wrong hemisphere.
    ///
    /// Which side of `E` the offset line is pushed to is fixed by `leg_precedes_edge` —
    /// the pair's order around the endpoint's normalised rotational walk — and by nothing
    /// else. A disk-preceding leg makes `E` the *second* of the pair and a disk-following
    /// one makes it the *first*, which negates `E`'s perpendicular; that is what puts the
    /// two flanking legs' strip verts on opposite sides on an ordinary mesh, and what lets
    /// a mis-wound flank push into the *other* flank's half-plane, meet its leg behind the
    /// endpoint, and collapse under the §7.5.2 span clamp (§7.8.2). Steering the push
    /// toward the sector face's own interior instead would suppress that collapse.
    ///
    /// Exactly one input carries the offset (`E`); the leg carries 0.
    pub(super) fn terminal_strip_vert(
        &self,
        vertex: VertexKey,
        edge: EdgeKey,
        leg: EdgeKey,
        face: FaceKey,
        leg_precedes_edge: bool,
        offset: f32,
    ) -> Vec3 {
        let endpoint = self.vertex_position(vertex);
        // The meet reads both flanks off their far endpoints — the raw, full-length
        // directions §7.5.1 binds, and the lengths regime 1's slide clamp measures
        // against. §7.5.4: a zero-length `E` flows through unchanged and the meet's own
        // degenerate guard collapses it onto the endpoint, rather than a special branch
        // here. A directionless *leg* has no line to meet at all.
        if self.edge_direction_from(leg, vertex, endpoint).is_none() {
            return endpoint;
        }

        let placement = terminal_leg_meet(TerminalLegMeet {
            endpoint,
            edge_far: self.edge_far_position(edge, vertex),
            leg_far: self.edge_far_position(leg, vertex),
            leg_precedes_edge,
            sector_plane: SectorPlane::Face(self.sector_plane_normal(face)),
            offset,
        });

        match placement {
            // §7.5.2: the zero-offset leg must not fling the meet past its own span.
            TerminalPlacement::LineSolve(meet) => {
                self.clamp_zero_offset_terminal(meet, vertex, leg)
            }
            // §7.5.1 returns regime 1 ahead of the clamp; see `TerminalPlacement`.
            TerminalPlacement::Degenerate(meet) => meet,
        }
    }

    /// §7.5.2 zero-offset span clamp: the unbeveled leg carries offset 0, so its
    /// offset line is the leg itself and the meet can land past the leg's far
    /// endpoint. The test is on the meet's **along-leg projection**, not on
    /// containment — only one of the endpoint's two strip verts lands on the leg at
    /// all, and the projection is the same number for either end of the
    /// closest-approach segment. Out of `[0, 1]`, the meet is snapped bodily onto the
    /// nearer endpoint of the leg's segment (`vertex` … leg's other end), lateral
    /// component discarded.
    ///
    /// Only ever handed a line solve: §7.5.1 returns regime 1's closed forms before
    /// reaching here, and `TerminalPlacement` carries that distinction across.
    pub(super) fn clamp_zero_offset_terminal(
        &self,
        meet: Vec3,
        vertex: VertexKey,
        leg: EdgeKey,
    ) -> Vec3 {
        let [a, b] = self.edge_vert_keys(leg);
        let far = if a == vertex { b } else { a };
        let near_position = self.vertex_position(vertex);
        let far_position = self.vertex_position(far);
        let segment = far_position - near_position;
        let length_squared = segment.length_squared();
        if length_squared <= f32::EPSILON {
            return near_position;
        }
        let parameter = (meet - near_position).dot(segment) / length_squared;
        if parameter < 0.0 {
            near_position
        } else if parameter > 1.0 {
            far_position
        } else {
            meet
        }
    }

    /// §7.5.1's **per-sector** quantity: the plane normal of the single face
    /// filling the sector the meet is computed in. Never hoisted out of the per-sector
    /// loop and never averaged — the whole of §7.8 is the case where a mis-wound sector
    /// face and any endpoint-wide surrogate for it disagree.
    ///
    /// The face's normal is taken **directly** — §7.5.1's source 1 does not inspect what
    /// it took — so a **zero-area** face yields the zero vector rather than a substitute.
    /// That is not a defect to be repaired one level up: the regime-1 push builds its
    /// perpendicular in this plane, and a zero plane normal is what collapses the push
    /// onto the endpoint. Falling through to the vertex normal here would be §7.5.1's
    /// source 3, which is reserved for a sector with **no face at all** — a condition
    /// this signature cannot express, since `face` is a `FaceKey` and both call paths
    /// pass one of the beveled edge's two radial faces. That face-less case is
    /// representable one level down, in `SectorPlane::NoFaceNormal`, and is currently
    /// unreachable from the mesh side.
    pub(super) fn sector_plane_normal(&self, face: FaceKey) -> Vec3 {
        self.face_newell_normal(face).unwrap_or(Vec3::ZERO)
    }

    /// §7.5.1 "surface normal at the terminal" / §8.8 junction `n_v`: the outward
    /// reference used to fix the *sign* of an edge-cross local normal (§7.5.1) and
    /// the forward/reflex sign of a `loop_slide` sub-meet (§8.8.2.1). The average of
    /// the unit normals of the faces incident to `vertex`, falling back to the
    /// stored vertex normal when that average is degenerate (near-zero length).
    ///
    /// Unit, or zero when neither source yields a usable direction — the stored normal is a
    /// user-settable field, so it is renormalised rather than trusted. Callers that project
    /// onto the tangent plane rely on that (a non-unit normal makes the projection non-
    /// orthogonal); callers that only read a sign are indifferent to it.
    pub(super) fn surface_normal_at(&self, vertex: VertexKey) -> Vec3 {
        let mut accumulator = Vec3::ZERO;
        for face in self.vert_face_keys(vertex) {
            accumulator += self.face_plane_normal(face);
        }
        let length = accumulator.length();
        if length > f32::EPSILON {
            accumulator / length
        } else {
            self.vertex_normal(vertex).normalize_or_zero()
        }
    }
}

/// One valence-≥4 terminal's endpoint fill (BEVEL.md §8.3.0.2). The fill ring walks
/// the beveled edge's through-chain `boundary` (`start` → interior peaks → `end`)
/// then one `leg_vert` per unbeveled leg not adjacent to E, in disk order — `leg_verts`
/// first entry is the leg vert ring-adjacent to `end`, its last the one adjacent to
/// `start`. The ring is fanned from its lone leg vert at **three** boundverts (valence
/// 4) and emitted whole at **four or more** (valence ≥ 5); see
/// [`build_terminal_fill`](LoopMesh::build_terminal_fill). Built for both fill methods —
/// the terminal fill is vmesh-independent (gated on "terminal", not `vmesh_method`).
/// `representative` is the §6.6.1 stage-1 inherit donor captured from a
/// terminal-incident face, read at every value of `material`.
pub(super) struct TerminalFill {
    pub(super) boundary: Vec<VertexKey>,
    pub(super) leg_verts: Vec<VertexKey>,
    pub(super) representative: FaceTemplate,
    /// The chain-owning boundvert's terminal-snapped profile cell (BEVEL.md §8.2.2.1),
    /// carried so the §8.3.0.2 custom-profile un-fan can test the lone leg vert against
    /// the terminal-far-end profile plane. `None` at the `segments = 1` close, where the
    /// un-fan is invisible (the ring is a bare triangle either way).
    pub(super) cell: Option<ProfileCell>,
    /// Whether the bevel uses a custom profile (BEVEL.md §8.6.3) — the discriminant for
    /// the three-boundvert un-fan (§8.3.0.2). Keyed on the profile *type*, never its value.
    pub(super) custom: bool,
}

/// One terminal endpoint's legs in **rotational disk order** (BEVEL.md §7.5 / §8.3.0.1),
/// as [`terminal_leg_layout`](LoopMesh::terminal_leg_layout) lays them out: `start_leg`
/// precedes the beveled edge around the disk and `end_leg` follows it (bounding
/// `start_face` / `end_face` with it), and `middle` runs the legs strictly between
/// `end_leg` and `start_leg`, forward from `end_leg` — the order the §8.3.0.2 fill
/// walks them, whose first entry is `leg₁` (ring-adjacent to `end`) and whose last is
/// `leg_(n−3)` (ring-adjacent to `start`, and the fan's pivot at three boundverts).
///
/// `disk` is the whole rotational walk this was read off — the arc, on an open fan, whose
/// `middle` therefore continues past the arc's end and back around to its other end. That
/// arc order *is* the fill's ring order, so an open fan builds the same fill as a closed
/// one, with the ring edge spanning the gap left as a boundary edge.
pub(super) struct TerminalLegs {
    pub(super) start_leg: EdgeKey,
    pub(super) end_leg: EdgeKey,
    pub(super) start_face: FaceKey,
    pub(super) end_face: FaceKey,
    pub(super) middle: Vec<EdgeKey>,
    pub(super) disk: Vec<EdgeKey>,
}

/// The distance a valence-≥4 terminal's non-`E`-adjacent leg vert slides along its
/// leg (BEVEL.md §8.3.0.1). The base is the beveled edge's **specified** (unreconciled)
/// offset — the fill's fan pivot, not a strip vert, so §7.6.4 width-reconciliation never
/// reaches it. It is lengthened by `√2` — the profile plane's box diagonal — exactly
/// when the superellipse cross-section is a **concave notch** (`profile < 0.25`, strict),
/// which can bow out toward the square's far corner; every `profile ≥ 0.25` shape (flat
/// chord, round arc, square) hugs the chord and takes the plain offset. The gate is
/// strict, so `profile = 0.25` (the straight chord, §8.6.2) is a plain-offset slide.
/// A **custom** profile takes the `√2` slide unconditionally (`custom == true`): like a
/// concave notch it can reach the profile box's far corner, so the fan pivot is pushed
/// out to `offset·√2` to keep the fill clear of the swept cross-section (§8.3.0.1).
pub(super) fn terminal_leg_slide_distance(offset: f32, profile: f32, custom: bool) -> f32 {
    if custom || profile < 0.25 {
        offset * std::f32::consts::SQRT_2
    } else {
        offset
    }
}
