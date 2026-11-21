//! Junction profile chains and welds (BEVEL.md §8.2.2 / §8.3.0.3).

use super::*;

impl LoopMesh {
    /// Place a junction's boundvert ring and build each sector's profile chain
    /// (BEVEL.md §8.2 / §8.3.0). One boundvert per sector at the §7.6.2 meet; one
    /// chain per consecutive boundvert pair, crossing the beveled edge shared by
    /// the two sectors. Records the cutoff-fill data for the junction.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn build_junction_chains(
        &mut self,
        vertex: VertexKey,
        is_beveled: &dyn Fn(EdgeKey) -> bool,
        affect: BevelAffect,
        offset: f32,
        segments: u32,
        profile: f32,
        custom: Option<&CustomChains>,
        orientation: &CustomOrientation,
        emission: ChainEmission,
        templates: &mut TemplateBank,
        chains: &mut Vec<EdgeEndChain>,
        face_subst: &mut Vec<FaceSubstitution>,
        junction_cuts: &mut Vec<JunctionCut>,
        cross_welds: &mut Vec<CrossWeld>,
        result: &mut BevelResult,
    ) {
        let corners = self.vertex_bevel_corner_ring(vertex, is_beveled);
        let sectors = sector_split(&corners, is_beveled);
        let vertex_position = self.vertex_position(vertex);

        let mut boundverts: Vec<VertexKey> = Vec::with_capacity(sectors.len());
        // Per sector, the beveled edge that *opens* it (`g_k` for `B_k`).
        let mut opening_edges: Vec<EdgeKey> = Vec::with_capacity(sectors.len());

        for sector in &sectors {
            let position = self.solve_meet_point(vertex, vertex_position, sector, offset);
            // §5.2.4: not yet minted from the corner.
            let boundvert = self.create_vertex(position);
            result.verts.push(boundvert);
            boundverts.push(boundvert);
            opening_edges.push(sector.bounding_edges[0]);

            // Each sector face collapses its corner to this one boundvert (1-for-1).
            for &(_, face) in &sector.corners {
                set_face_substitution(face_subst, vertex, face, vec![boundvert]);
            }
        }

        let outward = self.junction_outward_normal(vertex);
        let count = boundverts.len();

        // §8.5.5.5(f.5): the square parallel-offset grid's centreline seeds are line-line
        // meets on the beveled edges, projected/arbitrated against the pristine corner. The
        // beveled edges and their sector faces are consumed by the re-stitch, so their
        // geometry — each spoke's direction and tip, and the per-sector reflex verdict — is
        // captured here, exactly as the pipe / band verdicts are.
        let opening_edge_directions: Vec<Vec3> = opening_edges
            .iter()
            .map(|&edge| {
                self.edge_direction_from(edge, vertex, vertex_position)
                    .unwrap_or(Vec3::ZERO)
            })
            .collect();
        let opening_edge_tips: Vec<Vec3> = opening_edges
            .iter()
            .map(|&edge| {
                let [a, b] = self.edge_vert_keys(edge);
                let far = if a == vertex { b } else { a };
                self.vertex_position(far)
            })
            .collect();
        // §8.5.5.5(f.5) case 3's entry condition, taken through §8.9.1's own classifier so the
        // seeding pass and the miters read one predicate in one frame. A sector whose flanks
        // are too degenerate to classify joins case 4's suppressed set.
        let boundvert_non_reflex: Vec<bool> = sectors
            .iter()
            .map(|sector| {
                self.classify_miter_sector(vertex, vertex_position, sector)
                    .is_some_and(|verdict| verdict.class == MiterSectorClass::NonReflex)
            })
            .collect();
        // §8.3.4.1: read the pipe-override verdict off the *original* corner before the
        // chains are built — the beveled-edge directions and incident-face normals are
        // consumed by the re-stitch. It is captured here (not only carried on the cut for
        // the fill-time routing) because it also decides the chains' emission: a ring that
        // takes the override delegates its **connected** fill to the ordinary ADJ path, and
        // that path's rim must be byte-identical to a from-scratch ADJ bevel — i.e. built on
        // the **resampled** chain, exactly the emission a plain ADJ junction gets, not the
        // raw cutoff arc. (The `(segments − 1) · s` loose leftovers are minted separately at
        // the raw arc samples; only the connected rim moves to the resampled positions.)
        let pipe_override =
            self.pipe_override_admits(vertex, vertex_position, &opening_edges, &corners);
        let chain_emission = if pipe_override {
            ChainEmission::Resampled
        } else {
            emission
        };

        // §8.5.1 / §7.6.6: the dihedral-band verdict must be read from the *original*
        // corner — its edges and faces are about to be consumed by the re-stitch — so it is
        // captured now and carried on the cut for the fill-time routing decision.
        let dihedral_band = self.junction_dihedral_band(vertex, &corners);

        // §8.5.5.5(f.4a): the **edge-mode square-in weld**. At a tri-corner-eligible ring the
        // concave profile's two straight legs make consecutive sectors' chains coincide, and
        // the fill *identifies* them rather than leaving two coincident samples: sector `i`'s
        // first half, walking in from `B_i`, **is** sector `i−1`'s second half walking out to
        // it. The verdict is taken here — one phase before the fill — because the strips are
        // laid over the chains, so the identification has to already be in place when they
        // are built; and it rides the cut afterwards so the fill's central-`s`-gon step and
        // this layout cannot disagree about which junction welded.
        //
        // The gate is (f.1) step 3's, in its order — the ring's three sectors first, then the
        // dispatch keys: `emission` names the fill method (the
        // §8.2.2.1.1 chord resample is the [V-ADJ] cap's own boundary emission, so only an
        // ADJ junction asks for it); step 3 is edge-bevel only and refuses a **custom**
        // profile outright, before any geometry, so a hand-drawn curve never welds; step 2's
        // pipe is tested before step 3, so a pipe junction never reaches the weld's row; and
        // the exponent row is the exact `r = 0` sentinel. Vertex mode falls through to
        // (f.4b)'s ordinary general grid, which is a *different object* — it keeps the ring
        // quads this deletes.
        //
        // `affect` alone is documentary — this path is only reached under an edge bevel — and
        // is kept because it makes step 3's dispatch legible. `emission` is **live**: it is
        // `Raw` under a `Cutoff` vmesh, and a `Cutoff` junction reaches here unchanged, so
        // dropping the conjunct would lay welded chains under a §8.3 flat cut, which owes no
        // weld at all. `custom` is live for the same reason — a custom bevel reaches here on
        // whatever exponent its untouched `profile` slider resolves to, `r = 0` included.
        //
        // Step 3 keys on **valence**, and its predicate is three-valued: it must *fire*, and
        // *doesn't apply* is not the same answer. A ring whose three sectors are octant-shaped
        // but whose junction carries extra unbeveled in-plane edges — so the **edge** count
        // exceeds three — is "doesn't apply", and that answer reaches the tri-corner family
        // only through step 4's shortcut, whose own gate excludes `EXP_SQUARE_IN`. At
        // square-in it therefore lands on the general grid, not here. The boundvert count
        // alone cannot see the difference: a sector absorbs its in-plane passenger, and
        // neither half of the shape test refuses one — the meet geometry reads three
        // boundverts and `v`, which the passenger does not move, and the band asks for
        // exactly `edgecount − 3` near-flat edges, which one passenger at valence 4 supplies.
        // So the edge count is asked for separately, and only exactly three fires.
        let three_sector_ring: Option<[Vec3; 3]> = (count == 3).then(|| {
            [
                self.vertex_position(boundverts[0]),
                self.vertex_position(boundverts[1]),
                self.vertex_position(boundverts[2]),
            ]
        });
        let square_in_weld = three_sector_ring
            .filter(|ring| {
                segments >= 2
                    && affect == BevelAffect::Edges
                    && emission == ChainEmission::Resampled
                    && custom.is_none()
                    && !pipe_override
                    && profile_to_exponent(profile) == 0.0
                    && junction_incident_edges(&corners).len() == 3
                    && tri_corner_shape_admits(
                        vertex_position,
                        ring,
                        dihedral_band,
                        TRI_CORNER_BAND_SMOOTH,
                    )
            })
            // The layout's own frame origin `c`: the legs are stated in octant coordinates and
            // mapped through it. Taken as the frame's translation alone rather than through
            // the conditioned frame because that frame's guard answers a different question
            // (is the map invertible) and is allowed to *decline*, which here — the phase that
            // welds the chains — would leave the legs meeting nowhere: an open boundary in a
            // closed mesh. Nothing downstream needs the point; the fill maps nothing.
            .map(|ring| unit_octant_origin(ring, vertex_position));

        if count == 2 {
            // §8.3.0.3: a two-boundvert ring bounds no region — the two sectors' chains
            // span the *same* boundvert pair in opposite directions. They are welded into
            // one chain, minted once, and both arriving strips terminate on it, so the
            // corner closes with no fill face under either method.
            self.build_welded_chain(
                vertex,
                &boundverts,
                &opening_edges,
                &corners,
                is_beveled,
                segments,
                profile,
                custom,
                orientation,
                chains,
                cross_welds,
                result,
            );
        } else if let Some(anchor) = square_in_weld {
            self.build_square_in_weld_chains(
                vertex,
                &boundverts,
                &opening_edges,
                &corners,
                is_beveled,
                segments,
                anchor,
                chains,
                result,
            );
        } else {
            // One chain per consecutive boundvert pair (B_k → B_{k+1}), crossing the
            // beveled edge shared by sectors k and k+1 (= the edge opening sector k+1).
            for k in 0..count {
                let next = (k + 1) % count;
                let crossing = opening_edges[next];
                let rescue = self.mid_rescue(vertex, crossing, &corners, is_beveled);
                // §8.6.3.1(4).4's per-consumer table: a junction sector's ring is the cap's
                // own boundary, so a custom curve supplies it — read back off the `doubled`
                // chain under `Resampled` (the [V-ADJ] cap rim) and raw off `target` under
                // `Raw` (the §8.3 cutoff fill). Both arms live in `chain_positions`; what
                // this call site owes is the curve itself and — §8.6.3.1(6) Regime A — the
                // handedness the orientation walk marked *this* edge-end with. Sibling
                // sectors of one junction routinely disagree here: three mutually orthogonal
                // spokes are three separate runs with three independent seed bits.
                let reverse = orientation.reverse(crossing, vertex);
                let (interior, cell) = self.build_chain_between(
                    boundverts[k],
                    boundverts[next],
                    crossing,
                    segments,
                    profile,
                    custom.map(|chains| CustomChainRead::edge_sector(chains, reverse)),
                    rescue,
                    None,
                    chain_emission,
                    result,
                );
                chains.push(EdgeEndChain {
                    edge: crossing,
                    endpoint: vertex,
                    start: boundverts[k],
                    interior,
                    end: boundverts[next],
                    side_label: SideLabel::Unlabelled,
                    cell: Some(cell),
                    custom_reverse: reverse,
                });
            }
        }

        // §6.6.1: the odd-`segments` centre n-gon of the [V-ADJ] grid (§8.5) is the fourth
        // ring-spanning fill family — seam-free it takes `B_0`'s rep off this junction's own
        // ring, and the tie-break reaches it at a seam-carrying junction exactly as it
        // reaches the cap, the patch and the terminal fill.
        let central_representative = templates.register(self.anchored_ring_spanning_donor(
            vertex,
            is_beveled,
            affect,
            segments,
            self.ring_first_boundvert_rep(vertex, &opening_edges),
            templates.interp_components(),
        ));

        // §6.6.2's [V-ADJ] grid row (edge bevel): each sector's non-centre faces inherit
        // from that sector's **own** boundvert rep, and — at odd `segments` — its centre
        // column takes the own rep or the §6.6.1.4 tie-break between this and the next
        // boundvert's reps (`centre_column_donor`). Registered here against the pristine
        // mesh; the vertex-mode grid rule (the next-boundvert rep) is the caller's, not
        // this edge-junction path's.
        //
        // B_i **opens** sector `i` along `opening_edges[i]`, but the beveled edge it
        // *starts* in the CCW ring walk (§6.6.1's "a boundvert that starts a beveled leg
        // takes that leg's edge") is the sector's **closing** edge — `opening_edges[i + 1]`,
        // the one B_i's chain departs across. Its `prev_face` is the original face filling
        // sector `i`'s wedge (between `opening_edges[i]` and `opening_edges[i + 1]`), and the
        // §5.2.4.2 `e_i` B_i slid along. Reading `opening_edges[i]` instead takes the
        // *arriving* leg, whose `prev_face` is the previous sector's face — an off-by-one
        // rotation of the donor across the ring.
        let sector_count = boundverts.len();
        let own_edges: Vec<EdgeKey> = (0..sector_count)
            .map(|i| opening_edges[(i + 1) % sector_count])
            .collect();
        let sector_own_faces: Vec<Option<FaceKey>> = own_edges
            .iter()
            .map(|&edge| self.boundvert_representative(vertex, edge))
            .collect();
        let sector_representatives: Vec<FaceTemplate> = sector_own_faces
            .iter()
            .map(|&face| templates.register(face))
            .collect();
        let centre_column_representatives: Vec<FaceTemplate> = (0..sector_count)
            .map(|i| {
                let donor = self.centre_column_donor(
                    own_edges[i],
                    sector_own_faces[i],
                    sector_own_faces[(i + 1) % sector_count],
                    is_beveled(own_edges[i]),
                    templates.interp_components(),
                );
                templates.register(donor)
            })
            .collect();

        junction_cuts.push(JunctionCut {
            vertex,
            boundverts,
            opening_edges,
            outward,
            central_representative,
            sector_representatives,
            centre_column_representatives,
            dihedral_band,
            pipe_override,
            square_in_weld: square_in_weld.is_some(),
            opening_edge_directions,
            opening_edge_tips,
            boundvert_non_reflex,
        });
    }

    /// Weld the two chains of an `n == 2` junction ring into one (BEVEL.md §8.3.0.3).
    ///
    /// Both sectors' chains run between the same two boundverts, so they are identified
    /// vertex-for-vertex in reverse: the junction mints `segments − 1` interior verts
    /// **once**, and the single chain `[B_0, w_1 … w_(segments−1), B_1]` becomes the shared
    /// terminal rung of both arriving strips. Each chain edge then lands on exactly two
    /// faces — one quad per strip — so the corner is closed with zero fill faces.
    ///
    /// Both profiles' frames are derived from scratch here (rescue included), *then* the
    /// weld-plane override replaces each plane's normal, *then* both chains are sampled:
    /// the override is what makes the two samplings coincide, rather than papering over a
    /// difference by averaging. The merge below is BMesh's defensive midpoint — it does
    /// real work only in the fallback where the override was abandoned.
    ///
    /// A custom curve reaches here on the same terms as any other ring sector. §8.6.3.1(4).4's
    /// per-consumer table puts the weld in **row 6** — the raw `target` chain at every
    /// `segments`, with no cascade, no source polyline and no stride — which is what
    /// [`ChainEmission::Raw`] at the operation's own `segments` already asks for. The
    /// handedness is §8.6.3.1(6) **Regime A**: a `count == 2` ring is the walk's
    /// beveled-valence-2 cross-vertex continuation, so each half's bit is just its owner
    /// edge-end's mark, read straight from the pass (A.5 — "the weld is not special"). Reading
    /// the two halves through *equal* bits is the observable failure: the halves then sample
    /// mirrored cell coordinates, no re-planing can reconcile them, and the merge below papers
    /// the difference over with a midpoint that lies on neither cross-section.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn build_welded_chain(
        &mut self,
        vertex: VertexKey,
        boundverts: &[VertexKey],
        opening_edges: &[EdgeKey],
        corners: &[VertexCorner],
        is_beveled: &dyn Fn(EdgeKey) -> bool,
        segments: u32,
        profile: f32,
        custom: Option<&CustomChains>,
        orientation: &CustomOrientation,
        chains: &mut Vec<EdgeEndChain>,
        cross_welds: &mut Vec<CrossWeld>,
        result: &mut BevelResult,
    ) {
        // Sector 0's chain runs B_0 → B_1 across the edge opening sector 1, and sector
        // 1's runs B_1 → B_0 across the edge opening sector 0.
        let [first_boundvert, second_boundvert] = [boundverts[0], boundverts[1]];
        let first_edge = opening_edges[1];
        let second_edge = opening_edges[0];

        let mut first = self.derive_profile_cell(
            first_boundvert,
            second_boundvert,
            first_edge,
            self.mid_rescue(vertex, first_edge, corners, is_beveled),
        );
        let mut second = self.derive_profile_cell(
            second_boundvert,
            first_boundvert,
            second_edge,
            self.mid_rescue(vertex, second_edge, corners, is_beveled),
        );
        ProfileCell::override_weld_plane(&mut first, &mut second, self.vertex_position(vertex));

        // §8.6.3.1(6) A.5: each half reads the curve through *its own* edge-end's mark. The two
        // marks differ by the walk's parity across the cross-vertex step, which is precisely
        // what makes the two halves' cell coordinates transpose and their samples coincide.
        let first_reverse = orientation.reverse(first_edge, vertex);
        let second_reverse = orientation.reverse(second_edge, vertex);

        // The welded chain is no cap boundary — neither method fills this corner — so
        // both profiles emit their raw samples (§8.2.2.1.1's resample is a [V-ADJ] cap
        // property, and §8.6.3.1(4).4 row 6 says the same of a custom curve's chain), which
        // is why `Cutoff` and `Adj` land the same verts here.
        let first_samples = first.chain_positions(
            segments,
            profile,
            custom.map(|chains| CustomChainRead::edge_sector(chains, first_reverse)),
            ChainEmission::Raw,
        );
        let second_samples = second.chain_positions(
            segments,
            profile,
            custom.map(|chains| CustomChainRead::edge_sector(chains, second_reverse)),
            ChainEmission::Raw,
        );

        let mut interior = Vec::with_capacity(segments.saturating_sub(1) as usize);
        for index in 1..segments as usize {
            let from_first = first_samples[index];
            let from_second = second_samples[segments as usize - index];
            let position =
                merge_welded_sample(&first, &second, custom.is_some(), from_first, from_second);
            // §5.2.4: not yet minted from the corner.
            let welded = self.create_vertex(position);
            result.verts.push(welded);
            interior.push(welded);
        }

        if let Some(cross) = self.cross_weld(
            vertex,
            first_boundvert,
            second_boundvert,
            &interior,
            is_beveled,
        ) {
            cross_welds.push(cross);
        }

        let mut reversed = interior.clone();
        reversed.reverse();
        chains.push(EdgeEndChain {
            edge: first_edge,
            endpoint: vertex,
            start: first_boundvert,
            interior,
            end: second_boundvert,
            side_label: SideLabel::Unlabelled,
            cell: Some(first),
            custom_reverse: first_reverse,
        });
        chains.push(EdgeEndChain {
            edge: second_edge,
            endpoint: vertex,
            start: second_boundvert,
            interior: reversed,
            end: first_boundvert,
            side_label: SideLabel::Unlabelled,
            cell: Some(second),
            custom_reverse: second_reverse,
        });
    }

    /// Lay a §8.5.5.5(f.4a) square-in junction's chains as `s` **welded legs**.
    ///
    /// At `r = 0` the cross-section between two boundverts is not an arc at all: it runs in
    /// along one axis to the corner's inner point and back out along the other, so sector
    /// `i`'s chain lies on the two axes bounding that sector rather than between them. In
    /// octant coordinates, with `h = ⌊n/2⌋` and the step
    ///
    /// ```text
    ///    b  =  2 / n                     (n even)
    ///       =  2 / (2·h + √2)            (n odd)
    ///
    ///    (i, 0, k)      =  (1 − k·b) · e_i          0 <= k <= h
    ///    (i, 0, n − k)  =  (1 − k·b) · e_{i+1}
    /// ```
    ///
    /// where `b` is §8.6.4 step 3's closing-chord term solved on two straight unit legs.
    ///
    /// The layout is built **in the octant and mapped once**, as (f.1) legislates for every
    /// tri-corner grid: under the affine `unit_to_corner` the axis point `e_i` is `B_i` and
    /// the octant origin is the frame's `c` (`anchor`), so the whole chain is the straight run
    /// `c + (1 − k·b)·(B_i − c)`. Reading the two sectors' own 3-D cross-sections and merging
    /// their views of a shared leg vertex instead would agree only on a symmetric corner: on
    /// the oblique / absorbed-leg / reflex ring the band deliberately admits (§7.6.6) the two
    /// readings genuinely differ, and any merge of them invents a third point that is on
    /// neither leg.
    ///
    /// The topology is the other half, and the whole point: `(i, 0, k)` **is the same vertex
    /// as** `(i − 1, 0, n − k)` for `k <= h`, not a coincident twin. So each boundvert grows one
    /// inward *leg* of `h` verts, minted once and shared by the two chains meeting there;
    /// both arriving strips terminate on it and every leg edge lands on exactly two faces,
    /// one per strip. On **even** `n` the innermost sample of every leg is the octant origin,
    /// so the identification closes transitively over all `s` legs onto a single vertex and
    /// the corner needs no fill face at all. On **odd** `n` no sample reaches the origin and
    /// the `s` innermost verts are left for the fill's one central `s`-gon.
    ///
    /// The legs are spaced at (f.4a)'s printed `b` — the **raw** closed form — and not at a
    /// §8.2.2.1.1 chord read-back of it. §8.2.2.1.1's gate part (i) is a question about the
    /// *fill's* doubling cascade — reach `segments` from a 2-segment control cage by doubling
    /// and, if that overshoots, resampling back down — and the weld runs no such cascade: it
    /// has no cage, no doubling round and no interior, and (f.4a) states the layout in closed
    /// form at `n` directly. Nor is there a later step that would restore the closed form: the
    /// snap that does so for the tri-corner family's other members has no 3-D branch at
    /// square-in ((f.2)'s branch S is 2-D and pipe-only). So a read-back here would move the
    /// legs with nothing to put them back, and (f.4a)'s odd-`n` step — the whole point of the
    /// `√2` bookkeeping — could never be observed: these chords are equal by construction, so a
    /// chord fraction degenerates to an even-arclength split, i.e. exactly `b = 2/n` at every
    /// parity (identity at even `n`, and at `n = 3` it would drag the first leg sample from
    /// `0.41421·(B_i − c)` to `0.33333·(B_i − c)`). The guide does not state this exemption; it
    /// is inferred from the resample's own scope and from the absence of a restoring snap.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn build_square_in_weld_chains(
        &mut self,
        vertex: VertexKey,
        boundverts: &[VertexKey],
        opening_edges: &[EdgeKey],
        corners: &[VertexCorner],
        is_beveled: &dyn Fn(EdgeKey) -> bool,
        segments: u32,
        anchor: Vec3,
        chains: &mut Vec<EdgeEndChain>,
        result: &mut BevelResult,
    ) {
        let sectors = boundverts.len();
        let n = segments as usize;
        let half = n / 2;

        let mut cells: Vec<ProfileCell> = Vec::with_capacity(sectors);
        for k in 0..sectors {
            let next = (k + 1) % sectors;
            let crossing = opening_edges[next];
            let rescue = self.mid_rescue(vertex, crossing, corners, is_beveled);
            cells.push(self.derive_profile_cell(boundverts[k], boundverts[next], crossing, rescue));
        }

        let even = n % 2 == 0;
        let step = square_in_weld_leg_step(segments);
        let mut origin: Option<VertexKey> = None;
        let mut legs: Vec<Vec<VertexKey>> = Vec::with_capacity(sectors);
        for &boundvert in boundverts {
            let axis = self.vertex_position(boundvert) - anchor;
            let mut leg: Vec<VertexKey> = Vec::with_capacity(half);
            for k in 1..=half {
                if even && k == half {
                    // `1 − ⌊n/2⌋·b` is exactly zero at even `n`, so every leg's innermost
                    // sample is the octant origin and the pairwise identification closes
                    // transitively over all `s` legs onto one vertex.
                    let shared = match origin {
                        Some(shared) => shared,
                        None => {
                            // §5.2.4: not yet minted from the corner.
                            let shared = self.create_vertex(anchor);
                            result.verts.push(shared);
                            origin = Some(shared);
                            shared
                        }
                    };
                    leg.push(shared);
                    continue;
                }
                let position = anchor + (1.0 - k as f32 * step) * axis;
                // §5.2.4: not yet minted from the corner.
                let welded = self.create_vertex(position);
                result.verts.push(welded);
                leg.push(welded);
            }
            legs.push(leg);
        }

        for i in 0..sectors {
            let next = (i + 1) % sectors;
            let mut interior: Vec<VertexKey> = Vec::with_capacity(n - 1);
            for k in 1..n {
                interior.push(if k <= half {
                    legs[i][k - 1]
                } else {
                    legs[next][n - k - 1]
                });
            }
            // Inert, not a gap: §8.5.5.5(f.4a)'s ring test refuses a custom profile outright, so
            // every chain laid here is an analytic superellipse — diagonal-symmetric, and
            // §8.6.3.1(6)'s handedness bit is unobservable on those.
            chains.push(EdgeEndChain {
                edge: opening_edges[next],
                endpoint: vertex,
                start: boundverts[i],
                interior,
                end: boundverts[next],
                side_label: SideLabel::Unlabelled,
                cell: Some(cells[i]),
                custom_reverse: EdgeEndChain::UNMARKED_DEFAULT_REVERSE,
            });
        }
    }

    /// Classify the §8.3.0.3 **cross** weld and capture its donor legs, off the pristine
    /// disk: a valence-4 welded junction whose two beveled edges sit **opposite** each
    /// other in the disk, so the two unbeveled legs are opposite too and the welded chain
    /// reads as the continuation of the two crossing legs. Every other weld — valence 3,
    /// or valence 4 with the beveled edges adjacent — donates nothing.
    ///
    /// Both "opposite in the disk" and "disk-predecessor" are *rotational* statements, so
    /// they are read off the §7.6.2 walk ([`Self::walk_disk_edges`]) and never off the
    /// stored disk cycle, whose order is the mesh's insertion order. Get the walk's sense
    /// backwards and the two donors swap.
    ///
    /// `interior` is the welded chain's interior verts — empty at `segments = 1`, where
    /// the chain degenerates to the single chamfer edge between the two boundverts and
    /// takes the donation just the same (§8.3.0.3's scope box).
    pub(super) fn cross_weld(
        &self,
        vertex: VertexKey,
        first_boundvert: VertexKey,
        second_boundvert: VertexKey,
        interior: &[VertexKey],
        is_beveled: &dyn Fn(EdgeKey) -> bool,
    ) -> Option<CrossWeld> {
        // §7.2.1: "valence 4" is the *face-bearing* count — a wire spoke on the junction's
        // disk is no more a leg here than it is in the layout.
        let faced = self
            .iter_disk_edges(vertex)
            .filter(|&spoke| self.edge_radial_iter(spoke).next().is_some())
            .count();
        if faced != 4 {
            return None;
        }
        let disk = self.walk_disk_edges(vertex);
        if disk.len() != 4 {
            return None; // an open or non-manifold fan: no cyclic rotational order
        }
        let beveled: Vec<usize> = (0..4).filter(|&i| is_beveled(disk[i])).collect();
        let [&before, &after] = [beveled.first()?, beveled.get(1)?];
        if after - before != 2 {
            return None; // adjacent in the disk, not a cross
        }

        let mut chain = Vec::with_capacity(interior.len() + 2);
        chain.push(first_boundvert);
        chain.extend_from_slice(interior);
        chain.push(second_boundvert);
        Some(CrossWeld {
            chain,
            // Each beveled edge donates from its own disk-**predecessor** leg, so the two
            // runs have different donors — the two crossing legs, one each.
            donors: [before, after]
                .map(|index| (disk[index], disk[(index + 3) % 4]))
                .to_vec(),
        })
    }

    /// Apply the §8.3.0.3 cross-weld per-edge attribute rule, once every strip has laid
    /// the welded chain's edges down. The copy runs **once per beveled edge**, each run
    /// donating that edge's own disk-predecessor leg's whole per-edge block onto the
    /// *same* chain edges — so the two runs collide and **the later one prevails**. Both
    /// runs then apply the same two reconciliations, computed from **both** crossing legs
    /// and therefore order-independent: a seam only crosses if it crosses on both sides,
    /// and the shading stays smooth unless both sides are sharp.
    ///
    /// "Later" is the order the operator walks its beveled edges, and that is the **mesh's
    /// own edge order** — not the order the caller listed them in the input slot (the one
    /// place in bevel where element creation order is observable in the output). So the
    /// two donations are replayed in ascending edge key here rather than in input order.
    ///
    /// The donation belongs to the *weld*, not to the multi-segment chain builder: at
    /// `segments = 1` the chain is one chamfer edge and takes it identically.
    pub(super) fn apply_cross_weld_attributes(&mut self, cross: &CrossWeld) {
        let chain_edges: Vec<EdgeKey> = cross
            .chain
            .windows(2)
            .filter_map(|pair| self.query_edge_exists(pair[0], pair[1]))
            .collect();
        if chain_edges.is_empty() {
            return;
        }

        let legs: Vec<EdgeKey> = cross.donors.iter().map(|&(_, leg)| leg).collect();
        let seam = legs.iter().all(|&leg| self.edge_seam(leg));
        let sharp = legs.iter().all(|&leg| self.edge_sharp(leg));

        let mut donations = cross.donors.clone();
        donations.sort_by_key(|&(beveled, _)| beveled.as_usize());
        for (_, donor) in donations {
            for &chain_edge in &chain_edges {
                self.edge_attribs
                    .copy_into(chain_edge.as_usize(), donor.as_usize());
                self.set_edge_seam(chain_edge, seam);
                self.set_edge_sharp(chain_edge, sharp);
            }
        }
    }

    /// Build a profile chain's `segments − 1` interior verts between two ring
    /// boundverts crossing beveled edge `edge` (BEVEL.md §8.2.2.1 / §8.3.6). The
    /// endpoints are passed in; the interior arc samples are minted here. Returns the
    /// interior verts and the frame they were sampled through — the frame is *derived
    /// here, once*, so a caller that overrides it (the §8.3.0.3 weld) hands its own cell
    /// in via `derive_profile_cell` rather than re-deriving downstream.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn build_chain_between(
        &mut self,
        start: VertexKey,
        end: VertexKey,
        edge: EdgeKey,
        segments: u32,
        profile: f32,
        custom: Option<CustomChainRead<'_>>,
        rescue: MidRescue,
        terminal_vertex: Option<Vec3>,
        emission: ChainEmission,
        result: &mut BevelResult,
    ) -> (Vec<VertexKey>, ProfileCell) {
        let mut cell = self.derive_profile_cell(start, end, edge, rescue);
        // §8.2.2.1 terminal-far-end override: at a valence-≥3 terminal the default
        // `start–mid–end` mid-plane is tilted up out of the rim by `mid` (the on-edge
        // foot, which sits partway up the beveled edge), lifting the samples off the
        // surface. Snap the profile plane to the triangle through the terminal vertex
        // and the two chain endpoints, flattening the cross-section into the rim plane.
        if let Some(terminal) = terminal_vertex {
            cell.snap_terminal_plane(terminal);
        }
        let interior = self.mint_chain_interior(&cell, segments, profile, custom, emission, result);
        (interior, cell)
    }

    /// Derive one beveled-edge sector's profile frame from scratch (BEVEL.md §8.2.2.1,
    /// rescue included) — beat 1 of the derive → override → mark discipline.
    pub(super) fn derive_profile_cell(
        &self,
        start: VertexKey,
        end: VertexKey,
        edge: EdgeKey,
        rescue: MidRescue,
    ) -> ProfileCell {
        let [ea, eb] = self.edge_vert_keys(edge);
        let edge_point = self.vertex_position(ea);
        let edge_dir = (self.vertex_position(eb) - edge_point).normalize_or_zero();
        let mut cell = ProfileCell::new(
            self.vertex_position(start),
            self.vertex_position(end),
            edge_point,
            edge_dir,
        );
        cell.rescue_degenerate_mid(&rescue);
        cell
    }

    /// Mint the `segments − 1` interior verts of a chain sampled through `cell`.
    pub(super) fn mint_chain_interior(
        &mut self,
        cell: &ProfileCell,
        segments: u32,
        profile: f32,
        custom: Option<CustomChainRead<'_>>,
        emission: ChainEmission,
        result: &mut BevelResult,
    ) -> Vec<VertexKey> {
        let positions = cell.chain_positions(segments, profile, custom, emission);
        let mut interior = Vec::with_capacity(segments.saturating_sub(1) as usize);
        for position in positions.iter().take(positions.len() - 1).skip(1) {
            // §5.2.4: not yet minted from the corner.
            let vertex = self.create_vertex(*position);
            result.verts.push(vertex);
            interior.push(vertex);
        }
        interior
    }

    /// The §8.2.2.1 rescue context for the chain crossing `edge` at `vertex`: the
    /// original vertex, plus tier 2's two offset-line directions when — and only when —
    /// both of `edge`'s **rotational** neighbours at `vertex` are themselves beveled and
    /// the vertex carries `s ≥ 3` beveled edges. Read off the pristine corner ring, which
    /// is the rotational order §7.6.1 walks — the radial structure's own edge cycle is
    /// insertion-ordered and says nothing about which edges flank `edge` around the fan.
    /// The returned directions are ordered `(start-side, end-side)` to match the chain's
    /// `start` / `end` boundverts, since the ring the sectors were split from is the ring
    /// indexed here.
    pub(super) fn mid_rescue(
        &self,
        vertex: VertexKey,
        edge: EdgeKey,
        corners: &[VertexCorner],
        is_beveled: &dyn Fn(EdgeKey) -> bool,
    ) -> MidRescue {
        let vertex_position = self.vertex_position(vertex);
        let beveled_count = self
            .iter_disk_edges(vertex)
            .filter(|&e| is_beveled(e))
            .count();
        let neighbour_dirs = (beveled_count >= 3)
            .then(|| rotational_edge_neighbours(corners, edge))
            .flatten()
            .and_then(|(before, after)| {
                if !is_beveled(before) || !is_beveled(after) {
                    return None;
                }
                let direction_before = self.edge_direction_from(before, vertex, vertex_position)?;
                let direction_after = self.edge_direction_from(after, vertex, vertex_position)?;
                Some((direction_before, direction_after))
            });
        MidRescue {
            vertex: vertex_position,
            neighbour_dirs,
        }
    }

    /// The **raw coarse** `(i, 0, 1)` profile mid of a flank / strip ring edge at the
    /// `segments = 2` control-net resolution (BEVEL.md §8.5.5.5(d) / §8.9.5.3): the
    /// naive cross-section sample *before* the §8.2.2.1.1 chord-fraction resample. On a
    /// symmetric (equal-offset) cell this equals the emitted rim mid (the resample is the
    /// identity there); on the skewed arc-flanking cell it is the un-skewed point the
    /// extraordinary-centre stencil must see instead of the resampled rim value.
    pub(super) fn coarse_flank_mid<'c>(
        &self,
        start: VertexKey,
        end: VertexKey,
        edge: EdgeKey,
        profile: f32,
        custom: Option<CustomChainRead<'c>>,
        rescue: MidRescue,
    ) -> Vec3 {
        // §8.6.3.1(4).4 row 1: the coarse net is the read at `m = 2`, and this mid is only
        // ever consumed at `segments == 2`, where the schedule's `m == segments` arm returns
        // `target[1]` — not the strided `doubled[doubled_resolution / 2]`.
        self.ring_flank_profile(start, end, edge, profile, custom, rescue)
            .raw_positions(2)[1]
    }

    /// The profile **generator** of a ring edge that crosses a beveled edge (BEVEL.md
    /// §8.2.2.1 / §8.5.5.5(a)): that edge's cross-section cell between the two ring
    /// boundverts. Rebuilt from the same three inputs `build_chain_between` used, so the
    /// generator re-samples to exactly the polyline the emitted chain was read off — which
    /// is why `custom` must be the *same* read (curve **and** §8.6.3.1(6) handedness bit)
    /// the emitted chain was minted through.
    pub(super) fn ring_flank_profile<'c>(
        &self,
        start: VertexKey,
        end: VertexKey,
        edge: EdgeKey,
        profile: f32,
        custom: Option<CustomChainRead<'c>>,
        rescue: MidRescue,
    ) -> RingChainProfile<'c> {
        RingChainProfile::Cell {
            cell: self.derive_profile_cell(start, end, edge, rescue),
            profile,
            custom,
        }
    }

    /// The chain spanning sector `i`'s boundvert pair at this junction (BEVEL.md §8.2.2) —
    /// the one edge-end chain of `cut.vertex` whose two named ends are `B_i` and `B_{i+1}`,
    /// in whichever sense it was sampled. `None` when no chain spans the pair.
    ///
    /// The single place the sector → chain lookup lives. Every reader of a sector's rim has
    /// to agree on *which* chain a sector index names — the net's `k = 0` column
    /// ([`Self::adj_grid_vert`], via [`Self::oriented_chain`]) and the §8.5.5.5(f.6.1) rim
    /// replant index the same chain from the same boundvert, and a second copy of this
    /// predicate is exactly how those two would drift apart.
    pub(super) fn sector_chain<'c>(
        &self,
        cut: &JunctionCut,
        chains: &'c [EdgeEndChain],
        i: usize,
    ) -> Option<&'c EdgeEndChain> {
        let count = cut.boundverts.len();
        let start = cut.boundverts[i];
        let end = cut.boundverts[(i + 1) % count];
        chains.iter().find(|chain| {
            chain.endpoint == cut.vertex
                && ((chain.start == start && chain.end == end)
                    || (chain.start == end && chain.end == start))
        })
    }

    /// The profile chain of sector `i` at this junction, oriented from its opening
    /// boundvert `B_i` (BEVEL.md §8.2.2). `None` when no chain spans the sector's
    /// boundvert pair.
    pub(super) fn oriented_chain(
        &self,
        cut: &JunctionCut,
        chains: &[EdgeEndChain],
        i: usize,
    ) -> Option<Vec<VertexKey>> {
        Some(
            self.sector_chain(cut, chains, i)?
                .oriented_from(cut.boundverts[i]),
        )
    }

    /// One boundvert's down-pushed corner-vert position (BEVEL.md §8.3.3): `B_k`
    /// plus the **raw** cross of the two adjacent sectors' mid-plane normals (the
    /// fold crease, whose own length is `sin θ`), scaled by the average of the two
    /// profiles' `height/√2`, oriented into the corner. The true displacement
    /// distance is therefore `sin θ · (height/√2)` — the `sin θ` from the
    /// un-normalised crease is load-bearing (§8.3.3.1).
    pub(super) fn corner_vert_position(
        &self,
        cut: &JunctionCut,
        chains: &[EdgeEndChain],
        k: usize,
        profile: f32,
    ) -> Vec3 {
        let count = cut.boundverts.len();
        let boundvert = cut.boundverts[k];
        let boundvert_position = self.vertex_position(boundvert);

        // §8.3.3: the down-push height is binary and set by the sector's profile
        // shape alone, not its valence. At the straight chord (`r == 1`) the cell
        // map is bypassed and the push is exactly zero for every junction valence
        // `s`, so the corner vert stays coincident-in-position with its boundvert
        // `B_k` at the undisplaced ring. Only curved profiles push the full cell
        // diagonal. Gating here (not per-`count`) is what suppresses the §8.3.4
        // collapse at `s == 3` and keeps `s >= 4` corners on the ring.
        if profile_to_exponent(profile) == 1.0 {
            return boundvert_position;
        }

        let before = (k + count - 1) % count; // sector ending at B_k
        let after = k; // sector starting at B_k

        let cell_before = self.sector_cell(cut, chains, before);
        let cell_after = self.sector_cell(cut, chains, after);

        let (normal_before, height_before) = cell_before
            .map(|c| (c.mid_plane_normal, c.height()))
            .unwrap_or((Vec3::ZERO, 0.0));
        let (normal_after, height_after) = cell_after
            .map(|c| (c.mid_plane_normal, c.height()))
            .unwrap_or((Vec3::ZERO, 0.0));

        // §8.3.3: the fold-crease direction is the RAW cross product of the two
        // adjacent unit mid-plane normals. Its magnitude is `sin θ` (θ = dihedral
        // between those mid-planes), and that `sin θ` is *part of the push distance*
        // — the height scalar below is scaled by it. Re-normalising to unit length
        // silently drops the factor and over-pushes every corner by `1 / sin θ`
        // (§8.3.3.1). Keep it raw; only flip its sign to point into the corner.
        let mut down = normal_before.cross(normal_after);
        if down.length_squared() <= 1.0e-12 {
            return boundvert_position;
        }
        if down.dot(cut.outward) > 0.0 {
            down = -down;
        }

        let inv_sqrt2 = 1.0 / 2.0_f32.sqrt();
        let push = (height_before * inv_sqrt2 + height_after * inv_sqrt2) / 2.0;
        boundvert_position + down * push
    }

    /// The profile cell of sector `k` at the junction (BEVEL.md §8.2.2.1) — the frame the
    /// chain was actually sampled through, read back off the chain rather than re-derived
    /// (§8.2.2.1 "The recompute…": a derived-from-scratch frame would drop any override).
    pub(super) fn sector_cell(
        &self,
        cut: &JunctionCut,
        chains: &[EdgeEndChain],
        k: usize,
    ) -> Option<ProfileCell> {
        self.sector_chain(cut, chains, k)
            .and_then(|chain| chain.cell)
    }
}

/// One beveled edge's profile chain at one endpoint (BEVEL.md §8.2.2 / §8.3.0):
/// the two ring boundverts (`start` / `end`) it spans plus the `segments − 1`
/// interior arc samples between them.
///
/// `start` / `end` carry **no** cross-endpoint meaning: a junction chain is ordered
/// by the boundvert ring alone (never by the crossing edge's radial order), and the
/// ring rotates oppositely at the edge's two endpoints, so an edge's two chains are
/// generally *anti*-aligned. The strip's rung join must therefore pair the two ends'
/// samples topologically at build time — see `build_multisegment_strip` — rather
/// than trust this order.
/// One endpoint's two ring verts for a beveled edge, tagged with the **side label**
/// each was laid out with (BEVEL.md §7.8.3): `prev_side` is `P(v)`, the ring vert on
/// the sector *preceding* `E` in the endpoint's normalised rotational order, and
/// `next_side` is `N(v)`, the one on the sector following it.
///
/// The label is combinatorial and pre-geometric — fixed when the ring is laid out,
/// never recovered afterwards from where the vert landed or from which flank face
/// touches it. It is what pairs the strip's two rails, and the only formulation of
/// that pairing which survives an inconsistently-wound flank pair, where a single
/// flank can occupy the *same* slot at both endpoints.
#[derive(Clone, Copy)]
pub(super) struct RungEnds {
    pub(super) prev_side: VertexKey,
    pub(super) next_side: VertexKey,
    pub(super) prev_face: FaceKey,
    pub(super) next_face: FaceKey,
}

impl RungEnds {
    /// The ring vert the given flank face touches at this endpoint. Legitimate for
    /// per-face bookkeeping (a face's own corner); never for pairing the rails.
    pub(super) fn on_face(&self, face: FaceKey) -> VertexKey {
        debug_assert!(
            face == self.prev_face || face == self.next_face,
            "only a flank of the beveled edge touches a rung end",
        );
        if face == self.prev_face {
            self.prev_side
        } else {
            self.next_side
        }
    }

    /// The §7.8.3 side label the given flank face's ring vert carries at this endpoint.
    /// Shares `on_face`'s guard so a face that flanks neither side is caught rather than
    /// silently labelled `NextFirst`.
    pub(super) fn label_on_face(&self, face: FaceKey) -> SideLabel {
        debug_assert!(
            face == self.prev_face || face == self.next_face,
            "only a flank of the beveled edge touches a rung end",
        );
        if face == self.prev_face {
            SideLabel::PrevFirst
        } else {
            SideLabel::NextFirst
        }
    }
}

/// The BEVEL.md §7.8.3 side label an [`EdgeEndChain`]'s `start` vert carries: `P(v)`, the
/// ring vert on the sector *preceding* the crossed edge in the endpoint's normalised
/// rotational order, or `N(v)`, the one on the sector following it.
///
/// **Invariant: only junction-family chains are `Unlabelled`.** A junction chain is
/// ordered by its boundvert ring alone and the builder does not carry the endpoint's
/// rotational slot through to the chain; every terminal chain is labelled at layout time.
/// `Unlabelled` is therefore a statement about this crate, not about the endpoint — the
/// label is defined at every beveled endpoint of any valence (§7.8.6's junction bullet,
/// §8.3.0) — and where it appears the rail pairing must fall back to naming the side
/// topologically. See `build_multisegment_strip`.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum SideLabel {
    PrevFirst,
    NextFirst,
    Unlabelled,
}

pub(super) struct EdgeEndChain {
    pub(super) edge: EdgeKey,
    pub(super) endpoint: VertexKey,
    pub(super) start: VertexKey,
    pub(super) interior: Vec<VertexKey>,
    pub(super) end: VertexKey,
    /// Which of the endpoint's two §7.8.3 side labels `start` carries. When both of an
    /// edge's chains carry a label, the label — not the rails' existence — pairs their
    /// samples.
    pub(super) side_label: SideLabel,
    /// The frame the chain was sampled through (BEVEL.md §8.2.2.1), kept rather than
    /// re-derived: an overridden frame (the terminal snap, the §8.3.0.3 weld plane) is
    /// not recoverable from the boundverts alone, and a downstream re-derivation would
    /// silently restore the default mid-plane — the "recompute erases the override" bug.
    pub(super) cell: Option<ProfileCell>,
    /// The §8.6.3.1(6) handedness bit this chain's samples were read through, kept for the
    /// same reason `cell` is: the cap fill re-samples the sector's generator at its own
    /// resolutions and must read it the way the rim was read. Meaningless without a custom
    /// curve (every analytic cross-section is diagonal-symmetric).
    pub(super) custom_reverse: bool,
}

impl EdgeEndChain {
    /// What `custom_reverse` records where §8.6.3.1(6)'s orientation bit was never threaded to
    /// the builder: the orientation pass's unmarked default,
    /// [`CustomOrientation::reverse`]'s answer for an edge-end absent from the map.
    ///
    /// Inert on an **analytically**-sampled chain, where the diagonal-symmetric superellipse
    /// family makes the bit unobservable. Its one live use is the §8.5.5.5(f.4a) square-in weld,
    /// whose ring test refuses a custom profile outright, so no chain that reaches it can read a
    /// curve. The name says "unmarked", not "analytic", so that a future builder cannot read it
    /// as a licence to skip the pass: the two miter ring builders used to sit here and were
    /// genuine gaps until they were threaded onto their edge-ends' own marks.
    pub(super) const UNMARKED_DEFAULT_REVERSE: bool = true;

    /// The chain's `segments + 1` verts in order `start → interior → end`.
    pub(super) fn samples(&self) -> Vec<VertexKey> {
        let mut samples = Vec::with_capacity(self.interior.len() + 2);
        samples.push(self.start);
        samples.extend_from_slice(&self.interior);
        samples.push(self.end);
        samples
    }

    /// The chain's verts ordered to begin at `first` (either endpoint).
    pub(super) fn oriented_from(&self, first: VertexKey) -> Vec<VertexKey> {
        let mut samples = self.samples();
        if self.start != first {
            samples.reverse();
        }
        samples
    }

    /// Sample `index` of the walk that begins at `first` (either endpoint), without
    /// materialising [`EdgeEndChain::oriented_from`]'s vector. `None` past the last sample.
    pub(super) fn oriented_sample(&self, first: VertexKey, index: usize) -> Option<VertexKey> {
        let count = self.interior.len() + 2;
        if index >= count {
            return None;
        }
        let index = if self.start == first {
            index
        } else {
            count - 1 - index
        };
        if index == 0 {
            Some(self.start)
        } else if index == count - 1 {
            Some(self.end)
        } else {
            self.interior.get(index - 1).copied()
        }
    }
}

/// One chain of a junction's outermost boundvert ring, in ring order — the unit the
/// §6.7.5.3 seam / sharp walk steps over (BEVEL.md §6.7.5.4 case (5)).
///
/// A chain either crosses a beveled edge (a strip cross-section, `cross_edge =
/// Some`) or fills a **split** sector (a miter fill chain, `cross_edge = None`). The
/// walk counts only the former but steps over both, which is exactly why a span whose
/// ring path enters a split sector desynchronises.
pub(super) struct RingChain {
    pub(super) cross_edge: Option<EdgeKey>,
    pub(super) samples: Vec<VertexKey>,
}

/// One **cross** weld (BEVEL.md §8.3.0.3's attribute note): a valence-4 welded junction
/// whose two beveled edges are opposite in the disk, so the two unbeveled legs are
/// opposite too and the beveled edges run *through* the vertex with the legs crossing
/// them. The welded chain then continues the crossing legs, and takes their per-edge
/// attributes. A flag / customdata rule only — no topology, no placement.
pub(super) struct CrossWeld {
    /// The welded chain's `segments + 1` verts, `B_0 → w_1 … → B_1`. Its edges are laid
    /// down by the strips, so they are resolved from these verts afterwards.
    pub(super) chain: Vec<VertexKey>,
    /// Per beveled edge, the disk-**predecessor** leg it donates from. Two entries with
    /// two *different* donors (the two crossing legs, one each), both writing the same
    /// chain edges — the strip built last prevails.
    pub(super) donors: Vec<(EdgeKey, EdgeKey)>,
}

/// The context the §8.2.2.1 degenerate-`mid` rescue needs for one beveled-edge sector:
/// the original vertex the corner is being cut off, and — only for tier 2 — the two
/// offset-line directions the bulge follows when the crossed edge's own disk-neighbours
/// are themselves beveled.
#[derive(Clone, Copy)]
pub(super) struct MidRescue {
    pub(super) vertex: Vec3,
    /// The crossed beveled edge's disk-predecessor / disk-successor directions at the
    /// vertex. `Some` **only** when both of those disk-neighbours are themselves beveled
    /// *and* the vertex carries `s ≥ 3` beveled edges — the one case where neither chain
    /// endpoint is a leg point and there is no leg plane for tier 1 to stay in.
    pub(super) neighbour_dirs: Option<(Vec3, Vec3)>,
}

impl MidRescue {
    /// The rescue context for a vertex that cannot reach tier 2 by construction: tier 2
    /// needs `s ≥ 3` beveled edges at the vertex, so any site with a *known* smaller `s`
    /// (a terminal endpoint's `s = 1`, §7.5.1) states that here instead of paying for a
    /// corner-ring walk whose answer is fixed.
    pub(super) fn tier1_only(vertex: Vec3) -> Self {
        MidRescue {
            vertex,
            neighbour_dirs: None,
        }
    }
}

/// BEVEL.md §8.3.0.3's **merge rule** for one interior index of an `n == 2` junction's
/// welded chain: *"For each interior index `t = 1 … segments−1`, take `p_a` = `E_a`'s
/// profile sample at index `t` and `p_b` = `E_b`'s profile sample at index `segments − t`
/// (both post-override). Then **`w_t` is the midpoint of `p_a` and `p_b`** — except that
/// when **neither** profile's `profile_type` is `CUSTOM` and **exactly one** of the two
/// carries §8.2.2.1's **tier-3 linear flag**, `w_t` is the **unflagged** one's sample
/// instead."*
///
/// Three things the exception's predicate is **not**, each of which §8.3.0.3 calls out by
/// name:
///
/// - It is not the *stickiness* mark. §8.3.0.3: *"`profile_type == CUSTOM` is the
///   cross-section **kind**, not the stickiness mark … every profile at a weld gets
///   [the mark], by construction, the moment the override is attempted. Read the
///   exception's predicate as the stickiness mark and it can never fire at a weld at
///   all."* Hence `custom` here is the operation's `profile_type`, taken at the call site
///   from whether a curve was supplied — not a per-profile bit the weld sets. Both welded
///   profiles carry it or neither does: §8.9's miter carve-out, the one thing that can hand
///   a sector the analytic spacing under a custom bevel, cannot reach an `n == 2` ring at
///   all (§8.3.0.3 — an inner miter lifts the ring above two boundverts, and outer mitering
///   is only considered at `s >= 3`).
/// - It is not the emitted *shape*. §8.3.0.3: *"'Has degenerated to a straight chord' is a
///   fair description of a tier-3 profile but is the wrong test … Test the flag an actual
///   tier-3 collapse sets; do not re-derive the condition from the samples."* The whole
///   point is a profile that emits a straight chain **without** being flagged — the
///   zero-cell-axis case of §8.2.2.1, where the rescue never runs and the straightness
///   comes from the frame builder declining. Hence [`ProfileCell::is_tier3_collapsed`],
///   not [`ProfileCell::emits_straight_chord`].
/// - It is not observable at an `s == 2` weld. §8.3.0.3 proves the only reachable pairing
///   (one zero-axis profile, one rescued-and-collapsed partner) is a no-op, because both
///   halves then emit the same even chord and the merge reads them from opposite ends.
///   *"An implementation that omits the exception entirely is, at an `s == 2` weld,
///   indistinguishable from one that has it."* It is kept because it is what the reference
///   tests, not to explain any observed placement.
///
/// The exception is the fallback half of the same pair as the override's all-or-nothing
/// gate: a collapsed profile has no projection direction, so the very configuration that
/// makes the merge choose rather than average is the one in which the weld-plane override
/// was abandoned and the two samplings were never reconciled.
pub(super) fn merge_welded_sample(
    first: &ProfileCell,
    second: &ProfileCell,
    custom: bool,
    from_first: Vec3,
    from_second: Vec3,
) -> Vec3 {
    if custom {
        return (from_first + from_second) * 0.5;
    }
    match (first.is_tier3_collapsed(), second.is_tier3_collapsed()) {
        (true, false) => from_second,
        (false, true) => from_first,
        _ => (from_first + from_second) * 0.5,
    }
}
