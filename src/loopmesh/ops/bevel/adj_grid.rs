//! [V-ADJ] grid tiling, vert minting and cap winding (BEVEL.md §8.5.5 / §5.1 / §5.2.4).

use super::*;

impl LoopMesh {
    /// Tile one valence-`s` adjacency grid from its rim chains and an `interior` rule
    /// (BEVEL.md §8.5.5). The topology is identical for the cube-corner sphere fill
    /// (§8.5.5.1–§8.5.5.4) and the general subdivision-relaxation fill (§8.5.5.5) — only
    /// where interior positions come from differs. Even `segments` weld the `s` sub-patches
    /// at one shared centre vertex (no central face); odd `segments` bridge them with a
    /// ring of seam quads closed by one central `s`-gon (§8.5.5.5(e)).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn emit_adj_grid(
        &mut self,
        sectors: usize,
        seg: usize,
        half: usize,
        even: bool,
        chain_of: &[Vec<VertexKey>],
        interior: &AdjInterior<'_>,
        material: i32,
        sector_representatives: &[FaceTemplateRef<'_>],
        centre_column_representatives: &[FaceTemplateRef<'_>],
        central_representative: FaceTemplateRef<'_>,
        apex_blend: Option<&ApexBlend>,
        interior_blend: Option<&AdjInteriorBlend<'_>>,
        slots: &mut AdjNetSlots,
        result: &mut BevelResult,
    ) {
        debug_assert_eq!(
            sector_representatives.len(),
            sectors,
            "one representative per sector — §6.6.1's per-sector grid rule",
        );
        debug_assert_eq!(
            centre_column_representatives.len(),
            sectors,
            "one centre-column donor per sector — §6.6.2's odd-`segments` grid row",
        );
        // §5.1 / §8.5.3: the cap's winding is one bit for the whole grid, read off the rim it
        // welds to, not a per-face geometric test.
        let reverse = self.adj_grid_needs_reversing(sectors, seg, chain_of);
        let mut grid: std::collections::HashMap<(usize, usize, usize), VertexKey> =
            std::collections::HashMap::new();

        // Sub-patch interior cells (`half × half` quads each), shared between
        // sectors at the seams (even) or independent (odd, the seam strips bridge
        // them). Walk every cell of every sub-patch.
        for i in 0..sectors {
            for cell_j in 0..half {
                for cell_k in 0..half {
                    let a = self.adj_grid_vert(
                        i,
                        cell_j,
                        cell_k,
                        half,
                        seg,
                        sectors,
                        even,
                        chain_of,
                        interior,
                        apex_blend,
                        interior_blend,
                        &mut grid,
                        slots,
                        result,
                    );
                    let b = self.adj_grid_vert(
                        i,
                        cell_j,
                        cell_k + 1,
                        half,
                        seg,
                        sectors,
                        even,
                        chain_of,
                        interior,
                        apex_blend,
                        interior_blend,
                        &mut grid,
                        slots,
                        result,
                    );
                    let c = self.adj_grid_vert(
                        i,
                        cell_j + 1,
                        cell_k + 1,
                        half,
                        seg,
                        sectors,
                        even,
                        chain_of,
                        interior,
                        apex_blend,
                        interior_blend,
                        &mut grid,
                        slots,
                        result,
                    );
                    let d = self.adj_grid_vert(
                        i,
                        cell_j + 1,
                        cell_k,
                        half,
                        seg,
                        sectors,
                        even,
                        chain_of,
                        interior,
                        apex_blend,
                        interior_blend,
                        &mut grid,
                        slots,
                        result,
                    );
                    self.emit_patch_face(
                        &[a, b, c, d],
                        reverse,
                        material,
                        sector_representatives[i],
                        result,
                    );
                }
            }
        }

        if even {
            // The even centre is already welded: every sub-patch's innermost cell
            // meets the shared apex vertex, so no further faces are needed.
            return;
        }

        // Odd centre (§8.5.5.5(e)): bridge each pair of adjacent sub-patches with a
        // strip of `half` quads — sub-patch `i`'s inner column (`k = half`) against
        // sub-patch `(i+1)`'s inner row (`j = half`) — covering the chains' central
        // segments, then close the `G_i` ring with one central `s`-gon face. This bridge
        // strip is the sector's **centre column** — the `k == segments/2` run that meets
        // the centre n-gon (BEVEL.md §6.6.2 / §8.5) — so it carries the centre-column
        // donor, not the sector's non-centre grid donor. (For vertex bevel the two donors
        // coincide: both are the next-boundvert rep the caller supplies.)
        for i in 0..sectors {
            let next = (i + 1) % sectors;
            let mut column: Vec<VertexKey> = Vec::with_capacity(half + 1);
            let mut row: Vec<VertexKey> = Vec::with_capacity(half + 1);
            for r in 0..=half {
                column.push(self.adj_grid_vert(
                    i,
                    r,
                    half,
                    half,
                    seg,
                    sectors,
                    even,
                    chain_of,
                    interior,
                    apex_blend,
                    interior_blend,
                    &mut grid,
                    slots,
                    result,
                ));
                row.push(self.adj_grid_vert(
                    next,
                    half,
                    r,
                    half,
                    seg,
                    sectors,
                    even,
                    chain_of,
                    interior,
                    apex_blend,
                    interior_blend,
                    &mut grid,
                    slots,
                    result,
                ));
            }
            for r in 0..half {
                self.emit_patch_face(
                    &[column[r], row[r], row[r + 1], column[r + 1]],
                    reverse,
                    material,
                    centre_column_representatives[i],
                    result,
                );
            }
        }

        let mut central: Vec<VertexKey> = Vec::with_capacity(sectors);
        for i in 0..sectors {
            central.push(self.adj_grid_vert(
                i,
                half,
                half,
                half,
                seg,
                sectors,
                even,
                chain_of,
                interior,
                apex_blend,
                interior_blend,
                &mut grid,
                slots,
                result,
            ));
        }
        // §6.6.1 / §8.5.5.5(e): the centre n-gon is the grid's one **ring-spanning** face —
        // its corners span the whole boundvert ring — so it takes the ring-spanning donor,
        // not the sector-positional one the rest of the grid carries.
        self.emit_patch_face(&central, reverse, material, central_representative, result);
    }

    /// Resolve the adjacency-grid vert at sub-patch index `(i, j, k)` (BEVEL.md
    /// §8.5.5). Boundary rows (`j == 0` / `k == 0`) reuse the existing chain /
    /// boundvert keys so the patch shares its rim with the strips — the rim is the
    /// deterministic §8.2.2.1.1 read-back and is never re-derived here. Interior verts
    /// are created once from `interior` and cached. For even `segments` the inner column
    /// (`k == half`), inner row (`j == half`), and centre alias onto the neighbouring
    /// sub-patch (the welded seams / apex); odd `segments` keeps each sub-patch
    /// independent.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn adj_grid_vert(
        &mut self,
        i: usize,
        j: usize,
        k: usize,
        half: usize,
        seg: usize,
        sectors: usize,
        even: bool,
        chain_of: &[Vec<VertexKey>],
        interior: &AdjInterior<'_>,
        apex_blend: Option<&ApexBlend>,
        interior_blend: Option<&AdjInteriorBlend<'_>>,
        grid: &mut std::collections::HashMap<(usize, usize, usize), VertexKey>,
        slots: &mut AdjNetSlots,
        result: &mut BevelResult,
    ) -> VertexKey {
        // `j = 0` is the chain `i` boundary (from `B_i`); `k = 0` is the chain
        // `(i−1)` boundary (toward `B_i`, hence reversed). Their shared corner
        // `(0, 0)` is `B_i` from either chain.
        if j == 0 {
            return chain_of[i][k];
        }
        if k == 0 {
            return chain_of[(i + sectors - 1) % sectors][seg - j];
        }

        // Interior. Even `segments` welds the seams / apex by aliasing the inner
        // column / row / corner onto one canonical sub-patch; odd keeps `(i, j, k)`.
        let canon = if even {
            if j == half && k == half {
                (0, half, half) // the single welded apex
            } else if k == half {
                (i, j, half) // inner column = seam, owned by sub-patch i
            } else if j == half {
                ((i + sectors - 1) % sectors, k, half) // inner row aliases prev sub-patch's seam
            } else {
                (i, j, k)
            }
        } else {
            (i, j, k)
        };
        if let Some(&existing) = grid.get(&canon) {
            return existing;
        }

        let position = interior.position(canon.0, canon.1, canon.2, half);
        let vertex = match apex_blend {
            // §5.2.4.1: the even-`segments` vertex-mode welded apex is minted from the
            // corner `v` (baseline, so `NoInterp` layers keep `cd(v)`) and its
            // interpolable layers take the edge-linear snap blend toward `far`.
            Some(blend) if even && j == half && k == half => self.mint_welded_apex(position, blend),
            // §5.2.4.2: at even `segments ≥ 4` every other chain-interior vert is minted
            // from `v` and blended per corner in its canonical sector. On the sector
            // diagonal (`canon.j == canon.k`, non-apex) it takes the edge-linear snap onto
            // that sector's own boundvert edge `e_i`; off the diagonal it takes the plain
            // mean-value blend from the next boundvert's rep face `f2`.
            //
            // §5.2.4.2 (lines ~1574–1580): the mint-from-`v` baseline is parity-independent —
            // *every* interior grid vert is a §5.2.1 step-1 copy of `v`'s block, so a
            // `NoInterp` layer keeps `cd(v)` and interpolable layers keep the `cd(v)` baseline
            // rather than falling to a layer default. At **odd** `segments` the central n-gon's
            // `(i, h, h)` corner verts take the §5.2.4.3 own-rep interior blend (no-seam
            // branch); every other odd interior vert keeps the bare `cd(v)` baseline (its
            // donor is still a §5.2.5 stub). Edge mode carries no `interior_blend`, so it keeps
            // the plain `create_vertex` default.
            _ => match interior_blend {
                Some(blend) if even => self.mint_chain_interior_vert(position, canon, blend),
                Some(blend) if canon.1 == half && canon.2 == half => {
                    self.mint_centre_ngon_corner(position, canon.0, blend)
                }
                Some(blend) => self.mint_bevel_vertex(position, blend.corner_vertex),
                None => self.create_vertex(position),
            },
        };
        result.verts.push(vertex);
        slots.record(canon);
        grid.insert(canon, vertex);
        vertex
    }

    /// Mint the even-`segments` vertex-mode welded apex with the §5.2.4.1 edge-linear
    /// snap blend. The apex's whole per-vertex block starts as a copy of the corner
    /// `v`'s (§5.2.1 step 1), so a `NoInterp` layer keeps `cd(v)` (BMESH.md §10.1.4);
    /// its interpolable layers are then overwritten by `cd(v) + t·(cd(far) − cd(v))`,
    /// where `t` is the apex's clamped closest-point projection onto the `v → far`
    /// snap-edge segment taken at the apex's final `position`. `interpolate_from_snapshot`
    /// leaves the `NoInterp` layers untouched, so the two steps compose exactly.
    pub(super) fn mint_welded_apex(&mut self, position: Vec3, blend: &ApexBlend) -> VertexKey {
        let apex = self.mint_bevel_vertex(position, blend.corner_vertex);
        let axis = blend.snap_to - blend.snap_from;
        let length_squared = axis.length_squared();
        let t = if length_squared <= f32::EPSILON {
            0.0
        } else {
            ((position - blend.snap_from).dot(axis) / length_squared).clamp(0.0, 1.0)
        };
        self.vertex_attribs.interpolate_from_snapshot(
            apex.as_usize(),
            &blend.endpoints,
            &[1.0 - t, t],
        );
        apex
    }

    /// Mint a §5.2.4.2 even-`segments` [V-ADJ] chain-interior vert (a non-apex interior
    /// grid vert) from the beveled corner `v` and blend its per-vertex block by the
    /// per-corner rule of its canonical sector `canon.0`. Minting from `v` is the §5.2.1
    /// step-1 baseline, so a `NoInterp` layer keeps `cd(v)` instead of falling to a layer
    /// default (the pre-fix behaviour these verts had).
    ///
    /// On the sector diagonal (`canon.j == canon.k`) the corner is snapped onto the
    /// sector's own boundvert edge and takes the edge-linear closed form — the same
    /// `mint_welded_apex` machinery, keyed on that sector's snap. Off the diagonal it
    /// takes the plain mean-value blend of the next boundvert's representative face `f2`
    /// at the vert's own position (no snap). A sector with no snap edge / no `f2` donor —
    /// the gap leg of an open fan — keeps the bare `cd(v)` mint.
    pub(super) fn mint_chain_interior_vert(
        &mut self,
        position: Vec3,
        canon: (usize, usize, usize),
        blend: &AdjInteriorBlend<'_>,
    ) -> VertexKey {
        let (sector, j, k) = canon;
        if j == k {
            if let Some(Some(snap)) = blend.sector_snaps.get(sector).map(|s| s.as_ref()) {
                return self.mint_welded_apex(position, snap);
            }
            return self.mint_bevel_vertex(position, blend.corner_vertex);
        }
        match blend.sector_donors.get(sector).and_then(|d| *d) {
            Some(donor) => {
                let vertex = self.mint_bevel_vertex(position, blend.corner_vertex);
                let sample = donor.projection.project(position);
                let weights = mean_value_weights(&donor.proj_corners, sample);
                self.vertex_attribs.interpolate_from_snapshot(
                    vertex.as_usize(),
                    &donor.corner_vertex_blocks,
                    &weights,
                );
                vertex
            }
            None => self.mint_bevel_vertex(position, blend.corner_vertex),
        }
    }

    /// Mint an odd-`segments` centre n-gon corner `(i, h, h)` with the §5.2.4.3 no-seam
    /// blend. The corner is minted from the beveled corner `v` (§5.2.1 step-1 baseline, so a
    /// `NoInterp` layer keeps `cd(v)`), then its interpolable per-vertex layers are overwritten
    /// by the plain §10.1 interior mean-value blend of boundvert `sector`'s **own**
    /// representative face at the corner's final position — no weight-snap (the innermost vert
    /// sits well inside the donor, so no §10.1.3 boundary short-circuit fires). Each corner
    /// reads its own sector's rep, so the value is per-sector-private and anchor-free. A `None`
    /// donor (a seam-carrying corner — the deferred branch — or a gap leg) keeps the bare
    /// `cd(v)` baseline.
    pub(super) fn mint_centre_ngon_corner(
        &mut self,
        position: Vec3,
        sector: usize,
        blend: &AdjInteriorBlend<'_>,
    ) -> VertexKey {
        match blend.centre_donors.get(sector).and_then(|d| *d) {
            Some(donor) => {
                let vertex = self.mint_bevel_vertex(position, blend.corner_vertex);
                let sample = donor.projection.project(position);
                let weights = mean_value_weights(&donor.proj_corners, sample);
                self.vertex_attribs.interpolate_from_snapshot(
                    vertex.as_usize(),
                    &donor.corner_vertex_blocks,
                    &weights,
                );
                vertex
            }
            None => self.mint_bevel_vertex(position, blend.corner_vertex),
        }
    }

    /// Emit one adjacency-grid patch face (BEVEL.md §8.5.5). `verts` arrives in the cap's own
    /// walk order and `reverse` is the cap's one winding bit — decided once for the whole cap
    /// by [`Self::patch_rim_needs_reversing`], never per face and never from geometry, so
    /// every face of a cap is consistent with its siblings by construction. Consecutive
    /// duplicate corners (none expected, a guard) are dropped.
    pub(super) fn emit_patch_face(
        &mut self,
        verts: &[VertexKey],
        reverse: bool,
        material: i32,
        representative: FaceTemplateRef<'_>,
        result: &mut BevelResult,
    ) {
        let mut ring: Vec<VertexKey> = Vec::with_capacity(verts.len());
        for &vertex in verts {
            if ring.last() != Some(&vertex) {
                ring.push(vertex);
            }
        }
        if ring.len() >= 2 && ring.first() == ring.last() {
            ring.pop();
        }
        if ring.len() < 3 {
            return;
        }

        if reverse {
            ring.reverse();
        }

        let face = self.face_create_verts(&ring, FaceCreateFlags::NONE);
        self.assign_bevel_face_attributes(face, material, representative, BevelNewFaceKind::Cap);
        result.faces.push(face);
        self.collect_face_edges(face, result);
    }

    /// One rim edge's vote on a cap's winding (BEVEL.md §5.1), read with the edge walked in
    /// the cap's **own** sense `from → to`: `+1` keep that walk, `−1` turn it round, `0`
    /// abstain.
    ///
    /// §5.1's manifold condition is that the cap and the face it shares a rim edge with
    /// traverse that edge in *opposite* directions, so a neighbour walking `to → from` wants
    /// the cap's walk kept and one walking `from → to` wants it reversed. An absent edge, or
    /// one no face has reached yet — §5.1's face-less star — abstains for want of anything to
    /// be manifold against.
    ///
    /// **One vote per loop, not per edge — a deliberate weighting.** §5.1 is stated per
    /// *shared edge*, so counting each already-attached face separately is a choice, and it has
    /// a visible consequence: a rim edge that is already radial-2 and internally **consistent**
    /// cancels to an abstention (correctly — a third face cannot bring it to radial 2 either
    /// way, so it has nothing to say about the cap), while one that is already radial-2 and
    /// **inconsistent** — both its faces walking it the same way — casts a **double-weight**
    /// vote, so one conflicted rim edge can outvote one clean one. That is the intended
    /// reading: a conflicted rim edge has two loops the cap could still be made manifold
    /// against, and each is a claim on the cap's sense, whereas a clean radial-2 edge has none
    /// left. It is a reachable object here rather than a hypothetical — the (f.4a) weld's leg
    /// edges are genuinely radial-2 before the cap closes — and it is pinned by
    /// `patch_rim_winding_is_an_aggregate_over_the_whole_rim`'s doubled arms.
    pub(super) fn rim_edge_winding_vote(&self, from: VertexKey, to: VertexKey) -> i32 {
        let Some(rim_edge) = self.query_edge_exists(from, to) else {
            return 0;
        };
        let mut vote = 0;
        for loop_key in self.edge_radial_iter(rim_edge) {
            let start = self.loop_vertex(loop_key);
            let end = self.loop_vertex(self.loop_next(loop_key));
            debug_assert!(
                (start == from && end == to) || (start == to && end == from),
                "a loop on the rim edge walks one of its two ends to the other",
            );
            vote += if start == from && end == to { -1 } else { 1 };
        }
        vote
    }

    /// The one winding bit for a whole [V-ADJ] cap (BEVEL.md §5.1, applied to the grid by
    /// §8.5.3): does the cap's own walk need reversing to leave every rim edge it shares with
    /// an arriving strip at radial count 2? `rim` yields the cap's rim edges as directed
    /// pairs, each walked the way the cap itself walks it.
    ///
    /// Deciding it **once, combinatorially, for the whole cap** is the point. Every cell of
    /// every sub-patch, the odd centre's bridge strips and its centre `s`-gon are indexed off
    /// one `(i, j, k)` frame, so they agree with each other by construction; a per-face
    /// geometric test can only break that agreement, and cannot orient a cap over a
    /// non-planar (saddle) ring at all.
    ///
    /// **Aggregate, not first-found.** What licenses the rule is §5.1 itself: the manifold
    /// condition is stated *per shared edge*, every rim edge states it, and on a cap that has
    /// only one winding bit to give they can conflict — so the bit is the sum of
    /// [`Self::rim_edge_winding_vote`] over the whole rim (see there for the per-loop
    /// weighting), reversed **iff the sum is strictly negative**.
    ///
    /// The **tie-break** is §7.6.1.2's, and only the tie-break: a tie breaks toward doing
    /// nothing, so an even split, or a rim with nothing to be manifold against, leaves the walk
    /// standing, pinned by the disk-order convention alone (§5.1's face-less star, where the
    /// winding is not observable anyway). The rest of §7.6.1.2's box is a *different* rule and
    /// is not claimed here — it abstains only on "no forward face at all", and it resolves
    /// several candidate faces by **selection** (the walk's own choice among them) rather than
    /// by summing them.
    ///
    /// Both halves are load-bearing rather than decorative, because the votes genuinely can
    /// disagree. On an inconsistently wound input (§7.8) different sectors' strips inherit
    /// different senses from the original faces they were stitched to, and no cap winding
    /// satisfies all of them: a first-found answer there is a coin toss keyed on which sector
    /// the probe happened to reach first, the sum takes the majority instead, and an even split
    /// — reachable at even valence — is what the tie-break then has to settle.
    pub(super) fn patch_rim_needs_reversing(
        &self,
        rim: impl Iterator<Item = (VertexKey, VertexKey)>,
    ) -> bool {
        rim.map(|(from, to)| self.rim_edge_winding_vote(from, to))
            .sum::<i32>()
            < 0
    }

    /// [`Self::patch_rim_needs_reversing`] over a [V-ADJ] grid cap's rim: the grid's cell
    /// order walks its rim **ascending** along each sector chain, and the chains ascend in
    /// sector order — that is the disk order, which §5.1 says the cap runs *with*.
    ///
    /// **Why the answer can legitimately be `true` at all**, given that the ascent is not an
    /// independent convention: a chain ascends from `B_i`, and the ring the `B`s sit in is
    /// itself built by the §7.6.2 winding walk, so on a consistently wound corner the rim's
    /// ascent and the arriving strips' traversal of it flip **together** and this returns
    /// `false` every time. What makes `true` reachable is the aggregate, not the ascent: at a
    /// §7.8 junction the strips inherit their senses from *different* original faces, and when
    /// two of three sectors dissent from the sense the ring's own walk implies, the majority
    /// reverses the cap. So this stays a query and must **not** be hardened into an assertion
    /// that it always answers `false`.
    pub(super) fn adj_grid_needs_reversing(
        &self,
        sectors: usize,
        seg: usize,
        chain_of: &[Vec<VertexKey>],
    ) -> bool {
        let rim_edge = |i: usize, k: usize| (chain_of[i][k], chain_of[i][k + 1]);
        self.debug_assert_rim_agrees_within_each_sector(sectors, seg, rim_edge);
        self.patch_rim_needs_reversing(
            (0..sectors).flat_map(|i| (0..seg).map(move |k| rim_edge(i, k))),
        )
    }

    /// Sector-local agreement guard for [`Self::adj_grid_needs_reversing`]'s vote, and for the
    /// `segments = 2` pinwheel's, whose rim is the same object at `seg = 2`. `rim_edge(i, k)`
    /// is sector `i`'s `k`-th rim edge, walked the way the cap walks it.
    ///
    /// A sector's rim is the whole of one arriving strip's terminal boundary, so every faced
    /// edge of it carries faces of that one strip — which is internally consistent by
    /// construction. Disagreement is therefore only ever a *between*-sector fact, and that is
    /// exactly the §7.8 case the aggregate exists to decide. Asserting unanimity across the
    /// whole rim instead would fire on an inconsistently wound input, which is a mesh we are
    /// meant to survive rather than a bug in the probe.
    ///
    /// **What the "one sector, one strip" premise covers.** It is stated for a plain strip
    /// arrival, and it extends to the two arrivals that are not plain, but not equally firmly:
    ///
    /// - a **§7.8-collapsed flank** is covered outright — the collapse changes the rung's
    ///   *width*, not the build, and still lays `segments` quads across it in one pass
    ///   (§7.8.4), so the terminal chain still carries one strip's consistent traversal;
    /// - a **§8.9 miter** sector's rim is the arc run `B1 → bulge → B3` that the sector's
    ///   rebuilt face(s) re-stitch onto (§8.9.3), which is one rebuild's boundary rather than
    ///   one strip's. The miter ring fills route through here, so this guard is the thing that
    ///   would surface a counterexample; it is quiet across the miter fixtures, and a firing
    ///   here on a *consistently* wound miter would be a finding, not a false alarm.
    ///
    /// What the premise does **not** cover either way is a sector whose rim carries faces from
    /// two different builds — which no path here produces.
    pub(super) fn debug_assert_rim_agrees_within_each_sector(
        &self,
        sectors: usize,
        seg: usize,
        rim_edge: impl Fn(usize, usize) -> (VertexKey, VertexKey),
    ) {
        if !cfg!(debug_assertions) {
            return;
        }
        for i in 0..sectors {
            let votes: Vec<i32> = (0..seg)
                .map(|k| {
                    let (from, to) = rim_edge(i, k);
                    self.rim_edge_winding_vote(from, to)
                })
                .filter(|&vote| vote != 0)
                .collect();
            debug_assert!(
                votes
                    .windows(2)
                    .all(|pair| pair[0].signum() == pair[1].signum()),
                "sector {i}'s rim edges disagree on the cap winding ({votes:?}) — one \
                 sector's rim is one strip's boundary, so it cannot",
            );
        }
    }
}

/// `|sin θ|` bound below which two offset lines of the §8.5.5.5(f.5) square-grid
/// lattice count as **aligned** (parallel-but-distinct *or* coincident). The guide
/// pins the interior meet to be *routed on directions alone*: once the `k`-th and
/// `j`-th offset families share a direction — as a sector opens toward flat — the
/// line-line solve is meaningless (its closest-approach parameters blow up), so the
/// lattice interpolates the rim row toward the swept centreline instead.
///
/// The band has to be caught *here*, by this caller, because the shared solve has no
/// parallel band of its own to fall into — its bail is an exact zero (§7.6.2), so an
/// aligned-but-not-bit-identical pair would solve as a crossing tens of units out
/// rather than reporting `Collinear`. Wide enough to catch that, still far tighter
/// than any genuine crossing on a non-degenerate grid.
pub(super) const OFFSET_LINE_ALIGNED_TOLERANCE: f32 = 1.0e-3;

/// Where an adjacency grid's **interior** verts come from (BEVEL.md §8.5.5). The rim
/// (`j == 0` / `k == 0`) is never sourced here — it is always the already-emitted chain,
/// the deterministic §8.2.2.1.1 read-back — so this covers only `j, k ≥ 1`.
pub(super) enum AdjInterior<'a> {
    /// The raw bilinear sub-patch blend. Used **only** by the flat (`r = 1`) planar cap at
    /// even `segments = 2`, which welds at the ring-plane centroid (`fullness = 0`) with no
    /// interior ring to relax and no surface to snap onto — the blend *is* the fill
    /// (§8.3.0.2.1). Every other flat cap — all odd `segments`, and even `segments ≥ 4` whose
    /// interior grid is a *determined* quantity — takes the [`AdjInterior::Relaxed`] descent
    /// instead (§8.5.5.5(e) / (f.3) law **F2**).
    Blend { corners: &'a [[Vec3; 4]] },
    /// The §8.5.5.5(c) subdivision-relaxation net, read back at the target resolution by
    /// the §8.5.5.5(d) kernel — bilinear within the source quad, against each axis's own
    /// **profile-arc fractions** (`axis[i]`, the [`SourcePosition`] the §8.2.2.1.1
    /// read-back puts chain `i`'s target sample at). The kernel is the licensed Tier-2
    /// interior freedom (§8.5.5.7), and it is licensed for these interior rows *only* —
    /// the rim is never read back through it. Parametrising against the rim's own
    /// fractions is what keeps the interior consistent with the rim it meets.
    Relaxed {
        net: &'a AdjNet,
        /// Per ring chain, `segments + 1` [`SourcePosition`]s into the dense net — span
        /// index and remainder carried apart, never summed.
        axis: &'a [Vec<SourcePosition>],
        sectors: usize,
        segments: usize,
    },
    /// The same §8.5.5.5(c)/(d) net and read-back, but carried in **unit-octant**
    /// coordinates (§8.5.5.5(f.2)): each interior vert is read off the octant net, snapped
    /// onto the unit superellipsoid `u_x^r + u_y^r + u_z^r = 1`, and only then mapped into
    /// the corner. The snap is what the world-space sphere projection of §8.5.5.3 is a
    /// special case of; the map is applied once, at the very end, so nothing downstream
    /// needs the corner's (non-existent, for curved `r`) circumsphere.
    Octant {
        net: &'a AdjNet,
        axis: &'a [Vec<SourcePosition>],
        sectors: usize,
        segments: usize,
        exponent: f32,
        octant: UnitOctant,
    },
    /// The §8.5.5.5(f.5) **general-ring** square grid — the planar parallel-offset lattice.
    /// Interior verts are line-line meets of the two families of offset lines swept from the
    /// rim toward `v`; `v` survives as the exact welded innermost vert. `starts[i]` is
    /// `(B_i, C_i)`: the sector's opening boundvert and its centreline start — the exact
    /// §8.5.5.5(f.5) line-line seed on the shared beveled edge (from the boundvert-granularity
    /// seeding pass, *not* the sector chain's fold midpoint). The caller uses the same `C_i` to
    /// reposition the straight rim rows, so rim and interior agree on the seed.
    Parallel {
        starts: &'a [(Vec3, Vec3)],
        vertex: Vec3,
    },
    /// The §8.5.5.5(f.5) **tri-corner** square grid — three flat unit squares, closed-form in
    /// octant coordinates with no subdivision and no snap, mapped to the corner by `octant`.
    SquareTriCorner { octant: UnitOctant, segments: usize },
}

impl AdjInterior<'_> {
    /// The interior position of sub-patch `i`'s grid vert `(j, k)` at target half-width
    /// `half`. Callers pass the *canonical* `(i, j, k)` so a welded seam / centre resolves
    /// to one position.
    pub(super) fn position(&self, i: usize, j: usize, k: usize, half: usize) -> Vec3 {
        match self {
            AdjInterior::Blend { corners } => {
                let [p00, p0t, pt0, ptt] = corners[i];
                let s = k as f32 / half as f32;
                let u = j as f32 / half as f32;
                p00 * ((1.0 - s) * (1.0 - u))
                    + p0t * (s * (1.0 - u))
                    + pt0 * ((1.0 - s) * u)
                    + ptt * (s * u)
            }
            AdjInterior::Relaxed {
                net,
                axis,
                sectors,
                segments,
            } => {
                if let Some(centre) = net.welded_centre(*segments, j, k, half) {
                    return centre;
                }
                // Sub-patch `i`'s `k` axis is chain `i` read forward; its `j` axis is chain
                // `i − 1` read *backward* from `B_i`, so its source coordinate is the
                // reflection of that chain's position at target sample `segments − j`. Only
                // the `j` axis reflects, and the reflection is the entire difference between
                // them — including the sidedness, since it is what supplies `j`'s second
                // snap application and hence its from-below arm.
                let resolution = 2 * net.half;
                let previous = (i + sectors - 1) % sectors;
                let along_chain = axis[i][k].snapped();
                let along_previous = ring_source_position(axis[previous][segments - j], resolution);
                net.sample(i, along_previous, along_chain)
            }
            AdjInterior::Octant {
                net,
                axis,
                sectors,
                segments,
                exponent,
                octant,
            } => {
                let unit = match net.welded_centre(*segments, j, k, half) {
                    Some(centre) => centre,
                    None => {
                        let resolution = 2 * net.half;
                        let previous = (i + sectors - 1) % sectors;
                        let along_chain = axis[i][k].snapped();
                        let along_previous =
                            ring_source_position(axis[previous][segments - j], resolution);
                        net.sample(i, along_previous, along_chain)
                    }
                };
                octant.to_world(snap_to_unit_superellipsoid(unit, *exponent, false))
            }
            AdjInterior::Parallel { starts, vertex } => {
                let sectors = starts.len();
                let h = half as f32;
                let (b_i, c_i) = starts[i];
                let c_prev = starts[(i + sectors - 1) % sectors].1;

                // The welded innermost vert is `v` itself — the square profile's un-rounded
                // corner (§8.5.5.5(f.5)).
                if j == half && k == half {
                    return *vertex;
                }

                // The two swept centrelines. `CL_i` runs `C_i → v`; `CL_{i−1}` runs
                // `C_{i−1} → v`. Both reach `v` at step `h`.
                let cl_i_j = c_i + (j as f32 / h) * (*vertex - c_i);
                let cl_prev_k = c_prev + (k as f32 / h) * (*vertex - c_prev);

                // The seam column (`k == half`) *is* `CL_i` walked to `v` — pin it exactly so
                // the seam the two sub-patches share stays a straight centreline.
                if k == half {
                    return cl_i_j;
                }

                // The two straight rim rows the interior line-families are cast between.
                let row0_k = b_i + (k as f32 / h) * (c_i - b_i); // (i, 0, k)
                let col0_j = b_i + (j as f32 / h) * (c_prev - b_i); // (i, j, 0)

                // §8.5.5.5(f.5): the `k`-th offset line meets the `j`-th offset line. The two
                // swept lines are symmetric peers with no anchored edge, so the skew midpoint
                // is the final answer here (one of only two sites licensed to take it).
                let line_k = Line {
                    point: row0_k,
                    direction: cl_prev_k - row0_k,
                };
                let line_j = Line {
                    point: col0_j,
                    direction: cl_i_j - col0_j,
                };
                // (f.5)'s parallel branch: interpolate `(i, 0, k)` toward `CL_{i−1}[k]` by
                // `j/h`. Continuous with the crossing (they coincide wherever the lines meet),
                // so it is the correct limit as the two families align.
                let interpolate = row0_k + (j as f32 / h) * (cl_prev_k - row0_k);
                // Route on the offset lines' **directions alone** (§8.5.5.5(f.5), the
                // parallel-but-distinct paragraph): once they share a direction — parallel yet
                // cast from two different rim rows, so never coincident — the line-line solve
                // is unstable and its skew / degenerate fallback is *not* the lattice's answer.
                // A merely-coincident test (the solver's own `Collinear`) misses this, so test
                // the directions explicitly.
                let dir_k = line_k.direction.normalize_or_zero();
                let dir_j = line_j.direction.normalize_or_zero();
                let aligned = dir_k == Vec3::ZERO
                    || dir_j == Vec3::ZERO
                    || dir_k.cross(dir_j).length() <= OFFSET_LINE_ALIGNED_TOLERANCE;
                if aligned {
                    interpolate
                } else {
                    // The lattice's own reading of a skew pair is the midpoint between
                    // the two offset families, not §7.6.2's on-the-first-line answer:
                    // neither family is privileged here.
                    match intersect_lines(&line_k, &line_j) {
                        LineMeet::Crossing(point)
                        | LineMeet::Skew {
                            midpoint: point, ..
                        } => point,
                        LineMeet::Collinear => interpolate,
                    }
                }
            }
            AdjInterior::SquareTriCorner { octant, segments } => {
                // §8.5.5.5(f.5), closed form: `coord i = 1`, `coord (i+1) mod 3 = 2k/n`,
                // `coord (i+2) mod 3 = 2j/n`; no subdivision, no snap.
                let n = *segments as f32;
                let mut coord = [0.0_f32; 3];
                coord[i] = 1.0;
                coord[(i + 1) % 3] = 2.0 * k as f32 / n;
                coord[(i + 2) % 3] = 2.0 * j as f32 / n;
                octant.to_world(Vec3::new(coord[0], coord[1], coord[2]))
            }
        }
    }
}
