//! The pipe override and pipe replant (BEVEL.md §8.3.4.1 / §8.5.5.5(f.6)).

use super::*;

impl LoopMesh {
    /// Mint the §8.3.4.1 pipe-override **loose leftover verts** at a re-tiled junction.
    ///
    /// A CUTOFF ring that clears the pipe gate has its connected fill delegated to the
    /// ADJ grid, byte-identical to a from-scratch ADJ bevel (topology *and* positions,
    /// the locally flat neighbourhood of condition 3 removing every position freedom).
    /// But the cutoff pass had already laid down — per beveled boundvert — that
    /// boundvert's profile-arc `segments − 1` interior points as real verts, *before*
    /// the pipe test redirected the fill; the ADJ builder then rebuilt its own boundary
    /// arc and never consumed them. They persist as valence-0 verts in **no edge and no
    /// face**. This re-creates that surplus: `(segments − 1) · s` loose verts, `s` being
    /// the ring's beveled-edge count (= its boundvert count here, one beveled spoke per
    /// sector), placed at the **CUTOFF profile arc** interior samples — the raw
    /// (un-resampled) superellipse the cutoff fill reads (§8.2.2.1.1 Scope: the cutoff
    /// fill is not a [V-ADJ] cap, so it emits `ChainEmission::Raw`). At `segments = 2`
    /// each lands on its profile midpoint (coincident with the ADJ mid-edge ring node);
    /// at `segments ≥ 3` the cutoff samples differ from the ADJ interior samples and the
    /// leftovers sit at their own distinct positions. E and F are untouched.
    ///
    /// **Known gap — no custom awareness (§8.6.3).** The leftovers are recomputed from the
    /// analytic arc off `profile` alone, so under a hand-drawn curve they land on the
    /// superellipse the slider names rather than on the curve the fill used. Positions only —
    /// the count is topological and correct either way. One of the two sites still owed the
    /// threading, with [`pipe_cross_section_replant`].
    pub(super) fn mint_pipe_override_loose_verts(
        &mut self,
        cut: &JunctionCut,
        chains: &[EdgeEndChain],
        segments: u32,
        profile: f32,
        result: &mut BevelResult,
    ) {
        if segments < 2 {
            return;
        }
        let count = cut.boundverts.len();
        for k in 0..count {
            let next = (k + 1) % count;
            let start_boundvert = cut.boundverts[k];
            let end_boundvert = cut.boundverts[next];
            let chain = chains.iter().find(|c| {
                c.endpoint == cut.vertex
                    && ((c.start == start_boundvert && c.end == end_boundvert)
                        || (c.start == end_boundvert && c.end == start_boundvert))
            });
            // Every consecutive boundvert pair of a pipe-override ring was chained (with a
            // profile cell) by `build_junction_chains` just above; a missing chain or cell
            // is an internal-invariant break, not a graceful skip.
            let Some(chain) = chain else {
                debug_assert!(false, "pipe-override sector {k} has no chain to sample");
                return;
            };
            let Some(cell) = chain.cell.as_ref() else {
                debug_assert!(
                    false,
                    "pipe-override sector {k} chain carries no profile cell"
                );
                return;
            };
            let positions = cell.chain_positions(segments, profile, None, ChainEmission::Raw);
            for position in positions.iter().take(positions.len() - 1).skip(1) {
                // §5.1.1: minted loose — in no edge and no face. No later phase re-points
                // or frees it (the wire re-attach and endpoint-free passes key on named
                // endpoints, not on new verts), so it stays orphaned, matching the
                // reference operator's leftover.
                let leftover = self.create_vertex(*position);
                result.verts.push(leftover);
            }
        }
    }

    /// Replant the ring-chain rim samples the canonical slot walk reaches (BEVEL.md
    /// §8.5.5.5(f.6.1)).
    ///
    /// The `k = 0` column of sub-patch `i` is not interior at all — through the net's seam
    /// aliasing it *is* ring chain `i − 1`, walked backwards from `B_i` — so the replant's
    /// `j >= 1` block writes through to the strict second half of every chain
    /// ([`pipe_replant_rim_window`]), while the first half keeps the chord point
    /// §8.2.2.1.1's read-back left it at. Sample `k` and sample `n − k` are mirror partners
    /// and exactly one of the pair moves, so a symmetric tube's cap is deliberately **not**
    /// mirror-symmetric; replanting the whole rim (or none of it) gives the prettier uniform
    /// cap and is wrong.
    ///
    /// Runs after [`Self::build_adj_fill_base`], and only when it actually tiled a net: the
    /// order is part of the rule (replanting first would let the read-back re-pin the rim on
    /// top of it), and a declined base has no slots to walk at all.
    ///
    /// Each sector's chain is resolved through [`Self::sector_chain`] and walked from the
    /// same boundvert [`Self::oriented_chain`] walks it from, so the samples this touches are
    /// the ones [`Self::adj_grid_vert`] hands out for the net's `k = 0` column — the two must
    /// index the same chain the same way or the window means nothing.
    pub(super) fn replant_pipe_rim_window(
        &mut self,
        cut: &JunctionCut,
        chains: &[EdgeEndChain],
        segments: usize,
        pipe: &PipeCrossSection,
        exponent: f32,
    ) {
        let window = pipe_replant_rim_window(segments);
        if window.is_empty() {
            return;
        }
        let sectors = cut.boundverts.len();
        for sector in 0..sectors {
            let start = cut.boundverts[sector];
            let Some(chain) = self.sector_chain(cut, chains, sector) else {
                continue;
            };
            // A chain that did not sample to `segments + 1` verts carries no net slot to
            // alias onto, so the walk never reaches it. Redundant now that the caller only
            // runs this on an emitted net — every base builder refuses a short chain outright
            // — but the window means nothing without it, so it stays local to the walk.
            if chain.interior.len() + 2 != segments + 1 {
                continue;
            }
            for sample in window.clone() {
                let Some(vertex) = chain.oriented_sample(start, sample) else {
                    continue;
                };
                let base = self.vertex_position(vertex);
                // Every slot this half reaches sits in the net's `k = 0` column, and
                // §8.5.5.5(f.6.1)'s `land_on_corner` rule wants `k = ⌊n/2⌋` — which `k = 0`
                // is not at any `segments ≥ 2`. So the rim never lands on the corner.
                let replanted = pipe_cross_section_replant(
                    base,
                    pipe.axis,
                    &pipe.cell,
                    pipe.placement,
                    exponent,
                    false,
                );
                self.set_vertex_position(vertex, replanted);
            }
        }
    }

    /// Recover a §8.5.5.5(f.6) pipe junction's axis, the pipe boundvert's profile cell, and
    /// which of the two placements every station will take (BEVEL.md §8.5.5.5(f.6)).
    ///
    /// The **axis** is §8.3.4.1's own: the line shared by the pipe boundvert's spoke and its
    /// collinear opposite ([`pipe_boundvert_spokes`]), taken off the pristine directions the
    /// cut captured with the pipe verdict. It is deliberately not read back off a cell's
    /// projection direction — that field is the crossed edge's line only until §8.2.2.1's
    /// degenerate-`mid` rescue repoints it at the rescued *plane normal*, which at a planar
    /// junction is the one direction under which the flat cross-section would falsely
    /// survive projection.
    ///
    /// The **shape** is the pipe boundvert's own profile: the chain that crosses its own
    /// (pipe) edge. Only a pipe-edge chain spans the cross-section — the two boundverts
    /// flanking a *transverse* edge sit at the same transverse coordinate and differ only in
    /// station, so the projection that forgets the station carries both onto one point.
    ///
    /// `station_cell_placement` is [`pipe_station_placement`]'s verdict on that cell: whether
    /// the replant builds a station cell at all, or takes one of step 4's collapses at every
    /// station. A **planar** junction is the standard instance of the latter (§8.3.4.1's flat
    /// anti-parallel / wire-disk configuration): its whole cross-section lies in one plane
    /// that contains the pipe axis, and a plane containing the projection direction has a line
    /// for its image. The clamp still runs there — on a concave profile it moves real verts —
    /// but nothing downstream needs the cell path.
    ///
    /// `None` when no anti-parallel pair or no pipe-edge chain is recoverable, which leaves
    /// the base grid standing.
    pub(super) fn pipe_cross_section(
        &self,
        cut: &JunctionCut,
        chains: &[EdgeEndChain],
    ) -> Option<PipeCrossSection> {
        // The spoke scan indexes the directions; its answer indexes the edges.
        debug_assert_eq!(cut.opening_edges.len(), cut.opening_edge_directions.len());
        let (pipe_edge_index, collinear_opposite) =
            pipe_boundvert_spokes(&cut.opening_edge_directions)?;
        let axis = cut.opening_edge_directions[pipe_edge_index].normalize_or_zero();
        let pipe_edge = cut.opening_edges[pipe_edge_index];
        let cell = chains
            .iter()
            .filter(|chain| chain.endpoint == cut.vertex)
            .find(|chain| chain.edge == pipe_edge)
            .and_then(|chain| chain.cell)?;
        // The **two pipe boundverts** §8.5.5.5(f.6.1)'s `land_on_corner` rule names: the pair
        // whose own edges are the collinear spokes. A boundvert's own edge closes its sector,
        // so each sector index is one ring step behind its edge index.
        let count = cut.opening_edge_directions.len();
        let sector_of = |edge_index: usize| (edge_index + count - 1) % count;
        Some(PipeCrossSection {
            axis,
            cell,
            placement: pipe_station_placement(axis, &cell),
            sectors: (sector_of(pipe_edge_index), sector_of(collinear_opposite)),
        })
    }

    /// The §8.3.4.1 **pipe-override** gate on the **original** corner: whether this multi-edge
    /// junction re-tiles its `CUTOFF` ring as the all-quad adjacency grid instead of the
    /// per-sector cut faces. All three conditions of §8.3.4.1 must hold:
    ///
    /// 1. **Ring of 3 or 4.** The junction has exactly 3 or 4 beveled spokes (one per sector,
    ///    hence 3 or 4 boundverts). At `s ≥ 5` the ordinary cut faces stand even with an
    ///    anti-parallel pair present.
    /// 2. **A collinear opposite pair.** Two beveled spokes are exactly **anti-parallel**
    ///    through `vertex` — their rays out of `v` point in opposite directions to within the
    ///    §7.6.2 near-collinear angular tolerance of **`2°`**. The shared line is the **pipe
    ///    axis**.
    /// 3. **Incident faces parallel to the pipe axis.** Every face in the junction's fan has
    ///    its normal perpendicular to the pipe axis, so the pipe runs straight through a
    ///    locally flat neighbourhood rather than turning a corner. The fan is the disk's
    ///    per-slot **forward faces**; the window is a dot-product magnitude
    ///    ([`PIPE_PERPENDICULAR_DOT_EPS`], four orders of magnitude tighter than condition 2's
    ///    `2°`), not an angle.
    ///
    /// Read from the pristine corner (its edges and faces are about to be consumed by the
    /// re-stitch), so it is captured at ring-placement time and carried on the cut.
    pub(super) fn pipe_override_admits(
        &self,
        vertex: VertexKey,
        vertex_position: Vec3,
        opening_edges: &[EdgeKey],
        corners: &[VertexCorner],
    ) -> bool {
        // Condition 1: the ring is a 3- or 4-way junction.
        let count = opening_edges.len();
        if count != 3 && count != 4 {
            return false;
        }

        // Each sector's opening beveled spoke, as a unit ray out of `vertex`.
        let mut spoke_directions: Vec<Vec3> = Vec::with_capacity(count);
        for &edge in opening_edges {
            match self.edge_direction_from(edge, vertex, vertex_position) {
                Some(direction) => spoke_directions.push(direction),
                None => return false,
            }
        }

        // Condition 2: some pair of spokes is exactly anti-parallel through `vertex` — the
        // ray out of `v` along one is the reverse of the ray out along the other.
        let pipe_axis = match pipe_boundvert_spokes(&spoke_directions) {
            Some((own, _)) => spoke_directions[own],
            None => return false,
        };

        // Condition 3: every face in the fan has its normal perpendicular to the pipe axis,
        // i.e. the pipe threads a locally flat neighbourhood. Condition 2's `2°` window is
        // **not** this window — the two measure different quantities at scales four orders of
        // magnitude apart, and sharing a constant between them is the classic way to build
        // this predicate wrong. See [`PIPE_PERPENDICULAR_DOT_EPS`].
        //
        // The fan is enumerated as one **forward face per disk slot**, not as every face
        // incident to `vertex`: on a non-manifold spoke the latter tests strictly more faces,
        // so it declines junctions that qualify. On a manifold fan the corner ring *is* that
        // slot array, so `corners` is the enumeration. On a non-manifold slot step it is not:
        // where two faces both qualify as the one step's forward face, [`vertex_corner_ring`]
        // records both, and this loop then declines on whichever is tilted while §8.3.4.1 keeps
        // the branch. Known gap, safe direction (over-test ⇒ false-decline, never false-admit),
        // pinned by the ignored `bevel_pipe_gate_condition3_records_one_face_per_slot_doubled_fin_fan`;
        // the defect is in the ring's forward-face record, not in condition 3.
        for corner in corners {
            let out_of_plane = self.face_plane_normal(corner.face).dot(pipe_axis).abs();
            if out_of_plane > PIPE_PERPENDICULAR_DOT_EPS {
                return false;
            }
        }
        true
    }
}

/// Shape half of the §8.5.5.5(f.6) replant's degeneracy floor: the smallest **sagitta**
/// (distance from the projected `mid` to the `start'–end'` chord) the replant still reads
/// as a cell, as a fraction of the cell's **longer** axis. Only for a near-isotropic cell is
/// that the same thing as a `sin θ` floor on the two axes; for an anisotropic one it fires at
/// a considerably larger `sin θ`, which is the wanted behaviour.
///
/// It has to be a ratio of lengths rather than a floor on the determinant, because by
/// Lagrange the determinant **is** `|a_axis × b_axis|²` — a length⁴ quantity. A fixed floor
/// on it is therefore a floor on the bevel *offset*: a right-angle cell of extent `L` reads
/// `L⁴`, so an `f32::EPSILON` floor diverts every vert of a bevel below `offset ≈ 0.019` to
/// the chord, silently flattening the curved cross-section into a chamfer with no diagnostic.
///
/// The dimensionless quotient `determinant / (aa · bb)` would be scale-free too, and is the
/// shape the rework was asked for; this ships the sagitta instead. That quotient is `sin²θ`,
/// a *pure angle*, and an angle is not what the curve-versus-chord decision turns on: an
/// anisotropic cell (`|a_axis| ≫ |b_axis|`) has a sagitta near `|b_axis| · sin θ`, so it can
/// hold a perfectly healthy angle while its curve departs from its chord by less than the
/// inputs resolve. The sagitta measures that departure directly, and so also covers the
/// near-parallel case, where the same anisotropy makes the angle quotient the less
/// informative of the two.
pub(super) const REPLANT_CELL_SHAPE_RATIO: f64 = 1.0e-4;

/// Resolution half of the same floor, as a fraction of the control points' **world**
/// magnitude — the part a pure ratio cannot supply. The three control points arrive as
/// single-precision world positions, so a `mid` that lies exactly on the chord (a
/// collapsed profile, which §8.5.5.5(f.6) names as the routine flat-chamfer case) still
/// stores a sagitta of a few ulps of its own world coordinate. Below that the cell's shape
/// is not resolvable from its inputs at all, and a ratio-only floor would read the storage
/// noise as a curve and build a sliver cell out of it.
pub(super) const REPLANT_INPUT_ULP_RATIO: f64 = 4.0 * f32::EPSILON as f64;

/// The §8.5.5.5(f.6) replant's `land_on_corner` override for one canonical net slot, per
/// (f.6.1)'s "what an implementation must do": for **even** `segments` it is set on the slots
/// with `k = ⌊n/2⌋` that either sit in one of the two pipe boundverts' sectors, or sit at
/// `(i = 0, j = ⌊n/2⌋)` — the welded apex. Odd `segments` never set it.
///
/// It is computed for every exponent and consumed only by the sentinel branch of
/// [`snap_to_unit_superellipsoid`], so computing it unconditionally and computing it only under
/// that branch agree.
pub(super) fn pipe_replant_lands_on_corner(
    slot: (usize, usize, usize),
    segments: usize,
    pipe_sectors: (usize, usize),
) -> bool {
    if segments % 2 != 0 {
        return false;
    }
    let (sector, ring, along) = slot;
    let half = segments / 2;
    if along != half {
        return false;
    }
    sector == pipe_sectors.0 || sector == pipe_sectors.1 || (sector == 0 && ring == half)
}

/// The §8.5.5.5(f.6) replant's shape at one pipe junction: the pipe axis, the pipe
/// boundvert's cross-section cell, and which of (f.6)'s two placements every station takes.
#[derive(Clone, Copy)]
pub(super) struct PipeCrossSection {
    pub(super) axis: Vec3,
    pub(super) cell: ProfileCell,
    /// Which of (f.6)'s placements every vert in the domain takes. One value serves the whole
    /// junction because the criterion is a property of the cell and the axis alone — the
    /// station projection is an affine map, so it cannot vary from station to station — and
    /// it is what both the (f.1) step-4 sentinel and the per-vert replant read. A
    /// [`StationPlacement::Cell`] here is the only placement that reaches (f.2)'s surface
    /// snap; a planar junction, whose profile plane contains the pipe axis, is
    /// [`StationPlacement::Chord`].
    pub(super) placement: StationPlacement,
    /// The **two pipe boundverts**' sector indices — the boundverts whose own edges are the
    /// collinear pair. §8.5.5.5(f.6.1)'s `land_on_corner` rule is stated over them, and a
    /// boundvert's own edge is the one *closing* its sector, so sector `i` owns edge `i + 1`
    /// and the two sectors are one ring step behind the two edge indices.
    pub(super) sectors: (usize, usize),
}

/// BEVEL.md §8.3.4.1 **condition 3**'s perpendicularity window: the largest `|dot|` between
/// the unit pipe axis and a fan face's unit normal that still reads as "this face is parallel
/// to the axis".
///
/// It is a **dot-product magnitude**, not an angle, and it is `1e-4` — about `0.0057°` off
/// perpendicular, the same tight window §8.8.2's in-plane leg classification uses. Condition
/// 2's `EPS_ANG = 2°` is a *different test on a different quantity*: that one asks whether two
/// spoke rays are collinear enough for the junction to *have* an axis at all, and is generous
/// by design so a hand-modelled near-straight tube still takes the pipe path. This one asks
/// whether the cross-section is genuinely constant along that axis, and is deliberately
/// conservative — a fan that is merely *visibly* flat, its normals accumulated through float
/// arithmetic, fails it and takes the ordinary cutoff path.
///
/// The two must therefore never share a constant. Reusing `2°` here is a `|dot| ≤ 0.035`
/// window, 349× wider, and fires the pipe override on corners that merely turn gently.
pub(super) const PIPE_PERPENDICULAR_DOT_EPS: f32 = 1.0e-4;

/// The §8.3.4.1 **pipe boundvert**'s two collinear spokes, as indices into the ring's
/// per-sector spoke directions (BEVEL.md §8.3.4.1 condition 2 / §8.5.5.5(f.6)). The first
/// returned index is the pipe boundvert's *own* edge; the second is its collinear opposite.
/// Their shared line is the pipe axis. The §7.6.2 near-collinear window is 2°, so the two
/// rays out of `v` meet at ≥ 178° (`cos ≤ −cos 2°`).
///
/// The scan is over **boundverts**, not over spoke pairs, because that is the unit
/// §8.3.4.1's condition is stated in: three *consecutive* boundverts whose edges' first and
/// third are collinear, naming the first of the three. A boundvert's own edge is the one its
/// own profile chain crosses — chain `B_i → B_(i+1)` (§8.2.2), which crosses the edge
/// *closing* sector `i` — so boundvert `i` tests edges `i + 1` and `i + 3`.
///
/// That association is what makes (f.6) coherent rather than a labelling preference: the
/// replant's shape is the pipe boundvert's **own** profile, and (f.6) puts that profile's
/// chord endpoints at the pipe edge's two offsets on either side of the axis. Only under
/// this reading does the named boundvert's own chain cross a *pipe* edge; under the other,
/// it crosses a transverse edge, whose two boundverts share a transverse coordinate and
/// differ only in station, so the projection that forgets the station collapses its
/// cross-section to a single point.
///
/// The distinction is invisible until a ring carries **two** collinear pairs — the flat
/// 4-way cross — where it selects which pair is the axis and which is spanned by the
/// cross-section.
///
/// **The scan is not over all `C(count, 2)` spoke pairs, and at `count == 4` that is
/// count-bearing.** Reading §8.3.4.1's condition as *three consecutive* ring members makes it
/// a ring-**distance-2** test: at `count == 3` every pair is at distance 2 (the three
/// rotations name the same three pairs, and any of them yields the same axis line), but at
/// `count == 4` the four ring-*adjacent* pairs are not tested at all. A valence-4 junction
/// whose only anti-parallel pair is adjacent in the ring — e.g. beveled spokes at
/// `0°, 30°, 60°, 180°` on a flat fan — therefore reports `None`, and the §8.3.4.1 override
/// declines a junction an any-pair scan would have re-tiled. That is a change in `V`, `E` and
/// `F`, not in positions ([`cutoff_pipe_override_declines_on_a_ring_adjacent_pair`]).
pub(super) fn pipe_boundvert_spokes(directions: &[Vec3]) -> Option<(usize, usize)> {
    // Condition 2's own window: an **angle** of `EPS_ANG = 2°` between two unit spoke
    // directions oriented through the vertex. Deliberately not shared with condition 3's
    // [`PIPE_PERPENDICULAR_DOT_EPS`] — see that constant.
    let near_collinear_cos = (2.0_f32.to_radians()).cos();
    let count = directions.len();
    (0..count)
        .map(|boundvert| ((boundvert + 1) % count, (boundvert + 3) % count))
        .find(|&(own, opposite)| {
            own != opposite && directions[own].dot(directions[opposite]) <= -near_collinear_cos
        })
}

/// Which of the §8.5.5.5(f.6) replant's three placements a pipe junction takes.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum StationPlacement {
    /// Step 4's first collapse: `start'` and `end'` coincide, so the cross-section is a
    /// single point and every vert replants onto it.
    Point,
    /// Step 4's second collapse: the projected control triple is collinear, so the
    /// cross-section is the closed chord `start'–end'` and every vert replants onto the
    /// nearest point *of that segment*. The ordinary case at a planar §8.3.4.1 junction.
    Chord,
    /// Step 3: the projected triple spans a cell, so each vert resolves through the (f.2)
    /// surface snap.
    Cell,
}

/// Which placement (f.6)'s replant takes at a pipe junction, for the whole junction at once
/// (BEVEL.md §8.5.5.5(f.6) steps 3–4).
///
/// The guide's criterion is **local and station-independent**: the projected triple is
/// collinear exactly when the profile plane through `start`, `mid`, `end` is parallel to the
/// pipe direction, since a plane containing the projection direction has a line for its image.
/// The station projection differs from station to station only by a translation, and a
/// translation moves no shape, so one verdict serves every vert in the domain — which is what
/// lets the (f.1) step-4 sentinel ask this question *before* the base grid exists and get the
/// answer the replant will act on. Deciding the two with separate predicates (say a
/// junction-wide coplanarity fit for the sentinel and this test for the replant) puts a seam
/// between them wherever their tolerances disagree, and a junction that falls in the seam
/// gets a grid emitted for a placement that is not the one it takes.
///
/// The two collapses are judged against one floor, a length in the cell's own frame that is
/// part shape ratio and part input resolution (see [`REPLANT_CELL_SHAPE_RATIO`] and
/// [`REPLANT_INPUT_ULP_RATIO`]). It deliberately does **not** read the vert being replanted:
/// a floor that moved with `base` would make the classification station-dependent again, in
/// the one place where the guide says it cannot be.
///
/// Planarity is read off the cell's own control points rather than off `projection_dir`,
/// because a rescued planar cell reports its *plane normal* there (§8.2.2.1) — the one
/// direction under which a flat cross-section would falsely survive projection.
///
/// Being junction-wide, this is resolved **once** per pipe junction and carried on
/// [`PipeCrossSection`]; the per-vert replant is handed the answer rather than re-deriving it
/// for every slot it walks.
pub(super) fn pipe_station_placement(pipe_axis: Vec3, cell: &ProfileCell) -> StationPlacement {
    let axis = pipe_axis.as_dvec3().normalize_or_zero();
    let flatten = |offset: DVec3| offset - offset.dot(axis) * axis;
    let a_axis = flatten(cell.mid.as_dvec3() - cell.start.as_dvec3());
    let b_axis = flatten(cell.mid.as_dvec3() - cell.end.as_dvec3());
    let span = flatten(cell.end.as_dvec3() - cell.start.as_dvec3());

    let world_magnitude = cell
        .start
        .abs()
        .max(cell.mid.abs())
        .max(cell.end.abs())
        .max_element() as f64;
    let resolution = REPLANT_CELL_SHAPE_RATIO * a_axis.length().max(b_axis.length())
        + REPLANT_INPUT_ULP_RATIO * world_magnitude;

    if span.length() <= resolution {
        return StationPlacement::Point;
    }
    // `|a_axis × b_axis|` is `|span|` times the sagitta, so the determinant test is the
    // sagitta test. Cauchy–Schwarz makes it non-negative in exact arithmetic, which is what
    // licenses testing it with no `.abs()`; it proves nothing about the divide the cell path
    // then performs — that is settled by the strict inequality here, which clears the floor
    // by eight orders before the solve is reached.
    let aa = a_axis.dot(a_axis);
    let ab = a_axis.dot(b_axis);
    let bb = b_axis.dot(b_axis);
    if aa * bb - ab * ab <= resolution * resolution * span.length_squared() {
        return StationPlacement::Chord;
    }
    StationPlacement::Cell
}

/// Re-plant one interior grid vert onto a pipe junction's constant cross-section
/// (BEVEL.md §8.5.5.5(f.6), the cross-section replant). `base` is the vert's base-grid
/// position; the result lies on the pipe boundvert's profile curve, evaluated in the
/// plane through `base` whose normal is `pipe_axis`.
///
/// The three profile control points `start` / `mid` / `end` are carried into that station
/// plane by orthogonal projection along the axis — the operation that "forgets which
/// station you are at", which is what makes the cross-section constant along the pipe.
/// `base` is then read into that planar cell's `(u, v)` frame, snapped onto the §8.6.1
/// superellipse `u^r + v^r = 1` there (the §8.5.5.5(f.2) surface snap with a zero third
/// octant coordinate, so only its 2-D behaviour is exercised), and carried back out.
/// `land_on_corner` is the slot's (f.6.1) override, live only under the snap's sentinel
/// branch; it is this routine's one per-vert input beyond `base`.
///
/// Two shapes admit no curve and collapse to a nearest-point rule, and both are routine
/// (a pipe through a flat-chamfer profile takes the second): a cross-section degenerated
/// to a single point (`start' == end'`), and one degenerated to the straight chord
/// `start'–end'` (collinear projected control points).
///
/// **Known gap — no custom awareness (§8.6.3).** The solve is keyed on `exponent` and re-solves
/// the cross-section in the analytic superellipse frame; it takes no custom chain and cannot.
/// That is newly reachable: (f.1) sends a custom profile to the general grid "or on the pipe
/// post-pass over it", and [`adj_square_ring_claims`]'s `not a custom profile` conjunct means a
/// custom bevel whose untouched slider happens to read `≥ 0.95` no longer has the square ring
/// claim the junction and suppress this replant. So `profile ≥ 0.95` + a pipe junction + even
/// `segments` under a hand-drawn curve now replants onto the superellipse the slider names
/// rather than onto the curve — a positional divergence, no topology in it. This is the **second**
/// of the sites where the custom curve is not yet threaded, alongside
/// [`LoopMesh::mint_pipe_override_loose_verts`]; it carries its own parity-matrix row and is
/// deliberately not fixed here.
///
/// Everything below is carried **relative to `base`**, and that framing is the whole
/// numeric content of the routine (§8.5.5.5(f.6.2)). The station projection, the two cell
/// axes and the coordinate solve are all differences of world positions that lie within an
/// `offset` of each other, so taking them off `base` first makes every intermediate
/// cell-scale and leaves no large-magnitude cancellation anywhere. Written off the world
/// origin instead, `base − (start' + end' − mid')` cancels world-magnitude terms down to a
/// cell-magnitude answer, and the residue it manufactures is not a rounding one can shrug
/// at: it lands in the cell coordinate that should be zero at a vert coincident with a
/// projected control point, and the snap's `r`-norm raises that coordinate to the power
/// `r`, which for a concave exponent is a violent expansion rather than a contraction
/// (`(10⁻⁷)^0.354 ≈ 3.5·10⁻³` — a percent-scale inward shrink of a vert the replant is
/// supposed to be the identity on). Centred, the identity
/// `base − (start' + end' − mid') ≡ (base − mid') + (mid' − start') + (mid' − end')` holds
/// term by term at cell scale, and the corner-coincident case comes out *exactly*: the
/// solve reads `(u, v) = (1, 0)` bit-for-bit and the reconstruction returns `base`
/// unchanged. That exactness is centring's doing alone and would survive a single-precision
/// solve.
///
/// Double precision on top of it buys **authorship, not residue**, exactly as (f.6.2) says.
/// Off a coordinate direction the residue is *input-borne*: a grid vert stored in `f32`
/// cannot lie exactly on a tilted axis line through a control point, so the vert arrives
/// already carrying `ε ≈ ulp(world)/offset ≈ 4·10⁻⁷` of cell coordinate — the amplification
/// band (f.6.2) budgets for, which no internal precision closes (measured: the drift is the
/// same to within one quantum with the widening removed). What the widening does buy is that
/// the answer we emit is the exact-arithmetic answer *for the inputs as stored*, which inside
/// that band differs from the single-precision answer by tens of quanta; being the
/// reproducible one of the two is worth having. Nothing below pins it — the tests pass with
/// the widening removed.
pub(super) fn pipe_cross_section_replant(
    base: Vec3,
    pipe_axis: Vec3,
    cell: &ProfileCell,
    placement: StationPlacement,
    exponent: f32,
    land_on_corner: bool,
) -> Vec3 {
    let axis = pipe_axis.as_dvec3().normalize_or_zero();
    let base_position = base.as_dvec3();
    let to_station = |point: Vec3| {
        let from_base = point.as_dvec3() - base_position;
        from_base - from_base.dot(axis) * axis
    };
    let start = to_station(cell.start);
    let end = to_station(cell.end);
    let mid = to_station(cell.mid);
    let to_world = |station_relative: DVec3| (base_position + station_relative).as_vec3();

    let a_axis = mid - start;
    let b_axis = mid - end;
    let span = end - start;

    // `placement` is [`pipe_station_placement`]'s junction-wide verdict, resolved once by the
    // caller — the same value the (f.1) step-4 sentinel read. Re-deciding it here from the
    // centred projections would be a second predicate on the same question.
    match placement {
        // Degenerate: the cross-section is a single point — every vert replants onto it.
        StationPlacement::Point => return to_world(start),
        // Degenerate: the projected control points are collinear (a flat-chamfer
        // cross-section, and the ordinary case at a planar junction), so the curve is the
        // chord `start'–end'` and the vert replants onto its nearest point. The segment is
        // closed, so this is a clamp and not a projection.
        StationPlacement::Chord => {
            let along = (-start).dot(span) / span.length_squared();
            return to_world(start + span * along.clamp(0.0, 1.0));
        }
        StationPlacement::Cell => {}
    }

    // `base` in the planar cell's `(u, v)` frame (§8.2.2.1): solve
    // `base − origin = u·a_axis + v·b_axis` by the 2×2 normal equations, with the origin's
    // `start' + end' − mid'` folded into the centred right-hand side. `base` and the two axes
    // are coplanar (the station plane), so the projection error is zero. The classifier is the
    // divide guard as well: a `Cell` verdict means the determinant cleared a floor that is
    // itself a product of squares, by eight orders, and these centred axes differ from the
    // ones it judged only by the rounding of a common translation.
    let aa = a_axis.dot(a_axis);
    let ab = a_axis.dot(b_axis);
    let bb = b_axis.dot(b_axis);
    let determinant = aa * bb - ab * ab;
    debug_assert!(
        determinant > 0.0,
        "a `Cell` placement's centred determinant collapsed",
    );
    let cell_offset = mid - start - end;
    let a_projection = a_axis.dot(cell_offset);
    let b_projection = b_axis.dot(cell_offset);
    let u = (a_projection * bb - b_projection * ab) / determinant;
    let v = (aa * b_projection - ab * a_projection) / determinant;

    // The third cell coordinate is identically zero by construction — the station plane is
    // *defined* to pass through `base` — so it is written, never computed (§8.5.5.5(f.6)):
    // a literal zero has no rounding for the `r`-norm to amplify.
    let snapped =
        snap_to_unit_superellipsoid(Vec3::new(u as f32, v as f32, 0.0), exponent, land_on_corner);
    to_world(mid + (snapped.x as f64 - 1.0) * a_axis + (snapped.y as f64 - 1.0) * b_axis)
}

/// The ring-chain rim samples the §8.5.5.5(f.6) replant reaches (BEVEL.md §8.5.5.5(f.6.1)).
///
/// The replant's domain is the net's index-canonical `j >= 1` slot block (§8.5.2), and the
/// net's seam aliasing `(i, j, 0) == (i - 1, 0, n - j)` makes the `k = 0` column of every
/// sub-patch *be* a ring chain rather than interior. Working both parities of the canonical
/// `j` range through that alias gives the same index window either way — the chain's strict
/// **second half** `⌊n/2⌋ + 1 ..= n − 1`, never its first half and never either endpoint
/// (those are the boundverts). Empty at `segments <= 2`.
pub(super) fn pipe_replant_rim_window(segments: usize) -> std::ops::Range<usize> {
    (segments / 2 + 1)..segments
}
