//! The [V-ADJ] tri-corner family and its dihedral gate (BEVEL.md §8.5.1 / §8.5.2).

use super::*;

impl LoopMesh {
    /// The §8.5.1 forced-tri-corner trigger: whether this 3-way junction is the
    /// symmetric equal-offset orthonormal cube corner at the default circular
    /// profile that the ADJ tri-corner fill (§8.5.1–§8.5.5) is distilled for.
    ///
    /// The two geometric conditions (§8.5.5.3) are read off the boundvert ring,
    /// which is pairing-robust at a triangle (every pair of boundverts shares one
    /// beveled edge). Writing each ring vert as the §7.6.2 meet-point
    /// `B_i = v + offset·(d_a + d_b)` over the two orthonormal inward edge
    /// directions it bridges:
    /// - **equal offset** ⇒ the three `w_i = B_i − v` have equal length
    ///   (`|w_i| = offset·√2`); an unequal-offset corner breaks this;
    /// - **right-angle convex octant** ⇒ every pair `w_i, w_j` meets at 60°
    ///   (`cos = ½`, the signature of mutually perpendicular `d_*`); a non-octant
    ///   corner moves the cosine off ½.
    ///
    /// The exponent condition is the tri-corner family's **smooth** member
    /// (§8.5.5.5(f.1) step 3, the `otherwise` arm): every `r` that is neither square nor
    /// square-in — the circular default (`r = 2`), the whole curved band, and the straight
    /// chord (`r = 1`, whose octant grid is the flat chamfer plane). All of them are what
    /// the unit-octant grid of §8.5.5.5(f.2) builds. Only the two square exponents, which
    /// take closed-form grids of their own (§8.5.5.5(f.4)/(f.5)), are refused here.
    /// The gate keys on **3 boundverts** and a smooth exponent, then asks the per-edge
    /// dihedral band captured on the cut (BEVEL.md §8.5.1 / §7.6.6). The band **decides
    /// wherever it can be measured**: it admits an **oblique / absorbed-leg / reflex** corner
    /// the position-only meet geometry would decline (a reflex sector is not itself a
    /// disqualifier), and it *refuses* — with no appeal to the ring's shape — a corner with a
    /// fold past `3·π/4`, the wrong near-flat count, or a signed sum outside the window,
    /// dropping it through to the general grid of §8.5.5.5. The meet-geometry test (equal
    /// spokes, 60° octant — the symmetric cube corner) stands in for the **one** case the band
    /// cannot read at all: an open fan or a wire, where no closed fold exists
    /// ([`DihedralBand::Unmeasurable`]).
    ///
    /// `custom` is the predicate's **refusal** arm (§8.5.5.5(f.1), the three-valued predicate):
    /// a custom profile is refused outright, before any offset or dihedral is looked at, because
    /// the superellipsoid the whole family snaps onto has no meaning for a hand-drawn curve. It
    /// is a *refusal*, not a "doesn't apply", so step 4's shortcut back into the family honours
    /// it too — which this gate serves by being the one predicate both routes ask. The other
    /// unconditional refusal, vertex mode, is structural here: no vertex-mode fill reaches this
    /// call at all.
    pub(super) fn adj_tri_corner_trigger(
        &self,
        cut: &JunctionCut,
        profile: f32,
        custom: bool,
    ) -> bool {
        if custom {
            return false;
        }
        let exponent = profile_to_exponent(profile);
        // §8.5.5.5(f.1): the tri-corner family admits three exponents — the smooth octant
        // grid (f.2), and the two closed forms. The **square** sentinel takes (f.5)'s three
        // flat unit squares, and is admitted here on the *tighter* total-angle band: three
        // flat planes visibly kink at an only-approximately-orthonormal corner. The
        // square-in sentinel (`r → 0`) stays out — it takes the degenerate weld path, not
        // filled on this route.
        if exponent == PROFILE_SQUARE_EXPONENT {
            return self.adj_tri_corner_shape(cut, TRI_CORNER_BAND_SQUARE);
        }
        if !adj_octant_exponent(exponent) {
            return false;
        }
        self.adj_tri_corner_shape(cut, TRI_CORNER_BAND_SMOOTH)
    }

    /// Whether the ring's *shape* is one the tri-corner family owns (BEVEL.md §8.5.1 /
    /// §7.6.6): three boundverts, and the per-edge dihedral band's verdict — measured against
    /// the caller's total-angle window, honoured as-is when it is a hard refusal, and replaced
    /// by the position-only meet geometry only where it could not be measured.
    ///
    /// The window ([`TRI_CORNER_BAND_SMOOTH`] / [`TRI_CORNER_BAND_SQUARE`]) is the only thing
    /// that varies across the trigger's arms, so the construction lives here once; the
    /// stand-in ([`tri_corner_meet_geometry`], described on
    /// [`Self::adj_tri_corner_trigger`]) is band-free.
    pub(super) fn adj_tri_corner_shape(&self, cut: &JunctionCut, band_tolerance: f32) -> bool {
        if cut.boundverts.len() != 3 {
            return false;
        }
        let vertex = self.vertex_position(cut.vertex);
        let meet_points = [
            self.vertex_position(cut.boundverts[0]),
            self.vertex_position(cut.boundverts[1]),
            self.vertex_position(cut.boundverts[2]),
        ];
        tri_corner_shape_admits(vertex, &meet_points, cut.dihedral_band, band_tolerance)
    }

    /// Build the [V-ADJ] fill for the symmetric equal-offset 3-way cube corner at
    /// `segments ≥ 3`, at any **smooth** exponent (BEVEL.md §8.5.5.5(f.2)). The grid is
    /// **not** built in world space and there is no circumsphere to fit — for a curved
    /// `r` no sphere exists at all. It is built entirely in the **unit octant** (the
    /// three boundverts at the axis points `e_0, e_1, e_2`, the corner at `(1,1,1)`),
    /// where the target surface is the unit superellipsoid `u_x^r + u_y^r + u_z^r = 1`,
    /// and mapped into the corner **once**, at the end, by the affine `unit_to_corner`
    /// map of §8.5.2. The circular sphere-fit is the special case of this, not the
    /// other way round.
    ///
    /// The four stages, all in octant coordinates (§8.5.5.5(f.2)):
    ///
    /// 1. **Coarse net** at `segments = 2` — corners `e_i`, sector mids
    ///    `2^(−1/r)·(e_i + e_{i+1})`, centre `√⅓·(1,1,1)·PRESCALE`. The pre-scale
    ///    deliberately seeds the centre *off* the surface so the subdivision's smoothing
    ///    does not pull the cap in too far; it is invisible in the apex (the snap is a
    ///    radial projection, and radial projection is scale-invariant) and visible one
    ///    ring in.
    /// 2. **Subdivide** — the §8.5.5.5(c) stencil, doubling until the working resolution
    ///    reaches the target.
    /// 3. **Resample** — the §8.5.5.5(d) kernel, when the doubled resolution overshoots.
    /// 4. **Snap** — *every* grid vert, interior and emitted rim alike, onto the unit
    ///    superellipsoid ([`snap_to_unit_superellipsoid`]).
    ///
    /// The centre resolves by parity (§8.5.5.4): even `segments` welds the three
    /// sub-patches at one apex vertex — the closed form `3^(−1/r)·(1,1,1)` in octant
    /// coordinates, for every `r` — and odd `segments` leaves them separated by a ring
    /// of seam quads closed by a central `s`-gon, with no apex vertex on the body
    /// diagonal.
    ///
    /// The rim is *reused* from [P]/[S] (the deterministic §8.2.2.1.1 read-back the
    /// arriving strips carry), never re-minted — stage 4 only projects it back out onto
    /// the surface. Interior positions are an admissible implementation choice (Tier-2,
    /// `bevel_vmesh.md`); only their superellipsoid membership and the grid counts /
    /// parity are pinned (§8.5.5.6).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn build_adj_fill_octant(
        &mut self,
        cut: &JunctionCut,
        chains: &[EdgeEndChain],
        segments: u32,
        profile: f32,
        material: i32,
        templates: &TemplateBank,
        slots: &mut AdjNetSlots,
        result: &mut BevelResult,
    ) -> bool {
        let sectors = cut.boundverts.len(); // 3 (guarded by the caller)
        let seg = segments as usize;
        let half = seg / 2;
        let even = seg % 2 == 0;
        let exponent = profile_to_exponent(profile);

        // Each sector's profile chain, oriented from its opening boundvert `B_i`
        // (`seg + 1` samples; the two ends are the ring boundverts).
        let mut chain_of: Vec<Vec<VertexKey>> = Vec::with_capacity(sectors);
        for i in 0..sectors {
            match self.oriented_chain(cut, chains, i) {
                Some(chain) if chain.len() == seg + 1 => chain_of.push(chain),
                // A sector that did not sample to `seg + 1` verts (a collinear /
                // degenerate chain) is outside the forced cube-corner regime; skip
                // the whole fill rather than emit a torn patch.
                _ => return false,
            }
        }

        let octant = match self.unit_octant(cut) {
            Some(octant) => octant,
            None => return false,
        };

        // Stage 4 on the rim. The snap is the fill's own **grid-level** step and runs
        // **last** — after the §8.2.2.1.1 boundary resample, never before it — and on this
        // (tri-corner) path it covers the emitted rim as well as the interior, projecting
        // each chord-resampled chain sample back out onto the superellipsoid; the sample
        // survives at a *shifted arc angle* rather than at its raw even-angle one. It is
        // invoked whether or not a chord resample ran: at `segments ∈ {4, 8, …}` the raw
        // sample is already on the surface and the projection is the identity on the rim
        // (identity, not skipped — the same snap is what places the interior, below). The
        // two chain ends are ring boundverts and are already on it, so only the interior
        // samples are touched.
        for chain in &chain_of {
            for &sample in chain.iter().take(seg).skip(1) {
                let unit = octant.to_octant(self.vertex_position(sample));
                let snapped = octant.to_world(snap_to_unit_superellipsoid(unit, exponent, false));
                self.set_vertex_position(sample, snapped);
            }
        }

        // Stage 1: the coarse net, in octant coordinates. Each sector is the quarter
        // superellipse from `e_i` to `e_{i+1}` bulging toward the unit-square corner
        // `e_i + e_{i+1}` — the same profile cell the world-space chain rides, expressed
        // in the canonical frame, so the sector mid is the closed-form
        // `2^(−1/r)·(e_i + e_{i+1})` the [S] sampler already produces there.
        let unit_boundverts = [Vec3::X, Vec3::Y, Vec3::Z];
        let profiles: Vec<RingChainProfile> = (0..sectors)
            .map(|i| {
                let start = unit_boundverts[i];
                let end = unit_boundverts[(i + 1) % sectors];
                RingChainProfile::Cell {
                    cell: ProfileCell::from_control_points(start, end, start + end),
                    profile,
                    custom: None,
                }
            })
            .collect();
        let mids: Vec<Vec3> = profiles.iter().map(|p| p.raw_positions(2)[1]).collect();
        let centre = Vec3::splat((1.0_f32 / 3.0).sqrt() * adj_octant_prescale(exponent, segments));

        // Stages 2/3: the same subdivision-relaxation stencil and read-back the general
        // ring runs (§8.5.5.5(c)/(d)) — here fed octant chains, so its output is in octant
        // coordinates and stage 4 can snap it against the unit surface.
        let net = adj_relaxed_net(&unit_boundverts, &mids, centre, &profiles, segments);
        let axis = adj_interior_axis(&profiles, segments, 2 * net.half as u32);

        let sector_representatives = cut.resolve_sector_representatives(templates);
        let centre_column_representatives = cut.resolve_centre_column_representatives(templates);
        let central_representative = templates.resolve(cut.central_representative);
        self.emit_adj_grid(
            sectors,
            seg,
            half,
            even,
            &chain_of,
            &AdjInterior::Octant {
                net: &net,
                axis: &axis,
                sectors,
                segments: seg,
                exponent,
                octant,
            },
            material,
            &sector_representatives,
            &centre_column_representatives,
            central_representative,
            // §5.2.4.1 / §5.2.4.2 are vertex-mode-only rules; the edge-mode octant fill's
            // apex and interior are the isomorphism-invariant closed forms of §8.5.5.5(f.2).
            None,
            None,
            slots,
            result,
        );
        true
    }

    /// The affine `unit_to_corner` map of this tri-corner junction (BEVEL.md §8.5.2 /
    /// §8.5.5.5(f.2)): the frame in which the whole [V-ADJ] grid is built. `None` when the
    /// three boundverts and the corner are degenerate enough that the map is singular.
    pub(super) fn unit_octant(&self, cut: &JunctionCut) -> Option<UnitOctant> {
        if cut.boundverts.len() != 3 {
            return None;
        }
        UnitOctant::from_corner(
            [
                self.vertex_position(cut.boundverts[0]),
                self.vertex_position(cut.boundverts[1]),
                self.vertex_position(cut.boundverts[2]),
            ],
            self.vertex_position(cut.vertex),
        )
    }

    /// Build the **tri-corner** square [V-ADJ] fill — the three flat unit squares of
    /// BEVEL.md §8.5.5.5(f.5) (valence 3, square profile). In octant coordinates it is closed
    /// form, with **no subdivision and no snap**: sector `i`'s grid is the face of the unit
    /// cube where `coord i = 1`, restricted to the sub-square `(coord (i+1) mod 3, coord (i+2)
    /// mod 3) = (2k/n, 2j/n)`, then mapped to the corner by the affine `unit_to_corner` frame.
    ///
    /// It is reached only on **odd** `segments` (even+`s≥3` is claimed by the parallel-offset
    /// ring, (f.1) step 1), so the three squares stop at `(n−1)/n < 1` and leave the
    /// central-triangle hole the odd-`segments` centre `s`-gon then closes.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn build_adj_fill_square_tricorner(
        &mut self,
        cut: &JunctionCut,
        chains: &[EdgeEndChain],
        segments: u32,
        profile: f32,
        material: i32,
        templates: &TemplateBank,
        slots: &mut AdjNetSlots,
        result: &mut BevelResult,
    ) -> bool {
        let _ = profile;
        let sectors = cut.boundverts.len(); // 3 (guarded by the caller)
        let seg = segments as usize;
        let half = seg / 2;
        let even = seg % 2 == 0;
        if sectors != 3 || seg < 3 {
            return false;
        }

        let mut chain_of: Vec<Vec<VertexKey>> = Vec::with_capacity(sectors);
        for i in 0..sectors {
            match self.oriented_chain(cut, chains, i) {
                Some(chain) if chain.len() == seg + 1 => chain_of.push(chain),
                _ => return false,
            }
        }

        let octant = match self.unit_octant(cut) {
            Some(octant) => octant,
            None => return false,
        };

        let sector_representatives = cut.resolve_sector_representatives(templates);
        let centre_column_representatives = cut.resolve_centre_column_representatives(templates);
        let central_representative = templates.resolve(cut.central_representative);
        self.emit_adj_grid(
            sectors,
            seg,
            half,
            even,
            &chain_of,
            &AdjInterior::SquareTriCorner {
                octant,
                segments: seg,
            },
            material,
            &sector_representatives,
            &centre_column_representatives,
            central_representative,
            None,
            None,
            slots,
            result,
        );
        true
    }

    /// The §8.5.1 dihedral-band gate on the **original** corner (BEVEL.md §8.5.1 / §7.6.6),
    /// evaluated over every edge incident to `vertex` (beveled and unbeveled alike, so an
    /// absorbed leg's own fold is one of the measured edges). All of:
    ///
    /// - the corner is a **closed** fan — every incident edge is shared by two of its faces
    ///   (an open fan or a wire leaves a dihedral undefined and declines);
    /// - **no** edge is too-folded: `|signed dihedral| < 3·π/4`;
    /// - **exactly `edgecount − 3`** edges are near-flat (`|signed dihedral| ≤ π/4`);
    /// - the signed dihedrals — signed *relative to each other* in §7.6.2's disk-walk sense —
    ///   sum to within `π/4` of `3·π/2` in magnitude; see
    ///   [`Self::junction_dihedral_band`] for why the sum's own overall sign is not part
    ///   of the corner.
    ///
    /// Equal beveled offsets — the gate's other clause — hold structurally here: a junction
    /// bevels all its edges at the one op-wide `offset`.
    pub(super) fn junction_dihedral_band_admits(
        &self,
        vertex: VertexKey,
        corners: &[VertexCorner],
    ) -> bool {
        matches!(
            self.junction_dihedral_band(vertex, corners),
            DihedralBand::Deviation(deviation) if deviation <= std::f32::consts::FRAC_PI_4,
        )
    }

    /// The §8.5.1 / §7.6.6 dihedral band's verdict at one junction: its **total-angle
    /// deviation** — how far the signed dihedral sum sits from the `3·π/2` an orthonormal
    /// corner carries — or [`DihedralBand::Refused`] when one of the band's own hard clauses
    /// fails, or [`DihedralBand::Unmeasurable`] when there is no closed fan to fold across.
    /// The three are not interchangeable: a refusal is final, while an unmeasurable band hands
    /// the question to the position-only stand-in ([`tri_corner_shape_admits`]).
    ///
    /// The tolerance on the deviation is the caller's: `π/4` for every smooth exponent
    /// ([`Self::junction_dihedral_band_admits`]), tightened to `π/16` at a square exponent
    /// (BEVEL.md §8.5.5.5(f.1), the three-flat-squares grid is far less forgiving of an
    /// only-approximately-orthonormal corner). Captured on the cut so the fill-time routing
    /// can apply whichever tolerance the profile selects.
    pub(super) fn junction_dihedral_band(
        &self,
        vertex: VertexKey,
        corners: &[VertexCorner],
    ) -> DihedralBand {
        if corners.len() < 3 {
            return DihedralBand::Unmeasurable;
        }
        let vertex_position = self.vertex_position(vertex);

        let edges = junction_incident_edges(corners);
        let edgecount = edges.len();
        if edgecount < 3 {
            return DihedralBand::Unmeasurable;
        }

        let quarter = std::f32::consts::FRAC_PI_4;
        let three_quarter = 3.0 * quarter;
        let target = 3.0 * std::f32::consts::FRAC_PI_2;
        let mut in_plane = 0usize;
        let mut signed_sum = 0.0_f32;

        for &edge in &edges {
            let corner_out = corners.iter().find(|c| c.outgoing_edge == edge);
            let corner_in = corners.iter().find(|c| c.incoming_edge == edge);
            let (corner_a, corner_b) = match (corner_out, corner_in) {
                (Some(a), Some(b)) => (a, b),
                // A single-face (boundary / wire) edge has no dihedral: the band is
                // undefined here, not failed.
                _ => return DihedralBand::Unmeasurable,
            };
            if corner_a.face == corner_b.face {
                return DihedralBand::Unmeasurable;
            }
            let edge_direction = match self.edge_direction_from(edge, vertex, vertex_position) {
                Some(direction) => direction,
                None => return DihedralBand::Unmeasurable,
            };
            let normal_a = match self.corner_walk_normal(corner_a, vertex, vertex_position) {
                Some(normal) => normal,
                None => return DihedralBand::Unmeasurable,
            };
            let normal_b = match self.corner_walk_normal(corner_b, vertex, vertex_position) {
                Some(normal) => normal,
                None => return DihedralBand::Unmeasurable,
            };
            let magnitude = normal_a.dot(normal_b).clamp(-1.0, 1.0).acos();
            let orientation = normal_a.cross(normal_b).dot(edge_direction);
            let signed = magnitude * if orientation >= 0.0 { 1.0 } else { -1.0 };

            // §8.5.1: one hard crease at the vertex disqualifies the whole corner.
            if magnitude >= three_quarter {
                return DihedralBand::Refused;
            }
            if magnitude <= quarter {
                in_plane += 1;
            }
            signed_sum += signed;
        }

        if in_plane != edgecount - 3 {
            return DihedralBand::Refused;
        }
        // The sum's *overall* sign is not a property of the corner: inverting every face at
        // the junction reverses the disk walk and negates every term, and the two shells
        // carry identical local data — a solid's convex corner and the same corner seen from
        // inside are the same measurement. So the target is matched against the sum's
        // magnitude, and the per-edge chirality needs no convex-positive re-basing: which of
        // the two global signs a given walk lands on is unobservable by construction.
        //
        // What the fold does **not** discard is each edge's sign *relative to its
        // neighbours*, which is the whole content of the test: a fold that creases the other
        // way (a concave leg among convex ones reads `|Σ| = π/2`, a whole `π` off target), or
        // a §7.8 mis-wound face flipping the two edges it flanks, still pulls the sum off
        // `3·π/2` and refuses the corner. A sum of *magnitudes* would pass both.
        DihedralBand::Deviation((signed_sum.abs() - target).abs())
    }
}

/// The §8.5.1 dihedral band's verdict at one junction — **three-valued**, because the band
/// distinguishes "this corner is disqualified" from "this corner has no band to measure"
/// (BEVEL.md §8.5.1).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) enum DihedralBand {
    /// A **hard refusal**: an edge folded past `3·π/4` ("one hard crease at the vertex
    /// disqualifies the whole corner"), or a near-flat count other than `edgecount − 3`.
    /// Neither depends on a tolerance the caller picks, and neither is rescuable by the
    /// ring's shape — a refusal is the one answer §8.5.5.5(f.1) step 4's shortcut honours
    /// too, so it must not fall through to the position-only stand-in.
    Refused,
    /// The band is **not measurable** here: some edge at the junction carries no two faces
    /// to fold between (an open fan, a wire, a face meeting itself across the edge), or a
    /// flanking direction is degenerate. §8.5.1 states the band over a closed fan and leaves
    /// this case unstated, so the position-only meet geometry stands in.
    Unmeasurable,
    /// The band's **total-angle deviation** from `3·π/2` — the quantity the caller's window
    /// is applied to.
    Deviation(f32),
}

/// The exponent-free half of the §8.5.1 tri-corner test, on raw inputs: three boundverts
/// inside the per-edge dihedral band, whose total-angle window is the caller's to choose
/// (BEVEL.md §8.5.1 / §7.6.6).
///
/// **The band decides wherever it can be measured**, and the position-only meet geometry is
/// only its stand-in where it cannot. §8.5.1 states the gate as a conjunction — a corner that
/// fails the band is *refused*, and a refusal is the one answer §8.5.5.5(f.1) step 4's
/// shortcut honours too — so an octant-shaped ring cannot buy its way past a band the corner
/// fails. That holds for **both** ways of failing it: the caller's total-angle window, and the
/// band's own hard clauses ([`DihedralBand::Refused`] — a fold past `3·π/4`, the wrong
/// near-flat count). Only [`DihedralBand::Unmeasurable`] reaches the geometry, which is why
/// the verdict is three-valued rather than an `Option`.
///
/// That is what keeps a §7.8 mis-wound junction off the tri-corner family: its ring is
/// a perfect octant by position and says nothing about the incident faces' sense, while the
/// band's *signed* sum sees the dissenting face and pulls the total a whole `π` off `3·π/2`.
///
/// Taken on positions rather than on a [`JunctionCut`] because the §8.5.5.5(f.4a) weld has
/// to ask the same question during chain layout, one phase before any cut exists.
///
/// Note this answers the *shape* question only. It cannot distinguish (f.1)'s *fires* from
/// its *doesn't apply*: both readings pass here, and the two are separated by the junction's
/// **edge** count ([`junction_incident_edges`]), which callers that need the distinction test
/// alongside.
pub(super) fn tri_corner_shape_admits(
    vertex_position: Vec3,
    meet_points: &[Vec3],
    dihedral_band: DihedralBand,
    band_tolerance: f32,
) -> bool {
    meet_points.len() == 3
        && match dihedral_band {
            DihedralBand::Refused => false,
            DihedralBand::Deviation(deviation) => deviation <= band_tolerance,
            DihedralBand::Unmeasurable => tri_corner_meet_geometry(vertex_position, meet_points),
        }
}

/// The §8.5.1 forced-tri-corner geometry test on three meet-point positions
/// (BEVEL.md §8.5.5.3): equal spoke length and a mutually-60° convex octant
/// (`cos = ½`). This is the single position-based predicate, and it is the §8.5.1 cap's
/// alone: `adj_tri_corner_trigger` feeds it the boundvert-ring positions, and
/// [`tri_corner_shape_admits`] feeds it the same ring when the dihedral band is
/// unmeasurable. The inner miter never consults it — §8.9.3 / §8.9.3.2(a) put no
/// junction-level geometry gate on that miter at all.
pub(super) fn tri_corner_meet_geometry(vertex_position: Vec3, meet_points: &[Vec3]) -> bool {
    if meet_points.len() != 3 {
        return false;
    }
    let spokes = [
        meet_points[0] - vertex_position,
        meet_points[1] - vertex_position,
        meet_points[2] - vertex_position,
    ];
    let lengths = [spokes[0].length(), spokes[1].length(), spokes[2].length()];
    let longest = lengths.iter().copied().fold(0.0_f32, f32::max);
    if longest <= 1.0e-6 {
        return false;
    }
    let length_tol = longest * 1.0e-3;
    if (lengths[0] - lengths[1]).abs() > length_tol
        || (lengths[1] - lengths[2]).abs() > length_tol
        || (lengths[2] - lengths[0]).abs() > length_tol
    {
        return false;
    }
    let directions = [
        spokes[0] / lengths[0],
        spokes[1] / lengths[1],
        spokes[2] / lengths[2],
    ];
    let cos_tol = 1.0e-3;
    (directions[0].dot(directions[1]) - 0.5).abs() <= cos_tol
        && (directions[1].dot(directions[2]) - 0.5).abs() <= cos_tol
        && (directions[2].dot(directions[0]) - 0.5).abs() <= cos_tol
}

/// How far the corner's signed dihedral sum may sit from `3·π/2` and still take the
/// §8.5.1 tri-corner family at a **smooth** exponent (BEVEL.md §8.5.5.5(f.1)).
pub(super) const TRI_CORNER_BAND_SMOOTH: f32 = std::f32::consts::FRAC_PI_4;

/// The tighter window the **square** sentinel's three flat unit squares demand
/// (BEVEL.md §8.5.5.5(f.1)/(f.5)): three flat planes visibly kink at an
/// only-approximately-orthonormal corner, where a curved cap would not.
pub(super) const TRI_CORNER_BAND_SQUARE: f32 = std::f32::consts::PI / 16.0;

/// The affine `unit_to_corner` frame of a tri-corner junction (BEVEL.md §8.5.2,
/// restated for the grid in §8.5.5.5(f.2)): the map sending the three unit axis points to
/// the three boundverts, the unit origin to the corner sphere's **centre** `c`, and hence
/// the unit body-corner `(1,1,1)` to the junction vertex `v`. For the orthonormal
/// equal-offset cube corner it is a similarity — a permutation of the world axes scaled by
/// `offset` — so "octant coordinates" and "world coordinates measured from `c` in units of
/// `offset`" coincide up to an axis relabelling.
///
/// The `(1,1,1) → v` row is not an extra condition, it follows from the other four; keeping
/// it in view is what catches the natural mis-statement `origin → v`, which would put the
/// unit surface's centre *at* the corner and place every grid vert at twice its true
/// distance from `v`.
#[derive(Clone, Copy)]
pub(super) struct UnitOctant {
    pub(super) linear: Mat3,
    pub(super) inverse: Mat3,
    pub(super) translation: Vec3,
}

/// The image of the unit octant's origin under §8.5.2's `unit_to_corner` — the frame's
/// translation `(B_0 + B_1 + B_2 − v)/2`, i.e. the corner surface's centre `c`.
///
/// Split out from [`UnitOctant::from_corner`] because it is the one part of the frame that
/// needs **no** conditioning: it is a plain affine combination of the four points, where the
/// linear columns have to be invertible before anything may be read back through them. A
/// caller that only maps *forward* — §8.5.5.5(f.4a)'s legs, `c + (1 − k·b)·(B_i − c)` — must
/// not be refused by a guard that exists for the inverse.
pub(super) fn unit_octant_origin(boundverts: [Vec3; 3], vertex: Vec3) -> Vec3 {
    (boundverts[0] + boundverts[1] + boundverts[2] - vertex) * 0.5
}

impl UnitOctant {
    pub(super) fn from_corner(boundverts: [Vec3; 3], vertex: Vec3) -> Option<UnitOctant> {
        let [b0, b1, b2] = boundverts;
        let axes = [
            (b0 - b1 - b2 + vertex) * 0.5,
            (-b0 + b1 - b2 + vertex) * 0.5,
            (-b0 - b1 + b2 + vertex) * 0.5,
        ];

        // Degeneracy here is a question about the frame's *shape*, never its size: each axis
        // is one boundvert's spoke, so the raw determinant scales as the cube of the bevel
        // width (at the symmetric cube corner it is exactly ±offset³). Any absolute threshold
        // on it would therefore reject ordinary narrow bevels — a 1e-3 offset is a perfectly
        // conditioned corner. Guard the axis lengths against an absolute floor and against
        // each other, then test conditioning on the *normalised* frame, whose determinant is
        // the scale-free sine of the solid angle the three axes span. Same relative shape as
        // the tri-corner spoke test.
        let lengths = axes.map(|axis: Vec3| axis.length());
        let longest = lengths.iter().copied().fold(0.0_f32, f32::max);
        if longest <= 1.0e-6 || lengths.iter().any(|&length| length <= longest * 1.0e-3) {
            return None;
        }
        let conditioning = Mat3::from_cols(
            axes[0] / lengths[0],
            axes[1] / lengths[1],
            axes[2] / lengths[2],
        )
        .determinant()
        .abs();
        if conditioning <= 1.0e-4 {
            return None;
        }

        let linear = Mat3::from_cols(axes[0], axes[1], axes[2]);
        Some(UnitOctant {
            linear,
            inverse: linear.inverse(),
            translation: unit_octant_origin(boundverts, vertex),
        })
    }

    pub(super) fn to_world(&self, unit: Vec3) -> Vec3 {
        self.translation + self.linear * unit
    }

    pub(super) fn to_octant(&self, world: Vec3) -> Vec3 {
        self.inverse * (world - self.translation)
    }
}

/// Whether `exponent` selects the tri-corner family's **smooth** member — the unit-octant
/// grid of BEVEL.md §8.5.5.5(f.2). Its scope is stated exactly: *every* `r` that is neither
/// the square sentinel nor square-in, those two taking closed-form grids of their own
/// ((f.5)/(f.4)). So the straight chord (`r = 1`, the flat chamfer) **is** a member — its
/// snap is branch R's 1-norm radial projection onto the plane `x + y + z = 1`, i.e. the
/// chamfer plane through the three boundverts, and its apex is (f.2)'s `3^(−1/1)·(1,1,1)`,
/// the boundvert centroid. A flat cube corner builds the octant grid like any other smooth
/// exponent; only a *non*-tri-corner ring takes the planar `fullness = 0` cap
/// (§8.3.0.2.1).
///
/// The comparisons are exact float equalities against the snapped exponents of §8.6.2 —
/// which is what that snap exists for (§8.5.5.5(f.1a)): a raw logarithm would route every
/// junction down the curved path.
pub(super) fn adj_octant_exponent(exponent: f32) -> bool {
    exponent != PROFILE_SQUARE_EXPONENT && exponent != 0.0
}

/// The coarse centre's pre-scale (BEVEL.md §8.5.5.5(f.2), step 1). The centre starts at the
/// octant body-diagonal unit vector `√⅓·(1,1,1)` and is multiplied by a scalar that depends
/// only on the exponent *band* — deliberately pushing it **off** the surface so the
/// subdivision's smoothing does not pull the whole cap in too far. The band thresholds are
/// `1.5` and `0.75`, **not** `2` and `1`: the circular exponent sits comfortably inside the
/// bulging band, so the circular cube corner pre-scales by `1.4` too.
///
/// The `segments ≤ 2` guard is not cosmetic: at two segments the coarse net *is* the target
/// grid on this path (no subdivision round runs at all), so the centre is snapped straight
/// from its seed and the scale would be a pure no-op.
///
/// Its effect on the apex is nil by construction — the snap is a radial projection, and
/// radial projection is scale-invariant — so it is *invisible in the apex* and *visible in
/// the ring of interior verts around it*. Do not "verify" it by checking the apex.
pub(super) fn adj_octant_prescale(exponent: f32, segments: u32) -> f32 {
    if segments <= 2 {
        return 1.0;
    }
    if exponent > 1.5 {
        1.4
    } else if exponent < 0.75 {
        0.6
    } else {
        1.0
    }
}

/// Project an octant point onto the unit superellipsoid `u_x^r + u_y^r + u_z^r = 1` along
/// its own ray from the octant origin (BEVEL.md §8.5.5.5(f.2), step 4). The rule genuinely
/// **branches** on the exponent, and the branches are not each other's special cases:
///
/// - **the circular exponent** is a plain, **unclamped** normalise. Subdivision can push an
///   interior vert slightly negative in one octant coordinate (the smoothing overshoots
///   past a sector seam); the unclamped normalise keeps that sign, leaving the vert a hair
///   outside its sector rather than exactly on the seam. Clamping first would move those
///   verts — and the circular corner's positions are pinned;
/// - **the two square sentinels** take an **axis snap in the clamped unit square**, not a norm
///   at all: at `r = 1e4` every `a^r` overflows or flushes in `f32`, so the curved closed form
///   is numerically dead there, and at `r = 0` the `r`-norm diverges and would collapse the
///   whole cross-section to a point. The snap pushes the clamped point onto the nearer of the
///   two walls the sentinel names — `x == 1` / `y == 1` for `square` (the profile bulges out
///   to the cell's `(1, 1)` corner, the junction vertex) and `x == 0` / `y == 0` for
///   `square-in` (it bows away, onto the cell edges through `start + end − mid`). This branch
///   is strictly **2-D**: it is not a limit of the curved one, and its only caller is the
///   §8.5.5.5(f.6) pipe replant, which snaps inside a station plane and hands over a written
///   zero third coordinate;
/// - **every other (curved) exponent** must clamp into the octant first, because a negative
///   base raised to a fractional power is `NaN`. The clamp is a *necessity of the general
///   exponent*, not a shared step, and it folds an overshooting vert onto the seam plane.
///
/// `land_on_corner` is §8.5.5.5(f.6)'s "land on the square's *corner*, not on the nearer wall"
/// override. It exists for the sentinel branch alone — the circular and curved branches ignore
/// it — and without it the pipe's square cross-section collapses into a wedge. Which verts set
/// it is the replant's business ([`pipe_replant_lands_on_corner`]).
///
/// **Which grids reach which branch.** The three 3-D grid paths (the octant cap and the two
/// pinwheel closed forms) never hold a square exponent — §8.5.5.5(f.1) hands those to (f.5)'s
/// closed forms before the grid is built — so they only ever exercise the circular and curved
/// branches and pass `land_on_corner = false`. The *square* grids of (f.5) do reach this
/// routine, but only through the (f.6) replant that post-passes them, and only at the one
/// count each can carry: step 3's three flat unit squares are odd-`segments` only, and step 1's
/// parallel-offset ring outranks the pipe test outright ([`adj_square_ring_claims`]), so no
/// replant follows it at all.
///
/// A vert already on the surface is a fixed point in every branch, so the snap is safe to
/// apply unconditionally to the boundverts and to any rim sample that happens to be exact.
/// The curved branch divides through by the leading coordinate before taking the powers:
/// the two forms agree in exact arithmetic, but only this one keeps the intermediate
/// magnitudes near `1` at a large `r`. It has no ray to project along when the clamp
/// annihilates every coordinate, which is the one degenerate case the branch guards; past
/// that guard the leading coordinate of `scaled` is exactly `1`, so the `r`-norm is `≥ 1`
/// and the division is safe.
pub(super) fn snap_to_unit_superellipsoid(unit: Vec3, exponent: f32, land_on_corner: bool) -> Vec3 {
    if exponent == 2.0 {
        return unit.normalize_or_zero();
    }
    if exponent == PROFILE_SQUARE_EXPONENT || exponent == 0.0 {
        return snap_to_unit_square_walls(unit, exponent == 0.0, land_on_corner);
    }
    let clamped = Vec3::new(unit.x.max(0.0), unit.y.max(0.0), unit.z.max(0.0));
    let leading = clamped.x.max(clamped.y).max(clamped.z);
    if leading <= 0.0 {
        return clamped;
    }
    let scaled = clamped / leading;
    let norm = (scaled.x.powf(exponent) + scaled.y.powf(exponent) + scaled.z.powf(exponent))
        .powf(1.0 / exponent);
    scaled / norm
}

/// The sentinel arm of [`snap_to_unit_superellipsoid`] (BEVEL.md §8.5.5.5(f.2), branch S): an
/// **axis snap inside the clamped unit square**, in the plane the §8.5.5.5(f.6) replant works
/// in. `inward` selects the square-in sentinel's walls (`u == 0` / `v == 0`, the cell edges
/// through the `(0, 0)` corner `start + end − mid`) over the square sentinel's (`u == 1` /
/// `v == 1`, the two edges meeting at the cell's `(1, 1)` corner, which is the junction vertex).
///
/// Clamp first, then push onto the **nearer** wall — or onto both, landing exactly on the
/// square's corner, when the caller's slot carries §8.5.5.5(f.6)'s `land_on_corner`. Snapping
/// to one wall there would fold the pipe's square cross-section into a wedge.
///
/// Strictly 2-D: the third coordinate is a written zero at the one caller, and there is no
/// third wall for it to be pushed onto. At an exact tie the `u` wall wins; the guide's "the
/// nearer of the walls" leaves the tie open, both answers lie on the target square, and the
/// choice has to be *some* fixed one for the fill to be deterministic.
pub(super) fn snap_to_unit_square_walls(unit: Vec3, inward: bool, land_on_corner: bool) -> Vec3 {
    debug_assert!(
        unit.z == 0.0,
        "the sentinel snap is 2-D; its third coordinate is written, never computed",
    );
    let mut u = unit.x.clamp(0.0, 1.0);
    let mut v = unit.y.clamp(0.0, 1.0);
    let (wall, nearer_is_u) = if inward {
        (0.0, u <= v)
    } else {
        (1.0, 1.0 - u <= 1.0 - v)
    };
    if land_on_corner || nearer_is_u {
        u = wall;
    }
    if land_on_corner || !nearer_is_u {
        v = wall;
    }
    Vec3::new(u, v, 0.0)
}
