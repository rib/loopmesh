//! Vertex-mode bevel (BEVEL.md §3–§6, §8.11): frame capture, ring placement, the
//! incident-face re-stitch and the cap.

use super::*;

/// The incident edge a boundvert sits on — the inverse of the per-leg `ring_vertex_of`
/// map. Used to resolve a boundvert's own representative face (BEVEL.md §6.6.1's
/// per-sector grid rule, where each sector fill inherits from the next boundvert).
pub(super) fn edge_of_ring_vertex(
    ring_vert_for_edge: &[(EdgeKey, VertexKey)],
    ring_vertex: VertexKey,
) -> Option<EdgeKey> {
    ring_vert_for_edge
        .iter()
        .find(|&&(_, ring)| ring == ring_vertex)
        .map(|&(edge, _)| edge)
}

/// A re-stitched original: the *new* face key the rebuild produced, and its own
/// pre-bevel self in the donor bank. Every corner of a re-stitched original blends
/// from that face, pre-bevel (BEVEL.md §5.2.1.1), and the re-stitched originals are
/// the **last** faces to write (§5.2.1 fact (c)). `None` only when there is nothing
/// to blend.
pub(super) struct RebuiltOriginal {
    pub(super) face: FaceKey,
    pub(super) donor: Option<usize>,
}

/// One beveled vertex's local frame, captured against the *pristine* mesh
/// before any surgery (BEVEL.md §6.4.1: place every ring vert before cutting any
/// face, so a shared edge's original direction and length are still readable
/// when the second corner is placed). Holds the disk-ordered incident edges, the
/// ring vert minted on each (paired with that edge), the incident faces, and the
/// interior/boundary fan classification.
pub(super) struct BeveledCorner {
    pub(super) vertex: VertexKey,
    pub(super) ring_vert_for_edge: Vec<(EdgeKey, VertexKey)>,
    pub(super) incident_faces: Vec<FaceKey>,
    /// This corner's ring verts in **disk order** (BEVEL.md §3.2 / §5.1) — the cycle the
    /// cap walks. One entry per incident edge, wires included (§7.2.1's vertex-mode box).
    pub(super) ring_order: Vec<VertexKey>,
    /// Each of this corner's **wire** spokes (face-less, radial-0 incident edges) paired
    /// with the incident face that adopts its ring vert. Vertex mode keeps a wire in the
    /// disk as an ordinary ring member (BEVEL.md §7.2.1's vertex-mode box — the documented
    /// exception to the face-bearing filter edge mode applies): it gets its own ring vert,
    /// slid along itself by the offset.
    ///
    /// Only a **closed** fan hosts its wires on a face: the face chain wraps, so the slot
    /// where it runs out lies inside a faced sector, and that face picks the ring vert up as
    /// an extra corner (growing by *two* sides). On an open or multi-chain fan the slot is a
    /// faceless **gap**, so no rebuilt face adopts the wire and the list is empty — the cap
    /// alone catches the wire's ring vert (§7.2.1's vertex-mode box, open-fan half).
    pub(super) hosted_wires: Vec<(FaceKey, VertexKey)>,
    /// The representative incident face the cap's stage-1 inherit copies its whole
    /// attribute block from (BEVEL.md §6.6.1: the cap inherits an incident face of `v`,
    /// whatever `material` is). Captured against the pristine mesh in `place_ring_verts`
    /// before those faces are killed and rebuilt, so it survives the surgery.
    pub(super) representative: FaceTemplate,
    /// The `FaceKey` behind [`representative`](Self::representative) — the §6.6.1.4-picked
    /// override donor at a seam-carrying corner, else the positional pick. Kept alongside the
    /// template handle because §5.2.1.1's seam-carrying cap branch and §5.2.6's post-pass both
    /// need the donor face's *identity* (to index the [`DonorBank`] and to compare
    /// representative donors), which the opaque template handle does not expose. `None` only
    /// when the corner has no adjacent original at all (a face-less wire star).
    pub(super) representative_face: Option<FaceKey>,
}

/// Per-incident-edge placement record gathered in the slide pre-pass: the edge,
/// its far endpoint position, and the *unclamped* resolved slide distance and
/// per-offset slide rate `k` (BEVEL.md §4.3 / §4.4.2).
pub(super) struct EdgePlacement {
    pub(super) edge: EdgeKey,
    pub(super) far_vertex: VertexKey,
    pub(super) far_position: Vec3,
    pub(super) slide: f32,
    pub(super) rate: f32,
}

/// One beveled vertex's placement frame, captured against the pristine mesh
/// (BEVEL.md §6.4.1): its position and the per-incident-edge placements.
pub(super) struct BevelFrame {
    pub(super) vertex: VertexKey,
    pub(super) vertex_position: Vec3,
    pub(super) placements: Vec<EdgePlacement>,
    /// The full incident-edge set in the **disk order** of BEVEL.md §3.2 — wires included
    /// (§7.2.1's vertex-mode box). The ring, and hence the cap, is this cycle.
    pub(super) disk_order: Vec<EdgeKey>,
    /// Each wire spoke of this corner paired with the incident face that adopts its ring
    /// vert (BEVEL.md §7.2.1's vertex-mode box), in the order that face's rebuilt corner run
    /// wants them. Empty unless the face fan is **closed** — an open or multi-chain fan slots
    /// its wires into a gap, where no face can adopt them.
    pub(super) wire_hosts: Vec<(EdgeKey, FaceKey)>,
}

impl LoopMesh {
    /// The single-segment vertex bevel (BEVEL.md §3–§6) behind [`bevel`](Self::bevel)'s
    /// `segments = 1`, `affect = Vertices` dispatch.
    pub(super) fn bevel_vertices_single_segment(
        &mut self,
        geom: &[GeomKey],
        options: &BevelOptions,
    ) -> BevelResult {
        let mut result = BevelResult::default();

        // The set of beveled verts, de-duplicated and order-preserved. A vert is
        // beveled at most once even if it appears twice in `geom`.
        let mut beveled: Vec<VertexKey> = Vec::new();
        for entry in geom {
            if let GeomKey::Vert(vertex) = *entry {
                if !beveled.contains(&vertex) {
                    beveled.push(vertex);
                }
            }
        }

        // §6.4.1 — place-before-surgery. Two phases against the pristine mesh:
        //
        //   1. Capture each beveled vertex's local frame (disk-ordered incident
        //      edges, far positions, per-edge slide distances) while the mesh is
        //      still pristine, so an edge shared by two beveled verts is measured
        //      from its *original* direction/length at both ends. The slide
        //      pre-pass also feeds the §4.4 global `clamp_overlap` reduction.
        //   2. Mint every ring vert (§4), then run all face surgery and caps.
        //
        // Doing surgery per-vertex inline would re-point a shared edge after the
        // first corner and slide the second corner off the recreated chamfer
        // (§6.4.1), so the two phases must stay separated.
        let frames = self.gather_bevel_frames(
            &beveled,
            options.offset,
            options.offset_type,
            options.use_weights,
            options.vert_weights,
        );
        // `gather_bevel_frames` is the eligibility gate (§2.1, plus the all-wire-star
        // decline): a vertex it dropped is not beveled, and every pass below — the
        // two-ended collision clamp, the re-stitch's corner lookup, the loose-vertex free —
        // has to see the *accepted* set, or it operates on a corner that has no ring.
        let beveled: Vec<VertexKey> = frames.iter().map(|frame| frame.vertex).collect();
        let clamp_factor = if options.clamp_overlap {
            clamp_factor(options.offset, &beveled, &frames)
        } else {
            1.0
        };

        // §6.6.1: one bank for the whole run — every cap's donor block captured in a
        // single snapshot, and read back at every value of `material`.
        let mut templates = TemplateBank::new(self, options.segments);
        let corners = self.place_ring_verts(
            &frames,
            clamp_factor,
            options.segments,
            &mut templates,
            &mut result,
        );
        templates.seal(self);

        // §5.2.1 fact (a): the per-corner attribute blend's donors are the incident
        // faces **as they stood before the bevel** — captured here, while the ring
        // verts exist (so the mints are done) but no face has been re-stitched and no
        // corner edge re-pointed. Reading a donor back after the re-stitch finds a
        // polygon whose corners *are* the ring verts, and every blend collapses onto
        // the beveled corner's own block.
        let incident: Vec<FaceKey> = corners
            .iter()
            .flat_map(|corner| corner.incident_faces.iter().copied())
            .collect();
        let bank = self.capture_donor_bank(&incident);

        // §5.2.1.1: the cap's donor list is **per corner** — corner `i` takes
        // boundvert `i`'s own representative, the face whose winding arrives at `v`
        // along `e_i`, so the sample point lies on that donor's boundary edge `e_i`
        // rather than in the interior of a face it does not touch. Collapsing the list
        // to one face (the §6.6.1 per-face representative the cap's *block* inherit
        // takes) would break that premise. At `segments = 1` the cap's write is not the
        // one that decides the value — every ring vert is also a corner of a re-stitched
        // original, and that pass writes last (fact (c)) — but the two agree only
        // because the list is per corner, and at §5.2.4 the cap's own write stands.
        // Resolved pre-surgery, like the donors themselves.
        //
        // This is §5.2.1.1's **non-seam-carrying** branch. A seam-carrying corner (any
        // incident leg an interpolation seam by §6.6.1.4 — which includes every boundary leg)
        // replaces this whole list with one override donor face plus a per-corner weight-snap
        // edge (§5.2.1.1.1), planned in `cap_overrides` below; the override branch wins at
        // those corners and the positional list here is skipped.
        let cap_donors: Vec<Vec<Option<usize>>> = if bank.is_empty() {
            Vec::new()
        } else {
            corners
                .iter()
                .map(|corner| {
                    corner
                        .ring_vert_for_edge
                        .iter()
                        .map(|&(edge, _)| {
                            self.arriving_face(corner.vertex, edge)
                                .and_then(|face| bank.index_of(face))
                        })
                        .collect()
                })
                .collect()
        };

        // §5.2.1.1's **seam-carrying** branch: at a corner with any seamed incident leg the
        // whole per-boundvert list collapses to the single §6.6.1.4 override donor (the same
        // face the block inherit takes, `corner.representative_face`), and each ring vert gets
        // the §5.2.1.1.1 weight-snap. Planned pre-surgery, while `F`'s edges still meet the
        // beveled vertex.
        let cap_overrides: Vec<Option<OverrideCapPlan>> = if bank.is_empty() {
            Vec::new()
        } else {
            corners
                .iter()
                .map(|corner| {
                    if !self.corner_is_seam_carrying(corner.vertex) {
                        return None;
                    }
                    let override_face = corner.representative_face?;
                    let donor = bank.index_of(override_face)?;
                    Some(OverrideCapPlan {
                        donor,
                        samples: self.plan_override_cap(corner, override_face, bank.donor(donor)),
                    })
                })
                .collect()
        };

        // §5.2.6's pre-pass: per beveled vertex, per float2 layer, which UV island each
        // incident original face's corner sat in — recorded now, before the surgery frees the
        // beveled vertex and the post-pass reads it back.
        let float2_layers = self.float2_loop_layers();
        let prepass: Vec<HashMap<ElementAttributeKey, HashMap<FaceKey, usize>>> = corners
            .iter()
            .map(|corner| {
                self.build_prepass_buckets(corner.vertex, &corner.incident_faces, &float2_layers)
            })
            .collect();

        // §6.4.3: re-stitch each incident face exactly once, swapping *every*
        // beveled corner of that face at the same time. A face straddling a
        // shared edge has two beveled corners and must not be killed twice.
        //
        // The re-point this runs (§6.1) is also what discharges §8.3.0.4 for vertex mode:
        // a wire spoke is a full ring member here (§7.2.1), so it carries its own ring
        // vert and is re-pointed onto it like every other incident edge — and that ring
        // vert lies *on* the segment from the corner to the wire's far tip, so it is the
        // nearest boundvert by construction. Edge mode needs the explicit post-pass only
        // because there the wire has no ring vert of its own.
        let rebuilt = self.restitch_incident_faces(&beveled, &corners, &bank);

        // §5 step 4 / §6.2 / §6.4.4: cap every corner of valence >= 3 — interior, boundary,
        // multi-gap and face-less alike — over its own ring only, the caps being independent
        // across a shared edge. The cap *adopts* the chamfer edges the re-stitch minted and
        // *creates* the ones no face could mint (the gap steps of §2.3's table); that
        // adopt-or-create is the whole rule, not a boundary patch-up.
        let mut cap_faces: Vec<Option<FaceKey>> = Vec::with_capacity(corners.len());
        for corner in &corners {
            if corner.ring_order.len() >= 3 {
                let cap = self.build_cap_face(&corner.ring_order);
                // §6.6: the cap is a new face — inherit the representative incident
                // face's whole block, then stamp `material` over the slot if >= 0.
                self.assign_bevel_face_attributes(
                    cap.face,
                    options.material,
                    templates.resolve(corner.representative),
                    BevelNewFaceKind::Cap,
                );
                result.faces.push(cap.face);
                result.edges.extend(cap.ring_edges);
                cap_faces.push(Some(cap.face));
            } else {
                cap_faces.push(None);
                // Valence 2 (§2.3's one special case): a two-corner "n-gon" is not a face, so
                // no cap — but the single ring edge is still minted *here*, by the operator
                // itself. A face-less two-wire hub (§6.2.1) has no re-stitch to mint it.
                let ring_edge = self.create_edge(
                    [corner.ring_order[0], corner.ring_order[1]],
                    EdgeCreateFlags::NO_DUPLICATE,
                );
                // §6.3: an edge is reported iff it touches a minted bevel face — and there is
                // no cap here. The lone carve-out is the ring edge of a **face-less** corner:
                // it bounds nothing at all, so no rebuild walk ever gets to withhold it.
                if corner.incident_faces.is_empty() {
                    result.edges.push(ring_edge);
                }
            }
        }

        // §5.2.1 fact (c): the per-corner blend runs once per corner of every face
        // the operator builds *or rebuilds*, and the **last write wins** — so the
        // order below is part of the value rule, not housekeeping. The fills go
        // first, the re-stitched originals last.
        //
        // The cap's write is the *only* writer of the cap's own loops — a ring vert is
        // a corner of a re-stitched original too, so its per-vertex value is decided
        // there, but no other face owns the cap's corners. A cap corner left unblended
        // reads the layer default (§5.2.5's per-loop half).
        //
        // At a seam-carrying corner every cap corner takes the single override donor with the
        // §5.2.1.1.1 weight-snap sample (§5.2.1.1's seamed branch); the gap-leg corner is not
        // special-cased — it takes the override too (§5.2.6). At a seam-free corner each takes
        // its own boundvert's positional donor, un-snapped.
        for corner_index in 0..corners.len() {
            let Some(cap) = cap_faces[corner_index] else {
                continue;
            };
            if bank.is_empty() {
                continue;
            }
            let corner = &corners[corner_index];
            let override_plan = cap_overrides
                .get(corner_index)
                .and_then(|plan| plan.as_ref());
            for cap_corner in self.face_loop_keys(cap) {
                let ring_vertex = self.loop_vertex(cap_corner);
                if let Some(plan) = override_plan {
                    if let Some(&(_, sample)) =
                        plan.samples.iter().find(|(ring, _)| *ring == ring_vertex)
                    {
                        self.blend_corner_from_donor_at(cap_corner, bank.donor(plan.donor), sample);
                    }
                } else {
                    let donor = donor_for_ring_vertex(
                        &corner.ring_vert_for_edge,
                        &cap_donors[corner_index],
                        ring_vertex,
                    );
                    if let Some(donor) = donor {
                        self.blend_corner_from_donor(cap_corner, bank.donor(donor));
                    }
                }
            }
        }
        self.blend_restitched_originals(&rebuilt, &bank);

        // §5.2.6's UV-merge post-pass: reconcile the per-loop float2 values that meet at each
        // new ring / cap vertex, bucket-then-average, once at the very end.
        self.reconcile_cap_uv_islands(
            &corners,
            &bank,
            &rebuilt,
            &cap_faces,
            &float2_layers,
            &prepass,
        );

        // §5.1.1 / §6.3: every beveled vertex is now loose. Free each through
        // the same guarded path the edge bevel uses, so a re-point/re-stitch
        // that left a corner attached to live geometry panics rather than
        // silently killing it.
        //
        // The victim set is exactly the corners that carry a bevel record — the
        // vertices `gather_bevel_frames` accepted. A vertex the §2.1 eligibility test
        // *rejected* (isolated, or a single incident edge) never reaches `corners`, so
        // it is never freed: it survives the run untouched, as it must. An
        // implementation that recovers this set from the caller's tag instead deletes
        // the geometry it declined to bevel (§6.3).
        for corner in &corners {
            self.free_loose_vertex(corner.vertex);
        }

        result
    }

    /// Vertex-mode multi-segment cap (BEVEL.md §8.11, Stage [VV]).
    ///
    /// Handles the **general-valence** (`s ≥ 3`), `segments ≥ 2` corner (§8.11.4) at the
    /// default-circular profile **and** the square-in profile (`r = 0`, §8.5.5.5(f.4b)):
    /// the §4 slide ring (a, §8.11.1), the `mid = v` / `projection_dir = 0` profile
    /// (b, §8.11.2), and the [V-ADJ] general fullness grid entered unconditionally
    /// (c, §8.11.3 — vertex mode is uniform across valence and never takes the cube-corner
    /// octant fast path, so a valence-3/4/5/… corner all route through the same §8.5.5.5
    /// grid). The `segments = 2` coarse-net case takes the identical path — a 1-vert
    /// profile chain per sector and an even-parity welded apex (§8.5.5.5(e)) — differing
    /// from the higher counts only in that the general fill's single mandatory subdivision
    /// round (§8.5.5.5(d)) resamples straight back down to the coarse resolution.
    ///
    /// Square-in (§8.5.5.5(f.4b)) is *not* a special path here: it is the same grid, with
    /// the boundary ring sampled as the concave square-in profile (§8.6.4) and the centre
    /// seeded by the **negative** square-in fullness (§8.5.5.5(f.3) law F3 at `profile = 0`),
    /// which pushes the coarse centre inward past the boundvert centroid. Both knobs fall
    /// out of passing `options.profile` straight through — the fullness is `adj_fullness`
    /// and the ring is the profile sampler. There is no surface snap (the general grid never
    /// snaps).
    ///
    /// Returns `None` when the corner is outside that regime (a profile that is neither
    /// circular nor square-in, a valence below 3, or a wire spoke present); the caller then
    /// declines to the historical no-op. Single-segment vertex bevel and every edge-bevel
    /// path are untouched — this method is reached only for `affect == VERTICES` at
    /// `segments ≥ 2`.
    pub(super) fn bevel_vertices_multiseg(
        &mut self,
        geom: &[GeomKey],
        options: &BevelOptions,
    ) -> Option<BevelResult> {
        let segments = options.segments;

        // §8.11.4 bounds: `segments ≥ 2` and any SUPERELLIPSE exponent. Vertex mode routes
        // every exponent to the general fullness grid (§8.5.5.5(f.1) dispatch): steps 1 and
        // 3 — the parallel-offset square grid and the tri-corner family — are edge-bevel
        // only, and step 4's tri-corner shortcut is refused for `affect == VERTICES`
        // (§8.5.1), so the circle, the curved band, both square sentinels (square `r = ∞`
        // and square-in `r = 0`), and the **line / flat chamfer** (`r = 1`, `profile = 0.25`)
        // all fall to the same (a)–(e) grid. The exponent only shapes two knobs, both riding
        // on `options.profile`: the boundary-ring sampler (§8.6, `superellipse_chain`) and
        // the centre fullness (§8.5.5.5(f.3), `adj_fullness`). The line profile is not a
        // distinct construction — per §8.5.5.5(f.3) law **F2** (`r == 1.0` ⇒ `fullness = 0`)
        // it is the *same* general fill with a planar cap, threaded through `emit_general_adj_fill`'s
        // `flat` path below. A `profile_type = CUSTOM` corner takes neither knob from the
        // exponent — its ring comes from §8.6.3's curve and its fullness from law **F1** — and
        // is never `flat`, but it still routes through this same (a)–(e) grid.
        // The `segments = 2` coarse-net case routes through the *same* grid — §8.11.3(c):
        // vertex mode never takes the tri-corner octant fast path, so a corner of any
        // valence at `segments = 2` builds the fullness-seeded net, subdivides once
        // (§8.5.5.5(d) blockquote: always at least one round, even at seg=2), and resamples
        // the even-parity welded apex back down (§8.5.5.5(e)).
        let exponent = profile_to_exponent(options.profile);
        if segments < 2 {
            return None;
        }
        // §8.6.3: a custom profile supplies the boundary chain and its fullness (law F1),
        // and is never flat — it routes through the general fullness grid unchanged.
        // §8.6.3.1(4).4: both of its chains are materialised **here, once** for the whole
        // operation, so every later read is a stride of one of them rather than a fresh run
        // of the evaluation model at that read's resolution.
        let custom_chains = options
            .profile_type
            .custom()
            .map(|curve| CustomChains::new(curve, segments));
        let custom = custom_chains.as_ref();

        let mut result = BevelResult::default();

        let mut beveled: Vec<VertexKey> = Vec::new();
        for entry in geom {
            if let GeomKey::Vert(vertex) = *entry {
                if !beveled.contains(&vertex) {
                    beveled.push(vertex);
                }
            }
        }
        if beveled.is_empty() {
            return None;
        }

        // §6.4.1 place-before-surgery pre-pass, identical to the single-segment path:
        // the ring is segment-independent (§8.11.1), so the frame gather and slide are
        // reused verbatim.
        let frames = self.gather_bevel_frames(
            &beveled,
            options.offset,
            options.offset_type,
            options.use_weights,
            options.vert_weights,
        );
        if frames.is_empty() {
            return None;
        }

        // §8.11 scope gate: general valence (`s ≥ 3`, §8.11.4 — the cap is uniform
        // across valence, and the `[V-ADJ]` grid of §8.5.5.5 the only fill this path
        // builds needs at least three sectors to close), and no wire spokes (the grid's
        // rim must be a clean faced/gap ring). A valence-2 corner (§2.3) has no cap and
        // is out of scope for the multiseg path; any corner outside the regime declines
        // the whole call.
        for frame in &frames {
            if frame.disk_order.len() < 3 {
                return None;
            }
            for placement in &frame.placements {
                if self.edge_radial_iter(placement.edge).next().is_none() {
                    return None;
                }
            }
        }

        let beveled: Vec<VertexKey> = frames.iter().map(|frame| frame.vertex).collect();
        let clamp_factor = if options.clamp_overlap {
            clamp_factor(options.offset, &beveled, &frames)
        } else {
            1.0
        };

        let mut templates = TemplateBank::new(self, segments);
        let mut corners =
            self.place_ring_verts(&frames, clamp_factor, segments, &mut templates, &mut result);

        // §6.6.1's grid rule (affect == VERTICES): each sector's [V-ADJ] fill faces inherit
        // from the **next** boundvert's representative, distributing the donor across the
        // corner's incident faces rather than collapsing every sector onto one corner-wide
        // donor. The centre n-gon is the ring's one ring-spanning fill and still takes
        // `corner.representative` (the §6.6.1.4 tie-broken donor). Registered here, before the
        // bank is sealed and the incident faces are killed / rebuilt.
        let per_corner_sector_reps: Vec<Vec<FaceTemplate>> = corners
            .iter()
            .map(|corner| {
                let sectors = corner.ring_order.len();
                (0..sectors)
                    .map(|i| {
                        let next_boundvert = corner.ring_order[(i + 1) % sectors];
                        let donor = edge_of_ring_vertex(&corner.ring_vert_for_edge, next_boundvert)
                            .and_then(|edge| self.boundvert_representative(corner.vertex, edge));
                        templates.register(donor)
                    })
                    .collect()
            })
            .collect();

        templates.seal(self);

        let incident: Vec<FaceKey> = corners
            .iter()
            .flat_map(|corner| corner.incident_faces.iter().copied())
            .collect();
        let bank = self.capture_donor_bank(&incident);

        // §5.2.4.1: the even-`segments` welded apex snaps to the **last-visited**
        // sector's boundvert edge — the predecessor of the ring's start, i.e. the last
        // boundvert of the disk-ordered ring (`emit_adj_grid` walks the sectors in that
        // order, so the last sector writes the apex last). `far` is that edge's far
        // endpoint. Captured now, against the pristine mesh: `v` and `far`'s per-vertex
        // blocks are still untouched (the re-stitch below re-points edges but does not
        // blend, and `blend_restitched_originals` runs after the fill).
        let apex_blends: Vec<Option<ApexBlend>> = corners
            .iter()
            .map(|corner| {
                let snap_boundvert = *corner.ring_order.last()?;
                let snap_edge = edge_of_ring_vertex(&corner.ring_vert_for_edge, snap_boundvert)?;
                let [a, b] = self.edge_vert_keys(snap_edge);
                let far = if a == corner.vertex { b } else { a };
                let endpoints = self
                    .vertex_attribs()
                    .snapshot_slots(&[corner.vertex.as_usize(), far.as_usize()]);
                Some(ApexBlend {
                    corner_vertex: corner.vertex,
                    snap_from: self.vertex_position(corner.vertex),
                    snap_to: self.vertex_position(far),
                    endpoints,
                })
            })
            .collect();

        // §5.2.4.2: at even `segments ≥ 4` the [V-ADJ] fill mints non-apex chain-interior
        // verts too. Each is minted from `v`; a sector-diagonal corner takes the
        // edge-linear snap onto that sector's *own* boundvert edge `e_i` (the same closed
        // form as the apex, one per sector), and an off-diagonal corner takes the
        // mean-value blend of the **next** boundvert's rep face `f2`. Captured now against
        // the pristine mesh, in the same disk-ordered sector enumeration `emit_adj_grid`
        // walks; the donor indices resolve against the same `bank` the re-stitch reads.
        let per_corner_sector_snaps: Vec<Vec<Option<ApexBlend>>> = corners
            .iter()
            .map(|corner| {
                (0..corner.ring_order.len())
                    .map(|i| {
                        let snap_edge =
                            edge_of_ring_vertex(&corner.ring_vert_for_edge, corner.ring_order[i])?;
                        let [a, b] = self.edge_vert_keys(snap_edge);
                        let far = if a == corner.vertex { b } else { a };
                        let endpoints = self
                            .vertex_attribs()
                            .snapshot_slots(&[corner.vertex.as_usize(), far.as_usize()]);
                        Some(ApexBlend {
                            corner_vertex: corner.vertex,
                            snap_from: self.vertex_position(corner.vertex),
                            snap_to: self.vertex_position(far),
                            endpoints,
                        })
                    })
                    .collect()
            })
            .collect();
        let per_corner_sector_donor_idx: Vec<Vec<Option<usize>>> = corners
            .iter()
            .map(|corner| {
                let sectors = corner.ring_order.len();
                (0..sectors)
                    .map(|i| {
                        let next_boundvert = corner.ring_order[(i + 1) % sectors];
                        edge_of_ring_vertex(&corner.ring_vert_for_edge, next_boundvert)
                            .and_then(|edge| self.boundvert_representative(corner.vertex, edge))
                            .and_then(|face| bank.index_of(face))
                    })
                    .collect()
            })
            .collect();

        // §5.2.4.3 (no-seam branch): at odd `segments` each centre n-gon corner `(i, h, h)`
        // blends from boundvert `i`'s **own** representative face (the `prev_face` of its own
        // edge `e_i` by fan direction), with no snap — *not* the next-boundvert rep `f2` the
        // grid's off-diagonal verts take. The whole face chooses this no-seam list only when
        // `v` is **not seam-carrying**; a seam-carrying corner (any incident leg an
        // interpolation seam — the §6.6.1.4 override-donor + snap-edge branch) is deferred, so
        // all its slots are `None` and its centre corners keep the bare `cd(v)` baseline. Every
        // in-scope leg here is face-bearing, so the boundary-leg arm of "seam-carrying" never
        // fires; the seam is purely an interpolation discontinuity.
        let per_corner_centre_donor_idx: Vec<Vec<Option<usize>>> = corners
            .iter()
            .map(|corner| {
                let sectors = corner.ring_order.len();
                let seam_carrying = corner
                    .ring_vert_for_edge
                    .iter()
                    .any(|&(edge, _)| self.edge_is_interp_seam(edge));
                (0..sectors)
                    .map(|i| {
                        if seam_carrying {
                            return None;
                        }
                        let own_boundvert = corner.ring_order[i];
                        edge_of_ring_vertex(&corner.ring_vert_for_edge, own_boundvert)
                            .and_then(|edge| self.boundvert_representative(corner.vertex, edge))
                            .and_then(|face| bank.index_of(face))
                    })
                    .collect()
            })
            .collect();

        // Stage 2 (§8.11.2 / §8.11.4): one profile chain per disk sector, built with
        // `mid = v` and `projection_dir = 0`. `ProfileCell::from_control_points` is
        // exactly that cell — the frame is the §8.2.2.1 one with `mid` promoted to the
        // corner vertex, and it carries **no** projection (`projection_dir = 0`) while
        // staying non-linear (the bulge control corner is off the chord), so the
        // interior samples bulge toward `v` rather than collapsing to the straight
        // chord. The `segments − 1` interior verts are minted at the §8.2.2.1.1
        // emitted-rim positions, shared between the cap and the incident faces.
        let mut per_corner_chains: Vec<Vec<Vec<VertexKey>>> = Vec::with_capacity(corners.len());
        let mut per_corner_cells: Vec<Vec<ProfileCell>> = Vec::with_capacity(corners.len());
        let mut per_corner_hosted: Vec<Vec<(FaceKey, VertexKey)>> =
            Vec::with_capacity(corners.len());

        for corner in &corners {
            let vertex_position = self.vertex_position(corner.vertex);
            let ring = corner.ring_order.clone();
            let sectors = ring.len();

            let mut chains: Vec<Vec<VertexKey>> = Vec::with_capacity(sectors);
            let mut cells: Vec<ProfileCell> = Vec::with_capacity(sectors);
            for i in 0..sectors {
                let start = ring[i];
                let end = ring[(i + 1) % sectors];
                let cell = ProfileCell::from_control_points(
                    self.vertex_position(start),
                    self.vertex_position(end),
                    vertex_position,
                );
                let interior = self.mint_chain_interior(
                    &cell,
                    segments,
                    options.profile,
                    // §8.6.3.1(6) Regime B: a cap sector has no beveled edge to walk, so the
                    // whole cap shares one handedness bit.
                    custom.map(CustomChainRead::vertex_cap),
                    ChainEmission::Resampled,
                    &mut result,
                );
                let mut chain = Vec::with_capacity(segments as usize + 1);
                chain.push(start);
                chain.extend(interior);
                chain.push(end);
                chains.push(chain);
                cells.push(cell);
            }

            // §8.11.4 stage 2: the chain is the boundary the cap and the incident face
            // **share**. Thread each sector's chain interior through the incident face
            // that spans it, via the same per-face run-insertion channel a hosted wire
            // uses — so `restitch_incident_faces` is reused unchanged (it is
            // segment-independent) and the cap later *adopts* the edges it minted.
            let mut hosted: Vec<(FaceKey, VertexKey)> = corner.hosted_wires.clone();
            for &face in &corner.incident_faces {
                let Some((incoming, outgoing)) = self.face_corner_edges(face, corner.vertex) else {
                    continue;
                };
                let leading = self.ring_vertex_of(&corner.ring_vert_for_edge, incoming);
                let trailing = self.ring_vertex_of(&corner.ring_vert_for_edge, outgoing);
                for i in 0..sectors {
                    let start = ring[i];
                    let end = ring[(i + 1) % sectors];
                    let interior = &chains[i][1..chains[i].len() - 1];
                    if start == leading && end == trailing {
                        for &wire in interior {
                            hosted.push((face, wire));
                        }
                        break;
                    }
                    if start == trailing && end == leading {
                        for &wire in interior.iter().rev() {
                            hosted.push((face, wire));
                        }
                        break;
                    }
                }
            }

            per_corner_chains.push(chains);
            per_corner_cells.push(cells);
            per_corner_hosted.push(hosted);
        }

        for (corner, hosted) in corners.iter_mut().zip(per_corner_hosted) {
            corner.hosted_wires = hosted;
        }

        // §8.11.1 / §6.4.3: the re-stitch is segment-independent — reused verbatim. It
        // re-points each incident edge onto its ring vert and rebuilds each incident
        // face, now threading the sector's chain interior between the two boundverts.
        let rebuilt = self.restitch_incident_faces(&beveled, &corners, &bank);

        // Stages 3–5 (§8.11.3 / §8.11.4): the shared [V-ADJ] general fullness grid,
        // entered unconditionally (no cube-corner octant fast path in vertex mode).
        // The cap adopts the incident-face chain edges and mints the gap-sector rim
        // plus the relaxed interior.
        for (corner_index, (((corner, (chains, cells)), sector_reps), apex_blend)) in corners
            .iter()
            .zip(per_corner_chains.iter().zip(per_corner_cells.iter()))
            .zip(per_corner_sector_reps.iter())
            .zip(apex_blends.iter())
            .enumerate()
        {
            let profiles: Vec<RingChainProfile> = cells
                .iter()
                .map(|&cell| RingChainProfile::Cell {
                    cell,
                    profile: options.profile,
                    custom: custom.map(CustomChainRead::vertex_cap),
                })
                .collect();
            // Centre n-gon: the ring-spanning §6.6.1.4 donor. Sector fills: the per-sector
            // next-boundvert reps registered above (§6.6.1's grid rule).
            let central_representative = templates.resolve(corner.representative);
            let sector_representatives: Vec<FaceTemplateRef<'_>> = sector_reps
                .iter()
                .map(|&template| templates.resolve(template))
                .collect();
            // §5.2.4.2: resolve this corner's per-sector `f2` donor indices to the pristine
            // donors in `bank`; the sector snaps ride along by reference.
            let sector_donors: Vec<Option<&CornerDonor>> = per_corner_sector_donor_idx
                [corner_index]
                .iter()
                .map(|slot| slot.map(|index| bank.donor(index)))
                .collect();
            // §5.2.4.3: this corner's per-sector own-rep centre donors, resolved against the
            // same pristine `bank`. All `None` when `v` is seam-carrying (deferred branch).
            let centre_donors: Vec<Option<&CornerDonor>> = per_corner_centre_donor_idx
                [corner_index]
                .iter()
                .map(|slot| slot.map(|index| bank.donor(index)))
                .collect();
            let interior_blend = AdjInteriorBlend {
                corner_vertex: corner.vertex,
                sector_snaps: &per_corner_sector_snaps[corner_index],
                sector_donors: &sector_donors,
                centre_donors: &centre_donors,
            };
            // §8.5.5.5(f.3) law **F2**: the line / flat-chamfer exponent (`r == 1.0`,
            // `profile = 0.25`) is the *same* general fill with `fullness = 0` — a planar
            // cap on the boundvert-ring plane, reached via `emit_general_adj_fill`'s `flat`
            // path. Every other exponent (circle, the curved band, both square sentinels)
            // relaxes the fullness-seeded net instead.
            self.emit_general_adj_fill(
                corner.vertex,
                chains,
                &profiles,
                segments,
                options.profile,
                custom,
                exponent == 1.0 && custom.is_none(),
                None,
                options.material,
                &sector_representatives,
                // Vertex bevel: the centre column follows the grid — the next-boundvert
                // rep (§6.6.2's grid row, vertex arm) — with no edge-bevel own-vs-tie-break
                // exception. So the centre column reuses the sector reps.
                &sector_representatives,
                central_representative,
                apex_blend.as_ref(),
                Some(&interior_blend),
                // Vertex mode never reaches the §8.3.4.1 pipe test, so nothing reads the walk.
                &mut AdjNetSlots::off(),
                &mut result,
            );
        }

        // §5.2.1 fact (c): blend the re-stitched originals' ring-vert corners. The fill's
        // own chain-interior verts were already minted-from-`v`-and-blended per §5.2.4.2
        // (the even-`segments` grid) / §5.2.4.1 (the welded apex) inside the fill above.
        self.blend_restitched_originals(&rebuilt, &bank);

        for corner in &corners {
            self.free_loose_vertex(corner.vertex);
        }

        Some(result)
    }

    /// Phase 1a (BEVEL.md §6.4.1): capture each beveled vertex's per-edge
    /// placement frame against the pristine mesh — disk-ordered incident edges,
    /// far-endpoint positions, the resolved (unclamped) slide distance and the
    /// per-offset slide rate `k` each edge contributes. A vertex with fewer than
    /// two incident edges has no corner to round (§2.1) and is dropped here.
    ///
    /// Two incident edges is the **whole** eligibility test (§2.1): there is no manifoldness,
    /// face-count or fan-shape gate anywhere in vertex mode. A boundary corner, a bow-tie, a
    /// corner carrying wires, and a corner with **nothing but** wires (the face-less star of
    /// §6.2.1) are all beveled, and all by the same rules — the count delta is a function of
    /// the valence alone (§2.3).
    pub(super) fn gather_bevel_frames(
        &self,
        beveled: &[VertexKey],
        offset: f32,
        offset_type: BevelOffsetType,
        use_weights: bool,
        vert_weights: &[(VertexKey, f32)],
    ) -> Vec<BevelFrame> {
        let mut frames = Vec::with_capacity(beveled.len());
        for &vertex in beveled {
            let incident_edges: Vec<EdgeKey> = self.iter_disk_edges(vertex).collect();
            if incident_edges.len() < 2 {
                continue;
            }

            let wires: Vec<EdgeKey> = incident_edges
                .iter()
                .copied()
                .filter(|&edge| self.edge_radial_iter(edge).next().is_none())
                .collect();
            // Only wires need the ring here, and building it is O(F²) in the fan — so the
            // wire-free corner, which is nearly every corner, must not pay for it. And only a
            // **closed** fan hosts a wire on a face at all (§7.2.1's vertex-mode box): where an
            // open or multi-chain fan's chain runs out is a gap, not a faced sector.
            let wire_hosts = if wires.is_empty() {
                Vec::new()
            } else if self.face_fan_is_closed(&incident_edges, &self.vert_face_keys(vertex)) {
                let corner_ring = self.vertex_corner_ring(vertex);
                self.wire_host_faces(vertex, &corner_ring, &wires)
            } else {
                Vec::new()
            };
            let disk_order = self.vertex_mode_disk_order(vertex, &incident_edges, &wire_hosts);

            // §8.7.2 / §8.7.3: the per-corner weight prescales this vertex's
            // offset uniformly across all its incident edges, *before* the §4 /
            // §4.3 slide solve. Gate off ⇒ weight 1.0 (full offset); gate on but
            // absent from the slice ⇒ weight 0.0 (no slide for this corner).
            let weight = corner_weight(use_weights, vert_weights, vertex);
            let effective_offset = offset * weight;

            let vertex_position = self.vertex_position(vertex);
            let far_verts: Vec<VertexKey> = incident_edges
                .iter()
                .map(|&edge| {
                    let [a, b] = self.edge_vert_keys(edge);
                    if a == vertex {
                        b
                    } else {
                        a
                    }
                })
                .collect();
            let far_positions: Vec<Vec3> =
                far_verts.iter().map(|&f| self.vertex_position(f)).collect();
            let corner_axis = corner_axis(vertex_position, &far_positions);

            frames.push(BevelFrame {
                vertex,
                vertex_position,
                disk_order,
                wire_hosts,
                placements: incident_edges
                    .iter()
                    .zip(&far_verts)
                    .zip(&far_positions)
                    .map(|((&edge, &far_vertex), &far_position)| {
                        // §8.7.3: prescale the corner offset, then solve the §4 /
                        // §4.3 slide unchanged. The solve is linear in offset, so
                        // this is identical to scaling the resolved slide by the
                        // weight afterward.
                        let slide = slide_distance(
                            effective_offset,
                            offset_type,
                            vertex_position,
                            far_position,
                            corner_axis,
                        );
                        // §4.4.2 / §8.7.4: the per-offset slide rate `k = d_i /
                        // offset`, the §4.3 `offset_type` factor used by the
                        // collision clamp. Derived from the *weighted* slide
                        // against the requested offset, so the clamp sees the
                        // weighted rates (§8.7.4: the weight is the inner scale,
                        // the clamp the outer).
                        let rate = if offset != 0.0 { slide / offset } else { 0.0 };
                        EdgePlacement {
                            edge,
                            far_vertex,
                            far_position,
                            slide,
                            rate,
                        }
                    })
                    .collect(),
            });
        }
        frames
    }

    /// Phase 1b: mint every ring vert (§4) at its clamp-scaled slide position, and capture
    /// each beveled vertex's ring cycle and incident-face set. No face surgery happens here —
    /// only vertex creation.
    pub(super) fn place_ring_verts(
        &mut self,
        frames: &[BevelFrame],
        clamp_factor: f32,
        segments: u32,
        templates: &mut TemplateBank,
        result: &mut BevelResult,
    ) -> Vec<BeveledCorner> {
        let mut corners = Vec::with_capacity(frames.len());
        for frame in frames {
            let incident_faces = self.vert_face_keys(frame.vertex);

            // §6.6.1 / §6.6.1.1: the cap is a ring-spanning fill, so its stage-1 inherit
            // donor is `prev_face` of the fan's **head edge** `e[0]` — which §7.6.1.2's
            // vertex-bevel reduction pins to the spoke array's slot 0, i.e. to
            // `disk_order[0]`. Every spoke carries a boundvert in vertex mode, so slot 0
            // *is* the ring's first boundvert and there is no beveled-spoke scan on top.
            // Displaced by §6.6.1.4's tie-break at a seam-carrying corner, which every
            // corner touching the mesh boundary — and every corner carrying a wire — is. No
            // leg is beveled in vertex mode. Registered now, against the pristine mesh —
            // the bank is sealed before the incident faces are killed / rebuilt.
            let head_rep = frame
                .disk_order
                .first()
                .and_then(|&head| self.boundvert_representative(frame.vertex, head));
            let representative_face = self.anchored_ring_spanning_donor(
                frame.vertex,
                &|_| false,
                BevelAffect::Vertices,
                segments,
                head_rep,
                templates.interp_components(),
            );
            let representative = templates.register(representative_face);

            let mut ring_vert_for_edge: Vec<(EdgeKey, VertexKey)> =
                Vec::with_capacity(frame.placements.len());
            for placement in &frame.placements {
                let slide = placement.slide * clamp_factor;
                // §4.1: the ring vert is a plain slide, so it takes the shared
                // primitive's overshoot guard — reached here from any
                // `offset_type`, not just from an over-large `OFFSET`.
                let ring_position =
                    slide_along_edge(frame.vertex_position, placement.far_position, slide);
                // §5.2.1 step 1: mint from the beveled corner, not from the layer
                // defaults — the copy *is* the initialisation, and it is what
                // survives on any layer the blend of step 2 does not write.
                let ring_vertex = self.mint_bevel_vertex(ring_position, frame.vertex);
                ring_vert_for_edge.push((placement.edge, ring_vertex));
                result.verts.push(ring_vertex);
            }

            // §7.2.1's vertex-mode box, closed-fan half: a wire bounds no sector of its own, so
            // on a closed fan its ring vert is adopted by the face whose sector holds it —
            // resolved with the frame, against the pristine mesh. An open fan slots its wires
            // in the gap and hosts none.
            let hosted_wires: Vec<(FaceKey, VertexKey)> = frame
                .wire_hosts
                .iter()
                .map(|&(wire, host)| (host, self.ring_vertex_of(&ring_vert_for_edge, wire)))
                .collect();

            // §5.1: the ring the cap walks — the disk cycle, in disk order.
            let ring_order: Vec<VertexKey> = frame
                .disk_order
                .iter()
                .map(|&edge| self.ring_vertex_of(&ring_vert_for_edge, edge))
                .collect();

            corners.push(BeveledCorner {
                vertex: frame.vertex,
                ring_vert_for_edge,
                incident_faces,
                ring_order,
                hosted_wires,
                representative,
                representative_face,
            });
        }
        corners
    }

    /// Phase 2 (BEVEL.md §6.1 / §6.4.3): re-stitch every face incident to any
    /// beveled vertex, swapping *all* of that face's beveled corners at once.
    /// Each beveled corner `v` becomes the pair (ring vert of `v`'s incoming
    /// edge, ring vert of `v`'s outgoing edge), keeping the face winding; the
    /// only freshly minted edge is the chamfer edge between the pair.
    ///
    /// **Every incident edge of a beveled corner survives in place.** §6.1 says a
    /// re-stitched face's two corner edges become `prev — r_i` (the original
    /// incoming edge `e_i`) and `r_(i+1) — next` (the original outgoing edge
    /// `e_(i+1)`) — *the same edges re-pointed*, not new ones — and §6.4.3 states
    /// the general preservation principle: a surviving edge "keeps the original
    /// edge's identity (and any per-edge attributes), simply re-pointed". So each
    /// of a corner `v`'s incident edges `e` has its `v`-endpoint re-pointed to
    /// `e`'s ring vert for `v`. An edge with **one** beveled endpoint is swapped
    /// once (`v → r_v`); a **shared** edge with both endpoints beveled is swapped
    /// once from each corner (`a → r_a`, then `b → r_b`), so the both-beveled case
    /// is just the general re-point applied twice. That up-front pass
    /// (`repoint_incident_edges`) uses the in-place re-point primitive (no
    /// kill/recreate, so each edge keeps the **same `EdgeKey` and `edge_attribs`
    /// slot**). The subsequent face rebuild then *adopts* every re-pointed edge
    /// (the face builder reuses an existing edge rather than minting a duplicate,
    /// BEVEL.md §6.1 / §6.4.3) and mints only the chamfer edges fresh. An interior
    /// shared edge ends with radial count 2 — both straddling faces rebuilt onto
    /// the same surviving edge.
    pub(super) fn restitch_incident_faces(
        &mut self,
        beveled: &[VertexKey],
        corners: &[BeveledCorner],
        bank: &DonorBank,
    ) -> Vec<RebuiltOriginal> {
        // Collect each distinct incident face once, regardless of how many
        // beveled corners it has, so a shared-edge face is rebuilt a single time.
        let mut faces: Vec<FaceKey> = Vec::new();
        for corner in corners {
            for &face in &corner.incident_faces {
                if !faces.contains(&face) {
                    faces.push(face);
                }
            }
        }

        // Compute every face's rebuilt vertex cycle against the *current* mesh,
        // before any re-point or kill mutates the disks. Each face's cycle and
        // the edge lookups it depends on (which ring vert each beveled corner's
        // incoming / outgoing edge carries) are read while the original incident
        // edges still connect their original endpoints.
        // Each freshly-minted chamfer edge and the original incident edge whose
        // per-edge CD it inherits (BEVEL.md §6.5.1): the rebuilt face's
        // **winding-outgoing** edge at the corner — the one the face's corner at `v`
        // *leaves* along — and never the winding-incoming one, which is the same
        // donor rule the edge-bevel corner runs take (§7.7.3). The owning EdgeKey is
        // recorded against the original mesh and survives the in-place re-point below
        // (same key / same CD slot), so it is still readable after the rebuild mints
        // the chamfer edge.
        let mut chamfer_cd: Vec<ChamferInheritance> = Vec::new();

        let mut rebuilds: Vec<(FaceKey, Vec<VertexKey>)> = Vec::with_capacity(faces.len());
        for &face in &faces {
            let cycle_verts = self.face_vert_keys(face);
            let cycle_length = cycle_verts.len();

            let mut new_cycle: Vec<VertexKey> = Vec::with_capacity(cycle_length + 1);
            for (index, &w) in cycle_verts.iter().enumerate() {
                if beveled.contains(&w) {
                    let corner = corners
                        .iter()
                        .find(|c| c.vertex == w)
                        .expect("a beveled corner vertex has a ring");
                    let prev_vertex = cycle_verts[(index + cycle_length - 1) % cycle_length];
                    let next_vertex = cycle_verts[(index + 1) % cycle_length];

                    let incoming_edge = self
                        .query_edge_exists(prev_vertex, w)
                        .expect("incident face's incoming corner edge exists");
                    let outgoing_edge = self
                        .query_edge_exists(w, next_vertex)
                        .expect("incident face's outgoing corner edge exists");

                    let leading_ring =
                        self.ring_vertex_of(&corner.ring_vert_for_edge, incoming_edge);
                    let trailing_ring =
                        self.ring_vertex_of(&corner.ring_vert_for_edge, outgoing_edge);

                    // §6.5.1 (cap-frame leading): the chamfer edge
                    // (leading_ring, trailing_ring) inherits the per-edge CD of the
                    // **outgoing** original edge — the owning edge of the cap's
                    // leading endpoint (`trailing_ring` in this incident-face walk).
                    //
                    // §6.7.1.1: its seam / sharp bit, however, is the *logical AND*
                    // of the two flanking owning edges — the leading owner
                    // `incoming_edge` (`e_i`) and the trailing owner `outgoing_edge`
                    // (`e_(i+1)`) — so a single seamed incident edge leaves no
                    // dead-end spur on the cap. Record both owners.
                    //
                    // §7.2.1's vertex-mode box: on a face that hosts one of this corner's
                    // wires the corner run is `leading — wire… — trailing`, so the face
                    // grows by two sides (not one) and the run mints two chamfer segments
                    // (not one). Every segment takes the same owners: a wire spoke is a
                    // donor for nothing (§8.3.0.4 point 5), so it neither supplies a
                    // per-edge block nor participates in the seam / sharp AND.
                    //
                    // The run is written straight into `new_cycle` and read back off its
                    // tail — the wire-free corner (the common case) must not pay for a
                    // temporary it would never fill.
                    let run_start = new_cycle.len();
                    new_cycle.push(leading_ring);
                    for &(host, wire_ring) in &corner.hosted_wires {
                        if host == face {
                            new_cycle.push(wire_ring);
                        }
                    }
                    new_cycle.push(trailing_ring);

                    for segment in new_cycle[run_start..].windows(2) {
                        chamfer_cd.push(ChamferInheritance {
                            leading_ring: segment[0],
                            trailing_ring: segment[1],
                            cd_owner: outgoing_edge,
                            leading_owner: incoming_edge,
                            trailing_owner: outgoing_edge,
                        });
                    }
                } else {
                    new_cycle.push(w);
                }
            }
            rebuilds.push((face, new_cycle));
        }

        // §6.1 / §6.4.3: re-point every incident edge of every beveled corner in
        // place *after* the cycles are captured but *before* any face is killed,
        // preserving each edge's identity / per-edge CD. After this every
        // corner-adjacent edge connects its ring vert(s); the face rebuild below
        // reuses each one (the face builder's NO_DUPLICATE reuse) rather than
        // minting it fresh, leaving only the chamfer edges newly minted.
        self.repoint_incident_edges(corners);

        // Capture every rebuilt face's block in one snapshot, before any kill:
        // a rebuilt face may recycle a killed face's slab slot, and a minted
        // element's attribute block is default-stamped, so the dead face's slot
        // is no longer a source once the loop below is under way.
        let carried_slots: Vec<usize> = rebuilds.iter().map(|(f, _)| f.as_usize()).collect();
        let carried = self.face_attribs.snapshot_slots(&carried_slots);

        let mut rebuilt: Vec<RebuiltOriginal> = Vec::with_capacity(rebuilds.len());
        for (index, (face, new_cycle)) in rebuilds.into_iter().enumerate() {
            let donor = bank.index_of(face);
            debug_assert!(
                donor.is_some() || bank.is_empty(),
                "a re-stitched original was not captured as a donor pre-bevel",
            );
            self.face_kill(face);
            let new_face = self.face_create_verts(&new_cycle, FaceCreateFlags::NONE);
            self.face_attribs
                .restore_slot(new_face.as_usize(), &carried, index);
            rebuilt.push(RebuiltOriginal {
                face: new_face,
                donor,
            });
        }

        // §6.5.1: copy each owning (outgoing) edge's per-edge CD onto the chamfer
        // edge it owns. Run after every face is rebuilt so the chamfer edge exists
        // (the cap later *adopts* this same edge, §5.1, so no separate cap step is
        // needed). The chamfer edge joins two ring verts of one corner — never a
        // re-pointed surviving edge — so copying here cannot clobber a re-pointed
        // edge's own preserved CD.
        for inheritance in chamfer_cd {
            let Some(chamfer_edge) =
                self.query_edge_exists(inheritance.leading_ring, inheritance.trailing_ring)
            else {
                continue;
            };
            self.edge_attribs
                .copy_into(chamfer_edge.as_usize(), inheritance.cd_owner.as_usize());

            // §6.7.1.1: the whole-block copy above carried the trailing owner's
            // seam / sharp bit (mechanism (1)'s raw §6.5.1 copy). The corner
            // segment keeps the flag only when *both* flanking owners carry it, so
            // re-derive each bit as the AND of the leading and trailing owners,
            // overriding whatever the block copy left. Unconditional — not gated by
            // `mark_seam` / `mark_sharp` (§6.7.1 mechanism (1)).
            let seam = self.edge_seam(inheritance.leading_owner)
                && self.edge_seam(inheritance.trailing_owner);
            let sharp = self.edge_sharp(inheritance.leading_owner)
                && self.edge_sharp(inheritance.trailing_owner);
            self.set_edge_seam(chamfer_edge, seam);
            self.set_edge_sharp(chamfer_edge, sharp);
        }

        rebuilt
    }

    /// Re-point every incident edge of every beveled corner in place
    /// (BEVEL.md §6.1 / §6.4.3).
    ///
    /// For a beveled corner `v` and one of its incident edges `e` carrying ring
    /// vert `r`, this swaps `e`'s `v`-endpoint to `r` with the in-place edge
    /// re-point primitive — the same `EdgeKey`, the same `edge_attribs` slot,
    /// just a new endpoint — so the re-pointed edge *is* the original edge with
    /// its identity and per-edge CD preserved. The re-point also patches `e`'s
    /// radial loops; those loops belong to faces about to be killed and rebuilt,
    /// so the transient face-cycle inconsistency it leaves is immediately
    /// discarded.
    ///
    /// An edge with a **single** beveled endpoint is re-pointed exactly once
    /// (`v → r`). A **shared** edge `E = {a, b}` whose *both* endpoints are
    /// beveled is reached once from `a`'s ring and once from `b`'s ring, so the
    /// loop below swaps it `a → r_a` when processing `a` and `b → r_b` when
    /// processing `b`, leaving `E` as the surviving inner segment `{r_a, r_b}`
    /// (§6.4.3) — the both-beveled case is the general re-point applied twice. The
    /// `(edge, endpoint)` guard ensures each endpoint of each edge is swapped at
    /// most once even if a corner's ring lists the same edge twice (a degenerate
    /// self-edge cannot arise here, but the guard keeps the pass idempotent).
    pub(super) fn repoint_incident_edges(&mut self, corners: &[BeveledCorner]) {
        let mut done: Vec<(EdgeKey, VertexKey)> = Vec::new();
        for corner in corners {
            let v = corner.vertex;
            for &(edge, ring) in &corner.ring_vert_for_edge {
                if done.contains(&(edge, v)) {
                    continue;
                }
                // Precondition of the re-point primitive: the new vert (the ring
                // vert) must not already be on the edge. Ring verts are freshly
                // minted, so this holds for both the single- and both-beveled
                // cases.
                self.edge_replace_vertex_raw(edge, v, ring);
                done.push((edge, v));
            }
        }
    }

    pub(super) fn ring_vertex_of(
        &self,
        ring_vert_for_edge: &[(EdgeKey, VertexKey)],
        edge: EdgeKey,
    ) -> VertexKey {
        ring_vert_for_edge
            .iter()
            .find_map(|&(e, r)| (e == edge).then_some(r))
            .expect("every incident edge has a ring vertex")
    }

    /// Does the corner's face chain wrap all the way round — every faced spoke manifold
    /// (radial count 2) and as many faces as faced spokes? This decides **only** where a wire
    /// spoke's slot in the disk order falls (BEVEL.md §7.2.1's vertex-mode box): a closed fan
    /// runs out at a seam whose two spokes share a face, so that face hosts the wire; an open
    /// fan (a radial-1 spoke) or a multi-chain one (a bow-tie, a non-manifold spoke) runs out
    /// in a faceless **gap**, where nothing can host it.
    ///
    /// It is **not** the cap gate. Valence >= 3 is the cap gate, and it is the whole of it
    /// (§2.3, §6.2): a boundary corner, a bow-tie and a face-less wire star are capped exactly
    /// like an interior one.
    ///
    /// A **wire** spoke (radial count 0) is skipped here even though vertex mode keeps it in
    /// the disk as a ring member: it carries no face, so it neither closes nor opens the face
    /// fan.
    pub(super) fn face_fan_is_closed(
        &self,
        incident_edges: &[EdgeKey],
        incident_faces: &[FaceKey],
    ) -> bool {
        let mut manifold_spokes = 0usize;
        for &edge in incident_edges {
            match self.edge_radial_iter(edge).count() {
                0 => {}
                2 => manifold_spokes += 1,
                _ => return false,
            }
        }
        !incident_faces.is_empty() && incident_faces.len() == manifold_spokes
    }

    /// Each **wire** spoke of a **closed-fan** `vertex` paired with the incident face that
    /// adopts its ring vert in vertex mode (BEVEL.md §7.2.1's vertex-mode box) — the face that
    /// therefore grows by *two* sides rather than one, while every other rebuilt face grows by
    /// one.
    ///
    /// **Only a closed fan reaches here.** A wire shares no face with either disk neighbour, so
    /// nothing inside the walk pins its slot: it is appended where the face chain runs out. On a
    /// closed fan the chain wraps, so that slot is a seam *inside* a faced sector and the sector's
    /// face hosts the wire. On an open or multi-chain fan it is a faceless **gap** — no face can
    /// adopt it there, and the caller does not ask.
    ///
    /// Which face a closed fan hands the wire to is an order-of-construction artefact and not an
    /// invariant (§7.2.1 again) — which is exactly the licence for the *preferred* answer tried
    /// first here. When the wire's direction, flattened into the corner's tangent plane, falls
    /// inside the rotational sector of two consecutive **faced** spokes that bound a common face,
    /// that face hosts it. Every inconclusive read falls back to the chain end — an on-axis wire
    /// (it flattens to nothing and lies in no sector in particular), a corner with no flattenable
    /// faced spoke to take a bearing from, fewer than two spokes to bound a sector with, and no
    /// sector found (spokes that all share one bearing, or a projection that reordered them).
    ///
    /// Wires sharing a host come out in the order that host's rebuilt corner run wants them:
    /// the run enters on the face's incoming edge and leaves on its outgoing one, sweeping
    /// the sector *clockwise* about the corner's outward normal (the faces wind
    /// counter-clockwise about it), so they are keyed on that clockwise offset.
    pub(super) fn wire_host_faces(
        &self,
        vertex: VertexKey,
        corner_ring: &[VertexCorner],
        wires: &[EdgeKey],
    ) -> Vec<(EdgeKey, FaceKey)> {
        if wires.is_empty() {
            return Vec::new();
        }
        let chain_end = corner_ring
            .last()
            .expect("a corner hosting a wire has a face chain to host it on")
            .face;
        let on_chain_end = || wires.iter().map(|&wire| (wire, chain_end)).collect();

        let origin = self.vertex_position(vertex);
        // Unit, or the tangent plane is not a plane: `flatten` below would stop being an
        // orthogonal projection and its `sine` would stop being a sine, taking the scale-free
        // reading of `TANGENT_SINE_EPSILON` with it. Nothing to take a bearing in, so:
        let normal = self.surface_normal_at(vertex);
        if normal == Vec3::ZERO {
            return on_chain_end();
        }
        // A spoke's direction projected into the corner's tangent plane, paired with the
        // **sine** of its angle to the normal — a scale-free measure of how much direction
        // survived the projection, so the degeneracy tests below do not depend on the mesh's
        // units the way a bare length threshold would.
        let flatten = |edge: EdgeKey| -> (Vec3, f32) {
            let [a, b] = self.edge_vert_keys(edge);
            let far = if a == vertex { b } else { a };
            let direction = self.vertex_position(far) - origin;
            let flattened = direction - normal * direction.dot(normal);
            let length = direction.length();
            let sine = if length > 0.0 {
                flattened.length() / length
            } else {
                0.0
            };
            (flattened, sine)
        };

        let faced: Vec<(EdgeKey, Vec3)> = self
            .iter_disk_edges(vertex)
            .filter(|&edge| self.edge_radial_iter(edge).next().is_some())
            .filter_map(|edge| {
                let (flattened, sine) = flatten(edge);
                (sine > TANGENT_SINE_EPSILON).then_some((edge, flattened))
            })
            .collect();
        let Some(&(_, reference_direction)) = faced.first() else {
            return on_chain_end();
        };

        let reference = reference_direction.normalize();
        let cross_axis = normal.cross(reference);
        let bearing = |direction: Vec3| -> f32 {
            direction
                .dot(cross_axis)
                .atan2(direction.dot(reference))
                .rem_euclid(std::f32::consts::TAU)
        };

        let mut spokes: Vec<(f32, EdgeKey)> = faced
            .iter()
            .map(|&(edge, flattened)| (bearing(flattened), edge))
            .collect();
        spokes.sort_by(|a, b| a.0.total_cmp(&b.0));
        if spokes.len() < 2 {
            return on_chain_end();
        }

        let mut hosted: Vec<(EdgeKey, FaceKey, f32)> = Vec::with_capacity(wires.len());
        for &wire in wires {
            let (flattened, sine) = flatten(wire);
            if sine <= TANGENT_SINE_EPSILON {
                hosted.push((wire, chain_end, 0.0));
                continue;
            }
            let wire_bearing = bearing(flattened);
            let host = self
                .sector_face(corner_ring, &spokes, wire_bearing)
                .unwrap_or(chain_end);
            hosted.push((wire, host, wire_bearing));
        }

        // Stable, so wires the run cannot order geometrically (an on-axis one, a fallback
        // that landed on a face it does not point into) keep their disk order.
        hosted.sort_by(|left, right| {
            let run_offset = |&(_, host, wire_bearing): &(EdgeKey, FaceKey, f32)| {
                let Some((incoming, _)) = self.face_corner_edges(host, vertex) else {
                    return 0.0;
                };
                let (flattened, sine) = flatten(incoming);
                if sine <= TANGENT_SINE_EPSILON {
                    return 0.0;
                }
                (bearing(flattened) - wire_bearing).rem_euclid(std::f32::consts::TAU)
            };
            run_offset(left).total_cmp(&run_offset(right))
        });
        hosted
            .into_iter()
            .map(|(wire, host, _)| (wire, host))
            .collect()
    }

    /// The face bounded by the two rotationally consecutive faced spokes whose sector holds
    /// `wire_bearing`. `None` when no sector holds it, or when the holding sector's two
    /// spokes bound no common face — on an open fan, its face-less gap.
    pub(super) fn sector_face(
        &self,
        corner_ring: &[VertexCorner],
        spokes: &[(f32, EdgeKey)],
        wire_bearing: f32,
    ) -> Option<FaceKey> {
        let sector = (0..spokes.len()).find(|&index| {
            let (start, _) = spokes[index];
            let (end, _) = spokes[(index + 1) % spokes.len()];
            let span = (end - start).rem_euclid(std::f32::consts::TAU);
            let into_sector = (wire_bearing - start).rem_euclid(std::f32::consts::TAU);
            into_sector < span
        })?;

        let (_, first) = spokes[sector];
        let (_, second) = spokes[(sector + 1) % spokes.len()];
        corner_ring
            .iter()
            .find(|corner| {
                (corner.incoming_edge == first && corner.outgoing_edge == second)
                    || (corner.incoming_edge == second && corner.outgoing_edge == first)
            })
            .map(|corner| corner.face)
    }

    /// Close the corner with one n-gon over the **whole** ring (BEVEL.md §5 step 4 / §5.1).
    ///
    /// `ring` is the corner's ring verts in disk order, so the cap's boundary walk is simply
    /// that order: the cap runs **with** the disk, `r_0 → r_1 → … → r_(n−1)`, which is
    /// equivalently "opposite every rebuilt face on the chamfer edge they share" — a rebuilt
    /// face's corner run is descending, `r_(i+1), r_i` (§6.1). Both readings are one rule, and
    /// running the ring *against* the disk order is the inward-cap bug. The winding is
    /// re-derived here from a rebuilt face wherever one exists, so the observable half of the
    /// rule — every shared ring edge walked in opposite directions by its face and the cap, and
    /// so a radial count of 2 — holds whichever sense the disk walk came back in.
    ///
    /// Each ring edge is **adopted if it exists and created if it does not**: the faced steps'
    /// chamfer edges were minted by the re-stitch that cut those faces' corners, and the gap
    /// steps have no face to have minted theirs (§2.3's provenance table), so the cap mints
    /// them itself — one for an open fan, one per gap for a bow-tie, all `n` for a face-less
    /// wire star. Never a second, coincident edge over a pair that already has one, which
    /// would split one radial cycle of 2 into two boundary edges (§5.1).
    pub(super) fn build_cap_face(&mut self, ring: &[VertexKey]) -> CapFace {
        debug_assert!(ring.len() >= 3, "a cap spans at least three ring verts");

        let mut cap_ring = ring.to_vec();
        if self.ring_runs_with_a_rebuilt_face(&cap_ring) {
            cap_ring.reverse();
        }

        let ring_edges: Vec<EdgeKey> = (0..cap_ring.len())
            .map(|index| {
                let from = cap_ring[index];
                let to = cap_ring[(index + 1) % cap_ring.len()];
                // §6.5.1: a gap edge has no incident original face to inherit from, so the
                // mint's default per-edge block stands.
                self.create_edge([from, to], EdgeCreateFlags::NO_DUPLICATE)
            })
            .collect();

        let face = self.face_create(&cap_ring, &ring_edges, FaceCreateFlags::NONE);
        CapFace { face, ring_edges }
    }

    /// Does `ring`, walked in its own order, traverse some shared chamfer edge the **same**
    /// way the face on it does (BEVEL.md §5.1)? A rebuilt face runs its chamfer edge
    /// descending, so a correctly-wound cap never agrees with it; `true` means the ring came
    /// back reversed and the cap would face inward.
    ///
    /// Answers `false` on a face-less star, where no ring edge is shared with anything and the
    /// cap's orientation is fixed by the disk-order convention alone (and is unobservable).
    pub(super) fn ring_runs_with_a_rebuilt_face(&self, ring: &[VertexKey]) -> bool {
        for index in 0..ring.len() {
            let from = ring[index];
            let to = ring[(index + 1) % ring.len()];
            let Some(chamfer_edge) = self.query_edge_exists(from, to) else {
                continue; // a gap step: no face minted it, so it says nothing about winding
            };
            let Some(&loop_key) = self.edge_loop_keys(chamfer_edge).first() else {
                continue;
            };
            let start = self.loop_vertex(loop_key);
            let end = self.loop_vertex(self.loop_next(loop_key));
            return start == from && end == to;
        }
        false
    }

    /// The vertex-mode disk order (BEVEL.md §3.2, §5 step 1): the **full** incident-edge set of
    /// `vertex` — wire spokes included, per §7.2.1's vertex-mode box — as one cyclic sequence.
    /// This is *not* edge mode's face-bearing disk, which filters the wires out; reusing that
    /// here would return an empty ring on a face-less star and drop a wire's ring vert anywhere
    /// else.
    ///
    /// The order is the winding walk `e_i → f_i → e_(i+1)`, where `f_i` is the face whose
    /// corner at `vertex` *leaves* along `e_i`. The walk chains maximal runs of face-sharing
    /// spokes and **restarts** on the next unvisited one when a run ends, concatenating the
    /// chains into a single cycle — which is why a bow-tie corner, whose faces form two
    /// disjoint fans, still rings up as one cycle whose between-chain steps are the faceless
    /// gaps (§6.2).
    ///
    /// A wire shares a face with nobody, so the walk cannot place it: it is appended where the
    /// chain runs out. On a closed fan that seam lies inside a faced sector, and `wire_hosts`
    /// names the face — the wire slots between that face's two spokes, right after the one it
    /// *leaves* along, so the ascending disk order reads `e_i, wires…, e_(i+1)` and the host's
    /// descending corner run reads them back the other way (§6.1). Otherwise the wire lands in
    /// the trailing gap, adopted by no face and caught only by the cap.
    pub(super) fn vertex_mode_disk_order(
        &self,
        vertex: VertexKey,
        incident_edges: &[EdgeKey],
        wire_hosts: &[(EdgeKey, FaceKey)],
    ) -> Vec<EdgeKey> {
        // One step per incident face: in along the spoke the face leaves `vertex` on (`e_i`),
        // out along the spoke it arrives on (`e_(i+1)`).
        let mut steps: Vec<(EdgeKey, EdgeKey)> = Vec::new();
        for face in self.vert_face_keys(vertex) {
            if let Some((arrives_on, leaves_on)) = self.face_corner_edges(face, vertex) {
                if arrives_on != leaves_on {
                    steps.push((leaves_on, arrives_on));
                }
            }
        }

        let mut used = vec![false; steps.len()];
        let mut order: Vec<EdgeKey> = Vec::with_capacity(incident_edges.len());
        let mut runs = 0usize;
        while let Some(seed) = (0..steps.len()).find(|&index| !used[index]) {
            runs += 1;
            used[seed] = true;
            let (seed_enters, seed_leaves) = steps[seed];
            let mut chain = vec![seed_enters, seed_leaves];

            // Forward: the next face is the one that leaves `vertex` along the spoke this one
            // arrived on. A face-chain neighbour wound the other way matches on its own
            // arriving spoke instead, so take either and continue on the far side.
            loop {
                let frontier = *chain.last().expect("the chain is seeded with two spokes");
                let next = steps.iter().enumerate().find(|(index, &(enters, leaves))| {
                    !used[*index] && (enters == frontier || leaves == frontier)
                });
                let Some((index, &(enters, leaves))) = next else {
                    break;
                };
                used[index] = true;
                let onward = if enters == frontier { leaves } else { enters };
                if onward == chain[0] {
                    break; // the fan closed
                }
                chain.push(onward);
            }

            // Backward, since the seed can sit mid-chain on an open fan.
            loop {
                let frontier = chain[0];
                let previous = steps.iter().enumerate().find(|(index, &(enters, leaves))| {
                    !used[*index] && (enters == frontier || leaves == frontier)
                });
                let Some((index, &(enters, leaves))) = previous else {
                    break;
                };
                used[index] = true;
                let backward = if leaves == frontier { enters } else { leaves };
                if backward == *chain.last().expect("non-empty chain") {
                    break;
                }
                chain.insert(0, backward);
            }

            for edge in chain {
                if !order.contains(&edge) {
                    order.push(edge);
                }
            }
        }

        // A closed fan's wires: slotted inside their host's sector. `wire_hosts` is ordered as
        // the host's *descending* corner run wants them, and the disk order ascends, so
        // inserting each one at the same place reverses them into the order the run reads back.
        for &(wire, host) in wire_hosts {
            let leaves_on = self
                .face_corner_edges(host, vertex)
                .map(|(_, leaves_on)| leaves_on);
            match leaves_on.and_then(|edge| order.iter().position(|&e| e == edge)) {
                Some(slot) => order.insert(slot + 1, wire),
                None => order.push(wire),
            }
        }

        // Everything the walk could not place: the unhosted wires, which land in the gap where
        // the chain ran out — the tail of the concatenated order.
        for &edge in incident_edges {
            if !order.contains(&edge) {
                order.push(edge);
            }
        }
        debug_assert_eq!(
            order.len(),
            incident_edges.len(),
            "the disk order is a permutation of the incident-edge set",
        );
        self.anchor_vertex_mode_disk_order(vertex, order, &steps, runs)
    }

    /// Re-anchor the vertex-mode spoke array onto BEVEL.md [§7.6.1.2]'s **ring origin** — its
    /// vertex-bevel reduction, whose slot 0 is §6.6.1.1's head edge `e[0]` and therefore the
    /// vertex cap's donor. Three differences from the edge-bevel form, all from §7.6.1.2's
    /// scope bullets: step 3 disappears (every spoke carries a boundvert, so there is nothing
    /// to scan *for*), step 1 loses its beveled clause and falls back to the **disk anchor**,
    /// and the array's membership is every incident spoke, wires included — which the walk
    /// above already gives.
    ///
    /// That walk is the guide's step-2 array up to rotation and direction, so only those two
    /// have to be recovered:
    ///
    /// - **Direction.** Both are settled by the same aggregate vote (step 2: slot `i` votes
    ///   on the pair `(i, i + 1)` off its own forward face, and the array reverses iff the
    ///   sum is strictly negative), and that vote negates under reversal — so a strictly
    ///   negative tally here means the built walk runs against the sense the guide's array
    ///   ends in, whichever way step 2 happened to start. A **zero** tally is the one case
    ///   the two can disagree on with nothing to arbitrate: the guide leaves it unreversed,
    ///   i.e. pointing the way the walk out of the *seed* went, so that case follows the seed.
    /// - **Origin.** An unreversed step-2 array leaves the seed in slot 0; a reversed one
    ///   parks it in the last slot, so slot 0 is then the seed's cyclic successor. Which of
    ///   the two happened is read off the first step, which leaves the seed for whichever of
    ///   its **fan** neighbours comes first in the *stored* disk order — vacuous at a fan
    ///   end, whose other neighbour sits across the faceless gap.
    ///
    /// **Out of contract, and deliberately left on the walk's own arbitrary seed:** a disk
    /// whose faced spokes are not a single face-connected run, and a **branching** disk (a
    /// spoke carrying three or more faces). §7.6.1.2 orders the first by concatenating
    /// per-run walks under a seed rule of its own, and the second by a max-placement search;
    /// the walk above implements neither, so re-anchoring its output would move the origin
    /// without making it the guide's.
    pub(super) fn anchor_vertex_mode_disk_order(
        &self,
        vertex: VertexKey,
        order: Vec<EdgeKey>,
        steps: &[(EdgeKey, EdgeKey)],
        runs: usize,
    ) -> Vec<EdgeKey> {
        let count = order.len();
        if runs > 1 || count < 2 {
            return order;
        }
        if order
            .iter()
            .any(|&edge| self.edge_radial_iter(edge).take(3).count() > 2)
        {
            return order;
        }

        // Step 1: scan the stored disk order from the anchor. A radial-1 spoke (an open-fan
        // end) wins and the *last* one takes it; failing that the seed is the anchor itself,
        // which is where the scan began — the vertex-mode reduction of the beveled clause.
        let mut seed = None;
        for edge in self.iter_disk_edges(vertex) {
            if self.edge_radial_iter(edge).take(2).count() == 1 {
                seed = Some(edge);
            }
        }
        let Some(seed) = seed.or_else(|| self.iter_disk_edges(vertex).next()) else {
            return order;
        };
        let Some(seed_slot) = order.iter().position(|&edge| edge == seed) else {
            return order;
        };

        // A slot's forward face carries the two spokes as consecutive sides of its winding,
        // so `+1` is "some face leaves along `from` and arrives along `to`" and `-1` is that
        // same face read the other way round. Only "no such face at all" abstains.
        let vote = |from: EdgeKey, to: EdgeKey| -> i32 {
            if steps
                .iter()
                .any(|&(leaves, arrives)| leaves == from && arrives == to)
            {
                1
            } else if steps
                .iter()
                .any(|&(leaves, arrives)| leaves == to && arrives == from)
            {
                -1
            } else {
                0
            }
        };
        let tally: i32 = (0..count)
            .map(|slot| vote(order[slot], order[(slot + 1) % count]))
            .sum();

        let mut order = order;
        if tally < 0 {
            order.reverse();
        }

        let seed_slot = if tally < 0 {
            count - 1 - seed_slot
        } else {
            seed_slot
        };
        let forward = order[(seed_slot + 1) % count];
        let backward = order[(seed_slot + count - 1) % count];
        let shares_a_face = |left: EdgeKey, right: EdgeKey| vote(left, right) != 0;
        let mut follows_the_seed = match (
            shares_a_face(seed, forward),
            // A two-spoke disk folds the seed's two neighbours onto one spoke, so there is
            // no choice of first step to make and the seed stays in slot 0.
            forward != backward && shares_a_face(seed, backward),
        ) {
            (true, true) => {
                let rank = |edge: EdgeKey| {
                    self.iter_disk_edges(vertex)
                        .position(|candidate| candidate == edge)
                        .unwrap_or(usize::MAX)
                };
                rank(forward) < rank(backward)
            }
            (false, true) => false,
            // A fan end's only step, or a wire seed with no fan step at all: vacuous.
            _ => true,
        };
        if tally == 0 && !follows_the_seed {
            order.reverse();
            follows_the_seed = true;
        }

        let seed_slot = order
            .iter()
            .position(|&edge| edge == seed)
            .expect("the seed is a member of the disk order");
        let origin = if follows_the_seed {
            seed_slot
        } else {
            (seed_slot + 1) % count
        };
        order.rotate_left(origin);
        order
    }
}

/// The cap n-gon and the ring edges it bounds — the faced steps' chamfer edges, adopted, and
/// the gap steps', minted by the cap itself (BEVEL.md §2.3's provenance table).
pub(super) struct CapFace {
    pub(super) face: FaceKey,
    pub(super) ring_edges: Vec<EdgeKey>,
}

/// How one freshly-minted chamfer edge `(leading_ring, trailing_ring)` inherits
/// its per-edge attributes from the original incident edges of the beveled corner.
///
/// - `cd_owner` is the owning edge whose whole per-edge attribute block the
///   chamfer copies (BEVEL.md §6.5.1, cap-frame leading = the outgoing edge).
/// - `leading_owner` / `trailing_owner` are the two original incident edges
///   flanking the corner segment (the leading owner `e_i` of `leading_ring` and
///   the trailing owner `e_(i+1)` of `trailing_ring`). The chamfer's seam / sharp
///   bit is the AND of these two (BEVEL.md §6.7.1.1), overriding the block copy.
pub(super) struct ChamferInheritance {
    pub(super) leading_ring: VertexKey,
    pub(super) trailing_ring: VertexKey,
    pub(super) cd_owner: EdgeKey,
    pub(super) leading_owner: EdgeKey,
    pub(super) trailing_owner: EdgeKey,
}

/// Degeneracy floor for a spoke flattened into a corner's tangent plane (the §7.2.1
/// vertex-mode wire host read): the **sine** of the spoke's angle to the corner normal, so
/// the test is scale-free — a long spoke and a short one at the same angle read alike.
/// Below it the flattened direction carries no usable bearing.
pub(super) const TANGENT_SINE_EPSILON: f32 = 1.0e-4;
