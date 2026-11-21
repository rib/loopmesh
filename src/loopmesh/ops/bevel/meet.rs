//! The boundvert meet (BEVEL.md §7.6.2): offset lines, the two-line solve and the face drop.

use super::*;

impl LoopMesh {
    /// A junction vertex's outward surface normal: the summed plane normals of its
    /// incident faces (BEVEL.md §8.3.3 down-push sign reference). Used only to
    /// decide which way the fold crease points (into the solid).
    pub(super) fn junction_outward_normal(&self, vertex: VertexKey) -> Vec3 {
        let mut sum = Vec3::ZERO;
        for face in self.vert_face_keys(vertex) {
            sum += self.face_plane_normal(face);
        }
        sum.normalize_or_zero()
    }

    /// Place a sector's boundvert by the pairwise offset-line meet (BEVEL.md
    /// §7.6.2). The boundvert is fixed by *only* the sector's two flanking beveled
    /// edges — never a fit over every incident face. Build the offset line of
    /// `g_k`'s inner side and the offset line of `g_(k+1)`'s inner side, then
    /// intersect them. Three sector cases (no in-between edge / all-in-plane /
    /// non-planar) choose which plane each offset line is built in; the
    /// parallel / anti-parallel ladder handles the near-degenerate edge angles.
    pub(super) fn solve_meet_point(
        &self,
        vertex: VertexKey,
        vertex_position: Vec3,
        sector: &Sector,
        offset: f32,
    ) -> Vec3 {
        let layout = match self.sector_layout(vertex, vertex_position, sector) {
            Some(layout) => layout,
            None => return vertex_position,
        };

        let dir_first = layout.dir_first;
        let dir_second = layout.dir_second;

        // §7.6.2's near-collinear ladder, gated on the **direction angle** `θ_dir` under
        // the convention that fixes which branch is which: the first edge's direction is
        // taken *into* `v`, the second's *out* of `v`. Both `dir_first`/`dir_second` are
        // stored pointing *out* of `v`, so the into-`v` first direction is `-dir_first`.
        //
        // This convention is load-bearing: two collinear beveled edges running
        // straight *through* `v` (rays out of `v` pointing *opposite* ways) fold onto
        // the *same* into/out vector, so `θ_dir → 0` — the straight-through branch, whose
        // perpendicular push lands the boundvert one offset out along the sector's
        // leg. The doubled-back slide is reserved for the genuine double-back
        // (`θ_dir → 180°`: both edges leaving `v` the same way, a zero-area sliver).
        // Dispatching on the raw ray angle `dir_first · dir_second` swaps the two —
        // it sends the straight-through config to the slide, which puts the boundvert
        // on the wrong axis (along the edges instead of perpendicular to them).
        //
        // The band is the same `EPS_ANG` §7.5.1's regime 1 takes, on the same quantity;
        // both halves of the ladder then share one closed form in
        // `degenerate_meet_placement`, the doubled-back arm sliding along `bounding_edges[1]`
        // — the pair's second member, whose far position `far_second` is.
        let direction_angle = unit_chord_angle_between(-dir_first, dir_second);
        if is_degenerate_direction_angle(direction_angle) {
            return degenerate_meet_placement(
                vertex_position,
                layout.far_first,
                layout.far_second,
                direction_angle,
                layout.sector_normal,
                offset,
            );
        }

        // Build the two offset lines (§7.6.2 "Building the two offset lines"). The base
        // point is `v` pushed perpendicular to the edge, into the sector face, by
        // `offset`; the direction is the edge's own.
        //
        // Both directions are the **raw**, full-length flank vectors, each with the sign
        // the §7.6.2 convention gives it — the first taken *into* `v`, the second *out* of
        // it. The solve steps each base point along its direction to form the line's second
        // point (§7.6.2 "The arithmetic of the two-line solve"), and that step is a rounded
        // addition: stepping the other way rounds differently and costs a quantum at about
        // one shallow station in a thousand. The base points themselves are already right —
        // the perpendiculars were built on the *outward* rays with the cross order flipped
        // per role, which is the same computation bit for bit.
        let line_first = Line {
            point: vertex_position + layout.perp_first * offset,
            direction: vertex_position - layout.far_first,
        };
        let line_second = Line {
            point: vertex_position + layout.perp_second * offset,
            direction: layout.far_second - vertex_position,
        };

        // §7.6.2's face-drop gate, stated on the **sector** and not on the solve's
        // outcome: out-of-plane in-between edges *and* both flanking offsets `> 0`.
        // Whether the two case-3 offset lines happen to come out skew or exactly
        // crossing is a floating-point knife-edge — either way the provisional meet
        // floats above the surface and must be projected onto the first in-wedge sector
        // face. Reading the gate off the classification instead leaves a clean crossing
        // lifted above an in-wedge face: correct where no face is in-wedge (the crossing
        // stands, e.g. a reflex meet whose bisector lands on a rim vertex), wrong where
        // one is (e.g. an even-absorbed-leg reflex sector whose bisector falls inside a
        // straddle face — the crossing must drop onto it, §8.9.2 / §8.9.2.2).
        //
        // The other half of the same statement is what a **planar** sector does with a
        // skew pair: §7.6.2's "face-drop does not run" reading, which is the
        // closest-approach point on the *first* line — no midpoint, no projection. A
        // planar case-1/2 sector reaching `Skew` at all is `LINE_COPLANARITY_TOLERANCE`
        // being an absolute bound on a triple product that grows with the flanking
        // lengths, so it is reachable on a large mesh even where the sector's own
        // geometry is exactly coplanar.
        let face_drop_runs = layout.non_planar && offset > 0.0;
        match intersect_lines(&line_first, &line_second) {
            LineMeet::Crossing(point) if face_drop_runs => {
                self.drop_onto_sector_face(vertex_position, &layout, point)
            }
            LineMeet::Crossing(point) => point,
            LineMeet::Collinear => line_first.point,
            LineMeet::Skew { midpoint, .. } if face_drop_runs => {
                self.drop_onto_sector_face(vertex_position, &layout, midpoint)
            }
            LineMeet::Skew { on_first, .. } => on_first,
        }
    }

    /// The geometric layout a §7.6.2 meet needs from a sector: the two flanking
    /// edge directions and, per the three-case classification, the in-plane
    /// perpendicular each offset line is built along, plus the in-between faces
    /// used by the skew face-drop. `None` for a degenerate (zero-length) flank.
    pub(super) fn sector_layout(
        &self,
        vertex: VertexKey,
        vertex_position: Vec3,
        sector: &Sector,
    ) -> Option<SectorLayout> {
        if sector.bounding_edges.len() != 2 {
            // A junction's first sector can transiently carry a single bounding
            // edge only on a malformed open fan; the §7.6.1 closed junction always
            // has two. Fall back to the vertex position rather than guess.
            return None;
        }
        let edge_first = sector.bounding_edges[0];
        let edge_second = sector.bounding_edges[1];
        let dir_first = self.edge_direction_from(edge_first, vertex, vertex_position)?;
        let dir_second = self.edge_direction_from(edge_second, vertex, vertex_position)?;
        let far_first = self.edge_far_position(edge_first, vertex);
        let far_second = self.edge_far_position(edge_second, vertex);
        // The offset lines are built from the **raw**, full-length flank vectors, not the
        // unit ones (§7.6.2 "The arithmetic of the two-line solve"). Unitising them first
        // is the natural-looking simplification and it moves the boundvert by a canonical
        // quantum on 4.7% of shallow planar sectors. The unit `dir_*` above stay in use for
        // the quantities that are genuinely about *direction* — the ladder's angle dispatch,
        // the sector's edge-span classification and the face-drop wedges.
        let raw_first = far_first - vertex_position;
        let raw_second = far_second - vertex_position;

        // The sector's face run, in disk order: f0 (just past g_k) … f_last (just
        // before g_(k+1)). `corners` is `[(g_k,f0),(leg1,f1),…,(g_(k+1),f_last)]`,
        // so deduping consecutive faces recovers the run, and the unbeveled edges
        // strictly between the two flanking beveled edges are the in-between legs.
        let mut faces: Vec<FaceKey> = Vec::new();
        for &(_, face) in &sector.corners {
            if faces.last() != Some(&face) {
                faces.push(face);
            }
        }
        let mut between_edges: Vec<EdgeKey> = Vec::new();
        for &(edge, _) in &sector.corners {
            if edge != edge_first && edge != edge_second && !between_edges.contains(&edge) {
                between_edges.push(edge);
            }
        }

        let face_first = *faces.first()?;
        let face_last = *faces.last()?;
        // Newell over `face_first` feeds two readings a few lines apart — the offset-line
        // sign reference (`+Z`-rescued) and the ladder's source-1 normal — and it
        // allocates, so it is taken once.
        let newell_first = self.face_newell_normal(face_first);
        // The walk crosses the sector's first face entering through `g_k`; both branches
        // below want that corner — case 3 for `g_k`'s own leg, cases 1/2 for the
        // walk-sense sign reference the role-signed perpendiculars are built about.
        let walk_first = self.walk_corner(face_first, vertex, edge_first);

        // Classify the sector into the three §7.6.2 cases by the in-between edges:
        // out-of-plane (silhouette) legs are the ones whose flanking faces are not
        // coplanar. Cases 1 and 2 (no legs, or all legs in-plane) share one sector
        // plane; case 3 (≥1 out-of-plane leg) gives each line its own local normal.
        let has_out_of_plane = between_edges.iter().enumerate().any(|(index, _)| {
            let before = faces[index];
            let after = faces[index + 1];
            let dot = self
                .face_plane_normal(before)
                .dot(self.face_plane_normal(after));
            dot.abs() < 1.0 - MEET_PLANAR_TOLERANCE
        });

        // Each base-point perpendicular is in the sector plane and perpendicular to
        // its edge, oriented so the offset pushes *into* the sector. Cases 1/2 (one
        // shared plane) and case 3 (two local planes) build them differently.
        let (normal_first, normal_second, perp_first, perp_second) = if has_out_of_plane {
            // Case 3: two different local planes. Each line's normal is taken from
            // the corner at its own end of the sector — the cross of that flanking
            // beveled edge's direction with the nearest in-between leg's direction,
            // oriented by the adjacent face (walk-sense). The perpendicular is then
            // the pure §7.6.2 right-of-`g_k` / left-of-`g_(k+1)` rotation about that
            // line's own walk-sense normal — `normal.cross(dir)` for the forward
            // flank, `dir.cross(normal)` for the backward one. Because the normals
            // are already walk-sense oriented, the cross pushes each line *into* the
            // sector with a sign-consistent handedness at convex and reflex corners
            // alike, with no face-centroid reference (a reflex face's centroid sits
            // on the convex complement of the sector, so the centroid heuristic
            // would sign the push backwards there).
            //
            // `g_k`'s leg is the disk edge *next* after it — which is the other edge of
            // `face_first` at `v`, not simply the sector's first recorded in-between
            // edge: at an open fan's gap-crossing sector the disk walk leaves the ring
            // through a boundary leg that no later corner re-reports, so the recorded
            // list starts one edge late and would pair `g_k` with an edge from a
            // different face's corner entirely.
            //
            // A flank that runs **collinear** with its own leg — straight through `v`,
            // as an unbeveled spoke continuing a beveled one does — crosses to exactly
            // zero. That zero is the answer, not a hole to be plugged: it zeroes the
            // perpendicular, so that line passes through `v` un-offset and the meet
            // lands where the *other* flank's offset line crosses the flank's own line.
            // No ladder guards it, because the ladder tests the two flanks against each
            // other and they can be square while one is collinear with the leg between
            // them. See `orient_normal`.
            //
            // A leg with **no direction at all** — a degenerate leg whose far endpoint sits
            // on `v` — is the limiting case of that same collinearity, and gets the same
            // zero: flank × leg is zero either way, and §7.6.2's case-3 recipe has no
            // branch for a missing leg, so what answers is the shared zero-returning floor.
            // Handing over the sector face's normal instead is the one thing §7.6.2 (d)
            // rules out — the face enters the recipe as a *sign* and nothing else — and it
            // would push that base point a full `offset` off the flank's own line. On a
            // degenerate face it would also substitute `face_plane_normal`'s synthetic `+Z`
            // axis, first on the guide's list of what a rescue must not invent.
            let first_leg = walk_first.and_then(|(next_edge, _)| {
                self.edge_direction_from(next_edge, vertex, vertex_position)
            });
            let last_leg = between_edges
                .last()
                .and_then(|&edge| self.edge_direction_from(edge, vertex, vertex_position));
            let normal_first = first_leg.map_or(Vec3::ZERO, |leg| {
                orient_normal(dir_first.cross(leg), self.face_plane_normal(face_first))
            });
            let normal_second = last_leg.map_or(Vec3::ZERO, |leg| {
                orient_normal(dir_second.cross(leg), self.face_plane_normal(face_last))
            });
            let perp_first = normalise_at_direction_floor(normal_first.cross(raw_first));
            let perp_second = normalise_at_direction_floor(raw_second.cross(normal_second));
            (normal_first, normal_second, perp_first, perp_second)
        } else {
            // Cases 1 and 2: one shared sector plane. Its normal is the cross product
            // of the two **raw** flanking beveled-edge vectors at `v` (the plane those
            // edges span), unitised, then sign-fixed against the sector's first face
            // read in the walk's sense — from which nothing but the sign is taken.
            //
            // The span is the answer **unconditionally**, on a planar sector as much as
            // on a bent one. There is no coplanarity test here, by design: on a flat
            // sector the unit span and the face's own unit normal are the same
            // *direction* and different `f32` vectors — the face's is summed over
            // vertices the two flanks never touch — and substituting one for the other
            // moves the in-face perpendicular, hence the base points, hence the
            // boundvert, by an ulp. That ulp survives into the multi-segment profile
            // grid and the inner miter. It is also nearly invisible on an axis-aligned
            // sector plane (both spellings collapse to a signed unit axis), which is
            // why the gate that used to sit here read as harmless.
            //
            // The one face-normal substitution §7.6.2 does license in this branch is the
            // shallow **direction-angle** exception (`θ_dir < SMALL_ANG = 10°`, where the
            // short cross swings under perturbation) — a predicate on the flanks' angle,
            // not on the span's alignment with anything. It is **not** implemented here.
            let face_normal = newell_first.unwrap_or(Vec3::Z);
            let reference = walk_first.map_or(face_normal, |(_, walk_normal)| walk_normal);
            // Cross order is free: swapping it negates every component exactly, and the
            // sign fix undoes the flip identically.
            let edge_span = raw_first.cross(raw_second);
            // A zero-length span means an (anti-)parallel flank pair. The ladder claims
            // that geometry in `solve_meet_point`, not here, so the sign fix's zero branch
            // is **reached but inert**: the layout is still built, the zero is kept, and no
            // offset line is ever built from it.
            let normal = orient_normal(edge_span, reference);
            // Each perpendicular is the pure right-of-`g_k` / left-of-`g_(k+1)`
            // rotation about that normal: its sign comes from the edge's **role** in
            // the sector — forward flank versus backward flank — and never from the
            // face's interior. Orienting them toward a face **centroid** instead is a
            // different rule that agrees on a convex sector and inverts wherever the
            // sector's face is reflex at `v` (the centroid then sits on the sector's
            // convex complement), reflecting the meet through `v` at `2 × offset`.
            let perp_first = normalise_at_direction_floor(normal.cross(raw_first));
            let perp_second = normalise_at_direction_floor(raw_second.cross(normal));
            (normal, normal, perp_first, perp_second)
        };

        // The §7.6.2 face-drop walk, in disk order: the walk edges are `g_k` then each
        // in-between leg, and `faces[i]` is the face just forward of walk edge `i`. Each
        // candidate carries the angular wedge that face occupies at `v` — bounded by the
        // face's *own* two edges at `v`, which is what the walk edge's disk-successor is
        // for an interior face, and the only defensible reading for the face that flanks
        // an open fan's gap (where the sector's edge list skips the boundary leg the walk
        // never visits) — plus that face's normal read in the walk's sense.
        let walk_edges = std::iter::once(edge_first).chain(between_edges.iter().copied());
        let mut drop_candidates = Vec::with_capacity(faces.len());
        for (&face, walk_edge) in faces.iter().zip(walk_edges) {
            let (Some(dir_walk), Some((next_edge, normal))) = (
                self.edge_direction_from(walk_edge, vertex, vertex_position),
                self.walk_corner(face, vertex, walk_edge),
            ) else {
                continue;
            };
            let Some(dir_next) = self.edge_direction_from(next_edge, vertex, vertex_position)
            else {
                continue;
            };
            drop_candidates.push(DropCandidate {
                dir_walk,
                dir_next,
                normal,
            });
        }

        // §7.6.2's three-source precedence for the ladder's plane normal, each source
        // tried only when the previous one is unavailable:
        //
        //   1. the sector's own nominated face — the face just forward of the first
        //      flanking edge — taken **directly**: as the face hands it over, *not*
        //      re-normalised, *not* sign-fixed, and *not* rescued when it is zero. All
        //      three omissions are load-bearing, and the third is why a zero-area face's
        //      zero normal flows straight through and collapses the straight-through push
        //      onto `v` — substituting an axis there pushes a whole `offset` off-axis
        //      instead. Available for cases 1 and 2, where one face fills the sector.
        //
        //      "Not sign-fixed" covers the walk's sense too, and deliberately: unlike the
        //      offset-line perpendiculars above, whose signs come from the two flanks'
        //      *roles*, this normal is taken raw. A ring wound against the faces would
        //      have to be repaired by re-deriving those roles, never by a sign check on
        //      the normal.
        //   2. otherwise (case 3, no single face fills the sector) the mean of the
        //      forward-face normals over the same walk the face-drop uses. The divisor is
        //      inert — only the sum's direction survives the cross and normalisation —
        //      so the contributions are summed unweighted and left un-averaged.
        //   3. otherwise the vertex normal, when no face in the span contributes one.
        let sector_normal = if has_out_of_plane {
            let mut accumulated = Vec3::ZERO;
            let mut contributors = 0_usize;
            for &face in &faces {
                if let Some(normal) = self.face_newell_normal(face) {
                    accumulated += normal;
                    contributors += 1;
                }
            }
            if contributors == 0 {
                self.vertex_normal(vertex).normalize_or_zero()
            } else {
                accumulated
            }
        } else {
            newell_first.unwrap_or(Vec3::ZERO)
        };

        Some(SectorLayout {
            dir_first,
            dir_second,
            far_first,
            far_second,
            perp_first,
            perp_second,
            normal_first,
            normal_second,
            sector_normal,
            drop_candidates,
            non_planar: has_out_of_plane,
        })
    }

    /// The disk walk crosses `face` from `walk_edge` to the face's **other** edge at
    /// `vertex`; return that edge together with `face`'s normal **oriented to the walk's
    /// sense**. A face's interior at `vertex` sweeps positively (about its own normal)
    /// from the corner's outgoing edge to its incoming one, so a walk that enters by the
    /// incoming edge crosses the corner the other way round and must read the normal
    /// flipped. Both the §7.6.2 sign-fix and the in-wedge signed sweeps are measured in
    /// the walk's sense, and the corner ring's chaining direction is not canonical — so
    /// taking the raw normal makes them depend on mesh build order.
    /// `vertex`'s incident edges in **disk-walk order** (BEVEL.md §7.6.2, *"The sense of
    /// the disk walk"*): each forward face is entered through its winding-outgoing edge
    /// at `vertex` and left along its winding-incoming one, so the sequence runs
    /// `e_0 → f_0 → e_1 → f_1 → …` with `e_i` the winding-outgoing edge of `f_i`.
    ///
    /// The stored disk cycle is linked in insertion order and carries **no rotational
    /// meaning**, so it answers "which edges" and nothing more. Every *rotational* read —
    /// the cross weld's disk-predecessor donor (§8.3.0.3), "opposite in the disk" — must
    /// come from this walk instead. Faces are the only source of the winding, so a wire
    /// edge never appears; callers that care about valence must ask the disk separately.
    ///
    /// An open fan is rewound to its boundary edge first, so the walk starts at the fan's
    /// true beginning rather than mid-fan. On a closed fan every starting edge is
    /// equivalent (the order is cyclic) and the rewind simply lands on one of them.
    pub(super) fn walk_disk_edges(&self, vertex: VertexKey) -> Vec<EdgeKey> {
        // Per incident face, the step the walk takes across it: in by the winding-
        // outgoing edge at `vertex`, out by the winding-incoming one.
        let mut steps: Vec<(EdgeKey, EdgeKey)> = Vec::new();
        for face in self.vert_face_keys(vertex) {
            if let Some((incoming, outgoing)) = self.face_corner_edges(face, vertex) {
                steps.push((outgoing, incoming));
            }
        }
        let Some(&(first, _)) = steps.first() else {
            return Vec::new();
        };

        let mut start = first;
        for _ in 0..steps.len() {
            match steps.iter().find(|&&(_, leaves)| leaves == start) {
                Some(&(enters, _)) if enters != first => start = enters,
                _ => break,
            }
        }

        let mut order = vec![start];
        let mut current = start;
        while let Some(&(_, leaves)) = steps.iter().find(|&&(enters, _)| enters == current) {
            if leaves == start || order.len() > steps.len() {
                break;
            }
            order.push(leaves);
            current = leaves;
        }
        order
    }

    pub(super) fn walk_corner(
        &self,
        face: FaceKey,
        vertex: VertexKey,
        walk_edge: EdgeKey,
    ) -> Option<(EdgeKey, Vec3)> {
        let (incoming, outgoing) = self.face_corner_edges(face, vertex)?;
        let normal = self.face_plane_normal(face);
        if walk_edge == outgoing {
            Some((incoming, normal))
        } else if walk_edge == incoming {
            Some((outgoing, -normal))
        } else {
            None
        }
    }

    /// §7.6.2 face-drop: a skew-line midpoint floats off the surface, so project it
    /// onto the first in-walk sector face whose plane projection lands inside that
    /// face's angular wedge at `v` (the in-wedge predicate). If no face projection
    /// is in-wedge, the midpoint stands.
    ///
    /// Whether the drop runs at all — §7.6.2's "out-of-plane in-between edges *and*
    /// both flanking offsets `> 0`" — is the caller's gate, stated once at the single
    /// call site so the two halves of that conjunction cannot drift apart.
    pub(super) fn drop_onto_sector_face(
        &self,
        vertex_position: Vec3,
        layout: &SectorLayout,
        midpoint: Vec3,
    ) -> Vec3 {
        // Walk the sector's edges in disk order starting at the flanking beveled edge
        // `g_k` itself (not the first in-between leg), advancing up to — but not
        // including — `g_(k+1)`, and take the face just forward of each. The projection
        // plane is anchored at `v` (not the face's own supporting plane), oriented by
        // the face normal; on a warped sector face those two planes differ by up to the
        // offset itself.
        //
        // The in-wedge bounds are the walk edge being tested and the face's other edge
        // at `v` — the pair that actually bounds that face's wedge — with the *raw*
        // edge directions, not their projections into the drop plane: `point_in_wedge`
        // orients its signed sweeps by the (walk-sense) face normal, so the small
        // out-of-plane tilt of the bounds on a warped face is already absorbed by the
        // angular measure, and projecting first would shift the near-boundary two-petal
        // verdict off the wedge boundary its midpoint hugs.
        for candidate in &layout.drop_candidates {
            let normal = candidate.normal;
            let projected = midpoint - normal * (midpoint - vertex_position).dot(normal);
            let to_projected = projected - vertex_position;
            if to_projected.length() <= f32::EPSILON {
                continue;
            }
            if point_in_wedge(
                candidate.dir_walk,
                candidate.dir_next,
                to_projected.normalize(),
                normal,
            ) {
                return projected;
            }
        }
        midpoint
    }
}

/// An offset line (BEVEL.md §7.6.2), carried as its base point and the raw step to its
/// second point. §7.6.2 states the line as **two points**, and `solve_two_lines` forms
/// that second point and re-derives the direction from it — so `direction` here is the
/// step, not a pre-normalised heading, and its magnitude and sign are both observable.
pub(super) struct Line {
    pub(super) point: Vec3,
    pub(super) direction: Vec3,
}

/// The outcome of intersecting two offset lines (BEVEL.md §7.6.2 "Intersecting
/// the two offset lines").
pub(super) enum LineMeet {
    /// The lines cross — the intersection point.
    Crossing(Vec3),
    /// The lines coincide — caller picks a defined on-line point.
    Collinear,
    /// The lines are skew. §7.6.2 gives the skew outcome **two** readings and the
    /// caller picks by the face-drop's own gate, not by the classification: with the
    /// drop running it consumes `midpoint` as a provisional meet, and with the drop
    /// not running the answer is `on_first`, the closest-approach point on the first
    /// line — "no midpoint, no projection".
    Skew { on_first: Vec3, midpoint: Vec3 },
}

/// The per-sector geometry a §7.6.2 meet consumes (built by `sector_layout`): the
/// two flanking edge directions and their offset-line perpendiculars/normals, plus
/// the in-between faces/edges the skew face-drop walks.
pub(super) struct SectorLayout {
    pub(super) dir_first: Vec3,
    pub(super) dir_second: Vec3,
    /// The two flanking edges' far endpoints, in the pair's own order. The ladder's
    /// straight-through push sums the **raw** directions — which telescope to
    /// `far_second − far_first` — and its doubled-back slide runs to `far_second`.
    pub(super) far_first: Vec3,
    pub(super) far_second: Vec3,
    pub(super) perp_first: Vec3,
    pub(super) perp_second: Vec3,
    /// The plane normal each offset line's perpendicular is rotated about — **zero or
    /// unit, never anything between**. Above the direction floor `orient_normal` hands
    /// over a unit vector — a Newell normal enters only as the sign reference *inside*
    /// it, never as the field — and only its
    /// *direction* is read: the sole consumer crosses it with an edge direction and
    /// normalises, so a raw `sin θ`-magnitude cross would go unnoticed there. At or below
    /// the floor `orient_normal` hands over the **zero vector**, and that zero *is* the
    /// degenerate answer rather than a missing one — it zeroes the perpendicular, so that
    /// line runs through `v` un-offset. Same convention as `sector_normal`, arrived at
    /// differently: there the zero comes from a zero-area face read directly, here from a
    /// zero cross of a flank against its adjacent leg.
    pub(super) normal_first: Vec3,
    pub(super) normal_second: Vec3,
    /// §7.6.2's plane normal for the **near-collinear ladder** only — a different
    /// quantity from the sign-fixed `normal_first`/`normal_second` the offset lines are
    /// built about, resolved by the guide's three-source precedence and, at source 1,
    /// taken *directly* from the sector's own face. A zero-area face therefore hands
    /// over the zero vector instead of a substitute.
    pub(super) sector_normal: Vec3,
    /// The §7.6.2 face-drop walk, in disk order (one entry per sector face, starting
    /// at the face just forward of `g_k`).
    pub(super) drop_candidates: Vec<DropCandidate>,
    /// Whether this is a §7.6.2 **case 3** (non-planar) sector: at least one
    /// in-between leg is out-of-plane, so each offset line carries its own local
    /// normal. The face-drop step runs only for these sectors.
    pub(super) non_planar: bool,
}

/// One stop of the §7.6.2 face-drop walk: a sector face, the angular wedge it occupies
/// at the junction vertex (swept from the walk edge `dir_walk` to the face's other edge
/// `dir_next`), and its normal oriented to the walk's sense so the in-wedge sweeps and
/// the wedge itself are measured with the same handedness.
pub(super) struct DropCandidate {
    pub(super) dir_walk: Vec3,
    pub(super) dir_next: Vec3,
    pub(super) normal: Vec3,
}

/// Orient `candidate` to agree with `reference` (flip when they oppose), giving a
/// consistent local sector normal for a case-3 offset-line perpendicular.
///
/// The unitisation is §7.6.2's shared `normalise` — one reciprocal scaled into three
/// components, never three divisions, on [`SLIDE_DIRECTION_FLOOR_SQ`]. This normal is
/// what the offset-line perpendicular is rotated about, so its last bits reach the base
/// points and land on the guide's normalisation axis.
///
/// The operator is **not** yet uniform on that floor, and the exception is worth knowing
/// about before reading a divergence as a bug here. §7.5.1's terminal regime-3 normal, in
/// `terminal_meet_placement`, still hand-rolls this same construct on the older spelling —
/// `cross.length() <= f32::EPSILON`, plus its own copy of the sign fix. That is a floor
/// some eleven orders of magnitude higher, stated on the **unsquared** length and with the
/// `<=` sense rather than `>`. Worse, the directions it crosses are **raw**, so unlike
/// every `normalise` here the guard is **scale-dependent**: at a world scale of `~1e-4` an
/// ordinary right-angle terminal produces a cross of `~1e-8`, trips it, and diverts a
/// perfectly well-conditioned terminal into `degenerate_meet_placement`. That site's own
/// comment — "deep inside regime 1's band, so the dispatch above has already taken it" —
/// holds at unit scale and fails below it. Unifying it is its own change.
///
/// **The reference contributes a sign and nothing else** — there is no rescue on a
/// degenerate candidate. §7.6.2's case-3 recipe states the local normal as the cross of a
/// flank with its adjacent in-between leg, *oriented by* the face at that end of the
/// sector, and stops there — the face enters as an orientation and never as a value. The
/// only path from that normal into a coordinate is the base-point construction, which
/// takes `normalise(direction × normal)`, and the shared zero-returning `normalise` is
/// what absorbs a degenerate cross. (The sign fix's name is this file's shorthand; the
/// guide spells it as prose.) So substituting a synthetic normal (an axis, the vertex
/// normal, a neighbouring face's) pushes the offset line a whole `offset` off-axis where
/// the reference does not move the base point at all — and note the "no rescue" the guide
/// spells out for the near-collinear ladder's own plane normal is a *different* construct
/// that happens to reach the same conclusion.
/// A zero normal is *load-bearing*: it zeroes the perpendicular,
/// which collapses that line's sideways push onto `v`, so the line runs through `v`
/// along its own flank. That is the answer at a case-3 sector whose flanking beveled
/// edge runs **collinear** with its adjacent in-between leg — straight through `v` —
/// where the flank × leg cross is exactly zero and no §7.6.2 ladder guards it (the
/// ladder tests the two *flanks*' direction angle, and those can be square to each
/// other while one of them is collinear with the leg between them).
pub(super) fn orient_normal(candidate: Vec3, reference: Vec3) -> Vec3 {
    let unit = normalise_at_direction_floor(candidate);
    if unit.dot(reference) < 0.0 {
        -unit
    } else {
        unit
    }
}

/// The three outcomes of the two-line solve, in the shape the solve actually produces
/// them (BEVEL.md §7.6.2 "The arithmetic of the two-line solve").
pub(super) enum LineSolve {
    /// `direction_cross` is **exactly** zero: no isolated crossing exists.
    Parallel,
    /// The lines meet — a parameter along the **first** line, so the answer lies on
    /// that line rather than between two near-coincident readings.
    Crossing(Vec3),
    /// The lines pass without meeting: the closest point on the first line, plus the
    /// common-perpendicular offset that separates the two closest points. The point on
    /// the second line is `on_first − gap`, and their midpoint is `on_first − gap/2`.
    Skew { on_first: Vec3, gap: Vec3 },
}

/// The crossing of two coplanar lines as a parameter along the **first** one, in
/// BEVEL.md §7.6.2's own names:
///
/// ```text
///    along  =  dot(between_bases × dir_second, direction_cross)
///              ───────────────────────────────────────────────
///                 dot(direction_cross, direction_cross)
///
///    meet   =  base_first + dir_first · along
/// ```
///
/// This spelling is not interchangeable with a `2 × 2` Gram solve, nor with averaging a
/// closest point taken on each line. All three are the same value in exact arithmetic
/// and three different values in `f32`; on a near-parallel pair the placement of the
/// cancellation is what decides the last bit, and the last bit is a canonical quantum.
/// `direction_cross` is the caller's own `dir_first × dir_second`, passed in so the
/// crossing branch does not form it twice; it must be that product and not a re-scaled
/// or re-signed stand-in.
pub(super) fn crossing_along_first(
    base_first: Vec3,
    dir_first: Vec3,
    between_bases: Vec3,
    dir_second: Vec3,
    direction_cross: Vec3,
) -> Vec3 {
    let along =
        between_bases.cross(dir_second).dot(direction_cross) / direction_cross.dot(direction_cross);
    base_first + dir_first * along
}

/// The two-line solve of BEVEL.md §7.6.2, in the guide's stated arithmetic.
///
/// Each line is carried as **two points** — its base and a tip one raw direction along
/// it — and the direction the solve consumes is re-derived as their difference. That
/// round trip is not the identity in `f32`, and skipping it (carrying point-plus-
/// direction straight through) moves the meet by a canonical quantum on about one
/// shallow station in a thousand. The caller therefore owes this routine a **raw**,
/// full-length direction with the sign the §7.6.2 convention gives it — into the
/// junction vertex for the first line, out of it for the second — because the tip is a
/// rounded sum and stepping the other way rounds differently.
///
/// Both classification tests are the guide's: an **exact** `== 0` on
/// `|direction_cross|²` for the parallel bail (a small epsilon there would divert
/// nearly-but-not-quite-parallel lines into the degenerate answer, which is a far
/// larger error than the rounding this arithmetic is about), and an **absolute** bound
/// on the **un-normalised** `coplanarity` triple product for crossing-versus-skew. The
/// latter is deliberately not a distance or an angle: it scales with both directions'
/// lengths.
pub(super) fn solve_two_lines(first: &Line, second: &Line) -> LineSolve {
    let base_first = first.point;
    let tip_first = base_first + first.direction;
    let base_second = second.point;
    let tip_second = base_second + second.direction;

    let dir_first = tip_first - base_first;
    let dir_second = tip_second - base_second;
    let direction_cross = dir_first.cross(dir_second);
    if direction_cross.dot(direction_cross) == 0.0 {
        return LineSolve::Parallel;
    }

    let between_bases = base_second - base_first;
    let coplanarity = between_bases.dot(direction_cross);
    if coplanarity.abs() <= LINE_COPLANARITY_TOLERANCE {
        return LineSolve::Crossing(crossing_along_first(
            base_first,
            dir_first,
            between_bases,
            dir_second,
            direction_cross,
        ));
    }

    // Skew: slide the whole second line along the common perpendicular until the two are
    // coplanar, re-derive its two points (and hence its direction) there, run the same
    // crossing formula, and carry the answer back by the same offset.
    let gap = direction_cross
        * ((base_first - base_second).dot(direction_cross) / direction_cross.dot(direction_cross));
    let moved_base = base_second + gap;
    let moved_direction = (tip_second + gap) - moved_base;
    let moved_cross = dir_first.cross(moved_direction);
    if moved_cross.dot(moved_cross) == 0.0 {
        return LineSolve::Parallel;
    }
    let on_first = crossing_along_first(
        base_first,
        dir_first,
        moved_base - base_first,
        moved_direction,
        moved_cross,
    );
    LineSolve::Skew { on_first, gap }
}

/// The closest-approach **pair** of two lines: one point on each, in the order the
/// lines were passed. `None` when the pair is exactly parallel and no unique pair
/// exists.
///
/// Callers pick which of the two they want; there is no project-wide convention.
/// §7.5.1's terminal leg meet takes the point on the *first* line, §7.6.2's face-drop
/// path takes the midpoint (see `intersect_lines`). A crossing pair meets, so the two
/// points are the one meet — which is the same slot the skew outcome fills with its
/// point on the first line, not a different answer.
pub(super) fn closest_approach(first: &Line, second: &Line) -> Option<(Vec3, Vec3)> {
    match solve_two_lines(first, second) {
        LineSolve::Parallel => None,
        LineSolve::Crossing(meet) => Some((meet, meet)),
        LineSolve::Skew { on_first, gap } => Some((on_first, on_first - gap)),
    }
}

/// Intersect two lines in 3-D (BEVEL.md §7.6.2). Returns the crossing point when they
/// meet, `Collinear` when no isolated crossing exists, and **both** of §7.6.2's skew
/// readings when they pass without meeting — the point on the first line and the
/// closest-points midpoint — because which one is the answer is the *caller's* gate
/// (whether the face-drop runs), not a property of the solve.
pub(super) fn intersect_lines(first: &Line, second: &Line) -> LineMeet {
    match solve_two_lines(first, second) {
        LineSolve::Parallel => LineMeet::Collinear,
        LineSolve::Crossing(meet) => LineMeet::Crossing(meet),
        LineSolve::Skew { on_first, gap } => LineMeet::Skew {
            on_first,
            midpoint: on_first - gap * 0.5,
        },
    }
}

/// The §7.6.2 in-wedge predicate: is `dir_projected` inside the angular wedge that
/// runs from `dir_one` to `dir_two` in `normal`'s winding sense? Each unsigned
/// angle in `[0, π]` is reflexed to `2π − angle` when its directions' cross points
/// against `normal`, so reflex wedges are measured correctly. In-wedge when the
/// `dir_one → dir_projected` sweep does not exceed the `dir_one → dir_two` sweep.
pub(super) fn point_in_wedge(
    dir_one: Vec3,
    dir_two: Vec3,
    dir_projected: Vec3,
    normal: Vec3,
) -> bool {
    let swept = |from: Vec3, to: Vec3| -> f32 {
        let angle = from.dot(to).clamp(-1.0, 1.0).acos();
        if from.cross(to).dot(normal) < 0.0 {
            std::f32::consts::TAU - angle
        } else {
            angle
        }
    };
    let wedge = swept(dir_one, dir_two);
    let test = swept(dir_one, dir_projected);
    // Bounded-overshoot accept (§7.6.2): wedge − drop > −EPS_ANG.
    test <= wedge + WEDGE_OVERSHOOT_TOLERANCE
}

/// Coplanarity tolerance (on the dot of two face normals) for the §7.6.2 sector
/// classification: an in-between leg is out-of-plane (silhouette) when its two
/// flanking face normals differ from ±1 by more than this.
pub(super) const MEET_PLANAR_TOLERANCE: f32 = 1.0e-4;

/// The §7.6.2 ladder's band, shared by the junction sector meet and §7.5.1's regime 1:
/// within `EPS_ANG` of `0` (the two flanks run straight through the vertex) or of `π`
/// (they double back on each other). Neither offset line is well defined there, so no
/// line solve runs at all. Both tests are strict, and the two bands are disjoint.
///
/// The angle handed in is always [`unit_chord_angle_between`] of the pair bound as
/// §7.5.1 binds it — first member taken *into* the shared vertex, second taken *out* of
/// it (see [`terminal_meet_directions`]). §7.6.2's ladder writes it as
/// `θ_dir = angle_between(dir_first, dir_second)`, and §8.6.3.1 scopes that name to one
/// operator-wide measure: *"Carry **one** `angle_between`."* A second spelling here is
/// the failure §8.2.2.1 names by construction.
pub(super) fn is_degenerate_direction_angle(direction_angle: f32) -> bool {
    direction_angle < MEET_DIRECTION_EPS_ANGLE
        || direction_angle > std::f32::consts::PI - MEET_DIRECTION_EPS_ANGLE
}

/// §7.6.2's "parallel and anti-parallel special cases" — the near-collinear ladder — as
/// **one** closed form serving both the junction sector meet and §7.5.1's regime 1. The
/// two halves differ only in the pair they are handed, so they must not differ in the
/// arithmetic; a second expression of this is how the two drift apart.
///
/// `far_first` / `far_second` are the two flanking edges' far endpoints in the pair's own
/// order, and `offset` is the larger of the two sides' offsets — at a terminal the leg
/// carries `0`, so it is always the beveled edge's.
///
/// The band is split at `π/2` because that is where the two closed forms swap which
/// component survives: below it the directions still enclose a wedge whose bisector is
/// well defined and the placement is purely perpendicular, above it they lie back on one
/// another and the placement is purely along the common line. Nothing in `[EPS_ANG,
/// π − EPS_ANG]` reaches here, so any split strictly inside that gap picks out the same
/// two halves; `π/2` is simply the one that names them by which end of `[0, π]` the
/// angle is nearer.
pub(super) fn degenerate_meet_placement(
    vertex_position: Vec3,
    far_first: Vec3,
    far_second: Vec3,
    direction_angle: f32,
    sector_normal: Vec3,
    offset: f32,
) -> Vec3 {
    if direction_angle > std::f32::consts::FRAC_PI_2 {
        // Doubled back: the two edges leave the vertex the same way and lie back on one
        // another, so the chamfer runs into a zero-area sliver with no wedge to bisect.
        // The meet sits on the common line, slid out along the *second* edge — there is
        // no perpendicular component to take.
        //
        // This is a plain on-edge slide, so it goes through §4.1's shared primitive and
        // inherits the whole of it: the strict overshoot trigger against the *second*
        // edge's own length, the absolute `5e-5` landing, the sub-gap reflection and the
        // land-on-the-vertex regime at the direction floor. It is exempt from §7.5.2's
        // span clamp (which is armed on a meet, and no meet ran here) but not from this
        // one.
        return slide_along_edge(vertex_position, far_second, offset);
    }

    straight_through_push(
        vertex_position,
        far_second - far_first,
        sector_normal,
        direction_angle,
        offset,
    )
}

/// The straight-through half of the §7.6.2 ladder: both offset lines get pushed the same
/// way and land on top of each other, so the meet goes square to the common direction,
/// `offset / cos(θ_dir/2)` out.
///
/// `common_raw` is the **raw** sum of the pair's two bound directions — the first taken
/// *into* the vertex, the second *out* of it — which telescopes to `far_second −
/// far_first`. Normalising the two before adding is the natural-looking simplification
/// and it is wrong: the raw sum leans toward the longer flank, and on flanks of length
/// `2` and `3.00042` less than a degree apart that lean is 83 quanta of the canonical
/// `1e-5` grid at `offset = 0.5`, a thousand at `offset = 6`.
///
/// `sector_normal` is used exactly as its three sources hand it over — not re-normalised,
/// not sign-fixed, not rescued — and the cross order `common × normal` alone points the
/// push *into* the sector. Steering it by a face-centroid reference instead point-reflects
/// the meet through the vertex, `2d` away, wherever the two disagree: a reflex sector
/// face's centroid sits on the sector's convex complement. Nor is there any degenerate
/// rescue — what absorbs a degenerate input is the zero-returning normalisation itself, so
/// a zero-area sector face's zero normal collapses the push exactly onto the vertex, where
/// a substituted axis would push a whole `offset` off-axis.
///
/// The distance is the crossing case's `t / sin(θ_wedge/2)` written in the supplementary
/// angle, so the branch joins the general solve in the limit rather than stepping.
pub(super) fn straight_through_push(
    vertex_position: Vec3,
    common_raw: Vec3,
    sector_normal: Vec3,
    direction_angle: f32,
    offset: f32,
) -> Vec3 {
    let perpendicular = normalise_at_direction_floor(common_raw.cross(sector_normal));
    vertex_position + perpendicular * (offset / (direction_angle * 0.5).cos())
}

/// `EPS_ANG` — the near-degenerate band half-width (2°, in radians). A direction angle
/// within this of `0` or of `π` leaves no well-defined meet plane at all, and the ladder
/// gives that band its own closed forms (push out along the common perpendicular / slide
/// along the common line — `degenerate_meet_placement`).
///
/// §7.6.2 states this constant for **both** halves of the ladder, so the junction sector
/// meet and §7.5.1's regime 1 read the one value. Stating the junction's band as a
/// tolerance in *cosine* units instead is the same rule at `0.81°`, which hands the
/// `0.81°–2°` window to the general two-line solve — and there the two offset lines are
/// near-coincident, so the solve extrapolates the boundvert tens of units out on a
/// unit-scale fan.
///
/// The bound is also what keeps §7.5.1's regimes 2 and 3 reachable only on the bands they
/// describe: without it, `< SMALL_ANG` would swallow the whole near-collinear band into
/// regime 2, and the `> π − EPS_ANG` band would reach regime 3's `edge_out × leg_out` —
/// the short, direction-unstable cross product regime 2 already exists to avoid.
pub(super) const MEET_DIRECTION_EPS_ANGLE: f32 = 0.034_906_6;

/// §7.6.2's `EPS_COPLANAR` — the bound that separates a crossing from a skew pair,
/// stated on the **un-normalised** `coplanarity` triple product
/// `dot(between_bases, direction_cross)` and therefore **absolute**: it scales like
/// `|dir_first|·|dir_second|·sin(θ_dir)·gap`, so the *gap* it waves through as
/// "crossing" varies with the geometry — a sector just outside the `2°` band admits
/// some twenty-five times the gap a right-angled one does. That is deliberate.
/// Re-expressing it as a normalised gap is a different predicate that flips the branch
/// on sectors whose flanking edges are long or whose direction angle is shallow.
///
/// It is the **only** tolerance in the whole solve: the parallel bail above it is an
/// exact zero.
pub(super) const LINE_COPLANARITY_TOLERANCE: f32 = 1.0e-6;

/// Angular slack (radians) for the §7.6.2 in-wedge face-drop predicate. Per the
/// bounded-overshoot rule (§7.6.2): accept a face-drop when its `d1→dp` sweep
/// overshoots the `d1→d2` wedge by strictly less than `EPS_ANG = 2°`. This is a
/// genuine geometric slack, not a round-off guard — wide enough to admit a first
/// face whose drop overshoots its wedge by ~1.3° (the two-petal marginal accept),
/// yet still reject one that overshoots by tens of degrees (the three-petal case).
pub(super) const WEDGE_OVERSHOOT_TOLERANCE: f32 = 0.034_906_6;
