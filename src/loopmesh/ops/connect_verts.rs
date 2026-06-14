use std::collections::HashSet;

use glam::{Vec2, Vec3};

use crate::loopmesh::*;
use crate::triangulate::axis_dominant_basis_negated;

impl LoopMesh {
    /// Draw chord edges between tagged verts that share a face, splitting each
    /// such face into pieces.
    ///
    /// Follows `radial-mesh-guides/operators/CONNECT_VERTS.md`:
    ///
    /// - §3.1 collection: tag inputs and excludes, walk each input vert's
    ///   incident faces, collect n-gon candidates (triangles are never split,
    ///   §3.3).
    /// - §4 pairing: anchors are tagged corners with at least one untagged
    ///   neighbour (§4.1); consecutive-cycle pairing closing the loop, with
    ///   adjacent and duplicate-close suppression (§4.2/§4.3).
    /// - §5 filter: the `check_degenerate` legal filter (§5.2) or the default
    ///   optimal shared-face filter (§5.1).
    /// - §6 split: one [`Self::face_split`] (SFME) per surviving chord,
    ///   re-resolving the working sub-face each time (§6.1).
    ///
    /// `verts` are the tagged input corners; `faces_exclude` are faces that
    /// are never collected or split (§1); `check_degenerate` selects the
    /// legality filter over the optimal one (§1/§5).
    ///
    /// The edge-reuse variant (§6.2/§6.3) handles a chord whose endpoints are
    /// already joined by an interior/wire edge: the split adopts that edge
    /// instead of duplicating it, and any duplicate sub-face it leaves is
    /// removed in a deferred cleanup pass.
    pub fn connect_verts(
        &mut self,
        verts: &[VertexKey],
        faces_exclude: &[FaceKey],
        check_degenerate: bool,
    ) {
        self.connect_verts_out(verts, faces_exclude, check_degenerate);
    }

    /// Capturing form of [`Self::connect_verts`]: returns the `edges.out`
    /// buffer (CONNECT_VERTS.md §7) — every newly-created chord edge plus
    /// every pre-existing edge whose two endpoints are both tagged. The
    /// buffer is a set; enumeration order is not observable for parity (§7).
    pub fn connect_verts_out(
        &mut self,
        verts: &[VertexKey],
        faces_exclude: &[FaceKey],
        check_degenerate: bool,
    ) -> Vec<EdgeKey> {
        let tagged: HashSet<VertexKey> = verts.iter().copied().collect();
        let excluded: HashSet<FaceKey> = faces_exclude.iter().copied().collect();

        // §3.1 — walk input verts' incident faces, dedup candidate n-gons,
        // and seed already-connected adjacent pairs into edges.out.
        let mut edges_out: HashSet<EdgeKey> = HashSet::new();
        let mut seen_faces: HashSet<FaceKey> = HashSet::new();
        let mut candidate_faces: Vec<FaceKey> = Vec::new();
        for &v in verts {
            for f in self.vert_face_keys(v) {
                if excluded.contains(&f) {
                    continue;
                }
                if seen_faces.insert(f) && self.face_len(f) > 3 {
                    candidate_faces.push(f); // §3.3 skips triangles
                }
                self.seed_adjacent_pairs(f, &tagged, &mut edges_out);
            }
        }

        // §4–§6 — per candidate face.
        for f in candidate_faces {
            let chords = self.face_candidate_chords(f, &tagged);
            if chords.is_empty() {
                continue;
            }
            let chords = if check_degenerate {
                self.filter_legal(f, chords)
            } else {
                self.filter_optimal(f, chords)
            };
            self.cut_chords(f, &chords, &mut edges_out);
        }

        edges_out.into_iter().collect()
    }

    /// §3.1 step 4 — for each corner of `f` whose vert is tagged, if the
    /// previous or next corner is also tagged, the connecting perimeter edge
    /// joins two tagged verts and is reported in `edges.out` (a no-op pair).
    fn seed_adjacent_pairs(
        &self,
        f: FaceKey,
        tagged: &HashSet<VertexKey>,
        edges_out: &mut HashSet<EdgeKey>,
    ) {
        let edges = self.face_edge_keys(f);
        let n = edges.len();
        if n == 0 {
            return;
        }
        // Perimeter edge i joins corner i and corner i+1 (cycle order).
        for i in 0..n {
            let [a, b] = self.edge_vertices(edges[i]);
            if tagged.contains(&a) && tagged.contains(&b) {
                edges_out.insert(edges[i]);
            }
        }
    }

    /// §4 — anchors and consecutive-cycle pairing for a single face.
    /// Returns candidate chords as `(VertexKey, VertexKey)` pairs.
    fn face_candidate_chords(
        &self,
        f: FaceKey,
        tagged: &HashSet<VertexKey>,
    ) -> Vec<(VertexKey, VertexKey)> {
        let cycle = self.face_vert_keys(f);
        let n = cycle.len();

        // §4.1 — an anchor is a tagged corner with at least one untagged
        // neighbour (i.e. not strictly inside a contiguous tagged run).
        let mut anchor_positions: Vec<usize> = Vec::new();
        for i in 0..n {
            if !tagged.contains(&cycle[i]) {
                continue;
            }
            let prev = cycle[(i + n - 1) % n];
            let next = cycle[(i + 1) % n];
            if !tagged.contains(&prev) || !tagged.contains(&next) {
                anchor_positions.push(i);
            }
        }

        let m = anchor_positions.len();
        if m < 2 {
            return Vec::new();
        }

        // §4.2 — consecutive pairing closing the cycle, with adjacency and
        // duplicate-close suppression.
        let mut chords: Vec<(VertexKey, VertexKey)> = Vec::new();
        let pair_count = if m == 2 { 1 } else { m };
        for k in 0..pair_count {
            let pa = anchor_positions[k];
            let pb = anchor_positions[(k + 1) % m];
            // Adjacent anchors (immediate neighbours on the face) are already
            // joined by a perimeter edge — no chord.
            if (pa + 1) % n == pb || (pb + 1) % n == pa {
                continue;
            }
            chords.push((cycle[pa], cycle[pb]));
        }
        chords
    }

    /// §5.1 — optimal filter. For each chord, among the faces sharing both
    /// endpoints choose the highest-scoring (deterministic first-in-disk-order
    /// tie-break); null the chord when `f` is not the chosen owner. This
    /// routes a chord between a pair sharing two faces to a single face,
    /// avoiding coincident edges.
    fn filter_optimal(
        &self,
        f: FaceKey,
        chords: Vec<(VertexKey, VertexKey)>,
    ) -> Vec<(VertexKey, VertexKey)> {
        chords
            .into_iter()
            .filter(|&(a, b)| self.optimal_owner(a, b) == Some(f))
            .collect()
    }

    /// Faces sharing both `a` and `b`, in `a`'s disk-iteration order, scored
    /// by split quality; the winner replaces only on a strictly-higher score
    /// so the first-encountered face wins ties (CONNECT_VERTS.md §5.1).
    fn optimal_owner(&self, a: VertexKey, b: VertexKey) -> Option<FaceKey> {
        let faces_b: HashSet<FaceKey> = self.vert_face_keys(b).into_iter().collect();
        let mut best: Option<(FaceKey, f32)> = None;
        for f in self.vert_face_keys(a) {
            if !faces_b.contains(&f) {
                continue;
            }
            let score = self.split_quality(f, a, b);
            match best {
                Some((_, best_score)) if score <= best_score => {}
                _ => best = Some((f, score)),
            }
        }
        best.map(|(f, _)| f)
    }

    /// Split-quality dot for the chord `a`–`b` on face `f`: how cleanly the
    /// chord bisects the corners at `a` and `b`. The chord direction is
    /// scored against the inward bisector at each endpoint; higher is a
    /// cleaner split. Used only to pick a shared face (§5.1); the result is a
    /// deterministic comparison, and for symmetric shared faces both score
    /// equally so the disk-order tie-break decides.
    fn split_quality(&self, f: FaceKey, a: VertexKey, b: VertexKey) -> f32 {
        let cycle = self.face_vert_keys(f);
        let n = cycle.len();
        let pos_a = cycle.iter().position(|&v| v == a);
        let pos_b = cycle.iter().position(|&v| v == b);
        let (Some(ia), Some(ib)) = (pos_a, pos_b) else {
            return f32::NEG_INFINITY;
        };
        let pa = self.vertex_position(a);
        let pb = self.vertex_position(b);
        let chord = (pb - pa).normalize_or_zero();
        self.corner_dot(&cycle, n, ia, pa, chord) + self.corner_dot(&cycle, n, ib, pb, -chord)
    }

    fn corner_dot(&self, cycle: &[VertexKey], n: usize, i: usize, p: Vec3, dir: Vec3) -> f32 {
        let prev = self.vertex_position(cycle[(i + n - 1) % n]);
        let next = self.vertex_position(cycle[(i + 1) % n]);
        let to_prev = (prev - p).normalize_or_zero();
        let to_next = (next - p).normalize_or_zero();
        let bisector = (to_prev + to_next).normalize_or_zero();
        dir.dot(bisector)
    }

    /// §5.2 — legal filter. A convex face passes every chord (short-circuit);
    /// on a concave face each chord must be a valid interior diagonal of the
    /// face's projected polygon (it must not exit and re-enter, and must not
    /// cross another candidate chord).
    fn filter_legal(
        &self,
        f: FaceKey,
        chords: Vec<(VertexKey, VertexKey)>,
    ) -> Vec<(VertexKey, VertexKey)> {
        let cycle = self.face_vert_keys(f);
        let n = cycle.len();
        let normal = self.face_compute_normal(f);
        let (u, w) = axis_dominant_basis_negated(normal);
        let mut proj: Vec<Vec2> = cycle
            .iter()
            .map(|&v| {
                let p = self.vertex_position(v);
                Vec2::new(u.dot(p), w.dot(p))
            })
            .collect();

        // The projection's winding can come out either way depending on the
        // axis-dominant basis; the diagonal cone test assumes CCW, so flip to
        // a positive signed area first. The chord indices stay valid because
        // we only mirror the 2D coords, not reorder the cycle.
        let mut indices: Vec<usize> = (0..n).collect();
        if signed_area(&proj) < 0.0 {
            proj.reverse();
            indices.reverse();
        }
        // Map original cycle index -> position in the (possibly reversed) proj.
        let proj_pos = |orig: usize| indices.iter().position(|&x| x == orig).unwrap();

        if polygon_is_convex(&proj) {
            return chords;
        }

        chords
            .into_iter()
            .filter(|&(a, b)| {
                let ia = cycle.iter().position(|&v| v == a);
                let ib = cycle.iter().position(|&v| v == b);
                match (ia, ib) {
                    (Some(ia), Some(ib)) => {
                        is_interior_diagonal(&proj, n, proj_pos(ia), proj_pos(ib))
                    }
                    _ => false,
                }
            })
            .collect()
    }

    /// §6 — cut the surviving chords one at a time, re-resolving the working
    /// sub-face for each (§6.1). After a split a later chord's endpoints lie
    /// on exactly one of the two sub-faces (chords never cross, §4.3); the
    /// fresh per-endpoint corner lookup lands the split there.
    fn cut_chords(
        &mut self,
        f0: FaceKey,
        chords: &[(VertexKey, VertexKey)],
        edges_out: &mut HashSet<EdgeKey>,
    ) {
        let mut sub_faces: Vec<FaceKey> = vec![f0];
        // §6.3 — duplicate sub-faces from the reuse path are killed only after
        // the whole chord list is processed; killing mid-loop would invalidate
        // the working-face re-resolution of §6.1.
        let mut deferred_doubles: Vec<FaceKey> = Vec::new();
        for &(a, b) in chords {
            let Some(target) = self.find_sub_face_with_pair(&sub_faces, a, b) else {
                continue; // endpoint no longer a corner of any live sub-face
            };
            let working = sub_faces[target];
            // §6.2 — an interior/wire edge already joining a–b means the split
            // must adopt it (SFME edge-reuse variant) rather than allocate a
            // duplicate. Only this path can produce a duplicate sub-face.
            let reuse_existing_edge = self.query_edge_exists(a, b).is_some();
            let (new_face, new_edge) = self
                .face_split_reusing(working, a, b, reuse_existing_edge)
                .expect("connect_verts: face_split failed on a filtered chord");
            edges_out.insert(new_edge);
            sub_faces.push(new_face);
            if reuse_existing_edge {
                self.collect_doubles_after_split(working, new_face, &mut deferred_doubles);
            }
        }
        for f_dup in deferred_doubles {
            self.face_kill(f_dup);
        }
    }

    /// §6.3 — after a reuse split, either resulting sub-face may duplicate a
    /// pre-existing face (one whose corner cycle coincides). Record each
    /// sub-face that has such a twin so it can be killed once the chord list is
    /// done. The kept twin is the pre-existing face; the freshly-produced
    /// sub-face is the one recorded for removal.
    fn collect_doubles_after_split(
        &self,
        survived: FaceKey,
        new_face: FaceKey,
        deferred: &mut Vec<FaceKey>,
    ) {
        for sub_face in [survived, new_face] {
            if self.face_has_duplicate(sub_face) {
                deferred.push(sub_face);
            }
        }
    }

    /// True when some *other* live face has the same corner-vertex set as
    /// `face` (a duplicate sub-face, §6.3). The corner cycles of two faces over
    /// the same vertex set are equal up to rotation/reflection, so a set
    /// comparison suffices to detect the duplicate.
    fn face_has_duplicate(&self, face: FaceKey) -> bool {
        let target: HashSet<VertexKey> = self.face_vert_keys(face).into_iter().collect();
        self.face_keys().into_iter().any(|other| {
            other != face && {
                let verts = self.face_vert_keys(other);
                verts.len() == target.len() && verts.iter().all(|v| target.contains(v))
            }
        })
    }

    fn find_sub_face_with_pair(
        &self,
        sub_faces: &[FaceKey],
        a: VertexKey,
        b: VertexKey,
    ) -> Option<usize> {
        sub_faces.iter().position(|&f| {
            let verts = self.face_vert_keys(f);
            verts.contains(&a) && verts.contains(&b)
        })
    }
}

/// Signed area of the 2D polygon (positive for CCW winding).
pub(crate) fn signed_area(poly: &[Vec2]) -> f32 {
    let n = poly.len();
    let mut sum = 0.0_f32;
    for i in 0..n {
        let a = poly[i];
        let b = poly[(i + 1) % n];
        sum += a.perp_dot(b);
    }
    sum * 0.5
}

/// True when the 2D polygon (in cycle order) turns the same way at every
/// corner — i.e. it is convex. Used to short-circuit the §5.2 legal filter.
fn polygon_is_convex(poly: &[Vec2]) -> bool {
    let n = poly.len();
    if n < 4 {
        return true;
    }
    let mut sign = 0.0_f32;
    for i in 0..n {
        let a = poly[i];
        let b = poly[(i + 1) % n];
        let c = poly[(i + 2) % n];
        let cross = (b - a).perp_dot(c - b);
        if cross.abs() > 1e-12 {
            if sign == 0.0 {
                sign = cross.signum();
            } else if cross.signum() != sign {
                return false;
            }
        }
    }
    true
}

/// True when the chord between corners `ia` and `ib` is a valid interior
/// diagonal of the projected polygon `poly`: it stays inside the polygon
/// (it is in the interior cone at each endpoint) and crosses no perimeter
/// edge (CONNECT_VERTS.md §5.2).
pub(crate) fn is_interior_diagonal(poly: &[Vec2], n: usize, ia: usize, ib: usize) -> bool {
    if ia == ib {
        return false;
    }
    // Adjacent corners share a perimeter edge — not an interior diagonal.
    if (ia + 1) % n == ib || (ib + 1) % n == ia {
        return false;
    }
    if !in_cone(poly, n, ia, ib) || !in_cone(poly, n, ib, ia) {
        return false;
    }
    let a = poly[ia];
    let b = poly[ib];
    for i in 0..n {
        let j = (i + 1) % n;
        if i == ia || i == ib || j == ia || j == ib {
            continue; // edges incident to an endpoint can't be a crossing
        }
        if segments_properly_intersect(a, b, poly[i], poly[j]) {
            return false;
        }
    }
    true
}

/// True when the ray from corner `i` toward corner `j` enters the polygon's
/// interior — i.e. it lies within the open cone spanned by `i`'s two
/// incident edges. Handles both convex and reflex corners.
fn in_cone(poly: &[Vec2], n: usize, i: usize, j: usize) -> bool {
    let p = poly[i];
    let prev = poly[(i + n - 1) % n];
    let next = poly[(i + 1) % n];
    let to_j = poly[j] - p;
    // Convex corner: interior cone is the intersection of the two half-planes.
    if left_or_on(prev, p, next) {
        return left_strict(p, next, p + to_j) && left_strict(prev, p, p + to_j);
    }
    // Reflex corner: interior cone is the complement, the union of half-planes.
    !(left_or_on(p, next, p + to_j) && left_or_on(prev, p, p + to_j))
}

fn left_strict(a: Vec2, b: Vec2, c: Vec2) -> bool {
    (b - a).perp_dot(c - a) > 0.0
}

fn left_or_on(a: Vec2, b: Vec2, c: Vec2) -> bool {
    (b - a).perp_dot(c - a) >= 0.0
}

/// True when segments `a`–`b` and `c`–`d` cross at an interior point of both.
fn segments_properly_intersect(a: Vec2, b: Vec2, c: Vec2, d: Vec2) -> bool {
    let d1 = (b - a).perp_dot(c - a);
    let d2 = (b - a).perp_dot(d - a);
    let d3 = (d - c).perp_dot(a - c);
    let d4 = (d - c).perp_dot(b - c);
    ((d1 > 0.0) != (d2 > 0.0)) && ((d3 > 0.0) != (d4 > 0.0))
}

#[cfg(test)]
mod tests {
    use glam::Vec3;

    use crate::loopmesh::test_support::create_face_from_verts;
    use crate::loopmesh::*;

    /// Regular convex n-gon in the z=0 plane, returns (mesh, face, verts in
    /// cycle order).
    fn ngon(n: usize) -> (LoopMesh, FaceKey, Vec<VertexKey>) {
        let mut mesh = LoopMesh::new();
        let coords: Vec<Vec3> = (0..n)
            .map(|i| {
                let theta = (i as f32) * std::f32::consts::TAU / (n as f32);
                Vec3::new(theta.cos(), theta.sin(), 0.0)
            })
            .collect();
        let face = create_face_from_verts(&mut mesh, &coords);
        let verts = mesh.face_vert_keys(face);
        (mesh, face, verts)
    }

    #[test]
    fn quad_two_opposite_corners_one_diagonal_cut() {
        let (mut mesh, _face, v) = ngon(4);
        let created = mesh.connect_verts_out(&[v[0], v[2]], &[], false);
        assert_eq!(created.len(), 1, "one chord cut between opposite corners");
        assert_eq!(mesh.face_keys().len(), 2, "quad split into two triangles");
        for f in mesh.face_keys() {
            assert_eq!(mesh.face_len(f), 3);
        }
        assert!(mesh.query_edge_exists(v[0], v[2]).is_some());
    }

    #[test]
    fn quad_two_adjacent_corners_no_cut_but_reported() {
        let (mut mesh, _face, v) = ngon(4);
        let faces_before = mesh.face_keys().len();
        let created = mesh.connect_verts_out(&[v[0], v[1]], &[], false);
        // Adjacency suppression: no new chord, mesh unchanged.
        assert_eq!(mesh.face_keys().len(), faces_before);
        assert_eq!(mesh.face_len(_face), 4);
        // The pre-existing perimeter edge between the tagged pair is reported.
        let existing = mesh.query_edge_exists(v[0], v[1]).unwrap();
        assert_eq!(created, vec![existing]);
    }

    #[test]
    fn hexagon_three_nonadjacent_corners_chord_ring() {
        let (mut mesh, _face, v) = ngon(6);
        let created = mesh.connect_verts_out(&[v[0], v[2], v[4]], &[], false);
        // Three chords {0,2},{2,4},{4,0} → central tri + three caps = 4 faces.
        assert_eq!(created.len(), 3);
        assert_eq!(mesh.face_keys().len(), 4);
        // The chords never cross: each tagged corner gains exactly two chords.
        for &corner in &[v[0], v[2], v[4]] {
            // perimeter degree 2 + two chord ends = 4 incident edges.
            assert_eq!(mesh.vert_edge_keys(corner).len(), 4);
        }
    }

    #[test]
    fn hexagon_contiguous_run_one_endpoint_chord() {
        let (mut mesh, _face, v) = ngon(6);
        // Run 0,1,2: only endpoints 0 and 2 are anchors; one chord {0,2}.
        // (`created` also reports the pre-existing run edges 0-1, 1-2 — that
        // count is pinned separately by `edges_out_reports_pre_existing_tagged_pair`.)
        mesh.connect_verts_out(&[v[0], v[1], v[2]], &[], false);
        assert_eq!(
            mesh.face_keys().len(),
            2,
            "one endpoint chord cuts the run off"
        );
        assert!(mesh.query_edge_exists(v[0], v[2]).is_some());
        // Interior run corner 1 gained no chord (still perimeter degree 2).
        assert_eq!(mesh.vert_edge_keys(v[1]).len(), 2);
    }

    #[test]
    fn single_tagged_corner_is_noop() {
        let (mut mesh, _face, v) = ngon(5);
        let created = mesh.connect_verts_out(&[v[0]], &[], false);
        assert!(created.is_empty());
        assert_eq!(mesh.face_keys().len(), 1);
        assert_eq!(mesh.face_len(_face), 5);
    }

    #[test]
    fn triangle_never_split() {
        let (mut mesh, face, v) = ngon(3);
        let created = mesh.connect_verts_out(&[v[0], v[1], v[2]], &[], false);
        // No cut (triangle excluded at collection), but its three perimeter
        // edges between tagged pairs are reported.
        assert_eq!(mesh.face_keys().len(), 1);
        assert_eq!(mesh.face_len(face), 3);
        assert_eq!(created.len(), 3);
    }

    #[test]
    fn check_degenerate_blocks_chord_exiting_concave_face() {
        // Concave pentagon: reflex vertex at index 2 pulled inward so the
        // chord {1,3} would pass outside the polygon.
        let mut mesh = LoopMesh::new();
        let coords = [
            Vec3::new(-2.0, 0.0, 0.0),
            Vec3::new(-1.0, 2.0, 0.0),
            Vec3::new(0.0, 0.5, 0.0), // notch dipping toward centre
            Vec3::new(1.0, 2.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
        ];
        let face = create_face_from_verts(&mut mesh, &coords);
        let v = mesh.face_vert_keys(face);
        let created = mesh.connect_verts_out(&[v[1], v[3]], &[], true);
        // Legal filter nulls the exiting chord: face unchanged.
        assert!(created.is_empty());
        assert_eq!(mesh.face_keys().len(), 1);
        assert_eq!(mesh.face_len(face), 5);
    }

    #[test]
    fn check_degenerate_false_attempts_chord_on_concave_face() {
        let mut mesh = LoopMesh::new();
        let coords = [
            Vec3::new(-1.0, 0.0, 0.0),
            Vec3::new(-0.5, 1.0, 0.0),
            Vec3::new(0.0, 0.1, 0.0),
            Vec3::new(0.5, 1.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
        ];
        let face = create_face_from_verts(&mut mesh, &coords);
        let v = mesh.face_vert_keys(face);
        // Optimal filter does no geometry test; the single shared face owns
        // the chord, so it is cut despite exiting the polygon.
        let created = mesh.connect_verts_out(&[v[1], v[3]], &[], false);
        assert_eq!(created.len(), 1);
        assert_eq!(mesh.face_keys().len(), 2);
    }

    #[test]
    fn shared_face_pair_routes_single_chord() {
        // Two quads sharing only the corner pair b,d (no edge between them):
        // quad1 = [b, x1, d, y1], quad2 = [b, x2, d, y2]. The chord {b,d} is a
        // candidate on both; the optimal filter routes it to exactly one.
        let mut mesh = LoopMesh::new();
        let b = mesh.create_vertex(Vec3::new(0.0, 0.0, 0.0));
        let d = mesh.create_vertex(Vec3::new(2.0, 0.0, 0.0));
        let x1 = mesh.create_vertex(Vec3::new(1.0, 1.0, 0.0));
        let y1 = mesh.create_vertex(Vec3::new(1.0, -1.0, 0.0));
        let x2 = mesh.create_vertex(Vec3::new(1.0, 3.0, 0.0));
        let y2 = mesh.create_vertex(Vec3::new(1.0, 1.5, 0.0));

        let mk = |mesh: &mut LoopMesh, ring: [VertexKey; 4]| {
            let e: Vec<EdgeKey> = (0..4)
                .map(|i| {
                    let pair = [ring[i], ring[(i + 1) % 4]];
                    mesh.query_edge_exists(pair[0], pair[1])
                        .unwrap_or_else(|| mesh.create_edge(pair, EdgeCreateFlags::NONE))
                })
                .collect();
            mesh.face_create(&ring, &e, FaceCreateFlags::NONE)
        };
        mk(&mut mesh, [b, x1, d, y1]);
        mk(&mut mesh, [b, x2, d, y2]);

        let created = mesh.connect_verts_out(&[b, d], &[], false);
        // Exactly one chord drawn (routed to a single shared face), not two
        // coincident edges, not zero.
        assert_eq!(created.len(), 1);
        assert_eq!(mesh.face_keys().len(), 3, "one of the two quads is split");
        // The chord edge b–d now exists exactly once.
        assert!(mesh.query_edge_exists(b, d).is_some());
    }

    #[test]
    fn faces_exclude_protects_face() {
        // Two quads sharing edge 1-2: quad0 = [0,1,2,3], quad1 = [1,4,5,2].
        let mut mesh = LoopMesh::new();
        let p = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::new(2.0, 1.0, 0.0),
        ];
        let vk: Vec<VertexKey> = p.iter().map(|&c| mesh.create_vertex(c)).collect();
        let mk = |mesh: &mut LoopMesh, ring: &[VertexKey]| {
            let e: Vec<EdgeKey> = (0..ring.len())
                .map(|i| {
                    let pair = [ring[i], ring[(i + 1) % ring.len()]];
                    mesh.query_edge_exists(pair[0], pair[1])
                        .unwrap_or_else(|| mesh.create_edge(pair, EdgeCreateFlags::NONE))
                })
                .collect();
            mesh.face_create(ring, &e, FaceCreateFlags::NONE)
        };
        let quad0 = mk(&mut mesh, &[vk[0], vk[1], vk[2], vk[3]]);
        mk(&mut mesh, &[vk[1], vk[4], vk[5], vk[2]]);

        let faces_before = mesh.face_keys().len();
        // Verts 0 and 2 are a diagonal of quad0 only; exclude quad0.
        let created = mesh.connect_verts_out(&[vk[0], vk[2]], &[quad0], false);
        assert!(created.is_empty(), "excluded face's chord is suppressed");
        assert_eq!(mesh.face_keys().len(), faces_before, "mesh unchanged");
    }

    #[test]
    fn edge_exists_reuse_kills_duplicate_subfaces() {
        // §6.2 / §6.3 fixture: a quad [0,1,2,3] overlaid on two coincident
        // triangles [0,1,2] and [0,2,3] that already share the interior
        // diagonal edge {0,2}. Tagging 0 and 2 yields the single chord {0,2};
        // its endpoints are already joined by {0,2}, so the split must reuse
        // that edge and then kill the two duplicate sub-faces it leaves.
        let mut mesh = LoopMesh::new();
        let coords = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        ];
        let vk: Vec<VertexKey> = coords.iter().map(|&c| mesh.create_vertex(c)).collect();

        let edge_of = |mesh: &mut LoopMesh, a: VertexKey, b: VertexKey| {
            mesh.query_edge_exists(a, b)
                .unwrap_or_else(|| mesh.create_edge([a, b], EdgeCreateFlags::NONE))
        };
        let mk = |mesh: &mut LoopMesh, ring: &[VertexKey]| {
            let edges: Vec<EdgeKey> = (0..ring.len())
                .map(|i| edge_of(mesh, ring[i], ring[(i + 1) % ring.len()]))
                .collect();
            mesh.face_create(ring, &edges, FaceCreateFlags::NONE)
        };

        let quad = mk(&mut mesh, &[vk[0], vk[1], vk[2], vk[3]]);
        mk(&mut mesh, &[vk[0], vk[1], vk[2]]);
        mk(&mut mesh, &[vk[0], vk[2], vk[3]]);

        let diagonal = mesh.query_edge_exists(vk[0], vk[2]).unwrap();
        assert_eq!(mesh.vert_keys().len(), 4);
        assert_eq!(mesh.edge_keys().len(), 5);
        assert_eq!(mesh.face_keys().len(), 3);

        let created = mesh.connect_verts_out(&[vk[0], vk[2]], &[], false);

        // Net delta: -1 face, 0 edges, 0 verts. The reused diagonal is the
        // edge reported, not a freshly-created duplicate.
        assert_eq!(mesh.vert_keys().len(), 4, "no verts created or removed");
        assert_eq!(mesh.edge_keys().len(), 5, "diagonal reused, no edge added");
        assert_eq!(
            mesh.face_keys().len(),
            2,
            "quad consumed, duplicates killed"
        );
        assert_eq!(
            mesh.query_edge_exists(vk[0], vk[2]),
            Some(diagonal),
            "the original diagonal survives (not duplicated)"
        );
        assert!(
            created.contains(&diagonal),
            "the reused diagonal is reported in edges.out"
        );
        for f in mesh.face_keys() {
            assert_eq!(mesh.face_len(f), 3, "both survivors are triangles");
        }
        // The quad handle is gone (consumed into a triangle and then either
        // kept as a survivor or killed as a duplicate — either way no quad
        // remains).
        assert!(!mesh.face_keys().iter().any(|&f| mesh.face_len(f) == 4));
        let _ = quad;
    }

    #[test]
    fn edges_out_reports_pre_existing_tagged_pair() {
        // Tag three corners along a contiguous run so a pre-existing perimeter
        // edge between adjacent tagged verts is reported alongside the cut.
        let (mut mesh, _face, v) = ngon(6);
        let created = mesh.connect_verts_out(&[v[0], v[1], v[2]], &[], false);
        // One created chord {0,2} plus the two pre-existing run edges 0-1, 1-2.
        let e01 = mesh.query_edge_exists(v[0], v[1]).unwrap();
        let e12 = mesh.query_edge_exists(v[1], v[2]).unwrap();
        let chord = mesh.query_edge_exists(v[0], v[2]).unwrap();
        assert!(created.contains(&e01));
        assert!(created.contains(&e12));
        assert!(created.contains(&chord));
        assert_eq!(created.len(), 3);
    }
}
