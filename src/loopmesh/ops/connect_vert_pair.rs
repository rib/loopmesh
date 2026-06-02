use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashSet};

use glam::Vec3;

use crate::loopmesh::*;

/// Sentinel index meaning "no parent link" — the chain root (§5 walks until it
/// hits this).
const NO_PARENT: usize = usize::MAX;

/// Tolerance for the signed-plane side test (§3.3). A point within this
/// absolute distance of the cutting plane is treated as lying on it.
const PLANE_EPS: f32 = 1e-5;

/// One element the route walks: an existing vert it lands on, or an edge it
/// crosses (CONNECT_VERT_PAIR.md §4.1).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
enum RouteElement {
    Vert(VertexKey),
    Edge(EdgeKey),
}

/// The cutting plane fitted to the two endpoints and their normals (§3.3).
/// `normal` is the plane normal (the cross-product axis); `offset` is the
/// signed projection of either endpoint onto it.
struct CuttingPlane {
    normal: Vec3,
    offset: f32,
}

impl CuttingPlane {
    /// Signed distance of a position from the plane (§3.3).
    fn signed_distance(&self, position: Vec3) -> f32 {
        self.normal.dot(position) - self.offset
    }
}

/// One node of the route chain, stored in a flat arena. Each node records the
/// element walked onto plus the index of the link it was reached *from*, so the
/// winning route is recovered by a single parent-pointer walk (§5) without ever
/// cloning the partial chain during the search (§9).
#[derive(Clone, Copy)]
struct RouteLink {
    element: RouteElement,
    parent: usize,
}

/// A partial route in the search frontier. `length` is the accumulated 3D
/// distance walked so far; the heap pops the shortest first (§4). `link` is the
/// index into the arena of the route's last node — the chain is reconstructed
/// by following `parent` pointers (§5).
struct PathState {
    length: f32,
    last_point: Vec3,
    last_element: RouteElement,
    /// The face the route crossed to reach `last_element` (§4.2). The next
    /// expansion skips this face so the route never immediately doubles back
    /// through the same face — the §4.1/§4.3 backward candidate would otherwise
    /// re-enter the just-crossed face and produce a spurious chord. `None` when
    /// the route arrived along an on-plane edge rather than across a face.
    arrived_via: Option<FaceKey>,
    link: usize,
}

impl PartialEq for PathState {
    fn eq(&self, other: &Self) -> bool {
        self.length.total_cmp(&other.length) == Ordering::Equal
    }
}
impl Eq for PathState {}
impl PartialOrd for PathState {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for PathState {
    fn cmp(&self, other: &Self) -> Ordering {
        // `BinaryHeap` is a max-heap; reverse so the shortest length is "max".
        // `total_cmp` keeps the pop order a strict total order even on NaN, so
        // the route returned stays deterministic on pathological geometry (§10).
        other.length.total_cmp(&self.length)
    }
}

impl LoopMesh {
    /// Connect two specific verts with a cut that threads across the faces
    /// between them, splitting each.
    ///
    /// Follows `radial-mesh-guides/operators/CONNECT_VERT_PAIR.md`: fit a
    /// cutting plane to the two endpoints and their normals (§3.3), run a
    /// length-ordered shortest-path search confined to that plane (§4), split
    /// every edge the route crosses at its plane-crossing point (§5.2), then
    /// delegate the cut to [`Self::connect_verts`] with `check_degenerate`
    /// always on (§6).
    ///
    /// No route (disconnected components, or every route blocked by the
    /// exclusions), adjacent endpoints, and the same vert twice are all no-ops
    /// (§7), not errors.
    ///
    /// `verts_exclude` are verts the route may not step onto; `faces_exclude`
    /// are faces the route may not cross and that the cut never splits (§3.2).
    pub fn connect_vert_pair(
        &mut self,
        verts: &[VertexKey],
        verts_exclude: &[VertexKey],
        faces_exclude: &[FaceKey],
    ) {
        self.connect_vert_pair_out(verts, verts_exclude, faces_exclude);
    }

    /// Capturing form of [`Self::connect_vert_pair`]: returns the `edges.out`
    /// buffer forwarded verbatim from the `connect_verts` sub-pass
    /// (CONNECT_VERT_PAIR.md §1 / §6) — created chord edges plus any
    /// pre-existing edge already joining two route verts.
    pub fn connect_vert_pair_out(
        &mut self,
        verts: &[VertexKey],
        verts_exclude: &[VertexKey],
        faces_exclude: &[FaceKey],
    ) -> Vec<EdgeKey> {
        // §3.1 — exactly-two gate.
        if verts.len() != 2 {
            return Vec::new();
        }
        let source = verts[0];
        let target = verts[1];

        let excluded_faces: HashSet<FaceKey> = faces_exclude.iter().copied().collect();
        let excluded_verts: HashSet<VertexKey> = verts_exclude.iter().copied().collect();

        // §7 — same vert twice: no pair to chord.
        if source == target {
            return Vec::new();
        }

        let plane = self.fit_cutting_plane(source, target);

        let route = self.search_route(source, target, &plane, &excluded_faces, &excluded_verts);

        // §4.4 / §7 — no in-plane route: no cut, empty output.
        let Some(route) = route else {
            return Vec::new();
        };

        // §5 — realise the route as verts, splitting each crossed edge.
        let mut route_verts: Vec<VertexKey> = Vec::new();
        let mut route_vert_set: HashSet<VertexKey> = HashSet::new();
        let mut push_route_vert = |v: VertexKey,
                                   verts: &mut Vec<VertexKey>,
                                   set: &mut HashSet<VertexKey>| {
            if set.insert(v) {
                verts.push(v);
            }
        };
        for element in &route {
            match *element {
                RouteElement::Vert(v) => {
                    push_route_vert(v, &mut route_verts, &mut route_vert_set);
                }
                RouteElement::Edge(e) => {
                    let [a, b] = self.edge_vertices(e);
                    let fraction = self.plane_crossing_fraction(&plane, a, b);
                    let (v_new, _e_new) = self.edge_split(e, a, fraction);
                    push_route_vert(v_new, &mut route_verts, &mut route_vert_set);
                }
            }
        }
        // §5 — the two endpoints are always part of the cut.
        push_route_vert(source, &mut route_verts, &mut route_vert_set);
        push_route_vert(target, &mut route_verts, &mut route_vert_set);

        // §6 — delegate the cut to connect-verts, check_degenerate always on.
        self.connect_verts_out(&route_verts, faces_exclude, true)
    }

    /// §3.3 — fit the cutting plane to `source`, `target` and their normals.
    fn fit_cutting_plane(&self, source: VertexKey, target: VertexKey) -> CuttingPlane {
        let p_source = self.vertex_position(source);
        let p_target = self.vertex_position(target);
        let segment = (p_target - p_source).normalize_or_zero();

        let n_source = self.endpoint_normal(source);
        let n_target = self.endpoint_normal(target);

        // Project each normal off the segment direction so a normal pointing
        // along the segment contributes nothing, then sum with a sign-align so
        // two oppositely-wound faces reinforce rather than cancel (§3.3).
        let proj_source = (n_source - segment * n_source.dot(segment)).normalize_or_zero();
        let mut proj_target = (n_target - segment * n_target.dot(segment)).normalize_or_zero();
        if proj_source.dot(proj_target) < 0.0 {
            proj_target = -proj_target;
        }
        let in_surface = (proj_source + proj_target).normalize_or_zero();

        let mut normal = segment.cross(in_surface);

        // Degenerate: projected normals cancelled or were absent. Pick, per
        // endpoint, the incident face whose plane is most nearly aligned with
        // the segment direction, and use the better of the two (§3.3).
        if normal.length_squared() < PLANE_EPS * PLANE_EPS {
            let fallback = self
                .best_aligned_in_surface(source, segment)
                .or_else(|| self.best_aligned_in_surface(target, segment));
            if let Some(in_surface) = fallback {
                normal = segment.cross(in_surface);
            }
        }

        // Last resort: any axis orthogonal to the segment (§3.3).
        if normal.length_squared() < PLANE_EPS * PLANE_EPS {
            let helper = if segment.x.abs() < 0.9 { Vec3::X } else { Vec3::Y };
            normal = segment.cross(helper);
        }

        let normal = normal.normalize_or_zero();
        let offset = normal.dot(p_source);
        CuttingPlane { normal, offset }
    }

    /// The in-surface normal of `vert` for the plane fit. Computed fresh from
    /// the incident faces so the fit does not depend on stale cached normals;
    /// falls back to the stored vertex normal when there are no faces.
    fn endpoint_normal(&self, vert: VertexKey) -> Vec3 {
        let mut accumulated = Vec3::ZERO;
        for f in self.vert_face_keys(vert) {
            accumulated += self.face_compute_normal(f);
        }
        if accumulated.length_squared() > 0.0 {
            accumulated.normalize()
        } else {
            self.vertex_normal(vert)
        }
    }

    /// §3.3 degenerate fallback — among `vert`'s incident faces, the in-plane
    /// direction (segment projected into the face plane) of the face whose
    /// normal is most nearly perpendicular to the segment, i.e. whose plane
    /// most nearly contains the segment.
    fn best_aligned_in_surface(&self, vert: VertexKey, segment: Vec3) -> Option<Vec3> {
        let mut best: Option<(f32, Vec3)> = None;
        for f in self.vert_face_keys(vert) {
            let face_normal = self.face_compute_normal(f);
            if face_normal.length_squared() == 0.0 {
                continue;
            }
            let in_plane = (segment - face_normal * segment.dot(face_normal)).normalize_or_zero();
            if in_plane.length_squared() == 0.0 {
                continue;
            }
            // A face plane that contains the segment has a normal perpendicular
            // to it: minimise |normal·segment|.
            let alignment = 1.0 - face_normal.dot(segment).abs();
            match best {
                Some((best_alignment, _)) if alignment <= best_alignment => {}
                _ => best = Some((alignment, in_plane)),
            }
        }
        best.map(|(_, dir)| dir)
    }

    /// §5.3 — the crossing fraction `t` along edge `a`→`b` where the plane
    /// cuts it: `t = |dist(a)| / (|dist(a)| + |dist(b)|)`. A near-on-plane
    /// edge (both distances within tolerance) splits at the midpoint.
    fn plane_crossing_fraction(
        &self,
        plane: &CuttingPlane,
        a: VertexKey,
        b: VertexKey,
    ) -> f32 {
        let da = plane.signed_distance(self.vertex_position(a)).abs();
        let db = plane.signed_distance(self.vertex_position(b)).abs();
        let total = da + db;
        if total < PLANE_EPS {
            0.5
        } else {
            da / total
        }
    }

    /// §4 — shortest-path search confined to the cutting plane. Returns the
    /// winning route as a chain of elements from `source` to `target`, or
    /// `None` when no in-plane route reaches the target (§4.4).
    fn search_route(
        &self,
        source: VertexKey,
        target: VertexKey,
        plane: &CuttingPlane,
        excluded_faces: &HashSet<FaceKey>,
        excluded_verts: &HashSet<VertexKey>,
    ) -> Option<Vec<RouteElement>> {
        // §4.1 — the touched-once set is shared across all live states.
        let mut touched: HashSet<RouteElement> = HashSet::new();

        // Arena of route links; states hold an index into it and the winning
        // route is recovered by a single parent-pointer walk (§5), so no partial
        // chain is ever cloned during the search (§9).
        let mut links: Vec<RouteLink> = Vec::new();

        let mut heap: BinaryHeap<PathState> = BinaryHeap::new();
        let source_element = RouteElement::Vert(source);
        touched.insert(source_element);
        links.push(RouteLink {
            element: source_element,
            parent: NO_PARENT,
        });
        heap.push(PathState {
            length: 0.0,
            last_point: self.vertex_position(source),
            last_element: source_element,
            arrived_via: None,
            link: 0,
        });

        while let Some(state) = heap.pop() {
            // §4.4 — target popped (checked before expansion): shortest wins.
            if state.last_element == RouteElement::Vert(target) {
                return Some(self.recover_chain(&links, state.link));
            }

            let candidates =
                self.expand_state(&state, plane, excluded_faces, excluded_verts, target);

            for candidate in candidates {
                if touched.contains(&candidate.element) {
                    continue;
                }
                touched.insert(candidate.element);
                let length = state.length + state.last_point.distance(candidate.point);
                let link = links.len();
                links.push(RouteLink {
                    element: candidate.element,
                    parent: state.link,
                });
                heap.push(PathState {
                    length,
                    last_point: candidate.point,
                    last_element: candidate.element,
                    arrived_via: candidate.arrived_via,
                    link,
                });
            }
        }

        None
    }

    /// §5 — walk the arena's parent pointers from the winning link back to the
    /// source, returning the route in source→target order.
    fn recover_chain(&self, links: &[RouteLink], last: usize) -> Vec<RouteElement> {
        let mut chain: Vec<RouteElement> = Vec::new();
        let mut index = last;
        while index != NO_PARENT {
            chain.push(links[index].element);
            index = links[index].parent;
        }
        chain.reverse();
        chain
    }

    /// §4.2 — expand a state from its last element into the next candidate
    /// steps.
    fn expand_state(
        &self,
        state: &PathState,
        plane: &CuttingPlane,
        excluded_faces: &HashSet<FaceKey>,
        excluded_verts: &HashSet<VertexKey>,
        target: VertexKey,
    ) -> Vec<Candidate> {
        let mut candidates: Vec<Candidate> = Vec::new();
        match state.last_element {
            RouteElement::Vert(v) => {
                // Walk incident faces and collect their plane crossings,
                // skipping the face the route just crossed (§4.2 no-double-back).
                for f in self.vert_face_keys(v) {
                    if excluded_faces.contains(&f) || state.arrived_via == Some(f) {
                        continue;
                    }
                    self.collect_face_crossings(
                        f,
                        Some(v),
                        plane,
                        excluded_verts,
                        target,
                        &mut candidates,
                    );
                }
                // Walk incident on-plane edges directly (§4.2 vert-edge step).
                for e in self.vert_edge_keys(v) {
                    let [a, b] = self.edge_vertices(e);
                    let far = if a == v { b } else { a };
                    if excluded_verts.contains(&far) {
                        continue;
                    }
                    let far_pos = self.vertex_position(far);
                    let on_plane = plane.signed_distance(far_pos).abs() <= PLANE_EPS;
                    if on_plane || far == target {
                        // Walking along an on-plane edge crosses no face, so the
                        // no-double-back face rule (§4.2) does not apply here.
                        candidates.push(Candidate {
                            element: RouteElement::Vert(far),
                            point: far_pos,
                            arrived_via: None,
                        });
                    }
                }
            }
            RouteElement::Edge(e) => {
                // The route is mid-crossing: continue across both faces of `e`,
                // skipping the one it arrived through (§4.2 no-double-back).
                for f in self.edge_face_keys(e) {
                    if excluded_faces.contains(&f) || state.arrived_via == Some(f) {
                        continue;
                    }
                    self.collect_face_crossings(
                        f,
                        None,
                        plane,
                        excluded_verts,
                        target,
                        &mut candidates,
                    );
                }
            }
        }
        candidates
    }

    /// §4.3 — collect the plane crossings on a face, keeping the closest in
    /// each of the two directions along the plane. `from_vert` is the vert the
    /// route is standing on (its own crossing is skipped); `None` when the
    /// route entered the face mid-crossing through an edge.
    fn collect_face_crossings(
        &self,
        f: FaceKey,
        from_vert: Option<VertexKey>,
        plane: &CuttingPlane,
        excluded_verts: &HashSet<VertexKey>,
        target: VertexKey,
        out: &mut Vec<Candidate>,
    ) {
        let from_pos = from_vert.map(|v| self.vertex_position(v));

        // Reference point for direction bucketing: the vert we stand on, or
        // the face centroid when entering mid-crossing.
        let reference = from_pos.unwrap_or_else(|| self.face_centroid(f));

        let mut reference_direction: Option<Vec3> = None;
        let mut forward_best: Option<(f32, Candidate)> = None;
        let mut backward_best: Option<(f32, Candidate)> = None;

        let mut consider = |element: RouteElement, point: Vec3| {
            let offset = point - reference;
            let distance = offset.length();
            if distance <= PLANE_EPS {
                return;
            }
            let direction = offset / distance;
            // Every candidate found here was reached by crossing face `f`, so the
            // next expansion skips `f` (§4.2 no-double-back).
            let candidate = Candidate {
                element,
                point,
                arrived_via: Some(f),
            };
            match reference_direction {
                None => {
                    reference_direction = Some(direction);
                    forward_best = Some((distance, candidate));
                }
                Some(reference_direction) => {
                    if direction.dot(reference_direction) >= 0.0 {
                        if forward_best
                            .as_ref()
                            .map(|(d, _)| distance < *d)
                            .unwrap_or(true)
                        {
                            forward_best = Some((distance, candidate));
                        }
                    } else if backward_best
                        .as_ref()
                        .map(|(d, _)| distance < *d)
                        .unwrap_or(true)
                    {
                        backward_best = Some((distance, candidate));
                    }
                }
            }
        };

        // On-plane verts of the face (other than the one we stand on).
        for v in self.face_vert_keys(f) {
            if Some(v) == from_vert {
                continue;
            }
            // The target is never excluded even if it is named in the set.
            if excluded_verts.contains(&v) && v != target {
                continue;
            }
            let pos = self.vertex_position(v);
            if plane.signed_distance(pos).abs() <= PLANE_EPS {
                consider(RouteElement::Vert(v), pos);
            }
        }

        // Crossings on the face's edges.
        for e in self.face_edge_keys(f) {
            let [a, b] = self.edge_vertices(e);
            // An edge incident to the vert we stand on shares its endpoint; a
            // genuine interior crossing needs the two endpoints on strictly
            // opposite sides.
            let da = plane.signed_distance(self.vertex_position(a));
            let db = plane.signed_distance(self.vertex_position(b));
            let a_on = da.abs() <= PLANE_EPS;
            let b_on = db.abs() <= PLANE_EPS;
            if a_on || b_on {
                continue; // endpoint-on-plane handled by the vert pass above
            }
            if (da > 0.0) == (db > 0.0) {
                continue; // same side: no crossing
            }
            let fraction = da.abs() / (da.abs() + db.abs());
            let point = self
                .vertex_position(a)
                .lerp(self.vertex_position(b), fraction);
            consider(RouteElement::Edge(e), point);
        }

        if let Some((_, candidate)) = forward_best {
            out.push(candidate);
        }
        if let Some((_, candidate)) = backward_best {
            out.push(candidate);
        }
    }

    fn face_centroid(&self, f: FaceKey) -> Vec3 {
        let verts = self.face_vert_keys(f);
        if verts.is_empty() {
            return Vec3::ZERO;
        }
        let mut sum = Vec3::ZERO;
        for v in &verts {
            sum += self.vertex_position(*v);
        }
        sum / verts.len() as f32
    }
}

/// One candidate next step out of a state (§4.2 / §4.3).
struct Candidate {
    element: RouteElement,
    point: Vec3,
    /// The face crossed to reach `element` (§4.2), or `None` if reached by
    /// walking along an on-plane edge. The next expansion skips this face.
    arrived_via: Option<FaceKey>,
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use glam::Vec3;

    use super::{CuttingPlane, PathState, RouteElement};
    use crate::loopmesh::test_support::create_face_from_verts;
    use crate::loopmesh::*;

    /// Build a strip of `cols` quads in the z=0 plane. Returns the mesh and the
    /// bottom-row / top-row vert keys in column order. Bottom-left is
    /// `bottom[0]`, bottom-right `bottom[cols]`. The strip is geometrically
    /// asymmetric in x (column widths grow) so the shortest in-plane route is
    /// unique.
    fn quad_strip(cols: usize) -> (LoopMesh, Vec<VertexKey>, Vec<VertexKey>) {
        let mut mesh = LoopMesh::new();
        let mut x = 0.0_f32;
        let mut xs = vec![0.0_f32];
        for c in 0..cols {
            x += 1.0 + 0.5 * c as f32; // widening columns → asymmetric
            xs.push(x);
        }
        let bottom: Vec<VertexKey> = xs
            .iter()
            .map(|&x| mesh.create_vertex(Vec3::new(x, 0.0, 0.0)))
            .collect();
        let top: Vec<VertexKey> = xs
            .iter()
            .map(|&x| mesh.create_vertex(Vec3::new(x, 1.0, 0.0)))
            .collect();

        let edge_of = |mesh: &mut LoopMesh, a: VertexKey, b: VertexKey| {
            mesh.query_edge_exists(a, b)
                .unwrap_or_else(|| mesh.create_edge([a, b], EdgeCreateFlags::NONE))
        };
        for c in 0..cols {
            let ring = [bottom[c], bottom[c + 1], top[c + 1], top[c]];
            let edges: Vec<EdgeKey> = (0..4)
                .map(|i| edge_of(&mut mesh, ring[i], ring[(i + 1) % 4]))
                .collect();
            mesh.face_create(&ring, &edges, FaceCreateFlags::NONE);
        }
        mesh.mesh_normals_update();
        (mesh, bottom, top)
    }

    fn regular_quad() -> (LoopMesh, Vec<VertexKey>) {
        let mut mesh = LoopMesh::new();
        let coords = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        ];
        let face = create_face_from_verts(&mut mesh, &coords);
        let verts = mesh.face_vert_keys(face);
        mesh.mesh_normals_update();
        (mesh, verts)
    }

    #[test]
    fn shared_face_degenerates_to_single_chord() {
        // §7: two opposite corners of one quad → one diagonal cut, identical
        // to connect_verts. 4 V, 4→5 E, 1→2 F.
        let (mut mesh, v) = regular_quad();
        let created = mesh.connect_vert_pair_out(&[v[0], v[2]], &[], &[]);
        assert_eq!(mesh.vert_keys().len(), 4, "no verts created");
        assert_eq!(mesh.edge_keys().len(), 5, "one chord edge created");
        assert_eq!(mesh.face_keys().len(), 2, "quad split into two triangles");
        assert!(mesh.query_edge_exists(v[0], v[2]).is_some());
        assert_eq!(created.len(), 1, "the single chord {{0,2}} is reported");
    }

    #[test]
    fn adjacent_verts_no_cut() {
        // §7: two adjacent corners share an edge → adjacency suppression draws
        // no chord; the existing edge is reported. No topology change.
        let (mut mesh, v) = regular_quad();
        let existing = mesh.query_edge_exists(v[0], v[1]).unwrap();
        let created = mesh.connect_vert_pair_out(&[v[0], v[1]], &[], &[]);
        assert_eq!(mesh.vert_keys().len(), 4);
        assert_eq!(mesh.edge_keys().len(), 4, "no edge added");
        assert_eq!(mesh.face_keys().len(), 1, "no face split");
        assert_eq!(created, vec![existing], "the existing edge is reported");
    }

    #[test]
    fn strip_multi_face_cut() {
        // §2 / §5 / §6: a diagonal cut across a 3-quad strip. The endpoints are
        // the bottom-left and top-right corners, so the fitted plane (§3.3)
        // slices the two interior vertical edges; the route crosses each,
        // splits it into a midpoint vert (§5.2), and connect_verts chords each
        // of the 3 faces. Two crossings → +2 V; two splits +2 E and three
        // chords +3 E → +5 E; +3 F.
        let (mut mesh, bottom, top) = quad_strip(3);
        let v_before = mesh.vert_keys().len();
        let e_before = mesh.edge_keys().len();
        let f_before = mesh.face_keys().len();

        let a = bottom[0];
        let b = top[3];
        let created = mesh.connect_vert_pair_out(&[a, b], &[], &[]);

        assert_eq!(mesh.vert_keys().len(), v_before + 2, "two crossing verts");
        assert_eq!(
            mesh.face_keys().len(),
            f_before + 3,
            "each of the three strip faces is chorded"
        );
        // Two crossing splits (+2 E) and three chords (+3 E).
        assert_eq!(mesh.edge_keys().len(), e_before + 5);
        assert_eq!(created.len(), 3, "three chord edges reported");
    }

    #[test]
    fn disconnected_components_no_cut() {
        // §4.4 / §7: endpoints in two separate quads → no route → no cut.
        let (mut mesh, v0) = regular_quad();
        let coords = [
            Vec3::new(10.0, 0.0, 0.0),
            Vec3::new(11.0, 0.0, 0.0),
            Vec3::new(11.0, 1.0, 0.0),
            Vec3::new(10.0, 1.0, 0.0),
        ];
        let far_face = create_face_from_verts(&mut mesh, &coords);
        let far = mesh.face_vert_keys(far_face);
        mesh.mesh_normals_update();

        let v_before = mesh.vert_keys().len();
        let e_before = mesh.edge_keys().len();
        let f_before = mesh.face_keys().len();

        let created = mesh.connect_vert_pair_out(&[v0[0], far[0]], &[], &[]);
        assert!(created.is_empty(), "no chord on a disconnected pair");
        assert_eq!(mesh.vert_keys().len(), v_before);
        assert_eq!(mesh.edge_keys().len(), e_before);
        assert_eq!(mesh.face_keys().len(), f_before, "mesh unchanged");
    }

    #[test]
    fn faces_exclude_blocks_route() {
        // §3.2 / §4.2: excluding every strip face blocks the only route → no
        // cut; the same strip with no exclude fires the multi-face cut.
        let (mut mesh, bottom, top) = quad_strip(3);
        let faces: Vec<FaceKey> = mesh.face_keys();
        let v_before = mesh.vert_keys().len();
        let e_before = mesh.edge_keys().len();
        let f_before = mesh.face_keys().len();

        let created = mesh.connect_vert_pair_out(&[bottom[0], top[3]], &[], &faces);
        assert!(created.is_empty(), "all faces excluded → no route");
        assert_eq!(mesh.vert_keys().len(), v_before);
        assert_eq!(mesh.edge_keys().len(), e_before);
        assert_eq!(mesh.face_keys().len(), f_before, "strip unchanged");
    }

    #[test]
    fn verts_exclude_keeps_route_off_named_verts() {
        // §3.2: on a flat strip with bottom-row endpoints the natural route
        // walks vert-to-vert along the bottom edge, stepping onto the interior
        // bottom verts. Excluding those interior verts forbids that route: the
        // search must route around them or fail, and in neither case may an
        // excluded vert become a route vert.
        let (mut mesh, bottom, _top) = quad_strip(3);
        let degree_before: Vec<usize> = [bottom[1], bottom[2]]
            .iter()
            .map(|&v| mesh.vert_edge_keys(v).len())
            .collect();

        let _ = mesh.connect_vert_pair_out(&[bottom[0], bottom[3]], &[bottom[1], bottom[2]], &[]);

        // An excluded vert is never stepped onto, so it never gains a chord —
        // its incident-edge count is unchanged.
        for (i, &excluded) in [bottom[1], bottom[2]].iter().enumerate() {
            assert_eq!(
                mesh.vert_edge_keys(excluded).len(),
                degree_before[i],
                "excluded vert gains no chord"
            );
        }
    }

    #[test]
    /// §4.1 / §4.2 no-double-back regression, exercised directly on the
    /// expansion step (the unit where the rule lives, isolated from the §4.3
    /// route-selection it would otherwise be tangled with). A state standing on
    /// an interior crossing edge of the strip has two incident faces; the one it
    /// arrived through must be skipped, so no candidate it yields is reached *via
    /// that same face*. The negative control — the identical state with
    /// `arrived_via = None` — must surface candidates through *both* faces,
    /// proving the skip (not some other filter) is what removes them. This is
    /// the test that fails if the arrived-via field is left unread.
    fn expand_skips_the_arrived_face() {
        let (mesh, bottom, top) = quad_strip(3);
        // A diagonal endpoints' plane (as in `strip_multi_face_cut`) crosses the
        // interior vertical edges; stand the state on one of those crossings.
        let plane = mesh.fit_cutting_plane(bottom[0], top[3]);
        let crossing_edge = mesh
            .query_edge_exists(bottom[1], top[1])
            .expect("interior vertical edge exists");
        let [ea, eb] = mesh.edge_vertices(crossing_edge);
        let fraction = mesh.plane_crossing_fraction(&plane, ea, eb);
        let crossing_point = mesh
            .vertex_position(ea)
            .lerp(mesh.vertex_position(eb), fraction);

        let faces = mesh.edge_face_keys(crossing_edge);
        assert_eq!(faces.len(), 2, "interior edge is shared by two faces");
        let arrived = faces[0];
        let other = faces[1];

        let no_excludes_f: HashSet<FaceKey> = HashSet::new();
        let no_excludes_v: HashSet<VertexKey> = HashSet::new();

        let make_state = |arrived_via: Option<FaceKey>| PathState {
            length: 0.0,
            last_point: crossing_point,
            last_element: RouteElement::Edge(crossing_edge),
            arrived_via,
            link: 0,
        };

        // With the arrived face recorded, none of the candidates come back
        // through it.
        let with_skip = mesh.expand_state(
            &make_state(Some(arrived)),
            &plane,
            &no_excludes_f,
            &no_excludes_v,
            top[3],
        );
        assert!(
            with_skip.iter().all(|c| c.arrived_via != Some(arrived)),
            "no candidate re-enters the arrived face"
        );
        assert!(
            with_skip.iter().any(|c| c.arrived_via == Some(other)),
            "the forward face is still explored"
        );

        // Negative control: without the skip, the arrived face's own crossings
        // re-appear — confirming the rule, not another filter, removed them.
        let without_skip = mesh.expand_state(
            &make_state(None),
            &plane,
            &no_excludes_f,
            &no_excludes_v,
            top[3],
        );
        assert!(
            without_skip.iter().any(|c| c.arrived_via == Some(arrived)),
            "control: the arrived face is re-collected when nothing is skipped"
        );
    }

    #[test]
    fn same_vert_twice_is_noop() {
        let (mut mesh, v) = regular_quad();
        let created = mesh.connect_vert_pair_out(&[v[0], v[0]], &[], &[]);
        assert!(created.is_empty());
        assert_eq!(mesh.face_keys().len(), 1);
    }

    #[test]
    fn wrong_vert_count_is_noop() {
        let (mut mesh, v) = regular_quad();
        assert!(mesh.connect_vert_pair_out(&[v[0]], &[], &[]).is_empty());
        assert!(mesh
            .connect_vert_pair_out(&[v[0], v[1], v[2]], &[], &[])
            .is_empty());
        assert_eq!(mesh.face_keys().len(), 1, "mesh unchanged");
    }
}
