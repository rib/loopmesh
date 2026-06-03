use std::collections::{HashMap, HashSet};

use glam::Vec3;

use crate::loopmesh::*;

/// Below this squared distance from the component centroid a loop vertex is
/// treated as coincident with the centroid and cannot win the outermost-loop
/// test, so a degenerate component never seeds on a zero-length direction.
const OUTERMOST_DIST_SQ_EPSILON: f32 = 1e-12;

impl LoopMesh {
    /// Recalculate face normals: refresh each named face's cached normal from
    /// its corner positions, then re-wind faces so that, within each piece of
    /// surface connected by manifold edges, every face is wound consistently
    /// with its neighbours and the piece points outward.
    ///
    /// The operator walks connectivity across *all* faces of the mesh to build
    /// components and seed the outward test, but only re-winds faces that are
    /// both decided-to-flip and present in `faces`. Faces outside `faces` give
    /// connectivity and seeding context without being disturbed.
    ///
    /// With `inside` set, the outward pass is followed by a winding reversal of
    /// the whole named set, so the consistent surface points inward instead.
    ///
    /// The flood propagates only across manifold edges (radial count 2);
    /// boundary, wire, and non-manifold edges stop it, so faces meeting at a
    /// non-manifold join are left in whatever relative orientation they had.
    pub fn recalc_face_normals(&mut self, faces: &[FaceKey], inside: bool) {
        if faces.is_empty() {
            return;
        }

        // Phase A: refresh every input face's cached normal so the outward
        // seed test reads a normal consistent with the face's current winding.
        for &face in faces {
            self.face_normal_update(face);
        }

        let input_set: HashSet<FaceKey> = faces.iter().copied().collect();

        let mut visited: HashSet<FaceKey> = HashSet::new();
        let mut component: Vec<FaceKey> = Vec::new();
        let mut flip_state: HashMap<FaceKey, bool> = HashMap::new();
        let mut flips: Vec<FaceKey> = Vec::new();

        // Partition all faces into components connected by manifold edges and
        // process each independently.
        let all_faces = self.face_keys();
        for &start in &all_faces {
            if visited.contains(&start) {
                continue;
            }

            self.collect_manifold_component(start, &mut visited, &mut component);

            // A component with no named face has nothing to flip.
            if !component.iter().any(|f| input_set.contains(f)) {
                continue;
            }

            let (seed, seed_flip) = self.outward_seed(&component);

            self.flood_flip_state(seed, seed_flip, &mut flip_state);

            for (&face, &do_flip) in &flip_state {
                if do_flip && input_set.contains(&face) {
                    flips.push(face);
                }
            }
            flip_state.clear();
        }

        // Each flip is the per-face loop reverse plus a normal negate, exactly
        // the surgery `reverse_faces` performs.
        self.reverse_faces(&flips, false);

        // The `inside` variant is recalc-outside composed with a reverse sweep
        // of the same set: the surface stays consistent and now points inward.
        if inside {
            self.reverse_faces(faces, false);
        }
    }

    /// Gather, into `out`, the faces reachable from `start` along manifold
    /// edges (radial count 2). Marks each gathered face in `visited`.
    fn collect_manifold_component(
        &self,
        start: FaceKey,
        visited: &mut HashSet<FaceKey>,
        out: &mut Vec<FaceKey>,
    ) {
        out.clear();
        let mut stack = vec![start];
        visited.insert(start);
        while let Some(face) = stack.pop() {
            out.push(face);
            for loop_key in self.face_loop_keys(face) {
                let edge = self.loop_edge(loop_key);
                if !self.edge_is_manifold(edge) {
                    continue;
                }
                let other_loop = self.loop_radial_next(loop_key);
                let neighbour = self.loop_face(other_loop);
                if visited.insert(neighbour) {
                    stack.push(neighbour);
                }
            }
        }
    }

    /// Flood the per-face flip state across manifold edges from `seed`,
    /// carrying the §4.1 flip-parity payload.
    fn flood_flip_state(
        &self,
        seed: FaceKey,
        seed_flip: bool,
        flip_state: &mut HashMap<FaceKey, bool>,
    ) {
        flip_state.insert(seed, seed_flip);
        let mut stack = vec![seed];
        while let Some(face) = stack.pop() {
            let parent_flip = flip_state[&face];
            for loop_key in self.face_loop_keys(face) {
                let edge = self.loop_edge(loop_key);
                if !self.edge_is_manifold(edge) {
                    continue;
                }
                let other_loop = self.loop_radial_next(loop_key);
                let neighbour = self.loop_face(other_loop);
                if flip_state.contains_key(&neighbour) {
                    continue;
                }
                // Two faces are consistently wound across a shared edge when
                // their loops start at different vertices (the edge is
                // traversed head-to-tail one way by each). Same start vertex
                // means inconsistent, so the neighbour flips relative to the
                // parent.
                let inconsistent = self.loop_vertex(loop_key) == self.loop_vertex(other_loop);
                flip_state.insert(neighbour, parent_flip ^ inconsistent);
                stack.push(neighbour);
            }
        }
    }

    /// True if `edge` has exactly two incident face loops.
    fn edge_is_manifold(&self, edge: EdgeKey) -> bool {
        if edge.is_null() {
            return false;
        }
        let mut count = 0u32;
        for _ in self.edge_radial_iter(edge) {
            count += 1;
            if count > 2 {
                return false;
            }
        }
        count == 2
    }

    /// The outermost-vertex outward test: pick the component's seed face and
    /// decide whether its consistent orientation points inward (so the whole
    /// component must be flipped to point outward).
    ///
    /// Returns `(seed_face, seed_is_flipped)`.
    fn outward_seed(&self, component: &[FaceKey]) -> (FaceKey, bool) {
        let centroid = self.component_centroid(component);

        // Find the loop whose vertex is furthest from the centroid, breaking
        // ties by how radially the loop's incident edges point, then by how
        // strongly the loop corner normal aligns with the outward direction.
        let mut best: Option<LoopWinner> = None;
        for &face in component {
            let face_normal = self.face_normal(face);
            for loop_key in self.face_loop_keys(face) {
                let candidate = self.evaluate_loop(loop_key, face, face_normal, centroid);
                if candidate.dist_sq <= OUTERMOST_DIST_SQ_EPSILON {
                    continue;
                }
                if best.as_ref().is_none_or(|b| candidate.beats(b)) {
                    best = Some(candidate);
                }
            }
        }

        let winner = match best {
            Some(winner) => winner,
            // Every loop sits at the centroid (a degenerate component): no
            // robust outward direction, so seed unflipped.
            None => return (component[0], false),
        };

        // If the outermost corner normal points back toward the centroid the
        // seed face is oriented inward and the component must be flipped.
        let seed_flip = winner.outward_dot < 0.0;
        (winner.face, seed_flip)
    }

    /// Area-weighted centroid of the component's faces (median centre of each
    /// face scaled by its area, divided by the total area). Falls back to the
    /// unweighted median-of-medians when the total area is degenerate.
    fn component_centroid(&self, component: &[FaceKey]) -> Vec3 {
        let mut weighted = Vec3::ZERO;
        let mut total_area = 0.0f32;
        let mut median_sum = Vec3::ZERO;
        for &face in component {
            let (median, area) = self.face_median_and_area(face);
            weighted += median * area;
            total_area += area;
            median_sum += median;
        }
        if total_area > 0.0 {
            weighted / total_area
        } else {
            median_sum / component.len() as f32
        }
    }

    /// The face's median centre (average of its corner positions) and its area.
    fn face_median_and_area(&self, face: FaceKey) -> (Vec3, f32) {
        let mut corners: smallvec::SmallVec<[Vec3; 8]> = smallvec::SmallVec::new();
        for loop_key in self.face_loop_keys(face) {
            corners.push(self.vertex_position(self.loop_vertex(loop_key)));
        }
        let mut median = Vec3::ZERO;
        for &corner in &corners {
            median += corner;
        }
        median /= corners.len() as f32;
        let area = LoopMesh::calc_poly_area(&corners);
        (median, area)
    }

    /// Evaluate one loop for the outermost-vertex test, computing its squared
    /// distance from the centroid and the two tie-break keys.
    fn evaluate_loop(
        &self,
        loop_key: LoopKey,
        face: FaceKey,
        face_normal: Vec3,
        centroid: Vec3,
    ) -> LoopWinner {
        let vert = self.loop_vertex(loop_key);
        let position = self.vertex_position(vert);
        let to_vert = position - centroid;
        let dist_sq = to_vert.length_squared();

        let prev_pos = self.vertex_position(self.loop_vertex(self.loop_prev(loop_key)));
        let next_pos = self.vertex_position(self.loop_vertex(self.loop_next(loop_key)));
        let edge_to_next = next_pos - position;
        let edge_to_prev = prev_pos - position;

        let direction = to_vert.normalize_or_zero();
        let edge_dot = f32::max(
            direction.dot(edge_to_next.normalize_or_zero()).abs(),
            direction.dot(edge_to_prev.normalize_or_zero()).abs(),
        );

        // Corner normal: cross of the two incident edge directions, oriented to
        // agree with the face normal.
        let mut corner_normal = edge_to_next.cross(edge_to_prev);
        if corner_normal.dot(face_normal) < 0.0 {
            corner_normal = -corner_normal;
        }
        let corner_normal = corner_normal.normalize_or_zero();
        let normal_dot = direction.dot(corner_normal).abs();

        LoopWinner {
            face,
            dist_sq,
            edge_dot,
            normal_dot,
            outward_dot: corner_normal.dot(to_vert),
        }
    }
}

/// A candidate for the outermost-vertex test, ranked by squared distance from
/// the centroid then the two geometric tie-break keys.
struct LoopWinner {
    face: FaceKey,
    dist_sq: f32,
    edge_dot: f32,
    normal_dot: f32,
    /// Corner normal (face-oriented) dotted with the centroid→vertex direction;
    /// negative means the corner points inward.
    outward_dot: f32,
}

impl LoopWinner {
    fn beats(&self, other: &LoopWinner) -> bool {
        if self.dist_sq != other.dist_sq {
            return self.dist_sq > other.dist_sq;
        }
        if self.edge_dot != other.edge_dot {
            return self.edge_dot > other.edge_dot;
        }
        self.normal_dot > other.normal_dot
    }
}

#[cfg(test)]
mod tests {
    use glam::Vec3;

    use crate::loopmesh::*;

    /// A closed octahedron built with a chosen winding for one face so tests
    /// can invert it. Returns all six faces and the apex/equator vertices.
    ///
    /// The simplest closed orientable solid to wind deliberately is a square
    /// bipyramid (octahedron): four top triangles and four bottom triangles
    /// around an equator square, with a top and bottom apex.
    struct Octahedron {
        faces: Vec<FaceKey>,
    }

    fn build_octahedron(mesh: &mut LoopMesh) -> Octahedron {
        // Equator square in the z=0 plane, apexes on +z / -z.
        let e0 = mesh.create_vertex(Vec3::new(1.0, 0.0, 0.0));
        let e1 = mesh.create_vertex(Vec3::new(0.0, 1.0, 0.0));
        let e2 = mesh.create_vertex(Vec3::new(-1.0, 0.0, 0.0));
        let e3 = mesh.create_vertex(Vec3::new(0.0, -1.0, 0.0));
        let top = mesh.create_vertex(Vec3::new(0.0, 0.0, 1.0));
        let bot = mesh.create_vertex(Vec3::new(0.0, 0.0, -1.0));

        let mut faces = Vec::new();
        // Top fan: wound CCW seen from outside (+z) -> outward normals.
        let equator = [e0, e1, e2, e3];
        for i in 0..4 {
            let a = equator[i];
            let b = equator[(i + 1) % 4];
            faces.push(make_tri(mesh, a, b, top));
        }
        // Bottom fan: wound so the normal points down (outside is -z).
        for i in 0..4 {
            let a = equator[i];
            let b = equator[(i + 1) % 4];
            faces.push(make_tri(mesh, b, a, bot));
        }
        Octahedron { faces }
    }

    fn make_tri(mesh: &mut LoopMesh, a: VertexKey, b: VertexKey, c: VertexKey) -> FaceKey {
        let ab = edge_between(mesh, a, b);
        let bc = edge_between(mesh, b, c);
        let ca = edge_between(mesh, c, a);
        mesh.face_create(&[a, b, c], &[ab, bc, ca], FaceCreateFlags::NONE)
    }

    /// Find an existing edge between `a` and `b`, or create one.
    fn edge_between(mesh: &mut LoopMesh, a: VertexKey, b: VertexKey) -> EdgeKey {
        for e in mesh.edge_keys() {
            let [v0, v1] = mesh.edge_vertices(e);
            if (v0 == a && v1 == b) || (v0 == b && v1 == a) {
                return e;
            }
        }
        mesh.create_edge([a, b], EdgeCreateFlags::NONE)
    }

    /// True if every pair of faces sharing a manifold edge is wound
    /// consistently across it (their loops on the shared edge start at
    /// different vertices).
    fn all_consistent(mesh: &LoopMesh, faces: &[FaceKey]) -> bool {
        for &face in faces {
            for loop_key in mesh.face_loop_keys(face) {
                let edge = mesh.loop_edge(loop_key);
                if mesh.edge_face_keys(edge).len() != 2 {
                    continue;
                }
                let other = mesh.loop_radial_next(loop_key);
                if mesh.loop_vertex(loop_key) == mesh.loop_vertex(other) {
                    return false;
                }
            }
        }
        true
    }

    /// Every face's cached normal must point away from the solid's centre —
    /// the "outside" convention for a star-convex solid centred at the origin.
    fn all_point_outward(mesh: &LoopMesh, faces: &[FaceKey]) -> bool {
        for &face in faces {
            let mut centre = Vec3::ZERO;
            let verts = mesh.face_vert_keys(face);
            for &v in &verts {
                centre += mesh.vertex_position(v);
            }
            centre /= verts.len() as f32;
            if mesh.face_normal(face).dot(centre) <= 0.0 {
                return false;
            }
        }
        true
    }

    fn build_two_quads_inconsistent(mesh: &mut LoopMesh) -> (FaceKey, FaceKey, EdgeKey) {
        let v0 = mesh.create_vertex(Vec3::new(0.0, 0.0, 0.0));
        let v1 = mesh.create_vertex(Vec3::new(1.0, 0.0, 0.0));
        let v2 = mesh.create_vertex(Vec3::new(1.0, 1.0, 0.0));
        let v3 = mesh.create_vertex(Vec3::new(0.0, 1.0, 0.0));
        let v4 = mesh.create_vertex(Vec3::new(2.0, 0.0, 0.0));
        let v5 = mesh.create_vertex(Vec3::new(2.0, 1.0, 0.0));

        let e01 = mesh.create_edge([v0, v1], EdgeCreateFlags::NONE);
        let e12 = mesh.create_edge([v1, v2], EdgeCreateFlags::NONE);
        let e23 = mesh.create_edge([v2, v3], EdgeCreateFlags::NONE);
        let e30 = mesh.create_edge([v3, v0], EdgeCreateFlags::NONE);
        let e14 = mesh.create_edge([v1, v4], EdgeCreateFlags::NONE);
        let e45 = mesh.create_edge([v4, v5], EdgeCreateFlags::NONE);
        let e52 = mesh.create_edge([v5, v2], EdgeCreateFlags::NONE);

        let left = mesh.face_create(&[v0, v1, v2, v3], &[e01, e12, e23, e30], FaceCreateFlags::NONE);
        // Wound the *wrong* way: traverses the shared edge 1->2 same as left.
        let right =
            mesh.face_create(&[v1, v2, v5, v4], &[e12, e52, e45, e14], FaceCreateFlags::NONE);
        (left, right, e12)
    }

    /// Headline: a closed solid with exactly one face wound inward is corrected
    /// so the whole solid points outward and is consistently wound.
    #[test]
    fn one_inverted_face_corrected() {
        let mut mesh = LoopMesh::new();
        let oct = build_octahedron(&mut mesh);
        mesh.calc_face_normals();

        // Invert one face so it disagrees with its neighbours.
        mesh.reverse_faces(&[oct.faces[0]], false);
        assert!(!all_consistent(&mesh, &oct.faces), "fixture must start inconsistent");

        mesh.recalc_face_normals(&oct.faces, false);
        mesh.mesh_normals_update();

        assert!(all_consistent(&mesh, &oct.faces), "all faces consistently wound");
        assert!(all_point_outward(&mesh, &oct.faces), "whole solid points outward");
    }

    /// Already-consistent, already-outward solid: recalc-outside is a winding
    /// no-op (every face keeps its corner sequence).
    #[test]
    fn consistent_solid_is_winding_noop() {
        let mut mesh = LoopMesh::new();
        let oct = build_octahedron(&mut mesh);
        mesh.calc_face_normals();
        assert!(all_consistent(&mesh, &oct.faces));
        assert!(all_point_outward(&mesh, &oct.faces), "fixture starts outward");

        let before: Vec<Vec<VertexKey>> =
            oct.faces.iter().map(|&f| mesh.face_vert_keys(f)).collect();

        mesh.recalc_face_normals(&oct.faces, false);
        mesh.mesh_normals_update();

        for (i, &face) in oct.faces.iter().enumerate() {
            assert_eq!(mesh.face_vert_keys(face), before[i], "face {i} winding unchanged");
        }
        assert!(all_point_outward(&mesh, &oct.faces));
    }

    /// Outward seed decision: the whole solid is initially wound *inward*
    /// (consistent but inverted). The seed test must detect this and flip the
    /// entire component back to outward.
    #[test]
    fn outward_seed_flips_uniformly_inward_solid() {
        let mut mesh = LoopMesh::new();
        let oct = build_octahedron(&mut mesh);
        mesh.calc_face_normals();
        // Reverse every face -> consistent but pointing inward.
        mesh.reverse_faces(&oct.faces, false);
        assert!(all_consistent(&mesh, &oct.faces), "still consistent after global flip");
        assert!(!all_point_outward(&mesh, &oct.faces), "now points inward");

        mesh.recalc_face_normals(&oct.faces, false);
        mesh.mesh_normals_update();

        assert!(all_consistent(&mesh, &oct.faces));
        assert!(all_point_outward(&mesh, &oct.faces), "seed test re-orients outward");
    }

    /// Open inconsistent pair: two quads sharing an edge, the second wound the
    /// wrong way. Recalc makes them agree across the shared edge.
    #[test]
    fn open_inconsistent_pair_made_consistent() {
        let mut mesh = LoopMesh::new();
        let (left, right, _) = build_two_quads_inconsistent(&mut mesh);
        mesh.calc_face_normals();
        let faces = [left, right];
        assert!(!all_consistent(&mesh, &faces), "fixture starts inconsistent");

        mesh.recalc_face_normals(&faces, false);
        mesh.mesh_normals_update();

        assert!(all_consistent(&mesh, &faces), "pair now agrees across shared edge");
        // The two coplanar quads end up with normals pointing the same way.
        assert!(
            (mesh.face_normal(left) - mesh.face_normal(right)).length() < 1e-5,
            "consistent coplanar faces share a normal direction"
        );
    }

    /// Input-set restriction (§5.3): naming a strict subset re-winds only the
    /// named faces. An unnamed inverted face stays inverted (it is used for
    /// connectivity/seeding but never flipped); a named inverted face that the
    /// flood marks is corrected.
    #[test]
    fn input_set_restricts_flips_to_named_faces() {
        let mut mesh = LoopMesh::new();
        let oct = build_octahedron(&mut mesh);
        mesh.calc_face_normals();

        // Invert two faces; only name one of them for recalculation.
        let named_inverted = oct.faces[0];
        let unnamed_inverted = oct.faces[1];
        mesh.reverse_faces(&[named_inverted, unnamed_inverted], false);

        let unnamed_before = mesh.face_vert_keys(unnamed_inverted);
        let named_before = mesh.face_vert_keys(named_inverted);

        // Name every face EXCEPT the second inverted one.
        let named: Vec<FaceKey> = oct
            .faces
            .iter()
            .copied()
            .filter(|&f| f != unnamed_inverted)
            .collect();
        mesh.recalc_face_normals(&named, false);
        mesh.mesh_normals_update();

        // The unnamed inverted face is untouched.
        assert_eq!(
            mesh.face_vert_keys(unnamed_inverted),
            unnamed_before,
            "unnamed face is not re-wound",
        );
        // The named inverted face was flipped back to agree with the component.
        assert_ne!(
            mesh.face_vert_keys(named_inverted),
            named_before,
            "named inverted face is corrected",
        );
    }

    /// `inside` variant: recalc with `inside = true` on an already-outward
    /// solid flips every face to point inward while staying consistent.
    #[test]
    fn inside_variant_points_inward() {
        let mut mesh = LoopMesh::new();
        let oct = build_octahedron(&mut mesh);
        mesh.calc_face_normals();
        assert!(all_point_outward(&mesh, &oct.faces));

        mesh.recalc_face_normals(&oct.faces, true);
        mesh.mesh_normals_update();

        assert!(all_consistent(&mesh, &oct.faces), "inside stays consistent");
        // Every face now points inward (toward the centre).
        for &face in &oct.faces {
            let mut centre = Vec3::ZERO;
            let verts = mesh.face_vert_keys(face);
            for &v in &verts {
                centre += mesh.vertex_position(v);
            }
            centre /= verts.len() as f32;
            assert!(mesh.face_normal(face).dot(centre) < 0.0, "face points inward");
        }
    }

    /// `inside` of an already-outward solid equals reverse-faces-all of that
    /// solid: same windings, same normals.
    #[test]
    fn inside_equals_reverse_all_of_outside() {
        let mut a = LoopMesh::new();
        let oct_a = build_octahedron(&mut a);
        a.calc_face_normals();
        a.recalc_face_normals(&oct_a.faces, true);
        a.mesh_normals_update();

        let mut b = LoopMesh::new();
        let oct_b = build_octahedron(&mut b);
        b.calc_face_normals();
        // The fixture is already consistent + outward, so recalc-outside is a
        // no-op and `inside` reduces to a plain reverse-all.
        b.reverse_faces(&oct_b.faces, false);
        b.mesh_normals_update();

        for i in 0..oct_a.faces.len() {
            assert_eq!(
                a.face_vert_keys(oct_a.faces[i]),
                b.face_vert_keys(oct_b.faces[i]),
                "face {i} winding matches reverse-all",
            );
            assert!(
                (a.face_normal(oct_a.faces[i]) - b.face_normal(oct_b.faces[i])).length() < 1e-5,
                "face {i} normal matches reverse-all",
            );
        }
    }

    /// Non-manifold edge blocks propagation: three quads meeting at a shared
    /// (radial-3) edge. The flood does not cross the non-manifold edge, so a
    /// page on the far side keeps its own orientation rather than being forced
    /// consistent across the impossible join.
    #[test]
    fn nonmanifold_edge_blocks_propagation() {
        let mut mesh = LoopMesh::new();
        // Spine edge (s0, s1); three pages fan off it.
        let s0 = mesh.create_vertex(Vec3::new(0.0, 0.0, 0.0));
        let s1 = mesh.create_vertex(Vec3::new(0.0, 1.0, 0.0));
        let spine = mesh.create_edge([s0, s1], EdgeCreateFlags::NONE);

        let mut pages = Vec::new();
        let page_dirs = [
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(-1.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
        ];
        for dir in page_dirs {
            let a = mesh.create_vertex(Vec3::new(0.0, 0.0, 0.0) + dir);
            let b = mesh.create_vertex(Vec3::new(0.0, 1.0, 0.0) + dir);
            let e_s1_b = mesh.create_edge([s1, b], EdgeCreateFlags::NONE);
            let e_ba = mesh.create_edge([b, a], EdgeCreateFlags::NONE);
            let e_a_s0 = mesh.create_edge([a, s0], EdgeCreateFlags::NONE);
            let face = mesh.face_create(
                &[s0, s1, b, a],
                &[spine, e_s1_b, e_ba, e_a_s0],
                FaceCreateFlags::NONE,
            );
            pages.push(face);
        }
        mesh.calc_face_normals();

        // The spine edge is non-manifold (radial 3).
        assert_eq!(mesh.edge_face_keys(spine).len(), 3, "spine is radial-3");

        let before: Vec<Vec<VertexKey>> = pages.iter().map(|&f| mesh.face_vert_keys(f)).collect();

        // Recalc must not panic and must not force a consistent winding across
        // the non-manifold spine: every page is its own singleton component
        // (joined to the others only via the non-manifold spine), so the flood
        // never crosses it. With all pages already locally consistent (each is
        // a lone face) the windings are left untouched.
        mesh.recalc_face_normals(&pages, false);
        mesh.mesh_normals_update();

        for (i, &face) in pages.iter().enumerate() {
            assert_eq!(
                mesh.face_vert_keys(face),
                before[i],
                "page {i} not re-wound across non-manifold join",
            );
        }
    }
}
