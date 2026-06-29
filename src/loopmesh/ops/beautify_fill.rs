//! The standalone **beautify-fill** operator: given a triangle patch and a set
//! of rotatable interior edges, repeatedly rotate the shared diagonals toward a
//! better-shaped (more Delaunay-like) triangulation until no rotation improves
//! the patch under the chosen beauty metric.
//!
//! It creates and destroys no geometry — the vertex set, the patch boundary,
//! and the triangle *count* are invariant; only *which* interior diagonal is
//! present at each shared edge changes
//! (`radial-mesh-guides/operators/BEAUTIFY_FILL.md`).
//!
//! This layer sits on top of the single-edge [`LoopMesh::edge_rotate`] primitive
//! (the §2 rotation surgery, documented in
//! `radial-mesh-guides/operators/ROTATE_EDGES.md` §2) and the existing AREA cost
//! kernel ([`crate::triangulate::beautify_quad_cost`], BEAUTIFY_FILL.md §1.1 /
//! §4.2). What beautify-fill adds is:
//!
//! - the §3 edge-eligibility gates (patch-triangle membership, manifold-ness,
//!   the no-double guard, and the always-on degenerate lock the cost kernel
//!   consumes),
//! - the §5.1 greedy min-cost worklist that drains most-beneficial first,
//! - the §5.2 per-edge visited-state cycle guard, and
//! - the §5.3 local four-edge re-score after each rotation,
//! - the §1 / §7 `geom.out` slot (rotated edges + their two flanking faces).

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};

use glam::Vec3;

use crate::loopmesh::*;
use crate::triangulate::beautify_quad_cost;

/// The "never rotate" cost sentinel shared by both metrics (BEAUTIFY_FILL.md §4):
/// a degenerate / flip-producing candidate scores this so the descent never
/// enqueues it. Matches the AREA kernel's `f32::MAX` (`beautify_quad_cost`).
const NEVER_ROTATE: f32 = f32::MAX;

/// The §4.1 ANGLE cost of rotating the cyclic quad `(v0, v1, v2, v3)` from its
/// current diagonal `v0-v2` to the candidate diagonal `v1-v3`. The metric
/// measures the dihedral angle between the two flanking triangles' normals and
/// prefers the diagonal that folds the pair *flatter*; the cost is therefore
/// `angle_after − angle_before`, negative when the candidate is the flatter,
/// more Delaunay-like choice. Computed on the true 3D triangle normals — there
/// is no axis-dominant 2D projection here (unlike the AREA kernel). A candidate
/// triangle with a zero-length (degenerate, zero-area) normal scores
/// [`NEVER_ROTATE`] so a rotation *into* a fold is never enqueued (§3 gate 5,
/// §4.1 candidate zero-normal sentinel); the current state's degeneracy is left
/// to the §3 eligibility guard.
fn beautify_quad_cost_angle(v0: Vec3, v1: Vec3, v2: Vec3, v3: Vec3) -> f32 {
    // Current diagonal v0-v2: flanking triangles (v0,v3,v2) and (v0,v2,v1).
    let current_normal_a = (v3 - v0).cross(v2 - v0);
    let current_normal_b = (v2 - v0).cross(v1 - v0);
    let angle_before = dihedral_between(current_normal_a, current_normal_b);

    // Candidate diagonal v1-v3: flanking triangles (v1,v0,v3) and (v1,v3,v2).
    let candidate_normal_c = (v0 - v1).cross(v3 - v1);
    let candidate_normal_d = (v3 - v1).cross(v2 - v1);
    if candidate_normal_c.length_squared() == 0.0 || candidate_normal_d.length_squared() == 0.0 {
        return NEVER_ROTATE;
    }
    let angle_after = dihedral_between(candidate_normal_c, candidate_normal_d);

    angle_after - angle_before
}

/// The dihedral angle (radians, in `[0, π]`) between two triangle normals,
/// each normalised first. A zero-length input normalises to zero, yielding a
/// `π/2` reading — the ANGLE metric only treats a zero-length *candidate*
/// normal as degenerate (handled by the caller before this is reached for the
/// candidate pair).
fn dihedral_between(normal_a: Vec3, normal_b: Vec3) -> f32 {
    let unit_a = normal_a.normalize_or_zero();
    let unit_b = normal_b.normalize_or_zero();
    unit_a.dot(unit_b).clamp(-1.0, 1.0).acos()
}

/// Which beauty metric decides a rotation (BEAUTIFY_FILL.md §4). `Area` is the
/// zero-init default (an unset `method` runs the area-over-perimeter test);
/// `Angle` is the explicit opt-in dihedral-angle metric.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum BeautifyMethod {
    #[default]
    Area,
    Angle,
}

/// The two knobs of the beautify-fill BMOP (BEAUTIFY_FILL.md §1). The element
/// buffers (`faces`, `edges`) are passed as separate arguments to
/// [`LoopMesh::beautify_fill`].
#[derive(Clone, Copy, Debug, Default)]
pub struct BeautifyFillOptions {
    /// Confine rotation to tag-seam edges (apexes differing in tag). Out of
    /// scope this cycle.
    pub use_restrict_tag: bool,
    /// The beauty metric. Defaults to [`BeautifyMethod::Area`].
    pub method: BeautifyMethod,
}

/// The `geom.out` slot (BEAUTIFY_FILL.md §1 / §7): every edge that was actually
/// rotated, plus the two faces flanking each such edge. Reported as an
/// order-insensitive set — each rotated edge appears once and each flanking
/// face appears once.
#[derive(Clone, Debug, Default)]
pub struct BeautifyFillResult {
    edges: Vec<EdgeKey>,
    faces: Vec<FaceKey>,
}

impl BeautifyFillResult {
    /// True when nothing rotated (an already-optimal patch).
    pub fn is_empty(&self) -> bool {
        self.edges.is_empty()
    }
    /// The number of edges that rotated.
    pub fn len(&self) -> usize {
        self.edges.len()
    }
    /// The rotated edges (the new diagonal of each rotation).
    pub fn edges(&self) -> std::slice::Iter<'_, EdgeKey> {
        self.edges.iter()
    }
    /// The faces flanking the rotated edges.
    pub fn faces(&self) -> std::slice::Iter<'_, FaceKey> {
        self.faces.iter()
    }
}

/// The §5.2 combinatorial signature of a diagonal: the unordered endpoint pair
/// together with the unordered opposite-corner (apex) pair. Rotating an edge
/// swaps the two roles, so the post-rotation signature is [`Self::swapped`].
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct EdgeState {
    endpoints: (u32, u32),
    apexes: (u32, u32),
}

impl EdgeState {
    /// The signature this diagonal would land in if rotated — endpoints and
    /// apexes trade roles (§5.2 `rotated_state`).
    fn swapped(self) -> Self {
        EdgeState {
            endpoints: self.apexes,
            apexes: self.endpoints,
        }
    }
}

fn sorted_pair(a: VertexKey, b: VertexKey) -> (u32, u32) {
    let (x, y) = (a.0, b.0);
    if x <= y {
        (x, y)
    } else {
        (y, x)
    }
}

/// A worklist entry. The §5.1 min-cost heap is realised as a max-heap on
/// `priority = -cost`, so the most-negative (most beneficial) cost pops first.
/// `cost` is carried so a stale entry — one whose edge was re-scored to a
/// different cost since this entry was pushed — can be detected and skipped at
/// pop time (lazy deletion).
#[derive(Clone, Copy)]
struct HeapEntry {
    edge: EdgeKey,
    cost: f32,
    priority: f64,
}

impl PartialEq for HeapEntry {
    fn eq(&self, other: &Self) -> bool {
        self.priority == other.priority
    }
}
impl Eq for HeapEntry {}
impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        self.priority.total_cmp(&other.priority)
    }
}

impl LoopMesh {
    /// Rotate the interior diagonals of a triangle patch toward a better-shaped
    /// triangulation (BEAUTIFY_FILL.md). `faces` is the patch (non-triangles
    /// ignored, §3 gate 3); `edges` is the set of edges *allowed* to rotate
    /// (any edge omitted is a constraint edge and is never touched, §6.1).
    /// Returns the `geom.out` set: every rotated edge plus its two flanking
    /// faces (§7). An already-optimal patch rotates nothing and returns an
    /// empty result (§8).
    pub fn beautify_fill(
        &mut self,
        faces: &[FaceKey],
        edges: &[EdgeKey],
        options: BeautifyFillOptions,
    ) -> BeautifyFillResult {
        if options.use_restrict_tag {
            unimplemented!(
                "beautify_fill: use_restrict_tag not yet implemented (BEAUTIFY_FILL.md §6.2)"
            );
        }

        let mut result = BeautifyFillResult::default();

        // §3 gate 3 — only triangle members of the patch participate. The set
        // is kept current across rotations (a rotation replaces two patch
        // triangles with two new patch triangles).
        let mut patch: HashSet<FaceKey> = faces
            .iter()
            .copied()
            .filter(|&face| self.face_len(face) == 3)
            .collect();
        // The rotatable set grows as diagonals rotate: a rotated edge keeps its
        // eligibility under its own visited-state guard (§5.3), but the rotation
        // mints a fresh edge identity, so that new key must inherit membership
        // — otherwise a neighbour's later re-score could never re-admit it.
        let mut rotatable: HashSet<EdgeKey> = edges.iter().copied().collect();

        // §5.2 visited-state sets, keyed by the *current* edge identity and
        // transferred when the edge rotates to a new key.
        let mut visited: HashMap<EdgeKey, HashSet<EdgeState>> = HashMap::new();
        // The cost currently representing each edge in the heap (the lazy-
        // deletion marker for §5.3 re-scoring).
        let mut active: HashMap<EdgeKey, f32> = HashMap::new();
        let mut heap: BinaryHeap<HeapEntry> = BinaryHeap::new();

        // §5.1 seed — every rotatable edge that passes the §3 gates and scores
        // a strictly negative (improving) cost.
        for &edge in edges {
            if active.contains_key(&edge) {
                continue;
            }
            if let Some(cost) = self.beautify_edge_cost(edge, &patch, options.method) {
                if cost < 0.0 {
                    active.insert(edge, cost);
                    heap.push(HeapEntry {
                        edge,
                        cost,
                        priority: -(cost as f64),
                    });
                }
            }
        }

        // §5.1 descend.
        while let Some(entry) = heap.pop() {
            let edge = entry.edge;
            // Lazy deletion — drop a stale entry whose cost no longer matches
            // the edge's live heap cost.
            match active.get(&edge) {
                Some(&live_cost) if live_cost == entry.cost => {}
                _ => continue,
            }
            active.remove(&edge);

            let Some(signature) = self.beautify_edge_signature(edge) else {
                continue;
            };
            // §5.2 — refuse a rotation that would re-enter a state this diagonal
            // has already occupied.
            if visited
                .get(&edge)
                .is_some_and(|states| states.contains(&signature.swapped()))
            {
                continue;
            }
            let Some((old_face_a, old_face_b)) = self.beautify_flanking_faces(edge) else {
                continue;
            };

            let Some(new_edge) = self.edge_rotate(edge, true) else {
                // The primitive refused (e.g. its no-double guard) — leave the
                // edge in place (§3 gate 4 / §8).
                continue;
            };

            // §5.2 — carry the diagonal's visited-state set across the identity
            // change and record the state just left, so a later rotation cannot
            // ping-pong back into it.
            let mut states = visited.remove(&edge).unwrap_or_default();
            states.insert(signature);
            visited.insert(new_edge, states);

            // Transfer rotatable membership across the identity change (§5.3),
            // so the rotated diagonal can itself be re-admitted by a later
            // neighbour rotation.
            rotatable.insert(new_edge);

            // Keep the patch set current: the two new triangles replace the two
            // old ones.
            patch.remove(&old_face_a);
            patch.remove(&old_face_b);
            let (new_face_a, new_face_b) = self
                .beautify_flanking_faces(new_edge)
                .expect("rotated edge is 2-manifold");
            patch.insert(new_face_a);
            patch.insert(new_face_b);

            // §1 / §7 geom.out — report the *currently present* set of rotated
            // diagonals and their current flanking faces. A diagonal rotated
            // more than once supersedes its earlier key rather than appearing
            // twice (replace, don't append), and the two faces this rotation
            // re-corners replace the keys it consumed.
            match result.edges.iter().position(|&recorded| recorded == edge) {
                Some(slot) => result.edges[slot] = new_edge,
                None => result.edges.push(new_edge),
            }
            result
                .faces
                .retain(|&face| face != old_face_a && face != old_face_b);
            for face in [new_face_a, new_face_b] {
                if !result.faces.contains(&face) {
                    result.faces.push(face);
                }
            }

            // §5.3 — only the four boundary edges of the flanking quad can have
            // changed cost; re-score exactly those.
            for neighbour in self.beautify_quad_boundary_edges(new_edge) {
                if !rotatable.contains(&neighbour) {
                    continue;
                }
                let admit = self
                    .beautify_edge_cost(neighbour, &patch, options.method)
                    .and_then(|cost| {
                    if cost >= 0.0 {
                        return None;
                    }
                    let sig = self.beautify_edge_signature(neighbour)?;
                    let blocked = visited
                        .get(&neighbour)
                        .is_some_and(|states| states.contains(&sig.swapped()));
                    if blocked {
                        None
                    } else {
                        Some(cost)
                    }
                });
                match admit {
                    Some(cost) => {
                        active.insert(neighbour, cost);
                        heap.push(HeapEntry {
                            edge: neighbour,
                            cost,
                            priority: -(cost as f64),
                        });
                    }
                    None => {
                        active.remove(&neighbour);
                    }
                }
            }
        }

        result
    }

    /// The §4 beauty cost of rotating `edge` under `method`, or `None` if `edge`
    /// fails a §3 eligibility gate (non-manifold, a flanking face outside the
    /// triangle patch, or the no-double guard). A negative cost means rotating
    /// improves the patch. The always-on degenerate lock (§3 gate 5) is honoured
    /// by both metrics: the AREA kernel takes it as a flag and the ANGLE kernel
    /// bakes it in via its candidate zero-normal check — either way a degenerate
    /// candidate scores the [`NEVER_ROTATE`] sentinel and is never enqueued.
    fn beautify_edge_cost(
        &self,
        edge: EdgeKey,
        patch: &HashSet<FaceKey>,
        method: BeautifyMethod,
    ) -> Option<f32> {
        // §3 gate 2 — manifold, with a well-defined opposite diagonal.
        let (apex_a, apex_c) = self.edge_rotate_diagonal(edge, true)?;
        // §3 gate 3 — both flanking faces are members of the triangle patch.
        let (face_a, face_b) = self.beautify_flanking_faces(edge)?;
        if !patch.contains(&face_a) || !patch.contains(&face_b) {
            return None;
        }
        // §3 gate 4 — the no-double guard: refuse when the candidate diagonal
        // already exists as an edge.
        if self.query_edge_exists(apex_a, apex_c).is_some() {
            return None;
        }

        // §4 — score the cyclic quad (e0, apex_a, e1, apex_c): existing diagonal
        // e0-e1, candidate diagonal apex_a-apex_c. The metric selector picks the
        // ordering (§4.1 ANGLE: angle_after − angle_before; §4.2 AREA:
        // fac_before − fac_after).
        let [e0, e1] = self.edges.get(edge).vertices;
        let (pos_e0, pos_apex_a, pos_e1, pos_apex_c) = (
            self.vertex_position(e0),
            self.vertex_position(apex_a),
            self.vertex_position(e1),
            self.vertex_position(apex_c),
        );
        let cost = match method {
            BeautifyMethod::Area => {
                beautify_quad_cost(pos_e0, pos_apex_a, pos_e1, pos_apex_c, true)
            }
            BeautifyMethod::Angle => {
                beautify_quad_cost_angle(pos_e0, pos_apex_a, pos_e1, pos_apex_c)
            }
        };
        Some(cost)
    }

    /// The §5.2 signature of `edge` in its current state, or `None` if `edge`
    /// has no well-defined flanking quad.
    fn beautify_edge_signature(&self, edge: EdgeKey) -> Option<EdgeState> {
        let (apex_a, apex_c) = self.edge_rotate_diagonal(edge, true)?;
        let [e0, e1] = self.edges.get(edge).vertices;
        Some(EdgeState {
            endpoints: sorted_pair(e0, e1),
            apexes: sorted_pair(apex_a, apex_c),
        })
    }

    /// The two faces on `edge`'s radial cycle, or `None` if it is not
    /// 2-manifold.
    fn beautify_flanking_faces(&self, edge: EdgeKey) -> Option<(FaceKey, FaceKey)> {
        let mut radial = self.edge_radial_iter(edge);
        let loop_a = radial.next()?;
        let loop_b = radial.next()?;
        if radial.next().is_some() {
            return None;
        }
        let face_a = self.loops.get(loop_a).face;
        let face_b = self.loops.get(loop_b).face;
        if face_a == face_b {
            return None;
        }
        Some((face_a, face_b))
    }

    /// The four boundary edges of `edge`'s flanking quad (the §2 edges
    /// `v1-v2`, `v2-v3`, `v3-v4`, `v4-v1`) — the only edges whose cost a
    /// rotation of `edge` can change (§5.3).
    fn beautify_quad_boundary_edges(&self, edge: EdgeKey) -> Vec<EdgeKey> {
        let mut out: Vec<EdgeKey> = Vec::with_capacity(4);
        let Some((face_a, face_b)) = self.beautify_flanking_faces(edge) else {
            return out;
        };
        for face in [face_a, face_b] {
            let start = self.faces.get(face).first_edge_loop;
            for face_loop in self.face_loop_iter(start) {
                let boundary_edge = self.loops.get(face_loop).edge;
                if boundary_edge != edge && !out.contains(&boundary_edge) {
                    out.push(boundary_edge);
                }
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use glam::Vec3;

    use super::sorted_pair;
    use crate::loopmesh::*;

    fn edge_between(mesh: &LoopMesh, a: VertexKey, b: VertexKey) -> EdgeKey {
        mesh.query_edge_exists(a, b).expect("edge must exist")
    }

    fn tri(mesh: &mut LoopMesh, vs: [VertexKey; 3]) -> FaceKey {
        let es: Vec<EdgeKey> = (0..3)
            .map(|i| mesh.create_edge([vs[i], vs[(i + 1) % 3]], EdgeCreateFlags::NO_DUPLICATE))
            .collect();
        mesh.face_create(&vs, &es, FaceCreateFlags::NONE)
    }

    /// An elongated rhombus split into two triangles along its **long**
    /// diagonal — the unambiguously *wrong* diagonal. `v0/v2` are the far
    /// (long-axis) corners; `v1/v3` are the near (short-axis) corners. Returns
    /// `[v0, v1, v2, v3]` and the two faces. Rotating the shared `{v0,v2}`
    /// diagonal to `{v1,v3}` (the short diagonal) is the unique improvement.
    fn long_diagonal_strip(mesh: &mut LoopMesh) -> ([VertexKey; 4], [FaceKey; 2]) {
        let v0 = mesh.create_vertex(Vec3::new(-3.0, 0.0, 0.0));
        let v1 = mesh.create_vertex(Vec3::new(0.0, -0.6, 0.0));
        let v2 = mesh.create_vertex(Vec3::new(3.0, 0.0, 0.0));
        let v3 = mesh.create_vertex(Vec3::new(0.0, 0.6, 0.0));
        // Both triangles share the long diagonal {v0, v2}.
        let f0 = tri(mesh, [v0, v1, v2]);
        let f1 = tri(mesh, [v0, v2, v3]);
        ([v0, v1, v2, v3], [f0, f1])
    }

    /// The same rhombus split along its **short** diagonal {v1,v3} — already
    /// the optimal (Delaunay-like) triangulation. Returns the verts and faces.
    fn short_diagonal_strip(mesh: &mut LoopMesh) -> ([VertexKey; 4], [FaceKey; 2]) {
        let v0 = mesh.create_vertex(Vec3::new(-3.0, 0.0, 0.0));
        let v1 = mesh.create_vertex(Vec3::new(0.0, -0.6, 0.0));
        let v2 = mesh.create_vertex(Vec3::new(3.0, 0.0, 0.0));
        let v3 = mesh.create_vertex(Vec3::new(0.0, 0.6, 0.0));
        let f0 = tri(mesh, [v1, v2, v3]);
        let f1 = tri(mesh, [v1, v3, v0]);
        ([v0, v1, v2, v3], [f0, f1])
    }

    /// §5 / §8 + §9 strategy 2 — a sub-optimal strip rotates once to the short
    /// diagonal; `geom.out` carries exactly one edge and its two faces.
    #[test]
    fn irregular_strip_rotates_to_better_diagonal() {
        let mut mesh = LoopMesh::new();
        let ([v0, v1, v2, v3], [f0, f1]) = long_diagonal_strip(&mut mesh);
        let diag = edge_between(&mesh, v0, v2);

        let out = mesh.beautify_fill(
            &[f0, f1],
            &[diag],
            BeautifyFillOptions {
                use_restrict_tag: false,
                method: BeautifyMethod::Area,
            },
        );

        assert_eq!(out.edges().count(), 1, "one edge rotated");
        assert_eq!(out.faces().count(), 2, "two flanking faces reported");
        assert!(
            mesh.query_edge_exists(v1, v3).is_some(),
            "the short diagonal {{1,3}} now exists"
        );
        assert!(
            mesh.query_edge_exists(v0, v2).is_none(),
            "the long diagonal {{0,2}} is gone"
        );
        // The reported edge is the new short diagonal.
        let reported = *out.edges().next().unwrap();
        let [a, b] = mesh.edges.get(reported).vertices;
        assert!(
            (a == v1 && b == v3) || (a == v3 && b == v1),
            "geom.out reports the new diagonal"
        );
    }

    /// §8 + §9 strategy 1 — an already-optimal (short-diagonal) strip rotates
    /// nothing; `geom.out` is empty and the mesh is untouched.
    #[test]
    fn already_optimal_strip_is_noop() {
        let mut mesh = LoopMesh::new();
        let ([v0, v1, v2, v3], [f0, f1]) = short_diagonal_strip(&mut mesh);
        let diag = edge_between(&mesh, v1, v3);
        let counts = (mesh.vertex_count(), mesh.edge_count(), mesh.face_count());

        let out = mesh.beautify_fill(
            &[f0, f1],
            &[diag],
            BeautifyFillOptions {
                use_restrict_tag: false,
                method: BeautifyMethod::Area,
            },
        );

        assert!(out.is_empty(), "already-optimal patch: empty geom.out");
        assert_eq!(out.edges().count(), 0);
        assert_eq!(
            (mesh.vertex_count(), mesh.edge_count(), mesh.face_count()),
            counts,
            "mesh untouched"
        );
        assert!(
            mesh.query_edge_exists(v1, v3).is_some(),
            "the short diagonal survives"
        );
        let _ = (v0, v2);
    }

    /// §7 — beautify-fill is a pure re-triangulation: 4 V / 5 E / 2 F before and
    /// after the rotation. Only *which* interior diagonal exists changes.
    #[test]
    fn rotation_preserves_element_counts() {
        let mut mesh = LoopMesh::new();
        let (_v, [f0, f1]) = long_diagonal_strip(&mut mesh);
        let v = _v;
        let diag = edge_between(&mesh, v[0], v[2]);
        let before = (mesh.vertex_count(), mesh.edge_count(), mesh.face_count());
        assert_eq!(before, (4, 5, 2), "strip starts at 4 V / 5 E / 2 F");

        let out = mesh.beautify_fill(
            &[f0, f1],
            &[diag],
            BeautifyFillOptions {
                use_restrict_tag: false,
                method: BeautifyMethod::Area,
            },
        );

        assert!(!out.is_empty());
        assert_eq!(
            (mesh.vertex_count(), mesh.edge_count(), mesh.face_count()),
            before,
            "rotation is 0 V / 0 E / 0 F"
        );
    }

    /// An irregular convex hexagon `[v0..v5]` (boundary order, CCW) with a
    /// deliberately *bad* non-fan starting triangulation `{v0-v3, v0-v4, v1-v3}`
    /// (faces `(0,1,3),(0,3,4),(0,4,5),(1,2,3)`). Its unique area-optimal
    /// triangulation is `{v1-v5, v2-v4, v2-v5}` — flip-distance **four** away,
    /// while there are only three interior diagonals, so *every* descent path
    /// must rotate a diagonal that an earlier rotation *created* (the §5.3
    /// re-admission path). The starting and optimal diagonal sets are disjoint.
    fn irregular_hexagon_patch(mesh: &mut LoopMesh) -> ([VertexKey; 6], [FaceKey; 4]) {
        let coords = [
            (-8.0, 6.0),
            (-7.0, 4.0),
            (-3.0, 0.0),
            (5.0, -5.0),
            (5.0, 2.0),
            (0.0, 5.0),
        ];
        let v: Vec<VertexKey> = coords
            .iter()
            .map(|&(x, y)| mesh.create_vertex(Vec3::new(x, y, 0.0)))
            .collect();
        // CCW boundary order ⇒ increasing-index triangles wind consistently.
        let faces = [
            tri(mesh, [v[0], v[1], v[3]]),
            tri(mesh, [v[0], v[3], v[4]]),
            tri(mesh, [v[0], v[4], v[5]]),
            tri(mesh, [v[1], v[2], v[3]]),
        ];
        (v.try_into().unwrap(), faces)
    }

    /// §5.3 / §9 strategy 3 — a multi-rotation cascade in which a diagonal
    /// *created* by an earlier rotation must itself rotate again to reach the
    /// optimum. The patch's optimum is flip-distance four from the start but has
    /// only three interior diagonals, so a correct descent must re-admit a
    /// created diagonal: `v3-v5` (minted by rotating `v0-v4`) is re-scored by a
    /// later neighbour rotation and rotated again to `v2-v4`. This exercises the
    /// §5.3 re-admission path (the `rotatable` transfer keeps the rotated
    /// diagonal eligible across its identity change) and pins convergence to the
    /// unique optimum. It also pins the `geom.out` set semantics (§7): the
    /// superseded `v3-v5` must be reported once and replaced by `v2-v4`, never
    /// left lingering as a duplicate.
    #[test]
    fn multi_rotation_cascade_readmits_created_diagonal() {
        let mut mesh = LoopMesh::new();
        let (v, faces) = irregular_hexagon_patch(&mut mesh);
        let diagonals = [
            edge_between(&mesh, v[0], v[3]),
            edge_between(&mesh, v[0], v[4]),
            edge_between(&mesh, v[1], v[3]),
        ];
        let before = (mesh.vertex_count(), mesh.edge_count(), mesh.face_count());

        let out = mesh.beautify_fill(
            &faces,
            &diagonals,
            BeautifyFillOptions {
                use_restrict_tag: false,
                method: BeautifyMethod::Area,
            },
        );

        // Converged to the unique optimum: every original diagonal is gone and
        // all three optimal diagonals are present.
        for (a, b) in [(0, 3), (0, 4), (1, 3)] {
            assert!(
                mesh.query_edge_exists(v[a], v[b]).is_none(),
                "original diagonal {{{a},{b}}} was replaced"
            );
        }
        for (a, b) in [(1, 5), (2, 4), (2, 5)] {
            assert!(
                mesh.query_edge_exists(v[a], v[b]).is_some(),
                "optimal diagonal {{{a},{b}}} is present"
            );
        }
        // `v2-v4` is reachable only by re-rotating the created diagonal `v3-v5`
        // (the §5.3 re-admission path). The superseded `v3-v5` must be gone.
        assert!(
            mesh.query_edge_exists(v[3], v[5]).is_none(),
            "the superseded created diagonal {{3,5}} is gone"
        );

        // `geom.out` set semantics (§7): each currently-present rotated diagonal
        // once (three of them), each flanking face once (all four). A regression
        // that appended the stale `v3-v5` instead of replacing it would report
        // four edges.
        assert_eq!(out.edges().count(), 3, "three rotated diagonals, deduped");
        assert_eq!(out.faces().count(), 4, "four flanking faces, deduped");

        // Every reported edge is a live current diagonal (no stale keys): the
        // reported set is exactly {v1-v5, v2-v4, v2-v5}.
        let expected: HashSet<(u32, u32)> =
            [(v[1], v[5]), (v[2], v[4]), (v[2], v[5])]
                .iter()
                .map(|&(a, b)| sorted_pair(a, b))
                .collect();
        let reported: HashSet<(u32, u32)> = out
            .edges()
            .map(|&e| {
                let [a, b] = mesh.edges.get(e).vertices;
                sorted_pair(a, b)
            })
            .collect();
        assert_eq!(reported, expected, "geom.out reports the live diagonals");

        // Pure re-triangulation: counts invariant (§7).
        assert_eq!(
            (mesh.vertex_count(), mesh.edge_count(), mesh.face_count()),
            before,
            "rotation is 0 V / 0 E / 0 F"
        );
    }

    /// §6.1 — constraint protection by omission: the beneficial diagonal is
    /// *not* in `edges`, so it must not rotate even though doing so would
    /// improve the patch. `geom.out` is empty and the long diagonal survives.
    #[test]
    fn omitted_edge_is_constraint_and_does_not_rotate() {
        let mut mesh = LoopMesh::new();
        let ([v0, v1, v2, v3], [f0, f1]) = long_diagonal_strip(&mut mesh);
        let diag = edge_between(&mesh, v0, v2);
        let counts = (mesh.vertex_count(), mesh.edge_count(), mesh.face_count());

        // Pass the patch faces but an empty rotatable-edge set — the diagonal is
        // a constraint edge.
        let out = mesh.beautify_fill(
            &[f0, f1],
            &[],
            BeautifyFillOptions {
                use_restrict_tag: false,
                method: BeautifyMethod::Area,
            },
        );

        assert!(out.is_empty(), "no rotatable edge ⇒ empty geom.out");
        assert!(
            mesh.query_edge_exists(v0, v2).is_some(),
            "the constraint diagonal survives"
        );
        assert!(
            mesh.query_edge_exists(v1, v3).is_none(),
            "no rotation occurred"
        );
        assert_eq!(
            (mesh.vertex_count(), mesh.edge_count(), mesh.face_count()),
            counts
        );
        let _ = diag;
    }

    /// A **folded** two-triangle tent: both near corners `v1/v3` lifted out of
    /// the `v0..v2` line so the current `{v0,v2}` diagonal is a sharp ridge,
    /// while the candidate `{v1,v3}` diagonal lies flat. The §4.1 ANGLE metric
    /// prefers the flatter candidate, so it rotates. Returns `[v0,v1,v2,v3]`
    /// (the strip shares the `{v0,v2}` diagonal).
    fn folded_tent_strip(mesh: &mut LoopMesh) -> ([VertexKey; 4], [FaceKey; 2]) {
        let v0 = mesh.create_vertex(Vec3::new(-3.0, 0.0, 0.0));
        let v1 = mesh.create_vertex(Vec3::new(0.0, -0.6, 1.5));
        let v2 = mesh.create_vertex(Vec3::new(3.0, 0.0, 0.0));
        let v3 = mesh.create_vertex(Vec3::new(0.0, 0.6, 1.5));
        let f0 = tri(mesh, [v0, v1, v2]);
        let f1 = tri(mesh, [v0, v2, v3]);
        ([v0, v1, v2, v3], [f0, f1])
    }

    /// A **non-planar saddle** two-triangle strip split on `{v0,v2}`, engineered
    /// so the two §4 metrics *disagree*: the AREA metric rotates `{v0,v2}` →
    /// `{v1,v3}` (the projected quad's `{v0,v2}` split gives the worse
    /// area-over-perimeter ratio, cost ≈ −0.05) while the ANGLE metric **keeps**
    /// `{v0,v2}` (its flanking-triangle dihedral fold is flatter than
    /// `{v1,v3}`'s, cost ≈ +4.4°). This is the discriminating case: a
    /// metric-blind ANGLE path that fell through to AREA would rotate.
    fn saddle_discriminating_strip(mesh: &mut LoopMesh) -> ([VertexKey; 4], [FaceKey; 2]) {
        let v0 = mesh.create_vertex(Vec3::new(-2.5, -0.5, 1.5));
        let v1 = mesh.create_vertex(Vec3::new(-1.5, -2.5, 1.5));
        let v2 = mesh.create_vertex(Vec3::new(2.5, -0.5, 0.0));
        let v3 = mesh.create_vertex(Vec3::new(-1.5, 1.0, 2.0));
        let f0 = tri(mesh, [v0, v1, v2]);
        let f1 = tri(mesh, [v0, v2, v3]);
        ([v0, v1, v2, v3], [f0, f1])
    }

    /// A strip split on `{v0,v2}` whose **candidate** diagonal `{v1,v3}` would
    /// produce a degenerate (zero-area) triangle: the diagonal endpoint `v0` is
    /// collinear with both apexes `v1`/`v3`, so the candidate triangle
    /// `(v1,v0,v3)` has a zero-length normal. The §4.1 ANGLE metric must score
    /// the never-rotate sentinel and keep the edge out of the worklist.
    fn degenerate_candidate_strip(mesh: &mut LoopMesh) -> ([VertexKey; 4], [FaceKey; 2]) {
        let v0 = mesh.create_vertex(Vec3::new(0.0, -3.0, 0.0));
        let v1 = mesh.create_vertex(Vec3::new(0.0, -1.0, 0.0));
        let v2 = mesh.create_vertex(Vec3::new(3.0, 0.0, 0.0));
        let v3 = mesh.create_vertex(Vec3::new(0.0, 1.0, 0.0));
        let f0 = tri(mesh, [v0, v1, v2]);
        let f1 = tri(mesh, [v0, v2, v3]);
        ([v0, v1, v2, v3], [f0, f1])
    }

    /// §4.1 — the ANGLE metric rotates a folded patch toward the flatter
    /// (smaller-dihedral) diagonal. The ridge `{v0,v2}` is replaced by the flat
    /// `{v1,v3}`; `geom.out` carries the one rotated edge and its two faces.
    #[test]
    fn angle_metric_rotates_folded_tent() {
        let mut mesh = LoopMesh::new();
        let ([v0, v1, v2, v3], [f0, f1]) = folded_tent_strip(&mut mesh);
        let diag = edge_between(&mesh, v0, v2);

        let out = mesh.beautify_fill(
            &[f0, f1],
            &[diag],
            BeautifyFillOptions {
                use_restrict_tag: false,
                method: BeautifyMethod::Angle,
            },
        );

        assert_eq!(out.edges().count(), 1, "one edge rotated under ANGLE");
        assert_eq!(out.faces().count(), 2, "two flanking faces reported");
        assert!(
            mesh.query_edge_exists(v1, v3).is_some(),
            "the flatter diagonal {{1,3}} now exists"
        );
        assert!(
            mesh.query_edge_exists(v0, v2).is_none(),
            "the folded ridge diagonal {{0,2}} is gone"
        );
    }

    /// §4.1 / §8 (`method` disagreement) — on the saddle the ANGLE metric is a
    /// **no-op** (the current `{0,2}` fold is already the flatter one) even
    /// though the AREA metric would rotate the same patch. The two runs of the
    /// same fixture must diverge; this pins that the operator honours the metric
    /// selector rather than always falling through to AREA.
    #[test]
    fn angle_metric_keeps_saddle_where_area_rotates() {
        // ANGLE arm: keep {0,2}.
        let mut mesh = LoopMesh::new();
        let ([v0, v1, v2, v3], [f0, f1]) = saddle_discriminating_strip(&mut mesh);
        let diag = edge_between(&mesh, v0, v2);
        let counts = (mesh.vertex_count(), mesh.edge_count(), mesh.face_count());

        let out = mesh.beautify_fill(
            &[f0, f1],
            &[diag],
            BeautifyFillOptions {
                use_restrict_tag: false,
                method: BeautifyMethod::Angle,
            },
        );

        assert!(out.is_empty(), "ANGLE: saddle is already the flatter split");
        assert!(
            mesh.query_edge_exists(v0, v2).is_some(),
            "ANGLE keeps the {{0,2}} diagonal"
        );
        assert!(
            mesh.query_edge_exists(v1, v3).is_none(),
            "ANGLE never creates the {{1,3}} diagonal"
        );
        assert_eq!(
            (mesh.vertex_count(), mesh.edge_count(), mesh.face_count()),
            counts,
            "ANGLE arm: mesh untouched"
        );

        // AREA arm: same fixture rotates — proving the metrics genuinely diverge.
        let mut mesh_area = LoopMesh::new();
        let ([a0, a1, a2, a3], [g0, g1]) = saddle_discriminating_strip(&mut mesh_area);
        let area_diag = edge_between(&mesh_area, a0, a2);
        let area_out = mesh_area.beautify_fill(
            &[g0, g1],
            &[area_diag],
            BeautifyFillOptions {
                use_restrict_tag: false,
                method: BeautifyMethod::Area,
            },
        );
        assert!(!area_out.is_empty(), "AREA rotates the same saddle");
        assert!(
            mesh_area.query_edge_exists(a1, a3).is_some(),
            "AREA rotates to {{1,3}}"
        );
        assert!(
            mesh_area.query_edge_exists(a0, a2).is_none(),
            "AREA drops {{0,2}}"
        );
    }

    /// §4.1 (candidate zero-normal sentinel) / §3 gate 5 — a candidate diagonal
    /// whose flanking triangle is degenerate (collinear) scores the never-rotate
    /// sentinel under ANGLE, so the edge never enters the worklist: the patch is
    /// returned unchanged and `geom.out` is empty.
    #[test]
    fn angle_metric_degenerate_candidate_stays_out_of_worklist() {
        let mut mesh = LoopMesh::new();
        let ([v0, v1, v2, v3], [f0, f1]) = degenerate_candidate_strip(&mut mesh);
        let diag = edge_between(&mesh, v0, v2);
        let counts = (mesh.vertex_count(), mesh.edge_count(), mesh.face_count());

        let out = mesh.beautify_fill(
            &[f0, f1],
            &[diag],
            BeautifyFillOptions {
                use_restrict_tag: false,
                method: BeautifyMethod::Angle,
            },
        );

        assert!(out.is_empty(), "degenerate candidate ⇒ never-rotate sentinel");
        assert!(
            mesh.query_edge_exists(v0, v2).is_some(),
            "the current diagonal {{0,2}} survives"
        );
        assert!(
            mesh.query_edge_exists(v1, v3).is_none(),
            "the degenerate candidate {{1,3}} is never created"
        );
        assert_eq!(
            (mesh.vertex_count(), mesh.edge_count(), mesh.face_count()),
            counts,
            "mesh untouched"
        );
    }
}
