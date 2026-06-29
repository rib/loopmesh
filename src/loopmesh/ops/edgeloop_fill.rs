use std::collections::HashMap;

use smallvec::SmallVec;

use crate::loopmesh::*;

/// Parameters for [`LoopMesh::edgeloop_fill`], mirroring the operator's BMOP
/// slot knobs.
///
/// Algorithm in
/// [`EDGELOOP_FILL.md`](../../../../radial-mesh-guides/operators/EDGELOOP_FILL.md)
/// §1.
#[derive(Copy, Clone, Debug, Default)]
pub struct EdgeloopFillParams {
    /// Material index stamped onto every face the operator creates (§1/§5).
    pub mat_nr: i32,
    /// When set, each created face is flagged smooth-shaded (§1/§5). A per-face
    /// attribute only; it does not change topology.
    pub use_smooth: bool,
}

/// The `faces.out` of [`LoopMesh::edgeloop_fill`]: every **new** n-gon the
/// operator created — one per closed loop it capped (§1). A loop whose face
/// already existed (§6) contributes nothing.
#[derive(Clone, Debug, Default)]
pub struct EdgeloopFillResult {
    faces: Vec<FaceKey>,
}

impl EdgeloopFillResult {
    pub fn faces(&self) -> &[FaceKey] {
        &self.faces
    }
    pub fn is_empty(&self) -> bool {
        self.faces.is_empty()
    }
}

impl LoopMesh {
    /// Cap each simple closed loop in an undirected set of `edges` with a single
    /// n-gon (§1). The operator creates no vertices and no edges — every side of
    /// every face is an existing input loop edge — and performs no
    /// triangulation.
    ///
    /// Algorithm in
    /// [`EDGELOOP_FILL.md`](../../../../radial-mesh-guides/operators/EDGELOOP_FILL.md):
    /// the §3 vertex-collection + count gate (`#verts == #edges`, all-or-nothing
    /// rejection), the §4 valence gate (every loop vertex touches exactly two
    /// input edges; branch points and dangling ends are rejected), the §5 cycle
    /// walk that orders each loop and builds its n-gon with a radial-data winding
    /// vote (rim-of-a-hole loops wind coherent with the bordering surface; a free
    /// wire loop falls back to walk order), and the §6 already-faced idempotence
    /// skip.
    ///
    /// Several disjoint loops in one `edges` set are each walked and capped
    /// independently (§5/§8). Malformed input — an open chain, a branched net, or
    /// a sub-three-edge set — fills **nothing** (§3/§4/§7).
    pub fn edgeloop_fill(
        &mut self,
        edges: &[EdgeKey],
        params: &EdgeloopFillParams,
    ) -> EdgeloopFillResult {
        let mut result = EdgeloopFillResult::default();

        // §7 — fewer than three edges can bound no loop.
        let edge_count = edges.len();
        if edge_count < 3 {
            return result;
        }

        // §3 — collect loop vertices from the (undirected) input edge set,
        // building the per-vertex adjacency over only those edges. The count
        // gate is enforced as we go: a vertex collected once the array is
        // already full means more distinct verts than edges (an open chain), so
        // the whole operation is rejected.
        let mut adjacency: HashMap<VertexKey, SmallVec<[(VertexKey, EdgeKey); 2]>> = HashMap::new();
        let mut verts: Vec<VertexKey> = Vec::with_capacity(edge_count);
        for &edge in edges {
            let [a, b] = self.edge_vert_keys(edge);
            if a == b {
                return EdgeloopFillResult::default();
            }
            for vertex in [a, b] {
                if !adjacency.contains_key(&vertex) {
                    if verts.len() == edge_count {
                        return EdgeloopFillResult::default();
                    }
                    verts.push(vertex);
                    adjacency.insert(vertex, SmallVec::new());
                }
            }
            adjacency.get_mut(&a).unwrap().push((b, edge));
            adjacency.get_mut(&b).unwrap().push((a, edge));
        }
        // Fewer distinct verts than edges → degenerate / multi-edge input.
        if verts.len() != edge_count {
            return EdgeloopFillResult::default();
        }

        // §4 — valence gate: every collected vertex must touch exactly two of
        // the input edges. A valence ≠ 2 vertex is a dangling end (1) or a
        // branch / non-manifold junction (3+); either rejects the whole input.
        for vertex in &verts {
            if adjacency[vertex].len() != 2 {
                return EdgeloopFillResult::default();
            }
        }

        // §5 — walk each disjoint closed cycle and cap it with one n-gon. With
        // the valence-2 guarantee the walk is unambiguous: at each vertex take
        // the marked edge we did not arrive on.
        let mut consumed: HashMap<VertexKey, ()> = HashMap::with_capacity(verts.len());
        for &start in &verts {
            if consumed.contains_key(&start) {
                continue;
            }
            let mut cycle: Vec<VertexKey> = Vec::new();
            let mut current = start;
            let mut arrived_on: Option<EdgeKey> = None;
            loop {
                let neighbors = &adjacency[&current];
                let (next_vertex, next_edge) = if Some(neighbors[0].1) == arrived_on {
                    neighbors[1]
                } else {
                    neighbors[0]
                };
                cycle.push(current);
                consumed.insert(current, ());
                current = next_vertex;
                arrived_on = Some(next_edge);
                if current == start {
                    break;
                }
            }
            self.edgeloop_fill_cap_cycle(&cycle, params, &mut result);
        }

        result
    }

    /// Build the single n-gon for one ordered loop cycle (§5), honouring the §6
    /// already-faced skip and the §5 radial-data winding vote.
    fn edgeloop_fill_cap_cycle(
        &mut self,
        cycle: &[VertexKey],
        params: &EdgeloopFillParams,
        result: &mut EdgeloopFillResult,
    ) {
        // A loop needs at least three corners to bound a face (§7).
        if cycle.len() < 3 {
            return;
        }

        // §6 — idempotence: a loop whose face already exists is left untouched.
        if self.face_exists(cycle).is_some() {
            return;
        }

        // §5 — winding from radial data; reverse the walk-order cycle when the
        // vote says the n-gon would otherwise be incoherent with a bordering
        // surface. A free wire loop scores a zero tie and keeps walk order.
        let face = if self.edgeloop_fill_winding_flip(cycle) {
            let mut reversed: Vec<VertexKey> = cycle.to_vec();
            reversed.reverse();
            self.face_create_verts(&reversed, FaceCreateFlags::NO_DUPLICATE)
        } else {
            self.face_create_verts(cycle, FaceCreateFlags::NO_DUPLICATE)
        };

        if params.mat_nr != 0 {
            self.set_face_material_index(face, params.mat_nr);
        }
        // `use_smooth` has no per-face representation in loopmesh yet; the
        // matrix tracks it as a separate row.

        result.faces.push(face);
    }

    /// §5 winding vote: returns true when the loop's n-gon, built in walk order,
    /// would be wound incoherently with an existing bordering surface and should
    /// be reversed. Only loop edges that already carry exactly one face vote; a
    /// free wire loop tallies zero and keeps walk order.
    fn edgeloop_fill_winding_flip(&self, cycle: &[VertexKey]) -> bool {
        let n = cycle.len();
        let mut tally: i32 = 0;
        for i in 0..n {
            let from_vertex = cycle[i];
            let to_vertex = cycle[(i + 1) % n];
            let Some(edge) = self.query_edge_exists(from_vertex, to_vertex) else {
                continue;
            };
            let radial = self.edge_loop_keys(edge);
            if radial.len() != 1 {
                continue;
            }
            // Two coherent faces sharing a manifold edge traverse it in opposite
            // senses. If the existing face already traverses from_vertex→to_vertex
            // the same way our walk does, our walk is incoherent → vote to flip.
            if self.loop_vertex(radial[0]) == from_vertex {
                tally += 1;
            } else {
                tally -= 1;
            }
        }
        tally > 0
    }
}

#[cfg(test)]
mod tests {
    use std::f32::consts::TAU;

    use glam::Vec3;

    use super::EdgeloopFillParams;
    use crate::loopmesh::*;

    /// Build a free wire ring of `n` verts on a unit circle in the z=0 plane,
    /// returning the ring edges in arbitrary (set) order.
    fn convex_wire_ring(mesh: &mut LoopMesh, n: usize, radius: f32, center: Vec3) -> Vec<EdgeKey> {
        let verts: Vec<VertexKey> = (0..n)
            .map(|i| {
                let theta = (i as f32) * TAU / (n as f32);
                mesh.create_vertex(center + Vec3::new(radius * theta.cos(), radius * theta.sin(), 0.0))
            })
            .collect();
        (0..n)
            .map(|i| mesh.create_edge([verts[i], verts[(i + 1) % n]], EdgeCreateFlags::NO_DUPLICATE))
            .collect()
    }

    #[test]
    fn convex_quad_pentagon_hexagon_each_cap_to_single_ngon() {
        for n in [4usize, 5, 6] {
            let mut mesh = LoopMesh::new();
            let edges = convex_wire_ring(&mut mesh, n, 1.0, Vec3::ZERO);
            let result = mesh.edgeloop_fill(&edges, &EdgeloopFillParams::default());

            assert_eq!(mesh.vertex_count(), n, "no vertex created (n={n})");
            assert_eq!(mesh.edge_count(), n, "no edge created (n={n})");
            assert_eq!(mesh.face_count(), 1, "one n-gon (n={n})");
            assert_eq!(result.faces().len(), 1, "faces.out reports the new n-gon (n={n})");
            let face = mesh.face_keys()[0];
            assert_eq!(mesh.face_len(face), n, "the n-gon spans all corners (n={n})");
            mesh.validate_topology();
        }
    }

    #[test]
    fn concave_reflex_loop_caps_to_single_ngon() {
        // An L-shaped hexagon ring with one reflex corner — no convexity is
        // assumed; the n-gon is built straight from the cycle.
        let mut mesh = LoopMesh::new();
        let positions = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::new(2.0, 1.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0), // reflex corner
            Vec3::new(1.0, 2.0, 0.0),
            Vec3::new(0.0, 2.0, 0.0),
        ];
        let verts: Vec<VertexKey> = positions.iter().map(|&p| mesh.create_vertex(p)).collect();
        let n = verts.len();
        let edges: Vec<EdgeKey> = (0..n)
            .map(|i| mesh.create_edge([verts[i], verts[(i + 1) % n]], EdgeCreateFlags::NO_DUPLICATE))
            .collect();

        let result = mesh.edgeloop_fill(&edges, &EdgeloopFillParams::default());

        assert_eq!(mesh.vertex_count(), n, "no vertex created");
        assert_eq!(mesh.edge_count(), n, "no edge created");
        assert_eq!(mesh.face_count(), 1, "concave loop caps to one n-gon");
        assert_eq!(result.faces().len(), 1);
        assert_eq!(mesh.face_len(mesh.face_keys()[0]), n, "all six corners present");
        mesh.validate_topology();
    }

    #[test]
    fn two_disjoint_loops_cap_to_two_ngons_in_one_call() {
        let mut mesh = LoopMesh::new();
        let mut edges = convex_wire_ring(&mut mesh, 4, 1.0, Vec3::new(-5.0, 0.0, 0.0));
        edges.extend(convex_wire_ring(&mut mesh, 5, 1.0, Vec3::new(5.0, 0.0, 0.0)));

        let result = mesh.edgeloop_fill(&edges, &EdgeloopFillParams::default());

        assert_eq!(mesh.vertex_count(), 9, "no vertex created (4 + 5)");
        assert_eq!(mesh.edge_count(), 9, "no edge created (4 + 5)");
        assert_eq!(mesh.face_count(), 2, "one n-gon per loop");
        assert_eq!(result.faces().len(), 2, "faces.out reports both new n-gons");
        let mut arities: Vec<usize> = mesh.face_keys().iter().map(|&f| mesh.face_len(f)).collect();
        arities.sort_unstable();
        assert_eq!(arities, vec![4, 5], "the quad and the pentagon were each capped");
        mesh.validate_topology();
    }

    #[test]
    fn already_faced_loop_is_idempotent() {
        let mut mesh = LoopMesh::new();
        let edges = convex_wire_ring(&mut mesh, 5, 1.0, Vec3::ZERO);
        let first = mesh.edgeloop_fill(&edges, &EdgeloopFillParams::default());
        assert_eq!(first.faces().len(), 1, "first call caps the loop");
        assert_eq!(mesh.face_count(), 1);

        let second = mesh.edgeloop_fill(&edges, &EdgeloopFillParams::default());
        assert!(second.is_empty(), "second call is a no-op (already faced)");
        assert_eq!(mesh.face_count(), 1, "no duplicate face created");
        mesh.validate_topology();
    }

    #[test]
    fn open_chain_creates_nothing() {
        // Four edges, five verts (an open chain) → §3 count gate rejects.
        let mut mesh = LoopMesh::new();
        let verts: Vec<VertexKey> = (0..5)
            .map(|i| mesh.create_vertex(Vec3::new(i as f32, 0.0, 0.0)))
            .collect();
        let edges: Vec<EdgeKey> = (0..4)
            .map(|i| mesh.create_edge([verts[i], verts[i + 1]], EdgeCreateFlags::NO_DUPLICATE))
            .collect();

        let result = mesh.edgeloop_fill(&edges, &EdgeloopFillParams::default());

        assert!(result.is_empty(), "open chain fills nothing");
        assert_eq!(mesh.face_count(), 0, "no face created");
        mesh.validate_topology();
    }

    #[test]
    fn branch_point_creates_nothing() {
        // A closed quad ring plus one spoke edge to a fifth vertex: the spoke's
        // ring endpoint has valence 3 → §4 valence gate rejects all-or-nothing.
        let mut mesh = LoopMesh::new();
        let ring = convex_wire_ring(&mut mesh, 4, 1.0, Vec3::ZERO);
        let ring_vertex = mesh.edge_vert_keys(ring[0])[0];
        let spoke_tip = mesh.create_vertex(Vec3::new(3.0, 3.0, 0.0));
        let spoke = mesh.create_edge([ring_vertex, spoke_tip], EdgeCreateFlags::NO_DUPLICATE);

        let mut edges = ring.clone();
        edges.push(spoke);
        let result = mesh.edgeloop_fill(&edges, &EdgeloopFillParams::default());

        assert!(result.is_empty(), "branched net fills nothing");
        assert_eq!(mesh.face_count(), 0, "no face created");
        mesh.validate_topology();
    }

    #[test]
    fn under_edged_input_creates_nothing() {
        // Two edges (a single open segment doubled back) — fewer than three
        // edges can bound no loop (§7).
        let mut mesh = LoopMesh::new();
        let a = mesh.create_vertex(Vec3::new(0.0, 0.0, 0.0));
        let b = mesh.create_vertex(Vec3::new(1.0, 0.0, 0.0));
        let c = mesh.create_vertex(Vec3::new(0.5, 1.0, 0.0));
        let e0 = mesh.create_edge([a, b], EdgeCreateFlags::NO_DUPLICATE);
        let e1 = mesh.create_edge([b, c], EdgeCreateFlags::NO_DUPLICATE);

        let result = mesh.edgeloop_fill(&[e0, e1], &EdgeloopFillParams::default());

        assert!(result.is_empty(), "sub-three-edge set fills nothing");
        assert_eq!(mesh.face_count(), 0);
        mesh.validate_topology();
    }
}
