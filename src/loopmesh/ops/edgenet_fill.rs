use crate::loopmesh::*;

/// Parameters for [`LoopMesh::edgenet_fill`], mirroring the operator's BMOP slot
/// knobs.
///
/// Algorithm in
/// [`EDGENET_FILL.md`](../../../../radial-mesh-guides/operators/EDGENET_FILL.md)
/// §1.
#[derive(Copy, Clone, Debug, Default)]
pub struct EdgenetFillParams {
    /// The material index **stamped** onto every new cap face (§5). Survives only
    /// on caps in a fully free / isolated net — the §6 attribute flood overwrites
    /// it for any cap that reaches an existing surface (§6.1).
    pub mat_nr: i32,
    /// When set, every new cap face is flagged **smooth** (§5); when clear, caps
    /// are left flat. Like `mat_nr`, observable only on free-net caps — the §6
    /// flood overwrites it for caps reaching an existing surface (§6.1).
    pub use_smooth: bool,
    /// **Declared but inert** (§6.3). The slot carries a maximum cap perimeter in
    /// [`LoopMesh::holes_fill`], but edge-net fill does **not** read or act on it:
    /// every discoverable cap is kept regardless of size. Present for signature
    /// compatibility only.
    pub sides: usize,
}

/// The `faces.out` of [`LoopMesh::edgenet_fill`]: every **new** cap the operator
/// created — one n-gon per enclosed minimal cycle the fill found (§1). An edge
/// net component with no closable cycle (an open chain) contributes nothing.
#[derive(Clone, Debug, Default)]
pub struct EdgenetFillResult {
    faces: Vec<FaceKey>,
}

impl EdgenetFillResult {
    pub fn faces(&self) -> &[FaceKey] {
        &self.faces
    }
    pub fn is_empty(&self) -> bool {
        self.faces.is_empty()
    }
}

impl LoopMesh {
    /// Discover every minimal closed cycle an arbitrary set of `edges` encloses
    /// and cap each with a **single n-gon** (§1), stamping the caller's `mat_nr` /
    /// `use_smooth` onto the new faces (§5). The operator creates no vertex and no
    /// edge — every cap side is an existing tagged edge — and never bridges open
    /// endpoints (that is the separate `edgenet_prepare` op, §1.2).
    ///
    /// Algorithm in
    /// [`EDGENET_FILL.md`](../../../../radial-mesh-guides/operators/EDGENET_FILL.md):
    /// the §3 net tagging and §4 two-front minimal-cycle fill are the **shared
    /// core** ([`LoopMesh::edgenet_minimal_cycle_fill`], reused verbatim from
    /// holes-fill §4); §5 stamps the caller's material / smooth onto every new
    /// cap; §6 floods the surrounding surface's attributes inward, **overwriting**
    /// the §5 stamp (material, smooth, loop data, winding) for any cap that reaches
    /// an existing surface (§6.1). So the `mat_nr` / `use_smooth` knobs are
    /// observable only on caps in a fully free net component.
    ///
    /// Unlike holes-fill, `params.sides` is **inert** (§6.3): it is never read and
    /// no cap is ever deleted — the fill is monotonically additive.
    pub fn edgenet_fill(
        &mut self,
        edges: &[EdgeKey],
        params: &EdgenetFillParams,
    ) -> EdgenetFillResult {
        let mut result = EdgenetFillResult::default();

        // §3 + §4 — the shared minimal-cycle core: tag the net, cap every enclosed
        // cell with one n-gon, and refresh each cap's normal (NORMALS_CALC). The
        // same engine holes-fill runs.
        let new_caps = self.edgenet_minimal_cycle_fill(edges);

        // §5 — stamp the caller's material / smooth onto every new cap. This is the
        // *default*: §6 overwrites it on any cap the flood reaches (§6.1). The cap's
        // normal was already refreshed by the core (the §5 normal recompute).
        for &cap in &new_caps {
            self.set_face_material_index(cap, params.mat_nr);
            self.set_face_smooth(cap, params.use_smooth);
        }

        // §6.3 — `sides` is inert: no post-fill deletion pass. Every cap is kept,
        // regardless of perimeter. `params.sides` is deliberately not read.

        // §6 / §6.1 — flood the surrounding surface's attributes into the caps,
        // overwriting the §5 stamp for any cap reaching an existing surface. A cap
        // in a free net is never visited, so it keeps the caller's stamp (§6.1) and
        // its geometric-normal winding fallback (§6.2).
        self.holes_fill_inherit_attributes(&new_caps);

        result.faces = new_caps;
        result
    }
}

#[cfg(test)]
mod tests {
    use std::f32::consts::TAU;

    use glam::Vec3;

    use super::EdgenetFillParams;
    use crate::loopmesh::*;

    /// Build a free wire ring of `n` verts on a circle of `radius` in the z=0
    /// plane, returning the ring edges in creation order.
    fn convex_wire_ring(mesh: &mut LoopMesh, n: usize, radius: f32) -> Vec<EdgeKey> {
        let verts: Vec<VertexKey> = (0..n)
            .map(|i| {
                let theta = (i as f32) * TAU / (n as f32);
                mesh.create_vertex(Vec3::new(radius * theta.cos(), radius * theta.sin(), 0.0))
            })
            .collect();
        (0..n)
            .map(|i| {
                mesh.create_edge(
                    [verts[i], verts[(i + 1) % n]],
                    EdgeCreateFlags::NO_DUPLICATE,
                )
            })
            .collect()
    }

    /// Build a free `n`×`m` wire lattice (no faces) of unit cells in the z=0
    /// plane and return every wire edge. Encloses `n·m` cells.
    fn wire_grid_cells(mesh: &mut LoopMesh, n: usize, m: usize) -> Vec<EdgeKey> {
        let cols = m + 1;
        let rows = n + 1;
        let verts: Vec<VertexKey> = (0..rows)
            .flat_map(|row| (0..cols).map(move |col| (row, col)))
            .map(|(row, col)| mesh.create_vertex(Vec3::new(col as f32, row as f32, 0.0)))
            .collect();
        let at = |row: usize, col: usize| verts[row * cols + col];
        let mut edges = Vec::new();
        // Horizontal wires.
        for row in 0..rows {
            for col in 0..m {
                edges.push(mesh.create_edge(
                    [at(row, col), at(row, col + 1)],
                    EdgeCreateFlags::NO_DUPLICATE,
                ));
            }
        }
        // Vertical wires.
        for row in 0..n {
            for col in 0..cols {
                edges.push(mesh.create_edge(
                    [at(row, col), at(row + 1, col)],
                    EdgeCreateFlags::NO_DUPLICATE,
                ));
            }
        }
        edges
    }

    /// Build a 3×3 grid of unit quad cells with the centre cell omitted, leaving
    /// an open hole bordered by four ring faces. Returns the mesh and the four
    /// hole-rim edges (16 verts, 8 faces).
    fn grid_3x3_center_hole() -> (LoopMesh, [EdgeKey; 4]) {
        let mut mesh = LoopMesh::new();
        let verts: Vec<VertexKey> = (0..4)
            .flat_map(|row| (0..4).map(move |col| (row, col)))
            .map(|(row, col)| mesh.create_vertex(Vec3::new(col as f32, row as f32, 0.0)))
            .collect();
        let at = |row: usize, col: usize| verts[row * 4 + col];
        for row in 0..3 {
            for col in 0..3 {
                if row == 1 && col == 1 {
                    continue;
                }
                let ring = [
                    at(row, col),
                    at(row, col + 1),
                    at(row + 1, col + 1),
                    at(row + 1, col),
                ];
                mesh.face_create_verts(&ring, FaceCreateFlags::NO_DUPLICATE);
            }
        }
        let c = [at(1, 1), at(1, 2), at(2, 2), at(2, 1)];
        let rim = [
            mesh.query_edge_exists(c[0], c[1]).expect("top rim"),
            mesh.query_edge_exists(c[1], c[2]).expect("right rim"),
            mesh.query_edge_exists(c[2], c[3]).expect("bottom rim"),
            mesh.query_edge_exists(c[3], c[0]).expect("left rim"),
        ];
        (mesh, rim)
    }

    #[test]
    fn free_loop_stamps_caller_material_and_smooth() {
        // §5 / §6.1 — a free pentagon wire loop with no surrounding surface: the
        // flood never reaches its single cap, so the caller's mat_nr / use_smooth
        // survive on it.
        let mut mesh = LoopMesh::new();
        let ring = convex_wire_ring(&mut mesh, 5, 1.0);

        let result = mesh.edgenet_fill(
            &ring,
            &EdgenetFillParams {
                mat_nr: 7,
                use_smooth: true,
                sides: 0,
            },
        );

        assert_eq!(result.faces().len(), 1, "free loop caps to one n-gon");
        assert_eq!(mesh.vertex_count(), 5, "no vertex created");
        assert_eq!(mesh.edge_count(), 5, "no edge created (loop reused)");
        assert_eq!(mesh.face_count(), 1, "exactly one cap");
        let cap = result.faces()[0];
        assert_eq!(mesh.face_len(cap), 5, "the cap spans all five corners");
        assert_eq!(
            mesh.face_material_index(cap),
            7,
            "free-net cap keeps the caller's material"
        );
        assert!(
            mesh.is_face_smooth(cap),
            "free-net cap keeps the caller's smooth flag"
        );
        assert!(
            mesh.face_normal(cap).length_squared() > 1e-6,
            "§5/§6.2 fallback gives the isolated cap a well-defined normal"
        );
        mesh.validate_topology();
    }

    #[test]
    fn free_loop_zero_stamp_is_default_material_and_flat() {
        // The zero contrast that makes the stamp pins meaningful (§8): same free
        // loop, mat_nr = 0 / use_smooth = false → default material, flat.
        let mut mesh = LoopMesh::new();
        let ring = convex_wire_ring(&mut mesh, 5, 1.0);

        let result = mesh.edgenet_fill(
            &ring,
            &EdgenetFillParams {
                mat_nr: 0,
                use_smooth: false,
                sides: 0,
            },
        );

        assert_eq!(result.faces().len(), 1);
        let cap = result.faces()[0];
        assert_eq!(mesh.face_material_index(cap), 0, "default material slot 0");
        assert!(!mesh.is_face_smooth(cap), "default cap is flat");
        mesh.validate_topology();
    }

    #[test]
    fn free_multicell_net_caps_each_cell() {
        // §4 note / §7 — a free 2×2 wire lattice encloses four cells; each is
        // capped independently in one call, every cap carrying the caller's stamp.
        let mut mesh = LoopMesh::new();
        let net = wire_grid_cells(&mut mesh, 2, 2);

        let result = mesh.edgenet_fill(
            &net,
            &EdgenetFillParams {
                mat_nr: 4,
                use_smooth: true,
                sides: 0,
            },
        );

        assert_eq!(result.faces().len(), 4, "four quad caps, one per cell");
        assert_eq!(mesh.vertex_count(), 9, "no vertex created");
        assert_eq!(mesh.edge_count(), 12, "no edge created");
        assert_eq!(mesh.face_count(), 4);
        for &cap in result.faces() {
            assert_eq!(mesh.face_len(cap), 4, "each cap is a quad");
            assert_eq!(
                mesh.face_material_index(cap),
                4,
                "free cap keeps caller material"
            );
            assert!(mesh.is_face_smooth(cap), "free cap keeps caller smooth");
        }
        // The shared interior wire edges became manifold (two caps each); the outer
        // perimeter edges each border exactly one cap.
        let manifold = mesh
            .edge_keys()
            .into_iter()
            .filter(|&e| mesh.edge_loop_keys(e).len() == 2)
            .count();
        assert_eq!(
            manifold, 4,
            "the four interior cross wires are now manifold"
        );
        mesh.validate_topology();
    }

    #[test]
    fn inert_sides_keeps_oversized_cap() {
        // §6.3 — the no-op contrast with holes-fill: a free pentagon (perimeter 5)
        // filled with sides unbounded and with sides below the perimeter; BOTH keep
        // the cap, because edgenet_fill never reads `sides`.
        let mut mesh_unbounded = LoopMesh::new();
        let ring_a = convex_wire_ring(&mut mesh_unbounded, 5, 1.0);
        let result_a = mesh_unbounded.edgenet_fill(
            &ring_a,
            &EdgenetFillParams {
                mat_nr: 0,
                use_smooth: false,
                sides: 1000,
            },
        );
        assert_eq!(result_a.faces().len(), 1, "unbounded sides keeps the cap");
        assert_eq!(mesh_unbounded.face_count(), 1);

        let mut mesh_below = LoopMesh::new();
        let ring_b = convex_wire_ring(&mut mesh_below, 5, 1.0);
        let result_b = mesh_below.edgenet_fill(
            &ring_b,
            // sides = 4 < perimeter 5: holes_fill would delete this cap; edgenet
            // keeps it.
            &EdgenetFillParams {
                mat_nr: 0,
                use_smooth: false,
                sides: 4,
            },
        );
        assert_eq!(
            result_b.faces().len(),
            1,
            "sides below perimeter still keeps the cap"
        );
        assert_eq!(mesh_below.face_count(), 1, "no post-fill deletion");
        assert_eq!(mesh_below.face_len(result_b.faces()[0]), 5);
        mesh_unbounded.validate_topology();
        mesh_below.validate_topology();
    }

    #[test]
    fn surface_touching_cap_inherits_overriding_stamp() {
        // §6.1 — a cap that reaches an existing decorated surface inherits the
        // surface's material / smooth, OVERRIDING the caller's stamp. The caller's
        // mat_nr is not observable on it.
        let (mut mesh, rim) = grid_3x3_center_hole();
        for face in mesh.face_keys() {
            mesh.set_face_material_index(face, 3);
            mesh.set_face_smooth(face, true);
        }

        let result = mesh.edgenet_fill(
            &rim,
            // distinct caller mat_nr = 7, use_smooth = false — both overridden.
            &EdgenetFillParams {
                mat_nr: 7,
                use_smooth: false,
                sides: 0,
            },
        );

        assert_eq!(result.faces().len(), 1, "one cap fills the hole");
        let cap = result.faces()[0];
        assert_eq!(
            mesh.face_material_index(cap),
            3,
            "cap inherits the surface material, not the caller's 7"
        );
        assert!(
            mesh.is_face_smooth(cap),
            "cap inherits the surface smooth flag"
        );
        // §6.2 — coherent winding with the surrounding ring.
        for shared in mesh.face_edge_keys(cap) {
            let radial = mesh.edge_loop_keys(shared);
            assert_eq!(radial.len(), 2, "each rim edge now borders cap + ring face");
            assert_ne!(
                mesh.loop_vertex(radial[0]),
                mesh.loop_vertex(radial[1]),
                "coherent neighbours traverse the shared edge oppositely"
            );
        }
        mesh.validate_topology();
    }

    #[test]
    fn faces_out_free_loop_is_one_ngon() {
        // §1 — faces.out surfaces exactly the new caps: one n-gon for one cell.
        let mut mesh = LoopMesh::new();
        let ring = convex_wire_ring(&mut mesh, 5, 1.0);

        let result = mesh.edgenet_fill(&ring, &EdgenetFillParams::default());

        assert_eq!(result.faces().len(), 1, "one cap surfaced in faces.out");
        assert_eq!(mesh.face_len(result.faces()[0]), 5);
        mesh.validate_topology();
    }

    #[test]
    fn open_chain_creates_nothing() {
        // §7 — an open chain has no closable cycle, and edgenet_fill does NOT
        // bridge endpoints (that is edgenet_prepare). Nothing is created.
        let mut mesh = LoopMesh::new();
        let verts: Vec<VertexKey> = (0..4)
            .map(|i| mesh.create_vertex(Vec3::new(i as f32, 0.0, 0.0)))
            .collect();
        let chain: Vec<EdgeKey> = (0..3)
            .map(|i| mesh.create_edge([verts[i], verts[i + 1]], EdgeCreateFlags::NO_DUPLICATE))
            .collect();

        let result = mesh.edgenet_fill(&chain, &EdgenetFillParams::default());

        assert!(result.is_empty(), "open chain caps nothing");
        assert_eq!(mesh.face_count(), 0, "no face created");
        assert_eq!(mesh.vertex_count(), 4, "no vertex created");
        assert_eq!(mesh.edge_count(), 3, "no edge created");
        mesh.validate_topology();
    }

    #[test]
    fn under_edged_input_creates_nothing() {
        // §7 — two wire edges cannot close a cycle.
        let mut mesh = LoopMesh::new();
        let a = mesh.create_vertex(Vec3::new(0.0, 0.0, 0.0));
        let b = mesh.create_vertex(Vec3::new(1.0, 0.0, 0.0));
        let c = mesh.create_vertex(Vec3::new(0.5, 1.0, 0.0));
        let e0 = mesh.create_edge([a, b], EdgeCreateFlags::NO_DUPLICATE);
        let e1 = mesh.create_edge([b, c], EdgeCreateFlags::NO_DUPLICATE);

        let result = mesh.edgenet_fill(&[e0, e1], &EdgenetFillParams::default());

        assert!(result.is_empty(), "sub-three-edge input fills nothing");
        assert_eq!(mesh.face_count(), 0);
        mesh.validate_topology();
    }
}
