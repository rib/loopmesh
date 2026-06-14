use glam::Vec3;

use crate::loopmesh::*;

impl LoopMesh {
    /// Insert a single loose vertex at `co` and return it (CREATE_VERT.md §3).
    ///
    /// Create-vert is the smallest constructive operator: it makes exactly one
    /// new vertex at the supplied coordinate and touches nothing else. There is
    /// no edge, no face, and no walk — `ΔV = +1`, `ΔE = ΔLoop = ΔF = 0` (§3.2).
    ///
    /// The new vertex starts from the no-example primitive, so every per-vertex
    /// customdata layer reads back that layer's **default** value (never a copied
    /// or interpolated one — there is no template vertex), the cached normal is
    /// the zero vector, and the persistent header flags are cleared (§2). The
    /// operator never de-duplicates or snaps: two calls with the same `co` make
    /// two distinct coincident vertices (§4). Any finite coordinate is valid; the
    /// call always succeeds with exactly one new vertex (§3.2, §6).
    ///
    /// Whether to then select the returned vertex is the caller's concern, exactly
    /// as the bare BMOP leaves selection to its editor wrapper (§2.3).
    ///
    /// See `radial-mesh-guides/operators/CREATE_VERT.md` for the full contract.
    pub fn create_vert(&mut self, co: Vec3) -> VertexKey {
        self.create_vertex(co)
    }
}

#[cfg(test)]
mod tests {
    use glam::Vec3;

    use crate::loopmesh::*;

    /// §1 / §3 / §5.2 headline: create one vertex in an empty mesh; the mesh holds
    /// exactly one loose vertex at `co` and nothing else.
    #[test]
    fn empty_mesh_single_vertex_at_co() {
        let mut mesh = LoopMesh::new();
        let co = Vec3::new(1.5, -2.0, 0.25);

        let v = mesh.create_vert(co);

        assert_eq!(
            (mesh.vertex_count(), mesh.edge_count(), mesh.face_count()),
            (1, 0, 0)
        );
        assert_eq!(mesh.vertex_position(v), co);
        // A loose vertex has no incident faces, so its cached normal is zeroed
        // (§2.2) — not derived from `co`.
        assert_eq!(mesh.vertex_normal(v), Vec3::ZERO);
        mesh.validate_topology();
    }

    /// §3.2 / §5.2: creating a loose vertex off to the side of existing geometry is
    /// purely additive — the original quad is untouched (`ΔE = ΔF = 0`) and one new
    /// isolated vertex appears at `co`.
    #[test]
    fn populated_mesh_is_additive() {
        let mut mesh = LoopMesh::new();
        let quad_corners = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        ];
        let quad_verts: Vec<VertexKey> = quad_corners
            .iter()
            .map(|&p| mesh.create_vertex(p))
            .collect();
        let quad_edges: Vec<EdgeKey> = (0..4)
            .map(|i| {
                mesh.create_edge(
                    [quad_verts[i], quad_verts[(i + 1) % 4]],
                    EdgeCreateFlags::NO_DUPLICATE,
                )
            })
            .collect();
        mesh.face_create(&quad_verts, &quad_edges, FaceCreateFlags::NONE);

        let before = (mesh.vertex_count(), mesh.edge_count(), mesh.face_count());
        assert_eq!(before, (4, 4, 1));

        let co = Vec3::new(5.0, 5.0, 5.0);
        let v = mesh.create_vert(co);

        // Exactly one new vertex; edges and faces unchanged.
        assert_eq!(
            (mesh.vertex_count(), mesh.edge_count(), mesh.face_count()),
            (5, 4, 1)
        );
        assert_eq!(mesh.vertex_position(v), co);
        // The original corners are where they were.
        for (corner, &vert) in quad_verts.iter().enumerate() {
            assert_eq!(mesh.vertex_position(vert), quad_corners[corner]);
        }
        mesh.validate_topology();
    }

    /// §4 / §5.2: two creates at the *same* `co` make **two** distinct coincident
    /// loose vertices — create-vert never merges on create.
    #[test]
    fn two_at_same_co_make_two_vertices() {
        let mut mesh = LoopMesh::new();
        let co = Vec3::new(3.0, 3.0, 3.0);

        let a = mesh.create_vert(co);
        let b = mesh.create_vert(co);

        assert_ne!(a, b);
        assert_eq!(
            (mesh.vertex_count(), mesh.edge_count(), mesh.face_count()),
            (2, 0, 0)
        );
        assert_eq!(mesh.vertex_position(a), co);
        assert_eq!(mesh.vertex_position(b), co);
        mesh.validate_topology();
    }

    /// §2.2 / §5.1: with a per-vertex float CD layer registered, the new vertex
    /// reads the layer **default** (zero fill), never a copied or interpolated
    /// value — the operator takes no example vertex.
    #[test]
    fn new_vertex_carries_cd_default() {
        let mut mesh = LoopMesh::new();

        let weight = mesh
            .register_vertex_attribute(ElementAttributeDesc::numeric(
                AttributeKey::custom("weight"),
                AttrType::F32,
            ))
            .expect("vert weight layer registers");

        // An existing vertex carrying a non-default value, to prove nothing is
        // copied from it onto the new vertex.
        let existing = mesh.create_vert(Vec3::ZERO);
        mesh.vertex_attribs_mut()
            .set_f32(weight, existing.as_usize(), 7.0);

        let v = mesh.create_vert(Vec3::new(2.0, 0.0, 0.0));

        let stored = mesh.vertex_attribs().get_f32(weight, v.as_usize()).unwrap();
        assert_eq!(
            stored, 0.0,
            "new vertex reads the layer default, not a copy"
        );
        mesh.validate_topology();
    }
}
