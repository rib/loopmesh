use glam::Vec3;

use super::duplicate::GeomKey;
use crate::loopmesh::*;

/// Which elements the bevel replaces (BEVEL.md §1.2 / §2.2).
///
/// Only [`BevelAffect::Vertices`] is implemented today — the foundational
/// single-segment vertex bevel of BEVEL.md §3–§6. [`BevelAffect::Edges`] is a
/// documented-as-future stub (BEVEL.md §7).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum BevelAffect {
    Vertices,
    Edges,
}

impl Default for BevelAffect {
    fn default() -> Self {
        BevelAffect::Vertices
    }
}

/// How `offset` is interpreted (BEVEL.md §1.2 / §4). Only [`OffsetType::Offset`]
/// — the plain slide-along-the-edge distance — is implemented (BEVEL.md §4); the
/// other four modes re-measure the slide from a different target quantity and
/// are §7 stubs.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum OffsetType {
    Offset,
}

impl Default for OffsetType {
    fn default() -> Self {
        OffsetType::Offset
    }
}

/// Parameters for [`LoopMesh::bevel`]. The foundational case (BEVEL.md §3) is
/// reachable by setting only `offset` and leaving every other field at its
/// default: `affect = Vertices`, `segments = 1`, `offset_type = Offset`.
#[derive(Clone, Copy, Debug)]
pub struct BevelOptions {
    /// How much material the bevel consumes, measured per `offset_type`. A
    /// non-positive offset is a no-op (BEVEL.md §2.1).
    pub offset: f32,
    pub offset_type: OffsetType,
    pub segments: u32,
    pub affect: BevelAffect,
}

impl Default for BevelOptions {
    fn default() -> Self {
        BevelOptions {
            offset: 0.0,
            offset_type: OffsetType::Offset,
            segments: 1,
            affect: BevelAffect::Vertices,
        }
    }
}

/// New geometry created by the bevel, mirroring the operator's
/// `verts.out` / `edges.out` / `faces.out` slots (BEVEL.md §1.1). The buffers
/// are sets; downstream comparison ignores their enumeration order (§8).
#[derive(Default)]
pub struct BevelResult {
    pub verts: Vec<VertexKey>,
    pub edges: Vec<EdgeKey>,
    pub faces: Vec<FaceKey>,
}

/// One incident face of the beveled vertex, captured before any surgery: its
/// loop-cycle vertices in winding order and the index at which the beveled
/// vertex `v` sits in that cycle.
struct IncidentFaceCorner {
    face: FaceKey,
    cycle_verts: Vec<VertexKey>,
    corner_index: usize,
}

impl LoopMesh {
    /// Foundational single-segment vertex bevel (BEVEL.md §3–§6).
    ///
    /// For each beveled vertex `v` of valence `n`, this replaces `v` with a ring
    /// of `n` new verts — one per incident edge, slid `offset` toward the far
    /// endpoint (§4) — re-stitches each incident face's `v` corner onto the two
    /// ring verts of that face's bounding edges (§6.1), and, for an interior
    /// (closed-fan) corner, closes the ring with a single cap n-gon (§5/§5.1).
    /// A boundary (open-fan) corner gets no cap (§6.2). The original vertex is
    /// then removed as a now-loose vertex (§5.1.1).
    ///
    /// Only the foundational case is implemented: `affect = Vertices`,
    /// `segments = 1`, `offset_type = Offset`. Edge bevel, multiple segments,
    /// the other offset modes, profiles, miters, overlap clamping, and flag
    /// propagation are documented-as-future stubs (BEVEL.md §7) and are ignored
    /// here.
    ///
    /// A non-positive `offset`, or a beveled vertex with fewer than two incident
    /// edges, is a no-op for that vertex (BEVEL.md §2.1).
    ///
    /// See `radial-mesh-guides/operators/BEVEL.md`.
    pub fn bevel(&mut self, geom: &[GeomKey], options: BevelOptions) -> BevelResult {
        let mut result = BevelResult::default();

        if options.affect != BevelAffect::Vertices
            || options.segments != 1
            || options.offset_type != OffsetType::Offset
            || options.offset <= 0.0
        {
            return result;
        }

        for entry in geom {
            if let GeomKey::Vert(vertex) = *entry {
                self.bevel_one_vertex(vertex, options.offset, &mut result);
            }
        }

        result
    }

    fn bevel_one_vertex(&mut self, vertex: VertexKey, offset: f32, result: &mut BevelResult) {
        // §3.2 / §5: gather the incident edges (the disk cycle) and a ring vertex
        // per edge, slid `offset` toward the edge's far endpoint (§4).
        let incident_edges: Vec<EdgeKey> = self.iter_disk_edges(vertex).collect();
        if incident_edges.len() < 2 {
            return;
        }

        let vertex_position = self.vertex_position(vertex);
        let mut ring_vert_for_edge: Vec<(EdgeKey, VertexKey)> =
            Vec::with_capacity(incident_edges.len());
        for &edge in &incident_edges {
            let [a, b] = self.edge_vert_keys(edge);
            let far = if a == vertex { b } else { a };
            let far_position = self.vertex_position(far);
            let ring_position = ring_vert_position(vertex_position, far_position, offset);
            let ring_vertex = self.create_vertex(ring_position);
            ring_vert_for_edge.push((edge, ring_vertex));
            result.verts.push(ring_vertex);
        }

        // §6: every face that has `v` as a corner. Snapshot each face's loop
        // cycle before any surgery, since the rebuild below frees the old faces.
        let incident_faces: Vec<FaceKey> = self.vert_face_keys(vertex);
        let interior_fan = self.is_interior_fan(vertex, &incident_edges, &incident_faces);

        let corners: Vec<IncidentFaceCorner> = incident_faces
            .iter()
            .filter_map(|&face| {
                let cycle_verts = self.face_vert_keys(face);
                cycle_verts
                    .iter()
                    .position(|&w| w == vertex)
                    .map(|corner_index| IncidentFaceCorner {
                        face,
                        cycle_verts,
                        corner_index,
                    })
            })
            .collect();

        // §6.1: re-stitch each incident face. The single corner `v` becomes the
        // pair (ring vert of the incoming edge, ring vert of the outgoing edge),
        // in the order that keeps the face's winding. The chamfer edge between
        // those two ring verts is minted here as the face's new corner side.
        for corner in &corners {
            let cycle_length = corner.cycle_verts.len();
            let prev_vertex = corner.cycle_verts[(corner.corner_index + cycle_length - 1)
                % cycle_length];
            let next_vertex = corner.cycle_verts[(corner.corner_index + 1) % cycle_length];

            let incoming_edge = self
                .query_edge_exists(prev_vertex, vertex)
                .expect("incident face's incoming corner edge exists");
            let outgoing_edge = self
                .query_edge_exists(vertex, next_vertex)
                .expect("incident face's outgoing corner edge exists");

            let ring_incoming = self.ring_vertex_of(&ring_vert_for_edge, incoming_edge);
            let ring_outgoing = self.ring_vertex_of(&ring_vert_for_edge, outgoing_edge);

            let mut new_cycle: Vec<VertexKey> = Vec::with_capacity(cycle_length + 1);
            for (index, &w) in corner.cycle_verts.iter().enumerate() {
                if index == corner.corner_index {
                    new_cycle.push(ring_incoming);
                    new_cycle.push(ring_outgoing);
                } else {
                    new_cycle.push(w);
                }
            }

            self.face_kill(corner.face);
            let new_face = self.face_create_verts(&new_cycle, FaceCreateFlags::NONE);
            self.face_attribs.copy_into(new_face.as_usize(), corner.face.as_usize());
        }

        // §5 / §6.2: close an interior fan with a cap over the ring. The cap
        // adopts the chamfer edges the re-stitch already built — never mints a
        // second coincident edge (§5.1) — and is wound opposite the incident
        // faces so each shared chamfer edge ends with a manifold radial count of
        // two. A boundary open fan gets no cap.
        if interior_fan {
            if let Some(cap) = self.build_cap_face(&ring_vert_for_edge, &corners, vertex) {
                result.faces.push(cap.face);
                result.edges.extend(cap.chamfer_edges);
            }
        } else {
            // The open fan still produced chamfer edges during the re-stitch;
            // surface them as new geometry.
            self.collect_chamfer_edges(&ring_vert_for_edge, result);
        }

        // §5.1.1 / §6.3: every incident face is now re-stitched onto ring verts
        // and each original incident edge was re-pointed off `v`, so `v` is
        // loose. Free it (an empty-disk vertex removal, not an Euler join).
        self.vertex_kill(vertex);
    }

    fn ring_vertex_of(
        &self,
        ring_vert_for_edge: &[(EdgeKey, VertexKey)],
        edge: EdgeKey,
    ) -> VertexKey {
        ring_vert_for_edge
            .iter()
            .find_map(|&(e, r)| (e == edge).then_some(r))
            .expect("every incident edge has a ring vertex")
    }

    /// Classify the corner as an interior (closed) fan vs a boundary (open) fan
    /// (BEVEL.md §2.3 / §6). A corner is interior when every incident edge is
    /// manifold (exactly two incident faces) and the fan closes — the count of
    /// incident faces equals the count of incident edges. Any boundary edge
    /// (radial count below two) opens the fan.
    fn is_interior_fan(
        &self,
        vertex: VertexKey,
        incident_edges: &[EdgeKey],
        incident_faces: &[FaceKey],
    ) -> bool {
        let _ = vertex;
        let all_manifold = incident_edges
            .iter()
            .all(|&edge| self.edge_radial_iter(edge).count() == 2);
        all_manifold && incident_faces.len() == incident_edges.len()
    }

    /// Walk the disk's chamfer edges into an ordered ring and build the cap face
    /// over them, wound opposite the incident faces (BEVEL.md §5.1).
    fn build_cap_face(
        &mut self,
        ring_vert_for_edge: &[(EdgeKey, VertexKey)],
        corners: &[IncidentFaceCorner],
        vertex: VertexKey,
    ) -> Option<CapFace> {
        let _ = (corners, vertex);
        let ring_verts: Vec<VertexKey> = ring_vert_for_edge.iter().map(|&(_, r)| r).collect();

        // The chamfer edges minted by the re-stitch connect ring verts pairwise.
        // For an interior fan they form one closed cycle through all ring verts.
        let chamfer_edges = self.chamfer_edges_among(&ring_verts);
        if chamfer_edges.len() != ring_verts.len() {
            return None;
        }

        let ordered_ring = order_ring_cycle(self, &ring_verts, &chamfer_edges)?;

        // §5.1: wind the cap opposite to how the incident faces traverse the
        // chamfer edges. Take the first chamfer edge's single incident loop and
        // read its direction; if the face goes a -> b on that edge, the cap goes
        // b -> a, so reverse the ring whenever it currently runs the face's way.
        let first_a = ordered_ring[0];
        let first_b = ordered_ring[1];
        let shared_edge = self
            .query_edge_exists(first_a, first_b)
            .expect("ordered ring pair shares a chamfer edge");
        let face_runs_forward = {
            let incident_loops: Vec<LoopKey> = self.edge_loop_keys(shared_edge);
            incident_loops
                .first()
                .map(|&l| {
                    let start = self.loop_vertex(l);
                    let end = self.loop_vertex(self.loop_next(l));
                    start == first_a && end == first_b
                })
                .unwrap_or(false)
        };

        let mut cap_ring = ordered_ring;
        if face_runs_forward {
            cap_ring.reverse();
        }

        let cap_edges: Vec<EdgeKey> = (0..cap_ring.len())
            .map(|i| {
                let a = cap_ring[i];
                let b = cap_ring[(i + 1) % cap_ring.len()];
                self.query_edge_exists(a, b)
                    .expect("cap reuses the existing chamfer edge")
            })
            .collect();

        let cap = self.face_create(&cap_ring, &cap_edges, FaceCreateFlags::NONE);

        let mut unique_edges = cap_edges;
        unique_edges.sort_by_key(|e| e.as_usize());
        unique_edges.dedup();

        Some(CapFace {
            face: cap,
            chamfer_edges: unique_edges,
        })
    }

    fn collect_chamfer_edges(
        &self,
        ring_vert_for_edge: &[(EdgeKey, VertexKey)],
        result: &mut BevelResult,
    ) {
        let ring_verts: Vec<VertexKey> = ring_vert_for_edge.iter().map(|&(_, r)| r).collect();
        result.edges.extend(self.chamfer_edges_among(&ring_verts));
    }

    /// Every edge that connects two ring verts — the chamfer edges minted by the
    /// re-stitch. Excludes the original incident edges (which connect a ring vert
    /// to a far neighbour outside the ring set).
    fn chamfer_edges_among(&self, ring_verts: &[VertexKey]) -> Vec<EdgeKey> {
        let mut edges: Vec<EdgeKey> = Vec::new();
        for i in 0..ring_verts.len() {
            for j in (i + 1)..ring_verts.len() {
                if let Some(edge) = self.query_edge_exists(ring_verts[i], ring_verts[j]) {
                    if !edges.contains(&edge) {
                        edges.push(edge);
                    }
                }
            }
        }
        edges
    }
}

struct CapFace {
    face: FaceKey,
    chamfer_edges: Vec<EdgeKey>,
}

#[cfg(test)]
mod tests {
    use glam::Vec3;

    use super::*;
    use crate::loopmesh::test_support::mesh_add_cube;

    fn counts(mesh: &LoopMesh) -> (usize, usize, usize) {
        (mesh.totvert(), mesh.totedge(), mesh.totface())
    }

    /// BEVEL.md §6.1 / §2.3: an interior valence-3 cube corner is replaced by
    /// three ring verts, a triangular cap, and three re-stitched faces.
    /// Count delta +2 V / +3 E / +1 F.
    #[test]
    fn interior_closed_fan_cube_corner() {
        let mut mesh = LoopMesh::new();
        mesh_add_cube(&mut mesh);
        let before = counts(&mesh);
        assert_eq!(before, (8, 12, 6));

        // v0 = (1, -1, -1): an interior valence-3 corner (Front/Left/Bottom).
        let corner = mesh
            .vert_keys()
            .into_iter()
            .find(|&v| mesh.vertex_position(v) == Vec3::new(1.0, -1.0, -1.0))
            .expect("cube corner exists");

        let result = mesh.bevel(
            &[corner.into()],
            BevelOptions {
                offset: 0.25,
                ..Default::default()
            },
        );

        assert_eq!(counts(&mesh), (10, 15, 7), "interior +2 V / +3 E / +1 F");
        assert_eq!(result.verts.len(), 3, "three ring verts");
        assert_eq!(result.faces.len(), 1, "single triangular cap");

        // The cap is a triangle (n = 3).
        let cap = result.faces[0];
        assert_eq!(mesh.face_vert_keys(cap).len(), 3, "triangular cap");

        // The original corner is gone.
        assert!(
            !mesh
                .vert_keys()
                .into_iter()
                .any(|v| v == corner),
            "beveled vertex removed",
        );

        mesh.validate_topology();
    }

    /// BEVEL.md §4: ring verts slide `offset` along each incident edge toward the
    /// far endpoint. A larger offset slides them proportionally further.
    #[test]
    fn ring_verts_slide_proportionally() {
        let make = |offset: f32| {
            let mut mesh = LoopMesh::new();
            mesh_add_cube(&mut mesh);
            let corner = mesh
                .vert_keys()
                .into_iter()
                .find(|&v| mesh.vertex_position(v) == Vec3::new(1.0, -1.0, -1.0))
                .unwrap();
            let result = mesh.bevel(
                &[corner.into()],
                BevelOptions {
                    offset,
                    ..Default::default()
                },
            );
            // Distance of each ring vert from the (known) original corner.
            let origin = Vec3::new(1.0, -1.0, -1.0);
            result
                .verts
                .iter()
                .map(|&v| (mesh.vertex_position(v) - origin).length())
                .collect::<Vec<_>>()
        };

        let small = make(0.25);
        let large = make(0.5);
        for distance in &small {
            assert!((distance - 0.25).abs() < 1.0e-5, "small slide is the offset");
        }
        for distance in &large {
            assert!((distance - 0.5).abs() < 1.0e-5, "large slide is the offset");
        }
    }

    /// BEVEL.md §6.2 / §2.3: a boundary valence-2 corner of a flat grid quad
    /// becomes an open chamfer — two ring verts, one chamfer edge, no cap.
    /// Count delta +1 V / +1 E / +0 F.
    #[test]
    fn boundary_open_fan_grid_corner() {
        let mut mesh = LoopMesh::new();
        let positions = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        ];
        let verts: Vec<VertexKey> = positions.iter().map(|&p| mesh.create_vertex(p)).collect();
        let edges: Vec<EdgeKey> = (0..4)
            .map(|i| mesh.create_edge([verts[i], verts[(i + 1) % 4]], EdgeCreateFlags::NONE))
            .collect();
        mesh.face_create(&verts, &edges, FaceCreateFlags::NONE);

        let before = counts(&mesh);
        assert_eq!(before, (4, 4, 1));

        let corner = verts[0];
        let result = mesh.bevel(
            &[corner.into()],
            BevelOptions {
                offset: 0.25,
                ..Default::default()
            },
        );

        assert_eq!(counts(&mesh), (5, 5, 1), "boundary +1 V / +1 E / +0 F");
        assert_eq!(result.verts.len(), 2, "two ring verts");
        assert!(result.faces.is_empty(), "no cap on an open fan");

        // The single original face is now a pentagon (quad's corner cut off).
        let face = mesh.face_keys()[0];
        assert_eq!(mesh.face_vert_keys(face).len(), 5, "corner cut into two");

        mesh.validate_topology();
    }

    /// BEVEL.md §2.1: a non-positive offset is a no-op.
    #[test]
    fn non_positive_offset_is_noop() {
        let mut mesh = LoopMesh::new();
        mesh_add_cube(&mut mesh);
        let before = counts(&mesh);
        let corner = mesh.vert_keys()[0];

        let result = mesh.bevel(
            &[corner.into()],
            BevelOptions {
                offset: 0.0,
                ..Default::default()
            },
        );

        assert_eq!(counts(&mesh), before, "offset <= 0 changes nothing");
        assert!(result.verts.is_empty() && result.edges.is_empty() && result.faces.is_empty());
        mesh.validate_topology();
    }

    /// BEVEL.md §7: edge bevel and multi-segment are unimplemented stubs — they
    /// are no-ops here, not silent wrong results.
    #[test]
    fn unsupported_modes_are_noops() {
        let mut mesh = LoopMesh::new();
        mesh_add_cube(&mut mesh);
        let before = counts(&mesh);
        let corner = mesh.vert_keys()[0];

        let edges_mode = mesh.bevel(
            &[corner.into()],
            BevelOptions {
                offset: 0.25,
                affect: BevelAffect::Edges,
                ..Default::default()
            },
        );
        assert_eq!(counts(&mesh), before);
        assert!(edges_mode.verts.is_empty());

        let multi_segment = mesh.bevel(
            &[corner.into()],
            BevelOptions {
                offset: 0.25,
                segments: 2,
                ..Default::default()
            },
        );
        assert_eq!(counts(&mesh), before);
        assert!(multi_segment.verts.is_empty());
        mesh.validate_topology();
    }
}

fn ring_vert_position(vertex: Vec3, far: Vec3, offset: f32) -> Vec3 {
    // §4: slide `offset` along the edge toward the far endpoint. §4.1: guard the
    // degenerate overshoot — a non-positive or near-zero edge length cannot host
    // a slide, so the ring vert stays at the original corner.
    let direction = far - vertex;
    let length = direction.length();
    if length <= f32::EPSILON {
        return vertex;
    }
    let clamped = offset.min(length);
    vertex + (direction / length) * clamped
}

/// Order the ring verts into a cycle by following the chamfer edges (BEVEL.md
/// §5.1 — the disk order). Starts from `ring_verts[0]` and walks the chamfer
/// adjacency; returns `None` if the chamfer edges do not form a single closed
/// cycle over all ring verts (an unexpected non-manifold corner).
fn order_ring_cycle(
    mesh: &LoopMesh,
    ring_verts: &[VertexKey],
    chamfer_edges: &[EdgeKey],
) -> Option<Vec<VertexKey>> {
    let mut neighbours: Vec<(VertexKey, VertexKey)> = Vec::with_capacity(chamfer_edges.len());
    for &edge in chamfer_edges {
        let [a, b] = mesh.edge_vert_keys(edge);
        neighbours.push((a, b));
    }

    let adjacent = |current: VertexKey, skip: VertexKey| -> Option<VertexKey> {
        for &(a, b) in &neighbours {
            if a == current && b != skip {
                return Some(b);
            }
            if b == current && a != skip {
                return Some(a);
            }
        }
        None
    };

    let start = ring_verts[0];
    let mut ordered = vec![start];
    // First step: pick either chamfer neighbour of `start`.
    let mut previous = start;
    let mut current = neighbours
        .iter()
        .find_map(|&(a, b)| {
            if a == start {
                Some(b)
            } else if b == start {
                Some(a)
            } else {
                None
            }
        })?;

    while current != start {
        ordered.push(current);
        let next = adjacent(current, previous)?;
        previous = current;
        current = next;
        if ordered.len() > ring_verts.len() {
            return None;
        }
    }

    if ordered.len() == ring_verts.len() {
        Some(ordered)
    } else {
        None
    }
}
