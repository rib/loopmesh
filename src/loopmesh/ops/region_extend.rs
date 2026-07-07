use std::collections::HashSet;

use crate::loopmesh::*;

/// Parameters for [`LoopMesh::region_extend`] (REGION_EXTEND.md §1.2). The three
/// booleans together pick the walk: direction (`use_contract`), the element kind
/// that seeds the walk and what the ring is made of (`use_faces`), and which
/// incidence cycle each seed steps along (`use_face_step`).
#[derive(PartialEq, Eq, Clone, Copy, Debug, Default)]
pub struct RegionExtendOptions {
    /// `false` = grow (select-more), reporting the next ring *outside* the
    /// region; `true` = shrink (select-less), reporting the region's inner
    /// boundary ring (REGION_EXTEND.md §1.2, §5).
    pub use_contract: bool,
    /// `false` = vert-based: seed input verts, ring is verts + crossing edges;
    /// `true` = face-based: seed input faces, ring is faces only
    /// (REGION_EXTEND.md §3.1 vs §3.2).
    pub use_faces: bool,
    /// Switches each seed's step from the edge incidence to the face incidence —
    /// orthogonal vs diagonal neighbours (REGION_EXTEND.md §4, §3.2).
    pub use_face_step: bool,
}

/// Transient ORIG / RING bookkeeping for one region-extend run (REGION_EXTEND.md
/// §2.1). ORIG records "this element was already in the input region" so the walk
/// can tell the region's interior from its frontier; RING is set on the elements
/// the walk decides belong to the reported ring. The harvest reads RING.
///
/// These are run-local scratch sets — region-extend writes no persistent header
/// flag (REGION_EXTEND.md §2.4), so nothing here touches the mesh.
#[derive(Default)]
struct Marks {
    orig: HashSet<GeomKey>,
    ring: HashSet<GeomKey>,
}

impl Marks {
    fn set_orig(&mut self, g: GeomKey) {
        self.orig.insert(g);
    }
    fn is_orig(&self, g: GeomKey) -> bool {
        self.orig.contains(&g)
    }
    fn is_ring(&self, g: GeomKey) -> bool {
        self.ring.contains(&g)
    }
    /// Mark `g` RING. The set already dedups an element reached via several
    /// incidences, so the harvest reports it once; the parity compare is an
    /// order-independent position-keyed multiset (REGION_EXTEND.md §7), so no
    /// insertion-order bookkeeping is needed.
    fn set_ring(&mut self, g: GeomKey) {
        self.ring.insert(g);
    }
}

impl LoopMesh {
    /// Compute **one ring** of incident geometry to add to (grow) or peel from
    /// (shrink) the input region, and report it. This is BMesh's *select-more /
    /// select-less* kernel — a pure topological query that creates, kills, and
    /// moves nothing (`ΔV = ΔE = ΔF = 0`) and writes no persistent header flag
    /// (REGION_EXTEND.md §6.1, §2.4). The only result is the returned ring.
    ///
    /// `geom` is a mixed buffer of verts / edges / faces — the currently
    /// selected region. The walk seeds only the kind the parameters select
    /// (`use_faces = false` reads the input **verts**, `use_faces = true` reads
    /// the input **faces**); the other input kinds still count as "inside the
    /// region" for the frontier / boundary tests but are never themselves a seed
    /// (REGION_EXTEND.md §8, mixed input).
    ///
    /// The returned ring's contents differ by direction (REGION_EXTEND.md §5):
    /// **grow** reports the newly-reached geometry, disjoint from the input;
    /// **shrink** reports the input region's boundary elements, a subset of the
    /// input.
    ///
    /// See `radial-mesh-guides/operators/REGION_EXTEND.md` for the full walk and
    /// parity contract.
    pub fn region_extend(&self, geom: &[GeomKey], opts: RegionExtendOptions) -> Vec<GeomKey> {
        let mut marks = Marks::default();

        for &g in geom {
            marks.set_orig(g);
        }

        if opts.use_contract {
            if opts.use_faces {
                for &g in geom {
                    if let GeomKey::Face(f) = g {
                        self.shrink_face(f, opts.use_face_step, &mut marks);
                    }
                }
            } else {
                for &g in geom {
                    if let GeomKey::Vert(v) = g {
                        self.shrink_vert(v, opts.use_face_step, &mut marks);
                    }
                }
            }
        } else if opts.use_faces {
            for &g in geom {
                if let GeomKey::Face(f) = g {
                    self.grow_face(f, opts.use_face_step, &mut marks);
                }
            }
        } else {
            for &g in geom {
                if let GeomKey::Vert(v) = g {
                    if opts.use_face_step {
                        self.grow_vert_face_step(v, &mut marks);
                    } else {
                        self.grow_vert_edge_step(v, &mut marks);
                    }
                }
            }
        }

        marks.ring.into_iter().collect()
    }

    /// Allocation-free walk of the faces incident to `v` — the disk→radial
    /// composition of BMesh's `BM_FACES_OF_VERT`. A face touched by several of
    /// `v`'s disk edges is yielded once per incidence (no dedup); every
    /// region-extend caller either dedups through the RING guard or only asks
    /// `any`, so the repeats are harmless and save a per-corner heap `Vec`.
    fn vert_incident_faces(&self, v: VertexKey) -> impl Iterator<Item = FaceKey> + '_ {
        self.vertex_disk_iter(v)
            .flat_map(move |e| self.edge_radial_iter(e).map(move |l| self.loop_face(l)))
    }

    /// Allocation-free walk of the faces in `e`'s radial cycle (one per loop,
    /// repeating a face incident twice on a non-manifold edge) — the
    /// non-collecting form of `edge_face_keys`.
    fn edge_incident_faces(&self, e: EdgeKey) -> impl Iterator<Item = FaceKey> + '_ {
        self.edge_radial_iter(e).map(move |l| self.loop_face(l))
    }

    /// REGION_EXTEND.md §2.3 — a candidate neighbour on the **grow** path is
    /// skipped when its persistent hidden bit is set. (Shrink consults no hidden
    /// bit; its walk never calls this — see §5.2.)
    fn edge_hidden(&self, e: EdgeKey) -> bool {
        self.edges.get(e).eflags.contains(ElementFlags::HIDDEN)
    }
    fn face_hidden(&self, f: FaceKey) -> bool {
        self.faces.get(f).eflags.contains(ElementFlags::HIDDEN)
    }

    /// A wire edge bounds no face, so it is invisible to a faces-only step and
    /// gets its own pass (REGION_EXTEND.md §4.1).
    fn edge_is_wire(&self, e: EdgeKey) -> bool {
        self.edges.get(e).radial_link.is_none()
    }

    fn other_endpoint(&self, e: EdgeKey, v: VertexKey) -> VertexKey {
        self.edges
            .get(e)
            .other_vertex(v)
            .expect("edge in v's disk must contain v")
    }

    /// REGION_EXTEND.md §3.1 — vert-mode grow, edge step. A frontier vertex (one
    /// with a non-hidden incident edge that is not ORIG) rings each of its
    /// non-hidden incident edges and that edge's far endpoint. The frontier test
    /// keys off the *edge*'s ORIG mark, and a vert grow flags only verts ORIG, so
    /// an edge spanning two originally-selected verts is itself non-ORIG and gets
    /// reported as a crossing edge (the §3.1 fill-in subtlety).
    fn grow_vert_edge_step(&self, v: VertexKey, marks: &mut Marks) {
        let on_frontier = self.vertex_disk_iter(v).any(|e| {
            !marks.is_orig(GeomKey::Edge(e)) && !self.edge_hidden(e)
        });
        if !on_frontier {
            return;
        }
        for e in self.vertex_disk_iter(v) {
            if self.edge_hidden(e) {
                continue;
            }
            marks.set_ring(GeomKey::Edge(e));
            marks.set_ring(GeomKey::Vert(self.other_endpoint(e, v)));
        }
    }

    /// REGION_EXTEND.md §4.1 — vert-mode grow, face step. The frontier test is
    /// the same as the edge step; the step then rings every non-hidden incident
    /// **face** and flushes the mark to the face's whole rim (its verts and
    /// edges). Wire edges have no face, so they are picked up by a dedicated pass
    /// exactly like the edge step, ensuring a vertex still grows along a wire.
    fn grow_vert_face_step(&self, v: VertexKey, marks: &mut Marks) {
        let on_frontier = self.vertex_disk_iter(v).any(|e| {
            !marks.is_orig(GeomKey::Edge(e)) && !self.edge_hidden(e)
        });
        if !on_frontier {
            return;
        }
        for f in self.vert_incident_faces(v) {
            if marks.is_ring(GeomKey::Face(f)) || self.face_hidden(f) {
                continue;
            }
            marks.set_ring(GeomKey::Face(f));
            let start = self.face_first_loop(f);
            for l in self.face_loop_iter(start) {
                marks.set_ring(GeomKey::Edge(self.loop_edge(l)));
                marks.set_ring(GeomKey::Vert(self.loop_vertex(l)));
            }
        }
        for e in self.vertex_disk_iter(v) {
            if self.edge_is_wire(e) && !self.edge_hidden(e) {
                marks.set_ring(GeomKey::Edge(e));
                marks.set_ring(GeomKey::Vert(self.other_endpoint(e, v)));
            }
        }
    }

    /// REGION_EXTEND.md §3.2 — face-mode grow. For each corner of the seed face,
    /// visit the neighbour faces across that corner's edge (edge step) or that
    /// corner's vertex (vertex step, reaching diagonal faces that share only the
    /// corner). Each non-ORIG, non-hidden neighbour is ringed. The ring is faces
    /// only; the original faces are excluded by the ORIG guard (§5.1).
    fn grow_face(&self, f: FaceKey, use_face_step: bool, marks: &mut Marks) {
        let start = self.face_first_loop(f);
        for l in self.face_loop_iter(start) {
            if use_face_step {
                let corner_vert = self.loop_vertex(l);
                for nf in self.vert_incident_faces(corner_vert) {
                    self.try_ring_grow_face(nf, marks);
                }
            } else {
                let corner_edge = self.loop_edge(l);
                for nf in self.edge_incident_faces(corner_edge) {
                    self.try_ring_grow_face(nf, marks);
                }
            }
        }
    }

    fn try_ring_grow_face(&self, nf: FaceKey, marks: &mut Marks) {
        let g = GeomKey::Face(nf);
        if !marks.is_orig(g) && !marks.is_ring(g) && !self.face_hidden(nf) {
            marks.set_ring(g);
        }
    }

    /// REGION_EXTEND.md §5.2 — vert-mode shrink. A vertex is a boundary vertex
    /// when it has an incident element leading *out* of the region: a non-ORIG
    /// incident edge (edge step), or a non-ORIG incident face / wire edge (face
    /// step). Every boundary vertex is ringed along with **all** its incident
    /// edges. Shrink consults no hidden bit (§5.2), the opposite of grow's §2.3
    /// skip.
    fn shrink_vert(&self, v: VertexKey, use_face_step: bool, marks: &mut Marks) {
        let on_boundary = if use_face_step {
            let face_out = self
                .vert_incident_faces(v)
                .any(|f| !marks.is_orig(GeomKey::Face(f)));
            let wire_out = self
                .vertex_disk_iter(v)
                .any(|e| self.edge_is_wire(e) && !marks.is_orig(GeomKey::Edge(e)));
            face_out || wire_out
        } else {
            self.vertex_disk_iter(v)
                .any(|e| !marks.is_orig(GeomKey::Edge(e)))
        };
        if !on_boundary {
            return;
        }
        marks.set_ring(GeomKey::Vert(v));
        for e in self.vertex_disk_iter(v) {
            marks.set_ring(GeomKey::Edge(e));
        }
    }

    /// REGION_EXTEND.md §5.2 — face-mode shrink. A face is a boundary face when
    /// some corner's edge-adjacent (edge step) or vertex-adjacent (vertex step)
    /// neighbour is non-ORIG (outside the region). One outside neighbour is
    /// enough to ring `f`. Shrink consults no hidden bit (§5.2).
    fn shrink_face(&self, f: FaceKey, use_face_step: bool, marks: &mut Marks) {
        let start = self.face_first_loop(f);
        let mut on_boundary = false;
        for l in self.face_loop_iter(start) {
            let has_outside = if use_face_step {
                let corner_vert = self.loop_vertex(l);
                self.vert_incident_faces(corner_vert)
                    .any(|nf| !marks.is_orig(GeomKey::Face(nf)))
            } else {
                let corner_edge = self.loop_edge(l);
                self.edge_incident_faces(corner_edge)
                    .any(|nf| !marks.is_orig(GeomKey::Face(nf)))
            };
            if has_outside {
                on_boundary = true;
                break;
            }
        }
        if on_boundary {
            marks.set_ring(GeomKey::Face(f));
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use glam::Vec3;

    use crate::loopmesh::*;

    /// A `dim`×`dim` vertex lattice of unit quads. `verts[r * dim + c]`,
    /// `faces[r * (dim - 1) + c]`. With `dim = 3` this is the scaffold's
    /// `grid_3x3` (9 verts, 12 edges, 4 quads, centre vert index 4); with
    /// `dim = 4` it is `grid_3x3_faces` (16 verts, 24 edges, 9 quads, centre
    /// face index 4).
    fn vertex_lattice(dim: usize) -> (LoopMesh, Vec<VertexKey>, Vec<FaceKey>) {
        let mut mesh = LoopMesh::new();
        let mut verts = Vec::with_capacity(dim * dim);
        for r in 0..dim {
            for c in 0..dim {
                verts.push(mesh.create_vertex(Vec3::new(c as f32, r as f32, 0.0)));
            }
        }
        let mut faces = Vec::new();
        for r in 0..dim - 1 {
            for c in 0..dim - 1 {
                let v_bl = verts[r * dim + c];
                let v_br = verts[r * dim + c + 1];
                let v_tr = verts[(r + 1) * dim + c + 1];
                let v_tl = verts[(r + 1) * dim + c];
                let e_b = mesh.create_edge([v_bl, v_br], EdgeCreateFlags::NO_DUPLICATE);
                let e_r = mesh.create_edge([v_br, v_tr], EdgeCreateFlags::NO_DUPLICATE);
                let e_t = mesh.create_edge([v_tr, v_tl], EdgeCreateFlags::NO_DUPLICATE);
                let e_l = mesh.create_edge([v_tl, v_bl], EdgeCreateFlags::NO_DUPLICATE);
                faces.push(mesh.face_create(
                    &[v_bl, v_br, v_tr, v_tl],
                    &[e_b, e_r, e_t, e_l],
                    FaceCreateFlags::NONE,
                ));
            }
        }
        (mesh, verts, faces)
    }

    fn ring_set(ring: &[GeomKey]) -> HashSet<GeomKey> {
        ring.iter().copied().collect()
    }

    /// `grid_3x3` (four quads sharing the centre vert 4) plus one face-less wire
    /// edge dangling off the centre vert; the wire's tip vertex is lifted off the
    /// grid plane so the edge is unambiguously face-less. Returns the mesh, the 9
    /// lattice verts, the 4 faces, the wire edge, and its tip vert.
    fn grid_3x3_wire_at_center(
    ) -> (LoopMesh, Vec<VertexKey>, Vec<FaceKey>, EdgeKey, VertexKey) {
        let (mut mesh, verts, faces) = vertex_lattice(3);
        let tip = mesh.create_vertex(Vec3::new(1.0, 1.0, 1.0));
        let wire = mesh.create_edge([verts[4], tip], EdgeCreateFlags::NO_DUPLICATE);
        (mesh, verts, faces, wire, tip)
    }

    fn vert_set(mesh: &LoopMesh, idx: &[usize], verts: &[VertexKey]) -> HashSet<GeomKey> {
        let _ = mesh;
        idx.iter().map(|&i| GeomKey::Vert(verts[i])).collect()
    }

    /// Vert-mode grow, edge step (foundational): centre vert of grid_3x3.
    /// REGION_EXTEND.md §3.1. The ring is the four crossing edges plus the four
    /// far verts {1, 3, 5, 7}, disjoint from the input (§5.1).
    #[test]
    fn vert_grow_grid_center_rings_four_edges_four_verts() {
        let (mesh, verts, _faces) = vertex_lattice(3);
        let ring = mesh.region_extend(
            &[GeomKey::Vert(verts[4])],
            RegionExtendOptions {
                use_contract: false,
                use_faces: false,
                use_face_step: false,
            },
        );
        let set = ring_set(&ring);
        assert_eq!(ring.len(), 8, "ring is 4 crossing edges + 4 far verts");

        let edge_count = set.iter().filter(|g| matches!(g, GeomKey::Edge(_))).count();
        let vert_count = set.iter().filter(|g| matches!(g, GeomKey::Vert(_))).count();
        assert_eq!(edge_count, 4);
        assert_eq!(vert_count, 4);

        let far = vert_set(&mesh, &[1, 3, 5, 7], &verts);
        assert!(
            far.is_subset(&set),
            "the four edge-adjacent verts are in the ring"
        );
        assert!(
            !set.contains(&GeomKey::Vert(verts[4])),
            "the seed vert itself is not reported (grow ring is disjoint from input)"
        );
    }

    /// The public HIDDEN setter flips the per-element bit, and the bit drives
    /// the direction-dependent hidden handling of `region_extend`
    /// (REGION_EXTEND.md §2.3 / §5.2): a hidden incident edge is **excluded**
    /// from a vert-mode grow ring but **included** in a vert-mode shrink ring.
    /// This pins the grow-skips / shrink-ignores asymmetry through the new API.
    #[test]
    fn hidden_edge_skipped_by_grow_but_not_shrink() {
        let (mut mesh, verts, _faces) = vertex_lattice(3);
        // The "east" edge out of the centre vert 4 → far vert 5.
        let east = mesh.query_edge_exists(verts[4], verts[5]).unwrap();

        // Setter toggles the bit and the getter agrees.
        assert!(!mesh.is_edge_hidden(east));
        mesh.set_edge_hidden(east, true);
        assert!(mesh.is_edge_hidden(east));

        // Grow from the centre vert: the hidden east edge and its far vert 5 are
        // skipped, so the ring shrinks to 3 edges + 3 far verts {1,3,7} (§2.3).
        let grow = ring_set(&mesh.region_extend(
            &[GeomKey::Vert(verts[4])],
            RegionExtendOptions::default(),
        ));
        assert!(
            !grow.contains(&GeomKey::Edge(east)),
            "grow skips the hidden edge"
        );
        assert!(
            !grow.contains(&GeomKey::Vert(verts[5])),
            "grow does not reach across the hidden edge"
        );
        assert_eq!(grow.len(), 6, "3 non-hidden edges + 3 far verts");

        // Shrink consults no hidden bit (§5.2): the centre vert is a boundary
        // vert and **all** its incident edges are ringed, hidden one included.
        let shrink = ring_set(&mesh.region_extend(
            &[GeomKey::Vert(verts[4])],
            RegionExtendOptions {
                use_contract: true,
                use_faces: false,
                use_face_step: false,
            },
        ));
        assert!(
            shrink.contains(&GeomKey::Edge(east)),
            "shrink ignores the hidden bit and still rings the hidden edge"
        );

        // Clearing the bit restores the full grow ring (8 elements).
        mesh.set_edge_hidden(east, false);
        assert!(!mesh.is_edge_hidden(east));
        let grow_cleared = mesh.region_extend(
            &[GeomKey::Vert(verts[4])],
            RegionExtendOptions::default(),
        );
        assert_eq!(grow_cleared.len(), 8, "cleared bit restores the full ring");
    }

    /// REGION_EXTEND.md §3.1 fill-in subtlety: an edge spanning two
    /// originally-selected verts is non-ORIG (only verts are flagged) and so is
    /// reported as a crossing edge.
    #[test]
    fn vert_grow_reports_edge_between_two_selected_verts() {
        let (mesh, verts, _faces) = vertex_lattice(3);
        // verts 0 and 1 are adjacent along the bottom row; the edge {0,1}
        // spans two selected verts.
        let edge01 = mesh.query_edge_exists(verts[0], verts[1]).unwrap();
        let ring = mesh.region_extend(
            &[GeomKey::Vert(verts[0]), GeomKey::Vert(verts[1])],
            RegionExtendOptions::default(),
        );
        assert!(
            ring_set(&ring).contains(&GeomKey::Edge(edge01)),
            "the edge spanning the two selected verts is filled in"
        );
    }

    /// Vert-mode grow, **face step** (REGION_EXTEND.md §4.1) seeded at the centre
    /// vert of a 3×3 grid carrying a face-less wire edge. The face step rings all
    /// four incident faces and flushes each face's whole rim, so every one of the
    /// grid's 9 verts and 12 edges and 4 faces is reported (25 elements). The
    /// dedicated wire pass then adds the wire edge and its tip vert — a 27-element
    /// mixed ring spanning all three element kinds.
    #[test]
    fn vert_grow_face_step_rings_incident_faces_rims_and_wire() {
        let (mesh, verts, faces, wire, tip) = grid_3x3_wire_at_center();
        let ring = mesh.region_extend(
            &[GeomKey::Vert(verts[4])],
            RegionExtendOptions {
                use_contract: false,
                use_faces: false,
                use_face_step: true,
            },
        );
        let set = ring_set(&ring);
        assert_eq!(ring.len(), 27, "9 verts + 12 edges + 4 faces + wire + tip");

        let vert_count = set.iter().filter(|g| matches!(g, GeomKey::Vert(_))).count();
        let edge_count = set.iter().filter(|g| matches!(g, GeomKey::Edge(_))).count();
        let face_count = set.iter().filter(|g| matches!(g, GeomKey::Face(_))).count();
        assert_eq!(vert_count, 10, "9 grid verts + wire tip");
        assert_eq!(edge_count, 13, "12 grid edges + wire edge");
        assert_eq!(face_count, 4, "all four incident faces");

        // Every grid vert/edge and all four faces are flushed in via the face step.
        for &v in &verts {
            assert!(set.contains(&GeomKey::Vert(v)), "grid vert flushed into ring");
        }
        for &f in &faces {
            assert!(set.contains(&GeomKey::Face(f)), "incident face ringed");
        }
        // The dedicated wire pass picks up the face-less wire and its tip (§4.1).
        assert!(set.contains(&GeomKey::Edge(wire)), "wire edge added by wire pass");
        assert!(set.contains(&GeomKey::Vert(tip)), "wire tip vert added by wire pass");
    }

    /// Contrast against [`vert_grow_face_step_rings_incident_faces_rims_and_wire`]:
    /// the **edge step** (REGION_EXTEND.md §3.1) at the same seed rings only the
    /// five incident edges (the four grid edges plus the wire) and their five far
    /// verts — no faces — a 10-element ring. The two rings must differ, so a
    /// regression that ignored `use_face_step` or dropped the face/wire pass fails.
    #[test]
    fn vert_grow_edge_step_rings_incident_edges_and_far_verts_no_faces() {
        let (mesh, verts, _faces, wire, tip) = grid_3x3_wire_at_center();
        let ring = mesh.region_extend(
            &[GeomKey::Vert(verts[4])],
            RegionExtendOptions {
                use_contract: false,
                use_faces: false,
                use_face_step: false,
            },
        );
        let set = ring_set(&ring);
        assert_eq!(ring.len(), 10, "5 incident edges + 5 far verts");

        let edge_count = set.iter().filter(|g| matches!(g, GeomKey::Edge(_))).count();
        let vert_count = set.iter().filter(|g| matches!(g, GeomKey::Vert(_))).count();
        let face_count = set.iter().filter(|g| matches!(g, GeomKey::Face(_))).count();
        assert_eq!(edge_count, 5, "four grid edges + wire edge");
        assert_eq!(vert_count, 5, "verts 1,3,5,7 + wire tip");
        assert_eq!(face_count, 0, "edge step rings no faces");

        // The wire edge participates in the ordinary edge step too (§3.1).
        assert!(set.contains(&GeomKey::Edge(wire)), "wire edge crossed by edge step");
        assert!(set.contains(&GeomKey::Vert(tip)), "wire tip reached by edge step");
        let far = vert_set(&mesh, &[1, 3, 5, 7], &verts);
        assert!(far.is_subset(&set), "the four grid-edge far verts are reported");

        // The face step and edge step report genuinely different rings.
        let face_step = ring_set(&mesh.region_extend(
            &[GeomKey::Vert(verts[4])],
            RegionExtendOptions {
                use_contract: false,
                use_faces: false,
                use_face_step: true,
            },
        ));
        assert_ne!(set, face_step, "use_face_step genuinely changes the vert-mode ring");
    }

    /// Face-mode grow, edge step (foundational): centre face of grid_3x3_faces.
    /// REGION_EXTEND.md §3.2. The ring is the four edge-adjacent faces (the
    /// plus shape {1, 3, 5, 7}); the four diagonals {0, 2, 6, 8} are not
    /// reached, and the seed face is excluded (§5.1).
    #[test]
    fn face_grow_edge_step_rings_plus_shape() {
        let (mesh, _verts, faces) = vertex_lattice(4);
        let ring = mesh.region_extend(
            &[GeomKey::Face(faces[4])],
            RegionExtendOptions {
                use_contract: false,
                use_faces: true,
                use_face_step: false,
            },
        );
        let set = ring_set(&ring);
        let expected: HashSet<GeomKey> =
            [1, 3, 5, 7].iter().map(|&i| GeomKey::Face(faces[i])).collect();
        assert_eq!(set, expected, "edge-step ring is the four orthogonal faces");
        assert!(!set.contains(&GeomKey::Face(faces[4])), "seed face excluded");
    }

    /// `use_face_step` face-mode contrast: the vertex step reaches all eight
    /// surrounding faces (orthogonal + diagonal), a strict superset of the
    /// edge-step ring. REGION_EXTEND.md §3.2 / §7.1.
    #[test]
    fn face_grow_vertex_step_reaches_diagonals() {
        let (mesh, _verts, faces) = vertex_lattice(4);
        let opts = |use_face_step| RegionExtendOptions {
            use_contract: false,
            use_faces: true,
            use_face_step,
        };
        let edge_step = ring_set(&mesh.region_extend(&[GeomKey::Face(faces[4])], opts(false)));
        let vertex_step = ring_set(&mesh.region_extend(&[GeomKey::Face(faces[4])], opts(true)));

        let expected: HashSet<GeomKey> = (0..9)
            .filter(|&i| i != 4)
            .map(|i| GeomKey::Face(faces[i]))
            .collect();
        assert_eq!(vertex_step, expected, "vertex step reaches all 8 neighbours");
        assert!(
            edge_step.is_subset(&vertex_step) && edge_step != vertex_step,
            "use_face_step genuinely changes the ring"
        );
    }

    /// Vert-mode shrink (`use_contract`): a corner patch's boundary. Every vert
    /// of the bottom-left quad {0, 1, 3, 4} of grid_3x3 has an edge leading out
    /// of the patch, so all four are boundary verts; the ring is a subset of the
    /// input (verts are ORIG *and* RING, §5.2) and differs from the grow ring.
    #[test]
    fn vert_shrink_patch_boundary_is_input_subset() {
        let (mesh, verts, _faces) = vertex_lattice(3);
        let patch = [0usize, 1, 3, 4];
        let region: Vec<GeomKey> = patch.iter().map(|&i| GeomKey::Vert(verts[i])).collect();

        let shrink = mesh.region_extend(
            &region,
            RegionExtendOptions {
                use_contract: true,
                use_faces: false,
                use_face_step: false,
            },
        );
        let shrink_set = ring_set(&shrink);
        assert!(!shrink_set.is_empty(), "corner patch has a boundary");

        // All four patch verts are boundary verts and appear in the ring.
        for &i in &patch {
            assert!(
                shrink_set.contains(&GeomKey::Vert(verts[i])),
                "boundary vert {i} is reported"
            );
        }

        // Direction contrast: grow of the same region differs from shrink.
        let grow = ring_set(&mesh.region_extend(&region, RegionExtendOptions::default()));
        assert_ne!(shrink_set, grow, "shrink and grow report different rings");
    }

    /// Face-mode shrink (`use_contract`): the inner boundary band of an interior
    /// block. REGION_EXTEND.md §5.2 / §6. Selecting the central 3×3 block of a
    /// 5×5 face grid and shrinking (edge step) reports exactly the band of block
    /// faces that touch a non-selected neighbour — every block face except the
    /// fully-interior centre face. The ring is a **strict subset** of the input
    /// (boundary faces are ORIG *and* RING, §5.2).
    #[test]
    fn face_shrink_block_inner_band_excludes_interior() {
        // dim = 6 vertex lattice → 5×5 face grid; face (r,c) is index r*5 + c.
        let (mesh, _verts, faces) = vertex_lattice(6);
        let block_rc = [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3), (3, 1), (3, 2), (3, 3)];
        let idx = |r: usize, c: usize| r * 5 + c;
        let region: Vec<GeomKey> = block_rc
            .iter()
            .map(|&(r, c)| GeomKey::Face(faces[idx(r, c)]))
            .collect();

        let shrink = mesh.region_extend(
            &region,
            RegionExtendOptions {
                use_contract: true,
                use_faces: true,
                use_face_step: false,
            },
        );
        let shrink_set = ring_set(&shrink);

        // The inner band is the 3×3 block minus the fully-interior centre (2,2).
        let expected: HashSet<GeomKey> = [(1, 1), (1, 2), (1, 3), (2, 1), (2, 3), (3, 1), (3, 2), (3, 3)]
            .iter()
            .map(|&(r, c)| GeomKey::Face(faces[idx(r, c)]))
            .collect();
        assert_eq!(shrink_set, expected, "inner boundary band of the block");

        // The interior face (2,2) = index 12 is excluded — all four of its
        // edge-adjacent neighbours are inside the region (§5.2 boundary test).
        assert!(
            !shrink_set.contains(&GeomKey::Face(faces[idx(2, 2)])),
            "the fully-interior centre face is not on the boundary"
        );

        // Shrink reports a strict subset of the input region.
        let region_set: HashSet<GeomKey> = region.iter().copied().collect();
        assert!(
            shrink_set.is_subset(&region_set) && shrink_set != region_set,
            "shrink ring is a strict subset of the input (§5.2)"
        );
    }

    /// Fully-selected no-op: growing a region that is the whole face set reaches
    /// nothing new — every candidate neighbour is ORIG. REGION_EXTEND.md §8.
    #[test]
    fn face_grow_fully_selected_is_empty() {
        let (mesh, _verts, faces) = vertex_lattice(4);
        let region: Vec<GeomKey> = faces.iter().map(|&f| GeomKey::Face(f)).collect();
        let ring = mesh.region_extend(
            &region,
            RegionExtendOptions {
                use_contract: false,
                use_faces: true,
                use_face_step: false,
            },
        );
        assert!(ring.is_empty(), "fully-selected grow reaches nothing");
    }

    /// Empty input → empty ring, no panic (REGION_EXTEND.md §8).
    #[test]
    fn empty_input_is_empty_ring() {
        let (mesh, _verts, _faces) = vertex_lattice(3);
        let ring = mesh.region_extend(&[], RegionExtendOptions::default());
        assert!(ring.is_empty());
    }

    /// Region-extend mutates nothing: counts are unchanged across the call
    /// (REGION_EXTEND.md §6.1, `ΔV = ΔE = ΔF = 0`).
    #[test]
    fn mesh_is_unchanged_by_the_query() {
        let (mesh, verts, _faces) = vertex_lattice(3);
        let (v, e, f) = (mesh.vertex_count(), mesh.edge_count(), mesh.face_count());
        let _ = mesh.region_extend(&[GeomKey::Vert(verts[4])], RegionExtendOptions::default());
        assert_eq!((mesh.vertex_count(), mesh.edge_count(), mesh.face_count()), (v, e, f));
    }
}
