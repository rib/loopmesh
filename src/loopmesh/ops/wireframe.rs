use crate::loopmesh::*;

/// Near-flat guard for the inset-tangent bisector: a corner whose two edge
/// directions sum to (near-)zero is collinear, so the bisector is undefined and
/// the tangent falls back to a face-plane perpendicular (WIREFRAME.md §4.3).
const BISECTOR_EPS_SQ: f32 = 1.0e-12;

/// The generated geometry of a wireframe run: the strut quads lining the
/// region's edges, plus the rim-cap quads added under `use_boundary`. This is
/// the analogue of the operator's `faces.out` slot (WIREFRAME.md §1) — it holds
/// only freshly built strut faces, never the (retained) source faces.
#[derive(Default)]
pub struct WireframeResult {
    pub strut_faces: Vec<FaceKey>,
}

/// The two shell hub verts of one tagged vertex (WIREFRAME.md §4.1): `neg` at
/// the near distance, `pos` at the far distance, both along the vertex normal.
#[derive(Clone, Copy)]
struct ShellPair {
    neg: VertexKey,
    pos: VertexKey,
}

impl LoopMesh {
    /// Replace a region of faces with a lattice of solid struts running along
    /// the region's edges — a per-face inset frame plus a two-sided strut
    /// bridge (WIREFRAME.md §2).
    ///
    /// For every tagged face this insets a shrunken copy of the face's loop
    /// inward along the face-tangent direction (the same in-plane bisector
    /// [`Self::inset_region`] uses, §4.2–§4.3), and bridges the gap between each
    /// original edge and the inset edge with a pair of quads — one offset to
    /// each side of the surface along the vertex normals by half the thickness
    /// (the two-sided shell offset of [`Self::solidify`], §4.1 + §5). Where two
    /// tagged faces share an edge their two strut pairs wrap that edge into a
    /// closed interior tube (§11.3); a region-boundary edge gets only the inner
    /// pair unless `use_boundary` caps its outer side with a rim strut pair
    /// (§6).
    ///
    /// This kernel-operator form **retains** the original input faces (it does
    /// not delete them) and leaves the original vertices in place — the
    /// `use_replace` removal of §7 is a separate, not-yet-implemented axis. The
    /// returned [`WireframeResult`] carries only the generated strut/cap quads.
    ///
    /// See `radial-mesh-guides/operators/WIREFRAME.md` for the full algorithm
    /// and parity contract.
    pub fn wireframe(
        &mut self,
        faces: &[FaceKey],
        thickness: f32,
        use_boundary: bool,
    ) -> WireframeResult {
        if faces.is_empty() {
            return WireframeResult::default();
        }

        // The shell offset (§5) and the in-plane tangent (§4.3) both read the
        // stored normals, so freshen them first — exactly the precondition the
        // oracle carries.
        self.mesh_normals_update();

        // §5.1 with offset = 0: the cross-section straddles the surface
        // symmetrically — one shell vert half a thickness to each side, the
        // inset frame in the surface plane.
        let near_offset = -thickness * 0.5;
        let far_offset = thickness * 0.5;
        let inset_dist = thickness * 0.5;

        let face_tag = self.push_face_op_flag_layer();
        for &face in faces {
            self.face_op_flag_set(face_tag, face, 0);
        }

        // §3: tag every vertex of a tagged face (a tagged vertex gets a shell
        // pair). The shell-pair lookup is keyed by original vertex slot.
        let vert_watermark = self.vertices.key_watermark();
        let mut vert_tagged: Vec<bool> = vec![false; vert_watermark];
        for &face in faces {
            let mut loops = self.face_loops(face);
            while let Some(l) = loops.next(self) {
                vert_tagged[self.loop_vertex(l).as_usize()] = true;
            }
        }

        // §4.1: two shell verts per tagged vertex, placed along the vertex
        // normal at the near / far distances. Indexed by original vertex slot.
        let mut shell: Vec<Option<ShellPair>> = vec![None; vert_watermark];
        for slot in 0..vert_watermark {
            if !vert_tagged[slot] {
                continue;
            }
            let v = VertexKey::from_usize(slot);
            let base = self.vertex_position(v);
            let normal = self.vertex_normal(v);

            let neg = self.create_vertex(base + normal * near_offset);
            self.vertex_attribs.copy_into(neg.as_usize(), slot);
            let pos = self.create_vertex(base + normal * far_offset);
            self.vertex_attribs.copy_into(pos.as_usize(), slot);

            shell[slot] = Some(ShellPair { neg, pos });
        }

        // §4.2: one inset-frame vert per loop (face-corner), in the face plane,
        // pushed inward along the corner's face tangent by `thickness/2`. Keyed
        // by loop key so §4.4 can recover both ends of each edge.
        let loop_watermark = self.loops.key_watermark();
        let mut inset_of_loop: Vec<Option<VertexKey>> = vec![None; loop_watermark];
        for &face in faces {
            let face_normal = self.face_normal(face);
            let mut loops = self.face_loops(face);
            while let Some(l) = loops.next(self) {
                let v = self.loop_vertex(l);
                let tangent = self.wireframe_inward_tangent(l, face_normal);
                let corner = self.vertex_position(v);
                let frame = self.create_vertex(corner + tangent * inset_dist);
                self.vertex_attribs.copy_into(frame.as_usize(), v.as_usize());
                inset_of_loop[l.as_usize()] = Some(frame);
            }
        }

        // §3 boundary classification + §6.1 boundary-frame verts (one per
        // boundary vertex, created lazily so each appears once).
        let mut boundary_frame: Vec<Option<VertexKey>> = vec![None; vert_watermark];

        let mut result = WireframeResult::default();

        // §4.4 + §6.2: walk every tagged face's loops once more and bridge each
        // corner's edge. The current loop runs from corner `a` (its own vertex)
        // to corner `b` (the next loop's vertex) along one face edge.
        for &face in faces {
            let face_normal = self.face_normal(face);
            let first = self.face_first_loop(face);
            let mut l = first;
            loop {
                let next_l = self.loop_next(l);
                let edge = self.loop_edge(l);

                let va = self.loop_vertex(l);
                let vb = self.loop_vertex(next_l);
                let inset_a = inset_of_loop[l.as_usize()].unwrap();
                let inset_b = inset_of_loop[next_l.as_usize()].unwrap();
                let shell_a = shell[va.as_usize()].unwrap();
                let shell_b = shell[vb.as_usize()].unwrap();

                // §4.4: the `−` quad is (inset_a, inset_b, neg_b, neg_a); the
                // `+` quad is (inset_b, inset_a, pos_a, pos_b) — opposite
                // windings so both faces point outward from the wire.
                let neg_quad = self.wireframe_quad(
                    [inset_a, inset_b, shell_b.neg, shell_a.neg],
                    [l, next_l, next_l, l],
                    face,
                );
                let pos_quad = self.wireframe_quad(
                    [inset_b, inset_a, shell_a.pos, shell_b.pos],
                    [next_l, l, l, next_l],
                    face,
                );
                result.strut_faces.push(neg_quad);
                result.strut_faces.push(pos_quad);

                // §6: a region-boundary edge (no other tagged face uses it)
                // gets, under `use_boundary`, an outer rim cap that closes its
                // tube.
                if use_boundary && self.wireframe_edge_is_boundary(edge, face, face_tag) {
                    // §6.1: one boundary vert per boundary vertex, created the
                    // first time a boundary edge touches it.
                    let boundary_a = match boundary_frame[va.as_usize()] {
                        Some(existing) => existing,
                        None => {
                            let frame = self.wireframe_boundary_frame(
                                va, face, face_normal, face_tag, inset_dist,
                            );
                            boundary_frame[va.as_usize()] = Some(frame);
                            frame
                        }
                    };
                    let boundary_b = match boundary_frame[vb.as_usize()] {
                        Some(existing) => existing,
                        None => {
                            let frame = self.wireframe_boundary_frame(
                                vb, face, face_normal, face_tag, inset_dist,
                            );
                            boundary_frame[vb.as_usize()] = Some(frame);
                            frame
                        }
                    };

                    // §6.2: windings opposite the inner struts so the cap faces
                    // point outward on the rim's outer side.
                    let cap_neg = self.wireframe_quad(
                        [boundary_b, boundary_a, shell_a.neg, shell_b.neg],
                        [next_l, l, l, next_l],
                        face,
                    );
                    let cap_pos = self.wireframe_quad(
                        [boundary_a, boundary_b, shell_b.pos, shell_a.pos],
                        [l, next_l, next_l, l],
                        face,
                    );
                    result.strut_faces.push(cap_neg);
                    result.strut_faces.push(cap_pos);
                }

                l = next_l;
                if l == first {
                    break;
                }
            }
        }

        self.pop_face_op_flag_layer(face_tag);
        result
    }

    /// §4.3 — the unit inward face tangent at loop `l`: the bisector of the two
    /// edge directions leaving the corner, pointing into the face interior. A
    /// collinear corner (degenerate bisector) falls back to a face-plane
    /// perpendicular taken against the face normal.
    fn wireframe_inward_tangent(&self, l: LoopKey, face_normal: Vec3) -> Vec3 {
        let v = self.loop_vertex(l);
        let prev_v = self.loop_vertex(self.loop_prev(l));
        let next_v = self.loop_vertex(self.loop_next(l));
        let here = self.vertex_position(v);
        let to_prev = LoopMesh::normalize_or_zero(self.vertex_position(prev_v) - here);
        let to_next = LoopMesh::normalize_or_zero(self.vertex_position(next_v) - here);

        let bisector = to_prev + to_next;
        if bisector.dot(bisector) > BISECTOR_EPS_SQ {
            LoopMesh::normalize_or_zero(bisector)
        } else {
            // Collinear corner: take a perpendicular to the (straight) edge in
            // the face plane, oriented inward by a concavity check.
            let edge_dir = to_next;
            let perp = LoopMesh::normalize_or_zero(face_normal.cross(edge_dir));
            if perp.dot(to_prev) < 0.0 {
                -perp
            } else {
                perp
            }
        }
    }

    /// §3 — an edge is a region boundary with respect to the tagged region if
    /// no tagged face *other than* `owner` uses it (a real mesh boundary, or
    /// the only tagged user is `owner`).
    fn wireframe_edge_is_boundary(
        &self,
        edge: EdgeKey,
        owner: FaceKey,
        face_tag: crate::op_flags::FaceOpFlagLayer,
    ) -> bool {
        let mut radial = self.radial_loops(edge);
        while let Some(rl) = radial.next(self) {
            let face = self.loop_face(rl);
            if face != owner && self.face_op_flag_test(face_tag, face, 0) {
                return false;
            }
        }
        true
    }

    /// §6.1 — the outer-side boundary inset-frame vert at corner `v` of the
    /// boundary-edged face `face`: placed like the per-loop inset vert (§4.2)
    /// but pushed *outward* along the boundary tangent so the rim cap closes the
    /// tube. The boundary tangent is the negation of the inward bisector taken
    /// over the (up to two) boundary edges of `face` at `v`; a single boundary
    /// edge falls back to a face-plane perpendicular.
    fn wireframe_boundary_frame(
        &mut self,
        v: VertexKey,
        face: FaceKey,
        face_normal: Vec3,
        face_tag: crate::op_flags::FaceOpFlagLayer,
        inset_dist: f32,
    ) -> VertexKey {
        let here = self.vertex_position(v);

        // The (up to two) boundary edges of `face` incident to `v`, as unit
        // directions leaving the corner.
        let mut dirs: [Vec3; 2] = [Vec3::ZERO; 2];
        let mut count = 0usize;
        let first = self.face_first_loop(face);
        let mut l = first;
        loop {
            let lv = self.loop_vertex(l);
            let le = self.loop_edge(l);
            if (lv == v || self.loop_vertex(self.loop_next(l)) == v)
                && self.wireframe_edge_is_boundary(le, face, face_tag)
            {
                let [a, b] = self.edge_vertices(le);
                let other = if a == v { b } else { a };
                if other != v {
                    let dir = LoopMesh::normalize_or_zero(self.vertex_position(other) - here);
                    if dir != Vec3::ZERO && count < 2 {
                        dirs[count] = dir;
                        count += 1;
                    }
                }
            }
            l = self.loop_next(l);
            if l == first {
                break;
            }
        }

        let outward = match count {
            2 => {
                // Inward bisector of the two boundary edges → negate for the
                // outward (rim) direction.
                let inward = LoopMesh::normalize_or_zero(dirs[0] + dirs[1]);
                if inward != Vec3::ZERO {
                    -inward
                } else {
                    LoopMesh::normalize_or_zero(face_normal.cross(dirs[0]))
                }
            }
            1 => {
                // §6.1 single-boundary-edge degenerate case: derive the tangent
                // from the one edge alone (a face-plane perpendicular).
                LoopMesh::normalize_or_zero(face_normal.cross(dirs[0]))
            }
            _ => Vec3::ZERO,
        };

        let frame = self.create_vertex(here + outward * inset_dist);
        self.vertex_attribs.copy_into(frame.as_usize(), v.as_usize());
        frame
    }

    /// Build one strut quad from four corner verts, copying the source face's
    /// per-face customdata and each corner's per-loop customdata from the named
    /// source loop (§4.4 / §10). `src_loops[i]` is the source corner whose loop
    /// data corner `i` of the quad inherits.
    fn wireframe_quad(
        &mut self,
        verts: [VertexKey; 4],
        src_loops: [LoopKey; 4],
        src_face: FaceKey,
    ) -> FaceKey {
        let edges = [
            self.create_edge([verts[0], verts[1]], EdgeCreateFlags::NO_DUPLICATE),
            self.create_edge([verts[1], verts[2]], EdgeCreateFlags::NO_DUPLICATE),
            self.create_edge([verts[2], verts[3]], EdgeCreateFlags::NO_DUPLICATE),
            self.create_edge([verts[3], verts[0]], EdgeCreateFlags::NO_DUPLICATE),
        ];
        let face = self.face_create(&verts, &edges, FaceCreateFlags::NONE);
        self.face_attribs.copy_into(face.as_usize(), src_face.as_usize());

        if !self.loop_attribs.is_empty() {
            let mut ql = self.face_first_loop(face);
            for i in 0..4 {
                self.loop_attribs
                    .copy_into(ql.as_usize(), src_loops[i].as_usize());
                ql = self.loop_next(ql);
            }
        }

        face
    }
}

#[cfg(test)]
mod tests {
    use glam::Vec3;

    use crate::loopmesh::test_support::create_face_from_verts;
    use crate::loopmesh::*;

    /// A unit quad lying flat in the z = 0 plane, wound so its face normal
    /// points +Z.
    fn flat_quad(mesh: &mut LoopMesh) -> FaceKey {
        create_face_from_verts(
            mesh,
            &[
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(1.0, 1.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
            ],
        )
    }

    #[test]
    fn lone_quad_open_frame_counts() {
        // WIREFRAME.md §11.1 (kernel-layer counts): a lone tagged quad with
        // `use_boundary = false` builds an open four-strut frame — 8 strut
        // faces — and the kernel BMOP *retains* the source face, so the mesh
        // ends with 8 + 1 = 9 faces and 16 verts (4 original + 4 inset-frame +
        // 8 shell).
        let mut mesh = LoopMesh::new();
        let face = flat_quad(&mut mesh);

        let result = mesh.wireframe(&[face], 0.2, false);

        assert_eq!(result.strut_faces.len(), 8, "two struts per corner");
        assert_eq!(mesh.face_count(), 9, "8 struts + 1 retained source");
        assert_eq!(
            mesh.vertex_count(),
            16,
            "4 original + 4 inset-frame + 8 shell"
        );
        mesh.validate_topology();
    }

    #[test]
    fn lone_quad_capped_rim_counts() {
        // WIREFRAME.md §6 / §11.1: with `use_boundary = true` every rim edge is
        // capped by a second strut pair — 8 inner + 8 cap = 16 strut faces —
        // adding one boundary vert per corner. With the retained source: 17
        // faces, 20 verts (16 open + 4 boundary).
        let mut mesh = LoopMesh::new();
        let face = flat_quad(&mut mesh);

        let result = mesh.wireframe(&[face], 0.2, true);

        assert_eq!(result.strut_faces.len(), 16, "8 inner + 8 rim caps");
        assert_eq!(mesh.face_count(), 17, "16 struts + 1 retained source");
        assert_eq!(
            mesh.vertex_count(),
            20,
            "16 (open case) + 4 boundary verts"
        );
        mesh.validate_topology();
    }

    #[test]
    fn use_boundary_contrast_pins_flag_effect() {
        // The contrast pin (§6): the same lone quad differs by exactly the rim
        // cap — 8 extra strut faces and 4 extra boundary verts — between
        // `use_boundary` clear and set. An impl that ignored the flag would
        // produce identical counts.
        let mut open = LoopMesh::new();
        let fo = flat_quad(&mut open);
        let open_result = open.wireframe(&[fo], 0.2, false);

        let mut capped = LoopMesh::new();
        let fc = flat_quad(&mut capped);
        let capped_result = capped.wireframe(&[fc], 0.2, true);

        assert_eq!(
            capped_result.strut_faces.len() - open_result.strut_faces.len(),
            8,
            "use_boundary adds one cap pair per boundary edge"
        );
        assert_eq!(
            capped.vertex_count() - open.vertex_count(),
            4,
            "use_boundary adds one boundary vert per boundary vertex"
        );
    }

    #[test]
    fn grid_3x3_centre_face_wires_only_region() {
        // WIREFRAME.md §11.2: wiring the centre face of a 3×3 grid touches only
        // that face. Its four edges are all interior to the mesh but
        // wire-boundary with respect to the tagged region (the surrounding
        // faces are untagged), so with `use_boundary = false` it yields 8 strut
        // faces; the 8 surrounding faces and their verts are untouched.
        let mut mesh = LoopMesh::new();
        let mut verts = [[VertexKey::from_usize(0); 4]; 4];
        for (iy, row) in verts.iter_mut().enumerate() {
            for (ix, slot) in row.iter_mut().enumerate() {
                *slot = mesh.create_vertex(Vec3::new(ix as f32, iy as f32, 0.0));
            }
        }
        let mut faces = Vec::new();
        for y in 0..3 {
            for x in 0..3 {
                let ring = [
                    verts[y][x],
                    verts[y][x + 1],
                    verts[y + 1][x + 1],
                    verts[y + 1][x],
                ];
                let edges = [
                    mesh.create_edge([ring[0], ring[1]], EdgeCreateFlags::NO_DUPLICATE),
                    mesh.create_edge([ring[1], ring[2]], EdgeCreateFlags::NO_DUPLICATE),
                    mesh.create_edge([ring[2], ring[3]], EdgeCreateFlags::NO_DUPLICATE),
                    mesh.create_edge([ring[3], ring[0]], EdgeCreateFlags::NO_DUPLICATE),
                ];
                faces.push(mesh.face_create(&ring, &edges, FaceCreateFlags::NONE));
            }
        }
        let centre = faces[4];
        let surround: Vec<FaceKey> = faces.iter().copied().filter(|&f| f != centre).collect();
        let verts_before = mesh.vertex_count();

        let result = mesh.wireframe(&[centre], 0.2, false);

        assert_eq!(result.strut_faces.len(), 8, "all four edges are wire-boundary");
        // 9 grid faces + 8 struts (centre retained) = 17.
        assert_eq!(mesh.face_count(), 17);
        // New verts: 4 inset + 8 shell; the 16 grid verts persist.
        assert_eq!(mesh.vertex_count(), verts_before + 12);
        for &f in &surround {
            assert_eq!(mesh.face_len(f), 4, "surrounding faces untouched");
        }
        mesh.validate_topology();
    }

    #[test]
    fn two_quads_shared_edge_interior_tube_closes() {
        // WIREFRAME.md §11.3: two tagged faces sharing an edge make it a wire
        // *interior* edge — each contributes a strut pair (no rim cap on the
        // shared edge). Per-corner counts: L = 8 corners → 16 strut faces. Each
        // quad has 4 boundary edges of its own plus the shared one (7 distinct
        // tagged edges total; the shared one is interior). With `use_boundary`
        // off the shared edge gets no cap, so the four strut quads around it
        // close the tube via the shared shell hub edges.
        let mut mesh = LoopMesh::new();
        let v0 = mesh.create_vertex(Vec3::new(0.0, 0.0, 0.0));
        let v1 = mesh.create_vertex(Vec3::new(1.0, 0.0, 0.0));
        let v2 = mesh.create_vertex(Vec3::new(1.0, 1.0, 0.0));
        let v3 = mesh.create_vertex(Vec3::new(0.0, 1.0, 0.0));
        let v4 = mesh.create_vertex(Vec3::new(2.0, 0.0, 0.0));
        let v5 = mesh.create_vertex(Vec3::new(2.0, 1.0, 0.0));
        let e01 = mesh.create_edge([v0, v1], EdgeCreateFlags::NO_DUPLICATE);
        let e12 = mesh.create_edge([v1, v2], EdgeCreateFlags::NO_DUPLICATE);
        let e23 = mesh.create_edge([v2, v3], EdgeCreateFlags::NO_DUPLICATE);
        let e30 = mesh.create_edge([v3, v0], EdgeCreateFlags::NO_DUPLICATE);
        let e14 = mesh.create_edge([v1, v4], EdgeCreateFlags::NO_DUPLICATE);
        let e45 = mesh.create_edge([v4, v5], EdgeCreateFlags::NO_DUPLICATE);
        let e52 = mesh.create_edge([v5, v2], EdgeCreateFlags::NO_DUPLICATE);
        let a = mesh.face_create(&[v0, v1, v2, v3], &[e01, e12, e23, e30], FaceCreateFlags::NONE);
        let b = mesh.face_create(&[v1, v4, v5, v2], &[e14, e45, e52, e12], FaceCreateFlags::NONE);

        let result = mesh.wireframe(&[a, b], 0.2, false);

        assert_eq!(
            result.strut_faces.len(),
            16,
            "two strut pairs per face × 4 corners each = 16"
        );
        // 16 struts + 2 retained source faces.
        assert_eq!(mesh.face_count(), 18);
        // New verts: 8 inset-frame (one per loop) + 12 shell (2 × 6 tagged
        // verts). 6 originals persist.
        assert_eq!(mesh.vertex_count(), 6 + 8 + 12);

        // Tube closure: the shared edge's shell hub verts at v1 and v2 must each
        // be referenced by struts from *both* faces. The neg shell vert of v1 is
        // shared by A's and B's neg struts, so its radial edge fan is richer
        // than an open-rim shell vert's. Topology validity is the load-bearing
        // assertion — the four quads must weld into a manifold tube.
        mesh.validate_topology();
    }
}
