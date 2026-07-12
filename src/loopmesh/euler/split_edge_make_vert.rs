use crate::loopmesh::*;

impl LoopMesh {
    /// Splits `edge` at a new vertex co-located with `tail_vertex`. Returns
    /// `(new_vertex, new_edge)`. After the split:
    ///
    /// - `edge` connects `v_old` (the original "other" endpoint) and `new_vertex`
    /// - `new_edge` connects `new_vertex` and `tail_vertex`
    /// - every face that referenced `edge` gains an extra loop at `new_vertex`,
    ///   correctly distributed across `edge` and `new_edge`
    pub fn euler_split_edge_make_vert(
        &mut self,
        tail_vertex: VertexKey,
        edge: EdgeKey,
    ) -> (VertexKey, EdgeKey) {
        debug_assert!(
            self.edges.get(edge).contains(tail_vertex).is_some(),
            "tail_vertex must belong to the edge being split"
        );

        let v_old = self
            .edges
            .get(edge)
            .other_vertex(tail_vertex)
            .expect("edge must have two distinct endpoints");
        let tv_pos = self.vertices.get(tail_vertex).position;

        // 1) Create the new vertex. Its disk is empty.
        let v_new = self.create_vertex(tv_pos);

        // 2) Migrate `edge`'s endpoint from tail_vertex to v_new *before* the
        //    fresh half exists, so the retained half reaches the still-empty
        //    disk first and becomes v_new's **disk anchor** (§8.1 steps 2-3).
        //    The anchor is contractual — it seeds v_new's fan (§6.5), so a
        //    caller that ties on "first in the fan" reads this choice back.
        //      a) remove edge from tail_vertex's disk
        //      b) overwrite the endpoint slot
        //      c) append edge to v_new's (empty) disk
        self.vertex_disk_remove_edge(tail_vertex, edge);
        let slot = if self.edges.get(edge).vertices[0] == tail_vertex {
            0
        } else {
            1
        };
        self.edges.get_mut(edge).vertices[slot] = v_new;
        self.vertex_disk_append_edge(v_new, edge);

        // 3) Only now create tail_vertex <-> v_new. create_edge appends it to
        //    both endpoints' disks; in v_new's it joins an already-anchored
        //    cycle, so it does not displace the anchor.
        let e_new = self.create_edge([tail_vertex, v_new], EdgeCreateFlags::NONE);

        // 4) Split the radial cycle, walking it forward **from its radial
        //    entry** — the last-attached loop (§8.1 step 4). The walk order is
        //    the order the two halves are re-populated in, and each half's own
        //    entry is left on whichever face the walk handled last, so starting
        //    at the first-attached loop instead would leave both halves headed
        //    by the wrong face. Snapshot first: the walk rebuilds the very
        //    cycle it is reading.
        let original_loops: Vec<LoopKey> = match self.edge_radial_last_attached(edge) {
            Some(entry) => {
                let mut loops = Vec::new();
                let mut current = entry;
                loop {
                    loops.push(current);
                    current = self.loop_radial_next(current);
                    if current == entry {
                        break;
                    }
                }
                loops
            }
            None => Vec::new(),
        };

        // Detach every loop from `edge`'s radial cycle in one shot — the
        // per-loop reattach below decides which of (edge, e_new) it belongs to.
        self.edges.get_mut(edge).radial_link = None;
        for &l in &original_loops {
            let lref = self.loops.get_mut(l);
            // Transient null-edge state — the per-loop reattach below
            // assigns each one to either `edge` (re-anchored) or `e_new`.
            // The invariant "every reachable loop has a non-null edge"
            // is restored before SEMV returns.
            lref.edge = EdgeKey::NULL;
            lref.edge_radial_links = DoubleLinks { next: l, prev: l };
        }

        for &l in &original_loops {
            let l_face = self.loops.get(l).face;
            let l_next = self.loops.get(l).face_loop_links.next;
            let l_vertex = self.loops.get(l).vertex;

            // Bump face perimeter length — we're inserting one new loop.
            self.faces.get_mut(l_face).len += 1;

            // New loop sits at v_new, spliced between l and l_next:
            //     ... <-> l <-> l_new <-> l_next <-> ...
            let l_new = self.create_loop(v_new, l_face);
            self.loops.get_mut(l_new).face_loop_links.prev = l;
            self.loops.get_mut(l_new).face_loop_links.next = l_next;
            self.loops.get_mut(l).face_loop_links.next = l_new;
            self.loops.get_mut(l_next).face_loop_links.prev = l_new;

            // Each loop in our cycle covers (loop.vertex -> loop.next.vertex).
            // After the splice:
            //   l covers (l_vertex -> v_new)               via either `edge` or `e_new`
            //   l_new covers (v_new -> l_next.vertex)      via the other edge
            // `edge` now connects v_new <-> v_old; `e_new` connects v_new <-> tail_vertex.
            let (l_edge, l_new_edge) = if l_vertex == v_old {
                (edge, e_new)
            } else if l_vertex == tail_vertex {
                (e_new, edge)
            } else {
                panic!(
                    "loop vertex {} is not an endpoint of split edge ({} <-> {})",
                    l_vertex, v_old, tail_vertex
                );
            };
            self.face_loop_connect_edge(l_edge, l);
            self.face_loop_connect_edge(l_new_edge, l_new);
        }

        (v_new, e_new)
    }
}

#[cfg(test)]
mod tests {
    use glam::Vec3;

    use crate::loopmesh::test_support::create_face_from_verts;
    use crate::loopmesh::*;

    /// Sanity test for euler_split_edge_make_vert against a quad face.
    #[test]
    fn test_euler_split_edge_make_vert_quad() {
        let mut mesh = LoopMesh::new();

        let verts = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::new(2.0, 1.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        ];
        let face = create_face_from_verts(&mut mesh, &verts);
        let original_face_verts = mesh.face_vert_keys(face);
        assert_eq!(original_face_verts.len(), 4);

        // Pick the edge that runs between vert[0] and vert[1] in the face cycle.
        let v0 = original_face_verts[0];
        let v1 = original_face_verts[1];
        let edge = mesh
            .query_edge_exists(v0, v1)
            .expect("quad must have v0-v1 edge");

        // Split at v1 (i.e. tail_vertex = v1, so new vertex sits between v0 and v1
        // along the original edge, leaving `edge` connecting v0 <-> v_new and
        // a new edge connecting v_new <-> v1).
        let pre_totvert = mesh.vertex_count();
        let pre_totedge = mesh.edge_count();
        let pre_totface = mesh.face_count();

        let (v_new, e_new) = mesh.euler_split_edge_make_vert(v1, edge);

        // Topology checks.
        assert_eq!(mesh.vertex_count(), pre_totvert + 1, "one vertex added");
        assert_eq!(mesh.edge_count(), pre_totedge + 1, "one edge added");
        assert_eq!(mesh.face_count(), pre_totface, "no face change");

        // The face should now have 5 verts.
        let new_face_verts = mesh.face_vert_keys(face);
        assert_eq!(new_face_verts.len(), 5, "quad becomes pentagon");
        assert_eq!(mesh.face_len(face), 5);

        // v_new must appear exactly once in the new face cycle, sandwiched
        // between v0 and v1 in the original winding direction.
        let new_pos = new_face_verts
            .iter()
            .position(|&v| v == v_new)
            .expect("new vertex must appear in the face cycle");
        let prev_pos = (new_pos + new_face_verts.len() - 1) % new_face_verts.len();
        let next_pos = (new_pos + 1) % new_face_verts.len();
        assert_eq!(new_face_verts[prev_pos], v0);
        assert_eq!(new_face_verts[next_pos], v1);

        // Disk validation: v_new sits on exactly 2 edges (edge + e_new).
        assert_eq!(mesh.vertex_disk_count_edges(v_new), 2);
        let v_new_edges: Vec<EdgeKey> = mesh.vertex_disk_iter(v_new).collect();
        assert!(v_new_edges.contains(&edge));
        assert!(v_new_edges.contains(&e_new));

        // Edge endpoints: `edge` should now span v0 <-> v_new, and `e_new` v_new <-> v1.
        let edge_verts = mesh.edge_vertices(edge);
        assert!(
            (edge_verts == [v0, v_new]) || (edge_verts == [v_new, v0]),
            "edge endpoints after SEMV should be v0<->v_new, got {:?}",
            edge_verts
        );
        let e_new_verts = mesh.edge_vertices(e_new);
        assert!(
            (e_new_verts == [v_new, v1]) || (e_new_verts == [v1, v_new]),
            "e_new endpoints should be v_new<->v1, got {:?}",
            e_new_verts
        );

        // Each edge in the new pentagon has the face on its radial.
        for f_edge in mesh.face_edge_keys(face) {
            let radial: Vec<FaceKey> = mesh.edge_face_keys(f_edge);
            assert_eq!(radial.len(), 1);
            assert_eq!(radial[0], face);
        }
    }

    /// Two quads sharing one edge, built in a known order so the shared edge's
    /// radial attachment order is `[first_quad, second_quad]`.
    fn two_quads_sharing_an_edge(mesh: &mut LoopMesh) -> (FaceKey, FaceKey, EdgeKey, VertexKey) {
        let v: Vec<VertexKey> = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::new(2.0, 1.0, 0.0),
        ]
        .into_iter()
        .map(|p| mesh.create_vertex(p))
        .collect();

        let first = mesh.face_create_verts(&[v[0], v[1], v[2], v[3]], FaceCreateFlags::NONE);
        let second = mesh.face_create_verts(&[v[1], v[4], v[5], v[2]], FaceCreateFlags::NONE);
        let shared = mesh.query_edge_exists(v[1], v[2]).expect("shared edge");
        (first, second, shared, v[2])
    }

    fn fan_first_face(mesh: &LoopMesh, v: VertexKey) -> FaceKey {
        let mut fan = mesh.vert_fan_loops(v);
        let first = fan.next(mesh).expect("non-empty fan");
        mesh.loop_face(first)
    }

    /// RADIAL_EDGE_STRUCTURE.md §8.1's cycle-head contract, manifold arm.
    ///
    /// The retained half must reach `v_new`'s empty disk **before** the fresh
    /// half exists (steps 2-3), so `v_new`'s disk anchor is `edge`; and the
    /// radial rebuild must walk the pre-split cycle forward **from its radial
    /// entry** (step 4), so each half ends up headed by the face the walk
    /// handled last — for a radial-2 edge, the face attached to it **first**.
    /// Composing the two: `v_new`'s fan opens on that first-attached face.
    ///
    /// Each assertion is falsified on its own by exactly one half of the
    /// radial-anchor pair, so the two cannot silently cancel back into place.
    #[test]
    fn semv_seats_v_new_anchor_on_retained_half_and_heads_both_halves_with_first_attached_face() {
        let mut mesh = LoopMesh::new();
        let (first_quad, second_quad, edge, tail) = two_quads_sharing_an_edge(&mut mesh);

        // Pre-split sanity: the shared edge's entry is the quad that adopted it last.
        let entry = mesh.edge_radial_last_attached(edge).expect("radial-2 edge");
        assert_eq!(mesh.loop_face(entry), second_quad);

        let (v_new, e_new) = mesh.euler_split_edge_make_vert(tail, edge);
        mesh.validate_topology();

        assert_eq!(
            mesh.vertices.get(v_new).disk_loop_start,
            Some(edge),
            "v_new's disk anchor is the retained half, not the fresh one"
        );
        for half in [edge, e_new] {
            let head = mesh.edge_radial_last_attached(half).expect("radial-2 half");
            assert_eq!(
                mesh.loop_face(head),
                first_quad,
                "a split half is headed by the face attached first to the original edge"
            );
        }
        assert_eq!(fan_first_face(&mesh, v_new), first_quad);
    }

    /// §8.1's cycle-head contract, non-manifold arm — the case the radial-2
    /// coincidence cannot cover. Three pages share the spine, attached in build
    /// order. The rebuild walks forward from the entry (the **last**-attached
    /// page), i.e. `p2, p0, p1`, so it finishes on `p1` and both halves are
    /// headed by the page attached **second-to-last**.
    #[test]
    fn semv_nonmanifold_split_heads_both_halves_with_second_to_last_attached_face() {
        let mut mesh = LoopMesh::new();
        let spine_a = mesh.create_vertex(Vec3::new(0.0, 0.0, 0.0));
        let spine_b = mesh.create_vertex(Vec3::new(0.0, 1.0, 0.0));
        let pages: Vec<FaceKey> = (0..3)
            .map(|i| {
                let tip = mesh.create_vertex(Vec3::new(1.0 + i as f32, 0.5, 0.0));
                mesh.face_create_verts(&[spine_a, spine_b, tip], FaceCreateFlags::NONE)
            })
            .collect();
        let spine = mesh.query_edge_exists(spine_a, spine_b).expect("spine");

        let (v_new, e_new) = mesh.euler_split_edge_make_vert(spine_b, spine);
        mesh.validate_topology();

        assert_eq!(mesh.vertices.get(v_new).disk_loop_start, Some(spine));
        for half in [spine, e_new] {
            let head = mesh.edge_radial_last_attached(half).expect("radial-3 half");
            assert_eq!(
                mesh.loop_face(head),
                pages[1],
                "the rebuild ends on the page attached second-to-last"
            );
        }
        assert_eq!(fan_first_face(&mesh, v_new), pages[1]);
    }

    /// Splitting the same edge twice should give the same topology shape (hexagon).
    #[test]
    fn test_euler_split_edge_make_vert_twice() {
        let mut mesh = LoopMesh::new();
        let verts = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::new(2.0, 1.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        ];
        let face = create_face_from_verts(&mut mesh, &verts);
        let face_verts0 = mesh.face_vert_keys(face);
        let v0 = face_verts0[0];
        let v1 = face_verts0[1];
        let edge = mesh.query_edge_exists(v0, v1).unwrap();

        let (v_mid1, _e1) = mesh.euler_split_edge_make_vert(v1, edge);
        // After first split, `edge` now goes v0 <-> v_mid1. Split that again.
        let (_v_mid2, _e2) = mesh.euler_split_edge_make_vert(v_mid1, edge);

        assert_eq!(mesh.vertex_count(), 6);
        assert_eq!(mesh.edge_count(), 6);
        assert_eq!(mesh.face_len(face), 6);
    }
}
