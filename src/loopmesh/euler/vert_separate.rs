use crate::loopmesh::*;

impl LoopMesh {
    /// Separates the disjoint face-fans at a single vertex into multiple
    /// vertices. Returns the list of resulting vertices (the input vertex is
    /// always returned first; one extra vertex is created per additional
    /// fan).
    ///
    /// For a normal manifold vertex with a single connected disk this is a
    /// no-op and the returned Vec contains just the input.
    ///
    /// The surgery follows
    /// [`RADIAL_EDGE_STRUCTURE.md`](../../../radial-mesh-guides/RADIAL_EDGE_STRUCTURE.md)
    /// §8.7, whose ordering choices are all observable through the resulting disks:
    ///
    /// - each round is seeded at `v`'s **current** disk anchor, so the anchor's own
    ///   fan is the first to move off and `v` keeps the fan discovered **last**;
    /// - a fan is collected with a **LIFO frontier**, which decides the discovery
    ///   order whenever a reached edge has two unvisited corner-adjacent partners
    ///   (i.e. from fan size 3 up — below that LIFO and FIFO agree);
    /// - the fan's edges re-attach to the new vertex in the **reverse** of that
    ///   discovery order, so the new vertex anchors on the **last**-discovered edge
    ///   and its forward disk walk runs in reverse-discovery order.
    pub fn euler_vert_separate(&mut self, v: VertexKey) -> Vec<VertexKey> {
        let mut result = vec![v];
        let mut remaining: Vec<EdgeKey> = self.vertex_disk_iter(v).collect();
        if remaining.len() <= 1 {
            return result;
        }

        // Gather one fan (the connected component of the corner-adjacency
        // relation, two edges being adjacent when one face meets `v` along both)
        // starting from `seed`, returning its edges in **discovery order** — the
        // order the LIFO frontier surfaces them, not the order they were pushed.
        let gather_fan = |this: &Self, seed: EdgeKey| -> Vec<EdgeKey> {
            let mut reached = vec![seed];
            let mut frontier = vec![seed];
            let mut discovered = Vec::with_capacity(4);
            while let Some(e) = frontier.pop() {
                discovered.push(e);
                for l in this.edge_radial_iter(e) {
                    let l_at_v = if this.loops.get(l).vertex == v {
                        l
                    } else {
                        let next = this.loops.get(l).face_loop_links.next;
                        if this.loops.get(next).vertex == v {
                            next
                        } else {
                            continue;
                        }
                    };
                    // The face meets `v` along `l_at_v.edge` and along the edge of
                    // the loop before it; one of the two is `e`, the other is the
                    // partner this face contributes to the frontier.
                    let prev_l = this.loops.get(l_at_v).face_loop_links.prev;
                    for partner in [this.loops.get(prev_l).edge, this.loops.get(l_at_v).edge] {
                        if !partner.is_null() && !reached.contains(&partner) {
                            reached.push(partner);
                            frontier.push(partner);
                        }
                    }
                }
            }
            discovered
        };

        // `remaining` is the disk walk from `v`'s anchor, so after striking out
        // each collected fan its head is the earliest-attached survivor — which is
        // where the anchor re-seats to (§6.5's disk invariant). Collecting the fans
        // up front is therefore the same seeding as §8.7's interleaved drain, and
        // a fan's collection cannot see the edges of a fan that already left.
        let mut fans: Vec<Vec<EdgeKey>> = Vec::new();
        while !remaining.is_empty() {
            let fan = gather_fan(self, remaining[0]);
            remaining.retain(|e| !fan.contains(e));
            fans.push(fan);
        }

        let v_pos = self.vertices.get(v).position;
        let last_idx = fans.len() - 1;
        for (i, fan) in fans.iter().enumerate() {
            if i == last_idx {
                break;
            }
            let v_new = self.create_vertex(v_pos);
            for &e in fan.iter().rev() {
                self.edge_replace_vertex_raw(e, v, v_new);
            }
            // Vert split: each peeled vertex inherits a 1:1 copy of the
            // source vertex's customdata.
            self.vertex_attribs
                .copy_into(v_new.as_usize(), v.as_usize());
            result.push(v_new);
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use crate::loopmesh::*;
    use glam::Vec3;

    fn build(mesh: &mut LoopMesh, n_verts: usize, faces: &[[usize; 3]]) -> Vec<VertexKey> {
        let verts: Vec<VertexKey> = (0..n_verts)
            .map(|i| mesh.create_vertex(Vec3::new(i as f32, 0.0, 0.0)))
            .collect();
        for f in faces {
            mesh.face_create_verts(
                &[verts[f[0]], verts[f[1]], verts[f[2]]],
                FaceCreateFlags::NONE,
            );
        }
        verts
    }

    /// `verts[hub]`'s disk in §6.5 walk order (forward from its anchor), read as
    /// the `verts` index of each edge's far endpoint.
    fn disk_far_endpoints(mesh: &LoopMesh, verts: &[VertexKey], hub: usize) -> Vec<usize> {
        mesh.iter_disk_edges(verts[hub])
            .map(|e| {
                let [a, b] = mesh.edge_vertices(e);
                let far = if a == verts[hub] { b } else { a };
                verts.iter().position(|&v| v == far).expect("far endpoint")
            })
            .collect()
    }

    /// §8.7 — which fan stays, and where the new vertex's anchor lands.
    ///
    /// A bowtie's pinch vert gains `{0,2} {1,2}` (face 0, closing edge first) then
    /// `{2,4} {2,3}` (face 1), so its anchor is an edge of **face 0's** fan. The
    /// drain is seeded at the anchor, so that fan is the one that **moves off**,
    /// and the pinch keeps the last-discovered fan.
    ///
    /// Falsifies "`v` keeps the fan containing its original anchor" (the pinch
    /// would read `0 1`) and "the new vertex anchors on the first edge its fan walk
    /// discovered" (the new vertex would read `0 1`).
    #[test]
    fn vert_separate_moves_the_anchor_fan_off_and_reverse_seats_the_new_anchor() {
        let mut mesh = LoopMesh::new();
        let mut verts = build(&mut mesh, 5, &[[0, 1, 2], [2, 3, 4]]);
        assert_eq!(disk_far_endpoints(&mesh, &verts, 2), vec![0, 1, 4, 3]);

        let fans = mesh.euler_vert_separate(verts[2]);
        mesh.validate_topology();
        assert_eq!(fans.len(), 2);
        assert_eq!(fans[0], verts[2], "the input vertex comes back first");
        verts.push(fans[1]);

        assert_eq!(
            disk_far_endpoints(&mesh, &verts, 2),
            vec![4, 3],
            "the pinch keeps the LAST-discovered fan"
        );
        assert_eq!(
            disk_far_endpoints(&mesh, &verts, 5),
            vec![1, 0],
            "the new vertex anchors on its fan's LAST-discovered edge"
        );
    }

    /// §8.7's **LIFO frontier**, which the bowtie above cannot see: its fans are
    /// two edges each, so the collection walk never holds two frontier entries and
    /// LIFO and FIFO agree. Two open 3-face umbrellas pinched at the hub give the
    /// seed a real branch.
    ///
    /// Hub disk (closing-edge-first): `2 1 3 4 6 5 7 8`, anchor `{0,2}` — *interior*
    /// to fan A's edge path `{0,1} — {0,2} — {0,3} — {0,4}`. `{0,2}`'s radial walk
    /// starts at its entry (face 1) and then reaches face 0, so its partners are
    /// pushed `{0,3}` then `{0,1}`; popping most-recent-first discovers
    /// `{0,2} {0,1} {0,3} {0,4}` and the reverse drain seats the new vertex's anchor
    /// on `{0,4}`, giving `4 3 1 2`.
    ///
    /// A FIFO frontier would discover `2 3 1 4` and read `4 1 3 2` — same fan, same
    /// incidences, a different disk order, which a later fan read resolves
    /// differently.
    #[test]
    fn vert_separate_collects_each_fan_with_a_lifo_frontier() {
        let mut mesh = LoopMesh::new();
        let mut verts = build(
            &mut mesh,
            9,
            &[
                [0, 1, 2],
                [0, 2, 3],
                [0, 3, 4],
                [0, 5, 6],
                [0, 6, 7],
                [0, 7, 8],
            ],
        );
        assert_eq!(
            disk_far_endpoints(&mesh, &verts, 0),
            vec![2, 1, 3, 4, 6, 5, 7, 8],
            "the anchor 0-2 is interior to fan A's edge path"
        );

        let fans = mesh.euler_vert_separate(verts[0]);
        mesh.validate_topology();
        assert_eq!(fans.len(), 2);
        verts.push(fans[1]);

        assert_eq!(
            disk_far_endpoints(&mesh, &verts, 0),
            vec![6, 5, 7, 8],
            "the hub keeps the LAST-discovered fan, anchored on its earliest survivor"
        );
        assert_eq!(
            disk_far_endpoints(&mesh, &verts, 9),
            vec![4, 3, 1, 2],
            "LIFO discovery 2 1 3 4, re-attached in reverse"
        );
    }
}
