use crate::loopmesh::*;

impl LoopMesh {
    /// Separates the disjoint face-fans at a single vertex into multiple
    /// vertices. Returns the list of resulting vertices (the input vertex is
    /// always returned first; one extra vertex is created per additional
    /// fan).
    ///
    /// For a normal manifold vertex with a single connected disk this is a
    /// no-op and the returned Vec contains just the input.
    pub fn euler_vert_separate(&mut self, v: VertexKey) -> Vec<VertexKey> {
        let mut result = vec![v];
        let mut remaining: Vec<EdgeKey> = self.vertex_disk_iter(v).collect();
        if remaining.len() <= 1 {
            return result;
        }

        // Helper: gather one fan (connected component through shared faces)
        // starting from `seed`. Returns the edges in the fan.
        let gather_fan = |this: &Self, seed: EdgeKey| -> Vec<EdgeKey> {
            let mut fan = vec![seed];
            let mut frontier = vec![seed];
            while let Some(e) = frontier.pop() {
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
                    let prev_l = this.loops.get(l_at_v).face_loop_links.prev;
                    let other_edge = this.loops.get(prev_l).edge;
                    if !other_edge.is_null() && !fan.contains(&other_edge) {
                        fan.push(other_edge);
                        frontier.push(other_edge);
                    }
                    let this_edge = this.loops.get(l_at_v).edge;
                    if !this_edge.is_null() && !fan.contains(&this_edge) {
                        fan.push(this_edge);
                        frontier.push(this_edge);
                    }
                }
            }
            fan
        };

        // Walk fans starting from the first disk edge and move each fan EXCEPT
        // THE LAST onto a new vertex. The last fan stays anchored on `v`.
        let mut fans: Vec<Vec<EdgeKey>> = Vec::new();
        while !remaining.is_empty() {
            let fan = gather_fan(self, remaining[0]);
            remaining.retain(|e| !fan.contains(e));
            fans.push(fan);
        }

        let v_pos = self.vertices.get(v).position;
        // Move all but the last fan onto fresh verts.
        let last_idx = fans.len() - 1;
        for (i, fan) in fans.iter().enumerate() {
            if i == last_idx {
                break;
            }
            let v_new = self.create_vertex(v_pos);
            for &e in fan {
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
