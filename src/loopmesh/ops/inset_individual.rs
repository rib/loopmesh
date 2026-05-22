use glam::Vec3;

use crate::loopmesh::*;

#[derive(Clone, Copy, Default)]
pub struct InsetIndividualParams {
    pub thickness: f32,
    pub depth: f32,
    pub use_even_offset: bool,
    pub use_interpolate: bool,
    pub use_relative_offset: bool,
}

#[derive(Default)]
pub struct InsetIndividualResult {
    pub wall_faces: Vec<FaceKey>,
}

impl LoopMesh {
    /// Inset each input face independently — replace the face with a smaller
    /// copy of itself in its own plane and stitch a ring of wall quads around
    /// it. No coordination between adjacent input faces; each is processed in
    /// isolation, so a corner shared by two input faces ends up with two
    /// disjoint inset copies rather than one merged corner.
    ///
    /// The per-corner offset uses the same bisector + `1/cos(half-angle)`
    /// math as [`LoopMesh::inset_region`] — the difference is purely in which
    /// edges count as the "seam" (here: every edge of the face).
    ///
    /// `use_interpolate` (re-bake of face customdata after the corners move)
    /// is not yet implemented; the per-face cd is copied verbatim onto the
    /// new face identity, which matches BMesh's behaviour when
    /// `use_interpolate=false`.
    pub fn inset_individual(
        &mut self,
        faces: &[FaceKey],
        params: InsetIndividualParams,
    ) -> InsetIndividualResult {
        if faces.is_empty() {
            return InsetIndividualResult::default();
        }
        self.calc_face_normals();
        let mut wall_faces: Vec<FaceKey> = Vec::new();
        for &f in faces {
            self.inset_one_face(f, &params, &mut wall_faces);
        }
        InsetIndividualResult { wall_faces }
    }

    fn inset_one_face(
        &mut self,
        f: FaceKey,
        params: &InsetIndividualParams,
        wall_faces: &mut Vec<FaceKey>,
    ) {
        // Collect the face's loop cycle. `loops[i]` sits on the edge between
        // its own vertex and the next loop's vertex.
        let loops: Vec<LoopKey> = {
            let mut it = self.face_loops(f);
            let mut acc = Vec::new();
            while let Some(l) = it.next(self) {
                acc.push(l);
            }
            acc
        };
        let n_loops = loops.len();
        if n_loops < 3 {
            return;
        }

        let face_normal = self.face_normal(f);
        let old_verts: Vec<VertexKey> = loops.iter().map(|&l| self.loop_vertex(l)).collect();
        let old_edges: Vec<EdgeKey> = loops.iter().map(|&l| self.loop_edge(l)).collect();
        let old_positions: Vec<Vec3> = old_verts.iter().map(|&v| self.vertex_position(v)).collect();

        // Per-edge inward tangent in the face plane: n × normalize(next-pos).
        let mut tangents: Vec<Vec3> = Vec::with_capacity(n_loops);
        let mut lengths: Vec<f32> = Vec::with_capacity(n_loops);
        for i in 0..n_loops {
            let p_a = old_positions[i];
            let p_b = old_positions[(i + 1) % n_loops];
            let edge_dir = (p_b - p_a).normalize_or_zero();
            tangents.push(face_normal.cross(edge_dir).normalize_or_zero());
            lengths.push((p_b - p_a).length());
        }

        // Per-corner inset position. Corner `i` is between edge `i-1` (prev,
        // incoming) and edge `i` (next, outgoing).
        let mut new_positions: Vec<Vec3> = Vec::with_capacity(n_loops);
        for i in 0..n_loops {
            let i_prev = (i + n_loops - 1) % n_loops;
            let t_in = tangents[i_prev];
            let t_out = tangents[i];
            let bisector = (t_in + t_out).normalize_or_zero();
            let dir = if params.use_even_offset {
                let cos_half = t_in.dot(bisector).abs();
                if cos_half > 1.0e-6 {
                    bisector / cos_half
                } else {
                    bisector
                }
            } else {
                bisector
            };
            let scale = if params.use_relative_offset {
                0.5 * (lengths[i_prev] + lengths[i])
            } else {
                1.0
            };
            new_positions.push(old_positions[i] + dir * params.thickness * scale);
        }

        // Duplicate verts at the new positions; duplicate edges between
        // consecutive new verts.
        let mut new_verts: Vec<VertexKey> = Vec::with_capacity(n_loops);
        for i in 0..n_loops {
            let v_new = self.create_vertex(new_positions[i]);
            self.vertex_attribs
                .copy_into(v_new.as_usize(), old_verts[i].as_usize());
            new_verts.push(v_new);
        }
        let mut new_edges: Vec<EdgeKey> = Vec::with_capacity(n_loops);
        for i in 0..n_loops {
            let v0 = new_verts[i];
            let v1 = new_verts[(i + 1) % n_loops];
            let e_new = self.create_edge([v0, v1], EdgeCreateFlags::NONE);
            self.edge_attribs
                .copy_into(e_new.as_usize(), old_edges[i].as_usize());
            new_edges.push(e_new);
        }

        // Re-anchor `f`'s loops to the new verts and new edges. This leaves
        // the original verts/edges still attached to whatever neighbouring
        // faces they had — `f` simply slides off them.
        for i in 0..n_loops {
            let l = loops[i];
            let new_e = new_edges[i];
            let new_v = new_verts[i];
            self.edge_replace_in_face_loop(l, new_e);
            self.loops.get_mut(l).vertex = new_v;
        }

        // Build wall quads — one per original edge — connecting (v_i, v_{i+1},
        // v_{i+1}', v_i'). Side edges between matched original / new corners
        // are shared between two walls at the same corner.
        let mut side_edges: Vec<EdgeKey> = Vec::with_capacity(n_loops);
        for i in 0..n_loops {
            let e = self.create_edge([old_verts[i], new_verts[i]], EdgeCreateFlags::NONE);
            side_edges.push(e);
        }
        for i in 0..n_loops {
            let i_next = (i + 1) % n_loops;
            let verts = [
                old_verts[i],
                old_verts[i_next],
                new_verts[i_next],
                new_verts[i],
            ];
            let edges = [
                old_edges[i],
                side_edges[i_next],
                new_edges[i],
                side_edges[i],
            ];
            let wall = self.face_create(&verts, &edges, FaceCreateFlags::NONE);
            self.face_attribs.copy_into(wall.as_usize(), f.as_usize());
            wall_faces.push(wall);
        }

        // Depth pass: lift every new vert along the face's normal.
        if params.depth != 0.0 {
            for i in 0..n_loops {
                let p = self.vertex_position(new_verts[i]);
                let scale = if params.use_relative_offset {
                    let i_prev = (i + n_loops - 1) % n_loops;
                    0.5 * (lengths[i_prev] + lengths[i])
                } else {
                    1.0
                };
                self.set_vertex_position(new_verts[i], p + face_normal * params.depth * scale);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use glam::Vec3;

    use super::{InsetIndividualParams, InsetIndividualResult};
    use crate::loopmesh::test_support::create_face_from_verts;
    use crate::loopmesh::*;

    #[allow(dead_code)]
    fn _result_type_used(r: InsetIndividualResult) -> usize {
        r.wall_faces.len()
    }

    #[test]
    fn inset_individual_single_quad() {
        let mut mesh = LoopMesh::new();
        let verts = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        ];
        let face = create_face_from_verts(&mut mesh, &verts);
        let result = mesh.inset_individual(
            &[face],
            InsetIndividualParams {
                thickness: 0.1,
                use_even_offset: true,
                ..Default::default()
            },
        );
        assert_eq!(result.wall_faces.len(), 4);
        assert_eq!(mesh.totvert(), 8);
        assert_eq!(mesh.totface(), 5);
        mesh.validate_topology();
    }
}
