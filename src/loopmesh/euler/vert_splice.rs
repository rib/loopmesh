use crate::loopmesh::*;

impl LoopMesh {
    /// Splice every edge of `v_src` onto `v_dst`, then delete `v_src`.
    ///
    /// This is the inverse of [`LoopMesh::euler_vert_separate`]: after a
    /// vertex-separate call has produced two or more fan-instances at the
    /// same source position, callers may want to re-merge a subset of those
    /// fans (typically the ones consisting of wire edges or non-tagged faces
    /// in operators like inset that otherwise leave extra "loose" copies of
    /// the vertex behind).
    ///
    /// The destination vertex's customdata is left as-is — the splice does
    /// not interpolate.
    pub fn euler_vert_splice(&mut self, v_dst: VertexKey, v_src: VertexKey) {
        if v_dst == v_src {
            return;
        }
        let edges: Vec<EdgeKey> = self.vertex_disk_iter(v_src).collect();
        for e in edges {
            self.edge_replace_vertex_raw(e, v_src, v_dst);
        }
        self.remove_vertex_slab_entry(v_src);
    }
}

#[cfg(test)]
mod tests {
    use glam::Vec3;

    use crate::loopmesh::*;

    /// vert_separate followed by vert_splice round-trips the topology: a
    /// non-manifold "bowtie" pair of triangles sharing a single corner vertex,
    /// when separated and then spliced back together, restores the original
    /// edge count.
    #[test]
    fn vert_splice_round_trips_after_separate() {
        let mut mesh = LoopMesh::new();
        let verts: Vec<VertexKey> = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (2.0, 1.0), (2.0, 2.0)]
            .iter()
            .map(|&(x, y)| mesh.create_vertex(Vec3::new(x, y, 0.0)))
            .collect();
        let e01 = mesh.create_edge([verts[0], verts[1]], EdgeCreateFlags::NONE);
        let e12 = mesh.create_edge([verts[1], verts[2]], EdgeCreateFlags::NONE);
        let e02 = mesh.create_edge([verts[0], verts[2]], EdgeCreateFlags::NONE);
        let _ = mesh.face_create(&verts[..3], &[e01, e12, e02], FaceCreateFlags::NONE);
        let e23 = mesh.create_edge([verts[2], verts[3]], EdgeCreateFlags::NONE);
        let e34 = mesh.create_edge([verts[3], verts[4]], EdgeCreateFlags::NONE);
        let e24 = mesh.create_edge([verts[2], verts[4]], EdgeCreateFlags::NONE);
        let _ = mesh.face_create(&verts[2..], &[e23, e34, e24], FaceCreateFlags::NONE);

        let v_pinch = verts[2];
        let v_before = mesh.vertex_count();

        let fans = mesh.euler_vert_separate(v_pinch);
        assert_eq!(fans.len(), 2, "bowtie pinch should split into 2 fans");
        assert_eq!(mesh.vertex_count(), v_before + 1);

        mesh.euler_vert_splice(fans[0], fans[1]);
        assert_eq!(mesh.vertex_count(), v_before);
        mesh.validate_topology();
    }
}
