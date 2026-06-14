use glam::Vec3;
use smallvec::SmallVec;

use crate::loopmesh::*;

impl LoopMesh {
    /// Forced merge: fuse every input vert into a single survivor and park that
    /// survivor at `merge_co`. Unlike the doubles family this makes no geometric
    /// decision — the input set *is* the cluster, and which vert lives is fixed
    /// by position in the input order, not proximity.
    ///
    /// Implements `radial-mesh-guides/operators/POINTMERGE.md`:
    ///
    /// - **§3 survivor choice** — the survivor is the first vert in `verts`.
    ///   Its identity (handle, per-vert customdata, header flags) persists; every
    ///   subsequent vert maps onto it.
    /// - **§4 move-to-target** — the survivor's position is overwritten with
    ///   `merge_co` *before* the weld runs. This is the only position write the
    ///   operator performs; the non-survivors keep their own positions until they
    ///   are deleted by the weld, and nothing is averaged.
    /// - **§1 / §2 surgery** — the synthesised `(non-survivor → survivor)` map is
    ///   a flat all-onto-one relation handed straight to [`Self::weld_verts`]
    ///   with `use_centroid = false` (§1.1). All edge collapse / face reap /
    ///   duplicate-face / pre-split behaviour is the weld contract, inherited
    ///   unchanged.
    /// - **§5 edge cases** — a single input vert synthesises an empty map, so the
    ///   weld is a no-op and the operator is a pure move of that vert to
    ///   `merge_co`. An empty input set is a complete no-op (no survivor, no
    ///   write). Disconnected-island verts fuse unconditionally; connectivity is
    ///   never consulted.
    ///
    /// The input order is part of the contract: it decides which handle survives
    /// (§3) and therefore which per-vert customdata and flags the result carries.
    pub fn pointmerge(&mut self, verts: &[VertexKey], merge_co: [f32; 3]) {
        // §5 — empty input writes nothing at all.
        let Some((&survivor, rest)) = verts.split_first() else {
            return;
        };

        // §4 — the one position write, before any surgery.
        self.set_vertex_position(survivor, Vec3::from(merge_co));

        // §3 — the trivial all-onto-one map; the survivor is never a key, so the
        // relation is flat by construction (WELD_VERTS §1.1). A single input
        // vert leaves this empty, making the weld a no-op (§5).
        let pairs: SmallVec<[(VertexKey, VertexKey); 16]> =
            rest.iter().map(|&source| (source, survivor)).collect();

        // §1.1 — use_centroid is hard-wired off: the survivor stays at the
        // merge_co already written above, never the input centroid.
        self.weld_verts(&pairs, false);
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use glam::Vec3;

    use crate::loopmesh::*;

    /// A single n-gon over `n` source verts arranged on a circle of `radius`,
    /// plus an extra loose target vert. Mirrors the `pointmerge_cluster` fixture
    /// shape from the guide's §6 coverage notes.
    fn pointmerge_cluster(
        mesh: &mut LoopMesh,
        n: usize,
        radius: f32,
    ) -> (Vec<VertexKey>, VertexKey) {
        let sources: Vec<VertexKey> = (0..n)
            .map(|i| {
                let theta = std::f32::consts::TAU * (i as f32) / (n as f32);
                mesh.create_vertex(Vec3::new(radius * theta.cos(), radius * theta.sin(), 0.0))
            })
            .collect();
        let edges: Vec<EdgeKey> = (0..n)
            .map(|i| {
                mesh.create_edge(
                    [sources[i], sources[(i + 1) % n]],
                    EdgeCreateFlags::NO_DUPLICATE,
                )
            })
            .collect();
        mesh.face_create(&sources, &edges, FaceCreateFlags::NONE);
        let target = mesh.create_vertex(Vec3::new(5.0, 5.0, 0.0));
        (sources, target)
    }

    /// §1 + §2 + §3 — every input vert but the first fuses into the first; the
    /// source face collapses below 3 distinct corners and is reaped, leaving a
    /// single survivor.
    #[test]
    fn pointmerge_cluster_collapses_to_single_survivor() {
        let mut mesh = LoopMesh::new();
        let (sources, target) = pointmerge_cluster(&mut mesh, 4, 1.0);
        // Merge the whole cluster (face corners + the loose target) to one point.
        let mut verts = sources.clone();
        verts.push(target);

        mesh.pointmerge(&verts, [3.0, 0.0, 0.0]);

        assert_eq!(
            mesh.vertex_count(),
            1,
            "the whole input set fused to one survivor"
        );
        assert_eq!(
            mesh.face_count(),
            0,
            "the source face degenerated and was reaped"
        );
        let survivor = mesh.vert_keys()[0];
        assert_eq!(survivor, sources[0], "the first input vert is the survivor");
        mesh.validate_topology();
    }

    /// §4 + §6 — the survivor lands at exactly `merge_co`, NOT the input
    /// centroid. A regression running the inner weld with `use_centroid = true`
    /// would place it on the average; the position pin catches that.
    #[test]
    fn pointmerge_survivor_lands_at_merge_co_not_centroid() {
        let mut mesh = LoopMesh::new();
        let (sources, target) = pointmerge_cluster(&mut mesh, 4, 1.0);
        let mut verts = sources.clone();
        verts.push(target);

        let merge_co = [7.0, -2.0, 1.0];
        // The centroid of the inputs is far from merge_co (the loose target sits
        // at (5,5,0), so the average is nowhere near (7,-2,1)).
        mesh.pointmerge(&verts, merge_co);

        let survivor = mesh.vert_keys()[0];
        assert_eq!(
            mesh.vertex_position(survivor),
            Vec3::from(merge_co),
            "the survivor sits at merge_co, not the input centroid"
        );
        mesh.validate_topology();
    }

    /// §5 — two verts joined by an edge: the second maps onto the first, the
    /// edge collapses to zero length and is reaped, and the survivor relocates to
    /// `merge_co`.
    #[test]
    fn pointmerge_two_adjacent_verts_collapse_to_point() {
        let mut mesh = LoopMesh::new();
        let a = mesh.create_vertex(Vec3::new(0.0, 0.0, 0.0));
        let b = mesh.create_vertex(Vec3::new(1.0, 0.0, 0.0));
        mesh.create_edge([a, b], EdgeCreateFlags::NO_DUPLICATE);

        let merge_co = [4.0, 4.0, 0.0];
        mesh.pointmerge(&[a, b], merge_co);

        assert_eq!(mesh.vertex_count(), 1, "the pair fused to one survivor");
        assert_eq!(mesh.edge_count(), 0, "the collapsed edge is reaped");
        let survivor = mesh.vert_keys()[0];
        assert_eq!(survivor, a, "the first input vert survives");
        assert_eq!(
            mesh.vertex_position(survivor),
            Vec3::from(merge_co),
            "the survivor relocated to merge_co"
        );
        mesh.validate_topology();
    }

    /// §5 — a single input vert: empty synthesised map, weld is a no-op, but the
    /// survivor has still been moved to `merge_co`. Nothing is removed.
    #[test]
    fn pointmerge_single_vert_is_pure_move() {
        let mut mesh = LoopMesh::new();
        let a = mesh.create_vertex(Vec3::new(0.0, 0.0, 0.0));
        let b = mesh.create_vertex(Vec3::new(1.0, 0.0, 0.0));
        let c = mesh.create_vertex(Vec3::new(2.0, 0.0, 0.0));
        let before = (mesh.vertex_count(), mesh.edge_count(), mesh.face_count());

        let merge_co = [9.0, 9.0, 9.0];
        mesh.pointmerge(&[b], merge_co);

        assert_eq!(
            (mesh.vertex_count(), mesh.edge_count(), mesh.face_count()),
            before,
            "pure move removes nothing"
        );
        assert_eq!(
            mesh.vertex_position(b),
            Vec3::from(merge_co),
            "the lone input vert relocated to merge_co"
        );
        // The untouched verts keep their own positions.
        assert_eq!(mesh.vertex_position(a), Vec3::new(0.0, 0.0, 0.0));
        assert_eq!(mesh.vertex_position(c), Vec3::new(2.0, 0.0, 0.0));
        mesh.validate_topology();
    }

    /// §5 — input verts from two unconnected islands fuse into one survivor
    /// regardless of connectivity (no proximity test, no connected flood).
    #[test]
    fn pointmerge_disconnected_verts_fuse_unconditionally() {
        let mut mesh = LoopMesh::new();
        // Two separate wire islands with no incidence path between them.
        let a0 = mesh.create_vertex(Vec3::new(-5.0, 0.0, 0.0));
        let a1 = mesh.create_vertex(Vec3::new(-4.0, 0.0, 0.0));
        mesh.create_edge([a0, a1], EdgeCreateFlags::NO_DUPLICATE);
        let b0 = mesh.create_vertex(Vec3::new(5.0, 0.0, 0.0));
        let b1 = mesh.create_vertex(Vec3::new(4.0, 0.0, 0.0));
        mesh.create_edge([b0, b1], EdgeCreateFlags::NO_DUPLICATE);

        let merge_co = [0.0, 3.0, 0.0];
        // Fuse one endpoint from each island; they share nothing.
        mesh.pointmerge(&[a0, b0], merge_co);

        let live: HashSet<VertexKey> = mesh.vert_keys().into_iter().collect();
        assert!(live.contains(&a0), "the first input vert survives");
        assert!(
            !live.contains(&b0),
            "the second island vert fused onto the survivor"
        );
        assert_eq!(
            mesh.vertex_position(a0),
            Vec3::from(merge_co),
            "the survivor sits at merge_co"
        );
        // Both islands' remaining wire anchors stay; the survivor now bridges
        // their incident edges.
        assert!(
            live.contains(&a1) && live.contains(&b1),
            "the wire anchors stay"
        );
        mesh.validate_topology();
    }

    /// §5 — empty input set is a complete no-op: nothing written, nothing
    /// removed.
    #[test]
    fn pointmerge_empty_input_is_noop() {
        let mut mesh = LoopMesh::new();
        let a = mesh.create_vertex(Vec3::new(1.0, 2.0, 3.0));
        let before = (mesh.vertex_count(), mesh.edge_count(), mesh.face_count());

        mesh.pointmerge(&[], [0.0, 0.0, 0.0]);

        assert_eq!(
            (mesh.vertex_count(), mesh.edge_count(), mesh.face_count()),
            before
        );
        assert_eq!(
            mesh.vertex_position(a),
            Vec3::new(1.0, 2.0, 3.0),
            "no position written"
        );
        mesh.validate_topology();
    }
}
