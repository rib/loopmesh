use crate::loopmesh::*;

impl LoopMesh {
    /// Merge-by-distance cleanup: discover which input verts fall within `dist`
    /// of one another, pick one survivor per cluster, and weld every other
    /// member onto its survivor in place.
    ///
    /// Implements `radial-mesh-guides/operators/REMOVE_DOUBLES.md` as the named
    /// composition of its two sub-operators with no intervening edit:
    ///
    /// - **§3 composition seam** — the clustering pass is
    ///   [`Self::find_doubles`] verbatim ([`FIND_DOUBLES.md`] §3/§4/§6), and the
    ///   `(source -> survivor)` map it returns feeds straight into
    ///   [`Self::weld_verts`] ([`WELD_VERTS.md`]). No map is surfaced to the
    ///   caller. Because the arrows come from the find-doubles emission they are
    ///   flat (every value is a final survivor, never itself a key), satisfying
    ///   the weld's flat-map invariant for free.
    /// - **§1 / §1.1 parameters** — only `verts`, `dist`, and `use_connected`
    ///   are forwarded to the clustering pass. There is **no `keep_verts`
    ///   input**: the protected set is left empty, so within the input set any
    ///   member of a cluster may be welded away. A caller who needs protected
    ///   survivors must call [`Self::find_doubles`] (with its `keep_verts`) and
    ///   weld the map separately.
    /// - **§4 position contract** — the inner weld always runs with
    ///   `use_centroid = false`, so every survivor keeps its **own** position
    ///   (the find-doubles centroid only *selects* the survivor of a 3+ cluster,
    ///   it never moves it). There is no averaging.
    /// - **§5 edge cases** — an empty map (nothing within `dist`) is a no-op
    ///   ([`WELD_VERTS.md`] §7); a shared edge between two merged verts collapses
    ///   and reaps below-3-corner faces; the non-adjacent self-merge pre-split,
    ///   duplicate-face adoption, and the boundary conventions at exactly `dist`
    ///   are all inherited unchanged from the two sub-operators.
    ///
    /// The input order of `verts` is part of the contract: it decides the
    /// find-doubles survivor tie-break and which cluster claims a vert reachable
    /// from two, and therefore the merged mesh.
    pub fn remove_doubles(&mut self, verts: &[VertexKey], dist: f32, use_connected: bool) {
        // §1.1 — no keep_verts input: the protected set is always empty.
        let pairs = self.find_doubles(verts, &[], dist, use_connected);
        // §4 — use_centroid is hard-wired off; survivors keep their own
        // position. An empty map welds to a no-op (§5).
        self.weld_verts(&pairs, false);
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use glam::Vec3;

    use crate::loopmesh::*;

    /// A tight ring of `n` near-coincident verts around the origin at radius
    /// `r`, plus a far isolated vert. The cluster is *near* but not coincident
    /// so the survivor-position contract (§4) is observable.
    fn cluster_plus_far(mesh: &mut LoopMesh, n: usize, r: f32) -> (Vec<VertexKey>, VertexKey) {
        let cluster: Vec<VertexKey> = (0..n)
            .map(|i| {
                let theta = std::f32::consts::TAU * (i as f32) / (n as f32);
                mesh.create_vertex(Vec3::new(r * theta.cos(), r * theta.sin(), 0.0))
            })
            .collect();
        let far = mesh.create_vertex(Vec3::new(100.0, 0.0, 0.0));
        (cluster, far)
    }

    /// §2 / §3 — a near-coincident cluster collapses to one survivor; the far
    /// vert is untouched. The basic composition: find-doubles clusters, weld
    /// removes the non-survivors.
    #[test]
    fn composition_collapses_near_coincident_cluster() {
        let mut mesh = LoopMesh::new();
        let (cluster, far) = cluster_plus_far(&mut mesh, 3, 0.05);
        let mut verts = cluster.clone();
        verts.push(far);

        mesh.remove_doubles(&verts, 0.5, false);

        assert_eq!(mesh.totvert(), 2, "the triple collapsed to one survivor plus the far vert");
        let live: HashSet<VertexKey> = mesh.vert_keys().into_iter().collect();
        assert!(live.contains(&far), "the far vert is untouched");
        // Exactly one cluster member survives.
        let surviving_cluster: Vec<VertexKey> =
            cluster.iter().copied().filter(|v| live.contains(v)).collect();
        assert_eq!(surviving_cluster.len(), 1, "exactly one cluster member survives");
        mesh.validate_topology();
    }

    /// §5 / §7 — `dist` above the spread merges (vert count drops); `dist` below
    /// the spread is a no-op (mesh unchanged). The threshold fires-vs-blocks
    /// contrast.
    #[test]
    fn dist_threshold_fires_and_blocks() {
        // Fires.
        let mut fired = LoopMesh::new();
        let (cluster, far) = cluster_plus_far(&mut fired, 3, 0.05);
        let mut verts = cluster.clone();
        verts.push(far);
        fired.remove_doubles(&verts, 0.5, false);
        assert_eq!(fired.totvert(), 2, "dist above the spread merges the cluster");

        // Blocks.
        let mut blocked = LoopMesh::new();
        let (cluster_b, far_b) = cluster_plus_far(&mut blocked, 3, 0.05);
        let mut verts_b = cluster_b.clone();
        verts_b.push(far_b);
        let before = (blocked.totvert(), blocked.totedge(), blocked.totface());
        blocked.remove_doubles(&verts_b, 0.001, false);
        assert_eq!(
            (blocked.totvert(), blocked.totedge(), blocked.totface()),
            before,
            "dist below the spread leaves the mesh unchanged"
        );
        blocked.validate_topology();
    }

    /// §4 — with `use_centroid` off the survivor sits at its own input position,
    /// NOT the cluster centroid. A near-but-not-coincident pair makes this
    /// observable: the survivor must keep its exact coordinate.
    #[test]
    fn survivor_keeps_own_position_not_centroid() {
        let mut mesh = LoopMesh::new();
        let a = mesh.create_vertex(Vec3::new(0.0, 0.0, 0.0));
        let b = mesh.create_vertex(Vec3::new(0.1, 0.0, 0.0));
        let a_pos = mesh.vertex_position(a);
        let verts = [a, b];

        // Lower index `a` survives (find-doubles pair tie-break); it must stay
        // at its own position, not the midpoint (0.05, 0, 0).
        mesh.remove_doubles(&verts, 0.5, false);

        assert_eq!(mesh.totvert(), 1, "the pair merged to one vert");
        let survivor = mesh.vert_keys()[0];
        assert_eq!(survivor, a, "the lower-index member is the survivor");
        assert_eq!(
            mesh.vertex_position(survivor),
            a_pos,
            "the survivor keeps its own position, no centroid averaging"
        );
        mesh.validate_topology();
    }

    /// §5 — two near-coincident endpoints of a shared edge: the edge collapses
    /// and each adjacent quad drops a corner to a triangle (the weld surgery
    /// firing through the composition).
    #[test]
    fn shared_edge_collapses_and_reaps() {
        // Two quads sharing the vertical edge {b, c}; `b` is nudged near `c`.
        let mut mesh = LoopMesh::new();
        let a0 = mesh.create_vertex(Vec3::new(0.0, 0.0, 0.0));
        let b = mesh.create_vertex(Vec3::new(1.0, 0.98, 0.0));
        let c = mesh.create_vertex(Vec3::new(1.0, 1.0, 0.0));
        let a1 = mesh.create_vertex(Vec3::new(0.0, 1.0, 0.0));
        let d0 = mesh.create_vertex(Vec3::new(2.0, 0.0, 0.0));
        let d1 = mesh.create_vertex(Vec3::new(2.0, 1.0, 0.0));

        let mk = |mesh: &mut LoopMesh, vs: [VertexKey; 4]| {
            let es: Vec<EdgeKey> = (0..4)
                .map(|i| mesh.create_edge([vs[i], vs[(i + 1) % 4]], EdgeCreateFlags::NO_DUPLICATE))
                .collect();
            mesh.face_create(&vs, &es, FaceCreateFlags::NONE)
        };
        mk(&mut mesh, [a0, b, c, a1]);
        mk(&mut mesh, [b, d0, d1, c]);
        assert_eq!(mesh.totface(), 2);

        // dist 0.1 covers the 0.02 gap between b and c; nothing else is close.
        mesh.remove_doubles(&[a0, b, c, a1, d0, d1], 0.1, false);

        let live: HashSet<VertexKey> = mesh.vert_keys().into_iter().collect();
        assert!(!live.contains(&b) || !live.contains(&c), "one of the shared endpoints is gone");
        assert_eq!(mesh.totface(), 2, "both quads survive as triangles");
        for face in mesh.face_keys() {
            assert_eq!(mesh.face_len(face), 3, "each quad lost the collapsed corner");
        }
        mesh.validate_topology();
    }

    /// §5 / §7 (FIND_DOUBLES §4.3) — two near-coincident verts with no incidence
    /// path between them: the spatial pass merges them (mesh changes), the
    /// connected pass leaves them apart (no-op). The defining `use_connected`
    /// divergence, observable as differing meshes.
    #[test]
    fn use_connected_vs_spatial_diverge() {
        let build = |mesh: &mut LoopMesh| -> [VertexKey; 4] {
            let a = mesh.create_vertex(Vec3::new(0.0, 0.0, 0.0));
            let a_anchor = mesh.create_vertex(Vec3::new(0.0, 5.0, 0.0));
            let b = mesh.create_vertex(Vec3::new(0.01, 0.0, 0.0));
            let b_anchor = mesh.create_vertex(Vec3::new(0.01, -5.0, 0.0));
            mesh.create_edge([a, a_anchor], EdgeCreateFlags::NO_DUPLICATE);
            mesh.create_edge([b, b_anchor], EdgeCreateFlags::NO_DUPLICATE);
            [a, a_anchor, b, b_anchor]
        };

        let mut spatial = LoopMesh::new();
        let verts_s = build(&mut spatial);
        spatial.remove_doubles(&verts_s, 0.5, false);
        assert_eq!(spatial.totvert(), 3, "spatial merges the coincident disconnected pair");
        spatial.validate_topology();

        let mut connected = LoopMesh::new();
        let verts_c = build(&mut connected);
        let before = (connected.totvert(), connected.totedge(), connected.totface());
        connected.remove_doubles(&verts_c, 0.5, true);
        assert_eq!(
            (connected.totvert(), connected.totedge(), connected.totface()),
            before,
            "connected leaves the disconnected pair apart: no-op"
        );
        connected.validate_topology();
    }

    /// §5 — nothing within `dist` (and an empty input) are both no-ops, leaving
    /// the mesh bit-for-bit unchanged. The lower-bound contrast.
    #[test]
    fn empty_and_separated_are_noops() {
        let mut mesh = LoopMesh::new();
        let verts: Vec<VertexKey> = (0..4)
            .map(|i| mesh.create_vertex(Vec3::new(i as f32, 0.0, 0.0)))
            .collect();
        let before = (mesh.totvert(), mesh.totedge(), mesh.totface());

        // Well-separated input: empty map, no-op.
        mesh.remove_doubles(&verts, 0.5, false);
        assert_eq!((mesh.totvert(), mesh.totedge(), mesh.totface()), before);

        // Empty input slot: nothing considered, no-op.
        mesh.remove_doubles(&[], 0.5, false);
        assert_eq!((mesh.totvert(), mesh.totedge(), mesh.totface()), before);
        mesh.validate_topology();
    }

    /// §5 / §7 — a bowtie is accepted with no special non-manifold rejection:
    /// fusing one outer corner of each wing collapses through the shared hub
    /// just as the weld contract dictates.
    #[test]
    fn non_manifold_bowtie_accepted() {
        let mut mesh = LoopMesh::new();
        // Two triangles sharing only the apex `hub`. The outer corners a0 and b0
        // are nudged near each other but share no edge.
        let hub = mesh.create_vertex(Vec3::new(0.0, 0.0, 0.0));
        let a0 = mesh.create_vertex(Vec3::new(1.0, 1.0, 0.0));
        let a1 = mesh.create_vertex(Vec3::new(-1.0, 1.0, 0.0));
        let b0 = mesh.create_vertex(Vec3::new(1.01, 1.0, 0.0));
        let b1 = mesh.create_vertex(Vec3::new(1.0, -1.0, 0.0));

        let mk = |mesh: &mut LoopMesh, vs: [VertexKey; 3]| {
            let es: Vec<EdgeKey> = (0..3)
                .map(|i| mesh.create_edge([vs[i], vs[(i + 1) % 3]], EdgeCreateFlags::NO_DUPLICATE))
                .collect();
            mesh.face_create(&vs, &es, FaceCreateFlags::NONE)
        };
        mk(&mut mesh, [hub, a0, a1]);
        mk(&mut mesh, [hub, b0, b1]);
        assert_eq!(mesh.totface(), 2);

        // a0 and b0 are within 0.1 (gap 0.01); nothing else is close.
        mesh.remove_doubles(&[hub, a0, a1, b0, b1], 0.1, false);

        // No special rejection: the cleanup runs. a0 and b0 fuse to one vert;
        // each wing triangle keeps its three distinct corners.
        let live: HashSet<VertexKey> = mesh.vert_keys().into_iter().collect();
        assert!(!live.contains(&a0) || !live.contains(&b0), "the near pair fused");
        assert_eq!(mesh.totvert(), 4, "a0/b0 fused: hub + survivor + a1 + b1");
        assert_eq!(mesh.totface(), 2, "both wings survive as triangles");
        mesh.validate_topology();
    }
}
