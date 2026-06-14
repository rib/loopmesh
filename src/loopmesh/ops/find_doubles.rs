use std::collections::HashMap;

use glam::Vec3;
use smallvec::SmallVec;

use crate::loopmesh::*;

impl LoopMesh {
    /// Build the dedup map that `weld_verts` consumes, without touching
    /// topology: discover which input verts are close enough to be considered
    /// the same point, pick one survivor per cluster, and return one
    /// `(source -> target)` couple per non-survivor.
    ///
    /// Implements `radial-mesh-guides/operators/FIND_DOUBLES.md`:
    ///
    /// - **§3 spatial pass** (`use_connected == false`) — proximity is judged
    ///   purely on position over the whole input set, with the **non-strict**
    ///   `len² <= dist²` boundary (a vert at exactly `dist` clusters). Clusters
    ///   are grown by first-unclaimed-point order so the partition is
    ///   deterministic in input order (§7 first-claim-wins).
    /// - **§4 connected pass** (`use_connected == true`) — a flood from each
    ///   unclaimed seed, stepping across incident edges (the disk) *and* across
    ///   the non-edge-adjacent corners of incident n-gon faces (§4.1). Every
    ///   candidate is tested against the **cluster origin** (the seed position),
    ///   not the stepping vert (§4.2), under the **strict** `len² < dist²`
    ///   boundary (§4.1) — so a vert at exactly `dist` does *not* merge here.
    /// - **§4.4 survivor tie-break** — connected: the flood seed; spatial pair:
    ///   the lower input index; spatial 3+ cluster: the member closest to the
    ///   cluster centroid, lower index breaking ties (the centroid only
    ///   *selects*, it never moves the survivor).
    /// - **§5 `keep_verts`** — each protected vert is pre-seeded as its own
    ///   survivor, so it is never a source and always wins the survivor role of
    ///   its cluster; two keep-verts in one cluster both survive (§5.1).
    /// - **§6 self-mapping omission** — survivors and lone verts are *absent* as
    ///   keys; no `v -> v` couple is ever emitted, keeping the map flat for
    ///   `weld_verts`.
    ///
    /// The input order of `verts` is part of the contract: it decides the
    /// survivor tie-break and which cluster claims a vert reachable from two.
    pub fn find_doubles(
        &self,
        verts: &[VertexKey],
        keep_verts: &[VertexKey],
        dist: f32,
        use_connected: bool,
    ) -> Vec<(VertexKey, VertexKey)> {
        let n = verts.len();
        let dist_sq = dist * dist;

        // §3.2 target[i] in { -1, i, j }: unclassified, survivor, or duplicate
        // of input vert j.
        let mut target: Vec<isize> = vec![-1; n];

        // §3.3 / §5 — pre-seed protected verts as their own survivors.
        if !keep_verts.is_empty() {
            for i in 0..n {
                if keep_verts.contains(&verts[i]) {
                    target[i] = i as isize;
                }
            }
        }

        if use_connected {
            self.find_doubles_connected(verts, dist_sq, &mut target);
        } else {
            self.find_doubles_spatial(verts, dist_sq, &mut target);
        }

        // §6 — emit the sparse map: skip -1 (lone) and self-index (survivor).
        let mut map: Vec<(VertexKey, VertexKey)> = Vec::new();
        for i in 0..n {
            let t = target[i];
            if t == -1 || t == i as isize {
                continue;
            }
            map.push((verts[i], verts[t as usize]));
        }
        map
    }

    /// §3 — spatial duplicate clustering. Walk the input verts; for each not yet
    /// claimed vert, gather every still-unclaimed vert within `dist` of it (the
    /// non-strict `<= dist_sq` ball) into a cluster, choose the survivor by
    /// §4.4, and point every other member at it. Pre-seeded keep-verts act as
    /// fixed survivors and win their cluster's survivor role.
    fn find_doubles_spatial(&self, verts: &[VertexKey], dist_sq: f32, target: &mut [isize]) {
        let n = verts.len();
        for seed in 0..n {
            // Only an unclassified vert or a pre-seeded keep survivor starts a
            // cluster; a vert already mapped onto someone is skipped (§7).
            if target[seed] != -1 && target[seed] != seed as isize {
                continue;
            }
            let seed_pos = self.vertex_position(verts[seed]);

            // Members of this cluster: the seed plus every vert in the seed's
            // `dist` ball that is not yet claimed onto someone else. A vert is
            // eligible when its slot is `-1` (unclassified) or points at itself
            // (a pre-seeded keep survivor, §3.2) — the latter must participate
            // so the §5 keep-override in `choose_spatial_survivor` sees it and
            // wins, even when an ordinary vert seeds the cluster. A vert already
            // mapped onto a third vert is excluded (first-claim-wins, §7).
            let mut members: SmallVec<[usize; 16]> = SmallVec::new();
            members.push(seed);
            for other in 0..n {
                if other == seed || (target[other] != -1 && target[other] != other as isize) {
                    continue;
                }
                if self
                    .vertex_position(verts[other])
                    .distance_squared(seed_pos)
                    <= dist_sq
                {
                    members.push(other);
                }
            }

            if members.len() < 2 {
                // Lone vert: leave it -1 unless it was a pre-seeded keep (then
                // it stays its own survivor, still emitting nothing).
                continue;
            }

            let survivor = self.choose_spatial_survivor(verts, &members, target);
            for &member in &members {
                // A pre-seeded keep-vert that is not the chosen survivor stays
                // its own survivor — two keep-verts in one cluster both survive
                // (§5.1); never re-point one onto the other.
                if member != survivor && target[member] == member as isize {
                    continue;
                }
                target[member] = survivor as isize;
            }
        }
    }

    /// §4.4 — pick a cluster's survivor. A pre-seeded keep-vert (already its own
    /// survivor) overrides everything (§5). Otherwise: a pair takes the lower
    /// input index; a cluster of three or more takes the member closest to the
    /// centroid, lower index breaking ties.
    fn choose_spatial_survivor(
        &self,
        verts: &[VertexKey],
        members: &[usize],
        target: &[isize],
    ) -> usize {
        // §5 — a protected member is already a survivor and wins; the lowest
        // such index is the deterministic choice if two keep-verts coincide.
        if let Some(&kept) = members.iter().filter(|&&m| target[m] == m as isize).min() {
            return kept;
        }

        if members.len() == 2 {
            // §4.4 pair — lower input index.
            return *members.iter().min().unwrap();
        }

        // §4.4 3+ — centroid-closest, lower index breaking ties.
        let mut centroid = Vec3::ZERO;
        for &m in members {
            centroid += self.vertex_position(verts[m]);
        }
        centroid /= members.len() as f32;

        let mut best = members[0];
        let mut best_dist_sq = self.vertex_position(verts[best]).distance_squared(centroid);
        for &m in &members[1..] {
            let d = self.vertex_position(verts[m]).distance_squared(centroid);
            if d < best_dist_sq || (d == best_dist_sq && m < best) {
                best = m;
                best_dist_sq = d;
            }
        }
        best
    }

    /// §4 — connected flood. From each unclaimed seed, flood across incident
    /// edges and n-gon far corners, claiming every unclaimed input vert within
    /// the **strict** `dist` ball of the *seed origin* (§4.1 / §4.2). The seed
    /// is always the survivor (§4.4).
    fn find_doubles_connected(&self, verts: &[VertexKey], dist_sq: f32, target: &mut [isize]) {
        let n = verts.len();
        // Vert handle -> input index, built once (§4.1).
        let index_of: HashMap<VertexKey, usize> =
            verts.iter().enumerate().map(|(i, &v)| (v, i)).collect();

        let mut stack: SmallVec<[usize; 32]> = SmallVec::new();
        for seed in 0..n {
            if target[seed] != -1 && target[seed] != seed as isize {
                continue;
            }
            // The seed is the survivor of its cluster (§4.4); claim it if it was
            // unclassified (a pre-seeded keep already points at itself).
            if target[seed] == -1 {
                target[seed] = seed as isize;
            }
            let origin = self.vertex_position(verts[seed]);

            stack.clear();
            stack.push(seed);
            while let Some(stepping) = stack.pop() {
                let stepping_vert = verts[stepping];
                self.find_doubles_connected_step(
                    stepping_vert,
                    seed,
                    origin,
                    dist_sq,
                    &index_of,
                    target,
                    &mut stack,
                );
            }
        }
    }

    /// Collect the candidate neighbours of `stepping_vert` (edge step + n-gon
    /// far-corner step, §4.1) and claim each that is an unclaimed input vert
    /// strictly within `dist` of the seed origin.
    #[allow(clippy::too_many_arguments)]
    fn find_doubles_connected_step(
        &self,
        stepping_vert: VertexKey,
        seed: usize,
        origin: Vec3,
        dist_sq: f32,
        index_of: &HashMap<VertexKey, usize>,
        target: &mut [isize],
        stack: &mut SmallVec<[usize; 32]>,
    ) {
        let try_claim =
            |candidate: VertexKey, target: &mut [isize], stack: &mut SmallVec<[usize; 32]>| {
                let Some(&ci) = index_of.get(&candidate) else {
                    return;
                };
                // Already claimed (incl. a pre-seeded keep) — first-claim-wins.
                if target[ci] != -1 {
                    return;
                }
                // §4.1 / §4.2 — strict distance to the seed origin.
                if self.vertex_position(candidate).distance_squared(origin) < dist_sq {
                    target[ci] = seed as isize;
                    stack.push(ci);
                }
            };

        // Edge step (§4.1) — every vert across an incident edge (the disk).
        for edge in self.vert_edge_keys(stepping_vert) {
            if let Some(other) = self.edges.get(edge).other_vertex(stepping_vert) {
                try_claim(other, target, stack);
            }
        }

        // Face-corner step (§4.1) — for each incident face with more than three
        // sides, every corner not edge-adjacent to `stepping_vert`. Triangles
        // add nothing (every other corner is already an edge neighbour).
        for face in self.vert_face_keys(stepping_vert) {
            if self.face_len(face) <= 3 {
                continue;
            }
            let ring = self.face_vert_keys(face);
            let count = ring.len();
            let Some(here) = ring.iter().position(|&v| v == stepping_vert) else {
                continue;
            };
            for offset in 0..count {
                // Skip self and the two edge-adjacent neighbours.
                if offset == here || (here + 1) % count == offset || (offset + 1) % count == here {
                    continue;
                }
                try_claim(ring[offset], target, stack);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use glam::Vec3;

    use crate::loopmesh::*;

    /// A small ring of `n` near-coincident verts around the origin at radius
    /// `r`, plus a far isolated vert. Mirrors the `pointmerge_cluster`-style
    /// fixture the matrix rows lean on: a tight cluster + a far loner.
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

    fn as_set(map: &[(VertexKey, VertexKey)]) -> HashSet<(VertexKey, VertexKey)> {
        map.iter().copied().collect()
    }

    /// §3 — three near-coincident verts cluster into one survivor; the far vert
    /// is neither key nor value. Exactly two arrows, no self-arrow.
    #[test]
    fn spatial_clusters_near_coincident_triple() {
        let mut mesh = LoopMesh::new();
        let (cluster, far) = cluster_plus_far(&mut mesh, 3, 0.05);
        let mut verts = cluster.clone();
        verts.push(far);

        let map = mesh.find_doubles(&verts, &[], 0.5, false);

        assert_eq!(map.len(), 2, "three sources yield two arrows");
        for &(s, t) in &map {
            assert_ne!(s, t, "no self-mapping");
            assert!(
                cluster.contains(&s) && cluster.contains(&t),
                "arrows stay in the cluster"
            );
            assert_ne!(s, far);
            assert_ne!(t, far);
        }
        // All arrows share one survivor value.
        let values: HashSet<VertexKey> = map.iter().map(|&(_, t)| t).collect();
        assert_eq!(values.len(), 1, "one survivor for the whole cluster");
    }

    /// §1 / §7 — `dist` above the spread merges; below the spread the map is
    /// empty (the threshold fires-vs-blocks contrast).
    #[test]
    fn dist_threshold_fires_and_blocks() {
        let mut mesh = LoopMesh::new();
        let (cluster, far) = cluster_plus_far(&mut mesh, 3, 0.05);
        let mut verts = cluster.clone();
        verts.push(far);

        let fired = mesh.find_doubles(&verts, &[], 0.5, false);
        assert!(!fired.is_empty(), "dist above the spread merges");

        // Cluster spread (ring diameter at r=0.05 is ~0.0866); a dist well below
        // that separates every member.
        let blocked = mesh.find_doubles(&verts, &[], 0.001, false);
        assert!(
            blocked.is_empty(),
            "dist below the spread leaves the map empty"
        );
    }

    /// §3.1 vs §4.1 — a vert at exactly `dist` merges in spatial mode
    /// (non-strict `<=`) but not in connected mode (strict `<`).
    #[test]
    fn boundary_convention_is_mode_dependent() {
        let mut mesh = LoopMesh::new();
        let a = mesh.create_vertex(Vec3::new(0.0, 0.0, 0.0));
        let b = mesh.create_vertex(Vec3::new(1.0, 0.0, 0.0));
        let _e = mesh.create_edge([a, b], EdgeCreateFlags::NO_DUPLICATE);
        let verts = [a, b];

        let spatial = mesh.find_doubles(&verts, &[], 1.0, false);
        assert_eq!(spatial.len(), 1, "spatial merges at exactly dist");

        let connected = mesh.find_doubles(&verts, &[], 1.0, true);
        assert!(
            connected.is_empty(),
            "connected does not merge at exactly dist"
        );
    }

    /// §5 — protecting the higher-index member forces it to be the survivor;
    /// with no keep set the lower-index member survives (opposite arrow).
    #[test]
    fn keep_verts_forces_protected_survivor() {
        let mut mesh = LoopMesh::new();
        let a = mesh.create_vertex(Vec3::new(0.0, 0.0, 0.0));
        let b = mesh.create_vertex(Vec3::new(0.1, 0.0, 0.0));
        let verts = [a, b];

        let kept = mesh.find_doubles(&verts, &[b], 0.5, false);
        assert_eq!(kept, vec![(a, b)], "the kept vert is the survivor (a -> b)");

        let no_keep = mesh.find_doubles(&verts, &[], 0.5, false);
        assert_eq!(no_keep, vec![(b, a)], "lower index survives (b -> a)");
    }

    /// §5 / §3.3 — a keep-vert at a *higher* index than the ordinary members
    /// sharing its ball must still win the survivor role. The ordinary verts
    /// seed the cluster (lower indices come first), so the keep-vert is only
    /// reached as a gathered member; the keep-override must pull every ordinary
    /// member onto it rather than onto an ordinary survivor. Regression for the
    /// spatial membership-gathering defect that dropped pre-seeded keep-verts.
    #[test]
    fn keep_vert_wins_when_higher_index_than_ordinary_members() {
        let mut mesh = LoopMesh::new();
        let a = mesh.create_vertex(Vec3::new(0.0, 0.0, 0.0));
        let b = mesh.create_vertex(Vec3::new(0.05, 0.0, 0.0));
        let k = mesh.create_vertex(Vec3::new(0.025, 0.01, 0.0));
        // k is the highest index; a and b would otherwise win the survivor role.
        let verts = [a, b, k];

        let map = mesh.find_doubles(&verts, &[k], 0.5, false);

        assert_eq!(
            as_set(&map),
            [(a, k), (b, k)].into_iter().collect(),
            "both ordinary members map onto the keep-vert (a -> k, b -> k)"
        );
    }

    /// §5.1 — two protected verts in one cluster both survive; an ordinary near
    /// vert attaches to whichever the clustering reaches it through.
    #[test]
    fn two_keep_verts_both_survive() {
        let mut mesh = LoopMesh::new();
        let a = mesh.create_vertex(Vec3::new(0.0, 0.0, 0.0));
        let b = mesh.create_vertex(Vec3::new(0.05, 0.0, 0.0));
        let c = mesh.create_vertex(Vec3::new(0.025, 0.01, 0.0));
        let verts = [a, b, c];

        let map = mesh.find_doubles(&verts, &[a, b], 0.5, false);

        // Neither protected vert is a key; neither maps onto the other.
        for &(s, _t) in &map {
            assert_ne!(s, a, "keep-vert a is never a source");
            assert_ne!(s, b, "keep-vert b is never a source");
        }
        // The ordinary vert c is the only source, mapping onto a keep-vert.
        assert_eq!(map.len(), 1, "only the ordinary vert maps away");
        assert_eq!(map[0].0, c);
        assert!(
            map[0].1 == a || map[0].1 == b,
            "c maps onto a protected survivor"
        );
    }

    /// §4.4 — on a cluster of three positioned so the centroid-closest member
    /// is NOT the lowest index, the survivor is the centroid-closest one.
    #[test]
    fn tiebreak_centroid_closest_on_triple() {
        let mut mesh = LoopMesh::new();
        // Two far-apart verts and one central vert; centroid sits near the
        // central vert (index 2), which is not the lowest index.
        let v0 = mesh.create_vertex(Vec3::new(-0.1, 0.0, 0.0));
        let v1 = mesh.create_vertex(Vec3::new(0.1, 0.0, 0.0));
        let v2 = mesh.create_vertex(Vec3::new(0.0, 0.001, 0.0));
        let verts = [v0, v1, v2];

        let map = mesh.find_doubles(&verts, &[], 0.5, false);

        assert_eq!(map.len(), 2);
        let values: HashSet<VertexKey> = map.iter().map(|&(_, t)| t).collect();
        assert_eq!(
            values,
            [v2].into_iter().collect(),
            "centroid-closest v2 survives"
        );
    }

    /// §4.3 — two verts close in space but sharing no incidence path merge in
    /// spatial mode (proximity ignores connectivity) but not in connected mode
    /// (the connectivity guard). The defining mode divergence.
    #[test]
    fn connected_vs_spatial_diverge_on_disconnected_pair() {
        let mut mesh = LoopMesh::new();
        // Two near-coincident verts with NO edge and NO shared face between
        // them. Each is anchored to its own far edge so neither is fully
        // isolated, but no incidence path links the two.
        let a = mesh.create_vertex(Vec3::new(0.0, 0.0, 0.0));
        let a_anchor = mesh.create_vertex(Vec3::new(0.0, 5.0, 0.0));
        let b = mesh.create_vertex(Vec3::new(0.01, 0.0, 0.0));
        let b_anchor = mesh.create_vertex(Vec3::new(0.01, -5.0, 0.0));
        mesh.create_edge([a, a_anchor], EdgeCreateFlags::NO_DUPLICATE);
        mesh.create_edge([b, b_anchor], EdgeCreateFlags::NO_DUPLICATE);
        let verts = [a, b];

        let spatial = mesh.find_doubles(&verts, &[], 0.5, false);
        let connected = mesh.find_doubles(&verts, &[], 0.5, true);

        assert_eq!(spatial.len(), 1, "spatial merges the coincident pair");
        assert!(
            connected.is_empty(),
            "connected leaves the disconnected pair apart"
        );
        assert_ne!(
            as_set(&spatial),
            as_set(&connected),
            "the two modes diverge"
        );
    }

    /// §4.2 — within connected mode, a chain longer than `dist` end-to-end
    /// breaks into multiple clusters because each candidate is tested against
    /// the seed origin, not the stepping vert.
    #[test]
    fn connected_chain_breaks_at_origin_distance() {
        let mut mesh = LoopMesh::new();
        // Four verts spaced 0.4 apart, edge-linked. Span 1.2 > dist 0.5.
        let chain: Vec<VertexKey> = (0..4)
            .map(|i| mesh.create_vertex(Vec3::new(0.4 * i as f32, 0.0, 0.0)))
            .collect();
        for i in 0..3 {
            mesh.create_edge([chain[i], chain[i + 1]], EdgeCreateFlags::NO_DUPLICATE);
        }

        let connected = mesh.find_doubles(&chain, &[], 0.5, true);

        // Seeded at vert0: only vert1 (0.4) is within 0.5; vert2 (0.8) is out.
        // vert2 then seeds its own cluster claiming vert3. Two clusters.
        assert_eq!(connected.len(), 2, "the chain breaks into two clusters");
        let values: HashSet<VertexKey> = connected.iter().map(|&(_, t)| t).collect();
        assert_eq!(values, [chain[0], chain[2]].into_iter().collect());
    }

    /// §4.1 / §4.3 — the connected flood reaches a quad's far diagonal corner
    /// via the face-corner step even though no edge joins the two corners.
    #[test]
    fn connected_merges_ngon_far_corner() {
        let mut mesh = LoopMesh::new();
        // A quad whose corners 0 and 2 (the diagonal) are near-coincident, and
        // 1, 3 far apart. No edge joins 0 and 2.
        let v0 = mesh.create_vertex(Vec3::new(0.0, 0.0, 0.0));
        let v1 = mesh.create_vertex(Vec3::new(1.0, 0.0, 0.0));
        let v2 = mesh.create_vertex(Vec3::new(0.02, 0.0, 0.0));
        let v3 = mesh.create_vertex(Vec3::new(0.0, 1.0, 0.0));
        let verts = [v0, v1, v2, v3];
        let edges: Vec<EdgeKey> = (0..4)
            .map(|i| {
                mesh.create_edge(
                    [verts[i], verts[(i + 1) % 4]],
                    EdgeCreateFlags::NO_DUPLICATE,
                )
            })
            .collect();
        mesh.face_create(&verts, &edges, FaceCreateFlags::NONE);

        // dist 0.1 reaches across the 0.02 diagonal but not the unit edges.
        let map = mesh.find_doubles(&verts, &[], 0.1, true);

        assert_eq!(
            map.len(),
            1,
            "the far diagonal corner merges via the face-corner step"
        );
        // Seeded at v0, v2 is reached as the non-edge-adjacent far corner.
        assert_eq!(map[0], (v2, v0));
    }

    /// §6 — the chosen survivor is absent as a key, and no arrow is a self-arrow.
    #[test]
    fn survivor_omitted_from_map() {
        let mut mesh = LoopMesh::new();
        let (cluster, _far) = cluster_plus_far(&mut mesh, 3, 0.05);

        let map = mesh.find_doubles(&cluster, &[], 0.5, false);

        let survivor = map[0].1;
        let keys: HashSet<VertexKey> = map.iter().map(|&(s, _)| s).collect();
        assert!(!keys.contains(&survivor), "the survivor is never a key");
        for &(s, t) in &map {
            assert_ne!(s, t, "no v -> v self-arrow");
        }
    }

    /// §6 / §7 — all verts separated by more than `dist` yields an empty map.
    #[test]
    fn all_separated_is_empty() {
        let mut mesh = LoopMesh::new();
        let verts: Vec<VertexKey> = (0..4)
            .map(|i| mesh.create_vertex(Vec3::new(i as f32, 0.0, 0.0)))
            .collect();

        let map = mesh.find_doubles(&verts, &[], 0.5, false);
        assert!(map.is_empty(), "well-separated verts produce no arrows");
    }

    /// §4.3 — an isolated input vert (no edge, no face) merges with nothing in
    /// connected mode even when a coincident neighbour exists.
    #[test]
    fn connected_isolated_vert_does_not_merge() {
        let mut mesh = LoopMesh::new();
        let a = mesh.create_vertex(Vec3::new(0.0, 0.0, 0.0));
        let b = mesh.create_vertex(Vec3::new(0.001, 0.0, 0.0));
        let verts = [a, b];

        // Spatial: coincident pair merges.
        assert_eq!(mesh.find_doubles(&verts, &[], 0.5, false).len(), 1);
        // Connected: no incidence path, so nothing merges.
        assert!(mesh.find_doubles(&verts, &[], 0.5, true).is_empty());
    }

    /// §7 / §11 — find_doubles never mutates topology; the mesh is unchanged.
    #[test]
    fn find_doubles_does_not_mutate_topology() {
        let mut mesh = LoopMesh::new();
        let (cluster, far) = cluster_plus_far(&mut mesh, 3, 0.05);
        let mut verts = cluster.clone();
        verts.push(far);
        let before = (mesh.vertex_count(), mesh.edge_count(), mesh.face_count());

        let _ = mesh.find_doubles(&verts, &[], 0.5, false);

        assert_eq!(
            (mesh.vertex_count(), mesh.edge_count(), mesh.face_count()),
            before
        );
        mesh.validate_topology();
    }
}
