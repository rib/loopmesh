use std::collections::{HashMap, HashSet};

use crate::loopmesh::ops::{DeleteContext, MixedSeed};
use crate::loopmesh::*;

/// The output of [`LoopMesh::split`] (SPLIT.md §1).
///
/// Split is a thin composition — duplicate the selection's closure, then delete
/// the originals with the **faces** context — so its outputs are a reduced
/// subset of [`DuplicateResult`]: the clone elements, the (key-fixed) frontier
/// edge correspondence, and the isolated-vert correspondence. The full
/// bidirectional per-kind maps are not re-exposed (SPLIT.md §1); split reads
/// duplicate's `face_map` once internally to keep the active face on a survivor
/// (§3.2) and otherwise forwards only what a stitching consumer needs.
#[derive(Debug, Default)]
pub struct SplitResult {
    /// The clone elements — the new disjoint copy left standing where the
    /// selection used to be (SPLIT.md §1, copied verbatim from duplicate's
    /// `geom.out`).
    pub geom_out: Vec<GeomKey>,

    /// Frontier-edge correspondence with the §5 key fix-up applied. A frontier
    /// edge whose **source** survived (shared with an unselected face) is keyed
    /// `source -> clone`; a frontier edge whose source was deleted (solely owned
    /// by the selection) is a self-map `clone -> clone`, since the source key no
    /// longer exists.
    pub boundary_map: HashMap<EdgeKey, EdgeKey>,

    /// Isolated-vert correspondence, copied **verbatim** from duplicate with no
    /// §5 key fix-up. Each couple is `source -> clone`; under `use_only_faces`
    /// the source is deleted, so its key is a freed slot — the consumer reaches
    /// the survivor through the value (SPLIT.md §5, the isovert asymmetry).
    pub isovert_map: HashMap<VertexKey, VertexKey>,
}

impl LoopMesh {
    /// Tear a mixed selection loose, replacing it in place with a topologically
    /// disjoint, coincident copy of itself.
    ///
    /// Split is the composition `duplicate(geom)` then `delete(originals,
    /// `DeleteContext::Faces`)`: the closure is cloned (sharing nothing with the
    /// originals), then the originals are removed under the faces context so the
    /// clone is left standing, now parting from the surrounding mesh along a seam
    /// wherever the selection met unselected geometry.
    ///
    /// `use_only_faces` controls the selection's **loose** parts (wire edges with
    /// no face, loose verts with no edge). When clear (the default) the
    /// face-seeded delete ignores them, so a selected loose original is left
    /// behind doubled-up alongside its clone. When set, the original-removal is
    /// widened to sweep the selection's solely-owned loose geometry too, so only
    /// the clone survives (SPLIT.md §4).
    ///
    /// The returned [`SplitResult`] carries the clone elements, the frontier
    /// correspondence with the §5 boundary-map key fix-up applied, and the
    /// isolated-vert correspondence forwarded verbatim.
    ///
    /// For an isolated selection (no boundary shared with unselected geometry)
    /// split is a topological no-op: duplicate makes a coincident copy and the
    /// delete removes the original, leaving one copy in the same place with fresh
    /// element identities (SPLIT.md §2).
    ///
    /// See `radial-mesh-guides/operators/SPLIT.md` for the full composition and
    /// parity contract.
    pub fn split(&mut self, geom: &[GeomKey], use_only_faces: bool) -> SplitResult {
        // The non-face parts of the input `geom`, captured before the clone runs
        // so they name the *originals*. SPLIT.md §3.3: split forwards the entire
        // input `geom` (faces *and* its directly-named loose verts / wire edges)
        // into the removal, not just the faces. The named loose edges and verts
        // are torn out alongside the face closure; the faces-context keep-alive
        // then spares a torn wire edge's endpoints (they become loose).
        let mut named_edges: Vec<EdgeKey> = Vec::new();
        let mut named_verts: Vec<VertexKey> = Vec::new();
        for g in geom {
            match *g {
                GeomKey::Edge(e) => named_edges.push(e),
                GeomKey::Vert(v) => named_verts.push(v),
                GeomKey::Face(_) => {}
            }
        }

        // Step 1 (§3.1) — clone the closure. Nothing is deleted yet; the mesh now
        // holds both the originals and a coincident clone.
        let dupe = self.duplicate(geom, false);

        // The clone elements, so the source-key recovery below can tell an
        // original apart from its fresh clone (the per-kind maps are
        // bidirectional and carry both).
        let (clone_verts, clone_edges, clone_faces) = partition_clones(&dupe.geom_out);

        // The closure originals: every per-kind map key that is not itself a
        // clone (SPLIT.md §1 — duplicate computes these maps but split does not
        // re-expose them).
        let source_faces: Vec<FaceKey> = dupe
            .face_map
            .keys()
            .copied()
            .filter(|f| !clone_faces.contains(f))
            .collect();
        let source_edges: Vec<EdgeKey> = dupe
            .edge_map
            .keys()
            .copied()
            .filter(|e| !clone_edges.contains(e))
            .collect();

        let selected_face: HashSet<FaceKey> = source_faces.iter().copied().collect();

        // §5 fix-up authority: exactly which *frontier* source edges the
        // faces-context removal frees, computed while the originals are still
        // live. A frontier (boundary_map) edge is freed iff it is solely owned by
        // the selection — every face on its live radial cycle is a selected face.
        // (Boundary_map only ever carries face-frontier edges, so this face-based
        // test is the relevant one for the key fix-up; the named loose wire edges
        // never appear as keys.)
        let mut deleted_frontier_edges: HashSet<EdgeKey> = HashSet::new();
        for &edge in &source_edges {
            let mut has_selected_face = false;
            let mut has_outsider_face = false;
            for loop_key in self.iter_radial_loops(edge) {
                if selected_face.contains(&self.loop_face(loop_key)) {
                    has_selected_face = true;
                } else {
                    has_outsider_face = true;
                }
            }
            if has_selected_face && !has_outsider_face {
                deleted_frontier_edges.insert(edge);
            }
        }

        // §5 boundary-map fix-up — run BEFORE any source edge is freed, so the
        // source keys are still live to test. A surviving-source frontier edge
        // keeps the ordinary `source -> clone` couple; a deleted-source frontier
        // edge becomes the self-map `clone -> clone`.
        let mut boundary_map = HashMap::with_capacity(dupe.boundary_map.len());
        for (&source_edge, &clone_edge) in dupe.boundary_map.iter() {
            let key = if deleted_frontier_edges.contains(&source_edge) {
                clone_edge
            } else {
                source_edge
            };
            boundary_map.insert(key, clone_edge);
        }

        // Step 3 (§3.3) — delete the originals with the faces context.
        if use_only_faces {
            self.split_delete_widened(&selected_face);
        } else {
            self.split_delete_faces_and_named_loose(&source_faces, &named_edges, &named_verts);
        }

        SplitResult {
            geom_out: dupe.geom_out,
            boundary_map,
            isovert_map: dupe.isovert_map,
        }
    }

    /// Tear a mixed selection loose **into a separate destination mesh**: the
    /// cross-mesh sibling of [`split`](Self::split) (SPLIT.md §8).
    ///
    /// The `split` operation declares a `dest` slot but never reads it, so a
    /// genuine cross-mesh tear is a **consumer-built composition** rather than a
    /// `dest`-carrying `split` call (SPLIT.md §8 intro). This method builds that
    /// composition from two already-parity-correct primitives:
    ///
    /// 1. **Step 1 (§8.1 step 1)** — [`duplicate_into`](Self::duplicate_into)
    ///    clones the selection's closure into `dest`. The clone is born in
    ///    `dest`; `self` is left untouched, so the source mesh holds *no*
    ///    coincident clone — the headline cross-mesh simplification (§8).
    /// 2. **Step 2 (§8.1 step 2)** — the same-mesh active-face re-point does not
    ///    apply cross-mesh (a mesh's active face must be one of its own faces, and
    ///    the clone now lives in `dest`), so it is skipped.
    /// 3. **Step 3 (§8.1 step 3)** — the originals are removed from `self` under
    ///    the **faces** context, exactly as the in-place [`split`](Self::split)
    ///    runs its source removal. Because the source holds only originals (no
    ///    clone to spare), this is plain faces-context delete; the same
    ///    `use_only_faces` widening (§4) and named-loose sweep (§3.3) apply, and
    ///    cross-mesh the §8.3 widening sweeps the **source** only — the clone's
    ///    loose parts stay standing in `dest`, since the source-side passes never
    ///    reach them.
    ///
    /// **Output slots cross-mesh (SPLIT.md §8.2).** `geom_out` comes back
    /// **empty** — the clone lives in `dest`, not in `self`, so the source-side
    /// new-element scan finds nothing (mirroring [`duplicate_into`](Self::duplicate_into)).
    /// A cross-mesh consumer reads the clone through `dest` or the correspondence
    /// maps. The `boundary_map` / `isovert_map` slots are forwarded **forward-only**
    /// (`source → clone`, key in `self`, value in `dest`) exactly as
    /// `duplicate_into` produces them: the §5 boundary-map key fix-up (rewriting a
    /// deleted-source frontier edge to a `clone → clone` self-map) is **deferred**
    /// for this landing. Per the §8.2 handle-uniqueness caveat an index-keyed
    /// backend cannot safely hold the self-map branch (a `clone → clone` couple
    /// living entirely in `dest` collides with a surviving-source `source → clone`
    /// couple at equal indices), so the cross-mesh maps stay forward-only here. A
    /// solely-owned frontier edge's source key therefore dangles after the tear,
    /// just as the isovert map's deleted-source key does (§5); the consumer
    /// reaches the surviving clone through the couple **value**.
    ///
    /// See `radial-mesh-guides/operators/SPLIT.md` §8 for the cross-mesh
    /// composition and its per-mesh element-count deltas (§8.4).
    pub fn split_into(
        &mut self,
        geom: &[GeomKey],
        use_only_faces: bool,
        dest: &mut LoopMesh,
    ) -> SplitResult {
        // The named face / loose parts of the input `geom`, in `self`. A face's
        // closure adds only edges and verts, never another face, so the named
        // faces *are* the selected-face set the faces-context delete cascades
        // from (SPLIT.md §3.3, applied source-side per §8.1 step 3).
        let mut named_faces: Vec<FaceKey> = Vec::new();
        let mut named_edges: Vec<EdgeKey> = Vec::new();
        let mut named_verts: Vec<VertexKey> = Vec::new();
        for g in geom {
            match *g {
                GeomKey::Face(f) => named_faces.push(f),
                GeomKey::Edge(e) => named_edges.push(e),
                GeomKey::Vert(v) => named_verts.push(v),
            }
        }

        // Step 1 (§8.1 step 1) — clone the closure into `dest`. `self` is left
        // untouched; nothing of the clone enters the source mesh.
        let dupe = self.duplicate_into(geom, false, dest);

        // Step 2 (§8.1 step 2) — active-face re-point does not apply cross-mesh.

        // Step 3 (§8.1 step 3) — remove the originals from `self` with the faces
        // context. The source holds only originals, so this is the in-place
        // split's source removal verbatim; the §8.3 widening sweeps `self` only.
        if use_only_faces {
            let selected_face: HashSet<FaceKey> = named_faces.iter().copied().collect();
            self.split_delete_widened(&selected_face);
        } else {
            self.split_delete_faces_and_named_loose(&named_faces, &named_edges, &named_verts);
        }

        SplitResult {
            // §8.2: empty cross-mesh — the clone lives in `dest`.
            geom_out: dupe.geom_out,
            // §8.2: forward-only `source → clone`, no §5 cross-mesh fix-up (deferred).
            boundary_map: dupe.boundary_map,
            isovert_map: dupe.isovert_map,
        }
    }

    /// Default-path removal (`use_only_faces = false`, SPLIT.md §3.3): tear out
    /// the face closure with the faces context, then sweep the input `geom`'s
    /// directly-named loose parts.
    ///
    /// The named loose **wire edges** (those that survive the faces delete still
    /// holding no face) are removed with the loose-endpoint sweep skipped, so the
    /// faces-context keep-alive is honoured: the wire's endpoints are left loose
    /// rather than swept. The named loose **verts** that remain loose afterwards
    /// are then removed. A named edge still on a face (shared boundary the
    /// keep-alive kept) or a named vert still held by a surviving edge is left
    /// standing — only the selection's genuinely loose extremities are torn.
    fn split_delete_faces_and_named_loose(
        &mut self,
        source_faces: &[FaceKey],
        named_edges: &[EdgeKey],
        named_verts: &[VertexKey],
    ) {
        self.delete(source_faces, DeleteContext::Faces);

        let loose_named_edges: Vec<EdgeKey> = named_edges
            .iter()
            .copied()
            .filter(|&e| {
                self.edges.try_get(e).is_some() && self.iter_radial_loops(e).next().is_none()
            })
            .collect();
        if !loose_named_edges.is_empty() {
            // EdgesFaces (loose-endpoint sweep skipped) reproduces the §3.3
            // keep-alive for a torn wire: the endpoints survive as loose points.
            self.delete(&loose_named_edges, DeleteContext::EdgesFaces);
        }

        let loose_named_verts: Vec<VertexKey> = named_verts
            .iter()
            .copied()
            .filter(|&v| {
                self.vertices.try_get(v).is_some() && self.iter_disk_edges(v).next().is_none()
            })
            .collect();
        if !loose_named_verts.is_empty() {
            self.delete(&loose_named_verts, DeleteContext::Verts);
        }
    }

    /// `use_only_faces = true` removal (SPLIT.md §4): widen the kill set over the
    /// **whole mesh** — which, after the clone has run, also contains the freshly
    /// made clone loose parts — then tear out the whole widened set.
    ///
    /// The §4 widening is two solely-owned passes over every element, not just
    /// the source originals: an edge whose every face is flagged (or a wire edge,
    /// with no face, vacuously) and then a vert whose every edge is flagged (or a
    /// loose vert, vacuously). A clone wire edge / clone loose vert has no face /
    /// no edge holding it, so it passes vacuously and is swept exactly as BMesh
    /// sweeps it — the selection's loose parts vanish entirely (original *and*
    /// clone), leaving only the face-bounded clone.
    fn split_delete_widened(&mut self, selected_face: &HashSet<FaceKey>) {
        // (a) Every edge all of whose faces are flagged — wire edges (no face)
        //     pass vacuously. An edge a non-flagged face still holds is spared
        //     (the faces-context shared-boundary keep-alive, unchanged).
        let mut kill_edges: HashSet<EdgeKey> = HashSet::new();
        let mut edge_iter = MeshEdges::default();
        while let Some(edge) = edge_iter.next(self) {
            let held_by_outsider = self
                .iter_radial_loops(edge)
                .any(|loop_key| !selected_face.contains(&self.loop_face(loop_key)));
            if !held_by_outsider {
                kill_edges.insert(edge);
            }
        }

        // (b) Every vert all of whose edges are now flagged — loose verts (no
        //     edge) pass vacuously.
        let mut kill_verts: HashSet<VertexKey> = HashSet::new();
        let mut vert_iter = MeshVerts::default();
        while let Some(vert) = vert_iter.next(self) {
            let held_by_outsider = self
                .iter_disk_edges(vert)
                .any(|edge| !kill_edges.contains(&edge));
            if !held_by_outsider {
                kill_verts.insert(vert);
            }
        }

        let kill_faces: Vec<FaceKey> = selected_face.iter().copied().collect();
        let kill_edges: Vec<EdgeKey> = kill_edges.into_iter().collect();
        let kill_verts: Vec<VertexKey> = kill_verts.into_iter().collect();
        self.delete(
            MixedSeed {
                verts: &kill_verts,
                edges: &kill_edges,
                faces: &kill_faces,
            },
            DeleteContext::TaggedOnly,
        );
    }
}

/// Split `geom_out` into its three per-kind clone-key sets.
fn partition_clones(
    geom_out: &[GeomKey],
) -> (HashSet<VertexKey>, HashSet<EdgeKey>, HashSet<FaceKey>) {
    let mut verts = HashSet::new();
    let mut edges = HashSet::new();
    let mut faces = HashSet::new();
    for g in geom_out {
        match *g {
            GeomKey::Vert(v) => {
                verts.insert(v);
            }
            GeomKey::Edge(e) => {
                edges.insert(e);
            }
            GeomKey::Face(f) => {
                faces.insert(f);
            }
        }
    }
    (verts, edges, faces)
}

#[cfg(test)]
mod tests {
    use glam::Vec3;

    use crate::loopmesh::*;

    /// Build an `n × n` grid of quad faces (so `(n+1) × (n+1)` verts). Faces are
    /// returned row-major: face `(r, c)` is at index `r * n + c`.
    fn build_quad_grid(mesh: &mut LoopMesh, n: usize) -> Vec<FaceKey> {
        let side = n + 1;
        let mut verts = Vec::with_capacity(side * side);
        for r in 0..side {
            for c in 0..side {
                verts.push(mesh.create_vertex(Vec3::new(c as f32, r as f32, 0.0)));
            }
        }
        let vert_at = |r: usize, c: usize| verts[r * side + c];

        let mut faces = Vec::with_capacity(n * n);
        for r in 0..n {
            for c in 0..n {
                let ring = [
                    vert_at(r, c),
                    vert_at(r, c + 1),
                    vert_at(r + 1, c + 1),
                    vert_at(r + 1, c),
                ];
                let mut edges = [EdgeKey::NULL; 4];
                for i in 0..4 {
                    let a = ring[i];
                    let b = ring[(i + 1) % 4];
                    edges[i] = mesh
                        .query_edge_exists(a, b)
                        .unwrap_or_else(|| mesh.create_edge([a, b], EdgeCreateFlags::NONE));
                }
                faces.push(mesh.face_create(&ring, &edges, FaceCreateFlags::NONE));
            }
        }
        faces
    }

    fn unit_quad(mesh: &mut LoopMesh) -> FaceKey {
        build_quad_grid(mesh, 1)[0]
    }

    /// A lone quad (face 0) plus a disjoint wire edge {4,5} hanging in space and
    /// a loose vert (index 6) — the §4 contrast fixture. Returns
    /// `(face, wire_edge, loose_vert)`.
    fn face_with_loose_edge_and_vert(mesh: &mut LoopMesh) -> (FaceKey, EdgeKey, VertexKey) {
        let face = unit_quad(mesh);
        let a = mesh.create_vertex(Vec3::new(3.0, 0.0, 0.0));
        let b = mesh.create_vertex(Vec3::new(4.0, 0.0, 0.0));
        let wire = mesh.create_edge([a, b], EdgeCreateFlags::NONE);
        let loose = mesh.create_vertex(Vec3::new(6.0, 6.0, 0.0));
        (face, wire, loose)
    }

    #[test]
    fn isolated_quad_is_topological_noop() {
        // §2 / §6: a lone quad (the whole mesh) is fully isolated — every
        // boundary edge and vert is solely owned. Split clones it and deletes the
        // original, leaving exactly one quad. Net counts unchanged (+0/+0/+0).
        let mut mesh = LoopMesh::new();
        let face = unit_quad(&mut mesh);
        let (v0, e0, f0) = (mesh.vertex_count(), mesh.edge_count(), mesh.face_count());

        let result = mesh.split(&[face.into()], false);

        assert_eq!(mesh.vertex_count(), v0, "no net vert change");
        assert_eq!(mesh.edge_count(), e0, "no net edge change");
        assert_eq!(mesh.face_count(), f0, "no net face change");
        // The original face is gone; the survivor is the clone.
        assert!(mesh.faces.try_get(face).is_none(), "original face torn out");
        let clone_faces: Vec<FaceKey> = result
            .geom_out
            .iter()
            .filter_map(|g| match g {
                GeomKey::Face(f) => Some(*f),
                _ => None,
            })
            .collect();
        assert_eq!(clone_faces.len(), 1);
        assert!(
            mesh.faces.try_get(clone_faces[0]).is_some(),
            "clone survives"
        );

        // Every frontier edge was solely owned and deleted -> all self-maps.
        assert_eq!(result.boundary_map.len(), 4);
        for (&k, &v) in result.boundary_map.iter() {
            assert_eq!(k, v, "solely-owned frontier -> clone->clone self-map (§5)");
        }
        mesh.validate_topology();
    }

    #[test]
    fn interior_tear_keeps_face_count_adds_boundary() {
        // §6: a single quad torn from the interior of a 3×3 grid. All four
        // perimeter edges are shared with surrounding faces, so none is solely
        // owned: net +4 V, +4 E, +0 F. The four boundary_map entries are all
        // surviving-source `source -> clone` couples (no self-maps).
        let mut mesh = LoopMesh::new();
        let faces = build_quad_grid(&mut mesh, 3);
        let (v0, e0, f0) = (mesh.vertex_count(), mesh.edge_count(), mesh.face_count());

        let centre = faces[1 * 3 + 1];
        let result = mesh.split(&[centre.into()], false);

        assert_eq!(mesh.vertex_count() - v0, 4, "+4 verts (clone corners)");
        assert_eq!(mesh.edge_count() - e0, 4, "+4 edges (clone boundary)");
        assert_eq!(
            mesh.face_count(),
            f0,
            "face count unchanged (clone offsets original)"
        );
        assert!(
            mesh.faces.try_get(centre).is_none(),
            "original centre torn out"
        );

        assert_eq!(result.boundary_map.len(), 4);
        for (&k, &v) in result.boundary_map.iter() {
            assert_ne!(k, v, "shared frontier survives -> source->clone (§5)");
            assert!(
                mesh.edges.try_get(k).is_some(),
                "surviving source key is live"
            );
            assert!(mesh.edges.try_get(v).is_some(), "clone value is live");
        }
        mesh.validate_topology();
    }

    #[test]
    fn boundary_patch_self_maps_owned_frontier() {
        // §5 deleted-source branch: a corner quad of the grid (face 0) has two
        // perimeter edges on the mesh boundary (solely owned -> deleted) and two
        // shared with neighbours (kept). The owned pair self-maps `clone->clone`;
        // the shared pair stays `source->clone`.
        let mut mesh = LoopMesh::new();
        let faces = build_quad_grid(&mut mesh, 3);
        let corner = faces[0];

        let result = mesh.split(&[corner.into()], false);

        assert_eq!(
            result.boundary_map.len(),
            4,
            "all four perimeter edges frontier"
        );
        let self_maps = result.boundary_map.iter().filter(|(k, v)| k == v).count();
        let cross_maps = result.boundary_map.iter().filter(|(k, v)| k != v).count();
        assert_eq!(
            self_maps, 2,
            "two solely-owned boundary edges self-map (§5)"
        );
        assert_eq!(cross_maps, 2, "two shared edges keep source->clone (§5)");
        mesh.validate_topology();
    }

    #[test]
    fn use_only_faces_drops_loose_originals() {
        // §4 fires over the WHOLE mesh: a face + a disjoint wire edge + a loose
        // vert. With the flag set, the widening sweeps not only the loose
        // *originals* but — because it runs after the clone — the freshly made
        // clone loose parts too (a clone wire edge / clone loose vert is vacuously
        // solely owned: no face / no edge holds it). So the selection's loose
        // parts vanish entirely (original and clone), leaving only the
        // face-bounded clone (a single quad).
        let mut mesh = LoopMesh::new();
        let (face, wire, loose) = face_with_loose_edge_and_vert(&mut mesh);
        let [wa, wb] = mesh.edge_vertices(wire);

        mesh.split(&[face.into(), wire.into(), loose.into()], true);

        // All loose originals torn out.
        assert!(
            mesh.edges.try_get(wire).is_none(),
            "wire original swept (§4)"
        );
        assert!(
            mesh.vertices.try_get(wa).is_none(),
            "wire endpoint swept (§4)"
        );
        assert!(
            mesh.vertices.try_get(wb).is_none(),
            "wire endpoint swept (§4)"
        );
        assert!(
            mesh.vertices.try_get(loose).is_none(),
            "loose vert swept (§4)"
        );
        assert!(mesh.faces.try_get(face).is_none(), "original face torn out");

        // The whole-mesh widening also sweeps the clone wire (2V/1E) and clone
        // loose vert (1V), so only the face-bounded clone survives: a single quad,
        // 4 V / 4 E / 1 F.
        assert_eq!(
            mesh.vertex_count(),
            4,
            "only the face-bounded clone survives"
        );
        assert_eq!(mesh.edge_count(), 4);
        assert_eq!(mesh.face_count(), 1);
        mesh.validate_topology();
    }

    #[test]
    fn use_only_faces_off_keeps_loose_originals() {
        // §3.3 default: the entire input `geom` is forwarded into the faces
        // delete. The face closure tears, the named wire edge {4,5} and named
        // loose vert 6 are removed, but the faces-context keep-alive spares the
        // wire's endpoints — they become loose verts. No clone loose part is
        // swept (the §4 whole-mesh widening only runs under use_only_faces).
        let mut mesh = LoopMesh::new();
        let (face, wire, loose) = face_with_loose_edge_and_vert(&mut mesh);
        let [wa, wb] = mesh.edge_vertices(wire);

        mesh.split(&[face.into(), wire.into(), loose.into()], false);

        // Named loose originals torn out, but the wire's endpoints survive loose.
        assert!(
            mesh.edges.try_get(wire).is_none(),
            "named wire original torn (§3.3)"
        );
        assert!(
            mesh.vertices.try_get(wa).is_some(),
            "wire endpoint spared loose (§3.3)"
        );
        assert!(
            mesh.vertices.try_get(wb).is_some(),
            "wire endpoint spared loose (§3.3)"
        );
        assert!(
            mesh.vertices.try_get(loose).is_none(),
            "named loose vert torn (§3.3)"
        );
        assert!(mesh.faces.try_get(face).is_none(), "original face torn out");

        // Quad clone (4V/4E/1F) + wire clone (2V/1E) + clone loose vert (1V) +
        // the two spared original wire endpoints (now loose) = 4+2+1+2 = 9 V,
        // 4+1 = 5 E, 1 F. Net vs the pre-split mesh (7V/5E/1F): +2 V / +0 E / +0 F.
        assert_eq!(
            mesh.vertex_count(),
            9,
            "two spared wire endpoints survive loose"
        );
        assert_eq!(mesh.edge_count(), 5);
        assert_eq!(mesh.face_count(), 1);
        mesh.validate_topology();
    }

    #[test]
    fn isovert_map_carries_through() {
        // §5 isovert carry-through + §4 whole-mesh widening: a single loose vert
        // under use_only_faces. The isovert_map still records `source -> clone`
        // verbatim (no key fix-up — key stays the now-freed source). But the §4
        // widening runs over the whole mesh *after* the clone, so BOTH the
        // original loose vert AND the freshly made clone loose vert are vacuously
        // solely owned and swept — the mesh ends empty.
        let mut mesh = LoopMesh::new();
        let v = mesh.create_vertex(Vec3::new(0.5, 0.5, 0.0));

        let result = mesh.split(&[v.into()], true);

        assert_eq!(result.isovert_map.len(), 1, "the lone vert is an isovert");
        let (&src, &clone) = result.isovert_map.iter().next().unwrap();
        assert_eq!(src, v, "key is the verbatim source (no §5 fix-up)");
        assert_ne!(src, clone, "isovert couple is never a self-map");
        assert!(
            mesh.vertices.try_get(v).is_none(),
            "original swept under use_only_faces"
        );
        assert!(
            mesh.vertices.try_get(clone).is_none(),
            "clone loose vert swept too (§4 whole-mesh)"
        );
        assert_eq!(
            mesh.vertex_count(),
            0,
            "both original and clone loose verts swept"
        );
        mesh.validate_topology();
    }

    // ---- Cross-mesh tear (`split_into`, SPLIT.md §8) ----

    #[test]
    fn split_into_isolated_quad_moves_to_dest() {
        // §8.4 clean move: an isolated quad (the whole source) torn into an empty
        // dest. The clone is born in dest (+4/+4/+1); the source-side faces delete
        // removes the whole isolated original (every edge/vert solely owned), so
        // the source empties (−4/−4/−1). Cross-mesh this is not the same-mesh
        // no-op but a relocation (§8.4).
        let mut src = LoopMesh::new();
        let face = unit_quad(&mut src);
        let mut dest = LoopMesh::new();

        let result = src.split_into(&[face.into()], false, &mut dest);

        // Source emptied — nothing of the isolated selection survives.
        assert_eq!(
            (src.vertex_count(), src.edge_count(), src.face_count()),
            (0, 0, 0),
            "isolated source torn out entirely (§8.4 move)"
        );
        // Dest holds the full clone quad.
        assert_eq!(
            (dest.vertex_count(), dest.edge_count(), dest.face_count()),
            (4, 4, 1),
            "dest gains the whole clone (+Fc/+Ec/+Vc, §8.4)"
        );

        // §8.2: geom_out empty cross-mesh (the clone lives in dest).
        assert!(result.geom_out.is_empty(), "geom.out empty cross-mesh (§8.2)");
        // Forward-only boundary map: every perimeter edge is a frontier, keyed by
        // the (now-deleted) source edge, valued at the live dest clone.
        assert_eq!(result.boundary_map.len(), 4, "four frontier edges (§8.2)");
        for &clone_edge in result.boundary_map.values() {
            assert!(
                dest.edges.try_get(clone_edge).is_some(),
                "boundary_map value is a live dest clone edge"
            );
        }

        src.validate_topology();
        dest.validate_topology();
    }

    #[test]
    fn split_into_interior_quad_keeps_seam_clones_to_dest() {
        // §8.4 interior tear: the centre face of a 3×3 grid torn into an empty
        // dest. The four perimeter edges are shared with surrounding faces, so the
        // source-side faces delete keeps them (the §3.3 keep-alive) and only the
        // centre face is removed: source delta −0/−0/−1. The clone lands whole in
        // dest: +4/+4/+1.
        let mut src = LoopMesh::new();
        let faces = build_quad_grid(&mut src, 3);
        let (v0, e0, f0) = (src.vertex_count(), src.edge_count(), src.face_count());
        let centre = faces[1 * 3 + 1];
        let mut dest = LoopMesh::new();

        let result = src.split_into(&[centre.into()], false, &mut dest);

        // Source keeps the shared seam — only the face count drops.
        assert_eq!(src.vertex_count(), v0, "shared seam verts kept (−0 V, §8.4)");
        assert_eq!(src.edge_count(), e0, "shared seam edges kept (−0 E, §8.4)");
        assert_eq!(src.face_count(), f0 - 1, "only the centre face removed (−1 F, §8.4)");
        assert!(
            src.faces.try_get(centre).is_none(),
            "centre original torn out of the source"
        );
        // Dest gains the full clone quad.
        assert_eq!(
            (dest.vertex_count(), dest.edge_count(), dest.face_count()),
            (4, 4, 1),
            "dest gains the whole clone (+4/+4/+1, §8.4)"
        );

        // §8.2: geom_out empty; forward-only frontier map spans the two meshes
        // (source key survives the keep-alive, dest value is the clone).
        assert!(result.geom_out.is_empty(), "geom.out empty cross-mesh (§8.2)");
        assert_eq!(result.boundary_map.len(), 4, "four shared frontier edges");
        for (&source_edge, &clone_edge) in result.boundary_map.iter() {
            assert!(
                src.edges.try_get(source_edge).is_some(),
                "shared source key survives the keep-alive in src"
            );
            assert!(
                dest.edges.try_get(clone_edge).is_some(),
                "frontier clone lives in dest"
            );
        }

        src.validate_topology();
        dest.validate_topology();
    }
}
