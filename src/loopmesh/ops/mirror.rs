use glam::{Mat4, Vec3};

use crate::loopmesh::*;

/// Which mirror-space coordinate the reflection negates — equivalently, which
/// plane (perpendicular to that axis) the geometry reflects across. The seam
/// test reads the *same* axis coordinate (MIRROR.md §1, §4).
#[derive(PartialEq, Eq, Clone, Copy, Debug, Hash)]
pub enum Axis {
    X,
    Y,
    Z,
}

impl Axis {
    /// The component index this axis selects from a `Vec3`.
    fn index(self) -> usize {
        match self {
            Axis::X => 0,
            Axis::Y => 1,
            Axis::Z => 2,
        }
    }

    /// The unit scale that negates exactly this axis — the determinant-−1
    /// reflection of MIRROR.md §3.
    fn reflect_scale(self) -> Vec3 {
        let mut scale = Vec3::ONE;
        scale[self.index()] = -1.0;
        scale
    }
}

/// Parameters for [`LoopMesh::mirror`] (MIRROR.md §1).
#[derive(Clone, Copy, Debug)]
pub struct MirrorParams {
    /// The frame the reflection happens in. With the identity matrix the mirror
    /// plane is the coordinate plane through the origin perpendicular to `axis`.
    pub matrix: Mat4,
    /// The seam-weld tolerance. A clone welds onto its original iff the
    /// original's mirror-space `axis` coordinate is within `merge_dist` of zero
    /// (non-strict `<=`, §4).
    pub merge_dist: f32,
    /// Which mirror-space coordinate the reflection negates.
    pub axis: Axis,
    /// Mirror the cloned face loops' UV `u` coordinate. Accepted for interface
    /// stability; the non-default (`true`) UV-flip pass (§5) is a by-design
    /// deferral and is not yet implemented.
    pub mirror_u: bool,
    /// As `mirror_u` for the UV `v` coordinate. By-design deferral.
    pub mirror_v: bool,
    /// Modifies the UV reflection centre (in-tile vs unit square). By-design
    /// deferral; only meaningful alongside `mirror_u` / `mirror_v`.
    pub mirror_udim: bool,
}

/// Result of [`LoopMesh::mirror`]: the surviving mirrored clone — the reflected
/// half minus the seam clones that welded back onto their originals (MIRROR.md
/// §1 `geom.out`, §6).
#[derive(Debug, Default)]
pub struct MirrorResult {
    /// The new, post-weld geometry: every cloned vert, edge, and face that
    /// survived the seam weld. The originals are not in this buffer.
    pub geom_out: Vec<GeomKey>,
}

impl LoopMesh {
    /// Reflect a selection across a plane and weld the result back at the seam.
    ///
    /// A thin composition (MIRROR.md §6): duplicate the selection's closure,
    /// reflect the clone across the mirror plane (a determinant-−1 transform
    /// that flips each clone face's winding, §3), build the seam weld map (each
    /// clone whose *original* lies within `merge_dist` of the plane welds onto
    /// that original, §4), then weld.
    ///
    /// The seam test ranges over the verts **named directly in `geom`**, not the
    /// closure verts the duplicate pulled in (MIRROR.md §4): a face-only
    /// selection welds nothing. Callers that want a welded seam must name the
    /// seam verts (in practice every selection vert) in `geom` alongside the
    /// faces.
    ///
    /// `mirror_u` / `mirror_v` / `mirror_udim` are accepted for interface
    /// stability; their non-default (`true`) UV-flip pass (§5) is a by-design
    /// deferral and is not yet implemented.
    ///
    /// See `radial-mesh-guides/operators/MIRROR.md` for the full algorithm and
    /// parity contract.
    pub fn mirror(&mut self, geom: &[GeomKey], params: MirrorParams) -> MirrorResult {
        debug_assert!(
            !(params.mirror_u || params.mirror_v),
            "mirror: mirror_u/mirror_v UV flip is a by-design deferral (MIRROR.md §5)"
        );
        let _ = params.mirror_udim;

        // Step 1 (§6.1) — clone the selection's closure. The clone is coincident
        // with the source; `vert_map` relates each source vert to its clone.
        let dup = self.duplicate(geom, false);

        // Step 2 (§3) — reflect the clone's positions across the mirror plane.
        // This flips the winding of every clone face (determinant −1); the
        // winding flip is left to stand.
        self.reflect_clone_positions(&dup.geom_out, params.matrix, params.axis);

        // Step 3 (§4) — build the seam weld map over the verts NAMED IN `geom`
        // (not the closure verts). A named vert whose original lies within
        // `merge_dist` of the plane welds its clone onto it.
        let weld_pairs = self.build_seam_weld_map(geom, &dup.vert_map, params.matrix, params.axis, params.merge_dist);

        // Step 4 (§6.5) — weld the seam: each seam clone collapses onto its
        // original, seam-internal clone edges collapse, the original's
        // customdata survives. `use_centroid = false` keeps the survivor (the
        // original) at its own position so the seam sits on the originals (§4).
        self.weld_verts(&weld_pairs, false);

        // Step 5 (§1 / §6.6) — surface the surviving clone as `geom.out`: the
        // reflected half minus the welded-away seam clones.
        let welded_away: std::collections::HashSet<VertexKey> =
            weld_pairs.iter().map(|&(clone, _original)| clone).collect();
        let geom_out = dup
            .geom_out
            .into_iter()
            .filter(|g| match g {
                GeomKey::Vert(v) => self.vertices.try_get(*v).is_some() && !welded_away.contains(v),
                GeomKey::Edge(e) => self.edges.try_get(*e).is_some(),
                GeomKey::Face(f) => self.faces.try_get(*f).is_some(),
            })
            .collect();

        MirrorResult { geom_out }
    }

    /// §3 — reflect every clone vert's position across the mirror plane: map
    /// into the mirror frame, negate the `axis` coordinate, map back. A vert
    /// exactly on the plane maps to itself (negating zero is zero), which is what
    /// makes the seam weld a true coincident merge.
    fn reflect_clone_positions(&mut self, clone_geom: &[GeomKey], matrix: Mat4, axis: Axis) {
        let from_mirror = matrix.inverse();
        let scale = axis.reflect_scale();
        for entry in clone_geom {
            let GeomKey::Vert(clone) = *entry else {
                continue;
            };
            let world = self.vertex_position(clone);
            let in_mirror = matrix.transform_point3(world);
            let reflected = from_mirror.transform_point3(in_mirror * scale);
            self.set_vertex_position(clone, reflected);
        }
    }

    /// §4 — the seam test: range over the verts named directly in `geom`, and
    /// for each one whose original's mirror-space `axis` coordinate is within
    /// `merge_dist` of the plane, emit a `(clone, original)` weld couple. The map
    /// is naturally flat — every value is an original, never itself a key — so it
    /// satisfies weld-verts' flat-map invariant.
    fn build_seam_weld_map(
        &self,
        geom: &[GeomKey],
        vert_map: &std::collections::HashMap<VertexKey, VertexKey>,
        matrix: Mat4,
        axis: Axis,
        merge_dist: f32,
    ) -> Vec<(VertexKey, VertexKey)> {
        let axis_index = axis.index();
        let mut pairs: Vec<(VertexKey, VertexKey)> = Vec::new();
        let mut seen: std::collections::HashSet<VertexKey> = std::collections::HashSet::new();
        for entry in geom {
            let GeomKey::Vert(original) = *entry else {
                continue;
            };
            if !seen.insert(original) {
                continue;
            }
            let in_mirror = matrix.transform_point3(self.vertex_position(original));
            // Non-strict boundary (§4): a vert at exactly `merge_dist` welds.
            if in_mirror[axis_index].abs() <= merge_dist {
                if let Some(&clone) = vert_map.get(&original) {
                    // Guard against a degenerate `geom` that names an already-clone
                    // key: only weld a true original -> clone direction.
                    if clone != original {
                        pairs.push((clone, original));
                    }
                }
            }
        }
        pairs
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use glam::{Mat4, Vec3};

    use crate::loopmesh::*;

    /// Build the §2 strip of two quads to the +x side of the `x = 0` plane, the
    /// left edge {v0, v3} lying ON the plane. Returns the 6 verts row-major:
    /// `[v0(0,0) v1(1,0) v4(2,0) v3(0,1) v2(1,1) v5(2,1)]` indexed by position.
    fn seam_strip(mesh: &mut LoopMesh) -> (Vec<VertexKey>, Vec<FaceKey>) {
        let v0 = mesh.create_vertex(Vec3::new(0.0, 0.0, 0.0));
        let v1 = mesh.create_vertex(Vec3::new(1.0, 0.0, 0.0));
        let v4 = mesh.create_vertex(Vec3::new(2.0, 0.0, 0.0));
        let v3 = mesh.create_vertex(Vec3::new(0.0, 1.0, 0.0));
        let v2 = mesh.create_vertex(Vec3::new(1.0, 1.0, 0.0));
        let v5 = mesh.create_vertex(Vec3::new(2.0, 1.0, 0.0));

        let mk = |mesh: &mut LoopMesh, vs: [VertexKey; 4]| {
            let es: Vec<EdgeKey> = (0..4)
                .map(|i| mesh.create_edge([vs[i], vs[(i + 1) % 4]], EdgeCreateFlags::NO_DUPLICATE))
                .collect();
            mesh.face_create(&vs, &es, FaceCreateFlags::NONE)
        };
        let fa = mk(mesh, [v0, v1, v2, v3]);
        let fb = mk(mesh, [v1, v4, v5, v2]);
        (vec![v0, v1, v4, v3, v2, v5], vec![fa, fb])
    }

    /// The same two-quad strip translated to the +x side with its near edge at
    /// `x = 1` — NO vert lies on the `x = 0` plane.
    fn offset_strip(mesh: &mut LoopMesh) -> (Vec<VertexKey>, Vec<FaceKey>) {
        let v0 = mesh.create_vertex(Vec3::new(1.0, 0.0, 0.0));
        let v1 = mesh.create_vertex(Vec3::new(2.0, 0.0, 0.0));
        let v4 = mesh.create_vertex(Vec3::new(3.0, 0.0, 0.0));
        let v3 = mesh.create_vertex(Vec3::new(1.0, 1.0, 0.0));
        let v2 = mesh.create_vertex(Vec3::new(2.0, 1.0, 0.0));
        let v5 = mesh.create_vertex(Vec3::new(3.0, 1.0, 0.0));

        let mk = |mesh: &mut LoopMesh, vs: [VertexKey; 4]| {
            let es: Vec<EdgeKey> = (0..4)
                .map(|i| mesh.create_edge([vs[i], vs[(i + 1) % 4]], EdgeCreateFlags::NO_DUPLICATE))
                .collect();
            mesh.face_create(&vs, &es, FaceCreateFlags::NONE)
        };
        let fa = mk(mesh, [v0, v1, v2, v3]);
        let fb = mk(mesh, [v1, v4, v5, v2]);
        (vec![v0, v1, v4, v3, v2, v5], vec![fa, fb])
    }

    fn all_geom(verts: &[VertexKey], faces: &[FaceKey]) -> Vec<GeomKey> {
        faces
            .iter()
            .map(|&f| GeomKey::from(f))
            .chain(verts.iter().map(|&v| GeomKey::from(v)))
            .collect()
    }

    fn default_params(merge_dist: f32) -> MirrorParams {
        MirrorParams {
            matrix: Mat4::IDENTITY,
            merge_dist,
            axis: Axis::X,
            mirror_u: false,
            mirror_v: false,
            mirror_udim: false,
        }
    }

    /// §2 headline: the welded four-quad strip. Seam clones (the two on-plane
    /// verts) weld away, the seam clone edge collapses: net +4 V / +6 E / +2 F,
    /// i.e. 10 V / 13 E / 4 F.
    #[test]
    fn seam_strip_welds_into_four_quad_strip() {
        let mut mesh = LoopMesh::new();
        let (verts, faces) = seam_strip(&mut mesh);
        assert_eq!((mesh.totvert(), mesh.totedge(), mesh.totface()), (6, 7, 2));

        let geom = all_geom(&verts, &faces);
        mesh.mirror(&geom, default_params(0.5));

        assert_eq!(mesh.totvert(), 10, "6 originals + 4 surviving clones");
        assert_eq!(mesh.totedge(), 13, "+6 edges (seam clone edge collapsed)");
        assert_eq!(mesh.totface(), 4, "original 2 quads + 2 reflected quads");

        // The two on-plane originals survive; the reflected half lands on -x.
        let xs: Vec<f32> = mesh.vert_keys().iter().map(|&v| mesh.vertex_position(v).x).collect();
        assert!(xs.iter().any(|&x| x < -1.5), "a clone reflected to x = -2");
        assert!(xs.iter().any(|&x| (x - 2.0).abs() < 1e-6), "an original stays at x = 2");
        assert_eq!(
            xs.iter().filter(|&&x| x.abs() < 1e-6).count(),
            2,
            "exactly the two on-plane seam verts at x = 0 (clones welded away)"
        );
        mesh.validate_topology();
    }

    /// §4 / §7 "fires": with `merge_dist` catching the on-plane verts, the two
    /// halves share the seam edge — the result is one connected component.
    #[test]
    fn merge_dist_fires_welds_seam() {
        let mut mesh = LoopMesh::new();
        let (verts, faces) = seam_strip(&mut mesh);
        let geom = all_geom(&verts, &faces);

        let result = mesh.mirror(&geom, default_params(0.5));

        // The two seam clones welded away, so geom_out carries 4 surviving clone
        // verts (not 6) plus the reflected edges/faces.
        let clone_verts = result
            .geom_out
            .iter()
            .filter(|g| matches!(g, GeomKey::Vert(_)))
            .count();
        assert_eq!(clone_verts, 4, "the 2 on-plane seam clones welded away");

        // Connectivity: a single shared seam edge {v0, v3} carries faces from
        // both halves (radial count 2).
        let seam = mesh.query_edge_exists(verts[0], verts[3]).expect("seam edge survives");
        assert_eq!(
            mesh.iter_radial_loops(seam).count(),
            2,
            "the seam edge is shared by an original and a reflected face"
        );
        mesh.validate_topology();
    }

    /// §4.1 / §7 "blocks": `merge_dist = 0` on a strip with NO vert on the plane
    /// welds nothing — the reflected copy is a disconnected shell (doubled
    /// counts, 12 V / 14 E / 4 F).
    #[test]
    fn merge_dist_zero_disconnects_offset_strip() {
        let mut mesh = LoopMesh::new();
        let (verts, faces) = offset_strip(&mut mesh);
        assert_eq!((mesh.totvert(), mesh.totedge(), mesh.totface()), (6, 7, 2));

        let geom = all_geom(&verts, &faces);
        let result = mesh.mirror(&geom, default_params(0.0));

        assert_eq!(mesh.totvert(), 12, "full disjoint clone, nothing welded");
        assert_eq!(mesh.totedge(), 14);
        assert_eq!(mesh.totface(), 4);
        assert_eq!(
            result.geom_out.iter().filter(|g| matches!(g, GeomKey::Vert(_))).count(),
            6,
            "every clone vert survives — no seam weld"
        );

        // No edge bridges the +x originals and the -x clones: the two shells share
        // no vert. The reflected half lives at negative x.
        let neg: HashSet<VertexKey> = mesh
            .vert_keys()
            .into_iter()
            .filter(|&v| mesh.vertex_position(v).x < 0.0)
            .collect();
        assert_eq!(neg.len(), 6, "all 6 clone verts reflected to -x");
        for &v in &verts {
            assert!(!neg.contains(&v), "originals stay on +x");
        }
        mesh.validate_topology();
    }

    /// §3 / §3.1 winding flip: the reflection reverses each clone face's
    /// loop-cycle direction, so a reflected face is wound opposite to the
    /// original it mirrors. Read directly off the loop-cycle vertex order rather
    /// than through an up-to-sign normal (which a single face cannot disambiguate).
    #[test]
    fn reflection_flips_clone_face_winding() {
        let mut mesh = LoopMesh::new();
        // A single quad on the +x side, one edge on the plane, so its clone welds
        // its seam and we can identify the reflected face unambiguously.
        let v0 = mesh.create_vertex(Vec3::new(0.0, 0.0, 0.0));
        let v1 = mesh.create_vertex(Vec3::new(1.0, 0.0, 0.0));
        let v2 = mesh.create_vertex(Vec3::new(1.0, 1.0, 0.0));
        let v3 = mesh.create_vertex(Vec3::new(0.0, 1.0, 0.0));
        let es: Vec<EdgeKey> = (0..4)
            .map(|i| {
                mesh.create_edge(
                    [[v0, v1, v2, v3][i], [v0, v1, v2, v3][(i + 1) % 4]],
                    EdgeCreateFlags::NO_DUPLICATE,
                )
            })
            .collect();
        let face = mesh.face_create(&[v0, v1, v2, v3], &es, FaceCreateFlags::NONE);

        // Signed area of the face's loop polygon in the xy-plane (the shoelace
        // z-component). Its sign is the in-space winding: a determinant-−1
        // reflection negates it relative to a non-flipped copy (§3.1).
        let signed_area = |mesh: &LoopMesh, f: FaceKey| -> f32 {
            let ring = mesh.face_vert_keys(f);
            let n = ring.len();
            let mut sum = 0.0;
            for i in 0..n {
                let a = mesh.vertex_position(ring[i]);
                let b = mesh.vertex_position(ring[(i + 1) % n]);
                sum += a.x * b.y - b.x * a.y;
            }
            0.5 * sum
        };
        let original_area = signed_area(&mesh, face);

        let geom = all_geom(&[v0, v1, v2, v3], &[face]);
        mesh.mirror(&geom, default_params(0.5));

        // Find the reflected face: the one face touching a vert at negative x.
        let reflected = mesh
            .face_keys()
            .into_iter()
            .find(|&f| mesh.face_vert_keys(f).iter().any(|&v| mesh.vertex_position(v).x < 0.0))
            .expect("a reflected face exists");
        let reflected_area = signed_area(&mesh, reflected);

        // The reflection flips the winding: the reflected face's signed area has
        // the OPPOSITE sign to the original's. (Without the flip a verbatim
        // reflected-position copy would carry the same sign and shade inverted.)
        assert!(
            original_area.abs() > 1e-6 && reflected_area.abs() > 1e-6,
            "both faces are non-degenerate"
        );
        assert!(
            original_area.signum() != reflected_area.signum(),
            "reflected winding is flipped: original area {original_area}, reflected area {reflected_area}"
        );
        mesh.validate_topology();
    }

    /// §4: the seam test ranges over the NAMED input verts, not the closure. A
    /// face-only `geom` (no verts named) welds nothing even when an edge lies on
    /// the plane — the reflected copy comes back disconnected.
    #[test]
    fn face_only_geom_welds_nothing() {
        let mut mesh = LoopMesh::new();
        let (_verts, faces) = seam_strip(&mut mesh);

        // Name only the faces — no verts.
        let geom: Vec<GeomKey> = faces.iter().map(|&f| GeomKey::from(f)).collect();
        mesh.mirror(&geom, default_params(0.5));

        assert_eq!(mesh.totvert(), 12, "no seam weld: full disjoint clone");
        assert_eq!(mesh.totface(), 4);
        mesh.validate_topology();
    }

    /// §7 empty selection: a no-op overall.
    #[test]
    fn empty_selection_is_noop() {
        let mut mesh = LoopMesh::new();
        let (_verts, _faces) = seam_strip(&mut mesh);
        let before = (mesh.totvert(), mesh.totedge(), mesh.totface());

        let result = mesh.mirror(&[], default_params(0.5));

        assert!(result.geom_out.is_empty());
        assert_eq!((mesh.totvert(), mesh.totedge(), mesh.totface()), before);
        mesh.validate_topology();
    }
}
