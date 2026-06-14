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
    /// original's raw world `axis` coordinate is within `merge_dist` of zero
    /// (the world `axis = 0` plane; non-strict `<=`, §4). `matrix` does not
    /// participate in this test.
    pub merge_dist: f32,
    /// Which mirror-space coordinate the reflection negates.
    pub axis: Axis,
    /// When set, the cloned face loops' UV `u` coordinate is flipped
    /// `u ↦ 1 − u` across every Float2 (UV) loop layer (§5).
    pub mirror_u: bool,
    /// As `mirror_u` for the UV `v` coordinate (`v ↦ 1 − v`, §5).
    pub mirror_v: bool,
    /// Modifies the UV reflection centre: when set, each flipped UV coordinate
    /// reflects **within its UDIM tile** (about the tile's local centre,
    /// `c ↦ floor(c) + 1 − (c − floor(c))`) rather than about `0.5` of the unit
    /// square. The tile index (the floor) is preserved. Only meaningful
    /// alongside `mirror_u` / `mirror_v` (§5).
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
    /// `mirror_u` / `mirror_v` flip the cloned face loops' UVs (§5); with them
    /// clear, UVs are copied verbatim by the duplicate step. `mirror_udim`
    /// changes the reflection centre from `0.5` of the unit square to the
    /// within-tile midpoint, preserving the UDIM tile index (§5).
    ///
    /// See `radial-mesh-guides/operators/MIRROR.md` for the full algorithm and
    /// parity contract.
    pub fn mirror(&mut self, geom: &[GeomKey], params: MirrorParams) -> MirrorResult {
        // Step 1 (§6.1) — clone the selection's closure. The clone is coincident
        // with the source; `vert_map` relates each source vert to its clone.
        let dup = self.duplicate(geom, false);

        // Step 2 (§3) — reflect the clone's positions across the mirror plane.
        // This flips the winding of every clone face (determinant −1); the
        // winding flip is left to stand.
        self.reflect_clone_positions(&dup.geom_out, params.matrix, params.axis);

        // Step 3 (§4) — build the seam weld map over the verts NAMED IN `geom`
        // (not the closure verts). A named vert whose original lies within
        // `merge_dist` of the WORLD `axis = 0` plane welds its clone onto it.
        // `matrix` is deliberately NOT passed: it owns the §3 reflection frame
        // only, never the seam test (§4).
        let weld_pairs =
            self.build_seam_weld_map(geom, &dup.vert_map, params.axis, params.merge_dist);

        // Step 4 (§5) — flip the cloned faces' UVs, after the verbatim CD copy
        // (done by the clone in step 1) and before the weld. The flip touches
        // the clone half only; the originals keep their UVs untouched.
        if params.mirror_u || params.mirror_v {
            self.mirror_clone_uvs(
                &dup.geom_out,
                params.mirror_u,
                params.mirror_v,
                params.mirror_udim,
            );
        }

        // Step 5 (§6.5) — weld the seam: each seam clone collapses onto its
        // original, seam-internal clone edges collapse, the original's
        // customdata survives. `use_centroid = false` keeps the survivor (the
        // original) at its own position so the seam sits on the originals (§4).
        self.weld_verts(&weld_pairs, false);

        // Step 6 (§1 / §6.6) — surface the surviving clone as `geom.out`: the
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

    /// §5 — flip the cloned faces' loop UVs. Over every loop of every cloned
    /// face, for each Float2 (UV-shaped) loop layer, reflect the requested
    /// coordinate via [`mirror_uv_coord`]: `mirror_u` flips `u`, `mirror_v`
    /// flips `v`, both may apply together. With `mirror_udim` clear the centre is
    /// `0.5` of the unit square (`c ↦ 1 − c`); with it set the reflection happens
    /// within the coordinate's UDIM tile, preserving the tile index. Only
    /// `CD_PROP_FLOAT2`-shaped layers (storage `Vec2`) are touched; other loop
    /// customdata stays as the verbatim duplicate copy left it.
    fn mirror_clone_uvs(
        &mut self,
        clone_geom: &[GeomKey],
        mirror_u: bool,
        mirror_v: bool,
        mirror_udim: bool,
    ) {
        let uv_layers: Vec<ElementAttributeKey> = self
            .loop_attribs()
            .iter_descriptors()
            .filter_map(|(key, _)| {
                matches!(
                    self.loop_attribs().storage_of(key),
                    Some(AttributeStorage::Vec2(_))
                )
                .then_some(key)
            })
            .collect();
        if uv_layers.is_empty() {
            return;
        }

        for entry in clone_geom {
            let GeomKey::Face(face) = *entry else {
                continue;
            };
            let slots: Vec<usize> = self.iter_face_loops(face).map(|l| l.as_usize()).collect();
            for &layer in &uv_layers {
                for &slot in &slots {
                    let Some(mut uv) = self.loop_attribs().get_vec2(layer, slot) else {
                        continue;
                    };
                    if mirror_u {
                        uv[0] = mirror_uv_coord(uv[0], mirror_udim);
                    }
                    if mirror_v {
                        uv[1] = mirror_uv_coord(uv[1], mirror_udim);
                    }
                    self.loop_attribs_mut().set_vec2(layer, slot, uv);
                }
            }
        }
    }

    /// §4 — the seam test: range over the verts named directly in `geom`, and
    /// for each one whose original's raw world `axis` coordinate is within
    /// `merge_dist` of the world `axis = 0` plane, emit a `(clone, original)`
    /// weld couple. `matrix` is deliberately NOT consulted here — it owns the §3
    /// reflection frame only; the seam test always reads the raw world
    /// coordinate, so a shifted/rotated `matrix` reflects across the shifted
    /// plane but still welds only verts on the world `axis = 0` plane. The map is
    /// naturally flat — every value is an original, never itself a key — so it
    /// satisfies weld-verts' flat-map invariant.
    fn build_seam_weld_map(
        &self,
        geom: &[GeomKey],
        vert_map: &std::collections::HashMap<VertexKey, VertexKey>,
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
            let world = self.vertex_position(original);
            // Non-strict boundary (§4): a vert at exactly `merge_dist` welds.
            if world[axis_index].abs() <= merge_dist {
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

/// §5 — reflect a single UV coordinate. With `udim` clear the reflection centre
/// is `0.5` of the unit square (`c ↦ 1 − c`). With `udim` set the coordinate
/// reflects **within its UDIM tile**: the tile index (the floor) is preserved
/// and only the in-tile fraction flips about the tile centre, giving
/// `floor + (1 − frac)`. The fraction is the **non-negative** part `c − c.floor()`
/// (in `[0, 1)`) — not a signed `fract`, which would mis-handle negative `c` and
/// shift the tile.
fn mirror_uv_coord(c: f32, udim: bool) -> f32 {
    if udim {
        let floor_part = c.floor();
        let frac = c - floor_part;
        floor_part + (1.0 - frac)
    } else {
        1.0 - c
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

    /// The §2 strip rotated so the mirror axis is **y**: the seam edge {v0, v3}
    /// lies ON the `y = 0` plane and the strip extends into +y. The two quads
    /// stack along +y, with the in-plane coordinate carried by x. Returns the 6
    /// verts in the same row-major slot order as `seam_strip`:
    /// `[v0(0,0) v1(0,1) v4(0,2) v3(1,0) v2(1,1) v5(1,2)]` (x, y).
    fn seam_strip_y(mesh: &mut LoopMesh) -> (Vec<VertexKey>, Vec<FaceKey>) {
        let v0 = mesh.create_vertex(Vec3::new(0.0, 0.0, 0.0));
        let v1 = mesh.create_vertex(Vec3::new(0.0, 1.0, 0.0));
        let v4 = mesh.create_vertex(Vec3::new(0.0, 2.0, 0.0));
        let v3 = mesh.create_vertex(Vec3::new(1.0, 0.0, 0.0));
        let v2 = mesh.create_vertex(Vec3::new(1.0, 1.0, 0.0));
        let v5 = mesh.create_vertex(Vec3::new(1.0, 2.0, 0.0));

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
        let xs: Vec<f32> = mesh
            .vert_keys()
            .iter()
            .map(|&v| mesh.vertex_position(v).x)
            .collect();
        assert!(xs.iter().any(|&x| x < -1.5), "a clone reflected to x = -2");
        assert!(
            xs.iter().any(|&x| (x - 2.0).abs() < 1e-6),
            "an original stays at x = 2"
        );
        assert_eq!(
            xs.iter().filter(|&&x| x.abs() < 1e-6).count(),
            2,
            "exactly the two on-plane seam verts at x = 0 (clones welded away)"
        );
        mesh.validate_topology();
    }

    /// §1 / §4 axis-generic path: the same welded four-quad strip headline as
    /// `seam_strip_welds_into_four_quad_strip`, but reflected across the `y = 0`
    /// plane (`axis = Y`). Pins that the reflection negates the *y* coordinate
    /// and the seam test reads the *same* axis coordinate — a regression keying
    /// on x (the default) would not find the on-plane verts and would weld
    /// nothing, doubling the counts. Same 10 V / 13 E / 4 F welded outcome.
    #[test]
    fn seam_strip_welds_into_four_quad_strip_axis_y() {
        let mut mesh = LoopMesh::new();
        let (verts, faces) = seam_strip_y(&mut mesh);
        assert_eq!((mesh.totvert(), mesh.totedge(), mesh.totface()), (6, 7, 2));

        let mut params = default_params(0.5);
        params.axis = Axis::Y;

        let geom = all_geom(&verts, &faces);
        let result = mesh.mirror(&geom, params);

        // The Y analogue of the X headline's welded counts.
        assert_eq!(mesh.totvert(), 10, "6 originals + 4 surviving clones");
        assert_eq!(mesh.totedge(), 13, "+6 edges (seam clone edge collapsed)");
        assert_eq!(mesh.totface(), 4, "original 2 quads + 2 reflected quads");

        // The reflected half lands on −y; the two on-plane originals stay at y=0.
        let ys: Vec<f32> = mesh
            .vert_keys()
            .iter()
            .map(|&v| mesh.vertex_position(v).y)
            .collect();
        assert!(ys.iter().any(|&y| y < -1.5), "a clone reflected to y = -2");
        assert!(
            ys.iter().any(|&y| (y - 2.0).abs() < 1e-6),
            "an original stays at y = 2"
        );
        assert_eq!(
            ys.iter().filter(|&&y| y.abs() < 1e-6).count(),
            2,
            "exactly the two on-plane seam verts at y = 0 (clones welded away)"
        );

        // The seam genuinely welded: the two on-plane seam clones are gone, so
        // only 4 clone verts survive in geom_out (not 6).
        let clone_verts = result
            .geom_out
            .iter()
            .filter(|g| matches!(g, GeomKey::Vert(_)))
            .count();
        assert_eq!(clone_verts, 4, "the 2 on-plane seam clones welded away");

        // The seam edge {v0, v3} (on y = 0) is shared by an original and a
        // reflected face — radial count 2 proves the two halves fused.
        let seam = mesh
            .query_edge_exists(verts[0], verts[3])
            .expect("seam edge survives");
        assert_eq!(
            mesh.iter_radial_loops(seam).count(),
            2,
            "the seam edge is shared by an original and a reflected face"
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
        let seam = mesh
            .query_edge_exists(verts[0], verts[3])
            .expect("seam edge survives");
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
            result
                .geom_out
                .iter()
                .filter(|g| matches!(g, GeomKey::Vert(_)))
                .count(),
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
            .find(|&f| {
                mesh.face_vert_keys(f)
                    .iter()
                    .any(|&v| mesh.vertex_position(v).x < 0.0)
            })
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

    /// Register a Float2 (UV-shaped) loop layer and stamp every loop with a
    /// distinct, non-symmetric UV so a flip is visible: corner `i` of face `j`
    /// gets `[0.1 + 0.05*i, 0.2 + 0.05*j]`. Returns the layer key and the
    /// per-loop value map captured *before* the mirror runs.
    fn decorate_loops_with_uv(
        mesh: &mut LoopMesh,
        faces: &[FaceKey],
    ) -> (
        ElementAttributeKey,
        std::collections::HashMap<LoopKey, [f32; 2]>,
    ) {
        let uv = mesh
            .register_loop_attribute(ElementAttributeDesc::numeric(
                AttributeKey::custom("uv"),
                AttrType::Vec2,
            ))
            .expect("uv layer registers");
        let mut values: std::collections::HashMap<LoopKey, [f32; 2]> =
            std::collections::HashMap::new();
        for (j, &f) in faces.iter().enumerate() {
            for (i, l) in mesh.face_loop_keys(f).into_iter().enumerate() {
                let value = [0.1 + 0.05 * i as f32, 0.2 + 0.05 * j as f32];
                mesh.loop_attribs_mut().set_vec2(uv, l.as_usize(), value);
                values.insert(l, value);
            }
        }
        (uv, values)
    }

    /// §5 fires: with `mirror_u`/`mirror_v` set, the cloned loops' UVs are
    /// flipped `[u,v] ↦ [1−u, 1−v]`, while the originals keep their UVs verbatim.
    /// Uses the offset strip with `merge_dist = 0` so nothing welds — the clone
    /// faces are exactly the surviving `geom_out` faces and their loops are
    /// readable directly.
    #[test]
    fn mirror_uv_flips_clone_uvs_only() {
        let mut mesh = LoopMesh::new();
        let (verts, faces) = offset_strip(&mut mesh);
        let (uv, original_uvs) = decorate_loops_with_uv(&mut mesh, &faces);

        let original_loops: HashSet<LoopKey> = original_uvs.keys().copied().collect();

        let geom = all_geom(&verts, &faces);
        let mut params = default_params(0.0);
        params.mirror_u = true;
        params.mirror_v = true;
        let result = mesh.mirror(&geom, params);

        // Originals are untouched (§5: the flip is on the clone only).
        for (&l, &before) in &original_uvs {
            let after = mesh.loop_attribs().get_vec2(uv, l.as_usize()).unwrap();
            assert_eq!(after, before, "original loop UV is untouched");
        }

        // Every clone loop carries the flip of some original loop's UV.
        let clone_faces: Vec<FaceKey> = result
            .geom_out
            .iter()
            .filter_map(|g| match g {
                GeomKey::Face(f) => Some(*f),
                _ => None,
            })
            .collect();
        assert_eq!(clone_faces.len(), 2, "two reflected clone faces survive");

        let mut flipped_count = 0;
        for &f in &clone_faces {
            for l in mesh.face_loop_keys(f) {
                assert!(!original_loops.contains(&l), "clone loop is a fresh key");
                let [u, v] = mesh.loop_attribs().get_vec2(uv, l.as_usize()).unwrap();
                // The flipped value is the reflection of a value the source
                // carried: u ∈ {0.10,0.15,0.20,0.25} ↦ {0.90,0.85,0.80,0.75}.
                let source_u = 1.0 - u;
                let source_v = 1.0 - v;
                let matches_source = original_uvs
                    .values()
                    .any(|&[su, sv]| (su - source_u).abs() < 1e-5 && (sv - source_v).abs() < 1e-5);
                assert!(
                    matches_source,
                    "clone UV [{u},{v}] is the 1−x flip of a source UV"
                );
                flipped_count += 1;
            }
        }
        assert_eq!(flipped_count, 8, "all eight clone loops flipped");
        mesh.validate_topology();
    }

    /// §5 verbatim contrast: with `mirror_u`/`mirror_v` clear, the cloned loops'
    /// UVs are copied verbatim (the duplicate §4 behaviour). Every clone loop's
    /// UV matches an original's exactly — no `1−x` flip.
    #[test]
    fn mirror_uv_clear_copies_clone_uvs_verbatim() {
        let mut mesh = LoopMesh::new();
        let (verts, faces) = offset_strip(&mut mesh);
        let (uv, original_uvs) = decorate_loops_with_uv(&mut mesh, &faces);

        let source_values: HashSet<[u32; 2]> = original_uvs
            .values()
            .map(|&[u, v]| [u.to_bits(), v.to_bits()])
            .collect();

        let geom = all_geom(&verts, &faces);
        // Flags clear: verbatim copy.
        let result = mesh.mirror(&geom, default_params(0.0));

        let clone_faces: Vec<FaceKey> = result
            .geom_out
            .iter()
            .filter_map(|g| match g {
                GeomKey::Face(f) => Some(*f),
                _ => None,
            })
            .collect();
        assert_eq!(clone_faces.len(), 2);

        for &f in &clone_faces {
            for l in mesh.face_loop_keys(f) {
                let [u, v] = mesh.loop_attribs().get_vec2(uv, l.as_usize()).unwrap();
                assert!(
                    source_values.contains(&[u.to_bits(), v.to_bits()]),
                    "clone UV [{u},{v}] is a verbatim source UV (no flip)"
                );
            }
        }
        mesh.validate_topology();
    }

    /// Register a Float2 loop layer and stamp every loop with a UV inside the
    /// UDIM tile `(1, 0)` — `u ∈ (1, 2)`, `v ∈ (0, 1)` — chosen away from the
    /// tile centre so the in-tile flip moves it: corner `i` of face `j` gets
    /// `[1.1 + 0.05*i, 0.2 + 0.05*j]`. Returns the layer key.
    fn decorate_loops_with_tile_uv(mesh: &mut LoopMesh, faces: &[FaceKey]) -> ElementAttributeKey {
        let uv = mesh
            .register_loop_attribute(ElementAttributeDesc::numeric(
                AttributeKey::custom("uv"),
                AttrType::Vec2,
            ))
            .expect("uv layer registers");
        for (j, &f) in faces.iter().enumerate() {
            for (i, l) in mesh.face_loop_keys(f).into_iter().enumerate() {
                let value = [1.1 + 0.05 * i as f32, 0.2 + 0.05 * j as f32];
                mesh.loop_attribs_mut().set_vec2(uv, l.as_usize(), value);
            }
        }
        uv
    }

    /// §5 `mirror_udim` fires: a UV that starts in the non-zero UDIM tile
    /// `(1, 0)` (`u ∈ (1, 2)`) stays in that same tile after the flip. The
    /// within-tile reflection is `u ↦ floor(u) + 1 − (u − floor(u))`, which for
    /// `u ∈ (1, 2)` lands back in `(1, 2)` — never crossing the tile boundary.
    #[test]
    fn mirror_udim_keeps_uv_in_tile() {
        let mut mesh = LoopMesh::new();
        let (verts, faces) = offset_strip(&mut mesh);
        let uv = decorate_loops_with_tile_uv(&mut mesh, &faces);

        let geom = all_geom(&verts, &faces);
        let mut params = default_params(0.0);
        params.mirror_u = true;
        params.mirror_v = true;
        params.mirror_udim = true;
        let result = mesh.mirror(&geom, params);

        let clone_faces: Vec<FaceKey> = result
            .geom_out
            .iter()
            .filter_map(|g| match g {
                GeomKey::Face(f) => Some(*f),
                _ => None,
            })
            .collect();
        assert_eq!(clone_faces.len(), 2);

        for &f in &clone_faces {
            for l in mesh.face_loop_keys(f) {
                let [u, v] = mesh.loop_attribs().get_vec2(uv, l.as_usize()).unwrap();
                // u started in tile (1,2); the in-tile flip keeps it there.
                assert!((1.0..2.0).contains(&u), "u stayed in UDIM tile (1,2): {u}");
                // v started in tile (0,1); the in-tile flip keeps it there.
                assert!((0.0..1.0).contains(&v), "v stayed in UDIM tile (0,1): {v}");
                // The exact reflection: u ∈ {1.10,1.15,1.20,1.25} ↦ {1.90,..,1.75}.
                let source_u = 1.0 + 1.0 - (u - 1.0); // invert: floor=1, frac flip
                assert!(
                    (1.10..=1.25).contains(&source_u),
                    "u is an in-tile flip: {u}"
                );
            }
        }
        mesh.validate_topology();
    }

    /// §5 contrast: for the SAME tile-(1,0) input, `mirror_udim` set differs from
    /// `mirror_udim` clear. The clear path reflects about `0.5` (`u ↦ 1 − u`),
    /// sending `u ∈ (1, 2)` to `(−1, 0)` — a different tile — while the set path
    /// keeps it in `(1, 2)`. The two results must not coincide.
    #[test]
    fn mirror_udim_differs_from_unit_square() {
        let collect_clone_uvs = |mirror_udim: bool| -> Vec<[u32; 2]> {
            let mut mesh = LoopMesh::new();
            let (verts, faces) = offset_strip(&mut mesh);
            let uv = decorate_loops_with_tile_uv(&mut mesh, &faces);
            let geom = all_geom(&verts, &faces);
            let mut params = default_params(0.0);
            params.mirror_u = true;
            params.mirror_v = true;
            params.mirror_udim = mirror_udim;
            let result = mesh.mirror(&geom, params);
            let mut out: Vec<[u32; 2]> = Vec::new();
            for g in &result.geom_out {
                let GeomKey::Face(f) = g else { continue };
                for l in mesh.face_loop_keys(*f) {
                    let [u, v] = mesh.loop_attribs().get_vec2(uv, l.as_usize()).unwrap();
                    out.push([u.to_bits(), v.to_bits()]);
                }
            }
            out.sort();
            out
        };

        let udim = collect_clone_uvs(true);
        let unit_square = collect_clone_uvs(false);
        assert_ne!(
            udim, unit_square,
            "mirror_udim (within-tile) must differ from the unit-square reflection"
        );
        // Unit-square sends tile-(1,2) UVs to negative tile (−1, 0).
        for [ub, _vb] in &unit_square {
            assert!(
                f32::from_bits(*ub) < 0.0,
                "unit-square flip jumps u below 0"
            );
        }
    }

    /// §5: `mirror_u` alone flips only `u`; `v` is copied verbatim.
    #[test]
    fn mirror_u_alone_flips_only_u() {
        let mut mesh = LoopMesh::new();
        let (verts, faces) = offset_strip(&mut mesh);
        let (uv, _) = decorate_loops_with_uv(&mut mesh, &faces);

        let geom = all_geom(&verts, &faces);
        let mut params = default_params(0.0);
        params.mirror_u = true;
        let result = mesh.mirror(&geom, params);

        for g in &result.geom_out {
            let GeomKey::Face(f) = g else { continue };
            for l in mesh.face_loop_keys(*f) {
                let [u, v] = mesh.loop_attribs().get_vec2(uv, l.as_usize()).unwrap();
                // u was flipped into [0.75, 0.90]; v stays in its source band.
                assert!((0.75..=0.90).contains(&u), "u flipped: {u}");
                assert!((0.20..=0.25).contains(&v), "v unchanged: {v}");
            }
        }
        mesh.validate_topology();
    }

    /// §3 / §4 non-identity `matrix` (shifted mirror plane): a `matrix` that
    /// reflects across the plane `x = 2` (off the origin), run on the offset
    /// strip whose verts sit at x ∈ {1, 2, 3} — none on the world `x = 0` plane.
    /// §3 reflects the clone across the shifted plane, but §4's seam test reads
    /// the RAW WORLD `x` coordinate and welds only verts on world `x = 0`, so
    /// nothing welds: the result is a fully disconnected reflected copy (counts
    /// doubled). This pins that `matrix` reflects but the seam stays pinned to
    /// world `axis = 0` — the bug being fixed transformed the tested coordinate
    /// by `matrix`, which would have welded the x=2 verts across the shifted
    /// plane and corrupted the result.
    #[test]
    fn shifted_matrix_reflects_but_welds_nothing() {
        let mut mesh = LoopMesh::new();
        let (verts, faces) = offset_strip(&mut mesh);
        assert_eq!((mesh.totvert(), mesh.totedge(), mesh.totface()), (6, 7, 2));

        // Reflect across the world plane `x = 2`: translate so x=2 maps to the
        // mirror-frame origin; the §3 negate-then-inverse then mirrors about x=2.
        let mut params = default_params(0.0);
        params.matrix = Mat4::from_translation(Vec3::new(-2.0, 0.0, 0.0));

        let geom = all_geom(&verts, &faces);
        let result = mesh.mirror(&geom, params);

        // Seam test reads raw world x (none on x=0), so nothing welds: doubled.
        assert_eq!(mesh.totvert(), 12, "no seam weld: full disjoint clone");
        assert_eq!(mesh.totedge(), 14);
        assert_eq!(mesh.totface(), 4);
        assert_eq!(
            result
                .geom_out
                .iter()
                .filter(|g| matches!(g, GeomKey::Vert(_)))
                .count(),
            6,
            "every clone vert survives — the world `x = 0` seam test welds nothing"
        );

        // The clone reflected across x = 2: source x ∈ {1,2,3} maps to {3,2,1},
        // i.e. the reflected shell lives at x ≥ 1, never at negative x — proof
        // the shifted `matrix` (not the world plane) drove the reflection.
        for &v in &mesh.vert_keys() {
            assert!(
                mesh.vertex_position(v).x >= 1.0 - 1e-6,
                "reflection about x = 2 keeps everything at x >= 1"
            );
        }
        // The two shells share no vert: the +x originals at x ∈ {1,2,3} and the
        // clones reflected about x=2 are distinct keys with no welded seam.
        let originals: HashSet<VertexKey> = verts.iter().copied().collect();
        let clone_keys: Vec<VertexKey> = result
            .geom_out
            .iter()
            .filter_map(|g| match g {
                GeomKey::Vert(v) => Some(*v),
                _ => None,
            })
            .collect();
        for c in &clone_keys {
            assert!(
                !originals.contains(c),
                "clone keys are disjoint from originals"
            );
        }
        mesh.validate_topology();
    }
}
