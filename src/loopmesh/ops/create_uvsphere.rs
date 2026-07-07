use glam::{Mat4, Vec3};

use crate::loopmesh::*;

impl LoopMesh {
    /// Lay down a **longitude/latitude UV-sphere** — a stack of `v − 1` interior
    /// constant-latitude rings of `u` vertices each, joined by quad bands and
    /// closed off by a triangle fan at each pole — and append it to the mesh
    /// (CREATE_UVSPHERE.md §2 / §3).
    ///
    /// Lattice vertex `(a, j)` (latitude `1 ≤ a ≤ v − 1`, longitude `0 ≤ j < u`)
    /// sits at local `P = (r·sin φ_a·sin θ_j, r·sin φ_a·cos θ_j, r·cos φ_a)` with
    /// `φ_a = π·a/v` measured down from `+z` and `θ_j = 2π·j/u` (§2.1). The `+sin`
    /// sign winds each ring **clockwise** as seen from `+z` — the same convention
    /// as `create_cone`, the opposite of `create_circle` — which lands the bands
    /// and pole fans facing **outward** (§3). The north pole sits at `(0, 0, +r)`
    /// and the south pole at `(0, 0, −r)`.
    ///
    /// Both `u_segments` and `v_segments` are **clamped up to a minimum of `1`**
    /// (§1 / §7): a count of `0` builds a degenerate shell, **not** nothing — a
    /// deliberate divergence from `create_cone`'s `segments = 0` no-op, matching
    /// `create_grid`'s clamp. The well-formed regime is `u ≥ 3`, `v ≥ 2`; below it
    /// the bare kernel emits ill-formed collapsed geometry (§7), reproduced here by
    /// three floors: `v = 1` folds to two poles joined by one meridian edge (the
    /// bigon sphere); `u = 1` leaves two coincident meridian arcs sharing the poles
    /// (the doubled, multi-edge collapsed meridian); `u = 2` welds the coincident
    /// opposite-longitude band quads down to `F = v` (the flat lens). Each floor is
    /// guarded so the well-formed path is untouched.
    ///
    /// Each latitude band between rings `a` and `a + 1` contributes one quad per
    /// longitude `(P(a+1,j), P(a,j), P(a,j+1), P(a+1,j+1))`, wound outward (§3).
    /// The north band (`a = 0`) and south band (`a = v − 1`) degenerate — both of
    /// the pole's corners coincide — so the quad collapses to the pole-fan triangle
    /// the guide names, automatically.
    ///
    /// The 4×4 `matrix` is applied to **every** generated vertex in one final pass
    /// *after* the build and the UV unwrap (§2.3): the unwrap is anchored to the
    /// canonical (pre-`matrix`) sphere, so a mirror or rotation in `matrix` moves
    /// the geometry but leaves the UV layout untouched.
    ///
    /// When `calc_uvs` is set **and** the mesh carries an active per-loop
    /// 2-component-float (UV) layer, an **equirectangular** unwrap is painted
    /// across the new faces (§4): longitude → `u`, latitude → `v`, computed from
    /// the local positions, with pole-longitude averaging, a seam-wrap fix, and a
    /// final left-justify. With `calc_uvs` clear or no UV layer present the flag is
    /// a silent no-op — it never mints a layer.
    ///
    /// Create-uvsphere is purely constructive and **additive**: it reads no
    /// existing geometry, never clears the mesh, and never merges into prior verts
    /// (§ intro / §3 / §7). The created vertices are **returned**; the slot order
    /// is a documented don't-care (§8.2), so they are emitted in a clean
    /// pole-then-rings order (north, south, then each interior ring in `(a, j)`
    /// order).
    ///
    /// See `radial-mesh-guides/operators/CREATE_UVSPHERE.md` for the full contract.
    pub fn create_uvsphere(
        &mut self,
        u_segments: i32,
        v_segments: i32,
        radius: f32,
        matrix: Mat4,
        calc_uvs: bool,
    ) -> Vec<VertexKey> {
        use std::f64::consts::{PI, TAU};

        // §1 / §7 — clamp each segment count up to a minimum of 1 (NOT a no-op).
        let u = u_segments.max(1) as usize;
        let v = v_segments.max(1) as usize;

        // §7 — degenerate floors below the well-formed regime (`u ≥ 3`, `v ≥ 2`).
        // The bare kernel accepts these and emits ill-formed collapsed geometry the
        // editor layer would clamp away; each floor is its own collapse, handled
        // before the well-formed path so that path stays byte-identical.
        if v == 1 {
            return self.create_uvsphere_bigon(radius, matrix);
        }
        if u == 1 {
            return self.create_uvsphere_collapsed_meridian(v, radius, matrix);
        }

        // §2.3 — build at local positions; `matrix` is applied in a final pass.
        let north = self.create_vertex(Vec3::new(0.0, 0.0, radius));
        let south = self.create_vertex(Vec3::new(0.0, 0.0, -radius));

        // §2.1 — the `v − 1` interior rings, each `u` vertices wound clockwise.
        // The ring trig and radius multiply run in f64, casting to f32 only when
        // writing the final position: a one-time constructor cost that avoids a
        // round-half tie at cos φ = ½ rings where an f32 intermediate would round
        // the opposite way from the higher-precision reference.
        let radius = radius as f64;
        let mut interior: Vec<Vec<VertexKey>> = Vec::with_capacity(v - 1);
        for a in 1..v {
            let phi = PI * (a as f64) / (v as f64);
            let (sin_phi, cos_phi) = (phi.sin(), phi.cos());
            let mut ring: Vec<VertexKey> = Vec::with_capacity(u);
            for j in 0..u {
                let theta = TAU * (j as f64) / (u as f64);
                ring.push(self.create_vertex(Vec3::new(
                    (radius * sin_phi * theta.sin()) as f32,
                    (radius * sin_phi * theta.cos()) as f32,
                    (radius * cos_phi) as f32,
                )));
            }
            interior.push(ring);
        }

        // Lattice accessor: latitude `0` is the north pole, `v` the south pole,
        // and everything between is an interior ring vertex.
        let lattice = |a: usize, j: usize| -> VertexKey {
            if a == 0 {
                north
            } else if a == v {
                south
            } else {
                interior[a - 1][j]
            }
        };

        // §3 — one band per latitude, one quad per longitude. The north (`a = 0`)
        // and south (`a = v − 1`) bands dedup to the pole-fan triangles.
        //
        // §7 — at the `u = 2` flat-lens floor the two opposite-longitude faces of
        // every band land coincident (same corner *set*, opposite winding); the
        // reference operator's internal merge folds each such pair to one, so a
        // coincident-face weld keeps only the first of each pair (`F = v`). Guarded
        // to `u = 2`, where such pairs actually arise, so the well-formed path
        // builds every face.
        let weld_coincident = u == 2;
        let mut welded_sets: Vec<smallvec::SmallVec<[VertexKey; 4]>> = Vec::new();
        let mut faces: Vec<FaceKey> = Vec::with_capacity(u * v);
        for a in 0..v {
            for j in 0..u {
                let jn = (j + 1) % u;
                let corners = [
                    lattice(a + 1, j),
                    lattice(a, j),
                    lattice(a, jn),
                    lattice(a + 1, jn),
                ];
                let deduped = dedup_face_corners(&corners);
                if deduped.len() < 3 {
                    continue;
                }
                if weld_coincident {
                    let mut key = deduped.clone();
                    key.sort_unstable_by_key(|vertex| vertex.as_usize());
                    if welded_sets.contains(&key) {
                        continue;
                    }
                    welded_sets.push(key);
                }
                faces.push(self.face_create_verts(&deduped, FaceCreateFlags::NONE));
            }
        }

        // §4 — paint the optional unwrap from the LOCAL positions, before `matrix`.
        if calc_uvs {
            if let Some(uv_layer) = self.active_uv_loop_layer() {
                self.paint_sphere_uvs(&faces, north, south, uv_layer);
            }
        }

        // §2.3 — transform every generated vertex in one final pass.
        if matrix != Mat4::IDENTITY {
            for &vert in [north, south].iter().chain(interior.iter().flatten()) {
                let local = self.vertex_position(vert);
                self.set_vertex_position(vert, matrix.transform_point3(local));
            }
        }

        // §1 — recompute the new faces' then verts' normals (a run-flag post-pass),
        // over the final (transformed) geometry.
        for &face in &faces {
            self.face_calc_normal(face);
        }
        for &vert in [north, south].iter().chain(interior.iter().flatten()) {
            self.vert_normal_update(vert);
        }

        // §8.2 — verts.out: slot order is a don't-care, emit a clean
        // pole-then-rings order.
        let mut out: Vec<VertexKey> = Vec::with_capacity(u * (v - 1) + 2);
        out.push(north);
        out.push(south);
        for ring in &interior {
            out.extend_from_slice(ring);
        }
        out
    }

    /// §7 `v = 1` — the **bigon sphere**. The meridian arc is just the two poles;
    /// sweeping it produces two-sided (bigon) faces that the internal merge folds
    /// entirely against the polar axis, leaving the two poles joined by a single
    /// meridian edge and no faces — raw `(V, E, F) = (2, 1, 0)`. (The low-`u`
    /// partial-survival counts the reference emits for `u = 3/4` are a fragile
    /// merge-threshold artifact and a don't-care; the robust `u ≥ 5` full collapse
    /// is the pinned form, reproduced here for every `u`.)
    fn create_uvsphere_bigon(&mut self, radius: f32, matrix: Mat4) -> Vec<VertexKey> {
        let north = self.create_vertex(Vec3::new(0.0, 0.0, radius));
        let south = self.create_vertex(Vec3::new(0.0, 0.0, -radius));
        self.create_edge([north, south], EdgeCreateFlags::NO_DUPLICATE);

        if matrix != Mat4::IDENTITY {
            for vert in [north, south] {
                let local = self.vertex_position(vert);
                self.set_vertex_position(vert, matrix.transform_point3(local));
            }
        }
        self.vert_normal_update(north);
        self.vert_normal_update(south);

        vec![north, south]
    }

    /// §7 `u = 1` — the **collapsed meridian**. A single-longitude sweep rotates a
    /// full −2π in one step, landing the extruded copy back on the original
    /// meridian; the internal merge fuses the two poles but leaves the interior
    /// ring vertices doubled (a fragile merge artifact). The result is two
    /// coincident meridian arcs sharing the two poles — a collapsed, multi-edge
    /// shell of `(2v, 3v − 1, v)` elements: the two poles, two coincident verts per
    /// interior ring, a fan triangle at each pole, and `v − 2` collapsed quad
    /// bands. Genuinely ill-formed (§7); positions and counts are the pinned
    /// contract, the collapsed faces' winding a don't-care.
    fn create_uvsphere_collapsed_meridian(
        &mut self,
        v: usize,
        radius: f32,
        matrix: Mat4,
    ) -> Vec<VertexKey> {
        use std::f64::consts::PI;

        let radius_f64 = radius as f64;
        let north = self.create_vertex(Vec3::new(0.0, 0.0, radius));
        let south = self.create_vertex(Vec3::new(0.0, 0.0, -radius));

        // Two coincident interior verts per ring, both at longitude 0 (+y): the two
        // meridian arcs the full-turn sweep leaves superimposed.
        let mut arc_a: Vec<VertexKey> = Vec::with_capacity(v - 1);
        let mut arc_b: Vec<VertexKey> = Vec::with_capacity(v - 1);
        for a in 1..v {
            let phi = PI * (a as f64) / (v as f64);
            let position = Vec3::new(
                0.0,
                (radius_f64 * phi.sin()) as f32,
                (radius_f64 * phi.cos()) as f32,
            );
            arc_a.push(self.create_vertex(position));
            arc_b.push(self.create_vertex(position));
        }

        // Latitude `0` is the (shared) north pole, `v` the south pole, everything
        // between is that arc's interior ring vertex.
        let arc_vertex = |arc: &[VertexKey], a: usize| -> VertexKey {
            if a == 0 {
                north
            } else if a == v {
                south
            } else {
                arc[a - 1]
            }
        };

        // §3 — one collapsed band per meridian segment: the quad joining arc A's
        // edge `(a, a+1)` to arc B's, the pole segments deduping to fan triangles.
        let mut faces: Vec<FaceKey> = Vec::with_capacity(v);
        for a in 0..v {
            let corners = [
                arc_vertex(&arc_a, a),
                arc_vertex(&arc_a, a + 1),
                arc_vertex(&arc_b, a + 1),
                arc_vertex(&arc_b, a),
            ];
            let deduped = dedup_face_corners(&corners);
            if deduped.len() >= 3 {
                faces.push(self.face_create_verts(&deduped, FaceCreateFlags::NONE));
            }
        }

        if matrix != Mat4::IDENTITY {
            for &vert in [north, south].iter().chain(arc_a.iter()).chain(arc_b.iter()) {
                let local = self.vertex_position(vert);
                self.set_vertex_position(vert, matrix.transform_point3(local));
            }
        }

        for &face in &faces {
            self.face_calc_normal(face);
        }
        for &vert in [north, south].iter().chain(arc_a.iter()).chain(arc_b.iter()) {
            self.vert_normal_update(vert);
        }

        let mut out: Vec<VertexKey> = Vec::with_capacity(2 * v);
        out.push(north);
        out.push(south);
        for a in 0..(v - 1) {
            out.push(arc_a[a]);
            out.push(arc_b[a]);
        }
        out
    }

    /// §4 — paint the default equirectangular unwrap over the sphere's faces from
    /// their **local** positions. Each loop maps `longitude → u`, `latitude → v`
    /// (north pole at `v = 1`, south at `v = 0`). A pole-cap triangle's apex loop
    /// has no longitude of its own, so it takes the averaged longitude of the
    /// face's non-pole corners; a per-face wrap shift keeps a seam-straddling face
    /// from smearing; and a final pass left-justifies the whole unwrap to `u = 0`.
    fn paint_sphere_uvs(
        &mut self,
        faces: &[FaceKey],
        north: VertexKey,
        south: VertexKey,
        uv_layer: ElementAttributeKey,
    ) {
        use std::f32::consts::{PI, TAU};

        let mut painted: Vec<(LoopKey, [f32; 2])> = Vec::with_capacity(faces.len() * 4);
        let mut global_min_u = f32::INFINITY;

        let mut loops: smallvec::SmallVec<[LoopKey; 4]> = smallvec::SmallVec::new();
        // (loop, u, v, is_pole)
        let mut corners: smallvec::SmallVec<[(LoopKey, f32, f32, bool); 4]> = smallvec::SmallVec::new();

        for &face in faces {
            loops.clear();
            for loop_key in self.iter_face_loops(face) {
                loops.push(loop_key);
            }

            corners.clear();
            for &loop_key in &loops {
                let vertex = self.loop_vertex(loop_key);
                let position = self.vertex_position(vertex);
                let is_pole = vertex == north || vertex == south;

                let length = position.length();
                let cos_phi = if length > 0.0 {
                    (position.z / length).clamp(-1.0, 1.0)
                } else {
                    0.0
                };
                let v_uv = 1.0 - cos_phi.acos() / PI;

                let mut theta = position.y.atan2(position.x);
                // A loop at longitude ≈ +π is pinned to −π so the back seam reads
                // as the left edge rather than the right.
                if (theta - PI).abs() < 1e-4 {
                    theta = -PI;
                }
                let u_uv = 0.5 + theta / TAU;

                corners.push((loop_key, u_uv, v_uv, is_pole));
            }

            // Per-face wrap shift FIRST, over the rim (non-pole) corners: a corner
            // more than half the square left of the rim's right-most corner is
            // shifted right by a full 1.0. The pole apex has no longitude of its
            // own, so it is excluded from the shift and derived below from the
            // already-shifted rim — otherwise a seam-straddling pole cap's apex
            // would average un-shifted rim u's and land a full wrap to the left.
            let max_u = corners
                .iter()
                .filter(|corner| !corner.3)
                .fold(f32::NEG_INFINITY, |m, corner| m.max(corner.1));
            for corner in corners.iter_mut() {
                if !corner.3 && corner.1 < max_u - 0.5 {
                    corner.1 += 1.0;
                }
            }

            // Pole-cap apex: take the averaged longitude of the (shifted) rim.
            let mut non_pole_u_sum = 0.0f32;
            let mut non_pole_count = 0u32;
            for corner in &corners {
                if !corner.3 {
                    non_pole_u_sum += corner.1;
                    non_pole_count += 1;
                }
            }
            if non_pole_count > 0 {
                let avg_u = non_pole_u_sum / non_pole_count as f32;
                for corner in corners.iter_mut() {
                    if corner.3 {
                        corner.1 = avg_u;
                    }
                }
            }

            for &(loop_key, u_uv, v_uv, _) in &corners {
                global_min_u = global_min_u.min(u_uv);
                painted.push((loop_key, [u_uv, v_uv]));
            }
        }

        // Final left-justify: subtract the global minimum u.
        let shift = if global_min_u.is_finite() { global_min_u } else { 0.0 };
        for (loop_key, uv) in painted {
            self.loop_attribs_mut()
                .set_vec2(uv_layer, loop_key.as_usize(), [uv[0] - shift, uv[1]]);
        }
    }
}

/// Collapse consecutive (and wrap-around) duplicate vertices in a band's corner
/// list, so a pole band's quad dedups to a fan triangle (§3). The returned list
/// may still be shorter than three corners (a fully collapsed degenerate band),
/// which the caller drops.
fn dedup_face_corners(corners: &[VertexKey]) -> smallvec::SmallVec<[VertexKey; 4]> {
    let mut deduped: smallvec::SmallVec<[VertexKey; 4]> = smallvec::SmallVec::new();
    for &vertex in corners {
        if deduped.last() != Some(&vertex) {
            deduped.push(vertex);
        }
    }
    while deduped.len() >= 2 && deduped.first() == deduped.last() {
        deduped.pop();
    }
    deduped
}

#[cfg(test)]
mod tests {
    use glam::{Mat4, Quat, Vec3};

    use crate::loopmesh::*;

    const EPS: f32 = 1e-4;

    fn has_vertex_near(mesh: &LoopMesh, target: Vec3) -> bool {
        mesh.vert_keys()
            .into_iter()
            .any(|v| (mesh.vertex_position(v) - target).length() < EPS)
    }

    fn euler(mesh: &LoopMesh) -> i64 {
        mesh.vertex_count() as i64 - mesh.edge_count() as i64 + mesh.face_count() as i64
    }

    fn register_uv(mesh: &mut LoopMesh) -> ElementAttributeKey {
        mesh.register_loop_attribute(ElementAttributeDesc::numeric(
            AttributeKey::custom("uv"),
            AttrType::Vec2,
        ))
        .expect("uv layer registers")
    }

    /// §3 / §6 / §8.1 default sphere (headline): a well-formed `u × v` sphere
    /// (`u = 8`, `v = 6`) — `v − 1` interior rings, `v − 2` quad bands, a triangle
    /// fan at each pole. `ΔV = u·(v−1)+2`, `ΔE = u·(2v−1)`, `ΔF = u·v`,
    /// `ΔLoops = u·(4v−2)`, χ = 2. Exactly `2u` triangles, `u·(v−2)` quads.
    #[test]
    fn create_uvsphere_default_sphere() {
        let mut mesh = LoopMesh::new();
        let u = 8usize;
        let v = 6usize;
        let r = 2.0;

        mesh.create_uvsphere(u as i32, v as i32, r, Mat4::IDENTITY, false);

        assert_eq!(
            (mesh.vertex_count(), mesh.edge_count(), mesh.face_count()),
            (u * (v - 1) + 2, u * (2 * v - 1), u * v)
        );
        assert_eq!(mesh.loop_count(), u * (4 * v - 2));
        assert_eq!(euler(&mesh), 2, "closed sphere shell (χ = 2)");

        // Both poles present at the local axis points.
        assert!(has_vertex_near(&mesh, Vec3::new(0.0, 0.0, r)));
        assert!(has_vertex_near(&mesh, Vec3::new(0.0, 0.0, -r)));

        // Face mix: 2u triangle fans, u·(v−2) interior-band quads, every vertex
        // at distance r, every face wound outward.
        let mut triangles = 0;
        let mut quads = 0;
        for face in mesh.face_keys() {
            match mesh.face_vert_keys(face).len() {
                3 => triangles += 1,
                4 => quads += 1,
                n => panic!("unexpected face arity {n}"),
            }
            let verts = mesh.face_vert_keys(face);
            let centroid = verts
                .iter()
                .map(|&vert| mesh.vertex_position(vert))
                .fold(Vec3::ZERO, |a, b| a + b)
                / verts.len() as f32;
            assert!(
                mesh.face_normal(face).dot(centroid) > 0.0,
                "sphere faces wind outward"
            );
        }
        assert_eq!(triangles, 2 * u, "two pole fans of u triangles each");
        assert_eq!(quads, u * (v - 2), "v−2 quad bands of u quads each");

        for vert in mesh.vert_keys() {
            assert!(
                (mesh.vertex_position(vert).length() - r).abs() < EPS,
                "every vertex at distance r"
            );
        }
        mesh.validate_topology();
    }

    /// §6 / §7 / §8.1 bicone (`v = 2`): one equator ring, no quad bands — two
    /// triangle fans meeting at the equator. `ΔV = u+2`, `ΔE = 3u`, `ΔF = 2u`
    /// (all triangles), χ = 2. Paired with the `v = 6` sphere this pins the
    /// presence/absence of interior quad bands as a topological switch.
    #[test]
    fn create_uvsphere_bicone_v2() {
        let mut bicone = LoopMesh::new();
        let u = 8usize;
        let r = 1.0;

        bicone.create_uvsphere(u as i32, 2, r, Mat4::IDENTITY, false);

        assert_eq!(
            (bicone.vertex_count(), bicone.edge_count(), bicone.face_count()),
            (u + 2, 3 * u, 2 * u)
        );
        assert_eq!(euler(&bicone), 2, "bicone is a closed shell (χ = 2)");
        for face in bicone.face_keys() {
            assert_eq!(
                bicone.face_vert_keys(face).len(),
                3,
                "every bicone face is a triangle"
            );
        }

        // Contrast: a v = 6 sphere has interior quad bands, so it differs in count.
        let mut sphere = LoopMesh::new();
        sphere.create_uvsphere(u as i32, 6, r, Mat4::IDENTITY, false);
        assert_ne!(bicone.vertex_count(), sphere.vertex_count());
        assert_ne!(bicone.face_count(), sphere.face_count());
        bicone.validate_topology();
    }

    /// §1 / §7: `u_segments = 0` (or below) clamps **up to 1**, NOT a no-op (the
    /// headline divergence from create-cone's `segments = 0` build-nothing). The
    /// clamped build equals an explicit `u_segments = 1` build and is non-empty.
    #[test]
    fn create_uvsphere_u_segments_clamp_to_one() {
        let mut clamped = LoopMesh::new();
        clamped.create_uvsphere(0, 6, 1.0, Mat4::IDENTITY, false);

        let mut explicit = LoopMesh::new();
        explicit.create_uvsphere(1, 6, 1.0, Mat4::IDENTITY, false);

        assert_eq!(clamped.vertex_count(), explicit.vertex_count());
        assert_eq!(clamped.edge_count(), explicit.edge_count());
        assert_eq!(clamped.face_count(), explicit.face_count());
        assert!(clamped.vertex_count() > 0, "clamp is not a no-op");

        // A negative count clamps the same way.
        let mut negative = LoopMesh::new();
        negative.create_uvsphere(-3, 6, 1.0, Mat4::IDENTITY, false);
        assert_eq!(negative.vertex_count(), explicit.vertex_count());
        clamped.validate_topology();
    }

    /// §1 / §7: the latitude-axis clamp — `v_segments = 0` clamps to 1 and equals
    /// an explicit `v_segments = 1` build, non-empty (the second clamp axis).
    #[test]
    fn create_uvsphere_v_segments_clamp_to_one() {
        let mut clamped = LoopMesh::new();
        clamped.create_uvsphere(8, 0, 1.0, Mat4::IDENTITY, false);

        let mut explicit = LoopMesh::new();
        explicit.create_uvsphere(8, 1, 1.0, Mat4::IDENTITY, false);

        assert_eq!(clamped.vertex_count(), explicit.vertex_count());
        assert_eq!(clamped.edge_count(), explicit.edge_count());
        assert_eq!(clamped.face_count(), explicit.face_count());
        assert!(clamped.vertex_count() > 0, "clamp is not a no-op");
        clamped.validate_topology();
    }

    /// §2.1 / §8.1 radius scaling: every vertex lands at distance `radius` from
    /// the local origin, so two builds at distinct radii place their poles (and
    /// rings) at distinct positions.
    #[test]
    fn create_uvsphere_radius_scaling() {
        for &r in &[1.0_f32, 2.0_f32] {
            let mut mesh = LoopMesh::new();
            mesh.create_uvsphere(8, 6, r, Mat4::IDENTITY, false);
            assert!(has_vertex_near(&mesh, Vec3::new(0.0, 0.0, r)));
            for vert in mesh.vert_keys() {
                assert!((mesh.vertex_position(vert).length() - r).abs() < EPS);
            }
            mesh.validate_topology();
        }

        // A radius = 2 sphere puts no pole at the radius = 1 axis point.
        let mut big = LoopMesh::new();
        big.create_uvsphere(8, 6, 2.0, Mat4::IDENTITY, false);
        assert!(!has_vertex_near(&big, Vec3::new(0.0, 0.0, 1.0)));
    }

    /// §2.1: the lattice trig runs in f64 intermediates so a round-half tie at a
    /// cos φ = ½ ring quantises the same way as the higher-precision reference.
    /// For `radius = 2`, `u = 8`, `v = 6` the ring `a = 2` has cos φ = ½ (z = 1),
    /// and its diagonal verts (`θ = 45°`) sit at `x = y = 2·sin60°·sin45° =
    /// 1.22474487…`, which quantises (×100000, round-half) to 122474, not 122475.
    #[test]
    fn create_uvsphere_radius_two_cos_phi_half_ring() {
        let mut mesh = LoopMesh::new();
        mesh.create_uvsphere(8, 6, 2.0, Mat4::IDENTITY, false);

        let quantize = |c: f32| (c * 100000.0).round() as i64;
        let mut saw_diagonal = false;
        for vert in mesh.vert_keys() {
            let pos = mesh.vertex_position(vert);
            // The cos φ = ½ ring at radius 2 sits at z = 1; pick its diagonals.
            if (pos.z - 1.0).abs() < EPS && pos.x.abs() > 1.0 && pos.y.abs() > 1.0 {
                assert_eq!(quantize(pos.x.abs()), 122474, "x quantises to 122474");
                assert_eq!(quantize(pos.y.abs()), 122474, "y quantises to 122474");
                saw_diagonal = true;
            }
        }
        assert!(saw_diagonal, "found the cos φ = ½ ring's diagonal verts");
        mesh.validate_topology();
    }

    /// §2.3 / §8.1 matrix transform: a non-identity `matrix` is applied to every
    /// vertex in a final pass; the poles land at the transformed axis points and
    /// the untransformed positions do not survive.
    #[test]
    fn create_uvsphere_matrix_transform() {
        let mut mesh = LoopMesh::new();
        let r = 1.0;
        let matrix = Mat4::from_scale_rotation_translation(
            Vec3::ONE,
            Quat::from_rotation_x(std::f32::consts::FRAC_PI_3),
            Vec3::new(5.0, -2.0, 1.0),
        );

        mesh.create_uvsphere(8, 6, r, matrix, false);

        assert!(has_vertex_near(&mesh, matrix.transform_point3(Vec3::new(0.0, 0.0, r))));
        assert!(has_vertex_near(&mesh, matrix.transform_point3(Vec3::new(0.0, 0.0, -r))));
        // The untransformed pole does not survive.
        assert!(!has_vertex_near(&mesh, Vec3::new(0.0, 0.0, r)));

        // A non-identity build differs from an identity build.
        let mut identity = LoopMesh::new();
        identity.create_uvsphere(8, 6, r, Mat4::IDENTITY, false);
        assert!(has_vertex_near(&identity, Vec3::new(0.0, 0.0, r)));
        mesh.validate_topology();
    }

    /// intro / §3 / §7 / §8.1 append onto a populated mesh: the sphere is a
    /// disjoint island, prior geometry untouched, additive counts, no merge.
    #[test]
    fn create_uvsphere_appends_onto_populated_mesh() {
        let mut mesh = LoopMesh::new();
        let corners = [
            Vec3::new(10.0, 10.0, 0.0),
            Vec3::new(11.0, 10.0, 0.0),
            Vec3::new(11.0, 11.0, 0.0),
            Vec3::new(10.0, 11.0, 0.0),
        ];
        let island: Vec<VertexKey> = corners.iter().map(|&p| mesh.create_vertex(p)).collect();
        mesh.face_create_verts(&island, FaceCreateFlags::NONE);
        assert_eq!(
            (mesh.vertex_count(), mesh.edge_count(), mesh.face_count()),
            (4, 4, 1)
        );

        let u = 6usize;
        let v = 4usize;
        mesh.create_uvsphere(u as i32, v as i32, 1.0, Mat4::IDENTITY, false);

        assert_eq!(
            (mesh.vertex_count(), mesh.edge_count(), mesh.face_count()),
            (
                4 + u * (v - 1) + 2,
                4 + u * (2 * v - 1),
                1 + u * v
            )
        );
        for &p in &corners {
            assert!(has_vertex_near(&mesh, p), "prior island vertex {p:?} moved");
        }
        mesh.validate_topology();
    }

    /// §4: with `calc_uvs` and an active UV layer, the equirectangular unwrap
    /// paints longitude → `u`, latitude → `v`: north-pole loops at `v ≈ 1`,
    /// south-pole loops at `v ≈ 0`, every `u` in `[0, 1]` after the left-justify.
    /// A flag-clear contrast confirms the flag is load-bearing.
    #[test]
    fn create_uvsphere_calc_uvs_equirectangular() {
        let mut mesh = LoopMesh::new();
        let uv = register_uv(&mut mesh);
        let r = 1.0;

        mesh.create_uvsphere(8, 6, r, Mat4::IDENTITY, true);

        let north = Vec3::new(0.0, 0.0, r);
        let south = Vec3::new(0.0, 0.0, -r);
        let mut saw_north_top = false;
        let mut saw_south_bottom = false;
        let mut min_u = f32::INFINITY;
        for face in mesh.face_keys() {
            for loop_key in mesh.face_loop_keys(face) {
                let vertex = mesh.loop_vertex(loop_key);
                let pos = mesh.vertex_position(vertex);
                let [u, v] = mesh
                    .loop_attribs()
                    .get_vec2(uv, loop_key.as_usize())
                    .expect("uv painted");
                min_u = min_u.min(u);
                assert!((0.0..=1.0 + EPS).contains(&v), "v within unit square: {v}");
                if (pos - north).length() < EPS {
                    assert!((v - 1.0).abs() < EPS, "north pole loop at v = 1");
                    saw_north_top = true;
                }
                if (pos - south).length() < EPS {
                    assert!(v.abs() < EPS, "south pole loop at v = 0");
                    saw_south_bottom = true;
                }
            }
        }
        assert!(saw_north_top && saw_south_bottom, "poles unwrap to the v edges");
        assert!(min_u.abs() < EPS, "the unwrap is left-justified to u = 0");
    }

    /// §4 / §7: `calc_uvs = true` with no UV layer present is a silent no-op — it
    /// builds the topology and creates no layer.
    #[test]
    fn create_uvsphere_calc_uvs_no_layer_is_noop() {
        let mut mesh = LoopMesh::new();
        mesh.create_uvsphere(8, 6, 1.0, Mat4::IDENTITY, true);
        assert!(mesh.loop_attribs().is_empty(), "no UV layer was created");
        mesh.validate_topology();
    }

    /// §7 degenerate `u_segments = 1` — the **collapsed meridian**. A single
    /// longitude leaves two coincident meridian arcs sharing the two poles: a
    /// doubled, multi-edge shell of `(V, E, F) = (2v, 3v − 1, v)`; at `v = 6` that
    /// is `(12, 17, 6)`. The doubled interior verts and merged bands are the
    /// load-bearing degeneracy (raw counts are the pin; the collapsed faces' loop
    /// structure is a don't-care). The clamp routes `u = 0` here too.
    #[test]
    fn create_uvsphere_u_segments_one_collapsed_meridian() {
        let mut mesh = LoopMesh::new();
        let v = 6usize;
        mesh.create_uvsphere(1, v as i32, 1.0, Mat4::IDENTITY, false);

        assert_eq!(
            (mesh.vertex_count(), mesh.edge_count(), mesh.face_count()),
            (2 * v, 3 * v - 1, v),
            "u=1 collapsed meridian: doubled interior verts, merged multi-edge meridian"
        );

        // Each interior ring carries two coincident verts at longitude 0 (+y).
        for a in 1..v {
            let phi = std::f64::consts::PI * (a as f64) / (v as f64);
            let target = Vec3::new(0.0, phi.sin() as f32, phi.cos() as f32);
            let coincident = mesh
                .vert_keys()
                .into_iter()
                .filter(|&vert| (mesh.vertex_position(vert) - target).length() < EPS)
                .count();
            assert_eq!(coincident, 2, "ring a={a} carries two coincident verts");
        }

        // The clamp routes `u = 0` to the same degenerate build.
        let mut clamped = LoopMesh::new();
        clamped.create_uvsphere(0, v as i32, 1.0, Mat4::IDENTITY, false);
        assert_eq!(
            (clamped.vertex_count(), clamped.edge_count(), clamped.face_count()),
            (2 * v, 3 * v - 1, v)
        );

        // A well-formed `u = 8` build differs.
        let mut wellformed = LoopMesh::new();
        wellformed.create_uvsphere(8, v as i32, 1.0, Mat4::IDENTITY, false);
        assert_ne!(mesh.vertex_count(), wellformed.vertex_count());
        mesh.validate_topology();
    }

    /// §7 degenerate `u_segments = 2` — the **flat lens**. Two longitudes at
    /// `θ = 0` and `θ = π` form a planar lens; the coincident opposite-longitude
    /// band quads merge to `F = v`. `(V, E, F) = (2v, 3v − 1, v) = (12, 17, 6)` at
    /// `v = 6` — the same count triple as `u = 1`, but a distinct (two-column)
    /// mesh: two real longitude columns at `+y` and `−y`.
    #[test]
    fn create_uvsphere_u_segments_two_flat_lens() {
        let mut mesh = LoopMesh::new();
        let v = 6usize;
        mesh.create_uvsphere(2, v as i32, 1.0, Mat4::IDENTITY, false);

        assert_eq!(
            (mesh.vertex_count(), mesh.edge_count(), mesh.face_count()),
            (2 * v, 3 * v - 1, v),
            "u=2 flat lens: coincident opposite-longitude band quads merge to F=v"
        );

        // Two real longitude columns (unlike u=1's single collapsed column): the
        // first interior ring places one vertex at +y and one at −y.
        let phi = std::f64::consts::PI / (v as f64);
        let (s, c) = (phi.sin() as f32, phi.cos() as f32);
        assert!(has_vertex_near(&mesh, Vec3::new(0.0, s, c)), "j=0 column at +y");
        assert!(has_vertex_near(&mesh, Vec3::new(0.0, -s, c)), "j=1 column at -y");

        // The lens keeps half the well-formed `u·v` faces (the coincident pairs
        // welded), so it differs from an unwelded double count.
        assert_eq!(mesh.face_count(), v);
        mesh.validate_topology();
    }

    /// §7 degenerate `v_segments = 1` — the **bigon sphere**. One latitude segment
    /// leaves the two poles joined by a single meridian edge; the sweep's bigon
    /// faces fold entirely against the polar axis under the merge. The robust
    /// full collapse (`u ≥ 5`) is `(V, E, F) = (2, 1, 0)`. The clamp routes
    /// `v = 0` here too.
    #[test]
    fn create_uvsphere_v_segments_one_bigon_sphere() {
        let mut mesh = LoopMesh::new();
        mesh.create_uvsphere(6, 1, 1.0, Mat4::IDENTITY, false);

        assert_eq!(
            (mesh.vertex_count(), mesh.edge_count(), mesh.face_count()),
            (2, 1, 0),
            "v=1 bigon sphere: two poles joined by a single meridian edge, no faces"
        );
        assert!(has_vertex_near(&mesh, Vec3::new(0.0, 0.0, 1.0)));
        assert!(has_vertex_near(&mesh, Vec3::new(0.0, 0.0, -1.0)));

        // The clamp routes `v = 0` to the same collapse.
        let mut clamped = LoopMesh::new();
        clamped.create_uvsphere(6, 0, 1.0, Mat4::IDENTITY, false);
        assert_eq!(
            (clamped.vertex_count(), clamped.edge_count(), clamped.face_count()),
            (2, 1, 0)
        );
        mesh.validate_topology();
    }
}
