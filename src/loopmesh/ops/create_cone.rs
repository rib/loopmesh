use glam::{Mat4, Vec3};

use crate::loopmesh::*;

/// §4 — the default cap-disc unwrap places the top cap near `(0.25, 0.25)` and
/// the bottom cap near `(0.75, 0.25)`, each inscribed in a disc of UV radius
/// `UV_DISC_RADIUS`. (CREATE_CONE.md §4 states these as "near"/"≈" values.)
const UV_DISC_RADIUS: f32 = 0.24;
const TOP_CAP_CENTRE: [f32; 2] = [0.25, 0.25];
const BOTTOM_CAP_CENTRE: [f32; 2] = [0.75, 0.25];

impl LoopMesh {
    /// Lay down a **cone, cylinder, or truncated cone** — two parallel rings of
    /// `segments` verts (a bottom ring of radius `radius1` at local
    /// `z = -depth/2` and a top ring of radius `radius2` at `z = +depth/2`),
    /// joined by a wall of side faces, with each ring optionally filled by a cap
    /// — and append it to the mesh (CREATE_CONE.md §2 / §3).
    ///
    /// Ring vertex `a` (for `0 ≤ a < segments`) is placed at angle
    /// `θ = 2π·a/segments` at local `(+radius·sin θ, +radius·cos θ, ∓depth/2)`, so
    /// vertex `0` sits on the `+y` axis and the rings are traced **clockwise** as
    /// seen from `+z` (§2.1). Note the `+sin` sign — this is the *opposite* of
    /// create-circle's counter-clockwise ring, and is what winds the side wall
    /// and caps outward. Every vertex is mapped through `matrix` as a homogeneous
    /// point at creation (§2.4).
    ///
    /// The side wall is `segments` quads `(b_{a-1}, t_{a-1}, t_a, b_a)` joining
    /// adjacent rungs, wound radially outward (§3). When `cap_ends` is set, each
    /// open ring is filled: a single n-gon (`cap_tris = false`) built straight
    /// from the ring, or a triangle fan from that ring's centre apex
    /// (`cap_tris = true`) — `segments` triangles per cap (§3). `cap_tris` affects
    /// only the caps; the wall is never triangulated by it.
    ///
    /// A ring whose radius is `0` **collapses to a single apex vertex** (§2.3): a
    /// true cone with its point at that end. The side faces on that end become
    /// triangles meeting at the apex, and that end carries **no** cap (a cone's
    /// cap appears only on the open end, §3). Equal non-zero radii give a
    /// cylinder; distinct non-zero radii a truncated cone (frustum). This
    /// implementation builds the collapsed end directly — one apex, fanned
    /// triangles — the §2.3-blessed equivalent of the build-then-collapse route.
    ///
    /// Create-cone is purely constructive and **additive**: it reads no existing
    /// geometry, never clears the mesh, and never merges into prior verts (§ intro
    /// / §7). `segments = 0` is a **no-op** — nothing is built (§7).
    ///
    /// When `calc_uvs` is set **and** the mesh carries an active per-loop
    /// 2-component-float (UV) layer, a default unwrap is painted across all the
    /// new faces (§4): the side wall as a horizontal `segments`-column strip (the
    /// whole UV square when uncapped, its top half when capped) and each cap as a
    /// radial disc in the bottom half. Unlike create-circle, the side wall is
    /// unwrapped even when uncapped. With `calc_uvs` clear or no UV layer present,
    /// the flag is a silent no-op — it never mints a layer.
    ///
    /// The created vertices are **returned** in the `verts.out` slot order
    /// (§1 / §8): the two centre vertices first (when capped with a surviving
    /// centre — the fan caps), then the ring vertices in segment order, each
    /// segment contributing its bottom vertex then its top vertex. A collapsed
    /// ring contributes its single apex once. A `segments = 0` no-op returns an
    /// empty buffer.
    ///
    /// See `radial-mesh-guides/operators/CREATE_CONE.md` for the full contract.
    pub fn create_cone(
        &mut self,
        cap_ends: bool,
        cap_tris: bool,
        segments: i32,
        radius1: f32,
        radius2: f32,
        depth: f32,
        matrix: Mat4,
        calc_uvs: bool,
    ) -> Vec<VertexKey> {
        // §7 — `segments = 0` (or any non-positive value) builds nothing.
        if segments <= 0 {
            return Vec::new();
        }
        let segment_count = segments as usize;
        let half_depth = 0.5 * depth;

        // §2.3 — a zero-radius ring collapses to a single apex vertex.
        let bottom_apex = radius1 == 0.0;
        let top_apex = radius2 == 0.0;

        // §2.2 — fan-cap centres survive only on a non-collapsed, fan-capped end.
        // The n-gon path builds the cap straight from the ring (no centre, the
        // §3-blessed equivalent of fan-then-dissolve); a collapsed end has no cap.
        //
        // §7 — over a degenerate ring (`segments < 3`) the fan-then-dissolve that
        // would flatten a fan into an n-gon cannot operate (a 1- or 2-vertex ring
        // has no interior to dissolve), so a degenerate n-gon cap stays a fan and
        // its centre survives. Route those through the fan builder too, matching
        // the ill-formed degenerate output the kernel accepts (the editor clamps
        // to `segments >= 3`, the kernel does not).
        let want_fan = cap_ends && (cap_tris || segment_count < 3);
        let centre_bottom = if want_fan && !bottom_apex {
            Some(self.create_vertex(matrix.transform_point3(Vec3::new(0.0, 0.0, -half_depth))))
        } else {
            None
        };
        let centre_top = if want_fan && !top_apex {
            Some(self.create_vertex(matrix.transform_point3(Vec3::new(0.0, 0.0, half_depth))))
        } else {
            None
        };

        // §2.3 — for a collapsed end, mint the single shared apex once.
        let bottom_apex_vert = if bottom_apex {
            Some(self.create_vertex(matrix.transform_point3(Vec3::new(0.0, 0.0, -half_depth))))
        } else {
            None
        };
        let top_apex_vert = if top_apex {
            Some(self.create_vertex(matrix.transform_point3(Vec3::new(0.0, 0.0, half_depth))))
        } else {
            None
        };

        // §2.1 — lay both rings clockwise from +z (the +sin convention). A
        // collapsed ring reuses its shared apex for every angular slot.
        let mut bottom: Vec<VertexKey> = Vec::with_capacity(segment_count);
        let mut top: Vec<VertexKey> = Vec::with_capacity(segment_count);
        for a in 0..segment_count {
            let theta = std::f32::consts::TAU * (a as f32) / (segment_count as f32);
            let (sin, cos) = (theta.sin(), theta.cos());
            let b = match bottom_apex_vert {
                Some(apex) => apex,
                None => self.create_vertex(matrix.transform_point3(Vec3::new(
                    radius1 * sin,
                    radius1 * cos,
                    -half_depth,
                ))),
            };
            let t = match top_apex_vert {
                Some(apex) => apex,
                None => self.create_vertex(matrix.transform_point3(Vec3::new(
                    radius2 * sin,
                    radius2 * cos,
                    half_depth,
                ))),
            };
            bottom.push(b);
            top.push(t);
        }

        // §3 — the side wall: `segments` faces `(b_{a-1}, t_{a-1}, t_a, b_a)`,
        // wound radially outward. A collapsed end's corners coincide, so the quad
        // dedups to a triangle meeting the apex. `seg` runs 1..=n, with `seg = n`
        // the closing face that joins the last rung back to the first; its `cur`
        // rung index stays `n` (UV column 0) while the vertex wraps to slot 0.
        //
        // §7 — a `segments = 1` wall quad is `(b_0, t_0, t_0, b_0)`; consecutive
        // dedup leaves `(b_0, t_0, b_0)`. Its wrap-around `b_0 == b_0` is the
        // ring's self-loop closure: kept for an **open** (uncapped) ring, whose
        // boundary *is* that ring edge, and stripped to a doubled-edge 2-gon for a
        // **capped** ring, whose closure is carried by the cap. That is why the
        // minimum wall corner count depends on `cap_ends`.
        let wall_min_corners = if cap_ends { 2 } else { 3 };
        let mut side_faces: Vec<(FaceKey, usize, usize)> = Vec::with_capacity(segment_count);
        for seg in 1..=segment_count {
            let prev = seg - 1;
            let cur = seg;
            let corners = [
                bottom[prev],
                top[prev],
                top[cur % segment_count],
                bottom[cur % segment_count],
            ];
            if let Some(face) = self.cone_face_create(&corners, wall_min_corners) {
                side_faces.push((face, prev, cur));
            }
        }

        // §3 — the caps. Fan caps fan from the surviving centre; n-gon caps are
        // built straight from the ring. A collapsed end gets neither.
        let mut bottom_cap_faces: Vec<FaceKey> = Vec::new();
        let mut top_cap_faces: Vec<FaceKey> = Vec::new();
        if cap_ends {
            if !bottom_apex {
                if let Some(centre) = centre_bottom {
                    // §3 bottom fan triangle: (centre, b_{a-1}, b_a) — faces -z.
                    // §7 — over a single-vertex ring this pinches to the doubled-
                    // spoke 2-gon (centre, b_0); `min_corners = 2` admits it.
                    for seg in 1..=segment_count {
                        let corners = [centre, bottom[seg - 1], bottom[seg % segment_count]];
                        if let Some(face) = self.cone_face_create(&corners, 2) {
                            bottom_cap_faces.push(face);
                        }
                    }
                } else {
                    // §3 bottom n-gon — the ring in order faces -z.
                    if let Some(face) = self.cone_face_create(&bottom, 3) {
                        bottom_cap_faces.push(face);
                    }
                }
            }
            if !top_apex {
                if let Some(centre) = centre_top {
                    // §3 top fan triangle: (centre, t_a, t_{a-1}) — reversed, faces +z.
                    // §7 — pinches to the doubled-spoke 2-gon over a 1-vertex ring.
                    for seg in 1..=segment_count {
                        let corners = [centre, top[seg % segment_count], top[seg - 1]];
                        if let Some(face) = self.cone_face_create(&corners, 2) {
                            top_cap_faces.push(face);
                        }
                    }
                } else {
                    // §3 top n-gon — the ring reversed faces +z.
                    let reversed: Vec<VertexKey> = top.iter().rev().copied().collect();
                    if let Some(face) = self.cone_face_create(&reversed, 3) {
                        top_cap_faces.push(face);
                    }
                }
            }
        }

        // §1 — recompute the new faces' then verts' normals (a run-flag post-pass).
        for &(face, _, _) in &side_faces {
            self.face_calc_normal(face);
        }
        for &face in bottom_cap_faces.iter().chain(top_cap_faces.iter()) {
            self.face_calc_normal(face);
        }
        for vert in centre_bottom
            .into_iter()
            .chain(centre_top)
            .chain(bottom_apex_vert)
            .chain(top_apex_vert)
            .chain(bottom.iter().copied())
            .chain(top.iter().copied())
        {
            self.vert_normal_update(vert);
        }

        // §4 — optional default unwrap, painted only when the flag is set and an
        // active per-loop UV layer already exists (the flag never mints a layer).
        if calc_uvs {
            if let Some(uv_layer) = self.active_uv_loop_layer() {
                self.paint_cone_uvs(
                    cap_ends,
                    radius1,
                    radius2,
                    matrix,
                    segment_count,
                    &bottom,
                    &top,
                    &side_faces,
                    &bottom_cap_faces,
                    &top_cap_faces,
                    uv_layer,
                );
            }
        }

        // §1 / §8 — verts.out: the surviving centres first, then the ring verts in
        // segment order (each segment's bottom then top), a collapsed ring's apex
        // contributed once.
        let mut out: Vec<VertexKey> = Vec::with_capacity(2 + 2 * segment_count);
        out.extend(centre_bottom);
        out.extend(centre_top);
        let mut emitted_bottom_apex = false;
        let mut emitted_top_apex = false;
        for a in 0..segment_count {
            if bottom_apex {
                if !emitted_bottom_apex {
                    out.push(bottom[a]);
                    emitted_bottom_apex = true;
                }
            } else {
                out.push(bottom[a]);
            }
            if top_apex {
                if !emitted_top_apex {
                    out.push(top[a]);
                    emitted_top_apex = true;
                }
            } else {
                out.push(top[a]);
            }
        }
        out
    }

    /// Build a face from `corners`, collapsing consecutive duplicate vertices
    /// first, then stripping wrap-around duplicates down to `min_corners`.
    ///
    /// A collapsed-ring side quad (`radius = 0`, §2.3) dedups to a triangle: its
    /// `(apex, t_{a-1}, t_a, apex)` corners strip the wrap `apex == apex` down to
    /// the three distinct corners. A degenerate low-`segments` face (§7) keeps
    /// more: `min_corners = 3` retains a single-vertex ring's self-loop closure
    /// (the `segments = 1` self-edge on an **open** wall), while `min_corners = 2`
    /// lets a capped wall or a pinched cap fan collapse to a doubled-edge 2-gon.
    ///
    /// Returns `None` (building nothing) only when fewer than two distinct corners
    /// survive — the doubly-degenerate spindle (both radii `0`, §7 don't-care).
    fn cone_face_create(&mut self, corners: &[VertexKey], min_corners: usize) -> Option<FaceKey> {
        let mut deduped: Vec<VertexKey> = Vec::with_capacity(corners.len());
        for &v in corners {
            if deduped.last() != Some(&v) {
                deduped.push(v);
            }
        }
        while deduped.len() > min_corners && deduped.first() == deduped.last() {
            deduped.pop();
        }
        if deduped.len() < 2 {
            return None;
        }
        Some(self.face_create_verts(&deduped, FaceCreateFlags::NONE))
    }

    /// §4 — paint the default cone unwrap. The side wall maps to a horizontal
    /// `segment_count`-column strip (the whole UV square uncapped, the top half
    /// `v ∈ [0.5, 1]` capped); each cap maps to a radial disc in the bottom half.
    /// A cone's collapsed-end side faces (one radius `0`) route through the radial
    /// branch rather than the strip (§4 quirk) — a best-effort, unpinned path.
    #[allow(clippy::too_many_arguments)]
    fn paint_cone_uvs(
        &mut self,
        cap_ends: bool,
        radius1: f32,
        radius2: f32,
        matrix: Mat4,
        segment_count: usize,
        bottom: &[VertexKey],
        top: &[VertexKey],
        side_faces: &[(FaceKey, usize, usize)],
        bottom_cap_faces: &[FaceKey],
        top_cap_faces: &[FaceKey],
        uv_layer: ElementAttributeKey,
    ) {
        let inverse = matrix.inverse();
        let both_radii_nonzero = radius1 != 0.0 && radius2 != 0.0;
        let v_bottom = if cap_ends { 0.5 } else { 0.0 };
        let v_top = 1.0;

        if both_radii_nonzero {
            // §4 strip — rung index `r` maps to column `u = 1 - r/n`; the bottom
            // corner of each rung to the lower strip edge, the top corner to the
            // upper edge. Each side corner's UV is keyed by its (rung, ring half),
            // so it is fixed at face-build time, not by raw position — the seam
            // rung carries `u = 1` on one quad and `u = 0` on the wrap quad.
            let column = |rung: usize| 1.0 - (rung as f32) / (segment_count as f32);
            let mut corner_uv: smallvec::SmallVec<[(VertexKey, [f32; 2]); 4]> =
                smallvec::SmallVec::new();
            for &(face, prev, cur) in side_faces {
                let u_prev = column(prev);
                let u_cur = column(cur);
                corner_uv.clear();
                corner_uv.push((bottom[prev], [u_prev, v_bottom]));
                corner_uv.push((top[prev], [u_prev, v_top]));
                corner_uv.push((top[cur % segment_count], [u_cur, v_top]));
                corner_uv.push((bottom[cur % segment_count], [u_cur, v_bottom]));

                let loops: smallvec::SmallVec<[LoopKey; 4]> = self.iter_face_loops(face).collect();
                for loop_key in loops {
                    let vertex = self.loop_vertex(loop_key);
                    if let Some(&(_, uv)) = corner_uv.iter().find(|&&(v, _)| v == vertex) {
                        self.loop_attribs_mut()
                            .set_vec2(uv_layer, loop_key.as_usize(), uv);
                    }
                }
            }
        }

        // Caps as radial discs (§4): each cap loop's local x/y (scaled by
        // `UV_DISC_RADIUS / radius`, with the §4 fallback to the opposite ring's
        // radius) offset to that cap's centre.
        let bottom_scale = UV_DISC_RADIUS / nonzero_radius(radius1, radius2);
        let top_scale = UV_DISC_RADIUS / nonzero_radius(radius2, radius1);
        self.paint_cone_cap_disc(
            bottom_cap_faces,
            &inverse,
            bottom_scale,
            BOTTOM_CAP_CENTRE,
            uv_layer,
        );
        self.paint_cone_cap_disc(top_cap_faces, &inverse, top_scale, TOP_CAP_CENTRE, uv_layer);

        // §4 quirk — a cone's collapsed-end side faces route through the radial
        // branch rather than the strip (unpinned, best-effort).
        if !both_radii_nonzero {
            let (scale, centre) = if radius2 != 0.0 {
                (UV_DISC_RADIUS / radius2, TOP_CAP_CENTRE)
            } else if radius1 != 0.0 {
                (UV_DISC_RADIUS / radius1, BOTTOM_CAP_CENTRE)
            } else {
                (UV_DISC_RADIUS, TOP_CAP_CENTRE)
            };
            let faces: Vec<FaceKey> = side_faces.iter().map(|&(f, _, _)| f).collect();
            self.paint_cone_cap_disc(&faces, &inverse, scale, centre, uv_layer);
        }
    }

    fn paint_cone_cap_disc(
        &mut self,
        faces: &[FaceKey],
        inverse: &Mat4,
        scale: f32,
        centre: [f32; 2],
        uv_layer: ElementAttributeKey,
    ) {
        let mut loop_keys: smallvec::SmallVec<[LoopKey; 8]> = smallvec::SmallVec::new();
        for &face in faces {
            loop_keys.clear();
            for loop_key in self.iter_face_loops(face) {
                loop_keys.push(loop_key);
            }
            for &loop_key in &loop_keys {
                let vertex = self.loop_vertex(loop_key);
                let local = inverse.transform_point3(self.vertex_position(vertex));
                let uv = [centre[0] + scale * local.x, centre[1] + scale * local.y];
                self.loop_attribs_mut()
                    .set_vec2(uv_layer, loop_key.as_usize(), uv);
            }
        }
    }
}

fn nonzero_radius(primary: f32, fallback: f32) -> f32 {
    if primary != 0.0 {
        primary
    } else if fallback != 0.0 {
        fallback
    } else {
        UV_DISC_RADIUS
    }
}

#[cfg(test)]
mod tests {
    use glam::{Mat4, Quat, Vec3};

    use crate::loopmesh::*;

    const EPS: f32 = 1e-5;

    fn has_vertex_near(mesh: &LoopMesh, target: Vec3) -> bool {
        mesh.vert_keys()
            .into_iter()
            .any(|v| (mesh.vertex_position(v) - target).length() < EPS)
    }

    fn euler(mesh: &LoopMesh) -> i64 {
        mesh.vertex_count() as i64 - mesh.edge_count() as i64 + mesh.face_count() as i64
    }

    /// §3 / §6 / §8.1 uncapped headline: equal non-zero radii, `cap_ends = false`
    /// — an open tube of `2n` verts, `3n` edges, `n` quad faces, χ = 0. The ring
    /// verts wind **clockwise** from `+z` (the `+sin` convention): the quarter-turn
    /// vertex lands at `(+r, 0, ∓h)`, the opposite of create-circle.
    #[test]
    fn create_cone_uncapped_cylinder() {
        let mut mesh = LoopMesh::new();
        let n = 8usize;
        let r = 2.0;
        let depth = 3.0;
        let h = depth / 2.0;

        mesh.create_cone(false, false, n as i32, r, r, depth, Mat4::IDENTITY, false);

        assert_eq!(
            (mesh.vertex_count(), mesh.edge_count(), mesh.face_count()),
            (2 * n, 3 * n, n)
        );
        assert_eq!(mesh.loop_count(), 4 * n);
        assert_eq!(euler(&mesh), 0, "open tube is an annulus (χ = 0)");

        // Vertex 0 on +y at both ring heights.
        assert!(has_vertex_near(&mesh, Vec3::new(0.0, r, -h)));
        assert!(has_vertex_near(&mesh, Vec3::new(0.0, r, h)));
        // Quarter-turn vertex (a = n/4) at +x — the clockwise (+sin) sign.
        assert!(has_vertex_near(&mesh, Vec3::new(r, 0.0, -h)));
        assert!(has_vertex_near(&mesh, Vec3::new(r, 0.0, h)));

        // Every side face is a quad whose normal points radially outward.
        for face in mesh.face_keys() {
            assert_eq!(mesh.face_vert_keys(face).len(), 4, "side faces are quads");
            let normal = mesh.face_normal(face);
            let centroid: Vec3 = {
                let verts = mesh.face_vert_keys(face);
                let sum: Vec3 = verts
                    .iter()
                    .map(|&v| mesh.vertex_position(v))
                    .fold(Vec3::ZERO, |a, b| a + b);
                sum / verts.len() as f32
            };
            let radial = Vec3::new(centroid.x, centroid.y, 0.0);
            assert!(normal.dot(radial) > 0.0, "side face faces outward");
        }
        mesh.validate_topology();
    }

    /// §3 / §6 / §8.1 n-gon caps: `cap_ends = true, cap_tris = false` — wall quads
    /// plus one n-gon cap per end, no interior centre vertices. `ΔV = 2n, ΔE = 3n,
    /// ΔF = n + 2, ΔLoops = 6n`, χ = 2. Bottom cap faces `-z`, top cap `+z`.
    #[test]
    fn create_cone_ngon_capped_cylinder() {
        let mut mesh = LoopMesh::new();
        let n = 6usize;
        let r = 1.5;
        let depth = 2.0;

        mesh.create_cone(true, false, n as i32, r, r, depth, Mat4::IDENTITY, false);

        assert_eq!(
            (mesh.vertex_count(), mesh.edge_count(), mesh.face_count()),
            (2 * n, 3 * n, n + 2)
        );
        assert_eq!(mesh.loop_count(), 6 * n);
        assert_eq!(euler(&mesh), 2, "n-gon-capped solid is a sphere (χ = 2)");
        // No interior centre vertex survives the n-gon path.
        assert!(!has_vertex_near(&mesh, Vec3::new(0.0, 0.0, depth / 2.0)));
        assert!(!has_vertex_near(&mesh, Vec3::new(0.0, 0.0, -depth / 2.0)));

        // Exactly two n-gon cap faces, one facing -z, one +z.
        let mut saw_bottom_cap = false;
        let mut saw_top_cap = false;
        for face in mesh.face_keys() {
            if mesh.face_vert_keys(face).len() == n {
                let nz = mesh.face_normal(face).z;
                if nz < -0.9 {
                    saw_bottom_cap = true;
                }
                if nz > 0.9 {
                    saw_top_cap = true;
                }
            }
        }
        assert!(saw_bottom_cap && saw_top_cap, "one cap each facing ∓z");
        mesh.validate_topology();
    }

    /// §3 / §6 / §8.1 fan caps (`cap_tris` contrast): `cap_tris = true` — `n`
    /// triangles per cap around each ring's centre apex. `ΔV = 2n + 2, ΔE = 5n,
    /// ΔF = 3n, ΔLoops = 10n`. Paired with the n-gon case this pins `cap_tris` as a
    /// genuine topological switch: the two capped forms differ in every count.
    #[test]
    fn create_cone_fan_capped_cylinder() {
        let mut fan = LoopMesh::new();
        let n = 6usize;
        let r = 1.5;
        let depth = 2.0;

        fan.create_cone(true, true, n as i32, r, r, depth, Mat4::IDENTITY, false);

        assert_eq!(
            (fan.vertex_count(), fan.edge_count(), fan.face_count()),
            (2 * n + 2, 5 * n, 3 * n)
        );
        assert_eq!(fan.loop_count(), 10 * n);
        assert_eq!(euler(&fan), 2);
        // Both centre apexes survive.
        assert!(has_vertex_near(&fan, Vec3::new(0.0, 0.0, depth / 2.0)));
        assert!(has_vertex_near(&fan, Vec3::new(0.0, 0.0, -depth / 2.0)));

        // Contrast: the n-gon cap of the same rings differs in every count.
        let mut ngon = LoopMesh::new();
        ngon.create_cone(true, false, n as i32, r, r, depth, Mat4::IDENTITY, false);
        assert_ne!(fan.vertex_count(), ngon.vertex_count());
        assert_ne!(fan.edge_count(), ngon.edge_count());
        assert_ne!(fan.face_count(), ngon.face_count());
        fan.validate_topology();
    }

    /// §1 / §2.1 / §8.1 truncated frustum: distinct non-zero radii — same counts as
    /// the cylinder but the two rings sit at different radii, so a frustum's
    /// vertex set differs from a same-`radius1` cylinder's.
    #[test]
    fn create_cone_truncated_frustum() {
        let n = 8usize;
        let depth = 2.0;
        let h = depth / 2.0;

        let mut frustum = LoopMesh::new();
        frustum.create_cone(
            false,
            false,
            n as i32,
            2.0,
            1.0,
            depth,
            Mat4::IDENTITY,
            false,
        );
        assert_eq!(
            (
                frustum.vertex_count(),
                frustum.edge_count(),
                frustum.face_count()
            ),
            (2 * n, 3 * n, n)
        );
        // Bottom ring at radius 2, top ring at radius 1.
        assert!(has_vertex_near(&frustum, Vec3::new(0.0, 2.0, -h)));
        assert!(has_vertex_near(&frustum, Vec3::new(0.0, 1.0, h)));

        // A same-`radius1` cylinder has its top ring at radius 2, not 1.
        let mut cylinder = LoopMesh::new();
        cylinder.create_cone(
            false,
            false,
            n as i32,
            2.0,
            2.0,
            depth,
            Mat4::IDENTITY,
            false,
        );
        assert!(!has_vertex_near(&cylinder, Vec3::new(0.0, 1.0, h)));
        frustum.validate_topology();
    }

    /// §2.3 / §6 / §8.1 true cone (apex collapse): `radius1 = 0` collapses the
    /// bottom ring to a single apex, its side faces become triangles. Uncapped:
    /// `ΔV = n + 1, ΔE = 2n, ΔF = n`, χ = 1 (disc). The apex sits at the bottom
    /// axis point; the cone differs from the frustum in every count.
    #[test]
    fn create_cone_true_cone_apex_collapse() {
        let mut cone = LoopMesh::new();
        let n = 8usize;
        let depth = 2.0;
        let h = depth / 2.0;

        cone.create_cone(
            false,
            false,
            n as i32,
            0.0,
            1.0,
            depth,
            Mat4::IDENTITY,
            false,
        );

        assert_eq!(
            (cone.vertex_count(), cone.edge_count(), cone.face_count()),
            (n + 1, 2 * n, n)
        );
        assert_eq!(cone.loop_count(), 3 * n);
        assert_eq!(euler(&cone), 1, "open cone is a disc (χ = 1)");
        // The single apex at the bottom axis point.
        assert!(has_vertex_near(&cone, Vec3::new(0.0, 0.0, -h)));
        // Every side face is a triangle (the collapsed-ring quad lost a corner).
        for face in cone.face_keys() {
            assert_eq!(
                cone.face_vert_keys(face).len(),
                3,
                "cone side faces are tris"
            );
        }

        // A regression that left the bottom ring uncollapsed would be a frustum:
        // 2n verts, not n + 1.
        let mut frustum = LoopMesh::new();
        frustum.create_cone(
            false,
            false,
            n as i32,
            1.0,
            1.0,
            depth,
            Mat4::IDENTITY,
            false,
        );
        assert_ne!(cone.vertex_count(), frustum.vertex_count());
        cone.validate_topology();
    }

    /// §2.3 / §3 / §6: a capped cone carries a cap only on its **open** (top) end —
    /// the collapsed (bottom) end is a point. n-gon-capped: `ΔV = n + 1, ΔE = 2n,
    /// ΔF = n + 1` (the `n` triangle wall + one top cap), χ = 2.
    #[test]
    fn create_cone_capped_cone_single_cap() {
        let mut cone = LoopMesh::new();
        let n = 8usize;
        let depth = 2.0;

        cone.create_cone(
            true,
            false,
            n as i32,
            0.0,
            1.0,
            depth,
            Mat4::IDENTITY,
            false,
        );

        assert_eq!(
            (cone.vertex_count(), cone.edge_count(), cone.face_count()),
            (n + 1, 2 * n, n + 1)
        );
        assert_eq!(euler(&cone), 2, "capped cone is a closed solid (χ = 2)");
        // Exactly one n-gon cap face, facing +z (the open top end). No -z cap.
        let mut ngon_caps = 0;
        for face in cone.face_keys() {
            if cone.face_vert_keys(face).len() == n {
                ngon_caps += 1;
                assert!(
                    cone.face_normal(face).z > 0.9,
                    "the only cap is the top (+z)"
                );
            }
        }
        assert_eq!(ngon_caps, 1, "a cone caps only its open end");
        cone.validate_topology();
    }

    /// §7 degenerate `segments = 1`, uncapped: the bare kernel does not clamp to
    /// `segments >= 3`. A single-vertex ring closes on itself, so the wall quad
    /// `(b_0, t_0, t_0, b_0)` collapses to `(b_0, t_0, b_0)` — one rung edge plus
    /// the bottom ring's **self-loop** closure. Raw `(V, E, F) = (2, 2, 1)`; the
    /// self-edge is the load-bearing degeneracy (CanonicalMesh dedups edges and is
    /// blind to it, so the raw counts are the pin).
    #[test]
    fn create_cone_segments_one_uncapped_self_edge() {
        let mut mesh = LoopMesh::new();
        mesh.create_cone(false, false, 1, 1.0, 1.0, 2.0, Mat4::IDENTITY, false);
        assert_eq!(
            (mesh.vertex_count(), mesh.edge_count(), mesh.face_count()),
            (2, 2, 1),
            "segments = 1 uncapped: rung + one self-loop ring edge, one degenerate side quad"
        );
        assert!(has_vertex_near(&mesh, Vec3::new(0.0, 1.0, -1.0)));
        assert!(has_vertex_near(&mesh, Vec3::new(0.0, 1.0, 1.0)));
        mesh.validate_topology();
    }

    /// §7 degenerate `segments = 1`, n-gon capped: the fan-then-dissolve that would
    /// flatten a cap fan into an n-gon cannot operate over a single-vertex ring, so
    /// each cap stays a pinched fan and its centre survives. The wall's ring
    /// closure is carried by the caps, so the wall degenerates to a doubled-edge
    /// 2-gon (no self-loop). Raw `(V, E, F) = (4, 3, 3)`: two ring verts + two
    /// surviving cap centres; the rung + the two doubled cap spokes; the wall +
    /// two pinched cap 2-gons.
    #[test]
    fn create_cone_segments_one_ngon_capped_degenerate() {
        let mut mesh = LoopMesh::new();
        mesh.create_cone(true, false, 1, 1.0, 1.0, 2.0, Mat4::IDENTITY, false);
        assert_eq!(
            (mesh.vertex_count(), mesh.edge_count(), mesh.face_count()),
            (4, 3, 3),
            "segments = 1 n-gon capped: ring verts + surviving cap centres, doubled spokes"
        );
        // Both cap centres survive on the degenerate ring.
        assert!(has_vertex_near(&mesh, Vec3::new(0.0, 0.0, -1.0)));
        assert!(has_vertex_near(&mesh, Vec3::new(0.0, 0.0, 1.0)));
        mesh.validate_topology();
    }

    /// §7 degenerate `segments = 2`, uncapped: doubled ring edges merge into a
    /// 4-cycle and the two side quads coincide (oppositely wound, zero-area). Raw
    /// `(V, E, F) = (4, 4, 2)`: four ring verts, two rungs + one merged top + one
    /// merged bottom ring edge, two coincident side quads.
    #[test]
    fn create_cone_segments_two_uncapped_doubled_edges() {
        let mut mesh = LoopMesh::new();
        mesh.create_cone(false, false, 2, 1.0, 1.0, 2.0, Mat4::IDENTITY, false);
        assert_eq!(
            (mesh.vertex_count(), mesh.edge_count(), mesh.face_count()),
            (4, 4, 2),
            "segments = 2 uncapped: 4-cycle of doubled ring edges, two coincident side quads"
        );
        mesh.validate_topology();
    }

    /// §7 degenerate `segments = 2`, n-gon capped: the caps stay fans (dissolve
    /// cannot flatten a 2-vertex ring) with surviving centres. Raw
    /// `(V, E, F) = (6, 8, 6)`: four ring verts + two centres; two rungs + two
    /// merged ring edges + two bottom spokes + two top spokes; two side quads +
    /// two bottom cap tris + two top cap tris.
    #[test]
    fn create_cone_segments_two_ngon_capped_degenerate() {
        let mut mesh = LoopMesh::new();
        mesh.create_cone(true, false, 2, 1.0, 1.0, 2.0, Mat4::IDENTITY, false);
        assert_eq!(
            (mesh.vertex_count(), mesh.edge_count(), mesh.face_count()),
            (6, 8, 6),
            "segments = 2 n-gon capped: fan caps with surviving centres over the doubled ring"
        );
        assert!(has_vertex_near(&mesh, Vec3::new(0.0, 0.0, -1.0)));
        assert!(has_vertex_near(&mesh, Vec3::new(0.0, 0.0, 1.0)));
        mesh.validate_topology();
    }

    /// §1 / §2.1 / §8.1 depth separation: the rings sit at `z = ±depth/2`, so two
    /// builds at distinct depths place their rings at distinct heights.
    #[test]
    fn create_cone_depth_separation() {
        let n = 6usize;
        let r = 1.0;
        for &depth in &[1.0_f32, 3.0_f32] {
            let mut mesh = LoopMesh::new();
            mesh.create_cone(false, false, n as i32, r, r, depth, Mat4::IDENTITY, false);
            assert!(has_vertex_near(&mesh, Vec3::new(0.0, r, -depth / 2.0)));
            assert!(has_vertex_near(&mesh, Vec3::new(0.0, r, depth / 2.0)));
            mesh.validate_topology();
        }
        // depth = 3 puts no ring at the depth = 1 height.
        let mut deep = LoopMesh::new();
        deep.create_cone(false, false, n as i32, r, r, 3.0, Mat4::IDENTITY, false);
        assert!(!has_vertex_near(&deep, Vec3::new(0.0, r, 0.5)));
    }

    /// §2.4 / §8.1 matrix transform: every vertex (centres + both rings) is mapped
    /// through `matrix` as a homogeneous point at creation.
    #[test]
    fn create_cone_matrix_transform() {
        let mut mesh = LoopMesh::new();
        let n = 6usize;
        let r = 1.0;
        let depth = 2.0;
        let h = depth / 2.0;
        let matrix = Mat4::from_scale_rotation_translation(
            Vec3::ONE,
            Quat::from_rotation_z(std::f32::consts::FRAC_PI_4),
            Vec3::new(5.0, -2.0, 1.0),
        );

        mesh.create_cone(true, true, n as i32, r, r, depth, matrix, false);

        // The two centres at the transformed axis points.
        assert!(has_vertex_near(
            &mesh,
            matrix.transform_point3(Vec3::new(0.0, 0.0, -h))
        ));
        assert!(has_vertex_near(
            &mesh,
            matrix.transform_point3(Vec3::new(0.0, 0.0, h))
        ));
        // Bottom vertex 0's local (0, r, -h) maps through the matrix; the
        // untransformed position does not survive.
        let local0 = Vec3::new(0.0, r, -h);
        assert!(has_vertex_near(&mesh, matrix.transform_point3(local0)));
        assert!(!has_vertex_near(&mesh, local0));
        mesh.validate_topology();
    }

    /// §7: `segments = 0` is a no-op — nothing built (not a clamp, not an
    /// override), even with caps requested, and the prior mesh is untouched.
    #[test]
    fn create_cone_zero_segments_noop_on_populated_mesh() {
        let mut mesh = LoopMesh::new();
        let corners = [
            Vec3::new(10.0, 10.0, 0.0),
            Vec3::new(11.0, 10.0, 0.0),
            Vec3::new(11.0, 11.0, 0.0),
            Vec3::new(10.0, 11.0, 0.0),
        ];
        let island: Vec<VertexKey> = corners.iter().map(|&p| mesh.create_vertex(p)).collect();
        mesh.face_create_verts(&island, FaceCreateFlags::NONE);
        let before = (mesh.vertex_count(), mesh.edge_count(), mesh.face_count());

        let verts = mesh.create_cone(true, true, 0, 1.0, 1.0, 2.0, Mat4::IDENTITY, false);

        assert!(verts.is_empty(), "no-op returns an empty verts.out buffer");
        assert_eq!(
            (mesh.vertex_count(), mesh.edge_count(), mesh.face_count()),
            before,
            "segments = 0 built nothing"
        );
        mesh.validate_topology();
    }

    /// intro / §7 / §8.1 append onto a populated mesh: the cone is a disjoint
    /// island, prior geometry untouched, additive counts, no merge.
    #[test]
    fn create_cone_appends_onto_populated_mesh() {
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

        // n-gon-capped cylinder: ΔV = 2n, ΔE = 3n, ΔF = n + 2.
        let n = 6usize;
        mesh.create_cone(true, false, n as i32, 1.0, 1.0, 2.0, Mat4::IDENTITY, false);

        assert_eq!(
            (mesh.vertex_count(), mesh.edge_count(), mesh.face_count()),
            (4 + 2 * n, 4 + 3 * n, 1 + n + 2)
        );
        for &p in &corners {
            assert!(has_vertex_near(&mesh, p), "prior island vertex {p:?} moved");
        }
        mesh.validate_topology();
    }

    /// §1 / §8: the returned `verts.out` buffer is in slot order — the two centre
    /// vertices first (fan cap, surviving), then the ring vertices in segment
    /// order (each segment's bottom then top). Probed under a transform so a
    /// reordered or transform-dropping bug lands on the wrong position.
    #[test]
    fn create_cone_verts_out_order_capped_cylinder() {
        let n = 6usize;
        let r = 1.0;
        let depth = 2.0;
        let h = depth / 2.0;
        let matrix = Mat4::from_translation(Vec3::new(0.5, -1.0, 0.25))
            * Mat4::from_rotation_x(std::f32::consts::FRAC_PI_6);

        let mut mesh = LoopMesh::new();
        let verts = mesh.create_cone(true, true, n as i32, r, r, depth, matrix, false);

        assert_eq!(verts.len(), 2 * n + 2, "fan-capped cylinder reports 2n + 2");
        // Slot 0/1: the bottom then top centre.
        assert!(
            (mesh.vertex_position(verts[0]) - matrix.transform_point3(Vec3::new(0.0, 0.0, -h)))
                .length()
                < EPS,
            "verts.out[0] is the bottom centre"
        );
        assert!(
            (mesh.vertex_position(verts[1]) - matrix.transform_point3(Vec3::new(0.0, 0.0, h)))
                .length()
                < EPS,
            "verts.out[1] is the top centre"
        );
        // Then each segment's bottom then top, in angular order.
        for a in 0..n {
            let theta = std::f32::consts::TAU * (a as f32) / (n as f32);
            let (sin, cos) = (theta.sin(), theta.cos());
            let expect_b = matrix.transform_point3(Vec3::new(r * sin, r * cos, -h));
            let expect_t = matrix.transform_point3(Vec3::new(r * sin, r * cos, h));
            let got_b = mesh.vertex_position(verts[2 + 2 * a]);
            let got_t = mesh.vertex_position(verts[2 + 2 * a + 1]);
            assert!(
                (got_b - expect_b).length() < EPS,
                "slot {} bottom",
                2 + 2 * a
            );
            assert!(
                (got_t - expect_t).length() < EPS,
                "slot {} top",
                2 + 2 * a + 1
            );
        }
    }

    fn register_uv(mesh: &mut LoopMesh) -> ElementAttributeKey {
        mesh.register_loop_attribute(ElementAttributeDesc::numeric(
            AttributeKey::custom("uv"),
            AttrType::Vec2,
        ))
        .expect("uv layer registers")
    }

    /// §4: an n-gon-capped cylinder with `calc_uvs` and an active UV layer paints
    /// the side wall as a horizontal strip in the **top half** (`v ∈ {0.5, 1}`,
    /// `u` stepping the rung columns including the `u = 1` / `u = 0` seam) and each
    /// cap as a radial disc in the **bottom half** (`v < 0.5`). The cylinder strip
    /// is the exact, fully-pinned axis; the cap-disc centre/radius constants follow
    /// the guide's "near"/"≈" values.
    #[test]
    fn create_cone_calc_uvs_capped_cylinder() {
        let mut mesh = LoopMesh::new();
        let uv = register_uv(&mut mesh);
        let n = 6usize;
        let r = 1.0;
        let depth = 2.0;

        mesh.create_cone(true, false, n as i32, r, r, depth, Mat4::IDENTITY, true);

        let mut saw_strip_top = false; // a wall loop at v = 1.0
        let mut saw_strip_bottom = false; // a wall loop at v = 0.5
        let mut saw_seam_right = false; // u = 1.0
        let mut saw_seam_left = false; // u = 0.0
        for face in mesh.face_keys() {
            let is_wall = mesh.face_vert_keys(face).len() == 4;
            for loop_key in mesh.face_loop_keys(face) {
                let [u, v] = mesh
                    .loop_attribs()
                    .get_vec2(uv, loop_key.as_usize())
                    .unwrap();
                if is_wall {
                    assert!(v >= 0.5 - EPS, "wall UVs live in the top half: {v}");
                    if (v - 1.0).abs() < EPS {
                        saw_strip_top = true;
                    }
                    if (v - 0.5).abs() < EPS {
                        saw_strip_bottom = true;
                    }
                    if (u - 1.0).abs() < EPS {
                        saw_seam_right = true;
                    }
                    if u.abs() < EPS {
                        saw_seam_left = true;
                    }
                } else {
                    assert!(v < 0.5 + EPS, "cap UVs live in the bottom half: {v}");
                }
            }
        }
        assert!(
            saw_strip_top && saw_strip_bottom,
            "wall strip spans the top half"
        );
        assert!(
            saw_seam_right && saw_seam_left,
            "wall strip spans the u seam"
        );
        mesh.validate_topology();
    }

    /// §4 / §7: `calc_uvs = true` with no UV layer present is a silent no-op — it
    /// builds the topology and creates no layer.
    #[test]
    fn create_cone_calc_uvs_no_layer_is_noop() {
        let mut mesh = LoopMesh::new();
        mesh.create_cone(true, false, 6, 1.0, 1.0, 2.0, Mat4::IDENTITY, true);
        assert!(mesh.loop_attribs().is_empty(), "no UV layer was created");
        mesh.validate_topology();
    }
}
