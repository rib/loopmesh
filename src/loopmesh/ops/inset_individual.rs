use glam::{Vec2, Vec3};

use super::mean_value::{mean_value_weights, PlaneProjection};
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

/// One inset face's customdata baseline (INSET.md §10.2/§4.5, sized to a single
/// face). Captured before the corners move so the §7 rebake can resample the
/// *original* corner attributes at the *new* corner positions.
struct FaceCdSnapshot {
    /// Original corner positions projected into the face-local 2D plane, in
    /// loop-cycle order.
    proj_corners: Vec<Vec2>,
    /// Detached copy of each corner's loop customdata block.
    loop_cd: SlotSnapshot,
    /// Detached copy of each corner-vertex's customdata block.
    vert_cd: SlotSnapshot,
    /// Axis-dominant projection shared by the snapshot corners and the query
    /// points (§10.1.1).
    projection: PlaneProjection,
}

/// The four loop keys of one wall quad, named by the INSET.md §10.1
/// correspondence: `*_start`/`*_end` is the source-face edge's start/end
/// corner; `far_*` sits on the original (stayed-put) edge and `near_*` on the
/// inset-side (moved) edge. A degenerate wall (coincident corners) may leave
/// some slots `None`.
#[derive(Default)]
struct WallCornerLoops {
    far_start: Option<LoopKey>,
    far_end: Option<LoopKey>,
    near_start: Option<LoopKey>,
    near_end: Option<LoopKey>,
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
    /// `use_interpolate` (INSET.md §10.2) controls how the shrunken inset
    /// face's per-corner customdata is filled after the corners move:
    ///   off → each surviving corner carries the source face's loop/vert
    ///         customdata **verbatim** (the cheap path; correct in value but
    ///         no longer registered with the moved corner geometry).
    ///   on  → each inset-face corner is **rebaked** by the §7 mean-value
    ///         resample of the source face's pre-offset corner customdata
    ///         evaluated at the moved corner's projected position. The
    ///         snapshot scope is the single inset face; there is no §6.3
    ///         corner merge (the source-face corners are no longer shared
    ///         with anything after detachment).
    ///
    /// Wall loop customdata (INSET.md §10.1/§10.2) always comes from the
    /// *source (inset) face's own loops* — never from any neighbour — so an
    /// isolated face still has its walls filled. The flag only changes *which
    /// snapshot in time* of those loops the two wall sides read: the
    /// stayed-put original-edge (`far_*`) corners take the pre-offset value and
    /// the moved inset-side (`near_*`) corners take the post-rebake value
    /// (`use_interpolate` on), or both take the verbatim pre-offset value
    /// (`use_interpolate` off).
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

        // (Optional) customdata snapshot — INSET.md §10.2/§4.5, scoped to this
        // one face. Capture the source face's per-corner loop/vert customdata
        // and the corners' axis-dominant 2D projection *before* the corners
        // move so phase four (§7) can resample the original attributes at the
        // moved corner positions. The loop keys survive the re-anchor below
        // (only their `.vertex` changes), so each snapshot loop still names the
        // live corner at rebake time.
        let snapshot: Option<FaceCdSnapshot> = if params.use_interpolate
            && (!self.loop_attribs.is_empty() || !self.vertex_attribs.is_empty())
        {
            let projection = PlaneProjection::from_normal(face_normal);
            let proj_corners: Vec<Vec2> = old_positions
                .iter()
                .map(|&p| projection.project(p))
                .collect();
            let loop_slots: Vec<usize> = loops.iter().map(|l| l.as_usize()).collect();
            // The new verts carry verbatim copies of the old verts' customdata,
            // so snapshotting the old verts captures the same baseline the new
            // corners start from.
            let vert_slots: Vec<usize> = old_verts.iter().map(|v| v.as_usize()).collect();
            Some(FaceCdSnapshot {
                proj_corners,
                loop_cd: self.loop_attribs.snapshot_slots(&loop_slots),
                vert_cd: self.vertex_attribs.snapshot_slots(&vert_slots),
                projection,
            })
        } else {
            None
        };

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

        // Phase four (INSET.md §10.2/§7) — rebake the inset face's per-corner
        // customdata at the moved corner positions via the §10.1 mean-value
        // resample. Each current corner's projected position is the query point
        // over the snapshot's recorded corners; the resulting weight vector
        // blends every per-loop and per-vertex layer under its own policy.
        //
        // Two-phase wall timing (INSET.md §10.1/§10.2): the walls source *all*
        // their loop customdata from the inset face's own loops — never a
        // neighbour — but the original-edge (`far_*`, stayed-put) corners must
        // read the loops' *pre-offset* values while the inset-side (`near_*`,
        // moved) corners read the *post-rebake* values. The source loop for
        // corner `i` is `loops[i]`: its `.vertex` was re-anchored above but the
        // loop key still names corner `i`'s customdata block. So we build the
        // walls *before* the rebake, fill the `far_*` loops from the
        // still-pre-offset source loops, then run the rebake (which overwrites
        // the source loops at the moved positions), then fill the `near_*`
        // loops. With `use_interpolate` off there is no rebake, so both sides
        // read the verbatim pre-offset value and the wall is uniform per corner.

        // Side edges between matched original / new corners are shared between
        // two walls at the same corner.
        let mut side_edges: Vec<EdgeKey> = Vec::with_capacity(n_loops);
        for i in 0..n_loops {
            let e = self.create_edge([old_verts[i], new_verts[i]], EdgeCreateFlags::NONE);
            side_edges.push(e);
        }

        // Build each wall quad — one per original edge — connecting
        // (old_i, old_{i+1}, new_{i+1}, new_i). Record, per wall, the loop key
        // at each of its four corners so the §10.1 direct (non-swapped)
        // source-loop correspondence can be applied in two timing phases.
        let do_loop_interp = !self.loop_attribs.is_empty();
        let mut wall_corner_loops: Vec<WallCornerLoops> = Vec::with_capacity(n_loops);
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

            // Classify the wall's four loops by which source corner they share
            // (start = corner `i`, end = corner `i_next`) and which edge they
            // sit on (`far_*` on the original edge / `old_verts`, `near_*` on
            // the inset-side edge / `new_verts`). Direct, non-swapped per §10.1.
            let mut corners = WallCornerLoops::default();
            let mut wl = self.faces.get(wall).first_edge_loop;
            for _ in 0..verts.len() {
                let wv = self.loop_vertex(wl);
                if wv == old_verts[i] {
                    corners.far_start = Some(wl);
                } else if wv == old_verts[i_next] {
                    corners.far_end = Some(wl);
                } else if wv == new_verts[i] {
                    corners.near_start = Some(wl);
                } else if wv == new_verts[i_next] {
                    corners.near_end = Some(wl);
                }
                wl = self.loop_next(wl);
            }
            wall_corner_loops.push(corners);
        }

        // Phase A — `far_*` wall loops from the *pre-offset* source loops.
        if do_loop_interp {
            for i in 0..n_loops {
                let i_next = (i + 1) % n_loops;
                let corners = &wall_corner_loops[i];
                if let Some(dst) = corners.far_start {
                    self.loop_attribs
                        .copy_into(dst.as_usize(), loops[i].as_usize());
                }
                if let Some(dst) = corners.far_end {
                    self.loop_attribs
                        .copy_into(dst.as_usize(), loops[i_next].as_usize());
                }
            }
        }

        // Phase four (INSET.md §10.2/§7) — rebake the inset face's per-corner
        // customdata at the moved corner positions via the §10.1 mean-value
        // resample. Each current corner's projected position is the query point
        // over the snapshot's recorded corners; the resulting weight vector
        // blends every per-loop and per-vertex layer under its own policy.
        if let Some(snapshot) = &snapshot {
            for i in 0..n_loops {
                let q = snapshot.projection.project(new_positions[i]);
                let weights = mean_value_weights(&snapshot.proj_corners, q);
                self.loop_attribs.interpolate_from_snapshot(
                    loops[i].as_usize(),
                    &snapshot.loop_cd,
                    &weights,
                );
                self.vertex_attribs.interpolate_from_snapshot(
                    new_verts[i].as_usize(),
                    &snapshot.vert_cd,
                    &weights,
                );
            }
        }

        // Phase B — `near_*` wall loops from the (now possibly rebaked) source
        // loops. With `use_interpolate` off these still hold the pre-offset
        // value, so far and near match and the wall is uniform.
        if do_loop_interp {
            for i in 0..n_loops {
                let i_next = (i + 1) % n_loops;
                let corners = &wall_corner_loops[i];
                if let Some(dst) = corners.near_start {
                    self.loop_attribs
                        .copy_into(dst.as_usize(), loops[i].as_usize());
                }
                if let Some(dst) = corners.near_end {
                    self.loop_attribs
                        .copy_into(dst.as_usize(), loops[i_next].as_usize());
                }
            }
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
    use glam::{Vec2, Vec3};

    use super::super::mean_value::{mean_value_weights, PlaneProjection};
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

    /// A unit-square quad whose corners carry a UV layer. Returns the registered
    /// UV layer key and the four perimeter loop keys keyed by vertex.
    fn unit_quad_with_uv(
        mesh: &mut LoopMesh,
        uv_at: impl Fn(Vec3) -> [f32; 2],
    ) -> (
        FaceKey,
        crate::loopmesh::ElementAttributeKey,
        std::collections::HashMap<VertexKey, [f32; 2]>,
    ) {
        let verts = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        ];
        let face = create_face_from_verts(mesh, &verts);
        let uv = mesh
            .register_loop_attribute(ElementAttributeDesc::numeric(
                AttributeKey::custom("uv"),
                AttrType::Vec2,
            ))
            .unwrap();
        let mut expected: std::collections::HashMap<VertexKey, [f32; 2]> =
            std::collections::HashMap::new();
        let mut it = mesh.face_loops(face);
        let mut corner_loops: Vec<(VertexKey, LoopKey)> = Vec::new();
        while let Some(l) = it.next(mesh) {
            corner_loops.push((mesh.loop_vertex(l), l));
        }
        for (v, l) in corner_loops {
            let value = uv_at(mesh.vertex_position(v));
            mesh.loop_attribs_mut().set_vec2(uv, l.as_usize(), value);
            expected.insert(v, value);
        }
        (face, uv, expected)
    }

    /// The inset face's corner loops after `inset_individual`, keyed by the
    /// moved corner position. The inset face keeps the `face` identity; its
    /// loops are re-anchored to the new (moved) verts.
    fn inset_corner_uvs(
        mesh: &LoopMesh,
        face: FaceKey,
        uv: crate::loopmesh::ElementAttributeKey,
    ) -> Vec<(Vec3, [f32; 2])> {
        let mut out = Vec::new();
        let mut it = mesh.face_loops(face);
        while let Some(l) = it.next(mesh) {
            let p = mesh.vertex_position(mesh.loop_vertex(l));
            let value = mesh.loop_attribs().get_vec2(uv, l.as_usize()).unwrap();
            out.push((p, value));
        }
        out
    }

    #[test]
    fn inset_individual_no_interpolate_copies_uv_verbatim() {
        // Non-affine UV (u = x·y, v = x²). With use_interpolate off, each
        // surviving corner keeps the source corner's UV value-for-value even
        // though the corner has moved inward (INSET.md §10.2, cheap path).
        let mut mesh = LoopMesh::new();
        let (face, uv, expected) = unit_quad_with_uv(&mut mesh, |p| [p.x * p.y, p.x * p.x]);
        // Source corner→UV keyed by source position, taken straight from the
        // values the helper actually stored.
        let original: Vec<(Vec3, [f32; 2])> = expected
            .iter()
            .map(|(&v, &uv_val)| (mesh.vertex_position(v), uv_val))
            .collect();

        mesh.inset_individual(
            &[face],
            InsetIndividualParams {
                thickness: 0.2,
                use_interpolate: false,
                ..Default::default()
            },
        );

        // Each moved inset corner sits inward of its source corner; its UV is
        // the verbatim source value. Match each inset corner to the source
        // corner it descends from (the nearest source corner).
        let got = inset_corner_uvs(&mesh, face, uv);
        for (pos, value) in got {
            let (_, src_uv) = original
                .iter()
                .min_by(|a, b| {
                    (a.0 - pos)
                        .length()
                        .partial_cmp(&(b.0 - pos).length())
                        .unwrap()
                })
                .unwrap();
            assert!(
                (value[0] - src_uv[0]).abs() < 1.0e-6 && (value[1] - src_uv[1]).abs() < 1.0e-6,
                "verbatim copy expected {src_uv:?} at {pos:?}, got {value:?}"
            );
        }
    }

    #[test]
    fn inset_individual_interpolate_rebakes_nonaffine_uv() {
        // Same non-affine UV. With use_interpolate on, each moved corner's UV
        // must equal the mean-value resample (§10.1) of the original corner
        // UVs at the corner's new projected position — checked lane-for-lane,
        // so a wrong weight scheme (barycentric / bilinear / centroid) would
        // fail rather than slip past an `any_differs` sanity check (INSET.md
        // §10.2/§7).
        let mut mesh = LoopMesh::new();
        let (face, uv, _expected) = unit_quad_with_uv(&mut mesh, |p| [p.x * p.y, p.x * p.x]);

        // Snapshot the source corners in loop-cycle order *before* the inset:
        // the loop keys are stable across the inset (re-anchored to moved
        // verts), so each entry pins one corner across the operation.
        let face_normal = {
            mesh.calc_face_normals();
            mesh.face_normal(face)
        };
        let projection = PlaneProjection::from_normal(face_normal);
        let mut snap_loops: Vec<LoopKey> = Vec::new();
        let mut proj_corners: Vec<Vec2> = Vec::new();
        let mut corner_uvs: Vec<[f32; 2]> = Vec::new();
        {
            let mut it = mesh.face_loops(face);
            while let Some(l) = it.next(&mesh) {
                let p = mesh.vertex_position(mesh.loop_vertex(l));
                snap_loops.push(l);
                proj_corners.push(projection.project(p));
                corner_uvs.push(mesh.loop_attribs().get_vec2(uv, l.as_usize()).unwrap());
            }
        }

        mesh.inset_individual(
            &[face],
            InsetIndividualParams {
                thickness: 0.2,
                use_interpolate: true,
                ..Default::default()
            },
        );

        // For each stable loop, recompute the expected MVC blend at the moved
        // corner's projected position and assert lane-for-lane equality.
        let mut any_differs = false;
        let mut checked = 0;
        for (i, &l) in snap_loops.iter().enumerate() {
            let new_p = mesh.vertex_position(mesh.loop_vertex(l));
            let q = projection.project(new_p);
            let w = mean_value_weights(&proj_corners, q);
            let mut want = [0.0_f32; 2];
            for k in 0..w.len() {
                want[0] += w[k] * corner_uvs[k][0];
                want[1] += w[k] * corner_uvs[k][1];
            }
            let got = mesh.loop_attribs().get_vec2(uv, l.as_usize()).unwrap();
            assert!(
                (got[0] - want[0]).abs() < 1.0e-4 && (got[1] - want[1]).abs() < 1.0e-4,
                "corner {i}: got {got:?} want {want:?} (new_p={new_p:?})"
            );
            // The corner genuinely moved (smooth-interior path, not the
            // on-corner identity short-circuit).
            assert!(
                (q - proj_corners[i]).length() > 1.0e-3,
                "corner {i} did not move"
            );
            if (got[0] - corner_uvs[i][0]).abs() > 1.0e-5
                || (got[1] - corner_uvs[i][1]).abs() > 1.0e-5
            {
                any_differs = true;
            }
            checked += 1;
        }
        assert_eq!(checked, 4);
        // Sanity: a non-affine layer's MVC rebake genuinely departs from the
        // verbatim source value at the moved corners.
        assert!(
            any_differs,
            "MVC rebake of a non-affine UV must differ from a verbatim copy"
        );
    }

    #[test]
    fn inset_individual_interpolate_affine_uv_reproduces_position() {
        // For an affine layer (u = x, v = y), the mean-value resample
        // reproduces the query point exactly over a convex polygon, so each
        // moved corner's rebaked UV equals its own moved (x, y). This both
        // confirms the resample is a true MVC blend and shows affine layers
        // are indistinguishable between the two paths (INSET.md §10.2).
        let mut mesh = LoopMesh::new();
        let (face, uv, _expected) = unit_quad_with_uv(&mut mesh, |p| [p.x, p.y]);

        mesh.inset_individual(
            &[face],
            InsetIndividualParams {
                thickness: 0.2,
                use_interpolate: true,
                ..Default::default()
            },
        );

        let got = inset_corner_uvs(&mesh, face, uv);
        for (pos, value) in got {
            assert!(
                (value[0] - pos.x).abs() < 1.0e-4 && (value[1] - pos.y).abs() < 1.0e-4,
                "affine UV should track moved position {pos:?}, got {value:?}"
            );
        }
    }

    #[test]
    fn inset_individual_interpolate_vertex_layer_rebakes() {
        // A per-vertex non-affine layer is rebaked the same way as the loop
        // layer, using the same weight vector at each corner (INSET.md §10.2).
        let mut mesh = LoopMesh::new();
        let verts = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        ];
        let face = create_face_from_verts(&mut mesh, &verts);
        let scalar = mesh
            .register_vertex_attribute(ElementAttributeDesc::numeric(
                AttributeKey::custom("w"),
                AttrType::F32,
            ))
            .unwrap();
        // Set w = x² on each original corner vertex.
        let mut it = mesh.face_loops(face);
        let mut corner_verts: Vec<VertexKey> = Vec::new();
        while let Some(l) = it.next(&mesh) {
            corner_verts.push(mesh.loop_vertex(l));
        }
        for &v in &corner_verts {
            let x = mesh.vertex_position(v).x;
            mesh.vertex_attribs_mut()
                .set_f32(scalar, v.as_usize(), x * x);
        }

        // Snapshot the source corners in loop-cycle order *before* the inset.
        // The vertex layer is rebaked with the same weight vector as the loop
        // layer, sampling the source corner-vertex values; the snapshot order
        // is the source verts' values (the new verts start as verbatim copies).
        let face_normal = {
            mesh.calc_face_normals();
            mesh.face_normal(face)
        };
        let projection = PlaneProjection::from_normal(face_normal);
        let mut snap_loops: Vec<LoopKey> = Vec::new();
        let mut proj_corners: Vec<Vec2> = Vec::new();
        let mut corner_vals: Vec<f32> = Vec::new();
        {
            let mut it = mesh.face_loops(face);
            while let Some(l) = it.next(&mesh) {
                let v = mesh.loop_vertex(l);
                snap_loops.push(l);
                proj_corners.push(projection.project(mesh.vertex_position(v)));
                corner_vals.push(mesh.vertex_attribs().get_f32(scalar, v.as_usize()).unwrap());
            }
        }

        mesh.inset_individual(
            &[face],
            InsetIndividualParams {
                thickness: 0.2,
                use_interpolate: true,
                ..Default::default()
            },
        );

        // For each stable loop, the moved inset corner vert holds the MVC
        // resample of the source corner-vertex values at the moved projected
        // position — checked lane-for-lane.
        let mut any_differs = false;
        let mut checked = 0;
        for (i, &l) in snap_loops.iter().enumerate() {
            let v = mesh.loop_vertex(l);
            let new_p = mesh.vertex_position(v);
            let q = projection.project(new_p);
            let w = mean_value_weights(&proj_corners, q);
            let mut want = 0.0_f32;
            for k in 0..w.len() {
                want += w[k] * corner_vals[k];
            }
            let got = mesh.vertex_attribs().get_f32(scalar, v.as_usize()).unwrap();
            assert!(
                (got - want).abs() < 1.0e-4,
                "corner {i}: got {got} want {want} (new_p={new_p:?})"
            );
            assert!(
                (q - proj_corners[i]).length() > 1.0e-3,
                "corner {i} did not move"
            );
            if (got - corner_vals[i]).abs() > 1.0e-5 {
                any_differs = true;
            }
            checked += 1;
        }
        assert_eq!(checked, 4);
        assert!(
            any_differs,
            "per-vertex non-affine layer must be rebaked to an intermediate value"
        );
    }

    /// Classify a wall loop by its position: returns `true` if it sits on the
    /// original (stayed-put / `far_*`) edge, matching one of the four original
    /// corner positions to within tolerance; `false` if it is a moved
    /// inset-side (`near_*`) corner. `originals` are the source face's corner
    /// positions before the inset.
    fn loop_is_far(p: Vec3, originals: &[Vec3]) -> bool {
        originals.iter().any(|&o| (o - p).length() < 1.0e-6)
    }

    #[test]
    fn inset_individual_walls_source_from_inset_face_isolated() {
        // INSET.md §10.1: walls source their loop customdata from the inset
        // face's own loops, never a neighbour — so an *isolated* face (no
        // neighbour across any edge) still has all four walls filled. Use a
        // non-constant UV (= loop vertex x,y) so a per-corner copy is
        // discriminating; with use_interpolate off the wall is uniform per
        // source corner (verbatim), and no loop is left at the zero default.
        let originals = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        ];
        let mut mesh = LoopMesh::new();
        // Offset so every source corner UV is non-zero — lets a left-at-default
        // wall loop be distinguished from a legitimately-copied corner value.
        let (face, uv, expected) = unit_quad_with_uv(&mut mesh, |p| [p.x + 1.0, p.y + 1.0]);

        let result = mesh.inset_individual(
            &[face],
            InsetIndividualParams {
                thickness: 0.2,
                use_interpolate: false,
                ..Default::default()
            },
        );
        assert_eq!(result.wall_faces.len(), 4);

        // Every wall loop carries a source-face corner UV (the value seeded at
        // the original corner it shares), never the zero default and never
        // anything outside the source corner set.
        let mut filled = 0;
        for &w in &result.wall_faces {
            let mut it = mesh.face_loops(w);
            while let Some(l) = it.next(&mesh) {
                let got = mesh.loop_attribs().get_vec2(uv, l.as_usize()).unwrap();
                assert_ne!(got, [0.0, 0.0], "wall loop left at default zero UV");
                // The value must be one of the four source-corner UVs.
                let matches_source = expected
                    .values()
                    .any(|&s| (s[0] - got[0]).abs() < 1.0e-6 && (s[1] - got[1]).abs() < 1.0e-6);
                assert!(
                    matches_source,
                    "wall UV {got:?} not a source-face corner value"
                );
                filled += 1;
            }
        }
        // Four quad walls, four loops each.
        assert_eq!(filled, 16);
        // Sanity: the originals helper is exercised (far/near classifier used
        // by the timing tests below).
        let _ = loop_is_far(Vec3::ZERO, &originals);
    }

    #[test]
    fn inset_individual_walls_uniform_when_interpolate_off() {
        // INSET.md §10.2 (flag off): no rebake, so both wall sides read the
        // verbatim pre-offset source value — the wall is uniform per source
        // corner. For each wall, the far_* and near_* loops sharing a source
        // corner must hold the identical UV.
        let originals = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        ];
        let mut mesh = LoopMesh::new();
        // Non-affine UV so a rebake (if wrongly applied) would change near.
        let (face, uv, _expected) = unit_quad_with_uv(&mut mesh, |p| [p.x * p.y, p.x * p.x]);

        let result = mesh.inset_individual(
            &[face],
            InsetIndividualParams {
                thickness: 0.2,
                use_interpolate: false,
                ..Default::default()
            },
        );

        for &w in &result.wall_faces {
            // Collect (is_far, x_or_y_bucket, uv) per loop, then pair far/near
            // by the shared source corner. Since both source corners of a wall
            // sit at distinct positions, bucket by nearest original corner.
            let mut far: Vec<(usize, [f32; 2])> = Vec::new();
            let mut near: Vec<(usize, [f32; 2])> = Vec::new();
            let mut it = mesh.face_loops(w);
            while let Some(l) = it.next(&mesh) {
                let p = mesh.vertex_position(mesh.loop_vertex(l));
                let got = mesh.loop_attribs().get_vec2(uv, l.as_usize()).unwrap();
                // Nearest original corner index identifies the shared corner.
                let corner = (0..originals.len())
                    .min_by(|&a, &b| {
                        (originals[a] - p)
                            .length()
                            .partial_cmp(&(originals[b] - p).length())
                            .unwrap()
                    })
                    .unwrap();
                if loop_is_far(p, &originals) {
                    far.push((corner, got));
                } else {
                    near.push((corner, got));
                }
            }
            assert_eq!(far.len(), 2);
            assert_eq!(near.len(), 2);
            for (corner, far_uv) in &far {
                let (_, near_uv) = near
                    .iter()
                    .find(|(c, _)| c == corner)
                    .expect("near corner pairing");
                assert!(
                    (far_uv[0] - near_uv[0]).abs() < 1.0e-6
                        && (far_uv[1] - near_uv[1]).abs() < 1.0e-6,
                    "flag off: wall corner {corner} far {far_uv:?} != near {near_uv:?}"
                );
            }
        }
    }

    #[test]
    fn inset_individual_walls_two_phase_timing_interpolate() {
        // INSET.md §10.1/§10.2 (flag on): the original-edge (far_*) wall corners
        // take the *pre-offset* source value; the inset-side (near_*) wall
        // corners take the *post-rebake* source value. On a non-affine UV the
        // rebake moves the corner value, so far != near and the wall carries a
        // genuine gradient. We pin: far == the seeded source-corner value, and
        // near == the MVC-rebaked value at the moved corner (the same value the
        // inset face's loop now holds).
        let originals = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        ];
        let mut mesh = LoopMesh::new();
        let (face, uv, expected) = unit_quad_with_uv(&mut mesh, |p| [p.x * p.y, p.x * p.x]);

        let result = mesh.inset_individual(
            &[face],
            InsetIndividualParams {
                thickness: 0.2,
                use_interpolate: true,
                ..Default::default()
            },
        );

        // The inset face's (rebaked) corner UVs, keyed by nearest original
        // corner index — these are what the near_* wall corners must match.
        let nearest_corner = |p: Vec3| -> usize {
            (0..originals.len())
                .min_by(|&a, &b| {
                    (originals[a] - p)
                        .length()
                        .partial_cmp(&(originals[b] - p).length())
                        .unwrap()
                })
                .unwrap()
        };
        let mut near_expected: std::collections::HashMap<usize, [f32; 2]> =
            std::collections::HashMap::new();
        {
            let mut it = mesh.face_loops(face);
            while let Some(l) = it.next(&mesh) {
                let p = mesh.vertex_position(mesh.loop_vertex(l));
                near_expected.insert(
                    nearest_corner(p),
                    mesh.loop_attribs().get_vec2(uv, l.as_usize()).unwrap(),
                );
            }
        }
        // far value per corner = the seeded source value at the original corner.
        let original_uv_by_corner: std::collections::HashMap<usize, [f32; 2]> = (0..originals
            .len())
            .map(|c| {
                let v = expected
                    .iter()
                    .find(|(v, _)| (mesh.vertex_position(**v) - originals[c]).length() < 1.0e-6)
                    .map(|(_, &uvv)| uvv)
                    .unwrap();
                (c, v)
            })
            .collect();

        let mut any_gradient = false;
        for &w in &result.wall_faces {
            let mut far: Vec<(usize, [f32; 2])> = Vec::new();
            let mut near: Vec<(usize, [f32; 2])> = Vec::new();
            let mut it = mesh.face_loops(w);
            while let Some(l) = it.next(&mesh) {
                let p = mesh.vertex_position(mesh.loop_vertex(l));
                let got = mesh.loop_attribs().get_vec2(uv, l.as_usize()).unwrap();
                let corner = nearest_corner(p);
                if loop_is_far(p, &originals) {
                    far.push((corner, got));
                } else {
                    near.push((corner, got));
                }
            }
            for (corner, far_uv) in &far {
                let want = original_uv_by_corner[corner];
                assert!(
                    (far_uv[0] - want[0]).abs() < 1.0e-6 && (far_uv[1] - want[1]).abs() < 1.0e-6,
                    "far corner {corner}: got {far_uv:?} want pre-offset source {want:?}"
                );
            }
            for (corner, near_uv) in &near {
                let want = near_expected[corner];
                assert!(
                    (near_uv[0] - want[0]).abs() < 1.0e-5 && (near_uv[1] - want[1]).abs() < 1.0e-5,
                    "near corner {corner}: got {near_uv:?} want rebaked inset value {want:?}"
                );
                // Did this wall carry a gradient (far != near at a corner)?
                if let Some((_, far_uv)) = far.iter().find(|(c, _)| c == corner) {
                    if (far_uv[0] - near_uv[0]).abs() > 1.0e-4
                        || (far_uv[1] - near_uv[1]).abs() > 1.0e-4
                    {
                        any_gradient = true;
                    }
                }
            }
        }
        assert!(
            any_gradient,
            "a non-affine UV with use_interpolate on must give walls a far->near gradient"
        );
    }
}
