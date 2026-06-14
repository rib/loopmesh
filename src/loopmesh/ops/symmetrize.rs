use std::collections::HashSet;

use glam::Vec3;

use crate::loopmesh::ops::{BisectParams, GeomKey};
use crate::loopmesh::*;

/// The signed-axis enum naming which half is **kept and reflected**
/// ([`SYMMETRIZE.md`](../../../../radial-mesh-guides/operators/SYMMETRIZE.md)
/// §2). The named half is the source the symmetric result is built from; the
/// opposite half is discarded and overwritten by the reflection.
///
/// The integer values match the §2 table: `0=-X, 1=-Y, 2=-Z, 3=X, 4=Y, 5=Z`.
/// Two derived quantities drive the whole operator: the **axis** is
/// `value mod 3`, and the **plane normal** points in the positive axis
/// direction for the negative-named values (0–2) and the negative axis
/// direction for the positive-named values (3–5). The bisect always clears the
/// positive-normal side, so a negative-named direction keeps the negative half
/// and a positive-named direction keeps the positive half.
#[derive(PartialEq, Eq, Clone, Copy, Debug, Hash)]
pub enum SymmetrizeDirection {
    NegX,
    NegY,
    NegZ,
    PosX,
    PosY,
    PosZ,
}

impl SymmetrizeDirection {
    /// The component index the direction selects from a `Vec3` (`value mod 3`).
    fn axis_index(self) -> usize {
        match self {
            SymmetrizeDirection::NegX | SymmetrizeDirection::PosX => 0,
            SymmetrizeDirection::NegY | SymmetrizeDirection::PosY => 1,
            SymmetrizeDirection::NegZ | SymmetrizeDirection::PosZ => 2,
        }
    }

    /// True for the negative-named values (`-X, -Y, -Z`), whose kept half is the
    /// negative side and whose bisect normal points in the positive axis
    /// direction (§2).
    fn is_negative_named(self) -> bool {
        matches!(
            self,
            SymmetrizeDirection::NegX | SymmetrizeDirection::NegY | SymmetrizeDirection::NegZ
        )
    }

    /// The bisect plane normal (§2): `+axis` for the negative-named directions,
    /// `-axis` for the positive-named ones. The bisect clears the positive side
    /// of this normal, which is the half opposite the one this direction names.
    fn plane_normal(self) -> Vec3 {
        let mut normal = Vec3::ZERO;
        normal[self.axis_index()] = if self.is_negative_named() { 1.0 } else { -1.0 };
        normal
    }
}

/// Result of [`LoopMesh::symmetrize`]: the geometry making up the symmetric
/// result — the surviving bisected half plus the surviving reflected clone,
/// minus the seam clones that welded back onto the survivors
/// ([`SYMMETRIZE.md`](../../../../radial-mesh-guides/operators/SYMMETRIZE.md)
/// §1, the `geom.out` slot). The discarded half is gone and is not in this
/// buffer.
#[derive(Debug, Default)]
pub struct SymmetrizeResult {
    /// The `geom.out` slot (§1 / §6): the union of the bisect's survivors and
    /// the duplicate's clone, with the welded-away seam clones removed.
    pub geom_out: Vec<GeomKey>,
}

impl LoopMesh {
    /// Make `geom` mirror-symmetric across an origin coordinate plane by keeping
    /// the half `direction` names and reflecting it onto the other half.
    ///
    /// A thin composition
    /// ([`SYMMETRIZE.md`](../../../../radial-mesh-guides/operators/SYMMETRIZE.md)
    /// §6) over primitives that already have their own chapters:
    ///
    /// 1. **Bisect** along the origin plane fixed by `direction` (§2/§3),
    ///    clearing the positive-normal side (the half opposite the named one),
    ///    snapping near-plane verts onto the plane, and reporting the surviving
    ///    half plus the cut seam. `dist` is consumed here, once (§5).
    /// 2. **Duplicate** the surviving half into a coincident, disjoint clone with
    ///    a source→clone `vert_map`.
    /// 3. **Reflect** the clone by negating the axis coordinate, then
    ///    **explicitly reverse** every cloned face's loop cycle (§4): the
    ///    reflection flips winding once, the reverse flips it again, so the clone
    ///    ends up wound the *same* way as its source — the whole result is
    ///    uniformly oriented (the opposite of mirror, which lets the single flip
    ///    stand).
    /// 4. **Weld** each clone of a bisect-reported **seam vert** onto its
    ///    survivor (§5) — no post-reflect distance re-test; the bisect already
    ///    classified the on-plane set with `dist`.
    ///
    /// Returns the symmetric result's `geom.out` (§1): the surviving half plus
    /// the surviving clone, minus the welded-away seam clones.
    pub fn symmetrize(
        &mut self,
        geom: &[GeomKey],
        direction: SymmetrizeDirection,
        dist: f32,
    ) -> SymmetrizeResult {
        let axis = direction.axis_index();

        // Step 1 (§3) — bisect along the origin plane, clear the positive-normal
        // side (the discarded half), snap near-plane verts onto the plane, and
        // report the surviving half + the cut seam. `dist` is consumed here.
        let bisect = self.bisect_plane(
            geom,
            BisectParams {
                plane_co: Vec3::ZERO,
                plane_no: direction.plane_normal(),
                dist,
                use_snap_center: true,
                clear_inner: false,
                clear_outer: true,
            },
        );

        // Step 2 (§4) — duplicate the surviving half into a coincident clone.
        let dup = self.duplicate(&bisect.geom_out, false);

        // Step 3 (§4) — reflect the clone (negate the axis coordinate), then
        // re-reverse every cloned face. The net of the two winding flips returns
        // the clone to its source orientation (§4).
        self.reflect_symmetrize_clone(&dup.geom_out, axis);
        self.reverse_clone_faces(&dup.geom_out);

        // Step 4 (§5) — build the seam weld map from the bisect's reported seam
        // verts: weld each seam vert's clone onto the seam vert itself. No
        // distance re-test — membership was decided by the bisect's `dist`.
        let weld_pairs = self.build_symmetrize_seam_weld_map(&bisect.seam_verts, &dup.vert_map);

        // Step 5 (§5) — weld the seam: seam clones collapse onto the survivors,
        // seam-internal clone edges collapse, the survivors keep their position
        // (exactly on the plane after the snap) and customdata.
        self.weld_verts(&weld_pairs, false);

        // Step 6 (§1 / §6) — surface the surviving half + surviving clone, minus
        // the welded-away seam clones.
        let welded_away: HashSet<VertexKey> =
            weld_pairs.iter().map(|&(clone, _survivor)| clone).collect();
        let mut geom_out: Vec<GeomKey> = Vec::new();
        for entry in bisect.geom_out.iter().chain(dup.geom_out.iter()) {
            let live = match *entry {
                GeomKey::Vert(v) => self.vertices.try_get(v).is_some() && !welded_away.contains(&v),
                GeomKey::Edge(e) => self.edges.try_get(e).is_some(),
                GeomKey::Face(f) => self.faces.try_get(f).is_some(),
            };
            if live {
                geom_out.push(*entry);
            }
        }

        SymmetrizeResult { geom_out }
    }

    /// §4 — reflect every clone vert's position by negating the axis coordinate
    /// about the origin plane. A vert exactly on the plane (a seam vert, axis
    /// coordinate 0) reflects to itself, which is what lets §5's weld fuse it
    /// with no position change.
    fn reflect_symmetrize_clone(&mut self, clone_geom: &[GeomKey], axis: usize) {
        for entry in clone_geom {
            let GeomKey::Vert(clone) = *entry else {
                continue;
            };
            let mut position = self.vertex_position(clone);
            position[axis] = -position[axis];
            self.set_vertex_position(clone, position);
        }
    }

    /// §4 — the explicit rewind: reverse every cloned face's loop cycle so the
    /// reflection's determinant-−1 flip is undone and the clone is wound the same
    /// way as its source. Only the loop cycle is reversed; higher-order per-loop
    /// data is left untouched.
    fn reverse_clone_faces(&mut self, clone_geom: &[GeomKey]) {
        for entry in clone_geom {
            if let GeomKey::Face(clone) = *entry {
                self.euler_loop_reverse(clone);
            }
        }
    }

    /// §5 — turn the bisect's reported seam verts into the weld map: for each
    /// seam vert, weld its clone (`vert_map` lookup) onto the seam vert itself
    /// (the survivor). The pair is `(clone, survivor)`, so the survivor keeps its
    /// position and customdata and the clone collapses onto it. A seam vert with
    /// no clone entry (none of its incident geometry was duplicated) is skipped.
    fn build_symmetrize_seam_weld_map(
        &self,
        seam_verts: &[VertexKey],
        vert_map: &std::collections::HashMap<VertexKey, VertexKey>,
    ) -> Vec<(VertexKey, VertexKey)> {
        let mut pairs: Vec<(VertexKey, VertexKey)> = Vec::new();
        for &survivor in seam_verts {
            if let Some(&clone) = vert_map.get(&survivor) {
                pairs.push((clone, survivor));
            }
        }
        pairs
    }
}

#[cfg(test)]
mod tests {
    use glam::Vec3;

    use super::*;
    use crate::loopmesh::*;

    fn edge_of(mesh: &mut LoopMesh, a: VertexKey, b: VertexKey) -> EdgeKey {
        mesh.query_edge_exists(a, b)
            .unwrap_or_else(|| mesh.create_edge([a, b], EdgeCreateFlags::NONE))
    }

    fn make_face(mesh: &mut LoopMesh, ring: &[VertexKey]) -> FaceKey {
        let edges: Vec<EdgeKey> = (0..ring.len())
            .map(|i| edge_of(mesh, ring[i], ring[(i + 1) % ring.len()]))
            .collect();
        mesh.face_create(ring, &edges, FaceCreateFlags::NONE)
    }

    /// A strip of `cols` quads along x, spanning rows y = 0 and y = 1. Column
    /// `c` runs x = `x_start + c`. The faces are wound CCW in the z = 0 plane
    /// (normal +z). Returns the mesh.
    fn strip(x_start: f32, cols: usize) -> LoopMesh {
        let mut mesh = LoopMesh::new();
        let n = cols + 1;
        let mut bottom = Vec::new();
        let mut top = Vec::new();
        for i in 0..n {
            let x = x_start + i as f32;
            bottom.push(mesh.create_vertex(Vec3::new(x, 0.0, 0.0)));
            top.push(mesh.create_vertex(Vec3::new(x, 1.0, 0.0)));
        }
        for c in 0..cols {
            make_face(&mut mesh, &[bottom[c], bottom[c + 1], top[c + 1], top[c]]);
        }
        mesh
    }

    /// A strip whose columns sit at the given x coordinates (so a kept-side
    /// column can be placed just off the plane). Quads span y = 0..1, wound CCW.
    fn strip_at(xs: &[f32]) -> LoopMesh {
        let mut mesh = LoopMesh::new();
        let mut bottom = Vec::new();
        let mut top = Vec::new();
        for &x in xs {
            bottom.push(mesh.create_vertex(Vec3::new(x, 0.0, 0.0)));
            top.push(mesh.create_vertex(Vec3::new(x, 1.0, 0.0)));
        }
        for c in 0..xs.len() - 1 {
            make_face(&mut mesh, &[bottom[c], bottom[c + 1], top[c + 1], top[c]]);
        }
        mesh
    }

    fn whole_mesh_geom(mesh: &LoopMesh) -> Vec<GeomKey> {
        let mut geom: Vec<GeomKey> = Vec::new();
        geom.extend(mesh.vert_keys().into_iter().map(GeomKey::Vert));
        geom.extend(mesh.edge_keys().into_iter().map(GeomKey::Edge));
        geom.extend(mesh.face_keys().into_iter().map(GeomKey::Face));
        geom
    }

    /// True when, for every live vert at x = a, there is a live vert at x = -a
    /// with the same (y, z). The headline symmetry property (§8).
    fn is_x_symmetric(mesh: &LoopMesh) -> bool {
        let quant = |p: Vec3| {
            (
                (p.x * 1e4).round() as i64,
                (p.y * 1e4).round() as i64,
                (p.z * 1e4).round() as i64,
            )
        };
        let present: std::collections::HashSet<(i64, i64, i64)> = mesh
            .vert_keys()
            .iter()
            .map(|&v| quant(mesh.vertex_position(v)))
            .collect();
        present
            .iter()
            .all(|&(x, y, z)| present.contains(&(-x, y, z)))
    }

    #[test]
    fn direction_neg_x_keeps_negative_half_and_is_symmetric() {
        // §2 / §3 / §6: a strip straddling x = 0; direction = -X keeps x < 0,
        // reflects it onto x > 0, welds at the constructed seam. The result is
        // mirror-symmetric about x = 0 and every vert lands at |x| <= 2.
        let mut mesh = strip(-2.0, 4); // columns x = -2..2
        let geom = whole_mesh_geom(&mesh);
        mesh.symmetrize(&geom, SymmetrizeDirection::NegX, 0.0);

        assert!(
            is_x_symmetric(&mesh),
            "result is mirror-symmetric about x = 0"
        );
        // The kept half was x < 0; nothing of the discarded x > 0 *original*
        // survives un-reflected — every survivor is the kept half or its mirror.
        for v in mesh.vert_keys() {
            assert!(mesh.vertex_position(v).x.abs() <= 2.0 + 1e-4);
        }
        // The seam (x = 0 column) is shared, not doubled: there is at least one
        // vert exactly on the plane and the mesh stays connected across it.
        assert!(
            mesh.vert_keys()
                .iter()
                .any(|&v| mesh.vertex_position(v).x.abs() < 1e-5),
            "the seam column lies on x = 0"
        );
        mesh.validate_topology();
    }

    #[test]
    fn direction_selects_kept_side() {
        // §2: the same non-symmetric straddling fixture under -X vs X keeps
        // opposite halves, so the two results differ. A regression that hard-codes
        // one side would make these identical.
        let run = |dir: SymmetrizeDirection| {
            // Asymmetric strip: more columns on one side of the plane.
            let mut mesh = strip_at(&[-2.0, -1.0, 0.5, 2.0, 3.0]);
            let geom = whole_mesh_geom(&mesh);
            mesh.symmetrize(&geom, dir, 0.0);
            mesh.validate_topology();
            let mut xs: Vec<i64> = mesh
                .vert_keys()
                .iter()
                .map(|&v| (mesh.vertex_position(v).x * 1e3).round() as i64)
                .collect();
            xs.sort_unstable();
            (mesh.totvert(), mesh.totface(), xs)
        };

        let neg = run(SymmetrizeDirection::NegX);
        let pos = run(SymmetrizeDirection::PosX);

        // Both are symmetric, but built from different source halves.
        let neg_max = neg.2.iter().copied().max().unwrap();
        let pos_max = pos.2.iter().copied().max().unwrap();
        // -X keeps the negative half (extent reaches x = -2 → mirror to +2);
        // X keeps the positive half (extent reaches x = +3 → mirror to ±3).
        assert!(neg_max <= 2010, "-X built from the |x| <= 2 negative half");
        assert!(pos_max >= 2990, "X built from the x up to 3 positive half");
        assert_ne!(neg, pos, "direction selects the kept side; runs differ");
    }

    #[test]
    fn dist_fires_vs_zero_changes_seam_membership() {
        // §3 / §5 / §5.1: a kept-side column just off the plane (x = 0.08) under
        // direction = X (keep x > 0). dist = 0.2 snaps it onto the plane (seam
        // vert, its clone welds away); dist = 0 keeps it distinct (clone at
        // x = -0.08). The element counts differ — what makes `dist` observable.
        let run = |dist: f32| {
            let mut mesh = strip_at(&[-1.0, 0.08, 1.2]);
            let geom = whole_mesh_geom(&mesh);
            mesh.symmetrize(&geom, SymmetrizeDirection::PosX, dist);
            mesh.validate_topology();
            (mesh.totvert(), mesh.totedge(), mesh.totface())
        };

        let fires = run(0.2);
        let zero = run(0.0);
        assert_ne!(
            fires, zero,
            "dist fires (snap + weld) vs zero (distinct) produce different counts"
        );
        // When dist fires, the near-plane column became a seam vert: its mirror
        // welds onto it, so the result has fewer verts than the zero run, where
        // the column's mirror is a distinct vert at x = -0.08.
        assert!(
            fires.0 < zero.0,
            "the snapped+welded seam column saves verts vs the distinct mirror"
        );
        // The dist-zero run keeps a vert near x = -0.08 (the reflected near-plane
        // column); the dist-fires run does not (it snapped to 0).
        let mut mesh_zero = strip_at(&[-1.0, 0.08, 1.2]);
        let geom = whole_mesh_geom(&mesh_zero);
        mesh_zero.symmetrize(&geom, SymmetrizeDirection::PosX, 0.0);
        assert!(
            mesh_zero
                .vert_keys()
                .iter()
                .any(|&v| (mesh_zero.vertex_position(v).x + 0.08).abs() < 1e-4),
            "dist = 0 leaves the reflected near-plane column at x = -0.08"
        );
    }

    #[test]
    fn winding_is_uniform_after_reflect_and_rewind() {
        // §4 / §4.1 / §8: the kept half is wound CCW (+z). After reflect-and-
        // rewind the reflected clone must be wound the SAME way (+z), so EVERY
        // face in the welded result has a +z normal. Omitting the explicit
        // reverse would leave the reflected faces wound -z. The strip straddles
        // x = 0 *between* columns (no column on the plane) so the bisect cuts
        // cleanly, with no degenerate zero-width seam face to muddy the normals.
        let mut mesh = strip_at(&[-1.5, -0.5, 0.5, 1.5]); // crosses x = 0 mid-column
        let geom = whole_mesh_geom(&mesh);
        mesh.symmetrize(&geom, SymmetrizeDirection::NegX, 0.0);

        for f in mesh.face_keys() {
            let normal = mesh.face_compute_normal(f);
            assert!(
                normal.z > 0.0,
                "every face (kept + reflected) is wound +z; reflect-and-rewind \
                 keeps the reflected half oriented like its source"
            );
        }
        mesh.validate_topology();
    }

    #[test]
    fn geom_out_excludes_welded_seam_clones() {
        // §1 / §6: geom.out is the surviving half + surviving clone, minus the
        // welded-away seam clones. Every reported key must be live, and no
        // welded-away seam clone vert may appear.
        let mut mesh = strip(-2.0, 4);
        let geom = whole_mesh_geom(&mesh);
        let result = mesh.symmetrize(&geom, SymmetrizeDirection::NegX, 0.0);

        assert!(!result.geom_out.is_empty(), "non-empty result buffer");
        for entry in &result.geom_out {
            let live = match *entry {
                GeomKey::Vert(v) => mesh.vertices.try_get(v).is_some(),
                GeomKey::Edge(e) => mesh.edges.try_get(e).is_some(),
                GeomKey::Face(f) => mesh.faces.try_get(f).is_some(),
            };
            assert!(live, "every geom.out key is live");
        }
        mesh.validate_topology();
    }

    #[test]
    fn axis_y_keeps_negative_half() {
        // §2: the same algorithm on a different coordinate index. A strip along y
        // straddling y = 0; direction = -Y keeps y < 0 and reflects onto y > 0.
        let mut mesh = LoopMesh::new();
        let xs = [0.0_f32, 1.0];
        let ys = [-2.0_f32, -1.0, 0.5, 2.0];
        let mut grid: Vec<Vec<VertexKey>> = Vec::new();
        for &y in &ys {
            let mut row = Vec::new();
            for &x in &xs {
                row.push(mesh.create_vertex(Vec3::new(x, y, 0.0)));
            }
            grid.push(row);
        }
        for r in 0..ys.len() - 1 {
            make_face(
                &mut mesh,
                &[grid[r][0], grid[r][1], grid[r + 1][1], grid[r + 1][0]],
            );
        }
        let geom = whole_mesh_geom(&mesh);
        mesh.symmetrize(&geom, SymmetrizeDirection::NegY, 0.0);

        let quant = |p: Vec3| {
            (
                (p.x * 1e4).round() as i64,
                (p.y * 1e4).round() as i64,
                (p.z * 1e4).round() as i64,
            )
        };
        let present: std::collections::HashSet<(i64, i64, i64)> = mesh
            .vert_keys()
            .iter()
            .map(|&v| quant(mesh.vertex_position(v)))
            .collect();
        assert!(
            present
                .iter()
                .all(|&(x, y, z)| present.contains(&(x, -y, z))),
            "result is mirror-symmetric about y = 0"
        );
        // Kept negative half reaches y = -2; nothing beyond |y| = 2.
        for v in mesh.vert_keys() {
            assert!(mesh.vertex_position(v).y.abs() <= 2.0 + 1e-4);
        }
        mesh.validate_topology();
    }
}
