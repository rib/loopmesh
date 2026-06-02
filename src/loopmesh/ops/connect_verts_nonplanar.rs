use std::collections::HashSet;

use glam::{Vec2, Vec3};

use super::connect_verts::{is_interior_diagonal, signed_area};
use crate::loopmesh::*;
use crate::triangulate::axis_dominant_basis_negated;

/// A half's Newell normal shorter than this is treated as degenerate
/// (collinear / zero-area arc); the diagonal is rejected for that orientation
/// (CONNECT_VERTS_NONPLANAR.md §3.1).
const DEGENERATE_NORMAL_LEN_SQ: f32 = 1e-12;

/// The captured output buffers of [`LoopMesh::connect_verts_nonplanar_capture`]
/// (CONNECT_VERTS_NONPLANAR.md §1). Both are sets; enumeration order is not
/// observable for parity (§1, §6).
#[derive(Debug, Clone, Default)]
pub struct ConnectVertsNonplanarOut {
    /// `edges.out` — the new diagonal edges the operator cut, one per committed
    /// split (§3.4).
    pub edges: Vec<EdgeKey>,
    /// `faces.out` — every face the operator touched: for each committed split,
    /// both the surviving input-face handle and the newly created face are
    /// flagged. A face split several times by the recursion appears once
    /// (re-flagging is idempotent) and every leaf piece is present (§1, §3.4).
    pub faces: Vec<FaceKey>,
}

impl LoopMesh {
    /// Cut every non-planar face in `faces` along the diagonal that flattens it
    /// best, recursing on each half until every piece is flat within
    /// `angle_limit`.
    ///
    /// Follows `radial-mesh-guides/operators/CONNECT_VERTS_NONPLANAR.md`:
    ///
    /// - §3 work stack: seed from `faces` (only `face_len > 3`, §3.5), pop a
    ///   face, split it once, push back each resulting half that is still an
    ///   n-gon. LIFO over the deterministic seed.
    /// - §3.1/§3.2 search: O(n²) over non-adjacent corner pairs; each half gets
    ///   its own arc Newell normal and total-absolute-height planar error; the
    ///   lowest-error legal diagonal wins on a strict first-improvement.
    /// - §3.3 legality: the in-polygon diagonal test shared with
    ///   [`Self::connect_verts`]' `check_degenerate` path, applied lazily on a
    ///   running-best improvement.
    /// - §3.4 commit gate: cut only when the winning diagonal's fold cosine is
    ///   *below* `cos(angle_limit)` — `angle_limit` is a stop condition, not a
    ///   where-to-cut selector. The single cut is one [`Self::face_split`]
    ///   (SFME).
    ///
    /// Like the other automatic family members the caller names only faces; the
    /// operator chooses every cut and never creates or destroys a vertex.
    pub fn connect_verts_nonplanar(&mut self, faces: &[FaceKey], angle_limit: f32) {
        self.connect_verts_nonplanar_out(faces, angle_limit);
    }

    /// Capturing form of [`Self::connect_verts_nonplanar`]: returns the
    /// `edges.out` buffer (CONNECT_VERTS_NONPLANAR.md §1) — the new diagonal
    /// edges the operator cut, one per committed split. The buffer is a set;
    /// enumeration order is not observable for parity (§1, §6).
    ///
    /// This is the `edges.out`-only projection of
    /// [`Self::connect_verts_nonplanar_capture`], kept as a stable contract;
    /// call the capturing form when `faces.out` is also wanted.
    pub fn connect_verts_nonplanar_out(
        &mut self,
        faces: &[FaceKey],
        angle_limit: f32,
    ) -> Vec<EdgeKey> {
        self.connect_verts_nonplanar_capture(faces, angle_limit).edges
    }

    /// Capturing form returning both output buffers of
    /// connect-verts-nonplanar (CONNECT_VERTS_NONPLANAR.md §1):
    ///
    /// - `edges.out` — the new diagonal edges, one per committed split (§3.4).
    /// - `faces.out` — every touched face: for each committed split both the
    ///   surviving input-face handle and the newly created half are flagged, so
    ///   a face cut several times by the recursion appears once (re-flagging is
    ///   idempotent) and every leaf piece is present (§1, §3.4).
    ///
    /// Both are sets; enumeration order is not observable for parity (§1, §6).
    pub fn connect_verts_nonplanar_capture(
        &mut self,
        faces: &[FaceKey],
        angle_limit: f32,
    ) -> ConnectVertsNonplanarOut {
        let angle_limit_cos = angle_limit.cos();

        // §3 — LIFO work stack, seeded with the n-gon inputs only.
        let mut work_stack: Vec<FaceKey> = faces
            .iter()
            .copied()
            .filter(|&face| self.face_len(face) > 3)
            .collect();

        let mut cut_diagonals: HashSet<EdgeKey> = HashSet::new();
        // Dedup keeps faces.out idempotent (§1): a face split several times by
        // the recursion is re-flagged on each cut but appears once.
        let mut touched_faces: HashSet<FaceKey> = HashSet::new();
        while let Some(face) = work_stack.pop() {
            let Some((kept_half, new_half, new_edge)) =
                self.split_by_best_fold(face, angle_limit_cos)
            else {
                continue; // flat enough (or no legal diagonal): a leaf.
            };
            cut_diagonals.insert(new_edge);
            // §3.4 — flag both the surviving handle and the new half. Leaf
            // pieces are present because the half that does not recurse is
            // flagged here and never popped again.
            touched_faces.insert(kept_half);
            touched_faces.insert(new_half);
            for half in [kept_half, new_half] {
                if self.face_len(half) > 3 {
                    work_stack.push(half);
                }
            }
        }

        ConnectVertsNonplanarOut {
            edges: cut_diagonals.into_iter().collect(),
            faces: touched_faces.into_iter().collect(),
        }
    }

    /// §3.4 — find the flattest-halves legal diagonal of `face` and cut it iff
    /// its fold exceeds the tolerance. Returns `(surviving_face, new_face,
    /// new_edge)` on a committed cut, or `None` when the face is left whole.
    fn split_by_best_fold(
        &mut self,
        face: FaceKey,
        angle_limit_cos: f32,
    ) -> Option<(FaceKey, FaceKey, EdgeKey)> {
        let (corner_a, corner_b, fold_cos) = self.find_best_fold_diagonal(face)?;

        // A sharper fold has a smaller cosine; cut only when the fold is sharper
        // than the tolerance allows (§1 cosine convention).
        if fold_cos >= angle_limit_cos {
            return None;
        }

        let (new_face, new_edge) = self
            .face_split(face, corner_a, corner_b)
            .expect("connect_verts_nonplanar: face_split failed on a legal diagonal");
        Some((face, new_face, new_edge))
    }

    /// §3.2 — scan every non-adjacent corner pair of `face`, scoring the sum of
    /// the two halves' planar errors, and return the lowest-error legal
    /// diagonal together with its fold cosine. The strict `<` improvement makes
    /// the first-encountered diagonal win an error tie (§3.2/§5.2 cycle-order
    /// first-improvement).
    fn find_best_fold_diagonal(&self, face: FaceKey) -> Option<(VertexKey, VertexKey, f32)> {
        let cycle = self.face_vert_keys(face);
        let positions: Vec<Vec3> = cycle.iter().map(|&v| self.vertex_position(v)).collect();
        let n = cycle.len();

        // §3.3 — project the face onto its best-fit plane once; the legality
        // test reads the 2D polygon. Match connect_verts' winding-normalised
        // setup so the in-cone test sees a CCW polygon.
        let face_normal = self.face_compute_normal(face);
        let (basis_u, basis_w) = axis_dominant_basis_negated(face_normal);
        let mut projected: Vec<Vec2> = positions
            .iter()
            .map(|&p| Vec2::new(basis_u.dot(p), basis_w.dot(p)))
            .collect();
        let mut proj_index: Vec<usize> = (0..n).collect();
        if signed_area(&projected) < 0.0 {
            projected.reverse();
            proj_index.reverse();
        }
        let proj_pos = |original: usize| proj_index.iter().position(|&x| x == original).unwrap();

        let mut best: Option<(VertexKey, VertexKey, f32)> = None;
        let mut best_error = f32::INFINITY;

        for i in 0..n {
            for j in (i + 2)..n {
                // The closing wrap-adjacent pair (0, n-1) is the only remaining
                // adjacency once j starts at i+2.
                if i == 0 && j == n - 1 {
                    continue;
                }
                let Some(normal_a) = self.arc_newell_normal(&positions, i, j) else {
                    continue;
                };
                let Some(normal_b) = self.arc_newell_normal(&positions, j, i) else {
                    continue;
                };

                let error = arc_planar_error(&positions, i, j, normal_a)
                    + arc_planar_error(&positions, j, i, normal_b);

                if error < best_error
                    && is_interior_diagonal(&projected, n, proj_pos(i), proj_pos(j))
                {
                    best_error = error;
                    let fold_cos = normal_a.dot(normal_b);
                    best = Some((cycle[i], cycle[j], fold_cos));
                }
            }
        }
        best
    }

    /// §3.1 — Newell best-fit unit normal of one arc (half-face): the corners
    /// from index `start` round the cycle to `end` inclusive. `None` for a
    /// degenerate (collinear / zero-area) arc.
    fn arc_newell_normal(&self, positions: &[Vec3], start: usize, end: usize) -> Option<Vec3> {
        let arc = arc_indices(positions.len(), start, end);
        let mut normal = Vec3::ZERO;
        for window in 0..arc.len() {
            let current = positions[arc[window]];
            let next = positions[arc[(window + 1) % arc.len()]];
            normal.x += (current.y - next.y) * (current.z + next.z);
            normal.y += (current.z - next.z) * (current.x + next.x);
            normal.z += (current.x - next.x) * (current.y + next.y);
        }
        if normal.length_squared() < DEGENERATE_NORMAL_LEN_SQ {
            None
        } else {
            Some(normal.normalize())
        }
    }
}

/// §3.1 — planar error of one arc against its own best-fit normal: the total
/// absolute variation of corner height (projection onto the normal axis) walked
/// around the arc, including the wrap from the last corner back to the first.
/// Zero for a perfectly flat arc; grows with ripple count and amplitude.
fn arc_planar_error(positions: &[Vec3], start: usize, end: usize, normal: Vec3) -> f32 {
    let arc = arc_indices(positions.len(), start, end);
    let height = |index: usize| normal.dot(positions[index]);
    let mut previous_height = height(*arc.last().unwrap());
    let mut error = 0.0;
    for &index in &arc {
        let current_height = height(index);
        error += (current_height - previous_height).abs();
        previous_height = current_height;
    }
    error
}

/// The contiguous cycle indices of one arc: from `start` round to `end`
/// inclusive (wrapping past the end of the cycle when `end < start`).
fn arc_indices(cycle_len: usize, start: usize, end: usize) -> Vec<usize> {
    let mut indices = Vec::new();
    let mut index = start;
    loop {
        indices.push(index);
        if index == end {
            break;
        }
        index = (index + 1) % cycle_len;
    }
    indices
}

#[cfg(test)]
mod tests {
    use glam::Vec3;

    use crate::loopmesh::test_support::create_face_from_verts;
    use crate::loopmesh::*;

    const ANGLE_TIGHT: f32 = 0.1; // ~5.7°, smaller than the folded fixtures' creases
    const ANGLE_LOOSE: f32 = 2.0; // ~114°, larger than any crease used here

    /// §7 `folded_quad(fold)` — corners 0 and 2 on the base plane (the hinge),
    /// 1 lifted and 3 dropped so the quad creases about the {0,2} diagonal. The
    /// two diagonals are deliberately asymmetric: {0,2} stays flat (its halves
    /// are the two triangles, fold sharp) while {1,3} runs across the crease.
    fn folded_quad(mesh: &mut LoopMesh, fold: f32) -> (FaceKey, Vec<VertexKey>) {
        let coords = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 1.0, fold),
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::new(1.0, -1.0, -fold * 0.5),
        ];
        let face = create_face_from_verts(mesh, &coords);
        let verts = mesh.face_vert_keys(face);
        (face, verts)
    }

    /// §7 `nonplanar_pentagon(ridge)` — middle corners lifted onto a ridge while
    /// the outer corners stay near the base plane, a shallow tent. The ridge
    /// forces the best diagonal to isolate a fold and can drive the recursion.
    fn nonplanar_pentagon(mesh: &mut LoopMesh, ridge: f32) -> (FaceKey, Vec<VertexKey>) {
        // A crumpled (saddle-ish) pentagon: the corner heights alternate so no
        // single diagonal leaves both halves flat — the first cut isolates one
        // fold, and at least one resulting quad is still folded and recurses.
        let coords = [
            Vec3::new(0.0, 0.0, ridge),
            Vec3::new(1.0, 1.0, -ridge),
            Vec3::new(2.0, 0.8, ridge),
            Vec3::new(2.6, -0.6, -ridge),
            Vec3::new(0.8, -1.0, ridge),
        ];
        let face = create_face_from_verts(mesh, &coords);
        let verts = mesh.face_vert_keys(face);
        (face, verts)
    }

    #[test]
    fn planar_ngon_is_noop() {
        // §5.1 — a flat quad / pentagon / hexagon: every diagonal joins two
        // coplanar halves, so the best fold cosine is ~1 ≥ cos(angle_limit).
        for n in [4usize, 5, 6] {
            let mut mesh = LoopMesh::new();
            let coords: Vec<Vec3> = (0..n)
                .map(|i| {
                    let theta = (i as f32) * std::f32::consts::TAU / (n as f32);
                    Vec3::new(theta.cos(), theta.sin(), 0.0)
                })
                .collect();
            let face = create_face_from_verts(&mut mesh, &coords);

            let cut = mesh.connect_verts_nonplanar_out(&[face], ANGLE_TIGHT);

            assert!(cut.is_empty(), "flat {n}-gon yields no cut");
            assert_eq!(mesh.totface(), 1, "flat {n}-gon untouched");
            assert_eq!(mesh.face_len(face), n);
            mesh.validate_topology();
        }
    }

    #[test]
    fn folded_quad_one_diagonal_cut() {
        // §5.2 — a creased quad cut into two flat triangles: 4 V, 4→5 E, 1→2 F.
        let mut mesh = LoopMesh::new();
        let (face, _v) = folded_quad(&mut mesh, 0.8);
        let v0 = mesh.totvert();

        let cut = mesh.connect_verts_nonplanar_out(&[face], ANGLE_TIGHT);

        assert_eq!(mesh.totvert(), v0, "no verts created or destroyed");
        assert_eq!(cut.len(), 1, "exactly one diagonal cut");
        assert_eq!(mesh.totedge(), 5, "one new diagonal edge");
        assert_eq!(mesh.totface(), 2, "quad becomes two triangles");
        for f in mesh.face_keys() {
            assert_eq!(mesh.face_len(f), 3, "both pieces are triangles");
        }
        mesh.validate_topology();
    }

    #[test]
    fn angle_limit_gates_fold_fires_vs_blocks() {
        // §5.4 — the same creased fixture: a tight tolerance cuts, a loose one
        // (larger than the crease) leaves the quad whole. Isolates angle_limit.
        let mut mesh_fires = LoopMesh::new();
        let (face_fires, _) = folded_quad(&mut mesh_fires, 0.8);
        let cut_fires = mesh_fires.connect_verts_nonplanar_out(&[face_fires], ANGLE_TIGHT);
        assert_eq!(cut_fires.len(), 1, "tight tolerance cuts the crease");
        assert_eq!(mesh_fires.totface(), 2);
        mesh_fires.validate_topology();

        let mut mesh_blocks = LoopMesh::new();
        let (face_blocks, _) = folded_quad(&mut mesh_blocks, 0.8);
        let cut_blocks = mesh_blocks.connect_verts_nonplanar_out(&[face_blocks], ANGLE_LOOSE);
        assert!(cut_blocks.is_empty(), "loose tolerance leaves the crease whole");
        assert_eq!(mesh_blocks.totface(), 1);
        assert_eq!(mesh_blocks.face_len(face_blocks), 4);
        mesh_blocks.validate_topology();
    }

    #[test]
    fn nonplanar_pentagon_recursive_multi_cut() {
        // §5.3 — a ridged pentagon: the best diagonal isolates a fold and the
        // §3 recursion may cut again. Count is geometry-dependent, so pin the
        // structural invariants the guide makes deterministic.
        let mut mesh = LoopMesh::new();
        let (face, _) = nonplanar_pentagon(&mut mesh, 1.0);
        let v0 = mesh.totvert();
        let e0 = mesh.totedge();

        let cut = mesh.connect_verts_nonplanar_out(&[face], ANGLE_TIGHT);

        assert_eq!(mesh.totvert(), v0, "no verts created or destroyed");
        assert!(!cut.is_empty(), "a ridged pentagon forces at least one cut");
        // §2 net delta: +(p-1) edges and +(p-1) faces for p leaf pieces.
        let piece_count = mesh.totface();
        assert!(
            piece_count >= 3,
            "the ridge drives the recursion past a single cut (≥3 pieces)"
        );
        assert_eq!(cut.len(), piece_count - 1, "p-1 diagonals for p pieces");
        assert_eq!(mesh.totedge(), e0 + (piece_count - 1), "+(p-1) interior edges");
        for f in mesh.face_keys() {
            assert!(mesh.face_len(f) >= 3);
        }
        mesh.validate_topology();
    }

    #[test]
    fn folded_quad_faces_out_contains_both_triangles() {
        // §1 / §5.2 — one committed cut flags the surviving input-face handle
        // and the new half: faces.out is exactly the two resulting triangles.
        let mut mesh = LoopMesh::new();
        let (face, _) = folded_quad(&mut mesh, 0.8);

        let out = mesh.connect_verts_nonplanar_capture(&[face], ANGLE_TIGHT);

        assert_eq!(out.edges.len(), 1, "one diagonal cut");
        assert_eq!(out.faces.len(), 2, "both resulting triangles flagged");
        // The surviving handle is the original face; the other is the new half.
        assert!(out.faces.contains(&face), "surviving input handle flagged");
        let all_faces: std::collections::HashSet<FaceKey> = mesh.face_keys().into_iter().collect();
        for &f in &out.faces {
            assert!(all_faces.contains(&f), "faces.out entry is a live face");
            assert_eq!(mesh.face_len(f), 3, "each flagged leaf is a triangle");
        }
        // No-op edges.out projection still matches the historical contract.
        assert_eq!(
            mesh.connect_verts_nonplanar_out(&[], ANGLE_TIGHT).len(),
            0,
            "empty input is a no-op"
        );
        mesh.validate_topology();
    }

    #[test]
    fn faces_out_dedup_idempotent_on_recursion() {
        // §1 — a face split several times appears once. On a recursive multi-cut
        // pentagon, faces.out holds (p-1) surviving handles + (p-1) new halves
        // deduped to exactly p leaf pieces — re-flagging a re-split handle does
        // not double-count it.
        let mut mesh = LoopMesh::new();
        let (face, _) = nonplanar_pentagon(&mut mesh, 1.0);

        let out = mesh.connect_verts_nonplanar_capture(&[face], ANGLE_TIGHT);

        let piece_count = mesh.totface();
        assert!(piece_count >= 3, "the ridge drives recursion past one cut");
        // Every leaf piece is present and the set is deduped: |faces.out| == p,
        // strictly greater than the (p-1) new-half count alone — proving the
        // re-split surviving handles were flagged and merged idempotently.
        assert_eq!(out.faces.len(), piece_count, "faces.out == p leaf pieces");
        assert_eq!(out.edges.len(), piece_count - 1, "p-1 diagonals");
        assert!(
            out.faces.len() > out.edges.len(),
            "faces.out exceeds the leaf-only / edge count"
        );
        let unique: std::collections::HashSet<FaceKey> = out.faces.iter().copied().collect();
        assert_eq!(unique.len(), out.faces.len(), "faces.out carries no duplicates");
        let live: std::collections::HashSet<FaceKey> = mesh.face_keys().into_iter().collect();
        assert_eq!(unique, live, "faces.out is exactly the live leaf set");
        mesh.validate_topology();
    }

    #[test]
    fn triangle_is_noop() {
        // §3.5 — a triangle has no diagonals; excluded by the seed gate.
        let mut mesh = LoopMesh::new();
        let coords = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.5),
            Vec3::new(0.0, 1.0, 0.0),
        ];
        let face = create_face_from_verts(&mut mesh, &coords);

        let cut = mesh.connect_verts_nonplanar_out(&[face], ANGLE_TIGHT);

        assert!(cut.is_empty());
        assert_eq!(mesh.totface(), 1);
        assert_eq!(mesh.face_len(face), 3);
        mesh.validate_topology();
    }

    #[test]
    fn mixed_input_gates_each_face_independently() {
        // §6 — a folded quad plus a disjoint flat quad in one call: the folded
        // one is split, the flat one is left alone.
        let mut mesh = LoopMesh::new();
        let (folded, _) = folded_quad(&mut mesh, 0.8);
        let flat_coords = [
            Vec3::new(5.0, 0.0, 0.0),
            Vec3::new(6.0, 0.0, 0.0),
            Vec3::new(6.0, 1.0, 0.0),
            Vec3::new(5.0, 1.0, 0.0),
        ];
        let flat = create_face_from_verts(&mut mesh, &flat_coords);

        let cut = mesh.connect_verts_nonplanar_out(&[folded, flat], ANGLE_TIGHT);

        assert_eq!(cut.len(), 1, "only the folded face is cut");
        assert_eq!(mesh.face_len(flat), 4, "flat face untouched");
        assert_eq!(mesh.totface(), 3, "2 input faces become 3");
        mesh.validate_topology();
    }
}
