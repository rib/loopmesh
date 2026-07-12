use std::collections::HashMap;

use glam::DVec3;

use crate::loopmesh::*;

/// Below this corner-triangle area a vertex is treated as zero-area and pinned
/// (SMOOTH_LAPLACIAN_VERT.md §4.2 / §9). It is a degenerate guard, not a
/// parameter: well-shaped triangles sit far above it and a collinear (sliver)
/// triangle sits at ~0, so the exact magnitude only matters at the degenerate
/// boundary.
const AREA_EPSILON: f64 = 1e-12;

/// Below this rim-edge length the edge pins its endpoints rather than supplying
/// a `1/length` boundary weight (§6.2), and the post-solve gate skips an edge
/// this short to avoid dividing by zero.
const LENGTH_EPSILON: f64 = 1e-9;

/// Floor on the cross-product length (twice the triangle area) under which the
/// cotangent degrades to `0` instead of blowing up (§4.1).
const COT_EPSILON: f64 = 1e-12;

/// The §7.1 edge-length sanity gate's fixed ratio thresholds. After the solve
/// every mesh edge is checked: an edge whose solved length exceeds its original
/// length by more than `EDGE_LEN_MAX_RATIO ×` (too stretched) or falls below
/// `EDGE_LEN_MIN_RATIO ×` it (too collapsed) reverts **both** of its endpoints to
/// their input positions. They are fixed constants — not parameters, not derived
/// from `lambda_*`.
const EDGE_LEN_MAX_RATIO: f64 = 1.8;
const EDGE_LEN_MIN_RATIO: f64 = 0.15;

/// Parameters for [`LoopMesh::smooth_laplacian`] (SMOOTH_LAPLACIAN_VERT.md §1).
///
/// Laplacian smoothing assembles and solves **one** cotangent-weighted implicit
/// (backward-Euler) smoothing step `(I + λ·L̃) x = x_old` per call. It is a
/// position-only operator: no topology and no customdata change, only the
/// position layer and the cached normals it feeds.
#[derive(Clone, Copy, Debug)]
pub struct SmoothLaplacianParams {
    /// Interior smoothing strength λ, applied through the area-normalised
    /// cotangent Laplacian (§4). `0` makes every interior row the identity, so
    /// the interior is pinned (§3.2 / §9).
    pub lambda_factor: f32,
    /// Boundary smoothing strength, applied through a uniform edge-length
    /// weighting over the rim edges (§6). `0` pins the boundary verts.
    pub lambda_border: f32,
    /// Per-axis write-back gate (§7.2). The full 3-D solution is always solved;
    /// a coordinate lands in the live position only on an enabled axis.
    pub use_x: bool,
    pub use_y: bool,
    pub use_z: bool,
    /// When set, measure the signed volume before/after the solve and rescale
    /// every free vert about the origin to restore it (§8).
    pub preserve_volume: bool,
}

impl LoopMesh {
    /// Relax `verts` by one implicit cotangent-Laplacian smoothing step over the
    /// `region_faces` region (SMOOTH_LAPLACIAN_VERT.md §3 / §4).
    ///
    /// Two independent selections drive the operator (§5): `verts` are the
    /// **free** variables, and `region_faces` are the **smoothing region** whose
    /// corner triangles supply the cotangent weights and ring areas. Every other
    /// vert is **locked**, but a locked vert is *not* a Dirichlet condition pinned
    /// at its own position — the solve sees it at the **origin** (§3.2), so a free
    /// row's spoke to a locked neighbour is **dropped** (origin-pull, §4.7), never
    /// folded onto the RHS at the neighbour's position. To hold a rim at its true
    /// position, free it too with `lambda_border = 0` (§5), don't lock it.
    /// Interior free verts relax with the area-normalised cotangent Laplacian at
    /// `lambda_factor`; boundary free verts — those on a topological rim (a
    /// one-face edge) or bordering an out-of-region face (§6.1) — relax with a
    /// uniform edge-length weighting at `lambda_border`.
    ///
    /// The system `(I + λ·L̃) x = x_old` is assembled per-row-scaled (so the
    /// matrix is **not** symmetric, §4.3) and solved as a linear least-squares
    /// problem via the normal equations; the three coordinate channels share the
    /// matrix as three right-hand sides (§3.1). The `use_*` flags gate only which
    /// solved channels are written back (§7.2); `preserve_volume` brackets the
    /// write-back with a volume-restoring origin rescale (§8).
    ///
    /// `ΔV = ΔE = ΔF = 0` — no Euler operator, no customdata touched; only the
    /// position layer and the cached normals derived from it. A face-free mesh
    /// (§3.3) or an empty `verts` (§9) is a no-op. A free vert with no region
    /// support (zero ring area — a wire vert) or in a degenerate (sliver)
    /// triangle takes an identity row and stays put (§4.4 / §9). If the linear
    /// solve fails, all positions are left unchanged (§9).
    ///
    /// See `radial-mesh-guides/operators/SMOOTH_LAPLACIAN_VERT.md`.
    pub fn smooth_laplacian(
        &mut self,
        verts: &[VertexKey],
        region_faces: &[FaceKey],
        params: SmoothLaplacianParams,
    ) {
        // §3.3 — no faces: nothing to assemble, immediate no-op.
        if self.face_count() == 0 {
            return;
        }

        let vertex_slots = self.vertex_slot_count();
        let face_slots = self.face_slot_count();

        // §5 — the free-variable set (deduplicated, indexed densely for the solve).
        let mut is_free = vec![false; vertex_slots];
        let mut free_index = vec![usize::MAX; vertex_slots];
        let mut free_list: Vec<VertexKey> = Vec::new();
        for &vert in verts {
            let slot = vert.as_usize();
            if !is_free[slot] {
                is_free[slot] = true;
                free_index[slot] = free_list.len();
                free_list.push(vert);
            }
        }
        if free_list.is_empty() {
            // §9 — empty verts: every vert locked, nothing to solve.
            return;
        }
        let free_count = free_list.len();

        // §5 — the region (deduplicated face set).
        let mut in_region = vec![false; face_slots];
        let mut region_face_list: Vec<FaceKey> = Vec::new();
        for &face in region_faces {
            let slot = face.as_usize();
            if !in_region[slot] {
                in_region[slot] = true;
                region_face_list.push(face);
            }
        }

        // Cache every live vert's position once, in f64, indexed by slot.
        let mut position = vec![DVec3::ZERO; vertex_slots];
        for vert in self.vert_keys() {
            position[vert.as_usize()] = self.vertex_position(vert).as_dvec3();
        }

        // §6.1 — boundary classification: a vert is boundary if incident to a
        // topological-rim edge (exactly one face) OR to an out-of-region face.
        let mut boundary = vec![false; vertex_slots];
        for vert in self.vert_keys() {
            let mut is_boundary = false;
            let mut disk = self.disk_edges(vert);
            while let Some(edge) = disk.next(self) {
                let mut face_count = 0usize;
                let mut radial = self.radial_loops(edge);
                while let Some(loop_key) = radial.next(self) {
                    face_count += 1;
                    if !in_region[self.loop_face(loop_key).as_usize()] {
                        is_boundary = true;
                    }
                }
                if face_count == 1 {
                    is_boundary = true;
                }
            }
            boundary[vert.as_usize()] = is_boundary;
        }

        // §4.2 — per-corner accumulation over the region: ring areas, per-vertex
        // cotangent weights, and the sliver (zero-area) marks.
        let mut ring_area = vec![0.0f64; vertex_slots];
        let mut vweight = vec![0.0f64; vertex_slots];
        let mut zero_area = vec![false; vertex_slots];
        for &face in &region_face_list {
            let start = self.face_first_loop(face);
            let mut loop_key = start;
            loop {
                let curr = self.loop_vertex(loop_key);
                let next = self.loop_vertex(self.loop_next(loop_key));
                let prev = self.loop_vertex(self.loop_prev(loop_key));
                let pc = position[curr.as_usize()];
                let pn = position[next.as_usize()];
                let pp = position[prev.as_usize()];

                let area = triangle_area(pp, pc, pn);
                if area < AREA_EPSILON {
                    zero_area[curr.as_usize()] = true;
                }
                // All-three-corners accumulation (§4.2 / §4 callout): every corner
                // iteration deposits the corner-triangle area and its three halved
                // corner cotangents into *all three* of the triangle's verts, never
                // into the pivot alone. A triangle face therefore triple-counts its
                // area into each of its verts (three corner iterations, same whole
                // triangle). Each vert's `vweight` collects the cotangents at the
                // *other two* corners — the classical `cot α + cot β` spoke weight,
                // gathered one corner-triangle at a time. The triple-count appears
                // identically in the off-diagonal scale's denominator and the
                // accumulated spoke cotangents, so it cancels out of the row sum
                // (§4.3 / §4.6).
                let c_curr = cot_at(pc, pn, pp) * 0.5;
                let c_next = cot_at(pn, pp, pc) * 0.5;
                let c_prev = cot_at(pp, pc, pn) * 0.5;
                ring_area[prev.as_usize()] += area;
                ring_area[curr.as_usize()] += area;
                ring_area[next.as_usize()] += area;
                vweight[prev.as_usize()] += c_curr + c_next;
                vweight[curr.as_usize()] += c_next + c_prev;
                vweight[next.as_usize()] += c_curr + c_prev;

                loop_key = self.loop_next(loop_key);
                if loop_key == start {
                    break;
                }
            }
        }

        // §6.2 — the length-weight rim edge set: topological-rim edges (exactly
        // one face) that are not region-selected. The loopmesh region is carried
        // entirely by faces, so no edge is region-selected; every one-face edge
        // qualifies. A qualifying edge below the min length pins its endpoints.
        let mut length_sum = vec![0.0f64; vertex_slots];
        for edge in self.edge_keys() {
            let mut face_count = 0usize;
            let mut radial = self.radial_loops(edge);
            while radial.next(self).is_some() {
                face_count += 1;
            }
            if face_count != 1 {
                continue;
            }
            let [a, b] = self.edge_vertices(edge);
            let length = (position[a.as_usize()] - position[b.as_usize()]).length();
            if length < LENGTH_EPSILON {
                zero_area[a.as_usize()] = true;
                zero_area[b.as_usize()] = true;
            } else {
                length_sum[a.as_usize()] += 1.0 / length;
                length_sum[b.as_usize()] += 1.0 / length;
            }
        }

        // Rows over the free verts. A locked vert is seen at the origin, so its
        // coupling is dropped entirely (§3.2) — never moved to the right-hand side.
        let mut matrix_rows: Vec<Vec<(usize, f64)>> = vec![Vec::new(); free_count];
        let mut rhs = [
            vec![0.0f64; free_count],
            vec![0.0f64; free_count],
            vec![0.0f64; free_count],
        ];
        // The interior per-spoke off-diagonal scale, filled per eligible row.
        let mut scale = vec![0.0f64; vertex_slots];

        let lambda_factor = params.lambda_factor as f64;
        let lambda_border = params.lambda_border as f64;

        // §4.4 / §6.3 — diagonal, RHS, and (for boundary rows) the length-based
        // off-diagonals.
        for &vert in &free_list {
            let slot = vert.as_usize();
            let row = free_index[slot];
            let here = position[slot];
            rhs[0][row] = here.x;
            rhs[1][row] = here.y;
            rhs[2][row] = here.z;

            let degenerate = zero_area[slot] || ring_area[slot] == 0.0;
            if degenerate {
                // §4.4 / §9 — pure identity row: stays exactly put.
                matrix_rows[row].push((row, 1.0));
            } else if boundary[slot] {
                // §6.3 — boundary row: uniform edge-length weighting.
                matrix_rows[row].push((row, 1.0 + lambda_border * 2.0));
                // §6.3 empty-length_sum guard: 0, never −2λ/0 → nan.
                let length_scale = if length_sum[slot] == 0.0 {
                    0.0
                } else {
                    -lambda_border * 2.0 / length_sum[slot]
                };
                let mut disk = self.disk_edges(vert);
                while let Some(edge) = disk.next(self) {
                    let mut face_count = 0usize;
                    let mut radial = self.radial_loops(edge);
                    while radial.next(self).is_some() {
                        face_count += 1;
                    }
                    if face_count != 1 {
                        continue;
                    }
                    let other = match self.edges.get(edge).other_vertex(vert) {
                        Some(other) => other,
                        None => continue,
                    };
                    let length = (here - position[other.as_usize()]).length();
                    if length < LENGTH_EPSILON {
                        continue;
                    }
                    let coefficient = (1.0 / length) * length_scale;
                    lay_off_diagonal(
                        &mut matrix_rows,
                        &is_free,
                        &free_index,
                        row,
                        other,
                        coefficient,
                    );
                }
            } else {
                // §4.4 — interior row: cotangent Laplacian.
                matrix_rows[row].push((row, 1.0 + lambda_factor / (4.0 * ring_area[slot])));
                scale[slot] = -lambda_factor / (4.0 * vweight[slot] * ring_area[slot]);
            }
        }

        // §4.3 / §6.3 — interior cotangent off-diagonals: a second sweep over the
        // region corners. The matrix-add is gated on the **row/pivot** vert's
        // interior-eligibility alone (non-boundary, non-zero-area, with ring
        // area); the column vert's status never suppresses the add. An interior
        // row lays an off-diagonal toward *all* its corner-triangle neighbours —
        // boundary and zero-area ones included. What the column's status changes
        // is what the term does at solve time (§3.2 / §4.3): a **free** column
        // stays a live coupling and feeds its solved position back into row `i`; a
        // **locked** column is seen at the origin, so its spoke folds to zero and
        // is dropped — it reaches neither the matrix nor the RHS. A free interior
        // vert with a locked neighbour therefore keeps its full diagonal but loses
        // that spoke's off-diagonal, biasing the solve toward the origin (§4.7).
        // Boundary verts never own a cotangent row (the row gate), so they only
        // ever appear here as columns.
        {
            let eligible =
                |slot: usize| !boundary[slot] && !zero_area[slot] && ring_area[slot] > 0.0;
            let mut lay_cotangent = |i: VertexKey, j: VertexKey, cotangent: f64| {
                let si = i.as_usize();
                if !is_free[si] || !eligible(si) {
                    return;
                }
                let coefficient = cotangent * scale[si];
                lay_off_diagonal(
                    &mut matrix_rows,
                    &is_free,
                    &free_index,
                    free_index[si],
                    j,
                    coefficient,
                );
            };
            for &face in &region_face_list {
                let start = self.face_first_loop(face);
                let mut loop_key = start;
                loop {
                    let curr = self.loop_vertex(loop_key);
                    let next = self.loop_vertex(self.loop_next(loop_key));
                    let prev = self.loop_vertex(self.loop_prev(loop_key));
                    let pc = position[curr.as_usize()];
                    let pn = position[next.as_usize()];
                    let pp = position[prev.as_usize()];
                    let c_curr = cot_at(pc, pn, pp) * 0.5;
                    let c_next = cot_at(pn, pp, pc) * 0.5;
                    let c_prev = cot_at(pp, pc, pn) * 0.5;

                    // All-three-corners off-diagonals (§4.3): every *ordered* pair
                    // of the corner triangle's verts gets the cotangent opposite
                    // that spoke — `(i, j)` carries the cotangent at the third vert.
                    // Each corner iteration lays all six directed pairs, mirroring
                    // the §4.2 sweep that deposited into all three verts, so the
                    // accumulated spoke cotangent into row `i` sums (over every
                    // corner iteration touching both endpoints) to exactly
                    // `vweight[i]`. That cancels the diagonal's `λ/(4·ring_area[i])`
                    // — `Σ_j off[i][j] = vweight[i]·scale[i] = −λ/(4·ring_area[i])`
                    // — so the L̃ row sums to zero regardless of the triple-count.
                    lay_cotangent(curr, next, c_prev);
                    lay_cotangent(next, curr, c_prev);
                    lay_cotangent(curr, prev, c_next);
                    lay_cotangent(prev, curr, c_next);
                    lay_cotangent(next, prev, c_curr);
                    lay_cotangent(prev, next, c_curr);

                    loop_key = self.loop_next(loop_key);
                    if loop_key == start {
                        break;
                    }
                }
            }
        }

        // Normal equations: solve the (non-symmetric, square) system as least
        // squares — AᵀA x = Aᵀb — with the SPD sparse Cholesky (§3.1 / §4.3).
        let mut normal_matrix: HashMap<(usize, usize), f64> = HashMap::new();
        let mut normal_rhs = [
            vec![0.0f64; free_count],
            vec![0.0f64; free_count],
            vec![0.0f64; free_count],
        ];
        for row in 0..free_count {
            for &(p, value_p) in &matrix_rows[row] {
                normal_rhs[0][p] += value_p * rhs[0][row];
                normal_rhs[1][p] += value_p * rhs[1][row];
                normal_rhs[2][p] += value_p * rhs[2][row];
                for &(q, value_q) in &matrix_rows[row] {
                    *normal_matrix.entry((p, q)).or_insert(0.0) += value_p * value_q;
                }
            }
        }
        let triplets: Vec<(usize, usize, f64)> = normal_matrix
            .into_iter()
            .map(|((r, c), v)| (r, c, v))
            .collect();

        let factor = match SparseCholesky::factor(free_count, &triplets) {
            Ok(factor) => factor,
            Err(_) => return, // §9 — solve failure: leave positions unchanged.
        };
        let solution = match factor.solve_columns(&[&normal_rhs[0], &normal_rhs[1], &normal_rhs[2]])
        {
            Ok(solution) => solution,
            Err(_) => return,
        };
        let solved: Vec<DVec3> = (0..free_count)
            .map(|row| DVec3::new(solution[0][row], solution[1][row], solution[2][row]))
            .collect();

        // §7.1 — edge-length sanity gate: sweep **every** mesh edge (not gated on
        // region/selection) and revert verts whose solved edges blew up or
        // collapsed. Marks both endpoints zero-area → excluded from write-back, so
        // they keep their full input coordinate on every axis (the §7.2 axis gate
        // does not re-enter for a reverted vert). The revert is symmetric: a single
        // blown edge reverts the verts at *both* ends.
        // §7.1 / §3.2 — the gate reads the *solver's* variable values, not the
        // mesh positions: a free vert at its computed position, a **locked** vert
        // at the solver's default, the **origin**. So an edge from a free vert far
        // from the origin to a locked vert measures a large solved length and the
        // free vert is reverted to its input.
        let solved_position = |vert: VertexKey| -> DVec3 {
            let slot = vert.as_usize();
            if is_free[slot] {
                solved[free_index[slot]]
            } else {
                DVec3::ZERO
            }
        };
        for edge in self.edge_keys() {
            let [a, b] = self.edge_vertices(edge);
            // §7.1 — original = endpoints' input positions; solved = solver
            // variable values (locked endpoint at the origin via `solved_position`).
            let original = (position[a.as_usize()] - position[b.as_usize()]).length();
            let solved_length = (solved_position(a) - solved_position(b)).length();
            // Multiplication form (§7.1): never forms the ratio, so a zero
            // original_length never divides. Trip if too stretched or too collapsed.
            if solved_length > original * EDGE_LEN_MAX_RATIO
                || solved_length < original * EDGE_LEN_MIN_RATIO
            {
                zero_area[a.as_usize()] = true;
                zero_area[b.as_usize()] = true;
            }
        }

        // §8 — measure volume before write-back.
        let volume_before = if params.preserve_volume {
            self.mesh_signed_volume()
        } else {
            0.0
        };

        // §7.2 — axis-gated write-back. Gated/degenerate verts keep their input.
        for &vert in &free_list {
            let slot = vert.as_usize();
            if zero_area[slot] {
                continue;
            }
            let target = solved[free_index[slot]];
            let mut here = self.vertex_position(vert);
            if params.use_x {
                here.x = target.x as f32;
            }
            if params.use_y {
                here.y = target.y as f32;
            }
            if params.use_z {
                here.z = target.z as f32;
            }
            self.set_vertex_position(vert, here);
        }

        // §8 — restore the pre-smooth volume by an origin rescale of the free
        // verts on the enabled axes.
        if params.preserve_volume {
            let volume_after = self.mesh_signed_volume();
            if volume_after != 0.0 {
                let beta = (volume_before / volume_after).cbrt() as f32;
                for &vert in &free_list {
                    let mut here = self.vertex_position(vert);
                    if params.use_x {
                        here.x *= beta;
                    }
                    if params.use_y {
                        here.y *= beta;
                    }
                    if params.use_z {
                        here.z *= beta;
                    }
                    self.set_vertex_position(vert, here);
                }
            }
        }

        // §1 — recompute cached normals on the affected geometry; no topology and
        // no customdata is touched.
        self.refresh_normals_around_verts(&free_list);
    }

    /// Signed volume over all faces, fan-triangulated from each face's first
    /// loop vertex (SMOOTH_LAPLACIAN_VERT.md §8). For a closed surface the sum is
    /// triangulation-independent, so the fan origin does not affect the result.
    fn mesh_signed_volume(&self) -> f64 {
        let mut total = 0.0f64;
        for face in self.face_keys() {
            let start = self.face_first_loop(face);
            let apex = self.vertex_position(self.loop_vertex(start)).as_dvec3();
            let mut loop_key = self.loop_next(start);
            loop {
                let next = self.loop_next(loop_key);
                if next == start {
                    break;
                }
                let b = self.vertex_position(self.loop_vertex(loop_key)).as_dvec3();
                let c = self.vertex_position(self.loop_vertex(next)).as_dvec3();
                total += apex.dot(b.cross(c));
                loop_key = next;
            }
        }
        total / 6.0
    }
}

/// `cot(angle at A in triangle A,B,C) = (AB · AC) / |AB × AC|`
/// (SMOOTH_LAPLACIAN_VERT.md §4.1). Degenerate triangles yield `0`, never an
/// infinity.
fn cot_at(a: DVec3, b: DVec3, c: DVec3) -> f64 {
    let ab = b - a;
    let ac = c - a;
    let denom = ab.cross(ac).length();
    if denom > COT_EPSILON {
        ab.dot(ac) / denom
    } else {
        0.0
    }
}

fn triangle_area(a: DVec3, b: DVec3, c: DVec3) -> f64 {
    0.5 * (b - a).cross(c - a).length()
}

/// Lay one directed off-diagonal `matrix[row][col(j)] += coefficient` for a
/// **free** column. A **locked** column is seen by the solve at the **origin**
/// (SMOOTH_LAPLACIAN_VERT.md §3.2), so `coefficient · origin = 0`: the spoke
/// contributes nothing to the matrix **and** nothing to the right-hand side and
/// is simply dropped — it is *not* folded onto the RHS at the locked vert's
/// position. Dropping the spoke leaves the row's diagonal carrying its full
/// Laplacian magnitude (§4.4) but only its free off-diagonals, so the row
/// over-balances and biases the solve toward the origin (the §4.7 origin-pull).
fn lay_off_diagonal(
    matrix_rows: &mut [Vec<(usize, f64)>],
    is_free: &[bool],
    free_index: &[usize],
    row: usize,
    j: VertexKey,
    coefficient: f64,
) {
    let slot = j.as_usize();
    if is_free[slot] {
        matrix_rows[row].push((free_index[slot], coefficient));
    }
}

#[cfg(test)]
mod tests {
    use std::f32::consts::PI;

    use glam::Vec3;

    use crate::loopmesh::*;

    fn all_axes(lambda_factor: f32, lambda_border: f32) -> SmoothLaplacianParams {
        SmoothLaplacianParams {
            lambda_factor,
            lambda_border,
            use_x: true,
            use_y: true,
            use_z: true,
            preserve_volume: false,
        }
    }

    /// A hexagonal triangle fan: a centre vert and six unit-circle rim verts in
    /// the `z = 0` plane, with six triangle faces pivoting on the centre.
    /// Returns `(centre, ring)`. The centre is interior (every spoke has two
    /// faces); the rim verts are boundary (the ring edges are one-face).
    fn hex_fan(mesh: &mut LoopMesh, centre_pos: Vec3) -> (VertexKey, [VertexKey; 6]) {
        let centre = mesh.create_vertex(centre_pos);
        let mut ring = [centre; 6];
        for k in 0..6 {
            let angle = k as f32 * PI / 3.0;
            ring[k] = mesh.create_vertex(Vec3::new(angle.cos(), angle.sin(), 0.0));
        }
        for k in 0..6 {
            let a = ring[k];
            let b = ring[(k + 1) % 6];
            mesh.face_create_verts(&[centre, a, b], FaceCreateFlags::NO_DUPLICATE);
        }
        (centre, ring)
    }

    fn centre_ring_area(centre_pos: Vec3) -> f64 {
        fan_ring_area(centre_pos, 0.0)
    }

    /// Sum of the six fan-triangle areas of a hex fan whose centre is at
    /// `centre_pos` and whose rim sits on the `z = rim_z` plane at unit radius.
    /// Multiply by 3 for the all-three-corners ring area of the centre (§4.6).
    fn fan_ring_area(centre_pos: Vec3, rim_z: f32) -> f64 {
        let c = centre_pos.as_dvec3();
        let mut area = 0.0;
        for k in 0..6 {
            let a0 = k as f32 * PI / 3.0;
            let a1 = (k + 1) as f32 * PI / 3.0;
            let p0 = Vec3::new(a0.cos(), a0.sin(), rim_z).as_dvec3();
            let p1 = Vec3::new(a1.cos(), a1.sin(), rim_z).as_dvec3();
            area += 0.5 * (p0 - c).cross(p1 - c).length();
        }
        area
    }

    /// §3.2 / §4.7: the headline interior relaxation under a locked surround. The
    /// lone free interior vert, lifted above the rim, has its six rim neighbours
    /// **locked**, so each is seen by the solve at the **origin** and its cotangent
    /// spoke is **dropped** (§3.2 / §4.3) — not folded onto the RHS at the rim's
    /// position. Row `centre` collapses to its diagonal alone,
    /// `(1 + λ/(4·A))·x = x_old`, so the centre is pulled straight toward the
    /// origin by `1 / (1 + λ/(4·A))`. Here `x_old = (0, 0, height)` already sits on
    /// the z-axis, so the flat axes stay `0` and only the `z` bump shrinks — the
    /// observable headline relaxation. (The
    /// `free_rim_feeds_held_position_to_interior` test frees the rim instead — the
    /// §5 idiom — so the rim's true height feeds in.)
    ///
    /// `A` is the **all-three-corners** ring area (§4 callout / §4.6): each of the
    /// six fan faces is a triangle whose three corner iterations all deposit the
    /// face area into the centre, so the centre's ring area is `3 ·` the sum of
    /// the six fan-triangle areas — *not* the pivot-only single count.
    #[test]
    fn interior_vert_relaxes_toward_plane() {
        let mut mesh = LoopMesh::new();
        let height = 1.5;
        let (centre, _) = hex_fan(&mut mesh, Vec3::new(0.0, 0.0, height));
        let faces = mesh.face_keys();

        mesh.smooth_laplacian(&[centre], &faces, all_axes(2.0, 0.0));

        let area = 3.0 * centre_ring_area(Vec3::new(0.0, 0.0, height));
        let expected_z = height as f64 / (1.0 + 2.0 / (4.0 * area));
        let result = mesh.vertex_position(centre);
        assert!(
            result.x.abs() < 1e-5 && result.y.abs() < 1e-5,
            "flat axes fixed: {result:?}"
        );
        assert!(
            result.z > 0.0 && result.z < height,
            "z bump must shrink toward the plane: {}",
            result.z
        );
        assert!(
            (result.z as f64 - expected_z).abs() < 1e-4,
            "z = {} expected {expected_z}",
            result.z
        );
        mesh.validate_topology();
    }

    /// §5 / §6.3: the parity-correct held-boundary idiom. To relax the interior
    /// toward a genuinely held rim, the rim verts must be **free** (in the `verts`
    /// slot) with `lambda_border = 0` — *not* locked. A free rim vert with
    /// `lambda_border = 0` gets an identity row, so it stays at its **true**
    /// position, and being free it feeds that true position into the centre's
    /// interior cotangent row as a live column. Same hex fan with the rim lifted
    /// to `z = h_rim`: with the whole fan free the centre keeps all six spokes, so
    /// `Σ_j off[c][j] = −λ/(4·A)` and the held rim height flows in, giving
    /// `z_c = (h_c + h_rim·m) / (1 + m)` with `m = λ/(4·A)`. Locking the rim
    /// instead would drop every spoke (rim seen at the origin) and land the centre
    /// at the pure origin-pull `h_c/(1 + m)` (§4.7) — a *different*, smaller value
    /// this fixture distinguishes. This is the contrast the corrected model turns
    /// on: held-boundary needs a **free** rim, never a locked one.
    #[test]
    fn free_rim_feeds_held_position_to_interior() {
        let h_c = 1.5f32;
        let h_rim = 0.5f32;
        let lambda = 2.0f64;

        let mut mesh = LoopMesh::new();
        let centre = mesh.create_vertex(Vec3::new(0.0, 0.0, h_c));
        let mut ring = [centre; 6];
        for k in 0..6 {
            let angle = k as f32 * PI / 3.0;
            ring[k] = mesh.create_vertex(Vec3::new(angle.cos(), angle.sin(), h_rim));
        }
        for k in 0..6 {
            mesh.face_create_verts(
                &[centre, ring[k], ring[(k + 1) % 6]],
                FaceCreateFlags::NO_DUPLICATE,
            );
        }
        let faces = mesh.face_keys();

        // Free the whole fan (centre + rim); lambda_border = 0 holds the rim at
        // its true height via identity boundary rows.
        let mut free = vec![centre];
        free.extend(ring);
        mesh.smooth_laplacian(&free, &faces, all_axes(lambda as f32, 0.0));

        let area = 3.0 * fan_ring_area(Vec3::new(0.0, 0.0, h_c), h_rim);
        let m = lambda / (4.0 * area);
        let expected_z = (h_c as f64 + h_rim as f64 * m) / (1.0 + m);
        let origin_pull_z = h_c as f64 / (1.0 + m); // what locking the rim would give

        let result = mesh.vertex_position(centre);
        assert!(
            result.x.abs() < 1e-5 && result.y.abs() < 1e-5,
            "flat axes fixed by symmetry: {result:?}"
        );
        assert!(
            (result.z as f64 - expected_z).abs() < 1e-4,
            "z = {} expected held-rim {expected_z} (locked-rim origin-pull would be {origin_pull_z})",
            result.z
        );
        // The held-rim coupling must be observable: it differs measurably from the
        // origin-pull a locked rim would produce.
        assert!(
            (expected_z - origin_pull_z).abs() > 1e-3,
            "fixture must distinguish held-rim from origin-pull"
        );
        // The free rim stays at its true height (identity rows at lambda_border = 0).
        for &v in &ring {
            assert!(
                (mesh.vertex_position(v).z - h_rim).abs() < 1e-5,
                "free rim with lambda_border = 0 must stay at its true height"
            );
        }
        mesh.validate_topology();
    }

    /// §4.7: a free interior vert all of whose ring neighbours are **locked**
    /// collapses to a pure origin-pull — `x_C = x_old_C / (1 + λ/(4·ring_area))` —
    /// because every locked spoke is dropped (the locked vert is seen at the
    /// origin, not at its own position, §3.2 / §4.3), so no neighbour term
    /// survives. The centre is pulled straight toward the origin by a **uniform
    /// scalar shrink on every axis**, *not* toward the rim. (Guide's worked
    /// instance: `ring_area = 9`, `λ = 1` → factor `36/37`; here `ring_area` is
    /// read from the actual fan geometry.)
    #[test]
    fn locked_rim_pulls_interior_toward_origin() {
        let mut mesh = LoopMesh::new();
        let start = Vec3::new(0.3, 0.2, 0.5);
        let (centre, _) = hex_fan(&mut mesh, start);
        let faces = mesh.face_keys();
        let lambda = 1.0f64;

        // Only the centre is free; the six rim verts are locked → seen at origin.
        mesh.smooth_laplacian(&[centre], &faces, all_axes(lambda as f32, 0.0));

        let area = 3.0 * fan_ring_area(start, 0.0);
        let factor = 1.0 / (1.0 + lambda / (4.0 * area));
        let expected = start.as_dvec3() * factor;
        let result = mesh.vertex_position(centre).as_dvec3();
        assert!(
            (result - expected).length() < 1e-5,
            "origin-pull closed form: {result:?} expected {expected:?}"
        );
        // Pure origin-pull: a uniform shrink along the origin→input ray — the same
        // scalar on every axis — *not* a displacement toward the rim centroid
        // (which sits in the z = 0 plane, a different direction).
        let ratio_x = result.x / start.x as f64;
        let ratio_z = result.z / start.z as f64;
        assert!(
            (ratio_x - ratio_z).abs() < 1e-6,
            "uniform scalar shrink across axes: x×{ratio_x} vs z×{ratio_z}"
        );
        assert!(factor < 1.0, "must shrink toward the origin");
        mesh.validate_topology();
    }

    /// §7.1: the edge-length sanity gate reverts a free vert whose solved edge to
    /// a **locked** neighbour blows past the `1.8×` upper ratio. A hex fan offset
    /// far from the origin: the centre is free, the rim **locked** (seen at the
    /// origin, §3.2). The centre's solved origin-seen edge length is its whole
    /// distance to the origin (~the offset), while the original centre–rim edge is
    /// ~unit — a ratio of ~10, far above `1.8`. The gate trips and the centre is
    /// reverted to its **input** position, keeping the full input coordinate on
    /// every axis (the §7.2 axis gate does not re-enter). Contrast
    /// `locked_rim_pulls_interior_toward_origin`, whose origin-seen ratio ≈0.68
    /// sits inside `[0.15, 1.8]`, so the gate does *not* fire and the centre moves.
    #[test]
    fn gate_reverts_free_vert_with_blown_up_locked_edge() {
        let mut mesh = LoopMesh::new();
        let offset = Vec3::new(10.0, 0.0, 0.0);
        let centre = mesh.create_vertex(offset);
        let mut ring = [centre; 6];
        for k in 0..6 {
            let angle = k as f32 * PI / 3.0;
            ring[k] = mesh.create_vertex(offset + Vec3::new(angle.cos(), angle.sin(), 0.0));
        }
        for k in 0..6 {
            mesh.face_create_verts(
                &[centre, ring[k], ring[(k + 1) % 6]],
                FaceCreateFlags::NO_DUPLICATE,
            );
        }
        let faces = mesh.face_keys();

        // Only the centre is free; the locked rim is seen at the origin, so the
        // centre's origin-seen edge length ~10 >> 1.8 × (~1) original → gate trips.
        mesh.smooth_laplacian(&[centre], &faces, all_axes(2.0, 0.0));

        let result = mesh.vertex_position(centre);
        assert!(
            (result - offset).length() < 1e-6,
            "gate must revert the centre to its input position: {result:?}"
        );
        mesh.validate_topology();
    }

    /// §3.2 / §9: with both λ values zero every row is the identity, so the call
    /// is a position no-op even with every vert free.
    #[test]
    fn lambda_zero_is_noop() {
        let mut mesh = LoopMesh::new();
        let (centre, ring) = hex_fan(&mut mesh, Vec3::new(0.0, 0.0, 1.5));
        let faces = mesh.face_keys();
        let before: Vec<Vec3> = std::iter::once(centre)
            .chain(ring)
            .map(|v| mesh.vertex_position(v))
            .collect();

        let mut all_verts = vec![centre];
        all_verts.extend(ring);
        mesh.smooth_laplacian(&all_verts, &faces, all_axes(0.0, 0.0));

        for (v, &p) in std::iter::once(centre).chain(ring).zip(&before) {
            assert!((mesh.vertex_position(v) - p).length() < 1e-6);
        }
        mesh.validate_topology();
    }

    /// §7.2: per-axis write-back. With `use_x` only, the free interior vert's `x`
    /// relaxes (shrinks toward the origin) while its `y` and `z` are held exactly
    /// at the input, even though the full solve would have moved all three.
    #[test]
    fn per_axis_lock_writes_only_enabled_axis() {
        let mut mesh = LoopMesh::new();
        let start = Vec3::new(0.3, 0.2, 1.0);
        let (centre, _) = hex_fan(&mut mesh, start);
        let faces = mesh.face_keys();

        mesh.smooth_laplacian(
            &[centre],
            &faces,
            SmoothLaplacianParams {
                lambda_factor: 2.0,
                lambda_border: 0.0,
                use_x: true,
                use_y: false,
                use_z: false,
                preserve_volume: false,
            },
        );

        let result = mesh.vertex_position(centre);
        assert!((result.y - 0.2).abs() < 1e-6, "y locked: {}", result.y);
        assert!((result.z - 1.0).abs() < 1e-6, "z locked: {}", result.z);
        assert!(
            result.x > 0.0 && result.x < 0.3,
            "x relaxed inward: {}",
            result.x
        );
        mesh.validate_topology();
    }

    /// §6: boundary handling. Freeing the rim verts, `lambda_border = 0` pins
    /// them (identity rows) while `lambda_border > 0` relaxes them inward under
    /// the uniform edge-length weighting.
    #[test]
    fn lambda_border_holds_then_smooths_rim() {
        // lambda_border = 0 → rim pinned.
        let mut held = LoopMesh::new();
        let (_, ring) = hex_fan(&mut held, Vec3::new(0.0, 0.0, 0.0));
        let faces = held.face_keys();
        let before: Vec<Vec3> = ring.iter().map(|&v| held.vertex_position(v)).collect();
        held.smooth_laplacian(&ring, &faces, all_axes(0.0, 0.0));
        for (&v, &p) in ring.iter().zip(&before) {
            assert!(
                (held.vertex_position(v) - p).length() < 1e-6,
                "lambda_border = 0 must pin the rim"
            );
        }

        // lambda_border > 0 → rim contracts (each vert pulled toward its two
        // neighbours' midpoint, which lies inside the unit circle).
        let mut smoothed = LoopMesh::new();
        let (_, ring2) = hex_fan(&mut smoothed, Vec3::new(0.0, 0.0, 0.0));
        let faces2 = smoothed.face_keys();
        smoothed.smooth_laplacian(&ring2, &faces2, all_axes(0.0, 0.5));
        for &v in &ring2 {
            let radius = smoothed.vertex_position(v).length();
            assert!(
                radius < 1.0 - 1e-4,
                "rim must contract inward: r = {radius}"
            );
        }
        smoothed.validate_topology();
    }

    /// §4.3 / §4.4 / §5: a flat, **all-free** region is a fixed point. A regular
    /// planar grid is discretely harmonic — each interior vert sits at the
    /// cotangent-weighted average of its four axis neighbours by symmetry — so one
    /// implicit step leaves it exactly put. With **every** vert free (the §5
    /// idiom) each interior row keeps all its spokes, so `Σ_j off[i][j]` collapses
    /// to `−λ/(4·ring_area[i])` and exactly cancels the diagonal's Laplacian term;
    /// the row sums to one and the planar coordinates are preserved. The rim verts
    /// are boundary and, with `lambda_border = 0`, take identity rows: they stay
    /// put *and*, being free, feed their true positions into the interior rows.
    ///
    /// This pins the row-sum-zero property of the off-diagonal sweep under the
    /// accumulate-to-all-three model: each directed off-diagonal is laid once per
    /// corner iteration touching both endpoints (so an interior spoke is
    /// multi-counted), and the multi-count cancels because `vweight_raw[i]` in the
    /// per-spoke scale's denominator is multi-counted in lockstep. A regression
    /// breaking that lockstep would leave the row sum non-zero and drift the verts.
    /// The whole grid is free so the interior verts couple to one another as live
    /// **free–free** matrix entries — exactly where the multi-count manifests —
    /// while the held rim is the parity-correct boundary, **not** a locked surround
    /// (a locked surround would drop the interior→rim spokes and pull the interior
    /// toward the origin, §3.2 / §4.7).
    #[test]
    fn flat_interior_grid_is_fixed_point() {
        let mut mesh = LoopMesh::new();
        let grid: Vec<Vec<VertexKey>> = (0..6)
            .map(|i| {
                (0..6)
                    .map(|j| mesh.create_vertex(Vec3::new(i as f32, j as f32, 0.0)))
                    .collect()
            })
            .collect();
        for i in 0..5 {
            for j in 0..5 {
                mesh.face_create_verts(
                    &[
                        grid[i][j],
                        grid[i + 1][j],
                        grid[i + 1][j + 1],
                        grid[i][j + 1],
                    ],
                    FaceCreateFlags::NO_DUPLICATE,
                );
            }
        }
        let faces = mesh.face_keys();
        // The whole grid is free (§5): the rim holds at its true position with
        // lambda_border = 0 while the interior verts couple as live free–free
        // matrix entries.
        let free: Vec<VertexKey> = grid.iter().flatten().copied().collect();
        let before: Vec<Vec3> = free.iter().map(|&v| mesh.vertex_position(v)).collect();

        mesh.smooth_laplacian(&free, &faces, all_axes(2.0, 0.0));

        for (&v, &p) in free.iter().zip(&before) {
            let moved = (mesh.vertex_position(v) - p).length();
            assert!(
                moved < 1e-6,
                "flat all-free region must be a fixed point: vert moved {moved}"
            );
        }
        mesh.validate_topology();
    }

    /// §4.2 / §9: a zero-area (collinear) face marks its corners zero-area, so
    /// every vert takes an identity row and stays exactly put.
    #[test]
    fn degenerate_collinear_face_stays_put() {
        let mut mesh = LoopMesh::new();
        let a = mesh.create_vertex(Vec3::new(0.0, 0.0, 0.0));
        let b = mesh.create_vertex(Vec3::new(1.0, 0.0, 0.0));
        let c = mesh.create_vertex(Vec3::new(2.0, 0.0, 0.0));
        mesh.face_create_verts(&[a, b, c], FaceCreateFlags::NO_DUPLICATE);
        let faces = mesh.face_keys();

        mesh.smooth_laplacian(&[a, b, c], &faces, all_axes(5.0, 5.0));

        assert!((mesh.vertex_position(a) - Vec3::new(0.0, 0.0, 0.0)).length() < 1e-6);
        assert!((mesh.vertex_position(b) - Vec3::new(1.0, 0.0, 0.0)).length() < 1e-6);
        assert!((mesh.vertex_position(c) - Vec3::new(2.0, 0.0, 0.0)).length() < 1e-6);
    }
}
