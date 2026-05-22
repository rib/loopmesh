//! Non-destructive, per-face corner-triple triangulation.
//!
//! Bridges [`crate::loopmesh::LoopMesh`] with the
//! [`crate::triangulate`] kernel. Produces `[LoopKey; 3]` triples
//! identifying the corners of each output triangle, without mutating
//! the mesh.
//!
//! Algorithms are documented in
//! `radial-mesh-guides/BMESH_POLYGON_TRIANGULATION.md`. The three-case
//! dispatch (§2), the triangle fast path (§4), the render-time quad
//! flip detection (§5), and the n-gon ear-clipping path (§7) all live
//! here. The operator-only quad-method policies (§6) are not exposed —
//! they belong to a future destructive triangulate operator.

use glam::Vec2;

use crate::loopmesh::{FaceKey, LoopKey, LoopMesh, MeshIter};
use crate::nub_mesh_convert::TriangulationPolicy;
use crate::triangulate;

/// Triangulate one face, appending `[LoopKey; 3]` triples to
/// `out_tris`.
///
/// `scratch_loops` and `scratch_proj` are caller-owned reusable
/// buffers; they're cleared at the top of each call. Reusing them
/// across many faces keeps allocations out of the hot loop.
///
/// Three-case dispatch (§2):
/// - `face.len == 3` → one triple in cycle order (§4).
/// - `face.len == 4` → two triples picked by [`TriangulationPolicy`]
///   (§5 EarClip variant, fan-from-first-loop for Fan variant).
/// - `face.len >= 5` → n-gon ear-clip on the negated axis-dominant 2D
///   projection (§3, §7) for EarClip; fan for Fan.
///
/// Faces with fewer than 3 loops are skipped (no output).
pub fn corner_triangulation(
    mesh: &LoopMesh,
    face: FaceKey,
    policy: TriangulationPolicy,
    scratch_loops: &mut Vec<LoopKey>,
    scratch_proj: &mut Vec<Vec2>,
    out_tris: &mut Vec<[LoopKey; 3]>,
) {
    scratch_loops.clear();
    let mut it = mesh.face_loops(face);
    while let Some(l) = it.next(mesh) {
        scratch_loops.push(l);
    }
    let n = scratch_loops.len();
    if n < 3 {
        return;
    }

    if n == 3 {
        // §4 — emit one triple in cycle order.
        out_tris.push([scratch_loops[0], scratch_loops[1], scratch_loops[2]]);
        return;
    }

    match policy {
        TriangulationPolicy::Fan => emit_fan(scratch_loops, out_tris),
        TriangulationPolicy::EarClip => {
            if n == 4 {
                emit_quad_render_time(mesh, scratch_loops, out_tris);
            } else {
                emit_ngon_ear_clip(mesh, face, scratch_loops, scratch_proj, out_tris);
            }
        }
    }
}

/// Fan-from-first-loop. Cheap and correct for convex faces; produces
/// incorrect output on concave inputs. Preserved as a fallback for
/// callers that explicitly opt into [`TriangulationPolicy::Fan`].
fn emit_fan(loops: &[LoopKey], out_tris: &mut Vec<[LoopKey; 3]>) {
    let i0 = loops[0];
    for w in 1..loops.len() - 1 {
        out_tris.push([i0, loops[w], loops[w + 1]]);
    }
}

/// Render-time quad path (§5): pick the 0-2 diagonal by default;
/// rotate to 1-3 when the 0-2 split would produce a flipped triangle
/// (§5.1). The flip check examines only the 0-2 diagonal and
/// deliberately avoids the face normal (§5.1 commentary on stability).
fn emit_quad_render_time(mesh: &LoopMesh, loops: &[LoopKey], out_tris: &mut Vec<[LoopKey; 3]>) {
    debug_assert_eq!(loops.len(), 4);
    let v1 = mesh.vertex_position(mesh.loop_vertex(loops[0]));
    let v2 = mesh.vertex_position(mesh.loop_vertex(loops[1]));
    let v3 = mesh.vertex_position(mesh.loop_vertex(loops[2]));
    let v4 = mesh.vertex_position(mesh.loop_vertex(loops[3]));

    let d12 = v2 - v1;
    let d13 = v3 - v1;
    let d14 = v4 - v1;
    let cross_a = d12.cross(d13);
    let cross_b = d14.cross(d13);
    let flipped = cross_a.dot(cross_b) > 0.0;

    if !flipped {
        // §5.2 default split (0-2 diagonal).
        // tri-a = (loop0, loop1, loop2), tri-b = (loop0, loop2, loop3)
        out_tris.push([loops[0], loops[1], loops[2]]);
        out_tris.push([loops[0], loops[2], loops[3]]);
    } else {
        // §5.2 rotated split (1-3 diagonal) via in-place patching.
        // tri-a[2] = loop3 (was loop2), tri-b[0] = loop1 (was loop0).
        // Resulting tri-a = (loop0, loop1, loop3), tri-b = (loop1, loop2, loop3).
        out_tris.push([loops[0], loops[1], loops[3]]);
        out_tris.push([loops[1], loops[2], loops[3]]);
    }
}

/// N-gon ear-clip path (§7): compute the face normal, project the
/// loops to 2D with the negated axis-dominant frame (§3.1), run the
/// 2D ear-clip kernel with `coords_sign = +1` (CW input), and map the
/// resulting `[u32; 3]` triples back to `[LoopKey; 3]`.
fn emit_ngon_ear_clip(
    mesh: &LoopMesh,
    face: FaceKey,
    loops: &[LoopKey],
    scratch_proj: &mut Vec<Vec2>,
    out_tris: &mut Vec<[LoopKey; 3]>,
) {
    let normal = mesh.face_compute_normal(face);
    let (u, w) = triangulate::axis_dominant_basis_negated(normal);

    scratch_proj.clear();
    scratch_proj.reserve(loops.len());
    for &l in loops {
        let p = mesh.vertex_position(mesh.loop_vertex(l));
        scratch_proj.push(Vec2::new(u.dot(p), w.dot(p)));
    }

    let mut idx_tris: Vec<[u32; 3]> = Vec::with_capacity(loops.len() - 2);
    triangulate::ear_clip(scratch_proj, /*coords_sign=*/ 1, &mut idx_tris);

    for t in idx_tris {
        out_tris.push([
            loops[t[0] as usize],
            loops[t[1] as usize],
            loops[t[2] as usize],
        ]);
    }
}
