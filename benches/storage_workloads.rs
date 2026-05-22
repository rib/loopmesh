//! Criterion benchmarks measuring loopmesh's storage-layer + iterator
//! overhead. Establishes baselines for the upcoming refactor:
//!
//! - PR0 (this file): land the harness; record initial numbers.
//! - PR1: storage refactor (drop Rc/RefCell). Re-run; expect ~2–3× field
//!   access, ~2× normals_update.
//! - PR3: lending iterators land + per-element snapshot collectors gone.
//!   Re-run; alloc counts drop dramatically.
//!
//! Run timings:    `cargo bench -p loopmesh`
//! Run alloc count: `cargo bench -p loopmesh --features bench-allocs`
//!                  (the eprintln! lines at the end of `main` report counters).

use criterion::{black_box, criterion_group, BatchSize, Criterion};
use glam::Vec3;
use loopmesh::loopmesh::{EdgeCreateFlags, FaceCreateFlags, LoopMesh, VertexKey};

// Install the counting allocator if the feature is on. The bench targets
// don't read it inline (they'd perturb timings); after `criterion_main!`
// runs, the `main` shim below reports the cumulative deltas.
#[cfg(feature = "bench-allocs")]
#[global_allocator]
static ALLOC: loopmesh::bench_alloc::CountingAllocator = loopmesh::bench_alloc::CountingAllocator;

// ---- Fixture: planar quad grid ----
//
// Vertex layout for n=3 (XY plane, z=0):
//
//     6 - 7 - 8
//     |   |   |
//     3 - 4 - 5
//     |   |   |
//     0 - 1 - 2
//
// Returns the flat list of vertex keys in row-major order.
fn build_grid(mesh: &mut LoopMesh, n: usize) -> Vec<VertexKey> {
    let mut verts = Vec::with_capacity(n * n);
    for y in 0..n {
        for x in 0..n {
            verts.push(mesh.create_vertex(Vec3::new(x as f32, y as f32, 0.0)));
        }
    }
    for y in 0..(n - 1) {
        for x in 0..(n - 1) {
            let v0 = verts[y * n + x];
            let v1 = verts[y * n + x + 1];
            let v2 = verts[(y + 1) * n + x + 1];
            let v3 = verts[(y + 1) * n + x];
            let e0 = mesh.create_edge([v0, v1], EdgeCreateFlags::NO_DUPLICATE);
            let e1 = mesh.create_edge([v1, v2], EdgeCreateFlags::NO_DUPLICATE);
            let e2 = mesh.create_edge([v2, v3], EdgeCreateFlags::NO_DUPLICATE);
            let e3 = mesh.create_edge([v3, v0], EdgeCreateFlags::NO_DUPLICATE);
            mesh.face_create(&[v0, v1, v2, v3], &[e0, e1, e2, e3], FaceCreateFlags::NONE);
        }
    }
    verts
}

fn build_grid_mesh(n: usize) -> LoopMesh {
    let mut m = LoopMesh::new();
    build_grid(&mut m, n);
    m
}

// ---- bench_grid_normals_update ----
// Idempotent operation: build mesh once per batch, recompute normals N times.
fn bench_grid_normals_update(c: &mut Criterion) {
    let mut group = c.benchmark_group("grid_normals_update");
    for &n in &[16usize, 64, 128] {
        group.bench_function(format!("n={}", n), |b| {
            b.iter_batched_ref(
                || build_grid_mesh(n),
                |mesh| {
                    mesh.mesh_normals_update();
                    black_box(&*mesh);
                },
                BatchSize::LargeInput,
            );
        });
    }
    group.finish();
}

// ---- bench_disk_walk_sum ----
// Read-only: traverse every vertex's disk cycle, sum the edge count.
// A 200x200 grid has 40k verts, each ~4-valence interior.
fn bench_disk_walk_sum(c: &mut Criterion) {
    let mut group = c.benchmark_group("disk_walk_sum");
    for &n in &[64usize, 200] {
        let mesh = build_grid_mesh(n);
        group.bench_function(format!("n={}", n), |b| {
            b.iter(|| {
                let mut total: usize = 0;
                for v in mesh.vert_keys() {
                    total += mesh.vertex_disk_iter(v).count();
                }
                black_box(total)
            });
        });
    }
    group.finish();
}

// ---- bench_edge_split_cascade ----
// Destructive: subdivide every edge of a grid once (one cut per edge).
fn bench_edge_split_cascade(c: &mut Criterion) {
    let mut group = c.benchmark_group("edge_split_cascade");
    for &n in &[16usize, 32, 64] {
        group.bench_function(format!("n={}", n), |b| {
            b.iter_batched(
                || build_grid_mesh(n),
                |mut mesh| {
                    for e in mesh.edge_keys() {
                        let [v0, _v1] = mesh.edge_vertices(e);
                        let _ = mesh.edge_split(e, v0, 0.5);
                    }
                    black_box(mesh);
                },
                BatchSize::LargeInput,
            );
        });
    }
    group.finish();
}

// ---- bench_vert_dissolve_grid ----
// Destructive: dissolve every interior vertex of a grid (4-valence manifold).
// Note: order matters — dissolving one vertex changes neighbours' valences,
// so we snapshot the interior verts first and dissolve in row-major order.
// Some will fail mid-stream once their neighbours change; that's OK for the
// benchmark — we're measuring the work attempted.
fn bench_vert_dissolve_grid(c: &mut Criterion) {
    let mut group = c.benchmark_group("vert_dissolve_grid");
    for &n in &[8usize, 16, 32] {
        group.bench_function(format!("n={}", n), |b| {
            b.iter_batched(
                || {
                    let mesh = build_grid_mesh(n);
                    // Snapshot interior vert keys before mutation.
                    let mut interior = Vec::with_capacity((n - 2) * (n - 2));
                    let all = mesh.vert_keys();
                    // Row-major; skip the border (first/last row, first/last col).
                    for y in 1..(n - 1) {
                        for x in 1..(n - 1) {
                            interior.push(all[y * n + x]);
                        }
                    }
                    (mesh, interior)
                },
                |(mut mesh, interior)| {
                    for v in interior {
                        let _ = mesh.vert_dissolve(v);
                    }
                    black_box(mesh);
                },
                BatchSize::LargeInput,
            );
        });
    }
    group.finish();
}

// ---- bench_field_access_microbench ----
// Tight loop: read vertex_position over a key snapshot, then edge_vertices
// over an edge snapshot. Pure storage-layer cost.
fn bench_field_access_microbench(c: &mut Criterion) {
    let mut group = c.benchmark_group("field_access");
    let mesh = build_grid_mesh(128);
    let vkeys = mesh.vert_keys();
    let ekeys = mesh.edge_keys();
    group.bench_function("vertex_position_sweep", |b| {
        b.iter(|| {
            let mut acc = Vec3::ZERO;
            for &v in &vkeys {
                acc += mesh.vertex_position(v);
            }
            black_box(acc)
        });
    });
    group.bench_function("edge_vertices_sweep", |b| {
        b.iter(|| {
            let mut acc: u64 = 0;
            for &e in &ekeys {
                let [a, b_] = mesh.edge_vertices(e);
                // Mix the keys into the accumulator so the compiler can't elide.
                acc = acc.wrapping_add(loopmesh::loopmesh::ElementKey::as_usize(&a) as u64);
                acc = acc.wrapping_add(loopmesh::loopmesh::ElementKey::as_usize(&b_) as u64);
            }
            black_box(acc)
        });
    });
    group.finish();
}

/// Push a vert op-flag layer, mark every other vertex, count the marks.
/// Establishes the transient-mark baseline for the op-flag pool.
fn bench_op_flag_scan_grid(c: &mut Criterion) {
    let mut group = c.benchmark_group("op_flag_scan_grid");
    for &n in &[32usize, 128, 256] {
        group.bench_function(format!("grid_{n}x{n}"), |b| {
            let mut mesh = LoopMesh::new();
            let _verts = build_grid(&mut mesh, n);
            let all_verts: Vec<_> = mesh.vert_keys();
            b.iter(|| {
                let layer = mesh.push_vert_op_flag_layer();
                for (i, &v) in all_verts.iter().enumerate() {
                    if i & 1 == 0 {
                        mesh.vert_op_flag_set(layer, v, 0);
                    }
                }
                let count = mesh.iter_verts_with_op_flag(layer, 0).count();
                black_box(count);
                mesh.pop_vert_op_flag_layer(layer);
            });
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_grid_normals_update,
    bench_disk_walk_sum,
    bench_edge_split_cascade,
    bench_vert_dissolve_grid,
    bench_field_access_microbench,
    bench_op_flag_scan_grid,
);

// Custom main so we can report allocation counters after criterion's
// default `criterion_main` would have returned.
fn main() {
    let allocs_before = allocs_now();
    benches();
    Criterion::default().configure_from_args().final_summary();
    let allocs_after = allocs_now();
    if let Some((a_before, d_before, b_before)) = allocs_before {
        let (a_after, d_after, b_after) = allocs_after.unwrap();
        eprintln!("\n[bench-allocs] cumulative deltas during this run:");
        eprintln!("  allocs:    {}", a_after.saturating_sub(a_before));
        eprintln!("  deallocs:  {}", d_after.saturating_sub(d_before));
        eprintln!("  bytes:     {}", b_after.saturating_sub(b_before));
    }
}

#[cfg(feature = "bench-allocs")]
fn allocs_now() -> Option<(usize, usize, usize)> {
    Some((
        loopmesh::bench_alloc::alloc_count(),
        loopmesh::bench_alloc::dealloc_count(),
        loopmesh::bench_alloc::bytes_allocated(),
    ))
}

#[cfg(not(feature = "bench-allocs"))]
fn allocs_now() -> Option<(usize, usize, usize)> {
    None
}
