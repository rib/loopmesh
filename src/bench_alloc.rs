//! Allocation-counting wrapper around the system allocator, for use in
//! benchmarks that want to compare heap-allocation behaviour across
//! refactors.
//!
//! Enable by:
//! 1. Adding `--features bench-allocs` to `cargo bench`.
//! 2. Installing the allocator as the global allocator in your bench
//!    binary:
//!    ```ignore
//!    #[cfg(feature = "bench-allocs")]
//!    #[global_allocator]
//!    static A: loopmesh::bench_alloc::CountingAllocator =
//!        loopmesh::bench_alloc::CountingAllocator;
//!    ```
//! 3. Reading [`alloc_count`] / [`dealloc_count`] / [`bytes_allocated`]
//!    around the code of interest.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

static ALLOCS: AtomicUsize = AtomicUsize::new(0);
static DEALLOCS: AtomicUsize = AtomicUsize::new(0);
static BYTES: AtomicUsize = AtomicUsize::new(0);

/// Total number of `alloc` calls observed since the process started.
pub fn alloc_count() -> usize {
    ALLOCS.load(Ordering::Relaxed)
}

/// Total number of `dealloc` calls observed since the process started.
pub fn dealloc_count() -> usize {
    DEALLOCS.load(Ordering::Relaxed)
}

/// Cumulative bytes allocated (sum of every `Layout.size()` ever
/// allocated; not net-of-frees).
pub fn bytes_allocated() -> usize {
    BYTES.load(Ordering::Relaxed)
}

/// A `GlobalAlloc` that forwards to the system allocator and increments
/// the [`alloc_count`] / [`dealloc_count`] / [`bytes_allocated`] counters
/// on every call.
pub struct CountingAllocator;

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        BYTES.fetch_add(layout.size(), Ordering::Relaxed);
        System.alloc(layout)
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        DEALLOCS.fetch_add(1, Ordering::Relaxed);
        System.dealloc(ptr, layout)
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        BYTES.fetch_add(layout.size(), Ordering::Relaxed);
        System.alloc_zeroed(layout)
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // Count a realloc as one alloc and one dealloc.
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        DEALLOCS.fetch_add(1, Ordering::Relaxed);
        BYTES.fetch_add(new_size, Ordering::Relaxed);
        System.realloc(ptr, layout, new_size)
    }
}
