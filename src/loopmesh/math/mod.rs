//! Numeric primitives shared by mesh operators.
//!
//! These are non-Blender-derived building blocks (sparse linear solves, etc.)
//! tracked under `bmesh-parity/prerequisites/`. Unlike the comp-geom `geom`
//! module, these are pure linear-algebra/numeric helpers with no mesh topology.

mod sparse_solve;

pub use sparse_solve::{SparseCholesky, SparseSolveError};

#[cfg(test)]
mod sparse_solve_tests;
