//! Computational-geometry primitives shared by mesh operators.
//!
//! These are non-Blender-derived building blocks (convex hull, etc.) tracked
//! under `bmesh-parity/prerequisites/`.

mod cdt;
mod hull;

pub use cdt::{
    triangulate, CdtInput, CdtProvider, CdtTriangle, CdtTriangulation, SpadeCdt,
};
pub use hull::{convex_hull, ConvexHullProvider, HullFace, HullTopology, Quickhull};

#[cfg(test)]
mod cdt_tests;
#[cfg(test)]
mod hull_tests;
