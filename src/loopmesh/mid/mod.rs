//! Mid-level operations — euler compositions and wrappers that sit one tier
//! above the [`euler`](super::euler) primitives but below the high-level
//! operator framework in [`ops`](super::ops).
//!
//! These are operations that are useful in their own right but not likely to be
//! the direct target of user calls; they are more likely to be building blocks
//! for the high-level operators.

mod edge_collapse;
mod edge_rotate;
mod edge_split;
mod face_split;
mod faces_join;
mod vert_dissolve;
