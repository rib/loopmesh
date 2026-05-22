//! High-level operations
//!
//! These are the operations that user's are most likely to call directly, and
//! they are the ones that will eventually support dynamic dispatch via a
//! BMesh-style operator framework with slot dispatch and rich result bundles.
//!
//! They are implemented in terms of the mid-level operations in
//! [`mid`](super::mid), which in turn are implemented in terms of the low-level
//! Euler operators in [`euler`](super::euler).

mod extrude_region;
mod face_poke;
mod face_triangulate;
mod inset_individual;
mod inset_region;

pub use face_triangulate::{NgonMethod, QuadMethod};
pub use inset_individual::{InsetIndividualParams, InsetIndividualResult};
pub use inset_region::{InsetRegionParams, InsetRegionResult};
