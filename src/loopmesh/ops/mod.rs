//! High-level operations
//!
//! These are the operations that user's are most likely to call directly, and
//! they are the ones that will eventually support dynamic dispatch via a
//! BMesh-style operator framework with slot dispatch and rich result bundles.
//!
//! They are implemented in terms of the mid-level operations in
//! [`mid`](super::mid), which in turn are implemented in terms of the low-level
//! Euler operators in [`euler`](super::euler).

mod collapse;
mod connect_verts;
mod delete;
mod dissolve_degenerate;
mod dissolve_edges;
mod dissolve_faces;
mod dissolve_limit;
mod dissolve_verts;
mod duplicate;
mod extrude_discrete;
mod extrude_edge_only;
mod extrude_mixed;
mod extrude_region;
mod extrude_vert_indiv;
mod find_doubles;
mod face_poke;
mod face_triangulate;
mod inset_individual;
mod inset_region;
mod mean_value;
mod mirror;
mod pointmerge;
mod remove_doubles;
mod split;
mod subdivide_edges;
mod weld_verts;

pub use delete::{DeleteContext, DeleteSeed, MixedSeed};
pub use dissolve_edges::{DissolveEdgesParams, DissolveEdgesResult};
pub use dissolve_faces::{DissolveFacesOptions, DissolveFacesResult};
pub use dissolve_limit::{
    DissolveLimitParams, DissolveLimitResult, DELIMIT_MATERIAL, DELIMIT_NORMAL, DELIMIT_SEAM,
    DELIMIT_SHARP, DELIMIT_UV,
};
pub use dissolve_verts::DissolveVertsParams;
pub use duplicate::{DuplicateResult, GeomKey};
pub use extrude_mixed::ExtrudeMixedResult;
pub use extrude_vert_indiv::ExtrudeVertIndivResult;
pub use face_poke::{CenterMode, PokeResult};
pub use face_triangulate::{NgonMethod, QuadMethod};
pub use inset_individual::{InsetIndividualParams, InsetIndividualResult};
pub use inset_region::{InsetRegionParams, InsetRegionResult};
pub use mirror::{Axis, MirrorParams, MirrorResult};
pub use split::SplitResult;
pub use subdivide_edges::{
    QuadCornerType, SubdivideGeom, SubdivideParams, SubdivideResult,
};
