//! High-level operations
//!
//! These are the operations that user's are most likely to call directly, and
//! they are the ones that will eventually support dynamic dispatch via a
//! BMesh-style operator framework with slot dispatch and rich result bundles.
//!
//! They are implemented in terms of the mid-level operations in
//! [`mid`](super::mid), which in turn are implemented in terms of the low-level
//! Euler operators in [`euler`](super::euler).

mod beautify_fill;
mod bevel;
mod bisect_edges;
mod bisect_plane;
mod bridge_loops;
mod collapse;
mod connect_vert_pair;
mod connect_verts;
mod connect_verts_concave;
mod connect_verts_nonplanar;
mod convex_hull;
mod create_circle;
mod create_cone;
mod create_cube;
mod create_grid;
mod create_icosphere;
mod create_uvsphere;
mod create_vert;
mod delete;
mod dissolve_degenerate;
mod dissolve_edges;
mod dissolve_faces;
mod dissolve_limit;
mod dissolve_verts;
mod duplicate;
mod edgeloop_fill;
mod extrude_discrete;
mod extrude_edge_only;
mod extrude_mixed;
mod extrude_region;
mod extrude_vert_indiv;
mod face_attribute_fill;
mod face_poke;
mod face_triangulate;
mod find_doubles;
mod flip_quad_tessellation;
mod fractal_noise;
mod grid_fill;
mod holes_fill;
mod inset_individual;
mod inset_region;
mod join_triangles;
mod mean_value;
mod mirror;
mod offset_edgeloops;
mod planar_faces;
mod pointmerge;
mod recalc_face_normals;
mod region_extend;
mod remove_doubles;
mod reverse_colors;
mod reverse_faces;
mod reverse_uvs;
mod rotate;
mod rotate_colors;
mod rotate_edges;
mod rotate_uvs;
mod scale;
mod smooth_laplacian;
mod smooth_vert;
mod solidify;
mod spin;
mod split;
mod split_edges;
mod subdivide_edgering;
mod subdivide_edges;
mod symmetrize;
mod transform;
mod translate;
mod triangle_fill;
mod unsubdivide;
mod weld_verts;
mod wireframe;

pub use beautify_fill::{BeautifyFillOptions, BeautifyFillResult, BeautifyMethod};
pub use bevel::{BevelAffect, BevelMiter, BevelOffsetType, BevelOptions, BevelResult, BevelVmeshMethod};
pub use bisect_edges::BisectGeom;
pub use bisect_plane::{BisectParams, BisectResult};
pub use bridge_loops::{BridgeLoopsParams, BridgeLoopsResult};
pub use connect_verts_nonplanar::ConnectVertsNonplanarOut;
pub use convex_hull::ConvexHullResult;
pub use delete::{DeleteContext, DeleteSeed, MixedSeed};
pub use dissolve_edges::{DissolveEdgesParams, DissolveEdgesResult};
pub use dissolve_faces::{DissolveFacesOptions, DissolveFacesResult};
pub use dissolve_limit::{
    DissolveLimitParams, DissolveLimitResult, DELIMIT_MATERIAL, DELIMIT_NORMAL, DELIMIT_SEAM,
    DELIMIT_SHARP, DELIMIT_UV,
};
pub use dissolve_verts::DissolveVertsParams;
pub use duplicate::{DuplicateResult, GeomKey};
pub use edgeloop_fill::{EdgeloopFillParams, EdgeloopFillResult};
pub use face_attribute_fill::FaceAttributeFillResult;
pub use grid_fill::{GridFillParams, GridFillResult};
pub use holes_fill::{HolesFillParams, HolesFillResult};
pub use extrude_mixed::ExtrudeMixedResult;
pub use extrude_vert_indiv::ExtrudeVertIndivResult;
pub use face_poke::{CenterMode, PokeResult};
pub use face_triangulate::{NgonMethod, QuadMethod};
pub use inset_individual::{InsetIndividualParams, InsetIndividualResult};
pub use inset_region::{InsetRegionParams, InsetRegionResult};
pub use join_triangles::{JoinTrianglesParams, JoinTrianglesResult};
pub use mirror::{Axis, MirrorParams, MirrorResult};
pub use offset_edgeloops::OffsetEdgeLoopsResult;
pub use region_extend::RegionExtendOptions;
pub use rotate::RotateParams;
pub use rotate_edges::RotateEdgesResult;
pub use scale::ScaleParams;
pub use smooth_laplacian::SmoothLaplacianParams;
pub use smooth_vert::SmoothVertParams;
pub use split::SplitResult;
pub use subdivide_edgering::{
    ProfileShape, RingInterp, SubdivideEdgeRingParams, SubdivideEdgeRingResult,
};
pub use subdivide_edges::{
    QuadCornerType, SmoothFalloff, SubdivideGeom, SubdivideParams, SubdivideResult, SubdivideSlot,
};
pub use symmetrize::{SymmetrizeDirection, SymmetrizeResult};
pub use transform::TransformParams;
pub use triangle_fill::{TriangleFillParams, TriangleFillResult};
pub use translate::TranslateParams;
pub use wireframe::WireframeResult;
