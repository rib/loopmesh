//! Low-level Euler operators — the minimal topology-mutating primitives the
//! mid-level [`mid`](super::mid) operations compose.
//!
//! Named after the Euler operators of the V−E+F=2 polyhedron formula (SEMV /
//! SFME / JEKV / JFKE / JVKE) plus a few companion primitives at the same tier
//! (vert/edge separate and the inverse edge splice, unglue, loop reverse).
//!
//! These are children of `loopmesh`, so they call the module-private element
//! accessors and helpers directly without widening their visibility.

mod edge_separate;
mod edge_splice;
mod join_edge_kill_vert;
mod join_face_kill_edge;
mod join_vert_kill_edge;
mod loop_reverse;
mod split_edge_make_vert;
mod split_face_make_edge;
mod unglue_region_make_vert;
mod vert_separate;
mod vert_splice;
