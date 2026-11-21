//! Unit tests for the bevel operator, grouped by BEVEL.md topic.

use glam::Vec3;

use super::*;
use crate::loopmesh::test_support::mesh_add_cube;

mod adj_fill;
mod adj_general;
mod adj_square;
mod adj_tri_corner;
mod custom_profile;
mod cutoff;
mod donors;
mod edge_basic;
mod edge_customdata;
mod edge_decline;
mod edge_weld;
mod face_customdata;
mod folded_tee;
mod inner_miter;
mod junction_meet;
mod material;
mod offset_type;
mod outer_miter;
mod pipe;
mod profile_cell;
mod resample;
mod ring_order;
mod seam_sharp;
mod shading;
mod shared_edge;
mod slide_clamp;
mod strip_winding;
mod superellipse;
mod support;
mod terminal;
mod terminal_meet;
mod vertex;
mod vertex_clamp;
mod vertex_customdata;
mod weights;
mod winding;
mod wire;

use support::*;
