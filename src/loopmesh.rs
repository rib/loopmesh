#![allow(dead_code)]
#![allow(unused_variables)]

use std::marker::PhantomData;

use anyhow::{anyhow, Result};
use bitflags::bitflags;
use glam::*;
use slab::Slab;
use smallvec::SmallVec;

pub use crate::attr::{AttrRole, AttrType, AttributeKey};

bitflags! {
    struct LoopMeshFlags: u32 {
        const NONE      = 0;

        /// Element is selected in some UI / editor
        const VERTICES_INDEX_DIRTY  = 1<<0;
        const EDGES_INDEX_DIRTY  = 1<<1;
        const LOOPS_INDEX_DIRTY  = 1<<2;
        const FACES_INDEX_DIRTY  = 1<<3;

        const ALL_INDICES_DIRTY = (Self::VERTICES_INDEX_DIRTY.bits | Self::EDGES_INDEX_DIRTY.bits | Self::LOOPS_INDEX_DIRTY.bits | Self::FACES_INDEX_DIRTY.bits);

    }
}

bitflags! {
    pub struct ElementFlags: u32 {
        const NONE      = 0;

        /// Element is selected in some UI / editor
        const SELECTED  = 1<<0;

        /// Element has been marked as "hidden" - e.g. to simplify view while editing
        const HIDDEN    = 1<<1;

        const TAG0      = 1<<2;
        const TAG1      = 1<<3;
        const TAG2      = 1<<4;
    }
}

bitflags! {
    pub struct EdgeCreateFlags: u32 {
        const NONE          = 0;
        const NO_DUPLICATE  = 1<<0;
    }
}

bitflags! {
    pub struct FaceCreateFlags: u32 {
        const NONE          = 0;
        const NO_DUPLICATE  = 1<<0;
    }
}

pub use crate::keys::{EdgeKey, ElementKey, ElementKeyFromUsize, FaceKey, LoopKey, VertexKey};

// Operators live in child modules so they can call the module-private Euler
// primitives and element accessors without widening their visibility. Each
// one contributes an `impl LoopMesh` block.
mod attributes;
mod euler;
mod index;
mod iter;
mod mid;
mod normals;
mod ops;
mod queries;
mod radial;

pub use attributes::*;
pub use iter::*;
pub use mid::EdgeCollapseFlags;
pub use ops::{
    Axis, BevelAffect, BevelOffsetType, BevelOptions, BevelResult, BisectGeom, BisectParams,
    BisectResult, BridgeLoopsParams, BridgeLoopsResult, CenterMode, ConnectVertsNonplanarOut,
    DeleteContext, DeleteSeed, DissolveEdgesParams, DissolveEdgesResult, DissolveFacesOptions,
    DissolveFacesResult, DissolveLimitParams, DissolveLimitResult, DissolveVertsParams,
    DuplicateResult, ExtrudeVertIndivResult, GeomKey, InsetIndividualParams, InsetIndividualResult,
    InsetRegionParams, InsetRegionResult, JoinTrianglesParams, JoinTrianglesResult, MirrorParams,
    MirrorResult, MixedSeed, NgonMethod, ProfileShape, QuadCornerType, QuadMethod, RingInterp,
    RotateEdgesResult, RotateParams, ScaleParams, SmoothFalloff, SmoothVertParams,
    SubdivideEdgeRingParams, SubdivideEdgeRingResult, SubdivideGeom, SubdivideParams,
    SubdivideResult, SymmetrizeDirection, SymmetrizeResult, TransformParams, TranslateParams,
    DELIMIT_MATERIAL, DELIMIT_NORMAL, DELIMIT_SEAM, DELIMIT_SHARP, DELIMIT_UV,
};

// Shared fixtures for the per-operator unit tests that live in the `euler`,
// `mid`, and `ops` child modules.
#[cfg(test)]
pub(crate) mod test_support;

pub(crate) struct Vertex {
    pub(crate) eflags: ElementFlags,
    pub(crate) key: VertexKey,
    pub(crate) index: u32,

    pub(crate) position: Vec3,
    pub(crate) normal: Vec3,

    pub(crate) disk_loop_start: Option<EdgeKey>,
}

impl Vertex {}

#[derive(Clone, Copy)]
pub(crate) struct DoubleLinks<IndexType> {
    pub next: IndexType,
    pub prev: IndexType,
}

impl<IndexType> DoubleLinks<IndexType> {
    /// A utility for inserting a new node into a doubly linked
    ///
    /// It's tricky to generalize the linked lists here while they each have rather different
    /// data associations and so it's most practical to assume that the links for the
    /// end/start of the list are first read-by value and so then we want to determine
    /// the set of links for the end, new and start nodes in the list that can be written
    /// back.
    ///
    /// One notable gotcha with updating links like this is the start and end may correspond
    /// to the same node for a list with just one node, so this utility is
    /// careful to check for that case...
    ///
    /// Returns links as a tupled ordered like (prev, new, next)
    fn insert_link(
        prev_links: DoubleLinks<IndexType>,
        previ: IndexType,
        newi: IndexType,
        next_links: DoubleLinks<IndexType>,
        nexti: IndexType,
    ) -> (
        DoubleLinks<IndexType>,
        DoubleLinks<IndexType>,
        DoubleLinks<IndexType>,
    )
    where
        IndexType: Eq + Copy,
    {
        if nexti == previ {
            (
                DoubleLinks {
                    // end
                    next: newi,
                    prev: newi,
                },
                DoubleLinks {
                    // new
                    next: nexti,
                    prev: previ,
                },
                DoubleLinks {
                    // start
                    next: newi,
                    prev: newi,
                },
            )
        } else {
            (
                DoubleLinks {
                    // end
                    next: newi,
                    prev: prev_links.prev, // unchanged
                },
                DoubleLinks {
                    // new
                    next: nexti,
                    prev: previ,
                },
                DoubleLinks {
                    //start
                    next: next_links.next, // unchanged
                    prev: newi,
                },
            )
        }
    }

    fn remove_link(
        prev_links: DoubleLinks<IndexType>,
        previ: IndexType,
        oldi: IndexType,
        next_links: DoubleLinks<IndexType>,
        nexti: IndexType,
    ) -> (
        DoubleLinks<IndexType>,
        DoubleLinks<IndexType>,
        DoubleLinks<IndexType>,
    )
    where
        IndexType: Eq + Copy,
    {
        if previ == nexti {
            (
                DoubleLinks {
                    // prev
                    next: previ,
                    prev: previ,
                },
                DoubleLinks {
                    // old
                    next: oldi,
                    prev: oldi,
                },
                DoubleLinks {
                    // next
                    next: previ,
                    prev: previ,
                },
            )
        } else {
            (
                DoubleLinks {
                    // prev
                    next: nexti,
                    prev: prev_links.prev, // unchanged
                },
                DoubleLinks {
                    // old
                    next: oldi,
                    prev: oldi,
                },
                DoubleLinks {
                    //next
                    next: next_links.next, // unchanged
                    prev: previ,
                },
            )
        }
    }
}

pub(crate) struct Edge {
    pub(crate) eflags: ElementFlags,

    pub(crate) key: EdgeKey,
    pub(crate) index: u32,

    /// Edge vertices (arbitrary order)
    pub(crate) vertices: [VertexKey; 2],
    pub(crate) vertex_disk_links: [DoubleLinks<EdgeKey>; 2],

    pub(crate) radial_link: Option<LoopKey>,
}

impl Edge {
    /// Checks whether the given vertex is part of this edge
    pub fn contains(&self, vert: VertexKey) -> Option<usize> {
        if self.vertices[0] == vert {
            Some(0)
        } else if self.vertices[1] == vert {
            Some(1)
        } else {
            None
        }
    }

    /// Checks that the given vertex is part of this edge and returns
    /// the edge's other vertex
    pub fn other_vertex(&self, vert: VertexKey) -> Option<VertexKey> {
        self.contains(vert)
            .map(|i| self.vertices[(i == 0) as usize])
    }

    /// like: bmesh_disk_edge_link_from_vert
    pub fn disk_loop_links_for_vert(&self, vert: VertexKey) -> DoubleLinks<EdgeKey> {
        match self.contains(vert) {
            Some(i) => self.vertex_disk_links[i],
            None => {
                panic!("Expected vertex {vert} to belong to edge")
            }
        }
    }
    pub fn disk_loop_links_for_vert_mut(&mut self, vert: VertexKey) -> &mut DoubleLinks<EdgeKey> {
        match self.contains(vert) {
            Some(i) => &mut self.vertex_disk_links[i],
            None => {
                panic!("Expected vertex {vert} to belong to edge")
            }
        }
    }

    pub fn set_disk_loop_links_for_vert(&mut self, vert: VertexKey, links: DoubleLinks<EdgeKey>) {
        match self.contains(vert) {
            Some(i) => self.vertex_disk_links[i] = links,
            None => {
                panic!("Expected vertex to already belong to edge!");
            }
        }
    }
}

pub(crate) struct Face {
    pub(crate) eflags: ElementFlags,

    pub(crate) key: FaceKey,
    pub(crate) index: u32,

    /// The number of vertices and the number of edge loops that
    /// define the outline of the face
    pub(crate) len: u32,
    pub(crate) first_edge_loop: LoopKey,

    pub(crate) normal: Vec3,
}

impl Element for Face {
    fn index(&self) -> usize {
        self.index as usize
    }
    fn set_index(&mut self, index: usize) {
        self.index = index as u32;
    }
    fn flags(&self) -> ElementFlags {
        self.eflags
    }
    fn set_flags(&mut self, flags: ElementFlags) {
        self.eflags = flags;
    }
}

// =====================================================================
// Lending iterators
//
// These mirror the borrowing iterators above but their state is pure POD —
// they hold no reference to the mesh. Advance them by calling `next(&mesh)`
// explicitly; that way `&mut mesh` mutation is allowed between steps.
// =====================================================================

// ----- Mesh-wide lending iterators -----
//
// These walk the slab via a cursor that the iterator state owns. No mesh
// reference is retained, so `&mut self` mutations on the mesh between
// `next()` calls are fine.
//
// Cost per `next()`: increment cursor + `slab.get(i).is_some()` test until
// an occupied slot is found. Dense slabs (no removals) hit O(1) amortized.
// Sparse slabs pay O(holes-skipped). The high-water-mark from
// `key_watermark()` bounds the walk so freed slots beyond the highest
// live key are never even probed.
//
// Zero allocation throughout: callers can replace `for v in mesh.vert_keys()`
// (allocates `Vec<VertexKey>`) with a `MeshVerts` walk that allocates nothing.

pub(crate) struct Loop {
    pub(crate) key: LoopKey,
    pub(crate) index: u32,

    pub(crate) vertex: VertexKey,

    /// The edge this loop lies on. The invariant: for every loop reachable
    /// via `face_loops(face)` on a fully-constructed mesh, `edge !=
    /// EdgeKey::NULL` and `mesh.edges[edge]` contains both `self.vertex` and
    /// `loops[self.face_loop_links.next].vertex`. The value transiently
    /// equals `EdgeKey::NULL` inside [`LoopMesh::euler_split_edge_make_vert`]
    /// while the radial cycle is being rewoven; that is the only place a
    /// reachable loop is allowed to carry a null edge.
    pub(crate) edge: EdgeKey,
    pub(crate) face: FaceKey,

    pub(crate) edge_radial_links: DoubleLinks<LoopKey>,
    pub(crate) face_loop_links: DoubleLinks<LoopKey>,
}

pub(crate) trait Element {
    fn index(&self) -> usize;
    fn set_index(&mut self, index: usize);

    fn flags(&self) -> ElementFlags;
    fn set_flags(&mut self, flags: ElementFlags);

    fn is_selected(&self) -> bool {
        (self.flags() & ElementFlags::SELECTED) != ElementFlags::NONE
    }
    fn set_selected(&mut self) {
        self.set_flags(self.flags() | ElementFlags::SELECTED);
    }
    fn is_hidden(&self) -> bool {
        (self.flags() & ElementFlags::HIDDEN) != ElementFlags::NONE
    }
    fn set_hidden(&mut self) {
        self.set_flags(self.flags() | ElementFlags::HIDDEN);
    }
}

/// Index-typed wrapper around a `slab::Slab<T>`. The newtype-keyed access
/// (`get(VertexKey)` etc.) is the only thing this adds beyond the bare slab.
pub(crate) struct ElementSlab<I, T> {
    slab: Slab<T>,
    watermark: usize,
    _phantom: PhantomData<I>,
}

impl<I, ElementType> ElementSlab<I, ElementType>
where
    I: ElementKey + ElementKeyFromUsize + Copy,
{
    pub fn new() -> Self {
        Self {
            slab: Slab::new(),
            watermark: 0,
            _phantom: PhantomData,
        }
    }

    /// Iterate the keys of all live entries.
    pub fn iter(&self) -> impl Iterator<Item = I> + '_ {
        self.slab.iter().map(|(i, _)| I::from_usize(i))
    }

    /// Borrow an element by key. Panics if the key has been freed.
    pub(crate) fn get(&self, idx: I) -> &ElementType {
        &self.slab[idx.as_usize()]
    }

    /// Mutably borrow an element by key. Panics if the key has been freed.
    pub(crate) fn get_mut(&mut self, idx: I) -> &mut ElementType {
        &mut self.slab[idx.as_usize()]
    }

    /// Fallible read — returns `None` if `idx` is past the high-water mark
    /// or sits in a freed slot. Used by the mesh-wide cursor iterators to
    /// skip vacancies without panicking.
    pub(crate) fn try_get(&self, idx: I) -> Option<&ElementType> {
        self.slab.get(idx.as_usize())
    }

    /// Remove the entry at `idx` from the slab.
    pub(crate) fn remove(&mut self, idx: I) {
        self.slab.remove(idx.as_usize());
    }

    /// Clear every entry.
    pub(crate) fn clear(&mut self) {
        self.slab.clear();
        self.watermark = 0;
    }

    /// Number of live entries.
    #[allow(dead_code)]
    pub(crate) fn len(&self) -> usize {
        self.slab.len()
    }

    /// The slab's high-water-mark key (used by attribute storage to size
    /// parallel Vecs): one past the highest key ever issued. Monotonic
    /// except for `clear()` — `remove()` leaves vacant slots in place, so
    /// it never lowers the mark.
    pub(crate) fn key_watermark(&self) -> usize {
        self.watermark
    }

    /// Reserve a vacant slot for a new entry. The caller fills it via the
    /// returned VacantEntry. Counted against the watermark up front; an
    /// unfilled entry would over-count, but every caller fills it.
    pub(crate) fn vacant_entry(&mut self) -> slab::VacantEntry<'_, ElementType> {
        let entry = self.slab.vacant_entry();
        self.watermark = self.watermark.max(entry.key() + 1);
        entry
    }
}

/// Output of [`LoopMesh::extrude_region`]. The maps name every duplicated
/// element so callers can displace the new vertices, recompute normals on
/// the wall faces, etc.
#[derive(Debug)]
pub struct ExtrudeResult {
    /// `(source_vertex, duplicate)` for every vertex referenced by any face
    /// in the selection. The duplicate carries a copy of the source's
    /// customdata.
    pub vert_map: Vec<(VertexKey, VertexKey)>,
    /// `(source_edge, duplicate)` for every edge referenced by any face in
    /// the selection. The duplicate spans the corresponding duplicate
    /// vertices and carries a copy of the source's customdata.
    pub edge_map: Vec<(EdgeKey, EdgeKey)>,
    /// Newly-created "vertical" edges connecting each source vertex to its
    /// duplicate. One per source vertex (lazy creation; shared across all
    /// walls at that vertex). These take default attribute values.
    pub side_edges: Vec<EdgeKey>,
    /// Newly-created wall quad faces, one per selected loop on the
    /// pre-mutation radial of each boundary edge. On a manifold mesh this
    /// is one wall per boundary edge; on a non-manifold mesh it can be
    /// more.
    pub wall_faces: Vec<FaceKey>,
    /// Freshly-built duplicate cap faces, one per input face. Populated only
    /// in the `use_keep_orig` path (EXTRUDE.md §3.2 shell semantics), where
    /// the duplicate cap is genuinely new geometry alongside the kept
    /// original. Empty in the default path, where the original face is
    /// re-anchored to *become* the cap and so is not new geometry.
    pub cap_faces: Vec<FaceKey>,
}

/// Cross-pass record of which source vertices have already been duplicated by
/// an extrude pass, and which vertical/connecting edge was lifted at each.
///
/// A mixed-input extrude routes faces, directly-named edges, and isolated
/// verts to three separate passes (EXTRUDE.md §1–§2). When a vertex is shared
/// across kinds — e.g. a face corner that is also an endpoint of a
/// directly-named wire edge — §8.4 requires it be duplicated exactly once and
/// the two passes' walls to meet at that single duplicate, sharing one
/// vertical edge. Passing this ledger between the passes lets a later pass
/// reuse an earlier pass's duplicate (and its vertical edge) instead of
/// minting a parallel one. Slab-indexed by the source vertex's slot, sized to
/// the watermark captured before any pass runs (duplicates created during a
/// pass always land at higher slots, so they never need a ledger entry).
#[derive(Debug)]
pub(crate) struct ExtrudeVertDupLedger {
    dup: Vec<Option<VertexKey>>,
    side_edge: Vec<EdgeKey>,
}

impl ExtrudeVertDupLedger {
    pub(crate) fn new(vert_watermark: usize) -> Self {
        Self {
            dup: vec![None; vert_watermark],
            side_edge: vec![EdgeKey::NULL; vert_watermark],
        }
    }

    fn in_range(&self, v: VertexKey) -> bool {
        v.as_usize() < self.dup.len()
    }

    pub(crate) fn dup_of(&self, v: VertexKey) -> Option<VertexKey> {
        if !self.in_range(v) {
            return None;
        }
        self.dup[v.as_usize()]
    }

    pub(crate) fn record_dup(&mut self, source: VertexKey, dup: VertexKey) {
        if self.in_range(source) {
            self.dup[source.as_usize()] = Some(dup);
        }
    }

    pub(crate) fn side_edge_of(&self, v: VertexKey) -> Option<EdgeKey> {
        if !self.in_range(v) {
            return None;
        }
        let e = self.side_edge[v.as_usize()];
        if e.is_null() {
            None
        } else {
            Some(e)
        }
    }

    pub(crate) fn record_side_edge(&mut self, source: VertexKey, edge: EdgeKey) {
        if self.in_range(source) {
            self.side_edge[source.as_usize()] = edge;
        }
    }
}

pub struct LoopMesh {
    pub(crate) flags: LoopMeshFlags,

    pub(crate) vertices: ElementSlab<VertexKey, Vertex>,
    pub(crate) edges: ElementSlab<EdgeKey, Edge>,
    pub(crate) faces: ElementSlab<FaceKey, Face>,
    pub(crate) loops: ElementSlab<LoopKey, Loop>,

    pub(crate) vertices_index: Vec<VertexKey>,
    pub(crate) edges_index: Vec<EdgeKey>,
    pub(crate) faces_index: Vec<FaceKey>,
    pub(crate) loops_index: Vec<LoopKey>,

    // Reusable per-element-type scratch buffers. Algorithms that need
    // "snapshot all keys of type X for mutation" can `with_*_keys()` to
    // borrow one of these instead of allocating a fresh Vec per call.
    pub(crate) vert_key_scratch: Vec<VertexKey>,
    pub(crate) edge_key_scratch: Vec<EdgeKey>,
    pub(crate) face_key_scratch: Vec<FaceKey>,
    pub(crate) loop_key_scratch: Vec<LoopKey>,

    pub(crate) vertex_attribs: ElementAttributes,
    pub(crate) edge_attribs: ElementAttributes,
    pub(crate) loop_attribs: ElementAttributes,
    pub(crate) face_attribs: ElementAttributes,

    // Transient per-element bit fields for operators. See [`op_flags`].
    pub(crate) op_flags: crate::op_flags::OpFlagPool,
}

impl Default for LoopMesh {
    fn default() -> Self {
        Self::new()
    }
}

impl LoopMesh {
    pub fn new() -> LoopMesh {
        LoopMesh {
            flags: LoopMeshFlags::ALL_INDICES_DIRTY,

            vertices: ElementSlab::new(),
            edges: ElementSlab::new(),
            faces: ElementSlab::new(),
            loops: ElementSlab::new(),

            vertices_index: vec![],
            edges_index: vec![],
            faces_index: vec![],
            loops_index: vec![],

            vert_key_scratch: vec![],
            edge_key_scratch: vec![],
            face_key_scratch: vec![],
            loop_key_scratch: vec![],

            vertex_attribs: ElementAttributes::default(),
            edge_attribs: ElementAttributes::default(),
            loop_attribs: ElementAttributes::default(),
            face_attribs: ElementAttributes::default(),

            op_flags: crate::op_flags::OpFlagPool::default(),
        }
    }

    // ----- Element-index tables -----
    //
    // Each `*_ensure_index` populates a dense `Vec<Key>` AND sets the cached
    // `index` field on every live element, so subsequent algorithms can use
    // `mesh.X.get(k).index as usize` as a dense array index in O(1).
    //
    // Tracked via the matching `LoopMeshFlags::*_INDEX_DIRTY` bit:
    // - cleared after a successful `ensure_index`
    // - set by every mutating op (create/kill paths) so the next ensure
    //   re-walks the mesh.
    //

    // Index maintenance is split into two operations:
    //
    //   *_ensure_index_field  — assigns `element.index = i` on every live
    //                           element. Cheap; no Vec allocation. Used when
    //                           callers only need O(1) per-element dense-index
    //                           lookups.
    //
    //   *_ensure_table        — populates `self.*_index: Vec<Key>` so callers
    //                           can iterate by dense index. Shares the walk
    //                           with the index-field pass.
    //
    // One dirty bit drives both: when set, both the field and the table are
    // rebuilt on the next ensure. A caller needing only the field still pays
    // one walk but skips the Vec writes.

    // ----- Dirty-flag observability -----
    //
    // Read-only queries that don't force a rebuild. Useful for asserts and
    // for algorithms that want to fail loudly rather than silently re-walk.

    // ----- Operator-scratch flag pool API -----
    //
    // Push a layer to allocate (or reuse) a per-element bit field, mark
    // elements as the operator runs, pop the layer when done. Bits are
    // numbered `0..32`; the operator names them.
    //
    // For the cheap case (≤3 transient bits, no risk of conflicting with
    // other tag users), [`ElementFlags::TAG0`]..[`ElementFlags::TAG2`] is
    // strictly faster — no allocation, one-instruction access. The op-flag
    // pool is for richer cases.

    pub fn push_vert_op_flag_layer(&mut self) -> crate::op_flags::VertOpFlagLayer {
        let w = self.vertices.key_watermark();
        self.op_flags.push_vert(w)
    }

    pub fn pop_vert_op_flag_layer(&mut self, layer: crate::op_flags::VertOpFlagLayer) {
        self.op_flags.pop_vert(layer);
    }

    #[inline]
    pub fn vert_op_flag_set(
        &mut self,
        layer: crate::op_flags::VertOpFlagLayer,
        v: VertexKey,
        bit: u8,
    ) {
        self.op_flags.vert_set(layer, v.as_usize(), bit);
    }

    #[inline]
    pub fn vert_op_flag_clear(
        &mut self,
        layer: crate::op_flags::VertOpFlagLayer,
        v: VertexKey,
        bit: u8,
    ) {
        self.op_flags.vert_clear(layer, v.as_usize(), bit);
    }

    #[inline]
    pub fn vert_op_flag_test(
        &self,
        layer: crate::op_flags::VertOpFlagLayer,
        v: VertexKey,
        bit: u8,
    ) -> bool {
        self.op_flags.vert_test(layer, v.as_usize(), bit)
    }

    pub fn vert_op_flag_clear_all(&mut self, layer: crate::op_flags::VertOpFlagLayer) {
        self.op_flags.vert_clear_all(layer);
    }

    pub fn push_edge_op_flag_layer(&mut self) -> crate::op_flags::EdgeOpFlagLayer {
        let w = self.edges.key_watermark();
        self.op_flags.push_edge(w)
    }

    pub fn pop_edge_op_flag_layer(&mut self, layer: crate::op_flags::EdgeOpFlagLayer) {
        self.op_flags.pop_edge(layer);
    }

    #[inline]
    pub fn edge_op_flag_set(
        &mut self,
        layer: crate::op_flags::EdgeOpFlagLayer,
        e: EdgeKey,
        bit: u8,
    ) {
        self.op_flags.edge_set(layer, e.as_usize(), bit);
    }

    #[inline]
    pub fn edge_op_flag_clear(
        &mut self,
        layer: crate::op_flags::EdgeOpFlagLayer,
        e: EdgeKey,
        bit: u8,
    ) {
        self.op_flags.edge_clear(layer, e.as_usize(), bit);
    }

    #[inline]
    pub fn edge_op_flag_test(
        &self,
        layer: crate::op_flags::EdgeOpFlagLayer,
        e: EdgeKey,
        bit: u8,
    ) -> bool {
        self.op_flags.edge_test(layer, e.as_usize(), bit)
    }

    pub fn edge_op_flag_clear_all(&mut self, layer: crate::op_flags::EdgeOpFlagLayer) {
        self.op_flags.edge_clear_all(layer);
    }

    pub fn push_face_op_flag_layer(&mut self) -> crate::op_flags::FaceOpFlagLayer {
        let w = self.faces.key_watermark();
        self.op_flags.push_face(w)
    }

    pub fn pop_face_op_flag_layer(&mut self, layer: crate::op_flags::FaceOpFlagLayer) {
        self.op_flags.pop_face(layer);
    }

    #[inline]
    pub fn face_op_flag_set(
        &mut self,
        layer: crate::op_flags::FaceOpFlagLayer,
        f: FaceKey,
        bit: u8,
    ) {
        self.op_flags.face_set(layer, f.as_usize(), bit);
    }

    #[inline]
    pub fn face_op_flag_clear(
        &mut self,
        layer: crate::op_flags::FaceOpFlagLayer,
        f: FaceKey,
        bit: u8,
    ) {
        self.op_flags.face_clear(layer, f.as_usize(), bit);
    }

    #[inline]
    pub fn face_op_flag_test(
        &self,
        layer: crate::op_flags::FaceOpFlagLayer,
        f: FaceKey,
        bit: u8,
    ) -> bool {
        self.op_flags.face_test(layer, f.as_usize(), bit)
    }

    pub fn face_op_flag_clear_all(&mut self, layer: crate::op_flags::FaceOpFlagLayer) {
        self.op_flags.face_clear_all(layer);
    }

    pub fn push_loop_op_flag_layer(&mut self) -> crate::op_flags::LoopOpFlagLayer {
        let w = self.loops.key_watermark();
        self.op_flags.push_loop(w)
    }

    pub fn pop_loop_op_flag_layer(&mut self, layer: crate::op_flags::LoopOpFlagLayer) {
        self.op_flags.pop_loop(layer);
    }

    #[inline]
    pub fn loop_op_flag_set(
        &mut self,
        layer: crate::op_flags::LoopOpFlagLayer,
        l: LoopKey,
        bit: u8,
    ) {
        self.op_flags.loop_set(layer, l.as_usize(), bit);
    }

    #[inline]
    pub fn loop_op_flag_clear(
        &mut self,
        layer: crate::op_flags::LoopOpFlagLayer,
        l: LoopKey,
        bit: u8,
    ) {
        self.op_flags.loop_clear(layer, l.as_usize(), bit);
    }

    #[inline]
    pub fn loop_op_flag_test(
        &self,
        layer: crate::op_flags::LoopOpFlagLayer,
        l: LoopKey,
        bit: u8,
    ) -> bool {
        self.op_flags.loop_test(layer, l.as_usize(), bit)
    }

    pub fn loop_op_flag_clear_all(&mut self, layer: crate::op_flags::LoopOpFlagLayer) {
        self.op_flags.loop_clear_all(layer);
    }

    /// Iterate every live vertex whose `bit` is set in `layer`.
    pub fn iter_verts_with_op_flag(
        &self,
        layer: crate::op_flags::VertOpFlagLayer,
        bit: u8,
    ) -> impl Iterator<Item = VertexKey> + '_ {
        let mut it = MeshVerts::default();
        std::iter::from_fn(move || loop {
            let v = it.next(self)?;
            if self.op_flags.vert_test(layer, v.as_usize(), bit) {
                return Some(v);
            }
        })
    }

    pub fn iter_edges_with_op_flag(
        &self,
        layer: crate::op_flags::EdgeOpFlagLayer,
        bit: u8,
    ) -> impl Iterator<Item = EdgeKey> + '_ {
        let mut it = MeshEdges::default();
        std::iter::from_fn(move || loop {
            let e = it.next(self)?;
            if self.op_flags.edge_test(layer, e.as_usize(), bit) {
                return Some(e);
            }
        })
    }

    pub fn iter_faces_with_op_flag(
        &self,
        layer: crate::op_flags::FaceOpFlagLayer,
        bit: u8,
    ) -> impl Iterator<Item = FaceKey> + '_ {
        let mut it = MeshFaces::default();
        std::iter::from_fn(move || loop {
            let f = it.next(self)?;
            if self.op_flags.face_test(layer, f.as_usize(), bit) {
                return Some(f);
            }
        })
    }

    pub fn iter_loops_with_op_flag(
        &self,
        layer: crate::op_flags::LoopOpFlagLayer,
        bit: u8,
    ) -> impl Iterator<Item = LoopKey> + '_ {
        let mut it = MeshLoops::default();
        std::iter::from_fn(move || loop {
            let l = it.next(self)?;
            if self.op_flags.loop_test(layer, l.as_usize(), bit) {
                return Some(l);
            }
        })
    }

    /// Create a LoopMesh from an array of x,y,z position vectors.
    ///
    /// This will create the same number of input vertices with the same indexing order which will
    /// also be preserved if converting back to a vector (assuming none of them get deleted when editing)
    pub fn new_from_pos_f32(
        pos: &[f32],
        indices: Option<&[u32]>,
        topology: PrimitiveTopology,
    ) -> Result<LoopMesh> {
        let pos = bytemuck::try_cast_slice::<f32, [f32; 3]>(pos)
            .map_err(|e| anyhow!("Failed to cast positions into &[[f32;3]] slice"))?;

        let mut mesh = Self::new();

        for i in 0..pos.len() {
            let v = mesh.create_vertex(pos[i].into());

            // To avoid explicitly creating an index afterwards we instead just expect/assume that
            // the slab is going to use sequential indices for new vertices in a new (empty) mesh
            debug_assert!(v.0 == i as u32);
        }

        let num_vertices = if let Some(indices) = indices {
            indices.len()
        } else {
            pos.len()
        };
        for triangle in indices_iter(indices, 0, num_vertices).primitive_triangles(topology) {
            // A small layering violation, creating vertex indices manually here for the sake of avoiding creating
            // an explicit index at this point when we can also rely on sequential slab indices too.
            let v0 = VertexKey(triangle[0] as u32);
            let v1 = VertexKey(triangle[1] as u32);
            let v2 = VertexKey(triangle[2] as u32);

            let e0 = mesh.create_edge([v0, v1], EdgeCreateFlags::NO_DUPLICATE);
            let e1 = mesh.create_edge([v0, v1], EdgeCreateFlags::NO_DUPLICATE);
            let e2 = mesh.create_edge([v0, v1], EdgeCreateFlags::NO_DUPLICATE);

            let face = mesh.face_create(&[v0, v1, v2], &[e0, e1, e2], FaceCreateFlags::NONE);
        }

        Ok(mesh)
    }

    pub fn into_indexed_vec3_pos_triangles(&mut self) -> Result<(Vec<f32>, Vec<u32>)> {
        let mut pos = vec![];
        let mut indices = vec![];

        // By indexing the vertices then we can also refer to vertex.index while
        // iterating the faces
        self.vertices_ensure_index();

        for v in self.vertices_index.iter() {
            let v = &self.vertices.get(*v);
            pos.push(v.position.x);
            pos.push(v.position.y);
            pos.push(v.position.z);
        }

        for face in self.faces.iter() {
            let face_data = self.faces.get(face);
            assert_eq!(face_data.len, 3); // only triangle faces are supported here

            for (i, l) in self
                .face_loop_iter(self.faces.get(face).first_edge_loop)
                .enumerate()
            {
                let v = self.vertices.get(self.loops.get(l).vertex);
                indices.push(v.index);
            }
        }

        Ok((pos, indices))
    }

    pub fn clear(&mut self) {
        self.vertices.clear();
        self.edges.clear();
        self.loops.clear();
        self.faces.clear();

        self.vertex_attribs.clear();
        self.edge_attribs.clear();
        self.loop_attribs.clear();
        self.face_attribs.clear();
    }

    // ----- Topology cycle iterators (borrow `&self`) -----
    //
    // These walk a single cycle (vertex disk, edge radial, face loop) and
    // borrow the mesh for the iterator's lifetime, so the mesh cannot be
    // mutated while iterating. For mutation during iteration use the
    // snapshot-key helpers further down (`vert_keys`, `face_loop_keys`,
    // etc.) which return owned `Vec<Key>`.
    //
    // Naming convention:
    //   vertex_disk_*  — cycle of edges around a vertex
    //   face_loop_*    — cycle of loops that define a face's perimeter
    //   edge_radial_*  — fan of loops/faces around an edge

    // ----- Lending cycle iterators -----
    //
    // These return iterator state that does NOT borrow the mesh, so the
    // caller can hold them across `&mut self` mutations:
    //
    // ```ignore
    // let mut it = mesh.disk_edges(v);
    // while let Some(e) = it.next(mesh) {
    //     mesh.some_mutating_method(e);  // &mut self OK between next() calls
    // }
    // ```
    //
    // For read-only `for ... in` ergonomics, see the `iter_*` adapters.

    // ----- Mesh-wide snapshot iterators -----
    //
    // Each returns an owned `Vec<Key>` so the caller may freely mutate the
    // mesh while consuming the iteration: the snapshot-then-mutate idiom,
    // collecting keys first and editing afterward.

    // ----- Reusable scratch buffers -----
    //
    // For the "snapshot keys then mutate" pattern. The mesh owns one reusable
    // Vec per element type. `with_*_keys` temporarily transfers ownership
    // to the call site (mem::take), fills the buf via the zero-alloc
    // mesh-wide iterator, runs the user callback, then restores the buf.
    //
    // For callers that own their own buffer (e.g. amortising allocs
    // across a batch), `collect_*_keys_into` is the primitive — it just
    // clears + refills the supplied Vec.

    // ----- Per-element neighbor collectors -----

    // ----- Direct field accessors -----

    // ----- Internal slab-removal helpers (dirty-flag the matching index table) -----
    //
    // The Euler operators routinely free elements directly. Routing every
    // such free through these helpers keeps the *_INDEX_DIRTY bits honest
    // without sprinkling `self.flags |= …` everywhere.
}

#[test]
fn init_mesh() {
    let mesh = LoopMesh::new();
}

#[test]
fn create_vertex_disk() {
    let mut mesh = LoopMesh::new();
    let v0 = mesh.create_vertex(Vec3::new(0.0, 0.0, 0.0));
    let v1 = mesh.create_vertex(Vec3::new(0.0, 1.0, 0.0));
    let v2 = mesh.create_vertex(Vec3::new(0.0, 0.0, 1.0));
    let edge = mesh.create_edge([v0, v1], EdgeCreateFlags::NONE);
    let edge = mesh.create_edge([v0, v2], EdgeCreateFlags::NONE);
    mesh.vertex_disk_print_edges(v0);
    assert_eq!(mesh.vertex_disk_count_edges(v0), 2);
}

fn mesh_add_cube(mesh: &mut LoopMesh) {
    let v0 = mesh.create_vertex(Vec3::new(1.0, -1.0, -1.0));
    let v1 = mesh.create_vertex(Vec3::new(-1.0, -1.0, -1.0));
    let v2 = mesh.create_vertex(Vec3::new(-1.0, 1.0, -1.0));
    let v3 = mesh.create_vertex(Vec3::new(1.0, 1.0, -1.0));
    let v4 = mesh.create_vertex(Vec3::new(1.0, -1.0, 1.0));
    let v5 = mesh.create_vertex(Vec3::new(-1.0, -1.0, 1.0));
    let v6 = mesh.create_vertex(Vec3::new(-1.0, 1.0, 1.0));
    let v7 = mesh.create_vertex(Vec3::new(1.0, 1.0, 1.0));

    println!("XXX: Creating edge 0");
    let e0 = mesh.create_edge([v0, v1], EdgeCreateFlags::NONE);
    println!(
        "Vertex 0 is connected to {} edges...",
        mesh.vertex_disk_count_edges(v0)
    );
    for e in mesh.vertex_disk_iter(v0) {
        println!("v0, connected to edge {}", e);
    }
    println!("XXX: Creating edge 1");
    let e1 = mesh.create_edge([v1, v2], EdgeCreateFlags::NONE);
    let e2 = mesh.create_edge([v2, v3], EdgeCreateFlags::NONE);
    let e3 = mesh.create_edge([v3, v0], EdgeCreateFlags::NONE);
    println!(
        "Vertex 0 is connected to {} edges...",
        mesh.vertex_disk_count_edges(v0)
    );
    for e in mesh.vertex_disk_iter(v0) {
        println!("v0, connected to edge {}", e);
    }

    let e4 = mesh.create_edge([v4, v5], EdgeCreateFlags::NONE);
    let e5 = mesh.create_edge([v5, v6], EdgeCreateFlags::NONE);
    let e6 = mesh.create_edge([v6, v7], EdgeCreateFlags::NONE);
    let e7 = mesh.create_edge([v7, v4], EdgeCreateFlags::NONE);

    println!("XXX: Creating edge 2");
    let e8 = mesh.create_edge([v0, v4], EdgeCreateFlags::NONE);
    println!(
        "Vertex 0 is connected to {} edges...",
        mesh.vertex_disk_count_edges(v0)
    );
    for e in mesh.vertex_disk_iter(v0) {
        println!("v0, connected to edge {}", e);
    }
    let e9 = mesh.create_edge([v1, v5], EdgeCreateFlags::NONE);
    let e10 = mesh.create_edge([v2, v6], EdgeCreateFlags::NONE);
    let e11 = mesh.create_edge([v3, v7], EdgeCreateFlags::NONE);

    let f0 = mesh.face_create(&[v0, v1, v2, v3], &[e0, e1, e2, e3], FaceCreateFlags::NONE); // Front
    let f1 = mesh.face_create(&[v1, v5, v6, v2], &[e9, e5, e10, e1], FaceCreateFlags::NONE); // Right
    let f2 = mesh.face_create(&[v5, v4, v7, v6], &[e4, e7, e6, e5], FaceCreateFlags::NONE); // Back
    let f3 = mesh.face_create(&[v4, v0, v3, v7], &[e8, e3, e11, e7], FaceCreateFlags::NONE); // Left
    let f4 = mesh.face_create(
        &[v3, v2, v6, v7],
        &[e2, e10, e6, e11],
        FaceCreateFlags::NONE,
    ); // Top
    let f5 = mesh.face_create(&[v0, v4, v5, v1], &[e8, e4, e9, e0], FaceCreateFlags::NONE); // Bottom

    for (i, v) in [v0, v1, v2, v3, v4, v5, v6, v7].iter().enumerate() {
        println!(
            "Vertex {i} is connected to {} edges...",
            mesh.vertex_disk_count_edges(*v)
        );
    }
}

#[test]
fn create_cube_faces() {
    let mut mesh = LoopMesh::new();
    mesh_add_cube(&mut mesh);

    for v in mesh.vertices.iter() {
        assert_eq!(mesh.vertex_disk_count_edges(v), 3);
    }
}

#[test]
fn update_cube_normals() {
    let mut mesh = LoopMesh::new();
    mesh_add_cube(&mut mesh);
    mesh.calc_face_normals();

    let normals: Vec<Vec3> = mesh
        .faces
        .iter()
        .map(|f| mesh.faces.get(f).normal)
        .collect();
    assert_eq!(normals[0], Vec3::new(0.0, 0.0, -1.0));
    assert_eq!(normals[1], Vec3::new(-1.0, 0.0, 0.0));
    assert_eq!(normals[2], Vec3::new(0.0, 0.0, 1.0));
    assert_eq!(normals[3], Vec3::new(1.0, 0.0, 0.0));
    assert_eq!(normals[4], Vec3::new(0.0, 1.0, 0.0));
    assert_eq!(normals[5], Vec3::new(0.0, -1.0, 0.0));
}

#[test]
fn to_from_indexed_vec3() {
    let vertices: Vec<f32> = vec![
        -1.0, -1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0, 1.0, 1.0,
        -1.0, 1.0, 1.0, 1.0, 1.0, -1.0, 1.0, 1.0,
    ];

    let indices: Vec<u32> = vec![
        0, 1, 2, 2, 3, 0, 1, 5, 6, 6, 2, 1, 5, 4, 7, 7, 6, 5, 4, 0, 3, 3, 7, 4, 3, 2, 6, 6, 7, 3,
        0, 4, 5, 5, 1, 0,
    ];

    let mut mesh =
        LoopMesh::new_from_pos_f32(&vertices, Some(&indices), PrimitiveTopology::TriangleList)
            .unwrap();

    let (vertices_out, indices_out) = mesh.into_indexed_vec3_pos_triangles().unwrap();

    assert_eq!(vertices_out.len(), vertices.len());
    assert_eq!(indices_out.len(), indices.len());

    for i in 0..vertices.len() {
        assert_eq!(vertices[i], vertices_out[i]);
    }
    for i in 0..indices.len() {
        assert_eq!(indices[i], indices_out[i]);
    }
}

#[test]
fn test_calc_tri_quad_poly_area() {
    let area = LoopMesh::calc_tri_area([
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(1.0, 1.0, 0.0),
    ]);
    assert_eq!(area, 0.5);

    let area = LoopMesh::calc_tri_area([
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(0.0, 2.0, 0.0),
        Vec3::new(0.0, 2.0, 2.0),
    ]);
    assert_eq!(area, 2.0);

    let area = LoopMesh::calc_quad_area([
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(0.0, 2.0, 0.0),
        Vec3::new(0.0, 2.0, 2.0),
        Vec3::new(0.0, 0.0, 2.0),
    ]);
    assert_eq!(area, 4.0);

    let area = LoopMesh::calc_poly_area(&[
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(0.0, 2.0, 0.0),
        Vec3::new(0.0, 2.0, 2.0),
        Vec3::new(0.0, 0.0, 2.0),
    ]);
    assert_eq!(area, 4.0);

    //             5        4
    //             ______________ 3 (2,1)
    //            /|        |    /
    //          /  |        |  /
    // (-1,0) /____|________ /
    //        0    1        2
    let area = LoopMesh::calc_poly_area(&[
        Vec3::new(-1.0, 0.0, 0.0),
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(2.0, 1.0, 0.0),
        Vec3::new(1.0, 1.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
    ]);
    assert_eq!(area, 2.0);
}

#[test]
fn test_calc_tri_quad_poly_normals() {
    let normal = LoopMesh::calc_tri_normal([
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(2.0, 0.0, 0.0),
        Vec3::new(2.0, 2.0, 0.0),
    ]);
    assert_eq!(normal, Vec3::new(0.0, 0.0, 1.0));

    let normal = LoopMesh::calc_quad_normal([
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(2.0, 0.0, 0.0),
        Vec3::new(2.0, 1.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
    ]);
    assert_eq!(normal, Vec3::new(0.0, 0.0, 1.0));

    let normal = LoopMesh::calc_poly_normal(&[
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(1.0, 2.0, 0.0),
        Vec3::new(0.0, 2.0, 0.0),
    ]);
    assert_eq!(normal, Vec3::new(0.0, 0.0, 1.0));

    //             5        4
    //             ______________ 3 (2,1)
    //            /|        |    /
    //          /  |        |  /
    // (-1,0) /____|________ /
    //        0    1        2
    let normal = LoopMesh::calc_poly_normal(&[
        Vec3::new(-1.0, 0.0, 0.0),
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(2.0, 1.0, 0.0),
        Vec3::new(1.0, 1.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
    ]);
    assert_eq!(normal, Vec3::new(0.0, 0.0, 1.0));

    let normal = LoopMesh::calc_iter_poly_normal(
        [
            Vec3::new(-1.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(2.0, 1.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        ]
        .into_iter(),
    );
    assert_eq!(normal, Vec3::new(0.0, 0.0, 1.0));
}

#[test]
fn test_normalize_or_zero_unit_and_degenerate() {
    // Non-degenerate input normalises to unit length.
    let n = LoopMesh::normalize_or_zero(Vec3::new(0.0, 0.0, 5.0));
    assert_eq!(n, Vec3::new(0.0, 0.0, 1.0));

    // True zero falls back to the all-zero vector, never NaN.
    let n = LoopMesh::normalize_or_zero(Vec3::ZERO);
    assert_eq!(n, Vec3::ZERO);

    // A NaN component fails the `> 1.0e-35` test and takes the zero branch.
    let n = LoopMesh::normalize_or_zero(Vec3::new(f32::NAN, 0.0, 0.0));
    assert_eq!(n, Vec3::ZERO);
}

#[test]
fn test_degenerate_tri_normal_is_zero() {
    // Collinear corners -> zero-area triangle -> zero normal.
    let normal = LoopMesh::calc_tri_normal([
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(2.0, 0.0, 0.0),
    ]);
    assert_eq!(normal, Vec3::ZERO);
}

#[test]
fn test_degenerate_quad_normal_is_zero() {
    // Parallel diagonals (both along X) -> zero diagonal cross -> zero normal,
    // even though the quad's edges are non-zero.
    let normal = LoopMesh::calc_quad_normal([
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(2.0, 0.0, 0.0),
        Vec3::new(3.0, 0.0, 0.0),
    ]);
    assert_eq!(normal, Vec3::ZERO);

    // Coincident-corner sliver quad (two pairs collapse) -> zero normal.
    let normal = LoopMesh::calc_quad_normal([
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(1.0, 0.0, 0.0),
    ]);
    assert_eq!(normal, Vec3::ZERO);
}

#[test]
fn test_degenerate_poly_normal_is_zero() {
    // Collinear n-gon -> Newell accumulator is zero -> zero normal.
    let corners = [
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(2.0, 0.0, 0.0),
        Vec3::new(3.0, 0.0, 0.0),
        Vec3::new(4.0, 0.0, 0.0),
    ];
    assert_eq!(LoopMesh::calc_poly_normal(&corners), Vec3::ZERO);
    assert_eq!(
        LoopMesh::calc_iter_poly_normal(corners.into_iter()),
        Vec3::ZERO
    );
}

fn create_face_from_verts(mesh: &mut LoopMesh, coords: &[Vec3]) -> FaceKey {
    let mut verts = vec![];
    for v in coords {
        verts.push(mesh.create_vertex(*v));
    }
    mesh.face_create_verts(&verts, FaceCreateFlags::NONE)
}

#[test]
fn test_calc_face_normals() {
    let mut mesh = LoopMesh::new();

    let verts = [
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(2.0, 0.0, 0.0),
        Vec3::new(2.0, 2.0, 0.0),
    ];
    let face = create_face_from_verts(&mut mesh, &verts);
    mesh.face_calc_normal(face);
    let normal = mesh.faces.get(face).normal;
    assert_eq!(normal, Vec3::new(0.0, 0.0, 1.0));

    let verts = [
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(2.0, 0.0, 0.0),
        Vec3::new(2.0, 1.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
    ];
    let face = create_face_from_verts(&mut mesh, &verts);
    mesh.face_calc_normal(face);
    let normal = mesh.faces.get(face).normal;
    assert_eq!(normal, Vec3::new(0.0, 0.0, 1.0));

    //             5        4
    //             ______________ 3 (2,1)
    //            /|        |    /
    //          /  |        |  /
    // (-1,0) /____|________ /
    //        0    1        2
    let verts = [
        Vec3::new(-1.0, 0.0, 0.0),
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(2.0, 1.0, 0.0),
        Vec3::new(1.0, 1.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
    ];
    let face = create_face_from_verts(&mut mesh, &verts);
    mesh.face_calc_normal(face);
    let normal = mesh.faces.get(face).normal;
    assert_eq!(normal, Vec3::new(0.0, 0.0, 1.0));
}

#[test]
fn test_calc_face_area() {
    let mut mesh = LoopMesh::new();

    let verts = [
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(1.0, 1.0, 0.0),
    ];
    let face = create_face_from_verts(&mut mesh, &verts);
    let area = mesh.face_calc_area(face);
    assert_eq!(area, 0.5);

    let verts = [
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(0.0, 2.0, 0.0),
        Vec3::new(0.0, 2.0, 2.0),
        Vec3::new(0.0, 0.0, 2.0),
    ];
    let face = create_face_from_verts(&mut mesh, &verts);
    let area = mesh.face_calc_area(face);
    assert_eq!(area, 4.0);

    //             5        4
    //             ______________ 3 (2,1)
    //            /|        |    /
    //          /  |        |  /
    // (-1,0) /____|________ /
    //        0    1        2
    let verts = [
        Vec3::new(-1.0, 0.0, 0.0),
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(2.0, 1.0, 0.0),
        Vec3::new(1.0, 1.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
    ];
    let face = create_face_from_verts(&mut mesh, &verts);
    let area = mesh.face_calc_area(face);
    assert_eq!(area, 2.0);
}

// Operator + Euler unit tests now live alongside their implementations
// in loopmesh/euler/*.rs and loopmesh/ops/*.rs.

// ----- Element-index machinery tests -----

#[test]
fn is_dirty_starts_set_clears_after_ensure() {
    let mut mesh = LoopMesh::new();
    let _v = mesh.create_vertex(Vec3::ZERO);

    assert!(mesh.is_vertices_index_dirty());
    mesh.vertices_ensure_index();
    assert!(!mesh.is_vertices_index_dirty());

    // Re-set by a subsequent mutation.
    let _v2 = mesh.create_vertex(Vec3::X);
    assert!(mesh.is_vertices_index_dirty());
}

#[test]
fn ensure_index_field_clears_only_its_axis() {
    let mut mesh = LoopMesh::new();
    let v0 = mesh.create_vertex(Vec3::ZERO);
    let v1 = mesh.create_vertex(Vec3::X);
    let _e = mesh.create_edge([v0, v1], EdgeCreateFlags::NONE);

    assert!(mesh.is_vertices_index_dirty());
    assert!(mesh.is_edges_index_dirty());

    mesh.vertices_ensure_index_field();
    assert!(!mesh.is_vertices_index_dirty());
    // Edges still dirty.
    assert!(mesh.is_edges_index_dirty());
}

#[test]
fn ensure_index_field_assigns_dense_indices() {
    let mut mesh = LoopMesh::new();
    let v0 = mesh.create_vertex(Vec3::ZERO);
    let v1 = mesh.create_vertex(Vec3::X);
    let v2 = mesh.create_vertex(Vec3::Y);
    mesh.vertices_ensure_index_field();

    // Indices are 0..3 in slab order (no holes yet).
    assert_eq!(mesh.vertex_index(v0), 0);
    assert_eq!(mesh.vertex_index(v1), 1);
    assert_eq!(mesh.vertex_index(v2), 2);
    assert_eq!(mesh.vertex_index_table(), &[v0, v1, v2]);
}

#[test]
fn validate_indices_passes_after_full_ensure() {
    let mut mesh = LoopMesh::new();
    let verts = [
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(1.0, 1.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
    ];
    let _face = create_face_from_verts(&mut mesh, &verts);

    mesh.all_indices_ensure();
    mesh.validate_indices();
}

#[test]
#[should_panic(expected = "indices are dirty")]
fn validate_indices_panics_when_dirty() {
    let mut mesh = LoopMesh::new();
    let _v = mesh.create_vertex(Vec3::ZERO);
    // No ensure call → still dirty.
    mesh.validate_indices();
}

// ----- validate_topology tests -----

#[test]
fn validate_topology_passes_on_empty_mesh() {
    LoopMesh::new().validate_topology();
}

#[test]
fn validate_topology_passes_on_lone_vertex() {
    let mut mesh = LoopMesh::new();
    let _ = mesh.create_vertex(Vec3::ZERO);
    mesh.validate_topology();
}

#[test]
fn validate_topology_passes_on_quad() {
    let mut mesh = LoopMesh::new();
    let verts = [
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(1.0, 1.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
    ];
    let _f = create_face_from_verts(&mut mesh, &verts);
    mesh.validate_topology();
}

#[test]
fn validate_topology_passes_after_edge_split() {
    let mut mesh = LoopMesh::new();
    let verts = [
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(1.0, 1.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
    ];
    let f = create_face_from_verts(&mut mesh, &verts);
    // Split one edge to exercise SEMV's transient null-edge state.
    let l = mesh.face_first_loop(f);
    let e = mesh.loop_edge(l);
    let v_near = mesh.loop_vertex(l);
    mesh.edge_split(e, v_near, 0.5);
    mesh.validate_topology();
}

#[test]
fn validate_topology_passes_after_face_split_and_poke() {
    let mut mesh = LoopMesh::new();
    let verts = [
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(1.0, 1.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
    ];
    let f = create_face_from_verts(&mut mesh, &verts);
    let face_verts = mesh.face_vert_keys(f);
    let v0 = face_verts[0];
    let v2 = face_verts[2];
    let (new_face, _new_edge) = mesh.face_split(f, v0, v2).unwrap();
    mesh.validate_topology();
    // Poke one of the halves.
    mesh.face_poke(new_face);
    mesh.validate_topology();
}

// ----- Op-flag pool integration tests -----

#[test]
fn op_flag_layer_round_trip() {
    let mut mesh = LoopMesh::new();
    let v0 = mesh.create_vertex(Vec3::ZERO);
    let v1 = mesh.create_vertex(Vec3::X);
    let v2 = mesh.create_vertex(Vec3::Y);

    let layer = mesh.push_vert_op_flag_layer();
    assert!(!mesh.vert_op_flag_test(layer, v0, 3));
    mesh.vert_op_flag_set(layer, v0, 3);
    mesh.vert_op_flag_set(layer, v2, 3);
    assert!(mesh.vert_op_flag_test(layer, v0, 3));
    assert!(!mesh.vert_op_flag_test(layer, v1, 3));
    assert!(mesh.vert_op_flag_test(layer, v2, 3));

    let marked: Vec<_> = mesh.iter_verts_with_op_flag(layer, 3).collect();
    assert_eq!(marked, vec![v0, v2]);

    mesh.pop_vert_op_flag_layer(layer);
}

#[test]
fn op_flag_eager_growth_on_create_vertex() {
    let mut mesh = LoopMesh::new();
    let v0 = mesh.create_vertex(Vec3::ZERO);

    let layer = mesh.push_vert_op_flag_layer();
    mesh.vert_op_flag_set(layer, v0, 0);

    // Create more vertices after pushing the layer; eager growth should
    // extend the active layer so set/test work without panic.
    let v1 = mesh.create_vertex(Vec3::X);
    let v2 = mesh.create_vertex(Vec3::Y);

    // v0's bit is still set; v1/v2 default to zero.
    assert!(mesh.vert_op_flag_test(layer, v0, 0));
    assert!(!mesh.vert_op_flag_test(layer, v1, 0));
    mesh.vert_op_flag_set(layer, v2, 0);
    assert!(mesh.vert_op_flag_test(layer, v2, 0));

    mesh.pop_vert_op_flag_layer(layer);
}

#[test]
fn op_flag_layer_reuse_zero_initialises() {
    let mut mesh = LoopMesh::new();
    let v = mesh.create_vertex(Vec3::ZERO);

    let layer_a = mesh.push_vert_op_flag_layer();
    mesh.vert_op_flag_set(layer_a, v, 5);
    mesh.pop_vert_op_flag_layer(layer_a);

    let layer_b = mesh.push_vert_op_flag_layer();
    // Reused layer must be zero-cleared, regardless of layer-handle equality.
    assert!(!mesh.vert_op_flag_test(layer_b, v, 5));
    mesh.pop_vert_op_flag_layer(layer_b);
}

#[test]
fn op_flag_each_axis_is_independent() {
    let mut mesh = LoopMesh::new();
    let v0 = mesh.create_vertex(Vec3::ZERO);
    let v1 = mesh.create_vertex(Vec3::X);
    let e = mesh.create_edge([v0, v1], EdgeCreateFlags::NONE);

    let vl = mesh.push_vert_op_flag_layer();
    let el = mesh.push_edge_op_flag_layer();

    mesh.vert_op_flag_set(vl, v0, 0);
    mesh.edge_op_flag_set(el, e, 0);

    assert!(mesh.vert_op_flag_test(vl, v0, 0));
    assert!(!mesh.vert_op_flag_test(vl, v1, 0));
    assert!(mesh.edge_op_flag_test(el, e, 0));

    mesh.pop_edge_op_flag_layer(el);
    mesh.pop_vert_op_flag_layer(vl);
}

// extrude_region tests now live in loopmesh/ops/extrude_region.rs.

#[test]
fn i32_vertex_attrib_round_trips() {
    let mut mesh = LoopMesh::new();
    let v0 = mesh.create_vertex(Vec3::ZERO);
    let v1 = mesh.create_vertex(Vec3::X);

    let material = mesh
        .register_vertex_attribute(ElementAttributeDesc::numeric(
            AttributeKey::custom("material"),
            AttrType::I32,
        ))
        .unwrap();

    assert!(mesh
        .vertex_attribs_mut()
        .set_i32(material, v0.as_usize(), 7));
    assert!(mesh
        .vertex_attribs_mut()
        .set_i32(material, v1.as_usize(), -3));

    assert_eq!(
        mesh.vertex_attribs().get_i32(material, v0.as_usize()),
        Some(7)
    );
    assert_eq!(
        mesh.vertex_attribs().get_i32(material, v1.as_usize()),
        Some(-3)
    );
}

#[test]
fn i32_loop_attrib_round_trips() {
    let mut mesh = LoopMesh::new();
    let v0 = mesh.create_vertex(Vec3::ZERO);
    let v1 = mesh.create_vertex(Vec3::X);
    let v2 = mesh.create_vertex(Vec3::Y);
    let e0 = mesh.create_edge([v0, v1], EdgeCreateFlags::NONE);
    let e1 = mesh.create_edge([v1, v2], EdgeCreateFlags::NONE);
    let e2 = mesh.create_edge([v2, v0], EdgeCreateFlags::NONE);
    let f = mesh.face_create(&[v0, v1, v2], &[e0, e1, e2], FaceCreateFlags::NONE);

    let tag = mesh
        .register_loop_attribute(ElementAttributeDesc::numeric(
            AttributeKey::custom("tag"),
            AttrType::I32,
        ))
        .unwrap();

    let loops: Vec<LoopKey> = mesh.iter_face_loops(f).collect();
    for (i, l) in loops.iter().enumerate() {
        assert!(mesh
            .loop_attribs_mut()
            .set_i32(tag, l.as_usize(), i as i32 * 10));
    }
    for (i, l) in loops.iter().enumerate() {
        assert_eq!(
            mesh.loop_attribs().get_i32(tag, l.as_usize()),
            Some(i as i32 * 10)
        );
    }
}

#[test]
fn i32_setter_rejects_wrong_bucket_and_missing_key() {
    let mut mesh = LoopMesh::new();
    let v0 = mesh.create_vertex(Vec3::ZERO);

    // A float-bucket attribute must not accept an i32 write.
    let color = mesh
        .register_vertex_attribute(ElementAttributeDesc::numeric(
            AttributeKey::custom("color"),
            AttrType::Vec3,
        ))
        .unwrap();
    assert!(!mesh.vertex_attribs_mut().set_i32(color, v0.as_usize(), 1));
    assert_eq!(mesh.vertex_attribs().get_i32(color, v0.as_usize()), None);

    // An unregistered key resolves to nothing.
    let bogus = ElementAttributeKey(9999);
    assert!(!mesh.vertex_attribs_mut().set_i32(bogus, v0.as_usize(), 1));
    assert_eq!(mesh.vertex_attribs().get_i32(bogus, v0.as_usize()), None);
}
