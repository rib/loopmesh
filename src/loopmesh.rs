#![allow(dead_code)]
#![allow(unused_variables)]

use std::{fmt::Display, marker::PhantomData};

use anyhow::{anyhow, Result};
use bitflags::bitflags;
use glam::*;
use slab::Slab;
use smallvec::SmallVec;

pub use nub_mesh::attribute::{AttributeKey, AttributeSemantic};
pub use nub_mesh::format::{ComponentType, Format};

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
mod euler;
mod mid;
mod ops;

pub use mid::EdgeCollapseFlags;
pub use ops::{
    BisectGeom, BisectParams, BisectResult,
    BridgeLoopsParams, BridgeLoopsResult,
    ConnectVertsNonplanarOut,
    CenterMode, DeleteContext, DeleteSeed, DissolveEdgesParams, DissolveEdgesResult,
    DissolveFacesOptions, DissolveFacesResult, DissolveLimitParams, DissolveLimitResult,
    DissolveVertsParams, DuplicateResult, ExtrudeVertIndivResult, GeomKey, InsetIndividualParams,
    InsetIndividualResult, InsetRegionParams, InsetRegionResult, Axis, MirrorParams, MirrorResult,
    JoinTrianglesParams, JoinTrianglesResult,
    MixedSeed, NgonMethod, QuadCornerType,
    QuadMethod, RotateEdgesResult, RotateParams, ScaleParams, SmoothFalloff, SubdivideGeom, SubdivideParams, SubdivideResult,
    SymmetrizeDirection, SymmetrizeResult,
    TransformParams, TranslateParams,
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

pub struct DiskEdgesIterator<'a> {
    mesh: &'a LoopMesh,

    hub_vertex: VertexKey,
    first_edge: Option<EdgeKey>,
    next_edge: Option<EdgeKey>,
}

impl<'a> DiskEdgesIterator<'a> {
    pub fn new(mesh: &'a LoopMesh, hub_vertex: VertexKey) -> Self {
        let vertex = &mesh.vertices.get(hub_vertex);
        DiskEdgesIterator {
            mesh,
            hub_vertex,
            first_edge: vertex.disk_loop_start,
            next_edge: vertex.disk_loop_start,
        }
    }
}

impl<'a> Iterator for DiskEdgesIterator<'a> {
    type Item = EdgeKey;

    fn next(&mut self) -> Option<Self::Item> {
        let cur = self.next_edge;

        if let Some(cur) = cur {
            let links = self
                .mesh
                .edges
                .get(cur)
                .disk_loop_links_for_vert(self.hub_vertex);
            self.next_edge = if Some(links.next) == self.first_edge {
                None
            } else {
                Some(links.next)
            }
        }

        cur
    }
}

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

pub struct RadialLoopIterator<'a> {
    mesh: &'a LoopMesh,

    axis_edge: EdgeKey,
    first_loop: Option<LoopKey>,
    next_loop: Option<LoopKey>,
}

impl<'a> RadialLoopIterator<'a> {
    pub fn new(mesh: &'a LoopMesh, axis_edge: EdgeKey) -> Self {
        let edge = &mesh.edges.get(axis_edge);
        RadialLoopIterator {
            mesh,
            axis_edge,
            first_loop: edge.radial_link,
            next_loop: edge.radial_link,
        }
    }
}

impl<'a> Iterator for RadialLoopIterator<'a> {
    type Item = LoopKey;

    fn next(&mut self) -> Option<Self::Item> {
        let cur = self.next_loop;

        if let Some(cur) = cur {
            let links = self.mesh.loops.get(cur).edge_radial_links;
            self.next_loop = if Some(links.next) == self.first_loop {
                None
            } else {
                Some(links.next)
            }
        }

        cur
    }
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

pub struct FaceEdgeLoopIterator<'a> {
    mesh: &'a LoopMesh,

    first_edge_loop: Option<LoopKey>,
    next_edge_loop: Option<LoopKey>,
}

impl<'a> FaceEdgeLoopIterator<'a> {
    pub fn new(mesh: &'a LoopMesh, face_loop_start: LoopKey) -> Self {
        FaceEdgeLoopIterator {
            mesh,
            first_edge_loop: Some(face_loop_start),
            next_edge_loop: Some(face_loop_start),
        }
    }
}

impl<'a> Iterator for FaceEdgeLoopIterator<'a> {
    type Item = LoopKey;

    fn next(&mut self) -> Option<Self::Item> {
        let cur = self.next_edge_loop;

        if let Some(cur) = cur {
            let links = self.mesh.loops.get(cur).face_loop_links;
            self.next_edge_loop = if Some(links.next) == self.first_edge_loop {
                None
            } else {
                Some(links.next)
            }
        }

        cur
    }
}

pub struct FaceEdgeLoopReverseIterator<'a> {
    mesh: &'a LoopMesh,

    first_edge_loop: Option<LoopKey>,
    next_edge_loop: Option<LoopKey>,
}

impl<'a> FaceEdgeLoopReverseIterator<'a> {
    pub fn new(mesh: &'a LoopMesh, face_loop_start: LoopKey) -> Self {
        FaceEdgeLoopReverseIterator {
            mesh,
            first_edge_loop: Some(face_loop_start),
            next_edge_loop: Some(face_loop_start),
        }
    }
}

impl<'a> Iterator for FaceEdgeLoopReverseIterator<'a> {
    type Item = LoopKey;

    fn next(&mut self) -> Option<Self::Item> {
        let cur = self.next_edge_loop;

        if let Some(cur) = cur {
            let links = self.mesh.loops.get(cur).face_loop_links;
            self.next_edge_loop = if Some(links.prev) == self.first_edge_loop {
                None
            } else {
                Some(links.prev)
            }
        }

        cur
    }
}

// =====================================================================
// Lending iterators
//
// These mirror the borrowing iterators above but their state is pure POD —
// they hold no reference to the mesh. Advance them by calling `next(&mesh)`
// explicitly; that way `&mut mesh` mutation is allowed between steps.
// =====================================================================

/// Trait implemented by every lending iterator. The mesh reference is
/// passed in per call rather than held in the iterator, so the caller is
/// free to mutate the mesh between `next` invocations.
pub trait MeshIter {
    type Item;
    fn next(&mut self, mesh: &LoopMesh) -> Option<Self::Item>;
}

/// Cycle iterator over the edges incident to a vertex (the disk cycle).
pub struct DiskEdges {
    hub: VertexKey,
    first: Option<EdgeKey>,
    next: Option<EdgeKey>,
}

impl DiskEdges {
    fn new(mesh: &LoopMesh, v: VertexKey) -> Self {
        let start = mesh.vertices.get(v).disk_loop_start;
        Self {
            hub: v,
            first: start,
            next: start,
        }
    }
}

impl MeshIter for DiskEdges {
    type Item = EdgeKey;
    fn next(&mut self, mesh: &LoopMesh) -> Option<EdgeKey> {
        let cur = self.next?;
        let links = mesh.edges.get(cur).disk_loop_links_for_vert(self.hub);
        self.next = if Some(links.next) == self.first {
            None
        } else {
            Some(links.next)
        };
        Some(cur)
    }
}

/// Cycle iterator over the loops on an edge's radial fan.
pub struct RadialLoops {
    axis: EdgeKey,
    first: Option<LoopKey>,
    next: Option<LoopKey>,
}

impl RadialLoops {
    fn new(mesh: &LoopMesh, e: EdgeKey) -> Self {
        let start = mesh.edges.get(e).radial_link;
        Self {
            axis: e,
            first: start,
            next: start,
        }
    }
}

impl MeshIter for RadialLoops {
    type Item = LoopKey;
    fn next(&mut self, mesh: &LoopMesh) -> Option<LoopKey> {
        let cur = self.next?;
        let _ = self.axis; // axis retained for symmetry; not needed for the walk
        let links = mesh.loops.get(cur).edge_radial_links;
        self.next = if Some(links.next) == self.first {
            None
        } else {
            Some(links.next)
        };
        Some(cur)
    }
}

/// Cycle iterator over the loops of a face (forward winding).
pub struct FaceLoops {
    first: Option<LoopKey>,
    next: Option<LoopKey>,
}

impl FaceLoops {
    fn new(mesh: &LoopMesh, f: FaceKey) -> Self {
        let start = Some(mesh.faces.get(f).first_edge_loop);
        Self {
            first: start,
            next: start,
        }
    }
}

impl MeshIter for FaceLoops {
    type Item = LoopKey;
    fn next(&mut self, mesh: &LoopMesh) -> Option<LoopKey> {
        let cur = self.next?;
        let links = mesh.loops.get(cur).face_loop_links;
        self.next = if Some(links.next) == self.first {
            None
        } else {
            Some(links.next)
        };
        Some(cur)
    }
}

/// Cycle iterator over the loops of a face (reverse winding).
pub struct FaceLoopsReverse {
    first: Option<LoopKey>,
    next: Option<LoopKey>,
}

impl FaceLoopsReverse {
    fn new(mesh: &LoopMesh, f: FaceKey) -> Self {
        let start = Some(mesh.faces.get(f).first_edge_loop);
        Self {
            first: start,
            next: start,
        }
    }
}

impl MeshIter for FaceLoopsReverse {
    type Item = LoopKey;
    fn next(&mut self, mesh: &LoopMesh) -> Option<LoopKey> {
        let cur = self.next?;
        let links = mesh.loops.get(cur).face_loop_links;
        self.next = if Some(links.prev) == self.first {
            None
        } else {
            Some(links.prev)
        };
        Some(cur)
    }
}

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

/// Lending iterator over every live vertex in the mesh.
#[derive(Default)]
pub struct MeshVerts {
    cursor: usize,
}

impl MeshIter for MeshVerts {
    type Item = VertexKey;
    fn next(&mut self, mesh: &LoopMesh) -> Option<VertexKey> {
        let bound = mesh.vertices.key_watermark();
        while self.cursor < bound {
            let i = self.cursor;
            self.cursor += 1;
            if mesh.vertices.try_get(VertexKey::from_usize(i)).is_some() {
                return Some(VertexKey::from_usize(i));
            }
        }
        None
    }
}

/// Lending iterator over every live edge in the mesh.
#[derive(Default)]
pub struct MeshEdges {
    cursor: usize,
}

impl MeshIter for MeshEdges {
    type Item = EdgeKey;
    fn next(&mut self, mesh: &LoopMesh) -> Option<EdgeKey> {
        let bound = mesh.edges.key_watermark();
        while self.cursor < bound {
            let i = self.cursor;
            self.cursor += 1;
            if mesh.edges.try_get(EdgeKey::from_usize(i)).is_some() {
                return Some(EdgeKey::from_usize(i));
            }
        }
        None
    }
}

/// Lending iterator over every live face in the mesh.
#[derive(Default)]
pub struct MeshFaces {
    cursor: usize,
}

impl MeshIter for MeshFaces {
    type Item = FaceKey;
    fn next(&mut self, mesh: &LoopMesh) -> Option<FaceKey> {
        let bound = mesh.faces.key_watermark();
        while self.cursor < bound {
            let i = self.cursor;
            self.cursor += 1;
            if mesh.faces.try_get(FaceKey::from_usize(i)).is_some() {
                return Some(FaceKey::from_usize(i));
            }
        }
        None
    }
}

/// Lending iterator over every live loop in the mesh.
#[derive(Default)]
pub struct MeshLoops {
    cursor: usize,
}

impl MeshIter for MeshLoops {
    type Item = LoopKey;
    fn next(&mut self, mesh: &LoopMesh) -> Option<LoopKey> {
        let bound = mesh.loops.key_watermark();
        while self.cursor < bound {
            let i = self.cursor;
            self.cursor += 1;
            if mesh.loops.try_get(LoopKey::from_usize(i)).is_some() {
                return Some(LoopKey::from_usize(i));
            }
        }
        None
    }
}

/// Drain a [`MeshIter`] into a `SmallVec` so the caller can subsequently
/// mutate the mesh in ways that would invalidate the iteration (e.g.
/// killing elements on the cycle being traversed). Avoids heap allocation
/// for cycles of length ≤ N.
pub fn collect_iter<I: MeshIter, const N: usize>(
    mesh: &LoopMesh,
    mut it: I,
) -> smallvec::SmallVec<[I::Item; N]>
where
    [I::Item; N]: smallvec::Array<Item = I::Item>,
{
    let mut out = smallvec::SmallVec::<[I::Item; N]>::new();
    while let Some(x) = it.next(mesh) {
        out.push(x);
    }
    out
}

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
    _phantom: PhantomData<I>,
}

impl<I, ElementType> ElementSlab<I, ElementType>
where
    I: ElementKey + ElementKeyFromUsize + Copy,
{
    pub fn new() -> Self {
        Self {
            slab: Slab::new(),
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
    }

    /// Number of live entries.
    #[allow(dead_code)]
    pub(crate) fn len(&self) -> usize {
        self.slab.len()
    }

    /// The slab's high-water-mark key (used by attribute storage to size
    /// parallel Vecs). Forwards to the patched slab fork.
    pub(crate) fn key_watermark(&self) -> usize {
        self.slab.key_watermark()
    }

    /// Reserve a vacant slot for a new entry — direct passthrough to the
    /// underlying slab. The caller fills it via the returned VacantEntry.
    pub(crate) fn vacant_entry(&mut self) -> slab::VacantEntry<'_, ElementType> {
        self.slab.vacant_entry()
    }
}

/// What kind of value an element attribute holds.
///
/// `Numeric` attributes carry a [`Format`] from nub-mesh — that gives loopmesh
/// the same encoding vocabulary used by `nub-mesh::Mesh` on export
/// (normalize / as_float / F16 / packed Unorm10_10_10_2 / …) and is the
/// piece that round-trips through the GPU-side mesh representation.
///
/// `String` attributes mirror bmesh's `CD_PROP_STRING` — they live only on
/// the loopmesh editing structure and do not round-trip through a
/// `nub-mesh::Mesh` (no GPU representation).
#[derive(Clone, Debug)]
pub enum ElementAttributeKind {
    Numeric(Format),
    String,
}

/// Subset of [`Format`] variants that the in-memory storage currently
/// understands. `register_*_attribute` returns an error for any other
/// `Format`; expand this as new storage buckets are added.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum NumericBucket {
    U8,
    I32,
    U32,
    F32,
    Vec2,
    Vec3,
    Vec4,
}

impl NumericBucket {
    fn for_format(format: Format) -> Option<Self> {
        if format.normalize || format.as_float || format.packed.is_some() {
            // The wire format is more compact than the in-memory storage;
            // attributes get decoded to one of the canonical buckets above
            // before being stored. Caller should pick a decoded Format.
            return None;
        }
        let n = format.components;
        match (format.ty, n) {
            (ComponentType::U8, 1) => Some(NumericBucket::U8),
            (ComponentType::I32, 1) => Some(NumericBucket::I32),
            (ComponentType::U32, 1) => Some(NumericBucket::U32),
            (ComponentType::F32, 1) => Some(NumericBucket::F32),
            (ComponentType::F32, 2) => Some(NumericBucket::Vec2),
            (ComponentType::F32, 3) => Some(NumericBucket::Vec3),
            (ComponentType::F32, 4) => Some(NumericBucket::Vec4),
            _ => None,
        }
    }
}

/// Where an attribute's per-element values live inside [`ElementAttributes`].
/// Indexes into one of the typed buckets; the `Slab<ElementAttributeDesc>`
/// remembers which.
#[derive(Clone, Copy, Debug)]
pub(crate) enum AttributeStorage {
    U8(usize),
    I32(usize),
    U32(usize),
    F32(usize),
    Vec2(usize),
    Vec3(usize),
    Vec4(usize),
    String(usize),
}

/// How an attribute is blended when a mid-level op creates a new element
/// from a weighted set of source elements.
///
/// Default per kind:
/// - numeric float → [`Linear`](Self::Linear)
/// - numeric integer / byte → [`Nearest`](Self::Nearest)
/// - string → [`CopyFirst`](Self::CopyFirst) (the only sane default)
///
/// Override via [`ElementAttributeDesc::with_interp`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InterpPolicy {
    /// Weighted sum of source values, normalised by total weight.
    Linear,
    /// Source with the largest weight wins. Ties resolved by first-encountered.
    Nearest,
    /// Always pick `sources[0]`. Lossy; the only viable default for strings
    /// and "ID-like" attributes.
    CopyFirst,
    /// Leave the destination's default-init value untouched. Useful for
    /// attributes that have no defensible interpolation rule.
    NoInterp,
}

impl InterpPolicy {
    /// Default policy for a given attribute kind.
    pub fn default_for(kind: &ElementAttributeKind) -> Self {
        match kind {
            ElementAttributeKind::Numeric(format) => match format.ty {
                ComponentType::F16 | ComponentType::F32 => InterpPolicy::Linear,
                _ => InterpPolicy::Nearest,
            },
            ElementAttributeKind::String => InterpPolicy::CopyFirst,
        }
    }
}

#[derive(Clone, Debug)]
pub struct ElementAttributeDesc {
    pub key: AttributeKey,
    pub kind: ElementAttributeKind,
    pub interp: InterpPolicy,
    storage: AttributeStorage,
}

impl ElementAttributeDesc {
    /// Build a numeric attribute descriptor. `key` identifies the attribute;
    /// `format` describes the in-memory value layout. Interp policy defaults
    /// from the format's component type.
    pub fn numeric(key: impl Into<AttributeKey>, format: Format) -> Self {
        let kind = ElementAttributeKind::Numeric(format);
        let interp = InterpPolicy::default_for(&kind);
        Self {
            key: key.into(),
            kind,
            interp,
            // Placeholder — register_*_attribute resolves this when the
            // descriptor is inserted into a specific ElementAttributes.
            storage: AttributeStorage::U8(usize::MAX),
        }
    }

    /// Build a string attribute descriptor. Interp policy defaults to
    /// [`InterpPolicy::CopyFirst`].
    pub fn string(key: impl Into<AttributeKey>) -> Self {
        let kind = ElementAttributeKind::String;
        let interp = InterpPolicy::default_for(&kind);
        Self {
            key: key.into(),
            kind,
            interp,
            storage: AttributeStorage::String(usize::MAX),
        }
    }

    /// Override the default interpolation policy.
    pub fn with_interp(mut self, interp: InterpPolicy) -> Self {
        self.interp = interp;
        self
    }
}

#[derive(PartialEq, Eq, Clone, Copy, Debug, Hash)]
pub struct ElementAttributeKey(u32);

impl ElementAttributeKey {
    pub(crate) fn as_usize(self) -> usize {
        self.0 as usize
    }
}

/// Element attributes mirror the slab layout for the corresponding
/// element type. The descriptor `Slab` carries the public-facing key and
/// kind; the typed buckets carry the actual values, one slot per element-
/// slab key. Vacant element slots have undefined values in each bucket
/// (mirrors the parallel-Vec storage that was here before nub-mesh
/// integration).
#[derive(Default)]
pub struct ElementAttributes {
    attributes: Slab<ElementAttributeDesc>,
    attributes_u8: Vec<Vec<u8>>,
    attributes_i32: Vec<Vec<i32>>,
    attributes_u32: Vec<Vec<u32>>,
    attributes_f32: Vec<Vec<f32>>,
    attributes_vec2: Vec<Vec<[f32; 2]>>,
    attributes_vec3: Vec<Vec<[f32; 3]>>,
    attributes_vec4: Vec<Vec<[f32; 4]>>,
    attributes_string: Vec<Vec<Option<String>>>,
}

impl ElementAttributes {
    fn clear(&mut self) {
        self.attributes.clear();
        self.attributes_u8.clear();
        self.attributes_i32.clear();
        self.attributes_u32.clear();
        self.attributes_f32.clear();
        self.attributes_vec2.clear();
        self.attributes_vec3.clear();
        self.attributes_vec4.clear();
        self.attributes_string.clear();
    }

    /// Reserve a new attribute slot in the matching typed bucket and
    /// return the resolved [`AttributeStorage`]. Used by every
    /// `register_*_attribute`.
    fn insert_with_storage(
        &mut self,
        mut desc: ElementAttributeDesc,
        key_watermark: usize,
    ) -> Result<ElementAttributeKey> {
        let storage = match &desc.kind {
            ElementAttributeKind::Numeric(format) => {
                let bucket = NumericBucket::for_format(*format).ok_or_else(|| {
                    anyhow!(
                        "format {:?} has no in-memory bucket on LoopMesh; \
                         supported: F32x{{1..=4}}, U8x1, I32x1, U32x1 (with \
                         no normalize / as_float / packed)",
                        format,
                    )
                })?;
                match bucket {
                    NumericBucket::U8 => {
                        let idx = self.attributes_u8.len();
                        self.attributes_u8.push(vec![0; key_watermark]);
                        AttributeStorage::U8(idx)
                    }
                    NumericBucket::I32 => {
                        let idx = self.attributes_i32.len();
                        self.attributes_i32.push(vec![0; key_watermark]);
                        AttributeStorage::I32(idx)
                    }
                    NumericBucket::U32 => {
                        let idx = self.attributes_u32.len();
                        self.attributes_u32.push(vec![0; key_watermark]);
                        AttributeStorage::U32(idx)
                    }
                    NumericBucket::F32 => {
                        let idx = self.attributes_f32.len();
                        self.attributes_f32.push(vec![0.0; key_watermark]);
                        AttributeStorage::F32(idx)
                    }
                    NumericBucket::Vec2 => {
                        let idx = self.attributes_vec2.len();
                        self.attributes_vec2.push(vec![[0.0; 2]; key_watermark]);
                        AttributeStorage::Vec2(idx)
                    }
                    NumericBucket::Vec3 => {
                        let idx = self.attributes_vec3.len();
                        self.attributes_vec3.push(vec![[0.0; 3]; key_watermark]);
                        AttributeStorage::Vec3(idx)
                    }
                    NumericBucket::Vec4 => {
                        let idx = self.attributes_vec4.len();
                        self.attributes_vec4.push(vec![[0.0; 4]; key_watermark]);
                        AttributeStorage::Vec4(idx)
                    }
                }
            }
            ElementAttributeKind::String => {
                let idx = self.attributes_string.len();
                self.attributes_string.push(vec![None; key_watermark]);
                AttributeStorage::String(idx)
            }
        };
        desc.storage = storage;
        let attr_index = self.attributes.insert(desc);
        Ok(ElementAttributeKey(attr_index as u32))
    }

    /// Find an attribute by its public key.
    pub fn find(&self, key: &AttributeKey) -> Option<ElementAttributeKey> {
        self.attributes
            .iter()
            .find(|(_, d)| &d.key == key)
            .map(|(i, _)| ElementAttributeKey(i as u32))
    }

    pub fn descriptor(&self, key: ElementAttributeKey) -> Option<&ElementAttributeDesc> {
        self.attributes.get(key.as_usize())
    }

    pub fn iter_descriptors(
        &self,
    ) -> impl Iterator<Item = (ElementAttributeKey, &ElementAttributeDesc)> {
        self.attributes
            .iter()
            .map(|(i, d)| (ElementAttributeKey(i as u32), d))
    }

    /// True if no attributes have been registered on this domain. Used by
    /// mid-level ops to skip the snapshot+interp work when no per-element
    /// CD layers exist (the common case for the mesh-without-attribs
    /// workflow).
    pub fn is_empty(&self) -> bool {
        self.attributes.is_empty()
    }

    // ---- Typed per-slot accessors. The element-slab key supplies the slot
    // index; the descriptor supplies the bucket. None when the descriptor's
    // bucket doesn't match the accessor.

    pub fn get_f32(&self, attr: ElementAttributeKey, slot: usize) -> Option<f32> {
        match self.attributes.get(attr.as_usize())?.storage {
            AttributeStorage::F32(b) => self
                .attributes_f32
                .get(b)
                .and_then(|v| v.get(slot).copied()),
            _ => None,
        }
    }

    pub fn set_f32(&mut self, attr: ElementAttributeKey, slot: usize, value: f32) -> bool {
        let storage = match self.attributes.get(attr.as_usize()) {
            Some(d) => d.storage,
            None => return false,
        };
        match storage {
            AttributeStorage::F32(b) => match self.attributes_f32.get_mut(b) {
                Some(v) if slot < v.len() => {
                    v[slot] = value;
                    true
                }
                _ => false,
            },
            _ => false,
        }
    }

    pub fn get_i32(&self, attr: ElementAttributeKey, slot: usize) -> Option<i32> {
        match self.attributes.get(attr.as_usize())?.storage {
            AttributeStorage::I32(b) => self
                .attributes_i32
                .get(b)
                .and_then(|v| v.get(slot).copied()),
            _ => None,
        }
    }

    pub fn set_i32(&mut self, attr: ElementAttributeKey, slot: usize, value: i32) -> bool {
        let storage = match self.attributes.get(attr.as_usize()) {
            Some(d) => d.storage,
            None => return false,
        };
        match storage {
            AttributeStorage::I32(b) => match self.attributes_i32.get_mut(b) {
                Some(v) if slot < v.len() => {
                    v[slot] = value;
                    true
                }
                _ => false,
            },
            _ => false,
        }
    }

    pub fn get_vec2(&self, attr: ElementAttributeKey, slot: usize) -> Option<[f32; 2]> {
        match self.attributes.get(attr.as_usize())?.storage {
            AttributeStorage::Vec2(b) => self
                .attributes_vec2
                .get(b)
                .and_then(|v| v.get(slot).copied()),
            _ => None,
        }
    }

    pub fn set_vec2(&mut self, attr: ElementAttributeKey, slot: usize, value: [f32; 2]) -> bool {
        let storage = match self.attributes.get(attr.as_usize()) {
            Some(d) => d.storage,
            None => return false,
        };
        match storage {
            AttributeStorage::Vec2(b) => match self.attributes_vec2.get_mut(b) {
                Some(v) if slot < v.len() => {
                    v[slot] = value;
                    true
                }
                _ => false,
            },
            _ => false,
        }
    }

    pub fn get_vec3(&self, attr: ElementAttributeKey, slot: usize) -> Option<[f32; 3]> {
        match self.attributes.get(attr.as_usize())?.storage {
            AttributeStorage::Vec3(b) => self
                .attributes_vec3
                .get(b)
                .and_then(|v| v.get(slot).copied()),
            _ => None,
        }
    }

    pub fn set_vec3(&mut self, attr: ElementAttributeKey, slot: usize, value: [f32; 3]) -> bool {
        let storage = match self.attributes.get(attr.as_usize()) {
            Some(d) => d.storage,
            None => return false,
        };
        match storage {
            AttributeStorage::Vec3(b) => match self.attributes_vec3.get_mut(b) {
                Some(v) if slot < v.len() => {
                    v[slot] = value;
                    true
                }
                _ => false,
            },
            _ => false,
        }
    }

    pub fn get_vec4(&self, attr: ElementAttributeKey, slot: usize) -> Option<[f32; 4]> {
        match self.attributes.get(attr.as_usize())?.storage {
            AttributeStorage::Vec4(b) => self
                .attributes_vec4
                .get(b)
                .and_then(|v| v.get(slot).copied()),
            _ => None,
        }
    }

    pub fn set_vec4(&mut self, attr: ElementAttributeKey, slot: usize, value: [f32; 4]) -> bool {
        let storage = match self.attributes.get(attr.as_usize()) {
            Some(d) => d.storage,
            None => return false,
        };
        match storage {
            AttributeStorage::Vec4(b) => match self.attributes_vec4.get_mut(b) {
                Some(v) if slot < v.len() => {
                    v[slot] = value;
                    true
                }
                _ => false,
            },
            _ => false,
        }
    }

    pub fn get_string(&self, attr: ElementAttributeKey, slot: usize) -> Option<&str> {
        match self.attributes.get(attr.as_usize())?.storage {
            AttributeStorage::String(b) => self
                .attributes_string
                .get(b)
                .and_then(|v| v.get(slot))
                .and_then(|opt| opt.as_deref()),
            _ => None,
        }
    }

    pub fn set_string(
        &mut self,
        attr: ElementAttributeKey,
        slot: usize,
        value: Option<String>,
    ) -> bool {
        let storage = match self.attributes.get(attr.as_usize()) {
            Some(d) => d.storage,
            None => return false,
        };
        match storage {
            AttributeStorage::String(b) => match self.attributes_string.get_mut(b) {
                Some(v) if slot < v.len() => {
                    v[slot] = value;
                    true
                }
                _ => false,
            },
            _ => false,
        }
    }

    pub(crate) fn storage_of(&self, attr: ElementAttributeKey) -> Option<AttributeStorage> {
        self.attributes.get(attr.as_usize()).map(|d| d.storage)
    }

    pub(crate) fn vec3_bucket(&self, bucket: usize) -> Option<&[[f32; 3]]> {
        self.attributes_vec3.get(bucket).map(|v| v.as_slice())
    }

    pub(crate) fn vec2_bucket(&self, bucket: usize) -> Option<&[[f32; 2]]> {
        self.attributes_vec2.get(bucket).map(|v| v.as_slice())
    }

    pub(crate) fn vec4_bucket(&self, bucket: usize) -> Option<&[[f32; 4]]> {
        self.attributes_vec4.get(bucket).map(|v| v.as_slice())
    }

    pub(crate) fn vec3_bucket_mut(&mut self, bucket: usize) -> Option<&mut [[f32; 3]]> {
        self.attributes_vec3
            .get_mut(bucket)
            .map(|v| v.as_mut_slice())
    }

    pub(crate) fn vec2_bucket_mut(&mut self, bucket: usize) -> Option<&mut [[f32; 2]]> {
        self.attributes_vec2
            .get_mut(bucket)
            .map(|v| v.as_mut_slice())
    }

    pub(crate) fn vec4_bucket_mut(&mut self, bucket: usize) -> Option<&mut [[f32; 4]]> {
        self.attributes_vec4
            .get_mut(bucket)
            .map(|v| v.as_mut_slice())
    }

    pub(crate) fn f32_bucket(&self, bucket: usize) -> Option<&[f32]> {
        self.attributes_f32.get(bucket).map(|v| v.as_slice())
    }

    pub(crate) fn u8_bucket(&self, bucket: usize) -> Option<&[u8]> {
        self.attributes_u8.get(bucket).map(|v| v.as_slice())
    }

    pub(crate) fn i32_bucket(&self, bucket: usize) -> Option<&[i32]> {
        self.attributes_i32.get(bucket).map(|v| v.as_slice())
    }

    pub(crate) fn u32_bucket(&self, bucket: usize) -> Option<&[u32]> {
        self.attributes_u32.get(bucket).map(|v| v.as_slice())
    }

    pub(crate) fn string_bucket(&self, bucket: usize) -> Option<&[Option<String>]> {
        self.attributes_string.get(bucket).map(|v| v.as_slice())
    }

    /// Serialise every attribute's value at `slot` into `out` as a stable
    /// byte sequence. Used to dedup "logically identical" per-corner
    /// attribute tuples during GPU export (seam-split). The exact bytes
    /// don't matter as long as two slots with equal attribute values
    /// produce equal sequences, and unequal values produce unequal
    /// sequences.
    ///
    /// Float values are serialised via `to_le_bytes` (so bitwise-equal
    /// floats — including NaN — fingerprint equal). Strings include a
    /// terminator and a Some/None discriminant.
    pub fn write_fingerprint(&self, slot: usize, out: &mut Vec<u8>) {
        for (_, desc) in self.iter_descriptors() {
            match desc.storage {
                AttributeStorage::U8(b) => {
                    if let Some(v) = self.u8_bucket(b).and_then(|s| s.get(slot)) {
                        out.push(*v);
                    }
                }
                AttributeStorage::I32(b) => {
                    if let Some(v) = self.i32_bucket(b).and_then(|s| s.get(slot)) {
                        out.extend_from_slice(&v.to_le_bytes());
                    }
                }
                AttributeStorage::U32(b) => {
                    if let Some(v) = self.u32_bucket(b).and_then(|s| s.get(slot)) {
                        out.extend_from_slice(&v.to_le_bytes());
                    }
                }
                AttributeStorage::F32(b) => {
                    if let Some(v) = self.f32_bucket(b).and_then(|s| s.get(slot)) {
                        out.extend_from_slice(&v.to_le_bytes());
                    }
                }
                AttributeStorage::Vec2(b) => {
                    if let Some(v) = self.vec2_bucket(b).and_then(|s| s.get(slot)) {
                        out.extend_from_slice(&v[0].to_le_bytes());
                        out.extend_from_slice(&v[1].to_le_bytes());
                    }
                }
                AttributeStorage::Vec3(b) => {
                    if let Some(v) = self.vec3_bucket(b).and_then(|s| s.get(slot)) {
                        out.extend_from_slice(&v[0].to_le_bytes());
                        out.extend_from_slice(&v[1].to_le_bytes());
                        out.extend_from_slice(&v[2].to_le_bytes());
                    }
                }
                AttributeStorage::Vec4(b) => {
                    if let Some(v) = self.vec4_bucket(b).and_then(|s| s.get(slot)) {
                        for lane in v {
                            out.extend_from_slice(&lane.to_le_bytes());
                        }
                    }
                }
                AttributeStorage::String(b) => {
                    match self.string_bucket(b).and_then(|s| s.get(slot)) {
                        Some(Some(s)) => {
                            out.push(1);
                            out.extend_from_slice(s.as_bytes());
                            out.push(0);
                        }
                        Some(None) | None => out.push(0),
                    }
                }
            }
        }
    }

    /// Tolerant per-layer equality of two element slots, used by the corner
    /// customdata merge (see `INSET.md` §6.3 step-5 note). Unlike
    /// [`write_fingerprint`](Self::write_fingerprint), which is a byte-exact
    /// serialisation, this compares each layer under its *own* equality
    /// predicate:
    ///
    /// - **Math (float) layers** — descriptors whose interp policy is the
    ///   fractional [`Linear`](InterpPolicy::Linear) blend — compare by
    ///   **squared distance across components**, treating the two slots as
    ///   equal when that squared distance is below `tolerance_sq`. Two
    ///   independently-rebaked resamples that the operator should treat as
    ///   "the same value" differ only in their low float bits, so a byte
    ///   compare would spuriously call them unequal; the tolerant compare is
    ///   what the merge gate and the propagation value-match both need.
    /// - **Every other layer** (nearest-policy integers, strings, copy-first)
    ///   compares **exactly** — a nearest-policy integer has no intermediate
    ///   value, so its equality is bitwise.
    ///
    /// Returns `true` only if *all* layers agree under their respective
    /// predicate.
    pub fn slots_tolerant_equal(&self, a: usize, b: usize, tolerance_sq: f32) -> bool {
        for (_, desc) in self.iter_descriptors() {
            let is_math = matches!(desc.interp, InterpPolicy::Linear);
            let equal = match desc.storage {
                AttributeStorage::F32(bucket) if is_math => {
                    let s = self.f32_bucket(bucket);
                    match (s.and_then(|v| v.get(a)), s.and_then(|v| v.get(b))) {
                        (Some(va), Some(vb)) => {
                            let d = va - vb;
                            d * d <= tolerance_sq
                        }
                        (None, None) => true,
                        _ => false,
                    }
                }
                AttributeStorage::Vec2(bucket) if is_math => {
                    self.vec_lanes_within(self.vec2_bucket(bucket), a, b, tolerance_sq)
                }
                AttributeStorage::Vec3(bucket) if is_math => {
                    self.vec_lanes_within(self.vec3_bucket(bucket), a, b, tolerance_sq)
                }
                AttributeStorage::Vec4(bucket) if is_math => {
                    self.vec_lanes_within(self.vec4_bucket(bucket), a, b, tolerance_sq)
                }
                _ => {
                    // Non-math layer: exact per-layer compare (no allocation).
                    self.slot_layer_exact_equal(desc, a, b)
                }
            };
            if !equal {
                return false;
            }
        }
        true
    }

    fn vec_lanes_within<const N: usize>(
        &self,
        bucket: Option<&[[f32; N]]>,
        a: usize,
        b: usize,
        tolerance_sq: f32,
    ) -> bool {
        match (bucket.and_then(|s| s.get(a)), bucket.and_then(|s| s.get(b))) {
            (Some(va), Some(vb)) => {
                let mut acc = 0.0_f32;
                for lane in 0..N {
                    let d = va[lane] - vb[lane];
                    acc += d * d;
                }
                acc <= tolerance_sq
            }
            (None, None) => true,
            _ => false,
        }
    }

    /// Exact (bitwise for floats, value-wise otherwise) equality of one
    /// attribute layer at two slots, allocation-free. Used by the non-math
    /// branch of [`slots_tolerant_equal`]. A missing value on both slots
    /// counts as equal; on only one, unequal.
    fn slot_layer_exact_equal(&self, desc: &ElementAttributeDesc, a: usize, b: usize) -> bool {
        fn pair_eq<T: PartialEq>(bucket: Option<&[T]>, a: usize, b: usize) -> bool {
            match (bucket.and_then(|s| s.get(a)), bucket.and_then(|s| s.get(b))) {
                (Some(va), Some(vb)) => va == vb,
                (None, None) => true,
                _ => false,
            }
        }
        match desc.storage {
            AttributeStorage::U8(bk) => pair_eq(self.u8_bucket(bk), a, b),
            AttributeStorage::I32(bk) => pair_eq(self.i32_bucket(bk), a, b),
            AttributeStorage::U32(bk) => pair_eq(self.u32_bucket(bk), a, b),
            AttributeStorage::F32(bk) => pair_eq(self.f32_bucket(bk), a, b),
            AttributeStorage::Vec2(bk) => pair_eq(self.vec2_bucket(bk), a, b),
            AttributeStorage::Vec3(bk) => pair_eq(self.vec3_bucket(bk), a, b),
            AttributeStorage::Vec4(bk) => pair_eq(self.vec4_bucket(bk), a, b),
            AttributeStorage::String(bk) => pair_eq(self.string_bucket(bk), a, b),
        }
    }

    // ----- Customdata interpolation primitives -----

    /// Interpolate every attribute into `dst_slot` from a weighted blend of
    /// source slots. The per-attribute behaviour is governed by each
    /// descriptor's [`InterpPolicy`]:
    ///
    /// - [`Linear`](InterpPolicy::Linear) — weighted sum, normalised by the
    ///   total weight. Zero total weight leaves the destination unchanged.
    /// - [`Nearest`](InterpPolicy::Nearest) — copy from the source with the
    ///   largest weight (first wins on ties).
    /// - [`CopyFirst`](InterpPolicy::CopyFirst) — copy from `sources[0]`.
    /// - [`NoInterp`](InterpPolicy::NoInterp) — leave destination untouched.
    ///
    /// Empty `sources` is a no-op.
    pub fn interpolate_into(&mut self, dst_slot: usize, sources: &[(f32, usize)]) {
        if sources.is_empty() {
            return;
        }

        // Walk descriptors. We need access to the typed buckets while
        // iterating; collect descriptor info into a small Vec first to
        // release the immutable borrow.
        let plan: SmallVec<[(AttributeStorage, InterpPolicy); 8]> = self
            .attributes
            .iter()
            .map(|(_, d)| (d.storage, d.interp))
            .collect();

        for (storage, interp) in plan {
            if matches!(interp, InterpPolicy::NoInterp) {
                continue;
            }
            apply_interp(self, dst_slot, sources, storage, interp);
        }
    }

    /// Capture every attribute layer's values at `slots` into a detached
    /// [`SlotSnapshot`]. The snapshot is independent of the live buckets, so
    /// it survives later writes to those same slots — the prerequisite for an
    /// in-place resample where a destination slot is also one of its own
    /// sources (inset's `use_interpolate` rebake reads a face's *original*
    /// corner values to recompute the *same* corners' values; see
    /// `radial-mesh-guides/operators/INSET.md` §4.5/§7).
    pub fn snapshot_slots(&self, slots: &[usize]) -> SlotSnapshot {
        let mut layers = Vec::with_capacity(self.attributes.len());
        for (_, desc) in self.attributes.iter() {
            let values = match desc.storage {
                AttributeStorage::U8(b) => SnapshotValues::U8(gather(&self.attributes_u8[b], slots)),
                AttributeStorage::I32(b) => {
                    SnapshotValues::I32(gather(&self.attributes_i32[b], slots))
                }
                AttributeStorage::U32(b) => {
                    SnapshotValues::U32(gather(&self.attributes_u32[b], slots))
                }
                AttributeStorage::F32(b) => {
                    SnapshotValues::F32(gather(&self.attributes_f32[b], slots))
                }
                AttributeStorage::Vec2(b) => {
                    SnapshotValues::Vec2(gather(&self.attributes_vec2[b], slots))
                }
                AttributeStorage::Vec3(b) => {
                    SnapshotValues::Vec3(gather(&self.attributes_vec3[b], slots))
                }
                AttributeStorage::Vec4(b) => {
                    SnapshotValues::Vec4(gather(&self.attributes_vec4[b], slots))
                }
                AttributeStorage::String(b) => SnapshotValues::String(
                    slots
                        .iter()
                        .map(|&s| self.attributes_string[b].get(s).cloned().flatten())
                        .collect(),
                ),
            };
            layers.push((desc.storage, desc.interp, values));
        }
        SlotSnapshot { layers }
    }

    /// Blend a [`SlotSnapshot`]'s captured corners into `dst_slot` by
    /// `weights` (one weight per snapshot index, in the same order the slots
    /// were captured). Each layer composes the weights under its own
    /// [`InterpPolicy`] — the fixed-arity blend-descriptor handoff of
    /// `radial-mesh-guides/BMESH.md` §10.1.4. The weights are taken as already
    /// normalised to a partition of unity; `Linear` layers re-normalise by
    /// their own weight total for robustness.
    pub fn interpolate_from_snapshot(
        &mut self,
        dst_slot: usize,
        snapshot: &SlotSnapshot,
        weights: &[f32],
    ) {
        for (storage, interp, values) in &snapshot.layers {
            if matches!(interp, InterpPolicy::NoInterp) {
                continue;
            }
            apply_interp_from_snapshot(self, dst_slot, *storage, *interp, values, weights);
        }
    }

    /// 1:1 attribute copy: every attribute's `dst_slot` receives `src_slot`'s
    /// value. Useful for ops that duplicate an element without blending
    /// (e.g. `euler_vert_separate`, `face_split` inheriting the source
    /// face's attribute values).
    pub fn copy_into(&mut self, dst_slot: usize, src_slot: usize) {
        if dst_slot == src_slot {
            return;
        }
        let plan: SmallVec<[AttributeStorage; 8]> =
            self.attributes.iter().map(|(_, d)| d.storage).collect();

        for storage in plan {
            apply_copy(self, dst_slot, src_slot, storage);
        }
    }

    /// Reset every attribute's `slot` to its default-init value. Used when
    /// recycling a slab slot whose previous attribute values are no longer
    /// meaningful.
    pub fn clear_slot(&mut self, slot: usize) {
        let plan: SmallVec<[AttributeStorage; 8]> =
            self.attributes.iter().map(|(_, d)| d.storage).collect();

        for storage in plan {
            match storage {
                AttributeStorage::U8(b) => {
                    if let Some(v) = self.attributes_u8.get_mut(b) {
                        if slot < v.len() {
                            v[slot] = 0;
                        }
                    }
                }
                AttributeStorage::I32(b) => {
                    if let Some(v) = self.attributes_i32.get_mut(b) {
                        if slot < v.len() {
                            v[slot] = 0;
                        }
                    }
                }
                AttributeStorage::U32(b) => {
                    if let Some(v) = self.attributes_u32.get_mut(b) {
                        if slot < v.len() {
                            v[slot] = 0;
                        }
                    }
                }
                AttributeStorage::F32(b) => {
                    if let Some(v) = self.attributes_f32.get_mut(b) {
                        if slot < v.len() {
                            v[slot] = 0.0;
                        }
                    }
                }
                AttributeStorage::Vec2(b) => {
                    if let Some(v) = self.attributes_vec2.get_mut(b) {
                        if slot < v.len() {
                            v[slot] = [0.0; 2];
                        }
                    }
                }
                AttributeStorage::Vec3(b) => {
                    if let Some(v) = self.attributes_vec3.get_mut(b) {
                        if slot < v.len() {
                            v[slot] = [0.0; 3];
                        }
                    }
                }
                AttributeStorage::Vec4(b) => {
                    if let Some(v) = self.attributes_vec4.get_mut(b) {
                        if slot < v.len() {
                            v[slot] = [0.0; 4];
                        }
                    }
                }
                AttributeStorage::String(b) => {
                    if let Some(v) = self.attributes_string.get_mut(b) {
                        if slot < v.len() {
                            v[slot] = None;
                        }
                    }
                }
            }
        }
    }
}

/// A detached copy of every attribute layer's values at an ordered list of
/// slots, captured by [`ElementAttributes::snapshot_slots`]. Used as a
/// resample source that is immune to in-place writes on the slots it was
/// captured from.
pub struct SlotSnapshot {
    layers: Vec<(AttributeStorage, InterpPolicy, SnapshotValues)>,
}

impl SlotSnapshot {
    /// Number of captured slots (snapshot corners).
    pub fn len(&self) -> usize {
        self.layers.first().map_or(0, |(_, _, v)| v.len())
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

enum SnapshotValues {
    U8(Vec<u8>),
    I32(Vec<i32>),
    U32(Vec<u32>),
    F32(Vec<f32>),
    Vec2(Vec<[f32; 2]>),
    Vec3(Vec<[f32; 3]>),
    Vec4(Vec<[f32; 4]>),
    String(Vec<Option<String>>),
}

impl SnapshotValues {
    fn len(&self) -> usize {
        match self {
            SnapshotValues::U8(v) => v.len(),
            SnapshotValues::I32(v) => v.len(),
            SnapshotValues::U32(v) => v.len(),
            SnapshotValues::F32(v) => v.len(),
            SnapshotValues::Vec2(v) => v.len(),
            SnapshotValues::Vec3(v) => v.len(),
            SnapshotValues::Vec4(v) => v.len(),
            SnapshotValues::String(v) => v.len(),
        }
    }
}

/// Gather `slots` out of a bucket into a dense Vec (default-init for any
/// out-of-range slot, which should not occur for live element slots).
fn gather<T: Copy + Default>(bucket: &[T], slots: &[usize]) -> Vec<T> {
    slots
        .iter()
        .map(|&s| bucket.get(s).copied().unwrap_or_default())
        .collect()
}

/// Index of the snapshot corner carrying the largest weight (first wins on
/// ties); `None` when every weight is non-positive and no source qualifies.
fn nearest_snapshot_index(weights: &[f32]) -> Option<usize> {
    let mut best: Option<(f32, usize)> = None;
    for (i, &w) in weights.iter().enumerate() {
        if best.map_or(true, |(bw, _)| w > bw) {
            best = Some((w, i));
        }
    }
    best.map(|(_, i)| i)
}

/// Snapshot-source analogue of [`apply_interp`]: blend the captured corner
/// values by `weights` under `policy` and write the result to `dst`.
fn apply_interp_from_snapshot(
    attrs: &mut ElementAttributes,
    dst: usize,
    storage: AttributeStorage,
    policy: InterpPolicy,
    values: &SnapshotValues,
    weights: &[f32],
) {
    macro_rules! linear_vecn {
        ($bucket:expr, $src:expr, $lanes:expr, $zero:expr) => {{
            if let Some(out) = $bucket.get_mut(dst) {
                match policy {
                    InterpPolicy::Linear => {
                        let mut sum = $zero;
                        let mut wsum = 0.0_f32;
                        for (i, &w) in weights.iter().enumerate() {
                            if let Some(val) = $src.get(i) {
                                for lane in 0..$lanes {
                                    sum[lane] += w * val[lane];
                                }
                                wsum += w;
                            }
                        }
                        if wsum != 0.0 {
                            for lane in 0..$lanes {
                                sum[lane] /= wsum;
                            }
                            *out = sum;
                        }
                    }
                    InterpPolicy::Nearest | InterpPolicy::CopyFirst => {
                        let idx = match policy {
                            InterpPolicy::CopyFirst => 0,
                            _ => match nearest_snapshot_index(weights) {
                                Some(i) => i,
                                None => return,
                            },
                        };
                        if let Some(val) = $src.get(idx) {
                            *out = *val;
                        }
                    }
                    InterpPolicy::NoInterp => {}
                }
            }
        }};
    }

    macro_rules! pick_int {
        ($bucket:expr, $src:expr) => {{
            if let Some(out) = $bucket.get_mut(dst) {
                let idx = match policy {
                    InterpPolicy::CopyFirst => 0,
                    InterpPolicy::NoInterp => return,
                    // Integer attributes never produce a fractional blend
                    // (§10.1.4); Linear routes to highest-weight like Nearest.
                    _ => match nearest_snapshot_index(weights) {
                        Some(i) => i,
                        None => return,
                    },
                };
                if let Some(val) = $src.get(idx) {
                    *out = *val;
                }
            }
        }};
    }

    match (storage, values) {
        (AttributeStorage::F32(b), SnapshotValues::F32(src)) => {
            if let Some(v) = attrs.attributes_f32.get_mut(b) {
                if let Some(out) = v.get_mut(dst) {
                    match policy {
                        InterpPolicy::Linear => {
                            let mut sum = 0.0_f32;
                            let mut wsum = 0.0_f32;
                            for (i, &w) in weights.iter().enumerate() {
                                if let Some(&val) = src.get(i) {
                                    sum += w * val;
                                    wsum += w;
                                }
                            }
                            if wsum != 0.0 {
                                *out = sum / wsum;
                            }
                        }
                        InterpPolicy::Nearest | InterpPolicy::CopyFirst => {
                            let idx = if matches!(policy, InterpPolicy::CopyFirst) {
                                0
                            } else {
                                match nearest_snapshot_index(weights) {
                                    Some(i) => i,
                                    None => return,
                                }
                            };
                            if let Some(&val) = src.get(idx) {
                                *out = val;
                            }
                        }
                        InterpPolicy::NoInterp => {}
                    }
                }
            }
        }
        (AttributeStorage::Vec2(b), SnapshotValues::Vec2(src)) => {
            if let Some(v) = attrs.attributes_vec2.get_mut(b) {
                linear_vecn!(v, src, 2, [0.0_f32; 2]);
            }
        }
        (AttributeStorage::Vec3(b), SnapshotValues::Vec3(src)) => {
            if let Some(v) = attrs.attributes_vec3.get_mut(b) {
                linear_vecn!(v, src, 3, [0.0_f32; 3]);
            }
        }
        (AttributeStorage::Vec4(b), SnapshotValues::Vec4(src)) => {
            if let Some(v) = attrs.attributes_vec4.get_mut(b) {
                linear_vecn!(v, src, 4, [0.0_f32; 4]);
            }
        }
        (AttributeStorage::U8(b), SnapshotValues::U8(src)) => {
            if let Some(v) = attrs.attributes_u8.get_mut(b) {
                pick_int!(v, src);
            }
        }
        (AttributeStorage::I32(b), SnapshotValues::I32(src)) => {
            if let Some(v) = attrs.attributes_i32.get_mut(b) {
                pick_int!(v, src);
            }
        }
        (AttributeStorage::U32(b), SnapshotValues::U32(src)) => {
            if let Some(v) = attrs.attributes_u32.get_mut(b) {
                pick_int!(v, src);
            }
        }
        (AttributeStorage::String(b), SnapshotValues::String(src)) => {
            if let Some(v) = attrs.attributes_string.get_mut(b) {
                if let Some(out) = v.get_mut(dst) {
                    let idx = match policy {
                        InterpPolicy::CopyFirst | InterpPolicy::Linear => 0,
                        InterpPolicy::Nearest => match nearest_snapshot_index(weights) {
                            Some(i) => i,
                            None => return,
                        },
                        InterpPolicy::NoInterp => return,
                    };
                    *out = src.get(idx).cloned().flatten();
                }
            }
        }
        _ => {}
    }
}

/// Per-attribute dispatch for [`ElementAttributes::interpolate_into`].
fn apply_interp(
    attrs: &mut ElementAttributes,
    dst: usize,
    sources: &[(f32, usize)],
    storage: AttributeStorage,
    policy: InterpPolicy,
) {
    match storage {
        AttributeStorage::F32(b) => {
            if let Some(v) = attrs.attributes_f32.get_mut(b) {
                if dst >= v.len() {
                    return;
                }
                match policy {
                    InterpPolicy::Linear => {
                        let mut sum = 0.0_f32;
                        let mut wsum = 0.0_f32;
                        for &(w, s) in sources {
                            if let Some(&val) = v.get(s) {
                                sum += w * val;
                                wsum += w;
                            }
                        }
                        if wsum != 0.0 {
                            v[dst] = sum / wsum;
                        }
                    }
                    InterpPolicy::Nearest => {
                        if let Some(s) = nearest_source(sources, v.len()) {
                            v[dst] = v[s];
                        }
                    }
                    InterpPolicy::CopyFirst => {
                        let s = sources[0].1;
                        if s < v.len() {
                            v[dst] = v[s];
                        }
                    }
                    InterpPolicy::NoInterp => {}
                }
            }
        }
        AttributeStorage::Vec2(b) => {
            if let Some(v) = attrs.attributes_vec2.get_mut(b) {
                if dst >= v.len() {
                    return;
                }
                match policy {
                    InterpPolicy::Linear => {
                        let mut sum = [0.0_f32; 2];
                        let mut wsum = 0.0_f32;
                        for &(w, s) in sources {
                            if let Some(&val) = v.get(s) {
                                sum[0] += w * val[0];
                                sum[1] += w * val[1];
                                wsum += w;
                            }
                        }
                        if wsum != 0.0 {
                            v[dst] = [sum[0] / wsum, sum[1] / wsum];
                        }
                    }
                    InterpPolicy::Nearest => {
                        if let Some(s) = nearest_source(sources, v.len()) {
                            v[dst] = v[s];
                        }
                    }
                    InterpPolicy::CopyFirst => {
                        let s = sources[0].1;
                        if s < v.len() {
                            v[dst] = v[s];
                        }
                    }
                    InterpPolicy::NoInterp => {}
                }
            }
        }
        AttributeStorage::Vec3(b) => {
            if let Some(v) = attrs.attributes_vec3.get_mut(b) {
                if dst >= v.len() {
                    return;
                }
                match policy {
                    InterpPolicy::Linear => {
                        let mut sum = [0.0_f32; 3];
                        let mut wsum = 0.0_f32;
                        for &(w, s) in sources {
                            if let Some(&val) = v.get(s) {
                                sum[0] += w * val[0];
                                sum[1] += w * val[1];
                                sum[2] += w * val[2];
                                wsum += w;
                            }
                        }
                        if wsum != 0.0 {
                            v[dst] = [sum[0] / wsum, sum[1] / wsum, sum[2] / wsum];
                        }
                    }
                    InterpPolicy::Nearest => {
                        if let Some(s) = nearest_source(sources, v.len()) {
                            v[dst] = v[s];
                        }
                    }
                    InterpPolicy::CopyFirst => {
                        let s = sources[0].1;
                        if s < v.len() {
                            v[dst] = v[s];
                        }
                    }
                    InterpPolicy::NoInterp => {}
                }
            }
        }
        AttributeStorage::Vec4(b) => {
            if let Some(v) = attrs.attributes_vec4.get_mut(b) {
                if dst >= v.len() {
                    return;
                }
                match policy {
                    InterpPolicy::Linear => {
                        let mut sum = [0.0_f32; 4];
                        let mut wsum = 0.0_f32;
                        for &(w, s) in sources {
                            if let Some(&val) = v.get(s) {
                                sum[0] += w * val[0];
                                sum[1] += w * val[1];
                                sum[2] += w * val[2];
                                sum[3] += w * val[3];
                                wsum += w;
                            }
                        }
                        if wsum != 0.0 {
                            v[dst] = [sum[0] / wsum, sum[1] / wsum, sum[2] / wsum, sum[3] / wsum];
                        }
                    }
                    InterpPolicy::Nearest => {
                        if let Some(s) = nearest_source(sources, v.len()) {
                            v[dst] = v[s];
                        }
                    }
                    InterpPolicy::CopyFirst => {
                        let s = sources[0].1;
                        if s < v.len() {
                            v[dst] = v[s];
                        }
                    }
                    InterpPolicy::NoInterp => {}
                }
            }
        }
        AttributeStorage::U8(b) => {
            if let Some(v) = attrs.attributes_u8.get_mut(b) {
                interp_integer(v, dst, sources, policy, |a, b| a.wrapping_add(b), 0);
            }
        }
        AttributeStorage::I32(b) => {
            if let Some(v) = attrs.attributes_i32.get_mut(b) {
                interp_integer(v, dst, sources, policy, |a, b| a.wrapping_add(b), 0);
            }
        }
        AttributeStorage::U32(b) => {
            if let Some(v) = attrs.attributes_u32.get_mut(b) {
                interp_integer(v, dst, sources, policy, |a, b| a.wrapping_add(b), 0);
            }
        }
        AttributeStorage::String(b) => {
            if let Some(v) = attrs.attributes_string.get_mut(b) {
                if dst >= v.len() {
                    return;
                }
                match policy {
                    // For strings only CopyFirst / Nearest / NoInterp make
                    // sense; Linear collapses to "use the highest-weight"
                    // (no defensible "weighted blend" exists for strings).
                    InterpPolicy::CopyFirst | InterpPolicy::Linear => {
                        let s = sources[0].1;
                        v[dst] = v.get(s).cloned().unwrap_or(None);
                    }
                    InterpPolicy::Nearest => {
                        if let Some(s) = nearest_source(sources, v.len()) {
                            v[dst] = v[s].clone();
                        }
                    }
                    InterpPolicy::NoInterp => {}
                }
            }
        }
    }
}

/// Per-attribute dispatch for [`ElementAttributes::copy_into`].
fn apply_copy(attrs: &mut ElementAttributes, dst: usize, src: usize, storage: AttributeStorage) {
    match storage {
        AttributeStorage::F32(b) => {
            if let Some(v) = attrs.attributes_f32.get_mut(b) {
                if src < v.len() && dst < v.len() {
                    v[dst] = v[src];
                }
            }
        }
        AttributeStorage::Vec2(b) => {
            if let Some(v) = attrs.attributes_vec2.get_mut(b) {
                if src < v.len() && dst < v.len() {
                    v[dst] = v[src];
                }
            }
        }
        AttributeStorage::Vec3(b) => {
            if let Some(v) = attrs.attributes_vec3.get_mut(b) {
                if src < v.len() && dst < v.len() {
                    v[dst] = v[src];
                }
            }
        }
        AttributeStorage::Vec4(b) => {
            if let Some(v) = attrs.attributes_vec4.get_mut(b) {
                if src < v.len() && dst < v.len() {
                    v[dst] = v[src];
                }
            }
        }
        AttributeStorage::U8(b) => {
            if let Some(v) = attrs.attributes_u8.get_mut(b) {
                if src < v.len() && dst < v.len() {
                    v[dst] = v[src];
                }
            }
        }
        AttributeStorage::I32(b) => {
            if let Some(v) = attrs.attributes_i32.get_mut(b) {
                if src < v.len() && dst < v.len() {
                    v[dst] = v[src];
                }
            }
        }
        AttributeStorage::U32(b) => {
            if let Some(v) = attrs.attributes_u32.get_mut(b) {
                if src < v.len() && dst < v.len() {
                    v[dst] = v[src];
                }
            }
        }
        AttributeStorage::String(b) => {
            if let Some(v) = attrs.attributes_string.get_mut(b) {
                if src < v.len() && dst < v.len() {
                    v[dst] = v[src].clone();
                }
            }
        }
    }
}

/// Pick the source slot with the largest weight; ties go to the
/// first-encountered. Returns `None` if no source is in-range.
fn nearest_source(sources: &[(f32, usize)], len: usize) -> Option<usize> {
    let mut best: Option<(f32, usize)> = None;
    for &(w, s) in sources {
        if s >= len {
            continue;
        }
        if best.map_or(true, |(bw, _)| w > bw) {
            best = Some((w, s));
        }
    }
    best.map(|(_, s)| s)
}

/// Linear/Nearest/CopyFirst dispatch for integer buckets where "Linear"
/// degenerates to "weighted sum, rounded by integer truncation". `acc`
/// adds source contributions; `zero` is the additive identity. Used for
/// `u8`, `i32`, `u32`.
fn interp_integer<T: Copy>(
    v: &mut [T],
    dst: usize,
    sources: &[(f32, usize)],
    policy: InterpPolicy,
    _acc: impl Fn(T, T) -> T,
    _zero: T,
) where
    T: From<u8>,
{
    if dst >= v.len() {
        return;
    }
    match policy {
        InterpPolicy::Nearest | InterpPolicy::Linear => {
            // Per default, integer attributes are Nearest. We also route
            // Linear here so callers that explicitly set Linear on an int
            // attribute get a defensible behaviour (highest-weight wins).
            if let Some(s) = nearest_source(sources, v.len()) {
                v[dst] = v[s];
            }
        }
        InterpPolicy::CopyFirst => {
            let s = sources[0].1;
            if s < v.len() {
                v[dst] = v[s];
            }
        }
        InterpPolicy::NoInterp => {}
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

pub trait IndexType {
    fn as_usize(&self) -> usize;
}
impl IndexType for u8 {
    fn as_usize(&self) -> usize {
        *self as usize
    }
}
impl IndexType for u16 {
    fn as_usize(&self) -> usize {
        *self as usize
    }
}
impl IndexType for u32 {
    fn as_usize(&self) -> usize {
        *self as usize
    }
}

pub struct IndexIterator<'a, T>
where
    T: IndexType + Copy,
{
    indices: Option<&'a [T]>,
    idx: usize,
    end: usize,
}

impl<'a, T> Iterator for IndexIterator<'a, T>
where
    T: IndexType + Copy,
{
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(indices) = self.indices {
            let i = self.idx;
            if i < self.end {
                self.idx += 1;
                Some(indices[i].as_usize())
            } else {
                None
            }
        } else {
            let i = self.idx;
            if i < self.end {
                self.idx += 1;
                Some(i)
            } else {
                None
            }
        }
    }
}

pub fn indices_iter<T>(indices: Option<&[T]>, start: usize, count: usize) -> IndexIterator<'_, T>
where
    T: IndexType + Copy,
{
    IndexIterator {
        indices,
        idx: start,
        end: start + count,
    }
}

pub struct TriangleListIterator<I> {
    iter: I,
}

impl<I, T> Iterator for TriangleListIterator<I>
where
    //T: IndexPrimitive,
    I: Iterator<Item = T>,
{
    type Item = [T; 3];

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(v0) = self.iter.next() {
            if let Some(v1) = self.iter.next() {
                self.iter.next().map(|v2| [v0, v1, v2])
            } else {
                None
            }
        } else {
            None
        }
    }
}

pub struct TriangleStripIterator<I, T>
where
    I: Iterator<Item = T>,
{
    v0: Option<T>,
    v1: Option<T>,
    iter: I,
}

impl<I, T> Iterator for TriangleStripIterator<I, T>
where
    T: Copy,
    I: Iterator<Item = T>,
{
    type Item = [T; 3];

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(v0) = self.v0 {
            if let Some(v1) = self.v1 {
                if let Some(v2) = self.iter.next() {
                    self.v0 = Some(v1);
                    self.v1 = Some(v2);
                    Some([v0, v1, v2])
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            None
        }
    }
}

pub struct TriangleFanIterator<I, T>
where
    I: Iterator<Item = T>,
{
    v0: Option<T>,
    v1: Option<T>,
    iter: I,
}

impl<I, T> Iterator for TriangleFanIterator<I, T>
where
    T: Copy,
    //T: IndexPrimitive,
    I: Iterator<Item = T>,
{
    type Item = [T; 3];

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(v0) = self.v0 {
            if let Some(v1) = self.v1 {
                if let Some(v2) = self.iter.next() {
                    self.v1 = Some(v2);
                    Some([v0, v1, v2])
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            None
        }
    }
}

pub enum PrimitiveTopology {
    TriangleList,
    TriangleStrip,
    TriangleFan,
}

pub struct PrimitiveIterator<I, T>
where
    I: Iterator<Item = T>,
{
    topology: PrimitiveTopology,
    v0: Option<T>,
    v1: Option<T>,
    iter: I,
}

impl<I, T> Iterator for PrimitiveIterator<I, T>
where
    T: Copy,
    I: Iterator<Item = T>,
{
    type Item = [T; 3];

    fn next(&mut self) -> Option<Self::Item> {
        match self.topology {
            PrimitiveTopology::TriangleList => {
                if let Some(v0) = self.v0 {
                    if let Some(v1) = self.v1 {
                        if let Some(v2) = self.iter.next() {
                            self.v0 = self.iter.next();
                            self.v1 = self.iter.next();
                            Some([v0, v1, v2])
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                } else {
                    None
                }
            }
            PrimitiveTopology::TriangleStrip => {
                if let Some(v0) = self.v0 {
                    if let Some(v1) = self.v1 {
                        if let Some(v2) = self.iter.next() {
                            self.v0 = Some(v1);
                            self.v1 = Some(v2);
                            Some([v0, v1, v2])
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                } else {
                    None
                }
            }
            PrimitiveTopology::TriangleFan => {
                if let Some(v0) = self.v0 {
                    if let Some(v1) = self.v1 {
                        if let Some(v2) = self.iter.next() {
                            self.v1 = Some(v2);
                            Some([v0, v1, v2])
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                } else {
                    None
                }
            }
        }
    }
}

pub trait PrimitiveIterators: Iterator {
    fn list_triangles(self) -> TriangleListIterator<Self>
    where
        Self: Sized,
    {
        TriangleListIterator { iter: self }
    }

    fn strip_triangles(mut self) -> TriangleStripIterator<Self, Self::Item>
    where
        Self: Sized,
        Self: Iterator,
    {
        TriangleStripIterator {
            v0: self.next(),
            v1: self.next(),
            iter: self,
        }
    }

    fn fan_triangles(mut self) -> TriangleFanIterator<Self, Self::Item>
    where
        Self: Sized,
        Self: Iterator,
    {
        TriangleFanIterator {
            v0: self.next(),
            v1: self.next(),
            iter: self,
        }
    }

    fn primitive_triangles(
        mut self,
        topology: PrimitiveTopology,
    ) -> PrimitiveIterator<Self, Self::Item>
    where
        Self: Sized,
        Self: Iterator,
    {
        PrimitiveIterator {
            topology,
            v0: self.next(),
            v1: self.next(),
            iter: self,
        }
    }
}

impl<T: ?Sized> PrimitiveIterators for T where T: Iterator {}

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

    pub fn register_vertex_attribute(
        &mut self,
        desc: ElementAttributeDesc,
    ) -> Result<ElementAttributeKey> {
        let watermark = self.vertices.key_watermark();
        self.vertex_attribs.insert_with_storage(desc, watermark)
    }

    pub fn register_edge_attribute(
        &mut self,
        desc: ElementAttributeDesc,
    ) -> Result<ElementAttributeKey> {
        let watermark = self.edges.key_watermark();
        self.edge_attribs.insert_with_storage(desc, watermark)
    }

    pub fn register_loop_attribute(
        &mut self,
        desc: ElementAttributeDesc,
    ) -> Result<ElementAttributeKey> {
        let watermark = self.loops.key_watermark();
        self.loop_attribs.insert_with_storage(desc, watermark)
    }

    pub fn register_face_attribute(
        &mut self,
        desc: ElementAttributeDesc,
    ) -> Result<ElementAttributeKey> {
        let watermark = self.faces.key_watermark();
        self.face_attribs.insert_with_storage(desc, watermark)
    }

    pub fn vertex_attribs(&self) -> &ElementAttributes {
        &self.vertex_attribs
    }
    pub fn edge_attribs(&self) -> &ElementAttributes {
        &self.edge_attribs
    }
    pub fn loop_attribs(&self) -> &ElementAttributes {
        &self.loop_attribs
    }
    pub fn face_attribs(&self) -> &ElementAttributes {
        &self.face_attribs
    }

    pub fn vertex_attribs_mut(&mut self) -> &mut ElementAttributes {
        &mut self.vertex_attribs
    }
    pub fn edge_attribs_mut(&mut self) -> &mut ElementAttributes {
        &mut self.edge_attribs
    }
    pub fn loop_attribs_mut(&mut self) -> &mut ElementAttributes {
        &mut self.loop_attribs
    }
    pub fn face_attribs_mut(&mut self) -> &mut ElementAttributes {
        &mut self.face_attribs
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

    pub fn vertices_ensure_index_field(&mut self) {
        if self.flags & LoopMeshFlags::VERTICES_INDEX_DIRTY == LoopMeshFlags::NONE {
            return;
        }
        // We're about to walk every live vertex; populate the table at the
        // same time to keep both forms cheap.
        self.vertices_index.clear();
        let mut it = MeshVerts::default();
        let mut i: u32 = 0;
        while let Some(v) = it.next(self) {
            self.vertices.get_mut(v).index = i;
            self.vertices_index.push(v);
            i += 1;
        }
        self.flags &= !LoopMeshFlags::VERTICES_INDEX_DIRTY;
    }

    pub fn edges_ensure_index_field(&mut self) {
        if self.flags & LoopMeshFlags::EDGES_INDEX_DIRTY == LoopMeshFlags::NONE {
            return;
        }
        self.edges_index.clear();
        let mut it = MeshEdges::default();
        let mut i: u32 = 0;
        while let Some(e) = it.next(self) {
            self.edges.get_mut(e).index = i;
            self.edges_index.push(e);
            i += 1;
        }
        self.flags &= !LoopMeshFlags::EDGES_INDEX_DIRTY;
    }

    pub fn faces_ensure_index_field(&mut self) {
        if self.flags & LoopMeshFlags::FACES_INDEX_DIRTY == LoopMeshFlags::NONE {
            return;
        }
        self.faces_index.clear();
        let mut it = MeshFaces::default();
        let mut i: u32 = 0;
        while let Some(f) = it.next(self) {
            self.faces.get_mut(f).index = i;
            self.faces_index.push(f);
            i += 1;
        }
        self.flags &= !LoopMeshFlags::FACES_INDEX_DIRTY;
    }

    pub fn loops_ensure_index_field(&mut self) {
        if self.flags & LoopMeshFlags::LOOPS_INDEX_DIRTY == LoopMeshFlags::NONE {
            return;
        }
        self.loops_index.clear();
        let mut it = MeshLoops::default();
        let mut i: u32 = 0;
        while let Some(l) = it.next(self) {
            self.loops.get_mut(l).index = i;
            self.loops_index.push(l);
            i += 1;
        }
        self.flags &= !LoopMeshFlags::LOOPS_INDEX_DIRTY;
    }

    /// Populate `vertices_index: Vec<VertexKey>`. Implies
    /// [`vertices_ensure_index_field`].
    pub fn vertices_ensure_table(&mut self) {
        self.vertices_ensure_index_field();
    }

    pub fn edges_ensure_table(&mut self) {
        self.edges_ensure_index_field();
    }

    pub fn faces_ensure_table(&mut self) {
        self.faces_ensure_index_field();
    }

    pub fn loops_ensure_table(&mut self) {
        self.loops_ensure_index_field();
    }

    /// Convenience: ensure indices and tables are fresh on every axis.
    pub fn vertices_ensure_index(&mut self) {
        self.vertices_ensure_index_field();
    }

    pub fn edges_ensure_index(&mut self) {
        self.edges_ensure_index_field();
    }

    pub fn faces_ensure_index(&mut self) {
        self.faces_ensure_index_field();
    }

    pub fn loops_ensure_index(&mut self) {
        self.loops_ensure_index_field();
    }

    pub fn all_indices_ensure(&mut self) {
        self.vertices_ensure_index_field();
        self.edges_ensure_index_field();
        self.faces_ensure_index_field();
        self.loops_ensure_index_field();
    }

    // ----- Dirty-flag observability -----
    //
    // Read-only queries that don't force a rebuild. Useful for asserts and
    // for algorithms that want to fail loudly rather than silently re-walk.

    pub fn is_vertices_index_dirty(&self) -> bool {
        self.flags & LoopMeshFlags::VERTICES_INDEX_DIRTY != LoopMeshFlags::NONE
    }

    pub fn is_edges_index_dirty(&self) -> bool {
        self.flags & LoopMeshFlags::EDGES_INDEX_DIRTY != LoopMeshFlags::NONE
    }

    pub fn is_faces_index_dirty(&self) -> bool {
        self.flags & LoopMeshFlags::FACES_INDEX_DIRTY != LoopMeshFlags::NONE
    }

    pub fn is_loops_index_dirty(&self) -> bool {
        self.flags & LoopMeshFlags::LOOPS_INDEX_DIRTY != LoopMeshFlags::NONE
    }

    /// Debug pass: assert every live element's cached `index` matches its
    /// position in the dense table, and that no dirty bit is set. Run after
    /// [`all_indices_ensure`] to verify integrity.
    #[cfg(debug_assertions)]
    pub fn validate_indices(&self) {
        assert!(
            !self.is_vertices_index_dirty(),
            "validate_indices: vertex indices are dirty; call vertices_ensure_index() first"
        );
        assert!(
            !self.is_edges_index_dirty(),
            "validate_indices: edge indices are dirty"
        );
        assert!(
            !self.is_faces_index_dirty(),
            "validate_indices: face indices are dirty"
        );
        assert!(
            !self.is_loops_index_dirty(),
            "validate_indices: loop indices are dirty"
        );

        for (i, &v) in self.vertices_index.iter().enumerate() {
            let cached = self.vertices.get(v).index as usize;
            assert_eq!(
                cached, i,
                "vertex {v:?}: cached index {cached} disagrees with table position {i}"
            );
        }
        for (i, &e) in self.edges_index.iter().enumerate() {
            let cached = self.edges.get(e).index as usize;
            assert_eq!(
                cached, i,
                "edge {e:?}: cached index {cached} disagrees with table position {i}"
            );
        }
        for (i, &f) in self.faces_index.iter().enumerate() {
            let cached = self.faces.get(f).index as usize;
            assert_eq!(
                cached, i,
                "face {f:?}: cached index {cached} disagrees with table position {i}"
            );
        }
        for (i, &l) in self.loops_index.iter().enumerate() {
            let cached = self.loops.get(l).index as usize;
            assert_eq!(
                cached, i,
                "loop {l:?}: cached index {cached} disagrees with table position {i}"
            );
        }
    }

    /// Debug-only structural integrity pass over the whole mesh. Panics on
    /// the first inconsistency it finds. Run after a non-trivial op to
    /// catch broken radial / disk / face cycles, dangling references, and
    /// loop ↔ edge / loop ↔ face mismatches.
    ///
    /// Not part of the parity contract — it inspects loopmesh's internal
    /// link invariants rather than reproducing any specific BMesh check.
    /// Safe to call on a freshly-constructed mesh; runs in O(V + E + F + L).
    #[cfg(debug_assertions)]
    pub fn validate_topology(&self) {
        const MAX_CYCLE: usize = 1_000_000;

        // ----- 1. Vertex disk cycles -----
        let v_keys: Vec<VertexKey> = self.vertices.iter().collect();
        for v in v_keys {
            let start = self.vertices.get(v).disk_loop_start;
            let Some(start) = start else { continue };
            // The disk_loop_start edge must contain this vertex.
            assert!(
                self.edges.get(start).contains(v).is_some(),
                "vertex {v:?}: disk_loop_start {start:?} doesn't contain it"
            );
            // Walk the cycle. Verify every edge claims `v` and the doubly-
            // linked invariant holds.
            let mut cur = start;
            let mut steps = 0;
            loop {
                let links = self.edges.get(cur).disk_loop_links_for_vert(v);
                let next = links.next;
                let next_links = self.edges.get(next).disk_loop_links_for_vert(v);
                assert_eq!(
                    next_links.prev, cur,
                    "vertex {v:?}: disk cycle broken at {cur:?} -> {next:?} (next.prev = {:?})",
                    next_links.prev
                );
                assert!(
                    self.edges.get(next).contains(v).is_some(),
                    "vertex {v:?}: disk cycle visits {next:?} which doesn't contain v"
                );
                cur = next;
                steps += 1;
                if cur == start {
                    break;
                }
                assert!(
                    steps < MAX_CYCLE,
                    "vertex {v:?}: disk cycle didn't close in {MAX_CYCLE} steps"
                );
            }
        }

        // ----- 2. Edge radial cycles -----
        let e_keys: Vec<EdgeKey> = self.edges.iter().collect();
        for e in e_keys {
            // Every edge's two endpoints must contain `e` in their disks
            // (otherwise the disk above wouldn't have seen it). Sanity-
            // check the back-pointer: the radial_link's loop must claim
            // this edge.
            let Some(radial_start) = self.edges.get(e).radial_link else {
                continue;
            };
            assert_eq!(
                self.loops.get(radial_start).edge,
                e,
                "edge {e:?}: radial_link {radial_start:?} doesn't claim this edge"
            );
            let mut cur = radial_start;
            let mut steps = 0;
            loop {
                let links = self.loops.get(cur).edge_radial_links;
                let next = links.next;
                assert_eq!(
                    self.loops.get(next).edge,
                    e,
                    "edge {e:?}: radial cycle visits {next:?} whose edge field disagrees"
                );
                let next_links = self.loops.get(next).edge_radial_links;
                assert_eq!(
                    next_links.prev, cur,
                    "edge {e:?}: radial cycle broken at {cur:?} -> {next:?}"
                );
                cur = next;
                steps += 1;
                if cur == radial_start {
                    break;
                }
                assert!(
                    steps < MAX_CYCLE,
                    "edge {e:?}: radial cycle didn't close in {MAX_CYCLE} steps"
                );
            }
        }

        // ----- 3. Face loop cycles + loop / edge / face cross-checks -----
        let f_keys: Vec<FaceKey> = self.faces.iter().collect();
        for f in f_keys {
            let face = self.faces.get(f);
            let start = face.first_edge_loop;
            let face_len = face.len as usize;
            assert!(
                face_len >= 1,
                "face {f:?}: len = 0 (face must have at least one loop)"
            );
            assert_eq!(
                self.loops.get(start).face,
                f,
                "face {f:?}: first_edge_loop {start:?}'s face field disagrees"
            );
            let mut cur = start;
            let mut count = 0;
            loop {
                let l = self.loops.get(cur);
                assert_eq!(
                    l.face, f,
                    "face {f:?}: loop {cur:?} in cycle claims face {:?}",
                    l.face
                );
                assert!(
                    !l.edge.is_null(),
                    "face {f:?}: loop {cur:?} has null edge (invariant violation outside SEMV)"
                );
                let e = l.edge;
                // The loop's edge must contain the loop's vertex.
                assert!(
                    self.edges.get(e).contains(l.vertex).is_some(),
                    "face {f:?} loop {cur:?}: edge {e:?} doesn't contain loop's vertex {:?}",
                    l.vertex
                );
                // The loop's edge must contain the next-loop's vertex too —
                // each loop covers the segment (vertex -> next.vertex).
                let next = l.face_loop_links.next;
                let next_vertex = self.loops.get(next).vertex;
                assert!(
                    self.edges.get(e).contains(next_vertex).is_some(),
                    "face {f:?} loop {cur:?}: edge {e:?} doesn't contain next-loop's vertex {next_vertex:?}"
                );
                // Doubly-linked face cycle.
                let next_links = self.loops.get(next).face_loop_links;
                assert_eq!(
                    next_links.prev, cur,
                    "face {f:?}: face_loop_links broken at {cur:?} -> {next:?}"
                );
                cur = next;
                count += 1;
                if cur == start {
                    break;
                }
                assert!(
                    count <= face_len,
                    "face {f:?}: face_loop cycle exceeds declared len {face_len}"
                );
            }
            assert_eq!(
                count, face_len,
                "face {f:?}: face_loop cycle length {count} disagrees with face.len {face_len}"
            );
        }

        // ----- 4. Cross-check: every live loop is reachable from its face -----
        // We've already validated each face's cycle visits the right number
        // of loops with the right `face` field. The complementary check —
        // "no orphan loops" — falls out as: live-loop-count must equal the
        // sum of face.len over all live faces.
        let total_loops_live = self.loops.iter().count();
        let total_face_loops: usize = self
            .faces
            .iter()
            .map(|f| self.faces.get(f).len as usize)
            .sum();
        assert_eq!(
            total_loops_live, total_face_loops,
            "{total_loops_live} live loops but face.len sum is {total_face_loops} — orphan loop(s)"
        );
    }

    /// Read the cached dense index of a vertex. Caller is responsible for
    /// having called `vertices_ensure_index()` since the last mutation.
    pub fn vertex_index(&self, v: VertexKey) -> u32 {
        self.vertices.get(v).index
    }
    pub fn edge_index(&self, e: EdgeKey) -> u32 {
        self.edges.get(e).index
    }
    pub fn face_index(&self, f: FaceKey) -> u32 {
        self.faces.get(f).index
    }
    pub fn loop_index(&self, l: LoopKey) -> u32 {
        self.loops.get(l).index
    }

    /// Read the index-table slice (matches the keys' cached `index` field).
    pub fn vertex_index_table(&self) -> &[VertexKey] {
        &self.vertices_index
    }
    pub fn edge_index_table(&self) -> &[EdgeKey] {
        &self.edges_index
    }
    pub fn face_index_table(&self) -> &[FaceKey] {
        &self.faces_index
    }
    pub fn loop_index_table(&self) -> &[LoopKey] {
        &self.loops_index
    }

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

    pub fn vertex_disk_iter(&self, vertex: VertexKey) -> DiskEdgesIterator<'_> {
        DiskEdgesIterator::new(self, vertex)
    }
    pub fn edge_radial_iter(&self, axis_edge: EdgeKey) -> RadialLoopIterator<'_> {
        RadialLoopIterator::new(self, axis_edge)
    }
    pub fn face_loop_iter(&self, face_loop_start: LoopKey) -> FaceEdgeLoopIterator<'_> {
        FaceEdgeLoopIterator::new(self, face_loop_start)
    }
    pub fn face_loop_reverse_iter(
        &self,
        face_loop_start: LoopKey,
    ) -> FaceEdgeLoopReverseIterator<'_> {
        FaceEdgeLoopReverseIterator::new(self, face_loop_start)
    }

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

    /// Lending iterator over the edges in `v`'s disk cycle.
    pub fn disk_edges(&self, v: VertexKey) -> DiskEdges {
        DiskEdges::new(self, v)
    }

    /// Lending iterator over the loops in `e`'s radial cycle.
    pub fn radial_loops(&self, e: EdgeKey) -> RadialLoops {
        RadialLoops::new(self, e)
    }

    /// Lending iterator over the loops in `f`'s perimeter cycle (forward).
    pub fn face_loops(&self, f: FaceKey) -> FaceLoops {
        FaceLoops::new(self, f)
    }

    /// Lending iterator over the loops in `f`'s perimeter cycle (reverse).
    pub fn face_loops_reverse(&self, f: FaceKey) -> FaceLoopsReverse {
        FaceLoopsReverse::new(self, f)
    }

    /// Zero-alloc lending iterator over every live vertex.
    pub fn mesh_verts(&self) -> MeshVerts {
        MeshVerts::default()
    }

    /// Zero-alloc lending iterator over every live edge.
    pub fn mesh_edges(&self) -> MeshEdges {
        MeshEdges::default()
    }

    /// Zero-alloc lending iterator over every live face.
    pub fn mesh_faces(&self) -> MeshFaces {
        MeshFaces::default()
    }

    /// Zero-alloc lending iterator over every live loop.
    pub fn mesh_loops(&self) -> MeshLoops {
        MeshLoops::default()
    }

    /// Read-only std-`Iterator` adapter over `v`'s disk edges. Borrows the
    /// mesh; for the mutation-friendly form use [`Self::disk_edges`].
    pub fn iter_disk_edges<'a>(&'a self, v: VertexKey) -> impl Iterator<Item = EdgeKey> + 'a {
        let mut it = DiskEdges::new(self, v);
        std::iter::from_fn(move || it.next(self))
    }

    pub fn iter_radial_loops<'a>(&'a self, e: EdgeKey) -> impl Iterator<Item = LoopKey> + 'a {
        let mut it = RadialLoops::new(self, e);
        std::iter::from_fn(move || it.next(self))
    }

    pub fn iter_face_loops<'a>(&'a self, f: FaceKey) -> impl Iterator<Item = LoopKey> + 'a {
        let mut it = FaceLoops::new(self, f);
        std::iter::from_fn(move || it.next(self))
    }

    pub fn iter_face_loops_reverse<'a>(&'a self, f: FaceKey) -> impl Iterator<Item = LoopKey> + 'a {
        let mut it = FaceLoopsReverse::new(self, f);
        std::iter::from_fn(move || it.next(self))
    }

    // ----- Mesh-wide snapshot iterators -----
    //
    // Each returns an owned `Vec<Key>` so the caller may freely mutate the
    // mesh while consuming the iteration: the snapshot-then-mutate idiom,
    // collecting keys first and editing afterward.

    pub fn vert_keys(&self) -> Vec<VertexKey> {
        self.vertices.iter().collect()
    }

    pub fn edge_keys(&self) -> Vec<EdgeKey> {
        self.edges.iter().collect()
    }

    pub fn face_keys(&self) -> Vec<FaceKey> {
        self.faces.iter().collect()
    }

    /// All currently-allocated loops; callers normally reach loops via faces.
    pub fn loop_keys(&self) -> Vec<LoopKey> {
        self.loops.iter().collect()
    }

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

    /// Refill `buf` with the keys of every live vertex (clearing first).
    /// Zero allocation if `buf` already has enough capacity.
    pub fn collect_vert_keys_into(&self, buf: &mut Vec<VertexKey>) {
        buf.clear();
        let mut it = MeshVerts::default();
        while let Some(v) = it.next(self) {
            buf.push(v);
        }
    }

    pub fn collect_edge_keys_into(&self, buf: &mut Vec<EdgeKey>) {
        buf.clear();
        let mut it = MeshEdges::default();
        while let Some(e) = it.next(self) {
            buf.push(e);
        }
    }

    pub fn collect_face_keys_into(&self, buf: &mut Vec<FaceKey>) {
        buf.clear();
        let mut it = MeshFaces::default();
        while let Some(f) = it.next(self) {
            buf.push(f);
        }
    }

    pub fn collect_loop_keys_into(&self, buf: &mut Vec<LoopKey>) {
        buf.clear();
        let mut it = MeshLoops::default();
        while let Some(l) = it.next(self) {
            buf.push(l);
        }
    }

    /// Borrow the mesh-owned vert-key scratch buffer (pre-filled with the
    /// current keys) for the duration of `f`. The buf is `mem::take`-d
    /// from the mesh, so `f` gets a fresh `&mut self` for free; the buf is
    /// restored after `f` returns. **No restoration on panic** — the buf
    /// gets reset to empty next time, which is correctness-safe.
    pub fn with_vert_keys<R>(&mut self, f: impl FnOnce(&mut LoopMesh, &[VertexKey]) -> R) -> R {
        let mut buf = std::mem::take(&mut self.vert_key_scratch);
        self.collect_vert_keys_into(&mut buf);
        let r = f(self, &buf);
        self.vert_key_scratch = buf;
        r
    }

    pub fn with_edge_keys<R>(&mut self, f: impl FnOnce(&mut LoopMesh, &[EdgeKey]) -> R) -> R {
        let mut buf = std::mem::take(&mut self.edge_key_scratch);
        self.collect_edge_keys_into(&mut buf);
        let r = f(self, &buf);
        self.edge_key_scratch = buf;
        r
    }

    pub fn with_face_keys<R>(&mut self, f: impl FnOnce(&mut LoopMesh, &[FaceKey]) -> R) -> R {
        let mut buf = std::mem::take(&mut self.face_key_scratch);
        self.collect_face_keys_into(&mut buf);
        let r = f(self, &buf);
        self.face_key_scratch = buf;
        r
    }

    pub fn with_loop_keys<R>(&mut self, f: impl FnOnce(&mut LoopMesh, &[LoopKey]) -> R) -> R {
        let mut buf = std::mem::take(&mut self.loop_key_scratch);
        self.collect_loop_keys_into(&mut buf);
        let r = f(self, &buf);
        self.loop_key_scratch = buf;
        r
    }

    pub fn totvert(&self) -> usize {
        self.vertices.len()
    }
    pub fn totedge(&self) -> usize {
        self.edges.len()
    }
    pub fn totface(&self) -> usize {
        self.faces.len()
    }
    pub fn totloop(&self) -> usize {
        self.loops.len()
    }

    /// Slab-key watermark — one past the highest slab slot ever allocated
    /// for this element type. Differs from `totvert()` after kills: kills
    /// reduce `totvert` but the watermark stays high (slot recycling is
    /// the slab's own concern). Useful for sizing per-element scratch
    /// buffers indexed by `key.as_usize()`.
    pub fn totvert_watermark(&self) -> usize {
        self.vertices.key_watermark()
    }
    pub fn totedge_watermark(&self) -> usize {
        self.edges.key_watermark()
    }
    pub fn totface_watermark(&self) -> usize {
        self.faces.key_watermark()
    }
    pub fn totloop_watermark(&self) -> usize {
        self.loops.key_watermark()
    }

    // ----- Per-element neighbor collectors -----

    pub fn vert_edge_keys(&self, v: VertexKey) -> Vec<EdgeKey> {
        self.vertex_disk_iter(v).collect()
    }

    /// Collects each neighbour vertex (the other endpoint of each disk edge).
    pub fn vert_neighbor_keys(&self, v: VertexKey) -> Vec<VertexKey> {
        self.vertex_disk_iter(v)
            .filter_map(|e| self.edges.get(e).other_vertex(v))
            .collect()
    }

    /// Deduplicates faces reached via multiple incident edges.
    pub fn vert_face_keys(&self, v: VertexKey) -> Vec<FaceKey> {
        let mut seen: Vec<FaceKey> = Vec::new();
        for e in self.vertex_disk_iter(v) {
            for l in self.edge_radial_iter(e) {
                let f = self.loops.get(l).face;
                if !seen.contains(&f) {
                    seen.push(f);
                }
            }
        }
        seen
    }

    pub fn edge_vert_keys(&self, e: EdgeKey) -> [VertexKey; 2] {
        self.edges.get(e).vertices
    }

    pub fn edge_loop_keys(&self, e: EdgeKey) -> Vec<LoopKey> {
        self.edge_radial_iter(e).collect()
    }

    /// One entry per loop in the radial cycle (will repeat if the same
    /// face is incident to the edge twice — non-manifold).
    pub fn edge_face_keys(&self, e: EdgeKey) -> Vec<FaceKey> {
        self.edge_radial_iter(e)
            .map(|l| self.loops.get(l).face)
            .collect()
    }

    pub fn face_loop_keys(&self, f: FaceKey) -> Vec<LoopKey> {
        let start = self.faces.get(f).first_edge_loop;
        self.face_loop_iter(start).collect()
    }

    pub fn face_vert_keys(&self, f: FaceKey) -> Vec<VertexKey> {
        let start = self.faces.get(f).first_edge_loop;
        self.face_loop_iter(start)
            .map(|l| self.loops.get(l).vertex)
            .collect()
    }

    /// On a fully-constructed mesh every face loop has an edge; the
    /// transient `EdgeKey::NULL` state inside SEMV is filtered out for
    /// safety but should never be observable here.
    pub fn face_edge_keys(&self, f: FaceKey) -> Vec<EdgeKey> {
        let start = self.faces.get(f).first_edge_loop;
        self.face_loop_iter(start)
            .map(|l| self.loops.get(l).edge)
            .filter(|e| !e.is_null())
            .collect()
    }

    // ----- Direct field accessors -----

    pub fn vertex_position(&self, v: VertexKey) -> Vec3 {
        self.vertices.get(v).position
    }

    /// Overwrite the vertex's position.
    pub fn set_vertex_position(&mut self, v: VertexKey, pos: Vec3) {
        self.vertices.get_mut(v).position = pos;
    }

    pub fn vertex_normal(&self, v: VertexKey) -> Vec3 {
        self.vertices.get(v).normal
    }

    /// Overwrite the vertex's stored normal.
    pub fn set_vertex_normal(&mut self, v: VertexKey, n: Vec3) {
        self.vertices.get_mut(v).normal = n;
    }

    pub fn edge_vertices(&self, e: EdgeKey) -> [VertexKey; 2] {
        self.edges.get(e).vertices
    }

    /// Vertex/loop count of the face.
    pub fn face_len(&self, f: FaceKey) -> usize {
        self.faces.get(f).len as usize
    }

    pub fn face_normal(&self, f: FaceKey) -> Vec3 {
        self.faces.get(f).normal
    }

    /// One loop of the face's loop cycle.
    pub fn face_first_loop(&self, f: FaceKey) -> LoopKey {
        self.faces.get(f).first_edge_loop
    }

    pub fn loop_vertex(&self, l: LoopKey) -> VertexKey {
        self.loops.get(l).vertex
    }

    /// Returns the edge this loop sits on.
    ///
    /// The returned key equals [`EdgeKey::NULL`] only in two situations:
    /// 1. Transiently inside `euler_split_edge_make_vert` while the radial
    ///    cycle is being rewoven — internal to that Euler op, never observable
    ///    at op boundaries.
    /// 2. After a manual `face_loop_disconnect_edge` call on a loop that
    ///    hasn't yet been reconnected.
    ///
    /// On a fully-constructed mesh (one that passes `validate_topology`),
    /// the returned key is always a live edge.
    pub fn loop_edge(&self, l: LoopKey) -> EdgeKey {
        self.loops.get(l).edge
    }

    /// True if `l` is currently connected to an edge (`loop_edge != NULL`).
    /// Use this in code that legitimately runs during a mid-edit transient,
    /// otherwise prefer `loop_edge` directly.
    pub fn loop_has_edge(&self, l: LoopKey) -> bool {
        !self.loops.get(l).edge.is_null()
    }

    pub fn loop_face(&self, l: LoopKey) -> FaceKey {
        self.loops.get(l).face
    }

    pub fn loop_next(&self, l: LoopKey) -> LoopKey {
        self.loops.get(l).face_loop_links.next
    }

    pub fn loop_prev(&self, l: LoopKey) -> LoopKey {
        self.loops.get(l).face_loop_links.prev
    }

    pub fn loop_radial_next(&self, l: LoopKey) -> LoopKey {
        self.loops.get(l).edge_radial_links.next
    }

    pub fn loop_radial_prev(&self, l: LoopKey) -> LoopKey {
        self.loops.get(l).edge_radial_links.prev
    }

    /// Append an edge to the disk loop of this hub vertex
    /// like: bmesh_disk_edge_append
    pub(crate) fn vertex_disk_append_edge(&mut self, vertex: VertexKey, edge: EdgeKey) {
        let disk_loop_start = self.vertices.get(vertex).disk_loop_start;
        match disk_loop_start {
            Some(start_edge) => {
                let start_links = self.edges.get(start_edge).disk_loop_links_for_vert(vertex);
                let end_edge = start_links.prev;
                let end_links = self.edges.get(end_edge).disk_loop_links_for_vert(vertex);

                let (end_links, new_links, start_links) =
                    DoubleLinks::insert_link(end_links, end_edge, edge, start_links, start_edge);

                self.edges
                    .get_mut(end_edge)
                    .set_disk_loop_links_for_vert(vertex, end_links);
                self.edges
                    .get_mut(edge)
                    .set_disk_loop_links_for_vert(vertex, new_links);
                self.edges
                    .get_mut(start_edge)
                    .set_disk_loop_links_for_vert(vertex, start_links);
            }
            None => {
                // Vertex currently has an empty disk loop
                let links = DoubleLinks {
                    next: edge,
                    prev: edge,
                };
                self.edges
                    .get_mut(edge)
                    .set_disk_loop_links_for_vert(vertex, links);
                self.vertices.get_mut(vertex).disk_loop_start = Some(edge);
            }
        }
    }

    pub fn vertex_disk_print_links(&self, vertex: VertexKey) {
        let v = &self.vertices.get(vertex);
        let start = v.disk_loop_start;
        let mut cur = start;
        let mut i = 0;
        while let Some(current) = cur {
            let e = &self.edges.get(current);
            let links = e.disk_loop_links_for_vert(vertex);
            println!(
                "edge {current}: next = {}, prev = {}",
                links.next, links.prev
            );
            cur = Some(links.next);
            if cur == start {
                break;
            }
            i += 1;
        }
    }

    /// Removes an edge from the disk cycle of a vertex
    /// Panics if the edge isn't associated with the vertex!
    pub(crate) fn vertex_disk_remove_edge(&mut self, vertex: VertexKey, edge: EdgeKey) {
        let links_old = self.edges.get(edge).disk_loop_links_for_vert(vertex);

        // "Only edge in the disk" iff its own next is itself. (For a length-2
        // cycle the prev and next are both the OTHER edge, so comparing
        // prev != next would mis-identify it as length-1.)
        if links_old.next != edge {
            // If this isn't the last edge in the disk cycle...
            let edge_next = links_old.next;
            let edge_prev = links_old.prev;
            // NB: read edge_PREV's links into the prev-slot, and edge_NEXT's
            // links into the next-slot — earlier revisions of this code had
            // them swapped, which happened to work for length-2 cycles
            // (special case in remove_link) but produced stale links for
            // length-3+ disk cycles.
            let links_for_prev = self.edges.get(edge_prev).disk_loop_links_for_vert(vertex);
            let links_for_next = self.edges.get(edge_next).disk_loop_links_for_vert(vertex);

            let (new_prev_links, old_links, new_next_links) = DoubleLinks::remove_link(
                links_for_prev,
                edge_prev,
                edge,
                links_for_next,
                edge_next,
            );

            self.edges
                .get_mut(edge_prev)
                .set_disk_loop_links_for_vert(vertex, new_prev_links);
            self.edges
                .get_mut(edge)
                .set_disk_loop_links_for_vert(vertex, old_links);
            self.edges
                .get_mut(edge_next)
                .set_disk_loop_links_for_vert(vertex, new_next_links);

            if self.vertices.get(vertex).disk_loop_start == Some(edge) {
                self.vertices.get_mut(vertex).disk_loop_start = Some(edge_next);
            }
        } else {
            self.vertices.get_mut(vertex).disk_loop_start = None;
        }
    }

    pub(crate) fn vertex_disk_print_edges(&mut self, hub_vertex: VertexKey) {
        if let Some(start_edge) = self.vertices.get(hub_vertex).disk_loop_start {
            let mut i = 0;
            let mut next_edge = start_edge;
            loop {
                println!("Edge {i} = {next_edge}");
                let links = self
                    .edges
                    .get(next_edge)
                    .disk_loop_links_for_vert(hub_vertex);
                next_edge = links.next;
                if next_edge == start_edge {
                    break;
                }
                i += 1;
            }
        } else {
            println!("No disk loop for vertex");
        }
    }

    pub(crate) fn vertex_disk_replace_vertex(
        &mut self,
        edge: EdgeKey,
        old: VertexKey,
        new: VertexKey,
    ) {
    }

    /// Iterates through all the edges in the disk loop of the given `hub_vertex` to
    /// determine if the `other_vertex` is also contained in the disk loop
    ///
    /// like: bmesh_disk_edge_exists
    pub fn vertex_disk_contains_edge(
        &mut self,
        hub_vertex: VertexKey,
        other_vertex: VertexKey,
    ) -> Option<EdgeKey> {
        for edge in self.vertex_disk_iter(hub_vertex) {
            if self.edges.get(edge).contains(other_vertex).is_some() {
                return Some(edge);
            }
        }
        None
    }

    /// Iterates through all the edges in the disk loop of the given `hub_vertex` to
    /// count the number of edges in the loop.
    ///
    /// like: bmesh_disk_count
    pub fn vertex_disk_count_edges(&self, hub_vertex: VertexKey) -> usize {
        let mut count = 0;
        for _ in self.vertex_disk_iter(hub_vertex) {
            count += 1;
        }
        count
    }

    /// Iterates through all the edges in the disk loop of the given `hub_vertex` to
    /// count the number of edges in the loop, up to `max_count`.
    ///
    /// like: bmesh_disk_count
    pub(crate) fn vertex_disk_count_edges_capped(
        &self,
        hub_vertex: VertexKey,
        max_count: usize,
    ) -> usize {
        let mut count = 0;
        for _ in self.vertex_disk_iter(hub_vertex) {
            count += 1;
            if count >= max_count {
                return max_count;
            }
        }
        count
    }

    /// Validates the integrity of a disk loop around the given `hub_vertex` with respect to a
    /// known edge member and an expected edge count.
    ///
    /// like: bmesh_disk_validate
    pub(crate) fn vertex_disk_validate_edge_state(
        &mut self,
        hub_vertex: VertexKey,
        member_edge: EdgeKey,
        num_edges: usize,
    ) -> Result<()> {
        if num_edges == 0 {
            return Err(anyhow!("invalid, empty, disk loop"));
        }

        if self.edges.get(member_edge).contains(hub_vertex).is_none() {
            return Err(anyhow!(
                "invalid disk loop: member edge doesn't contain hub vertex"
            ));
        }

        let found_edges = self.vertex_disk_count_edges_capped(hub_vertex, num_edges + 1);
        if found_edges != num_edges {
            return Err(anyhow!("invalid disk loop: expected {num_edges} edges, found {found_edges} (capped search at {})", num_edges + 1));
        }

        Ok(())
    }

    pub fn create_vertex(&mut self, position: Vec3) -> VertexKey {
        self.flags |= LoopMeshFlags::VERTICES_INDEX_DIRTY;
        let entry = self.vertices.vacant_entry();
        let vi = VertexKey::from_usize(entry.key());
        entry.insert(Vertex {
            eflags: ElementFlags::NONE,
            key: vi,
            index: 0,
            position,
            normal: Vec3::ZERO,
            disk_loop_start: None,
        });

        let key_watermark = self.vertices.key_watermark();
        for attrib_vec in self.vertex_attribs.attributes_u8.iter_mut() {
            attrib_vec.resize(key_watermark, 0);
        }
        for attrib_vec in self.vertex_attribs.attributes_i32.iter_mut() {
            attrib_vec.resize(key_watermark, 0);
        }
        for attrib_vec in self.vertex_attribs.attributes_u32.iter_mut() {
            attrib_vec.resize(key_watermark, 0);
        }
        for attrib_vec in self.vertex_attribs.attributes_f32.iter_mut() {
            attrib_vec.resize(key_watermark, 0.0);
        }
        for attrib_vec in self.vertex_attribs.attributes_vec2.iter_mut() {
            attrib_vec.resize(key_watermark, [0.0; 2]);
        }
        for attrib_vec in self.vertex_attribs.attributes_vec3.iter_mut() {
            attrib_vec.resize(key_watermark, [0.0; 3]);
        }
        for attrib_vec in self.vertex_attribs.attributes_vec4.iter_mut() {
            attrib_vec.resize(key_watermark, [0.0; 4]);
        }
        for attrib_vec in self.vertex_attribs.attributes_string.iter_mut() {
            attrib_vec.resize(key_watermark, None);
        }
        self.op_flags.grow_vert_layers(key_watermark);

        vi
    }

    fn vertex_data_remove(&mut self, vertex: VertexKey) {
        self.remove_vertex_slab_entry(vertex);
    }

    // ----- Internal slab-removal helpers (dirty-flag the matching index table) -----
    //
    // The Euler operators routinely free elements directly. Routing every
    // such free through these helpers keeps the *_INDEX_DIRTY bits honest
    // without sprinkling `self.flags |= …` everywhere.

    fn remove_vertex_slab_entry(&mut self, v: VertexKey) {
        self.flags |= LoopMeshFlags::VERTICES_INDEX_DIRTY;
        self.vertices_index.clear();
        self.vertices.remove(v);
    }
    fn remove_edge_slab_entry(&mut self, e: EdgeKey) {
        self.flags |= LoopMeshFlags::EDGES_INDEX_DIRTY;
        self.edges_index.clear();
        self.edges.remove(e);
    }
    fn remove_face_slab_entry(&mut self, f: FaceKey) {
        self.flags |= LoopMeshFlags::FACES_INDEX_DIRTY;
        self.faces_index.clear();
        self.faces.remove(f);
    }
    fn remove_loop_slab_entry(&mut self, l: LoopKey) {
        self.flags |= LoopMeshFlags::LOOPS_INDEX_DIRTY;
        self.loops_index.clear();
        self.loops.remove(l);
    }

    pub fn vertex_kill(&mut self, vertex: VertexKey) {
        // Kill every edge on the vert's disk, then free the vert. We can't hold
        // a disk iterator across the kills (it would borrow the mesh), and we
        // can't pre-capture the cycle's `next` either: each `edge_kill` shortens
        // the disk in place, so any edge key captured before a kill can name an
        // edge a later kill has already freed. Instead we always re-read the
        // live `disk_loop_start` and kill whatever it currently points at; each
        // `edge_kill` clears or advances it, so the disk drains to empty.
        while let Some(edge) = self.vertices.get(vertex).disk_loop_start {
            self.edge_kill(edge);
        }

        self.vertex_data_remove(vertex);
    }

    pub fn create_edge(&mut self, vertices: [VertexKey; 2], flags: EdgeCreateFlags) -> EdgeKey {
        if flags & EdgeCreateFlags::NO_DUPLICATE != EdgeCreateFlags::NONE {
            if let Some(edge) = self.query_edge_exists(vertices[0], vertices[1]) {
                return edge;
            }
        }
        self.flags |= LoopMeshFlags::EDGES_INDEX_DIRTY;

        let new_entry = self.edges.vacant_entry();
        let edge = EdgeKey(new_entry.key() as u32);

        let self_links = DoubleLinks {
            next: edge,
            prev: edge,
        };
        new_entry.insert(Edge {
            eflags: ElementFlags::NONE,
            key: edge,
            index: 0,
            vertices,
            vertex_disk_links: [self_links, self_links],
            radial_link: None,
        });

        let key_watermark = self.edges.key_watermark();
        for attrib_vec in self.edge_attribs.attributes_u8.iter_mut() {
            attrib_vec.resize(key_watermark, 0);
        }
        for attrib_vec in self.edge_attribs.attributes_i32.iter_mut() {
            attrib_vec.resize(key_watermark, 0);
        }
        for attrib_vec in self.edge_attribs.attributes_u32.iter_mut() {
            attrib_vec.resize(key_watermark, 0);
        }
        for attrib_vec in self.edge_attribs.attributes_f32.iter_mut() {
            attrib_vec.resize(key_watermark, 0.0);
        }
        for attrib_vec in self.edge_attribs.attributes_vec2.iter_mut() {
            attrib_vec.resize(key_watermark, [0.0; 2]);
        }
        for attrib_vec in self.edge_attribs.attributes_vec3.iter_mut() {
            attrib_vec.resize(key_watermark, [0.0; 3]);
        }
        for attrib_vec in self.edge_attribs.attributes_vec4.iter_mut() {
            attrib_vec.resize(key_watermark, [0.0; 4]);
        }
        for attrib_vec in self.edge_attribs.attributes_string.iter_mut() {
            attrib_vec.resize(key_watermark, None);
        }
        self.op_flags.grow_edge_layers(key_watermark);

        self.vertex_disk_append_edge(vertices[0], edge);
        self.vertex_disk_append_edge(vertices[1], edge);

        edge
    }

    fn edge_data_remove(&mut self, edge: EdgeKey) {
        self.remove_edge_slab_entry(edge);
    }

    /// Removes an edge, and any associated faces (and loops) from the mesh
    pub fn edge_kill(&mut self, edge: EdgeKey) {
        // Can't use an iterator since that would take a shared reference to the mesh
        // and stop us removing the associated faces.
        // Kill every face on the edge's radial cycle. We can't pre-capture the
        // cycle's `next` loop: each `face_kill` splices its loop out of this
        // edge's radial cycle (and frees that loop), so a loop key captured
        // before a kill can name a freed loop, and the original `first` loop is
        // gone after the first kill — leaving the `cur == first` termination
        // unreachable. Instead we always re-read the live `radial_link` and kill
        // the face it currently points at; each `face_kill` clears or advances
        // it, so the radial cycle drains to empty.
        while let Some(loopi) = self.edges.get(edge).radial_link {
            let face = self.loops.get(loopi).face;
            self.face_kill(face);
        }

        let vertices = self.edges.get(edge).vertices;
        self.vertex_disk_remove_edge(vertices[0], edge);
        self.vertex_disk_remove_edge(vertices[1], edge);

        self.edge_data_remove(edge);
    }

    fn init_face_edge_loop_data(loopi: LoopKey, vertex: VertexKey, face: FaceKey) -> Loop {
        Loop {
            key: loopi,
            index: 0,
            vertex,
            edge: EdgeKey::NULL,
            face,
            edge_radial_links: DoubleLinks {
                next: loopi,
                prev: loopi,
            },
            face_loop_links: DoubleLinks {
                next: loopi,
                prev: loopi,
            },
        }
    }

    /// NB: The returned loop is disconnected/orphaned, in that it's not yet linked in to the radial cycle of
    /// and edge. Call edge_radial_append_loop to also connect the loop to an edge.
    pub fn create_loop(&mut self, vertex: VertexKey, face: FaceKey) -> LoopKey {
        self.flags |= LoopMeshFlags::LOOPS_INDEX_DIRTY;
        let new_entry = self.loops.vacant_entry();
        let loopi = LoopKey(new_entry.key() as u32);

        let loop_data = LoopMesh::init_face_edge_loop_data(loopi, vertex, face);
        new_entry.insert(loop_data);

        let key_watermark = self.loops.key_watermark();
        for attrib_vec in self.loop_attribs.attributes_u8.iter_mut() {
            attrib_vec.resize(key_watermark, 0);
        }
        for attrib_vec in self.loop_attribs.attributes_i32.iter_mut() {
            attrib_vec.resize(key_watermark, 0);
        }
        for attrib_vec in self.loop_attribs.attributes_u32.iter_mut() {
            attrib_vec.resize(key_watermark, 0);
        }
        for attrib_vec in self.loop_attribs.attributes_f32.iter_mut() {
            attrib_vec.resize(key_watermark, 0.0);
        }
        for attrib_vec in self.loop_attribs.attributes_vec2.iter_mut() {
            attrib_vec.resize(key_watermark, [0.0; 2]);
        }
        for attrib_vec in self.loop_attribs.attributes_vec3.iter_mut() {
            attrib_vec.resize(key_watermark, [0.0; 3]);
        }
        for attrib_vec in self.loop_attribs.attributes_vec4.iter_mut() {
            attrib_vec.resize(key_watermark, [0.0; 4]);
        }
        for attrib_vec in self.loop_attribs.attributes_string.iter_mut() {
            attrib_vec.resize(key_watermark, None);
        }
        self.op_flags.grow_loop_layers(key_watermark);

        loopi
    }

    pub fn query_edge_exists(&self, v0: VertexKey, v1: VertexKey) -> Option<EdgeKey> {
        if v0 == v1 {
            return None;
        }

        // iterate the disk cycle for each vertex in lockstep, looking for a reciprocal
        // pointer back to the other edge, and stop searching at the end of the
        // shortest cycle

        let start0 = self.vertices.get(v0).disk_loop_start;
        let start1 = self.vertices.get(v1).disk_loop_start;

        if start0.is_none() || start1.is_none() {
            // one or the other vertex doesn't belong to any edge...
            return None;
        }
        let start0 = start0.unwrap();
        let start1 = start1.unwrap();

        let mut next0 = start0;
        let mut next1 = start1;

        loop {
            if self.edges.get(next0).contains(v1).is_some() {
                return Some(next0);
            }
            if self.edges.get(next1).contains(v0).is_some() {
                return Some(next1);
            }

            let links0 = self.edges.get(next0).disk_loop_links_for_vert(v0);
            next0 = links0.next;
            if next0 == start0 {
                break;
            }
            let links1 = self.edges.get(next1).disk_loop_links_for_vert(v1);
            next1 = links1.next;
            if next1 == start1 {
                break;
            }
        }

        None
    }

    pub fn find_edge_duplicate(&self, edge: EdgeKey) -> Option<EdgeKey> {
        let v1 = self.edges.get(edge).vertices[0];
        let v2 = self.edges.get(edge).vertices[1];

        self.vertex_disk_iter(v1)
            .find(|&e| self.edges.get(e).contains(v2).is_some())
    }

    // like: bmesh_radial_loop_append
    //
    // Invariant: a loop with a non-null edge is linked into that edge's radial
    // cycle. A loop is only transiently edge-null while an Euler op rewires it.
    pub(crate) fn face_loop_connect_edge(&mut self, axis_edge: EdgeKey, loopi: LoopKey) {
        let cur = self.loops.get(loopi).edge;
        assert!(
            cur.is_null() || cur == axis_edge,
            "face_loop_connect_edge: loop is already on a different edge ({cur:?} vs {axis_edge:?})"
        );

        // Make sure we don't old Ref to axis edge, since we need to mutate later...
        let axis_radial_link = self.edges.get(axis_edge).radial_link;
        if let Some(starti) = axis_radial_link {
            let start_links = self.loops.get(starti).edge_radial_links;
            let endi = start_links.prev;
            let end_links = self.loops.get(endi).edge_radial_links;
            let (end_links, new_links, start_links) =
                DoubleLinks::insert_link(end_links, endi, loopi, start_links, starti);

            self.loops.get_mut(endi).edge_radial_links = end_links;
            self.loops.get_mut(loopi).edge_radial_links = new_links;
            self.loops.get_mut(starti).edge_radial_links = start_links;
        } else {
            self.edges.get_mut(axis_edge).radial_link = Some(loopi);
            self.loops.get_mut(loopi).edge_radial_links = DoubleLinks {
                next: loopi,
                prev: loopi,
            };
        }

        self.loops.get_mut(loopi).edge = axis_edge;
    }

    /// like: bmesh_radial_loop_remove
    pub(crate) fn face_loop_disconnect_edge(&mut self, loopi: LoopKey) {
        // Bind the edge value into a local first so subsequent get_mut
        // calls on the same loop don't conflict with an outstanding borrow.
        let axis_edge = self.loops.get(loopi).edge;
        if axis_edge.is_null() {
            return;
        }
        let old_links = self.loops.get(loopi).edge_radial_links;

        // Two cases:
        // - last loop on this edge's radial — set edge.radial_link = None.
        // - otherwise — splice out the loop. If edge.radial_link == loopi,
        //   redirect it to old_links.next.
        // Detect "only loop in the cycle" via `next == loopi` (a single-
        // element cycle self-points). A length-2 cycle has prev == next ==
        // the OTHER loop, so `prev != next` would incorrectly mark it as
        // length-1.
        if old_links.next != loopi {
            let previ = old_links.prev;
            let nexti = old_links.next;
            let prev_links = self.loops.get(previ).edge_radial_links;
            let next_links = self.loops.get(nexti).edge_radial_links;

            let (prev_links, old_links, next_links) =
                DoubleLinks::remove_link(prev_links, previ, loopi, next_links, nexti);

            self.loops.get_mut(previ).edge_radial_links = prev_links;
            self.loops.get_mut(loopi).edge_radial_links = old_links;
            self.loops.get_mut(nexti).edge_radial_links = next_links;

            if self.edges.get(axis_edge).radial_link == Some(loopi) {
                self.edges.get_mut(axis_edge).radial_link = Some(nexti);
            }
        } else {
            self.edges.get_mut(axis_edge).radial_link = None;
        }

        self.loops.get_mut(loopi).edge = EdgeKey::NULL;
    }

    pub(crate) fn face_loop_radial_len(&self, loopi: LoopKey) -> usize {
        let mut count = 0;
        for l in self.face_loop_iter(loopi) {
            count += 1;
        }
        count
    }

    /// Re-anchor a single face-loop from its current edge onto `new_edge`,
    /// leaving the face's loop cycle (face_loop_links) untouched. The new
    /// edge must connect the same two vertices the old edge did, so that
    /// the loop's `(vertex, next.vertex)` contract still holds.
    ///
    /// This is the primitive `extrude_region` uses to swap a selected
    /// face's perimeter from "shared with boundary" edges onto duplicate
    /// edges between the duplicate vertices. Unlike `euler_split_face_make_edge`
    /// it does not change the face's loop count — only which edge each
    /// loop is anchored to.
    pub(crate) fn edge_replace_in_face_loop(&mut self, loopi: LoopKey, new_edge: EdgeKey) {
        self.face_loop_disconnect_edge(loopi);
        self.face_loop_connect_edge(new_edge, loopi);
    }

    /// Queries whether some face already exists for the given set of vertices, with
    /// either a forwards or reverse winding order.
    pub fn face_exists(&self, verts: &[VertexKey]) -> Option<FaceKey> {
        let face_len = verts.len() as u32;
        assert!(face_len >= 1);

        // Look for another edge connected to the first vertex with any
        // associated (radial) face loops...
        for e in self.vertex_disk_iter(verts[0]) {
            for l in self.edge_radial_iter(e) {
                let radial_loop = &self.loops.get(l);
                let face = radial_loop.face;
                if radial_loop.vertex != verts[0] || self.faces.get(face).len != face_len {
                    continue;
                }

                // Try matching the vertices with a forwards winding order first...

                let iter_forward = self
                    .face_loop_iter(self.faces.get(face).first_edge_loop)
                    .skip(1);

                let mut i = 1;
                for fl in iter_forward {
                    if self.loops.get(fl).vertex != verts[i] {
                        break;
                    }
                    i += 1;
                }
                if i as u32 == face_len {
                    return Some(face);
                }

                // Then try matching the vertices with a reverse winding order...

                let iter_backwards = self
                    .face_loop_reverse_iter(self.faces.get(face).first_edge_loop)
                    .skip(1);

                i = 1;
                for fl in iter_backwards {
                    if self.loops.get(fl).vertex != verts[i] {
                        break;
                    }
                    i += 1;
                }
                if i as u32 == face_len {
                    return Some(face);
                }
            }
        }

        None
    }

    pub fn face_create(
        &mut self,
        verts: &[VertexKey],
        edges: &[EdgeKey],
        flags: FaceCreateFlags,
    ) -> FaceKey {
        // For now we allow degenerate edges/faces, since these are low-level APIs and the more general
        // representation may help avoid special cases during non-atomic mesh edits
        assert!(!verts.is_empty() && edges.len() == verts.len());

        if flags & FaceCreateFlags::NO_DUPLICATE != FaceCreateFlags::NONE {
            if let Some(face) = self.face_exists(verts) {
                return face;
            }
        }
        self.flags |= LoopMeshFlags::FACES_INDEX_DIRTY | LoopMeshFlags::LOOPS_INDEX_DIRTY;

        let new_face_entry = self.faces.vacant_entry();
        let face = FaceKey(new_face_entry.key() as u32);

        // Note: we don't use self.create_loop because the vacant_entry is holding a mutable reference
        // to the slab and that stops us invoking methods that would take a second mutable reference
        // to the mesh
        let new_loop_entry = self.loops.vacant_entry();
        let first_edge_loop = LoopKey(new_loop_entry.key() as u32);
        new_loop_entry.insert(LoopMesh::init_face_edge_loop_data(
            first_edge_loop,
            verts[0],
            face,
        ));

        let key_watermark = self.loops.key_watermark();
        for attrib_vec in self.loop_attribs.attributes_u8.iter_mut() {
            attrib_vec.resize(key_watermark, 0);
        }
        for attrib_vec in self.loop_attribs.attributes_i32.iter_mut() {
            attrib_vec.resize(key_watermark, 0);
        }
        for attrib_vec in self.loop_attribs.attributes_u32.iter_mut() {
            attrib_vec.resize(key_watermark, 0);
        }
        for attrib_vec in self.loop_attribs.attributes_f32.iter_mut() {
            attrib_vec.resize(key_watermark, 0.0);
        }
        for attrib_vec in self.loop_attribs.attributes_vec2.iter_mut() {
            attrib_vec.resize(key_watermark, [0.0; 2]);
        }
        for attrib_vec in self.loop_attribs.attributes_vec3.iter_mut() {
            attrib_vec.resize(key_watermark, [0.0; 3]);
        }
        for attrib_vec in self.loop_attribs.attributes_vec4.iter_mut() {
            attrib_vec.resize(key_watermark, [0.0; 4]);
        }
        for attrib_vec in self.loop_attribs.attributes_string.iter_mut() {
            attrib_vec.resize(key_watermark, None);
        }
        self.op_flags.grow_loop_layers(key_watermark);

        new_face_entry.insert(Face {
            eflags: ElementFlags::NONE,
            key: face,
            index: 0,
            len: verts.len() as u32,
            normal: Vec3::ZERO,
            first_edge_loop,
        });

        let key_watermark = self.faces.key_watermark();
        for attrib_vec in self.face_attribs.attributes_u8.iter_mut() {
            attrib_vec.resize(key_watermark, 0);
        }
        for attrib_vec in self.face_attribs.attributes_i32.iter_mut() {
            attrib_vec.resize(key_watermark, 0);
        }
        for attrib_vec in self.face_attribs.attributes_u32.iter_mut() {
            attrib_vec.resize(key_watermark, 0);
        }
        for attrib_vec in self.face_attribs.attributes_f32.iter_mut() {
            attrib_vec.resize(key_watermark, 0.0);
        }
        for attrib_vec in self.face_attribs.attributes_vec2.iter_mut() {
            attrib_vec.resize(key_watermark, [0.0; 2]);
        }
        for attrib_vec in self.face_attribs.attributes_vec3.iter_mut() {
            attrib_vec.resize(key_watermark, [0.0; 3]);
        }
        for attrib_vec in self.face_attribs.attributes_vec4.iter_mut() {
            attrib_vec.resize(key_watermark, [0.0; 4]);
        }
        for attrib_vec in self.face_attribs.attributes_string.iter_mut() {
            attrib_vec.resize(key_watermark, None);
        }
        self.op_flags.grow_face_layers(key_watermark);

        // Finish initializing the face edge loop
        self.face_loop_connect_edge(edges[0], first_edge_loop);

        // connect all the edges into the loop...
        let mut prev_loopi = first_edge_loop;
        for i in 1..verts.len() {
            let loopi = self.create_loop(verts[i], face);
            self.face_loop_connect_edge(edges[i], loopi);

            self.loops.get_mut(loopi).face_loop_links.prev = prev_loopi;
            self.loops.get_mut(prev_loopi).face_loop_links.next = loopi;
            prev_loopi = loopi;
        }

        // close the loop
        self.loops.get_mut(first_edge_loop).face_loop_links.prev = prev_loopi;
        self.loops.get_mut(prev_loopi).face_loop_links.next = first_edge_loop;

        face
    }

    /// Removes a Loop from the slab, without doing anything to verify/maintain
    /// the integrity of the mesh structure, such as unlinking from a face or
    /// radial loops (assumes caller is responsible for this)
    fn face_loop_data_remove(&mut self, loopi: LoopKey) {
        self.remove_loop_slab_entry(loopi);
    }

    /// Normalise to unit length, or return the all-zero vector when the
    /// input is (near-)zero or contains a NaN. The test is on the squared
    /// length against a threshold far below float epsilon, so any face with
    /// measurable area normalises normally; only a genuinely degenerate
    /// (zero-area) face takes the zero branch.
    fn normalize_or_zero(v: Vec3) -> Vec3 {
        let len_sq = v.dot(v);
        if len_sq > 1.0e-35 {
            v / len_sq.sqrt()
        } else {
            Vec3::ZERO
        }
    }

    fn calc_tri_cross(corners: [Vec3; 3]) -> Vec3 {
        let side0 = corners[0] - corners[1];
        let side1 = corners[1] - corners[2];
        side0.cross(side1)
    }

    fn calc_tri_normal(corners: [Vec3; 3]) -> Vec3 {
        LoopMesh::normalize_or_zero(LoopMesh::calc_tri_cross(corners))
    }

    fn calc_tri_area(corners: [Vec3; 3]) -> f32 {
        LoopMesh::calc_tri_cross(corners).length() * 0.5
    }

    fn calc_quad_normal(corners: [Vec3; 4]) -> Vec3 {
        let diagonal0 = corners[0] - corners[2];
        let diagonal1 = corners[1] - corners[3];
        let cross = diagonal0.cross(diagonal1);
        LoopMesh::normalize_or_zero(cross)
    }

    fn calc_quad_area(corners: [Vec3; 4]) -> f32 {
        let diagonal0 = corners[0] - corners[2];
        let diagonal1 = corners[1] - corners[3];
        let cross = diagonal0.cross(diagonal1);
        cross.length() * 0.5
    }

    fn calc_poly_newell_normal_not_normalized(corners: &[Vec3]) -> Vec3 {
        /*
        Newell Method:

        Pseudo code from: https://www.khronos.org/opengl/wiki/Calculating_a_Surface_Normal

        Set Vertex Normal to (0, 0, 0)

        Begin Cycle for Index in [0, Polygon.vertexNumber)

            Set Vertex Current to Polygon.verts[Index]
            Set Vertex Next    to Polygon.verts[(Index plus 1) mod Polygon.vertexNumber]

            Set Normal.x to Sum of Normal.x and (multiply (Current.y minus Next.y) by (Current.z plus Next.z))
            Set Normal.y to Sum of Normal.y and (multiply (Current.z minus Next.z) by (Current.x plus Next.x))
            Set Normal.z to Sum of Normal.z and (multiply (Current.x minus Next.x) by (Current.y plus Next.y))

        End Cycle

        Returning Normalize(Normal)
         */

        let mut normal = Vec3::ZERO;
        for i in 0..corners.len() {
            let cur = corners[i];
            let next = corners[(i + 1) % corners.len()];

            normal.x += (cur.y - next.y) * (cur.z + next.z);
            normal.y += (cur.z - next.z) * (cur.x + next.x);
            normal.z += (cur.x - next.x) * (cur.y + next.y);
        }

        normal
    }

    fn calc_poly_normal(corners: &[Vec3]) -> Vec3 {
        LoopMesh::normalize_or_zero(LoopMesh::calc_poly_newell_normal_not_normalized(corners))
    }

    fn calc_iter_newell_normal_not_normalized(mut corner_iter: impl Iterator<Item = Vec3>) -> Vec3 {
        match corner_iter.next() {
            Some(first_corner) => {
                let mut normal = Vec3::ZERO;
                let mut prev = first_corner;
                let mut end = false;
                loop {
                    let cur = match corner_iter.next() {
                        Some(cur) => cur,
                        None => {
                            end = true;
                            first_corner
                        }
                    };

                    normal.x += (prev.y - cur.y) * (prev.z + cur.z);
                    normal.y += (prev.z - cur.z) * (prev.x + cur.x);
                    normal.z += (prev.x - cur.x) * (prev.y + cur.y);

                    if end {
                        break;
                    }

                    prev = cur;
                }

                normal
            }
            None => Vec3::ONE,
        }
    }

    fn calc_iter_poly_normal(corner_iter: impl Iterator<Item = Vec3>) -> Vec3 {
        LoopMesh::normalize_or_zero(LoopMesh::calc_iter_newell_normal_not_normalized(corner_iter))
    }

    fn calc_poly_area(corners: &[Vec3]) -> f32 {
        f32::abs(LoopMesh::calc_poly_newell_normal_not_normalized(corners).length() * 0.5)
    }

    fn calc_iter_poly_area(corner_iter: impl Iterator<Item = Vec3>) -> f32 {
        f32::abs(LoopMesh::calc_iter_newell_normal_not_normalized(corner_iter).length() * 0.5)
    }

    pub fn face_calc_normal(&mut self, face: FaceKey) {
        let normal = self.face_compute_normal(face);
        self.faces.get_mut(face).normal = normal;
    }

    /// Compute (but do not store) the face's normal from its current loop
    /// positions. Uses Newell's method for non-triangle / non-quad faces
    /// so non-planar polygons get a reasonable result. The output is unit
    /// length for a non-degenerate face; for a fully degenerate (zero-area)
    /// face — collinear or coincident corners — the output is the all-zero
    /// vector `(0, 0, 0)`, never a `NaN` or a fixed-axis fallback.
    pub fn face_compute_normal(&self, face: FaceKey) -> Vec3 {
        let (face_len, first_edge_loop) = {
            let face_data = self.faces.get(face);
            (face_data.len, face_data.first_edge_loop)
        };
        let iter = self.face_loop_iter(first_edge_loop);
        let mut corner_iter = iter.map(|l| self.vertices.get(self.loops.get(l).vertex).position);
        match face_len {
            3 => {
                let mut pos = [Vec3::ZERO; 3];
                for i in 0..3 {
                    pos[i] = corner_iter.next().unwrap();
                }
                LoopMesh::calc_tri_normal(pos)
            }
            4 => {
                let mut pos = [Vec3::ZERO; 4];
                for i in 0..4 {
                    pos[i] = corner_iter.next().unwrap();
                }
                LoopMesh::calc_quad_normal(pos)
            }
            _ => LoopMesh::calc_iter_poly_normal(corner_iter),
        }
    }

    pub fn face_calc_area(&mut self, face: FaceKey) -> f32 {
        let face_data = self.faces.get(face);
        let iter = self.face_loop_iter(face_data.first_edge_loop);
        let mut corner_iter = iter.map(|l| self.vertices.get(self.loops.get(l).vertex).position);
        match face_data.len {
            3 => {
                let mut pos = [Vec3::ZERO; 3];
                for i in 0..3 {
                    pos[i] = corner_iter.next().unwrap();
                }
                LoopMesh::calc_tri_area(pos)
            }
            4 => {
                let mut pos = [Vec3::ZERO; 4];
                for i in 0..4 {
                    pos[i] = corner_iter.next().unwrap();
                }
                LoopMesh::calc_quad_area(pos)
            }
            _ => LoopMesh::calc_iter_poly_area(corner_iter),
        }
    }

    pub fn calc_face_normals(&mut self) {
        let mut it = self.mesh_faces();
        while let Some(face) = it.next(self) {
            self.face_calc_normal(face);
        }
    }

    /// Recompute a single face's normal. Alias for [`Self::face_calc_normal`].
    pub fn face_normal_update(&mut self, face: FaceKey) {
        self.face_calc_normal(face);
    }

    /// Recompute a single vertex's normal as the area-weighted average of
    /// incident face normals, normalised. Falls back to `Vec3::Z` for
    /// isolated / wire-only vertices.
    ///
    /// Note: callers should ensure face normals are current first (e.g.
    /// via [`Self::mesh_normals_update`]); the area weight is read from
    /// each face's stored `normal` field.
    pub fn vert_normal_update(&mut self, v: VertexKey) {
        // Walk v's incident faces (with dedup) using lending iterators —
        // SmallVec keeps the dedup set off the heap for typical valences.
        let mut seen: smallvec::SmallVec<[FaceKey; 8]> = smallvec::SmallVec::new();
        let mut disk = self.disk_edges(v);
        while let Some(e) = disk.next(self) {
            let mut radial = self.radial_loops(e);
            while let Some(l) = radial.next(self) {
                let f = self.loops.get(l).face;
                if !seen.contains(&f) {
                    seen.push(f);
                }
            }
        }
        if seen.is_empty() {
            self.set_vertex_normal(v, Vec3::Z);
            return;
        }
        let mut acc = Vec3::ZERO;
        for &f in &seen {
            let area = self.face_calc_area(f);
            let n_f = self.faces.get(f).normal;
            acc += area * n_f;
        }
        let n = if acc.length_squared() > 0.0 {
            acc.normalize()
        } else {
            Vec3::Z
        };
        self.set_vertex_normal(v, n);
    }

    /// Recompute every face normal, then every vertex normal. This is the
    /// canonical "freshen all cached normals" entry point.
    pub fn mesh_normals_update(&mut self) {
        let mut it_f = self.mesh_faces();
        while let Some(f) = it_f.next(self) {
            self.face_calc_normal(f);
        }
        let mut it_v = self.mesh_verts();
        while let Some(v) = it_v.next(self) {
            self.vert_normal_update(v);
        }
    }

    /// Updates the `edge` to connect to a `new` vertex instead of some `existing` vertex.
    /// Updates all associated (radial) face loops to also reference the `new` vertex.
    ///
    /// Assumes that the existing vertex is currently associated with the edge
    /// Assumes that the new vertex is not currently associated with edge
    ///
    /// Beware: This function will lead to inconsistent face loops that need to be repaired!
    ///
    /// Loops that reference the existing vertex and a different edge will end up
    /// referencing the new vertex which probably doesn't belong to the edge associated
    /// with that loop. For example...
    ///
    /// ```pre
    ///      E1             E2           E3
    /// V1 ======= V2 ============== V3 ===== V4 : EDGES
    /// e1v1-----> e2v2------------> e3v3----->  : LOOPS
    ///
    /// might become....
    ///      E1         E2                E3
    /// V1 ======= V2 ===== V5       V3 ===== V4 : EDGES
    /// e1v1-----> e2v2---> e3v5-------------->  : LOOPS
    ///                      ^__ invalid loop
    ///
    /// Or:
    ///      E1             E2           E3
    /// V1 ======= V2 ============== V3 ===== V4 : EDGES
    /// <--------e1v2 <------------e2v3 <---e3v4 : LOOPS
    ///
    /// might become....
    ///      E1         E2                E3
    /// V1 ======= V2 ===== V5       V3 ===== V4 ...
    /// <--------e1v2 <---e2v5 <------------e3v4
    ///                                      ^
    ///                               invalid loop
    /// ```
    ///
    /// like: bmesh_edge_vert_swap (but swapped dst/src arguments)
    fn edge_replace_vertex_raw(&mut self, edge: EdgeKey, existing: VertexKey, new: VertexKey) {
        self.vertex_disk_remove_edge(existing, edge);

        let slot = if self.edges.get(edge).vertices[0] == existing {
            0
        } else {
            1
        };
        self.edges.get_mut(edge).vertices[slot] = new;
        self.vertex_disk_append_edge(new, edge);

        // Iterate the radial cycle of face loops for the edge and also make sure to replace
        // the same vertex for face loops associated with this edge...
        //
        if let Some(first) = self.edges.get(edge).radial_link {
            let mut cur = first;
            loop {
                let links = self.loops.get(cur).edge_radial_links;

                // For loops whose perimeter span (loop.vertex -> loop.next.vertex)
                // crosses `existing`, swap to `new`. We tolerate "neither side
                // matches" silently — this happens during multi-step Euler
                // operations (notably JVKE) where some loops were already
                // patched in an earlier step.
                if self.loops.get(cur).vertex == existing {
                    self.loops.get_mut(cur).vertex = new;
                } else {
                    let loop_next = self.loops.get(cur).face_loop_links.next;
                    if self.loops.get(loop_next).vertex == existing {
                        self.loops.get_mut(loop_next).vertex = new;
                    }
                }

                cur = links.next;
                if cur == first {
                    break;
                }
            }
        }
    }

    /// Returns the loop in `face`'s cycle whose edge is `e`, if any.
    pub fn face_edge_share_loop(&self, face: FaceKey, e: EdgeKey) -> Option<LoopKey> {
        let first = self.faces.get(face).first_edge_loop;
        for l in self.face_loop_iter(first) {
            if self.loops.get(l).edge == e {
                return Some(l);
            }
        }
        None
    }

    /// Allocates an empty face with the given placeholder first_edge_loop,
    /// resizing face attribute storage. Caller is responsible for patching
    /// the face fields (len, first_edge_loop, etc.) once topology is settled.
    fn alloc_empty_face(&mut self, placeholder_first_loop: LoopKey) -> FaceKey {
        self.flags |= LoopMeshFlags::FACES_INDEX_DIRTY;
        let entry = self.faces.vacant_entry();
        let key = FaceKey(entry.key() as u32);
        entry.insert(Face {
            eflags: ElementFlags::NONE,
            key,
            index: 0,
            len: 0,
            first_edge_loop: placeholder_first_loop,
            normal: Vec3::ZERO,
        });
        let key_watermark = self.faces.key_watermark();
        for attrib_vec in self.face_attribs.attributes_u8.iter_mut() {
            attrib_vec.resize(key_watermark, 0);
        }
        for attrib_vec in self.face_attribs.attributes_i32.iter_mut() {
            attrib_vec.resize(key_watermark, 0);
        }
        for attrib_vec in self.face_attribs.attributes_u32.iter_mut() {
            attrib_vec.resize(key_watermark, 0);
        }
        for attrib_vec in self.face_attribs.attributes_f32.iter_mut() {
            attrib_vec.resize(key_watermark, 0.0);
        }
        for attrib_vec in self.face_attribs.attributes_vec2.iter_mut() {
            attrib_vec.resize(key_watermark, [0.0; 2]);
        }
        for attrib_vec in self.face_attribs.attributes_vec3.iter_mut() {
            attrib_vec.resize(key_watermark, [0.0; 3]);
        }
        for attrib_vec in self.face_attribs.attributes_vec4.iter_mut() {
            attrib_vec.resize(key_watermark, [0.0; 4]);
        }
        for attrib_vec in self.face_attribs.attributes_string.iter_mut() {
            attrib_vec.resize(key_watermark, None);
        }
        self.op_flags.grow_face_layers(key_watermark);
        key
    }

    pub fn query_vertex_shared_face(&self, v0: VertexKey, v1: VertexKey) -> Option<FaceKey> {
        for e0 in self.vertex_disk_iter(v0) {
            if let Some(loopi0) = self.edges.get(e0).radial_link {
                for l0 in self.face_loop_iter(loopi0) {
                    let face0 = self.loops.get(l0).face;

                    for e1 in self.vertex_disk_iter(v1) {
                        if let Some(loopi1) = self.edges.get(e1).radial_link {
                            for l1 in self.face_loop_iter(loopi1) {
                                let face1 = self.loops.get(l1).face;

                                if face0 == face1 {
                                    return Some(face0);
                                }
                            }
                        }
                    }
                }
            }
        }

        None
    }

    /// Merges the edge disks from a `src` vertex into a `dst` vertex and then kills the
    /// `src` vertex.
    ///
    /// The vertices must not belong to the same edge or be part of the same face.
    pub fn vertex_merge_disks(&mut self, dst: VertexKey, src: VertexKey) {
        debug_assert!(self.query_edge_exists(dst, src).is_none());
        debug_assert!(self.query_vertex_shared_face(dst, src).is_none());

        if src == dst {
            return;
        }

        loop {
            let loop_start = self.vertices.get(dst).disk_loop_start;
            match loop_start {
                Some(edge) => {
                    self.edge_replace_vertex_raw(edge, src, dst);
                }
                None => break,
            }
        }

        self.vertex_kill(src);
    }

    /// Checks whether merging the two vertices (via `vertex_merge_disks`) would result in
    /// a double edge.
    ///
    /// The vertices must not belong to the same edge or be part of the same face.
    ///
    pub fn vertex_merge_disks_check_for_double_edges(
        &mut self,
        dst: VertexKey,
        src: VertexKey,
    ) -> bool {
        debug_assert!(self.query_edge_exists(dst, src).is_none());
        debug_assert!(self.query_vertex_shared_face(dst, src).is_none());

        // FIXME: iterate the disk for each src/dst vertex just _once_ by tagging the 'other'
        // vertices during the iteration of the first vertex and then just check for the
        // flag when iterating the other vertices in the disk of the second vertex.

        for e_dst in self.vertex_disk_iter(dst) {
            let other_dst = self.edges.get(e_dst).other_vertex(dst);

            for e_src in self.vertex_disk_iter(src) {
                let other_src = self.edges.get(e_src).other_vertex(src);

                if other_src == other_dst {
                    return true;
                }
            }
        }

        false
    }

    pub fn face_kill(&mut self, face: FaceKey) {
        // NB: We can't use a regular iterator since that would hold a shared reference
        // to our mesh, stopping us from disconnecting and freeing the loops as we go...
        let first = self.faces.get(face).first_edge_loop;
        let mut cur = first;
        loop {
            let links = self.loops.get(cur).face_loop_links;

            self.face_loop_disconnect_edge(cur);
            self.face_loop_data_remove(cur);

            cur = links.next;
            if cur == first {
                break;
            }
        }

        self.remove_face_slab_entry(face);
    }
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
    let mut edges = vec![];
    for i in 0..(verts.len() - 1) {
        edges.push(mesh.create_edge([verts[i], verts[i + 1]], EdgeCreateFlags::NONE));
    }
    edges.push(mesh.create_edge([verts[verts.len() - 1], verts[0]], EdgeCreateFlags::NONE));

    mesh.face_create(&verts, &edges, FaceCreateFlags::NONE)
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
            Format::new(ComponentType::I32, 1),
        ))
        .unwrap();

    assert!(mesh.vertex_attribs_mut().set_i32(material, v0.as_usize(), 7));
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
            Format::new(ComponentType::I32, 1),
        ))
        .unwrap();

    let loops: Vec<LoopKey> = mesh.iter_face_loops(f).collect();
    for (i, l) in loops.iter().enumerate() {
        assert!(mesh.loop_attribs_mut().set_i32(tag, l.as_usize(), i as i32 * 10));
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
            Format::new(ComponentType::F32, 3),
        ))
        .unwrap();
    assert!(!mesh.vertex_attribs_mut().set_i32(color, v0.as_usize(), 1));
    assert_eq!(mesh.vertex_attribs().get_i32(color, v0.as_usize()), None);

    // An unregistered key resolves to nothing.
    let bogus = ElementAttributeKey(9999);
    assert!(!mesh.vertex_attribs_mut().set_i32(bogus, v0.as_usize(), 1));
    assert_eq!(mesh.vertex_attribs().get_i32(bogus, v0.as_usize()), None);
}
