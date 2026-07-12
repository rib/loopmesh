use super::*;

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

pub struct RadialLoopIterator<'a> {
    mesh: &'a LoopMesh,

    axis_edge: EdgeKey,
    first_loop: Option<LoopKey>,
    next_loop: Option<LoopKey>,
}

impl<'a> RadialLoopIterator<'a> {
    pub fn new(mesh: &'a LoopMesh, axis_edge: EdgeKey) -> Self {
        // §6.2: the radial traversal starts at the ENTRY, not at the stored
        // head. Seeding from `radial_link` yields the same cycle rotated by
        // one, and every consumer that takes the first element — or re-appends
        // in walk order — inherits the rotation.
        let entry = mesh.edge_radial_entry(axis_edge);
        RadialLoopIterator {
            mesh,
            axis_edge,
            first_loop: entry,
            next_loop: entry,
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
    pub(crate) fn new(mesh: &LoopMesh, v: VertexKey) -> Self {
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

/// Cycle iterator over the loops on an edge's radial fan, in the traversal
/// order of
/// [`RADIAL_EDGE_STRUCTURE.md`](../../../radial-mesh-guides/RADIAL_EDGE_STRUCTURE.md)
/// §6.2: forward from the edge's **radial entry**
/// ([`LoopMesh::edge_radial_entry`]), never from the stored head.
pub struct RadialLoops {
    axis: EdgeKey,
    first: Option<LoopKey>,
    next: Option<LoopKey>,
}

impl RadialLoops {
    pub(crate) fn new(mesh: &LoopMesh, e: EdgeKey) -> Self {
        let start = mesh.edge_radial_entry(e);
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

/// Cycle iterator over a vertex's **face-corner fan**
/// ([`RADIAL_EDGE_STRUCTURE.md`](../../../radial-mesh-guides/RADIAL_EDGE_STRUCTURE.md)
/// §6.5): the loop of every incident face whose vertex is the hub, each face
/// yielded exactly once, in the order that section pins.
///
/// The order is **observable**, so this is not interchangeable with a
/// disk-major / radial-minor walk filtered on the corner vertex. Two things
/// separate them, and a consumer that keeps a running best-so-far under a strict
/// comparison (subdivide's fewest-corners carrier search) can see both:
///
/// - the fan groups faces by their **corner edge** at the hub — the edge the
///   face traverses *away* from the hub, the one carrying the face's only loop
///   whose vertex is the hub. A face has exactly one, so it is reached once, not
///   once per incident edge; no de-duplication is needed or performed.
/// - the group sequence starts at the **first face's** corner edge, not at the
///   disk-seed edge: the seed's radial entry may sit at the far endpoint, and
///   corner-adjusting it onto the hub can land on the face's *other* edge. The
///   seed edge's own group is then reached last, on the disk wrap.
pub struct VertFanLoops {
    hub: VertexKey,
    /// The first face's corner edge — where the group sequence starts and, on
    /// the disk wrap, ends. `None` for a hub with no incident face.
    start_edge: Option<EdgeKey>,
    /// The corner edge whose radial group is currently draining.
    group_edge: Option<EdgeKey>,
    /// The loop the current group's radial walk began at (the wrap sentinel).
    group_first: Option<LoopKey>,
    /// The next loop to examine in the current group's radial walk.
    next_loop: Option<LoopKey>,
}

impl VertFanLoops {
    pub(crate) fn new(mesh: &LoopMesh, v: VertexKey) -> Self {
        let empty = Self {
            hub: v,
            start_edge: None,
            group_edge: None,
            group_first: None,
            next_loop: None,
        };
        let Some(anchor) = mesh.vertices.get(v).disk_loop_start else {
            return empty;
        };

        // The seed is the first non-wire edge walking the disk forward from its
        // anchor (the first-attached edge).
        let mut seed = None;
        let mut edge = anchor;
        loop {
            if mesh.edges.get(edge).radial_link.is_some() {
                seed = Some(edge);
                break;
            }
            let next = mesh.edges.get(edge).disk_loop_links_for_vert(v).next;
            if next == anchor {
                break;
            }
            edge = next;
        }
        let Some(seed) = seed else {
            return empty;
        };

        let entry = mesh
            .edge_radial_entry(seed)
            .expect("a non-wire edge carries a radial entry");
        // Corner-adjust the seed's radial entry onto the hub: a loop's vertex is
        // the tail of its edge, so the entry sits either at the hub already or
        // exactly one corner before it.
        let first = if mesh.loops.get(entry).vertex == v {
            entry
        } else {
            mesh.loops.get(entry).face_loop_links.next
        };
        debug_assert_eq!(
            mesh.loops.get(first).vertex,
            v,
            "the corner-adjust hop lands on the hub"
        );
        let corner_edge = mesh.loops.get(first).edge;

        Self {
            hub: v,
            start_edge: Some(corner_edge),
            group_edge: Some(corner_edge),
            group_first: Some(first),
            next_loop: Some(first),
        }
    }
}

impl MeshIter for VertFanLoops {
    type Item = LoopKey;
    fn next(&mut self, mesh: &LoopMesh) -> Option<LoopKey> {
        loop {
            // Drain the current corner edge's radial group, skipping the loops
            // whose vertex is the far endpoint — those belong to faces whose
            // corner at the hub lives on some other edge, and are yielded there.
            while let Some(current) = self.next_loop {
                let next = mesh.loops.get(current).edge_radial_links.next;
                self.next_loop = (Some(next) != self.group_first).then_some(next);
                if mesh.loops.get(current).vertex == self.hub {
                    return Some(current);
                }
            }

            let group = self.group_edge?;
            let next_edge = mesh
                .edges
                .get(group)
                .disk_loop_links_for_vert(self.hub)
                .next;
            if Some(next_edge) == self.start_edge {
                self.group_edge = None;
                return None;
            }
            self.group_edge = Some(next_edge);
            // A later group starts at its edge's radial entry — there is no
            // corner-adjust hop here; that happens only for the seed.
            self.group_first = mesh.edge_radial_entry(next_edge);
            self.next_loop = self.group_first;
        }
    }
}

/// Cycle iterator over the loops of a face (forward winding).
pub struct FaceLoops {
    first: Option<LoopKey>,
    next: Option<LoopKey>,
}

impl FaceLoops {
    pub(crate) fn new(mesh: &LoopMesh, f: FaceKey) -> Self {
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
    pub(crate) fn new(mesh: &LoopMesh, f: FaceKey) -> Self {
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

impl LoopMesh {
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

    /// Lending iterator over the edges in `v`'s disk cycle.
    pub fn disk_edges(&self, v: VertexKey) -> DiskEdges {
        DiskEdges::new(self, v)
    }

    /// Lending iterator over the loops in `e`'s radial cycle.
    pub fn radial_loops(&self, e: EdgeKey) -> RadialLoops {
        RadialLoops::new(self, e)
    }

    /// Lending iterator over `v`'s face-corner fan (§6.5): one loop per incident
    /// face, in corner-edge-major / radial-minor order. Order-sensitive; see
    /// [`VertFanLoops`].
    pub fn vert_fan_loops(&self, v: VertexKey) -> VertFanLoops {
        VertFanLoops::new(self, v)
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
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A non-manifold "book" of `n` triangular pages hinged on the spine edge
    /// `v0–v1`: page `i` is wound `[v0, v1, tip_i]`, so every page adopts the
    /// spine and its radial cycle has valence `n`.
    fn build_book(mesh: &mut LoopMesh, n: usize) -> (VertexKey, VertexKey, Vec<FaceKey>) {
        let v0 = mesh.create_vertex(Vec3::new(0.0, 0.0, 0.0));
        let v1 = mesh.create_vertex(Vec3::new(0.0, 1.0, 0.0));
        let faces = (0..n)
            .map(|i| {
                let tip = mesh.create_vertex(Vec3::new(1.0 + i as f32, 0.5, 0.0));
                mesh.face_create_verts(&[v0, v1, tip], FaceCreateFlags::NONE)
            })
            .collect();
        (v0, v1, faces)
    }

    fn fan_faces(mesh: &LoopMesh, v: VertexKey) -> Vec<FaceKey> {
        let mut fan = mesh.vert_fan_loops(v);
        let mut faces = Vec::new();
        while let Some(l) = fan.next(mesh) {
            assert_eq!(mesh.loop_vertex(l), v, "every fan loop is a corner at v");
            faces.push(mesh.loop_face(l));
        }
        faces
    }

    /// RADIAL_EDGE_STRUCTURE.md §6.5 worked example B — the `book(n)` spine pair
    /// discriminates the radial pick. Both spine verts share *every* page, yet:
    ///
    /// - `v0` anchors on its private spoke `v0–tip0` (face 0's closing edge,
    ///   radial length 1) → build-face 0, whatever the pick.
    /// - `v1` anchors on the **spine**, adopted by all `n` pages, so its first
    ///   corner is the spine's **last-attached** page — build-face `n-1`.
    ///
    /// A fan that read the *first*-attached loop would answer build-face 0 at
    /// `v1` too. This is the pin for that half of the radial-anchor pair; it
    /// fails on its own if the fan's radial pick regresses to the stored head.
    #[test]
    fn vert_fan_first_loop_book_spine_pair_is_last_attached() {
        let n = 5usize;
        let mut mesh = LoopMesh::new();
        let (v0, v1, faces) = build_book(&mut mesh, n);

        assert_eq!(fan_faces(&mesh, v0)[0], faces[0]);
        assert_eq!(
            fan_faces(&mesh, v1)[0],
            faces[n - 1],
            "the spine's radial pick is its last-attached page, not its first"
        );
    }

    /// The fan yields each incident face exactly once — once *per face*, not
    /// once per (face, incident-edge) pair (§6.5, "Composing the fan").
    #[test]
    fn vert_fan_yields_each_incident_face_once() {
        let n = 4usize;
        let mut mesh = LoopMesh::new();
        let (v0, v1, faces) = build_book(&mut mesh, n);

        for hub in [v0, v1] {
            let fanned = fan_faces(&mesh, hub);
            assert_eq!(fanned.len(), n, "one corner per incident face");
            let mut seen: Vec<usize> = fanned.iter().map(|f| f.as_usize()).collect();
            seen.sort_unstable();
            let mut expected: Vec<usize> = faces.iter().map(|f| f.as_usize()).collect();
            expected.sort_unstable();
            assert_eq!(seen, expected);
        }
    }

    /// A vertex with no incident face fans empty (a lone wire endpoint).
    #[test]
    fn vert_fan_of_wire_endpoint_is_empty() {
        let mut mesh = LoopMesh::new();
        let a = mesh.create_vertex(Vec3::ZERO);
        let b = mesh.create_vertex(Vec3::X);
        mesh.create_edge([a, b], EdgeCreateFlags::NONE);
        assert!(fan_faces(&mesh, a).is_empty());
    }
}
