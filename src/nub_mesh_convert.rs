//! Conversions between [`nub_mesh::Mesh`] (GPU-shaped) and
//! [`crate::loopmesh::LoopMesh`] (bmesh-shaped editing structure).
//!
//! The two structures encode different things:
//! - `nub_mesh::Mesh` is a packed, GPU-upload-ready snapshot — vertex N lives
//!   at `base + N * stride` in a byte buffer, attributes are described by
//!   `(offset, stride, Format, AttributeRate)`, topology is a flat index buffer.
//! - `LoopMesh` is a sparse editing structure — vertices/edges/loops/faces
//!   live in slabs, topology is implicit in the radial/loop link fields, and
//!   per-element attributes live in typed buckets keyed by slab index.
//!
//! ## Import (`from_nub_mesh`)
//!
//! Triangle topologies only. Every source vertex becomes one `LoopMesh`
//! vertex; every `PerVertex` float-decodable attribute (Position is consumed
//! into `Vertex::position` directly; the rest land in `vertex_attribs`).
//! Integer-decodable attributes (e.g. `JointIndex`) are skipped today.
//!
//! ## Export (`to_nub_mesh`)
//!
//! Faces triangulate by fan-from-first-loop. Each numeric vertex attribute
//! emits its own packed `[f32; N]` buffer.
//!
//! `loop_attribs` participate in the export through a *seam-split* pass: for
//! each source vertex, distinct loop-attribute tuples among its corners cause
//! that vertex to be duplicated in the packed output (canonical glTF
//! behaviour, default [`SeamSplitPolicy::Split`]).
//!
//! `edge_attribs`, `face_attribs`, and `String`-kind attributes have no
//! representation in `nub_mesh::Mesh` and are not exported.

use anyhow::{anyhow, Context, Result};
use glam::Vec3;
use nub_mesh::attribute::{AttributeRate, AttributeSemantic};
use nub_mesh::walk::LendingIterator;
use nub_mesh::{
    Attribute, AttributeKey, Buffer, ComponentType, Format, IndexBuffer, IndexFormat, Mesh,
    PrimitiveTopology,
};

use crate::loopmesh::{
    AttributeStorage, EdgeCreateFlags, ElementAttributeDesc, ElementAttributeKey,
    ElementAttributeKind, ElementKey, FaceCreateFlags, LoopKey, LoopMesh, MeshIter, VertexKey,
};

// ============================== Options ==============================

#[derive(Clone, Debug)]
pub struct FromNubMeshOptions {
    /// Abort if the source mesh has no `Position` attribute. When `false`
    /// vertices are placed at the origin. Default: true.
    pub require_position: bool,
}

impl Default for FromNubMeshOptions {
    fn default() -> Self {
        Self {
            require_position: true,
        }
    }
}

#[derive(Clone, Debug)]
pub struct ToNubMeshOptions {
    /// How to resolve loop attributes when a single source vertex has
    /// corners that disagree. Default `Split` matches glTF / OBJ semantics.
    pub seam_split: SeamSplitPolicy,

    /// Which primitive topology to emit. The packed-vertex set is shared
    /// across all three; only the index buffer differs.
    ///
    /// - `TriangleList` (default): tessellate each face per `triangulation`.
    /// - `LineList`: one line per `EdgeKey`, controlled by `wireframe`.
    /// - `PointList`: one index per source vertex.
    pub topology: PrimitiveTopology,

    /// How to triangulate faces with more than three loops when emitting
    /// `TriangleList`. See [`TriangulationPolicy`].
    pub triangulation: TriangulationPolicy,

    /// How to emit lines when `topology == LineList`. See [`WireframePolicy`].
    pub wireframe: WireframePolicy,
}

impl Default for ToNubMeshOptions {
    fn default() -> Self {
        Self {
            seam_split: SeamSplitPolicy::Split,
            topology: PrimitiveTopology::TriangleList,
            triangulation: TriangulationPolicy::EarClip,
            wireframe: WireframePolicy::EdgesOnce,
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub enum SeamSplitPolicy {
    /// Duplicate the source vertex once per distinct corner-attribute tuple.
    Split,
    /// Use the value from the first encountered corner (lossy).
    Merge,
    /// Reject the conversion if any vertex has differing corner attributes.
    Error,
}

/// How to break an n-gon face into triangles when emitting a
/// [`PrimitiveTopology::TriangleList`].
///
/// Algorithm details for each variant are documented in
/// `radial-mesh-guides/BMESH_POLYGON_TRIANGULATION.md`.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum TriangulationPolicy {
    /// Fan from the first loop of each face. Cheap and correct for convex
    /// faces; produces flipped or inverted triangles on concave quads and
    /// arbitrary n-gons. Preserved as a fallback for byte-compatibility
    /// with the original `to_nub_mesh` output.
    Fan,
    /// Ear-clipping with quad-flip detection, matching the render-time
    /// path described in `BMESH_POLYGON_TRIANGULATION.md` § 5.1 (quad) and
    /// § 7 (n-gon). Non-destructive, no beautify pass. This is what BMesh's
    /// equivalent render-time tessellation produces. Default.
    EarClip,
}

/// How to emit lines when [`ToNubMeshOptions::topology`] is
/// [`PrimitiveTopology::LineList`].
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum WireframePolicy {
    /// Emit one line per `EdgeKey`. When seam-split has duplicated a
    /// source vertex, either side's packed index is chosen arbitrarily —
    /// fine for wireframe overlay rendering where a one-pixel seam
    /// inconsistency is invisible.
    EdgesOnce,
}

// ============================ from_nub_mesh ============================

impl LoopMesh {
    /// Build a [`LoopMesh`] from a triangulated [`nub_mesh::Mesh`].
    pub fn from_nub_mesh(mesh: &Mesh) -> Result<Self> {
        Self::from_nub_mesh_with(mesh, &FromNubMeshOptions::default())
    }

    pub fn from_nub_mesh_with(mesh: &Mesh, opts: &FromNubMeshOptions) -> Result<Self> {
        let topology = mesh.topology();
        if !topology.is_triangles() {
            return Err(anyhow!(
                "from_nub_mesh requires a triangle topology; got {:?}",
                topology
            ));
        }

        let mut out = LoopMesh::new();
        let vertex_count = mesh.vertex_count();

        // ---- Step 1: positions → Vertex::position ----
        let pos_id = mesh.find_semantic(AttributeSemantic::Position);
        if pos_id.is_none() && opts.require_position {
            return Err(anyhow!("source mesh has no Position attribute"));
        }

        let mut src_to_vk: Vec<VertexKey> = Vec::with_capacity(vertex_count);
        if pos_id.is_some() {
            let mut it = mesh
                .walk()
                .read_f32x3(AttributeSemantic::Position, [0.0, 0.0, 0.0, 1.0])
                .context("declaring Position read")?
                .vertices()
                .context("building vertices walker")?;
            while let Some(cur) = it.next() {
                let p = cur.read_f32x3(0);
                src_to_vk.push(out.create_vertex(Vec3::new(p[0], p[1], p[2])));
            }
        } else {
            for _ in 0..vertex_count {
                src_to_vk.push(out.create_vertex(Vec3::ZERO));
            }
        }

        // ---- Step 2: other PerVertex float-decodable attributes ----
        // Collect the transferable list first so the borrow on `mesh` from
        // `attributes()` is released before we build the walker.
        struct Transfer {
            key: AttributeKey,
            n_components: u8,
        }
        let mut transferable: Vec<Transfer> = Vec::new();
        for attr in mesh.attributes() {
            if attr.rate != AttributeRate::PerVertex {
                continue;
            }
            if matches!(
                &attr.key,
                AttributeKey::Semantic(AttributeSemantic::Position)
            ) {
                continue;
            }
            if !attr.format.decodes_as_float() {
                // TODO: support integer-decode attributes (JointIndex etc.)
                // once loopmesh grows u32xN buckets.
                continue;
            }
            // Clamp into the F32x{1..=4} space loopmesh's buckets handle.
            let n = attr.format.components.clamp(1, 4);
            transferable.push(Transfer {
                key: attr.key.clone(),
                n_components: n,
            });
        }

        if !transferable.is_empty() {
            // Register the matching vertex attributes on loopmesh.
            let mut element_keys: Vec<ElementAttributeKey> = Vec::with_capacity(transferable.len());
            for t in &transferable {
                let lm_format = Format::new(ComponentType::F32, t.n_components);
                let key = out
                    .register_vertex_attribute(ElementAttributeDesc::numeric(
                        t.key.clone(),
                        lm_format,
                    ))
                    .with_context(|| format!("registering vertex attribute {:?}", t.key))?;
                element_keys.push(key);
            }

            // Build a single walk that reads all transferable attributes
            // (each as f32x4 with zero fill; we re-slice into the right bucket).
            let default = [0.0_f32, 0.0, 0.0, 0.0];
            let mut builder = mesh.walk();
            for t in &transferable {
                builder = builder
                    .read_f32x4(t.key.clone(), default)
                    .with_context(|| format!("declaring read of {:?}", t.key))?;
            }
            let mut it = builder.vertices().context("building vertices walker")?;
            let mut src_idx = 0usize;
            while let Some(cur) = it.next() {
                let vk = src_to_vk[src_idx];
                let slot = vk.as_usize();
                for (i, t) in transferable.iter().enumerate() {
                    let v4 = cur.read_f32x4(i);
                    let elem_key = element_keys[i];
                    let attrs = out.vertex_attribs_mut();
                    match t.n_components {
                        1 => {
                            attrs.set_f32(elem_key, slot, v4[0]);
                        }
                        2 => {
                            attrs.set_vec2(elem_key, slot, [v4[0], v4[1]]);
                        }
                        3 => {
                            attrs.set_vec3(elem_key, slot, [v4[0], v4[1], v4[2]]);
                        }
                        4 => {
                            attrs.set_vec4(elem_key, slot, v4);
                        }
                        _ => unreachable!(),
                    }
                }
                src_idx += 1;
            }
        }

        // ---- Step 3: topology (triangles → edges + faces) ----
        let indices: Vec<u32> = match mesh.index_iter() {
            Some(iter) => iter.collect(),
            None => (0..vertex_count as u32).collect(),
        };

        let mut i = 0;
        while i + 2 < indices.len() {
            let (i0, i1, i2) = (
                indices[i] as usize,
                indices[i + 1] as usize,
                indices[i + 2] as usize,
            );
            i += 3;
            if i0 == i1 || i1 == i2 || i0 == i2 {
                continue;
            }
            if i0 >= src_to_vk.len() || i1 >= src_to_vk.len() || i2 >= src_to_vk.len() {
                continue;
            }
            let v0 = src_to_vk[i0];
            let v1 = src_to_vk[i1];
            let v2 = src_to_vk[i2];
            let e0 = out.create_edge([v0, v1], EdgeCreateFlags::NO_DUPLICATE);
            let e1 = out.create_edge([v1, v2], EdgeCreateFlags::NO_DUPLICATE);
            let e2 = out.create_edge([v2, v0], EdgeCreateFlags::NO_DUPLICATE);
            out.face_create(&[v0, v1, v2], &[e0, e1, e2], FaceCreateFlags::NO_DUPLICATE);
        }

        Ok(out)
    }
}

// ============================ to_nub_mesh ==============================

impl LoopMesh {
    /// Export a packed [`nub_mesh::Mesh`] view of this loopmesh.
    pub fn to_nub_mesh(&mut self) -> Result<Mesh> {
        self.to_nub_mesh_with(&ToNubMeshOptions::default())
    }

    pub fn to_nub_mesh_with(&mut self, opts: &ToNubMeshOptions) -> Result<Mesh> {
        // TODO: parallelize via rayon once feature work has settled. Natural
        // fan-out points: position fill, the inner loops of
        // `pack_attribute_for_packed`, the across-attributes loop in
        // `emit_attribute_set`, and the triangle-index fill (prefix-sum
        // tri counts, then per-face write into disjoint slots). `&LoopMesh`
        // is already `Sync` (no interior mutability), so no storage changes
        // are required — only this file needs to be touched.
        self.vertices_ensure_table();

        let mut export = Export::new(self, opts)?;
        export.write_position()?;
        export.write_vertex_attributes()?;
        export.write_loop_attributes()?;
        match opts.topology {
            PrimitiveTopology::TriangleList => export.write_triangle_indices(opts)?,
            PrimitiveTopology::LineList => export.write_line_indices(opts)?,
            PrimitiveTopology::PointList => export.write_point_indices()?,
            other => {
                return Err(anyhow!(
                    "to_nub_mesh: unsupported topology {:?} (only TriangleList, \
                     LineList, PointList are emitted)",
                    other
                ));
            }
        }
        Ok(export.finish())
    }
}

/// Builder for one `to_nub_mesh` invocation. Owns the in-progress output
/// mesh and the packed-vertex resolution shared across all three topology
/// branches.
struct Export<'m> {
    mesh: &'m LoopMesh,
    packed: Vec<PackedVertex>,
    /// Indexed by `LoopKey::as_usize()`. `u32::MAX` for slab slots that
    /// have no loop in any face.
    loop_to_packed: Vec<u32>,
    /// Indexed by `VertexKey::as_usize()`. `u32::MAX` for unused slab
    /// slots. In the seam-split case, one source vertex may produce
    /// multiple packed vertices — this stores the first one assigned, used
    /// for wire-edge endpoint resolution when emitting `LineList`.
    vert_to_packed: Vec<u32>,
    out: Mesh,
}

impl<'m> Export<'m> {
    /// Phase 1 (packed-vertex resolution) is folded into construction so
    /// the output `Mesh` can be created with the final vertex count
    /// directly. Each "packed vertex" carries a (source_vert, source_loop)
    /// pair — source_vert for position / vertex_attribs, source_loop for
    /// loop_attribs. When `loop_attribs` is empty every corner of a
    /// source vertex collapses to one packed vertex (the fast path); when
    /// non-empty, `opts.seam_split` decides what happens when two corners
    /// disagree.
    fn new(mesh: &'m LoopMesh, opts: &ToNubMeshOptions) -> Result<Self> {
        let has_loop_attribs = !mesh.loop_attribs().is_empty();
        let resolved = if has_loop_attribs {
            build_packed_seam_split(mesh, opts.seam_split)?
        } else {
            build_packed_no_seam(mesh)
        };
        let out = Mesh::new(opts.topology, resolved.packed.len());
        Ok(Self {
            mesh,
            packed: resolved.packed,
            loop_to_packed: resolved.loop_to_packed,
            vert_to_packed: resolved.vert_to_packed,
            out,
        })
    }

    fn finish(self) -> Mesh {
        self.out
    }

    fn packed_count(&self) -> usize {
        self.packed.len()
    }

    fn write_position(&mut self) -> Result<()> {
        let n = self.packed_count();
        let mut pos_bytes = vec![0u8; n * core::mem::size_of::<[f32; 3]>()];
        {
            let positions: &mut [[f32; 3]] = bytemuck::cast_slice_mut(&mut pos_bytes);
            for (i, pv) in self.packed.iter().enumerate() {
                let p = self.mesh.vertex_position(pv.source_vert);
                positions[i] = [p.x, p.y, p.z];
            }
        }
        let pos_buf = self.out.add_buffer(Buffer::from_bytes(&pos_bytes));
        self.out
            .add_attribute(Attribute {
                key: AttributeKey::Semantic(AttributeSemantic::Position),
                buffer: pos_buf,
                offset: 0,
                stride: core::mem::size_of::<[f32; 3]>(),
                format: Format::new(ComponentType::F32, 3),
                rate: AttributeRate::PerVertex,
            })
            .context("adding Position attribute")?;
        Ok(())
    }

    fn write_vertex_attributes(&mut self) -> Result<()> {
        emit_attribute_set(
            self.mesh.vertex_attribs(),
            &self.packed,
            |pv| pv.source_vert.as_usize(),
            &mut self.out,
            /*skip_position=*/ true,
        )
    }

    fn write_loop_attributes(&mut self) -> Result<()> {
        emit_attribute_set(
            self.mesh.loop_attribs(),
            &self.packed,
            |pv| pv.source_loop.as_usize(),
            &mut self.out,
            /*skip_position=*/ false,
        )
    }

    fn write_triangle_indices(&mut self, opts: &ToNubMeshOptions) -> Result<()> {
        let mut tri_indices: Vec<u32> = Vec::with_capacity(self.mesh.totface() * 3);
        let mut scratch_loops: Vec<LoopKey> = Vec::with_capacity(8);
        let mut scratch_proj: Vec<glam::Vec2> = Vec::with_capacity(8);
        let mut tri_buf: Vec<[LoopKey; 3]> = Vec::with_capacity(8);

        let mut it = self.mesh.mesh_faces();
        while let Some(f) = it.next(self.mesh) {
            tri_buf.clear();
            crate::triangulate_corners::corner_triangulation(
                self.mesh,
                f,
                opts.triangulation,
                &mut scratch_loops,
                &mut scratch_proj,
                &mut tri_buf,
            );
            for &[a, b, c] in &tri_buf {
                tri_indices.push(self.loop_to_packed[a.as_usize()]);
                tri_indices.push(self.loop_to_packed[b.as_usize()]);
                tri_indices.push(self.loop_to_packed[c.as_usize()]);
            }
        }
        self.emit_index_buffer(&tri_indices)
    }

    fn write_line_indices(&mut self, opts: &ToNubMeshOptions) -> Result<()> {
        // `WireframePolicy::EdgesOnce`: one line per `EdgeKey`. With
        // seam-split, a single source vertex may map to multiple packed
        // vertices — we resolve via `vert_to_packed`, which stores the
        // first packed index assigned to each source vert. That's
        // arbitrary but stable; the alternative (PerCorner) would
        // double-emit at seams.
        match opts.wireframe {
            WireframePolicy::EdgesOnce => {}
        }
        let mut line_indices: Vec<u32> = Vec::with_capacity(self.mesh.totedge() * 2);
        let mut it = self.mesh.mesh_edges();
        while let Some(e) = it.next(self.mesh) {
            let [v_a, v_b] = self.mesh.edge_vert_keys(e);
            let pa = self.vert_to_packed[v_a.as_usize()];
            let pb = self.vert_to_packed[v_b.as_usize()];
            if pa == u32::MAX || pb == u32::MAX {
                // Endpoint isn't reachable via any face loop (genuine
                // wire edge on a vertex with no loops, OR a vertex slot
                // we haven't visited). Skip — we can't place it in the
                // packed-vertex set without inventing a new vertex.
                continue;
            }
            line_indices.push(pa);
            line_indices.push(pb);
        }
        self.emit_index_buffer(&line_indices)
    }

    fn write_point_indices(&mut self) -> Result<()> {
        // One index per source vertex that's reachable from at least one
        // face loop. Vertices with no incident loops (isolated verts /
        // dead slab slots) have `vert_to_packed[v] == u32::MAX` and are
        // skipped. For seam-split exports we pick the first packed index
        // assigned to each source vert — points are vertex-identity
        // markers, not corner-attribute consumers, so duplicating per
        // seam side would be wasteful.
        let mut point_indices: Vec<u32> = Vec::with_capacity(self.mesh.totvert());
        for &p in &self.vert_to_packed {
            if p != u32::MAX {
                point_indices.push(p);
            }
        }
        self.emit_index_buffer(&point_indices)
    }

    /// Pick the narrowest sufficient index width and write the buffer to
    /// `self.out`. No-op for empty input.
    fn emit_index_buffer(&mut self, indices: &[u32]) -> Result<()> {
        if indices.is_empty() {
            return Ok(());
        }
        let use_u32 = self.packed_count() > u16::MAX as usize;
        if use_u32 {
            let buf_id = self.out.add_buffer(Buffer::from_pod_slice(indices));
            self.out
                .set_indices(IndexBuffer {
                    buffer: buf_id,
                    offset: 0,
                    count: indices.len(),
                    format: IndexFormat::U32,
                })
                .context("setting U32 index buffer")
        } else {
            let u16s: Vec<u16> = indices.iter().map(|i| *i as u16).collect();
            let count = u16s.len();
            let buf_id = self.out.add_buffer(Buffer::from_pod_slice(&u16s));
            self.out
                .set_indices(IndexBuffer {
                    buffer: buf_id,
                    offset: 0,
                    count,
                    format: IndexFormat::U16,
                })
                .context("setting U16 index buffer")
        }
    }
}

/// A packed-vertex assignment: one entry per output vertex in the exported
/// nub-mesh, sourcing position and vertex attributes from `source_vert` and
/// loop attributes from `source_loop`.
#[derive(Copy, Clone, Debug)]
struct PackedVertex {
    source_vert: VertexKey,
    source_loop: LoopKey,
}

/// Output of the packed-vertex resolution phase.
struct PackedResolve {
    packed: Vec<PackedVertex>,
    /// Indexed by `LoopKey::as_usize()`. `u32::MAX` for unused slab slots.
    loop_to_packed: Vec<u32>,
    /// Indexed by `VertexKey::as_usize()`. `u32::MAX` for unused or
    /// face-less slab slots. In the seam-split case, only the first
    /// packed vertex assigned to each source vertex is recorded; later
    /// duplicates are visible via `loop_to_packed` but not here. Used for
    /// wire-edge endpoint resolution in the `LineList` path.
    vert_to_packed: Vec<u32>,
}

/// Fast path used when `loop_attribs` is empty. One packed vertex per
/// source vertex; every corner loop maps to the same packed index.
fn build_packed_no_seam(mesh: &LoopMesh) -> PackedResolve {
    let watermark = mesh.totloop_watermark();
    let mut packed: Vec<PackedVertex> = Vec::with_capacity(mesh.totvert());
    let mut loop_to_packed: Vec<u32> = vec![u32::MAX; watermark];
    let mut vert_to_packed: Vec<u32> = vec![u32::MAX; mesh.totvert_watermark()];

    let mut face_it = mesh.mesh_faces();
    while let Some(f) = face_it.next(mesh) {
        let mut loop_it = mesh.face_loops(f);
        while let Some(l) = loop_it.next(mesh) {
            let v = mesh.loop_vertex(l);
            let v_slot = v.as_usize();
            let idx = if vert_to_packed[v_slot] != u32::MAX {
                vert_to_packed[v_slot]
            } else {
                let i = packed.len() as u32;
                packed.push(PackedVertex {
                    source_vert: v,
                    source_loop: l,
                });
                vert_to_packed[v_slot] = i;
                i
            };
            loop_to_packed[l.as_usize()] = idx;
        }
    }
    PackedResolve {
        packed,
        loop_to_packed,
        vert_to_packed,
    }
}

/// Slow path used when `loop_attribs` is non-empty. Each (source_vert,
/// loop-attribute fingerprint) pair becomes a distinct packed vertex.
/// `SeamSplitPolicy` controls behaviour on conflicts.
fn build_packed_seam_split(mesh: &LoopMesh, policy: SeamSplitPolicy) -> Result<PackedResolve> {
    let watermark = mesh.totloop_watermark();
    let mut packed: Vec<PackedVertex> = Vec::new();
    let mut loop_to_packed: Vec<u32> = vec![u32::MAX; watermark];
    let mut vert_to_packed: Vec<u32> = vec![u32::MAX; mesh.totvert_watermark()];

    // Keyed by (source_vert, fingerprint). The fingerprint is the bytewise
    // serialisation of the loop_attribs at that loop's slot.
    let mut split_lookup: std::collections::HashMap<(VertexKey, Vec<u8>), u32> =
        std::collections::HashMap::new();

    // For Merge / Error: also remember the first packed_idx (and, for Error,
    // the first fingerprint) seen at each source vertex.
    let mut first_at_vert: std::collections::HashMap<VertexKey, (u32, Vec<u8>)> =
        std::collections::HashMap::new();

    let mut fp_scratch: Vec<u8> = Vec::with_capacity(64);
    let loop_attribs = mesh.loop_attribs();

    let mut face_it = mesh.mesh_faces();
    while let Some(f) = face_it.next(mesh) {
        let mut loop_it = mesh.face_loops(f);
        while let Some(l) = loop_it.next(mesh) {
            let v = mesh.loop_vertex(l);
            fp_scratch.clear();
            loop_attribs.write_fingerprint(l.as_usize(), &mut fp_scratch);

            let packed_idx = match policy {
                SeamSplitPolicy::Split => {
                    if let Some(&idx) = split_lookup.get(&(v, fp_scratch.clone())) {
                        idx
                    } else {
                        let i = packed.len() as u32;
                        packed.push(PackedVertex {
                            source_vert: v,
                            source_loop: l,
                        });
                        split_lookup.insert((v, fp_scratch.clone()), i);
                        i
                    }
                }
                SeamSplitPolicy::Merge => {
                    if let Some((idx, _)) = first_at_vert.get(&v) {
                        *idx
                    } else {
                        let i = packed.len() as u32;
                        packed.push(PackedVertex {
                            source_vert: v,
                            source_loop: l,
                        });
                        first_at_vert.insert(v, (i, fp_scratch.clone()));
                        i
                    }
                }
                SeamSplitPolicy::Error => {
                    if let Some((idx, first_fp)) = first_at_vert.get(&v) {
                        if first_fp != &fp_scratch {
                            return Err(anyhow!(
                                "to_nub_mesh: SeamSplitPolicy::Error — vertex {v:?} has \
                                 differing loop-attribute fingerprints across its corners; \
                                 use Split or Merge to allow this case",
                            ));
                        }
                        *idx
                    } else {
                        let i = packed.len() as u32;
                        packed.push(PackedVertex {
                            source_vert: v,
                            source_loop: l,
                        });
                        first_at_vert.insert(v, (i, fp_scratch.clone()));
                        i
                    }
                }
            };
            loop_to_packed[l.as_usize()] = packed_idx;
            let v_slot = v.as_usize();
            if vert_to_packed[v_slot] == u32::MAX {
                vert_to_packed[v_slot] = packed_idx;
            }
        }
    }

    Ok(PackedResolve {
        packed,
        loop_to_packed,
        vert_to_packed,
    })
}

/// Emit every descriptor in `attrs` as a `nub_mesh` attribute, sourcing the
/// per-packed-vertex value via `source_slot(packed_vertex)`. Used for both
/// `vertex_attribs` (source = source_vert) and `loop_attribs` (source =
/// source_loop). Skips `String` and `Position` (the latter only when
/// `skip_position` is true — Position is exported separately as the
/// canonical vertex field).
fn emit_attribute_set(
    attrs: &crate::loopmesh::ElementAttributes,
    packed: &[PackedVertex],
    source_slot: impl Fn(&PackedVertex) -> usize,
    out: &mut Mesh,
    skip_position: bool,
) -> Result<()> {
    let descriptors: Vec<(AttributeKey, AttributeStorage)> = attrs
        .iter_descriptors()
        .filter_map(|(k, d)| match d.kind {
            ElementAttributeKind::Numeric(_) => attrs.storage_of(k).map(|s| (d.key.clone(), s)),
            ElementAttributeKind::String => None,
        })
        .collect();

    for (attr_key, storage) in descriptors {
        if skip_position
            && matches!(
                &attr_key,
                AttributeKey::Semantic(AttributeSemantic::Position)
            )
        {
            continue;
        }
        let (format, bytes) = pack_attribute_for_packed(attrs, storage, packed, &source_slot)?;
        let buf_id = out.add_buffer(Buffer::from_bytes(&bytes));
        out.add_attribute(Attribute {
            key: attr_key.clone(),
            buffer: buf_id,
            offset: 0,
            stride: format.value_size(),
            format,
            rate: AttributeRate::PerVertex,
        })
        .with_context(|| format!("adding attribute {:?}", attr_key))?;
    }
    Ok(())
}

/// Pack a single attribute (vertex or loop) into a contiguous byte buffer
/// sized to the packed-vertex set. `source_slot` maps each packed-vertex
/// entry to the slab slot in `attrs`'s host element-type that supplies its
/// value (i.e. the source vertex or loop).
fn pack_attribute_for_packed(
    attrs: &crate::loopmesh::ElementAttributes,
    storage: AttributeStorage,
    packed: &[PackedVertex],
    source_slot: &impl Fn(&PackedVertex) -> usize,
) -> Result<(Format, Vec<u8>)> {
    let n = packed.len();
    match storage {
        AttributeStorage::F32(b) => {
            let src = attrs
                .f32_bucket(b)
                .ok_or_else(|| anyhow!("f32 bucket {b} missing"))?;
            let mut bytes = vec![0u8; n * 4];
            let lanes: &mut [f32] = bytemuck::cast_slice_mut(&mut bytes);
            for (i, pv) in packed.iter().enumerate() {
                if let Some(value) = src.get(source_slot(pv)).copied() {
                    lanes[i] = value;
                }
            }
            Ok((Format::new(ComponentType::F32, 1), bytes))
        }
        AttributeStorage::Vec2(b) => {
            let src = attrs
                .vec2_bucket(b)
                .ok_or_else(|| anyhow!("vec2 bucket {b} missing"))?;
            let mut bytes = vec![0u8; n * 8];
            let lanes: &mut [[f32; 2]] = bytemuck::cast_slice_mut(&mut bytes);
            for (i, pv) in packed.iter().enumerate() {
                if let Some(value) = src.get(source_slot(pv)).copied() {
                    lanes[i] = value;
                }
            }
            Ok((Format::new(ComponentType::F32, 2), bytes))
        }
        AttributeStorage::Vec3(b) => {
            let src = attrs
                .vec3_bucket(b)
                .ok_or_else(|| anyhow!("vec3 bucket {b} missing"))?;
            let mut bytes = vec![0u8; n * 12];
            let lanes: &mut [[f32; 3]] = bytemuck::cast_slice_mut(&mut bytes);
            for (i, pv) in packed.iter().enumerate() {
                if let Some(value) = src.get(source_slot(pv)).copied() {
                    lanes[i] = value;
                }
            }
            Ok((Format::new(ComponentType::F32, 3), bytes))
        }
        AttributeStorage::Vec4(b) => {
            let src = attrs
                .vec4_bucket(b)
                .ok_or_else(|| anyhow!("vec4 bucket {b} missing"))?;
            let mut bytes = vec![0u8; n * 16];
            let lanes: &mut [[f32; 4]] = bytemuck::cast_slice_mut(&mut bytes);
            for (i, pv) in packed.iter().enumerate() {
                if let Some(value) = src.get(source_slot(pv)).copied() {
                    lanes[i] = value;
                }
            }
            Ok((Format::new(ComponentType::F32, 4), bytes))
        }
        AttributeStorage::U8(_) | AttributeStorage::I32(_) | AttributeStorage::U32(_) => Err(
            anyhow!("integer/byte attribute export to nub-mesh not yet implemented"),
        ),
        AttributeStorage::String(_) => Err(anyhow!(
            "string attribute does not round-trip through nub-mesh"
        )),
    }
}
