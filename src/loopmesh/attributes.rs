use super::*;

/// Where an attribute's per-element values live inside [`ElementAttributes`].
/// Indexes into one of the typed buckets; the `Slab<ElementAttributeDesc>`
/// remembers which. One variant per [`AttrType`].
#[derive(Clone, Copy, Debug)]
pub(crate) enum AttributeStorage {
    Bool(usize),
    U8(usize),
    I32(usize),
    U32(usize),
    F32(usize),
    F64(usize),
    Vec2(usize),
    Vec3(usize),
    Vec4(usize),
    Vec4U8(usize),
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
    /// Default policy for a given attribute type: weighted `Linear` for the
    /// math types (float / vector / byte-color), `Nearest` for the integer /
    /// bool types, `CopyFirst` for strings.
    pub fn default_for(ty: AttrType) -> Self {
        if ty.is_string() {
            InterpPolicy::CopyFirst
        } else if ty.is_math() {
            InterpPolicy::Linear
        } else {
            InterpPolicy::Nearest
        }
    }
}

#[derive(Clone, Debug)]
pub struct ElementAttributeDesc {
    pub key: AttributeKey,
    pub ty: AttrType,
    /// Semantic role of the layer (colour / UV / generic). Must be compatible
    /// with `ty` (see [`AttrRole::accepts`]); registration rejects a
    /// mismatch. Defaults to [`AttrRole::Generic`].
    pub role: AttrRole,
    pub interp: InterpPolicy,
    storage: AttributeStorage,
}

impl ElementAttributeDesc {
    /// Build a numeric attribute descriptor. `key` identifies the attribute;
    /// `ty` is its [`AttrType`]. Interp policy defaults from the type, role
    /// defaults to [`AttrRole::Generic`]. (Despite the name, `ty` may be any
    /// non-string `AttrType` — use [`string`](Self::string) for strings.)
    pub fn numeric(key: impl Into<AttributeKey>, ty: AttrType) -> Self {
        Self {
            key: key.into(),
            ty,
            role: AttrRole::Generic,
            interp: InterpPolicy::default_for(ty),
            // Placeholder — register_*_attribute resolves this when the
            // descriptor is inserted into a specific ElementAttributes.
            storage: AttributeStorage::U8(usize::MAX),
        }
    }

    /// Build a string attribute descriptor. Interp policy defaults to
    /// [`InterpPolicy::CopyFirst`].
    pub fn string(key: impl Into<AttributeKey>) -> Self {
        Self {
            key: key.into(),
            ty: AttrType::String,
            role: AttrRole::Generic,
            interp: InterpPolicy::default_for(AttrType::String),
            storage: AttributeStorage::String(usize::MAX),
        }
    }

    /// Override the default interpolation policy.
    pub fn with_interp(mut self, interp: InterpPolicy) -> Self {
        self.interp = interp;
        self
    }

    /// Set the semantic [`AttrRole`]. Panics in debug builds if the role is
    /// not valid for this descriptor's value type; registration also rejects
    /// an invalid pairing (returning an error) regardless of build profile.
    pub fn with_role(mut self, role: AttrRole) -> Self {
        debug_assert!(
            role.accepts(self.ty),
            "role {role:?} is not valid for attribute type {:?}",
            self.ty
        );
        self.role = role;
        self
    }

    /// Convenience: a `Vec2` UV layer (role [`AttrRole::Uv`]).
    pub fn uv(key: impl Into<AttributeKey>) -> Self {
        Self::numeric(key, AttrType::Vec2).with_role(AttrRole::Uv)
    }

    /// Convenience: a colour layer of the given `ty` (role
    /// [`AttrRole::Color`]). `ty` must be `Vec3` / `Vec4` / `Vec4U8`.
    pub fn color(key: impl Into<AttributeKey>, ty: AttrType) -> Self {
        Self::numeric(key, ty).with_role(AttrRole::Color)
    }
}

#[derive(PartialEq, Eq, Clone, Copy, Debug, Hash)]
pub struct ElementAttributeKey(pub(crate) u32);

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
    attributes_bool: Vec<Vec<bool>>,
    attributes_u8: Vec<Vec<u8>>,
    attributes_i32: Vec<Vec<i32>>,
    attributes_u32: Vec<Vec<u32>>,
    attributes_f32: Vec<Vec<f32>>,
    attributes_f64: Vec<Vec<f64>>,
    attributes_vec2: Vec<Vec<[f32; 2]>>,
    attributes_vec3: Vec<Vec<[f32; 3]>>,
    attributes_vec4: Vec<Vec<[f32; 4]>>,
    attributes_vec4u8: Vec<Vec<[u8; 4]>>,
    attributes_string: Vec<Vec<Option<String>>>,
}

impl ElementAttributes {
    /// Resize every bucket to `key_watermark`, default-filling new slots, so
    /// every registered layer keeps one slot per live element after the slab
    /// grows. The single place that must enumerate every bucket type.
    pub(crate) fn grow_all(&mut self, key_watermark: usize) {
        for v in &mut self.attributes_bool {
            v.resize(key_watermark, false);
        }
        for v in &mut self.attributes_u8 {
            v.resize(key_watermark, 0);
        }
        for v in &mut self.attributes_i32 {
            v.resize(key_watermark, 0);
        }
        for v in &mut self.attributes_u32 {
            v.resize(key_watermark, 0);
        }
        for v in &mut self.attributes_f32 {
            v.resize(key_watermark, 0.0);
        }
        for v in &mut self.attributes_f64 {
            v.resize(key_watermark, 0.0);
        }
        for v in &mut self.attributes_vec2 {
            v.resize(key_watermark, [0.0; 2]);
        }
        for v in &mut self.attributes_vec3 {
            v.resize(key_watermark, [0.0; 3]);
        }
        for v in &mut self.attributes_vec4 {
            v.resize(key_watermark, [0.0; 4]);
        }
        for v in &mut self.attributes_vec4u8 {
            v.resize(key_watermark, [0; 4]);
        }
        for v in &mut self.attributes_string {
            v.resize(key_watermark, None);
        }
    }

    pub(crate) fn clear(&mut self) {
        self.attributes.clear();
        self.attributes_bool.clear();
        self.attributes_u8.clear();
        self.attributes_i32.clear();
        self.attributes_u32.clear();
        self.attributes_f32.clear();
        self.attributes_f64.clear();
        self.attributes_vec2.clear();
        self.attributes_vec3.clear();
        self.attributes_vec4.clear();
        self.attributes_vec4u8.clear();
        self.attributes_string.clear();
    }

    /// Reserve a new attribute slot in the matching typed bucket and
    /// return the resolved [`AttributeStorage`]. Used by every
    /// `register_*_attribute`. Total over [`AttrType`] — never fails on type
    /// (the `Result` is kept for API stability).
    fn insert_with_storage(
        &mut self,
        mut desc: ElementAttributeDesc,
        key_watermark: usize,
    ) -> Result<ElementAttributeKey> {
        if !desc.role.accepts(desc.ty) {
            return Err(anyhow!(
                "attribute role {:?} is not valid for value type {:?}",
                desc.role,
                desc.ty
            ));
        }
        let storage = match desc.ty {
            AttrType::Bool => {
                let idx = self.attributes_bool.len();
                self.attributes_bool.push(vec![false; key_watermark]);
                AttributeStorage::Bool(idx)
            }
            AttrType::U8 => {
                let idx = self.attributes_u8.len();
                self.attributes_u8.push(vec![0; key_watermark]);
                AttributeStorage::U8(idx)
            }
            AttrType::I32 => {
                let idx = self.attributes_i32.len();
                self.attributes_i32.push(vec![0; key_watermark]);
                AttributeStorage::I32(idx)
            }
            AttrType::U32 => {
                let idx = self.attributes_u32.len();
                self.attributes_u32.push(vec![0; key_watermark]);
                AttributeStorage::U32(idx)
            }
            AttrType::F32 => {
                let idx = self.attributes_f32.len();
                self.attributes_f32.push(vec![0.0; key_watermark]);
                AttributeStorage::F32(idx)
            }
            AttrType::F64 => {
                let idx = self.attributes_f64.len();
                self.attributes_f64.push(vec![0.0; key_watermark]);
                AttributeStorage::F64(idx)
            }
            AttrType::Vec2 => {
                let idx = self.attributes_vec2.len();
                self.attributes_vec2.push(vec![[0.0; 2]; key_watermark]);
                AttributeStorage::Vec2(idx)
            }
            AttrType::Vec3 => {
                let idx = self.attributes_vec3.len();
                self.attributes_vec3.push(vec![[0.0; 3]; key_watermark]);
                AttributeStorage::Vec3(idx)
            }
            AttrType::Vec4 => {
                let idx = self.attributes_vec4.len();
                self.attributes_vec4.push(vec![[0.0; 4]; key_watermark]);
                AttributeStorage::Vec4(idx)
            }
            AttrType::Vec4U8 => {
                let idx = self.attributes_vec4u8.len();
                self.attributes_vec4u8.push(vec![[0; 4]; key_watermark]);
                AttributeStorage::Vec4U8(idx)
            }
            AttrType::String => {
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
    /// attribute layers exist (the common case for the mesh-without-attribs
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

    pub(crate) fn bool_bucket(&self, bucket: usize) -> Option<&[bool]> {
        self.attributes_bool.get(bucket).map(|v| v.as_slice())
    }

    pub(crate) fn f64_bucket(&self, bucket: usize) -> Option<&[f64]> {
        self.attributes_f64.get(bucket).map(|v| v.as_slice())
    }

    pub(crate) fn vec4u8_bucket(&self, bucket: usize) -> Option<&[[u8; 4]]> {
        self.attributes_vec4u8.get(bucket).map(|v| v.as_slice())
    }

    pub fn get_bool(&self, attr: ElementAttributeKey, slot: usize) -> Option<bool> {
        match self.attributes.get(attr.as_usize())?.storage {
            AttributeStorage::Bool(b) => self
                .attributes_bool
                .get(b)
                .and_then(|v| v.get(slot).copied()),
            _ => None,
        }
    }

    pub fn set_bool(&mut self, attr: ElementAttributeKey, slot: usize, value: bool) -> bool {
        let storage = match self.attributes.get(attr.as_usize()) {
            Some(d) => d.storage,
            None => return false,
        };
        match storage {
            AttributeStorage::Bool(b) => match self.attributes_bool.get_mut(b) {
                Some(v) if slot < v.len() => {
                    v[slot] = value;
                    true
                }
                _ => false,
            },
            _ => false,
        }
    }

    pub fn get_f64(&self, attr: ElementAttributeKey, slot: usize) -> Option<f64> {
        match self.attributes.get(attr.as_usize())?.storage {
            AttributeStorage::F64(b) => self
                .attributes_f64
                .get(b)
                .and_then(|v| v.get(slot).copied()),
            _ => None,
        }
    }

    pub fn set_f64(&mut self, attr: ElementAttributeKey, slot: usize, value: f64) -> bool {
        let storage = match self.attributes.get(attr.as_usize()) {
            Some(d) => d.storage,
            None => return false,
        };
        match storage {
            AttributeStorage::F64(b) => match self.attributes_f64.get_mut(b) {
                Some(v) if slot < v.len() => {
                    v[slot] = value;
                    true
                }
                _ => false,
            },
            _ => false,
        }
    }

    pub fn get_vec4u8(&self, attr: ElementAttributeKey, slot: usize) -> Option<[u8; 4]> {
        match self.attributes.get(attr.as_usize())?.storage {
            AttributeStorage::Vec4U8(b) => self
                .attributes_vec4u8
                .get(b)
                .and_then(|v| v.get(slot).copied()),
            _ => None,
        }
    }

    pub fn set_vec4u8(&mut self, attr: ElementAttributeKey, slot: usize, value: [u8; 4]) -> bool {
        let storage = match self.attributes.get(attr.as_usize()) {
            Some(d) => d.storage,
            None => return false,
        };
        match storage {
            AttributeStorage::Vec4U8(b) => match self.attributes_vec4u8.get_mut(b) {
                Some(v) if slot < v.len() => {
                    v[slot] = value;
                    true
                }
                _ => false,
            },
            _ => false,
        }
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
                AttributeStorage::Bool(b) => {
                    if let Some(v) = self.bool_bucket(b).and_then(|s| s.get(slot)) {
                        out.push(*v as u8);
                    }
                }
                AttributeStorage::F64(b) => {
                    if let Some(v) = self.f64_bucket(b).and_then(|s| s.get(slot)) {
                        out.extend_from_slice(&v.to_le_bytes());
                    }
                }
                AttributeStorage::Vec4U8(b) => {
                    if let Some(v) = self.vec4u8_bucket(b).and_then(|s| s.get(slot)) {
                        out.extend_from_slice(v);
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
            AttributeStorage::Bool(bk) => pair_eq(self.bool_bucket(bk), a, b),
            AttributeStorage::U8(bk) => pair_eq(self.u8_bucket(bk), a, b),
            AttributeStorage::I32(bk) => pair_eq(self.i32_bucket(bk), a, b),
            AttributeStorage::U32(bk) => pair_eq(self.u32_bucket(bk), a, b),
            AttributeStorage::F32(bk) => pair_eq(self.f32_bucket(bk), a, b),
            AttributeStorage::F64(bk) => pair_eq(self.f64_bucket(bk), a, b),
            AttributeStorage::Vec2(bk) => pair_eq(self.vec2_bucket(bk), a, b),
            AttributeStorage::Vec3(bk) => pair_eq(self.vec3_bucket(bk), a, b),
            AttributeStorage::Vec4(bk) => pair_eq(self.vec4_bucket(bk), a, b),
            AttributeStorage::Vec4U8(bk) => pair_eq(self.vec4u8_bucket(bk), a, b),
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
                AttributeStorage::U8(b) => {
                    SnapshotValues::U8(gather(&self.attributes_u8[b], slots))
                }
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
                AttributeStorage::Bool(b) => {
                    SnapshotValues::Bool(gather(&self.attributes_bool[b], slots))
                }
                AttributeStorage::F64(b) => {
                    SnapshotValues::F64(gather(&self.attributes_f64[b], slots))
                }
                AttributeStorage::Vec4U8(b) => {
                    SnapshotValues::Vec4U8(gather(&self.attributes_vec4u8[b], slots))
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
                AttributeStorage::Bool(b) => {
                    if let Some(v) = self.attributes_bool.get_mut(b) {
                        if slot < v.len() {
                            v[slot] = false;
                        }
                    }
                }
                AttributeStorage::F64(b) => {
                    if let Some(v) = self.attributes_f64.get_mut(b) {
                        if slot < v.len() {
                            v[slot] = 0.0;
                        }
                    }
                }
                AttributeStorage::Vec4U8(b) => {
                    if let Some(v) = self.attributes_vec4u8.get_mut(b) {
                        if slot < v.len() {
                            v[slot] = [0; 4];
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
    Bool(Vec<bool>),
    U8(Vec<u8>),
    I32(Vec<i32>),
    U32(Vec<u32>),
    F32(Vec<f32>),
    F64(Vec<f64>),
    Vec2(Vec<[f32; 2]>),
    Vec3(Vec<[f32; 3]>),
    Vec4(Vec<[f32; 4]>),
    Vec4U8(Vec<[u8; 4]>),
    String(Vec<Option<String>>),
}

impl SnapshotValues {
    fn len(&self) -> usize {
        match self {
            SnapshotValues::Bool(v) => v.len(),
            SnapshotValues::U8(v) => v.len(),
            SnapshotValues::I32(v) => v.len(),
            SnapshotValues::U32(v) => v.len(),
            SnapshotValues::F32(v) => v.len(),
            SnapshotValues::F64(v) => v.len(),
            SnapshotValues::Vec2(v) => v.len(),
            SnapshotValues::Vec3(v) => v.len(),
            SnapshotValues::Vec4(v) => v.len(),
            SnapshotValues::Vec4U8(v) => v.len(),
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
        (AttributeStorage::Bool(b), SnapshotValues::Bool(src)) => {
            if let Some(v) = attrs.attributes_bool.get_mut(b) {
                pick_int!(v, src);
            }
        }
        (AttributeStorage::F64(b), SnapshotValues::F64(src)) => {
            if let Some(v) = attrs.attributes_f64.get_mut(b) {
                if let Some(out) = v.get_mut(dst) {
                    match policy {
                        InterpPolicy::Linear => {
                            let mut sum = 0.0_f64;
                            let mut wsum = 0.0_f64;
                            for (i, &w) in weights.iter().enumerate() {
                                if let Some(&val) = src.get(i) {
                                    sum += w as f64 * val;
                                    wsum += w as f64;
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
        (AttributeStorage::Vec4U8(b), SnapshotValues::Vec4U8(src)) => {
            if let Some(v) = attrs.attributes_vec4u8.get_mut(b) {
                if let Some(out) = v.get_mut(dst) {
                    match policy {
                        InterpPolicy::Linear => {
                            let mut sum = [0.0_f32; 4];
                            let mut wsum = 0.0_f32;
                            for (i, &w) in weights.iter().enumerate() {
                                if let Some(val) = src.get(i) {
                                    for lane in 0..4 {
                                        sum[lane] += w * val[lane] as f32;
                                    }
                                    wsum += w;
                                }
                            }
                            if wsum != 0.0 {
                                *out = round_vec4u8(sum, wsum);
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
        _ => {}
    }
}

/// Round a weighted f32 accumulator (divided by `wsum`) back to `[u8; 4]`,
/// clamping each lane into `0..=255`. Used for the explicit-`Linear` blend
/// of a `Vec4U8` layer.
fn round_vec4u8(sum: [f32; 4], wsum: f32) -> [u8; 4] {
    let mut out = [0u8; 4];
    for lane in 0..4 {
        out[lane] = (sum[lane] / wsum).round().clamp(0.0, 255.0) as u8;
    }
    out
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
        AttributeStorage::F64(b) => {
            if let Some(v) = attrs.attributes_f64.get_mut(b) {
                if dst >= v.len() {
                    return;
                }
                match policy {
                    InterpPolicy::Linear => {
                        let mut sum = 0.0_f64;
                        let mut wsum = 0.0_f64;
                        for &(w, s) in sources {
                            if let Some(&val) = v.get(s) {
                                sum += w as f64 * val;
                                wsum += w as f64;
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
        AttributeStorage::Vec4U8(b) => {
            if let Some(v) = attrs.attributes_vec4u8.get_mut(b) {
                if dst >= v.len() {
                    return;
                }
                match policy {
                    InterpPolicy::Linear => {
                        let mut sum = [0.0_f32; 4];
                        let mut wsum = 0.0_f32;
                        for &(w, s) in sources {
                            if let Some(&val) = v.get(s) {
                                for lane in 0..4 {
                                    sum[lane] += w * val[lane] as f32;
                                }
                                wsum += w;
                            }
                        }
                        if wsum != 0.0 {
                            v[dst] = round_vec4u8(sum, wsum);
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
        AttributeStorage::Bool(b) => {
            if let Some(v) = attrs.attributes_bool.get_mut(b) {
                if dst >= v.len() {
                    return;
                }
                match policy {
                    // Bool has no fractional blend; Linear degenerates to Nearest.
                    InterpPolicy::Nearest | InterpPolicy::Linear => {
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
        AttributeStorage::Bool(b) => {
            if let Some(v) = attrs.attributes_bool.get_mut(b) {
                if src < v.len() && dst < v.len() {
                    v[dst] = v[src];
                }
            }
        }
        AttributeStorage::F64(b) => {
            if let Some(v) = attrs.attributes_f64.get_mut(b) {
                if src < v.len() && dst < v.len() {
                    v[dst] = v[src];
                }
            }
        }
        AttributeStorage::Vec4U8(b) => {
            if let Some(v) = attrs.attributes_vec4u8.get_mut(b) {
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

impl LoopMesh {
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
}
