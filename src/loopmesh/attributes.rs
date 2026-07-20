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

/// How an attribute is blended when a mid-level op creates a new element from
/// a weighted set of source elements. The policy is a property of the layer's
/// *type*, not of whether its values look numeric
/// (`radial-mesh-guides/BMESH.md` §10.1.4).
///
/// There is deliberately **no** nearest / dominant-source policy: a blend never
/// picks a winning source. Where a source value survives unchanged it is
/// because of an identity (all sources agree, or an exact short-circuit), not
/// because anything elected one.
///
/// Override via [`ElementAttributeDesc::with_interp`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InterpPolicy {
    /// Weighted sum of source values, normalised by total weight. The float
    /// scalar / vector family (UVs, colours, masks, generic float layers).
    Linear,
    /// Weighted sum accumulated in floating point, then rounded **half away
    /// from zero** (`0.5 → 1`, `-0.5 → -1`). The 32-bit integer family.
    ///
    /// The sum is **not** renormalised by its own weight total: §10.1.4 hands
    /// the layer a weight vector that already partitions unity, and dividing by
    /// a re-accumulated total would perturb the sum by a few ulp — enough to
    /// knock a blend that lands exactly on a half off the tie the rounding rule
    /// exists to resolve.
    RoundedLinear,
    /// Logical OR: the result is `true` iff *any* source carrying a strictly
    /// positive weight is `true`. The boolean family. Note this is not a
    /// blend-and-round — a lone `true` at weight `0.125` still yields `true`.
    Disjunction,
    /// The layer's type defines no interpolation: the blend neither synthesises
    /// a value nor copies one from a source, so whatever the destination slot
    /// already holds survives. Strings, the byte / byte-vector types, opaque
    /// ids.
    NoInterp,
}

impl InterpPolicy {
    /// Default policy for a given attribute type, per `BMESH.md` §10.1.4:
    /// [`Linear`](Self::Linear) for the float family,
    /// [`RoundedLinear`](Self::RoundedLinear) for the 32-bit integer family,
    /// [`Disjunction`](Self::Disjunction) for booleans, and
    /// [`NoInterp`](Self::NoInterp) for everything else — the byte and
    /// byte-vector types and strings, whose types define no interpolation at
    /// all. (A `Vec4U8` used as a *byte colour* is the one type that wants an
    /// explicit opt-in to `Linear`; [`ElementAttributeDesc::color`] does that.)
    pub fn default_for(ty: AttrType) -> Self {
        match ty {
            AttrType::F32 | AttrType::F64 | AttrType::Vec2 | AttrType::Vec3 | AttrType::Vec4 => {
                InterpPolicy::Linear
            }
            AttrType::I32 | AttrType::U32 => InterpPolicy::RoundedLinear,
            AttrType::Bool => InterpPolicy::Disjunction,
            AttrType::U8 | AttrType::Vec4U8 | AttrType::String => InterpPolicy::NoInterp,
        }
    }
}

/// One row of `BEVEL.md` §6.6.1.4.0(a)'s closed **blendable** list, carrying its own
/// equality predicate (§6.6.1.4.0(b)). The guide's fourth row — the original-space
/// auxiliary — is a distinct 2-float type there; this crate stores it as `Vec2` like the
/// UV type and it takes the same threshold, so [`Float2`](Self::Float2) covers both.
///
/// `INSET.md` §6.3's **math layer** is the same class named twice, and both guides say in
/// as many words not to build a second predicate for it. So membership lives here, once,
/// and is keyed on **storage type alone** — never on the layer's
/// [`InterpPolicy`], its [`AttrRole`], its name, or its contents.
///
/// Known divergence: the guides key membership on the *layer type*, while this crate keys
/// it on the **storage shape**, and accepts the over-admission that follows for every row —
/// any `Vec2` layer is treated as the UV / original-space kind, any `Vec4` as float RGBA,
/// any `Vec4U8` as byte RGBA, whatever the layer was declared to mean. So a mesh whose only
/// `Vec2` layer is an original-space auxiliary also arms the float2 inventory gate, and a
/// generic `Vec4` layer is read by INSET's corner merge as a colour. Tracked in
/// `bmesh-parity/FOLLOWUPS.md`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum BlendableKind {
    Float2,
    ColorFloat,
    ColorByte,
}

impl BlendableKind {
    /// The closed list itself. `None` for every other storage — a scalar float and a
    /// 3-vector float are both excluded, the 3-vector being §6.6.1.4.0(a)'s sharpest
    /// exclusion, and so are the integer, byte, boolean and string families.
    pub(crate) fn of_storage(storage: AttributeStorage) -> Option<Self> {
        match storage {
            AttributeStorage::Vec2(_) => Some(BlendableKind::Float2),
            AttributeStorage::Vec4(_) => Some(BlendableKind::ColorFloat),
            AttributeStorage::Vec4U8(_) => Some(BlendableKind::ColorByte),
            // Spelled out rather than a catch-all: the list is normatively closed, so a new
            // storage shape must be an explicit decision, not a silent omission.
            AttributeStorage::Bool(_)
            | AttributeStorage::U8(_)
            | AttributeStorage::I32(_)
            | AttributeStorage::U32(_)
            | AttributeStorage::F32(_)
            | AttributeStorage::F64(_)
            | AttributeStorage::Vec3(_)
            | AttributeStorage::String(_) => None,
        }
    }

    /// The kind's own constant, as an upper bound on the **summed squared** difference.
    /// Absolute and coarse: a "painted to the same place" tolerance, not a
    /// round-trip-noise one.
    pub(crate) fn equality_threshold(self) -> f32 {
        match self {
            BlendableKind::Float2 => 1e-5,
            BlendableKind::ColorFloat | BlendableKind::ColorByte => 1e-3,
        }
    }
}

pub(crate) fn summed_squared_difference(left: &[f32], right: &[f32]) -> f32 {
    left.iter()
        .zip(right)
        .map(|(left_component, right_component)| {
            let difference = left_component - right_component;
            difference * difference
        })
        .sum()
}

#[derive(Clone, Copy)]
pub(crate) struct BlendableLayer {
    pub(crate) key: ElementAttributeKey,
    pub(crate) kind: BlendableKind,
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

    /// Build a string attribute descriptor. A string layer's type defines no
    /// interpolation, so the policy defaults to [`InterpPolicy::NoInterp`].
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
    ///
    /// Colours blend linearly whatever their storage width, so a `Vec4U8`
    /// colour layer opts into [`InterpPolicy::Linear`] here — the bare
    /// [`numeric`](Self::numeric) `Vec4U8` layer is an opaque byte vector and
    /// keeps the type's [`NoInterp`](InterpPolicy::NoInterp) default.
    pub fn color(key: impl Into<AttributeKey>, ty: AttrType) -> Self {
        Self::numeric(key, ty)
            .with_role(AttrRole::Color)
            .with_interp(InterpPolicy::Linear)
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
/// slab key.
///
/// A **vacant** element slot's buckets are not scrubbed on removal — they still
/// hold the dead element's values. That is invisible to callers because the slot
/// is default-stamped on the way back *in*: every element-creation site goes
/// through [`mint_slot`](Self::mint_slot), so a live element's attribute block
/// reads as the layer defaults from the moment it exists, whether its slab key
/// is fresh or recycled (`BMESH.md` §13).
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
    /// Prepare `slot`'s storage for a **freshly minted element**: grow every
    /// bucket to `key_watermark`, then stamp each layer's default onto `slot`.
    ///
    /// This is the whole of `BMESH.md` §13's "creation ... default-initialises
    /// the attribute block": a created element reads as the layer default on
    /// every layer of its domain, *unconditionally*. The grow alone would only
    /// achieve that for a **fresh** slab key — a recycled key's bucket slots are
    /// already in range, so the resize is a no-op there and the slot would still
    /// hold the dead element's values. Callers that mean to inherit data write it
    /// explicitly after the mint, so the reset never destroys wanted state.
    ///
    /// Cost is O(the layer slab's capacity on this domain); zero-layer domains
    /// return without touching memory.
    ///
    /// Argument order matches `OpFlagPool`'s `mint_*_slot`: the two are called
    /// back-to-back at every creation site with the same pair, and a
    /// transposition here would default-stamp a **live** element's attribute
    /// block. The `debug_assert` is what turns that transposition into a failing
    /// test rather than silent data loss.
    pub(crate) fn mint_slot(&mut self, slot: usize, key_watermark: usize) {
        debug_assert!(
            slot < key_watermark,
            "mint_slot: slot {slot} is not below the post-creation key watermark \
             {key_watermark} (arguments transposed?)"
        );
        if self.attributes.is_empty() {
            return;
        }
        self.grow_all(key_watermark);
        self.clear_slot(slot);
    }

    /// Resize every bucket to `key_watermark`, default-filling new slots, so
    /// every registered layer keeps one slot per live element after the slab
    /// grows. The single place that must enumerate every bucket type.
    ///
    /// Private on purpose: growing is *not* sufficient at an element-creation
    /// site (see [`mint_slot`](Self::mint_slot)), and a mint path that reached
    /// for `grow_all` would silently reintroduce the recycled-slot hazard.
    fn grow_all(&mut self, key_watermark: usize) {
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

    pub fn get_u32(&self, attr: ElementAttributeKey, slot: usize) -> Option<u32> {
        match self.attributes.get(attr.as_usize())?.storage {
            AttributeStorage::U32(b) => self
                .attributes_u32
                .get(b)
                .and_then(|v| v.get(slot).copied()),
            _ => None,
        }
    }

    pub fn set_u32(&mut self, attr: ElementAttributeKey, slot: usize, value: u32) -> bool {
        let storage = match self.attributes.get(attr.as_usize()) {
            Some(d) => d.storage,
            None => return false,
        };
        match storage {
            AttributeStorage::U32(b) => match self.attributes_u32.get_mut(b) {
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

    /// The layers of [`BlendableKind`]'s closed list, in descriptor order.
    ///
    /// Both guides scope the class to **loop** layers; `ElementAttributes` backs all four
    /// domains, so it is the caller's job to ask this of the loop attributes only.
    pub(crate) fn blendable_layers(&self) -> Vec<BlendableLayer> {
        self.iter_descriptors()
            .filter_map(|(key, desc)| {
                BlendableKind::of_storage(desc.storage).map(|kind| BlendableLayer { key, kind })
            })
            .collect()
    }

    /// Non-allocating "is the closed list non-empty" — `INSET.md` §6.3's corner-merge
    /// gate reads this rather than re-deriving membership. As with
    /// [`blendable_layers`](Self::blendable_layers), the class is a **loop**-layer notion
    /// in both guides, so ask it of the loop attributes.
    pub(crate) fn has_blendable_layer(&self) -> bool {
        self.iter_descriptors()
            .any(|(_, desc)| BlendableKind::of_storage(desc.storage).is_some())
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

    /// Tolerant equality of two element slots over the **blendable set only** —
    /// the comparison used by the corner customdata merge (`INSET.md` §6.3
    /// step-5 gate and its step-7 value match). Unlike
    /// [`write_fingerprint`](Self::write_fingerprint), which is a byte-exact
    /// serialisation over every layer, this reads *only* the closed list of
    /// [`BlendableKind`] — `INSET.md` §6.3's *math* layers, specified
    /// normatively by `BEVEL.md` §6.6.1.4.0(a).
    ///
    /// The set is as load-bearing as the predicate. §6.6.1.4.0(a) is explicit
    /// that a non-blendable layer is "skipped entirely and its values are never
    /// read here": a scalar float, a 3-vector float, an integer, a boolean or a
    /// string layer contributes **nothing**, however far apart its two values
    /// are. Reading them would not merely be stricter, it would be wrong in a
    /// way that shows up with *zero authored disagreement* — §7 resamples the
    /// two gate slots independently, so a scalar float layer's two values
    /// differ in their low bits from the arithmetic alone and an exact compare
    /// on it vetoes merges the reference performs.
    ///
    /// Each blendable layer compares under
    /// [`slots_agree_on_blendable_layer`](Self::slots_agree_on_blendable_layer):
    /// summed squared difference strictly below the *kind's own* constant, for
    /// exactly the same reason — two independently-rebaked resamples that the
    /// operator treats as "the same value" agree only to within that constant.
    /// Byte RGBA is on the list, but its constant is vacuous on the raw
    /// `0..=255` scale, so it compares exactly in practice.
    ///
    /// Returns `true` when every blendable layer agrees — vacuously `true` on a
    /// mesh with no blendable layer at all.
    pub(crate) fn slots_agree_on_blendable_layers(&self, a: usize, b: usize) -> bool {
        for (key, desc) in self.iter_descriptors() {
            let Some(kind) = BlendableKind::of_storage(desc.storage) else {
                continue;
            };
            if !self.slots_agree_on_blendable_layer(BlendableLayer { key, kind }, a, b) {
                return false;
            }
        }
        true
    }

    /// **The** equality predicate of the blendable class (`BEVEL.md` §6.6.1.4.0(b),
    /// `INSET.md` §6.3 step-5 note): form the difference vector over the whole layer value,
    /// sum the squares of its components, and agree when that sum is **strictly below** the
    /// kind's own constant. Absolute, not relative; whole-vector, not per-component.
    ///
    /// Byte colour is widened onto the raw `0..=255` scale rather than normalised, which is
    /// what makes its constant vacuous — one step of difference already contributes `1.0`.
    ///
    /// The strict `<` is load-bearing at NaN: a NaN component on either side propagates
    /// through the difference-and-square chain, the comparison is false, and the slots
    /// **disagree** — including a NaN against the bit-identical NaN. That is the arithmetic
    /// falling out, not a special case, and it is reproduced rather than fixed.
    pub(crate) fn slots_agree_on_blendable_layer(
        &self,
        layer: BlendableLayer,
        a: usize,
        b: usize,
    ) -> bool {
        // Both slots read the *same* layer, so a one-sided absence is a lookup bug, not a
        // value difference: debug builds assert, release builds report "disagree".
        let difference = match layer.kind {
            BlendableKind::Float2 => {
                match (self.get_vec2(layer.key, a), self.get_vec2(layer.key, b)) {
                    (Some(left), Some(right)) => summed_squared_difference(&left, &right),
                    (None, None) => return true,
                    _ => {
                        debug_assert!(false, "both slots index the same float2 layer");
                        return false;
                    }
                }
            }
            BlendableKind::ColorFloat => {
                match (self.get_vec4(layer.key, a), self.get_vec4(layer.key, b)) {
                    (Some(left), Some(right)) => summed_squared_difference(&left, &right),
                    (None, None) => return true,
                    _ => {
                        debug_assert!(false, "both slots index the same float RGBA layer");
                        return false;
                    }
                }
            }
            BlendableKind::ColorByte => {
                match (self.get_vec4u8(layer.key, a), self.get_vec4u8(layer.key, b)) {
                    (Some(left), Some(right)) => {
                        let widen = |bytes: [u8; 4]| bytes.map(f32::from);
                        summed_squared_difference(&widen(left), &widen(right))
                    }
                    (None, None) => return true,
                    _ => {
                        debug_assert!(false, "both slots index the same byte RGBA layer");
                        return false;
                    }
                }
            }
        };
        difference < layer.kind.equality_threshold()
    }

    // ----- Customdata interpolation primitives -----

    /// Interpolate every attribute into `dst_slot` from a weighted blend of
    /// source slots. The per-attribute behaviour is governed by each
    /// descriptor's [`InterpPolicy`]:
    ///
    /// - [`Linear`](InterpPolicy::Linear) — weighted sum, normalised by the
    ///   total weight.
    /// - [`RoundedLinear`](InterpPolicy::RoundedLinear) — weighted sum
    ///   accumulated in floating point, rounded half away from zero, *not*
    ///   renormalised.
    /// - [`Disjunction`](InterpPolicy::Disjunction) — true iff any
    ///   strictly-positive-weight source is true.
    /// - [`NoInterp`](InterpPolicy::NoInterp) — leave destination untouched.
    ///
    /// Empty `sources` is a no-op. A source set whose weights total zero is
    /// deliberately **asymmetric** across the policies (§10.1.4): the two
    /// weighted-sum policies have nothing to write and leave the destination
    /// untouched, whereas the disjunction is defined over the sources that carry
    /// *strictly positive* weight — of which there are none — and so writes
    /// `false`.
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
    pub(crate) fn snapshot_slots(&self, slots: &[usize]) -> SlotSnapshot {
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
    /// their own weight total for robustness, `RoundedLinear` layers do not
    /// (see [`InterpPolicy::RoundedLinear`]).
    ///
    /// Crate-internal: a snapshot carries no domain marker, so it is only sound
    /// against the same [`ElementAttributes`] it was captured from (or one with
    /// an identical layer layout). Keeping it out of the public surface is what
    /// makes that an invariant rather than a caller-triggerable panic.
    pub(crate) fn interpolate_from_snapshot(
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

    /// Write a [`SlotSnapshot`]'s captured slot `index` verbatim onto
    /// `dst_slot`: the 1:1 [`copy_into`](Self::copy_into) with a **detached**
    /// source, so it outlives the element it was captured from. Every layer is
    /// written, [`NoInterp`](InterpPolicy::NoInterp) included — this is a block
    /// copy, not a blend.
    ///
    /// The primitive a *rebuild* needs: an op that kills an element and
    /// re-creates it over a new cycle must capture the block **before** the kill,
    /// because the freed slot is default-stamped the moment the replacement
    /// recycles it ([`mint_slot`](Self::mint_slot)).
    pub(crate) fn restore_slot(&mut self, dst_slot: usize, snapshot: &SlotSnapshot, index: usize) {
        for (storage, _, values) in &snapshot.layers {
            match (*storage, values) {
                (AttributeStorage::Bool(b), SnapshotValues::Bool(src)) => {
                    write_slot(self.attributes_bool.get_mut(b), dst_slot, src.get(index))
                }
                (AttributeStorage::U8(b), SnapshotValues::U8(src)) => {
                    write_slot(self.attributes_u8.get_mut(b), dst_slot, src.get(index))
                }
                (AttributeStorage::I32(b), SnapshotValues::I32(src)) => {
                    write_slot(self.attributes_i32.get_mut(b), dst_slot, src.get(index))
                }
                (AttributeStorage::U32(b), SnapshotValues::U32(src)) => {
                    write_slot(self.attributes_u32.get_mut(b), dst_slot, src.get(index))
                }
                (AttributeStorage::F32(b), SnapshotValues::F32(src)) => {
                    write_slot(self.attributes_f32.get_mut(b), dst_slot, src.get(index))
                }
                (AttributeStorage::F64(b), SnapshotValues::F64(src)) => {
                    write_slot(self.attributes_f64.get_mut(b), dst_slot, src.get(index))
                }
                (AttributeStorage::Vec2(b), SnapshotValues::Vec2(src)) => {
                    write_slot(self.attributes_vec2.get_mut(b), dst_slot, src.get(index))
                }
                (AttributeStorage::Vec3(b), SnapshotValues::Vec3(src)) => {
                    write_slot(self.attributes_vec3.get_mut(b), dst_slot, src.get(index))
                }
                (AttributeStorage::Vec4(b), SnapshotValues::Vec4(src)) => {
                    write_slot(self.attributes_vec4.get_mut(b), dst_slot, src.get(index))
                }
                (AttributeStorage::Vec4U8(b), SnapshotValues::Vec4U8(src)) => {
                    write_slot(self.attributes_vec4u8.get_mut(b), dst_slot, src.get(index))
                }
                (AttributeStorage::String(b), SnapshotValues::String(src)) => {
                    if let Some(bucket) = self.attributes_string.get_mut(b) {
                        if let Some(out) = bucket.get_mut(dst_slot) {
                            *out = src.get(index).cloned().flatten();
                        }
                    }
                }
                // A snapshot layer's `SnapshotValues` variant is chosen from its
                // own `AttributeStorage` at capture, so the pair is always
                // matched. Panicking beats a silent no-op: a future storage kind
                // that forgets its arm here would otherwise present as
                // "attributes mysteriously not restored".
                (storage, _) => unreachable!(
                    "snapshot layer's value variant does not match its storage {storage:?}"
                ),
            }
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

    /// Build a destination-driven `(type, name)` remap plan from `src` into
    /// `self` (DUPLICATE.md §8.5.2). The walk is over **destination** layers;
    /// each yields one [`RemapStep`]:
    ///
    /// - A destination layer with a source layer of the **same type and the
    ///   same name** → a [`Copy`](RemapStep::Copy) step (matched layer copied
    ///   source → clone).
    /// - A destination-only layer with no matching source → a
    ///   [`SetDefault`](RemapStep::SetDefault) step: the clone receives the
    ///   layer type's default, written **explicitly**. Element creation already
    ///   guarantees that ([`mint_slot`](Self::mint_slot)), so the step is
    ///   belt-and-braces on a *freshly minted* clone — but the plan is also
    ///   replayed onto destinations the caller supplies, and it is the step that
    ///   makes the plan self-contained rather than dependent on how its
    ///   destination was obtained.
    ///
    /// A source-only layer (no matching destination layer) is never visited —
    /// the walk is over destination layers — so it is simply dropped. Build
    /// once per domain, replay per element with
    /// [`apply_remap_from`](Self::apply_remap_from). The same-mesh path is the
    /// degenerate plan where every layer matches itself — i.e. the verbatim
    /// copy of §4.
    pub(crate) fn build_remap_from(&self, src: &ElementAttributes) -> RemapPlan {
        let mut plan = RemapPlan::new();
        for (_, dst_desc) in self.iter_descriptors() {
            let matched = src
                .iter_descriptors()
                .find(|(_, src_desc)| src_desc.ty == dst_desc.ty && src_desc.key == dst_desc.key);
            match matched {
                Some((_, src_desc)) => plan.push(RemapStep::Copy {
                    dst: dst_desc.storage,
                    src: src_desc.storage,
                }),
                None => plan.push(RemapStep::SetDefault {
                    dst: dst_desc.storage,
                }),
            }
        }
        plan
    }

    /// Replay a [`build_remap_from`](Self::build_remap_from) plan into
    /// `self`'s `dst_slot` (DUPLICATE.md §8.5.2):
    ///
    /// - [`Copy`](RemapStep::Copy) → copy the matched layer's value from `src`'s
    ///   `src_slot`. A copy, never an interpolation — duplicate has exactly one
    ///   source per clone, so only the *offsets* differ from §4's verbatim copy.
    /// - [`SetDefault`](RemapStep::SetDefault) → write the destination-only
    ///   layer's type default onto the clone.
    pub(crate) fn apply_remap_from(
        &mut self,
        plan: &RemapPlan,
        dst_slot: usize,
        src: &ElementAttributes,
        src_slot: usize,
    ) {
        for step in plan {
            match *step {
                RemapStep::Copy {
                    dst,
                    src: src_storage,
                } => {
                    apply_copy_cross(self, dst, dst_slot, src, src_storage, src_slot);
                }
                RemapStep::SetDefault { dst } => set_storage_default(self, dst, dst_slot),
            }
        }
    }

    /// Reset every attribute's `slot` to its default-init value. The
    /// default-initialise behaviour of `BMESH.md` §10, applied to one element's
    /// whole attribute block; [`mint_slot`](Self::mint_slot) is its caller at
    /// element creation.
    ///
    /// Indexed rather than iterated so the descriptor borrow is released per
    /// layer: this is on the mint path and must not allocate a plan buffer.
    /// The sweep is over the descriptor slab's **capacity**, not its live count,
    /// so a domain that has removed layers pays for the vacant slots too — a
    /// non-issue while layer removal is rare and capacity stays within a few of
    /// the live count.
    pub fn clear_slot(&mut self, slot: usize) {
        for index in 0..self.attributes.capacity() {
            let Some(storage) = self.attributes.get(index).map(|desc| desc.storage) else {
                continue;
            };
            set_storage_default(self, storage, slot);
        }
    }
}

/// Stamp one layer's type default onto `slot` (zero / `false` / `None`).
/// Shared by [`ElementAttributes::clear_slot`] and the remap plan's
/// destination-only [`SetDefault`](RemapStep::SetDefault) branch — both need
/// "overwrite whatever is in this slot with the layer default".
fn set_storage_default(attrs: &mut ElementAttributes, storage: AttributeStorage, slot: usize) {
    match storage {
        AttributeStorage::U8(b) => {
            if let Some(v) = attrs.attributes_u8.get_mut(b) {
                if slot < v.len() {
                    v[slot] = 0;
                }
            }
        }
        AttributeStorage::I32(b) => {
            if let Some(v) = attrs.attributes_i32.get_mut(b) {
                if slot < v.len() {
                    v[slot] = 0;
                }
            }
        }
        AttributeStorage::U32(b) => {
            if let Some(v) = attrs.attributes_u32.get_mut(b) {
                if slot < v.len() {
                    v[slot] = 0;
                }
            }
        }
        AttributeStorage::F32(b) => {
            if let Some(v) = attrs.attributes_f32.get_mut(b) {
                if slot < v.len() {
                    v[slot] = 0.0;
                }
            }
        }
        AttributeStorage::Vec2(b) => {
            if let Some(v) = attrs.attributes_vec2.get_mut(b) {
                if slot < v.len() {
                    v[slot] = [0.0; 2];
                }
            }
        }
        AttributeStorage::Vec3(b) => {
            if let Some(v) = attrs.attributes_vec3.get_mut(b) {
                if slot < v.len() {
                    v[slot] = [0.0; 3];
                }
            }
        }
        AttributeStorage::Vec4(b) => {
            if let Some(v) = attrs.attributes_vec4.get_mut(b) {
                if slot < v.len() {
                    v[slot] = [0.0; 4];
                }
            }
        }
        AttributeStorage::Bool(b) => {
            if let Some(v) = attrs.attributes_bool.get_mut(b) {
                if slot < v.len() {
                    v[slot] = false;
                }
            }
        }
        AttributeStorage::F64(b) => {
            if let Some(v) = attrs.attributes_f64.get_mut(b) {
                if slot < v.len() {
                    v[slot] = 0.0;
                }
            }
        }
        AttributeStorage::Vec4U8(b) => {
            if let Some(v) = attrs.attributes_vec4u8.get_mut(b) {
                if slot < v.len() {
                    v[slot] = [0; 4];
                }
            }
        }
        AttributeStorage::String(b) => {
            if let Some(v) = attrs.attributes_string.get_mut(b) {
                if slot < v.len() {
                    v[slot] = None;
                }
            }
        }
    }
}

/// A detached copy of every attribute layer's values at an ordered list of
/// slots, captured by [`ElementAttributes::snapshot_slots`]. Used as a
/// resample source that is immune to in-place writes on the slots it was
/// captured from.
pub(crate) struct SlotSnapshot {
    layers: Vec<(AttributeStorage, InterpPolicy, SnapshotValues)>,
}

impl SlotSnapshot {
    /// A snapshot of a domain that carries no layers: every read of it is a no-op,
    /// so a caller with nothing to blend can skip both the slot list and the gather.
    pub(crate) fn empty() -> Self {
        Self { layers: Vec::new() }
    }

    /// Number of captured slots (snapshot corners).
    pub(crate) fn len(&self) -> usize {
        self.layers.first().map_or(0, |(_, _, v)| v.len())
    }

    #[allow(dead_code)]
    pub(crate) fn is_empty(&self) -> bool {
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

/// Write one captured value into a bucket slot, tolerating a missing bucket or
/// an out-of-range slot. Helper for [`ElementAttributes::restore_slot`].
fn write_slot<T: Copy>(bucket: Option<&mut Vec<T>>, slot: usize, value: Option<&T>) {
    if let (Some(bucket), Some(value)) = (bucket, value) {
        if let Some(out) = bucket.get_mut(slot) {
            *out = *value;
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

/// The rounding rule the integer family's blend uses (`BMESH.md` §10.1.4):
/// half **away from zero** (`0.5 → 1`, `-0.5 → -1`) — not half-to-even, not
/// truncation. `f32::round` already is this rule; the wrapper pins the choice
/// so it cannot drift silently.
#[inline]
fn round_half_away_from_zero(value: f32) -> f32 {
    value.round()
}

/// A bucket value viewed as `N` `f32` lanes, so one blend kernel serves the
/// float, byte and boolean storages. `from_lanes` is where a byte bucket
/// re-quantises (rounding half away from zero, saturating on overflow) and where
/// a boolean bucket collapses back to a bit. The 32-bit integer storages are
/// *not* here: they go through [`WideLane`], because `f32` cannot hold them
/// exactly.
trait BlendLanes<const N: usize>: Copy {
    fn to_lanes(self) -> [f32; N];
    fn from_lanes(lanes: [f32; N]) -> Self;
}

impl BlendLanes<1> for f32 {
    #[inline]
    fn to_lanes(self) -> [f32; 1] {
        [self]
    }
    #[inline]
    fn from_lanes(lanes: [f32; 1]) -> Self {
        lanes[0]
    }
}

impl<const N: usize> BlendLanes<N> for [f32; N] {
    #[inline]
    fn to_lanes(self) -> [f32; N] {
        self
    }
    #[inline]
    fn from_lanes(lanes: [f32; N]) -> Self {
        lanes
    }
}

impl BlendLanes<1> for u8 {
    #[inline]
    fn to_lanes(self) -> [f32; 1] {
        [self as f32]
    }
    #[inline]
    fn from_lanes(lanes: [f32; 1]) -> Self {
        round_half_away_from_zero(lanes[0]) as u8
    }
}

impl BlendLanes<4> for [u8; 4] {
    #[inline]
    fn to_lanes(self) -> [f32; 4] {
        [
            self[0] as f32,
            self[1] as f32,
            self[2] as f32,
            self[3] as f32,
        ]
    }
    #[inline]
    fn from_lanes(lanes: [f32; 4]) -> Self {
        let mut out = [0u8; 4];
        for lane in 0..4 {
            out[lane] = round_half_away_from_zero(lanes[lane]) as u8;
        }
        out
    }
}

impl BlendLanes<1> for bool {
    #[inline]
    fn to_lanes(self) -> [f32; 1] {
        [if self { 1.0 } else { 0.0 }]
    }
    #[inline]
    fn from_lanes(lanes: [f32; 1]) -> Self {
        lanes[0] != 0.0
    }
}

/// A boolean layer's blend is the disjunction whatever numeric-flavoured policy
/// its descriptor happens to carry; only `NoInterp` opts out (§10.1.4). Booleans
/// are dilated by interpolation, never averaged.
#[inline]
fn boolean_policy(policy: InterpPolicy) -> InterpPolicy {
    match policy {
        InterpPolicy::NoInterp => InterpPolicy::NoInterp,
        _ => InterpPolicy::Disjunction,
    }
}

/// The blend kernel (§10.1.4). `None` means "write nothing to the destination":
/// either the policy defines no interpolation, or no source carried weight.
#[inline]
fn blend_lanes<const N: usize>(
    policy: InterpPolicy,
    sources: impl Iterator<Item = (f32, [f32; N])>,
) -> Option<[f32; N]> {
    match policy {
        InterpPolicy::NoInterp => None,
        InterpPolicy::Disjunction => {
            let mut out = [0.0_f32; N];
            let mut saw_source = false;
            for (weight, lanes) in sources {
                saw_source = true;
                // Zero and negative weights contribute nothing; a mean-value
                // solve on a concave polygon can hand us either.
                if weight > 0.0 {
                    for lane in 0..N {
                        if lanes[lane] != 0.0 {
                            out[lane] = 1.0;
                        }
                    }
                }
            }
            saw_source.then_some(out)
        }
        InterpPolicy::Linear | InterpPolicy::RoundedLinear => {
            let mut sum = [0.0_f32; N];
            let mut weight_total = 0.0_f32;
            for (weight, lanes) in sources {
                for lane in 0..N {
                    sum[lane] += weight * lanes[lane];
                }
                weight_total += weight;
            }
            if weight_total == 0.0 {
                return None;
            }
            for lane in 0..N {
                match policy {
                    InterpPolicy::RoundedLinear => sum[lane] = round_half_away_from_zero(sum[lane]),
                    _ => sum[lane] /= weight_total,
                }
            }
            Some(sum)
        }
    }
}

/// The wide (`f64`-accumulating) blend kernel. Kept apart from [`blend_lanes`]
/// so the double-precision layer *and* the 32-bit integer layers accumulate
/// wider than their sources: an `f32` accumulator quantises above 2²⁴, which
/// would make even a one-hot weight vector a lossy "copy".
#[inline]
fn blend_wide(policy: InterpPolicy, sources: impl Iterator<Item = (f32, f64)>) -> Option<f64> {
    match policy {
        InterpPolicy::NoInterp => None,
        InterpPolicy::Disjunction => {
            let mut out = 0.0_f64;
            let mut saw_source = false;
            for (weight, value) in sources {
                saw_source = true;
                if weight > 0.0 && value != 0.0 {
                    out = 1.0;
                }
            }
            saw_source.then_some(out)
        }
        InterpPolicy::Linear | InterpPolicy::RoundedLinear => {
            let mut sum = 0.0_f64;
            let mut weight_total = 0.0_f64;
            for (weight, value) in sources {
                sum += weight as f64 * value;
                weight_total += weight as f64;
            }
            if weight_total == 0.0 {
                return None;
            }
            Some(match policy {
                InterpPolicy::RoundedLinear => sum.round(),
                _ => sum / weight_total,
            })
        }
    }
}

/// A bucket value blended through the wide kernel: the `f64` layer (itself) and
/// the 32-bit integer layers, whose magnitudes outrun `f32`'s exact range.
/// `from_wide` is where an integer bucket re-quantises — rounding half away from
/// zero, saturating on overflow.
trait WideLane: Copy {
    fn to_wide(self) -> f64;
    fn from_wide(value: f64) -> Self;
}

impl WideLane for f64 {
    #[inline]
    fn to_wide(self) -> f64 {
        self
    }
    #[inline]
    fn from_wide(value: f64) -> Self {
        value
    }
}

impl WideLane for i32 {
    #[inline]
    fn to_wide(self) -> f64 {
        self as f64
    }
    #[inline]
    fn from_wide(value: f64) -> Self {
        // Float→int `as` saturates at the type's bounds, which is the wanted
        // behaviour for a blend that overshot the lattice.
        value.round() as i32
    }
}

impl WideLane for u32 {
    #[inline]
    fn to_wide(self) -> f64 {
        self as f64
    }
    #[inline]
    fn from_wide(value: f64) -> Self {
        value.round() as u32
    }
}

/// Blend live bucket slots through the wide kernel.
fn blend_bucket_wide<T: WideLane>(
    values: &mut [T],
    dst: usize,
    sources: &[(f32, usize)],
    policy: InterpPolicy,
) {
    if dst >= values.len() {
        return;
    }
    let blended = blend_wide(
        policy,
        sources
            .iter()
            .filter_map(|&(weight, slot)| values.get(slot).map(|&value| (weight, value.to_wide()))),
    );
    if let Some(value) = blended {
        values[dst] = T::from_wide(value);
    }
}

/// Blend detached snapshot corners through the wide kernel.
fn blend_bucket_wide_from_snapshot<T: WideLane>(
    values: &mut [T],
    dst: usize,
    sources: &[T],
    weights: &[f32],
    policy: InterpPolicy,
) {
    if dst >= values.len() {
        return;
    }
    let blended = blend_wide(
        policy,
        weights
            .iter()
            .zip(sources)
            .map(|(&weight, &value)| (weight, value.to_wide())),
    );
    if let Some(value) = blended {
        values[dst] = T::from_wide(value);
    }
}

/// Blend live bucket slots (`sources` = weight + slot) into `dst`.
fn blend_bucket<T: BlendLanes<N>, const N: usize>(
    values: &mut [T],
    dst: usize,
    sources: &[(f32, usize)],
    policy: InterpPolicy,
) {
    if dst >= values.len() {
        return;
    }
    let blended = blend_lanes(
        policy,
        sources.iter().filter_map(|&(weight, slot)| {
            values.get(slot).map(|&value| (weight, value.to_lanes()))
        }),
    );
    if let Some(lanes) = blended {
        values[dst] = T::from_lanes(lanes);
    }
}

/// Blend detached snapshot corners into `dst`, one weight per captured corner.
fn blend_bucket_from_snapshot<T: BlendLanes<N>, const N: usize>(
    values: &mut [T],
    dst: usize,
    sources: &[T],
    weights: &[f32],
    policy: InterpPolicy,
) {
    if dst >= values.len() {
        return;
    }
    let blended = blend_lanes(
        policy,
        weights
            .iter()
            .zip(sources)
            .map(|(&weight, &value)| (weight, value.to_lanes())),
    );
    if let Some(lanes) = blended {
        values[dst] = T::from_lanes(lanes);
    }
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
    match (storage, values) {
        (AttributeStorage::F32(b), SnapshotValues::F32(src)) => {
            if let Some(v) = attrs.attributes_f32.get_mut(b) {
                blend_bucket_from_snapshot(v, dst, src, weights, policy);
            }
        }
        (AttributeStorage::Vec2(b), SnapshotValues::Vec2(src)) => {
            if let Some(v) = attrs.attributes_vec2.get_mut(b) {
                blend_bucket_from_snapshot(v, dst, src, weights, policy);
            }
        }
        (AttributeStorage::Vec3(b), SnapshotValues::Vec3(src)) => {
            if let Some(v) = attrs.attributes_vec3.get_mut(b) {
                blend_bucket_from_snapshot(v, dst, src, weights, policy);
            }
        }
        (AttributeStorage::Vec4(b), SnapshotValues::Vec4(src)) => {
            if let Some(v) = attrs.attributes_vec4.get_mut(b) {
                blend_bucket_from_snapshot(v, dst, src, weights, policy);
            }
        }
        (AttributeStorage::Vec4U8(b), SnapshotValues::Vec4U8(src)) => {
            if let Some(v) = attrs.attributes_vec4u8.get_mut(b) {
                blend_bucket_from_snapshot(v, dst, src, weights, policy);
            }
        }
        (AttributeStorage::U8(b), SnapshotValues::U8(src)) => {
            if let Some(v) = attrs.attributes_u8.get_mut(b) {
                blend_bucket_from_snapshot(v, dst, src, weights, policy);
            }
        }
        (AttributeStorage::I32(b), SnapshotValues::I32(src)) => {
            if let Some(v) = attrs.attributes_i32.get_mut(b) {
                blend_bucket_wide_from_snapshot(v, dst, src, weights, policy);
            }
        }
        (AttributeStorage::U32(b), SnapshotValues::U32(src)) => {
            if let Some(v) = attrs.attributes_u32.get_mut(b) {
                blend_bucket_wide_from_snapshot(v, dst, src, weights, policy);
            }
        }
        (AttributeStorage::Bool(b), SnapshotValues::Bool(src)) => {
            if let Some(v) = attrs.attributes_bool.get_mut(b) {
                blend_bucket_from_snapshot(v, dst, src, weights, boolean_policy(policy));
            }
        }
        (AttributeStorage::F64(b), SnapshotValues::F64(src)) => {
            if let Some(v) = attrs.attributes_f64.get_mut(b) {
                blend_bucket_wide_from_snapshot(v, dst, src, weights, policy);
            }
        }
        // A string layer's type defines no interpolation: a blend never reads
        // its sources and never writes its destination.
        (AttributeStorage::String(_), _) => {}
        (storage, _) => {
            unreachable!("snapshot layer's value variant does not match its storage {storage:?}")
        }
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
                blend_bucket(v, dst, sources, policy);
            }
        }
        AttributeStorage::Vec2(b) => {
            if let Some(v) = attrs.attributes_vec2.get_mut(b) {
                blend_bucket(v, dst, sources, policy);
            }
        }
        AttributeStorage::Vec3(b) => {
            if let Some(v) = attrs.attributes_vec3.get_mut(b) {
                blend_bucket(v, dst, sources, policy);
            }
        }
        AttributeStorage::Vec4(b) => {
            if let Some(v) = attrs.attributes_vec4.get_mut(b) {
                blend_bucket(v, dst, sources, policy);
            }
        }
        AttributeStorage::Vec4U8(b) => {
            if let Some(v) = attrs.attributes_vec4u8.get_mut(b) {
                blend_bucket(v, dst, sources, policy);
            }
        }
        AttributeStorage::U8(b) => {
            if let Some(v) = attrs.attributes_u8.get_mut(b) {
                blend_bucket(v, dst, sources, policy);
            }
        }
        AttributeStorage::I32(b) => {
            if let Some(v) = attrs.attributes_i32.get_mut(b) {
                blend_bucket_wide(v, dst, sources, policy);
            }
        }
        AttributeStorage::U32(b) => {
            if let Some(v) = attrs.attributes_u32.get_mut(b) {
                blend_bucket_wide(v, dst, sources, policy);
            }
        }
        AttributeStorage::Bool(b) => {
            if let Some(v) = attrs.attributes_bool.get_mut(b) {
                blend_bucket(v, dst, sources, boolean_policy(policy));
            }
        }
        AttributeStorage::F64(b) => {
            if let Some(v) = attrs.attributes_f64.get_mut(b) {
                blend_bucket_wide(v, dst, sources, policy);
            }
        }
        AttributeStorage::String(_) => {}
    }
}

/// One destination layer's remap action (DUPLICATE.md §8.5.2). The
/// destination-driven plan walks the *destination* layers; each becomes either
/// a copy of a `(type, name)`-matched source layer or an explicit default for a
/// destination-only layer.
#[derive(Clone, Copy, Debug)]
pub(crate) enum RemapStep {
    /// Matched layer: copy `src` storage → `dst` storage at the per-element slots.
    Copy {
        dst: AttributeStorage,
        src: AttributeStorage,
    },
    /// Destination-only layer: stamp the layer type's default onto `dst`.
    SetDefault { dst: AttributeStorage },
}

/// A destination-driven customdata remap plan (DUPLICATE.md §8.5.2): one
/// [`RemapStep`] per destination layer, built once per domain and replayed per
/// cloned element across two meshes.
pub(crate) type RemapPlan = Vec<RemapStep>;

/// Cross-layout single-layer copy (DUPLICATE.md §8.5.2): copy the value at
/// `src_slot` in `src`'s layer into `dst_slot` in `dst`'s layer. The two
/// storages share an [`AttrType`] (the plan matched them by `(type, name)`), so
/// only their bucket indices differ; a mismatched pair is a no-op.
fn apply_copy_cross(
    dst: &mut ElementAttributes,
    dst_storage: AttributeStorage,
    dst_slot: usize,
    src: &ElementAttributes,
    src_storage: AttributeStorage,
    src_slot: usize,
) {
    macro_rules! copy {
        ($dbucket:ident, $db:expr, $sbucket:ident, $sb:expr) => {{
            if let (Some(dv), Some(sv)) = (dst.$dbucket.get_mut($db), src.$sbucket.get($sb)) {
                if let (true, Some(value)) = (dst_slot < dv.len(), sv.get(src_slot)) {
                    dv[dst_slot] = value.clone();
                }
            }
        }};
    }
    match (dst_storage, src_storage) {
        (AttributeStorage::Bool(db), AttributeStorage::Bool(sb)) => {
            copy!(attributes_bool, db, attributes_bool, sb)
        }
        (AttributeStorage::U8(db), AttributeStorage::U8(sb)) => {
            copy!(attributes_u8, db, attributes_u8, sb)
        }
        (AttributeStorage::I32(db), AttributeStorage::I32(sb)) => {
            copy!(attributes_i32, db, attributes_i32, sb)
        }
        (AttributeStorage::U32(db), AttributeStorage::U32(sb)) => {
            copy!(attributes_u32, db, attributes_u32, sb)
        }
        (AttributeStorage::F32(db), AttributeStorage::F32(sb)) => {
            copy!(attributes_f32, db, attributes_f32, sb)
        }
        (AttributeStorage::F64(db), AttributeStorage::F64(sb)) => {
            copy!(attributes_f64, db, attributes_f64, sb)
        }
        (AttributeStorage::Vec2(db), AttributeStorage::Vec2(sb)) => {
            copy!(attributes_vec2, db, attributes_vec2, sb)
        }
        (AttributeStorage::Vec3(db), AttributeStorage::Vec3(sb)) => {
            copy!(attributes_vec3, db, attributes_vec3, sb)
        }
        (AttributeStorage::Vec4(db), AttributeStorage::Vec4(sb)) => {
            copy!(attributes_vec4, db, attributes_vec4, sb)
        }
        (AttributeStorage::Vec4U8(db), AttributeStorage::Vec4U8(sb)) => {
            copy!(attributes_vec4u8, db, attributes_vec4u8, sb)
        }
        (AttributeStorage::String(db), AttributeStorage::String(sb)) => {
            copy!(attributes_string, db, attributes_string, sb)
        }
        _ => {}
    }
}

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

#[cfg(test)]
mod tests {
    use super::*;

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

    // ---- Mint-time default-initialisation (BMESH.md §13 / §10) ----
    //
    // Every test below kills an element so its slab key is freed, mints a
    // replacement that recycles that key, and asserts the new element reads as
    // the layer default. Each *asserts the recycling itself* — without that the
    // test would silently degenerate into checking a fresh key, which the plain
    // grow already handled.

    /// A triangle, returned as (verts, edges, face).
    fn triangle(mesh: &mut LoopMesh) -> ([VertexKey; 3], [EdgeKey; 3], FaceKey) {
        let v0 = mesh.create_vertex(Vec3::ZERO);
        let v1 = mesh.create_vertex(Vec3::X);
        let v2 = mesh.create_vertex(Vec3::Y);
        let e0 = mesh.create_edge([v0, v1], EdgeCreateFlags::NONE);
        let e1 = mesh.create_edge([v1, v2], EdgeCreateFlags::NONE);
        let e2 = mesh.create_edge([v2, v0], EdgeCreateFlags::NONE);
        let f = mesh.face_create(&[v0, v1, v2], &[e0, e1, e2], FaceCreateFlags::NONE);
        ([v0, v1, v2], [e0, e1, e2], f)
    }

    #[test]
    fn minted_vertex_in_recycled_slot_reads_defaults() {
        let mut mesh = LoopMesh::new();
        let color = mesh
            .register_vertex_attribute(ElementAttributeDesc::numeric(
                AttributeKey::custom("v_color"),
                AttrType::Vec3,
            ))
            .unwrap();
        let name = mesh
            .register_vertex_attribute(ElementAttributeDesc::string(AttributeKey::custom("v_name")))
            .unwrap();

        let doomed = mesh.create_vertex(Vec3::X);
        mesh.vertex_attribs_mut()
            .set_vec3(color, doomed.as_usize(), [0.4, 0.5, 0.6]);
        mesh.vertex_attribs_mut()
            .set_string(name, doomed.as_usize(), Some("doomed".to_string()));
        mesh.vertex_kill(doomed);

        let minted = mesh.create_vertex(Vec3::Y);
        assert_eq!(minted, doomed, "the mint must recycle the freed slab key");
        assert_eq!(
            mesh.vertex_attribs().get_vec3(color, minted.as_usize()),
            Some([0.0; 3])
        );
        assert_eq!(
            mesh.vertex_attribs().get_string(name, minted.as_usize()),
            None
        );
    }

    #[test]
    fn minted_edge_in_recycled_slot_reads_defaults() {
        let mut mesh = LoopMesh::new();
        let seam = mesh
            .register_edge_attribute(ElementAttributeDesc::numeric(
                AttributeKey::custom("e_seam"),
                AttrType::Bool,
            ))
            .unwrap();

        let v0 = mesh.create_vertex(Vec3::ZERO);
        let v1 = mesh.create_vertex(Vec3::X);
        let v2 = mesh.create_vertex(Vec3::Y);

        let doomed = mesh.create_edge([v0, v1], EdgeCreateFlags::NONE);
        // The crease layer is an ordinary named per-edge F32 layer, registered
        // on first write — a built-in in name only, so it must obey the same
        // mint-time default as any custom layer.
        mesh.set_edge_crease(doomed, 0.75);
        mesh.edge_attribs_mut()
            .set_bool(seam, doomed.as_usize(), true);
        mesh.edge_kill(doomed);

        let minted = mesh.create_edge([v1, v2], EdgeCreateFlags::NONE);
        assert_eq!(minted, doomed, "the mint must recycle the freed slab key");
        assert_eq!(mesh.edge_crease(minted), 0.0);
        assert_eq!(
            mesh.edge_attribs().get_bool(seam, minted.as_usize()),
            Some(false)
        );
    }

    #[test]
    fn minted_face_and_loops_in_recycled_slots_read_defaults() {
        let mut mesh = LoopMesh::new();
        let f_color = mesh
            .register_face_attribute(ElementAttributeDesc::numeric(
                AttributeKey::custom("f_color"),
                AttrType::Vec3,
            ))
            .unwrap();
        let uv = mesh
            .register_loop_attribute(ElementAttributeDesc::uv(AttributeKey::custom("uv")))
            .unwrap();

        let (verts, edges, doomed_face) = triangle(&mut mesh);
        let doomed_loops: Vec<LoopKey> = mesh.iter_face_loops(doomed_face).collect();
        mesh.face_attribs_mut()
            .set_vec3(f_color, doomed_face.as_usize(), [0.9, 0.1, 0.2]);
        for (i, l) in doomed_loops.iter().enumerate() {
            mesh.loop_attribs_mut()
                .set_vec2(uv, l.as_usize(), [i as f32, 1.0]);
        }

        // Kills the face and its three loops; the verts and edges survive, so
        // the rebuilt face lands on exactly the freed face / loop keys.
        mesh.face_kill(doomed_face);

        let minted_face = mesh.face_create(&verts, &edges, FaceCreateFlags::NONE);
        assert_eq!(
            minted_face, doomed_face,
            "the mint must recycle the freed face key"
        );
        let minted_loops: Vec<LoopKey> = mesh.iter_face_loops(minted_face).collect();
        let mut recycled: Vec<LoopKey> = minted_loops.clone();
        recycled.retain(|l| doomed_loops.contains(l));
        assert_eq!(
            recycled.len(),
            doomed_loops.len(),
            "the mint must recycle every freed loop key"
        );

        assert_eq!(
            mesh.face_attribs()
                .get_vec3(f_color, minted_face.as_usize()),
            Some([0.0; 3])
        );
        for l in minted_loops {
            assert_eq!(
                mesh.loop_attribs().get_vec2(uv, l.as_usize()),
                Some([0.0; 2]),
                "loop {l:?} inherited a dead corner's UV"
            );
        }
    }

    /// The single-loop mint path (`create_loop`) is separate from `face_create`'s
    /// inlined first-loop mint; pin it independently.
    #[test]
    fn minted_loop_in_recycled_slot_reads_defaults() {
        let mut mesh = LoopMesh::new();
        let uv = mesh
            .register_loop_attribute(ElementAttributeDesc::uv(AttributeKey::custom("uv")))
            .unwrap();

        // `host` stays alive so the re-minted loops have a face to name; only
        // `doomed_face`'s loops are freed.
        let (verts, _, host) = triangle(&mut mesh);
        let (_, _, doomed_face) = triangle(&mut mesh);
        let doomed_loops: Vec<LoopKey> = mesh.iter_face_loops(doomed_face).collect();
        let doomed = doomed_loops[2];
        mesh.loop_attribs_mut()
            .set_vec2(uv, doomed.as_usize(), [3.0, 4.0]);

        mesh.face_kill(doomed_face);
        // create_loop yields a loop that is not yet spliced into any cycle, so
        // `host` is untouched; all we exercise is the mint.
        let minted: Vec<LoopKey> = verts.iter().map(|&v| mesh.create_loop(v, host)).collect();
        assert!(
            minted.contains(&doomed),
            "the mints must recycle the freed loop keys"
        );
        assert_eq!(
            mesh.loop_attribs().get_vec2(uv, doomed.as_usize()),
            Some([0.0; 2])
        );
    }

    // ----- BMESH.md §10.1.4: the per-layer-type interpolation policy -----

    #[test]
    fn policy_defaults_follow_the_layer_type() {
        for ty in [
            AttrType::F32,
            AttrType::F64,
            AttrType::Vec2,
            AttrType::Vec3,
            AttrType::Vec4,
        ] {
            assert_eq!(
                InterpPolicy::default_for(ty),
                InterpPolicy::Linear,
                "{ty:?}"
            );
        }
        for ty in [AttrType::I32, AttrType::U32] {
            assert_eq!(
                InterpPolicy::default_for(ty),
                InterpPolicy::RoundedLinear,
                "{ty:?}"
            );
        }
        assert_eq!(
            InterpPolicy::default_for(AttrType::Bool),
            InterpPolicy::Disjunction
        );
        for ty in [AttrType::U8, AttrType::Vec4U8, AttrType::String] {
            assert_eq!(
                InterpPolicy::default_for(ty),
                InterpPolicy::NoInterp,
                "{ty:?}"
            );
        }
    }

    /// Three vertices carrying `values`, blended into a fourth by `weights`.
    fn blend_i32_into_fresh_vertex(values: [i32; 2], weights: [f32; 2]) -> i32 {
        let mut mesh = LoopMesh::new();
        let sources = [mesh.create_vertex(Vec3::ZERO), mesh.create_vertex(Vec3::X)];
        let dst = mesh.create_vertex(Vec3::Y);
        let layer = mesh
            .register_vertex_attribute(ElementAttributeDesc::numeric(
                AttributeKey::custom("id"),
                AttrType::I32,
            ))
            .unwrap();
        for (v, value) in sources.iter().zip(values) {
            mesh.vertex_attribs_mut()
                .set_i32(layer, v.as_usize(), value);
        }
        let plan = [
            (weights[0], sources[0].as_usize()),
            (weights[1], sources[1].as_usize()),
        ];
        mesh.vertex_attribs_mut()
            .interpolate_into(dst.as_usize(), &plan);
        mesh.vertex_attribs()
            .get_i32(layer, dst.as_usize())
            .unwrap()
    }

    #[test]
    fn i32_layer_blends_then_rounds_half_away_from_zero() {
        // The blend is a float accumulation, so it can land off the lattice —
        // it is not a copy of the dominant source.
        assert_eq!(blend_i32_into_fresh_vertex([100, 104], [0.875, 0.125]), 101);
        assert_eq!(blend_i32_into_fresh_vertex([100, 101], [0.875, 0.125]), 100);

        // Exact halves round AWAY from zero, in both signs — not to even, not
        // toward zero.
        assert_eq!(blend_i32_into_fresh_vertex([2, 3], [0.5, 0.5]), 3);
        assert_eq!(blend_i32_into_fresh_vertex([7, 8], [0.5, 0.5]), 8);
        assert_eq!(blend_i32_into_fresh_vertex([-2, -3], [0.5, 0.5]), -3);
        assert_eq!(blend_i32_into_fresh_vertex([0, -1], [0.5, 0.5]), -1);
        // Half-to-even would give 4 here; away-from-zero gives 5.
        assert_eq!(blend_i32_into_fresh_vertex([4, 5], [0.5, 0.5]), 5);
        // A whole-number blend is untouched by the rounding.
        assert_eq!(blend_i32_into_fresh_vertex([10, 10], [0.5, 0.5]), 10);
    }

    #[test]
    fn i32_blend_does_not_renormalise_a_near_unity_weight_vector() {
        // §10.1.4's integer rule is "weighted sum, then round" — no division by
        // the re-accumulated weight total. The mean-value solve hands us weights
        // that sum to 1.0 only to within a few ulp, and the *only* place that
        // slop is observable is at an exact half — precisely the tie the
        // rounding rule exists to break.
        let mut mesh = LoopMesh::new();
        let low = mesh.create_vertex(Vec3::ZERO);
        let high = mesh.create_vertex(Vec3::X);
        let ballast = mesh.create_vertex(Vec3::Y);
        let dst = mesh.create_vertex(Vec3::Z);
        let id = mesh
            .register_vertex_attribute(ElementAttributeDesc::numeric(
                AttributeKey::custom("id"),
                AttrType::I32,
            ))
            .unwrap();
        mesh.vertex_attribs_mut().set_i32(id, low.as_usize(), 2);
        mesh.vertex_attribs_mut().set_i32(id, high.as_usize(), 3);
        // A zero-valued source at a tiny weight: it moves the weight total off
        // 1.0 without moving the weighted sum off the exact 2.5.
        mesh.vertex_attribs_mut().set_i32(id, ballast.as_usize(), 0);

        let slop = 2.0e-7_f32;
        let weight_total = 0.5 + 0.5 + slop;
        assert_ne!(weight_total, 1.0, "the test needs a non-unity weight total");
        assert_eq!(
            (2.5_f32 / weight_total).round() as i32,
            2,
            "a renormalising kernel would round this blend down; that is the bug \
             this test pins"
        );

        mesh.vertex_attribs_mut().interpolate_into(
            dst.as_usize(),
            &[
                (0.5, low.as_usize()),
                (0.5, high.as_usize()),
                (slop, ballast.as_usize()),
            ],
        );
        assert_eq!(
            mesh.vertex_attribs().get_i32(id, dst.as_usize()),
            Some(3),
            "the raw weighted sum is exactly 2.5, which rounds away from zero to 3"
        );
    }

    #[test]
    fn i32_blend_accumulates_wider_than_f32() {
        // An f32 accumulator quantises above 2^24, which would make even a
        // one-hot weight vector a lossy copy — the premove-corner overlay of
        // INSET.md §6.2 depends on that copy being exact.
        assert_eq!(
            blend_i32_into_fresh_vertex([2_000_000_001, 7], [1.0, 0.0]),
            2_000_000_001
        );
        assert_eq!(
            blend_i32_into_fresh_vertex([-2_000_000_001, 7], [1.0, 0.0]),
            -2_000_000_001
        );
        // And a genuine blend of two large sources lands between them, rather
        // than on the nearest float-representable multiple of 128.
        assert_eq!(
            blend_i32_into_fresh_vertex([2_000_000_000, 2_000_000_002], [0.5, 0.5]),
            2_000_000_001
        );
    }

    #[test]
    fn integer_blend_saturates_instead_of_wrapping() {
        // A weight vector that does not partition unity can push the sum off the
        // type's lattice; the re-quantisation clamps rather than wraps.
        assert_eq!(
            blend_i32_into_fresh_vertex([i32::MAX, i32::MAX], [1.0, 1.0]),
            i32::MAX
        );
        assert_eq!(
            blend_i32_into_fresh_vertex([i32::MIN, i32::MIN], [1.0, 1.0]),
            i32::MIN
        );
    }

    #[test]
    fn u32_layer_blends_and_rounds_like_its_signed_sibling() {
        let mut mesh = LoopMesh::new();
        let a = mesh.create_vertex(Vec3::ZERO);
        let b = mesh.create_vertex(Vec3::X);
        let dst = mesh.create_vertex(Vec3::Y);
        let id = mesh
            .register_vertex_attribute(ElementAttributeDesc::numeric(
                AttributeKey::custom("id"),
                AttrType::U32,
            ))
            .unwrap();

        let blend = |mesh: &mut LoopMesh, values: [u32; 2], weights: [f32; 2]| -> u32 {
            mesh.vertex_attribs_mut()
                .set_u32(id, a.as_usize(), values[0]);
            mesh.vertex_attribs_mut()
                .set_u32(id, b.as_usize(), values[1]);
            mesh.vertex_attribs_mut().interpolate_into(
                dst.as_usize(),
                &[(weights[0], a.as_usize()), (weights[1], b.as_usize())],
            );
            mesh.vertex_attribs().get_u32(id, dst.as_usize()).unwrap()
        };

        assert_eq!(blend(&mut mesh, [100, 104], [0.875, 0.125]), 101);
        assert_eq!(blend(&mut mesh, [2, 3], [0.5, 0.5]), 3);
        // Wide accumulation: exact past 2^24, where an f32 lane would quantise.
        assert_eq!(
            blend(&mut mesh, [4_000_000_001, 7], [1.0, 0.0]),
            4_000_000_001
        );
        // A negative weight (mean-value coordinates produce them on concave
        // polygons) can drive the sum below zero; it saturates at the bound.
        assert_eq!(blend(&mut mesh, [10, 0], [-1.0, 2.0]), 0);
    }

    #[test]
    fn zero_total_weight_is_asymmetric_across_the_policies() {
        // Deliberate, and §10.1.4-derived: the two weighted-sum policies have
        // nothing to write and leave the destination alone, while the
        // disjunction is defined over the strictly-positive-weight sources — of
        // which there are none — and so writes `false`.
        let mut mesh = LoopMesh::new();
        let a = mesh.create_vertex(Vec3::ZERO);
        let b = mesh.create_vertex(Vec3::X);
        let dst = mesh.create_vertex(Vec3::Y);
        let mask = mesh
            .register_vertex_attribute(ElementAttributeDesc::numeric(
                AttributeKey::custom("mask"),
                AttrType::F32,
            ))
            .unwrap();
        let id = mesh
            .register_vertex_attribute(ElementAttributeDesc::numeric(
                AttributeKey::custom("id"),
                AttrType::I32,
            ))
            .unwrap();
        let flag = mesh
            .register_vertex_attribute(ElementAttributeDesc::numeric(
                AttributeKey::custom("flag"),
                AttrType::Bool,
            ))
            .unwrap();
        for v in [a, b] {
            mesh.vertex_attribs_mut().set_f32(mask, v.as_usize(), 9.0);
            mesh.vertex_attribs_mut().set_i32(id, v.as_usize(), 9);
            mesh.vertex_attribs_mut().set_bool(flag, v.as_usize(), true);
        }
        mesh.vertex_attribs_mut().set_f32(mask, dst.as_usize(), 1.0);
        mesh.vertex_attribs_mut().set_i32(id, dst.as_usize(), 1);
        mesh.vertex_attribs_mut()
            .set_bool(flag, dst.as_usize(), true);

        mesh.vertex_attribs_mut()
            .interpolate_into(dst.as_usize(), &[(0.0, a.as_usize()), (0.0, b.as_usize())]);

        assert_eq!(
            mesh.vertex_attribs().get_f32(mask, dst.as_usize()),
            Some(1.0)
        );
        assert_eq!(mesh.vertex_attribs().get_i32(id, dst.as_usize()), Some(1));
        assert_eq!(
            mesh.vertex_attribs().get_bool(flag, dst.as_usize()),
            Some(false)
        );
    }

    #[test]
    fn bool_layer_blends_by_disjunction_not_by_rounding() {
        let mut mesh = LoopMesh::new();
        let heavy = mesh.create_vertex(Vec3::ZERO);
        let light = mesh.create_vertex(Vec3::X);
        let zero_weight = mesh.create_vertex(Vec3::Y);
        let dst = mesh.create_vertex(Vec3::Z);
        let flag = mesh
            .register_vertex_attribute(ElementAttributeDesc::numeric(
                AttributeKey::custom("flag"),
                AttrType::Bool,
            ))
            .unwrap();

        // A lone `true` at a tiny weight still dilates: blend-and-round would
        // give round(0.125) = 0 → false, which is the wrong reading.
        mesh.vertex_attribs_mut()
            .set_bool(flag, heavy.as_usize(), false);
        mesh.vertex_attribs_mut()
            .set_bool(flag, light.as_usize(), true);
        mesh.vertex_attribs_mut().interpolate_into(
            dst.as_usize(),
            &[(0.875, heavy.as_usize()), (0.125, light.as_usize())],
        );
        assert_eq!(
            mesh.vertex_attribs().get_bool(flag, dst.as_usize()),
            Some(true)
        );

        // Symmetric: which source carries the majority weight is irrelevant.
        mesh.vertex_attribs_mut()
            .set_bool(flag, dst.as_usize(), false);
        mesh.vertex_attribs_mut().interpolate_into(
            dst.as_usize(),
            &[(0.125, heavy.as_usize()), (0.875, light.as_usize())],
        );
        assert_eq!(
            mesh.vertex_attribs().get_bool(flag, dst.as_usize()),
            Some(true)
        );

        // An all-false source set is the only way to get false back.
        mesh.vertex_attribs_mut()
            .set_bool(flag, dst.as_usize(), true);
        mesh.vertex_attribs_mut().interpolate_into(
            dst.as_usize(),
            &[(0.5, heavy.as_usize()), (0.5, heavy.as_usize())],
        );
        assert_eq!(
            mesh.vertex_attribs().get_bool(flag, dst.as_usize()),
            Some(false)
        );

        // Zero and negative weights contribute nothing, however true they are.
        mesh.vertex_attribs_mut()
            .set_bool(flag, zero_weight.as_usize(), true);
        mesh.vertex_attribs_mut().interpolate_into(
            dst.as_usize(),
            &[
                (1.0, heavy.as_usize()),
                (0.0, zero_weight.as_usize()),
                (-0.25, light.as_usize()),
            ],
        );
        assert_eq!(
            mesh.vertex_attribs().get_bool(flag, dst.as_usize()),
            Some(false)
        );
    }

    #[test]
    fn byte_vector_layer_is_not_interpolated() {
        let mut mesh = LoopMesh::new();
        let a = mesh.create_vertex(Vec3::ZERO);
        let b = mesh.create_vertex(Vec3::X);
        let dst = mesh.create_vertex(Vec3::Y);

        // A bare Vec4U8 is an opaque byte vector: its type defines no
        // interpolation, so the blend leaves whatever the destination holds.
        let opaque = mesh
            .register_vertex_attribute(ElementAttributeDesc::numeric(
                AttributeKey::custom("opaque"),
                AttrType::Vec4U8,
            ))
            .unwrap();
        // The same storage under a colour role *does* blend.
        let color = mesh
            .register_vertex_attribute(ElementAttributeDesc::color(
                AttributeKey::custom("color"),
                AttrType::Vec4U8,
            ))
            .unwrap();

        for (attr, values) in [
            (opaque, ([0, 0, 0, 0], [100, 100, 100, 100])),
            (color, ([0, 0, 0, 0], [100, 100, 100, 100])),
        ] {
            mesh.vertex_attribs_mut()
                .set_vec4u8(attr, a.as_usize(), values.0);
            mesh.vertex_attribs_mut()
                .set_vec4u8(attr, b.as_usize(), values.1);
        }
        mesh.vertex_attribs_mut()
            .set_vec4u8(opaque, dst.as_usize(), [7, 7, 7, 7]);
        mesh.vertex_attribs_mut()
            .set_vec4u8(color, dst.as_usize(), [7, 7, 7, 7]);

        let plan = [(0.5, a.as_usize()), (0.5, b.as_usize())];
        mesh.vertex_attribs_mut()
            .interpolate_into(dst.as_usize(), &plan);

        assert_eq!(
            mesh.vertex_attribs().get_vec4u8(opaque, dst.as_usize()),
            Some([7, 7, 7, 7]),
            "an opaque byte-vector layer must survive the blend untouched"
        );
        assert_eq!(
            mesh.vertex_attribs().get_vec4u8(color, dst.as_usize()),
            Some([50, 50, 50, 50]),
            "a byte colour layer opts into the weighted sum"
        );
    }

    #[test]
    fn float_layer_still_blends_linearly() {
        let mut mesh = LoopMesh::new();
        let a = mesh.create_vertex(Vec3::ZERO);
        let b = mesh.create_vertex(Vec3::X);
        let dst = mesh.create_vertex(Vec3::Y);
        let mask = mesh
            .register_vertex_attribute(ElementAttributeDesc::numeric(
                AttributeKey::custom("mask"),
                AttrType::F32,
            ))
            .unwrap();
        mesh.vertex_attribs_mut().set_f32(mask, a.as_usize(), 100.0);
        mesh.vertex_attribs_mut().set_f32(mask, b.as_usize(), 104.0);
        mesh.vertex_attribs_mut().interpolate_into(
            dst.as_usize(),
            &[(0.875, a.as_usize()), (0.125, b.as_usize())],
        );
        // No rounding on the float family: the fractional value survives.
        let got = mesh.vertex_attribs().get_f32(mask, dst.as_usize()).unwrap();
        assert!((got - 100.5).abs() < 1e-5, "got {got}");
    }

    #[test]
    fn string_layer_is_not_interpolated() {
        let mut mesh = LoopMesh::new();
        let a = mesh.create_vertex(Vec3::ZERO);
        let b = mesh.create_vertex(Vec3::X);
        let dst = mesh.create_vertex(Vec3::Y);
        let note = mesh
            .register_vertex_attribute(ElementAttributeDesc::string(AttributeKey::custom("note")))
            .unwrap();
        mesh.vertex_attribs_mut()
            .set_string(note, a.as_usize(), Some("from-a".to_string()));
        mesh.vertex_attribs_mut()
            .set_string(note, b.as_usize(), Some("from-b".to_string()));
        mesh.vertex_attribs_mut()
            .set_string(note, dst.as_usize(), Some("seeded".to_string()));

        mesh.vertex_attribs_mut()
            .interpolate_into(dst.as_usize(), &[(0.9, a.as_usize()), (0.1, b.as_usize())]);

        assert_eq!(
            mesh.vertex_attribs().get_string(note, dst.as_usize()),
            Some("seeded"),
            "a string layer's blend must neither synthesise nor copy a value"
        );
    }

    #[test]
    fn snapshot_blend_applies_the_same_policies() {
        // The snapshot path (the resample of §10.1) must agree with the live
        // path — the bevel / inset rebakes go through it.
        let mut mesh = LoopMesh::new();
        let a = mesh.create_vertex(Vec3::ZERO);
        let b = mesh.create_vertex(Vec3::X);
        let dst = mesh.create_vertex(Vec3::Y);
        let id = mesh
            .register_vertex_attribute(ElementAttributeDesc::numeric(
                AttributeKey::custom("id"),
                AttrType::I32,
            ))
            .unwrap();
        let flag = mesh
            .register_vertex_attribute(ElementAttributeDesc::numeric(
                AttributeKey::custom("flag"),
                AttrType::Bool,
            ))
            .unwrap();
        mesh.vertex_attribs_mut().set_i32(id, a.as_usize(), 2);
        mesh.vertex_attribs_mut().set_i32(id, b.as_usize(), 3);
        mesh.vertex_attribs_mut()
            .set_bool(flag, a.as_usize(), false);
        mesh.vertex_attribs_mut().set_bool(flag, b.as_usize(), true);

        let snapshot = mesh
            .vertex_attribs()
            .snapshot_slots(&[a.as_usize(), b.as_usize()]);
        mesh.vertex_attribs_mut()
            .interpolate_from_snapshot(dst.as_usize(), &snapshot, &[0.5, 0.5]);
        assert_eq!(mesh.vertex_attribs().get_i32(id, dst.as_usize()), Some(3));

        mesh.vertex_attribs_mut().interpolate_from_snapshot(
            dst.as_usize(),
            &snapshot,
            &[0.875, 0.125],
        );
        assert_eq!(mesh.vertex_attribs().get_i32(id, dst.as_usize()), Some(2));
        assert_eq!(
            mesh.vertex_attribs().get_bool(flag, dst.as_usize()),
            Some(true),
            "the 0.125-weight `true` still dilates through the snapshot path"
        );
    }

    #[test]
    fn snapshot_one_hot_weight_copies_a_large_integer_exactly() {
        // The premove-corner overlay (INSET.md §6.2) is a one-hot snapshot
        // blend, and it claims to reproduce the captured corner exactly. That
        // holds for an integer only because the accumulator is wider than f32.
        let mut mesh = LoopMesh::new();
        let a = mesh.create_vertex(Vec3::ZERO);
        let b = mesh.create_vertex(Vec3::X);
        let dst = mesh.create_vertex(Vec3::Y);
        let id = mesh
            .register_vertex_attribute(ElementAttributeDesc::numeric(
                AttributeKey::custom("id"),
                AttrType::I32,
            ))
            .unwrap();
        mesh.vertex_attribs_mut()
            .set_i32(id, a.as_usize(), 2_000_000_001);
        mesh.vertex_attribs_mut().set_i32(id, b.as_usize(), -5);

        let snapshot = mesh
            .vertex_attribs()
            .snapshot_slots(&[a.as_usize(), b.as_usize()]);
        mesh.vertex_attribs_mut()
            .interpolate_from_snapshot(dst.as_usize(), &snapshot, &[1.0, 0.0]);
        assert_eq!(
            mesh.vertex_attribs().get_i32(id, dst.as_usize()),
            Some(2_000_000_001)
        );
    }
}
