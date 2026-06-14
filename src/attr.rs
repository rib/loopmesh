//! Attribute value types and attribute identity.
//!
//! [`AttrType`] is the type of a per-element custom attribute; [`AttributeKey`]
//! names a layer. The supported value types span booleans, integers, single-
//! and double-precision floats, 2/3/4-component float vectors, four-byte
//! values, and strings.

use std::borrow::Cow;

/// The type of an element attribute. Each variant is a complete value type
/// with one matching in-memory storage bucket.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AttrType {
    /// Boolean flag.
    Bool,
    /// Single unsigned byte.
    U8,
    /// Signed 32-bit integer.
    I32,
    /// Unsigned 32-bit integer.
    U32,
    /// 32-bit float.
    F32,
    /// 64-bit float.
    F64,
    /// 2 × f32 (e.g. UVs).
    Vec2,
    /// 3 × f32.
    Vec3,
    /// 4 × f32 (e.g. float colour).
    Vec4,
    /// 4 × u8 (e.g. a packed byte colour, or any four-byte payload).
    Vec4U8,
    /// UTF-8 string. No GPU representation.
    String,
}

impl AttrType {
    /// True for the one non-numeric type.
    pub fn is_string(self) -> bool {
        matches!(self, AttrType::String)
    }

    /// True for types that default to a weighted (`Linear`) blend. The
    /// float scalar / vector types. Integer, byte, bool, and string types
    /// default to `Nearest` / `CopyFirst` instead — including `Vec4U8`,
    /// which is opaque bytes by default; a caller treating it as a colour
    /// can opt into `Linear` explicitly (the blend dispatch supports it).
    pub fn is_math(self) -> bool {
        matches!(
            self,
            AttrType::F32 | AttrType::F64 | AttrType::Vec2 | AttrType::Vec3 | AttrType::Vec4
        )
    }
}

/// Identity of an attribute layer. loopmesh keys layers by name; the
/// render/export "semantic" mapping (position / normal / uv / …) lives in the
/// export bridge, not the kernel.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum AttributeKey {
    /// A named custom layer.
    Custom(Cow<'static, str>),
}

impl AttributeKey {
    /// A named custom attribute layer.
    pub fn custom(name: impl Into<Cow<'static, str>>) -> Self {
        AttributeKey::Custom(name.into())
    }

    /// The layer's name.
    pub fn name(&self) -> &str {
        match self {
            AttributeKey::Custom(n) => n,
        }
    }
}

impl From<&'static str> for AttributeKey {
    fn from(s: &'static str) -> Self {
        AttributeKey::Custom(Cow::Borrowed(s))
    }
}

impl From<String> for AttributeKey {
    fn from(s: String) -> Self {
        AttributeKey::Custom(Cow::Owned(s))
    }
}
