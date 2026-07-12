//! Attribute value types and attribute identity.
//!
//! [`AttrType`] is the type of a per-element custom attribute; [`AttributeKey`]
//! names a layer. The supported value types span booleans, integers, single-
//! and double-precision floats, 2/3/4-component float vectors, four-byte
//! values, and strings.

use crate::loopmesh::InterpPolicy;
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

/// The semantic role of an attribute layer — what an operator should treat it
/// as, independent of its value type. Lets semantic-targeting operators (colour
/// reverse/rotate, UV reverse/rotate) find the right layers without inferring
/// meaning from the value type alone. `Generic` is data with no special role.
///
/// A role is only valid on certain value types ([`AttrRole::accepts`]); the
/// registration path rejects mismatched pairings (a `Color` boolean, a `Uv`
/// scalar, …), so the role and type can't drift into nonsense.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum AttrRole {
    /// No special role; the value is generic data.
    #[default]
    Generic,
    /// An RGB / RGBA colour.
    Color,
    /// Texture coordinates.
    Uv,
}

impl AttrRole {
    /// Whether this role may be attached to a layer of value type `ty`.
    /// `Generic` accepts any type; `Color` needs a 3- or 4-component layout;
    /// `Uv` needs a 2-component layout.
    pub fn accepts(self, ty: AttrType) -> bool {
        match self {
            AttrRole::Generic => true,
            AttrRole::Color => {
                matches!(ty, AttrType::Vec3 | AttrType::Vec4 | AttrType::Vec4U8)
            }
            AttrRole::Uv => matches!(ty, AttrType::Vec2),
        }
    }
}

impl AttrType {
    /// True for the one non-numeric type.
    pub fn is_string(self) -> bool {
        matches!(self, AttrType::String)
    }

    /// True for types whose *default* interpolation policy is the plain
    /// weighted blend. Derived from [`InterpPolicy::default_for`] rather than
    /// restating the type→policy routing, which lives there alone.
    ///
    /// Note this is the type's default: a `Vec4U8` registered as a colour opts
    /// into `Linear` and blends, while `AttrType::Vec4U8.is_math()` is `false`.
    pub fn is_math(self) -> bool {
        matches!(InterpPolicy::default_for(self), InterpPolicy::Linear)
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
