use std::fmt::Display;

pub trait ElementKey {
    fn as_usize(&self) -> usize;
}

pub trait ElementKeyFromUsize {
    fn from_usize(i: usize) -> Self;
}

#[derive(PartialEq, Eq, Clone, Copy, Debug, Hash)]
pub struct VertexKey(pub(crate) u32);
impl Display for VertexKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}
impl ElementKey for VertexKey {
    fn as_usize(&self) -> usize {
        self.0 as usize
    }
}
impl ElementKeyFromUsize for VertexKey {
    fn from_usize(i: usize) -> Self {
        Self(i as u32)
    }
}

#[derive(PartialEq, Eq, Clone, Copy, Debug, Hash)]
pub struct EdgeKey(pub(crate) u32);
impl EdgeKey {
    /// Sentinel "no edge" value used by [`Loop::edge`] during the brief
    /// detached state inside [`LoopMesh::euler_split_edge_make_vert`].
    /// Every loop reachable via `face_loops(face)` on a fully-constructed
    /// mesh has `loop.edge != EdgeKey::NULL` — that is the invariant
    /// `validate_topology` checks.
    pub const NULL: EdgeKey = EdgeKey(u32::MAX);

    /// True if this key is the [`NULL`](Self::NULL) sentinel.
    #[inline]
    pub fn is_null(self) -> bool {
        self == Self::NULL
    }
}
impl Display for EdgeKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}
impl ElementKey for EdgeKey {
    fn as_usize(&self) -> usize {
        self.0 as usize
    }
}
impl ElementKeyFromUsize for EdgeKey {
    fn from_usize(i: usize) -> Self {
        Self(i as u32)
    }
}

#[derive(PartialEq, Eq, Clone, Copy, Debug, Hash)]
pub struct FaceKey(pub(crate) u32);
impl Display for FaceKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}
impl ElementKey for FaceKey {
    fn as_usize(&self) -> usize {
        self.0 as usize
    }
}
impl ElementKeyFromUsize for FaceKey {
    fn from_usize(i: usize) -> Self {
        Self(i as u32)
    }
}

#[derive(PartialEq, Eq, Clone, Copy, Debug, Hash)]
pub struct LoopKey(pub(crate) u32);
impl Display for LoopKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}
impl ElementKey for LoopKey {
    fn as_usize(&self) -> usize {
        self.0 as usize
    }
}
impl ElementKeyFromUsize for LoopKey {
    fn from_usize(i: usize) -> Self {
        Self(i as u32)
    }
}
