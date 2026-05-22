//! Operator-scratch flag pool — transient per-element bit fields, allocated
//! lazily and amortised across operator invocations.
//!
//! Conceptual analogue of BMesh's `BMOpFlag` layers: an operator pushes a
//! layer for the element type(s) it needs, marks elements during execution,
//! and pops the layer when done. The pool retains the underlying memory so
//! the next operator's push reuses it without re-allocating.
//!
//! ## Layers, bits, slots
//!
//! - **Layer** — one `Vec<u32>` per element type, indexed by slab slot. Each
//!   layer carries 32 independent bits.
//! - **Bit** — `0..32`, named by the operator.
//! - **Slot** — the element's slab-key `.as_usize()`. Stable until the slot
//!   is recycled.
//!
//! ## Growth strategy — eager
//!
//! When a new element is created, every active layer is extended to the new
//! `key_watermark` in `LoopMesh::create_*` /  `face_create`. This keeps the
//! `set` / `test` / `clear` hot path branch-free. The cost on element
//! creation is bounded by the number of currently-active layers (typically
//! 1–4), tracked in a `SmallVec`.

use smallvec::SmallVec;

use crate::loopmesh::{EdgeKey, ElementKey, FaceKey, LoopKey, VertexKey};

// ---------- Layer handle types ----------

#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct VertOpFlagLayer(u32);

#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct EdgeOpFlagLayer(u32);

#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct FaceOpFlagLayer(u32);

#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct LoopOpFlagLayer(u32);

impl VertOpFlagLayer {
    #[inline]
    fn idx(self) -> usize {
        self.0 as usize
    }
}
impl EdgeOpFlagLayer {
    #[inline]
    fn idx(self) -> usize {
        self.0 as usize
    }
}
impl FaceOpFlagLayer {
    #[inline]
    fn idx(self) -> usize {
        self.0 as usize
    }
}
impl LoopOpFlagLayer {
    #[inline]
    fn idx(self) -> usize {
        self.0 as usize
    }
}

// ---------- The pool ----------

#[derive(Default)]
pub(crate) struct OpFlagPool {
    vert_layers: Vec<Vec<u32>>,
    edge_layers: Vec<Vec<u32>>,
    face_layers: Vec<Vec<u32>>,
    loop_layers: Vec<Vec<u32>>,

    vert_free: Vec<u32>,
    edge_free: Vec<u32>,
    face_free: Vec<u32>,
    loop_free: Vec<u32>,

    // Indices of currently-pushed layers. New elements grow these layers
    // eagerly so the set/test/clear hot path is bounds-check-free.
    active_vert: SmallVec<[u32; 8]>,
    active_edge: SmallVec<[u32; 8]>,
    active_face: SmallVec<[u32; 8]>,
    active_loop: SmallVec<[u32; 8]>,
}

// Each push/pop/set/test/clear is implemented identically across the four
// element types; a macro keeps the bodies in lock-step.
macro_rules! impl_element_ops {
    (
        $domain:ident,                   // verbose name, e.g. vert
        $LayerTy:ident,                  // VertOpFlagLayer
        $layers_field:ident,             // vert_layers
        $free_field:ident,               // vert_free
        $active_field:ident,             // active_vert
        $push_fn:ident,                  // push_vert
        $pop_fn:ident,                   // pop_vert
        $set_fn:ident,                   // vert_set
        $clear_fn:ident,                 // vert_clear
        $test_fn:ident,                  // vert_test
        $clear_all_fn:ident,             // vert_clear_all
        $grow_fn:ident,                  // grow_vert_layers
        $active_count_fn:ident,          // active_vert_count
    ) => {
        impl OpFlagPool {
            pub(crate) fn $push_fn(&mut self, watermark: usize) -> $LayerTy {
                let idx = if let Some(reused) = self.$free_field.pop() {
                    let v = &mut self.$layers_field[reused as usize];
                    v.clear();
                    v.resize(watermark, 0);
                    reused
                } else {
                    let i = self.$layers_field.len() as u32;
                    self.$layers_field.push(vec![0u32; watermark]);
                    i
                };
                self.$active_field.push(idx);
                $LayerTy(idx)
            }

            pub(crate) fn $pop_fn(&mut self, layer: $LayerTy) {
                let pos = self
                    .$active_field
                    .iter()
                    .position(|&i| i == layer.0)
                    .expect(concat!(
                        "double-pop or unknown ",
                        stringify!($domain),
                        " op-flag layer"
                    ));
                self.$active_field.swap_remove(pos);
                self.$free_field.push(layer.0);
            }

            #[inline]
            pub(crate) fn $set_fn(&mut self, layer: $LayerTy, slot: usize, bit: u8) {
                debug_assert!(bit < 32);
                self.$layers_field[layer.idx()][slot] |= 1u32 << bit;
            }

            #[inline]
            pub(crate) fn $clear_fn(&mut self, layer: $LayerTy, slot: usize, bit: u8) {
                debug_assert!(bit < 32);
                self.$layers_field[layer.idx()][slot] &= !(1u32 << bit);
            }

            #[inline]
            pub(crate) fn $test_fn(&self, layer: $LayerTy, slot: usize, bit: u8) -> bool {
                debug_assert!(bit < 32);
                (self.$layers_field[layer.idx()][slot] >> bit) & 1 == 1
            }

            pub(crate) fn $clear_all_fn(&mut self, layer: $LayerTy) {
                self.$layers_field[layer.idx()].fill(0);
            }

            /// Extend every active layer's storage to `new_watermark` if
            /// shorter. Called from `LoopMesh::create_*` after the slab
            /// inserts a new slot.
            pub(crate) fn $grow_fn(&mut self, new_watermark: usize) {
                for &li in &self.$active_field {
                    let v = &mut self.$layers_field[li as usize];
                    if v.len() < new_watermark {
                        v.resize(new_watermark, 0);
                    }
                }
            }

            #[cfg(test)]
            pub(crate) fn $active_count_fn(&self) -> usize {
                self.$active_field.len()
            }
        }
    };
}

impl_element_ops!(
    vert,
    VertOpFlagLayer,
    vert_layers,
    vert_free,
    active_vert,
    push_vert,
    pop_vert,
    vert_set,
    vert_clear,
    vert_test,
    vert_clear_all,
    grow_vert_layers,
    active_vert_count,
);

impl_element_ops!(
    edge,
    EdgeOpFlagLayer,
    edge_layers,
    edge_free,
    active_edge,
    push_edge,
    pop_edge,
    edge_set,
    edge_clear,
    edge_test,
    edge_clear_all,
    grow_edge_layers,
    active_edge_count,
);

impl_element_ops!(
    face,
    FaceOpFlagLayer,
    face_layers,
    face_free,
    active_face,
    push_face,
    pop_face,
    face_set,
    face_clear,
    face_test,
    face_clear_all,
    grow_face_layers,
    active_face_count,
);

impl_element_ops!(
    r#loop,
    LoopOpFlagLayer,
    loop_layers,
    loop_free,
    active_loop,
    push_loop,
    pop_loop,
    loop_set,
    loop_clear,
    loop_test,
    loop_clear_all,
    grow_loop_layers,
    active_loop_count,
);

// Avoid "unused element-key import" warnings even on builds that don't
// exercise every layer type.
#[doc(hidden)]
const _: fn() = || {
    fn assert_keys<T: ElementKey>() {}
    assert_keys::<VertexKey>();
    assert_keys::<EdgeKey>();
    assert_keys::<FaceKey>();
    assert_keys::<LoopKey>();
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn push_pop_reuses_storage() {
        let mut pool = OpFlagPool::default();
        let a = pool.push_vert(8);
        pool.vert_set(a, 3, 5);
        pool.pop_vert(a);

        // Next push should reuse the freed layer (single layer slot exists).
        let b = pool.push_vert(8);
        assert_eq!(
            pool.vert_layers.len(),
            1,
            "second push should reuse, not allocate"
        );
        // Reused layer is zero-cleared on push.
        assert!(!pool.vert_test(b, 3, 5));
    }

    #[test]
    fn set_test_clear_round_trip() {
        let mut pool = OpFlagPool::default();
        let layer = pool.push_vert(4);
        assert!(!pool.vert_test(layer, 0, 7));
        pool.vert_set(layer, 0, 7);
        assert!(pool.vert_test(layer, 0, 7));
        // Other bits on the same slot are independent.
        assert!(!pool.vert_test(layer, 0, 6));
        pool.vert_clear(layer, 0, 7);
        assert!(!pool.vert_test(layer, 0, 7));
    }

    #[test]
    fn clear_all_zeros_layer() {
        let mut pool = OpFlagPool::default();
        let layer = pool.push_vert(4);
        for slot in 0..4 {
            pool.vert_set(layer, slot, 0);
        }
        pool.vert_clear_all(layer);
        for slot in 0..4 {
            assert!(!pool.vert_test(layer, slot, 0));
        }
    }

    #[test]
    fn two_layers_are_independent() {
        let mut pool = OpFlagPool::default();
        let a = pool.push_vert(4);
        let b = pool.push_vert(4);
        pool.vert_set(a, 2, 3);
        assert!(pool.vert_test(a, 2, 3));
        assert!(!pool.vert_test(b, 2, 3));
        pool.pop_vert(a);
        pool.pop_vert(b);
    }

    #[test]
    fn grow_extends_active_layers_only() {
        let mut pool = OpFlagPool::default();
        let _a = pool.push_vert(2);
        // Push a second layer then immediately pop — it becomes inactive but
        // still exists in storage.
        let b = pool.push_vert(2);
        pool.pop_vert(b);

        pool.grow_vert_layers(8);
        // Active layer grew.
        assert_eq!(pool.vert_layers[0].len(), 8);
        // Popped (inactive) layer stayed at its original length.
        assert_eq!(pool.vert_layers[1].len(), 2);
    }

    #[test]
    fn each_element_type_has_independent_pool() {
        let mut pool = OpFlagPool::default();
        let v = pool.push_vert(4);
        let e = pool.push_edge(4);
        let f = pool.push_face(4);
        let l = pool.push_loop(4);
        pool.vert_set(v, 0, 0);
        pool.edge_set(e, 0, 1);
        pool.face_set(f, 0, 2);
        pool.loop_set(l, 0, 3);
        assert!(pool.vert_test(v, 0, 0));
        assert!(pool.edge_test(e, 0, 1));
        assert!(pool.face_test(f, 0, 2));
        assert!(pool.loop_test(l, 0, 3));
        // Crosswise tests are independent.
        assert!(!pool.vert_test(v, 0, 1));
        assert!(!pool.edge_test(e, 0, 0));
    }
}
