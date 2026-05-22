[![Build Status](https://github.com/rib/loopmesh/actions/workflows/ci.yml/badge.svg)](https://github.com/rib/loopmesh/actions)
[![Docs](https://docs.rs/loopmesh/badge.svg)](https://docs.rs/loopmesh)
[![Crates.io](https://img.shields.io/crates/v/loopmesh.svg)](https://crates.io/crates/loopmesh)

# loopmesh

A radial-edge mesh data structure for Rust, a vertex / edge / loop / face
boundary representation wired together by *disk*, *radial*, and *loop* cycles,
capable of representing manifold and non-manifold topology.

The structure descends from Kevin Weiler's **Radial Edge Structure** (
Weiler, K. (1986). *Topological Structures for Geometric Modeling.* Ph.D.
thesis, Rensselaer Polytechnic Institute.), and follows the practical
simplifications implemented in Blender's *BMesh*.

## Goal: BMesh operation compatibility

`loopmesh` strives to be operation-compatible with Blender's BMesh API, to
express the same topological capabilities and support the same kinds of
operations, and to match BMesh's behaviour element-for-element.

Every operator lands with an A/B test in the companion `bmesh-compat-tests`
crate that builds the same mesh through both `loopmesh` and a thin FFI binding
to the real BMesh and asserts the results are identical.

## How loopmesh currently differs from BMesh

Most differences are deliberate: BMesh predates Blender's generic attribute
system and bakes a number of per-element properties as dedicated struct fields
or header flags that a fresh design models more cleanly as custom-data
attributes or derived caches.

- **No `smooth` header flag.** Smooth/flat shading and edge sharpness are better
  modelled as per-domain boolean *attributes* (`sharp_edge`, `sharp_face`), the
  approach modern Blender itself moved to, from which normals are derived. A
  per-face smooth bit is also largely redundant with per-edge sharpness.
- **No per-face `material` field.** Material is assumed to be tracked as an
  integer face-domain attribute (mirroring modern Blender's `material_index`),
  not a baked field. As an attribute it is also carried correctly through
  topology operators by the custom-data interpolation machinery (with a
  nearest / copy-first policy, since a material index must never be averaged).
- **No operator "slots".** BMesh operators pass element buffers and maps through
  a uniform, name-dispatched slot system. `loopmesh` operators are currently
  ordinary Rust methods with typed parameters and return values (e.g.
  `extrude_region` returns an `ExtrudeResult` carrying the old to new
  vertex/edge maps and the new wall faces). This composes via direct method
  calls and the type system rather than a runtime slot registry.

## License

Licensed under either of

* Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or https://www.apache.org/licenses/LICENSE-2.0)
* MIT license ([LICENSE-MIT](LICENSE-MIT) or https://opensource.org/license/mit)
