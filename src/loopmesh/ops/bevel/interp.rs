//! Attribute interpolation: face templates, donor capture, corner blends and the UV
//! merge (BEVEL.md §5.2 / §6.6).

use super::*;

/// A new bevel face's **representative original face** (BEVEL.md §6.6.1): the face
/// inherits that donor's *whole* per-face attribute block — every per-face layer, of
/// which the material slot is one field — not the material slot alone. The inherit is
/// stage 1 of §6.6 and runs at **every** value of `material`; an explicit slot only
/// overwrites the one field afterwards.
///
/// A handle into the run's one [`TemplateBank`], not an owned copy: a captured block
/// costs one buffer per registered layer, and a bevel has one donor per beveled corner
/// / edge / junction / terminal. An empty handle (no adjacent original — a boundary
/// corner, a wire edge) leaves the new face on the layer defaults.
#[derive(Clone, Copy, Default)]
pub(super) struct FaceTemplate {
    pub(super) donor: Option<usize>,
}

/// Every donor block of one bevel run (BEVEL.md §6.6.1), captured in a single batched
/// snapshot.
///
/// Donors are **registered** while the mesh is still pristine, and the whole set is
/// captured by [`seal`](Self::seal) before the first face kill: the re-stitch kills and
/// re-creates the adjacent originals, a rebuilt face can recycle a donor's attribute
/// slot, and a consumed corner's faces are no longer reachable from it at all — so by
/// the time the new faces are built the donor's key names nothing. This is the same
/// capture-before-kill discipline `restitch_consumed_endpoints` uses to carry a rebuilt
/// face's own block across its rebuild.
///
/// The bank is **not** gated on `material`: the stage-1 inherit runs at every value of
/// the parameter, and an explicit slot overwrites one field of the inherited block rather
/// than replacing it (§6.6). A donor query skipped under `material >= 0` would give the
/// new face the right slot over the wrong block.
#[derive(Default)]
pub(super) struct TemplateBank {
    pub(super) donor_slots: Vec<usize>,
    pub(super) blocks: Option<SlotSnapshot>,
    pub(super) components: Option<InterpComponents>,
}

impl TemplateBank {
    /// The bank for one bevel run, carrying §6.6.1.4.1's interpolation-component partition
    /// when key 1 of the ring-spanning tie-break is **live** (a float2 loop layer present
    /// *and* `segments` odd) and nothing at all otherwise. The partition is a mesh-wide
    /// flood fill, so it is built here — once, against the pristine mesh, before any
    /// surgery — rather than per beveled corner.
    pub(super) fn new(mesh: &LoopMesh, segments: u32) -> Self {
        Self {
            components: InterpComponents::live(mesh, segments)
                .then(|| InterpComponents::build(mesh)),
            ..Self::default()
        }
    }

    pub(super) fn interp_components(&self) -> Option<&InterpComponents> {
        self.components.as_ref()
    }

    /// Register one new-face site's donor against the pristine mesh, returning the handle
    /// the faces built there read their block back through.
    pub(super) fn register(&mut self, donor: Option<FaceKey>) -> FaceTemplate {
        match donor {
            Some(face) => {
                self.donor_slots.push(face.as_usize());
                FaceTemplate {
                    donor: Some(self.donor_slots.len() - 1),
                }
            }
            None => FaceTemplate::default(),
        }
    }

    /// Capture every registered donor's block at once. Runs after the last `register`
    /// and before the first face kill.
    pub(super) fn seal(&mut self, mesh: &LoopMesh) {
        if !self.donor_slots.is_empty() {
            self.blocks = Some(mesh.face_attribs().snapshot_slots(&self.donor_slots));
        }
    }

    pub(super) fn resolve(&self, template: FaceTemplate) -> FaceTemplateRef<'_> {
        debug_assert!(
            self.blocks.is_some() || template.donor.is_none(),
            "a registered donor was read before the bank was sealed — the new face would \
             silently land on the layer defaults, which §6.6.1 forbids",
        );
        FaceTemplateRef {
            block: self.blocks.as_ref().zip(template.donor),
        }
    }
}

/// A face the partition never labelled — either it did not exist when the partition was
/// built, or no partition covers it.
pub(super) const NO_COMPONENT: u32 = u32::MAX;

/// The mesh-wide interpolation-component partition of BEVEL.md §6.6.1.4.1 — its
/// equivalence **relation** only, never its ids.
///
/// Two faces share a component when a chain of **loop-contiguous** edges joins them: the
/// negation of §6.6.1.4's seam condition (2), read across the mesh's **blendable** loop
/// layers (§6.6.1.4.0(a)) — colours included, under §6.6.1.4.0(b)'s per-kind tolerant
/// compare.
///
/// Note the asymmetry §6.6.1.4.1 draws and this type keeps: the partition's *edges* are
/// decided by the whole blendable set, but whether it is **built** at all is the narrower
/// float2 *inventory* test of [`live`](Self::live). A colour-only discontinuity therefore
/// splits components on a mesh that carries a float2 layer anywhere, and leaves key 1 tied
/// on a mesh that carries none — because there no partition exists.
///
/// The labels this stores **are** §6.6.1.4.1's construction-order numbering: flood-fill
/// seeds taken in face-enumeration order, then the partial top/bottom label swaps. They are
/// therefore ordered, not merely compared for equality — key 1 keeps the lowest-numbered
/// component and the remaining keys decide *within* it. That numbering is reproducible by
/// an engine enumerating faces in the same order, but it is **not** an isomorphism
/// invariant, so a candidate set straddling components has a donor that is only pinned
/// against a same-order fixture.
pub(super) struct InterpComponents {
    pub(super) component_of: Vec<u32>,
}

impl InterpComponents {
    /// Key 1 exists only when a **float2** loop layer is present **and** `segments` is odd
    /// (§6.6.1.4.1). The arming gate is that one-row *type* test over the layer inventory —
    /// not blendability and not a role test — so it stays on the float2 layer even though
    /// the partition's own edges read the wider blendable set. In every other configuration
    /// key 1 is a constant, ties for every candidate, and no partition is built at all.
    pub(super) fn live(mesh: &LoopMesh, segments: u32) -> bool {
        segments % 2 == 1 && mesh.has_float2_loop_layer()
    }

    pub(super) fn build(mesh: &LoopMesh) -> Self {
        let blendable_layers = mesh.blendable_loop_layers();
        let mut component_of = vec![NO_COMPONENT; mesh.face_slot_count()];
        let mut next_component: u32 = 0;
        let mut pending: Vec<FaceKey> = Vec::new();

        for seed in mesh.face_keys() {
            if component_of[seed.as_usize()] != NO_COMPONENT {
                continue;
            }
            component_of[seed.as_usize()] = next_component;
            pending.push(seed);
            while let Some(face) = pending.pop() {
                for loop_key in mesh.iter_face_loops(face) {
                    let edge = mesh.loop_edge(loop_key);
                    let Some(across) =
                        mesh.interp_contiguous_neighbor(face, edge, &blendable_layers)
                    else {
                        continue;
                    };
                    if component_of[across.as_usize()] == NO_COMPONENT {
                        component_of[across.as_usize()] = next_component;
                        pending.push(across);
                    }
                }
            }
            next_component += 1;
        }

        let mut partition = Self { component_of };
        partition.nudge_extremal_components_low(mesh, next_component);
        partition
    }

    /// §6.6.1.4.1's *partial* canonicalisation, run only when the flood fill produced more
    /// than one component. Its intent is to bring the visually topmost and bottommost
    /// components onto the low ids; it does not reliably achieve that, and the artefact —
    /// not the intent — is what has to be reproduced, so the two swaps are keyed on the
    /// components of the **first two faces in enumeration order** rather than on the literal
    /// ids `0` and `1`.
    pub(super) fn nudge_extremal_components_low(&mut self, mesh: &LoopMesh, component_count: u32) {
        if component_count < 2 {
            return;
        }
        let enumeration = mesh.face_keys();

        let mut topmost: Option<(FaceKey, f32)> = None;
        let mut bottommost: Option<(FaceKey, f32)> = None;
        for &face in &enumeration {
            let height = mesh.face_bounding_box_center(face).z;
            // Strict replacement keeps the *first* face in enumeration order on a tie.
            if topmost.is_none_or(|(_, best)| height.total_cmp(&best).is_gt()) {
                topmost = Some((face, height));
            }
            if bottommost.is_none_or(|(_, best)| height.total_cmp(&best).is_lt()) {
                bottommost = Some((face, height));
            }
        }
        let (Some((topmost, _)), Some((bottommost, _))) = (topmost, bottommost) else {
            return;
        };

        let top = self.component_of[topmost.as_usize()];
        let bottom = self.component_of[bottommost.as_usize()];

        let first_component = self.component_of[enumeration[0].as_usize()];
        self.swap_labels(first_component, top);

        if bottom == top {
            return;
        }
        let Some(&second_face) = enumeration.get(1) else {
            return;
        };
        // Re-read the bottom component's label through the swap just applied, then swap it
        // with whatever component the second enumerated face now wears.
        let bottom = if bottom == first_component {
            top
        } else if bottom == top {
            first_component
        } else {
            bottom
        };
        let second_component = self.component_of[second_face.as_usize()];
        self.swap_labels(second_component, bottom);
    }

    pub(super) fn swap_labels(&mut self, left: u32, right: u32) {
        if left == right {
            return;
        }
        for component in &mut self.component_of {
            if *component == left {
                *component = right;
            } else if *component == right {
                *component = left;
            }
        }
    }

    /// A face's component id, or `None` for a face the partition never saw (one minted after
    /// the build). An unknown face cannot be ordered against a known one, so key 1 ties for
    /// it and the chain falls through.
    pub(super) fn component_of(&self, face: FaceKey) -> Option<u32> {
        match self.component_of.get(face.as_usize()).copied() {
            Some(component) if component != NO_COMPONENT => Some(component),
            _ => None,
        }
    }

    /// Do all of `faces` sit in one component — i.e. does key 1 **tie** across them? An
    /// unlabelled face answers `false`: a tie that cannot be demonstrated is not assumed.
    #[cfg(test)]
    pub(super) fn all_share_component(&self, faces: &[FaceKey]) -> bool {
        let mut shared: Option<u32> = None;
        for &face in faces {
            match self.component_of(face) {
                Some(component) => {
                    if *shared.get_or_insert(component) != component {
                        return false;
                    }
                }
                None => return false,
            }
        }
        true
    }
}

/// A new face's resolved donor: the sealed bank's block buffer plus this face's index
/// into it.
#[derive(Clone, Copy, Default)]
pub(super) struct FaceTemplateRef<'a> {
    pub(super) block: Option<(&'a SlotSnapshot, usize)>,
}

/// One beveled edge's multi-segment strip donors, registered against the pristine mesh
/// (BEVEL.md §6.6.2, the multi-segment profile-strip row). The run of `segments` quads is
/// **split about the strip's centre line**: the quads on the near flank's side inherit the
/// near flanking face's whole block, the quads strictly past it the far flanking face's.
///
/// `near_at_chain_start` is what *orients* that split onto §8.3.0's quad numbering, and it
/// is a fact of the pristine mesh: the chain's two ends are the boundverts the edge's two
/// flanking originals each collapse their corner onto, so which end is the near flank's is
/// read off the recorded substitutions before the re-stitch retires those faces' keys.
///
/// `straddle` is the odd-`segments` centre quad's own donor. It is the near flank
/// whenever the beveled edge is seam-free, and §6.6.1.4's tie-break winner between the
/// two flanks when it is not — the one entry of the split the near/far halves do not
/// determine. At even `segments` no quad straddles and the field goes unread.
pub(super) struct StripDonors {
    pub(super) near: FaceTemplate,
    pub(super) far: FaceTemplate,
    pub(super) straddle: FaceTemplate,
    pub(super) near_face: Option<FaceKey>,
    pub(super) near_at_chain_start: bool,
}

/// A strip's resolved donor pair plus the orientation its centre-line split is read in.
#[derive(Clone, Copy)]
pub(super) struct StripDonorRefs<'a> {
    pub(super) near: FaceTemplateRef<'a>,
    pub(super) far: FaceTemplateRef<'a>,
    pub(super) straddle: FaceTemplateRef<'a>,
    pub(super) near_at_chain_start: bool,
}

impl<'a> StripDonorRefs<'a> {
    /// The donor of strip quad `k` (1-based in §8.3.0's numbering, which runs along the
    /// anchor chain's samples). Counting from the near flank, the first `⌈segments / 2⌉`
    /// quads take the near flank and the rest the far one — the ceiling is what hands the
    /// quad *straddling* the centre line at odd `segments` the near flank, the one donor
    /// §6.6.2 says cannot be read off the split itself.
    ///
    /// That last quad is also the run's **only** re-routable donor (§6.6.2's strip-row
    /// scoping): at a seamed beveled edge `straddle` is the tie-break winner rather than
    /// the near flank, while every other quad — and every quad at even `segments` — keeps
    /// its positional half.
    pub(super) fn quad(&self, k: usize, segments: usize) -> FaceTemplateRef<'a> {
        let rank_from_near = if self.near_at_chain_start {
            k
        } else {
            segments + 1 - k
        };
        let straddle_rank = (segments + 1) / 2;
        if segments % 2 == 1 && rank_from_near == straddle_rank {
            return self.straddle;
        }
        if rank_from_near <= (segments + 1) / 2 {
            self.near
        } else {
            self.far
        }
    }
}

/// One **pre-bevel** donor polygon for the per-corner attribute blend of BEVEL.md
/// §5.2.1 step 2: the face's geometry *and* its corners' attribute blocks, read as
/// the face stood before the bevel touched it (§5.2.1 fact (a)).
///
/// Capturing is the alternative the guide sanctions to building every new face
/// while the originals are still alive: the re-stitch rewrites a face by killing it
/// and rebuilding it over the expanded corner list, so a donor read back *after*
/// that finds a polygon whose corners are the new verts themselves — the sample
/// point lands on a donor corner, the weight vector collapses to `1` there
/// (BMESH.md §10.1.3), and every new vertex comes out an exact copy of the beveled
/// corner's block. That is the wrong answer, reached through machinery that looks
/// right, so the capture is not optional.
///
/// Holds the axis-dominant projection of the donor's normal (BMESH.md §10.1.1), its
/// corners projected through it in loop-cycle order, and a snapshot of those corners'
/// **vertex** and **loop** blocks in that same order. Both domains are captured
/// because both are blended from the *one* weight vector the corner solves (§5.2.5:
/// one solve, two domains). The loop snapshot is as mandatory as the vertex one and
/// for a sharper reason: a re-stitched original's loops are killed with the face, so
/// its corner values exist *nowhere* once the rebuild starts.
///
/// No **weight-snap edge** rides along, because none is implemented: §5.2.1.1's seamed
/// branch — where a seam-carrying corner's whole per-boundvert donor list collapses to
/// one override face plus a per-corner snap edge that pulls the sample point onto one
/// of that face's two edges at the corner — is not built here. Every donor a corner
/// gets is the *un-snapped positional* one, so the sample point is used as it falls.
pub(super) struct CornerDonor {
    pub(super) projection: PlaneProjection,
    pub(super) proj_corners: Vec<Vec2>,
    pub(super) corner_vertex_blocks: SlotSnapshot,
    pub(super) corner_loop_blocks: SlotSnapshot,
}

/// Every pre-bevel donor polygon of one bevel run, keyed by the original face.
///
/// The key is a map, not a scan: a donor is looked up once per candidate face while
/// the bank is built, once per re-stitched original, and once per cap corner, so a
/// linear probe would make the whole per-vertex blend quadratic in a corner's
/// incident-face count.
#[derive(Default)]
pub(super) struct DonorBank {
    pub(super) index: HashMap<FaceKey, usize>,
    pub(super) donors: Vec<CornerDonor>,
}

impl DonorBank {
    pub(super) fn index_of(&self, face: FaceKey) -> Option<usize> {
        self.index.get(&face).copied()
    }

    pub(super) fn donor(&self, index: usize) -> &CornerDonor {
        &self.donors[index]
    }

    /// Empty exactly when the mesh carries neither a per-vertex nor a per-loop layer,
    /// in which case the whole blend is a no-op and nothing is captured.
    pub(super) fn is_empty(&self) -> bool {
        self.donors.is_empty()
    }
}

/// The §5.2.4.1 edge-linear snap blend for the even-`segments` vertex-mode welded
/// apex (BEVEL.md §5.2.4.1). The apex `(i, j, k)` with `j > 0` belongs to **no**
/// rebuilt original face, so it is minted from the beveled corner `v` (the §5.2.1
/// step-1 baseline — so `NoInterp` layers keep `cd(v)` per BMESH.md §10.1.4) and its
/// interpolable per-vertex layers then take the edge-linear blend
/// `cd(v) + t·(cd(far) − cd(v))` on the **last-visited** boundvert's incident edge:
/// the predecessor of the ring's start, whose sector writes the shared apex last.
/// `far` is that edge's far endpoint; `t` is the apex's clamped closest-point
/// projection onto the `v → far` segment, taken at the apex's final position. Because
/// the projected point lands on a donor boundary edge, §10.1.3's boundary
/// short-circuit collapses the mean-value resample to exactly this edge-linear form,
/// regardless of which face donates the interior.
pub(super) struct ApexBlend {
    /// The beveled corner `v` — the §5.2.1 step-1 example the apex is minted from.
    pub(super) corner_vertex: VertexKey,
    /// The snap edge's near endpoint, `pos(v)`.
    pub(super) snap_from: Vec3,
    /// The snap edge's far endpoint, `pos(far)`.
    pub(super) snap_to: Vec3,
    /// The pre-bevel per-vertex blocks of `[v, far]`, in that order — the two ends of
    /// the edge-linear blend, weighted `[1 − t, t]`.
    pub(super) endpoints: SlotSnapshot,
}

/// Per-sector §5.2.4.2 blend context for the even-`segments` [V-ADJ] **chain-interior**
/// verts (BEVEL.md §5.2.4.2). From `segments = 4` up a vertex-mode fill mints more
/// interior verts than the welded apex, and each is minted from the beveled corner `v`
/// (§5.2.1 step 1, so a `NoInterp` layer keeps `cd(v)` rather than a layer default), then
/// blended per fill-quad corner in its canonical sector `i`:
///
/// - **on the sector diagonal (`j == k`, non-seam):** the corner's position is snapped
///   onto the sector's own boundvert edge `e_i` and the mean-value blend collapses
///   (BMESH.md §10.1.3's boundary short-circuit) to the edge-linear
///   `cd(v) + t·(cd(far) − cd(v))` — the same closed form the apex takes, held per sector
///   in `sector_snaps[i]`;
/// - **off the diagonal (`j != k`):** the plain mean-value blend of the **next**
///   boundvert's representative face `f2` (`sector_donors[i]`) at the corner's own
///   position, with no snap.
///
/// Sector-private interior verts (both indices `< h`) are written by one sector only and
/// are isomorphism-invariant; the `j = h` seam verts and the welded apex are shared and
/// last-write-wins order-dependent (§5.2.4.2's private/seam/apex invariance split). The
/// welded apex itself is not routed through here — it keeps the dedicated `ApexBlend`
/// path of §5.2.4.1.
///
/// `centre_donors` carries the §5.2.4.3 odd-`segments` centre n-gon rule (the no-seam
/// branch): at odd `segments` the innermost `(i, h, h)` corner of each sector is minted
/// from `v` and its interpolable layers are overwritten by the plain §10.1 interior
/// mean-value blend of boundvert `i`'s **own** representative face (`prev_face` of `e_i`
/// by fan direction) — *not* the next-boundvert rep `f2` the off-diagonal grid verts use,
/// and with **no** weight-snap. Each corner reads its own sector's rep, so the value is
/// anchor-free / isomorphism-invariant. A seam-carrying corner `v` (the deferred branch)
/// has all `centre_donors` slots `None`, so its centre corners keep the bare `cd(v)`
/// baseline. `None` slots (seam-carrying, gap leg, or no donor) keep that baseline too.
pub(super) struct AdjInteriorBlend<'a> {
    pub(super) corner_vertex: VertexKey,
    pub(super) sector_snaps: &'a [Option<ApexBlend>],
    pub(super) sector_donors: &'a [Option<&'a CornerDonor>],
    pub(super) centre_donors: &'a [Option<&'a CornerDonor>],
}

/// The cap's donor for the corner sitting on `ring_vertex` (BEVEL.md §5.2.1.1, the
/// non-seam-carrying branch): boundvert `i`'s own representative, resolved against the
/// same per-leg order `ring_vert_for_edge` and `donors` were both built in. `None` when
/// no face arrives on that leg — the gap leg of an open fan — in which case the cap
/// corner is left unblended.
pub(super) fn donor_for_ring_vertex(
    ring_vert_for_edge: &[(EdgeKey, VertexKey)],
    donors: &[Option<usize>],
    ring_vertex: VertexKey,
) -> Option<usize> {
    ring_vert_for_edge
        .iter()
        .zip(donors)
        .find(|(&(_, ring), _)| ring == ring_vertex)
        .and_then(|(_, &donor)| donor)
}

/// BEVEL.md §5.2.1.1's seam-carrying cap plan: the single override donor face (as an index
/// into the run's [`DonorBank`]) and, per ring vert, the world position the per-corner blend
/// samples at after the §5.2.1.1.1 weight-snap.
pub(super) struct OverrideCapPlan {
    pub(super) donor: usize,
    pub(super) samples: Vec<(VertexKey, Vec3)>,
}

/// BEVEL.md §5.2.6's UV connectivity tolerance: two float2 values are coincident when their
/// summed squared difference is within `1e-4`. A public engine constant, distinct from (and
/// coarser than) the seam test's `1e-5`.
pub(super) const UV_MERGE_TOLERANCE: f32 = 1.0e-4;

pub(super) fn float2_within_merge_tolerance(a: [f32; 2], b: [f32; 2]) -> bool {
    let du = a[0] - b[0];
    let dv = a[1] - b[1];
    du * du + dv * dv <= UV_MERGE_TOLERANCE
}

/// BEVEL.md §5.2.6's `island_connected_at`: two new faces at a beveled vertex `ov` are
/// island-connected on `layer` when their representative donor faces are the **same** (a
/// layer-independent identity test) or when those donors' corners at `ov` fell in the **same**
/// pre-pass bucket for this layer. `corner_index` selects `ov`'s per-corner partition.
pub(super) fn island_connected_at(
    layer: ElementAttributeKey,
    corner_index: usize,
    face_a: FaceKey,
    face_b: FaceKey,
    representative_donor: &HashMap<FaceKey, FaceKey>,
    prepass: &[HashMap<ElementAttributeKey, HashMap<FaceKey, usize>>],
) -> bool {
    let (Some(&donor_a), Some(&donor_b)) = (
        representative_donor.get(&face_a),
        representative_donor.get(&face_b),
    ) else {
        return false;
    };
    if donor_a == donor_b {
        return true;
    }
    let Some(buckets) = prepass.get(corner_index).and_then(|byle| byle.get(&layer)) else {
        return false;
    };
    match (buckets.get(&donor_a), buckets.get(&donor_b)) {
        (Some(a), Some(b)) => a == b,
        _ => false,
    }
}

impl LoopMesh {
    /// Capture one face as a **pre-bevel donor polygon** (BEVEL.md §5.2.1 fact (a)):
    /// its corners flattened through the axis-dominant projection of its normal
    /// (BMESH.md §10.1.1), and a snapshot of those corners' per-vertex and per-loop
    /// blocks in the same loop-cycle order. Must run while the face still stands as it
    /// did before the bevel — for a face about to be re-stitched, that means before its
    /// corner edges are re-pointed, not merely before it is killed.
    ///
    /// A domain with no layers is skipped outright — no slot list, no gather — because
    /// this runs once per incident face of every beveled corner and a mesh decorated on
    /// only one of the two domains is the common case.
    pub(super) fn capture_corner_donor(&self, face: FaceKey) -> CornerDonor {
        let cycle = self.face_loop_keys(face);
        let projection = PlaneProjection::from_normal(self.face_plane_normal(face));

        let has_vertex_layers = !self.vertex_attribs().is_empty();
        let mut proj_corners = Vec::with_capacity(cycle.len());
        let mut vertex_slots = Vec::with_capacity(if has_vertex_layers { cycle.len() } else { 0 });
        for &corner in &cycle {
            let vertex = self.loop_vertex(corner);
            proj_corners.push(projection.project(self.vertex_position(vertex)));
            if has_vertex_layers {
                vertex_slots.push(vertex.as_usize());
            }
        }
        let corner_vertex_blocks = if has_vertex_layers {
            self.vertex_attribs().snapshot_slots(&vertex_slots)
        } else {
            SlotSnapshot::empty()
        };

        let corner_loop_blocks = if self.loop_attribs().is_empty() {
            SlotSnapshot::empty()
        } else {
            let loop_slots: Vec<usize> = cycle.iter().map(|corner| corner.as_usize()).collect();
            self.loop_attribs().snapshot_slots(&loop_slots)
        };

        CornerDonor {
            projection,
            proj_corners,
            corner_vertex_blocks,
            corner_loop_blocks,
        }
    }

    /// Capture the pre-bevel donor of every face in `faces` (deduplicated). A mesh
    /// with neither a per-vertex nor a per-loop layer has nothing for the blend to
    /// write, so it captures nothing and pays for none of this.
    pub(super) fn capture_donor_bank(&self, faces: &[FaceKey]) -> DonorBank {
        let mut bank = DonorBank::default();
        if self.vertex_attribs().is_empty() && self.loop_attribs().is_empty() {
            return bank;
        }
        for &face in faces {
            if bank.index.contains_key(&face) {
                continue;
            }
            let donor = self.capture_corner_donor(face);
            bank.index.insert(face, bank.donors.len());
            bank.donors.push(donor);
        }
        bank
    }

    /// The per-corner blend of BEVEL.md §5.2.1 step 2: overwrite the corner's whole
    /// **per-loop** block *and* its vertex's whole **per-vertex** block with `donor`'s
    /// corner blocks, resampled at the vertex's **final** position.
    ///
    /// Both domains ride one solve (§5.2.5 / BMESH.md §10.1.4): the same donor, the
    /// same weight vector, the loop half blending the donor's per-**corner** blocks
    /// where the vertex half blends its per-**vertex** ones. Re-deriving a second,
    /// separately-computed weight vector for the loop half is not a refactor — it is a
    /// different value rule.
    ///
    /// This is the **un-snapped** case of that call: §5.2.1.1's weight-snap edge (the
    /// seam-carrying branch, which would move the sample point onto an edge of an
    /// override donor face before the solve) is unimplemented, so the sample is taken at
    /// the corner vertex's position as it stands.
    ///
    /// The loop half has no §5.2.1 step-1 baseline to fall back on the way the vertex
    /// half does (a new vertex is minted from `v`'s block; a new loop is not minted
    /// from anything), so every corner that goes unblended keeps the layer defaults.
    /// That is what the fills outside the sites §5.2.5 pins still do.
    ///
    /// The weight is **positional, not parametric** — it is solved from where the
    /// vertex actually ended up, so it tracks the `offset_type` measure, the
    /// `clamp_overlap` reduction, the per-element bevel weights and the junction's
    /// width reconciliation without any of them being plumbed in. The primitive is
    /// the shared polygon-interior resample of BMESH.md §10.1, not a bevel-local
    /// flattening: a ring vert sits *on* a donor boundary edge, so §10.1.3's on-edge
    /// short-circuit collapses the blend to that edge's two endpoints and yields the
    /// closed form `(1 − t)·cd(v) + t·cd(far)` at the realised slide fraction
    /// (§5.2.2) — exactly, not approximately.
    ///
    /// It is a **blend**, never a block copy, even where the weight vector is an
    /// on-corner identity (§10.1.3: a fully clamped ring vert sits exactly on `far`;
    /// a surviving corner of a rebuilt face samples at itself). A block copy there
    /// would stamp the donor corner's *no-interp* layers too, and BMESH.md §10.1.4
    /// says the blend does not touch those at all — whatever the destination block
    /// already holds survives, which after the §5.2.1 step-1 mint is `cd(v)`'s value.
    /// The interpolable layers need no special case either: at a unit weight they
    /// already come out bit-exact under every policy.
    pub(super) fn blend_corner_from_donor(&mut self, corner: LoopKey, donor: &CornerDonor) {
        let sample_position = self.vertex_position(self.loop_vertex(corner));
        self.blend_corner_from_donor_at(corner, donor, sample_position);
    }

    /// The general form of [`blend_corner_from_donor`](Self::blend_corner_from_donor): resample
    /// the donor at an explicit world position rather than at the corner vertex's own.
    ///
    /// This is the mechanism behind §5.2.1.1's **weight-snap** (§5.2.1.1.1): a seam-carrying
    /// cap corner blends from the single override donor face, but its sample point is first
    /// moved onto one of that face's two boundary edges at the beveled vertex. Because the
    /// snapped point lands on a donor boundary edge, §10.1.3's on-edge short-circuit collapses
    /// the mean-value solve to that edge's two endpoints — the edge-linear form that keeps a
    /// seamed corner's value on the seam's own side of the discontinuity. The snap moves the
    /// *sample*, never the vertex, so the geometry is untouched.
    pub(super) fn blend_corner_from_donor_at(
        &mut self,
        corner: LoopKey,
        donor: &CornerDonor,
        sample_position: Vec3,
    ) {
        let vertex = self.loop_vertex(corner);
        let sample = donor.projection.project(sample_position);
        let weights = mean_value_weights(&donor.proj_corners, sample);
        self.loop_attribs.interpolate_from_snapshot(
            corner.as_usize(),
            &donor.corner_loop_blocks,
            &weights,
        );
        self.vertex_attribs.interpolate_from_snapshot(
            vertex.as_usize(),
            &donor.corner_vertex_blocks,
            &weights,
        );
    }

    /// Is `vertex` **seam-carrying** in the sense of BEVEL.md §5.2.1.1 — does *any* incident
    /// leg satisfy §6.6.1.4's corner-level seam test? This is the single all-or-nothing
    /// boolean that decides the whole ring-spanning cap's per-loop donor branch: when true
    /// every cap corner takes the single override donor face plus the §5.2.1.1.1 weight-snap;
    /// when false every corner takes its own boundvert's positional donor.
    ///
    /// Mirrors the seam predicate [`ring_spanning_donor`](Self::ring_spanning_donor) uses to
    /// decide whether to displace the positional block-inherit donor, so the block inherit and
    /// the per-loop blend agree on which branch a corner is on. (§5.2.1.1's odd-`segments`
    /// vertex-level contiguity arm is not modelled here, exactly as it is not there.)
    pub(super) fn corner_is_seam_carrying(&self, vertex: VertexKey) -> bool {
        self.iter_disk_edges(vertex)
            .any(|leg| self.leg_missing_flanking_face(leg) || self.edge_is_interp_seam(leg))
    }

    /// The loop of `face` sitting at `vertex`, or `None` when the face has no corner there.
    pub(super) fn corner_loop_at(&self, face: FaceKey, vertex: VertexKey) -> Option<LoopKey> {
        self.face_loop_keys(face)
            .into_iter()
            .find(|&loop_key| self.loop_vertex(loop_key) == vertex)
    }

    /// The loop layers of the 2-component-float type — BEVEL.md §5.2.6's scope, chosen by
    /// **type alone** (a UV layer or any other float2 loop layer), never by role or name.
    pub(super) fn float2_loop_layers(&self) -> Vec<ElementAttributeKey> {
        let attribs = self.loop_attribs();
        attribs
            .iter_descriptors()
            .filter(|&(key, _)| matches!(attribs.storage_of(key), Some(AttributeStorage::Vec2(_))))
            .map(|(key, _)| key)
            .collect()
    }

    /// Build BEVEL.md §5.2.1.1's **seam-carrying** per-corner blend plan for the cap at a
    /// beveled vertex: the single override donor face `F` reused for every corner, and one
    /// sample position per ring vert — the §5.2.1.1.1 weight-snap.
    ///
    /// `F`'s two edges at the beveled vertex are collected in the face's **loop (winding)
    /// order** (§5.2.1.1.1(i)); a ring vert that projects **inside** `F`'s axis-dominant image
    /// under the ray-crossing parity test (§5.2.1.1.1(ii)) — first three in ring order only
    /// (§5.2.1.1.1(iii)) — is given no snap and samples at its own position, while every other
    /// ring vert snaps onto the nearer of the two edges (squared segment distance, strictly
    /// nearer else the second). Runs pre-surgery, while the edges still meet the beveled
    /// vertex.
    pub(super) fn plan_override_cap(
        &self,
        corner: &BeveledCorner,
        override_face: FaceKey,
        donor: &CornerDonor,
    ) -> Vec<(VertexKey, Vec3)> {
        let vertex = corner.vertex;
        let vertex_position = self.vertex_position(vertex);

        // (i) F's two edges at the beveled vertex, in loop-cycle order, as (near, far) segments.
        let segments: Vec<(Vec3, Vec3)> = self
            .face_edge_keys(override_face)
            .into_iter()
            .filter(|&edge| self.edge_vert_keys(edge).contains(&vertex))
            .map(|edge| {
                let [a, b] = self.edge_vert_keys(edge);
                let far = if a == vertex { b } else { a };
                (vertex_position, self.vertex_position(far))
            })
            .collect();

        // (ii) + (iii): the ring verts that project inside F, capped at three in ring order.
        let mut internal: Vec<VertexKey> = Vec::new();
        for &ring_vertex in &corner.ring_order {
            let projected = donor.projection.project(self.vertex_position(ring_vertex));
            if point_in_polygon_parity(&donor.proj_corners, projected) {
                internal.push(ring_vertex);
                if internal.len() == 3 {
                    break;
                }
            }
        }

        corner
            .ring_order
            .iter()
            .map(|&ring_vertex| {
                let position = self.vertex_position(ring_vertex);
                if internal.contains(&ring_vertex) {
                    return (ring_vertex, position);
                }
                let sample = match segments.as_slice() {
                    [first, second] => {
                        let nearer = if dist_sq_to_segment(position, first.0, first.1)
                            < dist_sq_to_segment(position, second.0, second.1)
                        {
                            first
                        } else {
                            second
                        };
                        closest_point_on_segment(position, nearer.0, nearer.1)
                    }
                    [only] => closest_point_on_segment(position, only.0, only.1),
                    _ => position,
                };
                (ring_vertex, sample)
            })
            .collect()
    }

    /// BEVEL.md §5.2.6's per-beveled-vertex island partition, built pre-surgery from each
    /// float2 layer's own values: the original loops incident to `vertex` are bucketed by
    /// coincidence of their value (summed-squared tolerance `1e-4`, **greedy first-match** —
    /// not transitive closure), and the record is which bucket each incident face's corner
    /// fell in. Persisted because the beveled vertex is freed before the post-pass reads it.
    pub(super) fn build_prepass_buckets(
        &self,
        vertex: VertexKey,
        incident_faces: &[FaceKey],
        layers: &[ElementAttributeKey],
    ) -> HashMap<ElementAttributeKey, HashMap<FaceKey, usize>> {
        let mut partition = HashMap::new();
        for &layer in layers {
            let mut buckets: Vec<Vec<[f32; 2]>> = Vec::new();
            let mut face_bucket: HashMap<FaceKey, usize> = HashMap::new();
            for &face in incident_faces {
                let Some(loop_key) = self.corner_loop_at(face, vertex) else {
                    continue;
                };
                let value = self
                    .loop_attribs()
                    .get_vec2(layer, loop_key.as_usize())
                    .unwrap_or([0.0, 0.0]);
                let joined = buckets.iter().position(|bucket| {
                    bucket
                        .iter()
                        .any(|&member| float2_within_merge_tolerance(member, value))
                });
                let bucket_id = match joined {
                    Some(id) => {
                        buckets[id].push(value);
                        id
                    }
                    None => {
                        buckets.push(vec![value]);
                        buckets.len() - 1
                    }
                };
                face_bucket.insert(face, bucket_id);
            }
            partition.insert(layer, face_bucket);
        }
        partition
    }

    /// BEVEL.md §5.2.6's **UV-merge post-pass**: reconcile the per-loop float2 values that meet
    /// at each new ring / cap vertex. Runs once, at the very end, on every 2-float loop layer,
    /// and only ever changes per-loop values — never topology, never per-vertex data.
    ///
    /// Per new vertex `nv` (originating from beveled vertex `ov`), the loops at `nv` are
    /// **bucketed** greedily (first-match, not transitive closure): a loop joins the first
    /// bucket already holding a neighbour whose face is *island-connected* to it at `ov` — same
    /// representative donor face, or donors that fell in the same pre-pass island bucket
    /// (§5.2.6's `island_connected_at`). Each bucket of two or more loops is overwritten with
    /// its arithmetic mean. The `ra == rb` branch is layer-independent; the pre-pass-bucket
    /// branch is per layer, so each float2 layer buckets and averages on its own values.
    pub(super) fn reconcile_cap_uv_islands(
        &mut self,
        corners: &[BeveledCorner],
        bank: &DonorBank,
        rebuilt: &[RebuiltOriginal],
        cap_faces: &[Option<FaceKey>],
        float2_layers: &[ElementAttributeKey],
        prepass: &[HashMap<ElementAttributeKey, HashMap<FaceKey, usize>>],
    ) {
        if float2_layers.is_empty() {
            return;
        }

        // Each new face's stage-1 representative donor original face (§5.2.6): the cap's is the
        // ring-spanning override / positional pick, a re-stitched original's is the face
        // itself. `bank.index` inverts the re-stitch donor index back to that original face.
        let mut representative_donor: HashMap<FaceKey, FaceKey> = HashMap::new();
        for (corner_index, &cap) in cap_faces.iter().enumerate() {
            if let (Some(cap), Some(donor)) = (cap, corners[corner_index].representative_face) {
                representative_donor.insert(cap, donor);
            }
        }
        let donor_face_of_index: HashMap<usize, FaceKey> = bank
            .index
            .iter()
            .map(|(&face, &index)| (index, face))
            .collect();
        for original in rebuilt {
            if let Some(index) = original.donor {
                if let Some(&face) = donor_face_of_index.get(&index) {
                    representative_donor.insert(original.face, face);
                }
            }
        }

        // Which beveled corner each ring vert originated from (`ov`), collected in a stable
        // order so the pass is reproducible.
        let mut corner_of_ring: HashMap<VertexKey, usize> = HashMap::new();
        let mut ring_verts: Vec<VertexKey> = Vec::new();
        for (corner_index, corner) in corners.iter().enumerate() {
            for &(_, ring_vertex) in &corner.ring_vert_for_edge {
                if corner_of_ring.insert(ring_vertex, corner_index).is_none() {
                    ring_verts.push(ring_vertex);
                }
            }
        }

        for ring_vertex in ring_verts {
            let corner_index = corner_of_ring[&ring_vertex];
            let loops: Vec<(LoopKey, FaceKey)> = self
                .vert_face_keys(ring_vertex)
                .into_iter()
                .filter_map(|face| self.corner_loop_at(face, ring_vertex).map(|l| (l, face)))
                .collect();
            if loops.len() < 2 {
                continue;
            }
            for &layer in float2_layers {
                let mut buckets: Vec<Vec<LoopKey>> = Vec::new();
                let mut bucket_faces: Vec<Vec<FaceKey>> = Vec::new();
                for &(loop_key, face) in &loops {
                    let joined = (0..buckets.len()).find(|&bucket| {
                        bucket_faces[bucket].iter().any(|&other| {
                            island_connected_at(
                                layer,
                                corner_index,
                                face,
                                other,
                                &representative_donor,
                                prepass,
                            )
                        })
                    });
                    match joined {
                        Some(bucket) => {
                            buckets[bucket].push(loop_key);
                            bucket_faces[bucket].push(face);
                        }
                        None => {
                            buckets.push(vec![loop_key]);
                            bucket_faces.push(vec![face]);
                        }
                    }
                }
                for bucket in &buckets {
                    if bucket.len() < 2 {
                        continue;
                    }
                    let mut sum = [0.0f32; 2];
                    for &loop_key in bucket {
                        let value = self
                            .loop_attribs()
                            .get_vec2(layer, loop_key.as_usize())
                            .unwrap_or([0.0, 0.0]);
                        sum[0] += value[0];
                        sum[1] += value[1];
                    }
                    let count = bucket.len() as f32;
                    let mean = [sum[0] / count, sum[1] / count];
                    for &loop_key in bucket {
                        self.loop_attribs_mut()
                            .set_vec2(layer, loop_key.as_usize(), mean);
                    }
                }
            }
        }
    }

    /// Blend every corner of a re-stitched original from that face's own pre-bevel
    /// polygon (BEVEL.md §5.2.1.1). Runs on **all** its corners, the untouched
    /// originals included: those sample at their own donor corner, take weight `1`
    /// by the §10.1.3 on-corner short-circuit, and so keep their block exactly
    /// (§5.2.1 property (3)) — their interpolable layers because a unit weight
    /// reproduces the source value under every policy, their no-interp layers
    /// because the blend never writes those at all (BMESH.md §10.1.4). A rebuilt
    /// face must not perturb the vertex data of the mesh around it, and it must
    /// hand every surviving corner of its own back the loop values the killed face
    /// carried there.
    ///
    /// The **last** write of the run in both domains (§5.2.1 fact (c)), so every
    /// caller runs it after its own fills.
    pub(super) fn blend_restitched_originals(
        &mut self,
        rebuilt: &[RebuiltOriginal],
        bank: &DonorBank,
    ) {
        for original in rebuilt {
            let Some(index) = original.donor else {
                continue;
            };
            let donor = bank.donor(index);
            for corner in self.face_loop_keys(original.face) {
                self.blend_corner_from_donor(corner, donor);
            }
        }
    }

    /// Blend the single-segment strip quad's **own** four loops (BEVEL.md §5.2.5, the
    /// per-loop half of §5.2.1.1's third bullet). The quad owns these loops outright — no
    /// re-stitched original writes them — so an unblended corner keeps the layer default,
    /// exactly what the §7-fills' stub still does elsewhere.
    ///
    /// Where the quad's per-**face** block takes only the near flank (§6.6.2's strip row),
    /// its per-loop donor list is per corner, ordered **near/near/far/far**: each strip vert
    /// sits on a *leg* of one flank of `E` (§5.2.3), so its loop blends **that flank's**
    /// pre-bevel corner blocks. `near_flank` is `E`'s stored-first radial face (the quad's
    /// block donor); `far_flank` the other. §10.1.3's on-edge short-circuit collapses each
    /// solve to the leg-linear closed form at the strip vert's final position, so a
    /// position-contiguous layer reproduces `f(position)` exactly and either flank yields the
    /// same number — but the split is nailed down now so the seam-carrying override (a later
    /// row) only *adds* an override donor rather than reworking this mapping.
    ///
    /// Both domains ride one solve, so this also writes each strip vert's per-vertex block
    /// from its flank; the re-stitched originals overwrite that last (§5.2.1 fact (c)) with
    /// the identical value (the same flank donates the strip vert in the re-stitch), so the
    /// order is a formality here, not a value choice.
    pub(super) fn blend_strip_quad_loops(
        &mut self,
        strip_face: FaceKey,
        near_flank: FaceKey,
        far_flank: FaceKey,
        strip_verts_on_near: [VertexKey; 2],
        strip_verts_on_far: [VertexKey; 2],
        bank: &DonorBank,
    ) {
        if bank.is_empty() {
            return;
        }
        let near = bank.index_of(near_flank);
        let far = bank.index_of(far_flank);
        for corner in self.face_loop_keys(strip_face) {
            let vertex = self.loop_vertex(corner);
            let donor = if strip_verts_on_near.contains(&vertex) {
                near
            } else if strip_verts_on_far.contains(&vertex) {
                far
            } else {
                None
            };
            if let Some(donor) = donor {
                self.blend_corner_from_donor(corner, bank.donor(donor));
            }
        }
    }

    /// Assign the per-face attributes of one **new** bevel face (BEVEL.md §6.6), in the
    /// two independent stages the parameter is actually defined by:
    ///
    /// 1. **Inherit**, whatever `material` is: copy `representative`'s **whole per-face
    ///    attribute block** onto the new face (§6.6.1) — every per-face layer, with the
    ///    material slot riding along as one field of it. The representative is a
    ///    per-new-face choice (cap ← incident face of `v`, strip ← flanking face, patch /
    ///    fill ← junction-adjacent face), captured pristine by the caller before the
    ///    adjacent originals are killed / rebuilt. A fill that supplies **no**
    ///    representative mints template-less and keeps the layer defaults (§6.6.1.3 — the
    ///    cutoff fill).
    /// 2. **Stamp**, iff `material >= 0`: overwrite the material-slot field *alone*. The
    ///    rest of the inherited block stands. This is why the inherit above cannot be
    ///    skipped under an explicit slot — that would give the right slot over a default
    ///    block.
    ///
    /// Surviving re-stitched faces are never routed here, so they keep their own
    /// block and are never re-materialised (§6.6.2).
    ///
    /// This is also the single chokepoint every **new** bevel face flows through,
    /// so it doubles as the §8.10.1 provenance recorder: while a face-strength pass
    /// is active (`bevel_cap_faces` is `Some`), each [`BevelNewFaceKind::Cap`] face
    /// is recorded so the post-pass can stamp it weak. Strips need no recording —
    /// their medium sentinel equals the layer default.
    pub(super) fn assign_bevel_face_attributes(
        &mut self,
        face: FaceKey,
        material: i32,
        representative: FaceTemplateRef<'_>,
        kind: BevelNewFaceKind,
    ) {
        if let Some((blocks, donor)) = representative.block {
            self.face_attribs
                .restore_slot(face.as_usize(), blocks, donor);
        }
        if material >= 0 {
            self.set_face_material_index(face, material);
        }

        if kind == BevelNewFaceKind::Cap {
            if let Some(cap_faces) = self.bevel_cap_faces.as_mut() {
                cap_faces.push(face);
            }
        }
    }
}
