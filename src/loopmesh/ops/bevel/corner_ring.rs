//! The rotational corner ring around a vertex and its sectors (BEVEL.md §7.6.1 /
//! §7.6.1.2 / §7.8.1).

use super::*;

impl LoopMesh {
    /// One corner's face normal oriented to the **walk-positive** side (BEVEL.md §7.6.2):
    /// the face interior sweeps positively about this normal from the corner's outgoing
    /// edge round to its incoming one, so the wedge fixes the side and the true face plane
    /// fixes the tilt (a non-planar reflex quad keeps its Newell normal, not the flat wedge
    /// one). `None` when either flanking edge is degenerate.
    ///
    /// The side is fixed in the **ring's** direction of travel, so a shell wound wholly
    /// inward reads exactly as the same shell wound outward (the ring reverses with it, and
    /// every fold's two normals reverse together). What does *not* survive that reversal is a
    /// face whose own loop cycle **disagrees** with the ring — §7.8.1(b)'s per-sector ballot
    /// read for one corner, the `−1` term — and such a face contributes its plane upside down
    /// relative to its neighbours, so [`Self::corner_walk_normal_dissents`] negates it. On a
    /// consistently wound fan nothing dissents and the flip never fires; on the
    /// inconsistently wound fans of §7.8 it is the whole of what makes §8.5.1's dihedral sum
    /// a genuinely *signed* quantity rather than a sum of magnitudes.
    ///
    /// That ballot flip is what makes this the **band's** normal rather than a general-purpose
    /// oriented corner normal: it is chosen so that folding one corner's normal against its
    /// neighbour's reads the §8.5.1 dihedral, and a caller wanting the face's own geometric
    /// side wants [`Self::face_plane_normal`] instead.
    pub(super) fn corner_walk_normal(
        &self,
        corner: &VertexCorner,
        vertex: VertexKey,
        vertex_position: Vec3,
    ) -> Option<Vec3> {
        let d_in = self.edge_direction_from(corner.incoming_edge, vertex, vertex_position)?;
        let d_out = self.edge_direction_from(corner.outgoing_edge, vertex, vertex_position)?;
        let wedge = d_out.cross(d_in);
        let mut normal = self.face_plane_normal(corner.face);
        if normal.dot(wedge) < 0.0 {
            normal = -normal;
        }
        if self.corner_walk_normal_dissents(corner, vertex) {
            normal = -normal;
        }
        Some(normal)
    }

    /// §7.8.1(b): whether this corner's face carries the corner's two edges **against** the
    /// ring's direction of travel — the `−1` term of the [`Self::corner_ring_winding_tally`]
    /// ballot, read for one sector instead of summed. A corner whose face presents neither
    /// order (nothing to read) does not dissent.
    pub(super) fn corner_walk_normal_dissents(
        &self,
        corner: &VertexCorner,
        vertex: VertexKey,
    ) -> bool {
        self.corner_winding_vote(corner, vertex) < 0
    }

    /// One corner's vote in the BEVEL.md §7.8.1(a) / §7.6.1.2 step 2 winding ballot: `+1` when
    /// the corner's own face leaves `vertex` along the corner's **incoming** edge and arrives
    /// along its **outgoing** one (the §7.6.2 walk sense), `−1` when that face carries the
    /// pair the other way round, `0` when the corner presents no such face.
    ///
    /// The ring's aggregate ([`Self::corner_ring_winding_tally`]) and the band's per-sector
    /// read ([`Self::corner_walk_normal_dissents`]) are the same ballot at two scales, so they
    /// share this one reading rather than each spelling the two orders out.
    pub(super) fn corner_winding_vote(&self, corner: &VertexCorner, vertex: VertexKey) -> i32 {
        let Some((arrives_on, leaves_on)) = self.face_corner_edges(corner.face, vertex) else {
            return 0;
        };
        if corner.incoming_edge == leaves_on && corner.outgoing_edge == arrives_on {
            1
        } else if corner.incoming_edge == arrives_on && corner.outgoing_edge == leaves_on {
            -1
        } else {
            0
        }
    }

    /// The rotational ring of (edge-pair, face) corners around `vertex`: each
    /// incident face contributes its two edges meeting at `vertex`. Returned as a
    /// cyclic sequence of `VertexCorner { incoming_edge, face, outgoing_edge }`
    /// chained so consecutive corners share an edge — the disk walk order.
    pub(super) fn vertex_corner_ring(&self, vertex: VertexKey) -> Vec<VertexCorner> {
        // Per incident face: its two edges at `vertex`, ordered so that the ring
        // *enters* the corner on the edge the face **leaves** `vertex` on (its corner
        // edge, RADIAL_EDGE_STRUCTURE.md §6.5) and exits on the edge the face arrives
        // at `vertex` on. That makes the walk run against the face winding, uniformly
        // — so the face "just past" a corner's opening edge is that corner's own face,
        // which is the reference normal BEVEL.md §8.9.1's signed reflex test names.
        //
        // The order must come off the winding, not off whichever of the two edges a
        // face-loop scan happens to reach first: that would let the seed corner face
        // either way round, leaving the ring's rotational direction a property of the
        // fan's start face rather than of the surface orientation, and flipping the
        // reflex sign on fixtures that merely built their faces in a different order.
        let mut face_edges: Vec<(FaceKey, EdgeKey, EdgeKey)> = Vec::new();
        for face in self.vert_face_keys(vertex) {
            if let Some((arrives_on, leaves_on)) = self.face_corner_edges(face, vertex) {
                if arrives_on != leaves_on {
                    face_edges.push((face, leaves_on, arrives_on));
                }
            }
        }

        // Chain faces edge-to-edge. Start anywhere; follow the shared edge into
        // the next face until the ring closes.
        let mut ring: Vec<VertexCorner> = Vec::with_capacity(face_edges.len());
        if face_edges.is_empty() {
            return ring;
        }

        let mut used = vec![false; face_edges.len()];
        let (start_face, start_e0, start_e1) = face_edges[0];
        used[0] = true;
        ring.push(VertexCorner {
            incoming_edge: start_e0,
            face: start_face,
            outgoing_edge: start_e1,
        });
        let mut frontier_edge = start_e1;

        while ring.len() < face_edges.len() {
            let next = face_edges.iter().enumerate().find(|(index, (_, e0, e1))| {
                !used[*index] && (*e0 == frontier_edge || *e1 == frontier_edge)
            });
            match next {
                Some((index, &(face, e0, e1))) => {
                    used[index] = true;
                    let outgoing = if e0 == frontier_edge { e1 } else { e0 };
                    ring.push(VertexCorner {
                        incoming_edge: frontier_edge,
                        face,
                        outgoing_edge: outgoing,
                    });
                    frontier_edge = outgoing;
                }
                None => break, // open fan or non-manifold — stop chaining
            }
        }

        // The start face is arbitrary, so on an open fan (a boundary vertex) it can sit
        // mid-fan and the forward walk reaches only the faces after it. Walk backward off
        // the start's other edge too, or the ring is silently truncated — and a truncated
        // ring loses whole sectors and mis-answers the rotational-neighbour query.
        let mut leading: Vec<VertexCorner> = Vec::new();
        let mut back_edge = start_e0;
        while ring.len() + leading.len() < face_edges.len() {
            let previous = face_edges.iter().enumerate().find(|(index, (_, e0, e1))| {
                !used[*index] && (*e0 == back_edge || *e1 == back_edge)
            });
            match previous {
                Some((index, &(face, e0, e1))) => {
                    used[index] = true;
                    let incoming = if e0 == back_edge { e1 } else { e0 };
                    leading.push(VertexCorner {
                        incoming_edge: incoming,
                        face,
                        outgoing_edge: back_edge,
                    });
                    back_edge = incoming;
                }
                None => break,
            }
        }

        let ring = if leading.is_empty() {
            ring
        } else {
            leading.reverse();
            leading.extend(ring);
            leading
        };

        self.normalise_corner_ring(vertex, ring)
    }

    /// Normalise the rotational order's handedness (BEVEL.md §7.8.1(a)). The chained walk
    /// above seeds on an arbitrary incident face, so on a mesh whose faces do **not** all
    /// agree on direction the seed decides which way round the ring runs — and the §7.5.1
    /// sign reference reads that handedness. One ballot per sector: `+1` when the sector's
    /// face traverses the sector's two edges in the same sense the ring walks them, `−1`
    /// when it runs them the other way, `0` when the face presents no such corner. A
    /// **strictly negative** total reverses the ring; a tie and a positive total leave it
    /// alone.
    ///
    /// Reversing swaps every corner's two edges as well as the sequence, so the sector
    /// lying between a given pair of edges is still that same pair's sector afterwards —
    /// dropping that swap would read every sector's face off by one and mis-sign the leg
    /// meet on ordinary meshes too.
    pub(super) fn normalise_corner_ring(
        &self,
        vertex: VertexKey,
        ring: Vec<VertexCorner>,
    ) -> Vec<VertexCorner> {
        if self.corner_ring_winding_tally(vertex, &ring) >= 0 {
            return ring;
        }
        reverse_corner_ring(ring)
    }

    /// The winding ballot of BEVEL.md §7.8.1(a) / §7.6.1.2 step 2 ([`Self::corner_winding_vote`]),
    /// summed over a corner ring.
    ///
    /// The ballot reads each corner independently, so it is invariant under rotating the
    /// ring's origin and negates exactly when the ring is reversed.
    pub(super) fn corner_ring_winding_tally(
        &self,
        vertex: VertexKey,
        ring: &[VertexCorner],
    ) -> i32 {
        ring.iter()
            .map(|corner| self.corner_winding_vote(corner, vertex))
            .sum()
    }

    /// The junction's **slot array** in the sense of BEVEL.md §7.6.1.2: the corner ring of
    /// [`vertex_corner_ring`] re-anchored so slot 0 is the origin the guide pins, instead of
    /// wherever the chained walk's face-order seed happened to land. `is_beveled` is step 1's
    /// bevel selection.
    ///
    /// Only the *origin* is recovered here, because the chained ring already runs in the
    /// guide's final direction wherever the winding ballot is non-zero: both orientations are
    /// settled by the same "reverse iff the ballot is strictly negative" rule, and the ballot
    /// is origin-invariant. The one direction the two can disagree on is a **zero** ballot,
    /// which the guide resolves toward the walk out of the seed rather than toward whatever
    /// the chain built, so that case reverses here.
    ///
    /// An **open** fan needs neither: both of its ends are radial-1 and so cannot be in the
    /// selection (§2.1), which puts step 1's seed on an end and leaves step 3's scan falling
    /// through to the far end — the origin the chained ring already carries.
    pub(super) fn vertex_bevel_corner_ring(
        &self,
        vertex: VertexKey,
        is_beveled: &dyn Fn(EdgeKey) -> bool,
    ) -> Vec<VertexCorner> {
        let ring = self.vertex_corner_ring(vertex);
        let count = ring.len();
        if count < 2 {
            return ring;
        }
        let closed = ring[count - 1].outgoing_edge == ring[0].incoming_edge;
        let trailing_spoke = (!closed).then(|| ring[count - 1].outgoing_edge);

        // Step 1: scan the stored disk order from the anchor. A radial-1 faced spoke outranks
        // a beveled one and the *last* such wins; failing that, the *first* beveled spoke.
        let mut seed: Option<EdgeKey> = None;
        for edge in self.iter_disk_edges(vertex) {
            let in_array = trailing_spoke == Some(edge)
                || ring.iter().any(|corner| corner.incoming_edge == edge);
            if !in_array {
                continue;
            }
            if self.edge_radial_iter(edge).count() == 1 {
                seed = Some(edge);
            } else if seed.is_none() && is_beveled(edge) {
                seed = Some(edge);
            }
        }
        let Some(seed) = seed else {
            return ring;
        };

        // Step 2: the walk's first step leaves the seed for whichever of its fan neighbours
        // comes first in that same stored order.
        let seed_slot = ring.iter().position(|corner| corner.incoming_edge == seed);
        let forward_neighbour = seed_slot.map(|slot| ring[slot].outgoing_edge);
        let backward_neighbour = match seed_slot {
            Some(0) if !closed => None,
            Some(slot) => Some(ring[(slot + count - 1) % count].incoming_edge),
            None => Some(ring[count - 1].incoming_edge),
        };
        let stored_rank = |edge: EdgeKey| {
            self.iter_disk_edges(vertex)
                .position(|candidate| candidate == edge)
                .unwrap_or(usize::MAX)
        };
        let walk_follows_ring = match (forward_neighbour, backward_neighbour) {
            (Some(forward), Some(backward)) => stored_rank(forward) < stored_rank(backward),
            (_, None) => true,
            (None, Some(_)) => false,
        };

        let mut ring = ring;
        let mut walk_follows_ring = walk_follows_ring;
        if !walk_follows_ring && self.corner_ring_winding_tally(vertex, &ring) == 0 {
            ring = reverse_corner_ring(ring);
            walk_follows_ring = true;
        }
        if !closed {
            return ring;
        }

        // An unreversed array leaves the seed in slot 0; a reversed one parks it in the last
        // slot, so slot 0 is the seed's successor.
        let seed_slot = ring
            .iter()
            .position(|corner| corner.incoming_edge == seed)
            .unwrap_or(0);
        let origin_slot = if walk_follows_ring {
            seed_slot
        } else {
            (seed_slot + 1) % count
        };
        ring.rotate_left(origin_slot);
        ring
    }
}

/// One corner of the rotational ring around a vertex: an incident face flanked by
/// its two edges meeting at that vertex (BEVEL.md §7.6.1 disk walk). Consecutive
/// ring entries share an edge.
#[derive(Clone, Copy)]
pub(super) struct VertexCorner {
    pub(super) incoming_edge: EdgeKey,
    pub(super) face: FaceKey,
    pub(super) outgoing_edge: EdgeKey,
}

/// One angular sector of a junction (BEVEL.md §7.6.1): the run of faces between
/// two consecutive beveled edges around the disk. `bounding_edges` are the two
/// beveled edges delimiting it (the offset-line sources for the meet solve);
/// `corners` are every (edge, face) pair inside it.
pub(super) struct Sector {
    pub(super) bounding_edges: Vec<EdgeKey>,
    pub(super) corners: Vec<(EdgeKey, FaceKey)>,
}

/// The junction's distinct incident edges, in corner-walk order (BEVEL.md §7.6.2): every
/// corner contributes its incoming and outgoing edge and the pair-shared ones collapse, so a
/// closed fan yields one entry per spoke. This is the count the §8.5.1 band means by
/// `edgecount`, and the **valence** §8.5.5.5(f.1) step 3 keys on — which is *not* the
/// boundvert count, since a sector absorbs any unbeveled in-plane leg running through it.
pub(super) fn junction_incident_edges(corners: &[VertexCorner]) -> Vec<EdgeKey> {
    let mut edges: Vec<EdgeKey> = Vec::with_capacity(corners.len() + 1);
    for corner in corners {
        for edge in [corner.incoming_edge, corner.outgoing_edge] {
            if !edges.contains(&edge) {
                edges.push(edge);
            }
        }
    }
    edges
}

/// `edge`'s two neighbours in the vertex's **rotational** edge order — the order §7.6.1
/// walks the fan in, and the one §8.2.2.1's tier-2 gate means by an edge's "disk
/// neighbours". The corner ring is the only carrier of that order: the radial structure's
/// per-vertex edge cycle is linked in insertion order and its successor is generally not
/// the next edge *around* the vertex.
///
/// Returned as `(predecessor, successor)` in the ring's own direction, so the predecessor
/// flanks the sector that opens before `edge` — the sector whose boundvert is a crossing
/// chain's `start`. `None` at a fan end (an open fan's first / last edge has no
/// wrap-around neighbour, so no tier-2 pair exists there).
pub(super) fn rotational_edge_neighbours(
    corners: &[VertexCorner],
    edge: EdgeKey,
) -> Option<(EdgeKey, EdgeKey)> {
    let first = corners.first()?;
    let last = corners.last()?;
    let closed = last.outgoing_edge == first.incoming_edge;
    let index = corners.iter().position(|c| c.incoming_edge == edge)?;
    let predecessor = if index > 0 {
        corners[index - 1].incoming_edge
    } else if closed {
        last.incoming_edge
    } else {
        return None;
    };
    Some((predecessor, corners[index].outgoing_edge))
}

/// Reverse a corner ring, exchanging each spoke's forward and backward flanking faces
/// along with its slot (BEVEL.md §7.6.1.2, "Reversal carries the flanking-face roles with
/// it"): the sector lying between a given pair of edges is still that pair's sector
/// afterwards. Dropping the swap yields an array that is correctly ordered and uniformly
/// wrong — the meet-point comes out at the point reflection of the right answer, which no
/// normal-facing check detects.
pub(super) fn reverse_corner_ring(ring: Vec<VertexCorner>) -> Vec<VertexCorner> {
    ring.into_iter()
        .rev()
        .map(|corner| VertexCorner {
            incoming_edge: corner.outgoing_edge,
            face: corner.face,
            outgoing_edge: corner.incoming_edge,
        })
        .collect()
}

/// Split a vertex's rotational corner ring into sectors at the beveled edges
/// (BEVEL.md §7.6.1). Each sector runs from one beveled edge, through any unbeveled
/// edges/faces, to the next beveled edge, and yields one boundvert. Returns one
/// `Sector` per gap between consecutive beveled edges in disk order.
///
/// `corners` must be the §7.6.1.2 slot array ([`LoopMesh::vertex_bevel_corner_ring`]): the
/// origin scan below is stated on that array's slot numbering and is off by a whole ring
/// step on any other.
pub(super) fn sector_split(
    corners: &[VertexCorner],
    is_beveled: &dyn Fn(EdgeKey) -> bool,
) -> Vec<Sector> {
    if corners.is_empty() {
        return Vec::new();
    }
    let count = corners.len();

    // §7.6.1.2 step 3: the first sector opens on the first beveled spoke found scanning from
    // the array's **last** slot and wrapping — not from slot 0. The two readings part company
    // exactly when the last slot is beveled, and the resulting one-step rotation is what
    // hands a two-collinear-pair ring the wrong pipe axis (§8.3.4.1 condition 2).
    //
    // An open fan's true last slot is the trailing spoke `corners[count - 1].outgoing_edge`,
    // which has no corner of its own; being a fan end it is radial-1 and so never in the
    // selection (§2.1), so the scan falls through it to slot 0.
    let scan_start = if corners[count - 1].outgoing_edge == corners[0].incoming_edge {
        count - 1
    } else {
        0
    };
    let start = (0..count)
        .map(|step| (scan_start + step) % count)
        .find(|&slot| is_beveled(corners[slot].incoming_edge))
        .unwrap_or(0);

    let mut sectors: Vec<Sector> = Vec::new();
    let mut current = Sector {
        bounding_edges: vec![corners[start].incoming_edge],
        corners: Vec::new(),
    };

    for step in 0..count {
        let corner = corners[(start + step) % count];
        current.corners.push((corner.incoming_edge, corner.face));
        // A different (outgoing) beveled edge closes the current sector and opens
        // the next.
        if is_beveled(corner.outgoing_edge) {
            current.corners.push((corner.outgoing_edge, corner.face));
            if !current.bounding_edges.contains(&corner.outgoing_edge) {
                current.bounding_edges.push(corner.outgoing_edge);
            }
            sectors.push(current);
            current = Sector {
                bounding_edges: vec![corner.outgoing_edge],
                corners: Vec::new(),
            };
        }
    }

    sectors
}
