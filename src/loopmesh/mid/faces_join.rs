use crate::loopmesh::*;

impl LoopMesh {
    /// Joins N faces into one by repeatedly killing edges shared between
    /// pairs of remaining faces. Returns the surviving face, or `None` if
    /// the input faces are not transitively connected by shared edges.
    ///
    /// This is the iterative-JFKE form. For complex N-face joins
    /// (multiply-connected regions), a single boundary-walk would assemble the
    /// result directly; this iterative form matches the topology outcome on the
    /// common simply-connected case.
    pub fn faces_join(&mut self, faces: &[FaceKey]) -> Option<FaceKey> {
        if faces.is_empty() {
            return None;
        }
        if faces.len() == 1 {
            return Some(faces[0]);
        }
        let mut remaining: Vec<FaceKey> = faces.to_vec();

        'outer: loop {
            if remaining.len() == 1 {
                return Some(remaining[0]);
            }
            // Find any edge shared by two remaining faces (radial cycle of
            // length 2 with both faces in `remaining`).
            for (i, &fa) in remaining.iter().enumerate() {
                let edges: smallvec::SmallVec<[EdgeKey; 8]> =
                    collect_iter::<_, 8>(self, FaceLoops::new(self, fa))
                        .into_iter()
                        .map(|l| self.loops.get(l).edge)
                        .filter(|e| !e.is_null())
                        .collect();
                for &e in &edges {
                    // Read the 2-radial face pair via the lending iterator.
                    let mut rl = self.radial_loops(e);
                    let l0 = match rl.next(self) {
                        Some(l) => l,
                        None => continue,
                    };
                    let l1 = match rl.next(self) {
                        Some(l) => l,
                        None => continue,
                    };
                    if rl.next(self).is_some() {
                        continue;
                    } // not 2-manifold
                    let f0 = self.loops.get(l0).face;
                    let f1 = self.loops.get(l1).face;
                    let other = if f0 == fa {
                        f1
                    } else if f1 == fa {
                        f0
                    } else {
                        continue;
                    };
                    if other == fa {
                        continue;
                    }
                    if let Some(j) = remaining.iter().position(|&f| f == other) {
                        // Join fa and other along e. fa survives.
                        self.euler_join_face_kill_edge(fa, other, e)?;
                        remaining.swap_remove(j);
                        let _ = i;
                        continue 'outer;
                    }
                }
            }
            // No more shared edges available — the input wasn't fully connected.
            return None;
        }
    }
}
