//! Deterministic RNG + improved-Perlin turbulence underpinning the
//! `subdivide_edges` fractal displacement
//! ([`SUBDIVIDE_EDGES.md`](../../../../../radial-mesh-guides/operators/SUBDIVIDE_EDGES.md)
//! §9.2.2 / §9.2.2.1 / §9.2.3.1 / §9.2.3.2).
//!
//! Everything here is pure integer / IEEE-754 float arithmetic with no mesh
//! dependency, so the same `seed` always produces the same offset and the same
//! sample point always produces the same turbulence value (§9.2.5). The two
//! consumers are the §9.2.1 once-consumed seed → `fractal_offset` mapping
//! (`SeededRng`) and the §9.2.3 per-vert turbulence (`turbulence`).

/// The 256-byte base permutation that both the §9.2.2 RNG whitening and the
/// §9.2.3 improved-Perlin gradient lookup share (§9.2.2.1). Pure reference
/// data — a fixed permutation of `0..=255`, reproduced verbatim.
const PERM_BASE: [u8; 256] = [
    162, 160, 25, 59, 248, 235, 170, 238, 243, 28, 103, 40, 29, 237, 0, 222, 149, 46, 220, 63, 58,
    130, 53, 77, 108, 186, 54, 208, 246, 12, 121, 50, 209, 89, 244, 8, 139, 99, 137, 47, 184, 180,
    151, 131, 242, 143, 24, 199, 81, 20, 101, 135, 72, 32, 66, 168, 128, 181, 64, 19, 178, 34, 126,
    87, 188, 127, 107, 157, 134, 76, 200, 219, 124, 213, 37, 78, 90, 85, 116, 80, 205, 179, 122,
    187, 195, 203, 182, 226, 228, 236, 253, 152, 11, 150, 211, 158, 92, 161, 100, 241, 129, 97,
    225, 196, 36, 114, 73, 140, 144, 75, 132, 52, 56, 171, 120, 202, 31, 1, 215, 147, 17, 193, 88,
    169, 49, 249, 68, 109, 191, 51, 156, 95, 9, 148, 163, 133, 6, 198, 154, 30, 123, 70, 21, 48,
    39, 43, 27, 113, 60, 91, 214, 111, 98, 172, 79, 194, 192, 14, 177, 35, 167, 223, 71, 176, 119,
    105, 5, 233, 230, 231, 118, 115, 15, 254, 110, 155, 86, 239, 18, 165, 55, 252, 174, 217, 3,
    142, 221, 16, 185, 206, 201, 141, 218, 42, 189, 104, 23, 159, 190, 212, 10, 204, 210, 232, 67,
    61, 112, 183, 2, 125, 153, 216, 13, 96, 138, 4, 44, 62, 146, 229, 175, 83, 7, 224, 41, 166,
    197, 227, 245, 247, 74, 65, 38, 106, 22, 94, 82, 45, 33, 173, 240, 145, 255, 234, 84, 250, 102,
    26, 69, 57, 207, 117, 164, 136, 251, 93,
];

/// The doubled 512-entry permutation (§9.2.2.1): `PERM[i] = PERM_BASE[i & 255]`.
/// The §9.2.3 noise's additive index fold can exceed 255, so the doubling lets
/// every gather read without a runtime wrap.
const PERM: [u8; 512] = {
    let mut p = [0u8; 512];
    let mut i = 0;
    while i < 512 {
        p[i] = PERM_BASE[i & 255];
        i += 1;
    }
    p
};

/// The §9.2.2 RNG: a 48-bit linear congruential generator with the classic
/// `java.util.Random` constants, settled through the three-round hash whitening
/// of [`SeededRng::new_hashed`]. Consumed exactly three times per operator run
/// (§9.2.1) to derive the fixed `fractal_offset`.
pub(super) struct SeededRng {
    state: u64,
}

const MULTIPLIER: u64 = 0x5DEECE66D;
const ADDEND: u64 = 0xB;
const MASK_48: u64 = 0x0000_FFFF_FFFF_FFFF;

impl SeededRng {
    /// §9.2.2 — the hashed seeding the subdivide operator uses. A three-round
    /// whitening mixes the raw seed through `PERM_BASE` (indexed only at
    /// `& 0xFF`) before settling the 48-bit state, so small similar integer
    /// seeds still produce well-separated offsets.
    pub(super) fn new_hashed(seed: u32) -> Self {
        let mut state = seed_plain(seed.wrapping_add(PERM_BASE[(seed & 0xFF) as usize] as u32));
        let s1 = next_u32(&mut state);
        state = seed_plain(s1.wrapping_add(PERM_BASE[(s1 & 0xFF) as usize] as u32));
        let s2 = next_u32(&mut state);
        state = seed_plain(s2.wrapping_add(PERM_BASE[(s2 & 0xFF) as usize] as u32));
        Self { state }
    }

    /// §9.2.2 — one float draw in `[0, 1)`: the top 31 bits of the next 48-bit
    /// state reinterpreted as a non-negative signed integer over `2^31`.
    pub(super) fn next_float(&mut self) -> f32 {
        let i = next_u32(&mut self.state) as i32;
        (i as f32) / 2_147_483_648.0_f32
    }
}

/// §9.2.2 — pack a 32-bit seed into the high bits of the 48-bit state with the
/// fixed low constant `0x330E`.
fn seed_plain(seed: u32) -> u64 {
    const LOW: u64 = 0x330E;
    ((seed as u64) << 16) | LOW
}

/// §9.2.2 — one LCG step then take the top 31 bits (a right shift by 17).
fn next_u32(state: &mut u64) -> u32 {
    *state = MULTIPLIER.wrapping_mul(*state).wrapping_add(ADDEND) & MASK_48;
    (*state >> 17) as u32
}

/// §9.2.3.1 — the quintic improved-Perlin fade `6t^5 - 15t^4 + 10t^3`, in the
/// nested (Horner) form. *Not* the cubic smoothstep.
fn fade(t: f32) -> f32 {
    t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
}

/// §9.2.3.1 — linear interpolation, `t`-first: `a` at `t = 0`, `b` at `t = 1`.
fn lerp(t: f32, a: f32, b: f32) -> f32 {
    a + t * (b - a)
}

/// §9.2.3.1 — map the low 4 bits of a hashed corner value to one of 12 gradient
/// directions (cube edge-midpoints) and return that gradient's dot product with
/// the corner offset. The 16 codes fold onto 12 directions via the `v` split.
fn grad(hash_value: u8, x: f32, y: f32, z: f32) -> f32 {
    let h = (hash_value & 15) as u32;
    let u = if h < 8 { x } else { y };
    let v = if h < 4 {
        y
    } else if h == 12 || h == 14 {
        x
    } else {
        z
    };
    (if h & 1 == 0 { u } else { -u }) + (if h & 2 == 0 { v } else { -v })
}

/// §9.2.3.1 — the single-octave improved-Perlin noise, centred on `0` and
/// bounded by roughly `±0.866`.
fn perlin_signed(mut x: f32, mut y: f32, mut z: f32) -> f32 {
    let fx = x.floor();
    let fy = y.floor();
    let fz = z.floor();
    let x_cell = (fx as i32) & 255;
    let y_cell = (fy as i32) & 255;
    let z_cell = (fz as i32) & 255;
    x -= fx;
    y -= fy;
    z -= fz;

    let u = fade(x);
    let v = fade(y);
    let w = fade(z);

    let a = (PERM[x_cell as usize] as i32) + y_cell;
    let aa = (PERM[a as usize] as i32) + z_cell;
    let ab = (PERM[(a + 1) as usize] as i32) + z_cell;
    let b = (PERM[(x_cell + 1) as usize] as i32) + y_cell;
    let ba = (PERM[b as usize] as i32) + z_cell;
    let bb = (PERM[(b + 1) as usize] as i32) + z_cell;

    lerp(
        w,
        lerp(
            v,
            lerp(
                u,
                grad(PERM[aa as usize], x, y, z),
                grad(PERM[ba as usize], x - 1.0, y, z),
            ),
            lerp(
                u,
                grad(PERM[ab as usize], x, y - 1.0, z),
                grad(PERM[bb as usize], x - 1.0, y - 1.0, z),
            ),
        ),
        lerp(
            v,
            lerp(
                u,
                grad(PERM[(aa + 1) as usize], x, y, z - 1.0),
                grad(PERM[(ba + 1) as usize], x - 1.0, y, z - 1.0),
            ),
            lerp(
                u,
                grad(PERM[(ab + 1) as usize], x, y - 1.0, z - 1.0),
                grad(PERM[(bb + 1) as usize], x - 1.0, y - 1.0, z - 1.0),
            ),
        ),
    )
}

/// §9.2.3.1 — the signed noise remapped to roughly `[0.067, 0.933]`, centred on
/// `0.5`. The variant the §9.2.3.2 turbulence sums.
fn perlin_unsigned(x: f32, y: f32, z: f32) -> f32 {
    0.5 + 0.5 * perlin_signed(x, y, z)
}

/// §9.2.3.2 — the 16-octave turbulence: `oct = 0..=15` of `perlin_unsigned`,
/// amplitude halving / frequency doubling from `1`, soft (no-`fabs`) convention,
/// normalised by `2^15 / (2^16 - 1) = 32768 / 65535`. Centred near `0.5`.
pub(super) fn turbulence(x: f32, y: f32, z: f32) -> f32 {
    const OCT: i32 = 15;
    let mut sum = 0.0f32;
    let mut amp = 1.0f32;
    let mut freq_scale = 1.0f32;
    for _ in 0..=OCT {
        let t = perlin_unsigned(freq_scale * x, freq_scale * y, freq_scale * z);
        sum += t * amp;
        amp *= 0.5;
        freq_scale *= 2.0;
    }
    sum *= (1i32 << OCT) as f32 / ((1i32 << (OCT + 1)) - 1) as f32;
    sum
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn perm_base_is_a_permutation() {
        let mut seen = [false; 256];
        for &value in PERM_BASE.iter() {
            assert!(!seen[value as usize], "duplicate value {value}");
            seen[value as usize] = true;
        }
        assert!(seen.iter().all(|&s| s));
    }

    #[test]
    fn perm_doubles_the_base() {
        for i in 0..512 {
            assert_eq!(PERM[i], PERM_BASE[i & 255]);
        }
    }

    #[test]
    fn seeded_rng_is_deterministic_and_seed_separated() {
        let draw = |seed: u32| {
            let mut rng = SeededRng::new_hashed(seed);
            [rng.next_float(), rng.next_float(), rng.next_float()]
        };
        assert_eq!(draw(1), draw(1));
        assert_ne!(draw(1), draw(2));
        for component in draw(7) {
            assert!((0.0..1.0).contains(&component), "draw in [0,1): {component}");
        }
    }

    #[test]
    fn turbulence_is_centred_near_half_and_sub_unit() {
        // §9.2.3.1/§9.2.3.2 output range: roughly [0.067, 0.933], centred 0.5.
        for &(x, y, z) in &[
            (0.0, 0.0, 0.0),
            (3.7, 11.2, -5.5),
            (100.25, -42.0, 17.5),
            (1000.0, 1000.0, 1000.0),
        ] {
            let t = turbulence(x, y, z);
            assert!((0.0..=1.0).contains(&t), "turbulence in [0,1]: {t}");
            assert!((t - 0.5).abs() < 0.45, "centred near 0.5: {t}");
        }
    }

    #[test]
    fn turbulence_is_deterministic() {
        assert_eq!(turbulence(1.5, 2.5, 3.5), turbulence(1.5, 2.5, 3.5));
    }
}
