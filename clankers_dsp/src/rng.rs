/// Fast xorshift32 RNG — seeded from JS Math.random() on construction.
pub struct Rng(u32);

impl Rng {
    pub fn new(seed: u32) -> Self {
        Self(if seed == 0 { 0xdead_beef } else { seed })
    }

    #[inline]
    pub fn next_u32(&mut self) -> u32 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.0 = x;
        x
    }

    /// Uniform float in [-1, 1]
    #[inline]
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u32() as i32) as f32 / 2_147_483_648.0
    }

    /// LCG identical to AntigravityDrums KickVoice click noise
    /// state = state * 1664525 + 1013904223
    #[inline]
    pub fn lcg_f32(state: &mut u32) -> f32 {
        *state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (*state as i32) as f32 / 2_147_483_648.0
    }
}
