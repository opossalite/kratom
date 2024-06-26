use rand::prelude::*;
use super::Multiplier;



/// Modified Hadamard multiplier which dynamically sets the sample size
pub struct DynamicHadamardMultiplier {}
impl DynamicHadamardMultiplier {
    #[inline]
    pub fn new() -> Self {
        DynamicHadamardMultiplier { }
    }
}
impl Multiplier for DynamicHadamardMultiplier {
    #[inline]
    fn multiply(&mut self, a: &[f32], b: &[f32], sizes: (usize, usize, usize), c: &mut [f32]) {
        randomized_hadamard_transform(a, b, sizes, c, sizes.1);
    }
}


pub struct HadamardMultiplier {
    /// Recommended to be the same size as the shared dimension
    pub sample_size: usize,
}
impl HadamardMultiplier {
    #[inline]
    pub fn new(sample_size: usize) -> Self {
        HadamardMultiplier { sample_size }
    }
}
impl Multiplier for HadamardMultiplier {
    #[inline]
    fn multiply(&mut self, a: &[f32], b: &[f32], sizes: (usize, usize, usize), c: &mut [f32]) {
        randomized_hadamard_transform(a, b, sizes, c, self.sample_size);
    }
}


// Perform Randomized Hadamard Transform (RHT) for matrix multiplication approximation
pub fn randomized_hadamard_transform(a: &[f32], b: &[f32], sizes: (usize, usize, usize), c: &mut [f32], sample_size: usize) {
    let (m, n, p) = sizes;

    // generate random indices to sample columns from matrices
    let mut rng = thread_rng();
    let a_indices: Vec<usize> = (0..n).choose_multiple(&mut rng, sample_size);
    let b_indices: Vec<usize> = (0..n).choose_multiple(&mut rng, sample_size);

    // perform Hadamard
    for idx in 0..sample_size {
        let a_col_start = a_indices[idx] * m;
        let b_col_start = b_indices[idx] * p;

        for i in 0..m {
            for j in 0..p {
                c[i * p + j] += a[a_col_start + i] * b[b_col_start + j];
            }
        }
    }
}


