use rand::prelude::*;
use crate::tools::Multiplier;



pub struct HadamardMultiplier {
    pub d: usize, //number of columns to sample for RHT approximation
}
impl HadamardMultiplier {
    pub fn new(d: usize) -> Self {
        HadamardMultiplier { d }
    }
}
impl Multiplier for HadamardMultiplier {
    fn multiply(&mut self, a: &[f32], b: &[f32], sizes: (usize, usize, usize), c: &mut [f32]) {
        randomized_hadamard_transform(a, b, sizes, c, self.d);
    }
}


// Function to perform Randomized Hadamard Transform (RHT) for matrix multiplication approximation
pub fn randomized_hadamard_transform(
    a: &[f32], // Contiguous array representing matrix A (row-major order)
    b: &[f32], // Contiguous array representing matrix B (row-major order)
    //m: usize,  // Number of rows in matrix A
    //n: usize,  // Number of columns in matrix A (and rows in matrix B)
    //p: usize,  // Number of columns in matrix B
    sizes: (usize, usize, usize), //rows in A, columns in A / rows in B, columns in B
    c: &mut [f32], //output
    d: usize,  // Number of columns to sample for RHT approximation
) {

    let m = sizes.0;
    let n = sizes.1;
    let p = sizes.2;

    // Generate random indices to sample d columns from matrices A and B
    let mut rng = thread_rng();
    let a_indices: Vec<usize> = (0..n).choose_multiple(&mut rng, d);
    let b_indices: Vec<usize> = (0..n).choose_multiple(&mut rng, d);

    // Initialize result vector
    //let mut result = vec![0.0; m * p];

    // Perform Randomized Hadamard Transform approximation
    for idx in 0..d {
        let a_col_start = a_indices[idx] * m;
        let b_col_start = b_indices[idx] * p;

        for i in 0..m {
            for j in 0..p {
                c[i * p + j] += a[a_col_start + i] * b[b_col_start + j];
            }
        }
    }
}
