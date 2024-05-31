use crate::tools::Multiplier;



pub struct RegularMultiplier {}
impl Multiplier for RegularMultiplier {
    fn multiply(&mut self, a: &[f32], b: &[f32], sizes: (usize, usize, usize), c: &mut [f32]) {
        multiply(a, b, sizes, c)
    }
}
impl RegularMultiplier {
    pub fn new() -> Self {
        RegularMultiplier { }
    }
}


/// Standard matrix multiplication
pub fn multiply(a: &[f32], b: &[f32], sizes: (usize, usize, usize), c: &mut [f32]) {
    // note: sizes = (A height, A width && B height, B width)
    let (m, n, p) = sizes;

    for i in 0..m { //A height
        for j in 0..p { //B width
            // note: chooses a row in A and a column in B and sums the products
            let mut sum = 0.0;
            for k in 0..n { //A width and B height
                sum += a[i*n + k] * b[k*p + j];
            }
            c[i*p + j] = sum;
        }
    }
}


