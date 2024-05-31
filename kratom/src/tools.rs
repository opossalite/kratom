
pub trait Multiplier {
    fn multiply(&mut self, a: &[f32], b: &[f32], sizes: (usize, usize, usize), c: &mut [f32]) -> ();
}


pub fn normalize_vector(a: &mut [f32]) {
    let mean = a.iter().sum::<f32>() / a.len() as f32;
    let variance = a.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / a.len() as f32;
    for i in 0..a.len() {
        a[i] = (a[i] - mean) / (variance + 0.00000001).sqrt();
    }
}


