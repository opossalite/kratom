pub mod regular;
pub mod hadamard;



pub trait Multiplier {
    fn multiply(&mut self, a: &[f32], b: &[f32], sizes: (usize, usize, usize), c: &mut [f32]) -> ();
}


