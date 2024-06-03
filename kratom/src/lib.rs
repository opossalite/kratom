pub mod tools;
pub mod neuralnets;
pub mod multipliers;

use multipliers::{hadamard::DynamicHadamardMultiplier, regular::RegularMultiplier};
use neuralnets::standard::NeuralNetwork2L;



/// Returns a regular 2-hidden layer neural network with a CPU-based matrix multiplier
#[inline]
pub fn default(sizes: (usize, usize, usize, usize)) -> NeuralNetwork2L<RegularMultiplier> {
    two_layer_regular_cpu(sizes)
}


/// Returns a regular 2-hidden layer neural network with a regular CPU-based matrix multiplier
#[inline]
pub fn two_layer_regular_cpu(sizes: (usize, usize, usize, usize)) -> NeuralNetwork2L<RegularMultiplier> {
    NeuralNetwork2L::new(RegularMultiplier::new(), sizes)
}


/// Returns a regular 2-hidden layer neural network with a CPU-based Hadamard matrix multiplier
#[inline]
pub fn two_layer_hadamard_cpu(sizes: (usize, usize, usize, usize)) -> NeuralNetwork2L<DynamicHadamardMultiplier> {
    NeuralNetwork2L::new(DynamicHadamardMultiplier::new(), sizes)
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_creation_and_forward() {
        let mut neuralnet = default((4, 200, 300, 20));
        let inputs = vec![1.0, 2.0, 4.2, -1.0];
        let output = neuralnet.forward(&inputs);
        assert_eq!(20, output.len());
    }
}


