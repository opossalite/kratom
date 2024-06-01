pub mod standard;

use crate::multipliers::regular::RegularMultiplier;
use self::standard::NeuralNetwork;



/// Returns a regular 2-hidden layer neural network with a CPU-based matrix multiplier
#[inline]
pub fn default(sizes: (usize, usize, usize, usize)) -> NeuralNetwork<RegularMultiplier> {
    two_layer_regular_cpu(sizes)
}


/// Returns a regular 2-hidden layer neural network with a CPU-based matrix multiplier
pub fn two_layer_regular_cpu(sizes: (usize, usize, usize, usize)) -> NeuralNetwork<RegularMultiplier> {
    NeuralNetwork::new(RegularMultiplier::new(), sizes)
}


