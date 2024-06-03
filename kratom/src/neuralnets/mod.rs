pub mod standard;



pub enum NeuralNetworkError {
    BadBufferSize,
}


pub trait NeuralNetwork {
    fn forward(&mut self, input: &[f32]) -> Vec<f32>;
    fn backward(&mut self, input: &[f32], expected: &[f32]) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, f32);
    fn train(&mut self, batch: &[(Vec<f32>, Vec<f32>)], eta: f32) -> Vec<f32>;

    fn forward_buf(&mut self, input: &[f32], output_buf: &mut [f32]) -> Result<(), NeuralNetworkError>;
    //fn backward_buf(&mut self, input: &[f32], expected: &[f32], grad_bufs: &mut [Vec<f32>], delta_bufs: &mut [Vec<f32>]) -> f32;
    //fn train_buf(&mut self, batch: &[(Vec<f32>, Vec<f32>)], eta: f32, errors_buf: &mut [f32]);
}


