//use transpose::transpose;

use crate::tools::{normalize_vector, Multiplier};



/// A two-hidden-layer neural network with customizable layer sizes.
pub struct NeuralNetwork<T: Multiplier> {
    sizes: (usize, usize, usize, usize),
    multiplier: T,

    // hidden
    weights0: Vec<f32>,
    bias0: Vec<f32>,

    // hidden
    weights1: Vec<f32>,
    bias1: Vec<f32>,

    // output
    weights2: Vec<f32>,
    bias2: Vec<f32>,
}
impl<T: Multiplier> NeuralNetwork<T> {
    pub fn new(multiplier: T, sizes: (usize, usize, usize, usize)) -> Self {
        NeuralNetwork {
            sizes,
            multiplier,
            
            weights0: (0..sizes.0*sizes.1).into_iter().map(|_| rand::random::<f32>() - 0.5).collect(),
            bias0: (0..sizes.1).into_iter().map(|_| rand::random::<f32>() - 0.5).collect(),

            weights1: (0..sizes.1*sizes.2).into_iter().map(|_| rand::random::<f32>() - 0.5).collect(),
            bias1: (0..sizes.2).into_iter().map(|_| rand::random::<f32>() - 0.5).collect(),

            weights2: (0..sizes.2*sizes.3).into_iter().map(|_| rand::random::<f32>() - 0.5).collect(),
            bias2: (0..sizes.3).into_iter().map(|_| rand::random::<f32>() - 0.5).collect(),
        }
    }


    /// Forward propagation with the given input
    pub fn forward(&mut self, input: &[f32]) -> Vec<f32> {

        // first hidden layer
        let mut layer0 = vec![0.0; self.sizes.1];
        self.multiplier.multiply(&input, &self.weights0, (1, self.sizes.0, self.sizes.1), &mut layer0); //matrix mul
        for i in 0..layer0.len() { //apply bias + nonlinearity
            layer0[i] = (layer0[i] + self.bias0[i]).tanh();
        }
        
        // second hidden layer
        let mut layer1 = vec![0.0; self.sizes.2];
        self.multiplier.multiply(&layer0, &self.weights1, (1, self.sizes.1, self.sizes.2), &mut layer1); //matrix mul
        for i in 0..layer1.len() { //apply bias + nonlinearity
            layer1[i] = (layer1[i] + self.bias1[i]).tanh();
        }

        // output layer
        let mut output = vec![0.0; self.sizes.3];
        self.multiplier.multiply(&layer1, &self.weights2, (1, self.sizes.2, self.sizes.3), &mut output); //matrix mul
        for i in 0..output.len() { //apply bias + nonlinearity
            output[i] = (output[i] + self.bias2[i]).tanh();
        }

        output
    }


    /// Backward propagation with the given information
    pub fn backward(&mut self, input: &[f32], expected: &[f32]) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, f32) {

        // forward first hidden layer
        let mut layer0_z = vec![0.0; self.sizes.1];
        let mut layer0 = vec![0.0; self.sizes.1];
        self.multiplier.multiply(&input, &self.weights0, (1, self.sizes.0, self.sizes.1), &mut layer0); //matrix mul
        for i in 0..layer0.len() { //apply bias + nonlinearity
            layer0_z[i] = layer0[i] + self.bias0[i];
            layer0[i] = (layer0[i] + self.bias0[i]).tanh();
        }
        normalize_vector(&mut layer0_z);
        
        // forward second hidden layer
        let mut layer1_z = vec![0.0; self.sizes.2];
        let mut layer1 = vec![0.0; self.sizes.2];
        self.multiplier.multiply(&layer0, &self.weights1, (1, self.sizes.1, self.sizes.2), &mut layer1); //matrix mul
        for i in 0..layer1.len() { //apply bias + nonlinearity
            layer1_z[i] = layer1[i] + self.bias1[i];
            layer1[i] = (layer1[i] + self.bias1[i]).tanh();
        }
        normalize_vector(&mut layer1_z);

        // forward output layer
        let mut output_z = vec![0.0; self.sizes.3];
        let mut output = vec![0.0; self.sizes.3];
        self.multiplier.multiply(&layer1, &self.weights2, (1, self.sizes.2, self.sizes.3), &mut output); //matrix mul
        for i in 0..output.len() { //apply bias + nonlinearity
            output_z[i] = output[i] + self.bias2[i];
            output[i] = (output[i] + self.bias2[i]).tanh();
        }
        normalize_vector(&mut output_z);

        // delta output layer
        let mut delta2 = vec![0.0; self.sizes.3];
        let mut diffs = vec![0.0; self.sizes.3];
        for i in 0..delta2.len() {
            diffs[i] = output[i] - expected[i];
            delta2[i] = (output[i] - expected[i]) * (1.0 - output_z[i].tanh().powi(2)); //take difference and mul by invert tanh of z
            if delta2[i].is_infinite() || delta2[i].is_nan() {
                todo!();
            }
        }

        // delta second hidden layer
        let mut delta1 = vec![0.0; self.sizes.2];
        self.multiplier.multiply(&self.weights2, &delta2, (self.sizes.2, self.sizes.3, 1), &mut delta1);
        for i in 0..delta1.len() {
            delta1[i] *= layer1_z[i];
            if delta1[i].is_infinite() || delta1[i].is_nan() {
                todo!();
            }
        }

        // delta first hidden layer
        let mut delta0 = vec![0.0; self.sizes.1];
        self.multiplier.multiply(&self.weights1, &delta1, (self.sizes.1, self.sizes.2, 1), &mut delta0);
        for i in 0..delta0.len() {
            delta0[i] *= layer0_z[i];
            if delta0[i].is_infinite() || delta0[i].is_nan() {
                println!("{:?}", layer0_z[i]);
                todo!();
            }
        }

        // calculate gradients
        let mut grads_w0 = vec![0.0; self.weights0.len()];
        let mut grads_w1 = vec![0.0; self.weights1.len()];
        let mut grads_w2 = vec![0.0; self.weights2.len()];

        for n in 0..self.sizes.0 { //weights0 tall
            for m in 0..self.sizes.1 { //weights0 wide
                grads_w0[m + (n * self.sizes.1)] = input[n] * delta0[m];
                if grads_w0[m + (n * self.sizes.1)].is_infinite() || grads_w0[m + (n * self.sizes.1)].is_nan() {
                    todo!();
                }
            }
        }
        for n in 0..self.sizes.1 { //weights1 tall
            for m in 0..self.sizes.2 { //weights1 wide
                grads_w1[m + (n * self.sizes.2)] = layer0[n] * delta1[m];
                if grads_w1[m + (n * self.sizes.2)].is_infinite() || grads_w1[m + (n * self.sizes.2)].is_nan() {
                    todo!();
                }
            }
        }
        for n in 0..self.sizes.2 { //weights2 tall
            for m in 0..self.sizes.3 { //weights2 wide
                grads_w2[m + (n * self.sizes.3)] = layer1[n] * delta2[m];
                if grads_w2[m + (n * self.sizes.3)].is_infinite() || grads_w2[m + (n * self.sizes.3)].is_nan() {
                    todo!();
                }
            }
        }

        // calculate average error
        let avg_err = diffs.iter().sum::<f32>() / diffs.len() as f32;

        (grads_w0, grads_w1, grads_w2, delta0, delta1, delta2, avg_err) //deltas are equal to bias grads
    }


    /// Train the model on the provided batch and learning rate
    pub fn train(&mut self, batch: &[(Vec<f32>, Vec<f32>)], eta: f32) -> Vec<f32> {
        let mut grad_w0_sum = vec![0.0; self.weights0.len()];
        let mut grad_w1_sum = vec![0.0; self.weights1.len()];
        let mut grad_w2_sum = vec![0.0; self.weights2.len()];
        let mut grad_b0_sum = vec![0.0; self.bias0.len()];
        let mut grad_b1_sum = vec![0.0; self.bias1.len()];
        let mut grad_b2_sum = vec![0.0; self.bias2.len()];

        let mut errors = Vec::with_capacity(batch.len());

        // iterate through the batch
        for (x, y) in batch {
            let (grad_w0, grad_w1, grad_w2, grad_b0, grad_b1, grad_b2, avg_err) = self.backward(x, y);

            // sum all the gradients to the accumulator
            for i in 0..grad_w0_sum.len() {
                grad_w0_sum[i] += grad_w0[i];
            }
            for i in 0..grad_w1_sum.len() {
                grad_w1_sum[i] += grad_w1[i];
            }
            for i in 0..grad_w2_sum.len() {
                grad_w2_sum[i] += grad_w2[i];
            }
            for i in 0..grad_b0_sum.len() {
                grad_b0_sum[i] += grad_b0[i];
            }
            for i in 0..grad_b1_sum.len() {
                grad_b1_sum[i] += grad_b1[i];
            }
            for i in 0..grad_b2_sum.len() {
                grad_b2_sum[i] += grad_b2[i];
            }

            errors.push(avg_err);
        }

        // apply the updates to the weights and biases
        let frac = eta / batch.len() as f32;

        for i in 0..grad_w0_sum.len() {
            self.weights0[i] -= frac * grad_w0_sum[i];
        }
        for i in 0..grad_w1_sum.len() {
            self.weights1[i] -= frac * grad_w1_sum[i];
        }
        for i in 0..grad_w2_sum.len() {
            self.weights2[i] -= frac * grad_w2_sum[i];
        }
        for i in 0..grad_b0_sum.len() {
            self.bias0[i] -= frac * grad_b0_sum[i];
        }
        for i in 0..grad_b1_sum.len() {
            self.bias1[i] -= frac * grad_b1_sum[i];
        }
        for i in 0..grad_b2_sum.len() {
            self.bias2[i] -= frac * grad_b2_sum[i];
        }

        errors
    }
}


