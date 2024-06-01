pub mod tools;
pub mod neuralnets;
pub mod multipliers;

pub fn add(left: usize, right: usize) -> usize {
    left + right
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let result = add(2, 2);
        assert_eq!(result, 4);
    }

    #[test]
    fn test_creation_and_forward() {
        let mut neuralnet = neuralnets::default((4, 200, 300, 20));
        let inputs = vec![1.0, 2.0, 4.2, -1.0];
        let output = neuralnet.forward(&inputs);
        assert_eq!(20, output.len());
    }
}


