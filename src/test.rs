#[cfg(test)]
mod tests {
    use ndarray::arr1;
    use crate::{mlp::MLP, activations::atann};

    #[test]
    fn test_encode_and_decode_mlp_same_output() {
        let input_width:usize = 3;
        let neuron_width:usize = 10;
        let hidden_layers:usize = 3;
        let output_width:usize = 10;
    
        let mut m: MLP = MLP::new_random(
            &input_width,
            &neuron_width,
            &hidden_layers,
            &output_width,
            atann);
    
        let output = m.output(&arr1(&[0.000001,0.000001,0.000001]));
    
        let chrommy: Vec<f32> = m.to_chromossome();

        let mut new_mlp = MLP::from_chromossome(
            &chrommy,
            &input_width,
            &neuron_width,
            &hidden_layers,
            &output_width,
            atann,
        );
    
        let new_output = new_mlp.output(&arr1(&[0.00001,0.00001,0.00001]));

        assert_eq!(output, new_output);
    }

}