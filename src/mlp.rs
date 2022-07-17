use ndarray::{Array1, Array2, arr1, arr2};
use rand::Rng;
use crate::util::to_array2;

// use rand::random;

pub type ActivFun = fn(f32) -> f32;

#[derive(Clone)]
pub struct NeuronLayer {
    pub values: Array1<f32>,
    pub weights: Array2<f32>,
    pub bias: Array1<f32>,
    pub activation: ActivFun // activation function pointer
}

impl NeuronLayer {

    pub fn new(width: &usize, weights: Array2<f32>, bias: Array1<f32>, activation: ActivFun) -> Self {
        if weights != arr2(&[[]]) && bias != arr1(&[]){
            assert_eq!(*width, weights.shape()[0], "Weight matrix's row size doesn't match width.");
        }
        let values: Array1<f32> = arr1(&vec![0.0; *width]);
        Self {values, weights, bias, activation}
    }

    pub fn new_random(input_width: &usize, width: &usize, activation: ActivFun) -> Self {
        assert!(*width > 0);
        let mut rng = rand::thread_rng();
        let values_arr: Vec<f32> = vec![0.0; *width];
        let mut weight_arr: Vec<Array1<f32>> = Vec::with_capacity(*width);

        for _ in 0..*width {
            let mut temp_row = Vec::with_capacity(*input_width);
            for _ in 0..*input_width{
                temp_row.push(rng.gen());
            }
            weight_arr.push(arr1(&temp_row));
        }

        let mut bias_arr: Vec<f32>  = Vec::with_capacity(*width);
        for _ in 0..*width {
            bias_arr.push(rng.gen());
        }
        //bias_arr.iter_mut().for_each(|elem| *elem = *elem * 0.001);
        let values = arr1(&values_arr);
        let weights = to_array2(&weight_arr).unwrap();

        let bias = arr1(&bias_arr);

        Self {values, weights, bias, activation}
    }

    /// Apply an input to a neuron layer and calculate resulting values
    pub fn excite(&mut self, input: &Array1<f32>) {
        let mut temp_values = &self.weights.dot(input) + &self.bias;
        temp_values.iter_mut().for_each(|x| *x = (self.activation)(*x));
        self.values = temp_values;
    }
}

#[derive(Clone)]
pub struct MLP {
    pub layers: Vec<NeuronLayer>
}

impl MLP {
    /// Creates an MLP with random weights and biases based on the topology args
    pub fn new_random(input_width:&usize, neuron_width:&usize, hidden_layers:&usize, output_width:&usize, activation: ActivFun) -> Self { 
        let mut layers: Vec<NeuronLayer> = Vec::with_capacity(hidden_layers+2);

        for i in 0..hidden_layers+2 {
            if i == 0 { // Input layer 
                layers.push(NeuronLayer::new_random(&0, input_width, activation));
            }
            else if i == 1 { // First hidden layer
                layers.push(NeuronLayer::new_random(input_width, neuron_width, activation));
            }
            else if i == hidden_layers+1 { // Output layer
                layers.push(NeuronLayer::new_random(neuron_width, output_width, activation));
            }
            else { // Hidden layer
                layers.push(NeuronLayer::new_random(neuron_width, neuron_width, activation));
            }
        }
        Self {layers}
    }

    /// Encode the MLP into a simple chromossome structure
    /// Flattens the neuron layers' weights and biases into a linear vec
    pub fn to_chromossome(self) -> Vec<f32>{
        let mut result: Vec<f32> = vec![];

        // Goes from input to output, weights then biases
        // i starts at 1 since we dont need input layer weights and biases
        for i in 1..self.layers.len() {
            result = [
                result.as_slice(),
                self.layers[i].weights.as_slice().unwrap(),
                self.layers[i].bias.as_slice().unwrap()
            ].concat();
        }
        return result
    }

    /// Decode a chromossome into an MLP
    /// Needs the MLP topology metadata since the chromossome represents raw unlabeled data
    pub fn from_chromossome(
        chromossome: &Vec<f32>,
        input_width: &usize,
        neuron_width: &usize,
        hidden_layers: &usize,
        output_width: &usize,
        activation: ActivFun) -> Self {

        let mut layers = Vec::with_capacity(hidden_layers+2);
        let mut i = 0;
        let mut weights: Vec<Array1<f32>>;
        let mut bias: Array1<f32>;
        
        // Insert input layer
        layers.push(
            NeuronLayer::new(
                input_width,
                arr2(&[[]]),
                arr1(&[]),
                activation
            )
        );

        // First hidden layer
        weights = Vec::with_capacity(*neuron_width);
        for _ in 0..*neuron_width {
            weights.push(arr1(&chromossome[i..i+*input_width]));
            i += *input_width;
        }

        bias = arr1(&chromossome[i..i+*neuron_width]);
        i += *neuron_width;

        layers.push(
            NeuronLayer::new(
                neuron_width,
                to_array2(&weights).unwrap(),
                bias,
                activation
            )
        );

        // Hidden layers
        for _ in 0..hidden_layers-1 {
            weights = Vec::with_capacity(*neuron_width);
            for _ in 0..*neuron_width {
                weights.push(arr1(&chromossome[i..i+*neuron_width]));
                i += *neuron_width;
            }
    
            bias = arr1(&chromossome[i..i+*neuron_width]);
            i += *neuron_width;
    
            layers.push(
                NeuronLayer::new(
                    neuron_width,
                    to_array2(&weights).unwrap(),
                    bias,
                    activation
                )
            );
        }

        // Output layer
        weights = Vec::with_capacity(*output_width);
        for _ in 0..*output_width {
            weights.push(arr1(&chromossome[i..i+*neuron_width]));
            i += *neuron_width;
        }

        bias = arr1(&chromossome[i..i+*neuron_width]);
        i += *neuron_width;

        layers.push(
            NeuronLayer::new(
                neuron_width,
                to_array2(&weights).unwrap(),
                bias,
                activation
            )
        );

        Self { layers }
    }

    /// Excite the MLP input layer and get the resulting values of the output layer 
    pub fn output(&mut self, input: &Array1<f32>) -> Array1<f32> {
        for i in 0..self.layers.len() {
            if i == 0 {
                self.layers[i].values = input.clone();
                continue;
            }
            let inputs = self.layers[i-1].values.clone();
            self.layers[i].excite(&inputs);
        }
        return self.layers.last().unwrap().values.clone();
    }
}
