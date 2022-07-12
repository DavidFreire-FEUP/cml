use ndarray::{Array1, Array2, arr1, arr2};
use crate::consts::{self, ACTIVATION};
use consts::{
    HIDDEN_LAYERS,
    HIDDEN_SIZE,
    INPUT_SIZE,
    OUTPUT_SIZE
};
// use rand::random;

pub type ActivFun = fn(f32) -> f32;

pub struct NeuronLayer {
    pub values: Array1<f32>,
    pub weights: Array2<f32>,
    pub bias: Array1<f32>,
    pub activation: ActivFun // activation function pointer
}

impl NeuronLayer {
    pub fn excite(&mut self, input: &Array1<f32>) {
        let mut temp_values = &self.weights.dot(input) + &self.bias;
        //println!("temp vals {:?}", temp_values);
        temp_values.iter_mut().for_each(|x| *x = (self.activation)(*x));
        //println!("temp vals {:?}", temp_values);
        self.values = temp_values;
    }
}

pub struct MLP {
    pub layers: Vec<NeuronLayer>
}

impl MLP {
    pub fn new() -> Self { 
        let mut layers: Vec<NeuronLayer> = Vec::with_capacity(HIDDEN_LAYERS+2);
    
        (0..HIDDEN_LAYERS+2).for_each(|i| {
            let values: Array1<f32> ;
            let weights: Array2<f32>;
            let bias: Array1<f32>;

            if i == 0 {
                let values_arr: [f32; INPUT_SIZE]  = [0.0; INPUT_SIZE];
                let weight_arr: [[f32; 0]; 0] =  [[0.0; 0]; 0];
                let bias_arr: [f32; 0]  = [0.0; 0];
                values = arr1(&values_arr);
                weights = arr2(&weight_arr);
                bias = arr1(&bias_arr);
                
            }
            else if i == 1 {
                let values_arr: [f32; HIDDEN_SIZE]  = [0.0; HIDDEN_SIZE];
                let weight_arr: [[f32; INPUT_SIZE]; HIDDEN_SIZE] = rand::random();
                let bias_arr: [f32; HIDDEN_SIZE]  = rand::random();
                //bias_arr.iter_mut().for_each(|elem| *elem = *elem * 0.001);

                values = arr1(&values_arr);
                weights = arr2(&weight_arr);
                bias = arr1(&bias_arr);
            }
            else if i == HIDDEN_LAYERS+1 {
                let values_arr: [f32; OUTPUT_SIZE]  = [0.0; OUTPUT_SIZE];
                let weight_arr: [[f32; HIDDEN_SIZE]; HIDDEN_SIZE] = rand::random();
                let bias_arr: [f32; OUTPUT_SIZE]  = rand::random();
                //bias_arr.iter_mut().for_each(|elem| *elem = *elem * 0.001);
                values = arr1(&values_arr);
                weights = arr2(&weight_arr);
                bias = arr1(&bias_arr);
            }
            else {
                let values_arr: [f32; HIDDEN_SIZE]  = [0.0; HIDDEN_SIZE];
                let weight_arr: [[f32; HIDDEN_SIZE]; HIDDEN_SIZE] = rand::random();
                let bias_arr: [f32; HIDDEN_SIZE]  = rand::random();
                //bias_arr.iter_mut().for_each(|elem| *elem = *elem * 0.001);
                values = arr1(&values_arr);
                weights = arr2(&weight_arr);
                bias = arr1(&bias_arr);
            }
            layers.push(NeuronLayer {
                values,
                weights,
                bias,
                activation: ACTIVATION,
            })
        });
        Self {layers}
    }

    pub fn output(mut self, input: Array1<f32>) -> Array1<f32> {
        for i in 0..self.layers.len() {
            if i == 0 {
                self.layers[i].values = input.clone();
            }
            else {
                let inputs = self.layers[i-1].values.clone();
                self.layers[i].excite(&inputs);
            }
        }
        let output = self.layers[self.layers.len()-1].values.clone();
        return output;
    }
}
