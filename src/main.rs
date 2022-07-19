mod mlp;
mod epso;
mod activations;
mod util;
mod test;
mod importer;

use epso::Swarm;
use importer::{TRAINING_IMAGES, TRAINING_LABELS, TESTING_IMAGES, TESTING_LABELS};
use mlp::ActivFun;
use ndarray::arr1;

use crate::{mlp::MLP, activations::atann};

const INPUT_WIDTH: usize = 28*28;
const NEURON_WIDTH: usize = 10;
const HIDDEN_LAYERS: usize = 2;
const OUTPUT_WIDTH: usize = 10;
const ACTIVATION: ActivFun = atann;

#[test]
fn test_fitness_func(){
    let mlp = MLP::new_random(
        &INPUT_WIDTH,
        &NEURON_WIDTH,
        &HIDDEN_LAYERS,
        &OUTPUT_WIDTH,
        ACTIVATION
    );
    println!("{}", fitness(&mlp.to_chromossome()));
    assert!(true);
}


pub fn fitness(chromossome: &Vec<f32>) -> i32 {
    let mut mlp = MLP::from_chromossome(
        chromossome,
        &INPUT_WIDTH,
        &NEURON_WIDTH,
        &HIDDEN_LAYERS,
        &OUTPUT_WIDTH,
        ACTIVATION
    );
    let mut error_sum:f32 = 0.0;

    let mut float_image: Vec<f32> = Vec::with_capacity(TRAINING_IMAGES[0].len());
    for (image, label) in TRAINING_IMAGES.iter().zip(TRAINING_LABELS.iter()){
        //println!("{}",*label);
        for pixel in image {
            float_image.push(*pixel as f32);
        }
        error_sum += ((1.0 - mlp.output(&arr1(&float_image)).to_vec()[*label as usize])*100.0).powi(2);
        float_image.clear();
    }

    return - error_sum.floor() as i32;
}

fn main() {
    let mlp = MLP::new_random(
        &INPUT_WIDTH,
        &NEURON_WIDTH,
        &HIDDEN_LAYERS,
        &OUTPUT_WIDTH,
        ACTIVATION
    );
    let mut swarm = Swarm::new(
        &100,
        &mlp.to_chromossome().len(),
        &0.01,
        &0.5,
        &0.5,
        &0.5,
        fitness,
    );

    let generations = 100;

    println!("Global best fitness:");
    for _ in 0..generations {
        swarm.reproduce();
        swarm.travel();
        swarm.select();
        println!("{}", swarm.global_best.fitness);
    }

    println!("Winning chromossome");
    println!("{:?}", swarm.global_best.chromossome);

    println!("Accuracy on test set: {} %", test_final_model(&swarm.global_best.chromossome));

}


pub fn test_final_model(chromossome: &Vec<f32>) -> f32 {
    let mut mlp = MLP::from_chromossome(
        chromossome,
        &INPUT_WIDTH,
        &NEURON_WIDTH,
        &HIDDEN_LAYERS,
        &OUTPUT_WIDTH,
        ACTIVATION
    );

    let mut float_image: Vec<f32> = Vec::with_capacity(TESTING_IMAGES[0].len());
    let samples_size = TESTING_LABELS.len();
    let mut hits = 0;
    for (image, label) in TRAINING_IMAGES.iter().zip(TESTING_LABELS.iter()){
        //println!("{}",*label);
        for pixel in image {
            float_image.push(*pixel as f32);
        }
        let output = mlp.output(&arr1(&float_image)).to_vec();

        let mut max = &0.0;
        for i in 0..output.len() {
            if output[i] > *max {
                max = &output[i];
            }
        }
        if output[*label as usize] == *max {
            hits += 1;
        }
        float_image.clear();
    }

    return hits as f32/samples_size as f32 * 100.0;
}
