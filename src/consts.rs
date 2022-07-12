use crate::mlp::{ActivFun};
use crate::activations::relu;

pub const HIDDEN_LAYERS:usize = 3;
pub const HIDDEN_SIZE:usize = 10;
pub const INPUT_SIZE:usize = 3;
pub const OUTPUT_SIZE:usize = 10;
pub const ACTIVATION:ActivFun = relu;

pub const MUTATION_RATE:f32 = 0.01;