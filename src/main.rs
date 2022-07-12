mod mlp;
mod epso;
mod consts;
mod activations;

use mlp::MLP;
use ndarray::arr1;

fn main() {
    rand::random::<f64>();
    
    let m: MLP = MLP::new();
    let output = m.output(&arr1(&[0.00001,0.00001,0.00001]));
    println!("{:?}", output);
}

