use std::f32::consts::PI;

pub fn relu(input: f32) -> f32{
    if input>1.0 {
        return 1.0
    } 
    else if input<0.0 {
        return 0.0;
    }
    return input;
}

pub fn atann(input: f32) -> f32 {
    return 2.0 * input.atan() / PI;
}