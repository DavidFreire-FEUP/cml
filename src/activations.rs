
#[allow(dead_code)]
pub fn relu(input: f32) -> f32{
    if input>1.0 {
        return 1.0
    } 
    else if input< -1.0 {
        return -1.0;
    }
    return input;
}

#[allow(dead_code)]
pub fn atann(input: f32) -> f32 {
    input.tanh()
}