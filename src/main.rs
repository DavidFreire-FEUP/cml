
fn relu(input: f32) -> f32{
    if input>1.0 {
        return 1.0
    } 
    else if input<0.0 {
        return 0.0;
    }
    return input;
}

fn main() {
    println!("Hello, world!");
}
