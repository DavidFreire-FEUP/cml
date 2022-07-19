use lazy_static::lazy_static;

//thread_local!(pub static TRAINING_IMAGES: Vec<Vec<u8>> = import_images("data/train-images-idx3-ubyte".to_string()));
//thread_local!(pub static TRAINING_LABELS: Vec<u8> = import_labels("data/train-labels-idx1-ubyte".to_string()));

lazy_static! {
    pub static ref TRAINING_IMAGES: Vec<Vec<u8>> = import_images("data/train-images-idx3-ubyte".to_string());
    pub static ref TRAINING_LABELS: Vec<u8> = import_labels("data/train-labels-idx1-ubyte".to_string());
    pub static ref TESTING_IMAGES: Vec<Vec<u8>> = import_images("data/t10k-images-idx3-ubyte".to_string());
    pub static ref TESTING_LABELS: Vec<u8> = import_labels("data/t10k-labels-idx1-ubyte".to_string());
}

pub fn import_images(filename: String) -> Vec<Vec<u8>>{
    let mut images: Vec<Vec<u8>>;
    let bytes = std::fs::read(filename).unwrap();

    let image_ammount = i32::from_be_bytes(bytes[4..8].to_vec().try_into().unwrap());
    //println!("Importing {} images", image_ammount);
    images = Vec::with_capacity(image_ammount as usize);

    let rows = i32::from_be_bytes(bytes[8..12].to_vec().try_into().unwrap());
    let columns = i32::from_be_bytes(bytes[12..16].to_vec().try_into().unwrap());
    let pixel_ammount = (rows*columns) as usize;

    // let percent_unity: i32 = (bytes.len() as f32 * 0.01).floor() as i32; 
    // let mut progress: i32 = 0;

    let mut i = 16;
    while i < bytes.len() {
        // if i > progress as usize {
        //     println!("{}", progress / percent_unity);
        //     progress += percent_unity;
        // }
        images.push(bytes[i..i+pixel_ammount].to_vec());
        i += pixel_ammount;
    }
    images
}

pub fn import_labels(filename: String) -> Vec<u8>{
    let mut images: Vec<u8>;
    let bytes = std::fs::read(filename).unwrap();
    let label_ammount = i32::from_be_bytes(bytes[4..8].to_vec().try_into().unwrap());
    images = Vec::with_capacity(label_ammount as usize);
    for i in 8..bytes.len() {
        images.push(bytes[i]);
    }
    images
}