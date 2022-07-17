use ndarray::{Array1, Array2, arr2};


/// Converts vector of 1D arrays to 2D array
pub fn to_array2<T: Copy>(source: &[Array1<T>]) -> Result<Array2<T>, impl std::error::Error> {
    if source.len() == 0 {
        return Ok(arr2(&[[]]));
    }
    let width = source.len();
    let flattened: Array1<T> = source.into_iter().flat_map(|row| row.to_vec()).collect();
    let height = flattened.len() / width;
    flattened.into_shape((width, height))
}