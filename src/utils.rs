
pub fn as_bytes(seed: u64, tree_index: u64) -> [u8; 16] {

    // TODO copy the seeds with bit shifts to get rid of the transmutes...
    use std::mem::transmute;
    let seed_bytes: [u8; 8] = unsafe { transmute(seed.to_be()) };
    let tree_index_bytes: [u8; 8] = unsafe { transmute(tree_index.to_be()) };

    [
        *seed_bytes.get(0).unwrap(),
        *seed_bytes.get(1).unwrap(),
        *seed_bytes.get(2).unwrap(),
        *seed_bytes.get(3).unwrap(),
        *seed_bytes.get(4).unwrap(),
        *seed_bytes.get(5).unwrap(),
        *seed_bytes.get(6).unwrap(),
        *seed_bytes.get(7).unwrap(),
        *tree_index_bytes.get(0).unwrap(),
        *tree_index_bytes.get(1).unwrap(),
        *tree_index_bytes.get(2).unwrap(),
        *tree_index_bytes.get(3).unwrap(),
        *tree_index_bytes.get(4).unwrap(),
        *tree_index_bytes.get(5).unwrap(),
        *tree_index_bytes.get(6).unwrap(),
        *tree_index_bytes.get(7).unwrap(),
    ]
}
