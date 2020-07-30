extern crate hedgecut;

use hedgecut::dataset::ShoppingDataset;
use hedgecut::evaluation::end_to_end;

fn main() {

    let samples = ShoppingDataset::samples_from_csv("datasets/shopping-train.csv");
    let test_data = ShoppingDataset::samples_from_csv("datasets/shopping-test.csv");
    let dataset = ShoppingDataset::from_samples(&samples);

    let seed: u64 = 4545;
    let num_trees = 100;
    let min_leaf_size = 2;
    let max_tries_per_split = 5;

    end_to_end(
        "shopping",
        dataset,
        samples,
        test_data,
        seed,
        num_trees,
        min_leaf_size,
        max_tries_per_split
    );
}

