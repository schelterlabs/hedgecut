extern crate hedgecut;

use hedgecut::dataset::CardioDataset;
use hedgecut::evaluation::end_to_end;

fn main() {

    let samples = CardioDataset::samples_from_csv("datasets/cardio-train.csv");
    let test_data = CardioDataset::samples_from_csv("datasets/cardio-test.csv");

    let dataset = CardioDataset::from_samples(&samples);

    let seed: u64 = 4545;
    let num_trees = 100;
    let min_leaf_size = 2;
    let max_tries_per_split = 5;

    end_to_end(
        "cardio",
        dataset,
        samples,
        test_data,
        seed,
        num_trees,
        min_leaf_size,
        max_tries_per_split
    );
}

