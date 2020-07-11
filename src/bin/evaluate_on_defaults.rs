extern crate hedgecut;

use hedgecut::dataset::DefaultsDataset;
use hedgecut::evaluation::end_to_end;

fn main() {

    let samples = DefaultsDataset::samples_from_csv("datasets/defaults-train.csv");
    let test_data = DefaultsDataset::samples_from_csv("datasets/defaults-test.csv");
    let dataset = DefaultsDataset::from_samples(&samples);

    let seed: u64 = 4545;
    let num_trees = 100;
    let min_leaf_size = 2;
    let max_tries_per_split = 300;

    end_to_end(
        "defaults",
        dataset,
        samples,
        test_data,
        seed,
        num_trees,
        min_leaf_size,
        max_tries_per_split
    );
}

