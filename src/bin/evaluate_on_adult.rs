extern crate hedgecut;

use hedgecut::dataset::AdultDataset;
use hedgecut::evaluation::end_to_end;

fn main() {

    let samples = AdultDataset::samples_from_csv("datasets/adult-train.csv");
    let test_data = AdultDataset::samples_from_csv("datasets/adult-test.csv");
    let dataset = AdultDataset::from_samples(&samples);

    let seed: u64 = 4545;
    let num_trees = 100;
    let min_leaf_size = 2;
    let max_tries_per_split = 5;

    end_to_end(
        "adult",
        dataset,
        samples,
        test_data,
        seed,
        num_trees,
        min_leaf_size,
        max_tries_per_split
    );
}

